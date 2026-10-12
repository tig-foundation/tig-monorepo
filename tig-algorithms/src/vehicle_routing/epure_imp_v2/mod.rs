use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
mod compression {
use super::problem::{NodeData, Problem};
use super::sequence::Sequence;

pub struct ProblemCompression {
    pub compact: Problem,    
    pub chains: Vec<Vec<usize>>,
}

impl ProblemCompression {
    pub fn from_chains(data: &Problem, chains: Vec<Vec<usize>>) -> Option<Self> {
        let mut compact_chains: Vec<Vec<usize>> = Vec::with_capacity(chains.len());
        for chain in chains {
            debug_assert!(!chain.is_empty(), "Consensus chain should never be empty");
            let chain_seq = Self::sequence_on_chain(data, &chain);
            if chain_seq.tw == 0 && chain_seq.earliest_start() <= chain_seq.tau_plus {
                compact_chains.push(chain);
            } else {
                for id in chain {
                    compact_chains.push(vec![id]);
                }
            }
        }

        let compact_clients = compact_chains.len();
        debug_assert!(compact_clients <= data.nb_nodes - 1);
        if compact_clients == data.nb_nodes - 1 {
            return None;
        }

        let mut node_data: Vec<NodeData> = Vec::with_capacity(compact_clients + 1);
        let mut node_positions: Vec<(i32, i32)> = Vec::with_capacity(compact_clients + 1);
        let mut first_orig: Vec<usize> = vec![0usize; compact_clients + 1];
        let mut last_orig: Vec<usize> = vec![0usize; compact_clients + 1];
        let mut fixed_distance: i32 = 0;

        node_data.push(data.node_data[0]);
        node_positions.push(data.node_positions[0]);

        for chain in &compact_chains {
            let chain_seq = Self::sequence_on_chain(data, chain);
            let demand_sum = chain_seq.load;
            let mut sum_x: i64 = 0;
            let mut sum_y: i64 = 0;
            for &id in chain {
                sum_x += data.node_positions[id].0 as i64;
                sum_y += data.node_positions[id].1 as i64;
            }
            let chain_len = chain.len() as i64;
            let barycenter_x = (sum_x / chain_len) as i32;
            let barycenter_y = (sum_y / chain_len) as i32;

            fixed_distance += chain_seq.distance;

            node_data.push(NodeData {
                start_tw: chain_seq.earliest_start(),
                end_tw: chain_seq.tau_plus,
                service_time: chain_seq.duration(),
                demand: demand_sum,
            });
            node_positions.push((barycenter_x, barycenter_y));
            first_orig[node_data.len() - 1] = chain[0];
            last_orig[node_data.len() - 1] = chain[chain.len() - 1];
        }

        let nb_nodes = compact_clients + 1;
        let mut distance_matrix = vec![0i32; nb_nodes * nb_nodes];
        for i in 0..nb_nodes {
            let from_node = if i == 0 { 0 } else { last_orig[i] };
            for j in 0..nb_nodes {
                let to_node = if j == 0 { 0 } else { first_orig[j] };
                distance_matrix[i * nb_nodes + j] = data.dm(from_node, to_node);
            }
        }

        let total_demand = node_data
            .iter()
            .skip(1)
            .map(|nd| nd.demand as i64)
            .sum::<i64>();
        let lb_vehicles =
            ((total_demand + data.max_capacity as i64 - 1) / data.max_capacity as i64) as usize;
        debug_assert!(
            lb_vehicles <= data.nb_vehicles,
            "lb_vehicles exceeds available vehicles"
        );
        debug_assert!(
            lb_vehicles <= compact_clients,
            "lb_vehicles exceeds number of clients"
        );

        let distance_matrix_transposed = Problem::transpose_distances(&distance_matrix, nb_nodes);
        let compact = Problem {
            seed: data.seed,
            nb_nodes,
            root_nb_nodes: data.root_nb_nodes,
            nb_vehicles: data.nb_vehicles,
            lb_vehicles,
            fixed_distance_offset: data.fixed_distance_offset + fixed_distance as i64,
            max_capacity: data.max_capacity,
            distance_matrix,
            distance_matrix_transposed,
            node_positions,
            node_data,
        };

        Some(Self {
            compact,
            chains: compact_chains,
        })
    }

    #[inline]
    fn sequence_on_chain(data: &Problem, chain: &[usize]) -> Sequence {
        let mut seq = Sequence::singleton(data, chain[0]);
        for &id in chain.iter().skip(1) {
            let node = Sequence::singleton(data, id);
            seq = Sequence::join2(data, &seq, &node);
        }
        seq
    }
}
}
mod constructive {
use super::problem::Problem;
use rand::{rngs::SmallRng, Rng};

#[inline(always)]
fn copy_at<T: Copy>(slice: &[T], index: usize) -> T {
    debug_assert!(index < slice.len());
    unsafe { *slice.get_unchecked(index) }
}

const SHUFFLE_WINDOW: usize = 10;

#[derive(Default)]
struct RouteTimes {
    depart: Vec<i32>,
    edge: Vec<i32>,
    late: Vec<i32>,
}

pub struct Constructive;

impl Constructive {
    pub fn build_routes(data: &Problem, rng: &mut SmallRng, randomize: bool) -> Vec<Vec<usize>> {
        let mut routes = Vec::new();
        let mut nodes: Vec<usize> = (1..data.nb_nodes).collect();
        let n = nodes.len();
        nodes.sort_by(|&a, &b| data.dm(0, a).cmp(&data.dm(0, b)));

        if randomize {
            for i in 0..(n - 1) {
                nodes.swap(i, rng.gen_range(i + 1..=(i + SHUFFLE_WINDOW).min(n - 1)));
            }
        }

        let mut available = vec![true; data.nb_nodes];
        available[0] = false;
        let mut times = RouteTimes::default();

        while let Some(node) = nodes.pop() {
            if !copy_at(&available, node) {
                continue;
            }
            available[node] = false;
            let mut route = vec![0, node, 0];
            let mut route_demand = data.nd(node).demand;

            while let Some((best_node, best_pos)) = Self::find_best_insertion(
                &route,
                &nodes,
                &available,
                route_demand,
                data,
                &mut times,
            ) {
                available[best_node] = false;
                route_demand += data.nd(best_node).demand;
                route.insert(best_pos, best_node);
            }

            routes.push(route);
        }

        routes
    }

    pub fn build_nearest(data: &Problem) -> Vec<Vec<usize>> {
        let mut nodes: Vec<usize> = (1..data.nb_nodes).collect();
        nodes.sort_by(|&a, &b| data.dm(0, b).cmp(&data.dm(0, a)).then_with(|| a.cmp(&b)));
        Self::seed_and_insert(data, nodes)
    }

    pub fn build_sweep(data: &Problem) -> Vec<Vec<usize>> {
        let (dx0, dy0) = data.node_positions[0];
        let mut order: Vec<usize> = (1..data.nb_nodes).collect();
        order.sort_by(|&a, &b| {
            let angle = |id: usize| {
                let (x, y) = data.node_positions[id];
                let dx = (x - dx0) as f64;
                let dy = (y - dy0) as f64;
                dy.atan2(dx)
            };
            angle(a).total_cmp(&angle(b)).then_with(|| a.cmp(&b))
        });

        let horizon = data.nd(0).end_tw - data.nd(0).start_tw;
        let short_limit = horizon / 2;
        let mut available = vec![true; data.nb_nodes];
        available[0] = false;
        let mut routes = Vec::new();

        while available.iter().any(|&open| open) {
            let mut group = Vec::new();
            let mut demand = 0i32;
            for &id in &order {
                if !available[id] {
                    continue;
                }
                let dem = data.nd(id).demand;
                if !group.is_empty() && demand + dem > data.max_capacity {
                    break;
                }
                group.push(id);
                demand += dem;
            }
            let mut narrow = Vec::new();
            let mut wide = Vec::new();
            for id in group {
                available[id] = false;
                let nd = data.nd(id);
                if nd.end_tw - nd.start_tw <= short_limit {
                    narrow.push(id);
                } else {
                    wide.push(id);
                }
            }
            narrow.sort_by(|&a, &b| {
                data.nd(a)
                    .end_tw
                    .cmp(&data.nd(b).end_tw)
                    .then_with(|| a.cmp(&b))
            });
            let mut route = Vec::with_capacity(narrow.len() + wide.len() + 2);
            route.push(0);
            route.extend(narrow);
            route.push(0);
            for id in wide {
                let pos = Self::best_detour_pos(data, &route, id);
                route.insert(pos, id);
            }
            routes.push(route);
        }

        routes
    }

    pub fn discard_randomized_start(nb_customers: usize, rng: &mut SmallRng) {
        let n = nb_customers;
        if n <= 1 {
            return;
        }
        for i in 0..(n - 1) {
            let _ = rng.gen_range(i + 1..=(i + SHUFFLE_WINDOW).min(n - 1));
        }
    }

    fn seed_and_insert(data: &Problem, mut nodes: Vec<usize>) -> Vec<Vec<usize>> {
        let mut routes = Vec::new();
        let mut available = vec![true; data.nb_nodes];
        available[0] = false;
        let mut times = RouteTimes::default();

        while let Some(node) = nodes.pop() {
            if !copy_at(&available, node) {
                continue;
            }
            available[node] = false;
            let mut route = vec![0, node, 0];
            let mut route_demand = data.nd(node).demand;

            while let Some((best_node, best_pos)) = Self::find_best_insertion(
                &route,
                &nodes,
                &available,
                route_demand,
                data,
                &mut times,
            ) {
                available[best_node] = false;
                route_demand += data.nd(best_node).demand;
                route.insert(best_pos, best_node);
            }

            routes.push(route);
        }

        routes
    }

    fn best_detour_pos(data: &Problem, route: &[usize], id: usize) -> usize {
        let from_insert = data.distance_row(id);
        let to_insert = data.distance_column(id);
        let mut best = i32::MAX;
        let mut best_pos = 1usize;
        for pos in 1..route.len() {
            let prev = route[pos - 1];
            let next = route[pos];
            let extra = copy_at(to_insert, prev) + copy_at(from_insert, next) - data.dm(prev, next);
            if extra < best {
                best = extra;
                best_pos = pos;
            }
        }
        best_pos
    }

    fn find_best_insertion(
        route: &[usize],
        nodes: &[usize],
        available: &[bool],
        route_demand: i32,
        data: &Problem,
        times: &mut RouteTimes,
    ) -> Option<(usize, usize)> {
        let len = route.len();
        let last = len - 1;
        times.depart.clear();
        times.depart.resize(len, 0);
        times.edge.clear();
        times.edge.resize(len, 0);
        times.late.clear();
        times.late.resize(len, 0);
        let depart = &mut times.depart;
        let edge = &mut times.edge;
        let late = &mut times.late;

        let mut curr_time = 0;
        let mut curr_node = 0;
        for pos in 1..len {
            let next_node = copy_at(route, pos);
            let next_nd = data.nd(next_node);
            let travel = data.dm(curr_node, next_node);
            curr_time = next_nd.start_tw.max(curr_time + travel) + next_nd.service_time;
            unsafe {
                *edge.get_unchecked_mut(pos) = travel;
                *depart.get_unchecked_mut(pos) = curr_time;
            }
            curr_node = next_node;
        }
        unsafe {
            *late.get_unchecked_mut(last) = data.nd(copy_at(route, last)).end_tw;
        }
        for pos in (1..last).rev() {
            let nd = data.nd(copy_at(route, pos));
            let bound = copy_at(late, pos + 1) - copy_at(edge, pos + 1) - nd.service_time;
            unsafe {
                *late.get_unchecked_mut(pos) = nd.end_tw.min(bound);
            }
        }

        let max_capacity = data.max_capacity;
        let mut best_c2 = i32::MIN;
        let mut best_node = 0usize;
        let mut best_pos = 0usize;
        for &insert_node in nodes.iter() {
            let insert_nd = data.nd(insert_node);

            let skip = !copy_at(available, insert_node)
                | (route_demand + insert_nd.demand > max_capacity);
            if skip {
                continue;
            }

            let from_insert = data.distance_row(insert_node);
            let to_insert = data.distance_column(insert_node);
            let start_tw = insert_nd.start_tw;
            let end_tw = insert_nd.end_tw;
            let leave = insert_nd.service_time;
            let depot_trip = copy_at(to_insert, 0);
            let mut curr_node = 0;

            for pos in 1..len {
                let next_node = copy_at(route, pos);
                let d_in = copy_at(to_insert, curr_node);
                let arrival = start_tw.max(copy_at(depart, pos - 1) + d_in);
                if arrival > end_tw {
                    break;
                }
                let d_out = copy_at(from_insert, next_node);
                let c2 = depot_trip - (d_in + d_out - copy_at(edge, pos));
                let feasible = arrival + leave + d_out <= copy_at(late, pos);
                let better = (c2 > best_c2) & feasible;
                best_c2 = if better { c2 } else { best_c2 };
                best_node = if better { insert_node } else { best_node };
                best_pos = if better { pos } else { best_pos };
                curr_node = next_node;
            }
        }
        if best_c2 == i32::MIN {
            None
        } else {
            Some((best_node, best_pos))
        }
    }
}
}
mod genetic {
use super::compression::ProblemCompression;
use super::constructive::Constructive;
use super::individual::Individual;
use super::local_search::LocalSearch;
use super::params::Params;
use super::population::{BestMetric, ConsensusEpoch, Population};
use super::pred_queue::{IndexDeque, PredQueue};
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

pub(super) const MIN_FRESH_STARTS: usize = 2;
pub(super) const REPAIR_FACTORS: [usize; 3] = [5, 20, 100];

pub(super) fn canonical_routes(routes: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = routes.iter().filter(|r| r.len() > 2).cloned().collect();
    out.sort_unstable();
    out
}

pub struct Genetic {
    pub data: Arc<Problem>,
    pub root_data: Arc<Problem>,
    pub params: Params,
    pub population: Population,
    pub client_expansion: Vec<Vec<usize>>,
    pub fixed_routes: Vec<Vec<usize>>,
    pub inherited: Vec<Vec<Vec<usize>>>,
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
            inherited: Vec::new(),
        }
    }

    fn repair_and_maybe_add(&mut self, ls: &mut LocalSearch, rng: &mut SmallRng) {
        for factor in REPAIR_FACTORS {
            let routes = ls.continue_repair(rng, self.params, factor);
            let (distance, tw, load_excess) = ls.evaluated_metrics();
            let repaired = Individual::new_from_evaluated_routes(
                self.data.as_ref(),
                &self.params,
                routes,
                distance,
                tw,
                load_excess,
            );
            if repaired.load_excess == 0 && repaired.tw_violation == 0 {
                self.population.add(repaired, &self.params);
                return;
            }
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
        it: usize,
    ) {
        let routes_seed = if self.data.root_nb_nodes == 700 && (it == 1 || it == 2) {
            Constructive::discard_randomized_start(self.data.nb_nodes - 1, rng);
            if it == 1 {
                Constructive::build_nearest(self.data.as_ref())
            } else {
                Constructive::build_sweep(self.data.as_ref())
            }
        } else {
            Constructive::build_routes(self.data.as_ref(), rng, it > 0)
        };
        self.educate_and_add(rng, ls, &routes_seed);
    }

    fn educate_and_add(
        &mut self,
        rng: &mut SmallRng,
        ls: &mut LocalSearch,
        routes_seed: &[Vec<usize>],
    ) {
        let routes = ls.run_from_routes(routes_seed, &[], self.params, rng);
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
        routes: &[Vec<usize>],
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
    ) -> Option<Individual> {
        const MIN_NODES_FOR_EVOLUTION: usize = 5;

        let mut ls = LocalSearch::new(Arc::clone(&self.data), self.params);
        self.population.consensus_enabled = false;
        self.population.reset_consensus_epoch();
        let mut reserved_seed_ind =
            seed_routes.map(|seed| self.build_seed_reserved_individual(seed));

        let mut best_metric: Option<BestMetric> = None;
        let migrants: Vec<Vec<Vec<usize>>> = self.inherited.drain(..).collect();
        let nb_migrants = migrants
            .len()
            .min(self.params.mu_start.saturating_sub(MIN_FRESH_STARTS));
        let nb_fresh = self.params.mu_start - nb_migrants;
        for it in 0..nb_fresh {
            self.generate_initial_individual(rng, &mut ls, it);
            self.track_best_and_save(&mut best_metric, save_solution);
        }
        for routes in migrants.iter().take(nb_migrants) {
            self.educate_and_add(rng, &mut ls, routes);
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
                self.apply_compression(&mut ls, &mut reserved_seed_ind);
                self.population.reset_consensus_epoch();
            }
            it_total += 1;
        }

        if let Some(best) = self.population.best_feasible() {
            let routes = self.decompress_active_routes(&best.routes);
            let ind = Individual::new_from_routes(self.root_data.as_ref(), &self.params, routes);
            Some(ind)
        } else {
            None
        }
    }

    fn apply_compression(
        &mut self,
        ls: &mut LocalSearch,
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

        let Some(compression) = ProblemCompression::from_chains(self.data.as_ref(), chains)
        else {
            return;
        };
        let mut compact = compression.compact;
        compact.fixed_distance_offset += fixed_route_distance;
        let removed_routes = complete_routes.len();
        compact.nb_vehicles = compact.nb_vehicles.saturating_sub(removed_routes);

        let compact_data = Arc::new(compact);
        let compact_chains = compression.chains;
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

        let mut all_chains = compact_chains;
        all_chains.extend(complete_routes);

        let (mut new_population, compressed_seed) = if compact_data.nb_nodes <= 1 {
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
        *ls = LocalSearch::new(Arc::clone(&self.data), self.params);
    }

    fn decompress_active_routes(&self, routes: &[Vec<usize>]) -> Vec<Vec<usize>> {
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

    pub(super) fn export_feasible(&self, k: usize) -> Vec<Vec<Vec<usize>>> {
        let sub = &self.population.feasible;
        let count = sub.indivs.len().min(sub.biased_fitness.len());
        let mut order: Vec<usize> = (0..count).collect();
        order.sort_by(|&a, &b| {
            sub.biased_fitness[a]
                .total_cmp(&sub.biased_fitness[b])
                .then_with(|| sub.indivs[a].cost.cmp(&sub.indivs[b].cost))
                .then_with(|| a.cmp(&b))
        });
        let mut keys: Vec<Vec<Vec<usize>>> = Vec::with_capacity(k.min(count));
        let mut out: Vec<Vec<Vec<usize>>> = Vec::with_capacity(k.min(count));
        for idx in order {
            if out.len() >= k {
                break;
            }
            let mut routes = self.decompress_active_routes(&sub.indivs[idx].routes);
            routes.retain(|route| route.len() > 2);
            let key = canonical_routes(&routes);
            if keys.contains(&key) {
                continue;
            }
            keys.push(key);
            out.push(routes);
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
        routes: &[Vec<usize>],
        chains: &[Vec<usize>],
        old_to_new: &[usize],
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

    fn split_linear(&self, giant: &[usize], target_routes: usize) -> Vec<Vec<usize>> {
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
                let nil = queue.nil;
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
                                queue.remove_node(back);
                            }
                            queue.insert_back(new_pred);
                            queue.feas = if queue.feas == nil { new_pred } else { queue.feas };
                            queue.no_warp = if queue.no_warp == nil {
                                queue.feas
                            } else {
                                queue.no_warp
                            };
                        }

                        while let Some(front) = queue.front() {
                            if demand_x - copy_at(&demand_prefix, front) <= cap_limit {
                                break;
                            }
                            queue.remove_node(front);
                        }
                        if queue.head() == nil {
                            continue;
                        }

                        while let Some(b) = waiting.back() {
                            if copy_at(&tw_start, b) + (s_t - copy_at(&s_prefix, b)) > tw_start_t {
                                break;
                            }
                            waiting.pop_back();
                        }
                        waiting.push_back(t);

                        if queue.no_warp != nil {
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
                                    queue.no_warp = queue.succ(no_warp_node);
                                    let next_no_warp = queue.no_warp;
                                    if next_no_warp == nil {
                                        break;
                                    }
                                    no_warp_node = next_no_warp;
                                } else {
                                    break;
                                }
                            }
                        }

                        while queue.feas != nil {
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
                                queue.no_warp = if queue.no_warp == feas_pred {
                                    nil
                                } else {
                                    queue.no_warp
                                };
                                queue.remove_node(feas_pred);
                            }

                            queue.feas = queue.succ(feas_node);
                            if queue.feas == nil {
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
                                queue.no_warp = if queue.no_warp == feas_pred {
                                    nil
                                } else {
                                    queue.no_warp
                                };
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
                    (k - 1, cost_k_minus_1_at_n)
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
            let mut best: Option<(usize, usize)> = None;

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
                    best = if better { Some((rid, pos)) } else { best };
                }
            }

            let (rid, pos) = best.expect("At least one insertion position should exist");
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

    pub fn crossover_ox(
        &self,
        parent1: &[usize],
        parent2: &[usize],
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
}
mod individual {
use super::params::Params;
use super::problem::Problem;
use super::sequence::Sequence;

#[derive(Clone, Debug)]
pub struct Individual {
    pub routes: Vec<Vec<usize>>,
    pub nb_routes: usize,
    pub distance: i32,
    pub tw_violation: i32,
    pub load_excess: i32,
    pub cost: i64,
    pub pred: Vec<usize>,
    pub succ: Vec<usize>,
}

impl Individual {
    pub fn new_from_routes(data: &Problem, params: &Params, routes: Vec<Vec<usize>>) -> Self {
        let (distance, tw_violation, load_excess) = Self::evaluate_routes(data, &routes);
        Self::new_from_evaluated_routes(data, params, routes, distance, tw_violation, load_excess)
    }

    pub fn new_from_evaluated_routes(
        data: &Problem,
        params: &Params,
        routes: Vec<Vec<usize>>,
        distance: i32,
        tw_violation: i32,
        load_excess: i32,
    ) -> Self {
        let cost = Self::compute_penalized_cost(distance, tw_violation, load_excess, params);
        let (pred, succ, nb_routes) = Self::build_pred_succ_and_count(data, &routes);
        Self {
            routes,
            nb_routes,
            distance,
            tw_violation,
            load_excess,
            cost,
            pred,
            succ,
        }
    }

    pub fn evaluate_routes(data: &Problem, routes: &[Vec<usize>]) -> (i32, i32, i32) {
        let mut total_distance: i32 = 0;
        let mut total_tw_violation: i32 = 0;
        let mut total_load_excess: i32 = 0;
        for route in routes {
            if route.is_empty() {
                continue;
            }
            let mut prefix = Sequence::singleton(data, route[0]);
            for position in 1..route.len() {
                let node = Sequence::singleton(data, route[position]);
                prefix = Sequence::join2(data, &prefix, &node);
            }
            total_distance += prefix.distance;
            total_tw_violation += prefix.tw;
            let excess = (prefix.load - data.max_capacity).max(0);
            total_load_excess += excess;
        }
        (total_distance, total_tw_violation, total_load_excess)
    }

    #[inline]
    pub fn compute_penalized_cost(
        distance: i32,
        tw_violation: i32,
        load_excess: i32,
        params: &Params,
    ) -> i64 {
        (distance as i64)
            + (params.penalty_tw as i64) * (tw_violation as i64)
            + (params.penalty_capa as i64) * (load_excess as i64)
    }

    #[inline]
    pub fn recompute_cost(&mut self, params: &Params) {
        self.cost = Self::compute_penalized_cost(
            self.distance,
            self.tw_violation,
            self.load_excess,
            params,
        );
    }

    fn build_pred_succ_and_count(
        data: &Problem,
        routes: &[Vec<usize>],
    ) -> (Vec<usize>, Vec<usize>, usize) {
        let nb_nodes = data.nb_nodes;
        let mut pred = vec![0usize; nb_nodes];
        let mut succ = vec![0usize; nb_nodes];
        let mut seen = vec![false; nb_nodes];
        let mut nb_routes: usize = 0;

        for route in routes {
            if route.len() > 2 {
                nb_routes += 1;
            }
            for window in route.windows(3) {
                let client = window[1];
                debug_assert!(
                    client < nb_nodes && !seen[client],
                    "Client {} appears more than once in routes",
                    client
                );
                unsafe {
                    *seen.get_unchecked_mut(client) = true;
                    *pred.get_unchecked_mut(client) = window[0];
                    *succ.get_unchecked_mut(client) = window[2];
                }
            }
        }
        for client in 1..nb_nodes {
            debug_assert!(seen[client], "Client {} is missing from routes", client);
        }
        (pred, succ, nb_routes)
    }
}
}
mod loader_tig {
use super::problem::{NodeData, Problem};
use tig_challenges::vehicle_routing::*;

pub struct TigLoader;

impl TigLoader {
    pub fn load(challenge: &Challenge) -> Problem {
        let nb_nodes = challenge.num_nodes;
        let nb_vehicles = challenge.fleet_size;

        let mut node_data: Vec<NodeData> = (0..nb_nodes)
            .map(|i| NodeData {
                start_tw: challenge.ready_times[i],
                end_tw: challenge.due_times[i],
                service_time: if i == 0 { 0 } else { challenge.service_time },
                demand: challenge.demands[i],
            })
            .collect();

        let depot_start = node_data[0].start_tw;
        let depot_end = node_data[0].end_tw;
        for i in 1..nb_nodes {
            let d0i = challenge.distance_matrix[0][i];
            let di0 = challenge.distance_matrix[i][0];
            let nd = &mut node_data[i];
            nd.start_tw = nd.start_tw.max(depot_start + d0i);
            nd.end_tw = nd.end_tw.min(depot_end - nd.service_time - di0);
            debug_assert!(
                nd.start_tw <= nd.end_tw,
                "Node {} has empty tightened TW: [{}, {}]",
                i,
                nd.start_tw,
                nd.end_tw
            );
        }

        let total_demand: i64 = challenge.demands.iter().map(|&d| d as i64).sum();
        let lb_vehicles = ((total_demand + challenge.max_capacity as i64 - 1)
            / challenge.max_capacity as i64) as usize;
        let distance_matrix: Vec<i32> = challenge
            .distance_matrix
            .iter()
            .flatten()
            .copied()
            .collect();
        let distance_matrix_transposed = Problem::transpose_distances(&distance_matrix, nb_nodes);
        Problem {
            seed: challenge.seed,
            nb_nodes,
            root_nb_nodes: nb_nodes,
            nb_vehicles,
            lb_vehicles,
            fixed_distance_offset: 0,
            node_positions: challenge.node_positions.clone(),
            max_capacity: challenge.max_capacity,
            distance_matrix,
            distance_matrix_transposed,
            node_data,
        }
    }
}
}
mod local_search {
use super::params::Params;
use super::problem::Problem;
use super::sequence::Sequence;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use std::cmp::min;
use std::sync::Arc;

const NO_MOVE: i64 = i64::MAX;

#[inline(always)]
fn copy_at<T: Copy>(slice: &[T], index: usize) -> T {
    debug_assert!(index < slice.len());
    unsafe { *slice.get_unchecked(index) }
}

#[inline(always)]
fn node_at(nodes: &[Node], index: usize) -> &Node {
    debug_assert!(index < nodes.len());
    unsafe { nodes.get_unchecked(index) }
}

#[inline(always)]
fn route_at(routes: &[Route], index: usize) -> &Route {
    debug_assert!(index < routes.len());
    unsafe { routes.get_unchecked(index) }
}

#[derive(Default)]
pub struct Node {
    id: usize,
    dist_to_succ: i32,
    seq0_i: Sequence,
    seqi_n: Sequence,
    seq1: Sequence,
    seq12: Sequence,
    seq21: Sequence,
    seq123: Sequence,
}

impl Node {
    #[inline]
    fn new(data: &Problem, id: usize) -> Self {
        Self {
            id,
            seq1: Sequence::singleton(data, id),
            ..Default::default()
        }
    }
}

#[derive(Default)]
pub struct Route {
    cost: i64,
    distance: i32,
    load: i32,
    tw: i32,
    nodes: Vec<Node>,
}

impl Route {
    #[inline(always)]
    fn node(&self, position: usize) -> &Node {
        debug_assert!(position < self.nodes.len());
        unsafe { self.nodes.get_unchecked(position) }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ArcMin {
    version: u32,
    value: i32,
}

#[derive(Clone, Copy, Default)]
struct Slot {
    distance: i32,
    prefix_end: i32,
    suffix_latest: i32,
    warp: i32,
    pred: u16,
    next: u16,
    position: u32,
}

impl Slot {
    #[inline(always)]
    fn from(left: &Sequence, right: &Sequence, position: usize) -> Self {
        Self {
            distance: left.distance + right.distance,
            prefix_end: left.earliest_end,
            suffix_latest: right.tau_plus,
            warp: left.tw + right.tw,
            pred: left.last_node,
            next: right.first_node,
            position: position as u32,
        }
    }
}

#[derive(Clone, Default)]
struct RemovalSlots {
    version: u32,
    entries: Vec<Slot>,
}

pub struct LocalSearch {
    pub data: Arc<Problem>,
    neighbors_before: Vec<usize>,
    neighbors_before_offsets: Vec<usize>,
    neighbors_capacity_swap: Vec<usize>,
    neighbors_capacity_swap_offsets: Vec<usize>,
    pub loop_order_nodes: Vec<usize>,
    pub params: Params,
    pub cost: i64,
    pub routes: Vec<Route>,
    pub node_route: Vec<usize>,
    pub node_pos: Vec<usize>,
    pub empty_routes: Vec<usize>,
    empty_route_pos: Vec<usize>,
    pub when_last_modified: Vec<usize>,
    pub when_last_tested: Vec<usize>,
    pub nb_moves: usize,
    pub move_credit: i64,
    move_bar: i64,
    last_plan: MovePlan,
    route_version: Vec<u32>,
    version_counter: u32,
    arc_min: Vec<ArcMin>,
    removal_slots: Vec<RemovalSlots>,
    cut: Vec<i32>,
    batch: bool,
    batch_ok: bool,
    total_demand: i64,
}

const CUT: usize = 32;
const CUT_R_LOAD: usize = 0;
const CUT_R_DIST: usize = 1;
const CUT_R_COST: usize = 2;
const CUT_V: usize = 3;
const CUT_Y: usize = 4;
const CUT_YN: usize = 5;
const CUT_Y2N: usize = 6;
const CUT_REM0: usize = 7;
const CUT_V_DTS: usize = 8;
const CUT_Y_DTS: usize = 9;
const CUT_YN_DTS: usize = 10;
const CUT_V_LOAD: usize = 11;
const CUT_Y_LOAD: usize = 12;
const CUT_YN_LOAD: usize = 13;
const CUT_D_Y_V: usize = 14;
const CUT_FLAGS: usize = 15;
const CUT_P_END: usize = 16;
const CUT_P_TW: usize = 17;
const CUT_P_LOAD: usize = 18;
const CUT_P_DIST: usize = 19;
const CUT_S_LATEST: usize = 20;
const CUT_S_TW: usize = 21;
const CUT_S_LOAD: usize = 22;
const CUT_S_DIST: usize = 23;
const CUT_RM_PRED: usize = 24;
const CUT_RM_NEXT: usize = 25;
const CUT_RM_ARCS: usize = 26;
const CUT_RM_BRIDGE: usize = 27;
const CUT_RM_LOAD: usize = 28;
const CUT_RID: usize = 29;
const COST_CAP: i64 = 1 << 29;
const LANE_MASK: i32 = 1 << 28;

#[derive(Clone, Copy, Default)]
struct InterSrc {
    r1_load: i32,
    r1_dist: i32,
    r1_cost: i32,
    rem1_1: i32,
    rem1_2: i32,
    rem1_3: i32,
    send1_1: i32,
    send1_2: i32,
    send1_3: i32,
    d_upred_x: i32,
    d_upred_xnext: i32,
    d_upred_x2next: i32,
    d_x_u: i32,
    d_u_x: i32,
    d_x_xnext: i32,
    xd: bool,
    swap3: bool,
    u: usize,
    x: usize,
    xnext: usize,
    x2next: usize,
    upred: usize,
    left1_end: i32,
    left1_tw: i32,
    left1_load: i32,
    left1_dist: i32,
    right2_latest: i32,
    right2_tw: i32,
    right2_load: i32,
    right2_dist: i32,
    valid: bool,
    t_y: usize,
    t_yn: usize,
    t_y2n: usize,
    t_rem2_0: i32,
    t_v_dts: i32,
    t_y_dts: i32,
    t_yn_dts: i32,
    t_v_load: i32,
    t_y_load: i32,
    t_yn_load: i32,
    t_d_y_v: i32,
    t_yd: bool,
    t_ynd: bool,
}

#[derive(Clone, Copy, Default)]
struct BatchOut {
    inter_pass: u32,
    two_ok: u32,
    rev_pass: u32,
    scalar: u32,
    two_lb: [i32; 8],
    two_tw: [i32; 8],
}

#[derive(Clone, Copy, Debug)]
enum CandidateMove {
    InterRoute {
        r1: usize,
        pos1: usize,
        r2: usize,
        pos2: usize,
    },
    TwoOptStar {
        r1: usize,
        pos1: usize,
        r2: usize,
        pos2: usize,
    },
    SwapStar {
        r1: usize,
        pos1: usize,
        r2: usize,
        pos2: usize,
    },
    IntraRelocate {
        r1: usize,
        pos1: usize,
    },
    IntraOrOpt2 {
        r1: usize,
        pos1: usize,
    },
    IntraSwap {
        r1: usize,
        pos1: usize,
    },
    Intra2Opt {
        r1: usize,
        pos1: usize,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum MovePlan {
    #[default]
    None,
    InterRoute {
        send1: u8,
        send2: u8,
    },
    TwoOptStar,
    SwapStar {
        insert1: usize,
        insert2: usize,
    },
    IntraRelocate {
        target: usize,
    },
    IntraOrOpt2 {
        target: usize,
        reversed: bool,
    },
    IntraSwap {
        target: usize,
    },
    Intra2Opt {
        end: usize,
    },
}

impl LocalSearch {
    pub fn new(data: Arc<Problem>, params: Params) -> Self {
        let n = data.nb_nodes;
        debug_assert!(n <= (u16::MAX as usize) + 1);
        let cap = n.saturating_sub(2);
        let keep = min(params.granularity, cap);
        let mut neighbors_before = Vec::with_capacity(n.saturating_sub(1) * keep);
        let mut neighbors_before_offsets = vec![0usize; n + 1];
        let mut prox: Vec<u64> = Vec::with_capacity(cap);
        let mut neighbors_capacity_swap =
            Vec::with_capacity(n.saturating_sub(1) * params.granularity2);
        let mut neighbors_capacity_swap_offsets = vec![0usize; n + 1];
        let mut capacity_prox: Vec<u64> = Vec::with_capacity(cap);
        let tw_shifted: Vec<(i32, i32)> = data
            .node_data
            .iter()
            .map(|nd| (nd.start_tw + nd.service_time, nd.end_tw + nd.service_time))
            .collect();
        let proximity_swap = data.root_nb_nodes == 800;
        let mut ids_by_demand: Vec<Vec<usize>> = Vec::new();
        if !proximity_swap {
            let max_demand = data
                .node_data
                .iter()
                .map(|nd| nd.demand)
                .max()
                .unwrap_or(0)
                .max(0) as usize;
            ids_by_demand = vec![Vec::new(); max_demand + 1];
            for j in 1..n {
                ids_by_demand[data.nd(j).demand as usize].push(j);
            }
        }
        for i in 1..n {
            let ndi = data.nd(i);
            let distance_from_i = data.distance_row(i);
            let distance_to_i = data.distance_column(i);
            let (start_service_i, end_service_i) = copy_at(&tw_shifted, i);
            prox.clear();
            capacity_prox.clear();
            for j in 1..n {
                if j == i {
                    continue;
                }
                let tji = copy_at(distance_to_i, j);
                let (start_service_j, end_service_j) = copy_at(&tw_shifted, j);
                let wait = (ndi.start_tw - tji - end_service_j).max(0);
                let late = (start_service_j + tji - ndi.end_tw).max(0);
                let proxy10 = 10 * tji + 2 * wait + 10 * late;
                prox.push(((proxy10 as u64) << 16) | (j as u64));
                if proximity_swap {
                    let ndj = data.nd(j);
                    let tij = copy_at(distance_from_i, j);
                    let wait_ij = (ndj.start_tw - tij - end_service_i).max(0);
                    let late_ij = (start_service_i + tij - ndj.end_tw).max(0);
                    let gamma_ji = 10 * (tji as i64) + 2 * (wait as i64) + 10 * (late as i64);
                    let gamma_ij =
                        10 * (tij as i64) + 2 * (wait_ij as i64) + 10 * (late_ij as i64);
                    let gamma = gamma_ji.min(gamma_ij) as u64;
                    capacity_prox.push((gamma << 16) | (j as u64));
                }
            }

            if keep > 0 {
                if keep < prox.len() {
                    prox.select_nth_unstable(keep);
                }
                prox[..keep].sort_unstable();
                neighbors_before.extend(prox[..keep].iter().map(|&key| (key & 0xffff) as usize));
            }
            neighbors_before_offsets[i + 1] = neighbors_before.len();

            if !proximity_swap {
                let keep_similar = (params.swapstar_capa_filter * cap as f64).ceil() as usize;
                let max_demand = ids_by_demand.len().saturating_sub(1);
                let di_usize = ndi.demand as usize;
                for dd in 0..=max_demand {
                    if capacity_prox.len() >= keep_similar {
                        break;
                    }
                    let low = di_usize.checked_sub(dd).filter(|&d| d <= max_demand);
                    let high_value = di_usize + dd;
                    let high = (high_value <= max_demand && Some(high_value) != low)
                        .then_some(high_value);
                    let low_ids = low.map(|d| ids_by_demand[d].as_slice()).unwrap_or(&[]);
                    let high_ids = high.map(|d| ids_by_demand[d].as_slice()).unwrap_or(&[]);
                    let mut il = 0usize;
                    let mut ih = 0usize;
                    while capacity_prox.len() < keep_similar
                        && (il < low_ids.len() || ih < high_ids.len())
                    {
                        let take_low = ih >= high_ids.len()
                            || (il < low_ids.len() && low_ids[il] < high_ids[ih]);
                        let j = if take_low {
                            let j = low_ids[il];
                            il += 1;
                            j
                        } else {
                            let j = high_ids[ih];
                            ih += 1;
                            j
                        };
                        if j != i {
                            let d = copy_at(distance_from_i, j) as u64;
                            capacity_prox.push((d << 32) | ((dd as u64) << 16) | (j as u64));
                        }
                    }
                }
                debug_assert_eq!(capacity_prox.len(), keep_similar);
            }

            let m = capacity_prox.len().min(params.granularity2);
            if m > 0 {
                if m < capacity_prox.len() {
                    capacity_prox.select_nth_unstable(m);
                }
                capacity_prox[..m].sort_unstable();
            }
            neighbors_capacity_swap.extend(capacity_prox[..m].iter().filter_map(|&key| {
                let j = (key & 0xffff) as usize;
                if j < i {
                    Some(j)
                } else {
                    None
                }
            }));
            neighbors_capacity_swap_offsets[i + 1] = neighbors_capacity_swap.len();
        }

        let padded = neighbors_before.len() + 8;
        neighbors_before.resize(padded, 0);
        let padded = neighbors_capacity_swap.len() + 8;
        neighbors_capacity_swap.resize(padded, 0);
        let total_demand: i64 = data.node_data.iter().map(|nd| nd.demand as i64).sum();

        Self {
            data,
            neighbors_before,
            neighbors_before_offsets,
            neighbors_capacity_swap,
            neighbors_capacity_swap_offsets,
            loop_order_nodes: (1..n).collect(),
            params,
            cost: 0,
            routes: Vec::new(),
            node_route: Vec::new(),
            node_pos: Vec::new(),
            empty_routes: Vec::new(),
            empty_route_pos: Vec::new(),
            when_last_modified: Vec::new(),
            when_last_tested: vec![0; n],
            nb_moves: 0,
            move_credit: 0,
            move_bar: 0,
            last_plan: MovePlan::None,
            route_version: Vec::new(),
            version_counter: 0,
            arc_min: Vec::new(),
            removal_slots: vec![RemovalSlots::default(); n],
            cut: vec![0; n * CUT],
            batch: Self::batch_supported(),
            batch_ok: false,
            total_demand,
        }
    }

    #[inline]
    fn batch_supported() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    #[inline]
    fn register_accepted_delta(&mut self, delta: i64) {
        if self.move_credit < 0 {
            return;
        }
        if delta < 0 {
            let cap = self.params.max_credit_deterioration as i64;
            self.move_credit = (self.move_credit - delta).min(cap);
        } else if delta > 0 {
            debug_assert!(
                delta <= self.move_credit,
                "Accepted deterioration exceeds available credit"
            );
            self.move_credit -= delta;
        }
    }

    #[inline]
    fn finish_planned_one(&mut self, route: usize, old_cost: i64) -> i64 {
        self.nb_moves += 1;
        self.update_route(route);
        let delta = route_at(&self.routes, route).cost - old_cost;
        self.cost += delta;
        self.register_accepted_delta(delta);
        delta
    }

    #[inline]
    fn finish_planned_two(&mut self, route1: usize, route2: usize, old_cost: i64) -> i64 {
        self.nb_moves += 1;
        self.update_route(route1);
        self.update_route(route2);
        let delta =
            route_at(&self.routes, route1).cost + route_at(&self.routes, route2).cost - old_cost;
        self.cost += delta;
        self.register_accepted_delta(delta);
        delta
    }

    fn apply_planned_move(&mut self, mv: CandidateMove, plan: MovePlan) -> Option<i64> {
        match (mv, plan) {
            (CandidateMove::IntraRelocate { r1, pos1 }, MovePlan::IntraRelocate { target }) => {
                let old_cost = route_at(&self.routes, r1).cost;
                let insert_pos = if target > pos1 { target - 1 } else { target };
                let elem = self.routes[r1].nodes.remove(pos1);
                self.routes[r1].nodes.insert(insert_pos, elem);
                Some(self.finish_planned_one(r1, old_cost))
            }
            (
                CandidateMove::IntraOrOpt2 { r1, pos1 },
                MovePlan::IntraOrOpt2 { target, reversed },
            ) => {
                let old_cost = route_at(&self.routes, r1).cost;
                let insert_pos = if target > pos1 { target - 2 } else { target };
                let n1 = self.routes[r1].nodes.remove(pos1);
                let n2 = self.routes[r1].nodes.remove(pos1);
                let (a, b) = if reversed { (n2, n1) } else { (n1, n2) };
                self.routes[r1].nodes.insert(insert_pos, a);
                self.routes[r1].nodes.insert(insert_pos + 1, b);
                Some(self.finish_planned_one(r1, old_cost))
            }
            (CandidateMove::IntraSwap { r1, pos1 }, MovePlan::IntraSwap { target }) => {
                let old_cost = route_at(&self.routes, r1).cost;
                self.routes[r1].nodes.swap(pos1, target);
                Some(self.finish_planned_one(r1, old_cost))
            }
            (CandidateMove::Intra2Opt { r1, pos1 }, MovePlan::Intra2Opt { end }) => {
                let old_cost = route_at(&self.routes, r1).cost;
                self.routes[r1].nodes[pos1..=end].reverse();
                Some(self.finish_planned_one(r1, old_cost))
            }
            (CandidateMove::TwoOptStar { r1, pos1, r2, pos2 }, MovePlan::TwoOptStar) => {
                let old_cost = route_at(&self.routes, r1).cost + route_at(&self.routes, r2).cost;
                let mut suffix1 = self.routes[r1].nodes.split_off(pos1);
                let mut suffix2 = self.routes[r2].nodes.split_off(pos2);
                self.routes[r1].nodes.append(&mut suffix2);
                self.routes[r2].nodes.append(&mut suffix1);
                Some(self.finish_planned_two(r1, r2, old_cost))
            }
            (
                CandidateMove::SwapStar { r1, pos1, r2, pos2 },
                MovePlan::SwapStar { insert1, insert2 },
            ) => {
                let old_cost = route_at(&self.routes, r1).cost + route_at(&self.routes, r2).cost;
                let node_u = self.routes[r1].nodes.remove(pos1);
                let node_v = self.routes[r2].nodes.remove(pos2);
                let ins1 = if insert1 > pos1 { insert1 - 1 } else { insert1 };
                let ins2 = if insert2 > pos2 { insert2 - 1 } else { insert2 };
                self.routes[r1].nodes.insert(ins1, node_v);
                self.routes[r2].nodes.insert(ins2, node_u);
                Some(self.finish_planned_two(r1, r2, old_cost))
            }
            (
                CandidateMove::InterRoute { r1, pos1, r2, pos2 },
                MovePlan::InterRoute { send1, send2 },
            ) => {
                let old_cost = route_at(&self.routes, r1).cost + route_at(&self.routes, r2).cost;
                let mut take_block = |route_idx: usize, pos: usize, kind: u8| -> Vec<Node> {
                    let nodes = &mut self.routes[route_idx].nodes;
                    match kind {
                        0 => vec![],
                        1 => vec![nodes.remove(pos)],
                        2 => {
                            let n1 = nodes.remove(pos);
                            let n2 = nodes.remove(pos);
                            vec![n1, n2]
                        }
                        3 => {
                            let n1 = nodes.remove(pos);
                            let n2 = nodes.remove(pos);
                            vec![n2, n1]
                        }
                        4 => {
                            let n1 = nodes.remove(pos);
                            let n2 = nodes.remove(pos);
                            let n3 = nodes.remove(pos);
                            vec![n1, n2, n3]
                        }
                        _ => unreachable!("invalid inter-route block kind"),
                    }
                };
                let block1 = take_block(r1, pos1, send1);
                let block2 = take_block(r2, pos2, send2);
                for (offset, node) in block2.into_iter().enumerate() {
                    self.routes[r1].nodes.insert(pos1 + offset, node);
                }
                for (offset, node) in block1.into_iter().enumerate() {
                    self.routes[r2].nodes.insert(pos2 + offset, node);
                }
                Some(self.finish_planned_two(r1, r2, old_cost))
            }
            _ => None,
        }
    }

    pub fn run_from_routes(
        &mut self,
        routes: &[Vec<usize>],
        inherited_routes: &[bool],
        params: Params,
        rng: &mut SmallRng,
    ) -> Vec<Vec<usize>> {
        let mut routes = routes.to_vec();
        self.params = params;
        let n = self.data.nb_nodes;
        let fleet = self.data.nb_vehicles;

        if routes.len() <= fleet {
            routes.resize(fleet, vec![0, 0]);
        } else {
            let keep = fleet.saturating_sub(1);
            let extras = routes.split_off(fleet);
            let merged = &mut routes[keep];
            merged.pop();
            for r in &extras {
                if r.len() > 2 {
                    merged.extend_from_slice(&r[1..r.len() - 1]);
                }
            }
            merged.push(0);
            debug_assert_eq!(routes.len(), fleet);
        }

        self.routes.resize_with(routes.len(), Route::default);
        for (dst, ids) in self.routes.iter_mut().zip(routes.iter()) {
            dst.nodes.clear();
            dst.nodes.extend(
                ids.iter()
                    .copied()
                    .map(|id| Node::new(self.data.as_ref(), id)),
            );
        }
        self.node_route.clear();
        self.node_route.resize(n, 0);
        self.node_pos.clear();
        self.node_pos.resize(n, 0);
        self.empty_routes.clear();
        self.empty_route_pos.clear();
        self.empty_route_pos.resize(self.routes.len(), usize::MAX);
        self.when_last_modified.clear();
        self.when_last_modified.resize(self.routes.len(), 0);
        self.route_version.clear();
        self.route_version.resize(self.routes.len(), 0);
        let cache_len = self.routes.len() * n;
        if self.arc_min.len() != cache_len {
            self.arc_min.clear();
            self.arc_min.resize(cache_len, ArcMin::default());
        }
        self.when_last_tested.clear();
        self.when_last_tested.resize(n, 0);
        self.nb_moves = 1;

        for rid in 0..self.routes.len() {
            self.update_route(rid);
        }
        if !inherited_routes.is_empty() {
            debug_assert_eq!(
                inherited_routes.len(),
                self.routes.len(),
                "inherited_routes size must match"
            );
        }
        for rid in 0..self.routes.len() {
            let r = &self.routes[rid];
            let is_feasible = r.load <= self.data.max_capacity && r.tw == 0;
            let inherited = !inherited_routes.is_empty() && inherited_routes[rid];
            self.when_last_modified[rid] = if inherited && is_feasible {
                0
            } else {
                self.nb_moves
            };
        }
        self.cost = self.routes.iter().map(|r| r.cost).sum();
        self.search(rng);
        self.export_routes()
    }

    fn export_routes(&self) -> Vec<Vec<usize>> {
        self.routes
            .iter()
            .filter(|r| r.nodes.len() > 2)
            .map(|r| r.nodes.iter().map(|n| n.id).collect::<Vec<usize>>())
            .collect()
    }

    #[inline]
    pub fn evaluated_metrics(&self) -> (i32, i32, i32) {
        let mut distance = 0;
        let mut tw_violation = 0;
        let mut load_excess = 0;
        for route in &self.routes {
            distance += route.distance;
            tw_violation += route.tw;
            load_excess += (route.load - self.data.max_capacity).max(0);
        }
        (distance, tw_violation, load_excess)
    }

    fn update_route(&mut self, rid: usize) {
        let data = self.data.as_ref();
        debug_assert!(rid < self.routes.len());
        let r = unsafe { self.routes.get_unchecked_mut(rid) };
        let nodes = &mut r.nodes;
        let len = nodes.len();
        debug_assert!(len >= 2);
        let ptr = nodes.as_mut_ptr();

        let mut acc_fwd = unsafe { (*ptr).seq1 };
        let mut prev_id = unsafe { (*ptr).id };
        unsafe {
            (*ptr).seq0_i = acc_fwd;
            *self.node_route.get_unchecked_mut(prev_id) = rid;
            *self.node_pos.get_unchecked_mut(prev_id) = 0;
        }
        for pos in 1..len {
            let id = unsafe { (*ptr.add(pos)).id };
            let travel = data.dm(prev_id, id);
            let singleton = unsafe { (*ptr.add(pos)).seq1 };
            acc_fwd = Sequence::join2_with_travel(&acc_fwd, &singleton, travel);
            unsafe {
                (*ptr.add(pos - 1)).dist_to_succ = travel;
                (*ptr.add(pos)).seq0_i = acc_fwd;
                *self.node_route.get_unchecked_mut(id) = rid;
                *self.node_pos.get_unchecked_mut(id) = pos;
            }
            prev_id = id;
        }
        unsafe {
            (*ptr.add(len - 1)).dist_to_succ = 0;
        }

        let swap3 = self.params.allow_swap3;
        let mut acc_bwd = unsafe { (*ptr.add(len - 1)).seq1 };
        unsafe {
            (*ptr.add(len - 1)).seqi_n = acc_bwd;
        }
        let last_pair = len - 2;
        {
            let pos = last_pair;
            let id = unsafe { (*ptr.add(pos)).id };
            let singleton = unsafe { (*ptr.add(pos)).seq1 };
            let next = unsafe { (*ptr.add(pos + 1)).seq1 };
            let forward_travel = unsafe { (*ptr.add(pos)).dist_to_succ };
            acc_bwd = Sequence::join2_with_travel(&singleton, &acc_bwd, forward_travel);
            let seq12 = Sequence::join2_with_travel(&singleton, &next, forward_travel);
            let seq21 = Sequence::join2_with_travel(
                &next,
                &singleton,
                data.dm(unsafe { (*ptr.add(pos + 1)).id }, id),
            );
            unsafe {
                (*ptr.add(pos)).seqi_n = acc_bwd;
                (*ptr.add(pos)).seq12 = seq12;
                (*ptr.add(pos)).seq21 = seq21;
            }
        }
        for pos in (0..last_pair).rev() {
            let id = unsafe { (*ptr.add(pos)).id };
            let singleton = unsafe { (*ptr.add(pos)).seq1 };
            let next = unsafe { (*ptr.add(pos + 1)).seq1 };
            let forward_travel = unsafe { (*ptr.add(pos)).dist_to_succ };
            acc_bwd = Sequence::join2_with_travel(&singleton, &acc_bwd, forward_travel);
            let seq12 = Sequence::join2_with_travel(&singleton, &next, forward_travel);
            let seq21 = Sequence::join2_with_travel(
                &next,
                &singleton,
                data.dm(unsafe { (*ptr.add(pos + 1)).id }, id),
            );
            unsafe {
                (*ptr.add(pos)).seqi_n = acc_bwd;
                (*ptr.add(pos)).seq12 = seq12;
                (*ptr.add(pos)).seq21 = seq21;
            }
            if swap3 {
                let next2 = unsafe { (*ptr.add(pos + 2)).seq1 };
                let travel = unsafe { (*ptr.add(pos + 1)).dist_to_succ };
                unsafe {
                    (*ptr.add(pos)).seq123 =
                        Sequence::join2_with_travel(&seq12, &next2, travel);
                }
            }
        }

        let end = unsafe { (*ptr.add(len - 1)).seq0_i };
        r.load = end.load;
        r.tw = end.tw;
        r.distance = end.distance;
        r.cost = end.eval(data, &self.params);

        {
            let r_cost = r.cost.min(COST_CAP) as i32;
            let last = len - 1;
            let cut = self.cut.as_mut_ptr();
            for pos in 1..last {
                let c = unsafe { &*ptr.add(pos) };
                let v = unsafe { &*ptr.add(pos + 1) };
                let vd = (v.id == 0) as usize;
                let j_y = pos + 2 - vd;
                let y = unsafe { &*ptr.add(j_y) };
                let yd = (y.id == 0) as usize;
                let j_yn = j_y + 1 - yd;
                let yn = unsafe { &*ptr.add(j_yn) };
                let ynd = (yn.id == 0) as usize;
                let y2n = unsafe { &*ptr.add(j_yn + 1 - ynd) };
                let rec = unsafe { cut.add(c.id * CUT) };
                unsafe {
                    *rec.add(CUT_R_LOAD) = end.load;
                    *rec.add(CUT_R_DIST) = end.distance;
                    *rec.add(CUT_R_COST) = r_cost;
                    *rec.add(CUT_V) = v.id as i32;
                    *rec.add(CUT_Y) = y.id as i32;
                    *rec.add(CUT_YN) = yn.id as i32;
                    *rec.add(CUT_Y2N) = y2n.id as i32;
                    *rec.add(CUT_REM0) = c.dist_to_succ;
                    *rec.add(CUT_V_DTS) = v.dist_to_succ;
                    *rec.add(CUT_Y_DTS) = y.dist_to_succ;
                    *rec.add(CUT_YN_DTS) = yn.dist_to_succ;
                    *rec.add(CUT_V_LOAD) = v.seq1.load;
                    *rec.add(CUT_Y_LOAD) = y.seq1.load;
                    *rec.add(CUT_YN_LOAD) = yn.seq1.load;
                    *rec.add(CUT_D_Y_V) = data.dm(y.id, v.id);
                    *rec.add(CUT_FLAGS) = (vd | (yd << 1) | (ynd << 2)) as i32;
                    *rec.add(CUT_P_END) = c.seq0_i.earliest_end;
                    *rec.add(CUT_P_TW) = c.seq0_i.tw;
                    *rec.add(CUT_P_LOAD) = c.seq0_i.load;
                    *rec.add(CUT_P_DIST) = c.seq0_i.distance;
                    *rec.add(CUT_S_LATEST) = v.seqi_n.tau_plus;
                    *rec.add(CUT_S_TW) = v.seqi_n.tw;
                    *rec.add(CUT_S_LOAD) = v.seqi_n.load;
                    *rec.add(CUT_S_DIST) = v.seqi_n.distance;
                    let p = &*ptr.add(pos - 1);
                    *rec.add(CUT_RM_PRED) = p.id as i32;
                    *rec.add(CUT_RM_NEXT) = v.id as i32;
                    *rec.add(CUT_RM_ARCS) = p.dist_to_succ + c.dist_to_succ;
                    *rec.add(CUT_RM_BRIDGE) = data.dm(p.id, v.id);
                    *rec.add(CUT_RM_LOAD) = c.seq1.load;
                    *rec.add(CUT_RID) = rid as i32;
                }
            }
        }

        let is_empty = len == 2;
        debug_assert!(rid < self.empty_route_pos.len());
        let pos = unsafe { *self.empty_route_pos.get_unchecked(rid) };
        if is_empty && pos == usize::MAX {
            unsafe {
                *self.empty_route_pos.get_unchecked_mut(rid) = self.empty_routes.len();
            }
            self.empty_routes.push(rid);
        } else if !is_empty && pos != usize::MAX {
            self.empty_routes.swap_remove(pos);
            unsafe {
                *self.empty_route_pos.get_unchecked_mut(rid) = usize::MAX;
            }
            if pos < self.empty_routes.len() {
                let moved_rid = unsafe { *self.empty_routes.get_unchecked(pos) };
                debug_assert!(moved_rid < self.empty_route_pos.len());
                unsafe {
                    *self.empty_route_pos.get_unchecked_mut(moved_rid) = pos;
                }
            }
        }
        debug_assert!(rid < self.when_last_modified.len());
        self.version_counter += 1;
        unsafe {
            *self.when_last_modified.get_unchecked_mut(rid) = self.nb_moves;
            *self.route_version.get_unchecked_mut(rid) = self.version_counter;
        }
    }

    #[inline(always)]
    fn arc_scan<const HOLE: usize>(
        nodes: &[Node],
        pos: usize,
        from_arc: usize,
        to_c: &[i32],
        from_c: &[i32],
    ) -> i32 {
        let len = nodes.len();
        let live = if HOLE == 0 { len - 1 - from_arc } else { len - HOLE - 2 };
        if live == 0 {
            return i32::MAX;
        }
        let end = live - 1;
        let split = if HOLE == 0 { usize::MAX } else { pos - 1 };
        let base = if HOLE == 0 { from_arc } else { 0 };
        let mut best = i32::MAX;
        macro_rules! arc {
            ($s:expr) => {{
                let s = min($s, end);
                let t = if s < split { s + base } else { s + HOLE + 1 };
                let pred = node_at(nodes, t);
                let next = node_at(nodes, t + 1);
                best = best.min(copy_at(to_c, pred.id) + copy_at(from_c, next.id) - pred.dist_to_succ);
            }};
        }
        let mut i = 0usize;
        while i < live {
            arc!(i);
            arc!(i + 1);
            arc!(i + 2);
            arc!(i + 3);
            arc!(i + 4);
            arc!(i + 5);
            arc!(i + 6);
            arc!(i + 7);
            arc!(i + 8);
            arc!(i + 9);
            arc!(i + 10);
            arc!(i + 11);
            i += 12;
        }
        best
    }

    fn run_intra_route_relocate(&mut self, r1: usize, pos1: usize) -> i64 {
        let route = route_at(&self.routes, r1);
        let data = self.data.as_ref();
        let len = route.nodes.len();
        if len <= 3 {
            return NO_MOVE;
        }

        debug_assert!(pos1 > 0);
        debug_assert!(self.routes[r1].nodes[pos1].id != 0);

        let old_cost = route.cost;
        let max_acceptable_cost = old_cost + self.move_bar;
        let mut best_cost = i64::MAX;
        let mut best_pos: Option<usize> = None;
        let u_seq1 = route.node(pos1).seq1;
        let old_distance = route.distance as i64;
        let cap_pen =
            ((route.load - self.data.max_capacity).max(0) as i64) * self.params.penalty_capa as i64;
        let ptw = self.params.penalty_tw as i64;
        let max_distance_acceptable = max_acceptable_cost - cap_pen;
        let a_id = route.node(pos1 - 1).id;
        let u_id = route.node(pos1).id;
        let b_id = route.node(pos1 + 1).id;
        let distance_from_u = data.distance_row(u_id);
        let distance_to_u = data.distance_column(u_id);
        let removed_au_ub = route.node(pos1 - 1).dist_to_succ + route.node(pos1).dist_to_succ;
        let add_ab = data.dm(a_id, b_id);
        let fixed_delta = add_ab - removed_au_ub;

        let cheapest = Self::arc_scan::<1>(&route.nodes, pos1, 0, distance_to_u, distance_from_u);
        if old_distance + ((fixed_delta as i64) + (cheapest as i64)) > max_distance_acceptable {
            return NO_MOVE;
        }

        if pos1 > 1 {
            let mut right_excl_u = route.node(pos1 + 1).seqi_n;
            for t in (1..pos1).rev() {
                let node_t = route.node(t);
                let bridge_mask = -((t + 1 == pos1) as i32);
                let join_travel = (add_ab & bridge_mask) | (node_t.dist_to_succ & !bridge_mask);
                right_excl_u =
                    Sequence::join_tw_with_travel(&node_t.seq1, &right_excl_u, join_travel);
                let c_id = route.node(t - 1).id;
                let d_id = route.node(t).id;
                let d_cu = copy_at(distance_to_u, c_id);
                let d_ud = copy_at(distance_from_u, d_id);
                let delta_distance = fixed_delta + d_cu + d_ud - route.node(t - 1).dist_to_succ;
                let eligible = old_distance + (delta_distance as i64) <= max_distance_acceptable;
                let left = route.node(t - 1).seq0_i;
                let tw = Sequence::tw3_with_travel(&left, &u_seq1, &right_excl_u, d_cu, d_ud);
                let new_cost = old_distance + (delta_distance as i64) + cap_pen + (tw as i64) * ptw;
                let better =
                    eligible & (new_cost <= max_acceptable_cost) & (new_cost < best_cost);
                best_cost = if better { new_cost } else { best_cost };
                best_pos = if better { Some(t) } else { best_pos };
            }
        }

        if pos1 + 2 < len {
            let mut left_excl_u = Sequence::join_tw_with_travel(
                &route.node(pos1 - 1).seq0_i,
                &route.node(pos1 + 1).seq1,
                add_ab,
            );
            for t in (pos1 + 2)..len {
                let c_id = route.node(t - 1).id;
                let d_id = route.node(t).id;
                let d_cu = copy_at(distance_to_u, c_id);
                let d_ud = copy_at(distance_from_u, d_id);
                let delta_distance = fixed_delta + d_cu + d_ud - route.node(t - 1).dist_to_succ;
                let eligible = old_distance + (delta_distance as i64) <= max_distance_acceptable;
                let tw = Sequence::tw3_with_travel(
                    &left_excl_u,
                    &u_seq1,
                    &route.node(t).seqi_n,
                    d_cu,
                    d_ud,
                );
                let new_cost =
                    old_distance + (delta_distance as i64) + cap_pen + (tw as i64) * ptw;
                let better = eligible & (new_cost <= max_acceptable_cost) & (new_cost < best_cost);
                best_cost = if better { new_cost } else { best_cost };
                best_pos = if better { Some(t) } else { best_pos };
                left_excl_u = Sequence::join_tw_with_travel(
                    &left_excl_u,
                    &route.node(t).seq1,
                    route.node(t - 1).dist_to_succ,
                );
            }
        }

        if let Some(target) = best_pos {
            let selected_delta = best_cost - old_cost;
            self.last_plan = MovePlan::IntraRelocate { target };
            selected_delta
        } else {
            NO_MOVE
        }
    }

    fn run_intra_route_oropt2(&mut self, r1: usize, pos1: usize) -> i64 {
        let route = route_at(&self.routes, r1);
        let data = self.data.as_ref();
        let len = route.nodes.len();
        if pos1 + 2 >= len {
            return NO_MOVE;
        }

        debug_assert!(pos1 > 0);
        debug_assert!(self.routes[r1].nodes[pos1].id != 0);
        debug_assert!(self.routes[r1].nodes[pos1 + 1].id != 0);

        let old_cost = route.cost;
        let max_acceptable_cost = old_cost + self.move_bar;
        let mut best_cost = i64::MAX;
        let mut best_pos: Option<usize> = None;
        let mut best_reversed = false;
        let pair_fwd = route.node(pos1).seq12;
        let pair_rev = route.node(pos1).seq21;
        let old_distance = route.distance as i64;
        let cap_pen =
            ((route.load - self.data.max_capacity).max(0) as i64) * self.params.penalty_capa as i64;
        let ptw = self.params.penalty_tw as i64;
        let max_distance_acceptable = max_acceptable_cost - cap_pen;
        let a_id = route.node(pos1 - 1).id;
        let u_id = route.node(pos1).id;
        let x_id = route.node(pos1 + 1).id;
        let b_id = route.node(pos1 + 2).id;
        let distance_from_u = data.distance_row(u_id);
        let distance_to_u = data.distance_column(u_id);
        let distance_from_x = data.distance_row(x_id);
        let distance_to_x = data.distance_column(x_id);
        let removed_auxb = route.node(pos1 - 1).dist_to_succ
            + route.node(pos1).dist_to_succ
            + route.node(pos1 + 1).dist_to_succ;
        let add_ab = data.dm(a_id, b_id);
        let fixed_delta_fwd = add_ab + route.node(pos1).dist_to_succ - removed_auxb;
        let fixed_delta_rev = add_ab + copy_at(distance_from_x, u_id) - removed_auxb;

        let cheapest_fwd = Self::arc_scan::<2>(&route.nodes, pos1, 0, distance_to_u, distance_from_x);
        let cheapest_rev = Self::arc_scan::<2>(&route.nodes, pos1, 0, distance_to_x, distance_from_u);
        let cheapest = ((fixed_delta_fwd as i64) + (cheapest_fwd as i64))
            .min((fixed_delta_rev as i64) + (cheapest_rev as i64));
        if old_distance + cheapest > max_distance_acceptable {
            return NO_MOVE;
        }

        if pos1 > 1 {
            let mut right_excl_pair = route.node(pos1 + 2).seqi_n;
            for t in (1..pos1).rev() {
                let node_t = route.node(t);
                let bridge_mask = -((t + 1 == pos1) as i32);
                let join_travel = (add_ab & bridge_mask) | (node_t.dist_to_succ & !bridge_mask);
                right_excl_pair =
                    Sequence::join_tw_with_travel(&node_t.seq1, &right_excl_pair, join_travel);
                let c_id = route.node(t - 1).id;
                let d_id = route.node(t).id;
                let d_cd = route.node(t - 1).dist_to_succ;
                let d_cu = copy_at(distance_to_u, c_id);
                let d_xd = copy_at(distance_from_x, d_id);
                let d_cx = copy_at(distance_to_x, c_id);
                let d_ud = copy_at(distance_from_u, d_id);
                let delta_fwd = fixed_delta_fwd + d_cu + d_xd - d_cd;
                let delta_rev = fixed_delta_rev + d_cx + d_ud - d_cd;
                let can_pass_fwd = old_distance + (delta_fwd as i64) <= max_distance_acceptable;
                let can_pass_rev = old_distance + (delta_rev as i64) <= max_distance_acceptable;

                let left = route.node(t - 1).seq0_i;

                let tw_fwd =
                    Sequence::tw3_with_travel(&left, &pair_fwd, &right_excl_pair, d_cu, d_xd);
                let new_cost_fwd =
                    old_distance + (delta_fwd as i64) + cap_pen + (tw_fwd as i64) * ptw;
                let better = can_pass_fwd
                    & (new_cost_fwd <= max_acceptable_cost)
                    & (new_cost_fwd < best_cost);
                best_cost = if better { new_cost_fwd } else { best_cost };
                best_pos = if better { Some(t) } else { best_pos };
                best_reversed = if better { false } else { best_reversed };

                let tw_rev =
                    Sequence::tw3_with_travel(&left, &pair_rev, &right_excl_pair, d_cx, d_ud);
                let new_cost_rev =
                    old_distance + (delta_rev as i64) + cap_pen + (tw_rev as i64) * ptw;
                let better = can_pass_rev
                    & (new_cost_rev <= max_acceptable_cost)
                    & (new_cost_rev < best_cost);
                best_cost = if better { new_cost_rev } else { best_cost };
                best_pos = if better { Some(t) } else { best_pos };
                best_reversed = if better { true } else { best_reversed };
            }
        }

        if pos1 + 3 < len {
            let mut left_excl_pair = Sequence::join_tw_with_travel(
                &route.node(pos1 - 1).seq0_i,
                &route.node(pos1 + 2).seq1,
                add_ab,
            );
            for t in (pos1 + 3)..len {
                let c_id = route.node(t - 1).id;
                let d_id = route.node(t).id;
                let d_cd = route.node(t - 1).dist_to_succ;
                let d_cu = copy_at(distance_to_u, c_id);
                let d_xd = copy_at(distance_from_x, d_id);
                let d_cx = copy_at(distance_to_x, c_id);
                let d_ud = copy_at(distance_from_u, d_id);
                let delta_fwd = fixed_delta_fwd + d_cu + d_xd - d_cd;
                let delta_rev = fixed_delta_rev + d_cx + d_ud - d_cd;
                let right = route.node(t).seqi_n;

                let can_pass_fwd = old_distance + (delta_fwd as i64) <= max_distance_acceptable;
                let can_pass_rev = old_distance + (delta_rev as i64) <= max_distance_acceptable;
                let tw_fwd =
                    Sequence::tw3_with_travel(&left_excl_pair, &pair_fwd, &right, d_cu, d_xd);
                let new_cost_fwd =
                    old_distance + (delta_fwd as i64) + cap_pen + (tw_fwd as i64) * ptw;
                let better = can_pass_fwd
                    & (new_cost_fwd <= max_acceptable_cost)
                    & (new_cost_fwd < best_cost);
                best_cost = if better { new_cost_fwd } else { best_cost };
                best_pos = if better { Some(t) } else { best_pos };
                best_reversed = if better { false } else { best_reversed };

                let tw_rev =
                    Sequence::tw3_with_travel(&left_excl_pair, &pair_rev, &right, d_cx, d_ud);
                let new_cost_rev =
                    old_distance + (delta_rev as i64) + cap_pen + (tw_rev as i64) * ptw;
                let better = can_pass_rev
                    & (new_cost_rev <= max_acceptable_cost)
                    & (new_cost_rev < best_cost);
                best_cost = if better { new_cost_rev } else { best_cost };
                best_pos = if better { Some(t) } else { best_pos };
                best_reversed = if better { true } else { best_reversed };

                left_excl_pair = Sequence::join_tw_with_travel(
                    &left_excl_pair,
                    &route.node(t).seq1,
                    route.node(t - 1).dist_to_succ,
                );
            }
        }

        if let Some(target) = best_pos {
            let selected_delta = best_cost - old_cost;
            self.last_plan = MovePlan::IntraOrOpt2 {
                target,
                reversed: best_reversed,
            };
            selected_delta
        } else {
            NO_MOVE
        }
    }

    fn run_intra_route_swap(&mut self, r1: usize, pos1: usize) -> i64 {
        let route = route_at(&self.routes, r1);
        let len = route.nodes.len();
        if len <= 4 {
            return NO_MOVE;
        }
        let data = self.data.as_ref();
        let has_right = pos1 + 2 < len - 1;
        let has_left = pos1 >= 3;
        if !has_left && !has_right {
            return NO_MOVE;
        }

        debug_assert!(pos1 > 0);
        debug_assert!(self.routes[r1].nodes[pos1].id != 0);

        let old_cost = route.cost;
        let max_acceptable_cost = old_cost + self.move_bar;
        let mut best_cost = i64::MAX;
        let mut best_pos: Option<usize> = None;
        let old_distance = route.distance as i64;
        let cap_pen =
            ((route.load - self.data.max_capacity).max(0) as i64) * self.params.penalty_capa as i64;
        let ptw = self.params.penalty_tw as i64;
        let max_distance_acceptable = max_acceptable_cost - cap_pen;
        let pu = route.node(pos1 - 1).id;
        let u = route.node(pos1).id;
        let nu = route.node(pos1 + 1).id;
        let distance_from_pu = data.distance_row(pu);
        let distance_to_nu = data.distance_column(nu);
        let distance_from_u = data.distance_row(u);
        let distance_to_u = data.distance_column(u);
        let removed_u = route.node(pos1 - 1).dist_to_succ + route.node(pos1).dist_to_succ;

        macro_rules! fits {
            ($pos2:expr) => {{
                let pos2 = $pos2;
                let pv = route.node(pos2 - 1).id;
                let v = route.node(pos2).id;
                let nv = route.node(pos2 + 1).id;
                let removed_v = route.node(pos2 - 1).dist_to_succ + route.node(pos2).dist_to_succ;
                let delta_distance = (copy_at(distance_from_pu, v) + copy_at(distance_to_nu, v)
                    - removed_u)
                    + (copy_at(distance_to_u, pv) + copy_at(distance_from_u, nv) - removed_v);
                (old_distance + (delta_distance as i64) <= max_distance_acceptable) as u32
            }};
        }
        let mut first_right_potential: Option<usize> = None;
        if has_right {
            let count = len - 1 - (pos1 + 2);
            let last = count - 1;
            let mut c = 0usize;
            while c < count {
                let base = pos1 + 2;
                let mask = (fits!(base + min(c, last))
                    | (fits!(base + min(c + 1, last)) << 1)
                    | (fits!(base + min(c + 2, last)) << 2)
                    | (fits!(base + min(c + 3, last)) << 3)
                    | (fits!(base + min(c + 4, last)) << 4)
                    | (fits!(base + min(c + 5, last)) << 5)
                    | (fits!(base + min(c + 6, last)) << 6)
                    | (fits!(base + min(c + 7, last)) << 7)
                    | (fits!(base + min(c + 8, last)) << 8)
                    | (fits!(base + min(c + 9, last)) << 9)
                    | (fits!(base + min(c + 10, last)) << 10)
                    | (fits!(base + min(c + 11, last)) << 11))
                    & (u32::MAX >> (32 - min(count - c, 12)));
                if mask != 0 {
                    first_right_potential = Some(base + c + mask.trailing_zeros() as usize);
                    break;
                }
                c += 12;
            }
        }
        let mut first_left_potential: Option<usize> = None;
        if has_left {
            let count = pos1 - 2;
            let last = count - 1;
            let mut c = 0usize;
            while c < count {
                let base = pos1 - 2;
                let mask = (fits!(base - min(c, last))
                    | (fits!(base - min(c + 1, last)) << 1)
                    | (fits!(base - min(c + 2, last)) << 2)
                    | (fits!(base - min(c + 3, last)) << 3)
                    | (fits!(base - min(c + 4, last)) << 4)
                    | (fits!(base - min(c + 5, last)) << 5)
                    | (fits!(base - min(c + 6, last)) << 6)
                    | (fits!(base - min(c + 7, last)) << 7)
                    | (fits!(base - min(c + 8, last)) << 8)
                    | (fits!(base - min(c + 9, last)) << 9)
                    | (fits!(base - min(c + 10, last)) << 10)
                    | (fits!(base - min(c + 11, last)) << 11))
                    & (u32::MAX >> (32 - min(count - c, 12)));
                if mask != 0 {
                    first_left_potential = Some(base - c - mask.trailing_zeros() as usize);
                    break;
                }
                c += 12;
            }
        }
        if first_right_potential.is_none() && first_left_potential.is_none() {
            return NO_MOVE;
        }

        if let Some(first_pos2) = first_right_potential {
            let mut acc_mid = route.node(pos1 + 1).seq1;
            for middle in (pos1 + 2)..first_pos2 {
                acc_mid = Sequence::join_tw_with_travel(
                    &acc_mid,
                    &route.node(middle).seq1,
                    route.node(middle - 1).dist_to_succ,
                );
            }
            for pos2 in first_pos2..(len - 1) {
                let pv = route.node(pos2 - 1).id;
                let v = route.node(pos2).id;
                let nv = route.node(pos2 + 1).id;
                let removed_v = route.node(pos2 - 1).dist_to_succ + route.node(pos2).dist_to_succ;
                let d_puv = copy_at(distance_from_pu, v);
                let d_vnu = copy_at(distance_to_nu, v);
                let d_pvu = copy_at(distance_to_u, pv);
                let d_unv = copy_at(distance_from_u, nv);
                let delta_distance = (d_puv + d_vnu - removed_u) + (d_pvu + d_unv - removed_v);
                let eligible = old_distance + (delta_distance as i64) <= max_distance_acceptable;
                let tw = Sequence::tw5_with_travel(
                    &route.node(pos1 - 1).seq0_i,
                    &route.node(pos2).seq1,
                    &acc_mid,
                    &route.node(pos1).seq1,
                    &route.node(pos2 + 1).seqi_n,
                    d_puv,
                    d_vnu,
                    d_pvu,
                    d_unv,
                );
                let new_cost =
                    old_distance + (delta_distance as i64) + cap_pen + (tw as i64) * ptw;
                let better =
                    eligible & (new_cost <= max_acceptable_cost) & (new_cost < best_cost);
                best_cost = if better { new_cost } else { best_cost };
                best_pos = if better { Some(pos2) } else { best_pos };
                acc_mid = Sequence::join_tw_with_travel(
                    &acc_mid,
                    &route.node(pos2).seq1,
                    route.node(pos2 - 1).dist_to_succ,
                );
            }
        }

        if let Some(first_pos2) = first_left_potential {
            let mut acc_mid = route.node(pos1 - 1).seq1;
            for middle in ((first_pos2 + 1)..=(pos1 - 2)).rev() {
                acc_mid = Sequence::join_tw_with_travel(
                    &route.node(middle).seq1,
                    &acc_mid,
                    route.node(middle).dist_to_succ,
                );
            }
            for pos2 in (1..=first_pos2).rev() {
                let pv = route.node(pos2 - 1).id;
                let v = route.node(pos2).id;
                let nv = route.node(pos2 + 1).id;
                let removed_v = route.node(pos2 - 1).dist_to_succ + route.node(pos2).dist_to_succ;
                let d_puv = copy_at(distance_from_pu, v);
                let d_vnu = copy_at(distance_to_nu, v);
                let d_pvu = copy_at(distance_to_u, pv);
                let d_unv = copy_at(distance_from_u, nv);
                let delta_distance = (d_puv + d_vnu - removed_u) + (d_pvu + d_unv - removed_v);
                let eligible = old_distance + (delta_distance as i64) <= max_distance_acceptable;
                let tw = Sequence::tw5_with_travel(
                    &route.node(pos2 - 1).seq0_i,
                    &route.node(pos1).seq1,
                    &acc_mid,
                    &route.node(pos2).seq1,
                    &route.node(pos1 + 1).seqi_n,
                    d_pvu,
                    d_unv,
                    d_puv,
                    d_vnu,
                );
                let new_cost =
                    old_distance + (delta_distance as i64) + cap_pen + (tw as i64) * ptw;
                let better =
                    eligible & (new_cost <= max_acceptable_cost) & (new_cost < best_cost);
                best_cost = if better { new_cost } else { best_cost };
                best_pos = if better { Some(pos2) } else { best_pos };
                acc_mid = Sequence::join_tw_with_travel(
                    &route.node(pos2).seq1,
                    &acc_mid,
                    route.node(pos2).dist_to_succ,
                );
            }
        }

        if let Some(target) = best_pos {
            let selected_delta = best_cost - old_cost;
            self.last_plan = MovePlan::IntraSwap { target };
            selected_delta
        } else {
            NO_MOVE
        }
    }

    fn run_2optstar(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize) -> i64 {
        debug_assert!(r1 != r2);
        debug_assert!(pos1 > 0 && pos1 < self.routes[r1].nodes.len());
        debug_assert!(pos2 > 0 && pos2 < self.routes[r2].nodes.len());

        let route1 = route_at(&self.routes, r1);
        let route2 = route_at(&self.routes, r2);
        let old_cost = route1.cost + route2.cost;
        let max_acceptable_cost = old_cost + self.move_bar;

        let left1 = &route1.node(pos1 - 1).seq0_i;
        let right1 = &route2.node(pos2).seqi_n;
        let left2 = &route2.node(pos2 - 1).seq0_i;
        let right2 = &route1.node(pos1).seqi_n;

        let pcap = self.params.penalty_capa as i64;
        let max_cap = self.data.max_capacity;
        let travel1 = self
            .data
            .dm(left1.last_node as usize, right1.first_node as usize);
        let travel2 = self
            .data
            .dm(left2.last_node as usize, right2.first_node as usize);
        let dist1 = left1.distance + right1.distance + travel1;
        let dist2 = left2.distance + right2.distance + travel2;
        let load1 = left1.load + right1.load;
        let load2 = left2.load + right2.load;
        let lb_cost = (dist1 as i64)
            + (dist2 as i64)
            + ((load1 - max_cap).max(0) as i64) * pcap
            + ((load2 - max_cap).max(0) as i64) * pcap;
        if lb_cost > max_acceptable_cost {
            return NO_MOVE;
        }

        let tw = Sequence::tw2_with_travel(left1, right1, travel1)
            + Sequence::tw2_with_travel(left2, right2, travel2);
        let new_cost = lb_cost + (tw as i64) * self.params.penalty_tw as i64;
        if new_cost <= max_acceptable_cost {
            let selected_delta = new_cost - old_cost;
            self.last_plan = MovePlan::TwoOptStar;
            selected_delta
        } else {
            NO_MOVE
        }
    }

    fn run_2opt(&mut self, r1: usize, pos1: usize) -> i64 {
        let route = route_at(&self.routes, r1);
        let data = self.data.as_ref();
        let len = route.nodes.len();
        if len < pos1 + 3 {
            return NO_MOVE;
        }

        debug_assert!(pos1 > 0);
        debug_assert!(self.routes[r1].nodes[pos1].id != 0);

        let old_cost = route.cost;
        let max_acceptable_cost = old_cost + self.move_credit;
        let mut best_cost = i64::MAX;
        let mut best_pos: Option<usize> = None;
        let cap_pen =
            ((route.load - self.data.max_capacity).max(0) as i64) * self.params.penalty_capa as i64;
        let ptw = self.params.penalty_tw as i64;
        let max_distance_acceptable = max_acceptable_cost - cap_pen;
        let a_id = route.node(pos1 - 1).id;
        let u_id = route.node(pos1).id;
        let old_distance = route.distance as i64;
        let removed_au = route.node(pos1 - 1).dist_to_succ;
        let distance_from_a = data.distance_row(a_id);
        let distance_from_u = data.distance_row(u_id);

        let cheapest = Self::arc_scan::<0>(&route.nodes, pos1, pos1 + 1, distance_from_a, distance_from_u);
        if old_distance + ((cheapest as i64) - (removed_au as i64)) > max_distance_acceptable {
            return NO_MOVE;
        }

        let mut mid_rev = route.node(pos1).seq21;
        let mut mid_rev_distance = mid_rev.distance;
        for pos2 in (pos1 + 1)..(len - 1) {
            let v_id = route.node(pos2).id;
            let b_id = route.node(pos2 + 1).id;
            let d_av = copy_at(distance_from_a, v_id);
            let d_ub = copy_at(distance_from_u, b_id);
            let left = route.node(pos1 - 1).seq0_i;
            let right = route.node(pos2 + 1).seqi_n;
            let new_distance = left.distance + mid_rev_distance + right.distance + d_av + d_ub;
            let boundary_delta = d_av + d_ub - removed_au - route.node(pos2).dist_to_succ;
            let eligible = old_distance + boundary_delta as i64 <= max_distance_acceptable;
            let tw = Sequence::tw3_with_travel(&left, &mid_rev, &right, d_av, d_ub);
            let new_cost = (new_distance as i64) + cap_pen + (tw as i64) * ptw;
            let better = eligible & (new_cost <= max_acceptable_cost) & (new_cost < best_cost);
            best_cost = if better { new_cost } else { best_cost };
            best_pos = if better { Some(pos2) } else { best_pos };
            let next = route.node(pos2 + 1);
            let reversed_travel = data.dm(next.id, v_id);
            mid_rev_distance += reversed_travel;
            mid_rev = Sequence::join_tw_with_travel(&next.seq1, &mid_rev, reversed_travel);
        }

        if let Some(end) = best_pos {
            let selected_delta = best_cost - old_cost;
            self.last_plan = MovePlan::Intra2Opt { end };
            selected_delta
        } else {
            NO_MOVE
        }
    }

    fn run_inter_route(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize) -> i64 {
        {
            let data = self.data.as_ref();
            let ru = route_at(&self.routes, r1);
            let rv = route_at(&self.routes, r2);
            let u = ru.node(pos1);
            let v = rv.node(pos2);
            let u_pred = ru.node(pos1 - 1);
            let v_pred = rv.node(pos2 - 1);
            let x = ru.node(pos1 + 1);
            let distance_from_u_pred = data.distance_row(u_pred.id);
            let distance_from_v_pred = data.distance_row(v_pred.id);
            let distance_from_u = data.distance_row(u.id);
            let distance_from_v = data.distance_row(v.id);
            let distance_from_x = data.distance_row(x.id);
            debug_assert!(
                u.id != 0,
                "Should always apply inter-route with a client as first node"
            );
            debug_assert!(r1 != r2, "Should not test inter-route move on same route");

            let old_total = ru.cost + rv.cost;
            let max_acceptable_cost = old_total + self.move_bar;
            let pcap = self.params.penalty_capa as i64;
            let max_cap = data.max_capacity;
            let route_total_dist = ru.distance + rv.distance;
            let route1_load = ru.load;
            let route2_load = rv.load;

            macro_rules! cap_pen {
                ($load:expr) => {
                    ((($load) - max_cap).max(0) as i64) * pcap
                };
            }

            let send1_1_load = u.seq1.load;
            let rem1_1 = u_pred.dist_to_succ + u.dist_to_succ;
            let rem2_0 = v_pred.dist_to_succ;
            let d_upred_x = copy_at(distance_from_u_pred, x.id);
            let d_vpred_u = copy_at(distance_from_v_pred, u.id);
            let d_u_v = copy_at(distance_from_u, v.id);
            let cap_10 = cap_pen!(route1_load - send1_1_load) + cap_pen!(route2_load + send1_1_load);
            let lb10 =
                (route_total_dist - rem1_1 - rem2_0 + d_upred_x + d_vpred_u + d_u_v) as i64 + cap_10;

            let xd = (x.id == 0) as usize;
            let i_xn = pos1 + 2 - xd;
            let x_next = ru.node(i_xn);
            let xnd = (x_next.id == 0) as usize;
            let x2_next = ru.node(i_xn + 1 - xnd);
            let vd = (v.id == 0) as usize;
            let j_y = pos2 + 1 - vd;
            let y = rv.node(j_y);
            let yd = (y.id == 0) as usize;
            let j_yn = j_y + 1 - yd;
            let y_next = rv.node(j_yn);
            let ynd = (y_next.id == 0) as usize;
            let y2_next = rv.node(j_yn + 1 - ynd);

            let s3d = xnd | (!self.params.allow_swap3) as usize;
            let m_x = (xd as i64) << 58;
            let m_v = (vd as i64) << 58;
            let m_s = m_x | ((s3d as i64) << 58);
            let m_xv = m_x | m_v;
            let m_xvy = m_xv | ((yd as i64) << 58);
            let m_sv = m_s | m_v;
            let m_svy = m_sv | ((yd as i64) << 58);
            let m_svyn = m_svy | ((ynd as i64) << 58);

            let distance_from_y = data.distance_row(y.id);
            let distance_from_x_next = data.distance_row(x_next.id);

            let send1_2_load = u.seq1.load + x.seq1.load;
            let rem1_2 = rem1_1 + x.dist_to_succ;
            let send2_1_load = v.seq1.load;
            let rem2_1 = rem2_0 + v.dist_to_succ;
            let send2_2_load = send2_1_load + y.seq1.load;
            let rem2_2 = rem2_1 + y.dist_to_succ;

            let d_upred_xnext = copy_at(distance_from_u_pred, x_next.id);
            let d_upred_v = copy_at(distance_from_u_pred, v.id);
            let d_upred_y = copy_at(distance_from_u_pred, y.id);
            let d_vpred_x = copy_at(distance_from_v_pred, x.id);
            let d_x_u = copy_at(distance_from_x, u.id);
            let d_u_x = u.dist_to_succ;
            let d_x_v = copy_at(distance_from_x, v.id);
            let d_v_x = copy_at(distance_from_v, x.id);
            let d_u_y = copy_at(distance_from_u, y.id);
            let d_v_xnext = copy_at(distance_from_v, x_next.id);
            let d_x_y = copy_at(distance_from_x, y.id);
            let d_y_v = copy_at(distance_from_y, v.id);
            let d_y_xnext = copy_at(distance_from_y, x_next.id);
            let d_x_ynext = copy_at(distance_from_x, y_next.id);
            let d_u_ynext = copy_at(distance_from_u, y_next.id);

            let pfx_u = d_vpred_u + d_u_x;
            let pfx_x = d_vpred_x + d_x_u;
            let pfx_v = d_upred_v + v.dist_to_succ;
            let base_1_2 = route_total_dist - rem1_2;

            let mut lb_min = lb10;

            let cap_20_30 =
                cap_pen!(route1_load - send1_2_load) + cap_pen!(route2_load + send1_2_load);
            let head_20_30 = base_1_2 - rem2_0 + d_upred_xnext;
            let right_20 = pfx_u + d_x_v;
            let right_30 = pfx_x + d_u_v;
            lb_min = lb_min.min((head_20_30 + right_20.min(right_30)) as i64 + cap_20_30 + m_x);

            let cap_11 = cap_pen!(route1_load - send1_1_load + send2_1_load)
                + cap_pen!(route2_load - send2_1_load + send1_1_load);
            lb_min = lb_min.min(
                (route_total_dist - rem1_1 - rem2_1 + d_upred_v + d_v_x + d_vpred_u + d_u_y) as i64
                    + cap_11
                    + m_v,
            );

            let cap_21_31 = cap_pen!(route1_load - send1_2_load + send2_1_load)
                + cap_pen!(route2_load - send2_1_load + send1_2_load);
            let head_21_31 = base_1_2 - rem2_1 + d_upred_v + d_v_xnext;
            let right_21 = pfx_u + d_x_y;
            let right_31 = pfx_x + d_u_y;
            lb_min = lb_min.min((head_21_31 + right_21.min(right_31)) as i64 + cap_21_31 + m_xv);

            let cap_22_33 = cap_pen!(route1_load - send1_2_load + send2_2_load)
                + cap_pen!(route2_load - send2_2_load + send1_2_load);
            let left_fwd_dist = pfx_v + d_y_xnext;
            let left_rev_dist = d_upred_y + d_y_v + d_v_xnext;
            let right_fwd_dist = pfx_u + d_x_ynext;
            let right_rev_dist = pfx_x + d_u_ynext;
            lb_min = lb_min.min(
                (base_1_2 - rem2_2
                    + left_fwd_dist.min(left_rev_dist)
                    + right_fwd_dist.min(right_rev_dist)) as i64
                    + cap_22_33
                    + m_xvy,
            );

            if (x_next.id != 0) & self.params.allow_swap3 {
                let send1_3_load = send1_2_load + x_next.seq1.load;
                let rem1_3 = rem1_2 + x_next.dist_to_succ;
                let send2_3_load = send2_2_load + y_next.seq1.load;
                let rem2_3 = rem2_2 + y_next.dist_to_succ;
                let d_upred_x2next = copy_at(distance_from_u_pred, x2_next.id);
                let d_v_x2next = copy_at(distance_from_v, x2_next.id);
                let d_y_x2next = copy_at(distance_from_y, x2_next.id);
                let d_xnext_v = copy_at(distance_from_x_next, v.id);
                let d_xnext_y = copy_at(distance_from_x_next, y.id);
                let d_xnext_ynext = copy_at(distance_from_x_next, y_next.id);
                let d_xnext_y2next = copy_at(distance_from_x_next, y2_next.id);
                let d_ynext_x2next = copy_at(data.distance_row(y_next.id), x2_next.id);

                let head_4 = route_total_dist - rem1_3 + pfx_u + x.dist_to_succ;
                let cap_40 =
                    cap_pen!(route1_load - send1_3_load) + cap_pen!(route2_load + send1_3_load);
                lb_min =
                    lb_min.min((head_4 - rem2_0 + d_upred_x2next + d_xnext_v) as i64 + cap_40 + m_s);

                let cap_41 = cap_pen!(route1_load - send1_3_load + send2_1_load)
                    + cap_pen!(route2_load - send2_1_load + send1_3_load);
                lb_min = lb_min.min(
                    (head_4 - rem2_1 + d_upred_v + d_v_x2next + d_xnext_y) as i64 + cap_41 + m_sv,
                );

                let cap_42_43 = cap_pen!(route1_load - send1_3_load + send2_2_load)
                    + cap_pen!(route2_load - send2_2_load + send1_3_load);
                let head_42_43 = head_4 - rem2_2 + d_xnext_ynext;
                let left_42 = pfx_v + d_y_x2next;
                let left_43 = d_upred_y + d_y_v + d_v_x2next;
                lb_min = lb_min.min((head_42_43 + left_42.min(left_43)) as i64 + cap_42_43 + m_svy);

                let cap_44 = cap_pen!(route1_load - send1_3_load + send2_3_load)
                    + cap_pen!(route2_load - send2_3_load + send1_3_load);
                lb_min = lb_min.min(
                    (head_4 - rem2_3
                        + d_upred_v
                        + v.dist_to_succ
                        + y.dist_to_succ
                        + d_ynext_x2next
                        + d_xnext_y2next) as i64
                        + cap_44
                        + m_svyn,
                );
            }

            if lb_min > max_acceptable_cost {
                return NO_MOVE;
            }
        }
        self.run_inter_route_full(r1, pos1, r2, pos2)
    }

    #[inline(always)]
    fn run_inter_route_empty(&mut self, r1: usize, pos1: usize, r2: usize) -> i64 {
        let (lb_min, max_acceptable_cost) = {
            let data = self.data.as_ref();
            let ru = route_at(&self.routes, r1);
            let rv = route_at(&self.routes, r2);
            debug_assert!(rv.nodes.len() == 2);
            let u = ru.node(pos1);
            let u_pred = ru.node(pos1 - 1);
            let x = ru.node(pos1 + 1);
            let max_acceptable_cost = ru.cost + rv.cost + self.move_bar;
            let pcap = self.params.penalty_capa as i64;
            let max_cap = data.max_capacity;
            macro_rules! cap_pen {
                ($load:expr) => {
                    ((($load) - max_cap).max(0) as i64) * pcap
                };
            }
            let route_total_dist = ru.distance + rv.distance;
            let route1_load = ru.load;
            let route2_load = rv.load;
            let distance_from_u_pred = data.distance_row(u_pred.id);
            let from_depot = data.distance_row(0);
            let to_depot = data.distance_column(0);
            let rem2_0 = rv.node(0).dist_to_succ;

            let send1_1_load = u.seq1.load;
            let rem1_1 = u_pred.dist_to_succ + u.dist_to_succ;
            let d_upred_x = copy_at(distance_from_u_pred, x.id);
            let d_vpred_u = copy_at(from_depot, u.id);
            let d_u_v = copy_at(to_depot, u.id);
            let cap_10 = cap_pen!(route1_load - send1_1_load) + cap_pen!(route2_load + send1_1_load);
            let mut lb_min =
                (route_total_dist - rem1_1 - rem2_0 + d_upred_x + d_vpred_u + d_u_v) as i64 + cap_10;

            if x.id != 0 {
                let x_next = ru.node(pos1 + 2);
                let send1_2_load = send1_1_load + x.seq1.load;
                let rem1_2 = rem1_1 + x.dist_to_succ;
                let d_upred_xnext = copy_at(distance_from_u_pred, x_next.id);
                let d_vpred_x = copy_at(from_depot, x.id);
                let d_x_u = data.dm(x.id, u.id);
                let d_u_x = u.dist_to_succ;
                let d_x_v = copy_at(to_depot, x.id);
                let pfx_u = d_vpred_u + d_u_x;
                let pfx_x = d_vpred_x + d_x_u;
                let cap_20_30 =
                    cap_pen!(route1_load - send1_2_load) + cap_pen!(route2_load + send1_2_load);
                let head_20_30 = route_total_dist - rem1_2 - rem2_0 + d_upred_xnext;
                let right_20 = pfx_u + d_x_v;
                let right_30 = pfx_x + d_u_v;
                lb_min = lb_min.min((head_20_30 + right_20.min(right_30)) as i64 + cap_20_30);

                if (x_next.id != 0) & self.params.allow_swap3 {
                    let x2_next = ru.node(pos1 + 3);
                    let send1_3_load = send1_2_load + x_next.seq1.load;
                    let rem1_3 = rem1_2 + x_next.dist_to_succ;
                    let d_upred_x2next = copy_at(distance_from_u_pred, x2_next.id);
                    let d_xnext_v = copy_at(to_depot, x_next.id);
                    let head_4 = route_total_dist - rem1_3 + pfx_u + x.dist_to_succ;
                    let cap_40 =
                        cap_pen!(route1_load - send1_3_load) + cap_pen!(route2_load + send1_3_load);
                    lb_min = lb_min.min((head_4 - rem2_0 + d_upred_x2next + d_xnext_v) as i64 + cap_40);
                }
            }
            (lb_min, max_acceptable_cost)
        };
        if lb_min > max_acceptable_cost {
            return NO_MOVE;
        }
        self.run_inter_route_full(r1, pos1, r2, 1)
    }
    #[inline(never)]
    fn run_inter_route_full(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize) -> i64 {
        let data = self.data.as_ref();
        let ru = route_at(&self.routes, r1);
        let rv = route_at(&self.routes, r2);
        let u = ru.node(pos1);
        let v = rv.node(pos2);
        let u_pred = ru.node(pos1 - 1);
        let v_pred = rv.node(pos2 - 1);
        let x = ru.node(pos1 + 1);
        let distance_from_u_pred = data.distance_row(u_pred.id);
        let distance_from_v_pred = data.distance_row(v_pred.id);
        let distance_from_u = data.distance_row(u.id);
        let distance_from_v = data.distance_row(v.id);
        let distance_from_x = data.distance_row(x.id);
        debug_assert!(
            u.id != 0,
            "Should always apply inter-route with a client as first node"
        );
        debug_assert!(r1 != r2, "Should not test inter-route move on same route");

        let old_total = ru.cost + rv.cost;
        let max_acceptable_cost = old_total + self.move_bar;
        let pcap = self.params.penalty_capa as i64;
        let ptw = self.params.penalty_tw as i64;
        let max_cap = data.max_capacity;
        let route_total_dist = ru.distance + rv.distance;
        let route1_load = ru.load;
        let route2_load = rv.load;
        let mut best_cost = i64::MAX;
        let mut best_send1 = 0u8;
        let mut best_send2 = 0u8;

        macro_rules! cap_pen {
            ($load:expr) => {
                ((($load) - max_cap).max(0) as i64) * pcap
            };
        }
        macro_rules! consider {
            ($send1:expr, $send2:expr, $lb:expr, $tw:expr) => {{
                let lower_bound = $lb;
                let candidate = lower_bound + ($tw as i64) * ptw;
                let better = (lower_bound <= max_acceptable_cost)
                    & (candidate <= max_acceptable_cost)
                    & (candidate < best_cost);
                best_cost = if better { candidate } else { best_cost };
                best_send1 = if better { $send1 } else { best_send1 };
                best_send2 = if better { $send2 } else { best_send2 };
            }};
        }

        let send1_1_load = u.seq1.load;
        let rem1_1 = u_pred.dist_to_succ + u.dist_to_succ;
        let rem2_0 = v_pred.dist_to_succ;
        let d_upred_x = copy_at(distance_from_u_pred, x.id);
        let d_vpred_u = copy_at(distance_from_v_pred, u.id);
        let d_u_v = copy_at(distance_from_u, v.id);
        let cap_10 = cap_pen!(route1_load - send1_1_load) + cap_pen!(route2_load + send1_1_load);
        let lb10 =
            (route_total_dist - rem1_1 - rem2_0 + d_upred_x + d_vpred_u + d_u_v) as i64 + cap_10;

        {
            let tw10 = Sequence::tw2_with_travel(&u_pred.seq0_i, &x.seqi_n, d_upred_x)
                + Sequence::tw3_with_travel(&v_pred.seq0_i, &u.seq1, &v.seqi_n, d_vpred_u, d_u_v);
            consider!(1, 0, lb10, tw10);
        }

        if v.id != 0 {
            let y = rv.node(pos2 + 1);
            let send2_1_load = v.seq1.load;
            let rem2_1 = v_pred.dist_to_succ + v.dist_to_succ;
            let d_upred_v = copy_at(distance_from_u_pred, v.id);
            let d_v_x = copy_at(distance_from_v, x.id);
            let d_u_y = copy_at(distance_from_u, y.id);
            let cap_11 = cap_pen!(route1_load - send1_1_load + send2_1_load)
                + cap_pen!(route2_load - send2_1_load + send1_1_load);
            let lb11 = (route_total_dist - rem1_1 - rem2_1 + d_upred_v + d_v_x + d_vpred_u + d_u_y)
                as i64
                + cap_11;
            {
                let tw11 =
                    Sequence::tw3_with_travel(&u_pred.seq0_i, &v.seq1, &x.seqi_n, d_upred_v, d_v_x)
                        + Sequence::tw3_with_travel(
                            &v_pred.seq0_i,
                            &u.seq1,
                            &y.seqi_n,
                            d_vpred_u,
                            d_u_y,
                        );
                consider!(1, 1, lb11, tw11);
            }
        }

        if x.id != 0 {
            let x_next = ru.node(pos1 + 2);
            let distance_from_x_next = data.distance_row(x_next.id);
            let send1_2_load = u.seq1.load + x.seq1.load;
            let rem1_2 = u_pred.dist_to_succ + u.dist_to_succ + x.dist_to_succ;
            let d_upred_xnext = copy_at(distance_from_u_pred, x_next.id);
            let d_vpred_x = copy_at(distance_from_v_pred, x.id);
            let d_x_u = copy_at(distance_from_x, u.id);
            let d_u_x = u.dist_to_succ;
            let d_x_v = copy_at(distance_from_x, v.id);
            let cap_20_30 =
                cap_pen!(route1_load - send1_2_load) + cap_pen!(route2_load + send1_2_load);
            let dist_base_20_30 = route_total_dist - rem1_2 - rem2_0;
            let lb20 =
                (dist_base_20_30 + d_upred_xnext + d_vpred_u + d_u_x + d_x_v) as i64 + cap_20_30;
            let lb30 =
                (dist_base_20_30 + d_upred_xnext + d_vpred_x + d_x_u + d_u_v) as i64 + cap_20_30;
            {
                let route1_tw =
                    Sequence::tw2_with_travel(&u_pred.seq0_i, &x_next.seqi_n, d_upred_xnext);
                let tw20 = route1_tw
                    + Sequence::tw3_with_travel(
                        &v_pred.seq0_i,
                        &u.seq12,
                        &v.seqi_n,
                        d_vpred_u,
                        d_x_v,
                    );
                let tw30 = route1_tw
                    + Sequence::tw3_with_travel(
                        &v_pred.seq0_i,
                        &u.seq21,
                        &v.seqi_n,
                        d_vpred_x,
                        d_u_v,
                    );
                consider!(2, 0, lb20, tw20);
                consider!(3, 0, lb30, tw30);
            }

            if v.id != 0 {
                let y = rv.node(pos2 + 1);
                let send2_1_load = v.seq1.load;
                let rem2_1 = v_pred.dist_to_succ + v.dist_to_succ;
                let d_upred_v = copy_at(distance_from_u_pred, v.id);
                let d_v_xnext = copy_at(distance_from_v, x_next.id);
                let d_x_y = copy_at(distance_from_x, y.id);
                let d_u_y = copy_at(distance_from_u, y.id);
                let cap_21_31 = cap_pen!(route1_load - send1_2_load + send2_1_load)
                    + cap_pen!(route2_load - send2_1_load + send1_2_load);
                let dist_base_21_31 = route_total_dist - rem1_2 - rem2_1;
                let common_left = dist_base_21_31 + d_upred_v + d_v_xnext;
                let lb21 = (common_left + d_vpred_u + d_u_x + d_x_y) as i64 + cap_21_31;
                let lb31 = (common_left + d_vpred_x + d_x_u + d_u_y) as i64 + cap_21_31;
                {
                    let route1_tw = Sequence::tw3_with_travel(
                        &u_pred.seq0_i,
                        &v.seq1,
                        &x_next.seqi_n,
                        d_upred_v,
                        d_v_xnext,
                    );
                    let tw21 = route1_tw
                        + Sequence::tw3_with_travel(
                            &v_pred.seq0_i,
                            &u.seq12,
                            &y.seqi_n,
                            d_vpred_u,
                            d_x_y,
                        );
                    let tw31 = route1_tw
                        + Sequence::tw3_with_travel(
                            &v_pred.seq0_i,
                            &u.seq21,
                            &y.seqi_n,
                            d_vpred_x,
                            d_u_y,
                        );
                    consider!(2, 1, lb21, tw21);
                    consider!(3, 1, lb31, tw31);
                }

                if y.id != 0 {
                    let y_next = rv.node(pos2 + 2);
                    let distance_from_y = data.distance_row(y.id);
                    let send2_2_load = v.seq1.load + y.seq1.load;
                    let rem2_2 = v_pred.dist_to_succ + v.dist_to_succ + y.dist_to_succ;
                    let d_upred_y = copy_at(distance_from_u_pred, y.id);
                    let d_y_v = copy_at(distance_from_y, v.id);
                    let d_y_xnext = copy_at(distance_from_y, x_next.id);
                    let d_x_ynext = copy_at(distance_from_x, y_next.id);
                    let d_u_ynext = copy_at(distance_from_u, y_next.id);
                    let cap_22_33 = cap_pen!(route1_load - send1_2_load + send2_2_load)
                        + cap_pen!(route2_load - send2_2_load + send1_2_load);
                    let dist_base = route_total_dist - rem1_2 - rem2_2;
                    let left_fwd_dist = d_upred_v + v.dist_to_succ + d_y_xnext;
                    let left_rev_dist = d_upred_y + d_y_v + d_v_xnext;
                    let right_fwd_dist = d_vpred_u + d_u_x + d_x_ynext;
                    let right_rev_dist = d_vpred_x + d_x_u + d_u_ynext;
                    let lb22 = (dist_base + left_fwd_dist + right_fwd_dist) as i64 + cap_22_33;
                    let lb32 = (dist_base + left_fwd_dist + right_rev_dist) as i64 + cap_22_33;
                    let lb23 = (dist_base + left_rev_dist + right_fwd_dist) as i64 + cap_22_33;
                    let lb33 = (dist_base + left_rev_dist + right_rev_dist) as i64 + cap_22_33;
                    let can22 = lb22 <= max_acceptable_cost;
                    let can32 = lb32 <= max_acceptable_cost;
                    let can23 = lb23 <= max_acceptable_cost;
                    let can33 = lb33 <= max_acceptable_cost;

                    if can22 | can32 | can23 | can33 {
                        let left_fwd = Sequence::tw3_with_travel(
                            &u_pred.seq0_i,
                            &v.seq12,
                            &x_next.seqi_n,
                            d_upred_v,
                            d_y_xnext,
                        );
                        let left_rev = Sequence::tw3_with_travel(
                            &u_pred.seq0_i,
                            &v.seq21,
                            &x_next.seqi_n,
                            d_upred_y,
                            d_v_xnext,
                        );
                        let right_fwd = Sequence::tw3_with_travel(
                            &v_pred.seq0_i,
                            &u.seq12,
                            &y_next.seqi_n,
                            d_vpred_u,
                            d_x_ynext,
                        );
                        let right_rev = Sequence::tw3_with_travel(
                            &v_pred.seq0_i,
                            &u.seq21,
                            &y_next.seqi_n,
                            d_vpred_x,
                            d_u_ynext,
                        );

                        consider!(2, 2, lb22, left_fwd + right_fwd);
                        consider!(3, 2, lb32, left_fwd + right_rev);
                        consider!(2, 3, lb23, left_rev + right_fwd);
                        consider!(3, 3, lb33, left_rev + right_rev);
                    }
                }
            }

            if (x_next.id != 0) & self.params.allow_swap3 {
                let x2_next = ru.node(pos1 + 3);
                let send1_3_load = u.seq1.load + x.seq1.load + x_next.seq1.load;
                let rem1_3 =
                    u_pred.dist_to_succ + u.dist_to_succ + x.dist_to_succ + x_next.dist_to_succ;
                let d_upred_x2next = copy_at(distance_from_u_pred, x2_next.id);
                let d_x_xnext = x.dist_to_succ;
                let d_xnext_v = copy_at(distance_from_x_next, v.id);
                let cap_40 =
                    cap_pen!(route1_load - send1_3_load) + cap_pen!(route2_load + send1_3_load);
                let lb40 = (route_total_dist - rem1_3 - rem2_0
                    + d_upred_x2next
                    + d_vpred_u
                    + d_u_x
                    + d_x_xnext
                    + d_xnext_v) as i64
                    + cap_40;
                {
                    let tw40 =
                        Sequence::tw2_with_travel(&u_pred.seq0_i, &x2_next.seqi_n, d_upred_x2next)
                            + Sequence::tw3_with_travel(
                                &v_pred.seq0_i,
                                &u.seq123,
                                &v.seqi_n,
                                d_vpred_u,
                                d_xnext_v,
                            );
                    consider!(4, 0, lb40, tw40);
                }

                if v.id != 0 {
                    let y = rv.node(pos2 + 1);
                    let send2_1_load = v.seq1.load;
                    let rem2_1 = v_pred.dist_to_succ + v.dist_to_succ;
                    let d_upred_v = copy_at(distance_from_u_pred, v.id);
                    let d_v_x2next = copy_at(distance_from_v, x2_next.id);
                    let d_xnext_y = copy_at(distance_from_x_next, y.id);
                    let cap_41 = cap_pen!(route1_load - send1_3_load + send2_1_load)
                        + cap_pen!(route2_load - send2_1_load + send1_3_load);
                    let lb41 = (route_total_dist - rem1_3 - rem2_1
                        + d_upred_v
                        + d_v_x2next
                        + d_vpred_u
                        + d_u_x
                        + d_x_xnext
                        + d_xnext_y) as i64
                        + cap_41;
                    {
                        let tw41 = Sequence::tw3_with_travel(
                            &u_pred.seq0_i,
                            &v.seq1,
                            &x2_next.seqi_n,
                            d_upred_v,
                            d_v_x2next,
                        ) + Sequence::tw3_with_travel(
                            &v_pred.seq0_i,
                            &u.seq123,
                            &y.seqi_n,
                            d_vpred_u,
                            d_xnext_y,
                        );
                        consider!(4, 1, lb41, tw41);
                    }

                    if y.id != 0 {
                        let y_next = rv.node(pos2 + 2);
                        let distance_from_y = data.distance_row(y.id);
                        let send2_2_load = v.seq1.load + y.seq1.load;
                        let rem2_2 = v_pred.dist_to_succ + v.dist_to_succ + y.dist_to_succ;
                        let d_upred_y = copy_at(distance_from_u_pred, y.id);
                        let d_y_v = copy_at(distance_from_y, v.id);
                        let d_y_x2next = copy_at(distance_from_y, x2_next.id);
                        let d_xnext_ynext = copy_at(distance_from_x_next, y_next.id);
                        let cap_42_43 = cap_pen!(route1_load - send1_3_load + send2_2_load)
                            + cap_pen!(route2_load - send2_2_load + send1_3_load);
                        let dist_base = route_total_dist - rem1_3 - rem2_2;
                        let common_right = d_vpred_u + d_u_x + d_x_xnext + d_xnext_ynext;
                        let lb42 =
                            (dist_base + d_upred_v + v.dist_to_succ + d_y_x2next + common_right)
                                as i64
                                + cap_42_43;
                        let lb43 = (dist_base + d_upred_y + d_y_v + d_v_x2next + common_right)
                            as i64
                            + cap_42_43;
                        {
                            let right_tw = Sequence::tw3_with_travel(
                                &v_pred.seq0_i,
                                &u.seq123,
                                &y_next.seqi_n,
                                d_vpred_u,
                                d_xnext_ynext,
                            );
                            let tw42 = Sequence::tw3_with_travel(
                                &u_pred.seq0_i,
                                &v.seq12,
                                &x2_next.seqi_n,
                                d_upred_v,
                                d_y_x2next,
                            ) + right_tw;
                            let tw43 = Sequence::tw3_with_travel(
                                &u_pred.seq0_i,
                                &v.seq21,
                                &x2_next.seqi_n,
                                d_upred_y,
                                d_v_x2next,
                            ) + right_tw;
                            consider!(4, 2, lb42, tw42);
                            consider!(4, 3, lb43, tw43);
                        }

                        if y_next.id != 0 {
                            let y2_next = rv.node(pos2 + 3);
                            let send2_3_load = v.seq1.load + y.seq1.load + y_next.seq1.load;
                            let rem2_3 = v_pred.dist_to_succ
                                + v.dist_to_succ
                                + y.dist_to_succ
                                + y_next.dist_to_succ;
                            let d_ynext_x2next = copy_at(data.distance_row(y_next.id), x2_next.id);
                            let d_xnext_y2next = copy_at(distance_from_x_next, y2_next.id);
                            let cap_44 = cap_pen!(route1_load - send1_3_load + send2_3_load)
                                + cap_pen!(route2_load - send2_3_load + send1_3_load);
                            let lb44 = (route_total_dist - rem1_3 - rem2_3
                                + d_upred_v
                                + v.dist_to_succ
                                + y.dist_to_succ
                                + d_ynext_x2next
                                + d_vpred_u
                                + d_u_x
                                + d_x_xnext
                                + d_xnext_y2next) as i64
                                + cap_44;
                            {
                                let tw44 = Sequence::tw3_with_travel(
                                    &u_pred.seq0_i,
                                    &v.seq123,
                                    &x2_next.seqi_n,
                                    d_upred_v,
                                    d_ynext_x2next,
                                ) + Sequence::tw3_with_travel(
                                    &v_pred.seq0_i,
                                    &u.seq123,
                                    &y2_next.seqi_n,
                                    d_vpred_u,
                                    d_xnext_y2next,
                                );
                                consider!(4, 4, lb44, tw44);
                            }
                        }
                    }
                }
            }
        }

        if best_send1 == 0 && best_send2 == 0 {
            return NO_MOVE;
        }
        self.last_plan = MovePlan::InterRoute {
            send1: best_send1,
            send2: best_send2,
        };
        best_cost - old_total
    }

    #[inline(always)]
    fn arc_min_for(
        routes: &[Route],
        route_version: &[u32],
        arc_min: &mut [ArcMin],
        nb_nodes: usize,
        rid: usize,
        client: usize,
        to_c: &[i32],
        from_c: &[i32],
    ) -> i32 {
        let version = copy_at(route_version, rid);
        let key = rid * nb_nodes + client;
        debug_assert!(key < arc_min.len());
        let entry = unsafe { arc_min.get_unchecked_mut(key) };
        if entry.version == version {
            return entry.value;
        }
        let nodes = &route_at(routes, rid).nodes;
        let arcs = nodes.len() - 1;
        let last = arcs - 1;
        let mut best = i32::MAX;
        macro_rules! arc {
            ($s:expr) => {{
                let t = min($s, last);
                let pred = node_at(nodes, t);
                let next = node_at(nodes, t + 1);
                best = best.min(copy_at(to_c, pred.id) + copy_at(from_c, next.id) - pred.dist_to_succ);
            }};
        }
        let mut i = 0usize;
        while i < arcs {
            arc!(i);
            arc!(i + 1);
            arc!(i + 2);
            arc!(i + 3);
            arc!(i + 4);
            arc!(i + 5);
            arc!(i + 6);
            arc!(i + 7);
            arc!(i + 8);
            arc!(i + 9);
            arc!(i + 10);
            arc!(i + 11);
            i += 12;
        }
        *entry = ArcMin {
            version,
            value: best,
        };
        best
    }

    #[inline(always)]
    fn prepare_removal_slots(routes: &[Route], rid: usize, pos: usize, bridge: i32, slots: &mut RemovalSlots) {
        let nodes = &route_at(routes, rid).nodes;
        let len = nodes.len();
        let count = len - 2;
        slots.entries.clear();
        slots.entries.reserve(count);
        let out = slots.entries.as_mut_ptr();
        let mut k = 0usize;
        let mut right = node_at(nodes, pos + 1).seqi_n;
        for t in (1..pos + 1).rev() {
            let prev = node_at(nodes, t - 1);
            unsafe {
                out.add(k).write(Slot::from(&prev.seq0_i, &right, t));
            }
            k += 1;
            let bridge_mask = -((t == pos) as i32);
            let travel = (bridge & bridge_mask) | (prev.dist_to_succ & !bridge_mask);
            right = Sequence::join2_with_travel(&prev.seq1, &right, travel);
        }
        let mut left = Sequence::join2_with_travel(
            &node_at(nodes, pos - 1).seq0_i,
            &node_at(nodes, pos + 1).seq1,
            bridge,
        );
        for t in (pos + 2)..len {
            let node = node_at(nodes, t);
            unsafe {
                out.add(k).write(Slot::from(&left, &node.seqi_n, t));
            }
            k += 1;
            left = Sequence::join2_with_travel(&left, &node.seq1, node_at(nodes, t - 1).dist_to_succ);
        }
        debug_assert_eq!(k, count);
        unsafe {
            slots.entries.set_len(count);
        }
    }

    #[inline(never)]
    fn surviving_arc_min(nodes: &[Node], pos: usize, bridge: i32, to_c: &[i32], from_c: &[i32]) -> i32 {
        let pred = node_at(nodes, pos - 1);
        let next = node_at(nodes, pos + 1);
        let mut best = copy_at(to_c, pred.id) + copy_at(from_c, next.id) - bridge;
        for t in 1..pos {
            let a = node_at(nodes, t - 1);
            let b = node_at(nodes, t);
            best = best.min(copy_at(to_c, a.id) + copy_at(from_c, b.id) - a.dist_to_succ);
        }
        for t in (pos + 2)..nodes.len() {
            let a = node_at(nodes, t - 1);
            let b = node_at(nodes, t);
            best = best.min(copy_at(to_c, a.id) + copy_at(from_c, b.id) - a.dist_to_succ);
        }
        best
    }

    #[inline(always)]
    fn run_swapstar(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize) -> i64 {
        debug_assert!(r1 != r2);
        debug_assert!(pos1 > 0 && pos1 < self.routes[r1].nodes.len() - 1);
        debug_assert!(pos2 > 0 && pos2 < self.routes[r2].nodes.len() - 1);

        let (
            u,
            v,
            removed_distance1,
            removed_distance2,
            new_load1,
            new_load2,
            old_total,
            bridge_u,
            bridge_v,
        ) = {
            let route1 = route_at(&self.routes, r1);
            let route2 = route_at(&self.routes, r2);
            let node_u = route1.node(pos1);
            let node_v = route2.node(pos2);
            let pred_u = route1.node(pos1 - 1);
            let next_u = route1.node(pos1 + 1);
            let pred_v = route2.node(pos2 - 1);
            let next_v = route2.node(pos2 + 1);
            let bridge_u = self.data.dm(pred_u.id, next_u.id);
            let bridge_v = self.data.dm(pred_v.id, next_v.id);
            (
                node_u.id,
                node_v.id,
                route1.distance - pred_u.dist_to_succ - node_u.dist_to_succ + bridge_u,
                route2.distance - pred_v.dist_to_succ - node_v.dist_to_succ + bridge_v,
                route1.load - node_u.seq1.load + node_v.seq1.load,
                route2.load - node_v.seq1.load + node_u.seq1.load,
                route1.cost + route2.cost,
                bridge_u,
                bridge_v,
            )
        };

        let new_pen1 =
            ((new_load1 - self.data.max_capacity).max(0) as i64) * self.params.penalty_capa as i64;
        let new_pen2 =
            ((new_load2 - self.data.max_capacity).max(0) as i64) * self.params.penalty_capa as i64;
        let cost_lb_r1_after_removal = (removed_distance1 as i64) + new_pen1;
        let cost_lb_r2_after_removal = (removed_distance2 as i64) + new_pen2;
        let mut lb_new_total = cost_lb_r1_after_removal + cost_lb_r2_after_removal;
        let max_acceptable_cost = old_total + self.move_credit;

        if lb_new_total > max_acceptable_cost {
            return NO_MOVE;
        }

        let data = self.data.as_ref();
        let nb_nodes = data.nb_nodes;
        let distance_from_u = data.distance_row(u);
        let distance_to_u = data.distance_column(u);
        let distance_from_v = data.distance_row(v);
        let distance_to_v = data.distance_column(v);

        let (bridge_ins_v, bridge_ins_u) = {
            let route1_nodes = &route_at(&self.routes, r1).nodes;
            let route2_nodes = &route_at(&self.routes, r2).nodes;
            let pred_u = node_at(route1_nodes, pos1 - 1);
            let next_u = node_at(route1_nodes, pos1 + 1);
            let pred_v = node_at(route2_nodes, pos2 - 1);
            let next_v = node_at(route2_nodes, pos2 + 1);
            (
                copy_at(distance_to_v, pred_u.id) + copy_at(distance_from_v, next_u.id) - bridge_u,
                copy_at(distance_to_u, pred_v.id) + copy_at(distance_from_u, next_v.id) - bridge_v,
            )
        };
        let best_ins_v = bridge_ins_v.min(Self::arc_min_for(
            &self.routes,
            &self.route_version,
            &mut self.arc_min,
            nb_nodes,
            r1,
            v,
            distance_to_v,
            distance_from_v,
        ));

        lb_new_total += best_ins_v as i64;
        if lb_new_total > max_acceptable_cost {
            return NO_MOVE;
        }

        let max_acceptable_cost = old_total + self.move_bar;
        let best_ins_u = bridge_ins_u.min(Self::arc_min_for(
            &self.routes,
            &self.route_version,
            &mut self.arc_min,
            nb_nodes,
            r2,
            u,
            distance_to_u,
            distance_from_u,
        ));

        lb_new_total += best_ins_u as i64;
        if lb_new_total > max_acceptable_cost {
            return NO_MOVE;
        }

        let lb_cost2 = cost_lb_r2_after_removal + best_ins_u as i64;

        let ptw = self.params.penalty_tw as i64;

        let v_seq1 = route_at(&self.routes, r2).node(pos2).seq1;
        let mut best_t1: usize = pos1;
        let mut best_cost1 = i64::MAX / 4;
        {
            let version1 = copy_at(&self.route_version, r1);
            let slots = unsafe { self.removal_slots.get_unchecked_mut(u) };
            if slots.version != version1 {
                Self::prepare_removal_slots(&self.routes, r1, pos1, bridge_u, slots);
                slots.version = version1;
            }
            let entries = slots.entries.as_slice();
            let last = entries.len() - 1;
            macro_rules! slot1 {
                ($s:expr) => {{
                    let slot = copy_at(entries, min($s, last));
                    let d_av = copy_at(distance_to_v, slot.pred as usize);
                    let d_vb = copy_at(distance_from_v, slot.next as usize);
                    let distance = slot.distance + d_av + d_vb;
                    let route1_lb = (distance as i64) + new_pen1;
                    let ok = (route1_lb < best_cost1) & (route1_lb + lb_cost2 <= max_acceptable_cost);
                    let arrival = slot.prefix_end + d_av;
                    let extra = (arrival - v_seq1.tau_plus).max(0);
                    let end = v_seq1.earliest_end.max(arrival + v_seq1.duration_net);
                    let tw = slot.warp + extra.max(end + d_vb - slot.suffix_latest);
                    let cand = route1_lb + (tw as i64) * ptw;
                    let better = ok & (cand < best_cost1);
                    best_cost1 = if better { cand } else { best_cost1 };
                    best_t1 = if better { slot.position as usize } else { best_t1 };
                }};
            }
            let mut i = 0usize;
            while i < entries.len() {
                slot1!(i);
                slot1!(i + 1);
                slot1!(i + 2);
                slot1!(i + 3);
                i += 4;
            }
        }

        if best_cost1.saturating_add(lb_cost2) > max_acceptable_cost {
            return NO_MOVE;
        }

        let u_seq1 = route_at(&self.routes, r1).node(pos1).seq1;
        let mut best_t2: usize = pos2;
        let mut best_cost2 = i64::MAX / 4;
        {
            let version2 = copy_at(&self.route_version, r2);
            let slots = unsafe { self.removal_slots.get_unchecked_mut(v) };
            if slots.version != version2 {
                Self::prepare_removal_slots(&self.routes, r2, pos2, bridge_v, slots);
                slots.version = version2;
            }
            let entries = slots.entries.as_slice();
            let last = entries.len() - 1;
            macro_rules! slot2 {
                ($s:expr) => {{
                    let slot = copy_at(entries, min($s, last));
                    let d_au = copy_at(distance_to_u, slot.pred as usize);
                    let d_ub = copy_at(distance_from_u, slot.next as usize);
                    let distance = slot.distance + d_au + d_ub;
                    let route2_lb = (distance as i64) + new_pen2;
                    let ok = (route2_lb < best_cost2) & (best_cost1 + route2_lb <= max_acceptable_cost);
                    let arrival = slot.prefix_end + d_au;
                    let extra = (arrival - u_seq1.tau_plus).max(0);
                    let end = u_seq1.earliest_end.max(arrival + u_seq1.duration_net);
                    let tw = slot.warp + extra.max(end + d_ub - slot.suffix_latest);
                    let cand = route2_lb + (tw as i64) * ptw;
                    let better = ok & (cand < best_cost2);
                    best_cost2 = if better { cand } else { best_cost2 };
                    best_t2 = if better { slot.position as usize } else { best_t2 };
                }};
            }
            let mut i = 0usize;
            while i < entries.len() {
                slot2!(i);
                slot2!(i + 1);
                slot2!(i + 2);
                slot2!(i + 3);
                i += 4;
            }
        }

        let new_total = best_cost1 + best_cost2;
        if new_total > max_acceptable_cost {
            return NO_MOVE;
        }

        if best_cost1 + cost_lb_r2_after_removal > old_total + self.move_credit {
            let exact_ins_v = Self::surviving_arc_min(
                &route_at(&self.routes, r1).nodes,
                pos1,
                bridge_u,
                distance_to_v,
                distance_from_v,
            );
            if cost_lb_r1_after_removal + cost_lb_r2_after_removal + (exact_ins_v as i64)
                > old_total + self.move_credit
            {
                return NO_MOVE;
            }
        }

        self.last_plan = MovePlan::SwapStar {
            insert1: best_t1,
            insert2: best_t2,
        };
        new_total - old_total
    }


    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn batch_swap8(&mut self, r1: usize, pos1: usize, ids: &[usize; 8], live: u32) -> u32 {
        use std::arch::x86_64::*;
        let data = self.data.as_ref();
        let n = data.nb_nodes;
        let route1 = route_at(&self.routes, r1);
        if route1.cost >= COST_CAP {
            return 255;
        }
        let node_u = route1.node(pos1);
        let pred_u = route1.node(pos1 - 1);
        let next_u = route1.node(pos1 + 1);
        let bridge_u = data.dm(pred_u.id, next_u.id);
        let removed_distance1 = route1.distance - pred_u.dist_to_succ - node_u.dist_to_succ + bridge_u;
        let u = node_u.id;
        macro_rules! b {
            ($x:expr) => {
                _mm256_set1_epi32($x as i32)
            };
        }
        macro_rules! add {
            ($x:expr, $y:expr) => {
                _mm256_add_epi32($x, $y)
            };
        }
        macro_rules! sub {
            ($x:expr, $y:expr) => {
                _mm256_sub_epi32($x, $y)
            };
        }
        let lane_ids = _mm256_setr_epi32(
            ids[0] as i32,
            ids[1] as i32,
            ids[2] as i32,
            ids[3] as i32,
            ids[4] as i32,
            ids[5] as i32,
            ids[6] as i32,
            ids[7] as i32,
        );
        let rec0 = self.load_record_part(ids.as_ptr(), 0);
        let rec3 = self.load_record_part(ids.as_ptr(), 3);
        let zero = _mm256_setzero_si256();
        let pcap = b!(self.params.penalty_capa);
        let max_cap = b!(data.max_capacity);
        macro_rules! cap {
            ($load:expr) => {
                _mm256_mullo_epi32(_mm256_max_epi32(sub!($load, max_cap), zero), pcap)
            };
        }
        macro_rules! bits {
            ($m:expr) => {
                (_mm256_movemask_ps(_mm256_castsi256_ps($m)) as u32) & 255
            };
        }
        let r2_load = rec0[CUT_R_LOAD];
        let r2_dist = rec0[CUT_R_DIST];
        let r2_cost = rec0[CUT_R_COST];
        let rm_pred = rec3[CUT_RM_PRED - 24];
        let rm_next = rec3[CUT_RM_NEXT - 24];
        let rm_arcs = rec3[CUT_RM_ARCS - 24];
        let rm_bridge = rec3[CUT_RM_BRIDGE - 24];
        let v_load = rec3[CUT_RM_LOAD - 24];
        let rid2 = rec3[CUT_RID - 24];

        let over = _mm256_cmpgt_epi32(r2_cost, b!(COST_CAP as i32 - 1));
        let max_acc = add!(add!(b!(route1.cost), r2_cost), b!(self.move_credit));

        let removed_distance2 = add!(sub!(r2_dist, rm_arcs), rm_bridge);
        let new_load1 = add!(b!(route1.load - node_u.seq1.load), v_load);
        let new_load2 = add!(sub!(r2_load, v_load), b!(node_u.seq1.load));
        let lb1 = add!(add!(b!(removed_distance1), cap!(new_load1)), add!(removed_distance2, cap!(new_load2)));
        let mut reject = _mm256_cmpgt_epi32(lb1, max_acc);

        let from_upred = data.distance_row(pred_u.id);
        let to_nextu = data.distance_column(next_u.id);
        let bridge_ins_v = sub!(
            add!(
                _mm256_i32gather_epi32::<4>(from_upred.as_ptr(), lane_ids),
                _mm256_i32gather_epi32::<4>(to_nextu.as_ptr(), lane_ids)
            ),
            b!(bridge_u)
        );
        let version1 = copy_at(&self.route_version, r1);
        let key1 = add!(b!(r1 * n), lane_ids);
        let am = self.arc_min.as_ptr() as *const i32;
        let ver1 = _mm256_i32gather_epi32::<8>(am, key1);
        let hit1 = _mm256_cmpeq_epi32(ver1, b!(version1));
        let mut miss = bits!(_mm256_andnot_si256(_mm256_or_si256(reject, hit1), _mm256_set1_epi32(-1))) & live;
        while miss != 0 {
            let j = miss.trailing_zeros() as usize;
            miss &= miss - 1;
            let v = ids[j];
            let best = Self::arc_scan::<0>(&route1.nodes, 0, 0, data.distance_column(v), data.distance_row(v));
            *self.arc_min.get_unchecked_mut(r1 * n + v) = ArcMin {
                version: version1,
                value: best,
            };
        }
        let am = self.arc_min.as_ptr() as *const i32;
        let val1 = _mm256_i32gather_epi32::<8>(am.add(1), key1);
        let lb2 = add!(lb1, _mm256_min_epi32(bridge_ins_v, val1));
        reject = _mm256_or_si256(reject, _mm256_cmpgt_epi32(lb2, max_acc));

        let to_u = data.distance_column(u);
        let from_u = data.distance_row(u);
        let bridge_ins_u = sub!(
            add!(
                _mm256_i32gather_epi32::<4>(to_u.as_ptr(), rm_pred),
                _mm256_i32gather_epi32::<4>(from_u.as_ptr(), rm_next)
            ),
            rm_bridge
        );
        let key2 = add!(_mm256_mullo_epi32(rid2, b!(n)), b!(u));
        let ver2 = _mm256_i32gather_epi32::<8>(am, key2);
        let rv2 = _mm256_i32gather_epi32::<4>(self.route_version.as_ptr() as *const i32, rid2);
        let hit2 = _mm256_cmpeq_epi32(ver2, rv2);
        let mut miss = bits!(_mm256_andnot_si256(_mm256_or_si256(reject, hit2), _mm256_set1_epi32(-1))) & live;
        let mut rid_lanes = [0i32; 8];
        _mm256_storeu_si256(rid_lanes.as_mut_ptr() as *mut __m256i, rid2);
        while miss != 0 {
            let j = miss.trailing_zeros() as usize;
            miss &= miss - 1;
            let rid = rid_lanes[j] as usize;
            let best = Self::arc_scan::<0>(&route_at(&self.routes, rid).nodes, 0, 0, to_u, from_u);
            *self.arc_min.get_unchecked_mut(rid * n + u) = ArcMin {
                version: copy_at(&self.route_version, rid),
                value: best,
            };
        }
        let am = self.arc_min.as_ptr() as *const i32;
        let val2 = _mm256_i32gather_epi32::<8>(am.add(1), key2);
        let lb3 = add!(lb2, _mm256_min_epi32(bridge_ins_u, val2));
        let max_acc3 = add!(add!(b!(route1.cost), r2_cost), b!(self.move_bar.max(-COST_CAP)));
        reject = _mm256_or_si256(reject, _mm256_cmpgt_epi32(lb3, max_acc3));

        let reject = _mm256_andnot_si256(over, reject);
        (!bits!(reject)) & 255
    }

    #[cfg(not(target_arch = "x86_64"))]
    unsafe fn batch_swap8(&mut self, _r1: usize, _pos1: usize, _ids: &[usize; 8], _live: u32) -> u32 {
        255
    }

    #[inline(always)]
    fn inter_src<const HEAD: bool>(&self, r1: usize, pos1: usize) -> InterSrc {
        let data = self.data.as_ref();
        let ru = route_at(&self.routes, r1);
        let u = ru.node(pos1);
        let u_pred = ru.node(pos1 - 1);
        let x = ru.node(pos1 + 1);
        let xd = (x.id == 0) as usize;
        let i_xn = pos1 + 2 - xd;
        let x_next = ru.node(i_xn);
        let xnd = (x_next.id == 0) as usize;
        let x2_next = ru.node(i_xn + 1 - xnd);
        let distance_from_u_pred = data.distance_row(u_pred.id);
        let distance_from_x = data.distance_row(x.id);
        let left1 = &u_pred.seq0_i;
        let right2 = &u.seqi_n;
        let zero = Node::default();
        let (t_v, t_y, t_yn, t_y2n, t_yd, t_ynd) = if HEAD {
            let t_v = ru.node(1);
            let t_y = ru.node(2.min(ru.nodes.len() - 1));
            let t_yd = (t_y.id == 0) as usize;
            let t_yn = ru.node(3 - t_yd);
            let t_ynd = (t_yn.id == 0) as usize;
            (t_v, t_y, t_yn, ru.node(4 - t_yd - t_ynd), t_yd, t_ynd)
        } else {
            (&zero, &zero, &zero, &zero, 0, 0)
        };
        InterSrc {
            r1_load: ru.load,
            r1_dist: ru.distance,
            r1_cost: ru.cost.min(COST_CAP) as i32,
            rem1_1: u_pred.dist_to_succ + u.dist_to_succ,
            rem1_2: u_pred.dist_to_succ + u.dist_to_succ + x.dist_to_succ,
            rem1_3: u_pred.dist_to_succ + u.dist_to_succ + x.dist_to_succ + x_next.dist_to_succ,
            send1_1: u.seq1.load,
            send1_2: u.seq1.load + x.seq1.load,
            send1_3: u.seq1.load + x.seq1.load + x_next.seq1.load,
            d_upred_x: copy_at(distance_from_u_pred, x.id),
            d_upred_xnext: copy_at(distance_from_u_pred, x_next.id),
            d_upred_x2next: copy_at(distance_from_u_pred, x2_next.id),
            d_x_u: copy_at(distance_from_x, u.id),
            d_u_x: u.dist_to_succ,
            d_x_xnext: x.dist_to_succ,
            xd: xd == 1,
            swap3: (x_next.id != 0) & self.params.allow_swap3,
            u: u.id,
            x: x.id,
            xnext: x_next.id,
            x2next: x2_next.id,
            upred: u_pred.id,
            left1_end: left1.earliest_end,
            left1_tw: left1.tw,
            left1_load: left1.load,
            left1_dist: left1.distance,
            right2_latest: right2.tau_plus,
            right2_tw: right2.tw,
            right2_load: right2.load,
            right2_dist: right2.distance,
            valid: ru.cost < COST_CAP && self.move_credit.abs() < COST_CAP,
            t_y: t_y.id,
            t_yn: t_yn.id,
            t_y2n: t_y2n.id,
            t_rem2_0: ru.node(0).dist_to_succ,
            t_v_dts: t_v.dist_to_succ,
            t_y_dts: t_y.dist_to_succ,
            t_yn_dts: t_yn.dist_to_succ,
            t_v_load: t_v.seq1.load,
            t_y_load: t_y.seq1.load,
            t_yn_load: t_yn.seq1.load,
            t_d_y_v: if HEAD { data.dm(t_y.id, t_v.id) } else { 0 },
            t_yd: t_yd == 1,
            t_ynd: t_ynd == 1,
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn transpose8(r: &mut [std::arch::x86_64::__m256i; 8]) {
        use std::arch::x86_64::*;
        let t0 = _mm256_unpacklo_epi32(r[0], r[1]);
        let t1 = _mm256_unpackhi_epi32(r[0], r[1]);
        let t2 = _mm256_unpacklo_epi32(r[2], r[3]);
        let t3 = _mm256_unpackhi_epi32(r[2], r[3]);
        let t4 = _mm256_unpacklo_epi32(r[4], r[5]);
        let t5 = _mm256_unpackhi_epi32(r[4], r[5]);
        let t6 = _mm256_unpacklo_epi32(r[6], r[7]);
        let t7 = _mm256_unpackhi_epi32(r[6], r[7]);
        let u0 = _mm256_unpacklo_epi64(t0, t2);
        let u1 = _mm256_unpackhi_epi64(t0, t2);
        let u2 = _mm256_unpacklo_epi64(t1, t3);
        let u3 = _mm256_unpackhi_epi64(t1, t3);
        let u4 = _mm256_unpacklo_epi64(t4, t6);
        let u5 = _mm256_unpackhi_epi64(t4, t6);
        let u6 = _mm256_unpacklo_epi64(t5, t7);
        let u7 = _mm256_unpackhi_epi64(t5, t7);
        r[0] = _mm256_permute2x128_si256::<0x20>(u0, u4);
        r[1] = _mm256_permute2x128_si256::<0x20>(u1, u5);
        r[2] = _mm256_permute2x128_si256::<0x20>(u2, u6);
        r[3] = _mm256_permute2x128_si256::<0x20>(u3, u7);
        r[4] = _mm256_permute2x128_si256::<0x31>(u0, u4);
        r[5] = _mm256_permute2x128_si256::<0x31>(u1, u5);
        r[6] = _mm256_permute2x128_si256::<0x31>(u2, u6);
        r[7] = _mm256_permute2x128_si256::<0x31>(u3, u7);
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn load_record_part(&self, ids: *const usize, part: usize) -> [std::arch::x86_64::__m256i; 8] {
        use std::arch::x86_64::*;
        let cut = self.cut.as_ptr();
        let mut rows: [__m256i; 8] = [_mm256_setzero_si256(); 8];
        for j in 0..8 {
            let base = cut.add(*ids.add(j) * CUT + part * 8);
            rows[j] = _mm256_loadu_si256(base as *const __m256i);
        }
        Self::transpose8(&mut rows);
        rows
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn load_records(&self, ids: *const usize, out: &mut [std::arch::x86_64::__m256i; 24]) {
        for part in 0..3 {
            let rows = self.load_record_part(ids, part);
            for k in 0..8 {
                out[part * 8 + k] = rows[k];
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn batch8_rev(&self, src: &InterSrc, ids: *const usize) -> u32 {
        use std::arch::x86_64::*;
        let data = self.data.as_ref();
        let n = data.nb_nodes;
        macro_rules! b {
            ($x:expr) => {
                _mm256_set1_epi32($x as i32)
            };
        }
        macro_rules! add {
            ($x:expr, $y:expr) => {
                _mm256_add_epi32($x, $y)
            };
        }
        macro_rules! sub {
            ($x:expr, $y:expr) => {
                _mm256_sub_epi32($x, $y)
            };
        }
        macro_rules! mn {
            ($x:expr, $y:expr) => {
                _mm256_min_epi32($x, $y)
            };
        }
        macro_rules! gather {
            ($slice:expr, $idx:expr) => {
                _mm256_i32gather_epi32::<4>($slice.as_ptr(), $idx)
            };
        }
        let lane_ids = _mm256_setr_epi32(
            *ids as i32,
            *ids.add(1) as i32,
            *ids.add(2) as i32,
            *ids.add(3) as i32,
            *ids.add(4) as i32,
            *ids.add(5) as i32,
            *ids.add(6) as i32,
            *ids.add(7) as i32,
        );
        let rec0 = self.load_record_part(ids, 0);
        let rec1 = self.load_record_part(ids, 1);
        let rec3 = self.load_record_part(ids, 3);
        let zero = _mm256_setzero_si256();
        let pcap = b!(self.params.penalty_capa);
        let max_cap = b!(data.max_capacity);
        macro_rules! cap {
            ($load:expr) => {
                _mm256_mullo_epi32(_mm256_max_epi32(sub!($load, max_cap), zero), pcap)
            };
        }
        let r1_load = rec0[CUT_R_LOAD];
        let r1_dist = rec0[CUT_R_DIST];
        let r1_cost = rec0[CUT_R_COST];
        let x = rec0[CUT_V];
        let xn = rec0[CUT_Y];
        let x2n = rec0[CUT_YN];
        let d_u_x = rec0[CUT_REM0];
        let x_dts = rec1[CUT_V_DTS - 8];
        let xn_dts = rec1[CUT_Y_DTS - 8];
        let x_load = rec1[CUT_V_LOAD - 8];
        let xn_load = rec1[CUT_Y_LOAD - 8];
        let flags = rec1[CUT_FLAGS - 8];
        let upred = rec3[CUT_RM_PRED - 24];
        let rem1_1 = rec3[CUT_RM_ARCS - 24];
        let send1_1 = rec3[CUT_RM_LOAD - 24];

        let over = _mm256_cmpgt_epi32(r1_cost, b!(COST_CAP as i32 - 1));
        let scalar = (_mm256_movemask_ps(_mm256_castsi256_ps(over)) as u32) & 255;
        let max_acc = add!(add!(r1_cost, b!(src.r1_cost)), b!(self.move_bar.max(-COST_CAP)));
        let route_total_dist = add!(r1_dist, b!(src.r1_dist));

        let m_x = _mm256_slli_epi32::<28>(_mm256_and_si256(flags, b!(1)));
        let m_s = _mm256_or_si256(m_x, _mm256_slli_epi32::<27>(_mm256_and_si256(flags, b!(2))));
        let m_y = b!(if src.t_yd { LANE_MASK } else { 0 });
        let m_yn = b!(if src.t_ynd { LANE_MASK } else { 0 });
        let m_xy = _mm256_or_si256(m_x, m_y);
        let m_sy = _mm256_or_si256(m_s, m_y);

        let matrix = data.distance_matrix.as_slice();
        let nv = b!(n);
        let idx_upred = _mm256_mullo_epi32(upred, nv);
        let d_upred_x = gather!(matrix, add!(idx_upred, x));
        let d_upred_xnext = gather!(matrix, add!(idx_upred, xn));
        let d_x_u = gather!(matrix, add!(_mm256_mullo_epi32(x, nv), lane_ids));

        let v = src.u;
        let row0 = data.distance_row(0);
        let to_v = data.distance_column(v);
        let to_y = data.distance_column(src.t_y);
        let to_yn = data.distance_column(src.t_yn);
        let from_v = data.distance_row(v);
        let from_y = data.distance_row(src.t_y);
        let d_vpred_u = gather!(row0, lane_ids);
        let d_vpred_x = gather!(row0, x);
        let d_upred_v = gather!(to_v, upred);
        let d_upred_y = gather!(to_y, upred);
        let d_u_v = gather!(to_v, lane_ids);
        let d_u_y = gather!(to_y, lane_ids);
        let d_u_ynext = gather!(to_yn, lane_ids);
        let d_x_v = gather!(to_v, x);
        let d_x_y = gather!(to_y, x);
        let d_x_ynext = gather!(to_yn, x);
        let d_v_x = gather!(from_v, x);
        let d_v_xnext = gather!(from_v, xn);
        let d_y_xnext = gather!(from_y, xn);

        let send2_1 = b!(src.t_v_load);
        let send2_2 = b!(src.t_v_load + src.t_y_load);
        let rem2_0 = b!(src.t_rem2_0);
        let rem2_1 = b!(src.t_rem2_0 + src.t_v_dts);
        let rem2_2 = b!(src.t_rem2_0 + src.t_v_dts + src.t_y_dts);
        let r2_load = b!(src.r1_load);
        let v_dts = b!(src.t_v_dts);
        let rem1_2 = add!(rem1_1, x_dts);
        let send1_2 = add!(send1_1, x_load);

        let r1_m1 = sub!(r1_load, send1_1);
        let r1_m2 = sub!(r1_load, send1_2);
        let cap_10 = add!(cap!(r1_m1), cap!(add!(r2_load, send1_1)));
        let lb10 = add!(
            add!(add!(sub!(sub!(route_total_dist, rem1_1), rem2_0), d_upred_x), add!(d_vpred_u, d_u_v)),
            cap_10
        );
        let mut lb_min = lb10;

        let pfx_u = add!(d_vpred_u, d_u_x);
        let pfx_x = add!(d_vpred_x, d_x_u);
        let pfx_v = add!(d_upred_v, v_dts);
        let base_1_2 = sub!(route_total_dist, rem1_2);

        let cap_20_30 = add!(cap!(r1_m2), cap!(add!(r2_load, send1_2)));
        let head_20_30 = add!(sub!(base_1_2, rem2_0), d_upred_xnext);
        let right_20 = add!(pfx_u, d_x_v);
        let right_30 = add!(pfx_x, d_u_v);
        lb_min = mn!(lb_min, add!(add!(add!(head_20_30, mn!(right_20, right_30)), cap_20_30), m_x));

        let cap_11 = add!(cap!(add!(r1_m1, send2_1)), cap!(add!(sub!(r2_load, send2_1), send1_1)));
        lb_min = mn!(
            lb_min,
            add!(
                add!(sub!(sub!(route_total_dist, rem1_1), rem2_1), add!(d_upred_v, d_v_x)),
                add!(add!(d_vpred_u, d_u_y), cap_11)
            )
        );

        let cap_21_31 = add!(cap!(add!(r1_m2, send2_1)), cap!(add!(sub!(r2_load, send2_1), send1_2)));
        let head_21_31 = add!(add!(sub!(base_1_2, rem2_1), d_upred_v), d_v_xnext);
        let right_21 = add!(pfx_u, d_x_y);
        let right_31 = add!(pfx_x, d_u_y);
        lb_min = mn!(lb_min, add!(add!(add!(head_21_31, mn!(right_21, right_31)), cap_21_31), m_x));

        let cap_22_33 = add!(cap!(add!(r1_m2, send2_2)), cap!(add!(sub!(r2_load, send2_2), send1_2)));
        let left_fwd = add!(pfx_v, d_y_xnext);
        let left_rev = add!(add!(d_upred_y, b!(src.t_d_y_v)), d_v_xnext);
        let right_fwd = add!(pfx_u, d_x_ynext);
        let right_rev = add!(pfx_x, d_u_ynext);
        lb_min = mn!(
            lb_min,
            add!(
                add!(sub!(base_1_2, rem2_2), add!(mn!(left_fwd, left_rev), mn!(right_fwd, right_rev))),
                add!(cap_22_33, m_xy)
            )
        );

        if self.params.allow_swap3 {
            let from_yn = data.distance_row(src.t_yn);
            let to_y2n = data.distance_column(src.t_y2n);
            let d_upred_x2next = gather!(matrix, add!(idx_upred, x2n));
            let d_v_x2next = gather!(from_v, x2n);
            let d_y_x2next = gather!(from_y, x2n);
            let d_ynext_x2next = gather!(from_yn, x2n);
            let d_xnext_v = gather!(to_v, xn);
            let d_xnext_y = gather!(to_y, xn);
            let d_xnext_ynext = gather!(to_yn, xn);
            let d_xnext_y2next = gather!(to_y2n, xn);
            let rem1_3 = add!(rem1_2, xn_dts);
            let send1_3 = add!(send1_2, xn_load);
            let send2_3 = b!(src.t_v_load + src.t_y_load + src.t_yn_load);
            let rem2_3 = b!(src.t_rem2_0 + src.t_v_dts + src.t_y_dts + src.t_yn_dts);
            let r1_m3 = sub!(r1_load, send1_3);
            let head_4 = add!(add!(sub!(route_total_dist, rem1_3), pfx_u), x_dts);
            let cap_40 = add!(cap!(r1_m3), cap!(add!(r2_load, send1_3)));
            lb_min = mn!(
                lb_min,
                add!(add!(add!(sub!(head_4, rem2_0), d_upred_x2next), add!(d_xnext_v, cap_40)), m_s)
            );
            let cap_41 = add!(cap!(add!(r1_m3, send2_1)), cap!(add!(sub!(r2_load, send2_1), send1_3)));
            lb_min = mn!(
                lb_min,
                add!(
                    add!(add!(sub!(head_4, rem2_1), add!(d_upred_v, d_v_x2next)), add!(d_xnext_y, cap_41)),
                    m_s
                )
            );
            let cap_42_43 = add!(cap!(add!(r1_m3, send2_2)), cap!(add!(sub!(r2_load, send2_2), send1_3)));
            let head_42_43 = add!(sub!(head_4, rem2_2), d_xnext_ynext);
            let left_42 = add!(pfx_v, d_y_x2next);
            let left_43 = add!(add!(d_upred_y, b!(src.t_d_y_v)), d_v_x2next);
            lb_min = mn!(lb_min, add!(add!(add!(head_42_43, mn!(left_42, left_43)), cap_42_43), m_sy));
            let cap_44 = add!(cap!(add!(r1_m3, send2_3)), cap!(add!(sub!(r2_load, send2_3), send1_3)));
            lb_min = mn!(
                lb_min,
                add!(
                    add!(
                        add!(sub!(head_4, rem2_3), add!(d_upred_v, b!(src.t_v_dts + src.t_y_dts))),
                        add!(add!(d_ynext_x2next, d_xnext_y2next), cap_44)
                    ),
                    _mm256_or_si256(m_sy, m_yn)
                )
            );
        }

        let reject = _mm256_cmpgt_epi32(lb_min, max_acc);
        (!(_mm256_movemask_ps(_mm256_castsi256_ps(reject)) as u32) & 255) | scalar
    }

    #[cfg(not(target_arch = "x86_64"))]
    unsafe fn batch8_rev(&self, _src: &InterSrc, _ids: *const usize) -> u32 {
        255
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn batch8(&self, src: &InterSrc, ids: *const usize) -> BatchOut {
        use std::arch::x86_64::*;
        let data = self.data.as_ref();
        macro_rules! b {
            ($x:expr) => {
                _mm256_set1_epi32($x as i32)
            };
        }
        macro_rules! add {
            ($x:expr, $y:expr) => {
                _mm256_add_epi32($x, $y)
            };
        }
        macro_rules! sub {
            ($x:expr, $y:expr) => {
                _mm256_sub_epi32($x, $y)
            };
        }
        macro_rules! mn {
            ($x:expr, $y:expr) => {
                _mm256_min_epi32($x, $y)
            };
        }
        macro_rules! gather {
            ($slice:expr, $idx:expr) => {
                _mm256_i32gather_epi32::<4>($slice.as_ptr(), $idx)
            };
        }
        let lane_ids = _mm256_setr_epi32(
            *ids as i32,
            *ids.add(1) as i32,
            *ids.add(2) as i32,
            *ids.add(3) as i32,
            *ids.add(4) as i32,
            *ids.add(5) as i32,
            *ids.add(6) as i32,
            *ids.add(7) as i32,
        );
        let mut recs: [__m256i; 24] = [_mm256_setzero_si256(); 24];
        self.load_records(ids, &mut recs);
        macro_rules! rec {
            ($k:expr) => {
                recs[$k]
            };
        }
        let zero = _mm256_setzero_si256();
        let pcap = b!(self.params.penalty_capa);
        let max_cap = b!(data.max_capacity);
        macro_rules! cap {
            ($load:expr) => {
                _mm256_mullo_epi32(_mm256_max_epi32(sub!($load, max_cap), zero), pcap)
            };
        }

        let r2_load = rec!(CUT_R_LOAD);
        let r2_dist = rec!(CUT_R_DIST);
        let r2_cost = rec!(CUT_R_COST);
        let v = rec!(CUT_V);
        let y = rec!(CUT_Y);
        let yn = rec!(CUT_YN);
        let rem2_0 = rec!(CUT_REM0);
        let v_dts = rec!(CUT_V_DTS);
        let y_dts = rec!(CUT_Y_DTS);
        let yn_dts = rec!(CUT_YN_DTS);
        let v_load = rec!(CUT_V_LOAD);
        let y_load = rec!(CUT_Y_LOAD);
        let yn_load = rec!(CUT_YN_LOAD);
        let d_y_v = rec!(CUT_D_Y_V);
        let flags = rec!(CUT_FLAGS);

        let over = _mm256_cmpgt_epi32(r2_cost, b!(COST_CAP as i32 - 1));
        let scalar = _mm256_movemask_ps(_mm256_castsi256_ps(over)) as u32;

        let credit = b!(self.move_bar.max(-COST_CAP));
        let max_acc = add!(add!(b!(src.r1_cost), r2_cost), credit);
        let route_total_dist = add!(b!(src.r1_dist), r2_dist);

        let m_v = _mm256_slli_epi32::<28>(_mm256_and_si256(flags, b!(1)));
        let m_y = _mm256_slli_epi32::<27>(_mm256_and_si256(flags, b!(2)));
        let m_yn = _mm256_slli_epi32::<26>(_mm256_and_si256(flags, b!(4)));
        let m_x = b!(if src.xd { LANE_MASK } else { 0 });
        let m_xv = _mm256_or_si256(m_x, m_v);
        let m_xvy = _mm256_or_si256(m_xv, m_y);

        let to_u = data.distance_column(src.u);
        let to_x = data.distance_column(src.x);
        let from_upred = data.distance_row(src.upred);
        let from_u = data.distance_row(src.u);
        let from_x = data.distance_row(src.x);

        let d_vpred_u = gather!(to_u, lane_ids);
        let d_vpred_x = gather!(to_x, lane_ids);
        let d_upred_v = gather!(from_upred, v);
        let d_upred_y = gather!(from_upred, y);
        let d_u_v = gather!(from_u, v);
        let d_u_y = gather!(from_u, y);
        let d_u_ynext = gather!(from_u, yn);
        let d_x_v = gather!(from_x, v);
        let d_x_y = gather!(from_x, y);
        let d_x_ynext = gather!(from_x, yn);
        let d_v_x = gather!(to_x, v);

        let send2_1 = v_load;
        let send2_2 = add!(v_load, y_load);
        let rem2_1 = add!(rem2_0, v_dts);
        let rem2_2 = add!(rem2_1, y_dts);

        let r1_m1 = b!(src.r1_load - src.send1_1);
        let r1_m2 = b!(src.r1_load - src.send1_2);
        let cap_10 = add!(cap!(r1_m1), cap!(add!(r2_load, b!(src.send1_1))));
        let lb10 = add!(
            add!(
                add!(sub!(sub!(route_total_dist, b!(src.rem1_1)), rem2_0), b!(src.d_upred_x)),
                add!(d_vpred_u, d_u_v)
            ),
            cap_10
        );
        let mut lb_min = lb10;

        let pfx_u = add!(d_vpred_u, b!(src.d_u_x));
        let pfx_x = add!(d_vpred_x, b!(src.d_x_u));
        let pfx_v = add!(d_upred_v, v_dts);
        let base_1_2 = sub!(route_total_dist, b!(src.rem1_2));

        let cap_20_30 = add!(cap!(r1_m2), cap!(add!(r2_load, b!(src.send1_2))));
        let head_20_30 = add!(sub!(base_1_2, rem2_0), b!(src.d_upred_xnext));
        let right_20 = add!(pfx_u, d_x_v);
        let right_30 = add!(pfx_x, d_u_v);
        lb_min = mn!(
            lb_min,
            add!(add!(add!(head_20_30, mn!(right_20, right_30)), cap_20_30), m_x)
        );

        let cap_11 = add!(
            cap!(add!(r1_m1, send2_1)),
            cap!(add!(sub!(r2_load, send2_1), b!(src.send1_1)))
        );
        lb_min = mn!(
            lb_min,
            add!(
                add!(
                    add!(
                        sub!(sub!(route_total_dist, b!(src.rem1_1)), rem2_1),
                        add!(d_upred_v, d_v_x)
                    ),
                    add!(add!(d_vpred_u, d_u_y), cap_11)
                ),
                m_v
            )
        );

        let cap_21_31 = add!(
            cap!(add!(r1_m2, send2_1)),
            cap!(add!(sub!(r2_load, send2_1), b!(src.send1_2)))
        );
        let to_xnext = data.distance_column(src.xnext);
        let d_v_xnext = gather!(to_xnext, v);
        let d_y_xnext = gather!(to_xnext, y);
        let head_21_31 = add!(add!(sub!(base_1_2, rem2_1), d_upred_v), d_v_xnext);
        let right_21 = add!(pfx_u, d_x_y);
        let right_31 = add!(pfx_x, d_u_y);
        lb_min = mn!(
            lb_min,
            add!(add!(add!(head_21_31, mn!(right_21, right_31)), cap_21_31), m_xv)
        );

        let cap_22_33 = add!(
            cap!(add!(r1_m2, send2_2)),
            cap!(add!(sub!(r2_load, send2_2), b!(src.send1_2)))
        );
        let left_fwd = add!(pfx_v, d_y_xnext);
        let left_rev = add!(add!(d_upred_y, d_y_v), d_v_xnext);
        let right_fwd = add!(pfx_u, d_x_ynext);
        let right_rev = add!(pfx_x, d_u_ynext);
        lb_min = mn!(
            lb_min,
            add!(
                add!(
                    add!(sub!(base_1_2, rem2_2), add!(mn!(left_fwd, left_rev), mn!(right_fwd, right_rev))),
                    cap_22_33
                ),
                m_xvy
            )
        );

        if src.swap3 {
            let y2n = rec!(CUT_Y2N);
            let from_xnext = data.distance_row(src.xnext);
            let to_x2next = data.distance_column(src.x2next);
            let send2_3 = add!(send2_2, yn_load);
            let rem2_3 = add!(rem2_2, yn_dts);
            let r1_m3 = b!(src.r1_load - src.send1_3);
            let d_v_x2next = gather!(to_x2next, v);
            let d_y_x2next = gather!(to_x2next, y);
            let d_ynext_x2next = gather!(to_x2next, yn);
            let d_xnext_v = gather!(from_xnext, v);
            let d_xnext_y = gather!(from_xnext, y);
            let d_xnext_ynext = gather!(from_xnext, yn);
            let d_xnext_y2next = gather!(from_xnext, y2n);
            let head_4 = add!(
                add!(sub!(route_total_dist, b!(src.rem1_3)), pfx_u),
                b!(src.d_x_xnext)
            );
            let cap_40 = add!(cap!(r1_m3), cap!(add!(r2_load, b!(src.send1_3))));
            lb_min = mn!(
                lb_min,
                add!(
                    add!(add!(sub!(head_4, rem2_0), b!(src.d_upred_x2next)), add!(d_xnext_v, cap_40)),
                    m_x
                )
            );
            let cap_41 = add!(
                cap!(add!(r1_m3, send2_1)),
                cap!(add!(sub!(r2_load, send2_1), b!(src.send1_3)))
            );
            lb_min = mn!(
                lb_min,
                add!(
                    add!(
                        add!(sub!(head_4, rem2_1), add!(d_upred_v, d_v_x2next)),
                        add!(d_xnext_y, cap_41)
                    ),
                    m_xv
                )
            );
            let cap_42_43 = add!(
                cap!(add!(r1_m3, send2_2)),
                cap!(add!(sub!(r2_load, send2_2), b!(src.send1_3)))
            );
            let head_42_43 = add!(sub!(head_4, rem2_2), d_xnext_ynext);
            let left_42 = add!(pfx_v, d_y_x2next);
            let left_43 = add!(add!(d_upred_y, d_y_v), d_v_x2next);
            lb_min = mn!(
                lb_min,
                add!(add!(add!(head_42_43, mn!(left_42, left_43)), cap_42_43), m_xvy)
            );
            let cap_44 = add!(
                cap!(add!(r1_m3, send2_3)),
                cap!(add!(sub!(r2_load, send2_3), b!(src.send1_3)))
            );
            lb_min = mn!(
                lb_min,
                add!(
                    add!(
                        add!(sub!(head_4, rem2_3), add!(d_upred_v, add!(v_dts, y_dts))),
                        add!(add!(d_ynext_x2next, d_xnext_y2next), cap_44)
                    ),
                    _mm256_or_si256(m_xvy, m_yn)
                )
            );
        }

        let reject = _mm256_cmpgt_epi32(lb_min, max_acc);
        let inter_pass = !(_mm256_movemask_ps(_mm256_castsi256_ps(reject)) as u32) & 255;

        let p_end = rec!(CUT_P_END);
        let p_tw = rec!(CUT_P_TW);
        let p_load = rec!(CUT_P_LOAD);
        let p_dist = rec!(CUT_P_DIST);
        let s_latest = rec!(CUT_S_LATEST);
        let s_tw = rec!(CUT_S_TW);
        let s_load = rec!(CUT_S_LOAD);
        let s_dist = rec!(CUT_S_DIST);
        let dist1 = add!(add!(b!(src.left1_dist), s_dist), d_upred_v);
        let dist2 = add!(add!(p_dist, b!(src.right2_dist)), d_vpred_u);
        let load1 = add!(b!(src.left1_load), s_load);
        let load2 = add!(p_load, b!(src.right2_load));
        let two_lb = add!(add!(dist1, dist2), add!(cap!(load1), cap!(load2)));
        let warp1 = _mm256_max_epi32(sub!(add!(b!(src.left1_end), d_upred_v), s_latest), zero);
        let warp2 = _mm256_max_epi32(sub!(add!(p_end, d_vpred_u), b!(src.right2_latest)), zero);
        let two_tw = add!(add!(add!(b!(src.left1_tw), s_tw), add!(p_tw, b!(src.right2_tw))), add!(warp1, warp2));

        let two_reject = _mm256_cmpgt_epi32(two_lb, max_acc);
        let two_ok = (!(_mm256_movemask_ps(_mm256_castsi256_ps(two_reject)) as u32) & 255) | scalar;
        let mut out = BatchOut {
            inter_pass: inter_pass | scalar,
            two_ok,
            rev_pass: 0,
            scalar,
            two_lb: [0; 8],
            two_tw: [0; 8],
        };
        _mm256_storeu_si256(out.two_lb.as_mut_ptr() as *mut __m256i, two_lb);
        _mm256_storeu_si256(out.two_tw.as_mut_ptr() as *mut __m256i, two_tw);
        out
    }

    #[cfg(not(target_arch = "x86_64"))]
    unsafe fn batch8(&self, _src: &InterSrc, _ids: *const usize) -> BatchOut {
        BatchOut {
            inter_pass: 255,
            two_ok: 255,
            rev_pass: 255,
            scalar: 255,
            two_lb: [0; 8],
            two_tw: [0; 8],
        }
    }

    pub fn continue_repair(
        &mut self,
        rng: &mut SmallRng,
        params: Params,
        factor: usize,
    ) -> Vec<Vec<usize>> {
        let stale_triples = params.allow_swap3 && !self.params.allow_swap3;
        self.params = params;
        debug_assert!(
            !self.routes.is_empty(),
            "continue_repair requires a loaded LS state"
        );
        self.params.penalty_tw = factor.saturating_mul(self.params.penalty_tw).min(10_000);
        self.params.penalty_capa = factor.saturating_mul(self.params.penalty_capa).min(10_000);
        self.nb_moves += 1;
        for rid in 0..self.routes.len() {
            self.when_last_modified[rid] = 0;
            let r = &self.routes[rid];
            if stale_triples || r.load > self.data.max_capacity || r.tw > 0 {
                self.update_route(rid);
            }
        }
        self.search(rng);
        self.export_routes()
    }

    #[inline(never)]
    fn run_intra_block(&mut self, r1: usize, pos1: usize) -> (i64, usize) {
        let mut best = self.run_intra_route_relocate(r1, pos1);
        let mut kind = 4usize;
        let mut plan = self.last_plan;
        let delta = self.run_intra_route_oropt2(r1, pos1);
        let better = delta < best;
        best = if better { delta } else { best };
        kind = if better { 5 } else { kind };
        plan = if better { self.last_plan } else { plan };
        let delta = self.run_intra_route_swap(r1, pos1);
        let better = delta < best;
        best = if better { delta } else { best };
        kind = if better { 6 } else { kind };
        plan = if better { self.last_plan } else { plan };
        let delta = self.run_2opt(r1, pos1);
        let better = delta < best;
        best = if better { delta } else { best };
        kind = if better { 7 } else { kind };
        plan = if better { self.last_plan } else { plan };
        self.last_plan = plan;
        (best, kind)
    }

    #[inline(always)]
    fn evaluate_and_apply_best_move_for_customer(
        &mut self,
        c1: usize,
        last_tested: usize,
        loop_id: usize,
    ) -> i64 {
        self.move_bar = self.move_credit;
        let mut best_delta: i64 = i64::MAX;
        let mut bk: usize = 0;
        let mut b_r1: usize = 0;
        let mut b_pos1: usize = 0;
        let mut b_r2: usize = 0;
        let mut b_pos2: usize = 0;
        let mut best_plan = MovePlan::None;
        let r1 = copy_at(&self.node_route, c1);
        let pos1 = copy_at(&self.node_pos, c1);
        let ptw = self.params.penalty_tw as i64;

        macro_rules! take {
            ($delta:expr, $kind:expr, $r1:expr, $pos1:expr, $r2:expr, $pos2:expr, $plan:expr) => {{
                let delta: i64 = $delta;
                let better = delta < best_delta;
                best_delta = if better { delta } else { best_delta };
                if better {
                    let tightened = if self.data.root_nb_nodes == 1000 {
                        (delta as f64 * 0.5).round() as i64
                    } else {
                        delta.saturating_sub(1)
                    };
                    self.move_bar = self.move_credit.min(tightened);
                }
                bk = if better { $kind } else { bk };
                b_r1 = if better { $r1 } else { b_r1 };
                b_pos1 = if better { $pos1 } else { b_pos1 };
                b_r2 = if better { $r2 } else { b_r2 };
                b_pos2 = if better { $pos2 } else { b_pos2 };
                if $plan {
                    best_plan = if better { self.last_plan } else { best_plan };
                }
            }};
        }

        {
            let r1_last_mod = copy_at(&self.when_last_modified, r1);
            let need_r2_stale_check = r1_last_mod <= last_tested;
            let neighbors_start = copy_at(&self.neighbors_before_offsets, c1);
            let neighbors_end = copy_at(&self.neighbors_before_offsets, c1 + 1);
            macro_rules! nb_keep {
                ($k:expr) => {{
                    let c2 = copy_at(&self.neighbors_before, $k);
                    let r2 = copy_at(&self.node_route, c2);
                    let r2_fresh_enough = need_r2_stale_check
                        & (copy_at(&self.when_last_modified, r2) <= last_tested);
                    ((r1 != r2) & !r2_fresh_enough) as u32
                }};
            }
            macro_rules! nb_one {
                ($k:expr, $head:expr) => {{
                    let c2 = copy_at(&self.neighbors_before, $k);
                    let r2 = copy_at(&self.node_route, c2);
                    let pos2 = copy_at(&self.node_pos, c2);
                    let delta = self.run_inter_route(r1, pos1, r2, pos2 + 1);
                    take!(delta, 1, r1, pos1, r2, pos2 + 1, true);
                    if $head {
                        let delta = self.run_inter_route(r2, pos2, r1, pos1);
                        take!(delta, 1, r2, pos2, r1, pos1, true);
                    }
                    let delta = self.run_2optstar(r1, pos1, r2, pos2 + 1);
                    take!(delta, 2, r1, pos1, r2, pos2 + 1, false);
                }};
            }
            macro_rules! nb_one_b {
                ($k:expr, $j:expr, $head:expr, $out:expr) => {{
                    let c2 = copy_at(&self.neighbors_before, $k);
                    let r2 = copy_at(&self.node_route, c2);
                    let pos2 = copy_at(&self.node_pos, c2);
                    let bit = 1u32 << $j;
                    let lane_scalar = $out.scalar & bit != 0;

                    if $out.inter_pass & bit != 0 {
                        let delta = if lane_scalar {
                            self.run_inter_route(r1, pos1, r2, pos2 + 1)
                        } else {
                            self.run_inter_route_full(r1, pos1, r2, pos2 + 1)
                        };
                        take!(delta, 1, r1, pos1, r2, pos2 + 1, true);
                    }

                    if $head && ($out.rev_pass & bit != 0) {
                        let delta = if lane_scalar {
                            self.run_inter_route(r2, pos2, r1, pos1)
                        } else {
                            self.run_inter_route_full(r2, pos2, r1, pos1)
                        };
                        take!(delta, 1, r2, pos2, r1, pos1, true);
                    }

                    let delta = if lane_scalar {
                        self.run_2optstar(r1, pos1, r2, pos2 + 1)
                    } else {
                        let old_cost =
                            route_at(&self.routes, r1).cost + route_at(&self.routes, r2).cost;
                        let max_acceptable_cost = old_cost + self.move_credit;
                        let lb_cost = $out.two_lb[$j] as i64;
                        let new_cost = lb_cost + ($out.two_tw[$j] as i64) * ptw;
                        let ok = ($out.two_ok & bit != 0) & (new_cost <= max_acceptable_cost);
                        if ok { new_cost - old_cost } else { NO_MOVE }
                    };
                    take!(delta, 2, r1, pos1, r2, pos2 + 1, false);
                }};
            }
            macro_rules! sweep8 {
                ($keep:ident, $one:ident, $start:expr, $stop:expr, $head:expr) => {{
                    let mut k = $start;
                    while k < $stop {
                        let last = $stop - 1;
                        let rem = min($stop - k, 8);
                        let mut live = ($keep!(k)
                                | ($keep!(min(k + 1, last)) << 1)
                                | ($keep!(min(k + 2, last)) << 2)
                                | ($keep!(min(k + 3, last)) << 3)
                                | ($keep!(min(k + 4, last)) << 4)
                                | ($keep!(min(k + 5, last)) << 5)
                                | ($keep!(min(k + 6, last)) << 6)
                                | ($keep!(min(k + 7, last)) << 7)) & (u32::MAX >> (32 - rem));
                        while live != 0 {
                            let j = live.trailing_zeros() as usize;
                            live &= live - 1;
                            $one!(k + j, $head);
                        }
                        k += 8;
                    }
                }};
            }
            macro_rules! sweep8b {
                ($keep:ident, $one:ident, $start:expr, $stop:expr, $head:expr, $src:expr) => {{
                    let mut k = $start;
                    while k < $stop {
                        let last = $stop - 1;
                        let rem = min($stop - k, 8);
                        let live = ($keep!(k)
                                | ($keep!(min(k + 1, last)) << 1)
                                | ($keep!(min(k + 2, last)) << 2)
                                | ($keep!(min(k + 3, last)) << 3)
                                | ($keep!(min(k + 4, last)) << 4)
                                | ($keep!(min(k + 5, last)) << 5)
                                | ($keep!(min(k + 6, last)) << 6)
                                | ($keep!(min(k + 7, last)) << 7)) & (u32::MAX >> (32 - rem));
                        if live != 0 {
                            let mut out = unsafe { self.batch8(&$src, self.neighbors_before.as_ptr().add(k)) };
                            if $head {
                                out.rev_pass = unsafe { self.batch8_rev(&$src, self.neighbors_before.as_ptr().add(k)) };
                            }
                            let mut todo = live & (out.inter_pass | out.two_ok | out.rev_pass);
                            while todo != 0 {
                                let j = todo.trailing_zeros() as usize;
                                todo &= todo - 1;
                                $one!(k + j, j, $head, out);
                            }
                        }
                        k += 8;
                    }
                }};
            }
            let src = if !self.batch_ok {
                InterSrc::default()
            } else if pos1 == 1 {
                self.inter_src::<true>(r1, pos1)
            } else {
                self.inter_src::<false>(r1, pos1)
            };
            if self.batch_ok && src.valid {
                if pos1 == 1 {
                    sweep8b!(nb_keep, nb_one_b, neighbors_start, neighbors_end, true, src);
                } else {
                    sweep8b!(nb_keep, nb_one_b, neighbors_start, neighbors_end, false, src);
                }
            } else if pos1 == 1 {
                sweep8!(nb_keep, nb_one, neighbors_start, neighbors_end, true);
            } else {
                sweep8!(nb_keep, nb_one, neighbors_start, neighbors_end, false);
            }

            let capacity_start = copy_at(&self.neighbors_capacity_swap_offsets, c1);
            let capacity_end = copy_at(&self.neighbors_capacity_swap_offsets, c1 + 1);
            macro_rules! cap_keep {
                ($k:expr) => {{
                    let c2 = copy_at(&self.neighbors_capacity_swap, $k);
                    let r2 = copy_at(&self.node_route, c2);
                    let r2_fresh_enough = need_r2_stale_check
                        & (copy_at(&self.when_last_modified, r2) <= last_tested);
                    ((r1 != r2) & !r2_fresh_enough) as u32
                }};
            }
            macro_rules! cap_one {
                ($k:expr, $head:expr) => {{
                    let c2 = copy_at(&self.neighbors_capacity_swap, $k);
                    let r2 = copy_at(&self.node_route, c2);
                    let pos2 = copy_at(&self.node_pos, c2);
                    let delta = self.run_swapstar(r1, pos1, r2, pos2);
                    take!(delta, 3, r1, pos1, r2, pos2, true);
                }};
            }
            macro_rules! sweep8s {
                ($keep:ident, $one:ident, $start:expr, $stop:expr, $head:expr) => {{
                    let mut k = $start;
                    while k < $stop {
                        let last = $stop - 1;
                        let rem = min($stop - k, 8);
                        let mut live = ($keep!(k)
                                | ($keep!(min(k + 1, last)) << 1)
                                | ($keep!(min(k + 2, last)) << 2)
                                | ($keep!(min(k + 3, last)) << 3)
                                | ($keep!(min(k + 4, last)) << 4)
                                | ($keep!(min(k + 5, last)) << 5)
                                | ($keep!(min(k + 6, last)) << 6)
                                | ($keep!(min(k + 7, last)) << 7)) & (u32::MAX >> (32 - rem));
                        if live != 0 {
                            let mut lane_ids = [0usize; 8];
                            lane_ids.copy_from_slice(&self.neighbors_capacity_swap[k..k + 8]);
                            live &= unsafe { self.batch_swap8(r1, pos1, &lane_ids, live) };
                            while live != 0 {
                                let j = live.trailing_zeros() as usize;
                                live &= live - 1;
                                $one!(k + j, $head);
                            }
                        }
                        k += 8;
                    }
                }};
            }
            if self.batch_ok {
                sweep8s!(cap_keep, cap_one, capacity_start, capacity_end, false);
            } else {
                sweep8!(cap_keep, cap_one, capacity_start, capacity_end, false);
            }

            if loop_id > 1 && (loop_id == 2 || r1_last_mod > last_tested) {
                if let Some(&r2) = self.empty_routes.first() {
                    let pos2 = 1;

                    if pos1 > 1 {
                        let delta = self.run_2optstar(r1, pos1, r2, pos2);
                        take!(delta, 2, r1, pos1, r2, pos2, false);
                    }

                    let delta = self.run_inter_route_empty(r1, pos1, r2);
                    take!(delta, 1, r1, pos1, r2, pos2, true);
                }
            }
        }

        if copy_at(&self.when_last_modified, r1) > last_tested {
            let (delta, kind) = self.run_intra_block(r1, pos1);
            take!(delta, kind, r1, pos1, 0, 0, true);
        }

        if bk == 0 {
            return NO_MOVE;
        }
        let mv = match bk {
            1 => CandidateMove::InterRoute { r1: b_r1, pos1: b_pos1, r2: b_r2, pos2: b_pos2 },
            2 => CandidateMove::TwoOptStar { r1: b_r1, pos1: b_pos1, r2: b_r2, pos2: b_pos2 },
            3 => CandidateMove::SwapStar { r1: b_r1, pos1: b_pos1, r2: b_r2, pos2: b_pos2 },
            4 => CandidateMove::IntraRelocate { r1: b_r1, pos1: b_pos1 },
            5 => CandidateMove::IntraOrOpt2 { r1: b_r1, pos1: b_pos1 },
            6 => CandidateMove::IntraSwap { r1: b_r1, pos1: b_pos1 },
            _ => CandidateMove::Intra2Opt { r1: b_r1, pos1: b_pos1 },
        };
        let plan = if bk == 2 { MovePlan::TwoOptStar } else { best_plan };
        let applied_delta = self.apply_planned_move(mv, plan);
        debug_assert!(
            applied_delta.is_some(),
            "Best candidate move was expected to be applicable"
        );
        let Some(delta) = applied_delta else {
            return NO_MOVE;
        };
        debug_assert_eq!(
            delta,
            best_delta,
            "Applied move delta differs from evaluated best delta for {mv:?} with {plan:?}",
        );
        delta
    }

    fn search(&mut self, rng: &mut SmallRng) {
        if self.batch {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                self.search_avx2(rng);
                return;
            }
        }
        self.search_impl(rng);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn search_avx2(&mut self, rng: &mut SmallRng) {
        self.search_impl(rng);
    }

    #[inline(always)]
    fn search_impl(&mut self, rng: &mut SmallRng) {
        let mut improved = true;
        let mut loop_id = 0;
        self.move_credit = 0;
        self.batch_ok = self.batch
            && (self.params.penalty_capa as i64) * (2 * self.total_demand + 1) < COST_CAP
            && (self.params.max_credit_deterioration as i64) < COST_CAP;
        self.loop_order_nodes.shuffle(rng);
        while improved || loop_id < 2 {
            improved = false;
            loop_id += 1;
            if loop_id == 2 {
                self.move_credit = -1;
            }
            for idx in 0..self.loop_order_nodes.len() {
                let c1 = copy_at(&self.loop_order_nodes, idx);
                let mut c1_repeat = true;
                while c1_repeat {
                    c1_repeat = false;
                    let last_tested = copy_at(&self.when_last_tested, c1);
                    debug_assert!(c1 < self.when_last_tested.len());
                    unsafe {
                        *self.when_last_tested.get_unchecked_mut(c1) = self.nb_moves;
                    }
                    let delta = if loop_id != 2 && self.nb_moves == last_tested {
                        NO_MOVE
                    } else {
                        self.evaluate_and_apply_best_move_for_customer(c1, last_tested, loop_id)
                    };
                    if delta != NO_MOVE {
                        improved = true;
                        c1_repeat = delta < 0;
                    }
                }
            }
        }
    }
}
}
mod params {
use super::problem::Problem;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

const DEFAULT_EXPLORATION_LEVEL: usize = 4;

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct Params {
    pub exploration_level: usize,
    pub allow_swap3: bool,
    pub granularity: usize,
    pub granularity2: usize,
    pub swapstar_capa_filter: f64,
    pub penalty_tw: usize,
    pub penalty_capa: usize,
    pub target_ratio: f64,
    pub max_it_noimprov: usize,
    pub max_it_total: usize,
    pub nb_it_adapt_penalties: usize,
    pub nb_it_compression: usize,
    pub mu: usize,
    pub mu_start: usize,
    pub lambda: usize,
    pub nb_close: usize,
    pub nb_elite: usize,
    pub crossover_srex_percent: usize,
    pub max_cli_srex: usize,
    pub max_credit_deterioration: usize,
    pub selection_weight_feasible: usize,
    pub factor_split: f64,
    pub decomp_target_size: usize,
    pub decomp_nb_phases: usize,
    #[serde(default)]
    pub stop_distance: i64,
    #[serde(default)]
    pub giveup_at: f64,
    #[serde(default)]
    pub giveup_distance: i64,
    #[serde(default)]
    pub fuel_use: f64,
    #[serde(default)]
    pub ot_max: i64,
    #[serde(default)]
    pub ot_cap: f64,
    #[serde(default)]
    pub ot_window: i64,
    #[serde(default = "no_phase")]
    pub l1_permille: usize,
    #[serde(default = "one_thousand")]
    pub l2_permille: usize,
    #[serde(default = "one_thousand")]
    pub l3_permille: usize,
    #[serde(default = "six_hundred")]
    pub l4_permille: usize,
    #[serde(default = "thirty_two_hundred")]
    pub sub_it_permille: usize,
    #[serde(default = "one_thousand")]
    pub sub_mu_permille: usize,
    #[serde(default = "eight")]
    pub inh_k: usize,
    #[serde(default = "four")]
    pub inh_min_level: usize,
    #[serde(default = "default_sub_it_l4_permille")]
    pub sub_it_l4_permille: Option<usize>,
}

fn one_thousand() -> usize {
    1000
}

fn four() -> usize {
    4
}

fn no_phase() -> usize {
    0
}

fn six_hundred() -> usize {
    600
}

fn thirty_two_hundred() -> usize {
    3200
}

fn eight() -> usize {
    8
}

fn default_sub_it_l4_permille() -> Option<usize> {
    Some(1400)
}

struct Preset {
    allow_swap3: bool,
    granularity: usize,
    granularity2: usize,
    max_it_noimprov: usize,
    max_it_total: usize,
    nb_it_adapt_penalties: usize,
    nb_it_compression: usize,
    mu: usize,
    mu_start: usize,
    lambda: usize,
    nb_close: usize,
    nb_elite: usize,
    max_cli_srex: usize,
    decomp_target_size: usize,
    decomp_nb_phases: usize,
}

#[rustfmt::skip]
const PRESETS_BY_EXPLORATION_LEVEL: [Preset; 7] = [
    Preset { allow_swap3: true,  granularity: 30, granularity2: 40, max_it_noimprov:    0, max_it_total:     0,
             nb_it_adapt_penalties: 20, nb_it_compression:  75, mu:  2, mu_start:  1, lambda:  1,
             nb_close: 1, nb_elite: 1, max_cli_srex: 50, decomp_target_size: 250, decomp_nb_phases: 2 },
    Preset { allow_swap3: true,  granularity: 30, granularity2: 40, max_it_noimprov:    0, max_it_total:     0,
             nb_it_adapt_penalties: 20, nb_it_compression:  75, mu:  2, mu_start: 20, lambda:  1,
             nb_close: 1, nb_elite: 1, max_cli_srex: 50, decomp_target_size: 250, decomp_nb_phases: 2 },
    Preset { allow_swap3: true,  granularity: 30, granularity2: 40, max_it_noimprov:   40, max_it_total:   150,
             nb_it_adapt_penalties: 20, nb_it_compression:  75, mu:  3, mu_start:  6, lambda:  3,
             nb_close: 1, nb_elite: 1, max_cli_srex: 50, decomp_target_size: 250, decomp_nb_phases: 2 },
    Preset { allow_swap3: false, granularity: 20, granularity2: 30, max_it_noimprov:  100, max_it_total:   400,
             nb_it_adapt_penalties: 20, nb_it_compression:  75, mu:  7, mu_start: 10, lambda:  3,
             nb_close: 2, nb_elite: 2, max_cli_srex: 50, decomp_target_size: 250, decomp_nb_phases: 2 },
    Preset { allow_swap3: false, granularity: 20, granularity2: 30, max_it_noimprov:  200, max_it_total:  2000,
             nb_it_adapt_penalties: 50, nb_it_compression:  75, mu: 15, mu_start: 20, lambda:  5,
             nb_close: 2, nb_elite: 3, max_cli_srex: 50, decomp_target_size: 250, decomp_nb_phases: 2 },
    Preset { allow_swap3: false, granularity: 25, granularity2: 35, max_it_noimprov: 2000, max_it_total: 20000,
             nb_it_adapt_penalties: 50, nb_it_compression: 100, mu: 25, mu_start: 50, lambda: 40,
             nb_close: 5, nb_elite: 4, max_cli_srex: 75, decomp_target_size: 500, decomp_nb_phases: 3 },
    Preset { allow_swap3: false, granularity: 25, granularity2: 35, max_it_noimprov: 5000, max_it_total: 50000,
             nb_it_adapt_penalties: 50, nb_it_compression: 150, mu: 25, mu_start: 50, lambda: 40,
             nb_close: 5, nb_elite: 4, max_cli_srex: 75, decomp_target_size: 500, decomp_nb_phases: 8 },
];

struct TrackProfile {
    nb_nodes: usize,
    allow_swap3: bool,
    granularity: usize,
    granularity2: usize,
    swapstar_capa_filter: f64,
    max_credit_deterioration: usize,
    decomp_target_size: usize,
    decomp_nb_phases: usize,
    l4_permille: usize,
}

#[rustfmt::skip]
const TRACK_PROFILES: [TrackProfile; 5] = [
    TrackProfile { nb_nodes:  600, allow_swap3: false, granularity: 25, granularity2: 35,
                   swapstar_capa_filter: 0.3, max_credit_deterioration: 10,
                   decomp_target_size: 500, decomp_nb_phases:  3, l4_permille: 600 },
    TrackProfile { nb_nodes:  700, allow_swap3: true, granularity: 32, granularity2: 40,
                   swapstar_capa_filter: 0.45, max_credit_deterioration: 15,
                   decomp_target_size: 175, decomp_nb_phases: 29, l4_permille: 600 },
    TrackProfile { nb_nodes:  800, allow_swap3: false, granularity: 25, granularity2: 35,
                   swapstar_capa_filter: 0.3, max_credit_deterioration: 10,
                   decomp_target_size: 500, decomp_nb_phases:  3, l4_permille: 600 },
    TrackProfile { nb_nodes:  900, allow_swap3: true, granularity: 30, granularity2: 40,
                   swapstar_capa_filter: 0.5, max_credit_deterioration: 20,
                   decomp_target_size: 200, decomp_nb_phases: 12, l4_permille: 400 },
    TrackProfile { nb_nodes: 1000, allow_swap3: true, granularity: 30, granularity2: 40,
                   swapstar_capa_filter: 0.5, max_credit_deterioration: 20,
                   decomp_target_size: 200, decomp_nb_phases: 12, l4_permille: 600 },
];

impl Params {
    pub(super) fn preset(exploration_level: usize, data: &Problem) -> Self {
        let max_dist = data.distance_matrix.iter().copied().max().unwrap_or(1) as f64;
        let max_demand = data
            .node_data
            .iter()
            .skip(1)
            .map(|nd| nd.demand)
            .max()
            .unwrap_or(1) as f64;
        let penalty_capa = (max_dist / max_demand).clamp(1.0, 500.0).round() as usize;

        let preset = match PRESETS_BY_EXPLORATION_LEVEL.get(exploration_level) {
            Some(preset) => preset,
            None => return Self::defaults(data),
        };

        Self {
            exploration_level,
            allow_swap3: preset.allow_swap3,
            granularity: preset.granularity,
            granularity2: preset.granularity2,
            swapstar_capa_filter: 0.3,
            penalty_tw: 10,
            penalty_capa,
            target_ratio: 0.4,
            max_it_noimprov: preset.max_it_noimprov,
            max_it_total: preset.max_it_total,
            nb_it_adapt_penalties: preset.nb_it_adapt_penalties,
            nb_it_compression: preset.nb_it_compression,
            mu: preset.mu,
            mu_start: preset.mu_start,
            lambda: preset.lambda,
            nb_close: preset.nb_close,
            nb_elite: preset.nb_elite,
            crossover_srex_percent: 75,
            max_cli_srex: preset.max_cli_srex,
            max_credit_deterioration: 10,
            selection_weight_feasible: 2,
            factor_split: 1.3,
            decomp_target_size: preset.decomp_target_size,
            decomp_nb_phases: preset.decomp_nb_phases,
            stop_distance: 0,
            giveup_at: 0.0,
            giveup_distance: 0,
            fuel_use: 0.0,
            ot_max: 0,
            ot_cap: 0.0,
            ot_window: 0,
            l1_permille: no_phase(),
            l2_permille: one_thousand(),
            l3_permille: one_thousand(),
            l4_permille: six_hundred(),
            sub_it_permille: thirty_two_hundred(),
            sub_mu_permille: one_thousand(),
            inh_k: eight(),
            inh_min_level: four(),
            sub_it_l4_permille: default_sub_it_l4_permille(),
        }
    }

    pub fn defaults(data: &Problem) -> Self {
        if matches!(data.nb_nodes, 600 | 800) {
            return Self::preset(5, data);
        }
        let mut params = Self::preset(DEFAULT_EXPLORATION_LEVEL, data);
        apply_track_profile(&mut params, data.nb_nodes, true);
        params
    }

    fn with_overrides(base: Self, hyperparameters: &Option<Map<String, Value>>, data: &Problem) -> Self {
        let map = match hyperparameters {
            Some(map) if !map.is_empty() => map,
            _ => return base,
        };
        let mut merged_params = serde_json::to_value(base).expect("Params serializable");
        if let Value::Object(ref mut obj) = merged_params {
            for (k, v) in map {
                if k != "exploration_level" {
                    obj.insert(k.clone(), v.clone());
                }
            }
            for (key, default) in [
                ("inh_k", Value::from(eight())),
                ("inh_min_level", Value::from(four())),
                ("sub_it_l4_permille", Value::from(default_sub_it_l4_permille())),
            ] {
                let valid = match obj.get(key) {
                    Some(Value::Null) => key == "sub_it_l4_permille",
                    Some(value) => value
                        .as_u64()
                        .map_or(false, |u| usize::try_from(u).is_ok()),
                    None => false,
                };
                if !valid {
                    obj.insert(key.to_string(), default);
                }
            }
        }
        serde_json::from_value(merged_params).unwrap_or_else(|_| Self::defaults(data))
    }

    pub fn initialize(hyperparameters: &Option<Map<String, Value>>, data: &Problem) -> Self {
        let requested_level = requested_exploration_level(hyperparameters);

        let base_params = match requested_level {
            Some(level) if level != DEFAULT_EXPLORATION_LEVEL => Self::preset(level, data),
            _ => Self::defaults(data),
        };

        Self::with_overrides(base_params, hyperparameters, data)
    }

    pub(super) fn initialize_at_level(
        hyperparameters: &Option<Map<String, Value>>,
        exploration_level: usize,
        data: &Problem,
    ) -> Self {
        let mut params = Self::preset(exploration_level, data);
        let bare = hyperparameters.as_ref().map(|map| map.is_empty()).unwrap_or(true);
        if bare {
            apply_track_profile(&mut params, data.root_nb_nodes, false);
        }
        Self::with_overrides(params, hyperparameters, data)
    }
}

fn apply_track_profile(params: &mut Params, nb_nodes: usize, include_schedule: bool) {
    if matches!(nb_nodes, 600 | 800) {
        return;
    }
    let Some(profile) = TRACK_PROFILES
        .iter()
        .find(|profile| profile.nb_nodes == nb_nodes)
    else {
        return;
    };
    params.allow_swap3 = profile.allow_swap3;
    params.granularity = profile.granularity;
    params.granularity2 = profile.granularity2;
    params.swapstar_capa_filter = profile.swapstar_capa_filter;
    params.max_credit_deterioration = profile.max_credit_deterioration;
    params.decomp_target_size = profile.decomp_target_size;
    params.decomp_nb_phases = profile.decomp_nb_phases;
    if include_schedule {
        params.l4_permille = profile.l4_permille;
    }
}

fn requested_exploration_level(hyperparameters: &Option<Map<String, Value>>) -> Option<usize> {
    hyperparameters
        .as_ref()
        .and_then(|map| map.get("exploration_level"))
        .and_then(|value| match value {
            Value::Number(n) => n.as_u64().map(|u| u as usize),
            Value::String(s) => s.parse::<usize>().ok(),
            _ => None,
        })
}
}
mod population {
use super::individual::Individual;
use super::params::Params;
use super::problem::Problem;
use rand::{rngs::SmallRng, Rng};
use std::collections::VecDeque;
use std::sync::Arc;

#[derive(Default)]
pub struct Subpopulation {
    pub indivs: Vec<Individual>,
    pub prox: Vec<Vec<(f64, usize)>>,
    pub biased_fitness: Vec<f64>,
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

    pub fn best_feasible(&self) -> Option<&Individual> {
        if !self.feasible.indivs.is_empty() {
            return Some(&self.feasible.indivs[self.feasible.order_cost[0]]);
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
}
mod pred_queue {
pub struct IndexDeque {
    pub data: Vec<usize>,
    pub head: usize,
}

impl IndexDeque {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            data: Vec::with_capacity(cap),
            head: 0,
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.data.clear();
        self.head = 0;
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.head >= self.data.len()
    }

    #[inline]
    pub fn push_back(&mut self, v: usize) {
        self.data.push(v);
    }

    #[inline]
    pub fn back(&self) -> Option<usize> {
        if self.is_empty() {
            None
        } else {
            self.data.last().copied()
        }
    }

    #[inline]
    pub fn front(&self) -> Option<usize> {
        if self.is_empty() {
            None
        } else {
            Some(self.data[self.head])
        }
    }

    #[inline]
    pub fn pop_back(&mut self) {
        if self.is_empty() {
            return;
        }
        self.data.pop();
        if self.head >= self.data.len() {
            self.clear();
        }
    }

    #[inline]
    pub fn pop_front(&mut self) {
        if self.is_empty() {
            return;
        }
        self.head += 1;
        if self.head >= self.data.len() {
            self.clear();
        }
    }
}

pub struct PredQueue {
    pub prev: Vec<usize>,
    pub next: Vec<usize>,
    pub nil: usize,
    pub size: usize,
    pub feas: usize,
    pub no_warp: usize,
}

impl PredQueue {
    pub fn new(n: usize) -> Self {
        Self {
            prev: vec![n; n + 1],
            next: vec![n; n + 1],
            nil: n,
            size: 0,
            feas: n,
            no_warp: n,
        }
    }

    #[inline(always)]
    fn link(&self, id: usize) -> (usize, usize) {
        debug_assert!(id <= self.nil);
        unsafe {
            (
                *self.prev.get_unchecked(id),
                *self.next.get_unchecked(id),
            )
        }
    }

    #[inline(always)]
    pub fn head(&self) -> usize {
        self.link(self.nil).1
    }

    #[inline(always)]
    pub fn tail(&self) -> usize {
        self.link(self.nil).0
    }

    #[inline(always)]
    pub fn succ(&self, id: usize) -> usize {
        self.link(id).1
    }

    #[inline]
    pub fn front(&self) -> Option<usize> {
        let h = self.head();
        if h != self.nil {
            Some(h)
        } else {
            None
        }
    }

    #[inline]
    pub fn back(&self) -> Option<usize> {
        let t = self.tail();
        if t != self.nil {
            Some(t)
        } else {
            None
        }
    }

    #[inline]
    pub fn front2(&self) -> Option<usize> {
        let n = self.succ(self.head());
        if n != self.nil {
            Some(n)
        } else {
            None
        }
    }

    #[inline]
    pub fn insert_back(&mut self, id: usize) {
        let nil = self.nil;
        let t = self.tail();
        unsafe {
            *self.next.get_unchecked_mut(t) = id;
            *self.prev.get_unchecked_mut(id) = t;
            *self.next.get_unchecked_mut(id) = nil;
            *self.prev.get_unchecked_mut(nil) = id;
        }
        self.size += 1;
    }

    #[inline]
    pub fn remove_node(&mut self, id: usize) {
        let (p, n) = self.link(id);
        unsafe {
            *self.next.get_unchecked_mut(p) = n;
            *self.prev.get_unchecked_mut(n) = p;
        }
        self.feas = if self.feas == id { n } else { self.feas };
        self.no_warp = if self.no_warp == id { n } else { self.no_warp };
        self.size -= 1;
    }

    #[inline]
    pub fn remove_front(&mut self) {
        let h = self.head();
        if h != self.nil {
            self.remove_node(h);
        }
    }

    #[inline]
    pub fn remove_front2(&mut self) {
        if let Some(h2) = self.front2() {
            self.remove_node(h2);
        }
    }

    #[inline]
    pub fn feas_prev(&self) -> Option<usize> {
        if self.feas == self.nil {
            return None;
        }
        let p = self.link(self.feas).0;
        if p != self.nil {
            Some(p)
        } else {
            None
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        let nil = self.nil;
        unsafe {
            *self.next.get_unchecked_mut(nil) = nil;
            *self.prev.get_unchecked_mut(nil) = nil;
        }
        self.size = 0;
        self.feas = nil;
        self.no_warp = nil;
    }
}
}
mod problem {
#[derive(Copy, Clone)]
pub struct NodeData {
    pub start_tw: i32,
    pub end_tw: i32,
    pub service_time: i32,
    pub demand: i32,
}

pub struct Problem {
    pub seed: [u8; 32],
    pub nb_nodes: usize,
    pub root_nb_nodes: usize,
    pub nb_vehicles: usize,
    pub lb_vehicles: usize,
    pub fixed_distance_offset: i64,
    pub max_capacity: i32,
    pub distance_matrix: Vec<i32>,
    pub distance_matrix_transposed: Vec<i32>,
    pub node_positions: Vec<(i32, i32)>,
    pub node_data: Vec<NodeData>,
}

impl Problem {
    #[inline(always)]
    pub fn dm(&self, i: usize, j: usize) -> i32 {
        debug_assert!(i < self.nb_nodes && j < self.nb_nodes);
        let idx = i * self.nb_nodes + j;
        debug_assert!(idx < self.distance_matrix.len());
        unsafe { *self.distance_matrix.get_unchecked(idx) }
    }

    #[inline(always)]
    pub fn distance_row(&self, i: usize) -> &[i32] {
        debug_assert!(i < self.nb_nodes);
        let start = i * self.nb_nodes;
        unsafe {
            self.distance_matrix
                .get_unchecked(start..start + self.nb_nodes)
        }
    }

    #[inline(always)]
    pub fn distance_column(&self, j: usize) -> &[i32] {
        debug_assert!(j < self.nb_nodes);
        let start = j * self.nb_nodes;
        unsafe {
            self.distance_matrix_transposed
                .get_unchecked(start..start + self.nb_nodes)
        }
    }

    pub fn transpose_distances(distance_matrix: &[i32], nb_nodes: usize) -> Vec<i32> {
        let mut transposed = vec![0; distance_matrix.len()];
        for i in 0..nb_nodes {
            for j in 0..nb_nodes {
                transposed[j * nb_nodes + i] = distance_matrix[i * nb_nodes + j];
            }
        }
        transposed
    }

    #[inline(always)]
    pub fn nd(&self, i: usize) -> &NodeData {
        debug_assert!(i < self.nb_nodes);
        unsafe { self.node_data.get_unchecked(i) }
    }
}
}
mod reverse_mode {
use super::constructive::Constructive;
use super::genetic::{canonical_routes, Genetic, MIN_FRESH_STARTS, REPAIR_FACTORS};
use super::individual::Individual;
use super::local_search::LocalSearch;
use super::params::Params;
use super::problem::Problem;
use super::sequence::Sequence;
use anyhow::Result;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::vehicle_routing::*;

const MAX_PENALTY: usize = 10_000;
const NB_MASTER_STARTS: usize = 5;
const MIN_SUBPROBLEM_NODES: usize = 5;
const MAX_CARRIED: usize = 64;

struct BlockDonors {
    donors: Vec<Vec<Vec<usize>>>,
    fallback: Vec<Vec<usize>>,
}

fn assemble_carried(
    blocks: &[BlockDonors],
    k: usize,
    nb_nodes: usize,
) -> Vec<Vec<Vec<usize>>> {
    let depth = blocks
        .iter()
        .map(|block| block.donors.len())
        .max()
        .unwrap_or(0)
        .min(k);
    let mut out: Vec<Vec<Vec<usize>>> = Vec::with_capacity(depth);
    let mut seen = vec![false; nb_nodes];
    for j in 0..depth {
        let mut routes: Vec<Vec<usize>> = Vec::new();
        for block in blocks {
            match block.donors.get(j) {
                Some(solution) => routes.extend(solution.iter().cloned()),
                None => {
                    routes.extend(block.fallback.iter().cloned());
                }
            }
        }
        seen.fill(false);
        let mut covered = 0usize;
        let mut well_formed = true;
        for route in &routes {
            let len = route.len();
            if len < 3 || route[0] != 0 || route[len - 1] != 0 {
                well_formed = false;
                break;
            }
            for &id in &route[1..len - 1] {
                if id == 0 || id >= nb_nodes || seen[id] {
                    well_formed = false;
                    break;
                }
                seen[id] = true;
                covered += 1;
            }
            if !well_formed {
                break;
            }
        }
        if !well_formed || covered + 1 != nb_nodes {
            continue;
        }
        out.push(routes);
    }
    out
}

fn restrict_carried(
    carried: &[Vec<Vec<usize>>],
    sub: &Problem,
    global_to_local: &[usize],
    seed_sub_routes: &[Vec<usize>],
    cap: usize,
) -> Vec<Vec<Vec<usize>>> {
    let mut out: Vec<Vec<Vec<usize>>> = Vec::new();
    if cap == 0 {
        return out;
    }
    let nb_nodes = sub.nb_nodes;
    let seed_key = canonical_routes(seed_sub_routes);
    let mut keys: Vec<Vec<Vec<usize>>> = Vec::new();
    let mut seen = vec![false; nb_nodes];
    for solution in carried {
        if out.len() >= cap {
            break;
        }
        seen.fill(false);
        let mut covered = 0usize;
        let mut well_formed = true;
        let mut routes: Vec<Vec<usize>> = Vec::new();
        for route in solution {
            let len = route.len();
            if len < 3 || route[0] != 0 || route[len - 1] != 0 {
                well_formed = false;
                break;
            }
            let mut local: Vec<usize> = vec![0];
            for &gid in &route[1..len - 1] {
                let lid = match global_to_local.get(gid) {
                    Some(&lid) => lid,
                    None => {
                        well_formed = false;
                        break;
                    }
                };
                if lid == usize::MAX {
                    continue;
                }
                if lid == 0 || lid >= nb_nodes || seen[lid] {
                    well_formed = false;
                    break;
                }
                seen[lid] = true;
                covered += 1;
                local.push(lid);
            }
            if !well_formed {
                break;
            }
            if local.len() > 1 {
                local.push(0);
                routes.push(local);
            }
        }
        if !well_formed || covered + 1 != nb_nodes {
            continue;
        }
        if routes.len() > sub.nb_vehicles {
            continue;
        }
        let mut capa_ok = true;
        let mut tw_ok = true;
        for route in &routes {
            let mut seq = Sequence::singleton(sub, route[0]);
            for &id in &route[1..] {
                seq = Sequence::join2(sub, &seq, &Sequence::singleton(sub, id));
            }
            capa_ok &= seq.load <= sub.max_capacity;
            tw_ok &= seq.tw == 0;
        }
        if !capa_ok {
            continue;
        }
        if !tw_ok {
            continue;
        }
        let key = canonical_routes(&routes);
        if key == seed_key {
            continue;
        }
        if keys.contains(&key) {
            continue;
        }
        keys.push(key);
        out.push(routes);
    }
    out
}

#[inline]
fn scale_permille(value: usize, permille: usize) -> usize {
    value.saturating_mul(permille).saturating_add(500) / 1000
}

#[inline]
fn nb_clients_in_route(route: &[usize]) -> usize {
    route.len().saturating_sub(2)
}

fn route_barycenter(data: &Problem, route: &[usize]) -> (f64, f64) {
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut cnt = 0usize;
    for &id in route.iter().skip(1).take(nb_clients_in_route(route)) {
        sx += data.node_positions[id].0 as f64;
        sy += data.node_positions[id].1 as f64;
        cnt += 1;
    }
    (sx / cnt as f64, sy / cnt as f64)
}

fn cluster_route_indices(
    data: &Problem,
    routes: &[Vec<usize>],
    k: usize,
    rng: &mut SmallRng,
) -> Vec<Vec<usize>> {
    let m = routes.len();
    if m == 0 {
        return Vec::new();
    }
    debug_assert!(
        routes.iter().all(|r| r.len() > 2),
        "Reverse-mode decomposition should not receive empty routes"
    );
    let kk = k.max(1).min(m);

    let barycenters: Vec<(f64, f64)> = routes.iter().map(|r| route_barycenter(data, r)).collect();
    let route_clients: Vec<usize> = routes.iter().map(|r| nb_clients_in_route(r)).collect();

    let mut route_indices: Vec<usize> = (0..m).collect();
    route_indices.shuffle(rng);
    let center_routes: Vec<usize> = route_indices[..kk].to_vec();

    let mut assigned = vec![false; m];
    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); kk];
    let mut cluster_clients: Vec<usize> = vec![0; kk];

    for c in 0..kk {
        let rid = center_routes[c];
        assigned[rid] = true;
        clusters[c].push(rid);
        cluster_clients[c] += route_clients[rid];
    }

    let ordered_candidates: Vec<Vec<usize>> = center_routes
        .iter()
        .map(|&center_rid| {
            let (cx, cy) = barycenters[center_rid];
            let mut cand: Vec<(f64, usize)> = (0..m)
                .filter(|&rid| rid != center_rid)
                .map(|rid| {
                    let (x, y) = barycenters[rid];
                    let dx = x - cx;
                    let dy = y - cy;
                    (dx * dx + dy * dy, rid)
                })
                .collect();
            cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            cand.into_iter().map(|(_, rid)| rid).collect()
        })
        .collect();

    let mut candidate_pos = vec![0usize; kk];
    let mut remaining = m - kk;
    while remaining > 0 {
        let mut best_cluster: Option<usize> = None;

        for c in 0..kk {
            while candidate_pos[c] < ordered_candidates[c].len()
                && assigned[ordered_candidates[c][candidate_pos[c]]]
            {
                candidate_pos[c] += 1;
            }
            if candidate_pos[c] == ordered_candidates[c].len() {
                continue;
            }
            if best_cluster.is_none() || cluster_clients[c] < cluster_clients[best_cluster.unwrap()]
            {
                best_cluster = Some(c);
            }
        }

        let c = best_cluster.expect("At least one cluster should still have assignable routes");
        let rid = ordered_candidates[c][candidate_pos[c]];
        candidate_pos[c] += 1;
        assigned[rid] = true;
        clusters[c].push(rid);
        cluster_clients[c] += route_clients[rid];
        remaining -= 1;
    }

    clusters
}

fn build_subproblem(
    data: &Problem,
    clients: &[usize],
    vehicles_hint: usize,
) -> (Problem, Vec<usize>) {
    let mut local_to_global = Vec::with_capacity(clients.len() + 1);
    local_to_global.push(0);
    local_to_global.extend_from_slice(clients);

    let nb_nodes = local_to_global.len();
    let mut node_positions = Vec::with_capacity(nb_nodes);
    let mut node_data = Vec::with_capacity(nb_nodes);
    for &gid in &local_to_global {
        node_positions.push(data.node_positions[gid]);
        node_data.push(data.node_data[gid]);
    }

    let mut distance_matrix = vec![0i32; nb_nodes * nb_nodes];
    for i in 0..nb_nodes {
        let gi = local_to_global[i];
        for j in 0..nb_nodes {
            let gj = local_to_global[j];
            distance_matrix[i * nb_nodes + j] = data.dm(gi, gj);
        }
    }

    let total_demand: i64 = clients.iter().map(|&id| data.nd(id).demand as i64).sum();
    let lb_vehicles =
        ((total_demand + data.max_capacity as i64 - 1) / data.max_capacity as i64) as usize;
    let nb_vehicles = vehicles_hint
        .max(lb_vehicles)
        .max(1)
        .min(clients.len().max(1));
    let distance_matrix_transposed = Problem::transpose_distances(&distance_matrix, nb_nodes);

    (
        Problem {
            seed: data.seed,
            nb_nodes,
            root_nb_nodes: data.root_nb_nodes,
            nb_vehicles,
            lb_vehicles,
            fixed_distance_offset: 0,
            max_capacity: data.max_capacity,
            distance_matrix,
            distance_matrix_transposed,
            node_positions,
            node_data,
        },
        local_to_global,
    )
}

fn map_sub_routes_to_global(
    sub_routes: &[Vec<usize>],
    local_to_global: &[usize],
) -> Vec<Vec<usize>> {
    sub_routes
        .iter()
        .map(|r| r.iter().map(|&lid| local_to_global[lid]).collect())
        .collect()
}

fn build_progressive_master_routes(
    phase_base_routes: &[Vec<usize>],
    clusters: &[Vec<usize>],
    rebuilt_routes: &[Vec<usize>],
    next_cluster_idx: usize,
) -> Vec<Vec<usize>> {
    let mut routes = rebuilt_routes.to_vec();
    for route_group in clusters.iter().skip(next_cluster_idx) {
        for &rid in route_group {
            routes.push(phase_base_routes[rid].clone());
        }
    }
    routes
}

unsafe extern "C" {
    static __fuel_remaining: u64;
}
#[inline(always)]
fn fuel_left() -> u64 {
    unsafe { core::ptr::read_volatile(core::ptr::addr_of!(__fuel_remaining)) }
}

struct MasterState {
    routes: Vec<Vec<usize>>,
    ind: Individual,
}

fn promote_if_improved(
    save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    incumbent: &mut MasterState,
    candidate_routes: &[Vec<usize>],
    candidate_ind: &Individual,
) {
    if candidate_ind.cost < incumbent.ind.cost {
        incumbent.routes = candidate_routes.to_vec();
        incumbent.ind = candidate_ind.clone();
        if let Some(save) = save_solution {
            let _ = save(&Solution {
                routes: incumbent.routes.clone(),
            });
        }
    }
}

fn increase_penalties_for_infeasibility(params: &mut Params, ind: &Individual) {
    let had_capa = ind.load_excess > 0;
    let had_tw = ind.tw_violation > 0;
    if had_capa {
        params.penalty_capa = (2 * params.penalty_capa).min(MAX_PENALTY);
    }
    if had_tw {
        params.penalty_tw = (2 * params.penalty_tw).min(MAX_PENALTY);
    }
}

pub(super) fn solve_reversed_mode(
    data: Problem,
    mut params: Params,
    hyperparameters: &Option<Map<String, Value>>,
    save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
) -> Result<Option<Solution>> {
    let mut rng = SmallRng::from_seed(data.seed);
    let data_arc = Arc::new(data);

    let mut ls = LocalSearch::new(Arc::clone(&data_arc), params);
    let mut best_constructive_feasible: Option<(Vec<Vec<usize>>, Individual)> = None;
    let mut best_ls_repair_feasible: Option<(Vec<Vec<usize>>, Individual)> = None;

    for run_idx in 0..NB_MASTER_STARTS {
        let randomize = run_idx > 0;
        let constructed_routes = Constructive::build_routes(data_arc.as_ref(), &mut rng, randomize);
        let constructed_ind =
            Individual::new_from_routes(data_arc.as_ref(), &params, constructed_routes.clone());
        if constructed_ind.load_excess == 0 && constructed_ind.tw_violation == 0 {
            let replace = best_constructive_feasible
                .as_ref()
                .map_or(true, |(_, best)| constructed_ind.cost < best.cost);
            if replace {
                best_constructive_feasible = Some((constructed_routes.clone(), constructed_ind));
            }
        }

        let routes_after_ls = ls.run_from_routes(&constructed_routes, &[], params, &mut rng);
        let mut candidate_ind =
            Individual::new_from_routes(data_arc.as_ref(), &params, routes_after_ls.clone());
        let mut candidate_routes = routes_after_ls;

        if candidate_ind.load_excess > 0 || candidate_ind.tw_violation > 0 {
            increase_penalties_for_infeasibility(&mut params, &candidate_ind);
            for factor in REPAIR_FACTORS {
                let repaired_routes = ls.continue_repair(&mut rng, params, factor);
                let repaired_ind = Individual::new_from_routes(
                    data_arc.as_ref(),
                    &params,
                    repaired_routes.clone(),
                );
                if repaired_ind.load_excess == 0 && repaired_ind.tw_violation == 0 {
                    candidate_routes = repaired_routes;
                    candidate_ind = repaired_ind;
                    break;
                }
            }
        }

        if candidate_ind.load_excess == 0 && candidate_ind.tw_violation == 0 {
            let replace = best_ls_repair_feasible
                .as_ref()
                .map_or(true, |(_, best)| candidate_ind.cost < best.cost);
            if replace {
                best_ls_repair_feasible = Some((candidate_routes, candidate_ind));
            }
        }
    }

    let mut incumbent = if let Some((routes, ind)) = best_ls_repair_feasible {
        MasterState { routes, ind }
    } else if let Some((routes, ind)) = best_constructive_feasible {
        MasterState { routes, ind }
    } else {
        panic!("No feasible constructive solution found among the reversed-mode starts");
    };

    if let Some(save) = save_solution {
        let _ = save(&Solution {
            routes: incumbent.routes.clone(),
        });
    }

    let fuel_start = fuel_left();
    let fuel_budget = ((fuel_start as f64) * params.fuel_use).max(1.0) as u64;

    let mut exploration_schedule: Vec<usize> = Vec::new();
    if params.exploration_level == 0 {
        for _ in 0..params.decomp_nb_phases {
            exploration_schedule.push(0);
        }
    } else {
        let rung_permille = [
            params.l1_permille,
            params.l2_permille,
            params.l3_permille,
            params.l4_permille,
        ];
        for level in 1..=params.exploration_level {
            let permille = match rung_permille.get(level - 1) {
                Some(&value) => value,
                None => 1000,
            };
            let nb_phases = (params.decomp_nb_phases * permille + 500) / 1000;
            for _ in 0..nb_phases {
                exploration_schedule.push(level);
            }
        }
    }

    let carry_k = params.inh_k.min(MAX_CARRIED);
    let carry_min_level = params.inh_min_level.max(1);
    let carry_active = carry_k > 0 && params.exploration_level > 0;
    let mut carried: Vec<Vec<Vec<usize>>> = Vec::new();

    const ADAPT_MIN_TARGET: usize = 100;
    const ADAPT_MAX_TARGET: usize = 500;
    let base_target_size = params
        .decomp_target_size
        .clamp(ADAPT_MIN_TARGET, ADAPT_MAX_TARGET);
    let mut adaptive_target_size = base_target_size;
    let mut window_start_cost: i64 = incumbent.ind.cost;
    let mut phases_in_window: usize = 0;

    let mut phase_idx = 0usize;
    let mut ot_rounds: i64 = 0;
    while phase_idx < exploration_schedule.len() {
        let sub_exploration_level = exploration_schedule[phase_idx];
        let spent = fuel_start.saturating_sub(fuel_left());
        let prog = (spent as f64) / (fuel_budget as f64);
        if params.giveup_distance > 0
            && params.giveup_at > 0.0
            && prog >= params.giveup_at
            && (incumbent.ind.cost as i64) > params.giveup_distance
        {
            break;
        }
        if params.stop_distance > 0 && (incumbent.ind.cost as i64) <= params.stop_distance {
            break;
        }
        let phase_base_routes = incumbent.routes.clone();
        let target_size = if data_arc.root_nb_nodes == 600 {
            adaptive_target_size
        } else {
            params.decomp_target_size
        };
        let k = ((data_arc.nb_nodes as f64) / (target_size as f64)).round() as usize;
        let clusters = cluster_route_indices(data_arc.as_ref(), &phase_base_routes, k, &mut rng);
        let mut residual_free_routes =
            (data_arc.nb_vehicles as i32) - (phase_base_routes.len() as i32);
        let receive =
            carry_active && sub_exploration_level >= carry_min_level && !carried.is_empty();
        let export = carry_active
            && match exploration_schedule.get(phase_idx + 1) {
                Some(&next_level) => next_level >= carry_min_level,
                None => ot_rounds < params.ot_max && sub_exploration_level >= carry_min_level,
            };
        let mut blocks: Vec<BlockDonors> = Vec::new();

        let mut rebuilt_routes: Vec<Vec<usize>> = Vec::new();
        for (cid, route_group) in clusters.iter().enumerate() {
            let mut clients: Vec<usize> = Vec::new();
            for &rid in route_group {
                clients.extend_from_slice(
                    &phase_base_routes[rid][1..phase_base_routes[rid].len().saturating_sub(1)],
                );
            }
            if clients.is_empty() {
                continue;
            }

            let (sub_problem, local_to_global) = build_subproblem(
                data_arc.as_ref(),
                &clients,
                ((route_group.len() as i32) + residual_free_routes).max(1) as usize,
            );
            if sub_problem.nb_nodes < MIN_SUBPROBLEM_NODES {
                if export {
                    blocks.push(BlockDonors {
                        donors: Vec::new(),
                        fallback: route_group
                            .iter()
                            .map(|&rid| phase_base_routes[rid].clone())
                            .collect(),
                    });
                }
                rebuilt_routes.extend(
                    route_group
                        .iter()
                        .map(|&rid| phase_base_routes[rid].clone()),
                );
                continue;
            }
            let mut global_to_local = vec![usize::MAX; data_arc.nb_nodes];
            for (lid, &gid) in local_to_global.iter().enumerate() {
                global_to_local[gid] = lid;
            }
            let seed_sub_routes: Vec<Vec<usize>> = route_group
                .iter()
                .map(|&rid| {
                    phase_base_routes[rid]
                        .iter()
                        .map(|&gid| global_to_local[gid])
                        .collect()
                })
                .collect();
            let mut params_sub =
                Params::initialize_at_level(hyperparameters, sub_exploration_level, &sub_problem);
            params_sub.decomp_nb_phases = 0;
            if params.sub_mu_permille != 1000 {
                params_sub.mu_start =
                    scale_permille(params_sub.mu_start, params.sub_mu_permille).max(1);
            }
            let it_permille = match params.sub_it_l4_permille {
                Some(value) if sub_exploration_level >= 4 => value,
                _ => params.sub_it_permille,
            };
            if it_permille != 1000 {
                params_sub.max_it_noimprov =
                    scale_permille(params_sub.max_it_noimprov, it_permille);
                params_sub.max_it_total = scale_permille(params_sub.max_it_total, it_permille);
            }
            let seed_sub_ind =
                Individual::new_from_routes(&sub_problem, &params_sub, seed_sub_routes.clone());
            debug_assert!(
                seed_sub_ind.load_excess == 0 && seed_sub_ind.tw_violation == 0,
                "Invariant violated: subproblem seed should always be feasible in reversed mode"
            );
            let seed_sub_route_count = seed_sub_routes.len();
            let migrants = if receive {
                let cap = carry_k.min(params_sub.mu_start.saturating_sub(MIN_FRESH_STARTS));
                restrict_carried(&carried, &sub_problem, &global_to_local, &seed_sub_routes, cap)
            } else {
                Vec::new()
            };
            let mut sub_ga = Genetic::new(sub_problem, params_sub);
            sub_ga.inherited = migrants;
            let chosen_sub_routes: Vec<Vec<usize>> =
                match sub_ga.run(&mut rng, None, Some(&seed_sub_routes)) {
                    Some(candidate) => {
                        if candidate.cost < seed_sub_ind.cost {
                            candidate.routes
                        } else {
                            seed_sub_routes
                        }
                    }
                    None => seed_sub_routes,
                };
            if export {
                let donors: Vec<Vec<Vec<usize>>> = sub_ga
                    .export_feasible(carry_k)
                    .iter()
                    .map(|routes| map_sub_routes_to_global(routes, &local_to_global))
                    .collect();
                blocks.push(BlockDonors {
                    donors,
                    fallback: route_group
                        .iter()
                        .map(|&rid| phase_base_routes[rid].clone())
                        .collect(),
                });
            }
            residual_free_routes +=
                (seed_sub_route_count as i32) - (chosen_sub_routes.len() as i32);
            let sub_routes_global = map_sub_routes_to_global(&chosen_sub_routes, &local_to_global);

            rebuilt_routes.extend(sub_routes_global);

            let progressive_master_routes = build_progressive_master_routes(
                &phase_base_routes,
                &clusters,
                &rebuilt_routes,
                cid + 1,
            );
            let progressive_master_ind = Individual::new_from_routes(
                data_arc.as_ref(),
                &params,
                progressive_master_routes.clone(),
            );
            promote_if_improved(
                save_solution,
                &mut incumbent,
                &progressive_master_routes,
                &progressive_master_ind,
            );
        }

        carried = if export {
            assemble_carried(&blocks, carry_k, data_arc.nb_nodes)
        } else {
            Vec::new()
        };

        let merged_ind =
            Individual::new_from_routes(data_arc.as_ref(), &params, rebuilt_routes.clone());
        debug_assert!(
            merged_ind.load_excess == 0 && merged_ind.tw_violation == 0,
            "Invariant violated: merged master solution should be feasible before global LS"
        );
        promote_if_improved(save_solution, &mut incumbent, &rebuilt_routes, &merged_ind);
        let mut candidate_routes = ls.run_from_routes(&rebuilt_routes, &[], params, &mut rng);
        let mut candidate_ind =
            Individual::new_from_routes(data_arc.as_ref(), &params, candidate_routes.clone());
        if candidate_ind.load_excess > 0 || candidate_ind.tw_violation > 0 {
            increase_penalties_for_infeasibility(&mut params, &candidate_ind);
            let mut repaired: Option<(Vec<Vec<usize>>, Individual)> = None;
            for factor in REPAIR_FACTORS {
                let routes = ls.continue_repair(&mut rng, params, factor);
                let ind = Individual::new_from_routes(data_arc.as_ref(), &params, routes.clone());
                if ind.load_excess == 0 && ind.tw_violation == 0 {
                    repaired = Some((routes, ind));
                    break;
                }
            }
            if let Some((routes, ind)) = repaired {
                if ind.cost <= merged_ind.cost {
                    candidate_routes = routes;
                    candidate_ind = ind;
                } else {
                    candidate_routes = rebuilt_routes;
                    candidate_ind = merged_ind;
                }
            } else {
                candidate_routes = rebuilt_routes;
                candidate_ind = merged_ind;
            }
        }

        promote_if_improved(
            save_solution,
            &mut incumbent,
            &candidate_routes,
            &candidate_ind,
        );
        phase_idx += 1;

        phases_in_window += 1;
        if phases_in_window >= 3 {
            let current_cost = incumbent.ind.cost;
            let denom = window_start_cost.max(1) as f64;
            let improvement_ratio = (window_start_cost - current_cost) as f64 / denom;
            if improvement_ratio < 0.01 {
                adaptive_target_size = ((adaptive_target_size * 4) / 5).max(ADAPT_MIN_TARGET);
            } else if improvement_ratio > 0.03 {
                adaptive_target_size = ((adaptive_target_size * 5) / 4).min(base_target_size);
            }
            window_start_cost = current_cost;
            phases_in_window = 0;
        }

        if ot_rounds < params.ot_max
            && phase_idx >= exploration_schedule.len()
            && params.stop_distance > 0
            && params.ot_cap > 0.0
            && (incumbent.ind.cost as i64) > params.stop_distance
            && (incumbent.ind.cost as i64) <= params.stop_distance + params.ot_window
        {
            let spent = fuel_start.saturating_sub(fuel_left());
            if (spent as f64) < (fuel_start as f64) * params.ot_cap {
                ot_rounds += 1;
                let deepest = *exploration_schedule.last().unwrap_or(&1);
                let extra = params.decomp_nb_phases.max(1);
                for _ in 0..extra {
                    exploration_schedule.push(deepest);
                }
            }
        }
    }

    Ok(Some(Solution { routes: incumbent.routes }))
}
}
mod sequence {
use super::params::Params;
use super::problem::Problem;
use std::cmp::{max, min};

#[derive(Copy, Clone, Debug, Default)]
pub struct Sequence {
    pub earliest_end: i32,
    pub tau_plus: i32,
    pub duration_net: i32,
    pub tw: i32,
    pub load: i32,
    pub distance: i32,
    pub first_node: u16,
    pub last_node: u16,
}

impl Sequence {
    #[inline(always)]
    pub fn earliest_start(&self) -> i32 {
        self.earliest_end - self.duration_net
    }

    #[inline(always)]
    pub fn duration(&self) -> i32 {
        self.duration_net + self.tw
    }

    #[inline(always)]
    pub fn initialize(&mut self, data: &Problem, node: usize) {
        debug_assert!(u16::try_from(node).is_ok());
        let nd = data.nd(node);
        self.earliest_end = nd.start_tw + nd.service_time;
        self.tau_plus = nd.end_tw;
        self.duration_net = nd.service_time;
        self.tw = 0;
        self.load = nd.demand;
        self.distance = 0;
        self.first_node = node as u16;
        self.last_node = node as u16;
    }

    #[inline(always)]
    pub fn join2(data: &Problem, s1: &Sequence, s2: &Sequence) -> Sequence {
        let travel = data.dm(s1.last_node as usize, s2.first_node as usize);
        Self::join2_with_travel(s1, s2, travel)
    }

    #[inline(always)]
    pub fn join2_with_travel(s1: &Sequence, s2: &Sequence, travel: i32) -> Sequence {
        let mut out = Self::join_tw_with_travel(s1, s2, travel);
        out.distance = s1.distance + s2.distance + travel;
        out.load = s1.load + s2.load;
        out
    }

    #[inline(always)]
    pub fn join_tw_with_travel(s1: &Sequence, s2: &Sequence, travel: i32) -> Sequence {
        let arrival = s1.earliest_end + travel;
        let extra = max(arrival - s2.tau_plus, 0);
        let shifted = s1.duration_net + travel;
        Sequence {
            earliest_end: max(s2.earliest_end, arrival + s2.duration_net) - extra,
            tau_plus: min(s2.tau_plus - shifted + extra, s1.tau_plus),
            duration_net: max(shifted + s2.duration_net, s2.earliest_end - s1.tau_plus) - extra,
            tw: s1.tw + s2.tw + extra,
            load: 0,
            distance: 0,
            first_node: s1.first_node,
            last_node: s2.last_node,
        }
    }

    #[inline(always)]
    pub fn singleton(data: &Problem, node: usize) -> Sequence {
        let mut s = Sequence::default();
        s.initialize(data, node);
        s
    }

    #[inline(always)]
    pub fn eval(&self, data: &Problem, params: &Params) -> i64 {
        let ptw = params.penalty_tw as i64;
        let pcap = params.penalty_capa as i64;
        let load_excess = (self.load - data.max_capacity).max(0) as i64;
        (self.distance as i64) + load_excess * pcap + (self.tw as i64) * ptw
    }

    #[inline(always)]
    pub fn tw2_with_travel(s1: &Sequence, s2: &Sequence, travel: i32) -> i32 {
        s1.tw + s2.tw + max(s1.earliest_end + travel - s2.tau_plus, 0)
    }

    #[inline(always)]
    pub fn tw3_with_travel(
        s1: &Sequence,
        s2: &Sequence,
        s3: &Sequence,
        travel12: i32,
        travel23: i32,
    ) -> i32 {
        let arrival = s1.earliest_end + travel12;
        let extra12 = max(arrival - s2.tau_plus, 0);
        let end12 = max(s2.earliest_end, arrival + s2.duration_net);
        s1.tw + s2.tw + s3.tw + max(extra12, end12 + travel23 - s3.tau_plus)
    }

    #[inline(always)]
    pub fn tw5_with_travel(
        s0: &Sequence,
        s1: &Sequence,
        s2: &Sequence,
        s3: &Sequence,
        s4: &Sequence,
        travel01: i32,
        travel12: i32,
        travel23: i32,
        travel34: i32,
    ) -> i32 {
        let arrival1 = s0.earliest_end + travel01;
        let extra1 = max(arrival1 - s1.tau_plus, 0);
        let end1 = max(s1.earliest_end, arrival1 + s1.duration_net) - extra1;

        let arrival2 = end1 + travel12;
        let extra2 = max(arrival2 - s2.tau_plus, 0);
        let end2 = max(s2.earliest_end, arrival2 + s2.duration_net) - extra2;

        let arrival3 = end2 + travel23;
        let extra3 = max(arrival3 - s3.tau_plus, 0);
        let end3 = max(s3.earliest_end, arrival3 + s3.duration_net) - extra3;

        s0.tw + s1.tw + s2.tw + s3.tw + s4.tw
            + extra1
            + extra2
            + extra3
            + max(end3 + travel34 - s4.tau_plus, 0)
    }

    #[inline(always)]
    pub fn eval3(
        data: &Problem,
        params: &Params,
        s1: &Sequence,
        s2: &Sequence,
        s3: &Sequence,
    ) -> i64 {
        let ptw = params.penalty_tw as i64;
        let pcap = params.penalty_capa as i64;
        let travel12 = data.dm(s1.last_node as usize, s2.first_node as usize);
        let travel23 = data.dm(s2.last_node as usize, s3.first_node as usize);
        let distance = s1.distance + s2.distance + s3.distance + travel12 + travel23;
        let tw_viol = Self::tw3_with_travel(s1, s2, s3, travel12, travel23);
        let load = s1.load + s2.load + s3.load;
        let load_excess = (load - data.max_capacity).max(0) as i64;
        (distance as i64) + load_excess * pcap + (tw_viol as i64) * ptw
    }
}
}
mod solver {
use super::genetic::Genetic;
use super::loader_tig::TigLoader;
use super::params::Params;
use super::problem::Problem;
use super::reverse_mode;
use anyhow::Result;
use rand::{rngs::SmallRng, SeedableRng};
use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;

pub struct Solver;

impl Solver {
    fn solve(
        data: Problem,
        params: Params,
        hyperparameters: &Option<Map<String, Value>>,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    ) -> Result<Option<Solution>> {
        if params.decomp_nb_phases > 0 {
            return reverse_mode::solve_reversed_mode(data, params, hyperparameters, save_solution);
        }

        let mut rng = SmallRng::from_seed(data.seed);
        let mut ga = Genetic::new(data, params);
        Ok(ga
            .run(&mut rng, save_solution, None)
            .map(|individual| Solution { routes: individual.routes }))
    }

    pub fn solve_challenge_instance(
        challenge: &Challenge,
        hyperparameters: &Option<Map<String, Value>>,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    ) -> Result<Option<Solution>> {
        let data = TigLoader::load(challenge);
        let params = Params::initialize(hyperparameters, &data);

        match Self::solve(data, params, hyperparameters, save_solution) {
            Ok(Some(solution)) => Ok(Some(solution)),
            Ok(None) => Ok(None),
            Err(_) => Ok(None),
        }
    }
}
}
pub use solver::Solver;
use tig_challenges::vehicle_routing::*;

#[allow(dead_code)]
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    match Solver::solve_challenge_instance(challenge, hyperparameters, Some(save_solution))? {
        Some(solution) => {
            let _ = save_solution(&solution);
            Ok(())
        }
        None => Err(anyhow!("No feasible solution found")),
    }
}

#[allow(dead_code)]
pub fn help() {
    println!("No help information available.");
}
