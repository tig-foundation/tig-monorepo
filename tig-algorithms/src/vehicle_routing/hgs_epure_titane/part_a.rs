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
            root_nb_nodes: data.root_nb_nodes,
            nb_nodes,
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

            let skip =
                !copy_at(available, insert_node) | (route_demand + insert_nd.demand > max_capacity);
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
#[path = "part_a_genetic.rs"]
mod genetic;
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
    pub adj: Vec<u32>,
    pub adjacency_fingerprint: u64,
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
        let (pred, succ, adj, nb_routes) = Self::build_pred_succ_and_count(data, &routes);
        let adjacency_fingerprint = if data.root_nb_nodes == 900 {
            let mut fingerprint = 0xcbf29ce484222325u64;
            for client in 1..pred.len() {
                for value in [client, pred[client], succ[client]] {
                    fingerprint = (fingerprint ^ value as u64).wrapping_mul(0x100000001b3);
                }
            }
            fingerprint
        } else {
            0
        };
        Self {
            routes,
            nb_routes,
            distance,
            tw_violation,
            load_excess,
            cost,
            pred,
            succ,
            adj,
            adjacency_fingerprint,
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
    ) -> (Vec<usize>, Vec<usize>, Vec<u32>, usize) {
        let nb_nodes = data.nb_nodes;
        debug_assert!(nb_nodes < 65536);
        let mut pred = vec![0usize; nb_nodes];
        let mut succ = vec![0usize; nb_nodes];
        let mut adj = vec![0u32; nb_nodes];
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
                    *adj.get_unchecked_mut(client) =
                        window[0] as u32 | (window[2] as u32) << 16;
                }
            }
        }
        for client in 1..nb_nodes {
            debug_assert!(seen[client], "Client {} is missing from routes", client);
        }
        debug_assert!((0..nb_nodes).all(|c| {
            adj[c] == (pred[c] as u32 | (succ[c] as u32) << 16)
        }));
        (pred, succ, adj, nb_routes)
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
            root_nb_nodes: challenge.num_nodes,
            nb_nodes,
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
#[path = "part_a_local_search.rs"]
mod local_search;
#[path = "part_a_params.rs"]
mod params;
#[path = "part_a_population.rs"]
mod population;
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
        unsafe { (*self.prev.get_unchecked(id), *self.next.get_unchecked(id)) }
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
    pub root_nb_nodes: usize,
    pub nb_nodes: usize,
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
pub(super) mod quality {
use tig_challenges::vehicle_routing::{Challenge, Solution};
use anyhow::{anyhow, Result};
use serde_json::{Map, Value};

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let mut routes = Vec::new();

    let mut nodes: Vec<usize> = (1..challenge.num_nodes).collect();
    nodes.sort_by(|&a, &b| challenge.distance_matrix[0][a].cmp(&challenge.distance_matrix[0][b]));

    let mut remaining: Vec<bool> = vec![true; challenge.num_nodes];
    remaining[0] = false;

    while let Some(node) = nodes.pop() {
        if !remaining[node] {
            continue;
        }
        remaining[node] = false;
        let mut route = vec![0, node, 0];
        let mut route_demand = challenge.demands[node];

        while let Some((best_node, best_pos)) = find_best_insertion(
            &route,
            remaining
                .iter()
                .enumerate()
                .filter(|(n, &flag)| {
                    flag && route_demand + challenge.demands[*n] <= challenge.max_capacity
                })
                .map(|(n, _)| n)
                .collect(),
            &challenge.distance_matrix,
            challenge.service_time,
            &challenge.ready_times,
            &challenge.due_times,
        ) {
            remaining[best_node] = false;
            route_demand += challenge.demands[best_node];
            route.insert(best_pos, best_node);
        }

        routes.push(route);
    }

    let _ = save_solution(&Solution { routes });
    Ok(())
}

fn is_feasible(
    route: &Vec<usize>,
    distance_matrix: &Vec<Vec<i32>>,
    service_time: i32,
    ready_times: &Vec<i32>,
    due_times: &Vec<i32>,
    mut curr_node: usize,
    mut curr_time: i32,
    start_pos: usize,
) -> bool {
    let mut valid = true;
    for pos in start_pos..route.len() {
        let next_node = route[pos];
        curr_time += distance_matrix[curr_node][next_node];
        if curr_time > due_times[route[pos]] {
            valid = false;
            break;
        }
        curr_time = curr_time.max(ready_times[next_node]) + service_time;
        curr_node = next_node;
    }
    valid
}

fn find_best_insertion(
    route: &Vec<usize>,
    remaining_nodes: Vec<usize>,
    distance_matrix: &Vec<Vec<i32>>,
    service_time: i32,
    ready_times: &Vec<i32>,
    due_times: &Vec<i32>,
) -> Option<(usize, usize)> {
    let alpha1 = 1;
    let alpha2 = 0;
    let lambda = 1;

    let mut best_c2 = None;
    let mut best = None;
    for insert_node in remaining_nodes {
        let mut best_c1 = None;

        let mut curr_time = 0;
        let mut curr_node = 0;
        for pos in 1..route.len() {
            let next_node = route[pos];
            let new_arrival_time =
                ready_times[insert_node].max(curr_time + distance_matrix[curr_node][insert_node]);
            if new_arrival_time > due_times[insert_node] {
                continue;
            }
            let old_arrival_time =
                ready_times[next_node].max(curr_time + distance_matrix[curr_node][next_node]);

            let c11 = distance_matrix[curr_node][insert_node]
                + distance_matrix[insert_node][next_node]
                - distance_matrix[curr_node][next_node];

            let c12 = new_arrival_time - old_arrival_time;

            let c1 = -(alpha1 * c11 + alpha2 * c12);
            let c2 = lambda * distance_matrix[0][insert_node] + c1;

            if best_c1.is_none_or(|x| c1 > x)
                && best_c2.is_none_or(|x| c2 > x)
                && is_feasible(
                    route,
                    distance_matrix,
                    service_time,
                    ready_times,
                    due_times,
                    insert_node,
                    new_arrival_time + service_time,
                    pos,
                )
            {
                best_c1 = Some(c1);
                best_c2 = Some(c2);
                best = Some((insert_node, pos));
            }

            curr_time = ready_times[next_node]
                .max(curr_time + distance_matrix[curr_node][next_node])
                + service_time;
            curr_node = next_node;
        }
    }
    best
}

pub fn evaluate_total_distance(challenge: &Challenge, solution: &Solution) -> Result<i32> {
    if solution.routes.len() > challenge.fleet_size {
        return Err(anyhow!(
            "Number of routes ({}) exceeds fleet size ({})",
            solution.routes.len(),
            challenge.fleet_size
        ));
    }
    let mut total_distance = 0;
    let mut visited = vec![false; challenge.num_nodes];
    visited[0] = true;

    for route in &solution.routes {
        if route.len() <= 2 || route[0] != 0 || route[route.len() - 1] != 0 {
            return Err(anyhow!("Each route must start and end at node 0 (the depot), and visit at least one non-depot node"));
        }

        let mut capacity = challenge.max_capacity;
        let mut current_node = 0;
        let mut curr_time = 0;
        for &node in &route[1..route.len() - 1] {
            if visited[node] {
                return Err(anyhow!(
                    "The same non-depot node cannot be visited more than once"
                ));
            }
            if challenge.demands[node] > capacity {
                return Err(anyhow!(
                    "The total demand on each route must not exceed max capacity"
                ));
            }
            curr_time += challenge.distance_matrix[current_node][node];
            if curr_time > challenge.due_times[node] {
                return Err(anyhow!("Node must be visited before due time"));
            }
            if curr_time < challenge.ready_times[node] {
                curr_time = challenge.ready_times[node];
            }
            curr_time += challenge.service_time;
            visited[node] = true;
            capacity -= challenge.demands[node];
            total_distance += challenge.distance_matrix[current_node][node];
            current_node = node;
        }

        curr_time += challenge.distance_matrix[current_node][0];
        if curr_time > challenge.due_times[0] {
            return Err(anyhow!("Must return to depot before due time"));
        }
        total_distance += challenge.distance_matrix[current_node][0];
    }

    if visited.iter().any(|&v| !v) {
        return Err(anyhow!("All nodes must be visited"));
    }

    Ok(total_distance)
}

pub fn greedy_total_distance(challenge: &Challenge) -> Result<u32> {
    let solution = std::cell::RefCell::new(Solution::new());
    let save_solution_fn = |s: &Solution| -> Result<()> {
        *solution.borrow_mut() = s.clone();
        Ok(())
    };
    solve_challenge(challenge, &save_solution_fn, &None)?;
    Ok(evaluate_total_distance(challenge, &solution.into_inner())? as u32)
}

pub fn quality_of(greedy_total_distance: u32, total_distance: i32) -> i32 {
    const QUALITY_PRECISION: i32 = 1_000_000;
    let quality =
        (greedy_total_distance as f64 - total_distance as f64) / greedy_total_distance as f64;
    let quality = quality.clamp(-10.0, 10.0) * QUALITY_PRECISION as f64;
    let quality = quality.round() as i32;
    quality
}
}
#[path = "part_a_reverse_mode.rs"]
mod reverse_mode;
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

        s0.tw
            + s1.tw
            + s2.tw
            + s3.tw
            + s4.tw
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
use super::params::{effective_hyperparameters, Params};
use super::problem::Problem;
use super::reverse_mode;
use anyhow::Result;
use rand::{rngs::SmallRng, SeedableRng};
use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;

pub struct Solver;

impl Solver {
    fn solve(
        challenge: &Challenge,
        data: Problem,
        params: Params,
        hyperparameters: &Option<Map<String, Value>>,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    ) -> Result<Option<Solution>> {
        if params.decomp_nb_phases > 0 {
            return reverse_mode::solve_reversed_mode(
                challenge,
                data,
                params,
                hyperparameters,
                save_solution,
                None,
                None,
            );
        }

        let mut rng = SmallRng::from_seed(data.seed);
        let mut ga = Genetic::new(data, params);
        Ok(ga
            .run(&mut rng, save_solution, None)
            .map(|individual| Solution {
                routes: individual.routes,
            }))
    }

    pub fn solve_from_seed(
        challenge: &Challenge,
        data: Problem,
        params: Params,
        hyperparameters: &Option<Map<String, Value>>,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
        seed: Solution,
        greedy: Option<u32>,
    ) -> Result<Option<Solution>> {
        if params.decomp_nb_phases > 0 {
            return reverse_mode::solve_reversed_mode(
                challenge,
                data,
                params,
                hyperparameters,
                save_solution,
                Some(seed.routes),
                greedy,
            );
        }

        let mut rng = SmallRng::from_seed(data.seed);
        let mut ga = Genetic::new(data, params);
        Ok(ga
            .run(&mut rng, save_solution, Some(&seed.routes))
            .map(|individual| Solution {
                routes: individual.routes,
            }))
    }

    pub fn solve_challenge_instance(
        challenge: &Challenge,
        hyperparameters: &Option<Map<String, Value>>,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    ) -> Result<Option<Solution>> {
        let data = TigLoader::load(challenge);
        let hyperparameters = effective_hyperparameters(data.nb_nodes, hyperparameters);
        let params = Params::initialize(&hyperparameters, &data);

        match Self::solve(challenge, data, params, &hyperparameters, save_solution) {
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
    let baseline = std::cell::Cell::new(None);
    let save_with_quality = |solution: &Solution| -> Result<()> {
        let result = save_solution(solution);
        let greedy_distance = match baseline.get() {
            Some(distance) => distance,
            None => {
                let distance = quality::greedy_total_distance(challenge)?;
                baseline.set(Some(distance));
                distance
            }
        };
        let distance = challenge.evaluate_total_distance(solution)?;
        println!("DIAG quality={}", quality::quality_of(greedy_distance, distance));
        result
    };
    match Solver::solve_challenge_instance(challenge, hyperparameters, Some(&save_with_quality))? {
        Some(solution) => {
            let _ = save_with_quality(&solution);
            Ok(())
        }
        None => Err(anyhow!("No feasible solution found")),
    }
}

pub fn solve_band_from_seed(
    challenge: &Challenge,
    seed: Solution,
    greedy: u32,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    if challenge.num_nodes != 700 {
        return Ok(());
    }

    let has_overrides = hyperparameters.as_ref().map_or(false, |map| !map.is_empty());
    let hyperparameters = params::effective_hyperparameters(700, hyperparameters);
    let window = ["band_bar", "band_low", "band_high", "post_low", "post_high"]
        .map(|key| hyperparameters.as_ref()?.get(key)?.as_i64());
    let mut initialized = None;
    let (band_bar, band_low, band_high, post_low, post_high) = match window {
        [Some(bar), Some(low), Some(high), Some(post_low), Some(post_high)]
            if !has_overrides => (bar, low, high, post_low, post_high),
        _ => {
            let data = loader_tig::TigLoader::load(challenge);
            let params = params::Params::initialize(&hyperparameters, &data);
            let window = (
                params.band_bar, params.band_low, params.band_high,
                params.post_low, params.post_high,
            );
            initialized = Some((data, params));
            window
        }
    };
    let distance = challenge.evaluate_total_distance(&seed)?;
    let quality = quality::quality_of(greedy, distance) as i64;
    let lower = band_bar.saturating_sub(band_low.max(post_low));
    let upper = band_bar.saturating_add(band_high.max(post_high));
    if quality < lower || quality >= upper {
        return Ok(());
    }
    let (data, params) = initialized.unwrap_or_else(|| {
        let data = loader_tig::TigLoader::load(challenge);
        let params = params::Params::initialize(&hyperparameters, &data);
        (data, params)
    });

    let save_with_quality = |solution: &Solution| -> Result<()> {
        let result = save_solution(solution);
        let distance = challenge.evaluate_total_distance(solution)?;
        println!("DIAG quality={}", quality::quality_of(greedy, distance));
        result
    };
    match Solver::solve_from_seed(
        challenge,
        data,
        params,
        &hyperparameters,
        Some(&save_with_quality),
        seed,
        Some(greedy),
    )? {
        Some(solution) => {
            let _ = save_with_quality(&solution);
            Ok(())
        }
        None => Err(anyhow!("No feasible solution found")),
    }
}

#[allow(dead_code)]
pub fn help() {
    println!("No help information available.");
}
