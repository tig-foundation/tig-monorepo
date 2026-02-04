use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rand::seq::SliceRandom;
use std::time::Instant;
use std::cmp::{max, min};
use std::collections::VecDeque;
pub struct VrpData {
    pub seed: [u8; 32],
    pub nb_nodes: usize,
    pub nb_vehicles: usize,
    pub lb_vehicles: usize,
    pub demands: Vec<i32>,
    pub max_capacity: i32,
    pub distance_matrix: Vec<Vec<i32>>,
    pub node_positions: Vec<(i32, i32)>,
    pub service_times: Vec<i32>,
    pub start_tw: Vec<i32>,
    pub end_tw: Vec<i32>
}

impl VrpData {
    #[inline(always)]
    pub fn dm(&self, i: usize, j: usize) -> i32 {
        unsafe { *self.distance_matrix.get_unchecked(i).get_unchecked(j) }
    }
}

// Backwards-compatible aliases: many functions expect the original `Problem`/`Params` names.
pub type Problem = VrpData;
pub type Params = Config;

/// Configuration parameters for the VRPTW solver
/// 
/// TIG v0.0.5+ Compatibility:
/// This solver is fully compatible with the new multi-track submission format.
/// The Config::preset() function automatically adapts parameters based on problem size (nb_nodes),
/// making it suitable for random track selection where the track is chosen at precommit time.
/// 
/// Use generate_all_track_settings() to create settings for all tracks in the submission format.
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Config {
    pub exploration_level: usize,
    pub allow_swap3: bool,
    pub granularity: usize,
    pub granularity2: usize,
    pub penalty_tw: usize,
    pub penalty_capa: usize,
    pub target_ratio: f64,
    pub max_it_noimprov: usize,
    pub max_it_total: usize,
    pub nb_it_adapt_penalties: usize,
    pub nb_it_traces: usize,
    pub mu: usize,
    pub mu_start: usize,
    pub lambda: usize,
    pub nb_close: usize,
    pub nb_elite: usize,
}

impl Config {

    fn preset(exploration_level: usize, nb_nodes: usize) -> Self {

        let p = if nb_nodes <= 700 { 20 }
        else if nb_nodes <= 1000 { 30 }
        else if nb_nodes <= 1200 { 50 }
        else if nb_nodes <= 1500 { 80 }
        else if nb_nodes <= 2000 { 150 }
        else if nb_nodes <= 3000 { 200 }
        else { 500 };

        match exploration_level {
            0 => Self {
                exploration_level: 0, allow_swap3: true,
                granularity: 40, granularity2: 20,
                penalty_tw: p,  penalty_capa: p, target_ratio: 0.2,
                max_it_noimprov: 0, max_it_total: 0,
                nb_it_adapt_penalties: 100, nb_it_traces: 100,
                mu: 2, mu_start: 1, lambda: 1, nb_close: 1, nb_elite: 1
            },
            1 => Self {
                exploration_level: 0, allow_swap3: true,
                granularity: 40, granularity2: 20,
                penalty_tw: p,  penalty_capa: p, target_ratio: 0.2,
                max_it_noimprov: 0, max_it_total: 0,
                nb_it_adapt_penalties: 100, nb_it_traces: 100,
                mu: 2, mu_start: 5, lambda: 1, nb_close: 1, nb_elite: 1
            },
            2 => Self {
                exploration_level: 1, allow_swap3: true,
                granularity: 40, granularity2: 20,
                penalty_tw: p,  penalty_capa: p, target_ratio: 0.2,
                max_it_noimprov: 10, max_it_total: 50,
                nb_it_adapt_penalties: 100, nb_it_traces: 100,
                mu: 3, mu_start: 6, lambda: 3, nb_close: 1, nb_elite: 1
            },
            3 => Self {
                exploration_level: 2, allow_swap3: true,
                granularity: 40, granularity2: 20,
                penalty_tw: p,  penalty_capa: p, target_ratio: 0.2,
                max_it_noimprov: 100, max_it_total: 500,
                nb_it_adapt_penalties: 20, nb_it_traces: 20,
                mu: 5, mu_start: 10, lambda: 5, nb_close: 2, nb_elite: 2
            },
            4 => Self {
                exploration_level: 3, allow_swap3: false,
                granularity: 30, granularity2: 20,
                penalty_tw: p,  penalty_capa: p, target_ratio: 0.2,
                max_it_noimprov: 500, max_it_total: 5_000,
                nb_it_adapt_penalties: 20, nb_it_traces: 100,
                mu: 10, mu_start: 20, lambda: 10, nb_close: 2, nb_elite: 3
            },
            5 => Self {
                exploration_level: 4, allow_swap3: false,
                granularity: 30, granularity2: 20,
                penalty_tw: p,  penalty_capa: p, target_ratio: 0.2,
                max_it_noimprov: 5_000, max_it_total: 50_000,
                nb_it_adapt_penalties: 50, nb_it_traces: 200,
                mu: 12, mu_start: 24, lambda: 20, nb_close: 3, nb_elite: 4
            },
            6 => Self {
                exploration_level: 5, allow_swap3: false,
                granularity: 30, granularity2: 20,
                penalty_tw: p,  penalty_capa: p, target_ratio: 0.2,
                max_it_noimprov: 10_000, max_it_total: 200_000,
                nb_it_adapt_penalties: 50, nb_it_traces: 500,
                mu: 25, mu_start: 50, lambda: 40, nb_close: 3, nb_elite: 8
            },
            _ => Self::defaults(nb_nodes),
        }
    }

    pub fn defaults(nb_nodes: usize) -> Self { Self::preset(0, nb_nodes) }

    pub fn initialize(hyperparameters: &Option<Map<String, Value>>, nb_nodes: usize) -> Self {

        let mut base_params = Self::defaults(nb_nodes);

        if let Some(v) = hyperparameters.as_ref().and_then(|m| m.get("exploration_level")) {
            match v {
                Value::Number(n) => {
                    if let Some(u) = n.as_u64() { base_params = Self::preset(u as usize, nb_nodes); }
                }
                Value::String(s) => {
                    if let Ok(u) = s.parse::<usize>() { base_params = Self::preset(u, nb_nodes); }
                }
                _ => {}
            }
        }

        let mut merged_params = serde_json::to_value(base_params).expect("Params serializable");
        if let (Value::Object(ref mut obj), Some(map)) = (&mut merged_params, hyperparameters) {
            for (k, v) in map {
                if k == "exploration_level" { continue; }
                obj.insert(k.clone(), v.clone());
            }
        }

        serde_json::from_value(merged_params).unwrap_or_else(|_| Self::defaults(nb_nodes))
    }
}

#[derive(Copy, Clone, Default)]
pub struct Sequence {
    pub tau_minus: i32,
    pub tau_plus: i32,
    pub tmin: i32,
    pub tw: i32,
    pub total_service_duration: i32,
    pub load: i32,
    pub distance: i32,
    pub first_node: usize,
    pub last_node: usize,
}

impl Sequence {

    #[inline(always)]
    pub fn initialize(&mut self, data: &VrpData, node: usize) {
        let st  = data.start_tw[node];
        let et  = data.end_tw[node];
        let svc = data.service_times[node];
        let ld  = data.demands[node];
        self.tau_minus = st;
        self.tau_plus  = et;
        self.tmin      = svc;
        self.tw        = 0;
        self.total_service_duration = svc ;
        self.load      = ld;
        self.distance  = 0;
        self.first_node = node;
        self.last_node  = node;
    }

    #[inline(always)]
    pub fn join2(data: &Problem, s1: &Sequence, s2: &Sequence) -> Sequence {
        let travel = data.dm(s1.last_node, s2.first_node);
        let distance = s1.distance + s2.distance + travel;
        let temp = travel + s1.tmin - s1.tw;

        let wtij      = max(s2.tau_minus - temp - s1.tau_plus, 0);
        let twij      = max(temp + s1.tau_minus - s2.tau_plus, 0);
        let tw        = s1.tw + s2.tw + twij;
        let tmin      = temp + s1.tw + s2.tmin + wtij;
        let tau_minus = max(s2.tau_minus - temp - wtij, s1.tau_minus);
        let tau_plus  = min(s2.tau_plus  - temp + twij, s1.tau_plus);
        let load      = s1.load + s2.load;

        Sequence {
            tau_minus, tau_plus, tmin, tw,
            total_service_duration: s1.total_service_duration + s2.total_service_duration,
            load, distance,
            first_node: s1.first_node,
            last_node:  s2.last_node,
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
        let ptw  = params.penalty_tw  as i64;
        let pcap = params.penalty_capa as i64;
        let load_excess = (self.load - data.max_capacity).max(0) as i64;
        (self.distance as i64) + load_excess * pcap + (self.tw as i64) * ptw
    }

    #[inline(always)]
    pub fn eval2(data: &Problem, params: &Params, s1: &Sequence, s2: &Sequence) -> i64 {
        let ptw  = params.penalty_tw  as i64;
        let pcap = params.penalty_capa as i64;
        let travel   = data.dm(s1.last_node, s2.first_node);
        let distance = s1.distance + s2.distance + travel;
        let temp     = s1.tmin - s1.tw + travel;
        let tw_viol  = s1.tw + s2.tw + max(s1.tau_minus - s2.tau_plus + temp, 0);
        let load     = s1.load + s2.load;
        let load_excess = (load - data.max_capacity).max(0) as i64;
        (distance as i64) + load_excess * pcap + (tw_viol as i64) * ptw
    }

    #[inline(always)]
    pub fn eval3(data: &Problem, params: &Params, s1: &Sequence, s2: &Sequence, s3: &Sequence) -> i64 {
        let ptw  = params.penalty_tw  as i64;
        let pcap = params.penalty_capa as i64;

        let travel12   = data.dm(s1.last_node, s2.first_node);
        let distance12 = s1.distance + s2.distance + travel12;
        let temp   = travel12 + s1.tmin - s1.tw;

        let wtij       = max(s2.tau_minus - temp - s1.tau_plus, 0);
        let twij       = max(temp + s1.tau_minus - s2.tau_plus, 0);
        let tw_viol12  = s1.tw + s2.tw + twij;
        let tmin12     = temp + s1.tw + s2.tmin + wtij;
        let tau_m12    = max(s2.tau_minus - temp - wtij, s1.tau_minus);

        let travel23   = data.dm(s2.last_node, s3.first_node);
        let distance   = distance12 + s3.distance + travel23;
        let temp2      = travel23 + tmin12 - tw_viol12;

        let tw_viol    = tw_viol12 + s3.tw + max(tau_m12 - s3.tau_plus + temp2, 0);
        let load       = s1.load + s2.load + s3.load;

        let load_excess = (load - data.max_capacity).max(0) as i64;
        (distance as i64) + load_excess * pcap + (tw_viol as i64) * ptw
    }

    #[inline(always)]
    pub fn eval_n(data: &Problem, params: &Params, chain: &[Sequence]) -> i64 {
        let mut agg = chain[0];
        for s in &chain[1..chain.len()-1] { agg = Sequence::join2(data, &agg, s); }
        let last = &chain[chain.len()-1];
        Sequence::eval2(data, params, &agg, last)
    }
}

#[derive(Clone)]
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

    pub fn evaluate_routes(data: &Problem, routes: &Vec<Vec<usize>>) -> (i32, i32, i32) {
        let mut dist: i32 = 0;
        let mut tw: i32 = 0;
        let mut loadx: i32 = 0;
        for r in routes {
            if r.is_empty() { continue; }
            let mut acc = Sequence::singleton(data, r[0]);
            for idx in 1..r.len() {
                let next = Sequence::singleton(data, r[idx]);
                acc = Sequence::join2(data, &acc, &next);
            }
            dist += acc.distance;
            tw += acc.tw;
            let ex = (acc.load - data.max_capacity).max(0);
            loadx += ex;
        }
        (dist, tw, loadx)
    }

    #[inline]
    pub fn compute_penalized_cost(distance: i32, tw_violation: i32, load_excess: i32, params: &Params) -> i64 {
        (distance as i64)
            + (params.penalty_tw as i64) * (tw_violation as i64)
            + (params.penalty_capa as i64) * (load_excess as i64)
    }

    #[inline]
    pub fn recompute_cost(&mut self, params: &Params) {
        self.cost = Self::compute_penalized_cost(self.distance, self.tw_violation, self.load_excess, params);
    }

    fn build_pred_succ_and_count(data: &Problem, routes: &Vec<Vec<usize>>) -> (Vec<usize>, Vec<usize>, usize) {
        let n_all = data.nb_nodes;
        let mut pred = vec![0usize; n_all];
        let mut succ = vec![0usize; n_all];
        let mut nb_routes: usize = 0;

        for r in routes {
            if r.len() > 2 { nb_routes += 1; }
            if r.len() < 2 { continue; } 
            for p in 1..r.len() - 1 {
                let id = r[p];
                pred[id] = r[p - 1];
                succ[id] = r[p + 1];
            }
        }
        (pred, succ, nb_routes)
    }
}

pub struct Constructive;

impl Constructive {
    pub fn build_routes(data: &Problem, rng: &mut SmallRng, randomize: bool) -> Vec<Vec<usize>> {
        let mut routes = Vec::new();
        let mut nodes: Vec<usize> = (1..data.nb_nodes).collect();
        let n = nodes.len();
        nodes.sort_by(|&a, &b| data.dm(0,a).cmp(&data.dm(0,b)));

        if randomize {
            let window = if n < 1000 { 10 } else { 5 };
            for i in 0..(n - 1) { 
                nodes.swap(i, rng.gen_range(i + 1..=(i + window).min(n - 1))); 
            }
        }

        let mut available = vec![true; data.nb_nodes];
        available[0] = false; 

        while let Some(node) = nodes.pop() {
            if !available[node] { continue; }
            available[node] = false;
            let mut route = vec![0, node, 0];
            let mut route_demand = data.demands[node];

            while let Some((best_node, best_pos)) =
                Self::find_best_insertion(&route, &nodes, &available, route_demand,data)
            {
                available[best_node] = false;
                route_demand += data.demands[best_node];
                route.insert(best_pos, best_node);
            }

            routes.push(route);
        }

        routes
    }

    fn find_best_insertion(
        route: &Vec<usize>,
        nodes: &Vec<usize>,
        available: &Vec<bool>,
        route_demand: i32,
        data: &Problem,
    ) -> Option<(usize, usize)> {
        let mut best_c2 = None;
        let mut best = None;
        for &insert_node in nodes.iter() {

            if !available[insert_node] || route_demand + data.demands[insert_node] > data.max_capacity {
                continue;
            }

            let mut curr_time = 0;
            let mut curr_node = 0;

            for pos in 1..route.len() {
                let next_node = route[pos];
                let new_arrival_time_insert_node = data.start_tw[insert_node]
                    .max(curr_time + data.dm(curr_node,insert_node));
                if new_arrival_time_insert_node > data.end_tw[insert_node] {
                    break;
                }

                let c11 = data.dm(curr_node,insert_node)
                    + data.dm(insert_node,next_node)
                    - data.dm(curr_node,next_node);

                let c2 = data.dm(0,insert_node) - c11;

                let c2_is_better = match best_c2 {
                    None => true,
                    Some(x) => c2 > x,
                };

                if c2_is_better
                    && Self::is_feasible(
                        route,
                        insert_node,
                        new_arrival_time_insert_node + data.service_times[insert_node],
                        pos,
                        data,
                    )
                {
                    best_c2 = Some(c2);
                    best = Some((insert_node, pos));
                }

                curr_time = data.start_tw[next_node]
                    .max(curr_time + data.dm(curr_node,next_node))
                    + data.service_times[next_node];
                curr_node = next_node;
            }
        }
        best
    }

    fn is_feasible(
        route: &Vec<usize>,
        mut curr_node: usize,
        mut curr_time: i32,
        start_pos: usize,
        data: &Problem,
    ) -> bool {
        for pos in start_pos..route.len() {
            let next_node = route[pos];
            curr_time += data.dm(curr_node,next_node);
            if curr_time > data.end_tw[route[pos]] {
                return false;
            }
            curr_time = curr_time.max(data.start_tw[next_node]) + data.service_times[next_node];
            curr_node = next_node;
        }
        true
    }
}

#[derive(Clone, Default)]
pub struct Node {
    id: usize,
    seq0_i: Sequence,
    seqi_n: Sequence,
    seq1: Sequence,
    seq12: Sequence,
    seq21: Sequence,
    seq123: Sequence,
}

impl Node {
    #[inline]
    fn new(id: usize) -> Self {
        Self { id, ..Default::default() }
    }
}

#[derive(Clone, Default)]
pub struct Route {
    cost: i64,
    distance: i32,
    load: i32,
    tw: i32,
    nodes: Vec<Node>,
    // per-route position lookup: pos[node_id] = index in `nodes` (or usize::MAX if not present)
    pub pos: Vec<usize>,
}

impl Route {
    #[inline]
    fn new(ids: &[usize], nb_nodes: usize) -> Self {
        let mut nodes: Vec<Node> = ids.iter().copied().map(Node::new).collect();
        // emulate smallvec by reserving a modest inline capacity for common cases
        nodes.reserve(16);
        let mut pos = vec![usize::MAX; nb_nodes];
        for (i, n) in nodes.iter().enumerate() {
            pos[n.id] = i;
        }
        Self {
            nodes,
            pos,
            ..Default::default()
        }
    }

    // Update `pos` mapping from current `nodes` vector
    fn rebuild_pos(&mut self) {
        if self.pos.is_empty() { return; }
        for v in self.pos.iter_mut() { *v = usize::MAX; }
        for (i, n) in self.nodes.iter().enumerate() { self.pos[n.id] = i; }
    }
}

pub struct LocalSearch<'a> {
    pub data: &'a Problem,
    // delta tables to accelerate move evaluations
    pub delta: DeltaTables,
    pub neighbors_before: Vec<Vec<usize>>,
    pub neighbors_capacity_swap: Vec<Vec<usize>>,
    pub params: Params,
    pub cost: i64,
    pub routes: Vec<Route>,
    pub node_route: Vec<usize>,
    pub node_pos: Vec<usize>,
    pub empty_routes: Vec<usize>,
    pub when_last_modified: Vec<usize>,
    pub when_last_tested: Vec<usize>,
    pub nb_moves: usize,
}

impl<'a> LocalSearch<'a> {
    pub fn new(data: &'a Problem, params: Params) -> Self {

        let n = data.nb_nodes;
        let cap = n.saturating_sub(2);
        let keep = min(params.granularity as usize, cap);
        let mut neighbors_before: Vec<Vec<usize>> = vec![Vec::new(); n];

        for i in 1..n {
            let mut prox: Vec<(i32, usize)> = Vec::with_capacity(cap);
            for j in 1..n {
                if j == i { continue; }
                let tji  = data.dm(j,i);
                let wait = (data.start_tw[i] - tji - data.service_times[j] - data.end_tw[j]).max(0);
                let late = (data.start_tw[j] + data.service_times[j] + tji - data.end_tw[i]).max(0);
                let proxy10 = 10 * tji + 2 * wait + 10 * late;
                prox.push((proxy10, j));
            }
            prox.sort_by_key(|&(p, _)| p);
            neighbors_before[i] = prox[..keep].iter().map(|&(_, j)| j).collect();
        }

        let mut neighbors_capacity_swap: Vec<Vec<usize>> = vec![Vec::new(); n];
        let diff_limit = max(4, data.max_capacity / 20);
        for i in 1..n {
            let di = data.demands[i];
            let mut prox: Vec<(i32, usize)> = Vec::with_capacity(n.saturating_sub(1));
            for j in 1..n {
                if j == i { continue; }
                if (data.demands[j] - di).abs() <= diff_limit {
                    let dij = data.dm(i,j);
                    prox.push((dij, j));
                }
            }
            prox.sort_by_key(|&(d, _)| d);
            let m = prox.len().min(params.granularity2 as usize);
            neighbors_capacity_swap[i] = prox[..m].iter().map(|&(_, j)| j).collect();
        }

        Self {
            data,
            delta: DeltaTables::new(data),
            neighbors_before,
            neighbors_capacity_swap,
            params,
            cost: 0,
            routes: Vec::new(),
            node_route: Vec::new(),
            node_pos: Vec::new(),
            empty_routes: Vec::new(),
            when_last_modified: Vec::new(),
            when_last_tested: vec![0; n],
            nb_moves: 0,
        }
    }

    fn load_from_routes(&mut self, routes: &Vec<Vec<usize>>) {
        let n = self.data.nb_nodes;
        let fleet = self.data.nb_vehicles;

        let mut src: Vec<Vec<usize>> = Vec::new();

        if routes.len() <= fleet {
            src.extend(routes.iter().cloned());
        } else {
            let keep = fleet.saturating_sub(1);
            src.extend(routes.iter().take(keep).cloned());
            let mut merged = routes[keep].clone();
            merged.pop();
            for r in routes.iter().skip(fleet) {
                if r.len() > 2 { merged.extend_from_slice(&r[1..r.len() - 1]); }
            }
            merged.push(0);
            src.push(merged);
        }

        while src.len() < fleet {
            src.push(vec![0, 0]);
        }

        let all_routes: Vec<Route> = src.iter().map(|r| Route::new(r, self.data.nb_nodes)).collect();
        self.node_route = vec![0; n];
        self.node_pos = vec![0; n];
        self.empty_routes.clear();
        self.routes = all_routes;

        self.when_last_modified = vec![0; self.routes.len()];
        self.when_last_tested = vec![0; n];
        self.nb_moves = 1;

        for rid in 0..self.routes.len() { self.update_route(rid); }
        self.cost = self.routes.iter().map(|r| r.cost).sum();
    }

    fn write_back_to_routes(&self, out: &mut Vec<Vec<usize>>) {
        out.clear();
        out.extend(
            self.routes
                .iter()
                .filter(|r| r.nodes.len() > 2)
                .map(|r| r.nodes.iter().map(|n| n.id).collect::<Vec<usize>>())
        );
    }

    fn update_route(&mut self, rid: usize) {
        let data = self.data;
        let r = &mut self.routes[rid];
        let len = r.nodes.len();

        let mut acc_fwd = Sequence::singleton(data, r.nodes[0].id);
        r.nodes[0].seq0_i = acc_fwd;
        for pos in 1..len {
            let id = r.nodes[pos].id;
            acc_fwd = Sequence::join2(data, &acc_fwd, &Sequence::singleton(data, id));
            r.nodes[pos].seq0_i = acc_fwd;
        }

        let mut acc_bwd = Sequence::singleton(data, r.nodes[len - 1].id);
        r.nodes[len - 1].seqi_n = acc_bwd;
        for pos in (0..len - 1).rev() {
            let id = r.nodes[pos].id;
            acc_bwd = Sequence::join2(data, &Sequence::singleton(data, id), &acc_bwd);
            r.nodes[pos].seqi_n = acc_bwd;
        }

        for pos in 0..len {
            let id = r.nodes[pos].id;
            r.nodes[pos].seq1 = Sequence::singleton(data, id);
            if pos + 1 < len {
                let id_next = r.nodes[pos + 1].id;
                r.nodes[pos].seq12 = Sequence::join2(data,
                                                     &Sequence::singleton(data, id),
                                                     &Sequence::singleton(data, id_next));
                r.nodes[pos].seq21 = Sequence::join2(data,
                                                     &Sequence::singleton(data, id_next),
                                                     &Sequence::singleton(data, id));
                if pos + 2 < len {
                    let id_next2 = r.nodes[pos + 2].id;
                    r.nodes[pos].seq123 = Sequence::join2(data,
                                                          &r.nodes[pos].seq12,
                                                          &Sequence::singleton(data, id_next2));
                }
            }
        }

        let end = r.nodes[len - 1].seq0_i;
        r.load = end.load;
        r.tw   = end.tw;
        r.distance = end.distance;
        r.cost = end.eval(data, &self.params);

        for (pos, node) in r.nodes.iter().enumerate() {
            self.node_route[node.id] = rid;
            self.node_pos[node.id] = pos;
        }

        // update per-route position lookup
        r.rebuild_pos();

        let is_empty = self.routes[rid].nodes.len() == 2;
        let pos = self.empty_routes.iter().position(|&eid| eid == rid);
        match (is_empty, pos) {
            (true, None) => self.empty_routes.push(rid),
            (false, Some(i)) => { self.empty_routes.swap_remove(i); }
            _ => {}
        }
        self.when_last_modified[rid] = self.nb_moves;
    }

    pub fn run_intra_route_relocate(&mut self, r1: usize, pos1: usize) -> bool {
        let route = &self.routes[r1];
        let len = route.nodes.len();
        if len < pos1 + 4 { return false; }

        let mut left_excl: Vec<Sequence> = vec![Sequence::default(); len];
        let mut acc_left = route.nodes[0].seq0_i;
        for p in 1..len {
            left_excl[p] = acc_left;
            if p != pos1 { acc_left = Sequence::join2(self.data, &acc_left, &route.nodes[p].seq1); }
        }

        let mut right_excl: Vec<Sequence> = vec![Sequence::default(); len];
        let mut acc_right = route.nodes[len-1].seq1;
        right_excl[len-1] = acc_right;
        for p in (1..len - 1).rev() {
            if p != pos1 { acc_right = Sequence::join2(self.data, &route.nodes[p].seq1, &acc_right); }
            right_excl[p] = acc_right;
        }

        let old_cost = route.cost;
        let mut best_cost = old_cost;
        let mut best_pos: Option<usize> = None;

        for t in 1..len {
            if t == pos1 || t == pos1 + 1 { continue; }
            let new_cost = Sequence::eval3(self.data, &self.params, &left_excl[t], &route.nodes[pos1].seq1, &right_excl[t]);
            if new_cost < best_cost {
                best_cost = new_cost;
                best_pos = Some(t);
            }
        }

        if let Some(mypos) = best_pos {
            let insert_pos = if mypos > pos1 { mypos - 1 } else { mypos };
            let elem = self.routes[r1].nodes.remove(pos1);
            self.routes[r1].nodes.insert(insert_pos, elem);
            self.nb_moves += 1;
            self.update_route(r1);
            self.cost += self.routes[r1].cost - old_cost;
            true
        } else { false }
    }

    pub fn run_intra_route_swap_right(&mut self, r1: usize, pos1: usize) -> bool {
        let route = &self.routes[r1];
        let len   = route.nodes.len();
        if len < pos1 + 4 { return false; }

        let old_cost = route.cost;
        let mut best_cost = old_cost;
        let mut best_pos: Option<usize> = None;

        let mut acc_mid = route.nodes[pos1 + 1].seq1;
        for pos2 in (pos1 + 2)..(len - 1) {
            let new_cost = Sequence::eval_n(self.data, &self.params,
                                            &[route.nodes[pos1 - 1].seq0_i,
                                                route.nodes[pos2].seq1,
                                                acc_mid,
                                                route.nodes[pos1].seq1,
                                                route.nodes[pos2 + 1].seqi_n]);
            if new_cost < best_cost {
                best_cost = new_cost;
                best_pos = Some(pos2);
            }
            acc_mid = Sequence::join2(self.data, &acc_mid, &route.nodes[pos2].seq1);
        }

        if let Some(mypos) = best_pos {
            self.routes[r1].nodes.swap(pos1, mypos);
            self.nb_moves += 1;
            self.update_route(r1);
            self.cost += self.routes[r1].cost - old_cost;
            true
        } else { false }
    }

    pub fn run_2optstar(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize) -> bool {
        let route1 = &self.routes[r1];
        let route2 = &self.routes[r2];

        let new1 = Sequence::eval2(self.data, &self.params, &route1.nodes[pos1 - 1].seq0_i, &route2.nodes[pos2].seqi_n);
        let new2 = Sequence::eval2(self.data, &self.params, &route2.nodes[pos2 - 1].seq0_i, &route1.nodes[pos1].seqi_n);

        let old_cost = route1.cost + route2.cost;
        let new_cost = new1 + new2;

        if new_cost < old_cost {
            let mut suffix1 = self.routes[r1].nodes.split_off(pos1);
            let mut suffix2 = self.routes[r2].nodes.split_off(pos2);
            self.routes[r1].nodes.append(&mut suffix2);
            self.routes[r2].nodes.append(&mut suffix1);
            self.nb_moves += 1;
            self.update_route(r1);
            self.update_route(r2);
            self.cost += new_cost - old_cost;
            true
        } else { false }
    }

    pub fn run_2opt(&mut self, r1: usize, pos1: usize) -> bool {
        let route = &self.routes[r1];
        let len = route.nodes.len();
        if len < pos1 + 3 { return false; }

        let old_cost = route.cost;
        let mut best_cost = old_cost;
        let mut best_pos: Option<usize> = None;

        let mut mid_rev = route.nodes[pos1].seq21;
        for pos2 in (pos1 + 1)..(len - 1) {
            let new_cost = Sequence::eval3(self.data, &self.params, &route.nodes[pos1 - 1].seq0_i, &mid_rev, &route.nodes[pos2 + 1].seqi_n);
            if new_cost < best_cost {
                best_cost = new_cost;
                best_pos = Some(pos2);
            }
            if pos2 + 1 < len - 1 { mid_rev = Sequence::join2(self.data, &route.nodes[pos2 + 1].seq1, &mid_rev); }
        }

        if let Some(mypos) = best_pos {
            self.routes[r1].nodes[pos1..=mypos].reverse();
            self.nb_moves += 1;
            self.update_route(r1);
            self.cost += self.routes[r1].cost - old_cost;
            true
        } else { false }
    }

    pub fn run_inter_route(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize) -> bool {
        let data = self.data;
        let ru = &self.routes[r1];
        let rv = &self.routes[r2];
        let u = &ru.nodes[pos1];
        let v = &rv.nodes[pos2];
        let u_pred = &ru.nodes[pos1 - 1];
        let v_pred = &rv.nodes[pos2 - 1];
        let x = &ru.nodes[pos1 + 1];

        let old_total = ru.cost + rv.cost;
        let mut best_i = 0usize;
        let mut best_j = 0usize;
        let mut best_cost = old_total;

        let mut update_best = |i: usize, j: usize, cand: i64| {
            if cand < best_cost {
                best_cost = cand;
                best_i = i;
                best_j = j;
            }
        };

        let result10 = Sequence::eval2(data, &self.params, &u_pred.seq0_i, &x.seqi_n)
            + Sequence::eval3(data, &self.params, &v_pred.seq0_i, &u.seq1, &v.seqi_n);
        update_best(1,0,result10);

        if v.id != 0 {
            let result11 = Sequence::eval3(data, &self.params, &u_pred.seq0_i, &v.seq1, &x.seqi_n)
                + Sequence::eval3(data, &self.params, &v_pred.seq0_i, &u.seq1, &rv.nodes[pos2 + 1].seqi_n);
            update_best(1,1,result11);
        }

        if x.id != 0 {
            let x_next = &ru.nodes[pos1 + 2];
            let mut result20 = Sequence::eval2(data, &self.params, &u_pred.seq0_i, &x_next.seqi_n);
            let mut result30 = result20;
            result20 += Sequence::eval3(data, &self.params, &v_pred.seq0_i, &u.seq12, &v.seqi_n);
            result30 += Sequence::eval3(data, &self.params, &v_pred.seq0_i, &u.seq21, &v.seqi_n);
            update_best(2,0,result20);
            update_best(3,0,result30);

            if v.id != 0 {
                let y = &rv.nodes[pos2 + 1];
                let mut result21 = Sequence::eval3(data, &self.params, &u_pred.seq0_i, &v.seq1, &x_next.seqi_n);
                let mut result31 = result21;
                result21 += Sequence::eval3(data, &self.params, &v_pred.seq0_i, &u.seq12, &y.seqi_n);
                result31 += Sequence::eval3(data, &self.params, &v_pred.seq0_i, &u.seq21, &y.seqi_n);
                update_best(2,1,result21);
                update_best(3,1,result31);

                if y.id != 0 {
                    let mut result22 = Sequence::eval3(data, &self.params, &u_pred.seq0_i, &v.seq12, &x_next.seqi_n);
                    let mut result23 = Sequence::eval3(data, &self.params, &u_pred.seq0_i, &v.seq21, &x_next.seqi_n);
                    let mut result32 = result22;
                    let mut result33 = result23;

                    let y_next = &rv.nodes[pos2 + 2];
                    let tmp  = Sequence::eval3(data, &self.params, &v_pred.seq0_i, &u.seq12, &y_next.seqi_n);
                    let tmp2 = Sequence::eval3(data, &self.params, &v_pred.seq0_i, &u.seq21, &y_next.seqi_n);
                    result22 += tmp;  result23 += tmp;
                    result32 += tmp2; result33 += tmp2;
                    update_best(2,2,result22);
                    update_best(3,2,result32);
                    update_best(2,3,result23);
                    update_best(3,3,result33);
                }
            }

            if x_next.id != 0 && self.params.allow_swap3 {
                let x2_next = &ru.nodes[pos1 + 3];
                let result40 = Sequence::eval2(data, &self.params, &u_pred.seq0_i, &x2_next.seqi_n)
                    + Sequence::eval3(data, &self.params, &v_pred.seq0_i, &u.seq123, &v.seqi_n);
                update_best(4,0,result40);

                if v.id != 0 {
                    let y = &rv.nodes[pos2 + 1];
                    let result41 = Sequence::eval3(data, &self.params, &u_pred.seq0_i, &v.seq1, &x2_next.seqi_n)
                        + Sequence::eval3(data, &self.params, &v_pred.seq0_i, &u.seq123, &y.seqi_n);
                    update_best(4,1,result41);

                    if y.id != 0 {
                        let y_next = &rv.nodes[pos2 + 2];
                        let result42 = Sequence::eval3(data, &self.params, &u_pred.seq0_i, &v.seq12, &x2_next.seqi_n)
                            + Sequence::eval3(data, &self.params, &v_pred.seq0_i, &u.seq123, &y_next.seqi_n);
                        let result43 = Sequence::eval3(data, &self.params, &u_pred.seq0_i, &v.seq21, &x2_next.seqi_n)
                            + Sequence::eval3(data, &self.params, &v_pred.seq0_i, &u.seq123, &y_next.seqi_n);
                        update_best(4,2,result42);
                        update_best(4,3,result43);

                        if y_next.id != 0 {
                            let y2_next = &rv.nodes[pos2 + 3];
                            let result44 = Sequence::eval3(data, &self.params, &u_pred.seq0_i, &v.seq123, &x2_next.seqi_n)
                                + Sequence::eval3(data, &self.params, &v_pred.seq0_i, &u.seq123, &y2_next.seqi_n);
                            update_best(4,4,result44);
                        }
                    }
                }
            }
        }

        if best_i == 0 && best_j == 0 { return false; }

        let mut take_block = |route_idx: usize, pos: usize, kind: usize| -> Vec<Node> {
            let nodes = &mut self.routes[route_idx].nodes;
            match kind {
                0 => vec![],
                1 => { let n1 = nodes.remove(pos); vec![n1] }
                2 => { let n1 = nodes.remove(pos); let n2 = nodes.remove(pos); vec![n1, n2] }
                3 => { let n1 = nodes.remove(pos); let n2 = nodes.remove(pos); vec![n2, n1] }
                4 => {
                    let n1 = nodes.remove(pos);
                    let n2 = nodes.remove(pos);
                    let n3 = nodes.remove(pos);
                    vec![n1, n2, n3]
                }
                _ => vec![],
            }
        };

        let blk_from_r1 = take_block(r1, pos1, best_i);
        let blk_from_r2 = take_block(r2, pos2, best_j);

        let nodes1 = &mut self.routes[r1].nodes;
        for (k, node) in blk_from_r2.into_iter().enumerate() { nodes1.insert(pos1 + k, node); }
        let nodes2 = &mut self.routes[r2].nodes;
        for (k, node) in blk_from_r1.into_iter().enumerate() { nodes2.insert(pos2 + k, node); }

        self.nb_moves += 1;
        self.update_route(r1);
        self.update_route(r2);

        let new_total = self.routes[r1].cost + self.routes[r2].cost;
        self.cost += new_total - old_total;
        true
    }

    pub fn run_swapstar(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize) -> bool {
        let route1_len = self.routes[r1].nodes.len();
        let route2_len = self.routes[r2].nodes.len();
        let u = self.routes[r1].nodes[pos1].id;
        let v = self.routes[r2].nodes[pos2].id;
        let (pu, nu) = (self.routes[r1].nodes[pos1 - 1].id, self.routes[r1].nodes[pos1 + 1].id);
        let (pv, nv) = (self.routes[r2].nodes[pos2 - 1].id, self.routes[r2].nodes[pos2 + 1].id);

        let dr1 = self.data.dm(pu, nu) - self.data.dm(pu, u) - self.data.dm(u, nu);
        let dr2 = self.data.dm(pv, nv) - self.data.dm(pv, v) - self.data.dm(v, nv);
        let delta_demand = self.data.demands[v] - self.data.demands[u] ;
        let new_load1 = self.routes[r1].load + delta_demand;
        let new_load2 = self.routes[r2].load - delta_demand;
        let new_pen1 = ((new_load1 - self.data.max_capacity).max(0) as i64) * self.params.penalty_capa as i64;
        let new_pen2 = ((new_load2 - self.data.max_capacity).max(0) as i64) * self.params.penalty_capa as i64;
        let cost_lb_r1_after_removal = (self.routes[r1].distance + dr1) as i64 + new_pen1 ;
        let cost_lb_r2_after_removal = (self.routes[r2].distance + dr2) as i64 + new_pen2 ;
        let mut lb_new_total = cost_lb_r1_after_removal + cost_lb_r2_after_removal;
        let old_total = self.routes[r1].cost + self.routes[r2].cost;
        if lb_new_total > old_total { return false; }

        let hole_v = self.data.dm(pu, v) + self.data.dm(v, nu) - self.data.dm(pu, nu);
        let mut best_ins_v = hole_v;
        for t in 1..route1_len {
            let a_id = self.routes[r1].nodes[t - 1].id;
            let b_id = self.routes[r1].nodes[t].id;
            if a_id == u || b_id == u { continue; }
            let delta = self.data.dm(a_id, v) + self.data.dm(v, b_id) - self.data.dm(a_id, b_id);
            if delta < best_ins_v { best_ins_v = delta; }
        }

        let hole_u = self.data.dm(pv, u) + self.data.dm(u, nv) - self.data.dm(pv, nv);
        let mut best_ins_u = hole_u;
        for t in 1..route2_len {
            let a_id = self.routes[r2].nodes[t - 1].id;
            let b_id = self.routes[r2].nodes[t].id;
            if a_id == v || b_id == v { continue; }
            let delta = self.data.dm(a_id, u) + self.data.dm(u, b_id) - self.data.dm(a_id, b_id);
            if delta < best_ins_u { best_ins_u = delta; }
        }

        lb_new_total += (best_ins_v + best_ins_u) as i64;
        if lb_new_total > old_total { return false; }

        let mut left_excl1: Vec<Sequence> = vec![Sequence::default(); route1_len];
        let mut right_excl1: Vec<Sequence> = vec![Sequence::default(); route1_len];
        {
            let r = &self.routes[r1];
            let mut acc_left = r.nodes[0].seq0_i;
            for p in 1..route1_len {
                left_excl1[p] = acc_left;
                if p != pos1 { acc_left = Sequence::join2(self.data, &acc_left, &r.nodes[p].seq1); }
            }
            let mut acc_right = r.nodes[route1_len - 1].seq1;
            right_excl1[route1_len - 1] = acc_right;
            for p in (1..route1_len - 1).rev() {
                if p != pos1 { acc_right = Sequence::join2(self.data, &r.nodes[p].seq1, &acc_right); }
                right_excl1[p] = acc_right;
            }
        }

        let mut left_excl2: Vec<Sequence> = vec![Sequence::default(); route2_len];
        let mut right_excl2: Vec<Sequence> = vec![Sequence::default(); route2_len];
        {
            let r = &self.routes[r2];
            let mut acc_left = r.nodes[0].seq0_i;
            for p in 1..route2_len {
                left_excl2[p] = acc_left;
                if p != pos2 { acc_left = Sequence::join2(self.data, &acc_left, &r.nodes[p].seq1); }
            }
            let mut acc_right = r.nodes[route2_len - 1].seq1;
            right_excl2[route2_len - 1] = acc_right;
            for p in (1..route2_len - 1).rev() {
                if p != pos2 { acc_right = Sequence::join2(self.data, &r.nodes[p].seq1, &acc_right); }
                right_excl2[p] = acc_right;
            }
        }

        let v_seq1 = self.routes[r2].nodes[pos2].seq1;
        let mut best_cost1 = i64::MAX / 4;
        let mut best_t1: usize = 1;
        for t in 1..route1_len {
            let cand = Sequence::eval3(self.data, &self.params, &left_excl1[t], &v_seq1, &right_excl1[t]);
            if cand < best_cost1 { best_cost1 = cand; best_t1 = t; }
        }

        let u_seq1 = self.routes[r1].nodes[pos1].seq1;
        let mut best_cost2 = i64::MAX / 4;
        let mut best_t2: usize = 1;
        for t in 1..route2_len {
            let cand = Sequence::eval3(self.data, &self.params, &left_excl2[t], &u_seq1, &right_excl2[t]);
            if cand < best_cost2 { best_cost2 = cand; best_t2 = t; }
        }

        if best_cost1 + best_cost2 >= old_total { return false; }

        let node_u = self.routes[r1].nodes[pos1].clone();
        let node_v = self.routes[r2].nodes[pos2].clone();

        self.routes[r1].nodes.remove(pos1);
        self.routes[r2].nodes.remove(pos2);

        let ins1 = if best_t1 > pos1 { best_t1 - 1 } else { best_t1 };
        let ins2 = if best_t2 > pos2 { best_t2 - 1 } else { best_t2 };

        self.routes[r1].nodes.insert(ins1, node_v);
        self.routes[r2].nodes.insert(ins2, node_u);

        self.nb_moves += 1;
        self.update_route(r1);
        self.update_route(r2);
        let new_total = self.routes[r1].cost + self.routes[r2].cost;
        self.cost += new_total - old_total;
        true
    }

    pub fn runls(&mut self, routes: &mut Vec<Vec<usize>>, rng: &mut SmallRng, params: Params, is_repair: bool, factor: usize) {
        self.params = params;
        if !is_repair {
            self.load_from_routes(routes);
        }
        else {
            self.params.penalty_tw   = (factor * self.params.penalty_tw).min(10_000);
            self.params.penalty_capa = (factor * self.params.penalty_capa).min(10_000);
            self.nb_moves += 1;
            for rid in 0..self.routes.len() {
                let r = &self.routes[rid];
                if r.load > self.data.max_capacity || r.tw > 0 {
                    self.update_route(rid);
                }
            }
        }

        let mut improved = true;
        let mut loop_id = 0;
        let mut c1_order: Vec<usize> = (1..self.data.nb_nodes).collect();
        while improved {
            improved = false;
            loop_id += 1;
            c1_order.shuffle(rng);
            for &c1 in &c1_order {
                let last_tested = self.when_last_tested[c1];
                self.when_last_tested[c1] = self.nb_moves;
                let r1 = self.node_route[c1];
                let pos1 = self.node_pos[c1];

                let neigh_len = self.neighbors_before[c1].len();
                let start = rng.gen_range(0..neigh_len);
                for off in 0..neigh_len {
                    let c2 = self.neighbors_before[c1][(start + off) % neigh_len];
                    let r2 = self.node_route[c2];
                    let pos2 = self.node_pos[c2];
                    if r1 == r2 { continue; }

                    if self.when_last_modified[r1].max(self.when_last_modified[r2]) <= last_tested {
                        continue;
                    }

                    if self.run_inter_route(r1, pos1, r2, pos2+1) {
                        improved = true;
                        break;
                    }

                    if pos1 == 1 && self.run_inter_route(r2, pos2, r1, pos1) {
                        improved = true;
                        break;
                    }

                    if self.run_2optstar(r1, pos1, r2, pos2+1) {
                        improved = true;
                        break;
                    }
                }

                let r1 = self.node_route[c1];
                let pos1 = self.node_pos[c1];
                let swap_len = self.neighbors_capacity_swap[c1].len();
                if swap_len > 0 {
                    let start_s = rng.gen_range(0..swap_len);
                    for off in 0..swap_len {
                        let c2 = self.neighbors_capacity_swap[c1][(start_s + off) % swap_len];
                        let r2 = self.node_route[c2];
                        if r1 == r2 { continue; }

                        if c1 < c2 || self.when_last_modified[r1].max(self.when_last_modified[r2]) <= last_tested {
                            continue;
                        }

                        let pos2 = self.node_pos[c2];
                        if self.run_swapstar(r1, pos1, r2, pos2) {
                            improved = true;
                            break;
                        }
                    }
                }

                let r1 = self.node_route[c1];
                let pos1 = self.node_pos[c1];
                if loop_id > 1 && (loop_id == 2 || self.when_last_modified[r1] > last_tested) {
                    if let Some(&r2) = self.empty_routes.first() {
                        let pos2 = 1;

                        if self.run_2optstar(r1, pos1, r2, pos2) {
                            improved = true;
                            break;
                        }

                        if self.run_inter_route(r1, pos1, r2, pos2) {
                            improved = true;
                            break;
                        }
                    }
                }

                let r1 = self.node_route[c1];
                if self.when_last_modified[r1] > last_tested {
                    improved |= self.run_intra_route_relocate(self.node_route[c1], self.node_pos[c1]);
                    improved |= self.run_intra_route_swap_right(self.node_route[c1], self.node_pos[c1]);
                    improved |= self.run_2opt(self.node_route[c1], self.node_pos[c1]);
                }
            }
        }
        self.write_back_to_routes(routes);
    }
}

// Lightweight delta tables (keeps travel times; future deltas can be added)
pub struct DeltaTables {
    pub travel: Vec<Vec<i32>>,
}

impl DeltaTables {
    pub fn new(data: &Problem) -> Self {
        // clone distance matrix into contiguous structure for faster access
        Self { travel: data.distance_matrix.clone() }
    }

    #[inline(always)]
    pub fn travel(&self, i: usize, j: usize) -> i32 {
        unsafe { *self.travel.get_unchecked(i).get_unchecked(j) }
    }
}

#[derive(Default)]
pub struct Subpopulation {
    pub indivs: Vec<Individual>,
    pub prox: Vec<Vec<(f64, usize)>>,
    pub biased_fitness: Vec<f64>,
    pub order_cost: Vec<usize>,
}

pub struct Population<'a> {
    pub data: &'a Problem,
    pub feasible: Subpopulation,
    pub infeasible: Subpopulation,
    cap_window: VecDeque<bool>,
    tw_window: VecDeque<bool>,
    since_last_adapt: usize,
}

impl<'a> Population<'a> {
    pub fn new(data: &'a Problem) -> Self {
        Self {
            data,
            feasible: Subpopulation::default(),
            infeasible: Subpopulation::default(),
            cap_window: VecDeque::new(),
            tw_window: VecDeque::new(),
            since_last_adapt: 0,
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
        let sub = if is_feasible {
            &mut self.feasible
        } else {
            &mut self.infeasible
        };

        let new_idx = sub.indivs.len();
        sub.indivs.push(ind);
        Self::prox_add(sub, self.data, new_idx);

        Self::order_cost_rebuild(sub);

        Self::update_biased_fitnesses(sub, params);

        if sub.indivs.len() > params.mu + params.lambda {
            Self::survivors_selection(sub, params);
        }
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

            if frac_cap < params.target_ratio {
                params.penalty_capa =
                    (((params.penalty_capa as f64) * 1.3).ceil()).clamp(1.0, 10_000.0) as usize;
            } else {
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

    pub fn best_metric(&self) -> BestMetric {
        if !self.feasible.indivs.is_empty() {
            let mut best_d = i32::MAX;
            for ind in &self.feasible.indivs {
                if ind.distance < best_d {
                    best_d = ind.distance;
                }
            }
            return BestMetric {
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
        BestMetric {
            feasible: false,
            distance: best_dist,
            infeas_sum: best_sum,
        }
    }

    pub fn print_trace(
        &self,
        _it_total: usize,
        _it_no_improve: usize,
        _elapsed_sec: f64,
        _params: &Params,
    ) {
    }

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

    fn prox_add(sub: &mut Subpopulation, data: &Problem, new_idx: usize) {
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

    fn broken_pairs_distance(data: &Problem, indiv_a: &Individual, indiv_b: &Individual) -> f64 {
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
pub struct BestMetric {
    pub feasible: bool,
    pub distance: i32,
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

pub struct Genetic<'a> {
    pub data: &'a Problem,
    pub params: Params,
    pub population: Population<'a>,
}

impl<'a> Genetic<'a> {
    pub fn new(data: &'a Problem, params: Params) -> Self {
        let population = Population::new(data);
        Self { data, params, population }
    }

    fn repair_and_maybe_add(&mut self, ls: &mut LocalSearch, rng: &mut SmallRng) {
        let mut repaired_routes1: Vec<Vec<usize>> = Vec::new();
        ls.runls(&mut repaired_routes1, rng, self.params, true, 100);
        let repaired1 = Individual::new_from_routes(self.data, &self.params, repaired_routes1);

        if repaired1.load_excess == 0 && repaired1.tw_violation == 0 {
            self.population.add(repaired1, &self.params);
        }
    }

    pub fn generate_initial_individual(&mut self, rng: &mut SmallRng, ls: &mut LocalSearch, randomize: bool) {
        let mut routes: Vec<Vec<usize>> = Constructive::build_routes(self.data, rng, randomize);
        ls.runls(&mut routes, rng, self.params, false,0);
        let ind = Individual::new_from_routes(self.data, &self.params, routes);
        let is_capa_feasible = ind.load_excess == 0;
        let is_tw_feasible = ind.tw_violation == 0;

        self.population.add(ind, &self.params);
        self.population.record_and_adapt(is_capa_feasible, is_tw_feasible, &mut self.params);
        if !is_capa_feasible || !is_tw_feasible { self.repair_and_maybe_add(ls, rng); }
    }

    pub fn generate_crossover_individual(&mut self, rng: &mut SmallRng, ls: &mut LocalSearch) {
        // Build a small micro-population from the best feasible individuals (if available)
        let micro_k = 8usize;
        let mut micro: Vec<Individual> = if !self.population.feasible.indivs.is_empty() {
            let mut tmp = self.population.feasible.indivs.clone();
            tmp.sort_by_key(|i| i.cost);
            tmp.truncate(micro_k);
            tmp
        } else {
            // fallback to mixing feasible+infeasible
            let mut tmp = self.population.feasible.indivs.clone();
            tmp.extend(self.population.infeasible.indivs.clone());
            tmp.sort_by_key(|i| i.cost);
            tmp.truncate(micro_k);
            tmp
        };

        let (p1, p2) = if micro.len() >= 2 {
            let i1 = rng.gen_range(0..micro.len());
            let mut i2 = rng.gen_range(0..micro.len());
            while i2 == i1 { i2 = rng.gen_range(0..micro.len()); }
            (micro[i1].clone(), micro[i2].clone())
        } else {
            let p1_ref = self.population.get_binary_tournament(rng).clone();
            let mut p2_ref = self.population.get_binary_tournament(rng).clone();
            while p1_ref.pred == p2_ref.pred && p1_ref.succ == p2_ref.succ {
                p2_ref = self.population.get_binary_tournament(rng).clone();
            }
            (p1_ref, p2_ref)
        };
        let t1 = self.extract_giant_tour(&p1.routes);
        let t2 = self.extract_giant_tour(&p2.routes);
        let extra = if rng.gen_ratio(1, 10) { 1 } else { 0 }; 
        let target_routes = (p1.nb_routes + extra).clamp(self.data.lb_vehicles, self.data.nb_vehicles);

        // With some probability use arc-preserving crossover on the micro-population
        let use_arc = rng.gen_ratio(3, 10);
        let mut child_tour = if use_arc {
            self.arc_preserving_crossover(&t1, &t2, rng)
        } else {
            self.crossover_ox(&t1, &t2, rng)
        };
        self.mutate_tour(&mut child_tour, rng);

        let mut child_routes = self.split(&child_tour, target_routes);
        ls.runls(&mut child_routes, rng, self.params, false, 0);
        let child = Individual::new_from_routes(self.data, &self.params, child_routes);
        let is_capa_feasible = child.load_excess == 0;
        let is_tw_feasible = child.tw_violation == 0;

        self.population.add(child, &self.params);
        self.population.record_and_adapt(is_capa_feasible, is_tw_feasible, &mut self.params);
        if !is_capa_feasible || !is_tw_feasible { self.repair_and_maybe_add(ls, rng); }
    }

    pub fn run(
        &mut self,
        rng: &mut SmallRng,
        t0: &Instant,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    ) -> Option<(Vec<Vec<usize>>,i32)> {
        if let Some(save) = save_solution {
            let dummy_routes: Vec<Vec<usize>> = (1..self.data.nb_nodes)
                .map(|i| vec![0, i, 0])
                .collect();
            let _ = save(&Solution { routes: dummy_routes });
        }
        
        let mut ls = LocalSearch::new(self.data, self.params);

        let diversity_boost = if self.data.nb_nodes < 1000 { 3 } else { 1 };
        for it in 0..(self.params.mu_start + diversity_boost) {
            self.generate_initial_individual(rng, &mut ls, it > 0);
        }

        let mut best_metric: BestMetric = self.population.best_metric();
        let mut it_noimprov: usize = 0;
        let mut it_total: usize = 0;
        while it_noimprov < self.params.max_it_noimprov && it_total < self.params.max_it_total {

            self.generate_crossover_individual(rng, &mut ls);

            if it_total % self.params.nb_it_traces == 0 {
                self.population.print_trace(
                    it_total, it_noimprov, t0.elapsed().as_secs_f64(), &self.params );
            }

            let cur = self.population.best_metric();
            if cur.better_than(best_metric) {
                best_metric = cur;
                it_noimprov = 0;

                if let Some(best) = self.population.best_feasible()  {
                    if let Some(save) = save_solution {
                        let _ = save(&Solution {routes: best.routes});
                    }
                }
            }
            else { it_noimprov += 1; }
            it_total += 1;
        }

        if let Some(best) = self.population.best_feasible() {
            let mut best_routes = best.routes.clone();
            ls.runls(&mut best_routes, rng, self.params, false, 0);
            let best_after = Individual::new_from_routes(self.data, &self.params, best_routes);
            let chosen = if best_after.tw_violation == 0
                && best_after.load_excess == 0
                && best_after.distance < best.distance
            {
                best_after
            } else {
                best
            };
            if let Some(save) = save_solution {
                let _ = save(&Solution { routes: chosen.routes.clone() });
            }
            Some((chosen.routes, chosen.cost as i32))
        } else {
            None
        }
    }

    fn mutate_tour(&self, tour: &mut Vec<usize>, rng: &mut SmallRng) {
        let n = tour.len();
        if n < 4 { return; }
        if rng.gen_ratio(1, 5) {
            let i = rng.gen_range(0..n - 1);
            let j = rng.gen_range(i + 1..n);
            tour[i..=j].reverse();
        }
        if rng.gen_ratio(1, 6) {
            let i = rng.gen_range(0..n);
            let mut j = rng.gen_range(0..n);
            if j == i {
                j = (j + 1) % n;
            }
            let node = tour.remove(i);
            let pos = if j <= tour.len() { j } else { tour.len() };
            tour.insert(pos, node);
        }
        if n >= 8 && rng.gen_ratio(1, 8) {
            let mut cuts = [0usize; 4];
            for c in cuts.iter_mut() {
                *c = rng.gen_range(1..n);
            }
            cuts.sort_unstable();
            let a = cuts[0];
            let b = cuts[1];
            let c = cuts[2];
            let d = cuts[3];
            let mut new_tour = Vec::with_capacity(n);
            new_tour.extend_from_slice(&tour[0..a]);
            new_tour.extend_from_slice(&tour[c..d]);
            new_tour.extend_from_slice(&tour[b..c]);
            new_tour.extend_from_slice(&tour[a..b]);
            new_tour.extend_from_slice(&tour[d..n]);
            *tour = new_tour;
        }
    }

    fn arc_preserving_crossover(&self, t1: &Vec<usize>, t2: &Vec<usize>, rng: &mut SmallRng) -> Vec<usize> {
        let n = t1.len();
        if n == 0 { return Vec::new(); }

        // build successor maps
        let mut succ1 = vec![usize::MAX; self.data.nb_nodes];
        let mut succ2 = vec![usize::MAX; self.data.nb_nodes];
        for i in 0..n {
            let a = t1[i];
            let b = t1[(i + 1) % n];
            succ1[a] = b;
            let c = t2[i];
            let d = t2[(i + 1) % n];
            succ2[c] = d;
        }

        let mut used = vec![false; self.data.nb_nodes];
        let mut child: Vec<usize> = Vec::with_capacity(n);
        // start from a random node
        let mut cur = t1[rng.gen_range(0..n)];
        child.push(cur);
        used[cur] = true;

        while child.len() < n {
            let cand1 = succ1[cur];
            let cand2 = succ2[cur];
            let mut next = usize::MAX;
            if cand1 != usize::MAX && !used[cand1] && cand1 == cand2 { next = cand1; }
            else if cand1 != usize::MAX && !used[cand1] && cand2 != usize::MAX && !used[cand2] {
                // choose the closer successor by travel time
                let d1 = self.data.dm(cur, cand1) as i64;
                let d2 = self.data.dm(cur, cand2) as i64;
                next = if d1 <= d2 { cand1 } else { cand2 };
            } else if cand1 != usize::MAX && !used[cand1] { next = cand1; }
            else if cand2 != usize::MAX && !used[cand2] { next = cand2; }
            else {
                // pick nearest unused neighbor (from data.dm) among a small randomized sample
                let mut best = usize::MAX;
                let mut bestd = i64::MAX;
                // sample a few indices randomly to avoid O(n^2)
                for _ in 0..8 {
                    let cand = t1[rng.gen_range(0..n)];
                    if used[cand] { continue; }
                    let d = self.data.dm(cur, cand) as i64;
                    if d < bestd { bestd = d; best = cand; }
                }
                if best != usize::MAX { next = best; }
                else {
                    // full scan fallback
                    for &v in t1.iter() {
                        if !used[v] {
                            let d = self.data.dm(cur, v) as i64;
                            if d < bestd { bestd = d; best = v; }
                        }
                    }
                    next = best;
                }
            }

            // safety fallback
            if next == usize::MAX {
                for &v in t1.iter() { if !used[v] { next = v; break; } }
            }

            child.push(next);
            used[next] = true;
            cur = next;
        }
        child
    }

    pub fn split(&self, giant: &Vec<usize>, target_routes: usize) -> Vec<Vec<usize>> {
        let n = giant.len();
        if n == 0 { return Vec::new(); }

        let k = target_routes.max(1);
        let inf = i64::MAX / 4;

        let mut dp = vec![vec![inf; n + 1]; k + 1];
        let mut pred = vec![vec![0usize; n + 1]; k + 1];
        dp[0][0] = 0;

        let factor_split: f32 = 1.5;
        let cap_limit: i32 = (factor_split * (self.data.max_capacity as f32)) as i32;
        let depot = Sequence::singleton(self.data, 0);

        for kk in 1..=k {
            for i in (kk - 1)..n {
                let base = dp[kk - 1][i];
                if base >= inf { continue; }

                let mut acc = Sequence::join2(self.data, &depot, &Sequence::singleton(self.data, giant[i]));
                for j in (i + 1)..=n {
                    let cost = Sequence::eval2(self.data, &self.params, &acc, &depot);
                    let cand = base + cost;
                    if cand < dp[kk][j] {
                        dp[kk][j] = cand;
                        pred[kk][j] = i;
                    }
                    if acc.load > cap_limit { break; }
                    if j < n {
                        let next = Sequence::singleton(self.data, giant[j]);
                        acc = Sequence::join2(self.data, &acc, &next);
                    }
                }
            }
        }

        let mut best_k = 1usize;
        let mut best_val = dp[1][n];
        for kk in 2..=k {
            let val = dp[kk][n];
            if val < best_val {
                best_val = val;
                best_k = kk;
            }
        }

        if best_val >= inf {
            let mut routes: Vec<Vec<usize>> = Vec::with_capacity(n);
            for &id in giant {
                routes.push(vec![0, id, 0]);
            }
            return routes;
        }

        let mut routes: Vec<Vec<usize>> = Vec::with_capacity(best_k);
        let mut j = n;
        for kk in (1..=best_k).rev() {
            let i = pred[kk][j];
            let mut r: Vec<usize> = Vec::with_capacity((j - i) + 2);
            r.push(0);
            for p in i..j {
                r.push(giant[p]);
            }
            r.push(0);
            routes.push(r);
            j = i;
        }
        routes.reverse();
        routes
    }

    pub fn extract_giant_tour(&self, routes: &[Vec<usize>]) -> Vec<usize> {
        let (x0, y0) = (self.data.node_positions[0].0 as f64, self.data.node_positions[0].1 as f64);
        let mut route_angles: Vec<(f64, usize)> = Vec::new();

        for (r_idx, r) in routes.iter().enumerate() {
            if r.len() <= 2 { continue; }
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut cnt = 0usize;
            for &id in r.iter().skip(1).take(r.len().saturating_sub(2)) {
                sum_x += self.data.node_positions[id].0 as f64;
                sum_y += self.data.node_positions[id].1 as f64;
                cnt += 1;
            }
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
        tour
    }

    pub fn crossover_ox(&self, parent1: &Vec<usize>, parent2: &Vec<usize>, rng: &mut SmallRng) -> Vec<usize> {
        let n = self.data.nb_nodes - 1;

        let mut child = vec![0usize; n];
        let mut used = vec![false; self.data.nb_nodes];

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
            j += 1;
        }

        let mut pos = stop;
        for t in 0..n {
            let v = parent2[(stop + t) % n];
            if !used[v] {
                child[pos] = v;
                used[v] = true;
                pos = (pos + 1) % n;
            }
        }
        child
    }
}

pub struct TigLoader;

impl TigLoader {
    pub fn load(challenge: &Challenge) -> Problem {
        let nb_nodes = challenge.num_nodes;
        let nb_vehicles = challenge.fleet_size;

        let mut service_times = vec![challenge.service_time; nb_nodes];
        service_times[0] = 0;

        let total_demand: f64 = challenge.demands.iter().map(|&d| d as f64).sum();
        let ratio = total_demand / challenge.max_capacity as f64;
        let lb_vehicles = ratio.ceil() as usize;

        Problem {
            seed: challenge.seed,
            nb_nodes,
            nb_vehicles,
            lb_vehicles,
            demands: challenge.demands.clone(),
            node_positions: challenge.node_positions.clone(),
            max_capacity: challenge.max_capacity,
            distance_matrix: challenge.distance_matrix.clone(),
            service_times,
            start_tw: challenge.ready_times.clone(),
            end_tw: challenge.due_times.clone(),
        }
    }
}

pub struct Solver;

impl Solver {
    fn solve(
        data: Problem,
        params: Params,
        t0: &Instant,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    ) -> Result<Option<(Solution, i32, usize)>> {
        let mut rng = SmallRng::from_seed(data.seed);
        let mut ga = Genetic::new(&data, params);
        Ok(ga.run(&mut rng, t0, save_solution).map(|(routes, cost)| {
            (Solution { routes: routes.clone() }, cost, routes.len())
        }))
    }

    pub fn solve_challenge_instance(
        challenge: &Challenge,
        hyperparameters: &Option<Map<String, Value>>,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    ) -> Result<Option<Solution>> {
        let t0 = Instant::now();
        let data = TigLoader::load(&challenge);
        let params = Params::initialize(hyperparameters,data.nb_nodes);
        match Self::solve(data, params, &t0, save_solution) {
            Ok(Some((solution, _cost, _routes))) => Ok(Some(solution)),
            Ok(None) => Ok(None),
            Err(_) => Ok(None)
        }
    }
}

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
        None => Ok(()),
    }
}

pub fn help() {
    println!("Hybrid Genetic Algorithm with Adaptive Local Search for VRPTW");
    println!("");
    println!("RECOMMENDED SETTINGS:");
    println!("");
    println!("For best quality:      {{\"exploration_level\": 3}}");
    println!("For balanced quality:  {{\"exploration_level\": 1}}");
    println!("For fastest runtime:   {{\"exploration_level\": 0}} or null");
    println!("");
    println!("EXPLORATION LEVELS (0-6):");
    println!("  0: Minimal iterations, fastest (~40s total)");
    println!("  1: More initial diversity, slightly slower");
    println!("  2: Light exploration (50 iterations)");
    println!("  3: Balanced (500 iterations, recommended)");
    println!("  4: Deep search (5,000 iterations)");
    println!("  5: Very deep (50,000 iterations)");
    println!("  6: Maximum quality (200,000 iterations)");    
}

/// Generate hyperparameters for a specific track (problem size)
/// 
/// This function returns optimal hyperparameters for the given track based on
/// the number of nodes in the problem instance.
pub fn generate_track_hyperparameters(n_nodes: usize, exploration_level: usize) -> Map<String, Value> {
    let mut hyperparameters = Map::new();
    hyperparameters.insert("exploration_level".to_string(), Value::Number(exploration_level.into()));
    
    // The Config::preset function already handles adaptive parameter tuning based on nb_nodes
    // We just expose the exploration_level here, which is the main tuning parameter
    
    hyperparameters
}

/// Generate track settings for all common vehicle_routing tracks for TIG v0.0.5+
/// 
/// This prepares a map of track_id -> (hyperparameters, num_bundles) for the new
/// submission format where all tracks must be specified upfront before random selection.
/// 
/// # Arguments
/// * `base_exploration_level` - The exploration level to use (0-6), will be adapted per track
/// * `base_num_bundles` - Base number of bundles, will be scaled for larger tracks
/// 
/// # Returns
/// A HashMap mapping track_id to a tuple of (hyperparameters, num_bundles)
pub fn generate_all_track_settings(
    base_exploration_level: usize,
    base_num_bundles: u32,
) -> std::collections::HashMap<String, (Map<String, Value>, u32)> {
    use std::collections::HashMap;
    
    // Common vehicle_routing track sizes
    // These track IDs should match what's configured in the TIG protocol
    let track_configs = vec![
        ("vrp_100", 100),
        ("vrp_200", 200),
        ("vrp_300", 300),
        ("vrp_400", 400),
        ("vrp_500", 500),
        ("vrp_700", 700),
        ("vrp_1000", 1000),
    ];
    
    let mut track_settings = HashMap::new();
    
    for (track_id, n_nodes) in track_configs {
        let hyperparameters = generate_track_hyperparameters(n_nodes, base_exploration_level);
        
        // Scale num_bundles based on problem size
        // Larger problems may benefit from more bundles for better coverage
        let num_bundles = if n_nodes >= 700 {
            (base_num_bundles as f64 * 1.5).ceil() as u32
        } else {
            base_num_bundles
        };
        
        track_settings.insert(track_id.to_string(), (hyperparameters, num_bundles));
    }
    
    track_settings
}

/// Helper function to print track settings in a format suitable for submission
/// 
/// This is useful for debugging and understanding what settings will be used for each track
pub fn print_track_settings(track_settings: &std::collections::HashMap<String, (Map<String, Value>, u32)>) {
    println!("Track Settings for TIG v0.0.5+ Submission:");
    println!("==========================================");
    
    let mut tracks: Vec<_> = track_settings.keys().collect();
    tracks.sort();
    
    for track_id in tracks {
        if let Some((hyperparameters, num_bundles)) = track_settings.get(track_id) {
            println!("\nTrack: {}", track_id);
            println!("  Hyperparameters: {}", serde_json::to_string(hyperparameters).unwrap_or_default());
            println!("  Num Bundles: {}", num_bundles);
        }
    }
    println!("\n==========================================");
}

/// Example usage for TIG v0.0.5+ submission preparation
/// 
/// This demonstrates how to prepare settings for all tracks as required by the new API
pub fn example_v0_0_5_submission() {
    // Generate settings for all tracks with exploration_level=1 and 10 bundles base
    let track_settings = generate_all_track_settings(1, 10);
    
    print_track_settings(&track_settings);
    
    // In actual submission code, you would pass track_settings to submit_precommit
    // The system will randomly select one track and use its corresponding settings
}