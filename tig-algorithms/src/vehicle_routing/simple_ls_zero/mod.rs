/*
REFERENCES AND ACKNOWLEDGMENTS

This implementation is based on or inspired by existing work.
This is a Rust re-implementation of the local search component of the HGS algorithm,
adapted to the VRP with time windows:

Vidal, T., Crainic, T. G., Gendreau, M., Lahrichi, N., & Rei, W. (2012).
A hybrid genetic algorithm for multidepot and periodic vehicle routing problems.
Operations Research, 60(3), 611–624.
https://doi.org/10.1287/opre.1120.1048

Vidal, T. (2022).
Hybrid genetic search for the CVRP: Open-source implementation and SWAP* neighborhood.
Computers and Operations Research, 140, 105643.
https://doi.org/10.1016/j.cor.2021.105643

*/

use serde_json::{Map, Value};
use std::cmp::{max, min};
use std::collections::BTreeSet;
use std::mem::replace;
use std::mem::take;
use tig_challenges::vehicle_routing::*;

pub struct MoveArgs {
    pub route1: usize,
    pub pos1: usize,
    pub route2: usize,
    pub pos2: usize,
    pub profit: isize,
}

impl MoveArgs {
    pub fn new(route1: usize, pos1: usize, route2: usize, pos2: usize) -> Self {
        Self {
            route1,
            pos1,
            route2,
            pos2,
            profit: 0,
        }
    }
}

pub struct Parameters {
    pub file_name: String,
    pub seed: [u8; 32],
    pub initial_penalty: isize,
    pub granularity: usize,
}

pub struct Data {
    pub seed: [u8; 32],
    pub nb_nodes: usize,
    pub nb_routes: usize,
    pub demands: Vec<isize>,
    pub max_capacity: isize,
    pub distance_matrix: Vec<Vec<isize>>,
    pub service_times: Vec<isize>,
    pub start_tw: Vec<isize>,
    pub end_tw: Vec<isize>,
    pub penalty_capacity: isize,
    pub penalty_tw: isize,
    pub granularity: usize,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Sequence {
    pub tau_minus: isize,
    pub tau_plus: isize,
    pub tmin: isize,
    pub tw: isize,
    pub total_service_duration: isize,
    pub load: isize,
    pub distance: isize,
    pub first_node: usize,
    pub last_node: usize,
}

#[inline(always)]
fn penalty(x: isize, coef: isize) -> isize {
    if x > 0 {
        x * coef
    } else {
        0
    }
}

#[inline(always)]
fn dm(dm: &[Vec<isize>], i: usize, j: usize) -> isize {
    dm[i][j]
}

impl Sequence {
    #[inline(always)]
    pub fn initialize(&mut self, data: &Data, node: usize) {
        let st = data.start_tw[node];
        let et = data.end_tw[node];
        let svc = data.service_times[node];
        let ld = data.demands[node];
        self.tau_minus = st;
        self.tau_plus = et;
        self.tmin = svc;
        self.tw = 0;
        self.total_service_duration = svc;
        self.load = ld;
        self.distance = 0;
        self.first_node = node;
        self.last_node = node;
    }

    #[inline(always)]
    pub fn join2(data: &Data, s1: &Sequence, s2: &Sequence) -> Sequence {
        let travel = dm(&data.distance_matrix, s1.last_node, s2.first_node);
        let distance = s1.distance + s2.distance + travel;
        let temp = travel + s1.tmin - s1.tw;

        let wtij = max(s2.tau_minus - temp - s1.tau_plus, 0);
        let twij = max(temp + s1.tau_minus - s2.tau_plus, 0);
        let tw = s1.tw + s2.tw + twij;
        let tmin = temp + s1.tw + s2.tmin + wtij;
        let tau_minus = max(s2.tau_minus - temp - wtij, s1.tau_minus);
        let tau_plus = min(s2.tau_plus - temp + twij, s1.tau_plus);
        let load = s1.load + s2.load;

        Sequence {
            tau_minus,
            tau_plus,
            tmin,
            tw,
            total_service_duration: s1.total_service_duration + s2.total_service_duration,
            load,
            distance,
            first_node: s1.first_node,
            last_node: s2.last_node,
        }
    }

    #[inline(always)]
    pub fn singleton(data: &Data, node: usize) -> Sequence {
        let mut s = Sequence::default();
        s.initialize(data, node);
        s
    }

    #[inline(always)]
    pub fn eval(&self, data: &Data) -> isize {
        self.distance
            + penalty(self.load - data.max_capacity, data.penalty_capacity)
            + penalty(self.tw, data.penalty_tw)
    }

    #[inline(always)]
    pub fn eval2(data: &Data, s1: &Sequence, s2: &Sequence) -> isize {
        let travel = dm(&data.distance_matrix, s1.last_node, s2.first_node);
        let distance = s1.distance + s2.distance + travel;
        let temp = s1.tmin - s1.tw + travel;
        let tw_viol = s1.tw + s2.tw + max(s1.tau_minus - s2.tau_plus + temp, 0);
        let load = s1.load + s2.load;

        distance
            + penalty(load - data.max_capacity, data.penalty_capacity)
            + penalty(tw_viol, data.penalty_tw)
    }

    #[inline(always)]
    pub fn eval3(data: &Data, s1: &Sequence, s2: &Sequence, s3: &Sequence) -> isize {
        let travel12 = dm(&data.distance_matrix, s1.last_node, s2.first_node);
        let distance12 = s1.distance + s2.distance + travel12;
        let mut temp = travel12 + s1.tmin - s1.tw;

        let wtij = max(s2.tau_minus - temp - s1.tau_plus, 0);
        let twij = max(temp + s1.tau_minus - s2.tau_plus, 0);
        let tw_viol12 = s1.tw + s2.tw + twij;
        let tmin12 = temp + s1.tw + s2.tmin + wtij;
        let tau_m12 = max(s2.tau_minus - temp - wtij, s1.tau_minus);
        let tau_p12 = min(s2.tau_plus - temp + twij, s1.tau_plus);

        let travel23 = dm(&data.distance_matrix, s2.last_node, s3.first_node);
        let distance = distance12 + s3.distance + travel23;
        temp = travel23 + tmin12 - tw_viol12;

        let tw_viol = tw_viol12 + s3.tw + max(tau_m12 - s3.tau_plus + temp, 0);
        let _tmin123 = tw_viol12 + s3.tmin + max(s3.tau_minus - tau_p12, temp);
        let load = s1.load + s2.load + s3.load;

        distance
            + penalty(load - data.max_capacity, data.penalty_capacity)
            + penalty(tw_viol, data.penalty_tw)
    }

    #[inline(always)]
    pub fn eval_n(data: &Data, chain: &[Sequence]) -> isize {
        debug_assert!(chain.len() >= 3);
        let mut agg = chain[0];

        for s in &chain[1..chain.len() - 1] {
            agg = Sequence::join2(data, &agg, s);
        }
        let last = &chain[chain.len() - 1];
        Sequence::eval2(data, &agg, last)
    }
}

pub struct TigLoader;

impl TigLoader {
    pub fn load(params: Parameters, challenge: &Challenge) -> Data {
        Data {
            seed: challenge.seed,
            nb_nodes: challenge.num_nodes,
            nb_routes: challenge.num_nodes,
            demands: challenge.demands.iter().map(|&d| d as isize).collect(),
            max_capacity: challenge.max_capacity as isize,
            distance_matrix: challenge
                .distance_matrix
                .iter()
                .map(|row| row.iter().map(|&x| x as isize).collect())
                .collect(),
            service_times: std::iter::once(0)
                .chain(
                    std::iter::repeat(challenge.service_time as isize)
                        .take(challenge.num_nodes),
                )
                .collect(),
            start_tw: challenge.ready_times.iter().map(|&d| d as isize).collect(),
            end_tw: challenge.due_times.iter().map(|&d| d as isize).collect(),
            penalty_capacity: params.initial_penalty,
            penalty_tw: params.initial_penalty,
            granularity: params.granularity,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Node {
    pub id: usize,
    pub seq0_i: Sequence,
    pub seqi_n: Sequence,
    pub seq1: Sequence,
    pub seq12: Sequence,
    pub seq21: Sequence,
}

impl Node {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            seq0_i: Sequence::default(),
            seqi_n: Sequence::default(),
            seq1: Sequence::default(),
            seq12: Sequence::default(),
            seq21: Sequence::default(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Route {
    pub cost: isize,
    pub load: isize,
    pub tw: isize,
    pub nodes: Vec<Node>,
}

impl Route {
    pub fn new(data: &Data, node_ids: &Vec<usize>) -> Self {
        let mut route = Route {
            cost: 0,
            load: 0,
            tw: 0,
            nodes: Vec::new(),
        };

        for &node_id in node_ids {
            route.nodes.push(Node::new(node_id));
        }

        route.update_metrics(data);
        route
    }

    pub fn update_metrics(&mut self, data: &Data) {
        self.cost = 0;
        self.load = 0;
        self.tw = 0;

        let mut current_time = 0;
        for i in 0..self.nodes.len() - 1 {
            let from = self.nodes[i].id;
            let to = self.nodes[i + 1].id;

            self.cost += data.distance_matrix[from][to];

            if to != 0 {
                self.load += data.demands[to];
            }

            current_time += data.distance_matrix[from][to];
            if current_time < data.start_tw[to] {
                current_time = data.start_tw[to];
            }
            if current_time > data.end_tw[to] {
                self.tw += current_time - data.end_tw[to];
                current_time = data.end_tw[to];
            }
            current_time += data.service_times[to];
        }
    }

    pub fn preprocess(&mut self, data: &Data) {
        let len = self.nodes.len();

        let mut acc_fwd = Sequence::singleton(data, self.nodes[0].id);
        self.nodes[0].seq0_i = acc_fwd;
        for pos in 1..len {
            let id = self.nodes[pos].id;
            acc_fwd = Sequence::join2(data, &acc_fwd, &Sequence::singleton(data, id));
            self.nodes[pos].seq0_i = acc_fwd;
        }

        let mut acc_bwd = Sequence::singleton(data, self.nodes[len - 1].id);
        self.nodes[len - 1].seqi_n = acc_bwd;
        for pos in (0..len - 1).rev() {
            let id = self.nodes[pos].id;
            acc_bwd = Sequence::join2(data, &Sequence::singleton(data, id), &acc_bwd);
            self.nodes[pos].seqi_n = acc_bwd;
        }

        for pos in 0..len {
            let id = self.nodes[pos].id;
            self.nodes[pos].seq1 = Sequence::singleton(data, id);
            if pos + 1 < len {
                let id_next = self.nodes[pos + 1].id;
                self.nodes[pos].seq12 = Sequence::join2(
                    data,
                    &Sequence::singleton(data, id),
                    &Sequence::singleton(data, id_next),
                );
                self.nodes[pos].seq21 = Sequence::join2(
                    data,
                    &Sequence::singleton(data, id_next),
                    &Sequence::singleton(data, id),
                );
            }
        }

        let end = self.nodes[len - 1].seq0_i;
        self.load = end.load;
        self.tw = end.tw;
        self.cost = end.eval(data);
    }

    pub fn to_string(&self) -> String {
        let mut result = String::from("[ ");
        for node in &self.nodes {
            result.push_str(&format!("{} ", node.id));
        }
        result.push_str(&format!(
            "] (Cost: {}, Load: {}, TW: {})",
            self.cost, self.load, self.tw
        ));
        result
    }
}

#[derive(Debug, Clone)]
pub struct Individual {
    pub cost: isize,
    pub routes: Vec<Route>,
    pub node_route: Vec<usize>,
    pub node_pos: Vec<usize>,
}

impl Individual {
    pub fn new(data: &Data, routes: Vec<Vec<usize>>) -> Self {
        let mut ind = Individual {
            cost: 0,
            routes: routes.iter().map(|r| Route::new(data, r)).collect(),
            node_route: vec![0; data.nb_nodes],
            node_pos: vec![0; data.nb_nodes],
        };

        ind.cost = ind.routes.iter().map(|r| r.cost).sum();
        ind.update_node_mappings();

        ind
    }

    pub fn update_route_node_mappings(&mut self, route_id: usize) {
        for (pos, node) in self.routes[route_id].nodes.iter().enumerate() {
            self.node_route[node.id] = route_id;
            self.node_pos[node.id] = pos;
        }
    }

    pub fn update_node_mappings(&mut self) {
        for route_id in 0..self.routes.len() {
            self.update_route_node_mappings(route_id);
        }
    }
}

pub struct Swap;

impl Swap {
    #[inline(always)]
    pub fn evaluate(data: &Data, ind: &Individual, args: &mut MoveArgs) {
        let r1 = args.route1;
        let r2 = args.route2;
        let p1 = args.pos1;
        let p2 = args.pos2;

        let new1 = Sequence::eval3(
            data,
            &ind.routes[r1].nodes[p1 - 1].seq0_i,
            &ind.routes[r2].nodes[p2].seq1,
            &ind.routes[r1].nodes[p1 + 1].seqi_n,
        );

        let new2 = Sequence::eval3(
            data,
            &ind.routes[r2].nodes[p2 - 1].seq0_i,
            &ind.routes[r1].nodes[p1].seq1,
            &ind.routes[r2].nodes[p2 + 1].seqi_n,
        );

        let old_cost = ind.routes[r1].cost + ind.routes[r2].cost;
        let new_cost = new1 + new2;
        args.profit = old_cost - new_cost;
    }

    pub fn perform(data: &Data, ind: &mut Individual, args: &MoveArgs) {
        let temp = take(&mut ind.routes[args.route1].nodes[args.pos1]);
        ind.routes[args.route1].nodes[args.pos1] =
            replace(&mut ind.routes[args.route2].nodes[args.pos2], temp);
        ind.routes[args.route1].preprocess(&data);
        ind.update_route_node_mappings(args.route1);
        ind.routes[args.route2].preprocess(&data);
        ind.update_route_node_mappings(args.route2);
        ind.cost -= args.profit;
    }
}

pub struct Constructive;

impl Constructive {
    pub fn build_individual(data: &Data) -> Individual {
        let mut routes = Vec::new();
        let mut nodes: Vec<usize> = (1..data.nb_nodes).collect();
        nodes.sort_by(|&a, &b| data.distance_matrix[0][a].cmp(&data.distance_matrix[0][b]));
        let mut remaining: BTreeSet<usize> = nodes.iter().cloned().collect();

        while let Some(node) = nodes.pop() {
            if !remaining.remove(&node) {
                continue;
            }
            let mut route = vec![0, node, 0];
            let mut route_demand = data.demands[node];

            while let Some((best_node, best_pos)) = Self::find_best_insertion(
                &route,
                remaining
                    .iter()
                    .cloned()
                    .filter(|&n| route_demand + data.demands[n] <= data.max_capacity)
                    .collect(),
                data,
            ) {
                remaining.remove(&best_node);
                route_demand += data.demands[best_node];
                route.insert(best_pos, best_node);
            }

            routes.push(route);
        }

        Individual::new(data, routes)
    }

    fn find_best_insertion(
        route: &Vec<usize>,
        remaining_nodes: Vec<usize>,
        data: &Data,
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
                let new_arrival_time = data.start_tw[insert_node]
                    .max(curr_time + data.distance_matrix[curr_node][insert_node]);
                if new_arrival_time > data.end_tw[insert_node] {
                    continue;
                }
                let old_arrival_time = data.start_tw[next_node]
                    .max(curr_time + data.distance_matrix[curr_node][next_node]);

                let c11 = data.distance_matrix[curr_node][insert_node]
                    + data.distance_matrix[insert_node][next_node]
                    - data.distance_matrix[curr_node][next_node];

                let c12 = new_arrival_time - old_arrival_time;
                let c1 = -(alpha1 * c11 + alpha2 * c12);
                let c2 = lambda * data.distance_matrix[0][insert_node] + c1;

                let c1_is_better = match best_c1 {
                    None => true,
                    Some(x) => c1 > x,
                };

                let c2_is_better = match best_c2 {
                    None => true,
                    Some(x) => c2 > x,
                };

                if c1_is_better
                    && c2_is_better
                    && Self::is_feasible(
                        route,
                        insert_node,
                        new_arrival_time + data.service_times[next_node],
                        pos,
                        data,
                    )
                {
                    best_c1 = Some(c1);
                    best_c2 = Some(c2);
                    best = Some((insert_node, pos));
                }

                curr_time = data.start_tw[next_node]
                    .max(curr_time + data.distance_matrix[curr_node][next_node])
                    + data.service_times[next_node];
                curr_node = next_node;
            }
        }
        best
    }

    fn is_feasible(
        route: &Vec<usize>,
        mut curr_node: usize,
        mut curr_time: isize,
        start_pos: usize,
        data: &Data,
    ) -> bool {
        let mut valid = true;
        for pos in start_pos..route.len() {
            let next_node = route[pos];
            curr_time += data.distance_matrix[curr_node][next_node];
            if curr_time > data.end_tw[route[pos]] {
                valid = false;
                break;
            }
            curr_time = curr_time.max(data.start_tw[next_node]) + data.service_times[next_node];
            curr_node = next_node;
        }
        valid
    }
}

pub struct Relocate;

impl Relocate {
    #[inline(always)]
    pub fn evaluate(data: &Data, ind: &Individual, args: &mut MoveArgs) {
        let r1 = args.route1;
        let r2 = args.route2;
        let p1 = args.pos1;
        let p2 = args.pos2;

        let new1 = Sequence::eval2(
            data,
            &ind.routes[r1].nodes[p1 - 1].seq0_i,
            &ind.routes[r1].nodes[p1 + 1].seqi_n,
        );

        let new2 = Sequence::eval3(
            data,
            &ind.routes[r2].nodes[p2 - 1].seq0_i,
            &ind.routes[r1].nodes[p1].seq1,
            &ind.routes[r2].nodes[p2].seqi_n,
        );

        let old_cost = ind.routes[r1].cost + ind.routes[r2].cost;
        let new_cost = new1 + new2;
        args.profit = old_cost - new_cost;
    }

    pub fn perform(data: &Data, ind: &mut Individual, args: &MoveArgs) {
        let element = ind.routes[args.route1].nodes.remove(args.pos1);
        ind.routes[args.route2].nodes.insert(args.pos2, element);
        ind.routes[args.route1].preprocess(&data);
        ind.update_route_node_mappings(args.route1);
        ind.routes[args.route2].preprocess(&data);
        ind.update_route_node_mappings(args.route2);
        ind.cost -= args.profit;
    }
}

pub struct TwoOptStar;

impl TwoOptStar {
    #[inline(always)]
    pub fn evaluate(data: &Data, ind: &Individual, args: &mut MoveArgs) {
        let r1 = args.route1;
        let r2 = args.route2;
        let p1 = args.pos1;
        let p2 = args.pos2;

        let new1 = Sequence::eval2(
            data,
            &ind.routes[r1].nodes[p1 - 1].seq0_i,
            &ind.routes[r2].nodes[p2].seqi_n,
        );

        let new2 = Sequence::eval2(
            data,
            &ind.routes[r2].nodes[p2 - 1].seq0_i,
            &ind.routes[r1].nodes[p1].seqi_n,
        );

        let old_cost = ind.routes[r1].cost + ind.routes[r2].cost;
        let new_cost = new1 + new2;
        args.profit = old_cost - new_cost;
    }

    pub fn perform(data: &Data, ind: &mut Individual, args: &MoveArgs) {
        let mut suffix1 = ind.routes[args.route1].nodes.split_off(args.pos1);
        let mut suffix2 = ind.routes[args.route2].nodes.split_off(args.pos2);
        ind.routes[args.route1].nodes.append(&mut suffix2);
        ind.routes[args.route2].nodes.append(&mut suffix1);
        ind.routes[args.route1].preprocess(&data);
        ind.update_route_node_mappings(args.route1);
        ind.routes[args.route2].preprocess(&data);
        ind.update_route_node_mappings(args.route2);
        ind.cost -= args.profit;
    }
}

pub struct LocalSearch<'a> {
    pub data: &'a Data,

    pub neighbors_before: Vec<Vec<usize>>,
}

impl<'a> LocalSearch<'a> {
    pub fn new(data: &'a Data) -> Self {
        let mut neighbors_before: Vec<Vec<usize>> = vec![Vec::new(); data.nb_nodes];

        for i in 1..data.nb_nodes {
            let mut prox: Vec<(isize, usize)> = Vec::with_capacity(data.nb_nodes - 2);
            for j in 1..data.nb_nodes {
                if j == i {
                    continue;
                }
                let tji = data.distance_matrix[j][i];
                let wait = (data.start_tw[i] - tji - data.service_times[j] - data.end_tw[j]).max(0);
                let late = (data.start_tw[j] + data.service_times[j] + tji - data.end_tw[i]).max(0);
                let proxy10 = 10 * tji + 2 * wait + 10 * late;
                prox.push((proxy10, j));
            }

            prox.sort_by_key(|&(p, _)| p);
            let keep = min(data.granularity, data.nb_nodes - 2);
            neighbors_before[i] = prox[..keep].iter().map(|&(_, j)| j).collect();
        }

        Self {
            data,
            neighbors_before,
        }
    }

    pub fn run_intra_route_relocate(
        &mut self,
        ind: &mut Individual,
        r1: usize,
        pos1: usize,
    ) -> bool {
        let route = &ind.routes[r1];
        let len = route.nodes.len();
        if len < pos1 + 4 {
            return false;
        }

        debug_assert!(pos1 > 0);
        debug_assert!(ind.routes[r1].nodes[pos1].id != 0);

        let mut left_excl: Vec<Sequence> = vec![Sequence::default(); len];
        let mut acc_left = route.nodes[0].seq0_i;
        for p in 1..len {
            left_excl[p] = acc_left;
            if p != pos1 {
                acc_left = Sequence::join2(self.data, &acc_left, &route.nodes[p].seq1);
            }
        }

        let mut right_excl: Vec<Sequence> = vec![Sequence::default(); len];
        let mut acc_right = route.nodes[len - 1].seq1;
        right_excl[len - 1] = acc_right;
        for p in (1..len - 1).rev() {
            if p != pos1 {
                acc_right = Sequence::join2(self.data, &route.nodes[p].seq1, &acc_right);
            }
            right_excl[p] = acc_right;
        }

        let old_cost = route.cost;
        let mut best_cost = old_cost;
        let mut best_pos: Option<usize> = None;

        for t in 1..len {
            if t == pos1 || t == pos1 + 1 {
                continue;
            }
            let new_cost = Sequence::eval3(
                self.data,
                &left_excl[t],
                &route.nodes[pos1].seq1,
                &right_excl[t],
            );

            if new_cost < best_cost {
                best_cost = new_cost;
                best_pos = Some(t);
            }
        }

        if let Some(mypos) = best_pos {
            let insert_pos = if mypos > pos1 { mypos - 1 } else { mypos };
            let elem = ind.routes[r1].nodes.remove(pos1);
            ind.routes[r1].nodes.insert(insert_pos, elem);
            ind.routes[r1].preprocess(self.data);
            ind.update_route_node_mappings(r1);
            ind.cost += ind.routes[r1].cost - old_cost;

            return true;
        } else {
            return false;
        };
    }

    pub fn run_intra_route_swap_right(
        &mut self,
        ind: &mut Individual,
        r1: usize,
        pos1: usize,
    ) -> bool {
        let route = &ind.routes[r1];
        let len = route.nodes.len();
        if len < pos1 + 4 {
            return false;
        }

        debug_assert!(pos1 > 0);
        debug_assert!(ind.routes[r1].nodes[pos1].id != 0);

        let old_cost = route.cost;
        let mut best_cost = old_cost;
        let mut best_pos: Option<usize> = None;

        let mut acc_mid = route.nodes[pos1 + 1].seq1;
        for pos2 in (pos1 + 2)..(len - 1) {
            let new_cost = Sequence::eval_n(
                self.data,
                &[
                    route.nodes[pos1 - 1].seq0_i,
                    route.nodes[pos2].seq1,
                    acc_mid,
                    route.nodes[pos1].seq1,
                    route.nodes[pos2 + 1].seqi_n,
                ],
            );
            if new_cost < best_cost {
                best_cost = new_cost;
                best_pos = Some(pos2);
            }
            acc_mid = Sequence::join2(self.data, &acc_mid, &route.nodes[pos2].seq1);
        }

        if let Some(mypos) = best_pos {
            ind.routes[r1].nodes.swap(pos1, mypos);
            ind.routes[r1].preprocess(self.data);
            ind.update_route_node_mappings(r1);
            ind.cost += ind.routes[r1].cost - old_cost;

            return true;
        } else {
            return false;
        };
    }

    pub fn run_2opt(&mut self, ind: &mut Individual, r1: usize, pos1: usize) -> bool {
        let route = &ind.routes[r1];
        let len = route.nodes.len();
        if len < pos1 + 3 {
            return false;
        }

        debug_assert!(pos1 > 0);
        debug_assert!(ind.routes[r1].nodes[pos1].id != 0);

        let old_cost = route.cost;
        let mut best_cost = old_cost;
        let mut best_pos: Option<usize> = None;

        let mut mid_rev = route.nodes[pos1].seq21;
        for pos2 in (pos1 + 1)..(len - 1) {
            let new_cost = Sequence::eval3(
                self.data,
                &route.nodes[pos1 - 1].seq0_i,
                &mid_rev,
                &route.nodes[pos2 + 1].seqi_n,
            );
            if new_cost < best_cost {
                best_cost = new_cost;
                best_pos = Some(pos2);
            }
            if pos2 + 1 < len - 1 {
                mid_rev = Sequence::join2(self.data, &route.nodes[pos2 + 1].seq1, &mid_rev);
            }
        }

        if let Some(mypos) = best_pos {
            ind.routes[r1].nodes[pos1..=mypos].reverse();
            ind.routes[r1].preprocess(self.data);
            ind.update_route_node_mappings(r1);
            ind.cost += ind.routes[r1].cost - old_cost;

            return true;
        } else {
            return false;
        };
    }

    pub fn runls(&mut self, ind: &mut Individual) {
        for route in &mut ind.routes {
            route.preprocess(self.data);
        }

        let mut improved = true;
        let mut loop_id = 0;
        while improved {
            improved = false;
            loop_id += 1;
            let c1_order: Vec<usize> = (1..self.data.nb_nodes).collect();
            for c1 in c1_order {
                for &c2 in &self.neighbors_before[c1] {
                    let r1 = ind.node_route[c1];
                    let pos1 = ind.node_pos[c1];
                    let r2 = ind.node_route[c2];
                    let pos2 = ind.node_pos[c2] + 1;
                    if r1 == r2 {
                        continue;
                    }
                    let mut args = MoveArgs::new(r1, pos1, r2, pos2);

                    if ind.routes[r2].nodes[pos2].id != 0 {
                        Swap::evaluate(self.data, ind, &mut args);
                        if args.profit > 0 {
                            Swap::perform(self.data, ind, &args);
                            improved = true;
                            continue;
                        }
                    }

                    Relocate::evaluate(self.data, ind, &mut args);
                    if args.profit > 0 {
                        Relocate::perform(self.data, ind, &args);
                        improved = true;
                        continue;
                    }

                    TwoOptStar::evaluate(self.data, ind, &mut args);
                    if args.profit > 0 {
                        TwoOptStar::perform(self.data, ind, &args);
                        improved = true;
                        continue;
                    }
                }

                let mut tested_empty_route = false;
                for r2 in 0..ind.routes.len() {
                    let r1 = ind.node_route[c1];
                    let pos1 = ind.node_pos[c1];
                    let pos2 = 1;
                    if r1 == r2 {
                        continue;
                    }

                    if ind.routes[r2].nodes.len() == 2 {
                        if loop_id == 1 || tested_empty_route {
                            continue;
                        };
                        tested_empty_route = true;
                    }

                    let mut args = MoveArgs::new(r1, pos1, r2, pos2);
                    if ind.routes[r2].nodes[pos2].id != 0 {
                        Swap::evaluate(self.data, ind, &mut args);
                        if args.profit > 0 {
                            Swap::perform(self.data, ind, &args);
                            improved = true;
                            continue;
                        }
                    }

                    Relocate::evaluate(self.data, ind, &mut args);
                    if args.profit > 0 {
                        Relocate::perform(self.data, ind, &args);
                        improved = true;
                        continue;
                    }

                    TwoOptStar::evaluate(self.data, ind, &mut args);
                    if args.profit > 0 {
                        TwoOptStar::perform(self.data, ind, &args);
                        improved = true;
                        continue;
                    }
                }

                improved |= self.run_2opt(ind, ind.node_route[c1], ind.node_pos[c1]);
                improved |=
                    self.run_intra_route_relocate(ind, ind.node_route[c1], ind.node_pos[c1]);
                improved |=
                    self.run_intra_route_swap_right(ind, ind.node_route[c1], ind.node_pos[c1]);
            }
        }
    }
}

fn solve(data: Data) -> anyhow::Result<Option<Solution>> {
    let mut individual = Constructive::build_individual(&data);
    let mut my_local_search = LocalSearch::new(&data);
    my_local_search.runls(&mut individual);

    let usize_routes = individual
        .routes
        .iter()
        .map(|r| r.nodes.iter().map(|n| n.id).collect::<Vec<usize>>())
        .filter(|route| route.len() > 2)
        .collect::<Vec<Vec<usize>>>();

    Ok(Some(Solution {
        routes: usize_routes,
    }))
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let data = TigLoader::load(
        Parameters {
            file_name: String::new(),
            seed: challenge.seed,
            initial_penalty: 1000,
            granularity: 30,
        },
        &challenge,
    );

    match solve(data) {
        Ok(Some(solution)) => {
            let _ = save_solution(&solution);
        }
        Ok(None) => {
            eprintln!("No solution found for the sub-instance.");
        }
        Err(e) => {
            eprintln!("Error solving sub-instance: {}", e);
        }
    }
    Ok(())
}

pub fn help() {
    println!("No help information available.");
}
