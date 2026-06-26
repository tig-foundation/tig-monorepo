use super::instance::Instance;
use super::config::Config;
use super::route_eval::RouteEval;
use rand::rngs::SmallRng;
use rand::Rng;
use rand::seq::SliceRandom;
use std::cmp::{max, min};

#[derive(Clone, Default)]
pub struct Node {
    id: usize,
    seq0_i: RouteEval,
    seqi_n: RouteEval,
    seq1: RouteEval,
    seq12: RouteEval,
    seq21: RouteEval,
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
}

impl Route {
    #[inline]
    fn new(ids: &[usize]) -> Self {
        Self {
            nodes: ids.iter().copied().map(Node::new).collect(),
            ..Default::default()
        }
    }
}

pub struct LocalOps<'a> {
    pub data: &'a Instance,
    pub neighbors_before: Vec<Vec<usize>>,
    pub neighbors_capacity_swap: Vec<Vec<usize>>,
    pub params: Config,
    pub cost: i64,
    pub routes: Vec<Route>,
    pub node_route: Vec<usize>,
    pub node_pos: Vec<usize>,
    pub empty_routes: Vec<usize>,
    pub empty_route_pos: Vec<usize>,
    pub when_last_modified: Vec<usize>,
    pub when_last_tested: Vec<usize>,
    pub nb_moves: usize,
    scratch_a: Vec<RouteEval>,
    scratch_b: Vec<RouteEval>,
    scratch_c: Vec<RouteEval>,
    scratch_d: Vec<RouteEval>,
}

impl<'a> LocalOps<'a> {
    pub fn new(data: &'a Instance, params: Config) -> Self {
        let n = data.nb_nodes;
        let cap = n.saturating_sub(2);
        let keep = min(params.granularity as usize, cap);
        let mut neighbors_before: Vec<Vec<usize>> = vec![Vec::new(); n];

        for i in 1..n {
            let mut prox: Vec<(i32, usize)> = Vec::with_capacity(cap);
            for j in 1..n {
                if j == i {
                    continue;
                }
                let tji = data.dm(j, i);
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
                if j == i {
                    continue;
                }
                if (data.demands[j] - di).abs() <= diff_limit {
                    let dij = data.dm(i, j);
                    prox.push((dij, j));
                }
            }
            prox.sort_by_key(|&(d, _)| d);
            let m = prox.len().min(params.granularity2 as usize);
            neighbors_capacity_swap[i] = prox[..m].iter().map(|&(_, j)| j).collect();
        }

        Self {
            data,
            neighbors_before,
            neighbors_capacity_swap,
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
            scratch_a: Vec::new(),
            scratch_b: Vec::new(),
            scratch_c: Vec::new(),
            scratch_d: Vec::new(),
        }
    }

    fn eval_route_ids(&self, ids: &[usize]) -> i64 {
        if ids.is_empty() {
            return 0;
        }
        let mut acc = RouteEval::singleton(self.data, ids[0]);
        for &id in ids.iter().skip(1) {
            let seq = RouteEval::singleton(self.data, id);
            acc = RouteEval::join2(self.data, &acc, &seq);
        }
        acc.eval(self.data, &self.params)
    }

    fn reinsert_overflow_customers(&mut self, mut pending: Vec<usize>) {
        if pending.is_empty() {
            return;
        }

        pending.sort_by(|&a, &b| {
            let hard_a = 2 * self.data.dm(0, a) as i64 + self.data.service_times[a] as i64
                - (self.data.end_tw[a] - self.data.start_tw[a]) as i64;
            let hard_b = 2 * self.data.dm(0, b) as i64 + self.data.service_times[b] as i64
                - (self.data.end_tw[b] - self.data.start_tw[b]) as i64;
            hard_b.cmp(&hard_a)
        });

        while !pending.is_empty() {
            let scan_limit = pending.len().min(48);
            let mut best_choice: Option<(usize, usize, usize, i64, i64, bool, i64)> = None;

            for idx in 0..scan_limit {
                let customer = pending[idx];
                let seq1 = RouteEval::singleton(self.data, customer);

                let mut best_rid = 0usize;
                let mut best_pos = 1usize;
                let mut best_delta = i64::MAX / 4;
                let mut second_delta = i64::MAX / 4;
                let mut best_empty: Option<(usize, usize, i64)> = None;

                for rid in 0..self.routes.len() {
                    let route = &self.routes[rid];
                    for pos in 1..route.nodes.len() {
                        let cand = RouteEval::eval3(
                            self.data,
                            &self.params,
                            &route.nodes[pos - 1].seq0_i,
                            &seq1,
                            &route.nodes[pos].seqi_n,
                        );
                        let delta = cand - route.cost;

                        if route.nodes.len() == 2 {
                            match best_empty {
                                Some((_, _, cur_delta)) if cur_delta <= delta => {}
                                _ => best_empty = Some((rid, pos, delta)),
                            }
                        }

                        if delta < best_delta {
                            second_delta = best_delta;
                            best_delta = delta;
                            best_rid = rid;
                            best_pos = pos;
                        } else if delta < second_delta {
                            second_delta = delta;
                        }
                    }
                }

                if second_delta == i64::MAX / 4 {
                    second_delta = best_delta;
                }

                let mut chosen_rid = best_rid;
                let mut chosen_pos = best_pos;
                let mut chosen_delta = best_delta;
                let mut prefer_empty = false;

                if let Some((empty_rid, empty_pos, empty_delta)) = best_empty {
                    let margin = max(10_i64, best_delta.abs() / 20);
                    let stressed_best = self.routes[best_rid].tw > 0
                        || self.routes[best_rid].load > self.data.max_capacity;
                    if empty_delta <= best_delta + margin
                        && (stressed_best || self.routes[best_rid].nodes.len() > 2)
                    {
                        chosen_rid = empty_rid;
                        chosen_pos = empty_pos;
                        chosen_delta = empty_delta;
                        prefer_empty = true;
                    }
                }

                let regret = second_delta - best_delta;
                let hardness = 2 * self.data.dm(0, customer) as i64
                    + self.data.service_times[customer] as i64
                    - (self.data.end_tw[customer] - self.data.start_tw[customer]) as i64;

                let take = match best_choice {
                    None => true,
                    Some((_, _, _, best_delta0, best_regret0, best_empty0, best_hard0)) => {
                        regret > best_regret0
                            || (regret == best_regret0 && chosen_delta < best_delta0)
                            || (regret == best_regret0
                                && chosen_delta == best_delta0
                                && prefer_empty
                                && !best_empty0)
                            || (regret == best_regret0
                                && chosen_delta == best_delta0
                                && prefer_empty == best_empty0
                                && hardness > best_hard0)
                    }
                };

                if take {
                    best_choice = Some((
                        idx,
                        chosen_rid,
                        chosen_pos,
                        chosen_delta,
                        regret,
                        prefer_empty,
                        hardness,
                    ));
                }
            }

            let (idx, rid, pos, _, _, _, _) = best_choice.unwrap();
            let customer = pending.remove(idx);
            let old_cost = self.routes[rid].cost;

            let mut node = Node::new(customer);
            node.seq1 = RouteEval::singleton(self.data, customer);
            self.routes[rid].nodes.insert(pos, node);

            self.nb_moves += 1;
            self.update_route(rid);
            self.cost += self.routes[rid].cost - old_cost;
        }
    }

    fn load_from_routes(&mut self, routes: &Vec<Vec<usize>>) {
        let n = self.data.nb_nodes;
        let fleet = self.data.nb_vehicles;

        let mut src: Vec<Vec<usize>> = Vec::new();
        let mut overflow_customers: Vec<usize> = Vec::new();

        if routes.len() <= fleet {
            src.extend(routes.iter().cloned());
        } else {
            let mut scored_routes: Vec<(i64, usize, Vec<usize>)> = Vec::new();
            for (idx, route) in routes.iter().enumerate() {
                if route.len() > 2 {
                    let customers = route.len().saturating_sub(2) as i64;
                    let route_cost = self.eval_route_ids(route);
                    let score = route_cost.saturating_mul(1024) / customers.max(1) - customers;
                    scored_routes.push((score, idx, route.clone()));
                }
            }

            scored_routes.sort_by_key(|&(score, idx, _)| (score, idx));
            let keep_nonempty = scored_routes.len().min(fleet);

            for (rank, (_, _, route)) in scored_routes.into_iter().enumerate() {
                if rank < keep_nonempty {
                    src.push(route);
                } else {
                    overflow_customers.extend_from_slice(&route[1..route.len() - 1]);
                }
            }
        }

        while src.len() < fleet {
            src.push(vec![0, 0]);
        }

        let all_routes: Vec<Route> = src.iter().map(|r| Route::new(r)).collect();
        self.node_route.resize(n, 0);
        self.node_route.fill(0);
        self.node_pos.resize(n, 0);
        self.node_pos.fill(0);
        self.empty_routes.clear();
        self.routes = all_routes;
        for route in &mut self.routes {
            for node in &mut route.nodes {
                node.seq1 = RouteEval::singleton(self.data, node.id);
            }
        }
        self.empty_route_pos.resize(self.routes.len(), usize::MAX);
        self.empty_route_pos.fill(usize::MAX);

        self.when_last_modified.resize(self.routes.len(), 0);
        self.when_last_modified.fill(0);
        self.when_last_tested.fill(0);
        self.nb_moves = 1;

        for rid in 0..self.routes.len() {
            self.update_route(rid);
        }
        self.cost = self.routes.iter().map(|r| r.cost).sum();

        if !overflow_customers.is_empty() {
            self.reinsert_overflow_customers(overflow_customers);
            self.cost = self.routes.iter().map(|r| r.cost).sum();
        }
    }

    fn write_back_to_routes(&self, out: &mut Vec<Vec<usize>>) {
        out.clear();
        out.extend(
            self.routes
                .iter()
                .filter(|r| r.nodes.len() > 2)
                .map(|r| r.nodes.iter().map(|n| n.id).collect::<Vec<usize>>()),
        );
    }

    fn update_route(&mut self, rid: usize) {
        let data = self.data;
        let params = self.params;
        let nb_moves = self.nb_moves;
        let (routes, node_route, node_pos, empty_routes, empty_route_pos, when_last_modified) = (
            &mut self.routes,
            &mut self.node_route,
            &mut self.node_pos,
            &mut self.empty_routes,
            &mut self.empty_route_pos,
            &mut self.when_last_modified,
        );
        let r = &mut routes[rid];
        let len = r.nodes.len();

        let id0 = r.nodes[0].id;
        let mut acc_fwd = r.nodes[0].seq1;
        r.nodes[0].seq0_i = acc_fwd;
        node_route[id0] = rid;
        node_pos[id0] = 0;
        for pos in 1..len {
            let id = r.nodes[pos].id;
            let s1 = r.nodes[pos].seq1;
            acc_fwd = RouteEval::join2(data, &acc_fwd, &s1);
            r.nodes[pos].seq0_i = acc_fwd;
            node_route[id] = rid;
            node_pos[id] = pos;
        }

        r.nodes[len - 1].seqi_n = r.nodes[len - 1].seq1;
        for pos in (0..len - 1).rev() {
            let cur = r.nodes[pos].seq1;
            let next = r.nodes[pos + 1].seqi_n;
            r.nodes[pos].seqi_n = RouteEval::join2(data, &cur, &next);
        }

        for pos in 0..len - 1 {
            let s1 = r.nodes[pos].seq1;
            let s2 = r.nodes[pos + 1].seq1;
            let s12 = RouteEval::join2(data, &s1, &s2);
            r.nodes[pos].seq12 = s12;
            r.nodes[pos].seq21 = RouteEval::join2(data, &s2, &s1);
        }

        let end = r.nodes[len - 1].seq0_i;
        r.load = end.load;
        r.tw = end.tw;
        r.distance = end.distance;
        r.cost = end.eval(data, &params);

        let is_empty = len == 2;
        let pos = empty_route_pos[rid];
        if is_empty {
            if pos == usize::MAX {
                empty_route_pos[rid] = empty_routes.len();
                empty_routes.push(rid);
            }
        } else if pos != usize::MAX {
            let last = empty_routes.pop().unwrap();
            if pos < empty_routes.len() {
                empty_routes[pos] = last;
                empty_route_pos[last] = pos;
            }
            empty_route_pos[rid] = usize::MAX;
        }
        when_last_modified[rid] = nb_moves;
    }

    fn build_skip_one_scratch(
        data: &Instance,
        route: &Route,
        skip_pos: usize,
        scratch_left: &mut Vec<RouteEval>,
        scratch_right: &mut Vec<RouteEval>,
    ) {
        let len = route.nodes.len();
        scratch_left.resize(len, RouteEval::default());
        scratch_right.resize(len, RouteEval::default());

        let nodes = &route.nodes;
        let mut acc_left = nodes[0].seq0_i;
        for p in 1..len {
            scratch_left[p] = acc_left;
            if p != skip_pos {
                acc_left = RouteEval::join2(data, &acc_left, &nodes[p].seq1);
            }
        }

        let mut acc_right = nodes[len - 1].seq1;
        scratch_right[len - 1] = acc_right;
        for p in (1..len - 1).rev() {
            if p != skip_pos {
                acc_right = RouteEval::join2(data, &nodes[p].seq1, &acc_right);
            }
            scratch_right[p] = acc_right;
        }
    }

    pub fn run_intra_route_relocate(&mut self, r1: usize, pos1: usize) -> bool {
        let route = &self.routes[r1];
        let len = route.nodes.len();
        if len < pos1 + 4 {
            return false;
        }

        Self::build_skip_one_scratch(
            self.data,
            &self.routes[r1],
            pos1,
            &mut self.scratch_a,
            &mut self.scratch_b,
        );

        let old_cost = self.routes[r1].cost;
        let mut best_cost = old_cost;
        let mut best_pos: Option<usize> = None;

        let seq1_pos1 = self.routes[r1].nodes[pos1].seq1;
        for t in 1..len {
            if t == pos1 || t == pos1 + 1 {
                continue;
            }
            let new_cost = RouteEval::eval3(self.data, &self.params, &self.scratch_a[t], &seq1_pos1, &self.scratch_b[t]);
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
        } else {
            false
        }
    }

    pub fn run_intra_route_swap_right(&mut self, r1: usize, pos1: usize) -> bool {
        let route = &self.routes[r1];
        let len = route.nodes.len();
        if len < pos1 + 4 {
            return false;
        }

        let old_cost = route.cost;
        let mut best_cost = old_cost;
        let mut best_pos: Option<usize> = None;

        let mut acc_mid = route.nodes[pos1 + 1].seq1;
        let prefix = route.nodes[pos1 - 1].seq0_i;
        let seq1_pos1 = route.nodes[pos1].seq1;
        for pos2 in (pos1 + 2)..(len - 1) {
            let new_cost = RouteEval::eval5(
                self.data,
                &self.params,
                &prefix,
                &route.nodes[pos2].seq1,
                &acc_mid,
                &seq1_pos1,
                &route.nodes[pos2 + 1].seqi_n,
            );
            if new_cost < best_cost {
                best_cost = new_cost;
                best_pos = Some(pos2);
            }
            acc_mid = RouteEval::join2(self.data, &acc_mid, &route.nodes[pos2].seq1);
        }

        if let Some(mypos) = best_pos {
            self.routes[r1].nodes.swap(pos1, mypos);
            self.nb_moves += 1;
            self.update_route(r1);
            self.cost += self.routes[r1].cost - old_cost;
            true
        } else {
            false
        }
    }

    pub fn run_2optstar(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize) -> bool {
        let route1 = &self.routes[r1];
        let route2 = &self.routes[r2];

        let new1 = RouteEval::eval2(self.data, &self.params, &route1.nodes[pos1 - 1].seq0_i, &route2.nodes[pos2].seqi_n);
        let new2 = RouteEval::eval2(self.data, &self.params, &route2.nodes[pos2 - 1].seq0_i, &route1.nodes[pos1].seqi_n);

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
        } else {
            false
        }
    }

    pub fn run_2opt(&mut self, r1: usize, pos1: usize) -> bool {
        let route = &self.routes[r1];
        let len = route.nodes.len();
        if len < pos1 + 3 {
            return false;
        }

        let old_cost = route.cost;
        let mut best_cost = old_cost;
        let mut best_pos: Option<usize> = None;

        let mut mid_rev = route.nodes[pos1].seq21;
        for pos2 in (pos1 + 1)..(len - 1) {
            let new_cost = RouteEval::eval3(
                self.data,
                &self.params,
                &route.nodes[pos1 - 1].seq0_i,
                &mid_rev,
                &route.nodes[pos2 + 1].seqi_n,
            );
            if new_cost < best_cost {
                best_cost = new_cost;
                best_pos = Some(pos2);
            }
            if pos2 + 1 < len - 1 {
                mid_rev = RouteEval::join2(self.data, &route.nodes[pos2 + 1].seq1, &mid_rev);
            }
        }

        if let Some(mypos) = best_pos {
            self.routes[r1].nodes[pos1..=mypos].reverse();
            self.nb_moves += 1;
            self.update_route(r1);
            self.cost += self.routes[r1].cost - old_cost;
            true
        } else {
            false
        }
    }

    pub fn run_intra_route_oropt(&mut self, r1: usize, pos1: usize, l: usize) -> bool {
        if l < 2 || l > 3 {
            return false;
        }
        let old_cost = self.routes[r1].cost;

        let applied = {
            let route = &self.routes[r1];
            let len = route.nodes.len();
            if pos1 == 0 || pos1 >= len - 1 {
                return false;
            }
            if pos1 + l >= len {
                return false;
            }
            if l == 3 && pos1 + 2 >= len {
                return false;
            }

            let block_seq = if l == 2 {
                route.nodes[pos1].seq12
            } else {
                RouteEval::join2(self.data, &route.nodes[pos1].seq12, &route.nodes[pos1 + 2].seq1)
            };

            let mut best_cost = old_cost;
            let mut best_dir = 0i32;
            let mut best_t = 0usize;

            let suffix_start = pos1 + l;
            let suffix_fixed = route.nodes[suffix_start].seqi_n;

            if pos1 > 1 {
                let mut mid_seq = route.nodes[pos1 - 1].seq1;
                for t in (1..pos1).rev() {
                    let prefix_seq = route.nodes[t - 1].seq0_i;
                    let cand = RouteEval::eval4(self.data, &self.params, &prefix_seq, &block_seq, &mid_seq, &suffix_fixed);
                    if cand < best_cost {
                        best_cost = cand;
                        best_dir = -1;
                        best_t = t;
                    }
                    if t > 1 {
                        mid_seq = RouteEval::join2(self.data, &route.nodes[t - 1].seq1, &mid_seq);
                    }
                }
            }

            if pos1 + l < len - 1 {
                let prefix_seq = route.nodes[pos1 - 1].seq0_i;
                let mut mid_seq = route.nodes[pos1 + l].seq1;
                for t in (pos1 + l + 1)..len {
                    let suffix_seq = route.nodes[t].seqi_n;
                    let cand = RouteEval::eval4(self.data, &self.params, &prefix_seq, &mid_seq, &block_seq, &suffix_seq);
                    if cand < best_cost {
                        best_cost = cand;
                        best_dir = 1;
                        best_t = t;
                    }
                    if t < len - 1 {
                        mid_seq = RouteEval::join2(self.data, &mid_seq, &route.nodes[t].seq1);
                    }
                }
            }

            if best_dir == 0 {
                return false;
            }

            if best_dir > 0 {
                self.routes[r1].nodes[pos1..best_t].rotate_left(l);
            } else {
                self.routes[r1].nodes[best_t..pos1 + l].rotate_right(l);
            }

            true
        };

        if !applied {
            return false;
        }

        self.nb_moves += 1;
        self.update_route(r1);
        self.cost += self.routes[r1].cost - old_cost;
        true
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

        let result10 = RouteEval::eval2(data, &self.params, &u_pred.seq0_i, &x.seqi_n)
            + RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u.seq1, &v.seqi_n);
        update_best(1, 0, result10);

        if v.id != 0 {
            let result11 = RouteEval::eval3(data, &self.params, &u_pred.seq0_i, &v.seq1, &x.seqi_n)
                + RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u.seq1, &rv.nodes[pos2 + 1].seqi_n);
            update_best(1, 1, result11);
        }

        if x.id != 0 {
            let x_next = &ru.nodes[pos1 + 2];
            let mut result20 = RouteEval::eval2(data, &self.params, &u_pred.seq0_i, &x_next.seqi_n);
            let mut result30 = result20;
            result20 += RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u.seq12, &v.seqi_n);
            result30 += RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u.seq21, &v.seqi_n);
            update_best(2, 0, result20);
            update_best(3, 0, result30);

            if v.id != 0 {
                let y = &rv.nodes[pos2 + 1];
                let mut result21 = RouteEval::eval3(data, &self.params, &u_pred.seq0_i, &v.seq1, &x_next.seqi_n);
                let mut result31 = result21;
                result21 += RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u.seq12, &y.seqi_n);
                result31 += RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u.seq21, &y.seqi_n);
                update_best(2, 1, result21);
                update_best(3, 1, result31);

                if y.id != 0 {
                    let mut result22 = RouteEval::eval3(data, &self.params, &u_pred.seq0_i, &v.seq12, &x_next.seqi_n);
                    let mut result23 = RouteEval::eval3(data, &self.params, &u_pred.seq0_i, &v.seq21, &x_next.seqi_n);
                    let mut result32 = result22;
                    let mut result33 = result23;

                    let y_next = &rv.nodes[pos2 + 2];
                    let tmp = RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u.seq12, &y_next.seqi_n);
                    let tmp2 = RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u.seq21, &y_next.seqi_n);
                    result22 += tmp;
                    result23 += tmp;
                    result32 += tmp2;
                    result33 += tmp2;
                    update_best(2, 2, result22);
                    update_best(3, 2, result32);
                    update_best(2, 3, result23);
                    update_best(3, 3, result33);
                }
            }

            if x_next.id != 0 && self.params.allow_swap3 {
                let x2_next = &ru.nodes[pos1 + 3];
                let u_seq123 = RouteEval::join2(data, &u.seq12, &x.seq1);
                let result40 = RouteEval::eval2(data, &self.params, &u_pred.seq0_i, &x2_next.seqi_n)
                    + RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u_seq123, &v.seqi_n);
                update_best(4, 0, result40);

                if v.id != 0 {
                    let y = &rv.nodes[pos2 + 1];
                    let result41 = RouteEval::eval3(data, &self.params, &u_pred.seq0_i, &v.seq1, &x2_next.seqi_n)
                        + RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u_seq123, &y.seqi_n);
                    update_best(4, 1, result41);

                    if y.id != 0 {
                        let y_next = &rv.nodes[pos2 + 2];
                        let v_seq123 = RouteEval::join2(data, &v.seq12, &y.seq1);
                        let result42 = RouteEval::eval3(data, &self.params, &u_pred.seq0_i, &v.seq12, &x2_next.seqi_n)
                            + RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u_seq123, &y_next.seqi_n);
                        let result43 = RouteEval::eval3(data, &self.params, &u_pred.seq0_i, &v.seq21, &x2_next.seqi_n)
                            + RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u_seq123, &y_next.seqi_n);
                        update_best(4, 2, result42);
                        update_best(4, 3, result43);

                        if y_next.id != 0 {
                            let y2_next = &rv.nodes[pos2 + 3];
                            let result44 = RouteEval::eval3(data, &self.params, &u_pred.seq0_i, &v_seq123, &x2_next.seqi_n)
                                + RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u_seq123, &y2_next.seqi_n);
                            update_best(4, 4, result44);
                        }
                    }
                }
            }
        }

        if best_i == 0 && best_j == 0 {
            return false;
        }

        let mut take_block = |route_idx: usize, pos: usize, kind: usize| -> Vec<Node> {
            let nodes = &mut self.routes[route_idx].nodes;
            match kind {
                0 => vec![],
                1 => {
                    let n1 = nodes.remove(pos);
                    vec![n1]
                }
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
                _ => vec![],
            }
        };

        let blk_from_r1 = take_block(r1, pos1, best_i);
        let blk_from_r2 = take_block(r2, pos2, best_j);

        let nodes1 = &mut self.routes[r1].nodes;
        for (k, node) in blk_from_r2.into_iter().enumerate() {
            nodes1.insert(pos1 + k, node);
        }
        let nodes2 = &mut self.routes[r2].nodes;
        for (k, node) in blk_from_r1.into_iter().enumerate() {
            nodes2.insert(pos2 + k, node);
        }

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
        let delta_demand = self.data.nd(v).demand - self.data.nd(u).demand;
        let new_load1 = self.routes[r1].load + delta_demand;
        let new_load2 = self.routes[r2].load - delta_demand;
        let new_pen1 = ((new_load1 - self.data.max_capacity).max(0) as i64) * self.params.penalty_capa as i64;
        let new_pen2 = ((new_load2 - self.data.max_capacity).max(0) as i64) * self.params.penalty_capa as i64;
        let cost_lb_r1_after_removal = (self.routes[r1].distance + dr1) as i64 + new_pen1;
        let cost_lb_r2_after_removal = (self.routes[r2].distance + dr2) as i64 + new_pen2;
        let mut lb_new_total = cost_lb_r1_after_removal + cost_lb_r2_after_removal;
        let old_total = self.routes[r1].cost + self.routes[r2].cost;
        if lb_new_total > old_total {
            return false;
        }

        let hole_v = self.data.dm(pu, v) + self.data.dm(v, nu) - self.data.dm(pu, nu);
        let mut best_ins_v = hole_v;
        for t in 1..route1_len {
            let a_id = self.routes[r1].nodes[t - 1].id;
            let b_id = self.routes[r1].nodes[t].id;
            if a_id == u || b_id == u {
                continue;
            }
            let delta = self.data.dm(a_id, v) + self.data.dm(v, b_id) - self.data.dm(a_id, b_id);
            if delta < best_ins_v {
                best_ins_v = delta;
            }
        }

        let hole_u = self.data.dm(pv, u) + self.data.dm(u, nv) - self.data.dm(pv, nv);
        let mut best_ins_u = hole_u;
        for t in 1..route2_len {
            let a_id = self.routes[r2].nodes[t - 1].id;
            let b_id = self.routes[r2].nodes[t].id;
            if a_id == v || b_id == v {
                continue;
            }
            let delta = self.data.dm(a_id, u) + self.data.dm(u, b_id) - self.data.dm(a_id, b_id);
            if delta < best_ins_u {
                best_ins_u = delta;
            }
        }

        lb_new_total += (best_ins_v + best_ins_u) as i64;
        if lb_new_total > old_total {
            return false;
        }

        Self::build_skip_one_scratch(
            self.data,
            &self.routes[r1],
            pos1,
            &mut self.scratch_a,
            &mut self.scratch_b,
        );

        Self::build_skip_one_scratch(
            self.data,
            &self.routes[r2],
            pos2,
            &mut self.scratch_c,
            &mut self.scratch_d,
        );

        let v_seq1 = self.routes[r2].nodes[pos2].seq1;
        let mut best_cost1 = i64::MAX / 4;
        let mut best_t1: usize = 1;
        for t in 1..route1_len {
            let cand = RouteEval::eval3(self.data, &self.params, &self.scratch_a[t], &v_seq1, &self.scratch_b[t]);
            if cand < best_cost1 {
                best_cost1 = cand;
                best_t1 = t;
            }
        }

        let u_seq1 = self.routes[r1].nodes[pos1].seq1;
        let mut best_cost2 = i64::MAX / 4;
        let mut best_t2: usize = 1;
        for t in 1..route2_len {
            let cand = RouteEval::eval3(self.data, &self.params, &self.scratch_c[t], &u_seq1, &self.scratch_d[t]);
            if cand < best_cost2 {
                best_cost2 = cand;
                best_t2 = t;
            }
        }

        if best_cost1 + best_cost2 >= old_total {
            return false;
        }

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

    fn run_2optstar_pair_scan(&mut self, r1: usize, r2: usize) -> bool {
        let len1 = self.routes[r1].nodes.len();
        let len2 = self.routes[r2].nodes.len();
        if len1 <= 2 || len2 <= 2 {
            return false;
        }

        let old_total = self.routes[r1].cost + self.routes[r2].cost;
        let (best_p1, best_p2, best_total) = {
            let route1 = &self.routes[r1];
            let route2 = &self.routes[r2];
            let mut best_total = old_total;
            let mut best_p1 = 0usize;
            let mut best_p2 = 0usize;

            for p1 in 1..len1 {
                let prefix1 = route1.nodes[p1 - 1].seq0_i;
                let suffix1 = route1.nodes[p1].seqi_n;
                for p2 in 1..len2 {
                    let cand_total =
                        RouteEval::eval2(self.data, &self.params, &prefix1, &route2.nodes[p2].seqi_n)
                        + RouteEval::eval2(self.data, &self.params, &route2.nodes[p2 - 1].seq0_i, &suffix1);
                    if cand_total < best_total {
                        best_total = cand_total;
                        best_p1 = p1;
                        best_p2 = p2;
                    }
                }
            }

            (best_p1, best_p2, best_total)
        };

        if best_total >= old_total {
            return false;
        }

        let mut suffix1 = self.routes[r1].nodes.split_off(best_p1);
        let mut suffix2 = self.routes[r2].nodes.split_off(best_p2);
        self.routes[r1].nodes.append(&mut suffix2);
        self.routes[r2].nodes.append(&mut suffix1);

        self.nb_moves += 1;
        self.update_route(r1);
        self.update_route(r2);
        let new_total = self.routes[r1].cost + self.routes[r2].cost;
        self.cost += new_total - old_total;
        true
    }

    fn run_swapstar_route_scan(&mut self, r1: usize, pos1: usize, r2: usize) -> bool {
        let route1_len = self.routes[r1].nodes.len();
        let route2_len = self.routes[r2].nodes.len();
        if route1_len <= 2 || route2_len <= 2 {
            return false;
        }

        let old_total = self.routes[r1].cost + self.routes[r2].cost;
        let u = self.routes[r1].nodes[pos1].id;
        let u_seq1 = self.routes[r1].nodes[pos1].seq1;
        let (pu, nu) = (self.routes[r1].nodes[pos1 - 1].id, self.routes[r1].nodes[pos1 + 1].id);
        let dr1 = self.data.dm(pu, nu) - self.data.dm(pu, u) - self.data.dm(u, nu);
        let load1 = self.routes[r1].load;
        let distance1 = self.routes[r1].distance;
        let demand_u = self.data.nd(u).demand;

        Self::build_skip_one_scratch(
            self.data,
            &self.routes[r1],
            pos1,
            &mut self.scratch_a,
            &mut self.scratch_b,
        );

        let mut best_total = old_total;
        let mut best_pos2: Option<usize> = None;
        let mut best_t1: usize = 1;
        let mut best_t2: usize = 1;

        for pos2 in 1..(route2_len - 1) {
            let v = self.routes[r2].nodes[pos2].id;
            let (pv, nv) = (self.routes[r2].nodes[pos2 - 1].id, self.routes[r2].nodes[pos2 + 1].id);

            let dr2 = self.data.dm(pv, nv) - self.data.dm(pv, v) - self.data.dm(v, nv);
            let delta_demand = self.data.nd(v).demand - demand_u;
            let new_load1 = load1 + delta_demand;
            let new_load2 = self.routes[r2].load - delta_demand;
            let new_pen1 = ((new_load1 - self.data.max_capacity).max(0) as i64) * self.params.penalty_capa as i64;
            let new_pen2 = ((new_load2 - self.data.max_capacity).max(0) as i64) * self.params.penalty_capa as i64;
            let cost_lb_r1_after_removal = (distance1 + dr1) as i64 + new_pen1;
            let cost_lb_r2_after_removal = (self.routes[r2].distance + dr2) as i64 + new_pen2;
            let mut lb_new_total = cost_lb_r1_after_removal + cost_lb_r2_after_removal;
            if lb_new_total >= best_total {
                continue;
            }

            let hole_v = self.data.dm(pu, v) + self.data.dm(v, nu) - self.data.dm(pu, nu);
            let mut best_ins_v = hole_v;
            for t in 1..route1_len {
                let a_id = self.routes[r1].nodes[t - 1].id;
                let b_id = self.routes[r1].nodes[t].id;
                if a_id == u || b_id == u {
                    continue;
                }
                let delta = self.data.dm(a_id, v) + self.data.dm(v, b_id) - self.data.dm(a_id, b_id);
                if delta < best_ins_v {
                    best_ins_v = delta;
                }
            }

            let hole_u = self.data.dm(pv, u) + self.data.dm(u, nv) - self.data.dm(pv, nv);
            let mut best_ins_u = hole_u;
            for t in 1..route2_len {
                let a_id = self.routes[r2].nodes[t - 1].id;
                let b_id = self.routes[r2].nodes[t].id;
                if a_id == v || b_id == v {
                    continue;
                }
                let delta = self.data.dm(a_id, u) + self.data.dm(u, b_id) - self.data.dm(a_id, b_id);
                if delta < best_ins_u {
                    best_ins_u = delta;
                }
            }

            lb_new_total += (best_ins_v + best_ins_u) as i64;
            if lb_new_total >= best_total {
                continue;
            }

            let v_seq1 = self.routes[r2].nodes[pos2].seq1;
            let mut cand_cost1 = i64::MAX / 4;
            let mut cand_t1 = 1usize;
            for t in 1..route1_len {
                let cand = RouteEval::eval3(self.data, &self.params, &self.scratch_a[t], &v_seq1, &self.scratch_b[t]);
                if cand < cand_cost1 {
                    cand_cost1 = cand;
                    cand_t1 = t;
                }
            }

            Self::build_skip_one_scratch(
                self.data,
                &self.routes[r2],
                pos2,
                &mut self.scratch_c,
                &mut self.scratch_d,
            );

            let mut cand_cost2 = i64::MAX / 4;
            let mut cand_t2 = 1usize;
            for t in 1..route2_len {
                let cand = RouteEval::eval3(self.data, &self.params, &self.scratch_c[t], &u_seq1, &self.scratch_d[t]);
                if cand < cand_cost2 {
                    cand_cost2 = cand;
                    cand_t2 = t;
                }
            }

            let cand_total = cand_cost1 + cand_cost2;
            if cand_total < best_total {
                best_total = cand_total;
                best_pos2 = Some(pos2);
                best_t1 = cand_t1;
                best_t2 = cand_t2;
            }
        }

        if let Some(pos2) = best_pos2 {
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
        } else {
            false
        }
    }

    pub fn runls(
        &mut self,
        routes: &mut Vec<Vec<usize>>,
        rng: &mut SmallRng,
        params: &Config,
        is_repair: bool,
        factor: usize,
    ) {
        self.params = *params;

        if !is_repair {
            self.load_from_routes(routes);
        } else {
            self.params.penalty_tw = (factor * self.params.penalty_tw).min(10_000);
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

                let mut r1 = self.node_route[c1];
                let mut pos1 = self.node_pos[c1];

                let mut changed = true;
                while changed {
                    changed = false;
                    let mod_r1 = self.when_last_modified[r1];
                    let neighbors_before_c1 = self.neighbors_before[c1].clone();
                    for c2 in neighbors_before_c1 {
                        let r2 = self.node_route[c2];
                        if r1 == r2 {
                            continue;
                        }

                        if mod_r1.max(self.when_last_modified[r2]) <= last_tested {
                            continue;
                        }

                        let pos2 = self.node_pos[c2];

                        if self.run_inter_route(r1, pos1, r2, pos2 + 1) {
                            improved = true;
                            changed = true;
                            r1 = self.node_route[c1];
                            pos1 = self.node_pos[c1];
                            break;
                        }

                        if pos1 == 1 && self.run_inter_route(r2, pos2, r1, pos1) {
                            improved = true;
                            changed = true;
                            r1 = self.node_route[c1];
                            pos1 = self.node_pos[c1];
                            break;
                        }

                        if self.run_2optstar(r1, pos1, r2, pos2 + 1) {
                            improved = true;
                            changed = true;
                            r1 = self.node_route[c1];
                            pos1 = self.node_pos[c1];
                            break;
                        }
                    }
                }

                let mut mod_r1 = self.when_last_modified[r1];
                let swap_neighbors_c1 = self.neighbors_capacity_swap[c1].clone();
                let swap_len = swap_neighbors_c1.len();
                if swap_len > 0 {
                    let start_s = rng.gen_range(0..swap_len);
                    let mut idx_s = start_s;
                    for _ in 0..swap_len {
                        let c2 = swap_neighbors_c1[idx_s];
                        idx_s += 1;
                        if idx_s == swap_len {
                            idx_s = 0;
                        }
                        let r2 = self.node_route[c2];
                        if r1 == r2 {
                            continue;
                        }

                        if c1 < c2 || mod_r1.max(self.when_last_modified[r2]) <= last_tested {
                            continue;
                        }

                        let pos2 = self.node_pos[c2];
                        if self.run_swapstar(r1, pos1, r2, pos2) {
                            improved = true;
                            r1 = self.node_route[c1];
                            pos1 = self.node_pos[c1];
                            mod_r1 = self.when_last_modified[r1];
                            break;
                        }
                    }
                }

                if mod_r1 > last_tested {
                    let stressed_r1 = self.routes[r1].tw > 0 || self.routes[r1].load > self.data.max_capacity;
                    let len1 = self.routes[r1].nodes.len();
                    let route_seed_neighbors = self.neighbors_before[c1].clone();
                    let mut target_routes: Vec<usize> = Vec::with_capacity(2);
                    for &c2 in route_seed_neighbors.iter().take(10) {
                        let r2 = self.node_route[c2];
                        if r1 == r2 || target_routes.contains(&r2) {
                            continue;
                        }
                        let len2 = self.routes[r2].nodes.len();
                        if len2 <= 2 {
                            continue;
                        }
                        if mod_r1.max(self.when_last_modified[r2]) <= last_tested {
                            continue;
                        }
                        let stressed_pair = stressed_r1 || self.routes[r2].tw > 0 || self.routes[r2].load > self.data.max_capacity;
                        let len_limit = if stressed_pair { 56usize } else { 36usize };
                        if len1 > len_limit || len2 > len_limit {
                            continue;
                        }
                        let pair_budget = if stressed_pair { 1800usize } else { 900usize };
                        if len1.saturating_mul(len2) > pair_budget {
                            continue;
                        }
                        target_routes.push(r2);
                        if target_routes.len() == 2 {
                            break;
                        }
                    }

                    for r2 in target_routes {
                        if self.run_2optstar_pair_scan(r1, r2) {
                            improved = true;
                            r1 = self.node_route[c1];
                            pos1 = self.node_pos[c1];
                            mod_r1 = self.when_last_modified[r1];
                            break;
                        }

                        if self.run_swapstar_route_scan(r1, pos1, r2) {
                            improved = true;
                            r1 = self.node_route[c1];
                            pos1 = self.node_pos[c1];
                            mod_r1 = self.when_last_modified[r1];
                            break;
                        }
                    }
                }

                if loop_id > 1 && (loop_id == 2 || mod_r1 > last_tested) {
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

                if mod_r1 > last_tested {
                    if self.run_intra_route_relocate(r1, pos1) {
                        improved = true;
                        r1 = self.node_route[c1];
                        pos1 = self.node_pos[c1];
                    }
                    if self.run_intra_route_swap_right(r1, pos1) {
                        improved = true;
                        r1 = self.node_route[c1];
                        pos1 = self.node_pos[c1];
                    }
                    if self.run_2opt(r1, pos1) {
                        improved = true;
                        r1 = self.node_route[c1];
                        pos1 = self.node_pos[c1];
                    }
                    if self.run_intra_route_oropt(r1, pos1, 2) {
                        improved = true;
                        r1 = self.node_route[c1];
                        pos1 = self.node_pos[c1];
                    }
                    if self.params.allow_swap3 && self.run_intra_route_oropt(r1, pos1, 3) {
                        improved = true;
                    }
                }
            }
        }
        self.write_back_to_routes(routes);
    }
}