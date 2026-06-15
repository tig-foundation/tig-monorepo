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
    #[allow(dead_code)]
    scratch_a: Vec<RouteEval>,
    #[allow(dead_code)]
    scratch_b: Vec<RouteEval>,
    #[allow(dead_code)]
    scratch_c: Vec<RouteEval>,
    #[allow(dead_code)]
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

    #[inline]
    fn customer_reinsertion_hardness(&self, customer: usize) -> i64 {
        let tw_width = (self.data.end_tw[customer] - self.data.start_tw[customer]).max(0) as i64;
        let roundtrip = (self.data.dm(0, customer) + self.data.dm(customer, 0)) as i64;
        let neighbors = &self.neighbors_before[customer];
        let alt_take = neighbors.len().min(3);
        let mut alt_dist = 0i64;
        if alt_take == 0 {
            alt_dist = self.data.dm(0, customer) as i64;
        } else {
            for &other in neighbors.iter().take(alt_take) {
                alt_dist += self.data.dm(customer, other) as i64;
            }
            alt_dist /= alt_take as i64;
        }
        let demand_term =
            64_i64 * self.data.demands[customer] as i64 / max(1_i64, self.data.max_capacity as i64);

        6 * roundtrip
            + 4 * alt_dist
            + 8 * self.data.service_times[customer] as i64
            + demand_term
            - 2 * tw_width
    }

    #[inline]
    fn customer_route_cohesion(
        &self,
        customer: usize,
        route_owner_id: usize,
        owner_route: &[usize],
    ) -> i64 {
        let neighbors = &self.neighbors_before[customer];
        let take = neighbors.len().min(6);
        if take == 0 {
            return 0;
        }

        let mut same_hits = 0i64;
        let mut external_hits = 0i64;
        let mut best_same = i64::MAX / 8;
        let mut best_external = i64::MAX / 8;

        for &other in neighbors.iter().take(take) {
            let d = self.data.dm(customer, other) as i64;
            let owner = owner_route[other];
            if owner == route_owner_id {
                same_hits += 1;
                if d < best_same {
                    best_same = d;
                }
            } else if owner != usize::MAX {
                external_hits += 1;
                if d < best_external {
                    best_external = d;
                }
            }
        }

        let mut bonus = 6 * same_hits - 4 * external_hits;
        if best_same < i64::MAX / 8 && best_external < i64::MAX / 8 {
            bonus += ((best_external - best_same).max(0)) / 10;
        } else if best_same < i64::MAX / 8 {
            bonus += 4;
        } else if best_external < i64::MAX / 8 {
            bonus -= 4;
        }
        bonus
    }

    fn route_retention_score(
        &self,
        route: &[usize],
        route_owner_id: usize,
        customer_hardness: &[i64],
        owner_route: &[usize],
    ) -> i64 {
        let customers = route.len().saturating_sub(2);
        if customers == 0 {
            return i64::MAX / 8;
        }

        let customers_i64 = customers as i64;
        let route_cost = self.eval_route_ids(route);
        let mut route_distance = 0i64;
        let mut hardest1 = 0i64;
        let mut hardest2 = 0i64;
        let mut hard_sum = 0i64;
        let mut separate_distance = 0i64;
        let mut cohesion_bonus = 0i64;

        for w in route.windows(2) {
            route_distance += self.data.dm(w[0], w[1]) as i64;
        }
        for &customer in &route[1..route.len() - 1] {
            let h = customer_hardness[customer];
            hard_sum += h;
            if h >= hardest1 {
                hardest2 = hardest1;
                hardest1 = h;
            } else if h > hardest2 {
                hardest2 = h;
            }
            separate_distance += (self.data.dm(0, customer) + self.data.dm(customer, 0)) as i64;
            cohesion_bonus += self.customer_route_cohesion(customer, route_owner_id, owner_route);
        }

        let compactness = route_distance.saturating_mul(1024) / customers_i64;
        let penalty_like = route_cost.saturating_sub(route_distance);
        let difficulty = hard_sum / customers_i64 + hardest1 + hardest2 / 2;
        let route_synergy = separate_distance.saturating_sub(route_distance);
        let cohesion_value = route_synergy.saturating_mul(12) / customers_i64
            + cohesion_bonus.saturating_mul(48) / customers_i64;

        compactness
            .saturating_add(penalty_like.saturating_mul(128) / customers_i64)
            .saturating_sub(difficulty.saturating_mul(32))
            .saturating_sub(cohesion_value)
            .saturating_sub(customers_i64)
    }

    fn reinsert_overflow_customers(&mut self, pending: Vec<usize>) {
        if pending.is_empty() {
            return;
        }

        let mut pending: Vec<(usize, RouteEval, i64)> = pending
            .into_iter()
            .map(|customer| {
                let hardness = 2 * self.data.dm(0, customer) as i64
                    + self.data.service_times[customer] as i64
                    - (self.data.end_tw[customer] - self.data.start_tw[customer]) as i64;
                (customer, RouteEval::singleton(self.data, customer), hardness)
            })
            .collect();

        pending.sort_by(|a, b| b.2.cmp(&a.2));

        while !pending.is_empty() {
            let scan_limit = pending.len().min(48);
            let mut best_choice: Option<(usize, usize, usize, i64, i64, bool, i64)> = None;

            for idx in 0..scan_limit {
                let (_customer, seq1, hardness) = pending[idx];

                let mut best_rid = 0usize;
                let mut best_pos = 1usize;
                let mut best_delta = i64::MAX / 4;
                let mut second_delta = i64::MAX / 4;
                let mut best_empty: Option<(usize, usize, i64)> = None;

                for rid in 0..self.routes.len() {
                    let route = &self.routes[rid];
                    let nodes = &route.nodes;
                    let route_len = nodes.len();
                    let route_cost = route.cost;
                    for pos in 1..route_len {
                        let cand = RouteEval::eval3(
                            self.data,
                            &self.params,
                            &nodes[pos - 1].seq0_i,
                            &seq1,
                            &nodes[pos].seqi_n,
                        );
                        let delta = cand - route_cost;

                        if route_len == 2 {
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
            let (customer, seq1, _) = pending.remove(idx);
            let old_cost = self.routes[rid].cost;

            let mut node = Node::new(customer);
            node.seq1 = seq1;
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
            let mut customer_hardness = vec![0i64; n];
            for customer in 1..n {
                customer_hardness[customer] = self.customer_reinsertion_hardness(customer);
            }

            let mut owner_route = vec![usize::MAX; n];
            for (idx, route) in routes.iter().enumerate() {
                if route.len() <= 2 {
                    continue;
                }
                for &customer in &route[1..route.len() - 1] {
                    owner_route[customer] = idx;
                }
            }

            let mut scored_routes: Vec<(i64, usize)> = Vec::with_capacity(routes.len());
            for (idx, route) in routes.iter().enumerate() {
                if route.len() > 2 {
                    let score = self.route_retention_score(route, idx, &customer_hardness, &owner_route);
                    scored_routes.push((score, idx));
                }
            }

            let keep_nonempty = scored_routes.len().min(fleet);
            if keep_nonempty == scored_routes.len() {
                scored_routes.sort_unstable();
                for &(_, idx) in &scored_routes {
                    src.push(routes[idx].clone());
                }
            } else if keep_nonempty == 0 {
                for &(_, idx) in &scored_routes {
                    let route = &routes[idx];
                    overflow_customers.extend_from_slice(&route[1..route.len() - 1]);
                }
            } else {
                scored_routes.select_nth_unstable_by(keep_nonempty, |a, b| a.cmp(b));
                let (kept, dropped) = scored_routes.split_at_mut(keep_nonempty);
                kept.sort_unstable();
                for &(_, idx) in kept.iter() {
                    src.push(routes[idx].clone());
                }
                for &(_, idx) in dropped.iter() {
                    let route = &routes[idx];
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
        let nodes = &mut r.nodes;
        let len = nodes.len();

        let id0 = nodes[0].id;
        let mut acc_fwd = nodes[0].seq1;
        nodes[0].seq0_i = acc_fwd;
        node_route[id0] = rid;
        node_pos[id0] = 0;
        for pos in 1..len {
            let node = &mut nodes[pos];
            let id = node.id;
            acc_fwd = RouteEval::join2(data, &acc_fwd, &node.seq1);
            node.seq0_i = acc_fwd;
            node_route[id] = rid;
            node_pos[id] = pos;
        }

        let last_idx = len - 1;
        nodes[last_idx].seqi_n = nodes[last_idx].seq1;
        for pos in (0..last_idx).rev() {
            let next_seq1 = nodes[pos + 1].seq1;
            let next_seqi_n = nodes[pos + 1].seqi_n;
            let cur_seq1 = nodes[pos].seq1;
            let node = &mut nodes[pos];
            node.seq12 = RouteEval::join2(data, &cur_seq1, &next_seq1);
            node.seq21 = RouteEval::join2(data, &next_seq1, &cur_seq1);
            node.seqi_n = RouteEval::join2(data, &cur_seq1, &next_seqi_n);
        }

        let end = nodes[last_idx].seq0_i;
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

    #[inline]
    fn exact_best_insertion_with_skip(
        data: &Instance,
        params: &Config,
        route: &Route,
        skip_pos: usize,
        insert_seq: &RouteEval,
    ) -> (i64, usize) {
        let nodes = &route.nodes;
        let len = nodes.len();

        let prefix_fixed = nodes[skip_pos - 1].seq0_i;
        let suffix_fixed = nodes[skip_pos + 1].seqi_n;
        let mut best_cost = RouteEval::eval3(data, params, &prefix_fixed, insert_seq, &suffix_fixed);
        let mut best_t = skip_pos;

        if skip_pos > 1 {
            let mut mid_seq = nodes[skip_pos - 1].seq1;
            for t in (1..skip_pos).rev() {
                let cand = RouteEval::eval4(
                    data,
                    params,
                    &nodes[t - 1].seq0_i,
                    insert_seq,
                    &mid_seq,
                    &suffix_fixed,
                );
                if cand <= best_cost {
                    best_cost = cand;
                    best_t = t;
                }
                if t > 1 {
                    mid_seq = RouteEval::join2(data, &nodes[t - 1].seq1, &mid_seq);
                }
            }
        }

        if skip_pos + 2 < len {
            let mut mid_seq = nodes[skip_pos + 1].seq1;
            for t in (skip_pos + 2)..len {
                let cand = RouteEval::eval4(
                    data,
                    params,
                    &prefix_fixed,
                    &mid_seq,
                    insert_seq,
                    &nodes[t].seqi_n,
                );
                if cand < best_cost {
                    best_cost = cand;
                    best_t = t;
                }
                if t < len - 1 {
                    mid_seq = RouteEval::join2(data, &mid_seq, &nodes[t].seq1);
                }
            }
        }

        (best_cost, best_t)
    }

    pub fn run_intra_route_relocate(&mut self, r1: usize, pos1: usize) -> bool {
        let route = &self.routes[r1];
        let len = route.nodes.len();
        if len < pos1 + 4 {
            return false;
        }

        let old_cost = route.cost;
        let mut best_cost = old_cost;
        let mut best_pos: Option<usize> = None;
        let seq1_pos1 = route.nodes[pos1].seq1;

        if pos1 > 1 {
            let suffix_fixed = route.nodes[pos1 + 1].seqi_n;
            let mut mid_seq = route.nodes[pos1 - 1].seq1;
            for t in (1..pos1).rev() {
                let cand = RouteEval::eval4(
                    self.data,
                    &self.params,
                    &route.nodes[t - 1].seq0_i,
                    &seq1_pos1,
                    &mid_seq,
                    &suffix_fixed,
                );
                if cand < best_cost || (cand == best_cost && best_cost < old_cost) {
                    best_cost = cand;
                    best_pos = Some(t);
                }
                if t > 1 {
                    mid_seq = RouteEval::join2(self.data, &route.nodes[t - 1].seq1, &mid_seq);
                }
            }
        }

        if pos1 + 2 < len {
            let prefix_fixed = route.nodes[pos1 - 1].seq0_i;
            let mut mid_seq = route.nodes[pos1 + 1].seq1;
            for t in (pos1 + 2)..len {
                let cand = RouteEval::eval4(
                    self.data,
                    &self.params,
                    &prefix_fixed,
                    &mid_seq,
                    &seq1_pos1,
                    &route.nodes[t].seqi_n,
                );
                if cand < best_cost {
                    best_cost = cand;
                    best_pos = Some(t);
                }
                if t < len - 1 {
                    mid_seq = RouteEval::join2(self.data, &mid_seq, &route.nodes[t].seq1);
                }
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
                + RouteEval::eval3(
                    data,
                    &self.params,
                    &v_pred.seq0_i,
                    &u.seq1,
                    &rv.nodes[pos2 + 1].seqi_n,
                );
            update_best(1, 1, result11);
        }

        if x.id != 0 {
            let x_next = &ru.nodes[pos1 + 2];
            let remove_r1_len2 = RouteEval::eval2(data, &self.params, &u_pred.seq0_i, &x_next.seqi_n);
            let insert_u12 = RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u.seq12, &v.seqi_n);
            let insert_u21 = RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u.seq21, &v.seqi_n);
            update_best(2, 0, remove_r1_len2 + insert_u12);
            update_best(3, 0, remove_r1_len2 + insert_u21);

            if v.id != 0 {
                let y = &rv.nodes[pos2 + 1];
                let insert_v1_in_r1 =
                    RouteEval::eval3(data, &self.params, &u_pred.seq0_i, &v.seq1, &x_next.seqi_n);
                let insert_u12_after_v =
                    RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u.seq12, &y.seqi_n);
                let insert_u21_after_v =
                    RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u.seq21, &y.seqi_n);
                update_best(2, 1, insert_v1_in_r1 + insert_u12_after_v);
                update_best(3, 1, insert_v1_in_r1 + insert_u21_after_v);

                if y.id != 0 {
                    let y_next = &rv.nodes[pos2 + 2];
                    let insert_v12_in_r1 =
                        RouteEval::eval3(data, &self.params, &u_pred.seq0_i, &v.seq12, &x_next.seqi_n);
                    let insert_v21_in_r1 =
                        RouteEval::eval3(data, &self.params, &u_pred.seq0_i, &v.seq21, &x_next.seqi_n);
                    let insert_u12_after_v12 =
                        RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u.seq12, &y_next.seqi_n);
                    let insert_u21_after_v12 =
                        RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u.seq21, &y_next.seqi_n);

                    update_best(2, 2, insert_v12_in_r1 + insert_u12_after_v12);
                    update_best(3, 2, insert_v12_in_r1 + insert_u21_after_v12);
                    update_best(2, 3, insert_v21_in_r1 + insert_u12_after_v12);
                    update_best(3, 3, insert_v21_in_r1 + insert_u21_after_v12);
                }
            }

            if x_next.id != 0 && self.params.allow_swap3 {
                let x2_next = &ru.nodes[pos1 + 3];
                let u_seq123 = RouteEval::join2(data, &u.seq12, &x_next.seq1);
                let u_seq321 = RouteEval::join2(data, &x_next.seq1, &u.seq21);
                let remove_r1_len3 = RouteEval::eval2(data, &self.params, &u_pred.seq0_i, &x2_next.seqi_n);

                let insert_u123 = RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u_seq123, &v.seqi_n);
                let insert_u321 = RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u_seq321, &v.seqi_n);
                update_best(4, 0, remove_r1_len3 + insert_u123);
                update_best(5, 0, remove_r1_len3 + insert_u321);

                if v.id != 0 {
                    let y = &rv.nodes[pos2 + 1];
                    let insert_v1_in_r1 =
                        RouteEval::eval3(data, &self.params, &u_pred.seq0_i, &v.seq1, &x2_next.seqi_n);
                    let insert_u123_after_v =
                        RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u_seq123, &y.seqi_n);
                    let insert_u321_after_v =
                        RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u_seq321, &y.seqi_n);

                    update_best(4, 1, insert_v1_in_r1 + insert_u123_after_v);
                    update_best(5, 1, insert_v1_in_r1 + insert_u321_after_v);

                    if y.id != 0 {
                        let y_next = &rv.nodes[pos2 + 2];
                        let insert_v12_in_r1 =
                            RouteEval::eval3(data, &self.params, &u_pred.seq0_i, &v.seq12, &x2_next.seqi_n);
                        let insert_v21_in_r1 =
                            RouteEval::eval3(data, &self.params, &u_pred.seq0_i, &v.seq21, &x2_next.seqi_n);
                        let insert_u123_after_v12 =
                            RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u_seq123, &y_next.seqi_n);
                        let insert_u321_after_v12 =
                            RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u_seq321, &y_next.seqi_n);

                        update_best(4, 2, insert_v12_in_r1 + insert_u123_after_v12);
                        update_best(5, 2, insert_v12_in_r1 + insert_u321_after_v12);
                        update_best(4, 3, insert_v21_in_r1 + insert_u123_after_v12);
                        update_best(5, 3, insert_v21_in_r1 + insert_u321_after_v12);

                        if y_next.id != 0 {
                            let y2_next = &rv.nodes[pos2 + 3];
                            let v_seq123 = RouteEval::join2(data, &v.seq12, &y_next.seq1);
                            let v_seq321 = RouteEval::join2(data, &y_next.seq1, &v.seq21);
                            let insert_v123_in_r1 =
                                RouteEval::eval3(data, &self.params, &u_pred.seq0_i, &v_seq123, &x2_next.seqi_n);
                            let insert_v321_in_r1 =
                                RouteEval::eval3(data, &self.params, &u_pred.seq0_i, &v_seq321, &x2_next.seqi_n);
                            let insert_u123_after_v123 =
                                RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u_seq123, &y2_next.seqi_n);
                            let insert_u321_after_v123 =
                                RouteEval::eval3(data, &self.params, &v_pred.seq0_i, &u_seq321, &y2_next.seqi_n);

                            update_best(4, 4, insert_v123_in_r1 + insert_u123_after_v123);
                            update_best(5, 4, insert_v123_in_r1 + insert_u321_after_v123);
                            update_best(4, 5, insert_v321_in_r1 + insert_u123_after_v123);
                            update_best(5, 5, insert_v321_in_r1 + insert_u321_after_v123);
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
                5 => {
                    let n1 = nodes.remove(pos);
                    let n2 = nodes.remove(pos);
                    let n3 = nodes.remove(pos);
                    vec![n3, n2, n1]
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

        let v_seq1 = self.routes[r2].nodes[pos2].seq1;
        let (best_cost1, best_t1) = Self::exact_best_insertion_with_skip(
            self.data,
            &self.params,
            &self.routes[r1],
            pos1,
            &v_seq1,
        );

        let u_seq1 = self.routes[r1].nodes[pos1].seq1;
        let (best_cost2, best_t2) = Self::exact_best_insertion_with_skip(
            self.data,
            &self.params,
            &self.routes[r2],
            pos2,
            &u_seq1,
        );

        if best_cost1 + best_cost2 >= old_total {
            return false;
        }

        let node_u = self.routes[r1].nodes.remove(pos1);
        let node_v = self.routes[r2].nodes.remove(pos2);

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

        let mut top_edge1_delta = i32::MAX;
        let mut top_edge1_pos = 0usize;
        let mut top_edge2_delta = i32::MAX;
        let mut top_edge2_pos = 0usize;
        let mut top_edge3_delta = i32::MAX;
        let mut top_edge3_pos = 0usize;
        for t in 1..route2_len {
            let a_id = self.routes[r2].nodes[t - 1].id;
            let b_id = self.routes[r2].nodes[t].id;
            let delta = self.data.dm(a_id, u) + self.data.dm(u, b_id) - self.data.dm(a_id, b_id);
            if delta < top_edge1_delta {
                top_edge3_delta = top_edge2_delta;
                top_edge3_pos = top_edge2_pos;
                top_edge2_delta = top_edge1_delta;
                top_edge2_pos = top_edge1_pos;
                top_edge1_delta = delta;
                top_edge1_pos = t;
            } else if delta < top_edge2_delta {
                top_edge3_delta = top_edge2_delta;
                top_edge3_pos = top_edge2_pos;
                top_edge2_delta = delta;
                top_edge2_pos = t;
            } else if delta < top_edge3_delta {
                top_edge3_delta = delta;
                top_edge3_pos = t;
            }
        }

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
            let invalid1 = pos2;
            let invalid2 = pos2 + 1;
            let mut best_ins_u = hole_u;
            if top_edge1_pos != invalid1 && top_edge1_pos != invalid2 {
                if top_edge1_delta < best_ins_u {
                    best_ins_u = top_edge1_delta;
                }
            } else if top_edge2_pos != invalid1 && top_edge2_pos != invalid2 {
                if top_edge2_delta < best_ins_u {
                    best_ins_u = top_edge2_delta;
                }
            } else if top_edge3_pos != invalid1 && top_edge3_pos != invalid2 {
                if top_edge3_delta < best_ins_u {
                    best_ins_u = top_edge3_delta;
                }
            }

            lb_new_total += (best_ins_v + best_ins_u) as i64;
            if lb_new_total >= best_total {
                continue;
            }

            let v_seq1 = self.routes[r2].nodes[pos2].seq1;
            let (cand_cost1, cand_t1) = Self::exact_best_insertion_with_skip(
                self.data,
                &self.params,
                &self.routes[r1],
                pos1,
                &v_seq1,
            );

            let (cand_cost2, cand_t2) = Self::exact_best_insertion_with_skip(
                self.data,
                &self.params,
                &self.routes[r2],
                pos2,
                &u_seq1,
            );

            let cand_total = cand_cost1 + cand_cost2;
            if cand_total < best_total {
                best_total = cand_total;
                best_pos2 = Some(pos2);
                best_t1 = cand_t1;
                best_t2 = cand_t2;
            }
        }

        if let Some(pos2) = best_pos2 {
            let node_u = self.routes[r1].nodes.remove(pos1);
            let node_v = self.routes[r2].nodes.remove(pos2);

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

        let max_capacity = self.data.max_capacity;
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
                    let neighbors_ptr = self.neighbors_before[c1].as_ptr();
                    let neighbors_len = self.neighbors_before[c1].len();
                    let node_route_ptr = self.node_route.as_ptr();
                    let node_pos_ptr = self.node_pos.as_ptr();
                    let when_last_modified_ptr = self.when_last_modified.as_ptr();

                    let mut idx = 0usize;
                    while idx < neighbors_len {
                        let c2 = unsafe { *neighbors_ptr.add(idx) };
                        idx += 1;

                        let r2 = unsafe { *node_route_ptr.add(c2) };
                        if r1 == r2 {
                            continue;
                        }

                        let mod_r2 = unsafe { *when_last_modified_ptr.add(r2) };
                        if mod_r1.max(mod_r2) <= last_tested {
                            continue;
                        }

                        let pos2 = unsafe { *node_pos_ptr.add(c2) };

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
                let swap_neighbors_ptr = self.neighbors_capacity_swap[c1].as_ptr();
                let swap_len = self.neighbors_capacity_swap[c1].len();
                if swap_len > 0 {
                    let start_s = rng.gen_range(0..swap_len);
                    let node_route_ptr = self.node_route.as_ptr();
                    let node_pos_ptr = self.node_pos.as_ptr();
                    let when_last_modified_ptr = self.when_last_modified.as_ptr();
                    let mut idx_s = start_s;
                    for _ in 0..swap_len {
                        let c2 = unsafe { *swap_neighbors_ptr.add(idx_s) };
                        idx_s += 1;
                        if idx_s == swap_len {
                            idx_s = 0;
                        }
                        let r2 = unsafe { *node_route_ptr.add(c2) };
                        if r1 == r2 {
                            continue;
                        }

                        let mod_r2 = unsafe { *when_last_modified_ptr.add(r2) };
                        if c1 < c2 || mod_r1.max(mod_r2) <= last_tested {
                            continue;
                        }

                        let pos2 = unsafe { *node_pos_ptr.add(c2) };
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
                    let node_route_ptr = self.node_route.as_ptr();
                    let when_last_modified_ptr = self.when_last_modified.as_ptr();
                    let routes_ptr = self.routes.as_ptr();
                    let route_seed_ptr = self.neighbors_before[c1].as_ptr();
                    let route_seed_len = self.neighbors_before[c1].len().min(10);
                    let (stressed_r1, len1) = unsafe {
                        let route1 = &*routes_ptr.add(r1);
                        (route1.tw > 0 || route1.load > max_capacity, route1.nodes.len())
                    };
                    let mut target_routes: Vec<usize> = Vec::with_capacity(2);
                    let mut idx = 0usize;
                    while idx < route_seed_len {
                        let c2 = unsafe { *route_seed_ptr.add(idx) };
                        idx += 1;

                        let r2 = unsafe { *node_route_ptr.add(c2) };
                        if r1 == r2 || target_routes.contains(&r2) {
                            continue;
                        }

                        let route2 = unsafe { &*routes_ptr.add(r2) };
                        let len2 = route2.nodes.len();
                        if len2 <= 2 {
                            continue;
                        }

                        let mod_r2 = unsafe { *when_last_modified_ptr.add(r2) };
                        if mod_r1.max(mod_r2) <= last_tested {
                            continue;
                        }

                        let stressed_pair = stressed_r1 || route2.tw > 0 || route2.load > max_capacity;
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