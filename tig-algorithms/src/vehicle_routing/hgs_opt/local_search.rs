use super::problem::Problem;
use super::params::Params;
use super::sequence::Sequence;
use std::cmp::min;
use rand::{rngs::SmallRng, Rng};
use rand::seq::SliceRandom;

#[derive(Clone, Debug, Default)]
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

#[derive(Clone, Debug, Default)]
pub struct Route {
    cost: i64,
    distance: i32,
    load: i32,
    tw: i32,
    nodes: Vec<Node>,
}

impl Route {
    /// Build the node container; metrics are computed later by `update_route`.
    #[inline]
    fn new(ids: &[usize]) -> Self {
        Self {
            nodes: ids.iter().copied().map(Node::new).collect(),
            ..Default::default()
        }
    }
}

pub struct LocalSearch<'a> {
    pub data: &'a Problem,
    pub neighbors_before: Vec<Vec<usize>>,
    pub neighbors_capacity_swap: Vec<Vec<usize>>,
    pub params: Params,
    pub cost: i64,
    pub routes: Vec<Route>,
    pub node_route: Vec<usize>,
    pub node_pos: Vec<usize>,
    pub empty_routes: Vec<usize>,
    pub when_last_modified: Vec<usize>,   // per route
    pub when_last_tested: Vec<usize>,     // per customer id
    pub nb_moves: usize,                  // monotone counter of applied moves
    pub depot_seq: Sequence,              // cached depot singleton
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
        for i in 1..n {
            let di = data.demands[i];
            let mut prox: Vec<(i32, usize)> = Vec::with_capacity(n.saturating_sub(1));
            for j in 1..n {
                if j == i { continue; }
                if (data.demands[j] - di).abs() <= 1 {
                    let dij = data.dm(i,j); // distance “i->j” as proximity
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
            when_last_modified: Vec::new(),
            when_last_tested: vec![0; n],
            nb_moves: 0,
            depot_seq: Sequence::singleton(data, 0),
        }
    }

    fn load_from_routes(&mut self, routes: &Vec<Vec<usize>>) {
        let n = self.data.nb_nodes;
        let fleet = self.data.nb_vehicles;

        // Compact any extra routes into the last regular route
        let mut src: Vec<Vec<usize>> = Vec::new();

        if routes.len() <= fleet {
            src.extend(routes.iter().cloned());
        } else {
            // Keep the first (fleet - 1) routes as-is
            let keep = fleet.saturating_sub(1);
            src.extend(routes.iter().take(keep).cloned());

            // Start from the last regular route and append all extra customers to it
            let mut merged = routes[keep].clone();
            merged.pop(); // remove trailing depot
            for r in routes.iter().skip(fleet) {
                if r.len() > 2 { merged.extend_from_slice(&r[1..r.len() - 1]); }
            }
            merged.push(0); // re-add trailing depot
            src.push(merged);
        }

        // Ensure we have exactly 'fleet' routes by padding with empties if needed
        while src.len() < fleet {
            src.push(vec![0, 0]);
        }

        // Convert to Route nodes and fill the local search structures
        let all_routes: Vec<Route> = src.iter().map(|r| Route::new(r)).collect();
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
        debug_assert!(len >= 2);

        // forward pass: seq0_i
        let mut acc_fwd = Sequence::singleton(data, r.nodes[0].id);
        r.nodes[0].seq0_i = acc_fwd;
        for pos in 1..len {
            let id = r.nodes[pos].id;
            acc_fwd = Sequence::join2(data, &acc_fwd, &Sequence::singleton(data, id));
            r.nodes[pos].seq0_i = acc_fwd;
        }

        // backward pass: seqi_n
        let mut acc_bwd = Sequence::singleton(data, r.nodes[len - 1].id);
        r.nodes[len - 1].seqi_n = acc_bwd;
        for pos in (0..len - 1).rev() {
            let id = r.nodes[pos].id;
            acc_bwd = Sequence::join2(data, &Sequence::singleton(data, id), &acc_bwd);
            r.nodes[pos].seqi_n = acc_bwd;
        }

        // per-node: seq1, seq12, seq21
        for pos in 0..len {
            let id = r.nodes[pos].id;
            r.nodes[pos].seq1 = Sequence::singleton(data, id);
        }
        for pos in 0..len {
            if pos + 1 < len {
                let seq1_curr = r.nodes[pos].seq1;
                let seq1_next = r.nodes[pos + 1].seq1;
                r.nodes[pos].seq12 = Sequence::join2(data, &seq1_curr, &seq1_next);
                r.nodes[pos].seq21 = Sequence::join2(data, &seq1_next, &seq1_curr);
                if pos + 2 < len {
                    let seq1_next2 = r.nodes[pos + 2].seq1;
                    r.nodes[pos].seq123 = Sequence::join2(data, &r.nodes[pos].seq12, &seq1_next2);
                }
            }
        }

        let end = r.nodes[len - 1].seq0_i;
        r.load = end.load;
        r.tw   = end.tw;
        r.distance = end.distance;
        r.cost = end.eval(data, &self.params);

        // Update route node mappings
        for (pos, node) in self.routes[rid].nodes.iter().enumerate() {
            self.node_route[node.id] = rid;
            self.node_pos[node.id] = pos;
        }

        // Refresh vector of empty routes
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

        debug_assert!(pos1 > 0); // U is a client
        debug_assert!(self.routes[r1].nodes[pos1].id != 0); // U is a client

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
        if len < pos1 + 4 { return false; } // need at least [0, U, X, V, 0]

        debug_assert!(pos1 > 0); // U is a client
        debug_assert!(self.routes[r1].nodes[pos1].id != 0); // U is a client

        let old_cost = route.cost;
        let mut best_cost = old_cost;
        let mut best_pos: Option<usize> = None;

        // SAFETY: bounds guaranteed by len >= pos1 + 4 check above
        let nodes = &route.nodes;
        let seq0_i_prev = unsafe { nodes.get_unchecked(pos1 - 1).seq0_i };
        let seq1_curr = unsafe { nodes.get_unchecked(pos1).seq1 };
        let mut acc_mid = unsafe { nodes.get_unchecked(pos1 + 1).seq1 };
        
        for pos2 in (pos1 + 2)..(len - 1) {
            // SAFETY: pos2 in [pos1+2, len-2], so pos2+1 <= len-1 < len
            let seq1_pos2 = unsafe { nodes.get_unchecked(pos2).seq1 };
            let seqi_n_next = unsafe { nodes.get_unchecked(pos2 + 1).seqi_n };
            let new_cost = Sequence::eval_n(self.data, &self.params,
                                            &[seq0_i_prev,
                                                seq1_pos2,
                                                acc_mid,
                                                seq1_curr,
                                                seqi_n_next]);
            if new_cost < best_cost {
                best_cost = new_cost;
                best_pos = Some(pos2);
            }
            acc_mid = Sequence::join2(self.data, &acc_mid, &seq1_pos2);
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
        debug_assert!(r1 != r2);
        debug_assert!(pos1 > 0 && pos1 < self.routes[r1].nodes.len());
        debug_assert!(pos2 > 0 && pos2 < self.routes[r2].nodes.len());

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
        if len < pos1 + 3 { return false; } // need at least [0, U, V, 0]

        debug_assert!(pos1 > 0); // U is a client
        debug_assert!(self.routes[r1].nodes[pos1].id != 0); // U is a client

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
        let u = &ru.nodes[pos1]; // cannot be a depot
        let v = &rv.nodes[pos2]; // could be a depot
        let u_pred = &ru.nodes[pos1 - 1]; // could be a depot
        let v_pred = &rv.nodes[pos2 - 1]; // could be a depot
        let x = &ru.nodes[pos1 + 1]; // could be a depot
        debug_assert!(u.id != 0,"Should always apply inter-route with a client as first node");
        debug_assert!(r1 != r2,"Should not test inter-route move on same route");

        // result table (i: what r1 sends, j: what r2 sends)
        // 0 -> send nothing
        // 1 -> send single (U / V)
        // 2 -> send pair in forward order (U, X) / (V, Y)
        // 3 -> send pair reversed (X, U) / (Y, V)
        // 4 -> send triple forward     (U, X, Xnext) / (V, Y, Ynext)
        let old_total = ru.cost + rv.cost;
        let mut best_i = 0usize;
        let mut best_j = 0usize;
        let mut best_cost = old_total;

        // tiny helper to track best
        let mut update_best = |i: usize, j: usize, cand: i64| {
            if cand < best_cost {
                best_cost = cand;
                best_i = i;
                best_j = j;
            }
        };

        // Send {U}, receive {}
        let result10 = Sequence::eval2(data, &self.params, &u_pred.seq0_i, &x.seqi_n)
            + Sequence::eval3(data, &self.params, &v_pred.seq0_i, &u.seq1, &v.seqi_n);
        update_best(1,0,result10);

        // Send {U}, receive {V}
        if v.id != 0 {
            let result11 = Sequence::eval3(data, &self.params, &u_pred.seq0_i, &v.seq1, &x.seqi_n)
                + Sequence::eval3(data, &self.params, &v_pred.seq0_i, &u.seq1, &rv.nodes[pos2 + 1].seqi_n);
            update_best(1,1,result11);
        }

        if x.id != 0 {
            // Send {U,X} or {X,U}, receive {}
            let x_next = &ru.nodes[pos1 + 2];
            let mut result20 = Sequence::eval2(data, &self.params, &u_pred.seq0_i, &x_next.seqi_n);
            let mut result30 = result20;
            result20 += Sequence::eval3(data, &self.params, &v_pred.seq0_i, &u.seq12, &v.seqi_n);
            result30 += Sequence::eval3(data, &self.params, &v_pred.seq0_i, &u.seq21, &v.seqi_n);
            update_best(2,0,result20);
            update_best(3,0,result30);

            if v.id != 0 {
                // Send {U,X} or {X,U}, receive {V}
                let y = &rv.nodes[pos2 + 1];
                let mut result21 = Sequence::eval3(data, &self.params, &u_pred.seq0_i, &v.seq1, &x_next.seqi_n);
                let mut result31 = result21;
                result21 += Sequence::eval3(data, &self.params, &v_pred.seq0_i, &u.seq12, &y.seqi_n);
                result31 += Sequence::eval3(data, &self.params, &v_pred.seq0_i, &u.seq21, &y.seqi_n);
                update_best(2,1,result21);
                update_best(3,1,result31);

                if y.id != 0 {
                    // Send {U,X} or {X,U}, receive {V,Y} or {Y,V}
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

            // Send {U,X,Xnext}
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

        if best_i == 0 && best_j == 0 { return false; }     // no improvement

        // ---------------- apply move ----------------
        // Remove helper — returns Nodes in the order they must be inserted in the target
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
        debug_assert!(new_total - old_total < 0, "Applied non-improving move unexpectedly!");
        true
    }

    pub fn run_swapstar(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize) -> bool {
        debug_assert!(r1 != r2);
        debug_assert!(pos1 > 0 && pos1 < self.routes[r1].nodes.len() - 1);
        debug_assert!(pos2 > 0 && pos2 < self.routes[r2].nodes.len() - 1);

        let route1_len = self.routes[r1].nodes.len();
        let route2_len = self.routes[r2].nodes.len();
        let u = self.routes[r1].nodes[pos1].id;
        let v = self.routes[r2].nodes[pos2].id;
        let (pu, nu) = (self.routes[r1].nodes[pos1 - 1].id, self.routes[r1].nodes[pos1 + 1].id);
        let (pv, nv) = (self.routes[r2].nodes[pos2 - 1].id, self.routes[r2].nodes[pos2 + 1].id);

        // First filter on route costs
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

        // best detour for inserting v into r1 \ {u}
        let hole_v = self.data.dm(pu, v) + self.data.dm(v, nu) - self.data.dm(pu, nu);
        let mut best_ins_v = hole_v;
        for t in 1..route1_len {
            let a_id = self.routes[r1].nodes[t - 1].id;
            let b_id = self.routes[r1].nodes[t].id;
            if a_id == u || b_id == u { continue; } // skip arcs broken by removing U
            let delta = self.data.dm(a_id, v) + self.data.dm(v, b_id) - self.data.dm(a_id, b_id);
            if delta < best_ins_v { best_ins_v = delta; }
        }

        // best detour for inserting u into r2 \ {v}
        let hole_u = self.data.dm(pv, u) + self.data.dm(u, nv) - self.data.dm(pv, nv);
        let mut best_ins_u = hole_u;
        for t in 1..route2_len {
            let a_id = self.routes[r2].nodes[t - 1].id;
            let b_id = self.routes[r2].nodes[t].id;
            if a_id == v || b_id == v { continue; } // skip arcs broken by removing V
            let delta = self.data.dm(a_id, u) + self.data.dm(u, b_id) - self.data.dm(a_id, b_id);
            if delta < best_ins_u { best_ins_u = delta; }
        }

        // Second filter on route costs
        lb_new_total += (best_ins_v + best_ins_u) as i64;
        if lb_new_total > old_total { return false; }

        // Full evaluation considering time windows upon insertion
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

        // Build "route without V" sequences for r2 (exclude pos2)
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

        // Reinsertion of V into r1 \ {U}
        let v_seq1 = self.routes[r2].nodes[pos2].seq1;
        let mut best_cost1 = i64::MAX / 4;
        let mut best_t1: usize = 1;
        for t in 1..route1_len {
            let cand = Sequence::eval3(self.data, &self.params, &left_excl1[t], &v_seq1, &right_excl1[t]);
            if cand < best_cost1 { best_cost1 = cand; best_t1 = t; }
        }

        // Reinsertion of U into r2 \ {V}
        let u_seq1 = self.routes[r1].nodes[pos1].seq1;
        let mut best_cost2 = i64::MAX / 4;
        let mut best_t2: usize = 1;
        for t in 1..route2_len {
            let cand = Sequence::eval3(self.data, &self.params, &left_excl2[t], &u_seq1, &right_excl2[t]);
            if cand < best_cost2 { best_cost2 = cand; best_t2 = t; }
        }

        if best_cost1 + best_cost2 >= old_total { return false; }

        // Apply move
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
        debug_assert!(new_total - old_total < 0, "Applied non-improving SWAP* unexpectedly!");
        true
    }

    pub fn runls(&mut self, routes: &mut Vec<Vec<usize>>, rng: &mut SmallRng, params: Params, is_repair: bool, factor: usize) {
        self.params = params;
        if !is_repair {
            self.load_from_routes(routes);
        }
        else { // hot start for Repair phase
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
        while improved {
            improved = false;
            loop_id += 1;
            let mut c1_order: Vec<usize> = (1..self.data.nb_nodes).collect();
            c1_order.shuffle(rng);
            for c1 in c1_order {
                let last_tested = self.when_last_tested[c1];
                self.when_last_tested[c1] = self.nb_moves;
                let r1 = self.node_route[c1];
                let pos1 = self.node_pos[c1];

                let neigh_len = self.neighbors_before[c1].len();
                let start = rng.gen_range(0..neigh_len); // randomizing start of neighbor loop
                for off in 0..neigh_len {
                    let c2 = self.neighbors_before[c1][(start + off) % neigh_len];
                    let r2 = self.node_route[c2];
                    let pos2 = self.node_pos[c2];
                    if r1 == r2 { continue; }

                    // Skip if both routes unchanged since last tests for this customer
                    if self.when_last_modified[r1].max(self.when_last_modified[r2]) <= last_tested {
                        continue;
                    }

                    // We use pos2 + 1 for the SWAP and RELOCATE moves since c2 is a good predecessor for c1
                    // Moves listed here create the edge c2 => c1, but never insert immediately after a depot
                    if self.run_inter_route(r1, pos1, r2, pos2+1) {
                        improved = true;
                        break;
                    }

                    // Special case to manage insert immediately after a depot
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

                        // Skip if both routes unchanged since last tests for this customer
                        // Also break symmetry by eliminating moves such that c1 < c2
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

                // Moves involving an empty route (only tested after the first loop)
                let r1 = self.node_route[c1];
                let pos1 = self.node_pos[c1];
                if loop_id > 1 && (loop_id == 2 || self.when_last_modified[r1] > last_tested) {
                    if let Some(&r2) = self.empty_routes.first() {
                        let pos2 = 1;
                        debug_assert!(self.routes[r2].nodes.len() == 2);

                        // 2-opt* with an empty route (essentially cut the route in 2)
                        if self.run_2optstar(r1, pos1, r2, pos2) {
                            improved = true;
                            break;
                        }

                        // Insert in an empty route
                        if self.run_inter_route(r1, pos1, r2, pos2) {
                            improved = true;
                            break;
                        }
                    }
                }

                // Intra-route moves
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
