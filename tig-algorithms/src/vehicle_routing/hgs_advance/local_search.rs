use super::problem::Problem;
use super::params::Params;
use super::sequence::Sequence;
use std::cmp::min;
use std::sync::Arc;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

#[derive(Clone, Debug, Default)]
pub struct Node {
    id: usize,
    demand: i32,
    dist_to_pred: i32,
    dist_to_succ: i32,
    seqno1: Sequence,
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
            demand: data.nd(id).demand,
            ..Default::default()
        }
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
    fn new(data: &Problem, ids: &[usize]) -> Self {
        Self {
            nodes: ids.iter().copied().map(|id| Node::new(data, id)).collect(),
            ..Default::default()
        }
    }
}

pub struct LocalSearch {
    pub data: Arc<Problem>,
    pub neighbors_before: Vec<Vec<usize>>,
    pub neighbors_capacity_swap: Vec<Vec<usize>>,
    pub loop_order_nodes: Vec<usize>,
    pub params: Params,
    pub cost: i64,
    pub routes: Vec<Route>,
    pub node_route: Vec<usize>,
    pub node_pos: Vec<usize>,
    pub empty_routes: Vec<usize>,
    pub when_last_modified: Vec<usize>,   // per route
    pub when_last_tested: Vec<usize>,     // per customer id
    pub nb_moves: usize,                  // monotone counter of applied moves
    pub move_credit: i64,                 // deterioration budget; set to -1 from loop #2 onward
}

#[derive(Clone, Copy)]
enum CandidateMove {
    InterRoute { r1: usize, pos1: usize, r2: usize, pos2: usize },
    TwoOptStar { r1: usize, pos1: usize, r2: usize, pos2: usize },
    SwapStar { r1: usize, pos1: usize, r2: usize, pos2: usize },
    IntraRelocate { r1: usize, pos1: usize },
    IntraOrOpt2 { r1: usize, pos1: usize },
    IntraSwap { r1: usize, pos1: usize },
    Intra2Opt { r1: usize, pos1: usize },
}

impl LocalSearch {
    pub fn new(data: Arc<Problem>, params: Params, _rng: &mut SmallRng) -> Self {

        let n = data.nb_nodes;
        let cap = n.saturating_sub(2);
        let keep = min(params.granularity as usize, cap);
        let mut neighbors_before: Vec<Vec<usize>> = vec![Vec::new(); n];

        for i in 1..n {
            let ndi = data.nd(i);
            let mut prox: Vec<(i32, usize)> = Vec::with_capacity(cap);
            for j in 1..n {
                if j == i { continue; }
                let ndj = data.nd(j);
                let tji  = data.dm(j,i);
                let wait = (ndi.start_tw - tji - ndj.service_time - ndj.end_tw).max(0);
                let late = (ndj.start_tw + ndj.service_time + tji - ndi.end_tw).max(0);
                let proxy10 = 10 * tji + 2 * wait + 10 * late;
                prox.push((proxy10, j));
            }
            prox.sort_by_key(|&(p, _)| p);
            neighbors_before[i] = prox[..keep].iter().map(|&(_, j)| j).collect();
        }

        let mut neighbors_capacity_swap: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 1..n {
            let di = data.nd(i).demand;
            let mut similar_by_demand: Vec<(i32, usize)> = Vec::with_capacity(n.saturating_sub(2));
            for j in 1..n {
                if j == i { continue; }
                similar_by_demand.push(((data.nd(j).demand - di).abs(), j));
            }
            similar_by_demand.sort_by_key(|&(dd, _)| dd);
            let keep_similar =
                (params.swapstar_capa_filter * similar_by_demand.len() as f64).ceil() as usize;

            let mut prox: Vec<(i32, usize)> = Vec::with_capacity(keep_similar);
            for &(_, j) in similar_by_demand.iter().take(keep_similar) {
                prox.push((data.dm(i, j), j));
            }
            prox.sort_by_key(|&(d, _)| d);
            let m = prox.len().min(params.granularity2 as usize);
            neighbors_capacity_swap[i] = prox[..m]
                .iter()
                .filter_map(|&(_, j)| if j < i { Some(j) } else { None })
                .collect();
        }

        Self {
            data,
            neighbors_before,
            neighbors_capacity_swap,
            loop_order_nodes: (1..n).collect(),
            params,
            cost: 0,
            routes: Vec::new(),
            node_route: Vec::new(),
            node_pos: Vec::new(),
            empty_routes: Vec::new(),
            when_last_modified: Vec::new(),
            when_last_tested: vec![0; n],
            nb_moves: 0,
            move_credit: 0,
        }
    }

    #[inline]
    fn register_accepted_delta(&mut self, delta: i64) {
        if self.move_credit < 0 { return; }
        if delta < 0 {
            let cap = self.params.max_credit_deterioration as i64;
            self.move_credit = (self.move_credit - delta).min(cap);
        } else if delta > 0 {
            debug_assert!(delta <= self.move_credit, "Accepted deterioration exceeds available credit");
            self.move_credit -= delta;
        }
    }

    #[inline]
    fn apply_candidate_move(&mut self, mv: CandidateMove) -> Option<i64> {
        self.eval_candidate_delta_mode(mv, true)
    }

    #[inline]
    fn eval_candidate_delta(&mut self, mv: CandidateMove) -> Option<i64> {
        self.eval_candidate_delta_mode(mv, false)
    }

    #[inline]
    fn eval_candidate_delta_mode(&mut self, mv: CandidateMove, apply: bool) -> Option<i64> {
        match mv {
            CandidateMove::InterRoute { r1, pos1, r2, pos2 } => self.run_inter_route(r1, pos1, r2, pos2, apply),
            CandidateMove::TwoOptStar { r1, pos1, r2, pos2 } => self.run_2optstar(r1, pos1, r2, pos2, apply),
            CandidateMove::SwapStar { r1, pos1, r2, pos2 } => self.run_swapstar(r1, pos1, r2, pos2, apply),
            CandidateMove::IntraRelocate { r1, pos1 } => self.run_intra_route_relocate(r1, pos1, apply),
            CandidateMove::IntraOrOpt2 { r1, pos1 } => self.run_intra_route_oropt2(r1, pos1, apply),
            CandidateMove::IntraSwap { r1, pos1 } => self.run_intra_route_swap(r1, pos1, apply),
            CandidateMove::Intra2Opt { r1, pos1 } => self.run_2opt(r1, pos1, apply),
        }
    }

    pub fn run_from_routes(
        &mut self,
        routes: &[Vec<usize>],
        inherited_routes: &[bool],
        params: Params,
        rng: &mut SmallRng,
    ) -> Vec<Vec<usize>> {
        self.params = params;
        let n = self.data.nb_nodes;
        let fleet = self.data.nb_vehicles;

        // Normalize routes to exactly `fleet` entries.
        let mut src: Vec<Vec<usize>> = Vec::new();

        if routes.len() <= fleet {
            src.extend(routes.iter().cloned());
            // If needed, pad missing routes with empty depot-only routes.
            src.resize(fleet, vec![0, 0]);
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
            debug_assert_eq!(src.len(), fleet);
        }

        // Convert to Route nodes and fill the local search structures
        let all_routes: Vec<Route> = src.iter().map(|r| Route::new(self.data.as_ref(), r)).collect();
        self.node_route = vec![0; n];
        self.node_pos = vec![0; n];
        self.empty_routes.clear();
        self.routes = all_routes;
        self.when_last_modified = vec![0; self.routes.len()];
        self.when_last_tested = vec![0; n];
        self.nb_moves = 1;

        for rid in 0..self.routes.len() { self.update_route(rid); }
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
            // Set all routes that have not been inherited from the majority parent as modified
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
        let mut out: Vec<Vec<usize>> = self.routes
            .iter()
            .filter(|r| r.nodes.len() > 2)
            .map(|r| r.nodes.iter().map(|n| n.id).collect::<Vec<usize>>())
            .collect();

        // In CVRP mode, normalize route orientation to clockwise before returning.
        if !self.data.is_vrptw {
            for route in &mut out {
                if route.len() == 4 {
                    // With exactly two customers, enforce a deterministic orientation:
                    // smallest customer index first.
                    if route[1] > route[2] {
                        route.swap(1, 2);
                    }
                } else if Self::is_counter_clockwise(self.data.as_ref(), route) {
                    let n = route.len();
                    route[1..n - 1].reverse();
                }
            }
        }
        out
    }

    #[inline]
    fn is_counter_clockwise(data: &Problem, route: &[usize]) -> bool {
        if route.len() < 4 {
            return false;
        }
        // Shoelace on the closed polyline [0, ..., 0] as stored in the route.
        let mut area2: i64 = 0;
        for k in 0..route.len() - 1 {
            let (x1, y1) = data.node_positions[route[k]];
            let (x2, y2) = data.node_positions[route[k + 1]];
            area2 += (x1 as i64) * (y2 as i64) - (x2 as i64) * (y1 as i64);
        }
        area2 > 0
    }

    fn update_route(&mut self, rid: usize) {
        let data = self.data.as_ref();
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

        // per-node: seq1, seq12, seq21, seqno1
        for pos in 0..len {
            let id = r.nodes[pos].id;
            r.nodes[pos].seq1 = Sequence::singleton(data, id);
            r.nodes[pos].demand = data.nd(id).demand;
            r.nodes[pos].dist_to_pred = if pos > 0 { data.dm(r.nodes[pos - 1].id, id) } else { 0 };
            r.nodes[pos].dist_to_succ = if pos + 1 < len { data.dm(id, r.nodes[pos + 1].id) } else { 0 };
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
            if pos > 0 && pos + 1 < len {
                r.nodes[pos].seqno1 = Sequence::join2(data, &r.nodes[pos - 1].seq0_i, &r.nodes[pos + 1].seqi_n);
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

    pub fn run_intra_route_relocate(&mut self, r1: usize, pos1: usize, apply: bool) -> Option<i64> {
        let route = &self.routes[r1];
        let data = self.data.as_ref();
        let len = route.nodes.len();
        if len <= 3 { return None; } // no alternative insertion for single-client routes

        debug_assert!(pos1 > 0); // U is a client
        debug_assert!(self.routes[r1].nodes[pos1].id != 0); // U is a client

        let old_cost = route.cost;
        let max_acceptable_cost = old_cost + self.move_credit;
        let mut best_cost = i64::MAX;
        let mut best_pos: Option<usize> = None;
        let u_seq1 = route.nodes[pos1].seq1;
        let old_distance = route.distance as i64;
        let cap_pen = ((route.load - self.data.max_capacity).max(0) as i64) * self.params.penalty_capa as i64;
        let max_distance_acceptable = max_acceptable_cost - cap_pen;
        let a_id = route.nodes[pos1 - 1].id;
        let u_id = route.nodes[pos1].id;
        let b_id = route.nodes[pos1 + 1].id;
        let removed_au_ub = data.dm(a_id, u_id) + data.dm(u_id, b_id);
        let add_ab = data.dm(a_id, b_id);
        let fixed_delta = add_ab - removed_au_ub;

        // Insert U before t in [1 .. pos1-1]
        if pos1 > 1 {
            let mut right_excl_u = route.nodes[pos1 + 1].seqi_n;
            for t in (1..pos1).rev() {
                right_excl_u = Sequence::join2(self.data.as_ref(), &route.nodes[t].seq1, &right_excl_u);
                let c_id = route.nodes[t - 1].id;
                let d_id = route.nodes[t].id;
                let delta_distance = fixed_delta + data.dm(c_id, u_id) + data.dm(u_id, d_id) - data.dm(c_id, d_id);
                if old_distance + (delta_distance as i64) > max_distance_acceptable { continue; }
                let left = route.nodes[t - 1].seq0_i;
                let new_cost = Sequence::eval3(self.data.as_ref(), &self.params, &left, &u_seq1, &right_excl_u);
                if new_cost <= max_acceptable_cost && new_cost < best_cost {
                    best_cost = new_cost;
                    best_pos = Some(t);
                }
            }
        }

        // Insert U before t in [pos1+2 .. len-1]
        if pos1 + 2 < len {
            let mut left_excl_u = Sequence::join2(
                self.data.as_ref(),
                &route.nodes[pos1 - 1].seq0_i,
                &route.nodes[pos1 + 1].seq1,
            );
            for t in (pos1 + 2)..len {
                let c_id = route.nodes[t - 1].id;
                let d_id = route.nodes[t].id;
                let delta_distance = fixed_delta + data.dm(c_id, u_id) + data.dm(u_id, d_id) - data.dm(c_id, d_id);
                if old_distance + (delta_distance as i64) <= max_distance_acceptable {
                    let new_cost = Sequence::eval3(
                        self.data.as_ref(),
                        &self.params,
                        &left_excl_u,
                        &u_seq1,
                        &route.nodes[t].seqi_n,
                    );
                    if new_cost <= max_acceptable_cost && new_cost < best_cost {
                        best_cost = new_cost;
                        best_pos = Some(t);
                    }
                }
                if t + 1 < len {
                    left_excl_u = Sequence::join2(self.data.as_ref(), &left_excl_u, &route.nodes[t].seq1);
                }
            }
        }

        if let Some(mypos) = best_pos {
            let selected_delta = best_cost - old_cost;
            if apply {
                let insert_pos = if mypos > pos1 { mypos - 1 } else { mypos };
                let elem = self.routes[r1].nodes.remove(pos1);
                self.routes[r1].nodes.insert(insert_pos, elem);
                self.nb_moves += 1;
                self.update_route(r1);
                let delta = self.routes[r1].cost - old_cost;
                self.cost += delta;
                self.register_accepted_delta(delta);
                Some(delta)
            } else {
                Some(selected_delta)
            }
        } else { None }
    }

    pub fn run_intra_route_oropt2(&mut self, r1: usize, pos1: usize, apply: bool) -> Option<i64> {
        let route = &self.routes[r1];
        let data = self.data.as_ref();
        let len = route.nodes.len();
        if pos1 + 2 >= len { return None; } // pair does not exist

        debug_assert!(pos1 > 0); // U is a client
        debug_assert!(self.routes[r1].nodes[pos1].id != 0); // U is a client
        debug_assert!(self.routes[r1].nodes[pos1 + 1].id != 0); // successor is a client

        let old_cost = route.cost;
        let max_acceptable_cost = old_cost + self.move_credit;
        let mut best_cost = i64::MAX;
        let mut best_pos: Option<usize> = None;
        let mut best_reversed = false;
        let pair_fwd = route.nodes[pos1].seq12;
        let pair_rev = route.nodes[pos1].seq21;
        let old_distance = route.distance as i64;
        let cap_pen = ((route.load - self.data.max_capacity).max(0) as i64) * self.params.penalty_capa as i64;
        let max_distance_acceptable = max_acceptable_cost - cap_pen;
        let a_id = route.nodes[pos1 - 1].id;
        let u_id = route.nodes[pos1].id;
        let x_id = route.nodes[pos1 + 1].id;
        let b_id = route.nodes[pos1 + 2].id;
        let removed_auxb = data.dm(a_id, u_id) + data.dm(u_id, x_id) + data.dm(x_id, b_id);
        let add_ab = data.dm(a_id, b_id);
        let fixed_delta_fwd = add_ab + data.dm(u_id, x_id) - removed_auxb;
        let fixed_delta_rev = add_ab + data.dm(x_id, u_id) - removed_auxb;

        // Insert (U,X) or (X,U) before t in [1 .. pos1-1]
        if pos1 > 1 {
            let mut right_excl_pair = route.nodes[pos1 + 2].seqi_n;
            for t in (1..pos1).rev() {
                right_excl_pair = Sequence::join2(self.data.as_ref(), &route.nodes[t].seq1, &right_excl_pair);
                let c_id = route.nodes[t - 1].id;
                let d_id = route.nodes[t].id;
                let delta_fwd = fixed_delta_fwd + data.dm(c_id, u_id) + data.dm(x_id, d_id) - data.dm(c_id, d_id);
                let delta_rev = fixed_delta_rev + data.dm(c_id, x_id) + data.dm(u_id, d_id) - data.dm(c_id, d_id);
                let can_pass_fwd = old_distance + (delta_fwd as i64) <= max_distance_acceptable;
                let can_pass_rev = old_distance + (delta_rev as i64) <= max_distance_acceptable;
                if !can_pass_fwd && !can_pass_rev { continue; }
                let left = route.nodes[t - 1].seq0_i;

                let new_cost_fwd = Sequence::eval3(self.data.as_ref(), &self.params, &left, &pair_fwd, &right_excl_pair);
                if new_cost_fwd <= max_acceptable_cost && new_cost_fwd < best_cost {
                    best_cost = new_cost_fwd;
                    best_pos = Some(t);
                    best_reversed = false;
                }

                let new_cost_rev = Sequence::eval3(self.data.as_ref(), &self.params, &left, &pair_rev, &right_excl_pair);
                if new_cost_rev <= max_acceptable_cost && new_cost_rev < best_cost {
                    best_cost = new_cost_rev;
                    best_pos = Some(t);
                    best_reversed = true;
                }
            }
        }

        // Insert (U,X) or (X,U) before t in [pos1+3 .. len-1]
        if pos1 + 3 < len {
            let mut left_excl_pair = Sequence::join2(
                self.data.as_ref(),
                &route.nodes[pos1 - 1].seq0_i,
                &route.nodes[pos1 + 2].seq1,
            );
            for t in (pos1 + 3)..len {
                let c_id = route.nodes[t - 1].id;
                let d_id = route.nodes[t].id;
                let delta_fwd = fixed_delta_fwd + data.dm(c_id, u_id) + data.dm(x_id, d_id) - data.dm(c_id, d_id);
                let delta_rev = fixed_delta_rev + data.dm(c_id, x_id) + data.dm(u_id, d_id) - data.dm(c_id, d_id);
                let right = route.nodes[t].seqi_n;

                let can_pass_fwd = old_distance + (delta_fwd as i64) <= max_distance_acceptable;
                let can_pass_rev = old_distance + (delta_rev as i64) <= max_distance_acceptable;
                if can_pass_fwd || can_pass_rev {
                    let new_cost_fwd = Sequence::eval3(self.data.as_ref(), &self.params, &left_excl_pair, &pair_fwd, &right);
                    if new_cost_fwd <= max_acceptable_cost && new_cost_fwd < best_cost {
                        best_cost = new_cost_fwd;
                        best_pos = Some(t);
                        best_reversed = false;
                    }

                    let new_cost_rev = Sequence::eval3(self.data.as_ref(), &self.params, &left_excl_pair, &pair_rev, &right);
                    if new_cost_rev <= max_acceptable_cost && new_cost_rev < best_cost {
                        best_cost = new_cost_rev;
                        best_pos = Some(t);
                        best_reversed = true;
                    }
                }

                if t + 1 < len {
                    left_excl_pair = Sequence::join2(self.data.as_ref(), &left_excl_pair, &route.nodes[t].seq1);
                }
            }
        }

        if let Some(mypos) = best_pos {
            let selected_delta = best_cost - old_cost;
            if apply {
                let insert_pos = if mypos > pos1 { mypos - 2 } else { mypos };
                let n1 = self.routes[r1].nodes.remove(pos1);
                let n2 = self.routes[r1].nodes.remove(pos1);
                let (a, b) = if best_reversed { (n2, n1) } else { (n1, n2) };
                self.routes[r1].nodes.insert(insert_pos, a);
                self.routes[r1].nodes.insert(insert_pos + 1, b);
                self.nb_moves += 1;
                self.update_route(r1);
                let delta = self.routes[r1].cost - old_cost;
                self.cost += delta;
                self.register_accepted_delta(delta);
                Some(delta)
            } else {
                Some(selected_delta)
            }
        } else { None }
    }

    pub fn run_intra_route_swap(&mut self, r1: usize, pos1: usize, apply: bool) -> Option<i64> {
        let route = &self.routes[r1];
        let len   = route.nodes.len();
        if len <= 4 { return None; } // need at least 3 clients for a non-adjacent swap
        let data = self.data.as_ref();
        let has_right = pos1 + 2 < len - 1;
        let has_left = pos1 >= 3;
        if !has_left && !has_right { return None; }

        debug_assert!(pos1 > 0); // U is a client
        debug_assert!(self.routes[r1].nodes[pos1].id != 0); // U is a client

        let old_cost = route.cost;
        let max_acceptable_cost = old_cost + self.move_credit;
        let mut best_cost = i64::MAX;
        let mut best_pos: Option<usize> = None;
        let old_distance = route.distance as i64;
        let cap_pen = ((route.load - self.data.max_capacity).max(0) as i64) * self.params.penalty_capa as i64;
        let max_distance_acceptable = max_acceptable_cost - cap_pen;
        let pu = route.nodes[pos1 - 1].id;
        let u = route.nodes[pos1].id;
        let nu = route.nodes[pos1 + 1].id;
        let removed_u = route.nodes[pos1 - 1].dist_to_succ + route.nodes[pos1].dist_to_succ;

        // Distance-only prefilter: evaluate only sides that contain a potentially improving swap.
        let mut has_right_potential = false;
        if has_right {
            for pos2 in (pos1 + 2)..(len - 1) {
                let pv = route.nodes[pos2 - 1].id;
                let v = route.nodes[pos2].id;
                let nv = route.nodes[pos2 + 1].id;
                let removed_v = route.nodes[pos2 - 1].dist_to_succ + route.nodes[pos2].dist_to_succ;
                let delta_distance = (data.dm(pu, v) + data.dm(v, nu) - removed_u)
                    + (data.dm(pv, u) + data.dm(u, nv) - removed_v);
                if old_distance + (delta_distance as i64) <= max_distance_acceptable {
                    has_right_potential = true;
                    break;
                }
            }
        }
        let mut has_left_potential = false;
        if has_left {
            for pos2 in (1..=pos1 - 2).rev() {
                let pv = route.nodes[pos2 - 1].id;
                let v = route.nodes[pos2].id;
                let nv = route.nodes[pos2 + 1].id;
                let removed_v = route.nodes[pos2 - 1].dist_to_succ + route.nodes[pos2].dist_to_succ;
                let delta_distance = (data.dm(pu, v) + data.dm(v, nu) - removed_u)
                    + (data.dm(pv, u) + data.dm(u, nv) - removed_v);
                if old_distance + (delta_distance as i64) <= max_distance_acceptable {
                    has_left_potential = true;
                    break;
                }
            }
        }
        if !has_right_potential && !has_left_potential {
            return None;
        }

        if has_right_potential {
            let mut acc_mid = route.nodes[pos1 + 1].seq1;
            for pos2 in (pos1 + 2)..(len - 1) {
                let new_cost = Sequence::eval5(
                    data,
                    &self.params,
                    &route.nodes[pos1 - 1].seq0_i,
                    &route.nodes[pos2].seq1,
                    &acc_mid,
                    &route.nodes[pos1].seq1,
                    &route.nodes[pos2 + 1].seqi_n,
                );
                if new_cost <= max_acceptable_cost && new_cost < best_cost {
                    best_cost = new_cost;
                    best_pos = Some(pos2);
                }
                acc_mid = Sequence::join2(data, &acc_mid, &route.nodes[pos2].seq1);
            }
        }

        if has_left_potential {
            let mut acc_mid = route.nodes[pos1 - 1].seq1;
            for pos2 in (1..=pos1 - 2).rev() {
                let new_cost = Sequence::eval5(
                    data,
                    &self.params,
                    &route.nodes[pos2 - 1].seq0_i,
                    &route.nodes[pos1].seq1,
                    &acc_mid,
                    &route.nodes[pos2].seq1,
                    &route.nodes[pos1 + 1].seqi_n,
                );
                if new_cost <= max_acceptable_cost && new_cost < best_cost {
                    best_cost = new_cost;
                    best_pos = Some(pos2);
                }
                if pos2 > 1 {
                    acc_mid = Sequence::join2(data, &route.nodes[pos2].seq1, &acc_mid);
                }
            }
        }

        if let Some(mypos) = best_pos {
            let selected_delta = best_cost - old_cost;
            if apply {
                self.routes[r1].nodes.swap(pos1, mypos);
                self.nb_moves += 1;
                self.update_route(r1);
                let delta = self.routes[r1].cost - old_cost;
                self.cost += delta;
                self.register_accepted_delta(delta);
                Some(delta)
            } else {
                Some(selected_delta)
            }
        } else { None }
    }

    pub fn run_2optstar(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize, apply: bool) -> Option<i64> {
        debug_assert!(r1 != r2);
        debug_assert!(pos1 > 0 && pos1 < self.routes[r1].nodes.len());
        debug_assert!(pos2 > 0 && pos2 < self.routes[r2].nodes.len());

        let route1 = &self.routes[r1];
        let route2 = &self.routes[r2];
        let old_cost = route1.cost + route2.cost;
        let max_acceptable_cost = old_cost + self.move_credit;

        let left1 = &route1.nodes[pos1 - 1].seq0_i;
        let right1 = &route2.nodes[pos2].seqi_n;
        let left2 = &route2.nodes[pos2 - 1].seq0_i;
        let right2 = &route1.nodes[pos1].seqi_n;

        // Prefilter: only distance + load excess penalties (no TW penalties).
        let pcap = self.params.penalty_capa as i64;
        let max_cap = self.data.max_capacity;
        let dist1 = left1.distance + right1.distance + self.data.dm(left1.last_node, right1.first_node);
        let dist2 = left2.distance + right2.distance + self.data.dm(left2.last_node, right2.first_node);
        let load1 = left1.load + right1.load;
        let load2 = left2.load + right2.load;
        let lb_cost = (dist1 as i64)
            + (dist2 as i64)
            + ((load1 - max_cap).max(0) as i64) * pcap
            + ((load2 - max_cap).max(0) as i64) * pcap;
        if lb_cost > max_acceptable_cost {
            return None;
        }

        let new1 = Sequence::eval2(self.data.as_ref(), &self.params, left1, right1);
        let new2 = Sequence::eval2(self.data.as_ref(), &self.params, left2, right2);
        let new_cost = new1 + new2;
        if new_cost <= max_acceptable_cost {
            let selected_delta = new_cost - old_cost;
            if apply {
                let mut suffix1 = self.routes[r1].nodes.split_off(pos1);
                let mut suffix2 = self.routes[r2].nodes.split_off(pos2);
                self.routes[r1].nodes.append(&mut suffix2);
                self.routes[r2].nodes.append(&mut suffix1);
                self.nb_moves += 1;
                self.update_route(r1);
                self.update_route(r2);
                let applied_delta = self.routes[r1].cost + self.routes[r2].cost - old_cost;
                self.cost += applied_delta;
                self.register_accepted_delta(applied_delta);
                Some(applied_delta)
            } else {
                Some(selected_delta)
            }
        } else { None }
    }

    pub fn run_2opt(&mut self, r1: usize, pos1: usize, apply: bool) -> Option<i64> {
        let route = &self.routes[r1];
        let data = self.data.as_ref();
        let len = route.nodes.len();
        if len < pos1 + 3 { return None; } // need at least [0, U, V, 0]

        debug_assert!(pos1 > 0); // U is a client
        debug_assert!(self.routes[r1].nodes[pos1].id != 0); // U is a client

        let old_cost = route.cost;
        let max_acceptable_cost = old_cost + self.move_credit;
        let mut best_cost = i64::MAX;
        let mut best_pos: Option<usize> = None;
        let old_distance = route.distance as i64;
        let cap_pen = ((route.load - self.data.max_capacity).max(0) as i64) * self.params.penalty_capa as i64;
        let max_distance_acceptable = max_acceptable_cost - cap_pen;
        let a_id = route.nodes[pos1 - 1].id;
        let u_id = route.nodes[pos1].id;
        let removed_au = data.dm(a_id, u_id);

        let mut mid_rev = route.nodes[pos1].seq21;
        for pos2 in (pos1 + 1)..(len - 1) {
            let v_id = route.nodes[pos2].id;
            let b_id = route.nodes[pos2 + 1].id;
            let delta_distance = data.dm(a_id, v_id) + data.dm(u_id, b_id) - removed_au - data.dm(v_id, b_id);
            if old_distance + (delta_distance as i64) > max_distance_acceptable {
                if pos2 + 1 < len - 1 { mid_rev = Sequence::join2(self.data.as_ref(), &route.nodes[pos2 + 1].seq1, &mid_rev); }
                continue;
            }
            let new_cost = Sequence::eval3(self.data.as_ref(), &self.params, &route.nodes[pos1 - 1].seq0_i, &mid_rev, &route.nodes[pos2 + 1].seqi_n);
            if new_cost <= max_acceptable_cost && new_cost < best_cost {
                best_cost = new_cost;
                best_pos = Some(pos2);
            }
            if pos2 + 1 < len - 1 { mid_rev = Sequence::join2(self.data.as_ref(), &route.nodes[pos2 + 1].seq1, &mid_rev); }
        }

        if let Some(mypos) = best_pos {
            let selected_delta = best_cost - old_cost;
            if apply {
                self.routes[r1].nodes[pos1..=mypos].reverse();
                self.nb_moves += 1;
                self.update_route(r1);
                let delta = self.routes[r1].cost - old_cost;
                self.cost += delta;
                self.register_accepted_delta(delta);
                Some(delta)
            } else {
                Some(selected_delta)
            }
        } else { None }
    }

    pub fn run_inter_route(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize, apply: bool) -> Option<i64> {
        let data = self.data.as_ref();
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
        let mut best_cost = i64::MAX;
        let max_acceptable_cost = old_total + self.move_credit;

        // Lower-bound prefilter (distance + load excess only), computed
        // directly from local arc rewiring + post-move loads.
        let pcap = self.params.penalty_capa as i64;
        let max_cap = data.max_capacity;
        let route1_dist = ru.distance;
        let route2_dist = rv.distance;
        let route_total_dist = route1_dist + route2_dist;
        let route1_load = ru.load;
        let route2_load = rv.load;
        let cap_pen = |load: i32| ((load - max_cap).max(0) as i64) * pcap;
        let to_cost = |dist_only: i32, cap_only: i64| (dist_only as i64) + cap_only;

        let mut best_lb = i64::MAX;
        let mut update_lb = |cand: i64| {
            if cand < best_lb {
                best_lb = cand;
            }
        };

        let send1_1_load = u.demand;
        let rem1_1 = u.dist_to_pred + u.dist_to_succ;
        let rem2_0 = v.dist_to_pred; // insertion before V breaks (Vpred,V)

        let d_upred_x = data.dm(u_pred.id, x.id);
        let d_vpred_u = data.dm(v_pred.id, u.id);
        let d_u_v = data.dm(u.id, v.id);
        let dist_base_10 = route_total_dist - rem1_1 - rem2_0;
        let cap_10 = cap_pen(route1_load - send1_1_load) + cap_pen(route2_load + send1_1_load);
        let result10_lb = to_cost(dist_base_10 + d_upred_x + d_vpred_u + d_u_v, cap_10);
        update_lb(result10_lb);

        // Send {U}, receive {V}
        if v.id != 0 {
            let y = &rv.nodes[pos2 + 1];
            let send2_1_load = v.demand;
            let rem2_1 = v.dist_to_pred + v.dist_to_succ;
            let d_upred_v = data.dm(u_pred.id, v.id);
            let d_v_x = data.dm(v.id, x.id);
            let d_u_y = data.dm(u.id, y.id);
            let dist_base_11 = route_total_dist - rem1_1 - rem2_1;
            let cap_11 = cap_pen(route1_load - send1_1_load + send2_1_load)
                + cap_pen(route2_load - send2_1_load + send1_1_load);
            let result11_lb = to_cost(dist_base_11 + d_upred_v + d_v_x + d_vpred_u + d_u_y, cap_11);
            update_lb(result11_lb);
        }

        if x.id != 0 {
            let x_next = &ru.nodes[pos1 + 2];
            let send1_2_load = u.demand + x.demand;
            let rem1_2 = u.dist_to_pred + u.dist_to_succ + x.dist_to_succ;
            let d_upred_xnext = data.dm(u_pred.id, x_next.id);
            let d_vpred_x = data.dm(v_pred.id, x.id);
            let d_x_u = data.dm(x.id, u.id);
            let d_u_x = u.dist_to_succ;
            let d_x_v = data.dm(x.id, v.id);
            let dist_base_20_30 = route_total_dist - rem1_2 - rem2_0;
            let cap_20_30 = cap_pen(route1_load - send1_2_load) + cap_pen(route2_load + send1_2_load);

            // Send {U,X} or {X,U}, receive {}
            let result20_lb = to_cost(dist_base_20_30 + d_upred_xnext + d_vpred_u + d_u_x + d_x_v, cap_20_30);
            let result30_lb = to_cost(dist_base_20_30 + d_upred_xnext + d_vpred_x + d_x_u + d_u_v, cap_20_30);
            update_lb(result20_lb);
            update_lb(result30_lb);

            if v.id != 0 {
                let y = &rv.nodes[pos2 + 1];
                let send2_1_load = v.demand;
                let rem2_1 = v.dist_to_pred + v.dist_to_succ;
                let d_upred_v = data.dm(u_pred.id, v.id);
                let d_v_xnext = data.dm(v.id, x_next.id);
                let d_x_y = data.dm(x.id, y.id);
                let d_u_y = data.dm(u.id, y.id);
                let dist_base_21_31 = route_total_dist - rem1_2 - rem2_1;
                let cap_21_31 = cap_pen(route1_load - send1_2_load + send2_1_load)
                    + cap_pen(route2_load - send2_1_load + send1_2_load);

                // Send {U,X} or {X,U}, receive {V}
                let result21_lb = to_cost(dist_base_21_31 + d_upred_v + d_v_xnext + d_vpred_u + d_u_x + d_x_y, cap_21_31);
                let result31_lb = to_cost(dist_base_21_31 + d_upred_v + d_v_xnext + d_vpred_x + d_x_u + d_u_y, cap_21_31);
                update_lb(result21_lb);
                update_lb(result31_lb);

                if y.id != 0 {
                    let y_next = &rv.nodes[pos2 + 2];
                    let send2_2_load = v.demand + y.demand;
                    let rem2_2 = v.dist_to_pred + v.dist_to_succ + y.dist_to_succ;
                    let d_upred_y = data.dm(u_pred.id, y.id);
                    let d_y_v = data.dm(y.id, v.id);
                    let d_y_xnext = data.dm(y.id, x_next.id);
                    let d_x_ynext = data.dm(x.id, y_next.id);
                    let d_u_ynext = data.dm(u.id, y_next.id);
                    let dist_base_22_33 = route_total_dist - rem1_2 - rem2_2;
                    let cap_22_33 = cap_pen(route1_load - send1_2_load + send2_2_load)
                        + cap_pen(route2_load - send2_2_load + send1_2_load);

                    // Send {U,X} or {X,U}, receive {V,Y} or {Y,V}
                    let result22_lb = to_cost(
                        dist_base_22_33
                            + d_upred_v + v.dist_to_succ + d_y_xnext
                            + d_vpred_u + d_u_x + d_x_ynext,
                        cap_22_33,
                    );
                    let result23_lb = to_cost(
                        dist_base_22_33
                            + d_upred_y + d_y_v + d_v_xnext
                            + d_vpred_u + d_u_x + d_x_ynext,
                        cap_22_33,
                    );
                    let result32_lb = to_cost(
                        dist_base_22_33
                            + d_upred_v + v.dist_to_succ + d_y_xnext
                            + d_vpred_x + d_x_u + d_u_ynext,
                        cap_22_33,
                    );
                    let result33_lb = to_cost(
                        dist_base_22_33
                            + d_upred_y + d_y_v + d_v_xnext
                            + d_vpred_x + d_x_u + d_u_ynext,
                        cap_22_33,
                    );
                    update_lb(result22_lb);
                    update_lb(result23_lb);
                    update_lb(result32_lb);
                    update_lb(result33_lb);
                }
            }

            // Send {U,X,Xnext}
            if x_next.id != 0 && self.params.allow_swap3 {
                let x2_next = &ru.nodes[pos1 + 3];
                let send1_3_load = u.demand + x.demand + x_next.demand;
                let rem1_3 = u.dist_to_pred + u.dist_to_succ + x.dist_to_succ + x_next.dist_to_succ;
                let d_upred_x2next = data.dm(u_pred.id, x2_next.id);
                let d_u_xnext = u.dist_to_succ;
                let d_x_xnext = x.dist_to_succ;
                let d_xnext_v = data.dm(x_next.id, v.id);
                let dist_base_40 = route_total_dist - rem1_3 - rem2_0;
                let cap_40 = cap_pen(route1_load - send1_3_load) + cap_pen(route2_load + send1_3_load);

                let result40_lb = to_cost(
                    dist_base_40 + d_upred_x2next + d_vpred_u + d_u_xnext + d_x_xnext + d_xnext_v,
                    cap_40,
                );
                update_lb(result40_lb);

                if v.id != 0 {
                    let y = &rv.nodes[pos2 + 1];
                    let send2_1_load = v.demand;
                    let rem2_1 = v.dist_to_pred + v.dist_to_succ;
                    let d_upred_v = data.dm(u_pred.id, v.id);
                    let d_v_x2next = data.dm(v.id, x2_next.id);
                    let d_xnext_y = data.dm(x_next.id, y.id);
                    let dist_base_41 = route_total_dist - rem1_3 - rem2_1;
                    let cap_41 = cap_pen(route1_load - send1_3_load + send2_1_load)
                        + cap_pen(route2_load - send2_1_load + send1_3_load);

                    let result41_lb = to_cost(
                        dist_base_41
                            + d_upred_v + d_v_x2next
                            + d_vpred_u + d_u_xnext + d_x_xnext + d_xnext_y,
                        cap_41,
                    );
                    update_lb(result41_lb);

                    if y.id != 0 {
                        let y_next = &rv.nodes[pos2 + 2];
                        let send2_2_load = v.demand + y.demand;
                        let rem2_2 = v.dist_to_pred + v.dist_to_succ + y.dist_to_succ;
                        let d_upred_y = data.dm(u_pred.id, y.id);
                        let d_y_v = data.dm(y.id, v.id);
                        let d_y_x2next = data.dm(y.id, x2_next.id);
                        let d_xnext_ynext = data.dm(x_next.id, y_next.id);
                        let dist_base_42_43 = route_total_dist - rem1_3 - rem2_2;
                        let cap_42_43 = cap_pen(route1_load - send1_3_load + send2_2_load)
                            + cap_pen(route2_load - send2_2_load + send1_3_load);

                        let result42_lb = to_cost(
                            dist_base_42_43
                                + d_upred_v + v.dist_to_succ + d_y_x2next
                                + d_vpred_u + d_u_xnext + d_x_xnext + d_xnext_ynext,
                            cap_42_43,
                        );
                        let result43_lb = to_cost(
                            dist_base_42_43
                                + d_upred_y + d_y_v + d_v_x2next
                                + d_vpred_u + d_u_xnext + d_x_xnext + d_xnext_ynext,
                            cap_42_43,
                        );
                        update_lb(result42_lb);
                        update_lb(result43_lb);

                        if y_next.id != 0 {
                            let y2_next = &rv.nodes[pos2 + 3];
                            let send2_4_load = v.demand + y.demand + y_next.demand;
                            let rem2_4 = v.dist_to_pred + v.dist_to_succ + y.dist_to_succ + y_next.dist_to_succ;
                            let d_ynext_x2next = data.dm(y_next.id, x2_next.id);
                            let d_xnext_y2next = data.dm(x_next.id, y2_next.id);
                            let dist_base_44 = route_total_dist - rem1_3 - rem2_4;
                            let cap_44 = cap_pen(route1_load - send1_3_load + send2_4_load)
                                + cap_pen(route2_load - send2_4_load + send1_3_load);

                            let result44_lb = to_cost(
                                dist_base_44
                                    + d_upred_v + v.dist_to_succ + y.dist_to_succ + d_ynext_x2next
                                    + d_vpred_u + d_u_xnext + d_x_xnext + d_xnext_y2next,
                                cap_44,
                            );
                            update_lb(result44_lb);
                        }
                    }
                }
            }
        }

        if best_lb > max_acceptable_cost {
            return None;
        }

        // tiny helper to track best
        let mut update_best = |i: usize, j: usize, cand: i64| {
            if cand <= max_acceptable_cost && cand < best_cost {
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

        if best_i == 0 && best_j == 0 { return None; }     // no improvement

        let selected_delta = best_cost - old_total;
        if !apply {
            return Some(selected_delta);
        }

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
        let delta = new_total - old_total;
        self.cost += delta;
        self.register_accepted_delta(delta);
        Some(delta)
    }

    pub fn run_swapstar(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize, apply: bool) -> Option<i64> {
        debug_assert!(r1 != r2);
        debug_assert!(pos1 > 0 && pos1 < self.routes[r1].nodes.len() - 1);
        debug_assert!(pos2 > 0 && pos2 < self.routes[r2].nodes.len() - 1);

        let route1_nodes = &self.routes[r1].nodes;
        let route2_nodes = &self.routes[r2].nodes;
        let node_u = &route1_nodes[pos1];
        let node_v = &route2_nodes[pos2];

        // First filter on route costs
        let new_load1 = node_u.seqno1.load + node_v.demand;
        let new_load2 = node_v.seqno1.load + node_u.demand;
        let new_pen1 = ((new_load1 - self.data.max_capacity).max(0) as i64) * self.params.penalty_capa as i64;
        let new_pen2 = ((new_load2 - self.data.max_capacity).max(0) as i64) * self.params.penalty_capa as i64;
        let cost_lb_r1_after_removal = (node_u.seqno1.distance as i64) + new_pen1;
        let cost_lb_r2_after_removal = (node_v.seqno1.distance as i64) + new_pen2;
        let mut lb_new_total = cost_lb_r1_after_removal + cost_lb_r2_after_removal;
        let old_total = self.routes[r1].cost + self.routes[r2].cost;
        let max_acceptable_cost = old_total + self.move_credit;

        // first filter on route costs
        if lb_new_total > max_acceptable_cost { return None; }

        // Values needed only beyond first filter
        let data = self.data.as_ref();
        let route1_len = route1_nodes.len();
        let route2_len = route2_nodes.len();
        let u = node_u.id;
        let v = node_v.id;
        let (pu, nu) = (route1_nodes[pos1 - 1].id, route1_nodes[pos1 + 1].id);
        let (pv, nv) = (route2_nodes[pos2 - 1].id, route2_nodes[pos2 + 1].id);
        let d_pu_nu = data.dm(pu, nu);
        let d_pv_nv = data.dm(pv, nv);

        // best detour for inserting v into r1 \ {u}
        let mut best_ins_v = data.dm(pu, v) + data.dm(v, nu) - d_pu_nu; // insertion at U place
        for t in 1..pos1 {
            let a_id = route1_nodes[t - 1].id;
            let b_id = route1_nodes[t].id;
            let delta = data.dm(a_id, v) + data.dm(v, b_id) - route1_nodes[t - 1].dist_to_succ;
            if delta < best_ins_v { best_ins_v = delta; }
        }
        for t in (pos1 + 2)..route1_len {
            let a_id = route1_nodes[t - 1].id;
            let b_id = route1_nodes[t].id;
            let delta = data.dm(a_id, v) + data.dm(v, b_id) - route1_nodes[t - 1].dist_to_succ;
            if delta < best_ins_v { best_ins_v = delta; }
        }

        // Second filter on route costs
        lb_new_total += best_ins_v as i64;
        if lb_new_total > max_acceptable_cost { return None; }

        // best detour for inserting u into r2 \ {v}
        let mut best_ins_u = data.dm(pv, u) + data.dm(u, nv) - d_pv_nv; // insertion at V place
        for t in 1..pos2 {
            let a_id = route2_nodes[t - 1].id;
            let b_id = route2_nodes[t].id;
            let delta = data.dm(a_id, u) + data.dm(u, b_id) - route2_nodes[t - 1].dist_to_succ;
            if delta < best_ins_u { best_ins_u = delta; }
        }
        for t in (pos2 + 2)..route2_len {
            let a_id = route2_nodes[t - 1].id;
            let b_id = route2_nodes[t].id;
            let delta = data.dm(a_id, u) + data.dm(u, b_id) - route2_nodes[t - 1].dist_to_succ;
            if delta < best_ins_u { best_ins_u = delta; }
        }

        // Third filter on route costs
        lb_new_total += best_ins_u as i64;
        if lb_new_total > max_acceptable_cost { return None; }

        // Reinsertion of V into r1 \ {U}
        let v_seq1 = route2_nodes[pos2].seq1;
        let mut best_t1: usize = pos1;
        let mut best_cost1 = i64::MAX / 4;

        // t <= pos1: build right-excluding-U by prepending seq1 fragments
        let mut right_excl_u = route1_nodes[pos1 + 1].seqi_n;
        for t in (1..=pos1).rev() {
            let left = route1_nodes[t - 1].seq0_i;
            let cand = Sequence::eval3(data, &self.params, &left, &v_seq1, &right_excl_u);
            if cand < best_cost1 { best_cost1 = cand; best_t1 = t; }
            if t > 1 {
                right_excl_u = Sequence::join2(data, &route1_nodes[t - 1].seq1, &right_excl_u);
            }
        }

        // t > pos1: build left-excluding-U incrementally
        let mut left_excl_u = Sequence::join2(data, &route1_nodes[pos1 - 1].seq0_i, &route1_nodes[pos1 + 1].seq1);
        for t in (pos1 + 2)..route1_len {
            let cand = Sequence::eval3(data, &self.params, &left_excl_u, &v_seq1, &route1_nodes[t].seqi_n);
            if cand < best_cost1 { best_cost1 = cand; best_t1 = t; }
            if t + 1 < route1_len {
                left_excl_u = Sequence::join2(data, &left_excl_u, &route1_nodes[t].seq1);
            }
        }

        // Fourth filter: one route is exact (TW-aware), the other remains a lower bound
        let lb_cost2 = cost_lb_r2_after_removal + best_ins_u as i64;
        if best_cost1.saturating_add(lb_cost2) > max_acceptable_cost {
            return None;
        }

        // Reinsertion of U into r2 \ {V}
        let u_seq1 = route1_nodes[pos1].seq1;
        let mut best_t2: usize = pos2;
        let mut best_cost2 = i64::MAX / 4;

        // t <= pos2: build right-excluding-V by prepending seq1 fragments
        let mut right_excl_v = route2_nodes[pos2 + 1].seqi_n;
        for t in (1..=pos2).rev() {
            let left = route2_nodes[t - 1].seq0_i;
            let cand = Sequence::eval3(data, &self.params, &left, &u_seq1, &right_excl_v);
            if cand < best_cost2 { best_cost2 = cand; best_t2 = t; }
            if t > 1 {
                right_excl_v = Sequence::join2(data, &route2_nodes[t - 1].seq1, &right_excl_v);
            }
        }

        // t > pos2: build left-excluding-V incrementally
        let mut left_excl_v = Sequence::join2(data, &route2_nodes[pos2 - 1].seq0_i, &route2_nodes[pos2 + 1].seq1);
        for t in (pos2 + 2)..route2_len {
            let cand = Sequence::eval3(data, &self.params, &left_excl_v, &u_seq1, &route2_nodes[t].seqi_n);
            if cand < best_cost2 { best_cost2 = cand; best_t2 = t; }
            if t + 1 < route2_len {
                left_excl_v = Sequence::join2(data, &left_excl_v, &route2_nodes[t].seq1);
            }
        }

        let new_total = best_cost1 + best_cost2;
        if new_total > max_acceptable_cost {
            return None;
        }
        if !apply {
            return Some(new_total - old_total);
        }

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
        let delta = new_total - old_total;
        self.cost += delta;
        self.register_accepted_delta(delta);
        Some(delta)
    }

    pub fn continue_repair(
        &mut self,
        rng: &mut SmallRng,
        params: Params,
        factor: usize,
    ) -> Vec<Vec<usize>> {
        self.params = params;
        debug_assert!(!self.routes.is_empty(), "continue_repair requires a loaded LS state");
        self.params.penalty_tw = factor.saturating_mul(self.params.penalty_tw).min(10_000);
        self.params.penalty_capa = factor.saturating_mul(self.params.penalty_capa).min(10_000);
        self.nb_moves += 1;
        for rid in 0..self.routes.len() {
            self.when_last_modified[rid] = 0;
            let r = &self.routes[rid];
            if r.load > self.data.max_capacity || r.tw > 0 {
                self.update_route(rid);
            }
        }
        self.search(rng);
        self.export_routes()
    }

    #[inline(always)]
    fn evaluate_and_apply_best_move_for_customer(
        &mut self,
        c1: usize,
        last_tested: usize,
        loop_id: usize,
    ) -> Option<i64> {
        let mut best_delta: i64 = i64::MAX; // best acceptable move
        let mut best_move: Option<CandidateMove> = None;

        {
            let r1 = self.node_route[c1];
            let pos1 = self.node_pos[c1];
            let r1_last_mod = self.when_last_modified[r1];
            let need_r2_stale_check = r1_last_mod <= last_tested;
            for k in 0..self.neighbors_before[c1].len() {
                let c2 = self.neighbors_before[c1][k];
                let r2 = self.node_route[c2];

                // Skip if both routes unchanged since last tests for this customer
                if r1 != r2 && !(need_r2_stale_check && self.when_last_modified[r2] <= last_tested) {
                    // We use pos2 + 1 for the SWAP and RELOCATE moves since c2 is a good predecessor for c1
                    // Moves listed here create the edge c2 => c1, but never insert immediately after a depot
                    let pos2 = self.node_pos[c2];

                    let cand_inter = CandidateMove::InterRoute { r1, pos1, r2, pos2: pos2 + 1 };
                    if let Some(delta) = self.eval_candidate_delta(cand_inter) {
                        if delta < best_delta {
                            best_delta = delta;
                            best_move = Some(cand_inter);
                        }
                    }

                    // Special case to manage insert immediately after a depot
                    if pos1 == 1 {
                        let cand_inter_depot = CandidateMove::InterRoute { r1: r2, pos1: pos2, r2: r1, pos2: pos1 };
                        if let Some(delta) = self.eval_candidate_delta(cand_inter_depot) {
                            if delta < best_delta {
                                best_delta = delta;
                                best_move = Some(cand_inter_depot);
                            }
                        }
                    }

                    let cand_2opt = CandidateMove::TwoOptStar { r1, pos1, r2, pos2: pos2 + 1 };
                    if let Some(delta) = self.eval_candidate_delta(cand_2opt) {
                        if delta < best_delta {
                            best_delta = delta;
                            best_move = Some(cand_2opt);
                        }
                    }
                }
            }

            for k in 0..self.neighbors_capacity_swap[c1].len() {
                let c2 = self.neighbors_capacity_swap[c1][k];
                let r2 = self.node_route[c2];

                // Skip if both routes unchanged since last tests for this customer
                if r1 != r2 && !(need_r2_stale_check && self.when_last_modified[r2] <= last_tested) {
                    let pos2 = self.node_pos[c2];
                    let cand_swap = CandidateMove::SwapStar { r1, pos1, r2, pos2 };
                    if let Some(delta) = self.eval_candidate_delta(cand_swap) {
                        if delta < best_delta {
                            best_delta = delta;
                            best_move = Some(cand_swap);
                        }
                    }
                }
            }

            // Moves involving an empty route (only tested after the first loop)
            if loop_id > 1 && (loop_id == 2 || r1_last_mod > last_tested) {
                if let Some(&r2) = self.empty_routes.first() {
                    let pos2 = 1;

                    // 2-opt* with an empty route (essentially cut the route in 2)
                    // Skip whole-route transfer to another route index.
                    if pos1 > 1 {
                        let cand_empty_2opt = CandidateMove::TwoOptStar { r1, pos1, r2, pos2 };
                        if let Some(delta) = self.eval_candidate_delta(cand_empty_2opt) {
                            if delta < best_delta {
                                best_delta = delta;
                                best_move = Some(cand_empty_2opt);
                            }
                        }
                    }

                    // Insert in an empty route
                    let cand_empty_inter = CandidateMove::InterRoute { r1, pos1, r2, pos2 };
                    if let Some(delta) = self.eval_candidate_delta(cand_empty_inter) {
                        if delta < best_delta {
                            best_delta = delta;
                            best_move = Some(cand_empty_inter);
                        }
                    }
                }
            }
        }

        // Intra-route moves
        let r1 = self.node_route[c1];
        let pos1 = self.node_pos[c1];
        if self.when_last_modified[r1] > last_tested {
            let cand_intra_reloc = CandidateMove::IntraRelocate { r1, pos1 };
            if let Some(delta) = self.eval_candidate_delta(cand_intra_reloc) {
                if delta < best_delta {
                    best_delta = delta;
                    best_move = Some(cand_intra_reloc);
                }
            }

            let cand_intra_oropt2 = CandidateMove::IntraOrOpt2 { r1, pos1 };
            if let Some(delta) = self.eval_candidate_delta(cand_intra_oropt2) {
                if delta < best_delta {
                    best_delta = delta;
                    best_move = Some(cand_intra_oropt2);
                }
            }

            let cand_intra_swap = CandidateMove::IntraSwap { r1, pos1 };
            if let Some(delta) = self.eval_candidate_delta(cand_intra_swap) {
                if delta < best_delta {
                    best_delta = delta;
                    best_move = Some(cand_intra_swap);
                }
            }

            let cand_intra_2opt = CandidateMove::Intra2Opt { r1, pos1 };
            if let Some(delta) = self.eval_candidate_delta(cand_intra_2opt) {
                if delta < best_delta {
                    best_delta = delta;
                    best_move = Some(cand_intra_2opt);
                }
            }
        }

        if let Some(mv) = best_move {
            let applied_delta = self.apply_candidate_move(mv);
            debug_assert!(applied_delta.is_some(), "Best candidate move was expected to be applicable");
            let Some(delta) = applied_delta else { return None; };
            debug_assert_eq!(delta, best_delta, "Applied move delta differs from evaluated best delta");
            Some(delta)
        } else {
            None
        }
    }

    fn search(&mut self, rng: &mut SmallRng) {
        let mut improved = true;
        let mut loop_id = 0;
        self.move_credit = 0;
        self.loop_order_nodes.shuffle(rng);
        while improved || loop_id < 2 {
            improved = false;
            loop_id += 1;
            if loop_id == 2 {
                self.move_credit = -1;
            }
            for idx in 0..self.loop_order_nodes.len() {
                let c1 = self.loop_order_nodes[idx];
                let mut c1_repeat = true;
                while c1_repeat {
                    c1_repeat = false;
                    let last_tested = self.when_last_tested[c1];
                    self.when_last_tested[c1] = self.nb_moves;
                    if let Some(delta) = self.evaluate_and_apply_best_move_for_customer(c1, last_tested, loop_id) {
                        improved = true;
                        c1_repeat = delta < 0;
                    }
                }
            }
        }
    }
}
