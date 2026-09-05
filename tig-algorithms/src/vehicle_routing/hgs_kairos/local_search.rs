use super::params::Params;
use super::problem::Problem;
use super::sequence::Sequence;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use std::cmp::min;
use std::sync::Arc;

const NO_MOVE: i64 = i64::MAX;

/// Caller guarantees `index < slice.len()`; only the debug build checks it.
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

/// Cached sequences for the node at position `i` of a route of length `len`:
/// prefix `[0..=i]`, suffix `[i..=len-1]`, the node alone, then the forward
/// pair, the reversed pair and the forward triple starting at `i`.
/// `update_route` refreshes the pair fields only for `i <= len - 2` and the
/// triple only for `i <= len - 3`; outside those ranges they hold stale values.
#[derive(Clone, Debug, Default)]
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
            // Depends only on the problem and the id, so it survives a move.
            seq1: Sequence::singleton(data, id),
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
    #[inline(always)]
    fn node(&self, position: usize) -> &Node {
        debug_assert!(position < self.nodes.len());
        unsafe { self.nodes.get_unchecked(position) }
    }
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
    // Both stamps are compared against `nb_moves`, which never decreases.
    pub when_last_modified: Vec<usize>, // indexed by route
    pub when_last_tested: Vec<usize>,   // indexed by customer id
    pub nb_moves: usize,
    pub move_credit: i64, // deterioration budget; negative means disabled
    last_plan: MovePlan,
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
    pub fn new(data: Arc<Problem>, params: Params, _rng: &mut SmallRng) -> Self {
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
        let max_demand = data
            .node_data
            .iter()
            .map(|nd| nd.demand)
            .max()
            .unwrap_or(0)
            .max(0) as usize;
        let mut ids_by_demand = vec![Vec::<usize>::new(); max_demand + 1];
        for j in 1..n {
            ids_by_demand[data.nd(j).demand as usize].push(j);
        }
        for i in 1..n {
            let ndi = data.nd(i);
            let di = ndi.demand;
            let distance_from_i = data.distance_row(i);
            let distance_to_i = data.distance_column(i);
            prox.clear();
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
            }

            if keep > 0 {
                if keep < prox.len() {
                    prox.select_nth_unstable(keep);
                }
                prox[..keep].sort_unstable();
                neighbors_before.extend(prox[..keep].iter().map(|&key| (key & 0xffff) as usize));
            }
            neighbors_before_offsets[i + 1] = neighbors_before.len();

            let keep_similar = (params.swapstar_capa_filter * cap as f64).ceil() as usize;

            capacity_prox.clear();
            // Buckets are walked in (|d_j - d_i|, client id) order; the sort key
            // packs distance:32 | demand gap:16 | client id:16.
            let di_usize = di as usize;
            for dd in 0..=max_demand {
                if capacity_prox.len() >= keep_similar {
                    break;
                }
                let low = di_usize.checked_sub(dd).filter(|&d| d <= max_demand);
                let high_value = di_usize + dd;
                let high =
                    (high_value <= max_demand && Some(high_value) != low).then_some(high_value);
                let low_ids = low.map(|d| ids_by_demand[d].as_slice()).unwrap_or(&[]);
                let high_ids = high.map(|d| ids_by_demand[d].as_slice()).unwrap_or(&[]);
                let mut il = 0usize;
                let mut ih = 0usize;
                while capacity_prox.len() < keep_similar
                    && (il < low_ids.len() || ih < high_ids.len())
                {
                    let take_low =
                        ih >= high_ids.len() || (il < low_ids.len() && low_ids[il] < high_ids[ih]);
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
            last_plan: MovePlan::None,
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
                let node_u = route_at(&self.routes, r1).node(pos1).clone();
                let node_v = route_at(&self.routes, r2).node(pos2).clone();
                self.routes[r1].nodes.remove(pos1);
                self.routes[r2].nodes.remove(pos2);
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

        // Normalize to exactly `fleet` routes: pad with depot-only routes, or
        // merge every surplus route into the last kept one.
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

        // Cached sequences are refreshed by `update_route` below, so only the
        // customer ids are rebuilt here.
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
        let mut out: Vec<Vec<usize>> = self
            .routes
            .iter()
            .filter(|r| r.nodes.len() > 2)
            .map(|r| r.nodes.iter().map(|n| n.id).collect::<Vec<usize>>())
            .collect();

        // Symmetric (non-VRPTW) instances: pin one orientation per route so that
        // two mirror-image routes compare equal downstream.
        if !self.data.is_vrptw {
            for route in &mut out {
                if route.len() == 4 {
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

    /// (distance, time-window violation, load excess) of the loaded routes.
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

    #[inline]
    fn is_counter_clockwise(data: &Problem, route: &[usize]) -> bool {
        if route.len() < 4 {
            return false;
        }
        // Shoelace over the closed polyline [0, ..., 0] as stored in the route.
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
            let next2 = unsafe { (*ptr.add(pos + 2)).seq1 };
            let travel = unsafe { (*ptr.add(pos + 1)).dist_to_succ };
            let seq123 = Sequence::join2_with_travel(&seq12, &next2, travel);
            unsafe {
                (*ptr.add(pos)).seqi_n = acc_bwd;
                (*ptr.add(pos)).seq12 = seq12;
                (*ptr.add(pos)).seq21 = seq21;
                (*ptr.add(pos)).seq123 = seq123;
            }
        }

        let end = unsafe { (*ptr.add(len - 1)).seq0_i };
        r.load = end.load;
        r.tw = end.tw;
        r.distance = end.distance;
        r.cost = end.eval(data, &self.params);

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
        unsafe {
            *self.when_last_modified.get_unchecked_mut(rid) = self.nb_moves;
        }
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
        let max_acceptable_cost = old_cost + self.move_credit;
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

        // Insert U before t, t in [1 .. pos1-1]
        if pos1 > 1 {
            let mut right_excl_u = route.node(pos1 + 1).seqi_n;
            for t in (1..pos1).rev() {
                // Only the step adjacent to U crosses the bridge arc A-B.
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

        // Insert U before t, t in [pos1+2 .. len-1]
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
        let max_acceptable_cost = old_cost + self.move_credit;
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

        // Insert (U,X) or (X,U) before t, t in [1 .. pos1-1]
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

        // Insert (U,X) or (X,U) before t, t in [pos1+3 .. len-1]
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
        // A non-adjacent swap needs at least three clients.
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
        let max_acceptable_cost = old_cost + self.move_credit;
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

        // Distance-only prefilter: on each side, find the first position whose
        // distance delta alone can still fit under the acceptance bar.
        let mut first_right_potential: Option<usize> = None;
        if has_right {
            for pos2 in (pos1 + 2)..(len - 1) {
                let pv = route.node(pos2 - 1).id;
                let v = route.node(pos2).id;
                let nv = route.node(pos2 + 1).id;
                let removed_v = route.node(pos2 - 1).dist_to_succ + route.node(pos2).dist_to_succ;
                let delta_distance = (copy_at(distance_from_pu, v) + copy_at(distance_to_nu, v)
                    - removed_u)
                    + (copy_at(distance_to_u, pv) + copy_at(distance_from_u, nv) - removed_v);
                if old_distance + (delta_distance as i64) <= max_distance_acceptable {
                    first_right_potential = Some(pos2);
                    break;
                }
            }
        }
        let mut first_left_potential: Option<usize> = None;
        if has_left {
            for pos2 in (1..=pos1 - 2).rev() {
                let pv = route.node(pos2 - 1).id;
                let v = route.node(pos2).id;
                let nv = route.node(pos2 + 1).id;
                let removed_v = route.node(pos2 - 1).dist_to_succ + route.node(pos2).dist_to_succ;
                let delta_distance = (copy_at(distance_from_pu, v) + copy_at(distance_to_nu, v)
                    - removed_u)
                    + (copy_at(distance_to_u, pv) + copy_at(distance_from_u, nv) - removed_v);
                if old_distance + (delta_distance as i64) <= max_distance_acceptable {
                    first_left_potential = Some(pos2);
                    break;
                }
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
        let max_acceptable_cost = old_cost + self.move_credit;

        let left1 = &route1.node(pos1 - 1).seq0_i;
        let right1 = &route2.node(pos2).seqi_n;
        let left2 = &route2.node(pos2 - 1).seq0_i;
        let right2 = &route1.node(pos1).seqi_n;

        // Prefilter on distance and load excess only, time windows excluded.
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
        // A reversal needs at least [0, U, V, 0].
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

        // `mid_rev` grows through `join_tw_with_travel`, which zeroes `distance`;
        // the true reversed-segment length is tracked in `mid_rev_distance`.
        let mut mid_rev = route.node(pos1).seq21;
        let mut mid_rev_distance = mid_rev.distance;
        for pos2 in (pos1 + 1)..(len - 1) {
            let v_id = route.node(pos2).id;
            let b_id = route.node(pos2 + 1).id;
            let d_av = copy_at(distance_from_a, v_id);
            let d_ub = copy_at(distance_from_u, b_id);
            let left = route.node(pos1 - 1).seq0_i;
            let right = route.node(pos2 + 1).seqi_n;
            // Exact new distance: the reversed middle carries its own internal
            // arcs, which differ from the forward ones on asymmetric matrices.
            let new_distance = left.distance + mid_rev_distance + right.distance + d_av + d_ub;
            // Admission is decided on the four boundary arcs only, never on
            // `new_distance`: the two are not interchangeable.
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
            let max_acceptable_cost = old_total + self.move_credit;
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
        let max_acceptable_cost = old_total + self.move_credit;
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
    fn run_swapstar(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize) -> i64 {
        debug_assert!(r1 != r2);
        debug_assert!(pos1 > 0 && pos1 < self.routes[r1].nodes.len() - 1);
        debug_assert!(pos2 > 0 && pos2 < self.routes[r2].nodes.len() - 1);

        // Removing one customer has an exact three-arc distance delta; only
        // that scalar and the new loads are needed before the filters below.
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

        // Filter 1: both routes after removal, distance and capacity only.
        if lb_new_total > max_acceptable_cost {
            return NO_MOVE;
        }

        let data = self.data.as_ref();
        let route1_nodes = &route_at(&self.routes, r1).nodes;
        let route2_nodes = &route_at(&self.routes, r2).nodes;
        let route1_len = route1_nodes.len();
        let route2_len = route2_nodes.len();
        let distance_from_u = data.distance_row(u);
        let distance_to_u = data.distance_column(u);
        let distance_from_v = data.distance_row(v);
        let distance_to_v = data.distance_column(v);

        // Cheapest distance detour for reinserting one customer once the other
        // is gone. `live` counts the slots that survive the removal, `split` is
        // the first slot index that lands past the hole, and slot `end` is
        // repeated rather than guarded so the scan can be unrolled.
        macro_rules! detour {
            ($best:ident, $nodes:expr, $to:expr, $from:expr, $end:expr, $split:expr, $s:expr) => {{
                let s = min($s, $end);
                let t = if s < $split { s + 1 } else { s + 3 };
                let pred = node_at($nodes, t - 1);
                let next = node_at($nodes, t);
                $best = $best.min(
                    copy_at($to, pred.id) + copy_at($from, next.id) - pred.dist_to_succ,
                );
            }};
        }
        macro_rules! scan_detours {
            ($best:ident, $nodes:expr, $to:expr, $from:expr, $live:expr, $split:expr) => {{
                let end = $live.wrapping_sub(1);
                let mut i = 0usize;
                while i < $live {
                    detour!($best, $nodes, $to, $from, end, $split, i + 0);
                    detour!($best, $nodes, $to, $from, end, $split, i + 1);
                    detour!($best, $nodes, $to, $from, end, $split, i + 2);
                    detour!($best, $nodes, $to, $from, end, $split, i + 3);
                    detour!($best, $nodes, $to, $from, end, $split, i + 4);
                    detour!($best, $nodes, $to, $from, end, $split, i + 5);
                    detour!($best, $nodes, $to, $from, end, $split, i + 6);
                    detour!($best, $nodes, $to, $from, end, $split, i + 7);
                    detour!($best, $nodes, $to, $from, end, $split, i + 8);
                    detour!($best, $nodes, $to, $from, end, $split, i + 9);
                    detour!($best, $nodes, $to, $from, end, $split, i + 10);
                    detour!($best, $nodes, $to, $from, end, $split, i + 11);
                    i += 12;
                }
            }};
        }

        // The arcs incident to the removed customer are replaced by its bridge
        // arc, which is therefore seeded here instead of being scanned.
        let pred_u = node_at(route1_nodes, pos1 - 1);
        let next_u = node_at(route1_nodes, pos1 + 1);
        let mut best_ins_v =
            copy_at(distance_to_v, pred_u.id) + copy_at(distance_from_v, next_u.id) - bridge_u;
        let live_r1 = route1_len - 3;
        let split_r1 = pos1 - 1;
        scan_detours!(
            best_ins_v,
            route1_nodes,
            distance_to_v,
            distance_from_v,
            live_r1,
            split_r1
        );

        // Filter 2: r1 now carries its cheapest reinsertion of V.
        lb_new_total += best_ins_v as i64;
        if lb_new_total > max_acceptable_cost {
            return NO_MOVE;
        }

        let pred_v = node_at(route2_nodes, pos2 - 1);
        let next_v = node_at(route2_nodes, pos2 + 1);
        let mut best_ins_u =
            copy_at(distance_to_u, pred_v.id) + copy_at(distance_from_u, next_v.id) - bridge_v;
        let live_r2 = route2_len - 3;
        let split_r2 = pos2 - 1;
        scan_detours!(
            best_ins_u,
            route2_nodes,
            distance_to_u,
            distance_from_u,
            live_r2,
            split_r2
        );

        // Filter 3: r2 now carries its cheapest reinsertion of U.
        lb_new_total += best_ins_u as i64;
        if lb_new_total > max_acceptable_cost {
            return NO_MOVE;
        }

        // Lower bound kept for r2 while r1 is scanned exactly, and vice versa.
        let lb_cost2 = cost_lb_r2_after_removal + best_ins_u as i64;

        let ptw = self.params.penalty_tw as i64;

        // Reinsertion of V into r1 without U.
        let v_seq1 = node_at(route2_nodes, pos2).seq1;
        let mut best_t1: usize = pos1;
        let mut best_cost1 = i64::MAX / 4;

        // t <= pos1: grow the U-free suffix by prepending singletons.
        let mut right_excl_u = node_at(route1_nodes, pos1 + 1).seqi_n;
        for t in (1..pos1 + 1).rev() {
            let left = node_at(route1_nodes, t - 1).seq0_i;
            let d_av = copy_at(distance_to_v, left.last_node as usize);
            let d_vb = copy_at(distance_from_v, right_excl_u.first_node as usize);
            let distance = left.distance + right_excl_u.distance + d_av + d_vb;
            let route1_lb = (distance as i64) + new_pen1;
            let ok = (route1_lb < best_cost1) & (route1_lb + lb_cost2 <= max_acceptable_cost);
            let tw = Sequence::tw3_with_travel(&left, &v_seq1, &right_excl_u, d_av, d_vb);
            let cand = route1_lb + (tw as i64) * ptw;
            let better = ok & (cand < best_cost1);
            best_cost1 = if better { cand } else { best_cost1 };
            best_t1 = if better { t } else { best_t1 };
            let prev = node_at(route1_nodes, t - 1);
            let bridge_mask = -((t == pos1) as i32);
            let travel = (bridge_u & bridge_mask) | (prev.dist_to_succ & !bridge_mask);
            right_excl_u = Sequence::join2_with_travel(&prev.seq1, &right_excl_u, travel);
        }

        // t > pos1: grow the U-free prefix by appending singletons.
        let mut left_excl_u = Sequence::join2_with_travel(
            &node_at(route1_nodes, pos1 - 1).seq0_i,
            &node_at(route1_nodes, pos1 + 1).seq1,
            bridge_u,
        );
        for t in (pos1 + 2)..route1_len {
            let right = node_at(route1_nodes, t).seqi_n;
            let d_av = copy_at(distance_to_v, left_excl_u.last_node as usize);
            let d_vb = copy_at(distance_from_v, right.first_node as usize);
            let distance = left_excl_u.distance + right.distance + d_av + d_vb;
            let route1_lb = (distance as i64) + new_pen1;
            let ok = (route1_lb < best_cost1) & (route1_lb + lb_cost2 <= max_acceptable_cost);
            let tw = Sequence::tw3_with_travel(&left_excl_u, &v_seq1, &right, d_av, d_vb);
            let cand = route1_lb + (tw as i64) * ptw;
            let better = ok & (cand < best_cost1);
            best_cost1 = if better { cand } else { best_cost1 };
            best_t1 = if better { t } else { best_t1 };
            left_excl_u = Sequence::join2_with_travel(
                &left_excl_u,
                &node_at(route1_nodes, t).seq1,
                node_at(route1_nodes, t - 1).dist_to_succ,
            );
        }

        // Filter 4: r1 exact including time windows, r2 still a lower bound.
        if best_cost1.saturating_add(lb_cost2) > max_acceptable_cost {
            return NO_MOVE;
        }

        // Reinsertion of U into r2 without V.
        let u_seq1 = node_at(route1_nodes, pos1).seq1;
        let mut best_t2: usize = pos2;
        let mut best_cost2 = i64::MAX / 4;

        // t <= pos2: grow the V-free suffix by prepending singletons.
        let mut right_excl_v = node_at(route2_nodes, pos2 + 1).seqi_n;
        for t in (1..pos2 + 1).rev() {
            let left = node_at(route2_nodes, t - 1).seq0_i;
            let d_au = copy_at(distance_to_u, left.last_node as usize);
            let d_ub = copy_at(distance_from_u, right_excl_v.first_node as usize);
            let distance = left.distance + right_excl_v.distance + d_au + d_ub;
            let route2_lb = (distance as i64) + new_pen2;
            let ok = (route2_lb < best_cost2) & (best_cost1 + route2_lb <= max_acceptable_cost);
            let tw = Sequence::tw3_with_travel(&left, &u_seq1, &right_excl_v, d_au, d_ub);
            let cand = route2_lb + (tw as i64) * ptw;
            let better = ok & (cand < best_cost2);
            best_cost2 = if better { cand } else { best_cost2 };
            best_t2 = if better { t } else { best_t2 };
            let prev = node_at(route2_nodes, t - 1);
            let bridge_mask = -((t == pos2) as i32);
            let travel = (bridge_v & bridge_mask) | (prev.dist_to_succ & !bridge_mask);
            right_excl_v = Sequence::join2_with_travel(&prev.seq1, &right_excl_v, travel);
        }

        // t > pos2: grow the V-free prefix by appending singletons.
        let mut left_excl_v = Sequence::join2_with_travel(
            &node_at(route2_nodes, pos2 - 1).seq0_i,
            &node_at(route2_nodes, pos2 + 1).seq1,
            bridge_v,
        );
        for t in (pos2 + 2)..route2_len {
            let right = node_at(route2_nodes, t).seqi_n;
            let d_au = copy_at(distance_to_u, left_excl_v.last_node as usize);
            let d_ub = copy_at(distance_from_u, right.first_node as usize);
            let distance = left_excl_v.distance + right.distance + d_au + d_ub;
            let route2_lb = (distance as i64) + new_pen2;
            let ok = (route2_lb < best_cost2) & (best_cost1 + route2_lb <= max_acceptable_cost);
            let tw = Sequence::tw3_with_travel(&left_excl_v, &u_seq1, &right, d_au, d_ub);
            let cand = route2_lb + (tw as i64) * ptw;
            let better = ok & (cand < best_cost2);
            best_cost2 = if better { cand } else { best_cost2 };
            best_t2 = if better { t } else { best_t2 };
            left_excl_v = Sequence::join2_with_travel(
                &left_excl_v,
                &node_at(route2_nodes, t).seq1,
                node_at(route2_nodes, t - 1).dist_to_succ,
            );
        }

        let new_total = best_cost1 + best_cost2;
        if new_total > max_acceptable_cost {
            return NO_MOVE;
        }
        self.last_plan = MovePlan::SwapStar {
            insert1: best_t1,
            insert2: best_t2,
        };
        new_total - old_total
    }

    pub fn continue_repair(
        &mut self,
        rng: &mut SmallRng,
        params: Params,
        factor: usize,
    ) -> Vec<Vec<usize>> {
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
    ) -> i64 {
        let mut best_delta: i64 = i64::MAX; // best acceptable move
        let mut best_move: Option<CandidateMove> = None;
        let mut best_plan = MovePlan::None;
        let r1 = copy_at(&self.node_route, c1);
        let pos1 = copy_at(&self.node_pos, c1);

        {
            let r1_last_mod = copy_at(&self.when_last_modified, r1);
            let need_r2_stale_check = r1_last_mod <= last_tested;
            let neighbors_start = copy_at(&self.neighbors_before_offsets, c1);
            let neighbors_end = copy_at(&self.neighbors_before_offsets, c1 + 1);
            // Admission is decided for a group of eight neighbours at a time,
            // then the admitted ones are visited in increasing index order.
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
                    {
                        let pos2 = copy_at(&self.node_pos, c2);

                        let delta = self.run_inter_route(r1, pos1, r2, pos2 + 1);
                        if delta < best_delta {
                            best_delta = delta;
                            best_move = Some(CandidateMove::InterRoute {
                                r1,
                                pos1,
                                r2,
                                pos2: pos2 + 1,
                            });
                            best_plan = self.last_plan;
                        }

                        if $head {
                            let delta = self.run_inter_route(r2, pos2, r1, pos1);
                            if delta < best_delta {
                                best_delta = delta;
                                best_move = Some(CandidateMove::InterRoute {
                                    r1: r2,
                                    pos1: pos2,
                                    r2: r1,
                                    pos2: pos1,
                                });
                                best_plan = self.last_plan;
                            }
                        }

                        let delta = self.run_2optstar(r1, pos1, r2, pos2 + 1);
                        if delta < best_delta {
                            best_delta = delta;
                            best_move = Some(CandidateMove::TwoOptStar {
                                r1,
                                pos1,
                                r2,
                                pos2: pos2 + 1,
                            });
                            best_plan = self.last_plan;
                        }
                    }
                }};
            }
            // `$keep` admits a whole group; the tail of the last group re-tests
            // the final index rather than branching, and the mask drops it.
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
            // `pos1` is fixed for the whole sweep, so the head case is decided once.
            if pos1 == 1 {
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
                    {
                        let pos2 = copy_at(&self.node_pos, c2);
                        let delta = self.run_swapstar(r1, pos1, r2, pos2);
                        if delta < best_delta {
                            best_delta = delta;
                            best_move = Some(CandidateMove::SwapStar { r1, pos1, r2, pos2 });
                            best_plan = self.last_plan;
                        }
                    }
                }};
            }
            sweep8!(cap_keep, cap_one, capacity_start, capacity_end, false);

            // Moves onto an empty route, from the second loop onward.
            if loop_id > 1 && (loop_id == 2 || r1_last_mod > last_tested) {
                if let Some(&r2) = self.empty_routes.first() {
                    let pos2 = 1;

                    // Splits r1 in two. At pos1 == 1 it would only move the whole
                    // route to another index, so it is skipped.
                    if pos1 > 1 {
                        let delta = self.run_2optstar(r1, pos1, r2, pos2);
                        if delta < best_delta {
                            best_delta = delta;
                            best_move = Some(CandidateMove::TwoOptStar { r1, pos1, r2, pos2 });
                            best_plan = self.last_plan;
                        }
                    }

                    let delta = self.run_inter_route(r1, pos1, r2, pos2);
                    if delta < best_delta {
                        best_delta = delta;
                        best_move = Some(CandidateMove::InterRoute { r1, pos1, r2, pos2 });
                        best_plan = self.last_plan;
                    }
                }
            }
        }

        if copy_at(&self.when_last_modified, r1) > last_tested {
            let delta = self.run_intra_route_relocate(r1, pos1);
            if delta < best_delta {
                best_delta = delta;
                best_move = Some(CandidateMove::IntraRelocate { r1, pos1 });
                best_plan = self.last_plan;
            }

            let delta = self.run_intra_route_oropt2(r1, pos1);
            if delta < best_delta {
                best_delta = delta;
                best_move = Some(CandidateMove::IntraOrOpt2 { r1, pos1 });
                best_plan = self.last_plan;
            }

            let delta = self.run_intra_route_swap(r1, pos1);
            if delta < best_delta {
                best_delta = delta;
                best_move = Some(CandidateMove::IntraSwap { r1, pos1 });
                best_plan = self.last_plan;
            }

            let delta = self.run_2opt(r1, pos1);
            if delta < best_delta {
                best_delta = delta;
                best_move = Some(CandidateMove::Intra2Opt { r1, pos1 });
                best_plan = self.last_plan;
            }
        }

        if let Some(mv) = best_move {
            let applied_delta = self.apply_planned_move(mv, best_plan);
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
                "Applied move delta differs from evaluated best delta for {mv:?} with {best_plan:?}",
            );
            delta
        } else {
            NO_MOVE
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
                let c1 = copy_at(&self.loop_order_nodes, idx);
                let mut c1_repeat = true;
                while c1_repeat {
                    c1_repeat = false;
                    let last_tested = copy_at(&self.when_last_tested, c1);
                    debug_assert!(c1 < self.when_last_tested.len());
                    unsafe {
                        *self.when_last_tested.get_unchecked_mut(c1) = self.nb_moves;
                    }
                    let delta =
                        self.evaluate_and_apply_best_move_for_customer(c1, last_tested, loop_id);
                    if delta != NO_MOVE {
                        improved = true;
                        c1_repeat = delta < 0;
                    }
                }
            }
        }
    }
}
