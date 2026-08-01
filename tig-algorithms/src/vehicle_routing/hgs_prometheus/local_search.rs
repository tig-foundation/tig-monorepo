use super::params::Params;
use super::problem::Problem;
use super::sequence::Sequence;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use std::cmp::min;
use std::sync::Arc;

const NO_MOVE: i64 = i64::MAX;

/// Index a solver-owned table after its structural invariant has established
/// the bound. This keeps release builds from paying repeated bounds checks in
/// the innermost neighbourhood loops.
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
            // A singleton depends only on the immutable problem and client id,
            // so it remains valid while this node moves between routes.
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
    /// Build the node container; metrics are computed later by `update_route`.
    #[inline]
    fn new(data: &Problem, ids: &[usize]) -> Self {
        Self {
            nodes: ids.iter().copied().map(|id| Node::new(data, id)).collect(),
            ..Default::default()
        }
    }

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
    pub when_last_modified: Vec<usize>, // per route
    pub when_last_tested: Vec<usize>,   // per customer id
    pub nb_moves: usize,                // monotone counter of applied moves
    pub move_credit: i64,               // deterioration budget; set to -1 from loop #2 onward
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
        let keep = min(params.granularity as usize, cap);
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

            // Both neighbourhood rankings use the same O(n²) node scan.
            // Their exact tie keys remain unchanged.
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
            // Demands are bounded small integers.  Walk demand buckets in
            // exactly the old (|d_j-d_i|, client-id) order instead of doing a
            // second O(n) collection and selection for every client.
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
            let m = capacity_prox.len().min(params.granularity2 as usize);
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

        // Normalize routes to exactly `fleet` entries.
        if routes.len() <= fleet {
            // If needed, pad missing routes with empty depot-only routes.
            routes.resize(fleet, vec![0, 0]);
        } else {
            // Keep the first fleet routes, merging every excess route into
            // the last one in place.  Taking ownership avoids cloning every
            // route before building the LS node containers.
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

        // Reuse route/node buffers across GA children.  Every cached sequence
        // is refreshed below, so only the customer ids must be rebuilt here.
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
        let mut out: Vec<Vec<usize>> = self
            .routes
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

    /// Raw metrics of the currently loaded routes. These are the same values
    /// that `Individual::evaluate_routes` would reconstruct from the export.
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
        debug_assert!(rid < self.routes.len());
        let r = unsafe { self.routes.get_unchecked_mut(rid) };
        let nodes = &mut r.nodes;
        let len = nodes.len();
        debug_assert!(len >= 2);
        let ptr = nodes.as_mut_ptr();

        // Materialize every route arc once. Prefixes, suffixes, and the
        // short forward sequences below all reuse these exact same travels.
        for pos in 0..len {
            let id = unsafe { (*ptr.add(pos)).id };
            let dist_to_succ = if pos + 1 < len {
                data.dm(id, unsafe { (*ptr.add(pos + 1)).id })
            } else {
                0
            };
            unsafe {
                (*ptr.add(pos)).dist_to_succ = dist_to_succ;
            }
        }

        // forward pass: seq0_i
        let mut acc_fwd = unsafe { (*ptr).seq1 };
        unsafe {
            (*ptr).seq0_i = acc_fwd;
        }
        for pos in 1..len {
            let singleton = unsafe { (*ptr.add(pos)).seq1 };
            let travel = unsafe { (*ptr.add(pos - 1)).dist_to_succ };
            acc_fwd = Sequence::join2_with_travel(&acc_fwd, &singleton, travel);
            unsafe {
                (*ptr.add(pos)).seq0_i = acc_fwd;
            }
        }

        // backward pass: seqi_n
        let mut acc_bwd = unsafe { (*ptr.add(len - 1)).seq1 };
        unsafe {
            (*ptr.add(len - 1)).seqi_n = acc_bwd;
        }
        for pos in (0..len - 1).rev() {
            let singleton = unsafe { (*ptr.add(pos)).seq1 };
            let travel = unsafe { (*ptr.add(pos)).dist_to_succ };
            acc_bwd = Sequence::join2_with_travel(&singleton, &acc_bwd, travel);
            unsafe {
                (*ptr.add(pos)).seqi_n = acc_bwd;
            }
        }

        // Per-node short sequences used by the granular move evaluators.
        for pos in 0..len {
            let id = unsafe { (*ptr.add(pos)).id };
            let singleton = unsafe { (*ptr.add(pos)).seq1 };
            if pos + 1 < len {
                let next = unsafe { (*ptr.add(pos + 1)).seq1 };
                let forward_travel = unsafe { (*ptr.add(pos)).dist_to_succ };
                let seq12 = Sequence::join2_with_travel(&singleton, &next, forward_travel);
                let seq21 = Sequence::join2_with_travel(
                    &next,
                    &singleton,
                    data.dm(unsafe { (*ptr.add(pos + 1)).id }, id),
                );
                unsafe {
                    (*ptr.add(pos)).seq12 = seq12;
                    (*ptr.add(pos)).seq21 = seq21;
                }
                if pos + 2 < len {
                    let next2 = unsafe { (*ptr.add(pos + 2)).seq1 };
                    let travel = unsafe { (*ptr.add(pos + 1)).dist_to_succ };
                    unsafe {
                        (*ptr.add(pos)).seq123 =
                            Sequence::join2_with_travel(&seq12, &next2, travel);
                    }
                }
            }
        }

        let end = unsafe { (*ptr.add(len - 1)).seq0_i };
        r.load = end.load;
        r.tw = end.tw;
        r.distance = end.distance;
        r.cost = end.eval(data, &self.params);

        // Update route node mappings
        for (pos, node) in nodes.iter().enumerate() {
            debug_assert!(node.id < self.node_route.len() && node.id < self.node_pos.len());
            unsafe {
                *self.node_route.get_unchecked_mut(node.id) = rid;
                *self.node_pos.get_unchecked_mut(node.id) = pos;
            }
        }

        // Refresh vector of empty routes
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
        } // no alternative insertion for single-client routes

        debug_assert!(pos1 > 0); // U is a client
        debug_assert!(self.routes[r1].nodes[pos1].id != 0); // U is a client

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

        // Insert U before t in [1 .. pos1-1]
        if pos1 > 1 {
            let mut right_excl_u = route.node(pos1 + 1).seqi_n;
            for t in (1..pos1).rev() {
                let join_travel = if t + 1 == pos1 {
                    add_ab
                } else {
                    route.node(t).dist_to_succ
                };
                right_excl_u =
                    Sequence::join_tw_with_travel(&route.node(t).seq1, &right_excl_u, join_travel);
                let c_id = route.node(t - 1).id;
                let d_id = route.node(t).id;
                let d_cu = copy_at(distance_to_u, c_id);
                let d_ud = copy_at(distance_from_u, d_id);
                let delta_distance = fixed_delta + d_cu + d_ud - route.node(t - 1).dist_to_succ;
                if old_distance + (delta_distance as i64) > max_distance_acceptable {
                    continue;
                }
                let left = route.node(t - 1).seq0_i;
                let tw = Sequence::tw3_with_travel(&left, &u_seq1, &right_excl_u, d_cu, d_ud);
                let new_cost = old_distance + (delta_distance as i64) + cap_pen + (tw as i64) * ptw;
                if new_cost <= max_acceptable_cost && new_cost < best_cost {
                    best_cost = new_cost;
                    best_pos = Some(t);
                }
            }
        }

        // Insert U before t in [pos1+2 .. len-1]
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
                if old_distance + (delta_distance as i64) <= max_distance_acceptable {
                    let tw = Sequence::tw3_with_travel(
                        &left_excl_u,
                        &u_seq1,
                        &route.node(t).seqi_n,
                        d_cu,
                        d_ud,
                    );
                    let new_cost =
                        old_distance + (delta_distance as i64) + cap_pen + (tw as i64) * ptw;
                    if new_cost <= max_acceptable_cost && new_cost < best_cost {
                        best_cost = new_cost;
                        best_pos = Some(t);
                    }
                }
                if t + 1 < len {
                    left_excl_u = Sequence::join_tw_with_travel(
                        &left_excl_u,
                        &route.node(t).seq1,
                        route.node(t - 1).dist_to_succ,
                    );
                }
            }
        }

        if let Some(mypos) = best_pos {
            let selected_delta = best_cost - old_cost;
            self.last_plan = MovePlan::IntraRelocate { target: mypos };
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
        } // pair does not exist

        debug_assert!(pos1 > 0); // U is a client
        debug_assert!(self.routes[r1].nodes[pos1].id != 0); // U is a client
        debug_assert!(self.routes[r1].nodes[pos1 + 1].id != 0); // successor is a client

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

        // Insert (U,X) or (X,U) before t in [1 .. pos1-1]
        if pos1 > 1 {
            let mut right_excl_pair = route.node(pos1 + 2).seqi_n;
            for t in (1..pos1).rev() {
                let join_travel = if t + 1 == pos1 {
                    add_ab
                } else {
                    route.node(t).dist_to_succ
                };
                right_excl_pair = Sequence::join_tw_with_travel(
                    &route.node(t).seq1,
                    &right_excl_pair,
                    join_travel,
                );
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
                if !can_pass_fwd && !can_pass_rev {
                    continue;
                }
                let left = route.node(t - 1).seq0_i;

                if can_pass_fwd {
                    let tw =
                        Sequence::tw3_with_travel(&left, &pair_fwd, &right_excl_pair, d_cu, d_xd);
                    let new_cost_fwd =
                        old_distance + (delta_fwd as i64) + cap_pen + (tw as i64) * ptw;
                    if new_cost_fwd <= max_acceptable_cost && new_cost_fwd < best_cost {
                        best_cost = new_cost_fwd;
                        best_pos = Some(t);
                        best_reversed = false;
                    }
                }

                if can_pass_rev {
                    let tw =
                        Sequence::tw3_with_travel(&left, &pair_rev, &right_excl_pair, d_cx, d_ud);
                    let new_cost_rev =
                        old_distance + (delta_rev as i64) + cap_pen + (tw as i64) * ptw;
                    if new_cost_rev <= max_acceptable_cost && new_cost_rev < best_cost {
                        best_cost = new_cost_rev;
                        best_pos = Some(t);
                        best_reversed = true;
                    }
                }
            }
        }

        // Insert (U,X) or (X,U) before t in [pos1+3 .. len-1]
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
                if can_pass_fwd {
                    let tw =
                        Sequence::tw3_with_travel(&left_excl_pair, &pair_fwd, &right, d_cu, d_xd);
                    let new_cost_fwd =
                        old_distance + (delta_fwd as i64) + cap_pen + (tw as i64) * ptw;
                    if new_cost_fwd <= max_acceptable_cost && new_cost_fwd < best_cost {
                        best_cost = new_cost_fwd;
                        best_pos = Some(t);
                        best_reversed = false;
                    }
                }

                if can_pass_rev {
                    let tw =
                        Sequence::tw3_with_travel(&left_excl_pair, &pair_rev, &right, d_cx, d_ud);
                    let new_cost_rev =
                        old_distance + (delta_rev as i64) + cap_pen + (tw as i64) * ptw;
                    if new_cost_rev <= max_acceptable_cost && new_cost_rev < best_cost {
                        best_cost = new_cost_rev;
                        best_pos = Some(t);
                        best_reversed = true;
                    }
                }

                if t + 1 < len {
                    left_excl_pair = Sequence::join_tw_with_travel(
                        &left_excl_pair,
                        &route.node(t).seq1,
                        route.node(t - 1).dist_to_succ,
                    );
                }
            }
        }

        if let Some(mypos) = best_pos {
            let selected_delta = best_cost - old_cost;
            self.last_plan = MovePlan::IntraOrOpt2 {
                target: mypos,
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
        } // need at least 3 clients for a non-adjacent swap
        let data = self.data.as_ref();
        let has_right = pos1 + 2 < len - 1;
        let has_left = pos1 >= 3;
        if !has_left && !has_right {
            return NO_MOVE;
        }

        debug_assert!(pos1 > 0); // U is a client
        debug_assert!(self.routes[r1].nodes[pos1].id != 0); // U is a client

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

        // Distance-only prefilter: evaluate only sides that contain a potentially improving swap.
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
                if old_distance + (delta_distance as i64) <= max_distance_acceptable {
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
                    if new_cost <= max_acceptable_cost && new_cost < best_cost {
                        best_cost = new_cost;
                        best_pos = Some(pos2);
                    }
                }
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
                if old_distance + (delta_distance as i64) <= max_distance_acceptable {
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
                    if new_cost <= max_acceptable_cost && new_cost < best_cost {
                        best_cost = new_cost;
                        best_pos = Some(pos2);
                    }
                }
                if pos2 > 1 {
                    acc_mid = Sequence::join_tw_with_travel(
                        &route.node(pos2).seq1,
                        &acc_mid,
                        route.node(pos2).dist_to_succ,
                    );
                }
            }
        }

        if let Some(mypos) = best_pos {
            let selected_delta = best_cost - old_cost;
            self.last_plan = MovePlan::IntraSwap { target: mypos };
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

        // Prefilter: only distance + load excess penalties (no TW penalties).
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
        } // need at least [0, U, V, 0]

        debug_assert!(pos1 > 0); // U is a client
        debug_assert!(self.routes[r1].nodes[pos1].id != 0); // U is a client

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

        let mut mid_rev = route.node(pos1).seq21;
        let mut mid_rev_distance = mid_rev.distance;
        for pos2 in (pos1 + 1)..(len - 1) {
            let v_id = route.node(pos2).id;
            let b_id = route.node(pos2 + 1).id;
            let d_av = copy_at(distance_from_a, v_id);
            let d_ub = copy_at(distance_from_u, b_id);
            // Materialize the exact reversed-segment distance for accepted
            // candidates, including asymmetric internal arcs.
            let left = route.node(pos1 - 1).seq0_i;
            let right = route.node(pos2 + 1).seqi_n;
            let new_distance = left.distance + mid_rev_distance + right.distance + d_av + d_ub;
            // Preserve HGS's original four-boundary-arc eligibility filter.
            // Replacing it with the exact distance admits extra moves and
            // changes the downstream deterministic search trajectory.
            let legacy_delta = d_av + d_ub - removed_au - route.node(pos2).dist_to_succ;
            if old_distance + legacy_delta as i64 > max_distance_acceptable {
                if pos2 + 1 < len - 1 {
                    let next = route.node(pos2 + 1);
                    let reversed_travel = data.dm(next.id, v_id);
                    mid_rev_distance += reversed_travel;
                    mid_rev = Sequence::join_tw_with_travel(&next.seq1, &mid_rev, reversed_travel);
                }
                continue;
            }
            let tw = Sequence::tw3_with_travel(&left, &mid_rev, &right, d_av, d_ub);
            let new_cost = (new_distance as i64) + cap_pen + (tw as i64) * ptw;
            if new_cost <= max_acceptable_cost && new_cost < best_cost {
                best_cost = new_cost;
                best_pos = Some(pos2);
            }
            if pos2 + 1 < len - 1 {
                let next = route.node(pos2 + 1);
                let reversed_travel = data.dm(next.id, v_id);
                mid_rev_distance += reversed_travel;
                mid_rev = Sequence::join_tw_with_travel(&next.seq1, &mid_rev, reversed_travel);
            }
        }

        if let Some(mypos) = best_pos {
            let selected_delta = best_cost - old_cost;
            self.last_plan = MovePlan::Intra2Opt { end: mypos };
            selected_delta
        } else {
            NO_MOVE
        }
    }

    fn run_inter_route(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize) -> i64 {
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
                if lower_bound <= max_acceptable_cost {
                    let candidate = lower_bound + ($tw as i64) * ptw;
                    if candidate <= max_acceptable_cost && candidate < best_cost {
                        best_cost = candidate;
                        best_send1 = $send1;
                        best_send2 = $send2;
                    }
                }
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
        if lb10 <= max_acceptable_cost {
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
            if lb11 <= max_acceptable_cost {
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
            if lb20 <= max_acceptable_cost || lb30 <= max_acceptable_cost {
                let route1_tw =
                    Sequence::tw2_with_travel(&u_pred.seq0_i, &x_next.seqi_n, d_upred_xnext);
                if lb20 <= max_acceptable_cost {
                    let tw20 = route1_tw
                        + Sequence::tw3_with_travel(
                            &v_pred.seq0_i,
                            &u.seq12,
                            &v.seqi_n,
                            d_vpred_u,
                            d_x_v,
                        );
                    consider!(2, 0, lb20, tw20);
                }
                if lb30 <= max_acceptable_cost {
                    let tw30 = route1_tw
                        + Sequence::tw3_with_travel(
                            &v_pred.seq0_i,
                            &u.seq21,
                            &v.seqi_n,
                            d_vpred_x,
                            d_u_v,
                        );
                    consider!(3, 0, lb30, tw30);
                }
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
                if lb21 <= max_acceptable_cost || lb31 <= max_acceptable_cost {
                    let route1_tw = Sequence::tw3_with_travel(
                        &u_pred.seq0_i,
                        &v.seq1,
                        &x_next.seqi_n,
                        d_upred_v,
                        d_v_xnext,
                    );
                    if lb21 <= max_acceptable_cost {
                        let tw21 = route1_tw
                            + Sequence::tw3_with_travel(
                                &v_pred.seq0_i,
                                &u.seq12,
                                &y.seqi_n,
                                d_vpred_u,
                                d_x_y,
                            );
                        consider!(2, 1, lb21, tw21);
                    }
                    if lb31 <= max_acceptable_cost {
                        let tw31 = route1_tw
                            + Sequence::tw3_with_travel(
                                &v_pred.seq0_i,
                                &u.seq21,
                                &y.seqi_n,
                                d_vpred_x,
                                d_u_y,
                            );
                        consider!(3, 1, lb31, tw31);
                    }
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

                    let left_fwd = if can22 || can32 {
                        Some(Sequence::tw3_with_travel(
                            &u_pred.seq0_i,
                            &v.seq12,
                            &x_next.seqi_n,
                            d_upred_v,
                            d_y_xnext,
                        ))
                    } else {
                        None
                    };
                    let left_rev = if can23 || can33 {
                        Some(Sequence::tw3_with_travel(
                            &u_pred.seq0_i,
                            &v.seq21,
                            &x_next.seqi_n,
                            d_upred_y,
                            d_v_xnext,
                        ))
                    } else {
                        None
                    };
                    let right_fwd = if can22 || can23 {
                        Some(Sequence::tw3_with_travel(
                            &v_pred.seq0_i,
                            &u.seq12,
                            &y_next.seqi_n,
                            d_vpred_u,
                            d_x_ynext,
                        ))
                    } else {
                        None
                    };
                    let right_rev = if can32 || can33 {
                        Some(Sequence::tw3_with_travel(
                            &v_pred.seq0_i,
                            &u.seq21,
                            &y_next.seqi_n,
                            d_vpred_x,
                            d_u_ynext,
                        ))
                    } else {
                        None
                    };

                    if can22 {
                        consider!(2, 2, lb22, left_fwd.unwrap() + right_fwd.unwrap());
                    }
                    if can32 {
                        consider!(3, 2, lb32, left_fwd.unwrap() + right_rev.unwrap());
                    }
                    if can23 {
                        consider!(2, 3, lb23, left_rev.unwrap() + right_fwd.unwrap());
                    }
                    if can33 {
                        consider!(3, 3, lb33, left_rev.unwrap() + right_rev.unwrap());
                    }
                }
            }

            if x_next.id != 0 && self.params.allow_swap3 {
                let x2_next = ru.node(pos1 + 3);
                let send1_3_load = u.seq1.load + x.seq1.load + x_next.seq1.load;
                let rem1_3 =
                    u_pred.dist_to_succ + u.dist_to_succ + x.dist_to_succ + x_next.dist_to_succ;
                let d_upred_x2next = copy_at(distance_from_u_pred, x2_next.id);
                let d_u_xnext = u.dist_to_succ;
                let d_x_xnext = x.dist_to_succ;
                let d_xnext_v = copy_at(distance_from_x_next, v.id);
                let cap_40 =
                    cap_pen!(route1_load - send1_3_load) + cap_pen!(route2_load + send1_3_load);
                let lb40 = (route_total_dist - rem1_3 - rem2_0
                    + d_upred_x2next
                    + d_vpred_u
                    + d_u_xnext
                    + d_x_xnext
                    + d_xnext_v) as i64
                    + cap_40;
                if lb40 <= max_acceptable_cost {
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
                        + d_u_xnext
                        + d_x_xnext
                        + d_xnext_y) as i64
                        + cap_41;
                    if lb41 <= max_acceptable_cost {
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
                        let common_right = d_vpred_u + d_u_xnext + d_x_xnext + d_xnext_ynext;
                        let lb42 =
                            (dist_base + d_upred_v + v.dist_to_succ + d_y_x2next + common_right)
                                as i64
                                + cap_42_43;
                        let lb43 = (dist_base + d_upred_y + d_y_v + d_v_x2next + common_right)
                            as i64
                            + cap_42_43;
                        if lb42 <= max_acceptable_cost || lb43 <= max_acceptable_cost {
                            let right_tw = Sequence::tw3_with_travel(
                                &v_pred.seq0_i,
                                &u.seq123,
                                &y_next.seqi_n,
                                d_vpred_u,
                                d_xnext_ynext,
                            );
                            if lb42 <= max_acceptable_cost {
                                let tw42 = Sequence::tw3_with_travel(
                                    &u_pred.seq0_i,
                                    &v.seq12,
                                    &x2_next.seqi_n,
                                    d_upred_v,
                                    d_y_x2next,
                                ) + right_tw;
                                consider!(4, 2, lb42, tw42);
                            }
                            if lb43 <= max_acceptable_cost {
                                let tw43 = Sequence::tw3_with_travel(
                                    &u_pred.seq0_i,
                                    &v.seq21,
                                    &x2_next.seqi_n,
                                    d_upred_y,
                                    d_v_x2next,
                                ) + right_tw;
                                consider!(4, 3, lb43, tw43);
                            }
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
                                + d_u_xnext
                                + d_x_xnext
                                + d_xnext_y2next) as i64
                                + cap_44;
                            if lb44 <= max_acceptable_cost {
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

    #[allow(dead_code)]
    fn run_inter_route_legacy(
        &mut self,
        r1: usize,
        pos1: usize,
        r2: usize,
        pos2: usize,
    ) -> Option<i64> {
        let data = self.data.as_ref();
        let ru = route_at(&self.routes, r1);
        let rv = route_at(&self.routes, r2);
        let u = ru.node(pos1); // cannot be a depot
        let v = rv.node(pos2); // could be a depot
        let u_pred = ru.node(pos1 - 1); // could be a depot
        let v_pred = rv.node(pos2 - 1); // could be a depot
        let x = ru.node(pos1 + 1); // could be a depot
        debug_assert!(
            u.id != 0,
            "Should always apply inter-route with a client as first node"
        );
        debug_assert!(r1 != r2, "Should not test inter-route move on same route");

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
        let mut can = [[false; 5]; 5];
        let mut lower_bounds = [[i64::MAX; 5]; 5];
        // Connecting arcs for route 1 followed by route 2. A zero second arc
        // denotes a two-sequence route for the variants that remove a block.
        let mut travels = [[[0i32; 4]; 5]; 5];
        let mut update_lb = |cand: i64| {
            if cand < best_lb {
                best_lb = cand;
            }
        };

        let send1_1_load = u.seq1.load;
        let rem1_1 = u_pred.dist_to_succ + u.dist_to_succ;
        let rem2_0 = v_pred.dist_to_succ; // insertion before V breaks (Vpred,V)

        let d_upred_x = data.dm(u_pred.id, x.id);
        let d_vpred_u = data.dm(v_pred.id, u.id);
        let d_u_v = data.dm(u.id, v.id);
        let dist_base_10 = route_total_dist - rem1_1 - rem2_0;
        let cap_10 = cap_pen(route1_load - send1_1_load) + cap_pen(route2_load + send1_1_load);
        let result10_lb = to_cost(dist_base_10 + d_upred_x + d_vpred_u + d_u_v, cap_10);
        lower_bounds[1][0] = result10_lb;
        travels[1][0] = [d_upred_x, 0, d_vpred_u, d_u_v];
        can[1][0] = result10_lb <= max_acceptable_cost;
        update_lb(result10_lb);

        // Send {U}, receive {V}
        if v.id != 0 {
            let y = rv.node(pos2 + 1);
            let send2_1_load = v.seq1.load;
            let rem2_1 = v_pred.dist_to_succ + v.dist_to_succ;
            let d_upred_v = data.dm(u_pred.id, v.id);
            let d_v_x = data.dm(v.id, x.id);
            let d_u_y = data.dm(u.id, y.id);
            let dist_base_11 = route_total_dist - rem1_1 - rem2_1;
            let cap_11 = cap_pen(route1_load - send1_1_load + send2_1_load)
                + cap_pen(route2_load - send2_1_load + send1_1_load);
            let result11_lb = to_cost(dist_base_11 + d_upred_v + d_v_x + d_vpred_u + d_u_y, cap_11);
            lower_bounds[1][1] = result11_lb;
            travels[1][1] = [d_upred_v, d_v_x, d_vpred_u, d_u_y];
            can[1][1] = result11_lb <= max_acceptable_cost;
            update_lb(result11_lb);
        }

        if x.id != 0 {
            let x_next = ru.node(pos1 + 2);
            let send1_2_load = u.seq1.load + x.seq1.load;
            let rem1_2 = u_pred.dist_to_succ + u.dist_to_succ + x.dist_to_succ;
            let d_upred_xnext = data.dm(u_pred.id, x_next.id);
            let d_vpred_x = data.dm(v_pred.id, x.id);
            let d_x_u = data.dm(x.id, u.id);
            let d_u_x = u.dist_to_succ;
            let d_x_v = data.dm(x.id, v.id);
            let dist_base_20_30 = route_total_dist - rem1_2 - rem2_0;
            let cap_20_30 =
                cap_pen(route1_load - send1_2_load) + cap_pen(route2_load + send1_2_load);

            // Send {U,X} or {X,U}, receive {}
            let result20_lb = to_cost(
                dist_base_20_30 + d_upred_xnext + d_vpred_u + d_u_x + d_x_v,
                cap_20_30,
            );
            let result30_lb = to_cost(
                dist_base_20_30 + d_upred_xnext + d_vpred_x + d_x_u + d_u_v,
                cap_20_30,
            );
            lower_bounds[2][0] = result20_lb;
            lower_bounds[3][0] = result30_lb;
            travels[2][0] = [d_upred_xnext, 0, d_vpred_u, d_x_v];
            travels[3][0] = [d_upred_xnext, 0, d_vpred_x, d_u_v];
            can[2][0] = result20_lb <= max_acceptable_cost;
            can[3][0] = result30_lb <= max_acceptable_cost;
            update_lb(result20_lb);
            update_lb(result30_lb);

            if v.id != 0 {
                let y = rv.node(pos2 + 1);
                let send2_1_load = v.seq1.load;
                let rem2_1 = v_pred.dist_to_succ + v.dist_to_succ;
                let d_upred_v = data.dm(u_pred.id, v.id);
                let d_v_xnext = data.dm(v.id, x_next.id);
                let d_x_y = data.dm(x.id, y.id);
                let d_u_y = data.dm(u.id, y.id);
                let dist_base_21_31 = route_total_dist - rem1_2 - rem2_1;
                let cap_21_31 = cap_pen(route1_load - send1_2_load + send2_1_load)
                    + cap_pen(route2_load - send2_1_load + send1_2_load);

                // Send {U,X} or {X,U}, receive {V}
                let result21_lb = to_cost(
                    dist_base_21_31 + d_upred_v + d_v_xnext + d_vpred_u + d_u_x + d_x_y,
                    cap_21_31,
                );
                let result31_lb = to_cost(
                    dist_base_21_31 + d_upred_v + d_v_xnext + d_vpred_x + d_x_u + d_u_y,
                    cap_21_31,
                );
                lower_bounds[2][1] = result21_lb;
                lower_bounds[3][1] = result31_lb;
                travels[2][1] = [d_upred_v, d_v_xnext, d_vpred_u, d_x_y];
                travels[3][1] = [d_upred_v, d_v_xnext, d_vpred_x, d_u_y];
                can[2][1] = result21_lb <= max_acceptable_cost;
                can[3][1] = result31_lb <= max_acceptable_cost;
                update_lb(result21_lb);
                update_lb(result31_lb);

                if y.id != 0 {
                    let y_next = rv.node(pos2 + 2);
                    let send2_2_load = v.seq1.load + y.seq1.load;
                    let rem2_2 = v_pred.dist_to_succ + v.dist_to_succ + y.dist_to_succ;
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
                            + d_upred_v
                            + v.dist_to_succ
                            + d_y_xnext
                            + d_vpred_u
                            + d_u_x
                            + d_x_ynext,
                        cap_22_33,
                    );
                    let result23_lb = to_cost(
                        dist_base_22_33
                            + d_upred_y
                            + d_y_v
                            + d_v_xnext
                            + d_vpred_u
                            + d_u_x
                            + d_x_ynext,
                        cap_22_33,
                    );
                    let result32_lb = to_cost(
                        dist_base_22_33
                            + d_upred_v
                            + v.dist_to_succ
                            + d_y_xnext
                            + d_vpred_x
                            + d_x_u
                            + d_u_ynext,
                        cap_22_33,
                    );
                    let result33_lb = to_cost(
                        dist_base_22_33
                            + d_upred_y
                            + d_y_v
                            + d_v_xnext
                            + d_vpred_x
                            + d_x_u
                            + d_u_ynext,
                        cap_22_33,
                    );
                    lower_bounds[2][2] = result22_lb;
                    lower_bounds[2][3] = result23_lb;
                    lower_bounds[3][2] = result32_lb;
                    lower_bounds[3][3] = result33_lb;
                    travels[2][2] = [d_upred_v, d_y_xnext, d_vpred_u, d_x_ynext];
                    travels[2][3] = [d_upred_y, d_v_xnext, d_vpred_u, d_x_ynext];
                    travels[3][2] = [d_upred_v, d_y_xnext, d_vpred_x, d_u_ynext];
                    travels[3][3] = [d_upred_y, d_v_xnext, d_vpred_x, d_u_ynext];
                    can[2][2] = result22_lb <= max_acceptable_cost;
                    can[2][3] = result23_lb <= max_acceptable_cost;
                    can[3][2] = result32_lb <= max_acceptable_cost;
                    can[3][3] = result33_lb <= max_acceptable_cost;
                    update_lb(result22_lb);
                    update_lb(result23_lb);
                    update_lb(result32_lb);
                    update_lb(result33_lb);
                }
            }

            // Send {U,X,Xnext}
            if x_next.id != 0 && self.params.allow_swap3 {
                let x2_next = ru.node(pos1 + 3);
                let send1_3_load = u.seq1.load + x.seq1.load + x_next.seq1.load;
                let rem1_3 =
                    u_pred.dist_to_succ + u.dist_to_succ + x.dist_to_succ + x_next.dist_to_succ;
                let d_upred_x2next = data.dm(u_pred.id, x2_next.id);
                let d_u_xnext = u.dist_to_succ;
                let d_x_xnext = x.dist_to_succ;
                let d_xnext_v = data.dm(x_next.id, v.id);
                let dist_base_40 = route_total_dist - rem1_3 - rem2_0;
                let cap_40 =
                    cap_pen(route1_load - send1_3_load) + cap_pen(route2_load + send1_3_load);

                let result40_lb = to_cost(
                    dist_base_40 + d_upred_x2next + d_vpred_u + d_u_xnext + d_x_xnext + d_xnext_v,
                    cap_40,
                );
                lower_bounds[4][0] = result40_lb;
                travels[4][0] = [d_upred_x2next, 0, d_vpred_u, d_xnext_v];
                can[4][0] = result40_lb <= max_acceptable_cost;
                update_lb(result40_lb);

                if v.id != 0 {
                    let y = rv.node(pos2 + 1);
                    let send2_1_load = v.seq1.load;
                    let rem2_1 = v_pred.dist_to_succ + v.dist_to_succ;
                    let d_upred_v = data.dm(u_pred.id, v.id);
                    let d_v_x2next = data.dm(v.id, x2_next.id);
                    let d_xnext_y = data.dm(x_next.id, y.id);
                    let dist_base_41 = route_total_dist - rem1_3 - rem2_1;
                    let cap_41 = cap_pen(route1_load - send1_3_load + send2_1_load)
                        + cap_pen(route2_load - send2_1_load + send1_3_load);

                    let result41_lb = to_cost(
                        dist_base_41
                            + d_upred_v
                            + d_v_x2next
                            + d_vpred_u
                            + d_u_xnext
                            + d_x_xnext
                            + d_xnext_y,
                        cap_41,
                    );
                    lower_bounds[4][1] = result41_lb;
                    travels[4][1] = [d_upred_v, d_v_x2next, d_vpred_u, d_xnext_y];
                    can[4][1] = result41_lb <= max_acceptable_cost;
                    update_lb(result41_lb);

                    if y.id != 0 {
                        let y_next = rv.node(pos2 + 2);
                        let send2_2_load = v.seq1.load + y.seq1.load;
                        let rem2_2 = v_pred.dist_to_succ + v.dist_to_succ + y.dist_to_succ;
                        let d_upred_y = data.dm(u_pred.id, y.id);
                        let d_y_v = data.dm(y.id, v.id);
                        let d_y_x2next = data.dm(y.id, x2_next.id);
                        let d_xnext_ynext = data.dm(x_next.id, y_next.id);
                        let dist_base_42_43 = route_total_dist - rem1_3 - rem2_2;
                        let cap_42_43 = cap_pen(route1_load - send1_3_load + send2_2_load)
                            + cap_pen(route2_load - send2_2_load + send1_3_load);

                        let result42_lb = to_cost(
                            dist_base_42_43
                                + d_upred_v
                                + v.dist_to_succ
                                + d_y_x2next
                                + d_vpred_u
                                + d_u_xnext
                                + d_x_xnext
                                + d_xnext_ynext,
                            cap_42_43,
                        );
                        let result43_lb = to_cost(
                            dist_base_42_43
                                + d_upred_y
                                + d_y_v
                                + d_v_x2next
                                + d_vpred_u
                                + d_u_xnext
                                + d_x_xnext
                                + d_xnext_ynext,
                            cap_42_43,
                        );
                        lower_bounds[4][2] = result42_lb;
                        lower_bounds[4][3] = result43_lb;
                        travels[4][2] = [d_upred_v, d_y_x2next, d_vpred_u, d_xnext_ynext];
                        travels[4][3] = [d_upred_y, d_v_x2next, d_vpred_u, d_xnext_ynext];
                        can[4][2] = result42_lb <= max_acceptable_cost;
                        can[4][3] = result43_lb <= max_acceptable_cost;
                        update_lb(result42_lb);
                        update_lb(result43_lb);

                        if y_next.id != 0 {
                            let y2_next = rv.node(pos2 + 3);
                            let send2_4_load = v.seq1.load + y.seq1.load + y_next.seq1.load;
                            let rem2_4 = v_pred.dist_to_succ
                                + v.dist_to_succ
                                + y.dist_to_succ
                                + y_next.dist_to_succ;
                            let d_ynext_x2next = data.dm(y_next.id, x2_next.id);
                            let d_xnext_y2next = data.dm(x_next.id, y2_next.id);
                            let dist_base_44 = route_total_dist - rem1_3 - rem2_4;
                            let cap_44 = cap_pen(route1_load - send1_3_load + send2_4_load)
                                + cap_pen(route2_load - send2_4_load + send1_3_load);

                            let result44_lb = to_cost(
                                dist_base_44
                                    + d_upred_v
                                    + v.dist_to_succ
                                    + y.dist_to_succ
                                    + d_ynext_x2next
                                    + d_vpred_u
                                    + d_u_xnext
                                    + d_x_xnext
                                    + d_xnext_y2next,
                                cap_44,
                            );
                            lower_bounds[4][4] = result44_lb;
                            travels[4][4] = [d_upred_v, d_ynext_x2next, d_vpred_u, d_xnext_y2next];
                            can[4][4] = result44_lb <= max_acceptable_cost;
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
        let ptw = self.params.penalty_tw as i64;

        // Send {U}, receive {}
        if can[1][0] {
            let tr = travels[1][0];
            let tw = Sequence::tw2_with_travel(&u_pred.seq0_i, &x.seqi_n, tr[0])
                + Sequence::tw3_with_travel(&v_pred.seq0_i, &u.seq1, &v.seqi_n, tr[2], tr[3]);
            let result10 = lower_bounds[1][0] + (tw as i64) * ptw;
            update_best(1, 0, result10);
        }

        // Send {U}, receive {V}
        if v.id != 0 && can[1][1] {
            let tr = travels[1][1];
            let tw = Sequence::tw3_with_travel(&u_pred.seq0_i, &v.seq1, &x.seqi_n, tr[0], tr[1])
                + Sequence::tw3_with_travel(
                    &v_pred.seq0_i,
                    &u.seq1,
                    &rv.node(pos2 + 1).seqi_n,
                    tr[2],
                    tr[3],
                );
            let result11 = lower_bounds[1][1] + (tw as i64) * ptw;
            update_best(1, 1, result11);
        }

        if x.id != 0 {
            // Send {U,X} or {X,U}, receive {}
            let x_next = ru.node(pos1 + 2);
            if can[2][0] || can[3][0] {
                let route1_tw =
                    Sequence::tw2_with_travel(&u_pred.seq0_i, &x_next.seqi_n, travels[2][0][0]);
                if can[2][0] {
                    let tr = travels[2][0];
                    let tw = route1_tw
                        + Sequence::tw3_with_travel(
                            &v_pred.seq0_i,
                            &u.seq12,
                            &v.seqi_n,
                            tr[2],
                            tr[3],
                        );
                    let result20 = lower_bounds[2][0] + (tw as i64) * ptw;
                    update_best(2, 0, result20);
                }
                if can[3][0] {
                    let tr = travels[3][0];
                    let tw = route1_tw
                        + Sequence::tw3_with_travel(
                            &v_pred.seq0_i,
                            &u.seq21,
                            &v.seqi_n,
                            tr[2],
                            tr[3],
                        );
                    let result30 = lower_bounds[3][0] + (tw as i64) * ptw;
                    update_best(3, 0, result30);
                }
            }

            if v.id != 0 {
                // Send {U,X} or {X,U}, receive {V}
                let y = rv.node(pos2 + 1);
                if can[2][1] || can[3][1] {
                    let route1_tw = Sequence::tw3_with_travel(
                        &u_pred.seq0_i,
                        &v.seq1,
                        &x_next.seqi_n,
                        travels[2][1][0],
                        travels[2][1][1],
                    );
                    if can[2][1] {
                        let tr = travels[2][1];
                        let tw = route1_tw
                            + Sequence::tw3_with_travel(
                                &v_pred.seq0_i,
                                &u.seq12,
                                &y.seqi_n,
                                tr[2],
                                tr[3],
                            );
                        let result21 = lower_bounds[2][1] + (tw as i64) * ptw;
                        update_best(2, 1, result21);
                    }
                    if can[3][1] {
                        let tr = travels[3][1];
                        let tw = route1_tw
                            + Sequence::tw3_with_travel(
                                &v_pred.seq0_i,
                                &u.seq21,
                                &y.seqi_n,
                                tr[2],
                                tr[3],
                            );
                        let result31 = lower_bounds[3][1] + (tw as i64) * ptw;
                        update_best(3, 1, result31);
                    }
                }

                if y.id != 0 {
                    // Send {U,X} or {X,U}, receive {V,Y} or {Y,V}
                    let y_next = rv.node(pos2 + 2);
                    let left_fwd = if can[2][2] || can[3][2] {
                        let tr = travels[2][2];
                        Some(Sequence::tw3_with_travel(
                            &u_pred.seq0_i,
                            &v.seq12,
                            &x_next.seqi_n,
                            tr[0],
                            tr[1],
                        ))
                    } else {
                        None
                    };
                    let left_rev = if can[2][3] || can[3][3] {
                        let tr = travels[2][3];
                        Some(Sequence::tw3_with_travel(
                            &u_pred.seq0_i,
                            &v.seq21,
                            &x_next.seqi_n,
                            tr[0],
                            tr[1],
                        ))
                    } else {
                        None
                    };
                    let right_fwd = if can[2][2] || can[2][3] {
                        let tr = travels[2][2];
                        Some(Sequence::tw3_with_travel(
                            &v_pred.seq0_i,
                            &u.seq12,
                            &y_next.seqi_n,
                            tr[2],
                            tr[3],
                        ))
                    } else {
                        None
                    };
                    let right_rev = if can[3][2] || can[3][3] {
                        let tr = travels[3][2];
                        Some(Sequence::tw3_with_travel(
                            &v_pred.seq0_i,
                            &u.seq21,
                            &y_next.seqi_n,
                            tr[2],
                            tr[3],
                        ))
                    } else {
                        None
                    };

                    if can[2][2] {
                        let tw = left_fwd.unwrap() + right_fwd.unwrap();
                        update_best(2, 2, lower_bounds[2][2] + (tw as i64) * ptw);
                    }
                    if can[3][2] {
                        let tw = left_fwd.unwrap() + right_rev.unwrap();
                        update_best(3, 2, lower_bounds[3][2] + (tw as i64) * ptw);
                    }
                    if can[2][3] {
                        let tw = left_rev.unwrap() + right_fwd.unwrap();
                        update_best(2, 3, lower_bounds[2][3] + (tw as i64) * ptw);
                    }
                    if can[3][3] {
                        let tw = left_rev.unwrap() + right_rev.unwrap();
                        update_best(3, 3, lower_bounds[3][3] + (tw as i64) * ptw);
                    }
                }
            }

            // Send {U,X,Xnext}
            if x_next.id != 0 && self.params.allow_swap3 {
                let x2_next = ru.node(pos1 + 3);
                if can[4][0] {
                    let tr = travels[4][0];
                    let tw = Sequence::tw2_with_travel(&u_pred.seq0_i, &x2_next.seqi_n, tr[0])
                        + Sequence::tw3_with_travel(
                            &v_pred.seq0_i,
                            &u.seq123,
                            &v.seqi_n,
                            tr[2],
                            tr[3],
                        );
                    let result40 = lower_bounds[4][0] + (tw as i64) * ptw;
                    update_best(4, 0, result40);
                }

                if v.id != 0 {
                    let y = rv.node(pos2 + 1);
                    if can[4][1] {
                        let tr = travels[4][1];
                        let tw = Sequence::tw3_with_travel(
                            &u_pred.seq0_i,
                            &v.seq1,
                            &x2_next.seqi_n,
                            tr[0],
                            tr[1],
                        ) + Sequence::tw3_with_travel(
                            &v_pred.seq0_i,
                            &u.seq123,
                            &y.seqi_n,
                            tr[2],
                            tr[3],
                        );
                        let result41 = lower_bounds[4][1] + (tw as i64) * ptw;
                        update_best(4, 1, result41);
                    }

                    if y.id != 0 {
                        let y_next = rv.node(pos2 + 2);
                        if can[4][2] || can[4][3] {
                            let tr = travels[4][2];
                            let right_tw = Sequence::tw3_with_travel(
                                &v_pred.seq0_i,
                                &u.seq123,
                                &y_next.seqi_n,
                                tr[2],
                                tr[3],
                            );
                            if can[4][2] {
                                let tr = travels[4][2];
                                let tw = Sequence::tw3_with_travel(
                                    &u_pred.seq0_i,
                                    &v.seq12,
                                    &x2_next.seqi_n,
                                    tr[0],
                                    tr[1],
                                ) + right_tw;
                                let result42 = lower_bounds[4][2] + (tw as i64) * ptw;
                                update_best(4, 2, result42);
                            }
                            if can[4][3] {
                                let tr = travels[4][3];
                                let tw = Sequence::tw3_with_travel(
                                    &u_pred.seq0_i,
                                    &v.seq21,
                                    &x2_next.seqi_n,
                                    tr[0],
                                    tr[1],
                                ) + right_tw;
                                let result43 = lower_bounds[4][3] + (tw as i64) * ptw;
                                update_best(4, 3, result43);
                            }
                        }

                        if y_next.id != 0 && can[4][4] {
                            let y2_next = rv.node(pos2 + 3);
                            let tr = travels[4][4];
                            let tw = Sequence::tw3_with_travel(
                                &u_pred.seq0_i,
                                &v.seq123,
                                &x2_next.seqi_n,
                                tr[0],
                                tr[1],
                            ) + Sequence::tw3_with_travel(
                                &v_pred.seq0_i,
                                &u.seq123,
                                &y2_next.seqi_n,
                                tr[2],
                                tr[3],
                            );
                            let result44 = lower_bounds[4][4] + (tw as i64) * ptw;
                            update_best(4, 4, result44);
                        }
                    }
                }
            }
        }

        if best_i == 0 && best_j == 0 {
            return None;
        } // no improvement

        self.last_plan = MovePlan::InterRoute {
            send1: best_i as u8,
            send2: best_j as u8,
        };
        Some(best_cost - old_total)
    }

    fn run_swapstar(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize) -> i64 {
        debug_assert!(r1 != r2);
        debug_assert!(pos1 > 0 && pos1 < self.routes[r1].nodes.len() - 1);
        debug_assert!(pos2 > 0 && pos2 < self.routes[r2].nodes.len() - 1);

        // The old code materialized a complete prefix/suffix Sequence for
        // every customer on every route refresh, although swap-star only
        // needs its load and distance here.  Removing one customer has an
        // exact three-arc delta, so compute those two scalars on demand.
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

        // First filter on route costs
        let new_pen1 =
            ((new_load1 - self.data.max_capacity).max(0) as i64) * self.params.penalty_capa as i64;
        let new_pen2 =
            ((new_load2 - self.data.max_capacity).max(0) as i64) * self.params.penalty_capa as i64;
        let cost_lb_r1_after_removal = (removed_distance1 as i64) + new_pen1;
        let cost_lb_r2_after_removal = (removed_distance2 as i64) + new_pen2;
        let mut lb_new_total = cost_lb_r1_after_removal + cost_lb_r2_after_removal;
        let max_acceptable_cost = old_total + self.move_credit;

        // first filter on route costs
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

        // Minimum distance detour for reinserting V after removing U.  The
        // two edges incident to U disappear; include the replacement bridge
        // explicitly and scan the remaining original edges.
        let pred_u = node_at(route1_nodes, pos1 - 1);
        let next_u = node_at(route1_nodes, pos1 + 1);
        let mut best_ins_v =
            copy_at(distance_to_v, pred_u.id) + copy_at(distance_from_v, next_u.id) - bridge_u;
        for t in 1..pos1 {
            let pred = node_at(route1_nodes, t - 1);
            let next = node_at(route1_nodes, t);
            best_ins_v = best_ins_v.min(
                copy_at(distance_to_v, pred.id) + copy_at(distance_from_v, next.id)
                    - pred.dist_to_succ,
            );
        }
        for t in (pos1 + 2)..route1_len {
            let pred = node_at(route1_nodes, t - 1);
            let next = node_at(route1_nodes, t);
            best_ins_v = best_ins_v.min(
                copy_at(distance_to_v, pred.id) + copy_at(distance_from_v, next.id)
                    - pred.dist_to_succ,
            );
        }

        // Second filter on route costs
        lb_new_total += best_ins_v as i64;
        if lb_new_total > max_acceptable_cost {
            return NO_MOVE;
        }

        let pred_v = node_at(route2_nodes, pos2 - 1);
        let next_v = node_at(route2_nodes, pos2 + 1);
        let mut best_ins_u =
            copy_at(distance_to_u, pred_v.id) + copy_at(distance_from_u, next_v.id) - bridge_v;
        for t in 1..pos2 {
            let pred = node_at(route2_nodes, t - 1);
            let next = node_at(route2_nodes, t);
            best_ins_u = best_ins_u.min(
                copy_at(distance_to_u, pred.id) + copy_at(distance_from_u, next.id)
                    - pred.dist_to_succ,
            );
        }
        for t in (pos2 + 2)..route2_len {
            let pred = node_at(route2_nodes, t - 1);
            let next = node_at(route2_nodes, t);
            best_ins_u = best_ins_u.min(
                copy_at(distance_to_u, pred.id) + copy_at(distance_from_u, next.id)
                    - pred.dist_to_succ,
            );
        }

        // Third filter on route costs
        lb_new_total += best_ins_u as i64;
        if lb_new_total > max_acceptable_cost {
            return NO_MOVE;
        }

        // Exact TW evaluation for one route cannot rescue a distance/capacity
        // lower bound that already exceeds the accepted total.  Keep the
        // other route's tight insertion lower bound available in both scans.
        let lb_cost2 = cost_lb_r2_after_removal + best_ins_u as i64;

        // Values needed only beyond the distance/capacity filters.
        let ptw = self.params.penalty_tw as i64;

        // Reinsertion of V into r1 \ {U}
        let v_seq1 = node_at(route2_nodes, pos2).seq1;
        let mut best_t1: usize = pos1;
        let mut best_cost1 = i64::MAX / 4;

        // t <= pos1: build right-excluding-U by prepending seq1 fragments
        let mut right_excl_u = node_at(route1_nodes, pos1 + 1).seqi_n;
        for t in (1..=pos1).rev() {
            let left = node_at(route1_nodes, t - 1).seq0_i;
            let d_av = copy_at(distance_to_v, left.last_node as usize);
            let d_vb = copy_at(distance_from_v, right_excl_u.first_node as usize);
            let distance = left.distance + right_excl_u.distance + d_av + d_vb;
            let route1_lb = (distance as i64) + new_pen1;
            if route1_lb < best_cost1 && route1_lb + lb_cost2 <= max_acceptable_cost {
                let tw = Sequence::tw3_with_travel(&left, &v_seq1, &right_excl_u, d_av, d_vb);
                let cand = route1_lb + (tw as i64) * ptw;
                if cand < best_cost1 {
                    best_cost1 = cand;
                    best_t1 = t;
                }
            }
            if t > 1 {
                let travel = if t == pos1 {
                    bridge_u
                } else {
                    node_at(route1_nodes, t - 1).dist_to_succ
                };
                right_excl_u = Sequence::join2_with_travel(
                    &node_at(route1_nodes, t - 1).seq1,
                    &right_excl_u,
                    travel,
                );
            }
        }

        // t > pos1: build left-excluding-U incrementally
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
            if route1_lb < best_cost1 && route1_lb + lb_cost2 <= max_acceptable_cost {
                let tw = Sequence::tw3_with_travel(&left_excl_u, &v_seq1, &right, d_av, d_vb);
                let cand = route1_lb + (tw as i64) * ptw;
                if cand < best_cost1 {
                    best_cost1 = cand;
                    best_t1 = t;
                }
            }
            if t + 1 < route1_len {
                left_excl_u = Sequence::join2_with_travel(
                    &left_excl_u,
                    &node_at(route1_nodes, t).seq1,
                    node_at(route1_nodes, t - 1).dist_to_succ,
                );
            }
        }

        // Fourth filter: one route is exact (TW-aware), the other remains a lower bound
        if best_cost1.saturating_add(lb_cost2) > max_acceptable_cost {
            return NO_MOVE;
        }

        // Reinsertion of U into r2 \ {V}
        let u_seq1 = node_at(route1_nodes, pos1).seq1;
        let mut best_t2: usize = pos2;
        let mut best_cost2 = i64::MAX / 4;

        // t <= pos2: build right-excluding-V by prepending seq1 fragments
        let mut right_excl_v = node_at(route2_nodes, pos2 + 1).seqi_n;
        for t in (1..=pos2).rev() {
            let left = node_at(route2_nodes, t - 1).seq0_i;
            let d_au = copy_at(distance_to_u, left.last_node as usize);
            let d_ub = copy_at(distance_from_u, right_excl_v.first_node as usize);
            let distance = left.distance + right_excl_v.distance + d_au + d_ub;
            let route2_lb = (distance as i64) + new_pen2;
            if route2_lb < best_cost2 && best_cost1 + route2_lb <= max_acceptable_cost {
                let tw = Sequence::tw3_with_travel(&left, &u_seq1, &right_excl_v, d_au, d_ub);
                let cand = route2_lb + (tw as i64) * ptw;
                if cand < best_cost2 {
                    best_cost2 = cand;
                    best_t2 = t;
                }
            }
            if t > 1 {
                let travel = if t == pos2 {
                    bridge_v
                } else {
                    node_at(route2_nodes, t - 1).dist_to_succ
                };
                right_excl_v = Sequence::join2_with_travel(
                    &node_at(route2_nodes, t - 1).seq1,
                    &right_excl_v,
                    travel,
                );
            }
        }

        // t > pos2: build left-excluding-V incrementally
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
            if route2_lb < best_cost2 && best_cost1 + route2_lb <= max_acceptable_cost {
                let tw = Sequence::tw3_with_travel(&left_excl_v, &u_seq1, &right, d_au, d_ub);
                let cand = route2_lb + (tw as i64) * ptw;
                if cand < best_cost2 {
                    best_cost2 = cand;
                    best_t2 = t;
                }
            }
            if t + 1 < route2_len {
                left_excl_v = Sequence::join2_with_travel(
                    &left_excl_v,
                    &node_at(route2_nodes, t).seq1,
                    node_at(route2_nodes, t - 1).dist_to_succ,
                );
            }
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
            for k in neighbors_start..neighbors_end {
                let c2 = copy_at(&self.neighbors_before, k);
                let r2 = copy_at(&self.node_route, c2);

                // Skip if both routes unchanged since last tests for this customer
                if r1 != r2
                    && !(need_r2_stale_check
                        && copy_at(&self.when_last_modified, r2) <= last_tested)
                {
                    // We use pos2 + 1 for the SWAP and RELOCATE moves since c2 is a good predecessor for c1
                    // Moves listed here create the edge c2 => c1, but never insert immediately after a depot
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

                    // Special case to manage insert immediately after a depot
                    if pos1 == 1 {
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
            }

            let capacity_start = copy_at(&self.neighbors_capacity_swap_offsets, c1);
            let capacity_end = copy_at(&self.neighbors_capacity_swap_offsets, c1 + 1);
            for k in capacity_start..capacity_end {
                let c2 = copy_at(&self.neighbors_capacity_swap, k);
                let r2 = copy_at(&self.node_route, c2);

                // Skip if both routes unchanged since last tests for this customer
                if r1 != r2
                    && !(need_r2_stale_check
                        && copy_at(&self.when_last_modified, r2) <= last_tested)
                {
                    let pos2 = copy_at(&self.node_pos, c2);
                    let delta = self.run_swapstar(r1, pos1, r2, pos2);
                    if delta < best_delta {
                        best_delta = delta;
                        best_move = Some(CandidateMove::SwapStar { r1, pos1, r2, pos2 });
                        best_plan = self.last_plan;
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
                        let delta = self.run_2optstar(r1, pos1, r2, pos2);
                        if delta < best_delta {
                            best_delta = delta;
                            best_move = Some(CandidateMove::TwoOptStar { r1, pos1, r2, pos2 });
                            best_plan = self.last_plan;
                        }
                    }

                    // Insert in an empty route
                    let delta = self.run_inter_route(r1, pos1, r2, pos2);
                    if delta < best_delta {
                        best_delta = delta;
                        best_move = Some(CandidateMove::InterRoute { r1, pos1, r2, pos2 });
                        best_plan = self.last_plan;
                    }
                }
            }
        }

        // Intra-route moves
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
