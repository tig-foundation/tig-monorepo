use super::params::Params;
use super::problem::Problem;
use super::sequence::Sequence;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use std::cmp::min;
use std::sync::Arc;

const NO_MOVE: i64 = i64::MAX;

/// Requires `index < slice.len()`.
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
    pub when_last_modified: Vec<usize>, // indexed by route
    pub when_last_tested: Vec<usize>,   // indexed by customer id
    pub nb_moves: usize,
    pub move_credit: i64, // deterioration budget; negative means disabled
    /// Maximum accepted delta for the current customer.
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

        // Padding permits full eight-lane reads at the end of either list.
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

    // Routes contain depot endpoints; client ids index all node tables.
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
        // Reserve `count` slots, initialize each once, then set the length.
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
        let mut best_delta: i64 = i64::MAX; // best acceptable move
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
                    self.move_bar = self.move_credit.min(delta.saturating_sub(1));
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
