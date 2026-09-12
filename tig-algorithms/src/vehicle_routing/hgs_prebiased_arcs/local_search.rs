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

// The hot route record keeps only the parts consumed by each fragment role.
// Prefixes are left operands (E, W); suffixes are right operands (L, W).
#[derive(Clone, Copy, Debug, Default)]
struct Temporal { end: i32, latest: i32, duration: i32, warp: i32 }
impl Temporal {
    #[inline(always)]
    fn from(s: Sequence) -> Self {
        Self { end: s.earliest_end, latest: s.tau_plus, duration: s.duration_net, warp: s.tw }
    }
    #[inline(always)]
    fn expand(self) -> Sequence {
        Sequence { earliest_end: self.end, tau_plus: self.latest, duration_net: self.duration, tw: self.warp, ..Sequence::default() }
    }
}
#[derive(Clone, Copy, Debug, Default)]
struct Boundary { time: i32, warp: i32, load: i32, distance: i32 }

#[repr(C, align(16))]
#[derive(Clone, Debug, Default)]
pub struct Node {
    id: usize,
    dist_to_succ: i32,
    load: i32,
    prefix: Boundary,
    suffix: Boundary,
    t1: Temporal,
    t12: Temporal,
    t21: Temporal,
    t123: Temporal,
    reversed_travel: i32,
    bridge: i32,
    removal_delta: i32,
    row_start: u32,
}
impl Node {
    #[inline]
    fn new(data: &Problem, id: usize) -> Self {
        let singleton = Sequence::singleton(data,id);
        Self { id, row_start: (id * data.nb_nodes) as u32, load: singleton.load, t1: Temporal::from(singleton), ..Self::default() }
    }
    #[inline(always)]
    fn distance_row<'a>(&self, data: &'a Problem) -> &'a [i32] {
        unsafe { data.distance_matrix.get_unchecked(self.row_start as usize..self.row_start as usize+data.nb_nodes) }
    }
    #[inline(always)]
    fn seq1(&self) -> Sequence {
        Sequence { tw: 0, load: self.load, first_node: self.id as u16, last_node: self.id as u16, ..self.t1.expand() }
    }
    #[inline(always)]
    fn seq12(&self) -> Sequence { self.t12.expand() }
    #[inline(always)]
    fn seq21(&self) -> Sequence { Sequence { distance: self.reversed_travel, ..self.t21.expand() } }
    #[inline(always)]
    fn seq123(&self) -> Sequence { self.t123.expand() }
    #[inline(always)]
    fn seq0_i(&self) -> Sequence {
        Sequence { earliest_end: self.prefix.time, tw: self.prefix.warp,
            load: self.prefix.load, distance: self.prefix.distance, last_node: self.id as u16, ..Sequence::default() }
    }
    #[inline(always)]
    fn seqi_n(&self) -> Sequence {
        Sequence { tau_plus: self.suffix.time, tw: self.suffix.warp,
            load: self.suffix.load, distance: self.suffix.distance, first_node: self.id as u16, ..Sequence::default() }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Route {
    refresh_from:usize,refresh_to:usize,mapping_end:usize,transfer:bool,
    arc_ids:Vec<i32>,arc_travel:Vec<i32>,
    insertion_version: u32,
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

#[derive(Clone, Copy, Default)]
struct NodeLocation {
    route_and_position: u64,
}

#[derive(Clone, Copy, Default)]
struct InsertionLowerBound { version:u32, value:i32 }
#[derive(Clone,Copy,Default,Debug,PartialEq,Eq)]
struct NeighborMasks { before:u64 }
#[derive(Clone,Copy)]
struct InsertionBoundary { distance:i32,prefix_end:i32,suffix_latest:i32,warp:i32,pred:u16,next:u16,position:u32 }
impl InsertionBoundary {
    #[inline(always)]
    fn from(left:Sequence,right:Sequence,position:usize)->Self {
        Self{distance:left.distance+right.distance,prefix_end:left.earliest_end,suffix_latest:right.tau_plus,
            warp:left.tw+right.tw,pred:left.last_node,next:right.first_node,position:position as u32}
    }
}
#[derive(Default)]
struct RemovalCache { version:u32,entries:Vec<InsertionBoundary> }
#[repr(C,align(64))]
#[derive(Clone,Copy)]
struct BatchCut([i32;16]);
#[derive(Clone,Copy)]
struct SwapSource {pred:usize,next:usize,cost:i64,max_cost:i64,id:usize,removed_distance:i32,remaining_load:i32,load:i32,bridge:i32,row_start:u32,singleton:bool}

// These private pointers exist only while evaluating a customer. Route buffers
// are unchanged until the final selected move is applied, after their last use.
#[derive(Clone,Copy)]
struct InterSource {
    cost:i64,distance:i32,load:i32,
    sent:[i32;3],removed:[i32;3],bridges:[i32;3],reverse_pair:i32,singleton:bool,
}
#[repr(C,align(32))]
struct LanePermutation([i32;8]);
static CUT_LANE_PERMUTATIONS:[LanePermutation;256]=[
LanePermutation([0,0,0,0,0,0,0,0]),
LanePermutation([0,0,0,0,0,0,0,0]),
LanePermutation([1,0,0,0,0,0,0,0]),
LanePermutation([0,1,0,0,0,0,0,0]),
LanePermutation([2,0,0,0,0,0,0,0]),
LanePermutation([0,2,0,0,0,0,0,0]),
LanePermutation([1,2,0,0,0,0,0,0]),
LanePermutation([0,1,2,0,0,0,0,0]),
LanePermutation([3,0,0,0,0,0,0,0]),
LanePermutation([0,3,0,0,0,0,0,0]),
LanePermutation([1,3,0,0,0,0,0,0]),
LanePermutation([0,1,3,0,0,0,0,0]),
LanePermutation([2,3,0,0,0,0,0,0]),
LanePermutation([0,2,3,0,0,0,0,0]),
LanePermutation([1,2,3,0,0,0,0,0]),
LanePermutation([0,1,2,3,0,0,0,0]),
LanePermutation([4,0,0,0,0,0,0,0]),
LanePermutation([0,4,0,0,0,0,0,0]),
LanePermutation([1,4,0,0,0,0,0,0]),
LanePermutation([0,1,4,0,0,0,0,0]),
LanePermutation([2,4,0,0,0,0,0,0]),
LanePermutation([0,2,4,0,0,0,0,0]),
LanePermutation([1,2,4,0,0,0,0,0]),
LanePermutation([0,1,2,4,0,0,0,0]),
LanePermutation([3,4,0,0,0,0,0,0]),
LanePermutation([0,3,4,0,0,0,0,0]),
LanePermutation([1,3,4,0,0,0,0,0]),
LanePermutation([0,1,3,4,0,0,0,0]),
LanePermutation([2,3,4,0,0,0,0,0]),
LanePermutation([0,2,3,4,0,0,0,0]),
LanePermutation([1,2,3,4,0,0,0,0]),
LanePermutation([0,1,2,3,4,0,0,0]),
LanePermutation([5,0,0,0,0,0,0,0]),
LanePermutation([0,5,0,0,0,0,0,0]),
LanePermutation([1,5,0,0,0,0,0,0]),
LanePermutation([0,1,5,0,0,0,0,0]),
LanePermutation([2,5,0,0,0,0,0,0]),
LanePermutation([0,2,5,0,0,0,0,0]),
LanePermutation([1,2,5,0,0,0,0,0]),
LanePermutation([0,1,2,5,0,0,0,0]),
LanePermutation([3,5,0,0,0,0,0,0]),
LanePermutation([0,3,5,0,0,0,0,0]),
LanePermutation([1,3,5,0,0,0,0,0]),
LanePermutation([0,1,3,5,0,0,0,0]),
LanePermutation([2,3,5,0,0,0,0,0]),
LanePermutation([0,2,3,5,0,0,0,0]),
LanePermutation([1,2,3,5,0,0,0,0]),
LanePermutation([0,1,2,3,5,0,0,0]),
LanePermutation([4,5,0,0,0,0,0,0]),
LanePermutation([0,4,5,0,0,0,0,0]),
LanePermutation([1,4,5,0,0,0,0,0]),
LanePermutation([0,1,4,5,0,0,0,0]),
LanePermutation([2,4,5,0,0,0,0,0]),
LanePermutation([0,2,4,5,0,0,0,0]),
LanePermutation([1,2,4,5,0,0,0,0]),
LanePermutation([0,1,2,4,5,0,0,0]),
LanePermutation([3,4,5,0,0,0,0,0]),
LanePermutation([0,3,4,5,0,0,0,0]),
LanePermutation([1,3,4,5,0,0,0,0]),
LanePermutation([0,1,3,4,5,0,0,0]),
LanePermutation([2,3,4,5,0,0,0,0]),
LanePermutation([0,2,3,4,5,0,0,0]),
LanePermutation([1,2,3,4,5,0,0,0]),
LanePermutation([0,1,2,3,4,5,0,0]),
LanePermutation([6,0,0,0,0,0,0,0]),
LanePermutation([0,6,0,0,0,0,0,0]),
LanePermutation([1,6,0,0,0,0,0,0]),
LanePermutation([0,1,6,0,0,0,0,0]),
LanePermutation([2,6,0,0,0,0,0,0]),
LanePermutation([0,2,6,0,0,0,0,0]),
LanePermutation([1,2,6,0,0,0,0,0]),
LanePermutation([0,1,2,6,0,0,0,0]),
LanePermutation([3,6,0,0,0,0,0,0]),
LanePermutation([0,3,6,0,0,0,0,0]),
LanePermutation([1,3,6,0,0,0,0,0]),
LanePermutation([0,1,3,6,0,0,0,0]),
LanePermutation([2,3,6,0,0,0,0,0]),
LanePermutation([0,2,3,6,0,0,0,0]),
LanePermutation([1,2,3,6,0,0,0,0]),
LanePermutation([0,1,2,3,6,0,0,0]),
LanePermutation([4,6,0,0,0,0,0,0]),
LanePermutation([0,4,6,0,0,0,0,0]),
LanePermutation([1,4,6,0,0,0,0,0]),
LanePermutation([0,1,4,6,0,0,0,0]),
LanePermutation([2,4,6,0,0,0,0,0]),
LanePermutation([0,2,4,6,0,0,0,0]),
LanePermutation([1,2,4,6,0,0,0,0]),
LanePermutation([0,1,2,4,6,0,0,0]),
LanePermutation([3,4,6,0,0,0,0,0]),
LanePermutation([0,3,4,6,0,0,0,0]),
LanePermutation([1,3,4,6,0,0,0,0]),
LanePermutation([0,1,3,4,6,0,0,0]),
LanePermutation([2,3,4,6,0,0,0,0]),
LanePermutation([0,2,3,4,6,0,0,0]),
LanePermutation([1,2,3,4,6,0,0,0]),
LanePermutation([0,1,2,3,4,6,0,0]),
LanePermutation([5,6,0,0,0,0,0,0]),
LanePermutation([0,5,6,0,0,0,0,0]),
LanePermutation([1,5,6,0,0,0,0,0]),
LanePermutation([0,1,5,6,0,0,0,0]),
LanePermutation([2,5,6,0,0,0,0,0]),
LanePermutation([0,2,5,6,0,0,0,0]),
LanePermutation([1,2,5,6,0,0,0,0]),
LanePermutation([0,1,2,5,6,0,0,0]),
LanePermutation([3,5,6,0,0,0,0,0]),
LanePermutation([0,3,5,6,0,0,0,0]),
LanePermutation([1,3,5,6,0,0,0,0]),
LanePermutation([0,1,3,5,6,0,0,0]),
LanePermutation([2,3,5,6,0,0,0,0]),
LanePermutation([0,2,3,5,6,0,0,0]),
LanePermutation([1,2,3,5,6,0,0,0]),
LanePermutation([0,1,2,3,5,6,0,0]),
LanePermutation([4,5,6,0,0,0,0,0]),
LanePermutation([0,4,5,6,0,0,0,0]),
LanePermutation([1,4,5,6,0,0,0,0]),
LanePermutation([0,1,4,5,6,0,0,0]),
LanePermutation([2,4,5,6,0,0,0,0]),
LanePermutation([0,2,4,5,6,0,0,0]),
LanePermutation([1,2,4,5,6,0,0,0]),
LanePermutation([0,1,2,4,5,6,0,0]),
LanePermutation([3,4,5,6,0,0,0,0]),
LanePermutation([0,3,4,5,6,0,0,0]),
LanePermutation([1,3,4,5,6,0,0,0]),
LanePermutation([0,1,3,4,5,6,0,0]),
LanePermutation([2,3,4,5,6,0,0,0]),
LanePermutation([0,2,3,4,5,6,0,0]),
LanePermutation([1,2,3,4,5,6,0,0]),
LanePermutation([0,1,2,3,4,5,6,0]),
LanePermutation([7,0,0,0,0,0,0,0]),
LanePermutation([0,7,0,0,0,0,0,0]),
LanePermutation([1,7,0,0,0,0,0,0]),
LanePermutation([0,1,7,0,0,0,0,0]),
LanePermutation([2,7,0,0,0,0,0,0]),
LanePermutation([0,2,7,0,0,0,0,0]),
LanePermutation([1,2,7,0,0,0,0,0]),
LanePermutation([0,1,2,7,0,0,0,0]),
LanePermutation([3,7,0,0,0,0,0,0]),
LanePermutation([0,3,7,0,0,0,0,0]),
LanePermutation([1,3,7,0,0,0,0,0]),
LanePermutation([0,1,3,7,0,0,0,0]),
LanePermutation([2,3,7,0,0,0,0,0]),
LanePermutation([0,2,3,7,0,0,0,0]),
LanePermutation([1,2,3,7,0,0,0,0]),
LanePermutation([0,1,2,3,7,0,0,0]),
LanePermutation([4,7,0,0,0,0,0,0]),
LanePermutation([0,4,7,0,0,0,0,0]),
LanePermutation([1,4,7,0,0,0,0,0]),
LanePermutation([0,1,4,7,0,0,0,0]),
LanePermutation([2,4,7,0,0,0,0,0]),
LanePermutation([0,2,4,7,0,0,0,0]),
LanePermutation([1,2,4,7,0,0,0,0]),
LanePermutation([0,1,2,4,7,0,0,0]),
LanePermutation([3,4,7,0,0,0,0,0]),
LanePermutation([0,3,4,7,0,0,0,0]),
LanePermutation([1,3,4,7,0,0,0,0]),
LanePermutation([0,1,3,4,7,0,0,0]),
LanePermutation([2,3,4,7,0,0,0,0]),
LanePermutation([0,2,3,4,7,0,0,0]),
LanePermutation([1,2,3,4,7,0,0,0]),
LanePermutation([0,1,2,3,4,7,0,0]),
LanePermutation([5,7,0,0,0,0,0,0]),
LanePermutation([0,5,7,0,0,0,0,0]),
LanePermutation([1,5,7,0,0,0,0,0]),
LanePermutation([0,1,5,7,0,0,0,0]),
LanePermutation([2,5,7,0,0,0,0,0]),
LanePermutation([0,2,5,7,0,0,0,0]),
LanePermutation([1,2,5,7,0,0,0,0]),
LanePermutation([0,1,2,5,7,0,0,0]),
LanePermutation([3,5,7,0,0,0,0,0]),
LanePermutation([0,3,5,7,0,0,0,0]),
LanePermutation([1,3,5,7,0,0,0,0]),
LanePermutation([0,1,3,5,7,0,0,0]),
LanePermutation([2,3,5,7,0,0,0,0]),
LanePermutation([0,2,3,5,7,0,0,0]),
LanePermutation([1,2,3,5,7,0,0,0]),
LanePermutation([0,1,2,3,5,7,0,0]),
LanePermutation([4,5,7,0,0,0,0,0]),
LanePermutation([0,4,5,7,0,0,0,0]),
LanePermutation([1,4,5,7,0,0,0,0]),
LanePermutation([0,1,4,5,7,0,0,0]),
LanePermutation([2,4,5,7,0,0,0,0]),
LanePermutation([0,2,4,5,7,0,0,0]),
LanePermutation([1,2,4,5,7,0,0,0]),
LanePermutation([0,1,2,4,5,7,0,0]),
LanePermutation([3,4,5,7,0,0,0,0]),
LanePermutation([0,3,4,5,7,0,0,0]),
LanePermutation([1,3,4,5,7,0,0,0]),
LanePermutation([0,1,3,4,5,7,0,0]),
LanePermutation([2,3,4,5,7,0,0,0]),
LanePermutation([0,2,3,4,5,7,0,0]),
LanePermutation([1,2,3,4,5,7,0,0]),
LanePermutation([0,1,2,3,4,5,7,0]),
LanePermutation([6,7,0,0,0,0,0,0]),
LanePermutation([0,6,7,0,0,0,0,0]),
LanePermutation([1,6,7,0,0,0,0,0]),
LanePermutation([0,1,6,7,0,0,0,0]),
LanePermutation([2,6,7,0,0,0,0,0]),
LanePermutation([0,2,6,7,0,0,0,0]),
LanePermutation([1,2,6,7,0,0,0,0]),
LanePermutation([0,1,2,6,7,0,0,0]),
LanePermutation([3,6,7,0,0,0,0,0]),
LanePermutation([0,3,6,7,0,0,0,0]),
LanePermutation([1,3,6,7,0,0,0,0]),
LanePermutation([0,1,3,6,7,0,0,0]),
LanePermutation([2,3,6,7,0,0,0,0]),
LanePermutation([0,2,3,6,7,0,0,0]),
LanePermutation([1,2,3,6,7,0,0,0]),
LanePermutation([0,1,2,3,6,7,0,0]),
LanePermutation([4,6,7,0,0,0,0,0]),
LanePermutation([0,4,6,7,0,0,0,0]),
LanePermutation([1,4,6,7,0,0,0,0]),
LanePermutation([0,1,4,6,7,0,0,0]),
LanePermutation([2,4,6,7,0,0,0,0]),
LanePermutation([0,2,4,6,7,0,0,0]),
LanePermutation([1,2,4,6,7,0,0,0]),
LanePermutation([0,1,2,4,6,7,0,0]),
LanePermutation([3,4,6,7,0,0,0,0]),
LanePermutation([0,3,4,6,7,0,0,0]),
LanePermutation([1,3,4,6,7,0,0,0]),
LanePermutation([0,1,3,4,6,7,0,0]),
LanePermutation([2,3,4,6,7,0,0,0]),
LanePermutation([0,2,3,4,6,7,0,0]),
LanePermutation([1,2,3,4,6,7,0,0]),
LanePermutation([0,1,2,3,4,6,7,0]),
LanePermutation([5,6,7,0,0,0,0,0]),
LanePermutation([0,5,6,7,0,0,0,0]),
LanePermutation([1,5,6,7,0,0,0,0]),
LanePermutation([0,1,5,6,7,0,0,0]),
LanePermutation([2,5,6,7,0,0,0,0]),
LanePermutation([0,2,5,6,7,0,0,0]),
LanePermutation([1,2,5,6,7,0,0,0]),
LanePermutation([0,1,2,5,6,7,0,0]),
LanePermutation([3,5,6,7,0,0,0,0]),
LanePermutation([0,3,5,6,7,0,0,0]),
LanePermutation([1,3,5,6,7,0,0,0]),
LanePermutation([0,1,3,5,6,7,0,0]),
LanePermutation([2,3,5,6,7,0,0,0]),
LanePermutation([0,2,3,5,6,7,0,0]),
LanePermutation([1,2,3,5,6,7,0,0]),
LanePermutation([0,1,2,3,5,6,7,0]),
LanePermutation([4,5,6,7,0,0,0,0]),
LanePermutation([0,4,5,6,7,0,0,0]),
LanePermutation([1,4,5,6,7,0,0,0]),
LanePermutation([0,1,4,5,6,7,0,0]),
LanePermutation([2,4,5,6,7,0,0,0]),
LanePermutation([0,2,4,5,6,7,0,0]),
LanePermutation([1,2,4,5,6,7,0,0]),
LanePermutation([0,1,2,4,5,6,7,0]),
LanePermutation([3,4,5,6,7,0,0,0]),
LanePermutation([0,3,4,5,6,7,0,0]),
LanePermutation([1,3,4,5,6,7,0,0]),
LanePermutation([0,1,3,4,5,6,7,0]),
LanePermutation([2,3,4,5,6,7,0,0]),
LanePermutation([0,2,3,4,5,6,7,0]),
LanePermutation([1,2,3,4,5,6,7,0]),
LanePermutation([0,1,2,3,4,5,6,7])
];
pub struct LocalSearch {
    minpos_arcs:bool,
    sparse_bmi2:bool,
    batch_bounds_valid:bool,batch_input_valid:bool,batch_cuts:Vec<BatchCut>,
    query_credit:i64,
    removal_cache:Vec<RemovalCache>,
    vector_arcs:bool,
    factored_bounds_valid:bool,
    gate_audit_valid:bool,
    masked_neighbors:bool, masked_customers:Vec<bool>,
    reverse_neighbors:Vec<u32>, reverse_offsets:Vec<usize>,
    membership_routes:Vec<usize>, route_neighbor_masks:Vec<NeighborMasks>, mask_stride:usize,
    recent_prev:Vec<usize>,recent_next:Vec<usize>,recent_head:usize,
    insertion_clock: u32,
    insertion_bounds: Vec<InsertionLowerBound>,
    capacity_costs: Vec<i64>,
    capacity_cost_penalty: usize,
    pub data: Arc<Problem>,
    neighbors_before: Vec<u32>,
    neighbors_before_offsets: Vec<usize>,
    neighbors_capacity_swap: Vec<u32>,
    neighbors_capacity_swap_offsets: Vec<usize>,
    pub loop_order_nodes: Vec<usize>,
    pub params: Params,
    pub cost: i64,
    pub routes: Vec<Route>,
    node_locations: Vec<NodeLocation>,
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
                neighbors_before.extend(prox[..keep].iter().map(|&key| (key & 0xffff) as u32));
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
                    Some(j as u32)
                } else {
                    None
                }
            }));
            neighbors_capacity_swap_offsets[i + 1] = neighbors_capacity_swap.len();
        }

        let masked_customers:Vec<bool>=(0..n).map(|i|
            neighbors_before_offsets[i+1]-neighbors_before_offsets[i]
                +neighbors_capacity_swap_offsets[i+1]-neighbors_capacity_swap_offsets[i]<=64).collect();
        let masked_neighbors=masked_customers.iter().any(|&x|x);
        let mut reverse_neighbors=Vec::new();let mut reverse_offsets=vec![0;n+1];
        if masked_neighbors {
            let mut rows=vec![Vec::<u32>::new();n];
            for c in 1..n {
                if !masked_customers[c] {continue;}
                for (bit,&v) in neighbors_before[neighbors_before_offsets[c]..neighbors_before_offsets[c+1]].iter().enumerate() {
                    rows[v as usize].push(c as u32|((bit as u32)<<16));
                }
                for (bit,&v) in neighbors_capacity_swap[neighbors_capacity_swap_offsets[c]..neighbors_capacity_swap_offsets[c+1]].iter().enumerate() {
                    rows[v as usize].push(c as u32|(((bit+neighbors_before_offsets[c+1]-neighbors_before_offsets[c]) as u32)<<16));
                }
            }
            for (i,row) in rows.iter().enumerate() {
                reverse_neighbors.extend_from_slice(row);reverse_offsets[i+1]=reverse_neighbors.len();
            }
        }

        let sum_load = data.node_data.iter().map(|nd| nd.demand.max(0) as u64).sum::<u64>();
        let capacity_costs = if sum_load <= 65_536 && data.node_data[0].demand == 0 && data.node_data.iter().all(|nd| nd.demand >= 0) {
            (0..=sum_load as usize).map(|load| (load as i64-data.max_capacity as i64).max(0)*params.penalty_capa as i64).collect()
        } else { Vec::new() };
        let gate_audit_valid=data.distance_matrix.iter().all(|&d|d>=0)
            && data.node_data.iter().all(|v|v.service_time>=0 && v.start_tw<=v.end_tw)
            && (data.node_data.iter().map(|v|(v.start_tw as i64).abs().max((v.end_tw as i64).abs())+v.service_time as i64).max().unwrap_or(0)
                +data.distance_matrix.iter().copied().max().unwrap_or(0) as i64)*(4*n as i64+32)<(i32::MAX as i64)/2;
        #[cfg(target_arch="x86_64")]
        let vector_arcs=std::is_x86_feature_detected!("avx2") && gate_audit_valid;
        #[cfg(not(target_arch="x86_64"))]
        let vector_arcs=false;
        let batch_input_valid=vector_arcs && gate_audit_valid && n<=32767 && data.max_capacity>=0 && data.max_capacity<=67_108_864
            && data.distance_matrix.iter().copied().max().unwrap_or(0) as i64*(4*n as i64+32)<=67_108_864;
        #[cfg(target_arch="x86_64")]
        let sparse_bmi2=std::is_x86_feature_detected!("bmi2") && std::is_x86_feature_detected!("popcnt");
        #[cfg(not(target_arch="x86_64"))]
        let sparse_bmi2=false;
        let minpos_arcs=vector_arcs && data.distance_matrix.iter().all(|&d|d>=0 && d<=16_383);
        Self {
            minpos_arcs,
            sparse_bmi2,
            batch_bounds_valid:false,batch_input_valid,batch_cuts:vec![BatchCut([0;16]);n],
            query_credit:0,removal_cache:(0..n).map(|_|RemovalCache::default()).collect(),vector_arcs,
            gate_audit_valid,factored_bounds_valid:false,
            masked_neighbors,masked_customers,reverse_neighbors,reverse_offsets,
            membership_routes:vec![usize::MAX;n],route_neighbor_masks:Vec::new(),mask_stride:0,
            recent_prev:Vec::new(),recent_next:Vec::new(),recent_head:usize::MAX,
            insertion_clock:0, insertion_bounds:Vec::new(),
            capacity_costs, capacity_cost_penalty: params.penalty_capa,
            data,
            neighbors_before,
            neighbors_before_offsets,
            neighbors_capacity_swap,
            neighbors_capacity_swap_offsets,
            loop_order_nodes: (1..n).collect(),
            params,
            cost: 0,
            routes: Vec::new(),
            node_locations: Vec::new(),
            empty_routes: Vec::new(),
            empty_route_pos: Vec::new(),
            when_last_modified: Vec::new(),
            when_last_tested: vec![0; n],
            nb_moves: 0,
            move_credit: 0,
            last_plan: MovePlan::None,
        }
    }


    #[cfg(target_arch="x86_64")]
    #[target_feature(enable="avx2")]
    unsafe fn refill_positive_capacity(costs:&mut [i64],start:usize,penalty:i64) {
        use std::arch::x86_64::*;
        // Entries at load <= capacity are always zero for this immutable
        // problem and need no update. The suffix starts at one penalty.
        let mut k=start;let mut value=_mm256_setr_epi64x(penalty,2*penalty,3*penalty,4*penalty);
        let stride=_mm256_set1_epi64x(4*penalty);
        while k+16<=costs.len() {
            _mm256_storeu_si256(costs.as_mut_ptr().add(k) as *mut __m256i,value);value=_mm256_add_epi64(value,stride);
            _mm256_storeu_si256(costs.as_mut_ptr().add(k+4) as *mut __m256i,value);value=_mm256_add_epi64(value,stride);
            _mm256_storeu_si256(costs.as_mut_ptr().add(k+8) as *mut __m256i,value);value=_mm256_add_epi64(value,stride);
            _mm256_storeu_si256(costs.as_mut_ptr().add(k+12) as *mut __m256i,value);value=_mm256_add_epi64(value,stride);
            k+=16;
        }
        while k+4<=costs.len() {
            _mm256_storeu_si256(costs.as_mut_ptr().add(k) as *mut __m256i,value);value=_mm256_add_epi64(value,stride);k+=4;
        }
        let mut scalar=(k-start+1) as i64*penalty;
        while k<costs.len() {*costs.get_unchecked_mut(k)=scalar;scalar+=penalty;k+=1;}
    }

    fn refresh_capacity_costs(&mut self) {
        self.factored_bounds_valid=self.gate_audit_valid && !self.capacity_costs.is_empty()
            && self.params.penalty_capa<=1_000_000 && self.params.penalty_tw<=1_000_000;

        self.batch_bounds_valid=self.batch_input_valid && self.factored_bounds_valid
            && self.capacity_costs.len() as u64*self.params.penalty_capa as u64<=67_108_864;
        if self.capacity_cost_penalty != self.params.penalty_capa {
            self.capacity_cost_penalty = self.params.penalty_capa;
            #[cfg(target_arch="x86_64")]
            if self.vector_arcs && self.factored_bounds_valid && self.data.max_capacity>=0 {
                let start=(self.data.max_capacity as usize+1).min(self.capacity_costs.len());
                unsafe{Self::refill_positive_capacity(&mut self.capacity_costs,start,self.params.penalty_capa as i64);}
                
                return;
            }
            for (load,cost) in self.capacity_costs.iter_mut().enumerate() {
                *cost = (load as i64-self.data.max_capacity as i64).max(0)*self.params.penalty_capa as i64;
            }
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

    #[inline(never)]
    fn apply_planned_move(&mut self, mv: CandidateMove, plan: MovePlan) -> Option<i64> {
        // A transferred tail retains its own suffix and short summaries.
        // Refresh starts at the new splice; its far end is the first node of
        // the unchanged tail. Prefix and suffix refresh can meet at one cut.
        let spans=match (mv,plan) {
            (CandidateMove::TwoOptStar{r1,pos1,r2,pos2},MovePlan::TwoOptStar)=>Some((r1,pos1,pos1,r2,pos2,pos2)),
            (CandidateMove::InterRoute{r1,pos1,r2,pos2},MovePlan::InterRoute{send1,send2})=>{
                let count=|kind:u8|if kind==4 {3} else if kind==3 {2} else {kind as usize};
                Some((r1,pos1,pos1+count(send2),r2,pos2,pos2+count(send1)))
            },
            (CandidateMove::SwapStar{r1,pos1,r2,pos2},MovePlan::SwapStar{insert1,insert2})=>
                Some((r1,pos1.min(insert1),pos1.max(insert1)+1,r2,pos2.min(insert2),pos2.max(insert2)+1)),
            _=>None,
        };
        if let Some((r1,lo1,end1,r2,lo2,end2))=spans {
            let r=unsafe{self.routes.get_unchecked_mut(r1)};r.refresh_from=lo1;r.refresh_to=end1;r.mapping_end=usize::MAX;r.transfer=true;
            let r=unsafe{self.routes.get_unchecked_mut(r2)};r.refresh_from=lo2;r.refresh_to=end2;r.mapping_end=usize::MAX;r.transfer=true;
        }
        // Outside the enclosing interval the node order and all arc endpoints
        // are unchanged. Zero refresh_to means a complete refresh.
        let changed=match (mv,plan) {
            (CandidateMove::IntraRelocate{r1,pos1},MovePlan::IntraRelocate{target})=>Some((r1,pos1.min(target),pos1.max(target))),
            (CandidateMove::IntraOrOpt2{r1,pos1},MovePlan::IntraOrOpt2{target,..})=>Some((r1,pos1.min(target),pos1.max(target)+2)),
            (CandidateMove::IntraSwap{r1,pos1},MovePlan::IntraSwap{target})=>Some((r1,pos1.min(target),pos1.max(target))),
            (CandidateMove::Intra2Opt{r1,pos1},MovePlan::Intra2Opt{end})=>Some((r1,pos1,end)),
            _=>None,
        };
        if let Some((rid,lo,hi))=changed {let r=unsafe{self.routes.get_unchecked_mut(rid)};r.refresh_from=lo;r.refresh_to=(hi+1).min(r.nodes.len());r.mapping_end=r.refresh_to;r.transfer=false;}
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
        self.refresh_capacity_costs();
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
        self.insertion_bounds.resize(n*self.routes.len(),InsertionLowerBound::default());
        if self.masked_neighbors {
            let size=n*self.routes.len();
            self.mask_stride=self.routes.len();
            if self.route_neighbor_masks.len()!=size {
                self.route_neighbor_masks.clear();self.route_neighbor_masks.resize(size,NeighborMasks::default());
                self.membership_routes.fill(usize::MAX);
                self.recent_prev.clear();self.recent_next.clear();self.recent_head=usize::MAX;
                self.recent_prev.resize(self.routes.len(),usize::MAX);
                self.recent_next.resize(self.routes.len(),usize::MAX);
            }
        }

        self.node_locations.clear();
        self.node_locations.resize(n,NodeLocation::default());
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
            if inherited && is_feasible {
                // The route timestamp above covers every customer in this route.
            }
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

    fn refresh_metrics_partial(&mut self,rid:usize) {
        let data = self.data.as_ref();
        debug_assert!(rid < self.routes.len());
        let r = unsafe { self.routes.get_unchecked_mut(rid) };
        let nodes = &mut r.nodes;
        let len = nodes.len();
        debug_assert!(len >= 2);
        let ptr = nodes.as_mut_ptr();
        let partial=r.refresh_to>0;
        let first=if partial {r.refresh_from.min(len-1)} else {0};
        let end=if partial {r.refresh_to.min(len)} else {len};


        // Materialize every route arc once. Prefixes, suffixes, and the
        // short forward sequences below all reuse these exact same travels.
        for pos in first.saturating_sub(1)..(end+1).min(len) {
            let id = unsafe { (*ptr.add(pos)).id };
            let dist_to_succ = if pos + 1 < len {
                data.dm(id, unsafe { (*ptr.add(pos + 1)).id })
            } else {
                0
            };
            unsafe {
                (*ptr.add(pos)).dist_to_succ = dist_to_succ;
                if pos > 0 && pos + 1 < len {
                    let bridge = data.dm((*ptr.add(pos-1)).id, (*ptr.add(pos+1)).id);
                    (*ptr.add(pos)).bridge = bridge;
                    (*ptr.add(pos)).removal_delta = bridge - (*ptr.add(pos-1)).dist_to_succ - dist_to_succ;
                }
            }
        }

        if self.vector_arcs {
            let padded=(len-1+7)&!7;
            r.arc_ids.resize(padded+1,0);r.arc_travel.resize(padded,1_073_758_208);
            for k in 0..len {r.arc_ids[k]=unsafe{(*ptr.add(k)).id as i32};}
            r.arc_ids[len..].fill(0);
            for k in 0..len-1 {r.arc_travel[k]=16_384-unsafe{(*ptr.add(k)).dist_to_succ};}
            r.arc_travel[len-1..].fill(1_073_758_208);
        }

        // forward pass: seq0_i
        let forward_start=first.max(1);
        let mut acc_fwd=if partial && first>0 {unsafe{(*ptr.add(first-1)).seq0_i()}} else {unsafe{(*ptr).seq1()}};
        if !partial || first==0 {
            unsafe{(*ptr).prefix=Boundary{time:acc_fwd.earliest_end,warp:acc_fwd.tw,load:acc_fwd.load,distance:acc_fwd.distance};}
        }
        for pos in forward_start..len {
            let singleton = unsafe { (*ptr.add(pos)).seq1() };
            let travel = unsafe { (*ptr.add(pos - 1)).dist_to_succ };
            acc_fwd = Sequence::join2_with_travel(&acc_fwd, &singleton, travel);
            unsafe {
                (*ptr.add(pos)).prefix = Boundary { time: acc_fwd.earliest_end, warp: acc_fwd.tw, load: acc_fwd.load, distance: acc_fwd.distance };
            }
        }

        // backward pass: seqi_n
        let backward_start=end.min(len-1);
        let mut acc_bwd=if partial && end<len {unsafe{(*ptr.add(end)).seqi_n()}} else {unsafe{(*ptr.add(len-1)).seq1()}};
        if !partial || end>=len {
            unsafe{(*ptr.add(len-1)).suffix=Boundary{time:acc_bwd.tau_plus,warp:acc_bwd.tw,load:acc_bwd.load,distance:acc_bwd.distance};}
        }
        for pos in (0..backward_start).rev() {
            let singleton = unsafe { (*ptr.add(pos)).seq1() };
            let travel = unsafe { (*ptr.add(pos)).dist_to_succ };
            acc_bwd = Sequence::join2_with_travel(&singleton, &acc_bwd, travel);
            unsafe {
                (*ptr.add(pos)).suffix = Boundary { time: acc_bwd.tau_plus, warp: acc_bwd.tw, load: acc_bwd.load, distance: acc_bwd.distance };
            }
        }

        // Only short sequences meeting the changed interval can differ.
        for pos in first.saturating_sub(2)..end {
            let id = unsafe { (*ptr.add(pos)).id };
            let singleton = unsafe { (*ptr.add(pos)).seq1() };
            if pos + 1 < len {
                let next = unsafe { (*ptr.add(pos + 1)).seq1() };
                let forward_travel = unsafe { (*ptr.add(pos)).dist_to_succ };
                let seq12 = Sequence::join2_with_travel(&singleton, &next, forward_travel);
                let seq21 = Sequence::join2_with_travel(
                    &next,
                    &singleton,
                    data.dm(unsafe { (*ptr.add(pos + 1)).id }, id),
                );
                unsafe {
                    (*ptr.add(pos)).t12 = Temporal::from(seq12);
                    (*ptr.add(pos)).t21 = Temporal::from(seq21);
                    (*ptr.add(pos)).reversed_travel = seq21.distance;
                }
                if pos + 2 < len && self.params.allow_swap3 {
                    let next2 = unsafe { (*ptr.add(pos + 2)).seq1() };
                    let travel = unsafe { (*ptr.add(pos + 1)).dist_to_succ };
                    unsafe {
                        (*ptr.add(pos)).t123 = Temporal::from(Sequence::join2_with_travel(&seq12, &next2, travel));
                    }
                }
            }
        }

        let end = unsafe { (*ptr.add(len - 1)).seq0_i() };
        r.load = end.load;
        r.tw = end.tw;
        r.distance = end.distance;
        r.cost = end.eval(data, &self.params);

        if self.batch_input_valid {
            let warp_price=if self.factored_bounds_valid && r.cost<=67_108_864 && r.tw>0 {self.params.penalty_tw as i64} else {0};
            if warp_price!=0 {
                let route_budget=r.cost-r.distance as i64;
            for pos in 1..len-1 {
                let pred=unsafe{&*ptr.add(pos)};
                let v=unsafe{&*ptr.add(pos+1)};let y=unsafe{&*ptr.add((pos+2).min(len-1))};
                let z=unsafe{&*ptr.add((pos+3).min(len-1))};let w=unsafe{&*ptr.add((pos+4).min(len-1))};
                let before=unsafe{&*ptr.add(pos-1)};
                let after_tail=if self.params.allow_swap3 {w} else {z};
                let before_tail=if self.params.allow_swap3 {z} else {y};
                let receiver=(pred.prefix.warp+after_tail.suffix.warp).min(32767);
                let donor=(before.prefix.warp+before_tail.suffix.warp).min(32767);
                let difference=donor-receiver;
                let packed_fourth=(w.id as u32)|((difference as i16 as u16 as u32)<<16);
                let kept=receiver;
                
                let cut_budget=route_budget-kept as i64*warp_price;
                unsafe{*self.batch_cuts.get_unchecked_mut(pred.id)=BatchCut([r.load,r.distance,cut_budget as i32,
                    v.id as i32,y.id as i32,z.id as i32,packed_fourth as i32,
                    pred.dist_to_succ,v.dist_to_succ,y.dist_to_succ,v.reversed_travel,(*ptr.add(pos-1)).id as i32,(*ptr.add(pos-1)).dist_to_succ,pred.bridge,pred.load,pred.removal_delta]);}
            }
            } else {
            for pos in 1..len-1 {
                let pred=unsafe{&*ptr.add(pos)};
                let v=unsafe{&*ptr.add(pos+1)};let y=unsafe{&*ptr.add((pos+2).min(len-1))};
                let z=unsafe{&*ptr.add((pos+3).min(len-1))};let w=unsafe{&*ptr.add((pos+4).min(len-1))};
                unsafe{*self.batch_cuts.get_unchecked_mut(pred.id)=BatchCut([r.load,r.distance,(if r.cost>67_108_864 {67_108_865} else {r.cost-r.distance as i64}) as i32,
                    v.id as i32,y.id as i32,z.id as i32,w.id as i32,
                    pred.dist_to_succ,v.dist_to_succ,y.dist_to_succ,v.reversed_travel,(*ptr.add(pos-1)).id as i32,(*ptr.add(pos-1)).dist_to_succ,pred.bridge,pred.load,pred.removal_delta]);}
            }
            }
        }

    }
    fn update_route(&mut self,rid:usize) {
        self.refresh_metrics_partial(rid);
        let data=self.data.as_ref();
        let r=unsafe{self.routes.get_unchecked_mut(rid)};
        let partial=r.refresh_to>0;
        let mapping_start=if partial {r.refresh_from.min(r.nodes.len())} else {0};
        let mapping_end=if partial {r.mapping_end.min(r.nodes.len())} else {r.nodes.len()};
        let update_membership=!partial || r.transfer;
        r.refresh_from=0;r.refresh_to=0;r.mapping_end=0;r.transfer=false;
        let nodes=&mut r.nodes;let len=nodes.len();let ptr=nodes.as_mut_ptr();
        self.insertion_clock=self.insertion_clock.checked_add(1).expect("insertion clock exhausted");
        r.insertion_version=self.insertion_clock;

        if self.masked_neighbors && update_membership {
            for pos in mapping_start.max(1)..mapping_end.min(len-1) {
                let id=unsafe{(*ptr.add(pos)).id};let old=copy_at(&self.membership_routes,id);
                if old!=rid {
                    let start=copy_at(&self.reverse_offsets,id);let end=copy_at(&self.reverse_offsets,id+1);
                    for k in start..end {
                        let packed=copy_at(&self.reverse_neighbors,k);
                        let c=(packed&65535) as usize;let bit=1u64<<((packed>>16)&63);
                        let dest=unsafe{self.route_neighbor_masks.get_unchecked_mut(c*self.mask_stride+rid)};
                        dest.before|=bit;
                        if old!=usize::MAX {
                            let previous=unsafe{self.route_neighbor_masks.get_unchecked_mut(c*self.mask_stride+old)};
                            previous.before&=!bit;
                        }
                    }
                    unsafe{*self.membership_routes.get_unchecked_mut(id)=rid;}
                }
            }
        }

        // Update route node mappings
        for pos in mapping_start..mapping_end {
            let node=unsafe{&*ptr.add(pos)};
            debug_assert!(node.id < self.node_locations.len());
            unsafe {
                *self.node_locations.get_unchecked_mut(node.id) = NodeLocation {
                    route_and_position: ((rid as u64)<<32) | pos as u64,
                };
            }
        }

        if mapping_end<len {
            // The original full loop leaves the shared depot entry at the
            // final depot of this route, even if no customer index changed.
            unsafe{*self.node_locations.get_unchecked_mut(0)=NodeLocation{route_and_position:((rid as u64)<<32)|(len-1) as u64};}
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
        if self.masked_neighbors {self.touch_recent_route(rid);}
    }

    #[cfg(target_arch="x86_64")]
    #[target_feature(enable="avx2")]
    unsafe fn intra_arc_possible<const PAIR:bool>(route:&Route,pos:usize,to_u:&[i32],from_u:&[i32],to_x:&[i32],from_x:&[i32],fixed:i32,fixed_rev:i32,limit:i32)->bool {
        use std::arch::x86_64::*;
        let mut indices=_mm256_setr_epi32(0,1,2,3,4,5,6,7);
        let excluded_first=_mm256_set1_epi32(pos as i32-1);
        let excluded_last=_mm256_set1_epi32(pos as i32+if PAIR {1} else {0});
        let threshold=_mm256_set1_epi32(limit);
        for k in (0..route.arc_travel.len()).step_by(8) {
            let a=_mm256_loadu_si256(route.arc_ids.as_ptr().add(k) as *const __m256i);
            let b=_mm256_loadu_si256(route.arc_ids.as_ptr().add(k+1) as *const __m256i);
            let removed=_mm256_loadu_si256(route.arc_travel.as_ptr().add(k) as *const __m256i);
            let incoming=_mm256_i32gather_epi32(to_u.as_ptr(),a,4);
            let outgoing=_mm256_i32gather_epi32(if PAIR {from_x.as_ptr()} else {from_u.as_ptr()},b,4);
            let mut delta=_mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(incoming,outgoing),_mm256_set1_epi32(fixed-16_384)),removed);
            if PAIR {
                let incoming=_mm256_i32gather_epi32(to_x.as_ptr(),a,4);
                let outgoing=_mm256_i32gather_epi32(from_u.as_ptr(),b,4);
                let reverse=_mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(incoming,outgoing),_mm256_set1_epi32(fixed_rev-16_384)),removed);
                delta=_mm256_min_epi32(delta,reverse);
            }
            let outside=_mm256_or_si256(_mm256_cmpgt_epi32(excluded_first,indices),_mm256_cmpgt_epi32(indices,excluded_last));
            let possible=_mm256_andnot_si256(_mm256_cmpgt_epi32(delta,threshold),outside);
            if _mm256_movemask_epi8(possible)!=0 {return true;}
            indices=_mm256_add_epi32(indices,_mm256_set1_epi32(8));
        }
        false
    }
    #[inline(always)]
    fn run_intra_route_relocate<const TABLE:bool>(&mut self, r1: usize, pos1: usize) -> i64 {
        let route = route_at(&self.routes, r1);
        let data = self.data.as_ref();
        let len = route.nodes.len();
        if len <= 3 {
            return NO_MOVE;
        } // no alternative insertion for single-client routes

        debug_assert!(pos1 > 0); // U is a client
        debug_assert!(self.routes[r1].nodes[pos1].id != 0); // U is a client

        let old_cost = route.cost;
        let max_acceptable_cost = old_cost + self.query_credit;
        let mut best_cost = i64::MAX;
        let mut best_pos: Option<usize> = None;
        let u_seq1 = route.node(pos1).seq1();
        let old_distance = route.distance as i64;
        let cap_pen =
            if TABLE {copy_at(&self.capacity_costs,route.load as usize)}
            else {((route.load-self.data.max_capacity).max(0) as i64)*self.params.penalty_capa as i64};
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
        #[cfg(target_arch="x86_64")]
        if self.batch_bounds_valid && max_distance_acceptable-old_distance>= -67_108_864 && max_distance_acceptable-old_distance<=67_108_864 {
            
            if !unsafe{Self::intra_arc_possible::<false>(route,pos1,distance_to_u,distance_from_u,distance_to_u,distance_from_u,fixed_delta,fixed_delta,(max_distance_acceptable-old_distance) as i32)} {
                
                return NO_MOVE;
            }
        }


        // Insert U before t in [1 .. pos1-1]
        if pos1 > 1 {
            let mut right_excl_u = route.node(pos1 + 1).seqi_n();
            for t in (1..pos1).rev() {
                let join_travel = if t + 1 == pos1 {
                    add_ab
                } else {
                    route.node(t).dist_to_succ
                };
                right_excl_u =
                    Sequence::join_tw_with_travel(&route.node(t).seq1(), &right_excl_u, join_travel);
                let c_id = route.node(t - 1).id;
                let d_id = route.node(t).id;
                let d_cu = copy_at(distance_to_u, c_id);
                let d_ud = copy_at(distance_from_u, d_id);
                let delta_distance = fixed_delta + d_cu + d_ud - route.node(t - 1).dist_to_succ;
                if old_distance + (delta_distance as i64) > max_distance_acceptable {
                    continue;
                }
                let left = route.node(t - 1).seq0_i();
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
                &route.node(pos1 - 1).seq0_i(),
                &route.node(pos1 + 1).seq1(),
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
                        &route.node(t).seqi_n(),
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
                        &route.node(t).seq1(),
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

    #[inline(always)]
    fn run_intra_route_oropt2<const TABLE:bool>(&mut self, r1: usize, pos1: usize) -> i64 {
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
        let max_acceptable_cost = old_cost + self.query_credit;
        let mut best_cost = i64::MAX;
        let mut best_pos: Option<usize> = None;
        let mut best_reversed = false;
        let pair_fwd = route.node(pos1).seq12();
        let pair_rev = route.node(pos1).seq21();
        let old_distance = route.distance as i64;
        let cap_pen =
            if TABLE {copy_at(&self.capacity_costs,route.load as usize)}
            else {((route.load-self.data.max_capacity).max(0) as i64)*self.params.penalty_capa as i64};
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
        #[cfg(target_arch="x86_64")]
        if self.batch_bounds_valid && max_distance_acceptable-old_distance>= -67_108_864 && max_distance_acceptable-old_distance<=67_108_864 {
            
            if !unsafe{Self::intra_arc_possible::<true>(route,pos1,distance_to_u,distance_from_u,distance_to_x,distance_from_x,fixed_delta_fwd,fixed_delta_rev,(max_distance_acceptable-old_distance) as i32)} {
                
                return NO_MOVE;
            }
        }


        // Insert (U,X) or (X,U) before t in [1 .. pos1-1]
        if pos1 > 1 {
            let mut right_excl_pair = route.node(pos1 + 2).seqi_n();
            for t in (1..pos1).rev() {
                let join_travel = if t + 1 == pos1 {
                    add_ab
                } else {
                    route.node(t).dist_to_succ
                };
                right_excl_pair = Sequence::join_tw_with_travel(
                    &route.node(t).seq1(),
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
                let left = route.node(t - 1).seq0_i();

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
                &route.node(pos1 - 1).seq0_i(),
                &route.node(pos1 + 2).seq1(),
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
                let right = route.node(t).seqi_n();

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
                        &route.node(t).seq1(),
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

    #[inline(always)]
    fn run_intra_route_swap<const TABLE:bool>(&mut self, r1: usize, pos1: usize) -> i64 {
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
        let max_acceptable_cost = old_cost + self.query_credit;
        let mut best_cost = i64::MAX;
        let mut best_pos: Option<usize> = None;
        let old_distance = route.distance as i64;
        let cap_pen =
            if TABLE {copy_at(&self.capacity_costs,route.load as usize)}
            else {((route.load-self.data.max_capacity).max(0) as i64)*self.params.penalty_capa as i64};
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
            let mut acc_mid = route.node(pos1 + 1).seq1();
            for middle in (pos1 + 2)..first_pos2 {
                acc_mid = Sequence::join_tw_with_travel(
                    &acc_mid,
                    &route.node(middle).seq1(),
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
                        &route.node(pos1 - 1).seq0_i(),
                        &route.node(pos2).seq1(),
                        &acc_mid,
                        &route.node(pos1).seq1(),
                        &route.node(pos2 + 1).seqi_n(),
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
                    &route.node(pos2).seq1(),
                    route.node(pos2 - 1).dist_to_succ,
                );
            }
        }

        if let Some(first_pos2) = first_left_potential {
            let mut acc_mid = route.node(pos1 - 1).seq1();
            for middle in ((first_pos2 + 1)..=(pos1 - 2)).rev() {
                acc_mid = Sequence::join_tw_with_travel(
                    &route.node(middle).seq1(),
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
                        &route.node(pos2 - 1).seq0_i(),
                        &route.node(pos1).seq1(),
                        &acc_mid,
                        &route.node(pos2).seq1(),
                        &route.node(pos1 + 1).seqi_n(),
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
                        &route.node(pos2).seq1(),
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

    #[inline(always)]
    fn run_2optstar<const TABLE:bool>(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize) -> i64 {
        debug_assert!(r1 != r2);
        debug_assert!(pos1 > 0 && pos1 < self.routes[r1].nodes.len());
        debug_assert!(pos2 > 0 && pos2 < self.routes[r2].nodes.len());

        let route1 = route_at(&self.routes, r1);
        let route2 = route_at(&self.routes, r2);
        let old_cost = route1.cost + route2.cost;
        let max_acceptable_cost = old_cost + self.query_credit;

        let left1 = &route1.node(pos1 - 1).seq0_i();
        let right1 = &route2.node(pos2).seqi_n();
        let left2 = &route2.node(pos2 - 1).seq0_i();
        let right2 = &route1.node(pos1).seqi_n();

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
        let penalties=if TABLE {copy_at(&self.capacity_costs,load1 as usize)+copy_at(&self.capacity_costs,load2 as usize)}
            else {((load1-max_cap).max(0) as i64)*pcap+((load2-max_cap).max(0) as i64)*pcap};
        let lb_cost=dist1 as i64+dist2 as i64+penalties;
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

    #[inline(always)]
    fn run_2opt<const TABLE:bool>(&mut self, r1: usize, pos1: usize) -> i64 {
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
            if TABLE {copy_at(&self.capacity_costs,route.load as usize)}
            else {((route.load-self.data.max_capacity).max(0) as i64)*self.params.penalty_capa as i64};
        let ptw = self.params.penalty_tw as i64;
        let max_distance_acceptable = max_acceptable_cost - cap_pen;
        let a_id = route.node(pos1 - 1).id;
        let u_id = route.node(pos1).id;
        let old_distance = route.distance as i64;
        let removed_au = route.node(pos1 - 1).dist_to_succ;
        let distance_from_a = data.distance_row(a_id);
        let distance_from_u = data.distance_row(u_id);

        let can_stop=self.gate_audit_valid && self.params.penalty_capa<=1_000_000 && self.params.penalty_tw<=1_000_000;
        let max_reverse_warp_cost=max_acceptable_cost-cap_pen-route.node(pos1-1).prefix.distance as i64;
        let mut mid_rev = route.node(pos1).seq21();
        let mut mid_rev_distance = mid_rev.distance;
        for pos2 in (pos1 + 1)..(len - 1) {
            if can_stop && mid_rev.tw as i64*ptw>max_reverse_warp_cost {break;}

            let v_id = route.node(pos2).id;
            let b_id = route.node(pos2 + 1).id;
            let d_av = copy_at(distance_from_a, v_id);
            let d_ub = copy_at(distance_from_u, b_id);
            // Materialize the exact reversed-segment distance for accepted
            // candidates, including asymmetric internal arcs.
            let left = route.node(pos1 - 1).seq0_i();
            let right = route.node(pos2 + 1).seqi_n();
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
                    mid_rev = Sequence::join_tw_with_travel(&next.seq1(), &mid_rev, reversed_travel);
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
                mid_rev = Sequence::join_tw_with_travel(&next.seq1(), &mid_rev, reversed_travel);
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

    #[inline(always)]
    fn run_inter_route(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize) -> i64 {
        if route_at(&self.routes,r1).nodes.len()==3 {
            return self.run_singleton_inter::<false>(r1,r2,pos2);
        }

        if self.capacity_costs.is_empty() {
            if self.params.allow_swap3 { self.run_inter_special::<true,false>(r1,pos1,r2,pos2) }
            else { self.run_inter_special::<false,false>(r1,pos1,r2,pos2) }
        } else {
            if self.params.allow_swap3 { self.run_inter_special::<true,true>(r1,pos1,r2,pos2) }
            else { self.run_inter_special::<false,true>(r1,pos1,r2,pos2) }
        }
    }


    #[inline(always)]
    fn run_inter_mode<const THREE: bool, const TABLE: bool>(&mut self,r1:usize,pos1:usize,r2:usize,pos2:usize)->i64 {
        if route_at(&self.routes,r1).nodes.len()==3 { self.run_singleton_inter::<TABLE>(r1,r2,pos2) }
        else { self.run_inter_special::<THREE,TABLE>(r1,pos1,r2,pos2) }
    }

    #[inline(always)]
    fn run_singleton_inter<const TABLE:bool>(&mut self, r1:usize, r2:usize, pos2:usize) -> i64 {
        let data=self.data.as_ref();
        let a=route_at(&self.routes,r1); let b=route_at(&self.routes,r2);
        let pa=a.node(0);let u=a.node(1);let an=a.node(2);
        let pb=b.node(pos2-1);let v=b.node(pos2);
        let old=a.cost+b.cost;let limit=old+self.query_credit;
        let total=a.distance+b.distance;
        let pcap=self.params.penalty_capa as i64;let ptw=self.params.penalty_tw as i64;
        let cap=data.max_capacity;
        let pen=|load:i32| if TABLE {copy_at(&self.capacity_costs,load as usize)} else {(load-cap).max(0) as i64*pcap};
        let du=data.dm(pb.id,u.id);let uv=data.dm(u.id,v.id);
        let bridge=data.dm(pa.id,an.id);
        let dist10=total-pa.dist_to_succ-u.dist_to_succ-pb.dist_to_succ+bridge+du+uv;
        let lb10=dist10 as i64+pen(a.load-u.load)+pen(b.load+u.load);
        let mut best=i64::MAX;let mut kind=0u8;
        if lb10<=limit {
            let tw=Sequence::tw2_with_travel(&pa.seq0_i(),&an.seqi_n(),bridge)
                +Sequence::tw3_with_travel(&pb.seq0_i(),&u.seq1(),&v.seqi_n(),du,uv);
            let cost=lb10+tw as i64*ptw;
            if cost<=limit {best=cost;kind=1;}
        }
        if v.id!=0 {
            if b.nodes.len()==3 {
                if self.query_credit>=0 && old<best {best=old;kind=2;}
            } else {
                let vn=b.node(pos2+1);
                let av=data.dm(pa.id,v.id);let va=data.dm(v.id,an.id);let un=data.dm(u.id,vn.id);
                let dist11=total-pa.dist_to_succ-u.dist_to_succ-pb.dist_to_succ-v.dist_to_succ+av+va+du+un;
                let lb11=dist11 as i64+pen(a.load-u.load+v.load)+pen(b.load-v.load+u.load);
                if lb11<=limit && lb11<best {
                    let tw=Sequence::tw3_with_travel(&pa.seq0_i(),&v.seq1(),&an.seqi_n(),av,va)
                        +Sequence::tw3_with_travel(&pb.seq0_i(),&u.seq1(),&vn.seqi_n(),du,un);
                    let cost=lb11+tw as i64*ptw;
                    if cost<=limit && cost<best {best=cost;kind=2;}
                }
            }
        }
        if kind==0 {return NO_MOVE;}
        self.last_plan=MovePlan::InterRoute{send1:1,send2:kind-1};best-old
    }


    #[inline(always)]
    fn make_inter_source<const THREE:bool>(&self,rid:usize,pos:usize)->InterSource {
        let r=route_at(&self.routes,rid);let data=self.data.as_ref();let last=r.nodes.len()-1;
        let up=r.node(pos-1);let u=r.node(pos);let x=r.node(pos+1);
        let xn=r.node((pos+2).min(last));let xnn=r.node((pos+3).min(last));
        let send2=u.load+x.load;let remove1=up.dist_to_succ+u.dist_to_succ;let remove2=remove1+x.dist_to_succ;
        InterSource{
            cost:r.cost,distance:r.distance,load:r.load,sent:[u.load,send2,send2+xn.load],removed:[remove1,remove2,remove2+xn.dist_to_succ],
            bridges:[u.bridge,if x.id!=0 {data.dm(up.id,xn.id)}else{0},if THREE && xn.id!=0 {data.dm(up.id,xnn.id)}else{0}],
            reverse_pair:if x.id!=0 {data.dm(x.id,u.id)}else{0},singleton:r.nodes.len()==3}
    }
    #[inline(always)]
    fn run_inter_source<const THREE:bool,const TABLE:bool>(&mut self,r1:usize,p1:usize,r2:usize,p2:usize,src:&InterSource)->i64 {
        let fast=if src.singleton {self.run_singleton_inter::<TABLE>(r1,r2,p2)}
            else {self.run_inter_source_impl::<THREE,TABLE>(r1,p1,r2,p2,src)};
        // CHECK_INTER_SOURCE
        fast
    }
    #[inline(always)]
    fn run_inter_source_impl<const THREE: bool, const TABLE: bool>(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize,src:&InterSource) -> i64 {
        if self.factored_bounds_valid {self.run_inter_source_impl_keyed::<THREE,TABLE,true>(r1,pos1,r2,pos2,src)}else{self.run_inter_source_impl_keyed::<THREE,TABLE,false>(r1,pos1,r2,pos2,src)}
    }
    #[inline(always)]
    fn run_inter_source_impl_keyed<const THREE: bool, const TABLE: bool,const PACKED:bool>(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize,src:&InterSource) -> i64 {
        let data = self.data.as_ref();
        let ru = route_at(&self.routes, r1);
        let rv = route_at(&self.routes, r2);
        let u = ru.node(pos1);
        let v = rv.node(pos2);
        let u_pred = ru.node(pos1 - 1);
        let v_pred = rv.node(pos2 - 1);
        let x = ru.node(pos1 + 1);
        let distance_from_u_pred = u_pred.distance_row(data);
        let distance_from_v_pred = v_pred.distance_row(data);
        let distance_from_u = u.distance_row(data);
        let distance_from_v = v.distance_row(data);
        let distance_from_x = x.distance_row(data);
        debug_assert!(
            u.id != 0,
            "Should always apply inter-route with a client as first node"
        );
        debug_assert!(r1 != r2, "Should not test inter-route move on same route");

        let old_total = src.cost + rv.cost;
        let max_acceptable_cost = old_total + self.query_credit;
        let group_limit=if self.factored_bounds_valid {max_acceptable_cost} else {i64::MAX};

        let pcap = self.params.penalty_capa as i64;
        let ptw = self.params.penalty_tw as i64;
        let max_cap = data.max_capacity;
        let route_total_dist = src.distance + rv.distance;
        let route1_load = src.load;
        let route2_load = rv.load;
        let mut best_key=i64::MAX;
        let mut best_cost = i64::MAX;
        let mut best_send1 = 0u8;
        let mut best_send2 = 0u8;

        macro_rules! cap_pen {
            ($load:expr) => {{
                if TABLE { copy_at(&self.capacity_costs, $load as usize) }
                else { ((($load) - max_cap).max(0) as i64) * pcap }
            }};
        }
        macro_rules! consider {
            ($order:expr, $send1:expr, $send2:expr, $lb:expr, $tw:expr) => {{
                let lower_bound = $lb;
                if lower_bound <= max_acceptable_cost {
                    let candidate = lower_bound + ($tw as i64) * ptw;
                    if PACKED {
                        // Existing safe-domain checks bound every route sum
                        // to i32 and each penalty to 1,000,000. Multiplying a
                        // nonnegative two-route cost by 16 therefore fits i64.
                        // The low four bits encode original evaluation order.
                        
                        best_key=best_key.min(candidate*16+$order);
                    } else {
                    if candidate <= max_acceptable_cost && candidate < best_cost {
                        best_cost = candidate;
                        best_send1 = $send1;
                        best_send2 = $send2;
                    }
                    }
                }
            }};
        }

        let send1_1_load = src.sent[0];
        let rem1_1 = src.removed[0];
        let rem2_0 = v_pred.dist_to_succ;
        let d_upred_x = src.bridges[0];
        let d_vpred_u = copy_at(distance_from_v_pred, u.id);
        let d_u_v = copy_at(distance_from_u, v.id);
        let cap_10 = cap_pen!(route1_load - send1_1_load) + cap_pen!(route2_load + send1_1_load);
        let lb10 =
            (route_total_dist - rem1_1 - rem2_0 + d_upred_x + d_vpred_u + d_u_v) as i64 + cap_10;
        if lb10 <= max_acceptable_cost {
            let tw10 = Sequence::tw2_with_travel(&u_pred.seq0_i(), &x.seqi_n(), d_upred_x)
                + Sequence::tw3_with_travel(&v_pred.seq0_i(), &u.seq1(), &v.seqi_n(), d_vpred_u, d_u_v);
            consider!(0,1, 0, lb10, tw10);
        }

        if v.id != 0 {
            let y = rv.node(pos2 + 1);
            let send2_1_load = v.seq1().load;
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
                    Sequence::tw3_with_travel(&u_pred.seq0_i(), &v.seq1(), &x.seqi_n(), d_upred_v, d_v_x)
                        + Sequence::tw3_with_travel(
                            &v_pred.seq0_i(),
                            &u.seq1(),
                            &y.seqi_n(),
                            d_vpred_u,
                            d_u_y,
                        );
                consider!(1,1, 1, lb11, tw11);
            }
        }

        if x.id != 0 {
            let x_next = ru.node(pos1 + 2);
            let distance_from_x_next = x_next.distance_row(data);
            let send1_2_load = src.sent[1];
            let rem1_2 = src.removed[1];
            let d_upred_xnext = src.bridges[1];
            let d_vpred_x = copy_at(distance_from_v_pred, x.id);
            let d_x_u = src.reverse_pair;
            let d_u_x = u.dist_to_succ;
            let d_x_v = copy_at(distance_from_x, v.id);
            let cap_20_30 =
                cap_pen!(route1_load - send1_2_load) + cap_pen!(route2_load + send1_2_load);
            let dist_base_20_30 = route_total_dist - rem1_2 - rem2_0;
            if (dist_base_20_30.wrapping_add(d_upred_xnext).wrapping_add(
                d_vpred_u.wrapping_add(d_u_x).wrapping_add(d_x_v).min(d_vpred_x.wrapping_add(d_x_u).wrapping_add(d_u_v))) as i64).wrapping_add(cap_20_30)<=group_limit {
let lb20 =
                (dist_base_20_30 + d_upred_xnext + d_vpred_u + d_u_x + d_x_v) as i64 + cap_20_30;
            let lb30 =
                (dist_base_20_30 + d_upred_xnext + d_vpred_x + d_x_u + d_u_v) as i64 + cap_20_30;
            if lb20 <= max_acceptable_cost || lb30 <= max_acceptable_cost {
                let route1_tw =
                    Sequence::tw2_with_travel(&u_pred.seq0_i(), &x_next.seqi_n(), d_upred_xnext);
                if lb20 <= max_acceptable_cost {
                    let tw20 = route1_tw
                        + Sequence::tw3_with_travel(
                            &v_pred.seq0_i(),
                            &u.seq12(),
                            &v.seqi_n(),
                            d_vpred_u,
                            d_x_v,
                        );
                    consider!(2,2, 0, lb20, tw20);
                }
                if lb30 <= max_acceptable_cost {
                    let tw30 = route1_tw
                        + Sequence::tw3_with_travel(
                            &v_pred.seq0_i(),
                            &u.seq21(),
                            &v.seqi_n(),
                            d_vpred_x,
                            d_u_v,
                        );
                    consider!(3,3, 0, lb30, tw30);
                }
            }
            }

            if v.id != 0 {
                let y = rv.node(pos2 + 1);
                let send2_1_load = v.seq1().load;
                let rem2_1 = v_pred.dist_to_succ + v.dist_to_succ;
                let d_upred_v = copy_at(distance_from_u_pred, v.id);
                let d_v_xnext = copy_at(distance_from_v, x_next.id);
                let d_x_y = copy_at(distance_from_x, y.id);
                let d_u_y = copy_at(distance_from_u, y.id);
                let cap_21_31 = cap_pen!(route1_load - send1_2_load + send2_1_load)
                    + cap_pen!(route2_load - send2_1_load + send1_2_load);
                let dist_base_21_31 = route_total_dist - rem1_2 - rem2_1;
                let common_left = dist_base_21_31 + d_upred_v + d_v_xnext;
                if (common_left.wrapping_add(
                d_vpred_u.wrapping_add(d_u_x).wrapping_add(d_x_y).min(d_vpred_x.wrapping_add(d_x_u).wrapping_add(d_u_y))) as i64).wrapping_add(cap_21_31)<=group_limit {
let lb21 = (common_left + d_vpred_u + d_u_x + d_x_y) as i64 + cap_21_31;
                let lb31 = (common_left + d_vpred_x + d_x_u + d_u_y) as i64 + cap_21_31;
                if lb21 <= max_acceptable_cost || lb31 <= max_acceptable_cost {
                    let route1_tw = Sequence::tw3_with_travel(
                        &u_pred.seq0_i(),
                        &v.seq1(),
                        &x_next.seqi_n(),
                        d_upred_v,
                        d_v_xnext,
                    );
                    if lb21 <= max_acceptable_cost {
                        let tw21 = route1_tw
                            + Sequence::tw3_with_travel(
                                &v_pred.seq0_i(),
                                &u.seq12(),
                                &y.seqi_n(),
                                d_vpred_u,
                                d_x_y,
                            );
                        consider!(4,2, 1, lb21, tw21);
                    }
                    if lb31 <= max_acceptable_cost {
                        let tw31 = route1_tw
                            + Sequence::tw3_with_travel(
                                &v_pred.seq0_i(),
                                &u.seq21(),
                                &y.seqi_n(),
                                d_vpred_x,
                                d_u_y,
                            );
                        consider!(5,3, 1, lb31, tw31);
                    }
                }
            }

                if y.id != 0 {
                    let y_next = rv.node(pos2 + 2);
                    let distance_from_y = y.distance_row(data);
                    let send2_2_load = v.seq1().load + y.seq1().load;
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
                    if (dist_base.wrapping_add(left_fwd_dist.min(left_rev_dist)).wrapping_add(right_fwd_dist.min(right_rev_dist)) as i64).wrapping_add(cap_22_33)<=group_limit {
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
                            &u_pred.seq0_i(),
                            &v.seq12(),
                            &x_next.seqi_n(),
                            d_upred_v,
                            d_y_xnext,
                        ))
                    } else {
                        None
                    };
                    let left_rev = if can23 || can33 {
                        Some(Sequence::tw3_with_travel(
                            &u_pred.seq0_i(),
                            &v.seq21(),
                            &x_next.seqi_n(),
                            d_upred_y,
                            d_v_xnext,
                        ))
                    } else {
                        None
                    };
                    let right_fwd = if can22 || can23 {
                        Some(Sequence::tw3_with_travel(
                            &v_pred.seq0_i(),
                            &u.seq12(),
                            &y_next.seqi_n(),
                            d_vpred_u,
                            d_x_ynext,
                        ))
                    } else {
                        None
                    };
                    let right_rev = if can32 || can33 {
                        Some(Sequence::tw3_with_travel(
                            &v_pred.seq0_i(),
                            &u.seq21(),
                            &y_next.seqi_n(),
                            d_vpred_x,
                            d_u_ynext,
                        ))
                    } else {
                        None
                    };

                    if can22 {
                        consider!(6,2, 2, lb22, left_fwd.unwrap() + right_fwd.unwrap());
                    }
                    if can32 {
                        consider!(7,3, 2, lb32, left_fwd.unwrap() + right_rev.unwrap());
                    }
                    if can23 {
                        consider!(8,2, 3, lb23, left_rev.unwrap() + right_fwd.unwrap());
                    }
                    if can33 {
                        consider!(9,3, 3, lb33, left_rev.unwrap() + right_rev.unwrap());
                    }
                    }
                }
            }

            if THREE && x_next.id != 0 {
                let x2_next = ru.node(pos1 + 3);
                let send1_3_load = src.sent[2];
                let rem1_3 = src.removed[2];
                let d_upred_x2next = src.bridges[2];
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
                        Sequence::tw2_with_travel(&u_pred.seq0_i(), &x2_next.seqi_n(), d_upred_x2next)
                            + Sequence::tw3_with_travel(
                                &v_pred.seq0_i(),
                                &u.seq123(),
                                &v.seqi_n(),
                                d_vpred_u,
                                d_xnext_v,
                            );
                    consider!(10,4, 0, lb40, tw40);
                }

                if v.id != 0 {
                    let y = rv.node(pos2 + 1);
                    let send2_1_load = v.seq1().load;
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
                            &u_pred.seq0_i(),
                            &v.seq1(),
                            &x2_next.seqi_n(),
                            d_upred_v,
                            d_v_x2next,
                        ) + Sequence::tw3_with_travel(
                            &v_pred.seq0_i(),
                            &u.seq123(),
                            &y.seqi_n(),
                            d_vpred_u,
                            d_xnext_y,
                        );
                        consider!(11,4, 1, lb41, tw41);
                    }

                    if y.id != 0 {
                        let y_next = rv.node(pos2 + 2);
                        let distance_from_y = y.distance_row(data);
                        let send2_2_load = v.seq1().load + y.seq1().load;
                        let rem2_2 = v_pred.dist_to_succ + v.dist_to_succ + y.dist_to_succ;
                        let d_upred_y = copy_at(distance_from_u_pred, y.id);
                        let d_y_v = copy_at(distance_from_y, v.id);
                        let d_y_x2next = copy_at(distance_from_y, x2_next.id);
                        let d_xnext_ynext = copy_at(distance_from_x_next, y_next.id);
                        let cap_42_43 = cap_pen!(route1_load - send1_3_load + send2_2_load)
                            + cap_pen!(route2_load - send2_2_load + send1_3_load);
                        let dist_base = route_total_dist - rem1_3 - rem2_2;
                        let common_right = d_vpred_u + d_u_xnext + d_x_xnext + d_xnext_ynext;
                        if (dist_base.wrapping_add(common_right).wrapping_add(
                d_upred_v.wrapping_add(v.dist_to_succ).wrapping_add(d_y_x2next).min(d_upred_y.wrapping_add(d_y_v).wrapping_add(d_v_x2next))) as i64).wrapping_add(cap_42_43)<=group_limit {
let lb42 =
                            (dist_base + d_upred_v + v.dist_to_succ + d_y_x2next + common_right)
                                as i64
                                + cap_42_43;
                        let lb43 = (dist_base + d_upred_y + d_y_v + d_v_x2next + common_right)
                            as i64
                            + cap_42_43;
                        if lb42 <= max_acceptable_cost || lb43 <= max_acceptable_cost {
                            let right_tw = Sequence::tw3_with_travel(
                                &v_pred.seq0_i(),
                                &u.seq123(),
                                &y_next.seqi_n(),
                                d_vpred_u,
                                d_xnext_ynext,
                            );
                            if lb42 <= max_acceptable_cost {
                                let tw42 = Sequence::tw3_with_travel(
                                    &u_pred.seq0_i(),
                                    &v.seq12(),
                                    &x2_next.seqi_n(),
                                    d_upred_v,
                                    d_y_x2next,
                                ) + right_tw;
                                consider!(12,4, 2, lb42, tw42);
                            }
                            if lb43 <= max_acceptable_cost {
                                let tw43 = Sequence::tw3_with_travel(
                                    &u_pred.seq0_i(),
                                    &v.seq21(),
                                    &x2_next.seqi_n(),
                                    d_upred_y,
                                    d_v_x2next,
                                ) + right_tw;
                                consider!(13,4, 3, lb43, tw43);
                            }
                        }
            }

                        if y_next.id != 0 {
                            let y2_next = rv.node(pos2 + 3);
                            let send2_3_load = v.seq1().load + y.seq1().load + y_next.seq1().load;
                            let rem2_3 = v_pred.dist_to_succ
                                + v.dist_to_succ
                                + y.dist_to_succ
                                + y_next.dist_to_succ;
                            let d_ynext_x2next = copy_at(y_next.distance_row(data), x2_next.id);
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
                                    &u_pred.seq0_i(),
                                    &v.seq123(),
                                    &x2_next.seqi_n(),
                                    d_upred_v,
                                    d_ynext_x2next,
                                ) + Sequence::tw3_with_travel(
                                    &v_pred.seq0_i(),
                                    &u.seq123(),
                                    &y2_next.seqi_n(),
                                    d_vpred_u,
                                    d_xnext_y2next,
                                );
                                consider!(14,4, 4, lb44, tw44);
                            }
                        }
                    }
                }
            }
        }

        if PACKED {
            if best_key==i64::MAX {return NO_MOVE;}
            let cost=best_key>>4;
            if cost>max_acceptable_cost {return NO_MOVE;}
            const PLANS:[(u8,u8);15]=[(1,0),(1,1),(2,0),(3,0),(2,1),(3,1),(2,2),(3,2),(2,3),(3,3),(4,0),(4,1),(4,2),(4,3),(4,4)];
            let (send1,send2)=PLANS[(best_key&15) as usize];
            self.last_plan=MovePlan::InterRoute{send1,send2};return cost-old_total;
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
    fn run_inter_special<const THREE: bool, const TABLE: bool>(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize) -> i64 {
        if self.factored_bounds_valid {self.run_inter_special_keyed::<THREE,TABLE,true>(r1,pos1,r2,pos2)}else{self.run_inter_special_keyed::<THREE,TABLE,false>(r1,pos1,r2,pos2)}
    }
    #[inline(always)]
    fn run_inter_special_keyed<const THREE: bool, const TABLE: bool,const PACKED:bool>(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize) -> i64 {
        let data = self.data.as_ref();
        let ru = route_at(&self.routes, r1);
        let rv = route_at(&self.routes, r2);
        let u = ru.node(pos1);
        let v = rv.node(pos2);
        let u_pred = ru.node(pos1 - 1);
        let v_pred = rv.node(pos2 - 1);
        let x = ru.node(pos1 + 1);
        let distance_from_u_pred = u_pred.distance_row(data);
        let distance_from_v_pred = v_pred.distance_row(data);
        let distance_from_u = u.distance_row(data);
        let distance_from_v = v.distance_row(data);
        let distance_from_x = x.distance_row(data);
        debug_assert!(
            u.id != 0,
            "Should always apply inter-route with a client as first node"
        );
        debug_assert!(r1 != r2, "Should not test inter-route move on same route");

        let old_total = ru.cost + rv.cost;
        let max_acceptable_cost = old_total + self.query_credit;
        let group_limit=if self.factored_bounds_valid {max_acceptable_cost} else {i64::MAX};

        let pcap = self.params.penalty_capa as i64;
        let ptw = self.params.penalty_tw as i64;
        let max_cap = data.max_capacity;
        let route_total_dist = ru.distance + rv.distance;
        let route1_load = ru.load;
        let route2_load = rv.load;
        let mut best_key=i64::MAX;
        let mut best_cost = i64::MAX;
        let mut best_send1 = 0u8;
        let mut best_send2 = 0u8;

        macro_rules! cap_pen {
            ($load:expr) => {{
                if TABLE { copy_at(&self.capacity_costs, $load as usize) }
                else { ((($load) - max_cap).max(0) as i64) * pcap }
            }};
        }
        macro_rules! consider {
            ($order:expr, $send1:expr, $send2:expr, $lb:expr, $tw:expr) => {{
                let lower_bound = $lb;
                if lower_bound <= max_acceptable_cost {
                    let candidate = lower_bound + ($tw as i64) * ptw;
                    if PACKED {
                        // Existing safe-domain checks bound every route sum
                        // to i32 and each penalty to 1,000,000. Multiplying a
                        // nonnegative two-route cost by 16 therefore fits i64.
                        // The low four bits encode original evaluation order.
                        
                        best_key=best_key.min(candidate*16+$order);
                    } else {
                    if candidate <= max_acceptable_cost && candidate < best_cost {
                        best_cost = candidate;
                        best_send1 = $send1;
                        best_send2 = $send2;
                    }
                    }
                }
            }};
        }

        let send1_1_load = u.seq1().load;
        let rem1_1 = u_pred.dist_to_succ + u.dist_to_succ;
        let rem2_0 = v_pred.dist_to_succ;
        let d_upred_x = u.bridge;
        let d_vpred_u = copy_at(distance_from_v_pred, u.id);
        let d_u_v = copy_at(distance_from_u, v.id);
        let cap_10 = cap_pen!(route1_load - send1_1_load) + cap_pen!(route2_load + send1_1_load);
        let lb10 =
            (route_total_dist - rem1_1 - rem2_0 + d_upred_x + d_vpred_u + d_u_v) as i64 + cap_10;
        if lb10 <= max_acceptable_cost {
            let tw10 = Sequence::tw2_with_travel(&u_pred.seq0_i(), &x.seqi_n(), d_upred_x)
                + Sequence::tw3_with_travel(&v_pred.seq0_i(), &u.seq1(), &v.seqi_n(), d_vpred_u, d_u_v);
            consider!(0,1, 0, lb10, tw10);
        }

        if v.id != 0 {
            let y = rv.node(pos2 + 1);
            let send2_1_load = v.seq1().load;
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
                    Sequence::tw3_with_travel(&u_pred.seq0_i(), &v.seq1(), &x.seqi_n(), d_upred_v, d_v_x)
                        + Sequence::tw3_with_travel(
                            &v_pred.seq0_i(),
                            &u.seq1(),
                            &y.seqi_n(),
                            d_vpred_u,
                            d_u_y,
                        );
                consider!(1,1, 1, lb11, tw11);
            }
        }

        if x.id != 0 {
            let x_next = ru.node(pos1 + 2);
            let distance_from_x_next = x_next.distance_row(data);
            let send1_2_load = u.seq1().load + x.seq1().load;
            let rem1_2 = u_pred.dist_to_succ + u.dist_to_succ + x.dist_to_succ;
            let d_upred_xnext = copy_at(distance_from_u_pred, x_next.id);
            let d_vpred_x = copy_at(distance_from_v_pred, x.id);
            let d_x_u = copy_at(distance_from_x, u.id);
            let d_u_x = u.dist_to_succ;
            let d_x_v = copy_at(distance_from_x, v.id);
            let cap_20_30 =
                cap_pen!(route1_load - send1_2_load) + cap_pen!(route2_load + send1_2_load);
            let dist_base_20_30 = route_total_dist - rem1_2 - rem2_0;
            if (dist_base_20_30.wrapping_add(d_upred_xnext).wrapping_add(
                d_vpred_u.wrapping_add(d_u_x).wrapping_add(d_x_v).min(d_vpred_x.wrapping_add(d_x_u).wrapping_add(d_u_v))) as i64).wrapping_add(cap_20_30)<=group_limit {
let lb20 =
                (dist_base_20_30 + d_upred_xnext + d_vpred_u + d_u_x + d_x_v) as i64 + cap_20_30;
            let lb30 =
                (dist_base_20_30 + d_upred_xnext + d_vpred_x + d_x_u + d_u_v) as i64 + cap_20_30;
            if lb20 <= max_acceptable_cost || lb30 <= max_acceptable_cost {
                let route1_tw =
                    Sequence::tw2_with_travel(&u_pred.seq0_i(), &x_next.seqi_n(), d_upred_xnext);
                if lb20 <= max_acceptable_cost {
                    let tw20 = route1_tw
                        + Sequence::tw3_with_travel(
                            &v_pred.seq0_i(),
                            &u.seq12(),
                            &v.seqi_n(),
                            d_vpred_u,
                            d_x_v,
                        );
                    consider!(2,2, 0, lb20, tw20);
                }
                if lb30 <= max_acceptable_cost {
                    let tw30 = route1_tw
                        + Sequence::tw3_with_travel(
                            &v_pred.seq0_i(),
                            &u.seq21(),
                            &v.seqi_n(),
                            d_vpred_x,
                            d_u_v,
                        );
                    consider!(3,3, 0, lb30, tw30);
                }
            }
            }

            if v.id != 0 {
                let y = rv.node(pos2 + 1);
                let send2_1_load = v.seq1().load;
                let rem2_1 = v_pred.dist_to_succ + v.dist_to_succ;
                let d_upred_v = copy_at(distance_from_u_pred, v.id);
                let d_v_xnext = copy_at(distance_from_v, x_next.id);
                let d_x_y = copy_at(distance_from_x, y.id);
                let d_u_y = copy_at(distance_from_u, y.id);
                let cap_21_31 = cap_pen!(route1_load - send1_2_load + send2_1_load)
                    + cap_pen!(route2_load - send2_1_load + send1_2_load);
                let dist_base_21_31 = route_total_dist - rem1_2 - rem2_1;
                let common_left = dist_base_21_31 + d_upred_v + d_v_xnext;
                if (common_left.wrapping_add(
                d_vpred_u.wrapping_add(d_u_x).wrapping_add(d_x_y).min(d_vpred_x.wrapping_add(d_x_u).wrapping_add(d_u_y))) as i64).wrapping_add(cap_21_31)<=group_limit {
let lb21 = (common_left + d_vpred_u + d_u_x + d_x_y) as i64 + cap_21_31;
                let lb31 = (common_left + d_vpred_x + d_x_u + d_u_y) as i64 + cap_21_31;
                if lb21 <= max_acceptable_cost || lb31 <= max_acceptable_cost {
                    let route1_tw = Sequence::tw3_with_travel(
                        &u_pred.seq0_i(),
                        &v.seq1(),
                        &x_next.seqi_n(),
                        d_upred_v,
                        d_v_xnext,
                    );
                    if lb21 <= max_acceptable_cost {
                        let tw21 = route1_tw
                            + Sequence::tw3_with_travel(
                                &v_pred.seq0_i(),
                                &u.seq12(),
                                &y.seqi_n(),
                                d_vpred_u,
                                d_x_y,
                            );
                        consider!(4,2, 1, lb21, tw21);
                    }
                    if lb31 <= max_acceptable_cost {
                        let tw31 = route1_tw
                            + Sequence::tw3_with_travel(
                                &v_pred.seq0_i(),
                                &u.seq21(),
                                &y.seqi_n(),
                                d_vpred_x,
                                d_u_y,
                            );
                        consider!(5,3, 1, lb31, tw31);
                    }
                }
            }

                if y.id != 0 {
                    let y_next = rv.node(pos2 + 2);
                    let distance_from_y = y.distance_row(data);
                    let send2_2_load = v.seq1().load + y.seq1().load;
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
                    if (dist_base.wrapping_add(left_fwd_dist.min(left_rev_dist)).wrapping_add(right_fwd_dist.min(right_rev_dist)) as i64).wrapping_add(cap_22_33)<=group_limit {
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
                            &u_pred.seq0_i(),
                            &v.seq12(),
                            &x_next.seqi_n(),
                            d_upred_v,
                            d_y_xnext,
                        ))
                    } else {
                        None
                    };
                    let left_rev = if can23 || can33 {
                        Some(Sequence::tw3_with_travel(
                            &u_pred.seq0_i(),
                            &v.seq21(),
                            &x_next.seqi_n(),
                            d_upred_y,
                            d_v_xnext,
                        ))
                    } else {
                        None
                    };
                    let right_fwd = if can22 || can23 {
                        Some(Sequence::tw3_with_travel(
                            &v_pred.seq0_i(),
                            &u.seq12(),
                            &y_next.seqi_n(),
                            d_vpred_u,
                            d_x_ynext,
                        ))
                    } else {
                        None
                    };
                    let right_rev = if can32 || can33 {
                        Some(Sequence::tw3_with_travel(
                            &v_pred.seq0_i(),
                            &u.seq21(),
                            &y_next.seqi_n(),
                            d_vpred_x,
                            d_u_ynext,
                        ))
                    } else {
                        None
                    };

                    if can22 {
                        consider!(6,2, 2, lb22, left_fwd.unwrap() + right_fwd.unwrap());
                    }
                    if can32 {
                        consider!(7,3, 2, lb32, left_fwd.unwrap() + right_rev.unwrap());
                    }
                    if can23 {
                        consider!(8,2, 3, lb23, left_rev.unwrap() + right_fwd.unwrap());
                    }
                    if can33 {
                        consider!(9,3, 3, lb33, left_rev.unwrap() + right_rev.unwrap());
                    }
                    }
                }
            }

            if THREE && x_next.id != 0 {
                let x2_next = ru.node(pos1 + 3);
                let send1_3_load = u.seq1().load + x.seq1().load + x_next.seq1().load;
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
                        Sequence::tw2_with_travel(&u_pred.seq0_i(), &x2_next.seqi_n(), d_upred_x2next)
                            + Sequence::tw3_with_travel(
                                &v_pred.seq0_i(),
                                &u.seq123(),
                                &v.seqi_n(),
                                d_vpred_u,
                                d_xnext_v,
                            );
                    consider!(10,4, 0, lb40, tw40);
                }

                if v.id != 0 {
                    let y = rv.node(pos2 + 1);
                    let send2_1_load = v.seq1().load;
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
                            &u_pred.seq0_i(),
                            &v.seq1(),
                            &x2_next.seqi_n(),
                            d_upred_v,
                            d_v_x2next,
                        ) + Sequence::tw3_with_travel(
                            &v_pred.seq0_i(),
                            &u.seq123(),
                            &y.seqi_n(),
                            d_vpred_u,
                            d_xnext_y,
                        );
                        consider!(11,4, 1, lb41, tw41);
                    }

                    if y.id != 0 {
                        let y_next = rv.node(pos2 + 2);
                        let distance_from_y = y.distance_row(data);
                        let send2_2_load = v.seq1().load + y.seq1().load;
                        let rem2_2 = v_pred.dist_to_succ + v.dist_to_succ + y.dist_to_succ;
                        let d_upred_y = copy_at(distance_from_u_pred, y.id);
                        let d_y_v = copy_at(distance_from_y, v.id);
                        let d_y_x2next = copy_at(distance_from_y, x2_next.id);
                        let d_xnext_ynext = copy_at(distance_from_x_next, y_next.id);
                        let cap_42_43 = cap_pen!(route1_load - send1_3_load + send2_2_load)
                            + cap_pen!(route2_load - send2_2_load + send1_3_load);
                        let dist_base = route_total_dist - rem1_3 - rem2_2;
                        let common_right = d_vpred_u + d_u_xnext + d_x_xnext + d_xnext_ynext;
                        if (dist_base.wrapping_add(common_right).wrapping_add(
                d_upred_v.wrapping_add(v.dist_to_succ).wrapping_add(d_y_x2next).min(d_upred_y.wrapping_add(d_y_v).wrapping_add(d_v_x2next))) as i64).wrapping_add(cap_42_43)<=group_limit {
let lb42 =
                            (dist_base + d_upred_v + v.dist_to_succ + d_y_x2next + common_right)
                                as i64
                                + cap_42_43;
                        let lb43 = (dist_base + d_upred_y + d_y_v + d_v_x2next + common_right)
                            as i64
                            + cap_42_43;
                        if lb42 <= max_acceptable_cost || lb43 <= max_acceptable_cost {
                            let right_tw = Sequence::tw3_with_travel(
                                &v_pred.seq0_i(),
                                &u.seq123(),
                                &y_next.seqi_n(),
                                d_vpred_u,
                                d_xnext_ynext,
                            );
                            if lb42 <= max_acceptable_cost {
                                let tw42 = Sequence::tw3_with_travel(
                                    &u_pred.seq0_i(),
                                    &v.seq12(),
                                    &x2_next.seqi_n(),
                                    d_upred_v,
                                    d_y_x2next,
                                ) + right_tw;
                                consider!(12,4, 2, lb42, tw42);
                            }
                            if lb43 <= max_acceptable_cost {
                                let tw43 = Sequence::tw3_with_travel(
                                    &u_pred.seq0_i(),
                                    &v.seq21(),
                                    &x2_next.seqi_n(),
                                    d_upred_y,
                                    d_v_x2next,
                                ) + right_tw;
                                consider!(13,4, 3, lb43, tw43);
                            }
                        }
            }

                        if y_next.id != 0 {
                            let y2_next = rv.node(pos2 + 3);
                            let send2_3_load = v.seq1().load + y.seq1().load + y_next.seq1().load;
                            let rem2_3 = v_pred.dist_to_succ
                                + v.dist_to_succ
                                + y.dist_to_succ
                                + y_next.dist_to_succ;
                            let d_ynext_x2next = copy_at(y_next.distance_row(data), x2_next.id);
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
                                    &u_pred.seq0_i(),
                                    &v.seq123(),
                                    &x2_next.seqi_n(),
                                    d_upred_v,
                                    d_ynext_x2next,
                                ) + Sequence::tw3_with_travel(
                                    &v_pred.seq0_i(),
                                    &u.seq123(),
                                    &y2_next.seqi_n(),
                                    d_vpred_u,
                                    d_xnext_y2next,
                                );
                                consider!(14,4, 4, lb44, tw44);
                            }
                        }
                    }
                }
            }
        }

        if PACKED {
            if best_key==i64::MAX {return NO_MOVE;}
            let cost=best_key>>4;
            if cost>max_acceptable_cost {return NO_MOVE;}
            const PLANS:[(u8,u8);15]=[(1,0),(1,1),(2,0),(3,0),(2,1),(3,1),(2,2),(3,2),(2,3),(3,3),(4,0),(4,1),(4,2),(4,3),(4,4)];
            let (send1,send2)=PLANS[(best_key&15) as usize];
            self.last_plan=MovePlan::InterRoute{send1,send2};return cost-old_total;
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


    #[cfg(target_arch="x86_64")]
    #[target_feature(enable="avx2")]

    unsafe fn vector_arc_min(ids:&[i32],travel:&[i32],from:*const i32,to:*const i32)->i32 {
        use std::arch::x86_64::*;
        macro_rules! block {($k:expr)=>{{
            let a=_mm256_loadu_si256(ids.as_ptr().add($k) as *const __m256i);
            let b=_mm256_loadu_si256(ids.as_ptr().add($k+1) as *const __m256i);
            let ab=_mm256_loadu_si256(travel.as_ptr().add($k) as *const __m256i);
            let av=_mm256_i32gather_epi32::<4>(to,a);let vb=_mm256_i32gather_epi32::<4>(from,b);
            _mm256_add_epi32(_mm256_add_epi32(av,vb),ab)
        }}}
        // Route arrays are padded to a multiple of eight, with one extra ID.
        // The specialized arms read exactly the same blocks as the loop.
        let best=match travel.len() {
            8=>block!(0),
            16=>_mm256_min_epi32(block!(0),block!(8)),
            24=>_mm256_min_epi32(_mm256_min_epi32(block!(0),block!(8)),block!(16)),
            32=>_mm256_min_epi32(_mm256_min_epi32(block!(0),block!(8)),_mm256_min_epi32(block!(16),block!(24))),
            _=>{let mut best=_mm256_set1_epi32(i32::MAX);let mut k=0;
                while k<travel.len(){best=_mm256_min_epi32(best,block!(k));k+=8;}best},
        };
        let mut half=_mm_min_epi32(_mm256_castsi256_si128(best),_mm256_extracti128_si256::<1>(best));
        half=_mm_min_epi32(half,_mm_shuffle_epi32::<0x4e>(half));
        half=_mm_min_epi32(half,_mm_shuffle_epi32::<0xb1>(half));
        let result=_mm_cvtsi128_si32(half)-16_384;
        
        result
    }

    #[inline(always)]
    fn route_insertion_lower_bound(data:&Problem, route:&Route, entry:&mut InsertionLowerBound,
                                   from:&[i32],to:&[i32])->i32 {
        if entry.version==route.insertion_version { return entry.value; }
        #[cfg(target_arch="x86_64")]
        if !route.arc_travel.is_empty() {
            let best=unsafe{Self::vector_arc_min(&route.arc_ids,&route.arc_travel,from.as_ptr(),to.as_ptr())};
            *entry=InsertionLowerBound{version:route.insertion_version,value:best};return best;
        }
        let mut best=i32::MAX;
        for t in 1..route.nodes.len() {
            let pred=route.node(t-1);let next=route.node(t);
            best=best.min(copy_at(to,pred.id)+copy_at(from,next.id)-pred.dist_to_succ);
        }
        *entry=InsertionLowerBound{version:route.insertion_version,value:best};best
    }

    #[inline(never)]
    fn audit_insertion_gate(nodes:&[Node],pos:usize,from:&[i32],to:&[i32],bridge:i32,threshold:i64)->bool {
        let pred=node_at(nodes,pos-1);let next=node_at(nodes,pos+1);
        if (copy_at(to,pred.id)+copy_at(from,next.id)-bridge) as i64<=threshold {return true;}
        for t in 1..pos {
            let a=node_at(nodes,t-1);let b=node_at(nodes,t);
            if (copy_at(to,a.id)+copy_at(from,b.id)-a.dist_to_succ) as i64<=threshold {return true;}
        }
        for t in pos+2..nodes.len() {
            let a=node_at(nodes,t-1);let b=node_at(nodes,t);
            if (copy_at(to,a.id)+copy_at(from,b.id)-a.dist_to_succ) as i64<=threshold {return true;}
        }
        false
    }

    #[inline(never)]
    fn prepare_removal_cache(route:&Route,pos:usize,cache:&mut RemovalCache) {
        let nodes=&route.nodes;let len=nodes.len();let bridge=node_at(nodes,pos).bridge;
        cache.entries.clear();cache.entries.reserve(len-2);
        // Exactly len-2 records are written into the reserved allocation.
        // No route or allocation mutation occurs while the output pointer is live.
        let out=cache.entries.as_mut_ptr();let mut k=0;
        let mut right=node_at(nodes,pos+1).seqi_n();
        for t in (1..=pos).rev() {
            let left=node_at(nodes,t-1).seq0_i();
            unsafe{out.add(k).write(InsertionBoundary::from(left,right,t));}k+=1;
            if t>1 {
                let travel=if t==pos {bridge} else {node_at(nodes,t-1).dist_to_succ};
                right=Sequence::join2_with_travel(&node_at(nodes,t-1).seq1(),&right,travel);
            }
        }
        let mut left=Sequence::join2_with_travel(&node_at(nodes,pos-1).seq0_i(),&node_at(nodes,pos+1).seq1(),bridge);
        for t in pos+2..len {
            let right=node_at(nodes,t).seqi_n();
            unsafe{out.add(k).write(InsertionBoundary::from(left,right,t));}k+=1;
            if t+1<len {left=Sequence::join2_with_travel(&left,&node_at(nodes,t).seq1(),node_at(nodes,t-1).dist_to_succ);}
        }
        debug_assert_eq!(k,len-2);unsafe{cache.entries.set_len(k);}
        cache.version=route.insertion_version;
    }
    #[inline(always)]
    fn make_swap_source(&self,rid:usize,pos:usize)->SwapSource {
        let r=route_at(&self.routes,rid);let u=r.node(pos);
        SwapSource{pred:r.node(pos-1).id,next:r.node(pos+1).id,cost:r.cost,max_cost:r.cost+self.move_credit,id:u.id,removed_distance:r.distance+u.removal_delta,
            remaining_load:r.load-u.load,load:u.load,bridge:u.bridge,row_start:u.row_start,singleton:r.nodes.len()==3}
    }
    #[inline(always)]
    fn run_swapstar_source<const TABLE:bool>(&mut self,r1:usize,p1:usize,r2:usize,p2:usize,src:&SwapSource)->i64 {
        self.run_swapstar_source_fast::<TABLE>(r1,p1,r2,p2,src)
    }
    #[inline(always)]
    fn run_swapstar_source_fast<const TABLE:bool>(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize,src:&SwapSource) -> i64 {
        if src.singleton && route_at(&self.routes,r2).nodes.len()==3 {
            if self.query_credit<0 { return NO_MOVE; }
            self.last_plan=MovePlan::SwapStar{insert1:1,insert2:1};
            return 0;
        }

        debug_assert!(r1 != r2);
        debug_assert!(pos1 > 0 && pos1 < self.routes[r1].nodes.len() - 1);
        debug_assert!(pos2 > 0 && pos2 < self.routes[r2].nodes.len() - 1);

        // The old code materialized a complete prefix/suffix Sequence for
        // every customer on every route refresh, although swap-star only
        // needs its load and distance here.  Removing one customer has an
        // exact three-arc delta, so compute those two scalars on demand.
        let route2=route_at(&self.routes,r2);let node_v=route2.node(pos2);
        let u=src.id;let v=node_v.id;
        let removed_distance1=src.removed_distance;
        let removed_distance2=route2.distance+node_v.removal_delta;
        let new_load1=src.remaining_load+node_v.load;
        let new_load2=route2.load-node_v.load+src.load;
        let old_total=src.cost+route2.cost;
        let bridge_u=src.bridge;let bridge_v=node_v.bridge;

        // First filter on route costs
        let new_pen1=if TABLE {copy_at(&self.capacity_costs,new_load1 as usize)}
            else {((new_load1-self.data.max_capacity).max(0) as i64)*self.params.penalty_capa as i64};
        let new_pen2=if TABLE {copy_at(&self.capacity_costs,new_load2 as usize)}
            else {((new_load2-self.data.max_capacity).max(0) as i64)*self.params.penalty_capa as i64};
        let cost_lb_r1_after_removal = (removed_distance1 as i64) + new_pen1;
        let cost_lb_r2_after_removal = (removed_distance2 as i64) + new_pen2;
        let mut lb_new_total = cost_lb_r1_after_removal + cost_lb_r2_after_removal;
        let max_acceptable_cost = src.max_cost+route2.cost;

        // first filter on route costs
        if lb_new_total > max_acceptable_cost {
            return NO_MOVE;
        }

        let data = self.data.as_ref();
        let route1_nodes = &route_at(&self.routes, r1).nodes;
        let route2_nodes = &route_at(&self.routes, r2).nodes;
        let route1_len = route1_nodes.len();
        let route2_len = route2_nodes.len();
        let distance_from_u = unsafe{data.distance_matrix.get_unchecked(src.row_start as usize..src.row_start as usize+data.nb_nodes)};
        let distance_to_u = unsafe{data.distance_matrix_transposed.get_unchecked(src.row_start as usize..src.row_start as usize+data.nb_nodes)};
        let distance_from_v = data.distance_row(v);
        let distance_to_v = data.distance_column(v);

        // Minimum distance detour for reinserting V after removing U.  The
        // two edges incident to U disappear; include the replacement bridge
        // explicitly and scan the remaining original edges.
        let pred_u = node_at(route1_nodes, pos1 - 1);
        let next_u = node_at(route1_nodes, pos1 + 1);
        let mut best_ins_v =
            copy_at(distance_to_v, pred_u.id) + copy_at(distance_from_v, next_u.id) - bridge_u;
        best_ins_v=best_ins_v.min(Self::route_insertion_lower_bound(data,route_at(&self.routes,r1),
            unsafe{self.insertion_bounds.get_unchecked_mut(r1*data.nb_nodes+v)},distance_from_v,distance_to_v));

        // Second filter on route costs
        lb_new_total += best_ins_v as i64;
        if lb_new_total > max_acceptable_cost {
            return NO_MOVE;
        }

        let pred_v = node_at(route2_nodes, pos2 - 1);
        let next_v = node_at(route2_nodes, pos2 + 1);
        let mut best_ins_u =
            copy_at(distance_to_u, pred_v.id) + copy_at(distance_from_u, next_v.id) - bridge_v;
        best_ins_u=best_ins_u.min(Self::route_insertion_lower_bound(data,route_at(&self.routes,r2),
            unsafe{self.insertion_bounds.get_unchecked_mut(r2*data.nb_nodes+u)},distance_from_u,distance_to_u));

        let search_limit=old_total+self.query_credit;
        // Third filter on route costs
        lb_new_total += best_ins_u as i64;
        if lb_new_total > search_limit {
            return NO_MOVE;
        }

        // Each insertion detour is at least best_ins. If detour plus
        // service is nonnegative, deleting that inserted visit from any
        // schedule cannot increase total warp. Thus the exact removed-route
        // warp is a lower bound for every insertion location. This uses the
        // actual arc minimum; no metric-distance assumption is required.
        let warp_floor1=if self.factored_bounds_valid && best_ins_v as i64+node_at(route2_nodes,pos2).t1.duration as i64>=0 {
            pred_u.prefix.warp+next_u.suffix.warp+(pred_u.prefix.time+bridge_u-next_u.suffix.time).max(0)
        } else {0};
        let warp_floor2=if self.factored_bounds_valid && best_ins_u as i64+node_at(route1_nodes,pos1).t1.duration as i64>=0 {
            pred_v.prefix.warp+next_v.suffix.warp+(pred_v.prefix.time+bridge_v-next_v.suffix.time).max(0)
        } else {0};
        let warp_charge1=warp_floor1 as i64*self.params.penalty_tw as i64;
        let warp_charge2=warp_floor2 as i64*self.params.penalty_tw as i64;
        if lb_new_total+warp_charge1+warp_charge2>search_limit {
            return NO_MOVE;
        }

        // Exact TW evaluation for one route cannot rescue a distance/capacity
        // lower bound that already exceeds the accepted total.  Keep the
        // other route's tight insertion lower bound available in both scans.
        let lb_cost2 = cost_lb_r2_after_removal + best_ins_u as i64 + warp_charge2;

        // Values needed only beyond the distance/capacity filters.
        let ptw = self.params.penalty_tw as i64;

        let inserted=node_at(route2_nodes,pos2).seq1();
        let cache=unsafe{self.removal_cache.get_unchecked_mut(u)};
        if cache.version!=route_at(&self.routes,r1).insertion_version {
            Self::prepare_removal_cache(route_at(&self.routes,r1),pos1,cache);
        }
        let mut best_t1=pos1;let mut best_cost1=i64::MAX/4;
        for entry in &cache.entries {
            let incoming=copy_at(distance_to_v,entry.pred as usize);
            let outgoing=copy_at(distance_from_v,entry.next as usize);
            let route_lb=(entry.distance+incoming+outgoing) as i64+new_pen1;
            if route_lb+warp_charge1<best_cost1 && route_lb+warp_charge1+lb_cost2<=search_limit {
                let arrival=entry.prefix_end+incoming;
                let first=(arrival-inserted.tau_plus).max(0);
                let last=inserted.earliest_end.max(arrival+inserted.duration_net)+outgoing-entry.suffix_latest;
                let tw=entry.warp+first.max(last);
                let cost=route_lb+tw as i64*ptw;
                if cost<best_cost1 {best_cost1=cost;best_t1=entry.position as usize;}
            }
        }

        // Fourth filter: one route is exact (TW-aware), the other remains a lower bound
        if best_cost1.saturating_add(lb_cost2) > search_limit {
            return NO_MOVE;
        }

        let inserted=node_at(route1_nodes,pos1).seq1();
        let cache=unsafe{self.removal_cache.get_unchecked_mut(v)};
        if cache.version!=route_at(&self.routes,r2).insertion_version {
            Self::prepare_removal_cache(route_at(&self.routes,r2),pos2,cache);
        }
        let mut best_t2=pos2;let mut best_cost2=i64::MAX/4;
        for entry in &cache.entries {
            let incoming=copy_at(distance_to_u,entry.pred as usize);
            let outgoing=copy_at(distance_from_u,entry.next as usize);
            let route_lb=(entry.distance+incoming+outgoing) as i64+new_pen2;
            if route_lb+warp_charge2<best_cost2 && best_cost1+route_lb+warp_charge2<=search_limit {
                let arrival=entry.prefix_end+incoming;
                let first=(arrival-inserted.tau_plus).max(0);
                let last=inserted.earliest_end.max(arrival+inserted.duration_net)+outgoing-entry.suffix_latest;
                let tw=entry.warp+first.max(last);
                let cost=route_lb+tw as i64*ptw;
                if cost<best_cost2 {best_cost2=cost;best_t2=entry.position as usize;}
            }
        }

        let new_total = best_cost1 + best_cost2;
        if new_total > search_limit {
            return NO_MOVE;
        }
        self.last_plan = MovePlan::SwapStar {
            insert1: best_t1,
            insert2: best_t2,
        };
        // A found feasible total proves that the original third and fourth
        // lower bounds pass. The second historical gate still needs auditing:
        // it omits the possibly negative detour of the other insertion.
        if !self.gate_audit_valid || self.params.penalty_capa>1_000_000 || self.params.penalty_tw>1_000_000 {
            return self.run_swapstar_exact(r1,pos1,r2,pos2);
        }
        if best_cost1+cost_lb_r2_after_removal>max_acceptable_cost {
            let threshold=max_acceptable_cost-cost_lb_r1_after_removal-cost_lb_r2_after_removal;
            if !Self::audit_insertion_gate(route1_nodes,pos1,distance_from_v,distance_to_v,bridge_u,threshold) {return NO_MOVE;}
        }
        new_total-old_total
    }
    #[inline(always)]
    fn run_swapstar<const TABLE:bool>(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize) -> i64 {
        if route_at(&self.routes,r1).nodes.len()==3 && route_at(&self.routes,r2).nodes.len()==3 {
            if self.query_credit<0 { return NO_MOVE; }
            self.last_plan=MovePlan::SwapStar{insert1:1,insert2:1};
            return 0;
        }

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
            let bridge_u = node_u.bridge;
            let bridge_v = node_v.bridge;
            (
                node_u.id,
                node_v.id,
                route1.distance + node_u.removal_delta,
                route2.distance + node_v.removal_delta,
                route1.load - node_u.seq1().load + node_v.seq1().load,
                route2.load - node_v.seq1().load + node_u.seq1().load,
                route1.cost + route2.cost,
                bridge_u,
                bridge_v,
            )
        };

        // First filter on route costs
        let new_pen1=if TABLE {copy_at(&self.capacity_costs,new_load1 as usize)}
            else {((new_load1-self.data.max_capacity).max(0) as i64)*self.params.penalty_capa as i64};
        let new_pen2=if TABLE {copy_at(&self.capacity_costs,new_load2 as usize)}
            else {((new_load2-self.data.max_capacity).max(0) as i64)*self.params.penalty_capa as i64};
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
        best_ins_v=best_ins_v.min(Self::route_insertion_lower_bound(data,route_at(&self.routes,r1),
            unsafe{self.insertion_bounds.get_unchecked_mut(r1*data.nb_nodes+v)},distance_from_v,distance_to_v));

        // Second filter on route costs
        lb_new_total += best_ins_v as i64;
        if lb_new_total > max_acceptable_cost {
            return NO_MOVE;
        }

        let pred_v = node_at(route2_nodes, pos2 - 1);
        let next_v = node_at(route2_nodes, pos2 + 1);
        let mut best_ins_u =
            copy_at(distance_to_u, pred_v.id) + copy_at(distance_from_u, next_v.id) - bridge_v;
        best_ins_u=best_ins_u.min(Self::route_insertion_lower_bound(data,route_at(&self.routes,r2),
            unsafe{self.insertion_bounds.get_unchecked_mut(r2*data.nb_nodes+u)},distance_from_u,distance_to_u));

        let search_limit=old_total+self.query_credit;
        // Third filter on route costs
        lb_new_total += best_ins_u as i64;
        if lb_new_total > search_limit {
            return NO_MOVE;
        }

        // Each insertion detour is at least best_ins. If detour plus
        // service is nonnegative, deleting that inserted visit from any
        // schedule cannot increase total warp. Thus the exact removed-route
        // warp is a lower bound for every insertion location. This uses the
        // actual arc minimum; no metric-distance assumption is required.
        let warp_floor1=if self.factored_bounds_valid && best_ins_v as i64+node_at(route2_nodes,pos2).t1.duration as i64>=0 {
            pred_u.prefix.warp+next_u.suffix.warp+(pred_u.prefix.time+bridge_u-next_u.suffix.time).max(0)
        } else {0};
        let warp_floor2=if self.factored_bounds_valid && best_ins_u as i64+node_at(route1_nodes,pos1).t1.duration as i64>=0 {
            pred_v.prefix.warp+next_v.suffix.warp+(pred_v.prefix.time+bridge_v-next_v.suffix.time).max(0)
        } else {0};
        let warp_charge1=warp_floor1 as i64*self.params.penalty_tw as i64;
        let warp_charge2=warp_floor2 as i64*self.params.penalty_tw as i64;
        if lb_new_total+warp_charge1+warp_charge2>search_limit {
            return NO_MOVE;
        }

        // Exact TW evaluation for one route cannot rescue a distance/capacity
        // lower bound that already exceeds the accepted total.  Keep the
        // other route's tight insertion lower bound available in both scans.
        let lb_cost2 = cost_lb_r2_after_removal + best_ins_u as i64 + warp_charge2;

        // Values needed only beyond the distance/capacity filters.
        let ptw = self.params.penalty_tw as i64;

        let inserted=node_at(route2_nodes,pos2).seq1();
        let cache=unsafe{self.removal_cache.get_unchecked_mut(u)};
        if cache.version!=route_at(&self.routes,r1).insertion_version {
            Self::prepare_removal_cache(route_at(&self.routes,r1),pos1,cache);
        }
        let mut best_t1=pos1;let mut best_cost1=i64::MAX/4;
        for entry in &cache.entries {
            let incoming=copy_at(distance_to_v,entry.pred as usize);
            let outgoing=copy_at(distance_from_v,entry.next as usize);
            let route_lb=(entry.distance+incoming+outgoing) as i64+new_pen1;
            if route_lb+warp_charge1<best_cost1 && route_lb+warp_charge1+lb_cost2<=search_limit {
                let arrival=entry.prefix_end+incoming;
                let first=(arrival-inserted.tau_plus).max(0);
                let last=inserted.earliest_end.max(arrival+inserted.duration_net)+outgoing-entry.suffix_latest;
                let tw=entry.warp+first.max(last);
                let cost=route_lb+tw as i64*ptw;
                if cost<best_cost1 {best_cost1=cost;best_t1=entry.position as usize;}
            }
        }

        // Fourth filter: one route is exact (TW-aware), the other remains a lower bound
        if best_cost1.saturating_add(lb_cost2) > search_limit {
            return NO_MOVE;
        }

        let inserted=node_at(route1_nodes,pos1).seq1();
        let cache=unsafe{self.removal_cache.get_unchecked_mut(v)};
        if cache.version!=route_at(&self.routes,r2).insertion_version {
            Self::prepare_removal_cache(route_at(&self.routes,r2),pos2,cache);
        }
        let mut best_t2=pos2;let mut best_cost2=i64::MAX/4;
        for entry in &cache.entries {
            let incoming=copy_at(distance_to_u,entry.pred as usize);
            let outgoing=copy_at(distance_from_u,entry.next as usize);
            let route_lb=(entry.distance+incoming+outgoing) as i64+new_pen2;
            if route_lb+warp_charge2<best_cost2 && best_cost1+route_lb+warp_charge2<=search_limit {
                let arrival=entry.prefix_end+incoming;
                let first=(arrival-inserted.tau_plus).max(0);
                let last=inserted.earliest_end.max(arrival+inserted.duration_net)+outgoing-entry.suffix_latest;
                let tw=entry.warp+first.max(last);
                let cost=route_lb+tw as i64*ptw;
                if cost<best_cost2 {best_cost2=cost;best_t2=entry.position as usize;}
            }
        }

        let new_total = best_cost1 + best_cost2;
        if new_total > search_limit {
            return NO_MOVE;
        }
        self.last_plan = MovePlan::SwapStar {
            insert1: best_t1,
            insert2: best_t2,
        };
        // A found feasible total proves that the original third and fourth
        // lower bounds pass. The second historical gate still needs auditing:
        // it omits the possibly negative detour of the other insertion.
        if !self.gate_audit_valid || self.params.penalty_capa>1_000_000 || self.params.penalty_tw>1_000_000 {
            return self.run_swapstar_exact(r1,pos1,r2,pos2);
        }
        if best_cost1+cost_lb_r2_after_removal>max_acceptable_cost {
            let threshold=max_acceptable_cost-cost_lb_r1_after_removal-cost_lb_r2_after_removal;
            if !Self::audit_insertion_gate(route1_nodes,pos1,distance_from_v,distance_to_v,bridge_u,threshold) {return NO_MOVE;}
        }
        new_total-old_total
    }

    #[inline(never)]
    fn run_swapstar_exact(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize) -> i64 {
        if route_at(&self.routes,r1).nodes.len()==3 && route_at(&self.routes,r2).nodes.len()==3 {
            if self.move_credit<0 { return NO_MOVE; }
            self.last_plan=MovePlan::SwapStar{insert1:1,insert2:1};
            return 0;
        }

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
            let bridge_u = node_u.bridge;
            let bridge_v = node_v.bridge;
            (
                node_u.id,
                node_v.id,
                route1.distance + node_u.removal_delta,
                route2.distance + node_v.removal_delta,
                route1.load - node_u.seq1().load + node_v.seq1().load,
                route2.load - node_v.seq1().load + node_u.seq1().load,
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
        let v_seq1 = node_at(route2_nodes, pos2).seq1();
        let mut best_t1: usize = pos1;
        let mut best_cost1 = i64::MAX / 4;

        // t <= pos1: build right-excluding-U by prepending seq1 fragments
        let mut right_excl_u = node_at(route1_nodes, pos1 + 1).seqi_n();
        for t in (1..=pos1).rev() {
            let left = node_at(route1_nodes, t - 1).seq0_i();
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
                    &node_at(route1_nodes, t - 1).seq1(),
                    &right_excl_u,
                    travel,
                );
            }
        }

        // t > pos1: build left-excluding-U incrementally
        let mut left_excl_u = Sequence::join2_with_travel(
            &node_at(route1_nodes, pos1 - 1).seq0_i(),
            &node_at(route1_nodes, pos1 + 1).seq1(),
            bridge_u,
        );
        for t in (pos1 + 2)..route1_len {
            let right = node_at(route1_nodes, t).seqi_n();
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
                    &node_at(route1_nodes, t).seq1(),
                    node_at(route1_nodes, t - 1).dist_to_succ,
                );
            }
        }

        // Fourth filter: one route is exact (TW-aware), the other remains a lower bound
        if best_cost1.saturating_add(lb_cost2) > max_acceptable_cost {
            return NO_MOVE;
        }

        // Reinsertion of U into r2 \ {V}
        let u_seq1 = node_at(route1_nodes, pos1).seq1();
        let mut best_t2: usize = pos2;
        let mut best_cost2 = i64::MAX / 4;

        // t <= pos2: build right-excluding-V by prepending seq1 fragments
        let mut right_excl_v = node_at(route2_nodes, pos2 + 1).seqi_n();
        for t in (1..=pos2).rev() {
            let left = node_at(route2_nodes, t - 1).seq0_i();
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
                    &node_at(route2_nodes, t - 1).seq1(),
                    &right_excl_v,
                    travel,
                );
            }
        }

        // t > pos2: build left-excluding-V incrementally
        let mut left_excl_v = Sequence::join2_with_travel(
            &node_at(route2_nodes, pos2 - 1).seq0_i(),
            &node_at(route2_nodes, pos2 + 1).seq1(),
            bridge_v,
        );
        for t in (pos2 + 2)..route2_len {
            let right = node_at(route2_nodes, t).seqi_n();
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
                    &node_at(route2_nodes, t).seq1(),
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
        self.refresh_capacity_costs();
        self.nb_moves += 1;
        for rid in 0..self.routes.len() {
            self.when_last_modified[rid] = 0;
            let r = &self.routes[rid];
            if r.load > self.data.max_capacity || r.tw > 0 {
                self.update_route(rid);
            } else {
                // The route timestamp above covers every customer in this route.
            }
        }
        self.search(rng);
        self.export_routes()
    }


    #[inline(always)]
    fn touch_recent_route(&mut self,rid:usize) {
        if self.recent_head==rid {return;}
        let prev=copy_at(&self.recent_prev,rid);let next=copy_at(&self.recent_next,rid);
        unsafe {
            if prev!=usize::MAX {*self.recent_next.get_unchecked_mut(prev)=next;}
            if next!=usize::MAX {*self.recent_prev.get_unchecked_mut(next)=prev;}
            let head=self.recent_head;
            *self.recent_prev.get_unchecked_mut(rid)=usize::MAX;
            *self.recent_next.get_unchecked_mut(rid)=head;
            if head!=usize::MAX {*self.recent_prev.get_unchecked_mut(head)=rid;}
            self.recent_head=rid;
        }
    }
    fn reset_recent_routes(&mut self) {
        if !self.masked_neighbors {return;}
        self.recent_prev.fill(usize::MAX);self.recent_next.fill(usize::MAX);self.recent_head=usize::MAX;
        // Entry is either run_from_routes or continue_repair: every nonzero
        // timestamp is the current move counter; the others are inherited.
        for rid in 0..self.routes.len() {
            let modified=copy_at(&self.when_last_modified,rid);
            debug_assert!(modified==0 || modified==self.nb_moves);
            if modified!=0 {self.touch_recent_route(rid);}
        }
    }
    #[inline(always)]
    fn active_neighbor_masks(&self,c:usize,rid:usize,modified:usize,last:usize)->NeighborMasks {
        let n=self.data.nb_nodes;
        if modified>last {
            let a=copy_at(&self.neighbors_before_offsets,c+1)-copy_at(&self.neighbors_before_offsets,c);
            let b=copy_at(&self.neighbors_capacity_swap_offsets,c+1)-copy_at(&self.neighbors_capacity_swap_offsets,c);
            let len=a+b;
            let full=if len==64 {u64::MAX} else {(1u64<<len)-1};
            let own=copy_at(&self.route_neighbor_masks,c*self.mask_stride+rid);
            NeighborMasks{before:full&!own.before}
        } else {
            let mut result=NeighborMasks::default();let mut r=self.recent_head;
            while r!=usize::MAX && copy_at(&self.when_last_modified,r)>last {
                // rid itself cannot appear in this range: its timestamp <= last.
                let mask=copy_at(&self.route_neighbor_masks,c*self.mask_stride+r);
                result.before|=mask.before;
                r=copy_at(&self.recent_next,r);
            }
            result
        }
    }
    #[cfg(target_arch="x86_64")]
    #[inline(always)]
    unsafe fn compress_neighbor_lanes(neighbors:&[u32],eligible:u64,packed:&mut [u32;72])->usize {
        use std::arch::x86_64::*;
        let mut count=0usize;
        for base in (0..neighbors.len()).step_by(8) {
            let start=base.min(neighbors.len()-8);let excluded=base-start;
            let mask=(((eligible>>start)&255) as u32)&!((1u32<<excluded)-1);
            if mask==0 {continue;}
            let ids=_mm256_loadu_si256(neighbors.as_ptr().add(start) as *const __m256i);
            let permutation=_mm256_load_si256(CUT_LANE_PERMUTATIONS.get_unchecked(mask as usize).0.as_ptr() as *const __m256i);
            let compressed=_mm256_permutevar8x32_epi32(ids,permutation);
            _mm256_storeu_si256(packed.as_mut_ptr().add(count) as *mut __m256i,compressed);
            count+=mask.count_ones() as usize;
        }
        count
    }
    // The record at customer c describes the cut immediately after c.
    // Every lane is a distinct neighbour; orientation minima are lower bounds
    // only. Original evaluators still select the move and preserve tie order.
    #[cfg(target_arch="x86_64")]
    #[target_feature(enable="avx2")]
    unsafe fn batch_neighbor_bounds<const THREE:bool>(&self,r1:usize,pos1:usize,neighbors:&[u32],eligible:u64)->(u64,u64) {
        let active=eligible.count_ones() as usize;
        if !self.sparse_bmi2 || neighbors.len()<8 || neighbors.len()>64 || active<4
            || (active+7)/8>=(neighbors.len()+7)/8 {
            return self.batch_neighbor_bounds_dense::<THREE>(r1,pos1,neighbors,eligible);
        }
        let mut packed=[0u32;72];
        let count=Self::compress_neighbor_lanes(neighbors,eligible,&mut packed);
        debug_assert_eq!(count,active);
        let padded=(count+7)&!7;let bits=(1u64<<count)-1;
        let result=self.batch_neighbor_bounds_dense::<THREE>(r1,pos1,&packed[..padded],bits);
        let fast=(std::arch::x86_64::_pdep_u64(result.0,eligible),std::arch::x86_64::_pdep_u64(result.1,eligible));
        fast
    }
    #[cfg(target_arch="x86_64")]
    #[inline(always)]
    unsafe fn batch_neighbor_bounds_dense<const THREE:bool>(&self,r1:usize,pos1:usize,neighbors:&[u32],eligible:u64)->(u64,u64) {
        use std::arch::x86_64::*;
        const SAFE:i64=67_108_864;
        let ru=route_at(&self.routes,r1);
        if neighbors.len()<8 || !self.batch_bounds_valid || ru.cost>SAFE || self.query_credit< -SAFE || self.query_credit>SAFE
            || eligible.count_ones()<4 {return (eligible,eligible);}
        let data=self.data.as_ref();let u=ru.node(pos1);let up=ru.node(pos1-1);let x=ru.node(pos1+1);
        let zero=_mm256_setzero_si256();let all=_mm256_set1_epi32(-1);
        macro_rules! b {($x:expr)=>{_mm256_set1_epi32($x as i32)}}
        macro_rules! add {($x:expr,$y:expr)=>{_mm256_add_epi32($x,$y)}}
        macro_rules! sub {($x:expr,$y:expr)=>{_mm256_sub_epi32($x,$y)}}
        macro_rules! mn {($x:expr,$y:expr)=>{_mm256_min_epi32($x,$y)}}
        macro_rules! gather {($slice:expr,$ids:expr)=>{_mm256_i32gather_epi32($slice.as_ptr(),$ids,4)}}
        let from_up=data.distance_row(up.id);let from_u=data.distance_row(u.id);let from_x=data.distance_row(x.id);
        let to_u=data.distance_column(u.id);let to_x=data.distance_column(x.id);
        let mut allow_inter=eligible;let mut allow_two=eligible;
        for block in 0..(neighbors.len()+7)/8 {
            let start=(block*8).min(neighbors.len()-8);
            let chunk=(eligible>>start)&255;if chunk==0 {continue;}
            // Eight packed u32 ids; start is at most len-8, including the overlapped tail.
            let ids=_mm256_loadu_si256(neighbors.as_ptr().add(start) as *const __m256i);
            let indexes=_mm256_mullo_epi32(ids,b!(16));
            macro_rules! f {($k:expr)=>{_mm256_i32gather_epi32((self.batch_cuts.as_ptr() as *const i32).add($k),indexes,4)}}
            let rload=f!(0);let total=zero;let rcost=f!(2);
            let limit=add!(b!(self.donor_cut_budget(u.id)+self.query_credit),rcost);
            let invalid=_mm256_cmpgt_epi32(rcost,b!(SAFE));
            let v=f!(3);let y=f!(4);let z=f!(5);let w=_mm256_and_si256(f!(6),b!(65535));
            
            let prededge=f!(7);let vedge=f!(8);let yedge=f!(9);let reverse=f!(10);
            let valid_v=_mm256_cmpgt_epi32(v,zero);let valid_y=_mm256_cmpgt_epi32(y,zero);
            let valid_z=_mm256_cmpgt_epi32(z,zero);
            let common_capacity=_mm256_mullo_epi32(_mm256_max_epi32(add!(rload,b!(ru.load-2*data.max_capacity)),zero),b!(self.params.penalty_capa));
            // The guarded i32 domain bounds both the original sum and this
            // subtraction. A + capacity <= limit iff A <= limit - capacity.
            let limit=sub!(limit,common_capacity);
            macro_rules! cap { ($send1:expr,$send2:expr)=>{zero}; }
            let dpu=gather!(to_u,ids);let dpv=gather!(from_up,v);let duv=gather!(from_u,v);
            let dvx=gather!(to_x,v);let duy=gather!(from_u,y);
            let remu1=up.dist_to_succ+u.dist_to_succ;
            let base10=sub!(sub!(total,b!(remu1)),prededge);
            let lb10=add!(add!(add!(base10,b!(u.bridge)),add!(dpu,duv)),cap!(u.load,zero));
            let mut best_all=lb10;
            let mut best_valid_v=b!(i32::MAX);
            let mut best_valid_y=b!(i32::MAX);
            let mut best_valid_z=b!(i32::MAX);

            let remv1=add!(prededge,vedge);let remv2=add!(remv1,yedge);
            let lb11=add!(add!(sub!(sub!(total,b!(remu1)),remv1),add!(add!(dpv,dvx),add!(dpu,duy))),cap!(u.load,vload));
            best_valid_v=mn!(best_valid_v,lb11);
            if x.id!=0 {
                let xn=ru.node(pos1+2);let to_xn=data.distance_column(xn.id);let from_xn=data.distance_row(xn.id);
                let remu2=remu1+x.dist_to_succ;let send2=u.load+x.load;
                let dpx=gather!(to_x,ids);let dxv=gather!(from_x,v);let dxy=gather!(from_x,y);
                let dxz=gather!(from_x,z);let duz=gather!(from_u,z);
                let dvxn=gather!(to_xn,v);let dyxn=gather!(to_xn,y);let dpy=gather!(from_up,y);
                let forward=add!(dpu,b!(u.dist_to_succ));let backward=add!(dpx,b!(data.dm(x.id,u.id)));
                let min20=mn!(add!(forward,dxv),add!(backward,duv));
                let lb20=add!(add!(sub!(sub!(total,b!(remu2)),prededge),add!(b!(data.dm(up.id,xn.id)),min20)),cap!(send2,zero));
                best_all=mn!(best_all,lb20);
                let min21=mn!(add!(forward,dxy),add!(backward,duy));
                let lb21=add!(add!(sub!(sub!(total,b!(remu2)),remv1),add!(add!(dpv,dvxn),min21)),cap!(send2,vload));
                best_valid_v=mn!(best_valid_v,lb21);
                let left_min=mn!(add!(add!(dpv,vedge),dyxn),add!(add!(dpy,reverse),dvxn));
                let right_min=mn!(add!(forward,dxz),add!(backward,duz));
                let lb22=add!(add!(sub!(sub!(total,b!(remu2)),remv2),add!(left_min,right_min)),cap!(send2,pairload));
                best_valid_y=mn!(best_valid_y,lb22);
                if THREE && xn.id!=0 {
                    let xnn=ru.node(pos1+3);let to_xnn=data.distance_column(xnn.id);
                    let remu3=remu2+xn.dist_to_succ;let send3=send2+xn.load;
                    let path=add!(dpu,b!(u.dist_to_succ+x.dist_to_succ));
                    let dxnv=gather!(from_xn,v);let dxny=gather!(from_xn,y);
                    let dvxnn=gather!(to_xnn,v);let dyxnn=gather!(to_xnn,y);
                    let dxnz=gather!(from_xn,z);
                    let lb40=add!(add!(sub!(sub!(total,b!(remu3)),prededge),add!(b!(data.dm(up.id,xnn.id)),add!(path,dxnv))),cap!(send3,zero));
                    best_all=mn!(best_all,lb40);
                    let lb41=add!(add!(sub!(sub!(total,b!(remu3)),remv1),add!(add!(dpv,dvxnn),add!(path,dxny))),cap!(send3,vload));
                    best_valid_v=mn!(best_valid_v,lb41);
                    let left_min=mn!(add!(add!(dpv,vedge),dyxnn),add!(add!(dpy,reverse),dvxnn));
                    let lb42=add!(add!(sub!(sub!(total,b!(remu3)),remv2),add!(left_min,add!(path,dxnz))),cap!(send3,pairload));
                    best_valid_y=mn!(best_valid_y,lb42);
                    // The final removed edge is z -> w; its travel is recovered
                    // from the distance matrix because the common record is small.
                    let dzw=_mm256_i32gather_epi32(data.distance_matrix.as_ptr(),add!(_mm256_mullo_epi32(z,b!(data.nb_nodes)),w),4);
                    let dzxnn=gather!(to_xnn,z);let dxnw=gather!(from_xn,w);
                    let lb44=add!(add!(sub!(sub!(total,b!(remu3)),add!(remv2,dzw)),
                        add!(add!(add!(dpv,vedge),add!(yedge,dzxnn)),add!(path,dxnw))),cap!(send3,threeload));
                    best_valid_z=mn!(best_valid_z,lb44);
                }
            }
            
            let lb_two=add!(add!(sub!(sub!(total,b!(up.dist_to_succ)),prededge),add!(dpv,dpu)),cap!(u.suffix.load,tail));
            let possible_two=_mm256_or_si256(invalid,_mm256_andnot_si256(_mm256_cmpgt_epi32(lb_two,limit),all));
            // Minima are combined only within an identical validity mask;
            // move ordering is still determined by the unchanged exact kernel.
            let mut possible=_mm256_andnot_si256(_mm256_cmpgt_epi32(best_all,limit),all);
            possible=_mm256_or_si256(possible,_mm256_andnot_si256(_mm256_cmpgt_epi32(best_valid_v,limit),valid_v));
            possible=_mm256_or_si256(possible,_mm256_andnot_si256(_mm256_cmpgt_epi32(best_valid_y,limit),valid_y));
            possible=_mm256_or_si256(possible,_mm256_andnot_si256(_mm256_cmpgt_epi32(best_valid_z,limit),valid_z));
            possible=_mm256_or_si256(possible,invalid);
            let inter_bits=_mm256_movemask_ps(_mm256_castsi256_ps(possible)) as u64;
            let two_bits=_mm256_movemask_ps(_mm256_castsi256_ps(possible_two)) as u64;
            allow_inter &= !(255u64<<start) | (inter_bits<<start);
            allow_two &= !(255u64<<start) | (two_bits<<start);
        }
        (allow_inter,allow_two)
    }

    #[cfg(target_arch="x86_64")]
    #[target_feature(enable="avx2")]
    unsafe fn batch_reverse_bounds<const THREE:bool>(&self,r2:usize,pos2:usize,neighbors:&[u32],eligible:u64)->u64 {
        let active=eligible.count_ones() as usize;
        if !self.sparse_bmi2 || neighbors.len()<8 || neighbors.len()>64 || active<4
            || (active+7)/8>=(neighbors.len()+7)/8 {
            return self.batch_reverse_bounds_dense::<THREE>(r2,pos2,neighbors,eligible);
        }
        let mut packed=[0u32;72];
        let count=Self::compress_neighbor_lanes(neighbors,eligible,&mut packed);
        debug_assert_eq!(count,active);
        let padded=(count+7)&!7;let bits=(1u64<<count)-1;
        let result=self.batch_reverse_bounds_dense::<THREE>(r2,pos2,&packed[..padded],bits);
        let fast=std::arch::x86_64::_pdep_u64(result,eligible);
        fast
    }
    #[cfg(target_arch="x86_64")]
    #[inline(always)]
    unsafe fn batch_reverse_bounds_dense<const THREE:bool>(&self,r2:usize,pos2:usize,neighbors:&[u32],eligible:u64)->u64 {
        use std::arch::x86_64::*;
        const SAFE:i64=67_108_864;
        let rv=route_at(&self.routes,r2);
        if neighbors.len()<8 || !self.batch_bounds_valid || rv.cost>SAFE
            || self.query_credit< -SAFE || self.query_credit>SAFE || eligible.count_ones()<4 {return eligible;}
        let data=self.data.as_ref();let v=rv.node(pos2);let vp=rv.node(pos2-1);let y=rv.node(pos2+1);
        let zero=_mm256_setzero_si256();let all=_mm256_set1_epi32(-1);
        macro_rules! b {($x:expr)=>{_mm256_set1_epi32($x as i32)}}
        macro_rules! add {($x:expr,$y:expr)=>{_mm256_add_epi32($x,$y)}}
        macro_rules! sub {($x:expr,$y:expr)=>{_mm256_sub_epi32($x,$y)}}
        macro_rules! mn {($x:expr,$y:expr)=>{_mm256_min_epi32($x,$y)}}
        macro_rules! gather {($slice:expr,$ids:expr)=>{_mm256_i32gather_epi32($slice.as_ptr(),$ids,4)}}
        macro_rules! dm {($from:expr,$to:expr)=>{_mm256_i32gather_epi32(data.distance_matrix.as_ptr(),add!(_mm256_mullo_epi32($from,b!(data.nb_nodes)),$to),4)}}
        let from_vp=data.distance_row(vp.id);let from_v=data.distance_row(v.id);
        let to_v=data.distance_column(v.id);let to_y=data.distance_column(y.id);
        let mut allowed=eligible;
        for block in 0..(neighbors.len()+7)/8 {
            let start=(block*8).min(neighbors.len()-8);if (eligible>>start)&255==0 {continue;}
            let u=_mm256_loadu_si256(neighbors.as_ptr().add(start) as *const __m256i);
            let indexes=_mm256_mullo_epi32(u,b!(16));
            macro_rules! f {($k:expr)=>{_mm256_i32gather_epi32((self.batch_cuts.as_ptr() as *const i32).add($k),indexes,4)}}
            let rload=f!(0);let total=zero;let rcost=f!(2);
            let invalid=_mm256_cmpgt_epi32(rcost,b!(SAFE));
            let donor_adjust=_mm256_mullo_epi32(_mm256_srai_epi32::<16>(f!(6)),b!(self.params.penalty_tw));
            let limit=add!(sub!(rcost,donor_adjust),b!(self.donor_cut_budget(v.id)+self.query_credit));
            let up=f!(11);let x=f!(3);let xn=f!(4);let xnn=f!(5);
            
            let upedge=f!(12);let uedge=f!(7);let xedge=f!(8);let xnedge=f!(9);
            let remu1=add!(upedge,uedge);let remu2=add!(remu1,xedge);let remu3=add!(remu2,xnedge);
            let valid_x=_mm256_cmpgt_epi32(x,zero);let valid_xn=_mm256_cmpgt_epi32(xn,zero);
            let common_capacity=_mm256_mullo_epi32(_mm256_max_epi32(add!(rload,b!(rv.load-2*data.max_capacity)),zero),b!(self.params.penalty_capa));
            // The guarded i32 domain bounds both the original sum and this
            // subtraction. A + capacity <= limit iff A <= limit - capacity.
            let limit=sub!(limit,common_capacity);
            macro_rules! cap { ($send1:expr,$send2:expr)=>{zero}; }
            let dpu=gather!(from_vp,u);let dpv=gather!(to_v,up);let duv=gather!(to_v,u);
            let dvx=gather!(from_v,x);let duy=gather!(to_y,u);
            let lb10=add!(add!(sub!(sub!(total,remu1),b!(vp.dist_to_succ)),add!(dm!(up,x),add!(dpu,duv))),cap!(uload,0));
            let mut best_all=lb10;
            let mut best_valid_x=b!(i32::MAX);
            let mut best_valid_xn=b!(i32::MAX);

            let remv1=vp.dist_to_succ+v.dist_to_succ;let remv2=remv1+y.dist_to_succ;
            let lb11=add!(add!(sub!(sub!(total,remu1),b!(remv1)),add!(add!(dpv,dvx),add!(dpu,duy))),cap!(uload,v.load));
            best_all=mn!(best_all,lb11);
            if _mm256_movemask_ps(_mm256_castsi256_ps(valid_x))!=0 {
                let dpx=gather!(from_vp,x);let dxv=gather!(to_v,x);let dxy=gather!(to_y,x);
                let dvxn=gather!(from_v,xn);let forward=add!(dpu,uedge);let backward=add!(dpx,dm!(x,u));
                let min20=mn!(add!(forward,dxv),add!(backward,duv));
                let lb20=add!(add!(sub!(sub!(total,remu2),b!(vp.dist_to_succ)),add!(dm!(up,xn),min20)),cap!(pairload,0));
                best_valid_x=mn!(best_valid_x,lb20);
                let min21=mn!(add!(forward,dxy),add!(backward,duy));
                let lb21=add!(add!(sub!(sub!(total,remu2),b!(remv1)),add!(add!(dpv,dvxn),min21)),cap!(pairload,v.load));
                best_valid_x=mn!(best_valid_x,lb21);
                if y.id!=0 {
                    let yn=rv.node(pos2+2);let to_yn=data.distance_column(yn.id);let from_y=data.distance_row(y.id);
                    let dpy=gather!(to_y,up);let dyxn=gather!(from_y,xn);
                    let left_min=mn!(add!(add!(dpv,b!(v.dist_to_succ)),dyxn),add!(add!(dpy,b!(data.dm(y.id,v.id))),dvxn));
                    let right_min=mn!(add!(forward,gather!(to_yn,x)),add!(backward,gather!(to_yn,u)));
                    let lb22=add!(add!(sub!(sub!(total,remu2),b!(remv2)),add!(left_min,right_min)),cap!(pairload,v.load+y.load));
                    best_valid_x=mn!(best_valid_x,lb22);
                }
                if THREE && _mm256_movemask_ps(_mm256_castsi256_ps(valid_xn))!=0 {
                    let path=add!(dpu,add!(uedge,xedge));let dvxnn=gather!(from_v,xnn);
                    let lb40=add!(add!(sub!(sub!(total,remu3),b!(vp.dist_to_succ)),add!(dm!(up,xnn),add!(path,gather!(to_v,xn)))),cap!(threeload,0));
                    best_valid_xn=mn!(best_valid_xn,lb40);
                    let lb41=add!(add!(sub!(sub!(total,remu3),b!(remv1)),add!(add!(dpv,dvxnn),add!(path,gather!(to_y,xn)))),cap!(threeload,v.load));
                    best_valid_xn=mn!(best_valid_xn,lb41);
                    if y.id!=0 {
                        let yn=rv.node(pos2+2);let to_yn=data.distance_column(yn.id);let from_y=data.distance_row(y.id);
                        let dpy=gather!(to_y,up);let dyxnn=gather!(from_y,xnn);
                        let left_min=mn!(add!(add!(dpv,b!(v.dist_to_succ)),dyxnn),add!(add!(dpy,b!(data.dm(y.id,v.id))),dvxnn));
                        let lb42=add!(add!(sub!(sub!(total,remu3),b!(remv2)),add!(left_min,add!(path,gather!(to_yn,xn)))),cap!(threeload,v.load+y.load));
                        best_valid_xn=mn!(best_valid_xn,lb42);
                        if yn.id!=0 {
                            let ynn=rv.node(pos2+3);let to_ynn=data.distance_column(ynn.id);let from_yn=data.distance_row(yn.id);
                            let left=add!(add!(dpv,b!(v.dist_to_succ+y.dist_to_succ)),gather!(from_yn,xnn));
                            let right=add!(path,gather!(to_ynn,xn));
                            let lb44=add!(add!(sub!(sub!(total,remu3),b!(remv2+yn.dist_to_succ)),add!(left,right)),cap!(threeload,v.load+y.load+yn.load));
                            best_valid_xn=mn!(best_valid_xn,lb44);
                        }
                    }
                }
            }
            // Minima are combined only within an identical validity mask;
            // move ordering is still determined by the unchanged exact kernel.
            let mut possible=_mm256_andnot_si256(_mm256_cmpgt_epi32(best_all,limit),all);
            possible=_mm256_or_si256(possible,_mm256_andnot_si256(_mm256_cmpgt_epi32(best_valid_x,limit),valid_x));
            possible=_mm256_or_si256(possible,_mm256_andnot_si256(_mm256_cmpgt_epi32(best_valid_xn,limit),valid_xn));
            possible=_mm256_or_si256(possible,invalid);
            let bits=_mm256_movemask_ps(_mm256_castsi256_ps(possible)) as u64;
            allowed &= !(255u64<<start) | (bits<<start);
        }
        allowed
    }

    #[cfg(target_arch="x86_64")]
    #[inline(always)]
    unsafe fn batch_neighbor_bounds_inline<const THREE:bool>(&self,r1:usize,pos1:usize,neighbors:&[u32],eligible:u64)->(u64,u64) {
        let active=eligible.count_ones() as usize;
        if !self.sparse_bmi2 || neighbors.len()<8 || neighbors.len()>64 || active<4
            || (active+7)/8>=(neighbors.len()+7)/8 {
            return self.batch_neighbor_bounds_inline_dense::<THREE>(r1,pos1,neighbors,eligible);
        }
        let mut packed=[0u32;72];
        let count=Self::compress_neighbor_lanes(neighbors,eligible,&mut packed);
        debug_assert_eq!(count,active);
        let padded=(count+7)&!7;let bits=(1u64<<count)-1;
        let result=self.batch_neighbor_bounds_inline_dense::<THREE>(r1,pos1,&packed[..padded],bits);
        let fast=(std::arch::x86_64::_pdep_u64(result.0,eligible),std::arch::x86_64::_pdep_u64(result.1,eligible));
        fast
    }
    #[cfg(target_arch="x86_64")]
    #[inline(always)]
    unsafe fn batch_neighbor_bounds_inline_dense<const THREE:bool>(&self,r1:usize,pos1:usize,neighbors:&[u32],eligible:u64)->(u64,u64) {
        use std::arch::x86_64::*;
        const SAFE:i64=67_108_864;
        let ru=route_at(&self.routes,r1);
        if neighbors.len()<8 || !self.batch_bounds_valid || ru.cost>SAFE || self.query_credit< -SAFE || self.query_credit>SAFE
            || eligible.count_ones()<4 {return (eligible,eligible);}
        let data=self.data.as_ref();let u=ru.node(pos1);let up=ru.node(pos1-1);let x=ru.node(pos1+1);
        let zero=_mm256_setzero_si256();let all=_mm256_set1_epi32(-1);
        macro_rules! b {($x:expr)=>{_mm256_set1_epi32($x as i32)}}
        macro_rules! add {($x:expr,$y:expr)=>{_mm256_add_epi32($x,$y)}}
        macro_rules! sub {($x:expr,$y:expr)=>{_mm256_sub_epi32($x,$y)}}
        macro_rules! mn {($x:expr,$y:expr)=>{_mm256_min_epi32($x,$y)}}
        macro_rules! gather {($slice:expr,$ids:expr)=>{_mm256_i32gather_epi32($slice.as_ptr(),$ids,4)}}
        let from_up=data.distance_row(up.id);let from_u=data.distance_row(u.id);let from_x=data.distance_row(x.id);
        let to_u=data.distance_column(u.id);let to_x=data.distance_column(x.id);
        let mut allow_inter=eligible;let mut allow_two=eligible;
        for block in 0..(neighbors.len()+7)/8 {
            let start=(block*8).min(neighbors.len()-8);
            let chunk=(eligible>>start)&255;if chunk==0 {continue;}
            // Eight packed u32 ids; start is at most len-8, including the overlapped tail.
            let ids=_mm256_loadu_si256(neighbors.as_ptr().add(start) as *const __m256i);
            let indexes=_mm256_mullo_epi32(ids,b!(16));
            macro_rules! f {($k:expr)=>{_mm256_i32gather_epi32((self.batch_cuts.as_ptr() as *const i32).add($k),indexes,4)}}
            let rload=f!(0);let total=zero;let rcost=f!(2);
            let limit=add!(b!(self.donor_cut_budget(u.id)+self.query_credit),rcost);
            let invalid=_mm256_cmpgt_epi32(rcost,b!(SAFE));
            let v=f!(3);let y=f!(4);let z=f!(5);let w=_mm256_and_si256(f!(6),b!(65535));
            
            let prededge=f!(7);let vedge=f!(8);let yedge=f!(9);let reverse=f!(10);
            let valid_v=_mm256_cmpgt_epi32(v,zero);let valid_y=_mm256_cmpgt_epi32(y,zero);
            let valid_z=_mm256_cmpgt_epi32(z,zero);
            let common_capacity=_mm256_mullo_epi32(_mm256_max_epi32(add!(rload,b!(ru.load-2*data.max_capacity)),zero),b!(self.params.penalty_capa));
            // The guarded i32 domain bounds both the original sum and this
            // subtraction. A + capacity <= limit iff A <= limit - capacity.
            let limit=sub!(limit,common_capacity);
            macro_rules! cap { ($send1:expr,$send2:expr)=>{zero}; }
            let dpu=gather!(to_u,ids);let dpv=gather!(from_up,v);let duv=gather!(from_u,v);
            let dvx=gather!(to_x,v);let duy=gather!(from_u,y);
            let remu1=up.dist_to_succ+u.dist_to_succ;
            let base10=sub!(sub!(total,b!(remu1)),prededge);
            let lb10=add!(add!(add!(base10,b!(u.bridge)),add!(dpu,duv)),cap!(u.load,zero));
            let mut best_all=lb10;
            let mut best_valid_v=b!(i32::MAX);
            let mut best_valid_y=b!(i32::MAX);
            let mut best_valid_z=b!(i32::MAX);

            let remv1=add!(prededge,vedge);let remv2=add!(remv1,yedge);
            let lb11=add!(add!(sub!(sub!(total,b!(remu1)),remv1),add!(add!(dpv,dvx),add!(dpu,duy))),cap!(u.load,vload));
            best_valid_v=mn!(best_valid_v,lb11);
            if x.id!=0 {
                let xn=ru.node(pos1+2);let to_xn=data.distance_column(xn.id);let from_xn=data.distance_row(xn.id);
                let remu2=remu1+x.dist_to_succ;let send2=u.load+x.load;
                let dpx=gather!(to_x,ids);let dxv=gather!(from_x,v);let dxy=gather!(from_x,y);
                let dxz=gather!(from_x,z);let duz=gather!(from_u,z);
                let dvxn=gather!(to_xn,v);let dyxn=gather!(to_xn,y);let dpy=gather!(from_up,y);
                let forward=add!(dpu,b!(u.dist_to_succ));let backward=add!(dpx,b!(data.dm(x.id,u.id)));
                let min20=mn!(add!(forward,dxv),add!(backward,duv));
                let lb20=add!(add!(sub!(sub!(total,b!(remu2)),prededge),add!(b!(data.dm(up.id,xn.id)),min20)),cap!(send2,zero));
                best_all=mn!(best_all,lb20);
                let min21=mn!(add!(forward,dxy),add!(backward,duy));
                let lb21=add!(add!(sub!(sub!(total,b!(remu2)),remv1),add!(add!(dpv,dvxn),min21)),cap!(send2,vload));
                best_valid_v=mn!(best_valid_v,lb21);
                let left_min=mn!(add!(add!(dpv,vedge),dyxn),add!(add!(dpy,reverse),dvxn));
                let right_min=mn!(add!(forward,dxz),add!(backward,duz));
                let lb22=add!(add!(sub!(sub!(total,b!(remu2)),remv2),add!(left_min,right_min)),cap!(send2,pairload));
                best_valid_y=mn!(best_valid_y,lb22);
                if THREE && xn.id!=0 {
                    let xnn=ru.node(pos1+3);let to_xnn=data.distance_column(xnn.id);
                    let remu3=remu2+xn.dist_to_succ;let send3=send2+xn.load;
                    let path=add!(dpu,b!(u.dist_to_succ+x.dist_to_succ));
                    let dxnv=gather!(from_xn,v);let dxny=gather!(from_xn,y);
                    let dvxnn=gather!(to_xnn,v);let dyxnn=gather!(to_xnn,y);
                    let dxnz=gather!(from_xn,z);
                    let lb40=add!(add!(sub!(sub!(total,b!(remu3)),prededge),add!(b!(data.dm(up.id,xnn.id)),add!(path,dxnv))),cap!(send3,zero));
                    best_all=mn!(best_all,lb40);
                    let lb41=add!(add!(sub!(sub!(total,b!(remu3)),remv1),add!(add!(dpv,dvxnn),add!(path,dxny))),cap!(send3,vload));
                    best_valid_v=mn!(best_valid_v,lb41);
                    let left_min=mn!(add!(add!(dpv,vedge),dyxnn),add!(add!(dpy,reverse),dvxnn));
                    let lb42=add!(add!(sub!(sub!(total,b!(remu3)),remv2),add!(left_min,add!(path,dxnz))),cap!(send3,pairload));
                    best_valid_y=mn!(best_valid_y,lb42);
                    // The final removed edge is z -> w; its travel is recovered
                    // from the distance matrix because the common record is small.
                    let dzw=_mm256_i32gather_epi32(data.distance_matrix.as_ptr(),add!(_mm256_mullo_epi32(z,b!(data.nb_nodes)),w),4);
                    let dzxnn=gather!(to_xnn,z);let dxnw=gather!(from_xn,w);
                    let lb44=add!(add!(sub!(sub!(total,b!(remu3)),add!(remv2,dzw)),
                        add!(add!(add!(dpv,vedge),add!(yedge,dzxnn)),add!(path,dxnw))),cap!(send3,threeload));
                    best_valid_z=mn!(best_valid_z,lb44);
                }
            }
            
            let lb_two=add!(add!(sub!(sub!(total,b!(up.dist_to_succ)),prededge),add!(dpv,dpu)),cap!(u.suffix.load,tail));
            let possible_two=_mm256_or_si256(invalid,_mm256_andnot_si256(_mm256_cmpgt_epi32(lb_two,limit),all));
            // Minima are combined only within an identical validity mask;
            // move ordering is still determined by the unchanged exact kernel.
            let mut possible=_mm256_andnot_si256(_mm256_cmpgt_epi32(best_all,limit),all);
            possible=_mm256_or_si256(possible,_mm256_andnot_si256(_mm256_cmpgt_epi32(best_valid_v,limit),valid_v));
            possible=_mm256_or_si256(possible,_mm256_andnot_si256(_mm256_cmpgt_epi32(best_valid_y,limit),valid_y));
            possible=_mm256_or_si256(possible,_mm256_andnot_si256(_mm256_cmpgt_epi32(best_valid_z,limit),valid_z));
            possible=_mm256_or_si256(possible,invalid);
            let inter_bits=_mm256_movemask_ps(_mm256_castsi256_ps(possible)) as u64;
            let two_bits=_mm256_movemask_ps(_mm256_castsi256_ps(possible_two)) as u64;
            allow_inter &= !(255u64<<start) | (inter_bits<<start);
            allow_two &= !(255u64<<start) | (two_bits<<start);
        }
        (allow_inter,allow_two)
    }
    #[cfg(target_arch="x86_64")]
    #[inline(always)]
    unsafe fn batch_reverse_bounds_inline<const THREE:bool>(&self,r2:usize,pos2:usize,neighbors:&[u32],eligible:u64)->u64 {
        let active=eligible.count_ones() as usize;
        if !self.sparse_bmi2 || neighbors.len()<8 || neighbors.len()>64 || active<4
            || (active+7)/8>=(neighbors.len()+7)/8 {
            return self.batch_reverse_bounds_inline_dense::<THREE>(r2,pos2,neighbors,eligible);
        }
        let mut packed=[0u32;72];
        let count=Self::compress_neighbor_lanes(neighbors,eligible,&mut packed);
        debug_assert_eq!(count,active);
        let padded=(count+7)&!7;let bits=(1u64<<count)-1;
        let result=self.batch_reverse_bounds_inline_dense::<THREE>(r2,pos2,&packed[..padded],bits);
        let fast=std::arch::x86_64::_pdep_u64(result,eligible);
        fast
    }
    #[cfg(target_arch="x86_64")]
    #[inline(always)]
    unsafe fn batch_reverse_bounds_inline_dense<const THREE:bool>(&self,r2:usize,pos2:usize,neighbors:&[u32],eligible:u64)->u64 {
        use std::arch::x86_64::*;
        const SAFE:i64=67_108_864;
        let rv=route_at(&self.routes,r2);
        if neighbors.len()<8 || !self.batch_bounds_valid || rv.cost>SAFE
            || self.query_credit< -SAFE || self.query_credit>SAFE || eligible.count_ones()<4 {return eligible;}
        let data=self.data.as_ref();let v=rv.node(pos2);let vp=rv.node(pos2-1);let y=rv.node(pos2+1);
        let zero=_mm256_setzero_si256();let all=_mm256_set1_epi32(-1);
        macro_rules! b {($x:expr)=>{_mm256_set1_epi32($x as i32)}}
        macro_rules! add {($x:expr,$y:expr)=>{_mm256_add_epi32($x,$y)}}
        macro_rules! sub {($x:expr,$y:expr)=>{_mm256_sub_epi32($x,$y)}}
        macro_rules! mn {($x:expr,$y:expr)=>{_mm256_min_epi32($x,$y)}}
        macro_rules! gather {($slice:expr,$ids:expr)=>{_mm256_i32gather_epi32($slice.as_ptr(),$ids,4)}}
        macro_rules! dm {($from:expr,$to:expr)=>{_mm256_i32gather_epi32(data.distance_matrix.as_ptr(),add!(_mm256_mullo_epi32($from,b!(data.nb_nodes)),$to),4)}}
        let from_vp=data.distance_row(vp.id);let from_v=data.distance_row(v.id);
        let to_v=data.distance_column(v.id);let to_y=data.distance_column(y.id);
        let mut allowed=eligible;
        for block in 0..(neighbors.len()+7)/8 {
            let start=(block*8).min(neighbors.len()-8);if (eligible>>start)&255==0 {continue;}
            let u=_mm256_loadu_si256(neighbors.as_ptr().add(start) as *const __m256i);
            let indexes=_mm256_mullo_epi32(u,b!(16));
            macro_rules! f {($k:expr)=>{_mm256_i32gather_epi32((self.batch_cuts.as_ptr() as *const i32).add($k),indexes,4)}}
            let rload=f!(0);let total=zero;let rcost=f!(2);
            let invalid=_mm256_cmpgt_epi32(rcost,b!(SAFE));
            let donor_adjust=_mm256_mullo_epi32(_mm256_srai_epi32::<16>(f!(6)),b!(self.params.penalty_tw));
            let limit=add!(sub!(rcost,donor_adjust),b!(self.donor_cut_budget(v.id)+self.query_credit));
            let up=f!(11);let x=f!(3);let xn=f!(4);let xnn=f!(5);
            
            let upedge=f!(12);let uedge=f!(7);let xedge=f!(8);let xnedge=f!(9);
            let remu1=add!(upedge,uedge);let remu2=add!(remu1,xedge);let remu3=add!(remu2,xnedge);
            let valid_x=_mm256_cmpgt_epi32(x,zero);let valid_xn=_mm256_cmpgt_epi32(xn,zero);
            let common_capacity=_mm256_mullo_epi32(_mm256_max_epi32(add!(rload,b!(rv.load-2*data.max_capacity)),zero),b!(self.params.penalty_capa));
            // The guarded i32 domain bounds both the original sum and this
            // subtraction. A + capacity <= limit iff A <= limit - capacity.
            let limit=sub!(limit,common_capacity);
            macro_rules! cap { ($send1:expr,$send2:expr)=>{zero}; }
            let dpu=gather!(from_vp,u);let dpv=gather!(to_v,up);let duv=gather!(to_v,u);
            let dvx=gather!(from_v,x);let duy=gather!(to_y,u);
            let lb10=add!(add!(sub!(sub!(total,remu1),b!(vp.dist_to_succ)),add!(dm!(up,x),add!(dpu,duv))),cap!(uload,0));
            let mut best_all=lb10;
            let mut best_valid_x=b!(i32::MAX);
            let mut best_valid_xn=b!(i32::MAX);

            let remv1=vp.dist_to_succ+v.dist_to_succ;let remv2=remv1+y.dist_to_succ;
            let lb11=add!(add!(sub!(sub!(total,remu1),b!(remv1)),add!(add!(dpv,dvx),add!(dpu,duy))),cap!(uload,v.load));
            best_all=mn!(best_all,lb11);
            if _mm256_movemask_ps(_mm256_castsi256_ps(valid_x))!=0 {
                let dpx=gather!(from_vp,x);let dxv=gather!(to_v,x);let dxy=gather!(to_y,x);
                let dvxn=gather!(from_v,xn);let forward=add!(dpu,uedge);let backward=add!(dpx,dm!(x,u));
                let min20=mn!(add!(forward,dxv),add!(backward,duv));
                let lb20=add!(add!(sub!(sub!(total,remu2),b!(vp.dist_to_succ)),add!(dm!(up,xn),min20)),cap!(pairload,0));
                best_valid_x=mn!(best_valid_x,lb20);
                let min21=mn!(add!(forward,dxy),add!(backward,duy));
                let lb21=add!(add!(sub!(sub!(total,remu2),b!(remv1)),add!(add!(dpv,dvxn),min21)),cap!(pairload,v.load));
                best_valid_x=mn!(best_valid_x,lb21);
                if y.id!=0 {
                    let yn=rv.node(pos2+2);let to_yn=data.distance_column(yn.id);let from_y=data.distance_row(y.id);
                    let dpy=gather!(to_y,up);let dyxn=gather!(from_y,xn);
                    let left_min=mn!(add!(add!(dpv,b!(v.dist_to_succ)),dyxn),add!(add!(dpy,b!(data.dm(y.id,v.id))),dvxn));
                    let right_min=mn!(add!(forward,gather!(to_yn,x)),add!(backward,gather!(to_yn,u)));
                    let lb22=add!(add!(sub!(sub!(total,remu2),b!(remv2)),add!(left_min,right_min)),cap!(pairload,v.load+y.load));
                    best_valid_x=mn!(best_valid_x,lb22);
                }
                if THREE && _mm256_movemask_ps(_mm256_castsi256_ps(valid_xn))!=0 {
                    let path=add!(dpu,add!(uedge,xedge));let dvxnn=gather!(from_v,xnn);
                    let lb40=add!(add!(sub!(sub!(total,remu3),b!(vp.dist_to_succ)),add!(dm!(up,xnn),add!(path,gather!(to_v,xn)))),cap!(threeload,0));
                    best_valid_xn=mn!(best_valid_xn,lb40);
                    let lb41=add!(add!(sub!(sub!(total,remu3),b!(remv1)),add!(add!(dpv,dvxnn),add!(path,gather!(to_y,xn)))),cap!(threeload,v.load));
                    best_valid_xn=mn!(best_valid_xn,lb41);
                    if y.id!=0 {
                        let yn=rv.node(pos2+2);let to_yn=data.distance_column(yn.id);let from_y=data.distance_row(y.id);
                        let dpy=gather!(to_y,up);let dyxnn=gather!(from_y,xnn);
                        let left_min=mn!(add!(add!(dpv,b!(v.dist_to_succ)),dyxnn),add!(add!(dpy,b!(data.dm(y.id,v.id))),dvxnn));
                        let lb42=add!(add!(sub!(sub!(total,remu3),b!(remv2)),add!(left_min,add!(path,gather!(to_yn,xn)))),cap!(threeload,v.load+y.load));
                        best_valid_xn=mn!(best_valid_xn,lb42);
                        if yn.id!=0 {
                            let ynn=rv.node(pos2+3);let to_ynn=data.distance_column(ynn.id);let from_yn=data.distance_row(yn.id);
                            let left=add!(add!(dpv,b!(v.dist_to_succ+y.dist_to_succ)),gather!(from_yn,xnn));
                            let right=add!(path,gather!(to_ynn,xn));
                            let lb44=add!(add!(sub!(sub!(total,remu3),b!(remv2+yn.dist_to_succ)),add!(left,right)),cap!(threeload,v.load+y.load+yn.load));
                            best_valid_xn=mn!(best_valid_xn,lb44);
                        }
                    }
                }
            }
            // Minima are combined only within an identical validity mask;
            // move ordering is still determined by the unchanged exact kernel.
            let mut possible=_mm256_andnot_si256(_mm256_cmpgt_epi32(best_all,limit),all);
            possible=_mm256_or_si256(possible,_mm256_andnot_si256(_mm256_cmpgt_epi32(best_valid_x,limit),valid_x));
            possible=_mm256_or_si256(possible,_mm256_andnot_si256(_mm256_cmpgt_epi32(best_valid_xn,limit),valid_xn));
            possible=_mm256_or_si256(possible,invalid);
            let bits=_mm256_movemask_ps(_mm256_castsi256_ps(possible)) as u64;
            allowed &= !(255u64<<start) | (bits<<start);
        }
        allowed
    }
    #[cfg(target_arch="x86_64")]
    #[target_feature(enable="avx2")]
    #[inline(never)]
    unsafe fn evaluate_masked_customer_avx<const THREE: bool, const TABLE: bool>(
        &mut self,
        c1: usize,
        last_tested: usize,
        loop_id: usize,
    ) -> i64 {
        if loop_id != 2 && last_tested == self.nb_moves { return NO_MOVE; }
        self.query_credit=self.move_credit;
        let mut best_delta: i64 = i64::MAX; // best acceptable move
        let mut best_move: Option<CandidateMove> = None;
        let mut best_plan = MovePlan::None;
        let loc1 = copy_at(&self.node_locations,c1);
        let r1 = (loc1.route_and_position>>32) as usize;
        let pos1 = loc1.route_and_position as u32 as usize;
        let inter_source=self.make_inter_source::<THREE>(r1,pos1);

        {
            let r1_last_mod = copy_at(&self.when_last_modified,r1);
            let masks=self.active_neighbor_masks(c1,r1,r1_last_mod,last_tested);
            let neighbors_start = copy_at(&self.neighbors_before_offsets, c1);
            let neighbors_end = copy_at(&self.neighbors_before_offsets, c1 + 1);
            let count=neighbors_end-neighbors_start;
            let mut pending=masks.before & if count==64 {u64::MAX} else {(1u64<<count)-1};
            #[cfg(target_arch="x86_64")]
            let (batch_inter,batch_two)=if self.vector_arcs {
                unsafe{self.batch_neighbor_bounds_inline::<THREE>(r1,pos1,&self.neighbors_before[neighbors_start..neighbors_end],pending)}
            } else {(pending,pending)};
            #[cfg(not(target_arch="x86_64"))]
            let (batch_inter,batch_two)=(pending,pending);
            #[cfg(target_arch="x86_64")]
            let batch_reverse=if pos1==1 && self.vector_arcs {
                unsafe{self.batch_reverse_bounds_inline::<THREE>(r1,pos1,&self.neighbors_before[neighbors_start..neighbors_end],pending)}
            } else {pending};
            #[cfg(not(target_arch="x86_64"))]
            let batch_reverse=pending;
            pending &= batch_inter | batch_two | if pos1==1 {batch_reverse} else {0};
            while pending!=0 {
                let k=neighbors_start+pending.trailing_zeros() as usize;pending&=pending-1;
                let c2 = copy_at(&self.neighbors_before, k) as usize;
                let loc2 = copy_at(&self.node_locations,c2);
                let r2 = (loc2.route_and_position>>32) as usize;

                // Skip if both routes unchanged since last tests for this customer
                {
                    // We use pos2 + 1 for the SWAP and RELOCATE moves since c2 is a good predecessor for c1
                    // Moves listed here create the edge c2 => c1, but never insert immediately after a depot
                    let pos2 = loc2.route_and_position as u32 as usize;

                    let delta = if batch_inter&(1u64<<(k-neighbors_start))!=0 {
                        self.run_inter_source::<THREE,TABLE>(r1,pos1,r2,pos2+1,&inter_source)
                    } else {NO_MOVE};
                    if delta < best_delta {
                        best_delta = delta;
                        best_move = Some(CandidateMove::InterRoute {
                            r1,
                            pos1,
                            r2,
                            pos2: pos2 + 1,
                        });
                        best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
                    }

                    // Special case to manage insert immediately after a depot
                    if pos1 == 1 {
                        let delta = if batch_reverse&(1u64<<(k-neighbors_start))!=0 {
                            self.run_inter_mode::<THREE,TABLE>(r2,pos2,r1,pos1)
                        } else {NO_MOVE};
                        if delta < best_delta {
                            best_delta = delta;
                            best_move = Some(CandidateMove::InterRoute {
                                r1: r2,
                                pos1: pos2,
                                r2: r1,
                                pos2: pos1,
                            });
                            best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
                        }
                    }

                    let delta = if batch_two&(1u64<<(k-neighbors_start))!=0 {
                        self.run_2optstar::<TABLE>(r1,pos1,r2,pos2+1)
                    } else {NO_MOVE};
                    if delta < best_delta {
                        best_delta = delta;
                        best_move = Some(CandidateMove::TwoOptStar {
                            r1,
                            pos1,
                            r2,
                            pos2: pos2 + 1,
                        });
                        best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
                    }
                }
            }

            let swap_source=self.make_swap_source(r1,pos1);
            let capacity_start = copy_at(&self.neighbors_capacity_swap_offsets, c1);
            let capacity_end = copy_at(&self.neighbors_capacity_swap_offsets, c1 + 1);
            let mut pending=masks.before.checked_shr((neighbors_end-neighbors_start) as u32).unwrap_or(0);
            while pending!=0 {
                let k=capacity_start+pending.trailing_zeros() as usize;pending&=pending-1;
                let c2 = copy_at(&self.neighbors_capacity_swap, k) as usize;
                let loc2 = copy_at(&self.node_locations,c2);
                let r2 = (loc2.route_and_position>>32) as usize;

                // Skip if both routes unchanged since last tests for this customer
                {
                    let pos2 = loc2.route_and_position as u32 as usize;
                    let delta = self.run_swapstar_source::<TABLE>(r1,pos1,r2,pos2,&swap_source);
                    if delta < best_delta {
                        best_delta = delta;
                        best_move = Some(CandidateMove::SwapStar { r1, pos1, r2, pos2 });
                        best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
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
                        let delta = self.run_2optstar::<TABLE>(r1, pos1, r2, pos2);
                        if delta < best_delta {
                            best_delta = delta;
                            best_move = Some(CandidateMove::TwoOptStar { r1, pos1, r2, pos2 });
                            best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
                        }
                    }

                    // Insert in an empty route
                    let delta = self.run_inter_source::<THREE,TABLE>(r1,pos1,r2,pos2,&inter_source);
                    if delta < best_delta {
                        best_delta = delta;
                        best_move = Some(CandidateMove::InterRoute { r1, pos1, r2, pos2 });
                        best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
                    }
                }
            }
        }

        // Intra-route moves
        if copy_at(&self.when_last_modified, r1) > last_tested {
            let delta = self.run_intra_route_relocate::<TABLE>(r1, pos1);
            if delta < best_delta {
                best_delta = delta;
                best_move = Some(CandidateMove::IntraRelocate { r1, pos1 });
                best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
            }

            let delta = self.run_intra_route_oropt2::<TABLE>(r1, pos1);
            if delta < best_delta {
                best_delta = delta;
                best_move = Some(CandidateMove::IntraOrOpt2 { r1, pos1 });
                best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
            }

            let delta = self.run_intra_route_swap::<TABLE>(r1, pos1);
            if delta < best_delta {
                best_delta = delta;
                best_move = Some(CandidateMove::IntraSwap { r1, pos1 });
                best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
            }

            let delta = self.run_2opt::<TABLE>(r1, pos1);
            if delta < best_delta {
                best_delta = delta;
                best_move = Some(CandidateMove::Intra2Opt { r1, pos1 });
                best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
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
    #[inline(always)]
    fn evaluate_and_apply_best_move_for_customer<const THREE:bool,const TABLE:bool>(&mut self,c1:usize,last_tested:usize,loop_id:usize)->i64 {
        #[cfg(target_arch="x86_64")]
        if self.vector_arcs && copy_at(&self.masked_customers,c1) {
            return unsafe{if self.batch_bounds_valid && self.sparse_bmi2 {
                if self.minpos_arcs {self.evaluate_assured_customer_avx::<THREE,TABLE,true>(c1,last_tested,loop_id)} else {self.evaluate_assured_customer_avx::<THREE,TABLE,false>(c1,last_tested,loop_id)}
            } else {self.evaluate_masked_customer_avx::<THREE,TABLE>(c1,last_tested,loop_id)}};
        }
        if copy_at(&self.masked_customers,c1) {self.evaluate_masked_customer::<THREE,TABLE>(c1,last_tested,loop_id)}
        else if self.batch_bounds_valid && copy_at(&self.neighbors_before_offsets,c1+1)-copy_at(&self.neighbors_before_offsets,c1)<=64 {
            self.evaluate_scan_customer::<THREE,TABLE>(c1,last_tested,loop_id)
        }
        else {self.evaluate_scalar_customer::<THREE,TABLE>(c1,last_tested,loop_id)}
    }
    #[inline(never)]
    fn evaluate_masked_customer<const THREE: bool, const TABLE: bool>(
        &mut self,
        c1: usize,
        last_tested: usize,
        loop_id: usize,
    ) -> i64 {
        if loop_id != 2 && last_tested == self.nb_moves { return NO_MOVE; }
        self.query_credit=self.move_credit;
        let mut best_delta: i64 = i64::MAX; // best acceptable move
        let mut best_move: Option<CandidateMove> = None;
        let mut best_plan = MovePlan::None;
        let loc1 = copy_at(&self.node_locations,c1);
        let r1 = (loc1.route_and_position>>32) as usize;
        let pos1 = loc1.route_and_position as u32 as usize;
        let inter_source=self.make_inter_source::<THREE>(r1,pos1);

        {
            let r1_last_mod = copy_at(&self.when_last_modified,r1);
            let masks=self.active_neighbor_masks(c1,r1,r1_last_mod,last_tested);
            let neighbors_start = copy_at(&self.neighbors_before_offsets, c1);
            let neighbors_end = copy_at(&self.neighbors_before_offsets, c1 + 1);
            let count=neighbors_end-neighbors_start;
            let mut pending=masks.before & if count==64 {u64::MAX} else {(1u64<<count)-1};
            #[cfg(target_arch="x86_64")]
            let (batch_inter,batch_two)=if self.vector_arcs {
                unsafe{self.batch_neighbor_bounds::<THREE>(r1,pos1,&self.neighbors_before[neighbors_start..neighbors_end],pending)}
            } else {(pending,pending)};
            #[cfg(not(target_arch="x86_64"))]
            let (batch_inter,batch_two)=(pending,pending);
            #[cfg(target_arch="x86_64")]
            let batch_reverse=if pos1==1 && self.vector_arcs {
                unsafe{self.batch_reverse_bounds::<THREE>(r1,pos1,&self.neighbors_before[neighbors_start..neighbors_end],pending)}
            } else {pending};
            #[cfg(not(target_arch="x86_64"))]
            let batch_reverse=pending;
            pending &= batch_inter | batch_two | if pos1==1 {batch_reverse} else {0};
            while pending!=0 {
                let k=neighbors_start+pending.trailing_zeros() as usize;pending&=pending-1;
                let c2 = copy_at(&self.neighbors_before, k) as usize;
                let loc2 = copy_at(&self.node_locations,c2);
                let r2 = (loc2.route_and_position>>32) as usize;

                // Skip if both routes unchanged since last tests for this customer
                {
                    // We use pos2 + 1 for the SWAP and RELOCATE moves since c2 is a good predecessor for c1
                    // Moves listed here create the edge c2 => c1, but never insert immediately after a depot
                    let pos2 = loc2.route_and_position as u32 as usize;

                    let delta = if batch_inter&(1u64<<(k-neighbors_start))!=0 {
                        self.run_inter_source::<THREE,TABLE>(r1,pos1,r2,pos2+1,&inter_source)
                    } else {NO_MOVE};
                    if delta < best_delta {
                        best_delta = delta;
                        best_move = Some(CandidateMove::InterRoute {
                            r1,
                            pos1,
                            r2,
                            pos2: pos2 + 1,
                        });
                        best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
                    }

                    // Special case to manage insert immediately after a depot
                    if pos1 == 1 {
                        let delta = if batch_reverse&(1u64<<(k-neighbors_start))!=0 {
                            self.run_inter_mode::<THREE,TABLE>(r2,pos2,r1,pos1)
                        } else {NO_MOVE};
                        if delta < best_delta {
                            best_delta = delta;
                            best_move = Some(CandidateMove::InterRoute {
                                r1: r2,
                                pos1: pos2,
                                r2: r1,
                                pos2: pos1,
                            });
                            best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
                        }
                    }

                    let delta = if batch_two&(1u64<<(k-neighbors_start))!=0 {
                        self.run_2optstar::<TABLE>(r1,pos1,r2,pos2+1)
                    } else {NO_MOVE};
                    if delta < best_delta {
                        best_delta = delta;
                        best_move = Some(CandidateMove::TwoOptStar {
                            r1,
                            pos1,
                            r2,
                            pos2: pos2 + 1,
                        });
                        best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
                    }
                }
            }

            let swap_source=self.make_swap_source(r1,pos1);
            let capacity_start = copy_at(&self.neighbors_capacity_swap_offsets, c1);
            let capacity_end = copy_at(&self.neighbors_capacity_swap_offsets, c1 + 1);
            let mut pending=masks.before.checked_shr((neighbors_end-neighbors_start) as u32).unwrap_or(0);
            while pending!=0 {
                let k=capacity_start+pending.trailing_zeros() as usize;pending&=pending-1;
                let c2 = copy_at(&self.neighbors_capacity_swap, k) as usize;
                let loc2 = copy_at(&self.node_locations,c2);
                let r2 = (loc2.route_and_position>>32) as usize;

                // Skip if both routes unchanged since last tests for this customer
                {
                    let pos2 = loc2.route_and_position as u32 as usize;
                    let delta = self.run_swapstar_source::<TABLE>(r1,pos1,r2,pos2,&swap_source);
                    if delta < best_delta {
                        best_delta = delta;
                        best_move = Some(CandidateMove::SwapStar { r1, pos1, r2, pos2 });
                        best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
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
                        let delta = self.run_2optstar::<TABLE>(r1, pos1, r2, pos2);
                        if delta < best_delta {
                            best_delta = delta;
                            best_move = Some(CandidateMove::TwoOptStar { r1, pos1, r2, pos2 });
                            best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
                        }
                    }

                    // Insert in an empty route
                    let delta = self.run_inter_source::<THREE,TABLE>(r1,pos1,r2,pos2,&inter_source);
                    if delta < best_delta {
                        best_delta = delta;
                        best_move = Some(CandidateMove::InterRoute { r1, pos1, r2, pos2 });
                        best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
                    }
                }
            }
        }

        // Intra-route moves
        if copy_at(&self.when_last_modified, r1) > last_tested {
            let delta = self.run_intra_route_relocate::<TABLE>(r1, pos1);
            if delta < best_delta {
                best_delta = delta;
                best_move = Some(CandidateMove::IntraRelocate { r1, pos1 });
                best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
            }

            let delta = self.run_intra_route_oropt2::<TABLE>(r1, pos1);
            if delta < best_delta {
                best_delta = delta;
                best_move = Some(CandidateMove::IntraOrOpt2 { r1, pos1 });
                best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
            }

            let delta = self.run_intra_route_swap::<TABLE>(r1, pos1);
            if delta < best_delta {
                best_delta = delta;
                best_move = Some(CandidateMove::IntraSwap { r1, pos1 });
                best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
            }

            let delta = self.run_2opt::<TABLE>(r1, pos1);
            if delta < best_delta {
                best_delta = delta;
                best_move = Some(CandidateMove::Intra2Opt { r1, pos1 });
                best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
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

    #[inline(never)]
    fn evaluate_scan_customer<const THREE: bool, const TABLE: bool>(
        &mut self,
        c1: usize,
        last_tested: usize,
        loop_id: usize,
    ) -> i64 {
        if loop_id != 2 && last_tested == self.nb_moves { return NO_MOVE; }
        self.query_credit=self.move_credit;
        let mut best_delta: i64 = i64::MAX; // best acceptable move
        let mut best_move: Option<CandidateMove> = None;
        let mut best_plan = MovePlan::None;
        let loc1 = copy_at(&self.node_locations,c1);
        let r1 = (loc1.route_and_position>>32) as usize;
        let pos1 = loc1.route_and_position as u32 as usize;
        let inter_source=self.make_inter_source::<THREE>(r1,pos1);

        {
            let r1_last_mod = copy_at(&self.when_last_modified,r1);
            let need_r2_stale_check = r1_last_mod <= last_tested;
            let neighbors_start = copy_at(&self.neighbors_before_offsets, c1);
            let neighbors_end = copy_at(&self.neighbors_before_offsets, c1 + 1);
            let mut eligible=0u64;
            for k in neighbors_start..neighbors_end {
                let c2=copy_at(&self.neighbors_before,k) as usize;let loc2=copy_at(&self.node_locations,c2);
                let r2=(loc2.route_and_position>>32) as usize;
                if r1!=r2 && !(need_r2_stale_check && copy_at(&self.when_last_modified,r2)<=last_tested) {eligible |= 1u64<<(k-neighbors_start);}
            }
            #[cfg(target_arch="x86_64")]
            let (batch_inter,batch_two)=unsafe{self.batch_neighbor_bounds::<THREE>(r1,pos1,&self.neighbors_before[neighbors_start..neighbors_end],eligible)};
            #[cfg(not(target_arch="x86_64"))]
            let (batch_inter,batch_two)=(eligible,eligible);
            #[cfg(target_arch="x86_64")]
            let batch_reverse=if pos1==1 {unsafe{self.batch_reverse_bounds::<THREE>(r1,pos1,&self.neighbors_before[neighbors_start..neighbors_end],eligible)}} else {0};
            #[cfg(not(target_arch="x86_64"))]
            let batch_reverse=eligible;
            let mut pending=eligible & (batch_inter | batch_two | batch_reverse);
            while pending!=0 {
                let k=neighbors_start+pending.trailing_zeros() as usize;pending&=pending-1;
                let c2 = copy_at(&self.neighbors_before, k) as usize;
                let loc2 = copy_at(&self.node_locations,c2);
                let r2 = (loc2.route_and_position>>32) as usize;

                // Skip if both routes unchanged since last tests for this customer
                {
                    // We use pos2 + 1 for the SWAP and RELOCATE moves since c2 is a good predecessor for c1
                    // Moves listed here create the edge c2 => c1, but never insert immediately after a depot
                    let pos2 = loc2.route_and_position as u32 as usize;

                    let delta = if batch_inter&(1u64<<(k-neighbors_start))!=0 {self.run_inter_source::<THREE,TABLE>(r1,pos1,r2,pos2 + 1,&inter_source)} else {NO_MOVE};
                    if delta < best_delta {
                        best_delta = delta;
                        best_move = Some(CandidateMove::InterRoute {
                            r1,
                            pos1,
                            r2,
                            pos2: pos2 + 1,
                        });
                        best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
                    }

                    // Special case to manage insert immediately after a depot
                    if pos1 == 1 {
                        let delta = if batch_reverse&(1u64<<(k-neighbors_start))!=0 {self.run_inter_mode::<THREE,TABLE>(r2, pos2, r1, pos1)} else {NO_MOVE};
                        if delta < best_delta {
                            best_delta = delta;
                            best_move = Some(CandidateMove::InterRoute {
                                r1: r2,
                                pos1: pos2,
                                r2: r1,
                                pos2: pos1,
                            });
                            best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
                        }
                    }

                    let delta = if batch_two&(1u64<<(k-neighbors_start))!=0 {self.run_2optstar::<TABLE>(r1, pos1, r2, pos2 + 1)} else {NO_MOVE};
                    if delta < best_delta {
                        best_delta = delta;
                        best_move = Some(CandidateMove::TwoOptStar {
                            r1,
                            pos1,
                            r2,
                            pos2: pos2 + 1,
                        });
                        best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
                    }
                }
            }

            let swap_source=self.make_swap_source(r1,pos1);
            let capacity_start = copy_at(&self.neighbors_capacity_swap_offsets, c1);
            let capacity_end = copy_at(&self.neighbors_capacity_swap_offsets, c1 + 1);
            for k in capacity_start..capacity_end {
                let c2 = copy_at(&self.neighbors_capacity_swap, k) as usize;
                let loc2 = copy_at(&self.node_locations,c2);
                let r2 = (loc2.route_and_position>>32) as usize;

                // Skip if both routes unchanged since last tests for this customer
                if r1 != r2
                    && !(need_r2_stale_check
                        && copy_at(&self.when_last_modified,r2) <= last_tested)
                {
                    let pos2 = loc2.route_and_position as u32 as usize;
                    let delta = self.run_swapstar_source::<TABLE>(r1,pos1,r2,pos2,&swap_source);
                    if delta < best_delta {
                        best_delta = delta;
                        best_move = Some(CandidateMove::SwapStar { r1, pos1, r2, pos2 });
                        best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
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
                        let delta = self.run_2optstar::<TABLE>(r1, pos1, r2, pos2);
                        if delta < best_delta {
                            best_delta = delta;
                            best_move = Some(CandidateMove::TwoOptStar { r1, pos1, r2, pos2 });
                            best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
                        }
                    }

                    // Insert in an empty route
                    let delta = self.run_inter_source::<THREE,TABLE>(r1,pos1,r2,pos2,&inter_source);
                    if delta < best_delta {
                        best_delta = delta;
                        best_move = Some(CandidateMove::InterRoute { r1, pos1, r2, pos2 });
                        best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
                    }
                }
            }
        }

        // Intra-route moves
        if copy_at(&self.when_last_modified, r1) > last_tested {
            let delta = self.run_intra_route_relocate::<TABLE>(r1, pos1);
            if delta < best_delta {
                best_delta = delta;
                best_move = Some(CandidateMove::IntraRelocate { r1, pos1 });
                best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
            }

            let delta = self.run_intra_route_oropt2::<TABLE>(r1, pos1);
            if delta < best_delta {
                best_delta = delta;
                best_move = Some(CandidateMove::IntraOrOpt2 { r1, pos1 });
                best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
            }

            let delta = self.run_intra_route_swap::<TABLE>(r1, pos1);
            if delta < best_delta {
                best_delta = delta;
                best_move = Some(CandidateMove::IntraSwap { r1, pos1 });
                best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
            }

            let delta = self.run_2opt::<TABLE>(r1, pos1);
            if delta < best_delta {
                best_delta = delta;
                best_move = Some(CandidateMove::Intra2Opt { r1, pos1 });
                best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
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

    #[inline(never)]
    fn evaluate_scalar_customer<const THREE: bool, const TABLE: bool>(
        &mut self,
        c1: usize,
        last_tested: usize,
        loop_id: usize,
    ) -> i64 {
        if loop_id != 2 && last_tested == self.nb_moves { return NO_MOVE; }
        self.query_credit=self.move_credit;
        let mut best_delta: i64 = i64::MAX; // best acceptable move
        let mut best_move: Option<CandidateMove> = None;
        let mut best_plan = MovePlan::None;
        let loc1 = copy_at(&self.node_locations,c1);
        let r1 = (loc1.route_and_position>>32) as usize;
        let pos1 = loc1.route_and_position as u32 as usize;
        let inter_source=self.make_inter_source::<THREE>(r1,pos1);

        {
            let r1_last_mod = copy_at(&self.when_last_modified,r1);
            let need_r2_stale_check = r1_last_mod <= last_tested;
            let neighbors_start = copy_at(&self.neighbors_before_offsets, c1);
            let neighbors_end = copy_at(&self.neighbors_before_offsets, c1 + 1);
            for k in neighbors_start..neighbors_end {
                let c2 = copy_at(&self.neighbors_before, k) as usize;
                let loc2 = copy_at(&self.node_locations,c2);
                let r2 = (loc2.route_and_position>>32) as usize;

                // Skip if both routes unchanged since last tests for this customer
                if r1 != r2
                    && !(need_r2_stale_check
                        && copy_at(&self.when_last_modified,r2) <= last_tested)
                {
                    // We use pos2 + 1 for the SWAP and RELOCATE moves since c2 is a good predecessor for c1
                    // Moves listed here create the edge c2 => c1, but never insert immediately after a depot
                    let pos2 = loc2.route_and_position as u32 as usize;

                    let delta = self.run_inter_source::<THREE,TABLE>(r1,pos1,r2,pos2 + 1,&inter_source);
                    if delta < best_delta {
                        best_delta = delta;
                        best_move = Some(CandidateMove::InterRoute {
                            r1,
                            pos1,
                            r2,
                            pos2: pos2 + 1,
                        });
                        best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
                    }

                    // Special case to manage insert immediately after a depot
                    if pos1 == 1 {
                        let delta = self.run_inter_mode::<THREE,TABLE>(r2, pos2, r1, pos1);
                        if delta < best_delta {
                            best_delta = delta;
                            best_move = Some(CandidateMove::InterRoute {
                                r1: r2,
                                pos1: pos2,
                                r2: r1,
                                pos2: pos1,
                            });
                            best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
                        }
                    }

                    let delta = self.run_2optstar::<TABLE>(r1, pos1, r2, pos2 + 1);
                    if delta < best_delta {
                        best_delta = delta;
                        best_move = Some(CandidateMove::TwoOptStar {
                            r1,
                            pos1,
                            r2,
                            pos2: pos2 + 1,
                        });
                        best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
                    }
                }
            }

            let swap_source=self.make_swap_source(r1,pos1);
            let capacity_start = copy_at(&self.neighbors_capacity_swap_offsets, c1);
            let capacity_end = copy_at(&self.neighbors_capacity_swap_offsets, c1 + 1);
            for k in capacity_start..capacity_end {
                let c2 = copy_at(&self.neighbors_capacity_swap, k) as usize;
                let loc2 = copy_at(&self.node_locations,c2);
                let r2 = (loc2.route_and_position>>32) as usize;

                // Skip if both routes unchanged since last tests for this customer
                if r1 != r2
                    && !(need_r2_stale_check
                        && copy_at(&self.when_last_modified,r2) <= last_tested)
                {
                    let pos2 = loc2.route_and_position as u32 as usize;
                    let delta = self.run_swapstar_source::<TABLE>(r1,pos1,r2,pos2,&swap_source);
                    if delta < best_delta {
                        best_delta = delta;
                        best_move = Some(CandidateMove::SwapStar { r1, pos1, r2, pos2 });
                        best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
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
                        let delta = self.run_2optstar::<TABLE>(r1, pos1, r2, pos2);
                        if delta < best_delta {
                            best_delta = delta;
                            best_move = Some(CandidateMove::TwoOptStar { r1, pos1, r2, pos2 });
                            best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
                        }
                    }

                    // Insert in an empty route
                    let delta = self.run_inter_source::<THREE,TABLE>(r1,pos1,r2,pos2,&inter_source);
                    if delta < best_delta {
                        best_delta = delta;
                        best_move = Some(CandidateMove::InterRoute { r1, pos1, r2, pos2 });
                        best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
                    }
                }
            }
        }

        // Intra-route moves
        if copy_at(&self.when_last_modified, r1) > last_tested {
            let delta = self.run_intra_route_relocate::<TABLE>(r1, pos1);
            if delta < best_delta {
                best_delta = delta;
                best_move = Some(CandidateMove::IntraRelocate { r1, pos1 });
                best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
            }

            let delta = self.run_intra_route_oropt2::<TABLE>(r1, pos1);
            if delta < best_delta {
                best_delta = delta;
                best_move = Some(CandidateMove::IntraOrOpt2 { r1, pos1 });
                best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
            }

            let delta = self.run_intra_route_swap::<TABLE>(r1, pos1);
            if delta < best_delta {
                best_delta = delta;
                best_move = Some(CandidateMove::IntraSwap { r1, pos1 });
                best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
            }

            let delta = self.run_2opt::<TABLE>(r1, pos1);
            if delta < best_delta {
                best_delta = delta;
                best_move = Some(CandidateMove::Intra2Opt { r1, pos1 });
                best_plan = self.last_plan;
                        if self.factored_bounds_valid {self.query_credit=delta.saturating_sub(1);}
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
        self.reset_recent_routes();
        if self.capacity_costs.is_empty() {
            if self.params.allow_swap3 { self.search_mode::<true,false>(rng); }
            else { self.search_mode::<false,false>(rng); }
        } else {
            if self.params.allow_swap3 { self.search_mode::<true,true>(rng); }
            else { self.search_mode::<false,true>(rng); }
        }
    }
    fn search_mode<const THREE: bool, const TABLE: bool>(&mut self, rng: &mut SmallRng) {
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
                        self.evaluate_and_apply_best_move_for_customer::<THREE,TABLE>(c1, last_tested, loop_id);
                    if delta != NO_MOVE {
                        improved = true;
                        c1_repeat = delta < 0;
                    }
                }
            }
        }
    }
    #[inline(always)]
    fn run_inter_source_assured<const THREE:bool,const TABLE:bool>(&mut self,r1:usize,p1:usize,r2:usize,p2:usize,src:&InterSource)->i64 {
        let fast=if src.singleton {self.run_singleton_inter::<TABLE>(r1,r2,p2)}
            else {self.run_inter_source_impl_assured::<THREE,TABLE>(r1,p1,r2,p2,src)};
        // CHECK_INTER_SOURCE
        fast
    }
    #[inline(always)]
    fn run_inter_source_impl_assured<const THREE: bool, const TABLE: bool>(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize,src:&InterSource) -> i64 {
        if true {self.run_inter_source_impl_keyed_assured::<THREE,TABLE,true>(r1,pos1,r2,pos2,src)}else{self.run_inter_source_impl_keyed_assured::<THREE,TABLE,false>(r1,pos1,r2,pos2,src)}
    }
    #[inline(always)]
    fn run_inter_mode_assured<const THREE: bool, const TABLE: bool>(&mut self,r1:usize,pos1:usize,r2:usize,pos2:usize)->i64 {
        if route_at(&self.routes,r1).nodes.len()==3 { self.run_singleton_inter::<TABLE>(r1,r2,pos2) }
        else { self.run_inter_special_assured::<THREE,TABLE>(r1,pos1,r2,pos2) }
    }
    #[inline(always)]
    fn run_inter_special_assured<const THREE: bool, const TABLE: bool>(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize) -> i64 {
        if true {self.run_inter_special_keyed_assured::<THREE,TABLE,true>(r1,pos1,r2,pos2)}else{self.run_inter_special_keyed_assured::<THREE,TABLE,false>(r1,pos1,r2,pos2)}
    }
    #[inline(always)]
    fn run_swapstar_source_assured<const TABLE:bool>(&mut self,r1:usize,p1:usize,r2:usize,p2:usize,src:&SwapSource)->i64 {
        self.run_swapstar_source_fast_assured::<TABLE>(r1,p1,r2,p2,src)
    }
    #[inline(always)]
    fn run_swapstar_source_fast_assured<const TABLE:bool>(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize,src:&SwapSource) -> i64 {
        if src.singleton && route_at(&self.routes,r2).nodes.len()==3 {
            if self.query_credit<0 { return NO_MOVE; }
            self.last_plan=MovePlan::SwapStar{insert1:1,insert2:1};
            return 0;
        }

        debug_assert!(r1 != r2);
        debug_assert!(pos1 > 0 && pos1 < self.routes[r1].nodes.len() - 1);
        debug_assert!(pos2 > 0 && pos2 < self.routes[r2].nodes.len() - 1);

        // The old code materialized a complete prefix/suffix Sequence for
        // every customer on every route refresh, although swap-star only
        // needs its load and distance here.  Removing one customer has an
        // exact three-arc delta, so compute those two scalars on demand.
        let route2=route_at(&self.routes,r2);let node_v=route2.node(pos2);
        let u=src.id;let v=node_v.id;
        let removed_distance1=src.removed_distance;
        let removed_distance2=route2.distance+node_v.removal_delta;
        let new_load1=src.remaining_load+node_v.load;
        let new_load2=route2.load-node_v.load+src.load;
        let old_total=src.cost+route2.cost;
        let bridge_u=src.bridge;let bridge_v=node_v.bridge;

        // First filter on route costs
        let new_pen1=if TABLE {copy_at(&self.capacity_costs,new_load1 as usize)}
            else {((new_load1-self.data.max_capacity).max(0) as i64)*self.params.penalty_capa as i64};
        let new_pen2=if TABLE {copy_at(&self.capacity_costs,new_load2 as usize)}
            else {((new_load2-self.data.max_capacity).max(0) as i64)*self.params.penalty_capa as i64};
        let cost_lb_r1_after_removal = (removed_distance1 as i64) + new_pen1;
        let cost_lb_r2_after_removal = (removed_distance2 as i64) + new_pen2;
        let mut lb_new_total = cost_lb_r1_after_removal + cost_lb_r2_after_removal;
        let max_acceptable_cost = src.max_cost+route2.cost;

        // first filter on route costs
        if lb_new_total > max_acceptable_cost {
            return NO_MOVE;
        }

        let data = self.data.as_ref();
        let route1_nodes = &route_at(&self.routes, r1).nodes;
        let route2_nodes = &route_at(&self.routes, r2).nodes;
        let route1_len = route1_nodes.len();
        let route2_len = route2_nodes.len();
        let distance_from_u = unsafe{data.distance_matrix.get_unchecked(src.row_start as usize..src.row_start as usize+data.nb_nodes)};
        let distance_to_u = unsafe{data.distance_matrix_transposed.get_unchecked(src.row_start as usize..src.row_start as usize+data.nb_nodes)};
        let distance_from_v = data.distance_row(v);
        let distance_to_v = data.distance_column(v);

        // Minimum distance detour for reinserting V after removing U.  The
        // two edges incident to U disappear; include the replacement bridge
        // explicitly and scan the remaining original edges.
        let pred_u = node_at(route1_nodes, pos1 - 1);
        let next_u = node_at(route1_nodes, pos1 + 1);
        let mut best_ins_v =
            copy_at(distance_to_v, pred_u.id) + copy_at(distance_from_v, next_u.id) - bridge_u;
        best_ins_v=best_ins_v.min(Self::route_insertion_lower_bound_assured::<false>(data,route_at(&self.routes,r1),
            unsafe{self.insertion_bounds.get_unchecked_mut(r1*data.nb_nodes+v)},distance_from_v,distance_to_v));

        // Second filter on route costs
        lb_new_total += best_ins_v as i64;
        if lb_new_total > max_acceptable_cost {
            return NO_MOVE;
        }

        let pred_v = node_at(route2_nodes, pos2 - 1);
        let next_v = node_at(route2_nodes, pos2 + 1);
        let mut best_ins_u =
            copy_at(distance_to_u, pred_v.id) + copy_at(distance_from_u, next_v.id) - bridge_v;
        best_ins_u=best_ins_u.min(Self::route_insertion_lower_bound_assured::<false>(data,route_at(&self.routes,r2),
            unsafe{self.insertion_bounds.get_unchecked_mut(r2*data.nb_nodes+u)},distance_from_u,distance_to_u));

        let search_limit=old_total+self.query_credit;
        // Third filter on route costs
        lb_new_total += best_ins_u as i64;
        if lb_new_total > search_limit {
            return NO_MOVE;
        }

        // Each insertion detour is at least best_ins. If detour plus
        // service is nonnegative, deleting that inserted visit from any
        // schedule cannot increase total warp. Thus the exact removed-route
        // warp is a lower bound for every insertion location. This uses the
        // actual arc minimum; no metric-distance assumption is required.
        let warp_floor1=if true && best_ins_v as i64+node_at(route2_nodes,pos2).t1.duration as i64>=0 {
            pred_u.prefix.warp+next_u.suffix.warp+(pred_u.prefix.time+bridge_u-next_u.suffix.time).max(0)
        } else {0};
        let warp_floor2=if true && best_ins_u as i64+node_at(route1_nodes,pos1).t1.duration as i64>=0 {
            pred_v.prefix.warp+next_v.suffix.warp+(pred_v.prefix.time+bridge_v-next_v.suffix.time).max(0)
        } else {0};
        let warp_charge1=warp_floor1 as i64*self.params.penalty_tw as i64;
        let warp_charge2=warp_floor2 as i64*self.params.penalty_tw as i64;
        if lb_new_total+warp_charge1+warp_charge2>search_limit {
            return NO_MOVE;
        }

        // Exact TW evaluation for one route cannot rescue a distance/capacity
        // lower bound that already exceeds the accepted total.  Keep the
        // other route's tight insertion lower bound available in both scans.
        let lb_cost2 = cost_lb_r2_after_removal + best_ins_u as i64 + warp_charge2;

        // Values needed only beyond the distance/capacity filters.
        let ptw = self.params.penalty_tw as i64;

        let inserted=node_at(route2_nodes,pos2).seq1();
        let cache=unsafe{self.removal_cache.get_unchecked_mut(u)};
        if cache.version!=route_at(&self.routes,r1).insertion_version {
            Self::prepare_removal_cache(route_at(&self.routes,r1),pos1,cache);
        }
        let mut best_t1=pos1;let mut best_cost1=i64::MAX/4;
        for entry in &cache.entries {
            let incoming=copy_at(distance_to_v,entry.pred as usize);
            let outgoing=copy_at(distance_from_v,entry.next as usize);
            let route_lb=(entry.distance+incoming+outgoing) as i64+new_pen1;
            if route_lb+warp_charge1<best_cost1 && route_lb+warp_charge1+lb_cost2<=search_limit {
                let arrival=entry.prefix_end+incoming;
                let first=(arrival-inserted.tau_plus).max(0);
                let last=inserted.earliest_end.max(arrival+inserted.duration_net)+outgoing-entry.suffix_latest;
                let tw=entry.warp+first.max(last);
                let cost=route_lb+tw as i64*ptw;
                if cost<best_cost1 {best_cost1=cost;best_t1=entry.position as usize;}
            }
        }

        // Fourth filter: one route is exact (TW-aware), the other remains a lower bound
        if best_cost1.saturating_add(lb_cost2) > search_limit {
            return NO_MOVE;
        }

        let inserted=node_at(route1_nodes,pos1).seq1();
        let cache=unsafe{self.removal_cache.get_unchecked_mut(v)};
        if cache.version!=route_at(&self.routes,r2).insertion_version {
            Self::prepare_removal_cache(route_at(&self.routes,r2),pos2,cache);
        }
        let mut best_t2=pos2;let mut best_cost2=i64::MAX/4;
        for entry in &cache.entries {
            let incoming=copy_at(distance_to_u,entry.pred as usize);
            let outgoing=copy_at(distance_from_u,entry.next as usize);
            let route_lb=(entry.distance+incoming+outgoing) as i64+new_pen2;
            if route_lb+warp_charge2<best_cost2 && best_cost1+route_lb+warp_charge2<=search_limit {
                let arrival=entry.prefix_end+incoming;
                let first=(arrival-inserted.tau_plus).max(0);
                let last=inserted.earliest_end.max(arrival+inserted.duration_net)+outgoing-entry.suffix_latest;
                let tw=entry.warp+first.max(last);
                let cost=route_lb+tw as i64*ptw;
                if cost<best_cost2 {best_cost2=cost;best_t2=entry.position as usize;}
            }
        }

        let new_total = best_cost1 + best_cost2;
        if new_total > search_limit {
            return NO_MOVE;
        }
        self.last_plan = MovePlan::SwapStar {
            insert1: best_t1,
            insert2: best_t2,
        };
        // A found feasible total proves that the original third and fourth
        // lower bounds pass. The second historical gate still needs auditing:
        // it omits the possibly negative detour of the other insertion.
        if !true || false || false {
            return self.run_swapstar_exact(r1,pos1,r2,pos2);
        }
        if best_cost1+cost_lb_r2_after_removal>max_acceptable_cost {
            let threshold=max_acceptable_cost-cost_lb_r1_after_removal-cost_lb_r2_after_removal;
            if !Self::audit_insertion_gate(route1_nodes,pos1,distance_from_v,distance_to_v,bridge_u,threshold) {return NO_MOVE;}
        }
        new_total-old_total
    }
    #[inline(always)]
    fn run_swapstar_fused_cut<const TABLE:bool,const MINPOS:bool>(&mut self,r1:usize,pos1:usize,r2:usize,pos2:usize,v:usize,src:&SwapSource)->i64 {
        
        let delta=self.run_swapstar_fused_cut_impl::<TABLE,MINPOS>(r1,pos1,r2,pos2,v,src);
        
        delta
    }
    #[inline(always)]
    fn run_swapstar_fused_cut_impl<const TABLE:bool,const MINPOS:bool>(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize,v:usize,src:&SwapSource) -> i64 {
        let cut=unsafe{self.batch_cuts.get_unchecked(v)};
        let target_cost=route_at(&self.routes,r2).cost;
        if src.singleton && cut.0[11]==0 && cut.0[3]==0 {
            if self.query_credit<0 { return NO_MOVE; }
            self.last_plan=MovePlan::SwapStar{insert1:1,insert2:1};
            return 0;
        }

        debug_assert!(r1 != r2);
        debug_assert!(pos1 > 0 && pos1 < self.routes[r1].nodes.len() - 1);
        debug_assert!(pos2 > 0 && pos2 < self.routes[r2].nodes.len() - 1);

        // The old code materialized a complete prefix/suffix Sequence for
        // every customer on every route refresh, although swap-star only
        // needs its load and distance here.  Removing one customer has an
        // exact three-arc delta, so compute those two scalars on demand.
        let u=src.id;
        let removed_distance1=src.removed_distance;
        let removed_distance2=cut.0[1]+cut.0[15];
        let new_load1=src.remaining_load+cut.0[14];
        let new_load2=cut.0[0]-cut.0[14]+src.load;
        let old_total=src.cost+target_cost;
        let bridge_u=src.bridge;let bridge_v=cut.0[13];

        // First filter on route costs
        let new_pen1=if TABLE {copy_at(&self.capacity_costs,new_load1 as usize)}
            else {((new_load1-self.data.max_capacity).max(0) as i64)*self.params.penalty_capa as i64};
        let new_pen2=if TABLE {copy_at(&self.capacity_costs,new_load2 as usize)}
            else {((new_load2-self.data.max_capacity).max(0) as i64)*self.params.penalty_capa as i64};
        let cost_lb_r1_after_removal = (removed_distance1 as i64) + new_pen1;
        let cost_lb_r2_after_removal = (removed_distance2 as i64) + new_pen2;
        let mut lb_new_total = cost_lb_r1_after_removal + cost_lb_r2_after_removal;
        let max_acceptable_cost = src.max_cost+target_cost;

        // first filter on route costs
        if lb_new_total > max_acceptable_cost {
            return NO_MOVE;
        }

        let data = self.data.as_ref();
        let distance_from_v = data.distance_row(v);
        let distance_to_v = data.distance_column(v);

        // Minimum distance detour for reinserting V after removing U.  The
        // two edges incident to U disappear; include the replacement bridge
        // explicitly and scan the remaining original edges.
        let mut best_ins_v =
            copy_at(distance_to_v, src.pred) + copy_at(distance_from_v, src.next) - bridge_u;
        best_ins_v=best_ins_v.min(Self::route_insertion_lower_bound_assured::<MINPOS>(data,route_at(&self.routes,r1),
            unsafe{self.insertion_bounds.get_unchecked_mut(r1*data.nb_nodes+v)},distance_from_v,distance_to_v));

        // Second filter on route costs
        lb_new_total += best_ins_v as i64;
        if lb_new_total > max_acceptable_cost {
            return NO_MOVE;
        }

        let distance_from_u = unsafe{data.distance_matrix.get_unchecked(src.row_start as usize..src.row_start as usize+data.nb_nodes)};
        let distance_to_u = unsafe{data.distance_matrix_transposed.get_unchecked(src.row_start as usize..src.row_start as usize+data.nb_nodes)};
        let mut best_ins_u =
            copy_at(distance_to_u, cut.0[11] as usize) + copy_at(distance_from_u, cut.0[3] as usize) - bridge_v;
        best_ins_u=best_ins_u.min(Self::route_insertion_lower_bound_assured::<MINPOS>(data,route_at(&self.routes,r2),
            unsafe{self.insertion_bounds.get_unchecked_mut(r2*data.nb_nodes+u)},distance_from_u,distance_to_u));

        let search_limit=old_total+self.query_credit;
        // Third filter on route costs
        lb_new_total += best_ins_u as i64;
        if lb_new_total > search_limit {
            return NO_MOVE;
        }

        let route1_nodes = &route_at(&self.routes, r1).nodes;
        let route2_nodes = &route_at(&self.routes, r2).nodes;
        let route1_len = route1_nodes.len();
        let route2_len = route2_nodes.len();
        let pred_u = node_at(route1_nodes, pos1 - 1);
        let next_u = node_at(route1_nodes, pos1 + 1);
        let pred_v = node_at(route2_nodes, pos2 - 1);
        let next_v = node_at(route2_nodes, pos2 + 1);
        // Each insertion detour is at least best_ins. If detour plus
        // service is nonnegative, deleting that inserted visit from any
        // schedule cannot increase total warp. Thus the exact removed-route
        // warp is a lower bound for every insertion location. This uses the
        // actual arc minimum; no metric-distance assumption is required.
        let warp_floor1=if true && best_ins_v as i64+node_at(route2_nodes,pos2).t1.duration as i64>=0 {
            pred_u.prefix.warp+next_u.suffix.warp+(pred_u.prefix.time+bridge_u-next_u.suffix.time).max(0)
        } else {0};
        let warp_floor2=if true && best_ins_u as i64+node_at(route1_nodes,pos1).t1.duration as i64>=0 {
            pred_v.prefix.warp+next_v.suffix.warp+(pred_v.prefix.time+bridge_v-next_v.suffix.time).max(0)
        } else {0};
        let warp_charge1=warp_floor1 as i64*self.params.penalty_tw as i64;
        let warp_charge2=warp_floor2 as i64*self.params.penalty_tw as i64;
        if lb_new_total+warp_charge1+warp_charge2>search_limit {
            return NO_MOVE;
        }

        // Exact TW evaluation for one route cannot rescue a distance/capacity
        // lower bound that already exceeds the accepted total.  Keep the
        // other route's tight insertion lower bound available in both scans.
        let lb_cost2 = cost_lb_r2_after_removal + best_ins_u as i64 + warp_charge2;

        // Values needed only beyond the distance/capacity filters.
        let ptw = self.params.penalty_tw as i64;

        let inserted=node_at(route2_nodes,pos2).seq1();
        let cache=unsafe{self.removal_cache.get_unchecked_mut(u)};
        if cache.version!=route_at(&self.routes,r1).insertion_version {
            Self::prepare_removal_cache(route_at(&self.routes,r1),pos1,cache);
        }
        let mut best_t1=pos1;let mut best_cost1=i64::MAX/4;
        for entry in &cache.entries {
            let incoming=copy_at(distance_to_v,entry.pred as usize);
            let outgoing=copy_at(distance_from_v,entry.next as usize);
            let route_lb=(entry.distance+incoming+outgoing) as i64+new_pen1;
            if route_lb+warp_charge1<best_cost1 && route_lb+warp_charge1+lb_cost2<=search_limit {
                let arrival=entry.prefix_end+incoming;
                let first=(arrival-inserted.tau_plus).max(0);
                let last=inserted.earliest_end.max(arrival+inserted.duration_net)+outgoing-entry.suffix_latest;
                let tw=entry.warp+first.max(last);
                let cost=route_lb+tw as i64*ptw;
                if cost<best_cost1 {best_cost1=cost;best_t1=entry.position as usize;}
            }
        }

        // Fourth filter: one route is exact (TW-aware), the other remains a lower bound
        if best_cost1.saturating_add(lb_cost2) > search_limit {
            return NO_MOVE;
        }

        let inserted=node_at(route1_nodes,pos1).seq1();
        let cache=unsafe{self.removal_cache.get_unchecked_mut(v)};
        if cache.version!=route_at(&self.routes,r2).insertion_version {
            Self::prepare_removal_cache(route_at(&self.routes,r2),pos2,cache);
        }
        let mut best_t2=pos2;let mut best_cost2=i64::MAX/4;
        for entry in &cache.entries {
            let incoming=copy_at(distance_to_u,entry.pred as usize);
            let outgoing=copy_at(distance_from_u,entry.next as usize);
            let route_lb=(entry.distance+incoming+outgoing) as i64+new_pen2;
            if route_lb+warp_charge2<best_cost2 && best_cost1+route_lb+warp_charge2<=search_limit {
                let arrival=entry.prefix_end+incoming;
                let first=(arrival-inserted.tau_plus).max(0);
                let last=inserted.earliest_end.max(arrival+inserted.duration_net)+outgoing-entry.suffix_latest;
                let tw=entry.warp+first.max(last);
                let cost=route_lb+tw as i64*ptw;
                if cost<best_cost2 {best_cost2=cost;best_t2=entry.position as usize;}
            }
        }

        let new_total = best_cost1 + best_cost2;
        if new_total > search_limit {
            return NO_MOVE;
        }
        self.last_plan = MovePlan::SwapStar {
            insert1: best_t1,
            insert2: best_t2,
        };
        // A found feasible total proves that the original third and fourth
        // lower bounds pass. The second historical gate still needs auditing:
        // it omits the possibly negative detour of the other insertion.
        if !true || false || false {
            return self.run_swapstar_exact(r1,pos1,r2,pos2);
        }
        if best_cost1+cost_lb_r2_after_removal>max_acceptable_cost {
            let threshold=max_acceptable_cost-cost_lb_r1_after_removal-cost_lb_r2_after_removal;
            if !Self::audit_insertion_gate(route1_nodes,pos1,distance_from_v,distance_to_v,bridge_u,threshold) {return NO_MOVE;}
        }
        new_total-old_total
    }
    #[inline(always)]
    fn run_intra_route_relocate_assured<const TABLE:bool>(&mut self, r1: usize, pos1: usize) -> i64 {
        let route = route_at(&self.routes, r1);
        let data = self.data.as_ref();
        let len = route.nodes.len();
        if len <= 3 {
            return NO_MOVE;
        } // no alternative insertion for single-client routes

        debug_assert!(pos1 > 0); // U is a client
        debug_assert!(self.routes[r1].nodes[pos1].id != 0); // U is a client

        let old_cost = route.cost;
        let max_acceptable_cost = old_cost + self.query_credit;
        let mut best_cost = i64::MAX;
        let mut best_pos: Option<usize> = None;
        let u_seq1 = route.node(pos1).seq1();
        let old_distance = route.distance as i64;
        let cap_pen =
            if TABLE {copy_at(&self.capacity_costs,route.load as usize)}
            else {((route.load-self.data.max_capacity).max(0) as i64)*self.params.penalty_capa as i64};
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
        #[cfg(target_arch="x86_64")]
        if true && max_distance_acceptable-old_distance>= -67_108_864 && max_distance_acceptable-old_distance<=67_108_864 {
            
            if !unsafe{Self::intra_arc_possible::<false>(route,pos1,distance_to_u,distance_from_u,distance_to_u,distance_from_u,fixed_delta,fixed_delta,(max_distance_acceptable-old_distance) as i32)} {
                
                return NO_MOVE;
            }
        }


        // Insert U before t in [1 .. pos1-1]
        if pos1 > 1 {
            let mut right_excl_u = route.node(pos1 + 1).seqi_n();
            for t in (1..pos1).rev() {
                let join_travel = if t + 1 == pos1 {
                    add_ab
                } else {
                    route.node(t).dist_to_succ
                };
                right_excl_u =
                    Sequence::join_tw_with_travel(&route.node(t).seq1(), &right_excl_u, join_travel);
                let c_id = route.node(t - 1).id;
                let d_id = route.node(t).id;
                let d_cu = copy_at(distance_to_u, c_id);
                let d_ud = copy_at(distance_from_u, d_id);
                let delta_distance = fixed_delta + d_cu + d_ud - route.node(t - 1).dist_to_succ;
                if old_distance + (delta_distance as i64) > max_distance_acceptable {
                    continue;
                }
                let left = route.node(t - 1).seq0_i();
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
                &route.node(pos1 - 1).seq0_i(),
                &route.node(pos1 + 1).seq1(),
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
                        &route.node(t).seqi_n(),
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
                        &route.node(t).seq1(),
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
    #[inline(always)]
    fn run_intra_route_oropt2_assured<const TABLE:bool>(&mut self, r1: usize, pos1: usize) -> i64 {
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
        let max_acceptable_cost = old_cost + self.query_credit;
        let mut best_cost = i64::MAX;
        let mut best_pos: Option<usize> = None;
        let mut best_reversed = false;
        let pair_fwd = route.node(pos1).seq12();
        let pair_rev = route.node(pos1).seq21();
        let old_distance = route.distance as i64;
        let cap_pen =
            if TABLE {copy_at(&self.capacity_costs,route.load as usize)}
            else {((route.load-self.data.max_capacity).max(0) as i64)*self.params.penalty_capa as i64};
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
        #[cfg(target_arch="x86_64")]
        if true && max_distance_acceptable-old_distance>= -67_108_864 && max_distance_acceptable-old_distance<=67_108_864 {
            
            if !unsafe{Self::intra_arc_possible::<true>(route,pos1,distance_to_u,distance_from_u,distance_to_x,distance_from_x,fixed_delta_fwd,fixed_delta_rev,(max_distance_acceptable-old_distance) as i32)} {
                
                return NO_MOVE;
            }
        }


        // Insert (U,X) or (X,U) before t in [1 .. pos1-1]
        if pos1 > 1 {
            let mut right_excl_pair = route.node(pos1 + 2).seqi_n();
            for t in (1..pos1).rev() {
                let join_travel = if t + 1 == pos1 {
                    add_ab
                } else {
                    route.node(t).dist_to_succ
                };
                right_excl_pair = Sequence::join_tw_with_travel(
                    &route.node(t).seq1(),
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
                let left = route.node(t - 1).seq0_i();

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
                &route.node(pos1 - 1).seq0_i(),
                &route.node(pos1 + 2).seq1(),
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
                let right = route.node(t).seqi_n();

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
                        &route.node(t).seq1(),
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
    #[inline(always)]
    fn run_intra_route_swap_assured<const TABLE:bool>(&mut self, r1: usize, pos1: usize) -> i64 {
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
        let max_acceptable_cost = old_cost + self.query_credit;
        let mut best_cost = i64::MAX;
        let mut best_pos: Option<usize> = None;
        let old_distance = route.distance as i64;
        let cap_pen =
            if TABLE {copy_at(&self.capacity_costs,route.load as usize)}
            else {((route.load-self.data.max_capacity).max(0) as i64)*self.params.penalty_capa as i64};
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
            let mut acc_mid = route.node(pos1 + 1).seq1();
            for middle in (pos1 + 2)..first_pos2 {
                acc_mid = Sequence::join_tw_with_travel(
                    &acc_mid,
                    &route.node(middle).seq1(),
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
                        &route.node(pos1 - 1).seq0_i(),
                        &route.node(pos2).seq1(),
                        &acc_mid,
                        &route.node(pos1).seq1(),
                        &route.node(pos2 + 1).seqi_n(),
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
                    &route.node(pos2).seq1(),
                    route.node(pos2 - 1).dist_to_succ,
                );
            }
        }

        if let Some(first_pos2) = first_left_potential {
            let mut acc_mid = route.node(pos1 - 1).seq1();
            for middle in ((first_pos2 + 1)..=(pos1 - 2)).rev() {
                acc_mid = Sequence::join_tw_with_travel(
                    &route.node(middle).seq1(),
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
                        &route.node(pos2 - 1).seq0_i(),
                        &route.node(pos1).seq1(),
                        &acc_mid,
                        &route.node(pos2).seq1(),
                        &route.node(pos1 + 1).seqi_n(),
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
                        &route.node(pos2).seq1(),
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
    #[inline(always)]
    fn run_2opt_assured<const TABLE:bool>(&mut self, r1: usize, pos1: usize) -> i64 {
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
            if TABLE {copy_at(&self.capacity_costs,route.load as usize)}
            else {((route.load-self.data.max_capacity).max(0) as i64)*self.params.penalty_capa as i64};
        let ptw = self.params.penalty_tw as i64;
        let max_distance_acceptable = max_acceptable_cost - cap_pen;
        let a_id = route.node(pos1 - 1).id;
        let u_id = route.node(pos1).id;
        let old_distance = route.distance as i64;
        let removed_au = route.node(pos1 - 1).dist_to_succ;
        let distance_from_a = data.distance_row(a_id);
        let distance_from_u = data.distance_row(u_id);

        let can_stop=true && self.params.penalty_capa<=1_000_000 && self.params.penalty_tw<=1_000_000;
        let max_reverse_warp_cost=max_acceptable_cost-cap_pen-route.node(pos1-1).prefix.distance as i64;
        let mut mid_rev = route.node(pos1).seq21();
        let mut mid_rev_distance = mid_rev.distance;
        for pos2 in (pos1 + 1)..(len - 1) {
            if can_stop && mid_rev.tw as i64*ptw>max_reverse_warp_cost {break;}

            let v_id = route.node(pos2).id;
            let b_id = route.node(pos2 + 1).id;
            let d_av = copy_at(distance_from_a, v_id);
            let d_ub = copy_at(distance_from_u, b_id);
            // Materialize the exact reversed-segment distance for accepted
            // candidates, including asymmetric internal arcs.
            let left = route.node(pos1 - 1).seq0_i();
            let right = route.node(pos2 + 1).seqi_n();
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
                    mid_rev = Sequence::join_tw_with_travel(&next.seq1(), &mid_rev, reversed_travel);
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
                mid_rev = Sequence::join_tw_with_travel(&next.seq1(), &mid_rev, reversed_travel);
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
    #[inline(always)]
    fn run_2optstar_assured<const TABLE:bool>(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize) -> i64 {
        debug_assert!(r1 != r2);
        debug_assert!(pos1 > 0 && pos1 < self.routes[r1].nodes.len());
        debug_assert!(pos2 > 0 && pos2 < self.routes[r2].nodes.len());

        let route1 = route_at(&self.routes, r1);
        let route2 = route_at(&self.routes, r2);
        let old_cost = route1.cost + route2.cost;
        let max_acceptable_cost = old_cost + self.query_credit;

        let left1 = &route1.node(pos1 - 1).seq0_i();
        let right1 = &route2.node(pos2).seqi_n();
        let left2 = &route2.node(pos2 - 1).seq0_i();
        let right2 = &route1.node(pos1).seqi_n();

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
        let penalties=if TABLE {copy_at(&self.capacity_costs,load1 as usize)+copy_at(&self.capacity_costs,load2 as usize)}
            else {((load1-max_cap).max(0) as i64)*pcap+((load2-max_cap).max(0) as i64)*pcap};
        let lb_cost=dist1 as i64+dist2 as i64+penalties;
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
    #[cfg(target_arch="x86_64")]
    #[inline(always)]
    unsafe fn batch_neighbor_bounds_inline_assured<const THREE:bool>(&self,r1:usize,pos1:usize,neighbors:&[u32],eligible:u64)->(u64,u64) {
        let active=eligible.count_ones() as usize;
        if !self.sparse_bmi2 || neighbors.len()<8 || neighbors.len()>64 || active<4
            || (active+7)/8>=(neighbors.len()+7)/8 {
            return self.batch_neighbor_bounds_inline_assured_dense::<THREE>(r1,pos1,neighbors,eligible);
        }
        let mut packed=[0u32;72];
        let count=Self::compress_neighbor_lanes(neighbors,eligible,&mut packed);
        debug_assert_eq!(count,active);
        let padded=(count+7)&!7;let bits=(1u64<<count)-1;
        let result=self.batch_neighbor_bounds_inline_assured_dense::<THREE>(r1,pos1,&packed[..padded],bits);
        let fast=(std::arch::x86_64::_pdep_u64(result.0,eligible),std::arch::x86_64::_pdep_u64(result.1,eligible));
        fast
    }
    #[cfg(target_arch="x86_64")]
    #[inline(always)]
    unsafe fn batch_neighbor_bounds_inline_assured_dense<const THREE:bool>(&self,r1:usize,pos1:usize,neighbors:&[u32],eligible:u64)->(u64,u64) {
        let route=route_at(&self.routes,r1);
        let span=(route.nodes.len()-pos1-1).min(3);
        let fast=match span {
            1=>self.batch_neighbor_bounds_inline_assured_dense_dispatched::<THREE,1>(r1,pos1,neighbors,eligible),
            2=>self.batch_neighbor_bounds_inline_assured_dense_dispatched::<THREE,2>(r1,pos1,neighbors,eligible),
            _=>self.batch_neighbor_bounds_inline_assured_dense_dispatched::<THREE,3>(r1,pos1,neighbors,eligible),
        };
        fast
    }
    #[cfg(target_arch="x86_64")]
    #[inline(always)]
    unsafe fn batch_neighbor_bounds_inline_assured_dense_dispatched<const THREE:bool,const SPAN:usize>(&self,r1:usize,pos1:usize,neighbors:&[u32],eligible:u64)->(u64,u64) {
        use std::arch::x86_64::*;
        const SAFE:i64=67_108_864;
        let ru=route_at(&self.routes,r1);
        if neighbors.len()<8 || !true || ru.cost>SAFE || self.query_credit< -SAFE || self.query_credit>SAFE
            || eligible.count_ones()<4 {return (eligible,eligible);}
        let data=self.data.as_ref();let u=ru.node(pos1);let up=ru.node(pos1-1);let x=ru.node(pos1+1);
        let zero=_mm256_setzero_si256();let all=_mm256_set1_epi32(-1);
        macro_rules! b {($x:expr)=>{_mm256_set1_epi32($x as i32)}}
        macro_rules! add {($x:expr,$y:expr)=>{_mm256_add_epi32($x,$y)}}
        macro_rules! sub {($x:expr,$y:expr)=>{_mm256_sub_epi32($x,$y)}}
        macro_rules! mn {($x:expr,$y:expr)=>{_mm256_min_epi32($x,$y)}}
        macro_rules! gather {($slice:expr,$ids:expr)=>{_mm256_i32gather_epi32($slice.as_ptr(),$ids,4)}}
        let from_up=data.distance_row(up.id);let from_u=data.distance_row(u.id);let from_x=data.distance_row(x.id);
        let to_u=data.distance_column(u.id);let to_x=data.distance_column(x.id);
        let mut allow_inter=eligible;let mut allow_two=eligible;
        for block in 0..(neighbors.len()+7)/8 {
            let start=(block*8).min(neighbors.len()-8);
            let chunk=(eligible>>start)&255;if chunk==0 {continue;}
            // Eight packed u32 ids; start is at most len-8, including the overlapped tail.
            let ids=_mm256_loadu_si256(neighbors.as_ptr().add(start) as *const __m256i);
            let indexes=_mm256_mullo_epi32(ids,b!(16));
            macro_rules! f {($k:expr)=>{_mm256_i32gather_epi32((self.batch_cuts.as_ptr() as *const i32).add($k),indexes,4)}}
            let rload=f!(0);let total=zero;let rcost=f!(2);
            let limit=add!(b!(self.donor_cut_budget(u.id)+self.query_credit),rcost);
            let invalid=_mm256_cmpgt_epi32(rcost,b!(SAFE));
            let v=f!(3);let y=f!(4);let z=f!(5);let w=_mm256_and_si256(f!(6),b!(65535));
            
            let prededge=f!(7);let vedge=f!(8);let yedge=f!(9);let reverse=f!(10);
            let valid_v=_mm256_cmpgt_epi32(v,zero);let valid_y=_mm256_cmpgt_epi32(y,zero);
            let valid_z=_mm256_cmpgt_epi32(z,zero);
            let common_capacity=_mm256_mullo_epi32(_mm256_max_epi32(add!(rload,b!(ru.load-2*data.max_capacity)),zero),b!(self.params.penalty_capa));
            // The guarded i32 domain bounds both the original sum and this
            // subtraction. A + capacity <= limit iff A <= limit - capacity.
            let limit=sub!(limit,common_capacity);
            macro_rules! cap { ($send1:expr,$send2:expr)=>{zero}; }
            let dpu=gather!(to_u,ids);let dpv=gather!(from_up,v);let duv=gather!(from_u,v);
            let dvx=gather!(to_x,v);let duy=gather!(from_u,y);
            let remu1=up.dist_to_succ+u.dist_to_succ;
            let base10=sub!(sub!(total,b!(remu1)),prededge);
            let lb10=add!(add!(add!(base10,b!(u.bridge)),add!(dpu,duv)),cap!(u.load,zero));
            let mut best_all=lb10;
            let mut best_valid_v=b!(i32::MAX);
            let mut best_valid_y=b!(i32::MAX);
            let mut best_valid_z=b!(i32::MAX);

            let remv1=add!(prededge,vedge);let remv2=add!(remv1,yedge);
            let lb11=add!(add!(sub!(sub!(total,b!(remu1)),remv1),add!(add!(dpv,dvx),add!(dpu,duy))),cap!(u.load,vload));
            best_valid_v=mn!(best_valid_v,lb11);
            if SPAN>=2 {
                let xn=ru.node(pos1+2);let to_xn=data.distance_column(xn.id);let from_xn=data.distance_row(xn.id);
                let remu2=remu1+x.dist_to_succ;let send2=u.load+x.load;
                let dpx=gather!(to_x,ids);let dxv=gather!(from_x,v);let dxy=gather!(from_x,y);
                let dxz=gather!(from_x,z);let duz=gather!(from_u,z);
                let dvxn=gather!(to_xn,v);let dyxn=gather!(to_xn,y);let dpy=gather!(from_up,y);
                let forward=add!(dpu,b!(u.dist_to_succ));let backward=add!(dpx,b!(data.dm(x.id,u.id)));
                let min20=mn!(add!(forward,dxv),add!(backward,duv));
                let lb20=add!(add!(sub!(sub!(total,b!(remu2)),prededge),add!(b!(data.dm(up.id,xn.id)),min20)),cap!(send2,zero));
                best_all=mn!(best_all,lb20);
                let min21=mn!(add!(forward,dxy),add!(backward,duy));
                let lb21=add!(add!(sub!(sub!(total,b!(remu2)),remv1),add!(add!(dpv,dvxn),min21)),cap!(send2,vload));
                best_valid_v=mn!(best_valid_v,lb21);
                let left_min=mn!(add!(add!(dpv,vedge),dyxn),add!(add!(dpy,reverse),dvxn));
                let right_min=mn!(add!(forward,dxz),add!(backward,duz));
                let lb22=add!(add!(sub!(sub!(total,b!(remu2)),remv2),add!(left_min,right_min)),cap!(send2,pairload));
                best_valid_y=mn!(best_valid_y,lb22);
                if THREE && SPAN>=3 {
                    let xnn=ru.node(pos1+3);let to_xnn=data.distance_column(xnn.id);
                    let remu3=remu2+xn.dist_to_succ;let send3=send2+xn.load;
                    let path=add!(dpu,b!(u.dist_to_succ+x.dist_to_succ));
                    let dxnv=gather!(from_xn,v);let dxny=gather!(from_xn,y);
                    let dvxnn=gather!(to_xnn,v);let dyxnn=gather!(to_xnn,y);
                    let dxnz=gather!(from_xn,z);
                    let lb40=add!(add!(sub!(sub!(total,b!(remu3)),prededge),add!(b!(data.dm(up.id,xnn.id)),add!(path,dxnv))),cap!(send3,zero));
                    best_all=mn!(best_all,lb40);
                    let lb41=add!(add!(sub!(sub!(total,b!(remu3)),remv1),add!(add!(dpv,dvxnn),add!(path,dxny))),cap!(send3,vload));
                    best_valid_v=mn!(best_valid_v,lb41);
                    let left_min=mn!(add!(add!(dpv,vedge),dyxnn),add!(add!(dpy,reverse),dvxnn));
                    let lb42=add!(add!(sub!(sub!(total,b!(remu3)),remv2),add!(left_min,add!(path,dxnz))),cap!(send3,pairload));
                    best_valid_y=mn!(best_valid_y,lb42);
                    // The final removed edge is z -> w; its travel is recovered
                    // from the distance matrix because the common record is small.
                    let dzw=_mm256_i32gather_epi32(data.distance_matrix.as_ptr(),add!(_mm256_mullo_epi32(z,b!(data.nb_nodes)),w),4);
                    let dzxnn=gather!(to_xnn,z);let dxnw=gather!(from_xn,w);
                    let lb44=add!(add!(sub!(sub!(total,b!(remu3)),add!(remv2,dzw)),
                        add!(add!(add!(dpv,vedge),add!(yedge,dzxnn)),add!(path,dxnw))),cap!(send3,threeload));
                    best_valid_z=mn!(best_valid_z,lb44);
                }
            }
            
            let lb_two=add!(add!(sub!(sub!(total,b!(up.dist_to_succ)),prededge),add!(dpv,dpu)),cap!(u.suffix.load,tail));
            let possible_two=_mm256_or_si256(invalid,_mm256_andnot_si256(_mm256_cmpgt_epi32(lb_two,limit),all));
            // Minima are combined only within an identical validity mask;
            // move ordering is still determined by the unchanged exact kernel.
            let mut possible=_mm256_andnot_si256(_mm256_cmpgt_epi32(best_all,limit),all);
            possible=_mm256_or_si256(possible,_mm256_andnot_si256(_mm256_cmpgt_epi32(best_valid_v,limit),valid_v));
            possible=_mm256_or_si256(possible,_mm256_andnot_si256(_mm256_cmpgt_epi32(best_valid_y,limit),valid_y));
            possible=_mm256_or_si256(possible,_mm256_andnot_si256(_mm256_cmpgt_epi32(best_valid_z,limit),valid_z));
            possible=_mm256_or_si256(possible,invalid);
            let inter_bits=_mm256_movemask_ps(_mm256_castsi256_ps(possible)) as u64;
            let two_bits=_mm256_movemask_ps(_mm256_castsi256_ps(possible_two)) as u64;
            allow_inter &= !(255u64<<start) | (inter_bits<<start);
            allow_two &= !(255u64<<start) | (two_bits<<start);
        }
        (allow_inter,allow_two)
    }
    #[cfg(target_arch="x86_64")]
    #[inline(always)]
    unsafe fn batch_reverse_bounds_inline_assured<const THREE:bool>(&self,r2:usize,pos2:usize,neighbors:&[u32],eligible:u64)->u64 {
        let active=eligible.count_ones() as usize;
        if !self.sparse_bmi2 || neighbors.len()<8 || neighbors.len()>64 || active<4
            || (active+7)/8>=(neighbors.len()+7)/8 {
            return self.batch_reverse_bounds_inline_assured_dense::<THREE>(r2,pos2,neighbors,eligible);
        }
        let mut packed=[0u32;72];
        let count=Self::compress_neighbor_lanes(neighbors,eligible,&mut packed);
        debug_assert_eq!(count,active);
        let padded=(count+7)&!7;let bits=(1u64<<count)-1;
        let result=self.batch_reverse_bounds_inline_assured_dense::<THREE>(r2,pos2,&packed[..padded],bits);
        let fast=std::arch::x86_64::_pdep_u64(result,eligible);
        fast
    }
    #[cfg(target_arch="x86_64")]
    #[inline(always)]
    unsafe fn batch_reverse_bounds_inline_assured_dense<const THREE:bool>(&self,r2:usize,pos2:usize,neighbors:&[u32],eligible:u64)->u64 {
        use std::arch::x86_64::*;
        const SAFE:i64=67_108_864;
        let rv=route_at(&self.routes,r2);
        if neighbors.len()<8 || !true || rv.cost>SAFE
            || self.query_credit< -SAFE || self.query_credit>SAFE || eligible.count_ones()<4 {return eligible;}
        let data=self.data.as_ref();let v=rv.node(pos2);let vp=rv.node(pos2-1);let y=rv.node(pos2+1);
        let zero=_mm256_setzero_si256();let all=_mm256_set1_epi32(-1);
        macro_rules! b {($x:expr)=>{_mm256_set1_epi32($x as i32)}}
        macro_rules! add {($x:expr,$y:expr)=>{_mm256_add_epi32($x,$y)}}
        macro_rules! sub {($x:expr,$y:expr)=>{_mm256_sub_epi32($x,$y)}}
        macro_rules! mn {($x:expr,$y:expr)=>{_mm256_min_epi32($x,$y)}}
        macro_rules! gather {($slice:expr,$ids:expr)=>{_mm256_i32gather_epi32($slice.as_ptr(),$ids,4)}}
        macro_rules! dm {($from:expr,$to:expr)=>{_mm256_i32gather_epi32(data.distance_matrix.as_ptr(),add!(_mm256_mullo_epi32($from,b!(data.nb_nodes)),$to),4)}}
        let from_vp=data.distance_row(vp.id);let from_v=data.distance_row(v.id);
        let to_v=data.distance_column(v.id);let to_y=data.distance_column(y.id);
        let mut allowed=eligible;
        for block in 0..(neighbors.len()+7)/8 {
            let start=(block*8).min(neighbors.len()-8);if (eligible>>start)&255==0 {continue;}
            let u=_mm256_loadu_si256(neighbors.as_ptr().add(start) as *const __m256i);
            let indexes=_mm256_mullo_epi32(u,b!(16));
            macro_rules! f {($k:expr)=>{_mm256_i32gather_epi32((self.batch_cuts.as_ptr() as *const i32).add($k),indexes,4)}}
            let rload=f!(0);let total=zero;let rcost=f!(2);
            let invalid=_mm256_cmpgt_epi32(rcost,b!(SAFE));
            let donor_adjust=_mm256_mullo_epi32(_mm256_srai_epi32::<16>(f!(6)),b!(self.params.penalty_tw));
            let limit=add!(sub!(rcost,donor_adjust),b!(self.donor_cut_budget(v.id)+self.query_credit));
            let up=f!(11);let x=f!(3);let xn=f!(4);let xnn=f!(5);
            
            let upedge=f!(12);let uedge=f!(7);let xedge=f!(8);let xnedge=f!(9);
            let remu1=add!(upedge,uedge);let remu2=add!(remu1,xedge);let remu3=add!(remu2,xnedge);
            let valid_x=_mm256_cmpgt_epi32(x,zero);let valid_xn=_mm256_cmpgt_epi32(xn,zero);
            let common_capacity=_mm256_mullo_epi32(_mm256_max_epi32(add!(rload,b!(rv.load-2*data.max_capacity)),zero),b!(self.params.penalty_capa));
            // The guarded i32 domain bounds both the original sum and this
            // subtraction. A + capacity <= limit iff A <= limit - capacity.
            let limit=sub!(limit,common_capacity);
            macro_rules! cap { ($send1:expr,$send2:expr)=>{zero}; }
            let dpu=gather!(from_vp,u);let dpv=gather!(to_v,up);let duv=gather!(to_v,u);
            let dvx=gather!(from_v,x);let duy=gather!(to_y,u);
            let lb10=add!(add!(sub!(sub!(total,remu1),b!(vp.dist_to_succ)),add!(dm!(up,x),add!(dpu,duv))),cap!(uload,0));
            let mut best_all=lb10;
            let mut best_valid_x=b!(i32::MAX);
            let mut best_valid_xn=b!(i32::MAX);

            let remv1=vp.dist_to_succ+v.dist_to_succ;let remv2=remv1+y.dist_to_succ;
            let lb11=add!(add!(sub!(sub!(total,remu1),b!(remv1)),add!(add!(dpv,dvx),add!(dpu,duy))),cap!(uload,v.load));
            best_all=mn!(best_all,lb11);
            if _mm256_movemask_ps(_mm256_castsi256_ps(valid_x))!=0 {
                let dpx=gather!(from_vp,x);let dxv=gather!(to_v,x);let dxy=gather!(to_y,x);
                let dvxn=gather!(from_v,xn);let forward=add!(dpu,uedge);let backward=add!(dpx,dm!(x,u));
                let min20=mn!(add!(forward,dxv),add!(backward,duv));
                let lb20=add!(add!(sub!(sub!(total,remu2),b!(vp.dist_to_succ)),add!(dm!(up,xn),min20)),cap!(pairload,0));
                best_valid_x=mn!(best_valid_x,lb20);
                let min21=mn!(add!(forward,dxy),add!(backward,duy));
                let lb21=add!(add!(sub!(sub!(total,remu2),b!(remv1)),add!(add!(dpv,dvxn),min21)),cap!(pairload,v.load));
                best_valid_x=mn!(best_valid_x,lb21);
                if y.id!=0 {
                    let yn=rv.node(pos2+2);let to_yn=data.distance_column(yn.id);let from_y=data.distance_row(y.id);
                    let dpy=gather!(to_y,up);let dyxn=gather!(from_y,xn);
                    let left_min=mn!(add!(add!(dpv,b!(v.dist_to_succ)),dyxn),add!(add!(dpy,b!(data.dm(y.id,v.id))),dvxn));
                    let right_min=mn!(add!(forward,gather!(to_yn,x)),add!(backward,gather!(to_yn,u)));
                    let lb22=add!(add!(sub!(sub!(total,remu2),b!(remv2)),add!(left_min,right_min)),cap!(pairload,v.load+y.load));
                    best_valid_x=mn!(best_valid_x,lb22);
                }
                if THREE && _mm256_movemask_ps(_mm256_castsi256_ps(valid_xn))!=0 {
                    let path=add!(dpu,add!(uedge,xedge));let dvxnn=gather!(from_v,xnn);
                    let lb40=add!(add!(sub!(sub!(total,remu3),b!(vp.dist_to_succ)),add!(dm!(up,xnn),add!(path,gather!(to_v,xn)))),cap!(threeload,0));
                    best_valid_xn=mn!(best_valid_xn,lb40);
                    let lb41=add!(add!(sub!(sub!(total,remu3),b!(remv1)),add!(add!(dpv,dvxnn),add!(path,gather!(to_y,xn)))),cap!(threeload,v.load));
                    best_valid_xn=mn!(best_valid_xn,lb41);
                    if y.id!=0 {
                        let yn=rv.node(pos2+2);let to_yn=data.distance_column(yn.id);let from_y=data.distance_row(y.id);
                        let dpy=gather!(to_y,up);let dyxnn=gather!(from_y,xnn);
                        let left_min=mn!(add!(add!(dpv,b!(v.dist_to_succ)),dyxnn),add!(add!(dpy,b!(data.dm(y.id,v.id))),dvxnn));
                        let lb42=add!(add!(sub!(sub!(total,remu3),b!(remv2)),add!(left_min,add!(path,gather!(to_yn,xn)))),cap!(threeload,v.load+y.load));
                        best_valid_xn=mn!(best_valid_xn,lb42);
                        if yn.id!=0 {
                            let ynn=rv.node(pos2+3);let to_ynn=data.distance_column(ynn.id);let from_yn=data.distance_row(yn.id);
                            let left=add!(add!(dpv,b!(v.dist_to_succ+y.dist_to_succ)),gather!(from_yn,xnn));
                            let right=add!(path,gather!(to_ynn,xn));
                            let lb44=add!(add!(sub!(sub!(total,remu3),b!(remv2+yn.dist_to_succ)),add!(left,right)),cap!(threeload,v.load+y.load+yn.load));
                            best_valid_xn=mn!(best_valid_xn,lb44);
                        }
                    }
                }
            }
            // Minima are combined only within an identical validity mask;
            // move ordering is still determined by the unchanged exact kernel.
            let mut possible=_mm256_andnot_si256(_mm256_cmpgt_epi32(best_all,limit),all);
            possible=_mm256_or_si256(possible,_mm256_andnot_si256(_mm256_cmpgt_epi32(best_valid_x,limit),valid_x));
            possible=_mm256_or_si256(possible,_mm256_andnot_si256(_mm256_cmpgt_epi32(best_valid_xn,limit),valid_xn));
            possible=_mm256_or_si256(possible,invalid);
            let bits=_mm256_movemask_ps(_mm256_castsi256_ps(possible)) as u64;
            allowed &= !(255u64<<start) | (bits<<start);
        }
        allowed
    }
    #[inline(always)]
    fn route_insertion_lower_bound_assured<const MINPOS:bool>(data:&Problem, route:&Route, entry:&mut InsertionLowerBound,
                                   from:&[i32],to:&[i32])->i32 {
        if entry.version==route.insertion_version { return entry.value; }
        #[cfg(target_arch="x86_64")]
        if true {
            let best=unsafe{if MINPOS {Self::vector_arc_min_narrow(&route.arc_ids,&route.arc_travel,from.as_ptr(),to.as_ptr())}
                else {Self::vector_arc_min(&route.arc_ids,&route.arc_travel,from.as_ptr(),to.as_ptr())}};
            *entry=InsertionLowerBound{version:route.insertion_version,value:best};return best;
        }
        let mut best=i32::MAX;
        for t in 1..route.nodes.len() {
            let pred=route.node(t-1);let next=route.node(t);
            best=best.min(copy_at(to,pred.id)+copy_at(from,next.id)-pred.dist_to_succ);
        }
        *entry=InsertionLowerBound{version:route.insertion_version,value:best};best
    }
    #[inline(always)]
    fn run_inter_source_impl_keyed_assured<const THREE: bool, const TABLE: bool,const PACKED:bool>(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize,src:&InterSource) -> i64 {
        let data = self.data.as_ref();
        let ru = route_at(&self.routes, r1);
        let rv = route_at(&self.routes, r2);
        let u = ru.node(pos1);
        let v = rv.node(pos2);
        let u_pred = ru.node(pos1 - 1);
        let v_pred = rv.node(pos2 - 1);
        let x = ru.node(pos1 + 1);
        let distance_from_u_pred = u_pred.distance_row(data);
        let distance_from_v_pred = v_pred.distance_row(data);
        let distance_from_u = u.distance_row(data);
        let distance_from_v = v.distance_row(data);
        let distance_from_x = x.distance_row(data);
        debug_assert!(
            u.id != 0,
            "Should always apply inter-route with a client as first node"
        );
        debug_assert!(r1 != r2, "Should not test inter-route move on same route");

        let old_total = src.cost + rv.cost;
        let max_acceptable_cost = old_total + self.query_credit;
        let group_limit=if true {max_acceptable_cost} else {i64::MAX};

        let pcap = self.params.penalty_capa as i64;
        let ptw = self.params.penalty_tw as i64;
        let max_cap = data.max_capacity;
        let route_total_dist = src.distance + rv.distance;
        let route1_load = src.load;
        let route2_load = rv.load;
        let mut best_key=i64::MAX;
        let mut best_cost = i64::MAX;
        let mut best_send1 = 0u8;
        let mut best_send2 = 0u8;

        macro_rules! cap_pen {
            ($load:expr) => {{
                if TABLE { copy_at(&self.capacity_costs, $load as usize) }
                else { ((($load) - max_cap).max(0) as i64) * pcap }
            }};
        }
        macro_rules! consider {
            ($order:expr, $send1:expr, $send2:expr, $lb:expr, $tw:expr) => {{
                let lower_bound = $lb;
                if lower_bound <= max_acceptable_cost {
                    let candidate = lower_bound + ($tw as i64) * ptw;
                    if PACKED {
                        // Existing safe-domain checks bound every route sum
                        // to i32 and each penalty to 1,000,000. Multiplying a
                        // nonnegative two-route cost by 16 therefore fits i64.
                        // The low four bits encode original evaluation order.
                        
                        best_key=best_key.min(candidate*16+$order);
                    } else {
                    if candidate <= max_acceptable_cost && candidate < best_cost {
                        best_cost = candidate;
                        best_send1 = $send1;
                        best_send2 = $send2;
                    }
                    }
                }
            }};
        }

        let send1_1_load = src.sent[0];
        let rem1_1 = src.removed[0];
        let rem2_0 = v_pred.dist_to_succ;
        let d_upred_x = src.bridges[0];
        let d_vpred_u = copy_at(distance_from_v_pred, u.id);
        let d_u_v = copy_at(distance_from_u, v.id);
        let cap_10 = cap_pen!(route1_load - send1_1_load) + cap_pen!(route2_load + send1_1_load);
        let lb10 =
            (route_total_dist - rem1_1 - rem2_0 + d_upred_x + d_vpred_u + d_u_v) as i64 + cap_10;
        if lb10 <= max_acceptable_cost {
            let tw10 = Sequence::tw2_with_travel(&u_pred.seq0_i(), &x.seqi_n(), d_upred_x)
                + Sequence::tw3_with_travel(&v_pred.seq0_i(), &u.seq1(), &v.seqi_n(), d_vpred_u, d_u_v);
            consider!(0,1, 0, lb10, tw10);
        }

        if v.id != 0 {
            let y = rv.node(pos2 + 1);
            let send2_1_load = v.seq1().load;
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
                    Sequence::tw3_with_travel(&u_pred.seq0_i(), &v.seq1(), &x.seqi_n(), d_upred_v, d_v_x)
                        + Sequence::tw3_with_travel(
                            &v_pred.seq0_i(),
                            &u.seq1(),
                            &y.seqi_n(),
                            d_vpred_u,
                            d_u_y,
                        );
                consider!(1,1, 1, lb11, tw11);
            }
        }

        if x.id != 0 {
            let x_next = ru.node(pos1 + 2);
            let distance_from_x_next = x_next.distance_row(data);
            let send1_2_load = src.sent[1];
            let rem1_2 = src.removed[1];
            let d_upred_xnext = src.bridges[1];
            let d_vpred_x = copy_at(distance_from_v_pred, x.id);
            let d_x_u = src.reverse_pair;
            let d_u_x = u.dist_to_succ;
            let d_x_v = copy_at(distance_from_x, v.id);
            let cap_20_30 =
                cap_pen!(route1_load - send1_2_load) + cap_pen!(route2_load + send1_2_load);
            let dist_base_20_30 = route_total_dist - rem1_2 - rem2_0;
            if (dist_base_20_30.wrapping_add(d_upred_xnext).wrapping_add(
                d_vpred_u.wrapping_add(d_u_x).wrapping_add(d_x_v).min(d_vpred_x.wrapping_add(d_x_u).wrapping_add(d_u_v))) as i64).wrapping_add(cap_20_30)<=group_limit {
let lb20 =
                (dist_base_20_30 + d_upred_xnext + d_vpred_u + d_u_x + d_x_v) as i64 + cap_20_30;
            let lb30 =
                (dist_base_20_30 + d_upred_xnext + d_vpred_x + d_x_u + d_u_v) as i64 + cap_20_30;
            if lb20 <= max_acceptable_cost || lb30 <= max_acceptable_cost {
                let route1_tw =
                    Sequence::tw2_with_travel(&u_pred.seq0_i(), &x_next.seqi_n(), d_upred_xnext);
                if lb20 <= max_acceptable_cost {
                    let tw20 = route1_tw
                        + Sequence::tw3_with_travel(
                            &v_pred.seq0_i(),
                            &u.seq12(),
                            &v.seqi_n(),
                            d_vpred_u,
                            d_x_v,
                        );
                    consider!(2,2, 0, lb20, tw20);
                }
                if lb30 <= max_acceptable_cost {
                    let tw30 = route1_tw
                        + Sequence::tw3_with_travel(
                            &v_pred.seq0_i(),
                            &u.seq21(),
                            &v.seqi_n(),
                            d_vpred_x,
                            d_u_v,
                        );
                    consider!(3,3, 0, lb30, tw30);
                }
            }
            }

            if v.id != 0 {
                let y = rv.node(pos2 + 1);
                let send2_1_load = v.seq1().load;
                let rem2_1 = v_pred.dist_to_succ + v.dist_to_succ;
                let d_upred_v = copy_at(distance_from_u_pred, v.id);
                let d_v_xnext = copy_at(distance_from_v, x_next.id);
                let d_x_y = copy_at(distance_from_x, y.id);
                let d_u_y = copy_at(distance_from_u, y.id);
                let cap_21_31 = cap_pen!(route1_load - send1_2_load + send2_1_load)
                    + cap_pen!(route2_load - send2_1_load + send1_2_load);
                let dist_base_21_31 = route_total_dist - rem1_2 - rem2_1;
                let common_left = dist_base_21_31 + d_upred_v + d_v_xnext;
                if (common_left.wrapping_add(
                d_vpred_u.wrapping_add(d_u_x).wrapping_add(d_x_y).min(d_vpred_x.wrapping_add(d_x_u).wrapping_add(d_u_y))) as i64).wrapping_add(cap_21_31)<=group_limit {
let lb21 = (common_left + d_vpred_u + d_u_x + d_x_y) as i64 + cap_21_31;
                let lb31 = (common_left + d_vpred_x + d_x_u + d_u_y) as i64 + cap_21_31;
                if lb21 <= max_acceptable_cost || lb31 <= max_acceptable_cost {
                    let route1_tw = Sequence::tw3_with_travel(
                        &u_pred.seq0_i(),
                        &v.seq1(),
                        &x_next.seqi_n(),
                        d_upred_v,
                        d_v_xnext,
                    );
                    if lb21 <= max_acceptable_cost {
                        let tw21 = route1_tw
                            + Sequence::tw3_with_travel(
                                &v_pred.seq0_i(),
                                &u.seq12(),
                                &y.seqi_n(),
                                d_vpred_u,
                                d_x_y,
                            );
                        consider!(4,2, 1, lb21, tw21);
                    }
                    if lb31 <= max_acceptable_cost {
                        let tw31 = route1_tw
                            + Sequence::tw3_with_travel(
                                &v_pred.seq0_i(),
                                &u.seq21(),
                                &y.seqi_n(),
                                d_vpred_x,
                                d_u_y,
                            );
                        consider!(5,3, 1, lb31, tw31);
                    }
                }
            }

                if y.id != 0 {
                    let y_next = rv.node(pos2 + 2);
                    let distance_from_y = y.distance_row(data);
                    let send2_2_load = v.seq1().load + y.seq1().load;
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
                    if (dist_base.wrapping_add(left_fwd_dist.min(left_rev_dist)).wrapping_add(right_fwd_dist.min(right_rev_dist)) as i64).wrapping_add(cap_22_33)<=group_limit {
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
                            &u_pred.seq0_i(),
                            &v.seq12(),
                            &x_next.seqi_n(),
                            d_upred_v,
                            d_y_xnext,
                        ))
                    } else {
                        None
                    };
                    let left_rev = if can23 || can33 {
                        Some(Sequence::tw3_with_travel(
                            &u_pred.seq0_i(),
                            &v.seq21(),
                            &x_next.seqi_n(),
                            d_upred_y,
                            d_v_xnext,
                        ))
                    } else {
                        None
                    };
                    let right_fwd = if can22 || can23 {
                        Some(Sequence::tw3_with_travel(
                            &v_pred.seq0_i(),
                            &u.seq12(),
                            &y_next.seqi_n(),
                            d_vpred_u,
                            d_x_ynext,
                        ))
                    } else {
                        None
                    };
                    let right_rev = if can32 || can33 {
                        Some(Sequence::tw3_with_travel(
                            &v_pred.seq0_i(),
                            &u.seq21(),
                            &y_next.seqi_n(),
                            d_vpred_x,
                            d_u_ynext,
                        ))
                    } else {
                        None
                    };

                    if can22 {
                        consider!(6,2, 2, lb22, left_fwd.unwrap() + right_fwd.unwrap());
                    }
                    if can32 {
                        consider!(7,3, 2, lb32, left_fwd.unwrap() + right_rev.unwrap());
                    }
                    if can23 {
                        consider!(8,2, 3, lb23, left_rev.unwrap() + right_fwd.unwrap());
                    }
                    if can33 {
                        consider!(9,3, 3, lb33, left_rev.unwrap() + right_rev.unwrap());
                    }
                    }
                }
            }

            if THREE && x_next.id != 0 {
                let x2_next = ru.node(pos1 + 3);
                let send1_3_load = src.sent[2];
                let rem1_3 = src.removed[2];
                let d_upred_x2next = src.bridges[2];
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
                        Sequence::tw2_with_travel(&u_pred.seq0_i(), &x2_next.seqi_n(), d_upred_x2next)
                            + Sequence::tw3_with_travel(
                                &v_pred.seq0_i(),
                                &u.seq123(),
                                &v.seqi_n(),
                                d_vpred_u,
                                d_xnext_v,
                            );
                    consider!(10,4, 0, lb40, tw40);
                }

                if v.id != 0 {
                    let y = rv.node(pos2 + 1);
                    let send2_1_load = v.seq1().load;
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
                            &u_pred.seq0_i(),
                            &v.seq1(),
                            &x2_next.seqi_n(),
                            d_upred_v,
                            d_v_x2next,
                        ) + Sequence::tw3_with_travel(
                            &v_pred.seq0_i(),
                            &u.seq123(),
                            &y.seqi_n(),
                            d_vpred_u,
                            d_xnext_y,
                        );
                        consider!(11,4, 1, lb41, tw41);
                    }

                    if y.id != 0 {
                        let y_next = rv.node(pos2 + 2);
                        let distance_from_y = y.distance_row(data);
                        let send2_2_load = v.seq1().load + y.seq1().load;
                        let rem2_2 = v_pred.dist_to_succ + v.dist_to_succ + y.dist_to_succ;
                        let d_upred_y = copy_at(distance_from_u_pred, y.id);
                        let d_y_v = copy_at(distance_from_y, v.id);
                        let d_y_x2next = copy_at(distance_from_y, x2_next.id);
                        let d_xnext_ynext = copy_at(distance_from_x_next, y_next.id);
                        let cap_42_43 = cap_pen!(route1_load - send1_3_load + send2_2_load)
                            + cap_pen!(route2_load - send2_2_load + send1_3_load);
                        let dist_base = route_total_dist - rem1_3 - rem2_2;
                        let common_right = d_vpred_u + d_u_xnext + d_x_xnext + d_xnext_ynext;
                        if (dist_base.wrapping_add(common_right).wrapping_add(
                d_upred_v.wrapping_add(v.dist_to_succ).wrapping_add(d_y_x2next).min(d_upred_y.wrapping_add(d_y_v).wrapping_add(d_v_x2next))) as i64).wrapping_add(cap_42_43)<=group_limit {
let lb42 =
                            (dist_base + d_upred_v + v.dist_to_succ + d_y_x2next + common_right)
                                as i64
                                + cap_42_43;
                        let lb43 = (dist_base + d_upred_y + d_y_v + d_v_x2next + common_right)
                            as i64
                            + cap_42_43;
                        if lb42 <= max_acceptable_cost || lb43 <= max_acceptable_cost {
                            let right_tw = Sequence::tw3_with_travel(
                                &v_pred.seq0_i(),
                                &u.seq123(),
                                &y_next.seqi_n(),
                                d_vpred_u,
                                d_xnext_ynext,
                            );
                            if lb42 <= max_acceptable_cost {
                                let tw42 = Sequence::tw3_with_travel(
                                    &u_pred.seq0_i(),
                                    &v.seq12(),
                                    &x2_next.seqi_n(),
                                    d_upred_v,
                                    d_y_x2next,
                                ) + right_tw;
                                consider!(12,4, 2, lb42, tw42);
                            }
                            if lb43 <= max_acceptable_cost {
                                let tw43 = Sequence::tw3_with_travel(
                                    &u_pred.seq0_i(),
                                    &v.seq21(),
                                    &x2_next.seqi_n(),
                                    d_upred_y,
                                    d_v_x2next,
                                ) + right_tw;
                                consider!(13,4, 3, lb43, tw43);
                            }
                        }
            }

                        if y_next.id != 0 {
                            let y2_next = rv.node(pos2 + 3);
                            let send2_3_load = v.seq1().load + y.seq1().load + y_next.seq1().load;
                            let rem2_3 = v_pred.dist_to_succ
                                + v.dist_to_succ
                                + y.dist_to_succ
                                + y_next.dist_to_succ;
                            let d_ynext_x2next = copy_at(y_next.distance_row(data), x2_next.id);
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
                                    &u_pred.seq0_i(),
                                    &v.seq123(),
                                    &x2_next.seqi_n(),
                                    d_upred_v,
                                    d_ynext_x2next,
                                ) + Sequence::tw3_with_travel(
                                    &v_pred.seq0_i(),
                                    &u.seq123(),
                                    &y2_next.seqi_n(),
                                    d_vpred_u,
                                    d_xnext_y2next,
                                );
                                consider!(14,4, 4, lb44, tw44);
                            }
                        }
                    }
                }
            }
        }

        if PACKED {
            if best_key==i64::MAX {return NO_MOVE;}
            let cost=best_key>>4;
            if cost>max_acceptable_cost {return NO_MOVE;}
            const PLANS:[(u8,u8);15]=[(1,0),(1,1),(2,0),(3,0),(2,1),(3,1),(2,2),(3,2),(2,3),(3,3),(4,0),(4,1),(4,2),(4,3),(4,4)];
            let (send1,send2)=PLANS[(best_key&15) as usize];
            self.last_plan=MovePlan::InterRoute{send1,send2};return cost-old_total;
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
    fn run_inter_special_keyed_assured<const THREE: bool, const TABLE: bool,const PACKED:bool>(&mut self, r1: usize, pos1: usize, r2: usize, pos2: usize) -> i64 {
        let data = self.data.as_ref();
        let ru = route_at(&self.routes, r1);
        let rv = route_at(&self.routes, r2);
        let u = ru.node(pos1);
        let v = rv.node(pos2);
        let u_pred = ru.node(pos1 - 1);
        let v_pred = rv.node(pos2 - 1);
        let x = ru.node(pos1 + 1);
        let distance_from_u_pred = u_pred.distance_row(data);
        let distance_from_v_pred = v_pred.distance_row(data);
        let distance_from_u = u.distance_row(data);
        let distance_from_v = v.distance_row(data);
        let distance_from_x = x.distance_row(data);
        debug_assert!(
            u.id != 0,
            "Should always apply inter-route with a client as first node"
        );
        debug_assert!(r1 != r2, "Should not test inter-route move on same route");

        let old_total = ru.cost + rv.cost;
        let max_acceptable_cost = old_total + self.query_credit;
        let group_limit=if true {max_acceptable_cost} else {i64::MAX};

        let pcap = self.params.penalty_capa as i64;
        let ptw = self.params.penalty_tw as i64;
        let max_cap = data.max_capacity;
        let route_total_dist = ru.distance + rv.distance;
        let route1_load = ru.load;
        let route2_load = rv.load;
        let mut best_key=i64::MAX;
        let mut best_cost = i64::MAX;
        let mut best_send1 = 0u8;
        let mut best_send2 = 0u8;

        macro_rules! cap_pen {
            ($load:expr) => {{
                if TABLE { copy_at(&self.capacity_costs, $load as usize) }
                else { ((($load) - max_cap).max(0) as i64) * pcap }
            }};
        }
        macro_rules! consider {
            ($order:expr, $send1:expr, $send2:expr, $lb:expr, $tw:expr) => {{
                let lower_bound = $lb;
                if lower_bound <= max_acceptable_cost {
                    let candidate = lower_bound + ($tw as i64) * ptw;
                    if PACKED {
                        // Existing safe-domain checks bound every route sum
                        // to i32 and each penalty to 1,000,000. Multiplying a
                        // nonnegative two-route cost by 16 therefore fits i64.
                        // The low four bits encode original evaluation order.
                        
                        best_key=best_key.min(candidate*16+$order);
                    } else {
                    if candidate <= max_acceptable_cost && candidate < best_cost {
                        best_cost = candidate;
                        best_send1 = $send1;
                        best_send2 = $send2;
                    }
                    }
                }
            }};
        }

        let send1_1_load = u.seq1().load;
        let rem1_1 = u_pred.dist_to_succ + u.dist_to_succ;
        let rem2_0 = v_pred.dist_to_succ;
        let d_upred_x = u.bridge;
        let d_vpred_u = copy_at(distance_from_v_pred, u.id);
        let d_u_v = copy_at(distance_from_u, v.id);
        let cap_10 = cap_pen!(route1_load - send1_1_load) + cap_pen!(route2_load + send1_1_load);
        let lb10 =
            (route_total_dist - rem1_1 - rem2_0 + d_upred_x + d_vpred_u + d_u_v) as i64 + cap_10;
        if lb10 <= max_acceptable_cost {
            let tw10 = Sequence::tw2_with_travel(&u_pred.seq0_i(), &x.seqi_n(), d_upred_x)
                + Sequence::tw3_with_travel(&v_pred.seq0_i(), &u.seq1(), &v.seqi_n(), d_vpred_u, d_u_v);
            consider!(0,1, 0, lb10, tw10);
        }

        if v.id != 0 {
            let y = rv.node(pos2 + 1);
            let send2_1_load = v.seq1().load;
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
                    Sequence::tw3_with_travel(&u_pred.seq0_i(), &v.seq1(), &x.seqi_n(), d_upred_v, d_v_x)
                        + Sequence::tw3_with_travel(
                            &v_pred.seq0_i(),
                            &u.seq1(),
                            &y.seqi_n(),
                            d_vpred_u,
                            d_u_y,
                        );
                consider!(1,1, 1, lb11, tw11);
            }
        }

        if x.id != 0 {
            let x_next = ru.node(pos1 + 2);
            let distance_from_x_next = x_next.distance_row(data);
            let send1_2_load = u.seq1().load + x.seq1().load;
            let rem1_2 = u_pred.dist_to_succ + u.dist_to_succ + x.dist_to_succ;
            let d_upred_xnext = copy_at(distance_from_u_pred, x_next.id);
            let d_vpred_x = copy_at(distance_from_v_pred, x.id);
            let d_x_u = copy_at(distance_from_x, u.id);
            let d_u_x = u.dist_to_succ;
            let d_x_v = copy_at(distance_from_x, v.id);
            let cap_20_30 =
                cap_pen!(route1_load - send1_2_load) + cap_pen!(route2_load + send1_2_load);
            let dist_base_20_30 = route_total_dist - rem1_2 - rem2_0;
            if (dist_base_20_30.wrapping_add(d_upred_xnext).wrapping_add(
                d_vpred_u.wrapping_add(d_u_x).wrapping_add(d_x_v).min(d_vpred_x.wrapping_add(d_x_u).wrapping_add(d_u_v))) as i64).wrapping_add(cap_20_30)<=group_limit {
let lb20 =
                (dist_base_20_30 + d_upred_xnext + d_vpred_u + d_u_x + d_x_v) as i64 + cap_20_30;
            let lb30 =
                (dist_base_20_30 + d_upred_xnext + d_vpred_x + d_x_u + d_u_v) as i64 + cap_20_30;
            if lb20 <= max_acceptable_cost || lb30 <= max_acceptable_cost {
                let route1_tw =
                    Sequence::tw2_with_travel(&u_pred.seq0_i(), &x_next.seqi_n(), d_upred_xnext);
                if lb20 <= max_acceptable_cost {
                    let tw20 = route1_tw
                        + Sequence::tw3_with_travel(
                            &v_pred.seq0_i(),
                            &u.seq12(),
                            &v.seqi_n(),
                            d_vpred_u,
                            d_x_v,
                        );
                    consider!(2,2, 0, lb20, tw20);
                }
                if lb30 <= max_acceptable_cost {
                    let tw30 = route1_tw
                        + Sequence::tw3_with_travel(
                            &v_pred.seq0_i(),
                            &u.seq21(),
                            &v.seqi_n(),
                            d_vpred_x,
                            d_u_v,
                        );
                    consider!(3,3, 0, lb30, tw30);
                }
            }
            }

            if v.id != 0 {
                let y = rv.node(pos2 + 1);
                let send2_1_load = v.seq1().load;
                let rem2_1 = v_pred.dist_to_succ + v.dist_to_succ;
                let d_upred_v = copy_at(distance_from_u_pred, v.id);
                let d_v_xnext = copy_at(distance_from_v, x_next.id);
                let d_x_y = copy_at(distance_from_x, y.id);
                let d_u_y = copy_at(distance_from_u, y.id);
                let cap_21_31 = cap_pen!(route1_load - send1_2_load + send2_1_load)
                    + cap_pen!(route2_load - send2_1_load + send1_2_load);
                let dist_base_21_31 = route_total_dist - rem1_2 - rem2_1;
                let common_left = dist_base_21_31 + d_upred_v + d_v_xnext;
                if (common_left.wrapping_add(
                d_vpred_u.wrapping_add(d_u_x).wrapping_add(d_x_y).min(d_vpred_x.wrapping_add(d_x_u).wrapping_add(d_u_y))) as i64).wrapping_add(cap_21_31)<=group_limit {
let lb21 = (common_left + d_vpred_u + d_u_x + d_x_y) as i64 + cap_21_31;
                let lb31 = (common_left + d_vpred_x + d_x_u + d_u_y) as i64 + cap_21_31;
                if lb21 <= max_acceptable_cost || lb31 <= max_acceptable_cost {
                    let route1_tw = Sequence::tw3_with_travel(
                        &u_pred.seq0_i(),
                        &v.seq1(),
                        &x_next.seqi_n(),
                        d_upred_v,
                        d_v_xnext,
                    );
                    if lb21 <= max_acceptable_cost {
                        let tw21 = route1_tw
                            + Sequence::tw3_with_travel(
                                &v_pred.seq0_i(),
                                &u.seq12(),
                                &y.seqi_n(),
                                d_vpred_u,
                                d_x_y,
                            );
                        consider!(4,2, 1, lb21, tw21);
                    }
                    if lb31 <= max_acceptable_cost {
                        let tw31 = route1_tw
                            + Sequence::tw3_with_travel(
                                &v_pred.seq0_i(),
                                &u.seq21(),
                                &y.seqi_n(),
                                d_vpred_x,
                                d_u_y,
                            );
                        consider!(5,3, 1, lb31, tw31);
                    }
                }
            }

                if y.id != 0 {
                    let y_next = rv.node(pos2 + 2);
                    let distance_from_y = y.distance_row(data);
                    let send2_2_load = v.seq1().load + y.seq1().load;
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
                    if (dist_base.wrapping_add(left_fwd_dist.min(left_rev_dist)).wrapping_add(right_fwd_dist.min(right_rev_dist)) as i64).wrapping_add(cap_22_33)<=group_limit {
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
                            &u_pred.seq0_i(),
                            &v.seq12(),
                            &x_next.seqi_n(),
                            d_upred_v,
                            d_y_xnext,
                        ))
                    } else {
                        None
                    };
                    let left_rev = if can23 || can33 {
                        Some(Sequence::tw3_with_travel(
                            &u_pred.seq0_i(),
                            &v.seq21(),
                            &x_next.seqi_n(),
                            d_upred_y,
                            d_v_xnext,
                        ))
                    } else {
                        None
                    };
                    let right_fwd = if can22 || can23 {
                        Some(Sequence::tw3_with_travel(
                            &v_pred.seq0_i(),
                            &u.seq12(),
                            &y_next.seqi_n(),
                            d_vpred_u,
                            d_x_ynext,
                        ))
                    } else {
                        None
                    };
                    let right_rev = if can32 || can33 {
                        Some(Sequence::tw3_with_travel(
                            &v_pred.seq0_i(),
                            &u.seq21(),
                            &y_next.seqi_n(),
                            d_vpred_x,
                            d_u_ynext,
                        ))
                    } else {
                        None
                    };

                    if can22 {
                        consider!(6,2, 2, lb22, left_fwd.unwrap() + right_fwd.unwrap());
                    }
                    if can32 {
                        consider!(7,3, 2, lb32, left_fwd.unwrap() + right_rev.unwrap());
                    }
                    if can23 {
                        consider!(8,2, 3, lb23, left_rev.unwrap() + right_fwd.unwrap());
                    }
                    if can33 {
                        consider!(9,3, 3, lb33, left_rev.unwrap() + right_rev.unwrap());
                    }
                    }
                }
            }

            if THREE && x_next.id != 0 {
                let x2_next = ru.node(pos1 + 3);
                let send1_3_load = u.seq1().load + x.seq1().load + x_next.seq1().load;
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
                        Sequence::tw2_with_travel(&u_pred.seq0_i(), &x2_next.seqi_n(), d_upred_x2next)
                            + Sequence::tw3_with_travel(
                                &v_pred.seq0_i(),
                                &u.seq123(),
                                &v.seqi_n(),
                                d_vpred_u,
                                d_xnext_v,
                            );
                    consider!(10,4, 0, lb40, tw40);
                }

                if v.id != 0 {
                    let y = rv.node(pos2 + 1);
                    let send2_1_load = v.seq1().load;
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
                            &u_pred.seq0_i(),
                            &v.seq1(),
                            &x2_next.seqi_n(),
                            d_upred_v,
                            d_v_x2next,
                        ) + Sequence::tw3_with_travel(
                            &v_pred.seq0_i(),
                            &u.seq123(),
                            &y.seqi_n(),
                            d_vpred_u,
                            d_xnext_y,
                        );
                        consider!(11,4, 1, lb41, tw41);
                    }

                    if y.id != 0 {
                        let y_next = rv.node(pos2 + 2);
                        let distance_from_y = y.distance_row(data);
                        let send2_2_load = v.seq1().load + y.seq1().load;
                        let rem2_2 = v_pred.dist_to_succ + v.dist_to_succ + y.dist_to_succ;
                        let d_upred_y = copy_at(distance_from_u_pred, y.id);
                        let d_y_v = copy_at(distance_from_y, v.id);
                        let d_y_x2next = copy_at(distance_from_y, x2_next.id);
                        let d_xnext_ynext = copy_at(distance_from_x_next, y_next.id);
                        let cap_42_43 = cap_pen!(route1_load - send1_3_load + send2_2_load)
                            + cap_pen!(route2_load - send2_2_load + send1_3_load);
                        let dist_base = route_total_dist - rem1_3 - rem2_2;
                        let common_right = d_vpred_u + d_u_xnext + d_x_xnext + d_xnext_ynext;
                        if (dist_base.wrapping_add(common_right).wrapping_add(
                d_upred_v.wrapping_add(v.dist_to_succ).wrapping_add(d_y_x2next).min(d_upred_y.wrapping_add(d_y_v).wrapping_add(d_v_x2next))) as i64).wrapping_add(cap_42_43)<=group_limit {
let lb42 =
                            (dist_base + d_upred_v + v.dist_to_succ + d_y_x2next + common_right)
                                as i64
                                + cap_42_43;
                        let lb43 = (dist_base + d_upred_y + d_y_v + d_v_x2next + common_right)
                            as i64
                            + cap_42_43;
                        if lb42 <= max_acceptable_cost || lb43 <= max_acceptable_cost {
                            let right_tw = Sequence::tw3_with_travel(
                                &v_pred.seq0_i(),
                                &u.seq123(),
                                &y_next.seqi_n(),
                                d_vpred_u,
                                d_xnext_ynext,
                            );
                            if lb42 <= max_acceptable_cost {
                                let tw42 = Sequence::tw3_with_travel(
                                    &u_pred.seq0_i(),
                                    &v.seq12(),
                                    &x2_next.seqi_n(),
                                    d_upred_v,
                                    d_y_x2next,
                                ) + right_tw;
                                consider!(12,4, 2, lb42, tw42);
                            }
                            if lb43 <= max_acceptable_cost {
                                let tw43 = Sequence::tw3_with_travel(
                                    &u_pred.seq0_i(),
                                    &v.seq21(),
                                    &x2_next.seqi_n(),
                                    d_upred_y,
                                    d_v_x2next,
                                ) + right_tw;
                                consider!(13,4, 3, lb43, tw43);
                            }
                        }
            }

                        if y_next.id != 0 {
                            let y2_next = rv.node(pos2 + 3);
                            let send2_3_load = v.seq1().load + y.seq1().load + y_next.seq1().load;
                            let rem2_3 = v_pred.dist_to_succ
                                + v.dist_to_succ
                                + y.dist_to_succ
                                + y_next.dist_to_succ;
                            let d_ynext_x2next = copy_at(y_next.distance_row(data), x2_next.id);
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
                                    &u_pred.seq0_i(),
                                    &v.seq123(),
                                    &x2_next.seqi_n(),
                                    d_upred_v,
                                    d_ynext_x2next,
                                ) + Sequence::tw3_with_travel(
                                    &v_pred.seq0_i(),
                                    &u.seq123(),
                                    &y2_next.seqi_n(),
                                    d_vpred_u,
                                    d_xnext_y2next,
                                );
                                consider!(14,4, 4, lb44, tw44);
                            }
                        }
                    }
                }
            }
        }

        if PACKED {
            if best_key==i64::MAX {return NO_MOVE;}
            let cost=best_key>>4;
            if cost>max_acceptable_cost {return NO_MOVE;}
            const PLANS:[(u8,u8);15]=[(1,0),(1,1),(2,0),(3,0),(2,1),(3,1),(2,2),(3,2),(2,3),(3,3),(4,0),(4,1),(4,2),(4,3),(4,4)];
            let (send1,send2)=PLANS[(best_key&15) as usize];
            self.last_plan=MovePlan::InterRoute{send1,send2};return cost-old_total;
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
    #[cfg(target_arch="x86_64")]
    #[target_feature(enable="avx2,bmi2,popcnt")]
    #[inline(never)]
    unsafe fn evaluate_assured_customer_avx<const THREE: bool, const TABLE: bool,const MINPOS:bool>(
        &mut self,
        c1: usize,
        last_tested: usize,
        loop_id: usize,
    ) -> i64 {
        if loop_id != 2 && last_tested == self.nb_moves { return NO_MOVE; }
        self.query_credit=self.move_credit;
        let mut best_delta: i64 = i64::MAX; // best acceptable move
        let mut best_move: Option<CandidateMove> = None;
        let mut best_plan = MovePlan::None;
        let loc1 = copy_at(&self.node_locations,c1);
        let r1 = (loc1.route_and_position>>32) as usize;
        let pos1 = loc1.route_and_position as u32 as usize;
        let inter_source=self.make_inter_source::<THREE>(r1,pos1);

        {
            let r1_last_mod = copy_at(&self.when_last_modified,r1);
            let masks=self.active_neighbor_masks(c1,r1,r1_last_mod,last_tested);
            let neighbors_start = copy_at(&self.neighbors_before_offsets, c1);
            let neighbors_end = copy_at(&self.neighbors_before_offsets, c1 + 1);
            let count=neighbors_end-neighbors_start;
            let mut pending=masks.before & if count==64 {u64::MAX} else {(1u64<<count)-1};
            #[cfg(target_arch="x86_64")]
            let (batch_inter,batch_two)=if true {
                unsafe{self.batch_neighbor_bounds_inline_assured::<THREE>(r1,pos1,&self.neighbors_before[neighbors_start..neighbors_end],pending)}
            } else {(pending,pending)};
            #[cfg(not(target_arch="x86_64"))]
            let (batch_inter,batch_two)=(pending,pending);
            #[cfg(target_arch="x86_64")]
            let batch_reverse=if pos1==1 && true {
                unsafe{self.batch_reverse_bounds_inline_assured::<THREE>(r1,pos1,&self.neighbors_before[neighbors_start..neighbors_end],pending)}
            } else {pending};
            #[cfg(not(target_arch="x86_64"))]
            let batch_reverse=pending;
            pending &= batch_inter | batch_two | if pos1==1 {batch_reverse} else {0};
            while pending!=0 {
                let k=neighbors_start+pending.trailing_zeros() as usize;pending&=pending-1;
                let c2 = copy_at(&self.neighbors_before, k) as usize;
                let loc2 = copy_at(&self.node_locations,c2);
                let r2 = (loc2.route_and_position>>32) as usize;

                // Skip if both routes unchanged since last tests for this customer
                {
                    // We use pos2 + 1 for the SWAP and RELOCATE moves since c2 is a good predecessor for c1
                    // Moves listed here create the edge c2 => c1, but never insert immediately after a depot
                    let pos2 = loc2.route_and_position as u32 as usize;

                    let delta = if batch_inter&(1u64<<(k-neighbors_start))!=0 {
                        self.run_inter_source_assured::<THREE,TABLE>(r1,pos1,r2,pos2+1,&inter_source)
                    } else {NO_MOVE};
                    if delta < best_delta {
                        best_delta = delta;
                        best_move = Some(CandidateMove::InterRoute {
                            r1,
                            pos1,
                            r2,
                            pos2: pos2 + 1,
                        });
                        best_plan = self.last_plan;
                        if true {self.query_credit=delta.saturating_sub(1);}
                    }

                    // Special case to manage insert immediately after a depot
                    if pos1 == 1 {
                        let delta = if batch_reverse&(1u64<<(k-neighbors_start))!=0 {
                            self.run_inter_mode_assured::<THREE,TABLE>(r2,pos2,r1,pos1)
                        } else {NO_MOVE};
                        if delta < best_delta {
                            best_delta = delta;
                            best_move = Some(CandidateMove::InterRoute {
                                r1: r2,
                                pos1: pos2,
                                r2: r1,
                                pos2: pos1,
                            });
                            best_plan = self.last_plan;
                        if true {self.query_credit=delta.saturating_sub(1);}
                        }
                    }

                    let delta = if batch_two&(1u64<<(k-neighbors_start))!=0 {
                        self.run_2optstar_assured::<TABLE>(r1,pos1,r2,pos2+1)
                    } else {NO_MOVE};
                    if delta < best_delta {
                        best_delta = delta;
                        best_move = Some(CandidateMove::TwoOptStar {
                            r1,
                            pos1,
                            r2,
                            pos2: pos2 + 1,
                        });
                        best_plan = self.last_plan;
                        if true {self.query_credit=delta.saturating_sub(1);}
                    }
                }
            }

            let swap_source=self.make_swap_source(r1,pos1);
            let capacity_start = copy_at(&self.neighbors_capacity_swap_offsets, c1);
            let capacity_end = copy_at(&self.neighbors_capacity_swap_offsets, c1 + 1);
            let mut pending=masks.before.checked_shr((neighbors_end-neighbors_start) as u32).unwrap_or(0);
            while pending!=0 {
                let k=capacity_start+pending.trailing_zeros() as usize;pending&=pending-1;
                let c2 = copy_at(&self.neighbors_capacity_swap, k) as usize;
                let loc2 = copy_at(&self.node_locations,c2);
                let r2 = (loc2.route_and_position>>32) as usize;

                // Skip if both routes unchanged since last tests for this customer
                {
                    let pos2 = loc2.route_and_position as u32 as usize;
                    let delta = self.run_swapstar_fused_cut::<TABLE,MINPOS>(r1,pos1,r2,pos2,c2 as usize,&swap_source);
                    if delta < best_delta {
                        best_delta = delta;
                        best_move = Some(CandidateMove::SwapStar { r1, pos1, r2, pos2 });
                        best_plan = self.last_plan;
                        if true {self.query_credit=delta.saturating_sub(1);}
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
                        let delta = self.run_2optstar_assured::<TABLE>(r1, pos1, r2, pos2);
                        if delta < best_delta {
                            best_delta = delta;
                            best_move = Some(CandidateMove::TwoOptStar { r1, pos1, r2, pos2 });
                            best_plan = self.last_plan;
                        if true {self.query_credit=delta.saturating_sub(1);}
                        }
                    }

                    // Insert in an empty route
                    let delta = self.run_inter_source_assured::<THREE,TABLE>(r1,pos1,r2,pos2,&inter_source);
                    if delta < best_delta {
                        best_delta = delta;
                        best_move = Some(CandidateMove::InterRoute { r1, pos1, r2, pos2 });
                        best_plan = self.last_plan;
                        if true {self.query_credit=delta.saturating_sub(1);}
                    }
                }
            }
        }

        // Intra-route moves
        if copy_at(&self.when_last_modified, r1) > last_tested {
            let delta = self.run_intra_route_relocate_assured::<TABLE>(r1, pos1);
            if delta < best_delta {
                best_delta = delta;
                best_move = Some(CandidateMove::IntraRelocate { r1, pos1 });
                best_plan = self.last_plan;
                        if true {self.query_credit=delta.saturating_sub(1);}
            }

            let delta = self.run_intra_route_oropt2_assured::<TABLE>(r1, pos1);
            if delta < best_delta {
                best_delta = delta;
                best_move = Some(CandidateMove::IntraOrOpt2 { r1, pos1 });
                best_plan = self.last_plan;
                        if true {self.query_credit=delta.saturating_sub(1);}
            }

            let delta = self.run_intra_route_swap_assured::<TABLE>(r1, pos1);
            if delta < best_delta {
                best_delta = delta;
                best_move = Some(CandidateMove::IntraSwap { r1, pos1 });
                best_plan = self.last_plan;
                        if true {self.query_credit=delta.saturating_sub(1);}
            }

            let delta = self.run_2opt_assured::<TABLE>(r1, pos1);
            if delta < best_delta {
                best_delta = delta;
                best_move = Some(CandidateMove::Intra2Opt { r1, pos1 });
                best_plan = self.last_plan;
                        if true {self.query_credit=delta.saturating_sub(1);}
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
    #[inline(always)]
    fn donor_cut_budget(&self,id:usize)->i64 {
        let cut=unsafe{self.batch_cuts.get_unchecked(id)};
        cut.0[2] as i64-(cut.0[6]>>16) as i64*self.params.penalty_tw as i64
    }

    #[cfg(target_arch="x86_64")]
    #[target_feature(enable="avx2")]
    unsafe fn vector_arc_min_narrow(ids:&[i32],travel:&[i32],from:*const i32,to:*const i32)->i32 {
        use std::arch::x86_64::*;
        macro_rules! block {($k:expr)=>{{
            let a=_mm256_loadu_si256(ids.as_ptr().add($k) as *const __m256i);
            let b=_mm256_loadu_si256(ids.as_ptr().add($k+1) as *const __m256i);
            let ab=_mm256_loadu_si256(travel.as_ptr().add($k) as *const __m256i);
            let av=_mm256_i32gather_epi32::<4>(to,a);let vb=_mm256_i32gather_epi32::<4>(from,b);
            _mm256_add_epi32(_mm256_add_epi32(av,vb),ab)
        }}}
        // Route arrays are padded to a multiple of eight, with one extra ID.
        // The specialized arms read exactly the same blocks as the loop.
        let best=match travel.len() {
            8=>block!(0),
            16=>_mm256_min_epi32(block!(0),block!(8)),
            24=>_mm256_min_epi32(_mm256_min_epi32(block!(0),block!(8)),block!(16)),
            32=>_mm256_min_epi32(_mm256_min_epi32(block!(0),block!(8)),_mm256_min_epi32(block!(16),block!(24))),
            _=>{let mut best=_mm256_set1_epi32(i32::MAX);let mut k=0;
                while k<travel.len(){best=_mm256_min_epi32(best,block!(k));k+=8;}best},
        };
        // True detours lie in [-16383,32766]. Padding is positive and
        // saturates to 65535; at least one real arc always remains below it.
        let packed=_mm_packus_epi32(_mm256_castsi256_si128(best),_mm256_extracti128_si256::<1>(best));
        let result=(_mm_cvtsi128_si32(_mm_minpos_epu16(packed))&65535)-16_384;
        
        
        result
    }
}
