use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::knapsack::{Challenge, Solution};

#[allow(dead_code, unused_imports, clippy::all)]
mod inner {
    use anyhow::Result;
    use serde::{Deserialize, Serialize};
    use serde_json::{Map, Value};
    use tig_challenges::knapsack::*;

    pub type Ll  = i64;
    pub type Ull = u64;

    pub const NEG_INF: Ll = -9_000_000_000_000_000_000i64;

    #[derive(Serialize, Deserialize)]
    pub struct Hyperparameters {
        pub n_lambda_values: Option<usize>,
        pub ils_rounds:      Option<usize>,
        pub window_k:        Option<usize>,
        pub core_half_dp:    Option<usize>,
    }

    struct Hparams {
        n_lambda_values: usize,
        ils_rounds:      usize,
        window_k:        usize,
        core_half_dp:    usize,
    }

    impl Hparams {
        fn for_size(n: usize) -> Self {
            if n <= 1200 {
                Self {
                    n_lambda_values: 1600,
                    ils_rounds:      8,
                    window_k:        250,
                    core_half_dp:    50,
                }
            } else if n <= 2000 {
                Self {
                    n_lambda_values: 1600,
                    ils_rounds:      400,
                    window_k:        220,
                    core_half_dp:    60,
                }
            } else if n <= 4000 {
                Self {
                    n_lambda_values: 1600,
                    ils_rounds:      200,
                    window_k:        200,
                    core_half_dp:    50,
                }
            } else {
                Self {
                    n_lambda_values: 1600,
                    ils_rounds:      100,
                    window_k:        180,
                    core_half_dp:    40,
                }
            }
        }

        fn from_map(h: &Option<Map<String, Value>>, n: usize) -> Self {
            let mut p = Self::for_size(n);
            if let Some(m) = h {
                if let Some(v) = m.get("n_lambda_values").and_then(|v| v.as_u64()) {
                    p.n_lambda_values = v as usize;
                }
                if let Some(v) = m.get("ils_rounds").and_then(|v| v.as_u64()) {
                    p.ils_rounds = v as usize;
                }
                if let Some(v) = m.get("window_k").and_then(|v| v.as_u64()) {
                    p.window_k = v as usize;
                }
                if let Some(v) = m.get("core_half_dp").and_then(|v| v.as_u64()) {
                    p.core_half_dp = v as usize;
                }
            }
            p
        }
    }

    #[derive(Debug, Clone)]
    pub struct Edge { pub i: usize, pub j: usize, pub value: f64 }

    #[derive(Debug)]
    pub struct P1Instance {
        pub n_items:   usize,
        pub n_edges:   usize,
        pub edges:     Vec<Edge>,
        pub weights:   Vec<i32>,
        pub n_budgets: usize,
        pub budgets:   Vec<i32>,
    }

    pub fn challenge_to_p1(challenge: &Challenge) -> P1Instance {
        let n = challenge.num_items;
        let mut edges: Vec<Edge> = Vec::new();

        for i in 0..n {
            edges.push(Edge { i, j: i, value: challenge.values[i] as f64 });
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let v = challenge.interaction_values[i][j];
                if v != 0 {
                    edges.push(Edge { i, j, value: v as f64 });
                }
            }
        }

        let weights: Vec<i32> = challenge.weights.iter().map(|&w| w as i32).collect();
        let budgets: Vec<i32> = vec![challenge.max_weight as i32];

        P1Instance {
            n_items:   n,
            n_edges:   edges.len(),
            edges,
            weights,
            n_budgets: 1,
            budgets,
        }
    }

    #[derive(Debug)]
    pub struct BudgetResult {
        pub budget:         i32,
        pub ofv:            f64,
        pub cpu:            f64,
        pub selected_items: Vec<usize>,
    }

    #[derive(Debug)]
    pub struct QKPResult { pub results: Vec<BudgetResult> }

    pub struct UtilMatrix {
        pub n:      usize,
        pub data:   Vec<f64>,
        pub linear: Vec<f64>,
    }
    impl UtilMatrix {
        pub fn get(&self, r: usize, c: usize) -> f64 { self.data[r * self.n + c] }
        pub fn set(&mut self, r: usize, c: usize, v: f64) { self.data[r * self.n + c] = v; }
    }

    pub fn build_utility_matrix(n: usize, edges: &[Edge]) -> UtilMatrix {
        let mut data   = vec![0.0f64; n * n];
        let mut linear = vec![0.0f64; n];
        for e in edges {
            if e.i == e.j {
                linear[e.i] += e.value;
            } else {
                data[e.i * n + e.j] = e.value;
                data[e.j * n + e.i] = e.value;
            }
        }
        UtilMatrix { n, data, linear }
    }

    pub fn compute_ofv(sel: &[usize], n: usize, edges: &[Edge]) -> f64 {
        let mut chosen = vec![0u8; n];
        for &i in sel {
            if i < n { chosen[i] = 1; }
        }
        let mut ofv = 0.0f64;
        for e in edges {
            if e.i >= n || e.j >= n { continue; }
            if e.i == e.j {
                if chosen[e.i] != 0 { ofv += e.value; }
            } else if chosen[e.i] != 0 && chosen[e.j] != 0 {
                ofv += e.value;
            }
        }
        ofv
    }

    mod hpf {
        use super::*;

        pub struct HpfArc {
            pub from:       *mut HpfNode,
            pub to:         *mut HpfNode,
            pub flow:       f32,
            pub capacity:   f32,
            pub direction:  i32,
            pub capacities: Vec<f32>,
        }

        pub struct HpfNode {
            pub wt:              f32,
            pub cst:             f32,
            pub visited:         i32,
            pub num_adjacent:    i32,
            pub number:          i32,
            pub label:           i32,
            pub excess:          f32,
            pub parent:          *mut HpfNode,
            pub child_list:      *mut HpfNode,
            pub next_scan:       *mut HpfNode,
            pub num_out_of_tree: i32,
            pub out_of_tree:     Vec<*mut HpfArc>,
            pub next_arc:        i32,
            pub arc_to_parent:   *mut HpfArc,
            pub next:            *mut HpfNode,
            pub prev:            *mut HpfNode,
            pub breakpoint:      i32,
        }

        impl HpfNode {
            pub fn zeroed(num_params: i32) -> Self {
                HpfNode {
                    wt: 0.0, cst: 1.0, visited: 0, num_adjacent: 0,
                    number: 0, label: 0, excess: 0.0,
                    parent: std::ptr::null_mut(),
                    child_list: std::ptr::null_mut(),
                    next_scan: std::ptr::null_mut(),
                    num_out_of_tree: 0,
                    out_of_tree: Vec::new(),
                    next_arc: 0,
                    arc_to_parent: std::ptr::null_mut(),
                    next: std::ptr::null_mut(),
                    prev: std::ptr::null_mut(),
                    breakpoint: num_params + 1,
                }
            }
        }

        pub struct HpfRoot { pub start: *mut HpfNode, pub end: *mut HpfNode }

        pub struct HpfState {
            pub num_nodes:            i32,
            pub num_arcs:             i32,
            pub source:               i32,
            pub sink:                 i32,
            pub num_params:           i32,
            pub highest_strong_label: i32,
            pub max_degree_ratio:     f32,
            pub step:                 f32,
            pub adjacency_list:       Vec<HpfNode>,
            pub strong_roots:         Vec<HpfRoot>,
            pub label_count:          Vec<i32>,
            pub arc_list:             Vec<HpfArc>,
        }

        unsafe fn init_root(num_params: i32) -> HpfRoot {
            let start = Box::into_raw(Box::new(HpfNode::zeroed(num_params)));
            let end   = Box::into_raw(Box::new(HpfNode::zeroed(num_params)));
            (*start).next = end;
            (*end).prev   = start;
            HpfRoot { start, end }
        }

        unsafe fn free_root(root: &HpfRoot) {
            drop(Box::from_raw(root.start));
            drop(Box::from_raw(root.end));
        }

        unsafe fn add_to_strong_bucket(new_root: *mut HpfNode, root_end: *mut HpfNode) {
            (*new_root).next         = root_end;
            (*new_root).prev         = (*root_end).prev;
            (*root_end).prev         = new_root;
            (*(*new_root).prev).next = new_root;
        }

        unsafe fn lift_all(s: *mut HpfState, root_node: *mut HpfNode, theparam: i32) {
            let mut current = root_node;
            (*current).next_scan = (*current).child_list;
            (*s).label_count[(*current).label as usize] -= 1;
            (*current).label      = (*s).num_nodes;
            (*current).breakpoint = theparam + 1;
            loop {
                while !(*current).next_scan.is_null() {
                    let temp             = (*current).next_scan;
                    (*current).next_scan = (*(*current).next_scan).next;
                    current              = temp;
                    (*current).next_scan = (*current).child_list;
                    (*s).label_count[(*current).label as usize] -= 1;
                    (*current).label      = (*s).num_nodes;
                    (*current).breakpoint = theparam + 1;
                }
                if (*current).parent.is_null() { break; }
                current = (*current).parent;
            }
        }

        unsafe fn add_relationship(new_parent: *mut HpfNode, child: *mut HpfNode) {
            (*child).parent          = new_parent;
            (*child).next            = (*new_parent).child_list;
            (*new_parent).child_list = child;
        }

        unsafe fn break_relationship(old_parent: *mut HpfNode, child: *mut HpfNode) {
            (*child).parent = std::ptr::null_mut();
            if (*old_parent).child_list == child {
                (*old_parent).child_list = (*child).next;
                (*child).next = std::ptr::null_mut();
                return;
            }
            let mut current = (*old_parent).child_list;
            while (*current).next != child { current = (*current).next; }
            (*current).next = (*child).next;
            (*child).next   = std::ptr::null_mut();
        }

        unsafe fn hpf_merge(parent: *mut HpfNode, child: *mut HpfNode,
                             new_arc: *mut HpfArc) {
            let mut current    = child;
            let mut new_parent = parent;
            let mut new_arc    = new_arc;
            while !(*current).parent.is_null() {
                let old_arc              = (*current).arc_to_parent;
                (*current).arc_to_parent = new_arc;
                let old_parent           = (*current).parent;
                break_relationship(old_parent, current);
                add_relationship(new_parent, current);
                new_parent           = current;
                current              = old_parent;
                new_arc              = old_arc;
                (*new_arc).direction = 1 - (*new_arc).direction;
            }
            (*current).arc_to_parent = new_arc;
            add_relationship(new_parent, current);
        }

        unsafe fn push_upward(s: *mut HpfState, arc: *mut HpfArc,
                               child: *mut HpfNode, parent: *mut HpfNode,
                               res_cap: f32) {
            if res_cap >= (*child).excess {
                (*parent).excess += (*child).excess;
                (*arc).flow      += (*child).excess;
                (*child).excess   = 0.0;
                return;
            }
            (*arc).direction  = 0;
            (*parent).excess += res_cap;
            (*child).excess  -= res_cap;
            (*arc).flow       = (*arc).capacity;
            (*parent).out_of_tree.push(arc);
            (*parent).num_out_of_tree += 1;
            break_relationship(parent, child);
            let lbl = (*child).label as usize;
            add_to_strong_bucket(child, (*s).strong_roots[lbl].end);
        }

        unsafe fn push_downward(s: *mut HpfState, arc: *mut HpfArc,
                                 child: *mut HpfNode, parent: *mut HpfNode,
                                 flow: f32) {
            if flow >= (*child).excess {
                (*parent).excess += (*child).excess;
                (*arc).flow      -= (*child).excess;
                (*child).excess   = 0.0;
                return;
            }
            (*arc).direction  = 1;
            (*child).excess  -= flow;
            (*parent).excess += flow;
            (*arc).flow       = 0.0;
            (*parent).out_of_tree.push(arc);
            (*parent).num_out_of_tree += 1;
            break_relationship(parent, child);
            let lbl = (*child).label as usize;
            add_to_strong_bucket(child, (*s).strong_roots[lbl].end);
        }

        unsafe fn push_excess(s: *mut HpfState, strong_root: *mut HpfNode) {
            let mut current = strong_root;
            while (*current).excess > 0.0 && !(*current).parent.is_null() {
                let parent = (*current).parent;
                let arc    = (*current).arc_to_parent;
                if (*arc).direction != 0 {
                    push_upward(s, arc, current, parent, (*arc).capacity - (*arc).flow);
                } else {
                    push_downward(s, arc, current, parent, (*arc).flow);
                }
                current = parent;
            }
            if (*current).excess > 0.0 && (*current).next.is_null() {
                let lbl = (*current).label as usize;
                add_to_strong_bucket(current, (*s).strong_roots[lbl].end);
            }
        }

        unsafe fn find_weak_node(s: *mut HpfState, strong_node: *mut HpfNode,
                                  weak_node: *mut *mut HpfNode) -> *mut HpfArc {
            let target = (*s).highest_strong_label - 1;
            let size   = (*strong_node).num_out_of_tree as usize;
            let mut i  = (*strong_node).next_arc as usize;
            while i < size {
                let out = (*strong_node).out_of_tree[i];
                if (*(*out).to).label == target {
                    (*strong_node).next_arc = i as i32;
                    *weak_node = (*out).to;
                    let last = (*strong_node).num_out_of_tree as usize - 1;
                    (*strong_node).out_of_tree[i] = (*strong_node).out_of_tree[last];
                    (*strong_node).out_of_tree.pop();
                    (*strong_node).num_out_of_tree -= 1;
                    return out;
                } else if (*(*out).from).label == target {
                    (*strong_node).next_arc = i as i32;
                    *weak_node = (*out).from;
                    let last = (*strong_node).num_out_of_tree as usize - 1;
                    (*strong_node).out_of_tree[i] = (*strong_node).out_of_tree[last];
                    (*strong_node).out_of_tree.pop();
                    (*strong_node).num_out_of_tree -= 1;
                    return out;
                }
                i += 1;
            }
            (*strong_node).next_arc = (*strong_node).num_out_of_tree;
            std::ptr::null_mut()
        }

        unsafe fn check_children(s: *mut HpfState, cur: *mut HpfNode) {
            while !(*cur).next_scan.is_null() {
                if (*(*cur).next_scan).label == (*cur).label { return; }
                (*cur).next_scan = (*(*cur).next_scan).next;
            }
            (*s).label_count[(*cur).label as usize] -= 1;
            (*cur).label += 1;
            (*s).label_count[(*cur).label as usize] += 1;
            (*cur).next_arc = 0;
        }

        unsafe fn process_root(s: *mut HpfState, strong_root: *mut HpfNode) {
            let mut strong_node = strong_root;
            let mut weak_node: *mut HpfNode = std::ptr::null_mut();
            (*strong_root).next_scan = (*strong_root).child_list;
            let out = find_weak_node(s, strong_root, &mut weak_node);
            if !out.is_null() {
                hpf_merge(weak_node, strong_node, out);
                push_excess(s, strong_root);
                return;
            }
            check_children(s, strong_root);
            loop {
                while !(*strong_node).next_scan.is_null() {
                    let temp                 = (*strong_node).next_scan;
                    (*strong_node).next_scan = (*(*strong_node).next_scan).next;
                    strong_node              = temp;
                    (*strong_node).next_scan = (*strong_node).child_list;
                    let out = find_weak_node(s, strong_node, &mut weak_node);
                    if !out.is_null() {
                        hpf_merge(weak_node, strong_node, out);
                        push_excess(s, strong_root);
                        return;
                    }
                    check_children(s, strong_node);
                }
                if (*strong_node).parent.is_null() { break; }
                strong_node = (*strong_node).parent;
                check_children(s, strong_node);
            }
            let lbl = (*strong_root).label as usize;
            add_to_strong_bucket(strong_root, (*s).strong_roots[lbl].end);
            (*s).highest_strong_label += 1;
        }

        unsafe fn get_highest_strong_root(s: *mut HpfState,
                                           theparam: i32) -> *mut HpfNode {
            let mut i = (*s).highest_strong_label;
            while i > 0 {
                if (*(*s).strong_roots[i as usize].start).next
                    != (*s).strong_roots[i as usize].end
                {
                    (*s).highest_strong_label = i;
                    if (*s).label_count[(i - 1) as usize] > 0 {
                        let sr = (*(*s).strong_roots[i as usize].start).next;
                        (*(*sr).next).prev = (*sr).prev;
                        (*(*sr).prev).next = (*sr).next;
                        (*sr).next = std::ptr::null_mut();
                        return sr;
                    }
                    while (*(*s).strong_roots[i as usize].start).next
                        != (*s).strong_roots[i as usize].end
                    {
                        let sr = (*(*s).strong_roots[i as usize].start).next;
                        (*(*sr).next).prev = (*sr).prev;
                        (*(*sr).prev).next = (*sr).next;
                        lift_all(s, sr, theparam);
                    }
                }
                i -= 1;
            }
            if (*(*s).strong_roots[0].start).next == (*s).strong_roots[0].end {
                return std::ptr::null_mut();
            }
            while (*(*s).strong_roots[0].start).next != (*s).strong_roots[0].end {
                let sr = (*(*s).strong_roots[0].start).next;
                (*(*sr).next).prev = (*sr).prev;
                (*(*sr).prev).next = (*sr).next;
                (*sr).label = 1;
                (*s).label_count[0] -= 1;
                (*s).label_count[1] += 1;
                let lbl = (*sr).label as usize;
                add_to_strong_bucket(sr, (*s).strong_roots[lbl].end);
            }
            (*s).highest_strong_label = 1;
            let sr = (*(*s).strong_roots[1].start).next;
            (*(*sr).next).prev = (*sr).prev;
            (*(*sr).prev).next = (*sr).next;
            (*sr).next = std::ptr::null_mut();
            sr
        }

        unsafe fn update_capacities(s: *mut HpfState, theparam: i32) {
            let lambda = (*s).max_degree_ratio - theparam as f32 * (*s).step;
            let src_idx = ((*s).source - 1) as usize;
            let size    = (*s).adjacency_list[src_idx].num_out_of_tree as usize;
            for i in 0..size {
                let arc = (*s).adjacency_list[src_idx].out_of_tree[i];
                let nd  = (*arc).to;
                let v   = (*nd).wt - lambda * (*nd).cst;
                let new_capacity = if v > 0.0 { v } else { 0.0 };
                let delta = new_capacity - (*arc).capacity;
                if delta < 0.0 { return; }
                (*arc).capacity      = new_capacity;
                (*arc).flow         += delta;
                (*nd).excess        += delta;
                if (*nd).label < (*s).num_nodes && (*nd).excess > 0.0 {
                    push_excess(s, nd);
                }
            }
            let snk_idx = ((*s).sink - 1) as usize;
            let size    = (*s).adjacency_list[snk_idx].num_out_of_tree as usize;
            for i in 0..size {
                let arc = (*s).adjacency_list[snk_idx].out_of_tree[i];
                let nd  = (*arc).from;
                let v   = lambda * (*nd).cst - (*nd).wt;
                let new_capacity = if v > 0.0 { v } else { 0.0 };
                let delta = new_capacity - (*arc).capacity;
                if delta > 0.0 { return; }
                (*arc).capacity      = new_capacity;
                (*arc).flow         += delta;
                (*nd).excess        -= delta;
                if (*nd).label < (*s).num_nodes && (*nd).excess > 0.0 {
                    push_excess(s, nd);
                }
            }
            (*s).highest_strong_label = (*s).num_nodes - 1;
        }

        unsafe fn simple_initialization(s: *mut HpfState) {
            let src_idx = ((*s).source - 1) as usize;
            let snk_idx = ((*s).sink   - 1) as usize;
            let size = (*s).adjacency_list[src_idx].num_out_of_tree as usize;
            for i in 0..size {
                let arc = (*s).adjacency_list[src_idx].out_of_tree[i];
                (*arc).flow = (*arc).capacity;
                (*(*arc).to).excess += (*arc).capacity;
            }
            let size = (*s).adjacency_list[snk_idx].num_out_of_tree as usize;
            for i in 0..size {
                let arc = (*s).adjacency_list[snk_idx].out_of_tree[i];
                (*arc).flow = (*arc).capacity;
                (*(*arc).from).excess -= (*arc).capacity;
            }
            (*s).adjacency_list[src_idx].excess = 0.0;
            (*s).adjacency_list[snk_idx].excess = 0.0;
            for i in 0..(*s).num_nodes as usize {
                if (*s).adjacency_list[i].excess > 0.0 {
                    (*s).adjacency_list[i].label = 1;
                    (*s).label_count[1] += 1;
                    let nd  = &mut (*s).adjacency_list[i] as *mut HpfNode;
                    let end = (*s).strong_roots[1].end;
                    add_to_strong_bucket(nd, end);
                }
            }
            (*s).adjacency_list[src_idx].label      = (*s).num_nodes;
            (*s).adjacency_list[src_idx].breakpoint = 0;
            (*s).adjacency_list[snk_idx].label      = 0;
            (*s).adjacency_list[snk_idx].breakpoint = (*s).num_params + 2;
            (*s).label_count[0] = ((*s).num_nodes - 2) - (*s).label_count[1];
        }

        unsafe fn pseudoflow_phase1(s: *mut HpfState) {
            let mut theparam = 0i32;
            loop {
                let sr = get_highest_strong_root(s, theparam);
                if sr.is_null() { break; }
                process_root(s, sr);
            }
            theparam = 1;
            while theparam < (*s).num_params {
                update_capacities(s, theparam);
                loop {
                    let sr = get_highest_strong_root(s, theparam);
                    if sr.is_null() { break; }
                    process_root(s, sr);
                }
                theparam += 1;
            }
        }

        pub struct BreakpointSets {
            pub sets: Vec<(i32, Vec<usize>)>,
        }

        pub fn get_breakpoints(inst: &P1Instance,
                                n_lambda_values: usize) -> BreakpointSets {
            let n_items    = inst.n_items;
            let n_edges    = inst.n_edges;
            let num_nodes  = (n_items + 2) as i32;
            let num_arcs   = (n_edges + n_items) as i32;
            let num_params = n_lambda_values as i32;

            let mut adjacency_list: Vec<HpfNode> = (0..num_nodes as usize)
                .map(|i| {
                    let mut nd = HpfNode::zeroed(num_params);
                    nd.number  = (i + 1) as i32;
                    nd.cst     = 1.0;
                    nd
                })
                .collect();

            for i in 0..n_items {
                adjacency_list[i + 2].cst = inst.weights[i] as f32;
            }

            let mut arc_list: Vec<HpfArc> = Vec::with_capacity(num_arcs as usize);
            for _ in 0..num_arcs as usize {
                arc_list.push(HpfArc {
                    from: std::ptr::null_mut(), to: std::ptr::null_mut(),
                    flow: 0.0, capacity: 0.0, direction: 1, capacities: Vec::new(),
                });
            }

            let mut first = 0usize;
            for e in &inst.edges {
                let from_id = e.i;
                let to_id   = e.j;
                let cap     = e.value as f32;
                if from_id == to_id {
                    adjacency_list[from_id + 2].wt += cap;
                    continue;
                }
                arc_list[first].capacity   = cap;
                arc_list[first].direction  = 1;
                adjacency_list[from_id + 2].wt += cap;
                adjacency_list[from_id + 2].num_adjacent += 1;
                adjacency_list[to_id   + 2].num_adjacent += 1;
                first += 1;
            }

            let mut max_degree_ratio = 0.0f32;
            for i in 2..num_nodes as usize {
                let ratio = adjacency_list[i].wt / adjacency_list[i].cst;
                if ratio > max_degree_ratio { max_degree_ratio = ratio; }
            }

            let step: f32 = if num_params > 0 {
                max_degree_ratio / num_params as f32
            } else { 0.0 };
            let initial_lambda = if num_params > 0 { max_degree_ratio } else { 0.0 };

            for i in 2..num_nodes as usize {
                let wt  = adjacency_list[i].wt;
                let cst = adjacency_list[i].cst;

                let src_cap = if num_params > 0 {
                    let v = wt - initial_lambda * cst;
                    if v > 0.0 { v } else { 0.0 }
                } else { 0.0 };
                arc_list[first].capacity   = src_cap;
                arc_list[first].direction  = 1;
                adjacency_list[0].num_adjacent += 1;
                adjacency_list[i].num_adjacent += 1;
                first += 1;

                let snk_cap = if num_params > 0 {
                    let v = initial_lambda * cst - wt;
                    if v > 0.0 { v } else { 0.0 }
                } else { 0.0 };
                arc_list[first].capacity   = snk_cap;
                arc_list[first].direction  = 1;
                adjacency_list[i].num_adjacent += 1;
                adjacency_list[1].num_adjacent += 1;
                first += 1;
            }

            let strong_roots: Vec<HpfRoot> = unsafe {
                (0..num_nodes as usize).map(|_| init_root(num_params)).collect()
            };
            let label_count = vec![0i32; num_nodes as usize];

            let mut state = HpfState {
                num_nodes, num_arcs, source: 1, sink: 2,
                num_params,
                highest_strong_label: 1,
                max_degree_ratio,
                step,
                adjacency_list,
                strong_roots,
                label_count,
                arc_list,
            };

            unsafe {
                let s = &mut state as *mut HpfState;

                let mut first = 0usize;
                for e in &inst.edges {
                    let from_id = e.i;
                    let to_id   = e.j;
                    if from_id == to_id { continue; }
                    (*s).arc_list[first].from = &mut (*s).adjacency_list[from_id + 2];
                    (*s).arc_list[first].to   = &mut (*s).adjacency_list[to_id   + 2];
                    first += 1;
                }
                for i in 2..num_nodes as usize {
                    (*s).arc_list[first].from = &mut (*s).adjacency_list[0];
                    (*s).arc_list[first].to   = &mut (*s).adjacency_list[i];
                    first += 1;
                    (*s).arc_list[first].from = &mut (*s).adjacency_list[i];
                    (*s).arc_list[first].to   = &mut (*s).adjacency_list[1];
                    first += 1;
                }

                for i in 0..num_arcs as usize {
                    let to_num   = (*(*s).arc_list[i].to).number;
                    let from_num = (*(*s).arc_list[i].from).number;
                    let cap      = (*s).arc_list[i].capacity;
                    let source   = (*s).source;
                    let sink     = (*s).sink;
                    if source == to_num || sink == from_num || from_num == to_num {
                        continue;
                    }
                    if source == from_num && to_num == sink {
                        (*s).arc_list[i].flow = cap;
                    } else if from_num == source {
                        let arc_ptr = &mut (*s).arc_list[i] as *mut HpfArc;
                        (*s).adjacency_list[(from_num - 1) as usize].out_of_tree.push(arc_ptr);
                        (*s).adjacency_list[(from_num - 1) as usize].num_out_of_tree += 1;
                    } else if to_num == sink {
                        let arc_ptr = &mut (*s).arc_list[i] as *mut HpfArc;
                        (*s).adjacency_list[(to_num - 1) as usize].out_of_tree.push(arc_ptr);
                        (*s).adjacency_list[(to_num - 1) as usize].num_out_of_tree += 1;
                    } else {
                        let arc_ptr = &mut (*s).arc_list[i] as *mut HpfArc;
                        (*s).adjacency_list[(from_num - 1) as usize].out_of_tree.push(arc_ptr);
                        (*s).adjacency_list[(from_num - 1) as usize].num_out_of_tree += 1;
                    }
                }

                simple_initialization(s);
                pseudoflow_phase1(s);

                let mut pos_items: Vec<(i32, usize)> = Vec::new();
                for i in 0..num_nodes as usize {
                    let node_num = (*s).adjacency_list[i].number;
                    if node_num == 1 || node_num == 2 { continue; }
                    pos_items.push((
                        (*s).adjacency_list[i].breakpoint,
                        (node_num - 3) as usize,
                    ));
                }

                pos_items.sort_unstable_by_key(|&(pos, _)| pos);
                let mut sets: Vec<(i32, Vec<usize>)> = Vec::new();
                let mut start = 0usize;
                while start < pos_items.len() {
                    let pos = pos_items[start].0;
                    let mut end = start + 1;
                    while end < pos_items.len() && pos_items[end].0 == pos {
                        end += 1;
                    }
                    let mut nodes = Vec::with_capacity(end - start);
                    nodes.extend(pos_items[start..end].iter().map(|&(_, item)| item));
                    sets.push((pos, nodes));
                    start = end;
                }

                for i in 0..num_nodes as usize { free_root(&(*s).strong_roots[i]); }

                BreakpointSets { sets }
            }
        }
    } 

    pub use hpf::{BreakpointSets, get_breakpoints};

    pub type IntArray = Vec<usize>;
    pub type DblArray = Vec<f64>;

    #[inline] pub fn ia_contains(a: &IntArray, v: usize) -> bool { a.contains(&v) }

    fn get_initial_node(right_nodes: &IntArray, um: &UtilMatrix,
                        weights: &[i32], budget: i32) -> Option<usize> {
        let mut best: Option<usize> = None;
        let mut best_val = f64::NEG_INFINITY;
        for &nd in right_nodes {
            if weights[nd] > budget { continue; }
            let util: f64 = right_nodes.iter().map(|&m| um.get(nd, m)).sum::<f64>()
                / weights[nd] as f64;
            if util > best_val { best_val = util; best = Some(nd); }
        }
        best
    }

    struct WarmStartLeft {
        valid:                bool,
        candidate_nodes:      IntArray,
        candidate_contribs:   DblArray,
        current_total_weight: f64,
        left_nodes:           IntArray,
    }
    impl WarmStartLeft {
        fn new() -> Self {
            WarmStartLeft {
                valid: false,
                candidate_nodes: Vec::new(),
                candidate_contribs: Vec::new(),
                left_nodes: Vec::new(),
                current_total_weight: -1.0,
            }
        }
        fn reset(&mut self) {
            self.valid = false;
            self.candidate_nodes.clear();
            self.candidate_contribs.clear();
            self.left_nodes.clear();
            self.current_total_weight = -1.0;
        }
    }

    struct WarmStartRight {
        valid:                bool,
        candidate_nodes:      IntArray,
        candidate_contribs:   DblArray,
        current_total_weight: f64,
    }
    impl WarmStartRight {
        fn new() -> Self {
            WarmStartRight {
                valid: false,
                candidate_nodes: Vec::new(),
                candidate_contribs: Vec::new(),
                current_total_weight: 0.0,
            }
        }
        fn reset(&mut self) {
            self.valid = false;
            self.candidate_nodes.clear();
            self.candidate_contribs.clear();
            self.current_total_weight = 0.0;
        }
    }

    fn run_greedy_left_beta0(um:          &UtilMatrix,
                             mut left:    IntArray,
                             right_nodes: &IntArray,
                             budget:      i32,
                             weights:     &[i32],
                             mut ws:      Option<&mut WarmStartLeft>) -> IntArray {
        if left.is_empty() {
            if let Some(nd) = get_initial_node(right_nodes, um, weights, budget) {
                left.push(nd);
            }
        }
        let mut cur_w: f64 = match &ws {
            Some(w) if w.valid && w.current_total_weight >= 0.0 => w.current_total_weight,
            _ => left.iter().map(|&k| weights[k] as f64).sum(),
        };
        let mut cand_nodes:    IntArray = Vec::new();
        let mut cand_contribs: DblArray = Vec::new();
        let ws_valid          = ws.as_ref().map(|w| w.valid).unwrap_or(false);
        let ws_cands_nonempty = ws.as_ref().map(|w| !w.candidate_nodes.is_empty()).unwrap_or(false);
        let mut update_flag;
        if ws_valid && ws_cands_nonempty {
            let w = ws.as_ref().unwrap();
            let mut all_fit = true;
            let rem = budget as f64 - cur_w;
            for (idx, &nd) in w.candidate_nodes.iter().enumerate() {
                if weights[nd] as f64 <= rem {
                    cand_nodes.push(nd);
                    cand_contribs.push(w.candidate_contribs[idx]);
                } else { all_fit = false; }
            }
            update_flag = all_fit;
        } else {
            for &nd in right_nodes {
                if ia_contains(&left, nd) { continue; }
                if weights[nd] as f64 > budget as f64 - cur_w { continue; }
                cand_nodes.push(nd);
            }
            update_flag = true;
            for &nd in &cand_nodes {
                let mut contrib: f64 = left.iter()
                    .map(|&m| um.get(nd, m)).sum();
                contrib += um.linear[nd];
                contrib /= weights[nd] as f64;
                cand_contribs.push(contrib);
            }
        }
        loop {
            if cand_nodes.is_empty() { break; }
            let best_idx = cand_contribs.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap();
            let best_node = cand_nodes[best_idx];
            left.push(best_node);
            cur_w += weights[best_node] as f64;
            let mut new_cands:    IntArray = Vec::new();
            let mut new_contribs: DblArray = Vec::new();
            let mut all_fit = true;
            let rem = budget as f64 - cur_w;
            for (k, &nd) in cand_nodes.iter().enumerate() {
                if k == best_idx { continue; }
                if weights[nd] as f64 > rem { all_fit = false; continue; }
                new_cands.push(nd);
                new_contribs.push(cand_contribs[k]);
            }
            for (k, &nd) in new_cands.iter().enumerate() {
                new_contribs[k] += um.get(nd, best_node) / weights[nd] as f64;
            }
            if let Some(ref mut ws_ref) = ws {
                if update_flag && all_fit {
                    ws_ref.left_nodes           = left.clone();
                    ws_ref.candidate_nodes      = new_cands.clone();
                    ws_ref.candidate_contribs   = new_contribs.clone();
                    ws_ref.current_total_weight = cur_w;
                    ws_ref.valid                = true;
                } else { update_flag = false; }
            }
            cand_nodes    = new_cands;
            cand_contribs = new_contribs;
        }
        left
    }

    fn run_greedy_right_beta0(um:              &UtilMatrix,
                              mut right_nodes: IntArray,
                              budget:          i32,
                              weights:         &[i32],
                              ws:              Option<&mut WarmStartRight>) -> IntArray {
        if right_nodes.is_empty() { return right_nodes; }
        let mut cur_w: f64 = match &ws {
            Some(w) if w.valid => w.current_total_weight,
            _ => right_nodes.iter().map(|&k| weights[k] as f64).sum(),
        };
        let ws_valid          = ws.as_ref().map(|w| w.valid).unwrap_or(false);
        let ws_cands_nonempty = ws.as_ref().map(|w| !w.candidate_nodes.is_empty()).unwrap_or(false);
        let mut cand_nodes:    IntArray = Vec::new();
        let mut cand_contribs: DblArray = Vec::new();
        if ws_valid && ws_cands_nonempty {
            let w = ws.as_ref().unwrap();
            cand_nodes    = w.candidate_nodes.clone();
            cand_contribs = w.candidate_contribs.clone();
        } else {
            for &nd in &right_nodes {
                let mut contrib: f64 = right_nodes.iter()
                    .map(|&m| -um.get(nd, m)).sum();
                contrib -= um.linear[nd];
                contrib /= weights[nd] as f64;
                cand_nodes.push(nd);
                cand_contribs.push(contrib);
            }
        }
        while !cand_nodes.is_empty() && cur_w > budget as f64 {
            let best_idx = cand_contribs.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap();
            let best_node = cand_nodes[best_idx];
            right_nodes.retain(|&x| x != best_node);
            cur_w -= weights[best_node] as f64;
            let mut new_cands:    IntArray = Vec::new();
            let mut new_contribs: DblArray = Vec::new();
            for (k, &nd) in cand_nodes.iter().enumerate() {
                if k == best_idx { continue; }
                new_cands.push(nd);
                new_contribs.push(cand_contribs[k]);
            }
            for (k, &nd) in new_cands.iter().enumerate() {
                new_contribs[k] += um.get(nd, best_node) / weights[nd] as f64;
            }
            cand_nodes    = new_cands;
            cand_contribs = new_contribs;
        }
        right_nodes
    }

    fn run_greedy_left(um:          &UtilMatrix,
                       n_nodes:     usize,
                       mut left:    IntArray,
                       right_nodes: &IntArray,
                       budget:      i32,
                       beta:        f64,
                       weights:     &[i32],
                       mut ws:      Option<&mut WarmStartLeft>) -> IntArray {
        if left.is_empty() {
            if let Some(nd) = get_initial_node(right_nodes, um, weights, budget) {
                left.push(nd);
            }
        }
        let mut cur_w: f64 = match &ws {
            Some(w) if w.valid && w.current_total_weight >= 0.0 => w.current_total_weight,
            _ => left.iter().map(|&k| weights[k] as f64).sum(),
        };
        let mut cand_nodes:    IntArray = Vec::new();
        let mut cand_contribs: DblArray = Vec::new();
        let ws_valid          = ws.as_ref().map(|w| w.valid).unwrap_or(false);
        let ws_cands_nonempty = ws.as_ref().map(|w| !w.candidate_nodes.is_empty()).unwrap_or(false);
        let mut update_flag;
        if ws_valid && ws_cands_nonempty {
            let w = ws.as_ref().unwrap();
            let mut all_fit = true;
            let rem = budget as f64 - cur_w;
            for (idx, &nd) in w.candidate_nodes.iter().enumerate() {
                if weights[nd] as f64 <= rem {
                    cand_nodes.push(nd);
                    cand_contribs.push(w.candidate_contribs[idx]);
                } else { all_fit = false; }
            }
            update_flag = all_fit;
        } else {
            for &nd in right_nodes {
                if ia_contains(&left, nd) { continue; }
                if weights[nd] as f64 > budget as f64 - cur_w { continue; }
                cand_nodes.push(nd);
            }
            update_flag = true;
            for &nd in &cand_nodes {
                let mut contrib: f64 = left.iter()
                    .map(|&m| (1.0 + beta) * um.get(nd, m)).sum();
                contrib += um.linear[nd];
                if beta != 0.0 {
                    for &m in &cand_nodes { contrib -= beta * um.get(nd, m); }
                    for v in 0..n_nodes {
                        if !ia_contains(right_nodes, v) {
                            contrib -= beta * um.get(nd, v);
                        }
                    }
                }
                contrib /= weights[nd] as f64;
                cand_contribs.push(contrib);
            }
        }
        loop {
            if cand_nodes.is_empty() { break; }
            let best_idx = cand_contribs.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap();
            let best_node = cand_nodes[best_idx];
            left.push(best_node);
            cur_w += weights[best_node] as f64;
            let mut new_cands:    IntArray = Vec::new();
            let mut new_contribs: DblArray = Vec::new();
            let mut all_fit = true;
            let rem = budget as f64 - cur_w;
            for (k, &nd) in cand_nodes.iter().enumerate() {
                if k == best_idx { continue; }
                if weights[nd] as f64 > rem { all_fit = false; continue; }
                new_cands.push(nd);
                new_contribs.push(cand_contribs[k]);
            }
            for (k, &nd) in new_cands.iter().enumerate() {
                new_contribs[k] +=
                    (1.0 + 2.0 * beta) * um.get(nd, best_node) / weights[nd] as f64;
            }
            if let Some(ref mut ws_ref) = ws {
                if update_flag && all_fit {
                    ws_ref.left_nodes           = left.clone();
                    ws_ref.candidate_nodes      = new_cands.clone();
                    ws_ref.candidate_contribs   = new_contribs.clone();
                    ws_ref.current_total_weight = cur_w;
                    ws_ref.valid                = true;
                } else { update_flag = false; }
            }
            cand_nodes    = new_cands;
            cand_contribs = new_contribs;
        }
        left
    }

    fn run_greedy_right(um:              &UtilMatrix,
                        n_nodes:         usize,
                        mut right_nodes: IntArray,
                        budget:          i32,
                        beta:            f64,
                        weights:         &[i32],
                        ws:              Option<&mut WarmStartRight>) -> IntArray {
        if right_nodes.is_empty() { return right_nodes; }
        let mut cur_w: f64 = match &ws {
            Some(w) if w.valid => w.current_total_weight,
            _ => right_nodes.iter().map(|&k| weights[k] as f64).sum(),
        };
        let ws_valid          = ws.as_ref().map(|w| w.valid).unwrap_or(false);
        let ws_cands_nonempty = ws.as_ref().map(|w| !w.candidate_nodes.is_empty()).unwrap_or(false);
        let mut cand_nodes:    IntArray = Vec::new();
        let mut cand_contribs: DblArray = Vec::new();
        if ws_valid && ws_cands_nonempty {
            let w = ws.as_ref().unwrap();
            cand_nodes    = w.candidate_nodes.clone();
            cand_contribs = w.candidate_contribs.clone();
        } else {
            for &nd in &right_nodes {
                let mut contrib: f64 = right_nodes.iter()
                    .map(|&m| (-1.0 - beta) * um.get(nd, m)).sum();
                contrib -= um.linear[nd];
                if beta != 0.0 {
                    for v in 0..n_nodes {
                        if !ia_contains(&right_nodes, v) {
                            contrib += beta * um.get(nd, v);
                        }
                    }
                }
                contrib /= weights[nd] as f64;
                cand_nodes.push(nd);
                cand_contribs.push(contrib);
            }
        }
        while !cand_nodes.is_empty() && cur_w > budget as f64 {
            let best_idx = cand_contribs.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap();
            let best_node = cand_nodes[best_idx];
            right_nodes.retain(|&x| x != best_node);
            cur_w -= weights[best_node] as f64;
            let mut new_cands:    IntArray = Vec::new();
            let mut new_contribs: DblArray = Vec::new();
            for (k, &nd) in cand_nodes.iter().enumerate() {
                if k == best_idx { continue; }
                new_cands.push(nd);
                new_contribs.push(cand_contribs[k]);
            }
            for (k, &nd) in new_cands.iter().enumerate() {
                new_contribs[k] +=
                    (1.0 + 2.0 * beta) * um.get(nd, best_node) / weights[nd] as f64;
            }
            cand_nodes    = new_cands;
            cand_contribs = new_contribs;
        }
        right_nodes
    }

    struct GreedyResults {
        left_nodes:  Vec<IntArray>,
        right_nodes: Vec<IntArray>,
    }

    fn run_greedy(inst:       &P1Instance,
                  left_init:  IntArray,
                  right_init: IntArray) -> GreedyResults {
        let n_nodes = inst.n_items;
        let budget = inst.budgets[0];
        let um = build_utility_matrix(n_nodes, &inst.edges);
        let all_nodes: IntArray = (0..n_nodes).collect();

        let left = run_greedy_left_beta0(&um, left_init, &all_nodes,
                                         budget, &inst.weights, None);
        let after_right = run_greedy_right_beta0(&um, right_init,
                                                  budget, &inst.weights, None);
        let right = run_greedy_left_beta0(&um, after_right, &all_nodes,
                                          budget, &inst.weights, None);

        GreedyResults {
            left_nodes: vec![left],
            right_nodes: vec![right],
        }
    }

    pub fn run_bp_algorithm(inst: &P1Instance, n_lambda_values: usize) -> QKPResult {
        let bps = get_breakpoints(inst, n_lambda_values);
        let n_breakpoints = bps.sets.len();
        let budget = inst.budgets[0] as f64;
        let mut left_groups = 0usize;
        let mut right_groups = n_breakpoints;
        let mut cumsum = 0.0f64;

        if 0.0 >= budget {
            right_groups = 0;
        }
        for (i, (_, nodes)) in bps.sets.iter().enumerate() {
            for &nd in nodes {
                cumsum += inst.weights[nd] as f64;
            }
            if cumsum <= budget {
                left_groups = i + 1;
            }
            if right_groups == n_breakpoints && cumsum >= budget {
                right_groups = i + 1;
            }
        }

        let mut left_init = Vec::with_capacity(
            bps.sets[..left_groups].iter().map(|(_, nodes)| nodes.len()).sum()
        );
        for (_, nodes) in &bps.sets[..left_groups] {
            left_init.extend_from_slice(nodes);
        }

        let mut right_init = Vec::with_capacity(
            bps.sets[..right_groups].iter().map(|(_, nodes)| nodes.len()).sum()
        );
        for (_, nodes) in &bps.sets[..right_groups] {
            right_init.extend_from_slice(nodes);
        }

        let gr = run_greedy(inst, left_init, right_init);
        let mut results: Vec<BudgetResult> = Vec::with_capacity(inst.n_budgets);
        for bi in 0..inst.n_budgets {
            let ofv_left  = compute_ofv(&gr.left_nodes[bi],  inst.n_items, &inst.edges);
            let ofv_right = compute_ofv(&gr.right_nodes[bi], inst.n_items, &inst.edges);
            let (best_items, ofv) = if ofv_left >= ofv_right {
                (gr.left_nodes[bi].clone(), ofv_left)
            } else {
                (gr.right_nodes[bi].clone(), ofv_right)
            };
            results.push(BudgetResult {
                budget:         inst.budgets[bi],
                ofv,
                cpu:            0.0,
                selected_items: best_items,
            });
        }
        QKPResult { results }
    }

    pub struct P2Instance {
        pub n:                     usize,
        pub m:                     i64,
        pub capacity:              i32,
        pub w:                     Vec<i32>,
        pub q:                     Vec<i32>,
        pub interaction_potential: Vec<Ll>,
    }

    impl P2Instance {
        #[inline] pub fn q_get(&self, i: usize, j: usize) -> i32 { self.q[i * self.n + j] }
        #[inline] pub fn q_set(&mut self, i: usize, j: usize, v: i32) { self.q[i * self.n + j] = v; }
    }

    pub fn bridge_p1_to_p2(inst: &P1Instance, budget: i32) -> (P2Instance, Vec<Ll>) {
        let n = inst.n_items;
        let mut p2 = P2Instance {
            n,
            m:                     inst.n_edges as i64,
            capacity:              budget,
            w:                     inst.weights.clone(),
            q:                     vec![0i32; n * n],
            interaction_potential: vec![0; n],
        };
        for e in &inst.edges {
            let u  = e.i;
            let v  = e.j;
            let qv = e.value as i32;
            if u >= n || v >= n { continue; }
            if u == v {
                p2.q[u * n + u] = qv;
                p2.interaction_potential[u] += qv as Ll;
            } else {
                p2.q[u * n + v] = qv;
                p2.q[v * n + u] = qv;
                p2.interaction_potential[u] += qv as Ll;
                p2.interaction_potential[v] += qv as Ll;
            }
        }
        let total_interactions = p2.interaction_potential.clone();
        (p2, total_interactions)
    }

    pub struct P2State {
        pub ins:         P2Instance,
        pub sel:         Vec<u8>,
        pub best_sel:    Vec<u8>,
        pub contrib:     Vec<i32>,
        pub value:       Ll,
        pub weight:      i32,
        pub count:       i32,
        pub best_value:  Ll,
        pub best_weight: i32,
        pub best_count:  i32,
    }

    impl P2State {
        pub fn new(ins: P2Instance) -> Self {
            let n = ins.n;
            P2State {
                sel:         vec![0u8; n],
                best_sel:    vec![0u8; n],
                contrib:     vec![0i32; n],
                value:       0,
                weight:      0,
                count:       0,
                best_value:  NEG_INF,
                best_weight: 0,
                best_count:  0,
                ins,
            }
        }

        pub fn clear(&mut self) {
            self.sel    .iter_mut().for_each(|x| *x = 0);
            self.contrib.iter_mut().for_each(|x| *x = 0);
            self.value  = 0;
            self.weight = 0;
            self.count  = 0;
        }

        #[inline] pub fn slack(&self) -> i32 { self.ins.capacity - self.weight }

        pub fn add_item(&mut self, i: usize) {
            let n = self.ins.n;
            if i >= n || self.sel[i] != 0 { return; }
            let row = &self.ins.q[i * n..(i + 1) * n];
            self.value  += self.contrib[i] as Ll + row[i] as Ll;
            self.weight += self.ins.w[i];
            self.count  += 1;
            self.sel[i]  = 1;
            let contrib = self.contrib.as_mut_ptr();
            let values = row.as_ptr();
            unsafe {
                for j in 0..n {
                    *contrib.add(j) += *values.add(j);
                }
            }
        }

        pub fn remove_item(&mut self, i: usize) {
            let n = self.ins.n;
            if i >= n || self.sel[i] == 0 { return; }
            let row = &self.ins.q[i * n..(i + 1) * n];
            self.value  -= self.contrib[i] as Ll;
            self.weight -= self.ins.w[i];
            self.count  -= 1;
            self.sel[i]  = 0;
            let contrib = self.contrib.as_mut_ptr();
            let values = row.as_ptr();
            unsafe {
                for j in 0..n {
                    *contrib.add(j) -= *values.add(j);
                }
            }
        }

        pub fn replace_item(&mut self, rm: usize, add: usize) {
            self.remove_item(rm);
            self.add_item(add);
        }

        pub fn exchange_transaction(&mut self, removals: &[usize], additions: &[usize]) {
            let n = self.ins.n;
            let q = &self.ins.q;

            for (ri, &rm) in removals.iter().enumerate() {
                let mut loss = self.contrib[rm];
                for &prev in &removals[..ri] {
                    loss -= q[prev * n + rm];
                }
                self.value -= loss as Ll;
            }
            for (ai, &add) in additions.iter().enumerate() {
                let mut gain = self.contrib[add];
                for &rm in removals {
                    gain -= q[rm * n + add];
                }
                for &prev in &additions[..ai] {
                    gain += q[prev * n + add];
                }
                self.value += gain as Ll + q[add * n + add] as Ll;
            }

            for &rm in removals {
                self.weight -= self.ins.w[rm];
                self.count -= 1;
                self.sel[rm] = 0;
            }
            for &add in additions {
                self.weight += self.ins.w[add];
                self.count += 1;
                self.sel[add] = 1;
            }

            for j in 0..n {
                let c = &mut self.contrib[j];
                for &rm in removals {
                    *c -= q[rm * n + j];
                }
                for &add in additions {
                    *c += q[add * n + j];
                }
            }
        }

        pub fn save_best(&mut self) {
            if self.weight <= self.ins.capacity && self.value > self.best_value {
                self.best_value  = self.value;
                self.best_weight = self.weight;
                self.best_count  = self.count;
                self.best_sel.copy_from_slice(&self.sel);
            }
        }

        pub fn restore_best(&mut self) {
            self.clear();
            for i in 0..self.ins.n {
                if self.best_sel[i] != 0 {
                    self.add_item(i);
                }
            }
        }

        pub fn eval_selected(ins: &P2Instance, bits: &[u8]) -> Ll {
            let n = ins.n;
            let mut val: Ll = 0;
            for i in 0..n {
                if bits[i] == 0 { continue; }
                val += ins.q_get(i, i) as Ll;
                for j in (i+1)..n {
                    if bits[j] != 0 { val += ins.q_get(i, j) as Ll; }
                }
            }
            val
        }

        pub fn weight_selected(ins: &P2Instance, bits: &[u8]) -> i32 {
            (0..ins.n).filter(|&i| bits[i] != 0).map(|i| ins.w[i]).sum()
        }
    }



    pub fn bridge_load_solution(s: &mut P2State, br: &BudgetResult) {
        s.clear();
        for &id in &br.selected_items {
            if id < s.ins.n && s.sel[id] == 0
                && s.weight + s.ins.w[id] <= s.ins.capacity
            {
                s.add_item(id);
            }
        }
        s.save_best();
    }

    #[derive(Clone, Copy)]
    struct ItemScore { id: usize, score: Ll }

    fn build_best_unused(s: &P2State, max_k: usize) -> Vec<ItemScore> {
        let n = s.ins.n;
        if n < 300 || max_k < 4 {
            let mut arr: Vec<ItemScore> = (0..n)
                .filter(|&i| s.sel[i] == 0)
                .map(|i| {
                    let w = s.ins.w[i].max(1) as Ll;
                    let gain = s.contrib[i] as Ll + s.ins.q_get(i, i) as Ll;
                    ItemScore { id: i, score: gain * 1_000_000 / w }
                })
                .collect();
            arr.sort_unstable_by(|a, b| b.score.cmp(&a.score));
            arr.truncate(max_k);
            return arr;
        }

        let synergy_k = (max_k.min(56) / 4).max(1);
        let density_k = max_k.saturating_sub(synergy_k);
        let mut arr: Vec<ItemScore> = (0..n)
            .filter(|&i| s.sel[i] == 0)
            .map(|i| {
                let w = s.ins.w[i].max(1) as Ll;
                let gain = s.contrib[i] as Ll + s.ins.q_get(i, i) as Ll;
                ItemScore { id: i, score: gain * 1_000_000 / w }
            })
            .collect();
        arr.sort_unstable_by(|a, b| b.score.cmp(&a.score));
        arr.truncate(density_k);

        let mut present = vec![0u8; n];
        for is in &arr {
            present[is.id] = 1;
        }

        let mut synergy: Vec<ItemScore> = Vec::with_capacity(synergy_k);
        for i in 0..n {
            if s.sel[i] != 0 || present[i] != 0 { continue; }
            let score = s.ins.interaction_potential[i];
            if synergy.len() < synergy_k {
                synergy.push(ItemScore { id: i, score });
                continue;
            }

            let mut worst = 0usize;
            for j in 1..synergy.len() {
                if synergy[j].score < synergy[worst].score {
                    worst = j;
                }
            }
            if score > synergy[worst].score {
                synergy[worst] = ItemScore { id: i, score };
            }
        }

        synergy.sort_unstable_by(|a, b| b.score.cmp(&a.score));
        arr.extend(synergy);
        arr
    }

    fn build_worst_used(s: &P2State, max_k: usize) -> Vec<ItemScore> {
        let n = s.ins.n;
        if n < 300 || max_k < 4 {
            let mut arr: Vec<ItemScore> = (0..n)
                .filter(|&i| s.sel[i] != 0)
                .map(|i| {
                    let w = s.ins.w[i].max(1) as Ll;
                    ItemScore { id: i, score: s.contrib[i] as Ll * 1_000_000 / w }
                })
                .collect();
            arr.sort_unstable_by(|a, b| a.score.cmp(&b.score));
            arr.truncate(max_k);
            return arr;
        }

        let raw_k = (max_k.min(56) / 4).max(1);
        let density_k = max_k.saturating_sub(raw_k);
        let mut arr: Vec<ItemScore> = (0..n)
            .filter(|&i| s.sel[i] != 0)
            .map(|i| {
                let w = s.ins.w[i].max(1) as Ll;
                ItemScore { id: i, score: s.contrib[i] as Ll * 1_000_000 / w }
            })
            .collect();
        arr.sort_unstable_by(|a, b| a.score.cmp(&b.score));
        arr.truncate(density_k);

        let mut present = vec![0u8; n];
        for is in &arr {
            present[is.id] = 1;
        }

        let mut raw: Vec<ItemScore> = Vec::with_capacity(raw_k);
        for i in 0..n {
            if s.sel[i] == 0 || present[i] != 0 { continue; }
            let score = s.contrib[i] as Ll;
            if raw.len() < raw_k {
                raw.push(ItemScore { id: i, score });
                continue;
            }
            let mut worst = 0usize;
            for j in 1..raw.len() {
                if raw[j].score > raw[worst].score {
                    worst = j;
                }
            }
            if score < raw[worst].score {
                raw[worst] = ItemScore { id: i, score };
            }
        }

        raw.sort_unstable_by(|a, b| a.score.cmp(&b.score));
        arr.extend(raw);
        arr
    }

    struct Rng { state: u64 }

    impl Rng {
        fn new(seed: u64) -> Self { Rng { state: if seed == 0 { 1 } else { seed } } }

        #[inline]
        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 7;
            x ^= x >> 9;
            x ^= x << 8;
            self.state = x;
            x
        }

        #[inline]
        fn next_int(&mut self, bound: usize) -> usize {
            if bound == 0 { return 0; }
            (self.next_u64() % bound as u64) as usize
        }
    }

    fn apply_best_add1(s: &mut P2State) -> bool {
        let n   = s.ins.n;
        let rem = s.slack();
        if rem <= 0 { return false; }
        let mut best       = None;
        let mut best_delta = 0i32;
        for i in 0..n {
            if s.sel[i] != 0 || s.ins.w[i] > rem { continue; }
            let d = s.contrib[i] + s.ins.q_get(i, i);
            if d > best_delta { best_delta = d; best = Some(i); }
        }
        if let Some(b) = best {
            s.add_item(b);
            return true;
        }
        false
    }

    fn apply_best_add2(s: &mut P2State, window_k: usize) -> bool {
        let rem = s.slack();
        if rem <= 0 { return false; }
        let unused = build_best_unused(s, window_k);
        if unused.len() < 2 { return false; }
        let cu = unused.len();
        let mut best_delta: Ll = 0;
        let mut ba = None;
        let mut bb = None;
        for x in 0..cu {
            let a  = unused[x].id;
            let wa = s.ins.w[a];
            if wa >= rem { continue; }
            let ca = s.contrib[a] as Ll + s.ins.q_get(a, a) as Ll;
            for y in (x+1)..cu {
                let b = unused[y].id;
                if wa + s.ins.w[b] > rem { continue; }
                let d = ca + s.contrib[b] as Ll + s.ins.q_get(b, b) as Ll
                        + s.ins.q_get(a, b) as Ll;
                if d > best_delta { best_delta = d; ba = Some(a); bb = Some(b); }
            }
        }
        if let (Some(a), Some(b)) = (ba, bb) {
            s.add_item(a);
            s.add_item(b);
            return true;
        }
        false
    }

    fn apply_best_swap11(s: &mut P2State, window_k: usize) -> bool {
        let unused = build_best_unused(s, window_k);
        let used   = build_worst_used(s, window_k);
        let cu = unused.len();
        let cs = used.len();
        let mut best_delta: Ll = 0;
        let mut br = None;
        let mut ba = None;
        for x in 0..cs {
            let rm     = used[x].id;
            let budget = s.slack() + s.ins.w[rm];
            for y in 0..cu {
                let add = unused[y].id;
                if s.ins.w[add] > budget { continue; }
                let d = s.contrib[add] as Ll + s.ins.q_get(add, add) as Ll
                        - s.contrib[rm] as Ll - s.ins.q_get(add, rm) as Ll;
                if d > best_delta { best_delta = d; br = Some(rm); ba = Some(add); }
            }
        }
        if let (Some(r), Some(a)) = (br, ba) {
            s.replace_item(r, a);
            return true;
        }
        false
    }

    fn apply_best_swap12(s: &mut P2State, window_k: usize) -> bool {
        let unused = build_best_unused(s, window_k);
        let used   = build_worst_used(s, window_k);
        let cu = unused.len();
        let cs = used.len();
        let n = s.ins.n;
        let q = &s.ins.q;
        let mut add_bases = Vec::with_capacity(cu);
        for item in &unused {
            let id = item.id;
            add_bases.push(s.contrib[id] as Ll + q[id * n + id] as Ll);
        }
        let mut effective = vec![0; cu];
        let mut best_delta: Ll = 0;
        let mut br = None;
        let mut ba = None;
        let mut bb = None;
        for si in 0..cs {
            let rm     = used[si].id;
            let budget = s.slack() + s.ins.w[rm];
            let lost   = s.contrib[rm] as Ll;
            for x in 0..cu {
                effective[x] = add_bases[x] - q[unused[x].id * n + rm] as Ll;
            }
            for x in 0..cu {
                let a  = unused[x].id;
                let wa = s.ins.w[a];
                if wa >= budget { continue; }
                let ca_eff = effective[x];
                let row_a = &q[a * n..(a + 1) * n];
                for y in (x+1)..cu {
                    let b = unused[y].id;
                    if wa + s.ins.w[b] > budget { continue; }
                    let d = ca_eff + effective[y] + row_a[b] as Ll - lost;
                    if d > best_delta {
                        best_delta = d; br = Some(rm); ba = Some(a); bb = Some(b);
                    }
                }
            }
        }
        if let (Some(r), Some(a), Some(b)) = (br, ba, bb) {
            s.exchange_transaction(&[r], &[a, b]);
            return true;
        }
        false
    }

    fn apply_best_swap21(s: &mut P2State, window_k: usize) -> bool {
        let unused = build_best_unused(s, window_k);
        let used   = build_worst_used(s, window_k);
        let cu = unused.len();
        let cs = used.len();
        let n = s.ins.n;
        let q = &s.ins.q;
        let mut removal_pairs: Vec<(usize, usize, i32, Ll)> =
            Vec::with_capacity(cs.saturating_mul(cs.saturating_sub(1)) / 2);
        for x in 0..cs {
            let r1 = used[x].id;
            let row_r1 = &q[r1 * n..(r1 + 1) * n];
            for y in (x + 1)..cs {
                let r2 = used[y].id;
                removal_pairs.push((
                    r1,
                    r2,
                    s.ins.w[r1] + s.ins.w[r2],
                    s.contrib[r1] as Ll + s.contrib[r2] as Ll - row_r1[r2] as Ll,
                ));
            }
        }
        let mut best_delta: Ll = 0;
        let mut br1 = None;
        let mut br2 = None;
        let mut ba  = None;
        for ai in 0..cu {
            let add  = unused[ai].id;
            let row_add = &q[add * n..(add + 1) * n];
            let cadd = s.contrib[add] as Ll + row_add[add] as Ll;
            for &(r1, r2, removed_weight, lost) in &removal_pairs {
                if s.ins.w[add] > s.slack() + removed_weight { continue; }
                let gained = cadd - row_add[r1] as Ll
                                 - row_add[r2] as Ll;
                let d = gained - lost;
                if d > best_delta {
                    best_delta = d; br1 = Some(r1); br2 = Some(r2); ba = Some(add);
                }
            }
        }
        if let (Some(r1), Some(r2), Some(a)) = (br1, br2, ba) {
            s.exchange_transaction(&[r1, r2], &[a]);
            return true;
        }
        false
    }

    fn apply_best_swap22(s: &mut P2State, window_k: usize) -> bool {
        let k      = window_k.min(40);
        let unused = build_best_unused(s, k);
        let used   = build_worst_used(s, k);
        let cu = unused.len();
        let cs = used.len();
        let n = s.ins.n;
        let q = &s.ins.q;

        let mut unused_to_used = vec![0i32; cu * cs];
        for a in 0..cu {
            let row = &q[unused[a].id * n..(unused[a].id + 1) * n];
            let table_row = &mut unused_to_used[a * cs..(a + 1) * cs];
            for b in 0..cs {
                table_row[b] = row[used[b].id];
            }
        }

        let mut add_pairs: Vec<(usize, usize, i32, i32, Ll)> =
            Vec::with_capacity(cu.saturating_mul(cu.saturating_sub(1)) / 2);
        for a in 0..cu {
            let p = unused[a].id;
            let wp = s.ins.w[p];
            let row_p = &q[p * n..(p + 1) * n];
            let p_base = s.contrib[p] as Ll + row_p[p] as Ll;
            for b in (a+1)..cu {
                let q2 = unused[b].id;
                let pair_w = wp + s.ins.w[q2];
                let row_q2 = &q[q2 * n..(q2 + 1) * n];
                let base = p_base + s.contrib[q2] as Ll + row_q2[q2] as Ll
                           + row_p[q2] as Ll;
                add_pairs.push((a, b, wp, pair_w, base));
            }
        }

        let mut best_delta: Ll = 0;
        let mut br1 = None; let mut br2 = None;
        let mut ba1 = None; let mut ba2 = None;
        for x in 0..cs {
            let r1 = used[x].id;
            let row_r1 = &q[r1 * n..(r1 + 1) * n];
            for y in (x+1)..cs {
                let r2     = used[y].id;
                let budget = s.slack() + s.ins.w[r1] + s.ins.w[r2];
                let lost   = s.contrib[r1] as Ll + s.contrib[r2] as Ll
                             - row_r1[r2] as Ll;
                for &(p, q2, wp, pair_w, base) in &add_pairs {
                    if wp >= budget || pair_w > budget { continue; }
                    let d = base
                            - unused_to_used[p * cs + x] as Ll
                            - unused_to_used[p * cs + y] as Ll
                            - unused_to_used[q2 * cs + x] as Ll
                            - unused_to_used[q2 * cs + y] as Ll
                            - lost;
                    if d > best_delta {
                        best_delta = d;
                        br1 = Some(r1); br2 = Some(r2);
                        ba1 = Some(unused[p].id); ba2 = Some(unused[q2].id);
                    }
                }
            }
        }
        if let (Some(r1), Some(r2), Some(a1), Some(a2)) = (br1, br2, ba1, ba2) {
            s.exchange_transaction(&[r1, r2], &[a1, a2]);
            return true;
        }
        false
    }

    fn apply_best_refill_after_remove(s: &mut P2State, window_k: usize) -> bool {        
        let k = window_k.min(40);
        if k == 0 || s.count <= 0 { return false; }

        let unused = build_best_unused(s, k);
        let used   = build_worst_used(s, k);
        if unused.is_empty() || used.is_empty() { return false; }

        let n = s.ins.n;
        let q = &s.ins.q;
        let mut best_delta: Ll = 0;
        let mut best_rm = None;
        let mut best_pack: Vec<usize> = Vec::new();

        for us in &used {
            let rm = us.id;
            let budget = s.slack() + s.ins.w[rm];
            if budget <= 0 { continue; }

            let mut chosen: Vec<usize> = Vec::with_capacity(3);
            let mut chosen_w = 0i32;
            let mut running_delta = -(s.contrib[rm] as Ll);
            let mut best_local_delta: Ll = 0;
            let mut best_local_len = 0usize;

            for _ in 0..3 {
                let mut best_add = None;
                let mut best_marg = NEG_INF;

                for is in &unused {
                    let add = is.id;
                    if chosen.iter().any(|&x| x == add) { continue; }
                    let wt = s.ins.w[add];
                    if chosen_w + wt > budget { continue; }

                    let row_add = &q[add * n..(add + 1) * n];
                    let mut marginal = s.contrib[add] as Ll
                        + row_add[add] as Ll
                        - row_add[rm] as Ll;
                    for &other in &chosen {
                        marginal += row_add[other] as Ll;
                    }

                    if marginal > best_marg {
                        best_marg = marginal;
                        best_add = Some(add);
                    }
                }

                let add = match best_add {
                    Some(id) => id,
                    None => break,
                };

                chosen_w += s.ins.w[add];
                chosen.push(add);
                running_delta += best_marg;

                if running_delta > best_local_delta {
                    best_local_delta = running_delta;
                    best_local_len = chosen.len();
                }
            }

            if best_local_len > 0 && best_local_delta > best_delta {
                best_delta = best_local_delta;
                best_rm = Some(rm);
                best_pack.clear();
                best_pack.extend_from_slice(&chosen[..best_local_len]);
            }
        }

        if let Some(rm) = best_rm {
            let added_weight: i32 = best_pack.iter().map(|&i| s.ins.w[i]).sum();
            if !best_pack.is_empty()
                && s.weight - s.ins.w[rm] + added_weight <= s.ins.capacity
            {
                s.remove_item(rm);
                for add in best_pack {
                    s.add_item(add);
                }
                return true;
            }
        }

        false
    }

    fn apply_best_swap13(s: &mut P2State, window_k: usize) -> bool {
        let k = window_k.min(28);
        if k < 3 || s.count <= 0 { return false; }

        let unused = build_best_unused(s, k);
        let used = build_worst_used(s, k.min(32));
        if unused.len() < 3 || used.is_empty() { return false; }

        let n = s.ins.n;
        let q = &s.ins.q;
        let mut best_delta = 0i64;
        let mut best_rm = None;
        let mut best_a = None;
        let mut best_b = None;
        let mut best_c = None;

        for rm_score in &used {
            let rm = rm_score.id;
            let budget = s.slack() + s.ins.w[rm];
            let lost = s.contrib[rm] as Ll;

            for ai in 0..unused.len() {
                let a = unused[ai].id;
                let wa = s.ins.w[a];
                if wa > budget { continue; }
                let row_a = &q[a * n..(a + 1) * n];
                let da = s.contrib[a] as Ll + row_a[a] as Ll - row_a[rm] as Ll;

                for bi in (ai + 1)..unused.len() {
                    let b = unused[bi].id;
                    let wab = wa + s.ins.w[b];
                    if wab > budget { continue; }
                    let row_b = &q[b * n..(b + 1) * n];
                    let db = s.contrib[b] as Ll + row_b[b] as Ll - row_b[rm] as Ll;

                    for ci in (bi + 1)..unused.len() {
                        let c = unused[ci].id;
                        if wab + s.ins.w[c] > budget { continue; }
                        let row_c = &q[c * n..(c + 1) * n];
                        let dc = s.contrib[c] as Ll + row_c[c] as Ll - row_c[rm] as Ll;
                        let delta = da + db + dc
                            + row_a[b] as Ll
                            + row_a[c] as Ll
                            + row_b[c] as Ll
                            - lost;

                        if delta > best_delta {
                            best_delta = delta;
                            best_rm = Some(rm);
                            best_a = Some(a);
                            best_b = Some(b);
                            best_c = Some(c);
                        }
                    }
                }
            }
        }

        if let (Some(rm), Some(a), Some(b), Some(c)) =
            (best_rm, best_a, best_b, best_c)
        {
            s.exchange_transaction(&[rm], &[a, b, c]);
            return true;
        }
        false
    }

    fn apply_best_swap23(s: &mut P2State, window_k: usize) -> bool {
        let k = window_k.min(18);
        if k < 3 || s.count <= 1 { return false; }

        let unused = build_best_unused(s, k);
        let used = build_worst_used(s, k);
        if unused.len() < 3 || used.len() < 2 { return false; }

        let n = s.ins.n;
        let q = &s.ins.q;
        let mut best_delta = 0i64;
        let mut best_r1 = None;
        let mut best_r2 = None;
        let mut best_a = None;
        let mut best_b = None;
        let mut best_c = None;

        for ri in 0..used.len() {
            let r1 = used[ri].id;
            let row_r1 = &q[r1 * n..(r1 + 1) * n];
            for rj in (ri + 1)..used.len() {
                let r2 = used[rj].id;
                let budget = s.slack() + s.ins.w[r1] + s.ins.w[r2];
                let lost = s.contrib[r1] as Ll + s.contrib[r2] as Ll
                    - row_r1[r2] as Ll;

                for ai in 0..unused.len() {
                    let a = unused[ai].id;
                    let wa = s.ins.w[a];
                    if wa > budget { continue; }
                    let row_a = &q[a * n..(a + 1) * n];
                    let da = s.contrib[a] as Ll + row_a[a] as Ll
                        - row_a[r1] as Ll - row_a[r2] as Ll;

                    for bi in (ai + 1)..unused.len() {
                        let b = unused[bi].id;
                        let wab = wa + s.ins.w[b];
                        if wab > budget { continue; }
                        let row_b = &q[b * n..(b + 1) * n];
                        let db = s.contrib[b] as Ll + row_b[b] as Ll
                            - row_b[r1] as Ll - row_b[r2] as Ll;

                        for ci in (bi + 1)..unused.len() {
                            let c = unused[ci].id;
                            if wab + s.ins.w[c] > budget { continue; }
                            let row_c = &q[c * n..(c + 1) * n];
                            let dc = s.contrib[c] as Ll + row_c[c] as Ll
                                - row_c[r1] as Ll - row_c[r2] as Ll;
                            let delta = da + db + dc
                                + row_a[b] as Ll
                                + row_a[c] as Ll
                                + row_b[c] as Ll
                                - lost;
                            if delta > best_delta {
                                best_delta = delta;
                                best_r1 = Some(r1);
                                best_r2 = Some(r2);
                                best_a = Some(a);
                                best_b = Some(b);
                                best_c = Some(c);
                            }
                        }
                    }
                }
            }
        }

        if let (Some(r1), Some(r2), Some(a), Some(b), Some(c)) =
            (best_r1, best_r2, best_a, best_b, best_c)
        {
            s.exchange_transaction(&[r1, r2], &[a, b, c]);
            return true;
        }
        false
    }

    fn apply_best_swap31(s: &mut P2State, window_k: usize) -> bool {
        let k = window_k.min(24);
        if k < 3 || s.count < 3 { return false; }

        let unused = build_best_unused(s, k);
        let used = build_worst_used(s, k);
        if unused.is_empty() || used.len() < 3 { return false; }

        let n = s.ins.n;
        let q = &s.ins.q;
        let mut best_delta = 0i64;
        let mut best_r1 = None;
        let mut best_r2 = None;
        let mut best_r3 = None;
        let mut best_add = None;

        for ai in 0..unused.len() {
            let add = unused[ai].id;
            let row_add = &q[add * n..(add + 1) * n];
            let base_gain = s.contrib[add] as Ll + row_add[add] as Ll;

            for ri in 0..used.len() {
                let r1 = used[ri].id;
                let row_r1 = &q[r1 * n..(r1 + 1) * n];
                for rj in (ri + 1)..used.len() {
                    let r2 = used[rj].id;
                    let row_r2 = &q[r2 * n..(r2 + 1) * n];
                    for rk in (rj + 1)..used.len() {
                        let r3 = used[rk].id;
                        let new_weight = s.weight - s.ins.w[r1] - s.ins.w[r2]
                            - s.ins.w[r3] + s.ins.w[add];
                        if new_weight > s.ins.capacity { continue; }

                        let lost = s.contrib[r1] as Ll + s.contrib[r2] as Ll
                            + s.contrib[r3] as Ll
                            - row_r1[r2] as Ll
                            - row_r1[r3] as Ll
                            - row_r2[r3] as Ll;
                        let gained = base_gain
                            - row_add[r1] as Ll
                            - row_add[r2] as Ll
                            - row_add[r3] as Ll;
                        let delta = gained - lost;

                        if delta > best_delta {
                            best_delta = delta;
                            best_r1 = Some(r1);
                            best_r2 = Some(r2);
                            best_r3 = Some(r3);
                            best_add = Some(add);
                        }
                    }
                }
            }
        }

        if let (Some(r1), Some(r2), Some(r3), Some(add)) =
            (best_r1, best_r2, best_r3, best_add)
        {
            s.exchange_transaction(&[r1, r2, r3], &[add]);
            return true;
        }
        false
    }

    struct VndCertificateEntry {
        window_k: usize,
        heavy:    bool,
        value:    Ll,
        weight:   i32,
        count:    i32,
        sel:      Vec<u8>,
        contrib:  Vec<i32>,
    }

    impl VndCertificateEntry {
        fn matches(&self, s: &P2State, window_k: usize, heavy: bool) -> bool {
            self.window_k == window_k
                && self.heavy == heavy
                && self.value == s.value
                && self.weight == s.weight
                && self.count == s.count
                && self.sel.as_slice() == s.sel.as_slice()
        }
    }

    struct VndCertificate {
        entries: Vec<VndCertificateEntry>,
    }

    impl VndCertificate {
        fn new() -> Self {
            VndCertificate { entries: Vec::new() }
        }

        fn matches(&self, s: &P2State, window_k: usize, heavy: bool) -> bool {
            self.entries.iter().any(|e| e.matches(s, window_k, heavy))
        }

        fn store(&mut self, s: &P2State, window_k: usize, heavy: bool) {
            if self.matches(s, window_k, heavy) { return; }
            let cap = 16usize;
            if self.entries.len() < cap {
                self.entries.push(VndCertificateEntry {
                    window_k,
                    heavy,
                    value:   s.value,
                    weight:  s.weight,
                    count:   s.count,
                    sel:     s.sel.clone(),
                    contrib: s.contrib.clone(),
                });
                return;
            }

            let mut replace_idx = 0usize;
            let mut replace_val = self.entries[0].value;
            for i in 1..self.entries.len() {
                if self.entries[i].value < replace_val {
                    replace_val = self.entries[i].value;
                    replace_idx = i;
                }
            }
            if s.value <= replace_val { return; }

            let e = &mut self.entries[replace_idx];
            e.window_k = window_k;
            e.heavy    = heavy;
            e.value    = s.value;
            e.weight   = s.weight;
            e.count    = s.count;
            e.sel.clear();
            e.sel.extend_from_slice(&s.sel);
            e.contrib.clear();
            e.contrib.extend_from_slice(&s.contrib);
        }
    }

    fn local_search_vnd(s: &mut P2State, window_k: usize, heavy: bool,
                        cert: &mut VndCertificate) {
        if cert.matches(s, window_k, heavy) { return; }
        loop {
            let tight = s.count >= 4 && s.slack() <= s.ins.capacity / 10;
            if tight {
                if apply_best_swap11(s, window_k)     { s.save_best(); continue; }
                if apply_best_swap12(s, window_k / 2) { s.save_best(); continue; }
                if apply_best_refill_after_remove(s, window_k) {
                    s.save_best();
                    continue;
                }
                if apply_best_swap21(s, window_k / 2) { s.save_best(); continue; }
                if apply_best_add2(s, window_k)       { s.save_best(); continue; }
                if apply_best_add1(s)                 { s.save_best(); continue; }
            } else {
                if apply_best_add1(s)                 { s.save_best(); continue; }
                if apply_best_swap11(s, window_k)     { s.save_best(); continue; }
                if apply_best_add2(s, window_k)       { s.save_best(); continue; }
                if apply_best_swap12(s, window_k / 2) { s.save_best(); continue; }
                if apply_best_swap21(s, window_k / 2) { s.save_best(); continue; }
                if apply_best_refill_after_remove(s, window_k) {
                    s.save_best();
                    continue;
                }
            }
            if heavy {
                if apply_best_swap13(s, window_k) { s.save_best(); continue; }
                if s.count >= 6 && apply_best_swap23(s, window_k) {
                    s.save_best();
                    continue;
                }
                if apply_best_swap31(s, window_k) { s.save_best(); continue; }
                if apply_best_swap22(s, window_k) { s.save_best(); continue; }
            }
            break;
        }
        cert.store(s, window_k, heavy);
    }

    struct DpWorkspace {
        ord:    Vec<ItemScore>,
        target: Vec<u8>,
        dp:     Vec<Ll>,
        choose: Vec<u64>,
    }

    impl DpWorkspace {
        fn new() -> Self {
            DpWorkspace {
                ord:    Vec::new(),
                target: Vec::new(),
                dp:     Vec::new(),
                choose: Vec::new(),
            }
        }
    }

    fn dp_refinement(s: &mut P2State, core_half: usize, ws: &mut DpWorkspace) {
        let n   = s.ins.n;
        let cap = s.ins.capacity;

        ws.ord.clear();
        for i in 0..n {
            let w = s.ins.w[i].max(1) as Ll;
            let gain = s.contrib[i] as Ll + s.ins.q_get(i, i) as Ll;
            ws.ord.push(ItemScore { id: i, score: gain * 1_000_000 / w });
        }
        ws.ord.sort_unstable_by(|a, b| b.score.cmp(&a.score));

        let mut idx_last      = 0usize;
        let mut idx_first_rej = n;
        let mut rem           = cap;
        for (idx, is) in ws.ord.iter().enumerate() {
            let wt = s.ins.w[is.id];
            if wt <= rem { rem -= wt; idx_last = idx; }
            else if idx_first_rej == n { idx_first_rej = idx; }
        }

        let left  = if idx_first_rej > core_half + 1 { idx_first_rej - core_half - 1 } else { 0 };
        let right = (idx_last + core_half + 1).min(n);
        if left >= right { return; }

        ws.target.resize(n, 0);
        for x in ws.target.iter_mut() { *x = 0; }

        let mut locked_weight = 0i32;
        for i in 0..left {
            let item = ws.ord[i].id;
            if locked_weight + s.ins.w[item] <= cap {
                ws.target[item] = 1;
                locked_weight += s.ins.w[item];
            }
        }

        let rem_cap = cap - locked_weight;
        if rem_cap > 0 {
            let k = right - left;
            let total_core_w: i32 = (left..right).map(|t| s.ins.w[ws.ord[t].id]).sum();
            let max_w = (rem_cap.min(total_core_w)) as usize;

            if max_w > 0 && k > 0 && max_w <= 2_000_000 {
                let neg_inf_dp: Ll = NEG_INF / 4;
                let stride = max_w + 1;
                if ws.dp.len() < stride {
                    ws.dp.resize(stride, neg_inf_dp);
                }
                for v in ws.dp[..stride].iter_mut() { *v = neg_inf_dp; }

                let words_per_row = (stride + 63) >> 6;
                let choose_len = k * words_per_row;
                if ws.choose.len() < choose_len {
                    ws.choose.resize(choose_len, 0);
                }
                for v in ws.choose[..choose_len].iter_mut() { *v = 0; }

                ws.dp[0] = 0;
                let mut w_hi = 0usize;

                for t in 0..k {
                    let item = ws.ord[left + t].id;
                    let wt   = s.ins.w[item] as usize;
                    let val  = s.contrib[item] as Ll + s.ins.q_get(item, item) as Ll;
                    if wt > max_w { continue; }
                    let new_hi = (w_hi + wt).min(max_w);
                    for w in (wt..=new_hi).rev() {
                        let cand = ws.dp[w - wt] + val;
                        if cand > ws.dp[w] {
                            ws.dp[w] = cand;
                            ws.choose[t * words_per_row + (w >> 6)] |= 1u64 << (w & 63);
                        }
                    }
                    w_hi = new_hi;
                }

                let best_w = (0..=max_w).max_by_key(|&w| ws.dp[w]).unwrap_or(0);
                let mut cur_w = best_w;
                for t in (0..k).rev() {
                    let item = ws.ord[left + t].id;
                    let wt   = s.ins.w[item] as usize;
                    if wt <= cur_w
                        && (ws.choose[t * words_per_row + (cur_w >> 6)] & (1u64 << (cur_w & 63))) != 0
                    {
                        ws.target[item] = 1;
                        cur_w -= wt;
                    }
                }
            }
        }

        for i in 0..n {
            if s.sel[i] != 0 && ws.target[i] == 0 { s.remove_item(i); }
        }
        for i in 0..n {
            if s.sel[i] == 0 && ws.target[i] != 0
                && s.weight + s.ins.w[i] <= s.ins.capacity
            {
                s.add_item(i);
            }
        }
    }

    fn perturb(s:                  &mut P2State,
               rng:                &mut Rng,
               strength:           usize,
               strategy:           usize,
               total_interactions: &[Ll]) {
        let n = s.ins.n;
        let mut cand: Vec<ItemScore> = (0..n)
            .filter(|&i| s.sel[i] != 0)
            .map(|i| {
                let score = match strategy {
                    0 => s.contrib[i] as Ll,
                    1 => -(s.ins.w[i] as Ll),
                    2 => { let w = s.ins.w[i].max(1) as Ll;
                           s.contrib[i] as Ll * 1_000_000 / w },
                    3 => s.contrib[i] as Ll * 100 - total_interactions[i],
                    4 => total_interactions[i] - 2 * s.contrib[i] as Ll,
                    _ => (rng.next_u64() & 0x7fff_ffff) as Ll,
                };
                ItemScore { id: i, score }
            })
            .collect();

        let cnt = cand.len();
        cand.sort_unstable_by(|a, b| a.score.cmp(&b.score));

        let remove_n = strength.min(cnt);
        for i in 0..remove_n {
            s.remove_item(cand[i].id);
        }
    }

    fn greedy_reconstruct(s:                  &mut P2State,
                          strategy:           usize,
                          total_interactions: &[Ll]) {
        if strategy == 5 && s.ins.n <= 2500 {
            greedy_reconstruct_sign_aware(s, total_interactions);
            return;
        }

        let n = s.ins.n;
        let mut cand: Vec<ItemScore> = (0..n)
            .filter(|&i| s.sel[i] == 0)
            .map(|i| {
                let w = s.ins.w[i].max(1) as Ll;
                let score = match strategy {
                    0 => s.contrib[i] as Ll + s.ins.q_get(i, i) as Ll,
                    1 => (s.contrib[i] as Ll + s.ins.q_get(i, i) as Ll) * 1_000_000 / w,
                    2 => total_interactions[i] + s.contrib[i] as Ll + s.ins.q_get(i, i) as Ll,
                    3 => (s.contrib[i] as Ll + s.ins.q_get(i, i) as Ll) * 1_000_000 / w
                         + total_interactions[i] / 10,
                    _ => s.contrib[i] as Ll + s.ins.q_get(i, i) as Ll
                         + total_interactions[i] / 20 - s.ins.w[i] as Ll,
                };
                ItemScore { id: i, score }
            })
            .collect();

        cand.sort_unstable_by(|a, b| b.score.cmp(&a.score));

        for is in &cand {
            if s.weight + s.ins.w[is.id] <= s.ins.capacity {
                s.add_item(is.id);
            }
        }
    }

    fn greedy_reconstruct_sign_aware(s: &mut P2State, total_interactions: &[Ll]) {
        let n = s.ins.n;
        let mut cand: Vec<ItemScore> = (0..n)
            .filter(|&i| s.sel[i] == 0)
            .map(|i| {
                let w = s.ins.w[i].max(1) as Ll;
                let marginal = s.contrib[i] as Ll + s.ins.q_get(i, i) as Ll;
                ItemScore {
                    id: i,
                    score: marginal * 1_000_000 / w + total_interactions[i] / 10,
                }
            })
            .collect();

        cand.sort_unstable_by(|a, b| b.score.cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
        let likely: Vec<usize> = cand.iter().take(48).map(|is| is.id).collect();

        for is in cand.iter_mut().take(likely.len()) {
            let item = is.id;
            let marginal = s.contrib[item] as Ll + s.ins.q_get(item, item) as Ll;
            let mut negative_exposure = 0i64;
            for &other in &likely {
                if item == other { continue; }
                let interaction = s.ins.q_get(item, other) as Ll;
                if interaction < 0 {
                    negative_exposure -= interaction;
                }
            }

            let present_positive = marginal.max(0);
            let penalty = if negative_exposure > present_positive {
                negative_exposure.saturating_mul(2)
            } else {
                negative_exposure
            };
            is.score = marginal.saturating_mul(3)
                .saturating_add(total_interactions[item] / 12)
                .saturating_sub(penalty);
        }

        cand.sort_unstable_by(|a, b| b.score.cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
        for is in &cand {
            if s.weight + s.ins.w[is.id] <= s.ins.capacity {
                s.add_item(is.id);
            }
        }
    }

    fn greedy_reconstruct_cluster(s:                  &mut P2State,
                                  total_interactions: &[Ll]) -> bool {
        if s.count < 4 { return false; }

        let n = s.ins.n;
        let rem = s.slack();
        let mut anchors: Vec<ItemScore> = Vec::new();

        for i in 0..n {
            if s.sel[i] != 0 || s.ins.w[i] > rem { continue; }
            let marginal = s.contrib[i] as Ll + s.ins.q_get(i, i) as Ll;
            anchors.push(ItemScore {
                id: i,
                score: total_interactions[i] + marginal * 2,
            });
        }
        anchors.sort_unstable_by(|a, b| b.score.cmp(&a.score));
        anchors.truncate(36);

        let mut best_score = 0i64;
        let mut best_a = None;
        let mut best_b = None;

        for a in 0..anchors.len() {
            let ia = anchors[a].id;
            if anchors[a].score > best_score {
                best_score = anchors[a].score;
                best_a = Some(ia);
                best_b = None;
            }
            for b in (a + 1)..anchors.len() {
                let ib = anchors[b].id;
                if s.ins.w[ia] + s.ins.w[ib] > rem { continue; }
                let score = anchors[a].score + anchors[b].score
                    + 3 * s.ins.q_get(ia, ib) as Ll;
                if score > best_score {
                    best_score = score;
                    best_a = Some(ia);
                    best_b = Some(ib);
                }
            }
        }

        if let Some(a) = best_a {
            s.add_item(a);
            if let Some(b) = best_b {
                s.add_item(b);
            }
            greedy_reconstruct(s, 2, total_interactions);
            return true;
        }
        false
    }

    fn solve_hybrid(s:          &mut P2State,
                    rng:        &mut Rng,
                    ils_rounds: usize,
                    window_k:   usize,
                    core_half:  usize,
                    total_interactions: &[Ll]) {
        let mut dp_ws = DpWorkspace::new();
        let mut vnd_cert = VndCertificate::new();

        dp_refinement(s, core_half, &mut dp_ws);
        local_search_vnd(s, window_k, true, &mut vnd_cert);
        s.save_best();
        s.restore_best();

        let n = s.ins.n;
        if n <= 2000 && s.count >= 6 {
            let seed_strength = ((s.count as usize) / 8).clamp(2, 5);
            perturb(s, rng, seed_strength, 2, total_interactions);
            greedy_reconstruct(s, 0, total_interactions);
            local_search_vnd(s, window_k, true, &mut vnd_cert);
            s.save_best();
            s.restore_best();
        }

        if n <= 1200 && s.count >= 10 {
            let seed_strength = ((s.count as usize) / 10).clamp(2, 4);
            perturb(s, rng, seed_strength, 3, total_interactions);
            if !greedy_reconstruct_cluster(s, total_interactions) {
                greedy_reconstruct(s, 2, total_interactions);
            }
            local_search_vnd(s, window_k, true, &mut vnd_cert);
            s.save_best();
            s.restore_best();
        }

        let selected = (s.count as usize).max(1);
        let mut target_rounds = if n <= 1200 {
            ils_rounds.max(60).min(140)
        } else if n <= 2500 {
            ils_rounds.max(45).min(120)
        } else if n <= 4000 {
            ils_rounds.max(35).min(80)
        } else {
            ils_rounds.max(30).min(45)
        };

        if window_k >= 240 && n > 2500 {
            target_rounds = target_rounds.min(60);
        }
        if selected <= 8 {
            target_rounds = target_rounds.min(50);
        }

        let stall_limit = (target_rounds / 2).max(24);
        let mut stall = 0usize;

        for round in 0..target_rounds {
            let old_best = s.best_value;
            s.restore_best();

            let strength = {
                let base = 2 + round / 6 + stall / 2;
                let cap  = ((s.count as usize) / 2).max(1);
                base.min(cap).max(1)
            };

            perturb(s, rng, strength, round % 6, total_interactions);
            if stall >= 4 && s.count >= 4 && (round + stall) % 3 == 0 {
                if !greedy_reconstruct_cluster(s, total_interactions) {
                    greedy_reconstruct(s, round % 6, total_interactions);
                }
            } else {
                greedy_reconstruct(s, round % 6, total_interactions);
            }

            dp_refinement(s, core_half, &mut dp_ws);
            local_search_vnd(s, window_k, true, &mut vnd_cert);
            s.save_best();

            if s.best_value > old_best {
                stall = 0;
                s.restore_best();
                if round % 4 == 0 {
                    dp_refinement(s, core_half, &mut dp_ws);
                    local_search_vnd(s, window_k, false, &mut vnd_cert);
                    s.save_best();
                }
            } else {
                stall += 1;
                if stall >= stall_limit && round + 1 >= 30 {
                    break;
                }
            }
        }

        s.restore_best();

        for _ in 0..5 {
            let before = s.value;
            dp_refinement(s, core_half * 2, &mut dp_ws);
            local_search_vnd(s, window_k, true, &mut vnd_cert);
            s.save_best();
            if s.value <= before { break; }
        }

        s.restore_best();
    }

    pub struct Solver;

    impl Solver {
        pub fn solve(
            challenge: &Challenge,
            _save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
            hyperparameters: &Option<Map<String, Value>>,
        ) -> Result<Option<Solution>> {
            let n = challenge.num_items;
            let hp = Hparams::from_map(hyperparameters, n);

            let raw_seed = u64::from_le_bytes(
                challenge.seed[..8].try_into().unwrap_or([0u8; 8])
            );
            let seed = raw_seed.wrapping_add(0x9E3779B97F4A7C15);
            let mut rng = Rng::new(seed);

            let inst = challenge_to_p1(challenge);

            let qr = run_bp_algorithm(&inst, hp.n_lambda_values);

            let budget = inst.budgets[0];

            let (p2, total_interactions) = bridge_p1_to_p2(&inst, budget);
            let mut s = P2State::new(p2);
            bridge_load_solution(&mut s, &qr.results[0]);

            solve_hybrid(&mut s, &mut rng, hp.ils_rounds, hp.window_k, hp.core_half_dp, &total_interactions);

            let chk_val = P2State::eval_selected(&s.ins, &s.best_sel);
            let chk_wt  = P2State::weight_selected(&s.ins, &s.best_sel);
            if chk_val != s.best_value  { s.best_value  = chk_val; }
            if chk_wt  != s.best_weight { s.best_weight = chk_wt;  }
            if chk_wt > budget {
                return Ok(None);
            }

            let items: Vec<usize> = (0..s.ins.n)
                .filter(|&i| s.best_sel[i] != 0)
                .collect();

            Ok(Some(Solution { items }))
        }
    }

    pub fn solve_challenge(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        hyperparameters: &Option<Map<String, Value>>,
    ) -> Result<()> {
        if let Some(solution) = Solver::solve(challenge, Some(save_solution), hyperparameters)? {
            let _ = save_solution(&solution);
        }
        Ok(())
    }

} 

#[inline(always)]
pub fn solve(
    challenge: &Challenge,
    save: &dyn Fn(&Solution) -> Result<()>,
    hp: &Option<Map<String, Value>>,
) -> Result<()> {
    inner::solve_challenge(challenge, save, hp)
}