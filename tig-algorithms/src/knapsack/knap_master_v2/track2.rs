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
        fn for_size(n: usize, budget: u32) -> Self {
            if n <= 1200 {
                if budget <= 10 {
                    Self {
                        n_lambda_values: 1600,
                        ils_rounds:      30,
                        window_k:        250,
                        core_half_dp:    50,
                    }
                } else {
                    Self {
                        n_lambda_values: 1600,
                        ils_rounds:      10,
                        window_k:        250,
                        core_half_dp:    50,
                    }
                }
            } else {
                Self {
                    n_lambda_values: 1600,
                    ils_rounds:      40,
                    window_k:        250,
                    core_half_dp:    50,
                }
            }
        }

        fn from_map(h: &Option<Map<String, Value>>, n: usize, budget: u32) -> Self {
            let mut p = Self::for_size(n, budget);
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

    #[derive(Debug)]
    pub struct BudgetResult {
        pub budget:         i32,
        pub ofv:            f64,
        pub cpu:            f64,
        pub selected_items: Vec<usize>,
    }

    #[derive(Debug)]
    pub struct QKPResult { pub results: Vec<BudgetResult> }

    pub fn challenge_to_p1(challenge: &Challenge) -> P1Instance {
        let n = challenge.num_items;
        let mut edges: Vec<Edge> = Vec::new();

        let dense_shape = challenge.interaction_values.get(..n)
            .map_or(false, |rows| rows.iter().all(|row| row.len() >= n));
        if dense_shape {
            let max_vec_len = (isize::MAX as usize) / std::mem::size_of::<Edge>();
            if let Some(max_edges) = n.checked_add(1)
                .and_then(|next| n.checked_mul(next))
                .map(|count| count / 2)
                .filter(|&count| count <= max_vec_len)
            {
                let _ = edges.try_reserve_exact(max_edges);
            }
        }

        for i in 0..n {
            edges.push(Edge { i, j: i, value: challenge.values[i] as f64 });
        }
        for i in 0..n {
            let row = &challenge.interaction_values[i];
            for (offset, &v) in row[(i + 1)..n].iter().enumerate() {
                if v != 0 {
                    edges.push(Edge {
                        i,
                        j: i + 1 + offset,
                        value: v as f64,
                    });
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

    pub struct UtilMatrix {
        pub n:      usize,
        pub data:   Vec<f64>,
        pub linear: Vec<f64>,
    }
    impl UtilMatrix {
        pub fn get(&self, r: usize, c: usize) -> f64 { self.data[r * self.n + c] }
        pub fn set(&mut self, r: usize, c: usize, v: f64) { self.data[r * self.n + c] = v; }
    }



    fn compute_ofv_from_matrix(sel: &[usize], um: &UtilMatrix) -> f64 {
        let mut ofv = 0.0f64;
        for (idx, &a) in sel.iter().enumerate() {
            ofv += um.linear[a];
            for &b in &sel[(idx + 1)..] {
                ofv += um.get(a, b);
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
            pub adjacency_list:       Vec<HpfNode>,
            pub strong_roots:         Vec<HpfRoot>,
            pub label_count:          Vec<i32>,
            pub arc_list:             Vec<HpfArc>,
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
            let src_idx = ((*s).source - 1) as usize;
            let size    = (*s).adjacency_list[src_idx].num_out_of_tree as usize;
            for i in 0..size {
                let arc   = (*s).adjacency_list[src_idx].out_of_tree[i];
                let delta = (*arc).capacities[theparam as usize] - (*arc).capacity;
                if delta < 0.0 { return; }
                (*arc).capacity     += delta;
                (*arc).flow         += delta;
                (*(*arc).to).excess += delta;
                if (*(*arc).to).label < (*s).num_nodes && (*(*arc).to).excess > 0.0 {
                    push_excess(s, (*arc).to);
                }
            }
            let snk_idx = ((*s).sink - 1) as usize;
            let size    = (*s).adjacency_list[snk_idx].num_out_of_tree as usize;
            for i in 0..size {
                let arc   = (*s).adjacency_list[snk_idx].out_of_tree[i];
                let delta = (*arc).capacities[theparam as usize] - (*arc).capacity;
                if delta > 0.0 { return; }
                (*arc).capacity       += delta;
                (*arc).flow           += delta;
                (*(*arc).from).excess -= delta;
                if (*(*arc).from).label < (*s).num_nodes && (*(*arc).from).excess > 0.0 {
                    push_excess(s, (*arc).from);
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

        pub struct BreakpointSets { pub sets: Vec<(i32, Vec<usize>)> }

        pub fn get_breakpoints(inst: &P1Instance,
                                n_lambda_values: usize) -> BreakpointSets {
            let n_items    = inst.n_items;
            let n_edges    = inst.n_edges;
            let num_nodes  = (n_items + 2) as i32;
            let num_arcs   = (n_edges + 2 * n_items) as i32;
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
            for k in 0..n_edges {
                let from_id = inst.edges[k].i;
                let to_id   = inst.edges[k].j;
                let cap     = inst.edges[k].value as f32;
                arc_list.push(HpfArc {
                    from: std::ptr::null_mut(),
                    to: std::ptr::null_mut(),
                    flow: 0.0,
                    capacity: cap,
                    direction: 1,
                    capacities: vec![cap],
                });
                adjacency_list[from_id + 2].wt += cap;
                adjacency_list[from_id + 2].num_adjacent += 1;
                adjacency_list[to_id   + 2].num_adjacent += 1;
            }

            let mut max_degree_ratio = 0.0f32;
            for i in 2..num_nodes as usize {
                let ratio = adjacency_list[i].wt / adjacency_list[i].cst;
                if ratio > max_degree_ratio { max_degree_ratio = ratio; }
            }

            let step: f32 = if num_params > 0 {
                max_degree_ratio / num_params as f32
            } else { 0.0 };
            let params: Vec<f32> = (0..num_params as usize)
                .map(|i| max_degree_ratio - i as f32 * step)
                .collect();

            for i in 2..num_nodes as usize {
                let wt  = adjacency_list[i].wt;
                let cst = adjacency_list[i].cst;
                let src_caps: Vec<f32> = params.iter()
                    .map(|&p| { let v = wt - p * cst; if v > 0.0 { v } else { 0.0 } })
                    .collect();
                let src_capacity = src_caps.first().copied().unwrap_or(0.0);
                arc_list.push(HpfArc {
                    from: std::ptr::null_mut(),
                    to: std::ptr::null_mut(),
                    flow: 0.0,
                    capacity: src_capacity,
                    direction: 1,
                    capacities: src_caps,
                });
                adjacency_list[0].num_adjacent += 1;
                adjacency_list[i].num_adjacent += 1;

                let snk_caps: Vec<f32> = params.iter()
                    .map(|&p| { let v = p * cst - wt; if v > 0.0 { v } else { 0.0 } })
                    .collect();
                let snk_capacity = snk_caps.first().copied().unwrap_or(0.0);
                arc_list.push(HpfArc {
                    from: std::ptr::null_mut(),
                    to: std::ptr::null_mut(),
                    flow: 0.0,
                    capacity: snk_capacity,
                    direction: 1,
                    capacities: snk_caps,
                });
                adjacency_list[i].num_adjacent += 1;
                adjacency_list[1].num_adjacent += 1;
            }

            let sentinel_count = 2 * num_nodes as usize;
            let mut root_sentinels: Vec<HpfNode> = Vec::with_capacity(sentinel_count);
            for _ in 0..sentinel_count {
                root_sentinels.push(HpfNode::zeroed(num_params));
            }
            let strong_roots: Vec<HpfRoot> = unsafe {
                let base = root_sentinels.as_mut_ptr();
                (0..num_nodes as usize).map(|i| {
                    let start = base.add(2 * i);
                    let end = start.add(1);
                    (*start).next = end;
                    (*end).prev = start;
                    HpfRoot { start, end }
                }).collect()
            };
            let label_count = vec![0i32; num_nodes as usize];

            let mut state = HpfState {
                num_nodes, num_arcs, source: 1, sink: 2,
                num_params, highest_strong_label: 1,
                adjacency_list, strong_roots, label_count, arc_list,
            };

            unsafe {
                let s = &mut state as *mut HpfState;

                let mut first = 0usize;
                for k in 0..n_edges {
                    (*s).arc_list[first].from = &mut (*s).adjacency_list[inst.edges[k].i + 2];
                    (*s).arc_list[first].to   = &mut (*s).adjacency_list[inst.edges[k].j + 2];
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
                    if source == to_num || sink == from_num || from_num == to_num { continue; }
                    if source == from_num && to_num == sink {
                        (*s).arc_list[i].flow = cap;
                    } else if from_num == source {
                        let arc_ptr = &mut (*s).arc_list[i] as *mut HpfArc;
                        (*s).adjacency_list[(from_num-1) as usize].out_of_tree.push(arc_ptr);
                        (*s).adjacency_list[(from_num-1) as usize].num_out_of_tree += 1;
                    } else if to_num == sink {
                        let arc_ptr = &mut (*s).arc_list[i] as *mut HpfArc;
                        (*s).adjacency_list[(to_num-1) as usize].out_of_tree.push(arc_ptr);
                        (*s).adjacency_list[(to_num-1) as usize].num_out_of_tree += 1;
                    } else {
                        let arc_ptr = &mut (*s).arc_list[i] as *mut HpfArc;
                        (*s).adjacency_list[(from_num-1) as usize].out_of_tree.push(arc_ptr);
                        (*s).adjacency_list[(from_num-1) as usize].num_out_of_tree += 1;
                    }
                }

                simple_initialization(s);
                pseudoflow_phase1(s);

                let domain_len = num_params as usize + 2;
                let mut counts = vec![0usize; domain_len];
                for i in 0..num_nodes as usize {
                    let node_num = (*s).adjacency_list[i].number;
                    if node_num == 1 || node_num == 2 { continue; }
                    let breakpoint = (*s).adjacency_list[i].breakpoint;
                    assert!(breakpoint >= 0 && (breakpoint as usize) < domain_len);
                    counts[breakpoint as usize] += 1;
                }

                let mut offsets = vec![0usize; domain_len + 1];
                let mut total = 0usize;
                for pos in 0..domain_len {
                    let count = counts[pos];
                    counts[pos] = total;
                    total += count;
                    offsets[pos + 1] = total;
                }

                let mut grouped_items = vec![0usize; total];
                for i in 0..num_nodes as usize {
                    let node_num = (*s).adjacency_list[i].number;
                    if node_num == 1 || node_num == 2 { continue; }
                    let pos = (*s).adjacency_list[i].breakpoint as usize;
                    grouped_items[counts[pos]] = (node_num - 3) as usize;
                    counts[pos] += 1;
                }

                let mut sets: Vec<(i32, Vec<usize>)> = Vec::new();
                for pos in 0..domain_len {
                    let start = offsets[pos];
                    let end = offsets[pos + 1];
                    if start != end {
                        sets.push((pos as i32, grouped_items[start..end].to_vec()));
                    }
                }

                BreakpointSets { sets }
            }
        }
    } 

    pub use hpf::get_breakpoints;

    pub type IntArray = Vec<usize>;
    pub type DblArray = Vec<f64>;

    #[inline] pub fn ia_contains(a: &IntArray, v: usize) -> bool { a.contains(&v) }

    struct WarmStartLeft {
        valid:                bool,
        candidate_nodes:      IntArray,
        candidate_contribs:   DblArray,
        current_total_weight: f64,
        left_nodes:           IntArray,
    }
    impl WarmStartLeft {
        fn new() -> Self {
            WarmStartLeft { valid: false, candidate_nodes: Vec::new(),
                candidate_contribs: Vec::new(), current_total_weight: -1.0,
                left_nodes: Vec::new() }
        }
        fn reset(&mut self) {
            self.valid = false; self.candidate_nodes.clear();
            self.candidate_contribs.clear(); self.left_nodes.clear();
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
            WarmStartRight { valid: false, candidate_nodes: Vec::new(),
                candidate_contribs: Vec::new(), current_total_weight: 0.0 }
        }
        fn reset(&mut self) {
            self.valid = false; self.candidate_nodes.clear();
            self.candidate_contribs.clear(); self.current_total_weight = 0.0;
        }
    }

    fn seed_empty_left(left: &mut IntArray, interaction_totals: &[f64],
                       weights: &[i32], budget: i32) {
        if !left.is_empty() { return; }
        let mut best: Option<usize> = None;
        let mut best_val = f64::NEG_INFINITY;
        for nd in 0..interaction_totals.len() {
            if weights[nd] > budget { continue; }
            let util = interaction_totals[nd] / weights[nd] as f64;
            if util > best_val { best_val = util; best = Some(nd); }
        }
        if let Some(nd) = best { left.push(nd); }
    }

    fn run_greedy_left(um:          &UtilMatrix,
                       n_nodes:     usize,
                       mut left:    IntArray,
                       right_nodes: &IntArray,
                       budget:      i32,
                       beta:        f64,
                       weights:     &[i32],
                       mut ws:      Option<&mut WarmStartLeft>) -> IntArray {
        let mut left_mark = vec![false; n_nodes];
        for &nd in &left {
            left_mark[nd] = true;
        }
        let mut cur_w: f64 = match &ws {
            Some(w) if w.valid && w.current_total_weight >= 0.0 => w.current_total_weight,
            _ => left.iter().map(|&k| weights[k] as f64).sum(),
        };
        let ws_valid          = ws.as_ref().map(|w| w.valid).unwrap_or(false);
        let ws_cands_nonempty = ws.as_ref().map(|w| !w.candidate_nodes.is_empty()).unwrap_or(false);
        let mut cand_nodes:    IntArray = Vec::new();
        let mut cand_contribs: DblArray = Vec::new();
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
            update_flag = true;
            if beta == 0.0 {
                let mut marginals = um.linear.clone();
                for &m in &left {
                    let row = &um.data[m * n_nodes..(m + 1) * n_nodes];
                    for (marginal, &value) in marginals.iter_mut().zip(row.iter()) {
                        *marginal += value;
                    }
                }
                for &nd in right_nodes {
                    if left_mark[nd] { continue; }
                    if weights[nd] as f64 > budget as f64 - cur_w { continue; }
                    cand_nodes.push(nd);
                    cand_contribs.push(marginals[nd] / weights[nd] as f64);
                }
            } else {
                for &nd in right_nodes {
                    if left_mark[nd] { continue; }
                    if weights[nd] as f64 > budget as f64 - cur_w { continue; }
                    cand_nodes.push(nd);
                }
                for &nd in &cand_nodes {
                    let mut contrib: f64 = left.iter()
                        .map(|&m| (1.0 + beta) * um.get(nd, m)).sum();
                    contrib += um.linear[nd];
                    for &m in &cand_nodes { contrib -= beta * um.get(nd, m); }
                    for v in 0..n_nodes {
                        if !ia_contains(right_nodes, v) { contrib -= beta * um.get(nd, v); }
                    }
                    contrib /= weights[nd] as f64;
                    cand_contribs.push(contrib);
                }
            }
        }
        loop {
            if cand_nodes.is_empty() { break; }
            let best_idx = cand_contribs.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap();
            let best_node = cand_nodes[best_idx];
            left.push(best_node);
            left_mark[best_node] = true;
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
                        if !ia_contains(&right_nodes, v) { contrib += beta * um.get(nd, v); }
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

    fn run_greedy(inst: &P1Instance, beta: f64,
                  breakpoints: &[IntArray], bp_weights: &[f64]) -> GreedyResults {
        let n_budgets  = inst.n_budgets;
        let n_nodes    = inst.n_items;
        let n_bp_total = breakpoints.len();
        let mut um = UtilMatrix {
            n: n_nodes,
            data: vec![0.0f64; n_nodes * n_nodes],
            linear: vec![0.0f64; n_nodes],
        };
        let mut interaction_totals = vec![0.0f64; n_nodes];
        for e in &inst.edges {
            if e.i == e.j {
                um.linear[e.i] += e.value;
            } else {
                um.data[e.i * n_nodes + e.j] = e.value;
                um.data[e.j * n_nodes + e.i] = e.value;
                interaction_totals[e.i] += e.value;
                interaction_totals[e.j] += e.value;
            }
        }
        let all_nodes: IntArray = (0..n_nodes).collect();

        let mut left_nodes: Vec<IntArray> = Vec::with_capacity(n_budgets);

        for bi in 0..n_budgets {
            let budget = inst.budgets[bi];
            let mut bp_idx = 0usize;
            for k in 0..n_bp_total {
                if bp_weights[k] <= budget as f64 { bp_idx = k; }
            }
            let mut left_init = breakpoints[bp_idx].clone();
            seed_empty_left(&mut left_init, &interaction_totals, &inst.weights, budget);
            let res = run_greedy_left(&um, n_nodes, left_init, &all_nodes,
                                      budget, beta, &inst.weights, None);
            left_nodes.push(res);
        }

        let mut right_nodes: Vec<IntArray> = vec![Vec::new(); n_budgets];

        for bi in (0..n_budgets).rev() {
            let budget = inst.budgets[bi];
            let mut bp_idx = n_bp_total - 1;
            for k in 0..n_bp_total {
                if bp_weights[k] >= budget as f64 { bp_idx = k; break; }
            }
            let right_init = breakpoints[bp_idx].clone();
            let mut after_right = run_greedy_right(&um, n_nodes, right_init,
                                                    budget, beta, &inst.weights, None);
            seed_empty_left(&mut after_right, &interaction_totals, &inst.weights, budget);
            let final_left = run_greedy_left(&um, n_nodes, after_right, &all_nodes,
                                             budget, beta, &inst.weights, None);
            right_nodes[bi] = final_left;
        }

        GreedyResults { left_nodes, right_nodes }
    }

    pub fn run_bp_algorithm(inst: &P1Instance, n_lambda_values: usize) -> QKPResult {
        let bps = get_breakpoints(inst, n_lambda_values);
        let n_breakpoints = bps.sets.len();

        let mut total_weights_at_bp = vec![0.0f64; n_breakpoints];
        {
            let mut cumsum = 0.0f64;
            for (i, (_, nodes)) in bps.sets.iter().enumerate() {
                for &nd in nodes { cumsum += inst.weights[nd] as f64; }
                total_weights_at_bp[i] = cumsum;
            }
        }

        let n_bp_total = n_breakpoints + 1;
        let mut breakpoints: Vec<IntArray> = Vec::with_capacity(n_bp_total);
        let mut bp_weights:  Vec<f64>      = Vec::with_capacity(n_bp_total);
        breakpoints.push(Vec::new());
        bp_weights.push(0.0);
        for i in 0..n_breakpoints {
            let mut next = breakpoints[i].clone();
            for &nd in &bps.sets[i].1 { next.push(nd); }
            breakpoints.push(next);
            bp_weights.push(total_weights_at_bp[i]);
        }

        let gr = run_greedy(inst, 0.0, &breakpoints, &bp_weights);
        let mut selected = vec![0u8; inst.n_items];

        let mut results: Vec<BudgetResult> = Vec::with_capacity(inst.n_budgets);
        for bi in 0..inst.n_budgets {
            selected.fill(0);
            for &i in &gr.left_nodes[bi] { selected[i] |= 1; }
            for &i in &gr.right_nodes[bi] { selected[i] |= 2; }

            let mut ofv_left = 0.0f64;
            let mut ofv_right = 0.0f64;
            for e in &inst.edges {
                let common = selected[e.i] & selected[e.j];
                if common & 1 != 0 { ofv_left += e.value; }
                if common & 2 != 0 { ofv_right += e.value; }
            }

            let (best_items, ofv) = if ofv_left >= ofv_right {
                (gr.left_nodes[bi].clone(), ofv_left)
            } else {
                (gr.right_nodes[bi].clone(), ofv_right)
            };
            results.push(BudgetResult {
                budget: inst.budgets[bi], ofv,
                cpu: 0.0,
                selected_items: best_items,
            });
        }
        QKPResult { results }
    }

    pub struct P2Instance {
        pub n:        usize,
        pub m:        i64,
        pub capacity: i32,
        pub w:        Vec<i32>,
        pub q:        Vec<i32>,
    }

    impl P2Instance {
        #[inline] pub fn q_get(&self, i: usize, j: usize) -> i32 { self.q[i * self.n + j] }
        #[inline] pub fn q_set(&mut self, i: usize, j: usize, v: i32) { self.q[i * self.n + j] = v; }
    }

    pub fn bridge_p1_to_p2(inst: &P1Instance, budget: i32) -> P2Instance {
        let n = inst.n_items;
        let mut p2 = P2Instance {
            n, m: inst.n_edges as i64, capacity: budget,
            w: inst.weights.clone(), q: vec![0i32; n * n],
        };
        for e in &inst.edges {
            if e.i >= n || e.j >= n { continue; }
            let qv = e.value as i32;
            if e.i == e.j { p2.q_set(e.i, e.i, qv); }
            else { p2.q_set(e.i, e.j, qv); p2.q_set(e.j, e.i, qv); }
        }
        p2
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
                sel: vec![0u8; n], best_sel: vec![0u8; n], contrib: vec![0i32; n],
                value: 0, weight: 0, count: 0,
                best_value: NEG_INF, best_weight: 0, best_count: 0, ins,
            }
        }
        pub fn clear(&mut self) {
            self.sel.iter_mut().for_each(|x| *x = 0);
            self.contrib.iter_mut().for_each(|x| *x = 0);
            self.value = 0; self.weight = 0; self.count = 0;
        }
        #[inline] pub fn slack(&self) -> i32 { self.ins.capacity - self.weight }
        pub fn add_item(&mut self, i: usize) {
            if i >= self.ins.n || self.sel[i] != 0 { return; }
            self.value  += self.contrib[i] as Ll;
            self.weight += self.ins.w[i];
            self.count  += 1;
            self.sel[i]  = 1;
            let start = i * self.ins.n;
            let row = &self.ins.q[start..start + self.ins.n];
            for (c, &q) in self.contrib.iter_mut().zip(row.iter()) { *c += q; }
        }
        pub fn remove_item(&mut self, i: usize) {
            if i >= self.ins.n || self.sel[i] == 0 { return; }
            self.value  -= self.contrib[i] as Ll;
            self.weight -= self.ins.w[i];
            self.count  -= 1;
            self.sel[i]  = 0;
            let start = i * self.ins.n;
            let row = &self.ins.q[start..start + self.ins.n];
            for (c, &q) in self.contrib.iter_mut().zip(row.iter()) { *c -= q; }
        }
        pub fn replace_item(&mut self, rm: usize, add: usize) {
            self.remove_item(rm); self.add_item(add);
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
            let selected: Vec<usize> = (0..self.ins.n)
                .filter(|&i| self.best_sel[i] != 0).collect();
            for i in selected { self.add_item(i); }
        }
        pub fn eval_selected(ins: &P2Instance, bits: &[u8]) -> Ll {
            let n = ins.n; let mut val: Ll = 0;
            for i in 0..n {
                if bits[i] == 0 { continue; }
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

    fn compute_total_interactions(ins: &P2Instance) -> Vec<Ll> {
        let n = ins.n;
        (0..n).map(|i| (0..n).map(|j| ins.q_get(i, j) as Ll).sum()).collect()
    }

    pub fn bridge_load_solution(s: &mut P2State, br: &BudgetResult) {
        s.sel.fill(0);
        s.value = 0;
        s.weight = 0;
        s.count = 0;

        let n = s.ins.n;
        let mut accepted = Vec::with_capacity(br.selected_items.len());
        for &id in &br.selected_items {
            if id < n && s.sel[id] == 0
                && s.weight + s.ins.w[id] <= s.ins.capacity
            {
                s.sel[id] = 1;
                s.weight += s.ins.w[id];
                s.count += 1;
                accepted.push(id);
            }
        }

        if accepted.is_empty() {
            s.contrib.fill(0);
            s.save_best();
            return;
        }

        for (pos, &id) in accepted.iter().enumerate() {
            let mut marginal = 0i32;
            for &previous in &accepted[..pos] {
                marginal += s.ins.q[previous * n + id];
            }
            s.value += marginal as Ll;
        }

        let block_size = n.min(4096);
        let mut accumulator = vec![0i32; block_size];
        let mut start = 0usize;
        while start < n {
            let end = (start + block_size).min(n);
            let len = end - start;
            accumulator[..len].fill(0);
            for &id in &accepted {
                let row_start = id * n + start;
                let row = &s.ins.q[row_start..row_start + len];
                for (total, &value) in accumulator[..len].iter_mut().zip(row.iter()) {
                    *total += value;
                }
            }
            s.contrib[start..end].copy_from_slice(&accumulator[..len]);
            start = end;
        }

        s.save_best();
    }

    #[derive(Clone, Copy)]
    struct ItemScore { id: usize, score: Ll }

    fn build_best_unused(s: &P2State, max_k: usize) -> Vec<ItemScore> {
        if max_k == 0 { return Vec::new(); }
        let n = s.ins.n;
        let mut density: Vec<ItemScore> = (0..n).filter(|&i| s.sel[i] == 0).map(|i| {
            let w = s.ins.w[i].max(1) as Ll;
            ItemScore { id: i, score: s.contrib[i] as Ll * 1_000_000 / w }
        }).collect();
        if density.len() <= max_k {
            density.sort_unstable_by(|a, b| b.score.cmp(&a.score));
            return density;
        }

        let density_k = (max_k * 3 / 4).max(1);
        density.select_nth_unstable_by(density_k, |a, b| b.score.cmp(&a.score));
        density.truncate(density_k);
        let marginal_k = max_k - density.len();
        if marginal_k == 0 {
            density.sort_unstable_by(|a, b| b.score.cmp(&a.score));
            return density;
        }

        let mut marginal: Vec<ItemScore> = (0..n).filter(|&i| s.sel[i] == 0).map(|i| {
            ItemScore { id: i, score: s.contrib[i] as Ll }
        }).collect();
        marginal.select_nth_unstable_by(marginal_k, |a, b| b.score.cmp(&a.score));
        marginal.truncate(marginal_k);

        let mut present = vec![false; n];
        for is in &density { present[is.id] = true; }
        for is in marginal {
            if !present[is.id] {
                let w = s.ins.w[is.id].max(1) as Ll;
                density.push(ItemScore {
                    id: is.id,
                    score: s.contrib[is.id] as Ll * 1_000_000 / w,
                });
                present[is.id] = true;
            }
        }
        if density.len() < max_k {
            let mut fill: Vec<ItemScore> = (0..n)
                .filter(|&i| s.sel[i] == 0 && !present[i])
                .map(|i| {
                    let w = s.ins.w[i].max(1) as Ll;
                    ItemScore { id: i, score: s.contrib[i] as Ll * 1_000_000 / w }
                })
                .collect();
            let need = max_k - density.len();
            if fill.len() > need {
                fill.select_nth_unstable_by(need, |a, b| b.score.cmp(&a.score));
                fill.truncate(need);
            }
            density.extend(fill);
        }
        density.sort_unstable_by(|a, b| b.score.cmp(&a.score));
        density
    }

    fn build_worst_used(s: &P2State, max_k: usize) -> Vec<ItemScore> {
        if max_k == 0 { return Vec::new(); }
        let n = s.ins.n;
        let mut density: Vec<ItemScore> = (0..n).filter(|&i| s.sel[i] != 0).map(|i| {
            let w = s.ins.w[i].max(1) as Ll;
            ItemScore { id: i, score: s.contrib[i] as Ll * 1_000_000 / w }
        }).collect();
        if density.len() <= max_k {
            density.sort_unstable_by(|a, b| a.score.cmp(&b.score));
            return density;
        }

        let density_k = (max_k * 3 / 4).max(1);
        density.select_nth_unstable_by(density_k, |a, b| a.score.cmp(&b.score));
        density.truncate(density_k);
        let marginal_k = max_k - density.len();
        if marginal_k == 0 {
            density.sort_unstable_by(|a, b| a.score.cmp(&b.score));
            return density;
        }

        let mut marginal: Vec<ItemScore> = (0..n).filter(|&i| s.sel[i] != 0).map(|i| {
            ItemScore { id: i, score: s.contrib[i] as Ll }
        }).collect();
        marginal.select_nth_unstable_by(marginal_k, |a, b| a.score.cmp(&b.score));
        marginal.truncate(marginal_k);

        let mut present = vec![false; n];
        for is in &density { present[is.id] = true; }
        for is in marginal {
            if !present[is.id] {
                let w = s.ins.w[is.id].max(1) as Ll;
                density.push(ItemScore {
                    id: is.id,
                    score: s.contrib[is.id] as Ll * 1_000_000 / w,
                });
                present[is.id] = true;
            }
        }
        if density.len() < max_k {
            let mut fill: Vec<ItemScore> = (0..n)
                .filter(|&i| s.sel[i] != 0 && !present[i])
                .map(|i| {
                    let w = s.ins.w[i].max(1) as Ll;
                    ItemScore { id: i, score: s.contrib[i] as Ll * 1_000_000 / w }
                })
                .collect();
            let need = max_k - density.len();
            if fill.len() > need {
                fill.select_nth_unstable_by(need, |a, b| a.score.cmp(&b.score));
                fill.truncate(need);
            }
            density.extend(fill);
        }
        density.sort_unstable_by(|a, b| a.score.cmp(&b.score));
        density
    }

    struct Rng { state: u64 }
    impl Rng {
        fn new(seed: u64) -> Self { Rng { state: if seed == 0 { 1 } else { seed } } }
        #[inline]
        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 7; x ^= x >> 9; x ^= x << 8;
            self.state = x; x
        }
        #[inline]
        fn next_int(&mut self, bound: usize) -> usize {
            if bound == 0 { return 0; }
            (self.next_u64() % bound as u64) as usize
        }
    }

    fn apply_best_add1(s: &mut P2State) -> bool {
        let n = s.ins.n;
        let rem = s.slack();
        if rem <= 0 { return false; }
        let mut best = None;
        let mut best_delta = 0i32;
        for i in 0..n {
            if s.sel[i] != 0 || s.ins.w[i] > rem { continue; }
            let d = s.contrib[i];
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
            let a = unused[x].id; let wa = s.ins.w[a]; let ca = s.contrib[a] as Ll;
            if wa >= rem { continue; }
            for y in (x+1)..cu {
                let b = unused[y].id;
                if wa + s.ins.w[b] > rem { continue; }
                let d = ca + s.contrib[b] as Ll + s.ins.q_get(a, b) as Ll;
                if d > best_delta { best_delta = d; ba = Some(a); bb = Some(b); }
            }
        }
        if let (Some(a), Some(b)) = (ba, bb) {
            s.add_item(a); s.add_item(b);
            return true;
        }
        false
    }

    fn build_synergy_unused(s: &P2State) -> Vec<ItemScore> {
        const SEED_K: usize = 36;
        const BASE_K: usize = 12;
        const PAIR_K: usize = 6;
        const POOL_K: usize = 32;

        let ranked = build_best_unused(s, SEED_K);
        if ranked.is_empty() { return ranked; }

        let mut pool: Vec<ItemScore> = ranked.iter().take(BASE_K).copied().collect();
        let mut present = vec![false; s.ins.n];
        for is in &pool { present[is.id] = true; }

        let mut pairs: Vec<(Ll, usize, usize)> = Vec::new();
        for x in 0..ranked.len() {
            let a = ranked[x].id;
            for y in (x + 1)..ranked.len() {
                let b = ranked[y].id;
                let score = s.contrib[a] as Ll + s.contrib[b] as Ll
                    + s.ins.q_get(a, b) as Ll;
                pairs.push((score, a, b));
            }
        }
        if pairs.len() > PAIR_K {
            pairs.select_nth_unstable_by(PAIR_K, |a, b| b.0.cmp(&a.0));
            pairs.truncate(PAIR_K);
        }
        pairs.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        for (_, a, b) in pairs {
            for id in [a, b] {
                if !present[id] && pool.len() < POOL_K {
                    let w = s.ins.w[id].max(1) as Ll;
                    pool.push(ItemScore {
                        id,
                        score: s.contrib[id] as Ll * 1_000_000 / w,
                    });
                    present[id] = true;
                }
            }
        }

        if pool.len() < POOL_K {
            let mut extra: Vec<ItemScore> = (0..s.ins.n)
                .filter(|&i| s.sel[i] == 0 && !present[i])
                .map(|i| {
                    let mut best_pair = 0i64;
                    for seed in &pool {
                        let pair = s.ins.q_get(i, seed.id) as Ll;
                        if pair > best_pair { best_pair = pair; }
                    }
                    ItemScore { id: i, score: s.contrib[i] as Ll + best_pair }
                })
                .collect();
            let need = POOL_K - pool.len();
            if extra.len() > need {
                extra.select_nth_unstable_by(need, |a, b| b.score.cmp(&a.score));
                extra.truncate(need);
            }
            pool.extend(extra);
        }
        pool
    }

    fn apply_synergy_tail(s: &mut P2State) -> bool {
        if s.ins.n > 2000 { return false; }
        let unused = build_synergy_unused(s);

        {
            let used = build_worst_used(s, 32);
            if unused.len() >= 2 && !used.is_empty() {
                let mut best_delta: Ll = 0;
                let mut best = None;
                for rm_score in &used {
                    let rm = rm_score.id;
                    let budget = s.slack() + s.ins.w[rm];
                    let lost = s.contrib[rm] as Ll;
                    for x in 0..unused.len() {
                        let a = unused[x].id;
                        let wa = s.ins.w[a];
                        if wa >= budget { continue; }
                        let a_gain = s.contrib[a] as Ll - s.ins.q_get(a, rm) as Ll;
                        for y in (x + 1)..unused.len() {
                            let b = unused[y].id;
                            if wa + s.ins.w[b] > budget { continue; }
                            let delta = a_gain
                                + s.contrib[b] as Ll
                                - s.ins.q_get(b, rm) as Ll
                                + s.ins.q_get(a, b) as Ll
                                - lost;
                            if delta > best_delta {
                                best_delta = delta;
                                best = Some((rm, a, b));
                            }
                        }
                    }
                }
                if let Some((rm, a, b)) = best {
                    s.remove_item(rm);
                    s.add_item(a);
                    s.add_item(b);
                    return true;
                }
            }
        }

        {
            let limit = unused.len().min(24);
            let used = build_worst_used(s, 24);
            if limit >= 3 && !used.is_empty() {
                let mut best_delta: Ll = 0;
                let mut best = None;
                for rm_score in &used {
                    let rm = rm_score.id;
                    let budget = s.slack() + s.ins.w[rm];
                    let lost = s.contrib[rm] as Ll;
                    for x in 0..limit {
                        let a = unused[x].id;
                        let wa = s.ins.w[a];
                        if wa > budget { continue; }
                        let a_gain = s.contrib[a] as Ll - s.ins.q_get(a, rm) as Ll;
                        for y in (x + 1)..limit {
                            let b = unused[y].id;
                            let wab = wa + s.ins.w[b];
                            if wab > budget { continue; }
                            let b_gain = s.contrib[b] as Ll - s.ins.q_get(b, rm) as Ll;
                            for z in (y + 1)..limit {
                                let c = unused[z].id;
                                if wab + s.ins.w[c] > budget { continue; }
                                let delta = a_gain
                                    + b_gain
                                    + s.contrib[c] as Ll
                                    - s.ins.q_get(c, rm) as Ll
                                    + s.ins.q_get(a, b) as Ll
                                    + s.ins.q_get(a, c) as Ll
                                    + s.ins.q_get(b, c) as Ll
                                    - lost;
                                if delta > best_delta {
                                    best_delta = delta;
                                    best = Some((rm, a, b, c));
                                }
                            }
                        }
                    }
                }
                if let Some((rm, a, b, c)) = best {
                    s.remove_item(rm);
                    s.add_item(a);
                    s.add_item(b);
                    s.add_item(c);
                    return true;
                }
            }
        }

        if s.ins.n <= 1200 {
            let limit = unused.len().min(20);
            let used = build_worst_used(s, 20);
            if limit >= 3 && used.len() >= 2 {
                let n = s.ins.n;
                let q = &s.ins.q;
                let w = &s.ins.w;
                let contrib = &s.contrib;
                let slack = s.slack();

                let mut removed_pairs: Vec<(usize, usize, i32, Ll)> =
                    Vec::with_capacity(used.len() * (used.len() - 1) / 2);
                for x in 0..used.len() {
                    let r1 = used[x].id;
                    let r1_base = r1 * n;
                    for y in (x + 1)..used.len() {
                        let r2 = used[y].id;
                        let lost = contrib[r1] as Ll + contrib[r2] as Ll
                            - q[r1_base + r2] as Ll;
                        removed_pairs.push((r1, r2, slack + w[r1] + w[r2], lost));
                    }
                }

                let mut best_delta: Ll = 0;
                let mut best = None;
                for &(r1, r2, budget, lost) in &removed_pairs {
                    for x in 0..limit {
                        let a = unused[x].id;
                        let wa = w[a];
                        if wa > budget { continue; }
                        let a_base = a * n;
                        let a_gain = contrib[a] as Ll - q[a_base + r1] as Ll - q[a_base + r2] as Ll;
                        for y in (x + 1)..limit {
                            let b = unused[y].id;
                            let wab = wa + w[b];
                            if wab > budget { continue; }
                            let b_base = b * n;
                            let b_gain = contrib[b] as Ll - q[b_base + r1] as Ll - q[b_base + r2] as Ll;
                            for z in (y + 1)..limit {
                                let c = unused[z].id;
                                if wab + w[c] > budget { continue; }
                                let c_base = c * n;
                                let delta = a_gain + b_gain
                                    + contrib[c] as Ll
                                    - q[c_base + r1] as Ll
                                    - q[c_base + r2] as Ll
                                    + q[a_base + b] as Ll
                                    + q[a_base + c] as Ll
                                    + q[b_base + c] as Ll
                                    - lost;
                                if delta > best_delta {
                                    best_delta = delta;
                                    best = Some((r1, r2, a, b, c));
                                }
                            }
                        }
                    }
                }

                if let Some((r1, r2, a, b, c)) = best {
                    s.remove_item(r1);
                    s.remove_item(r2);
                    s.add_item(a);
                    s.add_item(b);
                    s.add_item(c);
                    return true;
                }
            }
        }

        if s.slack() <= 0 || unused.len() < 3 { return false; }
        let rem = s.slack();
        let mut best_delta: Ll = 0;
        let mut best = None;
        for x in 0..unused.len() {
            let a = unused[x].id;
            let wa = s.ins.w[a];
            if wa >= rem { continue; }
            for y in (x + 1)..unused.len() {
                let b = unused[y].id;
                let wab = wa + s.ins.w[b];
                if wab >= rem { continue; }
                for z in (y + 1)..unused.len() {
                    let c = unused[z].id;
                    if wab + s.ins.w[c] > rem { continue; }
                    let delta = s.contrib[a] as Ll + s.contrib[b] as Ll + s.contrib[c] as Ll
                        + s.ins.q_get(a, b) as Ll
                        + s.ins.q_get(a, c) as Ll
                        + s.ins.q_get(b, c) as Ll;
                    if delta > best_delta {
                        best_delta = delta;
                        best = Some((a, b, c));
                    }
                }
            }
        }
        if let Some((a, b, c)) = best {
            s.add_item(a);
            s.add_item(b);
            s.add_item(c);
            return true;
        }
        false
    }



    fn apply_best_swap11(s: &mut P2State, window_k: usize) -> bool {
        let unused = build_best_unused(s, window_k);
        let used   = build_worst_used(s, window_k);
        let cu = unused.len(); let cs = used.len();
        let n = s.ins.n;
        let slack = s.slack();
        let q = &s.ins.q;
        let w = &s.ins.w;
        let contrib = &s.contrib;
        let mut best_delta: Ll = 0;
        let mut best_ordinal = usize::MAX;
        let mut br = None; let mut ba = None;
        for y in 0..cu {
            let add = unused[y].id;
            let add_weight = w[add];
            let add_contrib = contrib[add] as Ll;
            let row_base = add * n;
            for x in 0..cs {
                let rm = used[x].id;
                if add_weight > slack + w[rm] { continue; }
                let d = add_contrib - contrib[rm] as Ll
                        - q[row_base + rm] as Ll;
                let ordinal = x * cu + y;
                if d > best_delta
                    || (d == best_delta && d > 0 && ordinal < best_ordinal)
                {
                    best_delta = d;
                    best_ordinal = ordinal;
                    br = Some(rm);
                    ba = Some(add);
                }
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
        let cu = unused.len(); let cs = used.len();
        let mut best_delta: Ll = 0;
        let mut br = None; let mut ba = None; let mut bb = None;
        for si in 0..cs {
            let rm = used[si].id; let budget = s.slack() + s.ins.w[rm];
            let lost = s.contrib[rm] as Ll;
            for x in 0..cu {
                let a = unused[x].id; let wa = s.ins.w[a];
                if wa >= budget { continue; }
                let ca_eff = s.contrib[a] as Ll - s.ins.q_get(a, rm) as Ll;
                for y in (x+1)..cu {
                    let b = unused[y].id;
                    if wa + s.ins.w[b] > budget { continue; }
                    let cb_eff = s.contrib[b] as Ll - s.ins.q_get(b, rm) as Ll;
                    let d = ca_eff + cb_eff + s.ins.q_get(a, b) as Ll - lost;
                    if d > best_delta {
                        best_delta = d; br = Some(rm); ba = Some(a); bb = Some(b);
                    }
                }
            }
        }
        if let (Some(r), Some(a), Some(b)) = (br, ba, bb) {
            s.remove_item(r); s.add_item(a); s.add_item(b);
            return true;
        }
        false
    }

    fn apply_best_swap21(s: &mut P2State, window_k: usize) -> bool {
        let unused = build_best_unused(s, window_k);
        let used   = build_worst_used(s, window_k);
        let cu = unused.len(); let cs = used.len();

        let n = s.ins.n;
        let q = &s.ins.q;
        let w = &s.ins.w;
        let contrib = &s.contrib;
        let budget_base = s.ins.capacity - s.weight;

        let mut used_pairs: Vec<(usize, usize, i32, Ll)> =
            Vec::with_capacity(cs * cs / 2);
        for x in 0..cs {
            let r1 = used[x].id;
            let r1_base = r1 * n;
            for y in (x+1)..cs {
                let r2 = used[y].id;
                let budget = budget_base + w[r1] + w[r2];
                let lost = contrib[r1] as Ll + contrib[r2] as Ll
                           - q[r1_base + r2] as Ll;
                used_pairs.push((r1, r2, budget, lost));
            }
        }

        let mut best_delta: Ll = 0;
        let mut br1 = None; let mut br2 = None; let mut ba = None;
        for ai in 0..cu {
            let add = unused[ai].id;
            let add_w = w[add];
            let cadd = contrib[add] as Ll;
            let add_base = add * n;
            let mut min_cross1 = Ll::MAX;
            let mut min_cross2 = Ll::MAX;
            for rm_score in &used {
                let cross = q[add_base + rm_score.id] as Ll;
                if cross < min_cross1 {
                    min_cross2 = min_cross1;
                    min_cross1 = cross;
                } else if cross < min_cross2 {
                    min_cross2 = cross;
                }
            }
            let min_cross = min_cross1 + min_cross2;
            for &(r1, r2, budget, lost) in &used_pairs {
                if add_w > budget { continue; }
                let upper_bound = cadd - min_cross - lost;
                if upper_bound <= best_delta { continue; }
                let gained = cadd - q[add_base + r1] as Ll
                             - q[add_base + r2] as Ll;
                let d = gained - lost;
                if d > best_delta {
                    best_delta = d; br1 = Some(r1); br2 = Some(r2); ba = Some(add);
                }
            }
        }
        if let (Some(r1), Some(r2), Some(a)) = (br1, br2, ba) {
            s.remove_item(r1); s.remove_item(r2); s.add_item(a);
            return true;
        }
        false
    }

    fn apply_best_swap22(s: &mut P2State, window_k: usize) -> bool {
        let k      = window_k.min(40);
        let unused = build_best_unused(s, k);
        let used   = build_worst_used(s, k);
        let cu = unused.len(); let cs = used.len();
        if cu < 2 || cs < 2 { return false; }

        let n = s.ins.n;
        let q = &s.ins.q;
        let w = &s.ins.w;
        let contrib = &s.contrib;
        let slack = s.slack();

        let mut used_pairs: Vec<(usize, usize, usize, usize, i32, Ll)> =
            Vec::with_capacity(cs * (cs - 1) / 2);
        for x in 0..cs {
            let r1 = used[x].id;
            let r1_base = r1 * n;
            for y in (x+1)..cs {
                let r2     = used[y].id;
                let budget = slack + w[r1] + w[r2];
                let lost   = contrib[r1] as Ll + contrib[r2] as Ll
                             - q[r1_base + r2] as Ll;
                used_pairs.push((x, y, r1, r2, budget, lost));
            }
        }

        let mut unused_pairs: Vec<(usize, usize, usize, usize, i32, i32, Ll)> =
            Vec::with_capacity(cu * (cu - 1) / 2);
        for a in 0..cu {
            let p  = unused[a].id;
            let wp = w[p];
            let p_base = p * n;
            for b in (a+1)..cu {
                let q2 = unused[b].id;
                let q2_base = q2 * n;
                let pair_w = wp + w[q2];
                let gain = contrib[p] as Ll + contrib[q2] as Ll
                           + q[p_base + q2] as Ll;
                unused_pairs.push((p, q2, p_base, q2_base, wp, pair_w, gain));
            }
        }

        let mut cross_cache = vec![0i64; unused_pairs.len() * cs];
        for (pair_idx, &(_, _, p_base, q2_base, _, _, _)) in unused_pairs.iter().enumerate() {
            let cache_row = &mut cross_cache[pair_idx * cs..(pair_idx + 1) * cs];
            for (used_idx, rm_score) in used.iter().enumerate() {
                let rm = rm_score.id;
                cache_row[used_idx] = q[p_base + rm] as Ll + q[q2_base + rm] as Ll;
            }
        }

        let mut best_delta: Ll = 0;
        let mut br1 = None; let mut br2 = None;
        let mut ba1 = None; let mut ba2 = None;
        for &(r1_idx, r2_idx, r1, r2, budget, lost) in &used_pairs {
            for (pair_idx, &(p, q2, _, _, wp, pair_w, gain)) in unused_pairs.iter().enumerate() {
                if wp >= budget || pair_w > budget { continue; }
                let upper_bound = gain - lost;
                if upper_bound <= best_delta { continue; }
                let cache_row = pair_idx * cs;
                let cross = cross_cache[cache_row + r1_idx]
                          + cross_cache[cache_row + r2_idx];
                let d = upper_bound - cross;
                if d > best_delta {
                    best_delta = d;
                    br1 = Some(r1); br2 = Some(r2);
                    ba1 = Some(p);  ba2 = Some(q2);
                }
            }
        }
        if let (Some(r1), Some(r2), Some(a1), Some(a2)) = (br1, br2, ba1, ba2) {
            s.remove_item(r1); s.remove_item(r2);
            s.add_item(a1);    s.add_item(a2);
            return true;
        }
        false
    }

    fn local_search_vnd(s: &mut P2State, window_k: usize, heavy: bool) {
        loop {
            if apply_best_add1(s)                 { continue; }
            if apply_best_swap11(s, window_k)     { continue; }
            if apply_best_add2(s, window_k)       { continue; }
            if apply_best_swap12(s, window_k / 2) { continue; }
            if apply_best_swap21(s, window_k / 2) { continue; }
            if heavy {
                if apply_best_swap22(s, window_k) { continue; }
                if apply_synergy_tail(s) { continue; }
            }
            break;
        }
        s.save_best();
    }

    fn dp_refinement(s: &mut P2State, core_half: usize) {
        let n = s.ins.n; let cap = s.ins.capacity;
        let mut ord: Vec<ItemScore> = (0..n).map(|i| {
            let w = s.ins.w[i].max(1) as Ll;
            ItemScore { id: i, score: s.contrib[i] as Ll * 1_000_000 / w }
        }).collect();
        ord.sort_unstable_by(|a, b| b.score.cmp(&a.score));

        let mut idx_last = 0usize; let mut idx_first_rej = n; let mut rem = cap;
        for (idx, is) in ord.iter().enumerate() {
            let wt = s.ins.w[is.id];
            if wt <= rem { rem -= wt; idx_last = idx; }
            else if idx_first_rej == n { idx_first_rej = idx; }
        }

        let left  = if idx_first_rej > core_half + 1 { idx_first_rej - core_half - 1 } else { 0 };
        let right = (idx_last + core_half + 1).min(n);
        if left >= right { return; }

        let mut target = vec![0u8; n]; let mut locked_weight = 0i32;
        for i in 0..left {
            let item = ord[i].id;
            if locked_weight + s.ins.w[item] <= cap {
                target[item] = 1; locked_weight += s.ins.w[item];
            }
        }

        let rem_cap = cap - locked_weight;
        if rem_cap > 0 {
            let k = right - left;
            let total_core_w: i32 = (left..right).map(|t| s.ins.w[ord[t].id]).sum();
            let max_w = (rem_cap.min(total_core_w)) as usize;
            if max_w > 0 && k > 0 && max_w <= 2_000_000 {
                let mut lower = 0i64;
                let mut upper = 0i64;
                for t in 0..k {
                    let val = s.contrib[ord[left + t].id] as i64;
                    if val < 0 { lower += val; } else { upper += val; }
                }
                let sentinel32 = lower.checked_sub(upper)
                    .and_then(|v| v.checked_sub(1))
                    .filter(|&v| {
                        lower >= i32::MIN as i64
                            && upper <= i32::MAX as i64
                            && v >= i32::MIN as i64
                            && v <= i32::MAX as i64
                            && v.checked_add(lower).map_or(false, |x| x >= i32::MIN as i64)
                            && v.checked_add(upper).map_or(false, |x| x <= i32::MAX as i64)
                    })
                    .map(|v| v as i32);
                let mut choose = vec![0u8; k * (max_w + 1)];
                let (best_w, _) = if let Some(neg_inf_dp) = sentinel32 {
                    let mut dp = vec![neg_inf_dp; max_w + 1];
                    dp[0] = 0;
                    let mut w_hi = 0usize;
                    for t in 0..k {
                        let item = ord[left + t].id;
                        let wt   = s.ins.w[item] as usize;
                        let val  = s.contrib[item];
                        if wt > max_w { continue; }
                        let new_hi = (w_hi + wt).min(max_w);
                        for w in (wt..=new_hi).rev() {
                            let cand = dp[w - wt] + val;
                            if cand > dp[w] { dp[w] = cand; choose[t*(max_w+1)+w] = 1; }
                        }
                        w_hi = new_hi;
                    }
                    let best_w = (0..=max_w).max_by_key(|&w| dp[w]).unwrap_or(0);
                    (best_w, dp[best_w] as Ll)
                } else {
                    let neg_inf_dp: Ll = NEG_INF / 4;
                    let mut dp = vec![neg_inf_dp; max_w + 1];
                    dp[0] = 0;
                    let mut w_hi = 0usize;
                    for t in 0..k {
                        let item = ord[left + t].id;
                        let wt   = s.ins.w[item] as usize;
                        let val  = s.contrib[item] as Ll;
                        if wt > max_w { continue; }
                        let new_hi = (w_hi + wt).min(max_w);
                        for w in (wt..=new_hi).rev() {
                            let cand = dp[w - wt] + val;
                            if cand > dp[w] { dp[w] = cand; choose[t*(max_w+1)+w] = 1; }
                        }
                        w_hi = new_hi;
                    }
                    let best_w = (0..=max_w).max_by_key(|&w| dp[w]).unwrap_or(0);
                    (best_w, dp[best_w])
                };
                let mut cur_w = best_w;
                for t in (0..k).rev() {
                    let item = ord[left + t].id;
                    let wt   = s.ins.w[item] as usize;
                    if wt <= cur_w && choose[t*(max_w+1)+cur_w] != 0 {
                        target[item] = 1; cur_w -= wt;
                    }
                }
            }
        }

        for i in 0..n {
            if s.sel[i] != 0 && target[i] == 0 { s.remove_item(i); }
        }
        for i in 0..n {
            if s.sel[i] == 0 && target[i] != 0
                && s.weight + s.ins.w[i] <= s.ins.capacity
            { s.add_item(i); }
        }
    }

    fn perturb(s: &mut P2State, rng: &mut Rng, strength: usize, strategy: usize,
               total_interactions: &[Ll]) {
        let n = s.ins.n;
        let mut cand: Vec<ItemScore> = (0..n).filter(|&i| s.sel[i] != 0).map(|i| {
            let score = match strategy {
                0 => s.contrib[i] as Ll,
                1 => -(s.ins.w[i] as Ll),
                2 => { let w = s.ins.w[i].max(1) as Ll; s.contrib[i] as Ll * 1_000_000 / w },
                3 => s.contrib[i] as Ll * 100 - total_interactions[i],
                4 => total_interactions[i] - 2 * s.contrib[i] as Ll,
                _ => (rng.next_u64() & 0x7fff_ffff) as Ll,
            };
            ItemScore { id: i, score }
        }).collect();
        let cnt = cand.len();
        cand.sort_unstable_by(|a, b| a.score.cmp(&b.score));
        let remove_n = strength.min(cnt);
        for i in 0..remove_n {
            s.remove_item(cand[i].id);
        }
    }

    fn reconstruction_score(s: &P2State, id: usize, strategy: usize,
                            total_interactions: &[Ll]) -> Ll {
        let w = s.ins.w[id].max(1) as Ll;
        match strategy {
            0 => s.contrib[id] as Ll,
            1 => s.contrib[id] as Ll * 1_000_000 / w,
            2 => total_interactions[id] + s.contrib[id] as Ll,
            3 => s.contrib[id] as Ll * 1_000_000 / w + total_interactions[id] / 10,
            _ => s.contrib[id] as Ll + total_interactions[id] / 20 - s.ins.w[id] as Ll,
        }
    }

    fn reconstruction_pool(s: &P2State, strategy: usize,
                           total_interactions: &[Ll], max_k: usize) -> Vec<ItemScore> {
        let mut pool: Vec<ItemScore> = (0..s.ins.n)
            .filter(|&i| s.sel[i] == 0 && s.ins.w[i] <= s.slack())
            .map(|i| ItemScore {
                id: i,
                score: reconstruction_score(s, i, strategy, total_interactions),
            })
            .collect();
        if pool.len() > max_k {
            pool.select_nth_unstable_by(max_k, |a, b| b.score.cmp(&a.score));
            pool.truncate(max_k);
        }
        pool
    }

    fn greedy_reconstruct(s: &mut P2State, strategy: usize,
                          total_interactions: &[Ll]) {
        if s.ins.n > 4000 {
            let mut cand: Vec<ItemScore> = (0..s.ins.n).filter(|&i| s.sel[i] == 0)
                .map(|i| ItemScore {
                    id: i,
                    score: reconstruction_score(s, i, strategy, total_interactions),
                })
                .collect();
            cand.sort_unstable_by(|a, b| b.score.cmp(&a.score));
            for is in &cand {
                if s.weight + s.ins.w[is.id] <= s.ins.capacity {
                    s.add_item(is.id);
                }
            }
            return;
        }

        const POOL_K: usize = 48;
        let mut pool = reconstruction_pool(s, strategy, total_interactions, POOL_K);
        let mut additions_since_refresh = 0usize;

        loop {
            let mut best: Option<(usize, Ll)> = None;
            for is in &pool {
                let score = reconstruction_score(s, is.id, strategy, total_interactions);
                if score > 0 && best.map_or(true, |(_, best_score)| score > best_score) {
                    best = Some((is.id, score));
                }
            }

            let Some((item, _)) = best else {
                if pool.is_empty() { break; }
                pool = reconstruction_pool(s, strategy, total_interactions, POOL_K);
                if pool.is_empty() { break; }
                let has_positive = pool.iter().any(|is| {
                    reconstruction_score(s, is.id, strategy, total_interactions) > 0
                });
                if !has_positive { break; }
                continue;
            };

            s.add_item(item);
            pool.retain(|is| is.id != item);
            additions_since_refresh += 1;

            if additions_since_refresh >= 8 {
                pool = reconstruction_pool(s, strategy, total_interactions, POOL_K);
                additions_since_refresh = 0;
            }
        }
    }

    fn solve_hybrid(s:          &mut P2State,
                    rng:        &mut Rng,
                    ils_rounds: usize,
                    window_k:   usize,
                    core_half:  usize) {
        let total_interactions = compute_total_interactions(&s.ins);
        let mut current_certified = false;
        let mut best_certified = false;
        let mut before_dp = s.sel.clone();
        let mut certified_refinement =
            |s: &mut P2State, half: usize, certified: &mut bool| {
                before_dp.copy_from_slice(&s.sel);
                dp_refinement(s, half);
                if before_dp.as_slice() != s.sel.as_slice() {
                    *certified = false;
                }
            };

        let best_before = s.best_value;
        certified_refinement(s, core_half, &mut current_certified);
        if !current_certified {
            local_search_vnd(s, window_k, true);
        }
        s.save_best();
        if s.best_value > best_before { best_certified = true; }
        s.restore_best();
        current_certified = best_certified;

        let active_ils_rounds = if ils_rounds > 0 && s.count >= 6 {
            ils_rounds
        } else {
            0
        };
        let mut round = 0usize;
        let mut stall = 0usize;

        while round < active_ils_rounds {
            let old_best = s.best_value;

            certified_refinement(s, core_half, &mut current_certified);
            if !current_certified {
                local_search_vnd(s, window_k, true);
            }
            s.save_best();
            if s.best_value > old_best { best_certified = true; }

            if s.best_value > old_best { stall = 0; } else { stall += 1; }

            s.restore_best();

            let strength = {
                let base = 1 + stall / 3 + round / 25;
                let cap  = (s.count as usize) / 4;
                base.min(cap).max(1)
            };

            let best_before = s.best_value;
            perturb(s, rng, strength, round % 6, &total_interactions);
            greedy_reconstruct(s, round % 5, &total_interactions);
            current_certified = false;

            certified_refinement(s, core_half, &mut current_certified);
            if !current_certified {
                local_search_vnd(s, window_k, true);
                current_certified = true;
            }
            s.save_best();
            if s.best_value > best_before { best_certified = true; }
            
            if stall >= 40{
                s.restore_best();
                let best_before = s.best_value;
                let ss = ((s.count as usize) / 6).max(3);
                perturb(s, rng, ss, 5, &total_interactions);
                let rs = rng.next_int(5);
                greedy_reconstruct(s, rs, &total_interactions);
                current_certified = false;
                certified_refinement(s, core_half, &mut current_certified);
                if !current_certified {
                    local_search_vnd(s, window_k, true);
                }
                s.save_best();
                if s.best_value > best_before { best_certified = true; }
                s.restore_best();
                current_certified = best_certified;
                stall = 0;
            }

            round += 1;
        }

        s.restore_best();
        current_certified = best_certified;

        for _ in 0..8 {
            let before = s.value;
            let best_before = s.best_value;
            certified_refinement(s, core_half * 2, &mut current_certified);
            if !current_certified {
                local_search_vnd(s, window_k, true);
            }
            s.save_best();
            if s.best_value > best_before { best_certified = true; }
            s.restore_best();
            current_certified = best_certified;
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
            let sum_w: u64 = challenge.weights.iter().map(|&w| w as u64).sum();
            let budget_pct = if sum_w > 0 { ((challenge.max_weight as u64) * 100 / sum_w) as u32 } else { 10 };
        
            let hp = Hparams::from_map(hyperparameters, n, budget_pct);

            let raw_seed = u64::from_le_bytes(
                challenge.seed[..8].try_into().unwrap_or([0u8; 8])
            );
            let seed = raw_seed.wrapping_add(0x9E3779B97F4A7C15);
            let mut rng = Rng::new(seed);

            let inst   = challenge_to_p1(challenge);
            let qr     = run_bp_algorithm(&inst, hp.n_lambda_values);
            let budget = inst.budgets[0];

            let p2    = bridge_p1_to_p2(&inst, budget);
            let mut s = P2State::new(p2);
            bridge_load_solution(&mut s, &qr.results[0]);

            solve_hybrid(&mut s, &mut rng, hp.ils_rounds, hp.window_k, hp.core_half_dp);

            let chk_val = P2State::eval_selected(&s.ins, &s.best_sel);
            let chk_wt  = P2State::weight_selected(&s.ins, &s.best_sel);
            if chk_val != s.best_value  { s.best_value  = chk_val; }
            if chk_wt  != s.best_weight { s.best_weight = chk_wt;  }
            if chk_wt > budget { return Ok(None); }

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