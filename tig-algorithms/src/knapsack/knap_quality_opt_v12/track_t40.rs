use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::knapsack::{Challenge, Solution};

mod inner {
    use anyhow::Result;
    use serde::{Deserialize, Serialize};
    use serde_json::{Map, Value};
    use tig_challenges::knapsack::*;

// Shared types

    pub type Ll = i64;

// Hyperparameter plumbing

    #[derive(Serialize, Deserialize)]
    pub struct Hyperparameters {
        pub n_lambda_values: Option<usize>,
        pub window_k:        Option<usize>,
        pub ils_rounds:      Option<usize>,
        pub core_half_dp:    Option<usize>,
    }

    struct Hparams {
        n_lambda_values:      usize,
        n_random_starts:      usize,
        n_crossover_gen:      usize,
        sa_rounds:            usize,
        sa_iter:              usize,
        n_sa_members:         usize,
        ils_rounds:           usize,
        ils_restart_interval: usize,
        perturb_base_frac:    usize,
        perturb_max_frac:     usize,
        ils_vnd_level:        usize,
        bounded_2_2_k:        usize,
        n_full_restarts:      usize,
        use_hub_pair:         bool,
        use_heavy_polish:     bool,
        window_k:             usize,
        core_half_dp:         usize,
    }

    impl Hparams {
        fn for_size(n: usize, budget: u32) -> Self {
            if n <= 1200 {
                if budget <= 5 {
                    Self {
                        n_lambda_values: 1600,
                        n_random_starts: 4, n_crossover_gen: 12, sa_rounds: 0,
                        sa_iter: 0, n_sa_members: 0, ils_rounds: 350,
                        ils_restart_interval: 12, perturb_base_frac: 3,
                        perturb_max_frac: 5, ils_vnd_level: 0, bounded_2_2_k: 10,
                        n_full_restarts: 1, use_hub_pair: true,
                        use_heavy_polish: true, window_k: n, core_half_dp: 60,
                    }
                } else if budget <= 10 {
                    Self {
                        n_lambda_values: 1600,
                        n_random_starts: 4, n_crossover_gen: 12, sa_rounds: 30,
                        sa_iter: 300, n_sa_members: 3, ils_rounds: 300,
                        ils_restart_interval: 12, perturb_base_frac: 4,
                        perturb_max_frac: 5, ils_vnd_level: 0, bounded_2_2_k: 10,
                        n_full_restarts: 1, use_hub_pair: true,
                        use_heavy_polish: true, window_k: n, core_half_dp: 60,
                    }
                } else {
                    Self {
                        n_lambda_values: 1600,
                        n_random_starts: 3, n_crossover_gen: 26, sa_rounds: 18,
                        sa_iter: 440, n_sa_members: 8, ils_rounds: 50,
                        ils_restart_interval: 1, perturb_base_frac: 100,
                        perturb_max_frac: 28, ils_vnd_level: 0, bounded_2_2_k: 0,
                        n_full_restarts: 12, use_hub_pair: true,
                        use_heavy_polish: false, window_k: 60, core_half_dp: 12,
                    }
                }
            } else {
                Self {
                    n_lambda_values: 1600,
                    n_random_starts: 4, n_crossover_gen: 0, sa_rounds: 0,
                    sa_iter: 0, n_sa_members: 0, ils_rounds: 120,
                    ils_restart_interval: 12, perturb_base_frac: 8,
                    perturb_max_frac: 5, ils_vnd_level: 0, bounded_2_2_k: 0,
                    n_full_restarts: 1, use_hub_pair: false,
                    use_heavy_polish: false, window_k: 200, core_half_dp: 50,
                }
            }
        }

        fn from_map(h: &Option<Map<String, Value>>, n: usize, budget: u32) -> Self {
            let mut p = Self::for_size(n, budget);
            if let Some(m) = h {
                if let Some(v) = m.get("n_lambda_values").and_then(|v| v.as_u64())  { p.n_lambda_values      = v as usize; }
                if let Some(v) = m.get("n_random_starts").and_then(|v| v.as_u64())  { p.n_random_starts      = v as usize; }
                if let Some(v) = m.get("n_crossover_gen").and_then(|v| v.as_u64())  { p.n_crossover_gen      = v as usize; }
                if let Some(v) = m.get("sa_rounds").and_then(|v| v.as_u64())        { p.sa_rounds            = v as usize; }
                if let Some(v) = m.get("sa_iter").and_then(|v| v.as_u64())          { p.sa_iter              = v as usize; }
                if let Some(v) = m.get("n_sa_members").and_then(|v| v.as_u64())     { p.n_sa_members         = v as usize; }
                if let Some(v) = m.get("ils_rounds").and_then(|v| v.as_u64())       { p.ils_rounds           = v as usize; }
                if let Some(v) = m.get("ils_restart_interval").and_then(|v| v.as_u64()) { p.ils_restart_interval = v as usize; }
                if let Some(v) = m.get("perturb_base_frac").and_then(|v| v.as_u64()){ p.perturb_base_frac    = v as usize; }
                if let Some(v) = m.get("perturb_max_frac").and_then(|v| v.as_u64()) { p.perturb_max_frac     = v as usize; }
                if let Some(v) = m.get("ils_vnd_level").and_then(|v| v.as_u64())    { p.ils_vnd_level        = v as usize; }
                if let Some(v) = m.get("bounded_2_2_k").and_then(|v| v.as_u64())    { p.bounded_2_2_k        = v as usize; }
                if let Some(v) = m.get("n_full_restarts").and_then(|v| v.as_u64())  { p.n_full_restarts      = v as usize; }
                if let Some(v) = m.get("window_k").and_then(|v| v.as_u64())         { p.window_k             = v as usize; }
                if let Some(v) = m.get("core_half_dp").and_then(|v| v.as_u64())     { p.core_half_dp         = v as usize; }
            }
            p
        }
    }

// Phase-1 QKBP instance

    #[derive(Debug, Clone)]
    pub struct Edge { pub i: usize, pub j: usize, pub value: f64 }

    #[derive(Debug)]
    pub struct Phase1Problem {
        pub n_items:   usize,
        pub n_edges:   usize,
        pub edges:     Vec<Edge>,
        pub weights:   Vec<i32>,
        pub n_budgets: usize,
        pub budgets:   Vec<i32>,
    }

    #[derive(Debug)]
    pub struct BudgetOutcome {
        pub budget:         i32,
        pub ofv:            f64,
        pub selected_items: Vec<usize>,
    }

    #[derive(Debug)]
    pub struct QkpOutcome { pub results: Vec<BudgetOutcome> }

    pub fn build_phase1(challenge: &Challenge) -> Phase1Problem {
        let n = challenge.num_items;
        let mut edges: Vec<Edge> = Vec::new();
        for i in 0..n {
            edges.push(Edge { i, j: i, value: challenge.values[i] as f64 });
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let v = challenge.interaction_values[i][j];
                if v != 0 { edges.push(Edge { i, j, value: v as f64 }); }
            }
        }
        let weights: Vec<i32> = challenge.weights.iter().map(|&w| w as i32).collect();
        let budgets: Vec<i32> = vec![challenge.max_weight as i32];
        Phase1Problem { n_items: n, n_edges: edges.len(), edges, weights, n_budgets: 1, budgets }
    }

// Utility matrix and objective value

    pub struct PairUtil { pub n: usize, pub data: Vec<f64>, pub linear: Vec<f64> }
    impl PairUtil {
        #[inline] pub fn get(&self, r: usize, c: usize) -> f64 { self.data[r * self.n + c] }
        #[inline] pub fn set(&mut self, r: usize, c: usize, v: f64) { self.data[r * self.n + c] = v; }
    }

    pub fn build_utility_matrix(n: usize, edges: &[Edge]) -> PairUtil {
        let mut data   = vec![0.0f64; n * n];
        let mut linear = vec![0.0f64; n];
        for e in edges {
            if e.i == e.j { linear[e.i] += e.value; }
            else { data[e.i * n + e.j] = e.value; data[e.j * n + e.i] = e.value; }
        }
        PairUtil { n, data, linear }
    }

    pub fn compute_ofv(sel: &[usize], n: usize, edges: &[Edge]) -> f64 {
        let um = build_utility_matrix(n, edges);
        let mut ofv = 0.0f64;
        for &k in sel { ofv += um.linear[k]; }
        for &a in sel { for &b in sel { ofv += um.get(a, b) / 2.0; } }
        ofv
    }

// Parametric max-flow min-cut

    mod hpf {
        use super::*;

        pub struct HpfArc {
            pub from: *mut HpfNode, pub to: *mut HpfNode,
            pub flow: f32, pub capacity: f32,
            pub direction: i32, pub capacities: Vec<f32>,
        }

        pub struct HpfNode {
            pub wt: f32, pub cst: f32, pub visited: i32,
            pub num_adjacent: i32, pub number: i32, pub label: i32,
            pub excess: f32,
            pub parent: *mut HpfNode, pub child_list: *mut HpfNode,
            pub next_scan: *mut HpfNode, pub num_out_of_tree: i32,
            pub out_of_tree: Vec<*mut HpfArc>, pub next_arc: i32,
            pub arc_to_parent: *mut HpfArc,
            pub next: *mut HpfNode, pub prev: *mut HpfNode,
            pub breakpoint: i32,
        }

        impl HpfNode {
            pub fn zeroed(num_params: i32) -> Self {
                HpfNode {
                    wt: 0.0, cst: 1.0, visited: 0, num_adjacent: 0,
                    number: 0, label: 0, excess: 0.0,
                    parent: std::ptr::null_mut(), child_list: std::ptr::null_mut(),
                    next_scan: std::ptr::null_mut(), num_out_of_tree: 0,
                    out_of_tree: Vec::new(), next_arc: 0,
                    arc_to_parent: std::ptr::null_mut(),
                    next: std::ptr::null_mut(), prev: std::ptr::null_mut(),
                    breakpoint: num_params + 1,
                }
            }
        }

        pub struct HpfRoot { pub start: *mut HpfNode, pub end: *mut HpfNode }

        pub struct HpfState {
            pub num_nodes: i32, pub num_arcs: i32,
            pub source: i32, pub sink: i32, pub num_params: i32,
            pub highest_strong_label: i32,
            pub adjacency_list: Vec<HpfNode>,
            pub strong_roots: Vec<HpfRoot>,
            pub label_count: Vec<i32>,
            pub arc_list: Vec<HpfArc>,
        }

        unsafe fn init_root(num_params: i32) -> HpfRoot {
            let start = Box::into_raw(Box::new(HpfNode::zeroed(num_params)));
            let end   = Box::into_raw(Box::new(HpfNode::zeroed(num_params)));
            (*start).next = end; (*end).prev = start;
            HpfRoot { start, end }
        }

        unsafe fn free_root(root: &HpfRoot) {
            drop(Box::from_raw(root.start));
            drop(Box::from_raw(root.end));
        }

        unsafe fn add_to_strong_bucket(new_root: *mut HpfNode, root_end: *mut HpfNode) {
            (*new_root).next = root_end; (*new_root).prev = (*root_end).prev;
            (*root_end).prev = new_root; (*(*new_root).prev).next = new_root;
        }

        unsafe fn lift_all(s: *mut HpfState, root_node: *mut HpfNode, theparam: i32) {
            let mut current = root_node;
            (*current).next_scan = (*current).child_list;
            (*s).label_count[(*current).label as usize] -= 1;
            (*current).label = (*s).num_nodes; (*current).breakpoint = theparam + 1;
            loop {
                while !(*current).next_scan.is_null() {
                    let temp = (*current).next_scan;
                    (*current).next_scan = (*(*current).next_scan).next;
                    current = temp;
                    (*current).next_scan = (*current).child_list;
                    (*s).label_count[(*current).label as usize] -= 1;
                    (*current).label = (*s).num_nodes; (*current).breakpoint = theparam + 1;
                }
                if (*current).parent.is_null() { break; }
                current = (*current).parent;
            }
        }

        unsafe fn add_relationship(new_parent: *mut HpfNode, child: *mut HpfNode) {
            (*child).parent = new_parent;
            (*child).next = (*new_parent).child_list;
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
            (*child).next = std::ptr::null_mut();
        }

        unsafe fn hpf_merge(parent: *mut HpfNode, child: *mut HpfNode, new_arc: *mut HpfArc) {
            let mut current = child; let mut new_parent = parent; let mut new_arc = new_arc;
            while !(*current).parent.is_null() {
                let old_arc = (*current).arc_to_parent;
                (*current).arc_to_parent = new_arc;
                let old_parent = (*current).parent;
                break_relationship(old_parent, current);
                add_relationship(new_parent, current);
                new_parent = current; current = old_parent;
                new_arc = old_arc; (*new_arc).direction = 1 - (*new_arc).direction;
            }
            (*current).arc_to_parent = new_arc;
            add_relationship(new_parent, current);
        }

        unsafe fn push_upward(s: *mut HpfState, arc: *mut HpfArc,
                               child: *mut HpfNode, parent: *mut HpfNode, res_cap: f32) {
            if res_cap >= (*child).excess {
                (*parent).excess += (*child).excess; (*arc).flow += (*child).excess;
                (*child).excess = 0.0; return;
            }
            (*arc).direction = 0; (*parent).excess += res_cap;
            (*child).excess -= res_cap; (*arc).flow = (*arc).capacity;
            (*parent).out_of_tree.push(arc); (*parent).num_out_of_tree += 1;
            break_relationship(parent, child);
            let lbl = (*child).label as usize;
            add_to_strong_bucket(child, (*s).strong_roots[lbl].end);
        }

        unsafe fn push_downward(s: *mut HpfState, arc: *mut HpfArc,
                                 child: *mut HpfNode, parent: *mut HpfNode, flow: f32) {
            if flow >= (*child).excess {
                (*parent).excess += (*child).excess; (*arc).flow -= (*child).excess;
                (*child).excess = 0.0; return;
            }
            (*arc).direction = 1; (*child).excess -= flow; (*parent).excess += flow;
            (*arc).flow = 0.0; (*parent).out_of_tree.push(arc); (*parent).num_out_of_tree += 1;
            break_relationship(parent, child);
            let lbl = (*child).label as usize;
            add_to_strong_bucket(child, (*s).strong_roots[lbl].end);
        }

        unsafe fn push_excess(s: *mut HpfState, strong_root: *mut HpfNode) {
            let mut current = strong_root;
            while (*current).excess > 0.0 && !(*current).parent.is_null() {
                let parent = (*current).parent; let arc = (*current).arc_to_parent;
                if (*arc).direction != 0 {
                    push_upward(s, arc, current, parent, (*arc).capacity - (*arc).flow);
                } else { push_downward(s, arc, current, parent, (*arc).flow); }
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
            let size = (*strong_node).num_out_of_tree as usize;
            let mut i = (*strong_node).next_arc as usize;
            while i < size {
                let out = (*strong_node).out_of_tree[i];
                if (*(*out).to).label == target {
                    (*strong_node).next_arc = i as i32; *weak_node = (*out).to;
                    let last = (*strong_node).num_out_of_tree as usize - 1;
                    (*strong_node).out_of_tree[i] = (*strong_node).out_of_tree[last];
                    (*strong_node).out_of_tree.pop(); (*strong_node).num_out_of_tree -= 1;
                    return out;
                } else if (*(*out).from).label == target {
                    (*strong_node).next_arc = i as i32; *weak_node = (*out).from;
                    let last = (*strong_node).num_out_of_tree as usize - 1;
                    (*strong_node).out_of_tree[i] = (*strong_node).out_of_tree[last];
                    (*strong_node).out_of_tree.pop(); (*strong_node).num_out_of_tree -= 1;
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
            if !out.is_null() { hpf_merge(weak_node, strong_node, out); push_excess(s, strong_root); return; }
            check_children(s, strong_root);
            loop {
                while !(*strong_node).next_scan.is_null() {
                    let temp = (*strong_node).next_scan;
                    (*strong_node).next_scan = (*(*strong_node).next_scan).next;
                    strong_node = temp;
                    (*strong_node).next_scan = (*strong_node).child_list;
                    let out = find_weak_node(s, strong_node, &mut weak_node);
                    if !out.is_null() { hpf_merge(weak_node, strong_node, out); push_excess(s, strong_root); return; }
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

        unsafe fn get_highest_strong_root(s: *mut HpfState, theparam: i32) -> *mut HpfNode {
            let mut i = (*s).highest_strong_label;
            while i > 0 {
                if (*(*s).strong_roots[i as usize].start).next != (*s).strong_roots[i as usize].end {
                    (*s).highest_strong_label = i;
                    if (*s).label_count[(i - 1) as usize] > 0 {
                        let sr = (*(*s).strong_roots[i as usize].start).next;
                        (*(*sr).next).prev = (*sr).prev; (*(*sr).prev).next = (*sr).next;
                        (*sr).next = std::ptr::null_mut(); return sr;
                    }
                    while (*(*s).strong_roots[i as usize].start).next != (*s).strong_roots[i as usize].end {
                        let sr = (*(*s).strong_roots[i as usize].start).next;
                        (*(*sr).next).prev = (*sr).prev; (*(*sr).prev).next = (*sr).next;
                        lift_all(s, sr, theparam);
                    }
                }
                i -= 1;
            }
            if (*(*s).strong_roots[0].start).next == (*s).strong_roots[0].end { return std::ptr::null_mut(); }
            while (*(*s).strong_roots[0].start).next != (*s).strong_roots[0].end {
                let sr = (*(*s).strong_roots[0].start).next;
                (*(*sr).next).prev = (*sr).prev; (*(*sr).prev).next = (*sr).next;
                (*sr).label = 1; (*s).label_count[0] -= 1; (*s).label_count[1] += 1;
                let lbl = (*sr).label as usize;
                add_to_strong_bucket(sr, (*s).strong_roots[lbl].end);
            }
            (*s).highest_strong_label = 1;
            let sr = (*(*s).strong_roots[1].start).next;
            (*(*sr).next).prev = (*sr).prev; (*(*sr).prev).next = (*sr).next;
            (*sr).next = std::ptr::null_mut(); sr
        }

        unsafe fn update_capacities(s: *mut HpfState, theparam: i32) {
            let src_idx = ((*s).source - 1) as usize;
            let size = (*s).adjacency_list[src_idx].num_out_of_tree as usize;
            for i in 0..size {
                let arc = (*s).adjacency_list[src_idx].out_of_tree[i];
                let delta = (*arc).capacities[theparam as usize] - (*arc).capacity;
                if delta < 0.0 { return; }
                (*arc).capacity += delta; (*arc).flow += delta; (*(*arc).to).excess += delta;
                if (*(*arc).to).label < (*s).num_nodes && (*(*arc).to).excess > 0.0 { push_excess(s, (*arc).to); }
            }
            let snk_idx = ((*s).sink - 1) as usize;
            let size = (*s).adjacency_list[snk_idx].num_out_of_tree as usize;
            for i in 0..size {
                let arc = (*s).adjacency_list[snk_idx].out_of_tree[i];
                let delta = (*arc).capacities[theparam as usize] - (*arc).capacity;
                if delta > 0.0 { return; }
                (*arc).capacity += delta; (*arc).flow += delta; (*(*arc).from).excess -= delta;
                if (*(*arc).from).label < (*s).num_nodes && (*(*arc).from).excess > 0.0 { push_excess(s, (*arc).from); }
            }
            (*s).highest_strong_label = (*s).num_nodes - 1;
        }

        unsafe fn simple_initialization(s: *mut HpfState) {
            let src_idx = ((*s).source - 1) as usize;
            let snk_idx = ((*s).sink - 1) as usize;
            let size = (*s).adjacency_list[src_idx].num_out_of_tree as usize;
            for i in 0..size {
                let arc = (*s).adjacency_list[src_idx].out_of_tree[i];
                (*arc).flow = (*arc).capacity; (*(*arc).to).excess += (*arc).capacity;
            }
            let size = (*s).adjacency_list[snk_idx].num_out_of_tree as usize;
            for i in 0..size {
                let arc = (*s).adjacency_list[snk_idx].out_of_tree[i];
                (*arc).flow = (*arc).capacity; (*(*arc).from).excess -= (*arc).capacity;
            }
            (*s).adjacency_list[src_idx].excess = 0.0;
            (*s).adjacency_list[snk_idx].excess = 0.0;
            for i in 0..(*s).num_nodes as usize {
                if (*s).adjacency_list[i].excess > 0.0 {
                    (*s).adjacency_list[i].label = 1; (*s).label_count[1] += 1;
                    let nd = &mut (*s).adjacency_list[i] as *mut HpfNode;
                    let end = (*s).strong_roots[1].end;
                    add_to_strong_bucket(nd, end);
                }
            }
            (*s).adjacency_list[src_idx].label = (*s).num_nodes;
            (*s).adjacency_list[src_idx].breakpoint = 0;
            (*s).adjacency_list[snk_idx].label = 0;
            (*s).adjacency_list[snk_idx].breakpoint = (*s).num_params + 2;
            (*s).label_count[0] = ((*s).num_nodes - 2) - (*s).label_count[1];
        }

        unsafe fn pseudoflow_phase1(s: *mut HpfState) {
            let mut theparam = 0i32;
            loop { let sr = get_highest_strong_root(s, theparam); if sr.is_null() { break; } process_root(s, sr); }
            theparam = 1;
            while theparam < (*s).num_params {
                update_capacities(s, theparam);
                loop { let sr = get_highest_strong_root(s, theparam); if sr.is_null() { break; } process_root(s, sr); }
                theparam += 1;
            }
        }

        pub struct BreakpointSets { pub sets: Vec<(i32, Vec<usize>)> }

        pub fn get_breakpoints(inst: &Phase1Problem, n_lambda_values: usize) -> BreakpointSets {
            let n_items    = inst.n_items;
            let n_edges    = inst.n_edges;
            let num_nodes  = (n_items + 2) as i32;
            let num_arcs   = (n_edges + 2 * n_items) as i32;
            let num_params = n_lambda_values as i32;

            let mut adjacency_list: Vec<HpfNode> = (0..num_nodes as usize)
                .map(|i| { let mut nd = HpfNode::zeroed(num_params); nd.number = (i + 1) as i32; nd.cst = 1.0; nd })
                .collect();
            for i in 0..n_items { adjacency_list[i + 2].cst = inst.weights[i] as f32; }

            let mut arc_list: Vec<HpfArc> = Vec::with_capacity(num_arcs as usize);
            for _ in 0..num_arcs as usize {
                arc_list.push(HpfArc { from: std::ptr::null_mut(), to: std::ptr::null_mut(),
                    flow: 0.0, capacity: 0.0, direction: 1, capacities: Vec::new() });
            }

            let mut first = 0usize;
            for k in 0..n_edges {
                let from_id = inst.edges[k].i; let to_id = inst.edges[k].j;
                let cap = inst.edges[k].value as f32;
                arc_list[first].capacities = vec![cap]; arc_list[first].capacity = cap;
                arc_list[first].direction = 1;
                adjacency_list[from_id + 2].wt += cap; adjacency_list[from_id + 2].num_adjacent += 1;
                adjacency_list[to_id + 2].num_adjacent += 1; first += 1;
            }

            let mut max_degree_ratio = 0.0f32;
            for i in 2..num_nodes as usize {
                let ratio = adjacency_list[i].wt / adjacency_list[i].cst;
                if ratio > max_degree_ratio { max_degree_ratio = ratio; }
            }
            let step: f32 = if num_params > 0 { max_degree_ratio / num_params as f32 } else { 0.0 };
            let params: Vec<f32> = (0..num_params as usize).map(|i| max_degree_ratio - i as f32 * step).collect();

            for i in 2..num_nodes as usize {
                let wt = adjacency_list[i].wt; let cst = adjacency_list[i].cst;
                let src_caps: Vec<f32> = params.iter().map(|&p| { let v = wt - p * cst; if v > 0.0 { v } else { 0.0 } }).collect();
                arc_list[first].capacity = src_caps.first().copied().unwrap_or(0.0);
                arc_list[first].capacities = src_caps; arc_list[first].direction = 1;
                adjacency_list[0].num_adjacent += 1; adjacency_list[i].num_adjacent += 1; first += 1;
                let snk_caps: Vec<f32> = params.iter().map(|&p| { let v = p * cst - wt; if v > 0.0 { v } else { 0.0 } }).collect();
                arc_list[first].capacity = snk_caps.first().copied().unwrap_or(0.0);
                arc_list[first].capacities = snk_caps; arc_list[first].direction = 1;
                adjacency_list[i].num_adjacent += 1; adjacency_list[1].num_adjacent += 1; first += 1;
            }

            let strong_roots: Vec<HpfRoot> = unsafe { (0..num_nodes as usize).map(|_| init_root(num_params)).collect() };
            let label_count = vec![0i32; num_nodes as usize];
            let mut state = HpfState {
                num_nodes, num_arcs, source: 1, sink: 2, num_params, highest_strong_label: 1,
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
                    (*s).arc_list[first].to   = &mut (*s).adjacency_list[i]; first += 1;
                    (*s).arc_list[first].from = &mut (*s).adjacency_list[i];
                    (*s).arc_list[first].to   = &mut (*s).adjacency_list[1]; first += 1;
                }
                for i in 0..num_arcs as usize {
                    let to_num   = (*(*s).arc_list[i].to).number;
                    let from_num = (*(*s).arc_list[i].from).number;
                    let cap      = (*s).arc_list[i].capacity;
                    let source   = (*s).source; let sink = (*s).sink;
                    if source == to_num || sink == from_num || from_num == to_num { continue; }
                    if source == from_num && to_num == sink { (*s).arc_list[i].flow = cap; }
                    else if from_num == source {
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

                let mut pos_items: Vec<(i32, usize)> = Vec::new();
                for i in 0..num_nodes as usize {
                    let node_num = (*s).adjacency_list[i].number;
                    if node_num == 1 || node_num == 2 { continue; }
                    pos_items.push(((*s).adjacency_list[i].breakpoint, (node_num - 3) as usize));
                }
                let mut positions: Vec<i32> = pos_items.iter().map(|&(p, _)| p).collect();
                positions.sort_unstable(); positions.dedup();
                let sets: Vec<(i32, Vec<usize>)> = positions.iter().map(|&pos| {
                    let nodes: Vec<usize> = pos_items.iter().filter(|&&(p, _)| p == pos).map(|&(_, item)| item).collect();
                    (pos, nodes)
                }).collect();
                for i in 0..num_nodes as usize { free_root(&(*s).strong_roots[i]); }
                BreakpointSets { sets }
            }
        }
    }

    pub use hpf::get_breakpoints;

// Greedy passes, left and right

    pub type IntArray = Vec<usize>;
    pub type DblArray = Vec<f64>;

    #[inline] pub fn ia_contains(a: &IntArray, v: usize) -> bool { a.contains(&v) }

    struct WarmStartLeft {
        valid: bool, candidate_nodes: IntArray, candidate_contribs: DblArray,
        current_total_weight: f64, left_nodes: IntArray,
    }
    impl WarmStartLeft {
        fn new() -> Self { WarmStartLeft { valid: false, candidate_nodes: Vec::new(), candidate_contribs: Vec::new(), current_total_weight: -1.0, left_nodes: Vec::new() } }
        fn reset(&mut self) { self.valid = false; self.candidate_nodes.clear(); self.candidate_contribs.clear(); self.left_nodes.clear(); self.current_total_weight = -1.0; }
    }

    struct WarmStartRight {
        valid: bool, candidate_nodes: IntArray, candidate_contribs: DblArray, current_total_weight: f64,
    }
    impl WarmStartRight {
        fn new() -> Self { WarmStartRight { valid: false, candidate_nodes: Vec::new(), candidate_contribs: Vec::new(), current_total_weight: 0.0 } }
        fn reset(&mut self) { self.valid = false; self.candidate_nodes.clear(); self.candidate_contribs.clear(); self.current_total_weight = 0.0; }
    }

    fn get_initial_node(right_nodes: &IntArray, um: &PairUtil, weights: &[i32], budget: i32) -> Option<usize> {
        let mut best: Option<usize> = None; let mut best_val = f64::NEG_INFINITY;
        for &nd in right_nodes {
            if weights[nd] > budget { continue; }
            let util: f64 = right_nodes.iter().map(|&m| um.get(nd, m)).sum::<f64>() / weights[nd] as f64;
            if util > best_val { best_val = util; best = Some(nd); }
        }
        best
    }

    fn run_greedy_left(um: &PairUtil, n_nodes: usize, mut left: IntArray, right_nodes: &IntArray,
                       budget: i32, beta: f64, weights: &[i32], mut ws: Option<&mut WarmStartLeft>) -> IntArray {
        if left.is_empty() {
            if let Some(nd) = get_initial_node(right_nodes, um, weights, budget) { left.push(nd); }
        }
        let mut cur_w: f64 = match &ws {
            Some(w) if w.valid && w.current_total_weight >= 0.0 => w.current_total_weight,
            _ => left.iter().map(|&k| weights[k] as f64).sum(),
        };
        let ws_valid = ws.as_ref().map(|w| w.valid).unwrap_or(false);
        let ws_cands = ws.as_ref().map(|w| !w.candidate_nodes.is_empty()).unwrap_or(false);
        let mut cand_nodes: IntArray = Vec::new(); let mut cand_contribs: DblArray = Vec::new();
        let mut update_flag;
        if ws_valid && ws_cands {
            let w = ws.as_ref().unwrap(); let mut all_fit = true; let rem = budget as f64 - cur_w;
            for (idx, &nd) in w.candidate_nodes.iter().enumerate() {
                if weights[nd] as f64 <= rem { cand_nodes.push(nd); cand_contribs.push(w.candidate_contribs[idx]); }
                else { all_fit = false; }
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
                let mut contrib: f64 = left.iter().map(|&m| (1.0 + beta) * um.get(nd, m)).sum();
                contrib += um.linear[nd];
                if beta != 0.0 {
                    for &m in &cand_nodes { contrib -= beta * um.get(nd, m); }
                    for v in 0..n_nodes { if !ia_contains(right_nodes, v) { contrib -= beta * um.get(nd, v); } }
                }
                contrib /= weights[nd] as f64; cand_contribs.push(contrib);
            }
        }
        loop {
            if cand_nodes.is_empty() { break; }
            let best_idx = cand_contribs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
            let best_node = cand_nodes[best_idx];
            left.push(best_node); cur_w += weights[best_node] as f64;
            let mut new_cands: IntArray = Vec::new(); let mut new_contribs: DblArray = Vec::new();
            let mut all_fit = true; let rem = budget as f64 - cur_w;
            for (k, &nd) in cand_nodes.iter().enumerate() {
                if k == best_idx { continue; }
                if weights[nd] as f64 > rem { all_fit = false; continue; }
                new_cands.push(nd); new_contribs.push(cand_contribs[k]);
            }
            for (k, &nd) in new_cands.iter().enumerate() {
                new_contribs[k] += (1.0 + 2.0 * beta) * um.get(nd, best_node) / weights[nd] as f64;
            }
            if let Some(ref mut ws_ref) = ws {
                if update_flag && all_fit {
                    ws_ref.left_nodes = left.clone(); ws_ref.candidate_nodes = new_cands.clone();
                    ws_ref.candidate_contribs = new_contribs.clone();
                    ws_ref.current_total_weight = cur_w; ws_ref.valid = true;
                } else { update_flag = false; }
            }
            cand_nodes = new_cands; cand_contribs = new_contribs;
        }
        left
    }

    fn run_greedy_right(um: &PairUtil, n_nodes: usize, mut right_nodes: IntArray,
                        budget: i32, beta: f64, weights: &[i32], ws: Option<&mut WarmStartRight>) -> IntArray {
        if right_nodes.is_empty() { return right_nodes; }
        let mut cur_w: f64 = match &ws {
            Some(w) if w.valid => w.current_total_weight,
            _ => right_nodes.iter().map(|&k| weights[k] as f64).sum(),
        };
        let ws_valid = ws.as_ref().map(|w| w.valid).unwrap_or(false);
        let ws_cands = ws.as_ref().map(|w| !w.candidate_nodes.is_empty()).unwrap_or(false);
        let mut cand_nodes: IntArray = Vec::new(); let mut cand_contribs: DblArray = Vec::new();
        if ws_valid && ws_cands {
            let w = ws.as_ref().unwrap(); cand_nodes = w.candidate_nodes.clone(); cand_contribs = w.candidate_contribs.clone();
        } else {
            for &nd in &right_nodes {
                let mut contrib: f64 = right_nodes.iter().map(|&m| (-1.0 - beta) * um.get(nd, m)).sum();
                contrib -= um.linear[nd];
                if beta != 0.0 { for v in 0..n_nodes { if !ia_contains(&right_nodes, v) { contrib += beta * um.get(nd, v); } } }
                contrib /= weights[nd] as f64; cand_nodes.push(nd); cand_contribs.push(contrib);
            }
        }
        while !cand_nodes.is_empty() && cur_w > budget as f64 {
            let best_idx = cand_contribs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
            let best_node = cand_nodes[best_idx];
            right_nodes.retain(|&x| x != best_node); cur_w -= weights[best_node] as f64;
            let mut new_cands: IntArray = Vec::new(); let mut new_contribs: DblArray = Vec::new();
            for (k, &nd) in cand_nodes.iter().enumerate() {
                if k == best_idx { continue; } new_cands.push(nd); new_contribs.push(cand_contribs[k]);
            }
            for (k, &nd) in new_cands.iter().enumerate() {
                new_contribs[k] += (1.0 + 2.0 * beta) * um.get(nd, best_node) / weights[nd] as f64;
            }
            cand_nodes = new_cands; cand_contribs = new_contribs;
        }
        right_nodes
    }

    struct GreedyResults { left_nodes: Vec<IntArray>, right_nodes: Vec<IntArray> }

    fn run_greedy(inst: &Phase1Problem, beta: f64, breakpoints: &[IntArray], bp_weights: &[f64]) -> GreedyResults {
        let n_budgets = inst.n_budgets; let n_nodes = inst.n_items; let n_bp_total = breakpoints.len();
        let um = build_utility_matrix(n_nodes, &inst.edges);
        let all_nodes: IntArray = (0..n_nodes).collect();
        let mut left_nodes: Vec<IntArray> = Vec::with_capacity(n_budgets);
        let mut ws_left = WarmStartLeft::new();
        for bi in 0..n_budgets {
            let budget = inst.budgets[bi]; let mut bp_idx = 0usize;
            for k in 0..n_bp_total { if bp_weights[k] <= budget as f64 { bp_idx = k; } }
            let left_init = breakpoints[bp_idx].clone();
            if ws_left.valid && ws_left.left_nodes.len() < left_init.len() { ws_left.reset(); }
            let res = run_greedy_left(&um, n_nodes, left_init, &all_nodes, budget, beta, &inst.weights, Some(&mut ws_left));
            left_nodes.push(res);
        }
        let mut right_nodes: Vec<IntArray> = vec![Vec::new(); n_budgets];
        let mut ws_right = WarmStartRight::new(); let mut selected_right: IntArray = Vec::new();
        for bi in (0..n_budgets).rev() {
            let budget = inst.budgets[bi]; let mut bp_idx = n_bp_total - 1;
            for k in 0..n_bp_total { if bp_weights[k] >= budget as f64 { bp_idx = k; break; } }
            let right_init = breakpoints[bp_idx].clone();
            if selected_right.is_empty() || selected_right.len() >= right_init.len() { selected_right = right_init; ws_right.reset(); }
            let after_right = run_greedy_right(&um, n_nodes, selected_right.clone(), budget, beta, &inst.weights, Some(&mut ws_right));
            selected_right = after_right.clone();
            let final_left = run_greedy_left(&um, n_nodes, after_right, &all_nodes, budget, beta, &inst.weights, None);
            right_nodes[bi] = final_left;
        }
        GreedyResults { left_nodes, right_nodes }
    }

    pub fn bound_and_prune(inst: &Phase1Problem, n_lambda_values: usize) -> QkpOutcome {
        let bps = get_breakpoints(inst, n_lambda_values);
        let n_breakpoints = bps.sets.len();
        let mut total_weights_at_bp = vec![0.0f64; n_breakpoints];
        { let mut cumsum = 0.0f64;
          for (i, (_, nodes)) in bps.sets.iter().enumerate() {
              for &nd in nodes { cumsum += inst.weights[nd] as f64; }
              total_weights_at_bp[i] = cumsum;
          }
        }
        let n_bp_total = n_breakpoints + 1;
        let mut breakpoints: Vec<IntArray> = Vec::with_capacity(n_bp_total);
        let mut bp_weights: Vec<f64> = Vec::with_capacity(n_bp_total);
        breakpoints.push(Vec::new()); bp_weights.push(0.0);
        for i in 0..n_breakpoints {
            let mut next = breakpoints[i].clone();
            for &nd in &bps.sets[i].1 { next.push(nd); }
            breakpoints.push(next); bp_weights.push(total_weights_at_bp[i]);
        }
        let gr = run_greedy(inst, 0.0, &breakpoints, &bp_weights);
        let mut results: Vec<BudgetOutcome> = Vec::with_capacity(inst.n_budgets);
        for bi in 0..inst.n_budgets {
            let ofv_left  = compute_ofv(&gr.left_nodes[bi],  inst.n_items, &inst.edges);
            let ofv_right = compute_ofv(&gr.right_nodes[bi], inst.n_items, &inst.edges);
            let (best_items, ofv) = if ofv_left >= ofv_right {
                (gr.left_nodes[bi].clone(), ofv_left)
            } else { (gr.right_nodes[bi].clone(), ofv_right) };
            results.push(BudgetOutcome { budget: inst.budgets[bi], ofv, selected_items: best_items });
        }
        QkpOutcome { results }
    }

// ILS state and helpers

    #[derive(Clone, Copy)]
    struct Rng { state: u64 }
    impl Rng {
        fn from_seed(seed: &[u8; 32]) -> Self {
            let mut s: u64 = 0x9E3779B97F4A7C15;
            for (i, &b) in seed.iter().enumerate() {
                s ^= (b as u64) << ((i & 7) * 8);
                s = s.rotate_left(7).wrapping_mul(0xBF58476D1CE4E5B9);
            }
            if s == 0 { s = 1; } Self { state: s }
        }
        #[inline] fn next_u64(&mut self) -> u64 { let mut x = self.state; x ^= x << 7; x ^= x >> 9; x ^= x << 8; self.state = x; x }
        #[inline] fn next_u32(&mut self) -> u32 { (self.next_u64() >> 32) as u32 }
        #[inline] fn next_f64(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
        #[inline] fn next_usize(&mut self, bound: usize) -> usize { if bound == 0 { return 0; } (self.next_u64() % bound as u64) as usize }
    }

    struct State<'a> {
        ch: &'a Challenge,
        selected_bit: Vec<bool>,
        contrib: Vec<i32>,
        total_value: i64,
        total_weight: u32,
        dp_cache: Vec<i64>,
        choose_cache: Vec<u8>,
    }

    impl<'a> State<'a> {
        fn new_empty(ch: &'a Challenge) -> Self {
            let n = ch.num_items;
            let mut contrib = vec![0i32; n];
            for i in 0..n { contrib[i] = ch.values[i] as i32; }
            Self { ch, selected_bit: vec![false; n], contrib, total_value: 0, total_weight: 0, dp_cache: Vec::new(), choose_cache: Vec::new() }
        }
        #[inline(always)] fn slack(&self) -> u32 { self.ch.max_weight - self.total_weight }
        #[inline(always)]
        fn add_item(&mut self, i: usize) {
            self.total_value += self.contrib[i] as i64;
            self.total_weight += self.ch.weights[i];
            let n = self.ch.num_items;
            let row_ptr = unsafe { self.ch.interaction_values.get_unchecked(i).as_ptr() };
            let contrib_ptr = self.contrib.as_mut_ptr();
            unsafe { for k in 0..n { let ck = contrib_ptr.add(k); *ck = (*ck).wrapping_add(*row_ptr.add(k)); } }
            self.selected_bit[i] = true;
        }
        #[inline(always)]
        fn remove_item(&mut self, j: usize) {
            self.total_value -= self.contrib[j] as i64;
            self.total_weight -= self.ch.weights[j];
            let n = self.ch.num_items;
            let row_ptr = unsafe { self.ch.interaction_values.get_unchecked(j).as_ptr() };
            let contrib_ptr = self.contrib.as_mut_ptr();
            unsafe { for k in 0..n { let ck = contrib_ptr.add(k); *ck = (*ck).wrapping_sub(*row_ptr.add(k)); } }
            self.selected_bit[j] = false;
        }
        #[inline(always)] fn replace_item(&mut self, rm: usize, cand: usize) { self.remove_item(rm); self.add_item(cand); }
        fn selected_items(&self) -> Vec<usize> { (0..self.ch.num_items).filter(|&i| self.selected_bit[i]).collect() }
        fn clone_solution(&self) -> SolState { SolState { bits: self.selected_bit.clone(), contrib: self.contrib.clone(), value: self.total_value, weight: self.total_weight } }
        fn restore_solution(&mut self, sol: &SolState) { self.selected_bit.clone_from(&sol.bits); self.contrib.clone_from(&sol.contrib); self.total_value = sol.value; self.total_weight = sol.weight; }
    }

    #[derive(Clone)]
    struct SolState { bits: Vec<bool>, contrib: Vec<i32>, value: i64, weight: u32 }

// Greedy constructors used by the ILS loop

    fn build_greedy_density(state: &mut State) {
        let n = state.ch.num_items; let cap = state.ch.max_weight;
        for i in 0..n { state.add_item(i); }
        while state.total_weight > cap {
            let mut worst = 0; let mut worst_s = i64::MAX;
            for i in 0..n {
                if state.selected_bit[i] {
                    let c = state.contrib[i] as i64;
                    let w = (state.ch.weights[i] as i64).max(1);
                    let s = (c * 1000) / w;
                    if s < worst_s { worst_s = s; worst = i; }
                }
            }
            state.remove_item(worst);
        }
        for _ in 0..2 {
            let mut by_density: Vec<usize> = (0..n).collect();
            let contrib = &state.contrib; let weights = &state.ch.weights;
            by_density.sort_unstable_by(|&a, &b| {
                let na = contrib[a] as i64; let nb = contrib[b] as i64;
                let wa = weights[a] as i64; let wb = weights[b] as i64;
                (na * wb).cmp(&(nb * wa)).reverse()
            });
            let mut target = vec![false; n]; let mut rem = cap;
            for &i in &by_density {
                if state.ch.weights[i] <= rem { target[i] = true; rem -= state.ch.weights[i]; }
            }
            let mut to_rm = Vec::new(); let mut to_add = Vec::new();
            for i in 0..n {
                if state.selected_bit[i] && !target[i] { to_rm.push(i); }
                if !state.selected_bit[i] && target[i] { to_add.push(i); }
            }
            if to_rm.is_empty() && to_add.is_empty() { break; }
            for &r in &to_rm { state.remove_item(r); }
            for &a in &to_add { state.add_item(a); }
        }
    }

    fn build_greedy_value(state: &mut State) {
        let n = state.ch.num_items; let cap = state.ch.max_weight;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by_key(|&i| std::cmp::Reverse(state.ch.values[i]));
        for &i in &order {
            if state.total_weight + state.ch.weights[i] <= cap { state.add_item(i); }
        }
    }

    fn build_greedy_hub(state: &mut State) {
        let n = state.ch.num_items; let cap = state.ch.max_weight;
        let mut hub_scores: Vec<(usize, i64)> = (0..n).map(|i| {
            let s: i64 = state.ch.interaction_values[i].iter().map(|&v| v as i64).sum();
            (i, s)
        }).collect();
        hub_scores.sort_unstable_by_key(|&(_, s)| std::cmp::Reverse(s));
        for &(i, _) in &hub_scores {
            if state.total_weight + state.ch.weights[i] <= cap { state.add_item(i); }
        }
    }

    fn build_greedy_synergy_weight(state: &mut State) {
        let n = state.ch.num_items; let cap = state.ch.max_weight;
        let mut scores: Vec<(usize, i64)> = (0..n).map(|i| {
            let avg_syn: i64 = if n > 1 {
                state.ch.interaction_values[i].iter().map(|&v| v as i64).sum::<i64>() / (n as i64 - 1)
            } else { 0 };
            let w = (state.ch.weights[i] as i64).max(1);
            (i, (state.ch.values[i] as i64 + avg_syn) * 100 / w)
        }).collect();
        scores.sort_unstable_by_key(|&(_, s)| std::cmp::Reverse(s));
        for &(i, _) in &scores {
            if state.total_weight + state.ch.weights[i] <= cap { state.add_item(i); }
        }
    }

    fn construct_forward_incremental(state: &mut State, mode: usize, rng: &mut Rng) {
        let n = state.ch.num_items;
        loop {
            let slack = state.slack();
            if slack == 0 { break; }
            let mut best_i: Option<usize> = None; let mut best_s: i64 = i64::MIN;
            let mut second_i: Option<usize> = None; let mut second_s: i64 = i64::MIN;
            for i in 0..n {
                if state.selected_bit[i] { continue; }
                if state.ch.weights[i] > slack { continue; }
                let c = state.contrib[i] as i64;
                if c <= 0 { continue; }
                let w = (state.ch.weights[i] as i64).max(1);
                let mut s = match mode {
                    2 => c,
                    3 => (c * 1000) / w + (state.ch.weights[i] as i64) * 3,
                    _ => (c * 1000) / w,
                };
                if mode >= 4 {
                    let mask = if mode >= 5 { 0x7F } else { 0x1F };
                    s += (rng.next_u32() & mask) as i64;
                }
                if s > best_s { second_s = best_s; second_i = best_i; best_s = s; best_i = Some(i); }
                else if s > second_s { second_s = s; second_i = Some(i); }
            }
            let pick = if mode >= 4 && second_i.is_some() {
                let m = if mode >= 5 { 1 } else { 3 };
                if (rng.next_u32() & m) == 0 { second_i } else { best_i }
            } else { best_i };
            if let Some(i) = pick { state.add_item(i); } else { break; }
        }
    }

    fn build_hub_pair_kth(state: &mut State, k: usize) {
        let n = state.ch.num_items; let cap = state.ch.max_weight;
        let mut pairs: Vec<(i32, usize, usize)> = Vec::new();
        for i in 0..n {
            for j in (i+1)..n {
                if state.ch.weights[i] + state.ch.weights[j] <= cap {
                    pairs.push((state.ch.interaction_values[i][j], i, j));
                }
            }
        }
        pairs.sort_unstable_by_key(|&(s, _, _)| std::cmp::Reverse(s));
        let mut used = Vec::new(); let mut count = 0;
        for &(_, pi, pj) in &pairs {
            if used.contains(&pi) || used.contains(&pj) { continue; }
            if count == k { state.add_item(pi); state.add_item(pj); break; }
            used.push(pi); used.push(pj); count += 1;
        }
        loop {
            let slack = state.slack();
            if slack == 0 { break; }
            let mut best_i: Option<usize> = None; let mut best_s: i64 = 0;
            for i in 0..n {
                if state.selected_bit[i] { continue; }
                if state.ch.weights[i] > slack { continue; }
                let c = state.contrib[i] as i64;
                if c <= 0 { continue; }
                let w = (state.ch.weights[i] as i64).max(1);
                let s = (c * 1000) / w;
                if s > best_s { best_s = s; best_i = Some(i); }
            }
            if let Some(i) = best_i { state.add_item(i); } else { break; }
        }
    }

    fn dp_refinement_hp(state: &mut State, core_half: usize) {
        let n = state.ch.num_items; let cap = state.ch.max_weight;
        let weights = &state.ch.weights;
        for _iter in 0..4 {
            let contrib = &state.contrib;
            let mut by_density: Vec<usize> = (0..n).collect();
            by_density.sort_unstable_by(|&a, &b| {
                let na = contrib[a] as i64; let nb = contrib[b] as i64;
                let wa = weights[a] as i64; let wb = weights[b] as i64;
                (na * wb).cmp(&(nb * wa)).reverse()
            });
            let mut idx_last_inserted = 0usize; let mut idx_first_rejected = n;
            let mut rem = cap;
            for (idx, &i) in by_density.iter().enumerate() {
                let w = weights[i];
                if w <= rem { rem -= w; idx_last_inserted = idx; }
                else if idx_first_rejected == n { idx_first_rejected = idx; }
            }
            let left  = idx_first_rejected.saturating_sub(core_half + 1);
            let right = (idx_last_inserted + core_half + 1).min(n);
            let locked: Vec<usize> = by_density[..left].to_vec();
            let core:   Vec<usize> = by_density[left..right].to_vec();
            let used_locked: u64 = locked.iter().map(|&i| weights[i] as u64).sum();
            let rem_cap = (cap as u64).saturating_sub(used_locked) as usize;
            let myk = core.len();
            if myk == 0 || rem_cap == 0 { break; }

            let mut total_core_weight: usize = 0;
            let mut total_pos_weight:  usize = 0;
            let mut all_pos_fit = true;
            for &it in &core {
                let wt = weights[it] as usize;
                total_core_weight += wt;
                if contrib[it] > 0 {
                    total_pos_weight += wt;
                    if total_pos_weight > rem_cap { all_pos_fit = false; }
                }
            }

            let target_sel = if all_pos_fit {
                let mut sel: Vec<usize> = locked.clone();
                for &it in &core { if contrib[it] > 0 { sel.push(it); } }
                sel.sort_unstable(); sel
            } else {
                let myw = rem_cap.min(total_core_weight);
                let dp_size = myw + 1;
                let choose_size = myk * dp_size;
                if state.dp_cache.len() < dp_size { state.dp_cache.resize(dp_size, i64::MIN / 4); }
                if state.choose_cache.len() < choose_size { state.choose_cache.resize(choose_size, 0); }
                let init_val = i64::MIN / 4;
                for v in &mut state.dp_cache[..dp_size] { *v = init_val; }
                state.choose_cache[..choose_size].fill(0);
                state.dp_cache[0] = 0;
                let mut w_hi: usize = 0;
                for (t, &it) in core.iter().enumerate() {
                    let wt = weights[it] as usize;
                    if wt > myw { continue; }
                    let val = contrib[it] as i64;
                    let new_hi = (w_hi + wt).min(myw);
                    for w in (wt..=new_hi).rev() {
                        let cand = state.dp_cache[w - wt] + val;
                        if cand > state.dp_cache[w] {
                            state.dp_cache[w] = cand;
                            state.choose_cache[t * dp_size + w] = 1;
                        }
                    }
                    w_hi = new_hi;
                }
                let mut sel: Vec<usize> = locked.clone();
                let mut w_star = (0..=myw).max_by_key(|&w| state.dp_cache[w]).unwrap_or(0);
                for t in (0..myk).rev() {
                    let it = core[t]; let wt = weights[it] as usize;
                    if wt <= w_star && state.choose_cache[t * dp_size + w_star] == 1 {
                        sel.push(it); w_star -= wt;
                    }
                }
                sel.sort_unstable(); sel
            };

            let mut to_rm = Vec::new(); let mut to_add = Vec::new();
            let mut j = 0; let m = target_sel.len();
            for i in 0..n {
                let in_target = j < m && target_sel[j] == i;
                if in_target { j += 1; }
                if state.selected_bit[i] && !in_target { to_rm.push(i); }
                else if in_target && !state.selected_bit[i] { to_add.push(i); }
            }
            if to_rm.is_empty() && to_add.is_empty() { break; }
            for r in to_rm { state.remove_item(r); }
            for a in to_add { state.add_item(a); }
        }
    }

    fn apply_best_add(state: &mut State) -> bool {
        let slack = state.slack(); if slack == 0 { return false; }
        let n = state.ch.num_items;
        let mut best_i: Option<usize> = None; let mut best_d: i32 = 0;
        for i in 0..n {
            if state.selected_bit[i] { continue; }
            if state.ch.weights[i] > slack { continue; }
            let d = state.contrib[i];
            if d > best_d { best_d = d; best_i = Some(i); }
        }
        if let Some(i) = best_i { state.add_item(i); true } else { false }
    }

    fn apply_best_swap_1_1(state: &mut State, selected: &[usize]) -> bool {
        let n = state.ch.num_items; let slack = state.slack();
        let mut best: Option<(usize, usize, i32)> = None;
        for &rm in selected {
            let w_rm = state.ch.weights[rm]; let max_w = w_rm + slack;
            for cand in 0..n {
                if state.selected_bit[cand] { continue; }
                let wc = state.ch.weights[cand];
                if wc > max_w { continue; }
                let delta = state.contrib[cand] - state.contrib[rm] - state.ch.interaction_values[cand][rm];
                if delta > 0 && best.map_or(true, |(_, _, bd)| delta > bd) { best = Some((cand, rm, delta)); }
            }
        }
        if let Some((cand, rm, _)) = best { state.replace_item(rm, cand); true } else { false }
    }

    fn apply_pair_add(state: &mut State) -> bool {
        let slack = state.slack(); if slack < 2 { return false; }
        let n = state.ch.num_items;
        let unsel: Vec<usize> = (0..n).filter(|&i| !state.selected_bit[i] && state.ch.weights[i] < slack).collect();
        let m = unsel.len(); if m < 2 { return false; }
        let mut best_delta: i64 = 0; let mut best_pair: Option<(usize, usize)> = None;
        for ai in 0..m {
            let a = unsel[ai]; let wa = state.ch.weights[a]; let ca = state.contrib[a] as i64;
            for bi in (ai+1)..m {
                let b = unsel[bi];
                if wa + state.ch.weights[b] > slack { continue; }
                let delta = ca + state.contrib[b] as i64 + state.ch.interaction_values[a][b] as i64;
                if delta > best_delta { best_delta = delta; best_pair = Some((a, b)); }
            }
        }
        if let Some((a, b)) = best_pair { state.add_item(a); state.add_item(b); true } else { false }
    }

    fn apply_chain_move(state: &mut State) -> bool {
        let n = state.ch.num_items;
        let mut sel: Vec<(usize, i32)> = (0..n).filter(|&i| state.selected_bit[i]).map(|i| (i, state.contrib[i])).collect();
        sel.sort_unstable_by_key(|&(_, c)| c);
        let sel_len = sel.len().min(80);
        let mut unsel: Vec<(usize, i32)> = (0..n).filter(|&i| !state.selected_bit[i]).map(|i| (i, state.contrib[i])).collect();
        unsel.sort_unstable_by_key(|&(_, c)| std::cmp::Reverse(c));
        let unsel_len = unsel.len().min(80);
        let cap = state.ch.max_weight;
        let mut best_delta: i64 = 0; let mut best_move: Option<(usize, usize, usize)> = None;
        for i_rm in 0..sel_len {
            let rm = sel[i_rm].0; let w_rm = state.ch.weights[rm] as i64;
            let c_rm = state.contrib[rm] as i64; let budget = state.slack() as i64 + w_rm;
            for ui in 0..unsel_len {
                let a1 = unsel[ui].0; let w_a1 = state.ch.weights[a1] as i64;
                if w_a1 >= budget { continue; }
                let c_a1 = state.contrib[a1] as i64 - state.ch.interaction_values[a1][rm] as i64;
                for uj in (ui+1)..unsel_len {
                    let a2 = unsel[uj].0; let w_a2 = state.ch.weights[a2] as i64;
                    if w_a1 + w_a2 > budget { continue; }
                    let c_a2 = state.contrib[a2] as i64 - state.ch.interaction_values[a2][rm] as i64;
                    let syn = state.ch.interaction_values[a1][a2] as i64;
                    let delta = c_a1 + c_a2 + syn - c_rm;
                    if delta > best_delta {
                        let new_w = state.total_weight as i64 - w_rm + w_a1 + w_a2;
                        if new_w <= cap as i64 { best_delta = delta; best_move = Some((rm, a1, a2)); }
                    }
                }
            }
        }
        if let Some((rm, a1, a2)) = best_move { state.remove_item(rm); state.add_item(a1); state.add_item(a2); true } else { false }
    }

    fn apply_reverse_chain(state: &mut State) -> bool {
        let n = state.ch.num_items;
        let mut sel: Vec<(usize, i32)> = (0..n).filter(|&i| state.selected_bit[i]).map(|i| (i, state.contrib[i])).collect();
        sel.sort_unstable_by_key(|&(_, c)| c);
        let sel_len = sel.len().min(80);
        let mut unsel: Vec<(usize, i32)> = (0..n).filter(|&i| !state.selected_bit[i]).map(|i| (i, state.contrib[i])).collect();
        unsel.sort_unstable_by_key(|&(_, c)| std::cmp::Reverse(c));
        let unsel_len = unsel.len().min(80);
        let cap = state.ch.max_weight;
        let mut best_delta: i64 = 0; let mut best_move: Option<(usize, usize, usize)> = None;
        for i_add in 0..unsel_len {
            let add = unsel[i_add].0; let w_add = state.ch.weights[add] as i64;
            let c_add = state.contrib[add] as i64;
            for si in 0..sel_len {
                let r1 = sel[si].0; let w_r1 = state.ch.weights[r1] as i64;
                let c_r1 = state.contrib[r1] as i64;
                let c_add_r1 = state.ch.interaction_values[add][r1] as i64;
                for sj in (si+1)..sel_len {
                    let r2 = sel[sj].0; let w_r2 = state.ch.weights[r2] as i64;
                    let freed = w_r1 + w_r2;
                    let new_w = state.total_weight as i64 - freed + w_add;
                    if new_w > cap as i64 || new_w < 0 { continue; }
                    let c_r2 = state.contrib[r2] as i64;
                    let syn_r1_r2 = state.ch.interaction_values[r1][r2] as i64;
                    let c_add_r2 = state.ch.interaction_values[add][r2] as i64;
                    let lost = c_r1 + c_r2 - syn_r1_r2;
                    let gained = c_add - c_add_r1 - c_add_r2;
                    let delta = gained - lost;
                    if delta > best_delta { best_delta = delta; best_move = Some((r1, r2, add)); }
                }
            }
        }
        if let Some((r1, r2, add)) = best_move { state.remove_item(r1); state.remove_item(r2); state.add_item(add); true } else { false }
    }

    fn apply_swap_2_2_bounded(state: &mut State, k: usize) -> bool {
        let n = state.ch.num_items;
        let mut sel_ranked: Vec<(usize, i32)> = (0..n).filter(|&i| state.selected_bit[i]).map(|i| (i, state.contrib[i])).collect();
        sel_ranked.sort_unstable_by_key(|&(_, c)| c); sel_ranked.truncate(k);
        let mut unsel_ranked: Vec<(usize, i32)> = (0..n).filter(|&i| !state.selected_bit[i]).map(|i| (i, state.contrib[i])).collect();
        unsel_ranked.sort_unstable_by_key(|&(_, c)| std::cmp::Reverse(c)); unsel_ranked.truncate(k);
        let cap = state.ch.max_weight;
        let mut best_delta: i64 = 0; let mut best_move: Option<(usize, usize, usize, usize)> = None;
        for si in 0..sel_ranked.len() {
            let r1 = sel_ranked[si].0; let w_r1 = state.ch.weights[r1] as i64; let c_r1 = state.contrib[r1] as i64;
            for sj in (si+1)..sel_ranked.len() {
                let r2 = sel_ranked[sj].0; let w_r2 = state.ch.weights[r2] as i64; let c_r2 = state.contrib[r2] as i64;
                let freed_weight = w_r1 + w_r2;
                let removed_syn = state.ch.interaction_values[r1][r2] as i64;
                let lost = c_r1 + c_r2 - removed_syn;
                let budget = state.slack() as i64 + freed_weight;
                for ui in 0..unsel_ranked.len() {
                    let a1 = unsel_ranked[ui].0; let w_a1 = state.ch.weights[a1] as i64;
                    if w_a1 > budget { continue; }
                    let c_a1 = state.contrib[a1] as i64 - state.ch.interaction_values[a1][r1] as i64 - state.ch.interaction_values[a1][r2] as i64;
                    for uj in (ui+1)..unsel_ranked.len() {
                        let a2 = unsel_ranked[uj].0; let w_a2 = state.ch.weights[a2] as i64;
                        if w_a1 + w_a2 > budget { continue; }
                        let c_a2 = state.contrib[a2] as i64 - state.ch.interaction_values[a2][r1] as i64 - state.ch.interaction_values[a2][r2] as i64;
                        let added_syn = state.ch.interaction_values[a1][a2] as i64;
                        let delta = c_a1 + c_a2 + added_syn - lost;
                        if delta > best_delta {
                            let new_weight = state.total_weight as i64 - freed_weight + w_a1 + w_a2;
                            if new_weight <= cap as i64 { best_delta = delta; best_move = Some((r1, r2, a1, a2)); }
                        }
                    }
                }
            }
        }
        if let Some((r1, r2, a1, a2)) = best_move { state.remove_item(r1); state.remove_item(r2); state.add_item(a1); state.add_item(a2); true } else { false }
    }

    fn local_search_vnd_fast(state: &mut State) {
        let n = state.ch.num_items;
        let mut selected_buf: Vec<usize> = Vec::with_capacity(n);
        for _ in 0..80 {
            if apply_best_add(state) { continue; }
            selected_buf.clear();
            for i in 0..n { if state.selected_bit[i] { selected_buf.push(i); } }
            if apply_best_swap_1_1(state, &selected_buf) { continue; }
            break;
        }
    }

    fn local_search_vnd_medium(state: &mut State, k: usize) {
        let n = state.ch.num_items;
        let mut selected_buf: Vec<usize> = Vec::with_capacity(n);
        for _ in 0..120 {
            if apply_best_add(state) { continue; }
            selected_buf.clear();
            for i in 0..n { if state.selected_bit[i] { selected_buf.push(i); } }
            if apply_best_swap_1_1(state, &selected_buf) { continue; }
            if apply_pair_add(state) { continue; }
            if apply_swap_2_2_bounded(state, k) { continue; }
            break;
        }
    }

    fn ils_vnd(state: &mut State, hp: &Hparams) {
        match hp.ils_vnd_level {
            0 => local_search_vnd_fast(state),
            1 => local_search_vnd_medium(state, hp.bounded_2_2_k),
            _ => local_search_vnd_heavy(state),
        }
    }

    fn local_search_vnd_heavy(state: &mut State) {
        let n = state.ch.num_items;
        let mut selected_buf: Vec<usize> = Vec::with_capacity(n);
        for _ in 0..300 {
            if apply_best_add(state) { continue; }
            selected_buf.clear();
            for i in 0..n { if state.selected_bit[i] { selected_buf.push(i); } }
            if apply_best_swap_1_1(state, &selected_buf) { continue; }
            if apply_pair_add(state) { continue; }
            if apply_swap_2_2_bounded(state, 25) { continue; }
            if apply_chain_move(state) { continue; }
            if apply_reverse_chain(state) { continue; }
            break;
        }
    }

    fn simulated_annealing(state: &mut State, rng: &mut Rng, n_rounds: usize, n_iter: usize) {
        let n = state.ch.num_items; let cap = state.ch.max_weight;
        let mut sel: Vec<usize> = Vec::with_capacity(n);
        let mut unsel: Vec<usize> = Vec::with_capacity(n);
        let mut pos_in_sel = vec![0usize; n]; let mut pos_in_unsel = vec![0usize; n];
        for i in 0..n {
            if state.selected_bit[i] { pos_in_sel[i] = sel.len(); sel.push(i); }
            else { pos_in_unsel[i] = unsel.len(); unsel.push(i); }
        }
        if sel.is_empty() || unsel.is_empty() { return; }
        let mut best_snap = state.clone_solution();
        let mut deltas: Vec<f64> = Vec::new();
        for _ in 0..100 {
            let rm = sel[rng.next_usize(sel.len())]; let add = unsel[rng.next_usize(unsel.len())];
            let d = state.contrib[add] as f64 - state.contrib[rm] as f64 - state.ch.interaction_values[add][rm] as f64;
            if d < 0.0 { deltas.push(-d); }
        }
        if deltas.is_empty() { return; }
        deltas.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let p75 = deltas[deltas.len() * 3 / 4];
        let t0 = p75 / 0.693;
        if t0 < 1.0 { return; }
        let alpha = 0.95f64; let mut temp = t0;
        for _ in 0..n_rounds {
            for _ in 0..n_iter {
                if sel.is_empty() || unsel.is_empty() { continue; }
                let coin = rng.next_u32() % 10;
                if coin < 8 {
                    let si = rng.next_usize(sel.len()); let ui = rng.next_usize(unsel.len());
                    let rm = sel[si]; let add = unsel[ui];
                    let w_new = state.total_weight - state.ch.weights[rm] + state.ch.weights[add];
                    if w_new > cap { continue; }
                    let delta = state.contrib[add] as i64 - state.contrib[rm] as i64 - state.ch.interaction_values[add][rm] as i64;
                    if delta > 0 || rng.next_f64() < (-delta as f64 / temp).exp() {
                        state.replace_item(rm, add);
                        let last_sel = *sel.last().unwrap(); sel[si] = last_sel; pos_in_sel[last_sel] = si; sel.pop(); pos_in_sel[rm] = 0;
                        let last_unsel = *unsel.last().unwrap(); unsel[ui] = last_unsel; pos_in_unsel[last_unsel] = ui; unsel.pop(); pos_in_unsel[add] = 0;
                        pos_in_sel[add] = sel.len(); sel.push(add); pos_in_unsel[rm] = unsel.len(); unsel.push(rm);
                    }
                } else if coin == 8 {
                    let slack = state.slack(); if slack == 0 { continue; }
                    let ui = rng.next_usize(unsel.len()); let add = unsel[ui];
                    if state.ch.weights[add] > slack { continue; }
                    let delta = state.contrib[add] as i64;
                    if delta > 0 || rng.next_f64() < (-delta as f64 / temp).exp() {
                        state.add_item(add);
                        let last_unsel = *unsel.last().unwrap(); unsel[ui] = last_unsel; pos_in_unsel[last_unsel] = ui; unsel.pop();
                        pos_in_sel[add] = sel.len(); sel.push(add);
                    }
                } else {
                    let si = rng.next_usize(sel.len()); let rm = sel[si];
                    let delta = -(state.contrib[rm] as i64);
                    if rng.next_f64() < (-delta as f64 / temp).exp() {
                        state.remove_item(rm);
                        let last_sel = *sel.last().unwrap(); sel[si] = last_sel; pos_in_sel[last_sel] = si; sel.pop();
                        pos_in_unsel[rm] = unsel.len(); unsel.push(rm);
                    }
                }
                if state.total_value > best_snap.value { best_snap = state.clone_solution(); }
            }
            temp *= alpha;
        }
        if best_snap.value > state.total_value { state.restore_solution(&best_snap); }
    }

    fn crossover_frequency(population: &[SolState], ch: &Challenge, rng: &mut Rng) -> Vec<bool> {
        let n = ch.num_items; let pop_size = population.len();
        let mut freq = vec![0usize; n];
        for sol in population { for i in 0..n { if sol.bits[i] { freq[i] += 1; } } }
        let threshold = (pop_size * 3) / 4;
        let mut child_bits = vec![false; n]; let mut child_weight: u32 = 0;
        let mut consensus: Vec<usize> = Vec::new(); let mut exploratory: Vec<usize> = Vec::new();
        for i in 0..n {
            if freq[i] > threshold { consensus.push(i); }
            else if freq[i] > 0 { exploratory.push(i); }
        }
        for &i in &consensus { if child_weight + ch.weights[i] <= ch.max_weight { child_bits[i] = true; child_weight += ch.weights[i]; } }
        for &i in &exploratory { if rng.next_u32() % 2 == 0 && child_weight + ch.weights[i] <= ch.max_weight { child_bits[i] = true; child_weight += ch.weights[i]; } }
        child_bits
    }

    fn crossover_uniform(sol_a: &SolState, sol_b: &SolState, ch: &Challenge, rng: &mut Rng) -> Vec<bool> {
        let n = ch.num_items; let mut bits = vec![false; n]; let mut weight: u32 = 0;
        for i in 0..n { if sol_a.bits[i] && sol_b.bits[i] { if weight + ch.weights[i] <= ch.max_weight { bits[i] = true; weight += ch.weights[i]; } } }
        for i in 0..n { if bits[i] { continue; } if sol_a.bits[i] || sol_b.bits[i] { if rng.next_u32() % 2 == 0 && weight + ch.weights[i] <= ch.max_weight { bits[i] = true; weight += ch.weights[i]; } } }
        bits
    }

    fn set_state_from_bits(state: &mut State, bits: &[bool]) {
        let n = state.ch.num_items;
        for i in (0..n).rev() { if state.selected_bit[i] { state.remove_item(i); } }
        for i in 0..n { if bits[i] { state.add_item(i); } }
    }

    fn build_windows(state: &State, k: usize) -> (Vec<usize>, Vec<usize>) {
        let n = state.ch.num_items;
        let mut unused_r: Vec<(usize, f64)> = Vec::with_capacity(n);
        let mut used_r: Vec<(usize, f64)> = Vec::with_capacity(n);
        for i in 0..n {
            let r = state.contrib[i] as f64 / (state.ch.weights[i] as f64).max(1.0);
            if state.selected_bit[i] { used_r.push((i, r)); } else { unused_r.push((i, r)); }
        }
        let ku = k.min(unused_r.len());
        if ku > 0 && ku < unused_r.len() { unused_r.select_nth_unstable_by(ku - 1, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)); }
        let ks = k.min(used_r.len());
        if ks > 0 && ks < used_r.len() { used_r.select_nth_unstable_by(ks - 1, |a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)); }
        (unused_r[..ku].iter().map(|x| x.0).collect(), used_r[..ks].iter().map(|x| x.0).collect())
    }

    fn local_search_vnd_windowed(state: &mut State, window_k: usize) {
        for _ in 0..80 {
            let (best_unused, worst_used) = build_windows(state, window_k);
            let slack = state.slack();
            if slack > 0 {
                let mut ba: Option<(usize, i32)> = None;
                for &c in &best_unused {
                    if state.ch.weights[c] > slack { continue; }
                    let d = state.contrib[c];
                    if d > 0 && ba.map_or(true, |(_, bd)| d > bd) { ba = Some((c, d)); }
                }
                if let Some((c, _)) = ba { state.add_item(c); continue; }
            }
            {
                let mut bs: Option<(usize, usize, i32)> = None;
                for &rm in &worst_used {
                    let max_w = state.ch.weights[rm] + state.slack();
                    for &c in &best_unused {
                        if state.ch.weights[c] > max_w { continue; }
                        let d = state.contrib[c] - state.contrib[rm] - state.ch.interaction_values[c][rm];
                        if d > 0 && bs.map_or(true, |(_, _, bd)| d > bd) { bs = Some((c, rm, d)); }
                    }
                }
                if let Some((c, rm, _)) = bs { state.replace_item(rm, c); continue; }
            }
            let slack = state.slack();
            if slack >= 2 {
                let fits: Vec<usize> = best_unused.iter().copied().filter(|&i| state.ch.weights[i] < slack).collect();
                let m = fits.len();
                if m >= 2 {
                    let mut bp: Option<(usize, usize, i64)> = None;
                    for ai in 0..m {
                        let a = fits[ai]; let wa = state.ch.weights[a]; let ca = state.contrib[a] as i64;
                        for bi in (ai+1)..m {
                            let b = fits[bi];
                            if wa + state.ch.weights[b] > slack { continue; }
                            let d = ca + state.contrib[b] as i64 + state.ch.interaction_values[a][b] as i64;
                            if d > 0 && bp.map_or(true, |(_, _, bd)| d > bd) { bp = Some((a, b, d)); }
                        }
                    }
                    if let Some((a, b, _)) = bp { state.add_item(a); state.add_item(b); continue; }
                }
            }
            {
                let mut bs: Option<(usize, usize, i32)> = None;
                for &rm in &worst_used {
                    let w_rm = state.ch.weights[rm] as i64;
                    for &c in &best_unused {
                        let w_c = state.ch.weights[c] as i64;
                        if w_c >= w_rm { continue; }
                        let dw = (w_rm - w_c) as usize;
                        if dw == 0 || dw > 4 { continue; }
                        let d = state.contrib[c] - state.contrib[rm] - state.ch.interaction_values[c][rm];
                        if d > 0 && bs.map_or(true, |(_, _, bd)| d > bd) { bs = Some((c, rm, d)); }
                    }
                }
                if let Some((c, rm, _)) = bs { state.replace_item(rm, c); continue; }
            }
            if state.slack() > 0 {
                let mut bs: Option<(usize, usize, f64)> = None;
                for &rm in &worst_used {
                    let w_rm = state.ch.weights[rm] as i64;
                    for &c in &best_unused {
                        let w_c = state.ch.weights[c] as i64;
                        if w_c <= w_rm { continue; }
                        let dw = w_c - w_rm;
                        if dw as usize > 4 || state.slack() < dw as u32 { continue; }
                        let d = state.contrib[c] - state.contrib[rm] - state.ch.interaction_values[c][rm];
                        if d > 0 { let r = d as f64 / dw as f64; if bs.map_or(true, |(_, _, br)| r > br) { bs = Some((c, rm, r)); } }
                    }
                }
                if let Some((c, rm, _)) = bs { state.replace_item(rm, c); continue; }
            }
            break;
        }
    }

    fn perturb_by_strategy(state: &mut State, strength: usize, stall_count: usize, strategy: usize, rng: &mut Rng, hp: &Hparams) {
        let selected = state.selected_items();
        if selected.is_empty() { return; }
        let mut removal_candidates: Vec<(usize, i64)>;
        match strategy {
            0 => { removal_candidates = selected.iter().map(|&i| (i, state.contrib[i] as i64)).collect(); removal_candidates.sort_unstable_by_key(|&(_, c)| c); },
            1 => { removal_candidates = selected.iter().map(|&i| (i, -(state.ch.weights[i] as i64))).collect(); removal_candidates.sort_unstable_by_key(|&(_, w)| w); },
            2 => { removal_candidates = selected.iter().map(|&i| { let syn = state.contrib[i] as i64 - state.ch.values[i] as i64; (i, syn) }).collect(); removal_candidates.sort_unstable_by_key(|&(_, s)| s); },
            3 => { removal_candidates = selected.iter().map(|&i| { let w = (state.ch.weights[i] as i64).max(1); (i, (state.contrib[i] as i64 * 1000) / w) }).collect(); removal_candidates.sort_unstable_by_key(|&(_, s)| s); },
            4 => { removal_candidates = selected.iter().map(|&i| { let w = (state.ch.weights[i] as i64).max(1); let density = (state.contrib[i] as i64 * 100) / w; (i, state.ch.weights[i] as i64 - density) }).collect(); removal_candidates.sort_unstable_by_key(|&(_, s)| s); },
            5 => { removal_candidates = selected.iter().map(|&i| { let w = (state.ch.weights[i] as i64).max(1); (i, (state.contrib[i] as i64 * 10000) / (w * w)) }).collect(); removal_candidates.sort_unstable_by_key(|&(_, s)| s); },
            6 => { let seed = selected[rng.next_usize(selected.len())]; removal_candidates = selected.iter().map(|&i| { if i == seed { (i, i64::MIN) } else { (i, -(state.ch.interaction_values[i][seed] as i64)) } }).collect(); removal_candidates.sort_unstable_by_key(|&(_, s)| s); },
            _ => { removal_candidates = selected.iter().map(|&i| (i, -(state.contrib[i] as i64))).collect(); removal_candidates.sort_unstable_by_key(|&(_, s)| s); }
        }
        let base_remove = (selected.len() / hp.perturb_base_frac).max(2);
        let adaptive_mult = 1 + (stall_count / 2);
        let n_remove = (base_remove * adaptive_mult).min(strength).min(selected.len() * 2 / hp.perturb_max_frac);
        for j in 0..n_remove { if j < removal_candidates.len() { state.remove_item(removal_candidates[j].0); } }
    }

    fn greedy_reconstruct(state: &mut State, strategy: usize) {
        let n = state.ch.num_items; let cap = state.ch.max_weight;
        let mut candidates: Vec<usize> = (0..n).filter(|&i| !state.selected_bit[i]).collect();
        match strategy % 4 {
            0 => candidates.sort_unstable_by_key(|&i| -state.contrib[i]),
            1 => candidates.sort_unstable_by(|&a, &b| state.ch.weights[a].cmp(&state.ch.weights[b]).then(state.contrib[b].cmp(&state.contrib[a]))),
            2 => candidates.sort_unstable_by_key(|&i| { let syn: i64 = state.ch.interaction_values[i].iter().take(n.min(100)).map(|&v| v as i64).sum(); -(syn + state.contrib[i] as i64 / 10) }),
            _ => candidates.sort_unstable_by_key(|&i| { let w = (state.ch.weights[i] as i64).max(1); -((state.contrib[i] as i64 * 100) / w) }),
        }
        for &i in &candidates { if state.total_weight + state.ch.weights[i] <= cap { state.add_item(i); } }
    }

    fn vnd_dispatch(state: &mut State, hp: &Hparams) {
        if hp.window_k < state.ch.num_items { local_search_vnd_windowed(state, hp.window_k); }
        else { ils_vnd(state, hp); }
    }

    fn try_add_elite_t40(pool: &mut Vec<Vec<bool>>, bits: &[bool]) {
        let hamming_ok = pool.iter().all(|e| e.iter().zip(bits.iter()).filter(|(&a, &b)| a != b).count() > 20);
        if hamming_ok { if pool.len() >= 4 { pool.remove(0); } pool.push(bits.to_vec()); }
    }

    fn path_relink_t40(challenge: &Challenge, source_bits: &[bool], guide_bits: &[bool], hp: &Hparams, ch_dp: usize) -> Option<(i64, Vec<bool>)> {
        let n = challenge.num_items;
        let mut state = State::new_empty(challenge);
        for i in 0..n { if source_bits[i] { state.add_item(i); } }
        let mut to_add: Vec<usize> = (0..n).filter(|&i| guide_bits[i] && !source_bits[i]).collect();
        let mut to_remove: Vec<usize> = (0..n).filter(|&i| source_bits[i] && !guide_bits[i]).collect();
        let total_moves = to_add.len() + to_remove.len();
        if total_moves == 0 { return None; }
        let checkpoint_interval = (total_moves / 4).max(3);
        let cap = challenge.max_weight;
        let mut best_pr_val = state.total_value; let mut best_pr_bits = state.selected_bit.clone();
        let mut move_count = 0usize;
        while !to_add.is_empty() || !to_remove.is_empty() {
            let mut best_delta = i64::MIN; let mut best_action: Option<(bool, usize)> = None;
            for (idx, &item) in to_add.iter().enumerate() {
                if state.total_weight + challenge.weights[item] <= cap {
                    let delta = state.contrib[item] as i64;
                    if delta > best_delta { best_delta = delta; best_action = Some((true, idx)); }
                }
            }
            for (idx, &item) in to_remove.iter().enumerate() {
                let delta = -(state.contrib[item] as i64);
                if delta > best_delta { best_delta = delta; best_action = Some((false, idx)); }
            }
            match best_action {
                Some((true, idx))  => { let item = to_add[idx];    state.add_item(item);    to_add.swap_remove(idx); }
                Some((false, idx)) => { let item = to_remove[idx]; state.remove_item(item); to_remove.swap_remove(idx); }
                None => break,
            }
            move_count += 1;
            if state.total_weight <= cap && state.total_value > best_pr_val { best_pr_val = state.total_value; best_pr_bits = state.selected_bit.clone(); }
            if move_count % checkpoint_interval == 0 && state.total_weight <= cap {
                let mut tmp = State::new_empty(challenge);
                for i in 0..n { if state.selected_bit[i] { tmp.add_item(i); } }
                local_search_vnd_fast(&mut tmp);
                dp_refinement_hp(&mut tmp, ch_dp);
                if tmp.total_value > best_pr_val { best_pr_val = tmp.total_value; best_pr_bits = tmp.selected_bit.clone(); }
            }
        }
        let mut final_st = State::new_empty(challenge);
        for i in 0..n { if best_pr_bits[i] { final_st.add_item(i); } }
        loop {
            let v_before = final_st.total_value;
            local_search_vnd_windowed(&mut final_st, hp.window_k);
            dp_refinement_hp(&mut final_st, hp.core_half_dp);
            if final_st.total_value <= v_before { break; }
        }
        if final_st.total_value > best_pr_val { best_pr_val = final_st.total_value; best_pr_bits = final_st.selected_bit.clone(); }
        if best_pr_val > 0 { Some((best_pr_val, best_pr_bits)) } else { None }
    }

// First pass seeded by bound-and-prune

    fn run_bp_seeded_instance(challenge: &Challenge, hp: &Hparams) -> SolState {
        let inst = build_phase1(challenge);
        let qr   = bound_and_prune(&inst, hp.n_lambda_values);
        let bp_items = &qr.results[0].selected_items;

        let mut state = State::new_empty(challenge);
        for &i in bp_items {
            if state.total_weight + challenge.weights[i] <= challenge.max_weight {
                state.add_item(i);
            }
        }
        dp_refinement_hp(&mut state, hp.core_half_dp);
        vnd_dispatch(&mut state, hp);
        state.clone_solution()
    }

// ILS instance

    fn run_one_instance(challenge: &Challenge, hp: &Hparams, rng_offset: usize) -> Solution {
        let n = challenge.num_items;
        let mut rng = Rng::from_seed(&challenge.seed);
        for _ in 0..rng_offset * 100 { rng.next_u32(); }
        let ch = hp.core_half_dp;

        let mut population: Vec<SolState> = Vec::with_capacity(16);

        let n_greedy = if n <= 1200 { 4 } else { 3 };
        for variant in 0..n_greedy {
            let mut st = State::new_empty(challenge);
            match variant {
                0 => build_greedy_density(&mut st),
                1 => build_greedy_value(&mut st),
                2 => build_greedy_synergy_weight(&mut st),
                _ => build_greedy_hub(&mut st),
            }
            dp_refinement_hp(&mut st, ch);
            if hp.use_heavy_polish { local_search_vnd_heavy(&mut st); } else { vnd_dispatch(&mut st, hp); }
            population.push(st.clone_solution());
        }

        for mode in 4..(4 + hp.n_random_starts) {
            let mut st = State::new_empty(challenge);
            let m = if mode < 6 { mode } else { mode - 2 };
            construct_forward_incremental(&mut st, m, &mut rng);
            dp_refinement_hp(&mut st, ch);
            vnd_dispatch(&mut st, hp);
            population.push(st.clone_solution());
        }

        if hp.use_hub_pair {
            for k in 0..4 {
                let mut st = State::new_empty(challenge);
                build_hub_pair_kth(&mut st, k);
                dp_refinement_hp(&mut st, ch);
                vnd_dispatch(&mut st, hp);
                population.push(st.clone_solution());
            }
        }

        population.sort_unstable_by_key(|s| std::cmp::Reverse(s.value));
        {
            let mut unique = Vec::with_capacity(8); let mut seen = Vec::with_capacity(8);
            for p in population.drain(..) {
                let mut h: u64 = 0;
                for (i, &b) in p.bits.iter().enumerate() { if b { let mut v: u64 = 0x517CC1B727220A95; v ^= (i as u64).wrapping_mul(0x9E3779B97F4A7C15); v = v.rotate_left(17).wrapping_mul(0xBF58476D1CE4E5B9); h ^= v; } }
                if !seen.contains(&h) { seen.push(h); unique.push(p); if unique.len() == 8 { break; } }
            }
            population = unique;
        }

        let mut state = State::new_empty(challenge);
        for gen in 0..hp.n_crossover_gen {
            let child_bits = crossover_frequency(&population, challenge, &mut rng);
            set_state_from_bits(&mut state, &child_bits);
            dp_refinement_hp(&mut state, ch); vnd_dispatch(&mut state, hp);
            population.push(state.clone_solution());
            if population.len() >= 2 {
                let a = gen % population.len().min(4); let b = (gen + 1) % population.len().min(4);
                if a != b {
                    let child_bits = crossover_uniform(&population[a], &population[b], challenge, &mut rng);
                    set_state_from_bits(&mut state, &child_bits);
                    dp_refinement_hp(&mut state, ch); vnd_dispatch(&mut state, hp);
                    population.push(state.clone_solution());
                }
            }
            population.sort_unstable_by_key(|s| std::cmp::Reverse(s.value));
            {
                let mut unique = Vec::with_capacity(8); let mut seen = Vec::with_capacity(8);
                for p in population.drain(..) {
                    let mut h: u64 = 0;
                    for (i, &b) in p.bits.iter().enumerate() { if b { let mut v: u64 = 0x517CC1B727220A95; v ^= (i as u64).wrapping_mul(0x9E3779B97F4A7C15); v = v.rotate_left(17).wrapping_mul(0xBF58476D1CE4E5B9); h ^= v; } }
                    if !seen.contains(&h) { seen.push(h); unique.push(p); if unique.len() == 8 { break; } }
                }
                population = unique;
            }
        }

        if hp.sa_rounds > 0 {
            for pi in 0..hp.n_sa_members.min(population.len()) {
                state.restore_solution(&population[pi]);
                simulated_annealing(&mut state, &mut rng, hp.sa_rounds, hp.sa_iter);
                vnd_dispatch(&mut state, hp);
                let sol = state.clone_solution();
                if sol.value > population[pi].value { population.push(sol); }
            }
            population.sort_unstable_by_key(|s| std::cmp::Reverse(s.value));
            {
                let mut unique = Vec::with_capacity(8); let mut seen = Vec::with_capacity(8);
                for p in population.drain(..) {
                    let mut h: u64 = 0;
                    for (i, &b) in p.bits.iter().enumerate() { if b { let mut v: u64 = 0x517CC1B727220A95; v ^= (i as u64).wrapping_mul(0x9E3779B97F4A7C15); v = v.rotate_left(17).wrapping_mul(0xBF58476D1CE4E5B9); h ^= v; } }
                    if !seen.contains(&h) { seen.push(h); unique.push(p); if unique.len() == 8 { break; } }
                }
                population = unique;
            }
        }

        state.restore_solution(&population[0]);
        let mut best_val = state.total_value;
        let mut best_sel: Vec<usize> = state.selected_items();

        let mut tabu_hashes: Vec<u64> = Vec::with_capacity(128);
                let zobrist_table: Vec<u64> = (0..n).map(|i| {
            let mut h: u64 = 0x517CC1B727220A95;
            h ^= (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
            h = h.rotate_left(17).wrapping_mul(0xBF58476D1CE4E5B9);
            h
        }).collect();
        let compute_hash = |bits: &[bool]| -> u64 {
            let mut h: u64 = 0;
            for i in 0..n { if bits[i] { h ^= zobrist_table[i]; } }
            h
        };
        tabu_hashes.push(compute_hash(&state.selected_bit));

        let mut stall_count = 0;
        let mut elite_pool: Vec<Vec<bool>> = Vec::new();
        try_add_elite_t40(&mut elite_pool, &state.selected_bit);

        for round in 0..hp.ils_rounds {
            let snap = state.clone_solution();

            dp_refinement_hp(&mut state, ch);
            vnd_dispatch(&mut state, hp);

            if state.total_value > best_val {
                best_val = state.total_value;
                best_sel = state.selected_items();
                stall_count = 0;
                try_add_elite_t40(&mut elite_pool, &state.selected_bit);
            }

            if state.total_value <= snap.value {
                state.restore_solution(&snap);
                stall_count += 1;

                if hp.ils_restart_interval > 0
                    && stall_count > 0
                    && stall_count % hp.ils_restart_interval == 0
                {
                    let pi = (stall_count / hp.ils_restart_interval) % population.len();
                    state.restore_solution(&population[pi]);
                }

                let strategy = round % 8;
                let strength  = 5 + round / 4;
                perturb_by_strategy(&mut state, strength, stall_count, strategy, &mut rng, hp);
                greedy_reconstruct(&mut state, strategy);
                vnd_dispatch(&mut state, hp);

                let h = compute_hash(&state.selected_bit);
                if tabu_hashes.contains(&h) {
                    let extra_strength = 10 + round / 3;
                    perturb_by_strategy(&mut state, extra_strength, stall_count + 3, 6, &mut rng, hp);
                    greedy_reconstruct(&mut state, 0);
                    vnd_dispatch(&mut state, hp);
                }
                let h2 = compute_hash(&state.selected_bit);
                if tabu_hashes.len() < 128 { tabu_hashes.push(h2); }
                else { tabu_hashes[round % 128] = h2; }

                if state.total_value > best_val {
                    best_val = state.total_value;
                    best_sel = state.selected_items();
                    stall_count = 0;
                }
            } else {
                stall_count = 0;
                let h = compute_hash(&state.selected_bit);
                if tabu_hashes.len() < 128 { tabu_hashes.push(h); }
            }
        }

        let mut final_state = State::new_empty(challenge);
        for &i in &best_sel { final_state.add_item(i); }

        if hp.use_heavy_polish {
            loop {
                let v_before = final_state.total_value;
                local_search_vnd_heavy(&mut final_state);
                dp_refinement_hp(&mut final_state, ch);
                if final_state.total_value <= v_before { break; }
            }
        } else {
            loop {
                let v_before = final_state.total_value;
                local_search_vnd_windowed(&mut final_state, hp.window_k);
                dp_refinement_hp(&mut final_state, ch);
                if final_state.total_value <= v_before { break; }
            }
        }

        if elite_pool.len() >= 2 {
            let best_bits: Vec<bool> = final_state.selected_bit.clone();
            let mut pr_best_val  = final_state.total_value;
            let mut pr_best_bits = best_bits.clone();
            for guide_bits in &elite_pool {
                if *guide_bits == best_bits { continue; }
                if let Some((pv, pb)) = path_relink_t40(challenge, &best_bits, guide_bits, hp, ch) {
                    if pv > pr_best_val { pr_best_val = pv; pr_best_bits = pb; }
                }
                if let Some((pv, pb)) = path_relink_t40(challenge, guide_bits, &best_bits, hp, ch) {
                    if pv > pr_best_val { pr_best_val = pv; pr_best_bits = pb; }
                }
            }
            if pr_best_val > final_state.total_value {
                for i in (0..n).rev() { if final_state.selected_bit[i] { final_state.remove_item(i); } }
                for i in 0..n { if pr_best_bits[i] { final_state.add_item(i); } }
            }
        }

        if final_state.total_value > best_val {
            Solution { items: final_state.selected_items() }
        } else {
            Solution { items: best_sel }
        }
    } // end run_one_instance

// Solver entry

    pub struct Solver;

    impl Solver {
        pub fn solve(
            challenge: &Challenge,
            _save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
            hyperparameters: &Option<Map<String, Value>>,
        ) -> Result<Option<Solution>> {
            let n = challenge.num_items;
            let sum_w: u64 = challenge.weights.iter().map(|&w| w as u64).sum();
            let budget_pct = if sum_w > 0 {
                ((challenge.max_weight as u64) * 100 / sum_w) as u32
            } else { 10 };
            let hp = Hparams::from_map(hyperparameters, n, budget_pct);
            let n_restarts = hp.n_full_restarts.max(1);

            let mut best_sol: Option<Solution> = None;
            let mut best_quality: i64 = i64::MIN;

            // pass 0: seeded from the bound-and-prune solution
            {
                let bp_sol = run_bp_seeded_instance(challenge, &hp);
                let val = bp_sol.value;
                if val > best_quality {
                    best_quality = val;
                    let items: Vec<usize> = (0..n)
                        .filter(|&i| bp_sol.bits[i])
                        .collect();
                    best_sol = Some(Solution { items });
                }
            }

            // remaining passes: independent ILS restarts
            for restart in 0..(n_restarts.saturating_sub(1)) {
                let sol = run_one_instance(challenge, &hp, restart);
                let mut val: i64 = 0;
                for &i in &sol.items {
                    val += challenge.values[i] as i64;
                    for &j in &sol.items {
                        if j > i { val += challenge.interaction_values[i][j] as i64; }
                    }
                }
                if val > best_quality {
                    best_quality = val;
                    best_sol = Some(sol);
                }
            }

            Ok(best_sol)
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

} // end mod inner

// Public entry point

#[inline(always)]
pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hp: &Option<Map<String, Value>>,
) -> Result<()> {
    inner::solve_challenge(challenge, save_solution, hp)
}