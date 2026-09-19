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
        pub rush_mode:       Option<usize>,
        pub rush_passes:     Option<usize>,
        pub rush_starts:     Option<usize>,
        pub rush_fast:       Option<usize>,
        pub rush_pool:       Option<usize>,
        pub rush_polish:     Option<usize>,
        pub stage_stop:      Option<usize>,
        pub vnd_cache:       Option<usize>,
        pub vnd_resume:      Option<usize>,
        pub no_synergy:      Option<usize>,
        pub syn_bound:       Option<usize>,
        pub ils_stall_stop:  Option<usize>,
        pub pert_ramp:       Option<usize>,
        pub bp_fast:         Option<usize>,
        pub prof_dup:        Option<usize>,
        pub vnd_fast:        Option<usize>,
        pub core_lo: Option<usize>, pub core_hi: Option<usize>,
        pub core_work: Option<usize>, pub core_lam: Option<usize>,
        pub core_fix: Option<usize>,
        pub core_kmax: Option<usize>,
        pub core_pre: Option<usize>,
        pub rng_stream: Option<usize>,
    }

    struct Hparams {
        n_lambda_values: usize,
        ils_rounds:      usize,
        window_k:        usize,
        core_half_dp:    usize,
        rush_mode:       usize,
        rush_passes:     usize,
        rush_starts:     usize,
        rush_fast:       usize,
        rush_pool:       usize,
        rush_polish:     usize,
        stage_stop:      usize,
        vnd_cache:       usize,
        vnd_resume:      usize,
        no_synergy:      usize,
        syn_bound:       usize,
        ils_stall_stop:  usize,
        pert_ramp:       usize,
        bp_fast:         usize,
        prof_dup:        usize,
        vnd_fast:        usize,
        core_lo: usize, core_hi: usize, core_work: usize, core_lam: usize, core_fix: usize, core_kmax: usize, core_pre: usize,
        rng_stream: usize,
    }

    impl Hparams {
        fn for_size(n: usize, budget: u32) -> Self {
            if n <= 1200 {
                if budget <= 10 {
                    Self {
                        n_lambda_values: 800,
                        ils_rounds:      9,
                        window_k:        150,
                        core_half_dp:    50,
                        rush_mode:       0,
                        rush_passes:     6,
                        rush_starts:     5,
                        rush_fast:       0,
                        rush_pool:       0,
                        rush_polish:     0,
                        stage_stop:      0,
                        vnd_cache:       1,
                        vnd_resume:      1,
                        no_synergy:      0,
                        syn_bound:       1,
                        ils_stall_stop:  0,
                        pert_ramp:       0,
                        bp_fast:         79,
                        prof_dup:        0,
                        vnd_fast:        531699872,
                        core_lo: 10, core_hi: 10, core_work: 6000000, core_lam: 2, core_fix: 0, core_kmax: 180, core_pre: 2, rng_stream: 0,
                    }
                } else {
                    Self {
                        n_lambda_values: 1600,
                        ils_rounds:      10,
                        window_k:        250,
                        core_half_dp:    50,
                        rush_mode:       0,
                        rush_passes:     6,
                        rush_starts:     5,
                        rush_fast:       0,
                        rush_pool:       0,
                        rush_polish:     0,
                        stage_stop:      0,
                        vnd_cache:       0,
                        vnd_resume:      0,
                        no_synergy:      0,
                        syn_bound:       0,
                        ils_stall_stop:  0,
                        pert_ramp:       0,
                        bp_fast:         7,
                        prof_dup:        0,
                        vnd_fast:        32,
                        core_lo: 0, core_hi: 0, core_work: 0, core_lam: 2, core_fix: 0, core_kmax: 100000, core_pre: 0, rng_stream: 0,
                    }
                }
            } else {
                Self {
                    n_lambda_values: 1600,
                    ils_rounds:      40,
                    window_k:        250,
                    core_half_dp:    50,
                    rush_mode:       0,
                    rush_passes:     6,
                    rush_starts:     5,
                    rush_fast:       0,
                    rush_pool:       0,
                    rush_polish:     0,
                    stage_stop:      0,
                    vnd_cache:       0,
                    vnd_resume:      0,
                    no_synergy:      0,
                    syn_bound:       0,
                    ils_stall_stop:  0,
                    pert_ramp:       0,
                    bp_fast:         7,
                    prof_dup:        0,
                    vnd_fast:        32,
                    core_lo: 0, core_hi: 0, core_work: 0, core_lam: 2, core_fix: 0, core_kmax: 100000, core_pre: 0, rng_stream: 0,
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
                if let Some(v) = m.get("rush_mode").and_then(|v| v.as_u64()) {
                    p.rush_mode = v as usize;
                }
                if let Some(v) = m.get("rush_passes").and_then(|v| v.as_u64()) {
                    p.rush_passes = v as usize;
                }
                if let Some(v) = m.get("rush_starts").and_then(|v| v.as_u64()) {
                    p.rush_starts = v as usize;
                }
                if let Some(v) = m.get("rush_fast").and_then(|v| v.as_u64()) {
                    p.rush_fast = v as usize;
                }
                if let Some(v) = m.get("rush_pool").and_then(|v| v.as_u64()) {
                    p.rush_pool = v as usize;
                }
                if let Some(v) = m.get("rush_polish").and_then(|v| v.as_u64()) {
                    p.rush_polish = v as usize;
                }
                if let Some(v) = m.get("stage_stop").and_then(|v| v.as_u64()) {
                    p.stage_stop = v as usize;
                }
                if let Some(v) = m.get("vnd_cache").and_then(|v| v.as_u64()) {
                    p.vnd_cache = v as usize;
                }
                if let Some(v) = m.get("vnd_resume").and_then(|v| v.as_u64()) {
                    p.vnd_resume = v as usize;
                }
                if let Some(v) = m.get("no_synergy").and_then(|v| v.as_u64()) {
                    p.no_synergy = v as usize;
                }
                if let Some(v) = m.get("syn_bound").and_then(|v| v.as_u64()) {
                    p.syn_bound = v as usize;
                }
                if let Some(v) = m.get("ils_stall_stop").and_then(|v| v.as_u64()) {
                    p.ils_stall_stop = v as usize;
                }
                if let Some(v) = m.get("pert_ramp").and_then(|v| v.as_u64()) {
                    p.pert_ramp = v as usize;
                }
                if let Some(v) = m.get("bp_fast").and_then(|v| v.as_u64()) {
                    p.bp_fast = v as usize;
                }
                if let Some(v) = m.get("prof_dup").and_then(|v| v.as_u64()) {
                    p.prof_dup = v as usize;
                }
                if let Some(v) = m.get("vnd_fast").and_then(|v| v.as_u64()) {
                    p.vnd_fast = v as usize;
                }
                if let Some(v) = m.get("core_lo").and_then(|v| v.as_u64()) { p.core_lo = v as usize; }
                if let Some(v) = m.get("core_hi").and_then(|v| v.as_u64()) { p.core_hi = v as usize; }
                if let Some(v) = m.get("core_work").and_then(|v| v.as_u64()) { p.core_work = v as usize; }
                if let Some(v) = m.get("core_lam").and_then(|v| v.as_u64()) { p.core_lam = v as usize; }
                if let Some(v) = m.get("core_fix").and_then(|v| v.as_u64()) { p.core_fix = v as usize; }
                if let Some(v) = m.get("core_kmax").and_then(|v| v.as_u64()) { p.core_kmax = v as usize; }
                if let Some(v) = m.get("core_pre").and_then(|v| v.as_u64()) { p.core_pre = v as usize; }
                if let Some(v) = m.get("rng_stream").and_then(|v| v.as_u64()) { p.rng_stream = v as usize; }
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
    pub struct QKPResult { pub results: Vec<BudgetResult>, pub bp: Vec<i32>, pub bcross: i32 }

    pub fn challenge_to_p1(challenge: &Challenge) -> P1Instance {
        let n = challenge.num_items;
        let mut edges: Vec<Edge> = Vec::new();

        let dense_shape = challenge.interaction_values.get(..n)
            .map_or(false, |rows| rows.iter().all(|row| row.len() >= n));
        if dense_shape {
            let max_vec_len = (isize::MAX as usize) / 24;
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
            pub max_bucket:           i32,
            pub max_degree_ratio:     f32,
            pub step:                 f32,
            pub src_ord:              Vec<u32>,
            pub src_rr:               Vec<f32>,
            pub src_live:             Vec<u32>,
            pub src_p:                usize,
            pub snk_ord:              Vec<u32>,
            pub snk_rr:               Vec<f32>,
            pub snk_live:             Vec<u32>,
            pub snk_p:                usize,
            pub f_bucket:             bool,
            pub f_prefix:             bool,
            pub f_fast:               bool,
            pub f_down:               bool,
            pub f_stop:               bool,
            pub stop_w:               f64,
            pub lifted_w:             f64,
            pub ext_pm:               i32,
            pub stop_at:              i32,
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
            if (*s).f_stop && (*current).number >= 3 { (*s).lifted_w += (*current).cst as f64; }
            loop {
                while !(*current).next_scan.is_null() {
                    let temp             = (*current).next_scan;
                    (*current).next_scan = (*(*current).next_scan).next;
                    current              = temp;
                    (*current).next_scan = (*current).child_list;
                    (*s).label_count[(*current).label as usize] -= 1;
                    (*current).label      = (*s).num_nodes;
                    (*current).breakpoint = theparam + 1;
                    if (*s).f_stop && (*current).number >= 3 { (*s).lifted_w += (*current).cst as f64; }
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
            if lbl as i32 > (*s).max_bucket { (*s).max_bucket = lbl as i32; }
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
            if lbl as i32 > (*s).max_bucket { (*s).max_bucket = lbl as i32; }
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
                if lbl as i32 > (*s).max_bucket { (*s).max_bucket = lbl as i32; }
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
            if lbl as i32 > (*s).max_bucket { (*s).max_bucket = lbl as i32; }
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
                        if (*s).f_down && i < (*s).max_bucket { (*s).max_bucket = i; }
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
                if lbl as i32 > (*s).max_bucket { (*s).max_bucket = lbl as i32; }
                add_to_strong_bucket(sr, (*s).strong_roots[lbl].end);
            }
            (*s).highest_strong_label = 1;
            let sr = (*(*s).strong_roots[1].start).next;
            (*(*sr).next).prev = (*sr).prev;
            (*(*sr).prev).next = (*sr).next;
            (*sr).next = std::ptr::null_mut();
            sr
        }

        #[inline]
        fn live_insert(v: &mut Vec<u32>, x: u32) {
            let mut lo = 0usize;
            let mut hi = v.len();
            while lo < hi {
                let mid = (lo + hi) >> 1;
                if v[mid] < x { lo = mid + 1; } else { hi = mid; }
            }
            v.insert(lo, x);
        }
        #[inline]
        fn live_remove(v: &mut Vec<u32>, x: u32) {
            let mut lo = 0usize;
            let mut hi = v.len();
            while lo < hi {
                let mid = (lo + hi) >> 1;
                if v[mid] < x { lo = mid + 1; } else { hi = mid; }
            }
            if lo < v.len() && v[lo] == x { v.remove(lo); }
        }

        unsafe fn update_capacities(s: *mut HpfState, theparam: i32) {
            let src_idx = ((*s).source - 1) as usize;
            let snk_idx = ((*s).sink   - 1) as usize;

            if (*s).f_prefix {
                let lambda = (*s).max_degree_ratio - theparam as f32 * (*s).step;
                let guard  = lambda - (*s).step;
                let ns     = (*s).src_ord.len();
                while (*s).src_p < ns && (*s).src_rr[(*s).src_p] > guard {
                    let pos = (*s).src_ord[(*s).src_p];
                    live_insert(&mut (*s).src_live, pos);
                    (*s).src_p += 1;
                }
                let hi = lambda + (*s).step;
                while (*s).snk_p > 0 && (*s).snk_rr[(*s).snk_p - 1] >= hi {
                    (*s).snk_p -= 1;
                    let pos = (*s).snk_ord[(*s).snk_p];
                    live_remove(&mut (*s).snk_live, pos);
                }
            }

            if (*s).f_fast {
                let lambda = (*s).max_degree_ratio - theparam as f32 * (*s).step;
                let nn     = (*s).num_nodes;
                let sbase  = (*s).adjacency_list[src_idx].out_of_tree.as_ptr();
                if (*s).f_prefix {
                    let lb = (*s).src_live.as_ptr();
                    let nl = (*s).src_live.len();
                    for i in 0..nl {
                        let arc = *sbase.add(*lb.add(i) as usize);
                        let nd  = (*arc).to;
                        let v   = (*nd).wt - lambda * (*nd).cst;
                        let new_capacity = if v > 0.0 { v } else { 0.0 };
                        let delta = new_capacity - (*arc).capacity;
                        (*arc).capacity = new_capacity;
                        (*arc).flow    += delta;
                        let ex = (*nd).excess + delta;
                        (*nd).excess    = ex;
                        if ex > 0.0 && (*nd).label < nn { push_excess(s, nd); }
                    }
                } else {
                    let size = (*s).adjacency_list[src_idx].num_out_of_tree as usize;
                    for i in 0..size {
                        let arc = *sbase.add(i);
                        let nd  = (*arc).to;
                        let v   = (*nd).wt - lambda * (*nd).cst;
                        let new_capacity = if v > 0.0 { v } else { 0.0 };
                        let delta = new_capacity - (*arc).capacity;
                        (*arc).capacity = new_capacity;
                        (*arc).flow    += delta;
                        let ex = (*nd).excess + delta;
                        (*nd).excess    = ex;
                        if ex > 0.0 && (*nd).label < nn { push_excess(s, nd); }
                    }
                }
                let kbase = (*s).adjacency_list[snk_idx].out_of_tree.as_ptr();
                if (*s).f_prefix {
                    let lb = (*s).snk_live.as_ptr();
                    let nl = (*s).snk_live.len();
                    for i in 0..nl {
                        let arc = *kbase.add(*lb.add(i) as usize);
                        let nd  = (*arc).from;
                        let v   = lambda * (*nd).cst - (*nd).wt;
                        let new_capacity = if v > 0.0 { v } else { 0.0 };
                        let delta = new_capacity - (*arc).capacity;
                        (*arc).capacity = new_capacity;
                        (*arc).flow    += delta;
                        let ex = (*nd).excess - delta;
                        (*nd).excess    = ex;
                        if ex > 0.0 && (*nd).label < nn { push_excess(s, nd); }
                    }
                } else {
                    let size = (*s).adjacency_list[snk_idx].num_out_of_tree as usize;
                    for i in 0..size {
                        let arc = *kbase.add(i);
                        let nd  = (*arc).from;
                        let v   = lambda * (*nd).cst - (*nd).wt;
                        let new_capacity = if v > 0.0 { v } else { 0.0 };
                        let delta = new_capacity - (*arc).capacity;
                        (*arc).capacity = new_capacity;
                        (*arc).flow    += delta;
                        let ex = (*nd).excess - delta;
                        (*nd).excess    = ex;
                        if ex > 0.0 && (*nd).label < nn { push_excess(s, nd); }
                    }
                }
            } else if (*s).f_prefix {
                let nl = (*s).src_live.len();
                for i in 0..nl {
                    let pos   = (*s).src_live[i] as usize;
                    let arc   = (*s).adjacency_list[src_idx].out_of_tree[pos];
                    let delta = (*arc).capacities[theparam as usize] - (*arc).capacity;
                    if delta < 0.0 { return; }
                    (*arc).capacity     += delta;
                    (*arc).flow         += delta;
                    (*(*arc).to).excess += delta;
                    if (*(*arc).to).label < (*s).num_nodes && (*(*arc).to).excess > 0.0 {
                        push_excess(s, (*arc).to);
                    }
                }
                let nl = (*s).snk_live.len();
                for i in 0..nl {
                    let pos   = (*s).snk_live[i] as usize;
                    let arc   = (*s).adjacency_list[snk_idx].out_of_tree[pos];
                    let delta = (*arc).capacities[theparam as usize] - (*arc).capacity;
                    if delta > 0.0 { return; }
                    (*arc).capacity       += delta;
                    (*arc).flow           += delta;
                    (*(*arc).from).excess -= delta;
                    if (*(*arc).from).label < (*s).num_nodes && (*(*arc).from).excess > 0.0 {
                        push_excess(s, (*arc).from);
                    }
                }
            } else {
                let size = (*s).adjacency_list[src_idx].num_out_of_tree as usize;
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
                let size = (*s).adjacency_list[snk_idx].num_out_of_tree as usize;
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
            }
            if (*s).f_bucket {
                let hi = (*s).num_nodes - 1;
                (*s).highest_strong_label = if (*s).max_bucket < hi { (*s).max_bucket } else { hi };
            } else {
                (*s).highest_strong_label = (*s).num_nodes - 1;
            }
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
                    if (*s).max_bucket < 1 { (*s).max_bucket = 1; }
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
            if (*s).f_stop && (*s).lifted_w >= (*s).stop_w && past_stop(s, 0) { return; }
            theparam = 1;
            while theparam < (*s).num_params {
                update_capacities(s, theparam);
                loop {
                    let sr = get_highest_strong_root(s, theparam);
                    if sr.is_null() { break; }
                    process_root(s, sr);
                }
                if (*s).f_stop && (*s).lifted_w >= (*s).stop_w && past_stop(s, theparam - 1) { return; }
                theparam += 1;
            }
        }

        #[inline]
        unsafe fn past_stop(s: *mut HpfState, done: i32) -> bool {
            if (*s).ext_pm <= 0 { return true; }
            if (*s).stop_at < 0 {
                let rem = ((*s).num_params - done).max(0) as i64;
                let ext = (rem * (*s).ext_pm as i64 + 999) / 1000;
                (*s).stop_at = done + ext as i32;
            }
            done >= (*s).stop_at
        }

        pub struct BreakpointSets { pub sets: Vec<(i32, Vec<usize>)> }

        pub fn get_breakpoints(inst: &P1Instance,
                                n_lambda_values: usize,
                                bp_fast: usize,
                                ext_pm: i32) -> BreakpointSets {
            let f_bucket = (bp_fast & 1) != 0;
            let f_prefix = (bp_fast & 2) != 0;
            let f_fast   = (bp_fast & 4) != 0;
            let f_stop   = (bp_fast & 8) != 0;
            let f_down   = (bp_fast & 16) != 0;
            let f_lazybp = (bp_fast & 32) != 0;
            let _ = f_lazybp;
            let f_reserve = (bp_fast & 64) != 0;
            let stop_w = inst.budgets.iter().fold(0i32, |a, &b| if b > a { b } else { a }) as f64;
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
                    capacities: if f_fast { Vec::new() } else { vec![cap] },
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
            let params: Vec<f32> = if f_fast {
                Vec::new()
            } else {
                (0..num_params as usize).map(|i| max_degree_ratio - i as f32 * step).collect()
            };
            let lambda0 = max_degree_ratio;

            for i in 2..num_nodes as usize {
                let wt  = adjacency_list[i].wt;
                let cst = adjacency_list[i].cst;
                let src_caps: Vec<f32> = if f_fast {
                    Vec::new()
                } else {
                    params.iter()
                        .map(|&p| { let v = wt - p * cst; if v > 0.0 { v } else { 0.0 } })
                        .collect()
                };
                let src_capacity = if f_fast {
                    if num_params > 0 { let v = wt - lambda0 * cst; if v > 0.0 { v } else { 0.0 } }
                    else { 0.0 }
                } else {
                    src_caps.first().copied().unwrap_or(0.0)
                };
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

                let snk_caps: Vec<f32> = if f_fast {
                    Vec::new()
                } else {
                    params.iter()
                        .map(|&p| { let v = p * cst - wt; if v > 0.0 { v } else { 0.0 } })
                        .collect()
                };
                let snk_capacity = if f_fast {
                    if num_params > 0 { let v = lambda0 * cst - wt; if v > 0.0 { v } else { 0.0 } }
                    else { 0.0 }
                } else {
                    snk_caps.first().copied().unwrap_or(0.0)
                };
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
                max_bucket: 1,
                max_degree_ratio, step,
                src_ord: Vec::new(), src_rr: Vec::new(), src_live: Vec::new(), src_p: 0,
                snk_ord: Vec::new(), snk_rr: Vec::new(), snk_live: Vec::new(), snk_p: 0,
                f_bucket, f_prefix, f_fast,
                f_down, f_stop, stop_w, lifted_w: 0.0,
                ext_pm, stop_at: -1,
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

                if f_reserve {
                    for i in 0..num_nodes as usize {
                        let need = (*s).adjacency_list[i].num_adjacent as usize;
                        (*s).adjacency_list[i].out_of_tree.reserve_exact(need);
                    }
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

                if f_prefix {
                    let si = ((*s).source - 1) as usize;
                    let ns = (*s).adjacency_list[si].num_out_of_tree as usize;
                    let mut key: Vec<(f32, u32)> = Vec::with_capacity(ns);
                    for i in 0..ns {
                        let nd = (*(*s).adjacency_list[si].out_of_tree[i]).to;
                        key.push(((*nd).wt / (*nd).cst, i as u32));
                    }
                    key.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
                    for &(r, i) in key.iter() { (*s).src_rr.push(r); (*s).src_ord.push(i); }
                    (*s).src_p = 0;
                    (*s).src_live = Vec::with_capacity(ns);

                    let ki = ((*s).sink - 1) as usize;
                    let nk = (*s).adjacency_list[ki].num_out_of_tree as usize;
                    let mut key: Vec<(f32, u32)> = Vec::with_capacity(nk);
                    for i in 0..nk {
                        let nd = (*(*s).adjacency_list[ki].out_of_tree[i]).from;
                        key.push(((*nd).wt / (*nd).cst, i as u32));
                    }
                    key.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
                    for &(r, i) in key.iter() { (*s).snk_rr.push(r); (*s).snk_ord.push(i); }
                    (*s).snk_p = nk;
                    (*s).snk_live = (0..nk as u32).collect();
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

    fn stage_stop_pick(weights: &[i32], budget: i32, tag: u64) -> Option<Solution> {
        let n = weights.len();
        if n == 0 { return Some(Solution { items: Vec::new() }); }
        let start = (tag as usize) % n;
        let mut best: Option<usize> = None;
        let mut best_w = -1i32;
        for k in 0..n {
            let i = (start + k) % n;
            if weights[i] <= budget && weights[i] > best_w {
                best_w = weights[i];
                best = Some(i);
            }
        }
        Some(Solution { items: best.into_iter().collect() })
    }

    pub fn run_bp_algorithm(inst: &P1Instance, n_lambda_values: usize,
                            stage_stop: usize, bp_tag: &mut u64, bp_fast: usize,
                            prof_dup: usize, ext_pm: i32) -> QKPResult {
        if prof_dup & 64 != 0 {
            let d = get_breakpoints(inst, n_lambda_values, bp_fast, 0);
            *bp_tag = (*bp_tag).wrapping_add(d.sets.len() as u64);
        }
        if prof_dup & 256 != 0 {
            let d = get_breakpoints(inst, 1, bp_fast, 0);
            *bp_tag = (*bp_tag).wrapping_add(d.sets.len() as u64);
        }
        let bps = get_breakpoints(inst, n_lambda_values, bp_fast, ext_pm);
        let n_breakpoints = bps.sets.len();
        if stage_stop == 2 {
            let mut t = n_breakpoints as u64;
            for (lv, nodes) in bps.sets.iter() { t = t.wrapping_add(*lv as u64 ^ nodes.len() as u64); }
            *bp_tag = t;
            return QKPResult { results: Vec::new(), bp: Vec::new(), bcross: -1 };
        }

        let mut total_weights_at_bp = vec![0.0f64; n_breakpoints];
        {
            let mut cumsum = 0.0f64;
            for (i, (_, nodes)) in bps.sets.iter().enumerate() {
                for &nd in nodes { cumsum += inst.weights[nd] as f64; }
                total_weights_at_bp[i] = cumsum;
            }
        }

        let n_bp_total = n_breakpoints + 1;
        let mut bp_weights:  Vec<f64>      = Vec::with_capacity(n_bp_total);
        bp_weights.push(0.0);
        for i in 0..n_breakpoints { bp_weights.push(total_weights_at_bp[i]); }

        let mut breakpoints: Vec<IntArray> = Vec::with_capacity(n_bp_total);
        if bp_fast & 32 != 0 {
            let mut needed = vec![false; n_bp_total];
            for bi in 0..inst.n_budgets {
                let budget = inst.budgets[bi] as f64;
                let mut l = 0usize;
                for k in 0..n_bp_total { if bp_weights[k] <= budget { l = k; } }
                needed[l] = true;
                let mut r = n_bp_total - 1;
                for k in 0..n_bp_total { if bp_weights[k] >= budget { r = k; break; } }
                needed[r] = true;
            }
            for _ in 0..n_bp_total { breakpoints.push(Vec::new()); }
            let mut cur: IntArray = Vec::new();
            if needed[0] { breakpoints[0] = cur.clone(); }
            for i in 0..n_breakpoints {
                for &nd in &bps.sets[i].1 { cur.push(nd); }
                if needed[i + 1] { breakpoints[i + 1] = cur.clone(); }
            }
        } else {
            breakpoints.push(Vec::new());
            for i in 0..n_breakpoints {
                let mut next = breakpoints[i].clone();
                for &nd in &bps.sets[i].1 { next.push(nd); }
                breakpoints.push(next);
            }
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
        let mut bp: Vec<i32> = Vec::new();
        let mut bcross = -1i32;
        if ext_pm > 0 {
            bp = vec![i32::MAX; inst.n_items];
            for (pos, nodes) in &bps.sets { for &nd in nodes { bp[nd] = *pos; } }
            let budget = inst.budgets[0] as f64;
            let mut cum = 0.0f64;
            for (pos, nodes) in &bps.sets {
                for &nd in nodes { cum += inst.weights[nd] as f64; }
                if cum >= budget { bcross = *pos; break; }
            }
        }
        QKPResult { results, bp, bcross }
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
        pub best_contrib: Vec<i32>,
        pub contrib:     Vec<i32>,
        pub value:       Ll,
        pub weight:      i32,
        pub count:       i32,
        pub best_value:  Ll,
        pub best_weight: i32,
        pub best_count:  i32,
        pub bu_cache:    Vec<(usize, Vec<ItemScore>)>,
        pub wu_cache:    Vec<(usize, Vec<ItemScore>)>,
        pub cache_on:    bool,
        pub vnd_resume:  bool,
        pub no_synergy:  bool,
        pub syn_bound:   bool,
        pub prof_dup:    usize,
        pub vnd_fast:    usize,
        pub wmax:        i32,
        pub sel_list:    Vec<u32>,
        pub unsel_list:  Vec<u32>,
        pub lists_ok:    bool,
    }

    impl P2State {
        pub fn new(ins: P2Instance) -> Self {
            let n = ins.n;
            P2State {
                sel: vec![0u8; n], best_sel: vec![0u8; n], contrib: vec![0i32; n],
                best_contrib: vec![0i32; n],
                value: 0, weight: 0, count: 0,
                best_value: NEG_INF, best_weight: 0, best_count: -1,
                bu_cache: Vec::new(), wu_cache: Vec::new(),
                cache_on: false, vnd_resume: false, no_synergy: false, syn_bound: false,
                prof_dup: 0, vnd_fast: 0,
                wmax: ins.w.iter().fold(0i32, |a, &b| if b > a { b } else { a }),
                sel_list: Vec::with_capacity(n), unsel_list: Vec::with_capacity(n),
                lists_ok: false,
                ins,
            }
        }

        #[inline]
        pub fn vnd_invalidate(&mut self) { self.bu_cache.clear(); self.wu_cache.clear(); }
        pub fn ensure_lists(&mut self) {
            if self.lists_ok { return; }
            self.sel_list.clear(); self.unsel_list.clear();
            for i in 0..self.ins.n {
                if self.sel[i] != 0 { self.sel_list.push(i as u32); }
                else { self.unsel_list.push(i as u32); }
            }
            self.lists_ok = true;
        }
        pub fn clear(&mut self) {
            self.lists_ok = false;
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
            self.lists_ok = false;
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
            self.lists_ok = false;
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
                self.best_contrib.copy_from_slice(&self.contrib);
            }
        }
        pub fn restore_best(&mut self) {
            if self.best_count < 0 {
                self.clear();
                return;
            }
            self.lists_ok = false;
            self.sel.copy_from_slice(&self.best_sel);
            self.contrib.copy_from_slice(&self.best_contrib);
            self.value  = self.best_value;
            self.weight = self.best_weight;
            self.count  = self.best_count;
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

    fn rush_build_dense(challenge: &Challenge, capacity: i32) -> P2Instance {
        let n = challenge.num_items;
        let mut q = vec![0i32; n * n];
        for i in 0..n {
            q[i * n + i] = (challenge.values[i] as f64) as i32;
            let row = &challenge.interaction_values[i];
            for (offset, &v) in row[(i + 1)..n].iter().enumerate() {
                if v != 0 {
                    let j  = i + 1 + offset;
                    let qv = (v as f64) as i32;
                    q[i * n + j] = qv;
                    q[j * n + i] = qv;
                }
            }
        }
        P2Instance {
            n, m: 0, capacity,
            w: challenge.weights.iter().map(|&w| w as i32).collect(),
            q,
        }
    }

    fn rush_build_pool(challenge: &Challenge, capacity: i32, pool: usize)
                       -> (P2Instance, Vec<usize>) {
        let n = challenge.num_items;
        let sum_w: i64 = challenge.weights.iter().map(|&w| w as i64).sum();
        let frac = if sum_w > 0 { capacity as f64 / sum_w as f64 } else { 1.0 };

        let mut rsum = vec![0f64; n];
        for i in 0..n {
            let row = &challenge.interaction_values[i];
            for (offset, &v) in row[(i + 1)..n].iter().enumerate() {
                if v != 0 {
                    let fv = v as f64;
                    rsum[i] += fv;
                    rsum[i + 1 + offset] += fv;
                }
            }
        }

        let mut scored: Vec<(f64, usize)> = (0..n)
            .map(|i| {
                let gain = challenge.values[i] as f64 + frac * rsum[i];
                let w    = challenge.weights[i] as f64;
                (if w > 0.0 { gain / w } else { gain }, i)
            })
            .collect();
        scored.sort_unstable_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });

        let k = pool.min(n);
        let mut ids: Vec<usize> = scored[..k].iter().map(|&(_, i)| i).collect();
        ids.sort_unstable();

        let mut q = vec![0i32; k * k];
        for a in 0..k {
            q[a * k + a] = (challenge.values[ids[a]] as f64) as i32;
            let row = &challenge.interaction_values[ids[a]];
            for b in (a + 1)..k {
                let v = row[ids[b]];
                if v != 0 {
                    let qv = (v as f64) as i32;
                    q[a * k + b] = qv;
                    q[b * k + a] = qv;
                }
            }
        }
        let ins = P2Instance {
            n: k, m: 0, capacity,
            w: ids.iter().map(|&i| challenge.weights[i] as i32).collect(),
            q,
        };
        (ins, ids)
    }

    pub fn bridge_load_solution(s: &mut P2State, br: &BudgetResult) {
        s.lists_ok = false;
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
                s.lists_ok = false;
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

    fn cached_best_unused(s: &mut P2State, max_k: usize) -> Vec<ItemScore> {
        if (s.vnd_fast & 268435456) != 0 { s.ensure_lists(); }
        if s.prof_dup & 1024 != 0 { let d = build_best_unused(s, max_k); if d.len() == usize::MAX { return d; } }
        if !s.cache_on { return build_best_unused(s, max_k); }
        for &(k, ref v) in s.bu_cache.iter() {
            if k == max_k { return v.clone(); }
        }
        let v = build_best_unused(s, max_k);
        s.bu_cache.push((max_k, v.clone()));
        v
    }

    fn cached_worst_used(s: &mut P2State, max_k: usize) -> Vec<ItemScore> {
        if (s.vnd_fast & 268435456) != 0 { s.ensure_lists(); }
        if s.prof_dup & 1024 != 0 { let d = build_worst_used(s, max_k); if d.len() == usize::MAX { return d; } }
        if !s.cache_on { return build_worst_used(s, max_k); }
        for &(k, ref v) in s.wu_cache.iter() {
            if k == max_k { return v.clone(); }
        }
        let v = build_worst_used(s, max_k);
        s.wu_cache.push((max_k, v.clone()));
        v
    }

    const LW2520: [Ll; 11] = [0, 2520, 1260, 840, 630, 504, 420, 360, 315, 280, 252];

    #[inline(always)]
    fn cbase(lim: i32, wc: usize, cu: usize) -> usize {
        let c = if lim < 1 { 0usize } else if (lim as usize) > wc { wc } else { lim as usize };
        c * (cu + 1)
    }

    #[inline(always)]
    fn dens_key(h258: bool, c: i32, w: i32) -> Ll {
        let w = if w > 1 { w } else { 1 };
        if h258 { c as Ll * LW2520[w as usize] } else { c as Ll * 1_000_000 / w as Ll }
    }

    fn build_best_unused(s: &P2State, max_k: usize) -> Vec<ItemScore> {
        if max_k == 0 { return Vec::new(); }
        let h258 = (s.vnd_fast & 134217728) != 0 && s.wmax <= 10;
        let h260 = (s.vnd_fast & 268435456) != 0 && s.lists_ok;
        let n = s.ins.n;
        let mut density: Vec<ItemScore> = if h260 {
            s.unsel_list.iter().map(|&u| { let i = u as usize;
                ItemScore { id: i, score: dens_key(h258, s.contrib[i], s.ins.w[i]) } }).collect()
        } else {
            (0..n).filter(|&i| s.sel[i] == 0).map(|i| {
                ItemScore { id: i, score: dens_key(h258, s.contrib[i], s.ins.w[i]) }
            }).collect()
        };
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

        let mut marginal: Vec<ItemScore> = if h260 {
            s.unsel_list.iter().map(|&u| { let i = u as usize;
                ItemScore { id: i, score: s.contrib[i] as Ll } }).collect()
        } else {
            (0..n).filter(|&i| s.sel[i] == 0).map(|i| {
                ItemScore { id: i, score: s.contrib[i] as Ll }
            }).collect()
        };
        marginal.select_nth_unstable_by(marginal_k, |a, b| b.score.cmp(&a.score));
        marginal.truncate(marginal_k);

        let mut present = vec![false; n];
        for is in &density { present[is.id] = true; }
        for is in marginal {
            if !present[is.id] {
                density.push(ItemScore {
                    id: is.id,
                    score: dens_key(h258, s.contrib[is.id], s.ins.w[is.id]),
                });
                present[is.id] = true;
            }
        }
        if density.len() < max_k {
            let mut fill: Vec<ItemScore> = if h260 {
                s.unsel_list.iter().map(|&u| u as usize).filter(|&i| !present[i])
                    .map(|i| ItemScore { id: i, score: dens_key(h258, s.contrib[i], s.ins.w[i]) })
                    .collect()
            } else {
                (0..n)
                    .filter(|&i| s.sel[i] == 0 && !present[i])
                    .map(|i| ItemScore { id: i, score: dens_key(h258, s.contrib[i], s.ins.w[i]) })
                    .collect()
            };
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
        let h258 = (s.vnd_fast & 134217728) != 0 && s.wmax <= 10;
        let h260 = (s.vnd_fast & 268435456) != 0 && s.lists_ok;
        let n = s.ins.n;
        let mut density: Vec<ItemScore> = if h260 {
            s.sel_list.iter().map(|&u| { let i = u as usize;
                ItemScore { id: i, score: dens_key(h258, s.contrib[i], s.ins.w[i]) } }).collect()
        } else {
            (0..n).filter(|&i| s.sel[i] != 0).map(|i| {
                ItemScore { id: i, score: dens_key(h258, s.contrib[i], s.ins.w[i]) }
            }).collect()
        };
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

        let mut marginal: Vec<ItemScore> = if h260 {
            s.sel_list.iter().map(|&u| { let i = u as usize;
                ItemScore { id: i, score: s.contrib[i] as Ll } }).collect()
        } else {
            (0..n).filter(|&i| s.sel[i] != 0).map(|i| {
                ItemScore { id: i, score: s.contrib[i] as Ll }
            }).collect()
        };
        marginal.select_nth_unstable_by(marginal_k, |a, b| a.score.cmp(&b.score));
        marginal.truncate(marginal_k);

        let mut present = vec![false; n];
        for is in &density { present[is.id] = true; }
        for is in marginal {
            if !present[is.id] {
                density.push(ItemScore {
                    id: is.id,
                    score: dens_key(h258, s.contrib[is.id], s.ins.w[is.id]),
                });
                present[is.id] = true;
            }
        }
        if density.len() < max_k {
            let mut fill: Vec<ItemScore> = if h260 {
                s.sel_list.iter().map(|&u| u as usize).filter(|&i| !present[i])
                    .map(|i| ItemScore { id: i, score: dens_key(h258, s.contrib[i], s.ins.w[i]) })
                    .collect()
            } else {
                (0..n)
                    .filter(|&i| s.sel[i] != 0 && !present[i])
                    .map(|i| ItemScore { id: i, score: dens_key(h258, s.contrib[i], s.ins.w[i]) })
                    .collect()
            };
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
        let unused = cached_best_unused(s, window_k);
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
        let h258 = (s.vnd_fast & 134217728) != 0 && s.wmax <= 10;
        let h260 = (s.vnd_fast & 268435456) != 0 && s.lists_ok;
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
                    pool.push(ItemScore {
                        id,
                        score: dens_key(h258, s.contrib[id], s.ins.w[id]),
                    });
                    present[id] = true;
                }
            }
        }

        if pool.len() < POOL_K {
            let mut extra: Vec<ItemScore> = Vec::new();
            {
                let mut emit = |i: usize, out: &mut Vec<ItemScore>| {
                    let mut best_pair = 0i64;
                    for seed in &pool {
                        let pair = s.ins.q_get(i, seed.id) as Ll;
                        if pair > best_pair { best_pair = pair; }
                    }
                    out.push(ItemScore { id: i, score: s.contrib[i] as Ll + best_pair });
                };
                if h260 {
                    for &u in s.unsel_list.iter() {
                        let i = u as usize;
                        if !present[i] { emit(i, &mut extra); }
                    }
                } else {
                    for i in 0..s.ins.n {
                        if s.sel[i] == 0 && !present[i] { emit(i, &mut extra); }
                    }
                }
            }
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
        if (s.vnd_fast & 268435456) != 0 { s.ensure_lists(); }
        if s.prof_dup & 512 != 0 { let d = build_synergy_unused(s); if d.len() == usize::MAX { return false; } }
        let unused = build_synergy_unused(s);

        let syn_bound = s.syn_bound;
        let ul = unused.len();
        let mut sc_suf: Vec<Ll> = Vec::new();
        let mut rowmax: Vec<Ll> = Vec::new();
        if syn_bound && ul > 0 {
            sc_suf = vec![Ll::MIN; ul + 1];
            for i in (0..ul).rev() {
                let v = s.contrib[unused[i].id] as Ll;
                sc_suf[i] = if v > sc_suf[i + 1] { v } else { sc_suf[i + 1] };
            }
            rowmax = vec![0; ul];
            for x in 0..ul {
                let base = unused[x].id * s.ins.n;
                let mut m: Ll = 0;
                for y in 0..ul {
                    let v = s.ins.q[base + unused[y].id] as Ll;
                    if v > m { m = v; }
                }
                rowmax[x] = m;
            }
        }

        let h245 = (s.vnd_fast & 65536) != 0;
        let h256 = (s.vnd_fast & 67108864) != 0 && s.wmax >= 1 && s.wmax <= 64;
        let wcu = if h256 { s.wmax as usize } else { 0 };
        let idrow = wcu + 1;
        let mut nxtu: Vec<u32> = vec![ul as u32; (wcu + 2) * (ul + 1)];
        {
            for c in 1..=wcu {
                let base = c * (ul + 1);
                for i in (0..ul).rev() {
                    nxtu[base + i] = if (s.ins.w[unused[i].id] as usize) <= c {
                        i as u32
                    } else {
                        nxtu[base + i + 1]
                    };
                }
            }
            let base = idrow * (ul + 1);
            for i in 0..=ul { nxtu[base + i] = i as u32; }
        }
        let idb = idrow * (ul + 1);
        let cbu = |lim: i32| -> usize {
            if !h256 { return idb; }
            let c = if lim < 1 { 0usize } else if (lim as usize) > wcu { wcu } else { lim as usize };
            c * (ul + 1)
        };
        let h251 = (s.vnd_fast & 8388608) != 0;
        let mut wmin1: Vec<i32> = Vec::new();
        let mut wmin2: Vec<i32> = Vec::new();
        let mut wmin3: Vec<i32> = Vec::new();
        if h245 && ul > 0 {
            const BIG: i32 = 1 << 29;
            wmin1 = vec![BIG; ul + 1];
            for i in (0..ul).rev() {
                let v = s.ins.w[unused[i].id];
                wmin1[i] = if v < wmin1[i + 1] { v } else { wmin1[i + 1] };
            }
            wmin2 = vec![BIG; ul + 1];
            for i in (0..ul).rev() {
                let v = if wmin1[i + 1] >= BIG { BIG } else { s.ins.w[unused[i].id] + wmin1[i + 1] };
                wmin2[i] = if v < wmin2[i + 1] { v } else { wmin2[i + 1] };
            }
            wmin3 = vec![BIG; ul + 1];
            for i in (0..ul).rev() {
                let v = if wmin2[i + 1] >= BIG { BIG } else { s.ins.w[unused[i].id] + wmin2[i + 1] };
                wmin3[i] = if v < wmin3[i + 1] { v } else { wmin3[i + 1] };
            }
        }

        {
            let used = cached_worst_used(s, 32);
            if unused.len() >= 2 && !used.is_empty() {
                let mut best_delta: Ll = 0;
                let mut best = None;
                for rm_score in &used {
                    let rm = rm_score.id;
                    let budget = s.slack() + s.ins.w[rm];
                    let lost = s.contrib[rm] as Ll;
                    let xb = cbu(budget - 1);
                    let mut x = nxtu[xb] as usize;
                    while x < ul {
                        if h245 && wmin2[x] > budget { break; }
                        let a = unused[x].id;
                        let wa = s.ins.w[a];
                        if !h256 && wa >= budget { x = nxtu[xb + x + 1] as usize; continue; }
                        let a_gain = s.contrib[a] as Ll - s.ins.q_get(a, rm) as Ll;
                        let yb = cbu(budget - wa);
                        let mut y = nxtu[yb + x + 1] as usize;
                        while y < ul {
                            if h245 && wa + wmin1[y] > budget { break; }
                            if syn_bound && a_gain
                                .saturating_add(sc_suf[y])
                                .saturating_add(rowmax[x])
                                .saturating_sub(lost) <= best_delta { break; }
                            let b = unused[y].id;
                            if !h256 && wa + s.ins.w[b] > budget { y = nxtu[yb + y + 1] as usize; continue; }
                            let delta = a_gain
                                + s.contrib[b] as Ll
                                - s.ins.q_get(b, rm) as Ll
                                + s.ins.q_get(a, b) as Ll
                                - lost;
                            if delta > best_delta {
                                best_delta = delta;
                                best = Some((rm, a, b));
                            }
                            y = nxtu[yb + y + 1] as usize;
                        }
                        x = nxtu[xb + x + 1] as usize;
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
            let used = cached_worst_used(s, 24);
            if limit >= 3 && !used.is_empty() {
                let mut best_delta: Ll = 0;
                let mut best = None;
                for rm_score in &used {
                    let rm = rm_score.id;
                    let budget = s.slack() + s.ins.w[rm];
                    let lost = s.contrib[rm] as Ll;
                    let xb = cbu(budget);
                    let mut x = nxtu[xb] as usize;
                    while x < limit {
                        if h245 && wmin3[x] > budget { break; }
                        let a = unused[x].id;
                        let wa = s.ins.w[a];
                        if !h256 && wa > budget { x = nxtu[xb + x + 1] as usize; continue; }
                        let a_gain = s.contrib[a] as Ll - s.ins.q_get(a, rm) as Ll;
                        let yb = cbu(budget - wa);
                        let mut y = nxtu[yb + x + 1] as usize;
                        while y < limit {
                            if h245 && wa + wmin2[y] > budget { break; }
                            let b = unused[y].id;
                            let wab = wa + s.ins.w[b];
                            if !h256 && wab > budget { y = nxtu[yb + y + 1] as usize; continue; }
                            let b_gain = s.contrib[b] as Ll - s.ins.q_get(b, rm) as Ll;
                            let pre2 = if h251 {
                                a_gain.saturating_add(b_gain)
                                      .saturating_add(s.ins.q_get(a, b) as Ll)
                            } else { 0 };
                            let zb = cbu(budget - wab);
                            let mut z = nxtu[zb + y + 1] as usize;
                            while z < limit {
                                if h245 && wab + wmin1[z] > budget { break; }
                                                                if h251 {
                                    if syn_bound && pre2
                                        .saturating_add(sc_suf[z])
                                        .saturating_add(rowmax[x])
                                        .saturating_add(rowmax[y])
                                        .saturating_sub(lost) <= best_delta { break; }
                                } else
                                if syn_bound && a_gain
                                    .saturating_add(b_gain)
                                    .saturating_add(s.ins.q_get(a, b) as Ll)
                                    .saturating_add(sc_suf[z])
                                    .saturating_add(rowmax[x])
                                    .saturating_add(rowmax[y])
                                    .saturating_sub(lost) <= best_delta { break; }
                                let c = unused[z].id;
                                if !h256 && wab + s.ins.w[c] > budget { z = nxtu[zb + z + 1] as usize; continue; }
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
                                z = nxtu[zb + z + 1] as usize;
                            }
                            y = nxtu[yb + y + 1] as usize;
                        }
                        x = nxtu[xb + x + 1] as usize;
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
            let used = cached_worst_used(s, 20);
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
                    let xb = cbu(budget);
                    let mut x = nxtu[xb] as usize;
                    while x < limit {
                        if h245 && wmin3[x] > budget { break; }
                        let a = unused[x].id;
                        let wa = w[a];
                        if !h256 && wa > budget { x = nxtu[xb + x + 1] as usize; continue; }
                        let a_base = a * n;
                        let a_gain = contrib[a] as Ll - q[a_base + r1] as Ll - q[a_base + r2] as Ll;
                        let yb = cbu(budget - wa);
                        let mut y = nxtu[yb + x + 1] as usize;
                        while y < limit {
                            if h245 && wa + wmin2[y] > budget { break; }
                            let b = unused[y].id;
                            let wab = wa + w[b];
                            if !h256 && wab > budget { y = nxtu[yb + y + 1] as usize; continue; }
                            let b_base = b * n;
                            let b_gain = contrib[b] as Ll - q[b_base + r1] as Ll - q[b_base + r2] as Ll;
                            let pre3 = if h251 {
                                a_gain.saturating_add(b_gain)
                                      .saturating_add(q[a_base + b] as Ll)
                            } else { 0 };
                            let zb = cbu(budget - wab);
                            let mut z = nxtu[zb + y + 1] as usize;
                            while z < limit {
                                if h245 && wab + wmin1[z] > budget { break; }
                                                                if h251 {
                                    if syn_bound && pre3
                                        .saturating_add(sc_suf[z])
                                        .saturating_add(rowmax[x])
                                        .saturating_add(rowmax[y])
                                        .saturating_sub(lost) <= best_delta { break; }
                                } else
                                if syn_bound && a_gain
                                    .saturating_add(b_gain)
                                    .saturating_add(q[a_base + b] as Ll)
                                    .saturating_add(sc_suf[z])
                                    .saturating_add(rowmax[x])
                                    .saturating_add(rowmax[y])
                                    .saturating_sub(lost) <= best_delta { break; }
                                let c = unused[z].id;
                                if !h256 && wab + w[c] > budget { z = nxtu[zb + z + 1] as usize; continue; }
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
                                z = nxtu[zb + z + 1] as usize;
                            }
                            y = nxtu[yb + y + 1] as usize;
                        }
                        x = nxtu[xb + x + 1] as usize;
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
            if h245 && wmin3[x] > rem { break; }
            let a = unused[x].id;
            let wa = s.ins.w[a];
            if wa >= rem { continue; }
            for y in (x + 1)..unused.len() {
                if h245 && wa + wmin2[y] > rem { break; }
                let b = unused[y].id;
                let wab = wa + s.ins.w[b];
                if wab >= rem { continue; }
                let pre4 = if h251 {
                    (s.contrib[a] as Ll).saturating_add(s.contrib[b] as Ll)
                                        .saturating_add(s.ins.q_get(a, b) as Ll)
                } else { 0 };
                for z in (y + 1)..unused.len() {
                    if h245 && wab + wmin1[z] > rem { break; }
                    if h251 {
                        if syn_bound && pre4
                            .saturating_add(sc_suf[z])
                            .saturating_add(rowmax[x])
                            .saturating_add(rowmax[y]) <= best_delta { break; }
                    } else
                    if syn_bound && (s.contrib[a] as Ll)
                        .saturating_add(s.contrib[b] as Ll)
                        .saturating_add(s.ins.q_get(a, b) as Ll)
                        .saturating_add(sc_suf[z])
                        .saturating_add(rowmax[x])
                        .saturating_add(rowmax[y]) <= best_delta { break; }
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
        let unused = cached_best_unused(s, window_k);
        let used   = cached_worst_used(s, window_k);
        let cu = unused.len(); let cs = used.len();
        let n = s.ins.n;
        let slack = s.slack();
        let q = &s.ins.q;
        let w = &s.ins.w;
        let contrib = &s.contrib;
        let mut best_delta: Ll = 0;
        let mut best_ordinal = usize::MAX;
        let mut br = None; let mut ba = None;
        let mut s11_mincontrib: Ll = Ll::MAX;
        for x in 0..cs {
            let v = contrib[used[x].id] as Ll;
            if v < s11_mincontrib { s11_mincontrib = v; }
        }
        for y in 0..cu {
            let add = unused[y].id;
            let add_weight = w[add];
            let add_contrib = contrib[add] as Ll;
            let row_base = add * n;
            if add_contrib.saturating_sub(s11_mincontrib) < best_delta { continue; }
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
        let vf12 = s.vnd_fast & 2;
        let h255 = (s.vnd_fast & 33554432) != 0 && s.wmax >= 1 && s.wmax <= 64;
        let unused = cached_best_unused(s, window_k);
        let used   = cached_worst_used(s, window_k);
        let cu = unused.len(); let cs = used.len();
        let mut best_delta: Ll = 0;
        let mut br = None; let mut ba = None; let mut bb = None;
        let n_items = s.ins.n;
        let mut effective: Vec<Ll> = vec![0; cu];
        let mut s12_rowsuf: Vec<Ll> = vec![Ll::MIN; cu];
        let h249 = (s.vnd_fast & 1048576) != 0;
        let h250 = (s.vnd_fast & 2097152) != 0;
        let mut sufbuf: Vec<Ll> = vec![Ll::MIN; cu + 1];
        let wc = if h255 { s.wmax as usize } else { 0 };
        let mut nxt: Vec<u32> = Vec::new();
        if h255 && cu > 0 {
            nxt = vec![cu as u32; (wc + 1) * (cu + 1)];
            for c in 1..=wc {
                let base = c * (cu + 1);
                nxt[base + cu] = cu as u32;
                for y in (0..cu).rev() {
                    nxt[base + y] = if (s.ins.w[unused[y].id] as usize) <= c {
                        y as u32
                    } else {
                        nxt[base + y + 1]
                    };
                }
            }
        }
        let use255 = h255 && !nxt.is_empty();
        for x in 0..cu {
            let ax = unused[x].id;
            let row_ax = &s.ins.q[ax * s.ins.n..(ax + 1) * s.ins.n];
            let mut m = Ll::MIN;
            for y in (x + 1)..cu {
                let v = row_ax[unused[y].id] as Ll;
                if v > m { m = v; }
            }
            s12_rowsuf[x] = m;
        }
        for si in 0..cs {
            let rm = used[si].id; let budget = s.slack() + s.ins.w[rm];
            let lost = s.contrib[rm] as Ll;
            if h250 {
                let row_rm = &s.ins.q[rm * n_items..(rm + 1) * n_items];
                for x in 0..cu {
                    let id = unused[x].id;
                    effective[x] = s.contrib[id] as Ll - row_rm[id] as Ll;
                }
            } else {
                for x in 0..cu {
                    let id = unused[x].id;
                    effective[x] = s.contrib[id] as Ll - s.ins.q[id * n_items + rm] as Ll;
                }
            }
            if !h249 { sufbuf = vec![Ll::MIN; cu + 1]; }
            let s12_sufmax = &mut sufbuf;
            for x in (0..cu).rev() {
                s12_sufmax[x] = if effective[x] > s12_sufmax[x + 1] { effective[x] } else { s12_sufmax[x + 1] };
            }
            let mut x = 0usize;
            if use255 { x = if budget >= 1 { nxt[cbase(budget - 1, wc, cu) + 0] as usize } else { cu }; }
            while x < cu {
                let a = unused[x].id; let wa = s.ins.w[a];
                if !use255 && wa >= budget { x += 1; continue; }
                let ca_eff = effective[x];
                if ca_eff
                    .saturating_add(s12_sufmax[x + 1])
                    .saturating_add(s12_rowsuf[x])
                    .saturating_sub(lost) <= best_delta {
                    x = if use255 { nxt[cbase(budget - 1, wc, cu) + x + 1] as usize } else { x + 1 };
                    continue;
                }
                let row_a = &s.ins.q[a * n_items..(a + 1) * n_items];
                if use255 {
                    let yb = cbase(budget - wa, wc, cu);
                    let mut y = nxt[yb + x + 1] as usize;
                    while y < cu {
                        if vf12 != 0 && ca_eff
                            .saturating_add(s12_sufmax[y])
                            .saturating_add(s12_rowsuf[x])
                            .saturating_sub(lost) <= best_delta { break; }
                        let b = unused[y].id;
                        let d = ca_eff + effective[y] + row_a[b] as Ll - lost;
                        if d > best_delta {
                            best_delta = d; br = Some(rm); ba = Some(a); bb = Some(b);
                        }
                        y = nxt[yb + y + 1] as usize;
                    }
                } else {
                for y in (x+1)..cu {
                    if vf12 != 0 && ca_eff
                        .saturating_add(s12_sufmax[y])
                        .saturating_add(s12_rowsuf[x])
                        .saturating_sub(lost) <= best_delta { break; }
                    let b = unused[y].id;
                    if wa + s.ins.w[b] > budget { continue; }
                    let d = ca_eff + effective[y] + row_a[b] as Ll - lost;
                    if d > best_delta {
                        best_delta = d; br = Some(rm); ba = Some(a); bb = Some(b);
                    }
                }
                }
                x = if use255 { nxt[cbase(budget - 1, wc, cu) + x + 1] as usize } else { x + 1 };
            }
        }
        if let (Some(r), Some(a), Some(b)) = (br, ba, bb) {
            s.remove_item(r); s.add_item(a); s.add_item(b);
            return true;
        }
        false
    }

    fn apply_best_swap21(s: &mut P2State, window_k: usize) -> bool {
        let s_vnd_fast = s.vnd_fast;
        let unused = cached_best_unused(s, window_k);
        let used   = cached_worst_used(s, window_k);
        let cu = unused.len(); let cs = used.len();

        let n = s.ins.n;
        let q = &s.ins.q;
        let w = &s.ins.w;
        let contrib = &s.contrib;
        let budget_base = s.ins.capacity - s.weight;

        let mut s21_min_lost: Ll = Ll::MAX;
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
                if lost < s21_min_lost { s21_min_lost = lost; }
            }
        }

        let mut best_delta: Ll = 0;
        let mut br1 = None; let mut br2 = None; let mut ba = None;
        let vf21 = s_vnd_fast & 4;
        let npu = used_pairs.len();
        let mut lsuf: Vec<Ll> = Vec::new();
        if vf21 != 0 {
            lsuf = vec![Ll::MAX; npu + 1];
            for i in (0..npu).rev() {
                let l = used_pairs[i].3;
                lsuf[i] = if l < lsuf[i + 1] { l } else { lsuf[i + 1] };
            }
        }
        let vf21r = (s_vnd_fast & 128) != 0;
        let npu = used_pairs.len();
        let mut lranked: Vec<(Ll, u32)> = Vec::new();
        if vf21r {
            lranked.reserve(npu);
            for i in 0..npu { lranked.push((used_pairs[i].3, i as u32)); }
            lranked.sort_unstable();
        }
        let mut best_pos21 = usize::MAX;
        let mut best_out21 = usize::MAX;
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
            if cadd.saturating_sub(min_cross).saturating_sub(s21_min_lost) <= best_delta { continue; }
            if vf21r {
                for t in 0..npu {
                    let (lost, pidx) = lranked[t];
                    let upper_bound = cadd - min_cross - lost;
                    if upper_bound <= best_delta { break; }
                    let pi = pidx as usize;
                    let (r1, r2, budget, _) = used_pairs[pi];
                    if add_w > budget { continue; }
                    let gained = cadd - q[add_base + r1] as Ll - q[add_base + r2] as Ll;
                    let d = gained - lost;
                    if d > best_delta
                        || (d == best_delta && best_out21 == ai && pi < best_pos21) {
                        best_delta = d; best_pos21 = pi; best_out21 = ai;
                        br1 = Some(r1); br2 = Some(r2); ba = Some(add);
                    }
                }
                continue;
            }
            for (ui, &(r1, r2, budget, lost)) in used_pairs.iter().enumerate() {
                if vf21 != 0
                    && cadd.saturating_sub(min_cross).saturating_sub(lsuf[ui]) <= best_delta {
                    break;
                }
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
        let s_vnd_fast = s.vnd_fast;
        let s_prof_dup = s.prof_dup;
        let k      = window_k.min(40);
        let unused = cached_best_unused(s, k);
        let used   = cached_worst_used(s, k);
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

        let mut cross_cache: Vec<i64> = Vec::new();
        if s_vnd_fast & 32 == 0 {
            cross_cache = vec![0i64; unused_pairs.len() * cs];
            for (pair_idx, &(_, _, p_base, q2_base, _, _, _)) in unused_pairs.iter().enumerate() {
                let cache_row = &mut cross_cache[pair_idx * cs..(pair_idx + 1) * cs];
                for (used_idx, rm_score) in used.iter().enumerate() {
                    let rm = rm_score.id;
                    cache_row[used_idx] = q[p_base + rm] as Ll + q[q2_base + rm] as Ll;
                }
            }
        }

        if s_prof_dup & 2048 != 0 {
            let mut d1: Vec<(usize, usize, usize, usize, i32, Ll)> = Vec::with_capacity(cs * (cs - 1) / 2);
            for x in 0..cs {
                let r1 = used[x].id; let r1_base = r1 * n;
                for y in (x+1)..cs {
                    let r2 = used[y].id;
                    d1.push((x, y, r1, r2, slack + w[r1] + w[r2],
                             contrib[r1] as Ll + contrib[r2] as Ll - q[r1_base + r2] as Ll));
                }
            }
            let mut d2: Vec<(usize, usize, usize, usize, i32, i32, Ll)> = Vec::with_capacity(cu * (cu - 1) / 2);
            for a in 0..cu {
                let p = unused[a].id; let wp = w[p]; let p_base = p * n;
                for b in (a+1)..cu {
                    let q2 = unused[b].id; let q2_base = q2 * n;
                    d2.push((p, q2, p_base, q2_base, wp, wp + w[q2],
                             contrib[p] as Ll + contrib[q2] as Ll + q[p_base + q2] as Ll));
                }
            }
            if d1.len() == usize::MAX || d2.len() == usize::MAX { return false; }
        }
        let mut best_delta: Ll = 0;
        let mut br1 = None; let mut br2 = None;
        let mut ba1 = None; let mut ba2 = None;
        let vf = s_vnd_fast;
        let np = unused_pairs.len();
        let mut gsuf0: Ll = Ll::MIN;
        if vf & 16 != 0 {
            for i in 0..np { let g = unused_pairs[i].6; if g > gsuf0 { gsuf0 = g; } }
        }
        let mut tsuf: Vec<Ll> = Vec::new();
        let mut asuf: Vec<Ll> = Vec::new();
        if vf & 1 != 0 {
            tsuf = vec![Ll::MIN; cu * (cu + 1)];
            for a in 0..cu {
                let abase = unused[a].id * n;
                let row = &mut tsuf[a * (cu + 1)..(a + 1) * (cu + 1)];
                for b in (0..cu).rev() {
                    let v = contrib[unused[b].id] as Ll + q[abase + unused[b].id] as Ll;
                    row[b] = if v > row[b + 1] { v } else { row[b + 1] };
                }
            }
            asuf = vec![Ll::MIN; cu + 1];
            for a in (0..cu).rev() {
                let ca = contrib[unused[a].id] as Ll;
                let v = if a + 1 < cu {
                    ca.saturating_add(tsuf[a * (cu + 1) + a + 1])
                } else { Ll::MIN };
                asuf[a] = if v > asuf[a + 1] { v } else { asuf[a + 1] };
            }
        }
        let drop_wp = (vf & 8) != 0;
        let gmax: Ll = if np > 0 { gsuf0 } else { Ll::MIN };
        if vf & 32 != 0 {
            let h236 = (s_vnd_fast & 4096) != 0;
            let mut ord: Vec<u32> = Vec::new();
            if h236 {
                let mut key: Vec<(Ll, u32)> = Vec::with_capacity(np);
                for i in 0..np { key.push((0 - unused_pairs[i].6, i as u32)); }
                key.sort_unstable();
                ord.reserve(np);
                for &(_, i) in key.iter() { ord.push(i); }
            } else {
                ord = (0..np as u32).collect();
                ord.sort_unstable_by(|&i, &j| {
                    let a = unused_pairs[i as usize].6;
                    let b = unused_pairs[j as usize].6;
                    b.cmp(&a).then(i.cmp(&j))
                });
            }
            let h235 = (vf & 2048) != 0;
            let mut rk: Vec<(Ll, u32, i32, u32, u32, u32, u32)> = Vec::new();
            let mut gsort: Vec<Ll> = Vec::new();
            if h235 {
                rk.reserve(np);
                for &i in ord.iter() {
                    let (p, q2, p_base, q2_base, wp, pair_w, gain) = unused_pairs[i as usize];
                    rk.push((gain, i, pair_w, p as u32, q2 as u32, p_base as u32, q2_base as u32));
                    let _ = wp;
                }
            } else {
                gsort = ord.iter().map(|&i| unused_pairs[i as usize].6).collect();
            }
            if s_prof_dup & 4096 != 0 {
                let mut key: Vec<(Ll, u32)> = Vec::with_capacity(np);
                for i in 0..np { key.push((0 - unused_pairs[i].6, i as u32)); }
                key.sort_unstable();
                if key.len() == usize::MAX { return false; }
            }
            let nt = ord.len();
            let mut best_pos  = usize::MAX;
            let mut best_out  = usize::MAX;
            let h242 = (vf & 32768) != 0;
            let h254 = (vf & 16777216) != 0 && h235;
            let mut bmax: i32 = 0;
            let mut cstart: Vec<u32> = Vec::new();
            let mut clen:   Vec<u32> = Vec::new();
            let mut cidx:   Vec<u32> = Vec::new();
            if h254 && nt > 0 {
                for t in 0..nt { let pw = rk[t].2; if pw > bmax { bmax = pw; } }
                let nb = bmax as usize + 1;
                let mut hist = vec![0u32; nb];
                for t in 0..nt { hist[rk[t].2 as usize] += 1; }
                clen = vec![0u32; nb];
                let mut acc = 0u32;
                for b in 0..nb { acc += hist[b]; clen[b] = acc; }
                cstart = vec![0u32; nb];
                let mut off = 0u32;
                for b in 0..nb { cstart[b] = off; off += clen[b]; }
                cidx = vec![0u32; off as usize];
                let mut fill: Vec<u32> = cstart.clone();
                for t in 0..nt {
                    let pw = rk[t].2 as usize;
                    for b in pw..nb {
                        let f = fill[b] as usize;
                        cidx[f] = t as u32;
                        fill[b] = f as u32 + 1;
                    }
                }
            }
            let nop = used_pairs.len();
            let mut oord: Vec<u32> = Vec::new();
            if h242 {
                let mut okey: Vec<(Ll, u32)> = Vec::with_capacity(nop);
                for i in 0..nop { okey.push((used_pairs[i].5, i as u32)); }
                okey.sort_unstable();
                oord.reserve(nop);
                for &(_, i) in okey.iter() { oord.push(i); }
            }
            for oiz in 0..nop {
                let oi = if h242 { oord[oiz] as usize } else { oiz };
                let (r1_idx, r2_idx, r1, r2, budget, lost) = used_pairs[oi];
                if h242 {
                    for t in 0..nt {
                        let (gain, pin, pair_w, p, q2, p_base, q2_base) = rk[t];
                        let upper_bound = gain - lost;
                        if upper_bound <= best_delta { break; }
                        if pair_w > budget { continue; }
                        let pb = p_base as usize;
                        let qb = q2_base as usize;
                        let cross = q[pb + r1] as Ll + q[qb + r1] as Ll
                                  + q[pb + r2] as Ll + q[qb + r2] as Ll;
                        let d = upper_bound - cross;
                        let pi = pin as usize;
                        if d > best_delta
                            || (d == best_delta && best_out != usize::MAX
                                && (oi < best_out || (oi == best_out && pi < best_pos)))
                        {
                            best_delta = d;
                            best_pos = pi; best_out = oi;
                            br1 = Some(r1); br2 = Some(r2);
                            ba1 = Some(p as usize);  ba2 = Some(q2 as usize);
                        }
                    }
                    continue;
                }
                if h254 {
                    if nt == 0 { continue; }
                    let bi = if budget >= bmax { bmax as usize }
                             else if budget < 0 { 0 } else { budget as usize };
                    let s0 = cstart[bi] as usize;
                    let s1 = s0 + clen[bi] as usize;
                    for ci in s0..s1 {
                        let (gain, pin, _pair_w, p, q2, p_base, q2_base) = rk[cidx[ci] as usize];
                        let upper_bound = gain - lost;
                        if upper_bound <= best_delta { break; }
                        let pb = p_base as usize;
                        let qb = q2_base as usize;
                        let cross = q[pb + r1] as Ll + q[qb + r1] as Ll
                                  + q[pb + r2] as Ll + q[qb + r2] as Ll;
                        let d = upper_bound - cross;
                        let pi = pin as usize;
                        if d > best_delta || (d == best_delta && best_out == oi && pi < best_pos) {
                            best_delta = d;
                            best_pos = pi; best_out = oi;
                            br1 = Some(r1); br2 = Some(r2);
                            ba1 = Some(p as usize);  ba2 = Some(q2 as usize);
                        }
                    }
                    continue;
                }
                if h235 {
                    for t in 0..nt {
                        let (gain, pin, pair_w, p, q2, p_base, q2_base) = rk[t];
                        let upper_bound = gain - lost;
                        if upper_bound <= best_delta { break; }
                        if pair_w > budget { continue; }
                        let pb = p_base as usize;
                        let qb = q2_base as usize;
                        let cross = q[pb + r1] as Ll + q[qb + r1] as Ll
                                  + q[pb + r2] as Ll + q[qb + r2] as Ll;
                        let d = upper_bound - cross;
                        let pi = pin as usize;
                        if d > best_delta || (d == best_delta && best_out == oi && pi < best_pos) {
                            best_delta = d;
                            best_pos = pi; best_out = oi;
                            br1 = Some(r1); br2 = Some(r2);
                            ba1 = Some(p as usize);  ba2 = Some(q2 as usize);
                        }
                    }
                    continue;
                }
                for t in 0..nt {
                    let gain = gsort[t];
                    let upper_bound = gain - lost;
                    if upper_bound <= best_delta { break; }
                    let pi = ord[t] as usize;
                    let (p, q2, p_base, q2_base, _wp, pair_w, _) = unused_pairs[pi];
                    if pair_w > budget { continue; }
                    let cross = q[p_base + r1] as Ll + q[q2_base + r1] as Ll
                              + q[p_base + r2] as Ll + q[q2_base + r2] as Ll;
                    let d = upper_bound - cross;
                    if d > best_delta || (d == best_delta && best_out == oi && pi < best_pos) {
                        best_delta = d;
                        best_pos = pi; best_out = oi;
                        br1 = Some(r1); br2 = Some(r2);
                        ba1 = Some(p);  ba2 = Some(q2);
                    }
                }
            }
            if s_prof_dup & 8192 != 0 {
                let mut bd: Ll = 0; let mut bp2 = usize::MAX; let mut bo = usize::MAX;
                let mut acc: Ll = 0;
                for (oi, &(_, _, r1, r2, budget, lost)) in used_pairs.iter().enumerate() {
                    for t in 0..nt {
                        let (gain, pin, pair_w, _p, _q2, p_base, q2_base) = rk[t];
                        let upper_bound = gain - lost;
                        if upper_bound <= bd { break; }
                        if pair_w > budget { continue; }
                        let pb = p_base as usize; let qb = q2_base as usize;
                        let cross = q[pb + r1] as Ll + q[qb + r1] as Ll
                                  + q[pb + r2] as Ll + q[qb + r2] as Ll;
                        let d = upper_bound - cross;
                        let pi = pin as usize;
                        if d > bd || (d == bd && bo == oi && pi < bp2) { bd = d; bp2 = pi; bo = oi; }
                    }
                    acc = acc.wrapping_add(bd);
                }
                if acc == Ll::MIN { return false; }
            }
            if let (Some(r1), Some(r2), Some(a1), Some(a2)) = (br1, br2, ba1, ba2) {
                s.remove_item(r1); s.remove_item(r2);
                s.add_item(a1);    s.add_item(a2);
                return true;
            }
            return false;
        }
        for &(r1_idx, r2_idx, r1, r2, budget, lost) in &used_pairs {
            if vf & 16 != 0 && gmax.saturating_sub(lost) <= best_delta { continue; }
            if vf & 1 != 0 {
                let mut pair_idx = 0usize;
                for a in 0..cu {
                    if asuf[a].saturating_sub(lost) <= best_delta { break; }
                    let ca  = contrib[unused[a].id] as Ll;
                    let row = &tsuf[a * (cu + 1)..(a + 1) * (cu + 1)];
                    if a + 1 >= cu { continue; }
                    if ca.saturating_add(row[a + 1]).saturating_sub(lost) <= best_delta {
                        pair_idx += cu - a - 1;
                        continue;
                    }
                    for b in (a + 1)..cu {
                        if ca.saturating_add(row[b]).saturating_sub(lost) <= best_delta {
                            pair_idx += cu - b;
                            break;
                        }
                        let (p, q2, _, _, wp, pair_w, gain) = unused_pairs[pair_idx];
                        pair_idx += 1;
                        if drop_wp {
                            if pair_w > budget { continue; }
                        } else {
                            if wp >= budget || pair_w > budget { continue; }
                        }
                        let upper_bound = gain - lost;
                        if upper_bound <= best_delta { continue; }
                        let cache_row = (pair_idx - 1) * cs;
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
            } else {
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
        }
        if let (Some(r1), Some(r2), Some(a1), Some(a2)) = (br1, br2, ba1, ba2) {
            s.remove_item(r1); s.remove_item(r2);
            s.add_item(a1);    s.add_item(a2);
            return true;
        }
        false
    }

    fn local_search_vnd(s: &mut P2State, window_k: usize, heavy: bool) {
        s.vnd_invalidate();
        if !s.vnd_resume {
            loop {
                if apply_best_swap11(s, window_k)     { s.vnd_invalidate(); continue; }
                if apply_best_swap12(s, window_k / 2) { s.vnd_invalidate(); continue; }
                if apply_best_swap21(s, window_k / 2) { s.vnd_invalidate(); continue; }
                if heavy {
                    if apply_best_swap22(s, window_k) { s.vnd_invalidate(); continue; }
                    if !s.no_synergy && apply_synergy_tail(s) { s.vnd_invalidate(); continue; }
                }
                break;
            }
        } else {
            let mut lvl = 0usize;
            loop {
                let fired = match lvl {
                    0 => apply_best_swap11(s, window_k),
                    1 => apply_best_swap12(s, window_k / 2),
                    2 => apply_best_swap21(s, window_k / 2),
                    3 => heavy && apply_best_swap22(s, window_k),
                    _ => heavy && !s.no_synergy && apply_synergy_tail(s),
                };
                if fired { s.vnd_invalidate(); continue; }
                if s.prof_dup != 0 {
                    let d = s.prof_dup;
                    match lvl {
                        0 => { if d & 1 != 0 { let _ = apply_best_swap11(s, window_k); } }
                        1 => { if d & 2 != 0 { let _ = apply_best_swap12(s, window_k / 2); } }
                        2 => { if d & 4 != 0 { let _ = apply_best_swap21(s, window_k / 2); } }
                        3 => { if d & 8 != 0 && heavy { let _ = apply_best_swap22(s, window_k); } }
                        _ => { if d & 16 != 0 && heavy && !s.no_synergy { let _ = apply_synergy_tail(s); } }
                    }
                }
                lvl += 1;
                if lvl > 4 { break; }
            }
        }
        s.save_best();
    }

    fn dp_refinement(s: &mut P2State, core_half: usize) {
        let h258 = (s.vnd_fast & 134217728) != 0 && s.wmax <= 10;
        let n = s.ins.n; let cap = s.ins.capacity;
        let mut ord: Vec<ItemScore> = (0..n).map(|i| {
            ItemScore { id: i, score: dens_key(h258, s.contrib[i], s.ins.w[i]) }
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
                    core_half:  usize,
                    stall_stop: usize,
                    pert_ramp:  usize) {
        let total_interactions = compute_total_interactions(&s.ins);
        let mut current_certified = false;
        let mut best_certified = false;
        let mut before_dp = s.sel.clone();
        let prof = s.prof_dup;
        let mut certified_refinement =
            |s: &mut P2State, half: usize, certified: &mut bool| {
                if prof & 32 != 0 {
                    let sv_sel = s.sel.clone(); let sv_c = s.contrib.clone();
                    let (v, w, c) = (s.value, s.weight, s.count);
                    dp_refinement(s, half);
                    s.sel.copy_from_slice(&sv_sel); s.contrib.copy_from_slice(&sv_c);
                    s.value = v; s.weight = w; s.count = c;
                    s.vnd_invalidate();
                }
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
            if stall_stop > 0 && stall >= stall_stop { break; }

            let strength = {
                let base = 1 + stall / 3 + round / 25
                           + if pert_ramp > 0 { round / pert_ramp } else { 0 };
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

    fn rush_search(s: &mut P2State, hp: &Hparams) {
        let total_interactions = compute_total_interactions(&s.ins);
        let (core_cap, win_cap) = if hp.rush_polish > 0 {
            ((24 * hp.rush_polish / 100).max(1), (120 * hp.rush_polish / 100).max(1))
        } else {
            (24, 120)
        };
        let core   = hp.core_half_dp.min(core_cap);
        let win    = hp.window_k.min(win_cap);
        let passes = if hp.rush_mode >= 3 { hp.rush_passes.max(1) } else { 1 };
        const STRAT_ORDER: [usize; 5] = [1, 0, 2, 3, 4];
        let starts = if hp.rush_mode >= 4 { hp.rush_starts.clamp(1, 5) } else { 1 };
        for k in 0..starts {
            if k > 0 { s.clear(); }
            greedy_reconstruct(s, STRAT_ORDER[k], &total_interactions);
            s.save_best();
            if hp.rush_mode >= 2 {
                for _ in 0..passes {
                    let before = s.value;
                    dp_refinement(s, core);
                    local_search_vnd(s, win, true);
                    s.save_best();
                    if s.value <= before { break; }
                }
            }
        }
        s.restore_best();
    }

    const CB_SC: i64 = 64;
    const CB_INF: i64 = i64::MAX / 8;

    struct CbFlow {
        to: Vec<u32>, cap: Vec<i64>, head: Vec<i32>, nxt: Vec<i32>,
        level: Vec<i32>, it: Vec<i32>, q: Vec<u32>, stk: Vec<u32>, pe: Vec<i32>,
        tarc: Vec<i32>, pre: bool,
    }

    impl CbFlow {
        fn new() -> Self {
            CbFlow { to: Vec::new(), cap: Vec::new(), head: Vec::new(), nxt: Vec::new(),
                     level: Vec::new(), it: Vec::new(), q: Vec::new(), stk: Vec::new(), pe: Vec::new(),
                     tarc: Vec::new(), pre: false }
        }
        fn reset(&mut self, n: usize) {
            self.to.clear(); self.cap.clear(); self.nxt.clear();
            self.head.clear(); self.head.resize(n, -1);
            self.level.clear(); self.level.resize(n, -1);
            self.it.clear(); self.it.resize(n, -1);
            self.pe.clear(); self.pe.resize(n, -1);
            self.tarc.clear(); self.tarc.resize(n, -1);
        }
        #[inline]
        fn add(&mut self, u: usize, v: usize, c: i64) {
            let e = self.to.len();
            if v + 1 == self.tarc.len() { self.tarc[u] = e as i32; }
            self.to.push(v as u32); self.cap.push(c); self.nxt.push(self.head[u]); self.head[u] = e as i32;
            self.to.push(u as u32); self.cap.push(0); self.nxt.push(self.head[v]); self.head[v] = (e + 1) as i32;
        }
        fn bfs(&mut self, s: usize, t: usize) -> bool {
            for x in self.level.iter_mut() { *x = -1; }
            self.level[s] = 0; self.q.clear(); self.q.push(s as u32);
            let mut qi = 0usize;
            while qi < self.q.len() {
                let u = self.q[qi] as usize; qi += 1;
                if self.level[t] >= 0 && self.level[u] >= self.level[t] { break; }
                let mut e = self.head[u];
                while e != -1 {
                    let ei = e as usize; let v = self.to[ei] as usize;
                    if self.cap[ei] > 0 && self.level[v] < 0 { self.level[v] = self.level[u] + 1; self.q.push(v as u32); }
                    e = self.nxt[ei];
                }
            }
            self.level[t] >= 0
        }
        fn augment(&mut self, s: usize, t: usize) -> i64 {
            let mut total = 0i64;
            self.stk.clear(); self.stk.push(s as u32);
            loop {
                let u = match self.stk.last() { Some(&u) => u as usize, None => return total };
                if u == t {
                    let mut f = CB_INF; let mut cut = 1usize;
                    for k in 1..self.stk.len() {
                        let v = self.stk[k] as usize; let c = self.cap[self.pe[v] as usize];
                        if c < f { f = c; cut = k; }
                    }
                    for k in 1..self.stk.len() { let v = self.stk[k] as usize; let ei = self.pe[v] as usize; self.cap[ei] -= f; self.cap[ei ^ 1] += f; }
                    total += f;
                    self.stk.truncate(cut);
                    continue;
                }
                let mut adv = false;
                while self.it[u] != -1 {
                    let ei = self.it[u] as usize; let v = self.to[ei] as usize;
                    if self.cap[ei] > 0 && self.level[v] == self.level[u] + 1 {
                        self.pe[v] = ei as i32; self.stk.push(v as u32); adv = true; break;
                    }
                    self.it[u] = self.nxt[ei];
                }
                if !adv {
                    self.level[u] = -1;
                    self.stk.pop();
                    if let Some(&p) = self.stk.last() { let p = p as usize; let ei = self.it[p] as usize; self.it[p] = self.nxt[ei]; }
                }
            }
        }
        fn maxflow(&mut self, s: usize, t: usize) -> i64 {
            let mut flow = 0i64;
            if self.pre {
                let mut e0 = self.head[s];
                while e0 != -1 {
                    let ei = e0 as usize;
                    if self.cap[ei] > 0 {
                        let a = self.to[ei] as usize;
                        let mut ea = self.head[a];
                        while ea != -1 {
                            let eai = ea as usize;
                            if self.cap[eai] > 0 {
                                let b = self.to[eai] as usize;
                                let tb = self.tarc[b];
                                if tb >= 0 {
                                    let tbi = tb as usize;
                                    let mut f = self.cap[ei];
                                    if self.cap[eai] < f { f = self.cap[eai]; }
                                    if self.cap[tbi] < f { f = self.cap[tbi]; }
                                    if f > 0 {
                                        self.cap[ei] -= f; self.cap[ei ^ 1] += f;
                                        self.cap[eai] -= f; self.cap[eai ^ 1] += f;
                                        self.cap[tbi] -= f; self.cap[tbi ^ 1] += f;
                                        flow += f;
                                        if self.cap[ei] == 0 { break; }
                                    }
                                }
                            }
                            ea = self.nxt[eai];
                        }
                    }
                    e0 = self.nxt[ei];
                }
            }
            while self.bfs(s, t) {
                for i in 0..self.head.len() { self.it[i] = self.head[i]; }
                flow += self.augment(s, t);
            }
            flow
        }
        fn source_side(&mut self, s: usize, out: &mut Vec<bool>) {
            out.clear(); out.resize(self.head.len(), false);
            self.stk.clear(); self.stk.push(s as u32); out[s] = true;
            while let Some(u) = self.stk.pop() {
                let mut e = self.head[u as usize];
                while e != -1 {
                    let ei = e as usize; let v = self.to[ei] as usize;
                    if self.cap[ei] > 0 && !out[v] { out[v] = true; self.stk.push(v as u32); }
                    e = self.nxt[ei];
                }
            }
        }
    }

    struct CutEng {
        nb_s: Vec<u32>, nb_j: Vec<u32>, nb_q: Vec<i64>, kk: Vec<i64>, degsum: u64, nint: u64,
        st: Vec<i8>, pin: Vec<i64>, pmax: Vec<i64>, qu: Vec<u32>, rid: Vec<i32>, rl: Vec<u32>,
        off: Vec<u32>, to: Vec<u32>, rev: Vec<u32>, cap: Vec<i64>, fill: Vec<u32>,
        level: Vec<i32>, it: Vec<u32>, q: Vec<u32>, stk: Vec<u32>, pe: Vec<u32>, reach: Vec<bool>,
        tarc: Vec<u32>, pre: bool,
    }

    impl CutEng {
        fn new() -> Self {
            CutEng { nb_s: Vec::new(), nb_j: Vec::new(), nb_q: Vec::new(), kk: Vec::new(), degsum: 0, nint: 0,
                     st: Vec::new(), pin: Vec::new(), pmax: Vec::new(), qu: Vec::new(), rid: Vec::new(), rl: Vec::new(),
                     off: Vec::new(), to: Vec::new(), rev: Vec::new(), cap: Vec::new(), fill: Vec::new(),
                     level: Vec::new(), it: Vec::new(), q: Vec::new(), stk: Vec::new(), pe: Vec::new(), reach: Vec::new(),
                     tarc: Vec::new(), pre: true }
        }

        fn flow(&mut self, nn: usize, s: usize, t: usize) {
            if self.pre {
                for e0 in self.off[s] as usize..self.off[s + 1] as usize {
                    let a = self.to[e0] as usize;
                    for ea in self.off[a] as usize..self.off[a + 1] as usize {
                        if self.cap[e0] == 0 { break; }
                        let b = self.to[ea] as usize;
                        if b >= s || self.cap[ea] == 0 { continue; }
                        let tb = self.tarc[b];
                        if tb == u32::MAX { continue; }
                        let tb = tb as usize;
                        let mut f = self.cap[e0];
                        if self.cap[ea] < f { f = self.cap[ea]; }
                        if self.cap[tb] < f { f = self.cap[tb]; }
                        if f > 0 {
                            self.cap[e0] -= f; let r0 = self.rev[e0] as usize; self.cap[r0] += f;
                            self.cap[ea] -= f; let ra = self.rev[ea] as usize; self.cap[ra] += f;
                            self.cap[tb] -= f; let rb = self.rev[tb] as usize; self.cap[rb] += f;
                        }
                    }
                }
            }
            loop {
                for x in self.level[..nn].iter_mut() { *x = -1; }
                self.level[s] = 0; self.q.clear(); self.q.push(s as u32);
                let mut qi = 0usize;
                while qi < self.q.len() {
                    let u = self.q[qi] as usize; qi += 1;
                    if self.level[t] >= 0 && self.level[u] >= self.level[t] { break; }
                    let lu = self.level[u] + 1;
                    for e in self.off[u] as usize..self.off[u + 1] as usize {
                        let v = self.to[e] as usize;
                        if self.cap[e] > 0 && self.level[v] < 0 { self.level[v] = lu; self.q.push(v as u32); }
                    }
                }
                if self.level[t] < 0 { break; }
                for i in 0..nn { self.it[i] = self.off[i]; }
                self.stk.clear(); self.stk.push(s as u32);
                loop {
                    let u = match self.stk.last() { Some(&u) => u as usize, None => break };
                    if u == t {
                        let mut f = i64::MAX; let mut cut = 1usize;
                        for k in 1..self.stk.len() {
                            let c = self.cap[self.pe[self.stk[k] as usize] as usize];
                            if c < f { f = c; cut = k; }
                        }
                        for k in 1..self.stk.len() {
                            let ei = self.pe[self.stk[k] as usize] as usize;
                            self.cap[ei] -= f; let r = self.rev[ei] as usize; self.cap[r] += f;
                        }
                        self.stk.truncate(cut);
                        continue;
                    }
                    let end = self.off[u + 1];
                    let lu = self.level[u] + 1;
                    let mut adv = false;
                    while self.it[u] < end {
                        let ei = self.it[u] as usize; let v = self.to[ei] as usize;
                        if self.cap[ei] > 0 && self.level[v] == lu {
                            self.pe[v] = ei as u32; self.stk.push(v as u32); adv = true; break;
                        }
                        self.it[u] += 1;
                    }
                    if !adv {
                        self.level[u] = -1;
                        self.stk.pop();
                        if let Some(&p) = self.stk.last() { self.it[p as usize] += 1; }
                    }
                }
            }
            for x in self.reach[..nn].iter_mut() { *x = false; }
            self.stk.clear(); self.stk.push(s as u32); self.reach[s] = true;
            while let Some(u) = self.stk.pop() {
                let u = u as usize;
                for e in self.off[u] as usize..self.off[u + 1] as usize {
                    let v = self.to[e] as usize;
                    if self.cap[e] > 0 && !self.reach[v] { self.reach[v] = true; self.stk.push(v as u32); }
                }
            }
        }
    }

    impl CoreBnb {
        fn prep(&mut self) {
            let m = self.idx.len();
            let ce = &mut self.ce;
            ce.nb_s.clear(); ce.nb_j.clear(); ce.nb_q.clear(); ce.kk.clear();
            let mut degsum = 0u64; let mut nint = 0u64;
            for a in 0..m {
                ce.nb_s.push(ce.nb_j.len() as u32);
                let ia = self.idx[a] as usize;
                let (b0, b1) = (self.adj_s[ia] as usize, self.adj_s[ia + 1] as usize);
                degsum += (b1 - b0) as u64;
                let mut up = 0i64;
                for x in b0..b1 {
                    let pb = self.pos[self.adj_j[x] as usize];
                    if pb >= 0 {
                        ce.nb_j.push(pb as u32); ce.nb_q.push(CB_SC * self.adj_q[x]);
                        if pb > a as i32 { up += self.adj_q[x]; nint += 1; }
                    }
                }
                ce.kk.push(CB_SC * (self.lin_cur[ia] + up));
            }
            ce.nb_s.push(ce.nb_j.len() as u32);
            ce.degsum = degsum; ce.nint = nint;
            self.prep_ok = true;
        }

        fn eval_side(&mut self, lam: i64) -> (i64, i64) {
            if !self.prep_ok { self.prep(); }
            let m = self.idx.len();
            let mut nterm = 0u64;
            for a in 0..m { if lam * self.w[self.idx[a] as usize] != self.ce.kk[a] { nterm += 1; } }
            self.work += self.ce.degsum + 2 * (self.ce.nint + nterm);
            let mut v = 0i64; let mut wt = 0i64;
            for a in 0..m {
                if !self.side[a] { continue; }
                let ia = self.idx[a] as usize;
                v += CB_SC * self.lin_cur[ia] - lam * self.w[ia]; wt += self.w[ia];
                for x in self.ce.nb_s[a] as usize..self.ce.nb_s[a + 1] as usize {
                    let b = self.ce.nb_j[x] as usize;
                    if b > a && self.side[b] { v += self.ce.nb_q[x]; }
                }
            }
            (v, wt)
        }

        fn closure_n(&mut self, lam: i64, lo: Option<&[bool]>, hi: Option<&[bool]>) -> (i64, i64) {
            if !self.prep_ok { self.prep(); }
            let m = self.idx.len();
            let ce = &mut self.ce;
            ce.st.clear(); ce.st.resize(m, -1);
            if let Some(l) = lo { for a in 0..m { if l[a] { ce.st[a] = 1; } } }
            if let Some(h) = hi { for a in 0..m { if !h[a] { ce.st[a] = 0; } } }
            ce.pin.clear(); ce.pin.resize(m, 0); ce.pmax.clear(); ce.pmax.resize(m, 0);
            ce.qu.clear();
            for a in 0..m {
                if ce.st[a] != -1 { continue; }
                let ia = self.idx[a] as usize;
                let mut pi = CB_SC * self.lin_cur[ia] - lam * self.w[ia];
                let mut pm = pi;
                for x in ce.nb_s[a] as usize..ce.nb_s[a + 1] as usize {
                    let b = ce.nb_j[x] as usize; let sb = ce.st[b];
                    if sb == 1 { pi += ce.nb_q[x]; pm += ce.nb_q[x]; } else if sb == -1 { pm += ce.nb_q[x]; }
                }
                ce.pin[a] = pi; ce.pmax[a] = pm;
                if pi > 0 || pm <= 0 { ce.qu.push(a as u32); }
            }
            let mut qh = 0usize;
            while qh < ce.qu.len() {
                let a = ce.qu[qh] as usize; qh += 1;
                if ce.st[a] != -1 { continue; }
                if ce.pin[a] > 0 {
                    ce.st[a] = 1;
                    for x in ce.nb_s[a] as usize..ce.nb_s[a + 1] as usize {
                        let b = ce.nb_j[x] as usize;
                        if ce.st[b] == -1 { ce.pin[b] += ce.nb_q[x]; if ce.pin[b] > 0 { ce.qu.push(b as u32); } }
                    }
                } else if ce.pmax[a] <= 0 {
                    ce.st[a] = 0;
                    for x in ce.nb_s[a] as usize..ce.nb_s[a + 1] as usize {
                        let b = ce.nb_j[x] as usize;
                        if ce.st[b] == -1 { ce.pmax[b] -= ce.nb_q[x]; if ce.pmax[b] <= 0 { ce.qu.push(b as u32); } }
                    }
                }
            }
            ce.rid.clear(); ce.rid.resize(m, -1); ce.rl.clear();
            for a in 0..m { if ce.st[a] == -1 { ce.rid[a] = ce.rl.len() as i32; ce.rl.push(a as u32); } }
            let r = ce.rl.len();
            if r > 0 {
                let s = r; let t = r + 1; let nn = r + 2;
                ce.fill.clear(); ce.fill.resize(nn + 1, 0);
                let mut nterm_s = 0u32; let mut nterm_t = 0u32;
                for x in 0..r {
                    let a = ce.rl[x] as usize;
                    let mut e = ce.pin[a];
                    for y in ce.nb_s[a] as usize..ce.nb_s[a + 1] as usize {
                        let b = ce.nb_j[y] as usize;
                        if b > a && ce.st[b] == -1 { e += ce.nb_q[y]; ce.fill[x] += 1; ce.fill[ce.rid[b] as usize] += 1; }
                    }
                    ce.pmax[a] = e;
                    if e > 0 { ce.fill[x] += 1; nterm_s += 1; } else if e < 0 { ce.fill[x] += 1; nterm_t += 1; }
                }
                ce.fill[s] = nterm_s; ce.fill[t] = nterm_t;
                ce.off.clear(); ce.off.resize(nn + 1, 0);
                for u in 0..nn { ce.off[u + 1] = ce.off[u] + ce.fill[u]; }
                let na = ce.off[nn] as usize;
                ce.tarc.clear(); ce.tarc.resize(r, u32::MAX);
                ce.to.clear(); ce.to.resize(na, 0); ce.rev.clear(); ce.rev.resize(na, 0); ce.cap.clear(); ce.cap.resize(na, 0);
                for u in 0..nn { ce.fill[u] = ce.off[u]; }
                for x in 0..r {
                    let a = ce.rl[x] as usize;
                    let e = ce.pmax[a];
                    if e > 0 {
                        let p = ce.fill[s] as usize; ce.fill[s] += 1; let q = ce.fill[x] as usize; ce.fill[x] += 1;
                        ce.to[p] = x as u32; ce.cap[p] = e; ce.rev[p] = q as u32;
                        ce.to[q] = s as u32; ce.cap[q] = 0; ce.rev[q] = p as u32;
                    } else if e < 0 {
                        let p = ce.fill[x] as usize; ce.fill[x] += 1; let q = ce.fill[t] as usize; ce.fill[t] += 1;
                        ce.to[p] = t as u32; ce.cap[p] = -e; ce.rev[p] = q as u32; ce.tarc[x] = p as u32;
                        ce.to[q] = x as u32; ce.cap[q] = 0; ce.rev[q] = p as u32;
                    }
                    for y in ce.nb_s[a] as usize..ce.nb_s[a + 1] as usize {
                        let b = ce.nb_j[y] as usize;
                        if b > a && ce.st[b] == -1 {
                            let xb = ce.rid[b] as usize;
                            let p = ce.fill[x] as usize; ce.fill[x] += 1; let q = ce.fill[xb] as usize; ce.fill[xb] += 1;
                            ce.to[p] = xb as u32; ce.cap[p] = ce.nb_q[y]; ce.rev[p] = q as u32;
                            ce.to[q] = x as u32; ce.cap[q] = 0; ce.rev[q] = p as u32;
                        }
                    }
                }
                if ce.level.len() < nn { ce.level.resize(nn, -1); ce.it.resize(nn, 0); ce.pe.resize(nn, 0); ce.reach.resize(nn, false); }
                ce.flow(nn, s, t);
                for x in 0..r { if ce.reach[x] { ce.st[ce.rl[x] as usize] = 1; } }
            }
            self.side.clear(); self.side.resize(m, false);
            for a in 0..m { self.side[a] = self.ce.st[a] == 1; }
            self.eval_side(lam)
        }
    }

    struct CoreBnb {
        k: usize,
        w: Vec<i64>,
        lin_cur: Vec<i64>,
        cap: i64,
        best_val: i64,
        best_sel: Vec<u8>,
        improved: bool,
        work: u64,
        work_cap: u64,
        lam_iters: usize,
        adj_s: Vec<u32>, adj_j: Vec<u32>, adj_q: Vec<i64>,
        pos: Vec<i32>,
        state: Vec<i8>,
        fl: CbFlow,
        idx: Vec<u32>,
        side: Vec<bool>,
        ce: CutEng, prep_ok: bool, eng: usize,
    }

    impl CoreBnb {
        fn closure(&mut self, lam: i64) -> (i64, i64) {
            if self.eng == 2 { return self.closure_n(lam, None, None); }
            let m = self.idx.len();
            let s = m; let t = m + 1;
            self.fl.reset(m + 2);
            let mut cst = 0i64;
            for a in 0..m {
                let ia = self.idx[a] as usize;
                let mut e = lam * self.w[ia] - CB_SC * self.lin_cur[ia];
                let (b0, b1) = (self.adj_s[ia] as usize, self.adj_s[ia + 1] as usize);
                for x in b0..b1 {
                    let pb = self.pos[self.adj_j[x] as usize];
                    if pb > a as i32 { e -= CB_SC * self.adj_q[x]; }
                }
                if e > 0 { self.fl.add(a, t, e); } else if e < 0 { cst += e; self.fl.add(s, a, -e); }
            }
            for a in 0..m {
                let ia = self.idx[a] as usize;
                let (b0, b1) = (self.adj_s[ia] as usize, self.adj_s[ia + 1] as usize);
                self.work += (b1 - b0) as u64;
                for x in b0..b1 {
                    let pb = self.pos[self.adj_j[x] as usize];
                    if pb > a as i32 { self.fl.add(a, pb as usize, CB_SC * self.adj_q[x]); }
                }
            }
            let cut = self.fl.maxflow(s, t);
            self.work += self.fl.to.len() as u64;
            let mut side = Vec::new();
            self.fl.source_side(s, &mut side);
            side.truncate(m);
            let mut wt = 0i64;
            for a in 0..m { if side[a] { wt += self.w[self.idx[a] as usize]; } }
            self.side = side;
            (-(cst + cut), wt)
        }

        fn try_incumbent(&mut self, base: i64) {
            let mut v = base;
            let m = self.idx.len();
            for a in 0..m {
                if !self.side[a] { continue; }
                let ia = self.idx[a] as usize;
                v += self.lin_cur[ia];
                let (b0, b1) = (self.adj_s[ia] as usize, self.adj_s[ia + 1] as usize);
                for x in b0..b1 {
                    let pb = self.pos[self.adj_j[x] as usize];
                    if pb > a as i32 && self.side[pb as usize] { v += self.adj_q[x]; }
                }
            }
            if v > self.best_val {
                self.best_val = v;
                self.improved = true;
                for i in 0..self.k { self.best_sel[i] = (self.state[i] == 1) as u8; }
                for a in 0..m { if self.side[a] { self.best_sel[self.idx[a] as usize] = 1; } }
            }
        }

        fn rebuild_idx(&mut self) {
            self.prep_ok = false;
            self.idx.clear();
            for i in 0..self.k {
                if self.state[i] == -1 { self.pos[i] = self.idx.len() as i32; self.idx.push(i as u32); }
                else { self.pos[i] = -1; }
            }
        }

        fn bound_at(&mut self, base: i64, c: i64, lam: i64) -> i64 {
            self.rebuild_idx();
            if self.idx.is_empty() {
                if base > self.best_val { self.side = Vec::new(); self.try_incumbent(base); }
                return base * CB_SC;
            }
            let (v, wt) = self.closure(lam);
            if wt <= c { self.try_incumbent(base); }
            base * CB_SC + lam * c + v
        }

        fn root_lambda(&mut self, base: i64, c: i64) -> (i64, i64) {
            let mut best = i64::MAX; let mut bl = 0i64;
            let mut lo = 0i64; let mut hi = 1i64;
            loop {
                self.rebuild_idx();
                let (v, wt) = self.closure(hi);
                if wt <= c { self.try_incumbent(base); }
                let b = base * CB_SC + hi * c + v;
                if b < best { best = b; bl = hi; }
                if wt <= c || hi > (1i64 << 40) || self.work >= self.work_cap { break; }
                lo = hi; hi *= 2;
            }
            while hi - lo > 1 {
                let mid = lo + (hi - lo) / 2;
                self.rebuild_idx();
                let (v, wt) = self.closure(mid);
                if wt <= c { self.try_incumbent(base); }
                let b = base * CB_SC + mid * c + v;
                if b < best { best = b; bl = mid; }
                if wt > c { lo = mid; } else { hi = mid; }
                if self.work >= self.work_cap { break; }
            }
            (best, bl)
        }

        fn set_in(&mut self, i: usize) {
            self.state[i] = 1;
            let (b0, b1) = (self.adj_s[i] as usize, self.adj_s[i + 1] as usize);
            for x in b0..b1 { let j = self.adj_j[x] as usize; self.lin_cur[j] += self.adj_q[x]; }
        }
        fn unset_in(&mut self, i: usize) {
            let (b0, b1) = (self.adj_s[i] as usize, self.adj_s[i + 1] as usize);
            for x in b0..b1 { let j = self.adj_j[x] as usize; self.lin_cur[j] -= self.adj_q[x]; }
            self.state[i] = -1;
        }

        fn root_fix(&mut self, base: &mut i64, c: &mut i64, lam: i64, passes: usize) {
            for _ in 0..passes {
                let mut changed = false;
                for i in 0..self.k {
                    if self.state[i] != -1 { continue; }
                    if self.work >= self.work_cap { return; }
                    let thr = (self.best_val + 1) * CB_SC;
                    let out_ok = if self.w[i] > *c { true } else {
                        let add = self.lin_cur[i];
                        self.set_in(i);
                        let b = self.bound_at(*base + add, *c - self.w[i], lam);
                        self.unset_in(i);
                        b < (self.best_val + 1) * CB_SC
                    };
                    let _ = thr;
                    if out_ok { self.state[i] = 0; changed = true; continue; }
                    self.state[i] = 0;
                    let b = self.bound_at(*base, *c, lam);
                    self.state[i] = -1;
                    if b < (self.best_val + 1) * CB_SC {
                        let add = self.lin_cur[i];
                        self.set_in(i);
                        *base += add; *c -= self.w[i];
                        changed = true;
                    }
                }
                if !changed { break; }
            }
        }

        fn closure_k(&mut self, lam: i64, known: &Vec<(i64, Vec<bool>, u8)>) -> (i64, i64) {
            if self.eng != 2 { return self.closure(lam); }
            let m = self.idx.len();
            let mut lo: Vec<bool> = Vec::new(); let mut hi: Vec<bool> = Vec::new();
            for kv in known.iter() {
                if kv.2 == 1 && kv.0 == lam {
                    self.side.clear(); self.side.extend_from_slice(&kv.1);
                    return self.eval_side(lam);
                }
            }
            for kv in known.iter() {
                if (kv.2 == 1 && kv.0 > lam) || (kv.2 == 3 && kv.0 >= lam) {
                    if lo.is_empty() { lo.extend_from_slice(&kv.1); } else { for a in 0..m { lo[a] |= kv.1[a]; } }
                }
                if (kv.2 == 1 && kv.0 < lam) || (kv.2 == 2 && kv.0 <= lam) {
                    if hi.is_empty() { hi.extend_from_slice(&kv.1); } else { for a in 0..m { hi[a] &= kv.1[a]; } }
                }
            }
            let lo_o = if lo.is_empty() { None } else { Some(&lo[..]) };
            let hi_o = if hi.is_empty() { None } else { Some(&hi[..]) };
            self.closure_n(lam, lo_o, hi_o)
        }

        fn dfs(&mut self, base: i64, c: i64, lam_in: i64, li: usize, mut known: Vec<(i64, Vec<bool>, u8)>) {
            if self.work >= self.work_cap { return; }
            self.prep_ok = false;
            self.idx.clear();
            for i in 0..self.k {
                if self.state[i] == -1 { self.pos[i] = self.idx.len() as i32; self.idx.push(i as u32); }
                else { self.pos[i] = -1; }
            }
            self.work += self.k as u64;
            if self.idx.is_empty() { return; }
            let thr = (self.best_val + 1) * CB_SC;
            let bsc = base * CB_SC;
            let mut n_cuts = 1usize;
            let (v, wt) = self.closure_k(lam_in, &known);
            if self.eng == 2 { known.push((lam_in, self.side.clone(), 1)); }
            let mut bound = bsc + lam_in * c + v;
            if wt <= c { self.try_incumbent(base); }
            if bound < thr { return; }
            let mut best_lam = lam_in;
            let mut best_side = self.side.clone();
            let mut lo; let mut hi;
            if wt > c {
                lo = lam_in;
                let mut step = 1i64.max(lam_in / 16);
                loop {
                    hi = lam_in + step;
                    let (v2, w2) = self.closure_k(hi, &known); n_cuts += 1;
                    if self.eng == 2 { known.push((hi, self.side.clone(), 1)); }
                    if self.work >= self.work_cap { return; }
                    let b2 = bsc + hi * c + v2;
                    if w2 <= c { self.try_incumbent(base); }
                    if b2 < bound { bound = b2; best_lam = hi; best_side = self.side.clone(); }
                    if bound < (self.best_val + 1) * CB_SC { return; }
                    if w2 <= c || n_cuts > li || step > (1i64 << 40) { break; }
                    lo = hi; step *= 2;
                }
            } else {
                hi = lam_in; lo = lam_in;
                if lam_in > 0 {
                    let mut step = 1i64.max(lam_in / 16);
                    loop {
                        lo = (lam_in - step).max(0);
                        let (v2, w2) = self.closure_k(lo, &known); n_cuts += 1;
                        if self.eng == 2 { known.push((lo, self.side.clone(), 1)); }
                        if self.work >= self.work_cap { return; }
                        let b2 = bsc + lo * c + v2;
                        if w2 <= c { self.try_incumbent(base); }
                        if b2 < bound { bound = b2; best_lam = lo; best_side = self.side.clone(); }
                        if bound < (self.best_val + 1) * CB_SC { return; }
                        if w2 > c || lo == 0 || n_cuts > li { break; }
                        hi = lo; step *= 2;
                    }
                }
            }
            while hi - lo > 1 && n_cuts <= li {
                let mid = lo + (hi - lo) / 2;
                let (v2, w2) = self.closure_k(mid, &known); n_cuts += 1;
                if self.eng == 2 { known.push((mid, self.side.clone(), 1)); }
                if self.work >= self.work_cap { return; }
                let b2 = bsc + mid * c + v2;
                if w2 <= c { self.try_incumbent(base); }
                if b2 < bound { bound = b2; best_lam = mid; best_side = self.side.clone(); }
                if bound < (self.best_val + 1) * CB_SC { return; }
                if w2 > c { lo = mid; } else { hi = mid; }
            }
            let _ = thr;
            let mut bi = usize::MAX; let mut bw = -1i64;
            for a in 0..self.idx.len() {
                if best_side[a] { let i = self.idx[a] as usize; if self.w[i] > bw { bw = self.w[i]; bi = i; } }
            }
            if bi == usize::MAX {
                for a in 0..self.idx.len() { let i = self.idx[a] as usize; if self.w[i] > bw { bw = self.w[i]; bi = i; } }
            }
            let mut kin: Vec<(i64, Vec<bool>, u8)> = Vec::new();
            let mut kout: Vec<(i64, Vec<bool>, u8)> = Vec::new();
            if self.eng == 2 {
                let mut pb = 0usize;
                for a in 0..self.idx.len() { if self.idx[a] as usize == bi { pb = a; } }
                let st = if known.len() > 12 { known.len() - 12 } else { 0 };
                for kv in known[st..].iter() {
                    let mut v = kv.1.clone(); let had = v.remove(pb);
                    if kv.2 == 1 {
                        kin.push((kv.0, v.clone(), if had { 1 } else { 3 }));
                        kout.push((kv.0, v, if had { 2 } else { 1 }));
                    } else if kv.2 == 3 { kin.push((kv.0, v, 3)); }
                    else { kout.push((kv.0, v, 2)); }
                }
            }
            if self.w[bi] <= c {
                let add = self.lin_cur[bi];
                self.state[bi] = 1;
                let (b0, b1) = (self.adj_s[bi] as usize, self.adj_s[bi + 1] as usize);
                for x in b0..b1 { let j = self.adj_j[x] as usize; self.lin_cur[j] += self.adj_q[x]; }
                self.dfs(base + add, c - self.w[bi], best_lam, self.lam_iters, kin);
                for x in b0..b1 { let j = self.adj_j[x] as usize; self.lin_cur[j] -= self.adj_q[x]; }
            }
            self.state[bi] = 0;
            self.dfs(base, c, best_lam, self.lam_iters, kout);
            self.state[bi] = -1;
        }
    }

    fn core_bnb(s: &mut P2State, bp: &[i32], bcross: i32, num_params: usize,
                hi_pm: usize, work_cap: usize, lam_iters: usize, fix_passes: usize, kmax: usize, pre_mode: usize) -> bool {
        let n = s.ins.n;
        if bp.len() != n || bcross < 0 || work_cap == 0 || s.best_count < 0 { return false; }
        let rem = (num_params as i64 + 1 - bcross as i64).max(0);
        let hi_steps = ((rem * hi_pm as i64 + 999) / 1000) as i32;
        let start = bcross - hi_steps;
        let mut fixed: Vec<usize> = Vec::new();
        let mut core: Vec<usize> = Vec::new();
        for i in 0..n {
            let b = bp[i];
            if b < start { fixed.push(i); }
            else if b <= num_params as i32 || s.best_sel[i] != 0 { core.push(i); }
        }
        let cap = s.ins.capacity as i64;
        let wf: i64 = fixed.iter().map(|&i| s.ins.w[i] as i64).sum();
        if wf > cap || core.is_empty() { return false; }
        let k = core.len();
        if k > kmax { return false; }
        let mut vf: i64 = 0;
        for (x, &i) in fixed.iter().enumerate() {
            vf += s.ins.q_get(i, i) as i64;
            for &j in &fixed[x + 1..] { vf += s.ins.q_get(i, j) as i64; }
        }
        let mut lin: Vec<i64> = Vec::with_capacity(k);
        let mut w: Vec<i64> = Vec::with_capacity(k);
        for &i in &core {
            let mut l = s.ins.q_get(i, i) as i64;
            for &j in &fixed { l += s.ins.q_get(i, j) as i64; }
            lin.push(l); w.push(s.ins.w[i] as i64);
        }
        let mut adj_s: Vec<u32> = Vec::with_capacity(k + 1);
        let mut adj_j: Vec<u32> = Vec::new();
        let mut adj_q: Vec<i64> = Vec::new();
        for a in 0..k {
            adj_s.push(adj_j.len() as u32);
            let ia = core[a];
            for b in 0..k {
                if b == a { continue; }
                let qq = s.ins.q_get(ia, core[b]);
                if qq > 0 { adj_j.push(b as u32); adj_q.push(qq as i64); }
            }
        }
        adj_s.push(adj_j.len() as u32);
        let mut bb = CoreBnb {
            k, w, lin_cur: lin, cap: cap - wf, best_val: s.best_value - vf, best_sel: vec![0u8; k],
            improved: false, work: (k * k) as u64, work_cap: work_cap as u64, lam_iters,
            adj_s, adj_j, adj_q, pos: vec![-1; k], state: vec![-1; k], fl: CbFlow::new(),
            idx: Vec::new(), side: Vec::new(),
            ce: CutEng::new(), prep_ok: false, eng: pre_mode,
        };
        bb.fl.pre = pre_mode > 0;
        let c0 = bb.cap;
        if fix_passes > 0 {
            let (_rb, rl) = bb.root_lambda(0, c0);
            let mut base = 0i64; let mut c = c0;
            bb.root_fix(&mut base, &mut c, rl.max(1), fix_passes);
            if c >= 0 && bb.work < bb.work_cap { bb.dfs(base, c, rl.max(1), 64, Vec::new()); }
        } else {
            bb.dfs(0, c0, 1, 64, Vec::new());
        }
        if !bb.improved { return false; }
        s.clear();
        for &i in &fixed { s.add_item(i); }
        for a in 0..k { if bb.best_sel[a] != 0 { s.add_item(core[a]); } }
        if s.weight > s.ins.capacity { s.restore_best(); return false; }
        s.save_best();
        true
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
            let seed = raw_seed.wrapping_add(0x9E3779B97F4A7C15)
                .wrapping_add((hp.rng_stream as u64).wrapping_mul(0xD1B54A32D192ED03));
            let mut rng = Rng::new(seed);

            if hp.rush_mode > 0 && (hp.rush_fast > 0 || hp.rush_pool > 0) {
                let cap = challenge.max_weight as i32;
                let (p2, ids) = if hp.rush_pool > 0 && hp.rush_pool < n {
                    let (p2, ids) = rush_build_pool(challenge, cap, hp.rush_pool);
                    (p2, Some(ids))
                } else {
                    (rush_build_dense(challenge, cap), None)
                };
                let mut s = P2State::new(p2);
                rush_search(&mut s, &hp);
                if P2State::weight_selected(&s.ins, &s.best_sel) > cap {
                    return Ok(None);
                }
                let items: Vec<usize> = (0..s.ins.n)
                    .filter(|&i| s.best_sel[i] != 0)
                    .map(|i| ids.as_ref().map_or(i, |v| v[i]))
                    .collect();
                return Ok(Some(Solution { items }));
            }

            let inst   = challenge_to_p1(challenge);
            let budget = inst.budgets[0];

            if hp.stage_stop == 1 {
                let tag = inst.n_edges as u64
                    ^ inst.weights.iter().map(|&w| w as u64).sum::<u64>();
                return Ok(stage_stop_pick(&inst.weights, budget, tag));
            }

            if hp.rush_mode > 0 {
                let p2 = bridge_p1_to_p2(&inst, budget);
                let mut s = P2State::new(p2);
                rush_search(&mut s, &hp);
                let chk_wt = P2State::weight_selected(&s.ins, &s.best_sel);
                if chk_wt > budget {
                    return Ok(None);
                }
                let items: Vec<usize> = (0..s.ins.n)
                    .filter(|&i| s.best_sel[i] != 0)
                    .collect();
                return Ok(Some(Solution { items }));
            }

            let mut bp_tag = 0u64;
            let ext_pm = if hp.core_work > 0 && n <= 1200 { hp.core_lo as i32 } else { 0 };
            let qr = run_bp_algorithm(&inst, hp.n_lambda_values, hp.stage_stop, &mut bp_tag, hp.bp_fast, hp.prof_dup, ext_pm);
            if hp.stage_stop == 2 {
                return Ok(stage_stop_pick(&inst.weights, budget, bp_tag));
            }
            if hp.stage_stop == 3 {
                let items = qr.results[0].selected_items.clone();
                let wt: i64 = items.iter().map(|&i| inst.weights[i] as i64).sum();
                if wt > budget as i64 {
                    let tag = qr.results[0].ofv as u64;
                    return Ok(stage_stop_pick(&inst.weights, budget, tag));
                }
                return Ok(Some(Solution { items }));
            }

            let p2    = bridge_p1_to_p2(&inst, budget);
            let mut s = P2State::new(p2);
            s.cache_on   = hp.vnd_cache  == 1;
            s.vnd_resume = hp.vnd_resume == 1;
            s.no_synergy = hp.no_synergy == 1;
            s.syn_bound  = hp.syn_bound  == 1;
            s.prof_dup   = hp.prof_dup;
            s.vnd_fast   = hp.vnd_fast;
            bridge_load_solution(&mut s, &qr.results[0]);
            if hp.stage_stop == 4 {
                let wt = P2State::weight_selected(&s.ins, &s.sel);
                if wt > budget {
                    return Ok(stage_stop_pick(&inst.weights, budget, s.value as u64));
                }
                let items: Vec<usize> = (0..s.ins.n).filter(|&i| s.sel[i] != 0).collect();
                return Ok(Some(Solution { items }));
            }

            solve_hybrid(&mut s, &mut rng, hp.ils_rounds, hp.window_k, hp.core_half_dp,
                         hp.ils_stall_stop, hp.pert_ramp);

            if ext_pm > 0 {
                s.restore_best();
                if core_bnb(&mut s, &qr.bp, qr.bcross, hp.n_lambda_values, hp.core_hi, hp.core_work, hp.core_lam, hp.core_fix, hp.core_kmax, hp.core_pre) {
                    dp_refinement(&mut s, hp.core_half_dp);
                    local_search_vnd(&mut s, hp.window_k, true);
                    s.save_best();
                }
                s.restore_best();
            }

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