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
        pub ils_floor:       Option<usize>,
        pub ils_stall_min:   Option<usize>,
        pub lean_pre:        Option<usize>,
        pub bp_fast:         Option<usize>,
        pub w_fast:          Option<usize>,
        pub k11: Option<usize>, pub k12: Option<usize>, pub k21: Option<usize>,
        pub k13: Option<usize>, pub k31: Option<usize>,
        pub core_lo: Option<usize>, pub core_hi: Option<usize>,
        pub core_work: Option<usize>, pub core_lam: Option<usize>,
        pub cut_engine: Option<usize>,
    }

    struct Hparams {
        n_lambda_values: usize,
        ils_rounds:      usize,
        window_k:        usize,
        core_half_dp:    usize,
        rush_mode:       usize,
        ils_floor:       usize,
        ils_stall_min:   usize,
        lean_pre:        usize,
        bp_fast:         usize,
        w_fast:          usize,
        k11: usize, k12: usize, k21: usize, k13: usize, k31: usize,
        core_lo: usize, core_hi: usize, core_work: usize, core_lam: usize,
        cut_engine: usize,
    }

    impl Hparams {
        fn for_size(n: usize) -> Self {
            if n <= 1200 {
                Self {
                    n_lambda_values: 800,
                    ils_rounds:      8,
                    window_k:        250,
                    core_half_dp:    10,
                    rush_mode:       0,
                    ils_floor:       20,
                    ils_stall_min:   30,
                    lean_pre:        0,
                    bp_fast:         12223,
                    w_fast:          49151,
                    k11: 100, k12: 125, k21: 60, k13: 16, k31: 20,
                    core_lo: 60, core_hi: 20, core_work: 750000, core_lam: 2, cut_engine: 2,
                }
            } else if n <= 2000 {
                Self {
                    n_lambda_values: 1600,
                    ils_rounds:      400,
                    window_k:        220,
                    core_half_dp:    60,
                    rush_mode:       0,
                    ils_floor:       60,
                    ils_stall_min:   30,
                    lean_pre:        0,
                    bp_fast:         7,
                    w_fast:          63,
                    k11: 250, k12: 125, k21: 125, k13: 28, k31: 24,
                    core_lo: 0, core_hi: 0, core_work: 0, core_lam: 2, cut_engine: 2,
                }
            } else if n <= 4000 {
                Self {
                    n_lambda_values: 1600,
                    ils_rounds:      200,
                    window_k:        200,
                    core_half_dp:    50,
                    rush_mode:       0,
                    ils_floor:       60,
                    ils_stall_min:   30,
                    lean_pre:        0,
                    bp_fast:         7,
                    w_fast:          63,
                    k11: 250, k12: 125, k21: 125, k13: 28, k31: 24,
                    core_lo: 0, core_hi: 0, core_work: 0, core_lam: 2, cut_engine: 2,
                }
            } else {
                Self {
                    n_lambda_values: 1600,
                    ils_rounds:      100,
                    window_k:        180,
                    core_half_dp:    40,
                    rush_mode:       0,
                    ils_floor:       60,
                    ils_stall_min:   30,
                    lean_pre:        0,
                    bp_fast:         7,
                    w_fast:          63,
                    k11: 250, k12: 125, k21: 125, k13: 28, k31: 24,
                    core_lo: 0, core_hi: 0, core_work: 0, core_lam: 2, cut_engine: 2,
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
                if let Some(v) = m.get("rush_mode").and_then(|v| v.as_u64()) {
                    p.rush_mode = v as usize;
                }
                if let Some(v) = m.get("ils_floor").and_then(|v| v.as_u64()) {
                    p.ils_floor = v as usize;
                }
                if let Some(v) = m.get("ils_stall_min").and_then(|v| v.as_u64()) {
                    p.ils_stall_min = v as usize;
                }
                if let Some(v) = m.get("lean_pre").and_then(|v| v.as_u64()) {
                    p.lean_pre = v as usize;
                }
                if let Some(v) = m.get("bp_fast").and_then(|v| v.as_u64()) {
                    p.bp_fast = v as usize;
                }
                if let Some(v) = m.get("w_fast").and_then(|v| v.as_u64()) {
                    p.w_fast = v as usize;
                }
                if let Some(v) = m.get("k11").and_then(|v| v.as_u64()) { p.k11 = v as usize; }
                if let Some(v) = m.get("k12").and_then(|v| v.as_u64()) { p.k12 = v as usize; }
                if let Some(v) = m.get("k21").and_then(|v| v.as_u64()) { p.k21 = v as usize; }
                if let Some(v) = m.get("k13").and_then(|v| v.as_u64()) { p.k13 = v as usize; }
                if let Some(v) = m.get("k31").and_then(|v| v.as_u64()) { p.k31 = v as usize; }
                if let Some(v) = m.get("core_lo").and_then(|v| v.as_u64()) { p.core_lo = v as usize; }
                if let Some(v) = m.get("core_hi").and_then(|v| v.as_u64()) { p.core_hi = v as usize; }
                if let Some(v) = m.get("core_work").and_then(|v| v.as_u64()) { p.core_work = v as usize; }
                if let Some(v) = m.get("core_lam").and_then(|v| v.as_u64()) { p.core_lam = v as usize; }
                if let Some(v) = m.get("cut_engine").and_then(|v| v.as_u64()) { p.cut_engine = v as usize; }
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

    pub fn challenge_to_p1(challenge: &Challenge, w_fast: usize) -> P1Instance {
        let n = challenge.num_items;
        let mut edges: Vec<Edge> = Vec::new();

        for i in 0..n {
            edges.push(Edge { i, j: i, value: challenge.values[i] as f64 });
        }
        if w_fast & 8192 != 0 {
            for i in 0..n {
                let row = &challenge.interaction_values[i];
                let need = n - i + 16;
                if edges.capacity() - edges.len() < need { edges.reserve(need); }
                let mut len = edges.len();
                let base = edges.as_mut_ptr();
                let mut j = i + 1;
                unsafe {
                    while j + 16 <= n {
                        let mut acc = 0i32;
                        for t in 0..16 { acc |= *row.get_unchecked(j + t); }
                        if acc != 0 {
                            for t in 0..16 {
                                let v = *row.get_unchecked(j + t);
                                std::ptr::write(base.add(len), Edge { i, j: j + t, value: v as f64 });
                                len += (v != 0) as usize;
                            }
                        }
                        j += 16;
                    }
                    while j < n {
                        let v = row[j];
                        std::ptr::write(base.add(len), Edge { i, j, value: v as f64 });
                        len += (v != 0) as usize;
                        j += 1;
                    }
                    edges.set_len(len);
                }
            }
        } else if w_fast & 2048 != 0 {
            for i in 0..n {
                let row = &challenge.interaction_values[i];
                let need = n - i;
                if edges.capacity() - edges.len() < need { edges.reserve(need); }
                let mut len = edges.len();
                let base = edges.as_mut_ptr();
                let mut j = i + 1;
                unsafe {
                    while j + 4 <= n {
                        let v0 = row[j]; let v1 = row[j + 1];
                        let v2 = row[j + 2]; let v3 = row[j + 3];
                        if (v0 | v1 | v2 | v3) != 0 {
                            std::ptr::write(base.add(len), Edge { i, j, value: v0 as f64 });
                            len += (v0 != 0) as usize;
                            std::ptr::write(base.add(len), Edge { i, j: j + 1, value: v1 as f64 });
                            len += (v1 != 0) as usize;
                            std::ptr::write(base.add(len), Edge { i, j: j + 2, value: v2 as f64 });
                            len += (v2 != 0) as usize;
                            std::ptr::write(base.add(len), Edge { i, j: j + 3, value: v3 as f64 });
                            len += (v3 != 0) as usize;
                        }
                        j += 4;
                    }
                    while j < n {
                        let v = row[j];
                        std::ptr::write(base.add(len), Edge { i, j, value: v as f64 });
                        len += (v != 0) as usize;
                        j += 1;
                    }
                    edges.set_len(len);
                }
            }
        } else if w_fast & 32 != 0 {
            for i in 0..n {
                let row = &challenge.interaction_values[i];
                let mut j = i + 1;
                while j + 4 <= n {
                    let v0 = row[j]; let v1 = row[j + 1];
                    let v2 = row[j + 2]; let v3 = row[j + 3];
                    if (v0 | v1 | v2 | v3) != 0 {
                        if v0 != 0 { edges.push(Edge { i, j,         value: v0 as f64 }); }
                        if v1 != 0 { edges.push(Edge { i, j: j + 1,  value: v1 as f64 }); }
                        if v2 != 0 { edges.push(Edge { i, j: j + 2,  value: v2 as f64 }); }
                        if v3 != 0 { edges.push(Edge { i, j: j + 3,  value: v3 as f64 }); }
                    }
                    j += 4;
                }
                while j < n {
                    let v = row[j];
                    if v != 0 { edges.push(Edge { i, j, value: v as f64 }); }
                    j += 1;
                }
            }
        } else {
        for i in 0..n {
            for j in (i + 1)..n {
                let v = challenge.interaction_values[i][j];
                if v != 0 {
                    edges.push(Edge { i, j, value: v as f64 });
                }
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
    pub struct QKPResult { pub results: Vec<BudgetResult>, pub bp: Vec<i32>, pub bcross: i32 }

    pub struct UtilMatrix<'a> {
        pub n: usize,
        pub q: &'a [i32],
    }
    impl<'a> UtilMatrix<'a> {
        #[inline] pub fn get(&self, r: usize, c: usize) -> f64 {
            if r == c { 0.0 } else { self.q[r * self.n + c] as f64 }
        }
        #[inline] pub fn linear(&self, i: usize) -> f64 { self.q[i * self.n + i] as f64 }
    }

    pub fn compute_ofv_q(sel: &[usize], n: usize, q: &[i32], seen: &mut Vec<u8>) -> f64 {
        seen.clear();
        seen.resize(n, 0);
        let mut ids: Vec<usize> = Vec::with_capacity(sel.len());
        for &i in sel {
            if i < n && seen[i] == 0 { seen[i] = 1; ids.push(i); }
        }
        let mut ofv = 0.0f64;
        for (a, &i) in ids.iter().enumerate() {
            ofv += q[i * n + i] as f64;
            let row = &q[i * n..(i + 1) * n];
            for &j in &ids[a + 1..] {
                ofv += row[j] as f64;
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
        }

        pub struct HpfNode {
            pub wt:              f32,
            pub cst:             f32,
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
            pub spos:            i32,
            pub lz:              i32,
            pub snk_arc:         *mut HpfArc,
        }

        impl HpfNode {
            pub fn zeroed(num_params: i32) -> Self {
                HpfNode {
                    wt: 0.0, cst: 1.0,
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
                    spos: -1, lz: 0, snk_arc: std::ptr::null_mut(),
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
            pub src_r:                Vec<f32>,
            pub snk_r:                Vec<f32>,
            pub src_k:                usize,
            pub snk_k:                usize,
            pub max_bucket:           i32,
            pub lazy:                 bool,
            pub nodead:               bool,
            pub dupsnk:               bool,
            pub dupacc:               f32,
            pub snk_dirty:            Vec<*mut HpfArc>,
            pub lam_prev:             f32,
            pub lam_cur:              f32,
            pub in_snk:               bool,
            pub fwn2:                 bool,
            pub lifted_w:             f64,
            pub budget_w:             f64,
            pub bpstop:               bool,
            pub ext_pm:               i32,
            pub stop_at:              i32,
            pub pxskip:               bool,
            pub cur_spos:             i32,
            pub snk_i:                usize,
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
            (*s).lifted_w += (*current).cst as f64;
            loop {
                while !(*current).next_scan.is_null() {
                    let temp             = (*current).next_scan;
                    (*current).next_scan = (*(*current).next_scan).next;
                    current              = temp;
                    (*current).next_scan = (*current).child_list;
                    (*s).label_count[(*current).label as usize] -= 1;
                    (*current).label      = (*s).num_nodes;
                    (*current).breakpoint = theparam + 1;
                    (*s).lifted_w += (*current).cst as f64;
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

        #[inline(always)]
        unsafe fn snk_sync(s: *mut HpfState, nd: *mut HpfNode) {
            if (*nd).lz == 0 { return; }
            let sp  = (*nd).spos;
            let lam = if (*s).in_snk && sp < (*s).cur_spos { (*s).lam_cur } else { (*s).lam_prev };
            let v   = lam * (*nd).cst - (*nd).wt;
            let c   = if v > 0.0 { v } else { 0.0 };
            let arc = (*nd).snk_arc;
            (*arc).capacity = c;
            (*nd).excess    = 0.0 - c;
            (*nd).lz        = 0;
            let mut lo = 0usize;
            let mut hi = (*s).snk_dirty.len();
            while lo < hi {
                let mid = (lo + hi) >> 1;
                let q = (*(*(*(*s).snk_dirty.as_ptr().add(mid))).from).spos;
                if q < sp { lo = mid + 1; } else { hi = mid; }
            }
            (*s).snk_dirty.insert(lo, arc);
            if (*s).in_snk && lo <= (*s).snk_i { (*s).snk_i += 1; }
        }

        unsafe fn snk_retire(s: *mut HpfState, j: usize) {
            let si  = ((*s).sink - 1) as usize;
            let arc = (*s).adjacency_list[si].out_of_tree[j];
            let nd  = (*arc).from;
            if (*nd).lz != 0 {
                let v = (*s).lam_prev * (*nd).cst - (*nd).wt;
                let c = if v > 0.0 { v } else { 0.0 };
                (*arc).capacity = c;
                (*nd).excess    = 0.0 - c;
                (*nd).lz        = 0;
                return;
            }
            let sp = (*nd).spos;
            let mut lo = 0usize;
            let mut hi = (*s).snk_dirty.len();
            while lo < hi {
                let mid = (lo + hi) >> 1;
                let q = (*(*(*(*s).snk_dirty.as_ptr().add(mid))).from).spos;
                if q < sp { lo = mid + 1; } else { hi = mid; }
            }
            if lo < (*s).snk_dirty.len()
                && (*(*(*(*s).snk_dirty.as_ptr().add(lo))).from).spos == sp {
                (*s).snk_dirty.remove(lo);
                if (*s).in_snk && lo <= (*s).snk_i && (*s).snk_i > 0 { (*s).snk_i -= 1; }
            }
        }

        unsafe fn push_excess(s: *mut HpfState, strong_root: *mut HpfNode) {
            if !(*s).pxskip { snk_sync(s, strong_root); }
            let mut current = strong_root;
            while (*current).excess > 0.0 && !(*current).parent.is_null() {
                let parent = (*current).parent;
                snk_sync(s, parent);
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
            if (*s).fwn2 {
                let base = (*strong_node).out_of_tree.as_ptr();
                while i < size {
                    let out = *base.add(i);
                    let lt  = (*(*out).to).label;
                    let lf  = (*(*out).from).label;
                    if (lt == target) | (lf == target) {
                        (*strong_node).next_arc = i as i32;
                        *weak_node = if lt == target { (*out).to } else { (*out).from };
                        let last = (*strong_node).num_out_of_tree as usize - 1;
                        (*strong_node).out_of_tree[i] = (*strong_node).out_of_tree[last];
                        (*strong_node).out_of_tree.pop();
                        (*strong_node).num_out_of_tree -= 1;
                        return out;
                    }
                    i += 1;
                }
                (*strong_node).next_arc = (*strong_node).num_out_of_tree;
                return std::ptr::null_mut();
            }
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

        unsafe fn update_capacities(s: *mut HpfState, theparam: i32) {
            let lambda = (*s).max_degree_ratio - theparam as f32 * (*s).step;
            (*s).lam_cur = lambda;
            let src_idx = ((*s).source - 1) as usize;
            let snk_idx = ((*s).sink   - 1) as usize;
            let guard = lambda - (*s).step;
            {
                let ns = (*s).src_r.len();
                if (*s).lazy {
                    while (*s).src_k < ns && (*s).src_r[(*s).src_k] > guard {
                        let arc = (*s).adjacency_list[src_idx].out_of_tree[(*s).src_k];
                        snk_sync(s, (*arc).to);
                        (*s).src_k += 1;
                    }
                    let hi = lambda + (*s).step;
                    while (*s).snk_k > 0 && (*s).snk_r[(*s).snk_k - 1] >= hi {
                        (*s).snk_k -= 1;
                        snk_retire(s, (*s).snk_k);
                    }
                } else {
                    while (*s).src_k < ns && (*s).src_r[(*s).src_k] > guard { (*s).src_k += 1; }
                    let hi = lambda + (*s).step;
                    while (*s).snk_k > 0 && (*s).snk_r[(*s).snk_k - 1] >= hi { (*s).snk_k -= 1; }
                }
            }
            let size    = (*s).src_k;
            let sbase = (*s).adjacency_list[src_idx].out_of_tree.as_ptr();
            let nn    = (*s).num_nodes;
            if (*s).nodead {
                for i in 0..size {
                    let arc = *sbase.add(i);
                    let nd  = (*arc).to;
                    let v   = (*nd).wt - lambda * (*nd).cst;
                    let new_capacity = if v > 0.0 { v } else { 0.0 };
                    let delta = new_capacity - (*arc).capacity;
                    (*arc).capacity      = new_capacity;
                    (*arc).flow         += delta;
                    let ex = (*nd).excess + delta;
                    (*nd).excess         = ex;
                    if ex > 0.0 && (*nd).label < nn {
                        push_excess(s, nd);
                    }
                }
            } else {
                for i in 0..size {
                    let arc = *sbase.add(i);
                    let nd  = (*arc).to;
                    let v   = (*nd).wt - lambda * (*nd).cst;
                    let new_capacity = if v > 0.0 { v } else { 0.0 };
                    let delta = new_capacity - (*arc).capacity;
                    if delta < 0.0 { return; }
                    (*arc).capacity      = new_capacity;
                    (*arc).flow         += delta;
                    let ex = (*nd).excess + delta;
                    (*nd).excess         = ex;
                    if ex > 0.0 && (*nd).label < nn {
                        push_excess(s, nd);
                    }
                }
            }
            let kbase = (*s).adjacency_list[snk_idx].out_of_tree.as_ptr();
            if (*s).lazy {
                (*s).in_snk = true;
                (*s).snk_i  = 0;
                if (*s).nodead {
                    while (*s).snk_i < (*s).snk_dirty.len() {
                        let arc = *(*s).snk_dirty.as_ptr().add((*s).snk_i);
                        let nd  = (*arc).from;
                        (*s).cur_spos = (*nd).spos;
                        let v   = lambda * (*nd).cst - (*nd).wt;
                        let new_capacity = if v > 0.0 { v } else { 0.0 };
                        let delta = new_capacity - (*arc).capacity;
                        (*arc).capacity      = new_capacity;
                        (*arc).flow         += delta;
                        let ex = (*nd).excess - delta;
                        (*nd).excess         = ex;
                        if ex > 0.0 && (*nd).label < nn {
                            push_excess(s, nd);
                        }
                        (*s).snk_i += 1;
                    }
                } else {
                    while (*s).snk_i < (*s).snk_dirty.len() {
                        let arc = *(*s).snk_dirty.as_ptr().add((*s).snk_i);
                        let nd  = (*arc).from;
                        (*s).cur_spos = (*nd).spos;
                        let v   = lambda * (*nd).cst - (*nd).wt;
                        let new_capacity = if v > 0.0 { v } else { 0.0 };
                        let delta = new_capacity - (*arc).capacity;
                        if delta > 0.0 { (*s).in_snk = false; return; }
                        (*arc).capacity      = new_capacity;
                        (*arc).flow         += delta;
                        let ex = (*nd).excess - delta;
                        (*nd).excess         = ex;
                        if ex > 0.0 && (*nd).label < nn {
                            push_excess(s, nd);
                        }
                        (*s).snk_i += 1;
                    }
                }
                (*s).in_snk = false;
            } else {
                let size = (*s).snk_k;
                for i in 0..size {
                    let arc = *kbase.add(i);
                    let nd  = (*arc).from;
                    let v   = lambda * (*nd).cst - (*nd).wt;
                    let new_capacity = if v > 0.0 { v } else { 0.0 };
                    let delta = new_capacity - (*arc).capacity;
                    if delta > 0.0 { return; }
                    (*arc).capacity      = new_capacity;
                    (*arc).flow         += delta;
                    let ex = (*nd).excess - delta;
                    (*nd).excess         = ex;
                    if ex > 0.0 && (*nd).label < nn {
                        push_excess(s, nd);
                    }
                }
            }
            (*s).lam_prev = lambda;
            if (*s).dupsnk {
                let size = (*s).snk_k;
                let mut acc = 0.0f32;
                for i in 0..size {
                    let arc = *kbase.add(i);
                    let nd  = (*arc).from;
                    let v   = lambda * (*nd).cst - (*nd).wt;
                    let new_capacity = if v > 0.0 { v } else { 0.0 };
                    let delta = new_capacity - (*arc).capacity;
                    if delta > 0.0 { return; }
                    acc += new_capacity + delta;
                    let ex = (*nd).excess - delta;
                    if ex > 0.0 && (*nd).label < nn {
                        acc += 1.0;
                    }
                }
                (*s).dupacc += acc;
            }
            let hi = (*s).num_nodes - 1;
            (*s).highest_strong_label = if (*s).max_bucket < hi { (*s).max_bucket } else { hi };
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
            theparam = 1;
            if (*s).bpstop && (*s).lifted_w > (*s).budget_w && past_stop(s, 0) { return; }
            while theparam < (*s).num_params {
                update_capacities(s, theparam);
                loop {
                    let sr = get_highest_strong_root(s, theparam);
                    if sr.is_null() { break; }
                    process_root(s, sr);
                }
                theparam += 1;
                if (*s).bpstop && (*s).lifted_w > (*s).budget_w && past_stop(s, theparam - 1) { return; }
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

        pub struct BreakpointSets {
            pub sets: Vec<(i32, Vec<usize>)>,
        }

        pub fn get_breakpoints(inst: &P1Instance,
                                n_lambda_values: usize,
                                bp_fast: usize,
                                w_fast: usize,
                                ext_pm: i32) -> BreakpointSets {
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

            let ctor = (w_fast & 1024) != 0 && (w_fast & 16) != 0;
            let mut arc_list: Vec<HpfArc> = Vec::with_capacity(num_arcs as usize);
            if ctor {
                unsafe { arc_list.set_len(num_arcs as usize); }
            } else {
            for _ in 0..num_arcs as usize {
                arc_list.push(HpfArc {
                    from: std::ptr::null_mut(), to: std::ptr::null_mut(),
                    flow: 0.0, capacity: 0.0, direction: 1,
                });
            }
            }

            let mut first = 0usize;
            let mut odeg: Vec<u32> = Vec::new();
            if ctor {
                odeg = vec![0u32; num_nodes as usize];
                for e in &inst.edges[..n_items] {
                    adjacency_list[e.i + 2].wt += e.value as f32;
                }
                for e in &inst.edges[n_items..] {
                    let from_id = e.i;
                    let cap     = e.value as f32;
                    arc_list[first].capacity   = cap;
                    arc_list[first].direction  = 1;
                    arc_list[first].flow       = 0.0;
                    adjacency_list[from_id + 2].wt += cap;
                    odeg[from_id + 2] += 1;
                    first += 1;
                }
            } else {
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
                first += 1;
            }
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
                if ctor { arc_list[first].flow = 0.0; }
                first += 1;

                let snk_cap = if num_params > 0 {
                    let v = initial_lambda * cst - wt;
                    if v > 0.0 { v } else { 0.0 }
                } else { 0.0 };
                arc_list[first].capacity   = snk_cap;
                arc_list[first].direction  = 1;
                if ctor { arc_list[first].flow = 0.0; }
                first += 1;
            }

            let mut sentinels: Vec<HpfNode> = Vec::new();
            let strong_roots: Vec<HpfRoot> = if ctor {
                sentinels = (0..2 * num_nodes as usize).map(|_| HpfNode::zeroed(num_params)).collect();
                let base = sentinels.as_mut_ptr();
                (0..num_nodes as usize).map(|i| unsafe {
                    let start = base.add(2 * i);
                    let end   = base.add(2 * i + 1);
                    (*start).next = end;
                    (*end).prev   = start;
                    HpfRoot { start, end }
                }).collect()
            } else { unsafe {
                (0..num_nodes as usize).map(|_| init_root(num_params)).collect()
            } };
            let label_count = vec![0i32; num_nodes as usize];

            let mut state = HpfState {
                lazy: (bp_fast & 1) != 0,
                nodead: (bp_fast & 2) != 0,
                dupsnk: (bp_fast & 64) != 0,
                dupacc: 0.0,
                snk_dirty: Vec::new(),
                lam_prev: initial_lambda,
                lam_cur: initial_lambda,
                in_snk: false,
                fwn2: (w_fast & 64) != 0,
                lifted_w: 0.0,
                budget_w: inst.budgets[0] as f64,
                bpstop: (w_fast & 256) != 0,
                ext_pm,
                stop_at: -1,
                pxskip: (w_fast & 128) != 0,
                cur_spos: 0,
                snk_i: 0,
                src_r: Vec::new(), snk_r: Vec::new(), src_k: 0, snk_k: 0, max_bucket: 1,
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
                if ctor {
                    for k in 2..num_nodes as usize {
                        (*s).adjacency_list[k].out_of_tree.reserve_exact(odeg[k] as usize);
                    }
                    (*s).adjacency_list[0].out_of_tree.reserve_exact(n_items);
                    (*s).adjacency_list[1].out_of_tree.reserve_exact(n_items);
                    for e in &inst.edges[n_items..] {
                        let from_id = e.i;
                        let to_id   = e.j;
                        (*s).arc_list[first].from = &mut (*s).adjacency_list[from_id + 2];
                        (*s).arc_list[first].to   = &mut (*s).adjacency_list[to_id   + 2];
                        let arc_ptr = &mut (*s).arc_list[first] as *mut HpfArc;
                        (*s).adjacency_list[from_id + 2].out_of_tree.push(arc_ptr);
                        (*s).adjacency_list[from_id + 2].num_out_of_tree += 1;
                        first += 1;
                    }
                    for i in 2..num_nodes as usize {
                        (*s).arc_list[first].from = &mut (*s).adjacency_list[0];
                        (*s).arc_list[first].to   = &mut (*s).adjacency_list[i];
                        let a1 = &mut (*s).arc_list[first] as *mut HpfArc;
                        (*s).adjacency_list[0].out_of_tree.push(a1);
                        (*s).adjacency_list[0].num_out_of_tree += 1;
                        first += 1;
                        (*s).arc_list[first].from = &mut (*s).adjacency_list[i];
                        (*s).arc_list[first].to   = &mut (*s).adjacency_list[1];
                        let a2 = &mut (*s).arc_list[first] as *mut HpfArc;
                        (*s).adjacency_list[1].out_of_tree.push(a2);
                        (*s).adjacency_list[1].num_out_of_tree += 1;
                        first += 1;
                    }
                } else if w_fast & 16 != 0 {
                    for e in &inst.edges {
                        let from_id = e.i;
                        let to_id   = e.j;
                        if from_id == to_id { continue; }
                        (*s).arc_list[first].from = &mut (*s).adjacency_list[from_id + 2];
                        (*s).arc_list[first].to   = &mut (*s).adjacency_list[to_id   + 2];
                        let arc_ptr = &mut (*s).arc_list[first] as *mut HpfArc;
                        (*s).adjacency_list[from_id + 2].out_of_tree.push(arc_ptr);
                        (*s).adjacency_list[from_id + 2].num_out_of_tree += 1;
                        first += 1;
                    }
                    for i in 2..num_nodes as usize {
                        (*s).arc_list[first].from = &mut (*s).adjacency_list[0];
                        (*s).arc_list[first].to   = &mut (*s).adjacency_list[i];
                        let a1 = &mut (*s).arc_list[first] as *mut HpfArc;
                        (*s).adjacency_list[0].out_of_tree.push(a1);
                        (*s).adjacency_list[0].num_out_of_tree += 1;
                        first += 1;
                        (*s).arc_list[first].from = &mut (*s).adjacency_list[i];
                        (*s).arc_list[first].to   = &mut (*s).adjacency_list[1];
                        let a2 = &mut (*s).arc_list[first] as *mut HpfArc;
                        (*s).adjacency_list[1].out_of_tree.push(a2);
                        (*s).adjacency_list[1].num_out_of_tree += 1;
                        first += 1;
                    }
                } else {
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
                }

                {
                    let si = ((*s).source - 1) as usize;
                    let ns = (*s).adjacency_list[si].num_out_of_tree as usize;
                    let mut key: Vec<(f32, usize)> = Vec::with_capacity(ns);
                    for i in 0..ns {
                        let nd = (*(*s).adjacency_list[si].out_of_tree[i]).to;
                        key.push(((*nd).wt / (*nd).cst, i));
                    }
                    key.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
                    let old: Vec<*mut HpfArc> = (0..ns).map(|i| (*s).adjacency_list[si].out_of_tree[i]).collect();
                    for (j, &(r, i)) in key.iter().enumerate() {
                        (*s).adjacency_list[si].out_of_tree[j] = old[i];
                        (*s).src_r.push(r);
                    }
                    let _ = ns;
                }
                {
                    let ki = ((*s).sink - 1) as usize;
                    let ns = (*s).adjacency_list[ki].num_out_of_tree as usize;
                    let mut key: Vec<(f32, usize)> = Vec::with_capacity(ns);
                    for i in 0..ns {
                        let nd = (*(*s).adjacency_list[ki].out_of_tree[i]).from;
                        key.push(((*nd).wt / (*nd).cst, i));
                    }
                    key.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
                    let old: Vec<*mut HpfArc> = (0..ns).map(|i| (*s).adjacency_list[ki].out_of_tree[i]).collect();
                    for (j, &(r, i)) in key.iter().enumerate() {
                        (*s).adjacency_list[ki].out_of_tree[j] = old[i];
                        (*s).snk_r.push(r);
                    }
                    (*s).snk_k = ns;
                    if (*s).lazy {
                        for j in 0..ns {
                            let arc = (*s).adjacency_list[ki].out_of_tree[j];
                            let nd  = (*arc).from;
                            (*nd).spos    = j as i32;
                            (*nd).snk_arc = arc;
                            (*nd).lz      = 1;
                        }
                    }
                }
                (*s).src_k = 0;
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

                if w_fast & 512 != 0 {
                    pos_items.sort_by_key(|&(pos, _)| pos);
                } else {
                    pos_items.sort_unstable_by_key(|&(pos, _)| pos);
                }
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

                if !ctor { for i in 0..num_nodes as usize { free_root(&(*s).strong_roots[i]); } }
                let _ = &sentinels;

                BreakpointSets { sets }
            }
        }
    } 

    pub use hpf::{BreakpointSets, get_breakpoints};

    pub type IntArray = Vec<usize>;
    pub type DblArray = Vec<f64>;

    #[inline] pub fn ia_contains(a: &IntArray, v: usize) -> bool { a.contains(&v) }

    fn get_initial_node(right_nodes: &IntArray, um: &UtilMatrix<'_>,
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

    fn run_greedy_left_beta0(um:          &UtilMatrix<'_>,
                             mut left:    IntArray,
                             right_nodes: &IntArray,
                             budget:      i32,
                             weights:     &[i32],
                             mut ws:      Option<&mut WarmStartLeft>,
                             w_fast:      usize) -> IntArray {
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
        } else if w_fast & 4 != 0 {
            let mut inleft = vec![0u8; weights.len()];
            for &m in &left { inleft[m] = 1; }
            let rem0 = budget as f64 - cur_w;
            for &nd in right_nodes {
                if inleft[nd] != 0 { continue; }
                if weights[nd] as f64 > rem0 { continue; }
                cand_nodes.push(nd);
            }
            update_flag = true;
            for &nd in &cand_nodes {
                let mut contrib: f64 = left.iter()
                    .map(|&m| um.get(nd, m)).sum();
                contrib += um.linear(nd);
                contrib /= weights[nd] as f64;
                cand_contribs.push(contrib);
            }
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
                contrib += um.linear(nd);
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

    fn run_greedy_right_beta0(um:              &UtilMatrix<'_>,
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
                contrib -= um.linear(nd);
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

    fn run_greedy_left(um:          &UtilMatrix<'_>,
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
                contrib += um.linear(nd);
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

    fn run_greedy_right(um:              &UtilMatrix<'_>,
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
                contrib -= um.linear(nd);
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
                  right_init: IntArray,
                  w_fast:     usize,
                  qmat:       &[i32]) -> GreedyResults {
        let n_nodes = inst.n_items;
        let budget = inst.budgets[0];
        let um = UtilMatrix { n: n_nodes, q: qmat };
        let all_nodes: IntArray = (0..n_nodes).collect();

        let left = run_greedy_left_beta0(&um, left_init, &all_nodes,
                                         budget, &inst.weights, None, w_fast);
        let after_right = run_greedy_right_beta0(&um, right_init,
                                                  budget, &inst.weights, None);
        let right = run_greedy_left_beta0(&um, after_right, &all_nodes,
                                          budget, &inst.weights, None, w_fast);

        GreedyResults {
            left_nodes: vec![left],
            right_nodes: vec![right],
        }
    }

    pub fn run_bp_algorithm(inst: &P1Instance, n_lambda_values: usize, bp_fast: usize, w_fast: usize, qmat: &[i32], ext_pm: i32) -> QKPResult {
        let bps = get_breakpoints(inst, n_lambda_values, bp_fast, w_fast, ext_pm);
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

        let gr = run_greedy(inst, left_init, right_init, w_fast, qmat);
        let mut results: Vec<BudgetResult> = Vec::with_capacity(inst.n_budgets);
        for bi in 0..inst.n_budgets {
            let mut seen: Vec<u8> = Vec::new();
            let ofv_left  = compute_ofv_q(&gr.left_nodes[bi],  inst.n_items, qmat, &mut seen);
            let ofv_right = compute_ofv_q(&gr.right_nodes[bi], inst.n_items, qmat, &mut seen);
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
        let mut bp: Vec<i32> = Vec::new();
        let mut bcross = -1i32;
        if ext_pm > 0 {
            bp = vec![i32::MAX; inst.n_items];
            for (pos, nodes) in &bps.sets { for &nd in nodes { bp[nd] = *pos; } }
            if right_groups > 0 && right_groups <= n_breakpoints { bcross = bps.sets[right_groups - 1].0; }
        }
        QKPResult { results, bp, bcross }
    }

    pub struct P2Instance {
        pub n:                     usize,
        pub m:                     i64,
        pub capacity:              i32,
        pub w:                     Vec<i32>,
        pub q:                     Vec<i32>,
        pub interaction_potential: Vec<Ll>,
        pub ip_rank:               Vec<u32>,
        pub bbu_ranked:            bool,
        pub s11_cap:               bool,
        pub s12_cap:               bool,
        pub s21_cap:               bool,
        pub s21_blk:               bool,
        pub s23_cap:               bool,
        pub s12_blk:               bool,
        pub s13_cap:               bool,
        pub s31_cap:               bool,
        pub s12_gq:                bool,
        pub xt_split:              bool,
        pub w_fast:                usize,
        pub k11: usize, pub k12: usize, pub k21: usize, pub k13: usize, pub k31: usize,
        pub qrowmax:               Vec<i32>,
    }

    impl P2Instance {
        #[inline] pub fn q_get(&self, i: usize, j: usize) -> i32 { self.q[i * self.n + j] }
        #[inline] pub fn q_set(&mut self, i: usize, j: usize, v: i32) { self.q[i * self.n + j] = v; }
    }

    pub fn bridge_p1_to_p2(inst: &P1Instance, budget: i32, w_fast_b: usize) -> (P2Instance, Vec<Ll>) {
        let n = inst.n_items;
        let mut p2 = P2Instance {
            n,
            m:                     inst.n_edges as i64,
            capacity:              budget,
            w:                     inst.weights.clone(),
            q:                     vec![0i32; n * n],
            interaction_potential: vec![0; n],
            qrowmax:               vec![0i32; n],
            ip_rank:               Vec::new(),
            bbu_ranked:            false,
            s11_cap:               false,
            s12_cap:               false,
            s21_cap:               false,
            s21_blk:               false,
            s23_cap:               false,
            s12_blk:               false,
            s13_cap:               false,
            s31_cap:               false,
            s12_gq:                false,
            xt_split:              false,
            w_fast:                63,
            k11: 250, k12: 125, k21: 125, k13: 28, k31: 24,
        };
        if w_fast_b & 32768 != 0 {
            for e in &inst.edges[..n] {
                let u = e.i; let qv = e.value as i32;
                p2.q[u * n + u] = qv;
                p2.interaction_potential[u] += qv as Ll;
            }
            for e in &inst.edges[n..] {
                let u = e.i; let v = e.j; let qv = e.value as i32;
                p2.q[u * n + v] = qv;
                p2.q[v * n + u] = qv;
                p2.interaction_potential[u] += qv as Ll;
                p2.interaction_potential[v] += qv as Ll;
                if qv > p2.qrowmax[u] { p2.qrowmax[u] = qv; }
                if qv > p2.qrowmax[v] { p2.qrowmax[v] = qv; }
            }
        } else {
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
                if qv > p2.qrowmax[u] { p2.qrowmax[u] = qv; }
                if qv > p2.qrowmax[v] { p2.qrowmax[v] = qv; }
            }
        }
        }
        let mut rank: Vec<u32> = (0..n as u32).collect();
        {
            let ip = &p2.interaction_potential;
            rank.sort_unstable_by(|&a, &b| ip[b as usize].cmp(&ip[a as usize]).then(a.cmp(&b)));
        }
        p2.ip_rank = rank;
        let total_interactions = p2.interaction_potential.clone();
        (p2, total_interactions)
    }

    pub struct P2State {
        pub ins:         P2Instance,
        pub ver:         u64,
        pub cs_ver:      u64,
        pub cs_unused:   Vec<ItemScore>,
        pub cs_used:     Vec<ItemScore>,
        pub cs_used_idx: Vec<usize>,
        pub cs_present:  Vec<u8>,
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
    }

    impl P2State {
        pub fn new(ins: P2Instance) -> Self {
            let n = ins.n;
            P2State {
                ver:         0,
                cs_ver:      u64::MAX,
                cs_unused:   Vec::with_capacity(n),
                cs_used:     Vec::with_capacity(n),
                cs_present:  vec![0u8; n],
                cs_used_idx: Vec::new(),
                sel:         vec![0u8; n],
                best_sel:    vec![0u8; n],
                best_contrib: vec![0i32; n],
                contrib:     vec![0i32; n],
                value:       0,
                weight:      0,
                count:       0,
                best_value:  NEG_INF,
                best_weight: 0,
                best_count:  -1,
                ins,
            }
        }

        pub fn clear(&mut self) {
            self.ver = self.ver.wrapping_add(1);
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
            self.ver = self.ver.wrapping_add(1);
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
            self.ver = self.ver.wrapping_add(1);
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
            self.ver = self.ver.wrapping_add(1);
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

            if self.ins.xt_split {
                let cptr = self.contrib.as_mut_ptr();
                for &rm in removals {
                    let row = &q[rm * n..(rm + 1) * n];
                    let vp = row.as_ptr();
                    unsafe { for j in 0..n { *cptr.add(j) -= *vp.add(j); } }
                }
                for &add in additions {
                    let row = &q[add * n..(add + 1) * n];
                    let vp = row.as_ptr();
                    unsafe { for j in 0..n { *cptr.add(j) += *vp.add(j); } }
                }
            } else {
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
            self.ver = self.ver.wrapping_add(1);
            self.sel.copy_from_slice(&self.best_sel);
            self.contrib.copy_from_slice(&self.best_contrib);
            self.value  = self.best_value;
            self.weight = self.best_weight;
            self.count  = self.best_count;
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
    pub struct ItemScore { pub id: usize, pub score: Ll }

    fn cs_refresh(s: &mut P2State) {
        if s.cs_ver == s.ver { return; }
        let n = s.ins.n;
        s.cs_unused.clear();
        s.cs_used.clear();
        s.cs_used_idx.clear();
        for i in 0..n {
            let w = s.ins.w[i].max(1) as Ll;
            if s.sel[i] == 0 {
                let gain = s.contrib[i] as Ll + s.ins.q_get(i, i) as Ll;
                s.cs_unused.push(ItemScore { id: i, score: gain * 1_000_000 / w });
            } else {
                s.cs_used.push(ItemScore { id: i, score: s.contrib[i] as Ll * 1_000_000 / w });
                s.cs_used_idx.push(i);
            }
        }
        s.cs_unused.sort_unstable_by(|a, b| b.score.cmp(&a.score));
        s.cs_used.sort_unstable_by(|a, b| a.score.cmp(&b.score));
        s.cs_ver = s.ver;
    }

    fn build_best_unused(s: &mut P2State, max_k: usize) -> Vec<ItemScore> {
        let n = s.ins.n;
        cs_refresh(s);
        if n < 300 || max_k < 4 {
            let mut arr: Vec<ItemScore> = Vec::with_capacity(max_k.min(s.cs_unused.len()));
            let take = max_k.min(s.cs_unused.len());
            arr.extend_from_slice(&s.cs_unused[..take]);
            return arr;
        }

        let synergy_k = (max_k.min(56) / 4).max(1);
        let density_k = max_k.saturating_sub(synergy_k);
        let take = density_k.min(s.cs_unused.len());
        let mut arr: Vec<ItemScore> = Vec::with_capacity(take + synergy_k);
        arr.extend_from_slice(&s.cs_unused[..take]);

        for is in &arr {
            s.cs_present[is.id] = 1;
        }

        let mut synergy: Vec<ItemScore> = Vec::with_capacity(synergy_k);
        if s.ins.bbu_ranked {
            for t in 0..n {
                let i = s.ins.ip_rank[t] as usize;
                if s.sel[i] != 0 || s.cs_present[i] != 0 { continue; }
                synergy.push(ItemScore { id: i, score: s.ins.interaction_potential[i] });
                if synergy.len() == synergy_k { break; }
            }
            for is in &arr {
                s.cs_present[is.id] = 0;
            }
            synergy.sort_unstable_by(|a, b| b.score.cmp(&a.score));
            arr.extend(synergy);
            return arr;
        }
        let mut worst = 0usize;
        let mut worst_score: Ll = 0;
        for i in 0..n {
            if s.sel[i] != 0 || s.cs_present[i] != 0 { continue; }
            let score = s.ins.interaction_potential[i];
            if synergy.len() < synergy_k {
                synergy.push(ItemScore { id: i, score });
                if synergy.len() == synergy_k {
                    worst = 0;
                    for j in 1..synergy.len() {
                        if synergy[j].score < synergy[worst].score { worst = j; }
                    }
                    worst_score = synergy[worst].score;
                }
                continue;
            }
            if score > worst_score {
                synergy[worst] = ItemScore { id: i, score };
                worst = 0;
                for j in 1..synergy.len() {
                    if synergy[j].score < synergy[worst].score { worst = j; }
                }
                worst_score = synergy[worst].score;
            }
        }

        for is in &arr {
            s.cs_present[is.id] = 0;
        }

        synergy.sort_unstable_by(|a, b| b.score.cmp(&a.score));
        arr.extend(synergy);
        arr
    }

    fn build_worst_used(s: &mut P2State, max_k: usize) -> Vec<ItemScore> {
        let n = s.ins.n;
        cs_refresh(s);
        if n < 300 || max_k < 4 {
            let take = max_k.min(s.cs_used.len());
            let mut arr: Vec<ItemScore> = Vec::with_capacity(take);
            arr.extend_from_slice(&s.cs_used[..take]);
            return arr;
        }

        let raw_k = (max_k.min(56) / 4).max(1);
        let density_k = max_k.saturating_sub(raw_k);
        let take = density_k.min(s.cs_used.len());
        let mut arr: Vec<ItemScore> = Vec::with_capacity(take + raw_k);
        arr.extend_from_slice(&s.cs_used[..take]);

        if take == s.cs_used.len() {
            return arr;
        }
        for is in &arr {
            s.cs_present[is.id] = 1;
        }

        let mut raw: Vec<ItemScore> = Vec::with_capacity(raw_k);
        let mut worst = 0usize;
        let mut worst_score: Ll = 0;
        let idxs: &[usize] = if s.ins.w_fast & 4096 != 0 { &s.cs_used_idx } else { &[] };
        let use_idx = s.ins.w_fast & 4096 != 0;
        let lim = if use_idx { idxs.len() } else { n };
        for t in 0..lim {
            let i = if use_idx { unsafe { *idxs.get_unchecked(t) } } else { t };
            if s.sel[i] == 0 || s.cs_present[i] != 0 { continue; }
            let score = s.contrib[i] as Ll;
            if raw.len() < raw_k {
                raw.push(ItemScore { id: i, score });
                if raw.len() == raw_k {
                    worst = 0;
                    for j in 1..raw.len() {
                        if raw[j].score > raw[worst].score { worst = j; }
                    }
                    worst_score = raw[worst].score;
                }
                continue;
            }
            if score < worst_score {
                raw[worst] = ItemScore { id: i, score };
                worst = 0;
                for j in 1..raw.len() {
                    if raw[j].score > raw[worst].score { worst = j; }
                }
                worst_score = raw[worst].score;
            }
        }

        for is in &arr {
            s.cs_present[is.id] = 0;
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
        let n = s.ins.n;
        let mut s11_maxbase: Ll = Ll::MIN;
        let mut ubase: Vec<Ll>    = Vec::with_capacity(cu);
        let mut uw:    Vec<i32>   = Vec::with_capacity(cu);
        let mut uid:   Vec<usize> = Vec::with_capacity(cu);
        for y in 0..cu {
            let a = unused[y].id;
            let v = s.contrib[a] as Ll + s.ins.q_get(a, a) as Ll;
            if v > s11_maxbase { s11_maxbase = v; }
            ubase.push(v); uw.push(s.ins.w[a]); uid.push(a);
        }
        let mut w11min: i32 = i32::MAX;
        for &w in &uw { if w < w11min { w11min = w; } }
        let mut sufmax: Vec<Ll> = Vec::new();
        if s.ins.s11_cap {
            sufmax = vec![Ll::MIN; cu + 1];
            for y in (0..cu).rev() {
                sufmax[y] = if ubase[y] > sufmax[y + 1] { ubase[y] } else { sufmax[y + 1] };
            }
        }
        for x in 0..cs {
            let rm     = used[x].id;
            let budget = s.slack() + s.ins.w[rm];
            let crm    = s.contrib[rm] as Ll;
            if budget < w11min { continue; }
            if s11_maxbase.saturating_sub(crm) <= best_delta { continue; }
            let row_rm = &s.ins.q[rm * n..(rm + 1) * n];
            if s.ins.s11_cap {
                for y in 0..cu {
                    if sufmax[y].saturating_sub(crm) <= best_delta { break; }
                    let d = ubase[y] - crm - row_rm[uid[y]] as Ll;
                    if d > best_delta && uw[y] <= budget {
                        best_delta = d; br = Some(rm); ba = Some(uid[y]);
                    }
                }
            } else {
                for y in 0..cu {
                    let d = ubase[y] - crm - row_rm[uid[y]] as Ll;
                    if d > best_delta && uw[y] <= budget {
                        best_delta = d; br = Some(rm); ba = Some(uid[y]);
                    }
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
        let s12cap = s.ins.s12_cap;
        let unused = build_best_unused(s, window_k);
        let used   = build_worst_used(s, window_k);
        let cu = unused.len();
        let cs = used.len();
        let n = s.ins.n;
        let q = &s.ins.q;
        let mut add_bases = Vec::with_capacity(cu);
        let mut uw:  Vec<i32>   = Vec::with_capacity(cu);
        let mut uid: Vec<usize> = Vec::with_capacity(cu);
        for item in &unused {
            let id = item.id;
            add_bases.push(s.contrib[id] as Ll + q[id * n + id] as Ll);
            uw.push(s.ins.w[id]); uid.push(id);
        }
        let mut wsuf: Vec<i32> = vec![i32::MAX; cu + 1];
        for x in (0..cu).rev() {
            wsuf[x] = if uw[x] < wsuf[x + 1] { uw[x] } else { wsuf[x + 1] };
        }
        let mut effective = vec![0; cu];
        let mut best_delta: Ll = 0;
        let mut br = None;
        let mut ba = None;
        let mut bb = None;
        let s12gq = s.ins.s12_gq && s12cap;
        let mut s12_rowsuf: Vec<Ll> = vec![Ll::MIN; cu];
        if s12gq {
            for x in 0..cu { s12_rowsuf[x] = s.ins.qrowmax[uid[x]] as Ll; }
        } else {
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
        }
        let mut s12_sufmax: Vec<Ll> = vec![Ll::MIN; cu + 1];
        let s12blk = s.ins.s12_blk && s12cap;
        let mut basesuf:   Vec<Ll> = Vec::new();
        let mut b2suf:     Vec<Ll> = Vec::new();
        let mut rowsufmax: Vec<Ll> = Vec::new();
        if s12blk {
            basesuf   = vec![Ll::MIN; cu + 1];
            b2suf     = vec![Ll::MIN; cu + 1];
            rowsufmax = vec![Ll::MIN; cu + 1];
            let mut t1 = Ll::MIN;
            let mut t2 = Ll::MIN;
            for x in (0..cu).rev() {
                let e = add_bases[x];
                if e > t1 { t2 = t1; t1 = e; } else if e > t2 { t2 = e; }
                basesuf[x] = t1;
                b2suf[x]   = t1.saturating_add(t2);
                rowsufmax[x] = if s12_rowsuf[x] > rowsufmax[x + 1] { s12_rowsuf[x] } else { rowsufmax[x + 1] };
            }
        }
        let row_rm_all = &s.ins.q;
        if s12blk {
            for si in 0..cs {
                let rm     = used[si].id;
                let budget = s.slack() + s.ins.w[rm];
                let lost   = s.contrib[rm] as Ll;
                let row_rm = &row_rm_all[rm * n..(rm + 1) * n];
                for x in 0..cu {
                    if b2suf[x].saturating_add(rowsufmax[x]).saturating_sub(lost) <= best_delta { break; }
                    let a  = uid[x];
                    let wa = uw[x];
                    if wa >= budget { continue; }
                    let ca_eff = add_bases[x] - row_rm[a] as Ll;
                    if ca_eff
                        .saturating_add(basesuf[x + 1])
                        .saturating_add(s12_rowsuf[x])
                        .saturating_sub(lost) <= best_delta { continue; }
                    let wlim = budget - wa;
                    if wlim < wsuf[x + 1] { continue; }
                    let row_a = &q[a * n..(a + 1) * n];
                    let cab = ca_eff - lost;
                    let rs  = s12_rowsuf[x];
                    for y in (x + 1)..cu {
                        if cab.saturating_add(basesuf[y]).saturating_add(rs) <= best_delta { break; }
                        let d = cab + (add_bases[y] - row_rm[uid[y]] as Ll) + row_a[uid[y]] as Ll;
                        if d > best_delta && uw[y] <= wlim {
                            best_delta = d; br = Some(rm); ba = Some(a); bb = Some(uid[y]);
                        }
                    }
                }
            }
            if let (Some(r), Some(a), Some(b)) = (br, ba, bb) {
                s.exchange_transaction(&[r], &[a, b]);
                return true;
            }
            return false;
        }
        for si in 0..cs {
            let rm     = used[si].id;
            let budget = s.slack() + s.ins.w[rm];
            let lost   = s.contrib[rm] as Ll;
            let row_rm = &row_rm_all[rm * n..(rm + 1) * n];
            s12_sufmax[cu] = Ll::MIN;
            for x in (0..cu).rev() {
                let e = add_bases[x] - row_rm[uid[x]] as Ll;
                effective[x] = e;
                s12_sufmax[x] = if e > s12_sufmax[x + 1] { e } else { s12_sufmax[x + 1] };
            }
            for x in 0..cu {
                let a  = unused[x].id;
                let wa = s.ins.w[a];
                if wa >= budget { continue; }
                let ca_eff = effective[x];
                if ca_eff
                    .saturating_add(s12_sufmax[x + 1])
                    .saturating_add(s12_rowsuf[x])
                    .saturating_sub(lost) <= best_delta { continue; }
                let wlim = budget - wa;
                if wlim < wsuf[x + 1] { continue; }
                let row_a = &q[a * n..(a + 1) * n];
                let cab = ca_eff - lost;
                if s12cap {
                    let rs = s12_rowsuf[x];
                    for y in (x+1)..cu {
                        if cab.saturating_add(s12_sufmax[y]).saturating_add(rs) <= best_delta { break; }
                        let d = cab + effective[y] + row_a[uid[y]] as Ll;
                        if d > best_delta && uw[y] <= wlim {
                            best_delta = d; br = Some(rm); ba = Some(a); bb = Some(uid[y]);
                        }
                    }
                } else {
                    for y in (x+1)..cu {
                        let d = cab + effective[y] + row_a[uid[y]] as Ll;
                        if d > best_delta && uw[y] <= wlim {
                            best_delta = d; br = Some(rm); ba = Some(a); bb = Some(uid[y]);
                        }
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
        let s21cap = s.ins.s21_cap;
        let unused = build_best_unused(s, window_k);
        let used   = build_worst_used(s, window_k);
        let cu = unused.len();
        let cs = used.len();
        let n = s.ins.n;
        let q = &s.ins.q;
        let mut s21_min_lost: Ll = Ll::MAX;
        let mut removal_pairs: Vec<(usize, usize, i32, Ll)> =
            Vec::with_capacity(cs.saturating_mul(cs.saturating_sub(1)) / 2);
        for x in 0..cs {
            let r1 = used[x].id;
            let row_r1 = &q[r1 * n..(r1 + 1) * n];
            for y in (x + 1)..cs {
                let r2 = used[y].id;
                let s21_lost = s.contrib[r1] as Ll + s.contrib[r2] as Ll - row_r1[r2] as Ll;
                removal_pairs.push((
                    r1,
                    r2,
                    s.ins.w[r1] + s.ins.w[r2],
                    s21_lost,
                ));
                if s21_lost < s21_min_lost { s21_min_lost = s21_lost; }
            }
        }
        let m_pairs = removal_pairs.len();
        let s21blk = s.ins.s21_blk && s21cap;
        let mut lostsuf: Vec<Ll> = Vec::new();
        let mut bsuf:    Vec<Ll> = Vec::new();
        if s21blk {
            lostsuf = vec![Ll::MAX; m_pairs];
            bsuf    = vec![Ll::MAX; cs + 1];
            let mut off = m_pairs;
            for x in (0..cs).rev() {
                let len = cs - 1 - x;
                off -= len;
                let mut run = Ll::MAX;
                for t in (0..len).rev() {
                    let l = removal_pairs[off + t].3;
                    if l < run { run = l; }
                    lostsuf[off + t] = run;
                }
                bsuf[x] = if run < bsuf[x + 1] { run } else { bsuf[x + 1] };
            }
        }
        let mut lostmin: Vec<Ll> = Vec::new();
        if s21cap && !s21blk {
            lostmin = vec![Ll::MAX; m_pairs + 1];
            for k in (0..m_pairs).rev() {
                let l = removal_pairs[k].3;
                lostmin[k] = if l < lostmin[k + 1] { l } else { lostmin[k + 1] };
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
            let mut min_cross1 = Ll::MAX;
            let mut min_cross2 = Ll::MAX;
            for u in used.iter() {
                let cross = row_add[u.id] as Ll;
                if cross < min_cross1 {
                    min_cross2 = min_cross1;
                    min_cross1 = cross;
                } else if cross < min_cross2 {
                    min_cross2 = cross;
                }
            }
            let min_cross = min_cross1.saturating_add(min_cross2);
            if cadd.saturating_sub(min_cross).saturating_sub(s21_min_lost) <= best_delta { continue; }
            let wadd  = s.ins.w[add];
            let slack = s.slack();
            if s21blk {
                let base = cadd.saturating_sub(min_cross);
                let mut off = 0usize;
                for x in 0..cs {
                    if base.saturating_sub(bsuf[x]) <= best_delta { break; }
                    let len = cs - 1 - x;
                    for t in 0..len {
                        let k = off + t;
                        if base.saturating_sub(lostsuf[k]) <= best_delta { break; }
                        let (r1, r2, removed_weight, lost) = removal_pairs[k];
                        let d = cadd - row_add[r1] as Ll - row_add[r2] as Ll - lost;
                        if d > best_delta && wadd <= slack + removed_weight {
                            best_delta = d; br1 = Some(r1); br2 = Some(r2); ba = Some(add);
                        }
                    }
                    off += len;
                }
                continue;
            }
            let mut kb = m_pairs;
            if s21cap {
                let base = cadd.saturating_sub(min_cross);
                let mut lo = 0usize;
                let mut hi = m_pairs;
                while lo < hi {
                    let mid = (lo + hi) >> 1;
                    if base.saturating_sub(lostmin[mid]) > best_delta { lo = mid + 1; } else { hi = mid; }
                }
                kb = lo;
            }
            for &(r1, r2, removed_weight, lost) in &removal_pairs[..kb] {
                let d = cadd - row_add[r1] as Ll - row_add[r2] as Ll - lost;
                if d > best_delta && wadd <= slack + removed_weight {
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

        let mut tu = vec![0i32; cs * cu];
        for a in 0..cu {
            let row = &q[unused[a].id * n..(unused[a].id + 1) * n];
            for b in 0..cs {
                tu[b * cu + a] = row[used[b].id];
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

        let mut m: Vec<Ll> = vec![0; cu];
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
                let (tx, ty) = (&tu[x * cu..x * cu + cu], &tu[y * cu..y * cu + cu]);
                for p in 0..cu { m[p] = tx[p] as Ll + ty[p] as Ll; }
                for &(p, q2, wp, pair_w, base) in &add_pairs {
                    if wp >= budget || pair_w > budget { continue; }
                    let d = base - m[p] - m[q2] - lost;
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
        let k = window_k.min(s.ins.k13);
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

        let cu2 = unused.len();
        let mut ubase: Vec<Ll>    = Vec::with_capacity(cu2);
        let mut uw:    Vec<i32>   = Vec::with_capacity(cu2);
        let mut uid:   Vec<usize> = Vec::with_capacity(cu2);
        for it in &unused {
            let a = it.id;
            ubase.push(s.contrib[a] as Ll + q[a * n + a] as Ll);
            uw.push(s.ins.w[a]); uid.push(a);
        }
        let mut wmin: i32 = i32::MAX;
        for &w in &uw { if w < wmin { wmin = w; } }
        let mut wsuf1: Vec<i32> = vec![i32::MAX; cu2 + 1];
        let mut wsuf2: Vec<i32> = vec![i32::MAX; cu2 + 1];
        for x in (0..cu2).rev() {
            let (a1, a2) = (wsuf1[x + 1], wsuf2[x + 1]);
            if uw[x] < a1 {
                wsuf1[x] = uw[x];
                wsuf2[x] = if a1 == i32::MAX { i32::MAX } else { a1 };
            } else {
                wsuf1[x] = a1;
                wsuf2[x] = if a1 != i32::MAX && uw[x] < a2 { uw[x] } else { a2 };
            }
        }
        let mut rmax: Vec<Ll> = vec![Ll::MIN; cu2];
        for i in 0..cu2 {
            let row_i = &q[uid[i] * n..(uid[i] + 1) * n];
            let mut m = Ll::MIN;
            for j in 0..cu2 {
                if j != i {
                    let v = row_i[uid[j]] as Ll;
                    if v > m { m = v; }
                }
            }
            rmax[i] = m;
        }
        let s13cap = s.ins.s13_cap;
        let mut hsuf13: Vec<Ll> = Vec::new();
        if s13cap { hsuf13 = vec![Ll::MIN; cu2 + 1]; }
        let mut g: Vec<Ll> = vec![0; cu2];
        for rm_score in &used {
            let rm = rm_score.id;
            let budget = s.slack() + s.ins.w[rm];
            if 3 * wmin > budget { continue; }
            let lost = s.contrib[rm] as Ll;
            let row_rm = &q[rm * n..(rm + 1) * n];
            for i in 0..cu2 { g[i] = ubase[i] - row_rm[uid[i]] as Ll; }
            {
                let (mut h1, mut h2, mut h3) = (Ll::MIN, Ll::MIN, Ll::MIN);
                for i in 0..cu2 {
                    let h = g[i].saturating_add(rmax[i]);
                    if h > h1 { h3 = h2; h2 = h1; h1 = h; }
                    else if h > h2 { h3 = h2; h2 = h; }
                    else if h > h3 { h3 = h; }
                }
                if h1.saturating_add(h2).saturating_add(h3).saturating_sub(lost) <= best_delta {
                    continue;
                }
            }

            let bud2 = budget - 2 * wmin;
            let bud1 = budget - wmin;
            if s13cap {
                for i in (0..cu2).rev() {
                    let v = g[i].saturating_add(rmax[i]).saturating_add(rmax[i]);
                    hsuf13[i] = if v > hsuf13[i + 1] { v } else { hsuf13[i + 1] };
                }
            }
            for ai in 0..cu2 {
                let a = uid[ai];
                let wa = uw[ai];
                if wa > bud2 { continue; }
                if wsuf2[ai + 1] == i32::MAX
                    || wa.saturating_add(wsuf1[ai + 1]).saturating_add(wsuf2[ai + 1]) > budget {
                    continue;
                }
                let row_a = &q[a * n..(a + 1) * n];
                let da = g[ai];

                for bi in (ai + 1)..cu2 {
                    let b = uid[bi];
                    let wab = wa + uw[bi];
                    if wab > bud1 { continue; }
                    if wab.saturating_add(wsuf1[bi + 1]) > budget { continue; }
                    let row_b = &q[b * n..(b + 1) * n];
                    let dab = da + g[bi] + row_a[b] as Ll - lost;
                    let wlim = budget - wab;

                    if s13cap {
                        for ci in (bi + 1)..cu2 {
                            if dab.saturating_add(hsuf13[ci]) <= best_delta { break; }
                            let cid = uid[ci];
                            let delta = dab + g[ci]
                                + row_a[cid] as Ll
                                + row_b[cid] as Ll;
                            if delta > best_delta && uw[ci] <= wlim {
                                best_delta = delta;
                                best_rm = Some(rm);
                                best_a = Some(a);
                                best_b = Some(b);
                                best_c = Some(cid);
                            }
                        }
                    } else {
                    for ci in (bi + 1)..cu2 {
                        let cid = uid[ci];
                        let delta = dab + g[ci]
                            + row_a[cid] as Ll
                            + row_b[cid] as Ll;

                        if delta > best_delta && uw[ci] <= wlim {
                            best_delta = delta;
                            best_rm = Some(rm);
                            best_a = Some(a);
                            best_b = Some(b);
                            best_c = Some(cid);
                        }
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

        let cu2 = unused.len();
        let cs2 = used.len();
        let mut ubase: Vec<Ll>    = Vec::with_capacity(cu2);
        let mut uw:    Vec<i32>   = Vec::with_capacity(cu2);
        let mut uid:   Vec<usize> = Vec::with_capacity(cu2);
        for it in &unused {
            let a = it.id;
            ubase.push(s.contrib[a] as Ll + q[a * n + a] as Ll);
            uw.push(s.ins.w[a]); uid.push(a);
        }
        let mut wmin: i32 = i32::MAX;
        for &w in &uw { if w < wmin { wmin = w; } }
        let mut wsuf1: Vec<i32> = vec![i32::MAX; cu2 + 1];
        let mut wsuf2: Vec<i32> = vec![i32::MAX; cu2 + 1];
        for x in (0..cu2).rev() {
            let (a1, a2) = (wsuf1[x + 1], wsuf2[x + 1]);
            if uw[x] < a1 {
                wsuf1[x] = uw[x];
                wsuf2[x] = if a1 == i32::MAX { i32::MAX } else { a1 };
            } else {
                wsuf1[x] = a1;
                wsuf2[x] = if a1 != i32::MAX && uw[x] < a2 { uw[x] } else { a2 };
            }
        }
        let mut rmax: Vec<Ll> = vec![Ll::MIN; cu2];
        for i in 0..cu2 {
            let row_i = &q[uid[i] * n..(uid[i] + 1) * n];
            let mut m = Ll::MIN;
            for j in 0..cu2 {
                if j != i {
                    let v = row_i[uid[j]] as Ll;
                    if v > m { m = v; }
                }
            }
            rmax[i] = m;
        }
        let mut g: Vec<Ll> = vec![0; cu2];
        let s23cap = s.ins.s23_cap;
        let mut hsuf: Vec<Ll> = Vec::new();
        if s23cap { hsuf = vec![Ll::MIN; cu2 + 1]; }
        for ri in 0..cs2 {
            let r1 = used[ri].id;
            let row_r1 = &q[r1 * n..(r1 + 1) * n];
            for rj in (ri + 1)..cs2 {
                let r2 = used[rj].id;
                let budget = s.slack() + s.ins.w[r1] + s.ins.w[r2];
                let lost = s.contrib[r1] as Ll + s.contrib[r2] as Ll
                    - row_r1[r2] as Ll;
                if 3 * wmin > budget { continue; }
                let row_r2 = &q[r2 * n..(r2 + 1) * n];
                for i in 0..cu2 {
                    let u = uid[i];
                    g[i] = ubase[i] - row_r1[u] as Ll - row_r2[u] as Ll;
                }
                if s23cap {
                    for i in (0..cu2).rev() {
                        let v = g[i].saturating_add(rmax[i]).saturating_add(rmax[i]);
                        hsuf[i] = if v > hsuf[i + 1] { v } else { hsuf[i + 1] };
                    }
                }
                let bud2 = budget - 2 * wmin;
                let bud1 = budget - wmin;
                {
                    let (mut h1, mut h2, mut h3) = (Ll::MIN, Ll::MIN, Ll::MIN);
                    for i in 0..cu2 {
                        let h = g[i].saturating_add(rmax[i]);
                        if h > h1 { h3 = h2; h2 = h1; h1 = h; }
                        else if h > h2 { h3 = h2; h2 = h; }
                        else if h > h3 { h3 = h; }
                    }
                    if h1.saturating_add(h2).saturating_add(h3).saturating_sub(lost) <= best_delta {
                        continue;
                    }
                }

                for ai in 0..cu2 {
                    let a = uid[ai];
                    let wa = uw[ai];
                    if wa > bud2 { continue; }
                    if wsuf2[ai + 1] == i32::MAX
                        || wa.saturating_add(wsuf1[ai + 1]).saturating_add(wsuf2[ai + 1]) > budget {
                        continue;
                    }
                    let row_a = &q[a * n..(a + 1) * n];
                    let da = g[ai];

                    for bi in (ai + 1)..cu2 {
                        let b = uid[bi];
                        let wab = wa + uw[bi];
                        if wab > bud1 { continue; }
                        if wab.saturating_add(wsuf1[bi + 1]) > budget { continue; }
                        let row_b = &q[b * n..(b + 1) * n];
                        let dab = da + g[bi] + row_a[b] as Ll - lost;
                        let wlim = budget - wab;

                        if s23cap {
                            for ci in (bi + 1)..cu2 {
                                if dab.saturating_add(hsuf[ci]) <= best_delta { break; }
                                let cid = uid[ci];
                                let delta = dab + g[ci]
                                    + row_a[cid] as Ll
                                    + row_b[cid] as Ll;
                                if delta > best_delta && uw[ci] <= wlim {
                                    best_delta = delta;
                                    best_r1 = Some(r1);
                                    best_r2 = Some(r2);
                                    best_a = Some(a);
                                    best_b = Some(b);
                                    best_c = Some(cid);
                                }
                            }
                        } else {
                        for ci in (bi + 1)..cu2 {
                            let cid = uid[ci];
                            let delta = dab + g[ci]
                                + row_a[cid] as Ll
                                + row_b[cid] as Ll;
                            if delta > best_delta && uw[ci] <= wlim {
                                best_delta = delta;
                                best_r1 = Some(r1);
                                best_r2 = Some(r2);
                                best_a = Some(a);
                                best_b = Some(b);
                                best_c = Some(cid);
                            }
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
        let k = window_k.min(s.ins.k31);
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

        let cs31 = used.len();
        let mut uid31: Vec<usize> = Vec::with_capacity(cs31);
        let mut wu31:  Vec<i32>   = Vec::with_capacity(cs31);
        for it in &used { uid31.push(it.id); wu31.push(s.ins.w[it.id]); }
        let mut qt31: Vec<Ll> = vec![0; cs31 * cs31];
        for i in 0..cs31 {
            let row_i = &q[uid31[i] * n..(uid31[i] + 1) * n];
            for j in 0..cs31 { qt31[i * cs31 + j] = row_i[uid31[j]] as Ll; }
        }
        let mut e31: Vec<Ll> = vec![0; cs31];
        let mut s31_rowmax: Vec<Ll> = vec![0; used.len()];
        for i in 0..used.len() {
            let ri_id = used[i].id;
            let row_i = &q[ri_id * n..(ri_id + 1) * n];
            let mut rm = 0i64;
            for j in 0..used.len() {
                if j != i {
                    let v = row_i[used[j].id] as Ll;
                    if v > rm { rm = v; }
                }
            }
            s31_rowmax[i] = rm;
        }
        for ai in 0..unused.len() {
            let add = unused[ai].id;
            let row_add = &q[add * n..(add + 1) * n];
            let base_gain = s.contrib[add] as Ll + row_add[add] as Ll;

            {
                let (mut h1, mut h2, mut h3) = (Ll::MIN, Ll::MIN, Ll::MIN);
                for (ui, u) in used.iter().enumerate() {
                    let h = s31_rowmax[ui] - (row_add[u.id] as Ll + s.contrib[u.id] as Ll);
                    if h > h1 { h3 = h2; h2 = h1; h1 = h; }
                    else if h > h2 { h3 = h2; h2 = h; }
                    else if h > h3 { h3 = h; }
                }
                let s31_ub = base_gain
                    .saturating_add(h1).saturating_add(h2).saturating_add(h3);
                if s31_ub <= best_delta { continue; }
            }
            for i in 0..cs31 {
                e31[i] = row_add[uid31[i]] as Ll + s.contrib[uid31[i]] as Ll;
            }
            let wbase = s.weight + s.ins.w[add] - s.ins.capacity;
            let s31cap = s.ins.s31_cap;
            let mut hs31: Vec<Ll> = Vec::new();
            if s31cap {
                hs31 = vec![Ll::MIN; cs31 + 1];
                for k in (0..cs31).rev() {
                    let v = s31_rowmax[k].saturating_add(s31_rowmax[k]).saturating_sub(e31[k]);
                    hs31[k] = if v > hs31[k + 1] { v } else { hs31[k + 1] };
                }
            }
            for ri in 0..cs31 {
                let qi = &qt31[ri * cs31..(ri + 1) * cs31];
                let bi0 = base_gain - e31[ri];
                let wi0 = wbase - wu31[ri];
                for rj in (ri + 1)..cs31 {
                    let qj = &qt31[rj * cs31..(rj + 1) * cs31];
                    let bij = bi0 - e31[rj] + qi[rj];
                    let wij = wi0 - wu31[rj];
                    if s31cap {
                        for rk in (rj + 1)..cs31 {
                            if bij.saturating_add(hs31[rk]) <= best_delta { break; }
                            let delta = bij - e31[rk] + qi[rk] + qj[rk];
                            if delta > best_delta && wu31[rk] >= wij {
                                best_delta = delta;
                                best_r1 = Some(used[ri].id);
                                best_r2 = Some(used[rj].id);
                                best_r3 = Some(used[rk].id);
                                best_add = Some(add);
                            }
                        }
                    } else {
                    for rk in (rj + 1)..cs31 {
                        let delta = bij - e31[rk] + qi[rk] + qj[rk];
                        if delta > best_delta && wu31[rk] >= wij {
                            best_delta = delta;
                            best_r1 = Some(used[ri].id);
                            best_r2 = Some(used[rj].id);
                            best_r3 = Some(used[rk].id);
                            best_add = Some(add);
                        }
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
                if apply_best_swap11(s, window_k.min(s.ins.k11))     { s.save_best(); continue; }
                if apply_best_swap12(s, (window_k / 2).min(s.ins.k12)) { s.save_best(); continue; }
                if apply_best_swap21(s, (window_k / 2).min(s.ins.k21)) { s.save_best(); continue; }
            } else {
                if apply_best_swap11(s, window_k.min(s.ins.k11))     { s.save_best(); continue; }
                if apply_best_swap12(s, (window_k / 2).min(s.ins.k12)) { s.save_best(); continue; }
                if apply_best_swap21(s, (window_k / 2).min(s.ins.k21)) { s.save_best(); continue; }
            }
            if heavy {
                if apply_best_swap13(s, window_k) { s.save_best(); continue; }
                if s.ins.w_fast & 1 == 0 && s.count >= 6 && apply_best_swap23(s, window_k) {
                    s.save_best();
                    continue;
                }
                if apply_best_swap31(s, window_k) { s.save_best(); continue; }
                #[allow(unused)] const _SWAP22_REMOVED: () = ();
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

        if s.ins.w_fast & 8 != 0 {
            let m = cand.len();
            let mut wsuf: Vec<i32> = vec![i32::MAX; m + 1];
            for t in (0..m).rev() {
                let w = s.ins.w[cand[t].id];
                wsuf[t] = if w < wsuf[t + 1] { w } else { wsuf[t + 1] };
            }
            let capf = s.ins.capacity;
            for (t, is) in cand.iter().enumerate() {
                if s.weight + wsuf[t] > capf { break; }
                if s.weight + s.ins.w[is.id] <= capf {
                    s.add_item(is.id);
                }
            }
            return;
        }
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
                    total_interactions: &[Ll],
                    ils_floor:     usize,
                    ils_stall_min: usize,
                    lean_pre:      usize) {
        let mut dp_ws = DpWorkspace::new();
        let mut vnd_cert = VndCertificate::new();

        dp_refinement(s, core_half, &mut dp_ws);
        local_search_vnd(s, window_k, true, &mut vnd_cert);
        s.save_best();
        s.restore_best();

        let n = s.ins.n;
        if lean_pre < 2 && n <= 2000 && s.count >= 6 {
            let seed_strength = ((s.count as usize) / 8).clamp(2, 5);
            perturb(s, rng, seed_strength, 2, total_interactions);
            greedy_reconstruct(s, 0, total_interactions);
            local_search_vnd(s, window_k, true, &mut vnd_cert);
            s.save_best();
            s.restore_best();
        }

        if lean_pre < 1 && n <= 1200 && s.count >= 10 {
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
            ils_rounds.max(ils_floor).min(140)
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

        let mut stall_limit = (target_rounds / 2).max(24);
        if ils_stall_min < 30 {
            stall_limit = stall_limit.min(ils_stall_min.max(1));
        }
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
                if stall >= stall_limit && round + 1 >= ils_stall_min {
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

    const CB_SC: i64 = 64;
    const CB_INF: i64 = i64::MAX / 8;

    struct CbFlow {
        to: Vec<u32>, cap: Vec<i64>, head: Vec<i32>, nxt: Vec<i32>,
        level: Vec<i32>, it: Vec<i32>, q: Vec<u32>, stk: Vec<u32>, pe: Vec<i32>,
    }

    impl CbFlow {
        fn new() -> Self {
            CbFlow { to: Vec::new(), cap: Vec::new(), head: Vec::new(), nxt: Vec::new(),
                     level: Vec::new(), it: Vec::new(), q: Vec::new(), stk: Vec::new(), pe: Vec::new() }
        }
        fn reset(&mut self, n: usize) {
            self.to.clear(); self.cap.clear(); self.nxt.clear();
            self.head.clear(); self.head.resize(n, -1);
            self.level.clear(); self.level.resize(n, -1);
            self.it.clear(); self.it.resize(n, -1);
            self.pe.clear(); self.pe.resize(n, -1);
        }
        #[inline]
        fn add(&mut self, u: usize, v: usize, c: i64) {
            let e = self.to.len();
            self.to.push(v as u32); self.cap.push(c); self.nxt.push(self.head[u]); self.head[u] = e as i32;
            self.to.push(u as u32); self.cap.push(0); self.nxt.push(self.head[v]); self.head[v] = (e + 1) as i32;
        }
        fn bfs(&mut self, s: usize, t: usize) -> bool {
            for x in self.level.iter_mut() { *x = -1; }
            self.level[s] = 0; self.q.clear(); self.q.push(s as u32);
            let mut qi = 0usize;
            while qi < self.q.len() {
                let u = self.q[qi] as usize; qi += 1;
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
            loop {
                self.stk.clear(); self.stk.push(s as u32);
                let mut found = false;
                while let Some(&u) = self.stk.last() {
                    let u = u as usize;
                    if u == t { found = true; break; }
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
                if !found { return total; }
                let mut f = CB_INF;
                for k in 1..self.stk.len() { let v = self.stk[k] as usize; let c = self.cap[self.pe[v] as usize]; if c < f { f = c; } }
                for k in 1..self.stk.len() { let v = self.stk[k] as usize; let ei = self.pe[v] as usize; self.cap[ei] -= f; self.cap[ei ^ 1] += f; }
                total += f;
            }
        }
        fn maxflow(&mut self, s: usize, t: usize) -> i64 {
            let mut flow = 0i64;
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
                hi_pm: usize, work_cap: usize, lam_iters: usize, eng: usize) -> bool {
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
            ce: CutEng::new(), prep_ok: false, eng,
        };
        let c0 = bb.cap;
        bb.dfs(0, c0, 1, 64, Vec::new());
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
            let hp = Hparams::from_map(hyperparameters, n);

            let raw_seed = u64::from_le_bytes(
                challenge.seed[..8].try_into().unwrap_or([0u8; 8])
            );
            let seed = raw_seed.wrapping_add(0x9E3779B97F4A7C15);
            let mut rng = Rng::new(seed);

            let inst = challenge_to_p1(challenge, hp.w_fast);
            let budget = inst.budgets[0];

            if hp.rush_mode > 0 {
                let (mut p2, total_interactions) = bridge_p1_to_p2(&inst, budget, hp.w_fast);
                p2.s11_cap = (hp.bp_fast & 4) != 0;
                p2.bbu_ranked = (hp.bp_fast & 8) != 0;
                p2.s12_cap = (hp.bp_fast & 16) != 0;
                p2.s21_cap = (hp.bp_fast & 32) != 0;
                p2.s21_blk = (hp.bp_fast & 128) != 0;
                p2.s23_cap = (hp.bp_fast & 256) != 0;
                p2.s12_blk = (hp.bp_fast & 512) != 0;
                p2.s13_cap = (hp.bp_fast & 1024) != 0;
                p2.s31_cap = (hp.bp_fast & 2048) != 0;
                p2.s12_gq  = (hp.bp_fast & 4096) != 0;
                p2.xt_split = (hp.bp_fast & 8192) != 0;
                p2.w_fast = hp.w_fast;
                p2.k11 = hp.k11; p2.k12 = hp.k12; p2.k21 = hp.k21; p2.k13 = hp.k13; p2.k31 = hp.k31;
                let mut s = P2State::new(p2);
                greedy_reconstruct(&mut s, 1, &total_interactions);
                s.save_best();
                if hp.rush_mode >= 2 {
                    let mut dp_ws = DpWorkspace::new();
                    let mut cert  = VndCertificate::new();
                    dp_refinement(&mut s, hp.core_half_dp.min(24), &mut dp_ws);
                    local_search_vnd(&mut s, hp.window_k.min(120), true, &mut cert);
                    s.save_best();
                }
                s.restore_best();
                let chk_wt = P2State::weight_selected(&s.ins, &s.best_sel);
                if chk_wt > budget {
                    return Ok(None);
                }
                let items: Vec<usize> = (0..s.ins.n)
                    .filter(|&i| s.best_sel[i] != 0)
                    .collect();
                return Ok(Some(Solution { items }));
            }

            let (mut p2, total_interactions) = bridge_p1_to_p2(&inst, budget, hp.w_fast);
                p2.s11_cap = (hp.bp_fast & 4) != 0;
                p2.bbu_ranked = (hp.bp_fast & 8) != 0;
                p2.s12_cap = (hp.bp_fast & 16) != 0;
                p2.s21_cap = (hp.bp_fast & 32) != 0;
                p2.s21_blk = (hp.bp_fast & 128) != 0;
                p2.s23_cap = (hp.bp_fast & 256) != 0;
                p2.s12_blk = (hp.bp_fast & 512) != 0;
                p2.s13_cap = (hp.bp_fast & 1024) != 0;
                p2.s31_cap = (hp.bp_fast & 2048) != 0;
                p2.s12_gq  = (hp.bp_fast & 4096) != 0;
                p2.xt_split = (hp.bp_fast & 8192) != 0;
                p2.w_fast = hp.w_fast;
                p2.k11 = hp.k11; p2.k12 = hp.k12; p2.k21 = hp.k21; p2.k13 = hp.k13; p2.k31 = hp.k31;
            let ext_pm = if hp.core_work > 0 && n <= 1200 { hp.core_lo as i32 } else { 0 };
            let qr = run_bp_algorithm(&inst, hp.n_lambda_values, hp.bp_fast, hp.w_fast, &p2.q, ext_pm);
            let mut s = P2State::new(p2);
            bridge_load_solution(&mut s, &qr.results[0]);

            solve_hybrid(&mut s, &mut rng, hp.ils_rounds, hp.window_k, hp.core_half_dp, &total_interactions,
                         hp.ils_floor, hp.ils_stall_min, hp.lean_pre);

            if ext_pm > 0 {
                s.restore_best();
                if core_bnb(&mut s, &qr.bp, qr.bcross, hp.n_lambda_values, hp.core_hi, hp.core_work, hp.core_lam, hp.cut_engine) {
                    let mut dp_ws = DpWorkspace::new();
                    let mut vnd_cert = VndCertificate::new();
                    dp_refinement(&mut s, hp.core_half_dp, &mut dp_ws);
                    local_search_vnd(&mut s, hp.window_k, true, &mut vnd_cert);
                    s.save_best();
                }
                s.restore_best();
            }

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