use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::knapsack::{Challenge, Solution};

mod inner {

    #[inline(always)]
    fn dw(x: i64, w: i64) -> i64 {
        x / w
    }
    use anyhow::Result;
    use serde::{Deserialize, Serialize};
    use serde_json::{Map, Value};
    use tig_challenges::knapsack::*;  

    #[derive(Serialize, Deserialize)]
    pub struct Hyperparameters {
        pub n_lambda_values: Option<usize>,
        pub window_k:        Option<usize>,
        pub ils_rounds:      Option<usize>,
        pub core_half_dp:    Option<usize>,
        pub rush_mode:       Option<usize>,
        pub rush_ils:        Option<usize>,
        pub rush_kick_stall: Option<usize>,
        pub rush_restarts:   Option<usize>,
        pub rush_tabu:       Option<usize>,
        pub ils_stall_stop:  Option<usize>,
        pub xr_elite:        Option<usize>,
        pub dp_passes:       Option<usize>,
        pub hub_top:         Option<usize>,
        pub neutral_mask:    Option<usize>,
        pub bp_seed:         Option<usize>,
        pub dedup_fast:      Option<usize>,
        pub use_hub_pair:    Option<usize>,
        pub use_heavy_polish:Option<usize>,
        pub stage_stop:      Option<usize>,
        pub exact_mask:      Option<usize>,
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
        rush_mode:            usize,
        rush_ils:             usize,
        rush_kick_stall:      usize,
        rush_restarts:        usize,
        rush_tabu:            usize,
        ils_stall_stop:       usize,
        xr_elite:             usize,
        dp_passes:            usize,
        bp_seed:              usize,
        hub_top:              usize,
        neutral_mask:         usize,
        dedup_fast:           usize,
        stage_stop:           usize,
        exact_mask:           usize,
        use_hub_pair:         bool,
        use_heavy_polish:     bool,
        window_k:             usize,
        core_half_dp:         usize,
        core_lo: usize, core_hi: usize, core_band: usize, core_work: usize, core_lam: usize, core_warm: usize, core_k: usize, core_rc: usize, core_up: usize, core_root: usize, core_tot: usize,
        cut_engine: usize,
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
                        n_full_restarts: 1, rush_mode: 0, rush_ils: 0, rush_kick_stall: 12, rush_restarts: 1, rush_tabu: 0, ils_stall_stop: 0, xr_elite: 0, dp_passes: 4, bp_seed: 1, hub_top: 0, neutral_mask: 0, dedup_fast: 0, stage_stop: 0, exact_mask: 0, use_hub_pair: true,
                        use_heavy_polish: true, window_k: n, core_half_dp: 60,
                        core_lo: 0, core_hi: 0, core_band: 0, core_work: 0, core_lam: 2, core_warm: 1, core_k: 0, core_rc: 0, core_up: 0, core_root: 0, core_tot: 0, cut_engine: 2,
                    }
                } else if budget <= 10 {
                    Self {
                        n_lambda_values: 1600,
                        n_random_starts: 4, n_crossover_gen: 12, sa_rounds: 30,
                        sa_iter: 300, n_sa_members: 3, ils_rounds: 300,
                        ils_restart_interval: 12, perturb_base_frac: 4,
                        perturb_max_frac: 5, ils_vnd_level: 0, bounded_2_2_k: 10,
                        n_full_restarts: 1, rush_mode: 0, rush_ils: 0, rush_kick_stall: 12, rush_restarts: 1, rush_tabu: 0, ils_stall_stop: 0, xr_elite: 0, dp_passes: 4, bp_seed: 1, hub_top: 0, neutral_mask: 0, dedup_fast: 0, stage_stop: 0, exact_mask: 0, use_hub_pair: true,
                        use_heavy_polish: true, window_k: n, core_half_dp: 60,
                        core_lo: 0, core_hi: 0, core_band: 0, core_work: 0, core_lam: 2, core_warm: 1, core_k: 0, core_rc: 0, core_up: 0, core_root: 0, core_tot: 0, cut_engine: 2,
                    }
                } else {
                    Self {
                        n_lambda_values: 800,
                        n_random_starts: 3, n_crossover_gen: 26, sa_rounds: 0,
                        sa_iter: 0, n_sa_members: 0, ils_rounds: 24,
                        ils_restart_interval: 1, perturb_base_frac: 100,
                        perturb_max_frac: 28, ils_vnd_level: 0, bounded_2_2_k: 0,
                        n_full_restarts: 6, rush_mode: 0, rush_ils: 0, rush_kick_stall: 12, rush_restarts: 1, rush_tabu: 0, ils_stall_stop: 3, xr_elite: 6, dp_passes: 2, bp_seed: 1, hub_top: 64, neutral_mask: 1480556543, dedup_fast: 1, stage_stop: 0, exact_mask: 4346431410023, use_hub_pair: true,
                        use_heavy_polish: false, window_k: 60, core_half_dp: 12,
                        core_lo: 0, core_hi: 0, core_band: 5, core_work: 750000, core_lam: 2, core_warm: 1, core_k: 0, core_rc: 0, core_up: 1, core_root: 3, core_tot: 1, cut_engine: 2,
                    }
                }
            } else {
                Self {
                    n_lambda_values: 1600,
                    n_random_starts: 4, n_crossover_gen: 0, sa_rounds: 0,
                    sa_iter: 0, n_sa_members: 0, ils_rounds: 120,
                    ils_restart_interval: 12, perturb_base_frac: 8,
                    perturb_max_frac: 5, ils_vnd_level: 0, bounded_2_2_k: 0,
                    n_full_restarts: 1, rush_mode: 0, rush_ils: 0, rush_kick_stall: 12, rush_restarts: 1, rush_tabu: 0, ils_stall_stop: 0, xr_elite: 0, dp_passes: 4, bp_seed: 1, hub_top: 0, neutral_mask: 0, dedup_fast: 0, stage_stop: 0, exact_mask: 0, use_hub_pair: false,
                    use_heavy_polish: false, window_k: 200, core_half_dp: 50,
                        core_lo: 0, core_hi: 0, core_band: 0, core_work: 0, core_lam: 2, core_warm: 1, core_k: 0, core_rc: 0, core_up: 0, core_root: 0, core_tot: 0, cut_engine: 2,
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
                if let Some(v) = m.get("rush_mode").and_then(|v| v.as_u64())        { p.rush_mode            = v as usize; }
                if let Some(v) = m.get("rush_ils").and_then(|v| v.as_u64())         { p.rush_ils             = v as usize; }
                if let Some(v) = m.get("rush_kick_stall").and_then(|v| v.as_u64())  { p.rush_kick_stall      = v as usize; }
                if let Some(v) = m.get("rush_restarts").and_then(|v| v.as_u64())    { p.rush_restarts        = v as usize; }
                if let Some(v) = m.get("rush_tabu").and_then(|v| v.as_u64())        { p.rush_tabu            = v as usize; }
                if let Some(v) = m.get("ils_stall_stop").and_then(|v| v.as_u64())   { p.ils_stall_stop       = v as usize; }
                if let Some(v) = m.get("xr_elite").and_then(|v| v.as_u64())         { p.xr_elite             = v as usize; }
                if let Some(v) = m.get("core_lo").and_then(|v| v.as_u64())   { p.core_lo   = v as usize; }
                if let Some(v) = m.get("core_hi").and_then(|v| v.as_u64())   { p.core_hi   = v as usize; }
                if let Some(v) = m.get("core_band").and_then(|v| v.as_u64()) { p.core_band = v as usize; }
                if let Some(v) = m.get("core_work").and_then(|v| v.as_u64()) { p.core_work = v as usize; }
                if let Some(v) = m.get("core_lam").and_then(|v| v.as_u64())  { p.core_lam  = v as usize; }
                if let Some(v) = m.get("core_warm").and_then(|v| v.as_u64()) { p.core_warm = v as usize; }
                if let Some(v) = m.get("core_k").and_then(|v| v.as_u64())    { p.core_k    = v as usize; }
                if let Some(v) = m.get("core_rc").and_then(|v| v.as_u64())   { p.core_rc   = v as usize; }
                if let Some(v) = m.get("core_up").and_then(|v| v.as_u64())   { p.core_up   = v as usize; }
                if let Some(v) = m.get("core_root").and_then(|v| v.as_u64()) { p.core_root = v as usize; }
                if let Some(v) = m.get("core_tot").and_then(|v| v.as_u64())  { p.core_tot  = v as usize; }
                if let Some(v) = m.get("cut_engine").and_then(|v| v.as_u64()) { p.cut_engine = v as usize; }
                if let Some(v) = m.get("window_k").and_then(|v| v.as_u64())         { p.window_k             = v as usize; }
                if let Some(v) = m.get("core_half_dp").and_then(|v| v.as_u64())     { p.core_half_dp         = v as usize; }
                if let Some(v) = m.get("dp_passes").and_then(|v| v.as_u64())        { p.dp_passes            = v as usize; }
                if let Some(v) = m.get("bp_seed").and_then(|v| v.as_u64())          { p.bp_seed              = v as usize; }
                if let Some(v) = m.get("hub_top").and_then(|v| v.as_u64())          { p.hub_top              = v as usize; }
                if let Some(v) = m.get("neutral_mask").and_then(|v| v.as_u64())     { p.neutral_mask         = v as usize; }
                if let Some(v) = m.get("dedup_fast").and_then(|v| v.as_u64())       { p.dedup_fast           = v as usize; }
                if let Some(v) = m.get("exact_mask").and_then(|v| v.as_u64())       { p.exact_mask           = v as usize; }
                if let Some(v) = m.get("stage_stop").and_then(|v| v.as_u64())       { p.stage_stop           = v as usize; }
                if let Some(v) = m.get("use_hub_pair").and_then(|v| v.as_u64())     { p.use_hub_pair         = v != 0; }
                if let Some(v) = m.get("use_heavy_polish").and_then(|v| v.as_u64()) { p.use_heavy_polish     = v != 0; }
            }
            p
        }
    }

    #[derive(Default)]
    pub struct DetHasher(u64);
    impl std::hash::Hasher for DetHasher {
        #[inline] fn finish(&self) -> u64 { self.0 }
        #[inline] fn write(&mut self, bytes: &[u8]) {
            for &b in bytes { self.0 = (self.0 ^ b as u64).wrapping_mul(0x100000001b3); }
        }
        #[inline] fn write_u64(&mut self, n: u64) {
            self.0 = n.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }
    pub type DetMap = std::collections::HashMap<u64, usize, std::hash::BuildHasherDefault<DetHasher>>;

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
        pub selected_items: Vec<usize>,
    }

    #[derive(Debug)]
    pub struct QKPResult { pub results: Vec<BudgetResult>, pub bp: Vec<i32>, pub bcross: i32, pub lam_c: (f64, f64) }

    pub fn challenge_to_p1(challenge: &Challenge) -> P1Instance { challenge_to_p1_m(challenge, 0) }
    pub fn challenge_to_p1_m(challenge: &Challenge, nm: usize) -> P1Instance {
        let n = challenge.num_items;
        let mut edges: Vec<Edge> = Vec::with_capacity(if n <= 1200 { n + 160_000 } else { n });
        for i in 0..n {
            edges.push(Edge { i, j: i, value: challenge.values[i] as f64 });
        }
        if nm & (1usize << 30) != 0 {
            let mut sel: Vec<usize> = vec![0usize; n + 4];
            for i in 0..n {
                let row = &challenge.interaction_values[i];
                let mut cnt = 0usize;
                unsafe {
                    let rp = row.as_ptr(); let sq = sel.as_mut_ptr();
                    let mut j = i + 1;
                    let m = j + ((n - j) & !3usize);
                    while j < m {
                        sq.add(cnt).write(j);     cnt += (*rp.add(j)     != 0) as usize;
                        sq.add(cnt).write(j + 1); cnt += (*rp.add(j + 1) != 0) as usize;
                        sq.add(cnt).write(j + 2); cnt += (*rp.add(j + 2) != 0) as usize;
                        sq.add(cnt).write(j + 3); cnt += (*rp.add(j + 3) != 0) as usize;
                        j += 4;
                    }
                    while j < n { sq.add(cnt).write(j); cnt += (*rp.add(j) != 0) as usize; j += 1; }
                }
                for t in 0..cnt {
                    let j = sel[t];
                    edges.push(Edge { i, j, value: row[j] as f64 });
                }
            }
        } else {
        for i in 0..n {
            for j in (i + 1)..n {
                let v = challenge.interaction_values[i][j];
                if v != 0 { edges.push(Edge { i, j, value: v as f64 }); }
            }
        }
        }
        let weights: Vec<i32> = challenge.weights.iter().map(|&w| w as i32).collect();
        let budgets: Vec<i32> = vec![challenge.max_weight as i32];
        P1Instance { n_items: n, n_edges: edges.len(), edges, weights, n_budgets: 1, budgets }
    }

    pub struct UtilMatrix<'a> { pub n: usize, pub data: Vec<f64>, pub linear: Vec<f64>, pub view: Option<&'a Challenge> }
    impl<'a> UtilMatrix<'a> {
        #[inline] pub fn get(&self, r: usize, c: usize) -> f64 {
            match self.view {
                Some(ch) => {
                    if r == c { return 0.0; }
                    let (a, b) = if r < c { (r, c) } else { (c, r) };
                    unsafe { *ch.interaction_values.get_unchecked(a).get_unchecked(b) as f64 }
                }
                None => self.data[r * self.n + c],
            }
        }
    }

    pub fn build_utility_matrix<'a>(n: usize, edges: &[Edge]) -> UtilMatrix<'a> {
        let mut data   = vec![0.0f64; n * n];
        let mut linear = vec![0.0f64; n];
        for e in edges {
            if e.i == e.j { linear[e.i] += e.value; }
            else { data[e.i * n + e.j] = e.value; data[e.j * n + e.i] = e.value; }
        }
        UtilMatrix { n, data, linear, view: None }
    }

    pub fn build_utility_view<'a>(n: usize, ch: &'a Challenge) -> UtilMatrix<'a> {
        let mut linear = vec![0.0f64; n];
        for i in 0..n { linear[i] = ch.values[i] as f64; }
        UtilMatrix { n, data: Vec::new(), linear, view: Some(ch) }
    }

    pub fn compute_ofv(sel: &[usize], n: usize, edges: &[Edge]) -> f64 { compute_ofv_m(sel, n, edges, 0) }
    pub fn compute_ofv2_m(a: &[usize], b: &[usize], n: usize, edges: &[Edge], nm: usize) -> (f64, f64) {
        let mut sa = vec![false; n]; for &i in a { sa[i] = true; }
        let mut sb = vec![false; n]; for &i in b { sb[i] = true; }
        let mut oa = 0.0f64; let mut ob = 0.0f64;
        if nm & (1usize << 23) != 0 {
            unsafe {
                let pa = sa.as_ptr(); let pb = sb.as_ptr();
                for e in edges {
                    let bits = e.value.to_bits();
                    let ka = (*pa.add(e.i) & *pa.add(e.j)) as u64;
                    let kb = (*pb.add(e.i) & *pb.add(e.j)) as u64;
                    oa += f64::from_bits(bits & 0u64.wrapping_sub(ka));
                    ob += f64::from_bits(bits & 0u64.wrapping_sub(kb));
                }
            }
            return (oa, ob);
        }
        for e in edges {
            if sa[e.i] && sa[e.j] { oa += e.value; }
            if sb[e.i] && sb[e.j] { ob += e.value; }
        }
        (oa, ob)
    }
    pub fn compute_ofv_m(sel: &[usize], n: usize, edges: &[Edge], nm: usize) -> f64 {
        let mut selected = vec![false; n];
        for &i in sel { selected[i] = true; }
        let mut ofv = 0.0f64;
        if nm & (1usize << 23) != 0 {
            unsafe {
                let sp = selected.as_ptr();
                for e in edges {
                    let keep = (*sp.add(e.i) & *sp.add(e.j)) as u64;
                    let mask = 0u64.wrapping_sub(keep);
                    ofv += f64::from_bits(e.value.to_bits() & mask);
                }
            }
            return ofv;
        }
        for e in edges {
            if selected[e.i] && selected[e.j] { ofv += e.value; }
        }
        ofv
    }

    mod hpf {
        use super::*;

        pub struct HpfArc {
            pub from: *mut HpfNode, pub to: *mut HpfNode,
            pub flow: f32, pub capacity: f32,
            pub direction: i32,
        }

        pub struct HpfNode {
            pub wt: f32, pub cst: f32,
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
                    wt: 0.0, cst: 1.0, num_adjacent: 0,
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
            pub max_bucket: i32, pub lean: usize,
            pub max_ratio: f32, pub step: f32,
            pub k_src: Vec<i32>, pub k_snk: Vec<i32>,
            pub awake_src: Vec<u32>, pub awake_snk: Vec<u32>,
            pub src_by_k: Vec<u32>, pub src_ptr: usize,
            pub wtv: Vec<f32>, pub cstv: Vec<f32>, pub csrc: Vec<f32>, pub csnk: Vec<f32>,
            pub snk_by_k: Vec<u32>, pub snk_ptr: usize,
            pub lifted_w: i64, pub budget_w: i64, pub nodew: Vec<i64>,
            pub ext_pm: i32, pub stop_at: i32,
            pub stop_on: bool, pub relift: u64,
            pub num_nodes: i32,
            pub source: i32, pub sink: i32, pub num_params: i32,
            pub highest_strong_label: i32,
            pub adjacency_list: Vec<HpfNode>,
            pub strong_roots: Vec<HpfRoot>,
            pub label_count: Vec<i32>,
            pub arc_list: Vec<HpfArc>,
            pub cap_arena: Vec<f32>,
            pub root_slab: Vec<HpfNode>,
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

        #[inline(always)]
        unsafe fn note_lift(s: *mut HpfState, nd: *mut HpfNode) {
            if (*nd).breakpoint > (*s).num_params {
                let slot = ((*nd).number - 1) as usize;
                (*s).lifted_w += *(*s).nodew.get_unchecked(slot);
            } else { (*s).relift += 1; }
        }
        unsafe fn lift_all(s: *mut HpfState, root_node: *mut HpfNode, theparam: i32) {
            let mut current = root_node;
            (*current).next_scan = (*current).child_list;
            (*s).label_count[(*current).label as usize] -= 1;
            (*current).label = (*s).num_nodes;
            if (*s).stop_on { note_lift(s, current); }
            (*current).breakpoint = theparam + 1;
            loop {
                while !(*current).next_scan.is_null() {
                    let temp = (*current).next_scan;
                    (*current).next_scan = (*(*current).next_scan).next;
                    current = temp;
                    (*current).next_scan = (*current).child_list;
                    (*s).label_count[(*current).label as usize] -= 1;
                    (*current).label = (*s).num_nodes;
                    if (*s).stop_on { note_lift(s, current); }
                    (*current).breakpoint = theparam + 1;
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
            if lbl as i32 > (*s).max_bucket { (*s).max_bucket = lbl as i32; }
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
            if lbl as i32 > (*s).max_bucket { (*s).max_bucket = lbl as i32; }
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
                if lbl as i32 > (*s).max_bucket { (*s).max_bucket = lbl as i32; }
                add_to_strong_bucket(current, (*s).strong_roots[lbl].end);
            }
        }

        unsafe fn find_weak_node(s: *mut HpfState, strong_node: *mut HpfNode,
                                  weak_node: *mut *mut HpfNode) -> *mut HpfArc {
            let target = (*s).highest_strong_label - 1;
            let size = (*strong_node).num_out_of_tree as usize;
            let mut i = (*strong_node).next_arc as usize;
            if (*s).lean & (1usize << 24) != 0 {
                let base = (*strong_node).out_of_tree.as_ptr();
                while i < size {
                    let out = *base.add(i);
                    let to = (*out).to; let fr = (*out).from;
                    let ht = ((*to).label == target) as u32;
                    let hf = ((*fr).label == target) as u32;
                    if ht | hf != 0 {
                        (*strong_node).next_arc = i as i32;
                        *weak_node = if ht != 0 { to } else { fr };
                        let last = (*strong_node).num_out_of_tree as usize - 1;
                        (*strong_node).out_of_tree[i] = (*strong_node).out_of_tree[last];
                        (*strong_node).out_of_tree.pop(); (*strong_node).num_out_of_tree -= 1;
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

        unsafe fn process_root_diag(s: *mut HpfState, theparam: i32) {
            let d = ((theparam as i64 * 10) / (*s).num_params.max(1) as i64) as usize;
            DIAG[d.min(9)] += 1; DIAG[10] += 1;
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
            if lbl as i32 > (*s).max_bucket { (*s).max_bucket = lbl as i32; }
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
                if lbl as i32 > (*s).max_bucket { (*s).max_bucket = lbl as i32; }
                add_to_strong_bucket(sr, (*s).strong_roots[lbl].end);
            }
            (*s).highest_strong_label = 1;
            let sr = (*(*s).strong_roots[1].start).next;
            (*(*sr).next).prev = (*sr).prev; (*(*sr).prev).next = (*sr).next;
            (*sr).next = std::ptr::null_mut(); sr
        }

        unsafe fn update_capacities(s: *mut HpfState, theparam: i32) {
            let n_items = ((*s).num_nodes - 2) as usize;
            if n_items == 0 { return; }
            let first_parametric = (*s).arc_list.len() - 2 * n_items;
            if (*s).lean & 36 == 36 && (*s).lean & (1usize << 29) != 0 {
                let p = (*s).max_ratio - theparam as f32 * (*s).step;
                let nn = (*s).num_nodes;
                while (*s).src_ptr < n_items {
                    let it = *(*s).src_by_k.get_unchecked((*s).src_ptr);
                    if *(*s).k_src.get_unchecked(it as usize) > theparam { break; }
                    let pos = (*s).awake_src.partition_point(|&x| x < it);
                    (*s).awake_src.insert(pos, it);
                    (*s).src_ptr += 1;
                }
                while (*s).snk_ptr < n_items {
                    let it = *(*s).snk_by_k.get_unchecked((*s).snk_ptr);
                    if *(*s).k_snk.get_unchecked(it as usize) >= theparam { break; }
                    let pos = (*s).awake_snk.partition_point(|&x| x < it);
                    (*s).awake_snk.remove(pos);
                    (*s).snk_ptr += 1;
                }
                let adj = (*s).adjacency_list.as_mut_ptr();
                let wtv = (*s).wtv.as_ptr(); let cstv = (*s).cstv.as_ptr();
                let csrc = (*s).csrc.as_mut_ptr(); let csnk = (*s).csnk.as_mut_ptr();
                let sp = (*s).awake_src.as_ptr(); let sl = (*s).awake_src.len();
                for t in 0..sl {
                    let i = *sp.add(t) as usize;
                    let nd = adj.add(i + 2);
                    let b = *wtv.add(i) - p * *cstv.add(i);
                    let d = b - *csrc.add(i);
                    *csrc.add(i) = b;
                    (*nd).excess += d;
                    if (*nd).label < nn && (*nd).excess > 0.0 { push_excess(s, nd); }
                }
                let kp = (*s).awake_snk.as_ptr(); let kl = (*s).awake_snk.len();
                macro_rules! snk1 { ($t:expr) => {{
                    let i = *kp.add($t) as usize;
                    let nd = adj.add(i + 2);
                    let raw = p * *cstv.add(i) - *wtv.add(i);
                    let b = if raw > 0.0 { raw } else { 0.0 };
                    let d = b - *csnk.add(i);
                    *csnk.add(i) = b;
                    (*nd).excess -= d;
                    if (*nd).excess > 0.0 && (*nd).label < nn { push_excess(s, nd); }
                }}; }
                let mut t = 0usize;
                if (*s).lean & (1usize << 32) != 0 {
                    let m = kl & !3usize;
                    while t < m {
                        let i0 = *kp.add(t) as usize;     let i1 = *kp.add(t + 1) as usize;
                        let i2 = *kp.add(t + 2) as usize; let i3 = *kp.add(t + 3) as usize;
                        let n0 = adj.add(i0 + 2); let n1 = adj.add(i1 + 2);
                        let n2 = adj.add(i2 + 2); let n3 = adj.add(i3 + 2);
                        let r0 = p * *cstv.add(i0) - *wtv.add(i0);
                        let r1 = p * *cstv.add(i1) - *wtv.add(i1);
                        let r2 = p * *cstv.add(i2) - *wtv.add(i2);
                        let r3 = p * *cstv.add(i3) - *wtv.add(i3);
                        let b0 = if r0 > 0.0 { r0 } else { 0.0 };
                        let b1 = if r1 > 0.0 { r1 } else { 0.0 };
                        let b2 = if r2 > 0.0 { r2 } else { 0.0 };
                        let b3 = if r3 > 0.0 { r3 } else { 0.0 };
                        let e0 = (*n0).excess - (b0 - *csnk.add(i0));
                        let e1 = (*n1).excess - (b1 - *csnk.add(i1));
                        let e2 = (*n2).excess - (b2 - *csnk.add(i2));
                        let e3 = (*n3).excess - (b3 - *csnk.add(i3));
                        let hit = ((e0 > 0.0) & ((*n0).label < nn)) as u32
                                | ((e1 > 0.0) & ((*n1).label < nn)) as u32
                                | ((e2 > 0.0) & ((*n2).label < nn)) as u32
                                | ((e3 > 0.0) & ((*n3).label < nn)) as u32;
                        if hit == 0 {
                            *csnk.add(i0) = b0; (*n0).excess = e0;
                            *csnk.add(i1) = b1; (*n1).excess = e1;
                            *csnk.add(i2) = b2; (*n2).excess = e2;
                            *csnk.add(i3) = b3; (*n3).excess = e3;
                        } else {
                            snk1!(t); snk1!(t + 1); snk1!(t + 2); snk1!(t + 3);
                        }
                        t += 4;
                    }
                }
                while t < kl { snk1!(t); t += 1; }
                (*s).highest_strong_label = if (*s).lean & 1 != 0 && (*s).max_bucket < (*s).num_nodes - 1 { (*s).max_bucket } else { (*s).num_nodes - 1 };
                return;
            }
            if (*s).lean & 36 == 36 {
                let p = (*s).max_ratio - theparam as f32 * (*s).step;
                let nn = (*s).num_nodes;
                while (*s).src_ptr < n_items {
                    let it = *(*s).src_by_k.get_unchecked((*s).src_ptr);
                    if *(*s).k_src.get_unchecked(it as usize) > theparam { break; }
                    let pos = (*s).awake_src.partition_point(|&x| x < it);
                    (*s).awake_src.insert(pos, it);
                    (*s).src_ptr += 1;
                }
                let sp = (*s).awake_src.as_ptr(); let sl = (*s).awake_src.len();
                for t in 0..sl {
                    let i = *sp.add(t) as usize;
                    let nd = (*s).adjacency_list.as_mut_ptr().add(i + 2);
                    let arc = (*s).arc_list.as_mut_ptr().add(first_parametric + 2 * i);
                    let delta = ((*nd).wt - p * (*nd).cst) - (*arc).capacity;
                    if delta < 0.0 { return; }
                    (*arc).capacity += delta; (*arc).flow += delta; (*nd).excess += delta;
                    if (*nd).label < nn && (*nd).excess > 0.0 { push_excess(s, nd); }
                }
                let kp = (*s).awake_snk.as_mut_ptr(); let kl = (*s).awake_snk.len();
                let mut wcur = 0usize;
                for t in 0..kl {
                    let i = *kp.add(t) as usize;
                    let live = theparam <= *(*s).k_snk.get_unchecked(i);
                    let nd = (*s).adjacency_list.as_mut_ptr().add(i + 2);
                    let arc = (*s).arc_list.as_mut_ptr().add(first_parametric + 2 * i + 1);
                    if live || (*arc).capacity != 0.0 {
                        let raw = p * (*nd).cst - (*nd).wt;
                        let delta = (if raw > 0.0 { raw } else { 0.0 }) - (*arc).capacity;
                        if delta > 0.0 { (*s).awake_snk.set_len(kl); return; }
                        (*arc).capacity += delta; (*arc).flow += delta; (*nd).excess -= delta;
                        if (*nd).label < nn && (*nd).excess > 0.0 { push_excess(s, nd); }
                    }
                    if live { *kp.add(wcur) = i as u32; wcur += 1; }
                }
                (*s).awake_snk.set_len(wcur);
                (*s).highest_strong_label = if (*s).lean & 1 != 0 && (*s).max_bucket < (*s).num_nodes - 1 { (*s).max_bucket } else { (*s).num_nodes - 1 };
                return;
            }
            let arena = (*s).cap_arena.as_ptr();
            let source_base = theparam as usize * 2 * n_items;
            for i in 0..n_items {
                let arc = &mut (*s).arc_list[first_parametric + 2 * i] as *mut HpfArc;
                let delta = *arena.add(source_base + i) - (*arc).capacity;
                if delta < 0.0 { return; }
                (*arc).capacity += delta; (*arc).flow += delta; (*(*arc).to).excess += delta;
                if (*(*arc).to).label < (*s).num_nodes && (*(*arc).to).excess > 0.0 { push_excess(s, (*arc).to); }
            }
            let sink_base = source_base + n_items;
            for i in 0..n_items {
                let arc = &mut (*s).arc_list[first_parametric + 2 * i + 1] as *mut HpfArc;
                let delta = *arena.add(sink_base + i) - (*arc).capacity;
                if delta > 0.0 { return; }
                (*arc).capacity += delta; (*arc).flow += delta; (*(*arc).from).excess -= delta;
                if (*(*arc).from).label < (*s).num_nodes && (*(*arc).from).excess > 0.0 { push_excess(s, (*arc).from); }
            }
            (*s).highest_strong_label = if (*s).lean & 1 != 0 && (*s).max_bucket < (*s).num_nodes - 1 { (*s).max_bucket } else { (*s).num_nodes - 1 };
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
                    if 1 > (*s).max_bucket { (*s).max_bucket = 1; }
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
            let diag = (*s).lean & (1usize << 35) != 0;
            loop { let sr = get_highest_strong_root(s, theparam); if sr.is_null() { break; } if diag { process_root_diag(s, theparam); } process_root(s, sr); }
            if (*s).stop_on && (*s).lifted_w >= (*s).budget_w && past_stop(s, 0) { return; }
            theparam = 1;
            while theparam < (*s).num_params {
                update_capacities(s, theparam);
                loop { let sr = get_highest_strong_root(s, theparam); if sr.is_null() { break; } if diag { process_root_diag(s, theparam); } process_root(s, sr); }
                if (*s).stop_on && (*s).lifted_w >= (*s).budget_w && past_stop(s, theparam) { break; }
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

        pub static mut DIAG: [u64; 12] = [0; 12];
        pub struct BreakpointSets { pub sets: Vec<(i32, Vec<usize>)>, pub maxr: f32, pub step: f32 }

        pub fn get_breakpoints(inst: &P1Instance, n_lambda_values: usize, lean: usize) -> BreakpointSets {
            get_breakpoints_x(inst, n_lambda_values, lean, 0)
        }
        pub fn get_breakpoints_x(inst: &P1Instance, n_lambda_values: usize, lean: usize, ext_pm: i32) -> BreakpointSets {
            let n_items    = inst.n_items;
            let n_edges    = inst.n_edges;
            let num_nodes  = (n_items + 2) as i32;
            let num_arcs   = (n_edges + 2 * n_items) as i32;
            let num_params = n_lambda_values as i32;

            let mut adjacency_list: Vec<HpfNode> = (0..num_nodes as usize)
                .map(|i| { let mut nd = HpfNode::zeroed(num_params); nd.number = (i + 1) as i32; nd.cst = 1.0; nd })
                .collect();
            for i in 0..n_items { adjacency_list[i + 2].cst = inst.weights[i] as f32; }

            let one_pass = lean & (1usize << 31) != 0;
            let dead_adj = lean & (1usize << 34) != 0;
            let mut arc_list: Vec<HpfArc> = Vec::with_capacity(num_arcs as usize);
            let mut outdeg: Vec<u32> = if lean & (1usize << 25) != 0 { vec![0u32; num_nodes as usize] } else { Vec::new() };
            let mut first = 0usize;
            if one_pass {
                let adjp = adjacency_list.as_mut_ptr();
                for k in 0..n_edges {
                    let from_id = inst.edges[k].i; let to_id = inst.edges[k].j;
                    let cap = inst.edges[k].value as f32;
                    unsafe {
                        arc_list.push(HpfArc { from: adjp.add(from_id + 2), to: adjp.add(to_id + 2),
                            flow: 0.0, capacity: cap, direction: 1 });
                        (*adjp.add(from_id + 2)).wt += cap;
                        if !dead_adj { (*adjp.add(from_id + 2)).num_adjacent += 1;
                                       (*adjp.add(to_id + 2)).num_adjacent += 1; }
                    }
                    if !outdeg.is_empty() { outdeg[from_id + 2] += 1; }
                    first += 1;
                }
            } else {
            for _ in 0..num_arcs as usize {
                arc_list.push(HpfArc { from: std::ptr::null_mut(), to: std::ptr::null_mut(),
                    flow: 0.0, capacity: 0.0, direction: 1 });
            }
            for k in 0..n_edges {
                let from_id = inst.edges[k].i; let to_id = inst.edges[k].j;
                let cap = inst.edges[k].value as f32;
                arc_list[first].capacity = cap;
                arc_list[first].direction = 1;
                adjacency_list[from_id + 2].wt += cap; adjacency_list[from_id + 2].num_adjacent += 1;
                adjacency_list[to_id + 2].num_adjacent += 1; first += 1;
                if !outdeg.is_empty() { outdeg[from_id + 2] += 1; }
            }
            }

            let mut max_degree_ratio = 0.0f32;
            for i in 2..num_nodes as usize {
                let ratio = adjacency_list[i].wt / adjacency_list[i].cst;
                if ratio > max_degree_ratio { max_degree_ratio = ratio; }
            }
            let step: f32 = if num_params > 0 { max_degree_ratio / num_params as f32 } else { 0.0 };
            let param_count = num_params as usize;
            let lean215 = lean & 36 == 36;
            let mut k_src: Vec<i32> = Vec::new();
            let mut k_snk: Vec<i32> = Vec::new();
            if lean215 {
                k_src = vec![num_params; n_items];
                k_snk = vec![0i32; n_items];
                for item in 0..n_items {
                    let i = item + 2;
                    let wt = adjacency_list[i].wt; let cst = adjacency_list[i].cst;
                    let (mut lo, mut hi) = (0i32, num_params);
                    while lo < hi {
                        let mid = lo + (hi - lo) / 2;
                        let p = max_degree_ratio - mid as f32 * step;
                        if wt - p * cst > 0.0 { hi = mid; } else { lo = mid + 1; }
                    }
                    k_src[item] = lo;
                    let (mut lo, mut hi) = (0i32, num_params);
                    while lo < hi {
                        let mid = lo + (hi - lo) / 2;
                        let p = max_degree_ratio - mid as f32 * step;
                        if p * cst - wt > 0.0 { lo = mid + 1; } else { hi = mid; }
                    }
                    k_snk[item] = lo;
                }
            }
            let mut capacity_arena = if lean215 { Vec::new() } else { vec![0.0f32; param_count * 2 * n_items] };
            if !lean215 {
            for param in 0..param_count {
                let p = max_degree_ratio - param as f32 * step;
                let source_base = param * 2 * n_items;
                let sink_base = source_base + n_items;
                for item in 0..n_items {
                    let i = item + 2;
                    let wt = adjacency_list[i].wt; let cst = adjacency_list[i].cst;
                    let src = wt - p * cst;
                    capacity_arena[source_base + item] = if src > 0.0 { src } else { 0.0 };
                    let snk = p * cst - wt;
                    capacity_arena[sink_base + item] = if snk > 0.0 { snk } else { 0.0 };
                }
            }
            }

            let p0 = max_degree_ratio;
            for item in 0..n_items {
                let i = item + 2;
                let (c0, c1) = if lean215 {
                    let wt = adjacency_list[i].wt; let cst = adjacency_list[i].cst;
                    let src = wt - p0 * cst; let snk = p0 * cst - wt;
                    (if src > 0.0 { src } else { 0.0 }, if snk > 0.0 { snk } else { 0.0 })
                } else { (capacity_arena[item], capacity_arena[n_items + item]) };
                if one_pass {
                    let adjp = adjacency_list.as_mut_ptr();
                    unsafe {
                        arc_list.push(HpfArc { from: adjp, to: adjp.add(i),
                            flow: 0.0, capacity: if param_count > 0 { c0 } else { 0.0 }, direction: 1 });
                        arc_list.push(HpfArc { from: adjp.add(i), to: adjp.add(1),
                            flow: 0.0, capacity: if param_count > 0 { c1 } else { 0.0 }, direction: 1 });
                    }
                    if !dead_adj { adjacency_list[0].num_adjacent += 1; adjacency_list[i].num_adjacent += 1;
                                   adjacency_list[i].num_adjacent += 1; adjacency_list[1].num_adjacent += 1; }
                    first += 2;
                } else {
                arc_list[first].capacity = if param_count > 0 { c0 } else { 0.0 };
                arc_list[first].direction = 1;
                adjacency_list[0].num_adjacent += 1; adjacency_list[i].num_adjacent += 1; first += 1;
                arc_list[first].capacity = if param_count > 0 { c1 } else { 0.0 };
                arc_list[first].direction = 1;
                adjacency_list[i].num_adjacent += 1; adjacency_list[1].num_adjacent += 1; first += 1;
                }
            }
            let cap_arena = if lean215 { Vec::new() } else { capacity_arena };

            let slab_on = lean & (1usize << 30) != 0;
            let mut root_slab: Vec<HpfNode> = Vec::new();
            let strong_roots: Vec<HpfRoot> = if slab_on {
                root_slab = (0..2 * num_nodes as usize).map(|_| HpfNode::zeroed(num_params)).collect();
                let base = root_slab.as_mut_ptr();
                unsafe {
                    (0..num_nodes as usize).map(|i| {
                        let start = base.add(2 * i); let end = base.add(2 * i + 1);
                        (*start).next = end; (*end).prev = start;
                        HpfRoot { start, end }
                    }).collect()
                }
            } else {
                unsafe { (0..num_nodes as usize).map(|_| init_root(num_params)).collect() }
            };
            let label_count = vec![0i32; num_nodes as usize];
            let mut src_by_k: Vec<u32> = Vec::new();
            let mut awake_snk: Vec<u32> = Vec::new();
            if lean215 {
                src_by_k = (0..n_items as u32).collect();
                src_by_k.sort_by_key(|&i| k_src[i as usize]);
                awake_snk = (0..n_items as u32).collect();
            }
            if !outdeg.is_empty() {
                outdeg[0] += n_items as u32; outdeg[1] += n_items as u32;
                for i in 0..num_nodes as usize {
                    adjacency_list[i].out_of_tree.reserve_exact(outdeg[i] as usize);
                }
            }
            let stop_on = lean & (1usize << 36) != 0 && inst.n_budgets == 1;
            let mut nodew: Vec<i64> = Vec::new();
            if stop_on {
                nodew = vec![0i64; num_nodes as usize];
                for i in 0..n_items { nodew[i + 2] = inst.weights[i] as i64; }
            }
            let budget_w = if inst.n_budgets == 1 { inst.budgets[0] as i64 } else { i64::MAX };
            let pf = lean215 && lean & (1usize << 29) != 0;
            let mut wtv: Vec<f32> = Vec::new(); let mut cstv: Vec<f32> = Vec::new();
            let mut csrc: Vec<f32> = Vec::new(); let mut csnk: Vec<f32> = Vec::new();
            let mut snk_by_k: Vec<u32> = Vec::new();
            if pf {
                wtv  = (0..n_items).map(|i| adjacency_list[i + 2].wt).collect();
                cstv = (0..n_items).map(|i| adjacency_list[i + 2].cst).collect();
                csrc = Vec::with_capacity(n_items); csnk = Vec::with_capacity(n_items);
                for i in 0..n_items {
                    csrc.push(arc_list[n_edges + 2 * i].capacity);
                    csnk.push(arc_list[n_edges + 2 * i + 1].capacity);
                }
                snk_by_k = (0..n_items as u32).collect();
                snk_by_k.sort_by_key(|&i| k_snk[i as usize]);
            }
            let mut state = HpfState {
                max_bucket: 1, lean,
                max_ratio: max_degree_ratio, step, k_src, k_snk,
                awake_src: Vec::with_capacity(if lean215 { n_items } else { 0 }),
                awake_snk, src_by_k, src_ptr: 0,
                wtv, cstv, csrc, csnk, snk_by_k, snk_ptr: 0,
                lifted_w: 0, budget_w, nodew, stop_on, relift: 0, ext_pm, stop_at: -1,
                num_nodes, source: 1, sink: 2, num_params, highest_strong_label: 1,
                adjacency_list, strong_roots, label_count, arc_list,
                cap_arena, root_slab,
            };

            unsafe {
                let s = &mut state as *mut HpfState;
                if !one_pass {
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
                pos_items.sort_unstable();
                let mut sets: Vec<(i32, Vec<usize>)> = Vec::new();
                for (pos, item) in pos_items {
                    if let Some((last_pos, nodes)) = sets.last_mut() {
                        if *last_pos == pos {
                            nodes.push(item);
                            continue;
                        }
                    }
                    sets.push((pos, vec![item]));
                }
                if !slab_on { for i in 0..num_nodes as usize { free_root(&(*s).strong_roots[i]); } }
                if lean & (1usize << 35) != 0 { eprintln!("LIFT relift={} lifted_w={} budget={} stop_on={}", (*s).relift, (*s).lifted_w, (*s).budget_w, (*s).stop_on); }
                BreakpointSets { sets, maxr: (*s).max_ratio, step: (*s).step }
            }
        }
    }

    pub use hpf::get_breakpoints;
    pub use hpf::get_breakpoints_x;

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

    fn get_initial_node(right_nodes: &IntArray, um: &UtilMatrix, weights: &[i32], budget: i32) -> Option<usize> {
        let mut best: Option<usize> = None; let mut best_val = f64::NEG_INFINITY;
        for &nd in right_nodes {
            if weights[nd] > budget { continue; }
            let util: f64 = right_nodes.iter().map(|&m| um.get(nd, m)).sum::<f64>() / weights[nd] as f64;
            if util > best_val { best_val = util; best = Some(nd); }
        }
        best
    }

    fn run_greedy_left(um: &UtilMatrix, n_nodes: usize, left: IntArray, right_nodes: &IntArray,
                       budget: i32, beta: f64, weights: &[i32], ws: Option<&mut WarmStartLeft>) -> IntArray {
        run_greedy_left_b(um, n_nodes, left, right_nodes, budget, beta, weights, ws, false)
    }
    fn run_greedy_left_b(um: &UtilMatrix, n_nodes: usize, mut left: IntArray, right_nodes: &IntArray,
                       budget: i32, beta: f64, weights: &[i32], mut ws: Option<&mut WarmStartLeft>,
                       bufs: bool) -> IntArray {
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
        let use_bufs = bufs && ws.is_none();
        let mut buf_n: IntArray = Vec::with_capacity(cand_nodes.len());
        let mut buf_c: DblArray = Vec::with_capacity(cand_nodes.len());
        loop {
            if cand_nodes.is_empty() { break; }
            let best_idx = cand_contribs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
            let best_node = cand_nodes[best_idx];
            left.push(best_node); cur_w += weights[best_node] as f64;
            if use_bufs {
                buf_n.clear(); buf_c.clear();
                let rem = budget as f64 - cur_w;
                for (k, &nd) in cand_nodes.iter().enumerate() {
                    if k == best_idx { continue; }
                    if weights[nd] as f64 > rem { continue; }
                    buf_n.push(nd); buf_c.push(cand_contribs[k]);
                }
                for (k, &nd) in buf_n.iter().enumerate() {
                    buf_c[k] += (1.0 + 2.0 * beta) * um.get(nd, best_node) / weights[nd] as f64;
                }
                let t = cand_nodes; cand_nodes = buf_n; buf_n = t;
                let t = cand_contribs; cand_contribs = buf_c; buf_c = t;
                continue;
            }
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

    fn run_greedy_right(um: &UtilMatrix, n_nodes: usize, mut right_nodes: IntArray,
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

    fn run_greedy_right_left_handoff(um: &UtilMatrix, n_nodes: usize, mut right_nodes: IntArray,
                                      budget: i32, weights: &[i32]) -> IntArray {
        let mut cur_w: f64 = right_nodes.iter().map(|&k| weights[k] as f64).sum();
        let mut marginal_density = Vec::with_capacity(n_nodes);
        for nd in 0..n_nodes {
            let mut marginal: f64 = right_nodes.iter().map(|&m| um.get(nd, m)).sum();
            marginal += um.linear[nd];
            marginal /= weights[nd] as f64;
            marginal_density.push(marginal);
        }

        while !right_nodes.is_empty() && cur_w > budget as f64 {
            let best_idx = right_nodes.iter().enumerate()
                .max_by(|a, b| (-marginal_density[*a.1]).partial_cmp(&(-marginal_density[*b.1])).unwrap())
                .map(|(i, _)| i).unwrap();
            let best_node = right_nodes.remove(best_idx);
            cur_w -= weights[best_node] as f64;
            if um.view.is_some() {
                for nd in 0..n_nodes { marginal_density[nd] -= um.get(best_node, nd) / weights[nd] as f64; }
            } else {
            let row = &um.data[best_node * n_nodes..(best_node + 1) * n_nodes];
            for nd in 0..n_nodes {
                marginal_density[nd] -= row[nd] / weights[nd] as f64;
            }
            }
        }

        if right_nodes.is_empty() {
            let all_nodes: IntArray = (0..n_nodes).collect();
            return run_greedy_left(um, n_nodes, right_nodes, &all_nodes, budget, 0.0, weights, None);
        }

        let mut selected = vec![false; n_nodes];
        for &nd in &right_nodes { selected[nd] = true; }
        let mut cand_nodes = Vec::with_capacity(n_nodes - right_nodes.len());
        let rem = budget as f64 - cur_w;
        for nd in 0..n_nodes {
            if selected[nd] || weights[nd] as f64 > rem { continue; }
            cand_nodes.push(nd);
        }

        while !cand_nodes.is_empty() {
            let best_idx = cand_nodes.iter().enumerate()
                .max_by(|a, b| marginal_density[*a.1].partial_cmp(&marginal_density[*b.1]).unwrap())
                .map(|(i, _)| i).unwrap();
            let best_node = cand_nodes[best_idx];
            right_nodes.push(best_node);
            cur_w += weights[best_node] as f64;

            let mut new_cands = Vec::with_capacity(cand_nodes.len() - 1);
            let rem = budget as f64 - cur_w;
            for (k, &nd) in cand_nodes.iter().enumerate() {
                if k != best_idx && weights[nd] as f64 <= rem { new_cands.push(nd); }
            }

            if um.view.is_some() {
                for &nd in &new_cands { marginal_density[nd] += um.get(best_node, nd) / weights[nd] as f64; }
            } else {
            let row = &um.data[best_node * n_nodes..(best_node + 1) * n_nodes];
            for &nd in &new_cands {
                marginal_density[nd] += row[nd] / weights[nd] as f64;
            }
            }
            cand_nodes = new_cands;
        }

        right_nodes
    }

    struct GreedyResults { left_nodes: Vec<IntArray>, right_nodes: Vec<IntArray> }

    fn run_greedy(inst: &P1Instance, beta: f64, breakpoints: &[IntArray], bp_weights: &[f64]) -> GreedyResults {
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

    pub fn run_bp_algorithm(inst: &P1Instance, n_lambda_values: usize, lean: usize) -> QKPResult {
        run_bp_algorithm_v(inst, n_lambda_values, lean, None)
    }
    pub fn run_bp_algorithm_v(inst: &P1Instance, n_lambda_values: usize, lean: usize, view: Option<&Challenge>) -> QKPResult {
        run_bp_algorithm_x(inst, n_lambda_values, lean, view, 0, false)
    }
    pub fn run_bp_algorithm_x(inst: &P1Instance, n_lambda_values: usize, lean: usize, view: Option<&Challenge>, ext_pm: i32, want_bp: bool) -> QKPResult {
        if lean & (1usize << 26) != 0 {
            let z = get_breakpoints(inst, n_lambda_values, lean);
            unsafe { BP_SINK = BP_SINK.wrapping_add(z.sets.len() as u64); }
        }
        let bps = get_breakpoints_x(inst, n_lambda_values, lean, ext_pm);
        let n_breakpoints = bps.sets.len();

        if inst.n_budgets == 1 {
            let budget = inst.budgets[0];
            let budget_f = budget as f64;
            let mut cumsum = 0.0f64;
            let mut left_take = 0usize;
            let mut right_take = if budget_f <= 0.0 { 0usize } else { n_breakpoints };
            for (i, (_, nodes)) in bps.sets.iter().enumerate() {
                for &nd in nodes { cumsum += inst.weights[nd] as f64; }
                if cumsum <= budget_f { left_take = i + 1; }
                if right_take == n_breakpoints && cumsum >= budget_f { right_take = i + 1; }
            }

            if lean & (1usize << 35) != 0 {
                let bp_left  = if left_take  > 0 { bps.sets[left_take  - 1].0 } else { -1 };
                let bp_right = if right_take > 0 && right_take <= n_breakpoints { bps.sets[right_take - 1].0 } else { -1 };
                let last     = if n_breakpoints > 0 { bps.sets[n_breakpoints - 1].0 } else { -1 };
                unsafe {
                    let d = &hpf::DIAG;
                    let tot = d[10].max(1);
                    let mut tail = 0u64; for k in 0..10 { if k as f64 / 10.0 >= (bp_right.max(0) as f64) / (n_lambda_values as f64) { tail += d[k]; } }
                    eprintln!("DIAG n_bp={} left_take={} right_take={} bp_left={} bp_right={} bp_last={} nlam={} proot_tot={} deciles={:?} proot_after_stop={} ({:.1}%)",
                        n_breakpoints, left_take, right_take, bp_left, bp_right, last, n_lambda_values, tot, d[0..10].to_vec(), tail, 100.0*tail as f64/tot as f64);
                }
            }
            let mut left_init: IntArray = Vec::with_capacity(inst.n_items);
            for i in 0..left_take {
                for &nd in &bps.sets[i].1 { left_init.push(nd); }
            }
            let mut right_init: IntArray = Vec::with_capacity(inst.n_items);
            for i in 0..right_take {
                for &nd in &bps.sets[i].1 { right_init.push(nd); }
            }

            let um = match view { Some(ch) => build_utility_view(inst.n_items, ch), None => build_utility_matrix(inst.n_items, &inst.edges) };
            let all_nodes: IntArray = (0..inst.n_items).collect();
            let mut ws_left = WarmStartLeft::new();
            let left_nodes = if lean & (1usize << 27) != 0 {
                run_greedy_left_b(&um, inst.n_items, left_init, &all_nodes, budget, 0.0, &inst.weights, None, lean & (1usize << 28) != 0)
            } else {
                run_greedy_left(&um, inst.n_items, left_init, &all_nodes, budget, 0.0, &inst.weights, Some(&mut ws_left))
            };
            let _ = &ws_left;
            let right_nodes = run_greedy_right_left_handoff(&um, inst.n_items, right_init, budget, &inst.weights);

            let (ofv_left, ofv_right) = if lean & (1usize << 33) != 0 {
                compute_ofv2_m(&left_nodes, &right_nodes, inst.n_items, &inst.edges, lean >> 23 << 23)
            } else {
                (compute_ofv_m(&left_nodes,  inst.n_items, &inst.edges, lean >> 23 << 23),
                 compute_ofv_m(&right_nodes, inst.n_items, &inst.edges, lean >> 23 << 23))
            };
            let best_items = if ofv_left >= ofv_right {
                left_nodes
            } else { right_nodes };
            let mut bp: Vec<i32> = Vec::new();
            let mut bcross = -1i32;
            if want_bp {
                bp = vec![i32::MAX; inst.n_items];
                for (pos, nodes) in &bps.sets { for &nd in nodes { bp[nd] = *pos; } }
                if right_take > 0 && right_take <= n_breakpoints { bcross = bps.sets[right_take - 1].0; }
            }
            let lam_c = if bcross >= 2 {
                ((bps.maxr as f64) - (bcross - 1) as f64 * bps.step as f64, (bps.maxr as f64) - (bcross - 2) as f64 * bps.step as f64)
            } else { (0.0, 0.0) };
            return QKPResult { results: vec![BudgetResult { selected_items: best_items }], bp, bcross, lam_c };
        }

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
        let mut results: Vec<BudgetResult> = Vec::with_capacity(inst.n_budgets);
        for bi in 0..inst.n_budgets {
            let ofv_left  = compute_ofv(&gr.left_nodes[bi],  inst.n_items, &inst.edges);
            let ofv_right = compute_ofv(&gr.right_nodes[bi], inst.n_items, &inst.edges);
            let best_items = if ofv_left >= ofv_right {
                gr.left_nodes[bi].clone()
            } else { gr.right_nodes[bi].clone() };
            results.push(BudgetResult { selected_items: best_items });
        }
        QKPResult { results, bp: Vec::new(), bcross: -1, lam_c: (0.0, 0.0) }
    }

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

    pub static mut SSB_MODE: bool = false;
    pub static mut SSB_PRED: bool = false;
    pub static mut SSB_RM: Vec<usize> = Vec::new();
    pub static mut SSB_AD: Vec<usize> = Vec::new();
    pub static mut BP_SINK: u64 = 0;

    struct State<'a> {
        ch: &'a Challenge,
        selected_bit: Vec<bool>,
        contrib: Vec<i32>,
        total_value: i64,
        total_weight: u32,
        dp_cache: Vec<i64>,
        choose_cache: Vec<u8>,
        ldiv: Vec<i64>,
        ssb: bool,
        dpb: Option<(Vec<usize>, Vec<bool>, Vec<usize>, Vec<usize>, Vec<i64>)>,
    }

    impl<'a> State<'a> {
        fn new_empty(ch: &'a Challenge) -> Self {
            let n = ch.num_items;
            let mut contrib = vec![0i32; n];
            for i in 0..n { contrib[i] = ch.values[i] as i32; }
            const LDIV0: [i64; 11] = [2520, 2520, 1260, 840, 630, 504, 420, 360, 315, 280, 252];
            let mut ldiv = vec![0i64; n];
            for i in 0..n { ldiv[i] = LDIV0[(ch.weights[i] as usize).clamp(1, 10)]; }
            Self { ch, selected_bit: vec![false; n], contrib, total_value: 0, total_weight: 0, dp_cache: Vec::new(), choose_cache: Vec::new(), ldiv, ssb: unsafe { SSB_MODE }, dpb: None }
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
        #[inline(always)]
        fn replace_item(&mut self, rm: usize, cand: usize) {
            let delta = self.contrib[cand] as i64
                - self.contrib[rm] as i64
                - self.ch.interaction_values[rm][cand] as i64;
            self.total_value += delta;
            self.total_weight -= self.ch.weights[rm];
            self.total_weight += self.ch.weights[cand];
            let n = self.ch.num_items;
            let rm_row_ptr = unsafe { self.ch.interaction_values.get_unchecked(rm).as_ptr() };
            let cand_row_ptr = unsafe { self.ch.interaction_values.get_unchecked(cand).as_ptr() };
            let contrib_ptr = self.contrib.as_mut_ptr();
            unsafe {
                for k in 0..n {
                    let ck = contrib_ptr.add(k);
                    *ck = (*ck)
                        .wrapping_sub(*rm_row_ptr.add(k))
                        .wrapping_add(*cand_row_ptr.add(k));
                }
            }
            self.selected_bit[rm] = false;
            self.selected_bit[cand] = true;
        }
        fn selected_items(&self) -> Vec<usize> { (0..self.ch.num_items).filter(|&i| self.selected_bit[i]).collect() }
        fn clone_solution(&self) -> SolState { SolState { bits: self.selected_bit.clone(), contrib: self.contrib.clone(), value: self.total_value, weight: self.total_weight } }
        fn restore_solution(&mut self, sol: &SolState) { self.selected_bit.clone_from(&sol.bits); self.contrib.clone_from(&sol.contrib); self.total_value = sol.value; self.total_weight = sol.weight; }
    }

    #[derive(Clone)]
    struct SolState { bits: Vec<bool>, contrib: Vec<i32>, value: i64, weight: u32 }

    fn build_greedy_density(state: &mut State) { build_greedy_density_m(state, 0) }

    fn build_greedy_density_m(state: &mut State, gm: usize) {
        let n = state.ch.num_items; let cap = state.ch.max_weight;
        for i in 0..n { state.add_item(i); }
        if gm & 1 != 0 {
            while state.total_weight > cap {
                let mut worst = 0usize; let mut worst_s = i64::MAX;
                unsafe {
                    let sbt = state.selected_bit.as_ptr();
                    let ctr = state.contrib.as_ptr();
                    let wts = state.ch.weights.as_ptr();
                    for i in 0..n {
                        let c = *ctr.add(i) as i64;
                        let w = (*wts.add(i) as i64).max(1);
                        let s = dw(c * 1000, w);
                        let take = *sbt.add(i) & (s < worst_s);
                        worst_s = if take { s } else { worst_s };
                        worst   = if take { i } else { worst };
                    }
                }
                state.remove_item(worst);
            }
        } else {
        while state.total_weight > cap {
            let mut worst = 0; let mut worst_s = i64::MAX;
            for i in 0..n {
                if state.selected_bit[i] {
                    let c = state.contrib[i] as i64;
                    let w = (state.ch.weights[i] as i64).max(1);
                    let s = dw(c * 1000, w);
                    if s < worst_s { worst_s = s; worst = i; }
                }
            }
            state.remove_item(worst);
        }
        }
        let mut by_density: Vec<usize> = (0..n).collect();
        let mut target = vec![false; n];
        let mut to_rm: Vec<usize> = Vec::with_capacity(n);
        let mut to_add: Vec<usize> = Vec::with_capacity(n);
        for _ in 0..2 {
            for i in 0..n { by_density[i] = i; }
            let contrib = &state.contrib; let weights = &state.ch.weights;
            const LDIV: [i64; 11] = [2520, 2520, 1260, 840, 630, 504, 420, 360, 315, 280, 252];
            let mut dkey = vec![0i64; n];
            for i in 0..n { dkey[i] = contrib[i] as i64 * LDIV[(weights[i] as usize).clamp(1, 10)]; }
            by_density.sort_unstable_by(|&a, &b| unsafe {
                dkey.get_unchecked(b).cmp(dkey.get_unchecked(a))
            });
            target.fill(false);
            let mut rem = cap;
            for &i in &by_density {
                if state.ch.weights[i] <= rem { target[i] = true; rem -= state.ch.weights[i]; }
            }
            to_rm.clear(); to_add.clear();
            if gm & 2 != 0 {
                to_rm.reserve(n + 4); to_add.reserve(n + 4);
                unsafe {
                    let rp = to_rm.as_mut_ptr(); let ap = to_add.as_mut_ptr();
                    let sp = state.selected_bit.as_ptr(); let tp = target.as_ptr();
                    let mut ri = 0usize; let mut ai = 0usize;
                    for i in 0..n {
                        let sb = *sp.add(i); let tb = *tp.add(i);
                        rp.add(ri).write(i); ri += (sb & !tb) as usize;
                        ap.add(ai).write(i); ai += (tb & !sb) as usize;
                    }
                    to_rm.set_len(ri); to_add.set_len(ai);
                }
            } else {
            for i in 0..n {
                if state.selected_bit[i] && !target[i] { to_rm.push(i); }
                if !state.selected_bit[i] && target[i] { to_add.push(i); }
            }
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

    fn build_greedy_hub(state: &mut State) { build_greedy_hub_s(state, None) }

    fn row_sums_of(challenge: &Challenge) -> Vec<i64> {
        let n = challenge.num_items;
        (0..n).map(|i| challenge.interaction_values[i].iter().map(|&v| v as i64).sum::<i64>()).collect()
    }

    fn build_greedy_hub_s(state: &mut State, rs: Option<&[i64]>) {
        let n = state.ch.num_items; let cap = state.ch.max_weight;
        let mut hub_scores: Vec<(usize, i64)> = match rs {
            Some(r) => (0..n).map(|i| (i, r[i])).collect(),
            None => (0..n).map(|i| {
                let s: i64 = state.ch.interaction_values[i].iter().map(|&v| v as i64).sum();
                (i, s)
            }).collect(),
        };
        hub_scores.sort_unstable_by_key(|&(_, s)| std::cmp::Reverse(s));
        for &(i, _) in &hub_scores {
            if state.total_weight + state.ch.weights[i] <= cap { state.add_item(i); }
        }
    }

    fn build_greedy_synergy_weight(state: &mut State) { build_greedy_synergy_weight_s(state, None) }

    fn build_greedy_synergy_weight_s(state: &mut State, rs: Option<&[i64]>) {
        let n = state.ch.num_items; let cap = state.ch.max_weight;
        let mut scores: Vec<(usize, i64)> = (0..n).map(|i| {
            let avg_syn: i64 = if n > 1 {
                let tot: i64 = match rs {
                    Some(r) => r[i],
                    None => state.ch.interaction_values[i].iter().map(|&v| v as i64).sum::<i64>(),
                };
                tot / (n as i64 - 1)
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
        let initial_slack = state.slack();
        let mut candidates: Vec<usize> = (0..n)
            .filter(|&i| !state.selected_bit[i] && state.ch.weights[i] <= initial_slack)
            .collect();
        loop {
            let slack = state.slack();
            if slack == 0 { break; }
            let mut best_i: Option<usize> = None; let mut best_s: i64 = i64::MIN;
            let mut second_i: Option<usize> = None; let mut second_s: i64 = i64::MIN;
            for &i in &candidates {
                let c = state.contrib[i] as i64;
                if c <= 0 { continue; }
                let w = (state.ch.weights[i] as i64).max(1);
                let mut s = match mode {
                    2 => c,
                    3 => dw(c * 1000, w) + (state.ch.weights[i] as i64) * 3,
                    _ => dw(c * 1000, w),
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
            if let Some(i) = pick {
                state.add_item(i);
                let new_slack = state.slack();
                candidates.retain(|&candidate| candidate != i && state.ch.weights[candidate] <= new_slack);
            } else { break; }
        }
    }

    fn build_hub_pair_kth_from_pairs(state: &mut State, pairs: &[(i32, usize, usize)], k: usize, pred: bool) {
        let n = state.ch.num_items;
        let mut used = Vec::new(); let mut count = 0;
        for &(_, pi, pj) in pairs {
            if used.contains(&pi) || used.contains(&pj) { continue; }
            if count == k { state.add_item(pi); state.add_item(pj); break; }
            used.push(pi); used.push(pj); count += 1;
        }
        loop {
            let slack = state.slack();
            if slack == 0 { break; }
            let mut best_i: Option<usize> = None; let mut best_s: i64 = 0;
            if pred {
                let mut thr: i64 = 1;
                unsafe {
                    let sp = state.selected_bit.as_ptr();
                    let wp = state.ch.weights.as_ptr();
                    let cp = state.contrib.as_ptr();
                    let mut bi = usize::MAX;
                    for i in 0..n {
                        let wu = *wp.add(i);
                        let wl = (wu as i64).max(1);
                        let x = *cp.add(i) as i64 * 1000;
                        if (!*sp.add(i)) & (wu <= slack) & (x >= thr * wl) {
                            best_s = x / wl; thr = best_s + 1; bi = i;
                        }
                    }
                    if bi != usize::MAX { best_i = Some(bi); }
                }
            } else {
            for i in 0..n {
                if state.selected_bit[i] { continue; }
                if state.ch.weights[i] > slack { continue; }
                let c = state.contrib[i] as i64;
                if c <= 0 { continue; }
                let w = (state.ch.weights[i] as i64).max(1);
                let s = dw(c * 1000, w);
                if s > best_s { best_s = s; best_i = Some(i); }
            }
            }
            if let Some(i) = best_i { state.add_item(i); } else { break; }
        }
    }

    fn dp_refinement_hp(state: &mut State, core_half: usize, hp: &Hparams) {
        let n = state.ch.num_items; let cap = state.ch.max_weight;
        let reuse_bufs = hp.neutral_mask & (1usize << 27) != 0;
        let (mut by_density, mut target, mut to_rm, mut to_add, mut lent_key) = match state.dpb.take() {
            Some(t) => t,
            None => ((0..n).collect::<Vec<usize>>(), vec![false; n], Vec::with_capacity(n), Vec::with_capacity(n), Vec::<i64>::new()),
        };
        if by_density.len() != n { by_density.clear(); by_density.resize(n, 0); }
        for i in 0..n { by_density[i] = i; }
        if target.len() != n { target.clear(); target.resize(n, false); }
        to_rm.clear(); to_add.clear();
        let weights = &state.ch.weights;
        let hoist = hp.neutral_mask & 2 != 0;
        if hoist && lent_key.len() != n { lent_key.clear(); lent_key.resize(n, 0); }
        let mut dkey_hoisted: Vec<i64> = lent_key;
        let stop_when_full = hp.neutral_mask & 1 != 0;
        let fuse = hp.neutral_mask & 16 != 0;
        let precomp = hp.neutral_mask & 128 != 0;
        for _iter in 0..hp.dp_passes {
            let contrib = &state.contrib;
            const LDIV: [i64; 11] = [2520, 2520, 1260, 840, 630, 504, 420, 360, 315, 280, 252];
            let mut dkey_owned: Vec<i64> = if hoist { Vec::new() } else { vec![0i64; n] };
            let dkey: &mut Vec<i64> = if hoist { &mut dkey_hoisted } else { &mut dkey_owned };
            if fuse {
                if precomp && hp.neutral_mask & 2048 != 0 {
                    unsafe {
                        let bp = by_density.as_mut_ptr();
                        let kp = dkey.as_mut_ptr();
                        let cp = contrib.as_ptr();
                        let lp = state.ldiv.as_ptr();
                        let m = n & !3usize;
                        let mut i = 0usize;
                        while i < m {
                            *bp.add(i)     = i;     *kp.add(i)     = *cp.add(i)     as i64 * *lp.add(i);
                            *bp.add(i + 1) = i + 1; *kp.add(i + 1) = *cp.add(i + 1) as i64 * *lp.add(i + 1);
                            *bp.add(i + 2) = i + 2; *kp.add(i + 2) = *cp.add(i + 2) as i64 * *lp.add(i + 2);
                            *bp.add(i + 3) = i + 3; *kp.add(i + 3) = *cp.add(i + 3) as i64 * *lp.add(i + 3);
                            i += 4;
                        }
                        while i < n { *bp.add(i) = i; *kp.add(i) = *cp.add(i) as i64 * *lp.add(i); i += 1; }
                    }
                } else if precomp {
                    let ldiv = &state.ldiv;
                    for i in 0..n { by_density[i] = i; dkey[i] = contrib[i] as i64 * ldiv[i]; }
                } else {
                for i in 0..n {
                    by_density[i] = i;
                    dkey[i] = contrib[i] as i64 * LDIV[(weights[i] as usize).clamp(1, 10)];
                }
                }
            } else {
            for i in 0..n { by_density[i] = i; }
            for i in 0..n { dkey[i] = contrib[i] as i64 * LDIV[(weights[i] as usize).clamp(1, 10)]; }
            }
            by_density.sort_unstable_by(|&a, &b| unsafe {
                dkey.get_unchecked(b).cmp(dkey.get_unchecked(a))
            });
            let mut idx_last_inserted = 0usize; let mut idx_first_rejected = n;
            let mut rem = cap;
            if stop_when_full && hp.exact_mask & (1usize << 39) != 0 {
                unsafe {
                    let bp = by_density.as_ptr(); let wp = weights.as_ptr();
                    let mut idx = 0usize;
                    while idx < n {
                        let w = *wp.add(*bp.add(idx));
                        let fits = w <= rem;
                        rem = if fits { rem - w } else { rem };
                        idx_last_inserted = if fits { idx } else { idx_last_inserted };
                        let first = (!fits) & (idx_first_rejected == n);
                        idx_first_rejected = if first { idx } else { idx_first_rejected };
                        if rem == 0 {
                            if idx_first_rejected == n && idx + 1 < n { idx_first_rejected = idx + 1; }
                            break;
                        }
                        idx += 1;
                    }
                }
            } else {
            for (idx, &i) in by_density.iter().enumerate() {
                let w = weights[i];
                if w <= rem { rem -= w; idx_last_inserted = idx; }
                else if idx_first_rejected == n { idx_first_rejected = idx; }
                if stop_when_full && rem == 0 {
                    if idx_first_rejected == n && idx + 1 < by_density.len() { idx_first_rejected = idx + 1; }
                    break;
                }
            }
            }
            let left  = idx_first_rejected.saturating_sub(core_half + 1);
            let right = (idx_last_inserted + core_half + 1).min(n);
            let fuse_locked = hp.exact_mask & (1usize << 40) != 0;
            let used_locked: u64 = if fuse_locked {
                target.fill(false);
                let mut acc = 0u64;
                unsafe {
                    let bp = by_density.as_ptr(); let wp = weights.as_ptr(); let tp = target.as_mut_ptr();
                    for q in 0..left { let i = *bp.add(q); acc += *wp.add(i) as u64; *tp.add(i) = true; }
                }
                acc
            } else {
                by_density[..left].iter().map(|&i| weights[i] as u64).sum()
            };
            let rem_cap = (cap as u64).saturating_sub(used_locked) as usize;
            let core = &by_density[left..right];
            let myk = core.len();
            if myk == 0 || rem_cap == 0 { break; }

            let mut total_core_weight: usize = 0;
            let mut total_pos_weight:  usize = 0;
            let mut all_pos_fit = true;
            for &it in core {
                let wt = weights[it] as usize;
                total_core_weight += wt;
                if contrib[it] > 0 {
                    total_pos_weight += wt;
                    if total_pos_weight > rem_cap { all_pos_fit = false; }
                }
            }

            if !fuse_locked {
                target.fill(false);
                for &it in &by_density[..left] { target[it] = true; }
            }
            if all_pos_fit {
                for &it in core {
                    if contrib[it] > 0 { target[it] = true; }
                }
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
                let predical = hp.exact_mask & (1usize << 38) != 0;
                for (t, &it) in core.iter().enumerate() {
                    let wt = weights[it] as usize;
                    if wt > myw { continue; }
                    let val = contrib[it] as i64;
                    let new_hi = (w_hi + wt).min(myw);
                    if predical {
                        unsafe {
                            let dp = state.dp_cache.as_mut_ptr();
                            let chp = state.choose_cache.as_mut_ptr().add(t * dp_size);
                            let mut w = new_hi;
                            loop {
                                let cand = *dp.add(w - wt) + val;
                                let old = *dp.add(w);
                                let take = cand > old;
                                *dp.add(w) = if take { cand } else { old };
                                *chp.add(w) = take as u8;
                                if w == wt { break; }
                                w -= 1;
                            }
                        }
                    } else {
                    for w in (wt..=new_hi).rev() {
                        let cand = state.dp_cache[w - wt] + val;
                        if cand > state.dp_cache[w] {
                            state.dp_cache[w] = cand;
                            state.choose_cache[t * dp_size + w] = 1;
                        }
                    }
                    }
                    w_hi = new_hi;
                }
                let mut w_star = (0..=myw).max_by_key(|&w| state.dp_cache[w]).unwrap_or(0);
                for t in (0..myk).rev() {
                    let it = core[t]; let wt = weights[it] as usize;
                    if wt <= w_star && state.choose_cache[t * dp_size + w_star] == 1 {
                        target[it] = true; w_star -= wt;
                    }
                }
            }

            to_rm.clear(); to_add.clear();
            if hp.neutral_mask & 4096 != 0 {
                to_rm.reserve(n + 4); to_add.reserve(n + 4);
                unsafe {
                    let rp = to_rm.as_mut_ptr(); let ap = to_add.as_mut_ptr();
                    let sp = state.selected_bit.as_ptr(); let tp = target.as_ptr();
                    let mut ri = 0usize; let mut ai = 0usize;
                    let m = n & !3usize; let mut i = 0usize;
                    while i < m {
                        let s0 = *sp.add(i);     let t0 = *tp.add(i);
                        rp.add(ri).write(i);     ri += (s0 & !t0) as usize;
                        ap.add(ai).write(i);     ai += (t0 & !s0) as usize;
                        let s1 = *sp.add(i + 1); let t1 = *tp.add(i + 1);
                        rp.add(ri).write(i + 1); ri += (s1 & !t1) as usize;
                        ap.add(ai).write(i + 1); ai += (t1 & !s1) as usize;
                        let s2 = *sp.add(i + 2); let t2 = *tp.add(i + 2);
                        rp.add(ri).write(i + 2); ri += (s2 & !t2) as usize;
                        ap.add(ai).write(i + 2); ai += (t2 & !s2) as usize;
                        let s3 = *sp.add(i + 3); let t3 = *tp.add(i + 3);
                        rp.add(ri).write(i + 3); ri += (s3 & !t3) as usize;
                        ap.add(ai).write(i + 3); ai += (t3 & !s3) as usize;
                        i += 4;
                    }
                    while i < n {
                        let sb = *sp.add(i); let tb = *tp.add(i);
                        rp.add(ri).write(i); ri += (sb & !tb) as usize;
                        ap.add(ai).write(i); ai += (tb & !sb) as usize;
                        i += 1;
                    }
                    to_rm.set_len(ri); to_add.set_len(ai);
                }
            } else {
            for i in 0..n {
                if state.selected_bit[i] && !target[i] { to_rm.push(i); }
                else if target[i] && !state.selected_bit[i] { to_add.push(i); }
            }
            }
            if to_rm.is_empty() && to_add.is_empty() { break; }
            for &r in &to_rm { state.remove_item(r); }
            for &a in &to_add { state.add_item(a); }
        }
        if reuse_bufs { state.dpb = Some((by_density, target, to_rm, to_add, dkey_hoisted)); }
    }

    fn apply_best_add(state: &mut State, pred: bool) -> bool {
        let slack = state.slack(); if slack == 0 { return false; }
        let n = state.ch.num_items;
        if pred {
            let mut bi: usize = usize::MAX; let mut bd: i32 = 0;
            unsafe {
                let sbt = state.selected_bit.as_ptr();
                let wts = state.ch.weights.as_ptr();
                let ctr = state.contrib.as_ptr();
                for i in 0..n {
                    let d = *ctr.add(i);
                    let take = !*sbt.add(i) & (*wts.add(i) <= slack) & (d > bd);
                    bd = if take { d } else { bd };
                    bi = if take { i } else { bi };
                }
            }
            if bi != usize::MAX { state.add_item(bi); return true; }
            return false;
        }
        let mut best_i: Option<usize> = None; let mut best_d: i32 = 0;
        for i in 0..n {
            if state.selected_bit[i] { continue; }
            if state.ch.weights[i] > slack { continue; }
            let d = state.contrib[i];
            if d > best_d { best_d = d; best_i = Some(i); }
        }
        if let Some(i) = best_i { state.add_item(i); true } else { false }
    }

    fn apply_best_swap_1_1(state: &mut State, selected: &[usize], pred: bool) -> bool {
        let n = state.ch.num_items; let slack = state.slack();
        if pred {
            let mut bd: i32 = 0; let mut br: usize = usize::MAX;
            let mut bc: usize = usize::MAX; let mut brm: usize = 0;
            let ns = selected.len();
            unsafe {
                let wts = state.ch.weights.as_ptr();
                let ctr = state.contrib.as_ptr();
                let sbt = state.selected_bit.as_ptr();
                let sel = selected.as_ptr();
                for cand in 0..n {
                    if *sbt.add(cand) { continue; }
                    let wc = *wts.add(cand);
                    let cand_contrib = *ctr.add(cand);
                    let row = state.ch.interaction_values.get_unchecked(cand).as_ptr();
                    let base = cand;
                    for rm_pos in 0..ns {
                        let rm = *sel.add(rm_pos);
                        let feas = wc <= *wts.add(rm) + slack;
                        let delta = cand_contrib - *ctr.add(rm) - *row.add(rm);
                        let rank = rm_pos * n + base;
                        let take = feas & (delta > 0) & ((delta > bd) | ((delta == bd) & (rank < br)));
                        bd  = if take { delta } else { bd };
                        br  = if take { rank }  else { br };
                        bc  = if take { cand }  else { bc };
                        brm = if take { rm }    else { brm };
                    }
                }
            }
            if bc != usize::MAX { state.replace_item(brm, bc); return true; }
            return false;
        }
        let mut best: Option<(usize, usize, i32, usize)> = None;
        for cand in 0..n {
            if state.selected_bit[cand] { continue; }
            let wc = state.ch.weights[cand];
            let cand_contrib = state.contrib[cand];
            let interaction_row = &state.ch.interaction_values[cand];
            for (rm_pos, &rm) in selected.iter().enumerate() {
                if wc > state.ch.weights[rm] + slack { continue; }
                let delta = cand_contrib - state.contrib[rm] - interaction_row[rm];
                if delta <= 0 { continue; }
                let rank = rm_pos * n + cand;
                if best.map_or(true, |(_, _, bd, br)| delta > bd || (delta == bd && rank < br)) {
                    best = Some((cand, rm, delta, rank));
                }
            }
        }
        if let Some((cand, rm, _, _)) = best { state.replace_item(rm, cand); true } else { false }
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

    fn local_search_vnd_fast(state: &mut State, pm: usize) {
        let n = state.ch.num_items;
        let mut selected_buf: Vec<usize> = Vec::with_capacity(n);
        for _ in 0..80 {
            if apply_best_add(state, pm & 2 != 0) { continue; }
            selected_buf.clear();
            for i in 0..n { if state.selected_bit[i] { selected_buf.push(i); } }
            if apply_best_swap_1_1(state, &selected_buf, pm & 1 != 0) { continue; }
            break;
        }
    }

    fn local_search_vnd_medium(state: &mut State, k: usize, pm: usize) {
        let n = state.ch.num_items;
        let mut selected_buf: Vec<usize> = Vec::with_capacity(n);
        for _ in 0..120 {
            if apply_best_add(state, pm & 2 != 0) { continue; }
            selected_buf.clear();
            for i in 0..n { if state.selected_bit[i] { selected_buf.push(i); } }
            if apply_best_swap_1_1(state, &selected_buf, pm & 1 != 0) { continue; }
            if apply_pair_add(state) { continue; }
            if apply_swap_2_2_bounded(state, k) { continue; }
            break;
        }
    }

    fn ils_vnd(state: &mut State, hp: &Hparams) {
        match hp.ils_vnd_level {
            0 => local_search_vnd_fast(state, (hp.exact_mask >> 10) & 3),
            1 => local_search_vnd_medium(state, hp.bounded_2_2_k, (hp.exact_mask >> 10) & 3),
            _ => local_search_vnd_heavy(state, (hp.exact_mask >> 10) & 3),
        }
    }

    fn local_search_vnd_heavy(state: &mut State, pm: usize) {
        let n = state.ch.num_items;
        let mut selected_buf: Vec<usize> = Vec::with_capacity(n);
        for _ in 0..300 {
            if apply_best_add(state, pm & 2 != 0) { continue; }
            selected_buf.clear();
            for i in 0..n { if state.selected_bit[i] { selected_buf.push(i); } }
            if apply_best_swap_1_1(state, &selected_buf, pm & 1 != 0) { continue; }
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
        crossover_frequency_m(population, ch, rng, 0)
    }
    fn crossover_frequency_m(population: &[SolState], ch: &Challenge, rng: &mut Rng, nm: usize) -> Vec<bool> {
        let n = ch.num_items; let pop_size = population.len();
        let mut freq = vec![0usize; n];
        let g = nm & (1usize << 17) != 0;
        if g {
            for sol in population {
                unsafe {
                    let b = sol.bits.as_ptr(); let f = freq.as_mut_ptr();
                    let m = n & !3usize; let mut i = 0usize;
                    while i < m {
                        *f.add(i)     += *b.add(i)     as usize;
                        *f.add(i + 1) += *b.add(i + 1) as usize;
                        *f.add(i + 2) += *b.add(i + 2) as usize;
                        *f.add(i + 3) += *b.add(i + 3) as usize;
                        i += 4;
                    }
                    while i < n { *f.add(i) += *b.add(i) as usize; i += 1; }
                }
            }
        } else {
        for sol in population { for i in 0..n { if sol.bits[i] { freq[i] += 1; } } }
        }
        let threshold = (pop_size * 3) / 4;
        let mut child_bits = vec![false; n]; let mut child_weight: u32 = 0;
        let mut consensus: Vec<usize> = Vec::new(); let mut exploratory: Vec<usize> = Vec::new();
        if g {
            consensus.reserve(n + 4); exploratory.reserve(n + 4);
            unsafe {
                let cp = consensus.as_mut_ptr(); let ep = exploratory.as_mut_ptr();
                let fp = freq.as_ptr();
                let mut ci = 0usize; let mut ei = 0usize;
                let m = n & !3usize; let mut i = 0usize;
                while i < m {
                    let f0 = *fp.add(i);     cp.add(ci).write(i);     ci += (f0 > threshold) as usize;     ep.add(ei).write(i);     ei += ((f0 > 0) & (f0 <= threshold)) as usize;
                    let f1 = *fp.add(i + 1); cp.add(ci).write(i + 1); ci += (f1 > threshold) as usize;     ep.add(ei).write(i + 1); ei += ((f1 > 0) & (f1 <= threshold)) as usize;
                    let f2 = *fp.add(i + 2); cp.add(ci).write(i + 2); ci += (f2 > threshold) as usize;     ep.add(ei).write(i + 2); ei += ((f2 > 0) & (f2 <= threshold)) as usize;
                    let f3 = *fp.add(i + 3); cp.add(ci).write(i + 3); ci += (f3 > threshold) as usize;     ep.add(ei).write(i + 3); ei += ((f3 > 0) & (f3 <= threshold)) as usize;
                    i += 4;
                }
                while i < n {
                    let f0 = *fp.add(i); cp.add(ci).write(i); ci += (f0 > threshold) as usize; ep.add(ei).write(i); ei += ((f0 > 0) & (f0 <= threshold)) as usize;
                    i += 1;
                }
                consensus.set_len(ci); exploratory.set_len(ei);
            }
        } else {
        for i in 0..n {
            if freq[i] > threshold { consensus.push(i); }
            else if freq[i] > 0 { exploratory.push(i); }
        }
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

    #[inline]
    fn zobrist_value(i: usize) -> u64 {
        let mut h: u64 = 0x517CC1B727220A95;
        h ^= (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
        h = h.rotate_left(17).wrapping_mul(0xBF58476D1CE4E5B9);
        h
    }

    fn build_zobrist_table(n: usize) -> Vec<u64> {
        (0..n).map(zobrist_value).collect()
    }

    #[inline]
    fn hash_bits_with_table(bits: &[bool], zobrist_table: &[u64]) -> u64 {
        let mut h: u64 = 0;
        for i in 0..bits.len() { if bits[i] { h ^= zobrist_table[i]; } }
        h
    }

    #[inline]
    fn hamming_distance(bits_a: &[bool], bits_b: &[bool]) -> usize {
        let mut distance = 0usize;
        for i in 0..bits_a.len() {
            if bits_a[i] != bits_b[i] { distance += 1; }
        }
        distance
    }

    #[inline]
    fn pack_bits(bits: &[bool], out: &mut Vec<u64>) {
        pack_bits_m(bits, out, 1)
    }

    fn pack_bits_m(bits: &[bool], out: &mut Vec<u64>, fast: usize) {
        let n = bits.len();
        let nw = (n + 63) / 64;
        out.clear(); out.resize(nw, 0u64);
        let p = bits.as_ptr() as *const u8;
        for k in 0..nw {
            let base = k << 6;
            let hi = (base + 64).min(n);
            let mut w = 0u64;
            let mut i = base;
            if fast != 0 {
                while i + 8 <= hi {
                    let chunk: [u8; 8] = unsafe { std::ptr::read_unaligned(p.add(i) as *const [u8; 8]) };
                    let v = u64::from_le_bytes(chunk);
                    w |= (v.wrapping_mul(0x0102_0408_1020_4080) >> 56) << (i - base);
                    i += 8;
                }
            }
            while i < hi { w |= (bits[i] as u64) << (i - base); i += 1; }
            out[k] = w;
        }
    }

    #[inline]
    fn hash_from_words(w: &[u64], table: &[u64]) -> u64 {
        let mut h: u64 = 0;
        for k in 0..w.len() {
            let mut x = w[k];
            let base = k << 6;
            while x != 0 {
                let b = x.trailing_zeros() as usize;
                h ^= table[base + b];
                x &= x - 1;
            }
        }
        h
    }

    #[inline]
    fn hamming_words(a: &[u64], b: &[u64]) -> usize {
        let mut d = 0u32;
        for k in 0..a.len() { d += (a[k] ^ b[k]).count_ones(); }
        d as usize
    }

    fn dedup_population_t8_w(population: &mut Vec<SolState>, zobrist_table: &[u64], fast: bool, words: bool, packhash: bool, packfast: usize) {
        population.sort_unstable_by_key(|s| std::cmp::Reverse(s.value));
        let mut unique = Vec::with_capacity(population.len());
        let mut seen = Vec::with_capacity(population.len());
        let nb = population.first().map(|p| p.bits.len()).unwrap_or(0);
        let nw = (nb + 63) / 64;
        let mut upacked: Vec<u64> = Vec::new();
        let mut tmp: Vec<u64> = Vec::with_capacity(nw);
        for p in population.drain(..) {
            let h = if packhash {
                pack_bits_m(&p.bits, &mut tmp, packfast);
                hash_from_words(&tmp, zobrist_table)
            } else {
                hash_bits_with_table(&p.bits, zobrist_table)
            };
            if !seen.contains(&h) {
                seen.push(h);
                if packhash { upacked.extend_from_slice(&tmp); }
                unique.push(p);
            }
        }
        let n0 = unique.len();
        let mut dcache: Vec<usize> = Vec::new();
        let mut alive: Vec<usize> = Vec::new();
        if fast && n0 > 8 {
            dcache = vec![0usize; n0 * n0];
            if words {
                let mut packed: Vec<u64> = Vec::new();
                if packhash { packed = upacked; } else {
                    packed = vec![0u64; n0 * nw];
                    let mut t2: Vec<u64> = Vec::with_capacity(nw);
                    for i in 0..n0 {
                        pack_bits_m(&unique[i].bits, &mut t2, packfast);
                        packed[i * nw..i * nw + nw].copy_from_slice(&t2);
                    }
                }
                for i in 0..n0 {
                    for j in (i + 1)..n0 {
                        let d = hamming_words(&packed[i * nw..i * nw + nw], &packed[j * nw..j * nw + nw]);
                        dcache[i * n0 + j] = d;
                        dcache[j * n0 + i] = d;
                    }
                }
            } else {
                for i in 0..n0 {
                    for j in (i + 1)..n0 {
                        let d = hamming_distance(&unique[i].bits, &unique[j].bits);
                        dcache[i * n0 + j] = d;
                        dcache[j * n0 + i] = d;
                    }
                }
            }
            alive = (0..n0).collect();
        }
        while unique.len() > 8 {
            let count = unique.len();
            let mut nearest_distance = vec![usize::MAX; count];
            for i in 0..count {
                for j in (i + 1)..count {
                    let distance = if fast {
                        dcache[alive[i] * n0 + alive[j]]
                    } else {
                        hamming_distance(&unique[i].bits, &unique[j].bits)
                    };
                    if distance < nearest_distance[i] { nearest_distance[i] = distance; }
                    if distance < nearest_distance[j] { nearest_distance[j] = distance; }
                }
            }

            let mut diversity_order: Vec<usize> = (0..count).collect();
            diversity_order.sort_unstable_by(|&a, &b| {
                nearest_distance[b]
                    .cmp(&nearest_distance[a])
                    .then_with(|| unique[b].value.cmp(&unique[a].value))
            });
            let mut diversity_rank = vec![0usize; count];
            for (rank, &index) in diversity_order.iter().enumerate() {
                diversity_rank[index] = rank;
            }

            let mut remove_index = 2usize;
            let mut worst_biased_fitness = 0usize;
            for i in 2..count {
                let biased_fitness = i * 10 + diversity_rank[i] * 4;
                if biased_fitness > worst_biased_fitness
                    || (biased_fitness == worst_biased_fitness
                        && unique[i].value < unique[remove_index].value)
                {
                    worst_biased_fitness = biased_fitness;
                    remove_index = i;
                }
            }
            unique.remove(remove_index);
            if fast { alive.remove(remove_index); }
            unique.sort_unstable_by_key(|s| std::cmp::Reverse(s.value));
        }
        *population = unique;
    }

    fn select_diverse_parent_pair(population: &[SolState], rng: &mut Rng) -> (usize, usize) {
        select_diverse_parent_pair_m(population, rng, 0)
    }
    fn select_diverse_parent_pair_m(population: &[SolState], rng: &mut Rng, nm: usize) -> (usize, usize) {
        let elite_count = population.len().min(4);
        let first = rng.next_usize(elite_count);
        let mut second = if first == 0 && population.len() > 1 { 1 } else { 0 };
        let mut best_score = 0usize;
        if nm & (1usize << 18) != 0 {
            let mut pf: Vec<u64> = Vec::new(); let mut pc: Vec<u64> = Vec::new();
            pack_bits(&population[first].bits, &mut pf);
            for candidate in 0..population.len() {
                if candidate == first { continue; }
                pack_bits(&population[candidate].bits, &mut pc);
                let distance = hamming_words(&pf, &pc);
                let quality_bonus = population.len() - candidate;
                let score = distance * 8 + quality_bonus;
                if score > best_score { best_score = score; second = candidate; }
            }
            return (first, second);
        }
        for candidate in 0..population.len() {
            if candidate == first { continue; }
            let distance = hamming_distance(&population[first].bits, &population[candidate].bits);
            let quality_bonus = population.len() - candidate;
            let score = distance * 8 + quality_bonus;
            if score > best_score {
                best_score = score;
                second = candidate;
            }
        }
        (first, second)
    }

    fn set_state_from_bits(state: &mut State, bits: &[bool]) {
        let n = state.ch.num_items;
        let mut differences = 0usize;
        let mut target_selected = 0usize;
        if state.ssb {
            unsafe {
                let bp = bits.as_ptr(); let sp = state.selected_bit.as_ptr();
                let m = n & !3usize; let mut i = 0usize;
                while i < m {
                    let b0 = *bp.add(i);     let s0 = *sp.add(i);     target_selected += b0 as usize; differences += (b0 != s0) as usize;
                    let b1 = *bp.add(i + 1); let s1 = *sp.add(i + 1); target_selected += b1 as usize; differences += (b1 != s1) as usize;
                    let b2 = *bp.add(i + 2); let s2 = *sp.add(i + 2); target_selected += b2 as usize; differences += (b2 != s2) as usize;
                    let b3 = *bp.add(i + 3); let s3 = *sp.add(i + 3); target_selected += b3 as usize; differences += (b3 != s3) as usize;
                    i += 4;
                }
                while i < n { let b = *bp.add(i); let sb = *sp.add(i); target_selected += b as usize; differences += (b != sb) as usize; i += 1; }
            }
        } else {
        for i in 0..n {
            if bits[i] { target_selected += 1; }
            if bits[i] != state.selected_bit[i] { differences += 1; }
        }
        }
        if differences == 0 { return; }
        if differences <= target_selected {
            if unsafe { SSB_PRED } {
                unsafe {
                let rm = &mut SSB_RM; let ad = &mut SSB_AD;
                if rm.capacity() < n + 4 { *rm = Vec::with_capacity(n + 4); }
                if ad.capacity() < n + 4 { *ad = Vec::with_capacity(n + 4); }
                rm.set_len(0); ad.set_len(0);
                    let bp = bits.as_ptr(); let sp = state.selected_bit.as_ptr();
                    let rp = rm.as_mut_ptr(); let ap = ad.as_mut_ptr();
                    let mut ri = 0usize; let mut ai = 0usize;
                    let m = n & !3usize; let mut i = 0usize;
                    while i < m {
                        let b0 = *bp.add(i);     let s0 = *sp.add(i);
                        rp.add(ri).write(i);     ri += (s0 & !b0) as usize;
                        ap.add(ai).write(i);     ai += (b0 & !s0) as usize;
                        let b1 = *bp.add(i + 1); let s1 = *sp.add(i + 1);
                        rp.add(ri).write(i + 1); ri += (s1 & !b1) as usize;
                        ap.add(ai).write(i + 1); ai += (b1 & !s1) as usize;
                        let b2 = *bp.add(i + 2); let s2 = *sp.add(i + 2);
                        rp.add(ri).write(i + 2); ri += (s2 & !b2) as usize;
                        ap.add(ai).write(i + 2); ai += (b2 & !s2) as usize;
                        let b3 = *bp.add(i + 3); let s3 = *sp.add(i + 3);
                        rp.add(ri).write(i + 3); ri += (s3 & !b3) as usize;
                        ap.add(ai).write(i + 3); ai += (b3 & !s3) as usize;
                        i += 4;
                    }
                    while i < n {
                        let b = *bp.add(i); let sb = *sp.add(i);
                        rp.add(ri).write(i); ri += (sb & !b) as usize;
                        ap.add(ai).write(i); ai += (b & !sb) as usize;
                        i += 1;
                    }
                    rm.set_len(ri); ad.set_len(ai);
                for k in (0..SSB_RM.len()).rev() { state.remove_item(SSB_RM[k]); }
                for k in 0..SSB_AD.len() { state.add_item(SSB_AD[k]); }
                }
                return;
            }
            for i in (0..n).rev() { if state.selected_bit[i] && !bits[i] { state.remove_item(i); } }
            for i in 0..n { if bits[i] && !state.selected_bit[i] { state.add_item(i); } }
            return;
        }

        state.selected_bit.clone_from_slice(bits);
        state.total_value = 0;
        state.total_weight = 0;
        for i in 0..n { state.contrib[i] = state.ch.values[i] as i32; }
        for i in 0..n {
            if !bits[i] { continue; }
            state.total_value += state.contrib[i] as i64;
            state.total_weight += state.ch.weights[i];
            let row_ptr = unsafe { state.ch.interaction_values.get_unchecked(i).as_ptr() };
            let contrib_ptr = state.contrib.as_mut_ptr();
            unsafe {
                for k in 0..n {
                    let ck = contrib_ptr.add(k);
                    *ck = (*ck).wrapping_add(*row_ptr.add(k));
                }
            }
        }
    }

    fn build_windows_into(state: &State, k: usize,
                          unused_r: &mut Vec<(usize, f64)>, used_r: &mut Vec<(usize, f64)>,
                          out_unused: &mut Vec<usize>, out_used: &mut Vec<usize>) {
        let n = state.ch.num_items;
        unused_r.clear(); used_r.clear();
        for i in 0..n {
            let (c, w, sel) = unsafe {
                (*state.contrib.get_unchecked(i),
                 *state.ch.weights.get_unchecked(i),
                 *state.selected_bit.get_unchecked(i))
            };
            let r = c as f64 / (w as f64).max(1.0);
            if sel { used_r.push((i, r)); } else { unused_r.push((i, r)); }
        }
        let ku = k.min(unused_r.len());
        if ku > 0 && ku < unused_r.len() { unused_r.select_nth_unstable_by(ku - 1, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)); }
        let ks = k.min(used_r.len());
        if ks > 0 && ks < used_r.len() { used_r.select_nth_unstable_by(ks - 1, |a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)); }
        out_unused.clear(); out_used.clear();
        out_unused.extend(unused_r[..ku].iter().map(|x| x.0));
        out_used.extend(used_r[..ks].iter().map(|x| x.0));
    }

    fn build_windows_into_int(state: &State, k: usize,
                              unused_r: &mut Vec<(usize, i64)>, used_r: &mut Vec<(usize, i64)>,
                              out_unused: &mut Vec<usize>, out_used: &mut Vec<usize>,
                              branchless: bool, precomp: bool, unroll: bool, wide: bool, nm: usize) {
        const LDIV: [i64; 11] = [2520, 2520, 1260, 840, 630, 504, 420, 360, 315, 280, 252];
        let n = state.ch.num_items;
        unused_r.clear(); used_r.clear();
        if branchless {
            unused_r.reserve(n + 8); used_r.reserve(n + 8);
            unsafe {
                let up = unused_r.as_mut_ptr();
                let sp = used_r.as_mut_ptr();
                let mut ui = 0usize; let mut si = 0usize;
                if unroll && precomp {
                    let ctr = state.contrib.as_ptr();
                    let ldv = state.ldiv.as_ptr();
                    let sbt = state.selected_bit.as_ptr();
                    if wide {
                        let m8 = n & !7usize;
                        let mut i = 0usize;
                        while i < m8 {
                            let r0 = *ctr.add(i) as i64 * *ldv.add(i);
                            let s0 = *sbt.add(i);
                            up.add(ui).write((i, r0)); sp.add(si).write((i, r0));
                            ui += (!s0) as usize; si += s0 as usize;
                            let r1 = *ctr.add(i + 1) as i64 * *ldv.add(i + 1);
                            let s1 = *sbt.add(i + 1);
                            up.add(ui).write((i + 1, r1)); sp.add(si).write((i + 1, r1));
                            ui += (!s1) as usize; si += s1 as usize;
                            let r2 = *ctr.add(i + 2) as i64 * *ldv.add(i + 2);
                            let s2 = *sbt.add(i + 2);
                            up.add(ui).write((i + 2, r2)); sp.add(si).write((i + 2, r2));
                            ui += (!s2) as usize; si += s2 as usize;
                            let r3 = *ctr.add(i + 3) as i64 * *ldv.add(i + 3);
                            let s3 = *sbt.add(i + 3);
                            up.add(ui).write((i + 3, r3)); sp.add(si).write((i + 3, r3));
                            ui += (!s3) as usize; si += s3 as usize;
                            let r4 = *ctr.add(i + 4) as i64 * *ldv.add(i + 4);
                            let s4 = *sbt.add(i + 4);
                            up.add(ui).write((i + 4, r4)); sp.add(si).write((i + 4, r4));
                            ui += (!s4) as usize; si += s4 as usize;
                            let r5 = *ctr.add(i + 5) as i64 * *ldv.add(i + 5);
                            let s5 = *sbt.add(i + 5);
                            up.add(ui).write((i + 5, r5)); sp.add(si).write((i + 5, r5));
                            ui += (!s5) as usize; si += s5 as usize;
                            let r6 = *ctr.add(i + 6) as i64 * *ldv.add(i + 6);
                            let s6 = *sbt.add(i + 6);
                            up.add(ui).write((i + 6, r6)); sp.add(si).write((i + 6, r6));
                            ui += (!s6) as usize; si += s6 as usize;
                            let r7 = *ctr.add(i + 7) as i64 * *ldv.add(i + 7);
                            let s7 = *sbt.add(i + 7);
                            up.add(ui).write((i + 7, r7)); sp.add(si).write((i + 7, r7));
                            ui += (!s7) as usize; si += s7 as usize;
                            i += 8;
                        }
                        while i < n {
                            let r = *ctr.add(i) as i64 * *ldv.add(i);
                            let sl = *sbt.add(i);
                            up.add(ui).write((i, r)); sp.add(si).write((i, r));
                            ui += (!sl) as usize; si += sl as usize;
                            i += 1;
                        }
                    } else {
                    let m = n & !3usize;
                    let mut i = 0usize;
                    while i < m {
                        let r0 = *ctr.add(i) as i64 * *ldv.add(i);
                        let s0 = *sbt.add(i);
                        up.add(ui).write((i, r0)); sp.add(si).write((i, r0));
                        ui += (!s0) as usize; si += s0 as usize;
                        let r1 = *ctr.add(i + 1) as i64 * *ldv.add(i + 1);
                        let s1 = *sbt.add(i + 1);
                        up.add(ui).write((i + 1, r1)); sp.add(si).write((i + 1, r1));
                        ui += (!s1) as usize; si += s1 as usize;
                        let r2 = *ctr.add(i + 2) as i64 * *ldv.add(i + 2);
                        let s2 = *sbt.add(i + 2);
                        up.add(ui).write((i + 2, r2)); sp.add(si).write((i + 2, r2));
                        ui += (!s2) as usize; si += s2 as usize;
                        let r3 = *ctr.add(i + 3) as i64 * *ldv.add(i + 3);
                        let s3 = *sbt.add(i + 3);
                        up.add(ui).write((i + 3, r3)); sp.add(si).write((i + 3, r3));
                        ui += (!s3) as usize; si += s3 as usize;
                        i += 4;
                    }
                    while i < n {
                        let r = *ctr.add(i) as i64 * *ldv.add(i);
                        let sl = *sbt.add(i);
                        up.add(ui).write((i, r)); sp.add(si).write((i, r));
                        ui += (!sl) as usize; si += sl as usize;
                        i += 1;
                    }
                    }
                } else {
                for i in 0..n {
                    let c   = *state.contrib.get_unchecked(i);
                    let w   = *state.ch.weights.get_unchecked(i);
                    let sel = *state.selected_bit.get_unchecked(i);
                    let r = if precomp { c as i64 * *state.ldiv.get_unchecked(i) } else { c as i64 * LDIV[(w as usize).clamp(1, 10)] };
                    up.add(ui).write((i, r));
                    sp.add(si).write((i, r));
                    ui += (!sel) as usize;
                    si += sel as usize;
                }
                }
                unused_r.set_len(ui); used_r.set_len(si);
            }
        } else {
        for i in 0..n {
            let (c, w, sel) = unsafe {
                (*state.contrib.get_unchecked(i),
                 *state.ch.weights.get_unchecked(i),
                 *state.selected_bit.get_unchecked(i))
            };
            let r = if precomp { c as i64 * unsafe { *state.ldiv.get_unchecked(i) } } else { c as i64 * LDIV[(w as usize).clamp(1, 10)] };
            if sel { used_r.push((i, r)); } else { unused_r.push((i, r)); }
        }
        }
        let ku = k.min(unused_r.len());
        if ku > 0 && ku < unused_r.len() { unused_r.select_nth_unstable_by(ku - 1, |a, b| b.1.cmp(&a.1)); }
        let ks = k.min(used_r.len());
        if ks > 0 && ks < used_r.len() { used_r.select_nth_unstable_by(ks - 1, |a, b| a.1.cmp(&b.1)); }
        out_unused.clear(); out_used.clear();
        out_unused.extend(unused_r[..ku].iter().map(|x| x.0));
        out_used.extend(used_r[..ks].iter().map(|x| x.0));
    }

    fn build_windows(state: &State, k: usize) -> (Vec<usize>, Vec<usize>) {
        let n = state.ch.num_items;
        let mut unused_r: Vec<(usize, f64)> = Vec::with_capacity(n);
        let mut used_r: Vec<(usize, f64)> = Vec::with_capacity(n);
        for i in 0..n {
            let (c, w, sel) = unsafe {
                (*state.contrib.get_unchecked(i),
                 *state.ch.weights.get_unchecked(i),
                 *state.selected_bit.get_unchecked(i))
            };
            let r = c as f64 / (w as f64).max(1.0);
            if sel { used_r.push((i, r)); } else { unused_r.push((i, r)); }
        }
        let ku = k.min(unused_r.len());
        if ku > 0 && ku < unused_r.len() { unused_r.select_nth_unstable_by(ku - 1, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)); }
        let ks = k.min(used_r.len());
        if ks > 0 && ks < used_r.len() { used_r.select_nth_unstable_by(ks - 1, |a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)); }
        (unused_r[..ku].iter().map(|x| x.0).collect(), used_r[..ks].iter().map(|x| x.0).collect())
    }

    fn local_search_vnd_windowed(state: &mut State, window_k: usize) {
        local_search_vnd_windowed_m(state, window_k, 0)
    }

    fn local_search_vnd_windowed_m(state: &mut State, window_k: usize, neutral_mask: usize) {
        let reuse = neutral_mask & 4 != 0;
        let intkey = neutral_mask & 8 != 0;
        let n_all = state.ch.num_items;
        let mut sc_ur: Vec<(usize, f64)> = if reuse && !intkey { Vec::with_capacity(n_all) } else { Vec::new() };
        let mut sc_us: Vec<(usize, f64)> = if reuse && !intkey { Vec::with_capacity(n_all) } else { Vec::new() };
        let mut sc_ui: Vec<(usize, i64)> = if reuse && intkey { Vec::with_capacity(n_all) } else { Vec::new() };
        let mut sc_si: Vec<(usize, i64)> = if reuse && intkey { Vec::with_capacity(n_all) } else { Vec::new() };
        let mut sc_bu: Vec<usize> = if reuse { Vec::with_capacity(n_all) } else { Vec::new() };
        let mut sc_wu: Vec<usize> = if reuse { Vec::with_capacity(n_all) } else { Vec::new() };
        let mut cand_live: Vec<usize> = Vec::with_capacity(n_all);
        let reuse_fits = neutral_mask & (1usize << 28) != 0;
        let mut fits_buf: Vec<usize> = Vec::with_capacity(n_all);
        for _ in 0..80 {
            let (best_unused, worst_used) = if reuse {
                if intkey {
                    build_windows_into_int(state, window_k, &mut sc_ui, &mut sc_si, &mut sc_bu, &mut sc_wu, neutral_mask & 32 != 0, neutral_mask & 128 != 0, neutral_mask & 1024 != 0, neutral_mask & (1usize << 20) != 0, neutral_mask);
                } else {
                    build_windows_into(state, window_k, &mut sc_ur, &mut sc_us, &mut sc_bu, &mut sc_wu);
                }
                (Vec::new(), Vec::new())
            } else {
                build_windows(state, window_k)
            };
            let (best_unused, worst_used): (&[usize], &[usize]) = if reuse {
                (&sc_bu, &sc_wu)
            } else {
                (&best_unused, &worst_used)
            };
            let slack = state.slack();
            if slack > 0 {
                let mut ba: Option<(usize, i32)> = None;
                for &c in best_unused {
                    if state.ch.weights[c] > slack { continue; }
                    let d = state.contrib[c];
                    if d > 0 && ba.map_or(true, |(_, bd)| d > bd) { ba = Some((c, d)); }
                }
                if let Some((c, _)) = ba { state.add_item(c); continue; }
            }
            {
                let mut bs: Option<(usize, usize, i32)> = None;
                let mut cmax = [i32::MIN; 11];
                for &c in best_unused {
                    let cw = (state.ch.weights[c] as usize).clamp(1, 10);
                    if state.contrib[c] > cmax[cw] { cmax[cw] = state.contrib[c]; }
                }
                for w in 1..=10 { if cmax[w - 1] > cmax[w] { cmax[w] = cmax[w - 1]; } }
                let mut rmin = [i32::MAX; 12];
                for &rm in worst_used {
                    let rw = (state.ch.weights[rm] as usize).clamp(1, 10);
                    if state.contrib[rm] < rmin[rw] { rmin[rw] = state.contrib[rm]; }
                }
                for w in (1..=10).rev() { if rmin[w + 1] < rmin[w] { rmin[w] = rmin[w + 1]; } }
                let slack_c = state.slack() as i32;
                cand_live.clear();
                for &c in best_unused {
                    let wc = state.ch.weights[c] as i32;
                    let lo = (wc - slack_c).max(1).min(10) as usize;
                    if state.contrib[c] > rmin[lo] { cand_live.push(c); }
                }
                let prune = neutral_mask & 64 != 0;
                let prune2 = neutral_mask & 512 != 0;
                let mut bdv: i64 = 0;
                if neutral_mask & 8192 != 0 {
                    let mut bc: usize = usize::MAX; let mut brm: usize = 0;
                    unsafe {
                        let wts = state.ch.weights.as_ptr();
                        let ctr = state.contrib.as_ptr();
                        for &rm in worst_used {
                            let c_rm = *ctr.add(rm);
                            let max_w = *wts.add(rm) + state.slack();
                            let cm = cmax[(max_w as usize).clamp(1, 10)];
                            if cm <= c_rm { continue; }
                            if prune && (cm as i64 - c_rm as i64) <= bdv { continue; }
                            let row_rm = state.ch.interaction_values.get_unchecked(rm).as_ptr();
                            for &c in &cand_live {
                                let feas = *wts.add(c) <= max_w;
                                let d = *ctr.add(c) - c_rm - *row_rm.add(c);
                                let take = feas & ((d as i64) > bdv);
                                bdv = if take { d as i64 } else { bdv };
                                bc  = if take { c } else { bc };
                                brm = if take { rm } else { brm };
                            }
                        }
                    }
                    if bc != usize::MAX { bs = Some((bc, brm, bdv as i32)); }
                } else {
                for &rm in worst_used {
                    let max_w = state.ch.weights[rm] + state.slack();
                    let cm = cmax[(max_w as usize).clamp(1, 10)];
                    if cm <= state.contrib[rm] { continue; }
                    if prune && (cm as i64 - state.contrib[rm] as i64) <= bdv { continue; }
                    let row_rm = unsafe { state.ch.interaction_values.get_unchecked(rm) };
                    for &c in &cand_live {
                        if state.ch.weights[c] > max_w { continue; }
                        if prune2 && (state.contrib[c] as i64 - state.contrib[rm] as i64) <= bdv { continue; }
                        let d = state.contrib[c] - state.contrib[rm] - unsafe { *row_rm.get_unchecked(c) };
                        if d > 0 && bs.map_or(true, |(_, _, bd)| d > bd) { bs = Some((c, rm, d)); bdv = d as i64; }
                    }
                }
                }
                if let Some((c, rm, _)) = bs { state.replace_item(rm, c); continue; }
            }
            let slack = state.slack();
            if slack >= 2 {
                let fits: Vec<usize> = if reuse_fits { Vec::new() } else {
                    best_unused.iter().copied().filter(|&i| state.ch.weights[i] < slack).collect()
                };
                if reuse_fits {
                    fits_buf.clear();
                    for &i in best_unused { if state.ch.weights[i] < slack { fits_buf.push(i); } }
                }
                let fits: &[usize] = if reuse_fits { &fits_buf } else { &fits };
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
            break;
        }
    }

    fn perturb_by_strategy(state: &mut State, strength: usize, stall_count: usize, strategy: usize, rng: &mut Rng, hp: &Hparams) {
        let n = state.ch.num_items;
        let selected_len = state.selected_bit.iter().filter(|&&b| b).count();
        if selected_len == 0 { return; }
        let mut removal_candidates: Vec<(usize, i64)> = Vec::with_capacity(selected_len);
        match strategy {
            0 => {
                for i in 0..n { if state.selected_bit[i] { removal_candidates.push((i, state.contrib[i] as i64)); } }
                removal_candidates.sort_unstable_by_key(|&(_, c)| c);
            },
            1 => {
                for i in 0..n { if state.selected_bit[i] { removal_candidates.push((i, -(state.ch.weights[i] as i64))); } }
                removal_candidates.sort_unstable_by_key(|&(_, w)| w);
            },
            2 => {
                for i in 0..n {
                    if state.selected_bit[i] {
                        let syn = state.contrib[i] as i64 - state.ch.values[i] as i64;
                        removal_candidates.push((i, syn));
                    }
                }
                removal_candidates.sort_unstable_by_key(|&(_, s)| s);
            },
            3 => {
                for i in 0..n {
                    if state.selected_bit[i] {
                        let w = (state.ch.weights[i] as i64).max(1);
                        removal_candidates.push((i, dw(state.contrib[i] as i64 * 1000, w)));
                    }
                }
                removal_candidates.sort_unstable_by_key(|&(_, s)| s);
            },
            4 => {
                for i in 0..n {
                    if state.selected_bit[i] {
                        let w = (state.ch.weights[i] as i64).max(1);
                        let density = dw(state.contrib[i] as i64 * 100, w);
                        removal_candidates.push((i, state.ch.weights[i] as i64 - density));
                    }
                }
                removal_candidates.sort_unstable_by_key(|&(_, s)| s);
            },
            5 => {
                for i in 0..n {
                    if state.selected_bit[i] {
                        let w = (state.ch.weights[i] as i64).max(1);
                        removal_candidates.push((i, (state.contrib[i] as i64 * 10000) / (w * w)));
                    }
                }
                removal_candidates.sort_unstable_by_key(|&(_, s)| s);
            },
            6 => {
                let seed_idx = rng.next_usize(selected_len);
                let mut seen = 0usize;
                let mut seed = 0usize;
                for i in 0..n {
                    if state.selected_bit[i] {
                        if seen == seed_idx { seed = i; break; }
                        seen += 1;
                    }
                }
                for i in 0..n {
                    if state.selected_bit[i] {
                        if i == seed { removal_candidates.push((i, i64::MIN)); }
                        else { removal_candidates.push((i, -(state.ch.interaction_values[i][seed] as i64))); }
                    }
                }
                removal_candidates.sort_unstable_by_key(|&(_, s)| s);
            },
            _ => {
                for i in 0..n { if state.selected_bit[i] { removal_candidates.push((i, -(state.contrib[i] as i64))); } }
                removal_candidates.sort_unstable_by_key(|&(_, s)| s);
            }
        }
        let base_remove = (selected_len / hp.perturb_base_frac).max(2);
        let adaptive_mult = 1 + (stall_count / 2);
        let n_remove = (base_remove * adaptive_mult).min(strength).min(selected_len * 2 / hp.perturb_max_frac);
        for j in 0..n_remove { if j < removal_candidates.len() { state.remove_item(removal_candidates[j].0); } }
    }

    fn greedy_reconstruct(state: &mut State, strategy: usize, static_synergy: &[i64]) {
        let n = state.ch.num_items; let cap = state.ch.max_weight;
        let mut candidates: Vec<usize> = (0..n).filter(|&i| !state.selected_bit[i]).collect();
        match strategy % 4 {
            0 => candidates.sort_unstable_by_key(|&i| -state.contrib[i]),
            1 => candidates.sort_unstable_by(|&a, &b| state.ch.weights[a].cmp(&state.ch.weights[b]).then(state.contrib[b].cmp(&state.contrib[a]))),
            2 => candidates.sort_unstable_by_key(|&i| -(static_synergy[i] + state.contrib[i] as i64 / 10)),
            _ => {
                let mut keys = vec![0i64; n];
                for &i in &candidates {
                    let w = (state.ch.weights[i] as i64).max(1);
                    keys[i] = -dw(state.contrib[i] as i64 * 100, w);
                }
                candidates.sort_unstable_by_key(|&i| keys[i]);
            },
        }
        for &i in &candidates { if state.total_weight + state.ch.weights[i] <= cap { state.add_item(i); } }
    }

    fn greedy_reconstruct_blocked(state: &mut State, strategy: usize, static_synergy: &[i64], blocked: &[bool]) {
        let n = state.ch.num_items; let cap = state.ch.max_weight;
        let mut candidates: Vec<usize> = (0..n).filter(|&i| !state.selected_bit[i] && !blocked[i]).collect();
        match strategy % 4 {
            0 => candidates.sort_unstable_by_key(|&i| -state.contrib[i]),
            1 => candidates.sort_unstable_by(|&a, &b| state.ch.weights[a].cmp(&state.ch.weights[b]).then(state.contrib[b].cmp(&state.contrib[a]))),
            2 => candidates.sort_unstable_by_key(|&i| -(static_synergy[i] + state.contrib[i] as i64 / 10)),
            _ => {
                let mut keys = vec![0i64; n];
                for &i in &candidates {
                    let w = (state.ch.weights[i] as i64).max(1);
                    keys[i] = -dw(state.contrib[i] as i64 * 100, w);
                }
                candidates.sort_unstable_by_key(|&i| keys[i]);
            },
        }
        for &i in &candidates { if state.total_weight + state.ch.weights[i] <= cap { state.add_item(i); } }
    }

    fn vnd_dispatch(state: &mut State, hp: &Hparams) {
        if hp.window_k < state.ch.num_items { local_search_vnd_windowed_m(state, hp.window_k, hp.neutral_mask); }
        else { ils_vnd(state, hp); }
    }

    fn refine_with_memo(
        state: &mut State,
        hp: &Hparams,
        core_half: usize,
        zobrist_table: &[u64],
        cache: &mut Vec<(u64, Vec<bool>, SolState)>,
        cache_index: &mut DetMap,
        replacement: usize,
    ) {
        let hash = hash_bits_with_table(&state.selected_bit, zobrist_table);
        if let Some(slot) = cache_index.get(&hash).copied() {
            if slot < cache.len()
                && cache[slot].0 == hash
                && cache[slot].1 == state.selected_bit
            {
                state.restore_solution(&cache[slot].2);
                return;
            }
        }

        let input_bits = state.selected_bit.clone();
        dp_refinement_hp(state, core_half, hp);
        vnd_dispatch(state, hp);
        let entry = (hash, input_bits, state.clone_solution());
        let slot = if cache.len() < 128 {
            let slot = cache.len();
            cache.push(entry);
            slot
        } else {
            let slot = replacement % 128;
            let old_hash = cache[slot].0;
            if cache_index.get(&old_hash).copied() == Some(slot) {
                cache_index.remove(&old_hash);
            }
            cache[slot] = entry;
            slot
        };
        cache_index.insert(hash, slot);
    }

    fn try_add_elite_t40(pool: &mut Vec<Vec<bool>>, bits: &[bool]) {
        let hamming_ok = pool.iter().all(|e| e.iter().zip(bits.iter()).filter(|(&a, &b)| a != b).count() > 20);
        if hamming_ok { if pool.len() >= 4 { pool.remove(0); } pool.push(bits.to_vec()); }
    }

    fn path_relink_select_action(state: &State, to_add: &[usize], to_remove: &[usize]) -> Option<(bool, usize)> {
        let mut best_delta = i64::MIN;
        let mut best_action = None;
        for (idx, &item) in to_add.iter().enumerate() {
            if state.total_weight + state.ch.weights[item] <= state.ch.max_weight {
                let delta = state.contrib[item] as i64;
                if delta > best_delta {
                    best_delta = delta;
                    best_action = Some((true, idx));
                }
            }
        }
        for (idx, &item) in to_remove.iter().enumerate() {
            let delta = -(state.contrib[item] as i64);
            if delta > best_delta {
                best_delta = delta;
                best_action = Some((false, idx));
            }
        }
        best_action
    }

    #[inline(always)]
    fn path_relink_transition(
        state: &mut State,
        action: (bool, usize),
        to_add: &mut Vec<usize>,
        to_remove: &mut Vec<usize>,
        add_pos: &mut [usize],
        remove_pos: &mut [usize],
    ) -> Option<(bool, usize)> {
        let (is_add, position) = action;
        let item;
        if is_add {
            item = to_add[position];
            let moved = *to_add.last().unwrap();
            to_add.swap_remove(position);
            add_pos[item] = usize::MAX;
            if position < to_add.len() { add_pos[moved] = position; }
            state.total_value += state.contrib[item] as i64;
            state.total_weight += state.ch.weights[item];
        } else {
            item = to_remove[position];
            let moved = *to_remove.last().unwrap();
            to_remove.swap_remove(position);
            remove_pos[item] = usize::MAX;
            if position < to_remove.len() { remove_pos[moved] = position; }
            state.total_value -= state.contrib[item] as i64;
            state.total_weight -= state.ch.weights[item];
        }

        let n = state.ch.num_items;
        let post_weight = state.total_weight;
        let cap = state.ch.max_weight;
        let weights = &state.ch.weights;
        let row_ptr = unsafe { state.ch.interaction_values.get_unchecked(item).as_ptr() };
        let contrib_ptr = state.contrib.as_mut_ptr();
        let mut best_delta = i64::MIN;
        let mut best_action: Option<(bool, usize)> = None;
        let mut consider = |k: usize, contribution: i32| {
            let position = add_pos[k];
            if position != usize::MAX && post_weight + weights[k] <= cap {
                let delta = contribution as i64;
                let better = delta > best_delta || (delta == best_delta && match best_action {
                    None => true,
                    Some((best_is_add, best_position)) => !best_is_add || position < best_position,
                });
                if better {
                    best_delta = delta;
                    best_action = Some((true, position));
                }
            }

            let position = remove_pos[k];
            if position != usize::MAX {
                let delta = -(contribution as i64);
                let better = delta > best_delta || (delta == best_delta && match best_action {
                    None => true,
                    Some((best_is_add, best_position)) => !best_is_add && position < best_position,
                });
                if better {
                    best_delta = delta;
                    best_action = Some((false, position));
                }
            }
        };

        if is_add {
            unsafe {
                for k in 0..n {
                    let contribution = (*contrib_ptr.add(k)).wrapping_add(*row_ptr.add(k));
                    *contrib_ptr.add(k) = contribution;
                    consider(k, contribution);
                }
            }
            state.selected_bit[item] = true;
        } else {
            unsafe {
                for k in 0..n {
                    let contribution = (*contrib_ptr.add(k)).wrapping_sub(*row_ptr.add(k));
                    *contrib_ptr.add(k) = contribution;
                    consider(k, contribution);
                }
            }
            state.selected_bit[item] = false;
        }

        best_action
    }

    fn path_relink_t40(challenge: &Challenge, source_bits: &[bool], guide_bits: &[bool], hp: &Hparams, ch_dp: usize) -> Option<(i64, Vec<bool>)> {
        let n = challenge.num_items;
        let mut state = State::new_empty(challenge);
        for i in 0..n { if source_bits[i] { state.add_item(i); } }
        let mut to_add: Vec<usize> = (0..n).filter(|&i| guide_bits[i] && !source_bits[i]).collect();
        let mut to_remove: Vec<usize> = (0..n).filter(|&i| source_bits[i] && !guide_bits[i]).collect();
        let total_moves = to_add.len() + to_remove.len();
        if total_moves == 0 { return None; }
        let mut add_pos = vec![usize::MAX; n];
        let mut remove_pos = vec![usize::MAX; n];
        for (position, &item) in to_add.iter().enumerate() { add_pos[item] = position; }
        for (position, &item) in to_remove.iter().enumerate() { remove_pos[item] = position; }
        let checkpoint_interval = (total_moves / 4).max(3);
        let cap = challenge.max_weight;
        let mut best_pr = state.clone_solution();
        let mut tmp = State::new_empty(challenge);
        let mut move_count = 0usize;
        let mut next_action = path_relink_select_action(&state, &to_add, &to_remove);
        let mut action_cached = true;
        while !to_add.is_empty() || !to_remove.is_empty() {
            if !action_cached {
                next_action = path_relink_select_action(&state, &to_add, &to_remove);
            }
            let action = match next_action {
                Some(action) => action,
                None => break,
            };
            next_action = path_relink_transition(
                &mut state,
                action,
                &mut to_add,
                &mut to_remove,
                &mut add_pos,
                &mut remove_pos,
            );
            action_cached = true;
            move_count += 1;
            if state.total_weight <= cap && state.total_value > best_pr.value {
                best_pr = state.clone_solution();
            }
            if move_count % checkpoint_interval == 0 && state.total_weight <= cap {
                tmp.selected_bit.clone_from(&state.selected_bit);
                tmp.contrib.clone_from(&state.contrib);
                tmp.total_value = state.total_value;
                tmp.total_weight = state.total_weight;
                local_search_vnd_fast(&mut tmp, (hp.exact_mask >> 10) & 3);
                dp_refinement_hp(&mut tmp, ch_dp, hp);
                if tmp.total_value > best_pr.value { best_pr = tmp.clone_solution(); }
                action_cached = false;
            }
        }
        let mut final_st = State::new_empty(challenge);
        final_st.restore_solution(&best_pr);
        loop {
            let v_before = final_st.total_value;
            local_search_vnd_windowed_m(&mut final_st, hp.window_k, hp.neutral_mask);
            dp_refinement_hp(&mut final_st, hp.core_half_dp, hp);
            if final_st.total_value <= v_before { break; }
        }
        if final_st.total_value > best_pr.value { best_pr = final_st.clone_solution(); }
        if best_pr.value > 0 { Some((best_pr.value, best_pr.bits)) } else { None }
    }

    fn xr_archive_push(pool: &mut Vec<(i64, Vec<bool>)>, val: i64, bits: &[bool], cap: usize) {
        if cap == 0 { return; }
        for e in pool.iter_mut() {
            let d = e.1.iter().zip(bits.iter()).filter(|(&a, &b)| a != b).count();
            if d <= 20 {
                if val > e.0 { e.0 = val; e.1.clear(); e.1.extend_from_slice(bits); }
                return;
            }
        }
        pool.push((val, bits.to_vec()));
        while pool.len() > cap {
            let mut worst = 0usize;
            for i in 1..pool.len() { if pool[i].0 < pool[worst].0 { worst = i; } }
            pool.remove(worst);
        }
    }

    fn run_bp_seeded_instance(challenge: &Challenge, hp: &Hparams, bp_out: &mut Vec<i32>, bcross_out: &mut i32, lam_out: &mut (f64, f64)) -> SolState {
        let inst = challenge_to_p1_m(challenge, hp.neutral_mask);
        let ext_pm = if hp.core_work > 0 { hp.core_lo as i32 } else { 0 };
        let mut qr = run_bp_algorithm_x(&inst, hp.n_lambda_values, hp.exact_mask,
                                      if hp.neutral_mask & (1usize << 19) != 0 { Some(challenge) } else { None }, ext_pm, hp.core_work > 0);
        *bcross_out = qr.bcross; *lam_out = qr.lam_c;
        bp_out.clear(); bp_out.append(&mut qr.bp);
        let bp_items = &qr.results[0].selected_items;

        let mut state = State::new_empty(challenge);
        for &i in bp_items {
            if state.total_weight + challenge.weights[i] <= challenge.max_weight {
                state.add_item(i);
            }
        }
        dp_refinement_hp(&mut state, hp.core_half_dp, hp);
        vnd_dispatch(&mut state, hp);
        state.clone_solution()
    }

    fn build_deterministic_seed_population(challenge: &Challenge, hp: &Hparams, ch: usize) -> (Vec<SolState>, Vec<SolState>) {
        let n = challenge.num_items;
        let mut greedy_population: Vec<SolState> = Vec::with_capacity(4);

        let n_greedy = if n <= 1200 { 4 } else { 3 };
        let shared_rows: Option<Vec<i64>> = if hp.neutral_mask & 65536 != 0 { Some(row_sums_of(challenge)) } else { None };
        let rs: Option<&[i64]> = shared_rows.as_deref();
        for variant in 0..n_greedy {
            let mut st = State::new_empty(challenge);
            match variant {
                0 => build_greedy_density_m(&mut st, (hp.neutral_mask >> 14) & 3),
                1 => build_greedy_value(&mut st),
                2 => build_greedy_synergy_weight_s(&mut st, rs),
                _ => build_greedy_hub_s(&mut st, rs),
            }
            dp_refinement_hp(&mut st, ch, hp);
            if hp.use_heavy_polish { local_search_vnd_heavy(&mut st, (hp.exact_mask >> 10) & 3); } else { vnd_dispatch(&mut st, hp); }
            greedy_population.push(st.clone_solution());
        }

        let mut hub_population: Vec<SolState> = Vec::with_capacity(4);
        if hp.use_hub_pair {
            let mut hub_pairs: Vec<(i32, usize, usize)> = Vec::new();
            let cap = challenge.max_weight;
            if hp.hub_top > 0 {
                let want = hp.hub_top;
                let mut thresh = i32::MIN;
                let prefilter = hp.exact_mask & 256 != 0;
                let mut hp_sel: Vec<usize> = if prefilter { vec![0usize; n + 4] } else { Vec::new() };
                for i in 0..n {
                    let wi = challenge.weights[i];
                    let row = &challenge.interaction_values[i];
                    if prefilter {
                        let t0 = thresh;
                        let mut cnt = 0usize;
                        unsafe {
                            let rp = row.as_ptr();
                            let sp = hp_sel.as_mut_ptr();
                            let mut j = i + 1;
                            let m = j + ((n - j) & !3usize);
                            while j < m {
                                sp.add(cnt).write(j);     cnt += (*rp.add(j)     > t0) as usize;
                                sp.add(cnt).write(j + 1); cnt += (*rp.add(j + 1) > t0) as usize;
                                sp.add(cnt).write(j + 2); cnt += (*rp.add(j + 2) > t0) as usize;
                                sp.add(cnt).write(j + 3); cnt += (*rp.add(j + 3) > t0) as usize;
                                j += 4;
                            }
                            while j < n {
                                sp.add(cnt).write(j); cnt += (*rp.add(j) > t0) as usize;
                                j += 1;
                            }
                        }
                        for k in 0..cnt {
                            let j = hp_sel[k];
                            let v = row[j];
                            if v <= thresh { continue; }
                            if wi + challenge.weights[j] > cap { continue; }
                            hub_pairs.push((v, i, j));
                            if hub_pairs.len() >= 4 * want {
                                hub_pairs.sort_unstable_by_key(|&(s, _, _)| std::cmp::Reverse(s));
                                hub_pairs.truncate(want);
                                thresh = hub_pairs[want - 1].0 - 1;
                            }
                        }
                        continue;
                    }
                    for j in (i+1)..n {
                        let v = row[j];
                        if v <= thresh { continue; }
                        if wi + challenge.weights[j] > cap { continue; }
                        hub_pairs.push((v, i, j));
                        if hub_pairs.len() >= 4 * want {
                            hub_pairs.sort_unstable_by_key(|&(s, _, _)| std::cmp::Reverse(s));
                            hub_pairs.truncate(want);
                            thresh = hub_pairs[want - 1].0 - 1;
                        }
                    }
                }
                hub_pairs.sort_unstable_by_key(|&(s, _, _)| std::cmp::Reverse(s));
                if hub_pairs.len() > want { hub_pairs.truncate(want); }
            } else {
            for i in 0..n {
                for j in (i+1)..n {
                    if challenge.weights[i] + challenge.weights[j] <= cap {
                        hub_pairs.push((challenge.interaction_values[i][j], i, j));
                    }
                }
            }
            hub_pairs.sort_unstable_by_key(|&(s, _, _)| std::cmp::Reverse(s));
            }
            for k in 0..4 {
                let mut st = State::new_empty(challenge);
                build_hub_pair_kth_from_pairs(&mut st, &hub_pairs, k, hp.exact_mask & (1usize << 37) != 0);
                dp_refinement_hp(&mut st, ch, hp);
                vnd_dispatch(&mut st, hp);
                hub_population.push(st.clone_solution());
            }
        }

        (greedy_population, hub_population)
    }

    fn run_one_instance_with_value(challenge: &Challenge, hp: &Hparams, rng_offset: usize, archive: &[SolState]) -> (Solution, i64, SolState) {
        let ch = hp.core_half_dp;
        let deterministic = build_deterministic_seed_population(challenge, hp, ch);
        run_one_instance_with_seed_cache_value(challenge, hp, rng_offset, &deterministic, archive, None)
    }

    fn run_one_instance_with_seed_cache_value(challenge: &Challenge, hp: &Hparams, rng_offset: usize, deterministic: &(Vec<SolState>, Vec<SolState>), archive: &[SolState], shared: Option<(&[u64], &[i64])>) -> (Solution, i64, SolState) {
        let n = challenge.num_items;
        let mut rng = Rng::from_seed(&challenge.seed);
        for _ in 0..rng_offset * 100 { rng.next_u32(); }
        let ch = hp.core_half_dp;
        let own_z: Vec<u64>;
        let own_s: Vec<i64>;
        let (zobrist_table, static_synergy): (&[u64], &[i64]) = match shared {
            Some((z, sy)) => (z, sy),
            None => {
                own_z = build_zobrist_table(n);
                let prefix_len = n.min(100);
                own_s = (0..n)
                    .map(|i| challenge.interaction_values[i].iter().take(prefix_len).map(|&v| v as i64).sum())
                    .collect();
                (&own_z, &own_s)
            }
        };

        let mut population: Vec<SolState> = Vec::with_capacity(16);
        population.extend(deterministic.0.iter().cloned());

        let ctor_is_noop = challenge.values.iter().all(|&v| v == 0);
        let mut rand_member: Option<SolState> = None;
        for mode in 4..(4 + hp.n_random_starts) {
            if ctor_is_noop {
                if let Some(m0) = rand_member.as_ref() {
                    population.push(m0.clone());
                    continue;
                }
            }
            let mut st = State::new_empty(challenge);
            let m = if mode < 6 { mode } else { mode - 2 };
            construct_forward_incremental(&mut st, m, &mut rng);
            dp_refinement_hp(&mut st, ch, hp);
            vnd_dispatch(&mut st, hp);
            let sol = st.clone_solution();
            if ctor_is_noop { rand_member = Some(sol.clone()); }
            population.push(sol);
        }

        population.extend(deterministic.1.iter().cloned());

        for s in archive.iter() { population.push(s.clone()); }

        dedup_population_t8_w(&mut population, zobrist_table, hp.dedup_fast > 0, hp.exact_mask & 2 != 0, hp.neutral_mask & 256 != 0, hp.exact_mask & 64);

        let mut state = State::new_empty(challenge);
        for _ in 0..hp.n_crossover_gen {
            let child_bits = crossover_frequency_m(&population, challenge, &mut rng, hp.neutral_mask);
            set_state_from_bits(&mut state, &child_bits);
            dp_refinement_hp(&mut state, ch, hp); vnd_dispatch(&mut state, hp);
            population.push(state.clone_solution());
            if population.len() >= 2 {
                dedup_population_t8_w(&mut population, zobrist_table, hp.dedup_fast > 0, hp.exact_mask & 2 != 0, hp.neutral_mask & 256 != 0, hp.exact_mask & 64);
                let (a, b) = select_diverse_parent_pair_m(&population, &mut rng, hp.neutral_mask);
                let child_bits = crossover_uniform(&population[a], &population[b], challenge, &mut rng);
                set_state_from_bits(&mut state, &child_bits);
                dp_refinement_hp(&mut state, ch, hp); vnd_dispatch(&mut state, hp);
                population.push(state.clone_solution());
            }
            dedup_population_t8_w(&mut population, zobrist_table, hp.dedup_fast > 0, hp.exact_mask & 2 != 0, hp.neutral_mask & 256 != 0, hp.exact_mask & 64);
        }

        if hp.sa_rounds > 0 {
            for pi in 0..hp.n_sa_members.min(population.len()) {
                state.restore_solution(&population[pi]);
                simulated_annealing(&mut state, &mut rng, hp.sa_rounds, hp.sa_iter);
                vnd_dispatch(&mut state, hp);
                let sol = state.clone_solution();
                if sol.value > population[pi].value { population.push(sol); }
            }
            dedup_population_t8_w(&mut population, zobrist_table, hp.dedup_fast > 0, hp.exact_mask & 2 != 0, hp.neutral_mask & 256 != 0, hp.exact_mask & 64);
        }

        state.restore_solution(&population[0]);
        let mut best_val = state.total_value;
        let mut best_snapshot = state.clone_solution();

        let mut tabu_hashes: Vec<u64> = Vec::with_capacity(128);
        let compute_hash = |bits: &[bool]| -> u64 { hash_bits_with_table(bits, zobrist_table) };
        tabu_hashes.push(compute_hash(&state.selected_bit));

        let mut stall_count = 0;
        let mut elite_pool: Vec<Vec<bool>> = Vec::new();
        try_add_elite_t40(&mut elite_pool, &state.selected_bit);
        let mut refinement_cache: Vec<(u64, Vec<bool>, SolState)> = Vec::with_capacity(128);
        let mut refinement_cache_index: DetMap = DetMap::with_capacity_and_hasher(128, Default::default());

        for round in 0..hp.ils_rounds {
            let snap = state.clone_solution();

            refine_with_memo(
                &mut state,
                hp,
                ch,
                zobrist_table,
                &mut refinement_cache,
                &mut refinement_cache_index,
                round,
            );

            if state.total_value > best_val {
                best_val = state.total_value;
                best_snapshot.bits.clone_from(&state.selected_bit);
                best_snapshot.contrib.clone_from(&state.contrib);
                best_snapshot.value = state.total_value;
                best_snapshot.weight = state.total_weight;
                stall_count = 0;
                try_add_elite_t40(&mut elite_pool, &state.selected_bit);
            }

            if state.total_value <= snap.value {
                state.restore_solution(&snap);
                stall_count += 1;

                if hp.ils_stall_stop > 0 && stall_count >= hp.ils_stall_stop { break; }

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
                greedy_reconstruct(&mut state, strategy, static_synergy);
                vnd_dispatch(&mut state, hp);

                let h = compute_hash(&state.selected_bit);
                if tabu_hashes.contains(&h) {
                    let extra_strength = 10 + round / 3;
                    perturb_by_strategy(&mut state, extra_strength, stall_count + 3, 6, &mut rng, hp);
                    greedy_reconstruct(&mut state, 0, static_synergy);
                    vnd_dispatch(&mut state, hp);
                }
                let h2 = compute_hash(&state.selected_bit);
                if tabu_hashes.len() < 128 { tabu_hashes.push(h2); }
                else { tabu_hashes[round % 128] = h2; }

                if state.total_value > best_val {
                    best_val = state.total_value;
                    best_snapshot.bits.clone_from(&state.selected_bit);
                    best_snapshot.contrib.clone_from(&state.contrib);
                    best_snapshot.value = state.total_value;
                    best_snapshot.weight = state.total_weight;
                    stall_count = 0;
                }
            } else {
                stall_count = 0;
                let h = compute_hash(&state.selected_bit);
                if tabu_hashes.len() < 128 { tabu_hashes.push(h); }
            }
        }

        let mut final_state = State::new_empty(challenge);
        final_state.restore_solution(&best_snapshot);

        if hp.use_heavy_polish {
            loop {
                let v_before = final_state.total_value;
                local_search_vnd_heavy(&mut final_state, (hp.exact_mask >> 10) & 3);
                dp_refinement_hp(&mut final_state, ch, hp);
                if final_state.total_value <= v_before { break; }
            }
        } else {
            loop {
                let v_before = final_state.total_value;
                local_search_vnd_windowed_m(&mut final_state, hp.window_k, hp.neutral_mask);
                dp_refinement_hp(&mut final_state, ch, hp);
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
            let snap_out = final_state.clone_solution();
            (Solution { items: final_state.selected_items() }, final_state.total_value, snap_out)
        } else {
            let items = (0..n).filter(|&i| best_snapshot.bits[i]).collect();
            (Solution { items }, best_val, best_snapshot)
        }
    } 

    const CB_SC: i64 = 64;
    const CB_INF: i64 = i64::MAX / 8;

    struct CbFlow {
        eu: Vec<u32>, ev: Vec<u32>, ec: Vec<i64>, n: usize,
        start: Vec<u32>, to: Vec<u32>, cap: Vec<i64>, rev: Vec<u32>,
        level: Vec<i32>, it: Vec<u32>, q: Vec<u32>, deg: Vec<u32>, sa: Vec<i32>,
    }

    impl CbFlow {
        fn new() -> Self {
            CbFlow { eu: Vec::new(), ev: Vec::new(), ec: Vec::new(), n: 0, start: Vec::new(), to: Vec::new(),
                     cap: Vec::new(), rev: Vec::new(), level: Vec::new(), it: Vec::new(), q: Vec::new(), deg: Vec::new(), sa: Vec::new() }
        }
        fn reset(&mut self, n: usize) { self.eu.clear(); self.ev.clear(); self.ec.clear(); self.n = n; }
        #[inline]
        fn add(&mut self, u: usize, v: usize, c: i64) { self.eu.push(u as u32); self.ev.push(v as u32); self.ec.push(c); }
        fn build(&mut self, t: usize) {
            let n = self.n; let m = self.eu.len();
            self.sa.clear(); self.sa.resize(n, -1);
            self.deg.clear(); self.deg.resize(n + 1, 0);
            for k in 0..m { self.deg[self.eu[k] as usize + 1] += 1; self.deg[self.ev[k] as usize + 1] += 1; }
            for i in 0..n { self.deg[i + 1] += self.deg[i]; }
            self.start.clear(); self.start.extend_from_slice(&self.deg);
            self.to.clear(); self.to.resize(2 * m, 0);
            self.cap.clear(); self.cap.resize(2 * m, 0);
            self.rev.clear(); self.rev.resize(2 * m, 0);
            for k in 0..m {
                let u = self.eu[k] as usize; let v = self.ev[k] as usize;
                let a = self.deg[u] as usize; self.deg[u] += 1;
                let b = self.deg[v] as usize; self.deg[v] += 1;
                self.to[a] = v as u32; self.cap[a] = self.ec[k]; self.rev[a] = b as u32;
                if v == t { self.sa[u] = a as i32; }
                self.to[b] = u as u32; self.rev[b] = a as u32;
            }
        }
        fn bfs(&mut self, s: usize, t: usize) -> bool {
            for x in self.level.iter_mut() { *x = -1; }
            self.level[s] = 0; self.q.clear(); self.q.push(s as u32);
            let mut qi = 0usize;
            while qi < self.q.len() {
                let u = self.q[qi] as usize; qi += 1;
                let lu = self.level[u] + 1;
                for e in self.start[u] as usize..self.start[u + 1] as usize {
                    let v = self.to[e] as usize;
                    if self.cap[e] > 0 && self.level[v] < 0 { self.level[v] = lu; self.q.push(v as u32); }
                }
            }
            self.level[t] >= 0
        }
        fn push(&mut self, u: usize, t: usize, f: i64) -> i64 {
            if u == t { return f; }
            let mut pushed = 0i64;
            let end = self.start[u + 1];
            while self.it[u] < end {
                let e = self.it[u] as usize; let v = self.to[e] as usize;
                if self.cap[e] > 0 && self.level[v] == self.level[u] + 1 {
                    let r = f - pushed; let c = self.cap[e];
                    let d = self.push(v, t, if c < r { c } else { r });
                    if d > 0 {
                        self.cap[e] -= d; let rv = self.rev[e] as usize; self.cap[rv] += d; pushed += d;
                        if pushed == f { return pushed; }
                    }
                }
                self.it[u] += 1;
            }
            self.level[u] = -1;
            pushed
        }
        fn maxflow(&mut self, s: usize, t: usize) -> i64 {
            self.build(t);
            let n = self.n;
            self.level.clear(); self.level.resize(n, -1);
            self.it.clear(); self.it.resize(n, 0);
            let mut flow = 0i64;
            for es in self.start[s] as usize..self.start[s + 1] as usize {
                let u = self.to[es] as usize;
                if self.cap[es] <= 0 { continue; }
                if self.sa[u] >= 0 {
                    let e2 = self.sa[u] as usize;
                    let d = if self.cap[es] < self.cap[e2] { self.cap[es] } else { self.cap[e2] };
                    if d > 0 {
                        self.cap[es] -= d; let r = self.rev[es] as usize; self.cap[r] += d;
                        self.cap[e2] -= d; let r2 = self.rev[e2] as usize; self.cap[r2] += d; flow += d;
                    }
                }
                for e in self.start[u] as usize..self.start[u + 1] as usize {
                    if self.cap[es] <= 0 { break; }
                    let v = self.to[e] as usize;
                    if self.sa[v] < 0 || self.cap[e] <= 0 { continue; }
                    let e2 = self.sa[v] as usize;
                    let mut d = self.cap[es]; if self.cap[e] < d { d = self.cap[e]; } if self.cap[e2] < d { d = self.cap[e2]; }
                    if d > 0 {
                        self.cap[es] -= d; let r = self.rev[es] as usize; self.cap[r] += d;
                        self.cap[e] -= d; let r1 = self.rev[e] as usize; self.cap[r1] += d;
                        self.cap[e2] -= d; let r2 = self.rev[e2] as usize; self.cap[r2] += d; flow += d;
                    }
                }
            }
            while self.bfs(s, t) {
                for i in 0..n { self.it[i] = self.start[i]; }
                flow += self.push(s, t, CB_INF);
            }
            flow
        }
        fn source_side(&mut self, s: usize, out: &mut Vec<bool>) {
            out.clear(); out.resize(self.n, false);
            self.q.clear(); self.q.push(s as u32); out[s] = true;
            while let Some(u) = self.q.pop() {
                for e in self.start[u as usize] as usize..self.start[u as usize + 1] as usize {
                    let v = self.to[e] as usize;
                    if self.cap[e] > 0 && !out[v] { out[v] = true; self.q.push(v as u32); }
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

    fn core_stage(ch: &Challenge, inc_bits: &[bool], inc_val: i64, bp: &[i32], bcross: i32, lam_c: (f64, f64), hp: &Hparams) -> Option<(Vec<bool>, i64)> {
        let n = ch.num_items;
        let num_params = hp.n_lambda_values as i64;
        let rem = (num_params + 1 - bcross as i64).max(0);
        let hi_steps = ((rem * hp.core_hi as i64 + 999) / 1000) as i32;
        let start = bcross - hi_steps;
        let mut fixed: Vec<usize> = Vec::new();
        let mut core: Vec<usize> = Vec::new();
        for i in 0..n {
            let b = bp[i];
            if b < start { fixed.push(i); }
            else if ((b as i64) <= num_params && (hp.core_up == 0 || b <= bcross + hp.core_up as i32 - 1)) || (hp.core_up == 0 && inc_bits[i]) { core.push(i); }
        }
        let cap = ch.max_weight as i64;
        let wf: i64 = fixed.iter().map(|&i| ch.weights[i] as i64).sum();
        if wf > cap || core.is_empty() { return None; }
        let mut vf: i64 = 0;
        for (x, &i) in fixed.iter().enumerate() {
            vf += ch.values[i] as i64;
            let row = &ch.interaction_values[i];
            for &j in &fixed[x + 1..] { vf += row[j] as i64; }
        }
        let build = |core: &[usize], fixed: &[usize]| -> CoreBnb {
            let k = core.len();
            let mut lin: Vec<i64> = Vec::with_capacity(k);
            let mut w: Vec<i64> = Vec::with_capacity(k);
            for &i in core {
                let row = &ch.interaction_values[i];
                let mut l = ch.values[i] as i64;
                for &j in fixed { l += row[j] as i64; }
                lin.push(l); w.push(ch.weights[i] as i64);
            }
            let mut adj_s: Vec<u32> = Vec::with_capacity(k + 1);
            let mut adj_j: Vec<u32> = Vec::new();
            let mut adj_q: Vec<i64> = Vec::new();
            for a in 0..k {
                adj_s.push(adj_j.len() as u32);
                let row = &ch.interaction_values[core[a]];
                for b in 0..k {
                    if b == a { continue; }
                    let qq = row[core[b]];
                    if qq > 0 { adj_j.push(b as u32); adj_q.push(qq as i64); }
                }
            }
            adj_s.push(adj_j.len() as u32);
            CoreBnb {
                k, w, lin_cur: lin, cap: 0, best_val: 0, best_sel: vec![0u8; k],
                improved: false, work: (k * k) as u64, work_cap: hp.core_work as u64, lam_iters: hp.core_lam,
                adj_s, adj_j, adj_q, pos: vec![-1; k], state: vec![-1; k], fl: CbFlow::new(),
                idx: Vec::new(), side: Vec::new(),
                ce: CutEng::new(), prep_ok: false, eng: hp.cut_engine,
            }
        };
        let mut used: u64 = 0;
        let mut lam0: i64 = 1;
        if hp.core_k > 0 && lam_c.1 > 0.0 {
            let lam = 0.5 * (lam_c.0 + lam_c.1);
            let mut g = vec![0i64; n];
            let refs: Vec<bool> = if hp.core_rc > 0 { (0..n).map(|i| bp[i] < bcross).collect() } else { inc_bits.to_vec() };
            for j in 0..n {
                if !refs[j] { continue; }
                let row = &ch.interaction_values[j];
                for i in 0..n { g[i] += row[i] as i64; }
            }
            let mut sgn = vec![false; n];
            let mut key: Vec<(i64, usize)> = (0..n).map(|i| {
                let d = (g[i] + ch.values[i] as i64) as f64 / ch.weights[i] as f64 - lam;
                sgn[i] = d > 0.0;
                (((if d < 0.0 { -d } else { d }) * 1024.0) as i64, i)
            }).collect();
            let kk = hp.core_k.min(n);
            key.select_nth_unstable(kk - 1);
            let mut free = vec![false; n];
            for x in 0..kk { free[key[x].1] = true; }
            fixed.clear(); core.clear();
            for i in 0..n { if free[i] { core.push(i); } else if (if hp.core_rc > 0 { sgn[i] } else { inc_bits[i] }) { fixed.push(i); } }
            let wfx: i64 = fixed.iter().map(|&i| ch.weights[i] as i64).sum();
            if wfx > cap { return None; }
            vf = 0;
            for (x, &i) in fixed.iter().enumerate() {
                vf += ch.values[i] as i64;
                let row = &ch.interaction_values[i];
                for &j in &fixed[x + 1..] { vf += row[j] as i64; }
            }
        } else
        if hp.core_band > 0 {
            let mut bb = build(&core, &fixed);
            let c1 = cap - wf;
            bb.idx.clear();
            for a in 0..bb.k { bb.pos[a] = a as i32; bb.idx.push(a as u32); }
            let (mut lo, mut f_lo, mut w_lo) = (0i64, 0i64, 0i64);
            let mut lam_max = 1i64;
            for a in 0..bb.k {
                w_lo += bb.w[a];
                let mut d = bb.lin_cur[a];
                let (b0, b1) = (bb.adj_s[a] as usize, bb.adj_s[a + 1] as usize);
                for x in b0..b1 { d += bb.adj_q[x]; if (bb.adj_j[x] as usize) > a { f_lo += bb.adj_q[x]; } }
                f_lo += bb.lin_cur[a];
                let l = (CB_SC * d) / bb.w[a].max(1) + 1;
                if l > lam_max { lam_max = l; }
            }
            f_lo *= CB_SC;
            let (mut hi, mut f_hi, mut w_hi) = (lam_max, 0i64, 0i64);
            if w_lo <= c1 { return None; }
            let mut guard = 0usize;
            let mut side_lo: Vec<bool> = Vec::new();
            let mut side_hi: Vec<bool> = Vec::new();
            let mut rk: Vec<(i64, Vec<bool>, u8)> = Vec::new();
            if hp.core_warm > 0 && lam_c.1 > 0.0 {
                let g_lo = ((CB_SC as f64) * lam_c.0 * 0.98) as i64;
                let g_hi = ((CB_SC as f64) * lam_c.1 * 1.02) as i64 + 1;
                if g_lo > lo && g_lo < hi {
                    let (v, wt) = bb.closure_k(g_lo, &rk); guard += 1; if bb.eng == 2 { rk.push((g_lo, bb.side.clone(), 1)); } let f = v + g_lo * wt;
                    if wt > c1 { lo = g_lo; f_lo = f; w_lo = wt; side_lo = bb.side.clone(); } else { hi = g_lo; f_hi = f; w_hi = wt; side_hi = bb.side.clone(); }
                }
                if g_hi > lo && g_hi < hi {
                    let (v, wt) = bb.closure_k(g_hi, &rk); guard += 1; if bb.eng == 2 { rk.push((g_hi, bb.side.clone(), 1)); } let f = v + g_hi * wt;
                    if wt > c1 { lo = g_hi; f_lo = f; w_lo = wt; side_lo = bb.side.clone(); } else { hi = g_hi; f_hi = f; w_hi = wt; side_hi = bb.side.clone(); }
                }
            }
            let mut newton = true;
            let rel = if hp.core_warm >= 2 { 2 * hp.core_band as i64 } else { 0 };
            while hi - lo > 1 && guard < 48 && !(rel > 0 && (hi - lo) * 1000 <= lo * rel && !side_lo.is_empty() && !side_hi.is_empty()) {
                guard += 1;
                let mut m = if newton && w_lo > w_hi { (f_lo - f_hi) / (w_lo - w_hi) } else { lo + (hi - lo) / 2 };
                let d = (hi - lo) / 8;
                if rel > 0 { if m < lo + d { m = lo + d; } if m > hi - d { m = hi - d; } }
                if m <= lo || m >= hi { m = lo + (hi - lo) / 2; }
                let (v, wt) = bb.closure_k(m, &rk); if bb.eng == 2 { rk.push((m, bb.side.clone(), 1)); }
                let f = v + m * wt;
                if wt > c1 {
                    if f == f_lo && wt == w_lo { newton = false; }
                    lo = m; f_lo = f; w_lo = wt; if rel > 0 { side_lo = bb.side.clone(); }
                } else {
                    if f == f_hi && wt == w_hi { newton = false; }
                    hi = m; f_hi = f; w_hi = wt; if rel > 0 { side_hi = bb.side.clone(); }
                }
            }
            let (fin, s2): (Vec<bool>, Vec<bool>) = if rel > 0 && !side_lo.is_empty() && !side_hi.is_empty() {
                (side_hi, side_lo)
            } else {
                let lhi = hi + (hi * hp.core_band as i64 + 999) / 1000;
                let llo = (lo - (lo * hp.core_band as i64 + 999) / 1000).max(0);
                bb.closure_k(lhi, &rk); let a: Vec<bool> = bb.side.clone();
                if bb.eng == 2 { rk.push((lhi, a.clone(), 1)); }
                bb.closure_k(llo, &rk); let b: Vec<bool> = bb.side.clone();
                (a, b)
            };
            used = bb.work;
            lam0 = hi;
            let mut fixed2 = fixed.clone();
            let mut core2: Vec<usize> = Vec::new();
            for a in 0..core.len() {
                if fin[a] { fixed2.push(core[a]); } else if s2[a] { core2.push(core[a]); }
            }
            let wf2: i64 = fixed2.iter().map(|&i| ch.weights[i] as i64).sum();
            if wf2 > cap || core2.is_empty() { return None; }
            fixed = fixed2; core = core2;
            vf = 0;
            for (x, &i) in fixed.iter().enumerate() {
                vf += ch.values[i] as i64;
                let row = &ch.interaction_values[i];
                for &j in &fixed[x + 1..] { vf += row[j] as i64; }
            }
        }
        let wf: i64 = fixed.iter().map(|&i| ch.weights[i] as i64).sum();
        let mut bb = build(&core, &fixed);
        bb.cap = cap - wf;
        bb.best_val = inc_val - vf;
        if hp.core_tot > 0 { bb.work += used; }
        let c0 = bb.cap;
        bb.dfs(0, c0, lam0, if hp.core_root > 0 { hp.core_root } else { 64 }, Vec::new());
        if !bb.improved { return None; }
        let mut bits = vec![false; n];
        for &i in &fixed { bits[i] = true; }
        for a in 0..core.len() { if bb.best_sel[a] != 0 { bits[core[a]] = true; } }
        let wt: i64 = (0..n).filter(|&i| bits[i]).map(|i| ch.weights[i] as i64).sum();
        if wt > cap { return None; }
        let val = bb.best_val + vf;
        if val > inc_val { Some((bits, val)) } else { None }
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
            let budget_pct = if sum_w > 0 {
                ((challenge.max_weight as u64) * 100 / sum_w) as u32
            } else { 10 };
            let hp = Hparams::from_map(hyperparameters, n, budget_pct);
            unsafe { SSB_MODE = hp.neutral_mask & (1usize << 21) != 0;
                     SSB_PRED = hp.exact_mask & (1usize << 41) != 0; }

            if hp.stage_stop > 0 {
                return Ok(Some(Solution { items: Vec::new() }));
            }

            if hp.rush_mode > 0 {
                let mut st = State::new_empty(challenge);
                let prefix_len = n.min(100);
                let static_synergy: Vec<i64> = (0..n)
                    .map(|i| challenge.interaction_values[i].iter().take(prefix_len).map(|&v| v as i64).sum())
                    .collect();
                let mut rng = Rng::from_seed(&challenge.seed);
                let mut champion: Option<SolState> = None;
                for walk in 0..hp.rush_restarts.max(1) {
                if walk > 0 {
                    st = State::new_empty(challenge);
                    match walk % 4 {
                        1 => build_greedy_synergy_weight(&mut st),
                        2 => build_greedy_hub(&mut st),
                        3 => build_greedy_value(&mut st),
                        _ => build_greedy_density(&mut st),
                    }
                } else
                if hp.rush_mode >= 4 {
                    build_greedy_density(&mut st);
                } else {
                    greedy_reconstruct(&mut st, 3, &static_synergy);
                }
                if hp.rush_mode >= 2 {
                    let core = hp.core_half_dp.min(24);
                    let win  = hp.window_k.min(120);
                    let passes = if hp.rush_mode >= 3 { 6 } else { 1 };
                    for _ in 0..passes {
                        let before = st.total_value;
                        dp_refinement_hp(&mut st, core, &hp);
                        local_search_vnd_windowed_m(&mut st, win, hp.neutral_mask);
                        if st.total_value <= before { break; }
                    }
                }
                if hp.rush_mode >= 5 && hp.rush_ils > 0 {
                    let core = hp.core_half_dp.min(24);
                    let win  = hp.window_k.min(120);
                    let mut best = st.clone_solution();
                    let mut local = st.clone_solution();
                    let mut stall = 0usize;
                    let mut kick = 0usize;
                    let tabu_on = hp.rush_tabu > 0;
                    let mut tabu_until: Vec<usize> = if tabu_on { vec![0usize; n] } else { Vec::new() };
                    let mut blocked: Vec<bool> = if tabu_on { vec![false; n] } else { Vec::new() };
                    let mut pre_bits: Vec<bool> = Vec::new();
                    for round in 0..hp.rush_ils {
                        if hp.rush_mode >= 6 && stall >= hp.rush_kick_stall {
                            st = State::new_empty(challenge);
                            match kick % 4 {
                                0 => build_greedy_synergy_weight(&mut st),
                                1 => build_greedy_hub(&mut st),
                                2 => build_greedy_value(&mut st),
                                _ => build_greedy_density(&mut st),
                            }
                            kick += 1;
                            dp_refinement_hp(&mut st, core, &hp);
                            local_search_vnd_windowed_m(&mut st, win, hp.neutral_mask);
                            local = st.clone_solution();
                            if local.value > best.value { best = st.clone_solution(); }
                            stall = 0;
                            continue;
                        }
                        if tabu_on {
                            pre_bits.clear();
                            pre_bits.extend_from_slice(&st.selected_bit);
                            perturb_by_strategy(&mut st, usize::MAX, stall, round % 7, &mut rng, &hp);
                            for i in 0..n {
                                if pre_bits[i] && !st.selected_bit[i] { tabu_until[i] = round + hp.rush_tabu; }
                                blocked[i] = tabu_until[i] > round;
                            }
                            greedy_reconstruct_blocked(&mut st, round % 4, &static_synergy, &blocked);
                        } else {
                            perturb_by_strategy(&mut st, usize::MAX, stall, round % 7, &mut rng, &hp);
                            greedy_reconstruct(&mut st, round % 4, &static_synergy);
                        }
                        dp_refinement_hp(&mut st, core, &hp);
                        local_search_vnd_windowed_m(&mut st, win, hp.neutral_mask);
                        if st.total_value > local.value {
                            local = st.clone_solution();
                            if local.value > best.value { best = st.clone_solution(); }
                            stall = 0;
                        } else {
                            st.restore_solution(&local);
                            stall += 1;
                        }
                    }
                    st.restore_solution(&best);
                }
                if st.total_weight <= challenge.max_weight
                    && champion.as_ref().map_or(true, |c| st.total_value > c.value)
                {
                    champion = Some(st.clone_solution());
                }
                }
                match champion {
                    Some(c) => { st.restore_solution(&c); }
                    None => { return Ok(None); }
                }
                return Ok(Some(Solution { items: st.selected_items() }));
            }

            let n_restarts = hp.n_full_restarts.max(1);

            let mut best_sol: Option<Solution> = None;
            let mut best_quality: i64 = i64::MIN;
            let mut best_bits: Vec<bool> = Vec::new();
            let mut xr_pool: Vec<(i64, Vec<bool>)> = Vec::new();
            let xr = hp.xr_elite;

            let mut h261_bp: Vec<i32> = Vec::new();
            let mut h261_bcross: i32 = -1;
            let mut h261_lam: (f64, f64) = (0.0, 0.0);
            if hp.bp_seed > 0 {
                let bp_sol = run_bp_seeded_instance(challenge, &hp, &mut h261_bp, &mut h261_bcross, &mut h261_lam);
                let val = bp_sol.value;
                if xr > 0 { xr_archive_push(&mut xr_pool, val, &bp_sol.bits, xr); }
                if val > best_quality {
                    best_quality = val;
                    best_bits.clear();
                    best_bits.extend_from_slice(&bp_sol.bits);
                    let items: Vec<usize> = (0..n)
                        .filter(|&i| bp_sol.bits[i])
                        .collect();
                    best_sol = Some(Solution { items });
                }
            }

            let deterministic_seed_cache = if n_restarts > 1 {
                Some(build_deterministic_seed_population(challenge, &hp, hp.core_half_dp))
            } else { None };

            let shared_tables: Option<(Vec<u64>, Vec<i64>)> = if hp.exact_mask & 512 != 0 {
                let prefix_len = n.min(100);
                Some((build_zobrist_table(n),
                      (0..n).map(|i| challenge.interaction_values[i].iter().take(prefix_len).map(|&v| v as i64).sum()).collect()))
            } else { None };

            let mut inject: Vec<SolState> = Vec::new();
            let mut inject_state = State::new_empty(challenge);
            for restart in 0..(n_restarts.saturating_sub(1)) {
                if xr > 0 {
                    inject.clear();
                    for entry in xr_pool.iter() {
                        set_state_from_bits(&mut inject_state, &entry.1);
                        inject.push(inject_state.clone_solution());
                    }
                }
                let lent: Option<(&[u64], &[i64])> = shared_tables.as_ref().map(|(z, sy)| (&z[..], &sy[..]));
                let (sol, val, snap) = if let Some(ref cache) = deterministic_seed_cache {
                    run_one_instance_with_seed_cache_value(challenge, &hp, restart, cache, &inject, lent)
                } else {
                    run_one_instance_with_value(challenge, &hp, restart, &inject)
                };
                if xr > 0 { xr_archive_push(&mut xr_pool, val, &snap.bits, xr); }
                if val > best_quality {
                    best_quality = val;
                    best_bits.clear();
                    best_bits.extend_from_slice(&snap.bits);
                    best_sol = Some(sol);
                }
            }

            if xr > 0 && xr_pool.len() >= 2 && !best_bits.is_empty() {
                let mut pr_val = best_quality;
                let mut pr_bits: Vec<bool> = Vec::new();
                for k in 0..xr_pool.len() {
                    if xr_pool[k].1 == best_bits { continue; }
                    let guide: Vec<bool> = xr_pool[k].1.clone();
                    if let Some((v, b)) = path_relink_t40(challenge, &best_bits, &guide, &hp, hp.core_half_dp) {
                        if v > pr_val { pr_val = v; pr_bits = b; }
                    }
                    if let Some((v, b)) = path_relink_t40(challenge, &guide, &best_bits, &hp, hp.core_half_dp) {
                        if v > pr_val { pr_val = v; pr_bits = b; }
                    }
                }
                if pr_val > best_quality && !pr_bits.is_empty() {
                    let items: Vec<usize> = (0..n).filter(|&i| pr_bits[i]).collect();
                    best_sol = Some(Solution { items });
                    best_quality = pr_val;
                    best_bits = pr_bits;
                }
            }

            if hp.core_work > 0 && h261_bcross >= 0 && h261_bp.len() == n && best_bits.len() == n {
                if let Some((bits, val)) = core_stage(challenge, &best_bits, best_quality, &h261_bp, h261_bcross, h261_lam, &hp) {
                    if val > best_quality {
                        let mut st = State::new_empty(challenge);
                        set_state_from_bits(&mut st, &bits);
                        let mut fin_bits = bits;
                        let mut fin_val = val;
                        if st.total_value == val && st.total_weight <= challenge.max_weight {
                            dp_refinement_hp(&mut st, hp.core_half_dp, &hp);
                            vnd_dispatch(&mut st, &hp);
                            if st.total_value > fin_val && st.total_weight <= challenge.max_weight {
                                fin_val = st.total_value; fin_bits = st.selected_bit.clone();
                            }
                        }
                        let _ = fin_val;
                        let items: Vec<usize> = (0..n).filter(|&i| fin_bits[i]).collect();
                        best_sol = Some(Solution { items });
                    }
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

} 

#[inline(always)]
pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hp: &Option<Map<String, Value>>,
) -> Result<()> {
    inner::solve_challenge(challenge, save_solution, hp)
}