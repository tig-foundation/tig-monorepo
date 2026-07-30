pub const INF: u32 = u32::MAX / 4;
pub const NONE_USIZE: usize = usize::MAX;
pub const NONE_U32: u32 = u32::MAX;

/// E21: index abstraction so the O(1) move estimators compile once for the `usize`-indexed
/// arrays of the cold paths and once for the `u32` arrays of the tabu hot loop.
/// Monomorphised: no dynamic dispatch, no runtime cost.
pub trait NodeIdx: Copy {
    fn is_none(self) -> bool;
    fn idx(self) -> usize;
}
impl NodeIdx for usize {
    #[inline(always)] fn is_none(self) -> bool { self == NONE_USIZE }
    #[inline(always)] fn idx(self) -> usize { self }
}
impl NodeIdx for u32 {
    #[inline(always)] fn is_none(self) -> bool { self == NONE_U32 }
    #[inline(always)] fn idx(self) -> usize { self as usize }
}

#[derive(Clone)]
pub struct OpInfo {
    pub machines: Vec<(usize, u32)>,
    pub min_pt: u32,
    pub avg_pt: f64,
    pub flex: usize,
    pub bn_avg: f64,
}

#[derive(Clone, Copy, Default)]
pub struct OpRoute {
    pub best_m: u8,
    pub best_w: u8,
    pub second_m: u8,
    pub second_w: u8,
}

pub type RoutePrefLite = Vec<Vec<OpRoute>>;

#[derive(Clone)]
pub struct Pre {
    pub job_products: Vec<usize>,
    pub job_ops_len: Vec<usize>,
    pub product_ops: Vec<Vec<OpInfo>>,
    pub product_suf_min: Vec<Vec<u32>>,
    pub product_suf_avg: Vec<Vec<f64>>,
    pub product_suf_bn: Vec<Vec<f64>>,
    pub product_next_min: Vec<Vec<u32>>,
    pub product_next_flex_inv: Vec<Vec<f64>>,
    pub machine_load0: Vec<f64>,
    pub machine_scarcity: Vec<f64>,
    pub machine_weight: Vec<f64>,
    pub machine_best_pop: Vec<f64>,
    pub avg_machine_load: f64,
    pub avg_machine_scarcity: f64,
    pub avg_op_min: f64,
    pub horizon: f64,
    pub time_scale: f64,
    pub max_ops: usize,
    pub max_job_avg_work: f64,
    pub max_job_bn: f64,
    pub flex_avg: f64,
    pub flex_factor: f64,
    pub hi_flex: bool,
    pub high_flex: f64,
    pub flow_like: f64,
    pub flow_w: f64,
    pub job_flow_pref: Vec<f64>,
    pub jobshopness: f64,
    pub bn_focus: f64,
    pub load_cv: f64,
    pub slack_base: f64,
    pub total_ops: usize,
    pub chaotic_like: bool,
    pub flow_route: Option<Vec<usize>>,
    pub flow_pt_by_job: Option<Vec<Vec<u32>>>,
    pub strict_route: Option<Vec<usize>>,
}

#[derive(Clone, Copy)]
pub struct Cand {
    pub job: usize,
    pub machine: usize,
    pub pt: u32,
    pub score: f64,
}

#[derive(Clone, Copy)]
pub struct RawCand {
    pub job: usize,
    pub machine: usize,
    pub pt: u32,
    pub base_score: f64,
    pub rigidity: f64,
    pub reg_n: f64,
}

#[derive(Clone, Copy)]
pub enum GreedyRule {
    MostWork,
    MostOps,
    LeastFlex,
    ShortestProc,
    LongestProc,
}

#[derive(Clone)]
pub struct DisjSchedule {
    pub n: usize,
    pub num_jobs: usize,
    pub num_machines: usize,
    pub job_offsets: Vec<usize>,
    pub job_succ: Vec<usize>,
    pub indeg_job: Vec<u16>,
    pub node_machine: Vec<usize>,
    pub node_pt: Vec<u32>,
    pub node_job: Vec<usize>,
    pub node_op: Vec<usize>,
    pub machine_seq: Vec<Vec<usize>>,
}

pub struct EvalBuf {
    pub indeg: Vec<u16>,
    pub start: Vec<u32>,
    pub best_pred: Vec<usize>,
    pub machine_succ: Vec<usize>,
    pub stack: Vec<usize>,
    /// E21-S2: the pop order of the last successful `eval_disj`, i.e. a valid topological
    /// order of the current disjunctive graph.  Iterating it in reverse is a valid
    /// reverse-topological order, which lets the tail (q-label) pass skip its own Kahn sort.
    pub topo: Vec<usize>,
    pub topo_len: usize,
}

impl EvalBuf {
    pub fn new(n: usize) -> Self {
        Self {
            indeg: vec![0u16; n],
            start: vec![0u32; n],
            best_pred: vec![NONE_USIZE; n],
            machine_succ: vec![NONE_USIZE; n],
            stack: vec![0usize; n + 1],
            topo: vec![0usize; n],
            topo_len: 0,
        }
    }
}

#[derive(Clone, Copy)]
pub struct MoveCand {
    pub kind: u8,
    pub m_from: usize,
    pub from: usize,
    pub m_to: usize,
    pub to: usize,
    pub new_pt: u32,
    pub score: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct EffortConfig {
    pub job_shop_iters: usize,
    pub hybrid_flow_shop_iters: usize,
    pub fjsp_medium_iters: usize,
    pub fjsp_high_iters: usize,
    /// flow_shop search-budget scale. FLOW_SHOP_ITERS_BASE == the built-in constants exactly.
    pub flow_shop_iters: usize,
    // -----------------------------------------------------------------------------------
    // E31 graft: budgets for the final CPDT tabu phase of the v9 hybrid_flow_shop engine
    // (`hybrid_flow_shop_v9.rs`).  Read *only* by that module, and that module runs only
    // when the `hfs_engine` knob == 1.  Values and clamps are copied verbatim from
    // adaptive_js_v9/types.rs:140-143 and :163-181 so the v9 engine sees exactly the
    // configuration it shipped with.  Adding fields cannot change any existing field's
    // value, so job_shop / flow_shop / fjsp_medium / fjsp_high are unaffected.
    // -----------------------------------------------------------------------------------
    pub hybrid_flow_shop_tabu_iters: usize,
    pub hybrid_flow_shop_tabu_seeds: usize,
    pub hybrid_flow_shop_tabu_stagnation: usize,
    pub hybrid_flow_shop_tabu_reassign_every: usize,
}

/// Denominator of the `flow_shop` budget scale.
pub const FLOW_SHOP_ITERS_BASE: usize = 2000;

impl EffortConfig {
    pub fn default_effort() -> Self {
        Self {
            job_shop_iters: 25000,
            hybrid_flow_shop_iters: 2000,
            fjsp_medium_iters: 2000,
            fjsp_high_iters: 2000,
            flow_shop_iters: FLOW_SHOP_ITERS_BASE,
            // v9 defaults (adaptive_js_v9/types.rs:150). Inert while hfs_engine == 0.
            hybrid_flow_shop_tabu_iters: 1800,
            hybrid_flow_shop_tabu_seeds: 5,
            hybrid_flow_shop_tabu_stagnation: 400,
            hybrid_flow_shop_tabu_reassign_every: 1,
        }
    }

    /// v9's own default constructor-restart budget for `hybrid_flow_shop`
    /// (adaptive_js_v9/types.rs:150).  Differs from the v10 default of 2000, so the v9
    /// engine has to be handed its own default explicitly -- see `solver::parse_effort`.
    pub const V9_HFS_DEFAULT_ITERS: usize = 3500;

    pub fn with_job_shop_iters(mut self, v: usize) -> Self {
        self.job_shop_iters = v.clamp(100, 4_000_000);
        self
    }

    pub fn with_hybrid_flow_shop_iters(mut self, v: usize) -> Self {
        self.hybrid_flow_shop_iters = v.clamp(100, 100000);
        self
    }

    pub fn with_fjsp_medium_iters(mut self, v: usize) -> Self {
        self.fjsp_medium_iters = v.clamp(100, 100000);
        self
    }

    pub fn with_fjsp_high_iters(mut self, v: usize) -> Self {
        self.fjsp_high_iters = v.clamp(100, 100000);
        self
    }
    pub fn with_flow_shop_iters(mut self, v: usize) -> Self {
        self.flow_shop_iters = v.clamp(100, 100000);
        self
    }

    // E31 graft: clamps copied verbatim from adaptive_js_v9/types.rs:163-181.
    pub fn with_hybrid_flow_shop_tabu_iters(mut self, v: usize) -> Self {
        self.hybrid_flow_shop_tabu_iters = v.clamp(100, 100000);
        self
    }

    pub fn with_hybrid_flow_shop_tabu_seeds(mut self, v: usize) -> Self {
        self.hybrid_flow_shop_tabu_seeds = v.clamp(1, 16);
        self
    }

    pub fn with_hybrid_flow_shop_tabu_stagnation(mut self, v: usize) -> Self {
        self.hybrid_flow_shop_tabu_stagnation = v.clamp(20, 10000);
        self
    }

    pub fn with_hybrid_flow_shop_tabu_reassign_every(mut self, v: usize) -> Self {
        self.hybrid_flow_shop_tabu_reassign_every = v.clamp(1, 32);
        self
    }
}

/// E21 --- cache-resident `u32` mirror of the disjunctive graph, owned by one tabu phase.
///
/// Rationale (see patch_E21_speed.md SECTION 1): 93% of a tabu iteration was two full
/// longest-path passes over 1500 nodes, and their random-access working set (~51 KB of
/// `usize` arrays) did not fit the 32 KB L1d of the EPYC 7702P.  Everything here is `u32`,
/// the machine arcs are maintained incrementally instead of being rebuilt per evaluation,
/// and the backward (tail) pass reuses the forward pass's topological order.
///
/// All of this is *representation only*: every label and every tie-break is bit-identical
/// to `eval_disj` + the original Kahn tail pass.
pub struct FastGraph {
    pub n: usize,
    // ---- static for the whole phase
    pub pt: Vec<u32>,
    pub job_succ: Vec<u32>,
    pub job_pred: Vec<u32>,
    pub indeg_job: Vec<u16>,
    // ---- maintained incrementally as machine sequences change
    pub mach_succ: Vec<u32>,
    pub mach_pred: Vec<u32>,
    pub indeg0: Vec<u16>,
    // ---- scratch, refilled per evaluation
    pub indeg: Vec<u16>,
    pub start: Vec<u32>,
    pub tail: Vec<u32>,
    pub best_pred: Vec<u32>,
    /// ready stack; physical length n+1 so a node can be stored speculatively at `stack[top]`
    pub stack: Vec<u32>,
    /// pop order of the last successful `eval()` == a topological order of the current graph
    pub topo: Vec<u32>,

    // =====================================================================
    // INCR (wave 4).  Everything below is inert unless `incr` is true, and
    // `mk_lowidx` is inert unless set; at the default knob values this struct
    // behaves exactly as the E23 version.
    // =====================================================================
    /// order-independent `mk_node`: lowest-index argmax of `start[i]+pt[i]`, and
    /// order-independent `best_pred`: lowest-index predecessor attaining `start[v]`.
    pub mk_lowidx: bool,
    /// maintain `ord`/`topo` across moves and use bounded incremental relabelling.
    pub incr: bool,
    /// maximum |dF|+|dB| a single Pearce-Kelly arc insertion may touch before the
    /// incremental path gives up and asks for a full pass.
    pub region_budget: usize,
    /// `ord[v]` == index of `v` in `topo`.  Valid only while `ord_valid`.
    pub ord: Vec<u32>,
    pub ord_valid: bool,
    /// `tail[]` matches the current graph (set by `tails()` / a successful incremental move).
    pub tails_fresh: bool,
    // ---- propagation scratch
    mark: Vec<u32>,
    pk_stamp: u32,
    pk_stack: Vec<u32>,
    pk_fwd: Vec<u32>,
    pk_bwd: Vec<u32>,
    pk_pos: Vec<u32>,
    seedbuf: Vec<u32>,

    // ---- measurement counters.  Plain u64 adds, a handful per tabu iteration against
    // ~10^5 cycles of work; they are never read back by the search, so they cannot affect
    // the trajectory (verified: the bit-identity gates below were run with them compiled in).
    pub st_moves: u64,
    pub st_bails: u64,
    pub st_cycles: u64,
    pub st_h_scan: u64,
    pub st_h_chg: u64,
    pub st_t_scan: u64,
    pub st_t_chg: u64,
    pub st_pk_calls: u64,
    pub st_pk_nodes: u64,
}

/// Outcome of one incremental move application.
pub enum IncrOut {
    /// labels are up to date; `(makespan, mk_node)`
    Ok(u32, usize),
    /// the move closed a cycle (exact, detected by Pearce-Kelly)
    Cycle,
    /// the perturbed region exceeded `region_budget`; machine arcs are consistent with
    /// `ds.machine_seq` but `ord`/labels are not -- the caller must run a full `eval()`.
    Bail,
}

enum Pk { Ok, Cycle, TooBig }

impl Drop for FastGraph {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering;
        let v = [self.st_moves, self.st_bails, self.st_cycles, self.st_h_scan,
                 self.st_h_chg, self.st_t_scan, self.st_t_chg,
                 self.st_pk_calls, self.st_pk_nodes];
        for i in 0..9 { super::knobs::INCR_ST[i].fetch_add(v[i], Ordering::Relaxed); }
    }
}

impl FastGraph {
    pub fn new(ds: &DisjSchedule) -> Self {
        let n = ds.n;
        let mut job_pred = vec![NONE_U32; n];
        for j in 0..ds.num_jobs {
            let base = ds.job_offsets[j];
            let end = ds.job_offsets[j + 1];
            for k in (base + 1)..end { job_pred[k] = (k - 1) as u32; }
        }
        let job_succ: Vec<u32> = (0..n)
            .map(|i| if ds.job_succ[i] == NONE_USIZE { NONE_U32 } else { ds.job_succ[i] as u32 })
            .collect();
        let incr = super::knobs::incr_eval() != 0;
        let mut g = Self {
            n,
            pt: ds.node_pt.clone(),
            job_succ,
            job_pred,
            indeg_job: ds.indeg_job.clone(),
            mach_succ: vec![NONE_U32; n],
            mach_pred: vec![NONE_U32; n],
            indeg0: vec![0u16; n],
            indeg: vec![0u16; n],
            start: vec![0u32; n],
            tail: vec![0u32; n],
            best_pred: vec![NONE_U32; n],
            stack: vec![0u32; n + 1],
            topo: vec![0u32; n],
            // incr state
            mk_lowidx: incr || super::knobs::mknode_lowidx() != 0,
            incr,
            region_budget: super::knobs::incr_region(128),
            ord: if incr { vec![0u32; n] } else { Vec::new() },
            ord_valid: false,
            tails_fresh: false,
            mark: if incr { vec![0u32; n] } else { Vec::new() },
            pk_stamp: 0,
            pk_stack: Vec::new(),
            pk_fwd: Vec::new(),
            pk_bwd: Vec::new(),
            pk_pos: Vec::new(),
            seedbuf: Vec::new(),
            st_moves: 0, st_bails: 0, st_cycles: 0,
            st_h_scan: 0, st_h_chg: 0,
            st_t_scan: 0, st_t_chg: 0,
            st_pk_calls: 0, st_pk_nodes: 0,
        };
        g.rebuild_machine_arcs(ds);
        g
    }

    /// `mknode_lowidx == 2` also makes `best_pred` order-independent.  Mode 1 changes only
    /// `mk_node`.  The incremental path requires mode 2 semantics and forces them on.
    #[inline(always)]
    fn bp_lowidx(&self) -> bool { self.incr || super::knobs::mknode_lowidx() >= 2 }

    /// Full O(n) rebuild of the machine arcs and base in-degrees from `ds.machine_seq`.
    /// Only needed after a wholesale sequence replacement (kick / restore-best).
    pub fn rebuild_machine_arcs(&mut self, ds: &DisjSchedule) {
        self.mach_succ.fill(NONE_U32);
        self.mach_pred.fill(NONE_U32);
        self.indeg0.copy_from_slice(&self.indeg_job);
        for seq in &ds.machine_seq {
            for i in 1..seq.len() {
                let a = seq[i - 1];
                let b = seq[i];
                self.mach_succ[a] = b as u32;
                self.mach_pred[b] = a as u32;
                self.indeg0[b] = self.indeg0[b].saturating_add(1);
            }
        }
    }

    /// O(hi-lo) repair after `relocate_machine_seq` rotated positions `lo..=hi` of `seq`.
    /// `old_first` must be `seq[0]` as it was *before* the rotation.
    ///
    /// Only arcs incident to positions `lo-1 ..= hi+1` can change (a rotation permutes the
    /// window and leaves every node outside it in place), and only position 0 can gain or
    /// lose a machine predecessor, so `indeg0` needs at most two O(1) fixes.
    #[inline]
    pub fn repair_machine_arcs(&mut self, seq: &[usize], lo: usize, hi: usize, old_first: usize) {
        let len = seq.len();
        if lo > 0 {
            let a = seq[lo - 1];
            self.mach_succ[a] = seq[lo] as u32;
            self.mach_pred[seq[lo]] = a as u32;
        } else {
            self.mach_pred[seq[0]] = NONE_U32;
        }
        for i in lo..hi {
            self.mach_succ[seq[i]] = seq[i + 1] as u32;
            self.mach_pred[seq[i + 1]] = seq[i] as u32;
        }
        if hi + 1 < len {
            self.mach_succ[seq[hi]] = seq[hi + 1] as u32;
            self.mach_pred[seq[hi + 1]] = seq[hi] as u32;
        } else {
            self.mach_succ[seq[hi]] = NONE_U32;
        }
        let new_first = seq[0];
        if new_first != old_first {
            self.indeg0[old_first] = self.indeg0[old_first].saturating_add(1);
            self.indeg0[new_first] = self.indeg0[new_first].saturating_sub(1);
        }
    }

    /// Forward longest-path (head labels).
    ///
    /// At `mk_lowidx == false` this is byte-for-byte the E23 pass: same stack discipline
    /// (ascending initial scan, LIFO pop), same strict `>` argmax so the same node wins a
    /// makespan tie, same `saturating_*` arithmetic.
    pub fn eval(&mut self) -> Option<(u32, usize)> {
        let r = if self.mk_lowidx {
            if self.bp_lowidx() { self.eval_impl::<true, true>() } else { self.eval_impl::<true, false>() }
        } else {
            self.eval_impl::<false, false>()
        };
        self.tails_fresh = false;
        if self.incr {
            if r.is_some() {
                for i in 0..self.n { self.ord[self.topo[i] as usize] = i as u32; }
                self.ord_valid = true;
            } else {
                self.ord_valid = false;
            }
        }
        r
    }

    #[inline(always)]
    fn eval_impl<const LOWIDX: bool, const BPLOW: bool>(&mut self) -> Option<(u32, usize)> {
        let n = self.n;
        self.indeg.copy_from_slice(&self.indeg0);
        self.start.fill(0);
        // `best_pred` is deliberately NOT cleared: `best_pred[v]` is written in exactly the
        // statement that raises `start[v]` above 0, so `best_pred[v]` is meaningful iff
        // `start[v] != 0`, and every reader stops at `start[v] == 0` (see the two walks in
        // job_shop.rs).  Saves a full n-word memset per evaluation.
        let mut top = 0usize;
        for i in 0..n {
            self.stack[top] = i as u32;
            top += (self.indeg[i] == 0) as usize;
        }
        let mut mk = 0u32;
        let mut mk_node = 0usize;
        let mut processed = 0usize;
        while top > 0 {
            top -= 1;
            let u = self.stack[top] as usize;
            self.topo[processed] = u as u32;
            processed += 1;
            let end_u = self.start[u].saturating_add(self.pt[u]);
            if !LOWIDX {
                if end_u > mk { mk = end_u; mk_node = u; }
            }
            let js = self.job_succ[u];
            if js != NONE_U32 {
                let v = js as usize;
                let s = self.start[v];
                let bp = self.best_pred[v];
                // BPLOW: on an exact tie the LOWEST-INDEX predecessor wins, which is
                // independent of the pop order and therefore reproducible incrementally.
                // The tie can only be taken when `s == end_u != 0`, and `start[v] != 0`
                // is exactly the condition under which `best_pred[v]` is meaningful.
                let better = if BPLOW {
                    s < end_u || (s == end_u && end_u != 0 && (u as u32) < bp)
                } else { s < end_u };
                self.start[v] = if s < end_u { end_u } else { s };
                self.best_pred[v] = if better { u as u32 } else { bp };
                let d = self.indeg[v].saturating_sub(1);
                self.indeg[v] = d;
                self.stack[top] = js;
                top += (d == 0) as usize;
            }
            let ms = self.mach_succ[u];
            if ms != NONE_U32 {
                let v = ms as usize;
                let s = self.start[v];
                let bp = self.best_pred[v];
                let better = if BPLOW {
                    s < end_u || (s == end_u && end_u != 0 && (u as u32) < bp)
                } else { s < end_u };
                self.start[v] = if s < end_u { end_u } else { s };
                self.best_pred[v] = if better { u as u32 } else { bp };
                let d = self.indeg[v].saturating_sub(1);
                self.indeg[v] = d;
                self.stack[top] = ms;
                top += (d == 0) as usize;
            }
        }
        if processed != n { return None; }
        if LOWIDX {
            // Order-independent makespan argmax: the LOWEST node index attaining `mk`.
            // One sequential pass over two u32 arrays; ~n cheap ops against the ~n random
            // accesses of the Kahn pass above, so the cost is in the noise.
            mk = 0; mk_node = 0;
            for i in 0..n {
                let e = self.start[i].saturating_add(self.pt[i]);
                if e > mk { mk = e; mk_node = i; }
            }
        }
        Some((mk, mk_node))
    }

    /// Backward longest-path (tail / q labels).  `topo` from the matching `eval()` is a
    /// topological order of this exact graph, so iterating it backwards visits every node
    /// after all of its successors -- a valid reverse-topological order.
    pub fn tails(&mut self) {
        self.tail.fill(0);
        for idx in (0..self.n).rev() {
            let nd = self.topo[idx] as usize;
            let contrib = self.pt[nd].saturating_add(self.tail[nd]);
            let jp = self.job_pred[nd];
            if jp != NONE_U32 {
                let p = jp as usize;
                self.tail[p] = self.tail[p].max(contrib);
            }
            let mp = self.mach_pred[nd];
            if mp != NONE_U32 {
                let p = mp as usize;
                self.tail[p] = self.tail[p].max(contrib);
            }
        }
        self.tails_fresh = true;
    }

    // =====================================================================
    // INCREMENTAL PATH
    // =====================================================================

    #[inline]
    fn next_pk_stamp(&mut self) -> u32 {
        self.pk_stamp = self.pk_stamp.wrapping_add(1);
        if self.pk_stamp == 0 { self.mark.fill(0); self.pk_stamp = 1; }
        self.pk_stamp
    }

    /// Head label + order-independent best predecessor of `v` from its two in-arcs.
    #[inline(always)]
    fn head_of(&self, v: usize) -> (u32, u32) {
        let jp = self.job_pred[v];
        let mp = self.mach_pred[v];
        let cj = if jp != NONE_U32 { self.start[jp as usize].saturating_add(self.pt[jp as usize]) } else { 0 };
        let cm = if mp != NONE_U32 { self.start[mp as usize].saturating_add(self.pt[mp as usize]) } else { 0 };
        // Branch-free argmax with the lowest-index tie-break.  NONE_U32 == u32::MAX makes the
        // `jp < mp` comparison do the right thing when one side is absent: the absent side
        // has contribution 0 and loses every tie it could take part in.
        let take_j = (cj > cm) | ((cj == cm) & (jp < mp));
        let h = if cj > cm { cj } else { cm };
        let bp = if take_j { jp } else { mp };
        (h, bp)
    }

    /// Tail label of `v` from its two out-arcs.
    #[inline(always)]
    fn tail_of(&self, v: usize) -> u32 {
        let js = self.job_succ[v];
        let ms = self.mach_succ[v];
        let a = if js != NONE_U32 { self.pt[js as usize].saturating_add(self.tail[js as usize]) } else { 0 };
        let b = if ms != NONE_U32 { self.pt[ms as usize].saturating_add(self.tail[ms as usize]) } else { 0 };
        if a > b { a } else { b }
    }

    /// Pearce & Kelly (2007) dynamic topological order maintenance for ONE inserted arc
    /// `x -> y`, on a graph whose `ord` is a valid topological order of everything else.
    ///
    /// If `ord[x] < ord[y]` nothing is needed.  Otherwise the affected region is
    /// `dF` = descendants of `y` with `ord < ord[x]` and `dB` = ancestors of `x` with
    /// `ord > ord[y]`; the positions they occupy are redistributed with all of `dB` before
    /// all of `dF`, which is exactly the PK reorder.  Every node outside `dF u dB` keeps its
    /// position, `dB` nodes only ever move down and `dF` nodes only ever move up, so no arc
    /// with an endpoint outside the region can be broken.
    ///
    /// Returns `Cycle` iff `y ~> x` already, which is an exact test.
    fn pk_insert(&mut self, x: u32, y: u32) -> Pk {
        let ub = self.ord[x as usize];
        let lb = self.ord[y as usize];
        if ub < lb { return Pk::Ok; }
        self.st_pk_calls += 1;
        let budget = self.region_budget;

        // ---- forward: descendants of y with ord < ord[x]
        let sf = self.next_pk_stamp();
        self.pk_fwd.clear();
        self.pk_stack.clear();
        self.mark[y as usize] = sf;
        self.pk_fwd.push(y);
        self.pk_stack.push(y);
        while let Some(uu) = self.pk_stack.pop() {
            let u = uu as usize;
            let succs = [self.job_succ[u], self.mach_succ[u]];
            for &w in succs.iter() {
                if w == NONE_U32 { continue; }
                if w == x { return Pk::Cycle; }
                let wi = w as usize;
                if self.ord[wi] < ub && self.mark[wi] != sf {
                    self.mark[wi] = sf;
                    self.pk_fwd.push(w);
                    self.pk_stack.push(w);
                    if self.pk_fwd.len() > budget { return Pk::TooBig; }
                }
            }
        }

        // ---- backward: ancestors of x with ord > ord[y]
        let sb = self.next_pk_stamp();
        self.pk_bwd.clear();
        self.pk_stack.clear();
        self.mark[x as usize] = sb;
        self.pk_bwd.push(x);
        self.pk_stack.push(x);
        while let Some(uu) = self.pk_stack.pop() {
            let u = uu as usize;
            let preds = [self.job_pred[u], self.mach_pred[u]];
            for &w in preds.iter() {
                if w == NONE_U32 { continue; }
                if w == y { return Pk::Cycle; }
                let wi = w as usize;
                if self.ord[wi] > lb && self.mark[wi] != sb {
                    self.mark[wi] = sb;
                    self.pk_bwd.push(w);
                    self.pk_stack.push(w);
                    if self.pk_bwd.len() > budget { return Pk::TooBig; }
                }
            }
        }

        // ---- reorder.  Sort keys (`ord` values) are distinct, so `sort_unstable_by_key`
        // is a total order and the result is deterministic.
        let mut fwd = core::mem::take(&mut self.pk_fwd);
        let mut bwd = core::mem::take(&mut self.pk_bwd);
        let mut pool = core::mem::take(&mut self.pk_pos);
        fwd.sort_unstable_by_key(|&v| self.ord[v as usize]);
        bwd.sort_unstable_by_key(|&v| self.ord[v as usize]);
        pool.clear();
        for &v in bwd.iter() { pool.push(self.ord[v as usize]); }
        for &v in fwd.iter() { pool.push(self.ord[v as usize]); }
        pool.sort_unstable();
        let mut k = 0usize;
        for &v in bwd.iter() {
            let p = pool[k]; k += 1;
            self.ord[v as usize] = p;
            self.topo[p as usize] = v;
        }
        for &v in fwd.iter() {
            let p = pool[k]; k += 1;
            self.ord[v as usize] = p;
            self.topo[p as usize] = v;
        }
        self.st_pk_nodes += (fwd.len() + bwd.len()) as u64;
        self.pk_fwd = fwd;
        self.pk_bwd = bwd;
        self.pk_pos = pool;
        Pk::Ok
    }

    /// Forward relabelling.  Every node whose in-arc set changed is seeded; a node is
    /// re-derived from its two predecessors, and its successors are only re-derived when its
    /// own head actually moved.  Because the scan follows `topo`, every predecessor of a node
    /// has already been finalised when that node is reached.
    fn propagate_heads(&mut self, skip: u32) -> bool {
        let mut first = usize::MAX;
        let mut last = 0usize;
        let mut any = false;
        let seeds = core::mem::take(&mut self.seedbuf);
        for &v in seeds.iter() {
            if v == skip { continue; }
            let o = self.ord[v as usize] as usize;
            if o < first { first = o; }
            if !any || o > last { last = o; }
            any = true;
        }
        self.seedbuf = seeds;
        if !any { return true; }
        // The perturbed set is DENSE inside its topological span (measured: 785 of the 1077
        // positions between the first and the last dirty node actually change), so a dirty
        // bitmap costs more in unpredictable branches and random flag accesses than it saves
        // in skipped recomputations.  Walk the whole span instead: recomputing a node whose
        // predecessors did not move is a no-op that writes back the value it already had.
        // `last` grows as changes propagate; successors always have a larger `ord`, so the
        // span is always extended forwards and every node is visited after all its
        // predecessors.
        let mut i = first;
        let mut chg = 0u64;
        while i <= last {
            let v = self.topo[i] as usize;
            let (h, bp) = self.head_of(v);
            self.best_pred[v] = bp;
            if h != self.start[v] {
                self.start[v] = h;
                chg += 1;
                let js = self.job_succ[v];
                if js != NONE_U32 {
                    let o = self.ord[js as usize] as usize;
                    if o > last { last = o; }
                }
                let ms = self.mach_succ[v];
                if ms != NONE_U32 {
                    let o = self.ord[ms as usize] as usize;
                    if o > last { last = o; }
                }
            }
            i += 1;
        }
        self.st_h_scan += (last + 1 - first) as u64;
        self.st_h_chg += chg;
        true
    }

    /// Backward relabelling, the mirror image.  Tails are pure `max` over successors, so no
    /// argmax has to be maintained and a node only wakes its predecessors when its own tail
    /// actually moved.
    fn propagate_tails(&mut self, skip: u32) -> bool {
        let mut first = usize::MAX;
        let mut last = 0usize;
        let mut any = false;
        let seeds = core::mem::take(&mut self.seedbuf);
        for &v in seeds.iter() {
            if v == skip { continue; }
            let o = self.ord[v as usize] as usize;
            if o < first { first = o; }
            if !any || o > last { last = o; }
            any = true;
        }
        self.seedbuf = seeds;
        if !any { return true; }
        let mut i = last;
        let mut chg = 0u64;
        loop {
            let v = self.topo[i] as usize;
            let t = self.tail_of(v);
            if t != self.tail[v] {
                self.tail[v] = t;
                chg += 1;
                let jp = self.job_pred[v];
                if jp != NONE_U32 {
                    let o = self.ord[jp as usize] as usize;
                    if o < first { first = o; }
                }
                let mp = self.mach_pred[v];
                if mp != NONE_U32 {
                    let o = self.ord[mp as usize] as usize;
                    if o < first { first = o; }
                }
            }
            if i == 0 || i <= first { break; }
            i -= 1;
        }
        self.st_t_scan += (last + 1 - i) as u64;
        self.st_t_chg += chg;
        true
    }

    /// Apply the machine-arc consequences of `relocate_machine_seq(seq, from, to)` (window
    /// `lo..=hi`, `seq` already rotated) and repair `ord`, `start`, `best_pred` and `tail`
    /// over just the perturbed region.
    ///
    /// Requires `ord_valid` and up-to-date `start`/`tail` for the PRE-move graph.
    pub fn incr_move(&mut self, seq: &[usize], lo: usize, hi: usize) -> IncrOut {
        self.st_moves += 1;
        let len = seq.len();
        let s0 = if lo > 0 { lo - 1 } else { lo };
        let s1 = if hi + 1 < len { hi + 1 } else { hi };
        // V := the node set at positions s0..=s1.  A rotation permutes lo..=hi and leaves the
        // two boundary positions alone, so V is the SAME node set before and after the move;
        // every machine arc that changed has both endpoints in V, and no other arc changed.
        let mut vb = core::mem::take(&mut self.seedbuf);
        vb.clear();
        for i in s0..=s1 { vb.push(seq[i] as u32); }

        // ---- 1. drop every old machine arc internal to V (indeg0 follows)
        for k in 0..vb.len() {
            let a = vb[k] as usize;
            let sc = self.mach_succ[a];
            if sc != NONE_U32 && vb.contains(&sc) {
                self.mach_succ[a] = NONE_U32;
                self.mach_pred[sc as usize] = NONE_U32;
                self.indeg0[sc as usize] = self.indeg0[sc as usize].saturating_sub(1);
            }
        }

        // ---- 2. install the new chain one arc at a time, repairing `ord` as we go.
        // Installing arc-by-arc keeps every Pearce-Kelly call in its single-arc setting: the
        // order is valid for the graph as it stands, plus exactly the one arc being inserted.
        let mut cyclic = false;
        let mut toobig = false;
        for i in s0..s1 {
            let x = seq[i] as u32;
            let y = seq[i + 1] as u32;
            self.mach_succ[x as usize] = y;
            self.mach_pred[y as usize] = x;
            self.indeg0[y as usize] = self.indeg0[y as usize].saturating_add(1);
            if !cyclic && !toobig {
                match self.pk_insert(x, y) {
                    Pk::Ok => {}
                    Pk::Cycle => { cyclic = true; }
                    Pk::TooBig => { toobig = true; }
                }
            }
        }
        self.seedbuf = vb;
        // The machine arcs and `indeg0` now describe `ds.machine_seq` exactly in ALL cases,
        // so a caller that undoes the move (`repair_machine_arcs`) or falls back to a full
        // `eval()` starts from consistent state.
        if cyclic { self.st_cycles += 1; self.ord_valid = false; self.tails_fresh = false; return IncrOut::Cycle; }
        if toobig { self.st_bails += 1; self.ord_valid = false; self.tails_fresh = false; return IncrOut::Bail; }

        // ---- 3. bounded relabelling.  Seeds are all of V: it is a superset of the nodes
        // whose in-arc set changed (heads) and of those whose out-arc set changed (tails).
        // `seq[s0]` (the node just before the window, when there is one) keeps both of its
        // predecessors, so its head cannot change -- it is only a TAIL seed.  Symmetrically
        // `seq[s1]` keeps both successors and is only a HEAD seed.
        let skip_head = if s0 < lo { seq[s0] as u32 } else { NONE_U32 };
        let skip_tail = if s1 > hi { seq[s1] as u32 } else { NONE_U32 };
        if !self.propagate_heads(skip_head) { self.st_bails += 1; self.ord_valid = false; self.tails_fresh = false; return IncrOut::Bail; }
        if !self.propagate_tails(skip_tail) { self.st_bails += 1; self.ord_valid = false; self.tails_fresh = false; return IncrOut::Bail; }
        self.tails_fresh = true;

        // ---- 4. makespan: lowest-index argmax over end times.  One sequential pass.
        let n = self.n;
        let mut mk = 0u32;
        let mut mk_node = 0usize;
        for i in 0..n {
            let e = self.start[i].saturating_add(self.pt[i]);
            if e > mk { mk = e; mk_node = i; }
        }
        IncrOut::Ok(mk, mk_node)
    }

    /// Debug/verification helper: recompute everything from scratch and compare.
    #[cfg(feature = "incr_verify")]
    pub fn verify_against_full(&mut self, ds: &DisjSchedule, mk: u32, mk_node: usize) -> Result<(), String> {
        let saved_start = self.start.clone();
        let saved_tail = self.tail.clone();
        let saved_bp = self.best_pred.clone();
        let saved_ms = self.mach_succ.clone();
        let saved_mp = self.mach_pred.clone();
        let saved_i0 = self.indeg0.clone();
        let saved_ord = self.ord.clone();
        let saved_topo = self.topo.clone();
        self.rebuild_machine_arcs(ds);
        if self.mach_succ != saved_ms { return Err("mach_succ".into()); }
        if self.mach_pred != saved_mp { return Err("mach_pred".into()); }
        if self.indeg0 != saved_i0 { return Err("indeg0".into()); }
        let r = self.eval();
        match r {
            None => return Err("full eval says cyclic".into()),
            Some((fmk, fnode)) => {
                if fmk != mk { return Err(format!("mk {} != {}", mk, fmk)); }
                if fnode != mk_node { return Err(format!("mk_node {} != {}", mk_node, fnode)); }
            }
        }
        if self.start != saved_start { return Err("heads differ".into()); }
        for v in 0..self.n {
            if self.start[v] != 0 && self.best_pred[v] != saved_bp[v] { return Err(format!("best_pred[{}]", v)); }
        }
        self.tails();
        if self.tail != saved_tail { return Err("tails differ".into()); }
        self.ord = saved_ord;
        self.topo = saved_topo;
        self.best_pred = saved_bp;
        self.tails_fresh = true;
        self.ord_valid = true;
        Ok(())
    }
}
