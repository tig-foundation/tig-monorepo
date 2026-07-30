//! Search-effort knobs, overridable via hyperparameters.
//!
//! Every knob defaults to 0 == "use the original hardcoded behaviour", so with no
//! hyperparameters this module is behaviourally identical to the baseline.
//! Values are written once from `solve_challenge` before any search begins and are
//! only read afterwards; the solver is single-threaded, so this is deterministic.
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

pub static TS_TENURE: AtomicUsize = AtomicUsize::new(0);
pub static TS_KICKS: AtomicUsize = AtomicUsize::new(0);
pub static EVAL_CAP: AtomicUsize = AtomicUsize::new(0);
pub static MAX_DIST: AtomicUsize = AtomicUsize::new(0);
pub static NUM_RESTARTS: AtomicUsize = AtomicUsize::new(0);
pub static TS_STARTS: AtomicUsize = AtomicUsize::new(0);
pub static NO_IMPROVE_DIV: AtomicUsize = AtomicUsize::new(0);
/// Acyclicity guard for `job_shop::tabu_search_phase` move generation.
/// 0 = original behaviour (moves are generated from O(1) head/tail estimates only, which
///     cannot see a cycle, so `eval_disj` can return `None` and abort the phase).
/// 1 = Balas & Vazacopoulos head/tail acyclicity tests are applied at every generation
///     site, so a cycle-creating move is never proposed.
pub static ACYCLIC_GUARD: AtomicUsize = AtomicUsize::new(0);

/// INCR wave 4 -- order-independent tie-breaks in the disjunctive longest-path pass.
/// 0 = original behaviour (`mk_node` is the first node in Kahn pop order attaining the
///     makespan; `best_pred` is the first predecessor in pop order attaining the head).
/// 1 = `mk_node` becomes the LOWEST-INDEX argmax of `start[i] + pt[i]` (order-independent).
/// 2 = additionally `best_pred[v]` becomes the LOWEST-INDEX predecessor attaining
///     `start[v]` (also order-independent).  Mode 2 is what incremental evaluation needs.
pub static MKNODE_LOWIDX: AtomicUsize = AtomicUsize::new(0);
/// 0 = off.  1 = replace the two full longest-path passes of the tabu inner loop with
/// bounded incremental head/tail maintenance (implies MKNODE_LOWIDX == 2 semantics).
pub static INCR_EVAL: AtomicUsize = AtomicUsize::new(0);
/// max |dF| + |dB| a single Pearce-Kelly arc insertion may touch before the incremental
/// path gives up and asks for a full pass.  0 = default (128).
pub static INCR_REGION: AtomicUsize = AtomicUsize::new(0);
/// Instrumentation only (feature `guard_stats`).
pub static INCR_MOVES: AtomicUsize = AtomicUsize::new(0);
pub static INCR_BAILS: AtomicUsize = AtomicUsize::new(0);
pub static INCR_CYCLES: AtomicUsize = AtomicUsize::new(0);
/// INCR measurement accumulators, flushed once per tabu phase from `FastGraph::drop`.
/// Order: moves, bails, cycles, h_scan, h_hit, h_chg, t_scan, t_hit, t_chg, pk_calls, pk_nodes.
pub static INCR_ST: [AtomicU64; 9] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0),
];
pub fn incr_st_dump() -> String {
    let names = ["moves","bails","cycles","h_scan","h_chg","t_scan","t_chg","pk_calls","pk_nodes"];
    let mut o = String::new();
    for i in 0..9 {
        o.push_str(&format!("{}={} ", names[i], INCR_ST[i].load(Ordering::Relaxed)));
    }
    o
}

/// Instrumentation only; compiled out unless the `guard_stats` feature is enabled.
/// Never read back by the solver, so they cannot influence the search.
pub static TS_PHASES: AtomicUsize = AtomicUsize::new(0);
pub static TS_ABORTS: AtomicUsize = AtomicUsize::new(0);
pub static TS_CANDIDATES: AtomicUsize = AtomicUsize::new(0);
pub static TS_GUARD_SKIPS: AtomicUsize = AtomicUsize::new(0);
pub static TS_GUARD_KICK_SKIPS: AtomicUsize = AtomicUsize::new(0);
pub static TS_FALLBACKS: AtomicUsize = AtomicUsize::new(0);

/// Cross-check counters; compiled out unless the `guard_verify` feature is enabled.
/// `VERIFY_*` classify a deterministic subsample of generated candidates by
/// (guard verdict) x (ground truth from a full `eval_disj` topological pass).
pub static VERIFY_TRUE_ACCEPT: AtomicUsize = AtomicUsize::new(0);
pub static VERIFY_UNSOUND_ACCEPT: AtomicUsize = AtomicUsize::new(0);
pub static VERIFY_TRUE_REJECT: AtomicUsize = AtomicUsize::new(0);
pub static VERIFY_FALSE_REJECT: AtomicUsize = AtomicUsize::new(0);
pub static VERIFY_D1_REJECT: AtomicUsize = AtomicUsize::new(0);

#[inline(always)]
pub fn stat(_c: &AtomicUsize) {
    #[cfg(feature = "guard_stats")]
    _c.fetch_add(1, Ordering::Relaxed);
}

#[inline(always)]
pub fn statv(_c: &AtomicUsize) {
    #[cfg(feature = "guard_verify")]
    _c.fetch_add(1, Ordering::Relaxed);
}

#[inline]
fn get(a: &AtomicUsize, dflt: usize) -> usize {
    let v = a.load(Ordering::Relaxed);
    if v == 0 { dflt } else { v }
}

#[inline] pub fn ts_tenure(dflt: usize) -> usize { get(&TS_TENURE, dflt) }
#[inline] pub fn ts_kicks(dflt: usize) -> usize { get(&TS_KICKS, dflt) }
#[inline] pub fn eval_cap(dflt: usize) -> usize { get(&EVAL_CAP, dflt) }
#[inline] pub fn max_dist(dflt: usize) -> usize { get(&MAX_DIST, dflt) }
#[inline] pub fn num_restarts(dflt: usize) -> usize { get(&NUM_RESTARTS, dflt) }
#[inline] pub fn ts_starts(dflt: usize) -> usize { get(&TS_STARTS, dflt) }
#[inline] pub fn no_improve_div(dflt: usize) -> usize { get(&NO_IMPROVE_DIV, dflt) }
#[inline] pub fn acyclic_guard() -> usize { ACYCLIC_GUARD.load(Ordering::Relaxed) }
#[inline] pub fn mknode_lowidx() -> usize { MKNODE_LOWIDX.load(Ordering::Relaxed) }
#[inline] pub fn incr_eval() -> usize { INCR_EVAL.load(Ordering::Relaxed) }
#[inline] pub fn incr_region(dflt: usize) -> usize { get(&INCR_REGION, dflt) }

// XTRACK audit knobs; every one defaults to 0 == original hardcoded behaviour.
pub static XT_FJH_SAME_MACHINE: AtomicUsize = AtomicUsize::new(0);
pub static XT_FJM_KEEP_BEST: AtomicUsize = AtomicUsize::new(0);
pub static XT_FJM_KICK_SPACING: AtomicUsize = AtomicUsize::new(0);
pub static XT_FJM_KICK_TIGHT: AtomicUsize = AtomicUsize::new(0);
#[inline] pub fn xt_fjh_same_machine() -> usize { XT_FJH_SAME_MACHINE.load(Ordering::Relaxed) }
#[inline] pub fn xt_fjm_keep_best()    -> usize { XT_FJM_KEEP_BEST.load(Ordering::Relaxed) }
#[inline] pub fn xt_fjm_kick_spacing() -> usize { XT_FJM_KICK_SPACING.load(Ordering::Relaxed) }
#[inline] pub fn xt_fjm_kick_tight()   -> usize { XT_FJM_KICK_TIGHT.load(Ordering::Relaxed) }

// hybrid_flow_shop audit knobs; 0 == original behaviour.
pub static HFS_LS_CYCLES: AtomicUsize = AtomicUsize::new(0);
pub static HFS_DEEP_THRESH_X100: AtomicUsize = AtomicUsize::new(0);
pub static HFS_ESCAPE_CD: AtomicUsize = AtomicUsize::new(0);
/// E31 graft: selects which hybrid_flow_shop implementation the `hybrid_flow_shop` track runs.
/// 0 (default) = `hybrid_flow_shop` (v10 lineage + our fixes) -- byte-identical to e30.
/// 1           = `hybrid_flow_shop_v9` (adaptive_js_v9's file, verbatim).
/// Read in exactly one place, `solver::solve_challenge`'s `Track::HybridFlowShop` arm, so it
/// cannot influence any other track. HFS_LS_CYCLES / HFS_DEEP_THRESH_X100 / HFS_ESCAPE_CD are
/// read only by `hybrid_flow_shop.rs` and are therefore inert when hfs_engine == 1.
pub static HFS_ENGINE: AtomicUsize = AtomicUsize::new(0);
#[inline] pub fn hfs_engine() -> usize { HFS_ENGINE.load(Ordering::Relaxed) }

#[inline] pub fn hfs_ls_cycles() -> usize { HFS_LS_CYCLES.load(Ordering::Relaxed) }
#[inline] pub fn hfs_deep_thresh_x100() -> usize { HFS_DEEP_THRESH_X100.load(Ordering::Relaxed) }
#[inline] pub fn hfs_escape_cd(dflt: usize) -> usize { get(&HFS_ESCAPE_CD, dflt) }

// ---------------------------------------------------------------------------
// Profiling scaffolding (feature "prof" only; compiles out completely otherwise).
// ---------------------------------------------------------------------------
#[cfg(feature = "prof")]
pub mod prof {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub static T: [AtomicU64; 8] = [
        AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
        AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    ];
    pub static C: [AtomicU64; 8] = [
        AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
        AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    ];
    #[inline(always)]
    pub fn now() -> u64 { unsafe { core::arch::x86_64::_rdtsc() } }
    #[inline(always)]
    pub fn acc(i: usize, t0: u64) {
        T[i].fetch_add(now().wrapping_sub(t0), Ordering::Relaxed);
        C[i].fetch_add(1, Ordering::Relaxed);
    }
    #[inline(always)]
    pub fn bump(i: usize, v: u64) { C[i].fetch_add(v, Ordering::Relaxed); }
    pub fn dump(label: &str) {
        let names = ["tail_pass", "critpath", "move_scan", "apply", "eval_disj", "iter_total", "n_crit_nodes", "n_cands"];
        let tot = T[5].load(Ordering::Relaxed).max(1);
        println!("--- prof {} ---", label);
        for i in 0..8 {
            let t = T[i].load(Ordering::Relaxed);
            let c = C[i].load(Ordering::Relaxed);
            if i < 6 {
                println!("  {:<12} cycles={:>15} calls={:>12} cyc/call={:>9.1} pct={:>5.1}%",
                    names[i], t, c, t as f64 / c.max(1) as f64, 100.0 * t as f64 / tot as f64);
            } else {
                println!("  {:<12} count={:>15}", names[i], c);
            }
        }
    }
}
