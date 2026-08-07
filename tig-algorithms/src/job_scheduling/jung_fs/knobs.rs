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
/// 1 restores the legacy HashMap-iteration vote tie-break (NONDETERMINISTIC). Default 0 = deterministic.
pub static FM_LEGACY_VOTES: AtomicUsize = AtomicUsize::new(0);
pub static XT_FJM_KICK_SPACING: AtomicUsize = AtomicUsize::new(0);
pub static XT_FJM_KICK_TIGHT: AtomicUsize = AtomicUsize::new(0);
#[inline] pub fn xt_fjh_same_machine() -> usize { XT_FJH_SAME_MACHINE.load(Ordering::Relaxed) }

// ---------------------------------------------------------------------------
// fjsp_high tail / variance knobs (R1, R2, R5). Every one defaults to 0 ==
// original hardcoded behaviour, so an absent key reproduces the baseline
// bit-for-bit. Read only by `fjsp_high.rs`, so no other track can be affected.
// ---------------------------------------------------------------------------

/// R2 -- diversity-first elite archive in `fjsp_high::push_top_solutions_diverse`.
/// 0 = original behaviour (purely elitist: always evict a worst-makespan entry, using the
///     nearest-neighbour signature distance only as a tie-break among equal-worst entries).
/// q > 0 = reserve `q` of the `cap` archive slots for diversity: the best `cap - q` entries by
///     makespan are protected, and the evicted entry is the most redundant (smallest
///     nearest-neighbour Hamming distance) of the remainder. `q` is clamped to `1..=cap-1`.
pub static FH_ELITE_DIV: AtomicUsize = AtomicUsize::new(0);
/// R1 -- number of independent search segments (portfolio width) in `fjsp_high::solve`.
/// 0 (and 1) = one segment seeded exactly as today == baseline.
/// S > 1 = run S segments, each with its own RNG stream derived from `challenge.seed`, its own
///     elite archive, learned consensus, bandit statistics and stagnation counters; keep the best.
pub static FH_PORTFOLIO: AtomicUsize = AtomicUsize::new(0);
/// R1 -- restart budget *per segment*. 0 = use `effort.fjsp_high_iters` unchanged.
pub static FH_SEG_ITERS: AtomicUsize = AtomicUsize::new(0);
/// R5 -- consensus re-learning period, in restarts. 0 = original behaviour (the learned job bias /
/// machine penalty / route preferences are refreshed only on an improving restart and freeze for
/// good after 10 such refreshes). P > 0 = additionally refresh every P restarts and re-arm the
/// improvement-driven refresh budget.
pub static FH_RELEARN: AtomicUsize = AtomicUsize::new(0);
#[inline] pub fn fh_elite_div() -> usize { FH_ELITE_DIV.load(Ordering::Relaxed) }
#[inline] pub fn fh_portfolio() -> usize { FH_PORTFOLIO.load(Ordering::Relaxed) }
#[inline] pub fn fh_seg_iters() -> usize { FH_SEG_ITERS.load(Ordering::Relaxed) }
#[inline] pub fn fh_relearn() -> usize { FH_RELEARN.load(Ordering::Relaxed) }
#[inline] pub fn xt_fjm_keep_best()    -> usize { XT_FJM_KEEP_BEST.load(Ordering::Relaxed) }
#[inline] pub fn fm_legacy_votes()     -> usize { FM_LEGACY_VOTES.load(Ordering::Relaxed) }
#[inline] pub fn xt_fjm_kick_spacing() -> usize { XT_FJM_KICK_SPACING.load(Ordering::Relaxed) }
#[inline] pub fn xt_fjm_kick_tight()   -> usize { XT_FJM_KICK_TIGHT.load(Ordering::Relaxed) }

// ---------------------------------------------------------------------------
// E45 -- `fm_fuel_lean`: fuel-per-iteration reductions for the `fjsp_medium` track.
//
// 0 = today's exact code path.  1 = the lean path.  EVERY change behind this knob is
// BIT-IDENTICAL by construction -- it removes instructions that are provably redundant
// (unreachable saturation arms, a re-derived topological order, per-decision recomputation
// of per-(product, op_idx) constants, a provably-never-taken guard, a discarded clone and
// a duplicated weight loop).  It never changes a value, a comparison outcome, an RNG draw
// count or an RNG draw order.
//
// Motivation: the .so is instrumented per basic block with Load/Store/GEP/Br priced at
// ZERO, so fuel tracks *executed non-free instructions*, not memory traffic.  Every item
// below deletes executed arithmetic at identical output.
//
// SCOPE NOTE: two of the sites live in `infra_shared.rs` (`eval_disj`,
// `push_top_solutions`), which is shared by all five tracks.  Both are gated on this knob,
// so with the knob unset the other four tracks are untouched.
// ---------------------------------------------------------------------------
pub static FM_FUEL_LEAN: AtomicUsize = AtomicUsize::new(0);
pub static FM_POS_BUF: AtomicUsize = AtomicUsize::new(0);
pub static FM_TAILS_TOPO: AtomicUsize = AtomicUsize::new(0);
#[inline] pub fn fm_fuel_lean() -> usize { FM_FUEL_LEAN.load(Ordering::Relaxed) }
#[inline] pub fn fm_pos_buf()         -> usize { FM_POS_BUF.load(Ordering::Relaxed) }
#[inline] pub fn fm_tails_topo()      -> usize { FM_TAILS_TOPO.load(Ordering::Relaxed) }

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

// E43: hybrid_flow_shop perturbation + evaluator knobs.
pub static HFS_IG_MODE: AtomicUsize = AtomicUsize::new(0);
pub static HFS_IG_D: AtomicUsize = AtomicUsize::new(0);
pub static HFS_KICKS: AtomicUsize = AtomicUsize::new(0);
pub static HFS_KICK_PCT: AtomicUsize = AtomicUsize::new(0);
pub static HFS_KICK_SWAPS: AtomicUsize = AtomicUsize::new(0);
pub static HFS_KICK_SPACE: AtomicUsize = AtomicUsize::new(0);
pub static HFS_EVAL_LEAN: AtomicUsize = AtomicUsize::new(0);

#[inline] pub fn hfs_ig_mode() -> usize { HFS_IG_MODE.load(Ordering::Relaxed).min(2) }
#[inline] pub fn hfs_ig_d() -> usize { get(&HFS_IG_D, 6).clamp(1, 32) }
#[inline] pub fn hfs_kicks() -> usize { get(&HFS_KICKS, 5).clamp(1, 64) }
#[inline] pub fn hfs_kick_pct() -> usize {
    let v = HFS_KICK_PCT.load(Ordering::Relaxed);
    if v == 0 { 0 } else { v.clamp(10, 100) }
}
#[inline] pub fn hfs_kick_swaps() -> usize { get(&HFS_KICK_SWAPS, 3).clamp(1, 16) }
#[inline] pub fn hfs_kick_space() -> usize { HFS_KICK_SPACE.load(Ordering::Relaxed) }
#[inline] pub fn hfs_eval_lean() -> usize { HFS_EVAL_LEAN.load(Ordering::Relaxed) }

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// E44 -- flow_shop wall-time knobs.  Read ONLY by `flow_shop.rs`, so no other
// track can be affected.  Every knob defaults to 0 == today's exact behaviour,
// so with no hyperparameters the `flow_shop` track is bit-identical to e43.
//
// Group A (R1/R2/R3/R6) are *implementation* knobs: they change how much work is
// done per candidate move, never which move is chosen.  Group B are *budget*
// knobs (plumbing only for now); they DO change per-nonce output and must be
// swept against the full 40-nonce bundle before adoption.
// ---------------------------------------------------------------------------

// --- Group A: bit-identical implementation knobs ---------------------------

/// R3 -- Kahn seeding in `flow_shop::eval_disj_prepared` / `eval_disj_trial`.
/// 0 = original behaviour (scan all `n` nodes for `indeg[i] == 0`).
/// 1 = seed from job roots only (`ds.job_offsets[j]`).  `build_disj_from_solution`
///     gives every non-first operation `indeg_job == 1`, so the indeg-0 set is
///     *exactly* the set of job roots, and job roots are visited in increasing
///     node id -- identical set, identical push order, identical pop order.
pub static FS_KAHN_ROOTS: AtomicUsize = AtomicUsize::new(0);
/// R1 -- trial-move evaluation in `flow_shop::descent_phase`.
/// 0 = original behaviour (full `eval_disj_stateful`: fills `best_pred`, always
///     runs the whole topological pass).
/// 1 = `eval_disj_trial`: makespan only (never touches `best_pred`) and aborts
///     as soon as the running makespan reaches the incumbent bound.  Trials
///     consume only `mk2`; the accepted move is still re-evaluated in full.
pub static FS_TRIAL_BOUND: AtomicUsize = AtomicUsize::new(0);
/// R2 -- per-iteration memo of trial-move evaluations in `flow_shop::descent_phase`,
/// keyed on `(kind, m_from, from, to)`.
/// 0 = original behaviour (every entry of `cands` is evaluated, duplicates included).
/// 1 = evaluate each distinct move at most once per descent iteration.  The move
///     shift list `[1, 2, max_shift]` emits literal duplicates whenever
///     `max_shift <= 2`, which is the common case for flow-shop critical blocks.
pub static FS_TRIAL_MEMO: AtomicUsize = AtomicUsize::new(0);
/// R6 -- allocation / redundant-evaluation bundle (see `flow_shop.rs` call sites).
/// 0 = original behaviour.
/// 1 = clone only `machine_seq` instead of the whole `DisjSchedule`; reuse
///     `cur_eval` instead of re-running `eval_disj` after `descent_phase`;
///     diff-based `patch_eval_machine_state_one_machine`; hoist the perturbation
///     base out of the 20-round loop; reuse the descent's eval for `job_bias`;
///     test the makespan before cloning a `Solution` into `commit_valid_best`.
pub static FS_ALLOC_LEAN: AtomicUsize = AtomicUsize::new(0);

#[inline] pub fn fs_kahn_roots() -> usize { FS_KAHN_ROOTS.load(Ordering::Relaxed) }
#[inline] pub fn fs_trial_bound() -> usize { FS_TRIAL_BOUND.load(Ordering::Relaxed) }
#[inline] pub fn fs_trial_memo() -> usize { FS_TRIAL_MEMO.load(Ordering::Relaxed) }
#[inline] pub fn fs_alloc_lean() -> usize { FS_ALLOC_LEAN.load(Ordering::Relaxed) }

// --- Group B: budget knobs (PLUMBING ONLY -- these change per-nonce output) --

/// Cap on the number of critical-block local-search runs seeded from `top_solutions`.
/// 0 = today's 15.
pub static FS_LS_RUNS: AtomicUsize = AtomicUsize::new(0);
/// `perturb_cycles` for those runs. 0 = today's `clamp(total_ops / 40, 10, 40)`.
pub static FS_PERTURB_CYCLES: AtomicUsize = AtomicUsize::new(0);
/// `max_iters` (tabu iterations per `descent_phase`) for those runs. 0 = today's 7.
pub static FS_LS_ITERS: AtomicUsize = AtomicUsize::new(0);
/// `top_cands` prescreen width for those runs. 0 = today's 80.  Note the effective
/// width is `min(top_cands, 48).max(8)`, so any value >= 48 is equivalent to 80.
pub static FS_LS_CANDS: AtomicUsize = AtomicUsize::new(0);
/// Number of rounds of the second (perturb + short local search) block.
/// 0 = today's 20.  Use `FS_SKIP_PERTURB` to run zero rounds.
pub static FS_PERTURB_ROUNDS: AtomicUsize = AtomicUsize::new(0);
/// 1 = skip the second perturbation block entirely (task_tree_h has no equivalent).
pub static FS_SKIP_PERTURB: AtomicUsize = AtomicUsize::new(0);
/// 1 = skip the `descent_phase(.., 1, 20)` polish inside the GRASP restart loop,
/// together with the `build_disj_from_solution` / `EvalBuf::new` it needs
/// (task_tree_h has no equivalent).
pub static FS_SKIP_GRASP_DESCENT: AtomicUsize = AtomicUsize::new(0);

#[inline] pub fn fs_ls_runs(dflt: usize) -> usize { get(&FS_LS_RUNS, dflt) }
#[inline] pub fn fs_perturb_cycles(dflt: usize) -> usize { get(&FS_PERTURB_CYCLES, dflt) }
#[inline] pub fn fs_ls_iters(dflt: usize) -> usize { get(&FS_LS_ITERS, dflt) }
#[inline] pub fn fs_ls_cands(dflt: usize) -> usize { get(&FS_LS_CANDS, dflt) }
#[inline] pub fn fs_perturb_rounds(dflt: usize) -> usize { get(&FS_PERTURB_ROUNDS, dflt) }
#[inline] pub fn fs_skip_perturb() -> usize { FS_SKIP_PERTURB.load(Ordering::Relaxed) }
#[inline] pub fn fs_skip_grasp_descent() -> usize { FS_SKIP_GRASP_DESCENT.load(Ordering::Relaxed) }

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
