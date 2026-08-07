use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::job_scheduling::*;

use super::types::EffortConfig;
use super::preprocess::build_pre;
use super::flow_shop;
use super::hybrid_flow_shop;
use super::hybrid_flow_shop_v9;
use super::job_shop;
use super::fjsp_medium;
use super::fjsp_high;
use super::knobs;
use std::sync::atomic::Ordering;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Track {
    FlowShop,
    HybridFlowShop,
    JobShop,
    FjspMedium,
    FjspHigh,
}

fn parse_track(hyperparameters: &Option<Map<String, Value>>) -> Track {
    if let Some(map) = hyperparameters {
        if let Some(Value::String(s)) = map.get("track") {
            return match s.to_lowercase().as_str() {
                "flow_shop" | "flow" => Track::FlowShop,
                "hybrid_flow_shop" | "hybrid" => Track::HybridFlowShop,
                "job_shop" | "job" => Track::JobShop,
                "fjsp_medium" | "medium" => Track::FjspMedium,
                "fjsp_high" | "high" | "fjsp" => Track::FjspHigh,
                _ => Track::FjspHigh,
            };
        }
    }
    Track::FjspHigh
}

/// True iff the caller asked for the v9 hybrid_flow_shop engine (`{"hfs_engine":1}`).
/// Reads the raw map rather than `knobs::hfs_engine()` so that `parse_effort` stays
/// independent of the order in which `set_knobs` runs.
fn wants_v9_hfs(hyperparameters: &Option<Map<String, Value>>) -> bool {
    if let Some(map) = hyperparameters {
        if let Some(Value::Number(n)) = map.get("hfs_engine") {
            return n.as_u64() == Some(1);
        }
    }
    false
}

fn parse_effort(hyperparameters: &Option<Map<String, Value>>) -> EffortConfig {
    let mut cfg = EffortConfig::default_effort();
    // E31 graft: the v9 hybrid engine shipped with its own restart-budget default of 3500
    // (adaptive_js_v9/types.rs:150) whereas the v10 lineage defaults to 2000.  The 40-nonce
    // median of 62,673 was measured for `adaptive_js_v9 {"track":"hybrid_flow_shop"}`, i.e.
    // at 3500, so hand the v9 engine its own default.  Applied *before* the explicit-key
    // pass below, so an explicit `hybrid_flow_shop_iters` still wins.  When hfs_engine != 1
    // this line is skipped and the returned config is bit-identical to e30's.
    if wants_v9_hfs(hyperparameters) {
        cfg = cfg.with_hybrid_flow_shop_iters(EffortConfig::V9_HFS_DEFAULT_ITERS);
    }
    if let Some(map) = hyperparameters {
        if let Some(Value::Number(n)) = map.get("job_shop_iters") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_job_shop_iters(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_iters") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_hybrid_flow_shop_iters(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("fjsp_medium_iters") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_fjsp_medium_iters(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("fjsp_high_iters") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_fjsp_high_iters(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("flow_shop_iters") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_flow_shop_iters(v as usize);
            }
        }
        // E31 graft: v9-hybrid-only budgets. Read only by `hybrid_flow_shop_v9`.
        if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_tabu_iters") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_hybrid_flow_shop_tabu_iters(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_tabu_seeds") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_hybrid_flow_shop_tabu_seeds(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_tabu_stagnation") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_hybrid_flow_shop_tabu_stagnation(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_tabu_reassign_every") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_hybrid_flow_shop_tabu_reassign_every(v as usize);
            }
        }
    }
    cfg
}

fn set_knobs(hyperparameters: &Option<Map<String, Value>>) {
    if let Some(map) = hyperparameters {
        for (key, slot) in [
            ("ts_tenure", &knobs::TS_TENURE),
            ("ts_kicks", &knobs::TS_KICKS),
            ("eval_cap", &knobs::EVAL_CAP),
            ("max_dist", &knobs::MAX_DIST),
            ("num_restarts", &knobs::NUM_RESTARTS),
            ("ts_starts", &knobs::TS_STARTS),
            ("no_improve_div", &knobs::NO_IMPROVE_DIV),
            ("acyclic_guard", &knobs::ACYCLIC_GUARD),
            ("xt_fjh_same_machine", &knobs::XT_FJH_SAME_MACHINE),
            ("xt_fjm_keep_best", &knobs::XT_FJM_KEEP_BEST),
            ("fm_legacy_votes", &knobs::FM_LEGACY_VOTES),
            ("xt_fjm_kick_spacing", &knobs::XT_FJM_KICK_SPACING),
            ("xt_fjm_kick_tight", &knobs::XT_FJM_KICK_TIGHT),
            // E45: fjsp_medium fuel-per-iteration knob. Bit-identical at any value; 0 ==
            // today's code path. Read by `fjsp_medium.rs` and, gated, by the two
            // `infra_shared.rs` sites listed in the knobs.rs comment.
            ("fm_fuel_lean", &knobs::FM_FUEL_LEAN),
            ("fm_pos_buf", &knobs::FM_POS_BUF),
            ("fm_tails_topo", &knobs::FM_TAILS_TOPO),
            ("hfs_ls_cycles", &knobs::HFS_LS_CYCLES),
            ("hfs_deep_thresh_x100", &knobs::HFS_DEEP_THRESH_X100),
            ("hfs_escape_cd", &knobs::HFS_ESCAPE_CD),
            // E43: hybrid_flow_shop perturbation + evaluator knobs.
            ("hfs_ig_mode", &knobs::HFS_IG_MODE),
            ("hfs_ig_d", &knobs::HFS_IG_D),
            ("hfs_kicks", &knobs::HFS_KICKS),
            ("hfs_kick_pct", &knobs::HFS_KICK_PCT),
            ("hfs_kick_swaps", &knobs::HFS_KICK_SWAPS),
            ("hfs_kick_space", &knobs::HFS_KICK_SPACE),
            ("hfs_eval_lean", &knobs::HFS_EVAL_LEAN),
            ("hfs_engine", &knobs::HFS_ENGINE),
            // E34 merge (INCR wave 4): job_shop-only knobs. Distinct keys, so the position
            // in this array cannot affect what any other knob resolves to.
            ("mknode_lowidx", &knobs::MKNODE_LOWIDX),
            ("incr_eval", &knobs::INCR_EVAL),
            ("incr_region", &knobs::INCR_REGION),
            // fjsp_high tail/variance knobs (R1/R2/R5). Distinct keys, read only by
            // `fjsp_high.rs`; every one defaults to 0 == original hardcoded behaviour.
            ("fh_elite_div", &knobs::FH_ELITE_DIV),
            ("fh_portfolio", &knobs::FH_PORTFOLIO),
            ("fh_seg_iters", &knobs::FH_SEG_ITERS),
            ("fh_relearn", &knobs::FH_RELEARN),
            // E44 flow_shop wall-time knobs; every one defaults to 0 == original behaviour.
            ("fs_kahn_roots", &knobs::FS_KAHN_ROOTS),
            ("fs_trial_bound", &knobs::FS_TRIAL_BOUND),
            ("fs_trial_memo", &knobs::FS_TRIAL_MEMO),
            ("fs_alloc_lean", &knobs::FS_ALLOC_LEAN),
            ("fs_ls_runs", &knobs::FS_LS_RUNS),
            ("fs_perturb_cycles", &knobs::FS_PERTURB_CYCLES),
            ("fs_ls_iters", &knobs::FS_LS_ITERS),
            ("fs_ls_cands", &knobs::FS_LS_CANDS),
            ("fs_perturb_rounds", &knobs::FS_PERTURB_ROUNDS),
            ("fs_skip_perturb", &knobs::FS_SKIP_PERTURB),
            ("fs_skip_grasp_descent", &knobs::FS_SKIP_GRASP_DESCENT),
        ] {
            if let Some(Value::Number(n)) = map.get(key) {
                if let Some(v) = n.as_u64() { slot.store(v as usize, Ordering::Relaxed); }
            }
        }
    }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let pre = build_pre(challenge)?;
    let track = parse_track(hyperparameters);
    let effort = parse_effort(hyperparameters);
    set_knobs(hyperparameters);
    // E43 containment. `hfs_eval_lean` is the only new knob read outside the v9 hybrid engine
    // (`infra_shared::eval_disj` is shared by all five tracks), and setting it leaves
    // `buf.topo` / `buf.topo_len` stale -- which `job_shop.rs` reads. Clear it on every path
    // except the one call tree that provably never reads them.
    if !(track == Track::HybridFlowShop && knobs::hfs_engine() == 1) {
        knobs::HFS_EVAL_LEAN.store(0, Ordering::Relaxed);
    }

    match track {
        Track::FlowShop => {
            flow_shop::solve(challenge, save_solution, &pre, &effort)
        }
        Track::HybridFlowShop => {
            // E31 graft. This is the ONLY call site of `hybrid_flow_shop_v9`; reaching it
            // requires track == hybrid_flow_shop AND hfs_engine == 1, so no other track can
            // ever enter the v9 code, and hfs_engine == 0 reproduces e30 exactly.
            if knobs::hfs_engine() == 1 {
                hybrid_flow_shop_v9::solve(challenge, save_solution, &pre, &effort)
            } else {
                hybrid_flow_shop::solve(challenge, save_solution, &pre, &effort)
            }
        }
        Track::JobShop => {
            job_shop::solve(challenge, save_solution, &pre, &effort)
        }
        Track::FjspMedium => {
            fjsp_medium::solve(challenge, save_solution, &pre, &effort)
        }
        Track::FjspHigh => {
            fjsp_high::solve(challenge, save_solution, &pre, &effort)
        }
    }
}

pub fn help() {
    println!("adaptive_js_v7 benchmarker hyperparameters");
    println!();
    println!("track (string):");
    println!("  selects which solver runs; each track is independent");
    println!("  accepted values:");
    println!("    \"flow_shop\" | \"flow\"");
    println!("    \"hybrid_flow_shop\" | \"hybrid\"");
    println!("    \"job_shop\" | \"job\"");
    println!("    \"fjsp_medium\" | \"medium\"");
    println!("    \"fjsp_high\" | \"high\" | \"fjsp\"");
    println!("  default if omitted or invalid: \"fjsp_high\"");
    println!();
    println!("job_shop_iters (integer):");
    println!("  affects track: job_shop (tabu search iteration budget)");
    println!("  range after clamp: 100..200000");
    println!("  default: 25000");
    println!();
    println!("hybrid_flow_shop_iters (integer):");
    println!("  affects track: hybrid_flow_shop (restart budget)");
    println!("  range after clamp: 100..100000");
    println!("  default: 2000");
    println!();
    println!("fjsp_medium_iters (integer):");
    println!("  affects track: fjsp_medium (restart budget; also scales tabu/cb/alns/ils budgets)");
    println!("  range after clamp: 100..100000");
    println!("  default: 2000");
    println!();
    println!("hfs_engine (integer):");
    println!("  affects track: hybrid_flow_shop only");
    println!("    0 (default) = v10-lineage engine (hybrid_flow_shop.rs)");
    println!("    1           = adaptive_js_v9 engine (hybrid_flow_shop_v9.rs, verbatim)");
    println!("  with hfs_engine=1 the restart-budget default becomes 3500 (v9's own default)");
    println!("  and these v9-only keys become live:");
    println!("    hybrid_flow_shop_tabu_iters          (100..100000, default 1800)");
    println!("    hybrid_flow_shop_tabu_seeds          (1..16,       default 5)");
    println!("    hybrid_flow_shop_tabu_stagnation     (20..10000,   default 400)");
    println!("    hybrid_flow_shop_tabu_reassign_every (1..32,       default 1)");
    println!("  they are ignored when hfs_engine=0");
    println!();
    println!("fh_elite_div (integer):");
    println!("  affects track: fjsp_high only");
    println!("    0 (default) = purely elitist elite archive (baseline)");
    println!("    q>0         = reserve q of the 15 archive slots for signature diversity");
    println!();
    println!("fh_portfolio (integer):");
    println!("  affects track: fjsp_high only");
    println!("    0/1 (default) = single search segment (baseline)");
    println!("    S>1           = S independent segments (own RNG/archive/consensus), keep best");
    println!();
    println!("fh_seg_iters (integer):");
    println!("  affects track: fjsp_high only; restart budget per segment");
    println!("    0 (default) = use fjsp_high_iters unchanged");
    println!();
    println!("fh_relearn (integer):");
    println!("  affects track: fjsp_high only");
    println!("    0 (default) = consensus freezes after 10 improvement-driven refreshes");
    println!("    P>0         = additionally refresh the consensus every P restarts");
    println!();
    println!("notes:");
    println!("  flow_shop: no tunable hyperparameter; iteration budget is fixed internally");
    println!("  fjsp_high: uses a fixed internal restart budget of 2000; not tunable via hyperparameters");
    println!("  all other hyperparameter keys are ignored");
}
