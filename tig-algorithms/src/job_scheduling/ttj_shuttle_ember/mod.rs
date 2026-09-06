use anyhow::Result;
use serde_json::{Map, Value, Number};
use tig_challenges::job_scheduling::*;

mod track_t47;
mod ref_greedy;
mod shuttle_finisher;
#[path = "ember_mod.rs"]
mod ember_stromboli;

#[derive(Debug, Clone, Copy)]
enum Track { FlowShop, HybridFlowShop, JobShop, FjspMedium, FjspHigh }

fn parse_track(hp: &Option<Map<String, Value>>) -> Option<Track> {
    if let Some(map) = hp {
        if let Some(Value::String(s)) = map.get("track") {
            return Some(match s.to_lowercase().as_str() {
                "flow_shop" | "flow" => Track::FlowShop,
                "hybrid_flow_shop" | "hybrid" => Track::HybridFlowShop,
                "job_shop" | "job" => Track::JobShop,
                "fjsp_medium" | "medium" => Track::FjspMedium,
                "fjsp_high" | "high" | "fjsp" => Track::FjspHigh,
                _ => return None,
            });
        }
    }
    None
}

fn merge_hp(user: &Option<Map<String, Value>>, defs: Vec<(String, Value)>) -> Option<Map<String, Value>> {
    let mut m = user.clone().unwrap_or_default();
    for (k, v) in defs { m.entry(k).or_insert(v); }
    Some(m)
}
fn n(v: u64) -> Value { Value::Number(Number::from(v)) }

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let track = parse_track(hyperparameters).unwrap_or_else(|| detect_track_simple(challenge));
    if !matches!(track, Track::HybridFlowShop) {
        // Exact public implementation, not a copy: the composition branch is
        // unreachable on all four non-hybrid tracks.
        return ember_stromboli::solve_challenge(
            challenge,
            save_solution,
            hyperparameters,
        );
    }
    solve_hybrid_composition(challenge, save_solution, hyperparameters)
}
pub fn help() { println!("task_tree_j - per-track job scheduling solver"); }

const FINISHER_FUEL_CAP: u64 = 120_000_000_000;

fn solve_hybrid_composition(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    use std::cell::{Cell, RefCell};

    // Phase 1 is byte-for-byte the public task_tree_j hybrid recipe.  The only
    // added state is an immutable clone of the best schedule passed to the
    // normal save sink, used as the finisher's warm start and safety floor.
    let best_mk = Cell::new(u32::MAX);
    let phase_best = RefCell::new(None::<Solution>);
    let phase_save = |s: &Solution| -> Result<()> {
        match challenge.evaluate_makespan(s) {
            Ok(mk) if mk <= best_mk.get() => {
                best_mk.set(mk);
                *phase_best.borrow_mut() = Some(s.clone());
                save_solution(s)
            }
            _ => Ok(()),
        }
    };

    let hp = merge_hp(hyperparameters, vec![
        ("track".into(), Value::String("hybrid_flow_shop".into())),
        ("hybrid_flow_shop_iters".into(), n(1000)),
        ("hybrid_flow_shop_n8".into(), n(0)), ("hybrid_flow_shop_ig_d".into(), n(6)), ("hybrid_flow_shop_accept".into(), n(0)),
        ("hybrid_flow_shop_ig_mode".into(), n(2)), ("hybrid_flow_shop_doe_mode".into(), n(0)), ("hybrid_flow_shop_ig_recon".into(), n(0)),
        ("hybrid_flow_shop_restarts".into(), n(0)), ("hybrid_flow_shop_ig_select".into(), n(0)), ("hybrid_flow_shop_tabu_iters".into(), n(60000)),
        ("hybrid_flow_shop_tabu_seeds".into(), n(5)), ("hybrid_flow_shop_path_relink".into(), n(1)), ("hybrid_flow_shop_kick_reentry".into(), n(0)),
        ("hybrid_flow_shop_tabu_kick_swaps".into(), n(3)), ("hybrid_flow_shop_tabu_stagnation".into(), n(8000)), ("hybrid_flow_shop_bottleneck_reassign".into(), n(0)),
    ]);
    let phase_result = track_t47::solve_challenge(challenge, &phase_save, &hp);

    // Preserve task_tree_j's final zero-effort greedy validity floor exactly.
    let greedy = RefCell::new(None);
    {
        let capture = |s: &Solution| -> Result<()> {
            *greedy.borrow_mut() = Some(s.clone());
            Ok(())
        };
        let _ = ref_greedy::solve_challenge_with_effort(challenge, &capture, 0);
    }
    if let Some(g) = greedy.borrow().as_ref() {
        let _ = phase_save(g);
    }
    phase_result?;

    let start = phase_best
        .borrow()
        .as_ref()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("task_tree_j produced no hybrid schedule"))?;
    let start_ms = challenge.evaluate_makespan(&start)?;

    shuttle_finisher::finish_from_schedule(
        challenge,
        &start,
        start_ms,
        &phase_save,
        FINISHER_FUEL_CAP,
    )?;
    Ok(())
}


fn detect_track_simple(challenge: &Challenge) -> Track {
    let mut total_flex = 0usize;
    let mut total_ops = 0usize;
    for p in 0..challenge.product_processing_times.len() {
        for op in &challenge.product_processing_times[p] {
            total_flex += op.len();
            total_ops += 1;
        }
    }
    let flex_avg = if total_ops > 0 { total_flex as f64 / total_ops as f64 } else { 1.0 };

    let mut max_ops = 0usize;
    let mut min_ops = usize::MAX;
    for p in 0..challenge.product_processing_times.len() {
        let nops = challenge.product_processing_times[p].len();
        if nops > max_ops { max_ops = nops; }
        if nops < min_ops { min_ops = nops; }
    }
    let uniform_routing = max_ops == min_ops;

    let is_flow_shop = if uniform_routing && flex_avg <= 1.5 && !challenge.product_processing_times.is_empty() {
        let n_ops = challenge.product_processing_times[0].len();
        let mut ok = true;
        'outer: for op_idx in 0..n_ops {
            let m0 = match challenge.product_processing_times[0][op_idx].keys().next() {
                Some(&m) => m,
                None => { ok = false; break; }
            };
            for p in 1..challenge.product_processing_times.len() {
                if !challenge.product_processing_times[p][op_idx].contains_key(&m0) {
                    ok = false;
                    break 'outer;
                }
            }
        }
        ok
    } else {
        false
    };

    if flex_avg > 5.0 {
        Track::FjspHigh
    } else if flex_avg > 1.5 && !uniform_routing {
        Track::FjspMedium
    } else if flex_avg > 1.5 {
        Track::HybridFlowShop
    } else if is_flow_shop {
        Track::FlowShop
    } else {
        Track::JobShop
    }
}
