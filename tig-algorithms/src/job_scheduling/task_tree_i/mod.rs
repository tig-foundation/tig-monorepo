use anyhow::Result;
use serde_json::{Map, Value, Number};
use tig_challenges::job_scheduling::*;

mod track_t44;
mod track_t45;
mod track_t46;
mod track_t47;
mod jss_engine;
mod ref_greedy;

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
    use std::cell::{Cell, RefCell};
    let best_mk = Cell::new(u32::MAX);
    let guarded_save = |s: &Solution| -> Result<()> {
        match challenge.evaluate_makespan(s) {
            Ok(mk) if mk <= best_mk.get() => { best_mk.set(mk); save_solution(s) }
            _ => Ok(()),
        }
    };
    let track = parse_track(hyperparameters).unwrap_or_else(|| detect_track_simple(challenge));
    let result = match track {
        Track::FjspHigh => {
            let hp = merge_hp(hyperparameters, vec![
                ("track".into(), Value::String("fjsp_high".into())),
                ("fjsp_high_iters".into(), n(7000)), ("fjsp_high_pool_width".into(), n(15)),
            ]);
            track_t44::solve_challenge(challenge, &guarded_save, &hp)
        }
        Track::FjspMedium => {
            let hp = merge_hp(hyperparameters, vec![("track".into(), Value::String("fjsp_medium".into()))]);
            track_t45::solve_challenge(challenge, &guarded_save, &hp)
        }
        Track::FlowShop => {
            let hp = merge_hp(hyperparameters, vec![
                ("track".into(), Value::String("flow_shop".into())), ("flow_engine".into(), n(1)),
            ]);
            track_t46::solve_challenge(challenge, &guarded_save, &hp)
        }
        Track::HybridFlowShop => {
            let hp = merge_hp(hyperparameters, vec![
                ("track".into(), Value::String("hybrid_flow_shop".into())),
                ("hybrid_flow_shop_iters".into(), n(1000)),
                ("hybrid_flow_shop_n8".into(), n(0)), ("hybrid_flow_shop_ig_d".into(), n(6)), ("hybrid_flow_shop_accept".into(), n(0)),
                ("hybrid_flow_shop_ig_mode".into(), n(2)), ("hybrid_flow_shop_doe_mode".into(), n(0)), ("hybrid_flow_shop_ig_recon".into(), n(0)),
                ("hybrid_flow_shop_restarts".into(), n(0)), ("hybrid_flow_shop_ig_select".into(), n(0)), ("hybrid_flow_shop_tabu_iters".into(), n(60000)),
                ("hybrid_flow_shop_tabu_seeds".into(), n(5)), ("hybrid_flow_shop_path_relink".into(), n(1)), ("hybrid_flow_shop_kick_reentry".into(), n(0)),
                ("hybrid_flow_shop_tabu_kick_swaps".into(), n(3)), ("hybrid_flow_shop_tabu_stagnation".into(), n(8000)), ("hybrid_flow_shop_bottleneck_reassign".into(), n(0)),
            ]);
            track_t47::solve_challenge(challenge, &guarded_save, &hp)
        }
        Track::JobShop => {
            let hp = merge_hp(hyperparameters, vec![
                ("track".into(), Value::String("job_shop".into())),
                ("job_shop_iters".into(), n(200000)), ("js_ts_fast_eval".into(), n(1)), ("js_seed_select_mode".into(), n(1)),
                ("js_seed_sbp_construct".into(), n(2)), ("js_seed_sbp_dual_cycles".into(), n(5)),
            ]);
            jss_engine::solver::solve_challenge(challenge, &guarded_save, &hp)
        }
    };
    let greedy = RefCell::new(None);
    { let capture = |s: &Solution| -> Result<()> { *greedy.borrow_mut() = Some(s.clone()); Ok(()) };
      let _ = ref_greedy::solve_challenge_with_effort(challenge, &capture, 0); }
    if let Some(g) = greedy.borrow().as_ref() { let _ = guarded_save(g); }
    result
}
pub fn help() { println!("task_tree_i - per-track job scheduling solver"); }


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
