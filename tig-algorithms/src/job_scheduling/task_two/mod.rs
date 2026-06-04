// TIG's UI uses the pattern `tig_challenges::job_scheduling` to automatically detect your algorithm's challenge
use anyhow::Result;
use serde_json::{Map, Value, Number};
use tig_challenges::job_scheduling::*;

mod track_t44;
mod track_t45;
mod track_t46;
mod track_t47;
mod track_t48;

#[derive(Debug, Clone, Copy)]
enum Track {
    FlowShop,
    HybridFlowShop,
    JobShop,
    FjspMedium,
    FjspHigh,
}

fn parse_track(hyperparameters: &Option<Map<String, Value>>) -> Option<Track> {
    if let Some(map) = hyperparameters {
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

fn merge_hp(user_hp: &Option<Map<String, Value>>, defaults: Vec<(&str, Value)>) -> Option<Map<String, Value>> {
    let mut m = user_hp.clone().unwrap_or_default();
    for (k, v) in defaults {
        m.entry(k.to_string()).or_insert(v);
    }
    Some(m)
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let track = parse_track(hyperparameters).unwrap_or_else(|| detect_track_simple(challenge));

    match track {
        Track::FjspHigh => {
            let hp = merge_hp(hyperparameters, vec![
                ("track".into(), Value::String("fjsp_high".into())),
                ("fjsp_high_iters".into(), Value::Number(Number::from(3500u64))),
            ]);
            track_t44::solve_challenge(challenge, save_solution, &hp)
        }
        Track::FjspMedium => {
            let hp = merge_hp(hyperparameters, vec![("track".into(), Value::String("fjsp_medium".into()))]);
            track_t45::solve_challenge(challenge, save_solution, &hp)
        }
        Track::FlowShop => {
            let hp = merge_hp(hyperparameters, vec![
                ("track".into(), Value::String("flow_shop".into())),
                ("extra_iters".into(), Value::Number(Number::from(1560u64))),
                ("total_iters".into(), Value::Number(Number::from(7800u64))),
                ("num_restarts".into(), Value::Number(Number::from(10u64))),
                ("perturb_cycles".into(), Value::Number(Number::from(7u64))),
            ]);
            track_t46::solve_challenge(challenge, save_solution, &hp)
        }
        Track::HybridFlowShop => {
            let hp = merge_hp(hyperparameters, vec![("track".into(), Value::String("hybrid_flow_shop".into()))]);
            track_t47::solve_challenge(challenge, save_solution, &hp)
        }
        Track::JobShop => {
            let hp = merge_hp(hyperparameters, vec![
                ("track".into(), Value::String("job_shop".into())),
                ("job_shop_iters".into(), Value::Number(Number::from(55000u64))),
            ]);
            track_t48::solve_challenge(challenge, save_solution, &hp)
        }
    }
}

pub fn help() {
    println!("job_twelve — per-track flat self-contained, baked HPs");
}
