// TIG's UI uses the pattern `tig_challenges::job_scheduling` to automatically detect your algorithm's challenge
use anyhow::Result;
use serde_json::{Map, Value, Number};
use tig_challenges::job_scheduling::*;

mod track_t44;
mod track_t45;
mod track_t46;
mod track_t47;
mod track_t48;
mod ref_greedy;

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
    use std::cell::{Cell, RefCell};

    // --- Validity + greedy-floor guard ------------------------------------------------
    // The runtime keeps the LAST saved solution. We wrap save_solution so a solution is
    // only persisted when it is (a) structurally valid (passes evaluate_makespan) and
    // (b) no worse (by makespan) than the best valid solution seen so far. This makes the
    // final output the best VALID solution found:
    //   * invalid schedules (a job operation starting before the previous finishes, from a
    //     stale/cyclic disjunctive eval) fail evaluate_makespan and are skipped;
    //   * combined with the reference-greedy floor below, the final makespan is never worse
    //     than the verifier's greedy baseline ("must be better than greedy baseline" reject).
    let best_mk = Cell::new(u32::MAX);
    let guarded_save = |s: &Solution| -> Result<()> {
        match challenge.evaluate_makespan(s) {
            Ok(mk) if mk <= best_mk.get() => {
                best_mk.set(mk);
                save_solution(s)
            }
            _ => Ok(()), // invalid or non-improving: do not persist
        }
    };

    let track = parse_track(hyperparameters).unwrap_or_else(|| detect_track_simple(challenge));

    let result = match track {
        Track::FjspHigh => {
            let hp = merge_hp(hyperparameters, vec![
                ("track".into(), Value::String("fjsp_high".into())),
                ("fjsp_high_iters".into(), Value::Number(Number::from(3500u64))),
            ]);
            track_t44::solve_challenge(challenge, &guarded_save, &hp)
        }
        Track::FjspMedium => {
            let hp = merge_hp(hyperparameters, vec![("track".into(), Value::String("fjsp_medium".into()))]);
            track_t45::solve_challenge(challenge, &guarded_save, &hp)
        }
        Track::FlowShop => {
            let hp = merge_hp(hyperparameters, vec![
                ("track".into(), Value::String("flow_shop".into())),
                ("extra_iters".into(), Value::Number(Number::from(1560u64))),
                ("total_iters".into(), Value::Number(Number::from(7800u64))),
                ("num_restarts".into(), Value::Number(Number::from(10u64))),
                ("perturb_cycles".into(), Value::Number(Number::from(7u64))),
            ]);
            track_t46::solve_challenge(challenge, &guarded_save, &hp)
        }
        Track::HybridFlowShop => {
            let hp = merge_hp(hyperparameters, vec![("track".into(), Value::String("hybrid_flow_shop".into()))]);
            track_t47::solve_challenge(challenge, &guarded_save, &hp)
        }
        Track::JobShop => {
            let hp = merge_hp(hyperparameters, vec![
                ("track".into(), Value::String("job_shop".into())),
                ("job_shop_iters".into(), Value::Number(Number::from(55000u64))),
            ]);
            track_t48::solve_challenge(challenge, &guarded_save, &hp)
        }
    };

    // --- Reference-greedy floor -------------------------------------------------------
    // dispatching_rules effort 0 is exactly the verifier's greedy baseline. Offered LAST
    // through guarded_save, it becomes the final output only if the main solver failed to
    // match or beat it — guaranteeing the submitted makespan is never worse than greedy.
    // Effort 0 is cheap (10 restarts), so this does not disturb cases that already win.
    let greedy = RefCell::new(None);
    {
        let capture = |s: &Solution| -> Result<()> {
            *greedy.borrow_mut() = Some(s.clone());
            Ok(())
        };
        let _ = ref_greedy::solve_challenge_with_effort(challenge, &capture, 0);
    }
    if let Some(g) = greedy.borrow().as_ref() {
        let _ = guarded_save(g);
    }

    result
}

pub fn help() {
    println!("job_twelve — per-track flat self-contained, baked HPs (robustness-fixed)");
}
