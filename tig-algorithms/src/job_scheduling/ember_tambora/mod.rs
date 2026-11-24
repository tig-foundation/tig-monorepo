use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::job_scheduling::*;

mod part_a;
mod part_b;

const JS_SBGLS_DIV: usize = 3;
static JS_SBGLS_DIV_HP: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(JS_SBGLS_DIV);

fn scenario(challenge: &Challenge) -> &'static str {
    let p = &challenge.product_processing_times;
    let flex = p[0].iter().map(|op| op.len()).max().unwrap_or(1);
    let common = p.iter().all(|q| {
        q.len() == p[0].len()
            && q.iter().zip(&p[0]).all(|(a, b)| a.len() == b.len() && a.keys().all(|m| b.contains_key(m)))
    });
    match (common, flex) {
        (true, 1) => "flow_shop",
        (true, _) => "hybrid_flow_shop",
        (false, 1) => "job_shop",
        (false, f) if f >= 6 => "fjsp_high",
        _ => "fjsp_medium",
    }
}

fn processing_row(
    op: &std::collections::HashMap<usize, u32>,
    num_machines: usize,
) -> Result<Vec<Option<u32>>> {
    let mut row = vec![None; num_machines];
    for (&m, &pt) in op {
        let slot = row
            .get_mut(m)
            .ok_or_else(|| anyhow::anyhow!("machine id out of range"))?;
        *slot = Some(pt);
    }
    Ok(row)
}

fn checked_makespan(challenge: &Challenge, solution: &Solution) -> Result<u32> {
    if solution.job_schedule.len() != challenge.num_jobs {
        return Err(anyhow::anyhow!("invalid job count"));
    }
    let mut rows = Vec::with_capacity(challenge.product_processing_times.len());
    for ops in &challenge.product_processing_times {
        let mut product = Vec::with_capacity(ops.len());
        for op in ops {
            product.push(processing_row(op, challenge.num_machines)?);
        }
        rows.push(product);
    }
    let mut usage = vec![Vec::new(); challenge.num_machines];
    let mut makespan = 0;
    let mut job = 0;
    for (product, &count) in challenge.jobs_per_product.iter().enumerate() {
        let processing = rows
            .get(product)
            .ok_or_else(|| anyhow::anyhow!("invalid product"))?;
        for _ in 0..count {
            let schedule = solution
                .job_schedule
                .get(job)
                .ok_or_else(|| anyhow::anyhow!("invalid product expansion"))?;
            if schedule.len() != processing.len() {
                return Err(anyhow::anyhow!("invalid operation count"));
            }
            let mut ready = 0;
            for (oi, &(machine, start)) in schedule.iter().enumerate() {
                if start < ready {
                    return Err(anyhow::anyhow!("job precedence violation"));
                }
                let pt = processing[oi]
                    .get(machine)
                    .copied()
                    .flatten()
                    .ok_or_else(|| anyhow::anyhow!("ineligible machine"))?;
                let end = start
                    .checked_add(pt)
                    .ok_or_else(|| anyhow::anyhow!("time overflow"))?;
                usage[machine].push((start, end));
                ready = end;
            }
            makespan = makespan.max(ready);
            job += 1;
        }
    }
    if job != challenge.num_jobs {
        return Err(anyhow::anyhow!("invalid product expansion"));
    }
    for machine in &mut usage {
        machine.sort_by_key(|&(start, _)| start);
        if machine.windows(2).any(|w| w[1].0 < w[0].1) {
            return Err(anyhow::anyhow!("machine overlap"));
        }
    }
    Ok(makespan)
}

#[allow(dead_code)]
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    if scenario(challenge) == "flow_shop" { return part_b::solve_challenge(challenge, save_solution, hyperparameters); }
    let s = scenario(challenge);
    if s == "job_shop" && challenge.num_jobs == 50 {
        let sbgls_div = hyperparameters.as_ref()
            .and_then(|hp| hp.get("js_sbgls_div"))
            .and_then(Value::as_u64)
            .map(|value| value.min(usize::MAX as u64) as usize)
            .unwrap_or(JS_SBGLS_DIV);
        JS_SBGLS_DIV_HP.store(sbgls_div, std::sync::atomic::Ordering::Relaxed);
        part_b::HALT.store(false, std::sync::atomic::Ordering::Relaxed);
        part_b::ARMED.store(false, std::sync::atomic::Ordering::Relaxed);
        part_b::ENV_PENDING.store(false, std::sync::atomic::Ordering::Relaxed);
        part_b::STOP.store(0, std::sync::atomic::Ordering::Relaxed);
        part_b::XS.store(0, std::sync::atomic::Ordering::Relaxed);
        part_b::set_job_salt(0);
        let pre = part_b::Pre::build(challenge);
        let kind = part_b::aim::kind(&pre);
        let mut goal = part_b::aim::Aim::read(kind, hyperparameters);
        let plan = part_b::aim::plan(&pre, &challenge.seed, &mut goal);
        let stop_at = match plan {
            part_b::aim::Plan::StopAt(m) => m,
            _ => 0,
        };
        if matches!(plan, part_b::aim::Plan::GiveUp) {
            let best_mk = std::cell::Cell::new(u32::MAX);
            let guarded = |sol: &Solution| -> Result<()> {
                if let Ok(mk) = checked_makespan(challenge, sol) {
                    if mk < best_mk.get() {
                        best_mk.set(mk);
                        save_solution(sol)?;
                    }
                }
                Ok(())
            };
            let _ = part_b::gate_floor::gate_floor(&pre, &challenge.seed, &guarded);
            return Ok(());
        }
        if let part_b::aim::Plan::StopAt(_) = &plan {
            part_b::STOP.store(stop_at, std::sync::atomic::Ordering::Relaxed);
            let margin = hyperparameters
                .as_ref()
                .and_then(|hp| hp.get("js_env_margin"))
                .and_then(Value::as_f64)
                .unwrap_or(0.004);
            part_b::arm_env_js(&goal, margin);
        }
        let best_mk = std::cell::Cell::new(u32::MAX);
        let stopped = std::cell::Cell::new(false);
        let guarded = |sol: &Solution| -> Result<()> {
            if stopped.get() {
                return Err(anyhow::anyhow!("stop"));
            }
            let mk = checked_makespan(challenge, sol)?;
            if mk <= best_mk.get() {
                best_mk.set(mk);
                part_b::note_best(mk);
                save_solution(sol)?;
                if mk <= stop_at || part_b::halted() {
                    stopped.set(true);
                    part_b::HALT.store(true, std::sync::atomic::Ordering::Relaxed);
                    return Err(anyhow::anyhow!("stop"));
                }
            }
            Ok(())
        };
        let _ = part_b::job_shop_engine::solver::solve_challenge(challenge, &guarded);
        part_b::STOP.store(0, std::sync::atomic::Ordering::Relaxed);
        part_b::ARMED.store(false, std::sync::atomic::Ordering::Relaxed);
        part_b::ENV_PENDING.store(false, std::sync::atomic::Ordering::Relaxed);
        part_b::HALT.store(false, std::sync::atomic::Ordering::Relaxed);
        return Ok(());
    }
    match s {
        "hybrid_flow_shop" => part_b::solve_challenge(challenge, save_solution, hyperparameters),
        _ => part_a::solve_challenge(challenge, save_solution, hyperparameters),
    }
}

#[allow(dead_code)]
pub fn help() {
    part_a::help();
    part_b::help();
}
