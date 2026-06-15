use serde::{Deserialize, Serialize};
#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {}

use tig_challenges::job_scheduling::*;
use anyhow::Result;
use serde_json::{Map, Value};
use std::cell::RefCell;
use std::collections::VecDeque;

pub mod types;
pub mod preprocess;
mod infra_shared;
pub mod solver;
pub mod flow_shop;
pub mod hybrid_flow_shop;
pub mod job_shop;
pub mod fjsp_medium;
pub mod fjsp_high;

use preprocess::build_pre;
use types::{EffortConfig, Pre};

pub use solver::help;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DispatchTrack {
    FlowShop,
    HybridFlowShop,
    JobShop,
    FjspMedium,
    FjspHigh,
}

#[derive(Clone, Copy)]
enum MachineOrderVariant {
    RawStart,
    ShortProcFirst,
    TailFirst,
}

fn requested_track(hyperparameters: &Option<Map<String, Value>>) -> Option<DispatchTrack> {
    let Some(map) = hyperparameters else { return None };
    let Some(Value::String(s)) = map.get("track") else { return None };
    match s.to_lowercase().as_str() {
        "flow_shop" | "flow" => Some(DispatchTrack::FlowShop),
        "hybrid_flow_shop" | "hybrid" => Some(DispatchTrack::HybridFlowShop),
        "job_shop" | "job" => Some(DispatchTrack::JobShop),
        "fjsp_medium" | "medium" => Some(DispatchTrack::FjspMedium),
        "fjsp_high" | "high" | "fjsp" => Some(DispatchTrack::FjspHigh),
        _ => None,
    }
}

fn parse_effort(hyperparameters: &Option<Map<String, Value>>) -> EffortConfig {
    let mut cfg = EffortConfig::default_effort();
    if let Some(map) = hyperparameters {
        if let Some(Value::Number(n)) = map.get("job_shop_iters") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_job_shop_iters(v as usize);
            }
        }
    }
    cfg
}

fn infer_track(pre: &Pre) -> DispatchTrack {
    if pre.strict_route.is_some() {
        return DispatchTrack::FlowShop;
    }
    if pre.flex_avg <= 1.35 {
        return DispatchTrack::JobShop;
    }
    if pre.flex_avg >= 6.0 {
        return DispatchTrack::FjspHigh;
    }
    if pre.flow_like >= 0.72 && pre.jobshopness <= 0.34 {
        return DispatchTrack::HybridFlowShop;
    }
    DispatchTrack::FjspMedium
}

fn backup_tracks(challenge: &Challenge, pre: &Pre, primary: DispatchTrack) -> Vec<DispatchTrack> {
    let single_product = challenge.product_processing_times.len() == 1;
    match primary {
        DispatchTrack::FlowShop => {
            if !single_product || pre.jobshopness >= 0.50 || pre.high_flex >= 0.12 {
                vec![DispatchTrack::JobShop]
            } else {
                Vec::new()
            }
        }
        DispatchTrack::HybridFlowShop => {
            let mut backups = Vec::with_capacity(2);
            if pre.jobshopness >= 0.54 && pre.flex_avg <= 3.5 {
                backups.push(DispatchTrack::JobShop);
            }
            if pre.high_flex >= 0.34 || pre.flex_avg >= 3.1 {
                backups.push(DispatchTrack::FjspMedium);
            }
            backups
        }
        DispatchTrack::JobShop => {
            let mut backups = Vec::with_capacity(2);
            if pre.flex_avg >= 1.5 || pre.high_flex >= 0.10 {
                backups.push(DispatchTrack::FjspMedium);
            }
            if pre.high_flex >= 0.24 || pre.flex_avg >= 2.6 {
                backups.push(DispatchTrack::FjspHigh);
            }
            backups
        }
        DispatchTrack::FjspMedium => {
            if pre.jobshopness >= 0.44 && pre.flex_avg <= 4.2 && !pre.chaotic_like {
                vec![DispatchTrack::JobShop]
            } else if pre.high_flex >= 0.55 || pre.flex_avg >= 5.2 {
                vec![DispatchTrack::FjspHigh]
            } else {
                Vec::new()
            }
        }
        DispatchTrack::FjspHigh => {
            if pre.jobshopness >= 0.52 && pre.flex_avg <= 4.4 && !pre.chaotic_like {
                vec![DispatchTrack::JobShop, DispatchTrack::FjspMedium]
            } else if pre.flow_like >= 0.84 && pre.high_flex <= 0.40 {
                vec![DispatchTrack::FjspMedium]
            } else {
                Vec::new()
            }
        }
    }
}

fn clamp01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

fn closeness(x: f64, center: f64, width: f64) -> f64 {
    if width <= 0.0 {
        return 0.0;
    }
    clamp01(1.0 - (x - center).abs() / width)
}

fn track_score(challenge: &Challenge, pre: &Pre, track: DispatchTrack) -> f64 {
    let flow = pre.flow_like;
    let job = pre.jobshopness;
    let flex = pre.flex_avg;
    let high = pre.high_flex;
    let strict_bonus = if pre.strict_route.is_some() && matches!(track, DispatchTrack::FlowShop) {
        4.0
    } else {
        0.0
    };
    let flow_route_bonus = if pre.flow_route.is_some() && matches!(track, DispatchTrack::FlowShop | DispatchTrack::HybridFlowShop) {
        0.6
    } else {
        0.0
    };
    let size_bias = if challenge.num_jobs <= 48 { 0.15 } else { 0.0 };

    let score = match track {
        DispatchTrack::FlowShop => {
            2.2 * closeness(flow, 0.95, 0.28)
                + 0.9 * closeness(job, 0.0, 0.35)
                - 1.4 * high
                - 0.18 * (flex - 1.0).max(0.0)
        }
        DispatchTrack::HybridFlowShop => {
            1.9 * closeness(flow, 0.82, 0.24)
                + 1.1 * closeness(job, 0.28, 0.22)
                + 0.8 * closeness(high, 0.16, 0.18)
                - 0.08 * (flex - 3.0).abs()
        }
        DispatchTrack::JobShop => {
            2.3 * closeness(job, 0.82, 0.32)
                + 1.1 * closeness(flex, 1.15, 1.9)
                + 0.5 * (1.0 - high)
                - 0.25 * flow
        }
        DispatchTrack::FjspMedium => {
            1.8 * closeness(flex, 3.0, 2.1)
                + 1.0 * closeness(high, 0.30, 0.28)
                + 0.5 * closeness(job, 0.42, 0.28)
        }
        DispatchTrack::FjspHigh => {
            2.4 * closeness(flex, 6.2, 4.2)
                + 1.5 * closeness(high, 0.70, 0.35)
                + 0.6 * job
        }
    };

    score + strict_bonus + flow_route_bonus + size_bias
}

fn ranked_tracks(challenge: &Challenge, pre: &Pre) -> Vec<DispatchTrack> {
    let mut tracks = vec![
        DispatchTrack::FlowShop,
        DispatchTrack::HybridFlowShop,
        DispatchTrack::JobShop,
        DispatchTrack::FjspMedium,
        DispatchTrack::FjspHigh,
    ];
    tracks.sort_unstable_by(|&a, &b| {
        track_score(challenge, pre, b)
            .partial_cmp(&track_score(challenge, pre, a))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| (a as u8).cmp(&(b as u8)))
    });
    tracks
}

fn scaled_effort(effort: &EffortConfig, primary: DispatchTrack, backup: DispatchTrack) -> EffortConfig {
    let mut scaled = *effort;
    let (num, den) = match (primary, backup) {
        (DispatchTrack::JobShop, DispatchTrack::FjspMedium) => (40usize, 100usize),
        (DispatchTrack::JobShop, DispatchTrack::FjspHigh) => (30usize, 100usize),
        (DispatchTrack::FjspMedium, DispatchTrack::JobShop) | (DispatchTrack::FjspHigh, DispatchTrack::JobShop) => (65usize, 100usize),
        (DispatchTrack::FlowShop, DispatchTrack::JobShop) | (DispatchTrack::HybridFlowShop, DispatchTrack::JobShop) => (65usize, 100usize),
        _ => (35usize, 100usize),
    };
    let scale = |v: usize| -> usize { ((v.saturating_mul(num)).saturating_add(den - 1)) / den };
    scaled.job_shop_iters = scale(scaled.job_shop_iters).max(100);
    scaled.hybrid_flow_shop_iters = scale(scaled.hybrid_flow_shop_iters).max(100);
    scaled.fjsp_medium_iters = scale(scaled.fjsp_medium_iters).max(100);
    scaled.fjsp_high_iters = scale(scaled.fjsp_high_iters).max(100);
    scaled
}

fn scaled_effort_ratio(effort: &EffortConfig, num: usize, den: usize) -> EffortConfig {
    let mut scaled = *effort;
    let scale = |v: usize| -> usize { ((v.saturating_mul(num)).saturating_add(den - 1)) / den };
    scaled.job_shop_iters = scale(scaled.job_shop_iters).max(100);
    scaled.hybrid_flow_shop_iters = scale(scaled.hybrid_flow_shop_iters).max(100);
    scaled.fjsp_medium_iters = scale(scaled.fjsp_medium_iters).max(100);
    scaled.fjsp_high_iters = scale(scaled.fjsp_high_iters).max(100);
    scaled
}

fn tuned_job_shop_effort(pre: &Pre, effort: &EffortConfig, primary: bool) -> EffortConfig {
    let mut tuned = *effort;
    let jobshopness = pre.jobshopness;
    let flex = pre.flex_avg;
    let boost = if jobshopness >= 0.60 || flex <= 1.55 {
        if primary { 150usize } else { 122usize }
    } else if jobshopness >= 0.42 || flex <= 2.10 {
        if primary { 132usize } else { 108usize }
    } else if pre.high_flex >= 0.35 {
        if primary { 88usize } else { 74usize }
    } else if primary {
        100usize
    } else {
        84usize
    };
    tuned.job_shop_iters = ((tuned.job_shop_iters.saturating_mul(boost)).saturating_add(99)) / 100;
    tuned.job_shop_iters = tuned.job_shop_iters.max(100);
    tuned
}

fn candidate_tracks(challenge: &Challenge, pre: &Pre) -> Vec<DispatchTrack> {
    let ranked = ranked_tracks(challenge, pre);
    let primary = ranked[0];
    let inferred = infer_track(pre);
    let mut tracks = Vec::with_capacity(3);

    tracks.push(primary);
    if inferred != primary {
        tracks.push(inferred);
    }

    let jobshop_like = pre.strict_route.is_none()
        && (pre.jobshopness >= 0.46 || (pre.flex_avg <= 2.0 && pre.high_flex <= 0.25));
    if jobshop_like && !tracks.contains(&DispatchTrack::JobShop) {
        tracks.push(DispatchTrack::JobShop);
    }

    for track in backup_tracks(challenge, pre, primary) {
        if !tracks.contains(&track) {
            tracks.push(track);
        }
        if tracks.len() >= 3 {
            return tracks;
        }
    }

    for &track in ranked.iter().skip(1) {
        if !tracks.contains(&track) {
            tracks.push(track);
        }
        if tracks.len() >= 3 {
            break;
        }
    }

    tracks
}

fn run_selected_track(
    track: DispatchTrack,
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    pre: &Pre,
    effort: &EffortConfig,
    job_shop_primary: bool,
) -> Result<()> {
    match track {
        DispatchTrack::FlowShop => flow_shop::solve(challenge, save_solution, pre, effort),
        DispatchTrack::HybridFlowShop => hybrid_flow_shop::solve(challenge, save_solution, pre, effort),
        DispatchTrack::JobShop => {
            let tuned = tuned_job_shop_effort(pre, effort, job_shop_primary);
            job_shop::solve(challenge, save_solution, pre, &tuned)
        }
        DispatchTrack::FjspMedium => fjsp_medium::solve(challenge, save_solution, pre, effort),
        DispatchTrack::FjspHigh => fjsp_high::solve(challenge, save_solution, pre, effort),
    }
}

#[derive(Clone, Copy)]
struct ScheduledOp {
    job: usize,
    op: usize,
    machine: usize,
    start: u32,
    duration: u32,
    tail: u32,
}

fn job_products(challenge: &Challenge) -> Vec<usize> {
    let mut products = vec![0usize; challenge.num_jobs];
    let mut job = 0usize;
    for (product, &count) in challenge.jobs_per_product.iter().enumerate() {
        for _ in 0..count {
            if job < products.len() {
                products[job] = product;
                job += 1;
            }
        }
    }
    products
}

fn compact_solution_variant(
    challenge: &Challenge,
    sol: &Solution,
    variant: MachineOrderVariant,
) -> Result<Option<Solution>> {
    if sol.job_schedule.len() != challenge.num_jobs {
        return Err(anyhow::anyhow!("solution job count mismatch"));
    }

    let products = job_products(challenge);
    let mut ops = Vec::new();
    let mut op_index_by_job: Vec<Vec<usize>> = Vec::with_capacity(challenge.num_jobs);
    let mut by_machine: Vec<Vec<usize>> = vec![Vec::new(); challenge.num_machines];

    for (job, job_sched) in sol.job_schedule.iter().enumerate() {
        let product = products[job];
        let route = &challenge.product_processing_times[product];
        if job_sched.len() != route.len() {
            return Err(anyhow::anyhow!("job {} operation count mismatch", job));
        }

        let mut job_indices = Vec::with_capacity(job_sched.len());
        let mut job_durations = Vec::with_capacity(job_sched.len());
        for (op, &(machine, start)) in job_sched.iter().enumerate() {
            let Some(&duration) = route[op].get(&machine) else {
                return Err(anyhow::anyhow!("job {} op {} uses ineligible machine", job, op));
            };
            if machine >= challenge.num_machines {
                return Err(anyhow::anyhow!("job {} op {} has invalid machine", job, op));
            }
            let idx = ops.len();
            ops.push(ScheduledOp {
                job,
                op,
                machine,
                start,
                duration,
                tail: 0,
            });
            job_indices.push(idx);
            job_durations.push(duration);
            by_machine[machine].push(idx);
        }

        let mut remaining = 0u32;
        for (local_op, &idx) in job_indices.iter().enumerate().rev() {
            remaining = remaining.saturating_add(job_durations[local_op]);
            ops[idx].tail = remaining;
        }
        op_index_by_job.push(job_indices);
    }

    if ops.is_empty() {
        return Ok(Some(sol.clone()));
    }

    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); ops.len()];
    let mut indegree = vec![0usize; ops.len()];

    for job_ops in &op_index_by_job {
        for pair in job_ops.windows(2) {
            successors[pair[0]].push(pair[1]);
            indegree[pair[1]] += 1;
        }
    }

    for machine_ops in &mut by_machine {
        match variant {
            MachineOrderVariant::RawStart => {
                machine_ops.sort_unstable_by_key(|&idx| (ops[idx].start, ops[idx].job, ops[idx].op));
            }
            MachineOrderVariant::ShortProcFirst => {
                machine_ops.sort_unstable_by_key(|&idx| {
                    (ops[idx].duration, ops[idx].start, ops[idx].job, ops[idx].op)
                });
            }
            MachineOrderVariant::TailFirst => {
                machine_ops.sort_unstable_by_key(|&idx| {
                    (
                        std::cmp::Reverse(ops[idx].tail),
                        ops[idx].start,
                        ops[idx].job,
                        ops[idx].op,
                    )
                });
            }
        }
        for pair in machine_ops.windows(2) {
            successors[pair[0]].push(pair[1]);
            indegree[pair[1]] += 1;
        }
    }

    let mut ready_time = vec![0u32; ops.len()];
    let mut compact_start = vec![0u32; ops.len()];
    let mut queue = VecDeque::with_capacity(ops.len());
    for (idx, &deg) in indegree.iter().enumerate() {
        if deg == 0 {
            queue.push_back(idx);
        }
    }

    let mut seen = 0usize;
    while let Some(idx) = queue.pop_front() {
        seen += 1;
        let finish = ready_time[idx].saturating_add(ops[idx].duration);
        compact_start[idx] = ready_time[idx];
        for &next in &successors[idx] {
            ready_time[next] = ready_time[next].max(finish);
            indegree[next] -= 1;
            if indegree[next] == 0 {
                queue.push_back(next);
            }
        }
    }

    if seen != ops.len() {
        return Ok(None);
    }

    let mut job_schedule: Vec<Vec<(usize, u32)>> = op_index_by_job
        .iter()
        .map(|job_ops| vec![(0usize, 0u32); job_ops.len()])
        .collect();
    for (idx, op) in ops.iter().enumerate() {
        job_schedule[op.job][op.op] = (op.machine, compact_start[idx]);
    }

    Ok(Some(Solution { job_schedule }))
}

fn compact_solution(challenge: &Challenge, sol: &Solution) -> Result<Solution> {
    let mut current = sol.clone();
    let mut current_mk = challenge.evaluate_makespan(&current)?;

    for _ in 0..2 {
        let baseline = compact_solution_variant(challenge, &current, MachineOrderVariant::RawStart)?
            .unwrap_or_else(|| current.clone());
        let mut best = (challenge.evaluate_makespan(&baseline)?, baseline);

        for variant in [MachineOrderVariant::ShortProcFirst, MachineOrderVariant::TailFirst] {
            let Some(candidate) = compact_solution_variant(challenge, &current, variant)? else {
                continue;
            };
            let makespan = challenge.evaluate_makespan(&candidate)?;
            if makespan < best.0 {
                best = (makespan, candidate);
            }
        }

        if best.0 < current_mk {
            current_mk = best.0;
            current = best.1;
        } else {
            break;
        }
    }

    Ok(current)
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let pre = build_pre(challenge)?;
    let effort = parse_effort(hyperparameters);
    let best_makespan = RefCell::new(None::<u32>);
    let best_solution = RefCell::new(None::<Solution>);
    let guarded_save = |sol: &Solution| -> Result<()> {
        let compacted = compact_solution(challenge, sol)?;
        let makespan = challenge.evaluate_makespan(&compacted)?;
        let improved = {
            let mut best = best_makespan.borrow_mut();
            if best.map_or(true, |cur| makespan < cur) {
                *best = Some(makespan);
                true
            } else {
                false
            }
        };
        if improved {
            *best_solution.borrow_mut() = Some(compacted.clone());
            save_solution(&compacted)?;
        }
        Ok(())
    };

    if let Some(track) = requested_track(hyperparameters) {
        run_selected_track(track, challenge, &guarded_save, &pre, &effort, track == DispatchTrack::JobShop)?;
    } else {
        let candidates = candidate_tracks(challenge, &pre);
        if let Some(&primary) = candidates.first() {
            run_selected_track(primary, challenge, &guarded_save, &pre, &effort, primary == DispatchTrack::JobShop)?;
        }
        if let Some(&backup) = candidates.get(1) {
            let backup_effort = scaled_effort(&effort, candidates[0], backup);
            run_selected_track(backup, challenge, &guarded_save, &pre, &backup_effort, false)?;
        }
        if let Some(&third) = candidates.get(2) {
            let third_effort = if third == DispatchTrack::JobShop {
                scaled_effort_ratio(&effort, 50, 100)
            } else {
                scaled_effort_ratio(&effort, 35, 100)
            };
            run_selected_track(third, challenge, &guarded_save, &pre, &third_effort, false)?;
        }
        if !candidates.contains(&DispatchTrack::FjspHigh)
            && (pre.high_flex >= 0.38 || pre.flex_avg >= 4.0)
            && pre.strict_route.is_none()
        {
            let rescue_effort = scaled_effort_ratio(&effort, 22, 100);
            run_selected_track(DispatchTrack::FjspHigh, challenge, &guarded_save, &pre, &rescue_effort, false)?;
        }

        if !candidates.contains(&DispatchTrack::JobShop)
            && pre.strict_route.is_none()
            && (pre.jobshopness >= 0.35 || (pre.flex_avg <= 2.5 && pre.high_flex <= 0.35))
        {
            let rescue_effort = scaled_effort_ratio(
                &effort,
                if pre.jobshopness >= 0.55 || pre.flex_avg <= 2.0 { 30 } else { 22 },
                100,
            );
            run_selected_track(DispatchTrack::JobShop, challenge, &guarded_save, &pre, &rescue_effort, false)?;
        }

    if pre.strict_route.is_none()
        && pre.jobshopness >= 0.55
        && candidates.first() != Some(&DispatchTrack::JobShop)
        && !candidates.contains(&DispatchTrack::JobShop)
    {
            let rescue_effort = scaled_effort_ratio(&effort, 35, 100);
            run_selected_track(DispatchTrack::JobShop, challenge, &guarded_save, &pre, &rescue_effort, false)?;
        }

        if let Some(&primary) = candidates.first() {
            let replay_effort = scaled_effort_ratio(&effort, 60, 100);
            run_selected_track(primary, challenge, &guarded_save, &pre, &replay_effort, primary == DispatchTrack::JobShop)?;
        }
    }

    if let Some(sol) = best_solution.borrow().as_ref() {
        save_solution(sol)?;
    }
    Ok(())
}

pub fn help() {
    println!("Prometheus solver");
}
