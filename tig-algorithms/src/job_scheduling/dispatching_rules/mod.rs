// Dispatching-rules baseline with adaptive weighted search and soft machine selection.
use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::cmp::Ordering;
use std::collections::HashMap;
use tig_challenges::job_scheduling::*;

#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct Hyperparameters {
    pub effort: usize,
    pub flow: Option<String>,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            effort: 5,
            flow: Some("chaotic".to_string()),
        }
    }
}

pub fn help() {
    println!("Dispatching-rules baseline with weighted search and soft machine selection.");
    println!("Hyperparameters (all optional):");
    println!("  effort: baseline effort (default 5).");
    println!("  flow: flow hint (strict | parallel | random | complex | chaotic).");
    println!("        default is chaotic; strict/parallel enable weighted search and soft machine selection.");
}

const WORK_MIN_WEIGHT: f64 = 0.3;
const MAX_SLACK_RATIO: f64 = 0.5;

#[derive(Clone, Copy)]
enum FlowHint {
    Strict,
    Parallel,
}

#[derive(Clone, Copy)]
struct RuleWeights {
    work: f64,
    ops: f64,
    flex: f64,
    end: f64,
    proc: f64,
}

impl Default for RuleWeights {
    fn default() -> Self {
        Self {
            work: 1.0,
            ops: 0.2,
            flex: 0.4,
            end: 0.9,
            proc: 0.2,
        }
    }
}

#[derive(Clone, Copy)]
struct PriorityScales {
    work: f64,
    ops: f64,
    proc: f64,
    end: f64,
}

#[derive(Clone, Copy)]
struct DispatchParams {
    rule: DispatchRule,
    weights: RuleWeights,
    slack_ratio: f64,
}

fn average_processing_time(operation: &HashMap<usize, u32>) -> f64 {
    if operation.is_empty() {
        return 0.0;
    }
    let sum: u32 = operation.values().sum();
    sum as f64 / operation.len() as f64
}

fn min_processing_time(operation: &HashMap<usize, u32>) -> f64 {
    operation.values().copied().min().unwrap_or(0) as f64
}

fn flow_hint_from_hyper(flow: &Option<String>) -> Result<Option<FlowHint>> {
    let Some(flow) = flow else {
        return Ok(None);
    };

    let parsed = flow.parse::<Flow>()?;
    Ok(match parsed {
        Flow::STRICT => Some(FlowHint::Strict),
        Flow::PARALLEL => Some(FlowHint::Parallel),
        Flow::RANDOM | Flow::COMPLEX | Flow::CHAOTIC => None,
    })
}

fn work_min_weight(flow_hint: FlowHint) -> f64 {
    match flow_hint {
        FlowHint::Strict => WORK_MIN_WEIGHT,
        FlowHint::Parallel => 0.15,
    }
}

fn base_weights(flow_hint: FlowHint) -> RuleWeights {
    match flow_hint {
        FlowHint::Strict => RuleWeights {
            work: 1.2,
            ops: 0.4,
            flex: 0.0,
            end: 1.3,
            proc: 0.3,
        },
        FlowHint::Parallel => RuleWeights {
            work: 1.0,
            ops: 0.2,
            flex: 0.8,
            end: 0.8,
            proc: 0.2,
        },
    }
}

fn build_product_work_times(
    product_processing_times: &[Vec<HashMap<usize, u32>>],
    work_min_weight: f64,
) -> Vec<Vec<f64>> {
    let mut product_work_times = Vec::with_capacity(product_processing_times.len());
    for product_ops in product_processing_times.iter() {
        let mut work_ops = Vec::with_capacity(product_ops.len());
        for op in product_ops.iter() {
            let avg = average_processing_time(op);
            let min = min_processing_time(op);
            let work = avg * (1.0 - work_min_weight) + min * work_min_weight;
            work_ops.push(work);
        }
        product_work_times.push(work_ops);
    }
    product_work_times
}

fn build_job_total_work(
    job_products: &[usize],
    product_work_times: &[Vec<f64>],
) -> Vec<f64> {
    let mut job_total_work: Vec<f64> = Vec::with_capacity(job_products.len());
    for &product in job_products.iter() {
        let work_ops = &product_work_times[product];
        job_total_work.push(work_ops.iter().sum());
    }
    job_total_work
}

fn jitter_weights(base: RuleWeights, rng: &mut SmallRng, jitter: f64) -> RuleWeights {
    let jitter = jitter.max(0.0).min(0.7);
    let mut scale = |value: f64| {
        if jitter == 0.0 {
            return value;
        }
        let factor = rng.gen_range(1.0 - jitter..=1.0 + jitter);
        (value * factor).max(0.0)
    };
    RuleWeights {
        work: scale(base.work),
        ops: scale(base.ops),
        flex: scale(base.flex),
        end: scale(base.end),
        proc: scale(base.proc),
    }
}

fn priority_scales(
    challenge: &Challenge,
    job_ops_len: &[usize],
    job_total_work: &[f64],
) -> PriorityScales {
    let work = job_total_work
        .iter()
        .copied()
        .fold(0.0, f64::max)
        .max(1.0);
    let ops = job_ops_len.iter().copied().max().unwrap_or(1) as f64;
    let mut max_proc = 1u32;
    for product_ops in challenge.product_processing_times.iter() {
        for op in product_ops.iter() {
            for &proc in op.values() {
                if proc > max_proc {
                    max_proc = proc;
                }
            }
        }
    }
    let proc = max_proc as f64;
    let end = (work + proc).max(1.0);
    PriorityScales {
        work,
        ops,
        proc,
        end,
    }
}

fn weighted_priority(
    remaining_work: f64,
    remaining_ops: usize,
    flexibility: usize,
    machine_end: u32,
    proc_time: u32,
    weights: RuleWeights,
    scales: PriorityScales,
) -> f64 {
    let flex = flexibility.max(1) as f64;
    let work_norm = remaining_work / scales.work;
    let ops_norm = remaining_ops as f64 / scales.ops;
    let flex_norm = 1.0 / flex;
    let end_norm = machine_end as f64 / scales.end;
    let proc_norm = proc_time as f64 / scales.proc;
    weights.work * work_norm
        + weights.ops * ops_norm
        + weights.flex * flex_norm
        - weights.end * end_norm
        - weights.proc * proc_norm
}

fn slack_allowance(proc_time: u32, slack_ratio: f64) -> u32 {
    if slack_ratio <= 0.0 {
        return 0;
    }
    let slack_ratio = slack_ratio.max(0.0).min(MAX_SLACK_RATIO);
    ((proc_time as f64) * slack_ratio).round() as u32
}

fn base_slack_ratio(flow_hint: FlowHint, effort: usize) -> f64 {
    match flow_hint {
        FlowHint::Strict => 0.0,
        FlowHint::Parallel => (0.18 + 0.03 * effort as f64).min(MAX_SLACK_RATIO),
    }
}

fn earliest_end_time(
    time: u32,
    machine_available_time: &[u32],
    operation: &HashMap<usize, u32>,
) -> u32 {
    let mut earliest_end = u32::MAX;
    for (&machine_id, &proc_time) in operation.iter() {
        let start = time.max(machine_available_time[machine_id]);
        let end = start + proc_time;
        if end < earliest_end {
            earliest_end = end;
        }
    }
    earliest_end
}

#[derive(Clone, Copy)]
enum DispatchRule {
    MostWorkRemaining,
    MostOpsRemaining,
    LeastFlexibility,
    ShortestProcTime,
    LongestProcTime,
    Weighted,
}

#[derive(Clone, Copy)]
struct Candidate {
    job: usize,
    priority: f64,
    machine_end: u32,
    proc_time: u32,
    flexibility: usize,
}

struct ScheduleResult {
    job_schedule: Vec<Vec<(usize, u32)>>,
    makespan: u32,
}

struct RestartResult {
    makespan: u32,
    rule: DispatchRule,
    random_top_k: usize,
    seed: u64,
    weights: RuleWeights,
    slack_ratio: f64,
}

fn better_candidate(candidate: &Candidate, best: &Candidate, eps: f64) -> bool {
    if candidate.priority > best.priority + eps {
        return true;
    }
    if (candidate.priority - best.priority).abs() <= eps {
        if candidate.machine_end < best.machine_end {
            return true;
        }
        if candidate.machine_end == best.machine_end {
            if candidate.proc_time < best.proc_time {
                return true;
            }
            if candidate.proc_time == best.proc_time {
                if candidate.flexibility < best.flexibility {
                    return true;
                }
                if candidate.flexibility == best.flexibility && candidate.job < best.job {
                    return true;
                }
            }
        }
    }
    false
}

fn run_dispatch_rule(
    challenge: &Challenge,
    job_products: &[usize],
    product_work_times: &[Vec<f64>],
    job_ops_len: &[usize],
    job_total_work: &[f64],
    params: DispatchParams,
    scales: PriorityScales,
    random_top_k: Option<usize>,
    rng: Option<&mut SmallRng>,
) -> Result<ScheduleResult> {
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;

    let mut job_next_op_idx = vec![0usize; num_jobs];
    let mut job_ready_time = vec![0u32; num_jobs];
    let mut machine_available_time = vec![0u32; num_machines];
    let mut job_schedule = job_ops_len
        .iter()
        .map(|&ops_len| Vec::with_capacity(ops_len))
        .collect::<Vec<_>>();
    let mut job_remaining_work = job_total_work.to_vec();

    let mut remaining_ops = job_ops_len.iter().sum::<usize>();
    let mut time = 0u32;
    let eps = 1e-9_f64;
    let random_top_k = random_top_k.unwrap_or(0);
    let mut rng = rng;
    let use_random = random_top_k > 1 && rng.is_some();

    while remaining_ops > 0 {
        let mut available_machines = (0..num_machines)
            .filter(|&m| machine_available_time[m] <= time)
            .collect::<Vec<usize>>();
        available_machines.sort_unstable();
        if use_random {
            available_machines.shuffle(rng.as_mut().unwrap());
        }

        let mut scheduled_any = false;
        for &machine in available_machines.iter() {
            let mut best_candidate: Option<Candidate> = None;

            if use_random {
                let mut candidates: Vec<Candidate> = Vec::new();

                for job in 0..num_jobs {
                    if job_next_op_idx[job] >= job_ops_len[job] {
                        continue;
                    }
                    if job_ready_time[job] > time {
                        continue;
                    }

                    let product = job_products[job];
                    let op_idx = job_next_op_idx[job];
                    let op_times = &challenge.product_processing_times[product][op_idx];
                    let proc_time = match op_times.get(&machine) {
                        Some(&value) => value,
                        None => continue,
                    };

                    let earliest_end = earliest_end_time(time, &machine_available_time, op_times);
                    let machine_end = time.max(machine_available_time[machine]) + proc_time;
                    let slack = slack_allowance(proc_time, params.slack_ratio);
                    if machine_end > earliest_end.saturating_add(slack) {
                        continue;
                    }

                    let flexibility = op_times.len();
                    let remaining_ops = job_ops_len[job] - job_next_op_idx[job];
                    let priority = match params.rule {
                        DispatchRule::MostWorkRemaining => job_remaining_work[job],
                        DispatchRule::MostOpsRemaining => remaining_ops as f64,
                        DispatchRule::LeastFlexibility => -(flexibility as f64),
                        DispatchRule::ShortestProcTime => -(proc_time as f64),
                        DispatchRule::LongestProcTime => proc_time as f64,
                        DispatchRule::Weighted => weighted_priority(
                            job_remaining_work[job],
                            remaining_ops,
                            flexibility,
                            machine_end,
                            proc_time,
                            params.weights,
                            scales,
                        ),
                    };

                    candidates.push(Candidate {
                        job,
                        priority,
                        machine_end,
                        proc_time,
                        flexibility,
                    });
                }

                if !candidates.is_empty() {
                    candidates.sort_by(|a, b| {
                        let ord = b
                            .priority
                            .partial_cmp(&a.priority)
                            .unwrap_or(Ordering::Equal);
                        if ord != Ordering::Equal {
                            return ord;
                        }
                        let ord = a.machine_end.cmp(&b.machine_end);
                        if ord != Ordering::Equal {
                            return ord;
                        }
                        let ord = a.proc_time.cmp(&b.proc_time);
                        if ord != Ordering::Equal {
                            return ord;
                        }
                        let ord = a.flexibility.cmp(&b.flexibility);
                        if ord != Ordering::Equal {
                            return ord;
                        }
                        a.job.cmp(&b.job)
                    });
                    let k = random_top_k.min(candidates.len());
                    let pick = rng.as_mut().unwrap().gen_range(0..k);
                    best_candidate = Some(candidates[pick]);
                }
            } else {
                for job in 0..num_jobs {
                    if job_next_op_idx[job] >= job_ops_len[job] {
                        continue;
                    }
                    if job_ready_time[job] > time {
                        continue;
                    }

                    let product = job_products[job];
                    let op_idx = job_next_op_idx[job];
                    let op_times = &challenge.product_processing_times[product][op_idx];
                    let proc_time = match op_times.get(&machine) {
                        Some(&value) => value,
                        None => continue,
                    };

                    let earliest_end = earliest_end_time(time, &machine_available_time, op_times);
                    let machine_end = time.max(machine_available_time[machine]) + proc_time;
                    let slack = slack_allowance(proc_time, params.slack_ratio);
                    if machine_end > earliest_end.saturating_add(slack) {
                        continue;
                    }

                    let flexibility = op_times.len();
                    let remaining_ops = job_ops_len[job] - job_next_op_idx[job];
                    let priority = match params.rule {
                        DispatchRule::MostWorkRemaining => job_remaining_work[job],
                        DispatchRule::MostOpsRemaining => remaining_ops as f64,
                        DispatchRule::LeastFlexibility => -(flexibility as f64),
                        DispatchRule::ShortestProcTime => -(proc_time as f64),
                        DispatchRule::LongestProcTime => proc_time as f64,
                        DispatchRule::Weighted => weighted_priority(
                            job_remaining_work[job],
                            remaining_ops,
                            flexibility,
                            machine_end,
                            proc_time,
                            params.weights,
                            scales,
                        ),
                    };

                    let candidate = Candidate {
                        job,
                        priority,
                        machine_end,
                        proc_time,
                        flexibility,
                    };

                    if best_candidate
                        .as_ref()
                        .map_or(true, |best| better_candidate(&candidate, best, eps))
                    {
                        best_candidate = Some(candidate);
                    }
                }
            }

            if let Some(candidate) = best_candidate {
                let job = candidate.job;
                let product = job_products[job];
                let op_idx = job_next_op_idx[job];
                let op_times = &challenge.product_processing_times[product][op_idx];
                let proc_time = op_times[&machine];

                let start_time = time.max(machine_available_time[machine]);
                let end_time = start_time + proc_time;

                job_schedule[job].push((machine, start_time));
                job_next_op_idx[job] += 1;
                job_ready_time[job] = end_time;
                machine_available_time[machine] = end_time;
                job_remaining_work[job] -= product_work_times[product][op_idx];
                if job_remaining_work[job] < 0.0 {
                    job_remaining_work[job] = 0.0;
                }

                remaining_ops -= 1;
                scheduled_any = true;
            }
        }

        if remaining_ops == 0 {
            break;
        }

        // Compute next event time (either machine becoming available or job becoming ready)
        let mut next_time: Option<u32> = None;
        for &t in machine_available_time.iter() {
            if t > time {
                next_time = Some(next_time.map_or(t, |best| best.min(t)));
            }
        }
        for job in 0..num_jobs {
            if job_next_op_idx[job] < job_ops_len[job] && job_ready_time[job] > time {
                let t = job_ready_time[job];
                next_time = Some(next_time.map_or(t, |best| best.min(t)));
            }
        }

        // Advance time to next event
        time = next_time.ok_or_else(|| {
            if scheduled_any {
                anyhow!("No next event time found while operations remain unscheduled")
            } else {
                anyhow!("No schedulable operations remain; dispatching rules stalled")
            }
        })?;
    }

    let makespan = job_ready_time.iter().copied().max().unwrap_or(0);
    Ok(ScheduleResult {
        job_schedule,
        makespan,
    })
}

fn run_baseline_search(
    challenge: &Challenge,
    save_best: &dyn Fn(&ScheduleResult) -> Result<()>,
    job_products: &[usize],
    product_work_times: &[Vec<f64>],
    job_ops_len: &[usize],
    job_total_work: &[f64],
    random_restarts: usize,
    top_k: usize,
    local_search_tries: usize,
) -> Result<ScheduleResult> {
    let scales = priority_scales(challenge, job_ops_len, job_total_work);
    let base_weights = RuleWeights::default();
    let rules = [
        DispatchRule::MostWorkRemaining,
        DispatchRule::MostOpsRemaining,
        DispatchRule::LeastFlexibility,
        DispatchRule::ShortestProcTime,
        DispatchRule::LongestProcTime,
    ];

    let mut best_result: Option<ScheduleResult> = None;
    for rule in rules.iter().copied() {
        let params = DispatchParams {
            rule,
            weights: base_weights,
            slack_ratio: 0.0,
        };
        let result = run_dispatch_rule(
            challenge,
            job_products,
            product_work_times,
            job_ops_len,
            job_total_work,
            params,
            scales,
            None,
            None,
        )?;
        let is_better = best_result
            .as_ref()
            .map_or(true, |best| result.makespan < best.makespan);
        if is_better {
            best_result = Some(result);
        }
    }

    let mut best_result = best_result.ok_or_else(|| anyhow!("No valid schedule produced"))?;
    save_best(&best_result)?;

    let mut top_restarts: Vec<RestartResult> = Vec::new();
    if random_restarts > 0 {
        let mut rng = SmallRng::from_seed(challenge.seed);
        for _ in 1..=random_restarts {
            let seed = rng.gen::<u64>();
            let rule = rules[rng.gen_range(0..rules.len())];
            let random_top_k = rng.gen_range(2..=5);
            let mut local_rng = SmallRng::seed_from_u64(seed);
            let params = DispatchParams {
                rule,
                weights: base_weights,
                slack_ratio: 0.0,
            };
            let result = run_dispatch_rule(
                challenge,
                job_products,
                product_work_times,
                job_ops_len,
                job_total_work,
                params,
                scales,
                Some(random_top_k),
                Some(&mut local_rng),
            )?;
            let makespan = result.makespan;
            let is_better = makespan < best_result.makespan;
            if is_better {
                best_result = result;
                save_best(&best_result)?;
            }

            if top_k > 0 {
                top_restarts.push(RestartResult {
                    makespan,
                    rule,
                    random_top_k,
                    seed,
                    weights: base_weights,
                    slack_ratio: 0.0,
                });
                top_restarts.sort_by(|a, b| a.makespan.cmp(&b.makespan));
                if top_restarts.len() > top_k {
                    top_restarts.pop();
                }
            }
        }
    }

    if !top_restarts.is_empty() {
        for restart in top_restarts.iter() {
            for attempt in 0..local_search_tries {
                let local_seed = restart.seed.wrapping_add(attempt as u64 + 1);
                let mut local_rng = SmallRng::seed_from_u64(local_seed);
                let local_k = match attempt % 3 {
                    0 => restart.random_top_k,
                    1 => restart.random_top_k.saturating_sub(1),
                    _ => restart.random_top_k.saturating_add(1),
                }
                .max(2);
                let params = DispatchParams {
                    rule: restart.rule,
                    weights: restart.weights,
                    slack_ratio: restart.slack_ratio,
                };
                let result = run_dispatch_rule(
                    challenge,
                    job_products,
                    product_work_times,
                    job_ops_len,
                    job_total_work,
                    params,
                    scales,
                    Some(local_k),
                    Some(&mut local_rng),
                )?;
                if result.makespan < best_result.makespan {
                    best_result = result;
                    save_best(&best_result)?;
                }
            }
        }
    }

    Ok(best_result)
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let hyperparameters = match hyperparameters {
        Some(hyperparameters) => {
            serde_json::from_value::<Hyperparameters>(Value::Object(hyperparameters.clone()))
                .map_err(|e| anyhow!("Failed to parse hyperparameters: {}", e))?
        }
        None => Hyperparameters::default(),
    };

    solve_challenge_with_settings(challenge, save_solution, &hyperparameters)
}

pub fn solve_challenge_with_effort(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    effort: usize,
) -> Result<()> {
    let hyperparameters = Hyperparameters {
        effort,
        ..Hyperparameters::default()
    };
    solve_challenge_with_settings(challenge, save_solution, &hyperparameters)
}

fn solve_challenge_with_settings(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Hyperparameters,
) -> Result<()> {
    let effort = hyperparameters.effort;
    let flow_hint = flow_hint_from_hyper(&hyperparameters.flow)?;
    let (random_restarts, top_k) = if effort == 0 {
        (10usize, 0usize)
    } else {
        let scaled_effort = effort.max(1);
        let random_restarts = 250usize.saturating_add(50usize.saturating_mul(scaled_effort));
        let top_k = if scaled_effort > 1 {
            2usize.saturating_add(2usize.saturating_mul(scaled_effort))
        } else {
            scaled_effort.saturating_add(1)
        };
        (random_restarts, top_k)
    };
    let local_search_tries = 1usize.saturating_add(3usize.saturating_mul(effort));
    solve_challenge_with_params(
        challenge,
        save_solution,
        random_restarts,
        top_k,
        local_search_tries,
        effort,
        flow_hint,
    )
}

fn solve_challenge_with_params(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    random_restarts: usize,
    top_k: usize,
    local_search_tries: usize,
    effort: usize,
    flow_hint: Option<FlowHint>,
) -> Result<()> {
    let save_best = |best: &ScheduleResult| -> Result<()> {
        save_solution(&Solution {
            job_schedule: best.job_schedule.clone(),
        })
    };
    let num_jobs = challenge.num_jobs;

    let mut job_products = Vec::with_capacity(num_jobs);
    for (product, count) in challenge.jobs_per_product.iter().enumerate() {
        for _ in 0..*count {
            job_products.push(product);
        }
    }
    if job_products.len() != num_jobs {
        return Err(anyhow!(
            "Job count mismatch. Expected {}, got {}",
            num_jobs,
            job_products.len()
        ));
    }

    let mut job_ops_len = Vec::with_capacity(num_jobs);
    for &product in job_products.iter() {
        job_ops_len.push(challenge.product_processing_times[product].len());
    }

    let baseline_product_work_times =
        build_product_work_times(&challenge.product_processing_times, WORK_MIN_WEIGHT);
    let baseline_job_total_work = build_job_total_work(&job_products, &baseline_product_work_times);
    let baseline_result = run_baseline_search(
        challenge,
        &save_best,
        &job_products,
        &baseline_product_work_times,
        &job_ops_len,
        &baseline_job_total_work,
        random_restarts,
        top_k,
        local_search_tries,
    )?;

    let Some(tuned_flow) = flow_hint else {
        save_solution(&Solution {
            job_schedule: baseline_result.job_schedule,
        })?;
        return Ok(());
    };

    let work_min_blend = work_min_weight(tuned_flow);
    let product_work_times =
        build_product_work_times(&challenge.product_processing_times, work_min_blend);
    let job_total_work = build_job_total_work(&job_products, &product_work_times);

    let scales = priority_scales(challenge, &job_ops_len, &job_total_work);
    let base_rule_weights = base_weights(tuned_flow);
    let slack_ratio = base_slack_ratio(tuned_flow, effort);
    let weight_jitter = (0.15 + 0.05 * effort as f64).min(0.6);

    let rules = [
        DispatchRule::MostWorkRemaining,
        DispatchRule::MostOpsRemaining,
        DispatchRule::LeastFlexibility,
        DispatchRule::ShortestProcTime,
        DispatchRule::LongestProcTime,
    ];

    let mut best_result: Option<ScheduleResult> = Some(baseline_result);
    for rule in rules.iter().copied() {
        let params = DispatchParams {
            rule,
            weights: base_rule_weights,
            slack_ratio,
        };
        let result = run_dispatch_rule(
            challenge,
            &job_products,
            &product_work_times,
            &job_ops_len,
            &job_total_work,
            params,
            scales,
            None,
            None,
        )?;
        let is_better = best_result
            .as_ref()
            .map_or(true, |best| result.makespan < best.makespan);
        if is_better {
            best_result = Some(result);
        }
    }

    let weighted_params = DispatchParams {
        rule: DispatchRule::Weighted,
        weights: base_rule_weights,
        slack_ratio,
    };
    let weighted_result = run_dispatch_rule(
        challenge,
        &job_products,
        &product_work_times,
        &job_ops_len,
        &job_total_work,
        weighted_params,
        scales,
        None,
        None,
    )?;
    let weighted_better = best_result
        .as_ref()
        .map_or(true, |best| weighted_result.makespan < best.makespan);
    if weighted_better {
        best_result = Some(weighted_result);
    }

    let mut best_result = best_result.ok_or_else(|| anyhow!("No valid schedule produced"))?;
    save_best(&best_result)?;

    let mut top_restarts: Vec<RestartResult> = Vec::new();

    if random_restarts > 0 {
        let mut rng = SmallRng::from_seed(challenge.seed);
        for _ in 1..=random_restarts {
            let seed = rng.gen::<u64>();
            let use_weighted = rng.gen_bool(0.5);
            let rule = if use_weighted {
                DispatchRule::Weighted
            } else {
                rules[rng.gen_range(0..rules.len())]
            };
            let weights = if use_weighted {
                jitter_weights(base_rule_weights, &mut rng, weight_jitter)
            } else {
                base_rule_weights
            };
            let base_top_k: usize = match tuned_flow {
                FlowHint::Strict => 2,
                FlowHint::Parallel => 3,
            };
            let max_top_k = base_top_k
                .saturating_add(effort)
                .saturating_add(2)
                .min(10);
            let random_top_k = rng.gen_range(base_top_k..=max_top_k.max(base_top_k));
            let restart_slack = slack_ratio;
            let mut local_rng = SmallRng::seed_from_u64(seed);
            let params = DispatchParams {
                rule,
                weights,
                slack_ratio: restart_slack,
            };
            let result = run_dispatch_rule(
                challenge,
                &job_products,
                &product_work_times,
                &job_ops_len,
                &job_total_work,
                params,
                scales,
                Some(random_top_k),
                Some(&mut local_rng),
            )?;
            let makespan = result.makespan;
            let is_better = makespan < best_result.makespan;
            if is_better {
                best_result = result;
                save_best(&best_result)?;
            }

            if top_k > 0 {
                top_restarts.push(RestartResult {
                    makespan,
                    rule,
                    random_top_k,
                    seed,
                    weights,
                    slack_ratio: restart_slack,
                });
                top_restarts.sort_by(|a, b| a.makespan.cmp(&b.makespan));
                if top_restarts.len() > top_k {
                    top_restarts.pop();
                }
            }
        }
    }

    if !top_restarts.is_empty() {
        for restart in top_restarts.iter() {
            for attempt in 0..local_search_tries {
                let local_seed = restart.seed.wrapping_add(attempt as u64 + 1);
                let mut local_rng = SmallRng::seed_from_u64(local_seed);
                let local_k = match attempt % 3 {
                    0 => restart.random_top_k,
                    1 => restart.random_top_k.saturating_sub(1),
                    _ => restart.random_top_k.saturating_add(1),
                }
                .max(2);
                let local_weights = if matches!(restart.rule, DispatchRule::Weighted) {
                    jitter_weights(restart.weights, &mut local_rng, weight_jitter * 0.5)
                } else {
                    restart.weights
                };
                let params = DispatchParams {
                    rule: restart.rule,
                    weights: local_weights,
                    slack_ratio: restart.slack_ratio,
                };
                let result = run_dispatch_rule(
                    challenge,
                    &job_products,
                    &product_work_times,
                    &job_ops_len,
                    &job_total_work,
                    params,
                    scales,
                    Some(local_k),
                    Some(&mut local_rng),
                )?;
                if result.makespan < best_result.makespan {
                    best_result = result;
                    save_best(&best_result)?;
                }
            }
        }
    }

    save_solution(&Solution {
        job_schedule: best_result.job_schedule,
    })?;
    Ok(())
}
