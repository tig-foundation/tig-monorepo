// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;
use tig_challenges::job_scheduling::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    // Optionally define hyperparameters here. Example:
    // pub param1: usize,
    // pub param2: f64,
}

pub fn help() {
    // Print help information about your algorithm here. It will be invoked with `help_algorithm` script
    println!("No help information provided.");
}

fn average_processing_time(operation: &HashMap<usize, u32>) -> f64 {
    if operation.is_empty() {
        return 0.0;
    }
    let sum: u32 = operation.values().sum();
    sum as f64 / operation.len() as f64
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
    ShortestProcTime,
    LongestProcTime,
}

struct ScheduleResult {
    job_schedule: Vec<Vec<(usize, u32)>>,
    makespan: u32,
}

fn run_dispatch_rule(
    challenge: &Challenge,
    job_products: &[usize],
    product_avg_times: &[Vec<f64>],
    job_ops_len: &[usize],
    job_total_work: &[f64],
    rule: DispatchRule,
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

    while remaining_ops > 0 {
        let mut available_machines = (0..num_machines)
            .filter(|&m| machine_available_time[m] <= time)
            .collect::<Vec<usize>>();
        available_machines.sort_unstable();

        let mut scheduled_any = false;
        for &machine in available_machines.iter() {
            let mut best_job: Option<usize> = None;
            let mut best_priority = f64::NEG_INFINITY;

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
                if machine_end != earliest_end {
                    continue;
                }

                let priority = match rule {
                    DispatchRule::MostWorkRemaining => job_remaining_work[job],
                    DispatchRule::MostOpsRemaining => {
                        (job_ops_len[job] - job_next_op_idx[job]) as f64
                    }
                    DispatchRule::ShortestProcTime => -(proc_time as f64),
                    DispatchRule::LongestProcTime => proc_time as f64,
                };

                if priority > best_priority + eps
                    || ((priority - best_priority).abs() <= eps
                        && best_job.map_or(true, |best| job < best))
                {
                    best_job = Some(job);
                    best_priority = priority;
                }
            }

            if let Some(job) = best_job {
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
                job_remaining_work[job] -= product_avg_times[product][op_idx];
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

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    // If you need random numbers, recommend using SmallRng with challenge.seed:
    // use rand::{rngs::SmallRng, Rng, SeedableRng};
    // let mut rng = SmallRng::from_seed(challenge.seed);

    // If you need HashMap or HashSet, make sure to use a deterministic hasher for consistent runtime_signature:
    // use crate::{seeded_hasher, HashMap, HashSet};
    // let hasher = seeded_hasher(&challenge.seed);
    // let map = HashMap::with_hasher(hasher);

    // Support hyperparameters if needed:
    // let hyperparameters = match hyperparameters {
    //     Some(hyperparameters) => {
    //         serde_json::from_value::<Hyperparameters>(Value::Object(hyperparameters.clone()))
    //             .map_err(|e| anyhow!("Failed to parse hyperparameters: {}", e))?
    //     }
    //     None => Hyperparameters { /* set default values here */ },
    // };

    // use save_solution(&Solution) to save your solution. Overwrites any previous solution

    // return Err(<msg>) if your algorithm encounters an error
    // return Ok(()) if your algorithm is finished
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;

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

    let mut product_avg_times = Vec::with_capacity(challenge.product_processing_times.len());
    for product_ops in challenge.product_processing_times.iter() {
        let mut avg_ops = Vec::with_capacity(product_ops.len());
        for op in product_ops.iter() {
            avg_ops.push(average_processing_time(op));
        }
        product_avg_times.push(avg_ops);
    }

    let mut job_ops_len = Vec::with_capacity(num_jobs);
    let mut job_total_work: Vec<f64> = Vec::with_capacity(num_jobs);
    for &product in job_products.iter() {
        let avg_ops = &product_avg_times[product];
        job_ops_len.push(avg_ops.len());
        job_total_work.push(avg_ops.iter().sum());
    }

    let rules = [
        DispatchRule::MostWorkRemaining,
        DispatchRule::MostOpsRemaining,
        DispatchRule::ShortestProcTime,
        DispatchRule::LongestProcTime,
    ];

    let mut best_result: Option<ScheduleResult> = None;
    for rule in rules.iter().copied() {
        let result = run_dispatch_rule(
            challenge,
            &job_products,
            &product_avg_times,
            &job_ops_len,
            &job_total_work,
            rule,
        )?;
        let is_better = best_result
            .as_ref()
            .map_or(true, |best| result.makespan < best.makespan);
        if is_better {
            best_result = Some(result);
        }
    }

    let best_result = best_result.ok_or_else(|| anyhow!("No valid schedule produced"))?;
    save_solution(&Solution {
        job_schedule: best_result.job_schedule,
    })?;
    Ok(())
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
