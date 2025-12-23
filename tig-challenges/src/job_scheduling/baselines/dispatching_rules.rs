use crate::job_scheduling::{Challenge, Solution};
use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use std::collections::HashMap;

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

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
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
    let mut job_remaining_work: Vec<f64> = Vec::with_capacity(num_jobs);
    for &product in job_products.iter() {
        let avg_ops = &product_avg_times[product];
        job_ops_len.push(avg_ops.len());
        job_remaining_work.push(avg_ops.iter().sum());
    }

    let mut job_next_op_idx = vec![0usize; num_jobs];
    let mut job_ready_time = vec![0u32; num_jobs];
    let mut machine_available_time = vec![0u32; num_machines];
    let mut job_schedule = job_ops_len
        .iter()
        .map(|&ops_len| Vec::with_capacity(ops_len))
        .collect::<Vec<_>>();

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
            let mut best_priority = -1.0_f64;

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

                let priority = job_remaining_work[job];
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

    save_solution(&Solution { job_schedule })?;
    Ok(())
}
