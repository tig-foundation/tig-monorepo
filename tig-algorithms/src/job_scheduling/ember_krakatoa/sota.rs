use tig_challenges::job_scheduling::{Challenge, Solution};
use anyhow::{anyhow, Result};
use rand::seq::SliceRandom;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::cmp::Ordering;

pub fn sota_makespan(challenge: &Challenge) -> Option<u32> {
    let best = solve_challenge_with_params(challenge, 200, 2, 4).ok()?;
    super::checked_makespan(
        challenge,
        &Solution {
            job_schedule: best.job_schedule,
        },
    )
    .ok()
}

pub fn sota_upper(challenge: &Challenge) -> Option<u32> {
    let best = solve_challenge_with_params(challenge, 0, 0, 0).ok()?;
    super::checked_makespan(
        challenge,
        &Solution {
            job_schedule: best.job_schedule,
        },
    )
    .ok()
}

const WORK_MIN_WEIGHT: f64 = 0.3;

struct OperationTimes {
    times: Vec<Option<u32>>,
    machines: Vec<usize>,
}

fn earliest_end_time(time: u32, machine_available_time: &[u32], operation: &OperationTimes) -> u32 {
    let mut earliest_end = u32::MAX;
    for &machine_id in &operation.machines {
        let start = time.max(machine_available_time[machine_id]);
        let proc_time = operation.times[machine_id].unwrap();
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
}

#[derive(Clone, Copy)]
struct Candidate {
    job: usize,
    priority: f64,
    machine_end: u32,
    proc_time: u32,
    flexibility: usize,
}

struct DispatchBuffers {
    job_next_op_idx: Vec<usize>,
    job_ready_time: Vec<u32>,
    machine_available_time: Vec<u32>,
    job_remaining_work: Vec<f64>,
    available_machines: Vec<usize>,
    ready_jobs: Vec<usize>,
    candidates: Vec<Candidate>,
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
    product_times: &[Vec<OperationTimes>],
    job_products: &[usize],
    product_work_times: &[Vec<f64>],
    job_ops_len: &[usize],
    job_total_work: &[f64],
    rule: DispatchRule,
    random_top_k: Option<usize>,
    rng: Option<&mut SmallRng>,
    buffers: &mut DispatchBuffers,
) -> Result<ScheduleResult> {
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;

    let DispatchBuffers {
        job_next_op_idx,
        job_ready_time,
        machine_available_time,
        job_remaining_work,
        available_machines,
        ready_jobs,
        candidates,
    } = buffers;
    job_next_op_idx.fill(0);
    job_ready_time.fill(0);
    machine_available_time.fill(0);
    let mut job_schedule = job_ops_len
        .iter()
        .map(|&ops_len| Vec::with_capacity(ops_len))
        .collect::<Vec<_>>();
    job_remaining_work.copy_from_slice(job_total_work);

    let mut remaining_ops = job_ops_len.iter().sum::<usize>();
    let mut time = 0u32;
    let eps = 1e-9_f64;
    let random_top_k = random_top_k.unwrap_or(0);
    let mut rng = rng;
    let use_random = random_top_k > 1 && rng.is_some();

    while remaining_ops > 0 {
        ready_jobs.clear();
        ready_jobs.extend(
            (0..num_jobs).filter(|&job| {
                job_next_op_idx[job] < job_ops_len[job] && job_ready_time[job] <= time
            }),
        );
        available_machines.clear();
        available_machines.extend((0..num_machines).filter(|&m| machine_available_time[m] <= time));
        if use_random {
            available_machines.shuffle(rng.as_mut().unwrap());
        }

        let mut scheduled_any = false;
        for &machine in available_machines.iter() {
            let mut best_candidate: Option<Candidate> = None;

            if use_random {
                candidates.clear();

                for &job in ready_jobs.iter() {
                    if job_next_op_idx[job] >= job_ops_len[job] {
                        continue;
                    }
                    if job_ready_time[job] > time {
                        continue;
                    }

                    let product = job_products[job];
                    let op_idx = job_next_op_idx[job];
                    let op_times = &product_times[product][op_idx];
                    let proc_time = match op_times.times[machine] {
                        Some(value) => value,
                        None => continue,
                    };

                    let earliest_end = earliest_end_time(time, machine_available_time, op_times);
                    let machine_end = time.max(machine_available_time[machine]) + proc_time;
                    if machine_end != earliest_end {
                        continue;
                    }

                    let flexibility = op_times.machines.len();
                    let priority = match rule {
                        DispatchRule::MostWorkRemaining => job_remaining_work[job],
                        DispatchRule::MostOpsRemaining => {
                            (job_ops_len[job] - job_next_op_idx[job]) as f64
                        }
                        DispatchRule::LeastFlexibility => -(flexibility as f64),
                        DispatchRule::ShortestProcTime => -(proc_time as f64),
                        DispatchRule::LongestProcTime => proc_time as f64,
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
                    let compare = |a: &Candidate, b: &Candidate| {
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
                    };
                    let k = random_top_k.min(candidates.len());
                    if k < candidates.len() {
                        candidates.select_nth_unstable_by(k, compare);
                    }
                    candidates[..k].sort_unstable_by(compare);
                    let pick = rng.as_mut().unwrap().gen_range(0..k);
                    best_candidate = Some(candidates[pick]);
                }
            } else {
                for &job in ready_jobs.iter() {
                    if job_next_op_idx[job] >= job_ops_len[job] {
                        continue;
                    }
                    if job_ready_time[job] > time {
                        continue;
                    }

                    let product = job_products[job];
                    let op_idx = job_next_op_idx[job];
                    let op_times = &product_times[product][op_idx];
                    let proc_time = match op_times.times[machine] {
                        Some(value) => value,
                        None => continue,
                    };

                    let earliest_end = earliest_end_time(time, machine_available_time, op_times);
                    let machine_end = time.max(machine_available_time[machine]) + proc_time;
                    if machine_end != earliest_end {
                        continue;
                    }

                    let flexibility = op_times.machines.len();
                    let priority = match rule {
                        DispatchRule::MostWorkRemaining => job_remaining_work[job],
                        DispatchRule::MostOpsRemaining => {
                            (job_ops_len[job] - job_next_op_idx[job]) as f64
                        }
                        DispatchRule::LeastFlexibility => -(flexibility as f64),
                        DispatchRule::ShortestProcTime => -(proc_time as f64),
                        DispatchRule::LongestProcTime => proc_time as f64,
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
                let proc_time = candidate.proc_time;

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

fn solve_challenge_with_params(
    challenge: &Challenge,
    random_restarts: usize,
    top_k: usize,
    local_search_tries: usize,
) -> Result<ScheduleResult> {
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

    let mut product_work_times = Vec::with_capacity(challenge.product_processing_times.len());
    let mut product_times = Vec::with_capacity(challenge.product_processing_times.len());
    for product_ops in challenge.product_processing_times.iter() {
        let mut work_ops = Vec::with_capacity(product_ops.len());
        let mut operation_times = Vec::with_capacity(product_ops.len());
        for op in product_ops.iter() {
            let row = super::processing_row(op, challenge.num_machines)?;
            let machines: Vec<_> = row
                .iter()
                .enumerate()
                .filter_map(|(m, t)| t.map(|_| m))
                .collect();
            let sum: u32 = row.iter().flatten().sum();
            let avg = if machines.is_empty() {
                0.0
            } else {
                sum as f64 / machines.len() as f64
            };
            let min = row.iter().flatten().copied().min().unwrap_or(0) as f64;
            let work = avg * (1.0 - WORK_MIN_WEIGHT) + min * WORK_MIN_WEIGHT;
            work_ops.push(work);
            operation_times.push(OperationTimes {
                times: row,
                machines,
            });
        }
        product_work_times.push(work_ops);
        product_times.push(operation_times);
    }

    let mut job_ops_len = Vec::with_capacity(num_jobs);
    let mut job_total_work: Vec<f64> = Vec::with_capacity(num_jobs);
    for &product in job_products.iter() {
        let work_ops = &product_work_times[product];
        job_ops_len.push(work_ops.len());
        job_total_work.push(work_ops.iter().sum());
    }

    let mut buffers = DispatchBuffers {
        job_next_op_idx: vec![0; num_jobs],
        job_ready_time: vec![0; num_jobs],
        machine_available_time: vec![0; challenge.num_machines],
        job_remaining_work: vec![0.0; num_jobs],
        available_machines: Vec::with_capacity(challenge.num_machines),
        ready_jobs: Vec::with_capacity(num_jobs),
        candidates: Vec::with_capacity(num_jobs),
    };

    let rules = [
        DispatchRule::MostWorkRemaining,
        DispatchRule::MostOpsRemaining,
        DispatchRule::LeastFlexibility,
        DispatchRule::ShortestProcTime,
        DispatchRule::LongestProcTime,
    ];

    let mut best_result: Option<ScheduleResult> = None;
    for rule in rules.iter().copied() {
        let result = run_dispatch_rule(
            challenge,
            &product_times,
            &job_products,
            &product_work_times,
            &job_ops_len,
            &job_total_work,
            rule,
            None,
            None,
            &mut buffers,
        )?;
        let is_better = best_result
            .as_ref()
            .map_or(true, |best| result.makespan < best.makespan);
        if is_better {
            best_result = Some(result);
        }
    }

    let mut best_result = best_result.ok_or_else(|| anyhow!("No valid schedule produced"))?;

    let mut top_restarts: Vec<RestartResult> = Vec::new();

    if random_restarts > 0 {
        let mut rng = SmallRng::from_seed(challenge.seed);
        for _ in 1..=random_restarts {
            let seed = rng.r#gen::<u64>();
            let rule = rules[rng.gen_range(0..rules.len())];
            let random_top_k = rng.gen_range(2..=5);
            let mut local_rng = SmallRng::seed_from_u64(seed);
            let result = run_dispatch_rule(
                challenge,
                &product_times,
                &job_products,
                &product_work_times,
                &job_ops_len,
                &job_total_work,
                rule,
                Some(random_top_k),
                Some(&mut local_rng),
                &mut buffers,
            )?;
            let makespan = result.makespan;
            let is_better = makespan < best_result.makespan;
            if is_better {
                best_result = result;
            }

            if top_k > 0 {
                top_restarts.push(RestartResult {
                    makespan,
                    rule,
                    random_top_k,
                    seed,
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
                let result = run_dispatch_rule(
                    challenge,
                    &product_times,
                    &job_products,
                    &product_work_times,
                    &job_ops_len,
                    &job_total_work,
                    restart.rule,
                    Some(local_k),
                    Some(&mut local_rng),
                    &mut buffers,
                )?;
                if result.makespan < best_result.makespan {
                    best_result = result;
                }
            }
        }
    }

    Ok(best_result)
}
