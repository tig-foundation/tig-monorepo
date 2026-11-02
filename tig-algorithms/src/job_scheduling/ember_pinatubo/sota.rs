use tig_challenges::job_scheduling::{Challenge, Solution};
use anyhow::{anyhow, Result};
use rand::seq::SliceRandom;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::cmp::Ordering;

pub fn sota_makespan(challenge: &Challenge) -> Option<u32> {
    let best = solve_challenge_with_params::<false>(challenge, 200, 2, 4).ok()?;
    super::checked_makespan(challenge, &Solution {
        job_schedule: best.job_schedule,
    }).ok()
}

pub fn sota_makespan_q(challenge: &Challenge) -> Option<u32> {
    let best = solve_challenge_with_params::<true>(challenge, 200, 2, 4).ok()?;
    let sota = super::checked_makespan(challenge, &Solution {
        job_schedule: best.job_schedule,
    }).ok()?;
    Some(sota)
}

pub fn sota_upper(challenge: &Challenge) -> Option<u32> {
    let best = solve_challenge_with_params::<false>(challenge, 0, 0, 0).ok()?;
    super::checked_makespan(challenge, &Solution {
        job_schedule: best.job_schedule,
    }).ok()
}

pub fn sota_partial(challenge: &Challenge, restarts: usize) -> Option<u32> {
    let best = solve_challenge_with_params::<false>(challenge, restarts, 0, 0).ok()?;
    super::checked_makespan(challenge, &Solution {
        job_schedule: best.job_schedule,
    }).ok()
}

const WORK_MIN_WEIGHT: f64 = 0.3;

struct OperationTimes {
    times: Vec<Option<u32>>,
    machines: Vec<usize>,
    min_remaining: u64,
}

fn earliest_end_time(
    time: u32,
    machine_available_time: &[u32],
    operation: &OperationTimes,
) -> u32 {
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
    machine_remaining_work: Vec<u64>,
    job_remaining_work: Vec<f64>,
    available_machines: Vec<usize>,
    machine_jobs: Vec<Vec<usize>>,
    candidates: Vec<Candidate>,
}

struct ScheduleResult {
    job_schedule: Vec<Vec<(usize, u32)>>,
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

fn run_dispatch_rule<const SQ: bool>(
    challenge: &Challenge,
    product_times: &[Vec<OperationTimes>],
    job_products: &[usize],
    product_work_times: &[Vec<f64>],
    job_ops_len: &[usize],
    job_total_work: &[f64],
    machine_total_work: &[u64],
    rule: DispatchRule,
    random_top_k: Option<usize>,
    rng: Option<&mut SmallRng>,
    buffers: &mut DispatchBuffers,
    job_schedule: &mut [Vec<(usize, u32)>],
    cutoff: Option<u32>,
) -> Result<Option<u32>> {
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;

    if let Some(cutoff) = cutoff {
        for &product in job_products {
            let remaining = match product_times[product].first() {
                Some(op) => op.min_remaining,
                None => 0,
            };
            if remaining >= cutoff as u64 {
                return Ok(None);
            }
        }
        for &remaining in machine_total_work {
            if remaining >= cutoff as u64 {
                return Ok(None);
            }
        }
    }

    let DispatchBuffers {
        job_next_op_idx, job_ready_time, machine_available_time,
        machine_remaining_work, job_remaining_work, available_machines, machine_jobs, candidates,
    } = buffers;
    job_next_op_idx.fill(0);
    job_ready_time.fill(0);
    machine_available_time.fill(0);
    machine_remaining_work.copy_from_slice(machine_total_work);
    for schedule in job_schedule.iter_mut() {
        schedule.clear();
    }
    for jobs in machine_jobs.iter_mut() {
        jobs.clear();
    }
    for job in 0..num_jobs {
        if let Some(op) = product_times[job_products[job]].first() {
            for &machine in &op.machines {
                machine_jobs[machine].push(job);
            }
        }
    }
    job_remaining_work.copy_from_slice(job_total_work);

    let mut remaining_ops = job_ops_len.iter().sum::<usize>();
    let mut time = 0u32;
    let eps = 1e-9_f64;
    let random_top_k = random_top_k.unwrap_or(0);
    let mut rng = rng;
    let use_random = random_top_k > 1 && rng.is_some();

    while remaining_ops > 0 {
        #[cfg(debug_assertions)]
        for (machine, jobs) in machine_jobs.iter().enumerate() {
            debug_assert!(jobs.windows(2).all(|pair| pair[0] < pair[1]));
            for job in 0..num_jobs {
                let expected = match product_times[job_products[job]].get(job_next_op_idx[job]) {
                    Some(op) => op.times[machine].is_some(),
                    None => false,
                };
                debug_assert_eq!(jobs.binary_search(&job).is_ok(), expected);
            }
        }
        let mut available_count = 0;
        for machine in 0..num_machines {
            available_machines[available_count] = machine;
            available_count += (machine_available_time[machine] <= time) as usize;
        }
        if use_random {
            available_machines[..available_count].shuffle(rng.as_mut().unwrap());
        }

        let mut scheduled_any = false;
        for &machine in &available_machines[..available_count] {
            let mut best_candidate: Option<Candidate> = None;

            if use_random {
                candidates.clear();

                for &job in &machine_jobs[machine] {
                    if job_ready_time[job] > time {
                        continue;
                    }

                    let product = job_products[job];
                    let op_idx = job_next_op_idx[job];
                    let op_times = &product_times[product][op_idx];
                    let proc_time = op_times.times[machine].unwrap();

                    let machine_end = time.max(machine_available_time[machine]) + proc_time;
                    if op_times.machines.len() > 1
                        && machine_end != earliest_end_time(time, machine_available_time, op_times)
                    {
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
                for &job in &machine_jobs[machine] {
                    if job_ready_time[job] > time {
                        continue;
                    }

                    let product = job_products[job];
                    let op_idx = job_next_op_idx[job];
                    let op_times = &product_times[product][op_idx];
                    let proc_time = op_times.times[machine].unwrap();

                    let machine_end = time.max(machine_available_time[machine]) + proc_time;
                    if op_times.machines.len() > 1
                        && machine_end != earliest_end_time(time, machine_available_time, op_times)
                    {
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
                #[cfg(debug_assertions)]
                let old_bound = cutoff.map(|_| job_ready_time[job] as u64 + product_times[product][op_idx].min_remaining);
                #[cfg(debug_assertions)]
                let old_machine_bound = cutoff.map(|_| machine_available_time[machine] as u64 + machine_remaining_work[machine]);

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
                if let Some(cutoff) = cutoff {
                    if product_times[product][op_idx].machines.len() == 1 {
                        machine_remaining_work[machine] -= proc_time as u64;
                    }
                    let machine_bound = end_time as u64 + machine_remaining_work[machine];
                    let remaining = match product_times[product].get(op_idx + 1) {
                        Some(op) => op.min_remaining,
                        None => 0,
                    };
                    let bound = end_time as u64 + remaining;
                    #[cfg(debug_assertions)]
                    {
                        debug_assert!(bound >= old_bound.unwrap());
                        debug_assert!(machine_bound >= old_machine_bound.unwrap());
                    }
                    if bound.max(machine_bound) >= cutoff as u64 { return Ok(None); }
                }
                for &eligible in &product_times[product][op_idx].machines {
                    if SQ && product_times[product].get(op_idx + 1)
                        .map_or(false, |op| op.times[eligible].is_some()) { continue; }
                    let jobs = &mut machine_jobs[eligible];
                    let position = jobs.binary_search(&job).unwrap();
                    jobs.remove(position);
                }
                if let Some(op) = product_times[product].get(op_idx + 1) {
                    for &eligible in &op.machines {
                        if SQ && product_times[product][op_idx].times[eligible].is_some() { continue; }
                        let jobs = &mut machine_jobs[eligible];
                        let position = jobs.binary_search(&job).unwrap_err();
                        jobs.insert(position, job);
                    }
                }
            }
        }

        if remaining_ops == 0 {
            break;
        }

        let mut next_time = u64::MAX;
        for &t in machine_available_time.iter() {
            let masked = t as u64 | (((t <= time) as u64) << 32);
            next_time = next_time.min(masked);
        }
        for job in 0..num_jobs {
            let t = job_ready_time[job];
            let excluded = (job_next_op_idx[job] >= job_ops_len[job]) | (t <= time);
            let masked = t as u64 | ((excluded as u64) << 32);
            next_time = next_time.min(masked);
        }

        if next_time > u32::MAX as u64 {
            return Err(if scheduled_any {
                anyhow!("No next event time found while operations remain unscheduled")
            } else {
                anyhow!("No schedulable operations remain; dispatching rules stalled")
            });
        }
        time = next_time as u32;
    }

    let makespan = job_ready_time.iter().copied().max().unwrap_or(0);
    Ok(Some(makespan))
}


fn solve_challenge_with_params<const SQ: bool>(
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
    let mut product_max_work = Vec::with_capacity(challenge.product_processing_times.len());
    let mut prune_safe = true;
    for product_ops in challenge.product_processing_times.iter() {
        let mut work_ops = Vec::with_capacity(product_ops.len());
        let mut operation_times = Vec::with_capacity(product_ops.len());
        let mut max_work = 0u64;
        for op in product_ops.iter() {
            let row = super::processing_row(op, challenge.num_machines)?;
            let machines: Vec<_> = row.iter().enumerate().filter_map(|(m, t)| t.map(|_| m)).collect();
            let sum: u32 = row.iter().flatten().sum();
            let avg = if machines.is_empty() { 0.0 } else { sum as f64 / machines.len() as f64 };
            let min = row.iter().flatten().copied().min().unwrap_or(0);
            prune_safe &= min > 0;
            max_work = max_work.saturating_add(row.iter().flatten().copied().max().unwrap_or(0) as u64);
            let work = avg * (1.0 - WORK_MIN_WEIGHT) + min as f64 * WORK_MIN_WEIGHT;
            work_ops.push(work);
            operation_times.push(OperationTimes {
                times: row,
                machines,
                min_remaining: min as u64,
            });
        }
        let mut remaining = 0u64;
        for op in operation_times.iter_mut().rev() {
            remaining = remaining.saturating_add(op.min_remaining);
            op.min_remaining = remaining;
        }
        product_work_times.push(work_ops);
        product_times.push(operation_times);
        product_max_work.push(max_work);
    }

    let mut job_ops_len = Vec::with_capacity(num_jobs);
    let mut job_total_work: Vec<f64> = Vec::with_capacity(num_jobs);
    let mut machine_total_work = vec![0u64; challenge.num_machines];
    let mut max_work = 0u64;
    for &product in job_products.iter() {
        let work_ops = &product_work_times[product];
        job_ops_len.push(work_ops.len());
        job_total_work.push(work_ops.iter().sum());
        max_work = max_work.saturating_add(product_max_work[product]);
        for op in &product_times[product] {
            if op.machines.len() == 1 {
                let machine = op.machines[0];
                machine_total_work[machine] = machine_total_work[machine]
                    .saturating_add(op.times[machine].unwrap() as u64);
            }
        }
    }
    prune_safe &= max_work <= u32::MAX as u64;

    let mut buffers = DispatchBuffers {
        job_next_op_idx: vec![0; num_jobs],
        job_ready_time: vec![0; num_jobs],
        machine_available_time: vec![0; challenge.num_machines],
        machine_remaining_work: vec![0; challenge.num_machines],
        job_remaining_work: vec![0.0; num_jobs],
        available_machines: vec![0; challenge.num_machines],
        machine_jobs: (0..challenge.num_machines).map(|_| Vec::with_capacity(num_jobs)).collect(),
        candidates: Vec::with_capacity(num_jobs),
    };
    let mut schedules = [Vec::with_capacity(num_jobs), Vec::with_capacity(num_jobs)];
    for schedule in &mut schedules {
        for &ops_len in &job_ops_len {
            schedule.push(Vec::with_capacity(ops_len));
        }
    }
    let mut schedule_index = 0;

    let rules = [
        DispatchRule::MostWorkRemaining,
        DispatchRule::MostOpsRemaining,
        DispatchRule::LeastFlexibility,
        DispatchRule::ShortestProcTime,
        DispatchRule::LongestProcTime,
    ];

    let mut best_result: Option<u32> = None;
    for rule in rules.iter().copied() {
        let Some(result) = run_dispatch_rule::<SQ>(
            challenge,
            &product_times,
            &job_products,
            &product_work_times,
            &job_ops_len,
            &job_total_work,
            &machine_total_work,
            rule,
            None,
            None,
            &mut buffers,
            &mut schedules[schedule_index],
            best_result.filter(|_| prune_safe),
        )? else { continue; };
        let is_better = best_result.map_or(true, |best| result < best);
        if is_better {
            best_result = Some(result);
            schedule_index ^= 1;
        }
    }

    let mut best_makespan = best_result.ok_or_else(|| anyhow!("No valid schedule produced"))?;

    let mut top_restarts: Vec<RestartResult> = Vec::new();

    if random_restarts > 0 {
        let mut rng = SmallRng::from_seed(challenge.seed);
        for _ in 1..=random_restarts {
            let seed = rng.r#gen::<u64>();
            let rule = rules[rng.gen_range(0..rules.len())];
            let random_top_k = rng.gen_range(2..=5);
            let mut local_rng = SmallRng::seed_from_u64(seed);
            let cutoff = if top_k == 0 || top_restarts.len() == top_k {
                Some(best_makespan.max(top_restarts.last().map_or(0, |r| r.makespan)))
            } else { None };
            let Some(result) = run_dispatch_rule::<SQ>(
                challenge,
                &product_times,
                &job_products,
                &product_work_times,
                &job_ops_len,
                &job_total_work,
                &machine_total_work,
                rule,
                Some(random_top_k),
                Some(&mut local_rng),
                &mut buffers,
                &mut schedules[schedule_index],
                cutoff.filter(|_| prune_safe),
            )? else { continue; };
            let makespan = result;
            let is_better = makespan < best_makespan;
            if is_better {
                best_makespan = makespan;
                schedule_index ^= 1;
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
        for entry in top_restarts.iter() {
            for attempt in 0..local_search_tries {
                let local_seed = entry.seed.wrapping_add(attempt as u64 + 1);
                let mut local_rng = SmallRng::seed_from_u64(local_seed);
                let local_k = match attempt % 3 {
                    0 => entry.random_top_k,
                    1 => entry.random_top_k.saturating_sub(1),
                    _ => entry.random_top_k.saturating_add(1),
                }
                .max(2);
                let Some(result) = run_dispatch_rule::<SQ>(
                    challenge,
                    &product_times,
                    &job_products,
                    &product_work_times,
                    &job_ops_len,
                    &job_total_work,
                    &machine_total_work,
                    entry.rule,
                    Some(local_k),
                    Some(&mut local_rng),
                    &mut buffers,
                    &mut schedules[schedule_index],
                    Some(best_makespan).filter(|_| prune_safe),
                )? else { continue; };
                if result < best_makespan {
                    best_makespan = result;
                    schedule_index ^= 1;
                }
            }
        }
    }

    let [first, second] = schedules;
    Ok(ScheduleResult {
        job_schedule: if schedule_index == 1 { first } else { second },
    })
}
