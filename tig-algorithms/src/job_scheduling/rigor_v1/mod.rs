// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use crate::{seeded_hasher, HashMap, HashSet};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::job_scheduling::*;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand::seq::SliceRandom;

// ============================================================================
// SolutionSaver — monotonically improving solution guard
// ============================================================================
struct SolutionSaver<'a> {
    save_fn: &'a dyn Fn(&Solution) -> Result<()>,
    best_makespan: u32,
}

impl<'a> SolutionSaver<'a> {
    fn new(save_fn: &'a dyn Fn(&Solution) -> Result<()>) -> Self {
        Self { save_fn, best_makespan: u32::MAX }
    }

    /// Save solution only if its makespan improves on the best seen so far.
    /// Returns true if the solution was actually saved.
    fn save(&mut self, solution: &Solution, makespan: u32) -> Result<bool> {
        if makespan < self.best_makespan {
            (self.save_fn)(solution)?;
            self.best_makespan = makespan;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

// ============================================================================
// Hyperparameters
// ============================================================================
#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub construction_restarts: usize,
    pub construction_top_k: usize,
    pub tabu_tenure: usize,
    pub max_iterations: usize,
    pub max_idle_iterations: usize,
    pub tabu_restarts: usize,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            construction_restarts: 50,
            construction_top_k: 3,
            tabu_tenure: 10,
            max_iterations: 0,
            max_idle_iterations: 1000,
            tabu_restarts: 5,
        }
    }
}

pub fn help() {
    println!("FJSP Solver v2: Construction Heuristic + Critical Path Tabu Search");
}

// ============================================================================
// Relation Enum
// ============================================================================
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Relation {
    Lesser,
    Greater,
    None,
}

// ============================================================================
// Core Data Structures
// ============================================================================
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct OpId {
    job: usize,
    op_idx: usize,
}

#[derive(Debug, Clone)]
struct OpInfo {
    id: OpId,
    product: usize,
    product_op_idx: usize,
    eligible_machines: Vec<usize>,
    processing_times: Vec<(usize, u32)>,
}

#[derive(Debug, Clone)]
struct Schedule {
    assignments: Vec<Vec<(usize, u32, u32)>>,
    machine_orders: Vec<Vec<OpId>>,
    makespan: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TabuMove {
    op_a: OpId,
    op_b: OpId,
    machine: usize,
}

struct ProblemData {
    num_jobs: usize,
    num_machines: usize,
    job_products: Vec<usize>,
    job_ops: Vec<Vec<OpInfo>>,
    total_ops: usize,
    is_flexible: bool,       // true if any op has >1 eligible machine
    avg_flexibility: f64,    // average number of eligible machines per op
}

// ============================================================================
// Problem Data Construction
// ============================================================================
fn build_problem_data(challenge: &Challenge) -> ProblemData {
    let job_products: Vec<usize> = challenge
        .jobs_per_product
        .iter()
        .enumerate()
        .flat_map(|(product, &count)| std::iter::repeat(product).take(count))
        .collect();

    let mut job_ops = Vec::with_capacity(challenge.num_jobs);
    let mut total_ops = 0usize;
    for job in 0..challenge.num_jobs {
        let product = job_products[job];
        let product_times = &challenge.product_processing_times[product];
        let mut ops = Vec::with_capacity(product_times.len());
        for (op_idx, op_machines) in product_times.iter().enumerate() {
            let mut processing_times: Vec<(usize, u32)> = op_machines
                .iter()
                .map(|(&machine, &ptime)| (machine, ptime))
                .collect();
            processing_times.sort_unstable();
            let eligible_machines: Vec<usize> = processing_times.iter().map(|&(m, _)| m).collect();
            ops.push(OpInfo {
                id: OpId { job, op_idx },
                product,
                product_op_idx: op_idx,
                eligible_machines,
                processing_times,
            });
            total_ops += 1;
        }
        job_ops.push(ops);
    }

    let mut flex_sum = 0usize;
    let is_flexible = job_ops.iter().any(|ops| ops.iter().any(|op| op.eligible_machines.len() > 1));
    for ops in &job_ops {
        for op in ops {
            flex_sum += op.eligible_machines.len();
        }
    }
    let avg_flexibility = if total_ops > 0 { flex_sum as f64 / total_ops as f64 } else { 1.0 };

    ProblemData {
        num_jobs: challenge.num_jobs,
        num_machines: challenge.num_machines,
        job_products,
        job_ops,
        total_ops,
        is_flexible,
        avg_flexibility,
    }
}

// ============================================================================
// Recompute Schedule: O(V+E) Topological Sort
// ============================================================================
fn recompute_schedule(schedule: &mut Schedule, data: &ProblemData) -> u32 {
    let num_jobs = data.num_jobs;

    // Build op_machine_pos
    let mut op_machine_pos: Vec<Vec<(usize, usize)>> = data
        .job_ops
        .iter()
        .map(|ops| vec![(0, 0); ops.len()])
        .collect();
    for (machine, order) in schedule.machine_orders.iter().enumerate() {
        for (pos, op_id) in order.iter().enumerate() {
            op_machine_pos[op_id.job][op_id.op_idx] = (machine, pos);
        }
    }

    // Compute in-degrees
    let mut in_degree: Vec<Vec<u32>> = data
        .job_ops
        .iter()
        .map(|ops| vec![0u32; ops.len()])
        .collect();

    for job in 0..num_jobs {
        for op_idx in 0..data.job_ops[job].len() {
            let mut deg = 0u32;
            if op_idx > 0 { deg += 1; }
            let (_, pos) = op_machine_pos[job][op_idx];
            if pos > 0 { deg += 1; }
            in_degree[job][op_idx] = deg;
        }
    }

    // Seed queue with zero in-degree ops
    let mut queue: Vec<OpId> = Vec::with_capacity(data.total_ops);
    for job in 0..num_jobs {
        for op_idx in 0..data.job_ops[job].len() {
            if in_degree[job][op_idx] == 0 {
                queue.push(OpId { job, op_idx });
            }
        }
    }

    let mut makespan = 0u32;
    let mut head = 0;

    while head < queue.len() {
        let op_id = queue[head];
        head += 1;

        let (machine, pos) = op_machine_pos[op_id.job][op_id.op_idx];

        let job_pred_finish = if op_id.op_idx == 0 { 0 } else {
            schedule.assignments[op_id.job][op_id.op_idx - 1].2
        };
        let machine_pred_finish = if pos == 0 { 0 } else {
            let prev_op = schedule.machine_orders[machine][pos - 1];
            schedule.assignments[prev_op.job][prev_op.op_idx].2
        };

        let start_time = job_pred_finish.max(machine_pred_finish);
        let proc_time = data.job_ops[op_id.job][op_id.op_idx]
            .processing_times
            .iter()
            .find(|&&(m, _)| m == machine)
            .map(|&(_, t)| t)
            .expect("Machine must be eligible");
        let finish_time = start_time + proc_time;

        schedule.assignments[op_id.job][op_id.op_idx] = (machine, start_time, finish_time);
        if finish_time > makespan { makespan = finish_time; }

        // Decrement job successor
        let next_op = op_id.op_idx + 1;
        if next_op < data.job_ops[op_id.job].len() {
            in_degree[op_id.job][next_op] -= 1;
            if in_degree[op_id.job][next_op] == 0 {
                queue.push(OpId { job: op_id.job, op_idx: next_op });
            }
        }

        // Decrement machine successor
        if pos + 1 < schedule.machine_orders[machine].len() {
            let succ = schedule.machine_orders[machine][pos + 1];
            in_degree[succ.job][succ.op_idx] -= 1;
            if in_degree[succ.job][succ.op_idx] == 0 {
                queue.push(succ);
            }
        }
    }

    schedule.makespan = makespan;
    makespan
}

// ============================================================================
// Tier 1: Construction Heuristic
// ============================================================================
#[derive(Clone, Copy, Debug)]
enum DispatchRule {
    MostWorkRemaining,
    MostOpsRemaining,
    LeastFlexibility,
    ShortestProcTime,
    LongestProcTime,
    MinLoadBalance,  // for flexible: prioritize balancing machine loads
}

const ALL_RULES: [DispatchRule; 5] = [
    DispatchRule::MostWorkRemaining,
    DispatchRule::MostOpsRemaining,
    DispatchRule::LeastFlexibility,
    DispatchRule::ShortestProcTime,
    DispatchRule::LongestProcTime,
];

const FLEX_RULES: [DispatchRule; 6] = [
    DispatchRule::MostWorkRemaining,
    DispatchRule::MostOpsRemaining,
    DispatchRule::LeastFlexibility,
    DispatchRule::ShortestProcTime,
    DispatchRule::LongestProcTime,
    DispatchRule::MinLoadBalance,
];

fn compute_work_remaining(data: &ProblemData) -> Vec<f64> {
    let mut work = Vec::with_capacity(data.num_jobs);
    for job in 0..data.num_jobs {
        let total: f64 = data.job_ops[job].iter().map(|op| {
            let avg: f64 = op.processing_times.iter().map(|&(_, t)| t as f64).sum::<f64>()
                / op.processing_times.len() as f64;
            let min: f64 = op.processing_times.iter().map(|&(_, t)| t as f64).fold(f64::MAX, f64::min);
            avg * 0.7 + min * 0.3
        }).sum();
        work.push(total);
    }
    work
}

fn construct_schedule(
    data: &ProblemData,
    challenge: &Challenge,
    rule: DispatchRule,
    random_top_k: usize,
    rng: Option<&mut SmallRng>,
    eet_slack: u32,  // 0 for strict EET, >0 allows near-EET assignments
) -> Schedule {
    let num_jobs = data.num_jobs;
    let num_machines = data.num_machines;

    let mut job_next_op = vec![0usize; num_jobs];
    let mut job_ready_time = vec![0u32; num_jobs];
    let mut machine_available = vec![0u32; num_machines];
    let mut job_remaining_work = compute_work_remaining(data);

    let mut assignments: Vec<Vec<(usize, u32, u32)>> = data
        .job_ops.iter().map(|ops| vec![(0, 0, 0); ops.len()]).collect();
    let mut machine_orders: Vec<Vec<OpId>> = vec![Vec::new(); num_machines];

    let mut remaining_ops = data.total_ops;
    let mut time = 0u32;
    let use_random = random_top_k > 1 && rng.is_some();
    let mut rng = rng;

    while remaining_ops > 0 {
        let mut machine_list: Vec<usize> = (0..num_machines)
            .filter(|&m| machine_available[m] <= time)
            .collect();
        if use_random {
            if let Some(ref mut r) = rng {
                for i in (1..machine_list.len()).rev() {
                    let j = r.gen_range(0..=i);
                    machine_list.swap(i, j);
                }
            }
        }

        for &machine in &machine_list {
            struct Candidate {
                job: usize,
                op_idx: usize,
                machine: usize,
                proc_time: u32,
                start_time: u32,
                priority: f64,
                flexibility: usize,
                machine_end: u32,
            }

            let mut candidates: Vec<Candidate> = Vec::new();

            for job in 0..num_jobs {
                let op_idx = job_next_op[job];
                if op_idx >= data.job_ops[job].len() { continue; }
                if job_ready_time[job] > time { continue; }

                let op_info = &data.job_ops[job][op_idx];
                let proc_time = match op_info.processing_times.iter().find(|&&(m, _)| m == machine) {
                    Some(&(_, t)) => t,
                    None => continue,
                };

                let earliest_end = op_info.processing_times.iter()
                    .map(|&(m, t)| time.max(machine_available[m]) + t)
                    .min().unwrap_or(u32::MAX);
                let machine_end = time.max(machine_available[machine]) + proc_time;
                if machine_end > earliest_end + eet_slack { continue; }

                let flexibility = op_info.eligible_machines.len();
                let priority = match rule {
                    DispatchRule::MostWorkRemaining => job_remaining_work[job],
                    DispatchRule::MostOpsRemaining => (data.job_ops[job].len() - op_idx) as f64,
                    DispatchRule::LeastFlexibility => -(flexibility as f64),
                    DispatchRule::ShortestProcTime => -(proc_time as f64),
                    DispatchRule::LongestProcTime => proc_time as f64,
                    DispatchRule::MinLoadBalance => {
                        // Prefer assigning to least loaded machine (negative load = higher priority)
                        -(machine_available[machine] as f64) + job_remaining_work[job] * 0.1
                    }
                };

                candidates.push(Candidate {
                    job, op_idx, machine, proc_time,
                    start_time: time.max(machine_available[machine]),
                    priority, flexibility, machine_end,
                });
            }

            if candidates.is_empty() { continue; }

            candidates.sort_by(|a, b| {
                b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.machine_end.cmp(&b.machine_end))
                    .then(a.proc_time.cmp(&b.proc_time))
                    .then(a.flexibility.cmp(&b.flexibility))
                    .then(a.job.cmp(&b.job))
            });

            let pick = if use_random && candidates.len() > 1 {
                let k = random_top_k.min(candidates.len());
                rng.as_mut().unwrap().gen_range(0..k)
            } else {
                0
            };

            let chosen = &candidates[pick];
            let job = chosen.job;
            let op_idx = chosen.op_idx;
            let machine_id = chosen.machine;
            let start_time = chosen.start_time;
            let finish_time = start_time + chosen.proc_time;

            assignments[job][op_idx] = (machine_id, start_time, finish_time);
            machine_orders[machine_id].push(OpId { job, op_idx });
            job_next_op[job] += 1;
            job_ready_time[job] = finish_time;
            machine_available[machine_id] = finish_time;

            let op_info = &data.job_ops[job][op_idx];
            let avg: f64 = op_info.processing_times.iter().map(|&(_, t)| t as f64).sum::<f64>()
                / op_info.processing_times.len() as f64;
            let min: f64 = op_info.processing_times.iter().map(|&(_, t)| t as f64).fold(f64::MAX, f64::min);
            job_remaining_work[job] -= avg * 0.7 + min * 0.3;
            if job_remaining_work[job] < 0.0 { job_remaining_work[job] = 0.0; }
            remaining_ops -= 1;
        }

        if remaining_ops == 0 { break; }

        let mut next_time = u32::MAX;
        for &t in machine_available.iter() {
            if t > time && t < next_time { next_time = t; }
        }
        for job in 0..num_jobs {
            if job_next_op[job] < data.job_ops[job].len() && job_ready_time[job] > time && job_ready_time[job] < next_time {
                next_time = job_ready_time[job];
            }
        }
        if next_time == u32::MAX { break; }
        time = next_time;
    }

    let makespan = assignments.iter()
        .map(|ops| ops.last().map(|&(_, _, f)| f).unwrap_or(0))
        .max().unwrap_or(0);

    Schedule { assignments, machine_orders, makespan }
}

/// Exact port of TIG's greedy baseline (effort=0).
/// Produces a Solution directly, matching TIG's dispatching_rules.rs exactly.
fn compute_tig_baseline(challenge: &Challenge) -> Solution {
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;
    
    // Build job_products mapping
    let mut job_products = Vec::with_capacity(num_jobs);
    for (product, count) in challenge.jobs_per_product.iter().enumerate() {
        for _ in 0..*count {
            job_products.push(product);
        }
    }
    
    // Compute work times: avg * 0.7 + min * 0.3
    let product_work_times: Vec<Vec<f64>> = challenge.product_processing_times.iter().map(|product_ops| {
        product_ops.iter().map(|op| {
            let avg = op.values().sum::<u32>() as f64 / op.len() as f64;
            let min = *op.values().min().unwrap_or(&0) as f64;
            avg * 0.7 + min * 0.3
        }).collect()
    }).collect();
    
    let job_ops_len: Vec<usize> = job_products.iter().map(|&p| challenge.product_processing_times[p].len()).collect();
    let job_total_work: Vec<f64> = job_products.iter().map(|&p| product_work_times[p].iter().sum()).collect();
    
    #[derive(Clone, Copy)]
    enum BaselineRule { MostWork, MostOps, LeastFlex, ShortProc, LongProc }
    const RULES: [BaselineRule; 5] = [
        BaselineRule::MostWork, BaselineRule::MostOps, BaselineRule::LeastFlex,
        BaselineRule::ShortProc, BaselineRule::LongProc,
    ];
    
    #[derive(Clone, Copy)]
    struct Cand { job: usize, priority: f64, machine_end: u32, proc_time: u32, flexibility: usize }
    
    let run_baseline = |rule: BaselineRule, random_top_k: usize, rng: Option<&mut SmallRng>| -> (Vec<Vec<(usize, u32)>>, u32) {
        let mut job_next_op = vec![0usize; num_jobs];
        let mut job_ready_time = vec![0u32; num_jobs];
        let mut machine_available = vec![0u32; num_machines];
        let mut job_schedule: Vec<Vec<(usize, u32)>> = job_ops_len.iter().map(|&n| Vec::with_capacity(n)).collect();
        let mut job_remaining_work = job_total_work.clone();
        let mut remaining_ops: usize = job_ops_len.iter().sum();
        let mut time = 0u32;
        let eps = 1e-9_f64;
        let use_random = random_top_k > 1 && rng.is_some();
        let mut rng = rng;
        
        while remaining_ops > 0 {
            let mut available_machines: Vec<usize> = (0..num_machines)
                .filter(|&m| machine_available[m] <= time)
                .collect();
            available_machines.sort_unstable();
            if use_random {
                if let Some(ref mut r) = rng {
                    available_machines.shuffle(*r);
                }
            }
            
            let mut scheduled_any = false;
            for &machine in &available_machines {
                let mut best_candidate: Option<Cand> = None;
                
                if use_random {
                    let mut candidates: Vec<Cand> = Vec::new();
                    for job in 0..num_jobs {
                        if job_next_op[job] >= job_ops_len[job] { continue; }
                        if job_ready_time[job] > time { continue; }
                        let product = job_products[job];
                        let op_idx = job_next_op[job];
                        let op_times = &challenge.product_processing_times[product][op_idx];
                        let proc_time = match op_times.get(&machine) {
                            Some(&v) => v,
                            None => continue,
                        };
                        // EET filter
                        let earliest_end = op_times.iter()
                            .map(|(&m, &t)| time.max(machine_available[m]) + t)
                            .min().unwrap_or(u32::MAX);
                        let machine_end = time.max(machine_available[machine]) + proc_time;
                        if machine_end != earliest_end { continue; }
                        
                        let flexibility = op_times.len();
                        let priority = match rule {
                            BaselineRule::MostWork => job_remaining_work[job],
                            BaselineRule::MostOps => (job_ops_len[job] - job_next_op[job]) as f64,
                            BaselineRule::LeastFlex => -(flexibility as f64),
                            BaselineRule::ShortProc => -(proc_time as f64),
                            BaselineRule::LongProc => proc_time as f64,
                        };
                        candidates.push(Cand { job, priority, machine_end, proc_time, flexibility });
                    }
                    if !candidates.is_empty() {
                        candidates.sort_by(|a, b| {
                            b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal)
                                .then(a.machine_end.cmp(&b.machine_end))
                                .then(a.proc_time.cmp(&b.proc_time))
                                .then(a.flexibility.cmp(&b.flexibility))
                                .then(a.job.cmp(&b.job))
                        });
                        let k = random_top_k.min(candidates.len());
                        let pick = rng.as_mut().unwrap().gen_range(0..k);
                        best_candidate = Some(candidates[pick]);
                    }
                } else {
                    // Deterministic: TIG's exact logic with epsilon comparison
                    for job in 0..num_jobs {
                        if job_next_op[job] >= job_ops_len[job] { continue; }
                        if job_ready_time[job] > time { continue; }
                        let product = job_products[job];
                        let op_idx = job_next_op[job];
                        let op_times = &challenge.product_processing_times[product][op_idx];
                        let proc_time = match op_times.get(&machine) {
                            Some(&v) => v,
                            None => continue,
                        };
                        let earliest_end = op_times.iter()
                            .map(|(&m, &t)| time.max(machine_available[m]) + t)
                            .min().unwrap_or(u32::MAX);
                        let machine_end = time.max(machine_available[machine]) + proc_time;
                        if machine_end != earliest_end { continue; }
                        
                        let flexibility = op_times.len();
                        let priority = match rule {
                            BaselineRule::MostWork => job_remaining_work[job],
                            BaselineRule::MostOps => (job_ops_len[job] - job_next_op[job]) as f64,
                            BaselineRule::LeastFlex => -(flexibility as f64),
                            BaselineRule::ShortProc => -(proc_time as f64),
                            BaselineRule::LongProc => proc_time as f64,
                        };
                        let cand = Cand { job, priority, machine_end, proc_time, flexibility };
                        let is_better = match best_candidate {
                            None => true,
                            Some(ref best) => {
                                if cand.priority > best.priority + eps { true }
                                else if (cand.priority - best.priority).abs() <= eps {
                                    if cand.machine_end < best.machine_end { true }
                                    else if cand.machine_end == best.machine_end {
                                        if cand.proc_time < best.proc_time { true }
                                        else if cand.proc_time == best.proc_time {
                                            if cand.flexibility < best.flexibility { true }
                                            else if cand.flexibility == best.flexibility { cand.job < best.job }
                                            else { false }
                                        } else { false }
                                    } else { false }
                                } else { false }
                            }
                        };
                        if is_better { best_candidate = Some(cand); }
                    }
                }
                
                if let Some(cand) = best_candidate {
                    let job = cand.job;
                    let start_time = time.max(machine_available[machine]);
                    let end_time = start_time + cand.proc_time;
                    let product = job_products[job];
                    let op_idx = job_next_op[job];
                    
                    job_schedule[job].push((machine, start_time));
                    job_next_op[job] += 1;
                    job_ready_time[job] = end_time;
                    machine_available[machine] = end_time;
                    job_remaining_work[job] -= product_work_times[product][op_idx];
                    if job_remaining_work[job] < 0.0 { job_remaining_work[job] = 0.0; }
                    remaining_ops -= 1;
                    scheduled_any = true;
                }
            }
            
            if remaining_ops == 0 { break; }
            let mut next_time: Option<u32> = None;
            for &t in machine_available.iter() {
                if t > time { next_time = Some(next_time.map_or(t, |best: u32| best.min(t))); }
            }
            for job in 0..num_jobs {
                if job_next_op[job] < job_ops_len[job] && job_ready_time[job] > time {
                    let t = job_ready_time[job];
                    next_time = Some(next_time.map_or(t, |best: u32| best.min(t)));
                }
            }
            match next_time {
                Some(t) => time = t,
                None => break,
            }
        }
        
        let makespan = job_ready_time.iter().copied().max().unwrap_or(0);
        (job_schedule, makespan)
    };
    
    // Run 5 deterministic rules
    let mut best_schedule: Option<Vec<Vec<(usize, u32)>>> = None;
    let mut best_makespan = u32::MAX;
    
    for &rule in &RULES {
        let (sched, ms) = run_baseline(rule, 0, None);
        if ms < best_makespan {
            best_makespan = ms;
            best_schedule = Some(sched);
        }
    }
    
    // 10 random restarts with TIG's exact seeding
    let mut baseline_rng = SmallRng::from_seed(challenge.seed);
    for _ in 0..10 {
        let seed: u64 = baseline_rng.gen();
        let rule = RULES[baseline_rng.gen_range(0..RULES.len())];
        let random_top_k = baseline_rng.gen_range(2..=5usize);
        let mut local_rng = SmallRng::seed_from_u64(seed);
        let (sched, ms) = run_baseline(rule, random_top_k, Some(&mut local_rng));
        if ms < best_makespan {
            best_makespan = ms;
            best_schedule = Some(sched);
        }
    }
    
    Solution { job_schedule: best_schedule.unwrap_or_default() }
}

fn evaluate_makespan_check(challenge: &Challenge, solution: &Solution) -> Result<u32> {
    if solution.job_schedule.len() != challenge.num_jobs {
        return Err(anyhow::anyhow!("Wrong job count"));
    }
    let mut job = 0;
    let mut machine_usage: std::collections::HashMap<usize, Vec<(u32, u32)>> = std::collections::HashMap::new();
    let mut makespan = 0u32;
    for (product, &num_jobs) in challenge.jobs_per_product.iter().enumerate() {
        for _ in 0..num_jobs {
            let schedule = &solution.job_schedule[job];
            let processing_times = &challenge.product_processing_times[product];
            if schedule.len() != processing_times.len() {
                return Err(anyhow::anyhow!("Wrong op count for job {}", job));
            }
            let mut min_start = 0u32;
            for (op_idx, &(machine, start_time)) in schedule.iter().enumerate() {
                let eligible = &processing_times[op_idx];
                if !eligible.contains_key(&machine) {
                    return Err(anyhow::anyhow!("Ineligible machine"));
                }
                if start_time < min_start {
                    return Err(anyhow::anyhow!("Op starts before previous completes"));
                }
                let finish = start_time + eligible[&machine];
                machine_usage.entry(machine).or_default().push((start_time, finish));
                min_start = finish;
            }
            if min_start > makespan { makespan = min_start; }
            job += 1;
        }
    }
    for (_, usage) in machine_usage.iter_mut() {
        usage.sort_by_key(|&(s, _)| s);
        for i in 1..usage.len() {
            if usage[i].0 < usage[i - 1].1 {
                return Err(anyhow::anyhow!("Overlapping jobs on machine"));
            }
        }
    }
    Ok(makespan)
}

// ============================================================================
// Tier 1b: Hierarchical Cooperative Construction
// ============================================================================
// 1. Build conflict graph: edge(job_a, job_b) = # shared eligible machines across ops
// 2. Hierarchical clustering: greedily merge highest-conflict pairs
// 3. The merge order defines job scheduling priority — jobs merged early
//    are in highest conflict and need to be interleaved carefully
// 4. Feed this priority into the existing time-simulation construction

fn build_conflict_graph(data: &ProblemData) -> Vec<Vec<u32>> {
    let n = data.num_jobs;
    let mut conflict = vec![vec![0u32; n]; n];

    // For each machine, collect which jobs have ops eligible on it
    let mut machine_jobs: Vec<Vec<usize>> = vec![Vec::new(); data.num_machines];
    for job in 0..n {
        // Track which machines this job uses (deduplicated)
        let mut job_machines = Vec::new();
        for op in &data.job_ops[job] {
            for &m in &op.eligible_machines {
                job_machines.push(m);
            }
        }
        job_machines.sort_unstable();
        job_machines.dedup();
        for m in job_machines {
            machine_jobs[m].push(job);
        }
    }

    // For each machine, every pair of jobs sharing it gets +1 conflict
    for m in 0..data.num_machines {
        let jobs = &machine_jobs[m];
        for i in 0..jobs.len() {
            for j in (i + 1)..jobs.len() {
                conflict[jobs[i]][jobs[j]] += 1;
                conflict[jobs[j]][jobs[i]] += 1;
            }
        }
    }

    conflict
}

fn hierarchical_cluster(data: &ProblemData) -> Vec<usize> {
    let n = data.num_jobs;
    let conflict = build_conflict_graph(data);

    // Union-Find for clustering
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<usize> = vec![0; n];

    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    fn union(parent: &mut Vec<usize>, rank: &mut Vec<usize>, a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra == rb { return; }
        if rank[ra] < rank[rb] { parent[ra] = rb; }
        else if rank[ra] > rank[rb] { parent[rb] = ra; }
        else { parent[rb] = ra; rank[ra] += 1; }
    }

    // Collect all edges, sort by conflict weight descending
    let mut edges: Vec<(u32, usize, usize)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if conflict[i][j] > 0 {
                edges.push((conflict[i][j], i, j));
            }
        }
    }
    edges.sort_by(|a, b| b.0.cmp(&a.0));

    // Merge order: track when each job gets merged (earlier = higher conflict)
    let mut merge_priority = vec![0usize; n]; // lower = higher priority
    let mut priority_counter = 0usize;

    for &(_, i, j) in &edges {
        let ri = find(&mut parent, i);
        let rj = find(&mut parent, j);
        if ri != rj {
            // Both jobs in this merge get high priority
            if merge_priority[i] == 0 && i != 0 { // skip job 0 default
                priority_counter += 1;
                merge_priority[i] = priority_counter;
            }
            if merge_priority[j] == 0 && j != 0 {
                priority_counter += 1;
                merge_priority[j] = priority_counter;
            }
            union(&mut parent, &mut rank, ri, rj);
        }
    }

    // Jobs that were never in a high-conflict merge get low priority
    for job in 0..n {
        if merge_priority[job] == 0 {
            priority_counter += 1;
            merge_priority[job] = priority_counter;
        }
    }

    // Build job ordering: sorted by merge_priority (low number = schedule first)
    let mut job_order: Vec<usize> = (0..n).collect();
    job_order.sort_by_key(|&j| merge_priority[j]);

    job_order
}

/// Construct schedule using hierarchical job priority + work-remaining weighting
fn construct_schedule_hierarchical(
    data: &ProblemData,
    challenge: &Challenge,
    rng: Option<&mut SmallRng>,
    eet_slack: u32,
) -> Schedule {
    let num_jobs = data.num_jobs;
    let num_machines = data.num_machines;

    let job_order = hierarchical_cluster(data);

    // Build job priority from cluster order + remaining work
    // Jobs earlier in job_order get higher base priority
    let mut job_base_priority = vec![0.0f64; num_jobs];
    for (rank, &job) in job_order.iter().enumerate() {
        // Higher priority for earlier in merge order
        // Also weight by total remaining work
        let work: f64 = data.job_ops[job].iter().map(|op| {
            let avg: f64 = op.processing_times.iter().map(|&(_, t)| t as f64).sum::<f64>()
                / op.processing_times.len() as f64;
            avg
        }).sum();
        // Combine: cluster priority (inverted rank) + work remaining
        job_base_priority[job] = (num_jobs - rank) as f64 * 1000.0 + work;
    }

    let mut job_next_op = vec![0usize; num_jobs];
    let mut job_ready_time = vec![0u32; num_jobs];
    let mut machine_available = vec![0u32; num_machines];

    let mut assignments: Vec<Vec<(usize, u32, u32)>> = data
        .job_ops.iter().map(|ops| vec![(0, 0, 0); ops.len()]).collect();
    let mut machine_orders: Vec<Vec<OpId>> = vec![Vec::new(); num_machines];

    let mut remaining_ops = data.total_ops;
    let mut time = 0u32;
    let use_random = rng.is_some();
    let mut rng = rng;

    while remaining_ops > 0 {
        let mut machine_list: Vec<usize> = (0..num_machines)
            .filter(|&m| machine_available[m] <= time)
            .collect();
        if use_random {
            if let Some(ref mut r) = rng {
                for i in (1..machine_list.len()).rev() {
                    let j = r.gen_range(0..=i);
                    machine_list.swap(i, j);
                }
            }
        }

        for &machine in &machine_list {
            struct Candidate {
                job: usize,
                op_idx: usize,
                machine: usize,
                proc_time: u32,
                start_time: u32,
                priority: f64,
                machine_end: u32,
            }

            let mut candidates: Vec<Candidate> = Vec::new();

            for job in 0..num_jobs {
                let op_idx = job_next_op[job];
                if op_idx >= data.job_ops[job].len() { continue; }
                if job_ready_time[job] > time { continue; }

                let op_info = &data.job_ops[job][op_idx];
                let proc_time = match op_info.processing_times.iter().find(|&&(m, _)| m == machine) {
                    Some(&(_, t)) => t,
                    None => continue,
                };

                // EET filter: allow slack for flexible problems
                let earliest_end = op_info.processing_times.iter()
                    .map(|&(m, t)| time.max(machine_available[m]) + t)
                    .min().unwrap_or(u32::MAX);
                let machine_end = time.max(machine_available[machine]) + proc_time;
                if machine_end > earliest_end + eet_slack { continue; }

                // Priority from hierarchical clustering + remaining work
                let priority = job_base_priority[job];

                candidates.push(Candidate {
                    job, op_idx, machine, proc_time,
                    start_time: time.max(machine_available[machine]),
                    priority, machine_end,
                });
            }

            if candidates.is_empty() { continue; }

            candidates.sort_by(|a, b| {
                b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.machine_end.cmp(&b.machine_end))
                    .then(a.proc_time.cmp(&b.proc_time))
                    .then(a.job.cmp(&b.job))
            });

            let pick = if use_random && candidates.len() > 1 {
                let k = 3.min(candidates.len());
                rng.as_mut().unwrap().gen_range(0..k)
            } else {
                0
            };

            let chosen = &candidates[pick];
            let job = chosen.job;
            let op_idx = chosen.op_idx;
            let machine_id = chosen.machine;
            let start_time = chosen.start_time;
            let finish_time = start_time + chosen.proc_time;

            assignments[job][op_idx] = (machine_id, start_time, finish_time);
            machine_orders[machine_id].push(OpId { job, op_idx });
            job_next_op[job] += 1;
            job_ready_time[job] = finish_time;
            machine_available[machine_id] = finish_time;
            remaining_ops -= 1;
        }

        if remaining_ops == 0 { break; }

        let mut next_time = u32::MAX;
        for &t in machine_available.iter() {
            if t > time && t < next_time { next_time = t; }
        }
        for job in 0..num_jobs {
            if job_next_op[job] < data.job_ops[job].len() && job_ready_time[job] > time && job_ready_time[job] < next_time {
                next_time = job_ready_time[job];
            }
        }
        if next_time == u32::MAX { break; }
        time = next_time;
    }

    let makespan = assignments.iter()
        .map(|ops| ops.last().map(|&(_, _, f)| f).unwrap_or(0))
        .max().unwrap_or(0);

    Schedule { assignments, machine_orders, makespan }
}

// ============================================================================
// Tier 2: Critical Path Tabu Search
// ============================================================================

fn get_relation(a: OpId, b: OpId) -> Relation {
    if a.job != b.job { Relation::None }
    else if a.op_idx < b.op_idx { Relation::Lesser }
    else if a.op_idx > b.op_idx { Relation::Greater }
    else { Relation::Lesser }
}

fn find_critical_path(schedule: &Schedule, data: &ProblemData) -> Vec<OpId> {
    let mut op_machine_pos: Vec<Vec<(usize, usize)>> = data
        .job_ops.iter().map(|ops| vec![(0, 0); ops.len()]).collect();
    for (machine, order) in schedule.machine_orders.iter().enumerate() {
        for (pos, op_id) in order.iter().enumerate() {
            op_machine_pos[op_id.job][op_id.op_idx] = (machine, pos);
        }
    }

    let mut end_op: Option<OpId> = None;
    for job in 0..data.num_jobs {
        let last_idx = data.job_ops[job].len() - 1;
        let (_, _, finish) = schedule.assignments[job][last_idx];
        if finish == schedule.makespan {
            end_op = Some(OpId { job, op_idx: last_idx });
            break;
        }
    }

    let mut path = Vec::new();
    let mut current = match end_op { Some(op) => op, None => return path };

    loop {
        path.push(current);
        let (machine, pos) = op_machine_pos[current.job][current.op_idx];
        let (_, start_time, _) = schedule.assignments[current.job][current.op_idx];
        if start_time == 0 { break; }

        let mut found_pred = false;
        if current.op_idx > 0 {
            let (_, _, prev_finish) = schedule.assignments[current.job][current.op_idx - 1];
            if prev_finish == start_time {
                current = OpId { job: current.job, op_idx: current.op_idx - 1 };
                found_pred = true;
            }
        }
        if !found_pred && pos > 0 {
            let prev_machine_op = &schedule.machine_orders[machine][pos - 1];
            let (_, _, prev_finish) = schedule.assignments[prev_machine_op.job][prev_machine_op.op_idx];
            if prev_finish == start_time {
                current = *prev_machine_op;
                found_pred = true;
            }
        }
        if !found_pred {
            if current.op_idx > 0 {
                current = OpId { job: current.job, op_idx: current.op_idx - 1 };
            } else { break; }
        }
    }
    path.reverse();
    path
}

fn find_critical_blocks(critical_path: &[OpId], schedule: &Schedule) -> Vec<(usize, Vec<OpId>)> {
    if critical_path.is_empty() { return Vec::new(); }
    let mut blocks = Vec::new();
    let mut current_machine = schedule.assignments[critical_path[0].job][critical_path[0].op_idx].0;
    let mut current_block = vec![critical_path[0]];

    for i in 1..critical_path.len() {
        let op = critical_path[i];
        let machine = schedule.assignments[op.job][op.op_idx].0;
        if machine == current_machine {
            current_block.push(op);
        } else {
            if current_block.len() >= 2 {
                blocks.push((current_machine, current_block.clone()));
            }
            current_machine = machine;
            current_block = vec![op];
        }
    }
    if current_block.len() >= 2 {
        blocks.push((current_machine, current_block));
    }
    blocks
}

fn generate_n5_moves(blocks: &[(usize, Vec<OpId>)]) -> Vec<TabuMove> {
    let mut moves = Vec::new();
    for &(machine, ref block) in blocks {
        let len = block.len();
        if len < 2 { continue; }
        if get_relation(block[0], block[1]) == Relation::None {
            moves.push(TabuMove { op_a: block[0], op_b: block[1], machine });
        }
        if len > 2 {
            if get_relation(block[len-2], block[len-1]) == Relation::None {
                moves.push(TabuMove { op_a: block[len-2], op_b: block[len-1], machine });
            }
        }
    }
    moves
}

/// Wider neighborhood: ALL adjacent pairs in critical blocks
fn generate_full_block_moves(blocks: &[(usize, Vec<OpId>)]) -> Vec<TabuMove> {
    let mut moves = Vec::new();
    for &(machine, ref block) in blocks {
        for i in 0..block.len().saturating_sub(1) {
            if get_relation(block[i], block[i + 1]) == Relation::None {
                moves.push(TabuMove { op_a: block[i], op_b: block[i + 1], machine });
            }
        }
    }
    moves
}

fn apply_swap(schedule: &mut Schedule, mv: &TabuMove) {
    let machine = mv.machine;
    let order = &mut schedule.machine_orders[machine];
    let mut pos_a = None;
    let mut pos_b = None;
    for (i, op) in order.iter().enumerate() {
        if *op == mv.op_a { pos_a = Some(i); }
        if *op == mv.op_b { pos_b = Some(i); }
        if pos_a.is_some() && pos_b.is_some() { break; }
    }
    if let (Some(pa), Some(pb)) = (pos_a, pos_b) {
        order.swap(pa, pb);
    }
}

fn generate_reassignment_moves(critical_path: &[OpId], schedule: &Schedule, data: &ProblemData) -> Vec<(OpId, usize, usize)> {
    let mut moves = Vec::new();
    for &op_id in critical_path {
        let op_info = &data.job_ops[op_id.job][op_id.op_idx];
        if op_info.eligible_machines.len() <= 1 { continue; }
        let current_machine = schedule.assignments[op_id.job][op_id.op_idx].0;
        for &new_machine in &op_info.eligible_machines {
            if new_machine != current_machine {
                moves.push((op_id, current_machine, new_machine));
            }
        }
    }

    // For flexible problems: also try reassigning ops on bottleneck machines
    if data.is_flexible {
        // Find bottleneck machine (highest load)
        let mut machine_load = vec![0u32; data.num_machines];
        for (machine, order) in schedule.machine_orders.iter().enumerate() {
            for op_id in order {
                let (_, _, finish) = schedule.assignments[op_id.job][op_id.op_idx];
                if finish > machine_load[machine] {
                    machine_load[machine] = finish;
                }
            }
        }

        // Sort machines by load descending, take top 3 bottleneck machines
        let mut machine_indices: Vec<usize> = (0..data.num_machines).collect();
        machine_indices.sort_by(|&a, &b| machine_load[b].cmp(&machine_load[a]));

        let bottleneck_count = 3.min(machine_indices.len());
        let critical_set: HashSet<OpId> = critical_path.iter().copied().collect();

        for &machine in &machine_indices[..bottleneck_count] {
            let mut count = 0;
            for op_id in &schedule.machine_orders[machine] {
                if count >= 5 { break; } // limit moves per machine to keep search fast
                if critical_set.contains(op_id) { continue; } // already handled above
                let op_info = &data.job_ops[op_id.job][op_id.op_idx];
                if op_info.eligible_machines.len() <= 1 { continue; }
                for &new_machine in &op_info.eligible_machines {
                    if new_machine != machine {
                        moves.push((*op_id, machine, new_machine));
                    }
                }
                count += 1;
            }
        }
    }

    moves
}

// Machine reassignment: remove from old, insert at specified position on new
// (Now done inline in tabu_search for optimal position selection)

fn tabu_search(
    schedule: &mut Schedule,
    data: &ProblemData,
    params: &Hyperparameters,
    saver: &mut SolutionSaver,
    rng: &mut SmallRng,
) -> u32 {
    let mut best_makespan = schedule.makespan;
    let mut best_machine_orders = schedule.machine_orders.clone();
    let mut best_assignments = schedule.assignments.clone();

    let mut tabu_list: Vec<(TabuMove, usize)> = Vec::new();
    let mut iteration = 0usize;
    let mut idle_count = 0usize;
    let mut perturbation_count = 0usize;
    let max_perturbations = if data.is_flexible { 6 } else { 3 };

    loop {
        if params.max_iterations > 0 && iteration >= params.max_iterations { break; }

        // When stuck: perturb instead of terminating
        if idle_count >= params.max_idle_iterations {
            if perturbation_count >= max_perturbations { break; }

            if data.is_flexible && perturbation_count % 2 == 0 {
                // Machine reassignment perturbation: move random ops from busiest machines
                let mut machine_finish = vec![0u32; data.num_machines];
                for (m, order) in schedule.machine_orders.iter().enumerate() {
                    for op_id in order {
                        let (_, _, f) = schedule.assignments[op_id.job][op_id.op_idx];
                        machine_finish[m] = machine_finish[m].max(f);
                    }
                }
                let mut machines_by_load: Vec<usize> = (0..data.num_machines).collect();
                machines_by_load.sort_by(|&a, &b| machine_finish[b].cmp(&machine_finish[a]));

                // Try to move 2-3 ops from top loaded machines
                for &m in &machines_by_load[..2.min(machines_by_load.len())] {
                    let order_snapshot: Vec<OpId> = schedule.machine_orders[m].clone();
                    for op_id in &order_snapshot {
                        let op_info = &data.job_ops[op_id.job][op_id.op_idx];
                        if op_info.eligible_machines.len() <= 1 { continue; }
                        // Pick a random alternative machine
                        let alternatives: Vec<usize> = op_info.eligible_machines.iter()
                            .filter(|&&em| em != m)
                            .copied()
                            .collect();
                        if alternatives.is_empty() { continue; }
                        let new_m = alternatives[rng.gen_range(0..alternatives.len())];
                        schedule.machine_orders[m].retain(|op| *op != *op_id);
                        schedule.machine_orders[new_m].push(*op_id);
                        break; // one move per machine per perturbation
                    }
                }
            } else {
                // Standard perturbation: shuffle ops on critical path machines
                let critical_path = find_critical_path(schedule, data);
                if critical_path.is_empty() { break; }
                for &op_id in &critical_path {
                    let machine = schedule.assignments[op_id.job][op_id.op_idx].0;
                    let order = &mut schedule.machine_orders[machine];
                    if order.len() >= 2 {
                        let i = rng.gen_range(0..order.len());
                        let j = rng.gen_range(0..order.len());
                        if i != j && order[i].job != order[j].job {
                            order.swap(i, j);
                        }
                    }
                }
            }
            recompute_schedule(schedule, data);
            tabu_list.clear();
            idle_count = 0;
            perturbation_count += 1;
            continue;
        }

        let critical_path = find_critical_path(schedule, data);
        if critical_path.is_empty() { break; }
        let blocks = find_critical_blocks(&critical_path, schedule);
        // Use wider neighborhood when getting stuck
        let swap_moves = if idle_count > params.max_idle_iterations / 2 {
            generate_full_block_moves(&blocks)
        } else {
            generate_n5_moves(&blocks)
        };
        let reassign_moves = generate_reassignment_moves(&critical_path, schedule, data);

        let mut best_move: Option<(TabuMove, u32)> = None;
        let mut best_reassign: Option<(OpId, usize, usize, usize, u32)> = None; // (op, old_m, new_m, insert_pos, makespan)

        for mv in &swap_moves {
            let relation = get_relation(mv.op_a, mv.op_b);
            if relation != Relation::None { continue; }

            let mut trial = schedule.clone();
            apply_swap(&mut trial, mv);
            let new_makespan = recompute_schedule(&mut trial, data);

            let reverse_move = TabuMove { op_a: mv.op_b, op_b: mv.op_a, machine: mv.machine };
            let is_tabu = tabu_list.iter().any(|(m, expiry)| (*m == *mv || *m == reverse_move) && *expiry > iteration);
            let dominated_by_tabu = is_tabu && new_makespan >= best_makespan;

            if !dominated_by_tabu {
                if best_move.as_ref().map_or(true, |&(_, best_ms)| new_makespan < best_ms) {
                    best_move = Some((*mv, new_makespan));
                }
            }
        }

        for &(op_id, old_machine, new_machine) in &reassign_moves {
            let mut trial = schedule.clone();
            trial.machine_orders[old_machine].retain(|op| *op != op_id);
            let order_len = trial.machine_orders[new_machine].len();

            // For flexible problems, try multiple insertion positions
            // For single-machine problems, just append (it's irrelevant)
            let positions_to_try = if data.is_flexible && order_len <= 20 {
                order_len + 1 // try all positions for small queues
            } else if data.is_flexible {
                // Sample a few positions for large queues: start, end, middle, quartiles
                5
            } else {
                1 // just append
            };

            let sample_mode = data.is_flexible && order_len > 20;
            for idx in 0..positions_to_try {
                let pos = if sample_mode {
                    match idx {
                        0 => 0,
                        1 => order_len,
                        2 => order_len / 2,
                        3 => order_len / 4,
                        _ => 3 * order_len / 4,
                    }
                } else if positions_to_try == 1 {
                    order_len // append
                } else {
                    idx
                };

                let mut trial2 = trial.clone();
                trial2.machine_orders[new_machine].insert(pos, op_id);
                let new_makespan = recompute_schedule(&mut trial2, data);

                if best_reassign.as_ref().map_or(true, |&(_, _, _, _, best_ms)| new_makespan < best_ms) {
                    best_reassign = Some((op_id, old_machine, new_machine, pos, new_makespan));
                }
            }
        }

        let swap_makespan = best_move.map(|(_, ms)| ms).unwrap_or(u32::MAX);
        let reassign_makespan = best_reassign.map(|(_, _, _, _, ms)| ms).unwrap_or(u32::MAX);
        if swap_makespan == u32::MAX && reassign_makespan == u32::MAX { break; }

        if swap_makespan <= reassign_makespan {
            if let Some((mv, _)) = best_move {
                apply_swap(schedule, &mv);
                recompute_schedule(schedule, data);
                let reverse = TabuMove { op_a: mv.op_b, op_b: mv.op_a, machine: mv.machine };
                tabu_list.push((reverse, iteration + params.tabu_tenure));
                if schedule.makespan < best_makespan {
                    best_makespan = schedule.makespan;
                    best_machine_orders = schedule.machine_orders.clone();
                    best_assignments = schedule.assignments.clone();
                    idle_count = 0;
                } else { idle_count += 1; }
            }
        } else {
            if let Some((op_id, old_machine, new_machine, insert_pos, _)) = best_reassign {
                schedule.machine_orders[old_machine].retain(|op| *op != op_id);
                schedule.machine_orders[new_machine].insert(insert_pos, op_id);
                recompute_schedule(schedule, data);
                if schedule.makespan < best_makespan {
                    best_makespan = schedule.makespan;
                    best_machine_orders = schedule.machine_orders.clone();
                    best_assignments = schedule.assignments.clone();
                    idle_count = 0;
                } else { idle_count += 1; }
            }
        }

        // Periodic save
        if iteration % 25 == 0 && iteration > 0 {
            let sol = schedule_to_solution_from_assignments(&best_assignments, data.num_jobs);
            let _ = saver.save(&sol, best_makespan);
        }

        if iteration % 50 == 0 {
            tabu_list.retain(|&(_, expiry)| expiry > iteration);
        }
        iteration += 1;
    }

    schedule.machine_orders = best_machine_orders;
    schedule.assignments = best_assignments;
    schedule.makespan = best_makespan;
    best_makespan
}

// ============================================================================
// Solution Conversion
// ============================================================================

fn schedule_to_solution(schedule: &Schedule, data: &ProblemData) -> Solution {
    schedule_to_solution_from_assignments(&schedule.assignments, data.num_jobs)
}

fn schedule_to_solution_from_assignments(assignments: &[Vec<(usize, u32, u32)>], num_jobs: usize) -> Solution {
    let mut job_schedule = Vec::with_capacity(num_jobs);
    for job in 0..num_jobs {
        let ops: Vec<(usize, u32)> = assignments[job]
            .iter()
            .map(|&(machine, start_time, _)| (machine, start_time))
            .collect();
        job_schedule.push(ops);
    }
    Solution { job_schedule }
}

// ============================================================================
// Main Entry Point
// ============================================================================

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let params = match hyperparameters {
        Some(hp) => serde_json::from_value::<Hyperparameters>(Value::Object(hp.clone()))
            .unwrap_or_default(),
        None => Hyperparameters::default(),
    };

    let mut rng = SmallRng::from_seed(challenge.seed);
    let data = build_problem_data(challenge);
    let mut saver = SolutionSaver::new(save_solution);

    println!("=== DIAGNOSTICS ===");
    println!("Flexibility: is_flexible={}, avg_flexibility={:.2}", data.is_flexible, data.avg_flexibility);

    // Compute EET slack based on flexibility
    // For non-flexible (flow_shop/job_shop): 0 (strict EET, proven behavior)
    // For flexible: small slack to explore different machine assignments
    // v1 used avg_flexibility * 5 ≈ 15 for hybrid_flow_shop, which worked well
    let base_eet_slack = if data.is_flexible { 
        (data.avg_flexibility * 5.0) as u32
    } else { 0 };
    let eet_slack = base_eet_slack;

    let rules: &[DispatchRule] = if data.is_flexible { &FLEX_RULES } else { &ALL_RULES };

    // ---- Tier 1: Construction Heuristic with eager saving ----
    let mut best_schedule: Option<Schedule> = None;

    // Deterministic rules
    for &rule in rules {
        let schedule = construct_schedule(&data, challenge, rule, 0, None, 0); // always strict EET for deterministic
        println!("Rule {:?}: makespan = {}", rule, schedule.makespan);
        if best_schedule.as_ref().map_or(true, |best| schedule.makespan < best.makespan) {
            let sol = schedule_to_solution(&schedule, &data);
            let _ = saver.save(&sol, schedule.makespan);
            best_schedule = Some(schedule);
        }
    }

    // Also try deterministic rules with EET slack for flexible problems
    if data.is_flexible {
        for &rule in rules {
            let schedule = construct_schedule(&data, challenge, rule, 0, None, eet_slack);
            println!("Rule {:?} (slack={}): makespan = {}", rule, eet_slack, schedule.makespan);
            if best_schedule.as_ref().map_or(true, |best| schedule.makespan < best.makespan) {
                let sol = schedule_to_solution(&schedule, &data);
                let _ = saver.save(&sol, schedule.makespan);
                best_schedule = Some(schedule);
            }
        }
    }

    // Hierarchical cooperative construction (deterministic)
    {
        let schedule = construct_schedule_hierarchical(&data, challenge, None, 0);
        println!("Hierarchical: makespan = {}", schedule.makespan);
        if best_schedule.as_ref().map_or(true, |best| schedule.makespan < best.makespan) {
            let sol = schedule_to_solution(&schedule, &data);
            let _ = saver.save(&sol, schedule.makespan);
            best_schedule = Some(schedule);
        }
    }
    if data.is_flexible {
        let schedule = construct_schedule_hierarchical(&data, challenge, None, eet_slack);
        println!("Hierarchical (slack={}): makespan = {}", eet_slack, schedule.makespan);
        if best_schedule.as_ref().map_or(true, |best| schedule.makespan < best.makespan) {
            let sol = schedule_to_solution(&schedule, &data);
            let _ = saver.save(&sol, schedule.makespan);
            best_schedule = Some(schedule);
        }
    }

    // Random restarts
    let num_restarts = if data.is_flexible { params.construction_restarts * 2 } else { params.construction_restarts };
    let mut restart_best = u32::MAX;
    let mut restart_worst = 0u32;
    for i in 0..num_restarts {
        let seed: u64 = rng.gen();
        let mut local_rng = SmallRng::seed_from_u64(seed);

        // Vary EET slack for flexible problems
        let slack = if data.is_flexible {
            match i % 4 {
                0 => 0,                    // strict EET
                1 => base_eet_slack / 2,
                2 => base_eet_slack,
                _ => base_eet_slack * 2,
            }
        } else { 0 };

        let schedule = if i % 4 == 0 {
            construct_schedule_hierarchical(&data, challenge, Some(&mut local_rng), slack)
        } else {
            let rule = rules[rng.gen_range(0..rules.len())];
            let random_k = rng.gen_range(2..=5);
            construct_schedule(&data, challenge, rule, random_k, Some(&mut local_rng), slack)
        };

        if best_schedule.as_ref().map_or(true, |best| schedule.makespan < best.makespan) {
            let sol = schedule_to_solution(&schedule, &data);
            let _ = saver.save(&sol, schedule.makespan);
            best_schedule = Some(schedule.clone());
        }
        restart_best = restart_best.min(schedule.makespan);
        restart_worst = restart_worst.max(schedule.makespan);
    }

    let mut best_schedule = best_schedule.unwrap();
    let construction_best = best_schedule.makespan;
    println!("Random restarts: best={}, worst={}", restart_best, restart_worst);
    println!("Construction best: {}", construction_best);

    // ---- Tier 2: Critical Path Tabu Search ----
    let num_tabu_restarts = if data.is_flexible { params.tabu_restarts + 3 } else { params.tabu_restarts };
    for restart in 0..num_tabu_restarts {
        let slack = if data.is_flexible { rng.gen_range(0..=base_eet_slack * 2) } else { 0 };
        let mut schedule = if restart == 0 {
            best_schedule.clone()
        } else if restart % 3 == 0 {
            let seed: u64 = rng.gen();
            let mut local_rng = SmallRng::seed_from_u64(seed);
            construct_schedule_hierarchical(&data, challenge, Some(&mut local_rng), slack)
        } else {
            let rule = rules[rng.gen_range(0..rules.len())];
            let seed: u64 = rng.gen();
            let mut local_rng = SmallRng::seed_from_u64(seed);
            construct_schedule(&data, challenge, rule, params.construction_top_k.max(2), Some(&mut local_rng), slack)
        };

        let input_makespan = schedule.makespan;
        let new_makespan = tabu_search(&mut schedule, &data, &params, &mut saver, &mut rng);
        println!("Tabu restart {}: {} -> {}", restart, input_makespan, new_makespan);

        if new_makespan < best_schedule.makespan {
            best_schedule = schedule;
        }
    }

    // ---- Tier 3: Rescue phase if in danger zone ----
    {
        let improvement_ratio = if construction_best > 0 {
            (construction_best as f64 - saver.best_makespan as f64) / construction_best as f64
        } else { 1.0 };
        println!("Tabu improvement over construction: {:.2}% ({} -> {})", 
            improvement_ratio * 100.0, construction_best, saver.best_makespan);

        // For flexible problems, always run rescue (cheap insurance)
        // For non-flexible, only run when tabu barely improved
        let needs_rescue = if data.is_flexible {
            improvement_ratio < 0.05  // more generous threshold for flexible
        } else {
            improvement_ratio < 0.02
        };

        if needs_rescue {
            println!("RESCUE: running extended construction + tabu");

            let rescue_restarts = if data.is_flexible { 200 } else { 100 };
            // Extended construction with eager saving
            let mut rescue_best: Option<Schedule> = None;
            for i in 0..rescue_restarts {
                let seed: u64 = rng.gen();
                let mut local_rng = SmallRng::seed_from_u64(seed);
                let slack = if data.is_flexible { rng.gen_range(0..=base_eet_slack * 2) } else { 0 };

                let schedule = if i % 5 == 0 {
                    construct_schedule_hierarchical(&data, challenge, Some(&mut local_rng), slack)
                } else {
                    let rule = rules[i % rules.len()];
                    let random_k = local_rng.gen_range(2..=6);
                    construct_schedule(&data, challenge, rule, random_k, Some(&mut local_rng), slack)
                };

                // Eager save during rescue construction
                if schedule.makespan < saver.best_makespan {
                    let sol = schedule_to_solution(&schedule, &data);
                    let _ = saver.save(&sol, schedule.makespan);
                }

                if rescue_best.as_ref().map_or(true, |best| schedule.makespan < best.makespan) {
                    rescue_best = Some(schedule);
                }
            }

            // Run multiple tabu restarts from rescue constructions
            let rescue_tabu_restarts = if data.is_flexible { 4 } else { 2 };
            for r in 0..rescue_tabu_restarts {
                let mut schedule = if r == 0 {
                    rescue_best.clone().unwrap_or_else(|| best_schedule.clone())
                } else {
                    let seed: u64 = rng.gen();
                    let mut local_rng = SmallRng::seed_from_u64(seed);
                    let slack = if data.is_flexible { rng.gen_range(0..=base_eet_slack * 2) } else { 0 };
                    if r % 2 == 0 {
                        construct_schedule_hierarchical(&data, challenge, Some(&mut local_rng), slack)
                    } else {
                        let rule = rules[rng.gen_range(0..rules.len())];
                        construct_schedule(&data, challenge, rule, rng.gen_range(2..=5), Some(&mut local_rng), slack)
                    }
                };

                let input = schedule.makespan;
                let result = tabu_search(&mut schedule, &data, &params, &mut saver, &mut rng);
                println!("RESCUE tabu restart {}: {} -> {}", r, input, result);

                if result < best_schedule.makespan {
                    best_schedule = schedule;
                }
            }

            println!("RESCUE final: {}", saver.best_makespan);
        }
    }

    println!("Final makespan: {}", saver.best_makespan);
    println!("=== END DIAGNOSTICS ===");
    Ok(())
}