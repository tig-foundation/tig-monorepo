// TIG's UI uses the pattern tig_challenges::<challenge_name> to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::job_scheduling::*;

#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct Hyperparameters {
    pub switch: String,
    pub strict_effort: usize,
    pub strict_random_restarts: Option<usize>,
    pub strict_neh_limit: Option<usize>,
    pub strict_local_iters: Option<usize>,
    pub strict_cds_limit: Option<usize>,
    pub strict_random_top_k: Option<usize>,
    pub strict_weight_jitter: Option<f64>,
    pub parallel_effort: usize,
    // --- random switch hyperparameters (uses f_parallel, prefixed with random_) ---
    pub random_effort: usize,
    // --- f_chaotic hyperparameters (prefixed with chaotic_) ---
    pub chaotic_effort: usize,
    pub chaotic_n_candidates: usize,
    pub chaotic_k_candidates: usize,
    pub chaotic_machine_top_k: usize,
    pub chaotic_weight_jitter: f64,
    // --- f_complex hyperparameters (prefixed with complex_) ---
    pub complex_effort: usize,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            switch: "strict".to_string(),
            strict_effort: 3,
            strict_random_restarts: None,
            strict_neh_limit: None,
            strict_local_iters: None,
            strict_cds_limit: None,
            strict_random_top_k: None,
            strict_weight_jitter: None,
            parallel_effort: 5,
            random_effort: 5,
            chaotic_effort: 2,
            chaotic_n_candidates: 320,
            chaotic_k_candidates: 8,
            chaotic_machine_top_k: 5,
            chaotic_weight_jitter: 0.25,
            complex_effort: 5,
        }
    }
}

pub fn help() {
    println!("Combined meta-algorithm that delegates to sub-algorithms based on the 'switch' hyperparameter.");
    println!("Hyperparameters:");
    println!("  switch: selects algorithm (\"strict\", \"parallel\", \"random\", \"chaotic\", or \"complex\", default \"strict\").");
    println!("  strict_effort: effort for strict algorithm (default 3).");
    println!("  strict_random_restarts: override random restarts for strict.");
    println!("  strict_neh_limit: override NEH insertion limit for strict.");
    println!("  strict_local_iters: override local search iterations for strict.");
    println!("  strict_cds_limit: override CDS orders for strict.");
    println!("  strict_random_top_k: override random top-k for strict.");
    println!("  strict_weight_jitter: override weight jitter for strict.");
    println!("  parallel_effort: effort for parallel algorithm (default 5).");
    println!("  random_effort: effort for random algorithm (uses f_parallel, default 5).");
    println!("  chaotic_effort: effort for chaotic algorithm (default 2).");
    println!("  chaotic_n_candidates: randomized constructive schedules (default 320).");
    println!("  chaotic_k_candidates: top candidates refined with tabu (default 8).");
    println!("  chaotic_machine_top_k: top-K fastest machines per job (default 5).");
    println!("  chaotic_weight_jitter: random weight jitter per restart (default 0.25).");
    println!("  complex_effort: effort for complex algorithm (default 5).");
}

fn build_sub_hyperparameters(combined: &Hyperparameters, prefix: &str) -> Map<String, Value> {
    let serialized = serde_json::to_value(combined).unwrap_or(Value::Object(Map::new()));
    let all_fields = match serialized {
        Value::Object(map) => map,
        _ => Map::new(),
    };
    let prefix_underscore = format!("{}_", prefix);
    let mut sub_hp = Map::new();
    for (key, value) in all_fields.iter() {
        if let Some(stripped) = key.strip_prefix(&prefix_underscore) {
            if value.is_null() {
                continue;
            }
            sub_hp.insert(stripped.to_string(), value.clone());
        }
    }
    sub_hp
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let combined: Hyperparameters = match hyperparameters {
        Some(hp) => serde_json::from_value::<Hyperparameters>(Value::Object(hp.clone()))
            .map_err(|e| anyhow!("Failed to parse hyperparameters: {}", e))?,
        None => Hyperparameters::default(),
    };

    match combined.switch.as_str() {
        "strict" => {
            let sub_hp = build_sub_hyperparameters(&combined, "strict");
            strict::solve_challenge(challenge, save_solution, &Some(sub_hp))
        }
        "parallel" => {
            let sub_hp = build_sub_hyperparameters(&combined, "parallel");
            parallel::solve_challenge(challenge, save_solution, &Some(sub_hp))
        }
        "random" => {
            let sub_hp = build_sub_hyperparameters(&combined, "random");
            parallel::solve_challenge(challenge, save_solution, &Some(sub_hp))
        }
        "chaotic" => {
            let sub_hp = build_sub_hyperparameters(&combined, "chaotic");
            chaotic::solve_challenge(challenge, save_solution, &Some(sub_hp))
        }
        "complex" => {
            let sub_hp = build_sub_hyperparameters(&combined, "complex");
            complex::solve_challenge(challenge, save_solution, &Some(sub_hp))
        }
        other => Err(anyhow!("Unknown switch value: '{}'. Expected 'strict', 'parallel', 'random', 'chaotic', or 'complex'.", other)),
    }
}

#[allow(dead_code)]
mod strict {
// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::cmp::Ordering;
use tig_challenges::job_scheduling::*;

#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct Hyperparameters {
    pub effort: usize,
    pub random_restarts: Option<usize>,
    pub neh_limit: Option<usize>,
    pub local_iters: Option<usize>,
    pub cds_limit: Option<usize>,
    pub random_top_k: Option<usize>,
    pub weight_jitter: Option<f64>,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            effort: 3,
            random_restarts: None,
            neh_limit: None,
            local_iters: None,
            cds_limit: None,
            random_top_k: None,
            weight_jitter: None,
        }
    }
}

pub fn help() {
    println!("Strict-flow optimized heuristic: flow-shop ordering + dispatch rules.");
    println!("Hyperparameters (all optional):");
    println!("  effort: scales ordering depth, local search, and restarts (default 3).");
    println!("  random_restarts: override number of randomized restarts.");
    println!("  neh_limit: override number of jobs inserted in NEH.");
    println!("  local_iters: override insertion local search iterations.");
    println!("  cds_limit: override number of CDS orders (max m-1).");
    println!("  random_top_k: override random choice among top-k candidates.");
    println!("  weight_jitter: override weight jitter for weighted rule (0.0..0.8).");
}

#[derive(Clone)]
struct OpInfo {
    machine: usize,
    duration: u32,
}

#[derive(Clone, Copy)]
struct RuleWeights {
    work: f64,
    ops: f64,
    end: f64,
    proc: f64,
}

impl Default for RuleWeights {
    fn default() -> Self {
        Self {
            work: 1.2,
            ops: 0.4,
            end: 1.3,
            proc: 0.3,
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
enum DispatchRule {
    MostWorkRemaining,
    MostOpsRemaining,
    ShortestProcTime,
    LongestProcTime,
    Weighted,
    BottleneckRemaining,
    OrderOnly,
}

#[derive(Clone, Copy)]
struct DispatchParams<'a> {
    rule: DispatchRule,
    weights: RuleWeights,
    order_rank: Option<&'a [usize]>,
    order_weight: f64,
    order_max: f64,
}

#[derive(Clone, Copy)]
struct Candidate {
    job: usize,
    priority: f64,
    machine_end: u32,
    proc_time: u32,
}

#[derive(Clone)]
struct ScheduleResult {
    job_schedule: Vec<Vec<(usize, u32)>>,
    makespan: u32,
}

#[derive(Clone, Copy)]
struct RestartResult {
    makespan: u32,
    rule: DispatchRule,
    random_top_k: usize,
    seed: u64,
}

fn build_job_products(challenge: &Challenge) -> Result<Vec<usize>> {
    let mut job_products = Vec::with_capacity(challenge.num_jobs);
    for (product, count) in challenge.jobs_per_product.iter().enumerate() {
        for _ in 0..*count {
            job_products.push(product);
        }
    }
    if job_products.len() != challenge.num_jobs {
        return Err(anyhow!(
            "Job count mismatch. Expected {}, got {}",
            challenge.num_jobs,
            job_products.len()
        ));
    }
    Ok(job_products)
}

fn build_op_info(challenge: &Challenge) -> Result<Vec<Vec<OpInfo>>> {
    let mut product_ops = Vec::with_capacity(challenge.product_processing_times.len());
    for (product_idx, ops) in challenge.product_processing_times.iter().enumerate() {
        let mut op_infos = Vec::with_capacity(ops.len());
        for (op_idx, op) in ops.iter().enumerate() {
            if op.is_empty() {
                return Err(anyhow!(
                    "Product {} op {} has no eligible machines",
                    product_idx,
                    op_idx
                ));
            }
            let mut min_time = u32::MAX;
            let mut min_machine = usize::MAX;
            for (&machine, &duration) in op.iter() {
                if machine >= challenge.num_machines {
                    return Err(anyhow!(
                        "Product {} op {} has invalid machine {}",
                        product_idx,
                        op_idx,
                        machine
                    ));
                }
                if duration < min_time || (duration == min_time && machine < min_machine) {
                    min_time = duration;
                    min_machine = machine;
                }
            }
            op_infos.push(OpInfo {
                machine: min_machine,
                duration: min_time,
            });
        }
        product_ops.push(op_infos);
    }
    Ok(product_ops)
}

fn build_product_work_times(product_ops: &[Vec<OpInfo>]) -> Vec<Vec<f64>> {
    let mut product_times = Vec::with_capacity(product_ops.len());
    for ops in product_ops.iter() {
        let mut times = Vec::with_capacity(ops.len());
        for op in ops.iter() {
            times.push(op.duration as f64);
        }
        product_times.push(times);
    }
    product_times
}

fn build_job_stats(
    job_products: &[usize],
    product_times: &[Vec<f64>],
) -> (Vec<usize>, Vec<f64>) {
    let mut job_ops_len = Vec::with_capacity(job_products.len());
    let mut job_total_work = Vec::with_capacity(job_products.len());
    for &product in job_products.iter() {
        let ops = &product_times[product];
        job_ops_len.push(ops.len());
        job_total_work.push(ops.iter().sum());
    }
    (job_ops_len, job_total_work)
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
    machine_end: u32,
    proc_time: u32,
    weights: RuleWeights,
    scales: PriorityScales,
) -> f64 {
    let work_norm = remaining_work / scales.work;
    let ops_norm = remaining_ops as f64 / scales.ops;
    let end_norm = machine_end as f64 / scales.end;
    let proc_norm = proc_time as f64 / scales.proc;
    weights.work * work_norm
        + weights.ops * ops_norm
        - weights.end * end_norm
        - weights.proc * proc_norm
}

fn compare_candidate(a: &Candidate, b: &Candidate) -> Ordering {
    let eps = 1e-9_f64;
    if (b.priority - a.priority).abs() > eps {
        return b.priority.partial_cmp(&a.priority).unwrap_or(Ordering::Equal);
    }
    if a.machine_end != b.machine_end {
        return a.machine_end.cmp(&b.machine_end);
    }
    if a.proc_time != b.proc_time {
        return a.proc_time.cmp(&b.proc_time);
    }
    a.job.cmp(&b.job)
}

fn run_dispatch_rule(
    challenge: &Challenge,
    job_products: &[usize],
    product_ops: &[Vec<OpInfo>],
    product_work_times: &[Vec<f64>],
    job_ops_len: &[usize],
    job_total_work: &[f64],
    job_bottleneck_work: Option<&[f64]>,
    bottleneck_machine: Option<usize>,
    params: DispatchParams,
    scales: PriorityScales,
    random_top_k: Option<usize>,
    shuffle_machines: bool,
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
    let mut job_bottleneck_remaining = job_bottleneck_work.map(|values| values.to_vec());

    let mut remaining_ops = job_ops_len.iter().sum::<usize>();
    let mut time = 0u32;
    let random_top_k = random_top_k.unwrap_or(0);
    let mut rng = rng;
    let use_random = random_top_k > 1 && rng.is_some();

    while remaining_ops > 0 {
        let mut available_machines = (0..num_machines)
            .filter(|&m| machine_available_time[m] <= time)
            .collect::<Vec<usize>>();
        available_machines.sort_unstable();
        if shuffle_machines && use_random {
            use rand::seq::SliceRandom;
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
                    let op = &product_ops[product][op_idx];
                    if op.machine != machine {
                        continue;
                    }
                    let proc_time = op.duration;
                    let machine_end = time.max(machine_available_time[machine]) + proc_time;

                    let remaining_ops = job_ops_len[job] - job_next_op_idx[job];
                    let mut priority = match params.rule {
                        DispatchRule::MostWorkRemaining => job_remaining_work[job],
                        DispatchRule::MostOpsRemaining => remaining_ops as f64,
                        DispatchRule::ShortestProcTime => -(proc_time as f64),
                        DispatchRule::LongestProcTime => proc_time as f64,
                        DispatchRule::Weighted => weighted_priority(
                            job_remaining_work[job],
                            remaining_ops,
                            machine_end,
                            proc_time,
                            params.weights,
                            scales,
                        ),
                        DispatchRule::BottleneckRemaining => job_bottleneck_remaining
                            .as_ref()
                            .map_or(0.0, |vals| vals[job]),
                        DispatchRule::OrderOnly => 0.0,
                    };

                    if let Some(rank) = params.order_rank {
                        let order_score = 1.0 - (rank[job] as f64 / params.order_max.max(1.0));
                        priority += params.order_weight * order_score;
                    }

                    candidates.push(Candidate {
                        job,
                        priority,
                        machine_end,
                        proc_time,
                    });
                }

                if !candidates.is_empty() {
                    candidates.sort_unstable_by(compare_candidate);
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
                    let op = &product_ops[product][op_idx];
                    if op.machine != machine {
                        continue;
                    }
                    let proc_time = op.duration;
                    let machine_end = time.max(machine_available_time[machine]) + proc_time;

                    let remaining_ops = job_ops_len[job] - job_next_op_idx[job];
                    let mut priority = match params.rule {
                        DispatchRule::MostWorkRemaining => job_remaining_work[job],
                        DispatchRule::MostOpsRemaining => remaining_ops as f64,
                        DispatchRule::ShortestProcTime => -(proc_time as f64),
                        DispatchRule::LongestProcTime => proc_time as f64,
                        DispatchRule::Weighted => weighted_priority(
                            job_remaining_work[job],
                            remaining_ops,
                            machine_end,
                            proc_time,
                            params.weights,
                            scales,
                        ),
                        DispatchRule::BottleneckRemaining => job_bottleneck_remaining
                            .as_ref()
                            .map_or(0.0, |vals| vals[job]),
                        DispatchRule::OrderOnly => 0.0,
                    };

                    if let Some(rank) = params.order_rank {
                        let order_score = 1.0 - (rank[job] as f64 / params.order_max.max(1.0));
                        priority += params.order_weight * order_score;
                    }

                    let candidate = Candidate {
                        job,
                        priority,
                        machine_end,
                        proc_time,
                    };

                    let dominated = best_candidate
                        .as_ref()
                        .map_or(false, |best| compare_candidate(&candidate, best) != Ordering::Less);
                    if !dominated {
                        best_candidate = Some(candidate);
                    }
                }
            }

            if let Some(candidate) = best_candidate {
                let job = candidate.job;
                let product = job_products[job];
                let op_idx = job_next_op_idx[job];
                let op = &product_ops[product][op_idx];
                let proc_time = op.duration;

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
                if let (Some(bottleneck), Some(ref mut remaining)) =
                    (bottleneck_machine, &mut job_bottleneck_remaining)
                {
                    if machine == bottleneck {
                        if let Some(value) = remaining.get_mut(job) {
                            *value = (*value - proc_time as f64).max(0.0);
                        }
                    }
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

fn extract_strict_sequences(
    product_ops: &[Vec<OpInfo>],
) -> (Vec<Vec<usize>>, Vec<Vec<u32>>) {
    let mut machines = Vec::with_capacity(product_ops.len());
    let mut proc_times = Vec::with_capacity(product_ops.len());
    for ops in product_ops.iter() {
        let mut machine_seq = Vec::with_capacity(ops.len());
        let mut proc_seq = Vec::with_capacity(ops.len());
        for op in ops.iter() {
            machine_seq.push(op.machine);
            proc_seq.push(op.duration);
        }
        machines.push(machine_seq);
        proc_times.push(proc_seq);
    }
    (machines, proc_times)
}

fn compute_makespan_order(
    job_order: &[usize],
    job_products: &[usize],
    job_ops_len: &[usize],
    product_machine_seq: &[Vec<usize>],
    product_proc_times: &[Vec<u32>],
    num_machines: usize,
) -> Result<u32> {
    let num_jobs = job_products.len();
    let mut included = vec![false; num_jobs];
    let mut job_rank = vec![usize::MAX; num_jobs];
    for (rank, &job) in job_order.iter().enumerate() {
        if job >= num_jobs {
            return Err(anyhow!("Invalid job id {}", job));
        }
        included[job] = true;
        job_rank[job] = rank;
    }

    let mut job_next_op_idx = vec![0usize; num_jobs];
    let mut job_ready_time = vec![0u32; num_jobs];
    let mut machine_available_time = vec![0u32; num_machines];
    let mut remaining_ops = job_order
        .iter()
        .map(|&job| job_ops_len[job])
        .sum::<usize>();
    let mut time = 0u32;

    while remaining_ops > 0 {
        let mut available_machines = (0..num_machines)
            .filter(|&m| machine_available_time[m] <= time)
            .collect::<Vec<usize>>();
        available_machines.sort_unstable();

        let mut scheduled_any = false;
        for &machine in available_machines.iter() {
            let mut best_job: Option<usize> = None;
            let mut best_rank = usize::MAX;

            for job in 0..num_jobs {
                if !included[job] {
                    continue;
                }
                if job_next_op_idx[job] >= job_ops_len[job] {
                    continue;
                }
                if job_ready_time[job] > time {
                    continue;
                }
                let product = job_products[job];
                let op_idx = job_next_op_idx[job];
                let op_machine = product_machine_seq[product][op_idx];
                if op_machine != machine {
                    continue;
                }
                let rank = job_rank[job];
                if rank < best_rank {
                    best_rank = rank;
                    best_job = Some(job);
                }
            }

            if let Some(job) = best_job {
                let product = job_products[job];
                let op_idx = job_next_op_idx[job];
                let proc_time = product_proc_times[product][op_idx];
                let start_time = time
                    .max(machine_available_time[machine])
                    .max(job_ready_time[job]);
                let end_time = start_time + proc_time;
                job_next_op_idx[job] += 1;
                job_ready_time[job] = end_time;
                machine_available_time[machine] = end_time;
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
            if !included[job] {
                continue;
            }
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

    let mut makespan = 0u32;
    for &job in job_order.iter() {
        if job_ready_time[job] > makespan {
            makespan = job_ready_time[job];
        }
    }
    Ok(makespan)
}

fn schedule_strict_order(
    job_order: &[usize],
    job_products: &[usize],
    job_ops_len: &[usize],
    product_machine_seq: &[Vec<usize>],
    product_proc_times: &[Vec<u32>],
    num_machines: usize,
) -> Result<ScheduleResult> {
    let num_jobs = job_products.len();
    if job_order.len() != num_jobs {
        return Err(anyhow!(
            "Strict scheduling requires a full job order (expected {}, got {})",
            num_jobs,
            job_order.len()
        ));
    }
    let mut job_rank = vec![0usize; num_jobs];
    for (rank, &job) in job_order.iter().enumerate() {
        if job >= num_jobs {
            return Err(anyhow!("Invalid job id {}", job));
        }
        job_rank[job] = rank;
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

    while remaining_ops > 0 {
        let mut available_machines = (0..num_machines)
            .filter(|&m| machine_available_time[m] <= time)
            .collect::<Vec<usize>>();
        available_machines.sort_unstable();

        let mut scheduled_any = false;
        for &machine in available_machines.iter() {
            let mut best_job: Option<usize> = None;
            let mut best_rank = usize::MAX;

            for job in 0..num_jobs {
                if job_next_op_idx[job] >= job_ops_len[job] {
                    continue;
                }
                if job_ready_time[job] > time {
                    continue;
                }
                let product = job_products[job];
                let op_idx = job_next_op_idx[job];
                let op_machine = product_machine_seq[product][op_idx];
                if op_machine != machine {
                    continue;
                }
                let rank = job_rank[job];
                if rank < best_rank {
                    best_rank = rank;
                    best_job = Some(job);
                }
            }

            if let Some(job) = best_job {
                let product = job_products[job];
                let op_idx = job_next_op_idx[job];
                let proc_time = product_proc_times[product][op_idx];
                let start_time = time
                    .max(machine_available_time[machine])
                    .max(job_ready_time[job]);
                let end_time = start_time + proc_time;

                job_schedule[job].push((machine, start_time));
                job_next_op_idx[job] += 1;
                job_ready_time[job] = end_time;
                machine_available_time[machine] = end_time;
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

fn order_by_total_work(job_total_work: &[f64], descending: bool) -> Vec<usize> {
    let mut jobs = (0..job_total_work.len()).collect::<Vec<usize>>();
    jobs.sort_unstable_by(|&a, &b| {
        let ord = if descending {
            job_total_work[b]
                .partial_cmp(&job_total_work[a])
                .unwrap_or(Ordering::Equal)
        } else {
            job_total_work[a]
                .partial_cmp(&job_total_work[b])
                .unwrap_or(Ordering::Equal)
        };
        if ord == Ordering::Equal {
            a.cmp(&b)
        } else {
            ord
        }
    });
    jobs
}

fn order_by_ops(job_ops_len: &[usize], descending: bool) -> Vec<usize> {
    let mut jobs = (0..job_ops_len.len()).collect::<Vec<usize>>();
    jobs.sort_unstable_by(|&a, &b| {
        let ord = if descending {
            job_ops_len[b].cmp(&job_ops_len[a])
        } else {
            job_ops_len[a].cmp(&job_ops_len[b])
        };
        if ord == Ordering::Equal {
            a.cmp(&b)
        } else {
            ord
        }
    });
    jobs
}

fn palmer_order(
    job_products: &[usize],
    product_proc_times: &[Vec<u32>],
) -> Vec<usize> {
    let num_jobs = job_products.len();
    let m = product_proc_times
        .first()
        .map(|ops| ops.len())
        .unwrap_or(0) as f64;
    let mut scores = vec![0.0; num_jobs];
    for job in 0..num_jobs {
        let product = job_products[job];
        let times = &product_proc_times[product];
        let mut score = 0.0;
        for (k, &p) in times.iter().enumerate() {
            let weight = m - 2.0 * (k as f64) - 1.0;
            score += weight * (p as f64);
        }
        scores[job] = score;
    }
    let mut jobs = (0..num_jobs).collect::<Vec<usize>>();
    jobs.sort_unstable_by(|&a, &b| {
        let ord = scores[b].partial_cmp(&scores[a]).unwrap_or(Ordering::Equal);
        if ord == Ordering::Equal {
            a.cmp(&b)
        } else {
            ord
        }
    });
    jobs
}

fn gupta_order(
    job_products: &[usize],
    product_proc_times: &[Vec<u32>],
) -> Vec<usize> {
    let num_jobs = job_products.len();
    let m = product_proc_times
        .first()
        .map(|ops| ops.len())
        .unwrap_or(0);
    if m < 3 {
        return (0..num_jobs).collect();
    }

    let mut min_adj = vec![0.0; num_jobs];
    let mut group_a: Vec<usize> = Vec::new();
    let mut group_b: Vec<usize> = Vec::new();
    for job in 0..num_jobs {
        let product = job_products[job];
        let times = &product_proc_times[product];
        let first = times.first().copied().unwrap_or(0) as f64;
        let last = times.last().copied().unwrap_or(0) as f64;
        let mut best = f64::INFINITY;
        for k in 0..(m - 1) {
            let sum = (times[k] + times[k + 1]) as f64;
            if sum < best {
                best = sum;
            }
        }
        min_adj[job] = best;
        if first < last {
            group_a.push(job);
        } else {
            group_b.push(job);
        }
    }

    group_a.sort_unstable_by(|&a, &b| {
        let ord = min_adj[a].partial_cmp(&min_adj[b]).unwrap_or(Ordering::Equal);
        if ord == Ordering::Equal {
            a.cmp(&b)
        } else {
            ord
        }
    });
    group_b.sort_unstable_by(|&a, &b| {
        let ord = min_adj[b].partial_cmp(&min_adj[a]).unwrap_or(Ordering::Equal);
        if ord == Ordering::Equal {
            a.cmp(&b)
        } else {
            ord
        }
    });

    let mut order = group_a;
    order.extend(group_b);
    order
}

fn johnson_order(a: &[f64], b: &[f64]) -> Vec<usize> {
    let mut left: Vec<usize> = Vec::new();
    let mut right: Vec<usize> = Vec::new();
    for job in 0..a.len() {
        if a[job] <= b[job] {
            left.push(job);
        } else {
            right.push(job);
        }
    }
    left.sort_unstable_by(|&i, &j| {
        let ord = a[i].partial_cmp(&a[j]).unwrap_or(Ordering::Equal);
        if ord == Ordering::Equal {
            i.cmp(&j)
        } else {
            ord
        }
    });
    right.sort_unstable_by(|&i, &j| {
        let ord = b[j].partial_cmp(&b[i]).unwrap_or(Ordering::Equal);
        if ord == Ordering::Equal {
            i.cmp(&j)
        } else {
            ord
        }
    });
    let mut order = left;
    order.extend(right);
    order
}

fn cds_orders(
    job_products: &[usize],
    product_proc_times: &[Vec<u32>],
    limit: usize,
) -> Vec<Vec<usize>> {
    let num_jobs = job_products.len();
    let m = product_proc_times
        .first()
        .map(|ops| ops.len())
        .unwrap_or(0);
    let max_k = m.saturating_sub(1).min(limit);
    let mut orders = Vec::new();
    for k in 1..=max_k {
        let mut a = vec![0.0; num_jobs];
        let mut b = vec![0.0; num_jobs];
        for job in 0..num_jobs {
            let product = job_products[job];
            let times = &product_proc_times[product];
            let mut sum_a = 0u64;
            let mut sum_b = 0u64;
            for idx in 0..k {
                sum_a += times[idx] as u64;
            }
            for idx in k..m {
                sum_b += times[idx] as u64;
            }
            a[job] = sum_a as f64;
            b[job] = sum_b as f64;
        }
        let order = johnson_order(&a, &b);
        orders.push(order);
    }
    orders
}

fn bottleneck_order(
    job_products: &[usize],
    product_ops: &[Vec<OpInfo>],
) -> Vec<usize> {
    let num_jobs = job_products.len();
    let num_machines = product_ops
        .iter()
        .flat_map(|ops| ops.iter().map(|op| op.machine))
        .max()
        .unwrap_or(0)
        + 1;
    let mut machine_load = vec![0u64; num_machines];

    for &product in job_products.iter() {
        for op in product_ops[product].iter() {
            machine_load[op.machine] += op.duration as u64;
        }
    }

    let (bottleneck, _) = machine_load
        .iter()
        .enumerate()
        .max_by_key(|&(_, load)| load)
        .unwrap_or((0usize, &0u64));

    let mut job_bottleneck = vec![0u64; num_jobs];
    for job in 0..num_jobs {
        let product = job_products[job];
        let mut sum = 0u64;
        for op in product_ops[product].iter() {
            if op.machine == bottleneck {
                sum += op.duration as u64;
            }
        }
        job_bottleneck[job] = sum;
    }

    let mut jobs = (0..num_jobs).collect::<Vec<usize>>();
    jobs.sort_unstable_by(|&a, &b| {
        let ord = job_bottleneck[b].cmp(&job_bottleneck[a]);
        if ord == Ordering::Equal {
            a.cmp(&b)
        } else {
            ord
        }
    });
    jobs
}

fn build_order_rank(order: &[usize], num_jobs: usize) -> Vec<usize> {
    let mut rank = vec![0usize; num_jobs];
    for (idx, &job) in order.iter().enumerate() {
        if job < num_jobs {
            rank[job] = idx;
        }
    }
    rank
}

fn neh_insertion(
    base_order: &[usize],
    job_products: &[usize],
    job_ops_len: &[usize],
    product_machine_seq: &[Vec<usize>],
    product_proc_times: &[Vec<u32>],
    num_machines: usize,
    insert_limit: usize,
) -> Result<Vec<usize>> {
    let mut order: Vec<usize> = Vec::new();
    let limit = insert_limit.min(base_order.len());
    for &job in base_order.iter().take(limit) {
        let mut best_pos = 0usize;
        let mut best_makespan = u32::MAX;
        for pos in 0..=order.len() {
            let mut candidate = order.clone();
            candidate.insert(pos, job);
            let makespan = compute_makespan_order(
                &candidate,
                job_products,
                job_ops_len,
                product_machine_seq,
                product_proc_times,
                num_machines,
            )?;
            if makespan < best_makespan {
                best_makespan = makespan;
                best_pos = pos;
            }
        }
        order.insert(best_pos, job);
    }
    order.extend(base_order.iter().skip(limit).copied());
    Ok(order)
}

fn insertion_local_search(
    order: &[usize],
    job_products: &[usize],
    job_ops_len: &[usize],
    product_machine_seq: &[Vec<usize>],
    product_proc_times: &[Vec<u32>],
    num_machines: usize,
    iterations: usize,
    rng: &mut SmallRng,
) -> Result<Vec<usize>> {
    if order.len() <= 2 || iterations == 0 {
        return Ok(order.to_vec());
    }
    let mut best_order = order.to_vec();
    let mut best_makespan = compute_makespan_order(
        &best_order,
        job_products,
        job_ops_len,
        product_machine_seq,
        product_proc_times,
        num_machines,
    )?;

    for _ in 0..iterations {
        let len = best_order.len();
        let remove_idx = rng.gen_range(0..len);
        let job = best_order[remove_idx];
        let mut candidate = best_order.clone();
        candidate.remove(remove_idx);

        let mut best_pos = 0usize;
        let mut best_candidate_ms = u32::MAX;
        let step = if len > 60 { 2 } else { 1 };
        let mut pos = 0usize;
        while pos <= candidate.len() {
            let mut trial = candidate.clone();
            trial.insert(pos, job);
            let ms = compute_makespan_order(
                &trial,
                job_products,
                job_ops_len,
                product_machine_seq,
                product_proc_times,
                num_machines,
            )?;
            if ms < best_candidate_ms {
                best_candidate_ms = ms;
                best_pos = pos;
            }
            pos += step;
        }

        candidate.insert(best_pos, job);
        if best_candidate_ms < best_makespan {
            best_makespan = best_candidate_ms;
            best_order = candidate;
        }
    }

    Ok(best_order)
}

fn compute_bottleneck_remaining(
    job_products: &[usize],
    product_ops: &[Vec<OpInfo>],
    num_machines: usize,
) -> (usize, Vec<f64>) {
    let num_jobs = job_products.len();
    let mut machine_load = vec![0u64; num_machines.max(1)];

    for &product in job_products.iter() {
        for op in product_ops[product].iter() {
            machine_load[op.machine] += op.duration as u64;
        }
    }

    let (bottleneck, _) = machine_load
        .iter()
        .enumerate()
        .max_by_key(|&(_, load)| load)
        .unwrap_or((0usize, &0u64));

    let mut job_bottleneck = vec![0.0; num_jobs];
    for job in 0..num_jobs {
        let product = job_products[job];
        let mut sum = 0.0;
        for op in product_ops[product].iter() {
            if op.machine == bottleneck {
                sum += op.duration as f64;
            }
        }
        job_bottleneck[job] = sum;
    }

    (bottleneck, job_bottleneck)
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

    let effort = hyperparameters.effort;
    let job_products = build_job_products(challenge)?;
    let product_ops = build_op_info(challenge)?;
    let product_work_times = build_product_work_times(&product_ops);
    let (job_ops_len, job_total_work) = build_job_stats(&job_products, &product_work_times);

    let total_ops = job_ops_len.iter().sum::<usize>();
    if total_ops == 0 {
        let job_schedule = vec![Vec::new(); challenge.num_jobs];
        save_solution(&Solution { job_schedule })?;
        return Ok(());
    }

    let (product_machine_seq, product_proc_times) = extract_strict_sequences(&product_ops);

    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;
    let scaled_effort = effort.max(1);
    let neh_limit = hyperparameters
        .neh_limit
        .unwrap_or(6 + scaled_effort * 2)
        .min(num_jobs);
    let local_iters = hyperparameters
        .local_iters
        .unwrap_or(4 + scaled_effort * 2)
        .max(1);
    let cds_limit = hyperparameters
        .cds_limit
        .unwrap_or(scaled_effort)
        .min(6);
    let random_restarts = hyperparameters
        .random_restarts
        .unwrap_or_else(|| if effort == 0 { 10 } else { 200 + 50 * scaled_effort });
    let random_top_k = hyperparameters
        .random_top_k
        .unwrap_or(2 + scaled_effort / 2)
        .max(3)
        .min(12);
    let weight_jitter = hyperparameters
        .weight_jitter
        .unwrap_or(0.1 + 0.05 * scaled_effort as f64)
        .clamp(0.0, 0.8);

    let mut best_makespan = u32::MAX;
    let mut best_schedule: Option<Vec<Vec<(usize, u32)>>> = None;
    let save_best =
        |best_makespan: &mut u32,
         best_schedule: &mut Option<Vec<Vec<(usize, u32)>>>,
         schedule: Vec<Vec<(usize, u32)>>,
         makespan: u32|
         -> Result<()> {
            if makespan < *best_makespan {
                *best_makespan = makespan;
                *best_schedule = Some(schedule.clone());
                save_solution(&Solution { job_schedule: schedule })?;
            }
            Ok(())
        };

    // Order-based heuristics (flow-shop style)
    let mut orders: Vec<Vec<usize>> = Vec::new();
    orders.push(order_by_total_work(&job_total_work, true));
    orders.push(order_by_total_work(&job_total_work, false));
    orders.push(order_by_ops(&job_ops_len, true));

    let uniform_ops = product_proc_times
        .iter()
        .map(|ops| ops.len())
        .all(|len| len == product_proc_times[0].len());

    if uniform_ops {
        orders.push(palmer_order(&job_products, &product_proc_times));
        orders.push(gupta_order(&job_products, &product_proc_times));
        let cds = cds_orders(&job_products, &product_proc_times, cds_limit);
        orders.extend(cds);
    }

    orders.push(bottleneck_order(&job_products, &product_ops));

    let base_order = order_by_total_work(&job_total_work, true);
    let neh_order = neh_insertion(
        &base_order,
        &job_products,
        &job_ops_len,
        &product_machine_seq,
        &product_proc_times,
        num_machines,
        neh_limit,
    )?;
    orders.push(neh_order.clone());

    // Evaluate top orders using strict scheduler
    let mut order_scores: Vec<(usize, u32)> = Vec::new();
    for (idx, order) in orders.iter().enumerate() {
        let makespan = compute_makespan_order(
            order,
            &job_products,
            &job_ops_len,
            &product_machine_seq,
            &product_proc_times,
            num_machines,
        )?;
        order_scores.push((idx, makespan));
    }
    order_scores.sort_unstable_by(|a, b| a.1.cmp(&b.1));
    let keep_orders = (2 + scaled_effort).min(order_scores.len()).max(1);

    for &(idx, order_makespan) in order_scores.iter().take(keep_orders) {
        if order_makespan >= best_makespan {
            continue;
        }
        let order = &orders[idx];
        let result = schedule_strict_order(
            order,
            &job_products,
            &job_ops_len,
            &product_machine_seq,
            &product_proc_times,
            num_machines,
        )?;
        save_best(
            &mut best_makespan,
            &mut best_schedule,
            result.job_schedule.clone(),
            result.makespan,
        )?;
    }

    // Local improvement on best NEH order
    let mut rng = SmallRng::from_seed(challenge.seed);
    let improved_order = insertion_local_search(
        &neh_order,
        &job_products,
        &job_ops_len,
        &product_machine_seq,
        &product_proc_times,
        num_machines,
        local_iters,
        &mut rng,
    )?;
    let improved_makespan = compute_makespan_order(
        &improved_order,
        &job_products,
        &job_ops_len,
        &product_machine_seq,
        &product_proc_times,
        num_machines,
    )?;
    if improved_makespan < best_makespan {
        let improved_result = schedule_strict_order(
            &improved_order,
            &job_products,
            &job_ops_len,
            &product_machine_seq,
            &product_proc_times,
            num_machines,
        )?;
        save_best(
            &mut best_makespan,
            &mut best_schedule,
            improved_result.job_schedule.clone(),
            improved_result.makespan,
        )?;
    }

    // Dispatching rules
    let scales = priority_scales(challenge, &job_ops_len, &job_total_work);
    let base_weights = RuleWeights::default();
    let (bottleneck_machine, job_bottleneck_work) =
        compute_bottleneck_remaining(&job_products, &product_ops, challenge.num_machines);

    let rules = [
        DispatchRule::MostWorkRemaining,
        DispatchRule::MostOpsRemaining,
        DispatchRule::ShortestProcTime,
        DispatchRule::LongestProcTime,
        DispatchRule::BottleneckRemaining,
        DispatchRule::Weighted,
    ];

    for rule in rules.iter().copied() {
        let params = DispatchParams {
            rule,
            weights: base_weights,
            order_rank: None,
            order_weight: 0.0,
            order_max: (num_jobs.saturating_sub(1)) as f64,
        };
        let result = run_dispatch_rule(
            challenge,
            &job_products,
            &product_ops,
            &product_work_times,
            &job_ops_len,
            &job_total_work,
            Some(&job_bottleneck_work),
            Some(bottleneck_machine),
            params,
            scales,
            None,
            false,
            None,
        )?;
        save_best(
            &mut best_makespan,
            &mut best_schedule,
            result.job_schedule.clone(),
            result.makespan,
        )?;
    }

    // Order-biased dispatching for top orders
    for &(idx, _) in order_scores.iter().take(keep_orders) {
        let order = &orders[idx];
        let rank = build_order_rank(order, num_jobs);
        let params = DispatchParams {
            rule: DispatchRule::OrderOnly,
            weights: base_weights,
            order_rank: Some(&rank),
            order_weight: 1.0,
            order_max: (num_jobs.saturating_sub(1)) as f64,
        };
        let result = run_dispatch_rule(
            challenge,
            &job_products,
            &product_ops,
            &product_work_times,
            &job_ops_len,
            &job_total_work,
            Some(&job_bottleneck_work),
            Some(bottleneck_machine),
            params,
            scales,
            None,
            false,
            None,
        )?;
        save_best(
            &mut best_makespan,
            &mut best_schedule,
            result.job_schedule.clone(),
            result.makespan,
        )?;
    }

    if random_restarts > 0 {
        let mut top_restarts: Vec<RestartResult> = Vec::new();
        let top_k_count = if effort == 0 { 0 } else { 2 + scaled_effort };
        let local_search_tries = 1 + 3 * scaled_effort.min(4);

        for _ in 0..random_restarts {
            let seed = rng.gen::<u64>();
            let mut local_rng = SmallRng::seed_from_u64(seed);
            let rule = rules[local_rng.gen_range(0..rules.len())];
            let restart_top_k = local_rng.gen_range(2..=random_top_k.max(3));
            let weights = if matches!(rule, DispatchRule::Weighted) {
                let mut jittered = base_weights;
                let mut jitter = |value: f64| {
                    let factor = local_rng.gen_range(1.0 - weight_jitter..=1.0 + weight_jitter);
                    (value * factor).max(0.0)
                };
                jittered.work = jitter(jittered.work);
                jittered.ops = jitter(jittered.ops);
                jittered.end = jitter(jittered.end);
                jittered.proc = jitter(jittered.proc);
                jittered
            } else {
                base_weights
            };

            let params = DispatchParams {
                rule,
                weights,
                order_rank: None,
                order_weight: 0.0,
                order_max: (num_jobs.saturating_sub(1)) as f64,
            };
            let result = run_dispatch_rule(
                challenge,
                &job_products,
                &product_ops,
                &product_work_times,
                &job_ops_len,
                &job_total_work,
                Some(&job_bottleneck_work),
                Some(bottleneck_machine),
                params,
                scales,
                Some(restart_top_k),
                true,
                Some(&mut local_rng),
            )?;
            let makespan = result.makespan;
            save_best(
                &mut best_makespan,
                &mut best_schedule,
                result.job_schedule,
                makespan,
            )?;

            if top_k_count > 0 {
                top_restarts.push(RestartResult {
                    makespan,
                    rule,
                    random_top_k: restart_top_k,
                    seed,
                });
                top_restarts.sort_unstable_by_key(|r| r.makespan);
                if top_restarts.len() > top_k_count {
                    top_restarts.pop();
                }
            }
        }

        // Local search around top restarts: vary seed and top_k
        for restart in top_restarts.iter() {
            for attempt in 0..local_search_tries {
                let local_seed = restart.seed.wrapping_add(attempt as u64 + 1);
                let mut local_rng = SmallRng::seed_from_u64(local_seed);
                let local_k = match attempt % 3 {
                    0 => restart.random_top_k,
                    1 => restart.random_top_k.saturating_sub(1).max(2),
                    _ => restart.random_top_k.saturating_add(1).min(8),
                };
                let weights = if matches!(restart.rule, DispatchRule::Weighted) {
                    let mut jittered = base_weights;
                    let mut jitter = |value: f64| {
                        let factor =
                            local_rng.gen_range(1.0 - weight_jitter..=1.0 + weight_jitter);
                        (value * factor).max(0.0)
                    };
                    jittered.work = jitter(jittered.work);
                    jittered.ops = jitter(jittered.ops);
                    jittered.end = jitter(jittered.end);
                    jittered.proc = jitter(jittered.proc);
                    jittered
                } else {
                    base_weights
                };

                let params = DispatchParams {
                    rule: restart.rule,
                    weights,
                    order_rank: None,
                    order_weight: 0.0,
                    order_max: (num_jobs.saturating_sub(1)) as f64,
                };
                let result = run_dispatch_rule(
                    challenge,
                    &job_products,
                    &product_ops,
                    &product_work_times,
                    &job_ops_len,
                    &job_total_work,
                    Some(&job_bottleneck_work),
                    Some(bottleneck_machine),
                    params,
                    scales,
                    Some(local_k),
                    true,
                    Some(&mut local_rng),
                )?;
                save_best(
                    &mut best_makespan,
                    &mut best_schedule,
                    result.job_schedule,
                    result.makespan,
                )?;
            }
        }
    }

    if best_schedule.is_none() {
        return Err(anyhow!("No valid schedule produced"));
    }

    Ok(())
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
}

#[allow(dead_code)]
mod parallel {
// PARALLEL-specialized variant of d_r with increased compute budget.
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
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self { effort: 5 }
    }
}

pub fn help() {
    println!("PARALLEL-specialized dispatching-rules solver with weighted search and soft machine selection.");
    println!("Hyperparameters (all optional):");
    println!("  effort: baseline effort (default 5).");
}

const WORK_MIN_WEIGHT: f64 = 0.3;
const MAX_SLACK_RATIO: f64 = 0.5;

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

// Pre-computed operation data (avoids HashMap lookups in hot loop)
struct OpData {
    // proc_time[product][op][machine] = processing time, u32::MAX if ineligible
    proc_time: Vec<Vec<Vec<u32>>>,
    // eligible[product][op] = sorted Vec of (machine_id, proc_time)
    eligible: Vec<Vec<Vec<(usize, u32)>>>,
    // flex[product][op] = number of eligible machines
    flex: Vec<Vec<usize>>,
}

fn precompute_ops(challenge: &Challenge) -> OpData {
    let num_machines = challenge.num_machines;
    let mut proc_time_data = Vec::with_capacity(challenge.product_processing_times.len());
    let mut eligible_data = Vec::with_capacity(challenge.product_processing_times.len());
    let mut flex_data = Vec::with_capacity(challenge.product_processing_times.len());

    for product_ops in challenge.product_processing_times.iter() {
        let mut p_proc = Vec::with_capacity(product_ops.len());
        let mut p_elig = Vec::with_capacity(product_ops.len());
        let mut p_flex = Vec::with_capacity(product_ops.len());

        for op in product_ops.iter() {
            let mut proc_vec = vec![u32::MAX; num_machines];
            let mut elig_vec = Vec::with_capacity(op.len());
            for (&machine, &pt) in op.iter() {
                proc_vec[machine] = pt;
                elig_vec.push((machine, pt));
            }
            elig_vec.sort_unstable_by_key(|&(m, _)| m);
            p_flex.push(elig_vec.len());
            p_proc.push(proc_vec);
            p_elig.push(elig_vec);
        }

        proc_time_data.push(p_proc);
        eligible_data.push(p_elig);
        flex_data.push(p_flex);
    }

    OpData {
        proc_time: proc_time_data,
        eligible: eligible_data,
        flex: flex_data,
    }
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

#[inline]
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

#[inline]
fn slack_allowance(proc_time: u32, slack_ratio: f64) -> u32 {
    if slack_ratio <= 0.0 {
        return 0;
    }
    let slack_ratio = slack_ratio.max(0.0).min(MAX_SLACK_RATIO);
    ((proc_time as f64) * slack_ratio).round() as u32
}

#[inline]
fn earliest_end_fast(time: u32, machine_avail: &[u32], eligible: &[(usize, u32)]) -> u32 {
    let mut earliest = u32::MAX;
    for &(machine, pt) in eligible.iter() {
        let end = time.max(machine_avail[machine]) + pt;
        if end < earliest {
            earliest = end;
        }
    }
    earliest
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

#[inline]
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
    num_jobs: usize,
    num_machines: usize,
    ops: &OpData,
    job_products: &[usize],
    product_work_times: &[Vec<f64>],
    job_ops_len: &[usize],
    job_total_work: &[f64],
    params: DispatchParams,
    scales: PriorityScales,
    random_top_k: Option<usize>,
    rng: Option<&mut SmallRng>,
) -> Result<ScheduleResult> {
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

    // Pre-allocate reusable buffers
    let mut available_machines: Vec<usize> = Vec::with_capacity(num_machines);
    let mut candidates: Vec<Candidate> = Vec::with_capacity(num_jobs);

    while remaining_ops > 0 {
        let mut scheduled_any = false;

        if use_random {
            // Random path: collect + shuffle machines
            available_machines.clear();
            for m in 0..num_machines {
                if machine_available_time[m] <= time {
                    available_machines.push(m);
                }
            }
            available_machines.shuffle(rng.as_mut().unwrap());

            for &machine in available_machines.iter() {
                candidates.clear();

                for job in 0..num_jobs {
                    if job_next_op_idx[job] >= job_ops_len[job] || job_ready_time[job] > time {
                        continue;
                    }

                    let product = job_products[job];
                    let op_idx = job_next_op_idx[job];
                    let proc_time = ops.proc_time[product][op_idx][machine];
                    if proc_time == u32::MAX {
                        continue;
                    }

                    let earliest_end = earliest_end_fast(
                        time,
                        &machine_available_time,
                        &ops.eligible[product][op_idx],
                    );
                    let machine_end = time.max(machine_available_time[machine]) + proc_time;
                    let slack = slack_allowance(proc_time, params.slack_ratio);
                    if machine_end > earliest_end.saturating_add(slack) {
                        continue;
                    }

                    let flexibility = ops.flex[product][op_idx];
                    let remaining_ops_job = job_ops_len[job] - job_next_op_idx[job];
                    let priority = match params.rule {
                        DispatchRule::MostWorkRemaining => job_remaining_work[job],
                        DispatchRule::MostOpsRemaining => remaining_ops_job as f64,
                        DispatchRule::LeastFlexibility => -(flexibility as f64),
                        DispatchRule::ShortestProcTime => -(proc_time as f64),
                        DispatchRule::LongestProcTime => proc_time as f64,
                        DispatchRule::Weighted => weighted_priority(
                            job_remaining_work[job],
                            remaining_ops_job,
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
                    candidates.sort_unstable_by(|a, b| {
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
                    let candidate = candidates[pick];

                    let job = candidate.job;
                    let product = job_products[job];
                    let op_idx = job_next_op_idx[job];
                    let start_time = time.max(machine_available_time[machine]);
                    let end_time = start_time + candidate.proc_time;

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
        } else {
            // Deterministic path: iterate machines directly (no Vec, no sort)
            for machine in 0..num_machines {
                if machine_available_time[machine] > time {
                    continue;
                }

                let mut best_candidate: Option<Candidate> = None;

                for job in 0..num_jobs {
                    if job_next_op_idx[job] >= job_ops_len[job] || job_ready_time[job] > time {
                        continue;
                    }

                    let product = job_products[job];
                    let op_idx = job_next_op_idx[job];
                    let proc_time = ops.proc_time[product][op_idx][machine];
                    if proc_time == u32::MAX {
                        continue;
                    }

                    let earliest_end = earliest_end_fast(
                        time,
                        &machine_available_time,
                        &ops.eligible[product][op_idx],
                    );
                    let machine_end = time.max(machine_available_time[machine]) + proc_time;
                    let slack = slack_allowance(proc_time, params.slack_ratio);
                    if machine_end > earliest_end.saturating_add(slack) {
                        continue;
                    }

                    let flexibility = ops.flex[product][op_idx];
                    let remaining_ops_job = job_ops_len[job] - job_next_op_idx[job];
                    let priority = match params.rule {
                        DispatchRule::MostWorkRemaining => job_remaining_work[job],
                        DispatchRule::MostOpsRemaining => remaining_ops_job as f64,
                        DispatchRule::LeastFlexibility => -(flexibility as f64),
                        DispatchRule::ShortestProcTime => -(proc_time as f64),
                        DispatchRule::LongestProcTime => proc_time as f64,
                        DispatchRule::Weighted => weighted_priority(
                            job_remaining_work[job],
                            remaining_ops_job,
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

                if let Some(candidate) = best_candidate {
                    let job = candidate.job;
                    let product = job_products[job];
                    let op_idx = job_next_op_idx[job];
                    let start_time = time.max(machine_available_time[machine]);
                    let end_time = start_time + candidate.proc_time;

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

fn run_baseline_search(
    num_jobs: usize,
    num_machines: usize,
    ops: &OpData,
    save_best: &dyn Fn(&ScheduleResult) -> Result<()>,
    job_products: &[usize],
    product_work_times: &[Vec<f64>],
    job_ops_len: &[usize],
    job_total_work: &[f64],
    random_restarts: usize,
    top_k: usize,
    local_search_tries: usize,
    seed: [u8; 32],
) -> Result<ScheduleResult> {
    let scales = priority_scales_fast(job_ops_len, job_total_work, ops);
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
            num_jobs,
            num_machines,
            ops,
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
        let mut rng = SmallRng::from_seed(seed);
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
                num_jobs,
                num_machines,
                ops,
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
            if makespan < best_result.makespan {
                best_result = result;
                save_best(&best_result)?;
            }

            if top_k > 0 {
                if top_restarts.len() < top_k
                    || makespan < top_restarts.last().unwrap().makespan
                {
                    top_restarts.push(RestartResult {
                        makespan,
                        rule,
                        random_top_k,
                        seed,
                        weights: base_weights,
                        slack_ratio: 0.0,
                    });
                    top_restarts.sort_unstable_by(|a, b| a.makespan.cmp(&b.makespan));
                    if top_restarts.len() > top_k {
                        top_restarts.pop();
                    }
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
                    num_jobs,
                    num_machines,
                    ops,
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

// Fast priority_scales using pre-computed OpData
fn priority_scales_fast(
    job_ops_len: &[usize],
    job_total_work: &[f64],
    ops: &OpData,
) -> PriorityScales {
    let work = job_total_work
        .iter()
        .copied()
        .fold(0.0, f64::max)
        .max(1.0);
    let ops_scale = job_ops_len.iter().copied().max().unwrap_or(1) as f64;
    let mut max_proc = 1u32;
    for product_eligible in ops.eligible.iter() {
        for op_eligible in product_eligible.iter() {
            for &(_, pt) in op_eligible.iter() {
                if pt > max_proc {
                    max_proc = pt;
                }
            }
        }
    }
    let proc = max_proc as f64;
    let end = (work + proc).max(1.0);
    PriorityScales {
        work,
        ops: ops_scale,
        proc,
        end,
    }
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
    let (baseline_random_restarts, baseline_top_k) = if effort == 0 {
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
    let baseline_local_search_tries = 1usize.saturating_add(3usize.saturating_mul(effort));

    let tuned_random_restarts = 350usize.saturating_add(70usize.saturating_mul(effort.max(1)));
    let tuned_top_k = 3usize.saturating_add(3usize.saturating_mul(effort.max(1)));
    let tuned_local_search_tries = 2usize.saturating_add(4usize.saturating_mul(effort));

    solve_challenge_with_params(
        challenge,
        save_solution,
        baseline_random_restarts,
        baseline_top_k,
        baseline_local_search_tries,
        tuned_random_restarts,
        tuned_top_k,
        tuned_local_search_tries,
        effort,
    )
}

fn solve_challenge_with_params(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    baseline_random_restarts: usize,
    baseline_top_k: usize,
    baseline_local_search_tries: usize,
    tuned_random_restarts: usize,
    tuned_top_k: usize,
    tuned_local_search_tries: usize,
    effort: usize,
) -> Result<()> {
    let save_best = |best: &ScheduleResult| -> Result<()> {
        save_solution(&Solution {
            job_schedule: best.job_schedule.clone(),
        })
    };
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

    let mut job_ops_len = Vec::with_capacity(num_jobs);
    for &product in job_products.iter() {
        job_ops_len.push(challenge.product_processing_times[product].len());
    }

    // Pre-compute operation data once (avoids HashMap lookups in hot loop)
    let ops = precompute_ops(challenge);

    // Baseline pass (guarantees non-negative quality)
    let baseline_product_work_times =
        build_product_work_times(&challenge.product_processing_times, WORK_MIN_WEIGHT);
    let baseline_job_total_work =
        build_job_total_work(&job_products, &baseline_product_work_times);
    let baseline_result = run_baseline_search(
        num_jobs,
        num_machines,
        &ops,
        &save_best,
        &job_products,
        &baseline_product_work_times,
        &job_ops_len,
        &baseline_job_total_work,
        baseline_random_restarts,
        baseline_top_k,
        baseline_local_search_tries,
        challenge.seed,
    )?;

    // Tuned pass: always run with PARALLEL-hardcoded values
    let work_min_blend = 0.15_f64;
    let product_work_times =
        build_product_work_times(&challenge.product_processing_times, work_min_blend);
    let job_total_work = build_job_total_work(&job_products, &product_work_times);

    let scales = priority_scales_fast(&job_ops_len, &job_total_work, &ops);
    let base_rule_weights = RuleWeights {
        work: 1.0,
        ops: 0.2,
        flex: 0.8,
        end: 0.8,
        proc: 0.2,
    };
    let slack_ratio = (0.18 + 0.03 * effort as f64).min(0.5);
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
            num_jobs,
            num_machines,
            &ops,
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
        num_jobs,
        num_machines,
        &ops,
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

    if tuned_random_restarts > 0 {
        let mut rng = SmallRng::from_seed(challenge.seed);
        for _ in 1..=tuned_random_restarts {
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
            let base_top_k: usize = 3;
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
                num_jobs,
                num_machines,
                &ops,
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
            if makespan < best_result.makespan {
                best_result = result;
                save_best(&best_result)?;
            }

            if tuned_top_k > 0 {
                if top_restarts.len() < tuned_top_k
                    || makespan < top_restarts.last().unwrap().makespan
                {
                    top_restarts.push(RestartResult {
                        makespan,
                        rule,
                        random_top_k,
                        seed,
                        weights,
                        slack_ratio: restart_slack,
                    });
                    top_restarts.sort_unstable_by(|a, b| a.makespan.cmp(&b.makespan));
                    if top_restarts.len() > tuned_top_k {
                        top_restarts.pop();
                    }
                }
            }
        }
    }

    if !top_restarts.is_empty() {
        for restart in top_restarts.iter() {
            for attempt in 0..tuned_local_search_tries {
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
                    num_jobs,
                    num_machines,
                    &ops,
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

// Important! Do not include any tests in this file, it will result in your submission being rejected
}

#[allow(dead_code)]
mod chaotic {
// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::cmp::Ordering;
use tig_challenges::job_scheduling::*;

use super::super::heuristic;

#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct Hyperparameters {
    pub effort: usize,
    pub n_candidates: usize,
    pub k_candidates: usize,
    pub machine_top_k: usize,
    pub weight_jitter: f64,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            effort: 2,
            n_candidates: 320,
            k_candidates: 8,
            machine_top_k: 5,
            weight_jitter: 0.25,
        }
    }
}

pub fn help() {
    println!("CHAOTIC-optimized scheduler: load-balanced greedy dispatch + light tabu refinement.");
}

const MIN_TIME_BLEND: f64 = 0.35;

#[derive(Clone, Copy)]
struct ScoreWeights {
    work: f64,
    ops: f64,
    wait: f64,
    end: f64,
    proc: f64,
    load: f64,
    delta: f64,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            work: 1.1,
            ops: 0.35,
            wait: 0.15,
            end: 0.8,
            proc: 0.25,
            load: 0.55,
            delta: 0.4,
        }
    }
}

#[derive(Clone, Copy)]
struct Scales {
    work: f64,
    ops: f64,
    proc: f64,
    end: f64,
    load: f64,
    wait: f64,
}

struct OpData {
    eligible: Vec<(usize, u32)>,
    min_proc: u32,
    avg_proc: f64,
}

struct ScheduleResult {
    job_schedule: Vec<Vec<(usize, u32)>>,
    makespan: u32,
}

#[derive(Clone, Copy)]
struct Candidate {
    job: usize,
    machine: usize,
    score: f64,
    end: u32,
    proc: u32,
    delta: u32,
}

fn build_job_products(challenge: &Challenge) -> Result<Vec<usize>> {
    let mut job_products = Vec::with_capacity(challenge.num_jobs);
    for (product, count) in challenge.jobs_per_product.iter().enumerate() {
        for _ in 0..*count {
            job_products.push(product);
        }
    }
    if job_products.len() != challenge.num_jobs {
        return Err(anyhow!(
            "Job count mismatch. Expected {}, got {}",
            challenge.num_jobs,
            job_products.len()
        ));
    }
    Ok(job_products)
}

fn precompute_ops(challenge: &Challenge) -> Result<(Vec<Vec<OpData>>, u32)> {
    let mut product_ops = Vec::with_capacity(challenge.product_processing_times.len());
    let mut max_proc = 1u32;

    for (product_idx, ops) in challenge.product_processing_times.iter().enumerate() {
        let mut ops_data = Vec::with_capacity(ops.len());
        for (op_idx, op) in ops.iter().enumerate() {
            if op.is_empty() {
                return Err(anyhow!(
                    "Product {} op {} has no eligible machines",
                    product_idx,
                    op_idx
                ));
            }
            let mut eligible = Vec::with_capacity(op.len());
            let mut sum = 0u64;
            let mut min_proc = u32::MAX;
            for (&machine, &proc) in op.iter() {
                if machine >= challenge.num_machines {
                    return Err(anyhow!(
                        "Product {} op {} has invalid machine {}",
                        product_idx,
                        op_idx,
                        machine
                    ));
                }
                eligible.push((machine, proc));
                sum += proc as u64;
                if proc < min_proc {
                    min_proc = proc;
                }
                if proc > max_proc {
                    max_proc = proc;
                }
            }
            eligible.sort_by_key(|&(m, _)| m);
            let avg_proc = sum as f64 / eligible.len() as f64;
            ops_data.push(OpData {
                eligible,
                min_proc,
                avg_proc,
            });
        }
        product_ops.push(ops_data);
    }

    Ok((product_ops, max_proc))
}

fn build_product_work_times(product_ops: &[Vec<OpData>], blend: f64) -> Vec<Vec<f64>> {
    let blend = blend.max(0.0).min(1.0);
    let mut product_times = Vec::with_capacity(product_ops.len());
    for ops in product_ops.iter() {
        let mut times = Vec::with_capacity(ops.len());
        for op in ops.iter() {
            let min = op.min_proc as f64;
            let avg = op.avg_proc;
            times.push(min * blend + avg * (1.0 - blend));
        }
        product_times.push(times);
    }
    product_times
}

fn build_job_stats(
    job_products: &[usize],
    product_work_times: &[Vec<f64>],
) -> (Vec<usize>, Vec<f64>, usize, f64, usize, f64) {
    let mut job_ops_len = Vec::with_capacity(job_products.len());
    let mut job_total_work = Vec::with_capacity(job_products.len());
    let mut total_ops = 0usize;
    let mut total_work = 0.0f64;
    let mut max_work = 0.0f64;
    let mut max_ops = 0usize;

    for &product in job_products.iter() {
        let ops = &product_work_times[product];
        let ops_len = ops.len();
        let work: f64 = ops.iter().sum();
        job_ops_len.push(ops_len);
        job_total_work.push(work);
        total_ops = total_ops.saturating_add(ops_len);
        total_work += work;
        if work > max_work {
            max_work = work;
        }
        if ops_len > max_ops {
            max_ops = ops_len;
        }
    }

    (job_ops_len, job_total_work, total_ops, max_work, max_ops, total_work)
}

fn compute_scales(
    max_work: f64,
    max_ops: usize,
    max_proc: u32,
    total_work: f64,
    total_ops: usize,
    num_machines: usize,
) -> Scales {
    let work = max_work.max(1.0);
    let ops = (max_ops.max(1)) as f64;
    let proc = (max_proc.max(1)) as f64;
    let end = (max_work + proc).max(1.0);
    let load = if num_machines == 0 {
        1.0
    } else {
        (total_work / num_machines as f64).max(1.0)
    };
    let wait = if total_ops == 0 {
        1.0
    } else {
        (total_work / total_ops as f64).max(1.0)
    };
    Scales {
        work,
        ops,
        proc,
        end,
        load,
        wait,
    }
}

fn jitter_weights(base: ScoreWeights, rng: &mut SmallRng, jitter: f64) -> ScoreWeights {
    let jitter = jitter.max(0.0).min(0.7);
    let mut scale = |value: f64| {
        if jitter == 0.0 {
            return value;
        }
        let factor = rng.gen_range(1.0 - jitter..=1.0 + jitter);
        (value * factor).max(0.0)
    };
    ScoreWeights {
        work: scale(base.work),
        ops: scale(base.ops),
        wait: scale(base.wait),
        end: scale(base.end),
        proc: scale(base.proc),
        load: scale(base.load),
        delta: scale(base.delta),
    }
}

fn insert_top_k(list: &mut Vec<(usize, u32)>, item: (usize, u32), k: usize) {
    if k == 0 {
        return;
    }
    let mut insert_at: Option<usize> = None;
    for (idx, &(m, proc)) in list.iter().enumerate() {
        if item.1 < proc || (item.1 == proc && item.0 < m) {
            insert_at = Some(idx);
            break;
        }
    }
    if let Some(idx) = insert_at {
        list.insert(idx, item);
    } else {
        list.push(item);
    }
    if list.len() > k {
        list.pop();
    }
}

fn candidate_cmp(a: &Candidate, b: &Candidate) -> Ordering {
    let eps = 1e-9_f64;
    if (b.score - a.score).abs() > eps {
        return b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal);
    }
    if a.end != b.end {
        return a.end.cmp(&b.end);
    }
    if a.proc != b.proc {
        return a.proc.cmp(&b.proc);
    }
    if a.delta != b.delta {
        return a.delta.cmp(&b.delta);
    }
    if a.job != b.job {
        return a.job.cmp(&b.job);
    }
    a.machine.cmp(&b.machine)
}

fn construct_schedule(
    challenge: &Challenge,
    job_products: &[usize],
    product_ops: &[Vec<OpData>],
    product_work_times: &[Vec<f64>],
    job_ops_len: &[usize],
    job_total_work: &[f64],
    scales: Scales,
    weights: ScoreWeights,
    machine_top_k: usize,
    score_jitter: f64,
    mut rng: Option<&mut SmallRng>,
) -> Result<ScheduleResult> {
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines.max(1);
    let machine_top_k = machine_top_k.max(1).min(num_machines);

    let mut job_next_op_idx = vec![0usize; num_jobs];
    let mut job_ready_time = vec![0u32; num_jobs];
    let mut machine_available_time = vec![0u32; num_machines];
    let mut machine_load = vec![0u32; num_machines];
    let mut job_schedule = job_ops_len
        .iter()
        .map(|&ops_len| Vec::with_capacity(ops_len))
        .collect::<Vec<_>>();
    let mut job_remaining_work = job_total_work.to_vec();

    let mut remaining_ops = job_ops_len.iter().sum::<usize>();
    let mut time = 0u32;

    let mut ready_jobs: Vec<usize> = Vec::with_capacity(num_jobs);
    let mut candidates: Vec<Candidate> = Vec::with_capacity(num_jobs * machine_top_k);
    let mut top_machines: Vec<(usize, u32)> = Vec::with_capacity(machine_top_k);
    let mut assigned_jobs = vec![0u32; num_jobs];
    let mut assigned_machines = vec![0u32; num_machines];
    let mut stamp: u32 = 1;
    let use_jitter = score_jitter > 0.0 && rng.is_some();

    while remaining_ops > 0 {
        ready_jobs.clear();
        candidates.clear();

        let mut available_count = 0usize;
        let mut next_machine_time: Option<u32> = None;
        for machine in 0..num_machines {
            let t = machine_available_time[machine];
            if t <= time {
                available_count += 1;
            } else {
                next_machine_time = Some(next_machine_time.map_or(t, |best| best.min(t)));
            }
        }

        let mut next_job_time: Option<u32> = None;
        for job in 0..num_jobs {
            if job_next_op_idx[job] >= job_ops_len[job] {
                continue;
            }
            let t = job_ready_time[job];
            if t <= time {
                ready_jobs.push(job);
            } else {
                next_job_time = Some(next_job_time.map_or(t, |best| best.min(t)));
            }
        }

        if available_count == 0 || ready_jobs.is_empty() {
            let next_time = match (next_machine_time, next_job_time) {
                (Some(machine_time), Some(job_time)) => Some(machine_time.min(job_time)),
                (Some(machine_time), None) => Some(machine_time),
                (None, Some(job_time)) => Some(job_time),
                (None, None) => None,
            };
            time = next_time.ok_or_else(|| {
                anyhow!("No schedulable operations remain; dispatching stalled")
            })?;
            continue;
        }

        for &job in ready_jobs.iter() {
            let product = job_products[job];
            let op_idx = job_next_op_idx[job];
            let op_data = &product_ops[product][op_idx];

            top_machines.clear();
            let mut best_end = u32::MAX;
            for &(machine, proc) in op_data.eligible.iter() {
                let start = time.max(machine_available_time[machine]);
                let end = start + proc;
                if end < best_end {
                    best_end = end;
                }
                if start == time {
                    insert_top_k(&mut top_machines, (machine, proc), machine_top_k);
                }
            }
            if top_machines.is_empty() {
                continue;
            }

            let remaining_ops = job_ops_len[job] - job_next_op_idx[job];
            let remaining_work = job_remaining_work[job];
            let wait = time.saturating_sub(job_ready_time[job]);

            let base_score = weights.work * (remaining_work / scales.work)
                + weights.ops * (remaining_ops as f64 / scales.ops)
                + weights.wait * (wait as f64 / scales.wait);

            for &(machine, proc) in top_machines.iter() {
                let end = time + proc;
                let delta = end.saturating_sub(best_end);
                let mut score = base_score
                    - weights.end * (end as f64 / scales.end)
                    - weights.proc * (proc as f64 / scales.proc)
                    - weights.load * (machine_load[machine] as f64 / scales.load)
                    - weights.delta * (delta as f64 / scales.proc);
                if use_jitter {
                    let jitter = rng
                        .as_deref_mut()
                        .unwrap()
                        .gen_range(-score_jitter..=score_jitter);
                    score += jitter;
                }
                candidates.push(Candidate {
                    job,
                    machine,
                    score,
                    end,
                    proc,
                    delta,
                });
            }
        }

        if candidates.is_empty() {
            let next_time = match (next_machine_time, next_job_time) {
                (Some(machine_time), Some(job_time)) => Some(machine_time.min(job_time)),
                (Some(machine_time), None) => Some(machine_time),
                (None, Some(job_time)) => Some(job_time),
                (None, None) => None,
            };
            time = next_time.ok_or_else(|| {
                anyhow!("No schedulable operations remain; dispatching stalled")
            })?;
            continue;
        }

        candidates.sort_unstable_by(candidate_cmp);
        let current_stamp = stamp;
        stamp = stamp.wrapping_add(1);
        if stamp == 0 {
            assigned_jobs.fill(0);
            assigned_machines.fill(0);
            stamp = 1;
        }

        let mut scheduled_any = false;
        let mut assigned_count = 0usize;
        let target = available_count;

        for cand in candidates.iter() {
            if assigned_jobs[cand.job] == current_stamp
                || assigned_machines[cand.machine] == current_stamp
            {
                continue;
            }
            let job = cand.job;
            let machine = cand.machine;
            let proc = cand.proc;

            let start_time = time;
            let end_time = time + proc;

            job_schedule[job].push((machine, start_time));
            job_next_op_idx[job] += 1;
            job_ready_time[job] = end_time;
            machine_available_time[machine] = end_time;
            machine_load[machine] = machine_load[machine].saturating_add(proc);

            let product = job_products[job];
            let op_idx = job_next_op_idx[job] - 1;
            job_remaining_work[job] -= product_work_times[product][op_idx];
            if job_remaining_work[job] < 0.0 {
                job_remaining_work[job] = 0.0;
            }

            remaining_ops -= 1;
            assigned_jobs[job] = current_stamp;
            assigned_machines[machine] = current_stamp;
            assigned_count += 1;
            scheduled_any = true;

            if assigned_count >= target {
                break;
            }
        }

        if !scheduled_any {
            let next_time = match (next_machine_time, next_job_time) {
                (Some(machine_time), Some(job_time)) => Some(machine_time.min(job_time)),
                (Some(machine_time), None) => Some(machine_time),
                (None, Some(job_time)) => Some(job_time),
                (None, None) => None,
            };
            time = next_time.ok_or_else(|| {
                anyhow!("No schedulable operations remain; dispatching stalled")
            })?;
        }
    }

    let makespan = job_ready_time.iter().copied().max().unwrap_or(0);
    Ok(ScheduleResult {
        job_schedule,
        makespan,
    })
}

fn estimate_stats(challenge: &Challenge) -> (usize, usize) {
    let mut total_ops = 0usize;
    let mut total_options = 0usize;

    for (product_idx, &count) in challenge.jobs_per_product.iter().enumerate() {
        let ops = &challenge.product_processing_times[product_idx];
        total_ops = total_ops.saturating_add(count.saturating_mul(ops.len()));
        for op in ops.iter() {
            total_options = total_options.saturating_add(count.saturating_mul(op.len()));
        }
    }

    let avg_flex = if total_ops == 0 {
        1usize
    } else {
        (total_options + total_ops - 1) / total_ops
    };

    (total_ops, avg_flex)
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let hyperparameters = match hyperparameters {
        Some(hyperparameters) => serde_json::from_value::<Hyperparameters>(Value::Object(
            hyperparameters.clone(),
        ))
        .map_err(|e| anyhow!("Failed to parse hyperparameters: {}", e))?,
        None => Hyperparameters::default(),
    };

    let effort = hyperparameters.effort.max(1);
    let n_candidates = hyperparameters.n_candidates;
    let k_candidates = hyperparameters.k_candidates.max(1);
    let mut machine_top_k = hyperparameters.machine_top_k.max(1);
    let weight_jitter = hyperparameters.weight_jitter;

    let job_products = build_job_products(challenge)?;
    let (product_ops, max_proc) = precompute_ops(challenge)?;
    let product_work_times = build_product_work_times(&product_ops, MIN_TIME_BLEND);

    let (job_ops_len, job_total_work, total_ops, max_work, max_ops, total_work) =
        build_job_stats(&job_products, &product_work_times);

    if total_ops == 0 {
        let job_schedule = vec![Vec::new(); challenge.num_jobs];
        save_solution(&Solution { job_schedule })?;
        return Ok(());
    }

    let scales = compute_scales(
        max_work,
        max_ops,
        max_proc,
        total_work,
        total_ops,
        challenge.num_machines,
    );

    if challenge.num_machines > 0 {
        machine_top_k = machine_top_k.min(challenge.num_machines);
    }

    let presets = [
        ScoreWeights::default(),
        ScoreWeights {
            work: 1.5,
            ops: 0.5,
            wait: 0.2,
            end: 0.6,
            proc: 0.15,
            load: 0.35,
            delta: 0.3,
        },
        ScoreWeights {
            work: 0.8,
            ops: 0.2,
            wait: 0.1,
            end: 1.0,
            proc: 0.5,
            load: 0.7,
            delta: 0.6,
        },
    ];

    let mut candidates: Vec<ScheduleResult> = Vec::new();
    for preset in presets.iter().copied() {
        let result = construct_schedule(
            challenge,
            &job_products,
            &product_ops,
            &product_work_times,
            &job_ops_len,
            &job_total_work,
            scales,
            preset,
            machine_top_k,
            0.0,
            None,
        )?;
        candidates.push(result);
    }

    let mut rng = SmallRng::from_seed(challenge.seed);
    let base_top_k = machine_top_k;
    for i in 0..n_candidates {
        let base = presets[i % presets.len()];
        let weights = jitter_weights(base, &mut rng, weight_jitter);
        let spread = (base_top_k / 2).max(1);
        let top_k = if base_top_k > 1 {
            rng.gen_range(base_top_k.saturating_sub(spread)..=base_top_k + spread)
        } else {
            1
        };
        let result = construct_schedule(
            challenge,
            &job_products,
            &product_ops,
            &product_work_times,
            &job_ops_len,
            &job_total_work,
            scales,
            weights,
            top_k,
            0.0001,
            Some(&mut rng),
        )?;
        candidates.push(result);
    }

    if candidates.is_empty() {
        return Err(anyhow!("No valid schedule produced"));
    }

    candidates.sort_by_key(|candidate| candidate.makespan);
    let keep = k_candidates.min(candidates.len());
    let selected_indices: Vec<usize> = (0..keep).collect();

    let (total_ops, avg_flex) = estimate_stats(challenge);
    let base_max_iters: usize = if total_ops <= 100 {
        220
    } else if total_ops <= 300 {
        140
    } else {
        80
    };
    let base_max_neighbors: usize = if total_ops <= 100 {
        45
    } else if total_ops <= 300 {
        30
    } else {
        20
    };

    let max_iters = base_max_iters.saturating_mul(effort).max(1);
    let max_neighbors = base_max_neighbors.saturating_mul(effort).max(1);
    let max_reassign = if avg_flex > 1 {
        (max_neighbors / 2).max(1)
    } else {
        0
    };
    let tabu_tenure = (7 + (total_ops / 50).min(12)).saturating_add(effort / 2);
    let stall_limit = (max_iters / 3).max(15);
    let params = heuristic::TabuParams {
        max_iters,
        max_neighbors,
        max_reassign,
        tabu_tenure,
        stall_limit,
    };

    let mut best: Option<(Solution, u32)> = None;
    for idx in selected_indices.into_iter() {
        let candidate = &candidates[idx];
        let candidate_solution = Solution {
            job_schedule: candidate.job_schedule.clone(),
        };
        let improved = heuristic::improve_solution_with_params(
            challenge,
            &candidate_solution,
            params,
            &mut rng,
        )?;
        let is_better = best
            .as_ref()
            .map_or(true, |current| improved.1 < current.1);
        if is_better {
            best = Some(improved);
        }
    }

    let best = best.ok_or_else(|| anyhow!("No valid schedule produced"))?;
    save_solution(&best.0)?;
    Ok(())
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
}

#[allow(dead_code)]
mod complex {
// f_complex: Baseline dispatch + tabu search, tuned for COMPLEX flow type.
//
// Phase 1: 5 dispatch rules + random restarts (d_r baseline, self-contained).
// Phase 2: Tabu search with high reassign ratio (exploits COMPLEX flex~3).
use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::cmp::Ordering;
use std::collections::HashMap;
use tig_challenges::job_scheduling::*;

pub fn help() {
    println!("Baseline dispatch + tabu search, tuned for COMPLEX flow type.");
    println!("Hyperparameters (all optional):");
    println!("  effort: baseline effort (default 5).");
}

const WORK_MIN_WEIGHT: f64 = 0.3;

#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct Hyperparameters {
    pub effort: usize,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self { effort: 5 }
    }
}

#[derive(Clone, Copy)]
enum DispatchRule {
    MostWorkRemaining,
    MostOpsRemaining,
    LeastFlexibility,
    ShortestProcTime,
    LongestProcTime,
}

const ALL_RULES: [DispatchRule; 5] = [
    DispatchRule::MostWorkRemaining,
    DispatchRule::MostOpsRemaining,
    DispatchRule::LeastFlexibility,
    DispatchRule::ShortestProcTime,
    DispatchRule::LongestProcTime,
];

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

fn build_product_work_times(
    product_processing_times: &[Vec<HashMap<usize, u32>>],
) -> Vec<Vec<f64>> {
    let mut product_work_times = Vec::with_capacity(product_processing_times.len());
    for product_ops in product_processing_times.iter() {
        let mut work_ops = Vec::with_capacity(product_ops.len());
        for op in product_ops.iter() {
            let avg = average_processing_time(op);
            let min = min_processing_time(op);
            let work = avg * (1.0 - WORK_MIN_WEIGHT) + min * WORK_MIN_WEIGHT;
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
        job_total_work.push(product_work_times[product].iter().sum());
    }
    job_total_work
}

fn dispatch_priority(
    rule: DispatchRule,
    remaining_work: f64,
    remaining_ops: usize,
    flexibility: usize,
    proc_time: u32,
) -> f64 {
    match rule {
        DispatchRule::MostWorkRemaining => remaining_work,
        DispatchRule::MostOpsRemaining => remaining_ops as f64,
        DispatchRule::LeastFlexibility => -(flexibility as f64),
        DispatchRule::ShortestProcTime => -(proc_time as f64),
        DispatchRule::LongestProcTime => proc_time as f64,
    }
}

fn run_dispatch(
    challenge: &Challenge,
    job_products: &[usize],
    product_work_times: &[Vec<f64>],
    job_ops_len: &[usize],
    job_total_work: &[f64],
    rule: DispatchRule,
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
            use rand::seq::SliceRandom;
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
                    if machine_end > earliest_end {
                        continue;
                    }

                    let flexibility = op_times.len();
                    let rem_ops = job_ops_len[job] - job_next_op_idx[job];
                    let priority = dispatch_priority(
                        rule,
                        job_remaining_work[job],
                        rem_ops,
                        flexibility,
                        proc_time,
                    );

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
                    if machine_end > earliest_end {
                        continue;
                    }

                    let flexibility = op_times.len();
                    let rem_ops = job_ops_len[job] - job_next_op_idx[job];
                    let priority = dispatch_priority(
                        rule,
                        job_remaining_work[job],
                        rem_ops,
                        flexibility,
                        proc_time,
                    );

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
    let mut best_result: Option<ScheduleResult> = None;
    for rule in ALL_RULES.iter().copied() {
        let result = run_dispatch(
            challenge,
            job_products,
            product_work_times,
            job_ops_len,
            job_total_work,
            rule,
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
            let rule = ALL_RULES[rng.gen_range(0..ALL_RULES.len())];
            let random_top_k = rng.gen_range(2..=5);
            let mut local_rng = SmallRng::seed_from_u64(seed);
            let result = run_dispatch(
                challenge,
                job_products,
                product_work_times,
                job_ops_len,
                job_total_work,
                rule,
                Some(random_top_k),
                Some(&mut local_rng),
            )?;
            let makespan = result.makespan;
            if makespan < best_result.makespan {
                best_result = result;
                save_best(&best_result)?;
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
                let result = run_dispatch(
                    challenge,
                    job_products,
                    product_work_times,
                    job_ops_len,
                    job_total_work,
                    restart.rule,
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
        Some(hp) => serde_json::from_value::<Hyperparameters>(Value::Object(hp.clone()))
            .map_err(|e| anyhow!("Failed to parse hyperparameters: {}", e))?,
        None => Hyperparameters::default(),
    };
    let effort = hyperparameters.effort;

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

    let product_work_times = build_product_work_times(&challenge.product_processing_times);
    let job_total_work = build_job_total_work(&job_products, &product_work_times);

    // ── Phase 1: Baseline dispatch search ──────────────────────────────
    let (random_restarts, top_k) = if effort == 0 {
        (10usize, 0usize)
    } else {
        let scaled = effort.max(1);
        let restarts = 250usize.saturating_add(50usize.saturating_mul(scaled));
        let tk = if scaled > 1 {
            2usize.saturating_add(2usize.saturating_mul(scaled))
        } else {
            scaled.saturating_add(1)
        };
        (restarts, tk)
    };
    let local_search_tries = 1usize.saturating_add(3usize.saturating_mul(effort));

    let save_best = |best: &ScheduleResult| -> Result<()> {
        save_solution(&Solution {
            job_schedule: best.job_schedule.clone(),
        })
    };

    let baseline = run_baseline_search(
        challenge,
        &save_best,
        &job_products,
        &product_work_times,
        &job_ops_len,
        &job_total_work,
        random_restarts,
        top_k,
        local_search_tries,
    )?;

    // ── Phase 2: COMPLEX-tuned tabu search ─────────────────────────────
    // COMPLEX has flex~3, so machine reassignment moves are very valuable.
    // Higher reassign ratio (1/2 instead of default 1/3).
    let total_ops: usize = job_ops_len.iter().sum();
    if total_ops == 0 {
        return Ok(());
    }

    let max_iters = if total_ops <= 100 {
        200
    } else if total_ops <= 300 {
        120
    } else {
        60
    };
    let max_neighbors = if total_ops <= 100 {
        40
    } else if total_ops <= 300 {
        25
    } else {
        15
    };
    let max_reassign = (max_neighbors / 2).max(2);
    let tabu_tenure = 7 + (total_ops / 50).min(10);
    let stall_limit = (max_iters / 3).max(10);

    let params = super::super::heuristic::TabuParams {
        max_iters,
        max_neighbors,
        max_reassign,
        tabu_tenure,
        stall_limit,
    };

    let baseline_solution = Solution {
        job_schedule: baseline.job_schedule,
    };

    let mut seed = challenge.seed;
    seed[0] ^= 0xFC;
    let mut rng = SmallRng::from_seed(seed);

    let (improved, _makespan) =
        super::super::heuristic::improve_solution_with_params(challenge, &baseline_solution, params, &mut rng)?;

    save_solution(&improved)?;
    Ok(())
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
