//! Job-shop track: rule-based construction, memetic recombination, shifting-bottleneck seeds and tabu search on a disjunctive graph.

mod types {
pub const NONE_USIZE: usize = usize::MAX;
pub const NONE_U32: u32 = u32::MAX;

/// Weight of the flexibility term for a single-machine-per-operation instance.
pub const FLEX_FACTOR: f64 = 2.2;

/// One operation of a product: its only eligible machine and processing time.
#[derive(Clone, Copy)]
pub struct OpInfo {
    pub machine: usize,
    pub pt: u32,
    /// Processing time weighted by the load of its machine.
    pub bn_avg: f64,
}

/// Instance statistics shared by every phase.
pub struct Pre {
    pub job_products: Vec<usize>,
    pub job_ops_len: Vec<usize>,
    pub product_ops: Vec<Vec<OpInfo>>,
    /// Processing time from operation i to the end of the product (`suffix[len] == 0`).
    pub product_suf_min: Vec<Vec<u32>>,
    /// Same suffix with bn_avg weights.
    pub product_suf_bn: Vec<Vec<f64>>,
    /// Processing time of operation i + 1 (0 at the end).
    pub product_next_min: Vec<Vec<u32>>,
    /// 1.0 when operation i has a successor, else 0.0.
    pub product_next_flag: Vec<Vec<f64>>,
    /// Total processing time per machine.
    pub machine_load: Vec<f64>,
    pub avg_machine_load: f64,
    pub avg_op_min: f64,
    pub time_scale: f64,
    pub max_ops: usize,
    pub max_job_work: f64,
    pub max_job_bn: f64,
    /// Normalised regret of a single-machine operation.
    pub reg_n: f64,
    pub flow_w: f64,
    pub job_flow_pref: Vec<f64>,
    pub jobshopness: f64,
    pub bn_focus: f64,
    pub load_cv: f64,
    pub slack_base: f64,
    pub total_ops: usize,
    /// Machine of each stage when every product follows the same route.
    pub flow_route: Option<Vec<usize>>,
    /// Processing time per job and stage, present with flow_route.
    pub flow_pt_by_job: Option<Vec<Vec<u32>>>,
}

#[derive(Clone, Copy)]
pub struct Cand {
    pub job: usize,
    pub machine: usize,
    pub pt: u32,
    pub score: f64,
}

#[derive(Clone, Copy)]
pub enum GreedyRule {
    MostWork,
    MostOps,
    EarliestEnd,
    ShortestProc,
    LongestProc,
}

/// Disjunctive-graph view of a schedule: operations are numbered job by job, route arcs are implicit, machine arcs follow machine_seq.
#[derive(Clone)]
pub struct DisjSchedule {
    pub n: usize,
    pub num_jobs: usize,
    pub num_machines: usize,
    pub job_offsets: Vec<usize>,
    pub job_succ: Vec<usize>,
    pub indeg_job: Vec<u16>,
    /// indeg_job_plus1[o] == indeg_job[o] + 1.
    pub indeg_job_plus1: Vec<u16>,
    pub node_machine: Vec<usize>,
    pub node_pt: Vec<u32>,
    pub node_job: Vec<usize>,
    pub node_op: Vec<usize>,
    pub machine_seq: Vec<Vec<usize>>,
}

/// Per-operation record used by the tabu search; NONE_U32 marks an absent neighbour.
#[derive(Clone, Copy)]
pub struct NodeRec {
    pub pt: u32,
    pub job_succ: u32,
    pub machine_succ: u32,
    pub indeg: u32,
    pub best_pred: u32,
    pub machine_pred: u32,
}

pub struct EvalBuf {
    pub indeg: Vec<u16>,
    pub start: Vec<u32>,
    pub best_pred: Vec<usize>,
    pub machine_succ: Vec<usize>,
    /// Each operation is pushed once when its in-degree reaches zero; the stack has n slots.
    pub stack: Vec<usize>,
    pub nd: Vec<NodeRec>,
}

impl EvalBuf {
    pub fn new(n: usize) -> Self {
        let empty = NodeRec { pt: 0, job_succ: NONE_U32, machine_succ: NONE_U32, indeg: 0, best_pred: NONE_U32, machine_pred: NONE_U32 };
        Self {
            indeg: vec![0u16; n],
            start: vec![0u32; n],
            best_pred: vec![NONE_USIZE; n],
            machine_succ: vec![NONE_USIZE; n],
            stack: vec![0usize; n],
            nd: vec![empty; n],
        }
    }
}

#[derive(Clone, Copy)]
pub struct MoveCand {
    pub swap: bool,
    pub m: usize,
    pub from: usize,
    pub to: usize,
    pub score: u32,
}

/// Search effort of the job-shop engine.
#[derive(Clone, Copy)]
pub struct EffortConfig {
    /// Iterations of each full tabu search.
    pub job_shop_iters: usize,
    /// Pooled schedules the first tabu searches start from.
    pub js_ts_starts: usize,
    /// Guided kicks spread over the stall window of a tabu search.
    pub js_ts_kick_spread: usize,
    /// Passes of the final shifting-bottleneck re-optimisation.
    pub js_sbp_rounds: usize,
    /// Re-optimisation cycles of the first shifting-bottleneck seed.
    pub js_seed_sbp_reopt_cycles: usize,
    /// Re-optimisation cycles of the second shifting-bottleneck seed.
    pub js_seed_sbp_dual_cycles: usize,
}

impl EffortConfig {
    pub const DEFAULT: EffortConfig = EffortConfig {
        job_shop_iters: 300_000,
        js_ts_starts: 4,
        js_ts_kick_spread: 4,
        js_sbp_rounds: 2,
        js_seed_sbp_reopt_cycles: 1,
        js_seed_sbp_dual_cycles: 5,
    };
}
}
mod infra_shared {
use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::job_scheduling::*;
use super::types::*;

/// Best of five dispatching rules plus ten randomised runs; the first solution saved by the solver.
pub fn run_simple_greedy_baseline(challenge: &Challenge, pre: &Pre) -> Result<(Solution, u32)> {
    let job_total_work: Vec<f64> = pre.job_products.iter()
        .map(|&p| pre.product_ops[p].iter().map(|op| op.pt as f64).sum())
        .collect();
    let rules = [GreedyRule::MostWork, GreedyRule::MostOps, GreedyRule::EarliestEnd, GreedyRule::ShortestProc, GreedyRule::LongestProc];
    let mut best_mk = u32::MAX;
    let mut best_sol: Option<Solution> = None;
    for rule in rules {
        let (sol, mk) = run_greedy_rule(challenge, pre, &job_total_work, rule, None)?;
        if mk < best_mk { best_mk = mk; best_sol = Some(sol); }
    }
    let mut rng = SmallRng::from_seed(challenge.seed);
    for _ in 0..10 {
        let seed = rng.gen::<u64>();
        let rule = rules[rng.gen_range(0..rules.len())];
        let random_top_k = rng.gen_range(2..=5);
        let mut local_rng = SmallRng::seed_from_u64(seed);
        let (sol, mk) = run_greedy_rule(challenge, pre, &job_total_work, rule, Some((random_top_k, &mut local_rng)))?;
        if mk < best_mk { best_mk = mk; best_sol = Some(sol); }
    }
    Ok((best_sol.ok_or_else(|| anyhow!("No greedy solution"))?, best_mk))
}

#[derive(Clone, Copy)]
struct GCandidate { job: usize, priority: f64, end: u32, pt: u32 }

fn greedy_priority(rule: GreedyRule, work_left: f64, ops_left: usize, pt: u32) -> f64 {
    match rule {
        GreedyRule::MostWork => work_left,
        GreedyRule::MostOps => ops_left as f64,
        GreedyRule::EarliestEnd => 0.0,
        GreedyRule::ShortestProc => -(pt as f64),
        GreedyRule::LongestProc => pt as f64,
    }
}

/// Event-driven dispatching: at each time, every idle machine takes the best ready job by rule,
/// or a random one of the top-k when a rng is supplied.
pub fn run_greedy_rule(
    challenge: &Challenge, pre: &Pre, job_total_work: &[f64],
    rule: GreedyRule, mut random_top_k: Option<(usize, &mut SmallRng)>,
) -> Result<(Solution, u32)> {
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;
    let mut job_next_op = vec![0usize; num_jobs];
    let mut job_ready = vec![0u32; num_jobs];
    let mut machine_avail = vec![0u32; num_machines];
    let mut job_schedule: Vec<Vec<(usize, u32)>> = pre.job_ops_len.iter().map(|&len| Vec::with_capacity(len)).collect();
    let mut job_work_left = job_total_work.to_vec();
    let mut remaining = pre.total_ops;
    let mut time = 0u32;
    let eps = 1e-9;
    let mut available_machines: Vec<usize> = Vec::with_capacity(num_machines);
    let mut candidates: Vec<GCandidate> = Vec::new();

    while remaining > 0 {
        available_machines.clear();
        for m in 0..num_machines {
            if machine_avail[m] <= time { available_machines.push(m); }
        }
        if let Some((_, ref mut rng)) = random_top_k { use rand::seq::SliceRandom; available_machines.shuffle(*rng); }

        for &m in &available_machines {
            candidates.clear();
            let mut best: Option<GCandidate> = None;
            for j in 0..num_jobs {
                if job_next_op[j] >= pre.job_ops_len[j] || job_ready[j] > time { continue; }
                let op = pre.product_ops[pre.job_products[j]][job_next_op[j]];
                if op.machine != m { continue; }
                let end = time.max(machine_avail[m]) + op.pt;
                let priority = greedy_priority(rule, job_work_left[j], pre.job_ops_len[j] - job_next_op[j], op.pt);
                let cand = GCandidate { job: j, priority, end, pt: op.pt };
                if random_top_k.is_some() {
                    candidates.push(cand);
                } else {
                    let better = match best {
                        Some(b) => {
                            if (cand.priority - b.priority).abs() > eps { cand.priority > b.priority }
                            else if cand.end != b.end { cand.end < b.end }
                            else if cand.pt != b.pt { cand.pt < b.pt }
                            else { cand.job < b.job }
                        }
                        None => true,
                    };
                    if better { best = Some(cand); }
                }
            }
            let chosen = if let Some((top_k, ref mut rng)) = random_top_k {
                if candidates.is_empty() { continue; }
                candidates.sort_by(|a, b| {
                    if (b.priority - a.priority).abs() > eps { b.priority.partial_cmp(&a.priority).unwrap() }
                    else if a.end != b.end { a.end.cmp(&b.end) }
                    else if a.pt != b.pt { a.pt.cmp(&b.pt) }
                    else { a.job.cmp(&b.job) }
                });
                let top = candidates.len().min(top_k);
                candidates[rng.gen_range(0..top)]
            } else {
                let Some(best) = best else { continue };
                best
            };
            let j = chosen.job;
            let st = time.max(machine_avail[m]);
            let end = st + chosen.pt;
            job_schedule[j].push((m, st));
            job_next_op[j] += 1;
            job_ready[j] = end;
            machine_avail[m] = end;
            job_work_left[j] -= chosen.pt as f64;
            if job_work_left[j] < 0.0 { job_work_left[j] = 0.0; }
            remaining -= 1;
        }

        if remaining == 0 { break; }
        let mut next = u32::MAX;
        for &t in &machine_avail { if t > time && t < next { next = t; } }
        for j in 0..num_jobs { if job_next_op[j] < pre.job_ops_len[j] && job_ready[j] > time && job_ready[j] < next { next = job_ready[j]; } }
        if next == u32::MAX { return Err(anyhow!("Greedy baseline stuck")); }
        time = next;
    }
    let mk = job_ready.iter().copied().max().unwrap_or(0);
    Ok((Solution { job_schedule }, mk))
}

pub fn build_disj_from_solution(pre: &Pre, challenge: &Challenge, sol: &Solution) -> Result<DisjSchedule> {
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;
    let mut job_offsets = vec![0usize; num_jobs + 1];
    for j in 0..num_jobs { job_offsets[j + 1] = job_offsets[j] + pre.job_ops_len[j]; }
    let n = job_offsets[num_jobs];
    if n == 0 { return Err(anyhow!("No operations")); }
    let mut node_machine = vec![0usize; n];
    let mut node_pt = vec![0u32; n];
    let mut node_job = vec![0usize; n];
    let mut node_op = vec![0usize; n];
    let mut per_machine: Vec<Vec<(u32, usize)>> = vec![Vec::new(); num_machines];
    for job in 0..num_jobs {
        let expected = pre.job_ops_len[job];
        if sol.job_schedule[job].len() != expected { return Err(anyhow!("Invalid solution: job {} ops len mismatch", job)); }
        let product = pre.job_products[job];
        for op_idx in 0..expected {
            let id = job_offsets[job] + op_idx;
            let (m, st) = sol.job_schedule[job][op_idx];
            let op = pre.product_ops[product][op_idx];
            if m != op.machine || m >= num_machines { return Err(anyhow!("Invalid solution: machine mismatch")); }
            node_machine[id] = m;
            node_pt[id] = op.pt;
            node_job[id] = job;
            node_op[id] = op_idx;
            per_machine[m].push((st, id));
        }
    }
    let mut machine_seq: Vec<Vec<usize>> = Vec::with_capacity(num_machines);
    for m in 0..num_machines {
        per_machine[m].sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        machine_seq.push(per_machine[m].iter().map(|&(_, id)| id).collect());
    }
    let mut job_succ = vec![NONE_USIZE; n];
    let mut indeg_job = vec![0u16; n];
    for job in 0..num_jobs {
        let len = pre.job_ops_len[job];
        let base = job_offsets[job];
        for k in 0..len { let id = base + k; if k + 1 < len { job_succ[id] = id + 1; indeg_job[id + 1] = indeg_job[id + 1].saturating_add(1); } }
    }
    let mut indeg_job_plus1 = indeg_job.clone();
    for d in indeg_job_plus1.iter_mut() { *d += 1; }
    Ok(DisjSchedule { n, num_jobs, num_machines, job_offsets, job_succ, indeg_job, indeg_job_plus1, node_machine, node_pt, node_job, node_op, machine_seq })
}

/// Rebuilds machine links and in-degrees from machine_seq; every operation sits on exactly one machine.
fn prepare_eval(ds: &DisjSchedule, buf: &mut EvalBuf) {
    buf.indeg.copy_from_slice(&ds.indeg_job_plus1);
    buf.start.fill(0);
    for seq in &ds.machine_seq {
        if seq.is_empty() { continue; }
        buf.indeg[seq[0]] -= 1;
        let mut prev = seq[0];
        for &v in &seq[1..] {
            buf.machine_succ[prev] = v;
            prev = v;
        }
        buf.machine_succ[prev] = NONE_USIZE;
    }
}

/// Longest-path makespan, or None when a cycle exists or the makespan reaches bound.
/// Start times are partial on early exit; call eval_disj before reading predecessors.
pub fn eval_disj_trial(ds: &DisjSchedule, buf: &mut EvalBuf, bound: u32) -> Option<u32> {
    let n = ds.n;
    prepare_eval(ds, buf);
    let mut sl = 0usize;
    for j in 0..ds.num_jobs {
        let h = ds.job_offsets[j];
        if h < ds.job_offsets[j + 1] && buf.indeg[h] == 0 { buf.stack[sl] = h; sl += 1; }
    }
    let mut processed = 0usize;
    let mut mk = 0u32;
    while sl > 0 {
        sl -= 1;
        let u = buf.stack[sl];
        processed += 1;
        let end_u = buf.start[u].saturating_add(ds.node_pt[u]);
        if end_u > mk {
            mk = end_u;
            if mk >= bound { return None; }
        }
        let js = ds.job_succ[u];
        if js != NONE_USIZE {
            if buf.start[js] < end_u { buf.start[js] = end_u; }
            buf.indeg[js] -= 1;
            if buf.indeg[js] == 0 { buf.stack[sl] = js; sl += 1; }
        }
        let ms = buf.machine_succ[u];
        if ms != NONE_USIZE {
            if buf.start[ms] < end_u { buf.start[ms] = end_u; }
            buf.indeg[ms] -= 1;
            if buf.indeg[ms] == 0 { buf.stack[sl] = ms; sl += 1; }
        }
    }
    if processed != n { return None; }
    Some(mk)
}

/// Longest-path evaluation: fills buf.start and buf.best_pred, returns (makespan, last node of a critical path).
pub fn eval_disj(ds: &DisjSchedule, buf: &mut EvalBuf) -> Option<(u32, usize)> {
    let n = ds.n;
    prepare_eval(ds, buf);
    let mut sl = 0usize;
    for j in 0..ds.num_jobs {
        let h = ds.job_offsets[j];
        if h < ds.job_offsets[j + 1] && buf.indeg[h] == 0 {
            buf.best_pred[h] = NONE_USIZE;
            buf.stack[sl] = h;
            sl += 1;
        }
    }
    let mut processed = 0usize;
    let mut mk = 0u32;
    let mut mk_node = 0usize;
    while sl > 0 {
        sl -= 1;
        let u = buf.stack[sl];
        processed += 1;
        let end_u = buf.start[u].saturating_add(ds.node_pt[u]);
        if end_u > mk { mk = end_u; mk_node = u; }
        let js = ds.job_succ[u];
        if js != NONE_USIZE {
            if buf.start[js] < end_u {
                buf.start[js] = end_u;
                buf.best_pred[js] = u;
            }
            buf.indeg[js] -= 1;
            if buf.indeg[js] == 0 { buf.stack[sl] = js; sl += 1; }
        }
        let ms = buf.machine_succ[u];
        if ms != NONE_USIZE {
            if buf.start[ms] < end_u {
                buf.start[ms] = end_u;
                buf.best_pred[ms] = u;
            }
            buf.indeg[ms] -= 1;
            if buf.indeg[ms] == 0 { buf.stack[sl] = ms; sl += 1; }
        }
    }
    if processed != n { return None; }
    Some((mk, mk_node))
}

pub fn disj_to_solution(pre: &Pre, ds: &DisjSchedule, start: &[u32]) -> Solution {
    let num_jobs = ds.num_jobs;
    let mut job_schedule: Vec<Vec<(usize, u32)>> = Vec::with_capacity(num_jobs);
    for j in 0..num_jobs {
        let len = pre.job_ops_len[j];
        let base = ds.job_offsets[j];
        let mut v = Vec::with_capacity(len);
        for k in 0..len { let id = base + k; v.push((ds.node_machine[id], start[id])); }
        job_schedule.push(v);
    }
    Solution { job_schedule }
}

/// Marks the critical path ending at mk_node.
fn mark_critical(buf: &EvalBuf, crit: &mut [bool], mk_node: usize) {
    crit.fill(false);
    let mut u = mk_node;
    while u != NONE_USIZE { crit[u] = true; u = buf.best_pred[u]; }
}

/// Maximal runs of consecutive critical operations on one machine, as (machine, first, last) with last > first.
fn critical_blocks(ds: &DisjSchedule, buf: &EvalBuf, crit: &[bool], blocks: &mut Vec<(usize, usize, usize)>) {
    blocks.clear();
    for m in 0..ds.num_machines {
        let seq = &ds.machine_seq[m];
        if seq.len() <= 1 { continue; }
        let mut i = 0usize;
        while i < seq.len() {
            if !crit[seq[i]] { i += 1; continue; }
            let bstart = i;
            let mut bend = i;
            while bend + 1 < seq.len() {
                let x = seq[bend];
                let y = seq[bend + 1];
                if !crit[y] || buf.start[y] != buf.start[x].saturating_add(ds.node_pt[x]) { break; }
                bend += 1;
            }
            if bend > bstart { blocks.push((m, bstart, bend)); }
            i = bend + 1;
        }
    }
}

/// Descent on critical blocks, then perturb_cycles rounds of two random block swaps followed by a new descent.
pub fn critical_block_move_local_search_ex(
    pre: &Pre, challenge: &Challenge, base_sol: &Solution,
    max_iters: usize, top_cands: usize, perturb_cycles: usize,
) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n);
    let mut crit = vec![false; ds.n];
    let mut cur_eval = match eval_disj(&ds, &mut buf) { Some(x) => x, None => return Ok(None) };
    let initial_mk = cur_eval.0;
    descent_phase(&mut ds, &mut buf, &mut crit, &mut cur_eval, max_iters, top_cands);
    let Some((mk_after, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let mut global_best_mk = mk_after;
    let mut global_best_ds = ds.clone();
    let mut sol_hash: u64 = 0;
    for m in 0..ds.num_machines.min(8) {
        if !ds.machine_seq[m].is_empty() {
            let first_node = ds.machine_seq[m][0];
            sol_hash ^= (first_node as u64).wrapping_mul(0xD2B54A6B68A5);
            sol_hash = sol_hash.rotate_left(7);
        }
    }
    let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (initial_mk as u64).wrapping_shl(16) ^ (ds.n as u64) ^ sol_hash;
    let mut blocks: Vec<(usize, usize, usize)> = Vec::new();
    for _ in 0..perturb_cycles {
        ds = global_best_ds.clone();
        let Some((_, mk_node)) = eval_disj(&ds, &mut buf) else { break };
        mark_critical(&buf, &mut crit, mk_node);
        critical_blocks(&ds, &buf, &crit, &mut blocks);
        if blocks.is_empty() { break; }
        for _ in 0..2 {
            pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
            let (m, bstart, bend) = blocks[(pseed as usize) % blocks.len()];
            let block_len = bend - bstart;
            pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
            let swap_pos = bstart + ((pseed as usize) % block_len);
            if swap_pos + 1 < ds.machine_seq[m].len() { ds.machine_seq[m].swap(swap_pos, swap_pos + 1); }
        }
        match eval_disj(&ds, &mut buf) { Some(x) => cur_eval = x, None => continue }
        descent_phase(&mut ds, &mut buf, &mut crit, &mut cur_eval, max_iters, top_cands);
        if let Some((mk_now, _)) = eval_disj(&ds, &mut buf) { if mk_now < global_best_mk { global_best_mk = mk_now; global_best_ds = ds.clone(); } }
    }
    if global_best_mk >= initial_mk { return Ok(None); }
    ds = global_best_ds;
    let Some((mk_final, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    Ok(Some((disj_to_solution(pre, &ds, &buf.start), mk_final)))
}

/// Steepest descent: block-end insertions and adjacent swaps around critical blocks, best of top_cands probes per step.
fn descent_phase(
    ds: &mut DisjSchedule, buf: &mut EvalBuf, crit: &mut [bool],
    cur_eval: &mut (u32, usize), max_iters: usize, top_cands: usize,
) {
    let mut cur_mk = cur_eval.0;
    let mut seen_probes: Vec<(bool, usize, usize, usize)> = Vec::with_capacity(64);
    let mut blocks: Vec<(usize, usize, usize)> = Vec::new();
    for _ in 0..max_iters {
        mark_critical(buf, crit, cur_eval.1);
        critical_blocks(ds, buf, crit, &mut blocks);
        let mut cands: Vec<MoveCand> = Vec::with_capacity(top_cands.min(64));
        for &(m, bstart, bend) in &blocks {
            let seq = &ds.machine_seq[m];
            let max_shift = bend - bstart;
            let mut shifts: [usize; 3] = [1, 2, max_shift];
            for sh in shifts.iter_mut() { if *sh > max_shift { *sh = 0; } }
            for &sh in &shifts {
                if sh == 0 { continue; }
                push_top_k_move(&mut cands, MoveCand { swap: false, m, from: bstart, to: bstart + sh, score: buf.start[seq[bstart + sh]] }, top_cands);
                push_top_k_move(&mut cands, MoveCand { swap: false, m, from: bend, to: bend - sh, score: buf.start[seq[bend]] }, top_cands);
            }
            if bstart > 0 { push_top_k_move(&mut cands, MoveCand { swap: true, m, from: bstart - 1, to: 0, score: buf.start[seq[bstart]] }, top_cands); }
            if bend + 1 < seq.len() { push_top_k_move(&mut cands, MoveCand { swap: true, m, from: bend, to: 0, score: buf.start[seq[bend]] }, top_cands); }
            push_top_k_move(&mut cands, MoveCand { swap: true, m, from: bstart, to: 0, score: buf.start[seq[bstart + 1]] }, top_cands);
            push_top_k_move(&mut cands, MoveCand { swap: true, m, from: bend - 1, to: 0, score: buf.start[seq[bend]] }, top_cands);
        }
        if cands.is_empty() { break; }
        let mut best_cand: Option<MoveCand> = None;
        let mut best_mk = cur_mk;
        seen_probes.clear();
        for cand in &cands {
            let key = (cand.swap, cand.m, cand.from, cand.to);
            if seen_probes.iter().any(|&k| k == key) { continue; }
            seen_probes.push(key);
            let m = cand.m;
            if cand.swap {
                if cand.from + 1 >= ds.machine_seq[m].len() { continue; }
                ds.machine_seq[m].swap(cand.from, cand.from + 1);
                if let Some(mk2) = eval_disj_trial(ds, buf, best_mk) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                ds.machine_seq[m].swap(cand.from, cand.from + 1);
            } else {
                if cand.from >= ds.machine_seq[m].len() { continue; }
                let new_idx = apply_insert(&mut ds.machine_seq[m], cand.from, cand.to);
                if let Some(mk2) = eval_disj_trial(ds, buf, best_mk) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                apply_insert(&mut ds.machine_seq[m], new_idx, cand.from);
            }
        }
        let Some(bc) = best_cand else { break };
        let m = bc.m;
        let accepted = if bc.swap {
            ds.machine_seq[m].swap(bc.from, bc.from + 1);
            match eval_disj(ds, buf) {
                Some(ne) if ne.0 < cur_mk => { *cur_eval = ne; cur_mk = ne.0; true }
                _ => { ds.machine_seq[m].swap(bc.from, bc.from + 1); false }
            }
        } else {
            let new_idx = apply_insert(&mut ds.machine_seq[m], bc.from, bc.to);
            match eval_disj(ds, buf) {
                Some(ne) if ne.0 < cur_mk => { *cur_eval = ne; cur_mk = ne.0; true }
                _ => { apply_insert(&mut ds.machine_seq[m], new_idx, bc.from); false }
            }
        };
        if !accepted { break; }
    }
}

/// Moves seq[from] so that it lands at index `to`; returns the final index.
#[inline]
pub fn apply_insert(seq: &mut [usize], from: usize, to: usize) -> usize {
    let len = seq.len();
    if len == 0 || from >= len { return from.min(len.saturating_sub(1)); }
    let t = to.min(len - 1);
    if t > from {
        seq[from..=t].rotate_left(1);
    } else if t < from {
        seq[t..=from].rotate_right(1);
    }
    t
}

/// Keeps the k highest scores, insertion-ordered among equal scores.
#[inline]
pub fn push_top_k_move(top: &mut Vec<MoveCand>, c: MoveCand, k: usize) {
    if k == 0 { return; }
    let mut pos = top.len();
    while pos > 0 && top[pos - 1].score < c.score { pos -= 1; }
    if pos >= k { return; }
    top.insert(pos, c);
    if top.len() > k { top.pop(); }
}

/// Per-job urgency in [0, 1] from the completion time of each job in a solution.
pub fn job_bias_from_solution(pre: &Pre, sol: &Solution) -> Vec<f64> {
    let num_jobs = pre.job_ops_len.len();
    let mut completion = vec![0u32; num_jobs];
    let mut makespan = 0u32;
    for job in 0..num_jobs {
        let product = pre.job_products[job];
        let mut end_j = 0u32;
        for (op_idx, &(_, st)) in sol.job_schedule[job].iter().enumerate() {
            end_j = end_j.max(st.saturating_add(pre.product_ops[product][op_idx].pt));
        }
        completion[job] = end_j;
        makespan = makespan.max(end_j);
    }
    let denom = (makespan as f64).max(1.0);
    let exp = 3.0 + 0.6 * pre.jobshopness;
    completion.into_iter().map(|c| ((c as f64) / denom).powf(exp).clamp(0.0, 1.0)).collect()
}

/// Per-machine penalty in [0, 1] mixing finishing time and load of each machine in a solution.
pub fn machine_penalty_from_solution(pre: &Pre, sol: &Solution, num_machines: usize) -> Vec<f64> {
    let num_jobs = pre.job_ops_len.len();
    let mut m_end = vec![0u32; num_machines];
    let mut m_sum = vec![0u64; num_machines];
    let mut makespan = 0u32;
    for job in 0..num_jobs {
        let product = pre.job_products[job];
        for (op_idx, &(m, st)) in sol.job_schedule[job].iter().enumerate() {
            let pt = pre.product_ops[product][op_idx].pt;
            let end = st.saturating_add(pt);
            if end > m_end[m] { m_end[m] = end; }
            m_sum[m] = m_sum[m].saturating_add(pt as u64);
            makespan = makespan.max(end);
        }
    }
    let mk = (makespan as f64).max(1.0);
    let total: u64 = m_sum.iter().copied().sum();
    let avg = ((total as f64) / (num_machines as f64).max(1.0)).max(1.0);
    let w_load = if pre.jobshopness > 0.45 { (0.20 + 0.12 * pre.jobshopness).clamp(0.18, 0.58) } else { 0.0 };
    let w_end = 1.0 - w_load;
    let exp = 2.0 + 0.55 * pre.jobshopness;
    let mut mp = vec![0.0f64; num_machines];
    for m in 0..num_machines {
        let endn = (m_end[m] as f64 / mk).clamp(0.0, 1.0);
        let loadr = ((m_sum[m] as f64) / avg).max(0.0);
        let loadn = (loadr / (loadr + 1.0)).clamp(0.0, 1.0);
        let mix = (w_end * endn + w_load * loadn).clamp(0.0, 1.0);
        mp[m] = mix.powf(exp).clamp(0.0, 1.0);
    }
    mp
}

/// Inserts sol into a list kept sorted by makespan at the position found by binary search, truncated to cap.
pub fn push_top_solutions(top: &mut Vec<(Solution, u32)>, sol: &Solution, mk: u32, cap: usize) {
    let pos = top.binary_search_by_key(&mk, |(_, m)| *m).unwrap_or_else(|e| e);
    top.insert(pos, (sol.clone(), mk));
    if top.len() > cap { top.truncate(cap); }
}

/// NEH insertion heuristic on the common route, for instances where every product follows it.
pub fn neh_flow_solution(pre: &Pre, num_jobs: usize, num_machines: usize) -> Result<(Solution, u32)> {
    let route = pre.flow_route.as_ref().ok_or_else(|| anyhow!("No flow route"))?;
    let pt = pre.flow_pt_by_job.as_ref().ok_or_else(|| anyhow!("No flow pt"))?;
    if route.is_empty() || pt.len() != num_jobs { return Err(anyhow!("Invalid flow data")); }
    let mut jobs: Vec<usize> = (0..num_jobs).collect();
    jobs.sort_unstable_by(|&a, &b| {
        let sa: u32 = pt[a].iter().copied().sum();
        let sb: u32 = pt[b].iter().copied().sum();
        sb.cmp(&sa).then_with(|| a.cmp(&b))
    });
    let mut seq: Vec<usize> = Vec::with_capacity(num_jobs);
    let mut mready = vec![0u32; num_machines];
    let mut tmp = Vec::with_capacity(num_jobs);
    for &j in &jobs {
        if seq.is_empty() { seq.push(j); continue; }
        let mut best_mk = u32::MAX;
        let mut best_pos = 0usize;
        for pos in 0..=seq.len() {
            tmp.clear();
            tmp.extend_from_slice(&seq[..pos]);
            tmp.push(j);
            tmp.extend_from_slice(&seq[pos..]);
            let mk = flow_route_makespan(&tmp, route, pt, &mut mready);
            if mk < best_mk { best_mk = mk; best_pos = pos; }
        }
        seq.insert(best_pos, j);
    }
    let mk = flow_route_makespan(&seq, route, pt, &mut mready);
    let mut job_schedule: Vec<Vec<(usize, u32)>> = vec![Vec::with_capacity(route.len()); num_jobs];
    mready.fill(0);
    for &j in &seq {
        let mut prev_end = 0u32;
        for (op_idx, &m) in route.iter().enumerate() {
            let st = prev_end.max(mready[m]);
            job_schedule[j].push((m, st));
            let end = st.saturating_add(pt[j][op_idx]);
            mready[m] = end;
            prev_end = end;
        }
    }
    Ok((Solution { job_schedule }, mk))
}

/// Makespan of a permutation on a fixed route (machines may repeat along the route).
fn flow_route_makespan(seq: &[usize], route: &[usize], pt: &[Vec<u32>], mready: &mut [u32]) -> u32 {
    mready.fill(0);
    let mut mk = 0u32;
    for &j in seq {
        let row = &pt[j];
        let mut prev = 0u32;
        for (op_idx, &m) in route.iter().enumerate() {
            let st = prev.max(mready[m]);
            let end = st.saturating_add(row[op_idx]);
            mready[m] = end;
            prev = end;
        }
        if prev > mk { mk = prev; }
    }
    mk
}

#[inline]
pub fn push_top_k(top: &mut Vec<Cand>, c: Cand, k: usize) {
    if k == 0 { return; }
    let mut pos = top.len();
    while pos > 0 && top[pos - 1].score < c.score { pos -= 1; }
    if pos >= k { return; }
    top.insert(pos, c);
    if top.len() > k { top.pop(); }
}
}
mod preprocess {
use anyhow::{anyhow, Result};
use tig_challenges::job_scheduling::*;
use super::types::*;

/// Permutation flow-shop makespan of seq with pt[job][stage].
fn flow_makespan(seq: &[usize], pt: &[Vec<u32>], comp: &mut [u32]) -> u32 {
    comp.fill(0);
    for &j in seq {
        let row = &pt[j];
        if row.is_empty() { continue; }
        comp[0] = comp[0].saturating_add(row[0]);
        for k in 1..row.len() {
            comp[k] = comp[k].max(comp[k - 1]).saturating_add(row[k]);
        }
    }
    *comp.last().unwrap_or(&0)
}

/// Requires exactly one eligible machine per operation.
pub fn build_pre(challenge: &Challenge) -> Result<Pre> {
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;

    let mut job_products = Vec::with_capacity(num_jobs);
    for (p, &cnt) in challenge.jobs_per_product.iter().enumerate() {
        for _ in 0..cnt { job_products.push(p); }
    }
    if job_products.len() != num_jobs {
        return Err(anyhow!("jobs_per_product sum mismatch"));
    }
    let num_products = challenge.product_processing_times.len();

    let mut product_ops: Vec<Vec<OpInfo>> = Vec::with_capacity(num_products);
    let mut machine_load = vec![0.0f64; num_machines];
    let mut total_ops: usize = 0;
    let mut total_work: f64 = 0.0;
    let mut max_ops: usize = 1;
    let mut max_job_work: f64 = 1.0;

    for (p, ops) in challenge.product_processing_times.iter().enumerate() {
        max_ops = max_ops.max(ops.len());
        let mut ops_info: Vec<OpInfo> = Vec::with_capacity(ops.len());
        let mut sum_pt: u64 = 0;
        let mut sum_pt_f: f64 = 0.0;
        for op in ops {
            if op.len() != 1 {
                return Err(anyhow!("expected exactly one eligible machine per operation"));
            }
            let (&m, &pt) = op.iter().next().unwrap();
            if m >= num_machines {
                return Err(anyhow!("machine id out of range"));
            }
            sum_pt += pt as u64;
            sum_pt_f += pt as f64;
            ops_info.push(OpInfo { machine: m, pt, bn_avg: 0.0 });
        }
        max_job_work = max_job_work.max(sum_pt_f);
        let cnt_f = challenge.jobs_per_product[p] as f64;
        total_ops += ops_info.len() * challenge.jobs_per_product[p];
        total_work += (sum_pt as f64) * cnt_f;
        for oi in &ops_info {
            machine_load[oi.machine] += (oi.pt as f64) * cnt_f;
        }
        product_ops.push(ops_info);
    }

    let job_ops_len: Vec<usize> = job_products.iter().map(|&p| product_ops[p].len()).collect();
    let avg_machine_load = (total_work / (num_machines as f64).max(1.0)).max(1.0);
    let avg_op_min = (total_work / (total_ops as f64).max(1.0)).max(1.0);
    let reg_n = (avg_op_min * 2.6 / avg_op_min.max(1.0)).clamp(0.0, 6.0);

    let load_cv = {
        let mean = avg_machine_load.max(1e-9);
        let mut var = 0.0f64;
        for &x in &machine_load {
            let d = (x / mean) - 1.0;
            var += d * d;
        }
        (var / (num_machines as f64)).sqrt().clamp(0.0, 2.5)
    };

    // Share of jobs whose stage i runs on the most common machine of that stage, averaged over stages.
    let mut flow_sum = 0.0f64;
    let mut flow_cnt = 0usize;
    let mut counts = vec![0u32; num_machines];
    for op_idx in 0..max_ops {
        counts.fill(0);
        let mut tot = 0u32;
        for p in 0..num_products {
            if op_idx >= product_ops[p].len() { continue; }
            let w = challenge.jobs_per_product[p] as u32;
            if w == 0 { continue; }
            let bm = product_ops[p][op_idx].machine;
            counts[bm] = counts[bm].saturating_add(w);
            tot = tot.saturating_add(w);
        }
        if tot > 0 {
            let mx = counts.iter().copied().max().unwrap_or(0);
            flow_sum += (mx as f64) / (tot as f64);
            flow_cnt += 1;
        }
    }
    let flow_like = if flow_cnt > 0 { (flow_sum / (flow_cnt as f64)).clamp(0.0, 1.0) } else { 0.5 };
    let jobshopness = (1.0 - flow_like).clamp(0.0, 1.0);

    let mut machine_weight = vec![1.0f64; num_machines];
    {
        let mean = avg_machine_load.max(1e-9);
        let exp = (1.10 + 0.35 * load_cv + 0.20 * jobshopness).clamp(1.05, 1.70);
        for m in 0..num_machines {
            let r = (machine_load[m] / mean).max(0.05);
            machine_weight[m] = r.powf(exp).clamp(0.55, 3.75);
        }
    }

    let bn_focus = (2.6 * (1.0 + 0.55 * load_cv) * (0.85 + 0.55 * jobshopness)).clamp(0.6, 3.4);

    let mut product_suf_min: Vec<Vec<u32>> = Vec::with_capacity(product_ops.len());
    let mut product_suf_bn: Vec<Vec<f64>> = Vec::with_capacity(product_ops.len());
    let mut product_next_min: Vec<Vec<u32>> = Vec::with_capacity(product_ops.len());
    let mut product_next_flag: Vec<Vec<f64>> = Vec::with_capacity(product_ops.len());
    let mut max_job_bn: f64 = 1e-9;
    for ops in product_ops.iter_mut() {
        let n = ops.len();
        let mut suf_m = vec![0u32; n + 1];
        let mut suf_bn = vec![0.0f64; n + 1];
        let mut nxt_m = vec![0u32; n + 1];
        let mut nxt_flag = vec![0.0f64; n + 1];
        for i in (0..n).rev() {
            let oi = &mut ops[i];
            oi.bn_avg = (oi.pt as f64) * machine_weight[oi.machine];
            suf_m[i] = suf_m[i + 1].saturating_add(oi.pt);
            suf_bn[i] = suf_bn[i + 1] + oi.bn_avg;
            if i + 1 < n {
                nxt_m[i] = ops[i + 1].pt;
                nxt_flag[i] = 1.0;
            }
        }
        max_job_bn = max_job_bn.max(suf_bn[0]);
        product_suf_min.push(suf_m);
        product_suf_bn.push(suf_bn);
        product_next_min.push(nxt_m);
        product_next_flag.push(nxt_flag);
    }

    let time_scale = (avg_machine_load * (2.65 + 0.15 * load_cv + 0.10 * jobshopness)).max(1.0);

    // Flow-like instances get a job order preference from a permutation flow-shop insertion heuristic.
    let mut job_flow_pref = vec![0.0f64; num_jobs];
    let use_flow_pref = flow_like > 0.82 && jobshopness < 0.38 && max_ops >= 2;
    if use_flow_pref {
        let m = max_ops.max(1);
        let mut job_pt: Vec<Vec<u32>> = Vec::with_capacity(num_jobs);
        for j in 0..num_jobs {
            let ops = &product_ops[job_products[j]];
            let mut v = vec![0u32; m];
            for s in 0..m.min(ops.len()) { v[s] = ops[s].pt; }
            job_pt.push(v);
        }
        let mut jobs2: Vec<usize> = (0..num_jobs).collect();
        jobs2.sort_unstable_by(|&a, &b| {
            let sa: u32 = job_pt[a].iter().copied().sum();
            let sb: u32 = job_pt[b].iter().copied().sum();
            sb.cmp(&sa).then_with(|| a.cmp(&b))
        });
        let mut perm: Vec<usize> = Vec::with_capacity(num_jobs);
        let mut comp = vec![0u32; m];
        let mut tmp: Vec<usize> = Vec::with_capacity(num_jobs);
        for &j in &jobs2 {
            if perm.is_empty() { perm.push(j); continue; }
            let mut best_mk = u32::MAX;
            let mut best_pos = 0usize;
            for pos in 0..=perm.len() {
                tmp.clear();
                tmp.extend_from_slice(&perm[..pos]);
                tmp.push(j);
                tmp.extend_from_slice(&perm[pos..]);
                let mk = flow_makespan(&tmp, &job_pt, &mut comp);
                if mk < best_mk { best_mk = mk; best_pos = pos; }
            }
            perm.insert(best_pos, j);
        }
        let n1 = (num_jobs.saturating_sub(1)) as f64;
        for (pos, &j) in perm.iter().enumerate() {
            job_flow_pref[j] = if n1 > 0.0 { 1.0 - (pos as f64) / n1 } else { 1.0 };
        }
    }
    let flow_w = if use_flow_pref {
        let t = ((flow_like - 0.82) / 0.18).clamp(0.0, 1.0);
        (0.10 + 0.26 * t).clamp(0.10, 0.36)
    } else {
        0.0
    };
    let slack_base = (0.04 + 0.14 * jobshopness).clamp(0.03, 0.22);

    // Common machine route across all products, when one exists.
    let mut flow_route: Option<Vec<usize>> = None;
    let mut flow_pt_by_job: Option<Vec<Vec<u32>>> = None;
    if !product_ops.is_empty() {
        let common_len = product_ops[0].len();
        let mut ok = common_len > 0 && product_ops.iter().all(|ops| ops.len() == common_len);
        if ok {
            let mut route: Vec<usize> = Vec::with_capacity(common_len);
            for i in 0..common_len {
                let m0 = product_ops[0][i].machine;
                if product_ops.iter().any(|ops| ops[i].machine != m0) { ok = false; break; }
                route.push(m0);
            }
            if ok {
                let pt_by_job: Vec<Vec<u32>> = job_products.iter()
                    .map(|&prod| (0..common_len).map(|i| product_ops[prod][i].pt).collect())
                    .collect();
                flow_route = Some(route);
                flow_pt_by_job = Some(pt_by_job);
            }
        }
    }

    Ok(Pre {
        job_products,
        job_ops_len,
        product_ops,
        product_suf_min,
        product_suf_bn,
        product_next_min,
        product_next_flag,
        machine_load,
        avg_machine_load,
        avg_op_min,
        time_scale,
        max_ops: max_ops.max(1),
        max_job_work: max_job_work.max(1.0),
        max_job_bn: max_job_bn.max(1e-9),
        reg_n,
        flow_w,
        job_flow_pref,
        jobshopness,
        bn_focus,
        load_cv,
        slack_base,
        total_ops,
        flow_route,
        flow_pt_by_job,
    })
}
}
mod job_shop {
use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::job_scheduling::*;
use super::types::*;
use super::infra_shared::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rule {
    Adaptive, BnHeavy, EndTight, CriticalPath, MostWork, EarlyEnd, Regret, ShortestProc,
}

const RULES: [Rule; 8] = [Rule::Adaptive, Rule::BnHeavy, Rule::EndTight, Rule::CriticalPath, Rule::MostWork, Rule::EarlyEnd, Rule::Regret, Rule::ShortestProc];

#[inline]
fn rule_idx(r: Rule) -> usize {
    match r { Rule::Adaptive => 0, Rule::BnHeavy => 1, Rule::EndTight => 2, Rule::CriticalPath => 3, Rule::MostWork => 4, Rule::EarlyEnd => 5, Rule::Regret => 6, Rule::ShortestProc => 7 }
}

/// Features of one ready (job, machine) pair at the current dispatch time.
struct CandCtx {
    job: usize,
    product: usize,
    op_idx: usize,
    ops_rem: usize,
    pt: u32,
    time: u32,
    target_mk: Option<u32>,
    /// Earliest end of the operation on its machine.
    end: u32,
    progress: f64,
    job_bias: f64,
    machine_penalty: f64,
    dynamic_load: f64,
    jitter: f64,
}

/// Urgency in [0, 4] from the slack of the job's lower bound against a target makespan.
#[inline]
fn slack_urgency(pre: &Pre, target_mk: Option<u32>, time: u32, product: usize, op_idx: usize) -> f64 {
    let Some(tgt) = target_mk else { return 0.0 };
    let lb = (time as u64).saturating_add(pre.product_suf_min[product][op_idx] as u64);
    let slack = (tgt as i64) - (lb as i64);
    let scale = (0.70 * pre.avg_op_min).max(1.0);
    let pos = (slack.max(0) as f64) / scale;
    let neg = ((-slack).max(0) as f64) / scale;
    (1.0 / (1.0 + pos)).clamp(0.0, 1.0) + (0.35 * neg).min(3.0)
}

#[inline(always)]
fn progress_phase(progress: f64) -> u8 {
    if progress < 0.34 { 0 } else if progress < 0.72 { 1 } else { 2 }
}

#[inline(always)]
fn rem_min_n(pre: &Pre, c: &CandCtx) -> f64 {
    (pre.product_suf_min[c.product][c.op_idx] as f64) / pre.avg_machine_load.max(1.0)
}

#[inline(always)]
fn rem_avg_n(pre: &Pre, c: &CandCtx) -> f64 {
    (pre.product_suf_min[c.product][c.op_idx] as f64) / pre.max_job_work.max(1e-9)
}

#[inline(always)]
fn ops_n(pre: &Pre, c: &CandCtx) -> f64 {
    (c.ops_rem as f64) / (pre.max_ops as f64).max(1.0)
}

#[inline(always)]
fn end_n(pre: &Pre, c: &CandCtx) -> f64 {
    (c.end as f64) / pre.time_scale.max(1.0)
}

#[inline(always)]
fn proc_n(pre: &Pre, c: &CandCtx) -> f64 {
    (c.pt as f64) / pre.avg_op_min.max(1.0)
}

#[inline(always)]
fn density_n(pre: &Pre, c: &CandCtx) -> f64 {
    let rem_min = pre.product_suf_min[c.product][c.op_idx] as f64;
    ((rem_min / (c.ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0)
}

#[inline(always)]
fn next_term_raw(pre: &Pre, c: &CandCtx) -> f64 {
    let next_min_n = (pre.product_next_min[c.product][c.op_idx] as f64) / pre.avg_machine_load.max(1.0);
    0.55 * next_min_n + 0.45 * pre.product_next_flag[c.product][c.op_idx]
}

#[inline(always)]
fn flow_term(pre: &Pre, c: &CandCtx, gain: f64) -> f64 {
    pre.flow_w * pre.job_flow_pref[c.job] * gain
}

#[inline(always)]
fn early_gain(progress: f64) -> f64 {
    0.65 + 0.70 * (1.0 - progress)
}

#[inline(always)]
fn next_w_base(progress: f64) -> f64 {
    0.12 + progress * progress * 0.28
}

#[inline]
fn score_critical_path(pre: &Pre, c: &CandCtx) -> f64 {
    let slack_u = slack_urgency(pre, c.target_mk, c.time, c.product, c.op_idx);
    let slack_w = pre.slack_base * (0.25 + 0.75 * c.progress);
    let slack_reg_boost = 1.0 + 0.40 * pre.reg_n * c.progress;
    let next_term = next_w_base(c.progress) * 0.30 * next_term_raw(pre, c);
    let slack_term = slack_w * slack_u * slack_reg_boost;
    1.03 * rem_min_n(pre, c) + 0.10 * ops_n(pre, c) + 0.24 + 0.20 * FLEX_FACTOR + next_term + 0.10 * slack_term
        - 0.70 * end_n(pre, c) + 0.45 * c.job_bias + flow_term(pre, c, early_gain(c.progress)) + c.jitter
}

#[inline]
fn score_most_work(pre: &Pre, c: &CandCtx) -> f64 {
    let next_term = next_w_base(c.progress) * 0.25 * next_term_raw(pre, c);
    1.00 * rem_avg_n(pre, c) + 0.12 * ops_n(pre, c) + 0.18 + 0.15 * FLEX_FACTOR + next_term
        - 0.62 * end_n(pre, c) + 0.45 * c.job_bias + flow_term(pre, c, early_gain(c.progress)) + c.jitter
}

#[inline]
fn score_early_end(pre: &Pre, c: &CandCtx) -> f64 {
    let next_term = next_w_base(c.progress) * 0.20 * next_term_raw(pre, c);
    1.00 + 0.28 * rem_min_n(pre, c) + 0.22 + next_term
        - 0.55 * end_n(pre, c) + 0.35 * c.job_bias + flow_term(pre, c, early_gain(c.progress)) + c.jitter
}

#[inline]
fn score_shortest_proc(pre: &Pre, c: &CandCtx) -> f64 {
    let next_term = next_w_base(c.progress) * 0.20 * next_term_raw(pre, c);
    -1.00 * proc_n(pre, c) + 0.25 * rem_min_n(pre, c) + 0.12 + next_term
        - 0.20 * end_n(pre, c) + 0.25 * c.job_bias + flow_term(pre, c, early_gain(c.progress)) + c.jitter
}

#[inline]
fn score_regret(pre: &Pre, c: &CandCtx) -> f64 {
    let next_term = next_w_base(c.progress) * 0.25 * next_term_raw(pre, c);
    1.05 * pre.reg_n + 0.55 * rem_min_n(pre, c) + 0.22 + next_term
        - 0.68 * end_n(pre, c) + 0.35 * c.job_bias + flow_term(pre, c, early_gain(c.progress)) + c.jitter
}

#[inline]
fn score_end_tight(pre: &Pre, c: &CandCtx) -> f64 {
    let js = pre.jobshopness;
    let mpen = c.machine_penalty.clamp(0.0, 1.0);
    let phase = progress_phase(c.progress);
    let (cp_w, phase_reg_w, next_w, end_w, flow_gain, slack_w, proc_w, flex_w, scarcity_w, mpen_w) = match phase {
        0 => (1.00 + 0.18 * js, 0.32 + 0.28 * js, 0.17 * (0.80 + 0.40 * js), 0.92, 1.24, 0.0, 0.18, 0.24 * FLEX_FACTOR, 0.14, 0.08 * FLEX_FACTOR),
        1 => (1.15 + 0.30 * js, 0.50 + 0.40 * js, 0.24 * (0.90 + 0.50 * js), 1.18, 0.96, pre.slack_base * (0.40 + 0.30 * js), 0.20, 0.30 * FLEX_FACTOR, 0.18, 0.10 * FLEX_FACTOR),
        _ => (1.30 + 0.32 * js, 0.72 + 0.42 * js, 0.18 * (0.70 + 0.45 * js), 1.58, 0.72, pre.slack_base * (0.92 + 0.34 * js), 0.24, 0.34 * FLEX_FACTOR, 0.20, 0.12 * FLEX_FACTOR),
    };
    let slack_term = if slack_w > 0.0 {
        let slack_u = slack_urgency(pre, c.target_mk, c.time, c.product, c.op_idx);
        let reg_boost = if phase == 2 { 0.50 } else { 0.30 };
        slack_w * slack_u * (1.0 + reg_boost * pre.reg_n)
    } else { 0.0 };
    let next_term = next_w * next_term_raw(pre, c);
    cp_w * rem_min_n(pre, c) + 0.12 * rem_avg_n(pre, c) + 0.08 * ops_n(pre, c) + scarcity_w + flex_w
        + (phase_reg_w * FLEX_FACTOR) * pre.reg_n + next_term + slack_term
        - end_w * end_n(pre, c) - proc_w * proc_n(pre, c) - mpen_w * mpen
        + 0.55 * c.job_bias + flow_term(pre, c, flow_gain) + c.jitter
}

#[inline]
fn score_bn_heavy(pre: &Pre, c: &CandCtx) -> f64 {
    let js = pre.jobshopness;
    let bn_n = pre.product_suf_bn[c.product][c.op_idx] / pre.max_job_bn.max(1e-9);
    let load_n = c.dynamic_load / pre.avg_machine_load.max(1e-9);
    let mpen = c.machine_penalty.clamp(0.0, 1.0);
    let phase = progress_phase(c.progress);
    let (bn_w, end_w, phase_reg_w, load_w, density_w, next_w, flow_gain, slack_w, proc_w, mpen_w, scarcity_w) = match phase {
        0 => ((0.44 + 0.26 * js) * pre.bn_focus, 0.58, 0.28 + 0.18 * js, (0.65 + 0.18 * js) * FLEX_FACTOR, 0.12, 0.16 * (0.70 + 0.50 * js), 1.26, 0.0, 0.16, (0.08 + 0.16 * js) * FLEX_FACTOR * 0.95, 0.16),
        1 => ((0.88 + 0.55 * js) * pre.bn_focus, 0.82, 0.60 + 0.25 * js, (0.55 + 0.24 * js) * FLEX_FACTOR, 0.22, 0.23 * (0.70 + 0.65 * js), 0.98, 0.0, 0.18, (0.12 + 0.30 * js) * FLEX_FACTOR * 0.95, 0.18),
        _ => ((0.60 + 0.30 * js) * pre.bn_focus, 1.10, 0.70 + 0.30 * js, (0.18 + 0.08 * js) * FLEX_FACTOR, 0.16, 0.18 * (0.55 + 0.55 * js), 0.70, pre.slack_base * (0.55 + 0.50 * js), 0.20, (0.14 + 0.34 * js) * FLEX_FACTOR, 0.20),
    };
    let slack_term = if slack_w > 0.0 {
        let slack_u = slack_urgency(pre, c.target_mk, c.time, c.product, c.op_idx);
        slack_w * slack_u * (1.0 + 0.45 * pre.reg_n)
    } else { 0.0 };
    let next_term = next_w * next_term_raw(pre, c);
    0.94 * rem_min_n(pre, c) + 0.30 * rem_avg_n(pre, c) + bn_w * bn_n + density_w * density_n(pre, c) + 0.10 * ops_n(pre, c)
        + 0.64 * FLEX_FACTOR + load_w * load_n + (phase_reg_w * FLEX_FACTOR) * pre.reg_n + scarcity_w + next_term + slack_term
        - end_w * end_n(pre, c) - proc_w * proc_n(pre, c) - mpen_w * mpen
        + 0.60 * c.job_bias + flow_term(pre, c, flow_gain) + c.jitter
}

#[inline]
fn score_adaptive(pre: &Pre, c: &CandCtx) -> f64 {
    let js = pre.jobshopness;
    let fl = 1.0 - js;
    let bn_n = pre.product_suf_bn[c.product][c.op_idx] / pre.max_job_bn.max(1e-9);
    let load_n = c.dynamic_load / pre.avg_machine_load.max(1e-9);
    let mpen = c.machine_penalty.clamp(0.0, 1.0);
    let phase = progress_phase(c.progress);
    let (bn_w, end_w, phase_reg_w, load_w, density_w, next_w, flow_gain, slack_w, proc_w, mpen_w, scarcity_w) = match phase {
        0 => ((0.20 + 0.18 * js) * pre.bn_focus, 0.86 * fl + 0.72 * js, 0.24 * fl + 0.46 * js, (0.60 * fl + 0.92 * js) * FLEX_FACTOR, 0.05 * fl + 0.12 * js, 0.16 * (0.65 * fl + 1.20 * js), 1.28, 0.0, 0.18 * fl + 0.14 * js, (0.05 * fl + 0.16 * js) * FLEX_FACTOR, 0.12 * FLEX_FACTOR),
        1 => ((0.48 + 0.42 * js) * pre.bn_focus, 1.00 * fl + 0.92 * js, 0.48 * fl + 0.78 * js, (0.42 * fl + 0.72 * js) * FLEX_FACTOR, 0.08 * fl + 0.20 * js, 0.23 * (0.55 * fl + 1.50 * js), 0.98, 0.0, 0.16 * fl + 0.12 * js, (0.08 * fl + 0.28 * js) * FLEX_FACTOR, 0.18 * FLEX_FACTOR),
        _ => ((0.34 + 0.30 * js) * pre.bn_focus, 1.22 + 0.28 * js, 0.72 * fl + 0.96 * js, (0.18 * fl + 0.32 * js) * FLEX_FACTOR, 0.04 * fl + 0.16 * js, 0.18 * (0.40 * fl + 1.25 * js), 0.72, pre.slack_base * (0.95 + 0.30 * js), 0.12 * fl + 0.10 * js, (0.10 * fl + 0.34 * js) * FLEX_FACTOR, 0.24 * FLEX_FACTOR),
    };
    let slack_term = if slack_w > 0.0 {
        let slack_u = slack_urgency(pre, c.target_mk, c.time, c.product, c.op_idx);
        slack_w * slack_u * (1.0 + 0.55 * pre.reg_n)
    } else { 0.0 };
    let next_term = next_w * next_term_raw(pre, c);
    1.03 * rem_min_n(pre, c) + 0.48 * rem_avg_n(pre, c) + bn_w * bn_n + density_w * density_n(pre, c) + 0.08 * ops_n(pre, c)
        + 0.60 * FLEX_FACTOR + load_w * load_n + (phase_reg_w * FLEX_FACTOR) * pre.reg_n + scarcity_w + next_term + slack_term
        - end_w * end_n(pre, c) - proc_w * proc_n(pre, c) - mpen_w * mpen
        + (0.60 + 0.06 * js) * c.job_bias + flow_term(pre, c, flow_gain) + c.jitter
}

#[inline]
fn score_candidate<const RULE: u8>(pre: &Pre, c: &CandCtx) -> f64 {
    match RULE {
        0 => score_adaptive(pre, c),
        1 => score_bn_heavy(pre, c),
        2 => score_end_tight(pre, c),
        3 => score_critical_path(pre, c),
        4 => score_most_work(pre, c),
        5 => score_early_end(pre, c),
        6 => score_regret(pre, c),
        _ => score_shortest_proc(pre, c),
    }
}

/// Softmax-free bandit over rules: exploit the best makespan per rule, explore rarely tried ones, sharpened late.
fn choose_rule_bandit(rng: &mut SmallRng, rule_best: &[u32], rule_tries: &[u32], global_best: u32, margin: u32, stuck: usize, late_phase: bool) -> Rule {
    let mut best_seen = global_best;
    for &mk in rule_best { if mk < best_seen { best_seen = mk; } }
    let scale = (margin as f64).max(1.0);
    let s = ((stuck as f64) / 140.0).clamp(0.0, 1.0);
    let explore_mix = (0.10 + 0.55 * s).clamp(0.10, 0.65);
    let mut w = [0.0f64; 8];
    for (i, &r) in RULES.iter().enumerate() {
        let idx = rule_idx(r);
        let mk = rule_best[idx];
        let t = rule_tries[idx].max(1) as f64;
        let delta = mk.saturating_sub(best_seen) as f64;
        let exploit = (-delta / scale).exp();
        let explore = (1.0 / t).sqrt();
        let mut ww = (1.0 - explore_mix) * exploit + explore_mix * explore;
        ww = ww.max(1e-6);
        if late_phase { ww = ww.powf(1.18); }
        w[i] = ww;
    }
    let mut sum = 0.0;
    for i in 0..RULES.len() { sum += w[i].max(0.0); }
    if !(sum > 0.0) { return RULES[rng.gen_range(0..RULES.len())]; }
    let mut r = rng.gen::<f64>() * sum;
    for i in 0..RULES.len() { r -= w[i].max(0.0); if r <= 0.0 { return RULES[i]; } }
    RULES[RULES.len() - 1]
}

/// Rank-weighted draw from a score-sorted list; temperature 1 is uniform, 0 favours the head.
#[inline]
fn choose_from_top_weighted_temp(rng: &mut SmallRng, top: &[Cand], temperature: f64) -> Cand {
    if top.len() <= 1 { return top[0]; }
    let flat = temperature.clamp(0.0, 1.0);
    let sharp = 1.0 - flat;
    let n = top.len();
    let mut sum = 0.0;
    for i in 0..n {
        sum += flat + sharp * ((n - i) as f64);
    }
    let mut r = rng.gen::<f64>() * sum;
    for i in 0..n {
        r -= flat + sharp * ((n - i) as f64);
        if r <= 0.0 { return top[i]; }
    }
    top[n - 1]
}

/// Optional per-job bias and per-machine penalty learned from an earlier solution.
type Guidance<'a> = Option<(&'a [f64], &'a [f64])>;

fn construct_solution_conflict(
    challenge: &Challenge, pre: &Pre, rule: Rule, k: usize, target_mk: Option<u32>,
    rng: &mut SmallRng, guidance: Guidance,
) -> Result<(Solution, u32)> {
    macro_rules! dispatch {
        ($g:expr) => {
            match rule {
                Rule::Adaptive => construct_impl::<0, $g>(challenge, pre, k, target_mk, rng, guidance),
                Rule::BnHeavy => construct_impl::<1, $g>(challenge, pre, k, target_mk, rng, guidance),
                Rule::EndTight => construct_impl::<2, $g>(challenge, pre, k, target_mk, rng, guidance),
                Rule::CriticalPath => construct_impl::<3, $g>(challenge, pre, k, target_mk, rng, guidance),
                Rule::MostWork => construct_impl::<4, $g>(challenge, pre, k, target_mk, rng, guidance),
                Rule::EarlyEnd => construct_impl::<5, $g>(challenge, pre, k, target_mk, rng, guidance),
                Rule::Regret => construct_impl::<6, $g>(challenge, pre, k, target_mk, rng, guidance),
                Rule::ShortestProc => construct_impl::<7, $g>(challenge, pre, k, target_mk, rng, guidance),
            }
        };
    }
    if guidance.is_some() { dispatch!(true) } else { dispatch!(false) }
}

/// Event-driven construction. At each time every idle machine scores its ready jobs; the candidates
/// of all machines compete, with a bonus for machines in demand, and the winner (best, or a
/// rank-weighted draw among the top k when k > 0) is dispatched. Repeats until no machine can start.
#[inline]
fn construct_impl<const RULE: u8, const GUIDED: bool>(
    challenge: &Challenge, pre: &Pre, k: usize, target_mk: Option<u32>,
    rng: &mut SmallRng, guidance: Guidance,
) -> Result<(Solution, u32)> {
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;
    let (job_bias, machine_penalty): (&[f64], &[f64]) = guidance.unwrap_or((&[], &[]));
    let mut job_next_op = vec![0usize; num_jobs];
    let mut job_ready_time = vec![0u32; num_jobs];
    let mut machine_avail = vec![0u32; num_machines];
    let mut machine_load = pre.machine_load.clone();
    let mut job_schedule: Vec<Vec<(usize, u32)>> = pre.job_ops_len.iter().map(|&len| Vec::with_capacity(len)).collect();
    let mut remaining_ops = pre.total_ops;
    let mut time = 0u32;
    let mut demand: Vec<u16> = vec![0u16; num_machines];
    let mut raw_by_machine: Vec<Vec<Cand>> = (0..num_machines).map(|_| Vec::with_capacity(12)).collect();
    let mut idle_machines: Vec<usize> = (0..num_machines).collect();
    let mut idle_pos: Vec<usize> = (0..num_machines).collect();
    let mut busy_machine_heap: std::collections::BinaryHeap<std::cmp::Reverse<(u32, usize)>> = std::collections::BinaryHeap::with_capacity(num_machines);
    let mut blocked_job_heap: std::collections::BinaryHeap<std::cmp::Reverse<(u32, usize)>> = std::collections::BinaryHeap::with_capacity(num_jobs);
    let mut touched_machines: Vec<usize> = Vec::with_capacity(num_machines);
    let mut top: Vec<Cand> = if k > 0 { Vec::with_capacity(k) } else { Vec::new() };
    // ready_by_machine[m] holds (job, generation, pt); an entry is stale once the job's generation moved on.
    let mut ready_by_machine: Vec<Vec<(usize, u32, u32)>> = (0..num_machines).map(|_| Vec::with_capacity(32)).collect();
    let mut job_gen: Vec<u32> = vec![0u32; num_jobs];
    // The conflict bonus of a single-machine operation only depends on the instance.
    let rigidity = (0.60 + 0.40_f64).clamp(0.0, 2.5);
    let regc = pre.reg_n.clamp(0.0, 4.5);
    let conflict_scale = (0.90 + 0.40 * FLEX_FACTOR).clamp(0.85, 1.75);
    for job in 0..num_jobs {
        if pre.job_ops_len[job] == 0 { continue; }
        let op = pre.product_ops[pre.job_products[job]][0];
        ready_by_machine[op.machine].push((job, job_gen[job], op.pt));
    }
    while remaining_ops > 0 {
        loop {
            while let Some(entry) = busy_machine_heap.peek() {
                let std::cmp::Reverse((t, m)) = *entry;
                if t > time { break; }
                busy_machine_heap.pop();
                if machine_avail[m] == t && idle_pos[m] == NONE_USIZE {
                    idle_pos[m] = idle_machines.len();
                    idle_machines.push(m);
                }
            }
            while let Some(entry) = blocked_job_heap.peek() {
                let std::cmp::Reverse((t, j)) = *entry;
                if t > time { break; }
                blocked_job_heap.pop();
                if job_next_op[j] >= pre.job_ops_len[j] || job_ready_time[j] != t { continue; }
                let op = pre.product_ops[pre.job_products[j]][job_next_op[j]];
                ready_by_machine[op.machine].push((j, job_gen[j], op.pt));
            }
            if idle_machines.is_empty() { break; }
            touched_machines.clear();
            let progress = 1.0 - (remaining_ops as f64) / (pre.total_ops as f64).max(1.0);
            let cap_per_machine = if k == 0 { 12usize } else { (k + 6).min(12) };
            for &m in &idle_machines {
                demand[m] = 0;
                raw_by_machine[m].clear();
                let list = &mut ready_by_machine[m];
                let mut write = 0usize;
                for read in 0..list.len() {
                    let (job, gen, pt) = list[read];
                    if job_gen[job] != gen { continue; }
                    let op_idx = job_next_op[job];
                    if op_idx >= pre.job_ops_len[job] || job_ready_time[job] > time { continue; }
                    list[write] = (job, gen, pt);
                    write += 1;
                    if demand[m] == 0 { touched_machines.push(m); }
                    demand[m] = demand[m].saturating_add(1);
                    let product = pre.job_products[job];
                    let ctx = CandCtx {
                        job,
                        product,
                        op_idx,
                        ops_rem: pre.job_ops_len[job] - op_idx,
                        pt,
                        time,
                        target_mk,
                        end: time.saturating_add(pt),
                        progress,
                        job_bias: if GUIDED { job_bias[job] } else { 0.0 },
                        machine_penalty: if GUIDED { machine_penalty[m] } else { 0.0 },
                        dynamic_load: machine_load[m],
                        jitter: if k > 0 { rng.gen::<f64>() * 1e-9 } else { 0.0 },
                    };
                    let base = score_candidate::<RULE>(pre, &ctx);
                    if raw_by_machine[m].len() < cap_per_machine || base >= raw_by_machine[m][cap_per_machine - 1].score {
                        push_top_k(&mut raw_by_machine[m], Cand { job, machine: m, pt, score: base }, cap_per_machine);
                    }
                }
                list.truncate(write);
            }
            touched_machines.sort_unstable();
            let denom = (idle_machines.len() as f64).max(1.0);
            let conflict_w = (0.09 + 0.26 * pre.jobshopness + 0.16 * (1.0 - progress)).clamp(0.05, 0.45);
            let mut best: Option<Cand> = None;
            top.clear();
            let mut demand_excess_sum = 0u32;
            let mut max_demand = 1u16;
            for &m in &touched_machines {
                let dem_u = demand[m];
                if dem_u > max_demand { max_demand = dem_u; }
                demand_excess_sum = demand_excess_sum.saturating_add(dem_u.saturating_sub(1) as u32);
                let dem = dem_u as f64;
                if dem <= 0.0 || raw_by_machine[m].is_empty() { continue; }
                let dem_n = ((dem - 1.0) / denom).clamp(0.0, 2.5);
                for rc in &raw_by_machine[m] {
                    let boost = conflict_w * conflict_scale * dem_n * (1.15 * rigidity + 0.85 * regc);
                    let c = Cand { job: rc.job, machine: rc.machine, pt: rc.pt, score: rc.score + boost };
                    if k == 0 {
                        if best.map_or(true, |bb| c.score > bb.score) { best = Some(c); }
                    } else if top.len() < k || c.score >= top[k - 1].score {
                        push_top_k(&mut top, c, k);
                    }
                }
            }
            let select_temp = if k == 0 || top.len() <= 1 { 0.0 } else {
                let conflict_avg = ((demand_excess_sum as f64) / denom).clamp(0.0, 3.0) / 3.0;
                let conflict_peak = ((max_demand.saturating_sub(1)) as f64).clamp(0.0, 4.0) / 4.0;
                ((1.0 - progress) * (0.22 + 0.58 * conflict_avg + 0.20 * conflict_peak)).clamp(0.0, 0.92)
            };
            let chosen = if k == 0 {
                match best { Some(c) => c, None => break }
            } else {
                if top.is_empty() { break; }
                choose_from_top_weighted_temp(rng, &top, select_temp)
            };
            let job = chosen.job;
            let machine = chosen.machine;
            let pt = chosen.pt;
            let op = pre.product_ops[pre.job_products[job]][job_next_op[job]];
            // Idle machines are exactly those with machine_avail <= time.
            debug_assert!(machine_avail[machine] <= time && op.machine == machine);
            let end_time = time.saturating_add(pt);
            job_schedule[job].push((machine, time));
            job_next_op[job] += 1;
            job_ready_time[job] = end_time;
            machine_avail[machine] = end_time;
            remaining_ops -= 1;
            job_gen[job] = job_gen[job].wrapping_add(1);
            let pos = idle_pos[machine];
            if pos != NONE_USIZE {
                let last = idle_machines.pop().unwrap();
                if pos < idle_machines.len() {
                    idle_machines[pos] = last;
                    idle_pos[last] = pos;
                }
                idle_pos[machine] = NONE_USIZE;
            }
            busy_machine_heap.push(std::cmp::Reverse((end_time, machine)));
            if job_next_op[job] < pre.job_ops_len[job] {
                blocked_job_heap.push(std::cmp::Reverse((end_time, job)));
            }
            let v = machine_load[op.machine] - op.pt as f64;
            machine_load[op.machine] = if v > 0.0 { v } else { 0.0 };
            if remaining_ops == 0 { break; }
        }
        if remaining_ops == 0 { break; }
        let next_machine_time = loop {
            match busy_machine_heap.peek() {
                Some(entry) => {
                    let std::cmp::Reverse((t, m)) = *entry;
                    if machine_avail[m] != t || t <= time {
                        busy_machine_heap.pop();
                        continue;
                    }
                    break Some(t);
                }
                None => break None,
            }
        };
        let next_job_time = loop {
            match blocked_job_heap.peek() {
                Some(entry) => {
                    let std::cmp::Reverse((t, j)) = *entry;
                    if job_next_op[j] >= pre.job_ops_len[j] || job_ready_time[j] != t || t <= time {
                        blocked_job_heap.pop();
                        continue;
                    }
                    break Some(t);
                }
                None => break None,
            }
        };
        time = match (next_machine_time, next_job_time) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => return Err(anyhow!("Stalled")),
        };
    }
    let mk = machine_avail.into_iter().max().unwrap_or(0);
    Ok((Solution { job_schedule }, mk))
}

#[inline]
fn rebuild_machine_pred_nodes(ds: &DisjSchedule, machine_pred_node: &mut [usize]) {
    machine_pred_node.fill(NONE_USIZE);
    for seq in &ds.machine_seq {
        for i in 1..seq.len() {
            machine_pred_node[seq[i]] = seq[i - 1];
        }
    }
}

fn rebuild_machine_succ_indeg(ds: &DisjSchedule, machine_succ: &mut [usize], indeg_full: &mut [u16]) {
    indeg_full.copy_from_slice(&ds.indeg_job_plus1);
    for seq in &ds.machine_seq {
        if seq.is_empty() { continue; }
        indeg_full[seq[0]] -= 1;
        let mut prev = seq[0];
        for &v in &seq[1..] {
            machine_succ[prev] = v;
            prev = v;
        }
        machine_succ[prev] = NONE_USIZE;
    }
}

#[inline]
fn u2n(x: usize) -> u32 { if x == NONE_USIZE { NONE_U32 } else { x as u32 } }

#[inline]
fn n2u(x: u32) -> usize { if x == NONE_U32 { NONE_USIZE } else { x as usize } }

type HGraph = super::super::hybrid_engine::graph::Graph;
type HModel = super::super::hybrid_engine::model::Model;

fn hg_load(hg: &mut HGraph, hm: &HModel, ds: &DisjSchedule) -> bool {
    let mut sched: Vec<Vec<(usize, u32)>> = (0..ds.num_jobs)
        .map(|j| vec![(0usize, 0u32); ds.job_offsets[j + 1] - ds.job_offsets[j]])
        .collect();
    for m in 0..ds.num_machines {
        for (i, &o) in ds.machine_seq[m].iter().enumerate() {
            sched[ds.node_job[o]][ds.node_op[o]] = (m, i as u32);
        }
    }
    hg.load_schedule(hm, &sched)
}

/// Evaluates heads and tails; returns (makespan, first sink reaching it, whether it is the only one).
#[inline]
fn hg_eval_nd(hg: &mut HGraph, node_pt: &[u32], sinks: &[usize]) -> Option<(u32, usize, bool)> {
    let mk = hg.evaluate()?;
    hg.compute_tails();
    let mut mk_node = NONE_USIZE;
    let mut single = true;
    for &u in sinks {
        if hg.head[u].saturating_add(node_pt[u]) == mk {
            if mk_node == NONE_USIZE { mk_node = u; } else { single = false; }
        }
    }
    Some((mk, mk_node, single))
}

/// With positive durations only a last operation of a job can end at the makespan.
#[inline]
fn hg_sinks(ds: &DisjSchedule) -> Vec<usize> {
    (0..ds.n).filter(|&u| ds.job_succ[u] == NONE_USIZE).collect()
}

/// Copies the static links of the disjunctive graph into the per-operation records.
fn nd_sync(ds: &DisjSchedule, buf: &mut EvalBuf, machine_pred_node: &[usize]) {
    for u in 0..ds.n {
        let r = &mut buf.nd[u];
        r.pt = ds.node_pt[u];
        r.job_succ = u2n(ds.job_succ[u]);
        r.machine_succ = u2n(buf.machine_succ[u]);
        r.best_pred = u2n(buf.best_pred[u]);
        r.machine_pred = u2n(machine_pred_node[u]);
    }
}

/// Tight predecessor of u (route first, then machine), or NONE_USIZE at a root.
/// SAFETY: u and every non-sentinel link are operation ids below nd.len() == head.len() == job_pred.len().
#[inline]
fn bp_of(nd: &[NodeRec], job_pred: &[usize], u: usize, head: &[u32]) -> usize {
    let h = unsafe { *head.get_unchecked(u) };
    if h == 0 { return NONE_USIZE; }
    let jp = unsafe { *job_pred.get_unchecked(u) };
    if jp != NONE_USIZE && unsafe { *head.get_unchecked(jp) }.saturating_add(unsafe { nd.get_unchecked(jp) }.pt) == h { return jp; }
    let mp = unsafe { nd.get_unchecked(u) }.machine_pred;
    if mp != NONE_U32 {
        let mp = mp as usize;
        if unsafe { *head.get_unchecked(mp) }.saturating_add(unsafe { nd.get_unchecked(mp) }.pt) == h { return mp; }
    }
    NONE_USIZE
}

/// Decides whether the critical path from mk_node is unambiguous. When both predecessors of some
/// node are tight, or when several sinks reach the makespan, best_pred is settled by a full pass.
/// Returns (path end, whether a full pass filled nd.best_pred).
/// SAFETY: mk_node and every non-sentinel predecessor are operation ids below nd.len() == head.len() == job_pred.len().
#[inline]
fn settle_path(ds: &DisjSchedule, buf: &mut EvalBuf, indeg_full: &[u16], job_pred: &[usize], mk_node: usize, single: bool, head: &[u32], mk: u32) -> Option<(usize, bool)> {
    let mut clear = single;
    if clear {
        let nd = &buf.nd;
        let rend = |i: usize| -> u32 { unsafe { *head.get_unchecked(i) }.saturating_add(unsafe { nd.get_unchecked(i) }.pt) };
        let mut u = mk_node;
        while u != NONE_USIZE {
            let h = unsafe { *head.get_unchecked(u) };
            if h == 0 { break; }
            let jp = unsafe { *job_pred.get_unchecked(u) };
            let jt = jp != NONE_USIZE && rend(jp) == h;
            let mp = unsafe { nd.get_unchecked(u) }.machine_pred;
            let mt = mp != NONE_U32 && rend(mp as usize) == h;
            if jt && mt { clear = false; break; }
            u = if jt { jp } else if mt { mp as usize } else { NONE_USIZE };
        }
    }
    if clear { return Some((mk_node, false)); }
    let node = settle_topo_nd(ds, buf, indeg_full, head, mk)?;
    Some((node, true))
}

/// Topological pass over the records: best_pred[v] is the first tight predecessor popped, the
/// path end is the first popped operation ending at mk. Roots are pushed job by job, route
/// successors are pushed before machine successors.
/// SAFETY: nd, indeg_full, head and stack have n slots; links are operation ids or sentinels; an
/// operation is pushed once when its in-degree reaches zero, so the stack holds at most n - popped
/// entries and the unconditional write below targets the next free slot.
fn settle_topo_nd(ds: &DisjSchedule, buf: &mut EvalBuf, indeg_full: &[u16], head: &[u32], mk: u32) -> Option<usize> {
    let n = ds.n;
    let nd = &mut buf.nd;
    for u in 0..n {
        let r = unsafe { nd.get_unchecked_mut(u) };
        r.indeg = unsafe { *indeg_full.get_unchecked(u) } as u32;
        r.best_pred = NONE_U32;
    }
    let mut sl = 0usize;
    for j in 0..ds.num_jobs {
        let h = ds.job_offsets[j];
        if h < ds.job_offsets[j + 1] && unsafe { nd.get_unchecked(h) }.indeg == 0 {
            unsafe { *buf.stack.get_unchecked_mut(sl) = h; }
            sl += 1;
        }
    }
    let mut popped = 0usize;
    let mut mk_node = NONE_USIZE;
    while sl > 0 {
        sl -= 1;
        let u = unsafe { *buf.stack.get_unchecked(sl) };
        popped += 1;
        let ru = unsafe { *nd.get_unchecked(u) };
        let end_u = unsafe { *head.get_unchecked(u) }.saturating_add(ru.pt);
        mk_node = if end_u == mk && mk_node == NONE_USIZE { u } else { mk_node };
        if ru.job_succ != NONE_U32 { settle_relax(nd, head, &mut buf.stack, &mut sl, u, end_u, ru.job_succ as usize); }
        if ru.machine_succ != NONE_U32 { settle_relax(nd, head, &mut buf.stack, &mut sl, u, end_u, ru.machine_succ as usize); }
    }
    if popped != n { return None; }
    Some(mk_node)
}

/// One arc u -> s of settle_topo_nd: records the first tight predecessor, decrements the in-degree, pushes s when it reaches zero.
#[inline(always)]
fn settle_relax(nd: &mut [NodeRec], head: &[u32], stack: &mut [usize], sl: &mut usize, u: usize, end_u: u32, s: usize) {
    let r = unsafe { nd.get_unchecked_mut(s) };
    let tight = end_u == unsafe { *head.get_unchecked(s) } && r.best_pred == NONE_U32;
    r.best_pred = if tight { u as u32 } else { r.best_pred };
    r.indeg -= 1;
    unsafe { *stack.get_unchecked_mut(*sl) = s; }
    *sl += (r.indeg == 0) as usize;
}

/// Makespan estimate after swapping adjacent u, v on their machine (heads before, tails after).
/// SAFETY: u, v and their non-sentinel links are operation ids below the lengths of nd, head, qend and job_pred.
#[inline]
fn estimate_swap_nd(u: usize, v: usize, nd: &[NodeRec], job_pred: &[usize], head: &[u32], qend: &[u32]) -> u32 {
    let rend = |i: usize| -> u32 { unsafe { *head.get_unchecked(i) }.saturating_add(unsafe { nd.get_unchecked(i) }.pt) };
    let q_at = |i: usize| -> u32 { unsafe { *qend.get_unchecked(i) } };
    let ru = unsafe { *nd.get_unchecked(u) };
    let rv = unsafe { *nd.get_unchecked(v) };
    let jp_u = unsafe { *job_pred.get_unchecked(u) };
    let jp_v = unsafe { *job_pred.get_unchecked(v) };
    let r_jp_v = if jp_v != NONE_USIZE { rend(jp_v) } else { 0 };
    let r_mp_u = if ru.machine_pred != NONE_U32 { rend(ru.machine_pred as usize) } else { 0 };
    let new_r_v = r_jp_v.max(r_mp_u);
    let r_jp_u = if jp_u != NONE_USIZE { rend(jp_u) } else { 0 };
    let new_r_u = r_jp_u.max(new_r_v.saturating_add(rv.pt));
    let q_js_u = if ru.job_succ != NONE_U32 { q_at(ru.job_succ as usize) } else { 0 };
    let q_ms_v = if rv.machine_succ != NONE_U32 { q_at(rv.machine_succ as usize) } else { 0 };
    let new_q_u = q_js_u.max(q_ms_v);
    let q_js_v = if rv.job_succ != NONE_U32 { q_at(rv.job_succ as usize) } else { 0 };
    let new_q_v = q_js_v.max(ru.pt.saturating_add(new_q_u));
    let path_v = new_r_v.saturating_add(rv.pt).saturating_add(new_q_v);
    let path_u = new_r_u.saturating_add(ru.pt).saturating_add(new_q_u);
    path_v.max(path_u)
}

/// Same estimate from plain arrays (heads in `heads`, tails in `qend`).
#[inline]
fn estimate_swap_mk(u: usize, v: usize, heads: &[u32], qend: &[u32], pt: &[u32], job_pred: &[usize], job_succ: &[usize], machine_pred: &[usize], machine_succ: &[usize]) -> u32 {
    let mp_u = machine_pred[u];
    let ms_v = machine_succ[v];
    let jp_v = job_pred[v];
    let jp_u = job_pred[u];
    let js_u = job_succ[u];
    let js_v = job_succ[v];
    let r_jp_v = if jp_v != NONE_USIZE { heads[jp_v].saturating_add(pt[jp_v]) } else { 0 };
    let r_mp_u = if mp_u != NONE_USIZE { heads[mp_u].saturating_add(pt[mp_u]) } else { 0 };
    let new_r_v = r_jp_v.max(r_mp_u);
    let r_jp_u = if jp_u != NONE_USIZE { heads[jp_u].saturating_add(pt[jp_u]) } else { 0 };
    let new_r_u = r_jp_u.max(new_r_v.saturating_add(pt[v]));
    let q_js_u = if js_u != NONE_USIZE { qend[js_u] } else { 0 };
    let q_ms_v = if ms_v != NONE_USIZE { qend[ms_v] } else { 0 };
    let new_q_u = q_js_u.max(q_ms_v);
    let q_js_v = if js_v != NONE_USIZE { qend[js_v] } else { 0 };
    let new_q_v = q_js_v.max(pt[u].saturating_add(new_q_u));
    let path_v = new_r_v.saturating_add(pt[v]).saturating_add(new_q_v);
    let path_u = new_r_u.saturating_add(pt[u]).saturating_add(new_q_u);
    path_v.max(path_u)
}

/// Fills qend[o] = tail[o] + pt[o] by a reverse topological pass from the current machine links.
fn compute_qend(ds: &DisjSchedule, machine_succ: &[usize], machine_pred_node: &[usize], job_pred_node: &[usize], qend: &mut [u32], back_deg: &mut [u16], stack: &mut Vec<usize>) {
    let n = ds.n;
    qend.copy_from_slice(&ds.node_pt);
    back_deg.fill(0);
    for i in 0..n {
        if ds.job_succ[i] != NONE_USIZE { back_deg[i] += 1; }
        if machine_succ[i] != NONE_USIZE { back_deg[i] += 1; }
    }
    stack.clear();
    for i in 0..n { if back_deg[i] == 0 { stack.push(i); } }
    while let Some(nd) = stack.pop() {
        let c = qend[nd];
        for p in [job_pred_node[nd], machine_pred_node[nd]] {
            if p == NONE_USIZE { continue; }
            let v = c.saturating_add(ds.node_pt[p]);
            if v > qend[p] { qend[p] = v; }
            back_deg[p] = back_deg[p].saturating_sub(1);
            if back_deg[p] == 0 { stack.push(p); }
        }
    }
}

#[inline]
fn rebuild_machine_pos_map(machine_seq: &[Vec<usize>], node_pos: &mut [usize]) {
    for seq in machine_seq {
        for (pos, &node) in seq.iter().enumerate() {
            node_pos[node] = pos;
        }
    }
}

/// One adjacent swap per machine along the critical path that moves an operation towards its position in ref_pos.
#[inline]
fn collect_guided_kick_moves(
    ds: &DisjSchedule,
    best_pred: &[usize],
    mk_node: usize,
    current_pos: &[usize],
    node_machine: &[usize],
    ref_pos: &[usize],
    used_machine: &mut [u8],
    out: &mut Vec<(u32, usize, usize)>,
) {
    out.clear();
    used_machine.fill(0);
    let mut u = mk_node;
    while u != NONE_USIZE {
        let m = node_machine[u];
        if used_machine[m] == 0 {
            let pos = current_pos[u];
            let target = ref_pos[u];
            let seq = &ds.machine_seq[m];
            if pos > target && pos > 0 {
                let left = seq[pos - 1];
                if ref_pos[u] < ref_pos[left] {
                    out.push(((pos - target) as u32, m, pos - 1));
                    used_machine[m] = 1;
                }
            } else if pos < target && pos + 1 < seq.len() {
                let right = seq[pos + 1];
                if ref_pos[u] > ref_pos[right] {
                    out.push(((target - pos) as u32, m, pos));
                    used_machine[m] = 1;
                }
            }
        }
        u = best_pred[u];
    }
}

/// Machine sequences of a solution as one id vector with a separator per machine.
fn ts_seed_key(pre: &Pre, num_machines: usize, sol: &Solution) -> Vec<u32> {
    let num_jobs = sol.job_schedule.len();
    if num_jobs != pre.job_ops_len.len() { return Vec::new(); }
    let mut job_offsets = vec![0usize; num_jobs + 1];
    for j in 0..num_jobs { job_offsets[j + 1] = job_offsets[j] + pre.job_ops_len[j]; }
    let mut per_machine: Vec<Vec<(u32, usize)>> = vec![Vec::new(); num_machines];
    for j in 0..num_jobs {
        if sol.job_schedule[j].len() != pre.job_ops_len[j] { return Vec::new(); }
        for (k, &(m, st)) in sol.job_schedule[j].iter().enumerate() {
            if m >= num_machines { return Vec::new(); }
            per_machine[m].push((st, job_offsets[j] + k));
        }
    }
    let mut key: Vec<u32> = Vec::with_capacity(job_offsets[num_jobs] + num_machines);
    for m in 0..num_machines {
        per_machine[m].sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        for &(_, id) in &per_machine[m] { key.push(id as u32); }
        key.push(u32::MAX);
    }
    key
}

#[inline]
fn ts_seed_key_distance(a: &[u32], b: &[u32]) -> u32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() { return 0; }
    let mut d = 0u32;
    for i in 0..a.len() {
        if a[i] != b[i] { d += 1; }
    }
    d
}

/// Walks from start towards target by the adjacent swap with the best estimate, one inversion at a
/// time, and returns the best schedule met in the middle 60% of the walk.
fn path_relink_midpoint(pre: &Pre, challenge: &Challenge, start: &Solution, target: &Solution) -> Result<Option<(Solution, u32)>> {
    let mut cur = build_disj_from_solution(pre, challenge, start)?;
    let tgt = build_disj_from_solution(pre, challenge, target)?;
    let n = cur.n;
    let mut buf = EvalBuf::new(n);
    let mut target_pos = vec![0usize; n];
    for m in 0..tgt.num_machines {
        for (pos, &op) in tgt.machine_seq[m].iter().enumerate() { target_pos[op] = pos; }
    }
    let mut job_pred_node = vec![NONE_USIZE; n];
    for j in 0..cur.num_jobs { let b = cur.job_offsets[j]; let e = cur.job_offsets[j + 1]; for k in (b + 1)..e { job_pred_node[k] = k - 1; } }
    let mut machine_pred_node = vec![NONE_USIZE; n];
    let mut qend = vec![0u32; n];
    let mut back_deg = vec![0u16; n];
    let mut stack: Vec<usize> = Vec::with_capacity(n);
    let mut dist = 0usize;
    for m in 0..cur.num_machines {
        let seq = &cur.machine_seq[m];
        for i in 0..seq.len() { for k in (i + 1)..seq.len() { if target_pos[seq[i]] > target_pos[seq[k]] { dist += 1; } } }
    }
    if dist < 10 { return Ok(None); }
    let lo = dist / 5;
    let hi = dist - dist / 5;
    let mut best: Option<(Solution, u32)> = None;
    for step in 0..=dist {
        let Some((mk, _)) = eval_disj(&cur, &mut buf) else { return Ok(best) };
        if step >= lo && step <= hi {
            if best.as_ref().map_or(true, |(_, bmk)| mk < *bmk) {
                best = Some((disj_to_solution(pre, &cur, &buf.start), mk));
            }
        }
        if step == dist { break; }
        rebuild_machine_pred_nodes(&cur, &mut machine_pred_node);
        compute_qend(&cur, &buf.machine_succ, &machine_pred_node, &job_pred_node, &mut qend, &mut back_deg, &mut stack);
        let mut best_move: Option<(usize, usize)> = None;
        let mut best_est = u32::MAX;
        for m in 0..cur.num_machines {
            let seq = &cur.machine_seq[m];
            for i in 0..seq.len().saturating_sub(1) {
                let (a, b) = (seq[i], seq[i + 1]);
                if target_pos[a] > target_pos[b] {
                    let est = estimate_swap_mk(a, b, &buf.start, &qend, &cur.node_pt, &job_pred_node, &cur.job_succ, &machine_pred_node, &buf.machine_succ);
                    if est < best_est { best_est = est; best_move = Some((m, i)); }
                }
            }
        }
        let Some((m, i)) = best_move else { break };
        cur.machine_seq[m].swap(i, i + 1);
    }
    Ok(best)
}

/// Random acceptance of the tie_n-th equal candidate with probability 1 / tie_n.
#[inline(always)]
fn tie_break(tie_n: &mut u64, tseed: &mut u64) -> bool {
    *tie_n = tie_n.saturating_add(1);
    *tseed ^= tseed.wrapping_shl(13);
    *tseed ^= tseed.wrapping_shr(7);
    *tseed ^= tseed.wrapping_shl(17);
    (*tseed % (*tie_n).max(1)) == 0
}

/// Best admissible move and best move overall; each is (machine, position, estimated makespan).
struct MoveSel {
    best: Option<(usize, usize, u32)>,
    best_key: u32,
    best_tie_n: u64,
    relaxed: Option<(usize, usize, u32)>,
    relaxed_key: u32,
    relaxed_tie_n: u64,
}

impl MoveSel {
    fn new() -> Self {
        Self { best: None, best_key: u32::MAX, best_tie_n: 0, relaxed: None, relaxed_key: u32::MAX, relaxed_tie_n: 0 }
    }

    #[inline(always)]
    fn offer(&mut self, m: usize, pos: usize, est_mk: u32, adj_mk: u32, admissible: bool, tseed: &mut u64) {
        if admissible {
            if adj_mk < self.best_key { self.best_key = adj_mk; self.best = Some((m, pos, est_mk)); self.best_tie_n = 1; }
            else if adj_mk == self.best_key && tie_break(&mut self.best_tie_n, tseed) { self.best = Some((m, pos, est_mk)); }
        }
        if adj_mk < self.relaxed_key { self.relaxed_key = adj_mk; self.relaxed = Some((m, pos, est_mk)); self.relaxed_tie_n = 1; }
        else if adj_mk == self.relaxed_key && tie_break(&mut self.relaxed_tie_n, tseed) { self.relaxed = Some((m, pos, est_mk)); }
    }
}

/// Calls f(first, last) for each maximal run of consecutive tight positions (last > first).
#[inline(always)]
fn for_each_block(positions: &[usize], seq: &[usize], head: &[u32], node_pt: &[u32], mut f: impl FnMut(usize, usize)) {
    let mut run_start = positions[0];
    let mut run_end = positions[0];
    let mut prev_pos = positions[0];
    let mut prev_node = seq[prev_pos];
    for &pos in &positions[1..] {
        let node = seq[pos];
        if pos == prev_pos + 1 && head[node] == head[prev_node].saturating_add(node_pt[prev_node]) {
            run_end = pos;
        } else {
            if run_end > run_start { f(run_start, run_end); }
            run_start = pos;
            run_end = pos;
        }
        prev_pos = pos;
        prev_node = node;
    }
    if run_end > run_start { f(run_start, run_end); }
}

/// Per-operation position bookkeeping of the tabu search.
struct TabuMirror {
    current_pos: Vec<usize>,
    node_machine: Vec<usize>,
    machine_pred_node: Vec<usize>,
    indeg_full: Vec<u16>,
}

/// Rebuilds every derived structure from ds.machine_seq and evaluates the graph.
fn reload_mirror(ds: &DisjSchedule, buf: &mut EvalBuf, tm: &mut TabuMirror, hg: &mut HGraph, hm: &HModel, sinks: &[usize]) -> Option<(u32, usize, bool)> {
    rebuild_machine_pred_nodes(ds, &mut tm.machine_pred_node);
    for m in 0..ds.num_machines {
        for (i, &node) in ds.machine_seq[m].iter().enumerate() {
            tm.current_pos[node] = i;
            tm.node_machine[node] = m;
        }
    }
    rebuild_machine_succ_indeg(ds, &mut buf.machine_succ, &mut tm.indeg_full);
    nd_sync(ds, buf, &tm.machine_pred_node);
    if !hg_load(hg, hm, ds) { return None; }
    hg_eval_nd(hg, &ds.node_pt, sinks)
}

/// Tabu search over adjacent swaps at the ends of critical blocks (N5), with guided kicks after
/// stalls, restarts from the best schedule, and a diversification bonus late in a stall.
/// Records better than published_mk are published when checked at the start of an ordinary
/// iteration. Returns None when the base schedule is not improved.
fn tabu_search_phase(
    pre: &Pre, challenge: &Challenge, base_sol: &Solution, max_iterations: usize, tenure: usize,
    kick_spread: usize, traj_salt: u64, restore_after_failed_kick: bool,
    publish: &dyn Fn(&Solution) -> Result<()>, published_mk: u32,
) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    if ds.node_pt.iter().any(|&p| p == 0) { return Ok(None); }
    let n = ds.n;
    let mut buf = EvalBuf::new(n);
    let Some((initial_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let mut best_global_mk = initial_mk;
    let mut best_global_machine_seq = ds.machine_seq.clone();
    let tenure_delta = (tenure / 3).max(2);
    let max_no_improve = (max_iterations / 2).max(60);
    let mut pair_offsets = vec![0usize; ds.num_machines + 1];
    for m in 0..ds.num_machines {
        let len = ds.machine_seq[m].len();
        pair_offsets[m + 1] = pair_offsets[m] + len.saturating_mul(len.saturating_sub(1)) / 2;
    }
    // Tabu status is attached to the unordered pair of base positions of the two swapped operations.
    let mut tabu_expiry = vec![0usize; pair_offsets[ds.num_machines]];
    let mut no_improve = 0usize;
    let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (initial_mk as u64).wrapping_shl(16) ^ (n as u64).wrapping_mul(0x517CC1B727220A95) ^ traj_salt.wrapping_mul(0x2545F4914F6CDD1D);
    let mut tseed: u64 = ((challenge.seed[0] as u64).wrapping_mul(0xD1B54A32D192ED03)
        ^ (initial_mk as u64).wrapping_mul(0xA24BAED4963EE407)
        ^ (n as u64).wrapping_shl(32)) | 1;
    let mut job_pred_node = vec![NONE_USIZE; n];
    for j in 0..ds.num_jobs { let base = ds.job_offsets[j]; let end = ds.job_offsets[j + 1]; for k in (base + 1)..end { job_pred_node[k] = k - 1; } }
    let crit_ml = ds.machine_seq.iter().map(|s| s.len()).max().unwrap_or(0).max(1);
    // Critical positions per machine, flat: crit_pos_flat[m * crit_ml .. m * crit_ml + crit_cnt[m]].
    let mut crit_pos_flat = vec![0usize; ds.num_machines * crit_ml];
    let mut crit_cnt = vec![0usize; ds.num_machines];
    let mut machine_last_pick = vec![0usize; ds.num_machines];
    let mut guided_kicks: Vec<(u32, usize, usize)> = Vec::with_capacity(ds.num_machines);
    let mut guided_used_machine = vec![0u8; ds.num_machines];
    let mut tm = TabuMirror {
        current_pos: vec![0usize; n],
        node_machine: vec![0usize; n],
        machine_pred_node: vec![NONE_USIZE; n],
        indeg_full: vec![0u16; n],
    };
    let hm = HModel::build(challenge);
    let mut hg = HGraph::new(&hm);
    let sinks = hg_sinks(&ds);
    let Some((mut cur_mk, mut mk_node, mut mk_single)) = reload_mirror(&ds, &mut buf, &mut tm, &mut hg, &hm, &sinks) else { return Ok(None) };
    let node_local = tm.current_pos.clone();
    let mut best_global_pos = node_local.clone();
    let diversify_threshold = (max_no_improve / 3).max(20);
    let diversify_unit = (pre.avg_op_min * (0.10 + 0.16 * pre.jobshopness)).max(1.0) as u32;
    let kick_step = (max_no_improve / kick_spread).max(1);
    let mut kicks_left = 4usize;
    for iter in 0..max_iterations {
        if no_improve >= max_no_improve {
            if kicks_left == 0 { break; }
            ds.machine_seq.clone_from(&best_global_machine_seq);
            no_improve = 0;
            kicks_left -= 1;
            tabu_expiry.fill(0);
            let Some((mk, node, single)) = reload_mirror(&ds, &mut buf, &mut tm, &mut hg, &hm, &sinks) else { break };
            cur_mk = mk; mk_node = node; mk_single = single;
            continue;
        }
        if no_improve > 0 && no_improve % kick_step == 0 && kicks_left > 0 && no_improve / kick_step <= 4 {
            let Some((node, exact)) = settle_path(&ds, &mut buf, &tm.indeg_full, &job_pred_node, mk_node, mk_single, &hg.head, cur_mk) else { break };
            mk_node = node;
            if exact { for u in 0..n { buf.best_pred[u] = n2u(buf.nd[u].best_pred); } }
            else { for u in 0..n { buf.best_pred[u] = bp_of(&buf.nd, &job_pred_node, u, &hg.head); } }
            collect_guided_kick_moves(&ds, &buf.best_pred, mk_node, &tm.current_pos, &tm.node_machine, &node_local, &mut guided_used_machine, &mut guided_kicks);
            let ref_pos = if guided_kicks.is_empty() {
                collect_guided_kick_moves(&ds, &buf.best_pred, mk_node, &tm.current_pos, &tm.node_machine, &best_global_pos, &mut guided_used_machine, &mut guided_kicks);
                &best_global_pos[..]
            } else {
                &node_local[..]
            };
            if !guided_kicks.is_empty() {
                guided_kicks.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)).then_with(|| a.2.cmp(&b.2)));
                let mut applied = 0usize;
                for &(_, m, pos) in &guided_kicks {
                    if applied >= 3 { break; }
                    if pos + 1 >= ds.machine_seq[m].len() { continue; }
                    let node_a = ds.machine_seq[m][pos];
                    let node_b = ds.machine_seq[m][pos + 1];
                    if ref_pos[node_a] <= ref_pos[node_b] { continue; }
                    ds.machine_seq[m].swap(pos, pos + 1);
                    tm.current_pos[node_a] = pos + 1;
                    tm.current_pos[node_b] = pos;
                    machine_last_pick[m] = iter;
                    applied += 1;
                }
            }
            kicks_left -= 1;
            no_improve += 1;
            // The kick may have created a cycle; the graph is reloaded as a whole.
            let ev = reload_mirror(&ds, &mut buf, &mut tm, &mut hg, &hm, &sinks);
            let Some((mk, node, single)) = ev else {
                if !restore_after_failed_kick { break; }
                ds.machine_seq.clone_from(&best_global_machine_seq);
                let Some((mk, node, single)) = reload_mirror(&ds, &mut buf, &mut tm, &mut hg, &hm, &sinks) else { break };
                cur_mk = mk; mk_node = node; mk_single = single;
                continue;
            };
            cur_mk = mk; mk_node = node; mk_single = single;
            continue;
        }
        if iter > 0 {
            if cur_mk < best_global_mk {
                if cur_mk < published_mk {
                    publish(&disj_to_solution(pre, &ds, &hg.head))?;
                }
                best_global_mk = cur_mk;
                best_global_machine_seq.clone_from(&ds.machine_seq);
                rebuild_machine_pos_map(&best_global_machine_seq, &mut best_global_pos);
                no_improve = 0;
            } else {
                no_improve += 1;
            }
        }

        crit_cnt.fill(0);
        let Some((node, exact)) = settle_path(&ds, &mut buf, &tm.indeg_full, &job_pred_node, mk_node, mk_single, &hg.head, cur_mk) else { break };
        mk_node = node;
        // SAFETY: u is an operation id; m < num_machines; a simple path visits each machine at most crit_ml times.
        let mut u = mk_node;
        while u != NONE_USIZE {
            let m = unsafe { *tm.node_machine.get_unchecked(u) };
            let c = unsafe { *crit_cnt.get_unchecked(m) };
            unsafe { *crit_pos_flat.get_unchecked_mut(m * crit_ml + c) = *tm.current_pos.get_unchecked(u); }
            unsafe { *crit_cnt.get_unchecked_mut(m) = c + 1; }
            u = if exact { n2u(buf.nd[u].best_pred) } else { bp_of(&buf.nd, &job_pred_node, u, &hg.head) };
        }
        let diversify = ((no_improve.saturating_sub(diversify_threshold) as f64) / (max_no_improve.saturating_sub(diversify_threshold).max(1) as f64)).clamp(0.0, 1.0);
        let mut sel = MoveSel::new();
        for m in 0..ds.num_machines {
            let cnt = crit_cnt[m];
            if cnt < 2 { continue; }
            let positions = &mut crit_pos_flat[m * crit_ml..m * crit_ml + cnt];
            let mut descending = true;
            for w in 1..positions.len() {
                if positions[w - 1] <= positions[w] { descending = false; break; }
            }
            if descending { positions.reverse(); } else { positions.sort_unstable(); }
            let seq = &ds.machine_seq[m];
            let age = iter.saturating_sub(machine_last_pick[m]).min(9) as u32;
            for_each_block(positions, seq, &hg.head, &ds.node_pt, |run_start, run_end| {
                let block_len = run_end - run_start + 1;
                let swap_positions = [run_start, run_end - 1];
                let num_swaps = if block_len >= 3 { 2 } else { 1 };
                for &pos in &swap_positions[..num_swaps] {
                    if pos + 1 >= seq.len() { continue; }
                    let node_u = seq[pos];
                    let node_v = seq[pos + 1];
                    let est_mk = estimate_swap_nd(node_u, node_v, &buf.nd, &job_pred_node, &hg.head, &hg.qend);
                    let lu = node_local[node_u];
                    let lv = node_local[node_v];
                    let (a, b) = if lu < lv { (lu, lv) } else { (lv, lu) };
                    let tabu_idx = pair_offsets[m] + b * (b - 1) / 2 + a;
                    let is_tabu = tabu_expiry[tabu_idx] > iter;
                    let aspiration = est_mk < best_global_mk;
                    let block_bonus = block_len.saturating_sub(2).min(3) as u32;
                    let div_bonus = if diversify > 0.0 {
                        ((diversify_unit as f64) * diversify * (0.35 * (age as f64) + 0.75 * (block_bonus as f64)) / 3.0) as u32
                    } else { 0 };
                    let adj_mk = est_mk.saturating_sub(div_bonus);
                    sel.offer(m, pos, est_mk, adj_mk, !is_tabu || aspiration, &mut tseed);
                }
            });
        }
        let Some((m, pos, _)) = sel.best.or(sel.relaxed) else { break };
        let node_a = ds.machine_seq[m][pos];
        let node_b = ds.machine_seq[m][pos + 1];
        ds.machine_seq[m].swap(pos, pos + 1);
        tm.current_pos[node_a] = pos + 1;
        tm.current_pos[node_b] = pos;
        machine_last_pick[m] = iter;
        let seq = &ds.machine_seq[m];
        let prev = if pos > 0 { seq[pos - 1] } else { NONE_USIZE };
        let next = if pos + 2 < seq.len() { seq[pos + 2] } else { NONE_USIZE };
        tm.machine_pred_node[node_b] = prev;
        tm.machine_pred_node[node_a] = node_b;
        buf.nd[node_b].machine_pred = u2n(prev);
        buf.nd[node_a].machine_pred = node_b as u32;
        if next != NONE_USIZE {
            tm.machine_pred_node[next] = node_a;
            buf.nd[next].machine_pred = node_a as u32;
        }
        if prev != NONE_USIZE {
            buf.machine_succ[prev] = node_b;
            buf.nd[prev].machine_succ = node_b as u32;
        }
        buf.machine_succ[node_b] = node_a;
        buf.nd[node_b].machine_succ = node_a as u32;
        buf.machine_succ[node_a] = next;
        buf.nd[node_a].machine_succ = u2n(next);
        if pos == 0 {
            tm.indeg_full[node_a] = tm.indeg_full[node_a].saturating_add(1);
            tm.indeg_full[node_b] = tm.indeg_full[node_b].saturating_sub(1);
        }
        pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
        let offset = (pseed % ((2 * tenure_delta + 1) as u64)) as usize;
        let progress = (iter as f64) / (max_iterations as f64);
        let late_bonus = if progress > 0.6 { ((progress - 0.6) * 10.0) as usize } else { 0 };
        let this_tenure = (tenure + offset + late_bonus).saturating_sub(tenure_delta);
        let la = node_local[node_a];
        let lb = node_local[node_b];
        let (a, b) = if la < lb { (la, lb) } else { (lb, la) };
        tabu_expiry[pair_offsets[m] + b * (b - 1) / 2 + a] = iter + this_tenure;
        hg.swap_adjacent(node_a as u32, node_b as u32);
        let Some((mk, node, single)) = hg_eval_nd(&mut hg, &ds.node_pt, &sinks) else { break };
        cur_mk = mk; mk_node = node; mk_single = single;
    }
    if cur_mk < best_global_mk { best_global_mk = cur_mk; best_global_machine_seq.clone_from(&ds.machine_seq); }
    if best_global_mk >= initial_mk { return Ok(None); }
    ds.machine_seq = best_global_machine_seq;
    let Some((mk_final, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    Ok(Some((disj_to_solution(pre, &ds, &buf.start), mk_final)))
}

#[inline]
fn relocate_machine_seq(seq: &mut [usize], from: usize, to: usize) {
    if from < to {
        seq[from..=to].rotate_left(1);
    } else if to < from {
        seq[to..=from].rotate_right(1);
    }
}

/// Crossover of one machine: starts from the better parent and moves at most one or two
/// operations that the worse parent and the elite consensus both place earlier.
#[inline]
fn inherit_machine_consensus_order(
    child_seq: &mut Vec<usize>,
    better_seq: &[usize],
    worse_seq: &[usize],
    elite_machine_seqs: &[&[usize]],
    pos_buf: &mut [usize],
    sum_buf: &mut [u32],
    cnt_buf: &mut [u16],
    pair_buf: &mut Vec<(usize, usize, i32)>,
) {
    if child_seq.len() != better_seq.len() {
        child_seq.clear();
        child_seq.extend_from_slice(better_seq);
    }
    if better_seq.len() <= 1 || better_seq.len() != worse_seq.len() {
        return;
    }
    for &node in better_seq {
        pos_buf[node] = 0;
        sum_buf[node] = 0;
        cnt_buf[node] = 0;
    }
    for (idx, &node) in better_seq.iter().enumerate() {
        pos_buf[node] = idx;
    }
    for &eseq in elite_machine_seqs {
        if eseq.len() != better_seq.len() { continue; }
        for (idx, &node) in eseq.iter().enumerate() {
            sum_buf[node] = sum_buf[node].saturating_add(idx as u32);
            cnt_buf[node] = cnt_buf[node].saturating_add(1);
        }
    }
    pair_buf.clear();
    for win in worse_seq.windows(2) {
        let u = win[0];
        let v = win[1];
        let pu = pos_buf[u];
        let pv = pos_buf[v];
        if pu < pv { continue; }
        let cu = cnt_buf[u];
        let cv = cnt_buf[v];
        if cu == 0 || cv == 0 { continue; }
        let avg_u = (sum_buf[u] as f64) / (cu as f64);
        let avg_v = (sum_buf[v] as f64) / (cv as f64);
        let gap = avg_v - avg_u;
        if gap <= 0.55 { continue; }
        let score = ((gap * 16.0) as i32) + ((pu - pv).min(10) as i32);
        pair_buf.push((u, v, score));
    }
    if pair_buf.is_empty() { return; }
    pair_buf.sort_unstable_by(|a, b| b.2.cmp(&a.2));
    let apply_cap = if better_seq.len() > 14 { 2usize } else { 1usize };
    let mut applied = 0usize;
    for &(u, v, _) in pair_buf.iter() {
        if applied >= apply_cap { break; }
        for (idx, &node) in child_seq.iter().enumerate() {
            pos_buf[node] = idx;
        }
        let from = pos_buf[u];
        let to = pos_buf[v];
        if from < to { continue; }
        child_seq[to..=from].rotate_right(1);
        applied += 1;
    }
}

#[inline]
fn invert_job_bias_guidance(jb: &[f64]) -> Vec<f64> {
    if jb.is_empty() { return Vec::new(); }
    let mean = jb.iter().copied().sum::<f64>() / (jb.len() as f64);
    jb.iter().map(|&v| (mean - v) * 0.75).collect()
}

#[inline]
fn invert_machine_penalty_guidance(mp: &[f64]) -> Vec<f64> {
    if mp.is_empty() { return Vec::new(); }
    let mean = mp.iter().copied().sum::<f64>() / (mp.len() as f64);
    mp.iter().map(|&v| (0.55 * mean + 0.45 * (1.0 - v.clamp(0.0, 1.0))).clamp(0.0, 1.0)).collect()
}

/// Descent on critical blocks with adjacent swaps at block ends; the two best estimates are
/// evaluated exactly at each step. Returns the improved makespan, or None.
fn critical_block_move_local_search_ex_disj(
    ds: &mut DisjSchedule,
    buf: &mut EvalBuf,
    max_rounds: usize,
    max_iters: usize,
) -> Option<u32> {
    let n = ds.n;
    let Some((initial_mk, mut mk_node)) = eval_disj(ds, buf) else { return None };
    let mut test_buf_a = EvalBuf::new(n);
    let mut test_buf_b = EvalBuf::new(n);
    let mut cur_mk = initial_mk;
    let mut qend = ds.node_pt.clone();
    let mut back_deg = vec![0u16; n];
    let mut back_stack: Vec<usize> = Vec::with_capacity(n);
    let mut machine_pred_node = vec![NONE_USIZE; n];
    let mut job_pred_node = vec![NONE_USIZE; n];
    let mut moves: Vec<(u32, u16, usize, usize)> = Vec::with_capacity(64);
    let mut current_pos = vec![0usize; n];
    let mut node_machine = vec![0usize; n];
    let mut crit_positions: Vec<(usize, usize)> = Vec::with_capacity(n);
    let mut positions: Vec<usize> = Vec::with_capacity(n);
    for j in 0..ds.num_jobs {
        let base = ds.job_offsets[j];
        let end = ds.job_offsets[j + 1];
        for k in (base + 1)..end {
            job_pred_node[k] = k - 1;
        }
    }
    for m in 0..ds.num_machines {
        for (i, &node) in ds.machine_seq[m].iter().enumerate() {
            current_pos[node] = i;
            node_machine[node] = m;
        }
    }
    let iter_limit = max_iters.max(max_rounds).max(1);
    for _ in 0..iter_limit {
        rebuild_machine_pred_nodes(ds, &mut machine_pred_node);
        compute_qend(ds, &buf.machine_succ, &machine_pred_node, &job_pred_node, &mut qend, &mut back_deg, &mut back_stack);
        crit_positions.clear();
        let mut u = mk_node;
        while u != NONE_USIZE {
            crit_positions.push((node_machine[u], current_pos[u]));
            u = buf.best_pred[u];
        }
        if crit_positions.len() > 1 { crit_positions.sort_unstable(); }
        moves.clear();
        let mut cp_i = 0usize;
        while cp_i < crit_positions.len() {
            let m = crit_positions[cp_i].0;
            let mut cp_j = cp_i;
            while cp_j < crit_positions.len() && crit_positions[cp_j].0 == m { cp_j += 1; }
            positions.clear();
            positions.extend(crit_positions[cp_i..cp_j].iter().map(|&(_, p)| p));
            cp_i = cp_j;
            let seq = &ds.machine_seq[m];
            for_each_block(&positions, seq, &buf.start, &ds.node_pt, |run_start, run_end| {
                let block_len = run_end - run_start + 1;
                let swap_positions = [run_start, run_end - 1];
                let num_swaps = if block_len >= 3 { 2 } else { 1 };
                let block_len_u16 = block_len.min(u16::MAX as usize) as u16;
                for &pos in &swap_positions[..num_swaps] {
                    if pos + 1 >= seq.len() { continue; }
                    let est_mk = estimate_swap_mk(seq[pos], seq[pos + 1], &buf.start, &qend, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ);
                    if est_mk < cur_mk {
                        moves.push((est_mk, block_len_u16, m, pos));
                    }
                }
            });
        }
        if moves.is_empty() { break; }
        if moves.len() > 1 {
            moves.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)).then_with(|| a.2.cmp(&b.2)).then_with(|| a.3.cmp(&b.3)));
        }
        let eval_cap = moves.len().min(2);
        let mut best_idx = NONE_USIZE;
        let mut best_actual_mk = cur_mk;
        let mut best_actual_node = NONE_USIZE;
        let mut tested = [(NONE_USIZE, NONE_USIZE); 2];
        for idx in 0..eval_cap {
            let (_, _, m, pos) = moves[idx];
            if pos + 1 >= ds.machine_seq[m].len() { continue; }
            tested[idx] = (m, pos);
            ds.machine_seq[m].swap(pos, pos + 1);
            let res = if idx == 0 { eval_disj(ds, &mut test_buf_a) } else { eval_disj(ds, &mut test_buf_b) };
            if let Some((new_mk, new_node)) = res {
                if new_mk < best_actual_mk {
                    best_actual_mk = new_mk;
                    best_actual_node = new_node;
                    best_idx = idx;
                }
            }
            ds.machine_seq[m].swap(pos, pos + 1);
        }
        if best_idx == NONE_USIZE { break; }
        let (m, pos) = tested[best_idx];
        let node_a = ds.machine_seq[m][pos];
        let node_b = ds.machine_seq[m][pos + 1];
        ds.machine_seq[m].swap(pos, pos + 1);
        current_pos[node_a] = pos + 1;
        current_pos[node_b] = pos;
        cur_mk = best_actual_mk;
        mk_node = best_actual_node;
        if best_idx == 0 {
            core::mem::swap(buf, &mut test_buf_a);
        } else {
            core::mem::swap(buf, &mut test_buf_b);
        }
    }
    if cur_mk < initial_mk { Some(cur_mk) } else { None }
}

/// Moves the largest critical block of the current path num_perturb times (block relocation,
/// rotation or swap by block length), then repairs by critical-block descent.
fn perturb_and_reoptimize_ils(
    pre: &Pre,
    challenge: &Challenge,
    base_sol: &Solution,
    num_perturb: usize,
    ls_rounds: usize,
    ls_iters: usize,
) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n);
    let Some((start_mk, mut mk_node)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let mut cur_mk = start_mk;
    let mut current_pos = vec![0usize; ds.n];
    let mut node_machine = vec![0usize; ds.n];
    let mut crit_pos_by_machine: Vec<Vec<usize>> = (0..ds.num_machines).map(|_| Vec::new()).collect();
    let mut crit_pos_machines: Vec<usize> = Vec::with_capacity(ds.num_machines);
    let mut moved = 0usize;

    for _ in 0..num_perturb {
        for m in 0..ds.num_machines {
            crit_pos_by_machine[m].clear();
            for (pos, &node) in ds.machine_seq[m].iter().enumerate() {
                current_pos[node] = pos;
                node_machine[node] = m;
            }
        }
        crit_pos_machines.clear();
        let mut u = mk_node;
        while u != NONE_USIZE {
            let m = node_machine[u];
            if crit_pos_by_machine[m].is_empty() { crit_pos_machines.push(m); }
            crit_pos_by_machine[m].push(current_pos[u]);
            u = buf.best_pred[u];
        }

        // (machine, a, b, kind, block_len, pt_sum); kind 0 relocates a to b, 1/2 rotate a..=a+2, 3 swaps a, a+1.
        let mut best_move: Option<(usize, usize, usize, u8, usize, u32)> = None;
        for &m in &crit_pos_machines {
            let positions = &mut crit_pos_by_machine[m];
            if positions.len() < 2 { continue; }
            positions.sort_unstable();
            let seq = &ds.machine_seq[m];
            for_each_block(positions, seq, &buf.start, &ds.node_pt, |run_start, run_end| {
                let block_len = run_end - run_start + 1;
                let pt_sum = seq[run_start..=run_end].iter().fold(0u32, |acc, &nd| acc.saturating_add(ds.node_pt[nd]));
                let cand = if block_len >= 5 {
                    let head_mass = ds.node_pt[seq[run_start]].saturating_add(ds.node_pt[seq[run_start + 1]]);
                    let tail_mass = ds.node_pt[seq[run_end - 1]].saturating_add(ds.node_pt[seq[run_end]]);
                    if head_mass <= tail_mass { (m, run_start, run_end - 1, 0u8, block_len, pt_sum) } else { (m, run_end, run_start + 1, 0u8, block_len, pt_sum) }
                } else if block_len == 4 {
                    let head_mass = ds.node_pt[seq[run_start]].saturating_add(ds.node_pt[seq[run_start + 1]]);
                    let tail_mass = ds.node_pt[seq[run_end - 1]].saturating_add(ds.node_pt[seq[run_end]]);
                    if head_mass <= tail_mass { (m, run_start, 0usize, 1u8, block_len, pt_sum) } else { (m, run_end - 2, 0usize, 2u8, block_len, pt_sum) }
                } else if block_len == 3 {
                    if ds.node_pt[seq[run_start]] <= ds.node_pt[seq[run_end]] { (m, run_start, 0usize, 1u8, block_len, pt_sum) } else { (m, run_start, 0usize, 2u8, block_len, pt_sum) }
                } else {
                    (m, run_start, 0usize, 3u8, block_len, pt_sum)
                };
                if best_move.as_ref().map_or(true, |&(_, _, _, _, best_len, best_sum)| block_len > best_len || (block_len == best_len && pt_sum > best_sum)) {
                    best_move = Some(cand);
                }
            });
        }

        let Some((m, a, b, kind, _, _)) = best_move else { break; };
        let apply = |seq: &mut [usize], undo: bool| match (kind, undo) {
            (0, false) => relocate_machine_seq(seq, a, b),
            (0, true) => relocate_machine_seq(seq, b, a),
            (1, false) | (2, true) => seq[a..=a + 2].rotate_left(1),
            (2, false) | (1, true) => seq[a..=a + 2].rotate_right(1),
            _ => seq.swap(a, a + 1),
        };
        apply(&mut ds.machine_seq[m], false);
        match eval_disj(&ds, &mut buf) {
            Some((new_mk, new_node)) => {
                cur_mk = new_mk;
                mk_node = new_node;
                moved += 1;
            }
            None => {
                apply(&mut ds.machine_seq[m], true);
                let Some((restored_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
                cur_mk = restored_mk;
                break;
            }
        }
    }

    if moved == 0 { return Ok(None); }

    let final_mk = match critical_block_move_local_search_ex_disj(&mut ds, &mut buf, ls_rounds, ls_iters) {
        Some(mk) => mk,
        None => match eval_disj(&ds, &mut buf) {
            Some((mk, _)) => mk,
            None => return Ok(None),
        },
    };
    if final_mk >= start_mk && cur_mk >= start_mk { return Ok(None); }
    Ok(Some((disj_to_solution(pre, &ds, &buf.start), final_mk.min(cur_mk))))
}

/// Node budget of the Carlier branch and bound on one machine.
const SBP_NODE_BUDGET: usize = 600;

/// Graph scratch space of the shifting-bottleneck procedure and the loaded one-machine problem.
struct SbpBuf {
    msucc: Vec<usize>,
    mpred: Vec<usize>,
    jpred: Vec<usize>,
    deg: Vec<u16>,
    stack: Vec<usize>,
    head: Vec<u32>,
    tail: Vec<u32>,
    nodes: Vec<usize>,
    r: Vec<u32>,
    p: Vec<u32>,
    q: Vec<u32>,
}

impl SbpBuf {
    fn new(n: usize) -> Self {
        Self {
            msucc: vec![NONE_USIZE; n], mpred: vec![NONE_USIZE; n], jpred: vec![NONE_USIZE; n],
            deg: vec![0u16; n], stack: Vec::with_capacity(n),
            head: vec![0u32; n], tail: vec![0u32; n],
            nodes: Vec::new(), r: Vec::new(), p: Vec::new(), q: Vec::new(),
        }
    }

    /// Loads the one-machine problem (release, duration, tail) of the given operations.
    fn load_machine(&mut self, ds: &DisjSchedule, members: &[usize]) {
        self.nodes.clear();
        self.nodes.extend_from_slice(members);
        self.r.clear();
        self.p.clear();
        self.q.clear();
        for &nd in &self.nodes {
            self.r.push(self.head[nd]);
            self.p.push(ds.node_pt[nd]);
            self.q.push(self.tail[nd]);
        }
    }
}

/// Scratch space of the Carlier branch and bound.
struct CarlierBuf {
    done: Vec<bool>,
    order: Vec<usize>,
    best_order: Vec<usize>,
}

impl CarlierBuf {
    fn new() -> Self {
        Self { done: Vec::new(), order: Vec::new(), best_order: Vec::new() }
    }
}

/// Heads and tails of every operation with the arcs of machine m_excl removed; false on a cycle.
fn sbp_heads_tails_excl(ds: &DisjSchedule, m_excl: usize, b: &mut SbpBuf) -> bool {
    let n = ds.n;
    for u in 0..n { b.msucc[u] = NONE_USIZE; b.mpred[u] = NONE_USIZE; b.jpred[u] = NONE_USIZE; }
    for (m, seq) in ds.machine_seq.iter().enumerate() {
        if m == m_excl || seq.len() < 2 { continue; }
        for w in seq.windows(2) { b.msucc[w[0]] = w[1]; b.mpred[w[1]] = w[0]; }
    }
    for u in 0..n {
        let js = ds.job_succ[u];
        if js != NONE_USIZE { b.jpred[js] = u; }
    }
    for u in 0..n {
        let mut d = ds.indeg_job[u];
        if b.mpred[u] != NONE_USIZE { d = d.saturating_add(1); }
        b.deg[u] = d;
        b.head[u] = 0;
    }
    b.stack.clear();
    for u in 0..n { if b.deg[u] == 0 { b.stack.push(u); } }
    let mut processed = 0usize;
    while let Some(u) = b.stack.pop() {
        processed += 1;
        let end_u = b.head[u].saturating_add(ds.node_pt[u]);
        for s in [ds.job_succ[u], b.msucc[u]] {
            if s == NONE_USIZE { continue; }
            if b.head[s] < end_u { b.head[s] = end_u; }
            b.deg[s] = b.deg[s].saturating_sub(1);
            if b.deg[s] == 0 { b.stack.push(s); }
        }
    }
    if processed != n { return false; }
    for u in 0..n {
        let mut d = 0u16;
        if ds.job_succ[u] != NONE_USIZE { d += 1; }
        if b.msucc[u] != NONE_USIZE { d += 1; }
        b.deg[u] = d;
        b.tail[u] = 0;
    }
    b.stack.clear();
    for u in 0..n { if b.deg[u] == 0 { b.stack.push(u); } }
    processed = 0;
    while let Some(u) = b.stack.pop() {
        processed += 1;
        let need = b.tail[u].saturating_add(ds.node_pt[u]);
        for p in [b.jpred[u], b.mpred[u]] {
            if p == NONE_USIZE { continue; }
            if b.tail[p] < need { b.tail[p] = need; }
            b.deg[p] = b.deg[p].saturating_sub(1);
            if b.deg[p] == 0 { b.stack.push(p); }
        }
    }
    processed == n
}

/// Value max(C_i + q_i) of a one-machine order.
#[inline]
fn sbp_eval_order(order: impl Iterator<Item = usize>, r: &[u32], p: &[u32], q: &[u32]) -> u32 {
    let mut t = 0u32;
    let mut value = 0u32;
    for i in order {
        let s = if t > r[i] { t } else { r[i] };
        let e = s.saturating_add(p[i]);
        let c = e.saturating_add(q[i]);
        if c > value { value = c; }
        t = e;
    }
    value
}

/// Schrage order: at each time, the released job with the largest tail.
fn sbp_schrage(r: &[u32], p: &[u32], q: &[u32], done: &mut Vec<bool>, order: &mut Vec<usize>) -> u32 {
    let k = r.len();
    done.clear();
    done.resize(k, false);
    order.clear();
    let mut t = u32::MAX;
    for i in 0..k { if r[i] < t { t = r[i]; } }
    let mut value = 0u32;
    for _ in 0..k {
        let mut pick = usize::MAX;
        for i in 0..k {
            if done[i] || r[i] > t { continue; }
            if pick == usize::MAX || q[i] > q[pick] { pick = i; }
        }
        if pick == usize::MAX {
            let mut nt = u32::MAX;
            for i in 0..k { if !done[i] && r[i] < nt { nt = r[i]; } }
            if nt == u32::MAX { break; }
            t = nt;
            for i in 0..k {
                if done[i] || r[i] > t { continue; }
                if pick == usize::MAX || q[i] > q[pick] { pick = i; }
            }
            if pick == usize::MAX { break; }
        }
        done[pick] = true;
        order.push(pick);
        let e = t.saturating_add(p[pick]);
        let c = e.saturating_add(q[pick]);
        if c > value { value = c; }
        t = e;
    }
    value
}

/// Carlier branch and bound on one machine (bounded node budget); best order left in cb.best_order.
fn sbp_carlier(r0: &[u32], p: &[u32], q0: &[u32], cb: &mut CarlierBuf) -> Option<u32> {
    let k = r0.len();
    if k < 2 { return None; }
    cb.best_order.clear();
    let mut ub = u32::MAX;
    let mut nodes = 0usize;
    let mut starts: Vec<u32> = Vec::with_capacity(k);
    let mut stack: Vec<(Vec<u32>, Vec<u32>)> = Vec::with_capacity(16);
    stack.push((r0.to_vec(), q0.to_vec()));

    while let Some((r, q)) = stack.pop() {
        if nodes >= SBP_NODE_BUDGET { break; }
        nodes += 1;
        let f = sbp_schrage(&r, p, &q, &mut cb.done, &mut cb.order);
        let true_val = sbp_eval_order(cb.order.iter().copied(), r0, p, q0);
        if true_val < ub {
            ub = true_val;
            cb.best_order.clear();
            cb.best_order.extend_from_slice(&cb.order);
        }
        if cb.order.len() != k { continue; }

        starts.clear();
        let mut t = 0u32;
        let mut pos_star = usize::MAX;
        for (pos, &i) in cb.order.iter().enumerate() {
            let s = if t > r[i] { t } else { r[i] };
            starts.push(s);
            t = s.saturating_add(p[i]);
            if t.saturating_add(q[i]) == f { pos_star = pos; }
        }
        if pos_star == usize::MAX { continue; }

        let mut a = pos_star;
        while a > 0 {
            let prev = cb.order[a - 1];
            if starts[a] != starts[a - 1].saturating_add(p[prev]) { break; }
            a -= 1;
        }
        let q_star = q[cb.order[pos_star]];
        let mut pos_c = usize::MAX;
        let mut pos = pos_star;
        while pos > a {
            pos -= 1;
            if q[cb.order[pos]] < q_star { pos_c = pos; break; }
        }
        if pos_c == usize::MAX { continue; }

        let c = cb.order[pos_c];
        let mut r_min = u32::MAX;
        let mut q_min = u32::MAX;
        let mut p_sum = 0u32;
        for &i in &cb.order[(pos_c + 1)..=pos_star] {
            if r[i] < r_min { r_min = r[i]; }
            if q[i] < q_min { q_min = q[i]; }
            p_sum = p_sum.saturating_add(p[i]);
        }
        if r_min == u32::MAX || q_min == u32::MAX { continue; }

        let h_j = r_min.saturating_add(p_sum).saturating_add(q_min);
        let r_min2 = if r[c] < r_min { r[c] } else { r_min };
        let q_min2 = if q[c] < q_min { q[c] } else { q_min };
        let h_jc = r_min2.saturating_add(p_sum).saturating_add(p[c]).saturating_add(q_min2);
        let lb = if h_j > h_jc { h_j } else { h_jc };
        if lb >= ub { continue; }
        if stack.len() + 2 > 2 * SBP_NODE_BUDGET { continue; }

        let forced_q = q_min.saturating_add(p_sum);
        if forced_q > q[c] {
            let mut q2 = q.clone();
            q2[c] = forced_q;
            stack.push((r.clone(), q2));
        }
        let forced_r = r_min.saturating_add(p_sum);
        if forced_r > r[c] {
            let mut r2 = r.clone();
            r2[c] = forced_r;
            stack.push((r2, q.clone()));
        }
    }
    if cb.best_order.len() == k { Some(ub) } else { None }
}

/// Re-sequences each machine (heaviest first) by Carlier with the other machines fixed, for rounds passes.
fn sbp_reoptimize(pre: &Pre, challenge: &Challenge, sol: &Solution, rounds: usize) -> Option<(Solution, u32)> {
    if rounds == 0 { return None; }
    let mut ds = build_disj_from_solution(pre, challenge, sol).ok()?;
    let mut buf = EvalBuf::new(ds.n);
    let (start_mk, _) = eval_disj(&ds, &mut buf)?;
    let mut cur_mk = start_mk;
    let mut b = SbpBuf::new(ds.n);
    let mut cb = CarlierBuf::new();
    let mut saved_seq: Vec<usize> = Vec::new();
    let num_machines = challenge.num_machines.min(ds.machine_seq.len());
    let mut m_order: Vec<usize> = (0..num_machines).filter(|&m| ds.machine_seq[m].len() >= 3).collect();
    let mut load: Vec<u64> = vec![0u64; num_machines];
    for m in 0..num_machines {
        for &nd in &ds.machine_seq[m] { load[m] = load[m].saturating_add(ds.node_pt[nd] as u64); }
    }
    m_order.sort_by(|&x, &y| load[y].cmp(&load[x]).then(x.cmp(&y)));
    if m_order.is_empty() { return None; }

    for _ in 0..rounds {
        let mut improved = false;
        for &m in &m_order {
            let k = ds.machine_seq[m].len();
            if k < 3 { continue; }
            if !sbp_heads_tails_excl(&ds, m, &mut b) { continue; }
            b.load_machine(&ds, &ds.machine_seq[m]);
            let cur_val = sbp_eval_order(0..k, &b.r, &b.p, &b.q);
            let Some(val) = sbp_carlier(&b.r, &b.p, &b.q, &mut cb) else { continue };
            if val >= cur_val { continue; }
            saved_seq.clear();
            saved_seq.extend_from_slice(&ds.machine_seq[m]);
            for pos in 0..k { ds.machine_seq[m][pos] = b.nodes[cb.best_order[pos]]; }
            match eval_disj(&ds, &mut buf) {
                Some((new_mk, _)) if new_mk < cur_mk => { cur_mk = new_mk; improved = true; }
                _ => {
                    ds.machine_seq[m].clear();
                    ds.machine_seq[m].extend_from_slice(&saved_seq);
                }
            }
        }
        if !improved { break; }
    }
    if cur_mk >= start_mk { return None; }
    let (final_mk, _) = eval_disj(&ds, &mut buf)?;
    if final_mk >= start_mk { return None; }
    Some((disj_to_solution(pre, &ds, &buf.start), final_mk))
}

/// Shifting-bottleneck construction: machines are sequenced one at a time by Carlier, the machine
/// with the largest value first, and the sequenced machines are re-optimised after each addition
/// (at most reopt_cycles passes, all machines when 0). `template` only supplies the machine of each operation.
fn sbp_construct_seed(
    pre: &Pre,
    challenge: &Challenge,
    template: &Solution,
    reopt_cycles: usize,
) -> Option<(Solution, u32)> {
    let mut ds = build_disj_from_solution(pre, challenge, template).ok()?;
    let n = ds.n;
    let num_machines = challenge.num_machines.min(ds.machine_seq.len());
    if num_machines == 0 { return None; }

    let mut members: Vec<Vec<usize>> = vec![Vec::new(); num_machines];
    for u in 0..n {
        let m = ds.node_machine[u];
        if m < num_machines { members[m].push(u); }
    }
    for m in 0..num_machines {
        ds.machine_seq[m].clear();
        if members[m].len() < 2 { ds.machine_seq[m].extend_from_slice(&members[m]); }
    }

    let mut b = SbpBuf::new(n);
    let mut cb = CarlierBuf::new();
    let mut buf = EvalBuf::new(n);
    let mut sequenced: Vec<bool> = (0..num_machines).map(|m| members[m].len() < 2).collect();
    let mut fixed_order: Vec<usize> = Vec::with_capacity(num_machines);
    let mut pending = sequenced.iter().filter(|&&s| !s).count();
    let mut best_seq: Vec<usize> = Vec::new();
    let mut cur_seq: Vec<usize> = Vec::new();
    let mut saved_seq: Vec<usize> = Vec::new();

    while pending > 0 {
        let mut best_m = NONE_USIZE;
        let mut best_val = 0u32;
        for m in 0..num_machines {
            if sequenced[m] { continue; }
            if !sbp_heads_tails_excl(&ds, m, &mut b) { continue; }
            b.load_machine(&ds, &members[m]);
            let Some(val) = sbp_carlier(&b.r, &b.p, &b.q, &mut cb) else { continue };
            if cb.best_order.len() != b.nodes.len() { continue; }
            if best_m == NONE_USIZE || val > best_val {
                best_val = val;
                best_m = m;
                best_seq.clear();
                for &pos in cb.best_order.iter() { best_seq.push(b.nodes[pos]); }
            }
        }
        if best_m == NONE_USIZE { return None; }

        ds.machine_seq[best_m].clear();
        ds.machine_seq[best_m].extend_from_slice(&best_seq);
        if !sbp_heads_tails_excl(&ds, usize::MAX, &mut b) {
            ds.machine_seq[best_m].clear();
            if !sbp_heads_tails_excl(&ds, best_m, &mut b) { return None; }
            let mut order: Vec<usize> = members[best_m].clone();
            order.sort_by(|&x, &y| b.head[x].cmp(&b.head[y]).then(x.cmp(&y)));
            ds.machine_seq[best_m].clear();
            ds.machine_seq[best_m].extend_from_slice(&order);
            if !sbp_heads_tails_excl(&ds, usize::MAX, &mut b) { return None; }
        }
        sequenced[best_m] = true;
        fixed_order.push(best_m);
        pending -= 1;

        let cap = if reopt_cycles == 0 { num_machines } else { reopt_cycles.min(num_machines) };
        for _ in 0..cap {
            let mut changed = false;
            for i in 0..fixed_order.len() {
                let m2 = fixed_order[i];
                if ds.machine_seq[m2].len() < 3 { continue; }
                if !sbp_heads_tails_excl(&ds, m2, &mut b) { continue; }
                b.load_machine(&ds, &members[m2]);
                cur_seq.clear();
                for &nd in &ds.machine_seq[m2] {
                    let Some(pos) = b.nodes.iter().position(|&x| x == nd) else { return None };
                    cur_seq.push(pos);
                }
                let cur_val = sbp_eval_order(cur_seq.iter().copied(), &b.r, &b.p, &b.q);
                let Some(val) = sbp_carlier(&b.r, &b.p, &b.q, &mut cb) else { continue };
                if val >= cur_val { continue; }
                if cb.best_order.len() != b.nodes.len() { continue; }
                saved_seq.clear();
                saved_seq.extend_from_slice(&ds.machine_seq[m2]);
                ds.machine_seq[m2].clear();
                for &pos in cb.best_order.iter() { ds.machine_seq[m2].push(b.nodes[pos]); }
                if !sbp_heads_tails_excl(&ds, usize::MAX, &mut b) {
                    ds.machine_seq[m2].clear();
                    ds.machine_seq[m2].extend_from_slice(&saved_seq);
                } else {
                    changed = true;
                }
            }
            if !changed { break; }
        }
    }

    let (mk, _) = eval_disj(&ds, &mut buf)?;
    Some((disj_to_solution(pre, &ds, &buf.start), mk))
}

/// Number of best solutions kept between phases.
const POOL_CAP: usize = 20;
/// Solutions kept while constructing.
const CONSTRUCT_POOL_CAP: usize = 15;
/// Randomised constructions after the first ranking pass.
const NUM_RESTARTS: usize = 450;
/// Largest top-k of a randomised construction.
const K_HI: usize = 6;

/// Publishes sol when it beats the best makespan seen so far.
fn record(best_mk: &mut u32, best: &mut Solution, sol: &Solution, mk: u32, save: &dyn Fn(&Solution) -> Result<()>) -> Result<bool> {
    if mk < *best_mk {
        *best_mk = mk;
        *best = sol.clone();
        save(sol)?;
        return Ok(true);
    }
    Ok(false)
}

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    pre: &Pre,
    effort: &EffortConfig,
) -> Result<()> {
    let (greedy_sol, greedy_mk) = run_simple_greedy_baseline(challenge, pre)?;
    save_solution(&greedy_sol)?;

    let mut rng = SmallRng::from_seed(challenge.seed);
    let mut best_makespan = greedy_mk;
    let mut best_solution: Solution = greedy_sol.clone();
    let mut top_solutions: Vec<(Solution, u32)> = Vec::new();
    push_top_solutions(&mut top_solutions, &greedy_sol, greedy_mk, CONSTRUCT_POOL_CAP);
    let target_margin: u32 = ((pre.avg_op_min * (0.9 + 0.6 * pre.jobshopness)).max(1.0)) as u32;

    if pre.flow_route.is_some() && pre.flow_pt_by_job.is_some() {
        if let Ok((sol, mk)) = neh_flow_solution(pre, challenge.num_jobs, challenge.num_machines) {
            record(&mut best_makespan, &mut best_solution, &sol, mk, save_solution)?;
            push_top_solutions(&mut top_solutions, &sol, mk, CONSTRUCT_POOL_CAP);
        }
    }

    // One deterministic construction per rule; the three best rules seed the randomised restarts.
    let mut ranked: Vec<(Rule, u32, Solution)> = Vec::with_capacity(RULES.len());
    for &rule in &RULES {
        let (sol, mk) = construct_solution_conflict(challenge, pre, rule, 0, None, &mut rng, None)?;
        record(&mut best_makespan, &mut best_solution, &sol, mk, save_solution)?;
        push_top_solutions(&mut top_solutions, &sol, mk, CONSTRUCT_POOL_CAP);
        ranked.push((rule, mk, sol));
    }
    ranked.sort_by_key(|x| x.1);
    let r0 = ranked[0].0;
    let r1 = ranked.get(1).map(|x| x.0).unwrap_or(r0);
    let r2 = ranked.get(2).map(|x| x.0).unwrap_or(r1);
    let mut rule_best: Vec<u32> = vec![u32::MAX; RULES.len()];
    let mut rule_tries: Vec<u32> = vec![0u32; RULES.len()];
    for (rr, mk, _) in &ranked {
        let idx = rule_idx(*rr);
        rule_best[idx] = rule_best[idx].min(*mk);
        rule_tries[idx] = rule_tries[idx].saturating_add(1);
    }

    let base = &ranked[0].2;
    let mut learned_jb = job_bias_from_solution(pre, base);
    let mut learned_mp = machine_penalty_from_solution(pre, base, challenge.num_machines);
    let mut learn_updates_left = 4usize;
    let mut stuck: usize = 0;
    for r in 0..NUM_RESTARTS {
        let late = r >= (NUM_RESTARTS * 2) / 3;
        let (k_min, k_max) = if stuck > 170 { (4usize, K_HI) } else if stuck > 90 { (3usize, K_HI) } else if stuck > 35 { (2usize, K_HI) } else { (2usize, K_HI.min(4)) };
        let rule = if r < 35 {
            let u: f64 = rng.gen();
            if u < 0.52 { r0 } else if u < 0.80 { r1 } else if u < 0.92 { r2 } else { RULES[rng.gen_range(0..RULES.len())] }
        } else {
            choose_rule_bandit(&mut rng, &rule_best, &rule_tries, best_makespan, target_margin, stuck, late)
        };
        let k = rng.gen_range(k_min..=k_max);
        let learn_base = (0.08 + 0.22 * pre.jobshopness).clamp(0.05, 0.42);
        let learn_boost = (1.0 + 0.35 * ((stuck as f64) / 120.0).clamp(0.0, 1.0)).clamp(1.0, 1.35);
        let learn_p = (learn_base * learn_boost).clamp(0.0, 0.60);
        let use_learn = rng.gen::<f64>() < learn_p;
        let target = Some(best_makespan.saturating_add(target_margin));
        let guidance: Guidance = if use_learn { Some((&learned_jb, &learned_mp)) } else { None };
        let (sol, mk) = construct_solution_conflict(challenge, pre, rule, k, target, &mut rng, guidance)?;
        let ridx = rule_idx(rule);
        rule_tries[ridx] = rule_tries[ridx].saturating_add(1);
        rule_best[ridx] = rule_best[ridx].min(mk);
        if record(&mut best_makespan, &mut best_solution, &sol, mk, save_solution)? {
            stuck = 0;
            if learn_updates_left > 0 {
                learned_jb = job_bias_from_solution(pre, &sol);
                learned_mp = machine_penalty_from_solution(pre, &sol, challenge.num_machines);
                learn_updates_left -= 1;
            }
        } else {
            stuck = stuck.saturating_add(1);
        }
        push_top_solutions(&mut top_solutions, &sol, mk, CONSTRUCT_POOL_CAP);
    }

    // Ten guided constructions per pooled solution, alternating its guidance and the inverse guidance.
    let mut refine_results: Vec<(Solution, u32)> = Vec::new();
    for (base_sol, _) in top_solutions.iter() {
        let jb = job_bias_from_solution(pre, base_sol);
        let mp = machine_penalty_from_solution(pre, base_sol, challenge.num_machines);
        let anti_jb = invert_job_bias_guidance(&jb);
        let anti_mp = invert_machine_penalty_guidance(&mp);
        let target_ls = Some(best_makespan.saturating_add(target_margin / 2));
        for attempt in 0..10 {
            let use_anti = attempt % 2 == 1;
            let rule = match attempt { 0 => r0, 1 => Rule::Adaptive, 2 => Rule::BnHeavy, 3 => Rule::EndTight, 4 => Rule::Regret, 5 => Rule::CriticalPath, 6 => Rule::EarlyEnd, 7 => Rule::MostWork, _ => r1 };
            let k = match attempt % 4 { 0 => 2, 1 => 3, 2 => 4, _ => 2 }.min(K_HI);
            let guidance: Guidance = if use_anti { Some((&anti_jb, &anti_mp)) } else { Some((&jb, &mp)) };
            let (sol, mk) = construct_solution_conflict(challenge, pre, rule, k, target_ls, &mut rng, guidance)?;
            record(&mut best_makespan, &mut best_solution, &sol, mk, save_solution)?;
            refine_results.push((sol, mk));
        }
    }
    for (sol, mk) in refine_results { push_top_solutions(&mut top_solutions, &sol, mk, CONSTRUCT_POOL_CAP); }

    let ts_iters = effort.job_shop_iters;
    let kick_spread = effort.js_ts_kick_spread;
    let ts_tenure = ((pre.total_ops as f64).sqrt() as usize * (100 + (pre.load_cv * 60.0) as usize) / 100).clamp(8, 24);

    // Tabu search from the best pooled solution and its farthest-first companions.
    {
        let ts_starts = top_solutions.len().min(effort.js_ts_starts.max(1));
        let pool_len = top_solutions.len();
        let keys: Vec<Vec<u32>> = top_solutions.iter()
            .map(|(s, _)| ts_seed_key(pre, challenge.num_machines, s))
            .collect();
        let mut picked: Vec<usize> = Vec::with_capacity(ts_starts);
        let mut taken = vec![false; pool_len];
        let mut min_d = vec![u32::MAX; pool_len];
        picked.push(0);
        taken[0] = true;
        while picked.len() < ts_starts {
            let last = *picked.last().unwrap();
            let mut best: Option<usize> = None;
            let mut best_d = 0u32;
            for i in 0..pool_len {
                if taken[i] { continue; }
                let d = ts_seed_key_distance(&keys[last], &keys[i]);
                if d < min_d[i] { min_d[i] = d; }
                if best.is_none() || min_d[i] > best_d {
                    best = Some(i);
                    best_d = min_d[i];
                }
            }
            let Some(b) = best else { break };
            picked.push(b);
            taken[b] = true;
        }
        let mut ts_results: Vec<(Solution, u32)> = Vec::new();
        for &idx in &picked {
            let res = tabu_search_phase(pre, challenge, &top_solutions[idx].0, ts_iters, ts_tenure, kick_spread, 0, true, save_solution, best_makespan)?;
            if let Some((sol2, mk2)) = res {
                record(&mut best_makespan, &mut best_solution, &sol2, mk2, save_solution)?;
                ts_results.push((sol2, mk2));
            }
        }
        for (sol2, mk2) in ts_results {
            push_top_solutions(&mut top_solutions, &sol2, mk2, POOL_CAP);
        }
    }

    // Relocation descent on the three most loaded machines of the six best solutions.
    {
        let bn_starts = top_solutions.len().min(6);
        let mut bn_buf: Option<EvalBuf> = None;
        let mut bn_results: Vec<(Solution, u32)> = Vec::new();
        let mut crit_pos: Vec<usize> = Vec::with_capacity(16);
        let mut move_pos: Vec<usize> = Vec::with_capacity(18);
        let mut machine_total_pt: Vec<u64> = vec![0u64; challenge.num_machines];
        let mut m_rank: Vec<usize> = Vec::with_capacity(challenge.num_machines);
        const RELOCATE_SPAN: usize = 2;
        const MAX_ROUNDS: usize = 2;
        for idx in 0..bn_starts {
            let mut ds = match build_disj_from_solution(pre, challenge, &top_solutions[idx].0) { Ok(d) => d, Err(_) => continue };
            let buf = bn_buf.get_or_insert_with(|| EvalBuf::new(ds.n));
            let Some((mut cur_mk, mut mk_node)) = eval_disj(&ds, buf) else { continue };

            machine_total_pt.fill(0);
            for m in 0..ds.machine_seq.len() {
                for &nd in &ds.machine_seq[m] {
                    machine_total_pt[m] = machine_total_pt[m].saturating_add(ds.node_pt[nd] as u64);
                }
            }
            m_rank.clear();
            for m in 0..ds.machine_seq.len() {
                if ds.machine_seq[m].len() > 1 { m_rank.push(m); }
            }
            m_rank.sort_by(|&a, &b| machine_total_pt[b].cmp(&machine_total_pt[a]));
            let num_bn_ls = m_rank.len().min(3);
            let mut any_improved = false;

            for _ in 0..MAX_ROUNDS {
                let mut round_improved = false;
                for bi in 0..num_bn_ls {
                    let m = m_rank[bi];
                    let seq_cap = ds.machine_seq[m].len().min(18);
                    if seq_cap < 2 { continue; }
                    crit_pos.clear();
                    {
                        let prefix = &ds.machine_seq[m][..seq_cap];
                        let mut u = mk_node;
                        while u != NONE_USIZE {
                            if let Some(pos) = prefix.iter().position(|&nd| nd == u) {
                                crit_pos.push(pos);
                            }
                            u = buf.best_pred[u];
                        }
                    }
                    if crit_pos.is_empty() { continue; }
                    crit_pos.sort_unstable();
                    crit_pos.dedup();
                    move_pos.clear();
                    for &pos in &crit_pos {
                        move_pos.push(pos);
                        if pos > 0 { move_pos.push(pos - 1); }
                        if pos + 1 < seq_cap { move_pos.push(pos + 1); }
                    }
                    move_pos.sort_unstable();
                    move_pos.dedup();

                    let mut best_move: Option<(usize, usize)> = None;
                    let mut best_move_mk = cur_mk;
                    for &from in &move_pos {
                        let lo = from.saturating_sub(RELOCATE_SPAN);
                        let hi = (from + RELOCATE_SPAN).min(seq_cap - 1);
                        for to in lo..=hi {
                            if to == from { continue; }
                            relocate_machine_seq(&mut ds.machine_seq[m][..seq_cap], from, to);
                            if let Some(new_mk) = eval_disj_trial(&ds, buf, best_move_mk) {
                                if new_mk < best_move_mk {
                                    best_move_mk = new_mk;
                                    best_move = Some((from, to));
                                }
                            }
                            relocate_machine_seq(&mut ds.machine_seq[m][..seq_cap], to, from);
                        }
                    }

                    let Some((from, to)) = best_move else {
                        let Some((restored_mk, restored_node)) = eval_disj(&ds, buf) else { continue };
                        cur_mk = restored_mk;
                        mk_node = restored_node;
                        continue;
                    };
                    relocate_machine_seq(&mut ds.machine_seq[m][..seq_cap], from, to);
                    match eval_disj(&ds, buf) {
                        Some((new_mk, new_node)) if new_mk < cur_mk => {
                            cur_mk = new_mk;
                            mk_node = new_node;
                            any_improved = true;
                            round_improved = true;
                        }
                        _ => {
                            relocate_machine_seq(&mut ds.machine_seq[m][..seq_cap], to, from);
                            let Some((restored_mk, restored_node)) = eval_disj(&ds, buf) else { continue };
                            cur_mk = restored_mk;
                            mk_node = restored_node;
                        }
                    }
                }
                if !round_improved { break; }
            }

            if any_improved {
                let sol_bn = disj_to_solution(pre, &ds, &buf.start);
                record(&mut best_makespan, &mut best_solution, &sol_bn, cur_mk, save_solution)?;
                bn_results.push((sol_bn, cur_mk));
            }
        }
        for (sol_bn, mk_bn) in bn_results {
            push_top_solutions(&mut top_solutions, &sol_bn, mk_bn, POOL_CAP);
        }
    }

    // Iterated local search on the six best solutions.
    {
        let ils_starts = top_solutions.len().min(6);
        let mut ils_results: Vec<(Solution, u32)> = Vec::new();
        for idx in 0..ils_starts {
            let ls_res = critical_block_move_local_search_ex(pre, challenge, &top_solutions[idx].0, 5, 400, 120);
            if let Ok(Some((ls_sol, ls_mk))) = ls_res {
                record(&mut best_makespan, &mut best_solution, &ls_sol, ls_mk, save_solution)?;
                ils_results.push((ls_sol.clone(), ls_mk));
                if let Some((sol3, mk3)) = perturb_and_reoptimize_ils(pre, challenge, &ls_sol, 2, 3, 220)? {
                    record(&mut best_makespan, &mut best_solution, &sol3, mk3, save_solution)?;
                    ils_results.push((sol3, mk3));
                }
            }
        }
        for (sol, mk) in ils_results {
            push_top_solutions(&mut top_solutions, &sol, mk, POOL_CAP);
        }
    }

    // Memetic recombination of non-bottleneck machine orders, then path relinking from the best
    // solution to the three best of the population, each followed by a short tabu search.
    {
        let num_machines = challenge.num_machines;
        let mut machine_rank: Vec<(usize, f64)> = (0..num_machines).map(|m| (m, pre.machine_load[m])).collect();
        machine_rank.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let num_bn = (num_machines / 5).max(3).min(8);
        let mut is_bottleneck = vec![false; num_machines];
        for i in 0..num_bn { is_bottleneck[machine_rank[i].0] = true; }

        let pop_cap = 10usize;
        let mut mem_pop: Vec<(Solution, u32)> = top_solutions.iter().take(pop_cap).cloned().collect();
        let mut mem_ds: Vec<Option<DisjSchedule>> = vec![None; mem_pop.len()];
        let num_generations = 18usize;
        let mut gen_no_improve = 0usize;
        let max_gen_no_improve = 12usize;
        let mut mem_buf: Option<EvalBuf> = None;
        let mut cross_pos: Vec<usize> = Vec::new();
        let mut cross_sum: Vec<u32> = Vec::new();
        let mut cross_cnt: Vec<u16> = Vec::new();
        let mut cross_pairs: Vec<(usize, usize, i32)> = Vec::new();
        let elite_cap = 4usize;
        for gen in 0..num_generations {
            if gen_no_improve >= max_gen_no_improve { break; }
            let cur_pop = mem_pop.len();
            if cur_pop < 2 { break; }
            let use_mutation = gen % 6 == 5;

            // Binary tournaments for two distinct parents.
            let ia = {
                let a = rng.gen_range(0..cur_pop);
                let b = rng.gen_range(0..cur_pop);
                if mem_pop[a].1 <= mem_pop[b].1 { a } else { b }
            };
            let ib = {
                let mut b = rng.gen_range(0..cur_pop);
                if b == ia { b = (b + 1) % cur_pop; }
                let c = rng.gen_range(0..cur_pop);
                let c = if c == ia { (c + 1) % cur_pop } else { c };
                if mem_pop[b].1 <= mem_pop[c].1 { b } else { c }
            };
            let mk_a = mem_pop[ia].1;
            let mk_b = mem_pop[ib].1;

            let mut elite_idx: Vec<usize> = (0..cur_pop).collect();
            elite_idx.sort_unstable_by_key(|&i| mem_pop[i].1);
            elite_idx.truncate(elite_cap.min(cur_pop));
            let mut build_failed = false;
            for &i in [ia, ib].iter().chain(elite_idx.iter()) {
                if mem_ds[i].is_none() {
                    match build_disj_from_solution(pre, challenge, &mem_pop[i].0) {
                        Ok(d) => mem_ds[i] = Some(d),
                        Err(_) => { build_failed = true; break; }
                    }
                }
            }
            if build_failed {
                gen_no_improve += 1;
                continue;
            }

            let (better_idx, worse_idx) = if mk_a <= mk_b { (ia, ib) } else { (ib, ia) };
            let mut child_ds = {
                let better_ds = mem_ds[better_idx].as_ref().unwrap();
                let worse_ds = mem_ds[worse_idx].as_ref().unwrap();
                let mut child_ds = better_ds.clone();
                if cross_pos.len() != child_ds.n {
                    cross_pos.resize(child_ds.n, 0);
                    cross_sum.resize(child_ds.n, 0);
                    cross_cnt.resize(child_ds.n, 0);
                }
                for m in 0..num_machines {
                    if is_bottleneck[m] { continue; }
                    if m >= better_ds.machine_seq.len() || m >= worse_ds.machine_seq.len() || m >= child_ds.machine_seq.len() { continue; }
                    if better_ds.machine_seq[m].len() != worse_ds.machine_seq[m].len() { continue; }
                    let mut elite_machine_seqs: Vec<&[usize]> = Vec::with_capacity(elite_idx.len());
                    for &ei in &elite_idx {
                        if let Some(ds_e) = mem_ds[ei].as_ref() {
                            if m < ds_e.machine_seq.len() && ds_e.machine_seq[m].len() == child_ds.machine_seq[m].len() {
                                elite_machine_seqs.push(&ds_e.machine_seq[m]);
                            }
                        }
                    }
                    inherit_machine_consensus_order(
                        &mut child_ds.machine_seq[m],
                        &better_ds.machine_seq[m],
                        &worse_ds.machine_seq[m],
                        &elite_machine_seqs,
                        &mut cross_pos,
                        &mut cross_sum,
                        &mut cross_cnt,
                        &mut cross_pairs,
                    );
                }
                child_ds
            };

            if use_mutation {
                let non_bn_machines: Vec<usize> = (0..num_machines).filter(|&m| !is_bottleneck[m] && child_ds.machine_seq[m].len() > 1).collect();
                if !non_bn_machines.is_empty() {
                    for _ in 0..2 {
                        let m = non_bn_machines[rng.gen_range(0..non_bn_machines.len())];
                        let pos = rng.gen_range(0..child_ds.machine_seq[m].len() - 1);
                        child_ds.machine_seq[m].swap(pos, pos + 1);
                    }
                }
                let bn_machines: Vec<usize> = (0..num_machines).filter(|&m| is_bottleneck[m] && child_ds.machine_seq[m].len() > 1).collect();
                if !bn_machines.is_empty() {
                    let m = bn_machines[rng.gen_range(0..bn_machines.len())];
                    let pos = rng.gen_range(0..child_ds.machine_seq[m].len() - 1);
                    child_ds.machine_seq[m].swap(pos, pos + 1);
                }
            }

            let child_buf = mem_buf.get_or_insert_with(|| EvalBuf::new(child_ds.n));
            let ls_mk = match critical_block_move_local_search_ex_disj(&mut child_ds, child_buf, 4, 250) {
                Some(mk) => mk,
                None => match eval_disj(&child_ds, child_buf) {
                    Some((mk, _)) => mk,
                    None => { gen_no_improve += 1; continue; }
                },
            };
            let ls_sol = disj_to_solution(pre, &child_ds, &child_buf.start);
            if record(&mut best_makespan, &mut best_solution, &ls_sol, ls_mk, save_solution)? {
                gen_no_improve = 0;
            } else {
                gen_no_improve += 1;
            }
            push_top_solutions(&mut top_solutions, &ls_sol, ls_mk, POOL_CAP);

            if cur_pop >= pop_cap {
                let worst_idx = mem_pop.iter().enumerate().max_by_key(|(_, (_, mk))| *mk).map(|(i, _)| i).unwrap_or(cur_pop - 1);
                if ls_mk < mem_pop[worst_idx].1 {
                    mem_pop[worst_idx] = (ls_sol, ls_mk);
                    mem_ds[worst_idx] = Some(child_ds);
                }
            } else {
                mem_pop.push((ls_sol, ls_mk));
                mem_ds.push(Some(child_ds));
            }
        }

        let mem_best: Vec<Solution> = {
            let mut sorted = mem_pop.clone();
            sorted.sort_by_key(|(_, mk)| *mk);
            sorted.into_iter().take(3).map(|(s, _)| s).collect()
        };
        for base_sol in &mem_best {
            let seed: Solution = match path_relink_midpoint(pre, challenge, &best_solution, base_sol)? {
                Some((mid, mid_mk)) => {
                    record(&mut best_makespan, &mut best_solution, &mid, mid_mk, save_solution)?;
                    mid
                }
                None => base_sol.clone(),
            };
            if let Some((ts_sol, ts_mk)) = tabu_search_phase(pre, challenge, &seed, ts_iters / 3, ts_tenure, kick_spread, 0, true, save_solution, best_makespan)? {
                record(&mut best_makespan, &mut best_solution, &ts_sol, ts_mk, save_solution)?;
                push_top_solutions(&mut top_solutions, &ts_sol, ts_mk, POOL_CAP);
            }
        }
    }

    // Two shifting-bottleneck seeds, each polished by a full tabu search; the better one competes with the best.
    const SBP_SALT: u64 = 0x3C79AC492BA7B653;
    let mut sbp_seed_best: Option<(Solution, u32)> = None;
    if let Some((seed_sol, _)) = sbp_construct_seed(pre, challenge, &greedy_sol, effort.js_seed_sbp_reopt_cycles) {
        sbp_seed_best = tabu_search_phase(pre, challenge, &seed_sol, ts_iters, ts_tenure, kick_spread, SBP_SALT, true, save_solution, best_makespan)?;
    }
    let dual_cycles = effort.js_seed_sbp_dual_cycles;
    if dual_cycles > 0 && dual_cycles != effort.js_seed_sbp_reopt_cycles {
        if let Some((seed_sol2, _)) = sbp_construct_seed(pre, challenge, &greedy_sol, dual_cycles) {
            if let Some((sol6, mk6)) = tabu_search_phase(pre, challenge, &seed_sol2, ts_iters, ts_tenure, kick_spread, SBP_SALT, true, save_solution, best_makespan)? {
                let better = match sbp_seed_best.as_ref() { Some((_, cur_mk)) => mk6 < *cur_mk, None => true };
                if better { sbp_seed_best = Some((sol6, mk6)); }
            }
        }
    }
    if let Some((sbp_sol, sbp_mk)) = sbp_seed_best {
        if sbp_mk < best_makespan { best_makespan = sbp_mk; best_solution = sbp_sol; }
    }

    // Final tabu search on the best solution, then a shifting-bottleneck re-optimisation.
    let mut final_mk = best_makespan;
    if let Some((sol4, mk4)) = tabu_search_phase(pre, challenge, &best_solution, ts_iters, ts_tenure, kick_spread, 0, false, save_solution, best_makespan)? {
        if mk4 < best_makespan { save_solution(&sol4)?; best_solution = sol4; final_mk = mk4; }
    }
    if let Some((sbp_sol, sbp_mk)) = sbp_reoptimize(pre, challenge, &best_solution, effort.js_sbp_rounds) {
        if sbp_mk < final_mk { best_solution = sbp_sol; }
    }
    save_solution(&best_solution)?;
    Ok(())
}
}

pub mod solver {
use anyhow::Result;
use tig_challenges::job_scheduling::*;

use super::types::EffortConfig;
use super::preprocess::build_pre;
use super::job_shop;

pub fn solve_challenge(challenge: &Challenge, save_solution: &dyn Fn(&Solution) -> Result<()>) -> Result<()> {
    let pre = build_pre(challenge)?;
    job_shop::solve(challenge, save_solution, &pre, &EffortConfig::DEFAULT)
}
}
