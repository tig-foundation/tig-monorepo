fn lower_bound(pre: &types::Pre, machines: usize) -> u64 {
    let mut load = vec![0u64; machines];
    let mut longest = 0u64;
    for &product in &pre.job_products {
        let mut total = 0u64;
        for op in &pre.product_ops[product] {
            load[op.machine] += op.pt as u64;
            total += op.pt as u64;
        }
        longest = longest.max(total);
    }
    let mut critical = 0usize;
    for m in 1..machines { if load[m] > load[critical] { critical = m; } }
    if machines == 0 { return longest; }
    let mut lb = longest.max(load[critical]);
    let mut tasks: Vec<(u64, u64, u64, usize)> = Vec::new();
    for &product in &pre.job_products {
        let ops = &pre.product_ops[product];
        let total: u64 = ops.iter().map(|o| o.pt as u64).sum();
        let mut head = 0u64;
        for op in ops {
            let p = op.pt as u64;
            if op.machine == critical { tasks.push((head, p, total - head - p, tasks.len())); }
            head += p;
        }
    }
    tasks.sort_unstable_by_key(|&(r, _, _, id)| (r, id));
    let mut ready = std::collections::BinaryHeap::new();
    let mut i = 0usize;
    let mut t = 0u64;
    while i < tasks.len() || !ready.is_empty() {
        if ready.is_empty() && i < tasks.len() { t = t.max(tasks[i].0); }
        while i < tasks.len() && tasks[i].0 <= t {
            let (_, p, q, id) = tasks[i];
            ready.push((q, std::cmp::Reverse(id), p));
            i += 1;
        }
        let Some((q, id, p)) = ready.pop() else { break };
        if i < tasks.len() && tasks[i].0 < t + p {
            let next = tasks[i].0;
            ready.push((q, id, p - (next - t)));
            t = next;
        } else {
            t += p;
            lb = lb.max(t + q);
        }
    }
    lb
}

struct BoundState {
    lb: u64,
}

impl BoundState {
    fn new(pre: &types::Pre, machines: usize) -> Self {
        Self { lb: lower_bound(pre, machines) }
    }

    fn offer(&self, pre: &types::Pre, sol: &tig_challenges::job_scheduling::Solution) -> bool {
        let mut mk = 0u64;
        for (j, ops) in sol.job_schedule.iter().enumerate() {
            let product = pre.job_products[j];
            for (o, &(_, start)) in ops.iter().enumerate() {
                mk = mk.max(start as u64 + pre.product_ops[product][o].pt as u64);
            }
        }
        mk == self.lb
    }
}

#[derive(Debug)]
struct BoundReached;
impl std::fmt::Display for BoundReached {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("bound reached") }
}
impl std::error::Error for BoundReached {}
const START_ITERS: usize = 70_000;
const SEED_ITERS: usize = 140_000;
const FINAL_ITERS: usize = 70_000;
const SBP_INC_ITERS: usize = 23_333;
const DEEP_ITERS: usize = 560_000;
const DIVERSE_STARTS: usize = 3;
const SBP_NODE_BUDGET: usize = 1_200;
const NUM_RESTARTS: usize = 150;
const BASELINE_TRIES: usize = 20;
const REFINE_TRIES: usize = 20;
const SBP_SEED_CYCLES: [usize; 2] = [4, 10];
const SBP_INC_ROUNDS: usize = 4;

mod types {
pub const NONE_USIZE: usize = usize::MAX;
pub const NONE_U32: u32 = u32::MAX;

pub const FLEX_FACTOR: f64 = 2.2;

#[derive(Clone, Copy)]
pub struct OpInfo {
    pub machine: usize,
    pub pt: u32,
    pub bn_avg: f64,
}

pub struct Pre {
    pub job_products: Vec<usize>,
    pub job_ops_len: Vec<usize>,
    pub product_ops: Vec<Vec<OpInfo>>,
    pub product_suf_min: Vec<Vec<u32>>,
    pub product_suf_bn: Vec<Vec<f64>>,
    pub product_next_min: Vec<Vec<u32>>,
    pub product_next_flag: Vec<Vec<f64>>,
    pub machine_load: Vec<f64>,
    pub avg_machine_load: f64,
    pub avg_op_min: f64,
    pub time_scale: f64,
    pub max_ops: usize,
    pub max_job_work: f64,
    pub max_job_bn: f64,
    pub reg_n: f64,
    pub flow_w: f64,
    pub job_flow_pref: Vec<f64>,
    pub jobshopness: f64,
    pub bn_focus: f64,
    pub load_cv: f64,
    pub slack_base: f64,
    pub total_ops: usize,
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

#[derive(Clone)]
pub struct DisjSchedule {
    pub n: usize,
    pub num_jobs: usize,
    pub num_machines: usize,
    pub job_offsets: Vec<usize>,
    pub job_succ: Vec<usize>,
    pub indeg_job: Vec<u16>,
    pub indeg_job_plus1: Vec<u16>,
    pub node_machine: Vec<usize>,
    pub node_pt: Vec<u32>,
    pub node_job: Vec<usize>,
    pub node_op: Vec<usize>,
    pub machine_seq: Vec<Vec<usize>>,
}

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
    pub stack: Vec<usize>,
    pub nd: Vec<NodeRec>,
}

impl EvalBuf {
    pub fn new(n: usize) -> Self {
        let empty = NodeRec { pt: 0, job_succ: NONE_U32, machine_succ: NONE_U32, indeg: 0, best_pred: NONE_U32, machine_pred: NONE_U32 };
        Self {
            indeg: vec![0u16; n + 1],
            start: vec![0u32; n + 1],
            best_pred: vec![NONE_USIZE; n + 1],
            machine_succ: vec![NONE_USIZE; n],
            stack: vec![0usize; n],
            nd: vec![empty; n + 1],
        }
    }
}

}
mod infra_shared {
use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::job_scheduling::*;
use super::types::*;

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
    for _ in 0..super::BASELINE_TRIES {
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

#[inline(always)]
fn greedy_priority(rule: GreedyRule, work_left: f64, ops_left: usize, pt: u32) -> f64 {
    match rule {
        GreedyRule::MostWork => work_left,
        GreedyRule::MostOps => ops_left as f64,
        GreedyRule::EarliestEnd => 0.0,
        GreedyRule::ShortestProc => -(pt as f64),
        GreedyRule::LongestProc => pt as f64,
    }
}

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
    let mut candidates: Vec<GCandidate> = Vec::with_capacity(num_jobs);

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

fn prepare_eval(ds: &DisjSchedule, buf: &mut EvalBuf) {
    buf.indeg[..ds.n].copy_from_slice(&ds.indeg_job_plus1);
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
    // SAFETY: start, indeg and best_pred have n + 1 slots; absent links map to sentinel n; operation arrays have n slots.
    unsafe {
        let start = buf.start.as_mut_ptr();
        let indeg = buf.indeg.as_mut_ptr();
        let stack = buf.stack.as_mut_ptr();
        let bp = buf.best_pred.as_mut_ptr();
        let msucc = buf.machine_succ.as_ptr();
        let jsucc = ds.job_succ.as_ptr();
        let pt = ds.node_pt.as_ptr();
        while sl > 0 {
            sl -= 1;
            let u = *stack.add(sl);
            processed += 1;
            let end_u = (*start.add(u)).saturating_add(*pt.add(u));
            let up = end_u > mk;
            mk_node = if up { u } else { mk_node };
            mk = mk.max(end_u);
            for s in [(*jsucc.add(u)).min(n), (*msucc.add(u)).min(n)] {
                let t = *start.add(s) < end_u;
                *start.add(s) = (*start.add(s)).max(end_u);
                *bp.add(s) = if t { u } else { *bp.add(s) };
                let d = (*indeg.add(s)).wrapping_sub(1);
                *indeg.add(s) = d;
                *stack.add(sl) = s;
                sl += (d == 0) as usize;
                *indeg.add(n) = u16::MAX;
            }
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

pub fn push_top_solutions(top: &mut Vec<(Solution, u32)>, sol: &Solution, mk: u32, cap: usize) {
    let pos = top.binary_search_by_key(&mk, |(_, m)| *m).unwrap_or_else(|e| e);
    top.insert(pos, (sol.clone(), mk));
    if top.len() > cap { top.truncate(cap); }
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

struct CandCtx {
    job: usize,
    product: usize,
    op_idx: usize,
    ops_rem: usize,
    pt: u32,
    time: u32,
    target_mk: Option<u32>,
    end: u32,
    progress: f64,
    job_bias: f64,
    machine_penalty: f64,
    dynamic_load: f64,
    jitter: f64,
}

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
    let mut raw_by_machine: Vec<Vec<Cand>> = (0..num_machines).map(|_| Vec::with_capacity(13)).collect();
    let mut idle_machines: Vec<usize> = (0..num_machines).collect();
    let mut idle_pos: Vec<usize> = (0..num_machines).collect();
    let mut busy_machine_heap: std::collections::BinaryHeap<std::cmp::Reverse<(u32, usize)>> = std::collections::BinaryHeap::with_capacity(num_machines);
    let mut blocked_job_heap: std::collections::BinaryHeap<std::cmp::Reverse<(u32, usize)>> = std::collections::BinaryHeap::with_capacity(num_jobs);
    let mut touched_machines: Vec<usize> = Vec::with_capacity(num_machines);
    let mut top: Vec<Cand> = if k > 0 { Vec::with_capacity(k + 1) } else { Vec::new() };
    let mut ready_by_machine: Vec<Vec<(usize, u32, u32)>> = (0..num_machines).map(|_| Vec::with_capacity(32)).collect();
    let mut job_gen: Vec<u32> = vec![0u32; num_jobs];
    let rigidity = (0.60 + 0.40_f64).clamp(0.0, 2.5);
    let regc = pre.reg_n.clamp(0.0, 4.5);
    let conflict_scale = (0.90 + 0.40 * FLEX_FACTOR).clamp(0.85, 1.75);
    let cap_per_machine = if k == 0 { 12usize } else { (k + 6).min(12) };
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
                let boost = conflict_w * conflict_scale * dem_n * (1.15 * rigidity + 0.85 * regc);
                for rc in &raw_by_machine[m] {
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

#[inline]
fn hg_eval_nd(hg: &mut HGraph, node_pt: &[u32], sinks: &[usize]) -> Option<(u32, usize, bool)> {
    let mk = hg.evaluate_js()?;
    hg.compute_tails_js();
    let mut mk_node = NONE_USIZE;
    let mut hits = 0usize;
    // SAFETY: sinks are operation ids below head.len() == node_pt.len().
    for &u in sinks {
        let hit = unsafe { *hg.head.get_unchecked(u) }.saturating_add(unsafe { *node_pt.get_unchecked(u) }) == mk;
        mk_node = if hit & (mk_node == NONE_USIZE) { u } else { mk_node };
        hits += hit as usize;
    }
    Some((mk, mk_node, hits <= 1))
}

#[inline]
fn hg_sinks(ds: &DisjSchedule) -> Vec<usize> {
    (0..ds.n).filter(|&u| ds.job_succ[u] == NONE_USIZE).collect()
}

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

// SAFETY: u and every non-sentinel link are operation ids below head.len() == job_pred.len(); nd has one additional sentinel.
#[inline(always)]
fn bp_of(nd: &[NodeRec], job_pred: &[usize], u: usize, head: &[u32]) -> usize {
    let (jt, mt, jp, mp) = tight_preds(nd, job_pred, u, head);
    if jt { jp } else if mt { mp } else { NONE_USIZE }
}

// SAFETY: as bp_of; absent links are read through the clamped id n - 1 and masked.
#[inline(always)]
fn tight_preds(nd: &[NodeRec], job_pred: &[usize], u: usize, head: &[u32]) -> (bool, bool, usize, usize) {
    let last = head.len() - 1;
    let h = unsafe { *head.get_unchecked(u) };
    let jp = unsafe { *job_pred.get_unchecked(u) };
    let jpc = jp.min(last);
    let jt = (jp != NONE_USIZE) & (unsafe { *head.get_unchecked(jpc) }.saturating_add(unsafe { nd.get_unchecked(jpc) }.pt) == h);
    let mp32 = unsafe { nd.get_unchecked(u) }.machine_pred;
    let mp = n2u(mp32);
    let mpc = mp.min(last);
    let mt = (mp32 != NONE_U32) & (unsafe { *head.get_unchecked(mpc) }.saturating_add(unsafe { nd.get_unchecked(mpc) }.pt) == h);
    (jt, mt, jp, mp)
}

// SAFETY: mk_node and every non-sentinel predecessor are operation ids below head.len() == job_pred.len(); nd has one additional sentinel.
#[inline(always)]
fn settle_path(ds: &DisjSchedule, buf: &mut EvalBuf, indeg_full: &[u16], job_pred: &[usize], mk_node: usize, single: bool, head: &[u32], mk: u32) -> Option<(usize, bool)> {
    let mut clear = single;
    if clear && mk_node != NONE_USIZE {
        let nd = &buf.nd;
        let mut u = mk_node;
        loop {
            let (jt, mt, jp, mp) = tight_preds(nd, job_pred, u, head);
            if jt == mt { clear = !jt; break; }
            u = if jt { jp } else { mp };
        }
    }
    if clear { return Some((mk_node, false)); }
    let node = settle_topo_nd(ds, buf, indeg_full, head, mk)?;
    Some((node, true))
}

// SAFETY: nd has n + 1 slots; other arrays have n. Links are ids or sentinels, and sl < n after a pop.
fn settle_topo_nd(ds: &DisjSchedule, buf: &mut EvalBuf, indeg_full: &[u16], head: &[u32], mk: u32) -> Option<usize> {
    let n = ds.n;
    let nd = &mut buf.nd;
    for u in 0..n {
        let r = unsafe { nd.get_unchecked_mut(u) };
        r.indeg = unsafe { *indeg_full.get_unchecked(u) } as u32;
        r.best_pred = NONE_U32;
    }
    nd[n].indeg = u32::MAX;
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
        mk_node = if (end_u == mk) & (mk_node == NONE_USIZE) { u } else { mk_node };
        settle_relax(nd, head, &mut buf.stack, &mut sl, u, end_u, (ru.job_succ as usize).min(n), n);
        settle_relax(nd, head, &mut buf.stack, &mut sl, u, end_u, (ru.machine_succ as usize).min(n), n);
    }
    if popped != n { return None; }
    Some(mk_node)
}

#[inline(always)]
fn settle_relax(nd: &mut [NodeRec], head: &[u32], stack: &mut [usize], sl: &mut usize, u: usize, end_u: u32, s: usize, n: usize) {
    let r = unsafe { nd.get_unchecked_mut(s) };
    let tight = (end_u == unsafe { *head.get_unchecked(s.min(n - 1)) }) & (r.best_pred == NONE_U32);
    r.best_pred = if tight { u as u32 } else { r.best_pred };
    r.indeg = r.indeg.wrapping_sub(1);
    unsafe { *stack.get_unchecked_mut(*sl) = s; }
    *sl += (r.indeg == 0) as usize;
}

// SAFETY: u, v and their non-sentinel links are operation ids below the lengths of nd, head, qend and job_pred.
#[inline(always)]
fn estimate_swap_nd(u: usize, v: usize, nd: &[NodeRec], job_pred: &[usize], head: &[u32], qend: &[u32]) -> u32 {
    let rend = |i: usize| -> u32 { unsafe { *head.get_unchecked(i) }.saturating_add(unsafe { nd.get_unchecked(i) }.pt) };
    let q_at = |i: usize| -> u32 { unsafe { *qend.get_unchecked(i) } };
    let last = head.len() - 1;
    let ru = unsafe { *nd.get_unchecked(u) };
    let rv = unsafe { *nd.get_unchecked(v) };
    let jp_u = unsafe { *job_pred.get_unchecked(u) };
    let jp_v = unsafe { *job_pred.get_unchecked(v) };
    let r_jp_v = rend(jp_v.min(last)) & 0u32.wrapping_sub((jp_v != NONE_USIZE) as u32);
    let r_mp_u = rend((ru.machine_pred as usize).min(last)) & 0u32.wrapping_sub((ru.machine_pred != NONE_U32) as u32);
    let new_r_v = r_jp_v.max(r_mp_u);
    let r_jp_u = rend(jp_u.min(last)) & 0u32.wrapping_sub((jp_u != NONE_USIZE) as u32);
    let new_r_u = r_jp_u.max(new_r_v.saturating_add(rv.pt));
    let q_js_u = q_at((ru.job_succ as usize).min(last)) & 0u32.wrapping_sub((ru.job_succ != NONE_U32) as u32);
    let q_ms_v = q_at((rv.machine_succ as usize).min(last)) & 0u32.wrapping_sub((rv.machine_succ != NONE_U32) as u32);
    let new_q_u = q_js_u.max(q_ms_v);
    let q_js_v = q_at((rv.job_succ as usize).min(last)) & 0u32.wrapping_sub((rv.job_succ != NONE_U32) as u32);
    let new_q_v = q_js_v.max(ru.pt.saturating_add(new_q_u));
    let path_v = new_r_v.saturating_add(rv.pt).saturating_add(new_q_v);
    let path_u = new_r_u.saturating_add(ru.pt).saturating_add(new_q_u);
    path_v.max(path_u)
}


#[inline]
fn rebuild_machine_pos_map(machine_seq: &[Vec<usize>], node_pos: &mut [usize]) {
    for seq in machine_seq {
        for (pos, &node) in seq.iter().enumerate() {
            node_pos[node] = pos;
        }
    }
}

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
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        d += (ai != bi) as u32;
    }
    d
}

#[inline(always)]
fn tie_break(tie_n: &mut u64, tseed: &mut u64) -> bool {
    *tie_n = tie_n.saturating_add(1);
    *tseed ^= tseed.wrapping_shl(13);
    *tseed ^= tseed.wrapping_shr(7);
    *tseed ^= tseed.wrapping_shl(17);
    (*tseed % (*tie_n).max(1)) == 0
}

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

#[inline(always)]
// SAFETY: positions is non-empty and holds indices into seq; seq holds operation ids below head.len() == node_pt.len().
fn for_each_block(positions: &[usize], seq: &[usize], head: &[u32], node_pt: &[u32], mut f: impl FnMut(usize, usize)) {
    let rend = |node: usize| -> u32 { unsafe { (*head.get_unchecked(node)).saturating_add(*node_pt.get_unchecked(node)) } };
    let mut run_start = positions[0];
    let mut prev_pos = run_start;
    let mut prev_end = rend(unsafe { *seq.get_unchecked(prev_pos) });
    for &pos in &positions[1..] {
        let node = unsafe { *seq.get_unchecked(pos) };
        let extend = (pos == prev_pos + 1) & (unsafe { *head.get_unchecked(node) } == prev_end);
        if !extend {
            if prev_pos > run_start { f(run_start, prev_pos); }
            run_start = pos;
        }
        prev_pos = pos;
        prev_end = rend(node);
    }
    if prev_pos > run_start { f(run_start, prev_pos); }
}

struct TabuMirror {
    current_pos: Vec<usize>,
    node_machine: Vec<usize>,
    machine_pred_node: Vec<usize>,
    indeg_full: Vec<u16>,
}

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

#[inline(always)]
fn tabu_search_phase(
    bound: &super::BoundState,
    pre: &Pre, challenge: &Challenge, base_sol: &Solution, max_iterations: usize, tenure: usize,
    stall_stop: usize,
    publish: &dyn Fn(&Solution) -> Result<()>, published_mk: u32,
) -> Result<Option<(Solution, u32)>> {
    tabu_search_phase_impl::<true>(bound, pre, challenge, base_sol, max_iterations, tenure, stall_stop, publish, published_mk)
}

fn tabu_search_phase_impl<const RESTORE: bool>(
    bound: &super::BoundState,
    pre: &Pre, challenge: &Challenge, base_sol: &Solution, max_iterations: usize, tenure: usize,
    stall_stop: usize,
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
    let mut tabu_expiry = vec![0usize; pair_offsets[ds.num_machines]];
    let mut no_improve = 0usize;
    let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (initial_mk as u64).wrapping_shl(16) ^ (n as u64).wrapping_mul(0x517CC1B727220A95);
    let mut tseed: u64 = ((challenge.seed[0] as u64).wrapping_mul(0xD1B54A32D192ED03)
        ^ (initial_mk as u64).wrapping_mul(0xA24BAED4963EE407)
        ^ (n as u64).wrapping_shl(32)) | 1;
    let mut job_pred_node = vec![NONE_USIZE; n];
    for j in 0..ds.num_jobs { let base = ds.job_offsets[j]; let end = ds.job_offsets[j + 1]; for k in (base + 1)..end { job_pred_node[k] = k - 1; } }
    let crit_ml = ds.machine_seq.iter().map(|s| s.len()).max().unwrap_or(0).max(1);
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
    let kick_step = (max_no_improve / 4).max(1);
    let mut kicks_left = 4usize;
    let mut stall_best = initial_mk;
    let mut stall_last_gain = 0usize;
    for iter in 0..max_iterations {
        if cur_mk as u64 == bound.lb {
            let sol = disj_to_solution(pre, &ds, &hg.head);
            publish(&sol)?;
            return Ok(Some((sol, cur_mk)));
        }
        if stall_stop > 0 {
            if cur_mk < stall_best {
                stall_best = cur_mk;
                stall_last_gain = iter;
            }
            if iter - stall_last_gain >= stall_stop { break; }
        }
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
            let ref_first = &node_local[..];
            let ref_second = &best_global_pos[..];
            collect_guided_kick_moves(&ds, &buf.best_pred, mk_node, &tm.current_pos, &tm.node_machine, ref_first, &mut guided_used_machine, &mut guided_kicks);
            let ref_pos = if guided_kicks.is_empty() {
                collect_guided_kick_moves(&ds, &buf.best_pred, mk_node, &tm.current_pos, &tm.node_machine, ref_second, &mut guided_used_machine, &mut guided_kicks);
                ref_second
            } else {
                ref_first
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
            let ev = reload_mirror(&ds, &mut buf, &mut tm, &mut hg, &hm, &sinks);
            let Some((mk, node, single)) = ev else {
                if !RESTORE { break; }
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
        if exact {
            while u != NONE_USIZE {
                let m = unsafe { *tm.node_machine.get_unchecked(u) };
                let c = unsafe { *crit_cnt.get_unchecked(m) };
                unsafe { *crit_pos_flat.get_unchecked_mut((m + 1) * crit_ml - 1 - c) = *tm.current_pos.get_unchecked(u); }
                unsafe { *crit_cnt.get_unchecked_mut(m) = c + 1; }
                u = n2u(unsafe { buf.nd.get_unchecked(u) }.best_pred);
            }
        } else {
            while u != NONE_USIZE {
                let m = unsafe { *tm.node_machine.get_unchecked(u) };
                let c = unsafe { *crit_cnt.get_unchecked(m) };
                unsafe { *crit_pos_flat.get_unchecked_mut((m + 1) * crit_ml - 1 - c) = *tm.current_pos.get_unchecked(u); }
                unsafe { *crit_cnt.get_unchecked_mut(m) = c + 1; }
                u = bp_of(&buf.nd, &job_pred_node, u, &hg.head);
            }
        }
        let diversify = ((no_improve.saturating_sub(diversify_threshold) as f64) / (max_no_improve.saturating_sub(diversify_threshold).max(1) as f64)).clamp(0.0, 1.0);
        let mut sel = MoveSel::new();
        // SAFETY: m < num_machines, cnt <= crit_ml, pos + 1 < seq.len(); node ids < n and pair indices < pair_offsets[m + 1].
        for m in 0..ds.num_machines {
            let cnt = unsafe { *crit_cnt.get_unchecked(m) };
            if cnt < 2 { continue; }
            let end = (m + 1) * crit_ml;
            let positions = unsafe { crit_pos_flat.get_unchecked(end - cnt..end) };
            let seq = unsafe { ds.machine_seq.get_unchecked(m) };
            let age = iter.saturating_sub(unsafe { *machine_last_pick.get_unchecked(m) }).min(9) as u32;
            let pair_base = unsafe { *pair_offsets.get_unchecked(m) };
            for_each_block(positions, seq, &hg.head, &ds.node_pt, |run_start, run_end| {
                let block_len = run_end - run_start + 1;
                let num_swaps = 1 + (block_len >= 3) as usize;
                let block_bonus = block_len.saturating_sub(2).min(3) as u32;
                let div_bonus = if diversify > 0.0 {
                    ((diversify_unit as f64) * diversify * (0.35 * (age as f64) + 0.75 * (block_bonus as f64)) / 3.0) as u32
                } else { 0 };
                for k in 0..num_swaps {
                    let pos = if k == 0 { run_start } else { run_end - 1 };
                    let node_u = unsafe { *seq.get_unchecked(pos) };
                    let node_v = unsafe { *seq.get_unchecked(pos + 1) };
                    let est_mk = estimate_swap_nd(node_u, node_v, &buf.nd, &job_pred_node, &hg.head, &hg.qend);
                    let lu = unsafe { *node_local.get_unchecked(node_u) };
                    let lv = unsafe { *node_local.get_unchecked(node_v) };
                    let a = lu.min(lv);
                    let b = lu.max(lv);
                    let tabu_idx = pair_base + b * (b - 1) / 2 + a;
                    let is_tabu = unsafe { *tabu_expiry.get_unchecked(tabu_idx) } > iter;
                    let aspiration = est_mk < best_global_mk;
                    let adj_mk = est_mk.saturating_sub(div_bonus);
                    sel.offer(m, pos, est_mk, adj_mk, !is_tabu | aspiration, &mut tseed);
                }
            });
        }
        let Some((m, pos, _)) = sel.best.or(sel.relaxed) else { break };
        let (node_a, node_b) = unsafe {
            let seq = ds.machine_seq.get_unchecked_mut(m);
            let node_a = *seq.get_unchecked(pos);
            let node_b = *seq.get_unchecked(pos + 1);
            *seq.get_unchecked_mut(pos) = node_b;
            *seq.get_unchecked_mut(pos + 1) = node_a;
            *tm.current_pos.get_unchecked_mut(node_a) = pos + 1;
            *tm.current_pos.get_unchecked_mut(node_b) = pos;
            *machine_last_pick.get_unchecked_mut(m) = iter;
            let prev = if pos > 0 { *seq.get_unchecked(pos - 1) } else { NONE_USIZE };
            let next = if pos + 2 < seq.len() { *seq.get_unchecked(pos + 2) } else { NONE_USIZE };
            *tm.machine_pred_node.get_unchecked_mut(node_b) = prev;
            *tm.machine_pred_node.get_unchecked_mut(node_a) = node_b;
            buf.nd.get_unchecked_mut(node_b).machine_pred = u2n(prev);
            buf.nd.get_unchecked_mut(node_a).machine_pred = node_b as u32;
            if next != NONE_USIZE {
                *tm.machine_pred_node.get_unchecked_mut(next) = node_a;
                buf.nd.get_unchecked_mut(next).machine_pred = node_a as u32;
            }
            if prev != NONE_USIZE {
                *buf.machine_succ.get_unchecked_mut(prev) = node_b;
                buf.nd.get_unchecked_mut(prev).machine_succ = node_b as u32;
            }
            *buf.machine_succ.get_unchecked_mut(node_b) = node_a;
            buf.nd.get_unchecked_mut(node_b).machine_succ = node_a as u32;
            *buf.machine_succ.get_unchecked_mut(node_a) = next;
            buf.nd.get_unchecked_mut(node_a).machine_succ = u2n(next);
            let first = (pos == 0) as u16;
            let a = tm.indeg_full.get_unchecked_mut(node_a);
            *a = a.saturating_add(first);
            let b = tm.indeg_full.get_unchecked_mut(node_b);
            *b = b.saturating_sub(first);
            (node_a, node_b)
        };
        pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
        let offset = (pseed % ((2 * tenure_delta + 1) as u64)) as usize;
        let progress = (iter as f64) / (max_iterations as f64);
        let late_bonus = if progress > 0.6 { ((progress - 0.6) * 10.0) as usize } else { 0 };
        let this_tenure = (tenure + offset + late_bonus).saturating_sub(tenure_delta);
        unsafe {
            let la = *node_local.get_unchecked(node_a);
            let lb = *node_local.get_unchecked(node_b);
            let (a, b) = (la.min(lb), la.max(lb));
            let idx = *pair_offsets.get_unchecked(m) + b * (b - 1) / 2 + a;
            *tabu_expiry.get_unchecked_mut(idx) = iter + this_tenure;
        }
        hg.swap_adjacent(node_a as u32, node_b as u32);
        let ev = hg_eval_nd(&mut hg, &ds.node_pt, &sinks);
        let (mk, node, single) = match ev {
            Some(value) => value,
            None => {
                ds.machine_seq.clone_from(&best_global_machine_seq);
                let Some(value) = reload_mirror(&ds, &mut buf, &mut tm, &mut hg, &hm, &sinks) else { break };
                value
            }
        };
        cur_mk = mk; mk_node = node; mk_single = single;
    }
    if cur_mk < best_global_mk { best_global_mk = cur_mk; best_global_machine_seq.clone_from(&ds.machine_seq); }
    if best_global_mk >= initial_mk { return Ok(None); }
    ds.machine_seq = best_global_machine_seq;
    let Some((mk_final, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    Ok(Some((disj_to_solution(pre, &ds, &buf.start), mk_final)))
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

use super::{SBP_NODE_BUDGET, NUM_RESTARTS};

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
            deg: vec![0u16; n + 1], stack: vec![0usize; n + 1],
            head: vec![0u32; n + 1], tail: vec![0u32; n + 1],
            nodes: Vec::new(), r: Vec::new(), p: Vec::new(), q: Vec::new(),
        }
    }

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

struct CarlierBuf {
    done: Vec<bool>,
    order: Vec<usize>,
    best_order: Vec<usize>,
    starts: Vec<u32>,
}

impl CarlierBuf {
    fn new() -> Self {
        Self { done: Vec::new(), order: Vec::new(), best_order: Vec::new(), starts: Vec::new() }
    }
}

#[inline(always)]
fn sbp_relax(values: &mut [u32], deg: &mut [u16], stack: &mut [usize], sl: &mut usize, next: usize, value: u32, n: usize) {
    let s = next.min(n);
    unsafe {
        let v = values.get_unchecked_mut(s);
        *v = (*v).max(value);
        let d = deg.get_unchecked_mut(s);
        *d = d.saturating_sub(1);
        *stack.get_unchecked_mut(*sl) = s;
        *sl += (*d == 0) as usize;
        *deg.get_unchecked_mut(n) = u16::MAX;
    }
}

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
    let mut sl = 0usize;
    b.deg[n] = u16::MAX;
    for u in 0..n {
        let d = ds.indeg_job[u].saturating_add((b.mpred[u] != NONE_USIZE) as u16);
        b.deg[u] = d;
        b.head[u] = 0;
        unsafe { *b.stack.get_unchecked_mut(sl) = u; }
        sl += (d == 0) as usize;
    }
    let mut processed = 0usize;
    while sl > 0 {
        sl -= 1;
        let u = unsafe { *b.stack.get_unchecked(sl) };
        processed += 1;
        let end_u = b.head[u].saturating_add(ds.node_pt[u]);
        for s in [ds.job_succ[u], b.msucc[u]] {
            sbp_relax(&mut b.head, &mut b.deg, &mut b.stack, &mut sl, s, end_u, n);
        }
    }
    if processed != n { return false; }
    sl = 0;
    b.deg[n] = u16::MAX;
    for u in 0..n {
        let d = (ds.job_succ[u] != NONE_USIZE) as u16 + (b.msucc[u] != NONE_USIZE) as u16;
        b.deg[u] = d;
        b.tail[u] = 0;
        unsafe { *b.stack.get_unchecked_mut(sl) = u; }
        sl += (d == 0) as usize;
    }
    processed = 0;
    while sl > 0 {
        sl -= 1;
        let u = unsafe { *b.stack.get_unchecked(sl) };
        processed += 1;
        let need = b.tail[u].saturating_add(ds.node_pt[u]);
        for p in [b.jpred[u], b.mpred[u]] {
            sbp_relax(&mut b.tail, &mut b.deg, &mut b.stack, &mut sl, p, need, n);
        }
    }
    processed == n
}

#[inline(always)]
fn sbp_eval_order(order: impl Iterator<Item = usize>, r: &[u32], p: &[u32], q: &[u32]) -> u32 {
    let mut t = 0u32;
    let mut value = 0u32;
    for i in order {
        let (ri, pi, qi) = unsafe { (*r.get_unchecked(i), *p.get_unchecked(i), *q.get_unchecked(i)) };
        let s = t.max(ri);
        let e = s.saturating_add(pi);
        let c = e.saturating_add(qi);
        value = value.max(c);
        t = e;
    }
    value
}

#[inline(always)]
fn schrage_pick(r: &[u32], q: &[u32], done: &[bool], t: u32) -> usize {
    let mut pick = usize::MAX;
    let mut best_q = 0u32;
    // SAFETY: callers provide equally sized r, q and done; i is below r.len().
    for i in 0..r.len() {
        let (ri, qi, used) = unsafe { (*r.get_unchecked(i), *q.get_unchecked(i), *done.get_unchecked(i)) };
        let take = (!used) & (ri <= t) & ((pick == usize::MAX) | (qi > best_q));
        let mask = 0usize.wrapping_sub(take as usize);
        pick = (pick & !mask) | (i & mask);
        let mask = 0u32.wrapping_sub(take as u32);
        best_q = (best_q & !mask) | (qi & mask);
    }
    pick
}

fn sbp_schrage(r: &[u32], p: &[u32], q: &[u32], done: &mut Vec<bool>, order: &mut Vec<usize>) -> u32 {
    let k = r.len();
    debug_assert_eq!(q.len(), k);
    debug_assert_eq!(p.len(), k);
    done.clear();
    done.resize(k, false);
    order.clear();
    let mut t = u32::MAX;
    for &ri in r { t = t.min(ri); }
    let mut value = 0u32;
    for _ in 0..k {
        let mut pick = schrage_pick(r, q, done, t);
        if pick == usize::MAX {
            let mut nt = u32::MAX;
            for (&ri, &used) in r.iter().zip(done.iter()) {
                let mask = 0u32.wrapping_sub(used as u32);
                nt = nt.min(ri | mask);
            }
            if nt == u32::MAX { break; }
            t = nt;
            pick = schrage_pick(r, q, done, t);
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

fn sbp_carlier_with_budget(r0: &[u32], p: &[u32], q0: &[u32], cb: &mut CarlierBuf, node_budget: usize) -> Option<u32> {
    let k = r0.len();
    if k < 2 { return None; }
    cb.best_order.clear();
    let mut ub = u32::MAX;
    let mut nodes = 0usize;
    let starts = &mut cb.starts;
    starts.clear();
    let mut stack: Vec<(Vec<u32>, Vec<u32>)> = Vec::with_capacity(16);
    stack.push((r0.to_vec(), q0.to_vec()));

    while let Some((r, q)) = stack.pop() {
        if nodes >= node_budget { break; }
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
            let (ri, pi, qi) = unsafe { (*r.get_unchecked(i), *p.get_unchecked(i), *q.get_unchecked(i)) };
            let s = t.max(ri);
            starts.push(s);
            t = s.saturating_add(pi);
            let mask = 0usize.wrapping_sub((t.saturating_add(qi) == f) as usize);
            pos_star = (pos_star & !mask) | (pos & mask);
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
            let (ri, pi, qi) = unsafe { (*r.get_unchecked(i), *p.get_unchecked(i), *q.get_unchecked(i)) };
            r_min = r_min.min(ri);
            q_min = q_min.min(qi);
            p_sum = p_sum.saturating_add(pi);
        }
        if r_min == u32::MAX || q_min == u32::MAX { continue; }

        let h_j = r_min.saturating_add(p_sum).saturating_add(q_min);
        let r_min2 = if r[c] < r_min { r[c] } else { r_min };
        let q_min2 = if q[c] < q_min { q[c] } else { q_min };
        let h_jc = r_min2.saturating_add(p_sum).saturating_add(p[c]).saturating_add(q_min2);
        let lb = if h_j > h_jc { h_j } else { h_jc };
        if lb >= ub { continue; }
        if stack.len() + 2 > 2 * node_budget { continue; }

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

fn sbp_reoptimize(pre: &Pre, challenge: &Challenge, sol: &Solution, rounds: usize) -> Option<(Solution, u32)> {
    sbp_reoptimize_mode::<false>(pre, challenge, sol, rounds)
}

fn sbp_reoptimize_mode<const OLD: bool>(pre: &Pre, challenge: &Challenge, sol: &Solution, rounds: usize) -> Option<(Solution, u32)> {
    if rounds == 0 { return None; }
    let mut ds = build_disj_from_solution(pre, challenge, sol).ok()?;
    let mut buf = EvalBuf::new(ds.n);
    let (start_mk, mk_node) = eval_disj(&ds, &mut buf)?;
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
    if OLD {
        m_order.sort_by(|&x, &y| load[y].cmp(&load[x]).then(x.cmp(&y)));
    } else {
        let mut critical_load = vec![0u64; num_machines];
        let mut node = if ds.n == 0 { NONE_USIZE } else { mk_node };
        while node != NONE_USIZE {
            let m = ds.node_machine[node];
            if m < num_machines {
                critical_load[m] = critical_load[m].saturating_add(ds.node_pt[node] as u64);
            }
            node = buf.best_pred[node];
        }
        m_order.sort_by(|&x, &y| critical_load[y].cmp(&critical_load[x])
            .then_with(|| load[y].cmp(&load[x])).then(x.cmp(&y)));
        m_order.truncate(if ds.n <= 3000 { 8 } else if ds.n <= 8000 { 6 } else { 4 });
    }
    if m_order.is_empty() { return None; }

    for _ in 0..rounds {
        let mut improved = false;
        for &m in &m_order {
            let k = ds.machine_seq[m].len();
            if k < 3 { continue; }
            if !sbp_heads_tails_excl(&ds, m, &mut b) { continue; }
            b.load_machine(&ds, &ds.machine_seq[m]);
            let cur_val = sbp_eval_order(0..k, &b.r, &b.p, &b.q);
            let Some(val) = sbp_carlier_with_budget(&b.r, &b.p, &b.q, &mut cb, if OLD { 600 } else { SBP_NODE_BUDGET }) else { continue };
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

fn sbp_construct_seed(pre: &Pre, challenge: &Challenge, template: &Solution, reopt_cycles: usize) -> Option<(Solution, u32)> {
    let mut ds = build_disj_from_solution(pre, challenge, template).ok()?;
    let n = ds.n;
    let num_machines = challenge.num_machines.min(ds.machine_seq.len());
    if num_machines == 0 { return None; }

    let mut members: Vec<Vec<usize>> = vec![Vec::new(); num_machines];
    let mut member_pos = vec![NONE_USIZE; n];
    for u in 0..n {
        let m = ds.node_machine[u];
        if m < num_machines { member_pos[u] = members[m].len(); members[m].push(u); }
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
            let Some(val) = sbp_carlier_with_budget(&b.r, &b.p, &b.q, &mut cb, SBP_NODE_BUDGET) else { continue };
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
                    let pos = member_pos[nd];
                    if b.nodes.get(pos).copied() != Some(nd) { return None; }
                    cur_seq.push(pos);
                }
                let cur_val = sbp_eval_order(cur_seq.iter().copied(), &b.r, &b.p, &b.q);
                let Some(val) = sbp_carlier_with_budget(&b.r, &b.p, &b.q, &mut cb, SBP_NODE_BUDGET) else { continue };
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

const CONSTRUCT_POOL_CAP: usize = 15;
const K_HI: usize = 6;

fn record(best_mk: &mut u32, best: &mut Solution, sol: &Solution, mk: u32, save: &dyn Fn(&Solution) -> Result<()>) -> Result<bool> {
    if mk < *best_mk {
        *best_mk = mk;
        *best = sol.clone();
        save(sol)?;
        return Ok(true);
    }
    Ok(false)
}

fn pick_competitive_alternative_seed(
    pre: &Pre,
    num_machines: usize,
    top: &[(Solution, u32)],
) -> Option<usize> {
    if top.len() < 2 { return None; }
    let best_mk = top[0].1;
    let quality_band = ((best_mk as f64) * (0.008 + 0.008 * pre.jobshopness))
        .max(pre.avg_op_min * 2.0) as u32;
    let keys: Vec<Vec<u32>> = top.iter().map(|(sol, _)| ts_seed_key(pre, num_machines, sol)).collect();
    let min_distance = ((keys[0].len() / 50).max(4)) as u32;
    let mut chosen: Option<usize> = None;
    let mut chosen_distance = 0u32;
    for i in 1..top.len() {
        if top[i].1 > best_mk.saturating_add(quality_band) { continue; }
        let distance = ts_seed_key_distance(&keys[0], &keys[i]);
        if distance < min_distance { continue; }
        let better = match chosen {
            None => true,
            Some(previous) => {
                distance > chosen_distance
                    || (distance == chosen_distance && top[i].1 < top[previous].1)
            }
        };
        if better {
            chosen = Some(i);
            chosen_distance = distance;
        }
    }
    chosen
}

fn pick_diverse_ts_seeds(
    pre: &Pre,
    num_machines: usize,
    top: &[(Solution, u32)],
    n_best: usize,
    n_diverse: usize,
) -> Vec<usize> {
    let pool_len = top.len();
    if pool_len == 0 { return Vec::new(); }
    let n_best = n_best.min(pool_len);
    let mut picked: Vec<usize> = (0..n_best).collect();
    let want = (n_best + n_diverse).min(pool_len);
    if picked.len() >= want { return picked; }
    let keys: Vec<Vec<u32>> = top.iter().map(|(s, _)| ts_seed_key(pre, num_machines, s)).collect();
    let mut taken = vec![false; pool_len];
    for &i in &picked { taken[i] = true; }
    let mut min_d = vec![u32::MAX; pool_len];
    for i in 0..pool_len {
        if taken[i] { continue; }
        for &p in &picked {
            let d = ts_seed_key_distance(&keys[p], &keys[i]);
            if d < min_d[i] { min_d[i] = d; }
        }
    }
    while picked.len() < want {
        let mut best: Option<usize> = None;
        let mut best_d = 0u32;
        for i in 0..pool_len {
            if taken[i] { continue; }
            if best.is_none() || min_d[i] > best_d {
                best = Some(i);
                best_d = min_d[i];
            }
        }
        let Some(b) = best else { break };
        picked.push(b);
        taken[b] = true;
        for i in 0..pool_len {
            if taken[i] { continue; }
            let d = ts_seed_key_distance(&keys[b], &keys[i]);
            if d < min_d[i] { min_d[i] = d; }
        }
    }
    picked
}

fn apply_sbp_incumbent(
    pre: &Pre,
    challenge: &Challenge,
    best_solution: &mut Solution,
    best_makespan: &mut u32,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<bool> {
    let Some((sol, mk)) = sbp_reoptimize(pre, challenge, best_solution, super::SBP_INC_ROUNDS) else {
        return Ok(false);
    };
    if !record(best_makespan, best_solution, &sol, mk, save_solution)? { return Ok(false); }
    Ok(true)
}

fn run_old_final(
    bound: &super::BoundState, pre: &Pre, challenge: &Challenge,
    best_solution: &mut Solution, best_makespan: &mut u32,
    tenure: usize,
    save: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let iters = 550_000;
    if let Some((sol, mk)) = tabu_search_phase_impl::<false>(bound, pre, challenge, best_solution, iters, tenure, 0, save, *best_makespan)? {
        record(best_makespan, best_solution, &sol, mk, save)?;
    }
    if let Some((sol, mk)) = sbp_reoptimize_mode::<true>(pre, challenge, best_solution, 2) {
        record(best_makespan, best_solution, &sol, mk, save)?;
    }
    Ok(())
}

fn solve_opt_tail(
    bound: &super::BoundState,
    pre: &Pre,
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    greedy_sol: &Solution,
    mut best_solution: Solution,
    mut best_makespan: u32,
    top_solutions: Vec<(Solution, u32)>,
) -> Result<()> {
    let ts_tenure = ((pre.total_ops as f64).sqrt() as usize * (100 + (pre.load_cv * 60.0) as usize) / 100).clamp(8, 24);
    let mut picked = pick_diverse_ts_seeds(pre, challenge.num_machines, &top_solutions, 2, super::DIVERSE_STARTS);
    if pre.total_ops <= 5000 {
        if let Some(idx) = pick_competitive_alternative_seed(pre, challenge.num_machines, &top_solutions) {
            if !picked.contains(&idx) { picked.push(idx); }
        }
    }
    for &idx in &picked {
        if let Some((sol, mk)) = tabu_search_phase(bound, pre, challenge, &top_solutions[idx].0, super::START_ITERS, ts_tenure, 0, save_solution, best_makespan)? {
            record(&mut best_makespan, &mut best_solution, &sol, mk, save_solution)?;
        }
    }
    let mut sbp_seeds: Vec<(Solution, u32)> = Vec::new();
    for cycles in super::SBP_SEED_CYCLES {
        if let Some((sol, mk)) = sbp_construct_seed(pre, challenge, greedy_sol, cycles) {
            record(&mut best_makespan, &mut best_solution, &sol, mk, save_solution)?;
            if let Some((_, previous_mk)) = sbp_seeds.last() {
                if mk == *previous_mk {
                    let k0 = ts_seed_key(pre, challenge.num_machines, &sbp_seeds[0].0);
                    let k1 = ts_seed_key(pre, challenge.num_machines, &sol);
                    if ts_seed_key_distance(&k0, &k1) == 0 { continue; }
                }
            }
            sbp_seeds.push((sol, mk));
        }
    }
    for (seed, _) in &sbp_seeds {
        if let Some((sol, mk)) = tabu_search_phase(bound, pre, challenge, seed, super::SEED_ITERS, ts_tenure, 0, save_solution, best_makespan)? {
            record(&mut best_makespan, &mut best_solution, &sol, mk, save_solution)?;
        }
    }
    apply_sbp_incumbent(pre, challenge, &mut best_solution, &mut best_makespan, save_solution)?;
    if let Some((sol, mk)) = tabu_search_phase(bound, pre, challenge, &best_solution, super::FINAL_ITERS, ts_tenure, 0, save_solution, best_makespan)? {
        record(&mut best_makespan, &mut best_solution, &sol, mk, save_solution)?;
    }
    if apply_sbp_incumbent(pre, challenge, &mut best_solution, &mut best_makespan, save_solution)? {
        if let Some((sol, mk)) = tabu_search_phase(bound, pre, challenge, &best_solution, super::SBP_INC_ITERS, ts_tenure, 0, save_solution, best_makespan)? {
            record(&mut best_makespan, &mut best_solution, &sol, mk, save_solution)?;
        }
        apply_sbp_incumbent(pre, challenge, &mut best_solution, &mut best_makespan, save_solution)?;
    }
    if pre.total_ops <= 3000 && challenge.num_machines <= 80 {
        let seed = best_solution.clone();
        if let Some((sol, mk)) = tabu_search_phase(bound, pre, challenge, &seed, super::DEEP_ITERS, ts_tenure, 300_000, save_solution, best_makespan)? {
            record(&mut best_makespan, &mut best_solution, &sol, mk, save_solution)?;
        }
        apply_sbp_incumbent(pre, challenge, &mut best_solution, &mut best_makespan, save_solution)?;
    }
    if (best_makespan as u64) > bound.lb {
        run_old_final(bound, pre, challenge, &mut best_solution, &mut best_makespan, ts_tenure, save_solution)?;
    }
    save_solution(&best_solution)?;
    Ok(())
}


pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    pre: &Pre,
) -> Result<bool> {
    let bound = super::BoundState::new(pre, challenge.num_machines);
    let guarded = |sol: &Solution| -> Result<()> {
        save_solution(sol)?;
        if bound.offer(pre, sol) { return Err(anyhow::Error::new(super::BoundReached)); }
        Ok(())
    };
    let result = solve_inner(&bound, challenge, &guarded, pre);
    match result {
        Err(e) if e.downcast_ref::<super::BoundReached>().is_some() => Ok(true),
        other => other.map(|()| false),
    }
}

fn solve_inner(
    bound: &super::BoundState, challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>, pre: &Pre,
) -> Result<()> {
    let (greedy_sol, greedy_mk) = run_simple_greedy_baseline(challenge, pre)?;
    save_solution(&greedy_sol)?;

    let mut rng = SmallRng::from_seed(challenge.seed);
    let mut best_makespan = greedy_mk;
    let mut best_solution: Solution = greedy_sol.clone();
    let mut top_solutions: Vec<(Solution, u32)> = Vec::new();
    push_top_solutions(&mut top_solutions, &greedy_sol, greedy_mk, CONSTRUCT_POOL_CAP);
    let target_margin: u32 = ((pre.avg_op_min * (0.9 + 0.6 * pre.jobshopness)).max(1.0)) as u32;

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
        let target = if best_makespan >= u32::MAX / 2 {
            None
        } else { Some(best_makespan.saturating_add(target_margin)) };
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

    let mut refine_results: Vec<(Solution, u32)> = Vec::new();
    for (base_sol, _) in top_solutions.iter() {
        let jb = job_bias_from_solution(pre, base_sol);
        let mp = machine_penalty_from_solution(pre, base_sol, challenge.num_machines);
        let anti_jb = invert_job_bias_guidance(&jb);
        let anti_mp = invert_machine_penalty_guidance(&mp);
        let target_ls = if best_makespan >= u32::MAX / 2 {
            None
        } else { Some(best_makespan.saturating_add(target_margin / 2)) };
        for attempt in 0..super::REFINE_TRIES {
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

    solve_opt_tail(bound, pre, challenge, save_solution, &greedy_sol, best_solution, best_makespan, top_solutions)
}
}

pub mod solver {
use anyhow::Result;
use tig_challenges::job_scheduling::*;

use super::preprocess::build_pre;
use super::job_shop;

pub fn solve_challenge(challenge: &Challenge, save_solution: &dyn Fn(&Solution) -> Result<()>) -> Result<bool> {
    let pre = build_pre(challenge)?;
    job_shop::solve(challenge, save_solution, &pre)
}
}
