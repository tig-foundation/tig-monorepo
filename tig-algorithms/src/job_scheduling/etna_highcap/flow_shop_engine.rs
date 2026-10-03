// Flow-shop track: strict-order search, NEH construction, GRASP with a critical-block descent,
// and an iterated greedy on permutation routes. Every operation has exactly one eligible machine.
use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};
use std::cmp::Reverse;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BinaryHeap, HashMap};
use std::hash::BuildHasherDefault;
use tig_challenges::job_scheduling::*;

const NONE: usize = usize::MAX;

// ------------------------------------------------------------------------------------------------
// Instance
// ------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Op {
    machine: usize,
    pt: u32,
}

/// Slots of the circular calendar of operation ends.
const CAL_SLOTS: usize = 1024;
const CAL_WORDS: usize = CAL_SLOTS / 64;
const NONE16: u16 = u16::MAX;
/// Sentinel machine index of the flat decoder (never free).
const DUMMY_MACHINE: usize = 63;

/// Flat tables of the non-delay decoder (64-stride layout).
struct FlowFast {
    /// Route length, 1..=63.
    r: usize,
    /// `route[k]`: machine of route position k, padded with `DUMMY_MACHINE`.
    route: Vec<usize>,
    /// `pt[(j << 6) | k]`: processing time of job j at route position k.
    pt: Vec<u32>,
    /// `tail_next[(j << 6) | k]`: processing time of job j after position k.
    tail_next: Vec<u32>,
    /// `mload[m]`: total load of machine m, padded with 0.
    mload: Vec<u32>,
}

/// Instance tables. Preconditions guaranteed by the challenge generator and relied upon below:
/// one eligible machine per operation, processing times in 1..=1023, at most 64 jobs, at most
/// 63 machines, routes of 1..=63 operations, every job non-empty.
struct Pre {
    num_jobs: usize,
    num_machines: usize,
    job_products: Vec<usize>,
    job_ops_len: Vec<usize>,
    product_ops: Vec<Vec<Op>>,
    /// Remaining processing time from a route position (one extra trailing zero).
    product_suf_min: Vec<Vec<u32>>,
    /// Remaining load-weighted processing time from a route position.
    product_suf_bn: Vec<Vec<f64>>,
    machine_load0: Vec<f64>,
    avg_machine_load: f64,
    avg_op_min: f64,
    horizon: f64,
    time_scale: f64,
    max_ops: usize,
    max_job_work: f64,
    max_job_bn: f64,
    bn_focus: f64,
    /// Normalised regret of a single-machine operation.
    regret_n: f64,
    /// Rank of each job in a flow-shop insertion order, scaled to [0, 1].
    job_flow_pref: Vec<f64>,
    total_ops: usize,
    /// Machine of each route position when every product follows the same route.
    route: Option<Vec<usize>>,
    /// `pt_by_job[j][k]`: processing time of job j at route position k.
    pt_by_job: Option<Vec<Vec<u32>>>,
    /// Total load of each machine over the common route.
    flow_mload: Vec<u32>,
    flow_fast: Option<FlowFast>,
}

fn flow_makespan(seq: &[usize], pt: &[Vec<u32>], comp: &mut [u32]) -> u32 {
    comp.fill(0);
    for &j in seq {
        let row = &pt[j];
        if row.is_empty() {
            continue;
        }
        comp[0] = comp[0].saturating_add(row[0]);
        for k in 1..row.len() {
            comp[k] = comp[k].max(comp[k - 1]).saturating_add(row[k]);
        }
    }
    *comp.last().unwrap_or(&0)
}

fn build_pre(challenge: &Challenge) -> Result<Pre> {
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;

    let mut job_products = Vec::with_capacity(num_jobs);
    for (p, &cnt) in challenge.jobs_per_product.iter().enumerate() {
        for _ in 0..cnt {
            job_products.push(p);
        }
    }
    if job_products.len() != num_jobs {
        return Err(anyhow!("jobs_per_product sum mismatch"));
    }

    let num_products = challenge.product_processing_times.len();
    let mut product_ops: Vec<Vec<Op>> = Vec::with_capacity(num_products);
    let mut machine_load0 = vec![0.0f64; num_machines];
    let mut total_ops: usize = 0;
    let mut total_min_work: f64 = 0.0;
    let mut max_ops: usize = 1;
    let mut max_job_work: f64 = 1.0;

    for (p, ops) in challenge.product_processing_times.iter().enumerate() {
        max_ops = max_ops.max(ops.len());
        let mut ops_info: Vec<Op> = Vec::with_capacity(ops.len());
        let mut sum_pt: u64 = 0;
        for op in ops {
            if op.len() != 1 {
                return Err(anyhow!("one eligible machine per operation expected"));
            }
            let (&machine, &pt) = op.iter().next().unwrap();
            if machine >= num_machines {
                return Err(anyhow!("machine id out of range"));
            }
            sum_pt += pt as u64;
            ops_info.push(Op { machine, pt });
        }
        max_job_work = max_job_work.max(sum_pt as f64);
        let cnt_u = challenge.jobs_per_product[p];
        let cnt_f = cnt_u as f64;
        total_ops += ops_info.len() * cnt_u;
        total_min_work += (sum_pt as f64) * cnt_f;
        for op in &ops_info {
            machine_load0[op.machine] += (op.pt as f64) * cnt_f;
        }
        product_ops.push(ops_info);
    }

    let job_ops_len: Vec<usize> = job_products.iter().map(|&p| product_ops[p].len()).collect();
    let avg_machine_load = (total_min_work / (num_machines as f64).max(1.0)).max(1.0);
    let horizon = avg_machine_load;
    let avg_op_min = (total_min_work / (total_ops as f64).max(1.0)).max(1.0);
    let regret_n = ((avg_op_min * 2.6) / avg_op_min).clamp(0.0, 6.0);

    let load_cv = {
        let mut var = 0.0f64;
        for &x in &machine_load0 {
            let d = (x / avg_machine_load) - 1.0;
            var += d * d;
        }
        (var / (num_machines as f64)).sqrt().clamp(0.0, 2.5)
    };

    let mut machine_weight = vec![1.0f64; num_machines];
    {
        let exp = (1.10 + 0.35 * load_cv).clamp(1.05, 1.70);
        for m in 0..num_machines {
            let r = (machine_load0[m] / avg_machine_load).max(0.05);
            machine_weight[m] = r.powf(exp).clamp(0.55, 3.75);
        }
    }

    let bn_focus = (2.6 * (1.0 + 0.55 * load_cv) * 0.85).clamp(0.6, 3.4);

    let mut product_suf_min: Vec<Vec<u32>> = Vec::with_capacity(product_ops.len());
    let mut product_suf_bn: Vec<Vec<f64>> = Vec::with_capacity(product_ops.len());
    let mut max_job_bn: f64 = 1e-9;
    for ops in product_ops.iter() {
        let n = ops.len();
        let mut suf_m = vec![0u32; n + 1];
        let mut suf_bn = vec![0.0f64; n + 1];
        for i in (0..n).rev() {
            let op = ops[i];
            suf_m[i] = suf_m[i + 1].saturating_add(op.pt);
            suf_bn[i] = suf_bn[i + 1] + (op.pt as f64) * machine_weight[op.machine];
        }
        max_job_bn = max_job_bn.max(suf_bn[0]);
        product_suf_min.push(suf_m);
        product_suf_bn.push(suf_bn);
    }

    let time_scale = (horizon * (2.65 + 0.15 * load_cv)).max(1.0);

    // Common route: every product has the same length and the same machine at each position.
    let mut route: Option<Vec<usize>> = None;
    let mut pt_by_job: Option<Vec<Vec<u32>>> = None;
    if let Some(first) = product_ops.first() {
        let common_len = first.len();
        let same_route = common_len > 0
            && product_ops.iter().all(|ops| {
                ops.len() == common_len && ops.iter().zip(first.iter()).all(|(a, b)| a.machine == b.machine)
            });
        if same_route {
            route = Some(first.iter().map(|op| op.machine).collect());
            pt_by_job = Some(
                job_products
                    .iter()
                    .map(|&p| product_ops[p].iter().map(|op| op.pt).collect())
                    .collect(),
            );
        }
    }

    let mut job_flow_pref = vec![0.0f64; num_jobs];
    if let Some(pt) = pt_by_job.as_ref() {
        let m = max_ops.max(1);
        let mut jobs: Vec<usize> = (0..num_jobs).collect();
        jobs.sort_unstable_by(|&a, &b| {
            let sa: u32 = pt[a].iter().copied().sum();
            let sb: u32 = pt[b].iter().copied().sum();
            sb.cmp(&sa).then_with(|| a.cmp(&b))
        });
        let mut perm: Vec<usize> = Vec::with_capacity(num_jobs);
        let mut comp = vec![0u32; m];
        let mut tmp: Vec<usize> = Vec::with_capacity(num_jobs);
        for &j in &jobs {
            if perm.is_empty() {
                perm.push(j);
                continue;
            }
            let mut best_mk = u32::MAX;
            let mut best_pos = 0usize;
            for pos in 0..=perm.len() {
                tmp.clear();
                tmp.extend_from_slice(&perm[..pos]);
                tmp.push(j);
                tmp.extend_from_slice(&perm[pos..]);
                let mk = flow_makespan(&tmp, pt, &mut comp);
                if mk < best_mk {
                    best_mk = mk;
                    best_pos = pos;
                }
            }
            perm.insert(best_pos, j);
        }
        let n1 = num_jobs.saturating_sub(1) as f64;
        for (pos, &j) in perm.iter().enumerate() {
            job_flow_pref[j] = if n1 > 0.0 { 1.0 - (pos as f64) / n1 } else { 1.0 };
        }
    }

    let mut flow_mload: Vec<u32> = vec![0u32; num_machines];
    let mut flow_fast: Option<FlowFast> = None;
    if let (Some(route), Some(pt)) = (route.as_ref(), pt_by_job.as_ref()) {
        let r = route.len();
        for row in pt.iter() {
            for k in 0..r {
                flow_mload[route[k]] = flow_mload[route[k]].saturating_add(row[k]);
            }
        }
        let ok = (1..=63).contains(&r)
            && num_jobs <= 64
            && num_machines <= DUMMY_MACHINE
            && pt
                .iter()
                .all(|row| row.iter().all(|&v| v >= 1 && (v as usize) < CAL_SLOTS));
        if ok {
            let mut flat_pt: Vec<u32> = vec![0u32; 64 * 64];
            let mut flat_tail: Vec<u32> = vec![0u32; 64 * 64];
            for j in 0..num_jobs {
                let mut tail = 0u32;
                for k in (0..r).rev() {
                    flat_pt[(j << 6) | k] = pt[j][k];
                    flat_tail[(j << 6) | k] = tail;
                    tail = tail.saturating_add(pt[j][k]);
                }
            }
            let mut route64 = vec![DUMMY_MACHINE; 64];
            route64[..r].copy_from_slice(route);
            let mut mload = vec![0u32; 64];
            mload[..num_machines].copy_from_slice(&flow_mload);
            flow_fast = Some(FlowFast {
                r,
                route: route64,
                pt: flat_pt,
                tail_next: flat_tail,
                mload,
            });
        }
    }

    Ok(Pre {
        num_jobs,
        num_machines,
        job_products,
        job_ops_len,
        product_ops,
        product_suf_min,
        product_suf_bn,
        machine_load0,
        avg_machine_load,
        avg_op_min,
        horizon,
        time_scale,
        max_ops: max_ops.max(1),
        max_job_work: max_job_work.max(1.0),
        max_job_bn: max_job_bn.max(1e-9),
        bn_focus,
        regret_n,
        job_flow_pref,
        total_ops,
        route,
        pt_by_job,
        flow_mload,
        flow_fast,
    })
}

// ------------------------------------------------------------------------------------------------
// Greedy baseline: most remaining work first, non-delay
// ------------------------------------------------------------------------------------------------

fn greedy_baseline(pre: &Pre) -> Result<(Solution, u32)> {
    let num_jobs = pre.num_jobs;
    let num_machines = pre.num_machines;
    let mut job_next_op = vec![0usize; num_jobs];
    let mut job_ready = vec![0u32; num_jobs];
    let mut machine_avail = vec![0u32; num_machines];
    let mut job_schedule: Vec<Vec<(usize, u32)>> = pre
        .job_ops_len
        .iter()
        .map(|&len| Vec::with_capacity(len))
        .collect();
    let mut work_left: Vec<u64> = pre
        .job_products
        .iter()
        .map(|&p| pre.product_ops[p].iter().map(|op| op.pt as u64).sum())
        .collect();
    let mut remaining: usize = pre.job_ops_len.iter().sum();
    let mut time = 0u32;
    let mut best_by_m: Vec<usize> = vec![NONE; num_machines];
    let mut best_work_by_m: Vec<u64> = vec![0; num_machines];
    while remaining > 0 {
        best_by_m.fill(NONE);
        for j in 0..num_jobs {
            let k = job_next_op[j];
            if k >= pre.job_ops_len[j] || job_ready[j] > time {
                continue;
            }
            let m = pre.product_ops[pre.job_products[j]][k].machine;
            if machine_avail[m] > time {
                continue;
            }
            if best_by_m[m] == NONE || work_left[j] > best_work_by_m[m] {
                best_work_by_m[m] = work_left[j];
                best_by_m[m] = j;
            }
        }
        let mut did_work = false;
        for m in 0..num_machines {
            let j = best_by_m[m];
            if machine_avail[m] > time || j == NONE {
                continue;
            }
            let op = pre.product_ops[pre.job_products[j]][job_next_op[j]];
            let end = time + op.pt;
            job_schedule[j].push((m, time));
            job_next_op[j] += 1;
            job_ready[j] = end;
            machine_avail[m] = end;
            work_left[j] = work_left[j].saturating_sub(op.pt as u64);
            remaining -= 1;
            did_work = true;
        }
        if remaining == 0 {
            break;
        }
        if !did_work {
            let mut next = u32::MAX;
            for &t in &machine_avail {
                if t > time && t < next {
                    next = t;
                }
            }
            for j in 0..num_jobs {
                if job_next_op[j] < pre.job_ops_len[j] && job_ready[j] > time && job_ready[j] < next {
                    next = job_ready[j];
                }
            }
            if next == u32::MAX {
                return Err(anyhow!("greedy stalled"));
            }
            time = next;
        }
    }
    let mk = job_ready.iter().copied().max().unwrap_or(0);
    Ok((Solution { job_schedule }, mk))
}

// ------------------------------------------------------------------------------------------------
// Disjunctive graph of a schedule, evaluation, critical-block descent
// ------------------------------------------------------------------------------------------------

#[derive(Clone)]
struct DisjSchedule {
    n: usize,
    num_jobs: usize,
    num_machines: usize,
    job_offsets: Vec<usize>,
    /// `job_succ[n]` is the sentinel slot every read of index n lands on.
    job_succ: Vec<usize>,
    indeg_job: Vec<u16>,
    node_machine: Vec<usize>,
    node_pt: Vec<u32>,
    node_job: Vec<usize>,
    node_op: Vec<usize>,
    machine_seq: Vec<Vec<usize>>,
}

struct EvalBuf {
    indeg: Vec<u16>,
    start: Vec<u32>,
    best_pred: Vec<usize>,
    machine_succ: Vec<usize>,
    /// Stack of the topological sort. Each operation is pushed at most once, when its in-degree
    /// reaches zero, so `n` slots suffice.
    tstack: Vec<usize>,
    /// Roots of the topological sort, at most one per machine.
    roots: Vec<usize>,
}

impl EvalBuf {
    fn new(n: usize) -> Self {
        Self {
            indeg: vec![0u16; n + 1],
            start: vec![0u32; n + 1],
            best_pred: vec![NONE; n + 1],
            machine_succ: vec![NONE; n + 1],
            tstack: vec![0usize; n + 1],
            roots: Vec::with_capacity(n.min(256)),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MoveKind {
    /// Move `from` to position `to`.
    Insert,
    /// Swap `from` with `from + 1`.
    AdjacentSwap,
    /// Swap `from` with `to`.
    Swap,
}

#[derive(Clone, Copy)]
struct MoveCand {
    kind: MoveKind,
    machine: usize,
    from: usize,
    to: usize,
    score: u32,
}

fn build_disj_from_solution(pre: &Pre, sol: &Solution) -> Result<DisjSchedule> {
    let num_jobs = pre.num_jobs;
    let num_machines = pre.num_machines;
    let mut job_offsets = vec![0usize; num_jobs + 1];
    for j in 0..num_jobs {
        job_offsets[j + 1] = job_offsets[j] + pre.job_ops_len[j];
    }
    let n = job_offsets[num_jobs];
    if n == 0 {
        return Err(anyhow!("No operations"));
    }
    let mut node_machine = vec![0usize; n];
    let mut node_pt = vec![0u32; n + 1];
    let mut node_job = vec![0usize; n];
    let mut node_op = vec![0usize; n];
    let mut per_machine: Vec<Vec<(u32, usize)>> = vec![Vec::new(); num_machines];
    for job in 0..num_jobs {
        let expected = pre.job_ops_len[job];
        if sol.job_schedule[job].len() != expected {
            return Err(anyhow!("Invalid solution"));
        }
        let product = pre.job_products[job];
        for op_idx in 0..expected {
            let id = job_offsets[job] + op_idx;
            let (m, st) = sol.job_schedule[job][op_idx];
            let op = pre.product_ops[product][op_idx];
            if op.machine != m {
                return Err(anyhow!("pt missing"));
            }
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
    let mut job_succ = vec![NONE; n + 1];
    let mut indeg_job = vec![0u16; n];
    for job in 0..num_jobs {
        let base = job_offsets[job];
        for k in 0..pre.job_ops_len[job] {
            let id = base + k;
            if k + 1 < pre.job_ops_len[job] {
                job_succ[id] = id + 1;
                indeg_job[id + 1] = indeg_job[id + 1].saturating_add(1);
            }
        }
    }
    Ok(DisjSchedule {
        n,
        num_jobs,
        num_machines,
        job_offsets,
        job_succ,
        indeg_job,
        node_machine,
        node_pt,
        node_job,
        node_op,
        machine_seq,
    })
}

fn rebuild_eval_machine_state_into(
    ds: &DisjSchedule,
    machine_succ: &mut [usize],
    indeg: &mut [u16],
    indeg_job: &[u16],
) {
    indeg[..indeg_job.len()].clone_from_slice(indeg_job);
    machine_succ.fill(NONE);
    for seq in &ds.machine_seq {
        if seq.len() <= 1 {
            continue;
        }
        for i in 0..(seq.len() - 1) {
            let u = seq[i];
            let v = seq[i + 1];
            machine_succ[u] = v;
            indeg[v] = indeg[v].saturating_add(1);
        }
    }
}

/// Reference machine state, recomputed in full (debug builds only).
#[cfg(debug_assertions)]
fn machine_state_consistent(ds: &DisjSchedule, buf: &EvalBuf, indeg_base: &[u16]) -> bool {
    let mut succ = vec![NONE; ds.n];
    let mut deg = ds.indeg_job.clone();
    for m in 0..ds.num_machines {
        let s = &ds.machine_seq[m];
        for i in 0..s.len().saturating_sub(1) {
            succ[s[i]] = s[i + 1];
            deg[s[i + 1]] = deg[s[i + 1]].saturating_add(1);
        }
    }
    succ.as_slice() == &buf.machine_succ[..ds.n] && deg.as_slice() == &indeg_base[..ds.n]
}

#[cfg(not(debug_assertions))]
#[inline(always)]
fn machine_state_consistent(_ds: &DisjSchedule, _buf: &EvalBuf, _indeg_base: &[u16]) -> bool {
    true
}

/// Repairs `machine_succ` and `indeg_base` after a move internal to `machine` that only touched
/// positions `lo..=hi` of its sequence. The links of positions `lo-1..=hi` are rewritten (`lo-1`
/// because its successor changed); the only in-degree that can change is the sequence head's,
/// so the caller supplies the head read before the move.
#[inline]
fn patch_eval_machine_state_window(
    ds: &DisjSchedule,
    buf: &mut EvalBuf,
    indeg_base: &mut [u16],
    machine: usize,
    lo: usize,
    hi: usize,
    old_head: usize,
) {
    if machine >= ds.num_machines {
        return;
    }
    let seq = &ds.machine_seq[machine];
    let len = seq.len();
    if len == 0 {
        return;
    }
    let a = lo.saturating_sub(1);
    let b = hi.min(len - 1);
    for i in a..=b {
        buf.machine_succ[seq[i]] = if i + 1 < len { seq[i + 1] } else { NONE };
    }
    let nh = seq[0];
    if nh != old_head {
        indeg_base[old_head] = indeg_base[old_head].saturating_add(1);
        indeg_base[nh] = indeg_base[nh].saturating_sub(1);
    }
    debug_assert!(
        machine_state_consistent(ds, buf, indeg_base),
        "machine_succ / indeg out of sync with machine_seq after a move"
    );
}

/// Longest-path evaluation over `buf.indeg` (already loaded). Returns the makespan and the node
/// that attains it; `None` when the graph has a cycle. Index `n` is the sentinel slot.
#[inline]
fn eval_disj_prepared(ds: &DisjSchedule, buf: &mut EvalBuf) -> Option<(u32, usize)> {
    let n = ds.n;
    if n == 0 || n >= (1usize << 40) {
        return if n == 0 { Some((0, 0)) } else { None };
    }
    let EvalBuf {
        indeg,
        start,
        best_pred,
        machine_succ,
        tstack,
        roots,
    } = buf;
    let indeg = &mut indeg[..n + 1];
    let start = &mut start[..n + 1];
    let best_pred = &mut best_pred[..n + 1];
    let tstack = &mut tstack[..n + 1];
    let machine_succ = &machine_succ[..n + 1];
    let node_pt = &ds.node_pt[..n + 1];
    let job_succ = &ds.job_succ[..n + 1];
    start.fill(0);
    best_pred.fill(NONE);
    // The sentinel absorbs one decrement per job end and per machine end: far below u16::MAX.
    indeg[n] = u16::MAX;
    roots.clear();
    for m in 0..ds.num_machines {
        if let Some(&h) = ds.machine_seq[m].first() {
            if h < n && indeg[h] == 0 {
                roots.push(h);
            }
        }
    }
    roots.sort_unstable();
    let mut sl = roots.len().min(n);
    tstack[..sl].copy_from_slice(&roots[..sl]);
    let mut processed = 0usize;
    let mut mk = 0u32;
    let mut mk_node = 0usize;
    while sl > 0 {
        sl -= 1;
        let u = tstack[sl.min(n)].min(n);
        processed += 1;
        let end_u = start[u].saturating_add(node_pt[u]);
        let up = end_u > mk;
        mk = if up { end_u } else { mk };
        mk_node = if up { u } else { mk_node };
        let js = job_succ[u].min(n);
        let ms = machine_succ[u].min(n);
        let sj = start[js];
        let lj = sj < end_u;
        start[js] = if lj { end_u } else { sj };
        best_pred[js] = if lj { u } else { best_pred[js] };
        let dj = indeg[js].saturating_sub(1);
        indeg[js] = dj;
        tstack[sl.min(n)] = js;
        sl += (dj == 0) as usize;
        let sm = start[ms];
        let lm = sm < end_u;
        start[ms] = if lm { end_u } else { sm };
        best_pred[ms] = if lm { u } else { best_pred[ms] };
        let dm = indeg[ms].saturating_sub(1);
        indeg[ms] = dm;
        tstack[sl.min(n)] = ms;
        sl += (dm == 0) as usize;
    }
    if processed != n {
        return None;
    }
    Some((mk, mk_node))
}

#[inline]
fn eval_disj_stateful(ds: &DisjSchedule, buf: &mut EvalBuf, indeg_base: &[u16]) -> Option<(u32, usize)> {
    buf.indeg[..indeg_base.len()].clone_from_slice(indeg_base);
    eval_disj_prepared(ds, buf)
}

/// Trial evaluation of a candidate: same pass without `best_pred`, aborted once the makespan
/// reaches `bound`. The makespan accumulator only grows, so `None` is a rejected candidate.
#[inline]
fn eval_disj_trial(ds: &DisjSchedule, buf: &mut EvalBuf, indeg_base: &[u16], bound: u32) -> Option<u32> {
    let n = ds.n;
    if n == 0 || n >= (1usize << 40) || indeg_base.len() < n {
        return if n == 0 { Some(0) } else { None };
    }
    let EvalBuf {
        indeg,
        start,
        machine_succ,
        tstack,
        roots,
        ..
    } = buf;
    let indeg = &mut indeg[..n + 1];
    let start = &mut start[..n + 1];
    let tstack = &mut tstack[..n + 1];
    let machine_succ = &machine_succ[..n + 1];
    let node_pt = &ds.node_pt[..n + 1];
    let job_succ = &ds.job_succ[..n + 1];
    indeg[..n].copy_from_slice(&indeg_base[..n]);
    indeg[n] = u16::MAX;
    start.fill(0);
    roots.clear();
    for m in 0..ds.num_machines {
        if let Some(&h) = ds.machine_seq[m].first() {
            if h < n && indeg[h] == 0 {
                roots.push(h);
            }
        }
    }
    roots.sort_unstable();
    let mut tail = roots.len().min(n);
    tstack[..tail].copy_from_slice(&roots[..tail]);
    let lim = bound.max(1);
    let mut head = 0usize;
    let mut mk = 0u32;
    while head < tail && mk < lim {
        let u = tstack[head.min(n)].min(n);
        head += 1;
        let end_u = start[u].saturating_add(node_pt[u]);
        mk = mk.max(end_u);
        let js = job_succ[u].min(n);
        let ms = machine_succ[u].min(n);
        start[js] = start[js].max(end_u);
        let dj = indeg[js].saturating_sub(1);
        indeg[js] = dj;
        start[ms] = start[ms].max(end_u);
        let dm = indeg[ms].saturating_sub(1);
        indeg[ms] = dm;
        tstack[tail.min(n)] = js;
        tail += (dj == 0) as usize;
        tstack[tail.min(n)] = ms;
        tail += (dm == 0) as usize;
    }
    if mk >= lim || head != n {
        None
    } else {
        Some(mk)
    }
}

fn eval_disj(ds: &DisjSchedule, buf: &mut EvalBuf) -> Option<(u32, usize)> {
    rebuild_eval_machine_state_into(ds, &mut buf.machine_succ, &mut buf.indeg, &ds.indeg_job);
    eval_disj_prepared(ds, buf)
}

fn disj_to_solution(pre: &Pre, ds: &DisjSchedule, start: &[u32]) -> Solution {
    let mut job_schedule: Vec<Vec<(usize, u32)>> = Vec::with_capacity(ds.num_jobs);
    for j in 0..ds.num_jobs {
        let base = ds.job_offsets[j];
        job_schedule.push(
            (0..pre.job_ops_len[j])
                .map(|k| (ds.node_machine[base + k], start[base + k]))
                .collect(),
        );
    }
    Solution { job_schedule }
}

/// Marks the critical chain ending at `mk_node`.
fn mark_critical(buf: &EvalBuf, mk_node: usize, crit: &mut [bool]) {
    crit.fill(false);
    let mut u = mk_node;
    while u != NONE {
        crit[u] = true;
        u = buf.best_pred[u];
    }
}

#[inline]
fn arc_key(machine: usize, left: usize, right: usize) -> u64 {
    ((machine as u64 + 1) << 42) ^ ((left as u64 + 1) << 21) ^ (right as u64 + 1)
}

#[inline]
fn collect_machine_arcs(seq: &[usize], machine: usize, out: &mut Vec<u64>) {
    out.clear();
    if seq.len() <= 1 {
        return;
    }
    for i in 0..(seq.len() - 1) {
        out.push(arc_key(machine, seq[i], seq[i + 1]));
    }
}

/// True when `cand` would recreate an arc that is still tabu.
fn move_hits_recent_arc(
    ds: &DisjSchedule,
    cand: &MoveCand,
    tabu_keys: &[u64],
    tabu_until: &[usize],
    step: usize,
) -> bool {
    let m = cand.machine;
    if m >= ds.num_machines {
        return false;
    }
    let seq = &ds.machine_seq[m];
    let len = seq.len();
    if len <= 1 {
        return false;
    }
    let hits = |left: usize, right: usize| -> bool {
        let key = arc_key(m, left, right);
        for i in 0..tabu_keys.len() {
            if tabu_keys[i] == key && tabu_until[i] > step {
                return true;
            }
        }
        false
    };
    match cand.kind {
        MoveKind::Insert => {
            let from = cand.from;
            if from >= len {
                return false;
            }
            let t = cand.to.min(len - 1);
            if t == from {
                return false;
            }
            let x = seq[from];
            if t < from {
                (t > 0 && hits(seq[t - 1], seq[t]))
                    || (from > 0 && hits(seq[from - 1], x))
                    || (from + 1 < len && hits(x, seq[from + 1]))
            } else {
                (from > 0 && hits(seq[from - 1], x))
                    || (from + 1 < len && hits(x, seq[from + 1]))
                    || (t + 1 < len && hits(seq[t], seq[t + 1]))
            }
        }
        MoveKind::AdjacentSwap => {
            let from = cand.from;
            if from + 1 >= len {
                return false;
            }
            (from > 0 && hits(seq[from - 1], seq[from]))
                || hits(seq[from], seq[from + 1])
                || (from + 2 < len && hits(seq[from + 1], seq[from + 2]))
        }
        MoveKind::Swap => {
            let from = cand.from;
            let to = cand.to;
            if from >= len || to >= len || from + 1 >= to {
                return false;
            }
            (from > 0 && hits(seq[from - 1], seq[from]))
                || hits(seq[from], seq[from + 1])
                || hits(seq[to - 1], seq[to])
                || (to + 1 < len && hits(seq[to], seq[to + 1]))
        }
    }
}

/// Makes tabu the machine arcs created by the accepted move.
fn protect_recent_created_arcs(
    before_seq: &[usize],
    after_seq: &[usize],
    machine: usize,
    tenure: usize,
    step: usize,
    tabu_keys: &mut [u64],
    tabu_until: &mut [usize],
    tabu_pos: &mut usize,
    old_arcs: &mut Vec<u64>,
    new_arcs: &mut Vec<u64>,
) {
    if tabu_keys.is_empty() {
        return;
    }
    collect_machine_arcs(before_seq, machine, old_arcs);
    collect_machine_arcs(after_seq, machine, new_arcs);
    for &key in new_arcs.iter() {
        if old_arcs.iter().any(|&k2| k2 == key) {
            continue;
        }
        let idx = *tabu_pos % tabu_keys.len();
        tabu_keys[idx] = key;
        tabu_until[idx] = step.saturating_add(tenure).saturating_add(1);
        *tabu_pos = idx + 1;
    }
}

/// Moves `seq[from]` to position `to_after_removal` (clamped) by rotating the segment between
/// them; returns the position actually taken.
#[inline]
fn apply_insert(seq: &mut [usize], from: usize, to_after_removal: usize) -> usize {
    let len = seq.len();
    if len == 0 || from >= len {
        return from.min(len.saturating_sub(1));
    }
    let t = to_after_removal.min(len - 1);
    if t < from {
        seq[t..=from].rotate_right(1);
    } else if t > from {
        seq[from..=t].rotate_left(1);
    }
    t
}

/// Applies `c` and patches the evaluation state. Returns the touched window and the position
/// needed to undo, or `None` when the move does not fit its sequence.
fn apply_move(
    ds: &mut DisjSchedule,
    buf: &mut EvalBuf,
    indeg_base: &mut [u16],
    c: &MoveCand,
) -> Option<(usize, usize, usize)> {
    let m = c.machine;
    if m >= ds.num_machines {
        return None;
    }
    let seq = &mut ds.machine_seq[m];
    let len = seq.len();
    let head0 = seq.first().copied().unwrap_or(NONE);
    let (lo, hi, back) = match c.kind {
        MoveKind::Insert => {
            if c.from >= len {
                return None;
            }
            let new_idx = apply_insert(seq, c.from, c.to);
            (new_idx.min(c.from), new_idx.max(c.from), new_idx)
        }
        MoveKind::AdjacentSwap => {
            if c.from + 1 >= len {
                return None;
            }
            seq.swap(c.from, c.from + 1);
            (c.from, c.from + 1, 0)
        }
        MoveKind::Swap => {
            if c.from >= len || c.to >= len || c.from + 1 >= c.to {
                return None;
            }
            seq.swap(c.from, c.to);
            (c.from, c.to, 0)
        }
    };
    patch_eval_machine_state_window(ds, buf, indeg_base, m, lo, hi, head0);
    Some((lo, hi, back))
}

fn undo_move(
    ds: &mut DisjSchedule,
    buf: &mut EvalBuf,
    indeg_base: &mut [u16],
    c: &MoveCand,
    applied: (usize, usize, usize),
) {
    let m = c.machine;
    let (lo, hi, back) = applied;
    let seq = &mut ds.machine_seq[m];
    let head1 = seq.first().copied().unwrap_or(NONE);
    match c.kind {
        MoveKind::Insert => {
            apply_insert(seq, back, c.from);
        }
        MoveKind::AdjacentSwap => seq.swap(c.from, c.from + 1),
        MoveKind::Swap => seq.swap(c.from, c.to),
    }
    patch_eval_machine_state_window(ds, buf, indeg_base, m, lo, hi, head1);
}

/// Keeps the `k` best entries of `top` by decreasing `key`, first come first kept on ties.
#[inline]
fn push_top_k_by<T, K: PartialOrd>(top: &mut Vec<T>, c: T, k: usize, key: impl Fn(&T) -> K) {
    if k == 0 {
        return;
    }
    let kc = key(&c);
    let mut pos = top.len();
    while pos > 0 && key(&top[pos - 1]) < kc {
        pos -= 1;
    }
    if pos >= k {
        return;
    }
    top.insert(pos, c);
    if top.len() > k {
        top.pop();
    }
}

/// Tabu descent over critical-block moves with a record-to-record acceptance band. Returns true
/// when the current makespan improved at least once.
fn descent_phase(
    ds: &mut DisjSchedule,
    buf: &mut EvalBuf,
    crit: &mut [bool],
    cur_eval: &mut (u32, usize),
    max_iters: usize,
    prescreen_k: usize,
) -> bool {
    let mut cur_mk = cur_eval.0;
    let mut best_seen = cur_mk;
    let mut improved = false;
    let mut aspiration: u32 = 0;

    let hlen = max_iters.max(1);
    let mut delta_hist: Vec<u32> = vec![0u32; hlen];
    let mut delta_sorted: Vec<u32> = Vec::with_capacity(hlen);
    let mut dhpos: usize = 0;
    let mut dhfill: usize = 0;

    let mut tenure = 2usize;
    let tabu_cap = 48usize;
    let mut tabu_keys: Vec<u64> = vec![0u64; tabu_cap];
    let mut tabu_until: Vec<usize> = vec![0usize; tabu_cap];
    let mut tabu_pos: usize = 0;
    let mut arc_old: Vec<u64> = Vec::new();
    let mut arc_new: Vec<u64> = Vec::new();
    let n = ds.n;
    let mut chain: Vec<usize> = Vec::with_capacity(n);
    let mut crit_tail = vec![0u32; n];
    let mut crit_rank = vec![NONE; n];
    let mut machine_pos = vec![NONE; n];
    let mut crit_blocks: Vec<(usize, usize, usize)> = Vec::new();
    let mut indeg_base = vec![0u16; n];
    // The shift list [1, 2, max_shift] repeats a move when max_shift <= 2; a repeat cannot win
    // under the strict comparison, so each distinct candidate is evaluated once.
    let mut seen_moves: Vec<(MoveKind, usize, usize, usize)> = Vec::with_capacity(64);
    rebuild_eval_machine_state_into(ds, &mut buf.machine_succ, &mut indeg_base, &ds.indeg_job);

    let mut recent_makespans: Vec<u32> = vec![0u32; 50];
    let mut recent_fill: usize = 0;
    let mut recent_idx: usize = 0;
    let mut non_improving_count: usize = 0;

    for iter_ix in 0..max_iters {
        mark_critical(buf, cur_eval.1, crit);

        chain.clear();
        let mut z = cur_eval.1;
        while z != NONE {
            chain.push(z);
            z = buf.best_pred[z];
        }
        chain.reverse();

        crit_tail.fill(0);
        crit_rank.fill(NONE);
        let mut carry = 0u32;
        for idx in (0..chain.len()).rev() {
            let v = chain[idx];
            carry = ds.node_pt[v].saturating_add(carry);
            crit_tail[v] = carry;
            crit_rank[v] = idx;
        }

        machine_pos.fill(NONE);
        crit_blocks.clear();
        for m in 0..ds.num_machines {
            let seq = &ds.machine_seq[m];
            let len = seq.len();
            let mut i = 0usize;
            while i < len {
                let a = seq[i];
                machine_pos[a] = i;
                if !crit[a] {
                    i += 1;
                    continue;
                }
                let bstart = i;
                let mut bend = i;
                while bend + 1 < len {
                    let y = seq[bend + 1];
                    machine_pos[y] = bend + 1;
                    if !crit[y] {
                        break;
                    }
                    let x = seq[bend];
                    if buf.start[y] != buf.start[x].saturating_add(ds.node_pt[x]) {
                        break;
                    }
                    bend += 1;
                }
                if bend > bstart {
                    crit_blocks.push((m, bstart, bend));
                }
                i = bend + 1;
            }
        }

        let cycle_detected = recent_fill > 0 && recent_makespans[..recent_fill].iter().any(|&x| x == cur_mk);
        if cycle_detected {
            tenure = 12;
        }

        let op_surrogate = |u: usize| -> u64 {
            let st = buf.start[u];
            let ptu = ds.node_pt[u];
            let end_u = st.saturating_add(ptu);
            let js = ds.job_succ[u];
            let ms = buf.machine_succ[u];
            let job_prev = if ds.node_op[u] > 0 { u - 1 } else { NONE };
            let m = ds.node_machine[u];
            let pos = machine_pos[u];
            let mach_prev = if pos > 0 && pos != NONE {
                ds.machine_seq[m][pos - 1]
            } else {
                NONE
            };
            let mach_next = if pos != NONE && pos + 1 < ds.machine_seq[m].len() {
                ds.machine_seq[m][pos + 1]
            } else {
                NONE
            };
            let gap_before = |p: usize| -> u32 {
                if p == NONE {
                    0
                } else {
                    st.saturating_sub(buf.start[p].saturating_add(ds.node_pt[p]))
                }
            };
            let gap_after = |v: usize| -> u32 {
                if v == NONE {
                    0
                } else {
                    buf.start[v].saturating_sub(end_u)
                }
            };
            let tight_before = |p: usize| -> u64 {
                if p == NONE {
                    0
                } else {
                    ds.node_pt[p].saturating_sub(gap_before(p).min(ds.node_pt[p])) as u64
                }
            };
            let tight_after = |v: usize| -> u64 {
                if v == NONE {
                    0
                } else {
                    ds.node_pt[v].saturating_sub(gap_after(v).min(ds.node_pt[v])) as u64
                }
            };
            let down_job = if js != NONE {
                ds.node_pt[js].saturating_add(gap_after(js))
            } else {
                0
            };
            let down_mach = if ms != NONE {
                ds.node_pt[ms].saturating_add(gap_after(ms))
            } else {
                0
            };
            let head_tail = if crit_tail[u] > 0 {
                crit_tail[u]
            } else {
                ptu.saturating_add(down_job.max(down_mach))
            };
            let near_chain = (job_prev != NONE && crit[job_prev])
                || (js != NONE && crit[js])
                || (mach_prev != NONE && crit[mach_prev])
                || (mach_next != NONE && crit[mach_next]);
            (end_u as u64) * 7
                + (ptu as u64) * 9
                + (head_tail as u64) * 11
                + tight_before(job_prev) * 3
                + tight_before(mach_prev) * 5
                + tight_after(js) * 4
                + tight_after(ms) * 6
                + if crit[u] {
                    (ptu as u64) * 3 + (head_tail as u64) * 2
                } else {
                    0
                }
                + if near_chain { (ptu as u64) * 2 } else { 0 }
        };

        let pair_surrogate = |u: usize, v: usize| -> u32 {
            let mut s = op_surrogate(u).saturating_add(op_surrogate(v));
            if crit[u] && crit[v] {
                s = s.saturating_add((ds.node_pt[u] as u64 + ds.node_pt[v] as u64) * 6);
            }
            let ru = crit_rank[u];
            let rv = crit_rank[v];
            if ru != NONE && rv != NONE {
                let dist = ru.max(rv) - ru.min(rv);
                s = s.saturating_add((chain.len().saturating_sub(dist) as u64) * 3);
            }
            s.min(u32::MAX as u64) as u32
        };

        let mut cands: Vec<MoveCand> = Vec::with_capacity(prescreen_k.min(64));
        seen_moves.clear();
        for &(m, bstart, bend) in &crit_blocks {
            let seq = &ds.machine_seq[m];
            let max_shift = bend - bstart;
            for &sh in &[1usize, 2, max_shift] {
                if sh == 0 || sh > max_shift {
                    continue;
                }
                let to_after = bstart + sh;
                if bstart < seq.len() && to_after <= seq.len() {
                    let tgt_idx = to_after.min(seq.len() - 1);
                    let score = pair_surrogate(seq[bstart], seq[tgt_idx]);
                    push_top_k_by(
                        &mut cands,
                        MoveCand {
                            kind: MoveKind::Insert,
                            machine: m,
                            from: bstart,
                            to: to_after,
                            score,
                        },
                        prescreen_k,
                        |c| c.score,
                    );
                }
                let to_after2 = bend - sh;
                let tgt_idx2 = to_after2.min(seq.len().saturating_sub(1));
                let score = pair_surrogate(seq[bend], seq[tgt_idx2]);
                push_top_k_by(
                    &mut cands,
                    MoveCand {
                        kind: MoveKind::Insert,
                        machine: m,
                        from: bend,
                        to: to_after2,
                        score,
                    },
                    prescreen_k,
                    |c| c.score,
                );
            }
            if bstart > 0 {
                let score = pair_surrogate(seq[bstart - 1], seq[bstart]);
                push_top_k_by(
                    &mut cands,
                    MoveCand {
                        kind: MoveKind::AdjacentSwap,
                        machine: m,
                        from: bstart - 1,
                        to: 0,
                        score,
                    },
                    prescreen_k,
                    |c| c.score,
                );
            }
            if bend + 1 < seq.len() {
                let score = pair_surrogate(seq[bend], seq[bend + 1]);
                push_top_k_by(
                    &mut cands,
                    MoveCand {
                        kind: MoveKind::AdjacentSwap,
                        machine: m,
                        from: bend,
                        to: 0,
                        score,
                    },
                    prescreen_k,
                    |c| c.score,
                );
            }
            let mid = (bstart + bend) / 2;
            let mut push_swap = |i1: usize, i2: usize| {
                let (lo, hi) = if i1 < i2 { (i1, i2) } else { (i2, i1) };
                if lo + 1 >= hi {
                    return;
                }
                let score = pair_surrogate(seq[lo], seq[hi]);
                push_top_k_by(
                    &mut cands,
                    MoveCand {
                        kind: MoveKind::Swap,
                        machine: m,
                        from: lo,
                        to: hi,
                        score,
                    },
                    prescreen_k,
                    |c| c.score,
                );
            };
            push_swap(bstart, bend);
            push_swap(bstart, mid);
            push_swap(mid, bend);
        }

        if cands.is_empty() {
            break;
        }

        let mut best_cand: Option<MoveCand> = None;
        let mut best_mk = u32::MAX;
        for cand in &cands {
            let key = (cand.kind, cand.machine, cand.from, cand.to);
            if seen_moves.iter().any(|&k| k == key) {
                continue;
            }
            seen_moves.push(key);
            let cand_tabu = move_hits_recent_arc(ds, cand, &tabu_keys, &tabu_until, iter_ix);
            let Some(applied) = apply_move(ds, buf, &mut indeg_base, cand) else {
                continue;
            };
            if let Some(mk2) = eval_disj_trial(ds, buf, &indeg_base, best_mk) {
                if mk2 < best_mk && (!cand_tabu || mk2 < best_seen.saturating_add(aspiration)) {
                    best_mk = mk2;
                    best_cand = Some(*cand);
                }
            }
            undo_move(ds, buf, &mut indeg_base, cand, applied);
        }

        let Some(bc) = best_cand else { break };
        let prev_mk = cur_mk;

        let d = if best_mk > best_seen {
            best_mk - best_seen
        } else {
            best_seen - best_mk
        };
        if dhfill < hlen {
            delta_hist[dhpos] = d;
            let pos = delta_sorted.binary_search(&d).unwrap_or_else(|p| p);
            delta_sorted.insert(pos, d);
            dhfill += 1;
        } else {
            let old = delta_hist[dhpos];
            if let Ok(pos) = delta_sorted.binary_search(&old) {
                delta_sorted.remove(pos);
            }
            delta_hist[dhpos] = d;
            let pos = delta_sorted.binary_search(&d).unwrap_or_else(|p| p);
            delta_sorted.insert(pos, d);
        }
        dhpos += 1;
        if dhpos >= hlen {
            dhpos = 0;
        }
        let band = if dhfill == 0 { 0 } else { delta_sorted[dhfill >> 1] };
        let rrt_limit = best_seen.saturating_add(band);

        let mut accepted = false;
        let mut global_improved = false;
        let m = bc.machine;
        let before_seq = ds.machine_seq[m].clone();
        if let Some(applied) = apply_move(ds, buf, &mut indeg_base, &bc) {
            let mut keep = false;
            if let Some(next_eval) = eval_disj_stateful(ds, buf, &indeg_base) {
                let next_mk = next_eval.0;
                if next_mk < prev_mk || next_mk <= rrt_limit {
                    *cur_eval = next_eval;
                    cur_mk = next_mk;
                    if next_mk < prev_mk {
                        improved = true;
                    }
                    if next_mk < best_seen {
                        best_seen = next_mk;
                        global_improved = true;
                    }
                    protect_recent_created_arcs(
                        &before_seq,
                        &ds.machine_seq[m],
                        m,
                        tenure,
                        iter_ix,
                        &mut tabu_keys,
                        &mut tabu_until,
                        &mut tabu_pos,
                        &mut arc_old,
                        &mut arc_new,
                    );
                    accepted = true;
                    keep = true;
                }
            }
            if !keep {
                undo_move(ds, buf, &mut indeg_base, &bc, applied);
            }
        }

        if !accepted {
            aspiration = (aspiration + 1).min(5);
        } else {
            aspiration = 0;
            if global_improved {
                tenure = 2;
                non_improving_count = 0;
            } else {
                non_improving_count += 1;
                tenure = (2 + non_improving_count).min(12);
            }
            if recent_fill < recent_makespans.len() {
                recent_makespans[recent_fill] = cur_mk;
                recent_fill += 1;
            } else {
                recent_makespans[recent_idx] = cur_mk;
                recent_idx = (recent_idx + 1) % recent_makespans.len();
            }
        }
    }
    improved
}

/// Descent from `base_sol`, then `perturb_cycles` rounds of two random critical-block swaps
/// followed by a descent, keeping the best end-of-descent schedule. `None` when nothing beats
/// the start.
fn critical_block_local_search(
    pre: &Pre,
    seed: &[u8; 32],
    base_sol: &Solution,
    perturb_cycles: usize,
) -> Result<Option<(Solution, u32)>> {
    const MAX_ITERS: usize = 28;
    const PRESCREEN_K: usize = 48;
    let mut ds = build_disj_from_solution(pre, base_sol)?;
    let mut buf = EvalBuf::new(ds.n);
    let mut crit = vec![false; ds.n];
    let mut cur_eval = match eval_disj(&ds, &mut buf) {
        Some(x) => x,
        None => return Ok(None),
    };
    let initial_mk = cur_eval.0;
    descent_phase(
        &mut ds,
        &mut buf,
        &mut crit,
        &mut cur_eval,
        MAX_ITERS,
        PRESCREEN_K,
    );
    let Some((mk_after, _)) = eval_disj(&ds, &mut buf) else {
        return Ok(None);
    };
    let mut global_best_mk = mk_after;
    let mut global_best_ds = ds.clone();
    let mut pseed: u64 = (seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ (initial_mk as u64).wrapping_shl(16)
        ^ (ds.n as u64);
    let mut blocks: Vec<(usize, usize, usize)> = Vec::new();
    for _ in 0..perturb_cycles {
        ds = global_best_ds.clone();
        let Some((_, mk_node)) = eval_disj(&ds, &mut buf) else {
            break;
        };
        mark_critical(&buf, mk_node, &mut crit);
        blocks.clear();
        for m in 0..ds.num_machines {
            let seq = &ds.machine_seq[m];
            if seq.len() <= 1 {
                continue;
            }
            let mut i = 0usize;
            while i < seq.len() {
                if !crit[seq[i]] {
                    i += 1;
                    continue;
                }
                let bstart = i;
                let mut bend = i;
                while bend + 1 < seq.len() {
                    let x = seq[bend];
                    let y = seq[bend + 1];
                    if !crit[y] || buf.start[y] != buf.start[x].saturating_add(ds.node_pt[x]) {
                        break;
                    }
                    bend += 1;
                }
                if bend > bstart {
                    blocks.push((m, bstart, bend));
                }
                i = bend + 1;
            }
        }
        if blocks.is_empty() {
            break;
        }
        for _ in 0..2 {
            pseed ^= pseed.wrapping_shl(13);
            pseed ^= pseed.wrapping_shr(7);
            pseed ^= pseed.wrapping_shl(17);
            let (m, bstart, bend) = blocks[(pseed as usize) % blocks.len()];
            let block_len = bend - bstart;
            if block_len == 0 {
                continue;
            }
            pseed ^= pseed.wrapping_shl(13);
            pseed ^= pseed.wrapping_shr(7);
            pseed ^= pseed.wrapping_shl(17);
            let swap_pos = bstart + ((pseed as usize) % block_len);
            if swap_pos + 1 < ds.machine_seq[m].len() {
                ds.machine_seq[m].swap(swap_pos, swap_pos + 1);
            }
        }
        match eval_disj(&ds, &mut buf) {
            Some(x) => cur_eval = x,
            None => continue,
        }
        descent_phase(
            &mut ds,
            &mut buf,
            &mut crit,
            &mut cur_eval,
            MAX_ITERS,
            PRESCREEN_K,
        );
        if let Some((mk_now, _)) = eval_disj(&ds, &mut buf) {
            if mk_now < global_best_mk {
                global_best_mk = mk_now;
                global_best_ds = ds.clone();
            }
        }
    }
    if global_best_mk >= initial_mk {
        return Ok(None);
    }
    ds = global_best_ds;
    let Some((mk_final, _)) = eval_disj(&ds, &mut buf) else {
        return Ok(None);
    };
    Ok(Some((disj_to_solution(pre, &ds, &buf.start), mk_final)))
}

// ------------------------------------------------------------------------------------------------
// Job orders on the common route: NEH construction and insertion improvement
// ------------------------------------------------------------------------------------------------

/// Makespan of a permutation schedule on a route that may revisit machines: jobs are placed
/// whole, in sequence order.
fn reentrant_makespan(seq: &[usize], route: &[usize], pt: &[Vec<u32>], mready: &mut [u32]) -> u32 {
    mready.fill(0);
    let mut mk = 0u32;
    for &j in seq {
        let row = &pt[j];
        let mut prev = 0u32;
        for (op_idx, &m) in route.iter().enumerate() {
            let end = prev.max(mready[m]).saturating_add(row[op_idx]);
            mready[m] = end;
            prev = end;
        }
        mk = mk.max(prev);
    }
    mk
}

fn johnson_order_from_ab(a: &[u32], b: &[u32]) -> Vec<usize> {
    let n = a.len().min(b.len());
    let mut front: Vec<(u32, usize)> = Vec::with_capacity(n);
    let mut back: Vec<(u32, usize)> = Vec::with_capacity(n);
    for j in 0..n {
        if a[j] <= b[j] {
            front.push((a[j], j));
        } else {
            back.push((b[j], j));
        }
    }
    front.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
    back.sort_unstable_by(|x, y| y.0.cmp(&x.0).then_with(|| x.1.cmp(&y.1)));
    front.iter().chain(back.iter()).map(|&(_, j)| j).collect()
}

fn palmer_order(pt: &[Vec<u32>]) -> Vec<usize> {
    let n = pt.len();
    let m = pt.first().map(|r| r.len()).unwrap_or(0);
    if m == 0 {
        return (0..n).collect();
    }
    let mm = m as i64;
    let mut jobs: Vec<(i64, usize)> = Vec::with_capacity(n);
    for j in 0..n {
        let mut s: i64 = 0;
        for k in 0..m {
            s += (mm - 2 * (k as i64) - 1) * (pt[j][k] as i64);
        }
        jobs.push((s, j));
    }
    jobs.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    jobs.into_iter().map(|x| x.1).collect()
}

fn cds_orders(pt: &[Vec<u32>]) -> Vec<Vec<usize>> {
    let n = pt.len();
    if n == 0 {
        return vec![];
    }
    let m = pt[0].len();
    if m <= 1 {
        return vec![(0..n).collect()];
    }
    let mut totals = vec![0u32; n];
    let mut prefix = vec![vec![0u32; m + 1]; n];
    for j in 0..n {
        let mut s = 0u32;
        for k in 0..m {
            s = s.saturating_add(pt[j][k]);
            prefix[j][k + 1] = s;
        }
        totals[j] = s;
    }
    let mut res: Vec<Vec<usize>> = Vec::with_capacity(m - 1);
    let mut a = vec![0u32; n];
    let mut b = vec![0u32; n];
    for k in 1..m {
        for j in 0..n {
            a[j] = prefix[j][k];
            b[j] = totals[j].saturating_sub(prefix[j][k]);
        }
        res.push(johnson_order_from_ab(&a, &b));
    }
    res
}

/// True when no machine appears twice on the route (plain permutation flow shop).
fn route_is_unique(route: &[usize], num_machines: usize) -> bool {
    if route.is_empty() {
        return false;
    }
    let mut seen = vec![false; num_machines.max(1)];
    for &m in route {
        if m >= seen.len() || seen[m] {
            return false;
        }
        seen[m] = true;
    }
    true
}

/// Scratch tables of the insertion evaluation on a permutation route.
#[derive(Default)]
struct PermInsBuf {
    f: Vec<u32>,
    b: Vec<u32>,
    e: Vec<u32>,
    comp: Vec<u32>,
}

impl PermInsBuf {
    fn ensure(&mut self, len: usize, m: usize) {
        let need = (len + 1) * m;
        if self.f.len() < need {
            self.f.resize(need, 0);
        }
        if self.b.len() < need {
            self.b.resize(need, 0);
        }
        if self.e.len() < m {
            self.e.resize(m, 0);
        }
        if self.comp.len() < m {
            self.comp.resize(m, 0);
        }
    }
}

/// Best insertion position of `job` into `seq` on a permutation route, with the resulting
/// makespan. First strict minimum kept.
fn perm_best_insert_pos(
    seq: &[usize],
    job: usize,
    pt: &[Vec<u32>],
    m: usize,
    buf: &mut PermInsBuf,
) -> (usize, u32) {
    let l = seq.len();
    if m == 0 {
        return (0, 0);
    }
    buf.ensure(l, m);
    let f = &mut buf.f;
    let b = &mut buf.b;
    let e = &mut buf.e;
    for k in 0..m {
        f[k] = 0;
    }
    for t in 1..=l {
        let row = &pt[seq[t - 1]];
        let base = t * m;
        let prev = (t - 1) * m;
        f[base] = f[prev].saturating_add(row[0]);
        for k in 1..m {
            f[base + k] = f[base + k - 1].max(f[prev + k]).saturating_add(row[k]);
        }
    }
    let base_l = l * m;
    for k in 0..m {
        b[base_l + k] = 0;
    }
    for t in (0..l).rev() {
        let row = &pt[seq[t]];
        let base = t * m;
        let next = (t + 1) * m;
        b[base + (m - 1)] = b[next + (m - 1)].saturating_add(row[m - 1]);
        for k in (0..m - 1).rev() {
            b[base + k] = b[base + k + 1].max(b[next + k]).saturating_add(row[k]);
        }
    }
    let prow = &pt[job];
    let mut best_pos = 0usize;
    let mut best_mk = u32::MAX;
    for pos in 0..=l {
        let fb = pos * m;
        e[0] = f[fb].saturating_add(prow[0]);
        for k in 1..m {
            e[k] = e[k - 1].max(f[fb + k]).saturating_add(prow[k]);
        }
        let mut mk = 0u32;
        for k in 0..m {
            mk = mk.max(e[k].saturating_add(b[fb + k]));
        }
        if mk < best_mk {
            best_mk = mk;
            best_pos = pos;
        }
    }
    (best_pos, best_mk)
}

fn neh_build_perm(order: &[usize], pt: &[Vec<u32>], m: usize, buf: &mut PermInsBuf) -> Vec<usize> {
    let mut seq: Vec<usize> = Vec::with_capacity(order.len());
    for &j in order {
        if seq.is_empty() {
            seq.push(j);
            continue;
        }
        let (pos, _) = perm_best_insert_pos(&seq, j, pt, m, buf);
        seq.insert(pos, j);
    }
    seq
}

/// Reinserts every job at its best position, up to `rounds` sweeps or until a sweep brings nothing.
fn improve_perm_seq(seq: &mut Vec<usize>, pt: &[Vec<u32>], rounds: usize, buf: &mut PermInsBuf) {
    let m = pt.first().map(|r| r.len()).unwrap_or(0);
    if seq.len() <= 2 || m == 0 {
        return;
    }
    buf.ensure(seq.len(), m);
    let mut cur_mk = flow_makespan(seq, pt, &mut buf.comp[..m]);
    for _ in 0..rounds {
        let mut improved_any = false;
        for i0 in 0..seq.len() {
            let job = seq.remove(i0);
            let (pos, mk) = perm_best_insert_pos(seq, job, pt, m, buf);
            seq.insert(pos, job);
            if mk < cur_mk {
                cur_mk = mk;
                improved_any = true;
            }
        }
        if !improved_any {
            break;
        }
    }
}

/// Insertion evaluation on a route that may visit a machine more than once. Jobs are placed
/// whole in sequence order, so the only machine arcs between consecutive jobs go from the last
/// visit of a machine to its first visit; heads and tails follow two O(R) recurrences per job.
#[derive(Default)]
struct ReentrantInsBuf {
    /// `f[(t+1)*w + r]`: end of stage r of the job at position t (row 0 is the empty prefix).
    f: Vec<u32>,
    /// `b[t*w + r]`: longest path from stage r of the job at position t to the end.
    b: Vec<u32>,
    /// Heads of the candidate job at the current position.
    e: Vec<u32>,
    first_r: Vec<usize>,
    last_r: Vec<usize>,
    /// One `(last stage, first stage)` pair per machine visited.
    arcs: Vec<(usize, usize)>,
    fcol: Vec<usize>,
    bcol: Vec<usize>,
    r_len: usize,
}

impl ReentrantInsBuf {
    fn set_route(&mut self, route: &[usize], num_machines: usize) {
        let r_len = route.len();
        self.r_len = r_len;
        let nm = num_machines.max(route.iter().copied().max().map_or(0, |x| x + 1));
        self.first_r.clear();
        self.first_r.resize(nm, usize::MAX);
        self.last_r.clear();
        self.last_r.resize(nm, usize::MAX);
        for (r, &m) in route.iter().enumerate() {
            if self.first_r[m] == usize::MAX {
                self.first_r[m] = r;
            }
            self.last_r[m] = r;
        }
        self.arcs.clear();
        for (r, &m) in route.iter().enumerate() {
            if self.last_r[m] == r {
                self.arcs.push((r, self.first_r[m]));
            }
        }
        self.fcol.clear();
        self.bcol.clear();
        for (r, &m) in route.iter().enumerate() {
            self.fcol.push(if self.first_r[m] == r {
                self.last_r[m].min(r_len)
            } else {
                r_len
            });
            self.bcol.push(if self.last_r[m] == r {
                self.first_r[m].min(r_len)
            } else {
                r_len
            });
        }
        self.e.clear();
        self.e.resize(r_len, 0);
    }

    /// Heads of every prefix and tails of every suffix of `seq`.
    fn rebuild(&mut self, seq: &[usize], pt: &[Vec<u32>]) {
        let l = seq.len();
        let r_len = self.r_len.min(1usize << 40);
        let w = r_len + 1;
        let sz = (l + 1) * w;
        self.f.clear();
        self.f.resize(sz, 0);
        self.b.clear();
        self.b.resize(sz, 0);
        let fcol = &self.fcol[..r_len];
        let bcol = &self.bcol[..r_len];
        for t in 0..l {
            let row = &pt[seq[t]][..r_len];
            let (prv, cur) = self.f.split_at_mut((t + 1) * w);
            let prv = &prv[t * w..(t + 1) * w];
            let cur = &mut cur[..w];
            let mut prev_end = 0u32;
            for r in 0..r_len {
                let st = prev_end.max(prv[fcol[r].min(r_len)]);
                prev_end = st.saturating_add(row[r]);
                cur[r] = prev_end;
            }
        }
        for t in (0..l).rev() {
            let row = &pt[seq[t]][..r_len];
            let (cur, nxt) = self.b.split_at_mut((t + 1) * w);
            let cur = &mut cur[t * w..(t + 1) * w];
            let nxt = &nxt[..w];
            let mut next_tail = 0u32;
            for r in (0..r_len).rev() {
                let v = next_tail.max(nxt[bcol[r].min(r_len)]);
                next_tail = v.saturating_add(row[r]);
                cur[r] = next_tail;
            }
        }
    }

    /// Makespan of `seq[..pos] + job + seq[pos..]` after `rebuild(seq)`; `pos <= seq.len()`.
    #[inline]
    fn makespan_at(&mut self, job: usize, pos: usize, pt: &[Vec<u32>]) -> u32 {
        let r_len = self.r_len.min(1usize << 40);
        let w = r_len + 1;
        let base = pos * w;
        let row = &pt[job][..r_len];
        let fcol = &self.fcol[..r_len];
        let prv = &self.f[base..base + w];
        let bnx = &self.b[base..base + w];
        let e = &mut self.e[..r_len];
        let mut prev_end = 0u32;
        for r in 0..r_len {
            let st = prev_end.max(prv[fcol[r].min(r_len)]);
            prev_end = st.saturating_add(row[r]);
            e[r] = prev_end;
        }
        let mut mk = 0u32;
        for &(lr, fr) in self.arcs.iter() {
            mk = mk.max(e[lr.min(r_len - 1)].saturating_add(bnx[fr.min(r_len)]));
        }
        mk
    }

    /// Best insertion position, first strict minimum kept.
    fn best_insert(&mut self, job: usize, seq_len: usize, pt: &[Vec<u32>]) -> (usize, u32) {
        let mut best_mk = u32::MAX;
        let mut best_pos = 0usize;
        for pos in 0..=seq_len {
            let mk = self.makespan_at(job, pos, pt);
            if mk < best_mk {
                best_mk = mk;
                best_pos = pos;
            }
        }
        (best_pos, best_mk)
    }
}

fn neh_build_reentrant(order: &[usize], pt: &[Vec<u32>], buf: &mut ReentrantInsBuf) -> Vec<usize> {
    let mut seq: Vec<usize> = Vec::with_capacity(order.len());
    for &j in order {
        if seq.is_empty() {
            seq.push(j);
            continue;
        }
        buf.rebuild(&seq, pt);
        let (pos, _) = buf.best_insert(j, seq.len(), pt);
        seq.insert(pos, j);
    }
    seq
}

fn improve_reentrant_seq(
    seq: &mut Vec<usize>,
    route: &[usize],
    pt: &[Vec<u32>],
    mready: &mut [u32],
    buf: &mut ReentrantInsBuf,
) {
    if seq.len() <= 2 || route.is_empty() {
        return;
    }
    let mut cur_mk = reentrant_makespan(seq, route, pt, mready);
    for _ in 0..8usize {
        let mut improved_any = false;
        for i0 in 0..seq.len() {
            let j = seq.remove(i0);
            buf.rebuild(seq, pt);
            let (pos, mk) = buf.best_insert(j, seq.len(), pt);
            seq.insert(pos, j);
            if mk < cur_mk {
                cur_mk = mk;
                improved_any = true;
            }
        }
        if !improved_any {
            break;
        }
    }
}

/// Permutation schedule of `seq` on the common route.
fn build_perm_solution_from_seq(
    seq: &[usize],
    route: &[usize],
    pt: &[Vec<u32>],
    num_jobs: usize,
    num_machines: usize,
) -> Solution {
    let mut job_schedule: Vec<Vec<(usize, u32)>> = vec![Vec::with_capacity(route.len()); num_jobs];
    let mut machine_ready = vec![0u32; num_machines];
    for &j in seq {
        if j >= num_jobs {
            continue;
        }
        let row = &pt[j];
        let mut prev_end = 0u32;
        for (op_idx, &m) in route.iter().enumerate() {
            if op_idx >= row.len() || m >= num_machines {
                break;
            }
            let st = prev_end.max(machine_ready[m]);
            job_schedule[j].push((m, st));
            let end = st.saturating_add(row[op_idx]);
            machine_ready[m] = end;
            prev_end = end;
        }
    }
    Solution { job_schedule }
}

/// Jobs ordered by the start of their first operation, ties by index.
fn order_from_solution_first_op_start(sol: &Solution, num_jobs: usize) -> Vec<usize> {
    let mut v: Vec<(u32, usize)> = Vec::with_capacity(num_jobs);
    for j in 0..num_jobs {
        if let Some(t) = sol.job_schedule.get(j).and_then(|ops| ops.first()).map(|x| x.1) {
            v.push((t, j));
        }
    }
    v.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut seen = vec![false; num_jobs];
    let mut ord: Vec<usize> = Vec::with_capacity(num_jobs);
    for &(_, j) in &v {
        if j < num_jobs && !seen[j] {
            seen[j] = true;
            ord.push(j);
        }
    }
    for j in 0..num_jobs {
        if !seen[j] {
            ord.push(j);
        }
    }
    ord
}

/// Best NEH sequence over the LPT, Palmer and CDS seed orders, each polished by reinsertion.
fn neh_best_sequence(
    route: &[usize],
    pt: &[Vec<u32>],
    num_jobs: usize,
    num_machines: usize,
) -> Result<Vec<usize>> {
    let ops = route.len();
    if ops == 0 || pt.len() != num_jobs {
        return Err(anyhow!("Invalid flow data"));
    }
    let mut candidates: Vec<Vec<usize>> = Vec::new();
    {
        let mut jobs: Vec<usize> = (0..num_jobs).collect();
        jobs.sort_unstable_by(|&a, &b| {
            let sa: u32 = pt[a].iter().copied().sum();
            let sb: u32 = pt[b].iter().copied().sum();
            sb.cmp(&sa).then_with(|| a.cmp(&b))
        });
        candidates.push(jobs);
    }
    candidates.push(palmer_order(pt));
    for o in cds_orders(pt) {
        if o.len() == num_jobs {
            candidates.push(o);
        }
    }
    let mut best_seq: Vec<usize> = (0..num_jobs).collect();
    let mut best_mk: u32 = u32::MAX;
    if route_is_unique(route, num_machines) {
        let mut buf = PermInsBuf::default();
        for ord in candidates.iter().filter(|o| o.len() == num_jobs) {
            let mut seq = neh_build_perm(ord, pt, ops, &mut buf);
            improve_perm_seq(&mut seq, pt, 8, &mut buf);
            buf.ensure(seq.len(), ops);
            let mk = flow_makespan(&seq, pt, &mut buf.comp[..ops]);
            if mk < best_mk {
                best_mk = mk;
                best_seq = seq;
            }
        }
    } else {
        let mut buf = ReentrantInsBuf::default();
        buf.set_route(route, num_machines);
        let mut mready = vec![0u32; num_machines];
        for ord in candidates.iter().filter(|o| o.len() == num_jobs) {
            let mut seq = neh_build_reentrant(ord, pt, &mut buf);
            improve_reentrant_seq(&mut seq, route, pt, &mut mready, &mut buf);
            let mk = reentrant_makespan(&seq, route, pt, &mut mready);
            if mk < best_mk {
                best_mk = mk;
                best_seq = seq;
            }
        }
    }
    Ok(best_seq)
}

// ------------------------------------------------------------------------------------------------
// Permutation route only: adjacent-swap polish and iterated greedy
// ------------------------------------------------------------------------------------------------

/// One left-to-right pass of adjacent swaps, each kept when it does not worsen the makespan.
fn perm_adjacent_swap_polish(seq: &mut [usize], pt: &[Vec<u32>], cur_mk: u32, buf: &mut PermInsBuf) -> u32 {
    let n = seq.len();
    let m = pt.first().map(|r| r.len()).unwrap_or(0);
    if n <= 1 || m == 0 {
        return cur_mk;
    }
    buf.ensure(n, m);
    let tail = &mut buf.b;
    let base_n = n * m;
    for k in 0..m {
        tail[base_n + k] = 0;
    }
    for t in (0..n).rev() {
        let row = &pt[seq[t]];
        let base = t * m;
        let next = (t + 1) * m;
        tail[base + (m - 1)] = tail[next + (m - 1)].saturating_add(row[m - 1]);
        for k in (0..m - 1).rev() {
            tail[base + k] = tail[base + k + 1].max(tail[next + k]).saturating_add(row[k]);
        }
    }
    let prefix = &mut buf.comp[..m];
    let first_after = &mut buf.e[..m];
    let swap_after = &mut buf.f[..m];
    prefix.fill(0);
    let mut best = cur_mk;
    for i in 0..(n - 1) {
        let row_a = &pt[seq[i]];
        let row_b = &pt[seq[i + 1]];
        first_after[0] = prefix[0].saturating_add(row_b[0]);
        for k in 1..m {
            first_after[k] = first_after[k - 1].max(prefix[k]).saturating_add(row_b[k]);
        }
        swap_after[0] = first_after[0].saturating_add(row_a[0]);
        for k in 1..m {
            swap_after[k] = swap_after[k - 1].max(first_after[k]).saturating_add(row_a[k]);
        }
        let base = (i + 2) * m;
        let mut mk2 = 0u32;
        for k in 0..m {
            mk2 = mk2.max(swap_after[k].saturating_add(tail[base + k]));
        }
        if mk2 <= best {
            seq.swap(i, i + 1);
            best = mk2;
            prefix.copy_from_slice(first_after);
        } else {
            prefix[0] = prefix[0].saturating_add(row_a[0]);
            for k in 1..m {
                prefix[k] = prefix[k].max(prefix[k - 1]).saturating_add(row_a[k]);
            }
        }
    }
    best
}

/// Destruction of `d` jobs, best-insertion reconstruction, adjacent-swap polish, and simulated
/// annealing acceptance with a geometric temperature.
fn iterated_greedy_search(
    init: &[usize],
    pt: &[Vec<u32>],
    iters: usize,
    d: usize,
    rng: &mut SmallRng,
) -> Vec<usize> {
    let n = init.len();
    if n <= 2 {
        return init.to_vec();
    }
    let m = pt.first().map(|r| r.len()).unwrap_or(0);
    if m == 0 {
        return init.to_vec();
    }
    let mut buf = PermInsBuf::default();
    buf.ensure(n, m);
    let mut cur = init.to_vec();
    let mut best = cur.clone();
    let mut cur_mk = flow_makespan(&cur, pt, &mut buf.comp[..m]);
    let mut best_mk = cur_mk;
    let mut temp = (cur_mk as f64) * 0.10 + 1.0;
    let dd = d.clamp(2, n.saturating_sub(1));
    let mut idxs: Vec<usize> = Vec::with_capacity(dd);
    let mut removed: Vec<usize> = Vec::with_capacity(dd);
    let mut partial: Vec<usize> = Vec::with_capacity(n);
    let mut remove_mark: Vec<u32> = vec![0u32; n];
    let mut mark_epoch: u32 = 1;
    for _ in 0..iters.max(1) {
        idxs.clear();
        while idxs.len() < dd {
            let x = rng.gen_range(0..n);
            if !idxs.iter().any(|&y| y == x) {
                idxs.push(x);
            }
        }
        idxs.sort_unstable();
        removed.clear();
        partial.clear();
        if mark_epoch == u32::MAX {
            remove_mark.fill(0);
            mark_epoch = 1;
        }
        let epoch = mark_epoch;
        mark_epoch += 1;
        for &ix in &idxs {
            remove_mark[ix] = epoch;
        }
        for &ix in idxs.iter().rev() {
            removed.push(cur[ix]);
        }
        for (pos, &job) in cur.iter().enumerate() {
            if remove_mark[pos] != epoch {
                partial.push(job);
            }
        }
        removed.shuffle(rng);
        for &j in &removed {
            let (pos, _) = perm_best_insert_pos(&partial, j, pt, m, &mut buf);
            partial.insert(pos, j);
        }
        let mut cand_mk = flow_makespan(&partial, pt, &mut buf.comp[..m]);
        if partial.len() >= 2 {
            cand_mk = perm_adjacent_swap_polish(&mut partial, pt, cand_mk, &mut buf);
        }
        if cand_mk < best_mk {
            best_mk = cand_mk;
            best.clear();
            best.extend_from_slice(&partial);
        }
        if cand_mk <= cur_mk {
            cur.clear();
            cur.extend_from_slice(&partial);
            cur_mk = cand_mk;
        } else {
            let delta = (cand_mk - cur_mk) as f64;
            let prob = (-delta / temp).exp();
            if rng.gen::<f64>() < prob {
                cur.clear();
                cur.extend_from_slice(&partial);
                cur_mk = cand_mk;
            }
        }
        temp = (temp * 0.995).max(1.0);
    }
    best
}

// ------------------------------------------------------------------------------------------------
// Strict-order decoder: non-delay list scheduling of a job priority per route position group
// ------------------------------------------------------------------------------------------------

/// Priority of one job at one route position: `rank[job * stride + col_group[op]]`.
#[derive(Clone, Copy)]
struct RankView<'a> {
    rank: &'a [usize],
    stride: usize,
    col_group: &'a [usize],
}

/// Route order, machine identity and no overlap on a machine (debug builds only).
#[cfg(debug_assertions)]
fn schedule_is_valid(pre: &Pre, route: &[usize], js: &[Vec<(usize, u32)>]) -> bool {
    let mut per_machine: Vec<Vec<(u32, u32)>> = vec![Vec::new(); pre.num_machines];
    for job in 0..pre.num_jobs {
        if js[job].len() != pre.job_ops_len[job] {
            return false;
        }
        let product = pre.job_products[job];
        let mut prev_end = 0u32;
        for (op, &(m, st)) in js[job].iter().enumerate() {
            let info = pre.product_ops[product][op];
            if route[op] != m || info.machine != m || st < prev_end {
                return false;
            }
            prev_end = st + info.pt;
            per_machine[m].push((st, prev_end));
        }
    }
    for v in per_machine.iter_mut() {
        v.sort_unstable();
        for i in 1..v.len() {
            if v[i].0 < v[i - 1].1 {
                return false;
            }
        }
    }
    true
}

/// Non-delay decoder on the flat tables. Returns the makespan and, with `BUILD_SOL`, the
/// schedule. With `USE_LB` the run aborts as soon as a lower bound reaches `bound`.
fn strict_run_flat<const BUILD_SOL: bool, const USE_LB: bool>(
    pre: &Pre,
    ff: &FlowFast,
    rank: RankView<'_>,
    bound: u32,
) -> Result<(u32, Vec<Vec<(usize, u32)>>)> {
    let n = pre.num_jobs.min(64);
    let r = ff.r;
    let route = &ff.route[..64];
    let pt = &ff.pt[..4096];
    let tail_next = &ff.tail_next[..4096];
    let stride = rank.stride;
    let mut rk = [0u8; 4096];
    {
        let src = &rank.rank[..n * stride];
        for j in 0..n {
            let row = &src[j * stride..(j + 1) * stride];
            for (g, &v) in row.iter().enumerate() {
                rk[(j << 6) | (g & 63)] = (v & 63) as u8;
            }
        }
    }
    let mut cg = [0u8; 64];
    for (k, &g) in rank.col_group[..r].iter().enumerate() {
        cg[k & 63] = (g & 63) as u8;
    }
    // ready[(m << 6) | k]: jobs whose next operation is on machine m with priority rank k.
    let mut ready = [0u64; 4096];
    let mut ready_ranks = [0u64; 64];
    let mut mach_avail = [0u32; 64];
    mach_avail[DUMMY_MACHINE] = u32::MAX;
    let mut job_next = [0u32; 64];
    let mut rem_load = [0u32; 64];
    if USE_LB {
        rem_load.copy_from_slice(&ff.mload[..64]);
    }
    // Circular calendar of operation ends: cal_head[slot] chains the jobs ending at that time.
    let mut cal_head = [NONE16; CAL_SLOTS];
    let mut cal_next = [NONE16; 64];
    let mut cal_occ = [0u64; CAL_WORDS];
    let mut job_schedule: Vec<Vec<(usize, u32)>> = if BUILD_SOL {
        pre.job_ops_len
            .iter()
            .map(|&len| Vec::with_capacity(len))
            .collect()
    } else {
        Vec::new()
    };
    let m0 = route[0] & 63;
    for j in 0..n {
        let k = rk[(j << 6) | (cg[0] as usize)] as usize;
        ready[(m0 << 6) | k] |= 1u64 << j;
        ready_ranks[m0] |= 1u64 << k;
    }
    let mut touched: u64 = 1u64 << m0;
    let mut remaining = n * r;
    let mut t = 0u32;
    let mut makespan = 0u32;
    loop {
        if touched == 0 {
            let s0 = ((t as usize) + 1) & (CAL_SLOTS - 1);
            let mut wi = s0 >> 6;
            let mut w = cal_occ[wi] & (u64::MAX << (s0 & 63));
            if w == 0 {
                let mut steps = 0usize;
                loop {
                    wi = (wi + 1) & (CAL_WORDS - 1);
                    w = cal_occ[wi];
                    if w != 0 || steps + 1 >= CAL_WORDS {
                        break;
                    }
                    steps += 1;
                }
                if w == 0 {
                    if remaining == 0 {
                        break;
                    }
                    return Err(anyhow!("stalled"));
                }
            }
            let slot = (wi << 6) | (w.trailing_zeros() as usize & 63);
            let d = slot.wrapping_sub(t as usize) & (CAL_SLOTS - 1);
            t = t.saturating_add(d as u32);
            cal_occ[wi] &= !(1u64 << (slot & 63));
            let mut j = cal_head[slot];
            cal_head[slot] = NONE16;
            while j != NONE16 {
                let jj = (j & 63) as usize;
                let op = job_next[jj] as usize;
                let mf = route[op.wrapping_sub(1) & 63] & 63;
                touched |= ((ready_ranks[mf] != 0) as u64) << mf;
                let m2 = route[op & 63] & 63;
                let k = rk[(jj << 6) | (cg[op & 63] as usize)] as usize;
                ready[(m2 << 6) | k] |= 1u64 << jj;
                ready_ranks[m2] |= 1u64 << k;
                touched |= ((mach_avail[m2] <= t) as u64) << m2;
                j = cal_next[jj];
            }
            continue;
        }
        let m = (touched.trailing_zeros() & 63) as usize;
        touched &= touched.wrapping_sub(1);
        let rks = ready_ranks[m];
        if rks == 0 {
            continue;
        }
        let k = (rks.trailing_zeros() & 63) as usize;
        let wi = (m << 6) | k;
        let w = ready[wi];
        let j = (w.trailing_zeros() & 63) as usize;
        let w2 = w & w.wrapping_sub(1);
        ready[wi] = w2;
        ready_ranks[m] = rks & !(((w2 == 0) as u64) << k);
        let op = job_next[j] as usize;
        let ji = (j << 6) | (op & 63);
        let p = pt[ji];
        let end = t.saturating_add(p);
        mach_avail[m] = end;
        if USE_LB {
            let lb = t
                .saturating_add(rem_load[m])
                .max(end.saturating_add(tail_next[ji]));
            if lb >= bound {
                return Err(anyhow!("bound"));
            }
            rem_load[m] -= p;
        } else if !BUILD_SOL && end >= bound {
            return Err(anyhow!("bound"));
        }
        if BUILD_SOL {
            job_schedule[j].push((m, t));
        }
        job_next[j] = (op + 1) as u32;
        remaining -= 1;
        makespan = makespan.max(end);
        let slot = (end as usize) & (CAL_SLOTS - 1);
        cal_next[j] = cal_head[slot];
        cal_head[slot] = j as u16;
        cal_occ[slot >> 6] |= 1u64 << (slot & 63);
    }
    if remaining != 0 {
        return Err(anyhow!("stalled"));
    }
    #[cfg(debug_assertions)]
    if BUILD_SOL {
        debug_assert!(
            schedule_is_valid(pre, &ff.route[..r], &job_schedule),
            "strict_run_flat built an invalid schedule"
        );
    }
    Ok((makespan, job_schedule))
}

fn strict_run<const BUILD_SOL: bool>(
    pre: &Pre,
    rank: RankView<'_>,
    bound: u32,
) -> Result<(u32, Vec<Vec<(usize, u32)>>)> {
    let ff = pre.flow_fast.as_ref().ok_or_else(|| anyhow!("no flat tables"))?;
    let num_jobs = pre.num_jobs;
    let stride = rank.stride;
    let fits = (1..=64).contains(&stride)
        && rank.col_group.len() >= ff.r
        && rank.rank.len() >= num_jobs * stride
        && rank.col_group[..ff.r].iter().all(|&g| g < stride)
        && rank.rank[..num_jobs * stride].iter().all(|&v| v < 64);
    if !fits {
        return Err(anyhow!("rank view out of range"));
    }
    if !BUILD_SOL && bound != u32::MAX {
        strict_run_flat::<BUILD_SOL, true>(pre, ff, rank, bound)
    } else {
        strict_run_flat::<BUILD_SOL, false>(pre, ff, rank, bound)
    }
}

/// One rank per job: every route position reads the same value.
const ZERO_GROUPS: [usize; 64] = [0usize; 64];

fn strict_makespan_bounded(pre: &Pre, rank: &[usize], bound: u32) -> Result<u32> {
    strict_run::<false>(
        pre,
        RankView {
            rank,
            stride: 1,
            col_group: &ZERO_GROUPS,
        },
        bound,
    )
    .map(|x| x.0)
}

fn strict_simulate(pre: &Pre, rank: &[usize]) -> Result<(Solution, u32)> {
    strict_run::<true>(
        pre,
        RankView {
            rank,
            stride: 1,
            col_group: &ZERO_GROUPS,
        },
        u32::MAX,
    )
    .map(|(makespan, job_schedule)| (Solution { job_schedule }, makespan))
}

/// Fingerprint of the product sequence: jobs of one product have identical processing times.
#[inline]
fn order_fingerprint(pre: &Pre, order: &[usize]) -> u64 {
    let mut h = 1469598103934665603u64 ^ (order.len() as u64);
    for &j in order {
        h ^= (pre.job_products[j] as u64).wrapping_add(1);
        h = h.wrapping_mul(1099511628211u64);
    }
    h
}

/// Memoised makespans, keyed by product-sequence fingerprint with a fixed hasher.
type EvalCache = HashMap<u64, (Vec<usize>, u32), BuildHasherDefault<DefaultHasher>>;

/// Bounded makespan of `order`, memoised by product sequence. A bound abort is memoised as
/// `u32::MAX`: the incumbents only decrease, so it stays a rejection for every later bound.
fn strict_makespan_cached(
    pre: &Pre,
    order: &[usize],
    rank: &mut [usize],
    eval_cache: &mut EvalCache,
    cache_cap: usize,
    bound: u32,
) -> Result<u32> {
    let key = order_fingerprint(pre, order);
    if let Some((cached_products, mk)) = eval_cache.get(&key) {
        if cached_products.len() == order.len()
            && cached_products
                .iter()
                .zip(order.iter())
                .all(|(&p, &j)| p == pre.job_products[j])
        {
            return Ok(*mk);
        }
    }
    for (pos, &j) in order.iter().enumerate() {
        rank[j] = pos;
    }
    let mk = strict_makespan_bounded(pre, rank, bound).unwrap_or(u32::MAX);
    if cache_cap > 0 {
        if !eval_cache.contains_key(&key) && eval_cache.len() >= cache_cap {
            eval_cache.clear();
        }
        eval_cache.insert(key, (order.iter().map(|&j| pre.job_products[j]).collect(), mk));
    }
    Ok(mk)
}

/// Writes `order` with the segment `order[start..start+seg_len]` moved to position `ins` of the
/// remaining jobs into `out`.
fn segment_move(order: &[usize], start: usize, seg_len: usize, ins: usize, out: &mut [usize]) {
    let rem_len = order.len() - seg_len;
    let mut o = 0usize;
    for r in 0..=rem_len {
        if r == ins {
            for t in 0..seg_len {
                out[o] = order[start + t];
                o += 1;
            }
        }
        if r == rem_len {
            break;
        }
        out[o] = order[if r < start { r } else { r + seg_len }];
        o += 1;
    }
}

/// Maximum number of per-group swap attempts after the single-order search.
const PASS_SWAP_BUDGET: usize = 10_500;

/// Route positions grouped at each revisit of the most loaded machine: (col_group, ngroups).
fn route_pass_groups(pre: &Pre, route: &[usize]) -> (Vec<usize>, usize) {
    let num_machines = pre.num_machines;
    let rl = route.len();
    let mut col_group = vec![0usize; rl];
    if pre.flow_mload.len() != num_machines || num_machines == 0 {
        return (col_group, 1);
    }
    let mut bn = 0usize;
    for m in 1..num_machines {
        if pre.flow_mload[m] > pre.flow_mload[bn] {
            bn = m;
        }
    }
    let visits: Vec<usize> = (0..rl).filter(|&k| route[k] == bn).collect();
    if visits.len() < 2 {
        return (col_group, 1);
    }
    let mut g = 0usize;
    let mut next_cut = 1usize;
    for k in 0..rl {
        if next_cut < visits.len() && k == visits[next_cut] {
            g += 1;
            next_cut += 1;
        }
        col_group[k] = g;
    }
    (col_group, g + 1)
}

/// Outcome of the per-group search: the final ranks, the ranks at the last strict improvement,
/// and the group layout both are read with.
struct GroupWalk {
    rank: Vec<usize>,
    ngroups: usize,
    col_group: Vec<usize>,
    mk: u32,
    /// `rank` as it was when `mk` was last lowered (the initial ranks until then).
    last_gain: Vec<usize>,
}

/// Random swaps over one job order per route position group, seeded with `base_order`.
/// `None` when the route has a single group or the instance has fewer than three jobs.
fn pass_group_search(
    pre: &Pre,
    seed: &[u8; 32],
    route: &[usize],
    base_order: &[usize],
    base_mk: u32,
) -> Option<GroupWalk> {
    let n = pre.num_jobs;
    if n < 3 || base_order.len() != n {
        return None;
    }
    let (col_group, s) = route_pass_groups(pre, route);
    if s < 2 {
        return None;
    }
    let mut orders: Vec<Vec<usize>> = (0..s).map(|_| base_order.to_vec()).collect();
    let mut rank = vec![0usize; n * s];
    for g in 0..s {
        for (pos, &j) in orders[g].iter().enumerate() {
            rank[j * s + g] = pos;
        }
    }
    let mut cur_mk = base_mk;
    let mut last_gain = rank.clone();
    let mut seed = *seed;
    seed[0] ^= 0x5D;
    let mut rng = SmallRng::from_seed(seed);
    for _ in 0..PASS_SWAP_BUDGET {
        let g = rng.gen_range(0..s);
        let i = rng.gen_range(0..n);
        let j = rng.gen_range(0..n);
        if i == j {
            continue;
        }
        orders[g].swap(i, j);
        rank[orders[g][i] * s + g] = i;
        rank[orders[g][j] * s + g] = j;
        let bound = cur_mk.saturating_add(1);
        let view = RankView {
            rank: &rank,
            stride: s,
            col_group: &col_group,
        };
        let mk = strict_run::<false>(pre, view, bound).map_or(u32::MAX, |x| x.0);
        if mk < bound {
            if mk < cur_mk {
                last_gain.clone_from(&rank);
            }
            cur_mk = mk;
        } else {
            orders[g].swap(i, j);
            rank[orders[g][i] * s + g] = i;
            rank[orders[g][j] * s + g] = j;
        }
    }
    Some(GroupWalk {
        rank,
        ngroups: s,
        col_group,
        mk: cur_mk,
        last_gain,
    })
}

/// Schedule decoded from `rank` under the group layout of `walk`.
fn group_schedule(pre: &Pre, walk: &GroupWalk, rank: &[usize]) -> Option<(Solution, u32)> {
    let view = RankView {
        rank,
        stride: walk.ngroups,
        col_group: &walk.col_group,
    };
    strict_run::<true>(pre, view, u32::MAX)
        .ok()
        .map(|(mk, js)| (Solution { job_schedule: js }, mk))
}

/// Local search over strict job orders: seed orders, single-job reinsertion, segment moves
/// with a sliding focus window, random swaps, segment reversals, then the per-group search.
/// Returns the best schedule, its makespan, and further descent starts of equal makespan.
fn strict_best_by_order_search(
    pre: &Pre,
    seed: &[u8; 32],
    route: &[usize],
    pt: &[Vec<u32>],
) -> Result<(Solution, u32, Vec<(Solution, u32)>)> {
    const MAX_PASSES: usize = 6;
    let n = pre.num_jobs;
    let mut cand_orders: Vec<Vec<usize>> = Vec::new();
    let total: Vec<u32> = pt.iter().map(|row| row.iter().copied().sum()).collect();
    {
        let mut lpt: Vec<usize> = (0..n).collect();
        lpt.sort_unstable_by(|&a, &b| total[b].cmp(&total[a]).then_with(|| a.cmp(&b)));
        cand_orders.push(lpt);
    }
    {
        let mut spt: Vec<usize> = (0..n).collect();
        spt.sort_unstable_by(|&a, &b| total[a].cmp(&total[b]).then_with(|| a.cmp(&b)));
        cand_orders.push(spt);
    }
    cand_orders.push(palmer_order(pt));
    for o in cds_orders(pt) {
        if o.len() == n {
            cand_orders.push(o);
        }
    }
    {
        let mut seed = *seed;
        seed[0] ^= 0x3C;
        let mut rng = SmallRng::from_seed(seed);
        for _ in 0..100usize {
            let mut r: Vec<usize> = (0..n).collect();
            r.shuffle(&mut rng);
            cand_orders.push(r);
        }
    }
    let cache_cap = n.saturating_mul(cand_orders.len().max(1)).max(1);
    let mut eval_cache = EvalCache::with_capacity_and_hasher(cand_orders.len().max(1), Default::default());
    let mut rank = vec![0usize; n];
    let mut best_mk = u32::MAX;
    let mut best_order: Vec<usize> = (0..n).collect();
    for ord in cand_orders.iter() {
        if ord.len() != n {
            continue;
        }
        let mk = strict_makespan_cached(pre, ord, &mut rank, &mut eval_cache, cache_cap, best_mk)?;
        if mk < best_mk {
            best_mk = mk;
            best_order.clone_from(ord);
        }
    }

    let mut cand_order: Vec<usize> = vec![0usize; n];
    for _ in 0..2 {
        let mut improved = false;
        for i in 0..n {
            let job = best_order[i];
            let mut best_pos = i;
            let mut best_local_mk = best_mk;
            for pos in 0..n {
                if pos == i {
                    continue;
                }
                if pos < i {
                    cand_order[..pos].copy_from_slice(&best_order[..pos]);
                    cand_order[pos] = job;
                    cand_order[pos + 1..=i].copy_from_slice(&best_order[pos..i]);
                    cand_order[i + 1..].copy_from_slice(&best_order[i + 1..]);
                } else {
                    cand_order[..i].copy_from_slice(&best_order[..i]);
                    cand_order[i..pos].copy_from_slice(&best_order[i + 1..=pos]);
                    cand_order[pos] = job;
                    cand_order[pos + 1..].copy_from_slice(&best_order[pos + 1..]);
                }
                let mk = strict_makespan_cached(
                    pre,
                    &cand_order,
                    &mut rank,
                    &mut eval_cache,
                    cache_cap,
                    best_local_mk,
                )?;
                if mk < best_local_mk {
                    best_local_mk = mk;
                    best_pos = pos;
                }
            }
            if best_local_mk < best_mk {
                best_mk = best_local_mk;
                if best_pos < i {
                    best_order[best_pos..=i].rotate_right(1);
                } else if best_pos > i {
                    best_order[i..=best_pos].rotate_left(1);
                }
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }

    let mut order = best_order.clone();
    let seg_lens: [usize; 2] = [2, 3];
    let base_window = (n / MAX_PASSES).max(MAX_PASSES).min(n);
    let mut focus: Option<(usize, usize)> = None;
    for _ in 0..MAX_PASSES {
        let (start_lo, start_hi) = focus.unwrap_or((0usize, base_window));
        let start_hi = start_hi.min(n);
        let mut best_local_mk = best_mk;
        let mut best_move: Option<(usize, usize, usize)> = None;
        for &seg_len in &seg_lens {
            if seg_len > n {
                continue;
            }
            let max_start = n - seg_len;
            let s0 = start_lo.min(max_start + 1);
            let s1 = start_hi.min(max_start + 1);
            if s0 >= s1 {
                continue;
            }
            let rem_len = n - seg_len;
            for start in s0..s1 {
                for ins in 0..=rem_len {
                    if ins == start {
                        continue;
                    }
                    segment_move(&order, start, seg_len, ins, &mut cand_order);
                    let mk = strict_makespan_cached(
                        pre,
                        &cand_order,
                        &mut rank,
                        &mut eval_cache,
                        cache_cap,
                        best_local_mk,
                    )?;
                    if mk < best_local_mk {
                        best_local_mk = mk;
                        best_move = Some((seg_len, start, ins));
                    }
                }
            }
        }
        let Some((seg_len, start, ins)) = best_move else {
            break;
        };
        segment_move(&order, start, seg_len, ins, &mut cand_order);
        order.clone_from(&cand_order);
        best_mk = best_local_mk;
        best_order.clone_from(&order);
        let min_pos = start.min(ins);
        let max_pos = start.max(ins);
        let lo = min_pos.saturating_sub(base_window.min(MAX_PASSES));
        let hi = (max_pos + base_window).min(n);
        focus = Some((lo, hi));
    }

    order.clone_from(&best_order);
    for (pos, &j) in order.iter().enumerate() {
        rank[j] = pos;
    }
    {
        let mut seed = *seed;
        seed[0] ^= 0xA5;
        let mut rng = SmallRng::from_seed(seed);
        let swap_budget = (n * 12).clamp(200, 800);
        for _ in 0..swap_budget {
            let i = rng.gen_range(0..n);
            let j = rng.gen_range(0..n);
            if i == j {
                continue;
            }
            order.swap(i, j);
            rank[order[i]] = i;
            rank[order[j]] = j;
            let mk = strict_makespan_cached(pre, &order, &mut rank, &mut eval_cache, cache_cap, best_mk)?;
            if mk < best_mk {
                best_mk = mk;
                best_order.clone_from(&order);
            } else {
                order.swap(i, j);
                rank[order[i]] = i;
                rank[order[j]] = j;
            }
        }
    }

    order.clone_from(&best_order);
    for (pos, &j) in order.iter().enumerate() {
        rank[j] = pos;
    }
    if n >= 2 {
        let max_seg = 5usize.min(n);
        for _ in 0..2 {
            let mut improved = false;
            for seg_len in 2..=max_seg {
                for start in 0..=(n - seg_len) {
                    order[start..start + seg_len].reverse();
                    for k in start..start + seg_len {
                        rank[order[k]] = k;
                    }
                    let mk =
                        strict_makespan_cached(pre, &order, &mut rank, &mut eval_cache, cache_cap, best_mk)?;
                    if mk < best_mk {
                        best_mk = mk;
                        best_order.clone_from(&order);
                        improved = true;
                    } else {
                        order[start..start + seg_len].reverse();
                        for k in start..start + seg_len {
                            rank[order[k]] = k;
                        }
                    }
                }
            }
            if !improved {
                break;
            }
        }
    }

    for (pos, &j) in best_order.iter().enumerate() {
        rank[j] = pos;
    }
    if let Some(walk) = pass_group_search(pre, seed, route, &best_order, best_mk) {
        if walk.mk < best_mk {
            if let Some((sol, mk)) = group_schedule(pre, &walk, &walk.rank) {
                let mut seeds: Vec<(Solution, u32)> = Vec::with_capacity(2);
                if walk.last_gain != walk.rank {
                    if let Some(sibling) = group_schedule(pre, &walk, &walk.last_gain) {
                        seeds.push(sibling);
                    }
                }
                return Ok((sol, mk, seeds));
            }
        }
    }
    let (sol, mk) = strict_simulate(pre, &rank)?;
    Ok((sol, mk, Vec::new()))
}

// ------------------------------------------------------------------------------------------------
// GRASP: randomised list scheduling under twelve priority rules, with conflict boosting
// ------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Rule {
    BnHeavy,
    MostWork,
    EndTight,
    ShortestProc,
    LeastFlex,
    CriticalPath,
    Regret,
    EarliestStart,
    MachineBalance,
    SlackRatio,
    BackwardCritical,
    WeightedCompletion,
}

const GRASP_RULES: [Rule; 12] = [
    Rule::BnHeavy,
    Rule::MostWork,
    Rule::EndTight,
    Rule::ShortestProc,
    Rule::LeastFlex,
    Rule::CriticalPath,
    Rule::Regret,
    Rule::EarliestStart,
    Rule::MachineBalance,
    Rule::SlackRatio,
    Rule::BackwardCritical,
    Rule::WeightedCompletion,
];

/// Gain applied to the conflict boost of a candidate.
const CONFLICT_GAIN: f64 = 0.45 * 1.8;
/// Weight of the flow-shop insertion rank in every rule.
const FLOW_PREF_WEIGHT: f64 = 0.36;

/// Multiplier of the conflict boost.
struct AdaptiveBoost {
    boost_strength: f64,
    ema_delta: f64,
}

impl AdaptiveBoost {
    fn new() -> Self {
        AdaptiveBoost {
            boost_strength: 1.0,
            ema_delta: 0.0,
        }
    }

    fn update_from_test(&mut self, mk_boost: u32, mk_no_boost: u32) {
        let delta = (mk_no_boost as f64 - mk_boost as f64) / mk_boost.max(1) as f64;
        let lr = 0.05;
        self.ema_delta += lr * (delta - self.ema_delta);
        self.boost_strength = (1.0 + 0.1 * self.ema_delta).clamp(0.5, 2.0);
    }
}

#[derive(Clone, Copy)]
struct Cand {
    job: usize,
    machine: usize,
    pt: u32,
    score: f64,
}

#[derive(Clone, Copy)]
struct RawCand {
    job: usize,
    machine: usize,
    pt: u32,
    base_score: f64,
    rigidity: f64,
}

#[allow(clippy::too_many_arguments)]
fn score_candidate(
    pre: &Pre,
    rule: Rule,
    job: usize,
    product: usize,
    op_idx: usize,
    ops_rem: usize,
    pt: u32,
    time: u32,
    best_end: u32,
    progress: f64,
    job_bias: f64,
    dynamic_load: f64,
    jitter: f64,
) -> f64 {
    let ops = &pre.product_ops[product];
    let rem_min = pre.product_suf_min[product][op_idx] as f64;
    let rem_bn = pre.product_suf_bn[product][op_idx];
    let rem_min_n = rem_min / pre.horizon;
    let ops_n = (ops_rem as f64) / (pre.max_ops as f64).max(1.0);
    let end_n = (best_end as f64) / pre.time_scale;
    let proc_n = (pt as f64) / pre.avg_op_min;
    let reg_n = pre.regret_n;
    let density_n = ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min).clamp(0.0, 4.0);
    let next_term_raw = if op_idx + 1 < ops.len() {
        0.55 * (ops[op_idx + 1].pt as f64 / pre.horizon) + 0.45
    } else {
        0.0
    };
    let next_w_base = 0.12 + progress * progress * 0.28;
    let flow_term = FLOW_PREF_WEIGHT * pre.job_flow_pref[job] * (0.65 + 0.70 * (1.0 - progress));

    match rule {
        Rule::BnHeavy => {
            let bn_w = 0.90 * pre.bn_focus;
            let end_w = 0.65 + 0.70 * progress;
            let reg_w = (0.60 + 0.25 * (1.0 - progress)) * 0.85;
            let next_term = next_w_base * 0.55 * next_term_raw;
            (0.95 * rem_min_n)
                + (bn_w * rem_bn / pre.max_job_bn)
                + (0.10 * ops_n)
                + (reg_w * 2.2) * reg_n
                + 0.18
                + next_term
                - end_w * end_n
                - 0.18 * proc_n
                + 0.60 * job_bias
                + flow_term
                + jitter
        }
        Rule::MostWork => {
            let next_term = next_w_base * 0.25 * next_term_raw;
            (1.00 * rem_min) / pre.max_job_work + (0.12 * ops_n) + 0.18 + next_term - (0.62 * end_n)
                + (0.45 * job_bias)
                + flow_term
                + jitter
        }
        Rule::EndTight => {
            let end_w = 1.10 + 1.00 * progress;
            let cp_w = 1.15;
            let reg_w = (0.55 + 0.20 * (1.0 - progress)) * 0.85;
            let next_term = next_w_base * 0.45 * next_term_raw;
            (cp_w * rem_min_n) + 0.08 * ops_n + 0.18 + (reg_w * 2.2) * reg_n + next_term
                - end_w * end_n
                - 0.22 * proc_n
                + 0.55 * job_bias
                + flow_term
                + jitter
        }
        Rule::ShortestProc => {
            let next_term = next_w_base * 0.20 * next_term_raw;
            (-1.00 * proc_n) + (0.25 * rem_min_n) + 0.12 + next_term - (0.20 * end_n)
                + (0.25 * job_bias)
                + flow_term
                + jitter
        }
        Rule::LeastFlex => {
            let next_term = next_w_base * 0.20 * next_term_raw;
            1.00 + (0.28 * rem_min_n) + 0.22 + next_term - (0.55 * end_n)
                + (0.35 * job_bias)
                + flow_term
                + jitter
        }
        Rule::CriticalPath => {
            let next_term = next_w_base * 0.30 * next_term_raw;
            (1.03 * rem_min_n) + (0.10 * ops_n) + 0.24 + next_term - (0.70 * end_n)
                + (0.45 * job_bias)
                + flow_term
                + jitter
        }
        Rule::Regret => {
            let next_term = next_w_base * 0.25 * next_term_raw;
            (1.05 * reg_n) + (0.55 * rem_min_n) + 0.22 + next_term - (0.68 * end_n)
                + (0.35 * job_bias)
                + flow_term
                + jitter
        }
        Rule::EarliestStart => {
            let start_n = (time as f64) / pre.time_scale;
            let next_term = next_w_base * 0.20 * next_term_raw;
            -(1.20 * start_n) + (0.40 * rem_min_n) + 0.15 + next_term - (0.30 * proc_n)
                + (0.30 * job_bias)
                + flow_term
                + jitter
        }
        Rule::MachineBalance => {
            let load_n = dynamic_load / pre.avg_machine_load;
            let next_term = next_w_base * 0.20 * next_term_raw;
            -(0.80 * load_n) + (0.50 * rem_min_n) + 0.25 + next_term - (0.45 * end_n)
                + (0.35 * job_bias)
                + flow_term
                + jitter
        }
        Rule::SlackRatio => {
            let time_to_horizon = (pre.horizon - time as f64).max(1.0);
            let cr = (rem_min / time_to_horizon).clamp(0.0, 4.0);
            let next_term = next_w_base * 0.25 * next_term_raw;
            (1.10 * cr) + (0.35 * rem_min_n) + 0.20 + next_term - (0.55 * end_n)
                + (0.40 * job_bias)
                + flow_term
                + jitter
        }
        Rule::BackwardCritical => {
            let bn_suf = rem_bn / pre.max_job_bn;
            let next_term = next_w_base * 0.30 * next_term_raw;
            (1.15 * bn_suf) + (0.45 * density_n) + 0.20 + next_term - (0.60 * end_n)
                + (0.40 * job_bias)
                + flow_term
                + jitter
        }
        Rule::WeightedCompletion => {
            let work_n = rem_min / pre.max_job_work;
            let wspt = if best_end > 0 {
                work_n / (best_end as f64 / pre.time_scale).max(0.01)
            } else {
                work_n
            };
            let next_term = next_w_base * 0.20 * next_term_raw;
            (1.20 * wspt) + (0.30 * rem_min_n) + 0.15 + next_term - (0.40 * end_n)
                + (0.35 * job_bias)
                + flow_term
                + jitter
        }
    }
}

/// Picks one of three score tiers uniformly, then one candidate of the tier uniformly.
#[inline]
fn choose_from_top_weighted(rng: &mut SmallRng, top: &[Cand]) -> Cand {
    let n = top.len();
    if n <= 1 {
        return top[0];
    }
    if n == 2 {
        return top[rng.gen_range(0..2)];
    }
    let b1 = n / 3;
    let b2 = (2 * n) / 3;
    let mut ranges: [(usize, usize); 3] = [(0, b1), (b1, b2), (b2, n)];
    let mut cnt = 0usize;
    for i in 0..3 {
        if ranges[i].0 < ranges[i].1 {
            ranges[cnt] = ranges[i];
            cnt += 1;
        }
    }
    let (s, e) = ranges[rng.gen_range(0..cnt)];
    top[s + rng.gen_range(0..(e - s))]
}

/// Non-delay list scheduling: whenever machines are free, the candidates of every free machine
/// are scored by `rule`, boosted by the contention on their machine, and one is placed (the best
/// when `k == 0`, otherwise a weighted draw among the `k` best). `target_mk` penalises
/// candidates whose remaining work already overshoots it.
fn construct_solution_conflict(
    pre: &Pre,
    rule: Rule,
    k: usize,
    target_mk: u32,
    rng: &mut SmallRng,
    boost_strength: f64,
    job_bias: Option<&[f64]>,
) -> Result<(Solution, u32)> {
    let num_jobs = pre.num_jobs;
    let num_machines = pre.num_machines;
    if num_jobs > 64 {
        return Err(anyhow!("at most 64 jobs"));
    }
    let mut job_next_op = vec![0usize; num_jobs];
    let mut job_ready_time = vec![0u32; num_jobs];
    let mut machine_avail = vec![0u32; num_machines];
    let mut machine_load = pre.machine_load0.clone();
    let mut job_schedule: Vec<Vec<(usize, u32)>> = pre
        .job_ops_len
        .iter()
        .map(|&len| Vec::with_capacity(len))
        .collect();
    let mut remaining_ops = pre.total_ops;
    let mut time = 0u32;
    let mut demand: Vec<u16> = vec![0u16; num_machines];
    let mut raw_by_machine: Vec<Vec<RawCand>> = (0..num_machines).map(|_| Vec::with_capacity(12)).collect();
    let mut idle_machines: Vec<usize> = Vec::with_capacity(num_machines);
    let mut future_jobs: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();
    // ready_on[m]: bitmask of the jobs whose next operation is on machine m and ready.
    let mut ready_on: Vec<u64> = vec![0u64; num_machines];
    let mut cand_list: Vec<usize> = Vec::with_capacity(num_jobs);
    let regc = pre.regret_n.clamp(0.0, 4.5);
    let cap_per_machine = if k == 0 { 12usize } else { (k + 6).min(12) };

    for job in 0..num_jobs {
        if pre.job_ops_len[job] > 0 {
            ready_on[pre.product_ops[pre.job_products[job]][0].machine] |= 1u64 << job;
        }
    }

    while remaining_ops > 0 {
        while let Some(Reverse((release, job))) = future_jobs.peek().copied() {
            if release > time {
                break;
            }
            future_jobs.pop();
            if job_next_op[job] < pre.job_ops_len[job] && job_ready_time[job] == release {
                ready_on[pre.product_ops[pre.job_products[job]][job_next_op[job]].machine] |= 1u64 << job;
            }
        }

        // Within a time step the set of free machines only shrinks: the chosen machine is
        // removed by hand, so the increasing order is preserved.
        idle_machines.clear();
        for m in 0..num_machines {
            if machine_avail[m] <= time {
                idle_machines.push(m);
            }
        }
        loop {
            if idle_machines.is_empty() {
                break;
            }
            for &m in &idle_machines {
                demand[m] = 0;
                raw_by_machine[m].clear();
            }
            let progress = 1.0 - (remaining_ops as f64) / (pre.total_ops as f64).max(1.0);

            let mut cm = 0u64;
            for &m in &idle_machines {
                cm |= ready_on[m];
            }
            cand_list.clear();
            while cm != 0 {
                cand_list.push(cm.trailing_zeros() as usize);
                cm &= cm.wrapping_sub(1);
            }
            for &job in cand_list.iter() {
                let op_idx = job_next_op[job];
                let product = pre.job_products[job];
                let op = pre.product_ops[product][op_idx];
                let m = op.machine;
                let best_end = time.saturating_add(op.pt);
                let ops_rem = pre.job_ops_len[job] - op_idx;
                let jb = job_bias.map(|v| v[job]).unwrap_or(0.0);
                let rem_min = pre.product_suf_min[product][op_idx] as f64;
                let excess = (best_end as f64 + rem_min - target_mk as f64).max(0.0);
                let cp_pen = (excess / pre.avg_op_min).clamp(0.0, 8.0);
                demand[m] = demand[m].saturating_add(1);
                let jitter = if k > 0 { rng.gen::<f64>() * 1e-9 } else { 0.0 };
                let base = score_candidate(
                    pre,
                    rule,
                    job,
                    product,
                    op_idx,
                    ops_rem,
                    op.pt,
                    time,
                    best_end,
                    progress,
                    jb,
                    machine_load[m],
                    jitter,
                );
                push_top_k_by(
                    &mut raw_by_machine[m],
                    RawCand {
                        job,
                        machine: m,
                        pt: op.pt,
                        base_score: base,
                        rigidity: cp_pen,
                    },
                    cap_per_machine,
                    |c| c.base_score,
                );
            }

            let denom = (idle_machines.len() as f64).max(1.0);
            let mut best: Option<Cand> = None;
            let mut top: Vec<Cand> = if k > 0 { Vec::with_capacity(k) } else { Vec::new() };
            for &m in &idle_machines {
                let dem = demand[m] as f64;
                if dem <= 0.0 || raw_by_machine[m].is_empty() {
                    continue;
                }
                let dem_n = ((dem - 1.0) / denom).clamp(0.0, 2.5);
                for rc in &raw_by_machine[m] {
                    let cp = rc.rigidity.clamp(0.0, 8.0);
                    let boost = CONFLICT_GAIN * dem_n * (0.85 * regc + 0.35 * cp) * boost_strength;
                    let c = Cand {
                        job: rc.job,
                        machine: rc.machine,
                        pt: rc.pt,
                        score: rc.base_score + boost,
                    };
                    if k == 0 {
                        if best.map_or(true, |bb| c.score > bb.score) {
                            best = Some(c);
                        }
                    } else {
                        push_top_k_by(&mut top, c, k, |x| x.score);
                    }
                }
            }

            let chosen = if k == 0 {
                match best {
                    Some(c) => c,
                    None => break,
                }
            } else {
                if top.is_empty() {
                    break;
                }
                choose_from_top_weighted(rng, &top)
            };

            let job = chosen.job;
            let machine = chosen.machine;
            let pt = chosen.pt;
            let product = pre.job_products[job];
            ready_on[machine] &= !(1u64 << job);
            let end_time = time.saturating_add(pt);
            job_schedule[job].push((machine, time));
            job_next_op[job] += 1;
            job_ready_time[job] = end_time;
            machine_avail[machine] = end_time;
            if end_time > time {
                if let Ok(pos_m) = idle_machines.binary_search(&machine) {
                    idle_machines.remove(pos_m);
                }
            }
            remaining_ops -= 1;
            let delta = pt as f64;
            if delta > 0.0 {
                let v = machine_load[machine] - delta;
                machine_load[machine] = if v > 0.0 { v } else { 0.0 };
            }
            if job_next_op[job] < pre.job_ops_len[job] {
                if end_time <= time {
                    ready_on[pre.product_ops[product][job_next_op[job]].machine] |= 1u64 << job;
                } else {
                    future_jobs.push(Reverse((end_time, job)));
                }
            }
            if remaining_ops == 0 {
                break;
            }
        }
        if remaining_ops == 0 {
            break;
        }
        let mut next_time: Option<u32> = None;
        for &t in &machine_avail {
            if t > time {
                next_time = Some(next_time.map_or(t, |b| b.min(t)));
            }
        }
        if let Some(Reverse((t, _))) = future_jobs.peek().copied() {
            if t > time {
                next_time = Some(next_time.map_or(t, |b| b.min(t)));
            }
        }
        time = next_time.ok_or_else(|| anyhow!("Stalled: no next event"))?;
    }
    let mk = machine_avail.into_iter().max().unwrap_or(0);
    Ok((Solution { job_schedule }, mk))
}

/// Processing time of the critical chain per job, scaled so that the largest is 5.
fn job_bias_from_solution(pre: &Pre, sol: &Solution) -> Option<Vec<f64>> {
    let ds = build_disj_from_solution(pre, sol).ok()?;
    let mut buf = EvalBuf::new(ds.n);
    let (_, mk_node) = eval_disj(&ds, &mut buf)?;
    let mut job_bias = vec![0.0f64; pre.num_jobs];
    let mut u = mk_node;
    while u != NONE {
        job_bias[ds.node_job[u]] += ds.node_pt[u] as f64;
        u = buf.best_pred[u];
    }
    let max_bias = job_bias.iter().cloned().fold(0.0f64, f64::max);
    if max_bias > 0.0 {
        let scale = 5.0 / max_bias;
        for b in &mut job_bias {
            *b *= scale;
        }
    }
    Some(job_bias)
}

// ------------------------------------------------------------------------------------------------
// Diverse pool of the best schedules
// ------------------------------------------------------------------------------------------------

/// The `ksig` jobs started first, in order: two schedules with the same signature are near-duplicates.
fn signature(s: &Solution, ksig: usize) -> Vec<usize> {
    let mut best: Vec<(u32, usize)> = Vec::with_capacity(ksig);
    for j in 0..s.job_schedule.len() {
        let t = s.job_schedule[j].first().map(|x| x.1).unwrap_or(u32::MAX);
        let mut pos = best.len();
        while pos > 0 {
            let (bt, bj) = best[pos - 1];
            if bt < t || (bt == t && bj < j) {
                break;
            }
            pos -= 1;
        }
        if pos >= ksig {
            continue;
        }
        best.insert(pos, (t, j));
        if best.len() > ksig {
            best.pop();
        }
    }
    best.into_iter().map(|(_, j)| j).collect()
}

fn similarity(a: &[usize], b: &[usize]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x == y).count()
}

/// Inserts `sol` into the pool of at most `cap` schedules: a near-duplicate replaces its twin
/// when better, otherwise the most crowded entry is evicted when the newcomer is less crowded
/// or about as crowded and better.
fn push_top_solutions(top: &mut Vec<(Solution, u32)>, sol: &Solution, mk: u32, cap: usize) {
    if cap == 0 {
        return;
    }
    let ksig = cap.min(sol.job_schedule.len().max(1));
    let sig_new = signature(sol, ksig);
    let sigs: Vec<Vec<usize>> = top.iter().map(|(s2, _)| signature(s2, ksig)).collect();
    let mut best_sim = 0usize;
    let mut best_idx = NONE;
    for (i, sig2) in sigs.iter().enumerate() {
        let sim = similarity(&sig_new, sig2);
        if sim > best_sim {
            best_sim = sim;
            best_idx = i;
        }
    }
    if best_idx != NONE && best_sim >= ksig {
        if mk < top[best_idx].1 {
            top[best_idx] = (sol.clone(), mk);
        }
        return;
    }
    if top.len() < cap {
        top.push((sol.clone(), mk));
        return;
    }
    let mut crowd_max: Vec<usize> = vec![0usize; top.len()];
    for i in 0..top.len() {
        for j in (i + 1)..top.len() {
            let sim = similarity(&sigs[i], &sigs[j]);
            crowd_max[i] = crowd_max[i].max(sim);
            crowd_max[j] = crowd_max[j].max(sim);
        }
    }
    let mut evict_idx = 0usize;
    let mut evict_crowd = crowd_max[0];
    for i in 1..top.len() {
        let crowd = crowd_max[i];
        if crowd > evict_crowd || (crowd == evict_crowd && top[i].1 > top[evict_idx].1) {
            evict_crowd = crowd;
            evict_idx = i;
        }
    }
    let new_crowd = sigs
        .iter()
        .map(|sig| similarity(&sig_new, sig))
        .max()
        .unwrap_or(0);
    if new_crowd < evict_crowd || (new_crowd <= evict_crowd + 1 && mk < top[evict_idx].1) {
        top[evict_idx] = (sol.clone(), mk);
    }
}

/// Up to `cap` schedules of `top` spread by signature: the best first, then repeatedly the one
/// least similar to those already picked (better makespan on ties).
fn pick_diverse(top: &[(Solution, u32)], cap: usize, ksig: usize) -> Vec<usize> {
    let sigs: Vec<Vec<usize>> = top.iter().map(|(s, _)| signature(s, ksig)).collect();
    let mut picked: Vec<usize> = Vec::with_capacity(cap);
    let mut first = 0usize;
    for i in 1..top.len() {
        if top[i].1 < top[first].1 {
            first = i;
        }
    }
    picked.push(first);
    while picked.len() < cap {
        let mut best_i = NONE;
        let mut best_max_sim = usize::MAX;
        let mut best_mk = u32::MAX;
        for i in 0..top.len() {
            if picked.iter().any(|&p| p == i) {
                continue;
            }
            let max_sim = picked
                .iter()
                .map(|&p| similarity(&sigs[i], &sigs[p]))
                .max()
                .unwrap_or(0);
            let mk_i = top[i].1;
            if max_sim < best_max_sim || (max_sim == best_max_sim && mk_i < best_mk) {
                best_max_sim = max_sim;
                best_mk = mk_i;
                best_i = i;
            }
        }
        if best_i == NONE {
            break;
        }
        picked.push(best_i);
    }
    picked
}

// ------------------------------------------------------------------------------------------------
// Entry point
// ------------------------------------------------------------------------------------------------

struct Incumbent<'a> {
    sol: Solution,
    mk: u32,
    save: &'a dyn Fn(&Solution) -> Result<()>,
}

impl Incumbent<'_> {
    /// Adopts `sol` when it is at least as good, and hands it to the host.
    fn offer_le(&mut self, sol: &Solution, mk: u32) -> Result<()> {
        if mk <= self.mk {
            self.mk = mk;
            self.sol = sol.clone();
            (self.save)(&self.sol)?;
        }
        Ok(())
    }

    /// Adopts `sol` only when strictly better.
    fn offer_lt(&mut self, sol: &Solution, mk: u32) -> Result<()> {
        if mk < self.mk {
            self.mk = mk;
            self.sol = sol.clone();
            (self.save)(&self.sol)?;
        }
        Ok(())
    }
}

/// Distinct job orders among `starts`, at most `cap`, all of length `n`.
fn distinct_orders(starts: Vec<Vec<usize>>, n: usize, cap: usize) -> Vec<Vec<usize>> {
    let mut uniq: Vec<Vec<usize>> = Vec::new();
    for ord in starts {
        if ord.len() == n && !uniq.iter().any(|u| *u == ord) {
            uniq.push(ord);
        }
        if uniq.len() >= cap {
            break;
        }
    }
    uniq
}

/// Up to `cap` starts of the critical-block search: the best pooled schedule, then `seeds`,
/// then the remaining pooled schedules by increasing makespan.
fn descent_starts(top: &[(Solution, u32)], seeds: Vec<(Solution, u32)>, cap: usize) -> Vec<(Solution, u32)> {
    let mut by_mk: Vec<usize> = (0..top.len()).collect();
    by_mk.sort_by_key(|&i| (top[i].1, i));
    let mut starts: Vec<(Solution, u32)> = Vec::with_capacity(cap);
    if let Some(&i) = by_mk.first() {
        starts.push(top[i].clone());
    }
    starts.extend(seeds);
    for &i in by_mk.iter().skip(1) {
        if starts.len() >= cap {
            break;
        }
        starts.push(top[i].clone());
    }
    starts.truncate(cap);
    starts
}

fn solve(challenge: &Challenge, save_solution: &dyn Fn(&Solution) -> Result<()>, pre: &Pre) -> Result<()> {
    let n = pre.num_jobs;
    let nm = pre.num_machines;
    let (greedy_sol, greedy_mk) = greedy_baseline(pre)?;
    save_solution(&greedy_sol)?;
    let mut inc = Incumbent {
        sol: greedy_sol,
        mk: greedy_mk,
        save: save_solution,
    };
    let mut top_solutions: Vec<(Solution, u32)> = Vec::new();
    push_top_solutions(&mut top_solutions, &inc.sol, inc.mk, 5);

    let common = pre.route.as_ref().zip(pre.pt_by_job.as_ref());
    let unique = common.map_or(false, |(route, _)| route_is_unique(route, nm));
    let mut strict_sol: Option<(Solution, u32)> = None;
    let mut descent_seeds: Vec<(Solution, u32)> = Vec::new();

    if let Some((route, pt)) = common {
        if let Ok((sol, mk, seeds)) = strict_best_by_order_search(pre, &challenge.seed, route, pt) {
            inc.offer_le(&sol, mk)?;
            if mk <= inc.mk {
                push_top_solutions(&mut top_solutions, &sol, mk, 5);
            }
            strict_sol = Some((sol, mk));
            descent_seeds = seeds;
        }

        if let Ok(neh_seq) = neh_best_sequence(route, pt, n, nm) {
            let perm_sol = build_perm_solution_from_seq(&neh_seq, route, pt, n, nm);
            if let Ok(mk) = challenge.evaluate_makespan(&perm_sol) {
                inc.offer_le(&perm_sol, mk)?;
                push_top_solutions(&mut top_solutions, &perm_sol, mk, 5);
            }
            let mut rank = vec![n; n];
            for (pos, &j) in neh_seq.iter().enumerate() {
                rank[j] = pos;
            }
            if let Ok((ssol, _)) = strict_simulate(pre, &rank) {
                if let Ok(mk) = challenge.evaluate_makespan(&ssol) {
                    inc.offer_le(&ssol, mk)?;
                    push_top_solutions(&mut top_solutions, &ssol, mk, 5);
                    descent_seeds.insert(0, (ssol.clone(), mk));
                }
            }
            if unique {
                let mut starts: Vec<Vec<usize>> = vec![neh_seq.clone()];
                if let Some((s, _)) = &strict_sol {
                    starts.push(order_from_solution_first_op_start(s, n));
                }
                starts.push(order_from_solution_first_op_start(&inc.sol, n));
                let uniq = distinct_orders(starts, n, 3);
                let mut seed = challenge.seed;
                seed[0] ^= 0x6B;
                let mut rng = SmallRng::from_seed(seed);
                let per = (2200 / uniq.len().max(1)).max(600);
                let m = route.len();
                let mut comp = vec![0u32; m];
                let mut best_ig_seq = neh_seq;
                let mut best_ig_mk = flow_makespan(&best_ig_seq, pt, &mut comp);
                for start_seq in uniq.iter() {
                    let cand_seq = iterated_greedy_search(start_seq, pt, per, 4, &mut rng);
                    let mk = flow_makespan(&cand_seq, pt, &mut comp);
                    if mk < best_ig_mk {
                        best_ig_mk = mk;
                        best_ig_seq = cand_seq;
                    }
                }
                let ig_perm_sol = build_perm_solution_from_seq(&best_ig_seq, route, pt, n, nm);
                if let Ok(mk) = challenge.evaluate_makespan(&ig_perm_sol) {
                    inc.offer_le(&ig_perm_sol, mk)?;
                    push_top_solutions(&mut top_solutions, &ig_perm_sol, mk, 5);
                }
            }
        }

        if !unique {
            grasp(pre, challenge, &mut inc, &mut top_solutions)?;
        }
    }

    if !unique {
        let perturb_cycles = (pre.total_ops / 160).clamp(1, 10);
        for (s0, _) in descent_starts(&top_solutions, descent_seeds, 3) {
            if let Ok(Some((sol2, mk2))) = critical_block_local_search(pre, &challenge.seed, &s0, perturb_cycles) {
                inc.offer_lt(&sol2, mk2)?;
                push_top_solutions(&mut top_solutions, &sol2, mk2, 15);
            }
        }
    } else if let Some((route, pt)) = common {
        perm_iterated_greedy(challenge, pre, route, pt, &mut inc, &top_solutions);
    }
    Ok(())
}

/// Eighty randomised constructions, the rule drawn by its average gain, each polished by one
/// descent iteration; the critical chain of near-best schedules biases the next constructions.
fn grasp(
    pre: &Pre,
    challenge: &Challenge,
    inc: &mut Incumbent<'_>,
    top_solutions: &mut Vec<(Solution, u32)>,
) -> Result<()> {
    let mut seed = challenge.seed;
    seed[0] ^= 0xF1;
    let mut rng = SmallRng::from_seed(seed);
    let mut adaptive_boost = AdaptiveBoost::new();
    let nr = GRASP_RULES.len();
    let mut attempts: Vec<u32> = vec![0u32; nr];
    let mut improves: Vec<u32> = vec![0u32; nr];
    let mut delta_sum: Vec<u64> = vec![0u64; nr];
    let mut total_attempts: u32 = 0;
    let mut job_bias: Option<Vec<f64>> = None;
    let num_restarts = 80usize;
    for r in 0..num_restarts {
        let do_test = (r % 45 == 0) && r > 0;
        let untried: Vec<usize> = (0..nr).filter(|&i| attempts[i] == 0).collect();
        let ridx = if !untried.is_empty() {
            untried[rng.gen_range(0..untried.len())]
        } else {
            let mut best_i = 0usize;
            let mut best_score = 0u64;
            let mut best_succ = 0u32;
            for i in 0..nr {
                let a = attempts[i].max(1) as u64;
                let score = (delta_sum[i] / a).saturating_add((total_attempts as u64) / a);
                if score > best_score || (score == best_score && improves[i] > best_succ) {
                    best_score = score;
                    best_succ = improves[i];
                    best_i = i;
                } else if score == best_score && improves[i] == best_succ && (rng.gen::<u32>() & 1) == 0 {
                    best_i = i;
                }
            }
            best_i
        };
        let rule = GRASP_RULES[ridx];
        let k = if r < nr { 0 } else { rng.gen_range(2..=5) };
        attempts[ridx] = attempts[ridx].saturating_add(1);
        total_attempts = total_attempts.saturating_add(1);

        let prev_best = inc.mk;
        let target = inc.mk.saturating_add(inc.mk / 20);
        let Ok((mut sol, mut mk)) = construct_solution_conflict(
            pre,
            rule,
            k,
            target,
            &mut rng,
            adaptive_boost.boost_strength,
            job_bias.as_deref(),
        ) else {
            continue;
        };
        if mk <= target {
            if let Ok(mut ds) = build_disj_from_solution(pre, &sol) {
                let mut buf = EvalBuf::new(ds.n);
                if let Some(initial) = eval_disj(&ds, &mut buf) {
                    let mut crit = vec![false; ds.n];
                    let mut cur_eval = initial;
                    if descent_phase(&mut ds, &mut buf, &mut crit, &mut cur_eval, 1, 20) {
                        if let Some((new_mk, _)) = eval_disj(&ds, &mut buf) {
                            if new_mk < mk {
                                sol = disj_to_solution(pre, &ds, &buf.start);
                                mk = new_mk;
                            }
                        }
                    }
                }
            }
        }
        if mk < prev_best {
            improves[ridx] = improves[ridx].saturating_add(1);
            delta_sum[ridx] = delta_sum[ridx].saturating_add((prev_best - mk) as u64);
        }
        inc.offer_lt(&sol, mk)?;
        push_top_solutions(top_solutions, &sol, mk, 15);
        if mk <= inc.mk.saturating_mul(11) / 10 {
            if let Some(bias) = job_bias_from_solution(pre, &sol) {
                job_bias = Some(bias);
            }
        }
        if do_test {
            let target = inc.mk.saturating_add(inc.mk / 20);
            if let Ok((_, mk_no_boost)) =
                construct_solution_conflict(pre, rule, k, target, &mut rng, 0.0, job_bias.as_deref())
            {
                adaptive_boost.update_from_test(mk, mk_no_boost);
            }
        }
    }
    Ok(())
}

/// Permutation route: iterated greedy from up to five diverse pool schedules, with a random
/// destruction of the best order once it stagnates.
fn perm_iterated_greedy(
    challenge: &Challenge,
    pre: &Pre,
    route: &[usize],
    pt: &[Vec<u32>],
    inc: &mut Incumbent<'_>,
    top_solutions: &[(Solution, u32)],
) {
    if pt.is_empty() || top_solutions.is_empty() {
        return;
    }
    let n = pre.num_jobs;
    let nm = pre.num_machines;
    let mut seed = challenge.seed;
    seed[0] ^= 0xD4;
    let mut rng = SmallRng::from_seed(seed);
    let m = route.len();
    let mut comp = vec![0u32; m];
    let initial_best_mk = inc.mk;
    let seed_cap = top_solutions.len().min(5);
    let picked = pick_diverse(top_solutions, seed_cap, seed_cap.min(n.max(1)));

    let mut best_ig_seq = order_from_solution_first_op_start(&inc.sol, n);
    let mut best_ig_mk = inc.mk;
    let mut stagnation = 0usize;
    let mut perturbation_attempts = 0usize;
    let perturb_max = 3usize;
    for &i in &picked {
        let perturb_mode = stagnation >= 2 && perturbation_attempts < perturb_max && best_ig_seq.len() == n;
        let start_ord = if perturb_mode {
            let ratio = (initial_best_mk - best_ig_mk) as f64 / initial_best_mk.max(1) as f64;
            let d_perturb = ((n as f64 * (0.08 - 0.05 * ratio)).max(2.0).min(6.0)) as usize;
            let mut seq = best_ig_seq.clone();
            let mut indices: Vec<usize> = (0..seq.len()).collect();
            indices.shuffle(&mut rng);
            let removed: Vec<usize> = indices.iter().take(d_perturb).map(|&idx| seq[idx]).collect();
            let mut to_remove: Vec<usize> = indices.iter().take(d_perturb).cloned().collect();
            to_remove.sort_unstable_by(|a, b| b.cmp(a));
            for idx in to_remove {
                seq.remove(idx);
            }
            for job in removed {
                let pos = rng.gen_range(0..=seq.len());
                seq.insert(pos, job);
            }
            seq
        } else {
            order_from_solution_first_op_start(&top_solutions[i].0, n)
        };
        if start_ord.len() != n {
            continue;
        }
        let cand_seq = iterated_greedy_search(&start_ord, pt, 320, 4, &mut rng);
        let mk = flow_makespan(&cand_seq, pt, &mut comp);
        if mk < best_ig_mk {
            best_ig_mk = mk;
            if mk < inc.mk {
                inc.mk = mk;
                inc.sol = build_perm_solution_from_seq(&cand_seq, route, pt, n, nm);
                let _ = (inc.save)(&inc.sol);
            }
            best_ig_seq = cand_seq;
            stagnation = 0;
        } else {
            stagnation += 1;
            if perturb_mode {
                perturbation_attempts += 1;
                if perturbation_attempts >= perturb_max {
                    stagnation = 0;
                }
            }
        }
    }
}

pub fn solve_challenge(challenge: &Challenge, save_solution: &dyn Fn(&Solution) -> Result<()>) -> Result<()> {
    let pre = build_pre(challenge)?;
    solve(challenge, save_solution, &pre)
}
