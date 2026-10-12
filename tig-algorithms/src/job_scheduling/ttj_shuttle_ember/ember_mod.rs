// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
//
// ember_stromboli: c007 job_scheduling.
use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde_json::{Map, Value};
use tig_challenges::job_scheduling::*;

#[path = "ember_flow_shop_engine.rs"]
mod flow_shop_engine;
#[path = "ember_job_shop_engine.rs"]
mod job_shop_engine;
#[path = "ember_gate_floor.rs"]
mod gate_floor;
#[path = "ember_sota.rs"]
mod sota;
#[path = "ember_hybrid_engine.rs"]
mod hybrid_engine;

const INF: u32 = u32::MAX / 4;
const FUEL_STOP: u64 = 100_000_000;
const JOBSHOPNESS_MEDIUM_OVERRIDE: Option<f64> = Some(0.55);

extern "C" {
    static __fuel_remaining: u64;
}
#[inline]
fn fuel_left() -> u64 {
    unsafe { core::ptr::read_volatile(&__fuel_remaining) }
}

struct Pre {
    num_jobs: usize,
    num_machines: usize,
    job_ops: Vec<Vec<Vec<(usize, u32)>>>,
    job_nops: Vec<usize>,
    n_ops: usize,
    job_off: Vec<usize>,
    op_job: Vec<usize>,
    op_idx: Vec<usize>,
    n_products: usize,
    max_flex: usize,
    op_w: Vec<f64>,
    // True when every eligible processing time is >= 1. The instance generator clamps with
    // `1.max(..)` so it always holds, but `critical_path` uses it to justify scanning only the
    // first operation of each job: with strictly positive durations, `heads[o] == 0` implies `o`
    // has no predecessor, hence `op_idx[o] == 0`. When it does not hold we keep the full sweep.
    dur_pos: bool,
}

impl Pre {
    fn build(challenge: &Challenge) -> Self {
        let num_jobs = challenge.num_jobs;
        let mut job_ops: Vec<Vec<Vec<(usize, u32)>>> = Vec::with_capacity(num_jobs);
        for (product, &count) in challenge.jobs_per_product.iter().enumerate() {
            let ops = &challenge.product_processing_times[product];
            for _ in 0..count {
                let mut this_job = Vec::with_capacity(ops.len());
                for op in ops.iter() {
                    let mut cands: Vec<(usize, u32)> = op.iter().map(|(&m, &t)| (m, t)).collect();
                    cands.sort_unstable_by_key(|&(m, _)| m);
                    this_job.push(cands);
                }
                job_ops.push(this_job);
            }
        }
        let job_nops: Vec<usize> = job_ops.iter().map(|j| j.len()).collect();
        let mut job_off = Vec::with_capacity(num_jobs);
        let mut op_job = Vec::new();
        let mut op_idx = Vec::new();
        let mut off = 0;
        for (j, &k) in job_nops.iter().enumerate() {
            job_off.push(off);
            for o in 0..k {
                op_job.push(j);
                op_idx.push(o);
            }
            off += k;
        }
        let max_flex = job_ops.iter().flat_map(|j| j.iter()).map(|o| o.len()).max().unwrap_or(1);
        let dur_pos = job_ops.iter().flat_map(|j| j.iter()).flat_map(|o| o.iter()).all(|&(_, t)| t >= 1);
        let mut op_w = Vec::with_capacity(off);
        for ops in job_ops.iter() {
            for op in ops.iter() {
                let s: u32 = op.iter().map(|&(_, t)| t).sum();
                let avg = s as f64 / op.len() as f64;
                let mn = op.iter().map(|&(_, t)| t).min().unwrap_or(0) as f64;
                op_w.push(avg * 0.7 + mn * 0.3);
            }
        }
        Pre {
            num_jobs, num_machines: challenge.num_machines, job_ops, job_nops,
            n_ops: off, job_off, op_job, op_idx, n_products: challenge.jobs_per_product.len(), max_flex,
            op_w, dur_pos,
        }
    }
    #[inline]
    fn op_id(&self, job: usize, oi: usize) -> usize { self.job_off[job] + oi }
}

#[inline]
fn earliest_end(time: u32, mach_avail: &[u32], op_times: &[(usize, u32)]) -> u32 {
    let mut e = u32::MAX;
    for &(m, pt) in op_times {
        let end = time.max(mach_avail[m]) + pt;
        if end < e { e = end; }
    }
    e
}

fn dispatch_construct(pre: &Pre, rule: u8, rng: Option<&mut SmallRng>, top_k: usize)
    -> (Vec<usize>, Vec<Vec<usize>>) {
    use rand::seq::SliceRandom;
    let n = pre.num_jobs;
    let nm = pre.num_machines;
    let mut job_next = vec![0usize; n];
    let mut job_ready = vec![0u32; n];
    let mut mach_avail = vec![0u32; nm];
    let mut op_machine = vec![0usize; pre.n_ops];
    let mut machine_seq: Vec<Vec<usize>> = vec![Vec::new(); nm];
    let mut rem_work = vec![0f64; n];
    for j in 0..n {
        for oi in 0..pre.job_nops[j] {
            rem_work[j] += pre.op_w[pre.op_id(j, oi)];
        }
    }
    let mut remaining: usize = pre.job_nops.iter().sum();
    let mut time = 0u32;
    let use_random = top_k > 1 && rng.is_some();
    let mut rng = rng;

    while remaining > 0 {
        let mut avail: Vec<usize> = (0..nm).filter(|&m| mach_avail[m] <= time).collect();
        if use_random { avail.shuffle(rng.as_mut().unwrap()); }
        for &machine in &avail {
            let mut cands: Vec<(f64, usize, u32)> = Vec::new();
            for j in 0..n {
                let oi = job_next[j];
                if oi >= pre.job_nops[j] { continue; }
                if job_ready[j] > time { continue; }
                let op = &pre.job_ops[j][oi];
                let Some(&(_, pt)) = op.iter().find(|&&(m, _)| m == machine) else { continue };
                let machine_end = time.max(mach_avail[machine]) + pt;
                if machine_end != earliest_end(time, &mach_avail, op) { continue; }
                let flex = op.len();
                let rem_ops = pre.job_nops[j] - oi;
                let pr = match rule {
                    0 => rem_work[j],
                    1 => rem_ops as f64,
                    2 => -(flex as f64),
                    3 => -(pt as f64),
                    _ => pt as f64,
                };
                cands.push((pr, j, pt));
            }
            if cands.is_empty() { continue; }
            let spt = pre.max_flex == 1;
            let (_, j, pt) = if use_random {
                if spt {
                    cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()
                        .then(a.2.cmp(&b.2)).then(a.1.cmp(&b.1)));
                } else {
                    cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
                }
                let k = top_k.min(cands.len());
                cands[rng.as_mut().unwrap().gen_range(0..k)]
            } else if spt {
                *cands.iter().max_by(|a, b| a.0.partial_cmp(&b.0).unwrap()
                    .then(b.2.cmp(&a.2)).then(b.1.cmp(&a.1))).unwrap()
            } else {
                *cands.iter().max_by(|a, b| a.0.partial_cmp(&b.0).unwrap()
                    .then(b.1.cmp(&a.1))).unwrap()
            };
            let oi = job_next[j];
            let start = time.max(mach_avail[machine]);
            let end = start + pt;
            let oid = pre.op_id(j, oi);
            op_machine[oid] = machine;
            machine_seq[machine].push(oid);
            job_next[j] += 1;
            job_ready[j] = end;
            mach_avail[machine] = end;
            rem_work[j] -= pre.op_w[oid];
            if rem_work[j] < 0.0 { rem_work[j] = 0.0; }
            remaining -= 1;
        }
        if remaining == 0 { break; }
        let mut next: Option<u32> = None;
        for &t in mach_avail.iter() { if t > time { next = Some(next.map_or(t, |b| b.min(t))); } }
        for j in 0..n {
            if job_next[j] < pre.job_nops[j] && job_ready[j] > time {
                let t = job_ready[j]; next = Some(next.map_or(t, |b| b.min(t)));
            }
        }
        match next { Some(t) => time = t, None => break }
    }
    (op_machine, machine_seq)
}

fn kacem_construct(pre: &Pre, rng: Option<&mut SmallRng>) -> (Vec<usize>, Vec<Vec<usize>>) {
    let n = pre.num_jobs;
    let nm = pre.num_machines;
    let mut charge = vec![0u64; nm];
    let mut op_machine = vec![0usize; pre.n_ops];
    let mut order: Vec<(u32, usize)> = (0..pre.n_ops).map(|o| {
        let j = pre.op_job[o]; let oi = pre.op_idx[o];
        let mn = pre.job_ops[j][oi].iter().map(|&(_, t)| t).min().unwrap_or(0);
        (mn, o)
    }).collect();
    order.sort_unstable();
    let jitter = rng.is_some();
    let mut rng = rng;
    for &(_, o) in &order {
        let j = pre.op_job[o]; let oi = pre.op_idx[o];
        let mut best_k = pre.job_ops[j][oi][0].0;
        let mut best_p = pre.job_ops[j][oi][0].1;
        let mut best_c = u64::MAX;
        for &(k, p) in &pre.job_ops[j][oi] {
            let mut c = charge[k] + p as u64;
            if jitter { c += (rng.as_mut().unwrap().gen::<u8>() as u64) & 3; }
            if c < best_c { best_c = c; best_k = k; best_p = p; }
        }
        op_machine[o] = best_k;
        charge[best_k] += best_p as u64;
    }
    let mut job_next = vec![0usize; n];
    let mut job_ready = vec![0u32; n];
    let mut mach_avail = vec![0u32; nm];
    let mut machine_seq: Vec<Vec<usize>> = vec![Vec::new(); nm];
    let mut remaining: usize = pre.job_nops.iter().sum();
    let mut op_dur = vec![0u32; pre.n_ops];
    for o in 0..pre.n_ops {
        let j = pre.op_job[o]; let oi = pre.op_idx[o];
        let m = op_machine[o];
        op_dur[o] = pre.job_ops[j][oi].iter().find(|&&(mm, _)| mm == m).map(|&(_, t)| t).unwrap();
    }
    while remaining > 0 {
        let mut best_end = u32::MAX;
        let mut best_j = usize::MAX;
        for j in 0..n {
            let oi = job_next[j];
            if oi >= pre.job_nops[j] { continue; }
            let o = pre.op_id(j, oi);
            let m = op_machine[o];
            let end = job_ready[j].max(mach_avail[m]) + op_dur[o];
            if end < best_end || (end == best_end && j < best_j) { best_end = end; best_j = j; }
        }
        let j = best_j;
        let oi = job_next[j];
        let o = pre.op_id(j, oi);
        let m = op_machine[o];
        let start = job_ready[j].max(mach_avail[m]);
        let e = start + op_dur[o];
        machine_seq[m].push(o);
        job_next[j] += 1;
        job_ready[j] = e;
        mach_avail[m] = e;
        remaining -= 1;
    }
    (op_machine, machine_seq)
}

struct Graph<'a> {
    pre: &'a Pre,
    op_machine: Vec<usize>,
    op_dur: Vec<u32>,
    machine_seq: Vec<Vec<usize>>,
    heads: Vec<u32>,
    // `hend[o] == heads[o] + op_dur[o]`, written by the forward pass of `eval` which already
    // computes that sum. Every consumer of the completion time reformed it from two loads and an
    // add; the binary searches of `best_machine_move_mg` reformed it once per probe.
    hend: Vec<u32>,
    // `qend[o] == tails[o] + op_dur[o]`. No consumer ever wanted the bare tail: the criticality
    // test, `swap_estimate`, `best_machine_move_mg` and `best_insertion` all reformed the
    // sum on every read. Stored pre-added it costs the backward pass one add per operation and
    // saves two loads and an add at each of the many reads.
    qend: Vec<u32>,
    // Fixed-size buffer plus its length instead of a `Vec` grown by `push`. Each operation is
    // pushed at most once (only when its in-degree reaches zero), so `n_ops` slots always
    // suffice; the LIFO/FIFO order is untouched.
    topo: Vec<usize>,
    mp: Vec<usize>,
    ms: Vec<usize>,
    // Position of each operation inside its own machine sequence, kept up to date by
    // `sync_arcs` / `repair_machine` (both already walk the sequence). `n5_moves` used to
    // rebuild the whole table from scratch at every polish iteration, and three call sites did a
    // linear `position()` search over a sequence.
    mpos: Vec<usize>,
    indeg: Vec<u8>,
    indeg0: Vec<u8>,
    // `pre.dur_pos` restricted to the durations actually loaded in `op_dur`; see `Pre::dur_pos`.
    dur_pos: bool,
}

impl<'a> Graph<'a> {
    fn new(pre: &'a Pre, op_machine: Vec<usize>, machine_seq: Vec<Vec<usize>>) -> Self {
        let n = pre.n_ops;
        let mut op_dur = vec![0u32; n];
        for oid in 0..n {
            let j = pre.op_job[oid];
            let o = pre.op_idx[oid];
            let m = op_machine[oid];
            op_dur[oid] = pre.job_ops[j][o].iter().find(|&&(mm, _)| mm == m).map(|&(_, t)| t).unwrap_or(0);
        }
        let dur_pos = pre.dur_pos && op_dur.iter().all(|&d| d >= 1);
        let mut g = Graph {
            pre, op_machine, op_dur, machine_seq,
            heads: vec![0; n], hend: vec![0; n], qend: vec![0; n],
            topo: vec![0; n],
            mp: vec![usize::MAX; n], ms: vec![usize::MAX; n], mpos: vec![0; n],
            indeg: vec![0; n], indeg0: vec![0; n], dur_pos,
        };
        g.sync_arcs();
        g
    }

    fn reset(&mut self, op_machine: &[usize], machine_seq: &[Vec<usize>]) {
        self.op_machine.copy_from_slice(op_machine);
        let mut dur_pos = self.pre.dur_pos;
        for oid in 0..self.op_machine.len() {
            let j = self.pre.op_job[oid];
            let o = self.pre.op_idx[oid];
            let m = self.op_machine[oid];
            let d = self.pre.job_ops[j][o].iter().find(|&&(mm, _)| mm == m)
                .map(|&(_, t)| t).unwrap_or(0);
            if d == 0 { dur_pos = false; }
            self.op_dur[oid] = d;
        }
        self.dur_pos = dur_pos;
        for (dst, src) in self.machine_seq.iter_mut().zip(machine_seq.iter()) {
            dst.clear();
            dst.extend_from_slice(src);
        }
        self.sync_arcs();
    }

    fn sync_arcs(&mut self) {
        let n = self.pre.n_ops;
        // `fill` compiles to a memset, which the fuel meter caps at 22 units whatever the length;
        // the hand-written loop it replaces was charged per element.
        self.mp.fill(usize::MAX);
        self.ms.fill(usize::MAX);
        let nm = self.machine_seq.len();
        for m in 0..nm {
            let len = self.machine_seq[m].len();
            for i in 0..len {
                let a = self.machine_seq[m][i];
                self.mpos[a] = i;
                if i + 1 < len {
                    let b = self.machine_seq[m][i + 1];
                    self.ms[a] = b;
                    self.mp[b] = a;
                }
            }
        }
        for oid in 0..n {
            let mut d = 0u8;
            if self.pre.op_idx[oid] > 0 { d += 1; }
            if self.mp[oid] != usize::MAX { d += 1; }
            self.indeg0[oid] = d;
        }
    }

    /// One pass instead of three. The old first loop reset `mp`/`ms` to `usize::MAX` and the
    /// second overwrote them from the neighbours; writing the neighbour (or `usize::MAX` at the
    /// ends) directly leaves exactly the same state, and `indeg0` is derived from the `mp` value
    /// computed on the spot rather than re-read on a third pass.
    fn repair_machine(&mut self, m: usize) {
        let len = self.machine_seq[m].len();
        for i in 0..len {
            let o = self.machine_seq[m][i];
            let p = if i > 0 { self.machine_seq[m][i - 1] } else { usize::MAX };
            let sc = if i + 1 < len { self.machine_seq[m][i + 1] } else { usize::MAX };
            self.mpos[o] = i;
            self.mp[o] = p;
            self.ms[o] = sc;
            let mut d = 0u8;
            if self.pre.op_idx[o] > 0 { d += 1; }
            if p != usize::MAX { d += 1; }
            self.indeg0[o] = d;
        }
    }

    #[inline]
    fn job_succ(&self, oid: usize) -> Option<usize> {
        let j = self.pre.op_job[oid];
        if self.pre.op_idx[oid] + 1 < self.pre.job_nops[j] { Some(oid + 1) } else { None }
    }

    fn eval(&mut self) -> Option<u32> {
        let n = self.pre.n_ops;
        self.indeg.copy_from_slice(&self.indeg0);
        // memset, capped at 22 fuel, instead of one charged store per operation.
        self.heads.fill(0);
        // `indeg0[o] == 0` requires `op_idx[o] == 0` (a job successor always contributes 1), so
        // the roots are a subset of the first operation of each job. `pre.job_off` lists exactly
        // those, in increasing operation id, so the seeded set AND its order are the same as the
        // old sweep over all `n_ops` -- for 50 jobs of 30 operations that is 50 tests instead of
        // 1500.
        let mut tl = 0usize;
        for &oid in self.pre.job_off.iter() {
            if self.indeg[oid] == 0 { self.topo[tl] = oid; tl += 1; }
        }
        let mut qi = 0;
        let mut makespan = 0u32;
        while qi < tl {
            let u = self.topo[qi]; qi += 1;
            let end_u = self.heads[u] + self.op_dur[u];
            self.hend[u] = end_u;
            if end_u > makespan { makespan = end_u; }
            let js = { let j = self.pre.op_job[u];
                if self.pre.op_idx[u] + 1 < self.pre.job_nops[j] { u + 1 } else { usize::MAX } };
            if js != usize::MAX {
                if end_u > self.heads[js] { self.heads[js] = end_u; }
                self.indeg[js] -= 1;
                if self.indeg[js] == 0 { self.topo[tl] = js; tl += 1; }
            }
            let mu = self.ms[u];
            if mu != usize::MAX {
                if end_u > self.heads[mu] { self.heads[mu] = end_u; }
                self.indeg[mu] -= 1;
                if self.indeg[mu] == 0 { self.topo[tl] = mu; tl += 1; }
            }
        }
        if tl != n { return None; }
        for idx in (0..tl).rev() {
            let u = self.topo[idx];
            let mut t = 0u32;
            let js = { let j = self.pre.op_job[u];
                if self.pre.op_idx[u] + 1 < self.pre.job_nops[j] { u + 1 } else { usize::MAX } };
            if js != usize::MAX { t = t.max(self.qend[js]); }
            let mu = self.ms[u];
            if mu != usize::MAX { t = t.max(self.qend[mu]); }
            self.qend[u] = t + self.op_dur[u];
        }
        Some(makespan)
    }

    fn to_solution(&self) -> Solution {
        let mut job_schedule: Vec<Vec<(usize, u32)>> =
            self.pre.job_nops.iter().map(|&k| Vec::with_capacity(k)).collect();
        for oid in 0..self.pre.n_ops {
            let j = self.pre.op_job[oid];
            job_schedule[j].push((self.op_machine[oid], self.heads[oid]));
        }
        Solution { job_schedule }
    }
}

fn swap_estimate(g: &Graph, m: usize, p: usize) -> u32 {
    let seq = &g.machine_seq[m];
    let a = seq[p];
    let b = seq[p + 1];
    let comp = |o: usize| g.hend[o];
    let tailc = |o: usize| g.qend[o];
    let jp_comp = |o: usize| -> u32 { if g.pre.op_idx[o] > 0 { comp(o - 1) } else { 0 } };
    let js_tail = |o: usize| -> u32 {
        let j = g.pre.op_job[o];
        if g.pre.op_idx[o] + 1 < g.pre.job_nops[j] { tailc(o + 1) } else { 0 }
    };
    let pm_comp = if p > 0 { comp(seq[p - 1]) } else { 0 };
    let sm_tail = if p + 2 < seq.len() { tailc(seq[p + 2]) } else { 0 };
    let h_b = jp_comp(b).max(pm_comp);
    let h_a = jp_comp(a).max(h_b + g.op_dur[b]);
    let t_a = js_tail(a).max(sm_tail);
    let t_b = js_tail(b).max(g.op_dur[a] + t_a);
    (h_a + g.op_dur[a] + t_a).max(h_b + g.op_dur[b] + t_b)
}

fn insert_pos(g: &Graph, mt: usize, o: usize) -> usize {
    let ho = g.heads[o];
    let seq = &g.machine_seq[mt];
    for (i, &x) in seq.iter().enumerate() {
        if g.heads[x] > ho { return i; }
    }
    seq.len()
}

fn best_machine_move_mg(
    g: &Graph, path: &[usize], best_real: u32,
    tabu: &std::collections::BTreeMap<(usize, usize), usize>, it: usize,
    tabu_hi: &[usize], sweep_cap: usize,
    w: &mut [u32], stamp: &mut [u32], gen: &mut u32,
) -> Option<(usize, usize, usize, u32)> {
    *gen = gen.wrapping_add(1);
    let cur = *gen;
    let mut best: Option<(usize, usize, usize, u32)> = None;
    let mut seen = 0usize;
    for &v in path {
        let j = g.pre.op_job[v];
        let oi = g.pre.op_idx[v];
        let elig = &g.pre.job_ops[j][oi];
        if elig.len() < 2 { continue; }
        seen += 1;
        if seen > sweep_cap { break; }
        let m_cur = g.op_machine[v];
        let rpj = if oi > 0 { g.hend[v - 1] } else { 0 };
        let qsj = if oi + 1 < g.pre.job_nops[j] { g.qend[v + 1] } else { 0 };
        let is_tabu = if tabu_hi[v] <= it { false }
                      else { tabu.get(&(v, usize::MAX)).map_or(false, |&e| e > it) };
        debug_assert_eq!(is_tabu, tabu.get(&(v, usize::MAX)).map_or(false, |&e| e > it));
        for &(k, pk) in elig {
            if k == m_cur { continue; }
            let seq = &g.machine_seq[k];
            let len = seq.len();
            let left = seq.partition_point(|&o| g.hend[o] <= rpj);
            let right = seq.partition_point(|&o| g.qend[o] > qsj);
            if left > right { continue; }
            for pos in left..=right {
                let l = if pos > left && pos < right {
                    let o = seq[pos];
                    if stamp[o] != cur {
                        let p = g.mp[o];
                        w[o] = g.hend[p] + g.qend[o];
                        stamp[o] = cur;
                    }
                    pk + w[o]
                } else {
                    let hv = if pos > 0 {
                        g.hend[seq[pos - 1]].max(rpj)
                    } else { rpj };
                    let tv = if pos < len {
                        g.qend[seq[pos]].max(qsj)
                    } else { qsj };
                    hv + pk + tv
                };
                let allowed = !is_tabu || l < best_real;
                if allowed && best.map_or(true, |(_, _, _, bl)| l < bl) {
                    best = Some((v, k, pos, l));
                }
            }
        }
    }
    best
}

/// Fills `path` with the critical path. The buffer is owned by the caller and reused across
/// iterations instead of being allocated and grown on every call.
fn critical_path(g: &Graph, makespan: u32, path: &mut Vec<usize>) {
    let n = g.pre.n_ops;
    let crit = |o: usize| g.heads[o] + g.qend[o] == makespan;
    let mut start = usize::MAX;
    if g.dur_pos {
        // With strictly positive durations any operation that has a predecessor starts strictly
        // after 0, so `heads[o] == 0` implies `op_idx[o] == 0`. `pre.job_off` lists those
        // operations in increasing id, so the first match is the same one the full sweep found.
        for &o in g.pre.job_off.iter() {
            if crit(o) && g.heads[o] == 0 { start = o; break; }
        }
    } else {
        for o in 0..n {
            if crit(o) && g.heads[o] == 0 { start = o; break; }
        }
    }
    if start == usize::MAX {
        let mut best = u32::MAX;
        for o in 0..n { if crit(o) && g.heads[o] < best { best = g.heads[o]; start = o; } }
    }
    path.clear();
    let mut cur = start;
    loop {
        path.push(cur);
        let end = g.hend[cur];
        let mut next = usize::MAX;
        if g.ms[cur] != usize::MAX {
            let v = g.ms[cur];
            if crit(v) && g.heads[v] == end { next = v; }
        }
        if next == usize::MAX {
            if let Some(v) = g.job_succ(cur) {
                if crit(v) && g.heads[v] == end { next = v; }
            }
        }
        if next == usize::MAX { break; }
        cur = next;
    }
}
fn n5_moves_full(g: &Graph, makespan: u32, moves: &mut Vec<(usize, usize)>) {
    let crit = |o: usize| g.heads[o] + g.qend[o] == makespan;
    moves.clear();
    for m in 0..g.pre.num_machines {
        let seq = &g.machine_seq[m];
        let mut pos = 0usize;
        while pos < seq.len() {
            if !crit(seq[pos]) { pos += 1; continue; }
            let run_start = pos;
            let mut run_end = pos;
            while run_end + 1 < seq.len() {
                let node = seq[run_end + 1];
                let prev = seq[run_end];
                if crit(node) && g.heads[node] == g.hend[prev] {
                    run_end += 1;
                } else {
                    break;
                }
            }
            let block_len = run_end - run_start + 1;
            if block_len >= 2 {
                moves.push((m, run_start));
                if block_len >= 3 {
                    moves.push((m, run_end - 1));
                }
            }
            pos = run_end + 1;
        }
    }
}

/// Two allocations and a full rebuild removed from the polish loop:
///  - the machine position of an operation is read from `g.mpos`, maintained by
///    `repair_machine`, instead of rebuilding a `vec![usize::MAX; n_ops]` table by walking every
///    machine sequence on every call;
///  - the blocks of the path are described by half-open index ranges into `path` instead of a
///    `Vec<Vec<usize>>` that heap-allocated one vector per block.
/// The blocks, their order and the emitted moves are unchanged.
fn n5_moves(g: &Graph, path: &[usize], moves: &mut Vec<(usize, usize)>, blk: &mut Vec<(u32, u32)>) {
    blk.clear();
    moves.clear();
    let mut s0 = 0usize;
    for i in 1..path.len() {
        let o = path[i];
        let prev = path[i - 1];
        if !(g.op_machine[o] == g.op_machine[prev] && g.mpos[o] == g.mpos[prev] + 1) {
            blk.push((s0 as u32, i as u32));
            s0 = i;
        }
    }
    if !path.is_empty() { blk.push((s0 as u32, path.len() as u32)); }

    let nb = blk.len();
    for bi in 0..nb {
        let bs = blk[bi].0 as usize;
        let be = blk[bi].1 as usize;
        if be - bs < 2 { continue; }
        let m = g.op_machine[path[bs]];
        if bi != 0 {
            moves.push((m, g.mpos[path[bs]]));
        }
        if bi != nb - 1 {
            let p = g.mpos[path[be - 2]];
            if !(bi != 0 && p == g.mpos[path[bs]]) { moves.push((m, p)); }
        }
    }
}

#[inline]
fn sweep_cap(pre: &Pre) -> usize {
    debug_assert!(pre.max_flex > 1);
    if pre.n_products >= 27 { 60 } else { 80 }
}

#[inline]
fn patience(pre: &Pre) -> usize {
    debug_assert!(pre.max_flex > 1);
    if pre.n_products >= 27 { 30_000 }
    else { 3_000 }
}

fn tabu_run(
    g: &mut Graph, max_iter: usize, tenure: usize,
    full: bool, max_kicks: usize, fuel_floor: u64, rng: &mut SmallRng,
) -> Option<u32> {
    let mut cur_real = g.eval()?;
    let mut best_om: Vec<usize> = g.op_machine.clone();
    let mut best_ms: Vec<Vec<usize>> = g.machine_seq.clone();
    let mut best_real = cur_real;
    let mut tabu: std::collections::BTreeMap<(usize, usize), usize> = std::collections::BTreeMap::new();
    let mut tabu_hi: Vec<usize> = vec![0; g.pre.n_ops];
    let mut no_improve = 0usize;
    let max_no_improve = if max_kicks > 0 { patience(g.pre) } else { 8 * tenure + 400 };
    let sweep_cap = sweep_cap(g.pre);
    let mut kicks_left = max_kicks;
    let n_ops = g.pre.n_ops;
    let mut mg_w: Vec<u32> = vec![0u32; n_ops];
    let mut mg_stamp: Vec<u32> = vec![0u32; n_ops];
    let mut mg_gen: u32 = 0;
    // Reused across iterations: the three buffers used to be allocated (and grown by `push`)
    // afresh on every single iteration.
    let mut path: Vec<usize> = Vec::with_capacity(n_ops);
    let mut moves: Vec<(usize, usize)> = Vec::with_capacity(n_ops);
    let mut blk: Vec<(u32, u32)> = Vec::with_capacity(n_ops);

    for it in 0..max_iter {
        if fuel_left() <= fuel_floor { break; }
        critical_path(g, cur_real, &mut path);
        if path.len() < 2 { break; }
        if full { n5_moves_full(g, cur_real, &mut moves) } else { n5_moves(g, &path, &mut moves, &mut blk) };
        if moves.is_empty() { break; }

        let mut best_swap: Option<(usize, usize)> = None;
        let mut best_est = INF;
        let mut best_pair = (0usize, 0usize);
        let mut fb_swap: Option<(usize, usize)> = None;
        let mut fb_est = INF;
        let mut fb_pair = (0usize, 0usize);
        for &(m, p) in &moves {
            let a = g.machine_seq[m][p];
            let b = g.machine_seq[m][p + 1];
            let pair = if a < b { (a, b) } else { (b, a) };
            let est = swap_estimate(g, m, p);
            let is_tabu = if tabu_hi[a] <= it || tabu_hi[b] <= it { false }
                          else { tabu.get(&pair).map_or(false, |&exp| exp > it) };
            debug_assert_eq!(is_tabu, tabu.get(&pair).map_or(false, |&exp| exp > it));
            if (!is_tabu || est < best_real) && est < best_est {
                best_est = est;
                best_swap = Some((m, p));
                best_pair = pair;
            }
            if est < fb_est {
                fb_est = est;
                fb_swap = Some((m, p));
                fb_pair = pair;
            }
        }
        let mm = best_machine_move_mg(g, &path, best_real, &tabu, it, &tabu_hi, sweep_cap,
                                      &mut mg_w, &mut mg_stamp, &mut mg_gen);
        let mm_l = mm.map_or(INF, |(_, _, _, l)| l);

        if best_swap.is_none() && mm.is_none() {
            if let Some((m, p)) = fb_swap {
                best_swap = Some((m, p));
                best_est = fb_est;
                best_pair = fb_pair;
            } else {
                break;
            }
        }
        let mut mg_revert: Option<(usize, usize, usize, u32)> = None;
        let mut swap_revert: Option<(usize, usize)> = None;
        if mm_l < best_est {
            let (v, k, pos, _) = mm.unwrap();
            let m_old = g.op_machine[v];
            // `mpos` is the position `position()` used to look up linearly; `repair_machine`
            // keeps it exact for every operation currently in a sequence.
            debug_assert_eq!(Some(g.mpos[v]), g.machine_seq[m_old].iter().position(|&x| x == v));
            let p_old = g.mpos[v];
            let old_dur = g.op_dur[v];
            let dt = g.pre.job_ops[g.pre.op_job[v]][g.pre.op_idx[v]].iter()
                .find(|&&(m2, _)| m2 == k).map(|&(_, d)| d).unwrap();
            g.machine_seq[m_old].remove(p_old);
            g.op_machine[v] = k;
            g.op_dur[v] = dt;
            g.machine_seq[k].insert(pos, v);
            g.repair_machine(m_old);
            g.repair_machine(k);
            tabu.insert((v, usize::MAX), it + tenure);
            if it + tenure > tabu_hi[v] { tabu_hi[v] = it + tenure; }
            mg_revert = Some((v, m_old, p_old, old_dur));
        } else {
            let (m, p) = best_swap.unwrap();
            g.machine_seq[m].swap(p, p + 1);
            g.repair_machine(m);
            tabu.insert(best_pair, it + tenure);
            if it + tenure > tabu_hi[best_pair.0] { tabu_hi[best_pair.0] = it + tenure; }
            if it + tenure > tabu_hi[best_pair.1] { tabu_hi[best_pair.1] = it + tenure; }
            swap_revert = Some((m, p));
        }
        cur_real = match g.eval() {
            Some(r) => r,
            None => {
                if let Some((v, m_old, p_old, old_dur)) = mg_revert {
                    let k = g.op_machine[v];
                    debug_assert_eq!(Some(g.mpos[v]), g.machine_seq[k].iter().position(|&x| x == v));
                    let pn = g.mpos[v];
                    g.machine_seq[k].remove(pn);
                    g.op_machine[v] = m_old;
                    g.op_dur[v] = old_dur;
                    g.machine_seq[m_old].insert(p_old, v);
                    g.repair_machine(k);
                    g.repair_machine(m_old);
                } else if let Some((m, p)) = swap_revert {
                    g.machine_seq[m].swap(p, p + 1);
                    g.repair_machine(m);
                }
                match g.eval() { Some(r) => r, None => break }
            }
        };
        if cur_real < best_real {
            best_real = cur_real;
            best_om.copy_from_slice(&g.op_machine);
            for (dst, src) in best_ms.iter_mut().zip(g.machine_seq.iter()) {
                dst.clear();
                dst.extend_from_slice(src);
            }
            no_improve = 0;
        } else {
            no_improve += 1;
            if no_improve >= max_no_improve {
                if kicks_left == 0 { break; }
                g.reset(&best_om, &best_ms);
                kick_critical(g, 3 + (max_kicks - kicks_left), rng);
                cur_real = match g.eval() { Some(r) => r, None => break };
                tabu.clear();
                for h in tabu_hi.iter_mut() { *h = 0; }
                no_improve = 0;
                kicks_left -= 1;
            }
        }
    }
    g.reset(&best_om, &best_ms);
    g.eval();
    Some(best_real)
}

fn kick(g: &mut Graph, k: usize, rng: &mut SmallRng) {
    if g.eval().is_none() { return; }
    let nm = g.pre.num_machines;
    let mut applied = 0usize;
    let mut attempts = 0usize;
    while applied < k && attempts < k * 5 {
        attempts += 1;
        if rng.gen_bool(0.7) {
            let m = rng.gen_range(0..nm);
            let len = g.machine_seq[m].len();
            if len < 2 { continue; }
            let p = rng.gen_range(0..len - 1);
            g.machine_seq[m].swap(p, p + 1);
            g.repair_machine(m);
            if g.eval().is_none() { g.machine_seq[m].swap(p, p + 1); g.repair_machine(m); } else { applied += 1; }
        } else {
            let o = rng.gen_range(0..g.pre.n_ops);
            let j = g.pre.op_job[o];
            let oi = g.pre.op_idx[o];
            let flex = g.pre.job_ops[j][oi].len();
            if flex < 2 { continue; }
            let m_old = g.op_machine[o];
            let (k2, dt) = g.pre.job_ops[j][oi][rng.gen_range(0..flex)];
            if k2 == m_old { continue; }
            debug_assert_eq!(Some(g.mpos[o]), g.machine_seq[m_old].iter().position(|&x| x == o));
            let p_old = g.mpos[o];
            let dur_old = g.op_dur[o];
            g.machine_seq[m_old].remove(p_old);
            g.op_machine[o] = k2;
            g.op_dur[o] = dt;
            let pins = insert_pos(g, k2, o);
            g.machine_seq[k2].insert(pins, o);
            g.repair_machine(m_old);
            g.repair_machine(k2);
            if g.eval().is_none() {
                debug_assert_eq!(Some(g.mpos[o]), g.machine_seq[k2].iter().position(|&x| x == o));
                let pn = g.mpos[o];
                g.machine_seq[k2].remove(pn);
                g.op_machine[o] = m_old;
                g.op_dur[o] = dur_old;
                g.machine_seq[m_old].insert(p_old, o);
                g.repair_machine(k2);
                g.repair_machine(m_old);
            } else {
                applied += 1;
            }
        }
    }
    let _ = g.eval();
}

fn relax_width(pre: &Pre, der: &Der) -> usize {
    match track_of(pre, der) {
        Track::Hybrid => 6,
        _ => 0,
    }
}

fn best_insertion(g: &Graph, o: usize, m: usize, d: u32) -> (u32, usize) {
    let seq = &g.machine_seq[m];
    let j = g.pre.op_job[o];
    let oi = g.pre.op_idx[o];
    let rj = if oi > 0 { g.hend[o - 1] } else { 0 };
    let qj = if oi + 1 < g.pre.job_nops[j] { g.qend[o + 1] } else { 0 };
    let (mut best, mut at, mut r) = (INF, 0usize, rj);
    for k in 0..=seq.len() {
        if k > 0 {
            let p = seq[k - 1];
            let v = g.hend[p];
            if v > r { r = v; }
        }
        if r + d + qj >= best { break; }
        let q = if k < seq.len() {
            let s = seq[k];
            g.qend[s].max(qj)
        } else { qj };
        let est = r + d + q;
        if est < best { best = est; at = k; }
    }
    (best, at)
}

fn relax_and_rebuild(g: &mut Graph, k: usize, rng: &mut SmallRng) -> Option<u32> {
    let before_mk = g.eval()?;
    if before_mk == 0 { return None; }

    let width = (before_mk / 4).max(1);
    let t0 = rng.gen_range(0..before_mk.saturating_sub(width).max(1));
    let t1 = t0 + width;

    let mut cands: Vec<usize> = Vec::new();
    for o in 0..g.pre.n_ops {
        let j = g.pre.op_job[o];
        let oi = g.pre.op_idx[o];
        if g.pre.job_ops[j][oi].len() < 2 { continue; }
        if g.heads[o] < t1 && g.hend[o] > t0 { cands.push(o); }
    }
    if cands.len() < 2 { return None; }
    let taken = k.min(cands.len());
    for i in 0..taken {
        let j = i + rng.gen_range(0..(cands.len() - i));
        cands.swap(i, j);
    }
    cands.truncate(taken);

    let saved_om = g.op_machine.clone();
    let saved_dur = g.op_dur.clone();
    let saved_seq = g.machine_seq.clone();

    for &o in &cands {
        let m = g.op_machine[o];
        // `mpos` is only valid for operations still held by a sequence; the candidates are
        // removed one after the other, so the positions of the later ones shift. This site keeps
        // the linear search on purpose -- it runs once per relax, not once per iteration.
        if let Some(p) = g.machine_seq[m].iter().position(|&x| x == o) {
            g.machine_seq[m].remove(p);
        }
    }
    g.sync_arcs();
    if g.eval().is_none() {
        g.op_machine.copy_from_slice(&saved_om);
        g.op_dur.copy_from_slice(&saved_dur);
        g.machine_seq.clone_from(&saved_seq);
        g.sync_arcs();
        let _ = g.eval();
        return None;
    }

    for &o in &cands {
        let j = g.pre.op_job[o];
        let oi = g.pre.op_idx[o];
        let (mut best, mut bm, mut bp, mut bd) = (INF, usize::MAX, 0usize, 0u32);
        for &(m, d) in &g.pre.job_ops[j][oi] {
            let (est, at) = best_insertion(g, o, m, d);
            if est < best { best = est; bm = m; bp = at; bd = d; }
        }
        if bm == usize::MAX { bm = g.op_machine[o]; bp = 0; bd = g.op_dur[o]; }
        let at = bp.min(g.machine_seq[bm].len());
        g.machine_seq[bm].insert(at, o);
        g.op_machine[o] = bm;
        g.op_dur[o] = bd;
        g.repair_machine(bm);
    }

    match g.eval() {
        Some(after_mk) if after_mk < before_mk => Some(after_mk),
        _ => {
            g.op_machine.copy_from_slice(&saved_om);
            g.op_dur.copy_from_slice(&saved_dur);
            g.machine_seq.clone_from(&saved_seq);
            g.sync_arcs();
            let _ = g.eval();
            None
        }
    }
}

fn kick_critical(g: &mut Graph, strength: usize, rng: &mut SmallRng) {
    let cur = match g.eval() { Some(r) => r, None => return };
    let mut path: Vec<usize> = Vec::new();
    critical_path(g, cur, &mut path);
    if path.len() < 2 { kick(g, strength, rng); return; }
    let mut cands: Vec<(usize, usize)> = Vec::with_capacity(path.len() * 2);
    for &u in &path {
        let m = g.op_machine[u];
        debug_assert_eq!(Some(g.mpos[u]), g.machine_seq[m].iter().position(|&x| x == u));
        {
            let pos = g.mpos[u];
            if pos > 0 { cands.push((m, pos - 1)); }
            if pos + 1 < g.machine_seq[m].len() { cands.push((m, pos)); }
        }
    }
    cands.sort_unstable();
    cands.dedup();
    if cands.is_empty() { kick(g, strength, rng); return; }
    let mut applied = 0usize;
    let mut attempts = 0usize;
    while applied < strength && attempts < strength * 5 {
        attempts += 1;
        let (m, pos) = cands[rng.gen_range(0..cands.len())];
        if pos + 1 < g.machine_seq[m].len() {
            g.machine_seq[m].swap(pos, pos + 1);
            g.repair_machine(m);
            if g.eval().is_none() { g.machine_seq[m].swap(pos, pos + 1); g.repair_machine(m); } else { applied += 1; }
        }
    }
    let _ = g.eval();
}

struct Der {
    op_min: Vec<u32>,
    suf_min: Vec<Vec<u32>>,
    machine_load0: Vec<f64>,
    machine_scarcity: Vec<f64>,
    machine_best_pop: Vec<f64>,
    avg_machine_load: f64,
    avg_machine_scarcity: f64,
    avg_op_min: f64,
    horizon: f64,
    time_scale: f64,
    flex_avg: f64,
    flex_factor: f64,
    high_flex: f64,
    jobshopness: f64,
    chaotic_like: bool,
    use_rich: bool,
}

impl Der {
    fn build(pre: &Pre) -> Der {
        let nm = pre.num_machines;
        let mut op_min = vec![0u32; pre.n_ops];
        let mut machine_load0 = vec![0.0f64; nm];
        let mut machine_scarcity = vec![0.0f64; nm];
        let mut machine_best_cnt = vec![0.0f64; nm];
        let mut total_min_work = 0.0f64;
        let mut total_flex = 0.0f64;
        for j in 0..pre.num_jobs {
            for oi in 0..pre.job_nops[j] {
                let oid = pre.op_id(j, oi);
                let elig = &pre.job_ops[j][oi];
                let flex = elig.len();
                let mut mn = u32::MAX;
                let mut best_m = elig[0].0;
                for &(m, pt) in elig { if pt < mn { mn = pt; best_m = m; } }
                op_min[oid] = mn;
                total_min_work += mn as f64;
                total_flex += flex as f64;
                machine_best_cnt[best_m] += 1.0;
                let ff = flex as f64;
                let delta = mn as f64 / ff;
                let delta_s = mn as f64 / (ff * ff);
                for &(m, _) in elig { machine_load0[m] += delta; machine_scarcity[m] += delta_s; }
            }
        }
        let mut suf_min = Vec::with_capacity(pre.num_jobs);
        for j in 0..pre.num_jobs {
            let n = pre.job_nops[j];
            let mut s = vec![0u32; n + 1];
            for oi in (0..n).rev() { s[oi] = s[oi + 1].saturating_add(op_min[pre.op_id(j, oi)]); }
            suf_min.push(s);
        }
        let n_ops = (pre.n_ops as f64).max(1.0);
        let avg_machine_load = (total_min_work / (nm as f64).max(1.0)).max(1.0);
        let horizon = avg_machine_load;
        let avg_op_min = (total_min_work / n_ops).max(1.0);
        let flex_avg = (total_flex / n_ops).max(1.0);
        let flex_factor = (3.0 / flex_avg).clamp(0.6, 2.2);
        let high_flex = ((flex_avg - 3.0) / 7.0).clamp(0.0, 1.0);
        let avg_machine_scarcity =
            (machine_scarcity.iter().sum::<f64>() / (nm as f64).max(1.0)).max(1e-9);
        let tot: f64 = machine_best_cnt.iter().sum();
        let mean = (tot / (nm as f64).max(1.0)).max(1e-9);
        let machine_best_pop = machine_best_cnt.iter()
            .map(|&c| { let r = (c / mean).clamp(0.0, 10.0); (r / (1.0 + r)).clamp(0.0, 1.0) })
            .collect();
        let chaotic_like = high_flex > 0.85;
        let use_rich = chaotic_like || (flex_avg >= 2.5 && flex_avg <= 6.0);
        let jobshopness = JOBSHOPNESS_MEDIUM_OVERRIDE
            .filter(|_| !chaotic_like)
            .unwrap_or(0.85);
        let time_scale = (horizon * (2.65 + 0.10 * jobshopness + 0.10 * high_flex)).max(1.0);
        Der {
            op_min, suf_min, machine_load0, machine_scarcity, machine_best_pop,
            avg_machine_load, avg_machine_scarcity, avg_op_min, horizon, time_scale,
            flex_avg, flex_factor, high_flex, jobshopness, chaotic_like, use_rich,
        }
    }
}

#[inline]
fn best_second_counts(time: u32, avail: &[u32], elig: &[(usize, u32)]) -> (u32, u32, usize) {
    let mut best = INF;
    let mut second = INF;
    let mut cnt = 0usize;
    for &(m, pt) in elig {
        let end = time.max(avail[m]).saturating_add(pt);
        if end < best { second = best; best = end; cnt = 1; }
        else if end == best { cnt += 1; }
        else if end < second { second = end; }
    }
    if cnt > 1 { second = best; }
    (best, second, cnt.max(1))
}

fn fjsp_construct(pre: &Pre, der: &Der, k: usize, rng: &mut SmallRng, route: Option<&[Vec<f32>]>)
    -> (Vec<usize>, Vec<Vec<usize>>) {
    let nj = pre.num_jobs;
    let nm = pre.num_machines;
    let mut job_next = vec![0usize; nj];
    let mut job_ready = vec![0u32; nj];
    let mut avail = vec![0u32; nm];
    let mut load = der.machine_load0.clone();
    let mut work = vec![0u64; nm];
    let mut op_machine = vec![0usize; pre.n_ops];
    let mut machine_seq: Vec<Vec<usize>> = vec![Vec::new(); nm];
    let mut remaining = pre.n_ops;
    let mut time = 0u32;
    let use_rand = k > 1;
    let js = der.jobshopness;
    let fl = 1.0 - js;
    let mut cands: Vec<(f64, usize, u32, usize)> = Vec::new();

    while remaining > 0 {
        let idle: Vec<usize> = (0..nm).filter(|&m| avail[m] <= time).collect();
        let sum_work: u64 = work.iter().sum();
        let avg_work = sum_work as f64 / (nm as f64).max(1.0);
        for &m in &idle {
            cands.clear();
            for j in 0..nj {
                let oi = job_next[j];
                if oi >= pre.job_nops[j] || job_ready[j] > time { continue; }
                let elig = &pre.job_ops[j][oi];
                let Some(&(_, pt)) = elig.iter().find(|&&(mm, _)| mm == m) else { continue };
                let (best_end, second_end, cnt) = best_second_counts(time, &avail, elig);
                if time.max(avail[m]).saturating_add(pt) != best_end { continue; }
                let oid = pre.op_id(j, oi);
                let flex = elig.len() as f64;
                let flex_inv = 1.0 / flex;
                let regret = if second_end >= INF { der.avg_op_min * 2.6 } else { (second_end - best_end) as f64 };
                let reg_n = (regret / der.avg_op_min.max(1.0)).clamp(0.0, 6.0);
                let scar_urg = 1.0 / (cnt as f64).max(1.0);
                let load_n = load[m] / der.avg_machine_load.max(1e-9);
                let scar_n = der.machine_scarcity[m] / der.avg_machine_scarcity;
                let scarce_match = scar_n * (flex_inv - 1.0 / der.flex_avg);
                let end_n = best_end as f64 / der.time_scale.max(1.0);
                let proc_n = pt as f64 / der.avg_op_min.max(1.0);
                let rem_min_n = der.suf_min[j][oi] as f64 / der.horizon.max(1.0);
                let progress = 1.0 - remaining as f64 / (pre.n_ops as f64).max(1.0);
                let end_w = (0.90 * fl + 0.72 * js) + (0.62 + 0.12 * fl) * progress + 0.18 * der.high_flex;
                let reg_w = (0.50 * fl + 0.78 * js) + 0.18 * (1.0 - progress);
                let load_w = if der.flex_avg >= 5.0 { -(0.45 * fl + 0.75 * js) * der.flex_factor }
                             else { (0.45 * fl + 0.75 * js) * der.flex_factor };
                let pop_pen = if der.chaotic_like && elig.len() >= 2 {
                    (0.07 + 0.15 * (1.0 - progress)).clamp(0.05, 0.24) * der.machine_best_pop[m] * der.flex_factor
                } else { 0.0 };
                let bal_pen = if der.chaotic_like {
                    let dw = (avg_work + (der.avg_op_min * 3.0).max(1.0)).max(1.0);
                    let r = work[m] as f64 / dw;
                    -0.07 * (r / (r + 1.0)).clamp(0.0, 1.0)
                } else { 0.0 };
                let jitter = if use_rand { rng.gen::<f64>() * 1e-9 } else { 0.0 };
                let route_bonus = route.map_or(0.0, |r| 0.90 * r[oid][m] as f64 * der.flex_factor);
                let score = 1.05 * rem_min_n
                    + reg_w * der.flex_factor * reg_n
                    + 0.62 * der.flex_factor * flex_inv
                    + 0.55 * der.flex_factor * scarce_match
                    + load_w * load_n
                    + 0.20 * der.flex_factor * scar_urg
                    - end_w * end_n
                    - (0.18 * fl + 0.12 * js) * proc_n
                    - pop_pen
                    + bal_pen
                    + route_bonus
                    + jitter;
                cands.push((score, j, pt, oid));
            }
            if cands.is_empty() { continue; }
            let (_, j, pt, oid) = if use_rand {
                cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
                if der.chaotic_like {
                    let progress = 1.0 - remaining as f64 / (pre.n_ops as f64).max(1.0);
                    let k_eff = if progress > 0.80 { 1 } else if progress > 0.60 { (k + 1) / 2 } else { k };
                    let kk = k_eff.max(1).min(cands.len()).min(8);
                    if kk <= 1 {
                        cands[0]
                    } else {
                        let smin = cands[kk - 1].0;
                        let mut wsum = 0.0f64;
                        for i in 0..kk { let d = cands[i].0 - smin; wsum += d * d; }
                        if wsum <= 1e-18 {
                            cands[rng.gen_range(0..kk)]
                        } else {
                            let mut r = rng.gen::<f64>() * wsum;
                            let mut sel = kk - 1;
                            for i in 0..kk { let d = cands[i].0 - smin; let wgt = d * d; if r < wgt { sel = i; break; } r -= wgt; }
                            cands[sel]
                        }
                    }
                } else {
                    let kk = k.min(cands.len());
                    cands[rng.gen_range(0..kk)]
                }
            } else {
                *cands.iter().max_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(b.1.cmp(&a.1))).unwrap()
            };
            let oi = job_next[j];
            let start = time.max(avail[m]);
            let end = start + pt;
            op_machine[oid] = m;
            machine_seq[m].push(oid);
            job_next[j] += 1;
            job_ready[j] = end;
            avail[m] = end;
            work[m] += pt as u64;
            let elig = &pre.job_ops[j][oi];
            let delta = der.op_min[oid] as f64 / (elig.len() as f64).max(1.0);
            for &(mm, _) in elig { load[mm] = (load[mm] - delta).max(0.0); }
            remaining -= 1;
        }
        if remaining == 0 { break; }
        let mut next: Option<u32> = None;
        for &t in &avail { if t > time { next = Some(next.map_or(t, |b| b.min(t))); } }
        for j in 0..nj {
            if job_next[j] < pre.job_nops[j] && job_ready[j] > time {
                let t = job_ready[j];
                next = Some(next.map_or(t, |b| b.min(t)));
            }
        }
        match next { Some(t) => time = t, None => break }
    }
    (op_machine, machine_seq)
}

struct PoolMember {
    mk: u32,
    om: Vec<usize>,
    ms: Vec<Vec<usize>>,
    deep: bool,
}

fn sol_distance(
    pre: &Pre, om_a: &[usize], ms_a: &[Vec<usize>], om_b: &[usize], ms_b: &[Vec<usize>],
) -> usize {
    let n = pre.n_ops;
    let mut d = 0usize;
    for o in 0..n { if om_a[o] != om_b[o] { d += 2; } }
    let mut pa = vec![0usize; n];
    let mut pb = vec![0usize; n];
    for seq in ms_a { for (p, &o) in seq.iter().enumerate() { pa[o] = p; } }
    for seq in ms_b { for (p, &o) in seq.iter().enumerate() { pb[o] = p; } }
    for o in 0..n { if pa[o] != pb[o] { d += 1; } }
    d
}

fn pool_consider(
    pool: &mut Vec<PoolMember>, pre: &Pre, mk: u32,
    om: &[usize], ms: &[Vec<usize>], cap: usize, min_dist: usize,
) -> bool {
    let mut nearest = usize::MAX;
    let mut nd = usize::MAX;
    for (i, e) in pool.iter().enumerate() {
        let dd = sol_distance(pre, &e.om, &e.ms, om, ms);
        if dd < nd { nd = dd; nearest = i; }
    }
    if nearest != usize::MAX && nd < min_dist {
        if mk < pool[nearest].mk {
            pool[nearest].mk = mk;
            pool[nearest].om.copy_from_slice(om);
            for (dst, src) in pool[nearest].ms.iter_mut().zip(ms.iter()) { dst.clear(); dst.extend_from_slice(src); }
            pool[nearest].deep = false;
            return true;
        }
        return false;
    }
    if pool.len() < cap {
        pool.push(PoolMember { mk, om: om.to_vec(), ms: ms.to_vec(), deep: false });
        return true;
    }
    let mut worst = 0usize;
    for i in 1..pool.len() { if pool[i].mk > pool[worst].mk { worst = i; } }
    if mk < pool[worst].mk {
        pool[worst] = PoolMember { mk, om: om.to_vec(), ms: ms.to_vec(), deep: false };
        return true;
    }
    false
}

fn build_consensus(pre: &Pre, pool: &[PoolMember]) -> Vec<Vec<f32>> {
    let nm = pre.num_machines;
    let mut route = vec![vec![0f32; nm]; pre.n_ops];
    if pool.is_empty() { return route; }
    let mut idx: Vec<usize> = (0..pool.len()).collect();
    idx.sort_by_key(|&i| pool[i].mk);
    let mut wsum = 0f32;
    for (rank, &i) in idx.iter().enumerate() {
        let w = 1.0 / (1.0 + rank as f32);
        wsum += w;
        for oid in 0..pre.n_ops { route[oid][pool[i].om[oid]] += w; }
    }
    if wsum > 0.0 { for r in route.iter_mut() { for v in r.iter_mut() { *v /= wsum; } } }
    route
}

fn make_seed(pre: &Pre, der: &Der, use_rich: bool, idx: usize, rng: &mut SmallRng)
    -> (Vec<usize>, Vec<Vec<usize>>) {
    if use_rich {
        if idx == 0 {
            fjsp_construct(pre, der, 0, rng, None)
        } else if idx % 5 == 4 {
            kacem_construct(pre, Some(rng))
        } else {
            let k = 2 + (idx % 4);
            fjsp_construct(pre, der, k, rng, None)
        }
    } else {
        if idx == 0 {
            dispatch_construct(pre, 0, None, 0)
        } else if idx % 4 == 3 {
            kacem_construct(pre, Some(rng))
        } else {
            let rule = (idx % 5) as u8;
            let top_k = 2 + (idx % 5);
            dispatch_construct(pre, rule, Some(rng), top_k)
        }
    }
}

fn sota_makespan(challenge: &Challenge) -> Option<u32> {
    let capture: std::cell::RefCell<Option<Solution>> = std::cell::RefCell::new(None);
    {
        let sink = |s: &Solution| -> Result<()> {
            *capture.borrow_mut() = Some(s.clone());
            Ok(())
        };
        sota::solve_challenge_with_effort(challenge, &sink, 1).ok()?;
    }
    let s = capture.into_inner()?;
    challenge.evaluate_makespan(&s).ok()
}

#[derive(Clone, Copy, PartialEq)]
enum Track { FjspHigh, FjspMedium, Hybrid }

fn track_of(pre: &Pre, der: &Der) -> Track {
    if der.chaotic_like { Track::FjspHigh }
    else if der.flex_avg >= 2.0 && pre.n_products >= 27 { Track::FjspMedium }
    else { Track::Hybrid }
}

fn quality_target(pre: &Pre, der: &Der) -> Option<i64> {
    if track_of(pre, der) == Track::FjspMedium {
        return None;
    }
    Some(match track_of(pre, der) {
        Track::FjspHigh   =>  70_000,
        Track::FjspMedium => 125_000,
        Track::Hybrid     =>  95_000,
    })
}

fn solve_pool(
    challenge: &Challenge, pre: &Pre, der: &Der,
    save_solution: &dyn Fn(&Solution) -> Result<()>, rng: &mut SmallRng,
) -> Result<()> {
    let use_rich = der.use_rich;
    let tenure = 8 + pre.num_jobs / 4;
    let n_ops = pre.n_ops;
    let cap = 16usize;
    let is_fjsp_medium = !der.chaotic_like && der.flex_avg >= 2.0 && pre.n_products >= 27;
    // This function is only reached when the dispatcher did NOT take the hybrid route, i.e.
    // `chaotic_like || n_products >= 27`: a "hybrid-like" arm here would be unreachable.
    let elites = if is_fjsp_medium { 3usize } else if der.chaotic_like { 6usize } else { 4usize };
    let min_dist = (n_ops / 16 + 2).max(2);
    let polish_iters = 4000usize;
    let deep_iters = 200000usize;

    const SOTA_BUDGET_MINI: u64 = 100_000_000_000;
    let avail_before = fuel_left();
    let relax = relax_width(pre, der);
    let target = quality_target(pre, der);
    let sota = if target.is_some() && avail_before >= SOTA_BUDGET_MINI {
        sota_makespan(challenge)
    } else {
        None
    };

    let avail_fuel = fuel_left();
    let work_cap = match track_of(pre, der) {
        Track::FjspHigh   => 100_000_000_000u64,
        Track::FjspMedium => 200_000_000_000u64,
        Track::Hybrid     => 200_000_000_000u64,
    };
    let total0 = avail_fuel.saturating_sub(FUEL_STOP).min(work_cap);
    let floor_fuel = avail_fuel.saturating_sub(total0);
    let build_floor = floor_fuel + total0 * 55 / 100;

    let (om0, ms0) = make_seed(pre, der, use_rich, 0, rng);
    let mut g = Graph::new(pre, om0, ms0);

    let mut pool: Vec<PoolMember> = Vec::new();
    let mut best_mk = INF;
    let mut best_om = g.op_machine.clone();
    let mut best_ms = g.machine_seq.clone();

    if der.chaotic_like {
        let a0_cap = 24usize;
        let a0_floor = floor_fuel + total0 * 85 / 100;
        let a0_draw_cap = 4000usize;
        let mut c = 0usize;
        let mut route: Vec<Vec<f32>> = Vec::new();
        while fuel_left() > a0_floor && c < a0_draw_cap {
            if c % 24 == 0 && pool.len() >= 3 { route = build_consensus(pre, &pool); }
            let k = 2 + (c % 5);
            let use_route = !route.is_empty() && c % 2 == 0;
            let (om, ms) = fjsp_construct(pre, der, k, rng, if use_route { Some(&route[..]) } else { None });
            g.reset(&om, &ms);
            if let Some(mk) = g.eval() {
                if mk < best_mk {
                    let s = g.to_solution();
                    if challenge.evaluate_makespan(&s).is_ok() {
                        best_mk = mk; save_solution(&s)?;
                        best_om.copy_from_slice(&g.op_machine);
                        for (d, sm) in best_ms.iter_mut().zip(g.machine_seq.iter()) { d.clear(); d.extend_from_slice(sm); }
                    }
                }
                pool_consider(&mut pool, pre, mk, &g.op_machine, &g.machine_seq, a0_cap, min_dist);
            }
            c += 1;
        }
    }

    if is_fjsp_medium {
        let a0_cap = 24usize;
        let a0_floor = floor_fuel + total0 * 85 / 100;
        let mut c = 0usize;
        while fuel_left() > a0_floor && c < 4000 {
            let k = 2 + (c % 5);
            let (om, ms) = fjsp_construct(pre, der, k, rng, None);
            g.reset(&om, &ms);
            if let Some(mk) = g.eval() {
                if mk < best_mk {
                    let s = g.to_solution();
                    if challenge.evaluate_makespan(&s).is_ok() {
                        best_mk = mk; save_solution(&s)?;
                        best_om.copy_from_slice(&g.op_machine);
                        for (d, sm) in best_ms.iter_mut().zip(g.machine_seq.iter()) { d.clear(); d.extend_from_slice(sm); }
                    }
                }
                pool_consider(&mut pool, pre, mk, &g.op_machine, &g.machine_seq, a0_cap, min_dist);
            }
            c += 1;
        }
    }

    let k_init = if der.chaotic_like || is_fjsp_medium { 0usize } else { 16usize };
    for idx in 0..k_init {
        if fuel_left() <= build_floor { break; }
        let (om, ms) = make_seed(pre, der, use_rich, idx, rng);
        g.reset(&om, &ms);
        if let Some(mk) = tabu_run(&mut g, polish_iters, tenure, false, 0, floor_fuel, rng) {
            if mk < best_mk {
                let s = g.to_solution();
                if challenge.evaluate_makespan(&s).is_ok() {
                    best_mk = mk;
                    save_solution(&s)?;
                    best_om.copy_from_slice(&g.op_machine);
                    for (d, sm) in best_ms.iter_mut().zip(g.machine_seq.iter()) { d.clear(); d.extend_from_slice(sm); }
                }
            }
            pool_consider(&mut pool, pre, mk, &g.op_machine, &g.machine_seq, cap, min_dist);
        }
    }
    if pool.is_empty() {
        if g.eval().is_some() {
            pool_consider(&mut pool, pre, g.eval().unwrap_or(INF), &g.op_machine, &g.machine_seq, cap, min_dist);
        }
    }

    let mut refresh_idx = k_init;
    let mut kk = 2usize;
    let mut stall = 0usize;

    while fuel_left() > floor_fuel {
        let fuel_round_start = fuel_left();
        pool.sort_by(|a, b| a.mk.cmp(&b.mk)
            .then_with(|| a.om.cmp(&b.om))
            .then_with(|| a.ms.cmp(&b.ms)));
        let e = elites.min(pool.len());
        let mut improved = false;

        for i in 0..e {
            if fuel_left() <= floor_fuel { break; }
            if pool[i].deep { continue; }
            let rem = fuel_left().saturating_sub(floor_fuel);
            let floor = (fuel_left() - rem / 4).max(floor_fuel);
            g.reset(&pool[i].om, &pool[i].ms);
            if let Some(mk) = tabu_run(&mut g, deep_iters, tenure, true, 8, floor, rng) {
                if mk < pool[i].mk {
                    pool[i].mk = mk;
                    pool[i].om.copy_from_slice(&g.op_machine);
                    for (d, sm) in pool[i].ms.iter_mut().zip(g.machine_seq.iter()) { d.clear(); d.extend_from_slice(sm); }
                    pool[i].deep = false;
                } else {
                    pool[i].deep = true;
                }
                if mk < best_mk {
                    let s = g.to_solution();
                    if challenge.evaluate_makespan(&s).is_ok() {
                        best_mk = mk; save_solution(&s)?; improved = true;
                        best_om.copy_from_slice(&g.op_machine);
                        for (d, sm) in best_ms.iter_mut().zip(g.machine_seq.iter()) { d.clear(); d.extend_from_slice(sm); }
                    }
                }
            }
        }

        if relax > 0 && fuel_left() > floor_fuel && best_mk != INF {
            g.reset(&best_om, &best_ms);
            if let Some(mk) = relax_and_rebuild(&mut g, relax, rng) {
                if mk < best_mk {
                    let s = g.to_solution();
                    if challenge.evaluate_makespan(&s).is_ok() {
                        best_mk = mk; save_solution(&s)?; improved = true;
                        best_om.copy_from_slice(&g.op_machine);
                        for (d, sm) in best_ms.iter_mut().zip(g.machine_seq.iter()) { d.clear(); d.extend_from_slice(sm); }
                    }
                }
                pool_consider(&mut pool, pre, mk, &g.op_machine, &g.machine_seq, cap, min_dist);
            }
        }

        if fuel_left() > floor_fuel && best_mk != INF {
            let rem = fuel_left().saturating_sub(floor_fuel);
            let floor = (fuel_left() - rem / 3).max(floor_fuel);
            g.reset(&best_om, &best_ms);
            kick_critical(&mut g, kk, rng);
            if let Some(mk) = tabu_run(&mut g, deep_iters, tenure, true, 6, floor, rng) {
                if mk < best_mk {
                    let s = g.to_solution();
                    if challenge.evaluate_makespan(&s).is_ok() {
                        best_mk = mk; save_solution(&s)?; improved = true;
                        best_om.copy_from_slice(&g.op_machine);
                        for (d, sm) in best_ms.iter_mut().zip(g.machine_seq.iter()) { d.clear(); d.extend_from_slice(sm); }
                    }
                }
                pool_consider(&mut pool, pre, mk, &g.op_machine, &g.machine_seq, cap, min_dist);
            }
        }

        if fuel_left() > floor_fuel {
            let (om, ms) = if der.chaotic_like && !pool.is_empty() && refresh_idx % 2 == 0 {
                let route = build_consensus(pre, &pool);
                fjsp_construct(pre, der, 3, rng, Some(&route))
            } else {
                make_seed(pre, der, use_rich, refresh_idx, rng)
            };
            refresh_idx += 1;
            g.reset(&om, &ms);
            if let Some(mk) = tabu_run(&mut g, polish_iters, tenure, false, 0, floor_fuel, rng) {
                if mk < best_mk {
                    let s = g.to_solution();
                    if challenge.evaluate_makespan(&s).is_ok() {
                        best_mk = mk; save_solution(&s)?; improved = true;
                        best_om.copy_from_slice(&g.op_machine);
                        for (d, sm) in best_ms.iter_mut().zip(g.machine_seq.iter()) { d.clear(); d.extend_from_slice(sm); }
                    }
                }
                pool_consider(&mut pool, pre, mk, &g.op_machine, &g.machine_seq, cap, min_dist);
            }
        }

        if improved {
            stall = 0; kk = 2;
        } else {
            stall += 1;
            if stall % 3 == 0 { kk = (kk + 1).min(8); }
        }

        if let (Some(c), Some(s)) = (target, sota) {
            if best_mk != INF && s > 0 {
                let q = (s as i64 - best_mk as i64) * 1_000_000 / s as i64;
                if q >= c { break; }
            }
        }

        if fuel_left() >= fuel_round_start { break; }
    }

    Ok(())
}

fn greedy_floor(pre: &Pre, guarded: &dyn Fn(&Solution) -> Result<()>) -> Result<()> {
    let (om0, ms0) = dispatch_construct(pre, 0, None, 0);
    let mut g = Graph::new(pre, om0, ms0);
    for rule in 0..5u8 {
        let (om, ms) = dispatch_construct(pre, rule, None, 0);
        g.reset(&om, &ms);
        if g.eval().is_some() {
            guarded(&g.to_solution())?;
        }
    }
    Ok(())
}

pub fn help() {
    println!("ember_stromboli (c007 job_scheduling)");
    println!();
    println!("No tunables. Any key passed is accepted and ignored.");
    println!("The fuel budget is read from `__fuel_remaining`; the solver is anytime.");
}

fn run_bounded(
    challenge: &Challenge,
    hp: Map<String, Value>,
    work_cap: u64,
    target: Option<i64>,
    guarded: &dyn Fn(&Solution) -> Result<()>,
    engine: fn(&Challenge, &dyn Fn(&Solution) -> Result<()>, &Option<Map<String, Value>>) -> Result<()>,
) -> Result<()> {
    let sota = if target.is_some() { sota_makespan(challenge) } else { None };
    let fuel_start = fuel_left();
    let floor_fuel = fuel_start.saturating_sub(work_cap.min(fuel_start.saturating_sub(FUEL_STOP)));

    let bound = |s: &Solution| -> Result<()> {
        guarded(s)?;
        if fuel_left() <= floor_fuel {
            return Err(anyhow::anyhow!("stop"));
        }
        if let (Some(c), Some(so)) = (target, sota) {
            if so > 0 {
                if let Ok(mk) = challenge.evaluate_makespan(s) {
                    if (so as i64 - mk as i64) * 1_000_000 / so as i64 >= c {
                        return Err(anyhow::anyhow!("stop"));
                    }
                }
            }
        }
        Ok(())
    };

    let _ = engine(challenge, &bound, &Some(hp));
    Ok(())
}

fn routes_to_hybrid(pre: &Pre, der: Option<&Der>) -> bool {
    match der {
        Some(d) => !d.chaotic_like && pre.max_flex > 1 && pre.n_products < 27,
        None => false,
    }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let pre = Pre::build(challenge);

    let best_mk = std::cell::Cell::new(u32::MAX);
    let guarded = |s: &Solution| -> Result<()> {
        if let Ok(mk) = challenge.evaluate_makespan(s) {
            if mk <= best_mk.get() {
                best_mk.set(mk);
                return save_solution(s);
            }
        }
        Ok(())
    };

    let der = if pre.max_flex > 1 { Some(Der::build(&pre)) } else { None };
    if !routes_to_hybrid(&pre, der.as_ref()) {
        // The floor pass costs a few e9 whatever the grant. On the sequencing route that is the
        // entire budget of a small grant, and the engine never gets to run; it saves a greedy
        // schedule of its own on entry, so the validity floor is kept either way.
        let lean = pre.max_flex == 1 && pre.n_products >= 27 && fuel_left() < 20_000_000_000u64;
        if !lean {
            gate_floor::gate_floor(&pre, &challenge.seed, 0, &guarded)?;
        }
    }

    if pre.max_flex == 1 && pre.n_products < 27 {
        let mut m = Map::new();
        m.insert("track".to_string(), Value::String("flow_shop".to_string()));
        const FLOW_CAP: u64 = 10_000_000_000u64;
        run_bounded(challenge, m, FLOW_CAP, None, &guarded,
                     flow_shop_engine::solver::solve_challenge)?;
        return greedy_floor(&pre, &guarded);
    }

    if pre.max_flex == 1 && pre.n_products >= 27 {
        let mut m = Map::new();
        m.insert("track".to_string(), Value::String("job_shop".to_string()));
        m.insert("job_shop_iters".to_string(), Value::from(300_000u64));
        m.insert("js_ts_fast_eval".to_string(), Value::from(1u64));
        m.insert("js_seed_select_mode".to_string(), Value::from(1u64));
        m.insert("js_seed_sbp_construct".to_string(), Value::from(2u64));
        m.insert("js_seed_sbp_dual_cycles".to_string(), Value::from(5u64));
        m.insert("js_ts_polish_rescue".to_string(), Value::from(1u64));
        m.insert("js_ts_kick_spread".to_string(), Value::from(4u64));
        m.insert("js_ts_tie_mode".to_string(), Value::from(1u64));
        m.insert("js_ts_starts".to_string(), Value::from(4u64));
        m.insert("js_sbp_rounds".to_string(), Value::from(2u64));
        run_bounded(challenge, m, u64::MAX, None, &guarded,
                     job_shop_engine::solver::solve_challenge)?;
        return greedy_floor(&pre, &guarded);
    }

    let der = der.expect("der");

    if routes_to_hybrid(&pre, Some(&der)) {
        let mut m = Map::new();
        m.insert("fuel_cap".to_string(), Value::from(400000000000u64));
        m.insert("restart_after".to_string(), Value::from(150000u64));
        run_bounded(challenge, m, u64::MAX, None, &guarded,
                     hybrid_engine::solve_challenge)?;
        return Ok(());
    }

    let mut rng = SmallRng::from_seed(challenge.seed);
    solve_pool(challenge, &pre, &der, &guarded, &mut rng)?;
    greedy_floor(&pre, &guarded)
}
