use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::job_scheduling::*;

use crate::{seeded_hasher, HashMap};

#[path = "ember_gate_floor.rs"]
mod gate_floor;
#[path = "ember_hfs_construct.rs"]
mod hfs_construct;
#[path = "ember_hybrid_engine.rs"]
mod hybrid_engine;

const INF: u32 = u32::MAX / 4;
const FUEL_STOP: u64 = 100_000_000;

extern "C" {
    #[allow(non_upper_case_globals)]
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
    dur_pos: bool,
    dur_tab: Vec<u32>,
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
                    let mut row = vec![u32::MAX; challenge.num_machines];
                    for (&m, &t) in op.iter() {
                        if m < challenge.num_machines { row[m] = t; }
                    }
                    let mut cands: Vec<(usize, u32)> = Vec::with_capacity(op.len());
                    for (m, &t) in row.iter().enumerate() {
                        if t != u32::MAX { cands.push((m, t)); }
                    }
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
        let nm = challenge.num_machines;
        let mut dur_tab = vec![u32::MAX; off * nm];
        for (j, ops) in job_ops.iter().enumerate() {
            for (oi, cands) in ops.iter().enumerate() {
                let oid = job_off[j] + oi;
                for &(m, t) in cands { dur_tab[oid * nm + m] = t; }
            }
        }
        Pre {
            num_jobs, num_machines: nm, job_ops, job_nops,
            n_ops: off, job_off, op_job, op_idx, n_products: challenge.jobs_per_product.len(), max_flex,
            op_w, dur_pos, dur_tab,
        }
    }
    #[inline]
    fn op_id(&self, job: usize, oi: usize) -> usize { self.job_off[job] + oi }
    #[inline(always)]
    fn dur_on(&self, o: usize, m: usize) -> u32 {
        unsafe { *self.dur_tab.get_unchecked(o * self.num_machines + m) }
    }
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
    let flat = pre.max_flex == 1 && pre.dur_pos;
    let mut bucket: Vec<Vec<usize>> = if flat { vec![Vec::new(); nm] } else { Vec::new() };

    while remaining > 0 {
        let mut avail: Vec<usize> = (0..nm).filter(|&m| mach_avail[m] <= time).collect();
        if use_random { avail.shuffle(rng.as_mut().unwrap()); }
        if flat {
            for b in bucket.iter_mut() { b.clear(); }
            for j in 0..n {
                let oi = job_next[j];
                if oi < pre.job_nops[j] && job_ready[j] <= time {
                    let m = pre.job_ops[j][oi][0].0;
                    if m < nm { bucket[m].push(j); }
                }
            }
        }
        for &machine in &avail {
            let mut cands: Vec<(f64, usize, u32)> = Vec::new();
            if flat {
                for &j in bucket[machine].iter() {
                    let oi = job_next[j];
                    let pt = pre.job_ops[j][oi][0].1;
                    let rem_ops = pre.job_nops[j] - oi;
                    let pr = match rule {
                        0 => rem_work[j],
                        1 => rem_ops as f64,
                        2 => -1.0,
                        3 => -(pt as f64),
                        _ => pt as f64,
                    };
                    cands.push((pr, j, pt));
                }
            }
            for j in 0..(if flat { 0 } else { n }) {
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

/// Kacem-style construction. Ops are ordered by their minimum eligible
/// processing time, then assigned to the machine minimizing `charge[k] + p`.
/// Charge ties are broken by the machine with the smaller remaining
/// eligibility count (a scarcity proxy, decremented as ops are assigned),
/// then by the smaller processing time, instead of first-in-list order.
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
    // Scarcity proxy: the number of operations each machine is eligible for.
    // Decremented as ops are assigned, so it tracks how many still-unassigned
    // operations the machine can serve. Used only to break charge ties.
    let mut machine_flex = vec![0u32; nm];
    for j in 0..pre.num_jobs {
        for oi in 0..pre.job_nops[j] {
            for &(k, _) in &pre.job_ops[j][oi] {
                machine_flex[k] += 1;
            }
        }
    }
    let jitter = rng.is_some();
    let mut rng = rng;
    for &(_, o) in &order {
        let j = pre.op_job[o]; let oi = pre.op_idx[o];
        let elig = &pre.job_ops[j][oi];
        // Running mean charge, recomputed once per operation from the live
        // `charge` vector. It is the reference the load-balance lookahead below
        // measures a machine's overshoot against.
        let charge_sum: u64 = charge.iter().sum();
        let mean_charge = charge_sum / (nm as u64).max(1);
        let mut best_k = elig[0].0;
        let mut best_p = elig[0].1;
        let mut best_c = u64::MAX;
        for &(k, p) in elig {
            // Post-assignment load-balance lookahead. `base` is the machine
            // charge after this operation is placed; a machine whose current
            // charge already sits above the running mean pays an extra penalty
            // proportional to that overshoot, so once a machine starts to run
            // away the constructor steers the next flexible operation to a less
            // loaded machine. Machines at or below the mean keep the pure
            // charge objective. All arithmetic saturates so the accumulator
            // cannot wrap. The scarcity tie-break is unchanged, so this only
            // reorders the primary charge objective.
            let base = charge[k].saturating_add(p as u64);
            let overload = charge[k].saturating_sub(mean_charge);
            let mut c = base.saturating_mul(4).saturating_add(overload);
            if jitter { c = c.saturating_add(((rng.as_mut().unwrap().gen::<u8>() as u64) & 3) * 4); }
            if (c, machine_flex[k], p) < (best_c, machine_flex[best_k], best_p) {
                best_c = c; best_k = k; best_p = p;
            }
        }
        op_machine[o] = best_k;
        charge[best_k] += best_p as u64;
        for &(k, _) in elig {
            if machine_flex[k] > 0 { machine_flex[k] -= 1; }
        }
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
    hend: Vec<u32>,
    qend: Vec<u32>,
    topo: Vec<usize>,
    mp: Vec<usize>,
    ms: Vec<usize>,
    mpos: Vec<usize>,
    indeg0: Vec<u8>,
    dur_pos: bool,
    js: Vec<u32>,
    nd: Vec<[u32; 8]>,
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
            indeg0: vec![0; n], dur_pos,
            js: (0..n).map(|oid| {
                let j = pre.op_job[oid];
                if pre.op_idx[oid] + 1 < pre.job_nops[j] { (oid + 1) as u32 } else { u32::MAX }
            }).collect(),
            nd: vec![[0u32; 8]; n],
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


    #[inline(always)]
    fn hd(&self, o: usize) -> u32 { unsafe { *self.heads.get_unchecked(o) } }
    #[inline(always)]
    fn he(&self, o: usize) -> u32 { unsafe { *self.hend.get_unchecked(o) } }
    #[inline(always)]
    fn qe(&self, o: usize) -> u32 { unsafe { *self.qend.get_unchecked(o) } }
    #[inline(always)]
    fn du(&self, o: usize) -> u32 { unsafe { *self.op_dur.get_unchecked(o) } }
    #[inline(always)]
    fn oidx(&self, o: usize) -> usize { unsafe { *self.pre.op_idx.get_unchecked(o) } }
    #[inline(always)]
    fn ojob(&self, o: usize) -> usize { unsafe { *self.pre.op_job.get_unchecked(o) } }
    #[inline(always)]
    fn jnops(&self, j: usize) -> usize { unsafe { *self.pre.job_nops.get_unchecked(j) } }

    fn eval(&mut self) -> Option<u32> {
        let n = self.pre.n_ops;
        const NONE: u32 = u32::MAX;
        {
            let nd = &mut self.nd;
            for o in 0..n {
                let ms = self.ms[o];
                nd[o] = [0, self.op_dur[o], self.js[o], if ms == usize::MAX { NONE } else { ms as u32 }, self.indeg0[o] as u32, 0, 0, 0];
            }
        }
        let mut tl = 0usize;
        for &oid in self.pre.job_off.iter() {
            if self.nd[oid][4] == 0 { self.topo[tl] = oid; tl += 1; }
        }
        let mut qi = 0;
        let mut makespan = 0u32;
        while qi < tl {
            let u = self.topo[qi]; qi += 1;
            let ru = self.nd[u];
            let end_u = ru[0] + ru[1];
            self.nd[u][6] = end_u;
            if end_u > makespan { makespan = end_u; }
            let js = ru[2];
            if js != NONE {
                let js = js as usize;
                let r = &mut self.nd[js];
                if end_u > r[0] { r[0] = end_u; }
                r[4] -= 1;
                if r[4] == 0 { self.topo[tl] = js; tl += 1; }
            }
            let mu = ru[3];
            if mu != NONE {
                let mu = mu as usize;
                let r = &mut self.nd[mu];
                if end_u > r[0] { r[0] = end_u; }
                r[4] -= 1;
                if r[4] == 0 { self.topo[tl] = mu; tl += 1; }
            }
        }
        if tl != n { return None; }
        for idx in (0..tl).rev() {
            let u = self.topo[idx];
            let ru = self.nd[u];
            let mut t = 0u32;
            if ru[2] != NONE { t = t.max(self.nd[ru[2] as usize][5]); }
            if ru[3] != NONE { t = t.max(self.nd[ru[3] as usize][5]); }
            self.nd[u][5] = t + ru[1];
        }
        for o in 0..n {
            let r = self.nd[o];
            self.heads[o] = r[0];
            self.hend[o] = r[6];
            self.qend[o] = r[5];
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

/// O(1) lower bound on the makespan after swapping the adjacent machine
/// operations `seq[p]` (a) and `seq[p + 1]` (b). The bound is the longest path
/// through a or b in the swapped graph, assembled from fixed boundary heads
/// and tails. Second-order neighbours are folded in as extra valid path
/// contributions: `pm2_comp`/`sm2_tail` are the completion/tail of the second
/// machine predecessor/successor of the pair, and `jp2_comp`/`js2_tail` are the
/// completion/tail of the second job predecessor/successor of the swapped
/// operations. Each term is a genuine path length in the swapped graph, so it
/// can never exceed the true new makespan; the extra max terms can therefore
/// only tighten the bound on the critical-block swaps that tabu_run actually
/// selects. Every index is length-checked and no graph evaluation happens
/// inside the estimate.
fn swap_estimate(g: &Graph, m: usize, p: usize) -> u32 {
    let seq = &g.machine_seq[m];
    let a = seq[p];
    let b = seq[p + 1];
    let comp = |o: usize| g.he(o);
    let tailc = |o: usize| g.qe(o);
    let jp_comp = |o: usize| -> u32 { if g.oidx(o) > 0 { comp(o - 1) } else { 0 } };
    let jp2_comp = |o: usize| -> u32 { if g.oidx(o) > 1 { comp(o - 2) } else { 0 } };
    let js_tail = |o: usize| -> u32 {
        let j = g.ojob(o);
        if g.oidx(o) + 1 < g.jnops(j) { tailc(o + 1) } else { 0 }
    };
    let js2_tail = |o: usize| -> u32 {
        let j = g.ojob(o);
        if g.oidx(o) + 2 < g.jnops(j) { tailc(o + 2) } else { 0 }
    };
    let pm_comp = if p > 0 { comp(seq[p - 1]) } else { 0 };
    let pm2_comp = if p >= 2 { comp(seq[p - 2]) } else { 0 };
    let sm_tail = if p + 2 < seq.len() { tailc(seq[p + 2]) } else { 0 };
    let sm2_tail = if p + 3 < seq.len() { tailc(seq[p + 3]) } else { 0 };
    let h_b = jp_comp(b).max(jp2_comp(b)).max(pm_comp).max(pm2_comp);
    let h_a = jp_comp(a).max(jp2_comp(a)).max(h_b + g.du(b));
    let t_a = js_tail(a).max(js2_tail(a)).max(sm_tail).max(sm2_tail);
    let t_b = js_tail(b).max(js2_tail(b)).max(g.du(a) + t_a);
    (h_a + g.du(a) + t_a).max(h_b + g.du(b) + t_b)
}

fn insert_pos(g: &Graph, mt: usize, o: usize) -> usize {
    let ho = g.heads[o];
    let seq = &g.machine_seq[mt];
    for (i, &x) in seq.iter().enumerate() {
        if g.heads[x] > ho { return i; }
    }
    seq.len()
}

struct Snap {
    off: Vec<usize>,
    hd: Vec<u32>,
    he: Vec<u32>,
    qe: Vec<u32>,
    gap: Vec<u32>,
}

impl Snap {
    fn new(n_ops: usize, nm: usize) -> Snap {
        Snap { off: vec![0; nm + 1], hd: vec![0; n_ops], he: vec![0; n_ops], qe: vec![0; n_ops], gap: vec![0; nm] }
    }

    fn fill(&mut self, g: &Graph) {
        let mut off = 0usize;
        for k in 0..g.machine_seq.len() {
            self.off[k] = off;
            let mut a = 0u32;
            let mut lo = u32::MAX;
            for &o in g.machine_seq[k].iter() {
                let he = g.he(o);
                let qe = g.qe(o);
                unsafe {
                    *self.hd.get_unchecked_mut(off) = g.hd(o);
                    *self.he.get_unchecked_mut(off) = he;
                    *self.qe.get_unchecked_mut(off) = qe;
                }
                let s = a + qe;
                if s < lo { lo = s; }
                a = he;
                off += 1;
            }
            if a < lo { lo = a; }
            self.gap[k] = lo;
        }
        self.off[g.machine_seq.len()] = off;
    }

    #[inline(always)]
    fn machine(&self, k: usize) -> (&[u32], &[u32], &[u32]) {
        let a = self.off[k];
        let b = self.off[k + 1];
        (&self.hd[a..b], &self.he[a..b], &self.qe[a..b])
    }
}

/// Machine-insert neighborhood on the critical path. The scan window and the
/// lower-bound arithmetic are unchanged; only the aspiration test is now
/// per-machine. A tabu insertion is admitted when its own bound undercuts the
/// record of the target machine `k` (`aspire[k]`) rather than a single
/// run-wide record, so a machine that already yielded a very low estimate no
/// longer blocks aspiration on a different, still-untried machine.
fn best_machine_move_mg(
    g: &Graph, snap: &Snap, path: &[usize], best_real: u32,
    re_exp: &[usize], it: usize,
    tabu_hi: &[usize], sweep_cap: usize, aspire: &[u32],
) -> Option<(usize, usize, usize, u32)> {
    let mut best: Option<(usize, usize, usize, u32)> = None;
    // Congestion keys of the destination machine that owns the current best
    // insertion. Consulted only when a candidate ties the incumbent bound `l`
    // exactly. `best_dst_gap` is the destination machine's snapshot insertion
    // gap (`Snap::gap[k]`, the smallest head-plus-tail length any insertion can
    // add on that machine); a small gap means the machine has internal slack to
    // absorb the inserted operation with little downstream displacement, so it
    // is the primary congestion key. `best_dst_ops` is that machine's operation
    // count and only breaks a further tie. A strictly better bound still wins.
    let mut best_dst_gap = u32::MAX;
    let mut best_dst_ops = usize::MAX;
    let mut seen = 0usize;
    for &v in path {
        let j = g.pre.op_job[v];
        let oi = g.pre.op_idx[v];
        let elig = &g.pre.job_ops[j][oi];
        if elig.len() < 2 { continue; }
        seen += 1;
        if seen > sweep_cap { break; }
        let m_cur = g.op_machine[v];
        let rpj = if oi > 0 { g.he(v - 1) } else { 0 };
        let qsj = if oi + 1 < g.jnops(j) { g.qe(v + 1) } else { 0 };
        let is_tabu = if tabu_hi[v] <= it { false } else { re_exp[v] > it };
        debug_assert_eq!(is_tabu, re_exp[v] > it);
        for &(k, pk) in elig {
            if k == m_cur { continue; }
            if let Some((_, _, _, bl)) = best { if rpj + pk + qsj >= bl || pk + snap.gap[k] >= bl { continue; } }
            let (_, he_s, qe_s) = snap.machine(k);
            let len = he_s.len();
            let left = he_s.partition_point(|&h| h <= rpj);
            let right = qe_s.partition_point(|&q| q > qsj);
            if left > right { continue; }
            let dst_gap = snap.gap[k];
            let dst_ops = snap.off[k + 1] - snap.off[k];
            for pos in left..=right {
                let l = if pos > left && pos < right {
                    pk + unsafe { *he_s.get_unchecked(pos - 1) + *qe_s.get_unchecked(pos) }
                } else {
                    let hv = if pos > 0 { unsafe { *he_s.get_unchecked(pos - 1) }.max(rpj) } else { rpj };
                    let tv = if pos < len { unsafe { *qe_s.get_unchecked(pos) }.max(qsj) } else { qsj };
                    hv + pk + tv
                };
                let allowed = !is_tabu || l < best_real || l < aspire[k];
                if allowed {
                    let better = match best {
                        None => true,
                        Some((_, _, _, bl)) => l < bl || (l == bl
                            && (dst_gap < best_dst_gap
                                || (dst_gap == best_dst_gap && dst_ops < best_dst_ops))),
                    };
                    if better {
                        best = Some((v, k, pos, l));
                        best_dst_gap = dst_gap;
                        best_dst_ops = dst_ops;
                    }
                }
            }
        }
    }
    best
}

/// Cross-machine swap neighborhood on the critical path. Same scan window and
/// bounds as before; only the aspiration test is now per-machine. A tabu swap
/// is admitted when its own bound undercuts the record of the target machine
/// `k` (`aspire[k]`, the machine op `v` is inserted into) instead of a single
/// run-wide record.
fn best_machine_swap_mg(
    g: &Graph, snap: &Snap, path: &[usize], best_real: u32,
    re_exp: &[usize], it: usize,
    tabu_hi: &[usize], sweep_cap: usize, aspire: &[u32],
) -> Option<(usize, usize, u32)> {
    let mut best: Option<(usize, usize, u32)> = None;
    let mut seen = 0usize;
    for &v in path {
        let jv = g.pre.op_job[v];
        let oiv = g.pre.op_idx[v];
        let elig_v = &g.pre.job_ops[jv][oiv];
        if elig_v.len() < 2 { continue; }
        seen += 1;
        if seen > sweep_cap { break; }
        let mv = g.op_machine[v];
        debug_assert_eq!(Some(g.mpos[v]), g.machine_seq[mv].iter().position(|&x| x == v));
        let pv = g.mpos[v];
        let rpv = if oiv > 0 { g.he(v - 1) } else { 0 };
        let qsv = if oiv + 1 < g.jnops(jv) { g.qe(v + 1) } else { 0 };
        let v_tabu = if tabu_hi[v] <= it { false } else { re_exp[v] > it };
        debug_assert_eq!(v_tabu, re_exp[v] > it);
        let (_, he_mv, qe_mv) = snap.machine(mv);
        let hu_base = if pv > 0 { he_mv[pv - 1] } else { 0 };
        let tu_base = if pv + 1 < he_mv.len() { qe_mv[pv + 1] } else { 0 };
        for &(k, pvk) in elig_v {
            if k == mv { continue; }
            if let Some((_, _, bl)) = best { if rpv + pvk + qsv >= bl { continue; } }
            let seq = &g.machine_seq[k];
            let len = seq.len();
            if len == 0 { continue; }
            let (_, he_s, qe_s) = snap.machine(k);
            let left = he_s.partition_point(|&h| h <= rpv);
            let right = qe_s.partition_point(|&q| q > qsv);
            if left > right { continue; }
            let hi = if right < len { right } else { len - 1 };
            for pos in left..=hi {
                let u = seq[pos];
                let hv = if pos > 0 { let e = he_s[pos - 1]; if e > rpv { e } else { rpv } } else { rpv };
                let tv = if pos + 1 < len { let q = qe_s[pos + 1]; if q > qsv { q } else { qsv } } else { qsv };
                let l_v = hv + pvk + tv;
                if let Some((_, _, bl)) = best { if l_v >= bl { continue; } }
                let pumv = g.pre.dur_on(u, mv);
                if pumv == u32::MAX { continue; }
                if let Some((_, _, bl)) = best { if hu_base + pumv + tu_base >= bl { continue; } }
                let ju = g.pre.op_job[u];
                let oiu = g.pre.op_idx[u];
                let rpu = if oiu > 0 { g.he(u - 1) } else { 0 };
                let qsu = if oiu + 1 < g.jnops(ju) { g.qe(u + 1) } else { 0 };
                let hu = if hu_base > rpu { hu_base } else { rpu };
                let tu = if tu_base > qsu { tu_base } else { qsu };
                let l_u = hu + pumv + tu;
                let l = if l_v > l_u { l_v } else { l_u };
                // Tie-break on the coarse lower bound. When two cross-swap
                // candidates share the same bound l, prefer the pair whose
                // partner operation u sits on the more near-critical chain,
                // measured by its critical slack `hd(u) + qe(u)`. The primary
                // bound keeps its strict '<'; the slack test fires only on
                // exact equality, so no strictly better candidate is discarded.
                if let Some((_, bu, bl)) = best {
                    if l > bl { continue; }
                    if l == bl {
                        let bs = g.hd(bu) + g.qe(bu);
                        let s = g.hd(u) + g.qe(u);
                        if s <= bs { continue; }
                    }
                }
                let u_tabu = if tabu_hi[u] <= it { false } else { re_exp[u] > it };
                debug_assert_eq!(u_tabu, re_exp[u] > it);
                if (v_tabu || u_tabu) && l >= best_real && l >= aspire[k] { continue; }
                best = Some((v, u, l));
            }
        }
    }
    best
}

fn critical_path(g: &Graph, makespan: u32, path: &mut Vec<usize>) {
    let n = g.pre.n_ops;
    let crit = |o: usize| g.hd(o) + g.qe(o) == makespan;
    let mut start = usize::MAX;
    if g.dur_pos {
        for &o in g.pre.job_off.iter() {
            if crit(o) && g.hd(o) == 0 { start = o; break; }
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
        let end = g.he(cur);
        let mut next = usize::MAX;
        let msc = unsafe { *g.ms.get_unchecked(cur) };
        if msc != usize::MAX {
            let v = msc;
            if crit(v) && g.hd(v) == end { next = v; }
        }
        if next == usize::MAX {
            if g.oidx(cur) + 1 < g.jnops(g.ojob(cur)) {
                let v = cur + 1;
                if crit(v) && g.hd(v) == end { next = v; }
            }
        }
        if next == usize::MAX { break; }
        cur = next;
    }
}
/// Critical-block adjacent-swap candidates for the full (deep) neighbourhood.
/// For every maximal critical block on every machine the adjacent pairs that
/// the swap ranker (`swap_estimate`) may examine are emitted. Both endpoint
/// pairs are always kept; a block with more than six adjacent pairs is capped
/// at the first three and the last three pairs, so a long block can no longer
/// hide a better interior pair behind the old endpoint-only restriction while
/// the candidate count stays bounded by a small constant. No extra search
/// phase, evaluation, or fuel is introduced.
fn n5_moves_full(snap: &Snap, makespan: u32, moves: &mut Vec<(usize, usize)>) {
    moves.clear();
    for m in 0..snap.gap.len() {
        let (hd, he, qe) = snap.machine(m);
        let len = hd.len();
        let crit = |i: usize| hd[i] + qe[i] == makespan;
        let mut pos = 0usize;
        while pos < len {
            if !crit(pos) { pos += 1; continue; }
            let run_start = pos;
            let mut run_end = pos;
            while run_end + 1 < len {
                if crit(run_end + 1) && hd[run_end + 1] == he[run_end] {
                    run_end += 1;
                } else {
                    break;
                }
            }
            let block_len = run_end - run_start + 1;
            if block_len >= 2 {
                let npairs = block_len - 1;
                if npairs <= 6 {
                    for i in 0..npairs {
                        moves.push((m, run_start + i));
                    }
                } else {
                    for i in 0..3 {
                        moves.push((m, run_start + i));
                    }
                    for i in 0..3 {
                        moves.push((m, run_end - 3 + i));
                    }
                }
            }
            pos = run_end + 1;
        }
    }
}

fn n5_moves(g: &Graph, path: &[usize], moves: &mut Vec<(usize, usize)>, blk: &mut Vec<(u32, u32)>) {
    blk.clear();
    moves.clear();
    let mut s0 = 0usize;
    for i in 1..path.len() {
        let o = unsafe { *path.get_unchecked(i) };
        let prev = unsafe { *path.get_unchecked(i - 1) };
        if !(unsafe { *g.op_machine.get_unchecked(o) == *g.op_machine.get_unchecked(prev) } && unsafe { *g.mpos.get_unchecked(o) == *g.mpos.get_unchecked(prev) + 1 }) {
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

type HGraph = hybrid_engine::graph::Graph;
type HModel = hybrid_engine::model::Model;

fn mirror_load(hg: &mut HGraph, hm: &HModel, g: &Graph) -> bool {
    let mut sched: Vec<Vec<(usize, u32)>> = g.pre.job_nops.iter().map(|&k| vec![(0usize, 0u32); k]).collect();
    for m in 0..g.machine_seq.len() {
        for (i, &o) in g.machine_seq[m].iter().enumerate() {
            let j = g.pre.op_job[o];
            let k = g.pre.op_idx[o];
            sched[j][k] = (m, i as u32);
        }
    }
    hg.load_schedule(hm, &sched)
}

#[inline]
fn mirror_eval(g: &mut Graph, hg: &mut HGraph) -> Option<u32> {
    let mk = hg.evaluate_tracked()?;
    hg.compute_tails_tracked();
    if hg.chg_full {
        let n = g.pre.n_ops;
        for o in 0..n {
            let h = hg.head[o];
            g.heads[o] = h;
            g.hend[o] = h + hg.dur[o];
            g.qend[o] = hg.qend[o];
        }
    } else {
        for i in 0..hg.chg_n {
            let o = hg.chg[i] as usize;
            let h = hg.head[o];
            g.heads[o] = h;
            g.hend[o] = h + hg.dur[o];
            g.qend[o] = hg.qend[o];
        }
    }
    hg.chg_n = 0;
    hg.chg_full = false;
    Some(mk)
}

fn tabu_run(
    g: &mut Graph, max_iter: usize, tenure: usize,
    full: bool, max_kicks: usize, fuel_floor: u64, rng: &mut SmallRng,
    hm: &HModel, use_gap: bool, abandon_above: u32, seed: &[u8; 32],
    bottleneck_m: usize,
) -> Option<u32> {
    let mut cur_real = g.eval()?;
    let mut hg = HGraph::new(hm);
    let mut inc = mirror_load(&mut hg, hm, g);
    if inc {
        match mirror_eval(g, &mut hg) { Some(mk) => { debug_assert_eq!(mk, cur_real); cur_real = mk; } None => { inc = false; } }
        if !inc { cur_real = g.eval()?; }
    }
    let mut best_om: Vec<usize> = g.op_machine.clone();
    let mut best_ms: Vec<Vec<usize>> = g.machine_seq.clone();
    let mut best_real = cur_real;
    let mut re_exp: Vec<usize> = vec![0; g.pre.n_ops];
    let mut swap_tabu: HashMap<(usize, usize), usize> =
        HashMap::with_hasher(seeded_hasher(seed));
    let mut tabu_hi: Vec<usize> = vec![0; g.pre.n_ops];
    let mut no_improve = 0usize;
    let mut kick_failed = false;
    // Per-machine aspiration record: `best_est_machine[k]` is the smallest move
    // estimate observed so far in this run for a move that acts on machine k.
    // A tabu move whose own estimate strictly undercuts the record of the
    // machine it acts on is admitted even while its tabu tenure is active
    // (classical per-machine aspiration criterion). This is finer grained than
    // a single run-wide record. Every entry starts at INF so iteration 0, whose
    // tabu lists are empty, is unaffected. No new candidates are generated and
    // no extra evaluation is performed; the only decision that changes is which
    // legal tabu move is admitted.
    let mut best_est_machine: Vec<u32> = vec![INF; g.pre.num_machines];
    let max_no_improve = if max_kicks > 0 { patience(g.pre) } else { 8 * tenure + 400 };
    let sweep_cap = sweep_cap(g.pre);
    let mut kicks_left = max_kicks;
    let n_ops = g.pre.n_ops;
    let mut snap = Snap::new(n_ops, g.machine_seq.len());
    let mut path: Vec<usize> = Vec::with_capacity(n_ops);
    let mut moves: Vec<(usize, usize)> = Vec::with_capacity(n_ops);
    let mut blk: Vec<(u32, u32)> = Vec::with_capacity(n_ops);

    for it in 0..max_iter {
        // Continuous stagnation abandon. A run whose incumbent is still above
        // the abandon bound and has stopped improving is handed back to
        // solve_pool so the kick and refresh rounds run inside the same total
        // envelope. The window scales with how far the incumbent sits above the
        // bound: a run more than ~2% over is very unlikely to catch up and is
        // cut after 1_500 stalled iterations, one more than ~1% over after
        // 3_000, and a run nearer the bound keeps the original 5_000 window.
        // The test runs every iteration (no modulo gate), so a qualifying run
        // fires exactly at its window instead of up to 2_000 iterations late.
        // This only re-schedules fuel already granted to this tabu_run; it does
        // not add fuel, rounds, or searches.
        if use_gap && it > 0 && best_real > abandon_above {
            let over = best_real - abandon_above;
            let need = if over > abandon_above / 50 {
                1_500
            } else if over > abandon_above / 100 {
                3_000
            } else {
                5_000
            };
            if no_improve >= need {
                break;
            }
        }
        if fuel_left() <= fuel_floor { break; }
        snap.fill(g);
        critical_path(g, cur_real, &mut path);
        if path.len() < 2 { break; }
        if full { n5_moves_full(&snap, cur_real, &mut moves) } else { n5_moves(g, &path, &mut moves, &mut blk) };
        if moves.is_empty() { break; }
        let mut best_swap: Option<(usize, usize)> = None;
        let mut best_est = INF;
        let mut best_tie = 0u32;
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
                          else { swap_tabu.get(&pair).map_or(false, |&exp| exp > it) };
            debug_assert_eq!(is_tabu, swap_tabu.get(&pair).map_or(false, |&exp| exp > it));
            // Tie-break on the adjacent-swap estimate. When two critical-block
            // swaps share the same lower bound, prefer the pair whose critical
            // operations carry the larger downstream tail, because shortening a
            // path next to a long chain has more leverage on the makespan. The
            // strict '<' on est is preserved; the tail test fires only on exact
            // equality, so the bound and the tabu admission test are unchanged.
            let tie = g.qe(a).max(g.qe(b));
            if (!is_tabu || est < best_real || est < best_est_machine[m])
                && (est < best_est || (est == best_est && tie > best_tie))
            {
                best_est = est;
                best_tie = tie;
                best_swap = Some((m, p));
                best_pair = pair;
            }
            if est < fb_est {
                fb_est = est;
                fb_swap = Some((m, p));
                fb_pair = pair;
            }
        }
        let mm = best_machine_move_mg(g, &snap, &path, best_real, &re_exp, it, &tabu_hi, sweep_cap, &best_est_machine);
        let mm_l = mm.map_or(INF, |(_, _, _, l)| l);
        let xs = if best_est >= cur_real && mm_l >= cur_real {
            best_machine_swap_mg(g, &snap, &path, best_real, &re_exp, it, &tabu_hi, sweep_cap, &best_est_machine)
        } else { None };
        let xs_l = xs.map_or(INF, |(_, _, l)| l);
        if best_swap.is_none() && mm.is_none() && xs.is_none() {
            if let Some((m, p)) = fb_swap {
                best_swap = Some((m, p));
                best_est = fb_est;
                best_pair = fb_pair;
            } else {
                break;
            }
        }
        // Fold this iteration's move estimates into the per-machine aspiration
        // records. Each estimate is charged to the machine the move acts on: an
        // adjacent swap acts on its own machine m, an insert acts on the target
        // machine k, and a cross-swap acts on the target machine of v (the
        // machine u currently sits on). Missing candidates are inert here. Done
        // after fallback selection so a forced fallback move still contributes
        // its estimate. Only the boolean gating on is_tabu uses these.
        if let Some((m, _)) = best_swap {
            if best_est < best_est_machine[m] { best_est_machine[m] = best_est; }
        }
        if let Some((_, k, _, l)) = mm {
            if l < best_est_machine[k] { best_est_machine[k] = l; }
        }
        if let Some((_, u, l)) = xs {
            let k = g.op_machine[u];
            if l < best_est_machine[k] { best_est_machine[k] = l; }
        }
        // Randomized tabu tenure (Taillard-style robust tabu search). Drawing
        // the tenure per applied move from a band centred on the tuned base
        // value breaks the fixed-period cycling a constant tenure can lock
        // into, while keeping the average amount of tabu memory equal to the
        // base value. The draw comes from the same `SmallRng` seeded by
        // `challenge.seed`, so the whole run stays deterministic. Only which
        // legal move is treated as tabu changes.
        let t_eff = tenure * 3 / 4 + rng.gen_range(0..=tenure / 2);
        let mut mg_revert: Option<(usize, usize, usize, u32)> = None;
        let mut swap_revert: Option<(usize, usize)> = None;
        let mut xs_revert: Option<(usize, usize, usize, u32, usize, usize, usize, u32)> = None;
        let mut xs_hg: Option<((u32, usize, u32), (u32, usize, u32))> = None;
        let mut hg_old: (u32, usize, u32) = (0, 0, 0);
        if xs_l < cur_real {
            let (v, u, _) = xs.unwrap();
            let mv = g.op_machine[v];
            let k = g.op_machine[u];
            debug_assert_ne!(mv, k);
            debug_assert_eq!(Some(g.mpos[v]), g.machine_seq[mv].iter().position(|&x| x == v));
            debug_assert_eq!(Some(g.mpos[u]), g.machine_seq[k].iter().position(|&x| x == u));
            let pv = g.mpos[v];
            let pu = g.mpos[u];
            let dv_old = g.op_dur[v];
            let du_old = g.op_dur[u];
            let dv_new = g.pre.job_ops[g.pre.op_job[v]][g.pre.op_idx[v]].iter()
                .find(|&&(m2, _)| m2 == k).map(|&(_, d)| d).unwrap();
            let du_new = g.pre.job_ops[g.pre.op_job[u]][g.pre.op_idx[u]].iter()
                .find(|&&(m2, _)| m2 == mv).map(|&(_, d)| d).unwrap();
            g.machine_seq[mv].remove(pv);
            g.machine_seq[k].remove(pu);
            g.machine_seq[k].insert(pu, v);
            g.machine_seq[mv].insert(pv, u);
            g.op_machine[v] = k;
            g.op_dur[v] = dv_new;
            g.op_machine[u] = mv;
            g.op_dur[u] = du_new;
            g.repair_machine(mv);
            g.repair_machine(k);
            if inc {
                let t1 = hg.reassign_at(v, k as u32, pu, dv_new);
                let t2 = hg.reassign_at(u, mv as u32, pv, du_new);
                xs_hg = Some((t1, t2));
            }
            re_exp[v] = it + t_eff;
            re_exp[u] = it + t_eff;
            if it + t_eff > tabu_hi[v] { tabu_hi[v] = it + t_eff; }
            if it + t_eff > tabu_hi[u] { tabu_hi[u] = it + t_eff; }
            xs_revert = Some((v, mv, pv, dv_old, u, k, pu, du_old));
        } else if mm_l < best_est {
            let (v, k, pos, _) = mm.unwrap();
            let m_old = g.op_machine[v];
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
            if inc { hg_old = hg.reassign_at(v, k as u32, pos, dt); }
            re_exp[v] = it + t_eff;
            if it + t_eff > tabu_hi[v] { tabu_hi[v] = it + t_eff; }
            mg_revert = Some((v, m_old, p_old, old_dur));
        } else if let Some((m, p)) = best_swap {
            g.machine_seq[m].swap(p, p + 1);
            g.repair_machine(m);
            if inc { hg.swap_adjacent(best_pair.0 as u32, best_pair.1 as u32); }
            swap_tabu.insert(best_pair, it + t_eff);
            if it + t_eff > tabu_hi[best_pair.0] { tabu_hi[best_pair.0] = it + t_eff; }
            if it + t_eff > tabu_hi[best_pair.1] { tabu_hi[best_pair.1] = it + t_eff; }
            swap_revert = Some((m, p));
        } else {
            break;
        }
        let ev = if inc { mirror_eval(g, &mut hg) } else { g.eval() };
        if inc {
            if let Some((v, _, _, _)) = mg_revert { g.hend[v] = g.heads[v] + g.op_dur[v]; }
            if let Some((v, _, _, _, u, _, _, _)) = xs_revert {
                g.hend[v] = g.heads[v] + g.op_dur[v];
                g.hend[u] = g.heads[u] + g.op_dur[u];
            }
        }
        cur_real = match ev {
            Some(r) => r,
            None => {
                if let Some((v, mv, pv, dv_old, u, k, pu, du_old)) = xs_revert {
                    debug_assert_eq!(g.machine_seq[mv][pv], u);
                    debug_assert_eq!(g.machine_seq[k][pu], v);
                    g.machine_seq[mv].remove(pv);
                    g.machine_seq[k].remove(pu);
                    g.machine_seq[k].insert(pu, u);
                    g.machine_seq[mv].insert(pv, v);
                    g.op_machine[v] = mv;
                    g.op_dur[v] = dv_old;
                    g.op_machine[u] = k;
                    g.op_dur[u] = du_old;
                    g.repair_machine(mv);
                    g.repair_machine(k);
                    if inc { if let Some((t1, t2)) = xs_hg { hg.undo_reassign(u, t2); hg.undo_reassign(v, t1); } }
                } else if let Some((v, m_old, p_old, old_dur)) = mg_revert {
                    let k = g.op_machine[v];
                    debug_assert_eq!(Some(g.mpos[v]), g.machine_seq[k].iter().position(|&x| x == v));
                    let pn = g.mpos[v];
                    g.machine_seq[k].remove(pn);
                    g.op_machine[v] = m_old;
                    g.op_dur[v] = old_dur;
                    g.machine_seq[m_old].insert(p_old, v);
                    g.repair_machine(k);
                    g.repair_machine(m_old);
                    if inc { hg.undo_reassign(v, hg_old); }
                } else if let Some((m, p)) = swap_revert {
                    let a = g.machine_seq[m][p]; let b = g.machine_seq[m][p + 1];
                    g.machine_seq[m].swap(p, p + 1);
                    g.repair_machine(m);
                    if inc { hg.swap_adjacent(a as u32, b as u32); }
                }
                let ev2 = if inc { mirror_eval(g, &mut hg) } else { g.eval() };
                if inc {
                    if let Some((v, _, _, _)) = mg_revert { g.hend[v] = g.heads[v] + g.op_dur[v]; }
                    if let Some((v, _, _, _, u, _, _, _)) = xs_revert {
                        g.hend[v] = g.heads[v] + g.op_dur[v];
                        g.hend[u] = g.heads[u] + g.op_dur[u];
                    }
                }
                match ev2 { Some(r) => r, None => break }
            }
        };
        if cur_real < best_real {
            best_real = cur_real;
            best_om.copy_from_slice(&g.op_machine);
            for (dst, src) in best_ms.iter_mut().zip(g.machine_seq.iter()) {
                dst.clear();
                dst.extend_from_slice(src);
            }
            // Age the per-machine aspiration records toward the new incumbent.
            // A record left over from an earlier region can sit far from the
            // current incumbent and gate the aspiration test (l < aspire[k]) in
            // a way that no longer reflects the region being searched. On every
            // incumbent improvement each finite record is pulled halfway toward
            // `best_real`; `INF` entries stay `INF`. This is O(num_machines) on
            // an improvement only, adds no candidate and no evaluation, and
            // leaves every bound, tenure and budget unchanged.
            for e in best_est_machine.iter_mut() {
                if *e != INF {
                    if *e < best_real {
                        *e += (best_real - *e) / 2;
                    } else {
                        *e -= (*e - best_real) / 2;
                    }
                }
            }
            no_improve = 0;
            kick_failed = false;
        } else {
            no_improve += 1;
            if no_improve >= max_no_improve {
                if kicks_left == 0 || kick_failed { break; }
                g.reset(&best_om, &best_ms);
                kick_critical(g, 3 + (max_kicks - kicks_left), rng, bottleneck_m);
                if inc { inc = mirror_load(&mut hg, hm, g); }
                cur_real = match if inc { mirror_eval(g, &mut hg) } else { g.eval() } { Some(r) => r, None => break };
                re_exp.fill(0);
                swap_tabu.clear();
                for h in tabu_hi.iter_mut() { *h = 0; }
                // Post-kick aspiration reset. A kick jumps to a new region, so
                // the per-machine aspiration records charged with pre-kick move
                // estimates are no longer comparable to the new region's
                // estimates. Clear them to INF alongside the other tabu
                // structures so aspiration reopens on the first post-kick
                // iteration. No bound, candidate, tenure, or budget changes, and
                // `best_real` / `best_om` / `best_ms` survive the kick.
                for e in best_est_machine.iter_mut() { *e = INF; }
                no_improve = 0;
                kicks_left -= 1;
                kick_failed = true;
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

/// Critical-path kick. Candidate adjacent-swap pairs are collected from every
/// operation on the current critical path and deduplicated. The deduplicated
/// list is then re-weighted: a pair whose machine is `bottleneck_m` is pushed
/// twice, so uniform sampling draws it twice as often and the perturbation
/// tends to reshuffle the machine most likely to be stretching the makespan.
/// Strength and the attempt cap are unchanged; on a balanced instance the
/// duplicate has little effect because the critical path is spread out.
fn kick_critical(g: &mut Graph, strength: usize, rng: &mut SmallRng, bottleneck_m: usize) {
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
    let mut weighted: Vec<(usize, usize)> = Vec::with_capacity(cands.len() * 2);
    for &c in &cands {
        weighted.push(c);
        if c.0 == bottleneck_m { weighted.push(c); }
    }
    let mut applied = 0usize;
    let mut attempts = 0usize;
    while applied < strength && attempts < strength * 5 {
        attempts += 1;
        let (m, pos) = weighted[rng.gen_range(0..weighted.len())];
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
    /// Index of the single machine carrying the largest initial load estimate
    /// (`argmax` of `machine_load0`). Read-only after construction.
    bottleneck_m: usize,
    /// `machine_load0[bottleneck_m]` divided by the mean load, clamped to at
    /// least 1.0. Near one on a balanced instance and grows on a congested one,
    /// so it can safely scale a penalty without disturbing the balanced case.
    bottleneck_share: f64,
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
        // Bottleneck machine: the argmax of the initial load estimate
        // `machine_load0` (each machine's summed `op_min / flex` contribution).
        // The share is that load over the mean load, clamped to at least 1.0 so
        // a balanced instance keeps a neutral multiplier. Both are derived only
        // from instance data already computed in this pass.
        let mut bottleneck_m = 0usize;
        let mut bottleneck_load = f64::NEG_INFINITY;
        for m in 0..nm {
            if machine_load0[m] > bottleneck_load {
                bottleneck_load = machine_load0[m];
                bottleneck_m = m;
            }
        }
        let bottleneck_share = (bottleneck_load / avg_machine_load).clamp(1.0, 4.0);
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
        let jobshopness = if chaotic_like { 0.85 } else { 0.55 };
        let time_scale = (horizon * (2.65 + 0.10 * jobshopness + 0.10 * high_flex)).max(1.0);
        Der {
            op_min, suf_min, machine_load0, machine_scarcity, machine_best_pop,
            bottleneck_m, bottleneck_share,
            avg_machine_load, avg_machine_scarcity, avg_op_min, horizon, time_scale,
            flex_avg, flex_factor, high_flex, jobshopness, chaotic_like, use_rich,
        }
    }
}

/// Best / runner-up completion pair over the eligible machines of one
/// operation, together with the number of machines that reach the best
/// completion. `best` is the smallest achievable completion over the eligible
/// machines; the runner-up is the second-best *machine*, whose identity is
/// carried through the scan and whose completion is rebuilt from that machine's
/// availability and processing time, so the regret charges the operation for
/// the machine it would actually be pushed to. When every eligible machine ties
/// the best value the runner-up collapses to `best` and the regret is zero;
/// with a single eligible machine there is no runner-up and `second` stays
/// `INF`, which the caller reads as "no alternative". The eligible-machine loop
/// keeps the same length and no candidate is added or removed.
#[inline]
fn best_second_counts(time: u32, avail: &[u32], elig: &[(usize, u32)]) -> (u32, u32, usize) {
    let mut best = INF;
    let mut best_m = usize::MAX;
    let mut best_pt = 0u32;
    let mut second = INF;
    let mut second_m = usize::MAX;
    let mut second_pt = 0u32;
    let mut cnt = 0usize;
    for &(m, pt) in elig {
        let end = time.max(avail[m]).saturating_add(pt);
        if end < best {
            second = best;
            second_m = best_m;
            second_pt = best_pt;
            best = end;
            best_m = m;
            best_pt = pt;
            cnt = 1;
        } else if end == best {
            cnt += 1;
        } else if end < second {
            second = end;
            second_m = m;
            second_pt = pt;
        }
    }
    if second_m != usize::MAX {
        second = time.max(avail[second_m]).saturating_add(second_pt);
    } else if cnt > 1 {
        second = best;
    }
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
        // Bottleneck machine: the machine carrying the largest running assigned
        // work, recomputed from the live `work` array once per scheduling round
        // so the penalty tracks construction as it progresses. The index stays
        // inactive (`usize::MAX`) while no machine stands out -- the maximum
        // must exceed the mean by at least 1/8 -- and while no work has been
        // assigned at all, so a balanced start behaves exactly as before.
        let mut bottleneck = usize::MAX;
        {
            let mut bwork = 0u64;
            let mut total = 0u64;
            for k in 0..nm {
                total += work[k];
                if work[k] > bwork { bwork = work[k]; bottleneck = k; }
            }
            if nm == 0 || total == 0 || bwork * (nm as u64) < total + total / 8 {
                bottleneck = usize::MAX;
            }
        }
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
                    // Amplify the popular-machine penalty exactly on the single
                    // bottleneck machine, so stacking flexible work on the
                    // already congested machine is discouraged more strongly.
                    // On a balanced instance `bottleneck_share` is 1.0 and the
                    // term is identical to before.
                    let amp = if m == der.bottleneck_m { der.bottleneck_share } else { 1.0 };
                    (0.07 + 0.15 * (1.0 - progress)).clamp(0.05, 0.24) * der.machine_best_pop[m] * der.flex_factor * amp
                } else { 0.0 };
                let bal_pen = if der.chaotic_like {
                    let dw = (avg_work + (der.avg_op_min * 3.0).max(1.0)).max(1.0);
                    let r = work[m] as f64 / dw;
                    -0.07 * (r / (r + 1.0)).clamp(0.0, 1.0)
                } else { 0.0 };
                // Explicit bottleneck penalty. `load_w * load_n` and `bal_pen`
                // are both diffuse: they react to average loading, not to the
                // single most congested machine that actually stretches the
                // makespan. When `m` is the current bottleneck and this
                // operation can run elsewhere (`elig.len() >= 2`), subtract a
                // small multiple of `der.flex_factor`, capped well below the
                // regret term, so flexible work is steered off the congested
                // machine. A rigid operation takes the else branch and is
                // scored exactly as before.
                let bott_pen = if m == bottleneck && elig.len() >= 2 {
                    (0.30 * der.flex_factor).min(0.55)
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
                    - bott_pen
                    + route_bonus
                    + jitter;
                cands.push((score, j, pt, oid));
            }
            if cands.is_empty() { continue; }
            let (_, j, pt, oid) = if use_rand {
                cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
                if der.chaotic_like {
                    let progress = 1.0 - remaining as f64 / (pre.n_ops as f64).max(1.0);
                    // Smooth top-k width. The previous three-step function of
                    // progress perturbed machine loading mid-construction
                    // because the exploration width changed discontinuously.
                    // A continuous power decay ramps selection pressure from
                    // wide exploration at the start to a single greedy choice
                    // near completion, removing the discontinuity without
                    // shifting the overall pressure. Rounded to a deterministic
                    // integer and clamped below.
                    let decay = (1.0 - progress).max(0.0);
                    let k_eff = 1 + ((k as f64 - 1.0) * decay.powf(1.5)).round() as usize;
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

/// Agreement-weighted consensus vote. For each operation we count how many
/// pool members assign it to each machine, then weight every member that chose
/// machine `m` by the fraction of the pool that agrees with that choice. The
/// number of members choosing `m` is itself `counts[m]`, so the accumulated
/// agreement for machine `m` on operation `oid` is exactly `counts[m]^2`; we
/// store `counts[m]^2` and normalize per operation by the sum of squares.
/// Members that agree with the crowd reinforce the consensus and outliers
/// contribute proportionally less, so the route is a true agreement vote rather
/// than a makespan-rank weighted average. The route is only consumed as a
/// construction bonus and a refresh seed, so a sharper consensus changes
/// construction decisions without adding any search.
fn build_consensus(pre: &Pre, pool: &[PoolMember]) -> Vec<Vec<f32>> {
    let nm = pre.num_machines;
    let mut route = vec![vec![0f32; nm]; pre.n_ops];
    if pool.is_empty() { return route; }
    let mut counts = vec![0u32; nm];
    for oid in 0..pre.n_ops {
        for c in counts.iter_mut() { *c = 0; }
        for e in pool.iter() { counts[e.om[oid]] += 1; }
        let mut sq = 0f32;
        for m in 0..nm {
            let c = counts[m] as f32;
            let v = c * c;
            route[oid][m] = v;
            sq += v;
        }
        if sq > 0.0 {
            let inv = 1.0f32 / sq;
            for v in route[oid].iter_mut() { *v *= inv; }
        }
    }
    route
}

/// Constructor mix for a seed index. On rich instances the randomized
/// `fjsp_construct` variants and the deterministic `kacem_construct` now
/// alternate every third index (rather than one in five), and the exploration
/// width `k` is drawn with a stride of 3 modulo 4 instead of a stride of 1, so
/// two successive seeds differ in both the constructor family and the width
/// instead of walking a fixed 2,3,4,5 ladder. Balanced instances keep the same
/// alternation over `dispatch_construct`, whose top-k width uses the same
/// stride; only the dispatch rule is still keyed off `idx % 5`. `idx == 0`
/// still yields the single greedy seed in both branches, so the first
/// construction, the A0 loop, and the number of refresh rounds are unchanged.
/// Only which seed a refresh round injects changes; the fuel budget and every
/// constructor body are untouched.
fn make_seed(pre: &Pre, der: &Der, idx: usize, rng: &mut SmallRng)
    -> (Vec<usize>, Vec<Vec<usize>>) {
    if der.use_rich {
        if idx == 0 {
            fjsp_construct(pre, der, 0, rng, None)
        } else if idx % 3 == 0 {
            kacem_construct(pre, Some(rng))
        } else {
            let k = 2 + ((idx * 3) % 4);
            fjsp_construct(pre, der, k, rng, None)
        }
    } else {
        if idx == 0 {
            dispatch_construct(pre, 0, None, 0)
        } else if idx % 3 == 0 {
            kacem_construct(pre, Some(rng))
        } else {
            let rule = (idx % 5) as u8;
            let top_k = 2 + ((idx * 3) % 4);
            dispatch_construct(pre, rule, Some(rng), top_k)
        }
    }
}

fn solution_makespan(pre: &Pre, solution: &Solution) -> Option<u32> {
    if solution.job_schedule.len() != pre.num_jobs {
        return None;
    }
    let mut machine_usage: Vec<Vec<(u32, u32)>> = vec![Vec::new(); pre.num_machines];
    let mut makespan = 0u32;
    for j in 0..pre.num_jobs {
        let schedule = &solution.job_schedule[j];
        if schedule.len() != pre.job_nops[j] {
            return None;
        }
        let mut min_start_time = 0u32;
        for (oi, &(machine, start_time)) in schedule.iter().enumerate() {
            if machine >= pre.num_machines {
                return None;
            }
            let pt = pre.job_ops[j][oi].iter().find(|&&(m, _)| m == machine)?.1;
            if start_time < min_start_time {
                return None;
            }
            let finish_time = start_time + pt;
            machine_usage[machine].push((start_time, finish_time));
            min_start_time = finish_time;
        }
        if min_start_time > makespan {
            makespan = min_start_time;
        }
    }
    for usage in &mut machine_usage {
        usage.sort_by_key(|&(start, _)| start);
        for i in 1..usage.len() {
            if usage[i].0 < usage[i - 1].1 {
                return None;
            }
        }
    }
    Some(makespan)
}

struct Best<'a> {
    mk: u32,
    om: Vec<usize>,
    ms: Vec<Vec<usize>>,
    save: &'a dyn Fn(&Solution) -> Result<()>,
}

impl Best<'_> {
    fn offer(&mut self, g: &Graph, mk: u32) -> Result<bool> {
        if mk >= self.mk { return Ok(false); }
        let s = g.to_solution();
        if solution_makespan(g.pre, &s).is_none() { return Ok(false); }
        self.mk = mk;
        (self.save)(&s)?;
        self.om.copy_from_slice(&g.op_machine);
        for (d, sm) in self.ms.iter_mut().zip(g.machine_seq.iter()) { d.clear(); d.extend_from_slice(sm); }
        Ok(true)
    }
}

fn solve_pool(
    challenge: &Challenge, pre: &Pre, der: &Der,
    save_solution: &dyn Fn(&Solution) -> Result<()>, rng: &mut SmallRng,
) -> Result<()> {
    let high = der.chaotic_like;
    let tenure = 8 + pre.num_jobs / 4;
    let cap = 16usize;
    let elites = if high { 6usize } else { 3usize };
    let min_dist = (pre.n_ops / 16 + 2).max(2);
    let polish_iters = 4000usize;
    let deep_iters = 200000usize;

    let hm = HModel::build(challenge);
    let avail_fuel = fuel_left();
    let work_cap = if high { 65_000_000_000u64 } else { 200_000_000_000u64 };
    let total0 = avail_fuel.saturating_sub(FUEL_STOP).min(work_cap);
    let floor_fuel = avail_fuel.saturating_sub(total0);

    let (om0, ms0) = make_seed(pre, der, 0, rng);
    let mut g = Graph::new(pre, om0, ms0);
    let mut pool: Vec<PoolMember> = Vec::new();
    let mut best = Best { mk: INF, om: g.op_machine.clone(), ms: g.machine_seq.clone(), save: save_solution };

    let a0_cap = 24usize;
    if high {
        let a0_fuel = (total0 * 15 / 100).max(15_000_000_000u64.min(total0 * 30 / 100));
        let a0_floor = floor_fuel + total0.saturating_sub(a0_fuel);
        let mut route: Vec<Vec<f32>> = Vec::new();
        let mut c = 0usize;
        while fuel_left() > a0_floor && c < 4000 {
            if c % 24 == 0 && pool.len() >= 3 { route = build_consensus(pre, &pool); }
            let k = 2 + (c % 5);
            let use_route = !route.is_empty() && c % 2 == 0;
            let (om, ms) = fjsp_construct(pre, der, k, rng, if use_route { Some(&route[..]) } else { None });
            g.reset(&om, &ms);
            if let Some(mk) = g.eval() {
                best.offer(&g, mk)?;
                pool_consider(&mut pool, pre, mk, &g.op_machine, &g.machine_seq, a0_cap, min_dist);
            }
            c += 1;
        }
    } else {
        let a0_floor = floor_fuel + total0 * 85 / 100;
        let mut c = 0usize;
        while fuel_left() > a0_floor && c < 4000 {
            let k = 2 + (c % 5);
            let (om, ms) = fjsp_construct(pre, der, k, rng, None);
            g.reset(&om, &ms);
            if let Some(mk) = g.eval() {
                best.offer(&g, mk)?;
                pool_consider(&mut pool, pre, mk, &g.op_machine, &g.machine_seq, a0_cap, min_dist);
            }
            c += 1;
        }
    }
    if pool.is_empty() {
        if let Some(mk) = g.eval() {
            pool_consider(&mut pool, pre, mk, &g.op_machine, &g.machine_seq, cap, min_dist);
        }
    }

    let mut refresh_idx = 0usize;
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
            let abandon_above = if high && i > 0 && best.mk != INF { best.mk + best.mk / 400 } else { INF };
            g.reset(&pool[i].om, &pool[i].ms);
            if let Some(mk) = tabu_run(&mut g, deep_iters, tenure, true, 8, floor, rng, &hm, high, abandon_above, &challenge.seed, der.bottleneck_m) {
                if mk < pool[i].mk {
                    pool[i].mk = mk;
                    pool[i].om.copy_from_slice(&g.op_machine);
                    for (d, sm) in pool[i].ms.iter_mut().zip(g.machine_seq.iter()) { d.clear(); d.extend_from_slice(sm); }
                    pool[i].deep = false;
                } else {
                    pool[i].deep = true;
                }
                if best.offer(&g, mk)? { improved = true; }
            }
        }

        if fuel_left() > floor_fuel && best.mk != INF {
            let rem = fuel_left().saturating_sub(floor_fuel);
            let slice = if high { (total0 * 6 / 100).min(rem) } else { rem / 3 };
            let floor = (fuel_left() - slice).max(floor_fuel);
            g.reset(&best.om, &best.ms);
            kick_critical(&mut g, kk, rng, der.bottleneck_m);
            if let Some(mk) = tabu_run(&mut g, deep_iters, tenure, true, 6, floor, rng, &hm, high, INF, &challenge.seed, der.bottleneck_m) {
                if best.offer(&g, mk)? { improved = true; }
                pool_consider(&mut pool, pre, mk, &g.op_machine, &g.machine_seq, cap, min_dist);
            }
        }

        if fuel_left() > floor_fuel {
            // Phase the refresh seed by fuel progress rather than parity: the
            // consensus route is only meaningful while the pool still holds
            // diverse elites, so consume it for the first half of the original
            // budget and inject fresh structural diversity from `make_seed`
            // once the run is more than half spent. `total0` and `floor_fuel`
            // are unchanged, so the number of refresh rounds, the `refresh_idx`
            // increments and the `polish_iters` budget are all identical.
            let rem_budget = fuel_left().saturating_sub(floor_fuel);
            let consumed = total0.saturating_sub(rem_budget);
            let late = consumed.saturating_mul(2) >= total0;
            let (om, ms) = if high && !pool.is_empty() && !late {
                let route = build_consensus(pre, &pool);
                fjsp_construct(pre, der, 3, rng, Some(&route))
            } else {
                make_seed(pre, der, refresh_idx, rng)
            };
            refresh_idx += 1;
            g.reset(&om, &ms);
            if let Some(mk) = tabu_run(&mut g, polish_iters, tenure, false, 0, floor_fuel, rng, &hm, high, INF, &challenge.seed, der.bottleneck_m) {
                if best.offer(&g, mk)? { improved = true; }
                pool_consider(&mut pool, pre, mk, &g.op_machine, &g.machine_seq, cap, min_dist);
            }
        }

        if improved {
            stall = 0; kk = 2;
        } else {
            stall += 1;
            if stall % 3 == 0 { kk = (kk + 1).min(8); }
        }

        if high && stall >= 3 { break; }
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

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let pre = Pre::build(challenge);
    let best_mk = std::cell::Cell::new(u32::MAX);
    let guarded = |s: &Solution| -> Result<()> {
        if let Some(mk) = solution_makespan(&pre, s) {
            if mk <= best_mk.get() {
                best_mk.set(mk);
                return save_solution(s);
            }
        }
        Ok(())
    };
    let der = Der::build(&pre);
    gate_floor::gate_floor(&pre, &challenge.seed, &guarded)?;
    let mut rng = SmallRng::from_seed(challenge.seed);
    solve_pool(challenge, &pre, &der, &guarded, &mut rng)?;
    greedy_floor(&pre, &guarded)?;
    Ok(())
}