use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use crate::{seeded_hasher, HashMap};
use super::{hybrid_engine, Der};
use tig_challenges::job_scheduling::*;

const INF: u32 = u32::MAX / 4;
const FUEL_STOP: u64 = 100_000_000;
const FJSP_WORK_CAP_HIGH: u64 = 105_000_000_000;
const FJSP_WORK_CAP_MEDIUM: u64 = 280_000_000_000;
pub(super) const FS_MED: u64 = 1200;
pub(super) const FM_CLAMP: u64 = 1;
pub(super) const FM_RESERVE: u64 = 1000;

struct LoadBias {
    machine: usize,
    share: f64,
}

impl LoadBias {
    fn build(der: &Der) -> Self {
        let mut machine = 0;
        let mut load = f64::NEG_INFINITY;
        for (m, &w) in der.machine_load0.iter().enumerate() {
            if w > load {
                load = w;
                machine = m;
            }
        }
        let share = (load / der.avg_machine_load).clamp(1.0, 4.0);
        Self { machine, share }
    }
}

#[inline(always)]
fn choose_usize(take: bool, yes: usize, no: usize) -> usize {
    let mask = 0usize.wrapping_sub(take as usize);
    (yes & mask) | (no & !mask)
}

#[inline(always)]
fn choose_u32(take: bool, yes: u32, no: u32) -> u32 {
    let mask = 0u32.wrapping_sub(take as u32);
    (yes & mask) | (no & !mask)
}

#[inline(always)]
fn choose_u64(take: bool, yes: u64, no: u64) -> u64 {
    let mask = 0u64.wrapping_sub(take as u64);
    (yes & mask) | (no & !mask)
}

#[inline(always)]
fn choose_u128(take: bool, yes: u128, no: u128) -> u128 {
    let mask = 0u128.wrapping_sub(take as u128);
    (yes & mask) | (no & !mask)
}

#[inline(always)]
fn choose_f64(take: bool, yes: f64, no: f64) -> f64 {
    f64::from_bits(choose_u64(take, yes.to_bits(), no.to_bits()))
}

#[inline(always)]
fn rank_key(estimate: u32, tie: u32) -> u64 {
    ((estimate as u64) << 32) | (!tie as u64)
}

#[inline(always)]
fn move_key(estimate: u32, gap: u32, operations: usize) -> u128 {
    ((estimate as u128) << 96) | ((gap as u128) << 64) | operations as u128
}

#[inline(always)]
fn age_aspiration(value: u32, best: u32) -> u32 {
    let midpoint = ((value as u64 + best as u64 + (value > best) as u64) / 2) as u32;
    choose_u32(value != INF, midpoint, value)
}

extern "C" {
    #[allow(non_upper_case_globals)]
    static __fuel_remaining: u64;
}
#[inline]
fn fuel_left() -> u64 {
    unsafe { core::ptr::read_volatile(&__fuel_remaining) }
}

struct Pre {
    seed: [u8; 32],
    num_jobs: usize,
    num_machines: usize,
    job_ops: Vec<Vec<Vec<(usize, u32)>>>,
    job_nops: Vec<usize>,
    n_ops: usize,
    job_off: Vec<usize>,
    op_job: Vec<usize>,
    op_idx: Vec<usize>,
    common_route: bool,
    max_flex: usize,
    op_w: Vec<f64>,
    dur_pos: bool,
    dur_tab: Vec<u32>,
    dur_first: Vec<u32>,
    elig_off: Vec<usize>,
    elig_flat: Vec<(usize, u32)>,
}

impl Pre {
    fn build(challenge: &Challenge, common_route: bool) -> Self {
        let num_jobs = challenge.num_jobs;
        let mut job_ops: Vec<Vec<Vec<(usize, u32)>>> = Vec::with_capacity(num_jobs);
        for (product, &count) in challenge.jobs_per_product.iter().enumerate() {
            let ops = &challenge.product_processing_times[product];
            for _ in 0..count {
                let mut this_job = Vec::with_capacity(ops.len());
                for op in ops.iter() {
                    let cands = super::processing_row(op, challenge.num_machines)
                        .expect("machine id out of range")
                        .into_iter()
                        .enumerate()
                        .filter_map(|(m, t)| t.map(|t| (m, t)))
                        .collect();
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
        let max_flex = job_ops
            .iter()
            .flat_map(|j| j.iter())
            .map(|o| o.len())
            .max()
            .unwrap_or(1);
        let dur_pos = job_ops
            .iter()
            .flat_map(|j| j.iter())
            .flat_map(|o| o.iter())
            .all(|&(_, t)| t >= 1);
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
                for &(m, t) in cands {
                    dur_tab[m * off + oid] = t;
                }
            }
        }
        let mut dur_first = vec![u32::MAX; off * nm];
        for (j, ops) in job_ops.iter().enumerate() {
            for (oi, cands) in ops.iter().enumerate() {
                let oid = job_off[j] + oi;
                for &(m, t) in cands.iter().rev() {
                    dur_first[oid * nm + m] = t;
                }
            }
        }
        let mut elig_off = Vec::with_capacity(off + 1);
        let mut elig_flat = Vec::new();
        for ops in &job_ops {
            for op in ops {
                elig_off.push(elig_flat.len());
                elig_flat.extend_from_slice(op);
            }
        }
        elig_off.push(elig_flat.len());
        Pre {
            seed: challenge.seed,
            num_jobs,
            num_machines: nm,
            job_ops,
            job_nops,
            n_ops: off,
            job_off,
            op_job,
            op_idx,
            common_route,
            max_flex,
            op_w,
            dur_pos,
            dur_tab,
            dur_first,
            elig_off,
            elig_flat,
        }
    }
    #[inline]
    fn op_id(&self, job: usize, oi: usize) -> usize {
        self.job_off[job] + oi
    }
    #[inline(always)]
    fn dur_first(&self, o: usize, m: usize) -> u32 {
        unsafe { *self.dur_first.get_unchecked(o * self.num_machines + m) }
    }
    #[inline(always)]
    fn durations_on(&self, m: usize) -> &[u32] {
        unsafe {
            self.dur_tab
                .get_unchecked(m * self.n_ops..(m + 1) * self.n_ops)
        }
    }
    #[inline(always)]
    fn eligible(&self, o: usize) -> &[(usize, u32)] {
        unsafe {
            let a = *self.elig_off.get_unchecked(o);
            let b = *self.elig_off.get_unchecked(o + 1);
            self.elig_flat.get_unchecked(a..b)
        }
    }
}

#[inline]
fn earliest_end(time: u32, mach_avail: &[u32], op_times: &[(usize, u32)]) -> u32 {
    let mut e = u32::MAX;
    for &(m, pt) in op_times {
        let end = time.max(mach_avail[m]) + pt;
        if end < e {
            e = end;
        }
    }
    e
}

fn dispatch_construct(
    pre: &Pre,
    rule: u8,
    rng: Option<&mut SmallRng>,
    top_k: usize,
) -> (Vec<usize>, Vec<Vec<usize>>) {
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
    let mut bucket: Vec<Vec<usize>> = if flat {
        vec![Vec::new(); nm]
    } else {
        Vec::new()
    };

    while remaining > 0 {
        let mut avail: Vec<usize> = (0..nm).filter(|&m| mach_avail[m] <= time).collect();
        if use_random {
            avail.shuffle(rng.as_mut().unwrap());
        }
        if flat {
            for b in bucket.iter_mut() {
                b.clear();
            }
            for j in 0..n {
                let oi = job_next[j];
                if oi < pre.job_nops[j] && job_ready[j] <= time {
                    let m = pre.job_ops[j][oi][0].0;
                    if m < nm {
                        bucket[m].push(j);
                    }
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
                if oi >= pre.job_nops[j] {
                    continue;
                }
                if job_ready[j] > time {
                    continue;
                }
                let op = &pre.job_ops[j][oi];
                let Some(&(_, pt)) = op.iter().find(|&&(m, _)| m == machine) else {
                    continue;
                };
                let machine_end = time.max(mach_avail[machine]) + pt;
                if machine_end != earliest_end(time, &mach_avail, op) {
                    continue;
                }
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
            if cands.is_empty() {
                continue;
            }
            let spt = pre.max_flex == 1;
            let (_, j, pt) = if use_random {
                if spt {
                    cands.sort_by(|a, b| {
                        b.0.partial_cmp(&a.0)
                            .unwrap()
                            .then(a.2.cmp(&b.2))
                            .then(a.1.cmp(&b.1))
                    });
                } else {
                    cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
                }
                let k = top_k.min(cands.len());
                cands[rng.as_mut().unwrap().gen_range(0..k)]
            } else if spt {
                *cands
                    .iter()
                    .max_by(|a, b| {
                        a.0.partial_cmp(&b.0)
                            .unwrap()
                            .then(b.2.cmp(&a.2))
                            .then(b.1.cmp(&a.1))
                    })
                    .unwrap()
            } else {
                *cands
                    .iter()
                    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(b.1.cmp(&a.1)))
                    .unwrap()
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
            if rem_work[j] < 0.0 {
                rem_work[j] = 0.0;
            }
            remaining -= 1;
        }
        if remaining == 0 {
            break;
        }
        let mut next: Option<u32> = None;
        for &t in mach_avail.iter() {
            if t > time {
                next = Some(next.map_or(t, |b| b.min(t)));
            }
        }
        for j in 0..n {
            if job_next[j] < pre.job_nops[j] && job_ready[j] > time {
                let t = job_ready[j];
                next = Some(next.map_or(t, |b| b.min(t)));
            }
        }
        match next {
            Some(t) => time = t,
            None => break,
        }
    }
    (op_machine, machine_seq)
}

fn assignment_construct(pre: &Pre, rng: Option<&mut SmallRng>) -> (Vec<usize>, Vec<Vec<usize>>) {
    let n = pre.num_jobs;
    let nm = pre.num_machines;
    let mut charge = vec![0u64; nm];
    let mut op_machine = vec![0usize; pre.n_ops];
    let mut order: Vec<(u32, usize)> = (0..pre.n_ops)
        .map(|o| {
            let j = pre.op_job[o];
            let oi = pre.op_idx[o];
            let mn = pre.job_ops[j][oi]
                .iter()
                .map(|&(_, t)| t)
                .min()
                .unwrap_or(0);
            (mn, o)
        })
        .collect();
    order.sort_unstable();
    let mut machine_flex = {
        let mut counts = vec![0u32; nm];
        for &(m, _) in &pre.elig_flat {
            counts[m] += 1;
        }
        counts
    };
    let jitter = rng.is_some();
    let mut rng = rng;
    for &(_, o) in &order {
        let j = pre.op_job[o];
        let oi = pre.op_idx[o];
        let mean_charge = charge.iter().sum::<u64>() / (nm as u64).max(1);
        let mut best_k = pre.job_ops[j][oi][0].0;
        let mut best_p = pre.job_ops[j][oi][0].1;
        let mut best_c = u64::MAX;
        for &(k, p) in &pre.job_ops[j][oi] {
            let c = {
                let mut c = charge[k]
                    .saturating_add(p as u64)
                    .saturating_mul(4)
                    .saturating_add(charge[k].saturating_sub(mean_charge));
                if jitter {
                    c = c.saturating_add(((rng.as_mut().unwrap().gen::<u8>() as u64) & 3) * 4);
                }
                c
            };
            let better = (c, machine_flex[k], p) < (best_c, machine_flex[best_k], best_p);
            if better {
                best_c = c;
                best_k = k;
                best_p = p;
            }
        }
        op_machine[o] = best_k;
        charge[best_k] += best_p as u64;
        {
            for &(k, _) in &pre.job_ops[j][oi] {
                machine_flex[k] = machine_flex[k].saturating_sub(1);
            }
        }
    }
    let mut job_next = vec![0usize; n];
    let mut job_ready = vec![0u32; n];
    let mut mach_avail = vec![0u32; nm];
    let mut machine_seq: Vec<Vec<usize>> = vec![Vec::new(); nm];
    let mut remaining: usize = pre.job_nops.iter().sum();
    let mut op_dur = vec![0u32; pre.n_ops];
    for o in 0..pre.n_ops {
        let j = pre.op_job[o];
        let oi = pre.op_idx[o];
        let m = op_machine[o];
        op_dur[o] = pre.job_ops[j][oi]
            .iter()
            .find(|&&(mm, _)| mm == m)
            .map(|&(_, t)| t)
            .unwrap();
    }
    while remaining > 0 {
        let mut best_end = u32::MAX;
        let mut best_j = usize::MAX;
        for j in 0..n {
            let oi = job_next[j];
            if oi >= pre.job_nops[j] {
                continue;
            }
            let o = pre.op_id(j, oi);
            let m = op_machine[o];
            let end = job_ready[j].max(mach_avail[m]) + op_dur[o];
            if end < best_end || (end == best_end && j < best_j) {
                best_end = end;
                best_j = j;
            }
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
            op_dur[oid] = pre.job_ops[j][o]
                .iter()
                .find(|&&(mm, _)| mm == m)
                .map(|&(_, t)| t)
                .unwrap_or(0);
        }
        let dur_pos = pre.dur_pos && op_dur.iter().all(|&d| d >= 1);
        let mut g = Graph {
            pre,
            op_machine,
            op_dur,
            machine_seq,
            heads: vec![0; n],
            hend: vec![0; n],
            qend: vec![0; n],
            topo: vec![0; n],
            mp: vec![usize::MAX; n],
            ms: vec![usize::MAX; n],
            mpos: vec![0; n],
            indeg0: vec![0; n],
            dur_pos,
            js: (0..n)
                .map(|oid| {
                    let j = pre.op_job[oid];
                    if pre.op_idx[oid] + 1 < pre.job_nops[j] {
                        (oid + 1) as u32
                    } else {
                        u32::MAX
                    }
                })
                .collect(),
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
            let d = self.pre.job_ops[j][o]
                .iter()
                .find(|&&(mm, _)| mm == m)
                .map(|&(_, t)| t)
                .unwrap_or(0);
            if d == 0 {
                dur_pos = false;
            }
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
            if self.pre.op_idx[oid] > 0 {
                d += 1;
            }
            if self.mp[oid] != usize::MAX {
                d += 1;
            }
            self.indeg0[oid] = d;
        }
    }

    fn repair_machine(&mut self, m: usize) {
        self.repair_span(m, 0, self.machine_seq[m].len());
    }

    #[inline(always)]
    fn repair_span(&mut self, m: usize, from: usize, to: usize) {
        let len = self.machine_seq[m].len();
        for i in from..to {
            unsafe {
                let seq = &self.machine_seq[m];
                let o = *seq.get_unchecked(i);
                let p = *seq.get_unchecked(i.saturating_sub(1))
                    | 0usize.wrapping_sub((i == 0) as usize);
                let sc = *seq.get_unchecked((i + 1).min(len - 1))
                    | 0usize.wrapping_sub((i + 1 == len) as usize);
                *self.mpos.get_unchecked_mut(o) = i;
                *self.mp.get_unchecked_mut(o) = p;
                *self.ms.get_unchecked_mut(o) = sc;
                *self.indeg0.get_unchecked_mut(o) =
                    (*self.pre.op_idx.get_unchecked(o) > 0) as u8 + (i > 0) as u8;
            }
        }
    }

    #[inline(always)]
    fn repair_adjacent(&mut self, m: usize, p: usize) {
        self.repair_span(
            m,
            p.saturating_sub(1),
            (p + 3).min(self.machine_seq[m].len()),
        );
    }

    #[inline(always)]
    fn hd(&self, o: usize) -> u32 {
        unsafe { *self.heads.get_unchecked(o) }
    }
    #[inline(always)]
    fn he(&self, o: usize) -> u32 {
        unsafe { *self.hend.get_unchecked(o) }
    }
    #[inline(always)]
    fn qe(&self, o: usize) -> u32 {
        unsafe { *self.qend.get_unchecked(o) }
    }
    #[inline(always)]
    fn du(&self, o: usize) -> u32 {
        unsafe { *self.op_dur.get_unchecked(o) }
    }
    #[inline(always)]
    fn oidx(&self, o: usize) -> usize {
        unsafe { *self.pre.op_idx.get_unchecked(o) }
    }
    #[inline(always)]
    fn ojob(&self, o: usize) -> usize {
        unsafe { *self.pre.op_job.get_unchecked(o) }
    }
    #[inline(always)]
    fn jnops(&self, j: usize) -> usize {
        unsafe { *self.pre.job_nops.get_unchecked(j) }
    }

    fn eval(&mut self) -> Option<u32> {
        let n = self.pre.n_ops;
        const NONE: u32 = u32::MAX;
        {
            let nd = &mut self.nd;
            for o in 0..n {
                let ms = self.ms[o];
                nd[o] = [
                    0,
                    self.op_dur[o],
                    self.js[o],
                    if ms == usize::MAX { NONE } else { ms as u32 },
                    self.indeg0[o] as u32,
                    0,
                    0,
                    0,
                ];
            }
        }
        let mut tl = 0usize;
        for &oid in self.pre.job_off.iter() {
            if self.nd[oid][4] == 0 {
                self.topo[tl] = oid;
                tl += 1;
            }
        }
        let mut qi = 0;
        let mut makespan = 0u32;
        while qi < tl {
            let u = self.topo[qi];
            qi += 1;
            let ru = self.nd[u];
            let end_u = ru[0] + ru[1];
            self.nd[u][6] = end_u;
            if end_u > makespan {
                makespan = end_u;
            }
            let js = ru[2];
            if js != NONE {
                let js = js as usize;
                let r = &mut self.nd[js];
                if end_u > r[0] {
                    r[0] = end_u;
                }
                r[4] -= 1;
                if r[4] == 0 {
                    self.topo[tl] = js;
                    tl += 1;
                }
            }
            let mu = ru[3];
            if mu != NONE {
                let mu = mu as usize;
                let r = &mut self.nd[mu];
                if end_u > r[0] {
                    r[0] = end_u;
                }
                r[4] -= 1;
                if r[4] == 0 {
                    self.topo[tl] = mu;
                    tl += 1;
                }
            }
        }
        if tl != n {
            return None;
        }
        for idx in (0..tl).rev() {
            let u = self.topo[idx];
            let ru = self.nd[u];
            let mut t = 0u32;
            if ru[2] != NONE {
                t = t.max(self.nd[ru[2] as usize][5]);
            }
            if ru[3] != NONE {
                t = t.max(self.nd[ru[3] as usize][5]);
            }
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
        let mut job_schedule: Vec<Vec<(usize, u32)>> = self
            .pre
            .job_nops
            .iter()
            .map(|&k| Vec::with_capacity(k))
            .collect();
        for oid in 0..self.pre.n_ops {
            let j = self.pre.op_job[oid];
            job_schedule[j].push((self.op_machine[oid], self.heads[oid]));
        }
        Solution { job_schedule }
    }
}

#[inline(always)]
fn swap_estimate(g: &Graph, snap: &Snap, m: usize, p: usize) -> u32 {
    let seq = unsafe { g.machine_seq.get_unchecked(m) };
    let a = unsafe { *seq.get_unchecked(p) };
    let b = unsafe { *seq.get_unchecked(p + 1) };
    let comp = |o: usize| g.he(o);
    let tailc = |o: usize| g.qe(o);
    let jp_comp =
        |o: usize| -> u32 { comp(o.saturating_sub(1)) & 0u32.wrapping_sub((g.oidx(o) > 0) as u32) };
    let js_tail = |o: usize| -> u32 {
        let j = g.ojob(o);
        tailc((o + 1).min(g.pre.n_ops - 1)) & 0u32.wrapping_sub((g.oidx(o) + 1 < g.jnops(j)) as u32)
    };
    let (pe, qc) = snap.insertion(m);
    let pm_comp = unsafe { *pe.get_unchecked(p) };
    let sm_tail = unsafe { *qc.get_unchecked(p + 2) };
    let h_b = jp_comp(b).max(pm_comp);
    let h_a = jp_comp(a).max(h_b + g.du(b));
    let t_a = js_tail(a).max(sm_tail);
    let t_b = js_tail(b).max(g.du(a) + t_a);
    (h_a + g.du(a) + t_a).max(h_b + g.du(b) + t_b)
}

fn insert_pos(g: &Graph, mt: usize, o: usize) -> usize {
    let ho = g.heads[o];
    let seq = &g.machine_seq[mt];
    for (i, &x) in seq.iter().enumerate() {
        if g.heads[x] > ho {
            return i;
        }
    }
    seq.len()
}

#[inline(always)]
fn critical_path(g: &Graph, makespan: u32, path: &mut Vec<usize>) {
    let n = g.pre.n_ops;
    let crit = |o: usize| g.hd(o) + g.qe(o) == makespan;
    let mut start = usize::MAX;
    if g.dur_pos {
        for &o in g.pre.job_off.iter() {
            if crit(o) & (g.hd(o) == 0) {
                start = o;
                break;
            }
        }
    } else {
        for o in 0..n {
            if crit(o) & (g.hd(o) == 0) {
                start = o;
                break;
            }
        }
    }
    if start == usize::MAX {
        let mut best = u32::MAX;
        for o in 0..n {
            let h = g.hd(o);
            let take = crit(o) & (h < best);
            best = choose_u32(take, h, best);
            start = choose_usize(take, o, start);
        }
    }
    path.clear();
    let mut cur = start;
    loop {
        path.push(cur);
        let end = g.he(cur);
        let msc = unsafe { *g.ms.get_unchecked(cur) };
        let m = msc.min(n - 1);
        let j = (cur + 1).min(n - 1);
        let take_m = (msc != usize::MAX) & crit(m) & (g.hd(m) == end);
        let take_j = (g.oidx(cur) + 1 < g.jnops(g.ojob(cur))) & crit(j) & (g.hd(j) == end);
        let next = choose_usize(take_m, m, choose_usize(take_j, j, usize::MAX));
        if next == usize::MAX {
            break;
        }
        cur = next;
    }
    #[cfg(debug_assertions)]
    {
        fn reference(g: &Graph, makespan: u32, path: &mut Vec<usize>) {
            let n = g.pre.n_ops;
            let crit = |o: usize| g.hd(o) + g.qe(o) == makespan;
            let mut start = usize::MAX;
            if g.dur_pos {
                for &o in g.pre.job_off.iter() {
                    if crit(o) && g.hd(o) == 0 {
                        start = o;
                        break;
                    }
                }
            } else {
                for o in 0..n {
                    if crit(o) && g.heads[o] == 0 {
                        start = o;
                        break;
                    }
                }
            }
            if start == usize::MAX {
                let mut best = u32::MAX;
                for o in 0..n {
                    if crit(o) && g.heads[o] < best {
                        best = g.heads[o];
                        start = o;
                    }
                }
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
                    if crit(v) && g.hd(v) == end {
                        next = v;
                    }
                }
                if next == usize::MAX {
                    if g.oidx(cur) + 1 < g.jnops(g.ojob(cur)) {
                        let v = cur + 1;
                        if crit(v) && g.hd(v) == end {
                            next = v;
                        }
                    }
                }
                if next == usize::MAX {
                    break;
                }
                cur = next;
            }
        }
        let mut expected = Vec::new();
        reference(g, makespan, &mut expected);
        debug_assert_eq!(*path, expected);
    }
}

#[inline(always)]
fn n5_moves(g: &Graph, path: &[usize], moves: &mut Vec<(usize, usize)>, blk: &mut Vec<(u32, u32)>) {
    blk.clear();
    moves.clear();
    let mut s0 = 0usize;
    for i in 1..path.len() {
        let o = unsafe { *path.get_unchecked(i) };
        let prev = unsafe { *path.get_unchecked(i - 1) };
        if !(unsafe { *g.op_machine.get_unchecked(o) == *g.op_machine.get_unchecked(prev) }
            & unsafe { *g.mpos.get_unchecked(o) == *g.mpos.get_unchecked(prev) + 1 })
        {
            blk.push((s0 as u32, i as u32));
            s0 = i;
        }
    }
    if !path.is_empty() {
        blk.push((s0 as u32, path.len() as u32));
    }

    let nb = blk.len();
    for bi in 0..nb {
        let bs = blk[bi].0 as usize;
        let be = blk[bi].1 as usize;
        if be - bs < 2 {
            continue;
        }
        let m = g.op_machine[path[bs]];
        if bi != 0 {
            moves.push((m, g.mpos[path[bs]]));
        }
        if bi != nb - 1 {
            let p = g.mpos[path[be - 2]];
            if !(bi != 0 && p == g.mpos[path[bs]]) {
                moves.push((m, p));
            }
        }
    }
}

#[inline]
fn sweep_cap(pre: &Pre) -> usize {
    debug_assert!(pre.max_flex > 1);
    if !pre.common_route {
        60
    } else {
        80
    }
}

#[inline]
fn patience(pre: &Pre) -> usize {
    debug_assert!(pre.max_flex > 1);
    if !pre.common_route {
        150_000
    } else {
        3_000
    }
}

type HGraph = hybrid_engine::graph::Graph;
type HModel = hybrid_engine::model::Model;

fn mirror_load(hg: &mut HGraph, hm: &HModel, g: &Graph) -> bool {
    let mut sched: Vec<Vec<(usize, u32)>> = g
        .pre
        .job_nops
        .iter()
        .map(|&k| vec![(0usize, 0u32); k])
        .collect();
    for m in 0..g.machine_seq.len() {
        for (i, &o) in g.machine_seq[m].iter().enumerate() {
            let j = g.pre.op_job[o];
            let k = g.pre.op_idx[o];
            sched[j][k] = (m, i as u32);
        }
    }
    hg.load_schedule(hm, &sched)
}

#[inline(always)]
fn mirror_eval(g: &mut Graph, hg: &mut HGraph) -> Option<u32> {
    let mk = hg.evaluate_tracked_js()?;
    hg.compute_tails_tracked_js();
    if hg.chg_full {
        let n = g.pre.n_ops;
        for o in 0..n {
            unsafe {
                let h = *hg.head.get_unchecked(o);
                *g.heads.get_unchecked_mut(o) = h;
                *g.hend.get_unchecked_mut(o) = h + *hg.dur.get_unchecked(o);
                *g.qend.get_unchecked_mut(o) = *hg.qend.get_unchecked(o);
            }
        }
    } else {
        for i in 0..hg.chg_n {
            unsafe {
                let o = *hg.chg.get_unchecked(i) as usize;
                let h = *hg.head.get_unchecked(o);
                *g.heads.get_unchecked_mut(o) = h;
                *g.hend.get_unchecked_mut(o) = h + *hg.dur.get_unchecked(o);
                *g.qend.get_unchecked_mut(o) = *hg.qend.get_unchecked(o);
            }
        }
    }
    hg.chg_n = 0;
    hg.chg_full = false;
    Some(mk)
}

fn tabu_run(
    g: &mut Graph,
    max_iter: usize,
    tenure: usize,
    full: bool,
    max_kicks: usize,
    fuel_floor: u64,
    rng: &mut SmallRng,
    hm: &HModel,
    use_gap: bool,
    abandon_above: u32,
    bias: &LoadBias,
    fm_pat: usize,
) -> Option<u32> {
    let mut cur_real = g.eval()?;
    let mut hg = HGraph::new(hm);
    let mut inc = mirror_load(&mut hg, hm, g);
    if inc {
        match mirror_eval(g, &mut hg) {
            Some(mk) => {
                debug_assert_eq!(mk, cur_real);
                cur_real = mk;
            }
            None => {
                inc = false;
            }
        }
        if !inc {
            cur_real = g.eval()?;
        }
    }
    let mut best_om: Vec<usize> = g.op_machine.clone();
    let mut best_ms: Vec<Vec<usize>> = g.machine_seq.clone();
    let mut best_real = cur_real;
    let mut re_exp: Vec<usize> = vec![0; g.pre.n_ops];
    let mut swap_tabu: HashMap<(usize, usize), usize> =
        HashMap::with_hasher(seeded_hasher(&g.pre.seed));
    let mut tabu_hi: Vec<usize> = vec![0; g.pre.n_ops];
    let mut freq = vec![0u32; g.pre.n_ops];
    let mut no_improve = 0usize;
    let mut kick_failed = false;
    let mut aspire = { vec![INF; g.pre.num_machines] };
    let max_no_improve = if max_kicks > 0 {
        if !g.pre.common_route {
            fm_pat
        } else {
            patience(g.pre)
        }
    } else {
        8 * tenure + 400
    };
    let sweep_cap = sweep_cap(g.pre);
    let mut kicks_left = max_kicks;
    let n_ops = g.pre.n_ops;
    let mut snap = Snap::new(n_ops, g.machine_seq.len());
    let mut path: Vec<usize> = Vec::with_capacity(n_ops);
    let mut moves: Vec<(usize, usize)> = Vec::with_capacity(n_ops + 2);
    let mut blk: Vec<(u32, u32)> = Vec::with_capacity(n_ops);

    for it in 0..max_iter {
        let over = best_real.saturating_sub(abandon_above);
        let need = choose_usize(
            over > abandon_above / 50,
            1_500,
            choose_usize(over > abandon_above / 100, 3_000, 5_000),
        );
        let abandon = use_gap & (it > 0) & (best_real > abandon_above) & (no_improve >= need);
        if abandon | (fuel_left() <= fuel_floor) {
            break;
        }
        debug_assert!(re_exp.iter().zip(&tabu_hi).all(|(&r, &h)| r <= h));
        snap.fill(g, true);
        critical_path(g, cur_real, &mut path);
        if path.len() < 2 {
            break;
        }
        if full {
            block_moves(g, cur_real, &mut moves);
        } else {
            n5_moves(g, &path, &mut moves, &mut blk)
        };
        if moves.is_empty() {
            break;
        }
        let mut best_idx = 0usize;
        let mut best_est = INF;
        let mut best_key = rank_key(INF, 0);
        let mut fb_idx = 0usize;
        let mut fb_est = INF;
        for (idx, &(m, p)) in moves.iter().enumerate() {
            let seq = unsafe { g.machine_seq.get_unchecked(m) };
            let a = unsafe { *seq.get_unchecked(p) };
            let b = unsafe { *seq.get_unchecked(p + 1) };
            let est = swap_estimate(g, &snap, m, p);
            let is_tabu =
                if unsafe { (*tabu_hi.get_unchecked(a) <= it) | (*tabu_hi.get_unchecked(b) <= it) }
                {
                    false
                } else {
                    swap_tabu
                        .get(&(a.min(b), a.max(b)))
                        .map_or(false, |&exp| exp > it)
                };
            debug_assert_eq!(
                is_tabu,
                swap_tabu
                    .get(&(a.min(b), a.max(b)))
                    .map_or(false, |&exp| exp > it)
            );
            let key = rank_key(est, g.qe(a).max(g.qe(b)));
            let admission = best_real.max(unsafe { *aspire.get_unchecked(m) });
            let take = (!is_tabu | (est < admission)) & (key < best_key);
            best_key = choose_u64(take, key, best_key);
            best_est = choose_u32(take, est, best_est);
            best_idx = choose_usize(take, idx, best_idx);
            let take_fb = est < fb_est;
            fb_est = choose_u32(take_fb, est, fb_est);
            fb_idx = choose_usize(take_fb, idx, fb_idx);
        }
        let mut best_swap = (best_est < INF).then_some(moves[best_idx]);
        let fb_swap = (fb_est < INF).then_some(moves[fb_idx]);
        let mm = best_machine_move_mg(g, &snap, &path, best_real, &re_exp, it, sweep_cap, &aspire);
        let mm_l = mm.map_or(INF, |(_, _, _, l)| l);
        let xs = if best_est.min(mm_l) >= cur_real {
            best_machine_swap_mg(g, &snap, &path, best_real, &re_exp, it, sweep_cap, &aspire)
        } else {
            None
        };
        let xs_l = xs.map_or(INF, |(_, _, l)| l);
        if best_swap.is_none() & mm.is_none() & xs.is_none() {
            if let Some((m, p)) = fb_swap {
                best_swap = Some((m, p));
                best_est = fb_est;
            } else {
                break;
            }
        }
        {
            if let Some((m, _)) = best_swap {
                aspire[m] = aspire[m].min(best_est);
            }
            if let Some((_, m, _, l)) = mm {
                aspire[m] = aspire[m].min(l);
            }
            if let Some((_, u, l)) = xs {
                let m = g.op_machine[u];
                aspire[m] = aspire[m].min(l);
            }
        }
        let t_eff = tenure * 3 / 4 + rng.gen_range(0..=tenure / 2);
        let t_eff = t_eff + (no_improve >> 7).min(tenure * 2);
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
            debug_assert_eq!(
                Some(g.mpos[v]),
                g.machine_seq[mv].iter().position(|&x| x == v)
            );
            debug_assert_eq!(
                Some(g.mpos[u]),
                g.machine_seq[k].iter().position(|&x| x == u)
            );
            let pv = g.mpos[v];
            let pu = g.mpos[u];
            let dv_old = g.op_dur[v];
            let du_old = g.op_dur[u];
            let dv_new = g.pre.dur_first(v, k);
            let du_new = g.pre.dur_first(u, mv);
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
            let ten_v = frequency_tenure(&mut freq, v, t_eff);
            let ten_u = frequency_tenure(&mut freq, u, t_eff);
            re_exp[v] = it + ten_v;
            re_exp[u] = it + ten_u;
            tabu_hi[v] = tabu_hi[v].max(it + ten_v);
            tabu_hi[u] = tabu_hi[u].max(it + ten_u);
            xs_revert = Some((v, mv, pv, dv_old, u, k, pu, du_old));
        } else if mm_l < best_est {
            let (v, k, pos, _) = mm.unwrap();
            let m_old = g.op_machine[v];
            debug_assert_eq!(
                Some(g.mpos[v]),
                g.machine_seq[m_old].iter().position(|&x| x == v)
            );
            let p_old = g.mpos[v];
            let old_dur = g.op_dur[v];
            let dt = g.pre.dur_first(v, k);
            g.machine_seq[m_old].remove(p_old);
            g.op_machine[v] = k;
            g.op_dur[v] = dt;
            g.machine_seq[k].insert(pos, v);
            g.repair_machine(m_old);
            g.repair_machine(k);
            if inc {
                hg_old = hg.reassign_at(v, k as u32, pos, dt);
            }
            let ten_v = frequency_tenure(&mut freq, v, t_eff);
            re_exp[v] = it + ten_v;
            tabu_hi[v] = tabu_hi[v].max(it + ten_v);
            mg_revert = Some((v, m_old, p_old, old_dur));
        } else if let Some((m, p)) = best_swap {
            let a = g.machine_seq[m][p];
            let b = g.machine_seq[m][p + 1];
            let best_pair = (a.min(b), a.max(b));
            g.machine_seq[m].swap(p, p + 1);
            g.repair_adjacent(m, p);
            if inc {
                hg.swap_adjacent(best_pair.0 as u32, best_pair.1 as u32);
            }
            let ten_pair = {
                let ta = frequency_tenure(&mut freq, best_pair.0, t_eff);
                let tb = frequency_tenure(&mut freq, best_pair.1, t_eff);
                ta.max(tb)
            };
            swap_tabu.insert(best_pair, it + ten_pair);
            tabu_hi[best_pair.0] = tabu_hi[best_pair.0].max(it + ten_pair);
            tabu_hi[best_pair.1] = tabu_hi[best_pair.1].max(it + ten_pair);
            swap_revert = Some((m, p));
        } else {
            break;
        }
        let ev = if inc {
            mirror_eval(g, &mut hg)
        } else {
            g.eval()
        };
        if inc {
            if let Some((v, _, _, _)) = mg_revert {
                g.hend[v] = g.heads[v] + g.op_dur[v];
            }
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
                    if inc {
                        if let Some((t1, t2)) = xs_hg {
                            hg.undo_reassign(u, t2);
                            hg.undo_reassign(v, t1);
                        }
                    }
                } else if let Some((v, m_old, p_old, old_dur)) = mg_revert {
                    let k = g.op_machine[v];
                    debug_assert_eq!(
                        Some(g.mpos[v]),
                        g.machine_seq[k].iter().position(|&x| x == v)
                    );
                    let pn = g.mpos[v];
                    g.machine_seq[k].remove(pn);
                    g.op_machine[v] = m_old;
                    g.op_dur[v] = old_dur;
                    g.machine_seq[m_old].insert(p_old, v);
                    g.repair_machine(k);
                    g.repair_machine(m_old);
                    if inc {
                        hg.undo_reassign(v, hg_old);
                    }
                } else if let Some((m, p)) = swap_revert {
                    let a = g.machine_seq[m][p];
                    let b = g.machine_seq[m][p + 1];
                    g.machine_seq[m].swap(p, p + 1);
                    g.repair_adjacent(m, p);
                    if inc {
                        hg.swap_adjacent(a as u32, b as u32);
                    }
                }
                let ev2 = if inc {
                    mirror_eval(g, &mut hg)
                } else {
                    g.eval()
                };
                if inc {
                    if let Some((v, _, _, _)) = mg_revert {
                        g.hend[v] = g.heads[v] + g.op_dur[v];
                    }
                    if let Some((v, _, _, _, u, _, _, _)) = xs_revert {
                        g.hend[v] = g.heads[v] + g.op_dur[v];
                        g.hend[u] = g.heads[u] + g.op_dur[u];
                    }
                }
                match ev2 {
                    Some(r) => r,
                    None => break,
                }
            }
        };
        if cur_real < best_real {
            best_real = cur_real;
            best_om.copy_from_slice(&g.op_machine);
            for (dst, src) in best_ms.iter_mut().zip(g.machine_seq.iter()) {
                dst.clear();
                dst.extend_from_slice(src);
            }
            {
                for e in &mut aspire {
                    *e = age_aspiration(*e, best_real);
                }
            }
            no_improve = 0;
            kick_failed = false;
        } else {
            no_improve += 1;
            if no_improve >= max_no_improve {
                if kicks_left == 0 || kick_failed {
                    break;
                }
                g.reset(&best_om, &best_ms);
                kick_medium(
                    g,
                    3 + (max_kicks - kicks_left),
                    rng,
                    bias,
                    &mut path,
                    &mut moves,
                );
                if inc {
                    inc = mirror_load(&mut hg, hm, g);
                }
                cur_real = match if inc {
                    mirror_eval(g, &mut hg)
                } else {
                    g.eval()
                } {
                    Some(r) => r,
                    None => break,
                };
                re_exp.fill(0);
                swap_tabu.clear();
                for h in tabu_hi.iter_mut() {
                    *h = 0;
                }
                aspire.fill(INF);
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

#[inline(always)]
fn frequency_tenure(freq: &mut [u32], op: usize, tenure: usize) -> usize {
    debug_assert!(op < freq.len());
    let frequency = unsafe { freq.get_unchecked_mut(op) };
    let value = tenure + ((*frequency >> 4) as usize).min(tenure);
    *frequency += 1;
    value
}

fn kick(g: &mut Graph, k: usize, rng: &mut SmallRng) {
    if g.eval().is_none() {
        return;
    }
    let nm = g.pre.num_machines;
    let mut applied = 0usize;
    let mut attempts = 0usize;
    while applied < k && attempts < k * 5 {
        attempts += 1;
        if rng.gen_bool(0.7) {
            let m = rng.gen_range(0..nm);
            let len = g.machine_seq[m].len();
            if len < 2 {
                continue;
            }
            let p = rng.gen_range(0..len - 1);
            g.machine_seq[m].swap(p, p + 1);
            g.repair_machine(m);
            if g.eval().is_none() {
                g.machine_seq[m].swap(p, p + 1);
                g.repair_machine(m);
            } else {
                applied += 1;
            }
        } else {
            let o = rng.gen_range(0..g.pre.n_ops);
            let j = g.pre.op_job[o];
            let oi = g.pre.op_idx[o];
            let flex = g.pre.job_ops[j][oi].len();
            if flex < 2 {
                continue;
            }
            let m_old = g.op_machine[o];
            let (k2, dt) = g.pre.job_ops[j][oi][rng.gen_range(0..flex)];
            if k2 == m_old {
                continue;
            }
            debug_assert_eq!(
                Some(g.mpos[o]),
                g.machine_seq[m_old].iter().position(|&x| x == o)
            );
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
                debug_assert_eq!(
                    Some(g.mpos[o]),
                    g.machine_seq[k2].iter().position(|&x| x == o)
                );
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

fn kick_medium(
    g: &mut Graph,
    strength: usize,
    rng: &mut SmallRng,
    bias: &LoadBias,
    path: &mut Vec<usize>,
    cands: &mut Vec<(usize, usize)>,
) {
    let cur = match g.eval() {
        Some(r) => r,
        None => return,
    };
    critical_path(g, cur, path);
    if path.len() < 2 {
        kick(g, strength, rng);
        return;
    }
    cands.clear();
    for &u in path.iter() {
        let m = g.op_machine[u];
        let pos = g.mpos[u];
        if pos > 0 {
            cands.push((m, pos - 1));
        }
        if pos + 1 < g.machine_seq[m].len() {
            cands.push((m, pos));
        }
    }
    cands.sort_unstable();
    cands.dedup();
    if cands.is_empty() {
        kick(g, strength, rng);
        return;
    }
    {
        let mut weighted = Vec::with_capacity(cands.len() * 2);
        for &c in cands.iter() {
            weighted.push(c);
            if c.0 == bias.machine {
                weighted.push(c);
            }
        }
        cands.clear();
        cands.extend_from_slice(&weighted);
    }
    let mut applied = 0usize;
    let mut attempts = 0usize;
    while applied < strength && attempts < strength * 5 {
        attempts += 1;
        if rng.gen_bool(0.35) {
            let u = path[rng.gen_range(0..path.len())];
            let elig = g.pre.eligible(u);
            if elig.len() < 2 {
                continue;
            }
            let m_old = g.op_machine[u];
            let (k2, dt) = elig[rng.gen_range(0..elig.len())];
            if k2 == m_old {
                continue;
            }
            let p_old = g.mpos[u];
            let dur_old = g.op_dur[u];
            g.machine_seq[m_old].remove(p_old);
            g.op_machine[u] = k2;
            g.op_dur[u] = dt;
            let pins = insert_pos(g, k2, u);
            g.machine_seq[k2].insert(pins, u);
            g.repair_machine(m_old);
            g.repair_machine(k2);
            if g.eval().is_none() {
                let pn = g.mpos[u];
                g.machine_seq[k2].remove(pn);
                g.op_machine[u] = m_old;
                g.op_dur[u] = dur_old;
                g.machine_seq[m_old].insert(p_old, u);
                g.repair_machine(k2);
                g.repair_machine(m_old);
            } else {
                applied += 1;
            }
        } else {
            let (m, pos) = cands[rng.gen_range(0..cands.len())];
            if pos + 1 < g.machine_seq[m].len() {
                g.machine_seq[m].swap(pos, pos + 1);
                g.repair_machine(m);
                if g.eval().is_none() {
                    g.machine_seq[m].swap(pos, pos + 1);
                    g.repair_machine(m);
                } else {
                    applied += 1;
                }
            }
        }
    }
    let _ = g.eval();
}

#[inline]
fn best_second_counts_ref(time: u32, avail: &[u32], elig: &[(usize, u32)]) -> (u32, u32, usize) {
    let mut best = INF;
    let mut second = INF;
    let mut cnt = 0usize;
    for &(m, pt) in elig {
        let end = time.max(avail[m]).saturating_add(pt);
        if end < best {
            second = best;
            best = end;
            cnt = 1;
        } else if end == best {
            cnt += 1;
        } else if end < second {
            second = end;
        }
    }
    if cnt > 1 && !(second < INF) {
        second = best;
    }
    (best, second, cnt.max(1))
}

fn fjsp_construct_ref(
    pre: &Pre,
    der: &Der,
    k: usize,
    rng: &mut SmallRng,
    route: Option<&[Vec<f32>]>,
    bias: &LoadBias,
) -> (Vec<usize>, Vec<Vec<usize>>) {
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
        let bottleneck = live_bottleneck(&work, sum_work);
        for &m in &idle {
            cands.clear();
            for j in 0..nj {
                let oi = job_next[j];
                if oi >= pre.job_nops[j] || job_ready[j] > time {
                    continue;
                }
                let elig = &pre.job_ops[j][oi];
                let Some(&(_, pt)) = elig.iter().find(|&&(mm, _)| mm == m) else {
                    continue;
                };
                let (best_end, second_end, cnt) = best_second_counts_ref(time, &avail, elig);
                if time.max(avail[m]).saturating_add(pt) != best_end {
                    continue;
                }
                let oid = pre.op_id(j, oi);
                let flex = elig.len() as f64;
                let flex_inv = 1.0 / flex;
                let regret = if second_end >= INF {
                    der.avg_op_min * 2.6
                } else {
                    (second_end - best_end) as f64
                };
                let reg_n = (regret / der.avg_op_min.max(1.0)).clamp(0.0, 6.0);
                let scar_urg = 1.0 / (cnt as f64).max(1.0);
                let load_n = load[m] / der.avg_machine_load.max(1e-9);
                let scar_n = der.machine_scarcity[m] / der.avg_machine_scarcity;
                let scarce_match = scar_n * (flex_inv - 1.0 / der.flex_avg);
                let end_n = best_end as f64 / der.time_scale.max(1.0);
                let proc_n = pt as f64 / der.avg_op_min.max(1.0);
                let rem_min_n = der.suf_min[j][oi] as f64 / der.horizon.max(1.0);
                let progress = 1.0 - remaining as f64 / (pre.n_ops as f64).max(1.0);
                let end_w =
                    (0.90 * fl + 0.72 * js) + (0.62 + 0.12 * fl) * progress + 0.18 * der.high_flex;
                let reg_w = (0.50 * fl + 0.78 * js) + 0.18 * (1.0 - progress);
                let load_w = if der.flex_avg >= 5.0 {
                    -(0.45 * fl + 0.75 * js) * der.flex_factor
                } else {
                    (0.45 * fl + 0.75 * js) * der.flex_factor
                };
                let mut pop_pen = if elig.len() >= 2 {
                    (0.07 + 0.15 * (1.0 - progress)).clamp(0.05, 0.24)
                        * der.machine_best_pop[m]
                        * der.flex_factor
                } else {
                    0.0
                };
                if m == bias.machine {
                    pop_pen *= bias.share;
                }
                let bal_pen = 0.0;
                let jitter = if use_rand {
                    rng.gen::<f64>() * 1e-9
                } else {
                    0.0
                };
                let route_bonus = route.map_or(0.0, |r| {
                    let weight = 0.90 * (1.0 - 0.55 * progress);
                    weight * r[oid][m] as f64 * der.flex_factor
                });
                let score = 1.05 * rem_min_n
                    + reg_w * der.flex_factor * reg_n
                    + 0.62 * der.flex_factor * flex_inv
                    + 0.55 * der.flex_factor * scarce_match
                    + load_w * load_n
                    + 0.20 * der.flex_factor * scar_urg
                    - end_w * end_n
                    - (0.18 * fl + 0.12 * js) * proc_n
                    - pop_pen
                    + bal_pen;
                let score = {
                    score
                        - if m == bottleneck && elig.len() >= 2 {
                            (0.30 * der.flex_factor).min(0.55)
                        } else {
                            0.0
                        }
                };
                let score = score + route_bonus + jitter;
                cands.push((score, j, pt, oid));
            }
            if cands.is_empty() {
                continue;
            }
            let (_, j, pt, oid) = if use_rand {
                cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
                {
                    let progress = 1.0 - remaining as f64 / (pre.n_ops as f64).max(1.0);
                    let kk = smooth_width(k, progress).max(1).min(cands.len()).min(8);
                    cands[rng.gen_range(0..kk)]
                }
            } else {
                *cands
                    .iter()
                    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(b.1.cmp(&a.1)))
                    .unwrap()
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
            for &(mm, _) in elig {
                load[mm] = (load[mm] - delta).max(0.0);
            }
            remaining -= 1;
        }
        if remaining == 0 {
            break;
        }
        let mut next: Option<u32> = None;
        for &t in &avail {
            if t > time {
                next = Some(next.map_or(t, |b| b.min(t)));
            }
        }
        for j in 0..nj {
            if job_next[j] < pre.job_nops[j] && job_ready[j] > time {
                let t = job_ready[j];
                next = Some(next.map_or(t, |b| b.min(t)));
            }
        }
        match next {
            Some(t) => time = t,
            None => break,
        }
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
    pre: &Pre,
    om_a: &[usize],
    ms_a: &[Vec<usize>],
    om_b: &[usize],
    pb: &[usize],
    pa: &mut [usize],
) -> usize {
    let n = pre.n_ops;
    let mut d = 0usize;
    for o in 0..n {
        d += 2 * (om_a[o] != om_b[o]) as usize;
    }
    for seq in ms_a {
        for (p, &o) in seq.iter().enumerate() {
            pa[o] = p;
        }
    }
    for o in 0..n {
        d += (pa[o] != pb[o]) as usize;
    }
    d
}

fn pool_consider(
    pool: &mut Vec<PoolMember>,
    pre: &Pre,
    mk: u32,
    om: &[usize],
    ms: &[Vec<usize>],
    cap: usize,
    min_dist: usize,
) -> bool {
    let mut nearest = usize::MAX;
    let mut nd = usize::MAX;
    let mut pa = vec![0usize; pre.n_ops];
    let mut pb = vec![0usize; pre.n_ops];
    for seq in ms {
        for (p, &o) in seq.iter().enumerate() {
            pb[o] = p;
        }
    }
    for (i, e) in pool.iter().enumerate() {
        let dd = sol_distance(pre, &e.om, &e.ms, om, &pb, &mut pa);
        if dd < nd {
            nd = dd;
            nearest = i;
        }
    }
    if nearest != usize::MAX && nd < min_dist {
        if mk < pool[nearest].mk {
            pool[nearest].mk = mk;
            pool[nearest].om.copy_from_slice(om);
            for (dst, src) in pool[nearest].ms.iter_mut().zip(ms.iter()) {
                dst.clear();
                dst.extend_from_slice(src);
            }
            pool[nearest].deep = false;
            return true;
        }
        return false;
    }
    if pool.len() < cap {
        pool.push(PoolMember {
            mk,
            om: om.to_vec(),
            ms: ms.to_vec(),
            deep: false,
        });
        return true;
    }
    let mut worst = 0usize;
    for i in 1..pool.len() {
        if pool[i].mk > pool[worst].mk {
            worst = i;
        }
    }
    if mk < pool[worst].mk {
        pool[worst] = PoolMember {
            mk,
            om: om.to_vec(),
            ms: ms.to_vec(),
            deep: false,
        };
        return true;
    }
    false
}

fn build_consensus(pre: &Pre, pool: &[PoolMember]) -> Vec<Vec<f32>> {
    let nm = pre.num_machines;
    let mut route = vec![vec![0f32; nm]; pre.n_ops];
    if pool.is_empty() {
        return route;
    }

    let mut idx: Vec<usize> = (0..pool.len()).collect();
    idx.sort_by_key(|&i| pool[i].mk);
    let best_mk = pool[idx[0]].mk as f32;
    let gap_scale = (best_mk * 0.03).max(1.0);
    let mut wsum = 0f32;
    for (rank, &i) in idx.iter().enumerate() {
        let w = 1.0 / (1.0 + rank as f32);
        let w = {
            let gap = (pool[i].mk as f32 - best_mk).max(0.0);
            w * (-gap / gap_scale).exp()
        };
        wsum += w;
        for oid in 0..pre.n_ops {
            route[oid][pool[i].om[oid]] += w;
        }
    }
    if wsum > 0.0 {
        for r in route.iter_mut() {
            for v in r.iter_mut() {
                *v /= wsum;
            }
        }
    }
    route
}

fn cross_elite(
    pre: &Pre,
    a: &PoolMember,
    b: &PoolMember,
    om: &mut Vec<usize>,
    ms: &mut [Vec<usize>],
    placed: &mut [bool],
) {
    om.clear();
    for o in 0..pre.n_ops {
        om.push(if a.om[o] == b.om[o] || (o & 1) == 0 {
            a.om[o]
        } else {
            b.om[o]
        });
    }
    for seq in ms.iter_mut() {
        seq.clear();
    }
    placed.fill(false);
    for parent in [b, a] {
        for m in 0..pre.num_machines {
            for &o in &parent.ms[m] {
                if om[o] == m && !placed[o] {
                    ms[m].push(o);
                    placed[o] = true;
                }
            }
        }
    }
    debug_assert!(placed.iter().all(|&p| p));
}

fn make_seed(
    pre: &Pre,
    der: &Der,
    idx: usize,
    rng: &mut SmallRng,
    bias: &LoadBias,
) -> (Vec<usize>, Vec<Vec<usize>>) {
    if der.use_rich {
        if idx == 0 {
            fjsp_construct(pre, der, 0, rng, None, bias)
        } else if idx % 3 == 0 {
            assignment_construct(pre, Some(rng))
        } else {
            let k = 2 + (idx * 3) % 4;
            fjsp_construct(pre, der, k, rng, None, bias)
        }
    } else {
        if idx == 0 {
            dispatch_construct(pre, 0, None, 0)
        } else if idx % 3 == 0 {
            assignment_construct(pre, Some(rng))
        } else {
            let rule = (idx % 5) as u8;
            let top_k = 2 + (idx * 3) % 4;
            dispatch_construct(pre, rule, Some(rng), top_k)
        }
    }
}

struct Best<'a> {
    mk: u32,
    om: Vec<usize>,
    ms: Vec<Vec<usize>>,
    save: &'a dyn Fn(&Solution) -> Result<()>,
}

impl Best<'_> {
    fn offer(&mut self, g: &Graph, mk: u32) -> Result<bool> {
        if mk >= self.mk {
            return Ok(false);
        }
        let s = g.to_solution();
        {
            if solution_makespan(g.pre, &s).is_none() {
                return Ok(false);
            }
        }
        self.mk = mk;
        (self.save)(&s)?;
        self.om.copy_from_slice(&g.op_machine);
        for (d, sm) in self.ms.iter_mut().zip(g.machine_seq.iter()) {
            d.clear();
            d.extend_from_slice(sm);
        }
        Ok(true)
    }
}

fn solve_pool<const TAKE: bool>(
    challenge: &Challenge,
    pre: &Pre,
    der: &Der,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    rng: &mut SmallRng,
    bias: &LoadBias,
    scale: u64,
    clamp: bool,
    reserve_permille: u64,
    fm_pat: usize,
    fm_deep: usize,
    fm_slice: u64,
    fm_take: u64,
    fm_tdiv: u64,
    fm_tx: bool,
) -> Result<()> {
    let tenure = 8 + pre.num_jobs / 4;
    let cap = 16usize;
    let elites = 3usize;
    let min_dist = (pre.n_ops / 16 + 2).max(2);
    let polish_iters = 4000usize;
    let deep_iters = fm_deep;

    let avail_fuel = fuel_left();
    let work_cap = FJSP_WORK_CAP_MEDIUM;
    let total0 = avail_fuel.saturating_sub(FUEL_STOP).min(work_cap);
    let mut floor_fuel = avail_fuel.saturating_sub(total0);
    let mut cal = Cal {
        f_ref: [0u64; 16],
        f_new: [0u64; 16],
        minus: 0,
        plus: 0,
    };

    let (om0, ms0) = if der.use_rich {
        fjsp_construct_ref(pre, der, 0, rng, None, bias)
    } else {
        make_seed(pre, der, 0, rng, bias)
    };
    let mut g = Graph::new(pre, om0, ms0);
    let hm = HModel::build(challenge);
    let mut pool: Vec<PoolMember> = Vec::new();
    let mut best = Best {
        mk: INF,
        om: g.op_machine.clone(),
        ms: g.machine_seq.clone(),
        save: save_solution,
    };

    let a0_cap = 24usize;
    {
        let a0_floor = floor_fuel + total0 * 85 / 100;
        let mut route: Vec<Vec<f32>> = Vec::new();
        let mut c = 0usize;
        while cal.left() > a0_floor && c < 4000 {
            if c % 24 == 0 && pool.len() >= 3 {
                route = build_consensus(pre, &pool);
            }
            let k = 2 + (c % 5);
            let use_route = !route.is_empty() && c % 2 == 0;
            let (om, ms) = cal.construct(
                pre,
                der,
                k,
                rng,
                if use_route { Some(&route[..]) } else { None },
                bias,
            );
            g.reset(&om, &ms);
            if let Some(mk) = g.eval() {
                best.offer(&g, mk)?;
                pool_consider(
                    &mut pool,
                    pre,
                    mk,
                    &g.op_machine,
                    &g.machine_seq,
                    a0_cap,
                    min_dist,
                );
            }
            c += 1;
        }
    }
    if pool.is_empty() {
        if let Some(mk) = g.eval() {
            pool_consider(
                &mut pool,
                pre,
                mk,
                &g.op_machine,
                &g.machine_seq,
                cap,
                min_dist,
            );
        }
    }

    let rem_ref = cal.left().saturating_sub(floor_fuel);
    let search = (rem_ref as u128 * scale as u128 / 1000) as u64;
    floor_fuel = if clamp {
        let reserve = cal.completion_reserve(total0.saturating_sub(rem_ref), reserve_permille);
        let left = fuel_left();
        let search = search.min(left.saturating_sub(FUEL_STOP).saturating_sub(reserve));
        left.saturating_sub(search)
    } else {
        fuel_left().saturating_sub(search)
    };
    let total0 = (total0 as u128 * scale as u128 / 1000) as u64;
    let final_floor = floor_fuel;
    let mut first_deep = true;
    let mut refresh_idx = 0usize;
    let mut kk = 2usize;
    let mut stall = 0usize;
    let mut kick_path = Vec::new();
    let mut kick_cands = Vec::new();
    let mut xom = Vec::new();
    let mut xms = vec![Vec::new(); pre.num_machines];
    let mut xplaced = vec![false; pre.n_ops];

    while fuel_left() > floor_fuel {
        let fuel_round_start = fuel_left();
        pool.sort_by(|a, b| {
            a.mk.cmp(&b.mk)
                .then_with(|| a.om.cmp(&b.om))
                .then_with(|| a.ms.cmp(&b.ms))
        });
        let e = elites.min(pool.len());
        let mut improved = false;

        for i in 0..e {
            if fuel_left() <= floor_fuel {
                break;
            }
            if pool[i].deep {
                continue;
            }
            let rem = fuel_left().saturating_sub(floor_fuel);
            let floor = if TAKE && first_deep {
                first_deep = false;
                let slice = rem / fm_slice;
                let reserve = (slice as u128 * fm_take as u128 / 1000) as u64;
                let shortened = (slice as u128 * (1000 - fm_take) as u128 / 1000) as u64;
                debug_assert!(reserve + shortened <= slice && slice - reserve - shortened <= 1);
                floor_fuel += reserve;
                fuel_left().saturating_sub(shortened).max(floor_fuel)
            } else {
                (fuel_left() - rem / fm_slice).max(floor_fuel)
            };
            let abandon_above = if i > 0 && best.mk != INF {
                best.mk + best.mk / 400
            } else {
                INF
            };
            g.reset(&pool[i].om, &pool[i].ms);
            if let Some(mk) = tabu_run(
                &mut g,
                deep_iters,
                tenure,
                true,
                8,
                floor,
                rng,
                &hm,
                true,
                abandon_above,
                bias,
                fm_pat,
            ) {
                if mk < pool[i].mk {
                    pool[i].mk = mk;
                    pool[i].om.copy_from_slice(&g.op_machine);
                    for (d, sm) in pool[i].ms.iter_mut().zip(g.machine_seq.iter()) {
                        d.clear();
                        d.extend_from_slice(sm);
                    }
                    pool[i].deep = false;
                } else {
                    pool[i].deep = true;
                }
                if best.offer(&g, mk)? {
                    improved = true;
                }
            }
        }

        if fuel_left() > floor_fuel && best.mk != INF {
            let rem = fuel_left().saturating_sub(floor_fuel);
            let slice = rem / fm_slice;
            let floor = (fuel_left() - slice).max(floor_fuel);
            if stall >= 2 && pool.len() >= 2 {
                let pi = 1 + (stall % (pool.len() - 1));
                cross_elite(pre, &pool[0], &pool[pi], &mut xom, &mut xms, &mut xplaced);
                g.reset(&xom, &xms);
                if g.eval().is_none() {
                    g.reset(&best.om, &best.ms);
                }
            } else {
                g.reset(&best.om, &best.ms);
            }
            if stall >= 4 && (stall & 1) == 0 {
                kick(&mut g, kk + 2, rng);
            } else {
                kick_medium(&mut g, kk, rng, bias, &mut kick_path, &mut kick_cands);
            }
            if let Some(mk) = tabu_run(
                &mut g, deep_iters, tenure, true, 6, floor, rng, &hm, false, INF, bias, fm_pat,
            ) {
                if best.offer(&g, mk)? {
                    improved = true;
                }
                pool_consider(
                    &mut pool,
                    pre,
                    mk,
                    &g.op_machine,
                    &g.machine_seq,
                    cap,
                    min_dist,
                );
            }
        }

        if fuel_left() > floor_fuel {
            let use_consensus = {
                let rem_budget = fuel_left().saturating_sub(floor_fuel);
                let consumed = total0.saturating_sub(rem_budget);
                !pool.is_empty() && consumed.saturating_mul(2) < total0
            };
            let (om, ms) = if use_consensus {
                let route = build_consensus(pre, &pool);
                fjsp_construct(pre, der, 3, rng, Some(&route), bias)
            } else {
                make_seed(pre, der, refresh_idx, rng, bias)
            };
            refresh_idx += 1;
            g.reset(&om, &ms);
            if let Some(mk) = tabu_run(
                &mut g,
                polish_iters,
                tenure,
                false,
                0,
                floor_fuel,
                rng,
                &hm,
                false,
                INF,
                bias,
                fm_pat,
            ) {
                if best.offer(&g, mk)? {
                    improved = true;
                }
                pool_consider(
                    &mut pool,
                    pre,
                    mk,
                    &g.op_machine,
                    &g.machine_seq,
                    cap,
                    min_dist,
                );
            }
        }

        if improved {
            stall = 0;
            kk = 2;
        } else {
            stall += 1;
            if stall % 3 == 0 {
                kk = (kk + 1).min(8);
            }
        }

        if fuel_left() >= fuel_round_start {
            if TAKE {
                return Ok(());
            }
            break;
        }
    }

    if TAKE {
        let floor_fuel = final_floor;
        let mut tail_cross = true;
        while fuel_left() > floor_fuel && best.mk != INF {
            let fuel_round_start = fuel_left();
            let mut improved = false;
            let rem = fuel_left().saturating_sub(floor_fuel);
            let slice = rem / fm_tdiv;
            let floor = (fuel_left() - slice).max(floor_fuel);
            if (if fm_tx { tail_cross } else { stall >= 2 }) && pool.len() >= 2 {
                if fm_tx {
                    pool.sort_by(|a, b| {
                        a.mk.cmp(&b.mk)
                            .then_with(|| a.om.cmp(&b.om))
                            .then_with(|| a.ms.cmp(&b.ms))
                    });
                }
                let pi = if fm_tx {
                    1
                } else {
                    1 + (stall % (pool.len() - 1))
                };
                cross_elite(pre, &pool[0], &pool[pi], &mut xom, &mut xms, &mut xplaced);
                g.reset(&xom, &xms);
                if g.eval().is_none() {
                    g.reset(&best.om, &best.ms);
                }
            } else {
                g.reset(&best.om, &best.ms);
            }
            if fm_tx {
                tail_cross = !tail_cross;
            }
            if stall >= 4 && (stall & 1) == 0 {
                kick(&mut g, kk + 2, rng);
            } else {
                kick_medium(&mut g, kk, rng, bias, &mut kick_path, &mut kick_cands);
            }
            if let Some(mk) = tabu_run(
                &mut g, deep_iters, tenure, true, 6, floor, rng, &hm, false, INF, bias, fm_pat,
            ) {
                if best.offer(&g, mk)? {
                    improved = true;
                }
                pool_consider(
                    &mut pool,
                    pre,
                    mk,
                    &g.op_machine,
                    &g.machine_seq,
                    cap,
                    min_dist,
                );
            }

            if improved {
                stall = 0;
                kk = 2;
            } else {
                stall += 1;
                if stall % 3 == 0 {
                    kk = (kk + 1).min(8);
                }
            }

            if fuel_left() >= fuel_round_start {
                break;
            }
        }
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

pub(super) fn solve<const TAKE: bool>(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    der: &Der,
    common_route: bool,
    scale: u64,
    clamp: bool,
    reserve_permille: u64,
    fm_pat: usize,
    fm_deep: usize,
    fm_slice: u64,
    fm_take: u64,
    fm_tdiv: u64,
    fm_tx: bool,
) -> Result<()> {
    let fm_pat = if fm_pat > 0 { fm_pat } else { 450_000 };
    let fm_deep = if fm_deep > 0 { fm_deep } else { 800_000 };
    let fm_slice = if fm_slice > 0 { fm_slice } else { 2 };
    let fm_tdiv = if (2..=16).contains(&fm_tdiv) {
        fm_tdiv
    } else {
        4
    };
    debug_assert!(fm_tdiv > 0);
    debug_assert!(!der.chaotic_like);
    let bias = LoadBias::build(der);

    let pre = Pre::build(challenge, common_route);
    let best_mk = std::cell::Cell::new(u32::MAX);
    let guarded = |s: &Solution| -> Result<()> {
        {
            if let Some(mk) = solution_makespan(&pre, s) {
                if mk <= best_mk.get() {
                    best_mk.set(mk);
                    return save_solution(s);
                }
            }
            return Ok(());
        }
    };
    let mut rng = SmallRng::from_seed(challenge.seed);
    solve_pool::<TAKE>(
        challenge,
        &pre,
        der,
        &guarded,
        &mut rng,
        &bias,
        scale,
        clamp,
        reserve_permille,
        fm_pat,
        fm_deep,
        fm_slice,
        fm_take,
        fm_tdiv,
        fm_tx,
    )?;
    greedy_floor(&pre, &guarded)?;
    Ok(())
}

#[inline(always)]
fn fjsp_construct(
    pre: &Pre,
    der: &Der,
    k: usize,
    rng: &mut SmallRng,
    route: Option<&[Vec<f32>]>,
    bias: &LoadBias,
) -> (Vec<usize>, Vec<Vec<usize>>) {
    if k > 1 {
        fjsp_construct_impl::<true>(pre, der, k, rng, route, bias)
    } else {
        fjsp_construct_impl::<false>(pre, der, k, rng, route, bias)
    }
}

fn fjsp_construct_impl<const RANDOM: bool>(
    pre: &Pre,
    der: &Der,
    k: usize,
    rng: &mut SmallRng,
    route: Option<&[Vec<f32>]>,
    bias: &LoadBias,
) -> (Vec<usize>, Vec<Vec<usize>>) {
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
    let use_rand = RANDOM;
    let js = der.jobshopness;
    let fl = 1.0 - js;
    let mut cands: Vec<(f64, usize, u32, usize)> = Vec::with_capacity(nj);

    let load_w = if der.flex_avg >= 5.0 {
        -(0.45 * fl + 0.75 * js) * der.flex_factor
    } else {
        (0.45 * fl + 0.75 * js) * der.flex_factor
    };
    let load_scale = der.avg_machine_load.max(1e-9);
    let op_scale = der.avg_op_min.max(1.0);
    let time_scale = der.time_scale.max(1.0);
    let horizon = der.horizon.max(1.0);
    let flex_base = 1.0 / der.flex_avg;
    let proc_w = 0.18 * fl + 0.12 * js;
    let mut total_work = 0u64;
    while remaining > 0 {
        let sum_work: u64 = total_work;
        let bottleneck = live_bottleneck(&work, sum_work);
        for m in 0..nm {
            if avail[m] > time {
                continue;
            }
            cands.clear();
            let progress = 1.0 - remaining as f64 / (pre.n_ops as f64).max(1.0);
            let limit = smooth_width(k, progress).max(1).min(8);
            let load_n = load[m] / load_scale;
            let scar_n = der.machine_scarcity[m] / der.avg_machine_scarcity;
            let end_w =
                (0.90 * fl + 0.72 * js) + (0.62 + 0.12 * fl) * progress + 0.18 * der.high_flex;
            let reg_w = (0.50 * fl + 0.78 * js) + 0.18 * (1.0 - progress);
            let mut pop_m = {
                (0.07 + 0.15 * (1.0 - progress)).clamp(0.05, 0.24)
                    * der.machine_best_pop[m]
                    * der.flex_factor
            };
            pop_m = choose_f64(m == bias.machine, pop_m * bias.share, pop_m);
            let bal_pen = 0.0;
            let start_m = time.max(avail[m]);
            let bottleneck_pen =
                choose_f64(m == bottleneck, (0.30 * der.flex_factor).min(0.55), 0.0);
            for j in 0..nj {
                let oi = unsafe { *job_next.get_unchecked(j) };
                if unsafe {
                    (oi >= *pre.job_nops.get_unchecked(j)) | (*job_ready.get_unchecked(j) > time)
                } {
                    continue;
                }
                let oid = unsafe { *pre.job_off.get_unchecked(j) + oi };
                let pt = pre.dur_first(oid, m);
                if pt == u32::MAX {
                    continue;
                }
                let elig = pre.eligible(oid);
                let (best_end, second_end, cnt) = best_second_counts(time, &avail, elig);
                if start_m.saturating_add(pt) != best_end {
                    continue;
                }
                let flex = elig.len() as f64;
                let flex_inv = 1.0 / flex;
                let regret = choose_f64(
                    second_end >= INF,
                    der.avg_op_min * 2.6,
                    second_end.wrapping_sub(best_end) as f64,
                );
                let reg_n = (regret / op_scale).clamp(0.0, 6.0);
                let scar_urg = 1.0 / (cnt as f64).max(1.0);
                let scarce_match = scar_n * (flex_inv - flex_base);
                let end_n = best_end as f64 / time_scale;
                let proc_n = pt as f64 / op_scale;
                let rem_min_n =
                    unsafe { *der.suf_min.get_unchecked(j).get_unchecked(oi) } as f64 / horizon;
                let pop_pen = choose_f64(elig.len() >= 2, pop_m, 0.0);
                let jitter = if use_rand {
                    rng.gen::<f64>() * 1e-9
                } else {
                    0.0
                };
                let route_bonus = route.map_or(0.0, |r| {
                    let weight = 0.90 * (1.0 - 0.55 * progress);
                    weight * r[oid][m] as f64 * der.flex_factor
                });
                let score = 1.05 * rem_min_n
                    + reg_w * der.flex_factor * reg_n
                    + 0.62 * der.flex_factor * flex_inv
                    + 0.55 * der.flex_factor * scarce_match
                    + load_w * load_n
                    + 0.20 * der.flex_factor * scar_urg
                    - end_w * end_n
                    - proc_w * proc_n
                    - pop_pen
                    + bal_pen;
                let score = { score - choose_f64(elig.len() >= 2, bottleneck_pen, 0.0) };
                let score = score + route_bonus + jitter;
                if RANDOM {
                    let c = (score, j, pt, oid);
                    let pos = cands.partition_point(|a| {
                        a.0.partial_cmp(&score).unwrap().is_gt() | ((a.0 == score) & (a.1 < j))
                    });
                    if pos < limit {
                        if cands.len() == limit {
                            cands.pop();
                        }
                        cands.insert(pos, c);
                    }
                } else {
                    cands.push((score, j, pt, oid));
                }
            }
            if cands.is_empty() {
                continue;
            }
            let (_, j, pt, oid) = if use_rand {
                let kk = limit.min(cands.len());
                cands[rng.gen_range(0..kk)]
            } else {
                *cands
                    .iter()
                    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(b.1.cmp(&a.1)))
                    .unwrap()
            };
            let start = time.max(avail[m]);
            let end = start + pt;
            op_machine[oid] = m;
            machine_seq[m].push(oid);
            job_next[j] += 1;
            job_ready[j] = end;
            avail[m] = end;
            work[m] += pt as u64;
            total_work += pt as u64;
            let elig = pre.eligible(oid);
            let delta = der.op_min[oid] as f64 / (elig.len() as f64).max(1.0);
            for &(mm, _) in elig {
                load[mm] = (load[mm] - delta).max(0.0);
            }
            remaining -= 1;
        }
        if remaining == 0 {
            break;
        }
        let mut next = u32::MAX;
        let mut found = false;
        for &t in &avail {
            let take = t > time;
            next = next.min(choose_u32(take, t, u32::MAX));
            found |= take;
        }
        for j in 0..nj {
            let t = unsafe { *job_ready.get_unchecked(j) };
            let take =
                unsafe { *job_next.get_unchecked(j) < *pre.job_nops.get_unchecked(j) } & (t > time);
            next = next.min(choose_u32(take, t, u32::MAX));
            found |= take;
        }
        if !found {
            break;
        }
        time = next;
    }
    (op_machine, machine_seq)
}

#[inline(always)]
fn best_second_counts(time: u32, avail: &[u32], elig: &[(usize, u32)]) -> (u32, u32, usize) {
    let mut best = INF;
    let mut second = INF;
    let mut cnt = 0usize;
    macro_rules! consider {
        ($candidate:expr) => {{
            let (m, pt) = $candidate;
            let end = time
                .max(unsafe { *avail.get_unchecked(m) })
                .saturating_add(pt);
            let lt = end < best;
            let eq = end == best;
            second = choose_u32(lt, best, choose_u32(!eq & (end < second), end, second));
            cnt = choose_usize(lt, 1, cnt + eq as usize);
            best = best.min(end);
        }};
    }
    let mut i = 0usize;
    while i + 1 < elig.len() {
        consider!(elig[i]);
        consider!(elig[i + 1]);
        i += 2;
    }
    if i < elig.len() {
        consider!(elig[i]);
    }
    second = choose_u32((cnt > 1) & (second >= INF), best, second);
    (best, second, cnt.max(1))
}

struct Cal {
    f_ref: [u64; 16],
    f_new: [u64; 16],
    minus: u64,
    plus: u64,
}
impl Cal {
    fn completion_reserve(&self, a0_ref: u64, permille: u64) -> u64 {
        let construct = self
            .f_ref
            .iter()
            .chain(&self.f_new)
            .copied()
            .max()
            .unwrap_or(0);
        let reserve = a0_ref.saturating_add(construct.saturating_mul(8));
        (reserve as u128 * permille as u128 / 1000).min(u64::MAX as u128) as u64
    }

    #[inline]
    fn left(&self) -> u64 {
        (fuel_left() + self.minus).saturating_sub(self.plus)
    }
    fn construct(
        &mut self,
        pre: &Pre,
        der: &Der,
        k: usize,
        rng: &mut SmallRng,
        route: Option<&[Vec<f32>]>,
        bias: &LoadBias,
    ) -> (Vec<usize>, Vec<Vec<usize>>) {
        let key = (k * 2 + route.is_some() as usize) & 15;
        if self.f_new[key] == 0 {
            let mut r2 = rng.clone();
            let f0 = fuel_left();
            let _ = fjsp_construct(pre, der, k, &mut r2, route, bias);
            let f1 = fuel_left();
            let res = fjsp_construct_ref(pre, der, k, rng, route, bias);
            let f2 = fuel_left();
            self.f_new[key] = (f0 - f1).max(1);
            self.f_ref[key] = f1 - f2;
            self.minus += f0 - f1;
            res
        } else {
            let f0 = fuel_left();
            let res = fjsp_construct(pre, der, k, rng, route, bias);
            let fd = f0 - fuel_left();
            self.minus += fd;
            self.plus += ((fd as u128 * self.f_ref[key] as u128) / self.f_new[key] as u128) as u64;
            res
        }
    }
}

struct Snap {
    off: Vec<usize>,
    he: Vec<u32>,
    qe: Vec<u32>,
    gap: Vec<u32>,
}

impl Snap {
    fn new(n_ops: usize, nm: usize) -> Snap {
        Snap {
            off: vec![0; nm + 1],
            he: vec![0; n_ops + nm],
            qe: vec![0; n_ops + nm],
            gap: vec![0; nm],
        }
    }

    #[inline(always)]
    fn fill_one(&mut self, g: &Graph, o: usize, off: usize, a: &mut u32, lo: &mut u32) {
        let he = g.he(o);
        let qe = g.qe(o);
        unsafe {
            *self.he.get_unchecked_mut(off + 1) = he;
            *self.qe.get_unchecked_mut(off) = qe;
        }
        *lo = (*lo).min(*a + qe);
        *a = he;
    }

    #[inline(always)]
    fn fill(&mut self, g: &Graph, use_gap: bool) {
        let mut off = 0usize;
        for (k, seq) in g.machine_seq.iter().enumerate() {
            let seq = seq.as_slice();
            self.off[k] = off;
            self.he[off] = 0;
            let mut a = 0u32;
            let mut lo = u32::MAX;
            let mut p = 0usize;
            let end = seq.len() & !3usize;
            while p < end {
                unsafe {
                    self.fill_one(g, *seq.get_unchecked(p), off, &mut a, &mut lo);
                    self.fill_one(g, *seq.get_unchecked(p + 1), off + 1, &mut a, &mut lo);
                    self.fill_one(g, *seq.get_unchecked(p + 2), off + 2, &mut a, &mut lo);
                    self.fill_one(g, *seq.get_unchecked(p + 3), off + 3, &mut a, &mut lo);
                }
                p += 4;
                off += 4;
            }
            while p < seq.len() {
                self.fill_one(g, unsafe { *seq.get_unchecked(p) }, off, &mut a, &mut lo);
                p += 1;
                off += 1;
            }
            self.gap[k] = lo.min(a) & 0u32.wrapping_sub(use_gap as u32);
            self.qe[off] = 0;
            off += 1;
        }
        self.off[g.machine_seq.len()] = off;
    }

    #[inline(always)]
    fn insertion(&self, k: usize) -> (&[u32], &[u32]) {
        unsafe {
            let a = *self.off.get_unchecked(k);
            let b = *self.off.get_unchecked(k + 1);
            (self.he.get_unchecked(a..b), self.qe.get_unchecked(a..b))
        }
    }
}

#[inline(always)]
fn ins_window(pe: &[u32], qc: &[u32], n: usize, r: u32, q: u32) -> (usize, usize) {
    if n >= 256 {
        let l = unsafe { pe.get_unchecked(1..n + 1) }.partition_point(|&h| h <= r);
        let rr = unsafe { qc.get_unchecked(..n) }.partition_point(|&x| x > q);
        return (l, rr);
    }
    let mut l = 0usize;
    let mut rr = 0usize;
    let mut size = n + 1;
    macro_rules! st {
        () => {
            let half = size / 2;
            let i = l + half;
            l = choose_usize(unsafe { *pe.get_unchecked(i) } <= r, i, l);
            let j = rr + half.saturating_sub(1);
            rr = choose_usize(unsafe { *qc.get_unchecked(j) } > q, rr + half, rr);
            size -= half;
        };
    }
    match n {
        0..=15 => {}
        16..=31 => {
            st!();
        }
        32..=63 => {
            st!();
            st!();
        }
        _ => {
            st!();
            st!();
            st!();
            st!();
        }
    }
    st!();
    st!();
    st!();
    st!();
    debug_assert_eq!(size, 1);
    (l, rr)
}

#[inline(always)]
fn best_machine_move_mg(
    g: &Graph,
    snap: &Snap,
    path: &[usize],
    best_real: u32,
    re_exp: &[usize],
    it: usize,
    sweep_cap: usize,
    aspire: &[u32],
) -> Option<(usize, usize, usize, u32)> {
    let (mut bv, mut bk, mut bp, mut bl) = (0usize, 0usize, 0usize, u32::MAX);
    let mut best_key = move_key(u32::MAX, u32::MAX, usize::MAX);
    let mut seen = 0usize;
    for &v in path {
        let j = g.ojob(v);
        let oi = g.oidx(v);
        let elig = g.pre.eligible(v);
        if elig.len() < 2 {
            continue;
        }
        seen += 1;
        if seen > sweep_cap {
            break;
        }
        let m_cur = unsafe { *g.op_machine.get_unchecked(v) };
        let rpj = g.he(v.saturating_sub(1)) & 0u32.wrapping_sub((oi > 0) as u32);
        let qsj =
            g.qe((v + 1).min(g.pre.n_ops - 1)) & 0u32.wrapping_sub((oi + 1 < g.jnops(j)) as u32);
        let is_tabu = unsafe { *re_exp.get_unchecked(v) > it };
        for &(k, pk) in elig {
            let gap = unsafe { *snap.gap.get_unchecked(k) };
            let admission = best_real.max(unsafe { *aspire.get_unchecked(k) });
            let limit = bl
                .saturating_add(1)
                .min(choose_u32(is_tabu, admission, u32::MAX));
            if (k == m_cur) | ((rpj + pk + qsj).max(pk + gap) >= limit) {
                continue;
            }
            let (pe_s, qc_s) = snap.insertion(k);
            let n = pe_s.len() - 1;
            let (left, right) = ins_window(pe_s, qc_s, n, rpj, qsj);
            if left > right {
                continue;
            }

            let mut pos = left;
            loop {
                let hv = unsafe { *pe_s.get_unchecked(pos) }.max(rpj);
                let tv = unsafe { *qc_s.get_unchecked(pos) }.max(qsj);
                let l = hv + pk + tv;
                let key = move_key(l, gap, n);
                let take = (!is_tabu | (l < admission)) & (key < best_key);
                bv = choose_usize(take, v, bv);
                bk = choose_usize(take, k, bk);
                bp = choose_usize(take, pos, bp);
                bl = choose_u32(take, l, bl);
                best_key = choose_u128(take, key, best_key);
                let p2 = pos + 1;
                let q2 = p2.min(n);
                let hv2 = unsafe { *pe_s.get_unchecked(q2) }.max(rpj);
                let tv2 = unsafe { *qc_s.get_unchecked(q2) }.max(qsj);
                let l2 = hv2 + pk + tv2;
                let key2 = move_key(l2, gap, n);
                let take2 = (p2 <= right) & (!is_tabu | (l2 < admission)) & (key2 < best_key);
                bv = choose_usize(take2, v, bv);
                bk = choose_usize(take2, k, bk);
                bp = choose_usize(take2, p2, bp);
                bl = choose_u32(take2, l2, bl);
                best_key = choose_u128(take2, key2, best_key);
                pos += 2;
                if pos > right {
                    break;
                }
            }
        }
    }
    if bl == u32::MAX {
        None
    } else {
        Some((bv, bk, bp, bl))
    }
}

#[inline(always)]
fn best_machine_swap_mg(
    g: &Graph,
    snap: &Snap,
    path: &[usize],
    best_real: u32,
    re_exp: &[usize],
    it: usize,
    sweep_cap: usize,
    aspire: &[u32],
) -> Option<(usize, usize, u32)> {
    let (mut bv, mut bu, mut bl) = (0usize, 0usize, u32::MAX);
    let mut best_slack = 0u32;
    let mut seen = 0usize;
    for &v in path {
        let jv = g.ojob(v);
        let oiv = g.oidx(v);
        let elig_v = g.pre.eligible(v);
        if elig_v.len() < 2 {
            continue;
        }
        seen += 1;
        if seen > sweep_cap {
            break;
        }
        let mv = unsafe { *g.op_machine.get_unchecked(v) };
        let pv = unsafe { *g.mpos.get_unchecked(v) };
        let rpv = g.he(v.saturating_sub(1)) & 0u32.wrapping_sub((oiv > 0) as u32);
        let qsv =
            g.qe((v + 1).min(g.pre.n_ops - 1)) & 0u32.wrapping_sub((oiv + 1 < g.jnops(jv)) as u32);
        let v_tabu = unsafe { *re_exp.get_unchecked(v) > it };
        let durations = g.pre.durations_on(mv);
        let (he_mv, qe_mv) = snap.insertion(mv);
        let hu_base = unsafe { *he_mv.get_unchecked(pv) };
        let tu_base = unsafe { *qe_mv.get_unchecked(pv + 1) };
        for &(k, pvk) in elig_v {
            let seq = unsafe { g.machine_seq.get_unchecked(k) }.as_slice();
            let len = seq.len();
            let admission = best_real.max(unsafe { *aspire.get_unchecked(k) });
            let initial_limit = bl.min(choose_u32(v_tabu, admission, u32::MAX));
            let prune = bl
                .saturating_add(1)
                .min(choose_u32(v_tabu, admission, u32::MAX));
            if (k == mv) | (rpv + pvk + qsv >= prune) | (len == 0) {
                continue;
            }
            let (pe_s, nq_s) = snap.insertion(k);
            let (left, right) = ins_window(pe_s, nq_s, len, rpv, qsv);
            let hi = right.min(len - 1);
            if left > hi {
                continue;
            }
            let mut pos = left;
            let mut limit = initial_limit;
            let mut best_key = rank_key(limit, choose_u32(limit == bl, best_slack, u32::MAX));
            let mut changed = false;
            macro_rules! consider {
                ($pos:expr) => {{
                    let pos = $pos;
                    let u = unsafe { *seq.get_unchecked(pos) };
                    let hv = unsafe { *pe_s.get_unchecked(pos) }.max(rpv);
                    let tv = unsafe { *nq_s.get_unchecked(pos + 1) }.max(qsv);
                    let l_v = hv + pvk + tv;
                    let pumv = unsafe { *durations.get_unchecked(u) };
                    let oiu = unsafe { *g.pre.op_idx.get_unchecked(u) };
                    let rpu = g.he(u.saturating_sub(1)) & 0u32.wrapping_sub((oiu > 0) as u32);
                    let qsu = g.qe((u + 1).min(g.pre.n_ops - 1))
                        & 0u32.wrapping_sub((unsafe { *g.js.get_unchecked(u) } != u32::MAX) as u32);
                    let l_u = hu_base
                        .max(rpu)
                        .wrapping_add(pumv)
                        .wrapping_add(tu_base.max(qsu));
                    let l = l_v.max(l_u);
                    let u_tabu = unsafe { *re_exp.get_unchecked(u) > it };
                    let slack = g.hd(u) + g.qe(u);
                    let key = rank_key(l, slack);
                    let take = (pumv != u32::MAX)
                        & (!(v_tabu | u_tabu) | (l < admission))
                        & (key < best_key);
                    best_slack = choose_u32(take, slack, best_slack);
                    best_key = choose_u64(take, key, best_key);
                    changed |= take;
                    bu = choose_usize(take, u, bu);
                    limit = choose_u32(take, l, limit);
                }};
            }
            while pos < hi {
                consider!(pos);
                consider!(pos + 1);
                pos += 2;
            }
            if pos <= hi {
                consider!(pos);
            }
            bv = choose_usize(changed, v, bv);
            bl = choose_u32(changed, limit, bl);
        }
    }
    if bl == u32::MAX {
        None
    } else {
        Some((bv, bu, bl))
    }
}

#[inline(always)]
fn n5_node(
    g: &Graph,
    makespan: u32,
    m: usize,
    pos: usize,
    node: usize,
    run: &mut usize,
    prev_end: &mut u32,
    dst: *mut (usize, usize),
    count: &mut usize,
) {
    let h = g.hd(node);
    let c = h + g.qe(node) == makespan;
    let link = c & (*run > 0) & (h == *prev_end);
    let first = link & (*run == 1);
    let last = !link & (*run >= 3);
    unsafe {
        dst.add(*count)
            .write((m, pos.wrapping_sub(2).wrapping_add(first as usize)));
    }
    *count += (first | last) as usize;
    *run = choose_usize(link, *run + 1, c as usize);
    *prev_end = g.he(node);
}

#[inline(always)]
fn n5_moves_full(g: &Graph, makespan: u32, moves: &mut Vec<(usize, usize)>) {
    moves.clear();
    moves.reserve(g.pre.n_ops + 2);
    let dst = moves.as_mut_ptr();
    let mut count = 0usize;
    for m in 0..g.pre.num_machines {
        let seq = g.machine_seq[m].as_slice();
        let mut run = 0usize;
        let mut prev_end = 0u32;
        let mut pos = 0usize;
        let end = seq.len() & !3usize;
        while pos < end {
            unsafe {
                n5_node(
                    g,
                    makespan,
                    m,
                    pos,
                    *seq.get_unchecked(pos),
                    &mut run,
                    &mut prev_end,
                    dst,
                    &mut count,
                );
                n5_node(
                    g,
                    makespan,
                    m,
                    pos + 1,
                    *seq.get_unchecked(pos + 1),
                    &mut run,
                    &mut prev_end,
                    dst,
                    &mut count,
                );
                n5_node(
                    g,
                    makespan,
                    m,
                    pos + 2,
                    *seq.get_unchecked(pos + 2),
                    &mut run,
                    &mut prev_end,
                    dst,
                    &mut count,
                );
                n5_node(
                    g,
                    makespan,
                    m,
                    pos + 3,
                    *seq.get_unchecked(pos + 3),
                    &mut run,
                    &mut prev_end,
                    dst,
                    &mut count,
                );
            }
            pos += 4;
        }
        while pos < seq.len() {
            n5_node(
                g,
                makespan,
                m,
                pos,
                unsafe { *seq.get_unchecked(pos) },
                &mut run,
                &mut prev_end,
                dst,
                &mut count,
            );
            pos += 1;
        }
        unsafe {
            dst.add(count).write((m, pos.wrapping_sub(2)));
        }
        count += (run >= 3) as usize;
    }
    unsafe {
        moves.set_len(count);
    }
    #[cfg(debug_assertions)]
    {
        let mut expected = Vec::new();
        for (m, seq) in g.machine_seq.iter().enumerate() {
            let mut start = 0;
            let mut run = 0;
            let mut prev_end = 0;
            for (pos, &o) in seq.iter().enumerate() {
                let h = g.hd(o);
                let critical = h + g.qe(o) == makespan;
                if critical && run > 0 && h == prev_end {
                    run += 1;
                } else {
                    if run >= 2 {
                        expected.push((m, start));
                    }
                    if run >= 3 {
                        expected.push((m, start + run - 2));
                    }
                    start = pos;
                    run = critical as usize;
                }
                prev_end = g.he(o);
            }
            if run >= 2 {
                expected.push((m, start));
            }
            if run >= 3 {
                expected.push((m, start + run - 2));
            }
        }
        debug_assert_eq!(*moves, expected);
    }
}

#[inline(always)]
fn live_bottleneck(work: &[u64], total: u64) -> usize {
    let mut machine = usize::MAX;
    let mut largest = 0;
    for (m, &w) in work.iter().enumerate() {
        machine = choose_usize(w > largest, m, machine);
        largest = largest.max(w);
    }
    choose_usize(
        (total != 0) & (largest * work.len() as u64 >= total + total / 8),
        machine,
        usize::MAX,
    )
}

#[inline(always)]
fn smooth_width(k: usize, progress: f64) -> usize {
    1 + ((k as f64 - 1.0) * (1.0 - progress).max(0.0).powf(1.5)).round() as usize
}

fn solution_makespan(pre: &Pre, solution: &Solution) -> Option<u32> {
    if solution.job_schedule.len() != pre.num_jobs {
        return None;
    }
    let mut usage = vec![Vec::new(); pre.num_machines];
    let mut makespan = 0;
    for (j, schedule) in solution.job_schedule.iter().enumerate() {
        if schedule.len() != pre.job_nops[j] {
            return None;
        }
        let mut ready = 0;
        for (oi, &(m, start)) in schedule.iter().enumerate() {
            if m >= pre.num_machines || start < ready {
                return None;
            }
            let pt = pre.job_ops[j][oi]
                .iter()
                .find(|&&(machine, _)| machine == m)?
                .1;
            let end = start.checked_add(pt)?;
            usage[m].push((start, end));
            ready = end;
        }
        makespan = makespan.max(ready);
    }
    for machine in &mut usage {
        machine.sort_by_key(|&(start, _)| start);
        if machine.windows(2).any(|w| w[1].0 < w[0].1) {
            return None;
        }
    }
    Some(makespan)
}

#[inline(always)]
fn block_emit(m: usize, start: usize, end: usize, dst: *mut (usize, usize), count: &mut usize) {
    let tail = (start + 3).max(end.saturating_sub(3));
    unsafe {
        dst.add(*count).write((m, start));
        dst.add(*count + 1).write((m, start + 1));
        dst.add(*count + 2).write((m, start + 2));
        dst.add(*count + 3).write((m, tail));
        dst.add(*count + 4).write((m, tail + 1));
        dst.add(*count + 5).write((m, tail + 2));
    }
    *count += (end - start).min(6);
}

#[inline(always)]
fn block_node(
    g: &Graph,
    makespan: u32,
    m: usize,
    pos: usize,
    o: usize,
    run: &mut usize,
    prev_end: &mut u32,
    dst: *mut (usize, usize),
    count: &mut usize,
) {
    let h = g.hd(o);
    let critical = h + g.qe(o) == makespan;
    let link = critical & (*run > 0) & (h == *prev_end);
    if !link & (*run >= 2) {
        block_emit(m, pos - *run, pos - 1, dst, count);
    }
    *run = choose_usize(link, *run + 1, critical as usize);
    *prev_end = g.he(o);
}

#[inline(always)]
fn block_moves(g: &Graph, makespan: u32, moves: &mut Vec<(usize, usize)>) {
    moves.clear();
    moves.reserve(g.pre.n_ops + 6);
    let dst = moves.as_mut_ptr();
    let mut count = 0usize;
    for (m, seq) in g.machine_seq.iter().enumerate() {
        let len = seq.len();
        let mut run = 0usize;
        let mut prev_end = 0u32;
        let mut pos = 0usize;
        if 1 < len {
            loop {
                let (pos0, emit0, emit1, run0, run1, end) = loop {
                    let pos0 = pos;
                    let o0 = unsafe { *seq.get_unchecked(pos0) };
                    let o1 = unsafe { *seq.get_unchecked(pos0 + 1) };
                    let run0 = run;
                    let h0 = g.hd(o0);
                    let critical0 = h0 + g.qe(o0) == makespan;
                    let link0 = critical0 & (run0 > 0) & (h0 == prev_end);
                    let emit0 = !link0 & (run0 >= 2);
                    let run1 = choose_usize(link0, run0 + 1, critical0 as usize);
                    let end0 = g.he(o0);
                    let h1 = g.hd(o1);
                    let critical1 = h1 + g.qe(o1) == makespan;
                    let link1 = critical1 & (run1 > 0) & (h1 == end0);
                    let emit1 = !link1 & (run1 >= 2);
                    run = choose_usize(link1, run1 + 1, critical1 as usize);
                    prev_end = g.he(o1);
                    pos = pos0 + 2;
                    let end = pos + 1 >= len;
                    if (emit0 | emit1) | end {
                        break (pos0, emit0, emit1, run0, run1, end);
                    }
                };
                if emit0 {
                    block_emit(m, pos0 - run0, pos0 - 1, dst, &mut count);
                }
                if emit1 {
                    block_emit(m, pos0 + 1 - run1, pos0, dst, &mut count);
                }
                if end {
                    break;
                }
            }
        }
        if pos < len {
            block_node(
                g,
                makespan,
                m,
                pos,
                unsafe { *seq.as_ptr().add(pos) },
                &mut run,
                &mut prev_end,
                dst,
                &mut count,
            );
        }
        if run >= 2 {
            block_emit(m, seq.len() - run, seq.len() - 1, dst, &mut count);
        }
    }
    unsafe {
        moves.set_len(count);
    }
    #[cfg(debug_assertions)]
    {
        fn critical_blocks(g: &Graph, makespan: u32, mut visit: impl FnMut(usize, usize, usize)) {
            for (m, seq) in g.machine_seq.iter().enumerate() {
                let mut start = usize::MAX;
                let mut prev_end = 0;
                for (pos, &o) in seq.iter().enumerate() {
                    let h = g.hd(o);
                    let critical = h + g.qe(o) == makespan;
                    if !(critical && start != usize::MAX && h == prev_end) {
                        if start != usize::MAX && pos - start >= 2 {
                            visit(m, start, pos - 1);
                        }
                        start = if critical { pos } else { usize::MAX };
                    }
                    prev_end = if critical { g.he(o) } else { 0 };
                }
                if start != usize::MAX && seq.len() - start >= 2 {
                    visit(m, start, seq.len() - 1);
                }
            }
        }

        fn reference(g: &Graph, makespan: u32, moves: &mut Vec<(usize, usize)>) {
            moves.clear();
            critical_blocks(g, makespan, |m, start, end| {
                if end - start <= 6 {
                    for p in start..end {
                        moves.push((m, p));
                    }
                } else {
                    for p in start..start + 3 {
                        moves.push((m, p));
                    }
                    for p in end - 3..end {
                        moves.push((m, p));
                    }
                }
            });
        }
        let mut expected = Vec::new();
        reference(g, makespan, &mut expected);
        debug_assert_eq!(*moves, expected);
    }
}

pub(super) mod high {
    use super::*;

    #[inline(always)]
    fn choose_u64(take: bool, yes: u64, no: u64) -> u64 {
        let mask = 0u64.wrapping_sub(take as u64);
        (yes & mask) | (no & !mask)
    }

    #[inline(always)]
    fn choose_u128(take: bool, yes: u128, no: u128) -> u128 {
        let mask = 0u128.wrapping_sub(take as u128);
        (yes & mask) | (no & !mask)
    }

    #[inline(always)]
    fn choose_f64(take: bool, yes: f64, no: f64) -> f64 {
        f64::from_bits(choose_u64(take, yes.to_bits(), no.to_bits()))
    }

    #[inline(always)]
    fn rank_key(estimate: u32, tie: u32) -> u64 {
        ((estimate as u64) << 32) | (!tie as u64)
    }

    #[inline(always)]
    fn move_key(estimate: u32, gap: u32, operations: usize) -> u128 {
        ((estimate as u128) << 96) | ((gap as u128) << 64) | operations as u128
    }

    #[inline(always)]
    fn age_aspiration(value: u32, best: u32) -> u32 {
        let midpoint = ((value as u64 + best as u64 + (value > best) as u64) / 2) as u32;
        choose_u32(value != INF, midpoint, value)
    }

    #[inline(always)]
    fn swap_estimate(g: &Graph, snap: &Snap, m: usize, p: usize) -> u32 {
        let seq = unsafe { g.machine_seq.get_unchecked(m) };
        let a = unsafe { *seq.get_unchecked(p) };
        let b = unsafe { *seq.get_unchecked(p + 1) };
        let comp = |o: usize| g.he(o);
        let tailc = |o: usize| g.qe(o);
        let jp_comp = |o: usize| -> u32 {
            comp(o.saturating_sub(1)) & 0u32.wrapping_sub((g.oidx(o) > 0) as u32)
        };
        let js_tail = |o: usize| -> u32 {
            let j = g.ojob(o);
            tailc((o + 1).min(g.pre.n_ops - 1))
                & 0u32.wrapping_sub((g.oidx(o) + 1 < g.jnops(j)) as u32)
        };
        let (pe, qc) = snap.insertion(m);
        let pm_comp = unsafe { *pe.get_unchecked(p) };
        let sm_tail = unsafe { *qc.get_unchecked(p + 2) };
        let h_b = jp_comp(b).max(pm_comp);
        let h_a = jp_comp(a).max(h_b + g.du(b));
        let t_a = js_tail(a).max(sm_tail);
        let t_b = js_tail(b).max(g.du(a) + t_a);
        (h_a + g.du(a) + t_a).max(h_b + g.du(b) + t_b)
    }

    #[inline(always)]
    fn critical_path(g: &Graph, makespan: u32, path: &mut Vec<usize>) {
        let n = g.pre.n_ops;
        let crit = |o: usize| g.hd(o) + g.qe(o) == makespan;
        let mut start = usize::MAX;
        if g.dur_pos {
            for &o in g.pre.job_off.iter() {
                if crit(o) & (g.hd(o) == 0) {
                    start = o;
                    break;
                }
            }
        } else {
            for o in 0..n {
                if crit(o) & (g.hd(o) == 0) {
                    start = o;
                    break;
                }
            }
        }
        if start == usize::MAX {
            let mut best = u32::MAX;
            for o in 0..n {
                let h = g.hd(o);
                let take = crit(o) & (h < best);
                best = choose_u32(take, h, best);
                start = choose_usize(take, o, start);
            }
        }
        path.clear();
        let mut cur = start;
        loop {
            path.push(cur);
            let end = g.he(cur);
            let msc = unsafe { *g.ms.get_unchecked(cur) };
            let m = msc.min(n - 1);
            let j = (cur + 1).min(n - 1);
            let take_m = (msc != usize::MAX) & crit(m) & (g.hd(m) == end);
            let take_j = (g.oidx(cur) + 1 < g.jnops(g.ojob(cur))) & crit(j) & (g.hd(j) == end);
            let next = choose_usize(take_m, m, choose_usize(take_j, j, usize::MAX));
            if next == usize::MAX {
                break;
            }
            cur = next;
        }
        #[cfg(debug_assertions)]
        {
            fn reference(g: &Graph, makespan: u32, path: &mut Vec<usize>) {
                let n = g.pre.n_ops;
                let crit = |o: usize| g.hd(o) + g.qe(o) == makespan;
                let mut start = usize::MAX;
                if g.dur_pos {
                    for &o in g.pre.job_off.iter() {
                        if crit(o) && g.hd(o) == 0 {
                            start = o;
                            break;
                        }
                    }
                } else {
                    for o in 0..n {
                        if crit(o) && g.heads[o] == 0 {
                            start = o;
                            break;
                        }
                    }
                }
                if start == usize::MAX {
                    let mut best = u32::MAX;
                    for o in 0..n {
                        if crit(o) && g.heads[o] < best {
                            best = g.heads[o];
                            start = o;
                        }
                    }
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
                        if crit(v) && g.hd(v) == end {
                            next = v;
                        }
                    }
                    if next == usize::MAX {
                        if g.oidx(cur) + 1 < g.jnops(g.ojob(cur)) {
                            let v = cur + 1;
                            if crit(v) && g.hd(v) == end {
                                next = v;
                            }
                        }
                    }
                    if next == usize::MAX {
                        break;
                    }
                    cur = next;
                }
            }
            let mut expected = Vec::new();
            reference(g, makespan, &mut expected);
            debug_assert_eq!(*path, expected);
        }
    }

    #[inline(always)]
    fn n5_moves(
        g: &Graph,
        path: &[usize],
        moves: &mut Vec<(usize, usize)>,
        blk: &mut Vec<(u32, u32)>,
    ) {
        blk.clear();
        moves.clear();
        let mut s0 = 0usize;
        for i in 1..path.len() {
            let o = unsafe { *path.get_unchecked(i) };
            let prev = unsafe { *path.get_unchecked(i - 1) };
            if !(unsafe { *g.op_machine.get_unchecked(o) == *g.op_machine.get_unchecked(prev) }
                & unsafe { *g.mpos.get_unchecked(o) == *g.mpos.get_unchecked(prev) + 1 })
            {
                blk.push((s0 as u32, i as u32));
                s0 = i;
            }
        }
        if !path.is_empty() {
            blk.push((s0 as u32, path.len() as u32));
        }

        let nb = blk.len();
        for bi in 0..nb {
            let bs = blk[bi].0 as usize;
            let be = blk[bi].1 as usize;
            if be - bs < 2 {
                continue;
            }
            let m = g.op_machine[path[bs]];
            if bi != 0 {
                moves.push((m, g.mpos[path[bs]]));
            }
            if bi != nb - 1 {
                let p = g.mpos[path[be - 2]];
                if !(bi != 0 && p == g.mpos[path[bs]]) {
                    moves.push((m, p));
                }
            }
        }
    }

    #[inline(always)]
    fn best_second_counts(time: u32, avail: &[u32], elig: &[(usize, u32)]) -> (u32, u32, usize) {
        let mut best = INF;
        let mut second = INF;
        let mut cnt = 0usize;
        macro_rules! consider {
            ($candidate:expr) => {{
                let (m, pt) = $candidate;
                let end = time
                    .max(unsafe { *avail.get_unchecked(m) })
                    .saturating_add(pt);
                let lt = end < best;
                let eq = end == best;
                second = choose_u32(lt, best, choose_u32(!eq & (end < second), end, second));
                cnt = choose_usize(lt, 1, cnt + eq as usize);
                best = best.min(end);
            }};
        }
        let mut i = 0usize;
        while i + 1 < elig.len() {
            consider!(elig[i]);
            consider!(elig[i + 1]);
            i += 2;
        }
        if i < elig.len() {
            consider!(elig[i]);
        }
        second = choose_u32((cnt > 1) & (second >= INF), best, second);
        (best, second, cnt.max(1))
    }

    #[inline(always)]
    fn best_machine_move_mg(
        g: &Graph,
        snap: &Snap,
        path: &[usize],
        best_real: u32,
        re_exp: &[usize],
        it: usize,
        sweep_cap: usize,
        aspire: &[u32],
    ) -> Option<(usize, usize, usize, u32)> {
        let (mut bv, mut bk, mut bp, mut bl) = (0usize, 0usize, 0usize, u32::MAX);
        let mut best_key = move_key(u32::MAX, u32::MAX, usize::MAX);
        let mut seen = 0usize;
        for &v in path {
            let j = g.ojob(v);
            let oi = g.oidx(v);
            let elig = g.pre.eligible(v);
            if elig.len() < 2 {
                continue;
            }
            seen += 1;
            if seen > sweep_cap {
                break;
            }
            let m_cur = unsafe { *g.op_machine.get_unchecked(v) };
            let rpj = g.he(v.saturating_sub(1)) & 0u32.wrapping_sub((oi > 0) as u32);
            let qsj = g.qe((v + 1).min(g.pre.n_ops - 1))
                & 0u32.wrapping_sub((oi + 1 < g.jnops(j)) as u32);
            let is_tabu = unsafe { *re_exp.get_unchecked(v) > it };
            for &(k, pk) in elig {
                let gap = unsafe { *snap.gap.get_unchecked(k) };
                let admission = best_real.max(unsafe { *aspire.get_unchecked(k) });
                let limit = bl
                    .saturating_add(1)
                    .min(choose_u32(is_tabu, admission, u32::MAX));
                if (k == m_cur) | ((rpj + pk + qsj).max(pk + gap) >= limit) {
                    continue;
                }
                let (pe_s, qc_s) = snap.insertion(k);
                let n = pe_s.len() - 1;
                let (left, right) = ins_window(pe_s, qc_s, n, rpj, qsj);
                if left > right {
                    continue;
                }

                let mut pos = left;
                loop {
                    let hv = unsafe { *pe_s.get_unchecked(pos) }.max(rpj);
                    let tv = unsafe { *qc_s.get_unchecked(pos) }.max(qsj);
                    let l = hv + pk + tv;
                    let key = move_key(l, gap, n);
                    let take = (!is_tabu | (l < admission)) & (key < best_key);
                    bv = choose_usize(take, v, bv);
                    bk = choose_usize(take, k, bk);
                    bp = choose_usize(take, pos, bp);
                    bl = choose_u32(take, l, bl);
                    best_key = choose_u128(take, key, best_key);
                    let p2 = pos + 1;
                    let q2 = p2.min(n);
                    let hv2 = unsafe { *pe_s.get_unchecked(q2) }.max(rpj);
                    let tv2 = unsafe { *qc_s.get_unchecked(q2) }.max(qsj);
                    let l2 = hv2 + pk + tv2;
                    let key2 = move_key(l2, gap, n);
                    let take2 = (p2 <= right) & (!is_tabu | (l2 < admission)) & (key2 < best_key);
                    bv = choose_usize(take2, v, bv);
                    bk = choose_usize(take2, k, bk);
                    bp = choose_usize(take2, p2, bp);
                    bl = choose_u32(take2, l2, bl);
                    best_key = choose_u128(take2, key2, best_key);
                    pos += 2;
                    if pos > right {
                        break;
                    }
                }
            }
        }
        if bl == u32::MAX {
            None
        } else {
            Some((bv, bk, bp, bl))
        }
    }

    #[inline(always)]
    fn best_machine_swap_mg(
        g: &Graph,
        snap: &Snap,
        path: &[usize],
        best_real: u32,
        re_exp: &[usize],
        it: usize,
        sweep_cap: usize,
        aspire: &[u32],
    ) -> Option<(usize, usize, u32)> {
        let (mut bv, mut bu, mut bl) = (0usize, 0usize, u32::MAX);
        let mut best_slack = 0u32;
        let mut seen = 0usize;
        for &v in path {
            let jv = g.ojob(v);
            let oiv = g.oidx(v);
            let elig_v = g.pre.eligible(v);
            if elig_v.len() < 2 {
                continue;
            }
            seen += 1;
            if seen > sweep_cap {
                break;
            }
            let mv = unsafe { *g.op_machine.get_unchecked(v) };
            let pv = unsafe { *g.mpos.get_unchecked(v) };
            let rpv = g.he(v.saturating_sub(1)) & 0u32.wrapping_sub((oiv > 0) as u32);
            let qsv = g.qe((v + 1).min(g.pre.n_ops - 1))
                & 0u32.wrapping_sub((oiv + 1 < g.jnops(jv)) as u32);
            let v_tabu = unsafe { *re_exp.get_unchecked(v) > it };
            let durations = g.pre.durations_on(mv);
            let (he_mv, qe_mv) = snap.insertion(mv);
            let hu_base = unsafe { *he_mv.get_unchecked(pv) };
            let tu_base = unsafe { *qe_mv.get_unchecked(pv + 1) };
            for &(k, pvk) in elig_v {
                let seq = unsafe { g.machine_seq.get_unchecked(k) }.as_slice();
                let len = seq.len();
                let admission = best_real.max(unsafe { *aspire.get_unchecked(k) });
                let initial_limit = bl.min(choose_u32(v_tabu, admission, u32::MAX));
                let prune = bl
                    .saturating_add(1)
                    .min(choose_u32(v_tabu, admission, u32::MAX));
                if (k == mv) | (rpv + pvk + qsv >= prune) | (len == 0) {
                    continue;
                }
                let (pe_s, nq_s) = snap.insertion(k);
                let (left, right) = ins_window(pe_s, nq_s, len, rpv, qsv);
                let hi = right.min(len - 1);
                if left > hi {
                    continue;
                }
                let mut pos = left;
                let mut limit = initial_limit;
                let mut best_key = rank_key(limit, choose_u32(limit == bl, best_slack, u32::MAX));
                let mut changed = false;
                macro_rules! consider {
                    ($pos:expr) => {{
                        let pos = $pos;
                        let u = unsafe { *seq.get_unchecked(pos) };
                        let hv = unsafe { *pe_s.get_unchecked(pos) }.max(rpv);
                        let tv = unsafe { *nq_s.get_unchecked(pos + 1) }.max(qsv);
                        let l_v = hv + pvk + tv;
                        let pumv = unsafe { *durations.get_unchecked(u) };
                        let oiu = unsafe { *g.pre.op_idx.get_unchecked(u) };
                        let rpu = g.he(u.saturating_sub(1)) & 0u32.wrapping_sub((oiu > 0) as u32);
                        let qsu = g.qe((u + 1).min(g.pre.n_ops - 1))
                            & 0u32.wrapping_sub(
                                (unsafe { *g.js.get_unchecked(u) } != u32::MAX) as u32,
                            );
                        let l_u = hu_base
                            .max(rpu)
                            .wrapping_add(pumv)
                            .wrapping_add(tu_base.max(qsu));
                        let l = l_v.max(l_u);
                        let u_tabu = unsafe { *re_exp.get_unchecked(u) > it };
                        let slack = g.hd(u) + g.qe(u);
                        let key = rank_key(l, slack);
                        let take = (pumv != u32::MAX)
                            & (!(v_tabu | u_tabu) | (l < admission))
                            & (key < best_key);
                        best_slack = choose_u32(take, slack, best_slack);
                        best_key = choose_u64(take, key, best_key);
                        changed |= take;
                        bu = choose_usize(take, u, bu);
                        limit = choose_u32(take, l, limit);
                    }};
                }
                while pos < hi {
                    consider!(pos);
                    consider!(pos + 1);
                    pos += 2;
                }
                if pos <= hi {
                    consider!(pos);
                }
                bv = choose_usize(changed, v, bv);
                bl = choose_u32(changed, limit, bl);
            }
        }
        if bl == u32::MAX {
            None
        } else {
            Some((bv, bu, bl))
        }
    }

    #[inline(always)]
    fn live_bottleneck(work: &[u64], total: u64) -> usize {
        let mut machine = usize::MAX;
        let mut largest = 0;
        for (m, &w) in work.iter().enumerate() {
            machine = choose_usize(w > largest, m, machine);
            largest = largest.max(w);
        }
        choose_usize(
            (total != 0) & (largest * work.len() as u64 >= total + total / 8),
            machine,
            usize::MAX,
        )
    }

    #[inline(always)]
    fn smooth_width(k: usize, progress: f64) -> usize {
        1 + ((k as f64 - 1.0) * (1.0 - progress).max(0.0).powf(1.5)).round() as usize
    }

    #[inline(always)]
    fn block_emit(m: usize, start: usize, end: usize, dst: *mut (usize, usize), count: &mut usize) {
        let tail = (start + 3).max(end.saturating_sub(3));
        unsafe {
            dst.add(*count).write((m, start));
            dst.add(*count + 1).write((m, start + 1));
            dst.add(*count + 2).write((m, start + 2));
            dst.add(*count + 3).write((m, tail));
            dst.add(*count + 4).write((m, tail + 1));
            dst.add(*count + 5).write((m, tail + 2));
        }
        *count += (end - start).min(6);
    }

    #[inline(always)]
    fn block_node(
        g: &Graph,
        makespan: u32,
        m: usize,
        pos: usize,
        o: usize,
        run: &mut usize,
        prev_end: &mut u32,
        dst: *mut (usize, usize),
        count: &mut usize,
    ) {
        let h = g.hd(o);
        let critical = h + g.qe(o) == makespan;
        let link = critical & (*run > 0) & (h == *prev_end);
        if !link & (*run >= 2) {
            block_emit(m, pos - *run, pos - 1, dst, count);
        }
        *run = choose_usize(link, *run + 1, critical as usize);
        *prev_end = g.he(o);
    }

    #[inline(always)]
    fn block_moves(g: &Graph, makespan: u32, moves: &mut Vec<(usize, usize)>) {
        moves.clear();
        moves.reserve(g.pre.n_ops + 6);
        let dst = moves.as_mut_ptr();
        let mut count = 0usize;
        for (m, seq) in g.machine_seq.iter().enumerate() {
            let len = seq.len();
            let mut run = 0usize;
            let mut prev_end = 0u32;
            let mut pos = 0usize;
            if 1 < len {
                loop {
                    let (pos0, emit0, emit1, run0, run1, end) = loop {
                        let pos0 = pos;
                        let o0 = unsafe { *seq.get_unchecked(pos0) };
                        let o1 = unsafe { *seq.get_unchecked(pos0 + 1) };
                        let run0 = run;
                        let h0 = g.hd(o0);
                        let critical0 = h0 + g.qe(o0) == makespan;
                        let link0 = critical0 & (run0 > 0) & (h0 == prev_end);
                        let emit0 = !link0 & (run0 >= 2);
                        let run1 = choose_usize(link0, run0 + 1, critical0 as usize);
                        let end0 = g.he(o0);
                        let h1 = g.hd(o1);
                        let critical1 = h1 + g.qe(o1) == makespan;
                        let link1 = critical1 & (run1 > 0) & (h1 == end0);
                        let emit1 = !link1 & (run1 >= 2);
                        run = choose_usize(link1, run1 + 1, critical1 as usize);
                        prev_end = g.he(o1);
                        pos = pos0 + 2;
                        let end = pos + 1 >= len;
                        if (emit0 | emit1) | end {
                            break (pos0, emit0, emit1, run0, run1, end);
                        }
                    };
                    if emit0 {
                        block_emit(m, pos0 - run0, pos0 - 1, dst, &mut count);
                    }
                    if emit1 {
                        block_emit(m, pos0 + 1 - run1, pos0, dst, &mut count);
                    }
                    if end {
                        break;
                    }
                }
            }
            if pos < len {
                block_node(
                    g,
                    makespan,
                    m,
                    pos,
                    unsafe { *seq.as_ptr().add(pos) },
                    &mut run,
                    &mut prev_end,
                    dst,
                    &mut count,
                );
            }
            if run >= 2 {
                block_emit(m, seq.len() - run, seq.len() - 1, dst, &mut count);
            }
        }
        unsafe {
            moves.set_len(count);
        }
        #[cfg(debug_assertions)]
        {
            fn critical_blocks(
                g: &Graph,
                makespan: u32,
                mut visit: impl FnMut(usize, usize, usize),
            ) {
                for (m, seq) in g.machine_seq.iter().enumerate() {
                    let mut start = usize::MAX;
                    let mut prev_end = 0;
                    for (pos, &o) in seq.iter().enumerate() {
                        let h = g.hd(o);
                        let critical = h + g.qe(o) == makespan;
                        if !(critical && start != usize::MAX && h == prev_end) {
                            if start != usize::MAX && pos - start >= 2 {
                                visit(m, start, pos - 1);
                            }
                            start = if critical { pos } else { usize::MAX };
                        }
                        prev_end = if critical { g.he(o) } else { 0 };
                    }
                    if start != usize::MAX && seq.len() - start >= 2 {
                        visit(m, start, seq.len() - 1);
                    }
                }
            }

            fn reference(g: &Graph, makespan: u32, moves: &mut Vec<(usize, usize)>) {
                moves.clear();
                critical_blocks(g, makespan, |m, start, end| {
                    if end - start <= 6 {
                        for p in start..end {
                            moves.push((m, p));
                        }
                    } else {
                        for p in start..start + 3 {
                            moves.push((m, p));
                        }
                        for p in end - 3..end {
                            moves.push((m, p));
                        }
                    }
                });
            }
            let mut expected = Vec::new();
            reference(g, makespan, &mut expected);
            debug_assert_eq!(*moves, expected);
        }
    }

    pub(in super::super) const FS_HI: u64 = 1900;
    pub(in super::super) const FH_CLAMP: u64 = 1;
    pub(in super::super) const FH_RESERVE: u64 = 1000;

    struct LoadBias {
        machine: usize,
        share: f64,
        active: bool,
    }

    impl LoadBias {
        fn build(der: &Der) -> Self {
            let mut machine = 0;
            let mut load = f64::NEG_INFINITY;
            for (m, &w) in der.machine_load0.iter().enumerate() {
                if w > load {
                    load = w;
                    machine = m;
                }
            }
            let share = (load / der.avg_machine_load).clamp(1.0, 4.0);
            let active = share != 1.0 && der.machine_best_pop[machine] != 0.0;
            Self {
                machine,
                share,
                active,
            }
        }
    }

    fn assignment_construct(
        pre: &Pre,
        rng: Option<&mut SmallRng>,
    ) -> (Vec<usize>, Vec<Vec<usize>>) {
        let n = pre.num_jobs;
        let nm = pre.num_machines;
        let mut charge = vec![0u64; nm];
        let mut op_machine = vec![0usize; pre.n_ops];
        let mut order: Vec<(u32, usize)> = (0..pre.n_ops)
            .map(|o| {
                let j = pre.op_job[o];
                let oi = pre.op_idx[o];
                let mn = pre.job_ops[j][oi]
                    .iter()
                    .map(|&(_, t)| t)
                    .min()
                    .unwrap_or(0);
                (mn, o)
            })
            .collect();
        order.sort_unstable();
        let mut machine_flex = {
            let mut counts = vec![0u32; nm];
            for &(m, _) in &pre.elig_flat {
                counts[m] += 1;
            }
            counts
        };
        let jitter = rng.is_some();
        let mut rng = rng;
        for &(_, o) in &order {
            let j = pre.op_job[o];
            let oi = pre.op_idx[o];
            let mean_charge = charge.iter().sum::<u64>() / (nm as u64).max(1);
            let mut best_k = pre.job_ops[j][oi][0].0;
            let mut best_p = pre.job_ops[j][oi][0].1;
            let mut best_c = u64::MAX;
            for &(k, p) in &pre.job_ops[j][oi] {
                let c = {
                    let mut c = charge[k]
                        .saturating_add(p as u64)
                        .saturating_mul(4)
                        .saturating_add(charge[k].saturating_sub(mean_charge));
                    if jitter {
                        c = c.saturating_add(((rng.as_mut().unwrap().gen::<u8>() as u64) & 3) * 4);
                    }
                    c
                };
                let better = (c, machine_flex[k], p) < (best_c, machine_flex[best_k], best_p);
                if better {
                    best_c = c;
                    best_k = k;
                    best_p = p;
                }
            }
            op_machine[o] = best_k;
            charge[best_k] += best_p as u64;
            {
                for &(k, _) in &pre.job_ops[j][oi] {
                    machine_flex[k] = machine_flex[k].saturating_sub(1);
                }
            }
        }
        let mut job_next = vec![0usize; n];
        let mut job_ready = vec![0u32; n];
        let mut mach_avail = vec![0u32; nm];
        let mut machine_seq: Vec<Vec<usize>> = vec![Vec::new(); nm];
        let mut remaining: usize = pre.job_nops.iter().sum();
        let mut op_dur = vec![0u32; pre.n_ops];
        for o in 0..pre.n_ops {
            let j = pre.op_job[o];
            let oi = pre.op_idx[o];
            let m = op_machine[o];
            op_dur[o] = pre.job_ops[j][oi]
                .iter()
                .find(|&&(mm, _)| mm == m)
                .map(|&(_, t)| t)
                .unwrap();
        }
        while remaining > 0 {
            let mut best_end = u32::MAX;
            let mut best_j = usize::MAX;
            for j in 0..n {
                let oi = job_next[j];
                if oi >= pre.job_nops[j] {
                    continue;
                }
                let o = pre.op_id(j, oi);
                let m = op_machine[o];
                let end = job_ready[j].max(mach_avail[m]) + op_dur[o];
                if end < best_end || (end == best_end && j < best_j) {
                    best_end = end;
                    best_j = j;
                }
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

    fn tabu_run(
        g: &mut Graph,
        max_iter: usize,
        tenure: usize,
        full: bool,
        max_kicks: usize,
        fuel_floor: u64,
        rng: &mut SmallRng,
        hm: &HModel,
        use_gap: bool,
        abandon_above: u32,
        bias: &LoadBias,
    ) -> Option<u32> {
        let mut cur_real = g.eval()?;
        let mut hg = HGraph::new(hm);
        let mut inc = mirror_load(&mut hg, hm, g);
        if inc {
            match mirror_eval(g, &mut hg) {
                Some(mk) => {
                    debug_assert_eq!(mk, cur_real);
                    cur_real = mk;
                }
                None => {
                    inc = false;
                }
            }
            if !inc {
                cur_real = g.eval()?;
            }
        }
        let mut best_om: Vec<usize> = g.op_machine.clone();
        let mut best_ms: Vec<Vec<usize>> = g.machine_seq.clone();
        let mut best_real = cur_real;
        let mut re_exp: Vec<usize> = vec![0; g.pre.n_ops];
        let mut swap_tabu: HashMap<(usize, usize), usize> =
            HashMap::with_hasher(seeded_hasher(&g.pre.seed));
        let mut tabu_hi: Vec<usize> = vec![0; g.pre.n_ops];
        let mut no_improve = 0usize;
        let mut kick_failed = false;
        let mut aspire = { vec![INF; g.pre.num_machines] };
        let max_no_improve = if max_kicks > 0 {
            patience(g.pre)
        } else {
            8 * tenure + 400
        };
        let sweep_cap = sweep_cap(g.pre);
        let mut kicks_left = max_kicks;
        let n_ops = g.pre.n_ops;
        let mut snap = Snap::new(n_ops, g.machine_seq.len());
        let mut path: Vec<usize> = Vec::with_capacity(n_ops);
        let mut moves: Vec<(usize, usize)> = Vec::with_capacity(n_ops + 2);
        let mut blk: Vec<(u32, u32)> = Vec::with_capacity(n_ops);

        for it in 0..max_iter {
            let over = best_real.saturating_sub(abandon_above);
            let need = choose_usize(
                over > abandon_above / 50,
                1_500,
                choose_usize(over > abandon_above / 100, 3_000, 5_000),
            );
            let abandon = use_gap & (it > 0) & (best_real > abandon_above) & (no_improve >= need);
            if abandon || (fuel_left() <= fuel_floor) {
                break;
            }
            debug_assert!(re_exp.iter().zip(&tabu_hi).all(|(&r, &h)| r <= h));
            snap.fill(g, true);
            critical_path(g, cur_real, &mut path);
            if path.len() < 2 {
                break;
            }
            if full {
                block_moves(g, cur_real, &mut moves);
            } else {
                n5_moves(g, &path, &mut moves, &mut blk)
            };
            if moves.is_empty() {
                break;
            }
            let mut best_idx = 0usize;
            let mut best_est = INF;
            let mut best_key = rank_key(INF, 0);
            let mut fb_idx = 0usize;
            let mut fb_est = INF;
            for (idx, &(m, p)) in moves.iter().enumerate() {
                let seq = unsafe { g.machine_seq.get_unchecked(m) };
                let a = unsafe { *seq.get_unchecked(p) };
                let b = unsafe { *seq.get_unchecked(p + 1) };
                let est = swap_estimate(g, &snap, m, p);
                let is_tabu = if unsafe {
                    (*tabu_hi.get_unchecked(a) <= it) | (*tabu_hi.get_unchecked(b) <= it)
                } {
                    false
                } else {
                    swap_tabu
                        .get(&(a.min(b), a.max(b)))
                        .map_or(false, |&exp| exp > it)
                };
                debug_assert_eq!(
                    is_tabu,
                    swap_tabu
                        .get(&(a.min(b), a.max(b)))
                        .map_or(false, |&exp| exp > it)
                );
                let key = rank_key(est, g.qe(a).max(g.qe(b)));
                let admission = best_real.max(unsafe { *aspire.get_unchecked(m) });
                let take = (!is_tabu | (est < admission)) & (key < best_key);
                best_key = choose_u64(take, key, best_key);
                best_est = choose_u32(take, est, best_est);
                best_idx = choose_usize(take, idx, best_idx);
                let take_fb = est < fb_est;
                fb_est = choose_u32(take_fb, est, fb_est);
                fb_idx = choose_usize(take_fb, idx, fb_idx);
            }
            let mut best_swap = (best_est < INF).then_some(moves[best_idx]);
            let fb_swap = (fb_est < INF).then_some(moves[fb_idx]);
            let mm =
                best_machine_move_mg(g, &snap, &path, best_real, &re_exp, it, sweep_cap, &aspire);
            let mm_l = mm.map_or(INF, |(_, _, _, l)| l);
            let xs = if best_est.min(mm_l) >= cur_real {
                best_machine_swap_mg(g, &snap, &path, best_real, &re_exp, it, sweep_cap, &aspire)
            } else {
                None
            };
            let xs_l = xs.map_or(INF, |(_, _, l)| l);
            if best_swap.is_none() & mm.is_none() & xs.is_none() {
                if let Some((m, p)) = fb_swap {
                    best_swap = Some((m, p));
                    best_est = fb_est;
                } else {
                    break;
                }
            }
            {
                if let Some((m, _)) = best_swap {
                    aspire[m] = aspire[m].min(best_est);
                }
                if let Some((_, m, _, l)) = mm {
                    aspire[m] = aspire[m].min(l);
                }
                if let Some((_, u, l)) = xs {
                    let m = g.op_machine[u];
                    aspire[m] = aspire[m].min(l);
                }
            }
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
                debug_assert_eq!(
                    Some(g.mpos[v]),
                    g.machine_seq[mv].iter().position(|&x| x == v)
                );
                debug_assert_eq!(
                    Some(g.mpos[u]),
                    g.machine_seq[k].iter().position(|&x| x == u)
                );
                let pv = g.mpos[v];
                let pu = g.mpos[u];
                let dv_old = g.op_dur[v];
                let du_old = g.op_dur[u];
                let dv_new = g.pre.dur_first(v, k);
                let du_new = g.pre.dur_first(u, mv);
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
                tabu_hi[v] = tabu_hi[v].max(it + t_eff);
                tabu_hi[u] = tabu_hi[u].max(it + t_eff);
                xs_revert = Some((v, mv, pv, dv_old, u, k, pu, du_old));
            } else if mm_l < best_est {
                let (v, k, pos, _) = mm.unwrap();
                let m_old = g.op_machine[v];
                debug_assert_eq!(
                    Some(g.mpos[v]),
                    g.machine_seq[m_old].iter().position(|&x| x == v)
                );
                let p_old = g.mpos[v];
                let old_dur = g.op_dur[v];
                let dt = g.pre.dur_first(v, k);
                g.machine_seq[m_old].remove(p_old);
                g.op_machine[v] = k;
                g.op_dur[v] = dt;
                g.machine_seq[k].insert(pos, v);
                g.repair_machine(m_old);
                g.repair_machine(k);
                if inc {
                    hg_old = hg.reassign_at(v, k as u32, pos, dt);
                }
                re_exp[v] = it + t_eff;
                tabu_hi[v] = tabu_hi[v].max(it + t_eff);
                mg_revert = Some((v, m_old, p_old, old_dur));
            } else if let Some((m, p)) = best_swap {
                let a = g.machine_seq[m][p];
                let b = g.machine_seq[m][p + 1];
                let best_pair = (a.min(b), a.max(b));
                g.machine_seq[m].swap(p, p + 1);
                g.repair_adjacent(m, p);
                if inc {
                    hg.swap_adjacent(best_pair.0 as u32, best_pair.1 as u32);
                }
                swap_tabu.insert(best_pair, it + t_eff);
                tabu_hi[best_pair.0] = tabu_hi[best_pair.0].max(it + t_eff);
                tabu_hi[best_pair.1] = tabu_hi[best_pair.1].max(it + t_eff);
                swap_revert = Some((m, p));
            } else {
                break;
            }
            let ev = if inc {
                mirror_eval(g, &mut hg)
            } else {
                g.eval()
            };
            if inc {
                if let Some((v, _, _, _)) = mg_revert {
                    g.hend[v] = g.heads[v] + g.op_dur[v];
                }
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
                        if inc {
                            if let Some((t1, t2)) = xs_hg {
                                hg.undo_reassign(u, t2);
                                hg.undo_reassign(v, t1);
                            }
                        }
                    } else if let Some((v, m_old, p_old, old_dur)) = mg_revert {
                        let k = g.op_machine[v];
                        debug_assert_eq!(
                            Some(g.mpos[v]),
                            g.machine_seq[k].iter().position(|&x| x == v)
                        );
                        let pn = g.mpos[v];
                        g.machine_seq[k].remove(pn);
                        g.op_machine[v] = m_old;
                        g.op_dur[v] = old_dur;
                        g.machine_seq[m_old].insert(p_old, v);
                        g.repair_machine(k);
                        g.repair_machine(m_old);
                        if inc {
                            hg.undo_reassign(v, hg_old);
                        }
                    } else if let Some((m, p)) = swap_revert {
                        let a = g.machine_seq[m][p];
                        let b = g.machine_seq[m][p + 1];
                        g.machine_seq[m].swap(p, p + 1);
                        g.repair_adjacent(m, p);
                        if inc {
                            hg.swap_adjacent(a as u32, b as u32);
                        }
                    }
                    let ev2 = if inc {
                        mirror_eval(g, &mut hg)
                    } else {
                        g.eval()
                    };
                    if inc {
                        if let Some((v, _, _, _)) = mg_revert {
                            g.hend[v] = g.heads[v] + g.op_dur[v];
                        }
                        if let Some((v, _, _, _, u, _, _, _)) = xs_revert {
                            g.hend[v] = g.heads[v] + g.op_dur[v];
                            g.hend[u] = g.heads[u] + g.op_dur[u];
                        }
                    }
                    match ev2 {
                        Some(r) => r,
                        None => break,
                    }
                }
            };
            if cur_real < best_real {
                best_real = cur_real;
                best_om.copy_from_slice(&g.op_machine);
                for (dst, src) in best_ms.iter_mut().zip(g.machine_seq.iter()) {
                    dst.clear();
                    dst.extend_from_slice(src);
                }
                {
                    for e in &mut aspire {
                        *e = age_aspiration(*e, best_real);
                    }
                }
                no_improve = 0;
                kick_failed = false;
            } else {
                no_improve += 1;
                if no_improve >= max_no_improve {
                    if kicks_left == 0 || kick_failed {
                        break;
                    }
                    g.reset(&best_om, &best_ms);
                    kick_critical(g, 3 + (max_kicks - kicks_left), rng, bias);
                    if inc {
                        inc = mirror_load(&mut hg, hm, g);
                    }
                    cur_real = match if inc {
                        mirror_eval(g, &mut hg)
                    } else {
                        g.eval()
                    } {
                        Some(r) => r,
                        None => break,
                    };
                    re_exp.fill(0);
                    swap_tabu.clear();
                    for h in tabu_hi.iter_mut() {
                        *h = 0;
                    }
                    aspire.fill(INF);
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

    fn kick_critical(g: &mut Graph, strength: usize, rng: &mut SmallRng, bias: &LoadBias) {
        let cur = match g.eval() {
            Some(r) => r,
            None => return,
        };
        let mut path: Vec<usize> = Vec::new();
        critical_path(g, cur, &mut path);
        if path.len() < 2 {
            kick(g, strength, rng);
            return;
        }
        let mut cands: Vec<(usize, usize)> = Vec::with_capacity(path.len() * 2);
        for &u in &path {
            let m = g.op_machine[u];
            debug_assert_eq!(
                Some(g.mpos[u]),
                g.machine_seq[m].iter().position(|&x| x == u)
            );
            {
                let pos = g.mpos[u];
                if pos > 0 {
                    cands.push((m, pos - 1));
                }
                if pos + 1 < g.machine_seq[m].len() {
                    cands.push((m, pos));
                }
            }
        }
        cands.sort_unstable();
        cands.dedup();
        if cands.is_empty() {
            kick(g, strength, rng);
            return;
        }
        {
            let mut weighted = Vec::with_capacity(cands.len() * 2);
            for &c in &cands {
                weighted.push(c);
                if c.0 == bias.machine {
                    weighted.push(c);
                }
            }
            cands = weighted;
        }
        let mut applied = 0usize;
        let mut attempts = 0usize;
        while applied < strength && attempts < strength * 5 {
            attempts += 1;
            let (m, pos) = cands[rng.gen_range(0..cands.len())];
            if pos + 1 < g.machine_seq[m].len() {
                g.machine_seq[m].swap(pos, pos + 1);
                g.repair_machine(m);
                if g.eval().is_none() {
                    g.machine_seq[m].swap(pos, pos + 1);
                    g.repair_machine(m);
                } else {
                    applied += 1;
                }
            }
        }
        let _ = g.eval();
    }

    #[inline]
    fn best_second_counts_ref(
        time: u32,
        avail: &[u32],
        elig: &[(usize, u32)],
    ) -> (u32, u32, usize) {
        let mut best = INF;
        let mut second = INF;
        let mut cnt = 0usize;
        for &(m, pt) in elig {
            let end = time.max(avail[m]).saturating_add(pt);
            if end < best {
                second = best;
                best = end;
                cnt = 1;
            } else if end == best {
                cnt += 1;
            } else if end < second {
                second = end;
            }
        }
        if cnt > 1 && !(second < INF) {
            second = best;
        }
        (best, second, cnt.max(1))
    }

    fn fjsp_construct_ref(
        pre: &Pre,
        der: &Der,
        k: usize,
        rng: &mut SmallRng,
        route: Option<&[Vec<f32>]>,
        bias: &LoadBias,
    ) -> (Vec<usize>, Vec<Vec<usize>>) {
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
            let bottleneck = live_bottleneck(&work, sum_work);
            for &m in &idle {
                cands.clear();
                for j in 0..nj {
                    let oi = job_next[j];
                    if oi >= pre.job_nops[j] || job_ready[j] > time {
                        continue;
                    }
                    let elig = &pre.job_ops[j][oi];
                    let Some(&(_, pt)) = elig.iter().find(|&&(mm, _)| mm == m) else {
                        continue;
                    };
                    let (best_end, second_end, cnt) = best_second_counts_ref(time, &avail, elig);
                    if time.max(avail[m]).saturating_add(pt) != best_end {
                        continue;
                    }
                    let oid = pre.op_id(j, oi);
                    let flex = elig.len() as f64;
                    let flex_inv = 1.0 / flex;
                    let regret = if second_end >= INF {
                        der.avg_op_min * 2.6
                    } else {
                        (second_end - best_end) as f64
                    };
                    let reg_n = (regret / der.avg_op_min.max(1.0)).clamp(0.0, 6.0);
                    let scar_urg = 1.0 / (cnt as f64).max(1.0);
                    let load_n = load[m] / der.avg_machine_load.max(1e-9);
                    let scar_n = der.machine_scarcity[m] / der.avg_machine_scarcity;
                    let scarce_match = scar_n * (flex_inv - 1.0 / der.flex_avg);
                    let end_n = best_end as f64 / der.time_scale.max(1.0);
                    let proc_n = pt as f64 / der.avg_op_min.max(1.0);
                    let rem_min_n = der.suf_min[j][oi] as f64 / der.horizon.max(1.0);
                    let progress = 1.0 - remaining as f64 / (pre.n_ops as f64).max(1.0);
                    let end_w = (0.90 * fl + 0.72 * js)
                        + (0.62 + 0.12 * fl) * progress
                        + 0.18 * der.high_flex;
                    let reg_w = (0.50 * fl + 0.78 * js) + 0.18 * (1.0 - progress);
                    let load_w = if der.flex_avg >= 5.0 {
                        -(0.45 * fl + 0.75 * js) * der.flex_factor
                    } else {
                        (0.45 * fl + 0.75 * js) * der.flex_factor
                    };
                    let mut pop_pen = if der.chaotic_like && elig.len() >= 2 {
                        (0.07 + 0.15 * (1.0 - progress)).clamp(0.05, 0.24)
                            * der.machine_best_pop[m]
                            * der.flex_factor
                    } else {
                        0.0
                    };
                    if bias.active && m == bias.machine {
                        pop_pen *= bias.share;
                    }
                    let bal_pen = if der.chaotic_like {
                        let dw = (avg_work + (der.avg_op_min * 3.0).max(1.0)).max(1.0);
                        let r = work[m] as f64 / dw;
                        -0.07 * (r / (r + 1.0)).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let jitter = if use_rand {
                        rng.gen::<f64>() * 1e-9
                    } else {
                        0.0
                    };
                    let route_bonus =
                        route.map_or(0.0, |r| 0.90 * r[oid][m] as f64 * der.flex_factor);
                    let score = 1.05 * rem_min_n
                        + reg_w * der.flex_factor * reg_n
                        + 0.62 * der.flex_factor * flex_inv
                        + 0.55 * der.flex_factor * scarce_match
                        + load_w * load_n
                        + 0.20 * der.flex_factor * scar_urg
                        - end_w * end_n
                        - (0.18 * fl + 0.12 * js) * proc_n
                        - pop_pen
                        + bal_pen;
                    let score = {
                        score
                            - if m == bottleneck && elig.len() >= 2 {
                                (0.30 * der.flex_factor).min(0.55)
                            } else {
                                0.0
                            }
                    };
                    let score = score + route_bonus + jitter;
                    cands.push((score, j, pt, oid));
                }
                if cands.is_empty() {
                    continue;
                }
                let (_, j, pt, oid) = if use_rand {
                    cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
                    if der.chaotic_like {
                        let progress = 1.0 - remaining as f64 / (pre.n_ops as f64).max(1.0);
                        let k_eff = smooth_width(k, progress);
                        let kk = k_eff.max(1).min(cands.len()).min(8);
                        if kk <= 1 {
                            cands[0]
                        } else {
                            let smin = cands[kk - 1].0;
                            let mut wsum = 0.0f64;
                            for i in 0..kk {
                                let d = cands[i].0 - smin;
                                wsum += d * d;
                            }
                            if wsum <= 1e-18 {
                                cands[rng.gen_range(0..kk)]
                            } else {
                                let mut r = rng.gen::<f64>() * wsum;
                                let mut sel = kk - 1;
                                for i in 0..kk {
                                    let d = cands[i].0 - smin;
                                    let wgt = d * d;
                                    if r < wgt {
                                        sel = i;
                                        break;
                                    }
                                    r -= wgt;
                                }
                                cands[sel]
                            }
                        }
                    } else {
                        let kk = k.min(cands.len());
                        cands[rng.gen_range(0..kk)]
                    }
                } else {
                    *cands
                        .iter()
                        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(b.1.cmp(&a.1)))
                        .unwrap()
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
                for &(mm, _) in elig {
                    load[mm] = (load[mm] - delta).max(0.0);
                }
                remaining -= 1;
            }
            if remaining == 0 {
                break;
            }
            let mut next: Option<u32> = None;
            for &t in &avail {
                if t > time {
                    next = Some(next.map_or(t, |b| b.min(t)));
                }
            }
            for j in 0..nj {
                if job_next[j] < pre.job_nops[j] && job_ready[j] > time {
                    let t = job_ready[j];
                    next = Some(next.map_or(t, |b| b.min(t)));
                }
            }
            match next {
                Some(t) => time = t,
                None => break,
            }
        }
        (op_machine, machine_seq)
    }

    fn build_consensus(pre: &Pre, pool: &[PoolMember]) -> Vec<Vec<f32>> {
        let nm = pre.num_machines;
        let mut route = vec![vec![0f32; nm]; pre.n_ops];
        if pool.is_empty() {
            return route;
        }
        debug_assert!(pool.len() <= 24);
        for (oid, row) in route.iter_mut().enumerate() {
            for e in pool {
                row[e.om[oid]] += 1.0;
            }
            let mut sq = 0f32;
            for v in row.iter_mut() {
                *v *= *v;
                sq += *v;
            }
            if sq > 0.0 {
                let inv = 1.0f32 / sq;
                for v in row {
                    *v *= inv;
                }
            }
        }
        route
    }

    fn make_seed(
        pre: &Pre,
        der: &Der,
        idx: usize,
        rng: &mut SmallRng,
        bias: &LoadBias,
    ) -> (Vec<usize>, Vec<Vec<usize>>) {
        {
            if idx == 0 {
                fjsp_construct(pre, der, 0, rng, None, bias)
            } else if idx % 3 == 0 {
                assignment_construct(pre, Some(rng))
            } else {
                let k = 2 + (idx * 3) % 4;
                fjsp_construct(pre, der, k, rng, None, bias)
            }
        }
    }

    struct Best<'a> {
        mk: u32,
        om: Vec<usize>,
        ms: Vec<Vec<usize>>,
        save: &'a dyn Fn(&Solution) -> Result<()>,
    }

    impl Best<'_> {
        fn offer(&mut self, g: &Graph, mk: u32) -> Result<bool> {
            if mk >= self.mk {
                return Ok(false);
            }
            let s = g.to_solution();
            {
                if solution_makespan(g.pre, &s).is_none() {
                    return Ok(false);
                }
            }
            self.mk = mk;
            (self.save)(&s)?;
            self.om.copy_from_slice(&g.op_machine);
            for (d, sm) in self.ms.iter_mut().zip(g.machine_seq.iter()) {
                d.clear();
                d.extend_from_slice(sm);
            }
            Ok(true)
        }
    }

    fn solve_pool(
        challenge: &Challenge,
        pre: &Pre,
        der: &Der,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        rng: &mut SmallRng,
        bias: &LoadBias,
        scale: u64,
        clamp: bool,
        reserve_permille: u64,
    ) -> Result<()> {
        let tenure = 8 + pre.num_jobs / 4;
        let cap = 16usize;
        let elites = 6usize;
        let min_dist = (pre.n_ops / 16 + 2).max(2);
        let polish_iters = 4000usize;
        let deep_iters = 200000usize;

        let avail_fuel = fuel_left();
        let work_cap = FJSP_WORK_CAP_HIGH;
        let total0 = avail_fuel.saturating_sub(FUEL_STOP).min(work_cap);
        let mut floor_fuel = avail_fuel.saturating_sub(total0);
        let mut cal = Cal {
            f_ref: [0u64; 16],
            f_new: [0u64; 16],
            minus: 0,
            plus: 0,
        };

        let (om0, ms0) = fjsp_construct_ref(pre, der, 0, rng, None, bias);
        let mut g = Graph::new(pre, om0, ms0);
        let hm = HModel::build(challenge);
        let mut pool: Vec<PoolMember> = Vec::new();
        let mut best = Best {
            mk: INF,
            om: g.op_machine.clone(),
            ms: g.machine_seq.clone(),
            save: save_solution,
        };

        let a0_cap = 24usize;
        {
            let a0_fuel = (total0 * 15 / 100).max(15_000_000_000u64.min(total0 * 30 / 100));
            let a0_floor = floor_fuel + total0.saturating_sub(a0_fuel);
            let mut route: Vec<Vec<f32>> = Vec::new();
            let mut c = 0usize;
            while cal.left() > a0_floor && c < 4000 {
                if c % 24 == 0 && pool.len() >= 3 {
                    route = build_consensus(pre, &pool);
                }
                let k = 2 + (c % 5);
                let use_route = !route.is_empty() && c % 2 == 0;
                let (om, ms) = cal.construct(
                    pre,
                    der,
                    k,
                    rng,
                    if use_route { Some(&route[..]) } else { None },
                    bias,
                );
                g.reset(&om, &ms);
                if let Some(mk) = g.eval() {
                    best.offer(&g, mk)?;
                    pool_consider(
                        &mut pool,
                        pre,
                        mk,
                        &g.op_machine,
                        &g.machine_seq,
                        a0_cap,
                        min_dist,
                    );
                }
                c += 1;
            }
        }
        if pool.is_empty() {
            if let Some(mk) = g.eval() {
                pool_consider(
                    &mut pool,
                    pre,
                    mk,
                    &g.op_machine,
                    &g.machine_seq,
                    cap,
                    min_dist,
                );
            }
        }

        let rem_ref = cal.left().saturating_sub(floor_fuel);
        let search = (rem_ref as u128 * scale as u128 / 1000) as u64;
        floor_fuel = if clamp {
            let reserve = cal.completion_reserve(total0.saturating_sub(rem_ref), reserve_permille);
            let left = fuel_left();
            let search = search.min(left.saturating_sub(FUEL_STOP).saturating_sub(reserve));
            left.saturating_sub(search)
        } else {
            fuel_left().saturating_sub(search)
        };
        let total0 = (total0 as u128 * scale as u128 / 1000) as u64;
        let mut refresh_idx = 0usize;
        let mut kk = 2usize;
        let mut stall = 0usize;

        while fuel_left() > floor_fuel {
            let fuel_round_start = fuel_left();
            pool.sort_by(|a, b| {
                a.mk.cmp(&b.mk)
                    .then_with(|| a.om.cmp(&b.om))
                    .then_with(|| a.ms.cmp(&b.ms))
            });
            let e = elites.min(pool.len());
            let mut improved = false;

            for i in 0..e {
                if fuel_left() <= floor_fuel {
                    break;
                }
                if pool[i].deep {
                    continue;
                }
                let rem = fuel_left().saturating_sub(floor_fuel);
                let floor = (fuel_left() - rem / 3).max(floor_fuel);
                let abandon_above = if i > 0 && best.mk != INF {
                    best.mk + best.mk / 400
                } else {
                    INF
                };
                g.reset(&pool[i].om, &pool[i].ms);
                if let Some(mk) = tabu_run(
                    &mut g,
                    deep_iters,
                    tenure,
                    true,
                    8,
                    floor,
                    rng,
                    &hm,
                    true,
                    abandon_above,
                    bias,
                ) {
                    if mk < pool[i].mk {
                        pool[i].mk = mk;
                        pool[i].om.copy_from_slice(&g.op_machine);
                        for (d, sm) in pool[i].ms.iter_mut().zip(g.machine_seq.iter()) {
                            d.clear();
                            d.extend_from_slice(sm);
                        }
                        pool[i].deep = false;
                    } else {
                        pool[i].deep = true;
                    }
                    if best.offer(&g, mk)? {
                        improved = true;
                    }
                }
            }

            if fuel_left() > floor_fuel && best.mk != INF {
                let rem = fuel_left().saturating_sub(floor_fuel);
                let slice = (total0 * 6 / 100).min(rem);
                let floor = (fuel_left() - slice).max(floor_fuel);
                g.reset(&best.om, &best.ms);
                kick_critical(&mut g, kk, rng, bias);
                if let Some(mk) = tabu_run(
                    &mut g, deep_iters, tenure, true, 6, floor, rng, &hm, true, INF, bias,
                ) {
                    if best.offer(&g, mk)? {
                        improved = true;
                    }
                    pool_consider(
                        &mut pool,
                        pre,
                        mk,
                        &g.op_machine,
                        &g.machine_seq,
                        cap,
                        min_dist,
                    );
                }
            }

            if fuel_left() > floor_fuel {
                let use_consensus = {
                    let rem_budget = fuel_left().saturating_sub(floor_fuel);
                    let consumed = total0.saturating_sub(rem_budget);
                    !pool.is_empty() && consumed.saturating_mul(2) < total0
                };
                let (om, ms) = if use_consensus {
                    let route = build_consensus(pre, &pool);
                    fjsp_construct(pre, der, 3, rng, Some(&route), bias)
                } else {
                    make_seed(pre, der, refresh_idx, rng, bias)
                };
                refresh_idx += 1;
                g.reset(&om, &ms);
                if let Some(mk) = tabu_run(
                    &mut g,
                    polish_iters,
                    tenure,
                    false,
                    0,
                    floor_fuel,
                    rng,
                    &hm,
                    true,
                    INF,
                    bias,
                ) {
                    if best.offer(&g, mk)? {
                        improved = true;
                    }
                    pool_consider(
                        &mut pool,
                        pre,
                        mk,
                        &g.op_machine,
                        &g.machine_seq,
                        cap,
                        min_dist,
                    );
                }
            }

            if improved {
                stall = 0;
                kk = 2;
            } else {
                stall += 1;
                if stall % 3 == 0 {
                    kk = (kk + 1).min(8);
                }
            }

            if stall >= 3 {
                break;
            }
            if fuel_left() >= fuel_round_start {
                break;
            }
        }

        Ok(())
    }

    pub(in super::super) fn solve(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        der: &Der,
        common_route: bool,
        scale: u64,
        clamp: bool,
        reserve_permille: u64,
    ) -> Result<()> {
        debug_assert!(der.chaotic_like);
        let bias = LoadBias::build(der);

        let pre = Pre::build(challenge, common_route);
        let best_mk = std::cell::Cell::new(u32::MAX);
        let guarded = |s: &Solution| -> Result<()> {
            {
                if let Some(mk) = solution_makespan(&pre, s) {
                    if mk <= best_mk.get() {
                        best_mk.set(mk);
                        return save_solution(s);
                    }
                }
                return Ok(());
            }
        };
        let mut rng = SmallRng::from_seed(challenge.seed);
        solve_pool(
            challenge,
            &pre,
            der,
            &guarded,
            &mut rng,
            &bias,
            scale,
            clamp,
            reserve_permille,
        )?;
        greedy_floor(&pre, &guarded)?;
        Ok(())
    }

    #[inline(always)]
    fn fjsp_construct(
        pre: &Pre,
        der: &Der,
        k: usize,
        rng: &mut SmallRng,
        route: Option<&[Vec<f32>]>,
        bias: &LoadBias,
    ) -> (Vec<usize>, Vec<Vec<usize>>) {
        if k > 1 {
            fjsp_construct_impl::<true>(pre, der, k, rng, route, bias)
        } else {
            fjsp_construct_impl::<false>(pre, der, k, rng, route, bias)
        }
    }

    fn fjsp_construct_impl<const RANDOM: bool>(
        pre: &Pre,
        der: &Der,
        k: usize,
        rng: &mut SmallRng,
        route: Option<&[Vec<f32>]>,
        bias: &LoadBias,
    ) -> (Vec<usize>, Vec<Vec<usize>>) {
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
        let use_rand = RANDOM;
        let js = der.jobshopness;
        let fl = 1.0 - js;
        let mut cands: Vec<(f64, usize, u32, usize)> = Vec::with_capacity(nj);

        let load_w = if der.flex_avg >= 5.0 {
            -(0.45 * fl + 0.75 * js) * der.flex_factor
        } else {
            (0.45 * fl + 0.75 * js) * der.flex_factor
        };
        let load_scale = der.avg_machine_load.max(1e-9);
        let op_scale = der.avg_op_min.max(1.0);
        let time_scale = der.time_scale.max(1.0);
        let horizon = der.horizon.max(1.0);
        let flex_base = 1.0 / der.flex_avg;
        let proc_w = 0.18 * fl + 0.12 * js;
        let mut total_work = 0u64;
        while remaining > 0 {
            let sum_work: u64 = total_work;
            let avg_work = sum_work as f64 / (nm as f64).max(1.0);
            let bottleneck = live_bottleneck(&work, sum_work);
            for m in 0..nm {
                if avail[m] > time {
                    continue;
                }
                cands.clear();
                let progress = 1.0 - remaining as f64 / (pre.n_ops as f64).max(1.0);
                let limit = smooth_width(k, progress).max(1).min(8);
                let load_n = load[m] / load_scale;
                let scar_n = der.machine_scarcity[m] / der.avg_machine_scarcity;
                let end_w =
                    (0.90 * fl + 0.72 * js) + (0.62 + 0.12 * fl) * progress + 0.18 * der.high_flex;
                let reg_w = (0.50 * fl + 0.78 * js) + 0.18 * (1.0 - progress);
                let mut pop_m = {
                    (0.07 + 0.15 * (1.0 - progress)).clamp(0.05, 0.24)
                        * der.machine_best_pop[m]
                        * der.flex_factor
                };
                pop_m = choose_f64(bias.active & (m == bias.machine), pop_m * bias.share, pop_m);
                let bal_pen = {
                    let dw = (avg_work + (der.avg_op_min * 3.0).max(1.0)).max(1.0);
                    let r = work[m] as f64 / dw;
                    -0.07 * (r / (r + 1.0)).clamp(0.0, 1.0)
                };
                let start_m = time.max(avail[m]);
                let bottleneck_pen =
                    choose_f64(m == bottleneck, (0.30 * der.flex_factor).min(0.55), 0.0);
                for j in 0..nj {
                    let oi = unsafe { *job_next.get_unchecked(j) };
                    if unsafe {
                        (oi >= *pre.job_nops.get_unchecked(j))
                            | (*job_ready.get_unchecked(j) > time)
                    } {
                        continue;
                    }
                    let oid = unsafe { *pre.job_off.get_unchecked(j) + oi };
                    let pt = pre.dur_first(oid, m);
                    if pt == u32::MAX {
                        continue;
                    }
                    let elig = pre.eligible(oid);
                    let (best_end, second_end, cnt) = best_second_counts(time, &avail, elig);
                    if start_m.saturating_add(pt) != best_end {
                        continue;
                    }
                    let flex = elig.len() as f64;
                    let flex_inv = 1.0 / flex;
                    let regret = choose_f64(
                        second_end >= INF,
                        der.avg_op_min * 2.6,
                        second_end.wrapping_sub(best_end) as f64,
                    );
                    let reg_n = (regret / op_scale).clamp(0.0, 6.0);
                    let scar_urg = 1.0 / (cnt as f64).max(1.0);
                    let scarce_match = scar_n * (flex_inv - flex_base);
                    let end_n = best_end as f64 / time_scale;
                    let proc_n = pt as f64 / op_scale;
                    let rem_min_n =
                        unsafe { *der.suf_min.get_unchecked(j).get_unchecked(oi) } as f64 / horizon;
                    let pop_pen = choose_f64(elig.len() >= 2, pop_m, 0.0);
                    let jitter = if use_rand {
                        rng.gen::<f64>() * 1e-9
                    } else {
                        0.0
                    };
                    let route_bonus =
                        route.map_or(0.0, |r| 0.90 * r[oid][m] as f64 * der.flex_factor);
                    let score = 1.05 * rem_min_n
                        + reg_w * der.flex_factor * reg_n
                        + 0.62 * der.flex_factor * flex_inv
                        + 0.55 * der.flex_factor * scarce_match
                        + load_w * load_n
                        + 0.20 * der.flex_factor * scar_urg
                        - end_w * end_n
                        - proc_w * proc_n
                        - pop_pen
                        + bal_pen;
                    let score = { score - choose_f64(elig.len() >= 2, bottleneck_pen, 0.0) };
                    let score = score + route_bonus + jitter;
                    if RANDOM {
                        let c = (score, j, pt, oid);
                        let pos = cands.partition_point(|a| {
                            a.0.partial_cmp(&score).unwrap().is_gt() | ((a.0 == score) & (a.1 < j))
                        });
                        if pos < limit {
                            if cands.len() == limit {
                                cands.pop();
                            }
                            cands.insert(pos, c);
                        }
                    } else {
                        cands.push((score, j, pt, oid));
                    }
                }
                if cands.is_empty() {
                    continue;
                }
                let (_, j, pt, oid) = if use_rand {
                    {
                        let kk = limit.min(cands.len());
                        if kk <= 1 {
                            cands[0]
                        } else {
                            let smin = cands[kk - 1].0;
                            let mut wsum = 0.0f64;
                            for i in 0..kk {
                                let d = cands[i].0 - smin;
                                wsum += d * d;
                            }
                            if wsum <= 1e-18 {
                                cands[rng.gen_range(0..kk)]
                            } else {
                                let mut r = rng.gen::<f64>() * wsum;
                                let mut sel = kk - 1;
                                for i in 0..kk {
                                    let d = cands[i].0 - smin;
                                    let wgt = d * d;
                                    if r < wgt {
                                        sel = i;
                                        break;
                                    }
                                    r -= wgt;
                                }
                                cands[sel]
                            }
                        }
                    }
                } else {
                    *cands
                        .iter()
                        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(b.1.cmp(&a.1)))
                        .unwrap()
                };
                let start = time.max(avail[m]);
                let end = start + pt;
                op_machine[oid] = m;
                machine_seq[m].push(oid);
                job_next[j] += 1;
                job_ready[j] = end;
                avail[m] = end;
                work[m] += pt as u64;
                total_work += pt as u64;
                let elig = pre.eligible(oid);
                let delta = der.op_min[oid] as f64 / (elig.len() as f64).max(1.0);
                for &(mm, _) in elig {
                    load[mm] = (load[mm] - delta).max(0.0);
                }
                remaining -= 1;
            }
            if remaining == 0 {
                break;
            }
            let mut next = u32::MAX;
            let mut found = false;
            for &t in &avail {
                let take = t > time;
                next = next.min(choose_u32(take, t, u32::MAX));
                found |= take;
            }
            for j in 0..nj {
                let t = unsafe { *job_ready.get_unchecked(j) };
                let take = unsafe { *job_next.get_unchecked(j) < *pre.job_nops.get_unchecked(j) }
                    & (t > time);
                next = next.min(choose_u32(take, t, u32::MAX));
                found |= take;
            }
            if !found {
                break;
            }
            time = next;
        }
        (op_machine, machine_seq)
    }

    struct Cal {
        f_ref: [u64; 16],
        f_new: [u64; 16],
        minus: u64,
        plus: u64,
    }

    impl Cal {
        fn completion_reserve(&self, a0_ref: u64, permille: u64) -> u64 {
            let construct = self
                .f_ref
                .iter()
                .chain(&self.f_new)
                .copied()
                .max()
                .unwrap_or(0);
            let reserve = a0_ref.saturating_add(construct.saturating_mul(8));
            (reserve as u128 * permille as u128 / 1000).min(u64::MAX as u128) as u64
        }

        #[inline]
        fn left(&self) -> u64 {
            (fuel_left() + self.minus).saturating_sub(self.plus)
        }
        fn construct(
            &mut self,
            pre: &Pre,
            der: &Der,
            k: usize,
            rng: &mut SmallRng,
            route: Option<&[Vec<f32>]>,
            bias: &LoadBias,
        ) -> (Vec<usize>, Vec<Vec<usize>>) {
            let key = (k * 2 + route.is_some() as usize) & 15;
            if self.f_new[key] == 0 {
                let mut r2 = rng.clone();
                let f0 = fuel_left();
                let _ = fjsp_construct(pre, der, k, &mut r2, route, bias);
                let f1 = fuel_left();
                let res = fjsp_construct_ref(pre, der, k, rng, route, bias);
                let f2 = fuel_left();
                self.f_new[key] = (f0 - f1).max(1);
                self.f_ref[key] = f1 - f2;
                self.minus += f0 - f1;
                res
            } else {
                let f0 = fuel_left();
                let res = fjsp_construct(pre, der, k, rng, route, bias);
                let fd = f0 - fuel_left();
                self.minus += fd;
                self.plus +=
                    ((fd as u128 * self.f_ref[key] as u128) / self.f_new[key] as u128) as u64;
                res
            }
        }
    }

    fn solution_makespan(pre: &Pre, solution: &Solution) -> Option<u32> {
        if solution.job_schedule.len() != pre.num_jobs {
            return None;
        }
        let mut usage = vec![Vec::new(); pre.num_machines];
        let mut makespan = 0;
        for (j, schedule) in solution.job_schedule.iter().enumerate() {
            if schedule.len() != pre.job_nops[j] {
                return None;
            }
            let mut ready = 0;
            for (oi, &(m, start)) in schedule.iter().enumerate() {
                if m >= pre.num_machines || start < ready {
                    return None;
                }
                let pt = pre.job_ops[j][oi]
                    .iter()
                    .find(|&&(machine, _)| machine == m)?
                    .1;
                let end = start.checked_add(pt)?;
                usage[m].push((start, end));
                ready = end;
            }
            makespan = makespan.max(ready);
        }
        for machine in &mut usage {
            machine.sort_by_key(|&(start, _)| start);
            if machine.windows(2).any(|w| w[1].0 < w[0].1) {
                return None;
            }
        }
        Some(makespan)
    }
}
