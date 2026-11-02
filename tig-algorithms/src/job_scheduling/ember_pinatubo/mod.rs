use anyhow::Result;
use rand::{rngs::SmallRng, Rng};
use serde_json::{Map, Value};
use tig_challenges::job_scheduling::*;

mod fjsp_pool;
mod flow_shop_engine;
mod gate_floor;
mod hfs_construct;
mod hybrid_engine;
mod job_shop_engine;
mod sota;

const FUEL_STOP: u64 = 100_000_000;
static SOTA_SPENT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

extern "C" {
    static __fuel_remaining: u64;
}
#[inline]
fn fuel_left() -> u64 {
    unsafe { core::ptr::read_volatile(&__fuel_remaining) }
}

fn processing_row(op: &std::collections::HashMap<usize, u32>, num_machines: usize) -> Result<Vec<Option<u32>>> {
    let mut row = vec![None; num_machines];
    for (&m, &pt) in op {
        let slot = row.get_mut(m).ok_or_else(|| anyhow::anyhow!("machine id out of range"))?;
        *slot = Some(pt);
    }
    Ok(row)
}

fn checked_makespan(challenge: &Challenge, solution: &Solution) -> Result<u32> {
    if solution.job_schedule.len() != challenge.num_jobs {
        return Err(anyhow::anyhow!("invalid job count"));
    }
    let mut rows = Vec::with_capacity(challenge.product_processing_times.len());
    for ops in &challenge.product_processing_times {
        let mut product = Vec::with_capacity(ops.len());
        for op in ops { product.push(processing_row(op, challenge.num_machines)?); }
        rows.push(product);
    }
    let mut usage = vec![Vec::new(); challenge.num_machines];
    let mut makespan = 0;
    let mut job = 0;
    for (product, &count) in challenge.jobs_per_product.iter().enumerate() {
        let processing = rows.get(product).ok_or_else(|| anyhow::anyhow!("invalid product"))?;
        for _ in 0..count {
            let schedule = solution.job_schedule.get(job).ok_or_else(|| anyhow::anyhow!("invalid product expansion"))?;
            if schedule.len() != processing.len() {
                return Err(anyhow::anyhow!("invalid operation count"));
            }
            let mut ready = 0;
            for (oi, &(machine, start)) in schedule.iter().enumerate() {
                if start < ready { return Err(anyhow::anyhow!("job precedence violation")); }
                let pt = processing[oi].get(machine).copied().flatten()
                    .ok_or_else(|| anyhow::anyhow!("ineligible machine"))?;
                let end = start.checked_add(pt).ok_or_else(|| anyhow::anyhow!("time overflow"))?;
                usage[machine].push((start, end));
                ready = end;
            }
            makespan = makespan.max(ready);
            job += 1;
        }
    }
    if job != challenge.num_jobs { return Err(anyhow::anyhow!("invalid product expansion")); }
    for machine in &mut usage {
        machine.sort_by_key(|&(start, _)| start);
        if machine.windows(2).any(|w| w[1].0 < w[0].1) {
            return Err(anyhow::anyhow!("machine overlap"));
        }
    }
    Ok(makespan)
}

fn common_route(job_ops: &[Vec<Vec<(usize, u32)>>]) -> bool {
    let Some(first) = job_ops.first() else { return true; };
    job_ops.iter().skip(1).all(|route| {
        route.len() == first.len() && route.iter().zip(first).all(|(op, reference)| {
            op.len() == reference.len() && op.iter().zip(reference).all(|(a, b)| a.0 == b.0)
        })
    })
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
    common_route: bool,
    max_flex: usize,
    op_w: Vec<f64>,
    dur_pos: bool,
}

impl Pre {
    fn build(challenge: &Challenge) -> Result<Self> {
        let num_jobs = challenge.num_jobs;
        let mut job_ops: Vec<Vec<Vec<(usize, u32)>>> = Vec::with_capacity(num_jobs);
        for (product, &count) in challenge.jobs_per_product.iter().enumerate() {
            let ops = &challenge.product_processing_times[product];
            for _ in 0..count {
                let mut this_job = Vec::with_capacity(ops.len());
                for op in ops.iter() {
                    if op.is_empty() {
                        return Err(anyhow::anyhow!("operation has no eligible machine"));
                    }
                    let cands = processing_row(op, challenge.num_machines)?
                        .into_iter().enumerate()
                        .filter_map(|(m, t)| t.map(|t| (m, t))).collect();
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
        let common_route = common_route(&job_ops);
        Ok(Pre {
            num_jobs, num_machines: challenge.num_machines, job_ops, job_nops,
            n_ops: off, job_off, op_job, op_idx, common_route, max_flex,
            op_w, dur_pos,
        })
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

struct Graph<'a> {
    pre: &'a Pre,
    op_machine: Vec<usize>,
    op_dur: Vec<u32>,
    machine_seq: Vec<Vec<usize>>,
    heads: Vec<u32>,
    topo: Vec<usize>,
    mp: Vec<usize>,
    ms: Vec<usize>,
    indeg0: Vec<u8>,
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
        let mut g = Graph {
            pre, op_machine, op_dur, machine_seq,
            heads: vec![0; n],
            topo: vec![0; n],
            mp: vec![usize::MAX; n], ms: vec![usize::MAX; n],
            indeg0: vec![0; n],
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
        for oid in 0..self.op_machine.len() {
            let j = self.pre.op_job[oid];
            let o = self.pre.op_idx[oid];
            let m = self.op_machine[oid];
            let d = self.pre.job_ops[j][o].iter().find(|&&(mm, _)| mm == m)
                .map(|&(_, t)| t).unwrap_or(0);
            self.op_dur[oid] = d;
        }
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
        for o in 0..n {
            let r = self.nd[o];
            self.heads[o] = r[0];
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
        let jobshopness = if chaotic_like { 0.85 } else { 0.55 };
        let time_scale = (horizon * (2.65 + 0.10 * jobshopness + 0.10 * high_flex)).max(1.0);
        Der {
            op_min, suf_min, machine_load0, machine_scarcity, machine_best_pop,
            avg_machine_load, avg_machine_scarcity, avg_op_min, horizon, time_scale,
            flex_avg, flex_factor, high_flex, jobshopness, chaotic_like, use_rich,
        }
    }
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

struct Triage<'a> {
    fuel0: std::cell::Cell<u64>,
    checkpoint: u64,
    lo: u64,
    hi: u64,
    bar: u64,
    boost: u64,
    sota: std::cell::Cell<Option<u32>>,
    early: std::cell::Cell<Option<u32>>,
    early_point: std::cell::Cell<Option<(u64, u64)>>,
    early_done: std::cell::Cell<bool>,
    high: Option<(u64, u64, u64)>,
    rearmed: std::cell::Cell<bool>,
    window: Option<(u64, u64, u64)>,
    level: Option<i128>,
    xq: Option<u64>,
    challenge: &'a Challenge,
    best_mk: &'a std::cell::Cell<u32>,
    class: std::cell::Cell<Option<u8>>,
}

impl<'a> Triage<'a> {
    fn new(challenge: &'a Challenge, best_mk: &'a std::cell::Cell<u32>,
        tc: u64, lo: u64, hi: u64, bar: u64, boost: u64,
        early_point: Option<(u64, u64)>) -> Result<Option<Self>> {
        if tc == 0 { return Ok(None); }
        let fuel0 = fuel_left();
        let sota = None;
        Ok(Some(Self {
            fuel0: std::cell::Cell::new(fuel0), checkpoint: tc.saturating_mul(1_000_000_000), lo, hi, bar, boost,
            sota: std::cell::Cell::new(sota), window: None, level: None, xq: None,
            early: std::cell::Cell::new(None), early_point: std::cell::Cell::new(early_point), early_done: std::cell::Cell::new(false),
            high: None, rearmed: std::cell::Cell::new(false),
            challenge, best_mk, class: std::cell::Cell::new(None),
        }))
    }

    fn check_at(&self, mut fuel: u64) -> bool {
        if let Some((checkpoint, threshold)) = self.early_point.get() {
            if !self.early_done.get() && self.class.get().is_none()
                && self.fuel0.get().saturating_sub(fuel) >= checkpoint
            {
                self.early_done.set(true);
                if self.rearmed.get() {
                    if let (Some((step, _, high_threshold)), Some(sota)) = (self.high, self.early.get()) {
                        let quality = (sota as f64 - self.best_mk.get() as f64) / sota as f64;
                        let q = (quality.clamp(-10.0, 10.0) * 1_000_000.0).round() as i32;
                        let high = (q as i128) > high_threshold as i128;
                        let rearm = !high && checkpoint + step < self.checkpoint;
                        if high { self.class.set(Some(2)); }
                        else if rearm {
                            self.early_point.set(Some((checkpoint + step, threshold)));
                            self.early_done.set(false);
                        }
                    }
                } else {
                    let elapsed = self.fuel0.get().saturating_sub(fuel);
                    let sota_before = fuel_left();
                    let sota = self.sota.get().or_else(|| sota::sota_makespan(self.challenge));
                    let spent = sota_before.saturating_sub(fuel_left());
                    SOTA_SPENT.fetch_add(spent, std::sync::atomic::Ordering::Relaxed);
                    self.fuel0.set(self.fuel0.get().saturating_sub(spent));
                    fuel = fuel.saturating_sub(spent);
                    debug_assert_eq!(self.fuel0.get().saturating_sub(fuel), elapsed);
                    self.early.set(sota);
                    let q = sota.map(|sota| {
                        let quality = (sota as f64 - self.best_mk.get() as f64) / sota as f64;
                        (quality.clamp(-10.0, 10.0) * 1_000_000.0).round() as i32
                    });
                    let cut = q.map_or(false, |q| (q as i128) < threshold as i128);
                    if cut { self.class.set(Some(0)); }
                    if !cut {
                        if let (Some((step, floor, high_threshold)), Some(q)) = (self.high, q) {
                            let high = (q as i128) > high_threshold as i128;
                            let rearm = !high && (q as i128) >= floor as i128
                                && checkpoint + step < self.checkpoint;
                            if high { self.class.set(Some(2)); }
                            else if rearm {
                                self.rearmed.set(true);
                                self.early_point.set(Some((checkpoint + step, threshold)));
                                self.early_done.set(false);
                            }
                        }
                    }
                }
            }
        }
        if self.class.get().is_none() && self.fuel0.get().saturating_sub(fuel) >= self.checkpoint {
            let best_mk = self.best_mk.get();
            let quality = |sota: u32| {
                let quality = (sota as f64 - best_mk as f64) / sota as f64;
                (quality.clamp(-10.0, 10.0) * 1_000_000.0).round() as i32
            };
            let sota_before = fuel_left();
            let upper = sota::sota_upper(self.challenge);
            let q = upper.and_then(|upper| {
                let q = quality(upper);
                if (q as i128) < self.bar as i128 - self.lo as i128 { Some(q) }
                else {
                    self.early.get().or_else(|| self.sota.get()).or_else(|| sota::sota_makespan(self.challenge)).map(|sota| {
                        self.sota.set(Some(sota));
                        quality(sota)
                    })
                }
            });
            SOTA_SPENT.fetch_add(sota_before.saturating_sub(fuel_left()), std::sync::atomic::Ordering::Relaxed);
            let class = match q {
                Some(q) if (q as i128) < self.bar as i128 - self.lo as i128 => 0,
                Some(q) if (q as i128) > self.bar as i128 + self.hi as i128 => 2,
                _ => 1,
            };
            self.class.set(Some(class));
        }
        matches!(self.class.get(), Some(0) | Some(2))
    }

    fn stop(&self) -> bool {
        if self.class.get().is_none() { return self.check_at(fuel_left()); }
        matches!(self.class.get(), Some(0) | Some(2))
    }

    fn uncertain(&self) -> bool {
        self.class.get() == Some(1)
    }

    fn quality(&self, mk: u32) -> Option<i32> {
        self.sota.get().map(|sota| {
            let quality = (sota as f64 - mk as f64) / sota as f64;
            (quality.clamp(-10.0, 10.0) * 1_000_000.0).round() as i32
        })
    }

    fn in_window(&self, mk: u32, wlo: u64, whi: u64) -> bool {
        self.uncertain() && self.quality(mk).map_or(false, |q| {
            self.bar as i128 - wlo as i128 <= q as i128
                && (q as i128) < self.bar as i128 + whi as i128
        })
    }

    fn scale(&self, iterations: usize) -> usize {
        if self.class.get() != Some(1) { return iterations; }
        ((iterations as u128 * self.boost as u128 / 100).min(usize::MAX as u128)) as usize
    }

}



pub fn help() {
    println!("job_scheduling (c007)");
    println!();
    println!("fs_hi: default 1900; range 0..=u64::MAX.\nfs_med: default 1200; range 0..=u64::MAX.");
    println!("fh_clamp: default 1; range 0..=u64::MAX.\nfh_reserve: default 1000; range 0..=u64::MAX.");
    println!("fh_fs: default 690; range 1..=4000.");
    println!("fh_rd: default 3; range 2..=24; invalid values use 3.");
    println!("fh_ne: default 1; range 0..=6; invalid values use 1.");
    println!("fh_tri: default 1; range 0..=1.");
    println!("fh_tc: default 400; range 1..=1000.");
    println!("fh_bd: default 80503; range 1..=10000000.");
    println!("fh_w0: default 2301; range 0..=100000; invalid values use 2301.");
    println!("fh_rs: default 1; range 0..=1; invalid values use 1.");
    println!("fh_rb: default 81500; range 1..=10000000; invalid values use 81500.");
    println!("fh_rw: default 625; range 1..=100000; invalid values use 625.");
    println!("fh_rf: default 1000; range 1..=4000; invalid values use 1000.");
    println!("fh_hs: default 85000; range 0..=10000000; 0 disables; invalid values use 85000.");
    println!("fh_hx: default 250; range 0..=1000; invalid values use 250.");
    println!("fh_hu: default 87300; range 1..=10000000; invalid values use 87300.");
    println!("fh_hq: default 85000; range 0..=1000000; 0 disables; invalid values use 85000.");
    println!("fh_ht: default 1; range 0..=1; 0 disables; invalid values use 1.");
    println!("fh_xf: default 250; range 0..=1000; invalid values use 250.");
    println!("fh_xl: default 82000; range 0..=10000000; invalid values use 82000.");
    println!("fh_xh: default 84500; range 0..=10000000; invalid values use 84500.");
    println!("fh_zl: default 85500; range 0..=10000000; invalid values use 85500.");
    println!("fh_zh: default 87263; range 0..=10000000; invalid values use 87263.");
    println!("fh_zf: default 60; range 0..=1000; invalid values use 60.");
    println!("fh_wf: default 250; range 0..=1000; invalid values use 250.");
    println!("fh_wl: default 82000; range 0..=10000000; invalid values use 82000.");
    println!("fh_wh: default 83000; range 0..=10000000; invalid values use 83000.");
    println!("fh_vf: default 250; range 0..=1000; invalid values use 250.");
    println!("fh_vl: default 85000; range 0..=10000000; invalid values use 85000.");
    println!("fh_vh: default 85500; range 0..=10000000; invalid values use 85500.");
    println!("fh_sr: default 1; range 0..=1; invalid values use 1.");
    println!("fh_sq: default 1; range 0..=1; invalid values use 1.");
    println!("fm_clamp: default 1; range 0..=u64::MAX.\nfm_reserve: default 1000; range 0..=u64::MAX.");
    println!("fm_pat: default 450000; range 1..=u64::MAX.");
    println!("fm_deep: default 800000; range 1..=u64::MAX.");
    println!("fm_slice: default 2; range 1..=u64::MAX.");
    println!("fm_take: default 300; range 1..=900.");
    println!("fh_alt: default 1; range 0..=1.");
    println!("fm_tdiv: default 4; range 2..=16.");
    println!("fl_x: default 100; range 1..=1000.");
    println!("fl_lb: default 1; range 0..=1.");
    println!("fl_rand: default 100; range 1..=100.");
    println!("fl_ins: default 2; range 1..=2.");
    println!("fl_seg: default 6; range 1..=6.");
    println!("fl_swp: default 1000; range 1..=1000.");
    println!("fl_nd: default 2; range 1..=3.");
    println!("fl_dp: default 250; range 1..=1000.");
    println!("fl_gr: default 1000; range 1..=1000.");
    println!("fl_triage: default 1; range 0..=1; invalid values use 1.");
    println!("fl_tc: default 4250; range 1..=17000; invalid values use 4250.");
    println!("fl_tlo: default 36000; range 0..=10000000; invalid values use 36000.");
    println!("fl_sr: default 10; range 0..=200; invalid values use 10.");
    println!("fl_ps: default 1; range 0..=1; invalid values use 1.");
    println!("hy_x: default 100; range 1..=1000.");
    println!("hy_triage: default 1; range 0..=1.");
    println!("hy_tc: default 72; range 1..=1000.");
    println!("hy_tlo: default 8000; range 1..=10000000.");
    println!("hy_thi: default 8000; range 1..=10000000.");
    println!("hy_bar: default 99193; range 1..=10000000.");
    println!("hy_win: default 1; range 0..=1.");
    println!("hy_wlo: default 5000; range 1..=10000000.");
    println!("hy_whi: default 2500; range 1..=10000000.");
    println!("hy_tw: default 400; range 1..=1000.");
    println!("hy_ea: default 1; range 0..=1; invalid values use 1.");
    println!("hy_e0: default 20; range 1..=1000; invalid values use 20.");
    println!("hy_et: default 75000; range 1..=10000000; invalid values use 75000.");
    println!("hy_h0: default 1; range 0..=1; invalid values use 1.");
    println!("hy_hs: default 10; range 1..=1000; invalid values use 10.");
    println!("hy_hl: default 90000; range 0..=10000000; invalid values use 90000.");
    println!("hy_hq: default 110000; range 1..=10000000; invalid values use 110000.");
    println!("js_x: default 100; range 1..=1000.");
    println!("js_triage: default 1; range 0..=1.");
    println!("js_tc: default 62; range 1..=1000.");
    println!("js_e0tc: default 10; range 1..=1000; invalid values use 10; disabled when js_e0tc >= js_tc.");
    println!("js_e0: default 62000; range 1..=10000000; invalid values use 62000.");
    println!("js_no_e0: default 0; range 0..=1; 1 disables early triage; invalid values use 0.");
    println!("js_h0: default 1; range 0..=1; 0 disables; invalid values use 1.");
    println!("js_hs: default 2; range 1..=1000; invalid values use 2.");
    println!("js_hl: default 90000; range 0..=10000000; invalid values use 90000.");
    println!("js_hq: default 107000; range 1..=10000000; invalid values use 107000.");
    println!("js_hk: default 1; range 0..=1; 0 disables; invalid values use 1.");
    println!("js_xq: default 107000; range 1..=10000000; invalid values use 107000.");
    println!("js_tlo: default 16000; range 1..=10000000.");
    println!("js_thi: default 5059; range 1..=10000000.");
    println!("js_bar: default 101941; range 1..=10000000.");
    println!("js_tb: default 100; range 1..=10000.");
    println!("js_tx: default 200; range 1..=100000.");
    println!("js_xs: default 1; range 0..=1.");
    println!("js_win: default 1; range 0..=1.");
    println!("js_wlo: default 2000; range 1..=10000000.");
    println!("js_whi: default 1000; range 1..=10000000.");
    println!("js_tw: default 300; range 1..=100000.");
    println!("js_nr: default 0; range 0..=2000.");
    println!("js_nb: default 0; range 0..=10000.");
    println!("no_take: default 0; range 0..=1.");
    println!("The fuel budget is read from `__fuel_remaining`; the solver is anytime.");
}

fn run_bounded(
    work_cap: u64,
    guarded: &dyn Fn(&Solution) -> Result<()>,
    engine: impl FnOnce(&dyn Fn(&Solution) -> Result<()>) -> Result<()>,
) {
    let fuel_start = fuel_left();
    let floor_fuel = fuel_start.saturating_sub(work_cap.min(fuel_start.saturating_sub(FUEL_STOP)));
    let bound = |s: &Solution| -> Result<()> {
        guarded(s)?;
        if fuel_left() <= floor_fuel {
            return Err(anyhow::anyhow!("stop"));
        }
        Ok(())
    };
    let _ = engine(&bound);
}

fn routes_to_hybrid(pre: &Pre, der: Option<&Der>) -> bool {
    match der {
        Some(d) => !d.chaotic_like && pre.max_flex > 1 && pre.common_route,
        None => false,
    }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    SOTA_SPENT.store(0, std::sync::atomic::Ordering::Relaxed);
    let save_error = std::cell::RefCell::new(None);
    let best_mk = std::cell::Cell::new(u32::MAX);
    let guarded = |s: &Solution| -> Result<()> {
        if let Ok(mk) = checked_makespan(challenge, s) {
            if mk <= best_mk.get() {
                best_mk.set(mk);
                return save_solution(s).map_err(|error| {
                    let mut first_error = save_error.borrow_mut();
                    if first_error.is_none() { *first_error = Some(error); }
                    anyhow::anyhow!("save_solution failed")
                });
            }
        }
        Ok(())
    };
    let result = solve_inner(challenge, &guarded, hyperparameters, &best_mk);
    match save_error.into_inner() {
        Some(error) => Err(error),
        None => result,
    }
}

fn solve_inner(
    challenge: &Challenge,
    guarded: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    best_mk: &std::cell::Cell<u32>,
) -> Result<()> {
    let pre = Pre::build(challenge)?;
    let der = if pre.max_flex > 1 { Some(Der::build(&pre)) } else { None };
    if !routes_to_hybrid(&pre, der.as_ref()) {
        let lean = pre.max_flex == 1 && !pre.common_route && fuel_left() < 20_000_000_000u64;
        let flow = pre.max_flex == 1 && pre.common_route;
        if !lean && !flow {
            gate_floor::gate_floor(&pre, &challenge.seed, &guarded)?;
        }
    }

    if pre.max_flex == 1 && pre.common_route {
        const FLOW_CAP: u64 = 10_000_000_000u64;
        let fl_x = hyperparameters.as_ref().and_then(|m| m.get("fl_x")).and_then(|v| v.as_u64()).unwrap_or(100);
        let fl_x = if (1..=1000).contains(&fl_x) { fl_x } else { 100 };
        let fl_lb = hyperparameters.as_ref().and_then(|m| m.get("fl_lb")).and_then(|v| v.as_u64()).unwrap_or(1) != 0;
        let work_hp = |key: &str, default: u64, max: u64| {
            hyperparameters.as_ref().and_then(|m| m.get(key)).and_then(|v| v.as_u64())
                .filter(|v| (1..=max).contains(v)).unwrap_or(default) as usize
        };
        let mut work = flow_shop_engine::Work {
            random_orders: work_hp("fl_rand", 100, 100),
            insertion_passes: work_hp("fl_ins", 2, 2),
            segment_passes: work_hp("fl_seg", 6, 6),
            swap_permille: work_hp("fl_swp", 1000, 1000),
            descent_starts: work_hp("fl_nd", 2, 3),
            descent_permille: work_hp("fl_dp", 250, 1000),
            grasp_permille: work_hp("fl_gr", 1000, 1000),
            triage_tc: 0,
            triage_lo: 0,
            triage_sota: 0,
            post_skip: false,
            triage_cut: std::cell::Cell::new(false),
        };
        let fl_triage = hyperparameters.as_ref().and_then(|m| m.get("fl_triage")).and_then(|v| v.as_u64())
            .filter(|v| (0..=1).contains(v)).unwrap_or(1);
        let fl_tc = work_hp("fl_tc", 4250, 17000);
        let fl_tlo = hyperparameters.as_ref().and_then(|m| m.get("fl_tlo")).and_then(|v| v.as_u64())
            .filter(|v| (0..=10_000_000).contains(v)).unwrap_or(36000);
        let fl_sr = hyperparameters.as_ref().and_then(|m| m.get("fl_sr")).and_then(|v| v.as_u64())
            .filter(|v| (0..=200).contains(v)).unwrap_or(10) as usize;
        let fl_ps = hyperparameters.as_ref().and_then(|m| m.get("fl_ps")).and_then(|v| v.as_u64())
            .filter(|v| (0..=1).contains(v)).unwrap_or(1);
        let fl_sota = if fl_triage == 1 { sota::sota_partial(challenge, fl_sr) } else { None };
        let armed = fl_triage == 1 && fl_sota.is_some();
        work.triage_tc = if armed { fl_tc } else { 0 };
        work.triage_lo = fl_tlo;
        work.triage_sota = fl_sota.unwrap_or(0);
        work.post_skip = fl_ps == 1;
        run_bounded(FLOW_CAP * fl_x / 100, &guarded, |bound| flow_shop_engine::solve_challenge(challenge, bound, fl_lb, &work));
        gate_floor::gate_floor(&pre, &challenge.seed, &guarded)?;
        return greedy_floor(&pre, &guarded);
    }

    if pre.max_flex == 1 && !pre.common_route {
        let js_x = hyperparameters.as_ref().and_then(|m| m.get("js_x")).and_then(|v| v.as_u64()).unwrap_or(100);
        let js_x = if (1..=1000).contains(&js_x) { js_x as usize } else { 100 };
        let js_nr = hyperparameters.as_ref().and_then(|m| m.get("js_nr")).and_then(|v| v.as_u64()).unwrap_or(0);
        let js_nr = if (1..=2000).contains(&js_nr) { js_nr as usize } else { 0 };
        let js_nb = hyperparameters.as_ref().and_then(|m| m.get("js_nb")).and_then(|v| v.as_u64()).unwrap_or(0);
        let js_nb = if (1..=10000).contains(&js_nb) { js_nb as usize } else { 0 };
        let js_triage = hyperparameters.as_ref().and_then(|m| m.get("js_triage")).and_then(|v| v.as_u64()).unwrap_or(1) != 0;
        let js_tc = hyperparameters.as_ref().and_then(|m| m.get("js_tc")).and_then(|v| v.as_u64()).unwrap_or(62);
        let js_tc = if (1..=1000).contains(&js_tc) { js_tc } else { 62 };
        let js_e0tc = hyperparameters.as_ref().and_then(|m| m.get("js_e0tc")).and_then(|v| v.as_u64()).unwrap_or(10);
        let js_e0tc = if (1..=1000).contains(&js_e0tc) { js_e0tc } else { 10 };
        let js_e0 = hyperparameters.as_ref().and_then(|m| m.get("js_e0")).and_then(|v| v.as_u64()).unwrap_or(62000);
        let js_e0 = if (1..=10_000_000).contains(&js_e0) { js_e0 } else { 62000 };
        let js_no_e0 = hyperparameters.as_ref().and_then(|m| m.get("js_no_e0")).and_then(|v| v.as_u64()).unwrap_or(0);
        let js_no_e0 = if (0..=1).contains(&js_no_e0) { js_no_e0 } else { 0 };
        let js_h0 = hyperparameters.as_ref().and_then(|m| m.get("js_h0")).and_then(|v| v.as_u64()).unwrap_or(1);
        let js_h0 = if (0..=1).contains(&js_h0) { js_h0 } else { 1 };
        let js_hs = hyperparameters.as_ref().and_then(|m| m.get("js_hs")).and_then(|v| v.as_u64()).unwrap_or(2);
        let js_hs = if (1..=1000).contains(&js_hs) { js_hs } else { 2 };
        let js_hl = hyperparameters.as_ref().and_then(|m| m.get("js_hl")).and_then(|v| v.as_u64()).unwrap_or(90_000);
        let js_hl = if (0..=10_000_000).contains(&js_hl) { js_hl } else { 90_000 };
        let js_hq = hyperparameters.as_ref().and_then(|m| m.get("js_hq")).and_then(|v| v.as_u64()).unwrap_or(107_000);
        let js_hq = if (1..=10_000_000).contains(&js_hq) { js_hq } else { 107_000 };
        let js_hk = hyperparameters.as_ref().and_then(|m| m.get("js_hk")).and_then(|v| v.as_u64()).unwrap_or(1);
        let js_hk = if (0..=1).contains(&js_hk) { js_hk } else { 1 };
        let js_xq = hyperparameters.as_ref().and_then(|m| m.get("js_xq")).and_then(|v| v.as_u64()).unwrap_or(107000);
        let js_xq = if (1..=10_000_000).contains(&js_xq) { js_xq } else { 107000 };
        let js_e0_armed = js_triage && js_no_e0 == 0 && js_e0tc < js_tc;
        let js_tlo = hyperparameters.as_ref().and_then(|m| m.get("js_tlo")).and_then(|v| v.as_u64()).unwrap_or(16000);
        let js_tlo = if (1..=10_000_000).contains(&js_tlo) { js_tlo } else { 16000 };
        let js_thi = hyperparameters.as_ref().and_then(|m| m.get("js_thi")).and_then(|v| v.as_u64()).unwrap_or(5059);
        let js_thi = if (1..=10_000_000).contains(&js_thi) { js_thi } else { 5059 };
        let js_bar = hyperparameters.as_ref().and_then(|m| m.get("js_bar")).and_then(|v| v.as_u64()).unwrap_or(101941);
        let js_bar = if (1..=10_000_000).contains(&js_bar) { js_bar } else { 101941 };
        let js_tb = hyperparameters.as_ref().and_then(|m| m.get("js_tb")).and_then(|v| v.as_u64()).unwrap_or(100);
        let js_tb = if (1..=10_000).contains(&js_tb) { js_tb } else { 100 };
        let js_tx = hyperparameters.as_ref().and_then(|m| m.get("js_tx")).and_then(|v| v.as_u64()).unwrap_or(200);
        let js_tx = if (1..=100_000).contains(&js_tx) { js_tx } else { 200 };
        let js_tx = if js_triage { js_tx } else { 0 };
        let js_xs = hyperparameters.as_ref().and_then(|m| m.get("js_xs")).and_then(|v| v.as_u64()).unwrap_or(1);
        let js_xs = if (0..=1).contains(&js_xs) { js_xs } else { 1 };
        let js_xs = if js_triage { js_xs } else { 0 };
        let js_win = hyperparameters.as_ref().and_then(|m| m.get("js_win")).and_then(|v| v.as_u64()).unwrap_or(1);
        let js_win = if (0..=1).contains(&js_win) { js_win } else { 1 };
        let js_win = if js_triage { js_win } else { 0 };
        let js_wlo = hyperparameters.as_ref().and_then(|m| m.get("js_wlo")).and_then(|v| v.as_u64()).unwrap_or(2000);
        let js_wlo = if (1..=10_000_000).contains(&js_wlo) { js_wlo } else { 2000 };
        let js_whi = hyperparameters.as_ref().and_then(|m| m.get("js_whi")).and_then(|v| v.as_u64()).unwrap_or(1000);
        let js_whi = if (1..=10_000_000).contains(&js_whi) { js_whi } else { 1000 };
        let js_tw = hyperparameters.as_ref().and_then(|m| m.get("js_tw")).and_then(|v| v.as_u64()).unwrap_or(300);
        let js_tw = if (1..=100_000).contains(&js_tw) { js_tw } else { 300 };
        let mut triage = if js_triage {
            Triage::new(challenge, &best_mk, js_tc, js_tlo, js_thi, js_bar, js_tb,
                if js_e0_armed { Some((js_e0tc * 1_000_000_000, js_e0)) } else { None })?
        } else { None };
        if js_win != 0 {
            if let Some(triage) = triage.as_mut() {
                triage.window = Some((js_wlo, js_whi, js_tw));
            }
        }
        if js_h0 == 1 && js_e0_armed {
            if let Some(triage) = triage.as_mut() {
                triage.high = Some((js_hs * 1_000_000_000, js_hl, js_hq));
                if js_hk == 1 { triage.level = Some(js_hq as i128 + 1); }
            }
        }
        if js_xq > 0 {
            if let Some(triage) = triage.as_mut() {
                triage.xq = Some(js_xq);
            }
        }
        let done = std::cell::Cell::new(false);
        run_bounded(u64::MAX, &guarded, |bound| {
            done.set(job_shop_engine::solver::solve_challenge(challenge, bound, js_x, js_tx, js_xs, js_nr, js_nb, triage.as_ref())?);
            Ok(())
        });
        if done.get() { return Ok(()); }
        return greedy_floor(&pre, &guarded);
    }

    let der = der.expect("der");

    if routes_to_hybrid(&pre, Some(&der)) {
        let hy_x = hyperparameters.as_ref().and_then(|m| m.get("hy_x")).and_then(|v| v.as_u64()).unwrap_or(100);
        let hy_x = if (1..=1000).contains(&hy_x) { hy_x } else { 100 };
        let hy_triage = hyperparameters.as_ref().and_then(|m| m.get("hy_triage")).and_then(|v| v.as_u64()).unwrap_or(1) != 0;
        let hy_tc = hyperparameters.as_ref().and_then(|m| m.get("hy_tc")).and_then(|v| v.as_u64()).unwrap_or(72);
        let hy_tc = if (1..=1000).contains(&hy_tc) { hy_tc } else { 72 };
        let hy_tlo = hyperparameters.as_ref().and_then(|m| m.get("hy_tlo")).and_then(|v| v.as_u64()).unwrap_or(8000);
        let hy_tlo = if (1..=10_000_000).contains(&hy_tlo) { hy_tlo } else { 8000 };
        let hy_thi = hyperparameters.as_ref().and_then(|m| m.get("hy_thi")).and_then(|v| v.as_u64()).unwrap_or(8000);
        let hy_thi = if (1..=10_000_000).contains(&hy_thi) { hy_thi } else { 8000 };
        let hy_bar = hyperparameters.as_ref().and_then(|m| m.get("hy_bar")).and_then(|v| v.as_u64()).unwrap_or(99193);
        let hy_bar = if (1..=10_000_000).contains(&hy_bar) { hy_bar } else { 99193 };
        let hy_ea = hyperparameters.as_ref().and_then(|m| m.get("hy_ea")).and_then(|v| v.as_u64()).unwrap_or(1);
        let hy_ea = if (0..=1).contains(&hy_ea) { hy_ea } else { 1 };
        let hy_e0 = hyperparameters.as_ref().and_then(|m| m.get("hy_e0")).and_then(|v| v.as_u64()).unwrap_or(20);
        let hy_e0 = if (1..=1000).contains(&hy_e0) { hy_e0 } else { 20 };
        let hy_et = hyperparameters.as_ref().and_then(|m| m.get("hy_et")).and_then(|v| v.as_u64()).unwrap_or(75000);
        let hy_et = if (1..=10_000_000).contains(&hy_et) { hy_et } else { 75000 };
        let hy_h0 = hyperparameters.as_ref().and_then(|m| m.get("hy_h0")).and_then(|v| v.as_u64()).unwrap_or(1);
        let hy_h0 = if (0..=1).contains(&hy_h0) { hy_h0 } else { 1 };
        let hy_hs = hyperparameters.as_ref().and_then(|m| m.get("hy_hs")).and_then(|v| v.as_u64()).unwrap_or(10);
        let hy_hs = if (1..=1000).contains(&hy_hs) { hy_hs } else { 10 };
        let hy_hl = hyperparameters.as_ref().and_then(|m| m.get("hy_hl")).and_then(|v| v.as_u64()).unwrap_or(90000);
        let hy_hl = if (0..=10_000_000).contains(&hy_hl) { hy_hl } else { 90000 };
        let hy_hq = hyperparameters.as_ref().and_then(|m| m.get("hy_hq")).and_then(|v| v.as_u64()).unwrap_or(110000);
        let hy_hq = if (1..=10_000_000).contains(&hy_hq) { hy_hq } else { 110000 };
        let mut triage = if hy_triage {
            Triage::new(challenge, &best_mk, hy_tc, hy_tlo, hy_thi, hy_bar, 100,
                if hy_ea == 1 && hy_e0 < hy_tc { Some((hy_e0 * 1_000_000_000, hy_et)) } else { None })?
        } else { None };
        if hy_h0 == 1 && hy_triage && hy_ea == 1 && hy_e0 < hy_tc {
            if let Some(triage) = triage.as_mut() {
                triage.high = Some((hy_hs * 1_000_000_000, hy_hl, hy_hq));
            }
        }
        run_bounded(u64::MAX, &guarded, |bound| hybrid_engine::solve_challenge(challenge, bound, hy_x, triage.as_ref(), hyperparameters));
        return Ok(());
    }

    let knob = |k: &str, d: u64| hyperparameters.as_ref().and_then(|m| m.get(k)).and_then(|v| v.as_u64()).unwrap_or(d);
    let fh_alt = knob("fh_alt", 1);
    let fh_alt = if fh_alt <= 1 { fh_alt } else { 1 };
    if der.chaotic_like {
        let bounded = |key: &str, default: u64, min: u64, max: u64| {
            let value = knob(key, default);
            if (min..=max).contains(&value) { value } else { default }
        };
        let fh_fs = knob("fh_fs", 690);
        let fh_fs = if (1..=4000).contains(&fh_fs) { fh_fs } else { 690 };
        let fh_hs = bounded("fh_hs", 85_000, 0, 10_000_000);
        let fh_hx = bounded("fh_hx", 250, 0, 1000);
        let fh_hu = if fh_hs > 0 && fh_hx > 0 { bounded("fh_hu", 87_300, 1, 10_000_000) } else { 87_300 };
        let fh_hq = if fh_hs > 0 && fh_hx > 0 { bounded("fh_hq", 85_000, 0, 1_000_000) } else { 0 };
        let fh_ht = if fh_hs > 0 && fh_hx > 0 { bounded("fh_ht", 1, 0, 1) } else { 0 };
        let hx = fjsp_pool::high::Hx::new(fh_hx, fh_hu, fh_hq);
        let fh_xf = bounded("fh_xf", 250, 0, 1000);
        let fh_xl = bounded("fh_xl", 82_000, 0, 10_000_000);
        let fh_xh = bounded("fh_xh", 84_500, 0, 10_000_000);
        let (fh_xl, fh_xh) = if fh_xl < fh_xh { (fh_xl, fh_xh) } else { (82_000, 84_500) };
        let fh_zl = bounded("fh_zl", 85_500, 0, 10_000_000);
        let fh_zh = bounded("fh_zh", 87_263, 0, 10_000_000);
        let (fh_zl, fh_zh) = if fh_zl < fh_zh { (fh_zl, fh_zh) } else { (85_500, 87_263) };
        let fh_zf = bounded("fh_zf", 60, 0, 1000);
        let fh_wf = bounded("fh_wf", 250, 0, 1000);
        let fh_wl = bounded("fh_wl", 82_000, 0, 10_000_000);
        let fh_wh = bounded("fh_wh", 83_000, 0, 10_000_000);
        let (fh_wl, fh_wh) = if fh_wl < fh_wh { (fh_wl, fh_wh) } else { (82_000, 83_000) };
        let fh_vf = bounded("fh_vf", 250, 0, 1000);
        let fh_vl = bounded("fh_vl", 85_000, 0, 10_000_000);
        let fh_vh = bounded("fh_vh", 85_500, 0, 10_000_000);
        let (fh_vl, fh_vh) = if fh_vl < fh_vh { (fh_vl, fh_vh) } else { (85_000, 85_500) };
        let fh_sr = knob("fh_sr", 1);
        let fh_sr = if (0..=1).contains(&fh_sr) { fh_sr } else { 1 };
        let fh_sq = knob("fh_sq", 1);
        let fh_sq = if (0..=1).contains(&fh_sq) { fh_sq == 1 } else { true };
        let fh_rd = knob("fh_rd", 3);
        let fh_rd = if (2..=24).contains(&fh_rd) { fh_rd } else { 3 };
        let fh_ne = knob("fh_ne", 1);
        let fh_ne = if fh_ne <= 6 { fh_ne as usize } else { 1 };
        let fh_tri = knob("fh_tri", 1);
        let fh_tri = if (0..=1).contains(&fh_tri) { fh_tri } else { 1 };
        let triage = if fh_tri == 1 {
            let w0 = bounded("fh_w0", 2301, 0, 100_000);
            Some((bounded("fh_tc", 400, 1, 1000), bounded("fh_bd", 80503, 1, 10_000_000),
                w0))
        } else { None };
        let triage_b = triage;
        macro_rules! high {
            (@mode, $method:ident, $alt:literal, $ht:literal, $($arg:expr),+ $(,)?) => {{
                if fh_hq > 0 {
                    fjsp_pool::high::$method::<$alt, true, true, true, $ht>($($arg),+, fh_hs, &hx)
                } else {
                    fjsp_pool::high::$method::<$alt, true, true, false, $ht>($($arg),+, fh_hs, &hx)
                }
            }};
            (@hx, $method:ident, $alt:literal, $($arg:expr),+ $(,)?) => {{
                if fh_ht == 1 {
                    high!(@mode, $method, $alt, true, $($arg),+)
                } else {
                    high!(@mode, $method, $alt, false, $($arg),+)
                }
            }};
            ($method:ident, $alt:literal, $($arg:expr),+ $(,)?) => {{
                if fh_hs == 0 {
                    fjsp_pool::high::$method::<$alt, false, false, false, false>($($arg),+, fh_hs, &hx)?
                } else {
                    let result = if fh_hx == 0 {
                        fjsp_pool::high::$method::<$alt, true, false, false, false>($($arg),+, fh_hs, &hx)
                    } else {
                        high!(@hx, $method, $alt, $($arg),+)
                    };
                    match result {
                        Ok(value) => value,
                        Err(error) => {
                            if error.downcast_ref::<fjsp_pool::high::Hs>().is_some() {
                                return Ok(());
                            }
                            return Err(error);
                        }
                    }
                }
            }};
        }
        let sota_a = if fh_alt == 1 {
            high!(solve, true, challenge, &guarded, &der, pre.common_route,
                knob("fs_hi", fjsp_pool::high::FS_HI),
                knob("fh_clamp", fjsp_pool::high::FH_CLAMP) != 0,
                knob("fh_reserve", fjsp_pool::high::FH_RESERVE), fh_fs, fh_rd, fh_ne, fh_sq, triage, &best_mk,
                (fh_xf, fh_xl, fh_xh, fh_zl, fh_zh, fh_zf, fh_wf, fh_wl, fh_wh, fh_vf, fh_vl, fh_vh))
        } else {
            high!(solve, false, challenge, &guarded, &der, pre.common_route,
                knob("fs_hi", fjsp_pool::high::FS_HI),
                knob("fh_clamp", fjsp_pool::high::FH_CLAMP) != 0,
                knob("fh_reserve", fjsp_pool::high::FH_RESERVE), fh_fs, fh_rd, fh_ne, fh_sq, triage, &best_mk,
                (fh_xf, fh_xl, fh_xh, fh_zl, fh_zh, fh_zf, fh_wf, fh_wl, fh_wh, fh_vf, fh_vl, fh_vh))
        };
        let fh_rs = bounded("fh_rs", 1, 0, 1);
        let fh_rb = bounded("fh_rb", 81_500, 1, 10_000_000);
        let fh_rw = bounded("fh_rw", 625, 1, 100_000);
        let fh_rf = bounded("fh_rf", 1000, 1, 4000);
        if fh_rs == 0 { return Ok(()); }

        let a_mk = best_mk.get();
        let sota = sota_a.or_else(|| if fh_sq {
            sota::sota_makespan_q(challenge)
        } else {
            sota::sota_makespan(challenge)
        })
            .ok_or_else(|| anyhow::anyhow!("SOTA baseline failed"))?;
        let quality = |mk: u32| {
            let q = (sota as f64 - mk as f64) / sota as f64;
            (q.clamp(-10.0, 10.0) * 1_000_000.0).round() as i64
        };
        let q_a = quality(a_mk);
        let gate = fh_rs == 1 && fh_rb as i64 - fh_rw as i64 <= q_a && q_a < fh_rb as i64;
        if gate {
            if fh_alt == 1 {
                high!(run_b, true, challenge, &guarded, &der, pre.common_route,
                    knob("fs_hi", fjsp_pool::high::FS_HI),
                    knob("fh_clamp", fjsp_pool::high::FH_CLAMP) != 0,
                    knob("fh_reserve", fjsp_pool::high::FH_RESERVE), fh_rf, fh_rd, fh_ne, fh_sq, triage_b,
                    if fh_sr == 1 { sota_a } else { None }, &best_mk)
            } else {
                high!(run_b, false, challenge, &guarded, &der, pre.common_route,
                    knob("fs_hi", fjsp_pool::high::FS_HI),
                    knob("fh_clamp", fjsp_pool::high::FH_CLAMP) != 0,
                    knob("fh_reserve", fjsp_pool::high::FH_RESERVE), fh_rf, fh_rd, fh_ne, fh_sq, triage_b,
                    if fh_sr == 1 { sota_a } else { None }, &best_mk)
            };
        }
        debug_assert!(best_mk.get() <= a_mk);
        return Ok(());
    }
    let fm_take = knob("fm_take", 300);
    let fm_take = if (1..=900).contains(&fm_take) { fm_take } else { 300 };
    macro_rules! solve_medium {
        ($alt:literal) => {{
            if knob("no_take", 0) != 1 {
                fjsp_pool::solve::<true, $alt>(challenge, &guarded, &der, pre.common_route,
                    knob("fs_med", fjsp_pool::FS_MED),
                    knob("fm_clamp", fjsp_pool::FM_CLAMP) != 0,
                    knob("fm_reserve", fjsp_pool::FM_RESERVE),
                    knob("fm_pat", 0).min(usize::MAX as u64) as usize,
                    knob("fm_deep", 0).min(usize::MAX as u64) as usize,
                    knob("fm_slice", 0), fm_take, knob("fm_tdiv", 0))
            } else {
                fjsp_pool::solve::<false, $alt>(challenge, &guarded, &der, pre.common_route,
                    knob("fs_med", fjsp_pool::FS_MED),
                    knob("fm_clamp", fjsp_pool::FM_CLAMP) != 0,
                    knob("fm_reserve", fjsp_pool::FM_RESERVE),
                    knob("fm_pat", 0).min(usize::MAX as u64) as usize,
                    knob("fm_deep", 0).min(usize::MAX as u64) as usize,
                    knob("fm_slice", 0), 0, 0)
            }
        }};
    }
    solve_medium!(false)
}

