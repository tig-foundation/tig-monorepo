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

fn processing_row(
    op: &std::collections::HashMap<usize, u32>,
    num_machines: usize,
) -> Result<Vec<Option<u32>>> {
    let mut row = vec![None; num_machines];
    for (&m, &pt) in op {
        let slot = row
            .get_mut(m)
            .ok_or_else(|| anyhow::anyhow!("machine id out of range"))?;
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
        for op in ops {
            product.push(processing_row(op, challenge.num_machines)?);
        }
        rows.push(product);
    }
    let mut usage = vec![Vec::new(); challenge.num_machines];
    let mut makespan = 0;
    let mut job = 0;
    for (product, &count) in challenge.jobs_per_product.iter().enumerate() {
        let processing = rows
            .get(product)
            .ok_or_else(|| anyhow::anyhow!("invalid product"))?;
        for _ in 0..count {
            let schedule = solution
                .job_schedule
                .get(job)
                .ok_or_else(|| anyhow::anyhow!("invalid product expansion"))?;
            if schedule.len() != processing.len() {
                return Err(anyhow::anyhow!("invalid operation count"));
            }
            let mut ready = 0;
            for (oi, &(machine, start)) in schedule.iter().enumerate() {
                if start < ready {
                    return Err(anyhow::anyhow!("job precedence violation"));
                }
                let pt = processing[oi]
                    .get(machine)
                    .copied()
                    .flatten()
                    .ok_or_else(|| anyhow::anyhow!("ineligible machine"))?;
                let end = start
                    .checked_add(pt)
                    .ok_or_else(|| anyhow::anyhow!("time overflow"))?;
                usage[machine].push((start, end));
                ready = end;
            }
            makespan = makespan.max(ready);
            job += 1;
        }
    }
    if job != challenge.num_jobs {
        return Err(anyhow::anyhow!("invalid product expansion"));
    }
    for machine in &mut usage {
        machine.sort_by_key(|&(start, _)| start);
        if machine.windows(2).any(|w| w[1].0 < w[0].1) {
            return Err(anyhow::anyhow!("machine overlap"));
        }
    }
    Ok(makespan)
}

fn common_route(job_ops: &[Vec<Vec<(usize, u32)>>]) -> bool {
    let Some(first) = job_ops.first() else {
        return true;
    };
    job_ops.iter().skip(1).all(|route| {
        route.len() == first.len()
            && route.iter().zip(first).all(|(op, reference)| {
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
    fn build(challenge: &Challenge) -> Self {
        let num_jobs = challenge.num_jobs;
        let mut job_ops: Vec<Vec<Vec<(usize, u32)>>> = Vec::with_capacity(num_jobs);
        for (product, &count) in challenge.jobs_per_product.iter().enumerate() {
            let ops = &challenge.product_processing_times[product];
            for _ in 0..count {
                let mut this_job = Vec::with_capacity(ops.len());
                for op in ops.iter() {
                    let cands = processing_row(op, challenge.num_machines)
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
        let common_route = common_route(&job_ops);
        Pre {
            num_jobs,
            num_machines: challenge.num_machines,
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
        }
    }
    #[inline]
    fn op_id(&self, job: usize, oi: usize) -> usize {
        self.job_off[job] + oi
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
            op_dur[oid] = pre.job_ops[j][o]
                .iter()
                .find(|&&(mm, _)| mm == m)
                .map(|&(_, t)| t)
                .unwrap_or(0);
        }
        let mut g = Graph {
            pre,
            op_machine,
            op_dur,
            machine_seq,
            heads: vec![0; n],
            topo: vec![0; n],
            mp: vec![usize::MAX; n],
            ms: vec![usize::MAX; n],
            indeg0: vec![0; n],
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
        for oid in 0..self.op_machine.len() {
            let j = self.pre.op_job[oid];
            let o = self.pre.op_idx[oid];
            let m = self.op_machine[oid];
            let d = self.pre.job_ops[j][o]
                .iter()
                .find(|&&(mm, _)| mm == m)
                .map(|&(_, t)| t)
                .unwrap_or(0);
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
            if self.pre.op_idx[oid] > 0 {
                d += 1;
            }
            if self.mp[oid] != usize::MAX {
                d += 1;
            }
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
        for o in 0..n {
            let r = self.nd[o];
            self.heads[o] = r[0];
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
                for &(m, pt) in elig {
                    if pt < mn {
                        mn = pt;
                        best_m = m;
                    }
                }
                op_min[oid] = mn;
                total_min_work += mn as f64;
                total_flex += flex as f64;
                machine_best_cnt[best_m] += 1.0;
                let ff = flex as f64;
                let delta = mn as f64 / ff;
                let delta_s = mn as f64 / (ff * ff);
                for &(m, _) in elig {
                    machine_load0[m] += delta;
                    machine_scarcity[m] += delta_s;
                }
            }
        }
        let mut suf_min = Vec::with_capacity(pre.num_jobs);
        for j in 0..pre.num_jobs {
            let n = pre.job_nops[j];
            let mut s = vec![0u32; n + 1];
            for oi in (0..n).rev() {
                s[oi] = s[oi + 1].saturating_add(op_min[pre.op_id(j, oi)]);
            }
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
        let machine_best_pop = machine_best_cnt
            .iter()
            .map(|&c| {
                let r = (c / mean).clamp(0.0, 10.0);
                (r / (1.0 + r)).clamp(0.0, 1.0)
            })
            .collect();
        let chaotic_like = high_flex > 0.85;
        let use_rich = chaotic_like || (flex_avg >= 2.5 && flex_avg <= 6.0);
        let jobshopness = if chaotic_like { 0.85 } else { 0.55 };
        let time_scale = (horizon * (2.65 + 0.10 * jobshopness + 0.10 * high_flex)).max(1.0);
        Der {
            op_min,
            suf_min,
            machine_load0,
            machine_scarcity,
            machine_best_pop,
            avg_machine_load,
            avg_machine_scarcity,
            avg_op_min,
            horizon,
            time_scale,
            flex_avg,
            flex_factor,
            high_flex,
            jobshopness,
            chaotic_like,
            use_rich,
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
    fuel0: u64,
    checkpoint: u64,
    lo: u64,
    hi: u64,
    bar: u64,
    boost: u64,
    sota: std::cell::Cell<Option<u32>>,
    window: Option<(u64, u64, u64)>,
    challenge: &'a Challenge,
    best_mk: &'a std::cell::Cell<u32>,
    class: std::cell::Cell<Option<u8>>,
    trace: bool,
}

impl<'a> Triage<'a> {
    fn new(
        challenge: &'a Challenge,
        best_mk: &'a std::cell::Cell<u32>,
        tc: u64,
        lo: u64,
        hi: u64,
        bar: u64,
        boost: u64,
        trace: bool,
    ) -> Result<Option<Self>> {
        if tc == 0 && !trace {
            return Ok(None);
        }
        let mut fuel0 = fuel_left();
        let sota = if trace {
            let sota = sota::sota_makespan(challenge)
                .ok_or_else(|| anyhow::anyhow!("SOTA baseline failed"))?;
            let fuel = fuel_left();
            eprintln!("DIAG s {} {}", sota, fuel0.saturating_sub(fuel));
            fuel0 = fuel;
            Some(sota)
        } else {
            None
        };
        if tc == 0 {
            return Ok(None);
        }
        Ok(Some(Self {
            fuel0,
            checkpoint: tc.saturating_mul(1_000_000_000),
            lo,
            hi,
            bar,
            boost,
            sota: std::cell::Cell::new(sota),
            window: None,
            challenge,
            best_mk,
            class: std::cell::Cell::new(None),
            trace,
        }))
    }

    fn check_at(&self, fuel: u64) -> bool {
        if self.class.get().is_none() && self.fuel0.saturating_sub(fuel) >= self.checkpoint {
            let best_mk = self.best_mk.get();
            let quality = |sota: u32| {
                let quality = (sota as f64 - best_mk as f64) / sota as f64;
                (quality.clamp(-10.0, 10.0) * 1_000_000.0).round() as i32
            };
            let sota_before = fuel_left();
            let upper = sota::sota_upper(self.challenge);
            let q = upper.and_then(|upper| {
                let q = quality(upper);
                if !self.trace && (q as i128) < self.bar as i128 - self.lo as i128 {
                    Some(q)
                } else {
                    self.sota
                        .get()
                        .or_else(|| sota::sota_makespan(self.challenge))
                        .map(|sota| {
                            self.sota.set(Some(sota));
                            quality(sota)
                        })
                }
            });
            SOTA_SPENT.fetch_add(
                sota_before.saturating_sub(fuel_left()),
                std::sync::atomic::Ordering::Relaxed,
            );
            let class = match q {
                Some(q) if (q as i128) < self.bar as i128 - self.lo as i128 => 0,
                Some(q) if (q as i128) > self.bar as i128 + self.hi as i128 => 2,
                _ => 1,
            };
            self.class.set(Some(class));
            if self.trace {
                match upper {
                    Some(upper) => eprintln!("DIAG u {}", upper),
                    None => eprintln!("DIAG u unavailable"),
                }
                eprintln!("DIAG c {} {}", class, q.unwrap_or(0));
            }
        }
        matches!(self.class.get(), Some(0) | Some(2))
    }

    fn stop(&self) -> bool {
        if self.class.get().is_none() {
            return self.check_at(fuel_left());
        }
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
        self.uncertain()
            && self.quality(mk).map_or(false, |q| {
                self.bar as i128 - wlo as i128 <= q as i128
                    && (q as i128) < self.bar as i128 + whi as i128
            })
    }

    fn scale(&self, iterations: usize) -> usize {
        if self.class.get() != Some(1) {
            return iterations;
        }
        ((iterations as u128 * self.boost as u128 / 100).min(usize::MAX as u128)) as usize
    }

    fn stopped_class(&self) -> Option<u8> {
        self.class.get().filter(|&c| c == 0 || c == 2)
    }
}

pub fn help() {
    println!("ember_krakatoa: job scheduling, one engine per track.");
    println!("Each track has its own defaults, selected by an empty map.");
    println!("A missing or out-of-range hyperparameter falls back to its default.");
    println!("Hyperparameters: {}.", HYPERPARAMETERS.join(", "));
    println!("The solver is anytime and reads its budget from `__fuel_remaining`.");
}

const HYPERPARAMETERS: [&str; 48] = [
    "trace",
    "fs_hi",
    "fs_med",
    "fh_clamp",
    "fh_reserve",
    "fm_clamp",
    "fm_reserve",
    "fm_pat",
    "fm_deep",
    "fm_slice",
    "fm_take",
    "fm_tdiv",
    "fm_tx",
    "js_take",
    "fl_x",
    "hy_x",
    "hy_triage",
    "hy_tc",
    "hy_tlo",
    "hy_thi",
    "hy_bar",
    "hy_win",
    "hy_wlo",
    "hy_whi",
    "hy_tw",
    "hy_c0",
    "hy_c2",
    "js_x",
    "js_triage",
    "js_tc",
    "js_tlo",
    "js_thi",
    "js_bar",
    "js_tb",
    "js_tx",
    "js_c0",
    "js_xs",
    "js_win",
    "js_wlo",
    "js_whi",
    "js_tw",
    "js_b1",
    "js_ils",
    "js_b2",
    "js_salt",
    "js_nr",
    "js_nb",
    "no_take",
];

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
    let fuel0 = fuel_left();
    let trace = hyperparameters
        .as_ref()
        .and_then(|m| m.get("trace"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let trace = trace == 1;
    let pre = Pre::build(challenge);

    let best_mk = std::cell::Cell::new(u32::MAX);
    let guarded = |s: &Solution| -> Result<()> {
        if let Ok(mk) = checked_makespan(challenge, s) {
            if mk <= best_mk.get() {
                best_mk.set(mk);
                if trace {
                    eprintln!("DIAG t {} {}", fuel0.saturating_sub(fuel_left()), mk);
                }
                return save_solution(s);
            }
        }
        Ok(())
    };

    let der = if pre.max_flex > 1 {
        Some(Der::build(&pre))
    } else {
        None
    };
    if !routes_to_hybrid(&pre, der.as_ref()) {
        let lean = pre.max_flex == 1 && !pre.common_route && fuel_left() < 20_000_000_000u64;
        let flow = pre.max_flex == 1 && pre.common_route;
        if !lean && !flow {
            gate_floor::gate_floor(&pre, &challenge.seed, &guarded)?;
        }
    }

    if pre.max_flex == 1 && pre.common_route {
        const FLOW_CAP: u64 = 10_000_000_000u64;
        let fl_x = hyperparameters
            .as_ref()
            .and_then(|m| m.get("fl_x"))
            .and_then(|v| v.as_u64())
            .unwrap_or(100);
        let fl_x = if (1..=1000).contains(&fl_x) {
            fl_x
        } else {
            100
        };
        run_bounded(FLOW_CAP * fl_x / 100, &guarded, |bound| {
            flow_shop_engine::solve_challenge(challenge, bound)
        });
        gate_floor::gate_floor(&pre, &challenge.seed, &guarded)?;
        return greedy_floor(&pre, &guarded);
    }

    if pre.max_flex == 1 && !pre.common_route {
        let js_take = hyperparameters
            .as_ref()
            .and_then(|m| m.get("js_take"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let js_take = if (1..=900).contains(&js_take) {
            js_take as usize
        } else {
            0
        };
        let js_x = hyperparameters
            .as_ref()
            .and_then(|m| m.get("js_x"))
            .and_then(|v| v.as_u64())
            .unwrap_or(100);
        let js_x = if (1..=1000).contains(&js_x) {
            js_x as usize
        } else {
            100
        };
        let js_b1 = hyperparameters
            .as_ref()
            .and_then(|m| m.get("js_b1"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let js_b1 = js_b1 == 1;
        let js_ils = hyperparameters
            .as_ref()
            .and_then(|m| m.get("js_ils"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let js_ils = js_ils == 1;
        let js_b2 = hyperparameters
            .as_ref()
            .and_then(|m| m.get("js_b2"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let js_b2 = js_b2 == 1;
        let js_salt = hyperparameters
            .as_ref()
            .and_then(|m| m.get("js_salt"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let js_salt = js_salt == 1;
        let js_nr = hyperparameters
            .as_ref()
            .and_then(|m| m.get("js_nr"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let js_nr = if (1..=2000).contains(&js_nr) {
            js_nr as usize
        } else {
            0
        };
        let js_nb = hyperparameters
            .as_ref()
            .and_then(|m| m.get("js_nb"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let js_nb = if (1..=10000).contains(&js_nb) {
            js_nb as usize
        } else {
            0
        };
        let js_triage = hyperparameters
            .as_ref()
            .and_then(|m| m.get("js_triage"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1)
            != 0;
        let js_tc = hyperparameters
            .as_ref()
            .and_then(|m| m.get("js_tc"))
            .and_then(|v| v.as_u64())
            .unwrap_or(62);
        let js_tc = if (1..=1000).contains(&js_tc) {
            js_tc
        } else {
            62
        };
        let js_tlo = hyperparameters
            .as_ref()
            .and_then(|m| m.get("js_tlo"))
            .and_then(|v| v.as_u64())
            .unwrap_or(8000);
        let js_tlo = if (1..=10_000_000).contains(&js_tlo) {
            js_tlo
        } else {
            8000
        };
        let js_thi = hyperparameters
            .as_ref()
            .and_then(|m| m.get("js_thi"))
            .and_then(|v| v.as_u64())
            .unwrap_or(7000);
        let js_thi = if (1..=10_000_000).contains(&js_thi) {
            js_thi
        } else {
            7000
        };
        let js_bar = hyperparameters
            .as_ref()
            .and_then(|m| m.get("js_bar"))
            .and_then(|v| v.as_u64())
            .unwrap_or(101941);
        let js_bar = if (1..=10_000_000).contains(&js_bar) {
            js_bar
        } else {
            101941
        };
        let js_tb = hyperparameters
            .as_ref()
            .and_then(|m| m.get("js_tb"))
            .and_then(|v| v.as_u64())
            .unwrap_or(100);
        let js_tb = if (1..=10_000).contains(&js_tb) {
            js_tb
        } else {
            100
        };
        let js_tx = hyperparameters
            .as_ref()
            .and_then(|m| m.get("js_tx"))
            .and_then(|v| v.as_u64())
            .unwrap_or(200);
        let js_tx = if (1..=100_000).contains(&js_tx) {
            js_tx
        } else {
            200
        };
        let js_tx = if js_triage { js_tx } else { 0 };
        let js_c0 = hyperparameters
            .as_ref()
            .and_then(|m| m.get("js_c0"))
            .and_then(|v| v.as_u64())
            .unwrap_or(80);
        let js_c0 = if (0..=1000).contains(&js_c0) {
            js_c0
        } else {
            80
        };
        let js_c0 = if js_triage { js_c0 } else { 0 };
        let js_xs = hyperparameters
            .as_ref()
            .and_then(|m| m.get("js_xs"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1);
        let js_xs = if (0..=1).contains(&js_xs) { js_xs } else { 1 };
        let js_xs = if js_triage { js_xs } else { 0 };
        let js_win = hyperparameters
            .as_ref()
            .and_then(|m| m.get("js_win"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1);
        let js_win = if (0..=1).contains(&js_win) { js_win } else { 1 };
        let js_win = if js_triage { js_win } else { 0 };
        let js_wlo = hyperparameters
            .as_ref()
            .and_then(|m| m.get("js_wlo"))
            .and_then(|v| v.as_u64())
            .unwrap_or(2000);
        let js_wlo = if (1..=10_000_000).contains(&js_wlo) {
            js_wlo
        } else {
            2000
        };
        let js_whi = hyperparameters
            .as_ref()
            .and_then(|m| m.get("js_whi"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1000);
        let js_whi = if (1..=10_000_000).contains(&js_whi) {
            js_whi
        } else {
            1000
        };
        let js_tw = hyperparameters
            .as_ref()
            .and_then(|m| m.get("js_tw"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1000);
        let js_tw = if (1..=100_000).contains(&js_tw) {
            js_tw
        } else {
            1000
        };
        if trace {
            eprintln!(
                "DIAG r {} {} {} {} {} {} {} {} {} {} {}",
                js_triage as u8,
                js_tc,
                js_tlo,
                js_thi,
                js_bar,
                js_tb,
                js_tx,
                js_win,
                js_wlo,
                js_whi,
                js_tw
            );
        }
        let mut triage = if js_triage {
            Triage::new(
                challenge, &best_mk, js_tc, js_tlo, js_thi, js_bar, js_tb, trace,
            )?
        } else {
            None
        };
        if js_win != 0 {
            if let Some(triage) = triage.as_mut() {
                triage.window = Some((js_wlo, js_whi, js_tw));
            }
        }
        let reached = std::cell::Cell::new(false);
        run_bounded(u64::MAX, &guarded, |bound| {
            reached.set(job_shop_engine::solver::solve_challenge(
                challenge,
                bound,
                js_take,
                js_x,
                js_tx,
                js_c0,
                js_xs,
                js_b1,
                js_ils,
                js_b2,
                js_salt,
                js_nr,
                js_nb,
                triage.as_ref(),
            )?);
            Ok(())
        });
        if reached.get() {
            return Ok(());
        }
        return greedy_floor(&pre, &guarded);
    }

    let der = der.expect("der");

    if routes_to_hybrid(&pre, Some(&der)) {
        let hy_x = hyperparameters
            .as_ref()
            .and_then(|m| m.get("hy_x"))
            .and_then(|v| v.as_u64())
            .unwrap_or(100);
        let hy_x = if (1..=1000).contains(&hy_x) {
            hy_x
        } else {
            100
        };
        let hy_triage = hyperparameters
            .as_ref()
            .and_then(|m| m.get("hy_triage"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1)
            != 0;
        let hy_tc = hyperparameters
            .as_ref()
            .and_then(|m| m.get("hy_tc"))
            .and_then(|v| v.as_u64())
            .unwrap_or(72);
        let hy_tc = if (1..=1000).contains(&hy_tc) {
            hy_tc
        } else {
            72
        };
        let hy_tlo = hyperparameters
            .as_ref()
            .and_then(|m| m.get("hy_tlo"))
            .and_then(|v| v.as_u64())
            .unwrap_or(5000);
        let hy_tlo = if (1..=10_000_000).contains(&hy_tlo) {
            hy_tlo
        } else {
            5000
        };
        let hy_thi = hyperparameters
            .as_ref()
            .and_then(|m| m.get("hy_thi"))
            .and_then(|v| v.as_u64())
            .unwrap_or(8000);
        let hy_thi = if (1..=10_000_000).contains(&hy_thi) {
            hy_thi
        } else {
            8000
        };
        let hy_bar = hyperparameters
            .as_ref()
            .and_then(|m| m.get("hy_bar"))
            .and_then(|v| v.as_u64())
            .unwrap_or(99193);
        let hy_bar = if (1..=10_000_000).contains(&hy_bar) {
            hy_bar
        } else {
            99193
        };
        if trace {
            eprintln!(
                "DIAG r {} {} {} {} {}",
                hy_triage as u8, hy_tc, hy_tlo, hy_thi, hy_bar
            );
        }
        let triage = if hy_triage {
            Triage::new(
                challenge, &best_mk, hy_tc, hy_tlo, hy_thi, hy_bar, 100, trace,
            )?
        } else {
            None
        };
        run_bounded(u64::MAX, &guarded, |bound| {
            hybrid_engine::solve_challenge(challenge, bound, hy_x, triage.as_ref(), hyperparameters)
        });
        return Ok(());
    }

    let knob = |k: &str, d: u64| {
        hyperparameters
            .as_ref()
            .and_then(|m| m.get(k))
            .and_then(|v| v.as_u64())
            .unwrap_or(d)
    };
    if der.chaotic_like {
        return fjsp_pool::high::solve(
            challenge,
            &guarded,
            &der,
            pre.common_route,
            knob("fs_hi", fjsp_pool::high::FS_HI),
            knob("fh_clamp", fjsp_pool::high::FH_CLAMP) != 0,
            knob("fh_reserve", fjsp_pool::high::FH_RESERVE),
        );
    }
    let fm_take = knob("fm_take", 300);
    let fm_take = if (1..=900).contains(&fm_take) {
        fm_take
    } else {
        300
    };
    if knob("no_take", 0) != 1 {
        return fjsp_pool::solve::<true>(
            challenge,
            &guarded,
            &der,
            pre.common_route,
            knob("fs_med", fjsp_pool::FS_MED),
            knob("fm_clamp", fjsp_pool::FM_CLAMP) != 0,
            knob("fm_reserve", fjsp_pool::FM_RESERVE),
            knob("fm_pat", 0).min(usize::MAX as u64) as usize,
            knob("fm_deep", 0).min(usize::MAX as u64) as usize,
            knob("fm_slice", 0),
            fm_take,
            knob("fm_tdiv", 0),
            knob("fm_tx", 0) == 1,
        );
    }
    fjsp_pool::solve::<false>(
        challenge,
        &guarded,
        &der,
        pre.common_route,
        knob("fs_med", fjsp_pool::FS_MED),
        knob("fm_clamp", fjsp_pool::FM_CLAMP) != 0,
        knob("fm_reserve", fjsp_pool::FM_RESERVE),
        knob("fm_pat", 0).min(usize::MAX as u64) as usize,
        knob("fm_deep", 0).min(usize::MAX as u64) as usize,
        knob("fm_slice", 0),
        0,
        0,
        false,
    )
}
