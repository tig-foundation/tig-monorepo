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

const FUEL_STOP: u64 = 100_000_000;

extern "C" {
    static __fuel_remaining: u64;
}
#[inline]
fn fuel_left() -> u64 {
    unsafe { core::ptr::read_volatile(&__fuel_remaining) }
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
        let common_route = common_route(&job_ops);
        Pre {
            num_jobs, num_machines: challenge.num_machines, job_ops, job_nops,
            n_ops: off, job_off, op_job, op_idx, common_route, max_flex,
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

pub fn help() {
    println!("ember_vesuvio (c007 job_scheduling)");
    println!();
    println!("No tunables. Any key passed is accepted and ignored.");
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
        let lean = pre.max_flex == 1 && !pre.common_route && fuel_left() < 20_000_000_000u64;
        let flow = pre.max_flex == 1 && pre.common_route;
        if !lean && !flow {
            gate_floor::gate_floor(&pre, &challenge.seed, &guarded)?;
        }
    }

    if pre.max_flex == 1 && pre.common_route {
        const FLOW_CAP: u64 = 10_000_000_000u64;
        run_bounded(FLOW_CAP, &guarded, |bound| flow_shop_engine::solve_challenge(challenge, bound));
        gate_floor::gate_floor(&pre, &challenge.seed, &guarded)?;
        return greedy_floor(&pre, &guarded);
    }

    if pre.max_flex == 1 && !pre.common_route {
        let reached = std::cell::Cell::new(false);
        run_bounded(u64::MAX, &guarded, |bound| {
            reached.set(job_shop_engine::solver::solve_challenge(challenge, bound)?);
            Ok(())
        });
        if reached.get() { return Ok(()); }
        return greedy_floor(&pre, &guarded);
    }

    let der = der.expect("der");

    if routes_to_hybrid(&pre, Some(&der)) {
        run_bounded(u64::MAX, &guarded, |bound| hybrid_engine::solve_challenge(challenge, bound));
        return Ok(());
    }

    fjsp_pool::solve(challenge, &guarded, &der, pre.common_route)
}
