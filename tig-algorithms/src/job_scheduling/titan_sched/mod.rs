use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde_json::{Map, Value};
use tig_challenges::job_scheduling::*;

const NONE: usize = usize::MAX;

// =========================================================
//  Instance: preprocessed problem data
// =========================================================

struct Inst {
    n_jobs: usize,
    n_mach: usize,
    total_ops: usize,
    prod: Vec<usize>,
    n_ops: Vec<usize>,
    op_base: Vec<usize>,
    op_job: Vec<usize>,
    elig: Vec<Vec<usize>>,
    pt_flat: Vec<u32>,
    flex: Vec<usize>,
    flex_avg: f64,
    j_succ: Vec<usize>,
    j_pred: Vec<usize>,
    job_work: Vec<f64>,
    op_work: Vec<f64>,
}

impl Inst {
    fn new(ch: &Challenge) -> Self {
        let n_jobs = ch.num_jobs;
        let n_mach = ch.num_machines;
        let mut prod = Vec::with_capacity(n_jobs);
        let mut n_ops_v = Vec::with_capacity(n_jobs);
        for (p, &count) in ch.jobs_per_product.iter().enumerate() {
            for _ in 0..count {
                prod.push(p);
                n_ops_v.push(ch.product_processing_times[p].len());
            }
        }
        let total_ops: usize = n_ops_v.iter().sum();
        let mut op_base = Vec::with_capacity(n_jobs);
        let mut base = 0;
        for &ops in &n_ops_v {
            op_base.push(base);
            base += ops;
        }
        let mut op_job = vec![0usize; total_ops];
        let mut elig = Vec::with_capacity(total_ops);
        let mut pt_flat = vec![0u32; total_ops * n_mach];
        let mut flex = vec![0usize; total_ops];
        let mut j_succ = vec![NONE; total_ops];
        let mut j_pred = vec![NONE; total_ops];
        let mut job_work = vec![0.0f64; n_jobs];
        let mut op_work = vec![0.0f64; total_ops];
        for job in 0..n_jobs {
            let p = prod[job];
            let ops = &ch.product_processing_times[p];
            for (lo, op_times) in ops.iter().enumerate() {
                let g = op_base[job] + lo;
                op_job[g] = job;
                let mut el = Vec::with_capacity(op_times.len());
                let mut sum_pt = 0u64;
                for (&m, &pt) in op_times.iter() {
                    el.push(m);
                    pt_flat[g * n_mach + m] = pt;
                    sum_pt += pt as u64;
                }
                el.sort_unstable();
                flex[g] = el.len();
                op_work[g] = if el.is_empty() { 0.0 } else { sum_pt as f64 / el.len() as f64 };
                job_work[job] += op_work[g];
                elig.push(el);
                if lo > 0 {
                    j_succ[g - 1] = g;
                    j_pred[g] = g - 1;
                }
            }
        }
        let flex_avg = flex.iter().map(|&f| f as f64).sum::<f64>() / total_ops as f64;
        Inst {
            n_jobs, n_mach, total_ops, prod, n_ops: n_ops_v, op_base,
            op_job, elig, pt_flat, flex, flex_avg,
            j_succ, j_pred, job_work, op_work,
        }
    }

    #[inline(always)]
    fn pt(&self, g: usize, m: usize) -> u32 {
        self.pt_flat[g * self.n_mach + m]
    }
}

// =========================================================
//  Sched: schedule state + disjunctive graph evaluation
// =========================================================

struct Sched {
    assign: Vec<usize>,
    mseq: Vec<Vec<usize>>,
    m_pred: Vec<usize>,
    m_succ: Vec<usize>,
    head: Vec<u32>,
    tail: Vec<u32>,
    pt: Vec<u32>,
    makespan: u32,
    in_deg: Vec<u32>,
    queue: Vec<usize>,
    topo: Vec<usize>,
}

impl Sched {
    fn new(inst: &Inst) -> Self {
        Sched {
            assign: vec![0; inst.total_ops],
            mseq: vec![Vec::new(); inst.n_mach],
            m_pred: vec![NONE; inst.total_ops],
            m_succ: vec![NONE; inst.total_ops],
            head: vec![0; inst.total_ops],
            tail: vec![0; inst.total_ops],
            pt: vec![0; inst.total_ops],
            makespan: u32::MAX,
            in_deg: vec![0; inst.total_ops],
            queue: Vec::with_capacity(inst.total_ops),
            topo: Vec::with_capacity(inst.total_ops),
        }
    }

    fn clone_from_sched(&mut self, other: &Sched, inst: &Inst) {
        self.assign.copy_from_slice(&other.assign);
        for m in 0..inst.n_mach {
            self.mseq[m].clear();
            self.mseq[m].extend_from_slice(&other.mseq[m]);
        }
        self.pt.copy_from_slice(&other.pt);
        self.build_mlinks(inst);
    }

    fn build_mlinks(&mut self, inst: &Inst) {
        for m in 0..inst.n_mach {
            self.build_mlinks_machine(m);
        }
    }

    #[inline]
    fn build_mlinks_machine(&mut self, m: usize) {
        let seq = &self.mseq[m];
        for (i, &g) in seq.iter().enumerate() {
            self.m_pred[g] = if i > 0 { seq[i - 1] } else { NONE };
            self.m_succ[g] = if i + 1 < seq.len() { seq[i + 1] } else { NONE };
        }
    }

    /// Returns false if a cycle is detected. Computes head, tail, makespan.
    fn eval(&mut self, inst: &Inst) -> bool {
        if !self.eval_forward(inst) { return false; }
        let n = inst.total_ops;
        for g in 0..n { self.tail[g] = 0; }
        for &g in self.topo.iter().rev() {
            let js = inst.j_succ[g];
            if js != NONE {
                let v = self.pt[js] + self.tail[js];
                if v > self.tail[g] { self.tail[g] = v; }
            }
            let ms_g = self.m_succ[g];
            if ms_g != NONE {
                let v = self.pt[ms_g] + self.tail[ms_g];
                if v > self.tail[g] { self.tail[g] = v; }
            }
        }
        true
    }

    /// Forward pass only: computes head, makespan, topo order. No tail computation.
    fn eval_forward(&mut self, inst: &Inst) -> bool {
        let n = inst.total_ops;
        for g in 0..n {
            let mut deg = 0u32;
            if inst.j_pred[g] != NONE { deg += 1; }
            if self.m_pred[g] != NONE { deg += 1; }
            self.in_deg[g] = deg;
        }
        self.queue.clear();
        self.topo.clear();
        for g in 0..n {
            self.head[g] = 0;
            if self.in_deg[g] == 0 {
                self.queue.push(g);
            }
        }
        let mut ms = 0u32;
        while let Some(g) = self.queue.pop() {
            self.topo.push(g);
            let end_g = self.head[g] + self.pt[g];
            if end_g > ms { ms = end_g; }
            let js = inst.j_succ[g];
            if js != NONE {
                if end_g > self.head[js] { self.head[js] = end_g; }
                self.in_deg[js] -= 1;
                if self.in_deg[js] == 0 { self.queue.push(js); }
            }
            let ms_g = self.m_succ[g];
            if ms_g != NONE {
                if end_g > self.head[ms_g] { self.head[ms_g] = end_g; }
                self.in_deg[ms_g] -= 1;
                if self.in_deg[ms_g] == 0 { self.queue.push(ms_g); }
            }
        }
        self.makespan = ms;
        self.topo.len() == n
    }

    fn to_solution(&self, inst: &Inst) -> Solution {
        let mut js = Vec::with_capacity(inst.n_jobs);
        for job in 0..inst.n_jobs {
            let mut ops = Vec::with_capacity(inst.n_ops[job]);
            for lo in 0..inst.n_ops[job] {
                let g = inst.op_base[job] + lo;
                ops.push((self.assign[g], self.head[g]));
            }
            js.push(ops);
        }
        Solution { job_schedule: js }
    }
}

// =========================================================
//  Critical block identification
// =========================================================

struct CritBlock {
    machine: usize,
    start: usize,
    end: usize,
}

fn find_critical_blocks(sched: &Sched, inst: &Inst) -> Vec<CritBlock> {
    let mut blocks = Vec::new();
    let ms = sched.makespan;
    for m in 0..inst.n_mach {
        let seq = &sched.mseq[m];
        if seq.len() < 2 { continue; }
        let mut i = 0;
        while i < seq.len() {
            let g = seq[i];
            if sched.head[g] + sched.pt[g] + sched.tail[g] != ms {
                i += 1;
                continue;
            }
            let block_start = i;
            let mut j = i + 1;
            while j < seq.len() {
                let gj = seq[j];
                if sched.head[gj] + sched.pt[gj] + sched.tail[gj] != ms { break; }
                let gp = seq[j - 1];
                if sched.head[gp] + sched.pt[gp] != sched.head[gj] { break; }
                j += 1;
            }
            if j - block_start >= 2 {
                blocks.push(CritBlock { machine: m, start: block_start, end: j });
            }
            i = j;
        }
    }
    blocks
}

// =========================================================
//  Construction: priority dispatching rules
// =========================================================

#[derive(Clone, Copy, PartialEq)]
enum Rule { MWR, MOR, LF, SPT, LPT, ECT, LWORK, MOPNR }

const ALL_RULES: [Rule; 8] = [
    Rule::MWR, Rule::MOR, Rule::LF, Rule::SPT, Rule::LPT,
    Rule::ECT, Rule::LWORK, Rule::MOPNR,
];

fn construct(inst: &Inst, rule: Rule, mut rng: Option<&mut SmallRng>, top_k: usize,
             mach_load_bias: f64, job_bias: Option<&[f64]>,
             mach_penalty_bias: Option<&[f64]>) -> Sched {
    let mut job_next = vec![0usize; inst.n_jobs];
    let mut job_ready = vec![0u32; inst.n_jobs];
    let mut mach_avail = vec![0u32; inst.n_mach];
    let mut mach_load = vec![0u64; inst.n_mach];
    let mut sched = Sched::new(inst);
    let mut remaining = inst.total_ops;
    let mut time = 0u32;
    let mut work_rem: Vec<f64> = inst.job_work.clone();
    let mut ops_rem: Vec<usize> = inst.n_ops.clone();
    let avg_mpb = mach_penalty_bias.map(|mpb| {
        let s: f64 = mpb.iter().sum();
        if inst.n_mach > 0 { s / inst.n_mach as f64 } else { 0.0 }
    }).unwrap_or(0.0);

    while remaining > 0 {
        let mut avail: Vec<usize> = (0..inst.n_mach)
            .filter(|&m| mach_avail[m] <= time)
            .collect();
        if let Some(ref mut r) = rng.as_deref_mut() {
            let len = avail.len();
            for i in (1..len).rev() {
                let j = r.gen_range(0..=i);
                avail.swap(i, j);
            }
        }

        let avg_load = if inst.n_mach > 0 {
            mach_load.iter().sum::<u64>() as f64 / inst.n_mach as f64
        } else { 0.0 };

        let mut scheduled = false;
        for &m in &avail {
            let mut candidates: Vec<(usize, f64, u32, u32, usize)> = Vec::new();
            for job in 0..inst.n_jobs {
                let lo = job_next[job];
                if lo >= inst.n_ops[job] { continue; }
                if job_ready[job] > time { continue; }
                let g = inst.op_base[job] + lo;
                let pt_m = inst.pt(g, m);
                if pt_m == 0 { continue; }
                let mach_end = time.max(mach_avail[m]) + pt_m;
                let mut earliest = u32::MAX;
                for &em in &inst.elig[g] {
                    let end = time.max(mach_avail[em]) + inst.pt(g, em);
                    if end < earliest { earliest = end; }
                }
                // Allow ECT slack when machine debiasing is active
                let ect_slack = if mach_penalty_bias.is_some() && inst.flex[g] > 3 {
                    (earliest / 30).max(1) // ~3% slack for high-flex ops
                } else { 0 };
                if mach_end > earliest + ect_slack { continue; }
                let flex_g = inst.flex[g];
                let mut priority = match rule {
                    Rule::MWR => work_rem[job],
                    Rule::MOR => (inst.n_ops[job] - lo) as f64,
                    Rule::LF => -(flex_g as f64),
                    Rule::SPT => -(pt_m as f64),
                    Rule::LPT => pt_m as f64,
                    Rule::ECT => -(mach_end as f64),
                    Rule::LWORK => {
                        let load_penalty = if avg_load > 0.0 {
                            mach_load_bias * (mach_load[m] as f64 / avg_load - 1.0)
                        } else { 0.0 };
                        work_rem[job] - load_penalty * work_rem[job]
                    }
                    Rule::MOPNR => ops_rem[job] as f64 + 0.1 * work_rem[job] / (inst.op_work[g] + 1.0),
                };
                if let Some(jb) = job_bias {
                    priority *= 1.0 + 0.5 * jb[job];
                }
                if let Some(mpb) = mach_penalty_bias {
                    let ratio = mpb[m] / (avg_mpb + 0.001);
                    if ratio > 1.2 {
                        priority *= 1.0 - 0.15 * (ratio - 1.2).min(1.0);
                    }
                }
                candidates.push((job, priority, mach_end, pt_m, flex_g));
            }
            if candidates.is_empty() { continue; }
            candidates.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.2.cmp(&b.2))
                    .then(a.3.cmp(&b.3))
                    .then(a.4.cmp(&b.4))
                    .then(a.0.cmp(&b.0))
            });
            let pick = if top_k > 1 {
                if let Some(ref mut r) = rng.as_deref_mut() {
                    r.gen_range(0..top_k.min(candidates.len()))
                } else { 0 }
            } else { 0 };
            let (job, _, _, _, _) = candidates[pick];
            let lo = job_next[job];
            let g = inst.op_base[job] + lo;
            let pt_m = inst.pt(g, m);
            let start = time.max(mach_avail[m]);
            let end = start + pt_m;
            sched.assign[g] = m;
            sched.pt[g] = pt_m;
            sched.mseq[m].push(g);
            job_next[job] += 1;
            ops_rem[job] -= 1;
            job_ready[job] = end;
            mach_avail[m] = end;
            mach_load[m] += pt_m as u64;
            work_rem[job] -= inst.op_work[g];
            if work_rem[job] < 0.0 { work_rem[job] = 0.0; }
            remaining -= 1;
            scheduled = true;
        }
        if remaining == 0 { break; }
        let _ = scheduled;
        let mut next_time = u32::MAX;
        for &t in &mach_avail {
            if t > time && t < next_time { next_time = t; }
        }
        for job in 0..inst.n_jobs {
            if job_next[job] < inst.n_ops[job] {
                let t = job_ready[job];
                if t > time && t < next_time { next_time = t; }
            }
        }
        if next_time == u32::MAX { break; }
        time = next_time;
    }
    sched.build_mlinks(inst);
    let _ = sched.eval(inst);
    sched
}

// =========================================================
//  Move types
// =========================================================

#[derive(Clone)]
struct Move {
    kind: u8,       // 0=swap, 1=reroute, 2=insert(shift on same machine)
    machine: usize,
    pos_a: usize,
    pos_b: usize,
    g: usize,
    old_machine: usize,
    old_pos: usize,
    new_machine: usize,
    new_pos: usize,
}

impl Move {
    fn swap(machine: usize, pos_a: usize, pos_b: usize) -> Self {
        Move { kind: 0, machine, pos_a, pos_b, g: 0, old_machine: 0, old_pos: 0, new_machine: 0, new_pos: 0 }
    }
    fn reroute(g: usize, old_machine: usize, old_pos: usize, new_machine: usize, new_pos: usize) -> Self {
        Move { kind: 1, machine: 0, pos_a: 0, pos_b: 0, g, old_machine, old_pos, new_machine, new_pos }
    }
    fn insert(machine: usize, from: usize, to: usize, g: usize) -> Self {
        Move { kind: 2, machine, pos_a: from, pos_b: to, g, old_machine: 0, old_pos: 0, new_machine: 0, new_pos: 0 }
    }
}

fn apply_move(sched: &mut Sched, inst: &Inst, mv: &Move) {
    match mv.kind {
        0 => {
            // Swap adjacent positions — O(1) mlinks update
            sched.mseq[mv.machine].swap(mv.pos_a, mv.pos_b);
            let seq = &sched.mseq[mv.machine];
            let ga = seq[mv.pos_a];
            let gb = seq[mv.pos_b];
            let pred = if mv.pos_a > 0 { seq[mv.pos_a - 1] } else { NONE };
            let succ = if mv.pos_b + 1 < seq.len() { seq[mv.pos_b + 1] } else { NONE };
            sched.m_pred[ga] = pred;
            sched.m_succ[ga] = gb;
            sched.m_pred[gb] = ga;
            sched.m_succ[gb] = succ;
            if pred != NONE { sched.m_succ[pred] = ga; }
            if succ != NONE { sched.m_pred[succ] = gb; }
        }
        1 => {
            // Reroute: remove from old_machine, insert into new_machine — O(1) mlinks
            let g = mv.g;
            sched.mseq[mv.old_machine].remove(mv.old_pos);
            {
                let seq = &sched.mseq[mv.old_machine];
                let pred = if mv.old_pos > 0 { seq[mv.old_pos - 1] } else { NONE };
                let succ = if mv.old_pos < seq.len() { seq[mv.old_pos] } else { NONE };
                if pred != NONE { sched.m_succ[pred] = succ; }
                if succ != NONE { sched.m_pred[succ] = pred; }
            }
            sched.mseq[mv.new_machine].insert(mv.new_pos, g);
            sched.assign[g] = mv.new_machine;
            sched.pt[g] = inst.pt(g, mv.new_machine);
            {
                let seq = &sched.mseq[mv.new_machine];
                let pred = if mv.new_pos > 0 { seq[mv.new_pos - 1] } else { NONE };
                let succ = if mv.new_pos + 1 < seq.len() { seq[mv.new_pos + 1] } else { NONE };
                sched.m_pred[g] = pred;
                sched.m_succ[g] = succ;
                if pred != NONE { sched.m_succ[pred] = g; }
                if succ != NONE { sched.m_pred[succ] = g; }
            }
        }
        2 => {
            // Insert on same machine — O(1) mlinks
            let g = sched.mseq[mv.machine].remove(mv.pos_a);
            {
                let seq = &sched.mseq[mv.machine];
                let pred = if mv.pos_a > 0 { seq[mv.pos_a - 1] } else { NONE };
                let succ = if mv.pos_a < seq.len() { seq[mv.pos_a] } else { NONE };
                if pred != NONE { sched.m_succ[pred] = succ; }
                if succ != NONE { sched.m_pred[succ] = pred; }
            }
            let adj = if mv.pos_b > mv.pos_a { mv.pos_b - 1 } else { mv.pos_b };
            sched.mseq[mv.machine].insert(adj, g);
            {
                let seq = &sched.mseq[mv.machine];
                sched.m_pred[g] = if adj > 0 { seq[adj - 1] } else { NONE };
                sched.m_succ[g] = if adj + 1 < seq.len() { seq[adj + 1] } else { NONE };
                if adj > 0 { sched.m_succ[seq[adj - 1]] = g; }
                if adj + 1 < seq.len() { sched.m_pred[seq[adj + 1]] = g; }
            }
        }
        _ => {}
    }
}

fn undo_move(sched: &mut Sched, inst: &Inst, mv: &Move) {
    match mv.kind {
        0 => {
            // Swap is self-inverse — O(1) mlinks
            sched.mseq[mv.machine].swap(mv.pos_a, mv.pos_b);
            let seq = &sched.mseq[mv.machine];
            let ga = seq[mv.pos_a];
            let gb = seq[mv.pos_b];
            let pred = if mv.pos_a > 0 { seq[mv.pos_a - 1] } else { NONE };
            let succ = if mv.pos_b + 1 < seq.len() { seq[mv.pos_b + 1] } else { NONE };
            sched.m_pred[ga] = pred;
            sched.m_succ[ga] = gb;
            sched.m_pred[gb] = ga;
            sched.m_succ[gb] = succ;
            if pred != NONE { sched.m_succ[pred] = ga; }
            if succ != NONE { sched.m_pred[succ] = gb; }
        }
        1 => {
            // Undo reroute: remove from new_machine at known pos, insert back to old_machine
            let g = mv.g;
            sched.mseq[mv.new_machine].remove(mv.new_pos);
            {
                let seq = &sched.mseq[mv.new_machine];
                let pred = if mv.new_pos > 0 { seq[mv.new_pos - 1] } else { NONE };
                let succ = if mv.new_pos < seq.len() { seq[mv.new_pos] } else { NONE };
                if pred != NONE { sched.m_succ[pred] = succ; }
                if succ != NONE { sched.m_pred[succ] = pred; }
            }
            sched.mseq[mv.old_machine].insert(mv.old_pos, g);
            sched.assign[g] = mv.old_machine;
            sched.pt[g] = inst.pt(g, mv.old_machine);
            {
                let seq = &sched.mseq[mv.old_machine];
                let pred = if mv.old_pos > 0 { seq[mv.old_pos - 1] } else { NONE };
                let succ = if mv.old_pos + 1 < seq.len() { seq[mv.old_pos + 1] } else { NONE };
                sched.m_pred[g] = pred;
                sched.m_succ[g] = succ;
                if pred != NONE { sched.m_succ[pred] = g; }
                if succ != NONE { sched.m_pred[succ] = g; }
            }
        }
        2 => {
            // Undo insert: move g from known adj position back to pos_a
            let adj = if mv.pos_b > mv.pos_a { mv.pos_b - 1 } else { mv.pos_b };
            let g = mv.g;
            sched.mseq[mv.machine].remove(adj);
            {
                let seq = &sched.mseq[mv.machine];
                let pred = if adj > 0 { seq[adj - 1] } else { NONE };
                let succ = if adj < seq.len() { seq[adj] } else { NONE };
                if pred != NONE { sched.m_succ[pred] = succ; }
                if succ != NONE { sched.m_pred[succ] = pred; }
            }
            sched.mseq[mv.machine].insert(mv.pos_a, g);
            {
                let seq = &sched.mseq[mv.machine];
                sched.m_pred[g] = if mv.pos_a > 0 { seq[mv.pos_a - 1] } else { NONE };
                sched.m_succ[g] = if mv.pos_a + 1 < seq.len() { seq[mv.pos_a + 1] } else { NONE };
                if mv.pos_a > 0 { sched.m_succ[seq[mv.pos_a - 1]] = g; }
                if mv.pos_a + 1 < seq.len() { sched.m_pred[seq[mv.pos_a + 1]] = g; }
            }
        }
        _ => {}
    }
}

fn find_insert_pos(sched: &Sched, inst: &Inst, g: usize, m: usize) -> Option<usize> {
    let seq = &sched.mseq[m];
    let mut min_pos = 0usize;
    let mut jp = inst.j_pred[g];
    while jp != NONE {
        for (i, &op) in seq.iter().enumerate() {
            if op == jp { min_pos = min_pos.max(i + 1); }
        }
        jp = inst.j_pred[jp];
    }
    let mut max_pos = seq.len();
    let mut js = inst.j_succ[g];
    while js != NONE {
        for (i, &op) in seq.iter().enumerate() {
            if op == js && i < max_pos { max_pos = i; }
        }
        js = inst.j_succ[js];
    }
    if min_pos > max_pos { return None; }
    let target = sched.head[g];
    let pos = (min_pos..=max_pos)
        .find(|&p| p == max_pos || sched.head[seq[p]] >= target)
        .unwrap_or(max_pos);
    Some(pos)
}

// =========================================================
//  Tabu search
// =========================================================

struct TabuParams {
    tenure: usize,
    stall_limit: usize,
    max_reroute_alts: usize,
    reroute_pt_mult: u32, // accept machines up to cur_pt * mult / 100
    n_perturb: usize,
}

fn tabu_search(
    inst: &Inst,
    sched: &mut Sched,
    rng: &mut SmallRng,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    global_best: &mut u32,
    max_iters: usize,
    params: &TabuParams,
) -> Result<()> {
    let mut tabu_list: Vec<(usize, usize)> = Vec::new();
    let mut iter = 0usize;
    let mut stall = 0usize;
    let mut local_best_ms = sched.makespan;
    let mut perturb_count = 0usize;
    let mut best_known = Sched::new(inst);
    best_known.clone_from_sched(sched, inst);
    let _ = best_known.eval(inst);
    let mut mach_penalty = vec![0i32; inst.n_mach];

    while iter < max_iters {
        let _ = sched.eval(inst);
        let blocks = find_critical_blocks(sched, inst);
        if blocks.is_empty() { break; }

        let mut best_mv: Option<Move> = None;
        let mut best_ms = u32::MAX;
        let mut best_tabu_mv: Option<Move> = None;
        let mut best_tabu_ms = u32::MAX;

        struct SwapCand { machine: usize, pa: usize, pb: usize, ga: usize, gb: usize }
        struct RerouteCand { g: usize, cur_m: usize, pos: usize, new_m: usize }
        struct InsertCand { machine: usize, from: usize, to: usize, g: usize }
        let mut swap_cands: Vec<SwapCand> = Vec::new();
        let mut reroute_cands: Vec<RerouteCand> = Vec::new();
        let mut insert_cands: Vec<InsertCand> = Vec::new();

        for block in &blocks {
            let seq = &sched.mseq[block.machine];
            let blen = block.end - block.start;

            // N5 swap moves: first pair and last pair of critical block
            if blen >= 2 {
                swap_cands.push(SwapCand {
                    machine: block.machine, pa: block.start, pb: block.start + 1,
                    ga: seq[block.start], gb: seq[block.start + 1],
                });
                if blen > 2 {
                    swap_cands.push(SwapCand {
                        machine: block.machine, pa: block.end - 2, pb: block.end - 1,
                        ga: seq[block.end - 2], gb: seq[block.end - 1],
                    });
                }
            }

            // N7 insert moves: move first/last op to positions within block
            // Generate all candidates but with Taillard estimates for screening
            if blen >= 3 {
                let first_g = seq[block.start];
                let last_g = seq[block.end - 1];
                let ms = sched.makespan;
                let jp_fg = inst.j_pred[first_g];
                let js_fg = inst.j_succ[first_g];
                let c_jp_fg = if jp_fg != NONE { sched.head[jp_fg] + sched.pt[jp_fg] } else { 0 };
                let r_js_fg = if js_fg != NONE { sched.pt[js_fg] + sched.tail[js_fg] } else { 0 };
                // First op to various positions: pred = seq[to-1], succ = seq[to]
                for to in (block.start + 2)..block.end {
                    let c_pred = sched.head[seq[to - 1]] + sched.pt[seq[to - 1]];
                    let r_succ = sched.pt[seq[to]] + sched.tail[seq[to]];
                    let est = c_jp_fg.max(c_pred) + sched.pt[first_g] + r_js_fg.max(r_succ);
                    if est <= ms {  // filter: only if estimate shows potential improvement
                        insert_cands.push(InsertCand {
                            machine: block.machine, from: block.start, to, g: first_g,
                        });
                    }
                }
                let jp_lg = inst.j_pred[last_g];
                let js_lg = inst.j_succ[last_g];
                let c_jp_lg = if jp_lg != NONE { sched.head[jp_lg] + sched.pt[jp_lg] } else { 0 };
                let r_js_lg = if js_lg != NONE { sched.pt[js_lg] + sched.tail[js_lg] } else { 0 };
                // Last op to various positions: pred = seq[to-1], succ = seq[to]
                for to in block.start..(block.end - 2) {
                    let c_pred = if to > 0 { sched.head[seq[to - 1]] + sched.pt[seq[to - 1]] } else { 0 };
                    let r_succ = sched.pt[seq[to]] + sched.tail[seq[to]];
                    let est = c_jp_lg.max(c_pred) + sched.pt[last_g] + r_js_lg.max(r_succ);
                    if est <= ms {
                        insert_cands.push(InsertCand {
                            machine: block.machine, from: block.end - 1, to, g: last_g,
                        });
                    }
                }
            }

            // Reroute candidates (skip entirely for low_flex — no alternatives exist)
            if inst.flex_avg > 1.5 {
                let rr_positions: Vec<usize> = if blen <= 6 {
                    (block.start..block.end).collect()
                } else {
                    vec![block.start, block.start + 1, block.end - 2, block.end - 1]
                };
                for pos in rr_positions {
                    let g = seq[pos];
                    if inst.flex[g] <= 1 { continue; }
                    let cur_m = sched.assign[g];
                    let cur_pt = sched.pt[g];
                    let threshold = (cur_pt as u64 * params.reroute_pt_mult as u64 / 100) as u32;
                    let mut alts: Vec<(usize, u32)> = inst.elig[g].iter()
                        .filter(|&&m| m != cur_m)
                        .map(|&m| (m, inst.pt(g, m)))
                        .filter(|&(_, pt_m)| pt_m <= threshold)
                        .collect();
                    alts.sort_by_key(|&(_, pt)| pt);
                    for &(m, _) in alts.iter().take(params.max_reroute_alts) {
                        reroute_cands.push(RerouteCand { g, cur_m, pos, new_m: m });
                    }
                }
            }
        }

        // Additional reroute: critical path ops not in any block (top 5 by tail)
        if inst.flex_avg > 1.5 {
            let ms = sched.makespan;
            let mut extra_crit: Vec<(usize, u32)> = Vec::new();
            for g in 0..inst.total_ops {
                if inst.flex[g] <= 1 { continue; }
                if sched.head[g] + sched.pt[g] + sched.tail[g] != ms { continue; }
                if reroute_cands.iter().any(|c| c.g == g) { continue; }
                extra_crit.push((g, sched.tail[g]));
            }
            extra_crit.sort_by(|a, b| b.1.cmp(&a.1));
            let extra_take = if inst.flex_avg > 6.0 { 6 } else { 3 };
            for &(g, _) in extra_crit.iter().take(extra_take) {
                let cur_m = sched.assign[g];
                let pos = match sched.mseq[cur_m].iter().position(|&x| x == g) {
                    Some(p) => p,
                    None => continue,
                };
                let cur_pt = sched.pt[g];
                let threshold = (cur_pt as u64 * params.reroute_pt_mult as u64 / 100) as u32;
                let mut alts: Vec<(usize, u32)> = inst.elig[g].iter()
                    .filter(|&&m| m != cur_m)
                    .map(|&m| (m, inst.pt(g, m)))
                    .filter(|&(_, pt_m)| pt_m <= threshold)
                    .collect();
                alts.sort_by_key(|&(_, pt)| pt);
                let extra_alts = if inst.flex_avg > 6.0 { 4 } else { 2 };
                for &(m, _) in alts.iter().take(extra_alts) {
                    reroute_cands.push(RerouteCand { g, cur_m, pos, new_m: m });
                }
            }
        }

        // Evaluate swap candidates (forward-only eval, always evaluate — cheap with only 2-4 per iteration)
        for cand in &swap_cands {
            let mv = Move::swap(cand.machine, cand.pa, cand.pb);
            apply_move(sched, inst, &mv);
            if sched.eval_forward(inst) {
                let ms = sched.makespan;
                let is_tabu = tabu_list.iter().any(|&(op, exp)| exp > iter && (op == cand.ga || op == cand.gb));
                if is_tabu {
                    if ms < best_tabu_ms { best_tabu_ms = ms; best_tabu_mv = Some(mv.clone()); }
                } else if ms < best_ms {
                    best_ms = ms; best_mv = Some(mv.clone());
                }
            }
            undo_move(sched, inst, &mv);
        }

        // Evaluate insert candidates (forward-only eval)
        for cand in &insert_cands {
            let mv = Move::insert(cand.machine, cand.from, cand.to, cand.g);
            apply_move(sched, inst, &mv);
            if sched.eval_forward(inst) {
                let ms = sched.makespan;
                let is_tabu = tabu_list.iter().any(|&(op, exp)| exp > iter && op == cand.g);
                if is_tabu {
                    if ms < best_tabu_ms { best_tabu_ms = ms; best_tabu_mv = Some(mv.clone()); }
                } else if ms < best_ms {
                    best_ms = ms; best_mv = Some(mv.clone());
                }
            }
            undo_move(sched, inst, &mv);
        }

        // Evaluate reroute candidates
        let mut reroute_improved = false;
        let mut reroute_eval_count = 0usize;

        if !reroute_cands.is_empty() {
            // Re-eval forward to restore correct head values for find_insert_pos
            let _ = sched.eval_forward(inst);

            // Sort reroute candidates by machine penalty + processing time
            reroute_cands.sort_by_key(|c| {
                let penalty = mach_penalty[c.new_m];
                let pt = inst.pt(c.g, c.new_m) as i32;
                penalty * 100 + pt
            });
        }

        let max_reroute_eval: usize = if inst.flex_avg > 6.0 { 50 } else { 20 };
        for cand in &reroute_cands {
            if reroute_eval_count >= max_reroute_eval { break; }
            let ins_pos = match find_insert_pos(sched, inst, cand.g, cand.new_m) {
                Some(p) => p,
                None => continue,
            };
            reroute_eval_count += 1;
            let mv = Move::reroute(cand.g, cand.cur_m, cand.pos, cand.new_m, ins_pos);
            apply_move(sched, inst, &mv);
            if sched.eval_forward(inst) {
                let ms = sched.makespan;
                let is_tabu = tabu_list.iter().any(|&(op, exp)| exp > iter && op == cand.g);
                if is_tabu {
                    if ms < best_tabu_ms { best_tabu_ms = ms; best_tabu_mv = Some(mv.clone()); }
                } else if ms < best_ms {
                    best_ms = ms; best_mv = Some(mv.clone());
                    reroute_improved = true;
                }
            }
            undo_move(sched, inst, &mv);
        }

        // Aspiration: accept tabu move if it beats global best
        if best_mv.is_none() || (best_tabu_ms < *global_best && best_tabu_ms < best_ms) {
            if let Some(tmv) = best_tabu_mv {
                best_mv = Some(tmv);
                best_ms = best_tabu_ms;
            }
        }

        // Update machine penalties based on reroute results
        if reroute_improved {
            if let Some(ref mv) = best_mv {
                if mv.kind == 1 {
                    mach_penalty[mv.new_machine] = (mach_penalty[mv.new_machine] - 3).max(0);
                }
            }
        } else if reroute_eval_count > 0 {
            for cand in reroute_cands.iter().take(reroute_eval_count) {
                mach_penalty[cand.new_m] = (mach_penalty[cand.new_m] + 1).min(20);
            }
        }

        if let Some(ref mv) = best_mv {
            apply_move(sched, inst, mv);
            match mv.kind {
                0 => {
                    let ga = sched.mseq[mv.machine][mv.pos_a];
                    let gb = sched.mseq[mv.machine][mv.pos_b];
                    tabu_list.push((ga, iter + params.tenure));
                    tabu_list.push((gb, iter + params.tenure));
                }
                1 => { tabu_list.push((mv.g, iter + params.tenure)); }
                2 => { tabu_list.push((mv.g, iter + params.tenure)); }
                _ => {}
            }
            tabu_list.retain(|&(_, exp)| exp > iter);

            if best_ms < local_best_ms {
                local_best_ms = best_ms;
                stall = 0;
            } else {
                stall += 1;
            }

            if best_ms < *global_best {
                *global_best = best_ms;
                let _ = sched.eval_forward(inst);
                best_known.clone_from_sched(sched, inst);
                save_solution(&sched.to_solution(inst))?;
                stall = 0;
            }
        } else {
            stall += 1;
        }

        // Perturbation when stuck
        if stall >= params.stall_limit {
            perturb_count += 1;

            // Every 6th perturbation: restart from best known solution
            if perturb_count % 6 == 0 && best_known.makespan < u32::MAX {
                sched.clone_from_sched(&best_known, inst);
                let _ = sched.eval(inst);
                // Apply a large random perturbation to explore new region
                let n_moves = params.n_perturb * 3;
                let blocks = find_critical_blocks(sched, inst);
                for _ in 0..n_moves {
                    if blocks.is_empty() { break; }
                    let bi = rng.gen_range(0..blocks.len());
                    let block = &blocks[bi];
                    let blen = block.end - block.start;
                    if blen < 2 { continue; }
                    let seq_len = sched.mseq[block.machine].len();
                    if block.start + 1 >= seq_len { continue; }
                    let mp = (block.end - 1).min(seq_len - 1);
                    if mp <= block.start { continue; }
                    let pos = block.start + rng.gen_range(0..mp - block.start);
                    sched.mseq[block.machine].swap(pos, pos + 1);
                    sched.build_mlinks_machine(block.machine);
                    if !sched.eval(inst) {
                        sched.mseq[block.machine].swap(pos, pos + 1);
                        sched.build_mlinks_machine(block.machine);
                    }
                }
            } else {
                // Escalating perturbation: more moves after repeated stalls
                let scale = 1 + (perturb_count / 4).min(2);
                let n_moves = params.n_perturb * scale;

                let _ = sched.eval(inst);
                let blocks = find_critical_blocks(sched, inst);
                for _ in 0..n_moves {
                    if blocks.is_empty() { break; }
                    let bi = rng.gen_range(0..blocks.len());
                    let block = &blocks[bi];
                    let blen = block.end - block.start;
                    if blen < 2 { continue; }

                    // Mix of swap and reroute perturbation
                    let do_reroute = if inst.flex_avg > 6.0 {
                        rng.gen_range(0..4) > 0 // 75% reroute for very high flex
                    } else {
                        inst.flex_avg > 2.0 && rng.gen_range(0..3) == 0
                    };
                    if do_reroute {
                        let seq_len = sched.mseq[block.machine].len();
                        let eff_end = block.end.min(seq_len);
                        if eff_end <= block.start { continue; }
                        let pos = block.start + rng.gen_range(0..eff_end - block.start);
                        let g = sched.mseq[block.machine][pos];
                        if inst.flex[g] > 1 {
                            let cur_m = sched.assign[g];
                            let alts: Vec<usize> = inst.elig[g].iter()
                                .filter(|&&m| m != cur_m)
                                .copied()
                                .collect();
                            if !alts.is_empty() {
                                let new_m = alts[rng.gen_range(0..alts.len())];
                                if let Some(ins_pos) = find_insert_pos(sched, inst, g, new_m) {
                                    sched.mseq[block.machine].remove(pos);
                                    sched.build_mlinks_machine(block.machine);
                                    let mx = sched.mseq[new_m].len();
                                    sched.mseq[new_m].insert(ins_pos.min(mx), g);
                                    sched.assign[g] = new_m;
                                    sched.pt[g] = inst.pt(g, new_m);
                                    sched.build_mlinks_machine(new_m);
                                }
                            }
                        }
                    } else {
                        let seq_len = sched.mseq[block.machine].len();
                        if block.start + 1 >= seq_len { continue; }
                        let mp = (block.end - 1).min(seq_len - 1);
                        if mp <= block.start { continue; }
                        let pos = block.start + rng.gen_range(0..mp - block.start);
                        sched.mseq[block.machine].swap(pos, pos + 1);
                        sched.build_mlinks_machine(block.machine);
                        if !sched.eval(inst) {
                            sched.mseq[block.machine].swap(pos, pos + 1);
                            sched.build_mlinks_machine(block.machine);
                        }
                    }
                }
            }
            tabu_list.clear();
            let _ = sched.eval(inst);
            local_best_ms = sched.makespan;
            stall = 0;
        }

        iter += 1;
    }
    // Restore best known solution
    if best_known.makespan < sched.makespan {
        sched.clone_from_sched(&best_known, inst);
        let _ = sched.eval(inst);
    }
    Ok(())
}

// =========================================================
//  Load-balanced construction (for high-flex FJSP)
// =========================================================

fn construct_loadbalanced(inst: &Inst, rng: &mut SmallRng, top_k: usize) -> Sched {
    let mut mach_load = vec![0u64; inst.n_mach];
    let mut assign = vec![0usize; inst.total_ops];
    let mut job_order: Vec<usize> = (0..inst.n_jobs).collect();
    for i in (1..inst.n_jobs).rev() {
        let j = rng.gen_range(0..=i);
        job_order.swap(i, j);
    }
    for &job in &job_order {
        for lo in 0..inst.n_ops[job] {
            let g = inst.op_base[job] + lo;
            let mut candidates: Vec<(usize, u64)> = inst.elig[g].iter()
                .map(|&m| (m, mach_load[m] + inst.pt(g, m) as u64))
                .collect();
            candidates.sort_by_key(|&(_, cost)| cost);
            let pick = if top_k > 1 && candidates.len() > 1 {
                rng.gen_range(0..top_k.min(candidates.len()))
            } else { 0 };
            let (m, _) = candidates[pick];
            assign[g] = m;
            mach_load[m] += inst.pt(g, m) as u64;
        }
    }
    construct_from_assign(inst, &assign, rng, 0.0)
}

// =========================================================
//  Solution-guided construction
// =========================================================

fn construct_from_assign(inst: &Inst, guide: &[usize], rng: &mut SmallRng, mutate_prob: f64) -> Sched {
    let mut assign = guide.to_vec();
    for g in 0..inst.total_ops {
        if inst.flex[g] > 1 && rng.gen::<f64>() < mutate_prob {
            let alt = &inst.elig[g];
            assign[g] = alt[rng.gen_range(0..alt.len())];
        }
    }
    let mut sched = Sched::new(inst);
    let mut job_next = vec![0usize; inst.n_jobs];
    let mut job_ready = vec![0u32; inst.n_jobs];
    let mut mach_avail = vec![0u32; inst.n_mach];
    let mut remaining = inst.total_ops;
    while remaining > 0 {
        let mut best_job = NONE;
        let mut best_end = u32::MAX;
        for job in 0..inst.n_jobs {
            let lo = job_next[job];
            if lo >= inst.n_ops[job] { continue; }
            let g = inst.op_base[job] + lo;
            let m = assign[g];
            let pt_m = inst.pt(g, m);
            if pt_m == 0 { continue; }
            let start = job_ready[job].max(mach_avail[m]);
            let end = start + pt_m;
            if end < best_end {
                best_end = end;
                best_job = job;
            }
        }
        if best_job == NONE { break; }
        let lo = job_next[best_job];
        let g = inst.op_base[best_job] + lo;
        let m = assign[g];
        let pt_m = inst.pt(g, m);
        let end = job_ready[best_job].max(mach_avail[m]) + pt_m;
        sched.assign[g] = m;
        sched.pt[g] = pt_m;
        sched.mseq[m].push(g);
        job_next[best_job] += 1;
        job_ready[best_job] = end;
        mach_avail[m] = end;
        remaining -= 1;
    }
    sched.build_mlinks(inst);
    let _ = sched.eval(inst);
    sched
}

// =========================================================
//  Flow shop optimization (NEH + local search)
// =========================================================

/// Detect flow shop: all ops have flex=1 and all products share same machine sequence.
/// Returns the common machine sequence if flow shop, None otherwise.
fn detect_flow_shop(inst: &Inst) -> Option<Vec<usize>> {
    if inst.flex_avg > 1.01 { return None; }
    let n_ops0 = inst.n_ops[0];
    let base0 = inst.op_base[0];
    let ref_seq: Vec<usize> = (0..n_ops0).map(|lo| inst.elig[base0 + lo][0]).collect();
    for job in 1..inst.n_jobs {
        if inst.n_ops[job] != n_ops0 { return None; }
        let base = inst.op_base[job];
        for lo in 0..n_ops0 {
            if inst.elig[base + lo].len() != 1 || inst.elig[base + lo][0] != ref_seq[lo] {
                return None;
            }
        }
    }
    Some(ref_seq)
}

/// Fast O(n_jobs × n_ops) makespan for a flow shop permutation (per-machine availability).
fn eval_perm_ms(perm: &[usize], inst: &Inst, mach_seq: &[usize]) -> u32 {
    let n_ops = mach_seq.len();
    let mut mach_avail = vec![0u32; inst.n_mach];
    let mut makespan = 0u32;
    for &job in perm {
        let base = inst.op_base[job];
        let mut job_prev = 0u32;
        for oi in 0..n_ops {
            let m = mach_seq[oi];
            let pt = inst.pt(base + oi, m);
            let start = job_prev.max(mach_avail[m]);
            let end = start + pt;
            mach_avail[m] = end;
            job_prev = end;
        }
        if job_prev > makespan { makespan = job_prev; }
    }
    makespan
}

/// NEH construction: sort jobs by descending total PT, insert each at best position.
fn neh_construction(inst: &Inst, mach_seq: &[usize]) -> Vec<usize> {
    let mut total_pt: Vec<(usize, u64)> = (0..inst.n_jobs).map(|job| {
        let base = inst.op_base[job];
        let sum: u64 = (0..inst.n_ops[job]).map(|lo| {
            inst.pt(base + lo, mach_seq[lo]) as u64
        }).sum();
        (job, sum)
    }).collect();
    total_pt.sort_by(|a, b| b.1.cmp(&a.1));

    let mut perm: Vec<usize> = vec![total_pt[0].0];
    for i in 1..inst.n_jobs {
        let job = total_pt[i].0;
        let mut best_ms = u32::MAX;
        let mut best_pos = 0;
        for pos in 0..=perm.len() {
            perm.insert(pos, job);
            let ms = eval_perm_ms(&perm, inst, mach_seq);
            if ms < best_ms { best_ms = ms; best_pos = pos; }
            perm.remove(pos);
        }
        perm.insert(best_pos, job);
    }
    perm
}


/// Convert a flow shop permutation to a Sched.
fn perm_to_sched(perm: &[usize], inst: &Inst, mach_seq: &[usize]) -> Sched {
    let mut sched = Sched::new(inst);
    let n_ops = mach_seq.len();
    for &job in perm {
        let base = inst.op_base[job];
        for oi in 0..n_ops {
            let g = base + oi;
            let m = mach_seq[oi];
            sched.assign[g] = m;
            sched.pt[g] = inst.pt(g, m);
            sched.mseq[m].push(g);
        }
    }
    sched.build_mlinks(inst);
    let _ = sched.eval(inst);
    sched
}

/// Permutation-based local search for flow shop.
/// Applies: insertion LS, adjacent swaps, random swaps, 2-opt segment reversal.
fn flow_shop_perm_search(
    inst: &Inst,
    mach_seq: &[usize],
    perm: &mut Vec<usize>,
    rng: &mut SmallRng,
) -> u32 {
    let n = perm.len();
    let mut best_ms = eval_perm_ms(perm, inst, mach_seq);

    // Pass 1: Insertion-based local search (2 passes)
    for _pass in 0..2 {
        let mut improved = false;
        for i in 0..n {
            let job = perm.remove(i);
            let mut best_pos = i;
            let mut best_insert_ms = u32::MAX;
            for pos in 0..=perm.len() {
                perm.insert(pos, job);
                let ms = eval_perm_ms(perm, inst, mach_seq);
                if ms < best_insert_ms {
                    best_insert_ms = ms;
                    best_pos = pos;
                }
                perm.remove(pos);
            }
            perm.insert(best_pos, job);
            if best_insert_ms < best_ms {
                best_ms = best_insert_ms;
                improved = true;
            }
        }
        if !improved { break; }
    }

    // Pass 2: Adjacent swap passes
    for _pass in 0..3 {
        let mut improved = false;
        for i in 0..n - 1 {
            perm.swap(i, i + 1);
            let ms = eval_perm_ms(perm, inst, mach_seq);
            if ms < best_ms {
                best_ms = ms;
                improved = true;
            } else {
                perm.swap(i, i + 1);
            }
        }
        if !improved { break; }
    }

    // Pass 3: Random swap phase
    let budget = (n * 8).clamp(160, 600);
    for _ in 0..budget {
        let i = rng.gen_range(0..n);
        let j = rng.gen_range(0..n);
        if i == j { continue; }
        perm.swap(i, j);
        let ms = eval_perm_ms(perm, inst, mach_seq);
        if ms < best_ms {
            best_ms = ms;
        } else {
            perm.swap(i, j);
        }
    }

    // Pass 4: 2-opt segment reversal (segments up to length 5)
    let max_seg = 5.min(n);
    for seg_len in 2..=max_seg {
        let mut improved = true;
        while improved {
            improved = false;
            for i in 0..=n - seg_len {
                perm[i..i + seg_len].reverse();
                let ms = eval_perm_ms(perm, inst, mach_seq);
                if ms < best_ms {
                    best_ms = ms;
                    improved = true;
                } else {
                    perm[i..i + seg_len].reverse();
                }
            }
        }
    }

    // Final: adjacent swap cleanup
    for _pass in 0..2 {
        let mut improved = false;
        for i in 0..n - 1 {
            perm.swap(i, i + 1);
            let ms = eval_perm_ms(perm, inst, mach_seq);
            if ms < best_ms {
                best_ms = ms;
                improved = true;
            } else {
                perm.swap(i, i + 1);
            }
        }
        if !improved { break; }
    }

    best_ms
}

/// Iterated Greedy for flow shop: destroy d jobs, reinsert via NEH, Metropolis acceptance.
fn iterated_greedy_flow_shop(
    inst: &Inst,
    mach_seq: &[usize],
    perm: &mut Vec<usize>,
    rng: &mut SmallRng,
    max_iters: usize,
) -> u32 {
    let n = perm.len();
    let d = 6.min(n / 2);
    let mut best_ms = eval_perm_ms(perm, inst, mach_seq);
    let mut best_perm = perm.clone();

    // Temperature based on average processing time
    let sum_pt: u64 = (0..inst.n_jobs).map(|job| {
        let base = inst.op_base[job];
        (0..inst.n_ops[job]).map(|lo| inst.pt(base + lo, mach_seq[lo]) as u64).sum::<u64>()
    }).sum();
    let temp = 0.4 * sum_pt as f64 / (n as f64 * inst.n_ops[0] as f64 * 10.0);

    let mut cur_ms = best_ms;
    let mut cur_perm = perm.clone();

    for _ in 0..max_iters {
        let prev_perm = cur_perm.clone();
        let prev_ms = cur_ms;

        // Destroy: remove d random jobs
        let mut removed = Vec::with_capacity(d);
        for _ in 0..d {
            let idx = rng.gen_range(0..cur_perm.len());
            removed.push(cur_perm.remove(idx));
        }

        // Construct: NEH-insertion of each removed job
        for &job in &removed {
            let mut best_pos = 0;
            let mut best_insert_ms = u32::MAX;
            for pos in 0..=cur_perm.len() {
                cur_perm.insert(pos, job);
                let ms = eval_perm_ms(&cur_perm, inst, mach_seq);
                if ms < best_insert_ms {
                    best_insert_ms = ms;
                    best_pos = pos;
                }
                cur_perm.remove(pos);
            }
            cur_perm.insert(best_pos, job);
        }

        let new_ms = eval_perm_ms(&cur_perm, inst, mach_seq);

        // Metropolis acceptance
        if new_ms <= prev_ms {
            cur_ms = new_ms;
        } else {
            let delta = (new_ms - prev_ms) as f64;
            let accept_prob = (-delta / temp).exp();
            if rng.gen::<f64>() < accept_prob {
                cur_ms = new_ms;
            } else {
                cur_perm = prev_perm;
                cur_ms = prev_ms;
            }
        }

        if cur_ms < best_ms {
            best_ms = cur_ms;
            best_perm = cur_perm.clone();
        }
    }

    *perm = best_perm;
    best_ms
}

// =========================================================
//  Learning: extract feedback from best solutions
// =========================================================

/// Machine penalty: penalize overloaded machines
fn compute_machine_penalty(sched: &Sched, inst: &Inst) -> Vec<f64> {
    let mut load = vec![0u64; inst.n_mach];
    let mut end_time = vec![0u32; inst.n_mach];
    for m in 0..inst.n_mach {
        for &g in &sched.mseq[m] {
            load[m] += sched.pt[g] as u64;
            let e = sched.head[g] + sched.pt[g];
            if e > end_time[m] { end_time[m] = e; }
        }
    }
    let avg_load = load.iter().sum::<u64>() as f64 / inst.n_mach as f64;
    let avg_end = end_time.iter().sum::<u32>() as f64 / inst.n_mach as f64;
    let mut penalty = vec![0.0f64; inst.n_mach];
    for m in 0..inst.n_mach {
        let load_ratio = if avg_load > 0.0 { load[m] as f64 / avg_load } else { 1.0 };
        let end_ratio = if avg_end > 0.0 { end_time[m] as f64 / avg_end } else { 1.0 };
        penalty[m] = 0.6 * end_ratio + 0.4 * load_ratio - 1.0;
    }
    penalty
}

/// Route preferences: for each (product, op_index), record top-2 machines used
fn extract_route_pref(sched: &Sched, inst: &Inst) -> Vec<Vec<[usize; 2]>> {
    // route_pref[product][op_idx] = [best_machine, second_best_machine]
    let n_products = inst.prod.iter().max().map_or(0, |&p| p + 1);
    let mut pref: Vec<Vec<[usize; 2]>> = Vec::with_capacity(n_products);
    for p in 0..n_products {
        // Find first job of this product
        let mut ops_count = 0;
        for job in 0..inst.n_jobs {
            if inst.prod[job] == p {
                ops_count = inst.n_ops[job];
                break;
            }
        }
        let mut op_pref = vec![[NONE, NONE]; ops_count];
        // Count machine usage for each op across jobs of this product
        let mut counts: Vec<Vec<(usize, u32)>> = vec![Vec::new(); ops_count];
        for job in 0..inst.n_jobs {
            if inst.prod[job] != p { continue; }
            for lo in 0..inst.n_ops[job] {
                let g = inst.op_base[job] + lo;
                let m = sched.assign[g];
                if let Some(entry) = counts[lo].iter_mut().find(|e| e.0 == m) {
                    entry.1 += 1;
                } else {
                    counts[lo].push((m, 1));
                }
            }
        }
        for lo in 0..ops_count {
            counts[lo].sort_by(|a, b| b.1.cmp(&a.1));
            if !counts[lo].is_empty() { op_pref[lo][0] = counts[lo][0].0; }
            if counts[lo].len() > 1 { op_pref[lo][1] = counts[lo][1].0; }
        }
        pref.push(op_pref);
    }
    pref
}

/// Job completion bias: jobs finishing late get high bias to be scheduled earlier next time
fn compute_job_bias(sched: &Sched, inst: &Inst) -> Vec<f64> {
    let ms = sched.makespan;
    if ms == 0 || ms == u32::MAX { return vec![0.0; inst.n_jobs]; }
    let exp = 3.0
        + if inst.flex_avg > 3.0 { 1.2 } else { 0.0 }
        + if inst.flex_avg < 1.5 { 0.6 } else { 0.0 };
    let mut bias = vec![0.0f64; inst.n_jobs];
    for job in 0..inst.n_jobs {
        let mut completion = 0u32;
        for lo in 0..inst.n_ops[job] {
            let g = inst.op_base[job] + lo;
            let end = sched.head[g] + sched.pt[g];
            if end > completion { completion = end; }
        }
        bias[job] = (completion as f64 / ms as f64).powf(exp);
    }
    bias
}

/// Construct from assignment with load-aware insertion ordering
fn construct_from_assign_smart(
    inst: &Inst,
    guide: &[usize],
    rng: &mut SmallRng,
    mutate_prob: f64,
    mach_penalty: &[f64],
) -> Sched {
    let mut assign = guide.to_vec();
    for g in 0..inst.total_ops {
        if inst.flex[g] > 1 && rng.gen::<f64>() < mutate_prob {
            let alt = &inst.elig[g];
            // Prefer machines with lower penalty
            if alt.len() <= 3 || rng.gen::<f64>() < 0.5 {
                // Random alternative
                assign[g] = alt[rng.gen_range(0..alt.len())];
            } else {
                // Weighted towards low-penalty machines
                let mut scored: Vec<(usize, f64)> = alt.iter()
                    .map(|&m| (m, -mach_penalty[m] - inst.pt(g, m) as f64 * 0.01))
                    .collect();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let pick = rng.gen_range(0..2.min(scored.len()));
                assign[g] = scored[pick].0;
            }
        }
    }
    let mut sched = Sched::new(inst);
    let mut job_next = vec![0usize; inst.n_jobs];
    let mut job_ready = vec![0u32; inst.n_jobs];
    let mut mach_avail = vec![0u32; inst.n_mach];
    let mut remaining = inst.total_ops;
    while remaining > 0 {
        let mut best_job = NONE;
        let mut best_end = u32::MAX;
        for job in 0..inst.n_jobs {
            let lo = job_next[job];
            if lo >= inst.n_ops[job] { continue; }
            let g = inst.op_base[job] + lo;
            let m = assign[g];
            let pt_m = inst.pt(g, m);
            if pt_m == 0 { continue; }
            let start = job_ready[job].max(mach_avail[m]);
            let end = start + pt_m;
            if end < best_end {
                best_end = end;
                best_job = job;
            }
        }
        if best_job == NONE { break; }
        let lo = job_next[best_job];
        let g = inst.op_base[best_job] + lo;
        let m = assign[g];
        let pt_m = inst.pt(g, m);
        let end = job_ready[best_job].max(mach_avail[m]) + pt_m;
        sched.assign[g] = m;
        sched.pt[g] = pt_m;
        sched.mseq[m].push(g);
        job_next[best_job] += 1;
        job_ready[best_job] = end;
        mach_avail[m] = end;
        remaining -= 1;
    }
    sched.build_mlinks(inst);
    let _ = sched.eval(inst);
    sched
}

/// Construct using route preferences from a previous solution
fn construct_from_route_pref(
    inst: &Inst,
    route_pref: &[Vec<[usize; 2]>],
    rng: &mut SmallRng,
    mutate_prob: f64,
) -> Sched {
    let mut assign = vec![0usize; inst.total_ops];
    for g in 0..inst.total_ops {
        let job = inst.op_job[g];
        let p = inst.prod[job];
        let lo = g - inst.op_base[job];
        if p < route_pref.len() && lo < route_pref[p].len() && rng.gen::<f64>() >= mutate_prob {
            let pref = route_pref[p][lo];
            if pref[0] != NONE && inst.pt(g, pref[0]) > 0 {
                assign[g] = pref[0];
            } else {
                assign[g] = inst.elig[g][rng.gen_range(0..inst.elig[g].len())];
            }
        } else {
            assign[g] = inst.elig[g][rng.gen_range(0..inst.elig[g].len())];
        }
    }
    construct_from_assign(inst, &assign, rng, 0.0)
}

// =========================================================
//  Entry point
// =========================================================

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let inst = Inst::new(challenge);
    let mut rng = SmallRng::from_seed(challenge.seed);

    // Scenario-adaptive parameters
    let high_flex = inst.flex_avg > 3.0;
    let very_high_flex = inst.flex_avg > 6.0;
    let low_flex = inst.flex_avg < 1.5;

    let params = TabuParams {
        tenure: if high_flex { 12 + (inst.total_ops / 120).min(18) }
                else { 10 + (inst.total_ops / 150).min(15) },
        stall_limit: if very_high_flex { 20 } else if high_flex { 25 } else if low_flex { 35 } else { 30 },
        max_reroute_alts: if very_high_flex { 8 } else if high_flex { 4 } else { 3 },
        reroute_pt_mult: if very_high_flex { 350 } else if high_flex { 200 } else { 150 },
        n_perturb: if low_flex { 6 } else if high_flex { 4 } else { 5 },
    };

    let mach_load_bias = if high_flex { 0.15 } else { 0.05 };

    // Phase 1: Adaptive rule ranking
    let mut rule_scores: Vec<(Rule, u32)> = Vec::new();
    for &rule in &ALL_RULES {
        let s = construct(&inst, rule, None, 0, mach_load_bias, None, None);
        rule_scores.push((rule, s.makespan));
    }
    rule_scores.sort_by_key(|&(_, ms)| ms);

    let mut global_best_ms = rule_scores[0].1;
    let best_rule_sched = construct(&inst, rule_scores[0].0, None, 0, mach_load_bias, None, None);
    save_solution(&best_rule_sched.to_solution(&inst))?;

    // Phase 2: Multi-start construction with early learning
    let n_restarts: usize = if low_flex { 90 } else if very_high_flex { 500 } else if high_flex { 400 } else { 500 };
    let top_k_keep: usize = if high_flex { 10 } else if low_flex { 3 } else { 8 };
    let mut top_scheds: Vec<Sched> = Vec::new();
    top_scheds.push(best_rule_sched);
    let mut best_assign: Vec<usize> = top_scheds[0].assign.clone();

    // Early learning: compute biases immediately from best construction
    let mut learned_penalty = compute_machine_penalty(&top_scheds[0], &inst);
    let mut learned_route = extract_route_pref(&top_scheds[0], &inst);
    let mut learned_job_bias = compute_job_bias(&top_scheds[0], &inst);

    // Machine popularity tracking for very_high_flex debiasing
    let mut mach_pop = vec![0u64; inst.n_mach];
    let mut mach_pop_total = 0u64;
    if very_high_flex {
        for g in 0..inst.total_ops {
            mach_pop[top_scheds[0].assign[g]] += 1;
        }
        mach_pop_total += inst.total_ops as u64;
    }

    for restart in 0..n_restarts {
        // Refresh learning data periodically
        if restart > 0 && restart % 80 == 0 && !top_scheds.is_empty() {
            learned_penalty = compute_machine_penalty(&top_scheds[0], &inst);
            learned_route = extract_route_pref(&top_scheds[0], &inst);
            learned_job_bias = compute_job_bias(&top_scheds[0], &inst);
        }

        let s = if low_flex {
            if restart >= 30 && rng.gen_range(0..3u32) == 0 {
                construct_from_assign(&inst, &best_assign, &mut rng, 0.15)
            } else {
                let rule = if restart % 20 < 9 { rule_scores[0].0 }
                           else if restart % 20 < 14 { rule_scores[1].0 }
                           else if restart % 20 < 17 { rule_scores[2].0 }
                           else { ALL_RULES[rng.gen_range(0..ALL_RULES.len())] };
                let top_k = rng.gen_range(2..=4);
                construct(&inst, rule, Some(&mut rng), top_k, mach_load_bias, None, None)
            }
        } else if very_high_flex {
            // Very high flex (chaotic): heavy loadbalanced + debiasing, minimal learning
            let r = rng.gen::<f64>();
            if r < 0.40 {
                // Loadbalanced construction: best for machine assignment diversity
                let lk = rng.gen_range(1..=3);
                construct_loadbalanced(&inst, &mut rng, lk)
            } else if r < 0.60 {
                // Mutated from best assignment
                construct_from_assign(&inst, &best_assign, &mut rng, 0.30)
            } else {
                // Rule-based with machine popularity debiasing
                let rule = if restart % 20 < 9 { rule_scores[0].0 }
                           else if restart % 20 < 14 { rule_scores[1].0 }
                           else if restart % 20 < 17 { rule_scores[2].0 }
                           else { ALL_RULES[rng.gen_range(0..ALL_RULES.len())] };
                let top_k = rng.gen_range(2..=6);
                let use_mpb = mach_pop_total > 0 && rng.gen::<f64>() < 0.5;
                let mpb: Vec<f64> = if use_mpb {
                    mach_pop.iter().map(|&c| c as f64 * inst.n_mach as f64 / mach_pop_total as f64).collect()
                } else { Vec::new() };
                construct(&inst, rule, Some(&mut rng), top_k, mach_load_bias,
                    None, if use_mpb { Some(&mpb) } else { None })
            }
        } else {
            let r = rng.gen::<f64>();
            if r < 0.20 {
                // Learning: assign-smart with penalties
                let mutate_p = if high_flex { 0.25 } else { 0.20 };
                construct_from_assign_smart(&inst, &best_assign, &mut rng, mutate_p, &learned_penalty)
            } else if r < 0.35 {
                // Learning: route preferences
                let mutate_p = if high_flex { 0.25 } else { 0.20 };
                construct_from_route_pref(&inst, &learned_route, &mut rng, mutate_p)
            } else if r < 0.50 {
                // Mutated from best assignment
                let mutate_p = if high_flex { 0.25 } else { 0.20 };
                construct_from_assign(&inst, &best_assign, &mut rng, mutate_p)
            } else if restart % 5 == 4 {
                let lk = rng.gen_range(1..=3);
                construct_loadbalanced(&inst, &mut rng, lk)
            } else {
                // Rule-based with optional job bias
                let rule = if restart % 20 < 9 { rule_scores[0].0 }
                           else if restart % 20 < 14 { rule_scores[1].0 }
                           else if restart % 20 < 17 { rule_scores[2].0 }
                           else { ALL_RULES[rng.gen_range(0..ALL_RULES.len())] };
                let top_k = if high_flex { rng.gen_range(2..=6) } else { rng.gen_range(2..=4) };
                let use_jb = rng.gen::<f64>() < 0.4;
                construct(&inst, rule, Some(&mut rng), top_k, mach_load_bias,
                    if use_jb { Some(&learned_job_bias) } else { None }, None)
            }
        };

        if s.makespan < global_best_ms {
            global_best_ms = s.makespan;
            best_assign = s.assign.clone();
            save_solution(&s.to_solution(&inst))?;
        }

        // Update machine popularity for very_high_flex
        if very_high_flex {
            for g in 0..inst.total_ops {
                mach_pop[s.assign[g]] += 1;
            }
            mach_pop_total += inst.total_ops as u64;
        }

        if top_scheds.len() < top_k_keep {
            top_scheds.push(s);
            top_scheds.sort_by_key(|sc| sc.makespan);
        } else if s.makespan < top_scheds.last().unwrap().makespan {
            top_scheds.pop();
            top_scheds.push(s);
            top_scheds.sort_by_key(|sc| sc.makespan);
        }
    }

    // Refinement phase: biased constructions from top schedules (skip for chaotic)
    if !low_flex && !very_high_flex {
        let refine_rules = ALL_RULES;
        let n_refine = top_scheds.len().min(5);
        for ti in 0..n_refine {
            let ref_bias = compute_job_bias(&top_scheds[ti], &inst);
            let ref_penalty = compute_machine_penalty(&top_scheds[ti], &inst);
            for &rule in &refine_rules {
                let top_k = if high_flex { 3 } else { 2 };
                let s = construct(&inst, rule, Some(&mut rng), top_k, mach_load_bias,
                    Some(&ref_bias), Some(&ref_penalty));
                if s.makespan < global_best_ms {
                    global_best_ms = s.makespan;
                    best_assign = s.assign.clone();
                    save_solution(&s.to_solution(&inst))?;
                }
                if top_scheds.len() < top_k_keep {
                    top_scheds.push(s);
                    top_scheds.sort_by_key(|sc| sc.makespan);
                } else if s.makespan < top_scheds.last().unwrap().makespan {
                    top_scheds.pop();
                    top_scheds.push(s);
                    top_scheds.sort_by_key(|sc| sc.makespan);
                }
            }
        }
    }

    // Flow shop: NEH construction + permutation-based local search
    let flow_mach_seq = detect_flow_shop(&inst);
    if let Some(ref mach_seq) = flow_mach_seq {
        let mut perm = neh_construction(&inst, mach_seq);
        let neh_ms = flow_shop_perm_search(&inst, mach_seq, &mut perm, &mut rng);
        let ns = perm_to_sched(&perm, &inst, mach_seq);
        if ns.makespan < global_best_ms {
            global_best_ms = ns.makespan;
            save_solution(&ns.to_solution(&inst))?;
        }
        if top_scheds.len() < top_k_keep {
            top_scheds.push(ns);
            top_scheds.sort_by_key(|sc| sc.makespan);
        } else if neh_ms < top_scheds.last().unwrap().makespan {
            top_scheds.pop();
            let ns2 = perm_to_sched(&perm, &inst, mach_seq);
            top_scheds.push(ns2);
            top_scheds.sort_by_key(|sc| sc.makespan);
        }

        // Try additional random starting permutations with perm search
        for _ in 0..4 {
            let mut rperm: Vec<usize> = (0..inst.n_jobs).collect();
            for i in (1..inst.n_jobs).rev() {
                let j = rng.gen_range(0..=i);
                rperm.swap(i, j);
            }
            let rms = flow_shop_perm_search(&inst, mach_seq, &mut rperm, &mut rng);
            if rms < global_best_ms {
                global_best_ms = rms;
                let rs = perm_to_sched(&rperm, &inst, mach_seq);
                save_solution(&rs.to_solution(&inst))?;
            }
            if rms < top_scheds.last().unwrap().makespan {
                top_scheds.pop();
                let rs = perm_to_sched(&rperm, &inst, mach_seq);
                top_scheds.push(rs);
                top_scheds.sort_by_key(|sc| sc.makespan);
            }
        }

        // Iterated Greedy: start from the best permutation found so far
        // Reconstruct best perm from best sched
        let best_sched_ref = &top_scheds[0];
        let ref_m = mach_seq[0];
        let mut ig_perm: Vec<usize> = best_sched_ref.mseq[ref_m].iter()
            .map(|&g| inst.op_job[g]).collect();
        // Fallback: if mseq doesn't give full perm, use NEH
        if ig_perm.len() != inst.n_jobs {
            ig_perm = neh_construction(&inst, mach_seq);
        }
        let ig_ms = iterated_greedy_flow_shop(&inst, mach_seq, &mut ig_perm, &mut rng, 600);
        if ig_ms < global_best_ms {
            global_best_ms = ig_ms;
            let ig_sched = perm_to_sched(&ig_perm, &inst, mach_seq);
            save_solution(&ig_sched.to_solution(&inst))?;
        }

        // Second IG run from a different starting point
        let mut ig_perm2 = neh_construction(&inst, mach_seq);
        let _ = flow_shop_perm_search(&inst, mach_seq, &mut ig_perm2, &mut rng);
        let ig_ms2 = iterated_greedy_flow_shop(&inst, mach_seq, &mut ig_perm2, &mut rng, 450);
        if ig_ms2 < global_best_ms {
            global_best_ms = ig_ms2;
            let ig_sched2 = perm_to_sched(&ig_perm2, &inst, mach_seq);
            save_solution(&ig_sched2.to_solution(&inst))?;
        }

    } else {
        // Non-flow shop: inject NEH if applicable (no change)
    }

    // Phase 3: Tabu search from top constructions
    let is_flow_shop = flow_mach_seq.is_some();
    if is_flow_shop {
        // Flow shop: IG above does most work, deeper tabu for refinement
        let mut primary = Sched::new(&inst);
        primary.clone_from_sched(&top_scheds[0], &inst);
        let _ = primary.eval(&inst);
        tabu_search(&inst, &mut primary, &mut rng, save_solution, &mut global_best_ms, 2000, &params)?;
        if top_scheds.len() >= 2 {
            let mut secondary = Sched::new(&inst);
            secondary.clone_from_sched(&top_scheds[1], &inst);
            let _ = secondary.eval(&inst);
            tabu_search(&inst, &mut secondary, &mut rng, save_solution, &mut global_best_ms, 1000, &params)?;
        }
    } else if low_flex {
        // Low flex: deep search (few constructions, long tabu)
        let mut primary = Sched::new(&inst);
        primary.clone_from_sched(&top_scheds[0], &inst);
        let _ = primary.eval(&inst);
        tabu_search(&inst, &mut primary, &mut rng, save_solution, &mut global_best_ms, 3500, &params)?;

        if top_scheds.len() >= 2 {
            let mut secondary = Sched::new(&inst);
            secondary.clone_from_sched(&top_scheds[1], &inst);
            let _ = secondary.eval(&inst);
            tabu_search(&inst, &mut secondary, &mut rng, save_solution, &mut global_best_ms, 1500, &params)?;
        }
    } else {
        // Medium/high flex: fewer starts, deeper search per start
        let n_searches = top_scheds.len().min(if very_high_flex { 3 } else if high_flex { 4 } else { 3 });
        let iters_first = if very_high_flex { 1200 } else if high_flex { 1100 } else { 1300 };
        let iters_rest = if very_high_flex { 600 } else if high_flex { 550 } else { 650 };
        for i in 0..n_searches {
            let iters = if i == 0 { iters_first } else { iters_rest };
            let mut s = Sched::new(&inst);
            s.clone_from_sched(&top_scheds[i], &inst);
            let _ = s.eval(&inst);
            tabu_search(&inst, &mut s, &mut rng, save_solution, &mut global_best_ms, iters, &params)?;
        }
    }

    Ok(())
}

pub fn help() {
    println!("titan_sched: Tabu Search + Iterated Greedy + Learning-guided for FJSP");
}
