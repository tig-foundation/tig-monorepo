use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde_json::{Map, Value};
use super::gate_floor::gate_dispatch;
use super::Pre;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Flow,
    Job,
    Hybrid,
    FjspMedium,
    FjspHigh,
}

const DEFAULT_TARGET: [f64; 5] = [0.0500, 0.1025, 0.1000, 0.1565, 0.0796];
const ACTIVE: [bool; 5] = [true, true, true, true, false];
const GIVE_UP: [f64; 5] = [0.006, 0.006, 0.006, 0.006, 0.006];
const STOP_ABOVE: [f64; 5] = [0.015, 0.008, 0.010, 0.010, 0.010];

pub const ENV_F0: [u64; 5] = [0, 0, 30_000_000_000, 100_000_000_000, 0];
pub const ENV_BIN: [u64; 5] = [1 << 62, 4_000_000_000, 2_000_000_000, 5_000_000_000, 1 << 62];
const ENV_MARGIN: [f64; 5] = [0.0, 0.008, 0.008, 0.02, 0.0];
pub const ENV_GAIN: [&[f64]; 5] = [
    &[],
    &[
        1.0, 0.0542, 0.0473, 0.0473, 0.0473, 0.0473, 0.0473, 0.0473, 0.0349, 0.0349, 0.0349, 0.0349, 0.0304,
        0.0304, 0.0304, 0.0279, 0.0279, 0.0265, 0.0180, 0.0179, 0.0179, 0.0179, 0.0179, 0.0179, 0.0126, 0.0107,
        0.0107, 0.0107, 0.0107, 0.0107, 0.0088, 0.0088, 0.0036, 0.0031, 0.0, 0.0,
    ],
    &[
        1.0, 0.1276, 0.112, 0.1071, 0.1071, 0.101, 0.101, 0.101, 0.101, 0.101, 0.095, 0.0947, 0.0871,
        0.074, 0.0451, 0.0304, 0.0224, 0.0169, 0.0145, 0.014, 0.0129, 0.0115, 0.0104, 0.0088, 0.0073, 0.0069,
        0.0069, 0.0069, 0.0059, 0.0059, 0.0059, 0.0059, 0.0059, 0.0043, 0.0043, 0.0043, 0.003, 0.003, 0.0026,
        0.0022, 0.0006, 0.0006, 0.0006, 0.0, 0.0, 0.0,
    ],
    &[
        1.0, 0.5522, 0.5522, 0.5522, 0.5522, 0.5522, 0.5522, 0.5522, 0.5522, 0.5522, 0.5522, 0.4445, 0.327,
        0.1023, 0.0992, 0.0992, 0.0992, 0.0992, 0.0992, 0.0992, 0.0992, 0.0992, 0.0992, 0.0946, 0.0932, 0.0932,
        0.0932, 0.0858, 0.0858, 0.0858, 0.0858, 0.0858, 0.0858, 0.0858, 0.0858, 0.0485, 0.0485, 0.0485, 0.0485,
        0.0078, 0.0006,
    ],
    &[],
];

pub fn kind(pre: &Pre) -> Kind {
    let n_ops = pre.n_ops.max(1) as f64;
    let flex: usize = pre.job_ops.iter().flat_map(|j| j.iter()).map(|o| o.len()).sum();
    let avg_flex = flex as f64 / n_ops;
    let route = |j: usize| pre.job_ops[j].iter().map(|o| o.iter().map(|&(m, _)| m));
    let one_route = (1..pre.num_jobs).all(|j| {
        pre.job_ops[j].len() == pre.job_ops[0].len() && route(j).zip(route(0)).all(|(a, b)| a.eq(b))
    });
    if pre.max_flex == 1 {
        if one_route { Kind::Flow } else { Kind::Job }
    } else if one_route {
        Kind::Hybrid
    } else if avg_flex >= 6.0 {
        Kind::FjspHigh
    } else {
        Kind::FjspMedium
    }
}

pub struct Aim {
    pub idx: usize,
    pub sota: u32,
    pub greedy: u32,
    pub env_margin: f64,
    pub on: bool,
    pub target: f64,
    pub give_up: f64,
    pub above: f64,
}

impl Aim {
    pub fn read(k: Kind, hp: &Option<Map<String, Value>>) -> Aim {
        let get = |key: &str| hp.as_ref().and_then(|m| m.get(key)).and_then(|v| v.as_f64());
        let idx = match k {
            Kind::Flow => 0,
            Kind::Job => 1,
            Kind::Hybrid => 2,
            Kind::FjspMedium => 3,
            Kind::FjspHigh => 4,
        };
        Aim {
            idx,
            sota: 0,
            greedy: 0,
            env_margin: ENV_MARGIN[idx],
            on: ACTIVE[idx],
            target: get("aim_target").map(|t| if t > 1.0 { t / 1e6 } else { t }).unwrap_or(DEFAULT_TARGET[idx]),
            give_up: GIVE_UP[idx],
            above: STOP_ABOVE[idx],
        }
    }
}

fn makespan(pre: &Pre, s: &tig_challenges::job_scheduling::Solution) -> u32 {
    let mut mk = 0u32;
    for (j, ops) in s.job_schedule.iter().enumerate() {
        if let Some(&(m, st)) = ops.last() {
            let oi = ops.len() - 1;
            let p = pre.job_ops[j][oi].iter().find(|&&(mm, _)| mm == m).map_or(0, |&(_, p)| p);
            mk = mk.max(st + p);
        }
    }
    mk
}

fn jackson(ops: &mut Vec<(u64, u64, u64)>) -> u64 {
    ops.sort_unstable_by_key(|o| o.0);
    let mut heap = std::collections::BinaryHeap::new();
    let (mut t, mut i, mut best) = (0u64, 0usize, 0u64);
    while i < ops.len() || !heap.is_empty() {
        if heap.is_empty() && t < ops[i].0 {
            t = ops[i].0;
        }
        while i < ops.len() && ops[i].0 <= t {
            heap.push((ops[i].2, ops[i].1));
            i += 1;
        }
        let (q, rem) = heap.pop().unwrap();
        let next = if i < ops.len() { ops[i].0 } else { u64::MAX };
        let run = rem.min(next - t);
        t += run;
        if run < rem {
            heap.push((q, rem - run));
        } else {
            best = best.max(t + q);
        }
    }
    best
}

pub fn lower_bound(pre: &Pre) -> u32 {
    let nm = pre.num_machines;
    let minp = |o: &Vec<(usize, u32)>| o.iter().map(|&(_, p)| p as u64).min().unwrap_or(0);
    let job_len: Vec<u64> = pre.job_ops.iter().map(|j| j.iter().map(minp).sum()).collect();
    let mut lb = job_len.iter().copied().max().unwrap_or(0);
    let total: u64 = job_len.iter().sum();
    lb = lb.max((total + nm as u64 - 1) / nm as u64);
    if pre.max_flex == 1 {
        let mut per: Vec<Vec<(u64, u64, u64)>> = vec![Vec::new(); nm];
        for (j, ops) in pre.job_ops.iter().enumerate() {
            let mut head = 0u64;
            for op in ops.iter() {
                let (m, p) = (op[0].0, op[0].1 as u64);
                per[m].push((head, p, job_len[j] - head - p));
                head += p;
            }
        }
        for ops in per.iter_mut() {
            if !ops.is_empty() {
                lb = lb.max(jackson(ops));
            }
        }
    }
    lb.min(u32::MAX as u64) as u32
}

struct Disp<'a> {
    pre: &'a Pre,
    next: Vec<usize>,
    ready: Vec<u32>,
    avail: Vec<u32>,
    rem: Vec<f64>,
    total: Vec<f64>,
    bucket: Vec<Vec<usize>>,
    machines: Vec<usize>,
    cands: Vec<(f64, u32, u32, usize, usize)>,
}

impl<'a> Disp<'a> {
    fn new(pre: &'a Pre) -> Self {
        let nj = pre.num_jobs;
        let nm = pre.num_machines;
        let total = (0..nj)
            .map(|j| (0..pre.job_nops[j]).map(|oi| pre.op_w[pre.job_off[j] + oi]).sum())
            .collect();
        Disp {
            pre,
            next: vec![0; nj],
            ready: vec![0; nj],
            avail: vec![0; nm],
            rem: vec![0.0; nj],
            total,
            bucket: vec![Vec::new(); nm],
            machines: Vec::with_capacity(nm),
            cands: Vec::new(),
        }
    }

    fn pick(&mut self, top_k: usize, rng: &mut Option<&mut SmallRng>) -> usize {
        if top_k > 1 && rng.is_some() {
            self.cands.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap()
                    .then(a.1.cmp(&b.1))
                    .then(a.2.cmp(&b.2))
                    .then(a.3.cmp(&b.3))
                    .then(a.4.cmp(&b.4))
            });
            let k = top_k.min(self.cands.len());
            return self.cands[rng.as_mut().unwrap().gen_range(0..k)].4;
        }
        let eps = 1e-9f64;
        let mut best = self.cands[0];
        for &c in self.cands.iter().skip(1) {
            let better = if c.0 > best.0 + eps {
                true
            } else if (c.0 - best.0).abs() <= eps {
                (c.1, c.2, c.3, c.4) < (best.1, best.2, best.3, best.4)
            } else {
                false
            };
            if better {
                best = c;
            }
        }
        best.4
    }

    fn run_flat(&mut self, rule: u8, top_k: usize, mut rng: Option<&mut SmallRng>) -> u32 {
        use rand::seq::SliceRandom;
        use std::cmp::Reverse;
        let pre = self.pre;
        let nj = pre.num_jobs;
        let nm = pre.num_machines;
        self.next.iter_mut().for_each(|x| *x = 0);
        self.ready.iter_mut().for_each(|x| *x = 0);
        self.avail.iter_mut().for_each(|x| *x = 0);
        self.rem.copy_from_slice(&self.total);
        for b in self.bucket.iter_mut() {
            b.clear();
        }
        for j in 0..nj {
            if pre.job_nops[j] > 0 {
                self.bucket[pre.job_ops[j][0][0].0].push(j);
            }
        }
        let mut pending: std::collections::BinaryHeap<Reverse<(u32, usize)>> = std::collections::BinaryHeap::new();
        let use_random = top_k > 1 && rng.is_some();
        let mut remaining: usize = pre.n_ops;
        let mut time = 0u32;
        while remaining > 0 {
            while let Some(&Reverse((t, j))) = pending.peek() {
                if t > time {
                    break;
                }
                pending.pop();
                let m = pre.job_ops[j][self.next[j]][0].0;
                let b = &mut self.bucket[m];
                let pos = b.partition_point(|&x| x < j);
                b.insert(pos, j);
            }
            self.machines.clear();
            for m in 0..nm {
                if self.avail[m] <= time {
                    self.machines.push(m);
                }
            }
            if use_random {
                self.machines.shuffle(rng.as_mut().unwrap());
            }
            for mi in 0..self.machines.len() {
                let machine = self.machines[mi];
                if self.bucket[machine].is_empty() {
                    continue;
                }
                self.cands.clear();
                for &job in self.bucket[machine].iter() {
                    let oi = self.next[job];
                    let pt = pre.job_ops[job][oi][0].1;
                    let pr = match rule {
                        0 => self.rem[job],
                        1 => (pre.job_nops[job] - oi) as f64,
                        2 => -1.0,
                        3 => -(pt as f64),
                        _ => pt as f64,
                    };
                    self.cands.push((pr, time + pt, pt, 1, job));
                }
                let job = self.pick(top_k, &mut rng);
                let b = &mut self.bucket[machine];
                let pos = b.partition_point(|&x| x < job);
                b.remove(pos);
                let oi = self.next[job];
                let end = time + pre.job_ops[job][oi][0].1;
                self.next[job] += 1;
                self.ready[job] = end;
                self.avail[machine] = end;
                self.rem[job] -= pre.op_w[pre.job_off[job] + oi];
                if self.rem[job] < 0.0 {
                    self.rem[job] = 0.0;
                }
                if self.next[job] < pre.job_nops[job] {
                    pending.push(Reverse((end, job)));
                }
                remaining -= 1;
            }
            if remaining == 0 {
                break;
            }
            let mut nt = pending.peek().map_or(u32::MAX, |r| r.0 .0);
            for &t in self.avail.iter() {
                if t > time && t < nt {
                    nt = t;
                }
            }
            if nt == u32::MAX {
                return u32::MAX;
            }
            time = nt;
        }
        self.ready.iter().copied().max().unwrap_or(0)
    }

    fn run(&mut self, rule: u8, top_k: usize, mut rng: Option<&mut SmallRng>) -> u32 {
        use rand::seq::SliceRandom;
        let pre = self.pre;
        if pre.max_flex == 1 {
            return self.run_flat(rule, top_k, rng);
        }
        let nj = pre.num_jobs;
        let nm = pre.num_machines;
        self.next.iter_mut().for_each(|x| *x = 0);
        self.ready.iter_mut().for_each(|x| *x = 0);
        self.avail.iter_mut().for_each(|x| *x = 0);
        self.rem.copy_from_slice(&self.total);
        let use_random = top_k > 1 && rng.is_some();
        let mut remaining: usize = pre.n_ops;
        let mut time = 0u32;
        while remaining > 0 {
            self.machines.clear();
            for m in 0..nm {
                if self.avail[m] <= time {
                    self.machines.push(m);
                }
            }
            if use_random {
                self.machines.shuffle(rng.as_mut().unwrap());
            }
            for b in self.bucket.iter_mut() {
                b.clear();
            }
            for j in 0..nj {
                let oi = self.next[j];
                if oi < pre.job_nops[j] && self.ready[j] <= time {
                    for &(m, _) in pre.job_ops[j][oi].iter() {
                        self.bucket[m].push(j);
                    }
                }
            }
            for mi in 0..self.machines.len() {
                let machine = self.machines[mi];
                self.cands.clear();
                for &job in self.bucket[machine].iter() {
                    if self.ready[job] > time {
                        continue;
                    }
                    let oi = self.next[job];
                    let op = &pre.job_ops[job][oi];
                    let mut pt = 0u32;
                    let mut earliest = u32::MAX;
                    for &(m2, p2) in op.iter() {
                        if m2 == machine {
                            pt = p2;
                        }
                        let e = time.max(self.avail[m2]) + p2;
                        if e < earliest {
                            earliest = e;
                        }
                    }
                    let machine_end = time.max(self.avail[machine]) + pt;
                    if machine_end != earliest {
                        continue;
                    }
                    let flex = op.len();
                    let pr = match rule {
                        0 => self.rem[job],
                        1 => (pre.job_nops[job] - oi) as f64,
                        2 => -(flex as f64),
                        3 => -(pt as f64),
                        _ => pt as f64,
                    };
                    self.cands.push((pr, machine_end, pt, flex, job));
                }
                if self.cands.is_empty() {
                    continue;
                }
                let job = self.pick(top_k, &mut rng);
                let oi = self.next[job];
                let pt = pre.job_ops[job][oi].iter().find(|&&(m, _)| m == machine).map_or(0, |&(_, p)| p);
                let end = time.max(self.avail[machine]) + pt;
                self.next[job] += 1;
                self.ready[job] = end;
                self.avail[machine] = end;
                self.rem[job] -= pre.op_w[pre.job_off[job] + oi];
                if self.rem[job] < 0.0 {
                    self.rem[job] = 0.0;
                }
                remaining -= 1;
            }
            if remaining == 0 {
                break;
            }
            let mut nt = u32::MAX;
            for &t in self.avail.iter() {
                if t > time && t < nt {
                    nt = t;
                }
            }
            for j in 0..nj {
                if self.next[j] < pre.job_nops[j] && self.ready[j] > time && self.ready[j] < nt {
                    nt = self.ready[j];
                }
            }
            if nt == u32::MAX {
                return u32::MAX;
            }
            time = nt;
        }
        self.ready.iter().copied().max().unwrap_or(0)
    }
}

pub fn sota(pre: &Pre, seed: &[u8; 32], mut enough: impl FnMut(u32) -> bool, mut greedy: impl FnMut(u32)) -> (u32, bool) {
    let mut disp = Disp::new(pre);
    let mut run = |rule: u8, k: usize, rng: Option<&mut SmallRng>| -> u32 {
        if pre.dur_pos {
            disp.run(rule, k, rng)
        } else {
            gate_dispatch(pre, rule, k, rng).map_or(u32::MAX, |s| makespan(pre, &s))
        }
    };
    let mut best = u32::MAX;
    for rule in 0..5u8 {
        best = best.min(run(rule, 0, None));
    }
    if enough(best) {
        return (best, false);
    }
    let mut top: Vec<(u32, u8, usize, u64)> = Vec::new();
    let mut rng = SmallRng::from_seed(*seed);
    for r in 0..200 {
        let s = rng.gen::<u64>();
        let rule = rng.gen_range(0..5usize) as u8;
        let k = rng.gen_range(2..=5usize);
        let mut local = SmallRng::seed_from_u64(s);
        let mk = run(rule, k, Some(&mut local));
        best = best.min(mk);
        top.push((mk, rule, k, s));
        top.sort_by(|a, b| a.0.cmp(&b.0));
        if top.len() > 2 {
            top.pop();
        }
        if r == 9 {
            greedy(best);
        }
        if enough(best) {
            return (best, false);
        }
    }
    for &(_, rule, k, s) in top.iter() {
        for attempt in 0..4usize {
            let mut local = SmallRng::seed_from_u64(s.wrapping_add(attempt as u64 + 1));
            let lk = match attempt % 3 {
                0 => k,
                1 => k.saturating_sub(1),
                _ => k + 1,
            }
            .max(2);
            best = best.min(run(rule, lk, Some(&mut local)));
            if enough(best) {
                return (best, false);
            }
        }
    }
    (best, true)
}

pub enum Plan {
    Full,
    GiveUp,
    StopAt(u32),
}

pub fn plan(pre: &Pre, seed: &[u8; 32], aim: &mut Aim) -> Plan {
    if !aim.on {
        return Plan::Full;
    }
    let lb = lower_bound(pre) as f64;
    let line = aim.target - aim.give_up;
    let mut g = u32::MAX;
    let (s, complete) = sota(pre, seed, |b| (b as f64 - lb) / (b as f64) < line, |b| g = b);
    aim.greedy = g;
    if !complete {
        return Plan::GiveUp;
    }
    aim.sota = s;
    let stop = ((s as f64) * (1.0 - aim.target - aim.above)).floor();
    Plan::StopAt(stop.max(0.0) as u32)
}
