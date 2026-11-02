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

const DEFAULT_TARGET: [f64; 5] = [0.0440, 0.1030, 0.1020, 0.1585, 0.0803];
const GIVE_UP: [f64; 5] = [0.006, 0.006, 0.006, 0.006, 0.006];
const STOP_ABOVE: [f64; 5] = [0.010, 0.004, 0.006, 0.010, 0.008];

pub const ENV_F0: [u64; 5] = [0, 0, 0, 30_000_000_000, 0];
pub const ENV_BIN: [u64; 5] = [1 << 62, 4_000_000_000, 5_000_000_000, 5_000_000_000, 2_000_000_000];
const ENV_MARGIN: [f64; 5] = [0.0, 0.004, 0.004, 0.012, 0.004];
pub const EXT_F: [f64; 5] = [0.0, 1.0, 1.0, 0.0, 0.0];
pub const EXT_W: [f64; 5] = [0.0, 0.008, 0.008, 0.0, 0.0];
pub const EXT_R: [u32; 5] = [1, 8, 1, 1, 1];
pub const EXT_SENT_ONLY: [bool; 5] = [false, true, false, false, false];
pub const ENV_LIN: [&[[f64; 4]]; 5] = [
    &[],
    &[
        [1.00000, 0.00000, 0.00000, 0.00000],
        [0.00753, 0.24832, -0.24507, 0.03015],
        [0.00340, 0.16713, -0.13000, 0.02479],
        [0.00167, 0.11952, -0.06315, 0.02304],
        [0.00237, 0.10314, -0.05607, 0.02002],
        [0.00151, 0.09673, -0.04095, 0.02049],
        [0.00108, 0.09227, -0.03203, 0.01688],
        [-0.00053, 0.10026, -0.03342, 0.01770],
        [-0.00033, 0.10592, -0.05275, 0.01457],
        [-0.00098, 0.09206, -0.03689, 0.01308],
        [-0.00073, 0.07583, -0.02673, 0.01298],
        [-0.00130, 0.06205, -0.00949, 0.01053],
        [-0.00076, 0.05022, -0.00945, 0.00958],
        [-0.00101, 0.03483, 0.00429, 0.01044],
        [-0.00103, 0.03038, 0.00328, 0.00813],
        [-0.00070, 0.02715, -0.00220, 0.00784],
        [-0.00057, 0.01964, 0.00294, 0.00835],
        [-0.00054, 0.01893, -0.00001, 0.00691],
        [-0.00022, 0.01725, -0.00350, 0.00566],
        [0.00007, 0.01549, -0.00653, 0.00538],
        [0.00019, 0.01204, -0.00660, 0.00575],
        [0.00003, 0.00757, -0.00148, 0.00607],
        [-0.00023, 0.00495, 0.00272, 0.00553],
        [-0.00032, 0.00439, 0.00230, 0.00303],
        [-0.00020, 0.00594, -0.00159, 0.00306],
        [-0.00006, 0.00330, -0.00064, 0.00310],
        [-0.00005, 0.00291, -0.00079, 0.00317],
        [-0.00009, 0.00171, -0.00004, 0.00208],
        [0.00033, 0.00244, -0.00479, 0.00202],
        [0.00006, 0.00111, -0.00178, 0.00016],
        [1.00000, 0.00000, 0.00000, 0.00000],
    ],
    &[],
    &[],
    &[],
];
pub const ENV_GAIN: [&[f64]; 5] = [
    &[],
    &[
        1.0, 0.0678, 0.0395, 0.0342, 0.0334, 0.0334, 0.0334, 0.0334, 0.0334, 0.0288, 0.0275, 0.0165, 0.0165,
        0.0139, 0.0122, 0.0118, 0.0084, 0.0084, 0.0079, 0.0079, 0.0079, 0.0052, 0.0042, 0.0042, 0.0042, 0.0028,
        0.0028, 0.0028, 0.0028, 0.0, 0.0, 0.0, 0.0,
    ],
    &[
        1.0, 0.1094, 0.1004, 0.0558, 0.03, 0.0185, 0.013, 0.013, 0.013, 0.012, 0.012, 0.012, 0.012,
        0.0108, 0.0099, 0.0072, 0.0065, 0.0065, 0.0057, 0.0057, 0.0057, 0.0057, 0.0046, 0.0046, 0.0046, 0.0046,
        0.0037, 0.0037, 0.0028, 0.0028, 0.0013, 0.0013, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011,
        0.0,
    ],
    &[
        1.0, 0.1802, 0.1802, 0.1802, 0.1802, 0.1802, 0.1802, 0.1802, 0.1802, 0.1802, 0.1802, 0.1802, 0.1802,
        0.1802, 0.1802, 0.1802, 0.1802, 0.1802, 0.1802, 0.1802, 0.1523, 0.0215, 0.0215, 0.0215, 0.0215, 0.0215,
        0.0215, 0.0215, 0.0215, 0.0215, 0.0215, 0.0215, 0.0215, 0.0192, 0.0178, 0.0178, 0.0178, 0.0178, 0.0178,
        0.0178, 0.0178, 0.0178, 0.0178, 0.0178, 0.0178, 0.0134, 0.0134, 0.0134, 0.0134, 0.0134, 0.0134, 0.0134,
        0.0134, 0.0134, 0.0134, 0.0134, 0.0134, 0.0082, 0.0073, 0.0073, 0.0073, 0.0073, 0.0073, 0.0073, 0.0049,
        0.0049, 0.0049, 0.0049, 0.0049, 0.0049, 0.0049, 0.0049, 0.0049, 0.0049, 0.0049, 0.0047, 0.0018, 0.0011,
        0.0006, 0.0002,
    ],
    &[
        1.0, 0.056, 0.0514, 0.0514, 0.0485, 0.0465, 0.0465, 0.0465, 0.0465, 0.0465, 0.0465, 0.0465, 0.0465,
        0.0465, 0.0465, 0.0465, 0.0465, 0.0465, 0.0465, 0.0465, 0.0465, 0.0465, 0.0427, 0.0062, 0.0062, 0.0062,
        0.0062, 0.0062, 0.0062, 0.0062, 0.0062, 0.0062, 0.0062, 0.0062, 0.0051, 0.0051, 0.0041, 0.0041, 0.0041,
        0.0041, 0.0038, 0.0038, 0.0038, 0.0038, 0.0038, 0.0033, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024,
        0.0024, 0.0024, 0.0024, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001,
        0.001, 0.001, 0.0006, 0.0006, 0.0006, 0.0006, 0.0006, 0.0006, 0.0006,
    ],
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
    pub gap: f64,
    pub greedy: u32,
    pub env_margin: f64,
    pub target: f64,
    pub give_up: f64,
    pub above: f64,
    pub sent: bool,
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
            gap: 0.0,
            greedy: 0,
            env_margin: ENV_MARGIN[idx],
            target: get("aim_target").map(|t| if t > 1.0 { t / 1e6 } else { t }).unwrap_or(DEFAULT_TARGET[idx]),
            give_up: GIVE_UP[idx],
            above: STOP_ABOVE[idx],
            sent: get("aim_target").is_some(),
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
            let cmp = |a: &(f64, u32, u32, usize, usize), b: &(f64, u32, u32, usize, usize)| {
                b.0.partial_cmp(&a.0)
                    .unwrap()
                    .then(a.1.cmp(&b.1))
                    .then(a.2.cmp(&b.2))
                    .then(a.3.cmp(&b.3))
                    .then(a.4.cmp(&b.4))
            };
            let k = top_k.min(self.cands.len());
            if k < self.cands.len() {
                self.cands.select_nth_unstable_by(k - 1, cmp);
            }
            self.cands[..k].sort_by(cmp);
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
    GiveUp,
    StopAt(u32),
}

pub fn plan(pre: &Pre, seed: &[u8; 32], aim: &mut Aim) -> Plan {
    let lb = lower_bound(pre) as f64;
    let line = aim.target - aim.give_up;
    let mut g = u32::MAX;
    let (s, complete) = sota(pre, seed, |b| (b as f64 - lb) / (b as f64) < line, |b| g = b);
    aim.greedy = g;
    if !complete {
        return Plan::GiveUp;
    }
    aim.sota = s;
    aim.gap = (s as f64 - lb) / s as f64;
    let stop = ((s as f64) * (1.0 - aim.target - aim.above)).floor();
    Plan::StopAt(stop.max(0.0) as u32)
}
