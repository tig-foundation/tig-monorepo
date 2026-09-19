//! Alternative start of the hybrid engine: rule-guided constructions with learned biases, refined
//! by critical-block descent and machine reassignment.
pub use search::construct;

pub mod types {
//! Instance summary and candidate records shared by the construction and the local searches.

/// Processing time standing for "no eligible machine".
pub const INF: u32 = u32::MAX / 4;
/// Absent index.
pub const NONE: usize = usize::MAX;

/// One operation of a product: eligible machines sorted by id, with summary statistics.
pub struct OpInfo {
    pub machines: Vec<(usize, u32)>,
    pub min_pt: u32,
    pub avg_pt: f64,
    pub flex: usize,
    /// Mean of the processing times weighted by machine load.
    pub bn_avg: f64,
}

impl OpInfo {
    #[inline]
    pub fn pt_on(&self, machine: usize) -> Option<u32> {
        self.machines
            .iter()
            .find(|&&(m, _)| m == machine)
            .map(|&(_, pt)| pt)
    }
}

/// Two most used machines of one product operation in a reference schedule, weighted in 0..=255.
#[derive(Clone, Copy, Default)]
pub struct OpRoute {
    pub best_m: u8,
    pub best_w: u8,
    pub second_m: u8,
    pub second_w: u8,
}

pub type RoutePref = Vec<Vec<OpRoute>>;

/// Instance summary: per-product operation data, suffix sums and normalisation scales.
pub struct Pre {
    pub job_products: Vec<usize>,
    pub job_ops_len: Vec<usize>,
    pub product_ops: Vec<Vec<OpInfo>>,
    pub product_suf_min: Vec<Vec<u32>>,
    pub product_suf_avg: Vec<Vec<f64>>,
    pub product_suf_bn: Vec<Vec<f64>>,
    pub product_next_min: Vec<Vec<u32>>,
    pub product_next_flex_inv: Vec<Vec<f64>>,
    pub machine_load0: Vec<f64>,
    pub machine_weight: Vec<f64>,
    pub machine_best_pop: Vec<f64>,
    pub avg_machine_load: f64,
    pub avg_op_min: f64,
    pub horizon: f64,
    pub time_scale: f64,
    pub max_job_avg_work: f64,
    pub max_job_bn: f64,
    pub flex_factor: f64,
    pub high_flex: f64,
    pub jobshopness: f64,
    pub bn_focus: f64,
    /// Weight of the flow-shop job order preference, 0 unless the instance is nearly a flow shop.
    pub flow_w: f64,
    pub job_flow_pref: Vec<f64>,
    pub total_ops: usize,
}

/// Scored (job, machine) candidate.
#[derive(Clone, Copy)]
pub struct Cand {
    pub job: usize,
    pub machine: usize,
    pub pt: u32,
    pub score: f64,
}

/// Candidate before the machine-conflict adjustment.
#[derive(Clone, Copy)]
pub struct RawCand {
    pub job: usize,
    pub machine: usize,
    pub pt: u32,
    pub base_score: f64,
    pub rigidity: f64,
    pub reg_n: f64,
}

/// x / (1 + x) for x > 0, else 0.
#[inline]
pub fn sat(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        x / (1.0 + x)
    }
}
}

pub mod preprocess {
//! Builds the instance summary.
use super::types::*;
use anyhow::{anyhow, Result};
use tig_challenges::job_scheduling::*;

/// Makespan of a permutation flow shop with one stage per operation rank.
#[inline]
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

pub fn build_pre(challenge: &Challenge) -> Result<Pre> {
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
    let mut product_ops: Vec<Vec<OpInfo>> = Vec::with_capacity(num_products);
    let mut best_machine_by_product: Vec<Vec<usize>> = Vec::with_capacity(num_products);
    let mut machine_load0 = vec![0.0f64; num_machines];
    let mut machine_best_cnt = vec![0.0f64; num_machines];
    let mut total_ops: usize = 0;
    let mut total_min_work: f64 = 0.0;
    let mut total_flex_weighted: f64 = 0.0;
    let mut max_ops: usize = 1;
    let mut max_job_avg_work: f64 = 1.0;

    for (p, ops) in challenge.product_processing_times.iter().enumerate() {
        max_ops = max_ops.max(ops.len());
        let mut ops_info: Vec<OpInfo> = Vec::with_capacity(ops.len());
        let mut bests: Vec<usize> = Vec::with_capacity(ops.len());
        let mut sum_min_u64: u64 = 0;
        let mut sum_avg_f: f64 = 0.0;

        for op in ops {
            if op.is_empty() {
                ops_info.push(OpInfo {
                    machines: vec![],
                    min_pt: INF,
                    avg_pt: 0.0,
                    flex: 0,
                    bn_avg: 0.0,
                });
                bests.push(0);
                continue;
            }
            let mut machines: Vec<(usize, u32)> = Vec::with_capacity(op.len());
            let mut min_pt = INF;
            let mut sum = 0u64;
            let mut best_m = 0usize;
            let mut best_pt = INF;
            for (&m, &pt) in op.iter() {
                if m >= num_machines {
                    return Err(anyhow!("machine id out of range"));
                }
                machines.push((m, pt));
                min_pt = min_pt.min(pt);
                sum += pt as u64;
                if pt < best_pt || (pt == best_pt && m < best_m) {
                    best_pt = pt;
                    best_m = m;
                }
            }
            let flex = machines.len().max(1);
            let avg_pt = (sum as f64) / (flex as f64);
            sum_min_u64 += min_pt.min(INF / 2) as u64;
            sum_avg_f += avg_pt;
            machines.sort_unstable_by_key(|x| x.0);
            ops_info.push(OpInfo {
                machines,
                min_pt,
                avg_pt,
                flex,
                bn_avg: 0.0,
            });
            bests.push(best_m);
        }

        max_job_avg_work = max_job_avg_work.max(sum_avg_f);
        let cnt_u = challenge.jobs_per_product[p] as usize;
        let cnt_f = challenge.jobs_per_product[p] as f64;
        total_ops += ops_info.len() * cnt_u;
        total_min_work += (sum_min_u64 as f64) * cnt_f;

        for (oi, &bm) in ops_info.iter().zip(bests.iter()) {
            total_flex_weighted += (oi.flex as f64) * cnt_f;
            if bm < num_machines {
                machine_best_cnt[bm] += cnt_f;
            }
            if oi.min_pt < INF && oi.flex > 0 && !oi.machines.is_empty() {
                let flex_f = (oi.flex as f64).max(1.0);
                let delta = (oi.min_pt as f64) * cnt_f / flex_f;
                for &(m, _) in &oi.machines {
                    machine_load0[m] += delta;
                }
            }
        }
        product_ops.push(ops_info);
        best_machine_by_product.push(bests);
    }

    let job_ops_len: Vec<usize> = job_products.iter().map(|&p| product_ops[p].len()).collect();
    let avg_machine_load = (total_min_work / (num_machines as f64).max(1.0)).max(1.0);
    let avg_op_min = (total_min_work / (total_ops as f64).max(1.0)).max(1.0);
    let flex_avg = (total_flex_weighted / (total_ops as f64).max(1.0)).max(1.0);
    let flex_factor = (3.0 / flex_avg).clamp(0.6, 2.2);
    let high_flex = ((flex_avg - 3.0) / 7.0).clamp(0.0, 1.0);

    let load_cv = {
        let mean = avg_machine_load.max(1e-9);
        let mut var = 0.0f64;
        for &x in &machine_load0 {
            let d = (x / mean) - 1.0;
            var += d * d;
        }
        (var / (num_machines as f64)).sqrt().clamp(0.0, 2.5)
    };

    // Share of the jobs whose fastest machine is the modal one, averaged over operation ranks.
    let mut flow_sum = 0.0f64;
    let mut flow_cnt = 0usize;
    let mut counts = vec![0u32; num_machines];
    for op_idx in 0..max_ops {
        counts.fill(0);
        let mut tot = 0u32;
        for p in 0..num_products {
            if op_idx >= best_machine_by_product[p].len() {
                continue;
            }
            let bm = best_machine_by_product[p][op_idx];
            let w = challenge.jobs_per_product[p] as u32;
            if w == 0 {
                continue;
            }
            counts[bm] = counts[bm].saturating_add(w);
            tot = tot.saturating_add(w);
        }
        if tot > 0 {
            let mut mx = 0u32;
            for &c in &counts {
                mx = mx.max(c);
            }
            flow_sum += (mx as f64) / (tot as f64);
            flow_cnt += 1;
        }
    }
    let flow_like = if flow_cnt > 0 {
        (flow_sum / (flow_cnt as f64)).clamp(0.0, 1.0)
    } else {
        0.5
    };
    let jobshopness = (1.0 - flow_like).clamp(0.0, 1.0);

    let mut machine_weight = vec![1.0f64; num_machines];
    {
        let mean = avg_machine_load.max(1e-9);
        let exp = (1.10 + 0.35 * load_cv + 0.20 * jobshopness).clamp(1.05, 1.70);
        for m in 0..num_machines {
            let r = (machine_load0[m] / mean).max(0.05);
            machine_weight[m] = r.powf(exp).clamp(0.55, 3.75);
        }
    }

    let machine_best_pop = {
        let tot: f64 = machine_best_cnt.iter().sum();
        let mean = (tot / (num_machines as f64).max(1.0)).max(1e-9);
        let mut pop = vec![0.0f64; num_machines];
        for m in 0..num_machines {
            let r = (machine_best_cnt[m] / mean).clamp(0.0, 10.0);
            pop[m] = (r / (1.0 + r)).clamp(0.0, 1.0);
        }
        pop
    };

    let bn_focus = ((3.0 / flex_avg).clamp(0.7, 2.6)
        * (1.0 + 0.55 * load_cv)
        * (0.85 + 0.55 * jobshopness))
        .clamp(0.6, 3.4);

    let mut product_suf_min: Vec<Vec<u32>> = Vec::with_capacity(product_ops.len());
    let mut product_suf_avg: Vec<Vec<f64>> = Vec::with_capacity(product_ops.len());
    let mut product_suf_bn: Vec<Vec<f64>> = Vec::with_capacity(product_ops.len());
    let mut product_next_min: Vec<Vec<u32>> = Vec::with_capacity(product_ops.len());
    let mut product_next_flex_inv: Vec<Vec<f64>> = Vec::with_capacity(product_ops.len());
    let mut max_job_bn: f64 = 1e-9;

    for ops in product_ops.iter_mut() {
        let n = ops.len();
        let mut suf_m = vec![0u32; n + 1];
        let mut suf_a = vec![0.0f64; n + 1];
        let mut suf_bn = vec![0.0f64; n + 1];
        let mut nxt_m = vec![0u32; n + 1];
        let mut nxt_fi = vec![0.0f64; n + 1];
        for i in (0..n).rev() {
            let oi = &mut ops[i];
            if oi.flex == 0 || oi.machines.is_empty() || oi.min_pt >= INF {
                oi.bn_avg = 0.0;
            } else {
                let mut sum = 0.0f64;
                for &(m, pt) in &oi.machines {
                    sum += (pt as f64) * machine_weight[m];
                }
                oi.bn_avg = sum / (oi.flex as f64);
            }
            suf_m[i] = suf_m[i + 1].saturating_add(oi.min_pt.min(INF / 2));
            suf_a[i] = suf_a[i + 1] + oi.avg_pt;
            suf_bn[i] = suf_bn[i + 1] + oi.bn_avg;
            if i + 1 < n {
                let next = &ops[i + 1];
                nxt_m[i] = next.min_pt;
                nxt_fi[i] = if next.flex > 0 {
                    1.0 / (next.flex as f64)
                } else {
                    0.0
                };
            }
        }
        max_job_bn = max_job_bn.max(suf_bn[0]);
        product_suf_min.push(suf_m);
        product_suf_avg.push(suf_a);
        product_suf_bn.push(suf_bn);
        product_next_min.push(nxt_m);
        product_next_flex_inv.push(nxt_fi);
    }

    // Nearly-flow-shop instances: jobs are ranked by an NEH insertion order on their fastest times.
    let mut job_flow_pref = vec![0.0f64; num_jobs];
    let use_flow_pref = flow_like > 0.82 && max_ops >= 2;
    if use_flow_pref {
        let m = max_ops.max(1);
        let mut job_pt: Vec<Vec<u32>> = Vec::with_capacity(num_jobs);
        for j in 0..num_jobs {
            let ops = &product_ops[job_products[j]];
            let mut v = vec![0u32; m];
            for s in 0..m.min(ops.len()) {
                v[s] = ops[s].min_pt.min(INF / 2);
            }
            job_pt.push(v);
        }
        let mut jobs: Vec<usize> = (0..num_jobs).collect();
        jobs.sort_unstable_by(|&a, &b| {
            let sa: u32 = job_pt[a].iter().copied().sum();
            let sb: u32 = job_pt[b].iter().copied().sum();
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
                let mk = flow_makespan(&tmp, &job_pt, &mut comp);
                if mk < best_mk {
                    best_mk = mk;
                    best_pos = pos;
                }
            }
            perm.insert(best_pos, j);
        }
        let n1 = (num_jobs.saturating_sub(1)) as f64;
        for (pos, &j) in perm.iter().enumerate() {
            job_flow_pref[j] = if n1 > 0.0 {
                1.0 - (pos as f64) / n1
            } else {
                1.0
            };
        }
    }
    let flow_w = if use_flow_pref {
        let t = ((flow_like - 0.82) / 0.18).clamp(0.0, 1.0);
        let base = (0.10 + 0.26 * t).clamp(0.10, 0.36);
        let flex_adj = (1.0 - 0.45 * high_flex).clamp(0.55, 1.0);
        base * flex_adj
    } else {
        0.0
    };

    let horizon = machine_load0
        .iter()
        .cloned()
        .fold(0.0f64, f64::max)
        .max(avg_machine_load);
    let time_scale =
        (horizon * (2.65 + 0.15 * load_cv + 0.10 * jobshopness + 0.10 * high_flex)).max(1.0);

    Ok(Pre {
        job_products,
        job_ops_len,
        product_ops,
        product_suf_min,
        product_suf_avg,
        product_suf_bn,
        product_next_min,
        product_next_flex_inv,
        machine_load0,
        machine_weight,
        machine_best_pop,
        avg_machine_load,
        avg_op_min,
        horizon,
        time_scale,
        max_job_avg_work: max_job_avg_work.max(1.0),
        max_job_bn: max_job_bn.max(1e-9),
        flex_factor,
        high_flex,
        jobshopness,
        bn_focus,
        flow_w,
        job_flow_pref,
        total_ops,
    })
}
}

pub mod greedy {
//! Dispatching-rule baseline: five deterministic rules and ten randomised top-k runs.
use anyhow::{anyhow, Result};
use rand::seq::SliceRandom;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::job_scheduling::*;

/// Randomised dispatching runs after the five deterministic rules.
const RANDOM_RUNS: usize = 10;

#[derive(Clone, Copy)]
enum Rule {
    MostWork,
    MostOps,
    LeastFlex,
    ShortestProc,
    LongestProc,
}

const RULES: [Rule; 5] = [
    Rule::MostWork,
    Rule::MostOps,
    Rule::LeastFlex,
    Rule::ShortestProc,
    Rule::LongestProc,
];

#[derive(Clone, Copy)]
struct Candidate {
    job: usize,
    priority: f64,
    end: u32,
    pt: u32,
    flex: usize,
}

impl Candidate {
    /// Higher priority, then earlier end, shorter processing time, lower flexibility, lower job id.
    #[inline]
    fn beats(&self, other: &Candidate, eps: f64) -> bool {
        if (self.priority - other.priority).abs() > eps {
            self.priority > other.priority
        } else if self.end != other.end {
            self.end < other.end
        } else if self.pt != other.pt {
            self.pt < other.pt
        } else if self.flex != other.flex {
            self.flex < other.flex
        } else {
            self.job < other.job
        }
    }
}

pub fn baseline(challenge: &Challenge) -> Result<(Solution, u32)> {
    let num_jobs = challenge.num_jobs;
    let mut job_products = Vec::with_capacity(num_jobs);
    for (p, &cnt) in challenge.jobs_per_product.iter().enumerate() {
        for _ in 0..cnt {
            job_products.push(p);
        }
    }
    let job_ops_len: Vec<usize> = job_products
        .iter()
        .map(|&p| challenge.product_processing_times[p].len())
        .collect();
    let job_total_work: Vec<f64> = job_products
        .iter()
        .map(|&p| {
            challenge.product_processing_times[p]
                .iter()
                .map(avg_pt)
                .sum()
        })
        .collect();

    let mut best_mk = u32::MAX;
    let mut best_sol: Option<Solution> = None;
    for rule in RULES {
        let (sol, mk) = run_rule(
            challenge,
            &job_products,
            &job_ops_len,
            &job_total_work,
            rule,
            None,
        )?;
        if mk < best_mk {
            best_mk = mk;
            best_sol = Some(sol);
        }
    }
    let mut rng = SmallRng::from_seed(challenge.seed);
    for _ in 0..RANDOM_RUNS {
        let seed = rng.gen::<u64>();
        let rule = RULES[rng.gen_range(0..RULES.len())];
        let top_k = rng.gen_range(2..=5);
        let mut local_rng = SmallRng::seed_from_u64(seed);
        let (sol, mk) = run_rule(
            challenge,
            &job_products,
            &job_ops_len,
            &job_total_work,
            rule,
            Some((top_k, &mut local_rng)),
        )?;
        if mk < best_mk {
            best_mk = mk;
            best_sol = Some(sol);
        }
    }
    Ok((
        best_sol.ok_or_else(|| anyhow!("No greedy solution"))?,
        best_mk,
    ))
}

#[inline]
fn avg_pt(op: &std::collections::HashMap<usize, u32>) -> f64 {
    op.values().sum::<u32>() as f64 / op.len().max(1) as f64
}

/// Event-driven dispatching state.
struct Dispatch<'a> {
    challenge: &'a Challenge,
    job_products: &'a [usize],
    job_ops_len: &'a [usize],
    rule: Rule,
    job_next_op: Vec<usize>,
    job_ready: Vec<u32>,
    machine_avail: Vec<u32>,
    job_work_left: Vec<f64>,
    job_schedule: Vec<Vec<(usize, u32)>>,
    remaining: usize,
    time: u32,
}

impl<'a> Dispatch<'a> {
    #[inline]
    fn op_times(&self, job: usize) -> &'a std::collections::HashMap<usize, u32> {
        &self.challenge.product_processing_times[self.job_products[job]][self.job_next_op[job]]
    }

    /// The next operation of `job` on `m`, when `m` finishes it no later than any other eligible
    /// machine.
    fn candidate(&self, job: usize, m: usize) -> Option<Candidate> {
        if self.job_next_op[job] >= self.job_ops_len[job] || self.job_ready[job] > self.time {
            return None;
        }
        let op_times = self.op_times(job);
        let pt = *op_times.get(&m)?;
        let earliest = op_times
            .iter()
            .map(|(&mm, &ppt)| self.time.max(self.machine_avail[mm]) + ppt)
            .min()
            .unwrap_or(u32::MAX);
        let end = self.time.max(self.machine_avail[m]) + pt;
        if end != earliest {
            return None;
        }
        let flex = op_times.len();
        let priority = match self.rule {
            Rule::MostWork => self.job_work_left[job],
            Rule::MostOps => (self.job_ops_len[job] - self.job_next_op[job]) as f64,
            Rule::LeastFlex => -(flex as f64),
            Rule::ShortestProc => -(pt as f64),
            Rule::LongestProc => pt as f64,
        };
        Some(Candidate {
            job,
            priority,
            end,
            pt,
            flex,
        })
    }

    fn schedule(&mut self, c: Candidate, m: usize) {
        let job = c.job;
        let work = avg_pt(self.op_times(job));
        let st = self.time.max(self.machine_avail[m]);
        let end = st + c.pt;
        self.job_schedule[job].push((m, st));
        self.job_next_op[job] += 1;
        self.job_ready[job] = end;
        self.machine_avail[m] = end;
        self.job_work_left[job] -= work;
        if self.job_work_left[job] < 0.0 {
            self.job_work_left[job] = 0.0;
        }
        self.remaining -= 1;
    }

    /// Earliest machine release or job readiness after the current time.
    fn next_event(&self) -> Option<u32> {
        let mut next = u32::MAX;
        for &t in &self.machine_avail {
            if t > self.time && t < next {
                next = t;
            }
        }
        for j in 0..self.job_ready.len() {
            if self.job_next_op[j] < self.job_ops_len[j]
                && self.job_ready[j] > self.time
                && self.job_ready[j] < next
            {
                next = self.job_ready[j];
            }
        }
        if next == u32::MAX {
            None
        } else {
            Some(next)
        }
    }
}

/// Each idle machine takes the best candidate by rule, or a random one of the top-k when a rng is
/// supplied.
fn run_rule(
    challenge: &Challenge,
    job_products: &[usize],
    job_ops_len: &[usize],
    job_total_work: &[f64],
    rule: Rule,
    mut random_top_k: Option<(usize, &mut SmallRng)>,
) -> Result<(Solution, u32)> {
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;
    let mut d = Dispatch {
        challenge,
        job_products,
        job_ops_len,
        rule,
        job_next_op: vec![0usize; num_jobs],
        job_ready: vec![0u32; num_jobs],
        machine_avail: vec![0u32; num_machines],
        job_work_left: job_total_work.to_vec(),
        job_schedule: job_ops_len
            .iter()
            .map(|&len| Vec::with_capacity(len))
            .collect(),
        remaining: job_ops_len.iter().sum::<usize>(),
        time: 0,
    };
    let eps = 1e-9;
    let mut available_machines: Vec<usize> = Vec::with_capacity(num_machines);
    let mut candidates: Vec<Candidate> = Vec::new();

    while d.remaining > 0 {
        available_machines.clear();
        for m in 0..num_machines {
            if d.machine_avail[m] <= d.time {
                available_machines.push(m);
            }
        }
        if let Some((_, ref mut rng)) = random_top_k {
            available_machines.shuffle(*rng);
        }

        for &m in &available_machines {
            let chosen = if let Some((top_k, ref mut rng)) = random_top_k {
                candidates.clear();
                candidates.extend((0..num_jobs).filter_map(|j| d.candidate(j, m)));
                if candidates.is_empty() {
                    continue;
                }
                candidates.sort_by(|a, b| {
                    if (b.priority - a.priority).abs() > eps {
                        b.priority.total_cmp(&a.priority)
                    } else if a.end != b.end {
                        a.end.cmp(&b.end)
                    } else if a.pt != b.pt {
                        a.pt.cmp(&b.pt)
                    } else if a.flex != b.flex {
                        a.flex.cmp(&b.flex)
                    } else {
                        a.job.cmp(&b.job)
                    }
                });
                let top = candidates.len().min(top_k);
                candidates[rng.gen_range(0..top)]
            } else {
                let mut best: Option<Candidate> = None;
                for j in 0..num_jobs {
                    if let Some(c) = d.candidate(j, m) {
                        if best.map_or(true, |b| c.beats(&b, eps)) {
                            best = Some(c);
                        }
                    }
                }
                let Some(best) = best else { continue };
                best
            };
            d.schedule(chosen, m);
        }

        if d.remaining == 0 {
            break;
        }
        d.time = d
            .next_event()
            .ok_or_else(|| anyhow!("Greedy baseline stuck"))?;
    }
    let mk = d.job_ready.iter().copied().max().unwrap_or(0);
    Ok((
        Solution {
            job_schedule: d.job_schedule,
        },
        mk,
    ))
}
}

pub mod disjunctive {
//! Disjunctive schedule: job chains plus one sequence per machine, evaluated by a topological pass.
use super::types::*;
use anyhow::{anyhow, Result};
use tig_challenges::job_scheduling::*;

#[derive(Clone)]
pub struct DisjSchedule {
    pub n: usize,
    pub num_jobs: usize,
    pub num_machines: usize,
    pub job_offsets: Vec<usize>,
    pub job_succ: Vec<usize>,
    pub indeg_job: Vec<u16>,
    pub node_machine: Vec<usize>,
    pub node_pt: Vec<u32>,
    pub node_job: Vec<usize>,
    pub node_op: Vec<usize>,
    pub machine_seq: Vec<Vec<usize>>,
}

/// Scratch space of `eval`; `start` and `best_pred` describe the last evaluated schedule.
pub struct EvalBuf {
    pub indeg: Vec<u16>,
    pub start: Vec<u32>,
    pub best_pred: Vec<usize>,
    pub machine_succ: Vec<usize>,
    pub stack: Vec<usize>,
}

impl EvalBuf {
    pub fn new(n: usize) -> Self {
        Self {
            indeg: vec![0u16; n],
            start: vec![0u32; n],
            best_pred: vec![NONE; n],
            machine_succ: vec![NONE; n],
            stack: Vec::with_capacity(n),
        }
    }
}

impl DisjSchedule {
    pub fn from_solution(
        pre: &Pre,
        challenge: &Challenge,
        sol: &Solution,
    ) -> Result<DisjSchedule> {
        let num_jobs = challenge.num_jobs;
        let num_machines = challenge.num_machines;
        let mut job_offsets = vec![0usize; num_jobs + 1];
        for j in 0..num_jobs {
            job_offsets[j + 1] = job_offsets[j] + pre.job_ops_len[j];
        }
        let n = job_offsets[num_jobs];
        if n == 0 {
            return Err(anyhow!("No operations"));
        }
        let mut node_machine = vec![0usize; n];
        let mut node_pt = vec![0u32; n];
        let mut node_job = vec![0usize; n];
        let mut node_op = vec![0usize; n];
        let mut per_machine: Vec<Vec<(u32, usize)>> = vec![Vec::new(); num_machines];
        for job in 0..num_jobs {
            let expected = pre.job_ops_len[job];
            if sol.job_schedule[job].len() != expected {
                return Err(anyhow!("Invalid solution: job {} ops len mismatch", job));
            }
            let product = pre.job_products[job];
            for op_idx in 0..expected {
                let id = job_offsets[job] + op_idx;
                let (m, st) = sol.job_schedule[job][op_idx];
                let op = &pre.product_ops[product][op_idx];
                let pt = op
                    .pt_on(m)
                    .ok_or_else(|| anyhow!("Invalid solution: pt missing"))?;
                if m >= num_machines {
                    return Err(anyhow!("Invalid solution: machine out of range"));
                }
                node_machine[id] = m;
                node_pt[id] = pt;
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
        let mut job_succ = vec![NONE; n];
        let mut indeg_job = vec![0u16; n];
        for job in 0..num_jobs {
            let len = pre.job_ops_len[job];
            let base = job_offsets[job];
            for k in 0..len {
                let id = base + k;
                if k + 1 < len {
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

    /// Earliest starts by a topological pass; returns the makespan and its last node, or None on a
    /// cycle.
    pub fn eval(&self, buf: &mut EvalBuf) -> Option<(u32, usize)> {
        let n = self.n;
        buf.indeg.clone_from_slice(&self.indeg_job);
        buf.start.fill(0);
        buf.best_pred.fill(NONE);
        buf.stack.clear();
        for seq in &self.machine_seq {
            if seq.is_empty() {
                continue;
            }
            let mut prev = seq[0];
            for &v in &seq[1..] {
                buf.machine_succ[prev] = v;
                buf.indeg[v] = buf.indeg[v].saturating_add(1);
                prev = v;
            }
            buf.machine_succ[prev] = NONE;
        }
        for i in 0..n {
            if buf.indeg[i] == 0 {
                buf.stack.push(i);
            }
        }
        let mut processed = 0usize;
        let mut mk = 0u32;
        let mut mk_node = 0usize;
        while let Some(u) = buf.stack.pop() {
            processed += 1;
            let end_u = buf.start[u].saturating_add(self.node_pt[u]);
            if end_u > mk {
                mk = end_u;
                mk_node = u;
            }
            for succ in [self.job_succ[u], buf.machine_succ[u]] {
                if succ == NONE {
                    continue;
                }
                if buf.start[succ] < end_u {
                    buf.start[succ] = end_u;
                    buf.best_pred[succ] = u;
                }
                buf.indeg[succ] = buf.indeg[succ].saturating_sub(1);
                if buf.indeg[succ] == 0 {
                    buf.stack.push(succ);
                }
            }
        }
        if processed != n {
            return None;
        }
        Some((mk, mk_node))
    }

    pub fn to_solution(&self, pre: &Pre, start: &[u32]) -> Solution {
        let mut job_schedule: Vec<Vec<(usize, u32)>> = Vec::with_capacity(self.num_jobs);
        for j in 0..self.num_jobs {
            let len = pre.job_ops_len[j];
            let base = self.job_offsets[j];
            let mut v = Vec::with_capacity(len);
            for k in 0..len {
                let id = base + k;
                v.push((self.node_machine[id], start[id]));
            }
            job_schedule.push(v);
        }
        Solution { job_schedule }
    }

    /// Marks the nodes of the critical path ending at `mk_node`.
    pub fn mark_critical(&self, buf: &EvalBuf, mk_node: usize, crit: &mut [bool]) {
        crit.fill(false);
        let mut u = mk_node;
        while u != NONE {
            crit[u] = true;
            u = buf.best_pred[u];
        }
    }

    /// Maximal runs of at least two critical, back-to-back operations on one machine, as (machine,
    /// first, last).
    pub fn critical_blocks(
        &self,
        buf: &EvalBuf,
        crit: &[bool],
        out: &mut Vec<(usize, usize, usize)>,
    ) {
        out.clear();
        for m in 0..self.num_machines {
            let seq = &self.machine_seq[m];
            if seq.len() <= 1 {
                continue;
            }
            let mut i = 0usize;
            while i < seq.len() {
                if !crit[seq[i]] {
                    i += 1;
                    continue;
                }
                let bend = self.block_end(buf, crit, seq, i);
                if bend > i {
                    out.push((m, i, bend));
                }
                i = bend + 1;
            }
        }
    }

    /// Last index of the critical back-to-back run starting at `bstart`.
    #[inline]
    pub fn block_end(
        &self,
        buf: &EvalBuf,
        crit: &[bool],
        seq: &[usize],
        bstart: usize,
    ) -> usize {
        let mut bend = bstart;
        while bend + 1 < seq.len() {
            let x = seq[bend];
            let y = seq[bend + 1];
            if !crit[y] || buf.start[y] != buf.start[x].saturating_add(self.node_pt[x]) {
                break;
            }
            bend += 1;
        }
        bend
    }

    /// Moves the node at `idx_from` of `m_from` to `m_to` at `idx_to`; returns (node, old pt,
    /// insertion index).
    pub fn reroute(
        &mut self,
        m_from: usize,
        idx_from: usize,
        m_to: usize,
        idx_to: usize,
        new_pt: u32,
    ) -> (usize, u32, usize) {
        let node = self.machine_seq[m_from].remove(idx_from);
        let old_pt = self.node_pt[node];
        self.node_machine[node] = m_to;
        self.node_pt[node] = new_pt;
        let ins = idx_to.min(self.machine_seq[m_to].len());
        self.machine_seq[m_to].insert(ins, node);
        (node, old_pt, ins)
    }

    pub fn undo_reroute(
        &mut self,
        m_from: usize,
        idx_from: usize,
        m_to: usize,
        ins_idx: usize,
        node: usize,
        old_pt: u32,
    ) {
        let x = self.machine_seq[m_to].remove(ins_idx);
        debug_assert_eq!(x, node);
        let ins_from = idx_from.min(self.machine_seq[m_from].len());
        self.machine_seq[m_from].insert(ins_from, node);
        self.node_machine[node] = m_from;
        self.node_pt[node] = old_pt;
    }
}

/// Moves `seq[from]` so that it lands at index `min(to, len - 1)`; returns that index.
#[inline]
pub fn apply_insert(seq: &mut [usize], from: usize, to: usize) -> usize {
    let len = seq.len();
    if len == 0 || from >= len {
        return from.min(len.saturating_sub(1));
    }
    let t = to.min(len - 1);
    if t > from {
        seq[from..=t].rotate_left(1);
    } else if t < from {
        seq[t..=from].rotate_right(1);
    }
    t
}

/// Swaps `seq[i]` and `seq[i + 1]`; false when out of range.
#[inline]
pub fn apply_swap(seq: &mut [usize], i: usize) -> bool {
    if i + 1 >= seq.len() {
        return false;
    }
    seq.swap(i, i + 1);
    true
}

/// First index of `seq` whose start is at least `desired_start`.
#[inline]
pub fn insert_pos_by_start(seq: &[usize], start: &[u32], desired_start: u32) -> usize {
    let mut lo = 0usize;
    let mut hi = seq.len();
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if start[seq[mid]] < desired_start {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// The two fastest (machine, pt) options of an operation, `(NONE, INF)` when absent.
pub fn best_two_by_pt(op: &OpInfo) -> [(usize, u32); 2] {
    let mut r = [(NONE, INF); 2];
    for &(m, pt) in &op.machines {
        if pt < r[0].1 {
            r[1] = r[0];
            r[0] = (m, pt);
        } else if pt < r[1].1 {
            r[1] = (m, pt);
        }
    }
    r
}

/// Inserts `c` into `top`, kept sorted by decreasing `key` and truncated to `k`.
#[inline]
pub fn push_top_k<T: Copy, K: PartialOrd>(
    top: &mut Vec<T>,
    c: T,
    k: usize,
    key: impl Fn(&T) -> K,
) {
    if k == 0 {
        return;
    }
    let mut pos = top.len();
    while pos > 0 && key(&top[pos - 1]) < key(&c) {
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
}

pub mod descent {
//! Critical-block descent with random block perturbations.
use super::disjunctive::*;
use super::types::*;
use anyhow::Result;
use tig_challenges::job_scheduling::*;

/// Block-swap perturbations applied per cycle.
const PERTURB_SWAPS: usize = 2;

#[derive(Clone, Copy, PartialEq, Eq)]
enum MoveKind {
    Insert,
    Reroute,
    Swap,
}

#[derive(Clone, Copy)]
struct MoveCand {
    kind: MoveKind,
    m_from: usize,
    from: usize,
    m_to: usize,
    to: usize,
    new_pt: u32,
    score: u32,
}

/// Descends from `base_sol`, then repeats `perturb_cycles` times a two-swap kick of the best
/// schedule followed by a descent; returns the improved schedule, or None when nothing beat the
/// start.
pub fn block_descent(
    pre: &Pre,
    challenge: &Challenge,
    base_sol: &Solution,
    max_iters: usize,
    top_cands: usize,
    perturb_cycles: usize,
) -> Result<Option<(Solution, u32)>> {
    let mut ds = DisjSchedule::from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n);
    let mut crit = vec![false; ds.n];
    let mut cur_eval = match ds.eval(&mut buf) {
        Some(x) => x,
        None => return Ok(None),
    };
    let initial_mk = cur_eval.0;
    descend(
        &mut ds,
        &mut buf,
        &mut crit,
        pre,
        &mut cur_eval,
        max_iters,
        top_cands,
    );
    let Some((mk_after, _)) = ds.eval(&mut buf) else {
        return Ok(None);
    };
    let mut global_best_mk = mk_after;
    let mut global_best_ds = ds.clone();

    let mut sol_hash: u64 = 0;
    for m in 0..ds.num_machines.min(8) {
        if !ds.machine_seq[m].is_empty() {
            let first_node = ds.machine_seq[m][0];
            sol_hash ^= (first_node as u64).wrapping_mul(0xD2B54A6B68A5);
            sol_hash = sol_hash.rotate_left(7);
        }
    }
    let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ (initial_mk as u64).wrapping_shl(16)
        ^ (ds.n as u64)
        ^ sol_hash;
    let mut blocks: Vec<(usize, usize, usize)> = Vec::new();

    for _ in 0..perturb_cycles {
        ds = global_best_ds.clone();
        let Some((_, mk_node)) = ds.eval(&mut buf) else {
            break;
        };
        ds.mark_critical(&buf, mk_node, &mut crit);
        ds.critical_blocks(&buf, &crit, &mut blocks);
        if blocks.is_empty() {
            break;
        }
        for _ in 0..PERTURB_SWAPS {
            pseed = xorshift(pseed);
            let (m, bstart, bend) = blocks[(pseed as usize) % blocks.len()];
            let block_len = bend - bstart;
            if block_len == 0 {
                continue;
            }
            pseed = xorshift(pseed);
            let swap_pos = bstart + ((pseed as usize) % block_len);
            if swap_pos + 1 < ds.machine_seq[m].len() {
                ds.machine_seq[m].swap(swap_pos, swap_pos + 1);
            }
        }
        match ds.eval(&mut buf) {
            Some(x) => cur_eval = x,
            None => continue,
        }
        descend(
            &mut ds,
            &mut buf,
            &mut crit,
            pre,
            &mut cur_eval,
            max_iters,
            top_cands,
        );
        if let Some((mk_now, _)) = ds.eval(&mut buf) {
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
    let Some((mk_final, _)) = ds.eval(&mut buf) else {
        return Ok(None);
    };
    Ok(Some((ds.to_solution(pre, &buf.start), mk_final)))
}

#[inline]
fn xorshift(mut x: u64) -> u64 {
    x ^= x.wrapping_shl(13);
    x ^= x.wrapping_shr(7);
    x ^= x.wrapping_shl(17);
    x
}

/// Best-improvement descent over the critical blocks; stops at the first iteration without a strict
/// improvement.
fn descend(
    ds: &mut DisjSchedule,
    buf: &mut EvalBuf,
    crit: &mut [bool],
    pre: &Pre,
    cur_eval: &mut (u32, usize),
    max_iters: usize,
    top_cands: usize,
) {
    let mut cur_mk = cur_eval.0;
    let mut cands: Vec<MoveCand> = Vec::with_capacity(top_cands.min(64));
    for _ in 0..max_iters {
        ds.mark_critical(buf, cur_eval.1, crit);
        cands.clear();
        collect_moves(ds, buf, crit, pre, top_cands, &mut cands);
        if cands.is_empty() {
            break;
        }
        let mut best_cand: Option<MoveCand> = None;
        let mut best_mk = cur_mk;
        for cand in &cands {
            if let Some(mk2) = try_move(ds, buf, *cand) {
                if mk2 < best_mk {
                    best_mk = mk2;
                    best_cand = Some(*cand);
                }
            }
        }
        let Some(bc) = best_cand else { break };
        match commit_move(ds, buf, bc, cur_mk) {
            Some(ne) => {
                *cur_eval = ne;
                cur_mk = ne.0;
            }
            None => break,
        }
    }
}

/// Insertions and swaps around each critical block, and reroutes of its end operations.
fn collect_moves(
    ds: &DisjSchedule,
    buf: &EvalBuf,
    crit: &[bool],
    pre: &Pre,
    top_cands: usize,
    cands: &mut Vec<MoveCand>,
) {
    let push =
        |cands: &mut Vec<MoveCand>, c: MoveCand| push_top_k(cands, c, top_cands, |x| x.score);
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
            let bend = ds.block_end(buf, crit, seq, i);
            if bend > bstart {
                let same = |from: usize, to: usize, score: u32| MoveCand {
                    kind: MoveKind::Insert,
                    m_from: m,
                    from,
                    m_to: m,
                    to,
                    new_pt: 0,
                    score,
                };
                let swap = |from: usize, score: u32| MoveCand {
                    kind: MoveKind::Swap,
                    m_from: m,
                    from,
                    m_to: m,
                    to: 0,
                    new_pt: 0,
                    score,
                };
                let max_shift = bend - bstart;
                let mut shifts: [usize; 3] = [1, 2, max_shift];
                for sh in shifts.iter_mut() {
                    if *sh > max_shift {
                        *sh = 0;
                    }
                }
                for &sh in &shifts {
                    if sh == 0 {
                        continue;
                    }
                    if bstart < seq.len() && bstart + sh <= seq.len() {
                        let tgt_idx = (bstart + sh).min(seq.len() - 1);
                        push(cands, same(bstart, bstart + sh, buf.start[seq[tgt_idx]]));
                    }
                    push(cands, same(bend, bend - sh, buf.start[seq[bend]]));
                }
                if bstart > 0 {
                    push(cands, swap(bstart - 1, buf.start[seq[bstart]]));
                }
                if bend + 1 < seq.len() {
                    push(cands, swap(bend, buf.start[seq[bend]]));
                }
                if bstart + 1 <= bend {
                    push(cands, swap(bstart, buf.start[seq[bstart + 1]]));
                    if bend >= 1 && bend - 1 >= bstart {
                        push(cands, swap(bend - 1, buf.start[seq[bend]]));
                    }
                }
                for &idx in &[bstart, bend] {
                    if idx >= seq.len() {
                        continue;
                    }
                    let node = seq[idx];
                    if !crit[node] {
                        continue;
                    }
                    let op =
                        &pre.product_ops[pre.job_products[ds.node_job[node]]][ds.node_op[node]];
                    if op.flex < 2 || op.machines.len() < 2 {
                        continue;
                    }
                    let old_m = ds.node_machine[node];
                    let old_pt = ds.node_pt[node];
                    let w_from = pre.machine_weight[old_m].max(1e-9);
                    for &(m_to, new_pt) in &best_two_by_pt(op) {
                        if m_to == NONE
                            || m_to >= ds.num_machines
                            || m_to == old_m
                            || new_pt >= INF
                        {
                            continue;
                        }
                        let w_to = pre.machine_weight[m_to].max(1e-9);
                        if !(new_pt + 1 < old_pt || w_to < w_from * 0.90) {
                            continue;
                        }
                        let desired = buf.start[node];
                        let pos0 =
                            insert_pos_by_start(&ds.machine_seq[m_to], &buf.start, desired);
                        for pos in [pos0, pos0.saturating_add(1)] {
                            if pos > ds.machine_seq[m_to].len() {
                                continue;
                            }
                            let diffw =
                                ((w_from - w_to).max(0.0) * pre.avg_op_min).max(0.0) as u32;
                            let difpt = old_pt.saturating_sub(new_pt);
                            let score = desired
                                .saturating_add(old_pt)
                                .saturating_add(diffw)
                                .saturating_add(difpt.saturating_mul(2));
                            push(
                                cands,
                                MoveCand {
                                    kind: MoveKind::Reroute,
                                    m_from: old_m,
                                    from: idx,
                                    m_to,
                                    to: pos,
                                    new_pt,
                                    score,
                                },
                            );
                        }
                    }
                }
            }
            i = bend + 1;
        }
    }
}

/// Applies `cand`, evaluates, and undoes it; None when the move is stale or creates a cycle.
fn try_move(ds: &mut DisjSchedule, buf: &mut EvalBuf, cand: MoveCand) -> Option<u32> {
    match cand.kind {
        MoveKind::Insert => {
            let m = cand.m_from;
            if m >= ds.num_machines || cand.from >= ds.machine_seq[m].len() {
                return None;
            }
            let new_idx = apply_insert(&mut ds.machine_seq[m], cand.from, cand.to);
            let mk = ds.eval(buf).map(|e| e.0);
            apply_insert(&mut ds.machine_seq[m], new_idx, cand.from);
            mk
        }
        MoveKind::Swap => {
            let m = cand.m_from;
            if m >= ds.num_machines || cand.from + 1 >= ds.machine_seq[m].len() {
                return None;
            }
            if !apply_swap(&mut ds.machine_seq[m], cand.from) {
                return None;
            }
            let mk = ds.eval(buf).map(|e| e.0);
            apply_swap(&mut ds.machine_seq[m], cand.from);
            mk
        }
        MoveKind::Reroute => {
            let (m_from, m_to) = (cand.m_from, cand.m_to);
            if m_from >= ds.num_machines
                || m_to >= ds.num_machines
                || cand.from >= ds.machine_seq[m_from].len()
            {
                return None;
            }
            let node = ds.machine_seq[m_from][cand.from];
            if ds.node_machine[node] != m_from {
                return None;
            }
            let (node2, old_pt, ins_idx) =
                ds.reroute(m_from, cand.from, m_to, cand.to, cand.new_pt);
            let mk = ds.eval(buf).map(|e| e.0);
            ds.undo_reroute(m_from, cand.from, m_to, ins_idx, node2, old_pt);
            mk
        }
    }
}

/// Applies `bc` and keeps it when the makespan strictly improves on `cur_mk`; otherwise undoes it.
fn commit_move(
    ds: &mut DisjSchedule,
    buf: &mut EvalBuf,
    bc: MoveCand,
    cur_mk: u32,
) -> Option<(u32, usize)> {
    match bc.kind {
        MoveKind::Insert => {
            let m = bc.m_from;
            let new_idx = apply_insert(&mut ds.machine_seq[m], bc.from, bc.to);
            match ds.eval(buf) {
                Some(ne) if ne.0 < cur_mk => Some(ne),
                _ => {
                    apply_insert(&mut ds.machine_seq[m], new_idx, bc.from);
                    None
                }
            }
        }
        MoveKind::Swap => {
            let m = bc.m_from;
            if m >= ds.num_machines
                || bc.from + 1 >= ds.machine_seq[m].len()
                || !apply_swap(&mut ds.machine_seq[m], bc.from)
            {
                return None;
            }
            match ds.eval(buf) {
                Some(ne) if ne.0 < cur_mk => Some(ne),
                _ => {
                    apply_swap(&mut ds.machine_seq[m], bc.from);
                    None
                }
            }
        }
        MoveKind::Reroute => {
            if bc.m_from >= ds.num_machines
                || bc.m_to >= ds.num_machines
                || bc.from >= ds.machine_seq[bc.m_from].len()
            {
                return None;
            }
            let (node2, old_pt, ins_idx) =
                ds.reroute(bc.m_from, bc.from, bc.m_to, bc.to, bc.new_pt);
            match ds.eval(buf) {
                Some(ne) if ne.0 < cur_mk => Some(ne),
                _ => {
                    ds.undo_reroute(bc.m_from, bc.from, bc.m_to, ins_idx, node2, old_pt);
                    None
                }
            }
        }
    }
}
}

pub mod reassign {
//! Greedy machine reassignment: moves single operations to another eligible machine when the
//! makespan drops.
use super::disjunctive::*;
use super::types::*;
use anyhow::{anyhow, Result};
use tig_challenges::job_scheduling::*;

/// Insertion positions tried per target machine.
const MAX_POSITIONS: usize = 7;

/// Repeats first-improvement passes over the flexible operations, most critical first.
pub fn reassign_pass(
    pre: &Pre,
    challenge: &Challenge,
    base_sol: &Solution,
) -> Result<Option<(Solution, u32)>> {
    let mut ds = DisjSchedule::from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n);
    let n = ds.n;
    let Some((mut current_mk, _)) = ds.eval(&mut buf) else {
        return Ok(None);
    };
    let initial_mk = current_mk;
    let mut job_pred = vec![NONE; n];
    for j in 0..ds.num_jobs {
        for k in (ds.job_offsets[j] + 1)..ds.job_offsets[j + 1] {
            job_pred[k] = k - 1;
        }
    }
    let mut cur_starts = buf.start.clone();
    let flex01 = (pre.high_flex + pre.jobshopness).clamp(0.0, 1.5);
    let focus_base = ((pre.avg_op_min * (1.8 + 1.2 * flex01)).max(1.0)) as u32;
    let max_rounds = if flex01 > 0.60 { 12 } else { 10 };
    let mut improved_any = false;

    for _ in 0..max_rounds {
        let Some((mk_now, _)) = ds.eval(&mut buf) else {
            return Ok(None);
        };
        current_mk = mk_now;
        cur_starts.clone_from(&buf.start);
        let tails = tails(&ds, &buf);
        let mut moved = false;

        if flex01 > 0.55 {
            let mut order =
                candidate_order(pre, &ds, &cur_starts, current_mk, &tails, Some(focus_base));
            if !order.is_empty() {
                order.truncate(((n / 3).max(24)).min(order.len()));
                for node in order {
                    if try_node(
                        pre,
                        &mut ds,
                        &mut buf,
                        &job_pred,
                        &mut cur_starts,
                        &mut current_mk,
                        node,
                    )? {
                        improved_any = true;
                        moved = true;
                        break;
                    }
                }
            }
        }
        if !moved {
            for node in candidate_order(pre, &ds, &cur_starts, current_mk, &tails, None) {
                if try_node(
                    pre,
                    &mut ds,
                    &mut buf,
                    &job_pred,
                    &mut cur_starts,
                    &mut current_mk,
                    node,
                )? {
                    improved_any = true;
                    moved = true;
                    break;
                }
            }
        }
        if !moved {
            break;
        }
    }

    if !improved_any || current_mk >= initial_mk {
        return Ok(None);
    }
    let Some((mk_now, _)) = ds.eval(&mut buf) else {
        return Ok(None);
    };
    if mk_now >= initial_mk {
        return Ok(None);
    }
    Ok(Some((ds.to_solution(pre, &buf.start), mk_now)))
}

/// Flexible operations by increasing slack, then by decreasing machine popularity, flexibility and
/// sequence length. With `focus`, operations whose slack exceeds the focus window are left out.
fn candidate_order(
    pre: &Pre,
    ds: &DisjSchedule,
    starts: &[u32],
    current_mk: u32,
    tails: &[u32],
    focus: Option<u32>,
) -> Vec<usize> {
    let mut order: Vec<(usize, u32, u16, usize, usize)> = Vec::with_capacity(ds.n);
    for node in 0..ds.n {
        let op_info = &pre.product_ops[pre.job_products[ds.node_job[node]]][ds.node_op[node]];
        let flex = op_info.machines.len();
        if flex <= 1 {
            continue;
        }
        let cur_machine = ds.node_machine[node];
        let path = starts[node]
            .saturating_add(ds.node_pt[node])
            .saturating_add(tails[node]);
        let slack = current_mk.saturating_sub(path);
        if let Some(focus_base) = focus {
            let extra = (flex.saturating_sub(2).min(3) as u32)
                .saturating_mul((pre.avg_op_min.max(1.0) as u32).max(1));
            if slack > focus_base.saturating_add(extra) {
                continue;
            }
        }
        let pop = (pre.machine_best_pop[cur_machine].clamp(0.0, 1.0) * 1000.0) as u16;
        order.push((node, slack, pop, flex, ds.machine_seq[cur_machine].len()));
    }
    order.sort_by(|a, b| {
        a.1.cmp(&b.1)
            .then_with(|| b.2.cmp(&a.2))
            .then_with(|| b.3.cmp(&a.3))
            .then_with(|| b.4.cmp(&a.4))
            .then_with(|| a.0.cmp(&b.0))
    });
    order.into_iter().map(|x| x.0).collect()
}

/// Moves `node` to the eligible machine and position that most reduces the makespan, if any.
fn try_node(
    pre: &Pre,
    ds: &mut DisjSchedule,
    buf: &mut EvalBuf,
    job_pred: &[usize],
    cur_starts: &mut Vec<u32>,
    current_mk: &mut u32,
    node: usize,
) -> Result<bool> {
    let op_info = &pre.product_ops[pre.job_products[ds.node_job[node]]][ds.node_op[node]];
    if op_info.machines.len() <= 1 {
        return Ok(false);
    }
    let cur_machine = ds.node_machine[node];
    let cur_pt = ds.node_pt[node];
    let Some(old_pos) = ds.machine_seq[cur_machine].iter().position(|&x| x == node) else {
        return Ok(false);
    };
    let cur_start = cur_starts[node];
    let mut best_m = cur_machine;
    let mut best_pt = cur_pt;
    let mut best_mk = *current_mk;
    let mut best_ins_pos = 0usize;

    {
        let seq = &mut ds.machine_seq[cur_machine];
        seq[old_pos..].rotate_left(1);
        seq.pop();
    }

    for &(new_m, new_pt) in &op_info.machines {
        if new_m == cur_machine {
            continue;
        }
        ds.node_machine[node] = new_m;
        ds.node_pt[node] = new_pt;
        let target_len = ds.machine_seq[new_m].len();
        let mut positions = insert_positions(ds, cur_starts, node, new_m, job_pred);
        let mut sorted_pos = target_len;
        for (k, &nd) in ds.machine_seq[new_m].iter().enumerate() {
            if cur_starts[nd] >= cur_start {
                sorted_pos = k;
                break;
            }
        }
        for p in [sorted_pos, sorted_pos.saturating_sub(1), target_len] {
            if p <= target_len && !positions.contains(&p) {
                positions.push(p);
            }
        }
        if target_len > 2 {
            let mid = (sorted_pos + target_len) / 2;
            if mid <= target_len && !positions.contains(&mid) {
                positions.push(mid);
            }
        }
        positions.truncate(MAX_POSITIONS);

        for insert_pos in positions {
            let ins = insert_pos.min(ds.machine_seq[new_m].len());
            ds.machine_seq[new_m].insert(ins, node);
            if let Some((test_mk, _)) = ds.eval(buf) {
                if test_mk < best_mk {
                    best_mk = test_mk;
                    best_m = new_m;
                    best_pt = new_pt;
                    best_ins_pos = ins;
                }
            }
            ds.machine_seq[new_m].remove(ins);
        }
    }

    if best_m != cur_machine {
        let ins = best_ins_pos.min(ds.machine_seq[best_m].len());
        ds.machine_seq[best_m].insert(ins, node);
        ds.node_machine[node] = best_m;
        ds.node_pt[node] = best_pt;
        let Some((mk_now, _)) = ds.eval(buf) else {
            return Err(anyhow!("Stalled greedy reassign"));
        };
        *current_mk = mk_now;
        cur_starts.clone_from(&buf.start);
        Ok(true)
    } else {
        let seq = &mut ds.machine_seq[cur_machine];
        seq.push(node);
        seq[old_pos..].rotate_right(1);
        ds.node_machine[node] = cur_machine;
        ds.node_pt[node] = cur_pt;
        Ok(false)
    }
}

/// Longest path from the end of each node to the end of the schedule.
fn tails(ds: &DisjSchedule, buf: &EvalBuf) -> Vec<u32> {
    let mut order: Vec<usize> = (0..ds.n).collect();
    order.sort_unstable_by(|&a, &b| buf.start[b].cmp(&buf.start[a]));
    let mut tails = vec![0u32; ds.n];
    for &nd in &order {
        let mut after = 0u32;
        for succ in [ds.job_succ[nd], buf.machine_succ[nd]] {
            if succ != NONE {
                after = after.max(ds.node_pt[succ].saturating_add(tails[succ]));
            }
        }
        tails[nd] = after;
    }
    tails
}

/// Up to six distinct insertion indexes on `new_machine`: around the job predecessor's end, around
/// the current start, and both ends.
fn insert_positions(
    ds: &DisjSchedule,
    starts: &[u32],
    node: usize,
    new_machine: usize,
    job_pred: &[usize],
) -> Vec<usize> {
    let seq = &ds.machine_seq[new_machine];
    let len = seq.len();
    if len == 0 {
        return vec![0];
    }
    let jp = job_pred[node];
    let job_pred_end = if jp != NONE {
        starts[jp].saturating_add(ds.node_pt[jp])
    } else {
        0
    };
    let cur_start = starts[node];
    let pos_after_jp = seq
        .iter()
        .position(|&nd| starts[nd] > job_pred_end)
        .unwrap_or(len);
    let pos_by_cur = seq
        .iter()
        .position(|&nd| starts[nd] >= cur_start)
        .unwrap_or(len);
    let mut out: Vec<usize> = Vec::with_capacity(6);
    for p in [
        pos_after_jp,
        pos_after_jp.saturating_sub(1),
        pos_by_cur,
        pos_by_cur.saturating_sub(1),
        0,
        len,
    ] {
        if p <= len && !out.contains(&p) {
            out.push(p);
        }
    }
    out
}
}

pub mod features {
//! Schedule-derived guidance: job bias, machine penalty and route preference, and the elite of such
//! triples.
use super::types::*;
use anyhow::{anyhow, Result};
use std::hash::{Hash, Hasher};
use tig_challenges::job_scheduling::*;

/// Guidance derived from one reference schedule.
#[derive(Clone)]
pub struct Features {
    pub job_bias: Vec<f64>,
    pub machine_penalty: Vec<f64>,
    pub route_pref: RoutePref,
}

impl Features {
    pub fn from_solution(pre: &Pre, challenge: &Challenge, sol: &Solution) -> Result<Features> {
        Ok(Features {
            job_bias: job_bias(pre, sol)?,
            machine_penalty: machine_penalty(pre, sol, challenge.num_machines)?,
            route_pref: route_pref(pre, sol, challenge)?,
        })
    }
}

/// Normalised job completion raised to a power that grows with flexibility.
fn job_bias(pre: &Pre, sol: &Solution) -> Result<Vec<f64>> {
    let num_jobs = pre.job_ops_len.len();
    let mut completion = vec![0u32; num_jobs];
    let mut makespan = 0u32;
    for job in 0..num_jobs {
        let product = pre.job_products[job];
        let mut end_j = 0u32;
        for (op_idx, &(m, st)) in sol.job_schedule[job].iter().enumerate() {
            let pt = pre.product_ops[product][op_idx]
                .pt_on(m)
                .ok_or_else(|| anyhow!("Missing pt in bias calc"))?;
            end_j = end_j.max(st.saturating_add(pt));
        }
        completion[job] = end_j;
        makespan = makespan.max(end_j);
    }
    let denom = (makespan as f64).max(1.0);
    let exp = 3.0 + 1.2 * pre.high_flex + 0.6 * pre.jobshopness;
    Ok(completion
        .into_iter()
        .map(|c| ((c as f64) / denom).powf(exp).clamp(0.0, 1.0))
        .collect())
}

/// Mix of the normalised machine end time and load, raised to a power that grows with flexibility.
fn machine_penalty(pre: &Pre, sol: &Solution, num_machines: usize) -> Result<Vec<f64>> {
    let num_jobs = pre.job_ops_len.len();
    let mut m_end = vec![0u32; num_machines];
    let mut m_sum = vec![0u64; num_machines];
    let mut makespan = 0u32;
    for job in 0..num_jobs {
        let product = pre.job_products[job];
        for (op_idx, &(m, st)) in sol.job_schedule[job].iter().enumerate() {
            let pt = pre.product_ops[product][op_idx]
                .pt_on(m)
                .ok_or_else(|| anyhow!("Missing pt in machine penalty"))?;
            let end = st.saturating_add(pt);
            if end > m_end[m] {
                m_end[m] = end;
            }
            m_sum[m] = m_sum[m].saturating_add(pt as u64);
            makespan = makespan.max(end);
        }
    }
    let mk = (makespan as f64).max(1.0);
    let total: u64 = m_sum.iter().copied().sum();
    let avg = ((total as f64) / (num_machines as f64).max(1.0)).max(1.0);
    let use_load = pre.high_flex > 0.35 || pre.jobshopness > 0.45;
    let w_load = if use_load {
        (0.20 + 0.30 * pre.high_flex + 0.12 * pre.jobshopness).clamp(0.18, 0.58)
    } else {
        0.0
    };
    let w_end = 1.0 - w_load;
    let exp = 2.0 + 1.2 * pre.high_flex + 0.55 * pre.jobshopness;
    let mut mp = vec![0.0f64; num_machines];
    for m in 0..num_machines {
        let endn = (m_end[m] as f64 / mk).clamp(0.0, 1.0);
        let loadr = ((m_sum[m] as f64) / avg).max(0.0);
        let loadn = (loadr / (loadr + 1.0)).clamp(0.0, 1.0);
        let mix = (w_end * endn + w_load * loadn).clamp(0.0, 1.0);
        mp[m] = mix.powf(exp).clamp(0.0, 1.0);
    }
    Ok(mp)
}

/// The two machines most used by each product operation, weighted by their share of the product's
/// jobs.
fn route_pref(pre: &Pre, sol: &Solution, challenge: &Challenge) -> Result<RoutePref> {
    let nm = challenge.num_machines;
    let np = challenge.product_processing_times.len();
    let ops_len: Vec<usize> = (0..np)
        .map(|p| challenge.product_processing_times[p].len())
        .collect();
    let mut counts: Vec<Vec<u16>> = (0..np)
        .map(|p| vec![0u16; ops_len[p].saturating_mul(nm)])
        .collect();
    for job in 0..challenge.num_jobs {
        let product = pre.job_products[job];
        let ol = ops_len[product];
        for (op_idx, &(m, _st)) in sol.job_schedule[job].iter().enumerate() {
            if op_idx >= ol || m >= nm {
                continue;
            }
            let idx = op_idx * nm + m;
            counts[product][idx] = counts[product][idx].saturating_add(1);
        }
    }
    let mut rp: RoutePref = Vec::with_capacity(np);
    for p in 0..np {
        let denom = (challenge.jobs_per_product[p].max(1) as u32).max(1);
        let mut v: Vec<OpRoute> = Vec::with_capacity(ops_len[p]);
        for op_idx in 0..ops_len[p] {
            let base = op_idx * nm;
            let (mut best_m, mut best_c, mut second_m, mut second_c) =
                (0usize, 0u16, 0usize, 0u16);
            for m in 0..nm {
                let c = counts[p][base + m];
                if c > best_c {
                    second_c = best_c;
                    second_m = best_m;
                    best_c = c;
                    best_m = m;
                } else if c > second_c && m != best_m {
                    second_c = c;
                    second_m = m;
                }
            }
            let weight = |c: u16| {
                (((c as u32).saturating_mul(255)).saturating_add(denom / 2) / denom).min(255)
                    as u8
            };
            v.push(OpRoute {
                best_m: best_m.min(255) as u8,
                best_w: weight(best_c),
                second_m: second_m.min(255) as u8,
                second_w: weight(second_c),
            });
        }
        rp.push(v);
    }
    Ok(rp)
}

/// Guidance triple with the makespan of the schedule it was derived from.
#[derive(Clone)]
pub struct Elite {
    pub features: Features,
    pub score: u32,
}

/// Hash over the machine choices of a schedule.
pub fn machine_sig(sol: &Solution) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for (j, ops) in sol.job_schedule.iter().enumerate() {
        for (o, (m, _t)) in ops.iter().enumerate() {
            (j, o, *m).hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// Hash over the machine choices and start times of a schedule.
pub fn exact_sig(sol: &Solution) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for (j, ops) in sol.job_schedule.iter().enumerate() {
        for (o, (m, t)) in ops.iter().enumerate() {
            (j, o, *m, *t).hash(&mut hasher);
        }
    }
    hasher.finish()
}

fn elite_sig(e: &Elite) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for (i, v) in e.features.job_bias.iter().enumerate() {
        (i as u32).hash(&mut hasher);
        v.to_bits().hash(&mut hasher);
    }
    for (i, v) in e.features.machine_penalty.iter().enumerate() {
        (i as u32).hash(&mut hasher);
        v.to_bits().hash(&mut hasher);
    }
    for (p, ops) in e.features.route_pref.iter().enumerate() {
        for (o, r) in ops.iter().enumerate() {
            (p as u32).hash(&mut hasher);
            (o as u32).hash(&mut hasher);
            r.best_m.hash(&mut hasher);
            r.second_m.hash(&mut hasher);
            r.best_w.hash(&mut hasher);
            r.second_w.hash(&mut hasher);
        }
    }
    hasher.finish()
}

#[inline]
fn dist(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Keeps at most `cap` members: the best, then the (score, distance-to-best) skyline by
/// farthest-first, then the rest by farthest-first; the result is sorted by score.
pub fn normalize_elite(elite: &mut Vec<Elite>, cap: usize) {
    if elite.is_empty() {
        return;
    }
    if elite.len() <= 1 {
        elite.truncate(cap);
        return;
    }
    let mut view: Vec<(u32, u64, usize)> = elite
        .iter()
        .enumerate()
        .map(|(idx, e)| (e.score, elite_sig(e), idx))
        .collect();
    view.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let best_sig = view[0].1;
    let cand: Vec<(u32, u32, u64, usize)> = view
        .iter()
        .map(|(s, sg, idx)| (*s, dist(*sg, best_sig), *sg, *idx))
        .collect();

    let mut keep = vec![true; cand.len()];
    for i in 0..cand.len() {
        if !keep[i] {
            continue;
        }
        for j in 0..cand.len() {
            if i == j || !keep[i] {
                continue;
            }
            let (si, di, _, _) = cand[i];
            let (sj, dj, _, _) = cand[j];
            let no_worse = sj <= si && dj >= di;
            let strictly_better = sj < si || dj > di;
            if no_worse && strictly_better {
                keep[i] = false;
            }
        }
    }
    let mut skyline: Vec<(u32, u64, usize)> = Vec::new();
    for (i, k) in keep.iter().copied().enumerate() {
        if k {
            let (s, _d, sg, idx) = cand[i];
            skyline.push((s, sg, idx));
        }
    }
    skyline.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let mut selected: Vec<(u32, u64, usize)> = vec![view[0]];
    farthest_first(&mut selected, &skyline, cap, elite.len());
    farthest_first(&mut selected, &view, cap, elite.len());

    let mut new_elite: Vec<Elite> = selected
        .into_iter()
        .map(|(_s, _sg, idx)| elite[idx].clone())
        .collect();
    new_elite.sort_by_key(|e| e.score);
    new_elite.truncate(cap);
    *elite = new_elite;
}

/// Adds members of `pool` to `selected`, each time the one farthest from the selection (ties: lower
/// score, lower signature).
fn farthest_first(
    selected: &mut Vec<(u32, u64, usize)>,
    pool: &[(u32, u64, usize)],
    cap: usize,
    total: usize,
) {
    while selected.len() < cap && selected.len() < total {
        let mut best_pick: Option<(u32, u64, usize, u32)> = None;
        for &(s, sg, idx) in pool {
            if selected.iter().any(|x| x.2 == idx) {
                continue;
            }
            let mut md = u32::MAX;
            for &(_ss, ssg, _ii) in selected.iter() {
                md = md.min(dist(sg, ssg));
            }
            match best_pick {
                None => best_pick = Some((s, sg, idx, md)),
                Some((bs, bsg, _bidx, bmd)) => {
                    if md > bmd || (md == bmd && (s < bs || (s == bs && sg < bsg))) {
                        best_pick = Some((s, sg, idx, md));
                    }
                }
            }
        }
        match best_pick {
            Some((s, sg, idx, _)) => selected.push((s, sg, idx)),
            None => break,
        }
    }
}

/// Offers `cand` to the elite: accepted outright below capacity, otherwise only when it beats the
/// worst member.
pub fn maybe_add_elite(elite: &mut Vec<Elite>, cand: Elite, cap: usize) {
    if elite.is_empty() {
        elite.push(cand);
        return;
    }
    if elite.len() < cap {
        elite.push(cand);
        normalize_elite(elite, cap);
        return;
    }
    normalize_elite(elite, cap);
    let worst = elite.last().map(|e| e.score).unwrap_or(u32::MAX);
    if cand.score < worst {
        match elite.last_mut() {
            Some(last) => *last = cand,
            None => elite.push(cand),
        }
        normalize_elite(elite, cap);
    }
}
}

pub mod construct {
//! Event-driven construction: at each decision time, every ready job scores its eligible idle
//! machines under a dispatching rule, machine contention adjusts the scores, and one (job, machine)
//! pair is scheduled.
use super::disjunctive::push_top_k;
use super::types::*;
use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use tig_challenges::job_scheduling::*;

/// Candidates kept per machine when the choice is greedy (k = 0).
const GREEDY_CAP_PER_MACHINE: usize = 12;
/// Candidates kept per machine beyond k when the choice is randomised.
const CAP_PER_MACHINE_EXTRA: usize = 6;
/// Top candidates that take part in the weighted draw.
const WEIGHTED_DRAW: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    Adaptive,
    BnHeavy,
    EndTight,
    CriticalPath,
    MostWork,
    LeastFlex,
    Regret,
    ShortestProc,
    FlexBalance,
}

impl Rule {
    pub const ALL: [Rule; 9] = [
        Rule::Adaptive,
        Rule::BnHeavy,
        Rule::EndTight,
        Rule::CriticalPath,
        Rule::MostWork,
        Rule::LeastFlex,
        Rule::Regret,
        Rule::ShortestProc,
        Rule::FlexBalance,
    ];

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

/// Guidance learned from reference schedules; `machine_penalty` and `route_pref` are optional.
pub struct Guide<'a> {
    pub job_bias: &'a [f64],
    pub machine_penalty: Option<&'a [f64]>,
    pub route_pref: Option<&'a RoutePref>,
}

/// Inputs of one construction.
#[derive(Clone, Copy)]
struct Inputs<'a> {
    challenge: &'a Challenge,
    pre: &'a Pre,
    rule: Rule,
    k: usize,
    target_mk: u32,
    job_bias: &'a [f64],
    machine_penalty: &'a [f64],
    route_pref: Option<&'a RoutePref>,
}

/// Unguided construction: greedy when `k == 0`, otherwise a weighted draw among the top-k.
pub fn construct(
    challenge: &Challenge,
    pre: &Pre,
    rule: Rule,
    k: usize,
    target: Option<u32>,
    rng: &mut SmallRng,
) -> Result<(Solution, u32)> {
    let inputs = Inputs {
        challenge,
        pre,
        rule,
        k,
        target_mk: target.unwrap_or(0),
        job_bias: &[],
        machine_penalty: &[],
        route_pref: None,
    };
    match target {
        None => run::<false, false, false, false>(&inputs, rng),
        Some(_) => run::<true, false, false, false>(&inputs, rng),
    }
}

/// Guided construction with a target makespan.
pub fn construct_guided(
    challenge: &Challenge,
    pre: &Pre,
    rule: Rule,
    k: usize,
    target: u32,
    rng: &mut SmallRng,
    guide: &Guide,
) -> Result<(Solution, u32)> {
    let inputs = Inputs {
        challenge,
        pre,
        rule,
        k,
        target_mk: target,
        job_bias: guide.job_bias,
        machine_penalty: guide.machine_penalty.unwrap_or(&[]),
        route_pref: guide.route_pref,
    };
    match (guide.machine_penalty.is_some(), guide.route_pref.is_some()) {
        (true, true) => run::<true, true, true, true>(&inputs, rng),
        (true, false) => run::<true, true, true, false>(&inputs, rng),
        (false, true) => run::<true, true, false, true>(&inputs, rng),
        (false, false) => run::<true, true, false, false>(&inputs, rng),
    }
}

/// Per-job terms that depend only on the job's next operation: the raw remaining minimum work and
/// saturated (x / (1 + x)) versions of the remaining work, density, next-operation and flexibility
/// terms.
#[derive(Clone, Copy, Default)]
struct JobTerms {
    rem_min_raw: u64,
    rem_min_u: f64,
    rem_avg_u: f64,
    bn_u: f64,
    dens_u: f64,
    next_u: f64,
    flex_inv: f64,
    flex_u: f64,
}

/// Normalisation scales of the instance.
struct Scales {
    avg_op_min: f64,
    horizon: f64,
    time_scale: f64,
    max_job_avg_work: f64,
    max_job_bn: f64,
    avg_machine_load: f64,
    flex_factor_nonneg: f64,
    bn_focus_u: f64,
    slack: f64,
    flex_regime: f64,
}

impl Scales {
    fn new(pre: &Pre) -> Scales {
        Scales {
            avg_op_min: pre.avg_op_min.max(1.0),
            horizon: pre.horizon.max(1.0),
            time_scale: pre.time_scale.max(1.0),
            max_job_avg_work: pre.max_job_avg_work.max(1e-9),
            max_job_bn: pre.max_job_bn.max(1e-9),
            avg_machine_load: pre.avg_machine_load.max(1e-9),
            flex_factor_nonneg: pre.flex_factor.max(0.0),
            bn_focus_u: if pre.bn_focus <= 0.0 {
                0.0
            } else {
                pre.bn_focus / (1.0 + pre.bn_focus)
            },
            slack: (0.70 * pre.avg_op_min).max(1.0),
            flex_regime: (pre.high_flex + pre.jobshopness).clamp(0.0, 1.5),
        }
    }

    fn job_terms(
        &self,
        pre: &Pre,
        product: usize,
        op_idx: usize,
        ops_rem: usize,
        op: &OpInfo,
    ) -> JobTerms {
        let rem_min_raw = pre.product_suf_min[product][op_idx] as u64;
        let rem_min = rem_min_raw as f64;
        let rem_min_n = rem_min / self.horizon;
        let rem_avg_n = pre.product_suf_avg[product][op_idx] / self.max_job_avg_work;
        let bn_n = pre.product_suf_bn[product][op_idx] / self.max_job_bn;
        let density_n =
            ((rem_min / (ops_rem as f64).max(1.0)) / self.avg_op_min).clamp(0.0, 4.0);
        let next_min_n = (pre.product_next_min[product][op_idx] as f64) / self.horizon;
        let next_term_raw = (0.55 * next_min_n
            + 0.45 * pre.product_next_flex_inv[product][op_idx])
            * (1.0 + 0.30 * density_n * pre.high_flex);
        let flex_inv = 1.0 / (op.flex as f64).max(1.0);
        let flex_term = flex_inv * self.flex_factor_nonneg;
        JobTerms {
            rem_min_raw,
            rem_min_u: sat(rem_min_n),
            rem_avg_u: sat(rem_avg_n),
            bn_u: sat(bn_n),
            dens_u: sat(density_n),
            next_u: sat(next_term_raw),
            flex_inv,
            flex_u: sat(flex_term),
        }
    }
}

/// Ready jobs and idle machines at the current time, with the pending releases of both.
struct Frontier {
    ready: Vec<usize>,
    ready_pos: Vec<usize>,
    in_ready: Vec<bool>,
    ready_heap: BinaryHeap<Reverse<(u32, usize)>>,
    idle: Vec<usize>,
    idle_pos: Vec<usize>,
    machine_heap: BinaryHeap<Reverse<(u32, usize, u32)>>,
    /// Bumped at each assignment; a heap entry with an older generation is stale.
    machine_gen: Vec<u32>,
}

impl Frontier {
    fn new(num_jobs: usize, num_machines: usize) -> Frontier {
        Frontier {
            ready: Vec::with_capacity(num_jobs),
            ready_pos: vec![NONE; num_jobs],
            in_ready: vec![false; num_jobs],
            ready_heap: BinaryHeap::new(),
            idle: (0..num_machines).collect(),
            idle_pos: (0..num_machines).collect(),
            machine_heap: BinaryHeap::new(),
            machine_gen: vec![0u32; num_machines],
        }
    }

    #[inline]
    fn push_ready(&mut self, job: usize) {
        self.in_ready[job] = true;
        self.ready_pos[job] = self.ready.len();
        self.ready.push(job);
    }

    #[inline]
    fn remove_ready(&mut self, job: usize) {
        self.in_ready[job] = false;
        let pos = self.ready_pos[job];
        if pos < self.ready.len() && self.ready[pos] == job {
            self.ready.swap_remove(pos);
            if pos < self.ready.len() {
                let moved = self.ready[pos];
                self.ready_pos[moved] = pos;
            }
        }
        self.ready_pos[job] = NONE;
    }

    #[inline]
    fn push_idle(&mut self, m: usize) {
        self.idle_pos[m] = self.idle.len();
        self.idle.push(m);
    }

    #[inline]
    fn remove_idle(&mut self, m: usize) {
        let pos = self.idle_pos[m];
        if pos < self.idle.len() {
            self.idle.swap_remove(pos);
            if pos < self.idle.len() {
                let moved = self.idle[pos];
                self.idle_pos[moved] = pos;
            }
        }
        self.idle_pos[m] = NONE;
    }

    /// Releases the machines and jobs due by `time`.
    fn release(
        &mut self,
        time: u32,
        job_next_op: &[usize],
        job_ops_len: &[usize],
        job_ready_time: &[u32],
        machine_avail: &[u32],
    ) {
        while let Some(Reverse((t, m, g))) = self.machine_heap.peek().copied() {
            if t > time {
                break;
            }
            self.machine_heap.pop();
            if g != self.machine_gen[m] || machine_avail[m] != t || self.idle_pos[m] != NONE {
                continue;
            }
            self.push_idle(m);
        }
        while let Some(Reverse((t, j))) = self.ready_heap.peek().copied() {
            if t > time {
                break;
            }
            self.ready_heap.pop();
            if self.in_ready[j] || job_next_op[j] >= job_ops_len[j] || job_ready_time[j] != t {
                continue;
            }
            self.push_ready(j);
        }
    }

    /// Moves `time` to the next release after it, dropping stale heap entries; None when nothing is
    /// pending.
    fn advance(
        &mut self,
        time: &mut u32,
        job_next_op: &[usize],
        job_ops_len: &[usize],
        job_ready_time: &[u32],
        machine_avail: &[u32],
    ) -> Option<u32> {
        self.release(
            *time,
            job_next_op,
            job_ops_len,
            job_ready_time,
            machine_avail,
        );
        let next_machine_time = loop {
            let Some(Reverse((t, m, g))) = self.machine_heap.peek().copied() else {
                break None;
            };
            if t <= *time
                || g != self.machine_gen[m]
                || machine_avail[m] != t
                || self.idle_pos[m] != NONE
            {
                self.machine_heap.pop();
                continue;
            }
            break Some(t);
        };
        let next_ready_time = loop {
            let Some(Reverse((t, j))) = self.ready_heap.peek().copied() else {
                break None;
            };
            if t <= *time
                || self.in_ready[j]
                || job_next_op[j] >= job_ops_len[j]
                || job_ready_time[j] != t
            {
                self.ready_heap.pop();
                continue;
            }
            break Some(t);
        };
        let nt = match (next_machine_time, next_ready_time) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => return None,
        };
        *time = nt;
        self.release(nt, job_next_op, job_ops_len, job_ready_time, machine_avail);
        Some(nt)
    }
}

#[inline]
fn route_bonus(rp: &RoutePref, product: usize, op_idx: usize, machine: usize) -> f64 {
    let r = rp[product][op_idx];
    let mu = machine.min(255) as u8;
    if mu == r.best_m {
        (r.best_w as f64) / 255.0
    } else if mu == r.second_m {
        (r.second_w as f64) / 255.0
    } else {
        0.0
    }
}

/// Earliest end over the eligible machines, the second earliest, and how many machines reach the
/// earliest in total and while idle.
#[inline]
fn best_second_and_counts(
    time: u32,
    machine_avail: &[u32],
    op: &OpInfo,
) -> (u32, u32, usize, usize) {
    let mut best = INF;
    let mut second = INF;
    let mut cnt_best = 0usize;
    let mut cnt_best_idle = 0usize;
    for &(m, pt) in &op.machines {
        let end = time.max(machine_avail[m]).saturating_add(pt);
        if end < best {
            second = best;
            best = end;
            cnt_best = 1;
            cnt_best_idle = if machine_avail[m] <= time { 1 } else { 0 };
        } else if end == best {
            cnt_best += 1;
            if machine_avail[m] <= time {
                cnt_best_idle += 1;
            }
        } else if end < second {
            second = end;
        }
    }
    if cnt_best > 1 {
        second = best;
    }
    (best, second, cnt_best.max(1), cnt_best_idle)
}

/// Draws among the leading candidates with weights quadratic in the score margin over the last one.
#[inline]
fn choose_weighted(rng: &mut SmallRng, top: &[Cand]) -> Cand {
    if top.len() <= 1 {
        return top[0];
    }
    let min_s = top.last().unwrap().score;
    let n = top.len().min(WEIGHTED_DRAW);
    let mut w = [0.0f64; WEIGHTED_DRAW];
    let mut sum = 0.0f64;
    for i in 0..n {
        let d = (top[i].score - min_s) + 1e-9;
        let wi = d * d;
        w[i] = wi;
        sum += wi;
    }
    if !(sum > 0.0) {
        return top[rng.gen_range(0..top.len())];
    }
    let mut r = rng.gen::<f64>() * sum;
    for i in 0..n {
        r -= w[i];
        if r <= 0.0 {
            return top[i];
        }
    }
    top[n - 1]
}

/// Among the top candidates, keeps those with the highest route bonus, then draws.
fn choose_routed(
    rng: &mut SmallRng,
    top: &mut Vec<Cand>,
    rp: &RoutePref,
    job_product: &[usize],
    job_next_op: &[usize],
    job_op: &[Option<&OpInfo>],
) -> Cand {
    let bonus = |c: &Cand| route_bonus(rp, job_product[c.job], job_next_op[c.job], c.machine);
    let mut best_rb: Option<f64> = None;
    let mut best_idx = 0usize;
    let mut keep_cnt = 0usize;
    for (i, c) in top.iter().enumerate() {
        if job_op[c.job].is_none() {
            continue;
        }
        let rb = bonus(c);
        match best_rb {
            None => {
                best_rb = Some(rb);
                best_idx = i;
                keep_cnt = 1;
            }
            Some(b) => {
                if rb > b {
                    best_rb = Some(rb);
                    best_idx = i;
                    keep_cnt = 1;
                } else if rb == b {
                    keep_cnt += 1;
                }
            }
        }
    }
    if keep_cnt == 0 {
        choose_weighted(rng, top)
    } else if keep_cnt == 1 {
        top[best_idx]
    } else {
        let best_rb = best_rb.unwrap();
        let mut write = 0usize;
        for i in 0..top.len() {
            let c = top[i];
            if job_op[c.job].is_none() {
                continue;
            }
            if bonus(&c) == best_rb {
                top[write] = c;
                write += 1;
            }
        }
        top.truncate(write);
        choose_weighted(rng, top)
    }
}

fn run<
    'a,
    const HAS_TARGET: bool,
    const HAS_JOB_BIAS: bool,
    const HAS_MACHINE_PENALTY: bool,
    const USE_ROUTE_PREF: bool,
>(
    inputs: &Inputs<'a>,
    rng: &mut SmallRng,
) -> Result<(Solution, u32)> {
    let Inputs {
        challenge,
        pre,
        rule,
        k,
        target_mk,
        job_bias,
        machine_penalty,
        route_pref,
    } = *inputs;
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;
    let sc = Scales::new(pre);
    let job_product = &pre.job_products;

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
    let mut raw_by_machine: Vec<Vec<RawCand>> =
        (0..num_machines).map(|_| Vec::with_capacity(12)).collect();
    // Next operation of each job, None once the job is complete.
    let mut job_op: Vec<Option<&'a OpInfo>> = vec![None; num_jobs];
    let mut job_terms: Vec<JobTerms> = vec![JobTerms::default(); num_jobs];
    let mut fr = Frontier::new(num_jobs, num_machines);
    let mut touched_machines: Vec<usize> = Vec::with_capacity(num_machines);
    let mut touched_gen: Vec<u32> = vec![0u32; num_machines];
    let mut cur_gen: u32 = 0;
    let mut top: Vec<Cand> = Vec::with_capacity(k);

    for j in 0..num_jobs {
        let job_len = pre.job_ops_len[j];
        if job_len == 0 {
            continue;
        }
        let product = job_product[j];
        let op = &pre.product_ops[product][0];
        job_op[j] = Some(op);
        job_terms[j] = sc.job_terms(pre, product, 0, job_len, op);
        fr.push_ready(j);
    }

    let cap_per_machine = if k == 0 {
        GREEDY_CAP_PER_MACHINE
    } else {
        (k + CAP_PER_MACHINE_EXTRA).min(GREEDY_CAP_PER_MACHINE)
    };
    let conflict_scale = (0.90 + 0.40 * pre.flex_factor).clamp(0.85, 1.75);

    while remaining_ops > 0 {
        if fr.idle.is_empty() {
            fr.advance(
                &mut time,
                &job_next_op,
                &pre.job_ops_len,
                &job_ready_time,
                &machine_avail,
            )
            .ok_or_else(|| anyhow!("Stalled"))?;
            continue;
        }

        touched_machines.clear();
        cur_gen = cur_gen.wrapping_add(1);
        if cur_gen == 0 {
            touched_gen.fill(0);
            cur_gen = 1;
        }
        let progress = 1.0 - (remaining_ops as f64) / (pre.total_ops as f64).max(1.0);
        let prog_gate = sat(progress);

        // Every ready job scores its idle eligible machines that finish within reach of its
        // earliest end.
        for &job in &fr.ready {
            let Some(op) = job_op[job] else { continue };
            let op_flex = op.flex;
            if op_flex == 0 || op.machines.is_empty() || op.min_pt >= INF {
                continue;
            }
            let (best_end, second_end, best_cnt_total, best_cnt_idle) =
                best_second_and_counts(time, &machine_avail, op);
            if best_end >= INF || best_cnt_idle == 0 {
                continue;
            }
            let jt = job_terms[job];
            let jb = if HAS_JOB_BIAS { job_bias[job] } else { 0.0 };
            let flow_term =
                pre.flow_w * pre.job_flow_pref[job] * (0.65 + 0.70 * (1.0 - progress));
            let bias = jb + flow_term;
            let flex_inv = jt.flex_inv;
            let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
            let regret = if second_end >= INF {
                pre.avg_op_min * 2.6
            } else {
                (second_end - best_end) as f64
            };
            let regn = (regret / sc.avg_op_min).clamp(0.0, 6.0);
            let rigidity = (0.60 * flex_inv + 0.40 * scarcity_urg).clamp(0.0, 2.5);
            // Only machines finishing within `near_band` of the earliest end are candidates.
            let exact_only = op_flex < 2
                || sc.flex_regime < 0.34
                || (k == 0 && progress < 0.10 && best_cnt_total > 1);
            let near_band = if exact_only {
                0u32
            } else {
                let base = ((pre.avg_op_min
                    * (0.35 + 0.30 * pre.high_flex + 0.28 * pre.jobshopness + 0.16 * progress))
                    .round() as u32)
                    .max(1);
                let regret_cap = if second_end >= INF {
                    base.max(op.min_pt / 2)
                } else {
                    second_end.saturating_sub(best_end).min(base.max(1))
                };
                regret_cap.max(1)
            };
            let allow_end = best_end.saturating_add(near_band);
            let detour_mult = if best_cnt_total <= 1 {
                0.72
            } else if best_cnt_total == 2 {
                0.86
            } else {
                1.0
            };
            let slack_u = if HAS_TARGET {
                let lb = (time as u64).saturating_add(jt.rem_min_raw);
                let slack = (target_mk as i64) - (lb as i64);
                let pos = (slack.max(0) as f64) / sc.slack;
                let neg = ((-slack).max(0) as f64) / sc.slack;
                (1.0 / (1.0 + pos)).clamp(0.0, 1.0) + (0.35 * neg).min(3.0)
            } else {
                0.0
            };
            let reg_u = sat(regn);
            let end_u = sat((best_end as f64) / sc.time_scale);
            let sat_scarcity = sat(scarcity_urg);
            let scarce_slack = scarcity_urg * slack_u;
            let scarce_reg = scarcity_urg * reg_u;

            for &(m, pt) in &op.machines {
                if fr.idle_pos[m] == NONE {
                    continue;
                }
                let end = time.saturating_add(pt);
                if end > allow_end {
                    continue;
                }
                if touched_gen[m] != cur_gen {
                    touched_gen[m] = cur_gen;
                    touched_machines.push(m);
                    demand[m] = 0;
                    raw_by_machine[m].clear();
                }
                demand[m] = demand[m].saturating_add(1);
                let mp = if HAS_MACHINE_PENALTY {
                    machine_penalty[m]
                } else {
                    0.0
                };
                let jitter = if k > 0 { rng.gen::<f64>() * 1e-9 } else { 0.0 };
                let load_n = machine_load[m] / sc.avg_machine_load;
                let proc_n = (pt as f64) / sc.avg_op_min;
                let mpen = mp.clamp(0.0, 1.0);
                let pop = pre.machine_best_pop[m].clamp(0.0, 1.2);
                let load_u = sat(load_n);
                let proc_u = sat(proc_n);
                let mpen_u = sat(mpen);
                let end_gap = end.saturating_sub(best_end);
                let end_gap_u = sat((end_gap as f64) / sc.avg_op_min);
                let preserve_pen = if op_flex >= 2 && sc.flex_regime > 0.35 {
                    (0.06 + 0.12 * sc.flex_regime.min(1.0) + 0.08 * progress)
                        * pop
                        * (1.0 - flex_inv)
                } else {
                    0.0
                };
                let detour_pen = if end_gap > 0 {
                    end_gap_u * (0.55 + 0.45 * rigidity + 0.18 * sat_scarcity) * detour_mult
                } else {
                    0.0
                };
                let detour_credit = if end_gap > 0 {
                    (0.08 + 0.10 * pre.jobshopness + 0.08 * pre.high_flex)
                        * (1.0 - load_u)
                        * (1.0 - pop.min(1.0))
                } else {
                    0.0
                };
                let scarce_machine_bonus = if op_flex >= 2 && sc.flex_regime > 0.35 {
                    (0.03 + 0.06 * progress) * (1.0 - pop.min(1.0)) * sat_scarcity
                } else {
                    0.0
                };
                let base_bias = bias + jitter;
                let base0 = match rule {
                    Rule::CriticalPath => {
                        let chain = jt.rem_min_u * (1.0 + jt.next_u);
                        let urgent = scarce_slack * (1.0 + scarce_reg * prog_gate);
                        chain + urgent + base_bias - end_u
                    }
                    Rule::MostWork => {
                        let work = jt.rem_avg_u * (1.0 + jt.dens_u);
                        let smooth = work * (1.0 + load_u);
                        smooth + base_bias - end_u
                    }
                    Rule::LeastFlex => {
                        let rigid = jt.flex_u * (1.0 + sat_scarcity);
                        rigid + jt.rem_min_u + jt.next_u + base_bias - end_u
                    }
                    Rule::ShortestProc => {
                        let short = 0.0 - proc_u;
                        short + jt.rem_min_u * (1.0 + jt.next_u) + sat_scarcity + base_bias
                            - end_u
                    }
                    Rule::Regret => {
                        let regret_focus = reg_u * (1.0 + sat_scarcity) * (1.0 + prog_gate);
                        regret_focus + jt.rem_min_u + jt.next_u + base_bias - end_u
                    }
                    Rule::EndTight => {
                        let tight = scarce_slack * (1.0 + scarce_reg);
                        let chain = jt.rem_min_u * (1.0 + prog_gate) * (1.0 + jt.next_u);
                        let penal =
                            end_u * (1.0 + prog_gate) + proc_u + mpen_u * pre.flex_factor;
                        chain + tight + base_bias - penal
                    }
                    Rule::BnHeavy => {
                        let bn_focus = jt.bn_u * (1.0 + jt.dens_u) * (1.0 + sc.bn_focus_u);
                        let chain = jt.rem_min_u * (1.0 + jt.next_u);
                        let penal = end_u
                            + proc_u
                            + load_u * pre.flex_factor
                            + mpen_u * pre.flex_factor;
                        bn_focus + chain + scarce_slack + reg_u + jt.flex_u + base_bias - penal
                    }
                    Rule::Adaptive => {
                        let js = pre.jobshopness;
                        let fl = 1.0 - js;
                        if js >= fl {
                            let hard = reg_u * (1.0 + scarce_reg)
                                + jt.flex_u
                                + jt.rem_min_u * (1.0 + jt.next_u);
                            hard + base_bias - (end_u + mpen_u * pre.flex_factor)
                        } else {
                            let flow =
                                jt.rem_avg_u * (1.0 + jt.dens_u) + (0.0 - proc_u) + slack_u;
                            flow + base_bias - (end_u + load_u * pre.flex_factor)
                        }
                    }
                    Rule::FlexBalance => {
                        let flexible = jt.flex_u * (1.0 + sat_scarcity);
                        let chain = (jt.rem_avg_u + jt.rem_min_u) * (1.0 + jt.next_u);
                        let penal =
                            end_u + load_u * pre.flex_factor + mpen_u * (1.0 + pre.flex_factor);
                        flexible + chain + base_bias - penal
                    }
                };
                let base =
                    base0 + scarce_machine_bonus + detour_credit - detour_pen - preserve_pen;
                push_top_k(
                    &mut raw_by_machine[m],
                    RawCand {
                        job,
                        machine: m,
                        pt,
                        base_score: base,
                        rigidity,
                        reg_n: regn,
                    },
                    cap_per_machine,
                    |c| c.base_score,
                );
            }
        }

        // Machine contention: candidates on a machine wanted by several jobs are boosted by their
        // rigidity and regret.
        let denom = (fr.idle.len() as f64).max(1.0);
        let conflict_w =
            (0.09 + 0.26 * pre.jobshopness + 0.11 * pre.high_flex + 0.16 * (1.0 - progress))
                .clamp(0.05, 0.45);
        let mut best: Option<Cand> = None;
        top.clear();
        if touched_machines.len() > 1 {
            touched_machines.sort_unstable();
        }
        for &m in &touched_machines {
            let dem = demand[m] as f64;
            if dem <= 0.0 || raw_by_machine[m].is_empty() {
                continue;
            }
            let dem_n = ((dem - 1.0) / denom).clamp(0.0, 2.5);
            let load_factor = machine_load[m] / sc.avg_machine_load;
            let cs = conflict_scale * (1.0 - 0.30 * load_factor).clamp(0.40, 1.60);
            for rc in &raw_by_machine[m] {
                let rig = rc.rigidity.clamp(0.0, 2.5);
                let regc = rc.reg_n.clamp(0.0, 4.5);
                let boost = conflict_w * cs * dem_n * (1.15 * rig + 0.85 * regc);
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
                    push_top_k(&mut top, c, k, |c| c.score);
                }
            }
        }

        let chosen = if k == 0 {
            match best {
                Some(c) => c,
                None => {
                    fr.advance(
                        &mut time,
                        &job_next_op,
                        &pre.job_ops_len,
                        &job_ready_time,
                        &machine_avail,
                    )
                    .ok_or_else(|| anyhow!("Stalled"))?;
                    continue;
                }
            }
        } else {
            if top.is_empty() {
                fr.advance(
                    &mut time,
                    &job_next_op,
                    &pre.job_ops_len,
                    &job_ready_time,
                    &machine_avail,
                )
                .ok_or_else(|| anyhow!("Stalled"))?;
                continue;
            }
            if USE_ROUTE_PREF {
                choose_routed(
                    rng,
                    &mut top,
                    route_pref.unwrap(),
                    job_product,
                    &job_next_op,
                    &job_op,
                )
            } else {
                choose_weighted(rng, &top)
            }
        };

        let (job, machine, pt) = (chosen.job, chosen.machine, chosen.pt);
        let product = job_product[job];
        let op = &pre.product_ops[product][job_next_op[job]];
        let end_time = time.saturating_add(pt);
        fr.remove_ready(job);
        fr.remove_idle(machine);
        job_schedule[job].push((machine, time));
        job_next_op[job] += 1;
        job_ready_time[job] = end_time;
        machine_avail[machine] = end_time;
        remaining_ops -= 1;

        if job_next_op[job] < pre.job_ops_len[job] {
            let new_op_idx = job_next_op[job];
            let next_op = &pre.product_ops[product][new_op_idx];
            job_op[job] = Some(next_op);
            job_terms[job] = sc.job_terms(
                pre,
                product,
                new_op_idx,
                pre.job_ops_len[job] - new_op_idx,
                next_op,
            );
            if end_time == time {
                fr.push_ready(job);
            } else {
                fr.ready_heap.push(Reverse((end_time, job)));
            }
        } else {
            job_op[job] = None;
        }

        fr.machine_gen[machine] = fr.machine_gen[machine].wrapping_add(1);
        if end_time == time {
            fr.push_idle(machine);
        } else {
            fr.machine_heap
                .push(Reverse((end_time, machine, fr.machine_gen[machine])));
        }

        // The scheduled operation no longer weighs on the load of its eligible machines.
        if op.min_pt < INF && op.flex > 0 && !op.machines.is_empty() {
            let delta = (op.min_pt as f64) / (op.flex as f64).max(1.0);
            if delta > 0.0 {
                for &(mm, _) in &op.machines {
                    let v = machine_load[mm] - delta;
                    machine_load[mm] = if v > 0.0 { v } else { 0.0 };
                }
            }
        }
    }

    let mk = machine_avail.into_iter().max().unwrap_or(0);
    Ok((Solution { job_schedule }, mk))
}
}

pub mod search {
//! Restart loop: rule selection by a bandit, guided constructions from an elite of reference
//! schedules, and local search on the promising ones.
use super::construct::{self, Guide, Rule};
use super::descent::block_descent;
use super::features::*;
use super::preprocess::build_pre;
use super::reassign::reassign_pass;
use super::types::*;
use crate::{seeded_hasher, HashMap};
use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::job_scheduling::*;

/// Best schedules kept for escapes and elite refreshes.
const TOP_KEEP: usize = 15;
/// Restarts that draw the rule from the three best initial rules before the bandit takes over.
const WARMUP_RESTARTS: usize = 35;
/// Restarts before another escape is attempted, after one ran.
const ESCAPE_COOLDOWN: usize = 14;
/// Restarts before another escape is attempted, after one was declined during a long stagnation.
const STALL_COOLDOWN: usize = 6;
/// Largest top-k of the weighted draw.
const K_MAX: usize = 6;
/// Cached local-search results before the cache is emptied.
const LS_CACHE_CAP: usize = 1024;

/// Local-search memo keyed by (exact schedule signature, parameters, kind).
type LsCache = HashMap<(u64, usize, usize, usize, u8), Option<(Solution, u32)>>;

/// Runs `iters` restarts of the construction and reports every improving schedule to
/// `save_solution`.
pub fn construct(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    iters: usize,
) -> Result<()> {
    let pre = build_pre(challenge)?;
    solve(challenge, save_solution, &pre, iters)
}

/// Best schedule so far, and the best makespan already handed to `save_solution`.
struct Incumbent {
    mk: u32,
    sol: Solution,
    published: u32,
}

impl Incumbent {
    /// Saves `sol` when it beats everything saved so far, without changing the incumbent.
    fn publish(
        &mut self,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        sol: &Solution,
        mk: u32,
    ) -> Result<()> {
        if mk < self.published {
            save_solution(sol)?;
            self.published = mk;
        }
        Ok(())
    }

    /// Adopts and saves `sol` when it beats the incumbent; reports whether it did.
    fn offer(
        &mut self,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        sol: &Solution,
        mk: u32,
    ) -> Result<bool> {
        if mk < self.mk {
            self.mk = mk;
            self.sol = sol.clone();
            self.publish(save_solution, sol, mk)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

/// Sorted list of the `cap` best (schedule, makespan) pairs.
fn push_top_solutions(top: &mut Vec<(Solution, u32)>, sol: &Solution, mk: u32, cap: usize) {
    let pos = top
        .binary_search_by_key(&mk, |(_, m)| *m)
        .unwrap_or_else(|e| e);
    top.insert(pos, (sol.clone(), mk));
    if top.len() > cap {
        top.truncate(cap);
    }
}

/// Tournament of two.
fn pick_elite_idx(rng: &mut SmallRng, elite: &[Elite]) -> usize {
    let len = elite.len();
    if len <= 1 {
        return 0;
    }
    let a = rng.gen_range(0..len);
    let b = rng.gen_range(0..len);
    if elite[a].score <= elite[b].score {
        a
    } else {
        b
    }
}

/// Tournament of two.
fn pick_top_idx(rng: &mut SmallRng, top: &[(Solution, u32)]) -> usize {
    let len = top.len();
    if len <= 1 {
        return 0;
    }
    let a = rng.gen_range(0..len);
    let b = rng.gen_range(0..len);
    if top[a].1 <= top[b].1 {
        a
    } else {
        b
    }
}

/// Among the first `scan` schedules of a non-empty `top`, the one whose machine choices differ most
/// from `best`; the first one on ties.
fn pick_diverse_top(best: &Solution, top: &[(Solution, u32)], scan: usize) -> usize {
    let sig_best = machine_sig(best);
    let mut best_idx = 0usize;
    let mut best_dist: u32 = 0;
    for i in 0..scan.min(top.len()) {
        let dist = (sig_best ^ machine_sig(&top[i].0)).count_ones();
        if i == 0 || dist > best_dist {
            best_idx = i;
            best_dist = dist;
        }
    }
    best_idx
}

/// Draws a rule with weight exp(-(best of rule - best seen) / margin) mixed with an exploration
/// term that decays with the rule's tries; the mix leans to exploration as the search stagnates.
fn choose_rule(
    rng: &mut SmallRng,
    rule_best: &[u32],
    rule_tries: &[u32],
    global_best: u32,
    margin: u32,
    stuck: usize,
    late: bool,
) -> Rule {
    let rules = &Rule::ALL;
    let mut best_seen = global_best;
    for &mk in rule_best {
        if mk < best_seen {
            best_seen = mk;
        }
    }
    let scale = (margin as f64).max(1.0);
    let s = ((stuck as f64) / 140.0).clamp(0.0, 1.0);
    let explore_mix = (0.10 + 0.55 * s).clamp(0.10, 0.65);
    let mut weights = [0.0f64; 9];
    let mut sum = 0.0;
    for (i, &r) in rules.iter().enumerate() {
        let mk = rule_best[r.index()];
        let t = rule_tries[r.index()].max(1) as f64;
        let delta = mk.saturating_sub(best_seen) as f64;
        let exploit = (-delta / scale).exp();
        let explore = (1.0 / t).sqrt();
        let mut ww = (1.0 - explore_mix) * exploit + explore_mix * explore;
        ww = ww.max(1e-6);
        if late {
            ww = ww.powf(1.18);
        }
        weights[i] = ww.max(0.0);
        sum += weights[i];
    }
    if !(sum > 0.0) {
        return rules[rng.gen_range(0..rules.len())];
    }
    let mut r = rng.gen::<f64>() * sum;
    for (i, &rule) in rules.iter().enumerate() {
        r -= weights[i];
        if r <= 0.0 {
            return rule;
        }
    }
    rules[rules.len() - 1]
}

fn cached_descent(
    pre: &Pre,
    challenge: &Challenge,
    sol: &Solution,
    p: (usize, usize, usize),
    cache: &mut LsCache,
) -> Result<Option<(Solution, u32)>> {
    let key = (exact_sig(sol), p.0, p.1, p.2, 0u8);
    if let Some(hit) = cache.get(&key) {
        return Ok(hit.clone());
    }
    let res = block_descent(pre, challenge, sol, p.0, p.1, p.2)?;
    if cache.len() >= LS_CACHE_CAP {
        cache.clear();
    }
    cache.insert(key, res.clone());
    Ok(res)
}

fn cached_reassign(
    pre: &Pre,
    challenge: &Challenge,
    sol: &Solution,
    cache: &mut LsCache,
) -> Result<Option<(Solution, u32)>> {
    let key = (exact_sig(sol), 0usize, 0usize, 0usize, 1u8);
    if let Some(hit) = cache.get(&key) {
        return Ok(hit.clone());
    }
    let res = reassign_pass(pre, challenge, sol)?;
    if cache.len() >= LS_CACHE_CAP {
        cache.clear();
    }
    cache.insert(key, res.clone());
    Ok(res)
}

/// Local search on a fresh construction, with a probability that depends on whether it improves the
/// incumbent, how close it is otherwise, and how long the search has stagnated. On flexible
/// instances the reassignment and the descent are chained, in one or both orders.
#[allow(clippy::too_many_arguments)]
fn intensify(
    pre: &Pre,
    challenge: &Challenge,
    rng: &mut SmallRng,
    sol: &Solution,
    mk: u32,
    best_mk: u32,
    target_margin: u32,
    stuck: usize,
    late: bool,
    cache: &mut LsCache,
) -> Result<Option<(Solution, u32)>> {
    let flex = (pre.high_flex + pre.jobshopness).clamp(0.0, 1.5);
    let near_best = mk <= best_mk.saturating_add((target_margin / 3).max(1));
    let very_near_best = mk <= best_mk.saturating_add((target_margin / 6).max(1));
    let do_ls = if mk < best_mk {
        late || stuck > 20 || flex >= 0.12 || rng.gen::<f64>() < 0.55
    } else if very_near_best && (late || stuck > 80) {
        rng.gen::<f64>() < (0.05 + 0.05 * flex).clamp(0.04, 0.11)
    } else if near_best && stuck > 140 {
        rng.gen::<f64>() < (0.035 + 0.045 * flex).clamp(0.03, 0.085)
    } else {
        false
    };
    if !do_ls {
        return Ok(None);
    }
    let params = if mk < best_mk {
        let bump = if flex > 0.60 { 1.0 } else { 0.0 };
        (38 + (6.0 * bump) as usize, 60 + (10.0 * bump) as usize, 12)
    } else if stuck > 180 {
        (30, 48, 10)
    } else {
        (24, 36, 8)
    };

    let chain_on = flex > 0.52 && (mk < best_mk || (very_near_best && (late || stuck > 110)));
    if chain_on {
        let mut best: Option<(Solution, u32)> = None;
        if let Some((s1, m1)) = cached_reassign(pre, challenge, sol, cache)? {
            if m1 < mk && best.as_ref().map_or(true, |b| m1 < b.1) {
                best = Some((s1.clone(), m1));
            }
            if let Some((s2, m2)) = cached_descent(pre, challenge, &s1, params, cache)? {
                if m2 < mk && best.as_ref().map_or(true, |b| m2 < b.1) {
                    best = Some((s2, m2));
                }
            }
        }
        if best.is_none() || flex < 0.85 {
            if let Some((s1, m1)) = cached_descent(pre, challenge, sol, params, cache)? {
                if m1 < mk && best.as_ref().map_or(true, |b| m1 < b.1) {
                    best = Some((s1.clone(), m1));
                }
                if m1 <= mk.saturating_add((target_margin / 8).max(1)) {
                    if let Some((s2, m2)) = cached_reassign(pre, challenge, &s1, cache)? {
                        if m2 < mk && best.as_ref().map_or(true, |b| m2 < b.1) {
                            best = Some((s2, m2));
                        }
                    }
                }
            }
        }
        return Ok(best);
    }

    if flex > 0.62 && near_best && late && rng.gen::<f64>() < 0.35 {
        if let Some(res) = cached_reassign(pre, challenge, sol, cache)? {
            return Ok(Some(res));
        }
    }
    cached_descent(pre, challenge, sol, params, cache)
}

/// Occasional descent from the top schedule least similar to the incumbent, once the search
/// stagnates.
#[allow(clippy::too_many_arguments)]
fn escape(
    pre: &Pre,
    challenge: &Challenge,
    rng: &mut SmallRng,
    top_solutions: &[(Solution, u32)],
    best_sol: &Solution,
    stuck: usize,
    flex01: f64,
    cache: &mut LsCache,
) -> Result<Option<(Solution, u32)>> {
    if top_solutions.is_empty() || stuck < 60 {
        return Ok(None);
    }
    let p = (0.040 + 0.060 * flex01 + 0.040 * ((stuck as f64) / 160.0).clamp(0.0, 1.0))
        .clamp(0.04, 0.14);
    if rng.gen::<f64>() >= p {
        return Ok(None);
    }
    let idx = pick_diverse_top(best_sol, top_solutions, top_solutions.len().min(TOP_KEEP));
    let bump = if flex01 > 0.55 { 1.0 } else { 0.0 };
    let params = (
        (34.0 + 8.0 * bump) as usize,
        (56.0 + 10.0 * bump) as usize,
        (10.0 + 2.0 * bump) as usize,
    );
    cached_descent(pre, challenge, &top_solutions[idx].0, params, cache)
}

/// Guidance of a schedule, computed once per distinct exact signature.
fn features_of<'a>(
    pre: &Pre,
    challenge: &Challenge,
    sol: &Solution,
    cache: &'a mut HashMap<u64, Features>,
) -> Result<&'a Features> {
    let sig = exact_sig(sol);
    if !cache.contains_key(&sig) {
        cache.insert(sig, Features::from_solution(pre, challenge, sol)?);
    }
    Ok(&cache[&sig])
}

fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    pre: &Pre,
    iters: usize,
) -> Result<()> {
    let (greedy_sol, greedy_mk) = super::greedy::baseline(challenge)?;
    save_solution(&greedy_sol)?;
    let mut cache: LsCache = HashMap::with_hasher(seeded_hasher(&challenge.seed));
    let mut feature_cache: HashMap<u64, Features> =
        HashMap::with_capacity_and_hasher(64, seeded_hasher(&challenge.seed));
    let mut rng = SmallRng::from_seed(challenge.seed);
    let flex01 = (pre.high_flex + pre.jobshopness).clamp(0.0, 1.0);
    let mut inc = Incumbent {
        mk: greedy_mk,
        sol: greedy_sol.clone(),
        published: greedy_mk,
    };
    let mut top_solutions: Vec<(Solution, u32)> = Vec::new();
    push_top_solutions(&mut top_solutions, &greedy_sol, greedy_mk, TOP_KEEP);
    let target_margin: u32 = ((pre.avg_op_min
        * (0.9 + 0.9 * pre.high_flex + 0.6 * pre.jobshopness))
        .max(1.0)) as u32;

    // One greedy construction per rule ranks the rules and seeds the elite.
    let mut ranked: Vec<(Rule, u32, Solution)> = Vec::with_capacity(Rule::ALL.len());
    for rule in Rule::ALL {
        let (sol, mk) = construct::construct(challenge, pre, rule, 0, None, &mut rng)?;
        inc.offer(save_solution, &sol, mk)?;
        push_top_solutions(&mut top_solutions, &sol, mk, TOP_KEEP);
        ranked.push((rule, mk, sol));
    }
    ranked.sort_by_key(|x| x.1);
    let r0 = ranked[0].0;
    let r1 = ranked[1].0;
    let r2 = ranked[2].0;
    let mut rule_best = [u32::MAX; 9];
    let mut rule_tries = [0u32; 9];
    for (rr, mk, _) in &ranked {
        let idx = rr.index();
        rule_best[idx] = rule_best[idx].min(*mk);
        rule_tries[idx] = rule_tries[idx].saturating_add(1);
    }

    let elite_cap: usize = (6usize + (2.0 * flex01).round() as usize).clamp(6, 8);
    let mut elite: Vec<Elite> = Vec::new();
    for (_, mk, sol) in ranked.iter().take(3) {
        let f = features_of(pre, challenge, sol, &mut feature_cache)?;
        elite.push(Elite {
            features: f.clone(),
            score: *mk,
        });
    }
    {
        let f = features_of(pre, challenge, &greedy_sol, &mut feature_cache)?;
        elite.push(Elite {
            features: f.clone(),
            score: greedy_mk,
        });
    }
    normalize_elite(&mut elite, elite_cap);

    let mut stuck: usize = 0;
    let mut escape_cooldown: usize = 0;

    for r in 0..iters {
        if escape_cooldown > 0 {
            escape_cooldown -= 1;
        }
        if escape_cooldown == 0 {
            if let Some((sol2, mk2)) = escape(
                pre,
                challenge,
                &mut rng,
                &top_solutions,
                &inc.sol,
                stuck,
                flex01,
                &mut cache,
            )? {
                escape_cooldown = ESCAPE_COOLDOWN;
                if inc.offer(save_solution, &sol2, mk2)? {
                    stuck = 0;
                    let f = features_of(pre, challenge, &sol2, &mut feature_cache)?;
                    maybe_add_elite(
                        &mut elite,
                        Elite {
                            features: f.clone(),
                            score: mk2,
                        },
                        elite_cap,
                    );
                } else {
                    stuck = stuck.saturating_add(1);
                }
                push_top_solutions(&mut top_solutions, &sol2, mk2, TOP_KEEP);
                continue;
            } else if stuck > 150 {
                escape_cooldown = STALL_COOLDOWN;
            }
        }

        let late = r >= (iters * 2) / 3;
        let (k_min, k_max) = if stuck > 170 {
            (4usize, K_MAX)
        } else if stuck > 90 {
            (3usize, K_MAX)
        } else if stuck > 35 {
            (2usize, K_MAX)
        } else {
            (2usize, 4usize)
        };

        let rule = if r < WARMUP_RESTARTS {
            let u: f64 = rng.gen();
            if u < 0.11 {
                Rule::FlexBalance
            } else if u < 0.18 {
                Rule::ShortestProc
            } else if u < 0.50 {
                r0
            } else if u < 0.75 {
                r1
            } else if u < 0.90 {
                r2
            } else {
                Rule::ALL[rng.gen_range(0..Rule::ALL.len())]
            }
        } else {
            choose_rule(
                &mut rng,
                &rule_best,
                &rule_tries,
                inc.mk,
                target_margin,
                stuck,
                late,
            )
        };

        let k = if stuck > 120 && rng.gen::<f64>() < 0.55 {
            k_max
        } else {
            rng.gen_range(k_min..=k_max)
        };
        let learn_base =
            (0.09 + 0.24 * pre.jobshopness + 0.20 * pre.high_flex).clamp(0.06, 0.44);
        let learn_boost =
            (1.0 + 0.38 * ((stuck as f64) / 120.0).clamp(0.0, 1.0)).clamp(1.0, 1.38);
        let learn_p = (learn_base * learn_boost).clamp(0.0, 0.65);

        if stuck > 80 && !top_solutions.is_empty() && rng.gen::<f64>() < 0.04 {
            let idx = pick_top_idx(&mut rng, &top_solutions);
            let (sref, mkref) = (&top_solutions[idx].0, top_solutions[idx].1);
            let f = features_of(pre, challenge, sref, &mut feature_cache)?;
            maybe_add_elite(
                &mut elite,
                Elite {
                    features: f.clone(),
                    score: mkref,
                },
                elite_cap,
            );
        }

        let use_learn = !elite.is_empty() && rng.gen::<f64>() < learn_p;
        let target = inc.mk.saturating_add(target_margin);

        let (mut sol, mut mk) = if use_learn {
            // Job bias from one elite member; machine penalty and route preference from possibly
            // other members, each dropped with a small probability.
            let mix_p = (0.055
                + 0.10 * pre.high_flex
                + 0.09 * pre.jobshopness
                + 0.16 * ((stuck as f64) / 160.0).clamp(0.0, 1.0))
            .clamp(0.05, 0.40);
            let base_idx = pick_elite_idx(&mut rng, &elite);
            let mut mp_idx = base_idx;
            let mut rp_idx = base_idx;
            if elite.len() > 1 && rng.gen::<f64>() < mix_p {
                mp_idx = pick_elite_idx(&mut rng, &elite);
            }
            if elite.len() > 1 && rng.gen::<f64>() < mix_p {
                rp_idx = pick_elite_idx(&mut rng, &elite);
            }
            let drop_mp_p = (0.030 + 0.060 * pre.high_flex).clamp(0.03, 0.10);
            let drop_rp_p = (0.030 + 0.070 * pre.jobshopness).clamp(0.03, 0.12);
            let machine_penalty = if rng.gen::<f64>() < drop_mp_p {
                None
            } else {
                Some(elite[mp_idx].features.machine_penalty.as_slice())
            };
            let route_pref = if rng.gen::<f64>() < drop_rp_p {
                None
            } else {
                Some(&elite[rp_idx].features.route_pref)
            };
            // One draw is consumed here for every guided construction.
            let _ = rng.gen::<f64>();
            let guide = Guide {
                job_bias: &elite[base_idx].features.job_bias,
                machine_penalty,
                route_pref,
            };
            construct::construct_guided(challenge, pre, rule, k, target, &mut rng, &guide)?
        } else {
            construct::construct(challenge, pre, rule, k, Some(target), &mut rng)?
        };

        // An improving construction is saved before the local search refines it.
        inc.publish(save_solution, &sol, mk)?;
        if let Some((sol2, mk2)) = intensify(
            pre,
            challenge,
            &mut rng,
            &sol,
            mk,
            inc.mk,
            target_margin,
            stuck,
            late,
            &mut cache,
        )? {
            sol = sol2;
            mk = mk2;
        }

        let ridx = rule.index();
        rule_tries[ridx] = rule_tries[ridx].saturating_add(1);
        rule_best[ridx] = rule_best[ridx].min(mk);
        if inc.offer(save_solution, &sol, mk)? {
            stuck = 0;
            let f = features_of(pre, challenge, &sol, &mut feature_cache)?;
            maybe_add_elite(
                &mut elite,
                Elite {
                    features: f.clone(),
                    score: mk,
                },
                elite_cap,
            );
        } else {
            stuck = stuck.saturating_add(1);
            let add_p = (0.075 + 0.025 * flex01).clamp(0.07, 0.11);
            if mk <= inc.mk.saturating_add(target_margin / 2) && rng.gen::<f64>() < add_p {
                let f = features_of(pre, challenge, &sol, &mut feature_cache)?;
                maybe_add_elite(
                    &mut elite,
                    Elite {
                        features: f.clone(),
                        score: mk,
                    },
                    elite_cap,
                );
            }
        }
        push_top_solutions(&mut top_solutions, &sol, mk, TOP_KEEP);
    }
    Ok(())
}
}
