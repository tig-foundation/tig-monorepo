// Job-shop and flow-shop tracks.
#![allow(dead_code, clippy::all)]
pub mod types {
pub const INF: u32 = u32::MAX / 4;
pub const NONE_USIZE: usize = usize::MAX;

#[derive(Clone)]
pub struct OpInfo {
    pub machines: Vec<(usize, u32)>,
    pub min_pt: u32,
    pub avg_pt: f64,
    pub flex: usize,
    pub bn_avg: f64,
}

#[derive(Clone, Copy, Default)]
pub struct OpRoute {
    pub best_m: u8,
    pub best_w: u8,
    pub second_m: u8,
    pub second_w: u8,
}

pub type RoutePrefLite = Vec<Vec<OpRoute>>;

#[derive(Clone)]
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
    pub machine_scarcity: Vec<f64>,
    pub machine_weight: Vec<f64>,
    pub machine_best_pop: Vec<f64>,
    pub avg_machine_load: f64,
    pub avg_machine_scarcity: f64,
    pub avg_op_min: f64,
    pub horizon: f64,
    pub time_scale: f64,
    pub max_ops: usize,
    pub max_job_avg_work: f64,
    pub max_job_bn: f64,
    pub flex_avg: f64,
    pub flex_factor: f64,
    pub hi_flex: bool,
    pub high_flex: f64,
    pub flow_like: f64,
    pub flow_w: f64,
    pub job_flow_pref: Vec<f64>,
    pub jobshopness: f64,
    pub bn_focus: f64,
    pub load_cv: f64,
    pub slack_base: f64,
    pub total_ops: usize,
    pub chaotic_like: bool,
    pub flow_route: Option<Vec<usize>>,
    pub flow_pt_by_job: Option<Vec<Vec<u32>>>,
    pub strict_route: Option<Vec<usize>>,
}

#[derive(Clone, Copy)]
pub struct Cand {
    pub job: usize,
    pub machine: usize,
    pub pt: u32,
    pub score: f64,
}

#[derive(Clone, Copy)]
pub struct RawCand {
    pub job: usize,
    pub machine: usize,
    pub pt: u32,
    pub base_score: f64,
    pub rigidity: f64,
    pub reg_n: f64,
}

#[derive(Clone, Copy)]
pub enum GreedyRule {
    MostWork,
    MostOps,
    LeastFlex,
    ShortestProc,
    LongestProc,
}

#[derive(Clone)]
pub struct DisjSchedule {
    pub n: usize,
    pub num_jobs: usize,
    pub num_machines: usize,
    pub job_offsets: Vec<usize>,
    pub job_succ: Vec<usize>,
    pub indeg_job: Vec<u16>,
    // `indeg_job[o] + 1`: the in-degree of an operation that has a machine predecessor.
    // This is a constant of the instance - the machine sequence does not change which
    // operations have a route predecessor, and only one operation per machine escapes the
    // machine predecessor (the head one). `eval_disj` therefore copies this array in one block
    // (`copy_from_slice` = memcpy, capped at 40 fuel by the counting pass) then subtracts 1
    // from the `num_machines` sequence heads, instead of rescanning the n operations one by one.
    pub indeg_job_plus1: Vec<u16>,
    pub node_machine: Vec<usize>,
    pub node_pt: Vec<u32>,
    pub node_job: Vec<usize>,
    pub node_op: Vec<usize>,
    pub machine_seq: Vec<Vec<usize>>,
}

pub struct EvalBuf {
    pub indeg: Vec<u16>,
    pub start: Vec<u32>,
    pub best_pred: Vec<usize>,
    pub machine_succ: Vec<usize>,
    // Fixed-size buffer of `n`, plus a `Vec` we `push`/`pop`. Each operation enters it only
    // once - it is only pushed at the instant its in-degree drops to zero, which happens once -
    // so `n` slots always suffice. The length lives in a local variable: this removes, on every
    // operation, the capacity check of `push` and the read/write in memory of the `len` field.
    // The LIFO order is unchanged.
    pub stack: Vec<usize>,
}

impl EvalBuf {
    pub fn new(n: usize) -> Self {
        Self {
            indeg: vec![0u16; n],
            start: vec![0u32; n],
            best_pred: vec![NONE_USIZE; n],
            machine_succ: vec![NONE_USIZE; n],
            stack: vec![0usize; n],
        }
    }
}

#[derive(Clone, Copy)]
pub struct MoveCand {
    pub kind: u8,
    pub m_from: usize,
    pub from: usize,
    pub m_to: usize,
    pub to: usize,
    pub new_pt: u32,
    pub score: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct EffortConfig {
    pub job_shop_iters: usize,
    pub hybrid_flow_shop_iters: usize,
    pub fjsp_medium_iters: usize,
    pub fjsp_high_iters: usize,
    pub js_seed_select_mode: usize,
    pub js_ts_fast_eval: usize,

    pub js_ts_starts: usize,
    
    pub js_ts_div_period: usize,
    
    pub js_ts_tie_mode: usize,

    pub js_ts_sel_key: usize,

    pub js_ts_exact_topk: usize,

    pub js_ts_traj_decorr: usize,

    pub js_ts_extra_draws: usize,

    pub js_ts_kick_spread: usize,

    pub js_ts_sa_accept: usize,

    pub js_mem_cross_scope: usize,

    pub js_mem_admit: usize,

    pub js_sbp_rounds: usize,

    pub js_ts_polish_rescue: usize,

    pub js_seed_sbp_construct: usize,

    pub js_seed_sbp_reopt_cycles: usize,

    pub js_seed_sbp_dual_cycles: usize,
}

impl EffortConfig {
    pub fn default_effort() -> Self {
        Self { job_shop_iters: 25000, hybrid_flow_shop_iters: 2000, fjsp_medium_iters: 2000, fjsp_high_iters: 2000, js_seed_select_mode: 1, js_ts_fast_eval: 0, js_ts_starts: 10, js_ts_div_period: 0, js_ts_tie_mode: 0, js_ts_sel_key: 0, js_ts_exact_topk: 0, js_ts_traj_decorr: 0, js_ts_extra_draws: 0, js_ts_kick_spread: 0, js_ts_sa_accept: 0, js_mem_cross_scope: 0, js_mem_admit: 0, js_sbp_rounds: 0, js_ts_polish_rescue: 0, js_seed_sbp_construct: 0, js_seed_sbp_reopt_cycles: 1, js_seed_sbp_dual_cycles: 0 }
    }

    pub fn with_js_seed_sbp_dual_cycles(mut self, v: usize) -> Self {
        self.js_seed_sbp_dual_cycles = if v <= 64 { v } else { 0 };
        self
    }

    pub fn with_js_seed_sbp_reopt_cycles(mut self, v: usize) -> Self {
        self.js_seed_sbp_reopt_cycles = if v <= 64 { v } else { 1 };
        self
    }

    pub fn with_js_seed_sbp_construct(mut self, v: usize) -> Self {
        self.js_seed_sbp_construct = if v == 1 || v == 2 { v } else { 0 };
        self
    }

    pub fn with_js_ts_polish_rescue(mut self, v: usize) -> Self {
        self.js_ts_polish_rescue = if v == 1 || v == 2 { v } else { 0 };
        self
    }

    pub fn with_js_sbp_rounds(mut self, v: usize) -> Self {
        self.js_sbp_rounds = v.min(64);
        self
    }

    pub fn with_js_mem_admit(mut self, v: usize) -> Self {
        self.js_mem_admit = if v <= 3 { v } else { 0 };
        self
    }
    pub fn with_js_mem_cross_scope(mut self, v: usize) -> Self {
        self.js_mem_cross_scope = if v == 1 || v == 2 { v } else { 0 };
        self
    }

    pub fn with_js_ts_sa_accept(mut self, v: usize) -> Self {
        self.js_ts_sa_accept = v.min(4000);
        self
    }

    pub fn with_js_ts_kick_spread(mut self, v: usize) -> Self {
        self.js_ts_kick_spread = v.min(64);
        self
    }

    pub fn with_js_ts_extra_draws(mut self, v: usize) -> Self {
        self.js_ts_extra_draws = v.min(4);
        self
    }

    pub fn with_js_ts_traj_decorr(mut self, v: usize) -> Self {
        self.js_ts_traj_decorr = if v <= 2 { v } else { 0 };
        self
    }

    pub fn with_js_ts_exact_topk(mut self, v: usize) -> Self {
        self.js_ts_exact_topk = if v <= 1 { 0 } else { v.min(4) };
        self
    }

    pub fn with_js_ts_sel_key(mut self, v: usize) -> Self {
        self.js_ts_sel_key = if v <= 2 { v } else { 0 };
        self
    }

    pub fn with_js_ts_tie_mode(mut self, v: usize) -> Self {
        self.js_ts_tie_mode = if v <= 3 { v } else { 0 };
        self
    }

    pub fn with_js_seed_select_mode(mut self, v: usize) -> Self {
        self.js_seed_select_mode = v.min(2);
        self
    }

    pub fn with_js_ts_fast_eval(mut self, v: usize) -> Self {
        self.js_ts_fast_eval = v.min(1);
        self
    }

    pub fn with_js_ts_starts(mut self, v: usize) -> Self {
        self.js_ts_starts = v.clamp(1, 15);
        self
    }

    pub fn with_js_ts_div_period(mut self, v: usize) -> Self {
        self.js_ts_div_period = if v == 0 { 0 } else { v.clamp(60, 300_000) };
        self
    }

    pub fn with_job_shop_iters(mut self, v: usize) -> Self {
        self.job_shop_iters = v.clamp(100, 500000);
        self
    }

    pub fn with_hybrid_flow_shop_iters(mut self, v: usize) -> Self {
        self.hybrid_flow_shop_iters = v.clamp(100, 100000);
        self
    }

    pub fn with_fjsp_medium_iters(mut self, v: usize) -> Self {
        self.fjsp_medium_iters = v.clamp(100, 100000);
        self
    }

    pub fn with_fjsp_high_iters(mut self, v: usize) -> Self {
        self.fjsp_high_iters = v.clamp(100, 100000);
        self
    }
}
}
mod infra_shared {
use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::job_scheduling::*;
use super::types::*;

pub fn run_simple_greedy_baseline(challenge: &Challenge) -> Result<(Solution, u32)> {
    let num_jobs = challenge.num_jobs;
    let mut job_products = Vec::with_capacity(num_jobs);
    for (p, &cnt) in challenge.jobs_per_product.iter().enumerate() {
        for _ in 0..cnt { job_products.push(p); }
    }
    let job_ops_len: Vec<usize> = job_products.iter()
        .map(|&p| challenge.product_processing_times[p].len()).collect();
    let job_total_work: Vec<f64> = job_products.iter().map(|&p| {
        challenge.product_processing_times[p].iter()
            .map(|op| op.values().sum::<u32>() as f64 / op.len().max(1) as f64).sum()
    }).collect();

    let rules = [GreedyRule::MostWork, GreedyRule::MostOps, GreedyRule::LeastFlex, GreedyRule::ShortestProc, GreedyRule::LongestProc];
    let mut best_mk = u32::MAX; let mut best_sol: Option<Solution> = None;
    for rule in rules {
        let (sol, mk) = run_greedy_rule(challenge, &job_products, &job_ops_len, &job_total_work, rule, None)?;
        if mk < best_mk { best_mk = mk; best_sol = Some(sol); }
    }
    let mut rng = SmallRng::from_seed(challenge.seed);
    for _ in 0..10 {
        let seed = rng.gen::<u64>(); let rule = rules[rng.gen_range(0..rules.len())];
        let random_top_k = rng.gen_range(2..=5); let mut local_rng = SmallRng::seed_from_u64(seed);
        let (sol, mk) = run_greedy_rule(challenge, &job_products, &job_ops_len, &job_total_work, rule, Some((random_top_k, &mut local_rng)))?;
        if mk < best_mk { best_mk = mk; best_sol = Some(sol); }
    }
    Ok((best_sol.ok_or_else(|| anyhow!("No greedy solution"))?, best_mk))
}

pub fn run_simple_greedy_baseline_weighted(challenge: &Challenge) -> Result<(Solution, u32)> {
    let num_jobs = challenge.num_jobs;
    let mut job_products = Vec::with_capacity(num_jobs);
    for (p, &cnt) in challenge.jobs_per_product.iter().enumerate() {
        for _ in 0..cnt { job_products.push(p); }
    }
    let job_ops_len: Vec<usize> = job_products.iter()
        .map(|&p| challenge.product_processing_times[p].len()).collect();
    let job_total_work: Vec<f64> = job_products.iter().map(|&p| {
        challenge.product_processing_times[p].iter()
            .map(|op| op.values().sum::<u32>() as f64 / op.len().max(1) as f64).sum()
    }).collect();

    let mut best_mk = u32::MAX; let mut best_sol: Option<Solution> = None;
    let base_weights = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for w in base_weights {
        let (sol, mk) = run_greedy_weighted(challenge, &job_products, &job_ops_len, &job_total_work, w, None)?;
        if mk < best_mk { best_mk = mk; best_sol = Some(sol); }
    }
    
    let mut rng = SmallRng::from_seed(challenge.seed);
    for _ in 0..10 {
        let seed = rng.gen::<u64>();
        let w = [
            rng.gen_range(0.0..1.0),
            rng.gen_range(0.0..1.0),
            rng.gen_range(-1.0..0.5),
            rng.gen_range(-1.0..1.0),
        ];
        let random_top_k = rng.gen_range(2..=5); let mut local_rng = SmallRng::seed_from_u64(seed);
        let (sol, mk) = run_greedy_weighted(challenge, &job_products, &job_ops_len, &job_total_work, w, Some((random_top_k, &mut local_rng)))?;
        if mk < best_mk { best_mk = mk; best_sol = Some(sol); }
    }
    Ok((best_sol.ok_or_else(|| anyhow!("No greedy solution"))?, best_mk))
}

pub fn run_greedy_weighted(
    challenge: &Challenge, job_products: &[usize], job_ops_len: &[usize], job_total_work: &[f64],
    weights: [f64; 4], mut random_top_k: Option<(usize, &mut SmallRng)>,
) -> Result<(Solution, u32)> {
    #[derive(Clone, Copy)]
    struct GCandidate { job: usize, priority: f64, end: u32, pt: u32, flex: usize }

    let num_jobs = challenge.num_jobs; let num_machines = challenge.num_machines;
    let mut job_next_op = vec![0usize; num_jobs]; let mut job_ready = vec![0u32; num_jobs]; let mut machine_avail = vec![0u32; num_machines];
    let mut job_schedule: Vec<Vec<(usize, u32)>> = job_ops_len.iter().map(|&len| Vec::with_capacity(len)).collect();
    let mut job_work_left = job_total_work.to_vec();
    let mut remaining = job_ops_len.iter().sum::<usize>(); let mut time = 0u32; let eps = 1e-9;
    let mut available_machines: Vec<usize> = Vec::with_capacity(num_machines);
    let mut candidates: Vec<GCandidate> = Vec::new();

    while remaining > 0 {
        available_machines.clear();
        for m in 0..num_machines {
            if machine_avail[m] <= time { available_machines.push(m); }
        }
        available_machines.sort_unstable();
        if let Some((_, ref mut rng)) = random_top_k { use rand::seq::SliceRandom; available_machines.shuffle(*rng); }

        if let Some((top_k, ref mut rng)) = random_top_k {
            for &m in &available_machines {
                candidates.clear();
                for j in 0..num_jobs {
                    if job_next_op[j] >= job_ops_len[j] || job_ready[j] > time { continue; }
                    let product = job_products[j]; let op_idx = job_next_op[j];
                    let op_times = &challenge.product_processing_times[product][op_idx];
                    let pt = match op_times.get(&m) { Some(&v) => v, None => continue };
                    let earliest = op_times.iter().map(|(&mm, &ppt)| time.max(machine_avail[mm]) + ppt).min().unwrap_or(u32::MAX);
                    let this_end = time.max(machine_avail[m]) + pt;
                    if this_end != earliest { continue; }
                    let flex = op_times.len(); let ops_left = job_ops_len[j] - job_next_op[j];
                    
                    let priority = weights[0] * job_work_left[j] 
                                 + weights[1] * (ops_left as f64) 
                                 + weights[2] * (flex as f64) 
                                 + weights[3] * (pt as f64);
                    
                    candidates.push(GCandidate { job: j, priority, end: this_end, pt, flex });
                }
                if candidates.is_empty() { continue; }

                candidates.sort_by(|a, b| { if (b.priority - a.priority).abs() > eps { b.priority.partial_cmp(&a.priority).unwrap() } else if a.end != b.end { a.end.cmp(&b.end) } else if a.pt != b.pt { a.pt.cmp(&b.pt) } else if a.flex != b.flex { a.flex.cmp(&b.flex) } else { a.job.cmp(&b.job) } });
                let top = candidates.len().min(top_k);
                let chosen = candidates[rng.gen_range(0..top)];
                let best_job = chosen.job;
                let product = job_products[best_job]; let op_idx = job_next_op[best_job];
                let op_times = &challenge.product_processing_times[product][op_idx];
                let avg_pt = op_times.values().sum::<u32>() as f64 / op_times.len().max(1) as f64;
                let st = time.max(machine_avail[m]); let end = st + chosen.pt;
                job_schedule[best_job].push((m, st)); job_next_op[best_job] += 1; job_ready[best_job] = end; machine_avail[m] = end;
                job_work_left[best_job] -= avg_pt; if job_work_left[best_job] < 0.0 { job_work_left[best_job] = 0.0; } remaining -= 1;
            }
        } else {
            for &m in &available_machines {
                let mut best: Option<GCandidate> = None;
                for j in 0..num_jobs {
                    if job_next_op[j] >= job_ops_len[j] || job_ready[j] > time { continue; }
                    let product = job_products[j]; let op_idx = job_next_op[j];
                    let op_times = &challenge.product_processing_times[product][op_idx];
                    let pt = match op_times.get(&m) { Some(&v) => v, None => continue };
                    let earliest = op_times.iter().map(|(&mm, &ppt)| time.max(machine_avail[mm]) + ppt).min().unwrap_or(u32::MAX);
                    let this_end = time.max(machine_avail[m]) + pt;
                    if this_end != earliest { continue; }
                    let flex = op_times.len(); let ops_left = job_ops_len[j] - job_next_op[j];
                    
                    let priority = weights[0] * job_work_left[j] 
                                 + weights[1] * (ops_left as f64) 
                                 + weights[2] * (flex as f64) 
                                 + weights[3] * (pt as f64);
                    
                    let cand = GCandidate {
                        job: j,
                        priority,
                        end: this_end,
                        pt,
                        flex,
                    };
                    let better = if let Some(b) = best {
                        if (cand.priority - b.priority).abs() > eps { cand.priority > b.priority } else if cand.end != b.end { cand.end < b.end } else if cand.pt != b.pt { cand.pt < b.pt } else if cand.flex != b.flex { cand.flex < b.flex } else { cand.job < b.job }
                    } else { true };
                    if better { best = Some(cand); }
                }
                let Some(best) = best else { continue };
                let best_job = best.job;
                let product = job_products[best_job]; let op_idx = job_next_op[best_job];
                let op_times = &challenge.product_processing_times[product][op_idx];
                let avg_pt = op_times.values().sum::<u32>() as f64 / op_times.len().max(1) as f64;
                let st = time.max(machine_avail[m]); let end = st + best.pt;
                job_schedule[best_job].push((m, st)); job_next_op[best_job] += 1; job_ready[best_job] = end; machine_avail[m] = end;
                job_work_left[best_job] -= avg_pt; if job_work_left[best_job] < 0.0 { job_work_left[best_job] = 0.0; } remaining -= 1;
            }
        }

        if remaining == 0 { break; }
        let mut next = u32::MAX;
        for &t in &machine_avail { if t > time && t < next { next = t; } }
        for j in 0..num_jobs { if job_next_op[j] < job_ops_len[j] && job_ready[j] > time && job_ready[j] < next { next = job_ready[j]; } }
        if next == u32::MAX { return Err(anyhow!("Greedy baseline stuck")); }
        time = next;
    }
    let mk = job_ready.iter().copied().max().unwrap_or(0);
    Ok((Solution { job_schedule }, mk))
}

pub fn run_greedy_rule(
    challenge: &Challenge, job_products: &[usize], job_ops_len: &[usize], job_total_work: &[f64],
    rule: GreedyRule, mut random_top_k: Option<(usize, &mut SmallRng)>,
) -> Result<(Solution, u32)> {
    #[derive(Clone, Copy)]
    struct GCandidate { job: usize, priority: f64, end: u32, pt: u32, flex: usize }

    let num_jobs = challenge.num_jobs; let num_machines = challenge.num_machines;
    let mut job_next_op = vec![0usize; num_jobs]; let mut job_ready = vec![0u32; num_jobs]; let mut machine_avail = vec![0u32; num_machines];
    let mut job_schedule: Vec<Vec<(usize, u32)>> = job_ops_len.iter().map(|&len| Vec::with_capacity(len)).collect();
    let mut job_work_left = job_total_work.to_vec();
    let mut remaining = job_ops_len.iter().sum::<usize>(); let mut time = 0u32; let eps = 1e-9;
    let mut available_machines: Vec<usize> = Vec::with_capacity(num_machines);
    let mut candidates: Vec<GCandidate> = Vec::new();

    while remaining > 0 {
        available_machines.clear();
        for m in 0..num_machines {
            if machine_avail[m] <= time { available_machines.push(m); }
        }
        available_machines.sort_unstable();
        if let Some((_, ref mut rng)) = random_top_k { use rand::seq::SliceRandom; available_machines.shuffle(*rng); }

        if let Some((top_k, ref mut rng)) = random_top_k {
            for &m in &available_machines {
                candidates.clear();
                for j in 0..num_jobs {
                    if job_next_op[j] >= job_ops_len[j] || job_ready[j] > time { continue; }
                    let product = job_products[j]; let op_idx = job_next_op[j];
                    let op_times = &challenge.product_processing_times[product][op_idx];
                    let pt = match op_times.get(&m) { Some(&v) => v, None => continue };
                    let earliest = op_times.iter().map(|(&mm, &ppt)| time.max(machine_avail[mm]) + ppt).min().unwrap_or(u32::MAX);
                    let this_end = time.max(machine_avail[m]) + pt;
                    if this_end != earliest { continue; }
                    let flex = op_times.len(); let ops_left = job_ops_len[j] - job_next_op[j];
                    let priority = match rule {
                        GreedyRule::MostWork => job_work_left[j], GreedyRule::MostOps => ops_left as f64,
                        GreedyRule::LeastFlex => -(flex as f64), GreedyRule::ShortestProc => -(pt as f64),
                        GreedyRule::LongestProc => pt as f64,
                    };
                    candidates.push(GCandidate { job: j, priority, end: this_end, pt, flex });
                }
                if candidates.is_empty() { continue; }

                candidates.sort_by(|a, b| { if (b.priority - a.priority).abs() > eps { b.priority.partial_cmp(&a.priority).unwrap() } else if a.end != b.end { a.end.cmp(&b.end) } else if a.pt != b.pt { a.pt.cmp(&b.pt) } else if a.flex != b.flex { a.flex.cmp(&b.flex) } else { a.job.cmp(&b.job) } });
                let top = candidates.len().min(top_k);
                let chosen = candidates[rng.gen_range(0..top)];
                let best_job = chosen.job;
                let product = job_products[best_job]; let op_idx = job_next_op[best_job];
                let op_times = &challenge.product_processing_times[product][op_idx];
                let avg_pt = op_times.values().sum::<u32>() as f64 / op_times.len().max(1) as f64;
                let st = time.max(machine_avail[m]); let end = st + chosen.pt;
                job_schedule[best_job].push((m, st)); job_next_op[best_job] += 1; job_ready[best_job] = end; machine_avail[m] = end;
                job_work_left[best_job] -= avg_pt; if job_work_left[best_job] < 0.0 { job_work_left[best_job] = 0.0; } remaining -= 1;
            }
        } else {
            for &m in &available_machines {
                let mut best: Option<GCandidate> = None;
                for j in 0..num_jobs {
                    if job_next_op[j] >= job_ops_len[j] || job_ready[j] > time { continue; }
                    let product = job_products[j]; let op_idx = job_next_op[j];
                    let op_times = &challenge.product_processing_times[product][op_idx];
                    let pt = match op_times.get(&m) { Some(&v) => v, None => continue };
                    let earliest = op_times.iter().map(|(&mm, &ppt)| time.max(machine_avail[mm]) + ppt).min().unwrap_or(u32::MAX);
                    let this_end = time.max(machine_avail[m]) + pt;
                    if this_end != earliest { continue; }
                    let flex = op_times.len(); let ops_left = job_ops_len[j] - job_next_op[j];
                    let cand = GCandidate {
                        job: j,
                        priority: match rule {
                            GreedyRule::MostWork => job_work_left[j], GreedyRule::MostOps => ops_left as f64,
                            GreedyRule::LeastFlex => -(flex as f64), GreedyRule::ShortestProc => -(pt as f64),
                            GreedyRule::LongestProc => pt as f64,
                        },
                        end: this_end,
                        pt,
                        flex,
                    };
                    let better = if let Some(b) = best {
                        if (cand.priority - b.priority).abs() > eps { cand.priority > b.priority } else if cand.end != b.end { cand.end < b.end } else if cand.pt != b.pt { cand.pt < b.pt } else if cand.flex != b.flex { cand.flex < b.flex } else { cand.job < b.job }
                    } else { true };
                    if better { best = Some(cand); }
                }
                let Some(best) = best else { continue };
                let best_job = best.job;
                let product = job_products[best_job]; let op_idx = job_next_op[best_job];
                let op_times = &challenge.product_processing_times[product][op_idx];
                let avg_pt = op_times.values().sum::<u32>() as f64 / op_times.len().max(1) as f64;
                let st = time.max(machine_avail[m]); let end = st + best.pt;
                job_schedule[best_job].push((m, st)); job_next_op[best_job] += 1; job_ready[best_job] = end; machine_avail[m] = end;
                job_work_left[best_job] -= avg_pt; if job_work_left[best_job] < 0.0 { job_work_left[best_job] = 0.0; } remaining -= 1;
            }
        }

        if remaining == 0 { break; }
        let mut next = u32::MAX;
        for &t in &machine_avail { if t > time && t < next { next = t; } }
        for j in 0..num_jobs { if job_next_op[j] < job_ops_len[j] && job_ready[j] > time && job_ready[j] < next { next = job_ready[j]; } }
        if next == u32::MAX { return Err(anyhow!("Greedy baseline stuck")); }
        time = next;
    }
    let mk = job_ready.iter().copied().max().unwrap_or(0);
    Ok((Solution { job_schedule }, mk))
}

pub fn build_disj_from_solution(pre: &Pre, challenge: &Challenge, sol: &Solution) -> Result<DisjSchedule> {
    let num_jobs = challenge.num_jobs; let num_machines = challenge.num_machines;
    let mut job_offsets = vec![0usize; num_jobs + 1];
    for j in 0..num_jobs { job_offsets[j + 1] = job_offsets[j] + pre.job_ops_len[j]; }
    let n = job_offsets[num_jobs];
    if n == 0 { return Err(anyhow!("No operations")); }
    let mut node_machine = vec![0usize; n]; let mut node_pt = vec![0u32; n]; let mut node_job = vec![0usize; n]; let mut node_op = vec![0usize; n];
    let mut per_machine: Vec<Vec<(u32, usize)>> = vec![Vec::new(); num_machines];
    for job in 0..num_jobs {
        let expected = pre.job_ops_len[job];
        if sol.job_schedule[job].len() != expected { return Err(anyhow!("Invalid solution: job {} ops len mismatch", job)); }
        let product = pre.job_products[job];
        for op_idx in 0..expected {
            let id = job_offsets[job] + op_idx; let (m, st) = sol.job_schedule[job][op_idx];
            let op = &pre.product_ops[product][op_idx];
            let pt = pt_from_op(op, m).ok_or_else(|| anyhow!("Invalid solution: pt missing"))?;
            if m >= num_machines { return Err(anyhow!("Invalid solution: machine out of range")); }
            node_machine[id] = m; node_pt[id] = pt; node_job[id] = job; node_op[id] = op_idx;
            per_machine[m].push((st, id));
        }
    }
    let mut machine_seq: Vec<Vec<usize>> = Vec::with_capacity(num_machines);
    for m in 0..num_machines {
        per_machine[m].sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        machine_seq.push(per_machine[m].iter().map(|&(_, id)| id).collect());
    }
    let mut job_succ = vec![NONE_USIZE; n]; let mut indeg_job = vec![0u16; n];
    for job in 0..num_jobs {
        let len = pre.job_ops_len[job]; let base = job_offsets[job];
        for k in 0..len { let id = base + k; if k + 1 < len { job_succ[id] = id + 1; indeg_job[id + 1] = indeg_job[id + 1].saturating_add(1); } }
    }
    let mut indeg_job_plus1 = indeg_job.clone();
    for d in indeg_job_plus1.iter_mut() { *d += 1; }
    Ok(DisjSchedule { n, num_jobs, num_machines, job_offsets, job_succ, indeg_job, indeg_job_plus1, node_machine, node_pt, node_job, node_op, machine_seq })
}

#[inline]
pub fn pt_from_op(op: &OpInfo, machine: usize) -> Option<u32> {
    for &(m, pt) in &op.machines { if m == machine { return Some(pt); } }
    None
}

/// Trial evaluation for probe sites: the same pass as `eval_disj` minus the `best_pred`
/// bookkeeping, with a monotone bound. The makespan accumulator only grows, so reaching
/// `bound` makes the final value >= bound: `None` is then equivalent to a rejected probe,
/// since every probe consumer tests strictly below its incumbent and the winning candidate
/// is re-evaluated in full before `start`/`best_pred` are read again.
pub fn eval_disj_trial(ds: &DisjSchedule, buf: &mut EvalBuf, bound: u32) -> Option<u32> {
    let n = ds.n;
    buf.indeg.copy_from_slice(&ds.indeg_job_plus1);
    buf.start.fill(0);
    let mut covered = 0usize;
    for seq in &ds.machine_seq {
        if seq.is_empty() { continue; }
        buf.indeg[seq[0]] -= 1;
        covered += seq.len();
        let mut prev = seq[0];
        for &v in &seq[1..] {
            buf.machine_succ[prev] = v;
            prev = v;
        }
        buf.machine_succ[prev] = NONE_USIZE;
    }
    if covered != n {
        buf.indeg.copy_from_slice(&ds.indeg_job);
        for seq in &ds.machine_seq {
            for &v in seq.iter().skip(1) { buf.indeg[v] = buf.indeg[v].saturating_add(1); }
        }
    }
    let mut sl = 0usize;
    for j in 0..ds.num_jobs {
        let h = ds.job_offsets[j];
        if h < ds.job_offsets[j + 1] && buf.indeg[h] == 0 { buf.stack[sl] = h; sl += 1; }
    }
    let mut processed = 0usize;
    let mut mk = 0u32;
    while sl > 0 {
        sl -= 1;
        let u = buf.stack[sl];
        processed += 1;
        let end_u = buf.start[u].saturating_add(ds.node_pt[u]);
        if end_u > mk {
            mk = end_u;
            if mk >= bound { return None; }
        }
        let js = ds.job_succ[u];
        if js != NONE_USIZE {
            if buf.start[js] < end_u { buf.start[js] = end_u; }
            buf.indeg[js] -= 1;
            if buf.indeg[js] == 0 { buf.stack[sl] = js; sl += 1; }
        }
        let ms = buf.machine_succ[u];
        if ms != NONE_USIZE {
            if buf.start[ms] < end_u { buf.start[ms] = end_u; }
            buf.indeg[ms] -= 1;
            if buf.indeg[ms] == 0 { buf.stack[sl] = ms; sl += 1; }
        }
    }
    if processed != n { return None; }
    Some(mk)
}

pub fn eval_disj(ds: &DisjSchedule, buf: &mut EvalBuf) -> Option<(u32, usize)> {
    let n = ds.n;
    // `copy_from_slice`/`fill` compile to memcpy/memset, which the counting pass caps at
    // 40 fuel whatever the size. The equivalent hand-written loop over n = 1500 operations cost
    // several thousand.
    buf.indeg.copy_from_slice(&ds.indeg_job_plus1);
    buf.start.fill(0);
    let mut covered = 0usize;
    for seq in &ds.machine_seq {
        if seq.is_empty() { continue; }
        // `indeg_job_plus1` has already counted one machine predecessor for every operation;
        // only the sequence head has none. One subtraction per machine replaces the n
        // increments of the former loop, and returns the same array as long as the machine
        // sequences partition the operations.
        buf.indeg[seq[0]] -= 1;
        covered += seq.len();
        let mut prev = seq[0];
        for &v in &seq[1..] {
            buf.machine_succ[prev] = v;
            prev = v;
        }
        buf.machine_succ[prev] = NONE_USIZE;
    }
    if covered != n {
        // A safety net, not an expected case: `sbp_construct_seed` empties machine sequences
        // before rebuilding them. An operation that is in no sequence at all has no machine
        // predecessor, whereas `indeg_job_plus1` counted one for it. We then start again from
        // the exact base, with the original loop. Thirty additions per evaluation buy not
        // having to assume the partition - equality with the former code becomes a theorem
        // rather than the result of a re-reading.
        buf.indeg.copy_from_slice(&ds.indeg_job);
        for seq in &ds.machine_seq {
            for &v in seq.iter().skip(1) { buf.indeg[v] = buf.indeg[v].saturating_add(1); }
        }
    }
    // A root necessarily has `indeg_job == 0`, that is, it is the first operation of a job.
    // There are `num_jobs` (50) of them, not `n` (1500). Since `job_offsets` is increasing,
    // scanning them in job order returns exactly the increasing identifier order of the former
    // `for i in 0..n`, so the starting stack - and the whole topological order that follows from
    // it - is unchanged.
    let mut sl = 0usize;
    for j in 0..ds.num_jobs {
        let h = ds.job_offsets[j];
        if h < ds.job_offsets[j + 1] && buf.indeg[h] == 0 {
            buf.best_pred[h] = NONE_USIZE;
            buf.stack[sl] = h;
            sl += 1;
        }
    }
    let mut processed = 0usize;
    let mut mk = 0u32;
    let mut mk_node = 0usize;
    while sl > 0 {
        sl -= 1;
        let u = buf.stack[sl];
        processed += 1;
        let end_u = buf.start[u].saturating_add(ds.node_pt[u]);
        if end_u > mk { mk = end_u; mk_node = u; }
        let js = ds.job_succ[u];
        if js != NONE_USIZE {
            if buf.start[js] < end_u {
                buf.start[js] = end_u;
                buf.best_pred[js] = u;
            }
            // `indeg[js] >= 1` here: we decrement only once per incoming arc (an operation is
            // pushed, hence popped, only once), so the original `saturating_sub` could never
            // saturate. In debug a faulty `-= 1` would panic: that is the canary.
            buf.indeg[js] -= 1;
            if buf.indeg[js] == 0 { buf.stack[sl] = js; sl += 1; }
        }
        let ms = buf.machine_succ[u];
        if ms != NONE_USIZE {
            if buf.start[ms] < end_u {
                buf.start[ms] = end_u;
                buf.best_pred[ms] = u;
            }
            buf.indeg[ms] -= 1;
            if buf.indeg[ms] == 0 { buf.stack[sl] = ms; sl += 1; }
        }
    }
    if processed != n { return None; }
    Some((mk, mk_node))
}

pub fn disj_to_solution(pre: &Pre, ds: &DisjSchedule, start: &[u32]) -> Result<Solution> {
    let num_jobs = ds.num_jobs;
    let mut job_schedule: Vec<Vec<(usize, u32)>> = Vec::with_capacity(num_jobs);
    for j in 0..num_jobs {
        let len = pre.job_ops_len[j]; let mut v = Vec::with_capacity(len); let base = ds.job_offsets[j];
        for k in 0..len { let id = base + k; v.push((ds.node_machine[id], start[id])); }
        job_schedule.push(v);
    }
    Ok(Solution { job_schedule })
}

pub fn critical_block_move_local_search_ex(
    pre: &Pre, challenge: &Challenge, base_sol: &Solution,
    max_iters: usize, top_cands: usize, perturb_cycles: usize,
) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n); let mut crit = vec![false; ds.n];
    let mut cur_eval = match eval_disj(&ds, &mut buf) { Some(x) => x, None => return Ok(None) };
    let initial_mk = cur_eval.0;
    descent_phase(&mut ds, &mut buf, &mut crit, pre, &mut cur_eval, max_iters, top_cands);
    let Some((mk_after, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let mut global_best_mk = mk_after; let mut global_best_ds = ds.clone();
    let mut sol_hash: u64 = 0;
    for m in 0..ds.num_machines.min(8) {
        if !ds.machine_seq[m].is_empty() {
            let first_node = ds.machine_seq[m][0];
            sol_hash ^= (first_node as u64).wrapping_mul(0xD2B54A6B68A5);
            sol_hash = sol_hash.rotate_left(7);
        }
    }
    let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (initial_mk as u64).wrapping_shl(16) ^ (ds.n as u64) ^ sol_hash;
    for _cycle in 0..perturb_cycles {
        ds = global_best_ds.clone();
        let Some((_, mk_node)) = eval_disj(&ds, &mut buf) else { break };
        crit.fill(false); let mut u = mk_node; while u != NONE_USIZE { crit[u] = true; u = buf.best_pred[u]; }
        let mut blocks: Vec<(usize, usize, usize)> = Vec::new();
        for m in 0..ds.num_machines {
            let seq = &ds.machine_seq[m]; if seq.len() <= 1 { continue; }
            let mut i = 0usize;
            while i < seq.len() {
                if !crit[seq[i]] { i += 1; continue; }
                let bstart = i; let mut bend = i;
                while bend + 1 < seq.len() { let x = seq[bend]; let y = seq[bend+1]; if !crit[y] { break; } if buf.start[y] != buf.start[x].saturating_add(ds.node_pt[x]) { break; } bend += 1; }
                if bend > bstart { blocks.push((m, bstart, bend)); } i = bend + 1;
            }
        }
        if blocks.is_empty() { break; }
        for _ in 0..2 {
            pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
            let bidx = (pseed as usize) % blocks.len(); let (m, bstart, bend) = blocks[bidx];
            let block_len = bend - bstart; if block_len == 0 { continue; }
            pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
            let swap_pos = bstart + ((pseed as usize) % block_len);
            if swap_pos + 1 < ds.machine_seq[m].len() { ds.machine_seq[m].swap(swap_pos, swap_pos + 1); }
        }
        match eval_disj(&ds, &mut buf) { Some(x) => cur_eval = x, None => continue }
        descent_phase(&mut ds, &mut buf, &mut crit, pre, &mut cur_eval, max_iters, top_cands);
        if let Some((mk_now, _)) = eval_disj(&ds, &mut buf) { if mk_now < global_best_mk { global_best_mk = mk_now; global_best_ds = ds.clone(); } }
    }
    if global_best_mk >= initial_mk { return Ok(None); }
    ds = global_best_ds;
    let Some((mk_final, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let sol = disj_to_solution(pre, &ds, &buf.start)?;
    Ok(Some((sol, mk_final)))
}

pub fn critical_block_move_local_search_ex_fjsp(
    pre: &Pre, challenge: &Challenge, base_sol: &Solution,
    max_iters: usize, top_cands: usize, perturb_cycles: usize,
) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n); let mut crit = vec![false; ds.n];
    let mut cur_eval = match eval_disj(&ds, &mut buf) { Some(x) => x, None => return Ok(None) };
    let initial_mk = cur_eval.0;
    descent_phase_fjsp(&mut ds, &mut buf, &mut crit, pre, &mut cur_eval, max_iters, top_cands);
    let Some((mk_after, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let mut global_best_mk = mk_after; let mut global_best_ds = ds.clone();
    let mut sol_hash: u64 = 0;
    for m in 0..ds.num_machines.min(8) {
        if !ds.machine_seq[m].is_empty() {
            let first_node = ds.machine_seq[m][0];
            sol_hash ^= (first_node as u64).wrapping_mul(0xD2B54A6B68A5);
            sol_hash = sol_hash.rotate_left(7);
        }
    }
    let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (initial_mk as u64).wrapping_shl(16) ^ (ds.n as u64) ^ sol_hash;
    for _cycle in 0..perturb_cycles {
        ds = global_best_ds.clone();
        let Some((_, mk_node)) = eval_disj(&ds, &mut buf) else { break };
        crit.fill(false); let mut u = mk_node; while u != NONE_USIZE { crit[u] = true; u = buf.best_pred[u]; }
        let mut blocks: Vec<(usize, usize, usize)> = Vec::new();
        for m in 0..ds.num_machines {
            let seq = &ds.machine_seq[m]; if seq.len() <= 1 { continue; }
            let mut i = 0usize;
            while i < seq.len() {
                if !crit[seq[i]] { i += 1; continue; }
                let bstart = i; let mut bend = i;
                while bend + 1 < seq.len() { let x = seq[bend]; let y = seq[bend+1]; if !crit[y] { break; } if buf.start[y] != buf.start[x].saturating_add(ds.node_pt[x]) { break; } bend += 1; }
                if bend > bstart { blocks.push((m, bstart, bend)); } i = bend + 1;
            }
        }
        if blocks.is_empty() { break; }
        let mut extracted = Vec::with_capacity(3);
        let num_extract = 3.min(blocks.len() * 2);
        for _ in 0..num_extract {
            pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
            let bidx = (pseed as usize) % blocks.len(); let (m, bstart, bend) = blocks[bidx];
            let block_len = bend - bstart; if block_len == 0 { continue; }
            pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
            let ext_pos = bstart + ((pseed as usize) % block_len);
            if ext_pos < ds.machine_seq[m].len() {
                extracted.push((m, ds.machine_seq[m].remove(ext_pos)));
            }
        }
        for (m, node) in extracted {
            let desired = buf.start[node];
            let mut pos = find_insert_pos_by_start(&ds.machine_seq[m], &buf.start, desired);
            pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
            let offset = (pseed as usize) % 3;
            if (pseed >> 3) & 1 == 0 { pos = pos.saturating_add(offset); } else { pos = pos.saturating_sub(offset); }
            let m_len = ds.machine_seq[m].len();
            ds.machine_seq[m].insert(pos.min(m_len), node);
        }
        match eval_disj(&ds, &mut buf) { Some(x) => cur_eval = x, None => continue }
        descent_phase_fjsp(&mut ds, &mut buf, &mut crit, pre, &mut cur_eval, max_iters, top_cands);
        if let Some((mk_now, _)) = eval_disj(&ds, &mut buf) { if mk_now < global_best_mk { global_best_mk = mk_now; global_best_ds = ds.clone(); } }
    }
    if global_best_mk >= initial_mk { return Ok(None); }
    ds = global_best_ds;
    let Some((mk_final, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let sol = disj_to_solution(pre, &ds, &buf.start)?;
    Ok(Some((sol, mk_final)))
}

fn descent_phase(
    ds: &mut DisjSchedule, buf: &mut EvalBuf, crit: &mut Vec<bool>, pre: &Pre,
    cur_eval: &mut (u32, usize), max_iters: usize, top_cands: usize,
) -> bool {
    let mut cur_mk = cur_eval.0; let mut improved = false;
    let mut seen_probes: Vec<(u8, usize, usize, usize, usize, u32)> = Vec::with_capacity(64);
    for _iter in 0..max_iters {
        crit.fill(false); let mut u = cur_eval.1; while u != NONE_USIZE { crit[u] = true; u = buf.best_pred[u]; }
        let mut cands: Vec<MoveCand> = Vec::with_capacity(top_cands.min(64));
        for m in 0..ds.num_machines {
            let seq = &ds.machine_seq[m]; if seq.len() <= 1 { continue; }
            let mut i = 0usize;
            while i < seq.len() {
                let a = seq[i]; if !crit[a] { i += 1; continue; }
                let bstart = i; let mut bend = i;
                while bend + 1 < seq.len() { let x = seq[bend]; let y = seq[bend+1]; if !crit[y] { break; } if buf.start[y] != buf.start[x].saturating_add(ds.node_pt[x]) { break; } bend += 1; }
                if bend > bstart {
                    let max_shift = bend - bstart;
                    let mut shifts: [usize; 3] = [1, 2, max_shift];
                    for sh in shifts.iter_mut() { if *sh > max_shift { *sh = 0; } }
                    for &sh in &shifts {
                        if sh == 0 { continue; }
                        { let from = bstart; let to_after = bstart + sh; if from < seq.len() && to_after <= seq.len() { let tgt_idx = (bstart+sh).min(seq.len()-1); push_top_k_move(&mut cands, MoveCand { kind: 0, m_from: m, from, m_to: m, to: to_after, new_pt: 0, score: buf.start[seq[tgt_idx]] }, top_cands); } }
                        { let from = bend; let to_after = bend - sh; push_top_k_move(&mut cands, MoveCand { kind: 0, m_from: m, from, m_to: m, to: to_after, new_pt: 0, score: buf.start[seq[bend]] }, top_cands); }
                    }
                    if bstart > 0 { push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bstart-1, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bstart]] }, top_cands); }
                    if bend + 1 < seq.len() { push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bend, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bend]] }, top_cands); }
                    if bstart + 1 <= bend {
                        push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bstart, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bstart+1]] }, top_cands);
                        if bend >= 1 && bend - 1 >= bstart { push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bend-1, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bend]] }, top_cands); }
                    }
                    for &idx in &[bstart, bend] {
                        if idx >= seq.len() { continue; }
                        let node = seq[idx]; if !crit[node] { continue; }
                        let job = ds.node_job[node]; let op_idx = ds.node_op[node]; let product = pre.job_products[job];
                        let op = &pre.product_ops[product][op_idx];
                        if op.flex < 2 || op.machines.len() < 2 { continue; }
                        let old_m = ds.node_machine[node]; let old_pt = ds.node_pt[node];
                        let w_from = pre.machine_weight[old_m].max(1e-9);
                        let best2 = best_two_by_pt(op);
                        for &(m_to, new_pt) in &best2 {
                            if m_to == NONE_USIZE || m_to >= ds.num_machines || m_to == old_m || new_pt >= INF { continue; }
                            let w_to = pre.machine_weight[m_to].max(1e-9);
                            if !(new_pt + 1 < old_pt || w_to < w_from * 0.90) { continue; }
                            let desired = buf.start[node];
                            let pos0 = find_insert_pos_by_start(&ds.machine_seq[m_to][..], &buf.start, desired);
                            for pos in [pos0, pos0.saturating_add(1)] {
                                if pos > ds.machine_seq[m_to].len() { continue; }
                                let diffw = ((w_from - w_to).max(0.0) * pre.avg_op_min).max(0.0) as u32;
                                let difpt = old_pt.saturating_sub(new_pt);
                                let score = desired.saturating_add(old_pt).saturating_add(diffw).saturating_add(difpt.saturating_mul(2));
                                push_top_k_move(&mut cands, MoveCand { kind: 1, m_from: old_m, from: idx, m_to, to: pos, new_pt, score }, top_cands);
                            }
                        }
                    }
                }
                i = bend + 1;
            }
        }
        if cands.is_empty() { break; }
        let mut best_cand: Option<MoveCand> = None; let mut best_mk = cur_mk;
        // The shift list emits literally duplicated probes on short blocks (up to six full
        // evaluations for one unique swap); a duplicate can never win, the strict comparison
        // already lowered the incumbent to its own value.
        seen_probes.clear();
        for cand in &cands {
            let key = (cand.kind, cand.m_from, cand.from, cand.m_to, cand.to, cand.new_pt);
            if seen_probes.iter().any(|&k| k == key) { continue; }
            seen_probes.push(key);
            if cand.kind == 0 {
                let m = cand.m_from; if m >= ds.num_machines || cand.from >= ds.machine_seq[m].len() { continue; }
                let new_idx = apply_insert(&mut ds.machine_seq[m], cand.from, cand.to);
                if let Some(mk2) = eval_disj_trial(ds, buf, best_mk) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                let _ = apply_insert(&mut ds.machine_seq[m], new_idx, cand.from);
            } else if cand.kind == 2 {
                let m = cand.m_from; if m >= ds.num_machines || cand.from + 1 >= ds.machine_seq[m].len() { continue; }
                if !apply_swap(&mut ds.machine_seq[m], cand.from) { continue; }
                if let Some(mk2) = eval_disj_trial(ds, buf, best_mk) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                let _ = apply_swap(&mut ds.machine_seq[m], cand.from);
            } else {
                let m_from = cand.m_from; let m_to = cand.m_to;
                if m_from >= ds.num_machines || m_to >= ds.num_machines || cand.from >= ds.machine_seq[m_from].len() { continue; }
                let node = ds.machine_seq[m_from][cand.from]; if ds.node_machine[node] != m_from { continue; }
                if let Some((node2, old_pt, ins_idx)) = apply_reroute(ds, m_from, cand.from, m_to, cand.to, cand.new_pt) {
                    if let Some(mk2) = eval_disj_trial(ds, buf, best_mk) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                    let _ = undo_reroute(ds, m_from, cand.from, m_to, ins_idx, node2, old_pt);
                }
            }
        }
        let Some(bc) = best_cand else { break };
        let mut accepted = false;
        if bc.kind == 0 {
            let m = bc.m_from; let new_idx = apply_insert(&mut ds.machine_seq[m], bc.from, bc.to);
            if let Some(ne) = eval_disj(ds, buf) { if ne.0 < cur_mk { *cur_eval = ne; cur_mk = ne.0; improved = true; accepted = true; } else { let _ = apply_insert(&mut ds.machine_seq[m], new_idx, bc.from); } }
            else { let _ = apply_insert(&mut ds.machine_seq[m], new_idx, bc.from); }
        } else if bc.kind == 2 {
            let m = bc.m_from;
            if m < ds.num_machines && bc.from + 1 < ds.machine_seq[m].len() {
                if apply_swap(&mut ds.machine_seq[m], bc.from) {
                    if let Some(ne) = eval_disj(ds, buf) { if ne.0 < cur_mk { *cur_eval = ne; cur_mk = ne.0; improved = true; accepted = true; } else { let _ = apply_swap(&mut ds.machine_seq[m], bc.from); } }
                    else { let _ = apply_swap(&mut ds.machine_seq[m], bc.from); }
                }
            }
        } else {
            if let Some((node2, old_pt, ins_idx)) = apply_reroute(ds, bc.m_from, bc.from, bc.m_to, bc.to, bc.new_pt) {
                if let Some(ne) = eval_disj(ds, buf) { if ne.0 < cur_mk { *cur_eval = ne; cur_mk = ne.0; improved = true; accepted = true; } else { let _ = undo_reroute(ds, bc.m_from, bc.from, bc.m_to, ins_idx, node2, old_pt); } }
                else { let _ = undo_reroute(ds, bc.m_from, bc.from, bc.m_to, ins_idx, node2, old_pt); }
            }
        }
        if !accepted { break; }
    }
    improved
}

fn descent_phase_fjsp(
    ds: &mut DisjSchedule, buf: &mut EvalBuf, crit: &mut Vec<bool>, pre: &Pre,
    cur_eval: &mut (u32, usize), max_iters: usize, top_cands: usize,
) -> bool {
    let mut cur_mk = cur_eval.0; let mut improved = false;
    let mut seen_probes: Vec<(u8, usize, usize, usize, usize, u32)> = Vec::with_capacity(64);
    for _iter in 0..max_iters {
        crit.fill(false); let mut u = cur_eval.1; while u != NONE_USIZE { crit[u] = true; u = buf.best_pred[u]; }
        let mut m_load = vec![0u32; ds.num_machines];
        for m in 0..ds.num_machines {
            if let Some(&last) = ds.machine_seq[m].last() {
                m_load[m] = buf.start[last].saturating_add(ds.node_pt[last]);
            }
        }
        let mut cands: Vec<MoveCand> = Vec::with_capacity(top_cands.min(64));
        for m in 0..ds.num_machines {
            let seq = &ds.machine_seq[m]; if seq.len() <= 1 { continue; }
            let mut i = 0usize;
            while i < seq.len() {
                let a = seq[i]; if !crit[a] { i += 1; continue; }
                let bstart = i; let mut bend = i;
                while bend + 1 < seq.len() { let x = seq[bend]; let y = seq[bend+1]; if !crit[y] { break; } if buf.start[y] != buf.start[x].saturating_add(ds.node_pt[x]) { break; } bend += 1; }
                if bend > bstart {
                    if bstart > 0 { push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bstart-1, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bstart]] }, top_cands); }
                    if bend + 1 < seq.len() { push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bend, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bend]] }, top_cands); }
                    if bstart + 1 <= bend {
                        push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bstart, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bstart+1]] }, top_cands);
                        if bend >= 1 && bend - 1 >= bstart { push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bend-1, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bend]] }, top_cands); }
                    }
                    let max_shift = bend - bstart;
                    if max_shift > 1 {
                        let to_after = bstart + max_shift;
                        if to_after <= seq.len() { let tgt_idx = (bstart+max_shift).min(seq.len()-1); push_top_k_move(&mut cands, MoveCand { kind: 0, m_from: m, from: bstart, m_to: m, to: to_after, new_pt: 0, score: buf.start[seq[tgt_idx]] }, top_cands); }
                        let to_after_bend = bend - max_shift;
                        push_top_k_move(&mut cands, MoveCand { kind: 0, m_from: m, from: bend, m_to: m, to: to_after_bend, new_pt: 0, score: buf.start[seq[bend]] }, top_cands);
                    }
                    for &idx in &[bstart, bend] {
                        if idx >= seq.len() { continue; }
                        let node = seq[idx]; if !crit[node] { continue; }
                        let job = ds.node_job[node]; let op_idx = ds.node_op[node]; let product = pre.job_products[job];
                        let op = &pre.product_ops[product][op_idx];
                        if op.flex < 2 || op.machines.len() < 2 { continue; }
                        let old_m = ds.node_machine[node]; let old_pt = ds.node_pt[node];
                        let w_from = pre.machine_weight[old_m].max(1e-9);
                        let desired = buf.start[node];
                        let mut best = (NONE_USIZE, INF, u64::MAX);
                        let mut second = (NONE_USIZE, INF, u64::MAX);
                        for &(m_to, pt) in &op.machines {
                            let end_time = m_load.get(m_to).copied().unwrap_or(INF).max(desired).saturating_add(pt);
                            let cost = (end_time as u64).saturating_mul(10000).saturating_add(pt as u64);
                            if cost < best.2 { second = best; best = (m_to, pt, cost); }
                            else if cost < second.2 { second = (m_to, pt, cost); }
                        }
                        let best2 = [(best.0, best.1), (second.0, second.1)];
                        for &(m_to, new_pt) in &best2 {
                            if m_to == NONE_USIZE || m_to >= ds.num_machines || m_to == old_m || new_pt >= INF { continue; }
                            let w_to = pre.machine_weight[m_to].max(1e-9);
                            if !(new_pt + 1 < old_pt || w_to < w_from * 0.90) { continue; }
                            let pos0 = find_insert_pos_by_start(&ds.machine_seq[m_to][..], &buf.start, desired);
                            for pos in [pos0, pos0.saturating_add(1)] {
                                if pos > ds.machine_seq[m_to].len() { continue; }
                                let diffw = ((w_from - w_to).max(0.0) * pre.avg_op_min).max(0.0) as u32;
                                let difpt = old_pt.saturating_sub(new_pt);
                                let score = desired.saturating_add(old_pt).saturating_add(diffw).saturating_add(difpt.saturating_mul(2));
                                push_top_k_move(&mut cands, MoveCand { kind: 1, m_from: old_m, from: idx, m_to, to: pos, new_pt, score }, top_cands);
                            }
                        }
                    }
                }
                i = bend + 1;
            }
        }
        if cands.is_empty() { break; }
        let mut best_cand: Option<MoveCand> = None; let mut best_mk = cur_mk;
        // The shift list emits literally duplicated probes on short blocks (up to six full
        // evaluations for one unique swap); a duplicate can never win, the strict comparison
        // already lowered the incumbent to its own value.
        seen_probes.clear();
        for cand in &cands {
            let key = (cand.kind, cand.m_from, cand.from, cand.m_to, cand.to, cand.new_pt);
            if seen_probes.iter().any(|&k| k == key) { continue; }
            seen_probes.push(key);
            if cand.kind == 0 {
                let m = cand.m_from; if m >= ds.num_machines || cand.from >= ds.machine_seq[m].len() { continue; }
                let new_idx = apply_insert(&mut ds.machine_seq[m], cand.from, cand.to);
                if let Some(mk2) = eval_disj_trial(ds, buf, best_mk) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                let _ = apply_insert(&mut ds.machine_seq[m], new_idx, cand.from);
            } else if cand.kind == 2 {
                let m = cand.m_from; if m >= ds.num_machines || cand.from + 1 >= ds.machine_seq[m].len() { continue; }
                if !apply_swap(&mut ds.machine_seq[m], cand.from) { continue; }
                if let Some(mk2) = eval_disj_trial(ds, buf, best_mk) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                let _ = apply_swap(&mut ds.machine_seq[m], cand.from);
            } else {
                let m_from = cand.m_from; let m_to = cand.m_to;
                if m_from >= ds.num_machines || m_to >= ds.num_machines || cand.from >= ds.machine_seq[m_from].len() { continue; }
                let node = ds.machine_seq[m_from][cand.from]; if ds.node_machine[node] != m_from { continue; }
                if let Some((node2, old_pt, ins_idx)) = apply_reroute(ds, m_from, cand.from, m_to, cand.to, cand.new_pt) {
                    if let Some(mk2) = eval_disj_trial(ds, buf, best_mk) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                    let _ = undo_reroute(ds, m_from, cand.from, m_to, ins_idx, node2, old_pt);
                }
            }
        }
        let Some(bc) = best_cand else { break };
        let mut accepted = false;
        if bc.kind == 0 {
            let m = bc.m_from; let new_idx = apply_insert(&mut ds.machine_seq[m], bc.from, bc.to);
            if let Some(ne) = eval_disj(ds, buf) { if ne.0 < cur_mk { *cur_eval = ne; cur_mk = ne.0; improved = true; accepted = true; } else { let _ = apply_insert(&mut ds.machine_seq[m], new_idx, bc.from); } }
            else { let _ = apply_insert(&mut ds.machine_seq[m], new_idx, bc.from); }
        } else if bc.kind == 2 {
            let m = bc.m_from;
            if m < ds.num_machines && bc.from + 1 < ds.machine_seq[m].len() {
                if apply_swap(&mut ds.machine_seq[m], bc.from) {
                    if let Some(ne) = eval_disj(ds, buf) { if ne.0 < cur_mk { *cur_eval = ne; cur_mk = ne.0; improved = true; accepted = true; } else { let _ = apply_swap(&mut ds.machine_seq[m], bc.from); } }
                    else { let _ = apply_swap(&mut ds.machine_seq[m], bc.from); }
                }
            }
        } else {
            if let Some((node2, old_pt, ins_idx)) = apply_reroute(ds, bc.m_from, bc.from, bc.m_to, bc.to, bc.new_pt) {
                if let Some(ne) = eval_disj(ds, buf) { if ne.0 < cur_mk { *cur_eval = ne; cur_mk = ne.0; improved = true; accepted = true; } else { let _ = undo_reroute(ds, bc.m_from, bc.from, bc.m_to, ins_idx, node2, old_pt); } }
                else { let _ = undo_reroute(ds, bc.m_from, bc.from, bc.m_to, ins_idx, node2, old_pt); }
            }
        }
        if !accepted { break; }
    }
    improved
}

#[inline]
pub fn apply_insert(seq: &mut Vec<usize>, from: usize, to_after_removal: usize) -> usize {
    let len = seq.len();
    if len == 0 || from >= len { return from.min(len.saturating_sub(1)); }
    let t = to_after_removal.min(len - 1);
    if t > from {
        seq[from..=t].rotate_left(1);
    } else if t < from {
        seq[t..=from].rotate_right(1);
    }
    t
}

#[inline]
pub fn apply_swap(seq: &mut [usize], i: usize) -> bool {
    if i + 1 >= seq.len() { return false; } seq.swap(i, i + 1); true
}

#[inline]
pub fn find_insert_pos_by_start(seq: &[usize], start: &[u32], desired_start: u32) -> usize {
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

#[inline]
pub fn apply_reroute(ds: &mut DisjSchedule, m_from: usize, idx_from: usize, m_to: usize, idx_to: usize, new_pt: u32) -> Option<(usize, u32, usize)> {
    if m_from >= ds.num_machines || m_to >= ds.num_machines || idx_from >= ds.machine_seq[m_from].len() { return None; }
    let node = ds.machine_seq[m_from].remove(idx_from); let old_pt = ds.node_pt[node];
    ds.node_machine[node] = m_to; ds.node_pt[node] = new_pt;
    let ins = idx_to.min(ds.machine_seq[m_to].len()); ds.machine_seq[m_to].insert(ins, node);
    Some((node, old_pt, ins))
}

#[inline]
pub fn undo_reroute(ds: &mut DisjSchedule, m_from: usize, idx_from: usize, m_to: usize, ins_idx: usize, node: usize, old_pt: u32) -> bool {
    if m_from >= ds.num_machines || m_to >= ds.num_machines || ins_idx >= ds.machine_seq[m_to].len() { return false; }
    let x = ds.machine_seq[m_to].remove(ins_idx);
    if x != node { let len_now = ds.machine_seq[m_to].len(); ds.machine_seq[m_to].insert(ins_idx.min(len_now), x); return false; }
    let ins_from = idx_from.min(ds.machine_seq[m_from].len());
    ds.machine_seq[m_from].insert(ins_from, node);
    ds.node_machine[node] = m_from; ds.node_pt[node] = old_pt; true
}

#[inline]
pub fn push_top_k_move(top: &mut Vec<MoveCand>, c: MoveCand, k: usize) {
    if k == 0 { return; }
    let mut pos = top.len(); while pos > 0 && top[pos-1].score < c.score { pos -= 1; }
    if pos >= k { return; } top.insert(pos, c); if top.len() > k { top.pop(); }
}

pub fn best_two_by_pt(op: &OpInfo) -> [(usize, u32); 2] {
    let mut r = [(NONE_USIZE, INF); 2];
    for &(m, pt) in &op.machines {
        if pt < r[0].1 { r[1] = r[0]; r[0] = (m, pt); }
        else if pt < r[1].1 { r[1] = (m, pt); }
    }
    r
}

pub fn job_bias_from_solution(pre: &Pre, sol: &Solution) -> Result<Vec<f64>> {
    let num_jobs = pre.job_ops_len.len();
    let mut completion = vec![0u32; num_jobs];
    let mut makespan = 0u32;

    for job in 0..num_jobs {
        let product = pre.job_products[job];
        let mut end_j = 0u32;
        for (op_idx, &(m, st)) in sol.job_schedule[job].iter().enumerate() {
            let op = &pre.product_ops[product][op_idx];
            let pt = pt_from_op(op, m).ok_or_else(|| anyhow!("Missing pt in bias calc"))?;
            end_j = end_j.max(st.saturating_add(pt));
        }
        completion[job] = end_j;
        makespan = makespan.max(end_j);
    }

    let denom = (makespan as f64).max(1.0);
    let exp = 3.0 + 1.2 * pre.high_flex + 0.6 * pre.jobshopness;
    Ok(completion.into_iter().map(|c| ((c as f64) / denom).powf(exp).clamp(0.0, 1.0)).collect())
}

pub fn machine_penalty_from_solution(pre: &Pre, sol: &Solution, num_machines: usize) -> Result<Vec<f64>> {
    let num_jobs = pre.job_ops_len.len();
    let mut m_end = vec![0u32; num_machines];
    let mut m_sum = vec![0u64; num_machines];
    let mut makespan = 0u32;

    for job in 0..num_jobs {
        let product = pre.job_products[job];
        for (op_idx, &(m, st)) in sol.job_schedule[job].iter().enumerate() {
            let op = &pre.product_ops[product][op_idx];
            let pt = pt_from_op(op, m).ok_or_else(|| anyhow!("Missing pt in machine penalty"))?;
            let end = st.saturating_add(pt);
            if end > m_end[m] { m_end[m] = end; }
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

pub fn route_pref_from_solution_lite(pre: &Pre, sol: &Solution, challenge: &Challenge) -> Result<RoutePrefLite> {
    let nm = challenge.num_machines;
    let np = challenge.product_processing_times.len();

    let mut counts: Vec<Vec<u16>> = Vec::with_capacity(np);
    let mut ops_len: Vec<usize> = Vec::with_capacity(np);
    for p in 0..np {
        let ol = challenge.product_processing_times[p].len();
        ops_len.push(ol);
        counts.push(vec![0u16; ol.saturating_mul(nm)]);
    }

    for job in 0..challenge.num_jobs {
        let product = pre.job_products[job];
        let ol = ops_len[product];
        for (op_idx, &(m, _st)) in sol.job_schedule[job].iter().enumerate() {
            if op_idx >= ol || m >= nm { continue; }
            let idx = op_idx * nm + m;
            counts[product][idx] = counts[product][idx].saturating_add(1);
        }
    }

    let mut rp: RoutePrefLite = Vec::with_capacity(np);
    for p in 0..np {
        let ol = ops_len[p];
        let denom_u32 = (challenge.jobs_per_product[p].max(1) as u32).max(1);
        let mut v: Vec<OpRoute> = Vec::with_capacity(ol);

        for op_idx in 0..ol {
            let base = op_idx * nm;
            let mut best_m = 0usize;
            let mut best_c = 0u16;
            let mut second_m = 0usize;
            let mut second_c = 0u16;

            for m in 0..nm {
                let c = counts[p][base + m];
                if c > best_c {
                    second_c = best_c; second_m = best_m;
                    best_c = c; best_m = m;
                } else if c > second_c && m != best_m {
                    second_c = c; second_m = m;
                }
            }

            let best_w = (((best_c as u32).saturating_mul(255)).saturating_add(denom_u32 / 2) / denom_u32).min(255) as u8;
            let second_w = (((second_c as u32).saturating_mul(255)).saturating_add(denom_u32 / 2) / denom_u32).min(255) as u8;

            v.push(OpRoute { best_m: best_m.min(255) as u8, best_w, second_m: second_m.min(255) as u8, second_w });
        }

        rp.push(v);
    }

    Ok(rp)
}

pub fn push_top_solutions(top: &mut Vec<(Solution, u32)>, sol: &Solution, mk: u32, cap: usize) {
    let pos = top.binary_search_by_key(&mk, |(_, m)| *m).unwrap_or_else(|e| e);
    top.insert(pos, (sol.clone(), mk));
    if top.len() > cap { top.truncate(cap); }
}

pub fn neh_reentrant_flow_solution(pre: &Pre, num_jobs: usize, num_machines: usize) -> Result<(Solution, u32)> {
    let route = pre.flow_route.as_ref().ok_or_else(|| anyhow!("No flow route"))?;
    let pt = pre.flow_pt_by_job.as_ref().ok_or_else(|| anyhow!("No flow pt"))?;
    let ops = route.len(); if ops == 0 || pt.len() != num_jobs { return Err(anyhow!("Invalid flow data")); }
    let mut jobs: Vec<usize> = (0..num_jobs).collect();
    jobs.sort_unstable_by(|&a, &b| { let sa: u32 = pt[a].iter().copied().sum(); let sb: u32 = pt[b].iter().copied().sum(); sb.cmp(&sa).then_with(|| a.cmp(&b)) });
    let mut seq: Vec<usize> = Vec::with_capacity(num_jobs); let mut mready = vec![0u32; num_machines];
    for &j in &jobs {
        if seq.is_empty() { seq.push(j); continue; }
        let mut best_mk = u32::MAX; let mut best_pos = 0usize; let mut tmp = seq.clone();
        for pos in 0..=seq.len() {
            tmp.clear(); tmp.extend_from_slice(&seq[..pos]); tmp.push(j); tmp.extend_from_slice(&seq[pos..]);
            let mk = reentrant_makespan_local(&tmp, route, pt, &mut mready);
            if mk < best_mk { best_mk = mk; best_pos = pos; }
        }
        seq.insert(best_pos, j);
    }
    let mk = reentrant_makespan_local(&seq, route, pt, &mut mready);
    let sol = build_perm_solution_from_seq_local(&seq, route, pt, num_jobs, num_machines);
    Ok((sol, mk))
}

fn reentrant_makespan_local(seq: &[usize], route: &[usize], pt: &[Vec<u32>], mready: &mut [u32]) -> u32 {
    mready.fill(0); let mut mk = 0u32;
    for &j in seq { let row = &pt[j]; let mut prev = 0u32; for (op_idx, &m) in route.iter().enumerate() { let p = row[op_idx]; let st = prev.max(mready[m]); let end = st.saturating_add(p); mready[m] = end; prev = end; } if prev > mk { mk = prev; } }
    mk
}

fn build_perm_solution_from_seq_local(seq: &[usize], route: &[usize], pt: &[Vec<u32>], num_jobs: usize, num_machines: usize) -> Solution {
    let ops = route.len(); let mut job_schedule: Vec<Vec<(usize, u32)>> = vec![Vec::with_capacity(ops); num_jobs]; let mut machine_ready = vec![0u32; num_machines];
    for &j in seq {
        if j >= num_jobs { continue; } let row = &pt[j]; let mut prev_end = 0u32;
        for (op_idx, &m) in route.iter().enumerate() {
            if op_idx >= row.len() || m >= num_machines { break; }
            let p = row[op_idx]; let st = prev_end.max(machine_ready[m]); job_schedule[j].push((m, st)); let end = st.saturating_add(p); machine_ready[m] = end; prev_end = end;
        }
    }
    Solution { job_schedule }
}

#[inline]
pub fn best_second_and_counts(time: u32, machine_avail: &[u32], op: &OpInfo) -> (u32, u32, usize, usize) {
    let mut best = INF; let mut second = INF; let mut cnt_best = 0usize; let mut cnt_best_idle = 0usize;
    for &(m, pt) in &op.machines {
        let end = time.max(machine_avail[m]).saturating_add(pt);
        if end < best { second = best; best = end; cnt_best = 1; cnt_best_idle = if machine_avail[m] <= time { 1 } else { 0 }; }
        else if end == best { cnt_best += 1; if machine_avail[m] <= time { cnt_best_idle += 1; } }
        else if end < second { second = end; }
    }
    if cnt_best > 1 { second = best; }
    (best, second, cnt_best.max(1), cnt_best_idle)
}

#[inline]
pub fn push_top_k(top: &mut Vec<Cand>, c: Cand, k: usize) {
    if k == 0 { return; }
    let mut pos = top.len(); while pos > 0 && top[pos-1].score < c.score { pos -= 1; }
    if pos >= k { return; } top.insert(pos, c); if top.len() > k { top.pop(); }
}

#[inline]
pub fn push_top_k_raw(top: &mut Vec<RawCand>, c: RawCand, k: usize) {
    if k == 0 { return; }
    let mut pos = top.len(); while pos > 0 && top[pos-1].base_score < c.base_score { pos -= 1; }
    if pos >= k { return; } top.insert(pos, c); if top.len() > k { top.pop(); }
}

#[inline]
pub fn choose_from_top_weighted(rng: &mut SmallRng, top: &[Cand]) -> Cand {
    if top.len() <= 1 { return top[0]; }
    let min_s = top.last().unwrap().score; let n = top.len().min(8);
    let mut w = [0.0f64; 8]; let mut sum = 0.0f64;
    for i in 0..n { let d = (top[i].score - min_s) + 1e-9; let wi = d * d; w[i] = wi; sum += wi; }
    if !(sum > 0.0) { return top[rng.gen_range(0..top.len())]; }
    let mut r = rng.gen::<f64>() * sum;
    for i in 0..n { r -= w[i]; if r <= 0.0 { return top[i]; } }
    top[n - 1]
}}
pub mod preprocess {
use anyhow::{anyhow, Result};
use tig_challenges::job_scheduling::*;
use super::types::*;

#[inline]
fn flow_makespan(seq: &[usize], pt: &[Vec<u32>], comp: &mut [u32]) -> u32 {
    comp.fill(0);
    for &j in seq {
        let row = &pt[j];
        if row.is_empty() { continue; }
        comp[0] = comp[0].saturating_add(row[0]);
        for k in 1..row.len() {
            let v = comp[k].max(comp[k - 1]).saturating_add(row[k]);
            comp[k] = v;
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
    let mut machine_scarcity = vec![0.0f64; num_machines];
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
                let delta_s = (oi.min_pt as f64) * cnt_f / (flex_f * flex_f);
                for &(m, _) in &oi.machines {
                    machine_load0[m] += delta;
                    machine_scarcity[m] += delta_s;
                }
            }
        }

        product_ops.push(ops_info);
        best_machine_by_product.push(bests);
    }

    let job_ops_len: Vec<usize> = job_products.iter().map(|&p| product_ops[p].len()).collect();

    let avg_machine_load = (total_min_work / (num_machines as f64).max(1.0)).max(1.0);
    let horizon = avg_machine_load;

    let avg_op_min = (total_min_work / (total_ops as f64).max(1.0)).max(1.0);
    let flex_avg = (total_flex_weighted / (total_ops as f64).max(1.0)).max(1.0);
    let flex_factor = (3.0 / flex_avg).clamp(0.6, 2.2);
    let hi_flex = flex_avg >= 5.0;
    let high_flex = ((flex_avg - 3.0) / 7.0).clamp(0.0, 1.0);

    let avg_machine_scarcity = {
        let s: f64 = machine_scarcity.iter().sum();
        (s / (num_machines as f64).max(1.0)).max(1e-9)
    };

    let load_cv = {
        let mean = avg_machine_load.max(1e-9);
        let mut var = 0.0f64;
        for &x in &machine_load0 {
            let d = (x / mean) - 1.0;
            var += d * d;
        }
        (var / (num_machines as f64)).sqrt().clamp(0.0, 2.5)
    };

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
            let w_u32 = challenge.jobs_per_product[p] as u32;
            if w_u32 == 0 {
                continue;
            }
            counts[bm] = counts[bm].saturating_add(w_u32);
            tot = tot.saturating_add(w_u32);
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
    let flow_like = if flow_cnt > 0 { (flow_sum / (flow_cnt as f64)).clamp(0.0, 1.0) } else { 0.5 };
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

    let bn_focus = ((3.0 / flex_avg).clamp(0.7, 2.6) * (1.0 + 0.55 * load_cv) * (0.85 + 0.55 * jobshopness)).clamp(0.6, 3.4);

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
                nxt_fi[i] = if next.flex > 0 { 1.0 / (next.flex as f64) } else { 0.0 };
            }
        }

        max_job_bn = max_job_bn.max(suf_bn[0]);

        product_suf_min.push(suf_m);
        product_suf_avg.push(suf_a);
        product_suf_bn.push(suf_bn);
        product_next_min.push(nxt_m);
        product_next_flex_inv.push(nxt_fi);
    }

    let time_scale = (horizon * (2.65 + 0.15 * load_cv + 0.10 * jobshopness + 0.10 * high_flex)).max(1.0);

    let mut job_flow_pref = vec![0.0f64; num_jobs];
    let use_flow_pref = flow_like > 0.82 && jobshopness < 0.38 && max_ops >= 2;

    if use_flow_pref {
        let m = max_ops.max(1);
        let mut job_pt: Vec<Vec<u32>> = Vec::with_capacity(num_jobs);
        for j in 0..num_jobs {
            let p = job_products[j];
            let ops = &product_ops[p];
            let mut v = vec![0u32; m];
            for s in 0..m.min(ops.len()) {
                v[s] = ops[s].min_pt.min(INF / 2);
            }
            job_pt.push(v);
        }

        let mut jobs2: Vec<usize> = (0..num_jobs).collect();
        jobs2.sort_unstable_by(|&a, &b| {
            let sa: u32 = job_pt[a].iter().copied().sum();
            let sb: u32 = job_pt[b].iter().copied().sum();
            sb.cmp(&sa).then_with(|| a.cmp(&b))
        });

        let mut perm: Vec<usize> = Vec::with_capacity(num_jobs);
        let mut comp = vec![0u32; m];
        let mut tmp: Vec<usize> = Vec::with_capacity(num_jobs);

        for &j in &jobs2 {
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
            job_flow_pref[j] = if n1 > 0.0 { 1.0 - (pos as f64) / n1 } else { 1.0 };
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

    let slack_base = (0.04 + 0.14 * jobshopness + 0.11 * high_flex).clamp(0.03, 0.22);

    let mut flow_route: Option<Vec<usize>> = None;
    let mut flow_pt_by_job: Option<Vec<Vec<u32>>> = None;
    let mut strict_route: Option<Vec<usize>> = None;
    if !product_ops.is_empty() {
        let common_len = product_ops[0].len();
        let mut ok = common_len > 0;

        for ops in &product_ops {
            if ops.len() != common_len {
                ok = false;
                break;
            }
        }

        if ok && flex_avg <= 1.25 {
            let mut route: Vec<usize> = Vec::with_capacity(common_len);
            for i in 0..common_len {
                let mut m0: Option<usize> = None;
                for p in 0..num_products {
                    let op = &product_ops[p][i];
                    if op.flex != 1 || op.machines.len() != 1 {
                        ok = false;
                        break;
                    }
                    let mid = op.machines[0].0;
                    if let Some(mm) = m0 {
                        if mm != mid {
                            ok = false;
                            break;
                        }
                    } else {
                        m0 = Some(mid);
                    }
                }
                if !ok {
                    break;
                }
                route.push(m0.unwrap());
            }

            if ok {
                strict_route = Some(route.clone());

                let mut pt_by_job: Vec<Vec<u32>> = Vec::with_capacity(num_jobs);
                for j in 0..num_jobs {
                    let prod = job_products[j];
                    let mut row = Vec::with_capacity(common_len);
                    for i in 0..common_len {
                        row.push(product_ops[prod][i].machines[0].1);
                    }
                    pt_by_job.push(row);
                }
                flow_route = Some(route);
                flow_pt_by_job = Some(pt_by_job);
            }
        }
    }

    let chaotic_like = high_flex > 0.85 && jobshopness > 0.75;

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
        machine_scarcity,
        machine_weight,
        machine_best_pop,
        avg_machine_load,
        avg_machine_scarcity,
        avg_op_min,
        horizon,
        time_scale,
        max_ops: max_ops.max(1),
        max_job_avg_work: max_job_avg_work.max(1.0),
        max_job_bn: max_job_bn.max(1e-9),
        flex_avg,
        flex_factor,
        hi_flex,
        high_flex,
        flow_like,
        flow_w,
        job_flow_pref,
        jobshopness,
        bn_focus,
        load_cv,
        slack_base,
        total_ops,
        chaotic_like,
        flow_route,
        flow_pt_by_job,
        strict_route,
    })
}
}

pub mod job_shop {
use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::job_scheduling::*;
use super::types::*;
use super::infra_shared::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rule {
    Adaptive, BnHeavy, EndTight, CriticalPath, MostWork, LeastFlex, Regret, ShortestProc, FlexBalance,
}

#[inline]
fn slack_urgency_js(pre: &Pre, target_mk: Option<u32>, time: u32, product: usize, op_idx: usize) -> f64 {
    let Some(tgt) = target_mk else { return 0.0 };
    let lb = (time as u64).saturating_add(pre.product_suf_min[product][op_idx] as u64);
    let slack = (tgt as i64) - (lb as i64);
    let scale = (0.70 * pre.avg_op_min).max(1.0);
    let pos = (slack.max(0) as f64) / scale; let neg = ((-slack).max(0) as f64) / scale;
    (1.0 / (1.0 + pos)).clamp(0.0, 1.0) + (0.35 * neg).min(3.0)
}

#[inline]
fn route_pref_bonus_js(rp: Option<&RoutePrefLite>, product: usize, op_idx: usize, machine: usize) -> f64 {
    let Some(rp) = rp else { return 0.0 };
    if product >= rp.len() || op_idx >= rp[product].len() { return 0.0; }
    let r = rp[product][op_idx]; let mu = js_cached_machine_u8(machine);
    if mu == r.best_m { (r.best_w as f64) / 255.0 } else if mu == r.second_m { (r.second_w as f64) / 255.0 } else { 0.0 }
}

#[inline(always)]
fn progress_mode_js(progress: f64) -> u8 {
    if progress < 0.34 { 0 } else if progress < 0.72 { 1 } else { 2 }
}

static mut JS_SCAR_N_PTR: *const f64 = std::ptr::null();
static mut JS_SCAR_N_LEN: usize = 0;
static mut JS_POP_PTR: *const f64 = std::ptr::null();
static mut JS_POP_LEN: usize = 0;
static mut JS_PEN_PTR: *const f64 = std::ptr::null();
static mut JS_PEN_LEN: usize = 0;
static mut JS_U8_PTR: *const u8 = std::ptr::null();
static mut JS_U8_LEN: usize = 0;

#[inline(always)]
unsafe fn set_js_machine_static_cache(scar_n: &[f64], pop: &[f64], pen: &[f64], mu8: &[u8]) {
    JS_SCAR_N_PTR = scar_n.as_ptr();
    JS_SCAR_N_LEN = scar_n.len();
    JS_POP_PTR = pop.as_ptr();
    JS_POP_LEN = pop.len();
    JS_PEN_PTR = pen.as_ptr();
    JS_PEN_LEN = pen.len();
    JS_U8_PTR = mu8.as_ptr();
    JS_U8_LEN = mu8.len();
}

#[inline(always)]
unsafe fn clear_js_machine_static_cache() {
    JS_SCAR_N_PTR = std::ptr::null();
    JS_SCAR_N_LEN = 0;
    JS_POP_PTR = std::ptr::null();
    JS_POP_LEN = 0;
    JS_PEN_PTR = std::ptr::null();
    JS_PEN_LEN = 0;
    JS_U8_PTR = std::ptr::null();
    JS_U8_LEN = 0;
}

struct JsMachineStaticCacheGuard;

impl Drop for JsMachineStaticCacheGuard {
    fn drop(&mut self) {
        unsafe { clear_js_machine_static_cache(); }
    }
}

#[inline(always)]
fn js_cached_scar_n(pre: &Pre, machine: usize) -> f64 {
    unsafe {
        if machine < JS_SCAR_N_LEN {
            *JS_SCAR_N_PTR.add(machine)
        } else {
            pre.machine_scarcity[machine] / pre.avg_machine_scarcity.max(1e-9)
        }
    }
}

#[inline(always)]
fn js_cached_pop(pre: &Pre, machine: usize) -> f64 {
    unsafe {
        if machine < JS_POP_LEN {
            *JS_POP_PTR.add(machine)
        } else {
            pre.machine_best_pop[machine]
        }
    }
}

#[inline(always)]
fn js_cached_penalty(machine: usize, machine_penalty: f64) -> f64 {
    unsafe {
        if machine < JS_PEN_LEN {
            *JS_PEN_PTR.add(machine)
        } else {
            machine_penalty.clamp(0.0, 1.0)
        }
    }
}

#[inline(always)]
fn js_cached_machine_u8(machine: usize) -> u8 {
    unsafe {
        if machine < JS_U8_LEN {
            *JS_U8_PTR.add(machine)
        } else {
            machine.min(255) as u8
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn score_candidate_critical_path(
    pre: &Pre, job: usize, product: usize, op_idx: usize,
    ops_rem: usize, op: &OpInfo, machine: usize, _pt: u32, time: u32,
    target_mk: Option<u32>, best_end: u32, second_end: u32, best_cnt_total: usize,
    progress: f64, job_bias: f64, _machine_penalty: f64, _dynamic_load: f64,
    route_pref: Option<&RoutePrefLite>, route_w: f64, jitter: f64,
) -> f64 {
    let rem_min = pre.product_suf_min[product][op_idx] as f64;
    let flex_f = (op.flex as f64).max(1.0); let flex_inv = 1.0 / flex_f;
    let rem_min_n = rem_min / pre.horizon.max(1.0); let ops_n = (ops_rem as f64) / (pre.max_ops as f64).max(1.0);
    let end_n = (best_end as f64) / pre.time_scale.max(1.0);
    let regret = if second_end >= INF { pre.avg_op_min * 2.6 } else { (second_end - best_end) as f64 };
    let reg_n = (regret / pre.avg_op_min.max(1.0)).clamp(0.0, 6.0);
    let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
    let density_n = ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0);
    let next_min = pre.product_next_min[product][op_idx] as f64; let next_min_n = next_min / pre.horizon.max(1.0);
    let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
    let p2 = progress * progress; let next_w_base = 0.12 + p2 * 0.28;
    let next_term_raw = (0.55*next_min_n + 0.45*next_flex_inv) * (1.0 + 0.30*density_n*pre.high_flex);
    let flow_term = pre.flow_w * pre.job_flow_pref[job] * (0.65 + 0.70*(1.0-progress));
    let slack_u = slack_urgency_js(pre, target_mk, time, product, op_idx);
    let slack_w = pre.slack_base * (0.25 + 0.75*progress); let slack_reg_boost = 1.0 + 0.40*reg_n*progress;
    let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop = js_cached_pop(pre, machine); (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
    let route_gain = (0.70 + 0.80*(1.0-progress)).clamp(0.70, 1.40);
    let route_term = if route_w > 0.0 && op.flex >= 2 { route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine) } else { 0.0 };
    let next_term = next_w_base*0.30*next_term_raw; let slack_term = slack_w*slack_u*slack_reg_boost;
    (1.03*rem_min_n)+(0.10*ops_n)+(0.24*scarcity_urg)+(0.20*pre.flex_factor)*flex_inv+next_term+0.10*slack_term-(0.70*end_n)-pop_pen+(0.45*job_bias)+flow_term+route_term+jitter
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn score_candidate_most_work(
    pre: &Pre, job: usize, product: usize, op_idx: usize,
    ops_rem: usize, op: &OpInfo, machine: usize, _pt: u32, _time: u32,
    _target_mk: Option<u32>, best_end: u32, _second_end: u32, best_cnt_total: usize,
    progress: f64, job_bias: f64, _machine_penalty: f64, _dynamic_load: f64,
    route_pref: Option<&RoutePrefLite>, route_w: f64, jitter: f64,
) -> f64 {
    let rem_min = pre.product_suf_min[product][op_idx] as f64;
    let rem_avg = pre.product_suf_avg[product][op_idx];
    let flex_f = (op.flex as f64).max(1.0); let flex_inv = 1.0 / flex_f;
    let rem_avg_n = rem_avg / pre.max_job_avg_work.max(1e-9); let ops_n = (ops_rem as f64) / (pre.max_ops as f64).max(1.0);
    let end_n = (best_end as f64) / pre.time_scale.max(1.0);
    let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
    let density_n = ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0);
    let next_min = pre.product_next_min[product][op_idx] as f64; let next_min_n = next_min / pre.horizon.max(1.0);
    let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
    let p2 = progress * progress; let next_w_base = 0.12 + p2 * 0.28;
    let next_term_raw = (0.55*next_min_n + 0.45*next_flex_inv) * (1.0 + 0.30*density_n*pre.high_flex);
    let flow_term = pre.flow_w * pre.job_flow_pref[job] * (0.65 + 0.70*(1.0-progress));
    let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop = js_cached_pop(pre, machine); (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
    let route_gain = (0.70 + 0.80*(1.0-progress)).clamp(0.70, 1.40);
    let route_term = if route_w > 0.0 && op.flex >= 2 { route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine) } else { 0.0 };
    let next_term = next_w_base*0.25*next_term_raw;
    (1.00*rem_avg_n)+(0.12*ops_n)+(0.18*scarcity_urg)+(0.15*pre.flex_factor)*flex_inv+next_term-(0.62*end_n)-pop_pen+(0.45*job_bias)+flow_term+route_term+jitter
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn score_candidate_least_flex(
    pre: &Pre, job: usize, product: usize, op_idx: usize,
    ops_rem: usize, op: &OpInfo, machine: usize, _pt: u32, _time: u32,
    _target_mk: Option<u32>, best_end: u32, _second_end: u32, best_cnt_total: usize,
    progress: f64, job_bias: f64, _machine_penalty: f64, _dynamic_load: f64,
    route_pref: Option<&RoutePrefLite>, route_w: f64, jitter: f64,
) -> f64 {
    let rem_min = pre.product_suf_min[product][op_idx] as f64;
    let flex_f = (op.flex as f64).max(1.0); let flex_inv = 1.0 / flex_f;
    let rem_min_n = rem_min / pre.horizon.max(1.0);
    let end_n = (best_end as f64) / pre.time_scale.max(1.0);
    let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
    let density_n = ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0);
    let next_min = pre.product_next_min[product][op_idx] as f64; let next_min_n = next_min / pre.horizon.max(1.0);
    let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
    let p2 = progress * progress; let next_w_base = 0.12 + p2 * 0.28;
    let next_term_raw = (0.55*next_min_n + 0.45*next_flex_inv) * (1.0 + 0.30*density_n*pre.high_flex);
    let flow_term = pre.flow_w * pre.job_flow_pref[job] * (0.65 + 0.70*(1.0-progress));
    let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop = js_cached_pop(pre, machine); (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
    let route_gain = (0.70 + 0.80*(1.0-progress)).clamp(0.70, 1.40);
    let route_term = if route_w > 0.0 && op.flex >= 2 { route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine) } else { 0.0 };
    let next_term = next_w_base*0.20*next_term_raw;
    (1.00*flex_inv)+(0.28*rem_min_n)+(0.22*scarcity_urg)+next_term-(0.55*end_n)-pop_pen+(0.35*job_bias)+flow_term+route_term+jitter
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn score_candidate_shortest_proc(
    pre: &Pre, job: usize, product: usize, op_idx: usize,
    ops_rem: usize, op: &OpInfo, machine: usize, pt: u32, _time: u32,
    _target_mk: Option<u32>, best_end: u32, _second_end: u32, best_cnt_total: usize,
    progress: f64, job_bias: f64, _machine_penalty: f64, _dynamic_load: f64,
    route_pref: Option<&RoutePrefLite>, route_w: f64, jitter: f64,
) -> f64 {
    let rem_min = pre.product_suf_min[product][op_idx] as f64;
    let rem_min_n = rem_min / pre.horizon.max(1.0);
    let end_n = (best_end as f64) / pre.time_scale.max(1.0); let proc_n = (pt as f64) / pre.avg_op_min.max(1.0);
    let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
    let density_n = ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0);
    let next_min = pre.product_next_min[product][op_idx] as f64; let next_min_n = next_min / pre.horizon.max(1.0);
    let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
    let p2 = progress * progress; let next_w_base = 0.12 + p2 * 0.28;
    let next_term_raw = (0.55*next_min_n + 0.45*next_flex_inv) * (1.0 + 0.30*density_n*pre.high_flex);
    let flow_term = pre.flow_w * pre.job_flow_pref[job] * (0.65 + 0.70*(1.0-progress));
    let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop = js_cached_pop(pre, machine); (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
    let route_gain = (0.70 + 0.80*(1.0-progress)).clamp(0.70, 1.40);
    let route_term = if route_w > 0.0 && op.flex >= 2 { route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine) } else { 0.0 };
    let next_term = next_w_base*0.20*next_term_raw;
    (-1.00*proc_n)+(0.25*rem_min_n)+(0.12*scarcity_urg)+next_term-(0.20*end_n)-pop_pen+(0.25*job_bias)+flow_term+route_term+jitter
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn score_candidate_regret(
    pre: &Pre, job: usize, product: usize, op_idx: usize,
    ops_rem: usize, op: &OpInfo, machine: usize, _pt: u32, _time: u32,
    _target_mk: Option<u32>, best_end: u32, second_end: u32, best_cnt_total: usize,
    progress: f64, job_bias: f64, _machine_penalty: f64, _dynamic_load: f64,
    route_pref: Option<&RoutePrefLite>, route_w: f64, jitter: f64,
) -> f64 {
    let rem_min = pre.product_suf_min[product][op_idx] as f64;
    let rem_min_n = rem_min / pre.horizon.max(1.0);
    let end_n = (best_end as f64) / pre.time_scale.max(1.0);
    let regret = if second_end >= INF { pre.avg_op_min * 2.6 } else { (second_end - best_end) as f64 };
    let reg_n = (regret / pre.avg_op_min.max(1.0)).clamp(0.0, 6.0);
    let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
    let density_n = ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0);
    let next_min = pre.product_next_min[product][op_idx] as f64; let next_min_n = next_min / pre.horizon.max(1.0);
    let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
    let p2 = progress * progress; let next_w_base = 0.12 + p2 * 0.28;
    let next_term_raw = (0.55*next_min_n + 0.45*next_flex_inv) * (1.0 + 0.30*density_n*pre.high_flex);
    let flow_term = pre.flow_w * pre.job_flow_pref[job] * (0.65 + 0.70*(1.0-progress));
    let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop = js_cached_pop(pre, machine); (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
    let route_gain = (0.70 + 0.80*(1.0-progress)).clamp(0.70, 1.40);
    let route_term = if route_w > 0.0 && op.flex >= 2 { route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine) } else { 0.0 };
    let next_term = next_w_base*0.25*next_term_raw;
    (1.05*reg_n)+(0.55*rem_min_n)+(0.22*scarcity_urg)+next_term-(0.68*end_n)-pop_pen+(0.35*job_bias)+flow_term+route_term+jitter
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn score_candidate_end_tight(
    pre: &Pre, job: usize, product: usize, op_idx: usize,
    ops_rem: usize, op: &OpInfo, machine: usize, pt: u32, time: u32,
    target_mk: Option<u32>, best_end: u32, second_end: u32, best_cnt_total: usize,
    progress: f64, job_bias: f64, machine_penalty: f64, _dynamic_load: f64,
    route_pref: Option<&RoutePrefLite>, route_w: f64, jitter: f64,
) -> f64 {
    let rem_min = pre.product_suf_min[product][op_idx] as f64;
    let rem_avg = pre.product_suf_avg[product][op_idx];
    let flex_f = (op.flex as f64).max(1.0); let flex_inv = 1.0 / flex_f;
    let rem_min_n = rem_min / pre.horizon.max(1.0); let rem_avg_n = rem_avg / pre.max_job_avg_work.max(1e-9);
    let ops_n = (ops_rem as f64) / (pre.max_ops as f64).max(1.0);
    let scar_n = js_cached_scar_n(pre, machine);
    let end_n = (best_end as f64) / pre.time_scale.max(1.0); let proc_n = (pt as f64) / pre.avg_op_min.max(1.0);
    let regret = if second_end >= INF { pre.avg_op_min * 2.6 } else { (second_end - best_end) as f64 };
    let reg_n = (regret / pre.avg_op_min.max(1.0)).clamp(0.0, 6.0);
    let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
    let density_n = ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0);
    let next_min = pre.product_next_min[product][op_idx] as f64; let next_min_n = next_min / pre.horizon.max(1.0);
    let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
    let next_term_raw = (0.55*next_min_n + 0.45*next_flex_inv) * (1.0 + 0.30*density_n*pre.high_flex);
    let js = pre.jobshopness;
    let avg_flex_inv = 1.0 / pre.flex_avg.max(1.0); let scarce_match = scar_n * (flex_inv - avg_flex_inv);
    let mpen = js_cached_penalty(machine, machine_penalty); let mpen_gain = 1.0 + 0.85*pre.high_flex;
    let phase = progress_mode_js(progress);
    let (cp_w, phase_reg_w, next_w, end_w, flow_gain, route_gain, slack_w, proc_w, flex_w, scarce_w, scarcity_w, mpen_w) = match phase {
        0 => (
            1.00 + 0.18*js,
            0.32 + 0.28*js,
            0.17*(0.80 + 0.40*js),
            0.92 + 0.08*pre.high_flex,
            1.24,
            1.24,
            0.0,
            0.18,
            0.24*pre.flex_factor,
            0.10*pre.flex_factor,
            0.14,
            (0.08 + 0.20*pre.high_flex)*pre.flex_factor,
        ),
        1 => (
            1.15 + 0.30*js,
            0.50 + 0.40*js,
            0.24*(0.90 + 0.50*js),
            1.18 + 0.15*pre.high_flex,
            0.96,
            0.98,
            pre.slack_base*(0.40 + 0.30*js),
            0.20,
            0.30*pre.flex_factor,
            0.18*pre.flex_factor,
            0.18,
            (0.10 + 0.34*pre.high_flex)*pre.flex_factor,
        ),
        _ => (
            1.30 + 0.32*js,
            0.72 + 0.42*js,
            0.18*(0.70 + 0.45*js),
            1.58 + 0.22*pre.high_flex,
            0.72,
            0.74,
            pre.slack_base*(0.92 + 0.34*js),
            0.24,
            0.34*pre.flex_factor,
            0.22*pre.flex_factor,
            0.20,
            (0.12 + 0.46*pre.high_flex)*pre.flex_factor,
        ),
    };
    let flow_term = pre.flow_w * pre.job_flow_pref[job] * flow_gain;
    let slack_term = if slack_w > 0.0 {
        let slack_u = slack_urgency_js(pre, target_mk, time, product, op_idx);
        let slack_reg_boost = 1.0 + if phase == 2 { 0.50 } else { 0.30 } * reg_n;
        slack_w * slack_u * slack_reg_boost
    } else { 0.0 };
    let pop_pen = if pre.chaotic_like && op.flex >= 2 {
        let pop = js_cached_pop(pre, machine);
        let ppw = match phase { 0 => 0.20, 1 => 0.14, _ => 0.08 };
        ppw * pop * pre.flex_factor
    } else { 0.0 };
    let route_term = if route_w > 0.0 && op.flex >= 2 { route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine) } else { 0.0 };
    let next_term = next_w * next_term_raw;
    (cp_w*rem_min_n)+0.12*rem_avg_n+0.08*ops_n+scarcity_w*scarcity_urg+flex_w*flex_inv+scarce_w*scarce_match+(phase_reg_w*pre.flex_factor)*reg_n+next_term+slack_term-end_w*end_n-proc_w*proc_n-pop_pen-(mpen_gain*mpen_w)*mpen+0.55*job_bias+flow_term+route_term+jitter
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn score_candidate_bn_heavy(
    pre: &Pre, job: usize, product: usize, op_idx: usize,
    ops_rem: usize, op: &OpInfo, machine: usize, pt: u32, time: u32,
    target_mk: Option<u32>, best_end: u32, second_end: u32, best_cnt_total: usize,
    progress: f64, job_bias: f64, machine_penalty: f64, dynamic_load: f64,
    route_pref: Option<&RoutePrefLite>, route_w: f64, jitter: f64,
) -> f64 {
    let rem_min = pre.product_suf_min[product][op_idx] as f64;
    let rem_avg = pre.product_suf_avg[product][op_idx];
    let rem_bn = pre.product_suf_bn[product][op_idx];
    let flex_f = (op.flex as f64).max(1.0); let flex_inv = 1.0 / flex_f;
    let rem_min_n = rem_min / pre.horizon.max(1.0); let rem_avg_n = rem_avg / pre.max_job_avg_work.max(1e-9);
    let bn_n = rem_bn / pre.max_job_bn.max(1e-9); let ops_n = (ops_rem as f64) / (pre.max_ops as f64).max(1.0);
    let load_n = dynamic_load / pre.avg_machine_load.max(1e-9); let scar_n = js_cached_scar_n(pre, machine);
    let end_n = (best_end as f64) / pre.time_scale.max(1.0); let proc_n = (pt as f64) / pre.avg_op_min.max(1.0);
    let regret = if second_end >= INF { pre.avg_op_min * 2.6 } else { (second_end - best_end) as f64 };
    let reg_n = (regret / pre.avg_op_min.max(1.0)).clamp(0.0, 6.0);
    let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
    let density_n = ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0);
    let next_min = pre.product_next_min[product][op_idx] as f64; let next_min_n = next_min / pre.horizon.max(1.0);
    let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
    let next_term_raw = (0.55*next_min_n + 0.45*next_flex_inv) * (1.0 + 0.30*density_n*pre.high_flex);
    let js = pre.jobshopness;
    let avg_flex_inv = 1.0 / pre.flex_avg.max(1.0); let scarce_match = scar_n * (flex_inv - avg_flex_inv);
    let mpen = js_cached_penalty(machine, machine_penalty); let mpen_gain = 1.0 + 0.85*pre.high_flex;
    let phase = progress_mode_js(progress);
    let (bn_w, end_w, phase_reg_w, load_w, density_w, next_w, flow_gain, route_gain, slack_w, proc_w, mpen_w, scarcity_w) = match phase {
        0 => (
            (0.44 + 0.26*js)*pre.bn_focus,
            0.58,
            0.28 + 0.18*js,
            if pre.hi_flex { -0.45 } else { 0.65 + 0.18*js } * pre.flex_factor,
            0.12,
            0.16*(0.70 + 0.50*js),
            1.26,
            1.26,
            0.0,
            0.16,
            (0.08 + 0.16*js)*pre.flex_factor*(0.95 + 0.45*pre.high_flex),
            0.16,
        ),
        1 => (
            (0.88 + 0.55*js)*pre.bn_focus,
            0.82,
            0.60 + 0.25*js,
            if pre.hi_flex { -0.28 } else { 0.55 + 0.24*js } * pre.flex_factor,
            0.22,
            0.23*(0.70 + 0.65*js),
            0.98,
            0.98,
            0.0,
            0.18,
            (0.12 + 0.30*js)*pre.flex_factor*(0.95 + 0.65*pre.high_flex),
            0.18,
        ),
        _ => (
            (0.60 + 0.30*js)*pre.bn_focus,
            1.10,
            0.70 + 0.30*js,
            if pre.hi_flex { -0.12 } else { 0.18 + 0.08*js } * pre.flex_factor,
            0.16,
            0.18*(0.55 + 0.55*js),
            0.70,
            0.72,
            pre.slack_base*(0.55 + 0.50*js),
            0.20,
            (0.14 + 0.34*js)*pre.flex_factor*(1.00 + 0.75*pre.high_flex),
            0.20,
        ),
    };
    let flow_term = pre.flow_w * pre.job_flow_pref[job] * flow_gain;
    let slack_term = if slack_w > 0.0 {
        let slack_u = slack_urgency_js(pre, target_mk, time, product, op_idx);
        slack_w * slack_u * (1.0 + 0.45*reg_n)
    } else { 0.0 };
    let pop_pen = if pre.chaotic_like && op.flex >= 2 {
        let pop = js_cached_pop(pre, machine);
        let ppw = match phase { 0 => 0.20, 1 => 0.14, _ => 0.09 };
        ppw * pop * pre.flex_factor
    } else { 0.0 };
    let route_term = if route_w > 0.0 && op.flex >= 2 { route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine) } else { 0.0 };
    let next_term = next_w * next_term_raw;
    (0.94*rem_min_n)+(0.30*rem_avg_n)+(bn_w*bn_n)+(density_w*density_n)+(0.10*ops_n)+(0.64*pre.flex_factor)*flex_inv+(0.34*pre.flex_factor)*scarce_match+load_w*load_n+(phase_reg_w*pre.flex_factor)*reg_n+scarcity_w*scarcity_urg+next_term+slack_term-end_w*end_n-proc_w*proc_n-pop_pen-(mpen_gain*mpen_w)*mpen+0.60*job_bias+flow_term+route_term+jitter
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn score_candidate_adaptive(
    pre: &Pre, job: usize, product: usize, op_idx: usize,
    ops_rem: usize, op: &OpInfo, machine: usize, pt: u32, time: u32,
    target_mk: Option<u32>, best_end: u32, second_end: u32, best_cnt_total: usize,
    progress: f64, job_bias: f64, machine_penalty: f64, dynamic_load: f64,
    route_pref: Option<&RoutePrefLite>, route_w: f64, jitter: f64,
) -> f64 {
    let rem_min = pre.product_suf_min[product][op_idx] as f64;
    let rem_avg = pre.product_suf_avg[product][op_idx];
    let rem_bn = pre.product_suf_bn[product][op_idx];
    let flex_f = (op.flex as f64).max(1.0); let flex_inv = 1.0 / flex_f;
    let rem_min_n = rem_min / pre.horizon.max(1.0); let rem_avg_n = rem_avg / pre.max_job_avg_work.max(1e-9);
    let bn_n = rem_bn / pre.max_job_bn.max(1e-9); let ops_n = (ops_rem as f64) / (pre.max_ops as f64).max(1.0);
    let load_n = dynamic_load / pre.avg_machine_load.max(1e-9); let scar_n = js_cached_scar_n(pre, machine);
    let end_n = (best_end as f64) / pre.time_scale.max(1.0); let proc_n = (pt as f64) / pre.avg_op_min.max(1.0);
    let regret = if second_end >= INF { pre.avg_op_min * 2.6 } else { (second_end - best_end) as f64 };
    let reg_n = (regret / pre.avg_op_min.max(1.0)).clamp(0.0, 6.0);
    let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
    let density_n = ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0);
    let next_min = pre.product_next_min[product][op_idx] as f64; let next_min_n = next_min / pre.horizon.max(1.0);
    let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
    let next_term_raw = (0.55*next_min_n + 0.45*next_flex_inv) * (1.0 + 0.30*density_n*pre.high_flex);
    let js = pre.jobshopness; let fl = 1.0 - js;
    let avg_flex_inv = 1.0 / pre.flex_avg.max(1.0); let scarce_match = scar_n * (flex_inv - avg_flex_inv);
    let mpen = js_cached_penalty(machine, machine_penalty); let mpen_gain = 1.0 + 0.85*pre.high_flex;
    let phase = progress_mode_js(progress);
    let load_sign = if pre.hi_flex { -1.0 } else { 1.0 };
    let (bn_w, end_w, phase_reg_w, load_w, density_w, next_w, flow_gain, route_gain, slack_w, proc_w, mpen_w, scarcity_w) = match phase {
        0 => (
            (0.20 + 0.18*js)*pre.bn_focus,
            0.86*fl + 0.72*js + 0.12*pre.high_flex,
            0.24*fl + 0.46*js,
            load_sign*(0.60*fl + 0.92*js)*pre.flex_factor,
            0.05*fl + 0.12*js,
            0.16*(0.65*fl + 1.20*js),
            1.28,
            1.28,
            0.0,
            0.18*fl + 0.14*js,
            (0.05*fl + 0.16*js)*pre.flex_factor*(1.0 + 0.70*pre.high_flex),
            0.12*pre.flex_factor,
        ),
        1 => (
            (0.48 + 0.42*js)*pre.bn_focus,
            1.00*fl + 0.92*js + 0.14*pre.high_flex,
            0.48*fl + 0.78*js,
            load_sign*(0.42*fl + 0.72*js)*pre.flex_factor,
            0.08*fl + 0.20*js,
            0.23*(0.55*fl + 1.50*js),
            0.98,
            0.98,
            0.0,
            0.16*fl + 0.12*js,
            (0.08*fl + 0.28*js)*pre.flex_factor*(1.0 + 0.85*pre.high_flex),
            0.18*pre.flex_factor,
        ),
        _ => (
            (0.34 + 0.30*js)*pre.bn_focus,
            1.22 + 0.28*js + 0.18*pre.high_flex,
            0.72*fl + 0.96*js,
            load_sign*(0.18*fl + 0.32*js)*pre.flex_factor,
            0.04*fl + 0.16*js,
            0.18*(0.40*fl + 1.25*js),
            0.72,
            0.72,
            pre.slack_base*(0.95 + 0.30*js),
            0.12*fl + 0.10*js,
            (0.10*fl + 0.34*js)*pre.flex_factor*(1.0 + 0.90*pre.high_flex),
            0.24*pre.flex_factor,
        ),
    };
    let flow_term = pre.flow_w * pre.job_flow_pref[job] * flow_gain;
    let slack_term = if slack_w > 0.0 {
        let slack_u = slack_urgency_js(pre, target_mk, time, product, op_idx);
        slack_w * slack_u * (1.0 + 0.55*reg_n)
    } else { 0.0 };
    let pop_pen = if pre.chaotic_like && op.flex >= 2 {
        let pop = js_cached_pop(pre, machine);
        let ppw = match phase { 0 => 0.20, 1 => 0.13, _ => 0.08 };
        ppw * pop * pre.flex_factor
    } else { 0.0 };
    let route_term = if route_w > 0.0 && op.flex >= 2 { route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine) } else { 0.0 };
    let next_term = next_w * next_term_raw;
    (1.03*rem_min_n)+(0.48*rem_avg_n)+(bn_w*bn_n)+density_w*density_n+(0.08*ops_n)+(0.60*pre.flex_factor)*flex_inv+(0.50*pre.flex_factor)*scarce_match+load_w*load_n+(phase_reg_w*pre.flex_factor)*reg_n+scarcity_w*scarcity_urg+next_term+slack_term-end_w*end_n-proc_w*proc_n-pop_pen-(mpen_gain*mpen_w)*mpen+(0.60+0.06*js)*job_bias+flow_term+route_term+jitter
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn score_candidate_flex_balance(
    pre: &Pre, job: usize, product: usize, op_idx: usize,
    ops_rem: usize, op: &OpInfo, machine: usize, pt: u32, time: u32,
    target_mk: Option<u32>, best_end: u32, second_end: u32, best_cnt_total: usize,
    progress: f64, job_bias: f64, machine_penalty: f64, dynamic_load: f64,
    route_pref: Option<&RoutePrefLite>, route_w: f64, jitter: f64,
) -> f64 {
    let rem_min = pre.product_suf_min[product][op_idx] as f64;
    let rem_avg = pre.product_suf_avg[product][op_idx];
    let rem_min_n = rem_min / pre.horizon.max(1.0); let rem_avg_n = rem_avg / pre.max_job_avg_work.max(1e-9);
    let ops_n = (ops_rem as f64) / (pre.max_ops as f64).max(1.0);
    let load_n = dynamic_load / pre.avg_machine_load.max(1e-9);
    let end_n = (best_end as f64) / pre.time_scale.max(1.0); let proc_n = (pt as f64) / pre.avg_op_min.max(1.0);
    let regret = if second_end >= INF { pre.avg_op_min * 2.6 } else { (second_end - best_end) as f64 };
    let reg_n = (regret / pre.avg_op_min.max(1.0)).clamp(0.0, 6.0);
    let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
    let density_n = ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0);
    let next_min = pre.product_next_min[product][op_idx] as f64; let next_min_n = next_min / pre.horizon.max(1.0);
    let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
    let p2 = progress * progress; let next_w_base = 0.12 + p2 * 0.28;
    let next_term_raw = (0.55*next_min_n + 0.45*next_flex_inv) * (1.0 + 0.30*density_n*pre.high_flex);
    let js = pre.jobshopness;
    let mpen = machine_penalty.clamp(0.0, 1.0);
    let flow_term = pre.flow_w * pre.job_flow_pref[job] * (0.65 + 0.70*(1.0-progress));
    let slack_u = slack_urgency_js(pre, target_mk, time, product, op_idx);
    let slack_w = pre.slack_base * (0.25 + 0.75*progress);
    let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop = js_cached_pop(pre, machine); (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
    let route_gain = (0.70 + 0.80*(1.0-progress)).clamp(0.70, 1.40);
    let route_term = if route_w > 0.0 && op.flex >= 2 { route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine) } else { 0.0 };
    let end_w=(0.85+0.70*progress+0.15*js).clamp(0.85,1.75); let cp_w=(1.00+0.30*js+0.15*(1.0-progress)).clamp(0.95,1.45); let load_w=(0.55+0.35*pre.high_flex).clamp(0.55,0.95)*pre.flex_factor; let mpen_w=(0.55+0.65*pre.high_flex).clamp(0.55,1.15); let reg_w=(0.35+0.25*(1.0-progress)).clamp(0.35,0.70); let next_term=next_w_base*0.40*next_term_raw;
    (cp_w*rem_min_n)+0.55*rem_avg_n+0.08*ops_n+0.06*density_n+0.08*scarcity_urg+next_term+(0.70*slack_w)*slack_u-end_w*end_n-0.16*proc_n-pop_pen-load_w*load_n-(mpen_w*(1.0+0.85*pre.high_flex))*mpen+(reg_w*pre.flex_factor)*reg_n+(0.58+0.10*pre.high_flex)*job_bias+flow_term+route_term+jitter
}

#[inline]
fn rule_idx(r: Rule) -> usize {
    match r { Rule::Adaptive=>0, Rule::BnHeavy=>1, Rule::EndTight=>2, Rule::CriticalPath=>3, Rule::MostWork=>4, Rule::LeastFlex=>5, Rule::Regret=>6, Rule::ShortestProc=>7, Rule::FlexBalance=>8 }
}

fn choose_rule_bandit(rng: &mut SmallRng, rules: &[Rule], rule_best: &[u32], rule_tries: &[u32], global_best: u32, margin: u32, stuck: usize, chaos_like: bool, late_phase: bool) -> Rule {
    if rules.is_empty() { return Rule::Adaptive; }
    let mut best_seen = global_best; for &mk in rule_best { if mk < best_seen { best_seen = mk; } }
    let scale = (margin as f64).max(1.0); let s = ((stuck as f64)/140.0).clamp(0.0,1.0); let explore_mix = (0.10+0.55*s).clamp(0.10,0.65);
    let mut w = [0.0f64; 9];
    for (i, &r) in rules.iter().enumerate() {
        let idx = rule_idx(r);
        let mk = rule_best[idx]; let t = rule_tries[idx].max(1) as f64;
        let delta = mk.saturating_sub(best_seen) as f64; let exploit = (-delta/scale).exp(); let explore = (1.0/t).sqrt();
        let mut ww = (1.0-explore_mix)*exploit+explore_mix*explore; ww = ww.max(1e-6);
        if chaos_like { ww = ww.powf(0.70); } else if late_phase { ww = ww.powf(1.18); }
        w[i] = ww;
    }
    let mut sum = 0.0; for i in 0..rules.len() { sum += w[i].max(0.0); }
    if !(sum > 0.0) { return rules[rng.gen_range(0..rules.len())]; }
    let mut r = rng.gen::<f64>() * sum;
    for i in 0..rules.len() { r -= w[i].max(0.0); if r <= 0.0 { return rules[i]; } }
    rules[rules.len()-1]
}

#[inline]
fn choose_from_top_weighted_temp(rng: &mut SmallRng, top: &[Cand], temperature: f64) -> Cand {
    if top.len() <= 1 { return top[0]; }
    let flat = temperature.clamp(0.0, 1.0);
    let sharp = 1.0 - flat;
    let n = top.len();
    let mut sum = 0.0;
    for i in 0..n {
        sum += flat + sharp * ((n - i) as f64);
    }
    let mut r = rng.gen::<f64>() * sum;
    for i in 0..n {
        r -= flat + sharp * ((n - i) as f64);
        if r <= 0.0 { return top[i]; }
    }
    top[n - 1]
}

#[inline]
fn score_candidate_specialized<const RULE: u8>(
    pre: &Pre, job: usize, product: usize, op_idx: usize,
    ops_rem: usize, op: &OpInfo, machine: usize, pt: u32, time: u32,
    target_mk: Option<u32>, best_end: u32, second_end: u32, best_cnt_total: usize,
    progress: f64, job_bias: f64, machine_penalty: f64, dynamic_load: f64,
    route_pref: Option<&RoutePrefLite>, route_w: f64, jitter: f64,
) -> f64 {
    if RULE == 0 {
        score_candidate_adaptive(pre, job, product, op_idx, ops_rem, op, machine, pt, time, target_mk, best_end, second_end, best_cnt_total, progress, job_bias, machine_penalty, dynamic_load, route_pref, route_w, jitter)
    } else if RULE == 1 {
        score_candidate_bn_heavy(pre, job, product, op_idx, ops_rem, op, machine, pt, time, target_mk, best_end, second_end, best_cnt_total, progress, job_bias, machine_penalty, dynamic_load, route_pref, route_w, jitter)
    } else if RULE == 2 {
        score_candidate_end_tight(pre, job, product, op_idx, ops_rem, op, machine, pt, time, target_mk, best_end, second_end, best_cnt_total, progress, job_bias, machine_penalty, dynamic_load, route_pref, route_w, jitter)
    } else if RULE == 3 {
        score_candidate_critical_path(pre, job, product, op_idx, ops_rem, op, machine, pt, time, target_mk, best_end, second_end, best_cnt_total, progress, job_bias, machine_penalty, dynamic_load, route_pref, route_w, jitter)
    } else if RULE == 4 {
        score_candidate_most_work(pre, job, product, op_idx, ops_rem, op, machine, pt, time, target_mk, best_end, second_end, best_cnt_total, progress, job_bias, machine_penalty, dynamic_load, route_pref, route_w, jitter)
    } else if RULE == 5 {
        score_candidate_least_flex(pre, job, product, op_idx, ops_rem, op, machine, pt, time, target_mk, best_end, second_end, best_cnt_total, progress, job_bias, machine_penalty, dynamic_load, route_pref, route_w, jitter)
    } else if RULE == 6 {
        score_candidate_regret(pre, job, product, op_idx, ops_rem, op, machine, pt, time, target_mk, best_end, second_end, best_cnt_total, progress, job_bias, machine_penalty, dynamic_load, route_pref, route_w, jitter)
    } else if RULE == 7 {
        score_candidate_shortest_proc(pre, job, product, op_idx, ops_rem, op, machine, pt, time, target_mk, best_end, second_end, best_cnt_total, progress, job_bias, machine_penalty, dynamic_load, route_pref, route_w, jitter)
    } else {
        score_candidate_flex_balance(pre, job, product, op_idx, ops_rem, op, machine, pt, time, target_mk, best_end, second_end, best_cnt_total, progress, job_bias, machine_penalty, dynamic_load, route_pref, route_w, jitter)
    }
}

#[inline]
fn construct_solution_conflict_dispatch<const USE_JB: bool, const USE_MP: bool, const USE_ROUTE: bool, const CHAOTIC: bool>(
    challenge: &Challenge, pre: &Pre, rule: Rule, k: usize, target_mk: Option<u32>,
    rng: &mut SmallRng, job_bias: Option<&[f64]>, machine_penalty: Option<&[f64]>,
    route_pref: Option<&RoutePrefLite>, route_w: f64,
) -> Result<(Solution, u32)> {
    match rule {
        Rule::Adaptive => construct_solution_conflict_impl::<0, USE_JB, USE_MP, USE_ROUTE, CHAOTIC>(challenge, pre, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        Rule::BnHeavy => construct_solution_conflict_impl::<1, USE_JB, USE_MP, USE_ROUTE, CHAOTIC>(challenge, pre, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        Rule::EndTight => construct_solution_conflict_impl::<2, USE_JB, USE_MP, USE_ROUTE, CHAOTIC>(challenge, pre, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        Rule::CriticalPath => construct_solution_conflict_impl::<3, USE_JB, USE_MP, USE_ROUTE, CHAOTIC>(challenge, pre, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        Rule::MostWork => construct_solution_conflict_impl::<4, USE_JB, USE_MP, USE_ROUTE, CHAOTIC>(challenge, pre, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        Rule::LeastFlex => construct_solution_conflict_impl::<5, USE_JB, USE_MP, USE_ROUTE, CHAOTIC>(challenge, pre, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        Rule::Regret => construct_solution_conflict_impl::<6, USE_JB, USE_MP, USE_ROUTE, CHAOTIC>(challenge, pre, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        Rule::ShortestProc => construct_solution_conflict_impl::<7, USE_JB, USE_MP, USE_ROUTE, CHAOTIC>(challenge, pre, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        Rule::FlexBalance => construct_solution_conflict_impl::<8, USE_JB, USE_MP, USE_ROUTE, CHAOTIC>(challenge, pre, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
    }
}

fn construct_solution_conflict(
    challenge: &Challenge, pre: &Pre, rule: Rule, k: usize, target_mk: Option<u32>,
    rng: &mut SmallRng, job_bias: Option<&[f64]>, machine_penalty: Option<&[f64]>,
    route_pref: Option<&RoutePrefLite>, route_w: f64,
) -> Result<(Solution, u32)> {
    let use_jb = job_bias.is_some();
    let use_mp = machine_penalty.is_some();
    let use_route = route_pref.is_some() && route_w > 0.0;
    match (pre.chaotic_like, use_jb, use_mp, use_route) {
        (false, false, false, false) => construct_solution_conflict_dispatch::<false, false, false, false>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (false, false, false, true) => construct_solution_conflict_dispatch::<false, false, true, false>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (false, false, true, false) => construct_solution_conflict_dispatch::<false, true, false, false>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (false, false, true, true) => construct_solution_conflict_dispatch::<false, true, true, false>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (false, true, false, false) => construct_solution_conflict_dispatch::<true, false, false, false>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (false, true, false, true) => construct_solution_conflict_dispatch::<true, false, true, false>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (false, true, true, false) => construct_solution_conflict_dispatch::<true, true, false, false>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (false, true, true, true) => construct_solution_conflict_dispatch::<true, true, true, false>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (true, false, false, false) => construct_solution_conflict_dispatch::<false, false, false, true>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (true, false, false, true) => construct_solution_conflict_dispatch::<false, false, true, true>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (true, false, true, false) => construct_solution_conflict_dispatch::<false, true, false, true>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (true, false, true, true) => construct_solution_conflict_dispatch::<false, true, true, true>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (true, true, false, false) => construct_solution_conflict_dispatch::<true, false, false, true>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (true, true, false, true) => construct_solution_conflict_dispatch::<true, false, true, true>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (true, true, true, false) => construct_solution_conflict_dispatch::<true, true, false, true>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (true, true, true, true) => construct_solution_conflict_dispatch::<true, true, true, true>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
    }
}

#[inline]
fn construct_solution_conflict_impl<const RULE: u8, const USE_JB: bool, const USE_MP: bool, const USE_ROUTE: bool, const CHAOTIC: bool>(
    challenge: &Challenge, pre: &Pre, k: usize, target_mk: Option<u32>,
    rng: &mut SmallRng, job_bias: Option<&[f64]>, machine_penalty: Option<&[f64]>,
    route_pref: Option<&RoutePrefLite>, route_w: f64,
) -> Result<(Solution, u32)> {
    let num_jobs = challenge.num_jobs; let num_machines = challenge.num_machines;
    let mut job_next_op = vec![0usize; num_jobs]; let mut job_ready_time = vec![0u32; num_jobs];
    let mut machine_avail = vec![0u32; num_machines]; let mut machine_load = pre.machine_load0.clone();
    let mut job_schedule: Vec<Vec<(usize, u32)>> = pre.job_ops_len.iter().map(|&len| Vec::with_capacity(len)).collect();
    let mut remaining_ops = pre.total_ops; let mut time = 0u32;
    let mut demand: Vec<u16> = vec![0u16; num_machines];
    let mut raw_by_machine: Vec<Vec<RawCand>> = (0..num_machines).map(|_| Vec::with_capacity(12)).collect();
    let mut idle_machines: Vec<usize> = (0..num_machines).collect();
    let mut idle_pos: Vec<usize> = (0..num_machines).collect();
    let mut busy_machine_heap: std::collections::BinaryHeap<std::cmp::Reverse<(u32, usize)>> = std::collections::BinaryHeap::with_capacity(num_machines);
    let mut blocked_job_heap: std::collections::BinaryHeap<std::cmp::Reverse<(u32, usize)>> = std::collections::BinaryHeap::with_capacity(num_jobs);
    let chaotic_like = CHAOTIC;
    let mut machine_work: Vec<u64> = if chaotic_like { vec![0u64; num_machines] } else { vec![] };
    let mut sum_work: u64 = 0;
    let mut touched_machines: Vec<usize> = Vec::with_capacity(num_machines);
    let mut round_stamp: u32 = 1;
    let mut top: Vec<Cand> = if k > 0 { Vec::with_capacity(k) } else { Vec::new() };
    let mut ready_by_machine: Vec<Vec<(usize, u32, u32)>> = (0..num_machines).map(|_| Vec::with_capacity(32)).collect();
    let mut job_gen: Vec<u32> = vec![0u32; num_jobs];
    let mut job_eval_stamp: Vec<u32> = vec![0u32; num_jobs];
    let mut job_best_end: Vec<u32> = vec![INF; num_jobs];
    let mut job_second_end: Vec<u32> = vec![INF; num_jobs];
    let mut job_best_cnt_total: Vec<usize> = vec![0usize; num_jobs];
    let mut job_best_cnt_idle: Vec<usize> = vec![0usize; num_jobs];
    let mut job_rigidity: Vec<f64> = vec![0.0; num_jobs];
    let mut job_regn: Vec<f64> = vec![0.0; num_jobs];
    let mut job_product_cur: Vec<usize> = vec![0usize; num_jobs];
    let mut job_ops_rem: Vec<usize> = vec![0usize; num_jobs];
    let mut job_op_ptr: Vec<*const OpInfo> = vec![std::ptr::null(); num_jobs];
    let mut job_bias_cur: Vec<f64> = if USE_JB { vec![0.0; num_jobs] } else { Vec::new() };
    let job_bias = if USE_JB { unsafe { job_bias.unwrap_unchecked() } } else { &[][..] };
    let machine_penalty = if USE_MP { unsafe { machine_penalty.unwrap_unchecked() } } else { &[][..] };
    let route_pref = if USE_ROUTE { Some(unsafe { route_pref.unwrap_unchecked() }) } else { None };
    let route_w = if USE_ROUTE { route_w } else { 0.0 };
    let machine_scar_n: Vec<f64> = if RULE == 0 || RULE == 1 || RULE == 2 {
        let denom = pre.avg_machine_scarcity.max(1e-9);
        pre.machine_scarcity.iter().map(|&v| v / denom).collect()
    } else {
        Vec::new()
    };
    let machine_pop: &[f64] = if CHAOTIC { &pre.machine_best_pop[..] } else { &[][..] };
    let machine_penalty_clamped: Vec<f64> = if USE_MP {
        machine_penalty.iter().map(|&v| v.clamp(0.0, 1.0)).collect()
    } else {
        Vec::new()
    };
    let machine_u8: Vec<u8> = if USE_ROUTE {
        (0..num_machines).map(|m| m.min(255) as u8).collect()
    } else {
        Vec::new()
    };
    unsafe { set_js_machine_static_cache(&machine_scar_n, machine_pop, &machine_penalty_clamped, &machine_u8); }
    let _machine_static_cache_guard = JsMachineStaticCacheGuard;
    for job in 0..num_jobs {
        if pre.job_ops_len[job] == 0 { continue; }
        let product = pre.job_products[job];
        let op = &pre.product_ops[product][0];
        if op.flex == 0 || op.machines.is_empty() || op.min_pt >= INF { continue; }
        let gen = job_gen[job];
        for &(m, pt) in &op.machines {
            ready_by_machine[m].push((job, gen, pt));
        }
    }
    while remaining_ops > 0 {
        loop {
            while let Some(entry) = busy_machine_heap.peek() {
                let std::cmp::Reverse((t, m)) = *entry;
                if t > time { break; }
                busy_machine_heap.pop();
                if machine_avail[m] == t && idle_pos[m] == NONE_USIZE {
                    idle_pos[m] = idle_machines.len();
                    idle_machines.push(m);
                }
            }
            while let Some(entry) = blocked_job_heap.peek() {
                let std::cmp::Reverse((t, j)) = *entry;
                if t > time { break; }
                blocked_job_heap.pop();
                if job_next_op[j] >= pre.job_ops_len[j] || job_ready_time[j] != t { continue; }
                let product = pre.job_products[j];
                let op_idx = job_next_op[j];
                let op = &pre.product_ops[product][op_idx];
                if op.flex == 0 || op.machines.is_empty() || op.min_pt >= INF { continue; }
                let gen = job_gen[j];
                for &(m, pt) in &op.machines {
                    ready_by_machine[m].push((j, gen, pt));
                }
            }
            if idle_machines.is_empty() { break; }
            let cur_stamp = round_stamp;
            round_stamp = round_stamp.wrapping_add(1);
            if round_stamp == 0 { job_eval_stamp.fill(0); round_stamp = 1; }
            touched_machines.clear();
            let progress = 1.0 - (remaining_ops as f64) / (pre.total_ops as f64).max(1.0);
            let cap_per_machine = if k == 0 { 12usize } else { (k+6).min(12) };
            for &m in &idle_machines {
                demand[m] = 0;
                raw_by_machine[m].clear();
                let list = &mut ready_by_machine[m];
                let mut write = 0usize;
                for read in 0..list.len() {
                    let (job, gen, pt) = list[read];
                    if job_gen[job] != gen { continue; }
                    let op_idx = job_next_op[job];
                    if op_idx >= pre.job_ops_len[job] || job_ready_time[job] > time { continue; }
                    list[write] = (job, gen, pt);
                    write += 1;
                    if job_eval_stamp[job] != cur_stamp {
                        job_eval_stamp[job] = cur_stamp;
                        let product = pre.job_products[job];
                        job_product_cur[job] = product;
                        job_ops_rem[job] = pre.job_ops_len[job] - op_idx;
                        if USE_JB { job_bias_cur[job] = job_bias[job]; }
                        let op = &pre.product_ops[product][op_idx];
                        if op.flex == 0 || op.machines.is_empty() || op.min_pt >= INF {
                            job_op_ptr[job] = std::ptr::null();
                            job_best_end[job] = INF;
                            job_second_end[job] = INF;
                            job_best_cnt_total[job] = 0;
                            job_best_cnt_idle[job] = 0;
                        } else {
                            job_op_ptr[job] = op as *const OpInfo;
                            let (best_end, second_end, best_cnt_total, best_cnt_idle) = best_second_and_counts(time, &machine_avail, op);
                            job_best_end[job] = best_end;
                            job_second_end[job] = second_end;
                            job_best_cnt_total[job] = best_cnt_total;
                            job_best_cnt_idle[job] = best_cnt_idle;
                            if best_end < INF && best_cnt_idle > 0 {
                                let flex_inv = 1.0/(op.flex as f64).max(1.0);
                                let scarcity_urg = 1.0/(best_cnt_total as f64).max(1.0);
                                let regret = if second_end >= INF { pre.avg_op_min*2.6 } else { (second_end-best_end) as f64 };
                                job_regn[job] = (regret/pre.avg_op_min.max(1.0)).clamp(0.0,6.0);
                                job_rigidity[job] = (0.60*flex_inv+0.40*scarcity_urg).clamp(0.0,2.5);
                            }
                        }
                    }
                    if job_best_end[job] >= INF || job_best_cnt_idle[job] == 0 { continue; }
                    if time.saturating_add(pt) != job_best_end[job] { continue; }
                    if demand[m] == 0 { touched_machines.push(m); }
                    demand[m] = demand[m].saturating_add(1);
                    let product = job_product_cur[job];
                    let op = unsafe { &*job_op_ptr[job] };
                    let ops_rem = job_ops_rem[job];
                    let jb = if USE_JB { job_bias_cur[job] } else { 0.0 };
                    let tie_idle = op.flex >= 2 && job_best_cnt_idle[job] > 1;
                    let tie_frac = if tie_idle {
                        ((job_best_cnt_idle[job].saturating_sub(1) as f64) / (op.flex as f64).max(1.0)).clamp(0.0, 1.0)
                    } else { 0.0 };
                    let mp_eff = if USE_MP {
                        if tie_frac > 0.0 {
                            (machine_penalty[m] * (0.72 + 0.20*progress) + (0.10 + 0.16*(1.0-progress))*tie_frac).clamp(0.0, 1.0)
                        } else {
                            machine_penalty[m]
                        }
                    } else { 0.0 };
                    let route_w_eff = if USE_ROUTE {
                        if tie_frac > 0.0 {
                            route_w * (0.60 + 0.22*progress - 0.16*tie_frac).clamp(0.36, 0.82)
                        } else {
                            route_w
                        }
                    } else { 0.0 };
                    let jitter = if k > 0 { rng.gen::<f64>()*1e-9 } else { 0.0 };
                    let dynamic_load_m = machine_load[m];
                    let base = score_candidate_specialized::<RULE>(pre, job, product, op_idx, ops_rem, op, m, pt, time, target_mk, job_best_end[job], job_second_end[job], job_best_cnt_total[job], progress, jb, mp_eff, dynamic_load_m, route_pref, route_w_eff, jitter);
                    if raw_by_machine[m].len() < cap_per_machine || base >= raw_by_machine[m][cap_per_machine - 1].base_score {
                        push_top_k_raw(&mut raw_by_machine[m], RawCand { job, machine: m, pt, base_score: base, rigidity: job_rigidity[job], reg_n: job_regn[job] }, cap_per_machine);
                    }
                }
                list.truncate(write);
            }
            touched_machines.sort_unstable();
            let denom = (idle_machines.len() as f64).max(1.0);
            let (conflict_w, conflict_scale) = if chaotic_like { (-(0.05+0.08*(1.0-progress)).clamp(0.04,0.14), (0.95+0.20*pre.flex_factor).clamp(0.90,1.20)) } else { ((0.09+0.26*pre.jobshopness+0.11*pre.high_flex+0.16*(1.0-progress)).clamp(0.05,0.45), (0.90+0.40*pre.flex_factor).clamp(0.85,1.75)) };
            let (bal_w, avg_work) = if chaotic_like { ((0.030+0.070*(1.0-progress)).clamp(0.025,0.11), (sum_work as f64)/(num_machines as f64).max(1.0)) } else { (0.0, 0.0) };
            let mut best: Option<Cand> = None; top.clear();
            let mut demand_excess_sum = 0u32;
            let mut max_demand = 1u16;
            for &m in &touched_machines {
                let dem_u = demand[m];
                if dem_u > max_demand { max_demand = dem_u; }
                demand_excess_sum = demand_excess_sum.saturating_add(dem_u.saturating_sub(1) as u32);
                let dem = dem_u as f64; if dem <= 0.0 || raw_by_machine[m].is_empty() { continue; }
                let dem_n = ((dem-1.0)/denom).clamp(0.0,2.5);
                let bal_pen = if chaotic_like && bal_w > 0.0 { let denomw=(avg_work+(pre.avg_op_min*3.0).max(1.0)).max(1.0); let r=(machine_work[m] as f64)/denomw; let done_n=(r/(r+1.0)).clamp(0.0,1.0); -bal_w*done_n } else { 0.0 };
                for rc in &raw_by_machine[m] {
                    let rig=rc.rigidity.clamp(0.0,2.5); let regc=rc.reg_n.clamp(0.0,4.5);
                    let mut boost=conflict_w*conflict_scale*dem_n*(1.15*rig+0.85*regc);
                    if chaotic_like { boost=boost.max(-0.26); }
                    let c = Cand { job: rc.job, machine: rc.machine, pt: rc.pt, score: rc.base_score+boost+bal_pen };
                    if k == 0 {
                        if best.map_or(true, |bb| c.score > bb.score) { best = Some(c); }
                    } else if top.len() < k || c.score >= top[k - 1].score {
                        push_top_k(&mut top, c, k);
                    }
                }
            }
            let select_temp = if k == 0 || top.len() <= 1 { 0.0 } else {
                let conflict_avg = ((demand_excess_sum as f64)/denom).clamp(0.0,3.0) / 3.0;
                let conflict_peak = ((max_demand.saturating_sub(1)) as f64).clamp(0.0,4.0) / 4.0;
                ((1.0-progress) * (0.22 + 0.58*conflict_avg + 0.20*conflict_peak)).clamp(0.0,0.92)
            };
            let chosen = if k == 0 { match best { Some(c) => c, None => break } } else { if top.is_empty() { break; } choose_from_top_weighted_temp(rng, &top, select_temp) };
            let job = chosen.job; let machine = chosen.machine; let pt = chosen.pt;
            let _product = job_product_cur[job]; let _op_idx = job_next_op[job]; let op = unsafe { &*job_op_ptr[job] };
            let best_end_now = job_best_end[job];
            let end_check = time.max(machine_avail[machine]).saturating_add(pt);
            if machine_avail[machine] > time || end_check != best_end_now { break; }
            let end_time = time.saturating_add(pt);
            job_schedule[job].push((machine, time)); job_next_op[job]+=1; job_ready_time[job]=end_time; machine_avail[machine]=end_time; remaining_ops-=1;
            job_gen[job] = job_gen[job].wrapping_add(1);
            let pos = idle_pos[machine];
            if pos != NONE_USIZE {
                let last = idle_machines.pop().unwrap();
                if pos < idle_machines.len() {
                    idle_machines[pos] = last;
                    idle_pos[last] = pos;
                }
                idle_pos[machine] = NONE_USIZE;
            }
            busy_machine_heap.push(std::cmp::Reverse((end_time, machine)));
            if job_next_op[job] < pre.job_ops_len[job] {
                if end_time <= time {
                    let next_product = pre.job_products[job];
                    let next_op = &pre.product_ops[next_product][job_next_op[job]];
                    if next_op.flex > 0 && !next_op.machines.is_empty() && next_op.min_pt < INF {
                        let gen = job_gen[job];
                        for &(m, pt2) in &next_op.machines {
                            ready_by_machine[m].push((job, gen, pt2));
                        }
                    }
                } else {
                    blocked_job_heap.push(std::cmp::Reverse((end_time, job)));
                }
            }
            if chaotic_like { machine_work[machine]=machine_work[machine].saturating_add(pt as u64); sum_work=sum_work.saturating_add(pt as u64); }

            if op.min_pt < INF && op.flex > 0 && !op.machines.is_empty() { let delta=(op.min_pt as f64)/(op.flex as f64).max(1.0); if delta>0.0 { for &(mm,_) in &op.machines { let v=machine_load[mm]-delta; machine_load[mm]=if v>0.0{v}else{0.0}; } } }
            if remaining_ops == 0 { break; }
        }
        if remaining_ops == 0 { break; }
        let next_machine_time = loop {
            match busy_machine_heap.peek() {
                Some(entry) => {
                    let std::cmp::Reverse((t, m)) = *entry;
                    if machine_avail[m] != t || t <= time {
                        busy_machine_heap.pop();
                        continue;
                    }
                    break Some(t);
                }
                None => break None,
            }
        };
        let next_job_time = loop {
            match blocked_job_heap.peek() {
                Some(entry) => {
                    let std::cmp::Reverse((t, j)) = *entry;
                    if job_next_op[j] >= pre.job_ops_len[j] || job_ready_time[j] != t || t <= time {
                        blocked_job_heap.pop();
                        continue;
                    }
                    break Some(t);
                }
                None => break None,
            }
        };
        time = match (next_machine_time, next_job_time) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => return Err(anyhow!("Stalled")),
        };
    }
    let mk = machine_avail.into_iter().max().unwrap_or(0);
    Ok((Solution { job_schedule }, mk))
}

#[inline]
fn rebuild_machine_pred_nodes(ds: &DisjSchedule, machine_pred_node: &mut [usize]) {
    machine_pred_node.fill(NONE_USIZE);
    for seq in &ds.machine_seq {
        for i in 1..seq.len() {
            machine_pred_node[seq[i]] = seq[i - 1];
        }
    }
}

fn rebuild_machine_succ_indeg(ds: &DisjSchedule, machine_succ: &mut [usize], indeg_full: &mut [u16]) {
    // Same gesture as in `eval_disj`, safety net included.
    indeg_full.copy_from_slice(&ds.indeg_job_plus1);
    let mut covered = 0usize;
    for seq in &ds.machine_seq {
        if seq.is_empty() { continue; }
        indeg_full[seq[0]] -= 1;
        covered += seq.len();
        let mut prev = seq[0];
        for &v in &seq[1..] {
            machine_succ[prev] = v;
            prev = v;
        }
        machine_succ[prev] = NONE_USIZE;
    }
    if covered != ds.n {
        indeg_full.copy_from_slice(&ds.indeg_job);
        for seq in &ds.machine_seq {
            for &v in seq.iter().skip(1) { indeg_full[v] = indeg_full[v].saturating_add(1); }
        }
    }
}

/// `topo` is a fixed-size buffer of `n` and `topo_len` its length, where it used to be a `Vec`.
/// `topo_len` is written even when the evaluation fails, exactly as the length of a partially
/// filled `Vec` would have been: the paths that re-read `topo` after a failure (the `exact_topk`
/// probe) therefore see the same prefix as before.
fn eval_disj_topo(ds: &DisjSchedule, buf: &mut EvalBuf, indeg_full: &[u16], topo: &mut [usize], topo_len: &mut usize) -> Option<(u32, usize)> {
    let n = ds.n;
    buf.indeg.copy_from_slice(indeg_full);
    buf.start.fill(0);
    // Same argument as in `eval_disj`: `indeg_full` is maintained as `indeg_job` plus the
    // machine predecessor, so a root is necessarily a job head. We sweep 50 candidates instead
    // of 1500, in the same increasing order.
    let mut sl = 0usize;
    for j in 0..ds.num_jobs {
        let h = ds.job_offsets[j];
        if h < ds.job_offsets[j + 1] && buf.indeg[h] == 0 {
            buf.best_pred[h] = NONE_USIZE;
            buf.stack[sl] = h;
            sl += 1;
        }
    }
    let mut tl = 0usize;
    let mut mk = 0u32;
    let mut mk_node = 0usize;
    while sl > 0 {
        sl -= 1;
        let u = buf.stack[sl];
        topo[tl] = u;
        tl += 1;
        let end_u = buf.start[u].saturating_add(ds.node_pt[u]);
        if end_u > mk { mk = end_u; mk_node = u; }
        let js = ds.job_succ[u];
        if js != NONE_USIZE {
            if buf.start[js] < end_u {
                buf.start[js] = end_u;
                buf.best_pred[js] = u;
            }
            buf.indeg[js] -= 1;
            if buf.indeg[js] == 0 { buf.stack[sl] = js; sl += 1; }
        }
        let ms = buf.machine_succ[u];
        if ms != NONE_USIZE {
            if buf.start[ms] < end_u {
                buf.start[ms] = end_u;
                buf.best_pred[ms] = u;
            }
            buf.indeg[ms] -= 1;
            if buf.indeg[ms] == 0 { buf.stack[sl] = ms; sl += 1; }
        }
    }
    *topo_len = tl;
    if tl != n { return None; }
    Some((mk, mk_node))
}

/// Writes `qend[o] = tail[o] + dur[o]` and not `tail[o]`.
///
/// No consumer wanted `tail` on its own: `estimate_swap_mk`, `estimate_swap_mk2` and this pass
/// itself all rebuilt the sum `pt[x] + tail[x]` on every read. Storing it pre-added removes two
/// loads and two saturating additions per operation in the backward pass, and three more per
/// swap estimate.
#[inline]
fn qend_from_topo(ds: &DisjSchedule, machine_succ: &[usize], topo: &[usize], qend: &mut [u32]) {
    for &nd in topo.iter().rev() {
        let js = ds.job_succ[nd];
        let mut t = if js != NONE_USIZE { qend[js] } else { 0 };
        let ms = machine_succ[nd];
        if ms != NONE_USIZE {
            let c = qend[ms];
            if c > t { t = c; }
        }
        qend[nd] = t.saturating_add(ds.node_pt[nd]);
    }
}

#[inline]
fn rebuild_machine_pos_map(machine_seq: &[Vec<usize>], node_pos: &mut [usize]) {
    for seq in machine_seq {
        for (pos, &node) in seq.iter().enumerate() {
            node_pos[node] = pos;
        }
    }
}

#[inline]
fn collect_guided_kick_moves(
    ds: &DisjSchedule,
    best_pred: &[usize],
    mk_node: usize,
    current_pos: &[usize],
    node_machine: &[usize],
    ref_pos: &[usize],
    used_machine: &mut [u8],
    out: &mut Vec<(u32, usize, usize)>,
) {
    out.clear();
    used_machine.fill(0);
    let mut u = mk_node;
    while u != NONE_USIZE {
        let m = node_machine[u];
        if used_machine[m] == 0 {
            let pos = current_pos[u];
            let target = ref_pos[u];
            let seq = &ds.machine_seq[m];
            if pos > target && pos > 0 {
                let left = seq[pos - 1];
                if ref_pos[u] < ref_pos[left] {
                    out.push(((pos - target) as u32, m, pos - 1));
                    used_machine[m] = 1;
                }
            } else if pos < target && pos + 1 < seq.len() {
                let right = seq[pos + 1];
                if ref_pos[u] > ref_pos[right] {
                    out.push(((target - pos) as u32, m, pos));
                    used_machine[m] = 1;
                }
            }
        }
        u = best_pred[u];
    }
}

fn ts_seed_key(pre: &Pre, num_machines: usize, sol: &Solution) -> Vec<u32> {
    let num_jobs = sol.job_schedule.len();
    if num_jobs != pre.job_ops_len.len() { return Vec::new(); }
    let mut job_offsets = vec![0usize; num_jobs + 1];
    for j in 0..num_jobs { job_offsets[j + 1] = job_offsets[j] + pre.job_ops_len[j]; }
    let mut per_machine: Vec<Vec<(u32, usize)>> = vec![Vec::new(); num_machines];
    for j in 0..num_jobs {
        if sol.job_schedule[j].len() != pre.job_ops_len[j] { return Vec::new(); }
        for (k, &(m, st)) in sol.job_schedule[j].iter().enumerate() {
            if m >= num_machines { return Vec::new(); }
            per_machine[m].push((st, job_offsets[j] + k));
        }
    }
    let mut key: Vec<u32> = Vec::with_capacity(job_offsets[num_jobs] + num_machines);
    for m in 0..num_machines {
        per_machine[m].sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        for &(_, id) in &per_machine[m] { key.push(id as u32); }
        key.push(u32::MAX);
    }
    key
}

#[inline]
fn ts_seed_key_distance(a: &[u32], b: &[u32]) -> u32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() { return 0; }
    let mut d = 0u32;
    for i in 0..a.len() {
        if a[i] != b[i] { d += 1; }
    }
    d
}

fn tabu_search_phase(pre: &Pre, challenge: &Challenge, base_sol: &Solution, max_iterations: usize, tenure_base: usize, fast_eval: bool, div_period: usize, tie_mode: usize, sel_key: usize, exact_topk: usize, traj_salt: u64, kick_spread: usize, sa_accept: usize, mid_abort_rescue: usize) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n); let n = ds.n;
    let Some((initial_mk, mut mk_node)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let mut cur_mk = initial_mk; let mut best_global_mk = initial_mk; let mut best_global_machine_seq = ds.machine_seq.clone();
    let tenure = tenure_base.max(5); let tenure_delta = (tenure/3).max(2);
    
    let max_no_improve = if div_period > 0 { div_period.max(60) } else { (max_iterations/2).max(60) };
    let mut pair_offsets = vec![0usize; ds.num_machines + 1];
    let mut node_local = vec![0usize; n];
    let mut current_pos = vec![0usize; n];
    let mut node_machine = vec![0usize; n];
    for m in 0..ds.num_machines {
        let seq = &ds.machine_seq[m];
        for (i, &node) in seq.iter().enumerate() {
            node_local[node] = i;
            current_pos[node] = i;
            node_machine[node] = m;
        }
        let len = seq.len();
        pair_offsets[m + 1] = pair_offsets[m] + len.saturating_mul(len.saturating_sub(1)) / 2;
    }
    let mut tabu_expiry = vec![0usize; pair_offsets[ds.num_machines]];
    let mut no_improve = 0usize;
    
    let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (initial_mk as u64).wrapping_shl(16) ^ (n as u64).wrapping_mul(0x517CC1B727220A95) ^ traj_salt.wrapping_mul(0x2545F4914F6CDD1D);
    
    let mut tseed: u64 = ((challenge.seed[0] as u64).wrapping_mul(0xD1B54A32D192ED03)
        ^ (initial_mk as u64).wrapping_mul(0xA24BAED4963EE407)
        ^ (n as u64).wrapping_shl(32)) | 1;

    let mut aseed: u64 = ((challenge.seed[0] as u64).wrapping_mul(0x2545F4914F6CDD1D)
        ^ traj_salt.wrapping_mul(0xBF58476D1CE4E5B9)
        ^ (initial_mk as u64).wrapping_shl(24)) | 1;
    // `qend[o] == tail[o] + node_pt[o]`. Starting value: the former `tail` was zero, so what
    // the consumers read was worth `node_pt[o]` - which is exactly `node_pt.clone()`.
    let mut qend = ds.node_pt.clone(); let mut back_deg = vec![0u16; n]; let mut back_stack: Vec<usize> = Vec::with_capacity(n);
    let mut machine_pred_node = vec![NONE_USIZE; n]; let mut job_pred_node = vec![NONE_USIZE; n];
    let mut job_back_deg = vec![0u16; n];
    let mut crit_pos_by_machine: Vec<Vec<usize>> = (0..ds.num_machines).map(|_| Vec::new()).collect();
    let mut crit_pos_machines: Vec<usize> = Vec::with_capacity(ds.num_machines);
    let mut machine_last_pick = vec![0usize; ds.num_machines];
    let mut best_global_pos = node_local.clone();
    let mut guided_kicks: Vec<(u32, usize, usize)> = Vec::with_capacity(ds.num_machines);
    let mut guided_used_machine = vec![0u8; ds.num_machines];
    for j in 0..ds.num_jobs { let base = ds.job_offsets[j]; let end = ds.job_offsets[j+1]; for k in (base+1)..end { job_pred_node[k] = k-1; } }
    for i in 0..n { if ds.job_succ[i] != NONE_USIZE { job_back_deg[i] += 1; } }
    rebuild_machine_pred_nodes(&ds, &mut machine_pred_node);
    let mut indeg_full = vec![0u16; n];
    // Fixed buffer plus length, instead of a `Vec`: see `eval_disj_topo`.
    let mut topo: Vec<usize> = vec![0usize; n];
    let mut topo_len = 0usize;
    if fast_eval {
        rebuild_machine_succ_indeg(&ds, &mut buf.machine_succ, &mut indeg_full);

        if eval_disj_topo(&ds, &mut buf, &indeg_full, &mut topo, &mut topo_len).is_none() { return Ok(None) }
    }
    let kick_threshold =(max_no_improve*2/3).max(40); let diversify_threshold = (max_no_improve/3).max(20); let diversify_unit = (pre.avg_op_min * (0.10 + 0.16*pre.jobshopness + 0.06*pre.high_flex)).max(1.0) as u32;
    let mut kicks_left = 4usize;
    
    let mut cycles_left = if div_period > 0 { (max_iterations / max_no_improve.max(1) / 2).max(1) } else { 0 };
    
    let mut exact_left = if exact_topk > 0 { (max_iterations / 16).max(1) } else { 0 };
    let mut top_cand = [(0usize, 0usize); 4];
    let mut top_n = 0usize;
    for iter in 0..max_iterations {
        if no_improve >= max_no_improve {
            
            if kicks_left == 0 {
                if cycles_left > 0 { cycles_left -= 1; kicks_left = 4; } else { break; }
            }
            ds.machine_seq.clone_from(&best_global_machine_seq); no_improve = 0; kicks_left -= 1; tabu_expiry.fill(0);
            rebuild_machine_pred_nodes(&ds, &mut machine_pred_node);
            for m in 0..ds.num_machines {
                for (i, &node) in ds.machine_seq[m].iter().enumerate() {
                    current_pos[node] = i;
                    node_machine[node] = m;
                }
            }
            if fast_eval { rebuild_machine_succ_indeg(&ds, &mut buf.machine_succ, &mut indeg_full); }
            let Some((mk, node)) = (if fast_eval { eval_disj_topo(&ds, &mut buf, &indeg_full, &mut topo, &mut topo_len) } else { eval_disj(&ds, &mut buf) }) else { if mid_abort_rescue > 0 { break } else { return Ok(None) } };
            cur_mk = mk; mk_node = node;
            continue;
        }
        
        let (kick_step, kick_ord) = if kick_spread > 0 {
            let step = (max_no_improve / kick_spread).max(1);
            (step, if step > 0 { no_improve / step } else { 0 })
        } else {
            (kick_threshold, no_improve / kick_threshold)
        };
        if no_improve > 0 && no_improve % kick_step == 0 && kicks_left > 0 && (kick_spread == 0 || kick_ord <= 4) {
            let kick_phase = if kick_spread > 0 { 1usize } else { kick_ord };
            let use_base_first = (kick_phase & 1) == 1;
            let ref_first = if use_base_first { &node_local[..] } else { &best_global_pos[..] };
            let ref_second = if use_base_first { &best_global_pos[..] } else { &node_local[..] };
            collect_guided_kick_moves(&ds, &buf.best_pred, mk_node, &current_pos, &node_machine, ref_first, &mut guided_used_machine, &mut guided_kicks);
            let ref_pos = if guided_kicks.is_empty() {
                collect_guided_kick_moves(&ds, &buf.best_pred, mk_node, &current_pos, &node_machine, ref_second, &mut guided_used_machine, &mut guided_kicks);
                ref_second
            } else {
                ref_first
            };
            if !guided_kicks.is_empty() {
                guided_kicks.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)).then_with(|| a.2.cmp(&b.2)));
                let num_kicks = (2 + kick_phase).min(4);
                let mut applied = 0usize;
                for &(_, m, pos) in &guided_kicks {
                    if applied >= num_kicks { break; }
                    if pos + 1 >= ds.machine_seq[m].len() { continue; }
                    let node_a = ds.machine_seq[m][pos];
                    let node_b = ds.machine_seq[m][pos + 1];
                    if ref_pos[node_a] <= ref_pos[node_b] { continue; }
                    ds.machine_seq[m].swap(pos, pos + 1);
                    current_pos[node_a] = pos + 1;
                    current_pos[node_b] = pos;
                    machine_last_pick[m] = iter;
                    applied += 1;
                }
            }
            kicks_left -= 1;
            
            if kick_spread > 0 { no_improve += 1; }
            rebuild_machine_pred_nodes(&ds, &mut machine_pred_node);
            if fast_eval { rebuild_machine_succ_indeg(&ds, &mut buf.machine_succ, &mut indeg_full); }
            
            let ev_kick = if fast_eval { eval_disj_topo(&ds, &mut buf, &indeg_full, &mut topo, &mut topo_len) } else { eval_disj(&ds, &mut buf) };
            let Some((mk, node)) = ev_kick else {
                if mid_abort_rescue >= 2 {
                    ds.machine_seq.clone_from(&best_global_machine_seq);
                    rebuild_machine_pred_nodes(&ds, &mut machine_pred_node);
                    for m in 0..ds.num_machines {
                        for (i, &node) in ds.machine_seq[m].iter().enumerate() {
                            current_pos[node] = i;
                            node_machine[node] = m;
                        }
                    }
                    if fast_eval { rebuild_machine_succ_indeg(&ds, &mut buf.machine_succ, &mut indeg_full); }
                    match if fast_eval { eval_disj_topo(&ds, &mut buf, &indeg_full, &mut topo, &mut topo_len) } else { eval_disj(&ds, &mut buf) } {
                        Some((mk_r, node_r)) => { cur_mk = mk_r; mk_node = node_r; continue; }
                        None => break,
                    }
                } else if mid_abort_rescue > 0 { break } else { return Ok(None) }
            };
            cur_mk = mk; mk_node = node;
            continue;
        }
        if iter > 0 { if cur_mk < best_global_mk { best_global_mk = cur_mk; best_global_machine_seq.clone_from(&ds.machine_seq); rebuild_machine_pos_map(&best_global_machine_seq, &mut best_global_pos); no_improve = 0; } else { no_improve += 1; } }

        if fast_eval {
            qend_from_topo(&ds, &buf.machine_succ, &topo[..topo_len], &mut qend);
        } else {
            // Same computation, written in the `qend` representation: `contrib > tail[x]` is
            // equivalent to `contrib + pt[x] > qend[x]`, the same comparison shifted by the
            // same constant.
            qend.copy_from_slice(&ds.node_pt);
            back_deg.copy_from_slice(&job_back_deg);
            for i in 0..n { if buf.machine_succ[i] != NONE_USIZE { back_deg[i] += 1; } }
            back_stack.clear(); for i in 0..n { if back_deg[i] == 0 { back_stack.push(i); } }
            while let Some(nd) = back_stack.pop() {
                let contrib = qend[nd];
                let jp = job_pred_node[nd]; if jp != NONE_USIZE { let c = contrib.saturating_add(ds.node_pt[jp]); if c > qend[jp] { qend[jp] = c; } back_deg[jp] = back_deg[jp].saturating_sub(1); if back_deg[jp] == 0 { back_stack.push(jp); } }
                let mp = machine_pred_node[nd]; if mp != NONE_USIZE { let c = contrib.saturating_add(ds.node_pt[mp]); if c > qend[mp] { qend[mp] = c; } back_deg[mp] = back_deg[mp].saturating_sub(1); if back_deg[mp] == 0 { back_stack.push(mp); } }
            }
        }
        for &m in &crit_pos_machines { crit_pos_by_machine[m].clear(); }
        crit_pos_machines.clear();
        let mut u = mk_node;
        while u != NONE_USIZE {
            let m = node_machine[u];
            if crit_pos_by_machine[m].is_empty() { crit_pos_machines.push(m); }
            crit_pos_by_machine[m].push(current_pos[u]);
            u = buf.best_pred[u];
        }
        // No more sort: the move loop now scans `0..num_machines`, skipping the machines with
        // no critical position, which returns the same set in the same increasing order.
        // `crit_pos_machines` is now only used to reset the lists that were touched.
        let diversify = ((no_improve.saturating_sub(diversify_threshold) as f64) / (max_no_improve.saturating_sub(diversify_threshold).max(1) as f64)).clamp(0.0, 1.0);
        let mut best_move: Option<(usize,usize,u32)> = None; let mut best_move_key = u32::MAX;
        
        top_n = 0;
        let mut relaxed_move: Option<(usize,usize,u32)> = None; let mut relaxed_key = u32::MAX;
        
        let mut best_tie_n: u64 = 0; let mut best_tie_block: usize = 0;
        let mut relaxed_tie_n: u64 = 0; let mut relaxed_tie_block: usize = 0;
        
        let mut best_snd: u32 = u32::MAX; let mut relaxed_snd: u32 = u32::MAX;
        for m in 0..ds.num_machines {
            let positions = &mut crit_pos_by_machine[m];
            if positions.len() < 2 { continue; }
            // The critical path is walked back from `mk_node` towards its predecessors. Two
            // operations on the same machine are ordered by their position (the machine arc
            // always goes from position i to i+1), so walking back they arrive in decreasing
            // position: `positions` is already sorted in reverse. Measurement: 4 141 000 lists
            // over 3 instances, 100 % strictly decreasing. We check it anyway in O(k) and fall
            // back on the sort otherwise - in both cases the array obtained is the same
            // increasing array, so no trajectory can change.
            let mut descending = true;
            for w in 1..positions.len() {
                if positions[w - 1] <= positions[w] { descending = false; break; }
            }
            if descending { positions.reverse(); } else { positions.sort_unstable(); }
            let seq = &ds.machine_seq[m];
            let mut run_start = positions[0];
            let mut run_end = positions[0];
            let mut prev_pos = positions[0];
            let mut prev_node = seq[prev_pos];
            for idx in 1..positions.len() {
                let pos = positions[idx];
                let node = seq[pos];
                if pos == prev_pos + 1 && buf.start[node] == buf.start[prev_node].saturating_add(ds.node_pt[prev_node]) {
                    run_end = pos;
                } else {
                    if run_end > run_start {
                        let block_len = run_end-run_start+1;
                        let mut swap_positions = [run_start,NONE_USIZE]; let num_swaps = if block_len>=3 { swap_positions[1]=run_end-1; 2 } else { 1 };
                        for si in 0..num_swaps {
                            let pos=swap_positions[si]; if pos+1>=seq.len() { continue; }
                            let node_u=seq[pos]; let node_v=seq[pos+1];
                            let (est_mk, est_snd) = estimate_swap_mk2(node_u, node_v, &buf.start, &qend, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ);
                            let lu = node_local[node_u]; let lv = node_local[node_v];
                            let (a, b) = if lu < lv { (lu, lv) } else { (lv, lu) };
                            let tabu_idx = pair_offsets[m] + b * (b - 1) / 2 + a;
                            let is_tabu = tabu_expiry[tabu_idx] > iter; let aspiration=est_mk<best_global_mk;
                            let age = iter.saturating_sub(machine_last_pick[m]).min(9) as u32;
                            let block_bonus = block_len.saturating_sub(2).min(3) as u32;
                            let div_bonus = if diversify > 0.0 {
                                ((diversify_unit as f64) * diversify * (0.35*(age as f64) + 0.75*(block_bonus as f64)) / 3.0) as u32
                            } else { 0 };
                            let adj_mk = est_mk.saturating_sub(div_bonus);

                            if !is_tabu||aspiration {
                                
                                if exact_topk != 0 {
                                    if adj_mk<best_move_key { top_cand[0] = (m, pos); top_n = 1; }
                                    else if adj_mk==best_move_key && top_n < exact_topk { top_cand[top_n] = (m, pos); top_n += 1; }
                                }
                                if adj_mk<best_move_key { best_move_key=adj_mk; best_move=Some((m,pos,est_mk)); best_tie_n=1; best_tie_block=block_len; best_snd=est_snd; }
                                else if sel_key!=0 && adj_mk==best_move_key { if sel_key_take(sel_key, est_snd, &mut best_snd) { best_move=Some((m,pos,est_mk)); } }
                                else if tie_mode!=0 && adj_mk==best_move_key && tie_break_take(tie_mode, block_len, &mut best_tie_n, &mut best_tie_block, &mut tseed) { best_move=Some((m,pos,est_mk)); }
                            }
                            if adj_mk<relaxed_key { relaxed_key=adj_mk; relaxed_move=Some((m,pos,est_mk)); relaxed_tie_n=1; relaxed_tie_block=block_len; relaxed_snd=est_snd; }
                            else if sel_key!=0 && adj_mk==relaxed_key { if sel_key_take(sel_key, est_snd, &mut relaxed_snd) { relaxed_move=Some((m,pos,est_mk)); } }
                            else if tie_mode!=0 && adj_mk==relaxed_key && tie_break_take(tie_mode, block_len, &mut relaxed_tie_n, &mut relaxed_tie_block, &mut tseed) { relaxed_move=Some((m,pos,est_mk)); }
                        }
                    }
                    run_start = pos;
                    run_end = pos;
                }
                prev_pos = pos;
                prev_node = node;
            }
            if run_end > run_start {
                let block_len = run_end-run_start+1;
                let mut swap_positions = [run_start,NONE_USIZE]; let num_swaps = if block_len>=3 { swap_positions[1]=run_end-1; 2 } else { 1 };
                for si in 0..num_swaps {
                    let pos=swap_positions[si]; if pos+1>=seq.len() { continue; }
                    let node_u=seq[pos]; let node_v=seq[pos+1];
                    let (est_mk, est_snd) = estimate_swap_mk2(node_u, node_v, &buf.start, &qend, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ);
                    let lu = node_local[node_u]; let lv = node_local[node_v];
                    let (a, b) = if lu < lv { (lu, lv) } else { (lv, lu) };
                    let tabu_idx = pair_offsets[m] + b * (b - 1) / 2 + a;
                    let is_tabu = tabu_expiry[tabu_idx] > iter; let aspiration=est_mk<best_global_mk;
                    let age = iter.saturating_sub(machine_last_pick[m]).min(9) as u32;
                    let block_bonus = block_len.saturating_sub(2).min(3) as u32;
                    let div_bonus = if diversify > 0.0 {
                        ((diversify_unit as f64) * diversify * (0.35*(age as f64) + 0.75*(block_bonus as f64)) / 3.0) as u32
                    } else { 0 };
                    let adj_mk = est_mk.saturating_sub(div_bonus);

                    if !is_tabu||aspiration {
                        
                        if exact_topk != 0 {
                            if adj_mk<best_move_key { top_cand[0] = (m, pos); top_n = 1; }
                            else if adj_mk==best_move_key && top_n < exact_topk { top_cand[top_n] = (m, pos); top_n += 1; }
                        }
                        if adj_mk<best_move_key { best_move_key=adj_mk; best_move=Some((m,pos,est_mk)); best_tie_n=1; best_tie_block=block_len; best_snd=est_snd; }
                        else if sel_key!=0 && adj_mk==best_move_key { if sel_key_take(sel_key, est_snd, &mut best_snd) { best_move=Some((m,pos,est_mk)); } }
                        else if tie_mode!=0 && adj_mk==best_move_key && tie_break_take(tie_mode, block_len, &mut best_tie_n, &mut best_tie_block, &mut tseed) { best_move=Some((m,pos,est_mk)); }
                    }
                    if adj_mk<relaxed_key { relaxed_key=adj_mk; relaxed_move=Some((m,pos,est_mk)); relaxed_tie_n=1; relaxed_tie_block=block_len; relaxed_snd=est_snd; }
                    else if sel_key!=0 && adj_mk==relaxed_key { if sel_key_take(sel_key, est_snd, &mut relaxed_snd) { relaxed_move=Some((m,pos,est_mk)); } }
                    else if tie_mode!=0 && adj_mk==relaxed_key && tie_break_take(tie_mode, block_len, &mut relaxed_tie_n, &mut relaxed_tie_block, &mut tseed) { relaxed_move=Some((m,pos,est_mk)); }
                }
            }
        }

        if exact_topk != 0 && top_n >= 2 && exact_left > 0 && best_move.is_some() {
            exact_left -= 1;
            let mut best_true = u32::MAX;
            let mut best_true_cand: Option<(usize, usize)> = None;
            for ci in 0..top_n {
                let (cm, cpos) = top_cand[ci];
                probe_swap_toggle(&mut ds, cm, cpos, &mut current_pos, &mut machine_pred_node, &mut buf.machine_succ, &mut indeg_full, fast_eval);
                let probe = if fast_eval { eval_disj_topo(&ds, &mut buf, &indeg_full, &mut topo, &mut topo_len) } else { eval_disj(&ds, &mut buf) };
                probe_swap_toggle(&mut ds, cm, cpos, &mut current_pos, &mut machine_pred_node, &mut buf.machine_succ, &mut indeg_full, fast_eval);
                if let Some((true_mk, _)) = probe {
                    if true_mk < best_true { best_true = true_mk; best_true_cand = Some((cm, cpos)); }
                }
            }
            if let Some((cm, cpos)) = best_true_cand { best_move = Some((cm, cpos, best_true)); }
        }
        let chosen = best_move.or(relaxed_move);

        if sa_accept > 0 {
            if let Some((_, _, cand_mk)) = chosen {
                if cand_mk > cur_mk {
                    let temp = (sa_accept as f64) * (1.0 - (iter as f64) / (max_iterations.max(1) as f64));
                    let accept = if temp <= 1e-9 {
                        false
                    } else {
                        aseed ^= aseed.wrapping_shl(13); aseed ^= aseed.wrapping_shr(7); aseed ^= aseed.wrapping_shl(17);
                        let u = ((aseed >> 11) as f64) * (1.0f64 / 9007199254740992.0);
                        u < (-((cand_mk - cur_mk) as f64) / temp).exp()
                    };
                    if !accept { continue; }
                }
            }
        }
        match chosen {
            Some((m,pos,_est)) => {
                let node_a=ds.machine_seq[m][pos]; let node_b=ds.machine_seq[m][pos+1];
                ds.machine_seq[m].swap(pos,pos+1);
                current_pos[node_a] = pos + 1;
                current_pos[node_b] = pos;
                machine_last_pick[m] = iter;
                let seq = &ds.machine_seq[m];
                let prev = if pos > 0 { seq[pos - 1] } else { NONE_USIZE };
                machine_pred_node[seq[pos]] = prev;
                machine_pred_node[seq[pos + 1]] = seq[pos];
                if pos + 2 < seq.len() { machine_pred_node[seq[pos + 2]] = seq[pos + 1]; }
                pseed^=pseed.wrapping_shl(13); pseed^=pseed.wrapping_shr(7); pseed^=pseed.wrapping_shl(17);
                let offset=(pseed%((2*tenure_delta+1) as u64)) as usize;
                let progress=(iter as f64)/(max_iterations as f64); let late_bonus=if progress>0.6{((progress-0.6)*10.0) as usize}else{0};
                let this_tenure=(tenure+offset+late_bonus).saturating_sub(tenure_delta);
                let la = node_local[node_a]; let lb = node_local[node_b];
                let (a, b) = if la < lb { (la, lb) } else { (lb, la) };
                let tabu_idx = pair_offsets[m] + b * (b - 1) / 2 + a;
                tabu_expiry[tabu_idx] = iter + this_tenure;
                if fast_eval {
                    if pos > 0 { buf.machine_succ[seq[pos - 1]] = seq[pos]; }
                    buf.machine_succ[seq[pos]] = seq[pos + 1];
                    buf.machine_succ[seq[pos + 1]] = if pos + 2 < seq.len() { seq[pos + 2] } else { NONE_USIZE };
                    if pos == 0 {
                        indeg_full[node_a] = indeg_full[node_a].saturating_add(1);
                        indeg_full[node_b] = indeg_full[node_b].saturating_sub(1);
                    }
                }
                let Some((mk, node)) = (if fast_eval { eval_disj_topo(&ds, &mut buf, &indeg_full, &mut topo, &mut topo_len) } else { eval_disj(&ds, &mut buf) }) else { if mid_abort_rescue > 0 { break } else { return Ok(None) } };
                cur_mk = mk; mk_node = node;
            }
            None => break,
        }
    }
    if cur_mk < best_global_mk { best_global_mk = cur_mk; best_global_machine_seq.clone_from(&ds.machine_seq); rebuild_machine_pos_map(&best_global_machine_seq, &mut best_global_pos); }
    if best_global_mk >= initial_mk { return Ok(None); }
    ds.machine_seq = best_global_machine_seq;
    let Some((mk_final,_)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let sol = disj_to_solution(pre, &ds, &buf.start)?;
    Ok(Some((sol, mk_final)))
}

#[inline]
fn estimate_swap_mk(u: usize, v: usize, heads: &[u32], qend: &[u32], pt: &[u32], job_pred: &[usize], job_succ: &[usize], machine_pred: &[usize], machine_succ: &[usize]) -> u32 {
    let mp_u=machine_pred[u]; let ms_v=machine_succ[v]; let jp_v=job_pred[v]; let jp_u=job_pred[u]; let js_u=job_succ[u]; let js_v=job_succ[v];
    let r_jp_v=if jp_v!=NONE_USIZE{heads[jp_v].saturating_add(pt[jp_v])}else{0}; let r_mp_u=if mp_u!=NONE_USIZE{heads[mp_u].saturating_add(pt[mp_u])}else{0};
    let new_r_v=r_jp_v.max(r_mp_u); let r_jp_u=if jp_u!=NONE_USIZE{heads[jp_u].saturating_add(pt[jp_u])}else{0}; let new_r_u=r_jp_u.max(new_r_v.saturating_add(pt[v]));
    // `qend[x]` is `pt[x] + tails[x]`: the sum is no longer rebuilt here.
    let q_js_u=if js_u!=NONE_USIZE{qend[js_u]}else{0}; let q_ms_v=if ms_v!=NONE_USIZE{qend[ms_v]}else{0};
    let new_q_u=q_js_u.max(q_ms_v); let q_js_v=if js_v!=NONE_USIZE{qend[js_v]}else{0}; let new_q_v=q_js_v.max(pt[u].saturating_add(new_q_u));
    let path_v=new_r_v.saturating_add(pt[v]).saturating_add(new_q_v); let path_u=new_r_u.saturating_add(pt[u]).saturating_add(new_q_u);
    path_v.max(path_u)
}

#[inline]
fn estimate_swap_mk2(u: usize, v: usize, heads: &[u32], qend: &[u32], pt: &[u32], job_pred: &[usize], job_succ: &[usize], machine_pred: &[usize], machine_succ: &[usize]) -> (u32, u32) {
    let mp_u=machine_pred[u]; let ms_v=machine_succ[v]; let jp_v=job_pred[v]; let jp_u=job_pred[u]; let js_u=job_succ[u]; let js_v=job_succ[v];
    let r_jp_v=if jp_v!=NONE_USIZE{heads[jp_v].saturating_add(pt[jp_v])}else{0}; let r_mp_u=if mp_u!=NONE_USIZE{heads[mp_u].saturating_add(pt[mp_u])}else{0};
    let new_r_v=r_jp_v.max(r_mp_u); let r_jp_u=if jp_u!=NONE_USIZE{heads[jp_u].saturating_add(pt[jp_u])}else{0}; let new_r_u=r_jp_u.max(new_r_v.saturating_add(pt[v]));
    // `qend[x]` is `pt[x] + tails[x]`: the sum is no longer rebuilt here.
    let q_js_u=if js_u!=NONE_USIZE{qend[js_u]}else{0}; let q_ms_v=if ms_v!=NONE_USIZE{qend[ms_v]}else{0};
    let new_q_u=q_js_u.max(q_ms_v); let q_js_v=if js_v!=NONE_USIZE{qend[js_v]}else{0}; let new_q_v=q_js_v.max(pt[u].saturating_add(new_q_u));
    let path_v=new_r_v.saturating_add(pt[v]).saturating_add(new_q_v); let path_u=new_r_u.saturating_add(pt[u]).saturating_add(new_q_u);
    (path_v.max(path_u), path_v.min(path_u))
}

#[inline(always)]
fn sel_key_take(sel_key: usize, snd: u32, best_snd: &mut u32) -> bool {
    match sel_key {
        1 => { if snd < *best_snd { *best_snd = snd; true } else { false } }
        2 => { if snd > *best_snd { *best_snd = snd; true } else { false } }
        _ => false,
    }
}

#[inline(always)]
fn tie_break_take(tie_mode: usize, block_len: usize, tie_n: &mut u64, tie_block: &mut usize, tseed: &mut u64) -> bool {
    *tie_n = tie_n.saturating_add(1);
    match tie_mode {
        1 => {
            *tseed ^= tseed.wrapping_shl(13);
            *tseed ^= tseed.wrapping_shr(7);
            *tseed ^= tseed.wrapping_shl(17);
            (*tseed % (*tie_n).max(1)) == 0
        }
        2 => { if block_len > *tie_block { *tie_block = block_len; true } else { false } }
        3 => true,
        _ => false,
    }
}

#[inline]
fn probe_swap_toggle(ds: &mut DisjSchedule, m: usize, pos: usize, current_pos: &mut [usize], machine_pred_node: &mut [usize], machine_succ: &mut [usize], indeg_full: &mut [u16], fast_eval: bool) {
    let node_a = ds.machine_seq[m][pos];
    let node_b = ds.machine_seq[m][pos + 1];
    ds.machine_seq[m].swap(pos, pos + 1);
    current_pos[node_a] = pos + 1;
    current_pos[node_b] = pos;
    let seq = &ds.machine_seq[m];
    let prev = if pos > 0 { seq[pos - 1] } else { NONE_USIZE };
    machine_pred_node[seq[pos]] = prev;
    machine_pred_node[seq[pos + 1]] = seq[pos];
    if pos + 2 < seq.len() { machine_pred_node[seq[pos + 2]] = seq[pos + 1]; }
    if fast_eval {
        if pos > 0 { machine_succ[seq[pos - 1]] = seq[pos]; }
        machine_succ[seq[pos]] = seq[pos + 1];
        machine_succ[seq[pos + 1]] = if pos + 2 < seq.len() { seq[pos + 2] } else { NONE_USIZE };
        if pos == 0 {
            indeg_full[node_a] = indeg_full[node_a].saturating_add(1);
            indeg_full[node_b] = indeg_full[node_b].saturating_sub(1);
        }
    }
}

#[inline]
fn relocate_machine_seq(seq: &mut [usize], from: usize, to: usize) {
    if from < to {
        seq[from..=to].rotate_left(1);
    } else if to < from {
        seq[to..=from].rotate_right(1);
    }
}

#[inline]
fn inherit_machine_consensus_order(
    child_seq: &mut Vec<usize>,
    better_seq: &[usize],
    worse_seq: &[usize],
    elite_machine_seqs: &[&[usize]],
    pos_buf: &mut [usize],
    sum_buf: &mut [u32],
    cnt_buf: &mut [u16],
    pair_buf: &mut Vec<(usize, usize, i32)>,
) {
    if better_seq.len() <= 1 || better_seq.len() != worse_seq.len() {
        if child_seq.len() != better_seq.len() {
            child_seq.clear();
            child_seq.extend_from_slice(better_seq);
        }
        return;
    }
    if child_seq.len() != better_seq.len() {
        child_seq.clear();
        child_seq.extend_from_slice(better_seq);
    }

    for &node in better_seq {
        pos_buf[node] = 0;
        sum_buf[node] = 0;
        cnt_buf[node] = 0;
    }
    for (idx, &node) in better_seq.iter().enumerate() {
        pos_buf[node] = idx;
    }
    for &eseq in elite_machine_seqs {
        if eseq.len() != better_seq.len() { continue; }
        for (idx, &node) in eseq.iter().enumerate() {
            sum_buf[node] = sum_buf[node].saturating_add(idx as u32);
            cnt_buf[node] = cnt_buf[node].saturating_add(1);
        }
    }

    pair_buf.clear();
    for win in worse_seq.windows(2) {
        let u = win[0];
        let v = win[1];
        let pu = pos_buf[u];
        let pv = pos_buf[v];
        if pu < pv { continue; }
        let cu = cnt_buf[u];
        let cv = cnt_buf[v];
        if cu == 0 || cv == 0 { continue; }
        let avg_u = (sum_buf[u] as f64) / (cu as f64);
        let avg_v = (sum_buf[v] as f64) / (cv as f64);
        let gap = avg_v - avg_u;
        if gap <= 0.55 { continue; }
        let score = ((gap * 16.0) as i32) + ((pu - pv).min(10) as i32);
        pair_buf.push((u, v, score));
    }
    if pair_buf.is_empty() { return; }

    pair_buf.sort_unstable_by(|a, b| b.2.cmp(&a.2));
    let apply_cap = if better_seq.len() > 14 { 2usize } else { 1usize };
    let mut applied = 0usize;
    for &(u, v, _) in pair_buf.iter() {
        if applied >= apply_cap { break; }
        for (idx, &node) in child_seq.iter().enumerate() {
            pos_buf[node] = idx;
        }
        let from = pos_buf[u];
        let to = pos_buf[v];
        if from < to { continue; }
        child_seq[to..=from].rotate_right(1);
        applied += 1;
    }
}

#[inline]
fn invert_job_bias_guidance(jb: &[f64]) -> Vec<f64> {
    if jb.is_empty() { return Vec::new(); }
    let mean = jb.iter().copied().sum::<f64>() / (jb.len() as f64);
    let mut anti = Vec::with_capacity(jb.len());
    for &v in jb {
        anti.push((mean - v) * 0.75);
    }
    anti
}

#[inline]
fn invert_machine_penalty_guidance(mp: &[f64]) -> Vec<f64> {
    if mp.is_empty() { return Vec::new(); }
    let mean = mp.iter().copied().sum::<f64>() / (mp.len() as f64);
    let mut anti = Vec::with_capacity(mp.len());
    for &v in mp {
        anti.push((0.55*mean + 0.45*(1.0 - v.clamp(0.0, 1.0))).clamp(0.0, 1.0));
    }
    anti
}

fn critical_block_move_local_search_ex_disj(
    ds: &mut DisjSchedule,
    buf: &mut EvalBuf,
    max_rounds: usize,
    max_iters: usize,
    stall_limit: usize,
) -> Option<u32> {
    let n = ds.n;
    let Some((initial_mk, mut mk_node)) = eval_disj(ds, buf) else { return None };
    let mut test_buf_a = EvalBuf::new(n);
    let mut test_buf_b = EvalBuf::new(n);
    let mut cur_mk = initial_mk;
    // Same pre-added representation as elsewhere: `qend[o] == tail[o] + node_pt[o]`.
    let mut qend = ds.node_pt.clone();
    let mut back_deg = vec![0u16; n];
    let mut back_stack: Vec<usize> = Vec::with_capacity(n);
    let mut machine_pred_node = vec![NONE_USIZE; n];
    let mut job_pred_node = vec![NONE_USIZE; n];
    let mut moves: Vec<(u32,u16,usize,usize)> = Vec::with_capacity(64);
    let mut current_pos = vec![0usize; n];
    let mut node_machine = vec![0usize; n];
    let mut crit_positions: Vec<(usize,usize)> = Vec::with_capacity(n);

    for j in 0..ds.num_jobs {
        let base = ds.job_offsets[j];
        let end = ds.job_offsets[j + 1];
        for k in (base + 1)..end {
            job_pred_node[k] = k - 1;
        }
    }
    for m in 0..ds.num_machines {
        for (i, &node) in ds.machine_seq[m].iter().enumerate() {
            current_pos[node] = i;
            node_machine[node] = m;
        }
    }

    let iter_limit = max_iters.max(max_rounds).max(1);
    let stall_cap = stall_limit.max(max_rounds).max(1);
    let mut stalled = 0usize;

    for _ in 0..iter_limit {
        rebuild_machine_pred_nodes(ds, &mut machine_pred_node);
        qend.copy_from_slice(&ds.node_pt);
        back_deg.fill(0);
        for i in 0..n {
            if ds.job_succ[i] != NONE_USIZE { back_deg[i] += 1; }
            if buf.machine_succ[i] != NONE_USIZE { back_deg[i] += 1; }
        }
        back_stack.clear();
        for i in 0..n {
            if back_deg[i] == 0 { back_stack.push(i); }
        }
        while let Some(nd) = back_stack.pop() {
            let contrib = qend[nd];
            let jp = job_pred_node[nd];
            if jp != NONE_USIZE {
                let c = contrib.saturating_add(ds.node_pt[jp]);
                if c > qend[jp] { qend[jp] = c; }
                back_deg[jp] = back_deg[jp].saturating_sub(1);
                if back_deg[jp] == 0 { back_stack.push(jp); }
            }
            let mp = machine_pred_node[nd];
            if mp != NONE_USIZE {
                let c = contrib.saturating_add(ds.node_pt[mp]);
                if c > qend[mp] { qend[mp] = c; }
                back_deg[mp] = back_deg[mp].saturating_sub(1);
                if back_deg[mp] == 0 { back_stack.push(mp); }
            }
        }

        crit_positions.clear();
        let mut u = mk_node;
        while u != NONE_USIZE {
            crit_positions.push((node_machine[u], current_pos[u]));
            u = buf.best_pred[u];
        }
        if crit_positions.len() > 1 { crit_positions.sort_unstable(); }

        moves.clear();
        let mut cp_i = 0usize;
        while cp_i < crit_positions.len() {
            let m = crit_positions[cp_i].0;
            let seq = &ds.machine_seq[m];
            let mut run_start = crit_positions[cp_i].1;
            let mut run_end = run_start;
            let mut prev_pos = run_start;
            let mut prev_node = seq[prev_pos];
            cp_i += 1;
            while cp_i < crit_positions.len() && crit_positions[cp_i].0 == m {
                let pos = crit_positions[cp_i].1;
                let node = seq[pos];
                if pos == prev_pos + 1 && buf.start[node] == buf.start[prev_node].saturating_add(ds.node_pt[prev_node]) {
                    run_end = pos;
                } else {
                    if run_end > run_start {
                        let block_len = run_end - run_start + 1;
                        let mut swap_positions = [run_start, NONE_USIZE];
                        let num_swaps = if block_len >= 3 { swap_positions[1] = run_end - 1; 2 } else { 1 };
                        let block_len_u16 = block_len.min(u16::MAX as usize) as u16;
                        for si in 0..num_swaps {
                            let pos = swap_positions[si];
                            if pos + 1 >= seq.len() { continue; }
                            let node_u = seq[pos];
                            let node_v = seq[pos + 1];
                            let est_mk = estimate_swap_mk(node_u, node_v, &buf.start, &qend, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ);
                            if est_mk < cur_mk {
                                moves.push((est_mk, block_len_u16, m, pos));
                            }
                        }
                    }
                    run_start = pos;
                    run_end = pos;
                }
                prev_pos = pos;
                prev_node = node;
                cp_i += 1;
            }
            if run_end > run_start {
                let block_len = run_end - run_start + 1;
                let mut swap_positions = [run_start, NONE_USIZE];
                let num_swaps = if block_len >= 3 { swap_positions[1] = run_end - 1; 2 } else { 1 };
                let block_len_u16 = block_len.min(u16::MAX as usize) as u16;
                for si in 0..num_swaps {
                    let pos = swap_positions[si];
                    if pos + 1 >= seq.len() { continue; }
                    let node_u = seq[pos];
                    let node_v = seq[pos + 1];
                    let est_mk = estimate_swap_mk(node_u, node_v, &buf.start, &qend, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ);
                    if est_mk < cur_mk {
                        moves.push((est_mk, block_len_u16, m, pos));
                    }
                }
            }
        }

        if moves.is_empty() { break; }
        if moves.len() > 1 {
            moves.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)).then_with(|| a.2.cmp(&b.2)).then_with(|| a.3.cmp(&b.3)));
        }

        let eval_cap = moves.len().min(2);
        let mut best_idx = NONE_USIZE;
        let mut best_actual_mk = cur_mk;
        let mut best_actual_node = NONE_USIZE;
        let mut tested = [(NONE_USIZE, NONE_USIZE); 2];

        for idx in 0..eval_cap {
            let (_, _, m, pos) = moves[idx];
            if pos + 1 >= ds.machine_seq[m].len() { continue; }
            tested[idx] = (m, pos);
            let node_a = ds.machine_seq[m][pos];
            let node_b = ds.machine_seq[m][pos + 1];
            ds.machine_seq[m].swap(pos, pos + 1);
            current_pos[node_a] = pos + 1;
            current_pos[node_b] = pos;
            let res = if idx == 0 {
                eval_disj(ds, &mut test_buf_a)
            } else {
                eval_disj(ds, &mut test_buf_b)
            };
            if let Some((new_mk, new_node)) = res {
                if new_mk < best_actual_mk {
                    best_actual_mk = new_mk;
                    best_actual_node = new_node;
                    best_idx = idx;
                }
            }
            ds.machine_seq[m].swap(pos, pos + 1);
            current_pos[node_a] = pos;
            current_pos[node_b] = pos + 1;
        }

        if best_idx != NONE_USIZE {
            let (m, pos) = tested[best_idx];
            let node_a = ds.machine_seq[m][pos];
            let node_b = ds.machine_seq[m][pos + 1];
            ds.machine_seq[m].swap(pos, pos + 1);
            current_pos[node_a] = pos + 1;
            current_pos[node_b] = pos;
            cur_mk = best_actual_mk;
            mk_node = best_actual_node;
            if best_idx == 0 {
                core::mem::swap(buf, &mut test_buf_a);
            } else {
                core::mem::swap(buf, &mut test_buf_b);
            }
            stalled = 0;
        } else {
            let _ = (stalled, stall_cap);
            break;
        }
    }

    if cur_mk < initial_mk { Some(cur_mk) } else { None }
}

#[inline]
fn perturb_and_reoptimize_ils(
    pre: &Pre,
    challenge: &Challenge,
    base_sol: &Solution,
    num_perturb: usize,
    ls_rounds: usize,
    ls_iters: usize,
    ls_stall: usize,
) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n);
    let Some((start_mk, mut mk_node)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let mut cur_mk = start_mk;

    let mut current_pos = vec![0usize; ds.n];
    let mut node_machine = vec![0usize; ds.n];
    let mut crit_pos_by_machine: Vec<Vec<usize>> = (0..ds.num_machines).map(|_| Vec::new()).collect();
    let mut crit_pos_machines: Vec<usize> = Vec::with_capacity(ds.num_machines);
    let mut moved = 0usize;

    for _ in 0..num_perturb {
        for m in 0..ds.num_machines {
            crit_pos_by_machine[m].clear();
            for (pos, &node) in ds.machine_seq[m].iter().enumerate() {
                current_pos[node] = pos;
                node_machine[node] = m;
            }
        }
        crit_pos_machines.clear();
        let mut u = mk_node;
        while u != NONE_USIZE {
            let m = node_machine[u];
            if crit_pos_by_machine[m].is_empty() { crit_pos_machines.push(m); }
            crit_pos_by_machine[m].push(current_pos[u]);
            u = buf.best_pred[u];
        }

        let mut best_move: Option<(usize, usize, usize, u8, usize, u32)> = None;
        for &m in &crit_pos_machines {
            let positions = &mut crit_pos_by_machine[m];
            if positions.len() < 2 { continue; }
            positions.sort_unstable();
            let seq = &ds.machine_seq[m];
            let mut run_start = positions[0];
            let mut run_end = positions[0];
            let mut prev_pos = positions[0];
            let mut prev_node = seq[prev_pos];

            for idx in 1..positions.len() {
                let pos = positions[idx];
                let node = seq[pos];
                if pos == prev_pos + 1 && buf.start[node] == buf.start[prev_node].saturating_add(ds.node_pt[prev_node]) {
                    run_end = pos;
                } else {
                    if run_end > run_start {
                        let block_len = run_end - run_start + 1;
                        let pt_sum = seq[run_start..=run_end].iter().fold(0u32, |acc, &nd| acc.saturating_add(ds.node_pt[nd]));
                        let cand = if block_len >= 5 {
                            let head_mass = ds.node_pt[seq[run_start]].saturating_add(ds.node_pt[seq[run_start + 1]]);
                            let tail_mass = ds.node_pt[seq[run_end - 1]].saturating_add(ds.node_pt[seq[run_end]]);
                            if head_mass <= tail_mass {
                                (m, run_start, run_end - 1, 0u8, block_len, pt_sum)
                            } else {
                                (m, run_end, run_start + 1, 0u8, block_len, pt_sum)
                            }
                        } else if block_len == 4 {
                            let head_mass = ds.node_pt[seq[run_start]].saturating_add(ds.node_pt[seq[run_start + 1]]);
                            let tail_mass = ds.node_pt[seq[run_end - 1]].saturating_add(ds.node_pt[seq[run_end]]);
                            if head_mass <= tail_mass {
                                (m, run_start, 0usize, 1u8, block_len, pt_sum)
                            } else {
                                (m, run_end - 2, 0usize, 2u8, block_len, pt_sum)
                            }
                        } else if block_len == 3 {
                            if ds.node_pt[seq[run_start]] <= ds.node_pt[seq[run_end]] {
                                (m, run_start, 0usize, 1u8, block_len, pt_sum)
                            } else {
                                (m, run_start, 0usize, 2u8, block_len, pt_sum)
                            }
                        } else {
                            (m, run_start, 0usize, 3u8, block_len, pt_sum)
                        };
                        if best_move.as_ref().map_or(true, |&(_, _, _, _, best_len, best_sum)| block_len > best_len || (block_len == best_len && pt_sum > best_sum)) {
                            best_move = Some(cand);
                        }
                    }
                    run_start = pos;
                    run_end = pos;
                }
                prev_pos = pos;
                prev_node = node;
            }

            if run_end > run_start {
                let block_len = run_end - run_start + 1;
                let pt_sum = seq[run_start..=run_end].iter().fold(0u32, |acc, &nd| acc.saturating_add(ds.node_pt[nd]));
                let cand = if block_len >= 5 {
                    let head_mass = ds.node_pt[seq[run_start]].saturating_add(ds.node_pt[seq[run_start + 1]]);
                    let tail_mass = ds.node_pt[seq[run_end - 1]].saturating_add(ds.node_pt[seq[run_end]]);
                    if head_mass <= tail_mass {
                        (m, run_start, run_end - 1, 0u8, block_len, pt_sum)
                    } else {
                        (m, run_end, run_start + 1, 0u8, block_len, pt_sum)
                    }
                } else if block_len == 4 {
                    let head_mass = ds.node_pt[seq[run_start]].saturating_add(ds.node_pt[seq[run_start + 1]]);
                    let tail_mass = ds.node_pt[seq[run_end - 1]].saturating_add(ds.node_pt[seq[run_end]]);
                    if head_mass <= tail_mass {
                        (m, run_start, 0usize, 1u8, block_len, pt_sum)
                    } else {
                        (m, run_end - 2, 0usize, 2u8, block_len, pt_sum)
                    }
                } else if block_len == 3 {
                    if ds.node_pt[seq[run_start]] <= ds.node_pt[seq[run_end]] {
                        (m, run_start, 0usize, 1u8, block_len, pt_sum)
                    } else {
                        (m, run_start, 0usize, 2u8, block_len, pt_sum)
                    }
                } else {
                    (m, run_start, 0usize, 3u8, block_len, pt_sum)
                };
                if best_move.as_ref().map_or(true, |&(_, _, _, _, best_len, best_sum)| block_len > best_len || (block_len == best_len && pt_sum > best_sum)) {
                    best_move = Some(cand);
                }
            }
        }

        let Some((m, a, b, kind, _, _)) = best_move else { break; };

        match kind {
            0 => {
                let seq = &mut ds.machine_seq[m];
                relocate_machine_seq(seq, a, b);
            }
            1 => {
                let seq = &mut ds.machine_seq[m];
                seq[a..=a + 2].rotate_left(1);
            }
            2 => {
                let seq = &mut ds.machine_seq[m];
                seq[a..=a + 2].rotate_right(1);
            }
            _ => {
                let seq = &mut ds.machine_seq[m];
                seq[a..=a + 1].swap(0, 1);
            }
        }

        match eval_disj(&ds, &mut buf) {
            Some((new_mk, new_node)) => {
                cur_mk = new_mk;
                mk_node = new_node;
                moved += 1;
            }
            None => {
                match kind {
                    0 => {
                        let seq = &mut ds.machine_seq[m];
                        relocate_machine_seq(seq, b, a);
                    }
                    1 => {
                        let seq = &mut ds.machine_seq[m];
                        seq[a..=a + 2].rotate_right(1);
                    }
                    2 => {
                        let seq = &mut ds.machine_seq[m];
                        seq[a..=a + 2].rotate_left(1);
                    }
                    _ => {
                        let seq = &mut ds.machine_seq[m];
                        seq[a..=a + 1].swap(0, 1);
                    }
                }
                let Some((restored_mk, restored_node)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
                cur_mk = restored_mk;
                mk_node = restored_node;
                break;
            }
        }
    }

    if moved == 0 { return Ok(None); }

    let final_mk = match critical_block_move_local_search_ex_disj(&mut ds, &mut buf, ls_rounds, ls_iters, ls_stall) {
        Some(mk) => mk,
        None => match eval_disj(&ds, &mut buf) {
            Some((mk, _)) => mk,
            None => return Ok(None),
        },
    };
    if final_mk >= start_mk && cur_mk >= start_mk { return Ok(None); }

    let sol = disj_to_solution(pre, &ds, &buf.start)?;
    Ok(Some((sol, final_mk.min(cur_mk))))
}

const SBP_NODE_BUDGET: usize = 600;

struct SbpBuf {
    msucc: Vec<usize>,
    mpred: Vec<usize>,
    jpred: Vec<usize>,
    deg: Vec<u16>,
    stack: Vec<usize>,
    head: Vec<u32>,
    tail: Vec<u32>,
    nodes: Vec<usize>,
    r: Vec<u32>,
    p: Vec<u32>,
    q: Vec<u32>,
    done: Vec<bool>,
    order: Vec<usize>,
    best_order: Vec<usize>,
    saved_seq: Vec<usize>,
}

impl SbpBuf {
    fn new(n: usize) -> Self {
        Self {
            msucc: vec![NONE_USIZE; n], mpred: vec![NONE_USIZE; n], jpred: vec![NONE_USIZE; n],
            deg: vec![0u16; n], stack: Vec::with_capacity(n),
            head: vec![0u32; n], tail: vec![0u32; n],
            nodes: Vec::new(), r: Vec::new(), p: Vec::new(), q: Vec::new(),
            done: Vec::new(), order: Vec::new(), best_order: Vec::new(), saved_seq: Vec::new(),
        }
    }
}

fn sbp_heads_tails_excl(ds: &DisjSchedule, m_excl: usize, b: &mut SbpBuf) -> bool {
    let n = ds.n;
    for u in 0..n { b.msucc[u] = NONE_USIZE; b.mpred[u] = NONE_USIZE; b.jpred[u] = NONE_USIZE; }
    for (m, seq) in ds.machine_seq.iter().enumerate() {
        if m == m_excl || seq.len() < 2 { continue; }
        for w in seq.windows(2) { b.msucc[w[0]] = w[1]; b.mpred[w[1]] = w[0]; }
    }
    for u in 0..n {
        let js = ds.job_succ[u];
        if js != NONE_USIZE { b.jpred[js] = u; }
    }

    for u in 0..n {
        let mut d = ds.indeg_job[u];
        if b.mpred[u] != NONE_USIZE { d = d.saturating_add(1); }
        b.deg[u] = d;
        b.head[u] = 0;
    }
    b.stack.clear();
    for u in 0..n { if b.deg[u] == 0 { b.stack.push(u); } }
    let mut processed = 0usize;
    while let Some(u) = b.stack.pop() {
        processed += 1;
        let end_u = b.head[u].saturating_add(ds.node_pt[u]);
        let js = ds.job_succ[u];
        if js != NONE_USIZE {
            if b.head[js] < end_u { b.head[js] = end_u; }
            b.deg[js] = b.deg[js].saturating_sub(1);
            if b.deg[js] == 0 { b.stack.push(js); }
        }
        let ms = b.msucc[u];
        if ms != NONE_USIZE {
            if b.head[ms] < end_u { b.head[ms] = end_u; }
            b.deg[ms] = b.deg[ms].saturating_sub(1);
            if b.deg[ms] == 0 { b.stack.push(ms); }
        }
    }
    if processed != n { return false; }

    for u in 0..n {
        let mut d = 0u16;
        if ds.job_succ[u] != NONE_USIZE { d += 1; }
        if b.msucc[u] != NONE_USIZE { d += 1; }
        b.deg[u] = d;
        b.tail[u] = 0;
    }
    b.stack.clear();
    for u in 0..n { if b.deg[u] == 0 { b.stack.push(u); } }
    processed = 0;
    while let Some(u) = b.stack.pop() {
        processed += 1;
        let need = b.tail[u].saturating_add(ds.node_pt[u]);
        let jp = b.jpred[u];
        if jp != NONE_USIZE {
            if b.tail[jp] < need { b.tail[jp] = need; }
            b.deg[jp] = b.deg[jp].saturating_sub(1);
            if b.deg[jp] == 0 { b.stack.push(jp); }
        }
        let mp = b.mpred[u];
        if mp != NONE_USIZE {
            if b.tail[mp] < need { b.tail[mp] = need; }
            b.deg[mp] = b.deg[mp].saturating_sub(1);
            if b.deg[mp] == 0 { b.stack.push(mp); }
        }
    }
    processed == n
}

#[inline]
fn sbp_eval_order(order: &[usize], r: &[u32], p: &[u32], q: &[u32]) -> u32 {
    let mut t = 0u32;
    let mut value = 0u32;
    for &i in order {
        let s = if t > r[i] { t } else { r[i] };
        let e = s.saturating_add(p[i]);
        let c = e.saturating_add(q[i]);
        if c > value { value = c; }
        t = e;
    }
    value
}

fn sbp_schrage(r: &[u32], p: &[u32], q: &[u32], done: &mut Vec<bool>, order: &mut Vec<usize>) -> u32 {
    let k = r.len();
    done.clear(); done.resize(k, false);
    order.clear();
    let mut t = u32::MAX;
    for i in 0..k { if r[i] < t { t = r[i]; } }
    let mut value = 0u32;
    for _ in 0..k {
        let mut pick = usize::MAX;
        for i in 0..k {
            if done[i] || r[i] > t { continue; }
            if pick == usize::MAX || q[i] > q[pick] { pick = i; }
        }
        if pick == usize::MAX {
            let mut nt = u32::MAX;
            for i in 0..k { if !done[i] && r[i] < nt { nt = r[i]; } }
            if nt == u32::MAX { break; }
            t = nt;
            for i in 0..k {
                if done[i] || r[i] > t { continue; }
                if pick == usize::MAX || q[i] > q[pick] { pick = i; }
            }
            if pick == usize::MAX { break; }
        }
        done[pick] = true;
        order.push(pick);
        let e = t.saturating_add(p[pick]);
        let c = e.saturating_add(q[pick]);
        if c > value { value = c; }
        t = e;
    }
    value
}

fn sbp_carlier(r0: &[u32], p: &[u32], q0: &[u32], b: &mut SbpBuf) -> Option<u32> {
    let k = r0.len();
    if k < 2 { return None; }
    b.best_order.clear();
    let mut ub = u32::MAX;
    let mut nodes = 0usize;
    let mut starts: Vec<u32> = Vec::with_capacity(k);
    let mut stack: Vec<(Vec<u32>, Vec<u32>)> = Vec::with_capacity(16);
    stack.push((r0.to_vec(), q0.to_vec()));

    while let Some((r, q)) = stack.pop() {
        if nodes >= SBP_NODE_BUDGET { break; }
        nodes += 1;

        let f = sbp_schrage(&r, p, &q, &mut b.done, &mut b.order);
        let true_val = sbp_eval_order(&b.order, r0, p, q0);
        if true_val < ub {
            ub = true_val;
            b.best_order.clear();
            b.best_order.extend_from_slice(&b.order);
        }
        if b.order.len() != k { continue; }

        starts.clear();
        let mut t = 0u32;
        let mut pos_star = usize::MAX;
        for (pos, &i) in b.order.iter().enumerate() {
            let s = if t > r[i] { t } else { r[i] };
            starts.push(s);
            t = s.saturating_add(p[i]);
            if t.saturating_add(q[i]) == f { pos_star = pos; }
        }
        if pos_star == usize::MAX { continue; }

        let mut a = pos_star;
        while a > 0 {
            let prev = b.order[a - 1];
            if starts[a] != starts[a - 1].saturating_add(p[prev]) { break; }
            a -= 1;
        }

        let q_star = q[b.order[pos_star]];
        let mut pos_c = usize::MAX;
        let mut pos = pos_star;
        while pos > a {
            pos -= 1;
            if q[b.order[pos]] < q_star { pos_c = pos; break; }
        }
        if pos_c == usize::MAX { continue; }

        let c = b.order[pos_c];
        let mut r_min = u32::MAX;
        let mut q_min = u32::MAX;
        let mut p_sum = 0u32;
        for &i in &b.order[(pos_c + 1)..=pos_star] {
            if r[i] < r_min { r_min = r[i]; }
            if q[i] < q_min { q_min = q[i]; }
            p_sum = p_sum.saturating_add(p[i]);
        }
        if r_min == u32::MAX || q_min == u32::MAX { continue; }

        let h_j = r_min.saturating_add(p_sum).saturating_add(q_min);
        let r_min2 = if r[c] < r_min { r[c] } else { r_min };
        let q_min2 = if q[c] < q_min { q[c] } else { q_min };
        let h_jc = r_min2.saturating_add(p_sum).saturating_add(p[c]).saturating_add(q_min2);
        let lb = if h_j > h_jc { h_j } else { h_jc };
        if lb >= ub { continue; }
        if stack.len() + 2 > 2 * SBP_NODE_BUDGET { continue; }

        let forced_q = q_min.saturating_add(p_sum);
        if forced_q > q[c] {
            let mut q2 = q.clone();
            q2[c] = forced_q;
            stack.push((r.clone(), q2));
        }
        let forced_r = r_min.saturating_add(p_sum);
        if forced_r > r[c] {
            let mut r2 = r.clone();
            r2[c] = forced_r;
            stack.push((r2, q.clone()));
        }
    }

    if b.best_order.len() == k { Some(ub) } else { None }
}

fn sbp_reoptimize(pre: &Pre, challenge: &Challenge, sol: &Solution, rounds: usize) -> Option<(Solution, u32)> {
    if rounds == 0 { return None; }
    let mut ds = build_disj_from_solution(pre, challenge, sol).ok()?;
    let mut buf = EvalBuf::new(ds.n);
    let (start_mk, _) = eval_disj(&ds, &mut buf)?;
    let mut cur_mk = start_mk;
    let mut b = SbpBuf::new(ds.n);

    let num_machines = challenge.num_machines.min(ds.machine_seq.len());
    let mut m_order: Vec<usize> = (0..num_machines).filter(|&m| ds.machine_seq[m].len() >= 3).collect();
    let mut load: Vec<u64> = vec![0u64; num_machines];
    for m in 0..num_machines {
        for &nd in &ds.machine_seq[m] { load[m] = load[m].saturating_add(ds.node_pt[nd] as u64); }
    }
    m_order.sort_by(|&x, &y| load[y].cmp(&load[x]).then(x.cmp(&y)));
    if m_order.is_empty() { return None; }

    for _ in 0..rounds {
        let mut improved = false;
        for mi in 0..m_order.len() {
            let m = m_order[mi];
            let k = ds.machine_seq[m].len();
            if k < 3 { continue; }
            if !sbp_heads_tails_excl(&ds, m, &mut b) { continue; }

            b.nodes.clear(); b.nodes.extend_from_slice(&ds.machine_seq[m]);
            b.r.clear(); b.p.clear(); b.q.clear();
            for &nd in &b.nodes {
                b.r.push(b.head[nd]);
                b.p.push(ds.node_pt[nd]);
                b.q.push(b.tail[nd]);
            }

            let mut cur_val = 0u32;
            {
                let mut t = 0u32;
                for i in 0..k {
                    let s = if t > b.r[i] { t } else { b.r[i] };
                    let e = s.saturating_add(b.p[i]);
                    let c = e.saturating_add(b.q[i]);
                    if c > cur_val { cur_val = c; }
                    t = e;
                }
            }

            let (r, p, q) = (b.r.clone(), b.p.clone(), b.q.clone());
            let Some(val) = sbp_carlier(&r, &p, &q, &mut b) else { continue };
            if val >= cur_val { continue; }

            b.saved_seq.clear(); b.saved_seq.extend_from_slice(&ds.machine_seq[m]);
            for pos in 0..k { ds.machine_seq[m][pos] = b.nodes[b.best_order[pos]]; }
            match eval_disj(&ds, &mut buf) {
                Some((new_mk, _)) if new_mk < cur_mk => { cur_mk = new_mk; improved = true; }
                _ => {
                    ds.machine_seq[m].clear();
                    ds.machine_seq[m].extend_from_slice(&b.saved_seq);
                }
            }
        }
        if !improved { break; }
    }

    if cur_mk >= start_mk { return None; }
    let (final_mk, _) = eval_disj(&ds, &mut buf)?;
    if final_mk >= start_mk { return None; }
    let out = disj_to_solution(pre, &ds, &buf.start).ok()?;
    Some((out, final_mk))
}

#[inline]
fn sbp_partial_acyclic(ds: &DisjSchedule, b: &mut SbpBuf) -> bool {
    sbp_heads_tails_excl(ds, usize::MAX, b)
}

fn sbp_construct_seed(
    pre: &Pre,
    challenge: &Challenge,
    template: &Solution,
    reopt: bool,
    reopt_cycles: usize,
) -> Option<(Solution, u32)> {
    let mut ds = build_disj_from_solution(pre, challenge, template).ok()?;
    let n = ds.n;
    let num_machines = challenge.num_machines.min(ds.machine_seq.len());
    if num_machines == 0 { return None; }

    let mut members: Vec<Vec<usize>> = vec![Vec::new(); num_machines];
    for u in 0..n {
        let m = ds.node_machine[u];
        if m < num_machines { members[m].push(u); }
    }

    for m in 0..num_machines {
        ds.machine_seq[m].clear();
        if members[m].len() < 2 { ds.machine_seq[m].extend_from_slice(&members[m]); }
    }

    let mut b = SbpBuf::new(n);
    let mut buf = EvalBuf::new(n);
    let mut sequenced: Vec<bool> = (0..num_machines).map(|m| members[m].len() < 2).collect();
    let mut fixed_order: Vec<usize> = Vec::with_capacity(num_machines);
    let mut pending = sequenced.iter().filter(|&&s| !s).count();

    while pending > 0 {
        let mut best_m = NONE_USIZE;
        let mut best_val = 0u32;
        let mut best_seq: Vec<usize> = Vec::new();
        for m in 0..num_machines {
            if sequenced[m] { continue; }
            if !sbp_heads_tails_excl(&ds, m, &mut b) { continue; }
            b.nodes.clear(); b.nodes.extend_from_slice(&members[m]);
            b.r.clear(); b.p.clear(); b.q.clear();
            for &nd in &b.nodes {
                b.r.push(b.head[nd]);
                b.p.push(ds.node_pt[nd]);
                b.q.push(b.tail[nd]);
            }
            let (r, p, q) = (b.r.clone(), b.p.clone(), b.q.clone());
            let Some(val) = sbp_carlier(&r, &p, &q, &mut b) else { continue };
            if b.best_order.len() != b.nodes.len() { continue; }
            if best_m == NONE_USIZE || val > best_val {
                best_val = val;
                best_m = m;
                best_seq.clear();
                for &pos in b.best_order.iter() { best_seq.push(b.nodes[pos]); }
            }
        }
        if best_m == NONE_USIZE { return None; }

        ds.machine_seq[best_m].clear();
        ds.machine_seq[best_m].extend_from_slice(&best_seq);
        if !sbp_partial_acyclic(&ds, &mut b) {
            ds.machine_seq[best_m].clear();
            if !sbp_heads_tails_excl(&ds, best_m, &mut b) { return None; }
            let mut order: Vec<usize> = members[best_m].clone();
            order.sort_by(|&x, &y| b.head[x].cmp(&b.head[y]).then(x.cmp(&y)));
            ds.machine_seq[best_m].clear();
            ds.machine_seq[best_m].extend_from_slice(&order);
            if !sbp_partial_acyclic(&ds, &mut b) { return None; }
        }
        sequenced[best_m] = true;
        fixed_order.push(best_m);
        pending -= 1;

        if reopt {
            let cap = if reopt_cycles == 0 { num_machines } else { reopt_cycles.min(num_machines) };
            for _cycle in 0..cap {
            let mut changed = false;
            for i in 0..fixed_order.len() {
                let m2 = fixed_order[i];
                let k2 = ds.machine_seq[m2].len();
                if k2 < 3 { continue; }
                if !sbp_heads_tails_excl(&ds, m2, &mut b) { continue; }
                b.nodes.clear(); b.nodes.extend_from_slice(&members[m2]);
                b.r.clear(); b.p.clear(); b.q.clear();
                for &nd in &b.nodes {
                    b.r.push(b.head[nd]);
                    b.p.push(ds.node_pt[nd]);
                    b.q.push(b.tail[nd]);
                }
                let cur_val = {
                    let seq: Vec<usize> = ds.machine_seq[m2].clone();
                    let mut t = 0u32;
                    let mut v = 0u32;
                    for &nd in seq.iter() {
                        let pos = match b.nodes.iter().position(|&x| x == nd) { Some(p) => p, None => return None };
                        let s = if t > b.r[pos] { t } else { b.r[pos] };
                        let e = s.saturating_add(b.p[pos]);
                        let c = e.saturating_add(b.q[pos]);
                        if c > v { v = c; }
                        t = e;
                    }
                    v
                };
                let (r, p, q) = (b.r.clone(), b.p.clone(), b.q.clone());
                let Some(val) = sbp_carlier(&r, &p, &q, &mut b) else { continue };
                if val >= cur_val { continue; }
                if b.best_order.len() != b.nodes.len() { continue; }
                b.saved_seq.clear(); b.saved_seq.extend_from_slice(&ds.machine_seq[m2]);
                ds.machine_seq[m2].clear();
                for &pos in b.best_order.iter() { ds.machine_seq[m2].push(b.nodes[pos]); }
                if !sbp_partial_acyclic(&ds, &mut b) {
                    ds.machine_seq[m2].clear();
                    ds.machine_seq[m2].extend_from_slice(&b.saved_seq);
                } else {
                    changed = true;
                }
            }
            if !changed { break; }
            }
        }
    }

    let (mk, _) = eval_disj(&ds, &mut buf)?;
    let sol = disj_to_solution(pre, &ds, &buf.start).ok()?;
    Some((sol, mk))
}

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    pre: &Pre,
    effort: &EffortConfig,
) -> Result<()> {
    let (greedy_sol, greedy_mk) = run_simple_greedy_baseline(challenge)?;
    save_solution(&greedy_sol)?;

    let mut rng = SmallRng::from_seed(challenge.seed);
    let allow_flex_balance = pre.high_flex > 0.60 && pre.jobshopness > 0.38;
    let mut rules: Vec<Rule> = vec![Rule::Adaptive, Rule::BnHeavy, Rule::EndTight, Rule::CriticalPath, Rule::MostWork, Rule::LeastFlex, Rule::Regret, Rule::ShortestProc];
    if allow_flex_balance { rules.push(Rule::FlexBalance); }

    let mut best_makespan = greedy_mk; let mut best_solution: Option<Solution> = Some(greedy_sol.clone()); let mut top_solutions: Vec<(Solution, u32)> = Vec::new();
    push_top_solutions(&mut top_solutions, &greedy_sol, greedy_mk, 15);
    let target_margin: u32 = ((pre.avg_op_min * (0.9 + 0.9*pre.high_flex + 0.6*pre.jobshopness)).max(1.0)) as u32;
    let route_w_base: f64 = if pre.chaotic_like { 0.0 } else { (0.040 + 0.10*pre.high_flex + 0.08*pre.jobshopness).clamp(0.04, 0.22) };

    if pre.flow_route.is_some() && pre.flow_pt_by_job.is_some() {
        if let Ok((sol, mk)) = neh_reentrant_flow_solution(pre, challenge.num_jobs, challenge.num_machines) {
            if mk < best_makespan { best_makespan=mk; best_solution=Some(sol.clone()); save_solution(&sol)?; }
            push_top_solutions(&mut top_solutions, &sol, mk, 15);
        }
    }

    let mut ranked: Vec<(Rule,u32,Solution)> = Vec::with_capacity(rules.len());
    for &rule in &rules {
        let (sol, mk) = construct_solution_conflict(challenge, pre, rule, 0, None, &mut rng, None, None, None, 0.0)?;
        if mk < best_makespan { best_makespan=mk; best_solution=Some(sol.clone()); save_solution(&sol)?; }
        push_top_solutions(&mut top_solutions, &sol, mk, 15); ranked.push((rule, mk, sol));
    }
    ranked.sort_by_key(|x| x.1);
    let r0=ranked[0].0; let r1=ranked.get(1).map(|x|x.0).unwrap_or(r0); let r2=ranked.get(2).map(|x|x.0).unwrap_or(r1);

    let mut rule_best: Vec<u32> = vec![u32::MAX; 9]; let mut rule_tries: Vec<u32> = vec![0u32; 9];
    for (rr,mk,_) in &ranked { let idx=rule_idx(*rr); rule_best[idx]=rule_best[idx].min(*mk); rule_tries[idx]=rule_tries[idx].saturating_add(1); }

    let base = &ranked[0].2;
    let mut learned_jb = Some(job_bias_from_solution(pre, base)?);
    let mut learned_mp = Some(machine_penalty_from_solution(pre, base, challenge.num_machines)?);
    let mut learned_rp = if route_w_base > 0.0 { Some(route_pref_from_solution_lite(pre, base, challenge)?) } else { None };
    let mut learn_updates_left = 4usize;
    let num_restarts = 450usize;

    let mut k_hi = if pre.flex_avg > 8.0 { 6 } else if pre.flex_avg > 6.5 { 4 } else if pre.flex_avg > 4.0 { 5 } else { 6 };
    if pre.jobshopness > 0.60 && k_hi < 6 { k_hi += 1; }
    k_hi = k_hi.min(6).max(2);
    let mut stuck: usize = 0;

    for r in 0..num_restarts {
        let late = r >= (num_restarts*2)/3;
        let (k_min,k_max) = if stuck>170 { (4usize,6usize.min(k_hi)) } else if stuck>90 { (3usize,6usize.min(k_hi.max(4))) } else if stuck>35 { (2usize,k_hi) } else { (2usize,k_hi.min(4)) };
        let rule = if r < 35 { let u: f64=rng.gen(); if allow_flex_balance&&pre.high_flex>0.82&&u<0.10{Rule::FlexBalance}else if u<0.52{r0}else if u<0.80{r1}else if u<0.92{r2}else{rules[rng.gen_range(0..rules.len())]} }
            else { choose_rule_bandit(&mut rng, &rules, &rule_best, &rule_tries, best_makespan, target_margin, stuck, pre.chaotic_like, late) };
        let k = if k_max<=k_min { k_min } else { rng.gen_range(k_min..=k_max) };
        let learn_base = if pre.chaotic_like { 0.0 } else { (0.08+0.22*pre.jobshopness+0.18*pre.high_flex).clamp(0.05,0.42) };
        let learn_boost = (1.0+0.35*((stuck as f64)/120.0).clamp(0.0,1.0)).clamp(1.0,1.35);
        let learn_p = (learn_base*learn_boost).clamp(0.0,0.60);
        let use_learn = learned_jb.is_some() && learned_mp.is_some() && rng.gen::<f64>()<learn_p && (route_w_base==0.0||learned_rp.is_some());
        let target = if best_makespan < (u32::MAX/2) { Some(best_makespan.saturating_add(target_margin)) } else { None };
        let (sol, mk) = if use_learn {
            construct_solution_conflict(challenge, pre, rule, k, target, &mut rng, learned_jb.as_deref(), learned_mp.as_deref(), learned_rp.as_ref(), route_w_base)?
        } else {
            construct_solution_conflict(challenge, pre, rule, k, target, &mut rng, None, None, None, 0.0)?
        };
        let ridx=rule_idx(rule); rule_tries[ridx]=rule_tries[ridx].saturating_add(1); rule_best[ridx]=rule_best[ridx].min(mk);
        if mk < best_makespan {
            best_makespan=mk; best_solution=Some(sol.clone()); save_solution(&sol)?; stuck=0;
            if learn_updates_left > 0 && !pre.chaotic_like {
                learned_jb=Some(job_bias_from_solution(pre,&sol)?); learned_mp=Some(machine_penalty_from_solution(pre,&sol,challenge.num_machines)?);
                if route_w_base>0.0 { learned_rp=Some(route_pref_from_solution_lite(pre,&sol,challenge)?); }
                learn_updates_left-=1;
            }
        } else { stuck=stuck.saturating_add(1); }
        push_top_solutions(&mut top_solutions, &sol, mk, 15);
    }

    let route_w_ls: f64 = if route_w_base>0.0 { (route_w_base*1.40).clamp(route_w_base,0.40) } else { 0.0 };
    let mut refine_results: Vec<(Solution,u32)> = Vec::new();
    for (base_sol, _) in top_solutions.iter() {
        let jb = job_bias_from_solution(pre, base_sol)?;
        let mp = machine_penalty_from_solution(pre, base_sol, challenge.num_machines)?;
        let anti_jb = invert_job_bias_guidance(&jb);
        let anti_mp = invert_machine_penalty_guidance(&mp);
        let rp = if route_w_ls>0.0 { Some(route_pref_from_solution_lite(pre, base_sol, challenge)?) } else { None };
        let target_ls = if best_makespan < (u32::MAX/2) { Some(best_makespan.saturating_add(target_margin/2)) } else { None };
        for attempt in 0..10 {
            let use_anti = attempt % 2 == 1;
            let rule = if pre.chaotic_like {
                match attempt { 0=>Rule::Regret, 1=>Rule::MostWork, 2=>Rule::ShortestProc, 3=>Rule::Adaptive, 4=>Rule::ShortestProc, 5=>Rule::Regret, 6=>Rule::MostWork, 7=>Rule::Adaptive, 8=>Rule::Adaptive, _=>Rule::ShortestProc }
            } else {
                match attempt { 0=>r0, 1=>Rule::Adaptive, 2=>Rule::BnHeavy, 3=>Rule::EndTight, 4=>Rule::Regret, 5=>Rule::CriticalPath, 6=>Rule::LeastFlex, 7=>Rule::MostWork, 8=>if allow_flex_balance{Rule::FlexBalance}else{r1}, _=>r1 }
            };
            let k = match attempt%4 { 0=>2, 1=>3, 2=>4, _=>2 }.min(k_hi);
            let jb_ref: Option<&[f64]> = if use_anti { Some(anti_jb.as_slice()) } else { Some(jb.as_slice()) };
            let mp_ref: Option<&[f64]> = if use_anti { Some(anti_mp.as_slice()) } else { Some(mp.as_slice()) };
            let rp_ref = if use_anti { None } else { rp.as_ref() };
            let route_w = if use_anti { 0.0 } else if rp.is_some() { route_w_ls } else { 0.0 };
            let (sol, mk) = construct_solution_conflict(challenge, pre, rule, k, target_ls, &mut rng, jb_ref, mp_ref, rp_ref, route_w)?;
            if mk < best_makespan { best_makespan=mk; best_solution=Some(sol.clone()); save_solution(&sol)?; }
            refine_results.push((sol, mk));
        }
    }
    for (sol, mk) in refine_results { push_top_solutions(&mut top_solutions, &sol, mk, 15); }

    let ts_starts = top_solutions.len().min(effort.js_ts_starts.max(1));
    let ts_iters = effort.job_shop_iters;
    let fast_eval = effort.js_ts_fast_eval == 1;
    
    let div_period = effort.js_ts_div_period;
    
    let tie_mode = effort.js_ts_tie_mode;
    
    let sel_key = effort.js_ts_sel_key;
    
    let exact_topk = effort.js_ts_exact_topk;
    
    let traj_decorr = effort.js_ts_traj_decorr;
    
    let extra_draws = effort.js_ts_extra_draws;
    
    let kick_spread = effort.js_ts_kick_spread;
    
    let sa_accept = effort.js_ts_sa_accept;
    
    let mem_cross_scope = effort.js_mem_cross_scope;
    
    let mem_admit = effort.js_mem_admit;
    
    let sbp_rounds = effort.js_sbp_rounds;
    
    let polish_rescue = effort.js_ts_polish_rescue;
    
    let sbp_construct = effort.js_seed_sbp_construct;
    
    let sbp_reopt_cycles = effort.js_seed_sbp_reopt_cycles;
    
    let sbp_dual_cycles = effort.js_seed_sbp_dual_cycles;
    let mut sbp_seed_best: Option<(Solution, u32)> = None;
    let mut ts_extra_best: Option<(Solution, u32)> = None;
    let ts_tenure =((pre.total_ops as f64).sqrt() as usize * (100 + (pre.load_cv * 60.0) as usize) / 100).clamp(8, 24);

    let ts_seed_idx: Vec<usize> = if ts_starts == 0 || effort.js_seed_select_mode == 0 {
        (0..ts_starts).collect()
    } else if effort.js_seed_select_mode == 2 {
        
        let mut picked: Vec<usize> = (0..ts_starts).collect();
        if top_solutions.len() > ts_starts {
            picked[ts_starts - 1] = ts_starts;
        }
        picked
    } else {
        let pool_len = top_solutions.len();
        let keys: Vec<Vec<u32>> = top_solutions.iter()
            .map(|(s, _)| ts_seed_key(pre, challenge.num_machines, s))
            .collect();
        let mut picked: Vec<usize> = Vec::with_capacity(ts_starts);
        let mut taken = vec![false; pool_len];
        let mut min_d = vec![u32::MAX; pool_len];
        picked.push(0);
        taken[0] = true;
        while picked.len() < ts_starts {
            let last = *picked.last().unwrap();
            let mut best: Option<usize> = None;
            let mut best_d = 0u32;
            for i in 0..pool_len {
                if taken[i] { continue; }
                let d = ts_seed_key_distance(&keys[last], &keys[i]);
                if d < min_d[i] { min_d[i] = d; }
                if best.is_none() || min_d[i] > best_d {
                    best = Some(i);
                    best_d = min_d[i];
                }
            }
            let Some(b) = best else { break };
            picked.push(b);
            taken[b] = true;
        }
        picked
    };
    {
        let mut ts_results: Vec<(Solution, u32)> = Vec::new();
        for (rank, &idx) in ts_seed_idx.iter().enumerate() {

            let traj_salt: u64 = match traj_decorr {
                0 => 0,
                1 => (rank as u64).wrapping_mul(0x9E3779B97F4A7C15),
                _ => (rank as u64).wrapping_mul(0xD1B54A32D192ED03),
            };
            let res = {
                let base_sol = &top_solutions[idx].0;
                tabu_search_phase(pre, challenge, base_sol, ts_iters, ts_tenure, fast_eval, div_period, tie_mode, sel_key, exact_topk, traj_salt, kick_spread, 0, 2)?
            };
            if let Some((sol2, mk2)) = res {
                if mk2 < best_makespan { best_makespan=mk2; best_solution=Some(sol2.clone()); save_solution(&sol2)?; }
                ts_results.push((sol2, mk2));
            }
        }

        for e in 0..extra_draws {
            if ts_seed_idx.is_empty() { break; }
            let idx = ts_seed_idx[e % ts_seed_idx.len()];
            
            let extra_salt: u64 = ((e as u64) + 1)
                .wrapping_mul(0x9E3779B97F4A7C15)
                ^ 0x517CC1B727220A95;
            let res = {
                let base_sol = &top_solutions[idx].0;
                tabu_search_phase(pre, challenge, base_sol, ts_iters, ts_tenure, fast_eval, div_period, tie_mode, sel_key, exact_topk, extra_salt, kick_spread, sa_accept, 2)?
            };
            if let Some((sol2, mk2)) = res {
                let better = match &ts_extra_best { None => true, Some((_, bmk)) => mk2 < *bmk };
                if better { ts_extra_best = Some((sol2, mk2)); }
            }
        }
        for (sol2, mk2) in ts_results {
            push_top_solutions(&mut top_solutions, &sol2, mk2, 20);
        }
    }

    {
        let bn_starts = top_solutions.len().min(if pre.total_ops > 240 { 6 } else { 8 });
        let mut shared_bn_buf: Option<(usize, EvalBuf)> = None;
        let mut bn_results: Vec<(Solution, u32)> = Vec::new();
        let mut crit_pos: Vec<usize> = Vec::with_capacity(16);
        let mut move_pos: Vec<usize> = Vec::with_capacity(18);
        let mut machine_total_pt: Vec<u64> = vec![0u64; challenge.num_machines];
        let mut m_rank: Vec<usize> = Vec::with_capacity(challenge.num_machines);
        let relocate_span = if pre.total_ops > 260 { 2usize } else if pre.jobshopness > 0.60 { 4usize } else { 3usize };
        let max_rounds = if pre.total_ops > 260 { 2usize } else { 3usize };
        for idx in 0..bn_starts {
            let mut ds = {
                let base_sol = &top_solutions[idx].0;
                match build_disj_from_solution(pre, challenge, base_sol) { Ok(d) => d, Err(_) => continue }
            };
            if shared_bn_buf.as_ref().map_or(true, |(n, _)| *n != ds.n) {
                shared_bn_buf = Some((ds.n, EvalBuf::new(ds.n)));
            }
            let (_, buf) = shared_bn_buf.as_mut().unwrap();
            let Some((mut cur_mk, mut mk_node)) = eval_disj(&ds, buf) else { continue };

            machine_total_pt.fill(0);
            for m in 0..challenge.num_machines {
                if m < ds.machine_seq.len() {
                    for &nd in &ds.machine_seq[m] {
                        if nd < ds.node_pt.len() {
                            machine_total_pt[m] = machine_total_pt[m].saturating_add(ds.node_pt[nd] as u64);
                        }
                    }
                }
            }
            m_rank.clear();
            for m in 0..challenge.num_machines {
                if m < ds.machine_seq.len() && ds.machine_seq[m].len() > 1 {
                    m_rank.push(m);
                }
            }
            m_rank.sort_by(|&a, &b| machine_total_pt[b].cmp(&machine_total_pt[a]));

            let num_bn_ls = m_rank.len().min(3);
            let mut any_improved = false;

            for _ in 0..max_rounds {
                let mut round_improved = false;
                for bi in 0..num_bn_ls {
                    let m = m_rank[bi];
                    let seq_cap = ds.machine_seq[m].len().min(18);
                    if seq_cap < 2 { continue; }

                    crit_pos.clear();
                    {
                        let prefix = &ds.machine_seq[m][..seq_cap];
                        let mut u = mk_node;
                        while u != NONE_USIZE {
                            if let Some(pos) = prefix.iter().position(|&nd| nd == u) {
                                crit_pos.push(pos);
                            }
                            u = buf.best_pred[u];
                        }
                    }
                    if crit_pos.is_empty() { continue; }
                    crit_pos.sort_unstable();
                    crit_pos.dedup();

                    move_pos.clear();
                    for &pos in &crit_pos {
                        move_pos.push(pos);
                        if pos > 0 { move_pos.push(pos - 1); }
                        if pos + 1 < seq_cap { move_pos.push(pos + 1); }
                    }
                    move_pos.sort_unstable();
                    move_pos.dedup();
                    if move_pos.is_empty() { continue; }

                    let mut best_move: Option<(usize, usize)> = None;
                    let mut best_move_mk = cur_mk;

                    for &from in &move_pos {
                        let lo = from.saturating_sub(relocate_span);
                        let hi = (from + relocate_span).min(seq_cap - 1);
                        for to in lo..=hi {
                            if to == from { continue; }
                            {
                                let seq = &mut ds.machine_seq[m][..seq_cap];
                                relocate_machine_seq(seq, from, to);
                            }
                            if let Some(new_mk) = eval_disj_trial(&ds, buf, best_move_mk) {
                                if new_mk < best_move_mk {
                                    best_move_mk = new_mk;
                                    best_move = Some((from, to));
                                }
                            }
                            {
                                let seq = &mut ds.machine_seq[m][..seq_cap];
                                relocate_machine_seq(seq, to, from);
                            }
                        }
                    }

                    let Some((from, to)) = best_move else {
                        let Some((restored_mk, restored_node)) = eval_disj(&ds, buf) else { continue };
                        cur_mk = restored_mk;
                        mk_node = restored_node;
                        continue;
                    };

                    {
                        let seq = &mut ds.machine_seq[m][..seq_cap];
                        relocate_machine_seq(seq, from, to);
                    }
                    match eval_disj(&ds, buf) {
                        Some((new_mk, new_node)) if new_mk < cur_mk => {
                            cur_mk = new_mk;
                            mk_node = new_node;
                            any_improved = true;
                            round_improved = true;
                        }
                        _ => {
                            {
                                let seq = &mut ds.machine_seq[m][..seq_cap];
                                relocate_machine_seq(seq, to, from);
                            }
                            let Some((restored_mk, restored_node)) = eval_disj(&ds, buf) else { continue };
                            cur_mk = restored_mk;
                            mk_node = restored_node;
                        }
                    }
                }
                if !round_improved { break; }
            }

            if any_improved {
                if let Ok(sol_bn) = disj_to_solution(pre, &ds, &buf.start) {
                    if cur_mk < best_makespan { best_makespan=cur_mk; best_solution=Some(sol_bn.clone()); save_solution(&sol_bn)?; }
                    bn_results.push((sol_bn, cur_mk));
                }
            }
        }
        for (sol_bn, mk_bn) in bn_results {
            push_top_solutions(&mut top_solutions, &sol_bn, mk_bn, 20);
        }
    }

    {
        let ils_starts = top_solutions.len().min(6);
        let mut ils_results: Vec<(Solution, u32)> = Vec::new();
        for idx in 0..ils_starts {
            let ls_res = {
                let base_sol = &top_solutions[idx].0;
                critical_block_move_local_search_ex(pre, challenge, base_sol, 5, 400, 120)
            };
            if let Ok(Some((ls_sol, ls_mk))) = ls_res {
                if ls_mk < best_makespan { best_makespan=ls_mk; best_solution=Some(ls_sol.clone()); save_solution(&ls_sol)?; }
                ils_results.push((ls_sol.clone(), ls_mk));
                let reopt_res = perturb_and_reoptimize_ils(pre, challenge, &ls_sol, 2, 3, 220, 80)?;
                if let Some((sol3, mk3)) = reopt_res {
                    if mk3 < best_makespan { best_makespan=mk3; best_solution=Some(sol3.clone()); save_solution(&sol3)?; }
                    ils_results.push((sol3, mk3));
                }
            }
        }
        for (sol, mk) in ils_results {
            push_top_solutions(&mut top_solutions, &sol, mk, 20);
        }
    }

    {
        let num_machines = challenge.num_machines;

        let mut machine_rank: Vec<(usize, f64)> = (0..num_machines).map(|m| {
            let scar = if m < pre.machine_scarcity.len() { pre.machine_scarcity[m] } else { 1.0 };
            (m, scar)
        }).collect();
        machine_rank.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let num_bn = (num_machines / 5).max(3).min(8);
        let mut is_bottleneck = vec![false; num_machines];
        for i in 0..num_bn { is_bottleneck[machine_rank[i].0] = true; }

        let pop_cap = 10usize;
        let mut mem_pop: Vec<(Solution, u32)> = top_solutions.iter().take(pop_cap).cloned().collect();
        let mut mem_ds: Vec<Option<DisjSchedule>> = vec![None; mem_pop.len()];

        let num_generations = 18usize;
        let mut gen_no_improve = 0usize;
        let max_gen_no_improve = 12usize;
        let mut shared_mem_buf: Option<(usize, EvalBuf)> = None;
        let mut cross_pos: Vec<usize> = Vec::new();
        let mut cross_sum: Vec<u32> = Vec::new();
        let mut cross_cnt: Vec<u16> = Vec::new();
        let mut cross_pairs: Vec<(usize, usize, i32)> = Vec::new();
        let elite_cap = 4usize;

        let mut admit_buf: Option<(usize, EvalBuf)> = None;
        let mut admit_saved: Vec<usize> = Vec::new();
        for gen in 0..num_generations {
            if gen_no_improve >= max_gen_no_improve { break; }
            let cur_pop = mem_pop.len();
            if cur_pop < 2 { break; }

            let use_mutation = gen % 6 == 5;

            let ia = {
                let a = rng.gen_range(0..cur_pop);
                let b = rng.gen_range(0..cur_pop);
                if mem_pop[a].1 <= mem_pop[b].1 { a } else { b }
            };
            let ib = {
                let mut b = rng.gen_range(0..cur_pop);
                if b == ia { b = (b + 1) % cur_pop; }
                let c = rng.gen_range(0..cur_pop);
                let c = if c == ia { (c + 1) % cur_pop } else { c };
                if mem_pop[b].1 <= mem_pop[c].1 { b } else { c }
            };

            let mk_a = mem_pop[ia].1;
            let mk_b = mem_pop[ib].1;

            if mem_ds[ia].is_none() {
                let built = match build_disj_from_solution(pre, challenge, &mem_pop[ia].0) {
                    Ok(d) => d,
                    Err(_) => { gen_no_improve += 1; continue; }
                };
                mem_ds[ia] = Some(built);
            }
            if mem_ds[ib].is_none() {
                let built = match build_disj_from_solution(pre, challenge, &mem_pop[ib].0) {
                    Ok(d) => d,
                    Err(_) => { gen_no_improve += 1; continue; }
                };
                mem_ds[ib] = Some(built);
            }

            let mut elite_idx: Vec<usize> = (0..cur_pop).collect();
            elite_idx.sort_unstable_by_key(|&i| mem_pop[i].1);
            elite_idx.truncate(elite_cap.min(cur_pop));
            let mut elite_failed = false;
            for &ei in &elite_idx {
                if mem_ds[ei].is_none() {
                    match build_disj_from_solution(pre, challenge, &mem_pop[ei].0) {
                        Ok(d) => mem_ds[ei] = Some(d),
                        Err(_) => {
                            elite_failed = true;
                            break;
                        }
                    }
                }
            }
            if elite_failed {
                gen_no_improve += 1;
                continue;
            }

            let (better_idx, worse_idx) = if mk_a <= mk_b { (ia, ib) } else { (ib, ia) };
            let mut child_ds = {
                let better_ds = mem_ds[better_idx].as_ref().unwrap();
                let worse_ds = mem_ds[worse_idx].as_ref().unwrap();
                let mut child_ds = better_ds.clone();

                if cross_pos.len() != child_ds.n {
                    cross_pos.resize(child_ds.n, 0);
                    cross_sum.resize(child_ds.n, 0);
                    cross_cnt.resize(child_ds.n, 0);
                }

                for m in 0..num_machines {
                    
                    let skip_machine = match mem_cross_scope {
                        1 => false,
                        2 => !is_bottleneck[m],
                        _ => is_bottleneck[m],
                    };
                    if skip_machine { continue; }
                    if m >= better_ds.machine_seq.len() || m >= worse_ds.machine_seq.len() || m >= child_ds.machine_seq.len() { continue; }
                    if better_ds.machine_seq[m].len() != worse_ds.machine_seq[m].len() { continue; }

                    let mut elite_machine_seqs: Vec<&[usize]> = Vec::with_capacity(elite_idx.len());
                    for &ei in &elite_idx {
                        if let Some(ds_e) = mem_ds[ei].as_ref() {
                            if m < ds_e.machine_seq.len() && ds_e.machine_seq[m].len() == child_ds.machine_seq[m].len() {
                                elite_machine_seqs.push(&ds_e.machine_seq[m]);
                            }
                        }
                    }

                    if mem_admit == 0 {
                        inherit_machine_consensus_order(
                            &mut child_ds.machine_seq[m],
                            &better_ds.machine_seq[m],
                            &worse_ds.machine_seq[m],
                            &elite_machine_seqs,
                            &mut cross_pos,
                            &mut cross_sum,
                            &mut cross_cnt,
                            &mut cross_pairs,
                        );
                        continue;
                    }

                    admit_saved.clear();
                    admit_saved.extend_from_slice(&child_ds.machine_seq[m]);
                    inherit_machine_consensus_order(
                        &mut child_ds.machine_seq[m],
                        &better_ds.machine_seq[m],
                        &worse_ds.machine_seq[m],
                        &elite_machine_seqs,
                        &mut cross_pos,
                        &mut cross_sum,
                        &mut cross_cnt,
                        &mut cross_pairs,
                    );
                    if child_ds.machine_seq[m] == admit_saved { continue; }
                    if admit_buf.as_ref().map_or(true, |(n, _)| *n != child_ds.n) {
                        admit_buf = Some((child_ds.n, EvalBuf::new(child_ds.n)));
                    }
                    let (_, abuf) = admit_buf.as_mut().unwrap();
                    if eval_disj(&child_ds, abuf).is_none() {
                        child_ds.machine_seq[m].clear();
                        child_ds.machine_seq[m].extend_from_slice(&admit_saved);
                    }
                }

                child_ds
            };

            if use_mutation {
                let non_bn_machines: Vec<usize> = (0..num_machines).filter(|&m| !is_bottleneck[m] && child_ds.machine_seq[m].len() > 1).collect();
                if !non_bn_machines.is_empty() {
                    for _ in 0..2 {
                        let m = non_bn_machines[rng.gen_range(0..non_bn_machines.len())];
                        let seq_len = child_ds.machine_seq[m].len();
                        if seq_len > 1 {
                            let pos = rng.gen_range(0..seq_len - 1);
                            child_ds.machine_seq[m].swap(pos, pos + 1);
                        }
                    }
                }
                let bn_machines: Vec<usize> = (0..num_machines).filter(|&m| is_bottleneck[m] && child_ds.machine_seq[m].len() > 1).collect();
                if !bn_machines.is_empty() {
                    let m = bn_machines[rng.gen_range(0..bn_machines.len())];
                    let seq_len = child_ds.machine_seq[m].len();
                    if seq_len > 1 {
                        let pos = rng.gen_range(0..seq_len - 1);
                        child_ds.machine_seq[m].swap(pos, pos + 1);
                    }
                }
            }

            if shared_mem_buf.as_ref().map_or(true, |(n, _)| *n != child_ds.n) {
                shared_mem_buf = Some((child_ds.n, EvalBuf::new(child_ds.n)));
            }
            let (_, child_buf) = shared_mem_buf.as_mut().unwrap();
            let ls_mk = match critical_block_move_local_search_ex_disj(&mut child_ds, child_buf, 4, 250, 100) {
                Some(mk) => mk,
                None => match eval_disj(&child_ds, child_buf) {
                    Some((mk, _)) => mk,
                    None => { gen_no_improve += 1; continue; }
                },
            };
            let mut ls_sol = match disj_to_solution(pre, &child_ds, &child_buf.start) {
                Ok(s) => s,
                Err(_) => { gen_no_improve += 1; continue; }
            };
            
            let mut ls_mk = ls_mk;
            if mem_admit >= 2 {
                let child_ts_iters = if mem_admit == 3 { ts_iters / 8 } else { ts_iters / 24 };
                if child_ts_iters > 0 {
                    if let Some((ts_sol, ts_mk)) = tabu_search_phase(pre, challenge, &ls_sol, child_ts_iters, ts_tenure, fast_eval, div_period, tie_mode, sel_key, exact_topk, 0, kick_spread, 0, 2)? {
                        if ts_mk < ls_mk {
                            ls_sol = ts_sol;
                            ls_mk = ts_mk;
                        }
                    }
                }
            }

            if ls_mk < best_makespan {
                best_makespan = ls_mk;
                best_solution = Some(ls_sol.clone());
                save_solution(&ls_sol)?;
                gen_no_improve = 0;
            } else {
                gen_no_improve += 1;
            }

            push_top_solutions(&mut top_solutions, &ls_sol, ls_mk, 20);

            if cur_pop >= pop_cap {
                let worst_idx = mem_pop.iter().enumerate().max_by_key(|(_, (_, mk))| *mk).map(|(i, _)| i).unwrap_or(cur_pop - 1);
                if ls_mk < mem_pop[worst_idx].1 {
                    mem_pop[worst_idx] = (ls_sol, ls_mk);
                    mem_ds[worst_idx] = Some(child_ds.clone());
                }
            } else {
                mem_pop.push((ls_sol, ls_mk));
                mem_ds.push(Some(child_ds.clone()));
            }
        }

        let mem_best: Vec<Solution> = {
            let mut sorted = mem_pop.clone();
            sorted.sort_by_key(|(_, mk)| *mk);
            sorted.into_iter().take(3).map(|(s, _)| s).collect()
        };
        for base_sol in &mem_best {
            
            if let Some((ts_sol, ts_mk)) = tabu_search_phase(pre, challenge, base_sol, ts_iters / 3, ts_tenure, fast_eval, div_period, tie_mode, sel_key, exact_topk, 0, kick_spread, 0, 2)? {
                if ts_mk < best_makespan {
                    best_makespan = ts_mk;
                    best_solution = Some(ts_sol.clone());
                    save_solution(&ts_sol)?;
                }
                push_top_solutions(&mut top_solutions, &ts_sol, ts_mk, 20);
            }
        }
    }

    if sbp_construct > 0 {
        if let Some((seed_sol, _seed_mk)) = sbp_construct_seed(pre, challenge, &greedy_sol, sbp_construct == 2, sbp_reopt_cycles) {
            let res = tabu_search_phase(pre, challenge, &seed_sol, ts_iters, ts_tenure, fast_eval, div_period, tie_mode, sel_key, exact_topk, 0x3C79AC492BA7B653, kick_spread, 0, 2)?;
            if let Some((sol5, mk5)) = res {
                sbp_seed_best = Some((sol5, mk5));
            }
        }

        if sbp_dual_cycles > 0 && sbp_dual_cycles != sbp_reopt_cycles {
            if let Some((seed_sol2, _seed_mk2)) = sbp_construct_seed(pre, challenge, &greedy_sol, sbp_construct == 2, sbp_dual_cycles) {
                let res2 = tabu_search_phase(pre, challenge, &seed_sol2, ts_iters, ts_tenure, fast_eval, div_period, tie_mode, sel_key, exact_topk, 0x3C79AC492BA7B653, kick_spread, 0, 2)?;
                if let Some((sol6, mk6)) = res2 {
                    let better = match sbp_seed_best.as_ref() {
                        Some((_, cur_mk)) => mk6 < *cur_mk,
                        None => true,
                    };
                    if better { sbp_seed_best = Some((sol6, mk6)); }
                }
            }
        }
    }

    let mut final_mk = best_makespan;
    if let Some(final_best) = best_solution.as_ref() {

        if let Some((sol4, mk4)) = tabu_search_phase(pre, challenge, final_best, ts_iters, ts_tenure, fast_eval, div_period, tie_mode, sel_key, exact_topk, 0, kick_spread, 0, polish_rescue)? {
            if mk4 < best_makespan { best_solution=Some(sol4.clone()); save_solution(&sol4)?; final_mk = mk4; }
        }
    }

    if sbp_rounds > 0 {
        let sbp_res = best_solution.as_ref().and_then(|cur| sbp_reoptimize(pre, challenge, cur, sbp_rounds));
        if let Some((sbp_sol, sbp_mk)) = sbp_res {
            if sbp_mk < final_mk { final_mk = sbp_mk; best_solution = Some(sbp_sol); }
        }
    }

    if let Some((extra_sol, extra_mk)) = ts_extra_best {
        if extra_mk < final_mk { best_solution = Some(extra_sol); final_mk = extra_mk; }
    }

    if let Some((sbp_sol, sbp_mk)) = sbp_seed_best {
        if sbp_mk < final_mk { best_solution = Some(sbp_sol); }
    }

    if let Some(sol) = best_solution { save_solution(&sol)?; }
    Ok(())
}}

pub mod solver {
use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::job_scheduling::*;

use super::types::EffortConfig;
use super::preprocess::build_pre;
use super::job_shop;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Track {
    FlowShop,
    HybridFlowShop,
    JobShop,
    FjspMedium,
    FjspHigh,
}

fn parse_track(hyperparameters: &Option<Map<String, Value>>) -> Track {
    if let Some(map) = hyperparameters {
        if let Some(Value::String(s)) = map.get("track") {
            return match s.to_lowercase().as_str() {
                "flow_shop" | "flow" => Track::FlowShop,
                "hybrid_flow_shop" | "hybrid" => Track::HybridFlowShop,
                "job_shop" | "job" => Track::JobShop,
                "fjsp_medium" | "medium" => Track::FjspMedium,
                "fjsp_high" | "high" | "fjsp" => Track::FjspHigh,
                _ => Track::FjspHigh,
            };
        }
    }
    Track::FjspHigh
}

fn parse_effort(hyperparameters: &Option<Map<String, Value>>) -> EffortConfig {
    let mut cfg = EffortConfig::default_effort();
    if let Some(map) = hyperparameters {
        if let Some(Value::Number(n)) = map.get("job_shop_iters") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_job_shop_iters(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_iters") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_hybrid_flow_shop_iters(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("fjsp_medium_iters") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_fjsp_medium_iters(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_seed_select_mode") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_seed_select_mode(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_ts_fast_eval") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_ts_fast_eval(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_ts_starts") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_ts_starts(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_ts_div_period") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_ts_div_period(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_ts_tie_mode") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_ts_tie_mode(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_ts_sel_key") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_ts_sel_key(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_ts_exact_topk") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_ts_exact_topk(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_ts_traj_decorr") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_ts_traj_decorr(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_ts_extra_draws") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_ts_extra_draws(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_ts_sa_accept") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_ts_sa_accept(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_mem_admit") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_mem_admit(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_mem_cross_scope") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_mem_cross_scope(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_ts_kick_spread") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_ts_kick_spread(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_sbp_rounds") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_sbp_rounds(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_ts_polish_rescue") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_ts_polish_rescue(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_seed_sbp_construct") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_seed_sbp_construct(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_seed_sbp_dual_cycles") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_seed_sbp_dual_cycles(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_seed_sbp_reopt_cycles") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_seed_sbp_reopt_cycles(v as usize);
            }
        }
    }
    cfg
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let pre = build_pre(challenge)?;
    let _track = parse_track(hyperparameters);
    let effort = parse_effort(hyperparameters);

    {
            job_shop::solve(challenge, save_solution, &pre, &effort)
        }
}

}
