// FLAT: track_t46.rs

pub use solver::{solve_challenge, help};

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
        pub stack: Vec<usize>,
    }

    impl EvalBuf {
        pub fn new(n: usize) -> Self {
            Self {
                indeg: vec![0u16; n],
                start: vec![0u32; n],
                best_pred: vec![NONE_USIZE; n],
                machine_succ: vec![NONE_USIZE; n],
                stack: Vec::with_capacity(n),
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
    }

    impl EffortConfig {
        pub fn default_effort() -> Self {
            Self { job_shop_iters: 10000, hybrid_flow_shop_iters: 2000, fjsp_medium_iters: 2000, fjsp_high_iters: 2000 }
        }

        pub fn with_job_shop_iters(mut self, v: usize) -> Self {
            self.job_shop_iters = v.clamp(100, 100000);
            self
        }
    }

}

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

    pub fn run_greedy_rule(
        challenge: &Challenge, job_products: &[usize], job_ops_len: &[usize], job_total_work: &[f64],
        rule: GreedyRule, mut random_top_k: Option<(usize, &mut SmallRng)>,
    ) -> Result<(Solution, u32)> {
        let num_jobs = challenge.num_jobs; let num_machines = challenge.num_machines;
        let mut job_next_op = vec![0usize; num_jobs]; let mut job_ready = vec![0u32; num_jobs]; let mut machine_avail = vec![0u32; num_machines];
        let mut job_schedule: Vec<Vec<(usize, u32)>> = job_ops_len.iter().map(|&len| Vec::with_capacity(len)).collect();
        let mut job_work_left = job_total_work.to_vec();
        let mut remaining = job_ops_len.iter().sum::<usize>(); let mut time = 0u32; let eps = 1e-9;

        while remaining > 0 {
            let mut available_machines: Vec<usize> = (0..num_machines).filter(|&m| machine_avail[m] <= time).collect();
            available_machines.sort_unstable();
            if let Some((_, ref mut rng)) = random_top_k { use rand::seq::SliceRandom; available_machines.shuffle(*rng); }

            for &m in &available_machines {
                #[derive(Clone)]
                struct GCandidate { job: usize, priority: f64, end: u32, pt: u32, flex: usize }
                let mut candidates: Vec<GCandidate> = Vec::new();
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

                let best_job = if let Some((top_k, ref mut rng)) = random_top_k {
                    candidates.sort_by(|a, b| { if (b.priority - a.priority).abs() > eps { b.priority.partial_cmp(&a.priority).unwrap() } else if a.end != b.end { a.end.cmp(&b.end) } else if a.pt != b.pt { a.pt.cmp(&b.pt) } else if a.flex != b.flex { a.flex.cmp(&b.flex) } else { a.job.cmp(&b.job) } });
                    let top = candidates.len().min(top_k); candidates[rng.gen_range(0..top)].job
                } else {
                    let mut best: Option<GCandidate> = None;
                    for cand in candidates { let better = if let Some(ref b) = best { if (cand.priority - b.priority).abs() > eps { cand.priority > b.priority } else if cand.end != b.end { cand.end < b.end } else if cand.pt != b.pt { cand.pt < b.pt } else if cand.flex != b.flex { cand.flex < b.flex } else { cand.job < b.job } } else { true }; if better { best = Some(cand); } }
                    best.ok_or_else(|| anyhow!("No candidate"))?.job
                };
                let product = job_products[best_job]; let op_idx = job_next_op[best_job];
                let op_times = &challenge.product_processing_times[product][op_idx];
                let pt = op_times[&m]; let avg_pt = op_times.values().sum::<u32>() as f64 / op_times.len().max(1) as f64;
                let st = time.max(machine_avail[m]); let end = st + pt;
                job_schedule[best_job].push((m, st)); job_next_op[best_job] += 1; job_ready[best_job] = end; machine_avail[m] = end;
                job_work_left[best_job] -= avg_pt; if job_work_left[best_job] < 0.0 { job_work_left[best_job] = 0.0; } remaining -= 1;
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
        Ok(DisjSchedule { n, num_jobs, num_machines, job_offsets, job_succ, indeg_job, node_machine, node_pt, node_job, node_op, machine_seq })
    }

    #[inline]
    pub fn pt_from_op(op: &OpInfo, machine: usize) -> Option<u32> {
        for &(m, pt) in &op.machines { if m == machine { return Some(pt); } }
        None
    }

    pub fn eval_disj(ds: &DisjSchedule, buf: &mut EvalBuf) -> Option<(u32, usize)> {
        let n = ds.n;
        buf.indeg.clone_from_slice(&ds.indeg_job);
        buf.start.fill(0); buf.best_pred.fill(NONE_USIZE); buf.machine_succ.fill(NONE_USIZE); buf.stack.clear();
        for seq in &ds.machine_seq { if seq.len() <= 1 { continue; } for i in 0..(seq.len()-1) { let u = seq[i]; let v = seq[i+1]; buf.machine_succ[u] = v; buf.indeg[v] = buf.indeg[v].saturating_add(1); } }
        for i in 0..n { if buf.indeg[i] == 0 { buf.stack.push(i); } }
        let mut processed = 0usize; let mut mk = 0u32; let mut mk_node = 0usize;
        while let Some(u) = buf.stack.pop() {
            processed += 1; let end_u = buf.start[u].saturating_add(ds.node_pt[u]);
            if end_u > mk { mk = end_u; mk_node = u; }
            let js = ds.job_succ[u]; if js != NONE_USIZE { if buf.start[js] < end_u { buf.start[js] = end_u; buf.best_pred[js] = u; } buf.indeg[js] = buf.indeg[js].saturating_sub(1); if buf.indeg[js] == 0 { buf.stack.push(js); } }
            let ms = buf.machine_succ[u]; if ms != NONE_USIZE { if buf.start[ms] < end_u { buf.start[ms] = end_u; buf.best_pred[ms] = u; } buf.indeg[ms] = buf.indeg[ms].saturating_sub(1); if buf.indeg[ms] == 0 { buf.stack.push(ms); } }
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

    fn descent_phase(
        ds: &mut DisjSchedule, buf: &mut EvalBuf, crit: &mut Vec<bool>, pre: &Pre,
        cur_eval: &mut (u32, usize), max_iters: usize, top_cands: usize,
    ) -> bool {
        let mut cur_mk = cur_eval.0; let mut improved = false;
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
            for cand in &cands {
                if cand.kind == 0 {
                    let m = cand.m_from; if m >= ds.num_machines || cand.from >= ds.machine_seq[m].len() { continue; }
                    let new_idx = apply_insert(&mut ds.machine_seq[m], cand.from, cand.to);
                    if let Some((mk2, _)) = eval_disj(ds, buf) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                    let _ = apply_insert(&mut ds.machine_seq[m], new_idx, cand.from);
                } else if cand.kind == 2 {
                    let m = cand.m_from; if m >= ds.num_machines || cand.from + 1 >= ds.machine_seq[m].len() { continue; }
                    if !apply_swap(&mut ds.machine_seq[m], cand.from) { continue; }
                    if let Some((mk2, _)) = eval_disj(ds, buf) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                    let _ = apply_swap(&mut ds.machine_seq[m], cand.from);
                } else {
                    let m_from = cand.m_from; let m_to = cand.m_to;
                    if m_from >= ds.num_machines || m_to >= ds.num_machines || cand.from >= ds.machine_seq[m_from].len() { continue; }
                    let node = ds.machine_seq[m_from][cand.from]; if ds.node_machine[node] != m_from { continue; }
                    if let Some((node2, old_pt, ins_idx)) = apply_reroute(ds, m_from, cand.from, m_to, cand.to, cand.new_pt) {
                        if let Some((mk2, _)) = eval_disj(ds, buf) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
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
        if seq.is_empty() || from >= seq.len() { return from.min(seq.len().saturating_sub(1)); }
        let x = seq.remove(from); let t = to_after_removal.min(seq.len()); seq.insert(t, x); t
    }

    #[inline]
    pub fn apply_swap(seq: &mut [usize], i: usize) -> bool {
        if i + 1 >= seq.len() { return false; } seq.swap(i, i + 1); true
    }

    #[inline]
    pub fn find_insert_pos_by_start(seq: &[usize], start: &[u32], desired_start: u32) -> usize {
        for (i, &id) in seq.iter().enumerate() { if start[id] >= desired_start { return i; } } seq.len()
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
    }
}

pub mod solver {
    use anyhow::Result;
    use serde_json::{Map, Value};
    use tig_challenges::job_scheduling::*;

    use super::types::EffortConfig;
    use super::preprocess::build_pre;
    use super::flow_shop;
    use super::hybrid_flow_shop;
    use super::job_shop;
    use super::fjsp_medium;
    use super::fjsp_high;

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
        }
        cfg
    }

    pub fn solve_challenge(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        hyperparameters: &Option<Map<String, Value>>,
    ) -> Result<()> {
        let pre = build_pre(challenge)?;
        let track = parse_track(hyperparameters);
        let effort = parse_effort(hyperparameters);

        match track {
            Track::FlowShop => {
                flow_shop::solve(challenge, save_solution, &pre, &effort)
            }
            Track::HybridFlowShop => {
                hybrid_flow_shop::solve(challenge, save_solution, &pre, &effort)
            }
            Track::JobShop => {
                job_shop::solve(challenge, save_solution, &pre, &effort)
            }
            Track::FjspMedium => {
                fjsp_medium::solve(challenge, save_solution, &pre, &effort)
            }
            Track::FjspHigh => {
                fjsp_high::solve(challenge, save_solution, &pre, &effort)
            }
        }
    }

    pub fn help() {
        println!("Job Scheduling Solver - Modular Independent Track Architecture v1");
        println!();
        println!("DESCRIPTION:");
        println!("  Each track is fully self-contained. The track file is depicted by using the correct hyperparameter 'track'");
        println!();
        println!("HYPERPARAMETERS:");
        println!("  track (required): \"flow_shop\" | \"hybrid_flow_shop\" | \"job_shop\" | \"fjsp_medium\" | \"fjsp_high\"");
        println!();
        println!("  job_shop_iters:  integer, default 10000, max 100000  (tabu search depth)");
        println!();
        println!("NOTES:");
        println!("  job_shop_iters scales strongly - higher values give meaningfully better quality at the cost of runtime.");
        println!("  All other tracks have fixed internal iteration counts and are not tunable.");
    }

}

pub mod flow_shop {
    use anyhow::{anyhow, Result};
    use rand::{rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};
    use std::cell::RefCell;
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    use tig_challenges::job_scheduling::*;
    use super::types::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Rule {
        BnHeavy,
        MostWork,
        EndTight,
        ShortestProc,
        LeastFlex,
        CriticalPath,
        Regret,
        EarliestStart,
        MachineBalance,
        SlackRatio,
        BackwardCritical,
        WeightedCompletion,
    }

    #[inline]
    fn score_candidate(
        pre: &Pre,
        rule: Rule,
        job: usize,
        product: usize,
        op_idx: usize,
        ops_rem: usize,
        op: &OpInfo,
        machine: usize,
        pt: u32,
        time: u32,
        target_mk: Option<u32>,
        best_end: u32,
        second_end: u32,
        best_cnt_total: usize,
        progress: f64,
        job_bias: f64,
        _machine_penalty: f64,
        dynamic_load: f64,
        route_pref: Option<&RoutePrefLite>,
        route_w: f64,
        jitter: f64,
    ) -> f64 {
        let rem_min = pre.product_suf_min[product][op_idx] as f64;
        let rem_avg = pre.product_suf_avg[product][op_idx];
        let rem_bn = pre.product_suf_bn[product][op_idx];
        let flex_f = (op.flex as f64).max(1.0);
        let flex_inv = 1.0 / flex_f;
        let rem_min_n = rem_min / pre.horizon.max(1.0);
        let _rem_avg_n = rem_avg / pre.max_job_avg_work.max(1e-9);
        let _bn_n = rem_bn / pre.max_job_bn.max(1e-9);
        let ops_n = (ops_rem as f64) / (pre.max_ops as f64).max(1.0);
        let _load_n = dynamic_load / pre.avg_machine_load.max(1e-9);
        let _scar_n = pre.machine_scarcity[machine] / pre.avg_machine_scarcity.max(1e-9);
        let end_n = (best_end as f64) / pre.time_scale.max(1.0);
        let proc_n = (pt as f64) / pre.avg_op_min.max(1.0);
        let regret = if second_end >= INF {
            pre.avg_op_min * 2.6
        } else {
            (second_end - best_end) as f64
        };
        let reg_n = (regret / pre.avg_op_min.max(1.0)).clamp(0.0, 6.0);
        let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
        let density_n =
            ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0);
        let next_min = pre.product_next_min[product][op_idx] as f64;
        let next_min_n = next_min / pre.horizon.max(1.0);
        let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
        let p2 = progress * progress;
        let next_w_base = 0.12 + p2 * 0.28;
        let next_term_raw =
            (0.55 * next_min_n + 0.45 * next_flex_inv) * (1.0 + 0.30 * density_n * pre.high_flex);
        let js = pre.jobshopness;
        let _fl = 1.0 - js;
        let pop_pen = if pre.chaotic_like && op.flex >= 2 {
            let pop = pre.machine_best_pop[machine];
            (0.07 + 0.15 * (1.0 - progress)).clamp(0.05, 0.24) * pop * pre.flex_factor
        } else {
            0.0
        };

        let slack_u = if let Some(tgt) = target_mk {
            let lb = (time as u64).saturating_add(pre.product_suf_min[product][op_idx] as u64);
            let slack = (tgt as i64) - (lb as i64);
            let scale = (0.70 * pre.avg_op_min).max(1.0);
            let pos = (slack.max(0) as f64) / scale;
            let neg = ((-slack).max(0) as f64) / scale;
            (1.0 / (1.0 + pos)).clamp(0.0, 1.0) + (0.35 * neg).min(3.0)
        } else {
            0.0
        };
        let _slack_w = pre.slack_base * (0.25 + 0.75 * progress);

        let flow_term = pre.flow_w * pre.job_flow_pref[job] * (0.65 + 0.70 * (1.0 - progress));
        let route_term = if route_w > 0.0 && op.flex >= 2 {
            let rp = route_pref;
            let bonus = if let Some(rp) = rp {
                if product < rp.len() && op_idx < rp[product].len() {
                    let r = rp[product][op_idx];
                    let mu = machine.min(255) as u8;
                    if mu == r.best_m {
                        (r.best_w as f64) / 255.0
                    } else if mu == r.second_m {
                        (r.second_w as f64) / 255.0
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let route_gain = (0.70 + 0.80 * (1.0 - progress)).clamp(0.70, 1.40);
            route_w * route_gain * bonus
        } else {
            0.0
        };

        let _ = (
            ops_n, _rem_avg_n, _bn_n, _load_n, _scar_n, _fl, _slack_w, slack_u, flex_inv,
        );

        match rule {
            Rule::BnHeavy => {
                let bn_w = (0.90 + 0.55 * js) * pre.bn_focus;
                let end_w = 0.65 + 0.70 * progress;
                let reg_w = (0.60 + 0.25 * (1.0 - progress)) * (0.85 + 0.35 * js);
                let next_term = next_w_base * (0.55 + 0.75 * js) * next_term_raw;
                (0.95 * rem_min_n)
                    + (bn_w * rem_bn / pre.max_job_bn.max(1e-9))
                    + (0.10 * ops_n)
                    + (reg_w * pre.flex_factor) * reg_n
                    + 0.18 * scarcity_urg
                    + next_term
                    - end_w * end_n
                    - 0.18 * proc_n
                    - pop_pen
                    + 0.60 * job_bias
                    + flow_term
                    + route_term
                    + jitter
            }
            Rule::MostWork => {
                let next_term = next_w_base * 0.25 * next_term_raw;
                (1.00 * rem_avg) / pre.max_job_avg_work.max(1e-9)
                    + (0.12 * ops_n)
                    + (0.18 * scarcity_urg)
                    + next_term
                    - (0.62 * end_n)
                    - pop_pen
                    + (0.45 * job_bias)
                    + flow_term
                    + route_term
                    + jitter
            }
            Rule::EndTight => {
                let end_w = 1.10 + 1.00 * progress + 0.35 * pre.high_flex;
                let cp_w = 1.15 + 0.30 * js;
                let reg_w = (0.55 + 0.20 * (1.0 - progress)) * (0.85 + 0.60 * js);
                let next_term = next_w_base * (0.45 + 0.55 * js) * next_term_raw;
                (cp_w * rem_min_n)
                    + 0.08 * ops_n
                    + 0.18 * scarcity_urg
                    + (reg_w * pre.flex_factor) * reg_n
                    + next_term
                    - end_w * end_n
                    - 0.22 * proc_n
                    - pop_pen
                    + 0.55 * job_bias
                    + flow_term
                    + route_term
                    + jitter
            }
            Rule::ShortestProc => {
                let next_term = next_w_base * 0.20 * next_term_raw;
                (-1.00 * proc_n)
                    + (0.25 * rem_min_n)
                    + (0.12 * scarcity_urg)
                    + next_term
                    - (0.20 * end_n)
                    - pop_pen
                    + (0.25 * job_bias)
                    + flow_term
                    + route_term
                    + jitter
            }
            Rule::LeastFlex => {
                let next_term = next_w_base * 0.20 * next_term_raw;
                (1.00 * flex_inv)
                    + (0.28 * rem_min_n)
                    + (0.22 * scarcity_urg)
                    + next_term
                    - (0.55 * end_n)
                    - pop_pen
                    + (0.35 * job_bias)
                    + flow_term
                    + route_term
                    + jitter
            }
            Rule::CriticalPath => {
                let next_term = next_w_base * 0.30 * next_term_raw;
                (1.03 * rem_min_n)
                    + (0.10 * ops_n)
                    + (0.24 * scarcity_urg)
                    + next_term
                    - (0.70 * end_n)
                    - pop_pen
                    + (0.45 * job_bias)
                    + flow_term
                    + route_term
                    + jitter
            }
            Rule::Regret => {
                let next_term = next_w_base * 0.25 * next_term_raw;
                (1.05 * reg_n)
                    + (0.55 * rem_min_n)
                    + (0.22 * scarcity_urg)
                    + next_term
                    - (0.68 * end_n)
                    - pop_pen
                    + (0.35 * job_bias)
                    + flow_term
                    + route_term
                    + jitter
            }
            Rule::EarliestStart => {
                let start_n = (time as f64) / pre.time_scale.max(1.0);
                let next_term = next_w_base * 0.20 * next_term_raw;
                -(1.20 * start_n)
                    + (0.40 * rem_min_n)
                    + (0.15 * scarcity_urg)
                    + next_term
                    - (0.30 * proc_n)
                    - pop_pen
                    + (0.30 * job_bias)
                    + flow_term
                    + route_term
                    + jitter
            }
            Rule::MachineBalance => {
                let load_n = dynamic_load / pre.avg_machine_load.max(1e-9);
                let next_term = next_w_base * 0.20 * next_term_raw;
                -(0.80 * load_n)
                    + (0.50 * rem_min_n)
                    + (0.25 * scarcity_urg)
                    + next_term
                    - (0.45 * end_n)
                    - pop_pen
                    + (0.35 * job_bias)
                    + flow_term
                    + route_term
                    + jitter
            }
            Rule::SlackRatio => {
                let time_to_horizon = (pre.horizon - time as f64).max(1.0);
                let cr = (rem_min / time_to_horizon).clamp(0.0, 4.0);
                let next_term = next_w_base * 0.25 * next_term_raw;
                (1.10 * cr)
                    + (0.35 * rem_min_n)
                    + (0.20 * scarcity_urg)
                    + next_term
                    - (0.55 * end_n)
                    - pop_pen
                    + (0.40 * job_bias)
                    + flow_term
                    + route_term
                    + jitter
            }
            Rule::BackwardCritical => {
                let bn_suf = pre.product_suf_bn[product][op_idx] as f64 / pre.max_job_bn.max(1e-9);
                let density =
                    ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0);
                let next_term = next_w_base * 0.30 * next_term_raw;
                (1.15 * bn_suf)
                    + (0.45 * density)
                    + (0.20 * scarcity_urg)
                    + next_term
                    - (0.60 * end_n)
                    - pop_pen
                    + (0.40 * job_bias)
                    + flow_term
                    + route_term
                    + jitter
            }
            Rule::WeightedCompletion => {
                let work_n = rem_avg / pre.max_job_avg_work.max(1e-9);
                let wspt = if best_end > 0 {
                    work_n / (best_end as f64 / pre.time_scale.max(1.0)).max(0.01)
                } else {
                    work_n
                };
                let next_term = next_w_base * 0.20 * next_term_raw;
                (1.20 * wspt)
                    + (0.30 * rem_min_n)
                    + (0.15 * scarcity_urg)
                    + next_term
                    - (0.40 * end_n)
                    - pop_pen
                    + (0.35 * job_bias)
                    + flow_term
                    + route_term
                    + jitter
            }
        }
    }

    fn construct_solution_conflict(
        challenge: &Challenge,
        pre: &Pre,
        rule: Rule,
        k: usize,
        target_mk: Option<u32>,
        rng: &mut SmallRng,
        job_bias: Option<&[f64]>,
        machine_penalty: Option<&[f64]>,
        route_pref: Option<&RoutePrefLite>,
        route_w: f64,
    ) -> Result<(Solution, u32)> {
        let num_jobs = challenge.num_jobs;
        let num_machines = challenge.num_machines;
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
        let mut idle_machines: Vec<usize> = Vec::with_capacity(num_machines);

        while remaining_ops > 0 {
            loop {
                idle_machines.clear();
                for m in 0..num_machines {
                    if machine_avail[m] <= time {
                        idle_machines.push(m);
                    }
                }
                if idle_machines.is_empty() {
                    break;
                }
                for &m in &idle_machines {
                    demand[m] = 0;
                    raw_by_machine[m].clear();
                }
                let progress = 1.0 - (remaining_ops as f64) / (pre.total_ops as f64).max(1.0);
                let cap_per_machine = if k == 0 { 12usize } else { (k + 6).min(12) };

                for job in 0..num_jobs {
                    let op_idx = job_next_op[job];
                    if op_idx >= pre.job_ops_len[job] || job_ready_time[job] > time {
                        continue;
                    }
                    let product = pre.job_products[job];
                    let op = &pre.product_ops[product][op_idx];
                    if op.flex == 0 || op.machines.is_empty() || op.min_pt >= INF {
                        continue;
                    }
                    let (best_end, second_end, best_cnt_total, best_cnt_idle) =
                        best_second_and_counts(time, &machine_avail, op);
                    if best_end >= INF || best_cnt_idle == 0 {
                        continue;
                    }
                    let ops_rem = pre.job_ops_len[job] - op_idx;
                    let jb = job_bias.map(|v| v[job]).unwrap_or(0.0);
                    let flex_inv = 1.0 / (op.flex as f64).max(1.0);
                    let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
                    let regret = if second_end >= INF {
                        pre.avg_op_min * 2.6
                    } else {
                        (second_end - best_end) as f64
                    };
                    let regn = (regret / pre.avg_op_min.max(1.0)).clamp(0.0, 6.0);
                    let rigidity = (0.60 * flex_inv + 0.40 * scarcity_urg).clamp(0.0, 2.5);

                    for &(m, pt) in &op.machines {
                        if machine_avail[m] > time {
                            continue;
                        }
                        let end = time.saturating_add(pt);
                        if end != best_end {
                            continue;
                        }
                        demand[m] = demand[m].saturating_add(1);
                        let mp = machine_penalty.map(|v| v[m]).unwrap_or(0.0);
                        let jitter = if k > 0 { rng.gen::<f64>() * 1e-9 } else { 0.0 };
                        let base = score_candidate(
                            pre,
                            rule,
                            job,
                            product,
                            op_idx,
                            ops_rem,
                            op,
                            m,
                            pt,
                            time,
                            target_mk,
                            best_end,
                            second_end,
                            best_cnt_total,
                            progress,
                            jb,
                            mp,
                            machine_load[m],
                            route_pref,
                            route_w,
                            jitter,
                        );
                        push_top_k_raw(
                            &mut raw_by_machine[m],
                            RawCand { job, machine: m, pt, base_score: base, rigidity, reg_n: regn },
                            cap_per_machine,
                        );
                    }
                }

                let denom = (idle_machines.len() as f64).max(1.0);
                let conflict_w = (0.09
                    + 0.26 * pre.jobshopness
                    + 0.11 * pre.high_flex
                    + 0.16
                        * (1.0
                            - (1.0 - (remaining_ops as f64) / (pre.total_ops as f64).max(1.0))))
                .clamp(0.05, 0.45);
                let conflict_scale = (0.90 + 0.40 * pre.flex_factor).clamp(0.85, 1.75);

                let mut best: Option<Cand> = None;
                let mut top: Vec<Cand> = if k > 0 { Vec::with_capacity(k) } else { Vec::new() };

                for &m in &idle_machines {
                    let dem = demand[m] as f64;
                    if dem <= 0.0 || raw_by_machine[m].is_empty() {
                        continue;
                    }
                    let dem_n = ((dem - 1.0) / denom).clamp(0.0, 2.5);
                    for rc in &raw_by_machine[m] {
                        let rig = rc.rigidity.clamp(0.0, 2.5);
                        let regc = rc.reg_n.clamp(0.0, 4.5);
                        let boost = conflict_w * conflict_scale * dem_n * (1.15 * rig + 0.85 * regc);
                        let c = Cand { job: rc.job, machine: rc.machine, pt: rc.pt, score: rc.base_score + boost };
                        if k == 0 {
                            if best.map_or(true, |bb| c.score > bb.score) {
                                best = Some(c);
                            }
                        } else {
                            push_top_k(&mut top, c, k);
                        }
                    }
                }

                let chosen = if k == 0 {
                    match best {
                        Some(c) => c,
                        None => break,
                    }
                } else {
                    if top.is_empty() {
                        break;
                    }
                    choose_from_top_weighted(rng, &top)
                };

                let job = chosen.job;
                let machine = chosen.machine;
                let pt = chosen.pt;
                let product = pre.job_products[job];
                let op_idx = job_next_op[job];
                let op = &pre.product_ops[product][op_idx];
                let (best_end_now, _, _, _) = best_second_and_counts(time, &machine_avail, op);
                let end_check = time.max(machine_avail[machine]).saturating_add(pt);
                if machine_avail[machine] > time || end_check != best_end_now {
                    break;
                }
                let end_time = time.saturating_add(pt);
                job_schedule[job].push((machine, time));
                job_next_op[job] += 1;
                job_ready_time[job] = end_time;
                machine_avail[machine] = end_time;
                remaining_ops -= 1;
                if op.min_pt < INF && op.flex > 0 && !op.machines.is_empty() {
                    let delta = (op.min_pt as f64) / (op.flex as f64).max(1.0);
                    if delta > 0.0 {
                        for &(mm, _) in &op.machines {
                            let v = machine_load[mm] - delta;
                            machine_load[mm] = if v > 0.0 { v } else { 0.0 };
                        }
                    }
                }
                if remaining_ops == 0 {
                    break;
                }
            }
            if remaining_ops == 0 {
                break;
            }
            let mut next_time: Option<u32> = None;
            for &t in &machine_avail {
                if t > time {
                    next_time = Some(next_time.map_or(t, |b| b.min(t)));
                }
            }
            for j in 0..num_jobs {
                let op_idx = job_next_op[j];
                if op_idx < pre.job_ops_len[j] && job_ready_time[j] > time {
                    let t = job_ready_time[j];
                    next_time = Some(next_time.map_or(t, |b| b.min(t)));
                }
            }
            time = next_time.ok_or_else(|| anyhow!("Stalled: no next event"))?;
        }
        let mk = machine_avail.into_iter().max().unwrap_or(0);
        Ok((Solution { job_schedule }, mk))
    }

    #[inline]
    fn best_second_and_counts(time: u32, machine_avail: &[u32], op: &OpInfo) -> (u32, u32, usize, usize) {
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

    #[inline]
    fn push_top_k(top: &mut Vec<Cand>, c: Cand, k: usize) {
        if k == 0 {
            return;
        }
        let mut pos = top.len();
        while pos > 0 && top[pos - 1].score < c.score {
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

    #[inline]
    fn push_top_k_raw(top: &mut Vec<RawCand>, c: RawCand, k: usize) {
        if k == 0 {
            return;
        }
        let mut pos = top.len();
        while pos > 0 && top[pos - 1].base_score < c.base_score {
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

    #[inline]
    fn choose_from_top_weighted(rng: &mut SmallRng, top: &[Cand]) -> Cand {    
        let n = top.len();
        if n <= 1 {
            return top[0];
        }
        if n == 2 {
            return top[rng.gen_range(0..2)];
        }

        let b1 = n / 3;
        let b2 = (2 * n) / 3;

        let mut ranges: [(usize, usize); 3] = [(0, b1), (b1, b2), (b2, n)];
        let mut cnt = 0usize;
        for i in 0..3 {
            if ranges[i].0 < ranges[i].1 {
                ranges[cnt] = ranges[i];
                cnt += 1;
            }
        }

        let (s, e) = ranges[rng.gen_range(0..cnt)];
        top[s + rng.gen_range(0..(e - s))]
    }

    #[inline]
    fn push_top_solutions(top: &mut Vec<(Solution, u32)>, sol: &Solution, mk: u32, cap: usize) {
        if cap == 0 {
            return;
        }

        let num_jobs = sol.job_schedule.len().max(1);
        let ksig = cap.min(num_jobs);

        let signature = |s: &Solution| -> Vec<usize> {
            let mut best: Vec<(u32, usize)> = Vec::with_capacity(ksig);
            for j in 0..s.job_schedule.len() {
                let t = s.job_schedule[j]
                    .first()
                    .map(|x| x.1)
                    .unwrap_or(u32::MAX);

                let mut pos = best.len();
                while pos > 0 {
                    let (bt, bj) = best[pos - 1];
                    if bt < t || (bt == t && bj < j) {
                        break;
                    }
                    pos -= 1;
                }
                if pos >= ksig {
                    continue;
                }
                best.insert(pos, (t, j));
                if best.len() > ksig {
                    best.pop();
                }
            }
            best.into_iter().map(|(_, j)| j).collect()
        };

        let similarity = |a: &[usize], b: &[usize]| -> usize {
            let len = a.len().min(b.len());
            let mut same = 0usize;
            for i in 0..len {
                if a[i] == b[i] {
                    same += 1;
                }
            }
            same
        };

        let sig_new = signature(sol);
        let mut sigs: Vec<Vec<usize>> = Vec::with_capacity(top.len());
        let mut best_sim = 0usize;
        let mut best_idx = NONE_USIZE;

        for (i, (s2, _)) in top.iter().enumerate() {
            let sig2 = signature(s2);
            let sim = similarity(&sig_new, &sig2);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
            sigs.push(sig2);
        }

        if best_idx != NONE_USIZE && best_sim >= ksig {
            if mk < top[best_idx].1 {
                top[best_idx] = (sol.clone(), mk);
            }
            return;
        }

        if top.len() < cap {
            top.push((sol.clone(), mk));
            return;
        }

        let mut crowd_max: Vec<usize> = vec![0usize; top.len()];
        for i in 0..top.len() {
            for j in (i + 1)..top.len() {
                let sim = similarity(&sigs[i], &sigs[j]);
                if sim > crowd_max[i] {
                    crowd_max[i] = sim;
                }
                if sim > crowd_max[j] {
                    crowd_max[j] = sim;
                }
            }
        }

        let mut evict_idx = 0usize;
        let mut evict_crowd = crowd_max[0];
        for i in 1..top.len() {
            let crowd = crowd_max[i];
            if crowd > evict_crowd || (crowd == evict_crowd && top[i].1 > top[evict_idx].1) {
                evict_crowd = crowd;
                evict_idx = i;
            }
        }

        let mut new_crowd = 0usize;
        for sig in &sigs {
            let sim = similarity(&sig_new, sig);
            if sim > new_crowd {
                new_crowd = sim;
            }
        }

        if new_crowd < evict_crowd || mk < top[evict_idx].1 {
            top[evict_idx] = (sol.clone(), mk);
        }
    }

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
                let v = comp[k].max(comp[k - 1]).saturating_add(row[k]);
                comp[k] = v;
            }
        }
        *comp.last().unwrap_or(&0)
    }

    #[inline]
    fn reentrant_makespan(seq: &[usize], route: &[usize], pt: &[Vec<u32>], mready: &mut [u32]) -> u32 {
        mready.fill(0);
        let mut mk = 0u32;
        for &j in seq {
            let row = &pt[j];
            let mut prev = 0u32;
            for (op_idx, &m) in route.iter().enumerate() {
                let p = row[op_idx];
                let st = prev.max(mready[m]);
                let end = st.saturating_add(p);
                mready[m] = end;
                prev = end;
            }
            if prev > mk {
                mk = prev;
            }
        }
        mk
    }

    fn build_disj_from_solution(pre: &Pre, challenge: &Challenge, sol: &Solution) -> Result<DisjSchedule> {
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
                return Err(anyhow!("Invalid solution"));
            }
            let product = pre.job_products[job];
            for op_idx in 0..expected {
                let id = job_offsets[job] + op_idx;
                let (m, st) = sol.job_schedule[job][op_idx];
                let op = &pre.product_ops[product][op_idx];
                let pt = op
                    .machines
                    .iter()
                    .find(|&&(mm, _)| mm == m)
                    .map(|&(_, p)| p)
                    .ok_or_else(|| anyhow!("pt missing"))?;
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
        let mut job_succ = vec![NONE_USIZE; n];
        let mut indeg_job = vec![0u16; n];
        for job in 0..num_jobs {
            let base = job_offsets[job];
            for k in 0..pre.job_ops_len[job] {
                let id = base + k;
                if k + 1 < pre.job_ops_len[job] {
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

    fn eval_disj(ds: &DisjSchedule, buf: &mut EvalBuf) -> Option<(u32, usize)> {
        let n = ds.n;
        buf.indeg.clone_from_slice(&ds.indeg_job);
        buf.start.fill(0);
        buf.best_pred.fill(NONE_USIZE);
        buf.machine_succ.fill(NONE_USIZE);
        buf.stack.clear();
        for seq in &ds.machine_seq {
            if seq.len() <= 1 {
                continue;
            }
            for i in 0..(seq.len() - 1) {
                let u = seq[i];
                let v = seq[i + 1];
                buf.machine_succ[u] = v;
                buf.indeg[v] = buf.indeg[v].saturating_add(1);
            }
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
            let end_u = buf.start[u].saturating_add(ds.node_pt[u]);
            if end_u > mk {
                mk = end_u;
                mk_node = u;
            }
            let js = ds.job_succ[u];
            if js != NONE_USIZE {
                if buf.start[js] < end_u {
                    buf.start[js] = end_u;
                    buf.best_pred[js] = u;
                }
                buf.indeg[js] = buf.indeg[js].saturating_sub(1);
                if buf.indeg[js] == 0 {
                    buf.stack.push(js);
                }
            }
            let ms = buf.machine_succ[u];
            if ms != NONE_USIZE {
                if buf.start[ms] < end_u {
                    buf.start[ms] = end_u;
                    buf.best_pred[ms] = u;
                }
                buf.indeg[ms] = buf.indeg[ms].saturating_sub(1);
                if buf.indeg[ms] == 0 {
                    buf.stack.push(ms);
                }
            }
        }
        if processed != n {
            return None;
        }
        Some((mk, mk_node))
    }

    fn disj_to_solution(pre: &Pre, ds: &DisjSchedule, start: &[u32]) -> Result<Solution> {
        let num_jobs = ds.num_jobs;
        let mut job_schedule: Vec<Vec<(usize, u32)>> = Vec::with_capacity(num_jobs);
        for j in 0..num_jobs {
            let len = pre.job_ops_len[j];
            let mut v = Vec::with_capacity(len);
            let base = ds.job_offsets[j];
            for k in 0..len {
                let id = base + k;
                v.push((ds.node_machine[id], start[id]));
            }
            job_schedule.push(v);
        }
        Ok(Solution { job_schedule })
    }

    #[inline]
    fn arc_key_fs(machine: usize, left: usize, right: usize) -> u64 {
        ((machine as u64 + 1) << 42) ^ ((left as u64 + 1) << 21) ^ (right as u64 + 1)
    }

    #[inline]
    fn collect_machine_arcs(seq: &[usize], machine: usize, out: &mut Vec<u64>) {
        out.clear();
        if seq.len() <= 1 {
            return;
        }
        for i in 0..(seq.len() - 1) {
            out.push(arc_key_fs(machine, seq[i], seq[i + 1]));
        }
    }

    #[inline]
    fn move_hits_recent_arc(
        ds: &DisjSchedule,
        cand: &MoveCand,
        tabu_keys: &[u64],
        tabu_until: &[usize],
        step: usize,
        old_arcs: &mut Vec<u64>,
        new_arcs: &mut Vec<u64>,
        tmp_seq: &mut Vec<usize>,
    ) -> bool {
        let m = cand.m_from;
        if m >= ds.num_machines {
            return false;
        }
        let seq = &ds.machine_seq[m];
        if seq.len() <= 1 {
            return false;
        }

        tmp_seq.clear();
        tmp_seq.extend_from_slice(seq);

        match cand.kind {
            0 => {
                if cand.from >= tmp_seq.len() {
                    return false;
                }
                let _ = apply_insert_fs(tmp_seq, cand.from, cand.to);
            }
            2 => {
                if cand.from + 1 >= tmp_seq.len() {
                    return false;
                }
                tmp_seq.swap(cand.from, cand.from + 1);
            }
            3 => {
                let len = tmp_seq.len();
                if cand.from >= len || cand.to >= len || cand.from + 1 >= cand.to {
                    return false;
                }
                tmp_seq.swap(cand.from, cand.to);
            }
            _ => return false,
        }

        collect_machine_arcs(seq, m, old_arcs);
        collect_machine_arcs(tmp_seq, m, new_arcs);

        for &key in old_arcs.iter() {
            let mut still_present = false;
            for &k2 in new_arcs.iter() {
                if k2 == key {
                    still_present = true;
                    break;
                }
            }
            if still_present {
                continue;
            }
            for i in 0..tabu_keys.len() {
                if tabu_keys[i] == key && tabu_until[i] > step {
                    return true;
                }
            }
        }
        false
    }

    #[inline]
    fn protect_recent_created_arcs(
        before_seq: &[usize],
        after_seq: &[usize],
        machine: usize,
        tenure: usize,
        step: usize,
        tabu_keys: &mut [u64],
        tabu_until: &mut [usize],
        tabu_pos: &mut usize,
        old_arcs: &mut Vec<u64>,
        new_arcs: &mut Vec<u64>,
    ) {
        if tabu_keys.is_empty() {
            return;
        }

        collect_machine_arcs(before_seq, machine, old_arcs);
        collect_machine_arcs(after_seq, machine, new_arcs);

        for &key in new_arcs.iter() {
            let mut existed = false;
            for &k2 in old_arcs.iter() {
                if k2 == key {
                    existed = true;
                    break;
                }
            }
            if existed {
                continue;
            }
            let idx = *tabu_pos % tabu_keys.len();
            tabu_keys[idx] = key;
            tabu_until[idx] = step.saturating_add(tenure).saturating_add(1);
            *tabu_pos = idx + 1;
        }
    }

    fn critical_block_move_local_search_ex(
        pre: &Pre,
        challenge: &Challenge,
        base_sol: &Solution,
        max_iters: usize,
        top_cands: usize,
        perturb_cycles: usize,
    ) -> Result<Option<(Solution, u32)>> {
        let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
        let mut buf = EvalBuf::new(ds.n);
        let mut crit = vec![false; ds.n];
        let mut cur_eval = match eval_disj(&ds, &mut buf) {
            Some(x) => x,
            None => return Ok(None),
        };
        let initial_mk = cur_eval.0;
        descent_phase(&mut ds, &mut buf, &mut crit, pre, &mut cur_eval, max_iters, top_cands);
        let Some((mk_after, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
        let mut global_best_mk = mk_after;
        let mut global_best_ds = ds.clone();
        let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15)
            ^ (initial_mk as u64).wrapping_shl(16)
            ^ (ds.n as u64);
        for _cycle in 0..perturb_cycles {
            ds = global_best_ds.clone();
            let Some((_, mk_node)) = eval_disj(&ds, &mut buf) else { break };
            crit.fill(false);
            let mut u = mk_node;
            while u != NONE_USIZE {
                crit[u] = true;
                u = buf.best_pred[u];
            }
            let mut blocks: Vec<(usize, usize, usize)> = Vec::new();
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
                    let mut bend = i;
                    while bend + 1 < seq.len() {
                        let x = seq[bend];
                        let y = seq[bend + 1];
                        if !crit[y] {
                            break;
                        }
                        let end_x = buf.start[x].saturating_add(ds.node_pt[x]);
                        if buf.start[y] != end_x {
                            break;
                        }
                        bend += 1;
                    }
                    if bend > bstart {
                        blocks.push((m, bstart, bend));
                    }
                    i = bend + 1;
                }
            }
            if blocks.is_empty() {
                break;
            }
            for _ in 0..2 {
                pseed ^= pseed.wrapping_shl(13);
                pseed ^= pseed.wrapping_shr(7);
                pseed ^= pseed.wrapping_shl(17);
                let bidx = (pseed as usize) % blocks.len();
                let (m, bstart, bend) = blocks[bidx];
                let block_len = bend - bstart;
                if block_len == 0 {
                    continue;
                }
                pseed ^= pseed.wrapping_shl(13);
                pseed ^= pseed.wrapping_shr(7);
                pseed ^= pseed.wrapping_shl(17);
                let swap_pos = bstart + ((pseed as usize) % block_len);
                if swap_pos + 1 < ds.machine_seq[m].len() {
                    ds.machine_seq[m].swap(swap_pos, swap_pos + 1);
                }
            }
            match eval_disj(&ds, &mut buf) {
                Some(x) => cur_eval = x,
                None => continue,
            }
            descent_phase(&mut ds, &mut buf, &mut crit, pre, &mut cur_eval, max_iters, top_cands);
            if let Some((mk_now, _)) = eval_disj(&ds, &mut buf) {
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
        let Some((mk_final, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
        let sol = disj_to_solution(pre, &ds, &buf.start)?;
        Ok(Some((sol, mk_final)))
    }

    fn descent_phase(
        ds: &mut DisjSchedule,
        buf: &mut EvalBuf,
        crit: &mut Vec<bool>,
        _pre: &Pre,
        cur_eval: &mut (u32, usize),
        max_iters: usize,
        top_cands: usize,
    ) -> bool {
        let mut cur_mk = cur_eval.0;
        let mut best_seen = cur_mk;
        let mut improved = false;

        let hlen = max_iters.max(1);
        let mut delta_hist: Vec<u32> = vec![0u32; hlen];
        let mut dhpos: usize = 0;
        let mut dhfill: usize = 0;

        let tenure = (3usize + max_iters.min(4) + if ds.num_machines >= 8 { 1 } else { 0 })
            .max(3)
            .min(8);
        let tabu_cap = (tenure * 6).max(12);
        let mut tabu_keys: Vec<u64> = vec![0u64; tabu_cap];
        let mut tabu_until: Vec<usize> = vec![0usize; tabu_cap];
        let mut tabu_pos: usize = 0;
        let mut arc_old: Vec<u64> = Vec::new();
        let mut arc_new: Vec<u64> = Vec::new();
        let mut tmp_seq: Vec<usize> = Vec::new();

        for iter_ix in 0..max_iters {
            crit.fill(false);
            let mut u = cur_eval.1;
            while u != NONE_USIZE {
                crit[u] = true;
                u = buf.best_pred[u];
            }

            let prescreen_k = top_cands.min(48).max(8);
            let n = ds.n;

            let mut chain: Vec<usize> = Vec::new();
            let mut z = cur_eval.1;
            while z != NONE_USIZE {
                chain.push(z);
                z = buf.best_pred[z];
            }
            chain.reverse();

            let mut crit_tail = vec![0u32; n];
            let mut crit_rank = vec![NONE_USIZE; n];
            let mut carry = 0u32;
            for idx in (0..chain.len()).rev() {
                let v = chain[idx];
                carry = ds.node_pt[v].saturating_add(carry);
                crit_tail[v] = carry;
                crit_rank[v] = idx;
            }

            let mut machine_pos = vec![NONE_USIZE; n];
            for m in 0..ds.num_machines {
                for (pos, &v) in ds.machine_seq[m].iter().enumerate() {
                    machine_pos[v] = pos;
                }
            }

            let op_surrogate = |u: usize| -> u64 {
                let st = buf.start[u];
                let ptu = ds.node_pt[u];
                let end_u = st.saturating_add(ptu);

                let js = ds.job_succ[u];
                let ms = buf.machine_succ[u];
                let job_prev = if ds.node_op[u] > 0 { u - 1 } else { NONE_USIZE };

                let m = ds.node_machine[u];
                let pos = machine_pos[u];
                let mach_prev = if pos > 0 && pos != NONE_USIZE {
                    ds.machine_seq[m][pos - 1]
                } else {
                    NONE_USIZE
                };
                let mach_next = if pos != NONE_USIZE && pos + 1 < ds.machine_seq[m].len() {
                    ds.machine_seq[m][pos + 1]
                } else {
                    NONE_USIZE
                };

                let gap_before = |p: usize| -> u32 {
                    if p == NONE_USIZE {
                        0
                    } else {
                        st.saturating_sub(buf.start[p].saturating_add(ds.node_pt[p]))
                    }
                };
                let gap_after = |v: usize| -> u32 {
                    if v == NONE_USIZE {
                        0
                    } else {
                        buf.start[v].saturating_sub(end_u)
                    }
                };
                let tight_before = |p: usize| -> u64 {
                    if p == NONE_USIZE {
                        0
                    } else {
                        ds.node_pt[p].saturating_sub(gap_before(p).min(ds.node_pt[p])) as u64
                    }
                };
                let tight_after = |v: usize| -> u64 {
                    if v == NONE_USIZE {
                        0
                    } else {
                        ds.node_pt[v].saturating_sub(gap_after(v).min(ds.node_pt[v])) as u64
                    }
                };

                let down_job = if js != NONE_USIZE {
                    ds.node_pt[js].saturating_add(gap_after(js))
                } else {
                    0
                };
                let down_mach = if ms != NONE_USIZE {
                    ds.node_pt[ms].saturating_add(gap_after(ms))
                } else {
                    0
                };
                let head_tail = if crit_tail[u] > 0 {
                    crit_tail[u]
                } else {
                    ptu.saturating_add(down_job.max(down_mach))
                };

                let near_chain =
                    (job_prev != NONE_USIZE && crit[job_prev])
                        || (js != NONE_USIZE && crit[js])
                        || (mach_prev != NONE_USIZE && crit[mach_prev])
                        || (mach_next != NONE_USIZE && crit[mach_next]);

                (end_u as u64) * 7
                    + (ptu as u64) * 9
                    + (head_tail as u64) * 11
                    + tight_before(job_prev) * 3
                    + tight_before(mach_prev) * 5
                    + tight_after(js) * 4
                    + tight_after(ms) * 6
                    + if crit[u] {
                        (ptu as u64) * 3 + (head_tail as u64) * 2
                    } else {
                        0
                    }
                    + if near_chain { (ptu as u64) * 2 } else { 0 }
            };

            let pair_surrogate = |u: usize, v: usize| -> u32 {
                let mut s = op_surrogate(u).saturating_add(op_surrogate(v));
                if crit[u] && crit[v] {
                    s = s.saturating_add((ds.node_pt[u] as u64 + ds.node_pt[v] as u64) * 6);
                }
                let ru = crit_rank[u];
                let rv = crit_rank[v];
                if ru != NONE_USIZE && rv != NONE_USIZE {
                    let dist = ru.max(rv) - ru.min(rv);
                    s = s.saturating_add((chain.len().saturating_sub(dist) as u64) * 3);
                }
                s.min(u32::MAX as u64) as u32
            };

            let mut cands: Vec<MoveCand> = Vec::with_capacity(prescreen_k.min(64));
            for m in 0..ds.num_machines {
                let seq = &ds.machine_seq[m];
                if seq.len() <= 1 {
                    continue;
                }
                let mut i = 0usize;
                while i < seq.len() {
                    let a = seq[i];
                    if !crit[a] {
                        i += 1;
                        continue;
                    }
                    let bstart = i;
                    let mut bend = i;
                    while bend + 1 < seq.len() {
                        let x = seq[bend];
                        let y = seq[bend + 1];
                        if !crit[y] {
                            break;
                        }
                        let end_x = buf.start[x].saturating_add(ds.node_pt[x]);
                        if buf.start[y] != end_x {
                            break;
                        }
                        bend += 1;
                    }

                    if bend > bstart {
                        let max_shift = bend - bstart;
                        for &sh in &[1usize, 2, max_shift] {
                            if sh == 0 || sh > max_shift {
                                continue;
                            }
                            let from = bstart;
                            let to_after = bstart + sh;
                            if from < seq.len() && to_after <= seq.len() {
                                let tgt_idx = (bstart + sh).min(seq.len() - 1);
                                let sc = pair_surrogate(seq[from], seq[tgt_idx]);
                                push_top_k_move_fs(
                                    &mut cands,
                                    MoveCand {
                                        kind: 0,
                                        m_from: m,
                                        from,
                                        m_to: m,
                                        to: to_after,
                                        new_pt: 0,
                                        score: sc,
                                    },
                                    prescreen_k,
                                );
                            }
                            let from2 = bend;
                            let to_after2 = bend - sh;
                            let tgt_idx2 = (bend - sh).min(seq.len().saturating_sub(1));
                            let sc2 = pair_surrogate(seq[from2], seq[tgt_idx2]);
                            push_top_k_move_fs(
                                &mut cands,
                                MoveCand {
                                    kind: 0,
                                    m_from: m,
                                    from: from2,
                                    m_to: m,
                                    to: to_after2,
                                    new_pt: 0,
                                    score: sc2,
                                },
                                prescreen_k,
                            );
                        }

                        if bstart > 0 {
                            let sc = pair_surrogate(seq[bstart - 1], seq[bstart]);
                            push_top_k_move_fs(
                                &mut cands,
                                MoveCand {
                                    kind: 2,
                                    m_from: m,
                                    from: bstart - 1,
                                    m_to: m,
                                    to: 0,
                                    new_pt: 0,
                                    score: sc,
                                },
                                prescreen_k,
                            );
                        }
                        if bend + 1 < seq.len() {
                            let sc = pair_surrogate(seq[bend], seq[bend + 1]);
                            push_top_k_move_fs(
                                &mut cands,
                                MoveCand {
                                    kind: 2,
                                    m_from: m,
                                    from: bend,
                                    m_to: m,
                                    to: 0,
                                    new_pt: 0,
                                    score: sc,
                                },
                                prescreen_k,
                            );
                        }

                        let mid = (bstart + bend) / 2;
                        let mut push_swap = |i1: usize, i2: usize| {
                            if i1 == i2 {
                                return;
                            }
                            let (lo, hi) = if i1 < i2 { (i1, i2) } else { (i2, i1) };
                            if lo + 1 >= hi {
                                return;
                            }
                            let sc = pair_surrogate(seq[lo], seq[hi]);
                            push_top_k_move_fs(
                                &mut cands,
                                MoveCand {
                                    kind: 3,
                                    m_from: m,
                                    from: lo,
                                    m_to: m,
                                    to: hi,
                                    new_pt: 0,
                                    score: sc,
                                },
                                prescreen_k,
                            );
                        };

                        push_swap(bstart, bend);
                        push_swap(bstart, mid);
                        push_swap(mid, bend);
                    }

                    i = bend + 1;
                }
            }

            if cands.is_empty() {
                break;
            }

            let mut best_cand: Option<MoveCand> = None;
            let mut best_mk = u32::MAX;

            for cand in &cands {
                let cand_tabu = move_hits_recent_arc(
                    ds,
                    cand,
                    &tabu_keys,
                    &tabu_until,
                    iter_ix,
                    &mut arc_old,
                    &mut arc_new,
                    &mut tmp_seq,
                );

                if cand.kind == 0 {
                    let m = cand.m_from;
                    if m >= ds.num_machines || cand.from >= ds.machine_seq[m].len() {
                        continue;
                    }
                    let new_idx = apply_insert_fs(&mut ds.machine_seq[m], cand.from, cand.to);
                    if let Some((mk2, _)) = eval_disj(ds, buf) {
                        if mk2 < best_mk && (!cand_tabu || mk2 < best_seen) {
                            best_mk = mk2;
                            best_cand = Some(*cand);
                        }
                    }
                    let _ = apply_insert_fs(&mut ds.machine_seq[m], new_idx, cand.from);
                } else if cand.kind == 2 {
                    let m = cand.m_from;
                    if m >= ds.num_machines || cand.from + 1 >= ds.machine_seq[m].len() {
                        continue;
                    }
                    ds.machine_seq[m].swap(cand.from, cand.from + 1);
                    if let Some((mk2, _)) = eval_disj(ds, buf) {
                        if mk2 < best_mk && (!cand_tabu || mk2 < best_seen) {
                            best_mk = mk2;
                            best_cand = Some(*cand);
                        }
                    }
                    ds.machine_seq[m].swap(cand.from, cand.from + 1);
                } else if cand.kind == 3 {
                    let m = cand.m_from;
                    if m >= ds.num_machines {
                        continue;
                    }
                    let len = ds.machine_seq[m].len();
                    if cand.from >= len || cand.to >= len || cand.from + 1 >= cand.to {
                        continue;
                    }
                    ds.machine_seq[m].swap(cand.from, cand.to);
                    if let Some((mk2, _)) = eval_disj(ds, buf) {
                        if mk2 < best_mk && (!cand_tabu || mk2 < best_seen) {
                            best_mk = mk2;
                            best_cand = Some(*cand);
                        }
                    }
                    ds.machine_seq[m].swap(cand.from, cand.to);
                }
            }

            let Some(bc) = best_cand else { break };

            let prev_mk = cur_mk;

            let bc_mk = best_mk;
            let d = if bc_mk > best_seen {
                bc_mk - best_seen
            } else {
                best_seen - bc_mk
            };
            delta_hist[dhpos] = d;
            dhpos += 1;
            if dhpos >= hlen {
                dhpos = 0;
            }
            if dhfill < hlen {
                dhfill += 1;
            }

            let band = if dhfill == 0 {
                0
            } else {
                let mut tmp: Vec<u32> = delta_hist[..dhfill].to_vec();
                let mid = tmp.len() >> 1;
                let (_, med, _) = tmp.select_nth_unstable(mid);
                *med
            };
            let rrt_limit = best_seen.saturating_add(band);

            let mut accepted = false;
            if bc.kind == 0 {
                let m = bc.m_from;
                let before_seq = ds.machine_seq[m].clone();
                let new_idx = apply_insert_fs(&mut ds.machine_seq[m], bc.from, bc.to);
                if let Some(next_eval) = eval_disj(ds, buf) {
                    let next_mk = next_eval.0;
                    if next_mk < prev_mk || next_mk <= rrt_limit {
                        *cur_eval = next_eval;
                        cur_mk = next_mk;
                        if next_mk < prev_mk {
                            improved = true;
                        }
                        if next_mk < best_seen {
                            best_seen = next_mk;
                        }
                        protect_recent_created_arcs(
                            &before_seq,
                            &ds.machine_seq[m],
                            m,
                            tenure,
                            iter_ix,
                            &mut tabu_keys,
                            &mut tabu_until,
                            &mut tabu_pos,
                            &mut arc_old,
                            &mut arc_new,
                        );
                        accepted = true;
                    } else {
                        let _ = apply_insert_fs(&mut ds.machine_seq[m], new_idx, bc.from);
                    }
                } else {
                    let _ = apply_insert_fs(&mut ds.machine_seq[m], new_idx, bc.from);
                }
            } else if bc.kind == 2 {
                let m = bc.m_from;
                if m < ds.num_machines && bc.from + 1 < ds.machine_seq[m].len() {
                    let before_seq = ds.machine_seq[m].clone();
                    ds.machine_seq[m].swap(bc.from, bc.from + 1);
                    if let Some(next_eval) = eval_disj(ds, buf) {
                        let next_mk = next_eval.0;
                        if next_mk < prev_mk || next_mk <= rrt_limit {
                            *cur_eval = next_eval;
                            cur_mk = next_mk;
                            if next_mk < prev_mk {
                                improved = true;
                            }
                            if next_mk < best_seen {
                                best_seen = next_mk;
                            }
                            protect_recent_created_arcs(
                                &before_seq,
                                &ds.machine_seq[m],
                                m,
                                tenure,
                                iter_ix,
                                &mut tabu_keys,
                                &mut tabu_until,
                                &mut tabu_pos,
                                &mut arc_old,
                                &mut arc_new,
                            );
                            accepted = true;
                        } else {
                            ds.machine_seq[m].swap(bc.from, bc.from + 1);
                        }
                    } else {
                        ds.machine_seq[m].swap(bc.from, bc.from + 1);
                    }
                }
            } else if bc.kind == 3 {
                let m = bc.m_from;
                if m < ds.num_machines {
                    let len = ds.machine_seq[m].len();
                    if bc.from < len && bc.to < len && bc.from + 1 < bc.to {
                        let before_seq = ds.machine_seq[m].clone();
                        ds.machine_seq[m].swap(bc.from, bc.to);
                        if let Some(next_eval) = eval_disj(ds, buf) {
                            let next_mk = next_eval.0;
                            if next_mk < prev_mk || next_mk <= rrt_limit {
                                *cur_eval = next_eval;
                                cur_mk = next_mk;
                                if next_mk < prev_mk {
                                    improved = true;
                                }
                                if next_mk < best_seen {
                                    best_seen = next_mk;
                                }
                                protect_recent_created_arcs(
                                    &before_seq,
                                    &ds.machine_seq[m],
                                    m,
                                    tenure,
                                    iter_ix,
                                    &mut tabu_keys,
                                    &mut tabu_until,
                                    &mut tabu_pos,
                                    &mut arc_old,
                                    &mut arc_new,
                                );
                                accepted = true;
                            } else {
                                ds.machine_seq[m].swap(bc.from, bc.to);
                            }
                        } else {
                            ds.machine_seq[m].swap(bc.from, bc.to);
                        }
                    }
                }
            }

            if !accepted {
                break;
            }
        }

        improved
    }

    #[inline]
    fn apply_insert_fs(seq: &mut Vec<usize>, from: usize, to_after_removal: usize) -> usize {
        if seq.is_empty() || from >= seq.len() {
            return from.min(seq.len().saturating_sub(1));
        }
        let x = seq.remove(from);
        let t = to_after_removal.min(seq.len());
        seq.insert(t, x);
        t
    }

    #[inline]
    fn push_top_k_move_fs(top: &mut Vec<MoveCand>, c: MoveCand, k: usize) {
        if k == 0 {
            return;
        }
        let mut pos = top.len();
        while pos > 0 && top[pos - 1].score < c.score {
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

    fn run_simple_greedy_baseline(challenge: &Challenge) -> Result<(Solution, u32)> {
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
                    .map(|op| op.values().sum::<u32>() as f64 / op.len().max(1) as f64)
                    .sum()
            })
            .collect();
        run_greedy_rule_fs(challenge, &job_products, &job_ops_len, &job_total_work)
    }

    fn run_greedy_rule_fs(
        challenge: &Challenge,
        job_products: &[usize],
        job_ops_len: &[usize],
        job_total_work: &[f64],
    ) -> Result<(Solution, u32)> {
        let num_jobs = challenge.num_jobs;
        let num_machines = challenge.num_machines;
        let mut job_next_op = vec![0usize; num_jobs];
        let mut job_ready = vec![0u32; num_jobs];
        let mut machine_avail = vec![0u32; num_machines];
        let mut job_schedule: Vec<Vec<(usize, u32)>> =
            job_ops_len.iter().map(|&len| Vec::with_capacity(len)).collect();
        let mut job_work_left = job_total_work.to_vec();
        let mut remaining = job_ops_len.iter().sum::<usize>();
        let mut time = 0u32;
        while remaining > 0 {
            let mut did_work = false;
            for m in 0..num_machines {
                if machine_avail[m] > time {
                    continue;
                }
                let mut best_job: Option<usize> = None;
                let mut best_priority = f64::NEG_INFINITY;
                for j in 0..num_jobs {
                    if job_next_op[j] >= job_ops_len[j] || job_ready[j] > time {
                        continue;
                    }
                    let product = job_products[j];
                    let op_idx = job_next_op[j];
                    let op_times = &challenge.product_processing_times[product][op_idx];
                    let pt = match op_times.get(&m) {
                        Some(&v) => v,
                        None => continue,
                    };
                    let earliest = op_times
                        .iter()
                        .map(|(&mm, &ppt)| time.max(machine_avail[mm]) + ppt)
                        .min()
                        .unwrap_or(u32::MAX);
                    if time + pt != earliest {
                        continue;
                    }
                    let priority = job_work_left[j];
                    if best_job.is_none() || priority > best_priority {
                        best_priority = priority;
                        best_job = Some(j);
                    }
                }
                if let Some(j) = best_job {
                    let product = job_products[j];
                    let op_idx = job_next_op[j];
                    let op_times = &challenge.product_processing_times[product][op_idx];
                    let pt = op_times[&m];
                    let avg_pt = op_times.values().sum::<u32>() as f64 / op_times.len().max(1) as f64;
                    let st = time.max(machine_avail[m]);
                    let end = st + pt;
                    job_schedule[j].push((m, st));
                    job_next_op[j] += 1;
                    job_ready[j] = end;
                    machine_avail[m] = end;
                    job_work_left[j] -= avg_pt;
                    if job_work_left[j] < 0.0 {
                        job_work_left[j] = 0.0;
                    }
                    remaining -= 1;
                    did_work = true;
                }
            }
            if remaining == 0 {
                break;
            }
            if !did_work {
                let mut next = u32::MAX;
                for &t in &machine_avail {
                    if t > time && t < next {
                        next = t;
                    }
                }
                for j in 0..num_jobs {
                    if job_next_op[j] < job_ops_len[j] && job_ready[j] > time && job_ready[j] < next {
                        next = job_ready[j];
                    }
                }
                if next == u32::MAX {
                    return Err(anyhow!("Greedy stuck"));
                }
                time = next;
            }
        }
        let mk = job_ready.iter().copied().max().unwrap_or(0);
        Ok((Solution { job_schedule }, mk))
    }

    fn johnson_order_from_ab(a: &[u32], b: &[u32]) -> Vec<usize> {
        let n = a.len().min(b.len());
        let mut front: Vec<(u32, usize)> = Vec::with_capacity(n);
        let mut back: Vec<(u32, usize)> = Vec::with_capacity(n);
        for j in 0..n {
            if a[j] <= b[j] {
                front.push((a[j], j));
            } else {
                back.push((b[j], j));
            }
        }
        front.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
        back.sort_unstable_by(|x, y| y.0.cmp(&x.0).then_with(|| x.1.cmp(&y.1)));
        let mut ord = Vec::with_capacity(n);
        for &(_, j) in &front {
            ord.push(j);
        }
        for &(_, j) in &back {
            ord.push(j);
        }
        ord
    }

    fn palmer_order(pt: &[Vec<u32>]) -> Vec<usize> {
        let n = pt.len();
        let m = pt.first().map(|r| r.len()).unwrap_or(0);
        let mut jobs: Vec<(i64, usize)> = Vec::with_capacity(n);
        if m == 0 {
            return (0..n).collect();
        }
        let mm = m as i64;
        for j in 0..n {
            let row = &pt[j];
            let mut s: i64 = 0;
            for k in 0..m {
                let w = mm - 2 * (k as i64) - 1;
                s += w * (row[k] as i64);
            }
            jobs.push((s, j));
        }
        jobs.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        jobs.into_iter().map(|x| x.1).collect()
    }

    fn cds_orders(pt: &[Vec<u32>]) -> Vec<Vec<usize>> {
        let n = pt.len();
        if n == 0 {
            return vec![];
        }
        let m = pt[0].len();
        if m <= 1 {
            return vec![(0..n).collect()];
        }
        let mut totals = vec![0u32; n];
        let mut prefix = vec![vec![0u32; m + 1]; n];
        for j in 0..n {
            let row = &pt[j];
            let mut s = 0u32;
            prefix[j][0] = 0;
            for k in 0..m {
                s = s.saturating_add(row[k]);
                prefix[j][k + 1] = s;
            }
            totals[j] = s;
        }
        let mut res: Vec<Vec<usize>> = Vec::with_capacity(m - 1);
        let mut a = vec![0u32; n];
        let mut b = vec![0u32; n];
        for k in 1..m {
            for j in 0..n {
                a[j] = prefix[j][k];
                b[j] = totals[j].saturating_sub(prefix[j][k]);
            }
            res.push(johnson_order_from_ab(&a, &b));
        }
        res
    }

    fn route_is_unique(route: &[usize], num_machines: usize) -> bool {
        if route.is_empty() {
            return false;
        }
        let mut seen = vec![false; num_machines.max(1)];
        for &m in route {
            if m >= seen.len() || seen[m] {
                return false;
            }
            seen[m] = true;
        }
        true
    }

    #[derive(Default, Clone)]
    struct TaillardInsBuf {
        f: Vec<u32>,
        b: Vec<u32>,
        e: Vec<u32>,
        comp: Vec<u32>,
    }
    impl TaillardInsBuf {
        fn ensure(&mut self, len: usize, m: usize) {
            let need = (len + 1) * m;
            if self.f.len() < need {
                self.f.resize(need, 0);
            }
            if self.b.len() < need {
                self.b.resize(need, 0);
            }
            if self.e.len() < m {
                self.e.resize(m, 0);
            }
            if self.comp.len() < m {
                self.comp.resize(m, 0);
            }
        }
    }

    thread_local! {
        static TL_TAILLARD: RefCell<TaillardInsBuf> = RefCell::new(TaillardInsBuf::default());
    }

    fn taillard_best_insert_pos(
        seq: &[usize],
        job: usize,
        pt: &[Vec<u32>],
        m: usize,
        buf: &mut TaillardInsBuf,
    ) -> (usize, u32) {
        let l = seq.len();
        if m == 0 {
            return (0, 0);
        }
        buf.ensure(l, m);
        let f = &mut buf.f;
        let b = &mut buf.b;
        let e = &mut buf.e;
        for k in 0..m {
            f[k] = 0;
        }
        for t in 1..=l {
            let jj = seq[t - 1];
            let row = &pt[jj];
            let base = t * m;
            let prev = (t - 1) * m;
            f[base] = f[prev].saturating_add(row[0]);
            for k in 1..m {
                f[base + k] = f[base + k - 1].max(f[prev + k]).saturating_add(row[k]);
            }
        }
        let base_l = l * m;
        for k in 0..m {
            b[base_l + k] = 0;
        }
        for t in (0..l).rev() {
            let jj = seq[t];
            let row = &pt[jj];
            let base = t * m;
            let next = (t + 1) * m;
            b[base + (m - 1)] = b[next + (m - 1)].saturating_add(row[m - 1]);
            if m >= 2 {
                for kk in 0..(m - 1) {
                    let k = (m - 2) - kk;
                    b[base + k] = b[base + k + 1].max(b[next + k]).saturating_add(row[k]);
                }
            }
        }
        let prow = &pt[job];
        let mut best_pos = 0usize;
        let mut best_mk = u32::MAX;
        for pos in 0..=l {
            let fb = pos * m;
            e[0] = f[fb].saturating_add(prow[0]);
            for k in 1..m {
                e[k] = e[k - 1].max(f[fb + k]).saturating_add(prow[k]);
            }
            let mut mk = 0u32;
            for k in 0..m {
                mk = mk.max(e[k].saturating_add(b[fb + k]));
            }
            if mk < best_mk {
                best_mk = mk;
                best_pos = pos;
            }
        }
        (best_pos, best_mk)
    }

    fn improve_perm_seq_taillard(seq: &mut Vec<usize>, pt: &[Vec<u32>], rounds: usize, buf: &mut TaillardInsBuf) {
        let m = pt.first().map(|r| r.len()).unwrap_or(0);
        if seq.len() <= 2 || m == 0 {
            return;
        }
        buf.ensure(seq.len(), m);
        let mut cur_mk = flow_makespan(seq, pt, &mut buf.comp[..m]);
        for _ in 0..rounds {
            let mut improved_any = false;
            for i0 in 0..seq.len() {
                let job = seq.remove(i0);
                let (pos, mk) = taillard_best_insert_pos(seq, job, pt, m, buf);
                seq.insert(pos, job);
                if mk < cur_mk {
                    cur_mk = mk;
                    improved_any = true;
                }
            }
            if !improved_any {
                break;
            }
        }
    }

    fn neh_build_seq(order: &[usize], route: &[usize], pt: &[Vec<u32>], num_machines: usize) -> Vec<usize> {
        let unique = route_is_unique(route, num_machines);
        if unique {
            let m = route.len();
            if m == 0 {
                return vec![];
            }
            return TL_TAILLARD.with(|cell| {
                let mut buf = cell.borrow_mut();
                let mut seq: Vec<usize> = Vec::with_capacity(order.len());
                for &j in order {
                    if seq.is_empty() {
                        seq.push(j);
                        continue;
                    }
                    let (pos, _mk) = taillard_best_insert_pos(&seq, j, pt, m, &mut buf);
                    seq.insert(pos, j);
                }
                seq
            });
        }
        let mut seq: Vec<usize> = Vec::with_capacity(order.len());
        let mut tmp: Vec<usize> = Vec::with_capacity(order.len());
        let mut mready = vec![0u32; num_machines];
        for &j in order {
            if seq.is_empty() {
                seq.push(j);
                continue;
            }
            let mut best_mk = u32::MAX;
            let mut best_pos = 0usize;
            for pos in 0..=seq.len() {
                tmp.clear();
                tmp.extend_from_slice(&seq[..pos]);
                tmp.push(j);
                tmp.extend_from_slice(&seq[pos..]);
                let mk = reentrant_makespan(&tmp, route, pt, &mut mready);
                if mk < best_mk {
                    best_mk = mk;
                    best_pos = pos;
                }
            }
            seq.insert(best_pos, j);
        }
        seq
    }

    fn fs_improve_reentrant_seq(seq: &mut Vec<usize>, route: &[usize], pt: &[Vec<u32>], num_machines: usize) {
        if seq.len() <= 2 || route.is_empty() {
            return;
        }
        if route_is_unique(route, num_machines) {
            TL_TAILLARD.with(|cell| {
                let mut buf = cell.borrow_mut();
                improve_perm_seq_taillard(seq, pt, 8, &mut buf);
            });
            return;
        }
        let mut mready = vec![0u32; num_machines];
        let mut cur_mk = reentrant_makespan(seq, route, pt, &mut mready);
        for _ in 0..8usize {
            let mut improved_any = false;
            for i0 in 0..seq.len() {
                let j = seq.remove(i0);
                let mut best_mk = u32::MAX;
                let mut best_pos = 0usize;
                for pos in 0..=seq.len() {
                    seq.insert(pos, j);
                    let mk = reentrant_makespan(seq, route, pt, &mut mready);
                    if mk < best_mk {
                        best_mk = mk;
                        best_pos = pos;
                    }
                    seq.remove(pos);
                }
                seq.insert(best_pos, j);
                if best_mk < cur_mk {
                    cur_mk = best_mk;
                    improved_any = true;
                }
            }
            if !improved_any {
                break;
            }
        }
    }

    fn build_perm_solution_from_seq(
        seq: &[usize],
        route: &[usize],
        pt: &[Vec<u32>],
        num_jobs: usize,
        num_machines: usize,
    ) -> Solution {
        let ops = route.len();
        let mut job_schedule: Vec<Vec<(usize, u32)>> = vec![Vec::with_capacity(ops); num_jobs];
        let mut machine_ready = vec![0u32; num_machines];
        for &j in seq {
            if j >= num_jobs {
                continue;
            }
            let row = &pt[j];
            let mut prev_end = 0u32;
            for (op_idx, &m) in route.iter().enumerate() {
                if op_idx >= row.len() || m >= num_machines {
                    break;
                }
                let p = row[op_idx];
                let st = prev_end.max(machine_ready[m]);
                job_schedule[j].push((m, st));
                let end = st.saturating_add(p);
                machine_ready[m] = end;
                prev_end = end;
            }
        }
        Solution { job_schedule }
    }

    fn order_from_solution_first_op_start(sol: &Solution, num_jobs: usize) -> Vec<usize> {
        let mut v: Vec<(u32, usize)> = Vec::with_capacity(num_jobs);
        for j in 0..num_jobs {
            if let Some(t) = sol
                .job_schedule
                .get(j)
                .and_then(|ops| ops.first())
                .map(|x| x.1)
            {
                v.push((t, j));
            }
        }
        v.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let mut seen = vec![false; num_jobs];
        let mut ord: Vec<usize> = Vec::with_capacity(num_jobs);
        for &(_, j) in &v {
            if j < num_jobs && !seen[j] {
                seen[j] = true;
                ord.push(j);
            }
        }
        for j in 0..num_jobs {
            if !seen[j] {
                ord.push(j);
            }
        }
        ord
    }

    fn neh_best_sequence(pre: &Pre, num_jobs: usize, num_machines: usize) -> Result<Vec<usize>> {
        let route = pre.flow_route.as_ref().ok_or_else(|| anyhow!("No flow route"))?;
        let pt = pre.flow_pt_by_job.as_ref().ok_or_else(|| anyhow!("No flow pt"))?;
        let ops = route.len();
        if ops == 0 || pt.len() != num_jobs {
            return Err(anyhow!("Invalid flow data"));
        }
        let mut candidates: Vec<Vec<usize>> = Vec::new();
        {
            let mut jobs: Vec<usize> = (0..num_jobs).collect();
            jobs.sort_unstable_by(|&a, &b| {
                let sa: u32 = pt[a].iter().copied().sum();
                let sb: u32 = pt[b].iter().copied().sum();
                sb.cmp(&sa).then_with(|| a.cmp(&b))
            });
            candidates.push(jobs);
        }
        candidates.push(palmer_order(pt));
        for o in cds_orders(pt) {
            if o.len() == num_jobs {
                candidates.push(o);
            }
        }
        let unique = route_is_unique(route, num_machines);
        let mut best_seq: Vec<usize> = (0..num_jobs).collect();
        let mut best_mk: u32 = u32::MAX;
        if unique {
            TL_TAILLARD.with(|cell| {
                let mut buf = cell.borrow_mut();
                let m = ops;
                for ord in candidates.iter() {
                    if ord.len() != num_jobs {
                        continue;
                    }
                    let mut seq = neh_build_seq(ord, route, pt, num_machines);
                    improve_perm_seq_taillard(&mut seq, pt, 8, &mut buf);
                    buf.ensure(seq.len(), m);
                    let mk = flow_makespan(&seq, pt, &mut buf.comp[..m]);
                    if mk < best_mk {
                        best_mk = mk;
                        best_seq = seq;
                    }
                }
            });
            return Ok(best_seq);
        }
        let mut mready = vec![0u32; num_machines];
        for ord in candidates.iter() {
            if ord.len() != num_jobs {
                continue;
            }
            let mut seq = neh_build_seq(ord, route, pt, num_machines);
            fs_improve_reentrant_seq(&mut seq, route, pt, num_machines);
            let mk = reentrant_makespan(&seq, route, pt, &mut mready);
            if mk < best_mk {
                best_mk = mk;
                best_seq = seq;
            }
        }
        Ok(best_seq)
    }

    fn iterated_greedy_search(init: &[usize], pt: &[Vec<u32>], iters: usize, d: usize, rng: &mut SmallRng) -> Vec<usize> {
        let n = init.len();
        if n <= 2 {
            return init.to_vec();
        }
        let m = pt.first().map(|r| r.len()).unwrap_or(0);
        if m == 0 {
            return init.to_vec();
        }
        let mut buf = TaillardInsBuf::default();
        buf.ensure(n, m);
        let mut cur = init.to_vec();
        let mut best = cur.clone();
        let mut cur_mk = flow_makespan(&cur, pt, &mut buf.comp[..m]);
        let mut best_mk = cur_mk;
        let mut temp = (cur_mk as f64) * 0.10 + 1.0;
        let dd = d.clamp(2, n.saturating_sub(1));
        let its = iters.max(1);
        let mut idxs: Vec<usize> = Vec::with_capacity(dd);
        let mut removed: Vec<usize> = Vec::with_capacity(dd);
        for _ in 0..its {
            idxs.clear();
            while idxs.len() < dd {
                let x = rng.gen_range(0..n);
                if !idxs.iter().any(|&y| y == x) {
                    idxs.push(x);
                }
            }
            idxs.sort_unstable();
            removed.clear();
            let mut partial = cur.clone();
            for &ix in idxs.iter().rev() {
                if ix < partial.len() {
                    removed.push(partial.remove(ix));
                }
            }
            removed.shuffle(rng);
            for &j in &removed {
                let (pos, _mk) = taillard_best_insert_pos(&partial, j, pt, m, &mut buf);
                partial.insert(pos, j);
            }
            let mut cand = partial;
            let mut cand_mk = flow_makespan(&cand, pt, &mut buf.comp[..m]);
            if cand.len() >= 2 {
                for i in 0..(cand.len() - 1) {
                    cand.swap(i, i + 1);
                    let mk2 = flow_makespan(&cand, pt, &mut buf.comp[..m]);
                    if mk2 <= cand_mk {
                        cand_mk = mk2;
                    } else {
                        cand.swap(i, i + 1);
                    }
                }
            }
            if cand_mk < best_mk {
                best_mk = cand_mk;
                best = cand.clone();
            }
            if cand_mk <= cur_mk {
                cur = cand;
                cur_mk = cand_mk;
            } else {
                let delta = (cand_mk - cur_mk) as f64;
                let prob = (-delta / temp).exp();
                if rng.gen::<f64>() < prob {
                    cur = cand;
                    cur_mk = cand_mk;
                }
            }
            temp = (temp * 0.995).max(1.0);
        }
        best
    }

    fn strict_makespan(challenge: &Challenge, pre: &Pre, rank: &[usize]) -> Result<u32> {
        let route = pre.flow_route.as_ref().ok_or_else(|| anyhow!("flow_route missing"))?;
        let num_jobs = challenge.num_jobs;
        let num_machines = challenge.num_machines;
        let mut job_next_op = vec![0usize; num_jobs];
        let mut job_ready = vec![0u32; num_jobs];
        let mut machine_avail = vec![0u32; num_machines];
        let mut remaining_ops = pre.total_ops;
        let mut future: Vec<BinaryHeap<Reverse<(u32, usize, usize)>>> =
            (0..num_machines).map(|_| BinaryHeap::new()).collect();
        let mut avail: Vec<BinaryHeap<Reverse<(usize, usize)>>> =
            (0..num_machines).map(|_| BinaryHeap::new()).collect();
        for job in 0..num_jobs {
            if pre.job_ops_len[job] == 0 {
                continue;
            }
            let m = route[0];
            future[m].push(Reverse((0u32, rank[job], job)));
        }
        let mut next_time: Vec<Option<u32>> = vec![None; num_machines];
        let mut machine_events: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();
        let compute_next_time = |m: usize,
                                machine_avail: &Vec<u32>,
                                future: &Vec<BinaryHeap<Reverse<(u32, usize, usize)>>>,
                                avail: &Vec<BinaryHeap<Reverse<(usize, usize)>>>|
         -> Option<u32> {
            if !avail[m].is_empty() {
                return Some(machine_avail[m]);
            }
            if let Some(Reverse((release, _, _))) = future[m].peek().copied() {
                return Some(machine_avail[m].max(release));
            }
            None
        };
        for m in 0..num_machines {
            let t = compute_next_time(m, &machine_avail, &future, &avail);
            next_time[m] = t;
            if let Some(tt) = t {
                machine_events.push(Reverse((tt, m)));
            }
        }
        let mut makespan = 0u32;
        while remaining_ops > 0 {
            let Reverse((t, m)) = machine_events.pop().ok_or_else(|| anyhow!("stalled"))?;
            if next_time[m] != Some(t) || machine_avail[m] > t {
                continue;
            }
            while let Some(Reverse((release, _, job))) = future[m].peek().copied() {
                if release > t {
                    break;
                }
                future[m].pop();
                avail[m].push(Reverse((rank[job], job)));
            }
            let Some(Reverse((_, job))) = avail[m].pop() else {
                let nt = compute_next_time(m, &machine_avail, &future, &avail);
                next_time[m] = nt;
                if let Some(tt) = nt {
                    machine_events.push(Reverse((tt, m)));
                }
                continue;
            };
            let op_idx = job_next_op[job];
            if op_idx >= pre.job_ops_len[job] {
                return Err(anyhow!("job complete but popped"));
            }
            if route[op_idx] != m {
                return Err(anyhow!("route mismatch"));
            }
            let start = t.max(job_ready[job]).max(machine_avail[m]);
            if start != t {
                avail[m].push(Reverse((rank[job], job)));
                let nt = compute_next_time(m, &machine_avail, &future, &avail);
                next_time[m] = nt;
                if let Some(tt) = nt {
                    machine_events.push(Reverse((tt, m)));
                }
                continue;
            }
            let product = pre.job_products[job];
            let ptv = *challenge.product_processing_times[product][op_idx]
                .get(&m)
                .ok_or_else(|| anyhow!("missing pt"))?;
            let end = start.saturating_add(ptv);
            job_next_op[job] += 1;
            job_ready[job] = end;
            machine_avail[m] = end;
            remaining_ops -= 1;
            makespan = makespan.max(end);
            if job_next_op[job] < pre.job_ops_len[job] {
                let next_op = job_next_op[job];
                let m2 = route[next_op];
                future[m2].push(Reverse((end, rank[job], job)));
                let nt2 = compute_next_time(m2, &machine_avail, &future, &avail);
                next_time[m2] = nt2;
                if let Some(tt) = nt2 {
                    machine_events.push(Reverse((tt, m2)));
                }
            }
            let nt = compute_next_time(m, &machine_avail, &future, &avail);
            next_time[m] = nt;
            if let Some(tt) = nt {
                machine_events.push(Reverse((tt, m)));
            }
        }
        Ok(makespan)
    }

    fn strict_simulate(challenge: &Challenge, pre: &Pre, rank: &[usize]) -> Result<(Solution, u32)> {
        let route = pre.flow_route.as_ref().ok_or_else(|| anyhow!("flow_route missing"))?;
        let num_jobs = challenge.num_jobs;
        let num_machines = challenge.num_machines;
        let mut job_next_op = vec![0usize; num_jobs];
        let mut job_ready = vec![0u32; num_jobs];
        let mut machine_avail = vec![0u32; num_machines];
        let mut job_schedule: Vec<Vec<(usize, u32)>> = pre
            .job_ops_len
            .iter()
            .map(|&len| Vec::with_capacity(len))
            .collect();
        let mut remaining_ops = pre.total_ops;
        let mut future: Vec<BinaryHeap<Reverse<(u32, usize, usize)>>> =
            (0..num_machines).map(|_| BinaryHeap::new()).collect();
        let mut avail: Vec<BinaryHeap<Reverse<(usize, usize)>>> =
            (0..num_machines).map(|_| BinaryHeap::new()).collect();
        for job in 0..num_jobs {
            if pre.job_ops_len[job] == 0 {
                continue;
            }
            let m = route[0];
            future[m].push(Reverse((0u32, rank[job], job)));
        }
        let mut next_time: Vec<Option<u32>> = vec![None; num_machines];
        let mut machine_events: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();
        let compute_next_time = |m: usize,
                                machine_avail: &Vec<u32>,
                                future: &Vec<BinaryHeap<Reverse<(u32, usize, usize)>>>,
                                avail: &Vec<BinaryHeap<Reverse<(usize, usize)>>>|
         -> Option<u32> {
            if !avail[m].is_empty() {
                return Some(machine_avail[m]);
            }
            if let Some(Reverse((release, _, _))) = future[m].peek().copied() {
                return Some(machine_avail[m].max(release));
            }
            None
        };
        for m in 0..num_machines {
            let t = compute_next_time(m, &machine_avail, &future, &avail);
            next_time[m] = t;
            if let Some(tt) = t {
                machine_events.push(Reverse((tt, m)));
            }
        }
        let mut makespan = 0u32;
        while remaining_ops > 0 {
            let Reverse((t, m)) = machine_events.pop().ok_or_else(|| anyhow!("stalled"))?;
            if next_time[m] != Some(t) || machine_avail[m] > t {
                continue;
            }
            while let Some(Reverse((release, _, job))) = future[m].peek().copied() {
                if release > t {
                    break;
                }
                future[m].pop();
                avail[m].push(Reverse((rank[job], job)));
            }
            let Some(Reverse((_, job))) = avail[m].pop() else {
                let nt = compute_next_time(m, &machine_avail, &future, &avail);
                next_time[m] = nt;
                if let Some(tt) = nt {
                    machine_events.push(Reverse((tt, m)));
                }
                continue;
            };
            let op_idx = job_next_op[job];
            if op_idx >= pre.job_ops_len[job] {
                return Err(anyhow!("job complete"));
            }
            if route[op_idx] != m {
                return Err(anyhow!("route mismatch"));
            }
            let start = t.max(job_ready[job]).max(machine_avail[m]);
            if start != t {
                avail[m].push(Reverse((rank[job], job)));
                let nt = compute_next_time(m, &machine_avail, &future, &avail);
                next_time[m] = nt;
                if let Some(tt) = nt {
                    machine_events.push(Reverse((tt, m)));
                }
                continue;
            }
            let product = pre.job_products[job];
            let ptv = *challenge.product_processing_times[product][op_idx]
                .get(&m)
                .ok_or_else(|| anyhow!("missing pt"))?;
            let end = start.saturating_add(ptv);
            job_schedule[job].push((m, start));
            job_next_op[job] += 1;
            job_ready[job] = end;
            machine_avail[m] = end;
            remaining_ops -= 1;
            makespan = makespan.max(end);
            if job_next_op[job] < pre.job_ops_len[job] {
                let next_op = job_next_op[job];
                let m2 = route[next_op];
                future[m2].push(Reverse((end, rank[job], job)));
                let nt2 = compute_next_time(m2, &machine_avail, &future, &avail);
                next_time[m2] = nt2;
                if let Some(tt) = nt2 {
                    machine_events.push(Reverse((tt, m2)));
                }
            }
            let nt = compute_next_time(m, &machine_avail, &future, &avail);
            next_time[m] = nt;
            if let Some(tt) = nt {
                machine_events.push(Reverse((tt, m)));
            }
        }
        Ok((Solution { job_schedule }, makespan))
    }

    fn strict_best_by_order_search(challenge: &Challenge, pre: &Pre, passes: usize) -> Result<(Solution, u32)> {
        if pre.flow_route.is_none() || pre.flex_avg > 1.25 {
            return Err(anyhow!("not strict-like"));
        }
        let n = challenge.num_jobs;
        let pt_stage: Vec<Vec<u32>> = if let Some(pt) = pre.flow_pt_by_job.as_ref() {
            pt.clone()
        } else {
            let mut tmp = vec![vec![0u32; pre.max_ops.max(1)]; n];
            for j in 0..n {
                let p = pre.job_products[j];
                let len = pre.job_ops_len[j];
                for k in 0..len {
                    tmp[j][k] = pre.product_ops[p][k]
                        .machines
                        .first()
                        .map(|x| x.1)
                        .unwrap_or(0);
                }
                tmp[j].truncate(len);
            }
            tmp
        };
        let mut cand_orders: Vec<Vec<usize>> = Vec::new();
        {
            let mut lpt: Vec<usize> = (0..n).collect();
            lpt.sort_unstable_by(|&a, &b| {
                let sa: u32 = pt_stage[a].iter().copied().sum();
                let sb: u32 = pt_stage[b].iter().copied().sum();
                sb.cmp(&sa).then_with(|| a.cmp(&b))
            });
            cand_orders.push(lpt);
        }
        {
            let mut spt: Vec<usize> = (0..n).collect();
            spt.sort_unstable_by(|&a, &b| {
                let sa: u32 = pt_stage[a].iter().copied().sum();
                let sb: u32 = pt_stage[b].iter().copied().sum();
                sa.cmp(&sb).then_with(|| a.cmp(&b))
            });
            cand_orders.push(spt);
        }
        cand_orders.push(palmer_order(&pt_stage));
        for o in cds_orders(&pt_stage) {
            if o.len() == n {
                cand_orders.push(o);
            }
        }
        {
            let mut seed = challenge.seed;
            seed[0] ^= 0x3C;
            let mut rng = SmallRng::from_seed(seed);
            for _ in 0..100usize {
                let mut r: Vec<usize> = (0..n).collect();
                r.shuffle(&mut rng);
                cand_orders.push(r);
            }
        }
        let mut rank = vec![0usize; n];
        let mut best_mk = u32::MAX;
        let mut best_order: Vec<usize> = (0..n).collect();
        for ord in cand_orders.iter() {
            if ord.len() != n {
                continue;
            }
            for (pos, &j) in ord.iter().enumerate() {
                rank[j] = pos;
            }
            let mk = strict_makespan(challenge, pre, &rank)?;
            if mk < best_mk {
                best_mk = mk;
                best_order.clone_from(ord);
            }
        }
        let max_passes = passes.max(1).min(6);
        let mut cand_order: Vec<usize> = vec![0usize; n];
        for _ in 0..max_passes.min(2) {
            let mut improved = false;
            for i in 0..n {
                let job = best_order[i];
                let mut best_pos = i;
                let mut best_local_mk = best_mk;
                for pos in 0..n {
                    if pos == i {
                        continue;
                    }
                    if pos < i {
                        cand_order[..pos].copy_from_slice(&best_order[..pos]);
                        cand_order[pos] = job;
                        cand_order[pos + 1..=i].copy_from_slice(&best_order[pos..i]);
                        cand_order[i + 1..].copy_from_slice(&best_order[i + 1..]);
                    } else {
                        cand_order[..i].copy_from_slice(&best_order[..i]);
                        cand_order[i..pos].copy_from_slice(&best_order[i + 1..=pos]);
                        cand_order[pos] = job;
                        cand_order[pos + 1..].copy_from_slice(&best_order[pos + 1..]);
                    }
                    for (p, &jj) in cand_order.iter().enumerate() {
                        rank[jj] = p;
                    }
                    let mk = strict_makespan(challenge, pre, &rank)?;
                    if mk < best_local_mk {
                        best_local_mk = mk;
                        best_pos = pos;
                    }
                }
                if best_local_mk < best_mk {
                    best_mk = best_local_mk;
                    if best_pos < i {
                        best_order[best_pos..=i].rotate_right(1);
                    } else if best_pos > i {
                        best_order[i..=best_pos].rotate_left(1);
                    }
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        let mut order = best_order.clone();
        for (pos, &j) in order.iter().enumerate() {
            rank[j] = pos;
        }

        let seg_lens: [usize; 2] = [2usize, 2usize + 1];
        let base_window = (n / max_passes.max(1)).max(max_passes).min(n);
        let mut focus: Option<(usize, usize)> = None;

        for _ in 0..max_passes {
            let (start_lo, start_hi) = focus.unwrap_or((0usize, base_window));
            let start_hi = start_hi.min(n);

            let mut best_local_mk = best_mk;
            let mut best_move: Option<(usize, usize, usize)> = None;

            for &seg_len in &seg_lens {
                if seg_len > n {
                    continue;
                }
                let max_start = n - seg_len;

                let s0 = start_lo.min(max_start + 1);
                let s1 = start_hi.min(max_start + 1);
                if s0 >= s1 {
                    continue;
                }

                let rem_len = n - seg_len;
                for start in s0..s1 {
                    for ins in 0..=rem_len {
                        if ins == start {
                            continue;
                        }

                        let mut out = 0usize;
                        for r in 0..=rem_len {
                            if r == ins {
                                for t in 0..seg_len {
                                    cand_order[out] = order[start + t];
                                    out += 1;
                                }
                            }
                            if r == rem_len {
                                break;
                            }
                            let orig = if r < start { r } else { r + seg_len };
                            cand_order[out] = order[orig];
                            out += 1;
                        }

                        for (pos, &jj) in cand_order.iter().enumerate() {
                            rank[jj] = pos;
                        }

                        let mk = strict_makespan(challenge, pre, &rank)?;
                        if mk < best_local_mk {
                            best_local_mk = mk;
                            best_move = Some((seg_len, start, ins));
                        }
                    }
                }
            }

            let Some((seg_len, start, ins)) = best_move else { break };

            let rem_len = n - seg_len;
            let mut out = 0usize;
            for r in 0..=rem_len {
                if r == ins {
                    for t in 0..seg_len {
                        cand_order[out] = order[start + t];
                        out += 1;
                    }
                }
                if r == rem_len {
                    break;
                }
                let orig = if r < start { r } else { r + seg_len };
                cand_order[out] = order[orig];
                out += 1;
            }

            order.clone_from(&cand_order);
            best_mk = best_local_mk;
            best_order = order.clone();

            let min_pos = start.min(ins);
            let max_pos = start.max(ins);
            let lo = min_pos.saturating_sub(base_window.min(max_passes));
            let hi = (max_pos + base_window).min(n);
            focus = Some((lo, hi));
        }
        order = best_order.clone();
        for (pos, &j) in order.iter().enumerate() {
            rank[j] = pos;
        }
        {
            let mut seed = challenge.seed;
            seed[0] ^= 0xA5;
            let mut rng = SmallRng::from_seed(seed);
            let swap_budget = (n * 8).clamp(160, 600);
            for _ in 0..swap_budget {
                let i = rng.gen_range(0..n);
                let j = rng.gen_range(0..n);
                if i == j {
                    continue;
                }
                order.swap(i, j);
                rank[order[i]] = i;
                rank[order[j]] = j;
                let mk = strict_makespan(challenge, pre, &rank)?;
                if mk < best_mk {
                    best_mk = mk;
                    best_order = order.clone();
                } else {
                    order.swap(i, j);
                    rank[order[i]] = i;
                    rank[order[j]] = j;
                }
            }
        }
        order = best_order.clone();
        for (pos, &j) in order.iter().enumerate() {
            rank[j] = pos;
        }
        if n >= 2 {
            let max_seg = 5usize.min(n);
            for _ in 0..2 {
                let mut improved = false;
                for seg_len in 2..=max_seg {
                    for start in 0..=(n - seg_len) {
                        order[start..start + seg_len].reverse();
                        for k in start..start + seg_len {
                            rank[order[k]] = k;
                        }
                        let mk = strict_makespan(challenge, pre, &rank)?;
                        if mk < best_mk {
                            best_mk = mk;
                            best_order = order.clone();
                            improved = true;
                        } else {
                            order[start..start + seg_len].reverse();
                            for k in start..start + seg_len {
                                rank[order[k]] = k;
                            }
                        }
                    }
                }
                if !improved {
                    break;
                }
            }
        }
        for (pos, &j) in best_order.iter().enumerate() {
            rank[j] = pos;
        }
        let (best_sol, mk2) = strict_simulate(challenge, pre, &rank)?;
        Ok((best_sol, if mk2 != best_mk { mk2 } else { best_mk }))
    }

    pub fn solve(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        pre: &Pre,
        _effort: &EffortConfig,
    ) -> Result<()> {
        let (greedy_sol, greedy_mk) = run_simple_greedy_baseline(challenge)?;
        save_solution(&greedy_sol)?;

        let mut best_sol = greedy_sol;
        let mut best_mk = greedy_mk;
        let mut top_solutions: Vec<(Solution, u32)> = Vec::new();
        push_top_solutions(&mut top_solutions, &best_sol, best_mk, 5);

        let mut strict_sol: Option<(Solution, u32)> = None;
        if pre.flow_route.is_some() && pre.flex_avg <= 1.25 {
            if let Ok((sol, mk)) = strict_best_by_order_search(challenge, pre, 6) {
                strict_sol = Some((sol.clone(), mk));
                if mk <= best_mk {
                    best_mk = mk;
                    best_sol = sol;
                    save_solution(&best_sol)?;
                    push_top_solutions(&mut top_solutions, &best_sol, best_mk, 5);
                }
            }
        }

        if let (Some(route), Some(pt)) = (&pre.flow_route, &pre.flow_pt_by_job) {
            if let Ok(neh_seq) = neh_best_sequence(pre, challenge.num_jobs, challenge.num_machines) {
                let perm_sol = build_perm_solution_from_seq(
                    &neh_seq,
                    route,
                    pt,
                    challenge.num_jobs,
                    challenge.num_machines,
                );
                if let Ok(mk) = challenge.evaluate_makespan(&perm_sol) {
                    if mk <= best_mk {
                        best_mk = mk;
                        best_sol = perm_sol.clone();
                        save_solution(&best_sol)?;
                    }
                    push_top_solutions(&mut top_solutions, &perm_sol, mk, 5);
                }
                if pre.flex_avg <= 1.25 {
                    let mut rank = vec![challenge.num_jobs; challenge.num_jobs];
                    for (pos, &j) in neh_seq.iter().enumerate() {
                        if j < challenge.num_jobs {
                            rank[j] = pos;
                        }
                    }
                    if let Ok((ssol, _)) = strict_simulate(challenge, pre, &rank) {
                        if let Ok(mk) = challenge.evaluate_makespan(&ssol) {
                            if mk <= best_mk {
                                best_mk = mk;
                                best_sol = ssol.clone();
                                save_solution(&best_sol)?;
                            }
                            push_top_solutions(&mut top_solutions, &ssol, mk, 5);
                        }
                    }
                }
                let unique = route_is_unique(route, challenge.num_machines);
                if unique && !neh_seq.is_empty() && route.len() == pt[neh_seq[0]].len() {
                    let mut starts: Vec<Vec<usize>> = Vec::new();
                    starts.push(neh_seq.clone());
                    if let Some((s, _mk)) = &strict_sol {
                        starts.push(order_from_solution_first_op_start(s, challenge.num_jobs));
                    }
                    starts.push(order_from_solution_first_op_start(&best_sol, challenge.num_jobs));
                    let mut uniq: Vec<Vec<usize>> = Vec::new();
                    for ord in starts {
                        if ord.len() != challenge.num_jobs {
                            continue;
                        }
                        let mut ok = true;
                        for u in &uniq {
                            if *u == ord {
                                ok = false;
                                break;
                            }
                        }
                        if ok {
                            uniq.push(ord);
                        }
                        if uniq.len() >= 3 {
                            break;
                        }
                    }
                    let mut seed = challenge.seed;
                    seed[0] ^= 0x6B;
                    let mut rng = SmallRng::from_seed(seed);
                    let total_iters = 2200usize;
                    let per = (total_iters / uniq.len().max(1)).max(600);
                    let d = 4usize;
                    let mut best_ig_seq = neh_seq;
                    TL_TAILLARD.with(|cell| {
                        let mut buf = cell.borrow_mut();
                        let m = route.len();
                        buf.ensure(best_ig_seq.len(), m);
                        let mk0 = flow_makespan(&best_ig_seq, pt, &mut buf.comp[..m]);
                        let mut best_ig_mk = mk0;
                        for start_seq in uniq.iter() {
                            let cand_seq = iterated_greedy_search(start_seq, pt, per, d, &mut rng);
                            buf.ensure(cand_seq.len(), m);
                            let mk = flow_makespan(&cand_seq, pt, &mut buf.comp[..m]);
                            if mk < best_ig_mk {
                                best_ig_mk = mk;
                                best_ig_seq = cand_seq;
                            }
                        }
                    });
                    let ig_perm_sol = build_perm_solution_from_seq(
                        &best_ig_seq,
                        route,
                        pt,
                        challenge.num_jobs,
                        challenge.num_machines,
                    );
                    if let Ok(mk) = challenge.evaluate_makespan(&ig_perm_sol) {
                        if mk <= best_mk {
                            best_mk = mk;
                            best_sol = ig_perm_sol.clone();
                            save_solution(&best_sol)?;
                        }
                        push_top_solutions(&mut top_solutions, &ig_perm_sol, mk, 5);
                    }
                }
            } else if let Ok(sol) = {
                let route = route;
                let pt = pt;
                let seq = neh_best_sequence(pre, challenge.num_jobs, challenge.num_machines);
                seq.map(|s| {
                    build_perm_solution_from_seq(&s, route, pt, challenge.num_jobs, challenge.num_machines)
                })
            } {
                if let Ok(mk) = challenge.evaluate_makespan(&sol) {
                    if mk <= best_mk {
                        best_mk = mk;
                        best_sol = sol.clone();
                        save_solution(&best_sol)?;
                    }
                    push_top_solutions(&mut top_solutions, &sol, mk, 5);
                }
            }
        }

        let flow_is_reentrant = pre.flow_route.is_some()
            && !route_is_unique(pre.flow_route.as_deref().unwrap_or(&[]), challenge.num_machines)
            && pre.flex_avg <= 1.25;

        if flow_is_reentrant {
            let mut seed = challenge.seed;
            seed[0] ^= 0xF1;
            let mut rng = SmallRng::from_seed(seed);
            let grasp_rules = [
                Rule::BnHeavy,
                Rule::MostWork,
                Rule::EndTight,
                Rule::ShortestProc,
                Rule::LeastFlex,
                Rule::CriticalPath,
                Rule::Regret,
                Rule::EarliestStart,
                Rule::MachineBalance,
                Rule::SlackRatio,
                Rule::BackwardCritical,
                Rule::WeightedCompletion,
            ];

            let mut attempts: Vec<u32> = vec![0u32; grasp_rules.len()];
            let mut improves: Vec<u32> = vec![0u32; grasp_rules.len()];
            let mut delta_sum: Vec<u64> = vec![0u64; grasp_rules.len()];
            let mut total_attempts: u32 = 0;

            let num_restarts = 500usize;
            for r in 0..num_restarts {
                let mut untried: Vec<usize> = Vec::new();
                for i in 0..grasp_rules.len() {
                    if attempts[i] == 0 {
                        untried.push(i);
                    }
                }

                let ridx = if !untried.is_empty() {
                    untried[rng.gen_range(0..untried.len())]
                } else {
                    let mut best_i = 0usize;
                    let mut best_score = 0u64;
                    let mut best_succ = 0u32;

                    for i in 0..grasp_rules.len() {
                        let a = attempts[i].max(1) as u64;
                        let avg_imp = delta_sum[i] / a;
                        let explore = (total_attempts as u64) / a;
                        let score = avg_imp.saturating_add(explore);

                        if score > best_score || (score == best_score && improves[i] > best_succ) {
                            best_score = score;
                            best_succ = improves[i];
                            best_i = i;
                        } else if score == best_score && improves[i] == best_succ {
                            if (rng.gen::<u32>() & 1) == 0 {
                                best_i = i;
                            }
                        }
                    }
                    best_i
                };

                let rule = grasp_rules[ridx];
                let k = if r < grasp_rules.len() {
                    0
                } else {
                    rng.gen_range(2..=5)
                };

                attempts[ridx] = attempts[ridx].saturating_add(1);
                total_attempts = total_attempts.saturating_add(1);

                let prev_best = best_mk;
                if let Ok((sol, mk)) = construct_solution_conflict(
                    challenge,
                    pre,
                    rule,
                    k,
                    Some(best_mk.saturating_add(best_mk / 20)),
                    &mut rng,
                    None,
                    None,
                    None,
                    0.0,
                ) {
                    if mk < prev_best {
                        improves[ridx] = improves[ridx].saturating_add(1);
                        delta_sum[ridx] = delta_sum[ridx].saturating_add((prev_best - mk) as u64);
                    }

                    if mk < best_mk {
                        best_mk = mk;
                        best_sol = sol.clone();
                        save_solution(&best_sol)?;
                    }
                    push_top_solutions(&mut top_solutions, &sol, mk, 15);
                }
            }
        }

        let is_strict_perm = route_is_unique(
            pre.flow_route.as_deref().unwrap_or(&[]),
            challenge.num_machines,
        ) && pre.flex_avg <= 1.25;

        if !is_strict_perm {
            let ls_runs = top_solutions.len().min(15);
            let perturb_cycles = 16usize;
            for i in 0..ls_runs {
                let base_sol = &top_solutions[i].0;
                if let Ok(Some((sol2, mk2))) =
                    critical_block_move_local_search_ex(pre, challenge, base_sol, 5, 64, perturb_cycles)
                {
                    if mk2 < best_mk {
                        best_mk = mk2;
                        best_sol = sol2.clone();
                        save_solution(&best_sol)?;
                    }
                    push_top_solutions(&mut top_solutions, &sol2, mk2, 15);
                }
            }
        } else {
            let extra_iters = 1200usize;
            if let (Some(route), Some(pt)) = (&pre.flow_route, &pre.flow_pt_by_job) {
                let unique = route_is_unique(route, challenge.num_machines);
                if unique && !pt.is_empty() {
                    let mut seed = challenge.seed;
                    seed[0] ^= 0xD4;
                    let mut rng = SmallRng::from_seed(seed);
                    let m = route.len();
                    TL_TAILLARD.with(|cell| {
                        let mut buf = cell.borrow_mut();

                        let seed_cap = top_solutions.len().min(5);
                        if seed_cap == 0 {
                            return;
                        }
                        let ksig = seed_cap.min(challenge.num_jobs.max(1));

                        let signature = |s: &Solution| -> Vec<usize> {
                            let mut best: Vec<(u32, usize)> = Vec::with_capacity(ksig);
                            for j in 0..s.job_schedule.len() {
                                let t = s.job_schedule[j]
                                    .first()
                                    .map(|x| x.1)
                                    .unwrap_or(u32::MAX);

                                let mut pos = best.len();
                                while pos > 0 {
                                    let (bt, bj) = best[pos - 1];
                                    if bt < t || (bt == t && bj < j) {
                                        break;
                                    }
                                    pos -= 1;
                                }
                                if pos >= ksig {
                                    continue;
                                }
                                best.insert(pos, (t, j));
                                if best.len() > ksig {
                                    best.pop();
                                }
                            }
                            best.into_iter().map(|(_, j)| j).collect()
                        };

                        let similarity = |a: &[usize], b: &[usize]| -> usize {
                            let len = a.len().min(b.len());
                            let mut same = 0usize;
                            for i in 0..len {
                                if a[i] == b[i] {
                                    same += 1;
                                }
                            }
                            same
                        };

                        let mut sigs: Vec<Vec<usize>> = Vec::with_capacity(top_solutions.len());
                        for (s, _) in top_solutions.iter() {
                            sigs.push(signature(s));
                        }

                        let mut picked: Vec<usize> = Vec::with_capacity(seed_cap);

                        let mut first = 0usize;
                        for i in 1..top_solutions.len() {
                            if top_solutions[i].1 < top_solutions[first].1 {
                                first = i;
                            }
                        }
                        picked.push(first);

                        while picked.len() < seed_cap {
                            let mut best_i = NONE_USIZE;
                            let mut best_max_sim = usize::MAX;
                            let mut best_mk = u32::MAX;

                            for i in 0..top_solutions.len() {
                                if picked.iter().any(|&p| p == i) {
                                    continue;
                                }
                                let mut max_sim = 0usize;
                                for &p in &picked {
                                    let sim = similarity(&sigs[i], &sigs[p]);
                                    if sim > max_sim {
                                        max_sim = sim;
                                    }
                                }
                                let mk_i = top_solutions[i].1;
                                if max_sim < best_max_sim || (max_sim == best_max_sim && mk_i < best_mk) {
                                    best_max_sim = max_sim;
                                    best_mk = mk_i;
                                    best_i = i;
                                }
                            }

                            if best_i == NONE_USIZE {
                                break;
                            }
                            picked.push(best_i);
                        }

                        for &i in &picked {
                            let start_ord =
                                order_from_solution_first_op_start(&top_solutions[i].0, challenge.num_jobs);
                            if start_ord.len() != challenge.num_jobs {
                                continue;
                            }
                            let cand_seq =
                                iterated_greedy_search(&start_ord, pt, extra_iters / 5, 4, &mut rng);
                            buf.ensure(cand_seq.len(), m);
                            let mk = flow_makespan(&cand_seq, pt, &mut buf.comp[..m]);
                            if mk < best_mk {
                                best_mk = mk;
                                let sol = build_perm_solution_from_seq(
                                    &cand_seq,
                                    route,
                                    pt,
                                    challenge.num_jobs,
                                    challenge.num_machines,
                                );
                                best_sol = sol;
                                let _ = save_solution(&best_sol);
                            }
                        }
                    });
                }
            }
        }

        Ok(())
    }
}

pub mod hybrid_flow_shop {
    use anyhow::{anyhow, Result};
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    use tig_challenges::job_scheduling::*;
    use super::types::*;
    use super::infra_shared::*;
    use crate::{seeded_hasher, HashMap};

    type DetCache = HashMap<(u64, usize, usize, usize, u8), Option<(Solution, u32)>>;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Rule {
        Adaptive, BnHeavy, EndTight, CriticalPath, MostWork, LeastFlex, Regret, ShortestProc, FlexBalance,
    }

    #[inline]
    fn slack_urgency_hfs(pre: &Pre, target_mk: Option<u32>, time: u32, product: usize, op_idx: usize) -> f64 {
        let Some(tgt) = target_mk else { return 0.0 };
        let lb = (time as u64).saturating_add(pre.product_suf_min[product][op_idx] as u64);
        let slack = (tgt as i64) - (lb as i64);
        let scale = (0.70 * pre.avg_op_min).max(1.0);
        let pos = (slack.max(0) as f64) / scale; let neg = ((-slack).max(0) as f64) / scale;
        (1.0 / (1.0 + pos)).clamp(0.0, 1.0) + (0.35 * neg).min(3.0)
    }

    #[inline]
    fn route_pref_bonus_hfs(rp: Option<&RoutePrefLite>, product: usize, op_idx: usize, machine: usize) -> f64 {
        let Some(rp) = rp else { return 0.0 };
        if product >= rp.len() || op_idx >= rp[product].len() { return 0.0; }
        let r = rp[product][op_idx]; let mu = machine.min(255) as u8;
        if mu == r.best_m { (r.best_w as f64) / 255.0 } else if mu == r.second_m { (r.second_w as f64) / 255.0 } else { 0.0 }
    }


    #[inline]
    fn rule_idx(r: Rule) -> usize {
        match r { Rule::Adaptive=>0, Rule::BnHeavy=>1, Rule::EndTight=>2, Rule::CriticalPath=>3, Rule::MostWork=>4, Rule::LeastFlex=>5, Rule::Regret=>6, Rule::ShortestProc=>7, Rule::FlexBalance=>8 }
    }

    fn choose_rule_bandit(rng: &mut SmallRng, rules: &[Rule], rule_best: &[u32], rule_tries: &[u32], global_best: u32, margin: u32, stuck: usize, chaos_like: bool, late_phase: bool) -> Rule {
        if rules.is_empty() { return Rule::Adaptive; }
        let mut best_seen = global_best; for &mk in rule_best { if mk < best_seen { best_seen = mk; } }
        let scale = (margin as f64).max(1.0); let s = ((stuck as f64)/140.0).clamp(0.0,1.0); let explore_mix = (0.10+0.55*s).clamp(0.10,0.65);

        let mut sum = 0.0;
        for &r in rules.iter() {
            let mk=rule_best[rule_idx(r)]; let t=rule_tries[rule_idx(r)].max(1) as f64;
            let delta=mk.saturating_sub(best_seen) as f64; let exploit=(-delta/scale).exp(); let explore=(1.0/t).sqrt();
            let mut ww=(1.0-explore_mix)*exploit+explore_mix*explore; ww=ww.max(1e-6);
            if chaos_like{ww=ww.powf(0.70);}else if late_phase{ww=ww.powf(1.18);}
            sum += ww.max(0.0);
        }

        if !(sum>0.0) { return rules[rng.gen_range(0..rules.len())]; }

        let mut r=rng.gen::<f64>()*sum;
        for &rule in rules.iter() {
            let mk=rule_best[rule_idx(rule)]; let t=rule_tries[rule_idx(rule)].max(1) as f64;
            let delta=mk.saturating_sub(best_seen) as f64; let exploit=(-delta/scale).exp(); let explore=(1.0/t).sqrt();
            let mut ww=(1.0-explore_mix)*exploit+explore_mix*explore; ww=ww.max(1e-6);
            if chaos_like{ww=ww.powf(0.70);}else if late_phase{ww=ww.powf(1.18);}
            r-=ww.max(0.0); if r<=0.0 { return rule; }
        }
        rules[rules.len()-1]
    }

    fn construct_solution_conflict_mode<const HAS_TARGET: bool, const HAS_JOB_BIAS: bool, const HAS_MACHINE_PENALTY: bool, const USE_ROUTE_PREF: bool>(
        challenge: &Challenge, pre: &Pre, rule: Rule, k: usize, target_mk: u32,
        rng: &mut SmallRng, job_bias: &[f64], machine_penalty: &[f64],
        route_pref: Option<&RoutePrefLite>, horizon: f64, time_scale: f64,
    ) -> Result<(Solution, u32)> {
        let num_jobs=challenge.num_jobs; let num_machines=challenge.num_machines;
        let mut job_next_op=vec![0usize;num_jobs]; let mut job_ready_time=vec![0u32;num_jobs];
        let mut machine_avail=vec![0u32;num_machines]; let mut machine_load=pre.machine_load0.clone();
        let mut job_schedule: Vec<Vec<(usize,u32)>>=pre.job_ops_len.iter().map(|&len|Vec::with_capacity(len)).collect();
        let mut remaining_ops=pre.total_ops; let mut time=0u32;

        let mut demand: Vec<u16>=vec![0u16;num_machines];
        let mut raw_by_machine: Vec<Vec<RawCand>>=(0..num_machines).map(|_|Vec::with_capacity(12)).collect();
        let mut idle_machines: Vec<usize>=Vec::with_capacity(num_machines);
        let chaotic_like=pre.chaotic_like;
        let mut machine_work: Vec<u64>=if chaotic_like{vec![0u64;num_machines]}else{vec![]};
        let mut sum_work: u64=0;

        let mut ready_jobs: Vec<usize> = Vec::with_capacity(num_jobs);
        let mut ready_pos: Vec<usize> = vec![usize::MAX; num_jobs];
        let mut in_ready: Vec<bool> = vec![false; num_jobs];
        let mut ready_heap: std::collections::BinaryHeap<std::cmp::Reverse<(u32, usize)>> =
            std::collections::BinaryHeap::new();

        for j in 0..num_jobs {
            if pre.job_ops_len[j] == 0 { continue; }
            in_ready[j] = true;
            ready_pos[j] = ready_jobs.len();
            ready_jobs.push(j);
        }

        let update_ready = |time: u32,
                                ready_jobs: &mut Vec<usize>,
                                ready_pos: &mut [usize],
                                in_ready: &mut [bool],
                                ready_heap: &mut std::collections::BinaryHeap<std::cmp::Reverse<(u32, usize)>>,
                                job_next_op: &[usize],
                                job_ready_time: &[u32]| {
            while let Some(std::cmp::Reverse((t, j))) = ready_heap.peek().copied() {
                if t > time { break; }
                ready_heap.pop();
                if j >= in_ready.len() { continue; }
                if in_ready[j] { continue; }
                if job_next_op[j] >= pre.job_ops_len[j] { continue; }
                if job_ready_time[j] != t { continue; }
                in_ready[j] = true;
                ready_pos[j] = ready_jobs.len();
                ready_jobs.push(j);
            }
        };

        while remaining_ops > 0 {
            update_ready(time, &mut ready_jobs, &mut ready_pos, &mut in_ready, &mut ready_heap, &job_next_op, &job_ready_time);

            loop {
                idle_machines.clear();
                for m in 0..num_machines { if machine_avail[m]<=time { idle_machines.push(m); } }
                if idle_machines.is_empty() { break; }

                for &m in &idle_machines { demand[m]=0; raw_by_machine[m].clear(); }
                let progress=1.0-(remaining_ops as f64)/(pre.total_ops as f64).max(1.0);
                let cap_per_machine=if k==0{12usize}else{(k+6).min(12)};

                for &job in &ready_jobs {
                    if !in_ready[job] { continue; }
                    let op_idx=job_next_op[job];
                    if op_idx>=pre.job_ops_len[job] || job_ready_time[job]>time { continue; }

                    let product=pre.job_products[job]; let op=&pre.product_ops[product][op_idx];
                    if op.flex==0||op.machines.is_empty()||op.min_pt>=INF{continue;}
                    let (best_end,second_end,best_cnt_total,best_cnt_idle)=best_second_and_counts(time,&machine_avail,op);
                    if best_end>=INF||best_cnt_idle==0{continue;}

                    let ops_rem=pre.job_ops_len[job]-op_idx; let jb=if HAS_JOB_BIAS { job_bias[job] } else { 0.0 };
                    let flex_inv=1.0/(op.flex as f64).max(1.0); let scarcity_urg=1.0/(best_cnt_total as f64).max(1.0);
                    let avg_op_min_scale=pre.avg_op_min.max(1.0); let horizon_scale=horizon.max(1.0); let time_scale_scale=time_scale.max(1.0);
                    let regret=if second_end>=INF{pre.avg_op_min*2.6}else{(second_end-best_end) as f64};
                    let regn=(regret/avg_op_min_scale).clamp(0.0,6.0); let rigidity=(0.60*flex_inv+0.40*scarcity_urg).clamp(0.0,2.5);

                    let rem_min=pre.product_suf_min[product][op_idx] as f64;
                    let rem_avg=pre.product_suf_avg[product][op_idx];
                    let rem_bn=pre.product_suf_bn[product][op_idx];
                    let rem_min_n=rem_min/horizon_scale;
                    let rem_avg_n=rem_avg/pre.max_job_avg_work.max(1e-9);
                    let bn_n=rem_bn/pre.max_job_bn.max(1e-9);
                    let density_n=((rem_min/(ops_rem as f64).max(1.0))/avg_op_min_scale).clamp(0.0,4.0);

                    let next_min=pre.product_next_min[product][op_idx] as f64;
                    let next_min_n=next_min/horizon_scale;
                    let next_flex_inv=pre.product_next_flex_inv[product][op_idx];
                    let next_term_raw=(0.55*next_min_n+0.45*next_flex_inv)*(1.0+0.30*density_n*pre.high_flex);

                    let flow_term=pre.flow_w*pre.job_flow_pref[job]*(0.65+0.70*(1.0-progress));
                    let slack_u=if HAS_TARGET { slack_urgency_hfs(pre,Some(target_mk),time,product,op_idx) } else { 0.0 };

                    let rem_min_u=if rem_min_n<=0.0{0.0}else{rem_min_n/(1.0+rem_min_n)};
                    let rem_avg_u=if rem_avg_n<=0.0{0.0}else{rem_avg_n/(1.0+rem_avg_n)};
                    let bn_u=if bn_n<=0.0{0.0}else{bn_n/(1.0+bn_n)};
                    let reg_u=if regn<=0.0{0.0}else{regn/(1.0+regn)};
                    let dens_u=if density_n<=0.0{0.0}else{density_n/(1.0+density_n)};
                    let next_u=if next_term_raw<=0.0{0.0}else{next_term_raw/(1.0+next_term_raw)};
                    let end_n=(best_end as f64)/time_scale_scale;
                    let end_u=if end_n<=0.0{0.0}else{end_n/(1.0+end_n)};
                    let flex_term=flex_inv*pre.flex_factor.max(0.0);
                    let flex_u=if flex_term<=0.0{0.0}else{flex_term/(1.0+flex_term)};
                    let sat_scarcity=if scarcity_urg<=0.0{0.0}else{scarcity_urg/(1.0+scarcity_urg)};
                    let scarce_slack=scarcity_urg*slack_u;
                    let scarce_reg=scarcity_urg*reg_u;
                    let prog_gate=if progress<=0.0{0.0}else{progress/(1.0+progress)};
                    let base_bias0=jb+flow_term;
                    let bn_focus_u=if pre.bn_focus<=0.0{0.0}else{pre.bn_focus/(1.0+pre.bn_focus)};

                    for &(m,pt) in &op.machines {
                        if machine_avail[m]>time{continue;}
                        let end=time.saturating_add(pt); if end!=best_end{continue;}
                        demand[m]=demand[m].saturating_add(1);
                        let mp=if HAS_MACHINE_PENALTY { machine_penalty[m] } else { 0.0 }; let jitter=if k>0{rng.gen::<f64>()*1e-9}else{0.0};
                        let load_n=machine_load[m]/pre.avg_machine_load.max(1e-9);
                        let proc_n=(pt as f64)/avg_op_min_scale;
                        let mpen=mp.clamp(0.0,1.0);
                        let pop_pen=if pre.chaotic_like&&op.flex>=2{
                            let pop=pre.machine_best_pop[m];
                            (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor
                        }else{
                            0.0
                        };
                        let load_u=if load_n<=0.0{0.0}else{load_n/(1.0+load_n)};
                        let proc_u=if proc_n<=0.0{0.0}else{proc_n/(1.0+proc_n)};
                        let mpen_u=if mpen<=0.0{0.0}else{mpen/(1.0+mpen)};
                        let base_bias=base_bias0+jitter;
                        let base=match rule {
                            Rule::CriticalPath => {
                                let chain=rem_min_u*(1.0+next_u);
                                let urgent=scarce_slack*(1.0+scarce_reg*prog_gate);
                                chain+urgent+base_bias-end_u-pop_pen
                            }
                            Rule::MostWork => {
                                let work=rem_avg_u*(1.0+dens_u);
                                let smooth=work*(1.0+load_u);
                                smooth+base_bias-end_u-pop_pen
                            }
                            Rule::LeastFlex => {
                                let rigid=flex_u*(1.0+sat_scarcity);
                                rigid+rem_min_u+next_u+base_bias-end_u-pop_pen
                            }
                            Rule::ShortestProc => {
                                let short=0.0-proc_u;
                                short+rem_min_u*(1.0+next_u)+sat_scarcity+base_bias-end_u-pop_pen
                            }
                            Rule::Regret => {
                                let regret_focus=reg_u*(1.0+sat_scarcity)*(1.0+prog_gate);
                                regret_focus+rem_min_u+next_u+base_bias-end_u-pop_pen
                            }
                            Rule::EndTight => {
                                let tight=scarce_slack*(1.0+scarce_reg);
                                let chain=rem_min_u*(1.0+prog_gate)*(1.0+next_u);
                                let penal=end_u*(1.0+prog_gate)+proc_u+mpen_u*pre.flex_factor;
                                chain+tight+base_bias-penal-pop_pen
                            }
                            Rule::BnHeavy => {
                                let bn_focus=bn_u*(1.0+dens_u)*(1.0+bn_focus_u);
                                let chain=rem_min_u*(1.0+next_u);
                                let penal=end_u+proc_u+load_u*pre.flex_factor+mpen_u*pre.flex_factor;
                                bn_focus+chain+scarce_slack+reg_u+flex_u+base_bias-penal-pop_pen
                            }
                            Rule::Adaptive => {
                                let js=pre.jobshopness;
                                let fl=1.0-js;
                                if js>=fl{
                                    let hard=reg_u*(1.0+scarce_reg)+flex_u+rem_min_u*(1.0+next_u);
                                    hard+base_bias-(end_u+mpen_u*pre.flex_factor)-pop_pen
                                }else{
                                    let flow=rem_avg_u*(1.0+dens_u)+(0.0-proc_u)+slack_u;
                                    flow+base_bias-(end_u+load_u*pre.flex_factor)-pop_pen
                                }
                            }
                            Rule::FlexBalance => {
                                let flexible=flex_u*(1.0+sat_scarcity);
                                let chain=(rem_avg_u+rem_min_u)*(1.0+next_u);
                                let penal=end_u+load_u*pre.flex_factor+mpen_u*(1.0+pre.flex_factor);
                                flexible+chain+base_bias-penal-pop_pen
                            }
                        };
                        push_top_k_raw(&mut raw_by_machine[m],RawCand{job,machine:m,pt,base_score:base,rigidity,reg_n:regn},cap_per_machine);
                    }
                }

                let denom=(idle_machines.len() as f64).max(1.0);
                let (conflict_w,conflict_scale)=if chaotic_like{(-(0.05+0.08*(1.0-progress)).clamp(0.04,0.14),(0.95+0.20*pre.flex_factor).clamp(0.90,1.20))}else{((0.09+0.26*pre.jobshopness+0.11*pre.high_flex+0.16*(1.0-progress)).clamp(0.05,0.45),(0.90+0.40*pre.flex_factor).clamp(0.85,1.75))};
                let (bal_w,avg_work)=if chaotic_like{((0.030+0.070*(1.0-progress)).clamp(0.025,0.11),(sum_work as f64)/(num_machines as f64).max(1.0))}else{(0.0,0.0)};

                let mut best: Option<Cand>=None; let mut top: Vec<Cand>=if k>0{Vec::with_capacity(k)}else{Vec::new()};
                for &m in &idle_machines {
                    let dem=demand[m] as f64; if dem<=0.0||raw_by_machine[m].is_empty(){continue;}
                    let dem_n=((dem-1.0)/denom).clamp(0.0,2.5);
                    let bal_pen=if chaotic_like&&bal_w>0.0{let denomw=(avg_work+(pre.avg_op_min*3.0).max(1.0)).max(1.0); let r=(machine_work[m] as f64)/denomw; let done_n=(r/(r+1.0)).clamp(0.0,1.0); -bal_w*done_n}else{0.0};
                    for rc in &raw_by_machine[m] {
                        let rig=rc.rigidity.clamp(0.0,2.5); let regc=rc.reg_n.clamp(0.0,4.5);
                        let mut boost=conflict_w*conflict_scale*dem_n*(1.15*rig+0.85*regc);
                        if chaotic_like{boost=boost.max(-0.26);}
                        let c=Cand{job:rc.job,machine:rc.machine,pt:rc.pt,score:rc.base_score+boost+bal_pen};
                        if k==0{if best.map_or(true,|bb|c.score>bb.score){best=Some(c);}}else{push_top_k(&mut top,c,k);}
                    }
                }

                let chosen=if k==0{
                    match best{Some(c)=>c,None=>break}
                }else{
                    if top.is_empty(){break;}
                    if USE_ROUTE_PREF{
                        let rp=route_pref.unwrap();
                        let mut best_rb: Option<f64>=None;
                        let mut best_idx: Vec<usize>=Vec::new();
                        for (i,c) in top.iter().enumerate(){
                            let job=c.job;
                            let op_idx=job_next_op[job];
                            if op_idx>=pre.job_ops_len[job]{continue;}
                            let product=pre.job_products[job];
                            let rb=route_pref_bonus_hfs(Some(rp),product,op_idx,c.machine);
                            match best_rb{
                                None=>{best_rb=Some(rb);best_idx.clear();best_idx.push(i);}
                                Some(b)=>{
                                    if rb>b{best_rb=Some(rb);best_idx.clear();best_idx.push(i);}
                                    else if rb==b{best_idx.push(i);}
                                }
                            }
                        }
                        if best_idx.len()==1{
                            top[best_idx[0]]
                        }else if best_idx.is_empty(){
                            choose_from_top_weighted(rng,&top)
                        }else{
                            let mut filtered: Vec<Cand>=Vec::with_capacity(best_idx.len());
                            for &i in &best_idx{filtered.push(top[i]);}
                            choose_from_top_weighted(rng,&filtered)
                        }
                    }else{
                        choose_from_top_weighted(rng,&top)
                    }
                };

                let job=chosen.job; let machine=chosen.machine; let pt=chosen.pt;
                let product=pre.job_products[job]; let op_idx=job_next_op[job]; let op=&pre.product_ops[product][op_idx];
                let end_time=time.saturating_add(pt);

                in_ready[job] = false;
                let pos = ready_pos[job];
                if pos < ready_jobs.len() && ready_jobs[pos] == job {
                    ready_jobs.swap_remove(pos);
                    if pos < ready_jobs.len() {
                        let moved = ready_jobs[pos];
                        ready_pos[moved] = pos;
                    }
                }
                ready_pos[job] = usize::MAX;

                job_schedule[job].push((machine,time)); job_next_op[job]+=1; job_ready_time[job]=end_time; machine_avail[machine]=end_time; remaining_ops-=1;

                if job_next_op[job] < pre.job_ops_len[job] {
                    ready_heap.push(std::cmp::Reverse((end_time, job)));
                }

                if chaotic_like{machine_work[machine]=machine_work[machine].saturating_add(pt as u64);sum_work=sum_work.saturating_add(pt as u64);}
                if op.min_pt<INF&&op.flex>0&&!op.machines.is_empty(){let delta=(op.min_pt as f64)/(op.flex as f64).max(1.0);if delta>0.0{for &(mm,_) in &op.machines{let v=machine_load[mm]-delta;machine_load[mm]=if v>0.0{v}else{0.0};}}}
                if remaining_ops==0{break;}
            }

            if remaining_ops==0{break;}

            let mut next_time: Option<u32>=None;
            for &t in &machine_avail{if t>time{next_time=Some(next_time.map_or(t,|b|b.min(t)));}}

            loop {
                let Some(std::cmp::Reverse((t, j))) = ready_heap.peek().copied() else { break; };
                if t <= time {
                    ready_heap.pop();
                    if j < num_jobs && !in_ready[j] && job_next_op[j] < pre.job_ops_len[j] && job_ready_time[j] == t {
                        in_ready[j] = true;
                        ready_pos[j] = ready_jobs.len();
                        ready_jobs.push(j);
                    }
                    continue;
                }
                if j >= num_jobs || in_ready[j] || job_next_op[j] >= pre.job_ops_len[j] || job_ready_time[j] != t {
                    ready_heap.pop();
                    continue;
                }
                next_time = Some(next_time.map_or(t, |b| b.min(t)));
                break;
            }

            time=next_time.ok_or_else(||anyhow!("Stalled"))?;
        }

        let mk=machine_avail.into_iter().max().unwrap_or(0);
        Ok((Solution{job_schedule},mk))
    }

    fn construct_solution_conflict(
        challenge: &Challenge, pre: &Pre, rule: Rule, k: usize, target_mk: Option<u32>,
        rng: &mut SmallRng, job_bias: Option<&[f64]>, machine_penalty: Option<&[f64]>,
        route_pref: Option<&RoutePrefLite>, route_w: f64, horizon: f64, time_scale: f64,
    ) -> Result<(Solution, u32)> {
        let empty: &[f64] = &[];
        let routed = if route_w > 0.0 { route_pref } else { None };

        if let Some(tgt) = target_mk {
            if let Some(jb) = job_bias {
                if let Some(mp) = machine_penalty {
                    if let Some(rp) = routed {
                        construct_solution_conflict_mode::<true,true,true,true>(challenge,pre,rule,k,tgt,rng,jb,mp,Some(rp),horizon,time_scale)
                    } else {
                        construct_solution_conflict_mode::<true,true,true,false>(challenge,pre,rule,k,tgt,rng,jb,mp,None,horizon,time_scale)
                    }
                } else if let Some(rp) = routed {
                    construct_solution_conflict_mode::<true,true,false,true>(challenge,pre,rule,k,tgt,rng,jb,empty,Some(rp),horizon,time_scale)
                } else {
                    construct_solution_conflict_mode::<true,true,false,false>(challenge,pre,rule,k,tgt,rng,jb,empty,None,horizon,time_scale)
                }
            } else if let Some(mp) = machine_penalty {
                if let Some(rp) = routed {
                    construct_solution_conflict_mode::<true,false,true,true>(challenge,pre,rule,k,tgt,rng,empty,mp,Some(rp),horizon,time_scale)
                } else {
                    construct_solution_conflict_mode::<true,false,true,false>(challenge,pre,rule,k,tgt,rng,empty,mp,None,horizon,time_scale)
                }
            } else if let Some(rp) = routed {
                construct_solution_conflict_mode::<true,false,false,true>(challenge,pre,rule,k,tgt,rng,empty,empty,Some(rp),horizon,time_scale)
            } else {
                construct_solution_conflict_mode::<true,false,false,false>(challenge,pre,rule,k,tgt,rng,empty,empty,None,horizon,time_scale)
            }
        } else if let Some(jb) = job_bias {
            if let Some(mp) = machine_penalty {
                if let Some(rp) = routed {
                    construct_solution_conflict_mode::<false,true,true,true>(challenge,pre,rule,k,0,rng,jb,mp,Some(rp),horizon,time_scale)
                } else {
                    construct_solution_conflict_mode::<false,true,true,false>(challenge,pre,rule,k,0,rng,jb,mp,None,horizon,time_scale)
                }
            } else if let Some(rp) = routed {
                construct_solution_conflict_mode::<false,true,false,true>(challenge,pre,rule,k,0,rng,jb,empty,Some(rp),horizon,time_scale)
            } else {
                construct_solution_conflict_mode::<false,true,false,false>(challenge,pre,rule,k,0,rng,jb,empty,None,horizon,time_scale)
            }
        } else if let Some(mp) = machine_penalty {
            if let Some(rp) = routed {
                construct_solution_conflict_mode::<false,false,true,true>(challenge,pre,rule,k,0,rng,empty,mp,Some(rp),horizon,time_scale)
            } else {
                construct_solution_conflict_mode::<false,false,true,false>(challenge,pre,rule,k,0,rng,empty,mp,None,horizon,time_scale)
            }
        } else if let Some(rp) = routed {
            construct_solution_conflict_mode::<false,false,false,true>(challenge,pre,rule,k,0,rng,empty,empty,Some(rp),horizon,time_scale)
        } else {
            construct_solution_conflict_mode::<false,false,false,false>(challenge,pre,rule,k,0,rng,empty,empty,None,horizon,time_scale)
        }
    }

    #[derive(Clone)]
    struct EliteParams {
        jb: Vec<f64>,
        mp: Vec<f64>,
        rp: RoutePrefLite,
        score: u32,
    }

    fn normalize_elite(elite: &mut Vec<EliteParams>, cap: usize) {
        if elite.is_empty() {
            return;
        }
        if elite.len() <= 1 {
            if elite.len() > cap {
                elite.truncate(cap);
            }
            return;
        }

        #[inline]
        fn elite_sig64(e: &EliteParams) -> u64 {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();

            for (i, v) in e.jb.iter().enumerate() {
                (i as u32).hash(&mut hasher);
                v.to_bits().hash(&mut hasher);
            }
            for (i, v) in e.mp.iter().enumerate() {
                (i as u32).hash(&mut hasher);
                v.to_bits().hash(&mut hasher);
            }
            for (p, ops) in e.rp.iter().enumerate() {
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

        let mut view: Vec<(u32, u64, usize)> = elite
            .iter()
            .enumerate()
            .map(|(idx, e)| (e.score, elite_sig64(e), idx))
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
                let (si, di, sgi, _) = cand[i];
                let (sj, dj, sgj, _) = cand[j];

                let no_worse = sj <= si && dj >= di;
                let strictly_better = sj < si || dj > di;
                if no_worse && strictly_better {
                    if sj == si && dj == di && sgj > sgi {
                        continue;
                    }
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

        let mut selected: Vec<(u32, u64, usize)> = Vec::new();
        selected.push(view[0]);

        #[inline]
        fn already_selected(selected: &[(u32, u64, usize)], idx: usize) -> bool {
            selected.iter().any(|x| x.2 == idx)
        }

        while selected.len() < cap && selected.len() < elite.len() {
            let mut best_pick: Option<(u32, u64, usize, u32)> = None;
            for &(s, sg, idx) in &skyline {
                if already_selected(&selected, idx) {
                    continue;
                }
                let mut md = u32::MAX;
                for &(_ss, ssg, _ii) in &selected {
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
            if let Some((s, sg, idx, _)) = best_pick {
                selected.push((s, sg, idx));
            } else {
                break;
            }
        }

        while selected.len() < cap && selected.len() < elite.len() {
            let mut best_pick: Option<(u32, u64, usize, u32)> = None;
            for &(s, sg, idx) in &view {
                if already_selected(&selected, idx) {
                    continue;
                }
                let mut md = u32::MAX;
                for &(_ss, ssg, _ii) in &selected {
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
            if let Some((s, sg, idx, _)) = best_pick {
                selected.push((s, sg, idx));
            } else {
                break;
            }
        }

        let mut new_elite: Vec<EliteParams> = selected
            .into_iter()
            .map(|(_s, _sg, idx)| elite[idx].clone())
            .collect();
        new_elite.sort_by_key(|e| e.score);

        if new_elite.len() > cap {
            new_elite.truncate(cap);
        }
        *elite = new_elite;
    }

    fn pick_elite_idx(rng: &mut SmallRng, elite: &[EliteParams]) -> usize {
        let len = elite.len(); if len <= 1 { return 0; }
        let a = rng.gen_range(0..len); let b = rng.gen_range(0..len);
        if elite[a].score <= elite[b].score { a } else { b }
    }

    fn pick_top_idx(rng: &mut SmallRng, top: &[(Solution, u32)]) -> usize {
        let len = top.len(); if len <= 1 { return 0; }
        let a = rng.gen_range(0..len); let b = rng.gen_range(0..len);
        if top[a].1 <= top[b].1 { a } else { b }
    }

    #[inline]
    fn solution_sig64(sol: &Solution) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for (j, ops) in sol.job_schedule.iter().enumerate() {
            for (o, (m, _t)) in ops.iter().enumerate() {
                std::hash::Hash::hash(&(j, o, *m), &mut hasher);
            }
        }
        std::hash::Hasher::finish(&hasher)
    }

    #[inline]
    fn pick_diverse_top_solution(best: &Solution, top: &[(Solution, u32)], scan: usize) -> Option<usize> {
        if top.is_empty() || scan == 0 { return None; }
        let sig_best = solution_sig64(best);
        let lim = scan.min(top.len());
        let mut best_idx: Option<usize> = None;
        let mut best_dist: u32 = 0;
        for i in 0..lim {
            let sig_i = solution_sig64(&top[i].0);
            let dist = (sig_best ^ sig_i).count_ones();
            if best_idx.is_none() || dist > best_dist {
                best_idx = Some(i);
                best_dist = dist;
            }
        }
        best_idx
    }

    #[inline]
    fn exact_solution_sig64(sol: &Solution) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for (j, ops) in sol.job_schedule.iter().enumerate() {
            for (o, (m, t)) in ops.iter().enumerate() {
                std::hash::Hash::hash(&(j, o, *m, *t), &mut hasher);
            }
        }
        std::hash::Hasher::finish(&hasher)
    }

    #[inline]
    fn unique_deterministic_phase_indices(top: &[(Solution, u32)], limit: usize) -> Vec<usize> {
        let lim = limit.min(top.len());
        let mut best_by_sig: std::collections::HashMap<u64, usize> =
            std::collections::HashMap::with_capacity(lim.saturating_mul(2).max(1));
        for i in 0..lim {
            let sig = exact_solution_sig64(&top[i].0);
            match best_by_sig.get_mut(&sig) {
                Some(best_i) => {
                    if top[i].1 < top[*best_i].1 {
                        *best_i = i;
                    }
                }
                None => {
                    best_by_sig.insert(sig, i);
                }
            }
        }
        let mut out: Vec<usize> = best_by_sig.into_values().collect();
        out.sort_by(|&a, &b| top[a].1.cmp(&top[b].1).then_with(|| a.cmp(&b)));
        out
    }


    fn maybe_add_elite(elite: &mut Vec<EliteParams>, cand: EliteParams, cap: usize) {
        if elite.is_empty() { elite.push(cand); return; }
        if elite.len() < cap { elite.push(cand); normalize_elite(elite, cap); return; }
        normalize_elite(elite, cap);
        let worst = elite.last().map(|e| e.score).unwrap_or(u32::MAX);
        if cand.score < worst {
            if let Some(last) = elite.last_mut() { *last = cand; } else { elite.push(cand); }
            normalize_elite(elite, cap);
        }
    }

    fn commit_best(save_solution: &dyn Fn(&Solution) -> Result<()>, best_mk: &mut u32, best_sol: &mut Option<Solution>, sol: &Solution, mk: u32) -> Result<bool> {
        if mk < *best_mk { *best_mk = mk; *best_sol = Some(sol.clone()); save_solution(sol)?; Ok(true) } else { Ok(false) }
    }

    fn cached_cbm(pre: &Pre, challenge: &Challenge, sol: &Solution, p1: usize, p2: usize, p3: usize, cache: &mut DetCache) -> Result<Option<(Solution, u32)>> {
        let key = (exact_solution_sig64(sol), p1, p2, p3, 0u8);
        if let Some(hit) = cache.get(&key) { return Ok(hit.clone()); }
        let res = critical_block_move_local_search_ex(pre, challenge, sol, p1, p2, p3)?;
        if cache.len() >= 1024 { cache.clear(); }
        cache.insert(key, res.clone());
        Ok(res)
    }

    fn cached_gr(pre: &Pre, challenge: &Challenge, sol: &Solution, cache: &mut DetCache) -> Result<Option<(Solution, u32)>> {
        let key = (exact_solution_sig64(sol), 0usize, 0usize, 0usize, 1u8);
        if let Some(hit) = cache.get(&key) { return Ok(hit.clone()); }
        let res = greedy_reassign_pass(pre, challenge, sol)?;
        if cache.len() >= 1024 { cache.clear(); }
        cache.insert(key, res.clone());
        Ok(res)
    }

    fn maybe_intensify_ls(pre: &Pre, challenge: &Challenge, rng: &mut SmallRng, sol: &Solution, mk: u32, best_mk: u32, target_margin: u32, stuck: usize, late: bool, cache: &mut DetCache) -> Result<Option<(Solution, u32)>> {
        let flex = (pre.high_flex + pre.jobshopness).clamp(0.0, 1.5);
        let near_best = mk <= best_mk.saturating_add((target_margin / 3).max(1));
        let very_near_best = mk <= best_mk.saturating_add((target_margin / 6).max(1));
        let do_ls = if mk < best_mk { late || stuck > 20 || flex >= 0.12 || rng.gen::<f64>() < 0.55 }
            else if very_near_best && (late || stuck > 80) { rng.gen::<f64>() < (0.05 + 0.05 * flex).clamp(0.04, 0.11) }
            else if near_best && stuck > 140 { rng.gen::<f64>() < (0.035 + 0.045 * flex).clamp(0.03, 0.085) }
            else { false };
        if !do_ls { return Ok(None); }
        let (p1, p2, p3) = if mk < best_mk { let bump = if flex > 0.60 { 1.0 } else { 0.0 }; (38 + (6.0 * bump) as usize, 60 + (10.0 * bump) as usize, 12) }
            else if stuck > 180 { (30, 48, 10) } else { (24, 36, 8) };
        cached_cbm(pre, challenge, sol, p1, p2, p3, cache)
    }

    fn maybe_escape_ls(
        pre: &Pre,
        challenge: &Challenge,
        rng: &mut SmallRng,
        top_solutions: &[(Solution, u32)],
        best_sol: &Solution,
        scan: usize,
        stuck: usize,
        flex01: f64,
        cache: &mut DetCache,
    ) -> Result<Option<(Solution, u32)>> {
        if top_solutions.is_empty() || stuck < 60 {
            return Ok(None);
        }
        let p = (0.040 + 0.060 * flex01 + 0.040 * ((stuck as f64) / 160.0).clamp(0.0, 1.0)).clamp(0.04, 0.14);
        if rng.gen::<f64>() >= p {
            return Ok(None);
        }

        let idx = pick_diverse_top_solution(best_sol, top_solutions, scan)
            .unwrap_or_else(|| pick_top_idx(rng, top_solutions));
        let base = &top_solutions[idx].0;

        let bump = if flex01 > 0.55 { 1.0 } else { 0.0 };
        let p1 = (34.0 + 8.0 * bump) as usize;
        let p2 = (56.0 + 10.0 * bump) as usize;
        let p3 = (10.0 + 2.0 * bump) as usize;
        cached_cbm(pre, challenge, base, p1, p2, p3, cache)
    }

    fn greedy_reassign_pass(pre: &Pre, challenge: &Challenge, base_sol: &Solution) -> Result<Option<(Solution, u32)>> {
        let mut ds=build_disj_from_solution(pre,challenge,base_sol)?; let mut buf=EvalBuf::new(ds.n); let n=ds.n;
        let Some((mut current_mk,_))=eval_disj(&ds,&mut buf) else{return Ok(None)};
        let initial_mk=current_mk; let mut improved=true; let mut passes=0; let max_passes=3;
        while improved&&passes<max_passes {
            improved=false; passes+=1;
            for node in 0..n {
                let job=ds.node_job[node]; let op_idx=ds.node_op[node]; let product=pre.job_products[job];
                let op_info=&pre.product_ops[product][op_idx]; if op_info.machines.len()<=1{continue;}
                let cur_machine=ds.node_machine[node]; let cur_pt=ds.node_pt[node];
                let old_pos=match ds.machine_seq[cur_machine].iter().position(|&x|x==node){Some(p)=>p,None=>continue};
                let mut best_m=cur_machine; let mut best_pt=cur_pt; let mut best_mk=current_mk; let mut best_ins_pos=0usize;

                {
                    let seq=&mut ds.machine_seq[cur_machine];
                    seq[old_pos..].rotate_left(1);
                    seq.pop();
                }

                for &(new_m,new_pt) in &op_info.machines {
                    if new_m==cur_machine{continue;}
                    ds.node_machine[node]=new_m; ds.node_pt[node]=new_pt;
                    let target_len=ds.machine_seq[new_m].len(); let cur_start=buf.start[node]; let mut sorted_pos=target_len;
                    for (k,&nd) in ds.machine_seq[new_m].iter().enumerate(){if buf.start[nd]>=cur_start{sorted_pos=k;break;}}

                    let pos0=sorted_pos;
                    {
                        let seq=&mut ds.machine_seq[new_m];
                        seq.push(node);
                        seq[pos0..].rotate_right(1);
                    }
                    if let Some((test_mk,_))=eval_disj(&ds,&mut buf){if test_mk<best_mk{best_mk=test_mk;best_m=new_m;best_pt=new_pt;best_ins_pos=pos0;}}
                    {
                        let seq=&mut ds.machine_seq[new_m];
                        seq[pos0..].rotate_left(1);
                        seq.pop();
                    }

                    let pos1=sorted_pos.saturating_sub(1);
                    if pos1!=pos0&&pos1<=target_len{
                        {
                            let seq=&mut ds.machine_seq[new_m];
                            seq.push(node);
                            seq[pos1..].rotate_right(1);
                        }
                        if let Some((test_mk,_))=eval_disj(&ds,&mut buf){if test_mk<best_mk{best_mk=test_mk;best_m=new_m;best_pt=new_pt;best_ins_pos=pos1;}}
                        {
                            let seq=&mut ds.machine_seq[new_m];
                            seq[pos1..].rotate_left(1);
                            seq.pop();
                        }
                    }

                    let pos2=target_len;
                    if pos2!=pos0&&pos2!=pos1&&pos2<=target_len{
                        {
                            let seq=&mut ds.machine_seq[new_m];
                            seq.push(node);
                            seq[pos2..].rotate_right(1);
                        }
                        if let Some((test_mk,_))=eval_disj(&ds,&mut buf){if test_mk<best_mk{best_mk=test_mk;best_m=new_m;best_pt=new_pt;best_ins_pos=pos2;}}
                        {
                            let seq=&mut ds.machine_seq[new_m];
                            seq[pos2..].rotate_left(1);
                            seq.pop();
                        }
                    }
                }

                if best_m!=cur_machine {
                    let ins=best_ins_pos.min(ds.machine_seq[best_m].len());
                    {
                        let seq=&mut ds.machine_seq[best_m];
                        seq.push(node);
                        seq[ins..].rotate_right(1);
                    }
                    ds.node_machine[node]=best_m; ds.node_pt[node]=best_pt;
                    current_mk=best_mk; improved=true;
                } else {
                    let seq=&mut ds.machine_seq[cur_machine];
                    seq.push(node);
                    seq[old_pos..].rotate_right(1);
                    ds.node_machine[node]=cur_machine; ds.node_pt[node]=cur_pt;
                }
            }
        }
        if current_mk>=initial_mk{return Ok(None);}
        let Some((_,_))=eval_disj(&ds,&mut buf) else{return Ok(None)};
        let sol=disj_to_solution(pre,&ds,&buf.start)?; Ok(Some((sol,current_mk)))
    }

    pub fn solve(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        pre: &Pre,
        effort: &EffortConfig,
    ) -> Result<()> {
        let (greedy_sol, greedy_mk) = run_simple_greedy_baseline(challenge)?;
        save_solution(&greedy_sol)?;
        let mut cache: DetCache = HashMap::with_hasher(seeded_hasher(&challenge.seed));
        let mut rng = SmallRng::from_seed(challenge.seed);
        let horizon = pre.machine_load0.iter().cloned().fold(0.0f64, f64::max).max(pre.horizon);
        let time_scale = (horizon * (2.65 + 0.15 * pre.load_cv + 0.10 * pre.jobshopness + 0.10 * pre.high_flex)).max(1.0);
        let rules: Vec<Rule> = vec![Rule::Adaptive,Rule::BnHeavy,Rule::EndTight,Rule::CriticalPath,Rule::MostWork,Rule::LeastFlex,Rule::Regret,Rule::ShortestProc,Rule::FlexBalance];
        let flex01 = (pre.high_flex + pre.jobshopness).clamp(0.0, 1.0);
        let mut best_makespan = greedy_mk;
        let mut best_solution: Option<Solution> = Some(greedy_sol.clone());
        let mut top_solutions: Vec<(Solution,u32)> = Vec::new();
        push_top_solutions(&mut top_solutions, &greedy_sol, greedy_mk, 15);
        let target_margin: u32 = ((pre.avg_op_min*(0.9+0.9*pre.high_flex+0.6*pre.jobshopness)).max(1.0)) as u32;
        let route_w_base: f64 = (0.040+0.10*pre.high_flex+0.08*pre.jobshopness).clamp(0.04,0.22);

        if pre.flow_route.is_some()&&pre.flow_pt_by_job.is_some() {
            if let Ok((sol,mk))=neh_reentrant_flow_solution(pre,challenge.num_jobs,challenge.num_machines) {
                commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol,mk)?;
                push_top_solutions(&mut top_solutions,&sol,mk,15);
            }
        }

        let mut ranked: Vec<(Rule,u32,Solution)>=Vec::with_capacity(rules.len());
        for &rule in &rules {
            let (sol,mk)=construct_solution_conflict(challenge,pre,rule,0,None,&mut rng,None,None,None,0.0,horizon,time_scale)?;
            commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol,mk)?;
            push_top_solutions(&mut top_solutions,&sol,mk,15); ranked.push((rule,mk,sol));
        }
        ranked.sort_by_key(|x|x.1);
        let r0=ranked[0].0; let r1=ranked.get(1).map(|x|x.0).unwrap_or(r0); let r2=ranked.get(2).map(|x|x.0).unwrap_or(r1);
        let mut rule_best: Vec<u32>=vec![u32::MAX;10]; let mut rule_tries: Vec<u32>=vec![0u32;10];
        for (rr,mk,_) in &ranked{let idx=rule_idx(*rr);rule_best[idx]=rule_best[idx].min(*mk);rule_tries[idx]=rule_tries[idx].saturating_add(1);}

        let elite_cap: usize = (6usize + (2.0 * flex01).round() as usize).clamp(6, 8);
        let mut elite: Vec<EliteParams> = Vec::new();
        for i in 0..ranked.len().min(3) {
            let sol=&ranked[i].2; let mk=ranked[i].1;
            let jb=job_bias_from_solution(pre,sol)?; let mp=machine_penalty_from_solution(pre,sol,challenge.num_machines)?; let rp=route_pref_from_solution_lite(pre,sol,challenge)?;
            elite.push(EliteParams{jb,mp,rp,score:mk});
        }
        {
            let jb=job_bias_from_solution(pre,&greedy_sol)?; let mp=machine_penalty_from_solution(pre,&greedy_sol,challenge.num_machines)?; let rp=route_pref_from_solution_lite(pre,&greedy_sol,challenge)?;
            elite.push(EliteParams{jb,mp,rp,score:greedy_mk});
        }
        normalize_elite(&mut elite, elite_cap);

        let num_restarts=effort.hybrid_flow_shop_iters;
        let k_hi=6usize;
        let mut stuck: usize=0;
        let mut escape_cooldown: usize=0;

        for r in 0..num_restarts {
            if escape_cooldown > 0 { escape_cooldown -= 1; }

            if escape_cooldown == 0 {
                if let Some((sol2,mk2)) = maybe_escape_ls(pre,challenge,&mut rng,&top_solutions,best_solution.as_ref().unwrap(),top_solutions.len().min(15),stuck,flex01,&mut cache)? {
                    escape_cooldown = 14;
                    let improved = commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol2,mk2)?;
                    if improved {
                        stuck = 0;
                        let jb=job_bias_from_solution(pre,&sol2)?; let mp=machine_penalty_from_solution(pre,&sol2,challenge.num_machines)?; let rp=route_pref_from_solution_lite(pre,&sol2,challenge)?;
                        maybe_add_elite(&mut elite,EliteParams{jb,mp,rp,score:mk2},elite_cap);
                    } else { stuck = stuck.saturating_add(1); }
                    push_top_solutions(&mut top_solutions,&sol2,mk2,15);
                    continue;
                } else if stuck > 150 { escape_cooldown = 6; }
            }

            let late = r >= (num_restarts*2)/3;
            let (k_min,k_max) = if stuck>170{(4usize,6usize)}else if stuck>90{(3usize,6usize)}else if stuck>35{(2usize,6usize)}else{(2usize,4usize)};

            let rule = if r < 35 {
                let u: f64=rng.gen();
                if u<0.11{Rule::FlexBalance}else if u<0.18{Rule::ShortestProc}else if u<0.50{r0}else if u<0.75{r1}else if u<0.90{r2}else{rules[rng.gen_range(0..rules.len())]}
            } else {
                choose_rule_bandit(&mut rng,&rules,&rule_best,&rule_tries,best_makespan,target_margin,stuck,false,late)
            };

            let k = if k_max<=k_min{k_min}else if stuck>120&&rng.gen::<f64>()<0.55{k_max}else{rng.gen_range(k_min..=k_max)}.min(k_hi);
            let learn_base = (0.09+0.24*pre.jobshopness+0.20*pre.high_flex).clamp(0.06,0.44);
            let learn_boost = (1.0+0.38*((stuck as f64)/120.0).clamp(0.0,1.0)).clamp(1.0,1.38);
            let learn_p = (learn_base*learn_boost).clamp(0.0,0.65);

            if stuck > 80 && !top_solutions.is_empty() && rng.gen::<f64>() < 0.04 {
                let idx=pick_top_idx(&mut rng,&top_solutions); let (sref,mkref)=(&top_solutions[idx].0,top_solutions[idx].1);
                let jb=job_bias_from_solution(pre,sref)?; let mp=machine_penalty_from_solution(pre,sref,challenge.num_machines)?; let rp=route_pref_from_solution_lite(pre,sref,challenge)?;
                maybe_add_elite(&mut elite,EliteParams{jb,mp,rp,score:mkref},elite_cap);
            }

            let use_learn = !elite.is_empty() && rng.gen::<f64>() < learn_p;
            let target = if best_makespan<(u32::MAX/2){Some(best_makespan.saturating_add(target_margin))}else{None};

            let (mut sol, mut mk) = if use_learn {
                let mix_p=(0.055+0.10*pre.high_flex+0.09*pre.jobshopness+0.16*((stuck as f64)/160.0).clamp(0.0,1.0)).clamp(0.05,0.40);
                let base_idx=pick_elite_idx(&mut rng,&elite); let mut mp_idx=base_idx; let mut rp_idx=base_idx;
                if elite.len()>1&&rng.gen::<f64>()<mix_p{mp_idx=pick_elite_idx(&mut rng,&elite);}
                if elite.len()>1&&rng.gen::<f64>()<mix_p{rp_idx=pick_elite_idx(&mut rng,&elite);}
                let drop_mp_p=(0.030+0.060*pre.high_flex).clamp(0.03,0.10); let drop_rp_p=(0.030+0.070*pre.jobshopness).clamp(0.03,0.12);
                let mp_opt=if rng.gen::<f64>()<drop_mp_p{None}else{Some(&elite[mp_idx].mp)};
                let rp_opt=if rng.gen::<f64>()<drop_rp_p{None}else{Some(&elite[rp_idx].rp)};
                let jitter=(0.80+0.70*rng.gen::<f64>()).clamp(0.65,1.55);
                let route_w=if rp_opt.is_some(){(route_w_base*jitter).clamp(route_w_base*0.55,0.45)}else{0.0};
                construct_solution_conflict(challenge,pre,rule,k,target,&mut rng,Some(&elite[base_idx].jb),mp_opt.map(|v|&**v),rp_opt,route_w,horizon,time_scale)?
            } else {
                construct_solution_conflict(challenge,pre,rule,k,target,&mut rng,None,None,None,0.0,horizon,time_scale)?
            };

            if let Some((sol2,mk2))=maybe_intensify_ls(pre,challenge,&mut rng,&sol,mk,best_makespan,target_margin,stuck,late,&mut cache)?{sol=sol2;mk=mk2;}

            let ridx=rule_idx(rule); rule_tries[ridx]=rule_tries[ridx].saturating_add(1); rule_best[ridx]=rule_best[ridx].min(mk);
            let improved=commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol,mk)?;

            if improved {
                stuck=0;
                let jb=job_bias_from_solution(pre,&sol)?; let mp=machine_penalty_from_solution(pre,&sol,challenge.num_machines)?; let rp=route_pref_from_solution_lite(pre,&sol,challenge)?;
                maybe_add_elite(&mut elite,EliteParams{jb,mp,rp,score:mk},elite_cap);
            } else {
                stuck=stuck.saturating_add(1);
                let add_p=(0.075+0.025*flex01).clamp(0.07,0.11);
                if mk<=best_makespan.saturating_add(target_margin/2)&&rng.gen::<f64>()<add_p {
                    let jb=job_bias_from_solution(pre,&sol)?; let mp=machine_penalty_from_solution(pre,&sol,challenge.num_machines)?; let rp=route_pref_from_solution_lite(pre,&sol,challenge)?;
                    maybe_add_elite(&mut elite,EliteParams{jb,mp,rp,score:mk},elite_cap);
                }
            }
            push_top_solutions(&mut top_solutions,&sol,mk,15);
        }

        let route_w_ls: f64=(route_w_base*1.40).clamp(route_w_base,0.40);
        let mut refine_results: Vec<(Solution,u32)>=Vec::new();
        let mut refine_cache: std::collections::HashMap<u64, (Vec<f64>, Vec<f64>, RoutePrefLite)> =
            std::collections::HashMap::with_capacity(top_solutions.len().saturating_mul(2).max(1));
        for (base_sol,_) in top_solutions.iter() {
            let sig=exact_solution_sig64(base_sol);
            if !refine_cache.contains_key(&sig) {
                let jb=job_bias_from_solution(pre,base_sol)?;
                let mp_base=machine_penalty_from_solution(pre,base_sol,challenge.num_machines)?;
                let rp_base=route_pref_from_solution_lite(pre,base_sol,challenge)?;
                refine_cache.insert(sig,(jb,mp_base,rp_base));
            }
            let (jb, mp_base, rp_base): (&[f64], &Vec<f64>, &RoutePrefLite) = {
                let cached=refine_cache.get(&sig).unwrap();
                (&cached.0, &cached.1, &cached.2)
            };
            let target_ls=if best_makespan<(u32::MAX/2){Some(best_makespan.saturating_add(target_margin/2))}else{None};
            let mix_ref_p=(0.045+0.10*pre.high_flex+0.09*pre.jobshopness).clamp(0.04,0.22);
            for attempt in 0..10 {
                let rule=match attempt{0=>r0,1=>Rule::Adaptive,2=>Rule::BnHeavy,3=>Rule::EndTight,4=>Rule::Regret,5=>Rule::CriticalPath,6=>Rule::LeastFlex,7=>Rule::MostWork,8=>Rule::FlexBalance,_=>Rule::ShortestProc};
                let k=match attempt%6{0=>2,1=>3,2=>4,3=>5,4=>3,_=>2}.min(k_hi);
                let mut mp_ref: Option<&Vec<f64>>=Some(mp_base); let mut rp_ref: Option<&RoutePrefLite>=Some(rp_base);
                if !elite.is_empty()&&rng.gen::<f64>()<mix_ref_p {
                    let eidx=pick_elite_idx(&mut rng,&elite);
                    if rng.gen::<f64>()<0.62{mp_ref=Some(&elite[eidx].mp);}
                    if rng.gen::<f64>()<0.72{rp_ref=Some(&elite[eidx].rp);}
                    if rng.gen::<f64>()<0.055{rp_ref=None;}
                }
                let rw_j=if rp_ref.is_some(){(route_w_ls*(0.86+0.50*rng.gen::<f64>())).clamp(route_w_ls*0.70,0.45)}else{0.0};
                let (mut sol,mut mk)=construct_solution_conflict(challenge,pre,rule,k,target_ls,&mut rng,Some(jb),mp_ref.map(|v|&**v),rp_ref,rw_j,horizon,time_scale)?;
                if let Some((sol2,mk2))=maybe_intensify_ls(pre,challenge,&mut rng,&sol,mk,best_makespan,target_margin,attempt,true,&mut cache)?{sol=sol2;mk=mk2;}
                if commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol,mk)? {
                    let jb2=job_bias_from_solution(pre,&sol)?; let mp2=machine_penalty_from_solution(pre,&sol,challenge.num_machines)?; let rp2=route_pref_from_solution_lite(pre,&sol,challenge)?;
                    maybe_add_elite(&mut elite,EliteParams{jb:jb2,mp:mp2,rp:rp2,score:mk},elite_cap);
                }
                refine_results.push((sol,mk));
            }
        }
        for (sol,mk) in refine_results { push_top_solutions(&mut top_solutions,&sol,mk,15); }

        let ls_base_indices=unique_deterministic_phase_indices(&top_solutions, top_solutions.len().min(15));
        for &i in &ls_base_indices {
            let base_sol=&top_solutions[i].0;
            if let Some((sol2,mk2))=cached_cbm(pre,challenge,base_sol,40,64,12,&mut cache)?{
                commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol2,mk2)?;
                push_top_solutions(&mut top_solutions,&sol2,mk2,15);
            }
        }

        if let Some(ref sol)=best_solution.clone() {
            if pre.high_flex+pre.jobshopness > 0.55 {
                if let Some((sol2,mk2))=cached_cbm(pre,challenge,sol,50,80,14,&mut cache)?{
                    commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol2,mk2)?;
                }
            }
        }

        if let Some(ref sol)=best_solution.clone() {
            let sig_best_exact=exact_solution_sig64(sol);
            let sig_best_div=solution_sig64(sol);
            let greedy_base_indices=unique_deterministic_phase_indices(&top_solutions, top_solutions.len().min(15));

            let mut base2: Option<&Solution> = None;
            let mut base2_dist: u32 = 0;
            for &idx in &greedy_base_indices {
                let cand=&top_solutions[idx].0;
                if exact_solution_sig64(cand)==sig_best_exact { continue; }
                let dist=(sig_best_div ^ solution_sig64(cand)).count_ones();
                if base2.is_none() || dist > base2_dist {
                    base2=Some(cand);
                    base2_dist=dist;
                }
            }

            let mut best_improved: Option<(Solution, u32)> = None;

            if let Ok(Some((sol2, mk2))) = cached_gr(pre, challenge, sol, &mut cache) {
                if mk2 < best_makespan { best_improved = Some((sol2, mk2)); }
            }

            if let Some(b2) = base2 {
                if let Ok(Some((sol2, mk2))) = cached_gr(pre, challenge, b2, &mut cache) {
                    if mk2 < best_makespan && best_improved.as_ref().map_or(true, |x| mk2 < x.1) {
                        best_improved = Some((sol2, mk2));
                    }
                }
            }

            if let Some((sol2, _mk2)) = best_improved {
                best_solution = Some(sol2.clone());
                save_solution(&sol2)?;
            }
        }

        if let Some(sol)=best_solution { save_solution(&sol)?; }

        Ok(())
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
        let r = rp[product][op_idx]; let mu = machine.min(255) as u8;
        if mu == r.best_m { (r.best_w as f64) / 255.0 } else if mu == r.second_m { (r.second_w as f64) / 255.0 } else { 0.0 }
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
        let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop = pre.machine_best_pop[machine]; (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
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
        let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop = pre.machine_best_pop[machine]; (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
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
        let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop = pre.machine_best_pop[machine]; (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
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
        let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop = pre.machine_best_pop[machine]; (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
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
        let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop = pre.machine_best_pop[machine]; (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
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
        let scar_n = pre.machine_scarcity[machine] / pre.avg_machine_scarcity.max(1e-9);
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
        let avg_flex_inv = 1.0 / pre.flex_avg.max(1.0); let scarce_match = scar_n * (flex_inv - avg_flex_inv);
        let mpen = machine_penalty.clamp(0.0, 1.0); let mpen_gain = 1.0 + 0.85*pre.high_flex;
        let flow_term = pre.flow_w * pre.job_flow_pref[job] * (0.65 + 0.70*(1.0-progress));
        let slack_u = slack_urgency_js(pre, target_mk, time, product, op_idx);
        let slack_w = pre.slack_base * (0.25 + 0.75*progress); let slack_reg_boost = 1.0 + 0.40*reg_n*progress;
        let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop = pre.machine_best_pop[machine]; (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
        let route_gain = (0.70 + 0.80*(1.0-progress)).clamp(0.70, 1.40);
        let route_term = if route_w > 0.0 && op.flex >= 2 { route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine) } else { 0.0 };
        let end_w=1.10+1.00*progress+0.35*pre.high_flex; let cp_w=1.15+0.30*js; let reg_w=(0.55+0.20*(1.0-progress))*(0.85+0.60*js); let mpen_w=(0.10+0.45*pre.high_flex)*pre.flex_factor; let next_term=next_w_base*(0.45+0.55*js)*next_term_raw; let slack_term=slack_w*(0.70+0.40*js)*slack_u*slack_reg_boost;
        (cp_w*rem_min_n)+0.12*rem_avg_n+0.08*ops_n+0.18*scarcity_urg+(0.30*pre.flex_factor)*flex_inv+(0.20*pre.flex_factor)*scarce_match+(reg_w*pre.flex_factor)*reg_n+next_term+slack_term-end_w*end_n-0.22*proc_n-pop_pen-(mpen_gain*mpen_w)*mpen+0.55*job_bias+flow_term+route_term+jitter
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
        let load_n = dynamic_load / pre.avg_machine_load.max(1e-9); let scar_n = pre.machine_scarcity[machine] / pre.avg_machine_scarcity.max(1e-9);
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
        let avg_flex_inv = 1.0 / pre.flex_avg.max(1.0); let scarce_match = scar_n * (flex_inv - avg_flex_inv);
        let mpen = machine_penalty.clamp(0.0, 1.0); let mpen_gain = 1.0 + 0.85*pre.high_flex;
        let flow_term = pre.flow_w * pre.job_flow_pref[job] * (0.65 + 0.70*(1.0-progress));
        let slack_u = slack_urgency_js(pre, target_mk, time, product, op_idx);
        let slack_w = pre.slack_base * (0.25 + 0.75*progress); let slack_reg_boost = 1.0 + 0.40*reg_n*progress;
        let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop = pre.machine_best_pop[machine]; (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
        let route_gain = (0.70 + 0.80*(1.0-progress)).clamp(0.70, 1.40);
        let route_term = if route_w > 0.0 && op.flex >= 2 { route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine) } else { 0.0 };
        let bn_w=(0.90+0.55*js)*pre.bn_focus; let end_w=0.65+0.70*progress; let reg_w=(0.60+0.25*(1.0-progress))*(0.85+0.35*js); let load_w=if pre.hi_flex{-0.35}else{0.55+0.25*js}; let mpen_w=(0.12+0.30*js)*pre.flex_factor*(0.95+0.65*pre.high_flex); let next_term=next_w_base*(0.55+0.75*js)*next_term_raw; let slack_term=slack_w*(0.45+0.55*js)*slack_u*slack_reg_boost;
        (0.95*rem_min_n)+(0.30*rem_avg_n)+(bn_w*bn_n)+(0.22*density_n)+(0.10*ops_n)+(0.65*pre.flex_factor)*flex_inv+(0.35*pre.flex_factor)*scarce_match+load_w*pre.flex_factor*load_n+(reg_w*pre.flex_factor)*reg_n+0.18*scarcity_urg+next_term+slack_term-end_w*end_n-0.18*proc_n-pop_pen-(mpen_gain*mpen_w)*mpen+0.60*job_bias+flow_term+route_term+jitter
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
        let load_n = dynamic_load / pre.avg_machine_load.max(1e-9); let scar_n = pre.machine_scarcity[machine] / pre.avg_machine_scarcity.max(1e-9);
        let end_n = (best_end as f64) / pre.time_scale.max(1.0); let proc_n = (pt as f64) / pre.avg_op_min.max(1.0);
        let regret = if second_end >= INF { pre.avg_op_min * 2.6 } else { (second_end - best_end) as f64 };
        let reg_n = (regret / pre.avg_op_min.max(1.0)).clamp(0.0, 6.0);
        let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
        let density_n = ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0);
        let next_min = pre.product_next_min[product][op_idx] as f64; let next_min_n = next_min / pre.horizon.max(1.0);
        let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
        let p2 = progress * progress; let next_w_base = 0.12 + p2 * 0.28;
        let next_term_raw = (0.55*next_min_n + 0.45*next_flex_inv) * (1.0 + 0.30*density_n*pre.high_flex);
        let js = pre.jobshopness; let fl = 1.0 - js;
        let avg_flex_inv = 1.0 / pre.flex_avg.max(1.0); let scarce_match = scar_n * (flex_inv - avg_flex_inv);
        let mpen = machine_penalty.clamp(0.0, 1.0); let mpen_gain = 1.0 + 0.85*pre.high_flex;
        let flow_term = pre.flow_w * pre.job_flow_pref[job] * (0.65 + 0.70*(1.0-progress));
        let slack_u = slack_urgency_js(pre, target_mk, time, product, op_idx);
        let slack_w = pre.slack_base * (0.25 + 0.75*progress); let slack_reg_boost = 1.0 + 0.40*reg_n*progress;
        let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop = pre.machine_best_pop[machine]; (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
        let route_gain = (0.70 + 0.80*(1.0-progress)).clamp(0.70, 1.40);
        let route_term = if route_w > 0.0 && op.flex >= 2 { route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine) } else { 0.0 };
        let end_w=(0.90*fl+0.72*js)+(0.62+0.12*fl)*progress+0.18*pre.high_flex; let reg_w=(0.50*fl+0.78*js)+0.18*(1.0-progress); let bn_w=((0.45+0.40*js)+0.25*(1.0-progress))*pre.bn_focus; let load_sign=if pre.hi_flex{-1.0}else{1.0}; let load_w=load_sign*(0.45*fl+0.75*js)*pre.flex_factor; let density_w=0.08*fl+0.20*js; let next_term=next_w_base*(0.50*fl+1.50*js)*next_term_raw; let mpen_w=(0.08*fl+0.28*js)*pre.flex_factor*(1.0+0.85*pre.high_flex); let slack_term=slack_w*(0.55*fl+0.85*js)*slack_u*slack_reg_boost;
        (1.05*rem_min_n)+(0.48*rem_avg_n)+(bn_w*bn_n)+density_w*density_n+(0.08*ops_n)+(0.62*pre.flex_factor)*flex_inv+(0.55*pre.flex_factor)*scarce_match+load_w*load_n+(reg_w*pre.flex_factor)*reg_n+0.20*pre.flex_factor*scarcity_urg+next_term+slack_term-end_w*end_n-(0.18*fl+0.12*js)*proc_n-pop_pen-(mpen_gain*mpen_w)*mpen+(0.62+0.06*js)*job_bias+flow_term+route_term+jitter
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
        let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop = pre.machine_best_pop[machine]; (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
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

    fn construct_solution_conflict(
        challenge: &Challenge, pre: &Pre, rule: Rule, k: usize, target_mk: Option<u32>,
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
        let chaotic_like = pre.chaotic_like;
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
        let score_candidate_for_rule: fn(&Pre, usize, usize, usize, usize, &OpInfo, usize, u32, u32, Option<u32>, u32, u32, usize, f64, f64, f64, f64, Option<&RoutePrefLite>, f64, f64) -> f64 = match rule {
            Rule::CriticalPath => score_candidate_critical_path,
            Rule::MostWork => score_candidate_most_work,
            Rule::LeastFlex => score_candidate_least_flex,
            Rule::ShortestProc => score_candidate_shortest_proc,
            Rule::Regret => score_candidate_regret,
            Rule::EndTight => score_candidate_end_tight,
            Rule::BnHeavy => score_candidate_bn_heavy,
            Rule::Adaptive => score_candidate_adaptive,
            Rule::FlexBalance => score_candidate_flex_balance,
        };
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
                            let op = &pre.product_ops[product][op_idx];
                            if op.flex == 0 || op.machines.is_empty() || op.min_pt >= INF {
                                job_best_end[job] = INF;
                                job_second_end[job] = INF;
                                job_best_cnt_total[job] = 0;
                                job_best_cnt_idle[job] = 0;
                            } else {
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
                        let product = pre.job_products[job];
                        let op = &pre.product_ops[product][op_idx];
                        let ops_rem = pre.job_ops_len[job] - op_idx;
                        let jb = job_bias.map(|v| v[job]).unwrap_or(0.0);
                        let mp = machine_penalty.map(|v| v[m]).unwrap_or(0.0);
                        let jitter = if k > 0 { rng.gen::<f64>()*1e-9 } else { 0.0 };
                        let dynamic_load_m = machine_load[m];
                        let base = score_candidate_for_rule(pre, job, product, op_idx, ops_rem, op, m, pt, time, target_mk, job_best_end[job], job_second_end[job], job_best_cnt_total[job], progress, jb, mp, dynamic_load_m, route_pref, route_w, jitter);
                        push_top_k_raw(&mut raw_by_machine[m], RawCand { job, machine: m, pt, base_score: base, rigidity: job_rigidity[job], reg_n: job_regn[job] }, cap_per_machine);
                    }
                    list.truncate(write);
                }
                touched_machines.sort_unstable();
                let denom = (idle_machines.len() as f64).max(1.0);
                let (conflict_w, conflict_scale) = if chaotic_like { (-(0.05+0.08*(1.0-progress)).clamp(0.04,0.14), (0.95+0.20*pre.flex_factor).clamp(0.90,1.20)) } else { ((0.09+0.26*pre.jobshopness+0.11*pre.high_flex+0.16*(1.0-progress)).clamp(0.05,0.45), (0.90+0.40*pre.flex_factor).clamp(0.85,1.75)) };
                let (bal_w, avg_work) = if chaotic_like { ((0.030+0.070*(1.0-progress)).clamp(0.025,0.11), (sum_work as f64)/(num_machines as f64).max(1.0)) } else { (0.0, 0.0) };
                let mut best: Option<Cand> = None; top.clear();
                for &m in &touched_machines {
                    let dem = demand[m] as f64; if dem <= 0.0 || raw_by_machine[m].is_empty() { continue; }
                    let dem_n = ((dem-1.0)/denom).clamp(0.0,2.5);
                    let bal_pen = if chaotic_like && bal_w > 0.0 { let denomw=(avg_work+(pre.avg_op_min*3.0).max(1.0)).max(1.0); let r=(machine_work[m] as f64)/denomw; let done_n=(r/(r+1.0)).clamp(0.0,1.0); -bal_w*done_n } else { 0.0 };
                    for rc in &raw_by_machine[m] {
                        let rig=rc.rigidity.clamp(0.0,2.5); let regc=rc.reg_n.clamp(0.0,4.5);
                        let mut boost=conflict_w*conflict_scale*dem_n*(1.15*rig+0.85*regc);
                        if chaotic_like { boost=boost.max(-0.26); }
                        let c = Cand { job: rc.job, machine: rc.machine, pt: rc.pt, score: rc.base_score+boost+bal_pen };
                        if k == 0 { if best.map_or(true, |bb| c.score > bb.score) { best = Some(c); } } else { push_top_k(&mut top, c, k); }
                    }
                }
                let chosen = if k == 0 { match best { Some(c) => c, None => break } } else { if top.is_empty() { break; } choose_from_top_weighted(rng, &top) };
                let job = chosen.job; let machine = chosen.machine; let pt = chosen.pt;
                let product = pre.job_products[job]; let op_idx = job_next_op[job]; let op = &pre.product_ops[product][op_idx];
                let (best_end_now,_,_,_) = best_second_and_counts(time, &machine_avail, op);
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

    pub fn tabu_search_phase(pre: &Pre, challenge: &Challenge, base_sol: &Solution, max_iterations: usize, tenure_base: usize) -> Result<Option<(Solution, u32)>> {
        let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
        let mut buf = EvalBuf::new(ds.n); let n = ds.n;
        let Some((initial_mk, mut mk_node)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
        let mut cur_mk = initial_mk; let mut best_global_mk = initial_mk; let mut best_global_machine_seq = ds.machine_seq.clone();
        let tenure = tenure_base.max(5); let tenure_delta = (tenure/3).max(2); let max_no_improve = (max_iterations/2).max(60);
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
        let mut crit = vec![false; n]; let mut no_improve = 0usize;
        let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (initial_mk as u64).wrapping_shl(16) ^ (n as u64).wrapping_mul(0x517CC1B727220A95);
        let mut tail = vec![0u32; n]; let mut back_deg = vec![0u16; n]; let mut back_stack: Vec<usize> = Vec::with_capacity(n);
        let mut machine_pred_node = vec![NONE_USIZE; n]; let mut job_pred_node = vec![NONE_USIZE; n];
        let mut kick_swaps: Vec<(usize,usize)> = Vec::with_capacity(n);
        let mut crit_machine_stamp = vec![0u32; ds.num_machines];
        let mut crit_min_pos = vec![0usize; ds.num_machines];
        let mut crit_max_pos = vec![0usize; ds.num_machines];
        let mut crit_round: u32 = 1;
        for j in 0..ds.num_jobs { let base = ds.job_offsets[j]; let end = ds.job_offsets[j+1]; for k in (base+1)..end { job_pred_node[k] = k-1; } }
        rebuild_machine_pred_nodes(&ds, &mut machine_pred_node);
        let kick_threshold = (max_no_improve*2/3).max(40); let mut kicks_left = 5usize;
        for iter in 0..max_iterations {
            if no_improve >= max_no_improve {
                if kicks_left == 0 { break; }
                ds.machine_seq.clone_from(&best_global_machine_seq); no_improve = 0; kicks_left -= 1; tabu_expiry.fill(0);
                rebuild_machine_pred_nodes(&ds, &mut machine_pred_node);
                for m in 0..ds.num_machines {
                    for (i, &node) in ds.machine_seq[m].iter().enumerate() {
                        current_pos[node] = i;
                        node_machine[node] = m;
                    }
                }
                let Some((mk, node)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
                cur_mk = mk; mk_node = node;
                continue;
            }
            if no_improve > 0 && no_improve % kick_threshold == 0 && kicks_left > 0 {
                crit.fill(false); let mut u = mk_node; while u != NONE_USIZE { crit[u]=true; u=buf.best_pred[u]; }
                kick_swaps.clear();
                let mut u = mk_node;
                while u != NONE_USIZE {
                    let m = node_machine[u];
                    let pos = current_pos[u];
                    if pos > 0 { kick_swaps.push((m, pos - 1)); }
                    if pos + 1 < ds.machine_seq[m].len() { kick_swaps.push((m, pos)); }
                    u = buf.best_pred[u];
                }
                if !kick_swaps.is_empty() {
                    kick_swaps.sort_unstable();
                    kick_swaps.dedup();
                    let num_kicks = (3 + (no_improve / kick_threshold)).min(5);
                    for _ in 0..num_kicks {
                        pseed^=pseed.wrapping_shl(13); pseed^=pseed.wrapping_shr(7); pseed^=pseed.wrapping_shl(17);
                        let idx=(pseed as usize)%kick_swaps.len(); let (m,pos)=kick_swaps[idx];
                        if pos+1<ds.machine_seq[m].len() {
                            let node_a = ds.machine_seq[m][pos];
                            let node_b = ds.machine_seq[m][pos+1];
                            ds.machine_seq[m].swap(pos,pos+1);
                            current_pos[node_a] = pos + 1;
                            current_pos[node_b] = pos;
                        }
                    }
                }
                kicks_left -= 1;
                rebuild_machine_pred_nodes(&ds, &mut machine_pred_node);
                let Some((mk, node)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
                cur_mk = mk; mk_node = node;
                continue;
            }
            if iter > 0 { if cur_mk < best_global_mk { best_global_mk = cur_mk; best_global_machine_seq.clone_from(&ds.machine_seq); no_improve = 0; } else { no_improve += 1; } }

            tail.fill(0); back_deg.fill(0);
            for i in 0..n { if ds.job_succ[i] != NONE_USIZE { back_deg[i] += 1; } if buf.machine_succ[i] != NONE_USIZE { back_deg[i] += 1; } }
            back_stack.clear(); for i in 0..n { if back_deg[i] == 0 { back_stack.push(i); } }
            while let Some(nd) = back_stack.pop() {
                let contrib = ds.node_pt[nd].saturating_add(tail[nd]);
                let jp = job_pred_node[nd]; if jp != NONE_USIZE { if contrib > tail[jp] { tail[jp] = contrib; } back_deg[jp] = back_deg[jp].saturating_sub(1); if back_deg[jp] == 0 { back_stack.push(jp); } }
                let mp = machine_pred_node[nd]; if mp != NONE_USIZE { if contrib > tail[mp] { tail[mp] = contrib; } back_deg[mp] = back_deg[mp].saturating_sub(1); if back_deg[mp] == 0 { back_stack.push(mp); } }
            }
            crit.fill(false);
            let cur_crit_round = crit_round;
            crit_round = crit_round.wrapping_add(1);
            if crit_round == 0 { crit_machine_stamp.fill(0); crit_round = 1; }
            let mut u = mk_node;
            while u != NONE_USIZE {
                crit[u] = true;
                let m = node_machine[u];
                let pos = current_pos[u];
                if crit_machine_stamp[m] != cur_crit_round {
                    crit_machine_stamp[m] = cur_crit_round;
                    crit_min_pos[m] = pos;
                    crit_max_pos[m] = pos;
                } else {
                    if pos < crit_min_pos[m] { crit_min_pos[m] = pos; }
                    if pos > crit_max_pos[m] { crit_max_pos[m] = pos; }
                }
                u = buf.best_pred[u];
            }
            let mut best_move: Option<(usize,usize,u32)> = None; let mut best_move_mk = u32::MAX;
            let mut fallback_move: Option<(usize,usize,u32)> = None; let mut fallback_mk = u32::MAX;
            for m in 0..ds.num_machines {
                if crit_machine_stamp[m] != cur_crit_round { continue; }
                let seq = &ds.machine_seq[m];
                if seq.is_empty() { continue; }
                let mut active = false;
                let mut run_start = 0usize;
                let mut run_end = 0usize;
                let mut prev_pos = 0usize;
                let mut prev_node = NONE_USIZE;
                for pos in crit_min_pos[m]..=crit_max_pos[m] {
                    let node = seq[pos];
                    if !crit[node] {
                        if active {
                            if run_end > run_start {
                                let block_len = run_end-run_start+1;
                                let mut swap_positions = [run_start,NONE_USIZE]; let num_swaps = if block_len>=3 { swap_positions[1]=run_end-1; 2 } else { 1 };
                                for si in 0..num_swaps {
                                    let pos=swap_positions[si]; if pos+1>=ds.machine_seq[m].len() { continue; }
                                    let node_u=ds.machine_seq[m][pos]; let node_v=ds.machine_seq[m][pos+1];
                                    let est_mk = estimate_swap_mk(node_u, node_v, &buf.start, &tail, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ);
                                    let lu = node_local[node_u]; let lv = node_local[node_v];
                                    let (a, b) = if lu < lv { (lu, lv) } else { (lv, lu) };
                                    let tabu_idx = pair_offsets[m] + b * (b - 1) / 2 + a;
                                    let is_tabu = tabu_expiry[tabu_idx] > iter; let aspiration=est_mk<best_global_mk;
                                    if (!is_tabu||aspiration) && est_mk<best_move_mk { best_move_mk=est_mk; best_move=Some((m,pos,est_mk)); }
                                    if est_mk<fallback_mk { fallback_mk=est_mk; fallback_move=Some((m,pos,est_mk)); }
                                }
                            }
                            active = false;
                        }
                        continue;
                    }
                    if !active {
                        active = true;
                        run_start = pos;
                        run_end = pos;
                        prev_pos = pos;
                        prev_node = node;
                        continue;
                    }
                    if pos == prev_pos + 1 && buf.start[node] == buf.start[prev_node].saturating_add(ds.node_pt[prev_node]) {
                        run_end = pos;
                    } else {
                        if run_end > run_start {
                            let block_len = run_end-run_start+1;
                            let mut swap_positions = [run_start,NONE_USIZE]; let num_swaps = if block_len>=3 { swap_positions[1]=run_end-1; 2 } else { 1 };
                            for si in 0..num_swaps {
                                let pos=swap_positions[si]; if pos+1>=ds.machine_seq[m].len() { continue; }
                                let node_u=ds.machine_seq[m][pos]; let node_v=ds.machine_seq[m][pos+1];
                                let est_mk = estimate_swap_mk(node_u, node_v, &buf.start, &tail, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ);
                                let lu = node_local[node_u]; let lv = node_local[node_v];
                                let (a, b) = if lu < lv { (lu, lv) } else { (lv, lu) };
                                let tabu_idx = pair_offsets[m] + b * (b - 1) / 2 + a;
                                let is_tabu = tabu_expiry[tabu_idx] > iter; let aspiration=est_mk<best_global_mk;
                                if (!is_tabu||aspiration) && est_mk<best_move_mk { best_move_mk=est_mk; best_move=Some((m,pos,est_mk)); }
                                if est_mk<fallback_mk { fallback_mk=est_mk; fallback_move=Some((m,pos,est_mk)); }
                            }
                        }
                        run_start = pos;
                        run_end = pos;
                    }
                    prev_pos = pos;
                    prev_node = node;
                }
                if active && run_end > run_start {
                    let block_len = run_end-run_start+1;
                    let mut swap_positions = [run_start,NONE_USIZE]; let num_swaps = if block_len>=3 { swap_positions[1]=run_end-1; 2 } else { 1 };
                    for si in 0..num_swaps {
                        let pos=swap_positions[si]; if pos+1>=ds.machine_seq[m].len() { continue; }
                        let node_u=ds.machine_seq[m][pos]; let node_v=ds.machine_seq[m][pos+1];
                        let est_mk = estimate_swap_mk(node_u, node_v, &buf.start, &tail, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ);
                        let lu = node_local[node_u]; let lv = node_local[node_v];
                        let (a, b) = if lu < lv { (lu, lv) } else { (lv, lu) };
                        let tabu_idx = pair_offsets[m] + b * (b - 1) / 2 + a;
                        let is_tabu = tabu_expiry[tabu_idx] > iter; let aspiration=est_mk<best_global_mk;
                        if (!is_tabu||aspiration) && est_mk<best_move_mk { best_move_mk=est_mk; best_move=Some((m,pos,est_mk)); }
                        if est_mk<fallback_mk { fallback_mk=est_mk; fallback_move=Some((m,pos,est_mk)); }
                    }
                }
            }
            let chosen = best_move.or(fallback_move);
            match chosen {
                Some((m,pos,_est)) => {
                    let node_a=ds.machine_seq[m][pos]; let node_b=ds.machine_seq[m][pos+1];
                    ds.machine_seq[m].swap(pos,pos+1);
                    current_pos[node_a] = pos + 1;
                    current_pos[node_b] = pos;
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
                    let Some((mk, node)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
                    cur_mk = mk; mk_node = node;
                }
                None => break,
            }
        }
        if cur_mk < best_global_mk { best_global_mk = cur_mk; best_global_machine_seq.clone_from(&ds.machine_seq); }
        if best_global_mk >= initial_mk { return Ok(None); }
        ds.machine_seq = best_global_machine_seq;
        let Some((mk_final,_)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
        let sol = disj_to_solution(pre, &ds, &buf.start)?;
        Ok(Some((sol, mk_final)))
    }

    #[inline]
    fn estimate_swap_mk(u: usize, v: usize, heads: &[u32], tails: &[u32], pt: &[u32], job_pred: &[usize], job_succ: &[usize], machine_pred: &[usize], machine_succ: &[usize]) -> u32 {
        let mp_u=machine_pred[u]; let ms_v=machine_succ[v]; let jp_v=job_pred[v]; let jp_u=job_pred[u]; let js_u=job_succ[u]; let js_v=job_succ[v];
        let r_jp_v=if jp_v!=NONE_USIZE{heads[jp_v].saturating_add(pt[jp_v])}else{0}; let r_mp_u=if mp_u!=NONE_USIZE{heads[mp_u].saturating_add(pt[mp_u])}else{0};
        let new_r_v=r_jp_v.max(r_mp_u); let r_jp_u=if jp_u!=NONE_USIZE{heads[jp_u].saturating_add(pt[jp_u])}else{0}; let new_r_u=r_jp_u.max(new_r_v.saturating_add(pt[v]));
        let q_js_u=if js_u!=NONE_USIZE{pt[js_u].saturating_add(tails[js_u])}else{0}; let q_ms_v=if ms_v!=NONE_USIZE{pt[ms_v].saturating_add(tails[ms_v])}else{0};
        let new_q_u=q_js_u.max(q_ms_v); let q_js_v=if js_v!=NONE_USIZE{pt[js_v].saturating_add(tails[js_v])}else{0}; let new_q_v=q_js_v.max(pt[u].saturating_add(new_q_u));
        let path_v=new_r_v.saturating_add(pt[v]).saturating_add(new_q_v); let path_u=new_r_u.saturating_add(pt[u]).saturating_add(new_q_u);
        path_v.max(path_u)
    }

    #[inline]
    fn lower_bound_machine_swap_jobpath(seq: &[usize], i: usize, j: usize, prefix_pt: &[u32], node_pt: &[u32], job_head_lb: &[u32], job_tail_lb: &[u32]) -> u32 {
        if i >= j || j >= seq.len() { return 0; }
        let node_i = seq[i];
        let node_j = seq[j];
        let pt_i = node_pt[node_i];
        let pt_j = node_pt[node_j];
        let plus = pt_j >= pt_i;
        let diff = if plus { pt_j - pt_i } else { pt_i - pt_j };
        let mut lb = 0u32;

        let start_j = prefix_pt[i].max(job_head_lb[node_j]);
        lb = lb.max(start_j.saturating_add(pt_j).saturating_add(job_tail_lb[node_j]));

        for p in (i + 1)..j {
            let node = seq[p];
            let mach_before = if plus { prefix_pt[p].saturating_add(diff) } else { prefix_pt[p].saturating_sub(diff) };
            let start = mach_before.max(job_head_lb[node]);
            lb = lb.max(start.saturating_add(node_pt[node]).saturating_add(job_tail_lb[node]));
        }

        let mach_before_i = if plus { prefix_pt[j].saturating_add(diff) } else { prefix_pt[j].saturating_sub(diff) };
        let start_i = mach_before_i.max(job_head_lb[node_i]);
        lb.max(start_i.saturating_add(pt_i).saturating_add(job_tail_lb[node_i]))
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
        let mut cur_mk = initial_mk;
        let mut crit = vec![false; n];
        let mut tail = vec![0u32; n];
        let mut back_deg = vec![0u16; n];
        let mut back_stack: Vec<usize> = Vec::with_capacity(n);
        let mut machine_pred_node = vec![NONE_USIZE; n];
        let mut job_pred_node = vec![NONE_USIZE; n];
        let mut moves: Vec<(usize,usize)> = Vec::with_capacity(64);
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
            tail.fill(0);
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
                let contrib = ds.node_pt[nd].saturating_add(tail[nd]);
                let jp = job_pred_node[nd];
                if jp != NONE_USIZE {
                    if contrib > tail[jp] { tail[jp] = contrib; }
                    back_deg[jp] = back_deg[jp].saturating_sub(1);
                    if back_deg[jp] == 0 { back_stack.push(jp); }
                }
                let mp = machine_pred_node[nd];
                if mp != NONE_USIZE {
                    if contrib > tail[mp] { tail[mp] = contrib; }
                    back_deg[mp] = back_deg[mp].saturating_sub(1);
                    if back_deg[mp] == 0 { back_stack.push(mp); }
                }
            }

            crit.fill(false);
            crit_positions.clear();
            let mut u = mk_node;
            while u != NONE_USIZE {
                crit[u] = true;
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
                            for si in 0..num_swaps {
                                let pos = swap_positions[si];
                                if pos + 1 >= seq.len() { continue; }
                                let node_u = seq[pos];
                                let node_v = seq[pos + 1];
                                let est_mk = estimate_swap_mk(node_u, node_v, &buf.start, &tail, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ);
                                if est_mk < cur_mk {
                                    moves.push((m, pos));
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
                    for si in 0..num_swaps {
                        let pos = swap_positions[si];
                        if pos + 1 >= seq.len() { continue; }
                        let node_u = seq[pos];
                        let node_v = seq[pos + 1];
                        let est_mk = estimate_swap_mk(node_u, node_v, &buf.start, &tail, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ);
                        if est_mk < cur_mk {
                            moves.push((m, pos));
                        }
                    }
                }
            }

            if moves.is_empty() { break; }

            let mut improved = false;
            for &(m, pos) in &moves {
                if pos + 1 >= ds.machine_seq[m].len() { continue; }
                let node_a = ds.machine_seq[m][pos];
                let node_b = ds.machine_seq[m][pos + 1];
                ds.machine_seq[m].swap(pos, pos + 1);
                match eval_disj(ds, buf) {
                    Some((new_mk, new_node)) if new_mk < cur_mk => {
                        cur_mk = new_mk;
                        mk_node = new_node;
                        current_pos[node_a] = pos + 1;
                        current_pos[node_b] = pos;
                        stalled = 0;
                        improved = true;
                        break;
                    }
                    _ => {
                        ds.machine_seq[m].swap(pos, pos + 1);
                    }
                }
            }

            if !improved {
                let Some((mk, node)) = eval_disj(ds, buf) else { return None };
                cur_mk = mk;
                mk_node = node;
                stalled += 1;
                if stalled >= stall_cap { break; }
            }
        }

        if cur_mk < initial_mk { Some(cur_mk) } else { None }
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
            let rp = if route_w_ls>0.0 { Some(route_pref_from_solution_lite(pre, base_sol, challenge)?) } else { None };
            let target_ls = if best_makespan < (u32::MAX/2) { Some(best_makespan.saturating_add(target_margin/2)) } else { None };
            for attempt in 0..10 {
                let rule = if pre.chaotic_like { match attempt%4 { 0=>Rule::Adaptive, 1=>Rule::ShortestProc, 2=>Rule::MostWork, _=>Rule::Regret } }
                    else { match attempt { 0=>r0, 1=>Rule::Adaptive, 2=>Rule::BnHeavy, 3=>Rule::EndTight, 4=>Rule::Regret, 5=>Rule::CriticalPath, 6=>Rule::LeastFlex, 7=>Rule::MostWork, 8=>if allow_flex_balance{Rule::FlexBalance}else{r1}, _=>r1 } };
                let k = match attempt%4 { 0=>2, 1=>3, 2=>4, _=>2 }.min(k_hi);
                let (sol, mk) = construct_solution_conflict(challenge, pre, rule, k, target_ls, &mut rng, Some(&jb), Some(&mp), rp.as_ref(), if rp.is_some(){route_w_ls}else{0.0})?;
                if mk < best_makespan { best_makespan=mk; best_solution=Some(sol.clone()); save_solution(&sol)?; }
                refine_results.push((sol, mk));
            }
        }
        for (sol, mk) in refine_results { push_top_solutions(&mut top_solutions, &sol, mk, 15); }

        let ts_starts = top_solutions.len().min(10);
        let ts_iters = effort.job_shop_iters;    
        let ts_tenure = ((pre.total_ops as f64).sqrt() as usize * (100 + (pre.load_cv * 60.0) as usize) / 100).clamp(8, 24);
        {
            let ts_pool: Vec<Solution> = top_solutions.iter().take(ts_starts).map(|(s,_)| s.clone()).collect();
            for base_sol in &ts_pool {
                if let Some((sol2, mk2)) = tabu_search_phase(pre, challenge, base_sol, ts_iters, ts_tenure)? {
                    if mk2 < best_makespan { best_makespan=mk2; best_solution=Some(sol2.clone()); save_solution(&sol2)?; }
                    push_top_solutions(&mut top_solutions, &sol2, mk2, 20);
                }
            }
        }

        {
            let bn_pool: Vec<Solution> = top_solutions.iter().take(8).map(|(s,_)| s.clone()).collect();
            let mut shared_bn_buf: Option<(usize, EvalBuf)> = None;
            for base_sol in &bn_pool {
                let mut ds = match build_disj_from_solution(pre, challenge, base_sol) { Ok(d) => d, Err(_) => continue };
                if shared_bn_buf.as_ref().map_or(true, |(n, _)| *n != ds.n) {
                    shared_bn_buf = Some((ds.n, EvalBuf::new(ds.n)));
                }
                let (_, buf) = shared_bn_buf.as_mut().unwrap();
                let Some((mut cur_mk, _)) = eval_disj(&ds, buf) else { continue };
                let mut job_head_lb = vec![0u32; ds.n];
                let mut job_tail_lb = vec![0u32; ds.n];
                for j in 0..ds.num_jobs {
                    let base = ds.job_offsets[j];
                    let end = ds.job_offsets[j + 1];
                    let mut acc = 0u32;
                    for nd in base..end {
                        job_head_lb[nd] = acc;
                        acc = acc.saturating_add(ds.node_pt[nd]);
                    }
                    acc = 0;
                    for nd in (base..end).rev() {
                        job_tail_lb[nd] = acc;
                        acc = acc.saturating_add(ds.node_pt[nd]);
                    }
                }
                let mut machine_total_pt = vec![0u64; challenge.num_machines];
                for m in 0..challenge.num_machines {
                    if m < ds.machine_seq.len() {
                        for &nd in &ds.machine_seq[m] {
                            if nd < ds.node_pt.len() {
                                machine_total_pt[m] = machine_total_pt[m].saturating_add(ds.node_pt[nd] as u64);
                            }
                        }
                    }
                }
                let mut m_rank: Vec<usize> = (0..challenge.num_machines).filter(|&m| m < ds.machine_seq.len() && ds.machine_seq[m].len() > 1).collect();
                m_rank.sort_by(|&a, &b| machine_total_pt[b].cmp(&machine_total_pt[a]));
                let num_bn_ls = m_rank.len().min(3);
                let mut any_improved = false;
                for bi in 0..num_bn_ls {
                    let m = m_rank[bi];
                    let seq_cap = ds.machine_seq[m].len().min(18);
                    let mut prefix_pt = vec![0u32; seq_cap + 1];
                    let mut found_improvement = true;
                    while found_improvement {
                        found_improvement = false;
                        prefix_pt[0] = 0;
                        for idx in 0..seq_cap {
                            prefix_pt[idx + 1] = prefix_pt[idx].saturating_add(ds.node_pt[ds.machine_seq[m][idx]]);
                        }
                        'swap_loop: for i in 0..seq_cap.saturating_sub(1) {
                            for j in (i+1)..seq_cap {
                                if j >= ds.machine_seq[m].len() { break; }
                                let lb = lower_bound_machine_swap_jobpath(&ds.machine_seq[m], i, j, &prefix_pt, &ds.node_pt, &job_head_lb, &job_tail_lb);
                                if lb >= cur_mk { continue; }
                                ds.machine_seq[m].swap(i, j);
                                if let Some((new_mk, _)) = eval_disj(&ds, buf) {
                                    if new_mk < cur_mk {
                                        cur_mk = new_mk;
                                        found_improvement = true;
                                        any_improved = true;
                                        break 'swap_loop;
                                    }
                                }
                                ds.machine_seq[m].swap(i, j);
                            }
                        }
                    }
                }
                if any_improved {
                    if let Some((mk_bn, _)) = eval_disj(&ds, buf) {
                        if let Ok(sol_bn) = disj_to_solution(pre, &ds, &buf.start) {
                            if mk_bn < best_makespan { best_makespan=mk_bn; best_solution=Some(sol_bn.clone()); save_solution(&sol_bn)?; }
                            push_top_solutions(&mut top_solutions, &sol_bn, mk_bn, 20);
                        }
                    }
                }
            }
        }

        {
            let ils_pool: Vec<Solution> = top_solutions.iter().take(6).map(|(s,_)| s.clone()).collect();
            for base_sol in &ils_pool {
                if let Ok(Some((ls_sol, ls_mk))) = critical_block_move_local_search_ex(pre, challenge, base_sol, 5, 400, 120) {
                    if ls_mk < best_makespan { best_makespan=ls_mk; best_solution=Some(ls_sol.clone()); save_solution(&ls_sol)?; }
                    push_top_solutions(&mut top_solutions, &ls_sol, ls_mk, 20);
                    if let Some((sol3, mk3)) = tabu_search_phase(pre, challenge, &ls_sol, ts_iters/2, ts_tenure)? {
                        if mk3 < best_makespan { best_makespan=mk3; best_solution=Some(sol3.clone()); save_solution(&sol3)?; }
                        push_top_solutions(&mut top_solutions, &sol3, mk3, 20);
                    }
                }
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

            let num_generations = 22usize;
            let mut gen_no_improve = 0usize;
            let max_gen_no_improve = 12usize;
            let mut shared_mem_buf: Option<(usize, EvalBuf)> = None;

            for gen in 0..num_generations {
                if gen_no_improve >= max_gen_no_improve { break; }
                let cur_pop = mem_pop.len();
                if cur_pop < 2 { break; }

                let use_mutation = gen % 5 == 4;

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

                let (sol_a, mk_a) = (&mem_pop[ia].0, mem_pop[ia].1);
                let (sol_b, mk_b) = (&mem_pop[ib].0, mem_pop[ib].1);

                let ds_a = match build_disj_from_solution(pre, challenge, sol_a) { Ok(d) => d, Err(_) => { gen_no_improve += 1; continue; } };
                let ds_b = match build_disj_from_solution(pre, challenge, sol_b) { Ok(d) => d, Err(_) => { gen_no_improve += 1; continue; } };

                let (better_ds, worse_ds) = if mk_a <= mk_b { (&ds_a, &ds_b) } else { (&ds_b, &ds_a) };
                let mut child_ds = better_ds.clone();

                for m in 0..num_machines {
                    if is_bottleneck[m] { continue; }
                    if m >= worse_ds.machine_seq.len() || m >= child_ds.machine_seq.len() { continue; }
                    if worse_ds.machine_seq[m].len() == child_ds.machine_seq[m].len() {
                        if rng.gen::<f64>() < 0.65 {
                            child_ds.machine_seq[m] = worse_ds.machine_seq[m].clone();
                        }
                    }
                }

                if use_mutation {
                    let non_bn_machines: Vec<usize> = (0..num_machines).filter(|&m| !is_bottleneck[m] && child_ds.machine_seq[m].len() > 1).collect();
                    if !non_bn_machines.is_empty() {
                        for _ in 0..3 {
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
                let ls_sol = match disj_to_solution(pre, &child_ds, &child_buf.start) {
                    Ok(s) => s,
                    Err(_) => { gen_no_improve += 1; continue; }
                };

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
                    }
                } else {
                    mem_pop.push((ls_sol, ls_mk));
                }
            }

            let mem_best: Vec<Solution> = {
                let mut sorted = mem_pop.clone();
                sorted.sort_by_key(|(_, mk)| *mk);
                sorted.into_iter().take(3).map(|(s, _)| s).collect()
            };
            for base_sol in &mem_best {
                if let Some((ts_sol, ts_mk)) = tabu_search_phase(pre, challenge, base_sol, ts_iters / 3, ts_tenure)? {
                    if ts_mk < best_makespan {
                        best_makespan = ts_mk;
                        best_solution = Some(ts_sol.clone());
                        save_solution(&ts_sol)?;
                    }
                    push_top_solutions(&mut top_solutions, &ts_sol, ts_mk, 20);
                }
            }
        }

        if let Some(ref final_best) = best_solution.clone() {
            if let Some((sol4, mk4)) = tabu_search_phase(pre, challenge, final_best, ts_iters, ts_tenure)? {
                if mk4 < best_makespan { best_solution=Some(sol4.clone()); save_solution(&sol4)?; }
            }
        }

        if let Some(sol) = best_solution { save_solution(&sol)?; }
        Ok(())
    }
}

pub mod fjsp_medium {
    use anyhow::{anyhow, Result};
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    use std::collections::HashMap;
    use tig_challenges::job_scheduling::*;
    use super::types::*;
    use super::infra_shared::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Rule {
        Adaptive, BnHeavy, EndTight, CriticalPath, MostWork, LeastFlex, Regret, ShortestProc, FlexBalance,
    }

    #[inline]
    fn slack_urgency_fm(pre: &Pre, target_mk: Option<u32>, time: u32, product: usize, op_idx: usize) -> f64 {
        let Some(tgt) = target_mk else { return 0.0 };
        let lb = (time as u64).saturating_add(pre.product_suf_min[product][op_idx] as u64);
        let slack = (tgt as i64) - (lb as i64);
        let scale = (0.70 * pre.avg_op_min).max(1.0);
        let pos = (slack.max(0) as f64) / scale; let neg = ((-slack).max(0) as f64) / scale;
        (1.0 / (1.0 + pos)).clamp(0.0, 1.0) + (0.35 * neg).min(3.0)
    }

    #[inline]
    fn route_pref_bonus_fm(rp: Option<&RoutePrefLite>, product: usize, op_idx: usize, machine: usize) -> f64 {
        let Some(rp) = rp else { return 0.0 };
        if product >= rp.len() || op_idx >= rp[product].len() { return 0.0; }
        let r = rp[product][op_idx]; let mu = machine.min(255) as u8;
        if mu == r.best_m { (r.best_w as f64) / 255.0 } else if mu == r.second_m { (r.second_w as f64) / 255.0 } else { 0.0 }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn score_candidate(
        pre: &Pre, rule: Rule, job: usize, product: usize, op_idx: usize,
        ops_rem: usize, op: &OpInfo, machine: usize, pt: u32, time: u32,
        target_mk: Option<u32>, best_end: u32, second_end: u32, best_cnt_total: usize,
        progress: f64, job_bias: f64, machine_penalty: f64, dynamic_load: f64,
        route_pref: Option<&RoutePrefLite>, route_w: f64, jitter: f64,
    ) -> f64 {
        let rem_min = pre.product_suf_min[product][op_idx] as f64;
        let rem_avg = pre.product_suf_avg[product][op_idx]; let rem_bn = pre.product_suf_bn[product][op_idx];
        let flex_f = (op.flex as f64).max(1.0); let flex_inv = 1.0/flex_f;
        let rem_min_n = rem_min/pre.horizon.max(1.0); let rem_avg_n = rem_avg/pre.max_job_avg_work.max(1e-9);
        let bn_n = rem_bn/pre.max_job_bn.max(1e-9); let ops_n = (ops_rem as f64)/(pre.max_ops as f64).max(1.0);
        let load_n = dynamic_load/pre.avg_machine_load.max(1e-9); let scar_n = pre.machine_scarcity[machine]/pre.avg_machine_scarcity.max(1e-9);
        let end_n = (best_end as f64)/pre.time_scale.max(1.0); let proc_n = (pt as f64)/pre.avg_op_min.max(1.0);
        let regret = if second_end >= INF { pre.avg_op_min*2.6 } else { (second_end-best_end) as f64 };
        let reg_n = (regret/pre.avg_op_min.max(1.0)).clamp(0.0,6.0);
        let scarcity_urg = 1.0/(best_cnt_total as f64).max(1.0);
        let density_n = ((rem_min/(ops_rem as f64).max(1.0))/pre.avg_op_min.max(1.0)).clamp(0.0,4.0);
        let next_min = pre.product_next_min[product][op_idx] as f64; let next_min_n = next_min/pre.horizon.max(1.0);
        let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
        let p2 = progress*progress; let next_w_base = 0.12+p2*0.28;
        let next_term_raw = (0.55*next_min_n+0.45*next_flex_inv)*(1.0+0.30*density_n*pre.high_flex);
        let js = pre.jobshopness; let fl = 1.0-js;
        let avg_flex_inv = 1.0/pre.flex_avg.max(1.0); let scarce_match = scar_n*(flex_inv-avg_flex_inv);
        let mpen = machine_penalty.clamp(0.0,1.0); let mpen_gain = 1.0+0.85*pre.high_flex;
        let flow_term = pre.flow_w*pre.job_flow_pref[job]*(0.65+0.70*(1.0-progress));
        let slack_u = slack_urgency_fm(pre, target_mk, time, product, op_idx);
        let slack_w = pre.slack_base*(0.25+0.75*progress); let slack_reg_boost = 1.0+0.40*reg_n*progress;
        let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop=pre.machine_best_pop[machine]; (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
        let route_gain = (0.70+0.80*(1.0-progress)).clamp(0.70,1.40);
        let route_term = if route_w>0.0 && op.flex>=2 { route_w*route_gain*route_pref_bonus_fm(route_pref,product,op_idx,machine) } else { 0.0 };
        match rule {
            Rule::CriticalPath => {
                let next_term = next_w_base * 0.30 * next_term_raw;
                let slack_term = slack_w * slack_u * slack_reg_boost;
                let base_score = (1.03 * rem_min_n) + (0.10 * ops_n) + (0.24 * scarcity_urg) + (0.20 * pre.flex_factor) * flex_inv + next_term + 0.10 * slack_term - (0.70 * end_n) - pop_pen + flow_term + route_term + jitter;
                let bias_factor = 0.45 * job_bias;
                base_score + bias_factor * base_score.abs()
            }
            Rule::MostWork => {
                let next_term = next_w_base * 0.25 * next_term_raw;
                let base_score = (1.00 * rem_avg_n) + (0.12 * ops_n) + (0.18 * scarcity_urg) + (0.15 * pre.flex_factor) * flex_inv + next_term - (0.62 * end_n) - pop_pen + flow_term + route_term + jitter;
                let bias_factor = 0.45 * job_bias;
                base_score + bias_factor * base_score.abs()
            }
            Rule::LeastFlex => {
                let next_term = next_w_base * 0.20 * next_term_raw;
                let base_score = (1.00 * flex_inv) + (0.28 * rem_min_n) + (0.22 * scarcity_urg) + next_term - (0.55 * end_n) - pop_pen + flow_term + route_term + jitter;
                let bias_factor = 0.35 * job_bias;
                base_score + bias_factor * base_score.abs()
            }
            Rule::ShortestProc => {
                let next_term = next_w_base * 0.20 * next_term_raw;
                let base_score = (-1.00 * proc_n) + (0.25 * rem_min_n) + (0.12 * scarcity_urg) + next_term - (0.20 * end_n) - pop_pen + flow_term + route_term + jitter;
                let bias_factor = 0.25 * job_bias;
                base_score + bias_factor * base_score.abs()
            }
            Rule::Regret => {
                let reg_scale = (1.0 + 0.35 * pre.bn_focus) * (1.0 + 0.25 * pre.load_cv);
                let next_term = next_w_base * 0.25 * next_term_raw;
                let scarce_w = 0.18 + 0.15 * pre.load_cv;
                let base_score = (reg_scale * 1.10 * reg_n) + (0.60 * rem_min_n) + (0.25 * scarcity_urg) + (scarce_w * pre.flex_factor) * flex_inv + next_term - (0.65 * end_n) - pop_pen + flow_term + route_term + jitter;
                let bias_factor = 0.38 * job_bias;
                base_score + bias_factor * base_score.abs()
            }
            Rule::EndTight => {
                let end_w = 1.10 + 1.00 * progress + 0.35 * pre.high_flex;
                let cp_w = 1.15 + 0.30 * js;
                let reg_w = (0.55 + 0.20 * (1.0 - progress)) * (0.85 + 0.60 * js);
                let mpen_w = (0.10 + 0.45 * pre.high_flex) * pre.flex_factor;
                let next_term = next_w_base * (0.45 + 0.55 * js) * next_term_raw;
                let slack_term = slack_w * (0.70 + 0.40 * js) * slack_u * slack_reg_boost;
                let base_score = (cp_w * rem_min_n) + 0.12 * rem_avg_n + 0.08 * ops_n + 0.18 * scarcity_urg + (0.30 * pre.flex_factor) * flex_inv + (0.20 * pre.flex_factor) * scarce_match + (reg_w * pre.flex_factor) * reg_n + next_term + slack_term - end_w * end_n - 0.22 * proc_n - pop_pen - (mpen_gain * mpen_w) * mpen + flow_term + route_term + jitter;
                let bias_factor = 0.55 * job_bias;
                base_score + bias_factor * base_score.abs()
            }
            Rule::BnHeavy => {
                let bn_w = (0.90 + 0.55 * js) * pre.bn_focus;
                let end_w = 0.65 + 0.70 * progress;
                let reg_w = (0.60 + 0.25 * (1.0 - progress)) * (0.85 + 0.35 * js);
                let load_w = if pre.hi_flex { -0.35 } else { 0.55 + 0.25 * js };
                let mpen_w = (0.12 + 0.30 * js) * pre.flex_factor * (0.95 + 0.65 * pre.high_flex);
                let next_term = next_w_base * (0.55 + 0.75 * js) * next_term_raw;
                let slack_term = slack_w * (0.45 + 0.55 * js) * slack_u * slack_reg_boost;
                let base_score = (0.95 * rem_min_n) + (0.30 * rem_avg_n) + (bn_w * bn_n) + (0.22 * density_n) + (0.10 * ops_n) + (0.65 * pre.flex_factor) * flex_inv + (0.35 * pre.flex_factor) * scarce_match + load_w * pre.flex_factor * load_n + (reg_w * pre.flex_factor) * reg_n + 0.18 * scarcity_urg + next_term + slack_term - end_w * end_n - 0.18 * proc_n - pop_pen - (mpen_gain * mpen_w) * mpen + flow_term + route_term + jitter;
                let bias_factor = 0.60 * job_bias;
                base_score + bias_factor * base_score.abs()
            }
            Rule::Adaptive => {
                let end_w = (0.90 * fl + 0.72 * js) + (0.62 + 0.12 * fl) * progress + 0.18 * pre.high_flex;
                let reg_scale = (1.0 + 0.40 * pre.bn_focus * (1.0 / pre.flex_avg.max(1.0)) * 2.5) * (1.0 + 0.30 * pre.load_cv);
                let reg_w = ((0.50 * fl + 0.78 * js) + 0.18 * (1.0 - progress)) * reg_scale;
                let bn_w = ((0.45 + 0.40 * js) + 0.25 * (1.0 - progress)) * pre.bn_focus;
                let load_sign = if pre.hi_flex { -1.0 } else { 1.0 };
                let load_w = load_sign * (0.45 * fl + 0.75 * js) * pre.flex_factor;
                let density_w = 0.08 * fl + 0.20 * js;
                let next_term = next_w_base * (0.50 * fl + 1.50 * js) * next_term_raw;
                let mpen_w = (0.08 * fl + 0.28 * js) * pre.flex_factor * (1.0 + 0.85 * pre.high_flex);
                let slack_term = slack_w * (0.55 * fl + 0.85 * js) * slack_u * slack_reg_boost;
                let route_scale = 1.0 + 0.45 * (1.0 / pre.flex_avg.max(1.0)) * 3.0 * (1.0 - 0.5 * pre.high_flex);
                let route_term_a = route_term * route_scale;
                let scarce_w = (0.55 + 0.25 * pre.load_cv) * pre.flex_factor;
                let base_score = (1.05 * rem_min_n) + (0.48 * rem_avg_n) + (bn_w * bn_n) + density_w * density_n + (0.08 * ops_n) + (0.62 * pre.flex_factor) * flex_inv + scarce_w * scarce_match + load_w * load_n + (reg_w * pre.flex_factor) * reg_n + 0.20 * pre.flex_factor * scarcity_urg + next_term + slack_term - end_w * end_n - (0.18 * fl + 0.12 * js) * proc_n - pop_pen - (mpen_gain * mpen_w) * mpen + flow_term + route_term_a + jitter;
                let bias_factor = (0.62 + 0.06 * js) * job_bias;
                base_score + bias_factor * base_score.abs()
            }
            Rule::FlexBalance => {
                let end_w = (0.85 + 0.70 * progress + 0.15 * js).clamp(0.85, 1.75);
                let cp_w = (1.00 + 0.30 * js + 0.15 * (1.0 - progress)).clamp(0.95, 1.45);
                let load_w = (0.55 + 0.35 * pre.high_flex).clamp(0.55, 0.95) * pre.flex_factor;
                let mpen_w = (0.55 + 0.65 * pre.high_flex).clamp(0.55, 1.15);
                let reg_w = (0.35 + 0.25 * (1.0 - progress)).clamp(0.35, 0.70);
                let next_term = next_w_base * 0.40 * next_term_raw;
                let base_score = (cp_w * rem_min_n) + 0.55 * rem_avg_n + 0.08 * ops_n + 0.06 * density_n + 0.08 * scarcity_urg + next_term + (0.70 * slack_w) * slack_u - end_w * end_n - 0.16 * proc_n - pop_pen - load_w * load_n - (mpen_w * (1.0 + 0.85 * pre.high_flex)) * mpen + (reg_w * pre.flex_factor) * reg_n + flow_term + route_term + jitter;
                let bias_factor = (0.58 + 0.10 * pre.high_flex) * job_bias;
                base_score + bias_factor * base_score.abs()
            }
        }
    }

    #[inline]
    fn rule_idx(r: Rule) -> usize {
        match r { Rule::Adaptive=>0, Rule::BnHeavy=>1, Rule::EndTight=>2, Rule::CriticalPath=>3, Rule::MostWork=>4, Rule::LeastFlex=>5, Rule::Regret=>6, Rule::ShortestProc=>7, Rule::FlexBalance=>8 }
    }

    fn choose_rule_bandit(rng: &mut SmallRng, rules: &[Rule], rule_best: &[u32], rule_tries: &[u32], global_best: u32, margin: u32, stuck: usize, chaos_like: bool, late_phase: bool) -> Rule {
        if rules.is_empty() { return Rule::Adaptive; }
        let mut best_seen = global_best; for &mk in rule_best { if mk < best_seen { best_seen = mk; } }
        let scale = (margin as f64).max(1.0); let s = ((stuck as f64)/140.0).clamp(0.0,1.0); let explore_mix = (0.10+0.55*s).clamp(0.10,0.65);
        let mut sum=0.0;
        for &r in rules.iter() {
            let mk=rule_best[rule_idx(r)]; let t=rule_tries[rule_idx(r)].max(1) as f64;
            let delta=mk.saturating_sub(best_seen) as f64; let exploit=(-delta/scale).exp(); let explore=(1.0/t).sqrt();
            let mut ww=(1.0-explore_mix)*exploit+explore_mix*explore; ww=ww.max(1e-6);
            if chaos_like{ww=ww.powf(0.70);}else if late_phase{ww=ww.powf(1.18);}
            sum+=ww.max(0.0);
        }
        if !(sum>0.0) { return rules[rng.gen_range(0..rules.len())]; }
        let mut r=rng.gen::<f64>()*sum;
        for &rule in rules.iter() {
            let mk=rule_best[rule_idx(rule)]; let t=rule_tries[rule_idx(rule)].max(1) as f64;
            let delta=mk.saturating_sub(best_seen) as f64; let exploit=(-delta/scale).exp(); let explore=(1.0/t).sqrt();
            let mut ww=(1.0-explore_mix)*exploit+explore_mix*explore; ww=ww.max(1e-6);
            if chaos_like{ww=ww.powf(0.70);}else if late_phase{ww=ww.powf(1.18);}
            r-=ww.max(0.0);
            if r<=0.0 { return rule; }
        }
        rules[rules.len()-1]
    }

    fn construct_solution_conflict(
        challenge: &Challenge, pre: &Pre, rule: Rule, k: usize, target_mk: Option<u32>,
        rng: &mut SmallRng, job_bias: Option<&[f64]>, machine_penalty: Option<&[f64]>,
        route_pref: Option<&RoutePrefLite>, route_w: f64,
    ) -> Result<(Solution, u32)> {
        if k == 0 {
            construct_solution_conflict_det(
                challenge,
                pre,
                rule,
                target_mk,
                job_bias,
                machine_penalty,
                route_pref,
                route_w,
            )
        } else {
            construct_solution_conflict_topk(
                challenge,
                pre,
                rule,
                k,
                target_mk,
                rng,
                job_bias,
                machine_penalty,
                route_pref,
                route_w,
            )
        }
    }

    fn construct_solution_conflict_det(
        challenge: &Challenge, pre: &Pre, rule: Rule, target_mk: Option<u32>,
        job_bias: Option<&[f64]>, machine_penalty: Option<&[f64]>,
        route_pref: Option<&RoutePrefLite>, route_w: f64,
    ) -> Result<(Solution, u32)> {
        let num_jobs = challenge.num_jobs;
        let num_machines = challenge.num_machines;

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

        let chaotic_like = pre.chaotic_like;
        let mut machine_work: Vec<u64> = if chaotic_like { vec![0u64; num_machines] } else { vec![] };
        let mut sum_work: u64 = 0;

        let mut ready_jobs: Vec<usize> = Vec::with_capacity(num_jobs);
        for job in 0..num_jobs {
            if pre.job_ops_len[job] > 0 {
                ready_jobs.push(job);
            }
        }
        let mut delayed_jobs: Vec<(u32, usize)> = Vec::with_capacity(num_jobs);
        let mut delayed_head = 0usize;

        while remaining_ops > 0 {
            loop {
                let mut any_idle = false;
                for m in 0..num_machines {
                    if machine_avail[m] <= time {
                        any_idle = true;
                        break;
                    }
                }
                if !any_idle || ready_jobs.is_empty() {
                    break;
                }

                let progress = 1.0 - (remaining_ops as f64) / (pre.total_ops as f64).max(1.0);
                let (bal_w, avg_work) = if chaotic_like {
                    (
                        (0.030 + 0.070 * (1.0 - progress)).clamp(0.025, 0.11),
                        (sum_work as f64) / (num_machines as f64).max(1.0),
                    )
                } else {
                    (0.0, 0.0)
                };

                let mut best: Option<Cand> = None;

                for &job in &ready_jobs {
                    let op_idx = job_next_op[job];
                    if op_idx >= pre.job_ops_len[job] {
                        continue;
                    }
                    let product = pre.job_products[job];
                    let op = &pre.product_ops[product][op_idx];
                    if op.flex == 0 || op.machines.is_empty() || op.min_pt >= INF {
                        continue;
                    }

                    let (best_end, second_end, best_cnt_total, best_cnt_idle) =
                        best_second_and_counts(time, &machine_avail, op);
                    if best_end >= INF || best_cnt_idle == 0 {
                        continue;
                    }

                    let ops_rem = pre.job_ops_len[job] - op_idx;
                    let jb = job_bias.map(|v| v[job]).unwrap_or(0.0);

                    for &(m, pt) in &op.machines {
                        if machine_avail[m] > time {
                            continue;
                        }
                        let end = time.saturating_add(pt);
                        if end != best_end {
                            continue;
                        }

                        let mp = machine_penalty.map(|v| v[m]).unwrap_or(0.0);

                        let base = score_candidate(
                            pre,
                            rule,
                            job,
                            product,
                            op_idx,
                            ops_rem,
                            op,
                            m,
                            pt,
                            time,
                            target_mk,
                            best_end,
                            second_end,
                            best_cnt_total,
                            progress,
                            jb,
                            mp,
                            machine_load[m],
                            route_pref,
                            route_w,
                            0.0,
                        );

                        let bal_pen = if chaotic_like && bal_w > 0.0 {
                            let denomw = (avg_work + (pre.avg_op_min * 3.0).max(1.0)).max(1.0);
                            let r = (machine_work[m] as f64) / denomw;
                            let done_n = (r / (r + 1.0)).clamp(0.0, 1.0);
                            -bal_w * done_n
                        } else {
                            0.0
                        };

                        let c = Cand { job, machine: m, pt, score: base + bal_pen };
                        if best.map_or(true, |bb| c.score > bb.score) {
                            best = Some(c);
                        }
                    }
                }

                let chosen = match best {
                    Some(c) => c,
                    None => break,
                };

                let job = chosen.job;
                let machine = chosen.machine;
                let pt = chosen.pt;

                let product = pre.job_products[job];
                let op_idx = job_next_op[job];
                let op = &pre.product_ops[product][op_idx];

                let (best_end_now, _, _, _) = best_second_and_counts(time, &machine_avail, op);
                let end_check = time.max(machine_avail[machine]).saturating_add(pt);
                if machine_avail[machine] > time || end_check != best_end_now {
                    break;
                }

                if let Ok(pos) = ready_jobs.binary_search(&job) {
                    ready_jobs.remove(pos);
                }

                let end_time = time.saturating_add(pt);
                job_schedule[job].push((machine, time));
                job_next_op[job] += 1;
                job_ready_time[job] = end_time;
                machine_avail[machine] = end_time;
                remaining_ops -= 1;

                if chaotic_like {
                    machine_work[machine] = machine_work[machine].saturating_add(pt as u64);
                    sum_work = sum_work.saturating_add(pt as u64);
                }

                if op.min_pt < INF && op.flex > 0 && !op.machines.is_empty() {
                    let delta = (op.min_pt as f64) / (op.flex as f64).max(1.0);
                    if delta > 0.0 {
                        for &(mm, _) in &op.machines {
                            let v = machine_load[mm] - delta;
                            machine_load[mm] = if v > 0.0 { v } else { 0.0 };
                        }
                    }
                }

                if job_next_op[job] < pre.job_ops_len[job] {
                    if job_ready_time[job] <= time {
                        let pos = ready_jobs.binary_search(&job).unwrap_or_else(|p| p);
                        ready_jobs.insert(pos, job);
                    } else {
                        let item = (job_ready_time[job], job);
                        let rel = delayed_jobs[delayed_head..]
                            .binary_search_by(|&(t, j)| {
                                if t < item.0 {
                                    std::cmp::Ordering::Less
                                } else if t > item.0 {
                                    std::cmp::Ordering::Greater
                                } else {
                                    j.cmp(&item.1)
                                }
                            })
                            .unwrap_or_else(|p| p);
                        delayed_jobs.insert(delayed_head + rel, item);
                    }
                }

                if remaining_ops == 0 {
                    break;
                }
            }

            if remaining_ops == 0 {
                break;
            }

            let mut next_time: Option<u32> = None;
            for &t in &machine_avail {
                if t > time {
                    next_time = Some(next_time.map_or(t, |b| b.min(t)));
                }
            }
            if delayed_head < delayed_jobs.len() {
                let t = delayed_jobs[delayed_head].0;
                if t > time {
                    next_time = Some(next_time.map_or(t, |b| b.min(t)));
                }
            }
            time = next_time.ok_or_else(|| anyhow!("Stalled"))?;

            while delayed_head < delayed_jobs.len() && delayed_jobs[delayed_head].0 <= time {
                let job = delayed_jobs[delayed_head].1;
                if job_next_op[job] < pre.job_ops_len[job] && job_ready_time[job] <= time {
                    let pos = ready_jobs.binary_search(&job).unwrap_or_else(|p| p);
                    ready_jobs.insert(pos, job);
                }
                delayed_head += 1;
            }
            if delayed_head > 64 && delayed_head * 2 >= delayed_jobs.len() {
                delayed_jobs.drain(0..delayed_head);
                delayed_head = 0;
            }
        }

        let mk = machine_avail.into_iter().max().unwrap_or(0);
        Ok((Solution { job_schedule }, mk))
    }

    fn construct_solution_conflict_topk(
        challenge: &Challenge, pre: &Pre, rule: Rule, k: usize, target_mk: Option<u32>,
        rng: &mut SmallRng, job_bias: Option<&[f64]>, machine_penalty: Option<&[f64]>,
        route_pref: Option<&RoutePrefLite>, route_w: f64,
    ) -> Result<(Solution, u32)> {
        let num_jobs = challenge.num_jobs;
        let num_machines = challenge.num_machines;

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

        let chaotic_like = pre.chaotic_like;
        let mut machine_work: Vec<u64> = if chaotic_like { vec![0u64; num_machines] } else { vec![] };
        let mut sum_work: u64 = 0;

        let mut ready_jobs: Vec<usize> = Vec::with_capacity(num_jobs);
        for job in 0..num_jobs {
            if pre.job_ops_len[job] > 0 {
                ready_jobs.push(job);
            }
        }
        let mut delayed_jobs: Vec<(u32, usize)> = Vec::with_capacity(num_jobs);
        let mut delayed_head = 0usize;

        while remaining_ops > 0 {
            loop {
                let mut any_idle = false;
                for m in 0..num_machines {
                    if machine_avail[m] <= time {
                        any_idle = true;
                        break;
                    }
                }
                if !any_idle || ready_jobs.is_empty() {
                    break;
                }

                let progress = 1.0 - (remaining_ops as f64) / (pre.total_ops as f64).max(1.0);
                let (bal_w, avg_work) = if chaotic_like {
                    (
                        (0.030 + 0.070 * (1.0 - progress)).clamp(0.025, 0.11),
                        (sum_work as f64) / (num_machines as f64).max(1.0),
                    )
                } else {
                    (0.0, 0.0)
                };

                let mut top: Vec<Cand> = Vec::with_capacity(k);

                for &job in &ready_jobs {
                    let op_idx = job_next_op[job];
                    if op_idx >= pre.job_ops_len[job] {
                        continue;
                    }
                    let product = pre.job_products[job];
                    let op = &pre.product_ops[product][op_idx];
                    if op.flex == 0 || op.machines.is_empty() || op.min_pt >= INF {
                        continue;
                    }

                    let (best_end, second_end, best_cnt_total, best_cnt_idle) =
                        best_second_and_counts(time, &machine_avail, op);
                    if best_end >= INF || best_cnt_idle == 0 {
                        continue;
                    }

                    let ops_rem = pre.job_ops_len[job] - op_idx;
                    let jb = job_bias.map(|v| v[job]).unwrap_or(0.0);

                    for &(m, pt) in &op.machines {
                        if machine_avail[m] > time {
                            continue;
                        }
                        let end = time.saturating_add(pt);
                        if end != best_end {
                            continue;
                        }

                        let mp = machine_penalty.map(|v| v[m]).unwrap_or(0.0);
                        let jitter = rng.gen::<f64>() * 1e-9;

                        let base = score_candidate(
                            pre,
                            rule,
                            job,
                            product,
                            op_idx,
                            ops_rem,
                            op,
                            m,
                            pt,
                            time,
                            target_mk,
                            best_end,
                            second_end,
                            best_cnt_total,
                            progress,
                            jb,
                            mp,
                            machine_load[m],
                            route_pref,
                            route_w,
                            jitter,
                        );

                        let bal_pen = if chaotic_like && bal_w > 0.0 {
                            let denomw = (avg_work + (pre.avg_op_min * 3.0).max(1.0)).max(1.0);
                            let r = (machine_work[m] as f64) / denomw;
                            let done_n = (r / (r + 1.0)).clamp(0.0, 1.0);
                            -bal_w * done_n
                        } else {
                            0.0
                        };

                        let c = Cand { job, machine: m, pt, score: base + bal_pen };
                        push_top_k(&mut top, c, k);
                    }
                }

                if top.is_empty() {
                    break;
                }
                let chosen = choose_from_top_weighted(rng, &top);

                let job = chosen.job;
                let machine = chosen.machine;
                let pt = chosen.pt;

                let product = pre.job_products[job];
                let op_idx = job_next_op[job];
                let op = &pre.product_ops[product][op_idx];

                let (best_end_now, _, _, _) = best_second_and_counts(time, &machine_avail, op);
                let end_check = time.max(machine_avail[machine]).saturating_add(pt);
                if machine_avail[machine] > time || end_check != best_end_now {
                    break;
                }

                if let Ok(pos) = ready_jobs.binary_search(&job) {
                    ready_jobs.remove(pos);
                }

                let end_time = time.saturating_add(pt);
                job_schedule[job].push((machine, time));
                job_next_op[job] += 1;
                job_ready_time[job] = end_time;
                machine_avail[machine] = end_time;
                remaining_ops -= 1;

                if chaotic_like {
                    machine_work[machine] = machine_work[machine].saturating_add(pt as u64);
                    sum_work = sum_work.saturating_add(pt as u64);
                }

                if op.min_pt < INF && op.flex > 0 && !op.machines.is_empty() {
                    let delta = (op.min_pt as f64) / (op.flex as f64).max(1.0);
                    if delta > 0.0 {
                        for &(mm, _) in &op.machines {
                            let v = machine_load[mm] - delta;
                            machine_load[mm] = if v > 0.0 { v } else { 0.0 };
                        }
                    }
                }

                if job_next_op[job] < pre.job_ops_len[job] {
                    if job_ready_time[job] <= time {
                        let pos = ready_jobs.binary_search(&job).unwrap_or_else(|p| p);
                        ready_jobs.insert(pos, job);
                    } else {
                        let item = (job_ready_time[job], job);
                        let rel = delayed_jobs[delayed_head..]
                            .binary_search_by(|&(t, j)| {
                                if t < item.0 {
                                    std::cmp::Ordering::Less
                                } else if t > item.0 {
                                    std::cmp::Ordering::Greater
                                } else {
                                    j.cmp(&item.1)
                                }
                            })
                            .unwrap_or_else(|p| p);
                        delayed_jobs.insert(delayed_head + rel, item);
                    }
                }

                if remaining_ops == 0 {
                    break;
                }
            }

            if remaining_ops == 0 {
                break;
            }

            let mut next_time: Option<u32> = None;
            for &t in &machine_avail {
                if t > time {
                    next_time = Some(next_time.map_or(t, |b| b.min(t)));
                }
            }
            if delayed_head < delayed_jobs.len() {
                let t = delayed_jobs[delayed_head].0;
                if t > time {
                    next_time = Some(next_time.map_or(t, |b| b.min(t)));
                }
            }
            time = next_time.ok_or_else(|| anyhow!("Stalled"))?;

            while delayed_head < delayed_jobs.len() && delayed_jobs[delayed_head].0 <= time {
                let job = delayed_jobs[delayed_head].1;
                if job_next_op[job] < pre.job_ops_len[job] && job_ready_time[job] <= time {
                    let pos = ready_jobs.binary_search(&job).unwrap_or_else(|p| p);
                    ready_jobs.insert(pos, job);
                }
                delayed_head += 1;
            }
            if delayed_head > 64 && delayed_head * 2 >= delayed_jobs.len() {
                delayed_jobs.drain(0..delayed_head);
                delayed_head = 0;
            }
        }

        let mk = machine_avail.into_iter().max().unwrap_or(0);
        Ok((Solution { job_schedule }, mk))
    }

    fn construct_solution_job_centric(
        challenge: &Challenge,
        pre: &Pre,
    ) -> Result<(Solution, u32)> {
        let num_jobs = challenge.num_jobs;
        let num_machines = challenge.num_machines;

        let mut job_priorities: Vec<(usize, u32)> = (0..num_jobs)
            .map(|j| {
                let product = pre.job_products[j];
                let total_min_pt: u32 = (0..pre.job_ops_len[j])
                    .map(|op_idx| pre.product_ops[product][op_idx].min_pt)
                    .sum();
                (j, total_min_pt)
            })
            .collect();
        
        job_priorities.sort_by_key(|&(_, work)| std::cmp::Reverse(work));
        let sorted_jobs: Vec<usize> = job_priorities.into_iter().map(|(j, _)| j).collect();

        let mut machine_avail = vec![0u32; num_machines];
        let mut job_schedule: Vec<Vec<(usize, u32)>> = (0..num_jobs)
            .map(|j| Vec::with_capacity(pre.job_ops_len[j]))
            .collect();

        for &job in &sorted_jobs {
            let product = pre.job_products[job];
            let num_ops = pre.job_ops_len[job];
            let mut last_op_completion_time = 0u32;

            for op_idx in 0..num_ops {
                let op_info = &pre.product_ops[product][op_idx];
                if op_info.machines.is_empty() {
                    continue;
                }

                let mut best_finish_time = u32::MAX;
                let mut best_machine = op_info.machines[0].0;
                let mut best_start_time = 0u32;

                for &(machine, pt) in &op_info.machines {
                    let start_time = last_op_completion_time.max(machine_avail[machine]);
                    let finish_time = start_time.saturating_add(pt);

                    if finish_time < best_finish_time {
                        best_finish_time = finish_time;
                        best_machine = machine;
                        best_start_time = start_time;
                    } else if finish_time == best_finish_time {
                        if machine_avail[machine] < machine_avail[best_machine] {
                             best_machine = machine;
                             best_start_time = start_time;
                        }
                    }
                }
                
                job_schedule[job].push((best_machine, best_start_time));
                machine_avail[best_machine] = best_finish_time;
                last_op_completion_time = best_finish_time;
            }
        }

        let mk = machine_avail.into_iter().max().unwrap_or(0);
        Ok((Solution { job_schedule }, mk))
    }


    fn exhaustive_critical_reroute_pass(pre: &Pre, challenge: &Challenge, base_sol: &Solution) -> Result<Option<(Solution, u32)>> {
        let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
        let mut buf = EvalBuf::new(ds.n);
        let Some((mut current_mk, mk_node0)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
        let initial_mk = current_mk;
        let mut improved = true;
        let mut passes = 0;
        let max_passes = 5;
        let mut buf_matches_current = true;
        let mut current_mk_node = mk_node0;

        while improved && passes < max_passes {
            improved = false;
            passes += 1;
            if !buf_matches_current {
                let Some((mk, mk_node)) = eval_disj(&ds, &mut buf) else { break };
                current_mk = mk;
                current_mk_node = mk_node;
                buf_matches_current = true;
            }
            let mut crit_nodes: Vec<usize> = Vec::with_capacity(128);
            let mut u = current_mk_node;
            while u != NONE_USIZE { crit_nodes.push(u); u = buf.best_pred[u]; }
            'node_loop: for &node in &crit_nodes {
                let job = ds.node_job[node]; let op_idx = ds.node_op[node]; let product = pre.job_products[job];
                let op_info = &pre.product_ops[product][op_idx];
                if op_info.machines.len() <= 1 { continue; }
                let cur_machine = ds.node_machine[node]; let cur_pt = ds.node_pt[node];
                let mut best_mk = current_mk; let mut best_m = cur_machine; let mut best_pt = cur_pt; let mut best_pos = 0usize;
                for &(new_m, new_pt) in &op_info.machines {
                    if new_m == cur_machine { continue; }
                    let old_pos = match ds.machine_seq[cur_machine].iter().position(|&x| x == node) { Some(p) => p, None => continue };
                    ds.machine_seq[cur_machine].remove(old_pos);
                    ds.node_machine[node] = new_m; ds.node_pt[node] = new_pt;
                    let target_len = ds.machine_seq[new_m].len();
                    for pos in 0..=target_len {
                        ds.machine_seq[new_m].insert(pos, node);
                        if let Some((test_mk, _)) = eval_disj(&ds, &mut buf) {
                            if test_mk < best_mk { best_mk = test_mk; best_m = new_m; best_pt = new_pt; best_pos = pos; }
                        }
                        ds.machine_seq[new_m].remove(pos);
                        buf_matches_current = false;
                    }
                    ds.machine_seq[cur_machine].insert(old_pos, node);
                    ds.node_machine[node] = cur_machine; ds.node_pt[node] = cur_pt;
                }
                if best_m != cur_machine {
                    let old_pos = ds.machine_seq[cur_machine].iter().position(|&x| x == node).unwrap();
                    ds.machine_seq[cur_machine].remove(old_pos);
                    let ins = best_pos.min(ds.machine_seq[best_m].len());
                    ds.machine_seq[best_m].insert(ins, node);
                    ds.node_machine[node] = best_m; ds.node_pt[node] = best_pt;
                    current_mk = best_mk;
                    improved = true;
                    buf_matches_current = false;
                    continue 'node_loop;
                }
            }
        }
        if current_mk >= initial_mk { return Ok(None); }
        if !buf_matches_current {
            let Some((mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
            current_mk = mk;
        }
        let sol = disj_to_solution(pre, &ds, &buf.start)?;
        Ok(Some((sol, current_mk)))
    }

    fn greedy_reassign_pass(pre: &Pre, challenge: &Challenge, base_sol: &Solution) -> Result<Option<(Solution, u32)>> {
        let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
        let mut buf = EvalBuf::new(ds.n);
        let n = ds.n;

        let Some((mut current_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
        let initial_mk = current_mk;

        let mut pos_of_node = vec![0usize; n];
        for seq in &ds.machine_seq {
            for (pos, &nd) in seq.iter().enumerate() {
                pos_of_node[nd] = pos;
            }
        }

        let mut passes = 0usize;
        let max_passes = 3;

        while passes < max_passes {
            passes += 1;

            let Some((cur_mk, _)) = eval_disj(&ds, &mut buf) else { break };
            current_mk = cur_mk;

            for seq in &ds.machine_seq {
                for (pos, &nd) in seq.iter().enumerate() {
                    pos_of_node[nd] = pos;
                }
            }

            let mut best_pass_mk = current_mk;
            let mut best_pass_move: Option<(usize, usize, u32, usize)> = None;

            for node in 0..n {
                let job = ds.node_job[node];
                let op_idx = ds.node_op[node];
                let product = pre.job_products[job];
                let op_info = &pre.product_ops[product][op_idx];
                if op_info.machines.len() <= 1 {
                    continue;
                }

                let cur_machine = ds.node_machine[node];
                let cur_pt = ds.node_pt[node];

                for &(new_m, new_pt) in &op_info.machines {
                    if new_m == cur_machine {
                        continue;
                    }

                    let old_pos = pos_of_node[node];
                    if old_pos >= ds.machine_seq[cur_machine].len() || ds.machine_seq[cur_machine][old_pos] != node {
                        continue;
                    }

                    ds.machine_seq[cur_machine].remove(old_pos);
                    for idx in old_pos..ds.machine_seq[cur_machine].len() {
                        let nd = ds.machine_seq[cur_machine][idx];
                        pos_of_node[nd] = idx;
                    }
                    ds.node_machine[node] = new_m;
                    ds.node_pt[node] = new_pt;

                    let target_len = ds.machine_seq[new_m].len();
                    let cur_start = buf.start[node];
                    let mut sorted_pos = target_len;
                    for (k, &nd) in ds.machine_seq[new_m].iter().enumerate() {
                        if buf.start[nd] >= cur_start {
                            sorted_pos = k;
                            break;
                        }
                    }

                    let mut positions: Vec<usize> = Vec::with_capacity(3);
                    for &p in &[sorted_pos, sorted_pos.saturating_sub(1), target_len] {
                        if p <= target_len && !positions.contains(&p) {
                            positions.push(p);
                        }
                    }

                    for &pos in &positions {
                        ds.machine_seq[new_m].insert(pos, node);
                        for idx in pos..ds.machine_seq[new_m].len() {
                            let nd = ds.machine_seq[new_m][idx];
                            pos_of_node[nd] = idx;
                        }
                        if let Some((test_mk, _)) = eval_disj(&ds, &mut buf) {
                            if test_mk < best_pass_mk {
                                best_pass_mk = test_mk;
                                best_pass_move = Some((node, new_m, new_pt, pos));
                            }
                        }
                        ds.machine_seq[new_m].remove(pos);
                        for idx in pos..ds.machine_seq[new_m].len() {
                            let nd = ds.machine_seq[new_m][idx];
                            pos_of_node[nd] = idx;
                        }
                    }

                    ds.machine_seq[cur_machine].insert(old_pos, node);
                    for idx in old_pos..ds.machine_seq[cur_machine].len() {
                        let nd = ds.machine_seq[cur_machine][idx];
                        pos_of_node[nd] = idx;
                    }
                    ds.node_machine[node] = cur_machine;
                    ds.node_pt[node] = cur_pt;
                }
            }

            let Some((node, best_m, best_pt, best_ins_pos)) = best_pass_move else { break };
            if best_pass_mk >= current_mk {
                break;
            }

            let cur_machine = ds.node_machine[node];
            let old_pos = pos_of_node[node];
            if old_pos >= ds.machine_seq[cur_machine].len() || ds.machine_seq[cur_machine][old_pos] != node {
                break;
            }
            ds.machine_seq[cur_machine].remove(old_pos);
            for idx in old_pos..ds.machine_seq[cur_machine].len() {
                let nd = ds.machine_seq[cur_machine][idx];
                pos_of_node[nd] = idx;
            }
            let ins = best_ins_pos.min(ds.machine_seq[best_m].len());
            ds.machine_seq[best_m].insert(ins, node);
            for idx in ins..ds.machine_seq[best_m].len() {
                let nd = ds.machine_seq[best_m][idx];
                pos_of_node[nd] = idx;
            }
            ds.node_machine[node] = best_m;
            ds.node_pt[node] = best_pt;

            current_mk = best_pass_mk;
        }

        if current_mk >= initial_mk {
            return Ok(None);
        }
        let Some((_, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
        let sol = disj_to_solution(pre, &ds, &buf.start)?;
        Ok(Some((sol, current_mk)))
    }

    enum MoveType { Swap{machine:usize,pos:usize}, Reassign{node:usize,new_machine:usize,new_pt:u32,insert_pos:usize} }

    fn tabu_search_hybrid(pre: &Pre, challenge: &Challenge, base_sol: &Solution, max_iterations: usize, tenure_base: usize) -> Result<Option<(Solution, u32)>> {
        let mut ds=build_disj_from_solution(pre,challenge,base_sol)?; let mut buf=EvalBuf::new(ds.n); let n=ds.n;
        let Some(init_eval)=eval_disj(&ds,&mut buf) else{return Ok(None)};
        let initial_mk=init_eval.0; let mut best_global_mk=initial_mk; let mut best_global_ds=ds.clone();
        let tenure=tenure_base.max(5); let tenure_delta=(tenure/3).max(2); let max_no_improve=(max_iterations/2).max(60);
        let mut tabu_swap: HashMap<(usize,usize),usize>=HashMap::with_capacity(tenure*8);
        let mut tabu_reassign: HashMap<(usize,usize),usize>=HashMap::with_capacity(tenure*4);
        let mut crit=vec![false;n]; let mut no_improve=0usize;
        let mut pseed: u64=(challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15)^(initial_mk as u64).wrapping_shl(16)^(n as u64).wrapping_mul(0x517CC1B727220A95);
        let mut tail=vec![0u32;n]; let mut back_deg=vec![0u16;n]; let mut back_stack: Vec<usize>=Vec::with_capacity(n);
        let mut machine_pred_node=vec![NONE_USIZE;n]; let mut job_pred_node=vec![NONE_USIZE;n];
        for j in 0..ds.num_jobs{let base=ds.job_offsets[j];let end=ds.job_offsets[j+1];for k in (base+1)..end{job_pred_node[k]=k-1;}}
        let kick_threshold=(max_no_improve*2/3).max(50); let mut kicks_left=3usize;
        for iter in 0..max_iterations {
            if no_improve>=max_no_improve{if kicks_left==0{break;}ds=best_global_ds.clone();no_improve=0;kicks_left-=1;tabu_swap.clear();tabu_reassign.clear();continue;}
            if no_improve>0&&no_improve%kick_threshold==0&&kicks_left>0 {
                let Some((_,kick_mk_node))=eval_disj(&ds,&mut buf) else{break};
                crit.fill(false); let mut u=kick_mk_node; while u!=NONE_USIZE{crit[u]=true;u=buf.best_pred[u];}
                let mut kick_swaps: Vec<(usize,usize)>=Vec::new();
                for m in 0..ds.num_machines{if ds.machine_seq[m].len()<=1{continue;}for i in 0..(ds.machine_seq[m].len()-1){if crit[ds.machine_seq[m][i]]&&crit[ds.machine_seq[m][i+1]]{kick_swaps.push((m,i));}}}
                if !kick_swaps.is_empty(){for _ in 0..2{pseed^=pseed.wrapping_shl(13);pseed^=pseed.wrapping_shr(7);pseed^=pseed.wrapping_shl(17);let idx=(pseed as usize)%kick_swaps.len();let (m,pos)=kick_swaps[idx];if pos+1<ds.machine_seq[m].len(){ds.machine_seq[m].swap(pos,pos+1);}}}
                kicks_left-=1; continue;
            }
            let Some((cur_mk,mk_node))=eval_disj(&ds,&mut buf) else{break};
            if iter>0{if cur_mk<best_global_mk{best_global_mk=cur_mk;best_global_ds=ds.clone();no_improve=0;}else{no_improve+=1;}}
            machine_pred_node.fill(NONE_USIZE);
            for seq in &ds.machine_seq{for i in 1..seq.len(){machine_pred_node[seq[i]]=seq[i-1];}}
            tail.fill(0); back_deg.fill(0);
            for i in 0..n{if ds.job_succ[i]!=NONE_USIZE{back_deg[i]+=1;}if buf.machine_succ[i]!=NONE_USIZE{back_deg[i]+=1;}}
            back_stack.clear(); for i in 0..n{if back_deg[i]==0{back_stack.push(i);}}
            while let Some(nd)=back_stack.pop(){let contrib=ds.node_pt[nd].saturating_add(tail[nd]);let jp=job_pred_node[nd];if jp!=NONE_USIZE{if contrib>tail[jp]{tail[jp]=contrib;}back_deg[jp]=back_deg[jp].saturating_sub(1);if back_deg[jp]==0{back_stack.push(jp);}}let mp=machine_pred_node[nd];if mp!=NONE_USIZE{if contrib>tail[mp]{tail[mp]=contrib;}back_deg[mp]=back_deg[mp].saturating_sub(1);if back_deg[mp]==0{back_stack.push(mp);}}}
            crit.fill(false); let mut u=mk_node; while u!=NONE_USIZE{crit[u]=true;u=buf.best_pred[u];}
            let mut best_move: Option<(MoveType,u32)>=None; let mut best_move_mk=u32::MAX;
            let mut fallback_move: Option<(MoveType,u32)>=None; let mut fallback_mk=u32::MAX;
            for m in 0..ds.num_machines {
                if ds.machine_seq[m].len()<=1{continue;}
                let mut blocks: Vec<(usize,usize)>=Vec::new(); let mut i=0;
                while i<ds.machine_seq[m].len(){if !crit[ds.machine_seq[m][i]]{i+=1;continue;}let bstart=i;let mut bend=i;while bend+1<ds.machine_seq[m].len(){let x=ds.machine_seq[m][bend];let y=ds.machine_seq[m][bend+1];if !crit[y]{break;}let end_x=buf.start[x].saturating_add(ds.node_pt[x]);if buf.start[y]!=end_x{break;}bend+=1;}if bend>bstart{blocks.push((bstart,bend));}i=bend+1;}
                for &(bstart,bend) in &blocks {
                    let block_len=bend-bstart+1; let mut swap_positions=[bstart,NONE_USIZE]; let num_swaps=if block_len>=3{swap_positions[1]=bend-1;2}else{1};
                    for si in 0..num_swaps {
                        let pos=swap_positions[si]; if pos+1>=ds.machine_seq[m].len(){continue;}
                        let node_u=ds.machine_seq[m][pos]; let node_v=ds.machine_seq[m][pos+1];
                        let est_mk=estimate_swap_mk_fm(node_u,node_v,&buf.start,&tail,&ds.node_pt,&job_pred_node,&ds.job_succ,&machine_pred_node,&buf.machine_succ);
                        let key=(node_u.min(node_v),node_u.max(node_v)); let is_tabu=tabu_swap.get(&key).map_or(false,|&exp|iter<exp); let aspiration=est_mk<best_global_mk;
                        if (!is_tabu||aspiration)&&est_mk<best_move_mk{best_move_mk=est_mk;best_move=Some((MoveType::Swap{machine:m,pos},est_mk));}
                        if est_mk<fallback_mk{fallback_mk=est_mk;fallback_move=Some((MoveType::Swap{machine:m,pos},est_mk));}
                    }
                }
            }
            let reassign_freq=3;
            if iter%reassign_freq==0 {
                for node in 0..n {
                    if !crit[node]{continue;}
                    let job=ds.node_job[node]; let op_idx=ds.node_op[node]; let product=pre.job_products[job];
                    let op_info=&pre.product_ops[product][op_idx]; if op_info.machines.len()<=1{continue;}
                    let cur_machine=ds.node_machine[node];
                    for &(new_m,new_pt) in &op_info.machines {
                        if new_m==cur_machine{continue;}
                        let key=(node,new_m); let is_tabu=tabu_reassign.get(&key).map_or(false,|&exp|iter<exp);
                        let positions=find_candidate_insert_positions_fm(&ds,&buf.start,node,new_m,new_pt,&job_pred_node);
                        for insert_pos in positions {
                            let est_mk=estimate_reassign_mk_fm(&ds,&buf.start,&tail,node,new_m,new_pt,insert_pos,&job_pred_node,&machine_pred_node,&buf.machine_succ);
                            let aspiration=est_mk<best_global_mk;
                            if (!is_tabu||aspiration)&&est_mk<best_move_mk{best_move_mk=est_mk;best_move=Some((MoveType::Reassign{node,new_machine:new_m,new_pt,insert_pos},est_mk));}
                            if est_mk<fallback_mk{fallback_mk=est_mk;fallback_move=Some((MoveType::Reassign{node,new_machine:new_m,new_pt,insert_pos},est_mk));}
                        }
                    }
                }
            }
            let chosen=best_move.or(fallback_move);
            match chosen {
                Some((MoveType::Swap{machine:m,pos},_)) => {
                    let node_a=ds.machine_seq[m][pos]; let node_b=ds.machine_seq[m][pos+1]; ds.machine_seq[m].swap(pos,pos+1);
                    pseed^=pseed.wrapping_shl(13);pseed^=pseed.wrapping_shr(7);pseed^=pseed.wrapping_shl(17);
                    let offset=(pseed%((2*tenure_delta+1) as u64)) as usize; let progress=(iter as f64)/(max_iterations as f64); let late_bonus=if progress>0.6{((progress-0.6)*10.0) as usize}else{0};
                    let this_tenure=(tenure+offset+late_bonus).saturating_sub(tenure_delta);
                    tabu_swap.insert((node_a.min(node_b),node_a.max(node_b)),iter+this_tenure);
                }
                Some((MoveType::Reassign{node,new_machine,new_pt,insert_pos},_)) => {
                    let old_machine=ds.node_machine[node]; let old_pos=ds.machine_seq[old_machine].iter().position(|&x|x==node);
                    if let Some(op)=old_pos{ds.machine_seq[old_machine].remove(op);}
                    ds.machine_seq[new_machine].insert(insert_pos,node); ds.node_machine[node]=new_machine; ds.node_pt[node]=new_pt;
                    pseed^=pseed.wrapping_shl(13);pseed^=pseed.wrapping_shr(7);pseed^=pseed.wrapping_shl(17);
                    let offset=(pseed%((2*tenure_delta+1) as u64)) as usize; let this_tenure=(tenure+offset).saturating_sub(tenure_delta/2);
                    tabu_reassign.insert((node,old_machine),iter+this_tenure);
                }
                None => break,
            }
        }
        let Some((final_mk,_))=eval_disj(&ds,&mut buf) else{return Ok(None)};
        if final_mk<best_global_mk{best_global_mk=final_mk;best_global_ds=ds.clone();}
        if best_global_mk>=initial_mk{return Ok(None);}
        ds=best_global_ds; let Some((_,_))=eval_disj(&ds,&mut buf) else{return Ok(None)};
        let sol=disj_to_solution(pre,&ds,&buf.start)?; Ok(Some((sol,best_global_mk)))
    }

    fn find_candidate_insert_positions_fm(
        ds: &DisjSchedule,
        starts: &[u32],
        node: usize,
        new_machine: usize,
        _new_pt: u32,
        job_pred: &[usize],
    ) -> Vec<usize> {
        let seq = &ds.machine_seq[new_machine];
        let len = seq.len();
        if len == 0 {
            return vec![0];
        }

        let jp = job_pred[node];
        let job_pred_end = if jp != NONE_USIZE {
            starts[jp].saturating_add(ds.node_pt[jp])
        } else {
            0
        };

        #[inline]
        fn lower_bound_start_gt(seq: &[usize], starts: &[u32], value: u32) -> usize {
            let mut lo = 0usize;
            let mut hi = seq.len();
            while lo < hi {
                let mid = (lo + hi) >> 1;
                if starts[seq[mid]] <= value {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo
        }

        #[inline]
        fn lower_bound_start_ge(seq: &[usize], starts: &[u32], value: u32) -> usize {
            let mut lo = 0usize;
            let mut hi = seq.len();
            while lo < hi {
                let mid = (lo + hi) >> 1;
                if starts[seq[mid]] < value {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo
        }

        let pos_after_jp = lower_bound_start_gt(seq, starts, job_pred_end).min(len);
        let cur_start = starts[node];
        let pos_by_cur = lower_bound_start_ge(seq, starts, cur_start).min(len);

        let mut out: Vec<usize> = Vec::with_capacity(5);
        #[inline]
        fn push_uniq(v: &mut Vec<usize>, p: usize, len: usize) {
            if p <= len && !v.contains(&p) {
                v.push(p);
            }
        }

        push_uniq(&mut out, pos_after_jp, len);
        push_uniq(&mut out, pos_after_jp.saturating_sub(1), len);

        push_uniq(&mut out, pos_by_cur, len);
        push_uniq(&mut out, pos_by_cur.saturating_sub(1), len);

        push_uniq(&mut out, 0, len);
        push_uniq(&mut out, len, len);

        if out.is_empty() {
            out.push(len);
        }
        if out.len() > 5 {
            out.truncate(5);
        }
        out
    }

    fn estimate_reassign_mk_fm(ds: &DisjSchedule, heads: &[u32], tails: &[u32], node: usize, new_machine: usize, new_pt: u32, insert_pos: usize, job_pred: &[usize], machine_pred: &[usize], machine_succ: &[usize]) -> u32 {
        let jp=job_pred[node]; let js=ds.job_succ[node]; let old_mp=machine_pred[node]; let old_ms=machine_succ[node];
        let jp_end=if jp!=NONE_USIZE{heads[jp].saturating_add(ds.node_pt[jp])}else{0};
        let new_seq=&ds.machine_seq[new_machine];
        let new_mp_end=if insert_pos>0&&!new_seq.is_empty(){let pred=new_seq[insert_pos.min(new_seq.len())-1];heads[pred].saturating_add(ds.node_pt[pred])}else{0};
        let new_start=jp_end.max(new_mp_end); let new_end=new_start.saturating_add(new_pt);
        let js_tail=if js!=NONE_USIZE{ds.node_pt[js].saturating_add(tails[js])}else{0};
        let new_ms_tail=if insert_pos<new_seq.len(){let succ=new_seq[insert_pos];ds.node_pt[succ].saturating_add(tails[succ])}else{0};
        let node_path=new_end.saturating_add(js_tail.max(new_ms_tail));
        let old_reconnect=if old_mp!=NONE_USIZE&&old_ms!=NONE_USIZE{let old_mp_end=heads[old_mp].saturating_add(ds.node_pt[old_mp]);old_mp_end.saturating_add(ds.node_pt[old_ms]).saturating_add(tails[old_ms])}else{0};
        node_path.max(old_reconnect)
    }

    #[inline]
    fn estimate_swap_mk_fm(u: usize, v: usize, heads: &[u32], tails: &[u32], pt: &[u32], job_pred: &[usize], job_succ: &[usize], machine_pred: &[usize], machine_succ: &[usize]) -> u32 {
        let mp_u=machine_pred[u];let ms_v=machine_succ[v];let jp_v=job_pred[v];let jp_u=job_pred[u];let js_u=job_succ[u];let js_v=job_succ[v];
        let r_jp_v=if jp_v!=NONE_USIZE{heads[jp_v].saturating_add(pt[jp_v])}else{0};let r_mp_u=if mp_u!=NONE_USIZE{heads[mp_u].saturating_add(pt[mp_u])}else{0};
        let new_r_v=r_jp_v.max(r_mp_u);let r_jp_u=if jp_u!=NONE_USIZE{heads[jp_u].saturating_add(pt[jp_u])}else{0};let new_r_u=r_jp_u.max(new_r_v.saturating_add(pt[v]));
        let q_js_u=if js_u!=NONE_USIZE{pt[js_u].saturating_add(tails[js_u])}else{0};let q_ms_v=if ms_v!=NONE_USIZE{pt[ms_v].saturating_add(tails[ms_v])}else{0};
        let new_q_u=q_js_u.max(q_ms_v);let q_js_v=if js_v!=NONE_USIZE{pt[js_v].saturating_add(tails[js_v])}else{0};let new_q_v=q_js_v.max(pt[u].saturating_add(new_q_u));
        (new_r_v.saturating_add(pt[v]).saturating_add(new_q_v)).max(new_r_u.saturating_add(pt[u]).saturating_add(new_q_u))
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
        let mut rules: Vec<Rule> = vec![Rule::Adaptive,Rule::BnHeavy,Rule::EndTight,Rule::CriticalPath,Rule::MostWork,Rule::LeastFlex,Rule::Regret,Rule::ShortestProc];
        if allow_flex_balance { rules.push(Rule::FlexBalance); }
        let mut best_makespan = greedy_mk; let mut best_solution: Option<Solution> = Some(greedy_sol); let mut top_solutions: Vec<(Solution,u32)> = Vec::new();
        let target_margin: u32 = ((pre.avg_op_min*(0.9+0.9*pre.high_flex+0.6*pre.jobshopness)).max(1.0)) as u32;
        let route_w_base: f64 = if pre.chaotic_like { 0.0 } else { (0.050+0.12*pre.high_flex+0.10*pre.jobshopness+(0.08/pre.flex_avg.max(1.0))).clamp(0.04,0.28) };

        if pre.flow_route.is_some()&&pre.flow_pt_by_job.is_some() {
            if let Ok((sol,mk))=neh_reentrant_flow_solution(pre,challenge.num_jobs,challenge.num_machines) {
                if mk<best_makespan{best_makespan=mk;best_solution=Some(sol.clone());save_solution(&sol)?;}
                push_top_solutions(&mut top_solutions,&sol,mk,15);
            }
        }
        let mut ranked: Vec<(Rule,u32,Solution)>=Vec::with_capacity(rules.len());
        for &rule in &rules {
            let (sol,mk)=construct_solution_conflict(challenge,pre,rule,0,None,&mut rng,None,None,None,0.0)?;
            if mk<best_makespan{best_makespan=mk;best_solution=Some(sol.clone());save_solution(&sol)?;}
            push_top_solutions(&mut top_solutions,&sol,mk,20); ranked.push((rule,mk,sol));
        }
        ranked.sort_by_key(|x|x.1);

        if let Ok((jc_sol, jc_mk)) = construct_solution_job_centric(challenge, pre) {
            if jc_mk < best_makespan {
                best_makespan = jc_mk;
                best_solution = Some(jc_sol.clone());
                save_solution(&jc_sol)?;
            }
            push_top_solutions(&mut top_solutions, &jc_sol, jc_mk, 20);
        }
        
        let r0=ranked[0].0; let r1=ranked.get(1).map(|x|x.0).unwrap_or(r0); let r2=ranked.get(2).map(|x|x.0).unwrap_or(r1);
        let mut rule_best: Vec<u32>=vec![u32::MAX;10]; let mut rule_tries: Vec<u32>=vec![0u32;10];
        for (rr,mk,_) in &ranked{let idx=rule_idx(*rr);rule_best[idx]=rule_best[idx].min(*mk);rule_tries[idx]=rule_tries[idx].saturating_add(1);}

        let base = best_solution.as_ref().ok_or_else(|| anyhow!("No initial solution found"))?;
        let mut learned_jb=Some(job_bias_from_solution(pre, base)?);
        let mut learned_mp=Some(machine_penalty_from_solution(pre,base,challenge.num_machines)?);
        let mut learned_rp=if route_w_base>0.0{Some(route_pref_from_solution_lite(pre,base,challenge)?)}else{None};
        let mut learn_updates_left=10usize;
        let num_restarts=effort.fjsp_medium_iters;
        let mut k_hi=if pre.flex_avg>8.0{6}else if pre.flex_avg>6.5{4}else if pre.flex_avg>4.0{5}else{6};
        if pre.jobshopness>0.60&&k_hi<6{k_hi+=1;} k_hi=k_hi.min(6).max(2);
        let mut stuck: usize=0;
        for r in 0..num_restarts {
            let late=r>=(num_restarts*2)/3;
            let (k_min,k_max)=if stuck>170{(4usize,6usize.min(k_hi))}else if stuck>90{(3usize,6usize.min(k_hi.max(4)))}else if stuck>35{(2usize,k_hi)}else{(2usize,k_hi.min(4))};
            let rule=if r<35{let u: f64=rng.gen();if allow_flex_balance&&pre.high_flex>0.82&&u<0.10{Rule::FlexBalance}else if u<0.52{r0}else if u<0.80{r1}else if u<0.92{r2}else{rules[rng.gen_range(0..rules.len())]}}
                else{choose_rule_bandit(&mut rng,&rules,&rule_best,&rule_tries,best_makespan,target_margin,stuck,pre.chaotic_like,late)};
            let k=if k_max<=k_min{k_min}else{rng.gen_range(k_min..=k_max)};
            let learn_base=if pre.chaotic_like{0.0}else{(0.08+0.22*pre.jobshopness+0.18*pre.high_flex).clamp(0.05,0.42)};
            let learn_boost=(1.0+0.35*((stuck as f64)/120.0).clamp(0.0,1.0)).clamp(1.0,1.35);
            let learn_p=(learn_base*learn_boost).clamp(0.0,0.60);
            let use_learn=learned_jb.is_some()&&learned_mp.is_some()&&rng.gen::<f64>()<learn_p&&(route_w_base==0.0||learned_rp.is_some());
            let target=if best_makespan<(u32::MAX/2){Some(best_makespan.saturating_add(target_margin))}else{None};
            let (sol,mk)=if use_learn{construct_solution_conflict(challenge,pre,rule,k,target,&mut rng,learned_jb.as_deref(),learned_mp.as_deref(),learned_rp.as_ref(),route_w_base)?}
                else{construct_solution_conflict(challenge,pre,rule,k,target,&mut rng,None,None,None,0.0)?};
            let ridx=rule_idx(rule);rule_tries[ridx]=rule_tries[ridx].saturating_add(1);rule_best[ridx]=rule_best[ridx].min(mk);
            if mk<best_makespan{best_makespan=mk;best_solution=Some(sol.clone());save_solution(&sol)?;stuck=0;}else{stuck=stuck.saturating_add(1);}
            push_top_solutions(&mut top_solutions,&sol,mk,20);

            if learn_updates_left > 0 && !pre.chaotic_like && !top_solutions.is_empty() {
                let refresh = (r > 0 && r % 35 == 0) || stuck == 90 || stuck == 170;
                if refresh {
                    let pool_size = top_solutions.len().min(10);
                    let mut elite: Vec<(u32, usize)> = top_solutions
                        .iter()
                        .take(pool_size)
                        .enumerate()
                        .map(|(i, (_s, mk))| (*mk, i))
                        .collect();
                    elite.sort_by_key(|x| x.0);
                    let rep_i = elite[pool_size / 2].1;
                    let rep_sol = &top_solutions[rep_i].0;

                    learned_jb = Some(job_bias_from_solution(pre, rep_sol)?);
                    learned_mp = Some(machine_penalty_from_solution(pre, rep_sol, challenge.num_machines)?);
                    if route_w_base > 0.0 {
                        learned_rp = Some(route_pref_from_solution_lite(pre, rep_sol, challenge)?);
                    }
                    learn_updates_left -= 1;
                }
            }
        }
        let route_w_ls: f64=if route_w_base>0.0{(route_w_base*1.40).clamp(route_w_base,0.40)}else{0.0};
        let mut refine_results: Vec<(Solution,u32)>=Vec::new();
        for (base_sol,_) in top_solutions.iter() {
            let jb=job_bias_from_solution(pre,base_sol)?; let mp=machine_penalty_from_solution(pre,base_sol,challenge.num_machines)?;
            let rp=if route_w_ls>0.0{Some(route_pref_from_solution_lite(pre,base_sol,challenge)?)}else{None};
            let target_ls=if best_makespan<(u32::MAX/2){Some(best_makespan.saturating_add(target_margin/2))}else{None};
            for attempt in 0..10 {
                let rule=if pre.chaotic_like{match attempt%4{0=>Rule::Adaptive,1=>Rule::ShortestProc,2=>Rule::MostWork,_=>Rule::Regret}}else{match attempt{0=>r0,1=>Rule::Adaptive,2=>Rule::BnHeavy,3=>Rule::EndTight,4=>Rule::Regret,5=>Rule::CriticalPath,6=>Rule::LeastFlex,7=>Rule::MostWork,8=>if allow_flex_balance{Rule::FlexBalance}else{r1},_=>r1}};
                let k=match attempt%4{0=>2,1=>3,2=>4,_=>2}.min(k_hi);
                let (sol,mk)=construct_solution_conflict(challenge,pre,rule,k,target_ls,&mut rng,Some(&jb),Some(&mp),rp.as_ref(),if rp.is_some(){route_w_ls}else{0.0})?;
                if mk<best_makespan{best_makespan=mk;best_solution=Some(sol.clone());save_solution(&sol)?;}
                refine_results.push((sol,mk));
            }
        }
        for (sol,mk) in refine_results{push_top_solutions(&mut top_solutions,&sol,mk,15);}
        let ts_starts=top_solutions.len().min(12); let ts_iters=(effort.fjsp_medium_iters*3/4).max(60);
        let ts_tenure=((pre.total_ops as f64).sqrt() as usize).clamp(5,12);
        for i in 0..ts_starts {
            let base_sol=&top_solutions[i].0;
            if let Some((sol2,mk2))=tabu_search_hybrid(pre,challenge,base_sol,ts_iters,ts_tenure)?{
                if mk2<best_makespan{best_makespan=mk2;best_solution=Some(sol2.clone());save_solution(&sol2)?;}
            }
        }
        if let Some(sol) = best_solution.as_ref() {
            if let Some((improved_sol, improved_mk)) = greedy_reassign_pass(pre, challenge, sol)? {
                if improved_mk < best_makespan {
                    best_makespan = improved_mk;
                    best_solution = Some(improved_sol.clone());
                    save_solution(&improved_sol)?;
                }
            }
        }

        if let Some(sol) = best_solution.as_ref() {
            if let Ok(Some((ecr_sol, ecr_mk))) = exhaustive_critical_reroute_pass(pre, challenge, sol) {
                if ecr_mk < best_makespan { best_makespan = ecr_mk; best_solution = Some(ecr_sol.clone()); save_solution(&ecr_sol)?; }
            }
        }

        let cb_passes = if effort.fjsp_medium_iters > 200 { 6 } else { 5 };
        let cb_iters = (pre.total_ops / 8).max(30).min(120);
        let cb_no_improve = cb_iters / 2;

        let cb_top_n = top_solutions.len().min(8);
        for ci in 0..cb_top_n {
            let base_sol = &top_solutions[ci].0;
            if let Ok(Some((cb_sol, cb_mk))) = critical_block_move_local_search_ex(pre, challenge, base_sol, cb_passes, cb_iters, cb_no_improve) {
                if cb_mk < best_makespan {
                    best_makespan = cb_mk;
                    best_solution = Some(cb_sol.clone());
                    save_solution(&cb_sol)?;
                }
                push_top_solutions(&mut top_solutions, &cb_sol, cb_mk, 20);
            }
        }

        if let Some(sol) = best_solution.as_ref() {
            if let Ok(Some((cb_sol, cb_mk))) = critical_block_move_local_search_ex(pre, challenge, sol, cb_passes, cb_iters, cb_no_improve) {
                if cb_mk < best_makespan {
                    best_makespan = cb_mk;
                    best_solution = Some(cb_sol.clone());
                    save_solution(&cb_sol)?;
                }
            }
        }

        if let Some(sol) = best_solution.as_ref() {
            if let Ok(Some((bmr_sol, bmr_mk))) = bottleneck_machine_relief_pass(pre, challenge, sol, 20) {
                if bmr_mk < best_makespan {
                    best_makespan = bmr_mk;
                    best_solution = Some(bmr_sol.clone());
                    save_solution(&bmr_sol)?;
                }
                push_top_solutions(&mut top_solutions, &bmr_sol, bmr_mk, 20);
            }
        }

        let ils_rounds = if effort.fjsp_medium_iters > 300 { 30 } else { 20 };
        let mut ils_best_sol = best_solution.clone();
        let mut ils_best_mk = best_makespan;
        let mut ils_no_improve = 0usize;
        let ils_max_no_improve = (ils_rounds * 3) / 4 + 3;

        const NUM_PERTURB_OPS: usize = 4;
        const LEARNING_RATE: f64 = 0.25;
        let mut op_weights = vec![10.0; NUM_PERTURB_OPS];

        for ils_r in 0..ils_rounds {
            if ils_no_improve >= ils_max_no_improve { break; }
            let Some(base) = ils_best_sol.as_ref() else { break };
            let mut ds = build_disj_from_solution(pre, challenge, base)?;
            let mut buf = EvalBuf::new(ds.n);
            let Some((_, mk_node)) = eval_disj(&ds, &mut buf) else { continue };
            let n = ds.n;
            let mut perturb_seed: u64 = (ils_r as u64).wrapping_mul(0x517CC1B727220A95)
                .wrapping_add(ils_best_mk as u64)
                .wrapping_add(challenge.seed[0] as u64)
                .wrapping_add((ils_r as u64).wrapping_mul(0xDEADBEEF));
            let k_perturb = (3 + ils_r / 3).min(8);

            let total_weight: f64 = op_weights.iter().sum();
            let mut choice_val = rng.gen::<f64>() * total_weight;
            let mut strategy = NUM_PERTURB_OPS - 1;
            for (i, &weight) in op_weights.iter().enumerate() {
                if choice_val < weight {
                    strategy = i;
                    break;
                }
                choice_val -= weight;
            }

            if strategy == 0 {
                let mut crit_nodes: Vec<usize> = Vec::with_capacity(64);
                let mut u = mk_node;
                while u != NONE_USIZE { crit_nodes.push(u); u = buf.best_pred[u]; }
                let mut perturbed = 0; let mut attempts = 0;
                while perturbed < k_perturb && attempts < crit_nodes.len() * 4 {
                    attempts += 1;
                    perturb_seed ^= perturb_seed.wrapping_shl(13); perturb_seed ^= perturb_seed.wrapping_shr(7); perturb_seed ^= perturb_seed.wrapping_shl(17);
                    if crit_nodes.is_empty() { break; }
                    let idx = (perturb_seed as usize) % crit_nodes.len();
                    let node = crit_nodes[idx];
                    let job = ds.node_job[node]; let op_idx = ds.node_op[node]; let product = pre.job_products[job];
                    let op_info = &pre.product_ops[product][op_idx];
                    if op_info.machines.len() <= 1 { continue; }
                    let cur_machine = ds.node_machine[node];
                    perturb_seed ^= perturb_seed.wrapping_shl(13); perturb_seed ^= perturb_seed.wrapping_shr(7); perturb_seed ^= perturb_seed.wrapping_shl(17);
                    let alt_idx = (perturb_seed as usize) % op_info.machines.len();
                    let (new_m, new_pt) = op_info.machines[alt_idx];
                    if new_m == cur_machine { continue; }
                    let old_pos = match ds.machine_seq[cur_machine].iter().position(|&x| x == node) { Some(p) => p, None => continue };
                    ds.machine_seq[cur_machine].remove(old_pos);
                    ds.node_machine[node] = new_m; ds.node_pt[node] = new_pt;
                    let cur_start = buf.start[node];
                    let mut ins_pos = ds.machine_seq[new_m].len();
                    for (ki, &nd) in ds.machine_seq[new_m].iter().enumerate() { if buf.start[nd] >= cur_start { ins_pos = ki; break; } }
                    ds.machine_seq[new_m].insert(ins_pos, node);
                    perturbed += 1;
                }
            } else if strategy == 1 {
                let mut machine_loads = vec![0u32; ds.num_machines];
                for node in 0..n { let m = ds.node_machine[node]; machine_loads[m] = machine_loads[m].saturating_add(ds.node_pt[node]); }
                let worst_m = machine_loads.iter().enumerate().max_by_key(|&(_, &v)| v).map(|(i, _)| i).unwrap_or(0);
                if ds.machine_seq[worst_m].is_empty() { continue; }
                let mut perturbed = 0; let mut attempts = 0;
                while perturbed < k_perturb && attempts < ds.machine_seq[worst_m].len() * 4 {
                    attempts += 1;
                    perturb_seed ^= perturb_seed.wrapping_shl(13); perturb_seed ^= perturb_seed.wrapping_shr(7); perturb_seed ^= perturb_seed.wrapping_shl(17);
                    let cur_seq_len = ds.machine_seq[worst_m].len();
                    if cur_seq_len == 0 { break; }
                    let seq_idx = (perturb_seed as usize) % cur_seq_len;
                    let node = ds.machine_seq[worst_m][seq_idx];
                    let job = ds.node_job[node]; let op_idx = ds.node_op[node]; let product = pre.job_products[job];
                    let op_info = &pre.product_ops[product][op_idx];
                    if op_info.machines.len() <= 1 { continue; }
                    let mut best_alt_m = worst_m; let mut best_alt_pt = ds.node_pt[node];
                    for &(am, apt) in &op_info.machines { if am != worst_m && apt < best_alt_pt { best_alt_pt = apt; best_alt_m = am; } }
                    if best_alt_m == worst_m { continue; }
                    let old_pos = match ds.machine_seq[worst_m].iter().position(|&x| x == node) { Some(p) => p, None => continue };
                    ds.machine_seq[worst_m].remove(old_pos);
                    ds.node_machine[node] = best_alt_m; ds.node_pt[node] = best_alt_pt;
                    let cur_start = buf.start[node];
                    let mut ins_pos = ds.machine_seq[best_alt_m].len();
                    for (ki, &nd) in ds.machine_seq[best_alt_m].iter().enumerate() { if buf.start[nd] >= cur_start { ins_pos = ki; break; } }
                    ds.machine_seq[best_alt_m].insert(ins_pos, node);
                    perturbed += 1;
                }
            } else if strategy == 2 {
                let mut crit_nodes: Vec<usize> = Vec::with_capacity(64);
                let mut crit_machines: Vec<usize> = Vec::with_capacity(16);
                let mut u = mk_node;
                while u != NONE_USIZE {
                    crit_nodes.push(u);
                    let m = ds.node_machine[u];
                    if !crit_machines.contains(&m) { crit_machines.push(m); }
                    u = buf.best_pred[u];
                }
                let k_reassign = k_perturb / 2;
                let mut perturbed = 0; let mut attempts = 0;
                while perturbed < k_reassign && attempts < crit_nodes.len() * 3 {
                    attempts += 1;
                    perturb_seed ^= perturb_seed.wrapping_shl(13); perturb_seed ^= perturb_seed.wrapping_shr(7); perturb_seed ^= perturb_seed.wrapping_shl(17);
                    if crit_nodes.is_empty() { break; }
                    let idx = (perturb_seed as usize) % crit_nodes.len();
                    let node = crit_nodes[idx];
                    let job = ds.node_job[node]; let op_idx = ds.node_op[node]; let product = pre.job_products[job];
                    let op_info = &pre.product_ops[product][op_idx];
                    if op_info.machines.len() <= 1 { continue; }
                    let cur_machine = ds.node_machine[node];
                    perturb_seed ^= perturb_seed.wrapping_shl(13); perturb_seed ^= perturb_seed.wrapping_shr(7); perturb_seed ^= perturb_seed.wrapping_shl(17);
                    let alt_idx = (perturb_seed as usize) % op_info.machines.len();
                    let (new_m, new_pt) = op_info.machines[alt_idx];
                    if new_m == cur_machine { continue; }
                    let old_pos = match ds.machine_seq[cur_machine].iter().position(|&x| x == node) { Some(p) => p, None => continue };
                    ds.machine_seq[cur_machine].remove(old_pos);
                    ds.node_machine[node] = new_m; ds.node_pt[node] = new_pt;
                    let cur_start = buf.start[node];
                    let mut ins_pos = ds.machine_seq[new_m].len();
                    for (ki, &nd) in ds.machine_seq[new_m].iter().enumerate() { if buf.start[nd] >= cur_start { ins_pos = ki; break; } }
                    ds.machine_seq[new_m].insert(ins_pos, node);
                    perturbed += 1;
                }
                let k_swaps = k_perturb - k_reassign;
                let mut swapped = 0;
                for _ in 0..(k_swaps * 4) {
                    if swapped >= k_swaps || crit_machines.is_empty() { break; }
                    perturb_seed ^= perturb_seed.wrapping_shl(13); perturb_seed ^= perturb_seed.wrapping_shr(7); perturb_seed ^= perturb_seed.wrapping_shl(17);
                    let m = crit_machines[(perturb_seed as usize) % crit_machines.len()];
                    if ds.machine_seq[m].len() < 2 { continue; }
                    perturb_seed ^= perturb_seed.wrapping_shl(13); perturb_seed ^= perturb_seed.wrapping_shr(7); perturb_seed ^= perturb_seed.wrapping_shl(17);
                    let pos = (perturb_seed as usize) % (ds.machine_seq[m].len() - 1);
                    ds.machine_seq[m].swap(pos, pos + 1);
                    swapped += 1;
                }
            } else {
                let mut swapped = 0; let mut attempts = 0;
                while swapped < k_perturb && attempts < 100 {
                    attempts += 1;
                    perturb_seed ^= perturb_seed.wrapping_shl(13); perturb_seed ^= perturb_seed.wrapping_shr(7); perturb_seed ^= perturb_seed.wrapping_shl(17);
                    let m = (perturb_seed as usize) % ds.num_machines;
                    if ds.machine_seq[m].len() < 2 { continue; }
                    perturb_seed ^= perturb_seed.wrapping_shl(13); perturb_seed ^= perturb_seed.wrapping_shr(7); perturb_seed ^= perturb_seed.wrapping_shl(17);
                    let pos = (perturb_seed as usize) % (ds.machine_seq[m].len() - 1);
                    ds.machine_seq[m].swap(pos, pos + 1);
                    swapped += 1;
                }
            }

            let Some((_, _)) = eval_disj(&ds, &mut buf) else { ils_no_improve += 1; continue };
            let perturbed_sol = match disj_to_solution(pre, &ds, &buf.start) { Ok(s) => s, Err(_) => { ils_no_improve += 1; continue; } };
            let after_reassign = match greedy_reassign_pass(pre, challenge, &perturbed_sol)? {
                Some((s, mk)) => (s, mk),
                None => { if let Some((pmk, _)) = eval_disj(&ds, &mut buf) { (perturbed_sol.clone(), pmk) } else { ils_no_improve += 1; continue; } }
            };
            let ls_result = critical_block_move_local_search_ex(pre, challenge, &after_reassign.0, cb_passes, cb_iters, cb_no_improve);
            let (candidate_sol, candidate_mk) = if let Ok(Some((ls_sol, ls_mk))) = ls_result {
                (ls_sol, ls_mk)
            } else {
                (after_reassign.0.clone(), after_reassign.1)
            };
            
            let reward = if candidate_mk < ils_best_mk {
                let improvement = (ils_best_mk - candidate_mk) as f64;
                let normalized_improvement = improvement / pre.avg_op_min.max(1.0);
                let mut r = 1.0 + normalized_improvement.min(10.0);
                if candidate_mk < best_makespan {
                    r *= 2.5;
                }
                r
            } else {
                0.05
            };

            op_weights[strategy] = op_weights[strategy] * (1.0 - LEARNING_RATE) + reward * LEARNING_RATE;

            if candidate_mk < best_makespan {
                best_makespan = candidate_mk; best_solution = Some(candidate_sol.clone()); save_solution(&candidate_sol)?;
            }
            
            if candidate_mk < ils_best_mk {
                ils_best_mk = candidate_mk; ils_best_sol = Some(candidate_sol); ils_no_improve = 0;
            } else {
                ils_no_improve += 1;
            }
        }

        if let Some(sol) = best_solution.as_ref() {
            if let Ok(Some((bmr_sol, bmr_mk))) = bottleneck_machine_relief_pass(pre, challenge, sol, 15) {
                if bmr_mk < best_makespan {
                    best_makespan = bmr_mk;
                    best_solution = Some(bmr_sol.clone());
                    save_solution(&bmr_sol)?;
                }
            }
        }

        {
            let alns_rounds = if effort.fjsp_medium_iters > 300 { 50 } else { 35 };
            let mut alns_sa_mk = best_makespan;
            let mut alns_sa_sol = best_solution.clone();
            let mut alns_best_mk = best_makespan;
            let mut alns_no_improve = 0usize;
            let alns_max_no_improve = alns_rounds / 2 + 4;
            let t_init = (best_makespan as f64) * 0.015;
            let t_final = (best_makespan as f64) * 0.0005;
            let cooling = if alns_rounds > 1 { (t_final / t_init.max(1.0)).powf(1.0 / (alns_rounds as f64)) } else { 0.95 };
            let mut temperature = t_init;
            let mut alns_seed: u64 = (challenge.seed[0] as u64).wrapping_mul(0xB7E151628AED2A6Bu64)
                .wrapping_add(best_makespan as u64)
                .wrapping_add(0x9E3779B97F4A7C15u64);

            for alns_r in 0..alns_rounds {
                if alns_no_improve >= alns_max_no_improve { break; }
                let Some(base) = alns_sa_sol.as_ref() else { break };
                let mut ds = match build_disj_from_solution(pre, challenge, base) { Ok(d) => d, Err(_) => { alns_no_improve += 1; temperature *= cooling; continue; } };
                let mut buf = EvalBuf::new(ds.n);
                let Some((_cur_mk, mk_node)) = eval_disj(&ds, &mut buf) else { alns_no_improve += 1; temperature *= cooling; continue };
                let n = ds.n;

                let mut crit_set = vec![false; n];
                let mut uu = mk_node;
                while uu != NONE_USIZE { crit_set[uu] = true; uu = buf.best_pred[uu]; }

                alns_seed ^= alns_seed.wrapping_shl(13); alns_seed ^= alns_seed.wrapping_shr(7); alns_seed ^= alns_seed.wrapping_shl(17);
                let k_destroy = 6 + (alns_r % 9);

                let mut scored: Vec<(f64, bool, usize)> = Vec::with_capacity(n);
                for nd in 0..n {
                    let job = ds.node_job[nd];
                    let op_idx = ds.node_op[nd];
                    let product = pre.job_products[job];
                    let flex = pre.product_ops[product][op_idx].flex.max(1) as f64;
                    let flex_inv = 1.0 / flex;
                    let m = ds.node_machine[nd];
                    let scarcity = pre.machine_scarcity[m];
                    scored.push((scarcity * flex_inv, crit_set[nd], nd));
                }

                scored.sort_unstable_by(|a, b| {
                    b.0.total_cmp(&a.0).then_with(|| b.1.cmp(&a.1))
                });

                let mut destroyed: Vec<usize> = Vec::new();
                if !scored.is_empty() {
                    let base = k_destroy.min(scored.len());
                    let window = if scored.len() > k_destroy {
                        (k_destroy + ((alns_seed as usize) % k_destroy.max(1))).min(scored.len())
                    } else {
                        base
                    };

                    destroyed = scored.iter().take(window).map(|x| x.2).collect();

                    for i in 0..destroyed.len() {
                        alns_seed ^= alns_seed.wrapping_shl(13); alns_seed ^= alns_seed.wrapping_shr(7); alns_seed ^= alns_seed.wrapping_shl(17);
                        let j = i + (alns_seed as usize) % (destroyed.len() - i);
                        destroyed.swap(i, j);
                    }
                    destroyed.truncate(k_destroy.min(destroyed.len()));
                }

                if destroyed.is_empty() { alns_no_improve += 1; temperature *= cooling; continue; }

                let mut removed_set = vec![false; n];
                for &nd in &destroyed {
                    removed_set[nd] = true;
                    let m = ds.node_machine[nd];
                    if let Some(pos) = ds.machine_seq[m].iter().position(|&x| x == nd) {
                        ds.machine_seq[m].remove(pos);
                    }
                }

                let _ = eval_disj(&ds, &mut buf);

                let mut to_ins: Vec<usize> = destroyed.clone();
                let max_repair = to_ins.len() * 6;
                let mut rep_iter = 0;
                while !to_ins.is_empty() && rep_iter < max_repair {
                    rep_iter += 1;
                    let mut best_regret = -1.0f64;
                    let mut best_ni = 0usize;
                    let mut best_ins_m = NONE_USIZE;
                    let mut best_ins_pt = 0u32;
                    let mut best_ins_pos = 0usize;
                    let mut found_any = false;

                    for (ti, &nd) in to_ins.iter().enumerate() {
                        let job = ds.node_job[nd];
                        let op_idx = ds.node_op[nd];
                        let product = pre.job_products[job];
                        let op_info = &pre.product_ops[product][op_idx];
                        let job_start = ds.job_offsets[job];
                        let jp = if nd > job_start { nd - 1 } else { NONE_USIZE };
                        let jp_end = if jp != NONE_USIZE && !removed_set[jp] {
                            buf.start[jp].saturating_add(ds.node_pt[jp])
                        } else if jp != NONE_USIZE && removed_set[jp] {
                            u32::MAX / 2
                        } else { 0u32 };
                        if jp_end >= u32::MAX / 2 { continue; }

                        let mut node_best = u32::MAX;
                        let mut node_second = u32::MAX;
                        let mut node_bm = NONE_USIZE;
                        let mut node_bpt = 0u32;
                        let mut node_bpos = 0usize;

                        for &(m, pt) in &op_info.machines {
                            let seq = &ds.machine_seq[m];
                            let mut pos_costs: Vec<(usize, u32)> = Vec::with_capacity(seq.len() + 1);
                            for pos in 0..=seq.len() {
                                let mp_end = if pos > 0 {
                                    let pred = seq[pos - 1];
                                    if !removed_set[pred] { buf.start[pred].saturating_add(ds.node_pt[pred]) } else { 0 }
                                } else { 0 };
                                let st = jp_end.max(mp_end);
                                let et = st.saturating_add(pt);
                                let suf = pre.product_suf_min[product][op_idx] as u32;
                                let succ_pen = if pos < seq.len() {
                                    let succ = seq[pos];
                                    if !removed_set[succ] {
                                        let new_succ_st = et.max(buf.start[succ]);
                                        if new_succ_st > buf.start[succ] { (new_succ_st - buf.start[succ]) / 2 } else { 0 }
                                    } else { 0 }
                                } else { 0 };
                                let cost = et.saturating_add(suf).saturating_add(succ_pen);
                                pos_costs.push((pos, cost));
                            }
                            pos_costs.sort_by_key(|&(_, c)| c);
                            for &(pos, cost) in pos_costs.iter().take(3) {
                                if cost < node_best {
                                    node_second = node_best;
                                    node_best = cost;
                                    node_bm = m; node_bpt = pt; node_bpos = pos;
                                } else if cost < node_second {
                                    node_second = cost;
                                }
                            }
                        }

                        if node_bm == NONE_USIZE { continue; }
                        found_any = true;
                        let regret = if node_second < u32::MAX { (node_second - node_best) as f64 } else { pre.avg_op_min * 3.0 };
                        if regret > best_regret {
                            best_regret = regret; best_ni = ti;
                            best_ins_m = node_bm; best_ins_pt = node_bpt;
                            best_ins_pos = node_bpos;
                        }
                    }

                    if !found_any || best_ins_m == NONE_USIZE {
                        for ti in 0..to_ins.len() {
                            let nd = to_ins[ti];
                            let job = ds.node_job[nd]; let op_idx = ds.node_op[nd]; let product = pre.job_products[job];
                            let op_info = &pre.product_ops[product][op_idx];
                            if let Some(&(m, pt)) = op_info.machines.first() {
                                let ins = ds.machine_seq[m].len();
                                ds.machine_seq[m].insert(ins, nd);
                                ds.node_machine[nd] = m; ds.node_pt[nd] = pt;
                                removed_set[nd] = false; to_ins.remove(ti);
                                break;
                            }
                        }
                        continue;
                    }

                    let nd = to_ins[best_ni];
                    let ins = best_ins_pos.min(ds.machine_seq[best_ins_m].len());
                    ds.machine_seq[best_ins_m].insert(ins, nd);
                    ds.node_machine[nd] = best_ins_m; ds.node_pt[nd] = best_ins_pt;
                    removed_set[nd] = false; to_ins.remove(best_ni);
                    let _ = eval_disj(&ds, &mut buf);
                }
                for &nd in &to_ins {
                    let job = ds.node_job[nd]; let op_idx = ds.node_op[nd]; let product = pre.job_products[job];
                    let op_info = &pre.product_ops[product][op_idx];
                    if let Some(&(m, pt)) = op_info.machines.first() {
                        let ins = ds.machine_seq[m].len();
                        ds.machine_seq[m].insert(ins, nd);
                        ds.node_machine[nd] = m; ds.node_pt[nd] = pt;
                    }
                }

                let Some((repaired_mk, _)) = eval_disj(&ds, &mut buf) else { alns_no_improve += 1; temperature *= cooling; continue };
                let repaired_sol = match disj_to_solution(pre, &ds, &buf.start) { Ok(s) => s, Err(_) => { alns_no_improve += 1; temperature *= cooling; continue } };

                let after_gr = match greedy_reassign_pass(pre, challenge, &repaired_sol) {
                    Ok(Some((s, mk))) => (s, mk),
                    _ => (repaired_sol, repaired_mk),
                };
                let (alns_cand_sol, alns_cand_mk) = if let Ok(Some((ls_sol, ls_mk))) = critical_block_move_local_search_ex(pre, challenge, &after_gr.0, cb_passes, cb_iters, cb_no_improve) {
                    (ls_sol, ls_mk)
                } else { (after_gr.0, after_gr.1) };

                if alns_cand_mk < best_makespan {
                    best_makespan = alns_cand_mk;
                    best_solution = Some(alns_cand_sol.clone());
                    save_solution(&alns_cand_sol)?;
                }
                if alns_cand_mk < alns_best_mk {
                    alns_best_mk = alns_cand_mk;
                    alns_no_improve = 0;
                } else { alns_no_improve += 1; }

                let delta = alns_cand_mk as f64 - alns_sa_mk as f64;
                alns_seed ^= alns_seed.wrapping_shl(13); alns_seed ^= alns_seed.wrapping_shr(7); alns_seed ^= alns_seed.wrapping_shl(17);
                let rand_val = (alns_seed as f64) / (u64::MAX as f64);
                if delta < 0.0 || (temperature > 0.0 && rand_val < (-delta / temperature).exp()) {
                    alns_sa_mk = alns_cand_mk;
                    alns_sa_sol = Some(alns_cand_sol);
                }
                temperature *= cooling;
            }
        }

        if top_solutions.len() >= 5 {
            let vote_result2 = crossover_majority_vote(pre, challenge, &top_solutions, cb_passes + 1, cb_iters, cb_no_improve)?;
            if let Some((vote_sol, vote_mk)) = vote_result2 {
                if vote_mk < best_makespan {
                    best_makespan = vote_mk;
                    best_solution = Some(vote_sol.clone());
                    save_solution(&vote_sol)?;
                }
            }
        }

        if let Some(sol) = best_solution.as_ref() {
            if let Ok(Some((bmr_sol, bmr_mk))) = bottleneck_machine_relief_pass(pre, challenge, sol, 20) {
                if bmr_mk < best_makespan {
                    best_makespan = bmr_mk;
                    best_solution = Some(bmr_sol.clone());
                    save_solution(&bmr_sol)?;
                }
            }
        }

        if let Some(sol) = best_solution.as_ref() {
            if let Some((improved_sol, improved_mk)) = greedy_reassign_pass(pre, challenge, sol)? {
                if improved_mk < best_makespan { best_makespan = improved_mk; save_solution(&improved_sol)?; best_solution = Some(improved_sol); }
            }
        }

        if let Some(sol) = best_solution.as_ref() {
            if let Ok(Some((ecr_sol, ecr_mk))) = exhaustive_critical_reroute_pass(pre, challenge, sol) {
                if ecr_mk < best_makespan { best_makespan = ecr_mk; best_solution = Some(ecr_sol.clone()); save_solution(&ecr_sol)?; }
            }
        }

        if let Some(sol) = best_solution.as_ref() {
            if let Ok(Some((cb_sol, cb_mk))) = critical_block_move_local_search_ex(pre, challenge, sol, cb_passes + 2, cb_iters, cb_no_improve) {
                if cb_mk < best_makespan { best_makespan = cb_mk; best_solution = Some(cb_sol.clone()); save_solution(&cb_sol)?; }
            }
        }

        if let Some(ref sol) = best_solution.clone() {
            if let Some((improved_sol, improved_mk)) = greedy_reassign_pass(pre, challenge, sol)? {
                if improved_mk < best_makespan { best_makespan = improved_mk; save_solution(&improved_sol)?; best_solution = Some(improved_sol); }
            }
        }

        if let Some(ref sol) = best_solution.clone() {
            if let Ok(Some((ecr_sol, ecr_mk))) = exhaustive_critical_reroute_pass(pre, challenge, sol) {
                if ecr_mk < best_makespan { best_makespan = ecr_mk; best_solution = Some(ecr_sol.clone()); save_solution(&ecr_sol)?; }
            }
        }

        if let Some(sol) = best_solution.as_ref() {
            if let Ok(Some((bmr_sol, bmr_mk))) = bottleneck_machine_relief_pass(pre, challenge, sol, 10) {
                if bmr_mk < best_makespan {
                    best_makespan = bmr_mk;
                    best_solution = Some(bmr_sol.clone());
                    save_solution(&bmr_sol)?;
                }
            }
        }

        {
            let final_ils_rounds = if effort.fjsp_medium_iters > 300 { 12 } else { 8 };
            let mut final_ils_best_mk = best_makespan;
            let mut final_ils_best_sol = best_solution.clone();
            let mut final_no_improve = 0usize;
            let final_max_no_improve = final_ils_rounds / 2 + 2;
            let mut fpseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0xDEADC0DEu64)
                .wrapping_add(best_makespan as u64)
                .wrapping_add(0xFEEDFACEu64);

            for fir in 0..final_ils_rounds {
                if final_no_improve >= final_max_no_improve { break; }
                let Some(base) = final_ils_best_sol.as_ref() else { break };
                let mut ds = match build_disj_from_solution(pre, challenge, base) { Ok(d) => d, Err(_) => { final_no_improve += 1; continue; } };
                let mut buf = EvalBuf::new(ds.n);
                let Some((_, mk_node)) = eval_disj(&ds, &mut buf) else { final_no_improve += 1; continue };
                let n = ds.n;

                let k_perturb = 5 + fir / 2;
                let mut machine_loads = vec![0u32; ds.num_machines];
                for nd in 0..n { let m = ds.node_machine[nd]; machine_loads[m] = machine_loads[m].saturating_add(ds.node_pt[nd]); }
                let worst_m = machine_loads.iter().enumerate().max_by_key(|&(_, &v)| v).map(|(i, _)| i).unwrap_or(0);

                let mut crit_nodes: Vec<usize> = Vec::with_capacity(64);
                let mut u = mk_node;
                while u != NONE_USIZE { crit_nodes.push(u); u = buf.best_pred[u]; }
                let bn_nodes: Vec<usize> = ds.machine_seq[worst_m].clone();

                let mut combined: Vec<usize> = crit_nodes.clone();
                for &nd in &bn_nodes {
                    if !combined.contains(&nd) { combined.push(nd); }
                }

                let mut perturbed = 0;
                for _ in 0..(k_perturb * 4) {
                    if perturbed >= k_perturb || combined.is_empty() { break; }
                    fpseed ^= fpseed.wrapping_shl(13); fpseed ^= fpseed.wrapping_shr(7); fpseed ^= fpseed.wrapping_shl(17);
                    let idx = (fpseed as usize) % combined.len();
                    let node = combined[idx];
                    let job = ds.node_job[node]; let op_idx = ds.node_op[node]; let product = pre.job_products[job];
                    let op_info = &pre.product_ops[product][op_idx];
                    if op_info.machines.len() <= 1 { continue; }
                    let cur_machine = ds.node_machine[node];
                    fpseed ^= fpseed.wrapping_shl(13); fpseed ^= fpseed.wrapping_shr(7); fpseed ^= fpseed.wrapping_shl(17);
                    let alt_idx = (fpseed as usize) % op_info.machines.len();
                    let (new_m, new_pt) = op_info.machines[alt_idx];
                    if new_m == cur_machine { continue; }
                    let old_pos = match ds.machine_seq[cur_machine].iter().position(|&x| x == node) { Some(p) => p, None => continue };
                    ds.machine_seq[cur_machine].remove(old_pos);
                    ds.node_machine[node] = new_m; ds.node_pt[node] = new_pt;
                    let cur_start = buf.start[node];
                    let mut ins_pos = ds.machine_seq[new_m].len();
                    for (ki, &nd) in ds.machine_seq[new_m].iter().enumerate() { if buf.start[nd] >= cur_start { ins_pos = ki; break; } }
                    ds.machine_seq[new_m].insert(ins_pos, node);
                    perturbed += 1;
                }

                let Some((_, _)) = eval_disj(&ds, &mut buf) else { final_no_improve += 1; continue };
                let perturbed_sol = match disj_to_solution(pre, &ds, &buf.start) { Ok(s) => s, Err(_) => { final_no_improve += 1; continue; } };

                let after_gr = match greedy_reassign_pass(pre, challenge, &perturbed_sol) {
                    Ok(Some((s, mk))) => (s, mk),
                    _ => { if let Some((pmk, _)) = eval_disj(&ds, &mut buf) { (perturbed_sol, pmk) } else { final_no_improve += 1; continue; } }
                };

                let (cand_sol, cand_mk) = if let Ok(Some((ls_sol, ls_mk))) = critical_block_move_local_search_ex(pre, challenge, &after_gr.0, cb_passes + 1, cb_iters, cb_no_improve) {
                    (ls_sol, ls_mk)
                } else { (after_gr.0, after_gr.1) };

                if cand_mk < best_makespan {
                    best_makespan = cand_mk; best_solution = Some(cand_sol.clone()); save_solution(&cand_sol)?;
                }
                if cand_mk < final_ils_best_mk {
                    final_ils_best_mk = cand_mk; final_ils_best_sol = Some(cand_sol); final_no_improve = 0;
                } else { final_no_improve += 1; }
            }
        }

        if top_solutions.len() >= 4 {
            let vote_result3 = crossover_majority_vote(pre, challenge, &top_solutions, cb_passes + 2, cb_iters, cb_no_improve)?;
            if let Some((vote_sol, vote_mk)) = vote_result3 {
                if vote_mk < best_makespan {
                    best_makespan = vote_mk;
                    best_solution = Some(vote_sol.clone());
                    save_solution(&vote_sol)?;
                }
            }
        }

        if let Some(sol) = best_solution.as_ref() {
            if let Some((improved_sol, improved_mk)) = greedy_reassign_pass(pre, challenge, sol)? {
                if improved_mk < best_makespan { best_makespan = improved_mk; save_solution(&improved_sol)?; best_solution = Some(improved_sol); }
            }
        }
        if let Some(sol) = best_solution.as_ref() {
            if let Ok(Some((ecr_sol, ecr_mk))) = exhaustive_critical_reroute_pass(pre, challenge, sol) {
                if ecr_mk < best_makespan { best_makespan = ecr_mk; best_solution = Some(ecr_sol.clone()); save_solution(&ecr_sol)?; }
            }
        }
        if let Some(sol) = best_solution.as_ref() {
            if let Ok(Some((bmr_sol, bmr_mk))) = bottleneck_machine_relief_pass(pre, challenge, sol, 15) {
                if bmr_mk < best_makespan { best_solution = Some(bmr_sol.clone()); save_solution(&bmr_sol)?; }
            }
        }

        if let Some(sol) = best_solution { save_solution(&sol)?; }
        Ok(())
    }

    fn crossover_majority_vote(
        pre: &Pre,
        challenge: &Challenge,
        top_solutions: &[(Solution, u32)],
        cb_passes: usize,
        cb_iters: usize,
        cb_no_improve: usize,
    ) -> Result<Option<(Solution, u32)>> {
        let num_jobs = challenge.num_jobs;
        let num_machines = challenge.num_machines;
        let pool_size = top_solutions.len().min(10);
        if pool_size < 2 { return Ok(None); }

        let mut job_machine_choices: Vec<Vec<usize>> = Vec::with_capacity(num_jobs);
        for job in 0..num_jobs {
            let num_ops = pre.job_ops_len[job];
            let mut vote_counts: Vec<HashMap<usize, usize>> = vec![HashMap::new(); num_ops];
            for (sol, _mk) in top_solutions.iter().take(pool_size) {
                if sol.job_schedule.len() <= job { continue; }
                let job_sched = &sol.job_schedule[job];
                for op_idx in 0..num_ops.min(job_sched.len()) {
                    let (machine, _) = job_sched[op_idx];
                    *vote_counts[op_idx].entry(machine).or_insert(0) += 1;
                }
            }
            let product = pre.job_products[job];
            let mut choices: Vec<usize> = Vec::with_capacity(num_ops);
            for op_idx in 0..num_ops {
                let op_info = &pre.product_ops[product][op_idx];
                let mut best_machine = op_info.machines.first().map(|&(m, _)| m).unwrap_or(0);
                let mut best_votes = 0usize;
                for (&m, &cnt) in &vote_counts[op_idx] {
                    if !op_info.machines.iter().any(|&(em, _)| em == m) { continue; }
                    if cnt > best_votes {
                        best_machine = m;
                        best_votes = cnt;
                    }
                }
                if best_votes == 0 {
                    best_machine = op_info.machines.first().map(|&(m, _)| m).unwrap_or(0);
                }
                choices.push(best_machine);
            }
            job_machine_choices.push(choices);
        }

        let mut job_next_op = vec![0usize; num_jobs];
        let mut job_ready_time = vec![0u32; num_jobs];
        let mut machine_avail = vec![0u32; num_machines];
        let mut job_schedule: Vec<Vec<(usize, u32)>> = vec![Vec::new(); num_jobs];
        let total_ops = pre.total_ops;
        let mut scheduled = 0usize;
        let mut time = 0u32;
        let mut stall_guard = 0usize;

        while scheduled < total_ops && stall_guard < total_ops * 6 {
            stall_guard += 1;
            let mut any = false;
            for job in 0..num_jobs {
                let op_idx = job_next_op[job];
                if op_idx >= job_machine_choices[job].len() { continue; }
                if job_ready_time[job] > time { continue; }
                let machine = job_machine_choices[job][op_idx];
                if machine_avail[machine] > time { continue; }
                let product = pre.job_products[job];
                let op_info = &pre.product_ops[product][op_idx];
                let pt = op_info.machines.iter().find(|&&(m, _)| m == machine).map(|&(_, pt)| pt).unwrap_or(0);
                let end = time.saturating_add(pt);
                job_schedule[job].push((machine, time));
                job_next_op[job] += 1;
                job_ready_time[job] = end;
                machine_avail[machine] = end;
                scheduled += 1;
                any = true;
            }
            if !any {
                let mut next_t = u32::MAX;
                for &t in &machine_avail { if t > time { next_t = next_t.min(t); } }
                for j in 0..num_jobs { if job_ready_time[j] > time { next_t = next_t.min(job_ready_time[j]); } }
                if next_t == u32::MAX { break; }
                time = next_t;
            }
        }

        if scheduled < total_ops { return Ok(None); }
        let vote_sol = Solution { job_schedule };

        let ds = match build_disj_from_solution(pre, challenge, &vote_sol) { Ok(d) => d, Err(_) => return Ok(None) };
        let mut buf = EvalBuf::new(ds.n);
        let Some((base_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };

        let (gr_sol, gr_mk) = match greedy_reassign_pass(pre, challenge, &vote_sol) {
            Ok(Some((s, mk))) => (s, mk),
            _ => (vote_sol, base_mk),
        };

        let (final_sol, final_mk) = if let Ok(Some((cb_sol, cb_mk))) = critical_block_move_local_search_ex(pre, challenge, &gr_sol, cb_passes, cb_iters, cb_no_improve) {
            (cb_sol, cb_mk)
        } else {
            (gr_sol, gr_mk)
        };

        let (result_sol, result_mk) = if let Ok(Some((ecr_sol, ecr_mk))) = exhaustive_critical_reroute_pass(pre, challenge, &final_sol) {
            if ecr_mk < final_mk { (ecr_sol, ecr_mk) } else { (final_sol, final_mk) }
        } else {
            (final_sol, final_mk)
        };

        Ok(Some((result_sol, result_mk)))
    }

    fn bottleneck_machine_relief_pass(
        pre: &Pre,
        challenge: &Challenge,
        base_sol: &Solution,
        max_iters: usize,
    ) -> Result<Option<(Solution, u32)>> {
        let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
        let mut buf = EvalBuf::new(ds.n);
        let Some((mut current_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
        let initial_mk = current_mk;
        let n = ds.n;
        let num_machines = ds.num_machines;
        let mut any_improvement = false;
        let mut machine_loads = vec![0u64; num_machines];
        for nd in 0..n {
            let m = ds.node_machine[nd];
            machine_loads[m] = machine_loads[m].saturating_add(ds.node_pt[nd] as u64);
        }

        for iter in 0..max_iters {
            let target_machine = if iter % 2 == 0 {
                machine_loads.iter().enumerate()
                    .max_by_key(|&(_, &v)| v)
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            } else {
                let Some((_, mk_node)) = eval_disj(&ds, &mut buf) else { break };
                let mut crit_machine_count = vec![0usize; num_machines];
                let mut u = mk_node;
                while u != NONE_USIZE {
                    crit_machine_count[ds.node_machine[u]] += 1;
                    u = buf.best_pred[u];
                }
                crit_machine_count.iter().enumerate()
                    .max_by_key(|&(_, &v)| v)
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            };

            let seq_len = ds.machine_seq[target_machine].len();
            if seq_len <= 1 { break; }

            let has_alternatives = ds.machine_seq[target_machine].iter().any(|&nd| {
                let job = ds.node_job[nd];
                let op_idx = ds.node_op[nd];
                let product = pre.job_products[job];
                pre.product_ops[product][op_idx].machines.len() > 1
            });
            if !has_alternatives { break; }

            let mut best_iter_mk = current_mk;
            let mut best_node = NONE_USIZE;
            let mut best_new_m = NONE_USIZE;
            let mut best_new_pt = 0u32;
            let mut best_ins_pos = 0usize;

            let target_nodes: Vec<usize> = ds.machine_seq[target_machine].clone();

            for &node in &target_nodes {
                let job = ds.node_job[node];
                let op_idx = ds.node_op[node];
                let product = pre.job_products[job];
                let op_info = &pre.product_ops[product][op_idx];
                if op_info.machines.len() <= 1 { continue; }

                let cur_machine = ds.node_machine[node];
                let cur_pt = ds.node_pt[node];

                let old_pos = match ds.machine_seq[cur_machine].iter().position(|&x| x == node) {
                    Some(p) => p,
                    None => continue,
                };
                ds.machine_seq[cur_machine].remove(old_pos);

                for &(new_m, new_pt) in &op_info.machines {
                    if new_m == cur_machine { continue; }

                    ds.node_machine[node] = new_m;
                    ds.node_pt[node] = new_pt;

                    let tgt_len = ds.machine_seq[new_m].len();
                    let jp_end = if node > ds.job_offsets[job] {
                        let jp = node - 1;
                        buf.start[jp].saturating_add(ds.node_pt[jp])
                    } else { 0u32 };

                    let mut pos_estimates: Vec<(usize, u32)> = Vec::with_capacity(tgt_len + 1);
                    for pos in 0..=tgt_len {
                        let mp_end = if pos > 0 {
                            let pred = ds.machine_seq[new_m][pos - 1];
                            buf.start[pred].saturating_add(ds.node_pt[pred])
                        } else { 0u32 };
                        let start_est = jp_end.max(mp_end);
                        pos_estimates.push((pos, start_est));
                    }
                    pos_estimates.sort_by_key(|&(_, s)| s);

                    for &(pos, _) in &pos_estimates {
                        ds.machine_seq[new_m].insert(pos, node);
                        if let Some((test_mk, _)) = eval_disj(&ds, &mut buf) {
                            if test_mk < best_iter_mk {
                                best_iter_mk = test_mk;
                                best_node = node;
                                best_new_m = new_m;
                                best_new_pt = new_pt;
                                best_ins_pos = pos;
                            }
                        }
                        ds.machine_seq[new_m].remove(pos);
                    }
                }

                ds.machine_seq[cur_machine].insert(old_pos, node);
                ds.node_machine[node] = cur_machine;
                ds.node_pt[node] = cur_pt;
            }

            if best_node != NONE_USIZE && best_iter_mk < current_mk {
                let cur_machine = ds.node_machine[best_node];
                let cur_pt = ds.node_pt[best_node];
                let old_pos = ds.machine_seq[cur_machine].iter().position(|&x| x == best_node).unwrap();
                ds.machine_seq[cur_machine].remove(old_pos);
                let ins = best_ins_pos.min(ds.machine_seq[best_new_m].len());
                ds.machine_seq[best_new_m].insert(ins, best_node);
                ds.node_machine[best_node] = best_new_m;
                ds.node_pt[best_node] = best_new_pt;
                machine_loads[cur_machine] = machine_loads[cur_machine].saturating_sub(cur_pt as u64);
                machine_loads[best_new_m] = machine_loads[best_new_m].saturating_add(best_new_pt as u64);
                current_mk = best_iter_mk;
                any_improvement = true;
                let _ = eval_disj(&ds, &mut buf);
            } else {
                break;
            }
        }

        if !any_improvement || current_mk >= initial_mk { return Ok(None); }
        let Some((_, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
        let sol = disj_to_solution(pre, &ds, &buf.start)?;
        Ok(Some((sol, current_mk)))
    }
}

pub mod fjsp_high {
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
    fn slack_urgency_fh(pre: &Pre, target_mk: Option<u32>, time: u32, product: usize, op_idx: usize) -> f64 {
        let Some(tgt) = target_mk else { return 0.0 };
        let lb = (time as u64).saturating_add(pre.product_suf_min[product][op_idx] as u64);
        let slack = (tgt as i64) - (lb as i64);
        let scale = (0.70 * pre.avg_op_min).max(1.0);
        let pos = (slack.max(0) as f64) / scale; let neg = ((-slack).max(0) as f64) / scale;
        (1.0 / (1.0 + pos)).clamp(0.0, 1.0) + (0.35 * neg).min(3.0)
    }

    #[inline]
    fn route_pref_bonus_fh(rp: Option<&RoutePrefLite>, product: usize, op_idx: usize, machine: usize) -> f64 {
        let Some(rp) = rp else { return 0.0 };
        if product >= rp.len() || op_idx >= rp[product].len() { return 0.0; }
        let r = rp[product][op_idx]; let mu = machine.min(255) as u8;
        if mu == r.best_m { (r.best_w as f64) / 255.0 } else if mu == r.second_m { (r.second_w as f64) / 255.0 } else { 0.0 }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn score_candidate(
        pre: &Pre, rule: Rule, job: usize, product: usize, op_idx: usize,
        ops_rem: usize, op: &OpInfo, machine: usize, pt: u32, time: u32,
        target_mk: Option<u32>, best_end: u32, second_end: u32, best_cnt_total: usize,
        progress: f64, job_bias: f64, machine_penalty: f64, dynamic_load: f64,
        route_pref: Option<&RoutePrefLite>, route_w: f64, jitter: f64,
    ) -> f64 {
        let rem_min = pre.product_suf_min[product][op_idx] as f64;
        let rem_avg = pre.product_suf_avg[product][op_idx]; let rem_bn = pre.product_suf_bn[product][op_idx];
        let flex_f = (op.flex as f64).max(1.0); let flex_inv = 1.0/flex_f;
        let rem_min_n = rem_min/pre.horizon.max(1.0); let rem_avg_n = rem_avg/pre.max_job_avg_work.max(1e-9);
        let bn_n = rem_bn/pre.max_job_bn.max(1e-9); let ops_n = (ops_rem as f64)/(pre.max_ops as f64).max(1.0);
        let load_n = dynamic_load/pre.avg_machine_load.max(1e-9); let scar_n = pre.machine_scarcity[machine]/pre.avg_machine_scarcity.max(1e-9);
        let end_n = (best_end as f64)/pre.time_scale.max(1.0); let proc_n = (pt as f64)/pre.avg_op_min.max(1.0);
        let regret = if second_end >= INF { pre.avg_op_min*2.6 } else { (second_end-best_end) as f64 };
        let reg_n = (regret/pre.avg_op_min.max(1.0)).clamp(0.0,6.0);
        let scarcity_urg = 1.0/(best_cnt_total as f64).max(1.0);
        let density_n = ((rem_min/(ops_rem as f64).max(1.0))/pre.avg_op_min.max(1.0)).clamp(0.0,4.0);
        let next_min = pre.product_next_min[product][op_idx] as f64; let next_min_n = next_min/pre.horizon.max(1.0);
        let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
        let p2 = progress*progress; let next_w_base = 0.12+p2*0.28;
        let next_term_raw = (0.55*next_min_n+0.45*next_flex_inv)*(1.0+0.30*density_n*pre.high_flex);
        let js = pre.jobshopness; let fl = 1.0-js;
        let avg_flex_inv = 1.0/pre.flex_avg.max(1.0); let scarce_match = scar_n*(flex_inv-avg_flex_inv);
        let mpen = machine_penalty.clamp(0.0,1.0); let mpen_gain = 1.0+0.85*pre.high_flex;
        let flow_term = pre.flow_w*pre.job_flow_pref[job]*(0.65+0.70*(1.0-progress));
        let slack_u = slack_urgency_fh(pre, target_mk, time, product, op_idx);
        let slack_w = pre.slack_base*(0.25+0.75*progress); let slack_reg_boost = 1.0+0.40*reg_n*progress;
        let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop=pre.machine_best_pop[machine]; (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
        let route_gain = (0.70+0.80*(1.0-progress)).clamp(0.70,1.40);
        let route_term = if route_w>0.0 && op.flex>=2 { route_w*route_gain*route_pref_bonus_fh(route_pref,product,op_idx,machine) } else { 0.0 };
        match rule {
            Rule::CriticalPath => { let next_term=next_w_base*0.30*next_term_raw; let slack_term=slack_w*slack_u*slack_reg_boost; (1.03*rem_min_n)+(0.10*ops_n)+(0.24*scarcity_urg)+(0.20*pre.flex_factor)*flex_inv+next_term+0.10*slack_term-(0.70*end_n)-pop_pen+(0.45*job_bias)+flow_term+route_term+jitter }
            Rule::MostWork => { let next_term=next_w_base*0.25*next_term_raw; (1.00*rem_avg_n)+(0.12*ops_n)+(0.18*scarcity_urg)+(0.15*pre.flex_factor)*flex_inv+next_term-(0.62*end_n)-pop_pen+(0.45*job_bias)+flow_term+route_term+jitter }
            Rule::LeastFlex => { let next_term=next_w_base*0.20*next_term_raw; (1.00*flex_inv)+(0.28*rem_min_n)+(0.22*scarcity_urg)+next_term-(0.55*end_n)-pop_pen+(0.35*job_bias)+flow_term+route_term+jitter }
            Rule::ShortestProc => { let next_term=next_w_base*0.20*next_term_raw; (-1.00*proc_n)+(0.25*rem_min_n)+(0.12*scarcity_urg)+next_term-(0.20*end_n)-pop_pen+(0.25*job_bias)+flow_term+route_term+jitter }
            Rule::Regret => { let next_term=next_w_base*0.25*next_term_raw; (1.05*reg_n)+(0.55*rem_min_n)+(0.22*scarcity_urg)+next_term-(0.68*end_n)-pop_pen+(0.35*job_bias)+flow_term+route_term+jitter }
            Rule::EndTight => { let end_w=1.10+1.00*progress+0.35*pre.high_flex; let cp_w=1.15+0.30*js; let reg_w=(0.55+0.20*(1.0-progress))*(0.85+0.60*js); let mpen_w=(0.10+0.45*pre.high_flex)*pre.flex_factor; let next_term=next_w_base*(0.45+0.55*js)*next_term_raw; let slack_term=slack_w*(0.70+0.40*js)*slack_u*slack_reg_boost; (cp_w*rem_min_n)+0.12*rem_avg_n+0.08*ops_n+0.18*scarcity_urg+(0.30*pre.flex_factor)*flex_inv+(0.20*pre.flex_factor)*scarce_match+(reg_w*pre.flex_factor)*reg_n+next_term+slack_term-end_w*end_n-0.22*proc_n-pop_pen-(mpen_gain*mpen_w)*mpen+0.55*job_bias+flow_term+route_term+jitter }
            Rule::BnHeavy => { let bn_w=(0.90+0.55*js)*pre.bn_focus; let end_w=0.65+0.70*progress; let reg_w=(0.60+0.25*(1.0-progress))*(0.85+0.35*js); let load_w=if pre.hi_flex{-0.35}else{0.55+0.25*js}; let mpen_w=(0.12+0.30*js)*pre.flex_factor*(0.95+0.65*pre.high_flex); let next_term=next_w_base*(0.55+0.75*js)*next_term_raw; let slack_term=slack_w*(0.45+0.55*js)*slack_u*slack_reg_boost; (0.95*rem_min_n)+(0.30*rem_avg_n)+(bn_w*bn_n)+(0.22*density_n)+(0.10*ops_n)+(0.65*pre.flex_factor)*flex_inv+(0.35*pre.flex_factor)*scarce_match+load_w*pre.flex_factor*load_n+(reg_w*pre.flex_factor)*reg_n+0.18*scarcity_urg+next_term+slack_term-end_w*end_n-0.18*proc_n-pop_pen-(mpen_gain*mpen_w)*mpen+0.60*job_bias+flow_term+route_term+jitter }
            Rule::Adaptive => { let end_w=(0.90*fl+0.72*js)+(0.62+0.12*fl)*progress+0.18*pre.high_flex; let reg_w=(0.50*fl+0.78*js)+0.18*(1.0-progress); let bn_w=((0.45+0.40*js)+0.25*(1.0-progress))*pre.bn_focus; let load_sign=if pre.hi_flex{-1.0}else{1.0}; let load_w=load_sign*(0.45*fl+0.75*js)*pre.flex_factor; let density_w=0.08*fl+0.20*js; let next_term=next_w_base*(0.50*fl+1.50*js)*next_term_raw; let mpen_w=(0.08*fl+0.28*js)*pre.flex_factor*(1.0+0.85*pre.high_flex); let slack_term=slack_w*(0.55*fl+0.85*js)*slack_u*slack_reg_boost; (1.05*rem_min_n)+(0.48*rem_avg_n)+(bn_w*bn_n)+density_w*density_n+(0.08*ops_n)+(0.62*pre.flex_factor)*flex_inv+(0.55*pre.flex_factor)*scarce_match+load_w*load_n+(reg_w*pre.flex_factor)*reg_n+0.20*pre.flex_factor*scarcity_urg+next_term+slack_term-end_w*end_n-(0.18*fl+0.12*js)*proc_n-pop_pen-(mpen_gain*mpen_w)*mpen+(0.62+0.06*js)*job_bias+flow_term+route_term+jitter }
            Rule::FlexBalance => { let end_w=(0.85+0.70*progress+0.15*js).clamp(0.85,1.75); let cp_w=(1.00+0.30*js+0.15*(1.0-progress)).clamp(0.95,1.45); let load_w=(0.55+0.35*pre.high_flex).clamp(0.55,0.95)*pre.flex_factor; let mpen_w=(0.55+0.65*pre.high_flex).clamp(0.55,1.15); let reg_w=(0.35+0.25*(1.0-progress)).clamp(0.35,0.70); let next_term=next_w_base*0.40*next_term_raw; (cp_w*rem_min_n)+0.55*rem_avg_n+0.08*ops_n+0.06*density_n+0.08*scarcity_urg+next_term+(0.70*slack_w)*slack_u-end_w*end_n-0.16*proc_n-pop_pen-load_w*load_n-(mpen_w*(1.0+0.85*pre.high_flex))*mpen+(reg_w*pre.flex_factor)*reg_n+(0.58+0.10*pre.high_flex)*job_bias+flow_term+route_term+jitter }
        }
    }

    #[inline]
    fn rule_idx(r: Rule) -> usize {
        match r { Rule::Adaptive=>0, Rule::BnHeavy=>1, Rule::EndTight=>2, Rule::CriticalPath=>3, Rule::MostWork=>4, Rule::LeastFlex=>5, Rule::Regret=>6, Rule::ShortestProc=>7, Rule::FlexBalance=>8 }
    }

    fn choose_rule_bandit(rng: &mut SmallRng, rules: &[Rule], rule_best: &[u32], rule_tries: &[u32], global_best: u32, margin: u32, stuck: usize, chaos_like: bool, late_phase: bool) -> Rule {
        if rules.is_empty() { return Rule::Adaptive; }
        let mut best_seen = global_best; for &mk in rule_best { if mk < best_seen { best_seen = mk; } }
        let scale = (margin as f64).max(1.0); let s = ((stuck as f64)/140.0).clamp(0.0,1.0); let explore_mix = (0.10+0.55*s).clamp(0.10,0.65);
        let mut w = vec![0.0f64; rules.len()];
        for (i, &r) in rules.iter().enumerate() {
            let mk=rule_best[rule_idx(r)]; let t=rule_tries[rule_idx(r)].max(1) as f64;
            let delta=mk.saturating_sub(best_seen) as f64; let exploit=(-delta/scale).exp(); let explore=(1.0/t).sqrt();
            let mut ww=(1.0-explore_mix)*exploit+explore_mix*explore; ww=ww.max(1e-6);
            if chaos_like{ww=ww.powf(0.70);}else if late_phase{ww=ww.powf(1.18);}
            w[i]=ww;
        }
        let mut sum=0.0; for &ww in &w { sum+=ww.max(0.0); }
        if !(sum>0.0) { return rules[rng.gen_range(0..rules.len())]; }
        let mut r=rng.gen::<f64>()*sum;
        for (i,&ww) in w.iter().enumerate() { r-=ww.max(0.0); if r<=0.0 { return rules[i]; } }
        rules[rules.len()-1]
    }

    fn construct_solution_conflict(
        challenge: &Challenge, pre: &Pre, rule: Rule, k: usize, target_mk: Option<u32>,
        rng: &mut SmallRng, job_bias: Option<&[f64]>, machine_penalty: Option<&[f64]>,
        route_pref: Option<&RoutePrefLite>, route_w: f64,
    ) -> Result<(Solution, u32)> {
        let num_jobs=challenge.num_jobs; let num_machines=challenge.num_machines;
        let mut job_next_op=vec![0usize;num_jobs]; let mut job_ready_time=vec![0u32;num_jobs];
        let mut machine_avail=vec![0u32;num_machines]; let mut machine_load=pre.machine_load0.clone();
        let mut job_schedule: Vec<Vec<(usize,u32)>>=pre.job_ops_len.iter().map(|&len|Vec::with_capacity(len)).collect();
        let mut remaining_ops=pre.total_ops; let mut time=0u32;
        let mut demand: Vec<u16>=vec![0u16;num_machines];
        let mut raw_by_machine: Vec<Vec<RawCand>>=(0..num_machines).map(|_|Vec::with_capacity(12)).collect();
        let mut idle_machines: Vec<usize>=Vec::with_capacity(num_machines);
        let chaotic_like=pre.chaotic_like;
        let mut machine_work: Vec<u64>=if chaotic_like{vec![0u64;num_machines]}else{vec![]};
        let mut sum_work: u64=0;
        while remaining_ops > 0 {
            loop {
                idle_machines.clear();
                for m in 0..num_machines { if machine_avail[m]<=time { idle_machines.push(m); } }
                if idle_machines.is_empty() { break; }
                for &m in &idle_machines { demand[m]=0; raw_by_machine[m].clear(); }
                let progress=1.0-(remaining_ops as f64)/(pre.total_ops as f64).max(1.0);
                let cap_per_machine=if k==0{12usize}else{(k+6).min(12)};
                for job in 0..num_jobs {
                    let op_idx=job_next_op[job]; if op_idx>=pre.job_ops_len[job]||job_ready_time[job]>time{continue;}
                    let product=pre.job_products[job]; let op=&pre.product_ops[product][op_idx];
                    if op.flex==0||op.machines.is_empty()||op.min_pt>=INF{continue;}
                    let (best_end,second_end,best_cnt_total,best_cnt_idle)=best_second_and_counts(time,&machine_avail,op);
                    if best_end>=INF||best_cnt_idle==0{continue;}
                    let ops_rem=pre.job_ops_len[job]-op_idx; let jb=job_bias.map(|v|v[job]).unwrap_or(0.0);
                    let flex_inv=1.0/(op.flex as f64).max(1.0); let scarcity_urg=1.0/(best_cnt_total as f64).max(1.0);
                    let regret=if second_end>=INF{pre.avg_op_min*2.6}else{(second_end-best_end) as f64};
                    let regn=(regret/pre.avg_op_min.max(1.0)).clamp(0.0,6.0); let rigidity=(0.60*flex_inv+0.40*scarcity_urg).clamp(0.0,2.5);
                    for &(m,pt) in &op.machines {
                        if machine_avail[m]>time{continue;}
                        let end=time.saturating_add(pt); if end!=best_end{continue;}
                        demand[m]=demand[m].saturating_add(1);
                        let mp=machine_penalty.map(|v|v[m]).unwrap_or(0.0); let jitter=if k>0{rng.gen::<f64>()*1e-9}else{0.0};
                        let base=score_candidate(pre,rule,job,product,op_idx,ops_rem,op,m,pt,time,target_mk,best_end,second_end,best_cnt_total,progress,jb,mp,machine_load[m],route_pref,route_w,jitter);
                        push_top_k_raw(&mut raw_by_machine[m],RawCand{job,machine:m,pt,base_score:base,rigidity,reg_n:regn},cap_per_machine);
                    }
                }
                let denom=(idle_machines.len() as f64).max(1.0);
                let (conflict_w,conflict_scale)=if chaotic_like{(-(0.05+0.08*(1.0-progress)).clamp(0.04,0.14),(0.95+0.20*pre.flex_factor).clamp(0.90,1.20))}else{((0.09+0.26*pre.jobshopness+0.11*pre.high_flex+0.16*(1.0-progress)).clamp(0.05,0.45),(0.90+0.40*pre.flex_factor).clamp(0.85,1.75))};
                let (bal_w,avg_work)=if chaotic_like{((0.030+0.070*(1.0-progress)).clamp(0.025,0.11),(sum_work as f64)/(num_machines as f64).max(1.0))}else{(0.0,0.0)};
                let mut best: Option<Cand>=None; let mut top: Vec<Cand>=if k>0{Vec::with_capacity(k)}else{Vec::new()};
                for &m in &idle_machines {
                    let dem=demand[m] as f64; if dem<=0.0||raw_by_machine[m].is_empty(){continue;}
                    let dem_n=((dem-1.0)/denom).clamp(0.0,2.5);
                    let bal_pen=if chaotic_like&&bal_w>0.0{let denomw=(avg_work+(pre.avg_op_min*3.0).max(1.0)).max(1.0); let r=(machine_work[m] as f64)/denomw; let done_n=(r/(r+1.0)).clamp(0.0,1.0); -bal_w*done_n}else{0.0};
                    for rc in &raw_by_machine[m] {
                        let rig=rc.rigidity.clamp(0.0,2.5); let regc=rc.reg_n.clamp(0.0,4.5);
                        let mut boost=conflict_w*conflict_scale*dem_n*(1.15*rig+0.85*regc);
                        if chaotic_like{boost=boost.max(-0.26);}
                        let c=Cand{job:rc.job,machine:rc.machine,pt:rc.pt,score:rc.base_score+boost+bal_pen};
                        if k==0{if best.map_or(true,|bb|c.score>bb.score){best=Some(c);}}else{push_top_k(&mut top,c,k);}
                    }
                }
                let chosen=if k==0{match best{Some(c)=>c,None=>break}}else{if top.is_empty(){break;}choose_from_top_weighted(rng,&top)};
                let job=chosen.job; let machine=chosen.machine; let pt=chosen.pt;
                let product=pre.job_products[job]; let op_idx=job_next_op[job]; let op=&pre.product_ops[product][op_idx];
                let (best_end_now,_,_,_)=best_second_and_counts(time,&machine_avail,op);
                let end_check=time.max(machine_avail[machine]).saturating_add(pt);
                if machine_avail[machine]>time||end_check!=best_end_now{break;}
                let end_time=time.saturating_add(pt);
                job_schedule[job].push((machine,time)); job_next_op[job]+=1; job_ready_time[job]=end_time; machine_avail[machine]=end_time; remaining_ops-=1;
                if chaotic_like{machine_work[machine]=machine_work[machine].saturating_add(pt as u64);sum_work=sum_work.saturating_add(pt as u64);}
                if op.min_pt<INF&&op.flex>0&&!op.machines.is_empty(){let delta=(op.min_pt as f64)/(op.flex as f64).max(1.0);if delta>0.0{for &(mm,_) in &op.machines{let v=machine_load[mm]-delta;machine_load[mm]=if v>0.0{v}else{0.0};}}}
                if remaining_ops==0{break;}
            }
            if remaining_ops==0{break;}
            let mut next_time: Option<u32>=None;
            for &t in &machine_avail{if t>time{next_time=Some(next_time.map_or(t,|b|b.min(t)));}}
            for j in 0..num_jobs{let op_idx=job_next_op[j];if op_idx<pre.job_ops_len[j]&&job_ready_time[j]>time{let t=job_ready_time[j];next_time=Some(next_time.map_or(t,|b|b.min(t)));}}
            time=next_time.ok_or_else(||anyhow!("Stalled"))?;
        }
        let mk=machine_avail.into_iter().max().unwrap_or(0);
        Ok((Solution{job_schedule},mk))
    }

    fn greedy_reassign_pass(pre: &Pre, challenge: &Challenge, base_sol: &Solution) -> Result<Option<(Solution, u32)>> {
        let mut ds=build_disj_from_solution(pre,challenge,base_sol)?; let mut buf=EvalBuf::new(ds.n); let n=ds.n;
        let Some((mut current_mk,_))=eval_disj(&ds,&mut buf) else{return Ok(None)};
        let initial_mk=current_mk; let mut improved=true; let mut passes=0; let max_passes=3;
        while improved&&passes<max_passes {
            improved=false; passes+=1;
            for node in 0..n {
                let job=ds.node_job[node]; let op_idx=ds.node_op[node]; let product=pre.job_products[job];
                let op_info=&pre.product_ops[product][op_idx]; if op_info.machines.len()<=1{continue;}
                let cur_machine=ds.node_machine[node]; let cur_pt=ds.node_pt[node];
                let mut best_m=cur_machine; let mut best_pt=cur_pt; let mut best_mk=current_mk; let mut best_ins_pos=0usize;
                for &(new_m,new_pt) in &op_info.machines {
                    if new_m==cur_machine{continue;}
                    let old_pos=match ds.machine_seq[cur_machine].iter().position(|&x|x==node){Some(p)=>p,None=>continue};
                    ds.machine_seq[cur_machine].remove(old_pos); ds.node_machine[node]=new_m; ds.node_pt[node]=new_pt;
                    let target_len=ds.machine_seq[new_m].len(); let cur_start=buf.start[node]; let mut sorted_pos=target_len;
                    for (k,&nd) in ds.machine_seq[new_m].iter().enumerate(){if buf.start[nd]>=cur_start{sorted_pos=k;break;}}
                    let mut positions: Vec<usize>=Vec::with_capacity(3);
                    for &p in &[sorted_pos,sorted_pos.saturating_sub(1),target_len]{if p<=target_len&&!positions.contains(&p){positions.push(p);}}
                    for &pos in &positions {
                        ds.machine_seq[new_m].insert(pos,node);
                        if let Some((test_mk,_))=eval_disj(&ds,&mut buf){if test_mk<best_mk{best_mk=test_mk;best_m=new_m;best_pt=new_pt;best_ins_pos=pos;}}
                        ds.machine_seq[new_m].remove(pos);
                    }
                    ds.machine_seq[cur_machine].insert(old_pos,node); ds.node_machine[node]=cur_machine; ds.node_pt[node]=cur_pt;
                }
                if best_m!=cur_machine {
                    let old_pos=ds.machine_seq[cur_machine].iter().position(|&x|x==node).unwrap();
                    ds.machine_seq[cur_machine].remove(old_pos);
                    let ins=best_ins_pos.min(ds.machine_seq[best_m].len());
                    ds.machine_seq[best_m].insert(ins,node); ds.node_machine[node]=best_m; ds.node_pt[node]=best_pt;
                    current_mk=best_mk; improved=true;
                }
            }
        }
        if current_mk>=initial_mk{return Ok(None);}
        let Some((_,_))=eval_disj(&ds,&mut buf) else{return Ok(None)};
        let sol=disj_to_solution(pre,&ds,&buf.start)?; Ok(Some((sol,current_mk)))
    }

    fn iterative_cp_descent(
        pre: &Pre,
        challenge: &Challenge,
        base_sol: &Solution,
        max_iters: usize,
    ) -> Result<Option<(Solution, u32)>> {
        let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
        let mut buf = EvalBuf::new(ds.n);
        let Some((mut current_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None); };
        let initial_mk = current_mk;
        let n = ds.n;

        let mut job_op_to_node: Vec<Vec<usize>> = vec![vec![]; challenge.num_jobs];
        for nd in 0..n {
            let job = ds.node_job[nd];
            let op_idx = ds.node_op[nd];
            if op_idx >= job_op_to_node[job].len() {
                job_op_to_node[job].resize(op_idx + 1, usize::MAX);
            }
            job_op_to_node[job][op_idx] = nd;
        }

        for _iter in 0..max_iters {
            let mut finish = vec![0u32; n];
            for nd in 0..n { finish[nd] = buf.start[nd].saturating_add(ds.node_pt[nd]); }
            let makespan = current_mk;

            let mut on_cp = vec![false; n];
            for nd in 0..n { if finish[nd] == makespan { on_cp[nd] = true; } }

            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| buf.start[b].cmp(&buf.start[a]));

            for &nd in &order {
                if !on_cp[nd] { continue; }
                let job = ds.node_job[nd];
                let op_idx = ds.node_op[nd];
                if op_idx > 0 {
                    let pred_op = op_idx - 1;
                    if pred_op < job_op_to_node[job].len() {
                        let pred = job_op_to_node[job][pred_op];
                        if pred != usize::MAX && finish[pred] == buf.start[nd] { on_cp[pred] = true; }
                    }
                }
                let machine = ds.node_machine[nd];
                let seq = &ds.machine_seq[machine];
                if let Some(pos) = seq.iter().position(|&x| x == nd) {
                    if pos > 0 {
                        let pred = seq[pos - 1];
                        if finish[pred] == buf.start[nd] { on_cp[pred] = true; }
                    }
                }
            }

            let mut cp_flex: Vec<usize> = (0..n).filter(|&nd| {
                if !on_cp[nd] { return false; }
                let job = ds.node_job[nd]; let op_idx = ds.node_op[nd];
                let product = pre.job_products[job];
                op_idx < pre.product_ops[product].len() && pre.product_ops[product][op_idx].machines.len() > 1
            }).collect();
            cp_flex.sort_by_key(|&nd| buf.start[nd]);

            if cp_flex.is_empty() { break; }

            let mut iter_improved = false;

            for &nd in &cp_flex {
                let job = ds.node_job[nd]; let op_idx = ds.node_op[nd];
                let product = pre.job_products[job];
                if op_idx >= pre.product_ops[product].len() { continue; }
                let op_info = &pre.product_ops[product][op_idx];
                let cur_m = ds.node_machine[nd]; let cur_pt = ds.node_pt[nd];
                let mut best_m = cur_m; let mut best_pt = cur_pt;
                let mut best_mk = current_mk; let mut best_ins = 0usize;

                for &(new_m, new_pt) in &op_info.machines {
                    if new_m == cur_m { continue; }
                    let old_pos = match ds.machine_seq[cur_m].iter().position(|&x| x == nd) { Some(p)=>p, None=>continue };
                    ds.machine_seq[cur_m].remove(old_pos);
                    ds.node_machine[nd] = new_m; ds.node_pt[nd] = new_pt;
                    let _ = eval_disj(&ds, &mut buf);
                    let cur_start = buf.start[nd];
                    let tlen = ds.machine_seq[new_m].len();
                    let mut spos = tlen;
                    for (ki, &nd2) in ds.machine_seq[new_m].iter().enumerate() {
                        if buf.start[nd2] >= cur_start { spos = ki; break; }
                    }
                    let mut positions: Vec<usize> = Vec::with_capacity(4);
                    for &p in &[spos, spos.saturating_sub(1), (spos+1).min(tlen), tlen] {
                        if p <= tlen && !positions.contains(&p) { positions.push(p); }
                    }
                    for &pos in &positions {
                        ds.machine_seq[new_m].insert(pos, nd);
                        if let Some((tmk, _)) = eval_disj(&ds, &mut buf) {
                            if tmk < best_mk { best_mk = tmk; best_m = new_m; best_pt = new_pt; best_ins = pos; }
                        }
                        ds.machine_seq[new_m].remove(pos);
                    }
                    ds.machine_seq[cur_m].insert(old_pos, nd);
                    ds.node_machine[nd] = cur_m; ds.node_pt[nd] = cur_pt;
                }
                if best_m != cur_m {
                    let old_pos = ds.machine_seq[cur_m].iter().position(|&x| x == nd).unwrap();
                    ds.machine_seq[cur_m].remove(old_pos);
                    let ins = best_ins.min(ds.machine_seq[best_m].len());
                    ds.machine_seq[best_m].insert(ins, nd);
                    ds.node_machine[nd] = best_m; ds.node_pt[nd] = best_pt;
                    current_mk = best_mk;
                    let _ = eval_disj(&ds, &mut buf);
                    iter_improved = true;
                }
            }

            if cp_flex.len() >= 2 {
                let pairs: Vec<(usize, usize)> = cp_flex.windows(2).map(|w| (w[0], w[1])).collect();

                let cand_positions = |ds: &DisjSchedule, buf: &EvalBuf, m: usize, nd: usize| -> Vec<usize> {
                    let cur_start = buf.start[nd];
                    let tlen = ds.machine_seq[m].len();
                    let mut spos = tlen;
                    for (ki, &nx) in ds.machine_seq[m].iter().enumerate() {
                        if buf.start[nx] >= cur_start {
                            spos = ki;
                            break;
                        }
                    }
                    let mut positions: Vec<usize> = Vec::with_capacity(3);
                    for &p in &[spos, spos.saturating_sub(1), tlen] {
                        if p <= tlen && !positions.contains(&p) {
                            positions.push(p);
                        }
                    }
                    positions
                };

                for (nd1, nd2) in pairs {
                    let job1 = ds.node_job[nd1];
                    let op1 = ds.node_op[nd1];
                    let job2 = ds.node_job[nd2];
                    let op2 = ds.node_op[nd2];
                    let prod1 = pre.job_products[job1];
                    let prod2 = pre.job_products[job2];
                    if op1 >= pre.product_ops[prod1].len() || op2 >= pre.product_ops[prod2].len() {
                        continue;
                    }
                    let op_info1 = &pre.product_ops[prod1][op1];
                    let op_info2 = &pre.product_ops[prod2][op2];
                    if op_info1.machines.len() <= 1 && op_info2.machines.len() <= 1 {
                        continue;
                    }

                    let cur_m1 = ds.node_machine[nd1];
                    let cur_pt1 = ds.node_pt[nd1];
                    let cur_m2 = ds.node_machine[nd2];
                    let cur_pt2 = ds.node_pt[nd2];

                    let mut opts1: Vec<(usize, u32)> = Vec::new();
                    opts1.push((cur_m1, cur_pt1));
                    opts1.extend(
                        op_info1
                            .machines
                            .iter()
                            .copied()
                            .filter(|&(m, _)| m != cur_m1)
                            .take(4),
                    );

                    let mut opts2: Vec<(usize, u32)> = Vec::new();
                    opts2.push((cur_m2, cur_pt2));
                    opts2.extend(
                        op_info2
                            .machines
                            .iter()
                            .copied()
                            .filter(|&(m, _)| m != cur_m2)
                            .take(4),
                    );

                    let pos1 = match ds.machine_seq[cur_m1].iter().position(|&x| x == nd1) {
                        Some(p) => p,
                        None => continue,
                    };
                    let pos2 = match ds.machine_seq[cur_m2].iter().position(|&x| x == nd2) {
                        Some(p) => p,
                        None => continue,
                    };

                    if cur_m1 == cur_m2 {
                        if pos1 > pos2 {
                            ds.machine_seq[cur_m1].remove(pos1);
                            ds.machine_seq[cur_m2].remove(pos2);
                        } else {
                            ds.machine_seq[cur_m2].remove(pos2);
                            ds.machine_seq[cur_m1].remove(pos1);
                        }
                    } else {
                        ds.machine_seq[cur_m1].remove(pos1);
                        ds.machine_seq[cur_m2].remove(pos2);
                    }

                    let mut best_mk_pair = current_mk;
                    let mut best_config: Option<(usize, u32, usize, usize, u32, usize, u8)> = None;

                    for &(m1, pt1) in &opts1 {
                        for &(m2, pt2) in &opts2 {
                            if m1 != m2 {
                                ds.node_machine[nd1] = m1;
                                ds.node_pt[nd1] = pt1;
                                ds.node_machine[nd2] = m2;
                                ds.node_pt[nd2] = pt2;

                                let p1s = cand_positions(&ds, &buf, m1, nd1);
                                let p2s = cand_positions(&ds, &buf, m2, nd2);

                                for &p1i in &p1s {
                                    ds.machine_seq[m1].insert(p1i, nd1);
                                    for &p2i in &p2s {
                                        ds.machine_seq[m2].insert(p2i, nd2);
                                        if let Some((tmk, _)) = eval_disj(&ds, &mut buf) {
                                            if tmk < best_mk_pair {
                                                best_mk_pair = tmk;
                                                best_config = Some((m1, pt1, p1i, m2, pt2, p2i, 0));
                                            }
                                        }
                                        ds.machine_seq[m2].remove(p2i);
                                    }
                                    ds.machine_seq[m1].remove(p1i);
                                }
                            } else {
                                let m = m1;
                                ds.node_machine[nd1] = m;
                                ds.node_pt[nd1] = pt1;
                                ds.node_machine[nd2] = m;
                                ds.node_pt[nd2] = pt2;

                                for order_flag in 0u8..=1u8 {
                                    let (a, b) = if order_flag == 0 { (nd1, nd2) } else { (nd2, nd1) };

                                    let pa = cand_positions(&ds, &buf, m, a);
                                    for &pai in &pa {
                                        ds.machine_seq[m].insert(pai, a);
                                        let pb = cand_positions(&ds, &buf, m, b);
                                        for &pbi in &pb {
                                            ds.machine_seq[m].insert(pbi, b);
                                            if let Some((tmk, _)) = eval_disj(&ds, &mut buf) {
                                                if tmk < best_mk_pair {
                                                    best_mk_pair = tmk;
                                                    if order_flag == 0 {
                                                        best_config = Some((m, pt1, pai, m, pt2, pbi, 0));
                                                    } else {
                                                        best_config = Some((m, pt1, pbi, m, pt2, pai, 1));
                                                    }
                                                }
                                            }
                                            ds.machine_seq[m].remove(pbi);
                                        }
                                        ds.machine_seq[m].remove(pai);
                                    }
                                }
                            }
                        }
                    }

                    if let Some((bm1, bpt1, bp1, bm2, bpt2, bp2, order_flag)) = best_config {
                        if best_mk_pair < current_mk {
                            ds.node_machine[nd1] = bm1;
                            ds.node_pt[nd1] = bpt1;
                            ds.node_machine[nd2] = bm2;
                            ds.node_pt[nd2] = bpt2;

                            if bm1 != bm2 {
                                let ins1 = bp1.min(ds.machine_seq[bm1].len());
                                ds.machine_seq[bm1].insert(ins1, nd1);
                                let ins2 = bp2.min(ds.machine_seq[bm2].len());
                                ds.machine_seq[bm2].insert(ins2, nd2);
                            } else {
                                let m = bm1;
                                if order_flag == 0 {
                                    let ins1 = bp1.min(ds.machine_seq[m].len());
                                    ds.machine_seq[m].insert(ins1, nd1);
                                    let ins2 = bp2.min(ds.machine_seq[m].len());
                                    ds.machine_seq[m].insert(ins2, nd2);
                                } else {
                                    let ins2 = bp2.min(ds.machine_seq[m].len());
                                    ds.machine_seq[m].insert(ins2, nd2);
                                    let ins1 = bp1.min(ds.machine_seq[m].len());
                                    ds.machine_seq[m].insert(ins1, nd1);
                                }
                            }

                            current_mk = best_mk_pair;
                            let _ = eval_disj(&ds, &mut buf);
                            iter_improved = true;
                            continue;
                        }
                    }

                    ds.node_machine[nd1] = cur_m1;
                    ds.node_pt[nd1] = cur_pt1;
                    ds.node_machine[nd2] = cur_m2;
                    ds.node_pt[nd2] = cur_pt2;

                    if cur_m1 != cur_m2 {
                        let ins1 = pos1.min(ds.machine_seq[cur_m1].len());
                        ds.machine_seq[cur_m1].insert(ins1, nd1);
                        let ins2 = pos2.min(ds.machine_seq[cur_m2].len());
                        ds.machine_seq[cur_m2].insert(ins2, nd2);
                    } else {
                        let m = cur_m1;
                        if pos1 <= pos2 {
                            let ins1 = pos1.min(ds.machine_seq[m].len());
                            ds.machine_seq[m].insert(ins1, nd1);
                            let ins2 = pos2.min(ds.machine_seq[m].len());
                            ds.machine_seq[m].insert(ins2, nd2);
                        } else {
                            let ins2 = pos2.min(ds.machine_seq[m].len());
                            ds.machine_seq[m].insert(ins2, nd2);
                            let ins1 = pos1.min(ds.machine_seq[m].len());
                            ds.machine_seq[m].insert(ins1, nd1);
                        }
                    }

                    let _ = eval_disj(&ds, &mut buf);
                }
            }

            if !iter_improved { break; }
        }

        if current_mk >= initial_mk { return Ok(None); }
        let Some((final_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None); };
        if final_mk >= initial_mk { return Ok(None); }
        let sol = disj_to_solution(pre, &ds, &buf.start)?;
        Ok(Some((sol, final_mk)))
    }

    fn cp_window_exhaustive(
        pre: &Pre,
        challenge: &Challenge,
        base_sol: &Solution,
        max_iters: usize,
    ) -> Result<Option<(Solution, u32)>> {
        let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
        let mut buf = EvalBuf::new(ds.n);
        let Some((mut current_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None); };
        let initial_mk = current_mk;
        let n = ds.n;

        let mut job_op_to_node: Vec<Vec<usize>> = vec![vec![]; challenge.num_jobs];
        for nd in 0..n {
            let job = ds.node_job[nd];
            let op_idx = ds.node_op[nd];
            if op_idx >= job_op_to_node[job].len() {
                job_op_to_node[job].resize(op_idx + 1, usize::MAX);
            }
            job_op_to_node[job][op_idx] = nd;
        }

        for _iter in 0..max_iters {
            let mut finish = vec![0u32; n];
            for nd in 0..n { finish[nd] = buf.start[nd].saturating_add(ds.node_pt[nd]); }
            let makespan = current_mk;

            let mut on_cp = vec![false; n];
            for nd in 0..n { if finish[nd] == makespan { on_cp[nd] = true; } }

            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| buf.start[b].cmp(&buf.start[a]));
            for &nd in &order {
                if !on_cp[nd] { continue; }
                let job = ds.node_job[nd];
                let op_idx = ds.node_op[nd];
                if op_idx > 0 {
                    let pred_op = op_idx - 1;
                    if pred_op < job_op_to_node[job].len() {
                        let pred = job_op_to_node[job][pred_op];
                        if pred != usize::MAX && finish[pred] == buf.start[nd] { on_cp[pred] = true; }
                    }
                }
                let machine = ds.node_machine[nd];
                let seq = &ds.machine_seq[machine];
                if let Some(pos) = seq.iter().position(|&x| x == nd) {
                    if pos > 0 {
                        let pred_nd = seq[pos - 1];
                        if finish[pred_nd] == buf.start[nd] { on_cp[pred_nd] = true; }
                    }
                }
            }

            let mut cp_flex: Vec<usize> = (0..n).filter(|&nd| {
                if !on_cp[nd] { return false; }
                let job = ds.node_job[nd]; let op_idx = ds.node_op[nd];
                let product = pre.job_products[job];
                op_idx < pre.product_ops[product].len() && pre.product_ops[product][op_idx].machines.len() > 1
            }).collect();
            cp_flex.sort_by_key(|&nd| buf.start[nd]);

            if cp_flex.is_empty() { break; }

            let mut machine_cp_count = vec![0usize; challenge.num_machines];
            for &nd in &cp_flex { machine_cp_count[ds.node_machine[nd]] += 1; }
            let bottleneck_m = machine_cp_count.iter().enumerate()
                .max_by_key(|&(_, &c)| c).map(|(m, _)| m).unwrap_or(0);

            let window_sizes = [4usize, 3, 2];
            let mut best_window_improvement = false;

            'window_search: for &wsize in &window_sizes {
                if cp_flex.len() < wsize { continue; }
                let bottleneck_ops: Vec<usize> = cp_flex.iter().copied()
                    .filter(|&nd| ds.node_machine[nd] == bottleneck_m)
                    .collect();
                if bottleneck_ops.is_empty() { continue; }

                let num_windows = cp_flex.len().saturating_sub(wsize) + 1;
                for w_start in 0..num_windows {
                    let window: Vec<usize> = cp_flex[w_start..w_start+wsize].to_vec();
                    let has_bottleneck = window.iter().any(|&nd| ds.node_machine[nd] == bottleneck_m);
                    if !has_bottleneck { continue; }

                    let total_alts: usize = window.iter().map(|&nd| {
                        let job = ds.node_job[nd]; let op_idx = ds.node_op[nd];
                        let product = pre.job_products[job];
                        if op_idx < pre.product_ops[product].len() {
                            pre.product_ops[product][op_idx].machines.len()
                        } else { 1 }
                    }).product::<usize>();
                    if total_alts > 512 { continue; }

                    let orig: Vec<(usize, u32, usize)> = window.iter().map(|&nd| {
                        let cur_m = ds.node_machine[nd];
                        let cur_pt = ds.node_pt[nd];
                        let cur_pos = ds.machine_seq[cur_m].iter().position(|&x| x == nd).unwrap_or(0);
                        (cur_m, cur_pt, cur_pos)
                    }).collect();

                    let options: Vec<Vec<(usize, u32)>> = window.iter().map(|&nd| {
                        let job = ds.node_job[nd]; let op_idx = ds.node_op[nd];
                        let product = pre.job_products[job];
                        if op_idx < pre.product_ops[product].len() {
                            pre.product_ops[product][op_idx].machines.iter().copied().collect()
                        } else { vec![] }
                    }).collect();

                    let mut best_combo_mk = current_mk;
                    let mut best_combo: Option<Vec<(usize, u32)>> = None;

                    let total_nodes = window.len();
                    let mut combo_indices = vec![0usize; total_nodes];
                    'combo_loop: loop {
                        let mut apply_ok = true;
                        for (wi, &nd) in window.iter().enumerate() {
                            if options[wi].is_empty() { apply_ok = false; break; }
                            let (new_m, new_pt) = options[wi][combo_indices[wi]];
                            let cur_m = ds.node_machine[nd];
                            if new_m == cur_m {
                                continue;
                            }
                            let old_pos = match ds.machine_seq[cur_m].iter().position(|&x| x == nd) {
                                Some(p) => p, None => { apply_ok = false; break; }
                            };
                            ds.machine_seq[cur_m].remove(old_pos);
                            ds.node_machine[nd] = new_m;
                            ds.node_pt[nd] = new_pt;
                            ds.machine_seq[new_m].push(nd);
                        }

                        if apply_ok {
                            let affected_machines: Vec<usize> = {
                                let mut ms: Vec<usize> = window.iter().map(|&nd| ds.node_machine[nd]).collect();
                                ms.extend(orig.iter().map(|&(m,_,_)| m));
                                ms.sort(); ms.dedup(); ms
                            };
                            for &m in &affected_machines {
                                let seq = &mut ds.machine_seq[m];
                                seq.sort_by_key(|&nd2| buf.start[nd2]);
                            }

                            if let Some((tmk, _)) = eval_disj(&ds, &mut buf) {
                                if tmk < best_combo_mk {
                                    best_combo_mk = tmk;
                                    best_combo = Some(window.iter().map(|&nd| {
                                        (ds.node_machine[nd], ds.node_pt[nd])
                                    }).collect());
                                }
                            }
                        }

                        for (wi, &nd) in window.iter().enumerate() {
                            let (orig_m, orig_pt, orig_pos) = orig[wi];
                            let cur_m = ds.node_machine[nd];
                            if cur_m != orig_m {
                                if let Some(p) = ds.machine_seq[cur_m].iter().position(|&x| x == nd) {
                                    ds.machine_seq[cur_m].remove(p);
                                }
                                ds.node_machine[nd] = orig_m;
                                ds.node_pt[nd] = orig_pt;
                                let ins = orig_pos.min(ds.machine_seq[orig_m].len());
                                ds.machine_seq[orig_m].insert(ins, nd);
                            }
                        }
                        let _ = eval_disj(&ds, &mut buf);

                        let mut carry = true;
                        for wi in (0..total_nodes).rev() {
                            if carry {
                                combo_indices[wi] += 1;
                                if combo_indices[wi] < options[wi].len() {
                                    carry = false;
                                } else {
                                    combo_indices[wi] = 0;
                                }
                            }
                        }
                        if carry { break 'combo_loop; }
                    }

                    if let Some(best_assign) = best_combo {
                        if best_combo_mk < current_mk {
                            for (wi, &nd) in window.iter().enumerate() {
                                let (new_m, new_pt) = best_assign[wi];
                                let cur_m = ds.node_machine[nd];
                                if new_m != cur_m {
                                    let old_pos = ds.machine_seq[cur_m].iter().position(|&x| x == nd).unwrap();
                                    ds.machine_seq[cur_m].remove(old_pos);
                                    ds.node_machine[nd] = new_m;
                                    ds.node_pt[nd] = new_pt;
                                    ds.machine_seq[new_m].push(nd);
                                }
                            }
                            let affected_machines: Vec<usize> = {
                                let mut ms: Vec<usize> = window.iter().map(|&nd| ds.node_machine[nd]).collect();
                                ms.extend(orig.iter().map(|&(m,_,_)| m));
                                ms.sort(); ms.dedup(); ms
                            };
                            for &m in &affected_machines {
                                ds.machine_seq[m].sort_by_key(|&nd2| buf.start[nd2]);
                            }
                            let _ = eval_disj(&ds, &mut buf);
                            current_mk = best_combo_mk;
                            best_window_improvement = true;
                            break 'window_search;
                        }
                    }
                }
                if best_window_improvement { break; }
            }

            if !best_window_improvement { break; }
        }

        if current_mk >= initial_mk { return Ok(None); }
        let Some((final_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None); };
        if final_mk >= initial_mk { return Ok(None); }
        let sol = disj_to_solution(pre, &ds, &buf.start)?;
        Ok(Some((sol, final_mk)))
    }

    #[inline]
    fn solution_machine_signature(pre: &Pre, challenge: &Challenge, sol: &Solution) -> Vec<u16> {
        let num_machines = challenge.num_machines;
        let mut sig: Vec<u16> = Vec::with_capacity(pre.total_ops);
        for job in 0..challenge.num_jobs {
            let lim = pre.job_ops_len[job].min(sol.job_schedule[job].len());
            for op_idx in 0..pre.job_ops_len[job] {
                let m_u16 = if op_idx < lim {
                    let m = sol.job_schedule[job][op_idx].0;
                    if m < num_machines { m as u16 } else { u16::MAX }
                } else {
                    u16::MAX
                };
                sig.push(m_u16);
            }
        }
        sig
    }

    #[inline]
    fn hamming_distance_sig(a: &[u16], b: &[u16]) -> u32 {
        let mut d = 0u32;
        let n = a.len().min(b.len());
        for i in 0..n {
            if a[i] != b[i] {
                d = d.saturating_add(1);
            }
        }
        d
    }

    fn push_top_solutions_diverse(
        pre: &Pre,
        challenge: &Challenge,
        pool: &mut Vec<(Solution, u32, Vec<u16>)>,
        sol: &Solution,
        mk: u32,
        cap: usize,
    ) {
        let sig = solution_machine_signature(pre, challenge, sol);
        pool.push((sol.clone(), mk, sig));

        while pool.len() > cap {
            let len = pool.len();
            let mut min_nn = vec![u32::MAX; len];
            for i in 0..len {
                for j in (i + 1)..len {
                    let d = hamming_distance_sig(&pool[i].2, &pool[j].2);
                    if d < min_nn[i] {
                        min_nn[i] = d;
                    }
                    if d < min_nn[j] {
                        min_nn[j] = d;
                    }
                }
            }

            let mut worst_mk = pool[0].1;
            for i in 1..len {
                if pool[i].1 > worst_mk {
                    worst_mk = pool[i].1;
                }
            }

            let mut drop_idx: Option<usize> = None;
            let mut drop_min_nn = u32::MAX;
            for i in 0..len {
                if pool[i].1 != worst_mk {
                    continue;
                }
                let mnn = min_nn[i];
                if drop_idx.is_none() || mnn < drop_min_nn {
                    drop_idx = Some(i);
                    drop_min_nn = mnn;
                }
            }
            let di = drop_idx.unwrap_or(0);
            pool.swap_remove(di);
        }
    }

    fn consensus_learning_from_elites(
        pre: &Pre,
        challenge: &Challenge,
        elites: &[(Solution, u32, Vec<u16>)],
    ) -> Result<(Vec<f64>, Vec<f64>, RoutePrefLite)> {
        if elites.is_empty() {
            return Err(anyhow!("No elites for consensus learning"));
        }
        let num_jobs = challenge.num_jobs;
        let num_machines = challenge.num_machines;

        let mut jb_sum = vec![0.0f64; num_jobs];
        for (sol, _, _) in elites.iter() {
            let jb = job_bias_from_solution(pre, sol)?;
            for j in 0..num_jobs {
                jb_sum[j] += jb[j];
            }
        }
        let denom = elites.len() as f64;
        for j in 0..num_jobs {
            jb_sum[j] /= denom;
        }

        let mut bottleneck_cnt = vec![0u32; num_machines];
        let mut machine_end = vec![0u32; num_machines];
        for (sol, mk, _) in elites.iter() {
            let mk = *mk;
            machine_end.fill(0);
            for job in 0..num_jobs {
                let product = pre.job_products[job];
                if product >= pre.product_ops.len() {
                    continue;
                }
                let sched = &sol.job_schedule[job];
                let ops = &pre.product_ops[product];
                let lim = sched.len().min(ops.len());
                for op_idx in 0..lim {
                    let (m, st) = sched[op_idx];
                    if m >= num_machines {
                        continue;
                    }
                    let op_info = &ops[op_idx];
                    let mut pt = None;
                    for &(mm, p) in &op_info.machines {
                        if mm == m {
                            pt = Some(p);
                            break;
                        }
                    }
                    let pt = pt.unwrap_or(op_info.min_pt);
                    if pt >= INF {
                        continue;
                    }
                    let end = st.saturating_add(pt);
                    if end > machine_end[m] {
                        machine_end[m] = end;
                    }
                }
            }
            for m in 0..num_machines {
                if machine_end[m] == mk {
                    bottleneck_cnt[m] = bottleneck_cnt[m].saturating_add(1);
                }
            }
        }
        let mut machine_penalty = vec![0.0f64; num_machines];
        for m in 0..num_machines {
            machine_penalty[m] = (bottleneck_cnt[m] as f64) / denom;
        }

        let mut rp = route_pref_from_solution_lite(pre, &elites[0].0, challenge)?;
        let num_products = pre.product_ops.len();
        let mut counts: Vec<Vec<Vec<u32>>> = pre
            .product_ops
            .iter()
            .map(|ops| ops.iter().map(|_| vec![0u32; num_machines]).collect())
            .collect();

        for (sol, _, _) in elites.iter() {
            for job in 0..num_jobs {
                let product = pre.job_products[job];
                if product >= num_products {
                    continue;
                }
                let sched = &sol.job_schedule[job];
                let ops_len = counts[product].len();
                for (op_idx, (m, _)) in sched.iter().enumerate() {
                    if op_idx >= ops_len {
                        break;
                    }
                    if *m < num_machines {
                        counts[product][op_idx][*m] = counts[product][op_idx][*m].saturating_add(1);
                    }
                }
            }
        }

        for product in 0..num_products {
            let ops_len = counts[product].len();
            if product >= rp.len() {
                break;
            }
            for op_idx in 0..ops_len {
                if op_idx >= rp[product].len() {
                    break;
                }
                let row = &counts[product][op_idx];
                let mut total = 0u32;
                for &c in row.iter() {
                    total = total.saturating_add(c);
                }
                if total == 0 {
                    continue;
                }

                let mut best_m = 0usize;
                let mut best_c = 0u32;
                let mut second_m = 0usize;
                let mut second_c = 0u32;
                for (m, &c) in row.iter().enumerate() {
                    if c > best_c {
                        second_m = best_m;
                        second_c = best_c;
                        best_m = m;
                        best_c = c;
                    } else if m != best_m && c > second_c {
                        second_m = m;
                        second_c = c;
                    }
                }
                if second_c == 0 {
                    second_m = best_m;
                }

                let total64 = total as u64;
                let best_w = ((best_c as u64) * 255u64 / total64) as u8;
                let second_w = ((second_c as u64) * 255u64 / total64) as u8;

                rp[product][op_idx].best_m = (best_m.min(255)) as u8;
                rp[product][op_idx].best_w = best_w;
                rp[product][op_idx].second_m = (second_m.min(255)) as u8;
                rp[product][op_idx].second_w = second_w;
            }
        }

        Ok((jb_sum, machine_penalty, rp))
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
        let rules: Vec<Rule> = vec![Rule::Adaptive,Rule::BnHeavy,Rule::EndTight,Rule::CriticalPath,Rule::MostWork,Rule::LeastFlex,Rule::Regret,Rule::ShortestProc,Rule::FlexBalance];
        let mut best_makespan = greedy_mk;
        let mut best_solution: Option<Solution> = Some(greedy_sol);
        let mut top_solutions: Vec<(Solution, u32, Vec<u16>)> = Vec::new();
        let target_margin: u32 = ((pre.avg_op_min*(0.9+0.9*pre.high_flex+0.6*pre.jobshopness)).max(1.0)) as u32;
        let route_w_base: f64 = (0.040+0.10*pre.high_flex+0.08*pre.jobshopness).clamp(0.04,0.22);

        if pre.flow_route.is_some()&&pre.flow_pt_by_job.is_some() {
            if let Ok((sol,mk))=neh_reentrant_flow_solution(pre,challenge.num_jobs,challenge.num_machines) {
                if mk<best_makespan{best_makespan=mk;best_solution=Some(sol.clone());save_solution(&sol)?;}
                push_top_solutions_diverse(pre, challenge, &mut top_solutions, &sol, mk, 15);
            }
        }
        let mut ranked: Vec<(Rule,u32,Solution)>=Vec::with_capacity(rules.len());
        for &rule in &rules {
            let (sol,mk)=construct_solution_conflict(challenge,pre,rule,0,None,&mut rng,None,None,None,0.0)?;
            if mk<best_makespan{best_makespan=mk;best_solution=Some(sol.clone());save_solution(&sol)?;}
            push_top_solutions_diverse(pre, challenge, &mut top_solutions, &sol, mk, 15);
            ranked.push((rule,mk,sol));
        }
        ranked.sort_by_key(|x|x.1);
        let r0=ranked[0].0; let r1=ranked.get(1).map(|x|x.0).unwrap_or(r0); let r2=ranked.get(2).map(|x|x.0).unwrap_or(r1);
        let mut rule_best: Vec<u32>=vec![u32::MAX;10]; let mut rule_tries: Vec<u32>=vec![0u32;10];
        for (rr,mk,_) in &ranked{let idx=rule_idx(*rr);rule_best[idx]=rule_best[idx].min(*mk);rule_tries[idx]=rule_tries[idx].saturating_add(1);}
        let (jb0, mp0, rp0) = consensus_learning_from_elites(pre, challenge, &top_solutions)?;
        let mut learned_jb=Some(jb0);
        let mut learned_mp=Some(mp0);
        let mut learned_rp=Some(rp0);
        let mut learn_updates_left=10usize;
        let num_restarts=effort.fjsp_high_iters;
        let k_hi=if pre.flex_avg>8.0{8}else if pre.flex_avg>6.5{7}else{6};
        let mut stuck: usize=0;
        for r in 0..num_restarts {
            let late=r>=(num_restarts*2)/3;
            let (k_min,k_max)=if stuck>170{(4usize,6usize.min(k_hi))}else if stuck>90{(3usize,6usize.min(k_hi))}else if stuck>35{(2usize,6usize.min(k_hi))}else{(2usize,4usize.min(k_hi))};
            let rule=if r<35{let u: f64=rng.gen();if u<0.12{Rule::FlexBalance}else if u<0.50{r0}else if u<0.75{r1}else if u<0.90{r2}else{rules[rng.gen_range(0..rules.len())]}}
                else{choose_rule_bandit(&mut rng,&rules,&rule_best,&rule_tries,best_makespan,target_margin,stuck,false,late)};
            let k=if k_max<=k_min{k_min}else{rng.gen_range(k_min..=k_max)};
            let learn_base=(0.08+0.22*pre.jobshopness+0.18*pre.high_flex).clamp(0.05,0.42);
            let learn_boost=(1.0+0.35*((stuck as f64)/120.0).clamp(0.0,1.0)).clamp(1.0,1.35);
            let learn_p=(learn_base*learn_boost).clamp(0.0,0.60);
            let use_learn=learned_jb.is_some()&&learned_mp.is_some()&&rng.gen::<f64>()<learn_p&&learned_rp.is_some();
            let target=if best_makespan<(u32::MAX/2){Some(best_makespan.saturating_add(target_margin))}else{None};
            let (sol,mk)=if use_learn{construct_solution_conflict(challenge,pre,rule,k,target,&mut rng,learned_jb.as_deref(),learned_mp.as_deref(),learned_rp.as_ref(),route_w_base)?}
                else{construct_solution_conflict(challenge,pre,rule,k,target,&mut rng,None,None,None,0.0)?};
            let ridx=rule_idx(rule);rule_tries[ridx]=rule_tries[ridx].saturating_add(1);rule_best[ridx]=rule_best[ridx].min(mk);
            push_top_solutions_diverse(pre, challenge, &mut top_solutions, &sol, mk, 15);
            if mk<best_makespan{best_makespan=mk;best_solution=Some(sol.clone());save_solution(&sol)?;stuck=0;if learn_updates_left>0{let (jb,mp,rp)=consensus_learning_from_elites(pre,challenge,&top_solutions)?;learned_jb=Some(jb);learned_mp=Some(mp);learned_rp=Some(rp);learn_updates_left-=1;}}else{stuck=stuck.saturating_add(1);}
        }
        let route_w_ls: f64=(route_w_base*1.40).clamp(route_w_base,0.40);
        let mut refine_results: Vec<(Solution,u32)>=Vec::new();
        for (base_sol,_,_) in top_solutions.iter() {
            let jb=job_bias_from_solution(pre,base_sol)?; let mp=machine_penalty_from_solution(pre,base_sol,challenge.num_machines)?;
            let rp=Some(route_pref_from_solution_lite(pre,base_sol,challenge)?);
            let target_ls=if best_makespan<(u32::MAX/2){Some(best_makespan.saturating_add(target_margin/2))}else{None};
            for attempt in 0..10 {
                let rule=match attempt{0=>r0,1=>Rule::Adaptive,2=>Rule::BnHeavy,3=>Rule::EndTight,4=>Rule::Regret,5=>Rule::CriticalPath,6=>Rule::LeastFlex,7=>Rule::MostWork,8=>Rule::FlexBalance,_=>r1};
                let k=match attempt%4{0=>2,1=>3,2=>4,_=>2}.min(k_hi);
                let (sol,mk)=construct_solution_conflict(challenge,pre,rule,k,target_ls,&mut rng,Some(&jb),Some(&mp),rp.as_ref(),route_w_ls)?;
                if mk<best_makespan{best_makespan=mk;best_solution=Some(sol.clone());save_solution(&sol)?;}
                refine_results.push((sol,mk));
            }
        }
        for (sol,mk) in refine_results{push_top_solutions_diverse(pre, challenge, &mut top_solutions, &sol, mk, 15);}
        let ls_runs=top_solutions.len().min(15);
        for i in 0..ls_runs {
            let base_sol=&top_solutions[i].0;
            if let Some((sol2,mk2))=critical_block_move_local_search_ex(pre,challenge,base_sol,8,128,24)?{
                if mk2<best_makespan{best_makespan=mk2;best_solution=Some(sol2.clone());save_solution(&sol2)?;}
                push_top_solutions_diverse(pre, challenge, &mut top_solutions, &sol2, mk2, 15);
            }
        }
        if let Some(ref sol)=best_solution.clone(){if let Ok(Some((sol2,mk2)))=greedy_reassign_pass(pre,challenge,sol){if mk2<best_makespan{best_makespan=mk2;best_solution=Some(sol2.clone());save_solution(&sol2)?;push_top_solutions_diverse(pre, challenge, &mut top_solutions, &sol2, mk2, 15);}}}

        let cp_runs = top_solutions.len().min(12);
        for i in 0..cp_runs {
            let base_sol = top_solutions[i].0.clone();
            if let Ok(Some((sol2, mk2))) = iterative_cp_descent(pre, challenge, &base_sol, 8) {
                if mk2 < best_makespan {
                    best_makespan = mk2;
                    best_solution = Some(sol2.clone());
                    save_solution(&sol2)?;
                    push_top_solutions_diverse(pre, challenge, &mut top_solutions, &sol2, mk2, 15);
                }
                if let Ok(Some((sol3, mk3))) = greedy_reassign_pass(pre, challenge, &sol2) {
                    if mk3 < best_makespan {
                        best_makespan = mk3;
                        best_solution = Some(sol3.clone());
                        save_solution(&sol3)?;
                        push_top_solutions_diverse(pre, challenge, &mut top_solutions, &sol3, mk3, 15);
                    }
                }
            }
        }

        let cpw_runs = top_solutions.len().min(10);
        for i in 0..cpw_runs {
            let base_sol = top_solutions[i].0.clone();
            if let Ok(Some((sol2, mk2))) = cp_window_exhaustive(pre, challenge, &base_sol, 6) {
                if mk2 < best_makespan {
                    best_makespan = mk2;
                    best_solution = Some(sol2.clone());
                    save_solution(&sol2)?;
                    push_top_solutions_diverse(pre, challenge, &mut top_solutions, &sol2, mk2, 15);
                }
                if let Ok(Some((sol3, mk3))) = iterative_cp_descent(pre, challenge, &sol2, 5) {
                    if mk3 < best_makespan {
                        best_makespan = mk3;
                        best_solution = Some(sol3.clone());
                        save_solution(&sol3)?;
                        push_top_solutions_diverse(pre, challenge, &mut top_solutions, &sol3, mk3, 15);
                    }
                }
            }
        }

        if let Some(ref sol) = best_solution.clone() {
            if let Ok(Some((sol2, mk2))) = iterative_cp_descent(pre, challenge, sol, 15) {
                if mk2 < best_makespan {
                    best_makespan = mk2;
                    best_solution = Some(sol2.clone());
                    save_solution(&sol2)?;
                    push_top_solutions_diverse(pre, challenge, &mut top_solutions, &sol2, mk2, 15);
                    if let Ok(Some((sol3, mk3))) = greedy_reassign_pass(pre, challenge, &sol2) {
                        if mk3 < best_makespan {
                            best_solution = Some(sol3.clone());
                            save_solution(&sol3)?;
                        }
                    }
                }
            }
        }

        if let Some(sol)=best_solution{save_solution(&sol)?;}
        Ok(())
    }
}