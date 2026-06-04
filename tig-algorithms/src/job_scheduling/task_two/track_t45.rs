// FLAT: track_t45.rs — auto-generated from t45/ sub-dir

pub use solver::{solve_challenge, help};

pub mod types {
    pub const INF: u32 = u32::MAX / 4;
    pub const NONE_USIZE: usize = usize::MAX;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
        pub num_restarts: usize,
        pub job_shop_iters: usize,
        pub flow_shop_iters: usize,
        pub hybrid_flow_shop_iters: usize,
        pub fjsp_medium_iters: usize,
        pub fjsp_high_iters: usize,
    }

    impl EffortConfig {
        pub fn default_effort() -> Self {
            Self { num_restarts: 2000, job_shop_iters: 100000, flow_shop_iters: 20000, hybrid_flow_shop_iters: 8000, fjsp_medium_iters: 1800, fjsp_high_iters: 5000 }
        }

        pub fn from_str(s: &str) -> Self {
            match s.to_lowercase().as_str() {
                "medium" => Self { num_restarts: 3000, job_shop_iters: 10000, flow_shop_iters: 12000, hybrid_flow_shop_iters: 8000, fjsp_medium_iters: 8000, fjsp_high_iters: 8000 },
                "high" => Self { num_restarts: 4000, job_shop_iters: 15000, flow_shop_iters: 18000, hybrid_flow_shop_iters: 12000, fjsp_medium_iters: 12000, fjsp_high_iters: 12000 },
                "extreme" => Self { num_restarts: 6000, job_shop_iters: 20000, flow_shop_iters: 25000, hybrid_flow_shop_iters: 15000, fjsp_medium_iters: 15000, fjsp_high_iters: 15000 },
                _ => Self::default_effort(),
            }
        }

        pub fn from_value(v: usize) -> Self {
            Self { num_restarts: v.clamp(1, 20000), job_shop_iters: 100000, flow_shop_iters: 20000, hybrid_flow_shop_iters: 5000, fjsp_medium_iters: 5000, fjsp_high_iters: 5000 }
        }

        pub fn with_job_shop_iters(mut self, v: usize) -> Self {
            self.job_shop_iters = v.clamp(100, 200000);
            self
        }

        pub fn with_flow_shop_iters(mut self, v: usize) -> Self {
            self.flow_shop_iters = v.clamp(100, 50000);
            self
        }

        pub fn with_hybrid_flow_shop_iters(mut self, v: usize) -> Self {
            self.hybrid_flow_shop_iters = v.clamp(100, 50000);
            self
        }

        pub fn with_fjsp_medium_iters(mut self, v: usize) -> Self {
            self.fjsp_medium_iters = v.clamp(100, 50000);
            self
        }

        pub fn with_fjsp_high_iters(mut self, v: usize) -> Self {
            self.fjsp_high_iters = v.clamp(100, 50000);
            self
        }
    }

}

pub mod preprocess {
    use anyhow::{anyhow, Result};
    use tig_challenges::job_scheduling::*;
    use super::types::*;
    use super::infra::flow_makespan;

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

pub mod detect {
    use super::types::Pre;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum DetectedTrack {
        FlowShop,
        HybridFlowShop,
        JobShop,
        FjspMedium,
        FjspHigh,
    }

    pub fn detect_track(pre: &Pre) -> DetectedTrack {
        if pre.chaotic_like {
            DetectedTrack::FjspHigh
        } else if pre.flow_like > 0.82 && pre.jobshopness < 0.38 && pre.high_flex < 0.3 {
            DetectedTrack::FlowShop
        } else if pre.flow_like > 0.45 && pre.jobshopness < 0.55 && pre.flex_avg > 2.0 && pre.flex_avg < 4.0 && pre.high_flex < 0.1 {
            DetectedTrack::HybridFlowShop
        } else if pre.jobshopness > 0.5 && pre.high_flex < 0.3 && pre.flow_like > 0.35 {
            DetectedTrack::JobShop
        } else if pre.high_flex > 0.4 && pre.jobshopness > 0.4 {
            DetectedTrack::FjspMedium
        } else {
            DetectedTrack::FjspMedium
        }
    }

}

pub mod infra {
    use anyhow::{anyhow, Result};
    use rand::{rngs::SmallRng, Rng, SeedableRng, seq::SliceRandom};
    use tig_challenges::job_scheduling::*;
    use super::types::*;

    #[inline]
    pub fn pt_from_op(op: &OpInfo, machine: usize) -> Option<u32> {
        for &(m, pt) in &op.machines {
            if m == machine {
                return Some(pt);
            }
        }
        None
    }

    #[inline]
    pub fn push_top_k(top: &mut Vec<Cand>, c: Cand, k: usize) {
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
    pub fn push_top_k_raw(top: &mut Vec<RawCand>, c: RawCand, k: usize) {
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
    pub fn best_second_and_counts(time: u32, machine_avail: &[u32], op: &OpInfo) -> (u32, u32, usize, usize) {
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
    pub fn choose_from_top_weighted(rng: &mut SmallRng, top: &[Cand]) -> Cand {
        if top.len() <= 1 {
            return top[0];
        }
        let min_s = top.last().unwrap().score;
        let n = top.len().min(8);
        let mut w: [f64; 8] = [0.0; 8];
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

    #[inline]
    pub fn push_top_k_move(top: &mut Vec<MoveCand>, c: MoveCand, k: usize) {
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
    pub fn best_two_by_pt(op: &OpInfo) -> [(usize, u32); 2] {
        let mut best_m = NONE_USIZE;
        let mut best_pt = INF;
        let mut second_m = NONE_USIZE;
        let mut second_pt = INF;

        for &(m, pt) in &op.machines {
            if pt < best_pt || (pt == best_pt && m < best_m) {
                second_m = best_m;
                second_pt = best_pt;
                best_m = m;
                best_pt = pt;
            } else if m != best_m && (pt < second_pt || (pt == second_pt && m < second_m)) {
                second_m = m;
                second_pt = pt;
            }
        }

        [(best_m, best_pt), (second_m, second_pt)]
    }

    #[inline]
    pub fn push_top_solutions(top: &mut Vec<(Solution, u32)>, sol: &Solution, mk: u32, cap: usize) {
        let pos = top.binary_search_by_key(&mk, |(_, m)| *m).unwrap_or_else(|e| e);
        top.insert(pos, (sol.clone(), mk));
        if top.len() > cap {
            top.truncate(cap);
        }
    }

    #[inline]
    pub fn flow_makespan(seq: &[usize], pt: &[Vec<u32>], comp: &mut [u32]) -> u32 {
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
    pub fn reentrant_makespan(seq: &[usize], route: &[usize], pt: &[Vec<u32>], mready: &mut [u32]) -> u32 {
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

    #[inline]
    pub fn slack_urgency(pre: &Pre, target_mk: Option<u32>, time: u32, product: usize, op_idx: usize) -> f64 {
        let Some(tgt) = target_mk else { return 0.0 };
        let lb = (time as u64).saturating_add(pre.product_suf_min[product][op_idx] as u64);
        let slack = (tgt as i64) - (lb as i64);

        let scale = (0.70 * pre.avg_op_min).max(1.0);
        let pos = (slack.max(0) as f64) / scale;
        let neg = ((-slack).max(0) as f64) / scale;

        (1.0 / (1.0 + pos)).clamp(0.0, 1.0) + (0.35 * neg).min(3.0)
    }

    #[inline]
    pub fn route_pref_bonus_lite(rp: Option<&RoutePrefLite>, product: usize, op_idx: usize, machine: usize) -> f64 {
        let Some(rp) = rp else { return 0.0 };
        if product >= rp.len() || op_idx >= rp[product].len() {
            return 0.0;
        }
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

    #[inline]
    pub fn score_candidate(
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
        machine_penalty: f64,
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
        let rem_avg_n = rem_avg / pre.max_job_avg_work.max(1e-9);
        let bn_n = rem_bn / pre.max_job_bn.max(1e-9);
        let ops_n = (ops_rem as f64) / (pre.max_ops as f64).max(1.0);

        let load_n = dynamic_load / pre.avg_machine_load.max(1e-9);
        let scar_n = pre.machine_scarcity[machine] / pre.avg_machine_scarcity.max(1e-9);

        let end_n = (best_end as f64) / pre.time_scale.max(1.0);
        let proc_n = (pt as f64) / pre.avg_op_min.max(1.0);

        let regret = if second_end >= INF {
            pre.avg_op_min * 2.6
        } else {
            (second_end - best_end) as f64
        };
        let reg_n = (regret / pre.avg_op_min.max(1.0)).clamp(0.0, 6.0);

        let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
        let density_n = ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0);

        let next_min = pre.product_next_min[product][op_idx] as f64;
        let next_min_n = next_min / pre.horizon.max(1.0);
        let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
        let next_term = 0.55 * next_min_n + 0.45 * next_flex_inv;

        let js = pre.jobshopness;
        let fl = 1.0 - js;

        let avg_flex_inv = 1.0 / pre.flex_avg.max(1.0);
        let scarce_match = scar_n * (flex_inv - avg_flex_inv);

        let mpen = machine_penalty.clamp(0.0, 1.0);
        let mpen_gain = 1.0 + 0.85 * pre.high_flex;

        let flow_term = pre.flow_w * pre.job_flow_pref[job] * (0.65 + 0.70 * (1.0 - progress));

        let slack_u = slack_urgency(pre, target_mk, time, product, op_idx);
        let slack_w = pre.slack_base * (0.25 + 0.75 * progress);

        let pop_pen = if pre.chaotic_like && op.flex >= 2 {
            let pop = pre.machine_best_pop[machine];
            (0.07 + 0.15 * (1.0 - progress)).clamp(0.05, 0.24) * pop * pre.flex_factor
        } else {
            0.0
        };

        let route_gain = (0.70 + 0.80 * (1.0 - progress)).clamp(0.70, 1.40);
        let route_term = if route_w > 0.0 && op.flex >= 2 {
            route_w * route_gain * route_pref_bonus_lite(route_pref, product, op_idx, machine)
        } else {
            0.0
        };

        match rule {
            Rule::CriticalPath => {
                (1.03 * rem_min_n)
                    + (0.10 * ops_n)
                    + (0.24 * scarcity_urg)
                    + (0.20 * pre.flex_factor) * flex_inv
                    - (0.70 * end_n)
                    - pop_pen
                    + (0.45 * job_bias)
                    + flow_term
                    + route_term
                    + jitter
            }
            Rule::MostWork => {
                (1.00 * rem_avg_n)
                    + (0.12 * ops_n)
                    + (0.18 * scarcity_urg)
                    + (0.15 * pre.flex_factor) * flex_inv
                    - (0.62 * end_n)
                    - pop_pen
                    + (0.45 * job_bias)
                    + flow_term
                    + route_term
                    + jitter
            }
            Rule::LeastFlex => {
                (1.00 * flex_inv)
                    + (0.28 * rem_min_n)
                    + (0.22 * scarcity_urg)
                    - (0.55 * end_n)
                    - pop_pen
                    + (0.35 * job_bias)
                    + flow_term
                    + route_term
                    + jitter
            }
            Rule::ShortestProc => {
                (-1.00 * proc_n)
                    + (0.25 * rem_min_n)
                    + (0.12 * scarcity_urg)
                    - (0.20 * end_n)
                    - pop_pen
                    + (0.25 * job_bias)
                    + flow_term
                    + route_term
                    + jitter
            }
            Rule::Regret => {
                (1.05 * reg_n)
                    + (0.55 * rem_min_n)
                    + (0.22 * scarcity_urg)
                    - (0.68 * end_n)
                    - pop_pen
                    + (0.35 * job_bias)
                    + flow_term
                    + route_term
                    + jitter
            }
            Rule::EndTight => {
                let end_w = 1.10 + 1.00 * progress + 0.35 * pre.high_flex;
                let cp_w = 1.15 + 0.30 * js;
                let reg_w = (0.55 + 0.20 * (1.0 - progress)) * (0.85 + 0.60 * js);
                let mpen_w = (0.10 + 0.45 * pre.high_flex) * pre.flex_factor;

                (cp_w * rem_min_n)
                    + 0.12 * rem_avg_n
                    + 0.08 * ops_n
                    + 0.18 * scarcity_urg
                    + (0.30 * pre.flex_factor) * flex_inv
                    + (0.20 * pre.flex_factor) * scarce_match
                    + (reg_w * pre.flex_factor) * reg_n
                    + (0.10 + 0.35 * js) * next_term
                    + (slack_w * (0.70 + 0.40 * js)) * slack_u
                    - end_w * end_n
                    - 0.22 * proc_n
                    - pop_pen
                    - (mpen_gain * mpen_w) * mpen
                    + 0.55 * job_bias
                    + flow_term
                    + route_term
                    + jitter
            }
            Rule::BnHeavy => {
                let bn_w = (0.90 + 0.55 * js) * pre.bn_focus;
                let end_w = 0.65 + 0.70 * progress;
                let reg_w = (0.60 + 0.25 * (1.0 - progress)) * (0.85 + 0.35 * js);
                let load_w = if pre.hi_flex { -0.35 } else { 0.55 + 0.25 * js };
                let mpen_w = (0.12 + 0.30 * js) * pre.flex_factor * (0.95 + 0.65 * pre.high_flex);

                (0.95 * rem_min_n)
                    + (0.30 * rem_avg_n)
                    + (bn_w * bn_n)
                    + (0.22 * density_n)
                    + (0.10 * ops_n)
                    + (0.65 * pre.flex_factor) * flex_inv
                    + (0.35 * pre.flex_factor) * scarce_match
                    + load_w * pre.flex_factor * load_n
                    + (reg_w * pre.flex_factor) * reg_n
                    + 0.18 * scarcity_urg
                    + (0.20 + 0.55 * js) * next_term
                    + (slack_w * (0.45 + 0.55 * js)) * slack_u
                    - end_w * end_n
                    - 0.18 * proc_n
                    - pop_pen
                    - (mpen_gain * mpen_w) * mpen
                    + 0.60 * job_bias
                    + flow_term
                    + route_term
                    + jitter
            }
            Rule::Adaptive => {
                let end_w = (0.90 * fl + 0.72 * js) + (0.62 + 0.12 * fl) * progress + 0.18 * pre.high_flex;
                let reg_w = (0.50 * fl + 0.78 * js) + 0.18 * (1.0 - progress);
                let bn_w = ((0.45 + 0.40 * js) + 0.25 * (1.0 - progress)) * pre.bn_focus;

                let load_sign = if pre.hi_flex { -1.0 } else { 1.0 };
                let load_w = load_sign * (0.45 * fl + 0.75 * js) * pre.flex_factor;

                let density_w = 0.08 * fl + 0.20 * js;
                let next_w = 0.18 * fl + 0.60 * js;

                let mpen_w = (0.08 * fl + 0.28 * js) * pre.flex_factor * (1.0 + 0.85 * pre.high_flex);

                (1.05 * rem_min_n)
                    + (0.48 * rem_avg_n)
                    + (bn_w * bn_n)
                    + density_w * density_n
                    + (0.08 * ops_n)
                    + (0.62 * pre.flex_factor) * flex_inv
                    + (0.55 * pre.flex_factor) * scarce_match
                    + load_w * load_n
                    + (reg_w * pre.flex_factor) * reg_n
                    + 0.20 * pre.flex_factor * scarcity_urg
                    + next_w * next_term
                    + (slack_w * (0.55 * fl + 0.85 * js)) * slack_u
                    - end_w * end_n
                    - (0.18 * fl + 0.12 * js) * proc_n
                    - pop_pen
                    - (mpen_gain * mpen_w) * mpen
                    + (0.62 + 0.06 * js) * job_bias
                    + flow_term
                    + route_term
                    + jitter
            }
            Rule::FlexBalance => {
                let end_w = (0.85 + 0.70 * progress + 0.15 * js).clamp(0.85, 1.75);
                let cp_w = (1.00 + 0.30 * js + 0.15 * (1.0 - progress)).clamp(0.95, 1.45);
                let load_w = (0.55 + 0.35 * pre.high_flex).clamp(0.55, 0.95) * pre.flex_factor;
                let mpen_w = (0.55 + 0.65 * pre.high_flex).clamp(0.55, 1.15);
                let reg_w = (0.35 + 0.25 * (1.0 - progress)).clamp(0.35, 0.70);

                (cp_w * rem_min_n)
                    + 0.55 * rem_avg_n
                    + 0.08 * ops_n
                    + 0.06 * density_n
                    + 0.08 * scarcity_urg
                    + 0.15 * next_term
                    + (0.70 * slack_w) * slack_u
                    - end_w * end_n
                    - 0.16 * proc_n
                    - pop_pen
                    - load_w * load_n
                    - (mpen_w * (1.0 + 0.85 * pre.high_flex)) * mpen
                    + (reg_w * pre.flex_factor) * reg_n
                    + (0.58 + 0.10 * pre.high_flex) * job_bias
                    + flow_term
                    + route_term
                    + jitter
            }
        }
    }

    pub fn construct_solution_conflict(
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

        let mut job_schedule: Vec<Vec<(usize, u32)>> = pre.job_ops_len.iter().map(|&len| Vec::with_capacity(len)).collect();

        let mut remaining_ops = pre.total_ops;
        let mut time = 0u32;

        let mut demand: Vec<u16> = vec![0u16; num_machines];
        let mut raw_by_machine: Vec<Vec<RawCand>> = (0..num_machines).map(|_| Vec::with_capacity(12)).collect();
        let mut idle_machines: Vec<usize> = Vec::with_capacity(num_machines);

        let chaotic_like = pre.chaotic_like;
        let mut machine_work: Vec<u64> = if chaotic_like { vec![0u64; num_machines] } else { vec![] };
        let mut sum_work: u64 = 0;

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

                // Phased k-decay: reduce randomness in late construction
                let ek = if k > 1 {
                    if chaotic_like {
                        // FjspHigh: aggressive decay (tested +3.6%)
                        if progress > 0.80 { 1usize }
                        else if progress > 0.60 { ((k + 1) / 2).max(1) }
                        else { k }
                    } else if pre.flex_avg > 1.5 {
                        // HFS/FjspMedium: conservative decay
                        if progress > 0.90 { 1usize }
                        else if progress > 0.75 { ((k + 1) / 2).max(1) }
                        else { k }
                    } else {
                        k
                    }
                } else {
                    k
                };

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

                    let (best_end, second_end, best_cnt_total, best_cnt_idle) = best_second_and_counts(time, &machine_avail, op);
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
                            RawCand {
                                job,
                                machine: m,
                                pt,
                                base_score: base,
                                rigidity,
                                reg_n: regn,
                            },
                            cap_per_machine,
                        );
                    }
                }

                let denom = (idle_machines.len() as f64).max(1.0);
                let (conflict_w, conflict_scale) = if chaotic_like {
                    let w = -(0.05 + 0.08 * (1.0 - progress)).clamp(0.04, 0.14);
                    let s = (0.95 + 0.20 * pre.flex_factor).clamp(0.90, 1.20);
                    (w, s)
                } else {
                    let w = (0.09 + 0.26 * pre.jobshopness + 0.11 * pre.high_flex + 0.16 * (1.0 - progress)).clamp(0.05, 0.45);
                    let s = (0.90 + 0.40 * pre.flex_factor).clamp(0.85, 1.75);
                    (w, s)
                };

                let (bal_w, avg_work) = if chaotic_like {
                    let aw = (sum_work as f64) / (num_machines as f64).max(1.0);
                    let bw = (0.030 + 0.070 * (1.0 - progress)).clamp(0.025, 0.11);
                    (bw, aw)
                } else {
                    (0.0, 0.0)
                };

                let mut best: Option<Cand> = None;
                let mut top: Vec<Cand> = if ek > 0 { Vec::with_capacity(ek) } else { Vec::new() };

                for &m in &idle_machines {
                    let dem = demand[m] as f64;
                    if dem <= 0.0 || raw_by_machine[m].is_empty() {
                        continue;
                    }
                    let dem_n = ((dem - 1.0) / denom).clamp(0.0, 2.5);

                    let bal_pen = if chaotic_like && bal_w > 0.0 {
                        let denomw = (avg_work + (pre.avg_op_min * 3.0).max(1.0)).max(1.0);
                        let r = (machine_work[m] as f64) / denomw;
                        let done_n = (r / (r + 1.0)).clamp(0.0, 1.0);
                        -bal_w * done_n
                    } else {
                        0.0
                    };

                    for rc in &raw_by_machine[m] {
                        let rig = rc.rigidity.clamp(0.0, 2.5);
                        let regc = rc.reg_n.clamp(0.0, 4.5);

                        let mut boost = conflict_w * conflict_scale * dem_n * (1.15 * rig + 0.85 * regc);
                        if chaotic_like {
                            boost = boost.max(-0.26);
                        }

                        let c = Cand {
                            job: rc.job,
                            machine: rc.machine,
                            pt: rc.pt,
                            score: rc.base_score + boost + bal_pen,
                        };

                        if ek == 0 {
                            if best.map_or(true, |bb| c.score > bb.score) {
                                best = Some(c);
                            }
                        } else {
                            push_top_k(&mut top, c, ek);
                        }
                    }
                }

                let chosen = if ek == 0 {
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

                let start_time = time;
                let end_time = start_time.saturating_add(pt);

                job_schedule[job].push((machine, start_time));
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
                    next_time = Some(next_time.map_or(t, |bestt| bestt.min(t)));
                }
            }
            for j in 0..num_jobs {
                let op_idx = job_next_op[j];
                if op_idx < pre.job_ops_len[j] && job_ready_time[j] > time {
                    let t = job_ready_time[j];
                    next_time = Some(next_time.map_or(t, |bestt| bestt.min(t)));
                }
            }
            time = next_time.ok_or_else(|| anyhow!("Stalled: no next event"))?;
        }

        let mk = machine_avail.into_iter().max().unwrap_or(0);
        Ok((Solution { job_schedule }, mk))
    }

    pub fn improve_reentrant_seq(seq: &mut Vec<usize>, route: &[usize], pt: &[Vec<u32>], num_machines: usize) {
        if seq.len() <= 2 || route.is_empty() {
            return;
        }
        let mut mready = vec![0u32; num_machines];

        for pass in 0..2usize {
            let indices: Vec<usize> = if pass == 0 { (0..seq.len()).collect() } else { (0..seq.len()).rev().collect() };
            let mut improved_any = false;

            for &i0 in &indices {
                if i0 >= seq.len() {
                    continue;
                }
                let cur = reentrant_makespan(seq, route, pt, &mut mready);
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
                if best_mk < cur {
                    improved_any = true;
                }
            }

            if !improved_any {
                break;
            }
        }
    }

    pub fn neh_reentrant_flow_solution(pre: &Pre, num_jobs: usize, num_machines: usize) -> Result<(Solution, u32)> {
        let route = pre.flow_route.as_ref().ok_or_else(|| anyhow!("NEH requested but no flow route"))?;
        let pt = pre.flow_pt_by_job.as_ref().ok_or_else(|| anyhow!("NEH requested but no flow pt"))?;
        let ops = route.len();
        if ops == 0 || pt.len() != num_jobs {
            return Err(anyhow!("Invalid flow data"));
        }

        let mut jobs: Vec<usize> = (0..num_jobs).collect();
        jobs.sort_unstable_by(|&a, &b| {
            let sa: u32 = pt[a].iter().copied().sum();
            let sb: u32 = pt[b].iter().copied().sum();
            sb.cmp(&sa).then_with(|| a.cmp(&b))
        });

        let mut seq: Vec<usize> = Vec::with_capacity(num_jobs);
        let mut tmp: Vec<usize> = Vec::with_capacity(num_jobs);
        let mut mready = vec![0u32; num_machines];

        for &j in &jobs {
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

        improve_reentrant_seq(&mut seq, route, pt, num_machines);

        let mut job_schedule: Vec<Vec<(usize, u32)>> = vec![Vec::with_capacity(ops); num_jobs];
        let mut machine_ready = vec![0u32; num_machines];

        for &j in &seq {
            let row = &pt[j];
            let mut prev_end = 0u32;
            for op_idx in 0..ops {
                let m = route[op_idx];
                let p = row[op_idx];
                let st = prev_end.max(machine_ready[m]);
                job_schedule[j].push((m, st));
                let end = st.saturating_add(p);
                machine_ready[m] = end;
                prev_end = end;
            }
        }

        let mk = machine_ready.into_iter().max().unwrap_or(0);
        Ok((Solution { job_schedule }, mk))
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
                if op_idx >= ol || m >= nm {
                    continue;
                }
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
                        second_c = best_c;
                        second_m = best_m;
                        best_c = c;
                        best_m = m;
                    } else if c > second_c && m != best_m {
                        second_c = c;
                        second_m = m;
                    }
                }

                let best_w = (((best_c as u32).saturating_mul(255)).saturating_add(denom_u32 / 2) / denom_u32).min(255) as u8;
                let second_w = (((second_c as u32).saturating_mul(255)).saturating_add(denom_u32 / 2) / denom_u32).min(255) as u8;

                v.push(OpRoute {
                    best_m: best_m.min(255) as u8,
                    best_w,
                    second_m: second_m.min(255) as u8,
                    second_w,
                });
            }

            rp.push(v);
        }

        Ok(rp)
    }

    #[inline]
    pub fn rule_idx(r: Rule) -> usize {
        match r {
            Rule::Adaptive => 0,
            Rule::BnHeavy => 1,
            Rule::EndTight => 2,
            Rule::CriticalPath => 3,
            Rule::MostWork => 4,
            Rule::LeastFlex => 5,
            Rule::Regret => 6,
            Rule::ShortestProc => 7,
            Rule::FlexBalance => 8,
        }
    }

    #[inline]
    pub fn sample_roulette(rng: &mut SmallRng, weights: &[f64]) -> usize {
        let mut sum = 0.0;
        for &w in weights {
            sum += w.max(0.0);
        }
        if !(sum > 0.0) {
            return rng.gen_range(0..weights.len());
        }
        let mut r = rng.gen::<f64>() * sum;
        for (i, &w) in weights.iter().enumerate() {
            r -= w.max(0.0);
            if r <= 0.0 {
                return i;
            }
        }
        weights.len().saturating_sub(1)
    }

    pub fn choose_rule_bandit(
        rng: &mut SmallRng,
        rules: &[Rule],
        rule_best: &[u32],
        rule_tries: &[u32],
        global_best: u32,
        margin: u32,
        stuck: usize,
        chaos_like: bool,
        late_phase: bool,
    ) -> Rule {
        if rules.is_empty() {
            return Rule::Adaptive;
        }

        let mut best_seen = global_best;
        for &mk in rule_best {
            if mk < best_seen {
                best_seen = mk;
            }
        }

        let scale = (margin as f64).max(1.0);
        let s = ((stuck as f64) / 140.0).clamp(0.0, 1.0);
        let explore_mix = (0.10 + 0.55 * s).clamp(0.10, 0.65);

        let mut w = vec![0.0f64; rules.len()];
        for (i, &r) in rules.iter().enumerate() {
            let mk = rule_best[rule_idx(r)];
            let t = rule_tries[rule_idx(r)].max(1) as f64;

            let delta = mk.saturating_sub(best_seen) as f64;
            let exploit = (-delta / scale).exp();

            let explore = (1.0 / t).sqrt();

            let mut ww = (1.0 - explore_mix) * exploit + explore_mix * explore;
            ww = ww.max(1e-6);

            if chaos_like {
                ww = ww.powf(0.70);
            } else if late_phase {
                ww = ww.powf(1.18);
            }

            w[i] = ww;
        }

        let idx = sample_roulette(rng, &w);
        rules[idx]
    }

    pub fn run_simple_greedy_baseline(challenge: &Challenge) -> Result<(Solution, u32)> {
        let num_jobs = challenge.num_jobs;
        let mut job_products = Vec::with_capacity(num_jobs);
        for (p, &cnt) in challenge.jobs_per_product.iter().enumerate() {
            for _ in 0..cnt {
                job_products.push(p);
            }
        }

        let job_ops_len: Vec<usize> = job_products.iter()
            .map(|&p| challenge.product_processing_times[p].len())
            .collect();

        let job_total_work: Vec<f64> = job_products.iter().map(|&p| {
            challenge.product_processing_times[p].iter()
                .map(|op| {
                    let avg: f64 = op.values().sum::<u32>() as f64 / op.len().max(1) as f64;
                    avg
                })
                .sum()
        }).collect();

        let rules = [GreedyRule::MostWork, GreedyRule::MostOps, GreedyRule::LeastFlex, GreedyRule::ShortestProc, GreedyRule::LongestProc];
        let mut best_mk = u32::MAX;
        let mut best_sol: Option<Solution> = None;

        for rule in rules {
            let (sol, mk) = run_greedy_rule(challenge, &job_products, &job_ops_len, &job_total_work, rule, None)?;
            if mk < best_mk {
                best_mk = mk;
                best_sol = Some(sol);
            }
        }

        let mut rng = SmallRng::from_seed(challenge.seed);
        for _ in 0..10 {
            let seed = rng.gen::<u64>();
            let rule = rules[rng.gen_range(0..rules.len())];
            let random_top_k = rng.gen_range(2..=5);
            let mut local_rng = SmallRng::seed_from_u64(seed);

            let (sol, mk) = run_greedy_rule(challenge, &job_products, &job_ops_len, &job_total_work, rule, Some((random_top_k, &mut local_rng)))?;
            if mk < best_mk {
                best_mk = mk;
                best_sol = Some(sol);
            }
        }

        Ok((best_sol.ok_or_else(|| anyhow!("No greedy solution"))?, best_mk))
    }

    pub fn run_greedy_rule(
        challenge: &Challenge,
        job_products: &[usize],
        job_ops_len: &[usize],
        job_total_work: &[f64],
        rule: GreedyRule,
        mut random_top_k: Option<(usize, &mut SmallRng)>,
    ) -> Result<(Solution, u32)> {
        let num_jobs = challenge.num_jobs;
        let num_machines = challenge.num_machines;

        let mut job_next_op = vec![0usize; num_jobs];
        let mut job_ready = vec![0u32; num_jobs];
        let mut machine_avail = vec![0u32; num_machines];
        let mut job_schedule: Vec<Vec<(usize, u32)>> = job_ops_len.iter()
            .map(|&len| Vec::with_capacity(len))
            .collect();
        let mut job_work_left = job_total_work.to_vec();

        let mut remaining = job_ops_len.iter().sum::<usize>();
        let mut time = 0u32;
        let eps = 1e-9;

        while remaining > 0 {
            let mut available_machines: Vec<usize> = (0..num_machines)
                .filter(|&m| machine_avail[m] <= time)
                .collect();
            available_machines.sort_unstable();
            if let Some((_, ref mut rng)) = random_top_k {
                available_machines.shuffle(*rng);
            }

            for &m in &available_machines {
                #[derive(Clone)]
                struct GCandidate {
                    job: usize,
                    priority: f64,
                    end: u32,
                    pt: u32,
                    flex: usize,
                }

                let mut candidates: Vec<GCandidate> = Vec::new();

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

                    let earliest = op_times.iter()
                        .map(|(&mm, &ppt)| time.max(machine_avail[mm]) + ppt)
                        .min().unwrap_or(u32::MAX);
                    let this_end = time.max(machine_avail[m]) + pt;
                    if this_end != earliest {
                        continue;
                    }

                    let flex = op_times.len();
                    let ops_left = job_ops_len[j] - job_next_op[j];
                    let priority = match rule {
                        GreedyRule::MostWork => job_work_left[j],
                        GreedyRule::MostOps => ops_left as f64,
                        GreedyRule::LeastFlex => -(flex as f64),
                        GreedyRule::ShortestProc => -(pt as f64),
                        GreedyRule::LongestProc => pt as f64,
                    };

                    candidates.push(GCandidate { job: j, priority, end: this_end, pt, flex });
                }

                if candidates.is_empty() {
                    continue;
                }

                let best_job = if let Some((top_k, ref mut rng)) = random_top_k {
                    candidates.sort_by(|a, b| {
                        if (b.priority - a.priority).abs() > eps {
                            b.priority.partial_cmp(&a.priority).unwrap()
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
                    candidates[rng.gen_range(0..top)].job
                } else {
                    let mut best: Option<GCandidate> = None;
                    for cand in candidates {
                        let better = if let Some(ref b) = best {
                            if (cand.priority - b.priority).abs() > eps {
                                cand.priority > b.priority
                            } else if cand.end != b.end {
                                cand.end < b.end
                            } else if cand.pt != b.pt {
                                cand.pt < b.pt
                            } else if cand.flex != b.flex {
                                cand.flex < b.flex
                            } else {
                                cand.job < b.job
                            }
                        } else {
                            true
                        };
                        if better {
                            best = Some(cand);
                        }
                    }
                    best.ok_or_else(|| anyhow!("No candidate"))?.job
                };

                let product = job_products[best_job];
                let op_idx = job_next_op[best_job];
                let op_times = &challenge.product_processing_times[product][op_idx];
                let pt = op_times[&m];
                let avg_pt = op_times.values().sum::<u32>() as f64 / op_times.len().max(1) as f64;

                let st = time.max(machine_avail[m]);
                let end = st + pt;

                job_schedule[best_job].push((m, st));
                job_next_op[best_job] += 1;
                job_ready[best_job] = end;
                machine_avail[m] = end;
                job_work_left[best_job] -= avg_pt;
                if job_work_left[best_job] < 0.0 {
                    job_work_left[best_job] = 0.0;
                }
                remaining -= 1;
            }

            if remaining == 0 {
                break;
            }

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
                return Err(anyhow!("Greedy baseline stuck"));
            }
            time = next;
        }

        let mk = job_ready.iter().copied().max().unwrap_or(0);
        Ok((Solution { job_schedule }, mk))
    }

    /// Evaluate the makespan of a solution (returns None if invalid/cyclic).
    pub fn eval_solution_makespan(pre: &Pre, challenge: &Challenge, sol: &Solution) -> Option<u32> {
        let ds = build_disj_from_solution(pre, challenge, sol).ok()?;
        let mut buf = EvalBuf::new(ds.n);
        eval_disj(&ds, &mut buf).map(|(mk, _)| mk)
    }

    pub fn build_disj_from_solution(pre: &Pre, challenge: &Challenge, sol: &Solution) -> Result<DisjSchedule> {
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
                let pt = pt_from_op(op, m).ok_or_else(|| anyhow!("Invalid solution: pt missing"))?;
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
            let mut seq = Vec::with_capacity(per_machine[m].len());
            for &(_st, id) in &per_machine[m] {
                seq.push(id);
            }
            machine_seq.push(seq);
        }

        let mut job_succ = vec![NONE_USIZE; n];
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

    pub fn eval_disj(ds: &DisjSchedule, buf: &mut EvalBuf) -> Option<(u32, usize)> {
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

    #[inline]
    pub fn apply_insert(seq: &mut Vec<usize>, from: usize, to_after_removal: usize) -> usize {
        if seq.is_empty() || from >= seq.len() {
            return from.min(seq.len().saturating_sub(1));
        }
        let x = seq.remove(from);
        let t = to_after_removal.min(seq.len());
        seq.insert(t, x);
        t
    }

    #[inline]
    pub fn apply_swap(seq: &mut [usize], i: usize) -> bool {
        if i + 1 >= seq.len() {
            return false;
        }
        seq.swap(i, i + 1);
        true
    }

    #[inline]
    pub fn find_insert_pos_by_start(seq: &[usize], start: &[u32], desired_start: u32) -> usize {
        for (i, &id) in seq.iter().enumerate() {
            if start[id] >= desired_start {
                return i;
            }
        }
        seq.len()
    }

    #[inline]
    pub fn apply_reroute(
        ds: &mut DisjSchedule,
        m_from: usize,
        idx_from: usize,
        m_to: usize,
        idx_to: usize,
        new_pt: u32,
    ) -> Option<(usize, u32, usize)> {
        if m_from >= ds.num_machines || m_to >= ds.num_machines {
            return None;
        }
        if idx_from >= ds.machine_seq[m_from].len() {
            return None;
        }
        let node = ds.machine_seq[m_from].remove(idx_from);
        let old_pt = ds.node_pt[node];
        ds.node_machine[node] = m_to;
        ds.node_pt[node] = new_pt;
        let ins = idx_to.min(ds.machine_seq[m_to].len());
        ds.machine_seq[m_to].insert(ins, node);
        Some((node, old_pt, ins))
    }

    #[inline]
    pub fn undo_reroute(
        ds: &mut DisjSchedule,
        m_from: usize,
        idx_from: usize,
        m_to: usize,
        ins_idx: usize,
        node: usize,
        old_pt: u32,
    ) -> bool {
        if m_from >= ds.num_machines || m_to >= ds.num_machines {
            return false;
        }
        if ins_idx >= ds.machine_seq[m_to].len() {
            return false;
        }
        let x = ds.machine_seq[m_to].remove(ins_idx);
        if x != node {
            let len_now = ds.machine_seq[m_to].len();
            let back_pos = ins_idx.min(len_now);
            ds.machine_seq[m_to].insert(back_pos, x);
            return false;
        }
        let ins_back = idx_from.min(ds.machine_seq[m_from].len());
        ds.machine_seq[m_from].insert(ins_back, node);
        ds.node_machine[node] = m_from;
        ds.node_pt[node] = old_pt;
        true
    }

    pub fn disj_to_solution(pre: &Pre, ds: &DisjSchedule, start: &[u32]) -> Result<Solution> {
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

    pub fn descent_phase(
        ds: &mut DisjSchedule,
        buf: &mut EvalBuf,
        crit: &mut Vec<bool>,
        pre: &Pre,
        cur_eval: &mut (u32, usize),
        max_iters: usize,
        top_cands: usize,
    ) -> bool {
        let mut cur_mk = cur_eval.0;
        let mut improved = false;

        for _iter in 0..max_iters {
            crit.fill(false);
            let mut u = cur_eval.1;
            while u != NONE_USIZE {
                crit[u] = true;
                u = buf.best_pred[u];
            }

            let mut cands: Vec<MoveCand> = Vec::with_capacity(top_cands.min(64));

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

                            {
                                let from = bstart;
                                let to_after = bstart + sh;
                                if from < seq.len() && to_after <= seq.len() {
                                    let tgt_idx = (bstart + sh).min(seq.len() - 1);
                                    let score = buf.start[seq[tgt_idx]];
                                    push_top_k_move(
                                        &mut cands,
                                        MoveCand { kind: 0, m_from: m, from, m_to: m, to: to_after, new_pt: 0, score },
                                        top_cands,
                                    );
                                }
                            }
                            {
                                let from = bend;
                                let to_after = bend - sh;
                                let score = buf.start[seq[bend]];
                                push_top_k_move(
                                    &mut cands,
                                    MoveCand { kind: 0, m_from: m, from, m_to: m, to: to_after, new_pt: 0, score },
                                    top_cands,
                                );
                            }
                        }

                        {
                            if bstart > 0 {
                                let score = buf.start[seq[bstart]];
                                push_top_k_move(
                                    &mut cands,
                                    MoveCand { kind: 2, m_from: m, from: bstart - 1, m_to: m, to: 0, new_pt: 0, score },
                                    top_cands,
                                );
                            }
                            if bend + 1 < seq.len() {
                                let score = buf.start[seq[bend]];
                                push_top_k_move(
                                    &mut cands,
                                    MoveCand { kind: 2, m_from: m, from: bend, m_to: m, to: 0, new_pt: 0, score },
                                    top_cands,
                                );
                            }
                            if bstart + 1 <= bend {
                                let score = buf.start[seq[bstart + 1]];
                                push_top_k_move(
                                    &mut cands,
                                    MoveCand { kind: 2, m_from: m, from: bstart, m_to: m, to: 0, new_pt: 0, score },
                                    top_cands,
                                );
                                if bend >= 1 && bend - 1 >= bstart {
                                    let score2 = buf.start[seq[bend]];
                                    push_top_k_move(
                                        &mut cands,
                                        MoveCand { kind: 2, m_from: m, from: bend - 1, m_to: m, to: 0, new_pt: 0, score: score2 },
                                        top_cands,
                                    );
                                }
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

                            let job = ds.node_job[node];
                            let op_idx = ds.node_op[node];
                            let product = pre.job_products[job];
                            let op = &pre.product_ops[product][op_idx];

                            if op.flex < 2 || op.machines.len() < 2 {
                                continue;
                            }

                            let old_m = ds.node_machine[node];
                            let old_pt = ds.node_pt[node];
                            let w_from = pre.machine_weight[old_m].max(1e-9);

                            let best2 = best_two_by_pt(op);
                            for &(m_to, new_pt) in &best2 {
                                if m_to == NONE_USIZE || m_to >= ds.num_machines || m_to == old_m || new_pt >= INF {
                                    continue;
                                }
                                let w_to = pre.machine_weight[m_to].max(1e-9);

                                if !(new_pt + 1 < old_pt || w_to < w_from * 0.90) {
                                    continue;
                                }

                                let desired = buf.start[node];
                                let pos0 = find_insert_pos_by_start(&ds.machine_seq[m_to], &buf.start, desired);
                                for pos in [pos0, pos0.saturating_add(1)] {
                                    if pos > ds.machine_seq[m_to].len() {
                                        continue;
                                    }

                                    let diffw = ((w_from - w_to).max(0.0) * pre.avg_op_min).max(0.0) as u32;
                                    let difpt = old_pt.saturating_sub(new_pt);
                                    let score = desired
                                        .saturating_add(old_pt)
                                        .saturating_add(diffw)
                                        .saturating_add(difpt.saturating_mul(2));

                                    push_top_k_move(
                                        &mut cands,
                                        MoveCand { kind: 1, m_from: old_m, from: idx, m_to, to: pos, new_pt, score },
                                        top_cands,
                                    );
                                }
                            }
                        }
                    }

                    i = bend + 1;
                }
            }

            if cands.is_empty() {
                break;
            }

            let mut best_cand: Option<MoveCand> = None;
            let mut best_mk = cur_mk;

            for cand in &cands {
                if cand.kind == 0 {
                    let m = cand.m_from;
                    if m >= ds.num_machines || cand.from >= ds.machine_seq[m].len() {
                        continue;
                    }
                    let new_idx = apply_insert(&mut ds.machine_seq[m], cand.from, cand.to);
                    if let Some((mk2, _)) = eval_disj(ds, buf) {
                        if mk2 < best_mk {
                            best_mk = mk2;
                            best_cand = Some(*cand);
                        }
                    }
                    let _ = apply_insert(&mut ds.machine_seq[m], new_idx, cand.from);
                } else if cand.kind == 2 {
                    let m = cand.m_from;
                    if m >= ds.num_machines || cand.from + 1 >= ds.machine_seq[m].len() {
                        continue;
                    }
                    if !apply_swap(&mut ds.machine_seq[m], cand.from) {
                        continue;
                    }
                    if let Some((mk2, _)) = eval_disj(ds, buf) {
                        if mk2 < best_mk {
                            best_mk = mk2;
                            best_cand = Some(*cand);
                        }
                    }
                    let _ = apply_swap(&mut ds.machine_seq[m], cand.from);
                } else {
                    let m_from = cand.m_from;
                    let m_to = cand.m_to;
                    if m_from >= ds.num_machines || m_to >= ds.num_machines {
                        continue;
                    }
                    if cand.from >= ds.machine_seq[m_from].len() {
                        continue;
                    }
                    let node = ds.machine_seq[m_from][cand.from];
                    if ds.node_machine[node] != m_from {
                        continue;
                    }

                    let applied = apply_reroute(ds, m_from, cand.from, m_to, cand.to, cand.new_pt);
                    if let Some((node2, old_pt, ins_idx)) = applied {
                        if let Some((mk2, _)) = eval_disj(ds, buf) {
                            if mk2 < best_mk {
                                best_mk = mk2;
                                best_cand = Some(*cand);
                            }
                        }
                        let _ = undo_reroute(ds, m_from, cand.from, m_to, ins_idx, node2, old_pt);
                    }
                }
            }

            let Some(bc) = best_cand else { break };

            let mut accepted = false;

            if bc.kind == 0 {
                let m = bc.m_from;
                let new_idx = apply_insert(&mut ds.machine_seq[m], bc.from, bc.to);
                if let Some(next_eval) = eval_disj(ds, buf) {
                    if next_eval.0 < cur_mk {
                        *cur_eval = next_eval;
                        cur_mk = cur_eval.0;
                        improved = true;
                        accepted = true;
                    } else {
                        let _ = apply_insert(&mut ds.machine_seq[m], new_idx, bc.from);
                    }
                } else {
                    let _ = apply_insert(&mut ds.machine_seq[m], new_idx, bc.from);
                }
            } else if bc.kind == 2 {
                let m = bc.m_from;
                if m < ds.num_machines && bc.from + 1 < ds.machine_seq[m].len() {
                    if apply_swap(&mut ds.machine_seq[m], bc.from) {
                        if let Some(next_eval) = eval_disj(ds, buf) {
                            if next_eval.0 < cur_mk {
                                *cur_eval = next_eval;
                                cur_mk = cur_eval.0;
                                improved = true;
                                accepted = true;
                            } else {
                                let _ = apply_swap(&mut ds.machine_seq[m], bc.from);
                            }
                        } else {
                            let _ = apply_swap(&mut ds.machine_seq[m], bc.from);
                        }
                    }
                }
            } else {
                let applied = apply_reroute(ds, bc.m_from, bc.from, bc.m_to, bc.to, bc.new_pt);
                if let Some((node2, old_pt, ins_idx)) = applied {
                    if let Some(next_eval) = eval_disj(ds, buf) {
                        if next_eval.0 < cur_mk {
                            *cur_eval = next_eval;
                            cur_mk = cur_eval.0;
                            improved = true;
                            accepted = true;
                        } else {
                            let _ = undo_reroute(ds, bc.m_from, bc.from, bc.m_to, ins_idx, node2, old_pt);
                        }
                    } else {
                        let _ = undo_reroute(ds, bc.m_from, bc.from, bc.m_to, ins_idx, node2, old_pt);
                    }
                }
            }

            if !accepted {
                break;
            }
        }

        improved
    }

    pub fn critical_block_move_local_search(
        pre: &Pre,
        challenge: &Challenge,
        base_sol: &Solution,
        max_iters: usize,
        top_cands: usize,
    ) -> Result<Option<(Solution, u32)>> {
        critical_block_move_local_search_ex(pre, challenge, base_sol, max_iters, top_cands, 3)
    }

    pub fn critical_block_move_local_search_ex(
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

        let mut sol_hash: u64 = 0;
        for m in 0..ds.num_machines.min(8) {
            if !ds.machine_seq[m].is_empty() {
                let first_node = ds.machine_seq[m][0];
                sol_hash ^= (first_node as u64).wrapping_mul(0xD2B54A6B68A5);
                sol_hash = sol_hash.rotate_left(7);
            }
        }

        let mut pseed: u64 = (challenge.seed[0] as u64)
            .wrapping_mul(0x9E3779B97F4A7C15)
            ^ (initial_mk as u64).wrapping_shl(16)
            ^ (ds.n as u64)
            ^ sol_hash;

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

            let num_swaps = 2 + (_cycle / 2);
            for _ in 0..num_swaps {
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

    pub fn machine_reassign_local_search(
        challenge: &Challenge,
        pre: &Pre,
        base_sol: &Solution,
        max_iters: usize,
    ) -> Result<Option<(Solution, u32)>> {
        let num_jobs = challenge.num_jobs;
        let num_machines = challenge.num_machines;

        let mut sol = base_sol.clone();
        let mut best_mk = evaluate_solution_makespan(challenge, &sol)?;
        let initial_mk = best_mk;
        let mut best_sol = sol.clone();

        let mut machine_load = vec![0u32; num_machines];
        for (job_idx, sched) in sol.job_schedule.iter().enumerate() {
            let product = pre.job_products[job_idx];
            for (op_idx, &(m, _start)) in sched.iter().enumerate() {
                let pt = challenge.product_processing_times[product][op_idx]
                    .get(&m)
                    .copied()
                    .unwrap_or(0);
                machine_load[m] = machine_load[m].saturating_add(pt);
            }
        }

        let mut pseed: u64 = (challenge.seed[0] as u64)
            .wrapping_mul(0x9E3779B97F4A7C15)
            ^ (initial_mk as u64).wrapping_shl(16)
            ^ (num_jobs as u64).wrapping_mul(0x517CC1B727220A95);

        let mut no_improve = 0usize;
        let max_no_improve = max_iters / 3;

        for _iter in 0..max_iters {
            if no_improve > max_no_improve {
                sol = best_sol.clone();
                machine_load.fill(0);
                for (job_idx, sched) in sol.job_schedule.iter().enumerate() {
                    let product = pre.job_products[job_idx];
                    for (op_idx, &(m, _)) in sched.iter().enumerate() {
                        let pt = challenge.product_processing_times[product][op_idx]
                            .get(&m)
                            .copied()
                            .unwrap_or(0);
                        machine_load[m] = machine_load[m].saturating_add(pt);
                    }
                }
                no_improve = 0;
            }

            pseed ^= pseed.wrapping_shl(13);
            pseed ^= pseed.wrapping_shr(7);
            pseed ^= pseed.wrapping_shl(17);
            let job_idx = (pseed as usize) % num_jobs;

            let product = pre.job_products[job_idx];
            let ops = &challenge.product_processing_times[product];

            pseed ^= pseed.wrapping_shl(13);
            pseed ^= pseed.wrapping_shr(7);
            pseed ^= pseed.wrapping_shl(17);
            let op_idx = (pseed as usize) % ops.len();

            let eligible = &ops[op_idx];
            if eligible.len() <= 1 {
                continue;
            }

            let old_m = sol.job_schedule[job_idx][op_idx].0;
            let old_pt = eligible.get(&old_m).copied().unwrap_or(0);

            let mut best_new_m = old_m;
            let mut best_delta = 0i64;

            for (&new_m, &new_pt) in eligible.iter() {
                if new_m == old_m {
                    continue;
                }

                let old_load_old_m = machine_load[old_m];
                let old_load_new_m = machine_load[new_m];
                let new_load_old_m = old_load_old_m.saturating_sub(old_pt);
                let new_load_new_m = old_load_new_m.saturating_add(new_pt);

                let old_max = old_load_old_m.max(old_load_new_m) as i64;
                let new_max = new_load_old_m.max(new_load_new_m) as i64;
                let delta = old_max - new_max;

                if delta > best_delta || (delta == best_delta && new_pt < old_pt) {
                    best_delta = delta;
                    best_new_m = new_m;
                }
            }

            if best_new_m != old_m {
                let new_pt = eligible.get(&best_new_m).copied().unwrap_or(0);
                machine_load[old_m] = machine_load[old_m].saturating_sub(old_pt);
                machine_load[best_new_m] = machine_load[best_new_m].saturating_add(new_pt);
                sol.job_schedule[job_idx][op_idx].0 = best_new_m;
            }

            let new_sol = reschedule_solution(challenge, pre, &sol)?;
            let new_mk = evaluate_solution_makespan(challenge, &new_sol)?;

            if new_mk < best_mk {
                best_mk = new_mk;
                best_sol = new_sol.clone();
                sol = new_sol;
                no_improve = 0;
            } else {
                no_improve += 1;
            }
        }

        if best_mk >= initial_mk {
            return Ok(None);
        }

        Ok(Some((best_sol, best_mk)))
    }

    fn reschedule_solution(
        challenge: &Challenge,
        pre: &Pre,
        sol: &Solution,
    ) -> Result<Solution> {
        let num_machines = challenge.num_machines;
        let mut machine_avail = vec![0u32; num_machines];
        let mut new_schedule: Vec<Vec<(usize, u32)>> = Vec::with_capacity(sol.job_schedule.len());

        for (job_idx, sched) in sol.job_schedule.iter().enumerate() {
            let product = pre.job_products[job_idx];
            let mut job_time = 0u32;
            let mut new_job_sched: Vec<(usize, u32)> = Vec::with_capacity(sched.len());

            for (op_idx, &(m, _old_start)) in sched.iter().enumerate() {
                let pt = challenge.product_processing_times[product][op_idx]
                    .get(&m)
                    .copied()
                    .unwrap_or(1);
                let start = job_time.max(machine_avail[m]);
                let end = start.saturating_add(pt);
                machine_avail[m] = end;
                job_time = end;
                new_job_sched.push((m, start));
            }
            new_schedule.push(new_job_sched);
        }

        Ok(Solution { job_schedule: new_schedule })
    }

    fn evaluate_solution_makespan(challenge: &Challenge, sol: &Solution) -> Result<u32> {
        challenge.evaluate_makespan(sol)
    }

    pub fn greedy_reassign_pass(
        pre: &Pre,
        challenge: &Challenge,
        base_sol: &Solution,
    ) -> Result<Option<(Solution, u32)>> {
        let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
        let mut buf = EvalBuf::new(ds.n);
        let n = ds.n;

        let Some((mut current_mk, _)) = eval_disj(&ds, &mut buf) else {
            return Ok(None);
        };
        let initial_mk = current_mk;

        let mut improved = true;
        let mut passes = 0;
        let max_passes = 5;

        while improved && passes < max_passes {
            improved = false;
            passes += 1;

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

                let mut best_m = cur_machine;
                let mut best_pt = cur_pt;
                let mut best_mk = current_mk;

                for &(new_m, new_pt) in &op_info.machines {
                    if new_m == cur_machine {
                        continue;
                    }

                    let old_pos = ds.machine_seq[cur_machine].iter().position(|&x| x == node);
                    if old_pos.is_none() {
                        continue;
                    }
                    let old_pos = old_pos.unwrap();

                    ds.machine_seq[cur_machine].remove(old_pos);
                    ds.machine_seq[new_m].push(node);
                    ds.node_machine[node] = new_m;
                    ds.node_pt[node] = new_pt;

                    if let Some((test_mk, _)) = eval_disj(&ds, &mut buf) {
                        if test_mk < best_mk {
                            best_mk = test_mk;
                            best_m = new_m;
                            best_pt = new_pt;
                        }
                    }

                    ds.machine_seq[new_m].pop();
                    ds.machine_seq[cur_machine].insert(old_pos, node);
                    ds.node_machine[node] = cur_machine;
                    ds.node_pt[node] = cur_pt;
                }

                if best_m != cur_machine {
                    let old_pos = ds.machine_seq[cur_machine].iter().position(|&x| x == node).unwrap();
                    ds.machine_seq[cur_machine].remove(old_pos);
                    ds.machine_seq[best_m].push(node);
                    ds.node_machine[node] = best_m;
                    ds.node_pt[node] = best_pt;
                    current_mk = best_mk;
                    improved = true;
                }
            }
        }

        if current_mk >= initial_mk {
            return Ok(None);
        }

        let Some((_, _)) = eval_disj(&ds, &mut buf) else {
            return Ok(None);
        };
        let sol = disj_to_solution(pre, &ds, &buf.start)?;
        Ok(Some((sol, current_mk)))
    }

    /// LNS on disjunctive graph: segment reversal, node insertion, pairwise swap
    pub fn lns_disj_post(
        pre: &Pre,
        challenge: &Challenge,
        base_sol: &Solution,
        iters: usize,
    ) -> Result<Option<(Solution, u32)>> {
        let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
        let mut buf = EvalBuf::new(ds.n);
        let Some(init_eval) = eval_disj(&ds, &mut buf) else { return Ok(None) };
        let mut best_mk = init_eval.0;
        let mut best_ds = ds.clone();
        let mut cur_eval = init_eval;

        let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15)
            ^ (best_mk as u64).wrapping_shl(16) ^ (ds.n as u64);

        for _ in 0..iters {
            pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);

            let machines_with_jobs: Vec<usize> = (0..ds.num_machines)
                .filter(|&m| ds.machine_seq[m].len() >= 2)
                .collect();
            if machines_with_jobs.is_empty() { break; }
            let m = machines_with_jobs[(pseed as usize) % machines_with_jobs.len()];
            let seq_len = ds.machine_seq[m].len();

            pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
            let move_type = (pseed as usize) % 3;

            match move_type {
                0 => {
                    if seq_len < 2 { continue; }
                    let max_seg = seq_len.min(6);
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    let seg_len = 2 + (pseed as usize) % (max_seg - 1);
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    let start = (pseed as usize) % (seq_len - seg_len + 1);
                    ds.machine_seq[m][start..start+seg_len].reverse();
                    if let Some(ev) = eval_disj(&ds, &mut buf) {
                        if ev.0 < best_mk { best_mk = ev.0; best_ds = ds.clone(); cur_eval = ev; }
                        else if ev.0 <= cur_eval.0 { cur_eval = ev; }
                        else { ds.machine_seq[m][start..start+seg_len].reverse(); }
                    } else { ds.machine_seq[m][start..start+seg_len].reverse(); }
                }
                1 => {
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    let from = (pseed as usize) % seq_len;
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    let to = (pseed as usize) % seq_len;
                    if from == to { continue; }
                    let node = ds.machine_seq[m].remove(from);
                    let ins = to.min(ds.machine_seq[m].len());
                    ds.machine_seq[m].insert(ins, node);
                    if let Some(ev) = eval_disj(&ds, &mut buf) {
                        if ev.0 < best_mk { best_mk = ev.0; best_ds = ds.clone(); cur_eval = ev; }
                        else if ev.0 <= cur_eval.0 { cur_eval = ev; }
                        else { ds.machine_seq[m].remove(ins); ds.machine_seq[m].insert(from, node); }
                    } else { ds.machine_seq[m].remove(ins); ds.machine_seq[m].insert(from, node); }
                }
                _ => {
                    if seq_len < 3 { continue; }
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    let i = (pseed as usize) % seq_len;
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    let j = (pseed as usize) % seq_len;
                    if i == j { continue; }
                    ds.machine_seq[m].swap(i, j);
                    if let Some(ev) = eval_disj(&ds, &mut buf) {
                        if ev.0 < best_mk { best_mk = ev.0; best_ds = ds.clone(); cur_eval = ev; }
                        else if ev.0 <= cur_eval.0 { cur_eval = ev; }
                        else { ds.machine_seq[m].swap(i, j); }
                    } else { ds.machine_seq[m].swap(i, j); }
                }
            }
        }

        if best_mk >= init_eval.0 { return Ok(None); }
        ds = best_ds;
        let Some((mk_final, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
        let sol = disj_to_solution(pre, &ds, &buf.start)?;
        Ok(Some((sol, mk_final)))
    }

    // ============================================================================
    // ELITE DIVERSITY POOL
    // ============================================================================

    #[inline(always)]
    pub fn machine_signature(sol: &Solution, pre: &Pre) -> Vec<u8> {
        let mut sig = Vec::with_capacity(pre.total_ops);
        for j in 0..pre.job_products.len() {
            for &(m, _) in &sol.job_schedule[j] {
                sig.push(m as u8);
            }
        }
        sig
    }

    #[inline(always)]
    pub fn hamming_distance(a: &[u8], b: &[u8]) -> usize {
        a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
    }

    pub fn push_diverse(
        pool: &mut Vec<(Solution, u32, Vec<u8>)>,
        sol: Solution,
        mk: u32,
        sig: Vec<u8>,
        cap: usize,
    ) {
        if pool.iter().any(|(_, pmk, psig)| *pmk == mk && psig == &sig) {
            return;
        }
        pool.push((sol, mk, sig));
        pool.sort_unstable_by_key(|(_, m, _)| *m);
        if pool.len() > cap {
            let mut min_dist = usize::MAX;
            let mut drop_idx = pool.len() - 1;
            for i in 0..pool.len() {
                for j in (i + 1)..pool.len() {
                    let dist = hamming_distance(&pool[i].2, &pool[j].2);
                    if dist < min_dist {
                        min_dist = dist;
                        drop_idx = j;
                    }
                }
            }
            pool.remove(drop_idx);
        }
    }

    // ============================================================================
    // CP PAIR DESCENT — Joint optimization of adjacent critical path nodes
    // ============================================================================

    pub fn cp_pair_descent(
        pre: &Pre,
        challenge: &Challenge,
        base_sol: &Solution,
        max_iters: usize,
    ) -> Result<Option<(Solution, u32)>> {
        let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
        let mut buf = EvalBuf::new(ds.n);
        let Some((mut best_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
        let initial_mk = best_mk;

        let mut improved = true;
        let mut iters = 0;

        while improved && iters < max_iters {
            improved = false;
            iters += 1;

            // Find terminal node (makespan maker)
            let mut terminal = NONE_USIZE;
            let mut max_end = 0u32;
            for i in 0..ds.n {
                let end = buf.start[i].saturating_add(ds.node_pt[i]);
                if end > max_end { max_end = end; terminal = i; }
            }
            if terminal == NONE_USIZE { break; }

            // Trace critical path
            let mut cp: Vec<usize> = Vec::new();
            let mut curr = terminal;
            while curr != NONE_USIZE { cp.push(curr); curr = buf.best_pred[curr]; }
            cp.reverse();

            for idx in 0..cp.len().saturating_sub(1) {
                let u = cp[idx];
                let v = cp[idx + 1];

                let j_u = ds.node_job[u]; let op_u = ds.node_op[u];
                let j_v = ds.node_job[v]; let op_v = ds.node_op[v];

                let opts_u = &pre.product_ops[pre.job_products[j_u]][op_u].machines;
                let opts_v = &pre.product_ops[pre.job_products[j_v]][op_v].machines;
                if opts_u.len() < 2 && opts_v.len() < 2 { continue; }

                let orig_m_u = ds.node_machine[u]; let orig_pt_u = ds.node_pt[u];
                let orig_m_v = ds.node_machine[v]; let orig_pt_v = ds.node_pt[v];

                let mut best_pair_mk = best_mk;
                let mut best_move: Option<(usize, u32, usize, usize, u32, usize)> = None;

                for &(m_u, pt_u) in opts_u.iter().take(4) {
                    for &(m_v, pt_v) in opts_v.iter().take(4) {
                        if m_u == orig_m_u && m_v == orig_m_v { continue; }

                        let old_seq_mu = ds.machine_seq[orig_m_u].clone();
                        let old_seq_mv = if orig_m_u != orig_m_v { ds.machine_seq[orig_m_v].clone() } else { vec![] };

                        ds.machine_seq[orig_m_u].retain(|&x| x != u && x != v);
                        if orig_m_u != orig_m_v { ds.machine_seq[orig_m_v].retain(|&x| x != u && x != v); }

                        ds.node_machine[u] = m_u; ds.node_pt[u] = pt_u;
                        ds.node_machine[v] = m_v; ds.node_pt[v] = pt_v;

                        let len_u = ds.machine_seq[m_u].len();
                        let pos_u_opts: Vec<usize> = if len_u == 0 { vec![0] } else if len_u <= 2 { (0..=len_u).collect() } else { vec![0, len_u / 2, len_u] };

                        for &p_u in &pos_u_opts {
                            ds.machine_seq[m_u].insert(p_u, u);
                            let len_v = ds.machine_seq[m_v].len();
                            let pos_v_opts: Vec<usize> = if len_v == 0 { vec![0] } else if len_v <= 2 { (0..=len_v).collect() } else { vec![0, len_v / 2, len_v] };

                            for &p_v in &pos_v_opts {
                                ds.machine_seq[m_v].insert(p_v, v);
                                if let Some((cand_mk, _)) = eval_disj(&ds, &mut buf) {
                                    if cand_mk < best_pair_mk {
                                        best_pair_mk = cand_mk;
                                        best_move = Some((m_u, pt_u, p_u, m_v, pt_v, p_v));
                                    }
                                }
                                ds.machine_seq[m_v].remove(p_v);
                            }
                            ds.machine_seq[m_u].remove(p_u);
                        }

                        // Restore
                        ds.node_machine[u] = orig_m_u; ds.node_pt[u] = orig_pt_u;
                        ds.node_machine[v] = orig_m_v; ds.node_pt[v] = orig_pt_v;
                        ds.machine_seq[orig_m_u] = old_seq_mu;
                        if orig_m_u != orig_m_v { ds.machine_seq[orig_m_v] = old_seq_mv; }
                    }
                }

                if let Some((m_u, pt_u, p_u, m_v, pt_v, p_v)) = best_move {
                    ds.machine_seq[orig_m_u].retain(|&x| x != u && x != v);
                    if orig_m_u != orig_m_v { ds.machine_seq[orig_m_v].retain(|&x| x != u && x != v); }
                    ds.node_machine[u] = m_u; ds.node_pt[u] = pt_u;
                    ds.node_machine[v] = m_v; ds.node_pt[v] = pt_v;
                    ds.machine_seq[m_u].insert(p_u, u);
                    ds.machine_seq[m_v].insert(p_v, v);
                    best_mk = best_pair_mk;
                    improved = true;
                    eval_disj(&ds, &mut buf);
                    break; // Restart on new CP
                }
            }
        }

        if best_mk >= initial_mk { return Ok(None); }
        let sol = disj_to_solution(pre, &ds, &buf.start)?;
        Ok(Some((sol, best_mk)))
    }

    // ============================================================================
    // PULSAR ENGINE: SCHRAGE ORACLE + SLACK EJECTION
    // ============================================================================

    /// Compute tail values: tail[nd] = longest path from END of nd to makespan.
    pub fn compute_tails_pulsar(ds: &DisjSchedule, buf: &EvalBuf) -> Vec<u32> {
        let n = ds.n;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| buf.start[b].cmp(&buf.start[a]));
        let mut tails = vec![0u32; n];
        for &nd in &order {
            let mut max_after = 0u32;
            let js = ds.job_succ[nd];
            if js != NONE_USIZE { max_after = max_after.max(ds.node_pt[js].saturating_add(tails[js])); }
            let ms = buf.machine_succ[nd];
            if ms != NONE_USIZE { max_after = max_after.max(ds.node_pt[ms].saturating_add(tails[ms])); }
            tails[nd] = max_after;
        }
        tails
    }

    /// Release dates excluding machine excl_m's arcs (correct r_j for single-machine subproblem).
    fn release_dates_excl_machine(ds: &DisjSchedule, excl_m: usize) -> Vec<u32> {
        let n = ds.n;
        let mut indeg = ds.indeg_job.clone();
        let mut msucc = vec![NONE_USIZE; n];
        for m in 0..ds.num_machines {
            if m == excl_m { continue; }
            let seq = &ds.machine_seq[m];
            for i in 0..seq.len().saturating_sub(1) {
                let u = seq[i]; let v = seq[i+1];
                msucc[u] = v;
                indeg[v] = indeg[v].saturating_add(1);
            }
        }
        let mut start = vec![0u32; n];
        let mut stack: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
        while let Some(u) = stack.pop() {
            let eu = start[u].saturating_add(ds.node_pt[u]);
            let js = ds.job_succ[u];
            if js != NONE_USIZE {
                if start[js] < eu { start[js] = eu; }
                indeg[js] = indeg[js].saturating_sub(1);
                if indeg[js] == 0 { stack.push(js); }
            }
            let ms = msucc[u];
            if ms != NONE_USIZE {
                if start[ms] < eu { start[ms] = eu; }
                indeg[ms] = indeg[ms].saturating_sub(1);
                if indeg[ms] == 0 { stack.push(ms); }
            }
        }
        start
    }

    /// Schrage's algorithm for 1|r_j,q_j|C_max. Choose max-tail among available jobs. O(N^2).
    fn schrage_seq_pulsar(nodes: &[usize], r: &[u32], p: &[u32], q: &[u32]) -> Vec<usize> {
        let m = nodes.len();
        if m <= 1 { return nodes.to_vec(); }
        let mut by_r: Vec<usize> = nodes.to_vec();
        by_r.sort_unstable_by_key(|&nd| r[nd]);
        let mut result = Vec::with_capacity(m);
        let mut t = r[by_r[0]];
        let mut i = 0usize;
        let mut avail: Vec<usize> = Vec::with_capacity(m);
        while result.len() < m {
            while i < by_r.len() && r[by_r[i]] <= t { avail.push(by_r[i]); i += 1; }
            if avail.is_empty() {
                if i < by_r.len() { t = r[by_r[i]]; continue; }
                break;
            }
            let bp = avail.iter().enumerate().max_by_key(|&(_, &nd)| q[nd]).map(|(p2,_)| p2).unwrap_or(0);
            let chosen = avail.swap_remove(bp);
            result.push(chosen);
            t = t.saturating_add(p[chosen]);
        }
        result
    }

    /// Apply Schrage resequencing to critical path machines iteratively.
    pub fn schrage_pass(
        pre: &Pre, challenge: &Challenge, base_sol: &Solution, max_iters: usize,
    ) -> Result<Option<(Solution, u32)>> {
        let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
        let mut buf = EvalBuf::new(ds.n);
        let Some((mut cur_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None); };
        let init_mk = cur_mk;
        for _ in 0..max_iters {
            let tails = compute_tails_pulsar(&ds, &buf);
            let mut mc = vec![0usize; ds.num_machines];
            for nd in 0..ds.n {
                if buf.start[nd].saturating_add(ds.node_pt[nd]) == cur_mk { mc[ds.node_machine[nd]] += 1; }
            }
            let mut m_order: Vec<usize> = (0..ds.num_machines)
                .filter(|&m| mc[m] > 0 && ds.machine_seq[m].len() >= 2).collect();
            m_order.sort_unstable_by(|&a, &b| mc[b].cmp(&mc[a]));
            let mut improved = false;
            for m in m_order {
                let r_excl = release_dates_excl_machine(&ds, m);
                let nodes = ds.machine_seq[m].clone();
                let new_seq = schrage_seq_pulsar(&nodes, &r_excl, &ds.node_pt, &tails);
                if new_seq == nodes { continue; }
                let old_seq = ds.machine_seq[m].clone();
                ds.machine_seq[m] = new_seq;
                if let Some((new_mk, _)) = eval_disj(&ds, &mut buf) {
                    if new_mk < cur_mk { cur_mk = new_mk; improved = true; break; }
                }
                ds.machine_seq[m] = old_seq;
                let _ = eval_disj(&ds, &mut buf);
            }
            if !improved { break; }
        }
        if cur_mk < init_mk { Ok(Some((disj_to_solution(pre, &ds, &buf.start)?, cur_mk))) } else { Ok(None) }
    }

    /// D-ILS: iterative perturbation + full Schrage sweep over all machines.
    /// Each attempt: random adjacent swap on bottleneck machine → sweep all machines
    /// with Schrage resequencing → if no improvement, revert perturbation.
    /// Fuel-based (no wall-clock). Runs num_attempts perturbation+sweep cycles.
    pub fn dils_schrage_phase(
        pre: &Pre,
        challenge: &Challenge,
        start_sol: &Solution,
        num_attempts: usize,
        rng: &mut SmallRng,
    ) -> Result<Option<(Solution, u32)>> {
        let mut ds = build_disj_from_solution(pre, challenge, start_sol)?;
        let mut buf = EvalBuf::new(ds.n);
        let Some((init_mk, _)) = eval_disj(&ds, &mut buf) else {
            return Ok(None);
        };
        let mut best_mk = init_mk;
        let mut best_ds = ds.clone();
        let mut best_start = buf.start.clone();

        for _ in 0..num_attempts {
            let pre_mk = best_mk;
            let pre_ds = ds.clone();

            // Find bottleneck machine: machine m maximising total_pt × (cp_count + 1)
            let tails = compute_tails_pulsar(&ds, &buf);
            let mut machine_score = vec![0u64; ds.num_machines];
            for m in 0..ds.num_machines {
                let mut total_pt = 0u64;
                let mut cp_cnt = 0u64;
                for &nd in &ds.machine_seq[m] {
                    total_pt = total_pt.saturating_add(ds.node_pt[nd] as u64);
                    let end = (buf.start[nd] as u64).saturating_add(ds.node_pt[nd] as u64);
                    if end.saturating_add(tails[nd] as u64) >= best_mk as u64 {
                        cp_cnt += 1;
                    }
                }
                machine_score[m] = total_pt * (cp_cnt + 1);
            }
            let candidates: Vec<usize> = (0..ds.num_machines)
                .filter(|&m| ds.machine_seq[m].len() >= 2)
                .collect();
            if candidates.is_empty() { return Ok(None); }
            let bottleneck = candidates.iter().copied()
                .max_by_key(|&m| machine_score[m])
                .unwrap_or_else(|| candidates[rng.gen_range(0..candidates.len())]);

            // Perturb: one adjacent swap on the bottleneck machine
            let blen = ds.machine_seq[bottleneck].len();
            let pos = rng.gen_range(0..blen - 1);
            ds.machine_seq[bottleneck].swap(pos, pos + 1);
            if eval_disj(&ds, &mut buf).is_none() {
                ds = pre_ds;
                let _ = eval_disj(&ds, &mut buf);
                continue;
            }

            // Full Schrage sweep over ALL machines in random order
            let mut m_order: Vec<usize> = (0..ds.num_machines).collect();
            m_order.shuffle(rng);
            for &m in &m_order {
                let seq = ds.machine_seq[m].clone();
                if seq.len() < 2 { continue; }
                // release dates and tails are node-ID indexed (matches schrage_seq_pulsar expectation)
                let r = release_dates_excl_machine(&ds, m);
                let tails2 = compute_tails_pulsar(&ds, &buf);
                let new_order = schrage_seq_pulsar(&seq, &r, &ds.node_pt, &tails2);
                if new_order == seq { continue; }
                ds.machine_seq[m] = new_order;
                if let Some((new_mk, _)) = eval_disj(&ds, &mut buf) {
                    if new_mk < best_mk {
                        best_mk = new_mk;
                        best_ds = ds.clone();
                        best_start = buf.start.clone();
                    } else {
                        ds.machine_seq[m] = seq;
                        let _ = eval_disj(&ds, &mut buf); // restore buf after revert
                    }
                } else {
                    ds.machine_seq[m] = seq;
                    let _ = eval_disj(&ds, &mut buf); // restore buf after revert
                }
            }

            // If perturbation+sweep found no improvement, revert to pre-perturb state
            if best_mk >= pre_mk {
                ds = pre_ds;
                let _ = eval_disj(&ds, &mut buf);
            }
        }

        if best_mk < init_mk {
            let sol = disj_to_solution(pre, &best_ds, &best_start)?;
            Ok(Some((sol, best_mk)))
        } else {
            Ok(None)
        }
    }

    /// Slack Ejection: reassign CP flex nodes, Schrage positioning. Fixed save/restore.
    pub fn slack_ejection_schrage(
        pre: &Pre, challenge: &Challenge, base_sol: &Solution, max_iters: usize,
    ) -> Result<Option<(Solution, u32)>> {
        let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
        let mut buf = EvalBuf::new(ds.n);
        let Some((mut cur_mk, mut terminal)) = eval_disj(&ds, &mut buf) else { return Ok(None); };
        let init_mk = cur_mk;
        for _ in 0..max_iters {
            let tails = compute_tails_pulsar(&ds, &buf);
            let mut cp: Vec<usize> = Vec::new();
            let mut curr = terminal;
            while curr != NONE_USIZE { cp.push(curr); curr = buf.best_pred[curr]; }
            let cp_flex: Vec<usize> = cp.iter().filter(|&&nd| {
                if ds.node_job[nd] >= pre.job_products.len() { return false; }
                let prod = pre.job_products[ds.node_job[nd]]; let op = ds.node_op[nd];
                prod < pre.product_ops.len() && op < pre.product_ops[prod].len()
                    && pre.product_ops[prod][op].machines.len() > 1
            }).cloned().collect();
            if cp_flex.is_empty() { break; }
            let mut improved = false;
            'node_loop2: for &nd in &cp_flex {
                let job = ds.node_job[nd]; let op_idx = ds.node_op[nd];
                let orig_m = ds.node_machine[nd]; let orig_pt = ds.node_pt[nd];
                if job >= pre.job_products.len() { continue; }
                let prod = pre.job_products[job];
                if op_idx >= pre.product_ops[prod].len() { continue; }
                let op_info = &pre.product_ops[prod][op_idx];
                let mut best_mk2 = cur_mk;
                let mut best_commit: Option<(usize, u32, Vec<usize>, Vec<usize>)> = None;
                for &(new_m, new_pt) in &op_info.machines {
                    if new_m == orig_m { continue; }
                    let saved_orig = ds.machine_seq[orig_m].clone();
                    let saved_new = ds.machine_seq[new_m].clone();
                    ds.machine_seq[orig_m].retain(|&x| x != nd);
                    ds.node_machine[nd] = new_m; ds.node_pt[nd] = new_pt;
                    ds.machine_seq[new_m].push(nd);
                    if eval_disj(&ds, &mut buf).is_some() {
                        let tmp_tails = compute_tails_pulsar(&ds, &buf);
                        let nodes_nm: Vec<usize> = ds.machine_seq[new_m].clone();
                        let r_excl = release_dates_excl_machine(&ds, new_m);
                        let sseq = schrage_seq_pulsar(&nodes_nm, &r_excl, &ds.node_pt, &tmp_tails);
                        ds.machine_seq[new_m] = sseq;
                        if let Some((cand_mk, _)) = eval_disj(&ds, &mut buf) {
                            if cand_mk < best_mk2 {
                                best_mk2 = cand_mk;
                                best_commit = Some((new_m, new_pt, ds.machine_seq[orig_m].clone(), ds.machine_seq[new_m].clone()));
                            }
                        }
                    }
                    ds.machine_seq[orig_m] = saved_orig;
                    ds.machine_seq[new_m] = saved_new;
                    ds.node_machine[nd] = orig_m; ds.node_pt[nd] = orig_pt;
                    let _ = eval_disj(&ds, &mut buf);
                }
                if let Some((new_m, new_pt, new_orig, new_new)) = best_commit {
                    ds.machine_seq[orig_m] = new_orig;
                    ds.machine_seq[new_m] = new_new;
                    ds.node_machine[nd] = new_m; ds.node_pt[nd] = new_pt;
                    if let Some((new_mk2, new_term)) = eval_disj(&ds, &mut buf) {
                        cur_mk = new_mk2; terminal = new_term; improved = true; break 'node_loop2;
                    }
                }
            }
            if !improved { break; }
            let _ = tails;
        }
        if cur_mk < init_mk { Ok(Some((disj_to_solution(pre, &ds, &buf.start)?, cur_mk))) } else { Ok(None) }
    }

    /// CP Descent Brute-Force: reassign CP flex nodes, try every insertion position.
    /// Unlike slack_ejection_schrage, does NOT reorder other ops in the new machine.
    /// Conservative: preserves optimized orderings found by previous LS passes.
    pub fn cp_descent_bf(
        pre: &Pre, challenge: &Challenge, base_sol: &Solution, max_iters: usize,
    ) -> Result<Option<(Solution, u32)>> {
        let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
        let mut buf = EvalBuf::new(ds.n);
        let Some((mut cur_mk, mut terminal)) = eval_disj(&ds, &mut buf) else { return Ok(None); };
        let init_mk = cur_mk;
        for _ in 0..max_iters {
            let mut cp: Vec<usize> = Vec::new();
            let mut curr = terminal;
            while curr != NONE_USIZE { cp.push(curr); curr = buf.best_pred[curr]; }
            let cp_flex: Vec<usize> = cp.iter().filter(|&&nd| {
                if ds.node_job[nd] >= pre.job_products.len() { return false; }
                let prod = pre.job_products[ds.node_job[nd]]; let op = ds.node_op[nd];
                prod < pre.product_ops.len() && op < pre.product_ops[prod].len()
                    && pre.product_ops[prod][op].machines.len() > 1
            }).cloned().collect();
            if cp_flex.is_empty() { break; }
            let mut improved = false;
            'node_loop: for &nd in &cp_flex {
                let job = ds.node_job[nd]; let op_idx = ds.node_op[nd];
                let orig_m = ds.node_machine[nd]; let orig_pt = ds.node_pt[nd];
                if job >= pre.job_products.len() { continue; }
                let prod = pre.job_products[job];
                if op_idx >= pre.product_ops[prod].len() { continue; }
                let op_info = &pre.product_ops[prod][op_idx];
                let mut best_mk2 = cur_mk;
                let mut best_patch: Option<(usize, u32, Vec<usize>, Vec<usize>)> = None;
                for &(new_m, new_pt) in &op_info.machines {
                    if new_m == orig_m { continue; }
                    let saved_orig = ds.machine_seq[orig_m].clone();
                    let saved_new = ds.machine_seq[new_m].clone();
                    // Remove nd from orig_m
                    let old_pos = match ds.machine_seq[orig_m].iter().position(|&x| x == nd) {
                        Some(p) => p, None => continue,
                    };
                    ds.machine_seq[orig_m].remove(old_pos);
                    ds.node_machine[nd] = new_m;
                    ds.node_pt[nd] = new_pt;
                    // Try each insertion position in new_m
                    let tlen = ds.machine_seq[new_m].len();
                    for ins in 0..=tlen {
                        ds.machine_seq[new_m].insert(ins, nd);
                        if let Some((cand_mk, _)) = eval_disj(&ds, &mut buf) {
                            if cand_mk < best_mk2 {
                                best_mk2 = cand_mk;
                                best_patch = Some((new_m, new_pt, ds.machine_seq[orig_m].clone(), ds.machine_seq[new_m].clone()));
                            }
                        }
                        ds.machine_seq[new_m].remove(ins);
                    }
                    // Restore state exactly
                    ds.machine_seq[orig_m] = saved_orig;
                    ds.machine_seq[new_m] = saved_new;
                    ds.node_machine[nd] = orig_m;
                    ds.node_pt[nd] = orig_pt;
                    let _ = eval_disj(&ds, &mut buf);
                }
                if let Some((new_m, new_pt, new_orig_seq, new_m_seq)) = best_patch {
                    ds.machine_seq[orig_m] = new_orig_seq;
                    ds.machine_seq[new_m] = new_m_seq;
                    ds.node_machine[nd] = new_m;
                    ds.node_pt[nd] = new_pt;
                    if let Some((new_mk2, new_term)) = eval_disj(&ds, &mut buf) {
                        cur_mk = new_mk2; terminal = new_term; improved = true; break 'node_loop;
                    }
                }
            }
            if !improved { break; }
        }
        if cur_mk < init_mk { Ok(Some((disj_to_solution(pre, &ds, &buf.start)?, cur_mk))) } else { Ok(None) }
    }

    // ============================================================================
    // WINDOW EXHAUSTIVE SEARCH — Brute-force bottleneck machine permutations
    // ============================================================================

    fn heap_permutations(a: &mut [usize], size: usize, res: &mut Vec<Vec<usize>>) {
        if size == 1 { res.push(a.to_vec()); return; }
        for i in 0..size {
            heap_permutations(a, size - 1, res);
            if size % 2 == 1 { a.swap(0, size - 1); } else { a.swap(i, size - 1); }
        }
    }

    pub fn window_exhaustive(
        pre: &Pre,
        challenge: &Challenge,
        base_sol: &Solution,
        window_sizes: &[usize],
    ) -> Result<Option<(Solution, u32)>> {
        let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
        let mut buf = EvalBuf::new(ds.n);
        let Some((mut best_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
        let initial_mk = best_mk;

        let mut improved_any = false;
        let mut loop_improved = true;

        while loop_improved {
            loop_improved = false;

            // Find terminal + CP
            let mut terminal = NONE_USIZE;
            let mut max_end = 0u32;
            for i in 0..ds.n {
                let end = buf.start[i].saturating_add(ds.node_pt[i]);
                if end > max_end { max_end = end; terminal = i; }
            }
            if terminal == NONE_USIZE { break; }

            let mut cp_set = vec![false; ds.n];
            let mut curr = terminal;
            while curr != NONE_USIZE { cp_set[curr] = true; curr = buf.best_pred[curr]; }

            // Find bottleneck machine (most CP nodes)
            let mut m_counts = vec![0usize; ds.num_machines];
            for i in 0..ds.n { if cp_set[i] { m_counts[ds.node_machine[i]] += 1; } }
            let bn_m = m_counts.iter().enumerate().max_by_key(|&(_, &c)| c).map(|(m, _)| m).unwrap_or(0);

            for &w_size in window_sizes {
                let seq_len = ds.machine_seq[bn_m].len();
                if seq_len < w_size { continue; }
                let mut window_improved = false;

                for start in 0..=(seq_len - w_size) {
                    let window: Vec<usize> = ds.machine_seq[bn_m][start..start + w_size].to_vec();
                    let mut perms = Vec::with_capacity(24);
                    let mut perm_buf = window.clone();
                    heap_permutations(&mut perm_buf, w_size, &mut perms);

                    let mut local_best_mk = best_mk;
                    let mut best_perm: Option<Vec<usize>> = None;

                    for perm in &perms {
                        if *perm == window { continue; }
                        ds.machine_seq[bn_m][start..start + w_size].copy_from_slice(perm);

                        if let Some((cand_mk, _)) = eval_disj(&ds, &mut buf) {
                            if cand_mk < local_best_mk {
                                local_best_mk = cand_mk;
                                best_perm = Some(perm.clone());
                            }
                        }
                    }

                    if let Some(ref perm) = best_perm {
                        ds.machine_seq[bn_m][start..start + w_size].copy_from_slice(perm);
                        best_mk = local_best_mk;
                        eval_disj(&ds, &mut buf);
                        window_improved = true;
                        loop_improved = true;
                        improved_any = true;
                        break;
                    } else {
                        ds.machine_seq[bn_m][start..start + w_size].copy_from_slice(&window);
                    }
                }
                if window_improved { break; }
            }
        }

        if best_mk >= initial_mk { return Ok(None); }
        let sol = disj_to_solution(pre, &ds, &buf.start)?;
        Ok(Some((sol, best_mk)))
    }

    /// Bottleneck Machine Relief Pass: identifies the most-loaded or most-CP machine,
    /// tries to move flexible ops from it to alternative machines with smart position ordering.
    /// Breaks as soon as no improvement is found (like av5).
    pub fn bottleneck_machine_relief_pass(
        pre: &Pre,
        challenge: &Challenge,
        base_sol: &Solution,
        max_iters: usize,
    ) -> Result<Option<(Solution, u32)>> {
        let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
        let mut buf = EvalBuf::new(ds.n);
        let Some((mut current_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None); };
        let initial_mk = current_mk;
        let nm = ds.machine_seq.len();
        let mut any_improvement = false;

        // Maintain machine loads for efficient load-based targeting
        let mut machine_loads = vec![0u64; nm];
        for nd in 0..ds.n {
            let m = ds.node_machine[nd];
            if m < nm {
                machine_loads[m] = machine_loads[m].saturating_add(ds.node_pt[nd] as u64);
            }
        }

        for iter in 0..max_iters {
            let target_m = if iter % 2 == 0 {
                machine_loads.iter().enumerate()
                    .max_by_key(|&(_, &v)| v)
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            } else {
                // CP-based: count nodes on critical path per machine
                let Some((_, _)) = eval_disj(&ds, &mut buf) else { break };
                let mut cp_count = vec![0usize; nm];
                for nd in 0..ds.n {
                    if buf.start[nd] + ds.node_pt[nd] == current_mk {
                        let m = ds.node_machine[nd];
                        if m < nm { cp_count[m] += 1; }
                    }
                }
                cp_count.iter().enumerate().max_by_key(|&(_, &v)| v).map(|(i, _)| i).unwrap_or(0)
            };

            let seq_len = ds.machine_seq[target_m].len();
            if seq_len <= 1 { break; }

            let target_nodes: Vec<usize> = ds.machine_seq[target_m].clone();
            let has_flex = target_nodes.iter().any(|&nd| {
                let job = ds.node_job[nd];
                let prod = pre.job_products[job];
                let op = ds.node_op[nd];
                pre.product_ops[prod][op].machines.len() > 1
            });
            if !has_flex { break; }

            let mut best_iter_mk = current_mk;
            let mut best_nd = usize::MAX;
            let mut best_new_m = usize::MAX;
            let mut best_new_pt = 0u32;
            let mut best_ins_pos = 0usize;

            for &nd in &target_nodes {
                let job = ds.node_job[nd];
                let op = ds.node_op[nd];
                let prod = pre.job_products[job];
                let op_info = &pre.product_ops[prod][op];
                if op_info.machines.len() <= 1 { continue; }

                let cur_m = ds.node_machine[nd];
                let cur_pt = ds.node_pt[nd];
                let old_pos = match ds.machine_seq[cur_m].iter().position(|&x| x == nd) {
                    Some(p) => p,
                    None => continue,
                };

                // Remove nd from cur_m once, restore at end
                ds.machine_seq[cur_m].remove(old_pos);

                // Estimate job predecessor end time for position sorting
                let n_in_job = ds.n; // used as sentinel check
                let _ = n_in_job;
                let job_pred_end = if nd > 0 && ds.node_job[nd - 1] == job {
                    buf.start[nd - 1].saturating_add(ds.node_pt[nd - 1])
                } else { 0u32 };

                for &(new_m, new_pt) in &op_info.machines {
                    if new_m == cur_m { continue; }

                    ds.node_machine[nd] = new_m;
                    ds.node_pt[nd] = new_pt;

                    let tgt_len = ds.machine_seq[new_m].len();

                    // Sort positions by estimated start time (earlier = try first)
                    let mut pos_estimates: Vec<(usize, u32)> = Vec::with_capacity(tgt_len + 1);
                    for pos in 0..=tgt_len {
                        let mp_end = if pos > 0 {
                            let pred = ds.machine_seq[new_m][pos - 1];
                            buf.start[pred].saturating_add(ds.node_pt[pred])
                        } else { 0u32 };
                        let start_est = job_pred_end.max(mp_end);
                        pos_estimates.push((pos, start_est));
                    }
                    pos_estimates.sort_unstable_by_key(|&(_, s)| s);

                    for &(pos, _) in &pos_estimates {
                        ds.machine_seq[new_m].insert(pos, nd);
                        if let Some((test_mk, _)) = eval_disj(&ds, &mut buf) {
                            if test_mk < best_iter_mk {
                                best_iter_mk = test_mk;
                                best_nd = nd;
                                best_new_m = new_m;
                                best_new_pt = new_pt;
                                best_ins_pos = pos;
                            }
                        }
                        ds.machine_seq[new_m].remove(pos);
                    }

                    ds.node_machine[nd] = cur_m;
                    ds.node_pt[nd] = cur_pt;
                }

                // Restore nd to cur_m
                ds.machine_seq[cur_m].insert(old_pos, nd);
            }

            if best_nd != usize::MAX && best_iter_mk < current_mk {
                let cur_m = ds.node_machine[best_nd];
                let cur_pt = ds.node_pt[best_nd];
                let old_pos = ds.machine_seq[cur_m].iter().position(|&x| x == best_nd).unwrap();
                ds.machine_seq[cur_m].remove(old_pos);
                let ins = best_ins_pos.min(ds.machine_seq[best_new_m].len());
                ds.machine_seq[best_new_m].insert(ins, best_nd);
                machine_loads[cur_m] = machine_loads[cur_m].saturating_sub(cur_pt as u64);
                machine_loads[best_new_m] = machine_loads[best_new_m].saturating_add(best_new_pt as u64);
                ds.node_machine[best_nd] = best_new_m;
                ds.node_pt[best_nd] = best_new_pt;
                current_mk = best_iter_mk;
                any_improvement = true;
                eval_disj(&ds, &mut buf);
            } else {
                break; // No improvement found — stop
            }
        }

        if !any_improvement || current_mk >= initial_mk { return Ok(None); }
        let sol = disj_to_solution(pre, &ds, &buf.start)?;
        Ok(Some((sol, current_mk)))
    }

}

pub mod solver {
    // TIG's UI uses the pattern `tig_challenges::job_scheduling` to automatically detect your algorithm's challenge
    use anyhow::Result;
    use serde::{Deserialize, Serialize};
    use serde_json::{Map, Value};
    use tig_challenges::job_scheduling::*;

    use super::types::EffortConfig;
    use super::preprocess::build_pre;
    use super::infra::run_simple_greedy_baseline;
    use super::detect;
    use super::flow_shop;
    use super::hybrid_flow_shop;
    use super::job_shop;
    use super::fjsp_medium;
    use super::fjsp_high;

    #[derive(Serialize, Deserialize)]
    pub struct Hyperparameters {
        #[serde(default)]
        pub track: Option<String>,
        #[serde(default)]
        pub effort: Option<String>,
        #[serde(default)]
        pub num_restarts: Option<u64>,
        #[serde(default)]
        pub job_shop_iters: Option<u64>,
        #[serde(default)]
        pub flow_shop_iters: Option<u64>,
        #[serde(default)]
        pub hybrid_flow_shop_iters: Option<u64>,
        #[serde(default)]
        pub fjsp_high_iters: Option<u64>,
    }

    fn parse_effort(hyperparameters: &Option<Map<String, Value>>) -> EffortConfig {
        let mut cfg = EffortConfig::default_effort();
        if let Some(map) = hyperparameters {
            if let Some(Value::Number(n)) = map.get("num_restarts") {
                if let Some(v) = n.as_u64() {
                    cfg = EffortConfig::from_value(v as usize);
                }
            }
            if let Some(Value::String(s)) = map.get("effort") {
                cfg = EffortConfig::from_str(s);
            }
            if let Some(Value::Number(n)) = map.get("job_shop_iters") {
                if let Some(v) = n.as_u64() {
                    cfg = cfg.with_job_shop_iters(v as usize);
                }
            }
            if let Some(Value::Number(n)) = map.get("flow_shop_iters") {
                if let Some(v) = n.as_u64() {
                    cfg = cfg.with_flow_shop_iters(v as usize);
                }
            }
            if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_iters") {
                if let Some(v) = n.as_u64() {
                    cfg = cfg.with_hybrid_flow_shop_iters(v as usize);
                }
            }
            if let Some(Value::Number(n)) = map.get("fjsp_high_iters") {
                if let Some(v) = n.as_u64() {
                    cfg = cfg.with_fjsp_high_iters(v as usize);
                }
            }
        }
        cfg
    }

    #[derive(Debug, Clone, Copy)]
    enum Track { FlowShop, HybridFlowShop, JobShop, FjspMedium, FjspHigh }

    fn parse_track(hyperparameters: &Option<Map<String, Value>>) -> Option<Track> {
        if let Some(map) = hyperparameters {
            if let Some(Value::String(s)) = map.get("track") {
                return Some(match s.to_lowercase().as_str() {
                    "flow_shop" | "flow" => Track::FlowShop,
                    "hybrid_flow_shop" | "hybrid" => Track::HybridFlowShop,
                    "job_shop" | "job" => Track::JobShop,
                    "fjsp_medium" | "medium" => Track::FjspMedium,
                    "fjsp_high" | "high" | "fjsp" => Track::FjspHigh,
                    _ => return None,
                });
            }
        }
        None
    }

    fn detect_track_simple(challenge: &Challenge) -> Track {
        let mut total_flex = 0usize;
        let mut total_ops = 0usize;
        for p in 0..challenge.product_processing_times.len() {
            for op in &challenge.product_processing_times[p] {
                total_flex += op.len();
                total_ops += 1;
            }
        }
        let flex_avg = if total_ops > 0 { total_flex as f64 / total_ops as f64 } else { 1.0 };

        let mut max_ops = 0usize;
        let mut min_ops = usize::MAX;
        for p in 0..challenge.product_processing_times.len() {
            let nops = challenge.product_processing_times[p].len();
            if nops > max_ops { max_ops = nops; }
            if nops < min_ops { min_ops = nops; }
        }
        let uniform_routing = max_ops == min_ops;

        // Check if it's a true flow shop: all products use the SAME machine for each op position
        let is_flow_shop = if uniform_routing && flex_avg <= 1.5 && !challenge.product_processing_times.is_empty() {
            let n_ops = challenge.product_processing_times[0].len();
            let mut ok = true;
            'outer: for op_idx in 0..n_ops {
                let m0 = match challenge.product_processing_times[0][op_idx].keys().next() {
                    Some(&m) => m,
                    None => { ok = false; break; }
                };
                for p in 1..challenge.product_processing_times.len() {
                    if !challenge.product_processing_times[p][op_idx].contains_key(&m0) {
                        ok = false;
                        break 'outer;
                    }
                }
            }
            ok
        } else {
            false
        };

        if flex_avg > 5.0 {
            Track::FjspHigh
        } else if flex_avg > 1.5 && !uniform_routing {
            Track::FjspMedium
        } else if flex_avg > 1.5 {
            Track::HybridFlowShop
        } else if is_flow_shop {
            Track::FlowShop
        } else {
            Track::JobShop
        }
    }

    pub fn solve_challenge(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        hyperparameters: &Option<Map<String, Value>>,
    ) -> Result<()> {
        // HP "track" overrides auto-detection — guarantees correct routing per track
        let track = parse_track(hyperparameters).unwrap_or_else(|| detect_track_simple(challenge));

        match track {
            Track::FjspMedium => {
                let effort = parse_effort(hyperparameters);
                let (greedy_sol, greedy_mk) = run_simple_greedy_baseline(challenge)?;
                save_solution(&greedy_sol)?;
                let pre = build_pre(challenge)?;
                fjsp_medium::solve(challenge, save_solution, &pre, greedy_sol, greedy_mk, &effort)
            }
            Track::HybridFlowShop => {
                let effort = parse_effort(hyperparameters);
                let (greedy_sol, greedy_mk) = run_simple_greedy_baseline(challenge)?;
                save_solution(&greedy_sol)?;
                let pre = build_pre(challenge)?;
                hybrid_flow_shop::solve(challenge, save_solution, &pre, greedy_sol, greedy_mk, &effort)
            }
            Track::FlowShop => {
                let effort = parse_effort(hyperparameters);
                let (greedy_sol, greedy_mk) = run_simple_greedy_baseline(challenge)?;
                save_solution(&greedy_sol)?;
                let pre = build_pre(challenge)?;
                flow_shop::solve(challenge, save_solution, &pre, greedy_sol, greedy_mk, &effort)
            }
            Track::JobShop => {
                let effort = parse_effort(hyperparameters);
                let pre = build_pre(challenge)?;
                job_shop::solve(challenge, save_solution, &pre, &effort)
            }
            Track::FjspHigh => {
                let effort = parse_effort(hyperparameters);
                let (greedy_sol, greedy_mk) = run_simple_greedy_baseline(challenge)?;
                save_solution(&greedy_sol)?;
                let pre = build_pre(challenge)?;
                fjsp_high::solve(challenge, save_solution, &pre, greedy_sol, greedy_mk, &effort)
            }
        }
    }

    pub fn help() {
        println!("job_four - Per-track FJSP solver");
        println!();
        println!("HYPERPARAMETERS:");
        println!("  track: \"flow_shop\" | \"hybrid_flow_shop\" | \"job_shop\" | \"fjsp_medium\" | \"fjsp_high\"");
        println!("  effort: \"default\" | \"medium\" | \"high\" | \"extreme\"");
        println!("  job_shop_iters: integer (default 12000)");
        println!("  hybrid_flow_shop_iters: integer (default 12000)");
        println!("  fjsp_high_iters: integer (default 5000)");
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
        pre: &Pre, rule: Rule, job: usize, product: usize, op_idx: usize,
        ops_rem: usize, op: &OpInfo, machine: usize, pt: u32, time: u32,
        target_mk: Option<u32>, best_end: u32, second_end: u32,
        best_cnt_total: usize, progress: f64, job_bias: f64,
        _machine_penalty: f64, dynamic_load: f64,
        route_pref: Option<&RoutePrefLite>, route_w: f64, jitter: f64,
    ) -> f64 {
        let rem_min = pre.product_suf_min[product][op_idx] as f64;
        let rem_avg = pre.product_suf_avg[product][op_idx];
        let rem_bn  = pre.product_suf_bn[product][op_idx];
        let flex_f  = (op.flex as f64).max(1.0);
        let flex_inv = 1.0 / flex_f;
        let rem_min_n = rem_min / pre.horizon.max(1.0);
        let _rem_avg_n = rem_avg / pre.max_job_avg_work.max(1e-9);
        let _bn_n = rem_bn / pre.max_job_bn.max(1e-9);
        let ops_n = (ops_rem as f64) / (pre.max_ops as f64).max(1.0);
        let _load_n = dynamic_load / pre.avg_machine_load.max(1e-9);
        let _scar_n = pre.machine_scarcity[machine] / pre.avg_machine_scarcity.max(1e-9);
        let end_n   = (best_end as f64) / pre.time_scale.max(1.0);
        let proc_n  = (pt as f64) / pre.avg_op_min.max(1.0);
        let regret  = if second_end >= INF { pre.avg_op_min * 2.6 } else { (second_end - best_end) as f64 };
        let reg_n   = (regret / pre.avg_op_min.max(1.0)).clamp(0.0, 6.0);
        let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
        let density_n = ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0);
        let next_min   = pre.product_next_min[product][op_idx] as f64;
        let next_min_n = next_min / pre.horizon.max(1.0);
        let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
        let p2 = progress * progress;
        let next_w_base = 0.12 + p2 * 0.28;
        let next_term_raw = (0.55 * next_min_n + 0.45 * next_flex_inv) * (1.0 + 0.30 * density_n * pre.high_flex);
        let js = pre.jobshopness;
        let _fl = 1.0 - js;
        let pop_pen = if pre.chaotic_like && op.flex >= 2 {
            let pop = pre.machine_best_pop[machine];
            (0.07 + 0.15 * (1.0 - progress)).clamp(0.05, 0.24) * pop * pre.flex_factor
        } else { 0.0 };

        let slack_u = if let Some(tgt) = target_mk {
            let lb = (time as u64).saturating_add(pre.product_suf_min[product][op_idx] as u64);
            let slack = (tgt as i64) - (lb as i64);
            let scale = (0.70 * pre.avg_op_min).max(1.0);
            let pos = (slack.max(0) as f64) / scale;
            let neg = ((-slack).max(0) as f64) / scale;
            (1.0 / (1.0 + pos)).clamp(0.0, 1.0) + (0.35 * neg).min(3.0)
        } else { 0.0 };
        let _slack_w = pre.slack_base * (0.25 + 0.75 * progress);

        let flow_term = pre.flow_w * pre.job_flow_pref[job] * (0.65 + 0.70 * (1.0 - progress));
        let route_term = if route_w > 0.0 && op.flex >= 2 {
            let rp = route_pref;
            let bonus = if let Some(rp) = rp {
                if product < rp.len() && op_idx < rp[product].len() {
                    let r = rp[product][op_idx];
                    let mu = machine.min(255) as u8;
                    if mu == r.best_m { (r.best_w as f64) / 255.0 }
                    else if mu == r.second_m { (r.second_w as f64) / 255.0 }
                    else { 0.0 }
                } else { 0.0 }
            } else { 0.0 };
            let route_gain = (0.70 + 0.80 * (1.0 - progress)).clamp(0.70, 1.40);
            route_w * route_gain * bonus
        } else { 0.0 };

        let _ = (ops_n, _rem_avg_n, _bn_n, _load_n, _scar_n, _fl, _slack_w, slack_u, flex_inv);

        match rule {
            Rule::BnHeavy => {
                let bn_w = (0.90 + 0.55 * js) * pre.bn_focus;
                let end_w = 0.65 + 0.70 * progress;
                let reg_w = (0.60 + 0.25 * (1.0 - progress)) * (0.85 + 0.35 * js);
                let next_term = next_w_base * (0.55 + 0.75 * js) * next_term_raw;
                (0.95 * rem_min_n) + (bn_w * rem_bn / pre.max_job_bn.max(1e-9))
                    + (0.10 * ops_n) + (reg_w * pre.flex_factor) * reg_n
                    + 0.18 * scarcity_urg + next_term
                    - end_w * end_n - 0.18 * proc_n - pop_pen
                    + 0.60 * job_bias + flow_term + route_term + jitter
            }
            Rule::MostWork => {
                let next_term = next_w_base * 0.25 * next_term_raw;
                (1.00 * rem_avg) / pre.max_job_avg_work.max(1e-9)
                    + (0.12 * ops_n) + (0.18 * scarcity_urg) + next_term
                    - (0.62 * end_n) - pop_pen + (0.45 * job_bias)
                    + flow_term + route_term + jitter
            }
            Rule::EndTight => {
                let end_w = 1.10 + 1.00 * progress + 0.35 * pre.high_flex;
                let cp_w = 1.15 + 0.30 * js;
                let reg_w = (0.55 + 0.20 * (1.0 - progress)) * (0.85 + 0.60 * js);
                let next_term = next_w_base * (0.45 + 0.55 * js) * next_term_raw;
                (cp_w * rem_min_n) + 0.08 * ops_n + 0.18 * scarcity_urg
                    + (reg_w * pre.flex_factor) * reg_n + next_term
                    - end_w * end_n - 0.22 * proc_n - pop_pen
                    + 0.55 * job_bias + flow_term + route_term + jitter
            }
            Rule::ShortestProc => {
                let next_term = next_w_base * 0.20 * next_term_raw;
                (-1.00 * proc_n) + (0.25 * rem_min_n) + (0.12 * scarcity_urg)
                    + next_term - (0.20 * end_n) - pop_pen
                    + (0.25 * job_bias) + flow_term + route_term + jitter
            }
            Rule::LeastFlex => {
                let next_term = next_w_base * 0.20 * next_term_raw;
                (1.00 * flex_inv) + (0.28 * rem_min_n) + (0.22 * scarcity_urg)
                    + next_term - (0.55 * end_n) - pop_pen
                    + (0.35 * job_bias) + flow_term + route_term + jitter
            }
            Rule::CriticalPath => {
                let next_term = next_w_base * 0.30 * next_term_raw;
                (1.03 * rem_min_n) + (0.10 * ops_n) + (0.24 * scarcity_urg)
                    + next_term - (0.70 * end_n) - pop_pen
                    + (0.45 * job_bias) + flow_term + route_term + jitter
            }
            Rule::Regret => {
                let next_term = next_w_base * 0.25 * next_term_raw;
                (1.05 * reg_n) + (0.55 * rem_min_n) + (0.22 * scarcity_urg)
                    + next_term - (0.68 * end_n) - pop_pen
                    + (0.35 * job_bias) + flow_term + route_term + jitter
            }
            Rule::EarliestStart => {
                let start_n = (time as f64) / pre.time_scale.max(1.0);
                let next_term = next_w_base * 0.20 * next_term_raw;
                -(1.20 * start_n) + (0.40 * rem_min_n) + (0.15 * scarcity_urg)
                    + next_term - (0.30 * proc_n) - pop_pen
                    + (0.30 * job_bias) + flow_term + route_term + jitter
            }
            Rule::MachineBalance => {
                let load_n = dynamic_load / pre.avg_machine_load.max(1e-9);
                let next_term = next_w_base * 0.20 * next_term_raw;
                -(0.80 * load_n) + (0.50 * rem_min_n) + (0.25 * scarcity_urg)
                    + next_term - (0.45 * end_n) - pop_pen
                    + (0.35 * job_bias) + flow_term + route_term + jitter
            }
            Rule::SlackRatio => {
                let time_to_horizon = (pre.horizon - time as f64).max(1.0);
                let cr = (rem_min / time_to_horizon).clamp(0.0, 4.0);
                let next_term = next_w_base * 0.25 * next_term_raw;
                (1.10 * cr) + (0.35 * rem_min_n) + (0.20 * scarcity_urg)
                    + next_term - (0.55 * end_n) - pop_pen
                    + (0.40 * job_bias) + flow_term + route_term + jitter
            }
            Rule::BackwardCritical => {
                let bn_suf = pre.product_suf_bn[product][op_idx] as f64 / pre.max_job_bn.max(1e-9);
                let density = ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0);
                let next_term = next_w_base * 0.30 * next_term_raw;
                (1.15 * bn_suf) + (0.45 * density) + (0.20 * scarcity_urg)
                    + next_term - (0.60 * end_n) - pop_pen
                    + (0.40 * job_bias) + flow_term + route_term + jitter
            }
            Rule::WeightedCompletion => {
                let work_n = rem_avg / pre.max_job_avg_work.max(1e-9);
                let wspt = if best_end > 0 { work_n / (best_end as f64 / pre.time_scale.max(1.0)).max(0.01) } else { work_n };
                let next_term = next_w_base * 0.20 * next_term_raw;
                (1.20 * wspt) + (0.30 * rem_min_n) + (0.15 * scarcity_urg)
                    + next_term - (0.40 * end_n) - pop_pen
                    + (0.35 * job_bias) + flow_term + route_term + jitter
            }
        }
    }

    fn construct_solution_conflict(
        challenge: &Challenge, pre: &Pre, rule: Rule, k: usize,
        target_mk: Option<u32>, rng: &mut SmallRng,
        job_bias: Option<&[f64]>, machine_penalty: Option<&[f64]>,
        route_pref: Option<&RoutePrefLite>, route_w: f64,
    ) -> Result<(Solution, u32)> {
        let num_jobs = challenge.num_jobs;
        let num_machines = challenge.num_machines;
        let mut job_next_op = vec![0usize; num_jobs];
        let mut job_ready_time = vec![0u32; num_jobs];
        let mut machine_avail = vec![0u32; num_machines];
        let mut machine_load = pre.machine_load0.clone();
        let mut job_schedule: Vec<Vec<(usize, u32)>> = pre.job_ops_len.iter().map(|&len| Vec::with_capacity(len)).collect();
        let mut remaining_ops = pre.total_ops;
        let mut time = 0u32;
        let mut demand: Vec<u16> = vec![0u16; num_machines];
        let mut raw_by_machine: Vec<Vec<RawCand>> = (0..num_machines).map(|_| Vec::with_capacity(12)).collect();
        let mut idle_machines: Vec<usize> = Vec::with_capacity(num_machines);

        while remaining_ops > 0 {
            loop {
                idle_machines.clear();
                for m in 0..num_machines { if machine_avail[m] <= time { idle_machines.push(m); } }
                if idle_machines.is_empty() { break; }
                for &m in &idle_machines { demand[m] = 0; raw_by_machine[m].clear(); }
                let progress = 1.0 - (remaining_ops as f64) / (pre.total_ops as f64).max(1.0);
                let cap_per_machine = if k == 0 { 12usize } else { (k + 6).min(12) };

                for job in 0..num_jobs {
                    let op_idx = job_next_op[job];
                    if op_idx >= pre.job_ops_len[job] || job_ready_time[job] > time { continue; }
                    let product = pre.job_products[job];
                    let op = &pre.product_ops[product][op_idx];
                    if op.flex == 0 || op.machines.is_empty() || op.min_pt >= INF { continue; }
                    let (best_end, second_end, best_cnt_total, best_cnt_idle) = best_second_and_counts(time, &machine_avail, op);
                    if best_end >= INF || best_cnt_idle == 0 { continue; }
                    let ops_rem = pre.job_ops_len[job] - op_idx;
                    let jb = job_bias.map(|v| v[job]).unwrap_or(0.0);
                    let flex_inv = 1.0 / (op.flex as f64).max(1.0);
                    let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
                    let regret = if second_end >= INF { pre.avg_op_min * 2.6 } else { (second_end - best_end) as f64 };
                    let regn = (regret / pre.avg_op_min.max(1.0)).clamp(0.0, 6.0);
                    let rigidity = (0.60 * flex_inv + 0.40 * scarcity_urg).clamp(0.0, 2.5);

                    for &(m, pt) in &op.machines {
                        if machine_avail[m] > time { continue; }
                        let end = time.saturating_add(pt);
                        if end != best_end { continue; }
                        demand[m] = demand[m].saturating_add(1);
                        let mp = machine_penalty.map(|v| v[m]).unwrap_or(0.0);
                        let jitter = if k > 0 { rng.gen::<f64>() * 1e-9 } else { 0.0 };
                        let base = score_candidate(pre, rule, job, product, op_idx, ops_rem, op, m, pt, time, target_mk, best_end, second_end, best_cnt_total, progress, jb, mp, machine_load[m], route_pref, route_w, jitter);
                        push_top_k_raw(&mut raw_by_machine[m], RawCand { job, machine: m, pt, base_score: base, rigidity, reg_n: regn }, cap_per_machine);
                    }
                }

                let denom = (idle_machines.len() as f64).max(1.0);
                let conflict_w = (0.09 + 0.26 * pre.jobshopness + 0.11 * pre.high_flex + 0.16 * (1.0 - (1.0 - (remaining_ops as f64) / (pre.total_ops as f64).max(1.0)))).clamp(0.05, 0.45);
                let conflict_scale = (0.90 + 0.40 * pre.flex_factor).clamp(0.85, 1.75);

                let mut best: Option<Cand> = None;
                let mut top: Vec<Cand> = if k > 0 { Vec::with_capacity(k) } else { Vec::new() };

                for &m in &idle_machines {
                    let dem = demand[m] as f64;
                    if dem <= 0.0 || raw_by_machine[m].is_empty() { continue; }
                    let dem_n = ((dem - 1.0) / denom).clamp(0.0, 2.5);
                    for rc in &raw_by_machine[m] {
                        let rig = rc.rigidity.clamp(0.0, 2.5);
                        let regc = rc.reg_n.clamp(0.0, 4.5);
                        let boost = conflict_w * conflict_scale * dem_n * (1.15 * rig + 0.85 * regc);
                        let c = Cand { job: rc.job, machine: rc.machine, pt: rc.pt, score: rc.base_score + boost };
                        if k == 0 { if best.map_or(true, |bb| c.score > bb.score) { best = Some(c); } }
                        else { push_top_k(&mut top, c, k); }
                    }
                }

                let chosen = if k == 0 { match best { Some(c) => c, None => break } }
                else { if top.is_empty() { break; } choose_from_top_weighted(rng, &top) };

                let job = chosen.job; let machine = chosen.machine; let pt = chosen.pt;
                let product = pre.job_products[job];
                let op_idx = job_next_op[job];
                let op = &pre.product_ops[product][op_idx];
                let (best_end_now, _, _, _) = best_second_and_counts(time, &machine_avail, op);
                let end_check = time.max(machine_avail[machine]).saturating_add(pt);
                if machine_avail[machine] > time || end_check != best_end_now { break; }
                let end_time = time.saturating_add(pt);
                job_schedule[job].push((machine, time));
                job_next_op[job] += 1;
                job_ready_time[job] = end_time;
                machine_avail[machine] = end_time;
                remaining_ops -= 1;
                if op.min_pt < INF && op.flex > 0 && !op.machines.is_empty() {
                    let delta = (op.min_pt as f64) / (op.flex as f64).max(1.0);
                    if delta > 0.0 { for &(mm, _) in &op.machines { let v = machine_load[mm] - delta; machine_load[mm] = if v > 0.0 { v } else { 0.0 }; } }
                }
                if remaining_ops == 0 { break; }
            }
            if remaining_ops == 0 { break; }
            let mut next_time: Option<u32> = None;
            for &t in &machine_avail { if t > time { next_time = Some(next_time.map_or(t, |b| b.min(t))); } }
            for j in 0..num_jobs { let op_idx = job_next_op[j]; if op_idx < pre.job_ops_len[j] && job_ready_time[j] > time { let t = job_ready_time[j]; next_time = Some(next_time.map_or(t, |b| b.min(t))); } }
            time = next_time.ok_or_else(|| anyhow!("Stalled: no next event"))?;
        }
        let mk = machine_avail.into_iter().max().unwrap_or(0);
        Ok((Solution { job_schedule }, mk))
    }

    #[inline]
    fn best_second_and_counts(time: u32, machine_avail: &[u32], op: &OpInfo) -> (u32, u32, usize, usize) {
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
    fn push_top_k(top: &mut Vec<Cand>, c: Cand, k: usize) {
        if k == 0 { return; }
        let mut pos = top.len();
        while pos > 0 && top[pos - 1].score < c.score { pos -= 1; }
        if pos >= k { return; }
        top.insert(pos, c);
        if top.len() > k { top.pop(); }
    }

    #[inline]
    fn push_top_k_raw(top: &mut Vec<RawCand>, c: RawCand, k: usize) {
        if k == 0 { return; }
        let mut pos = top.len();
        while pos > 0 && top[pos - 1].base_score < c.base_score { pos -= 1; }
        if pos >= k { return; }
        top.insert(pos, c);
        if top.len() > k { top.pop(); }
    }

    #[inline]
    fn choose_from_top_weighted(rng: &mut SmallRng, top: &[Cand]) -> Cand {
        if top.len() <= 1 { return top[0]; }
        let min_s = top.last().unwrap().score;
        let n = top.len().min(8);
        let mut w = [0.0f64; 8]; let mut sum = 0.0f64;
        for i in 0..n { let d = (top[i].score - min_s) + 1e-9; let wi = d * d; w[i] = wi; sum += wi; }
        if !(sum > 0.0) { return top[rng.gen_range(0..top.len())]; }
        let mut r = rng.gen::<f64>() * sum;
        for i in 0..n { r -= w[i]; if r <= 0.0 { return top[i]; } }
        top[n - 1]
    }

    #[inline]
    fn push_top_solutions(top: &mut Vec<(Solution, u32)>, sol: &Solution, mk: u32, cap: usize) {
        let pos = top.binary_search_by_key(&mk, |(_, m)| *m).unwrap_or_else(|e| e);
        top.insert(pos, (sol.clone(), mk));
        if top.len() > cap { top.truncate(cap); }
    }

    #[inline]
    fn flow_makespan(seq: &[usize], pt: &[Vec<u32>], comp: &mut [u32]) -> u32 {
        comp.fill(0);
        for &j in seq {
            let row = &pt[j];
            if row.is_empty() { continue; }
            comp[0] = comp[0].saturating_add(row[0]);
            for k in 1..row.len() { let v = comp[k].max(comp[k - 1]).saturating_add(row[k]); comp[k] = v; }
        }
        *comp.last().unwrap_or(&0)
    }

    #[inline]
    fn reentrant_makespan(seq: &[usize], route: &[usize], pt: &[Vec<u32>], mready: &mut [u32]) -> u32 {
        mready.fill(0); let mut mk = 0u32;
        for &j in seq {
            let row = &pt[j]; let mut prev = 0u32;
            for (op_idx, &m) in route.iter().enumerate() {
                let p = row[op_idx]; let st = prev.max(mready[m]); let end = st.saturating_add(p);
                mready[m] = end; prev = end;
            }
            if prev > mk { mk = prev; }
        }
        mk
    }

    fn build_disj_from_solution(pre: &Pre, challenge: &Challenge, sol: &Solution) -> Result<DisjSchedule> {
        let num_jobs = challenge.num_jobs; let num_machines = challenge.num_machines;
        let mut job_offsets = vec![0usize; num_jobs + 1];
        for j in 0..num_jobs { job_offsets[j + 1] = job_offsets[j] + pre.job_ops_len[j]; }
        let n = job_offsets[num_jobs];
        if n == 0 { return Err(anyhow!("No operations")); }
        let mut node_machine = vec![0usize; n]; let mut node_pt = vec![0u32; n];
        let mut node_job = vec![0usize; n]; let mut node_op = vec![0usize; n];
        let mut per_machine: Vec<Vec<(u32, usize)>> = vec![Vec::new(); num_machines];
        for job in 0..num_jobs {
            let expected = pre.job_ops_len[job];
            if sol.job_schedule[job].len() != expected { return Err(anyhow!("Invalid solution")); }
            let product = pre.job_products[job];
            for op_idx in 0..expected {
                let id = job_offsets[job] + op_idx;
                let (m, st) = sol.job_schedule[job][op_idx];
                let op = &pre.product_ops[product][op_idx];
                let pt = op.machines.iter().find(|&&(mm, _)| mm == m).map(|&(_, p)| p).ok_or_else(|| anyhow!("pt missing"))?;
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
            let base = job_offsets[job];
            for k in 0..pre.job_ops_len[job] { let id = base + k; if k + 1 < pre.job_ops_len[job] { job_succ[id] = id + 1; indeg_job[id + 1] = indeg_job[id + 1].saturating_add(1); } }
        }
        Ok(DisjSchedule { n, num_jobs, num_machines, job_offsets, job_succ, indeg_job, node_machine, node_pt, node_job, node_op, machine_seq })
    }

    fn eval_disj(ds: &DisjSchedule, buf: &mut EvalBuf) -> Option<(u32, usize)> {
        let n = ds.n;
        buf.indeg.clone_from_slice(&ds.indeg_job);
        buf.start.fill(0); buf.best_pred.fill(NONE_USIZE); buf.machine_succ.fill(NONE_USIZE); buf.stack.clear();
        for seq in &ds.machine_seq { if seq.len() <= 1 { continue; } for i in 0..(seq.len()-1) { let u = seq[i]; let v = seq[i+1]; buf.machine_succ[u] = v; buf.indeg[v] = buf.indeg[v].saturating_add(1); } }
        for i in 0..n { if buf.indeg[i] == 0 { buf.stack.push(i); } }
        let mut processed = 0usize; let mut mk = 0u32; let mut mk_node = 0usize;
        while let Some(u) = buf.stack.pop() {
            processed += 1;
            let end_u = buf.start[u].saturating_add(ds.node_pt[u]);
            if end_u > mk { mk = end_u; mk_node = u; }
            let js = ds.job_succ[u];
            if js != NONE_USIZE { if buf.start[js] < end_u { buf.start[js] = end_u; buf.best_pred[js] = u; } buf.indeg[js] = buf.indeg[js].saturating_sub(1); if buf.indeg[js] == 0 { buf.stack.push(js); } }
            let ms = buf.machine_succ[u];
            if ms != NONE_USIZE { if buf.start[ms] < end_u { buf.start[ms] = end_u; buf.best_pred[ms] = u; } buf.indeg[ms] = buf.indeg[ms].saturating_sub(1); if buf.indeg[ms] == 0 { buf.stack.push(ms); } }
        }
        if processed != n { return None; }
        Some((mk, mk_node))
    }

    fn disj_to_solution(pre: &Pre, ds: &DisjSchedule, start: &[u32]) -> Result<Solution> {
        let num_jobs = ds.num_jobs;
        let mut job_schedule: Vec<Vec<(usize, u32)>> = Vec::with_capacity(num_jobs);
        for j in 0..num_jobs {
            let len = pre.job_ops_len[j]; let mut v = Vec::with_capacity(len); let base = ds.job_offsets[j];
            for k in 0..len { let id = base + k; v.push((ds.node_machine[id], start[id])); }
            job_schedule.push(v);
        }
        Ok(Solution { job_schedule })
    }

    fn critical_block_move_local_search_ex(pre: &Pre, challenge: &Challenge, base_sol: &Solution, max_iters: usize, top_cands: usize, perturb_cycles: usize) -> Result<Option<(Solution, u32)>> {
        let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
        let mut buf = EvalBuf::new(ds.n);
        let mut crit = vec![false; ds.n];
        let mut cur_eval = match eval_disj(&ds, &mut buf) { Some(x) => x, None => return Ok(None) };
        let initial_mk = cur_eval.0;
        descent_phase(&mut ds, &mut buf, &mut crit, pre, &mut cur_eval, max_iters, top_cands);
        let Some((mk_after, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
        let mut global_best_mk = mk_after; let mut global_best_ds = ds.clone();
        let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (initial_mk as u64).wrapping_shl(16) ^ (ds.n as u64);
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
                    while bend + 1 < seq.len() { let x = seq[bend]; let y = seq[bend+1]; if !crit[y] { break; } let end_x = buf.start[x].saturating_add(ds.node_pt[x]); if buf.start[y] != end_x { break; } bend += 1; }
                    if bend > bstart { blocks.push((m, bstart, bend)); }
                    i = bend + 1;
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

    fn descent_phase(ds: &mut DisjSchedule, buf: &mut EvalBuf, crit: &mut Vec<bool>, _pre: &Pre, cur_eval: &mut (u32, usize), max_iters: usize, top_cands: usize) -> bool {
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
                    while bend + 1 < seq.len() { let x = seq[bend]; let y = seq[bend+1]; if !crit[y] { break; } let end_x = buf.start[x].saturating_add(ds.node_pt[x]); if buf.start[y] != end_x { break; } bend += 1; }
                    if bend > bstart {
                        let max_shift = bend - bstart;
                        for &sh in &[1usize, 2, max_shift] {
                            if sh == 0 || sh > max_shift { continue; }
                            let from = bstart; let to_after = bstart + sh;
                            if from < seq.len() && to_after <= seq.len() {
                                let tgt_idx = (bstart + sh).min(seq.len() - 1);
                                push_top_k_move_fs(&mut cands, MoveCand { kind: 0, m_from: m, from, m_to: m, to: to_after, new_pt: 0, score: buf.start[seq[tgt_idx]] }, top_cands);
                            }
                            let from2 = bend; let to_after2 = bend - sh;
                            push_top_k_move_fs(&mut cands, MoveCand { kind: 0, m_from: m, from: from2, m_to: m, to: to_after2, new_pt: 0, score: buf.start[seq[bend]] }, top_cands);
                        }
                        if bstart > 0 { push_top_k_move_fs(&mut cands, MoveCand { kind: 2, m_from: m, from: bstart-1, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bstart]] }, top_cands); }
                        if bend + 1 < seq.len() { push_top_k_move_fs(&mut cands, MoveCand { kind: 2, m_from: m, from: bend, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bend]] }, top_cands); }
                    }
                    i = bend + 1;
                }
            }
            if cands.is_empty() { break; }
            let mut best_cand: Option<MoveCand> = None; let mut best_mk = cur_mk;
            for cand in &cands {
                if cand.kind == 0 {
                    let m = cand.m_from; if m >= ds.num_machines || cand.from >= ds.machine_seq[m].len() { continue; }
                    let new_idx = apply_insert_fs(&mut ds.machine_seq[m], cand.from, cand.to);
                    if let Some((mk2, _)) = eval_disj(ds, buf) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                    let _ = apply_insert_fs(&mut ds.machine_seq[m], new_idx, cand.from);
                } else if cand.kind == 2 {
                    let m = cand.m_from; if m >= ds.num_machines || cand.from + 1 >= ds.machine_seq[m].len() { continue; }
                    ds.machine_seq[m].swap(cand.from, cand.from + 1);
                    if let Some((mk2, _)) = eval_disj(ds, buf) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                    ds.machine_seq[m].swap(cand.from, cand.from + 1);
                }
            }
            let Some(bc) = best_cand else { break };
            let mut accepted = false;
            if bc.kind == 0 {
                let m = bc.m_from; let new_idx = apply_insert_fs(&mut ds.machine_seq[m], bc.from, bc.to);
                if let Some(next_eval) = eval_disj(ds, buf) { if next_eval.0 < cur_mk { *cur_eval = next_eval; cur_mk = cur_eval.0; improved = true; accepted = true; } else { let _ = apply_insert_fs(&mut ds.machine_seq[m], new_idx, bc.from); } }
                else { let _ = apply_insert_fs(&mut ds.machine_seq[m], new_idx, bc.from); }
            } else if bc.kind == 2 {
                let m = bc.m_from;
                if m < ds.num_machines && bc.from + 1 < ds.machine_seq[m].len() {
                    ds.machine_seq[m].swap(bc.from, bc.from + 1);
                    if let Some(next_eval) = eval_disj(ds, buf) { if next_eval.0 < cur_mk { *cur_eval = next_eval; cur_mk = cur_eval.0; improved = true; accepted = true; } else { ds.machine_seq[m].swap(bc.from, bc.from + 1); } }
                    else { ds.machine_seq[m].swap(bc.from, bc.from + 1); }
                }
            }
            if !accepted { break; }
        }
        improved
    }

    #[inline]
    fn apply_insert_fs(seq: &mut Vec<usize>, from: usize, to_after_removal: usize) -> usize {
        if seq.is_empty() || from >= seq.len() { return from.min(seq.len().saturating_sub(1)); }
        let x = seq.remove(from); let t = to_after_removal.min(seq.len()); seq.insert(t, x); t
    }

    #[inline]
    fn push_top_k_move_fs(top: &mut Vec<MoveCand>, c: MoveCand, k: usize) {
        if k == 0 { return; }
        let mut pos = top.len();
        while pos > 0 && top[pos - 1].score < c.score { pos -= 1; }
        if pos >= k { return; }
        top.insert(pos, c);
        if top.len() > k { top.pop(); }
    }

    fn run_simple_greedy_baseline(challenge: &Challenge) -> Result<(Solution, u32)> {
        let num_jobs = challenge.num_jobs;
        let mut job_products = Vec::with_capacity(num_jobs);
        for (p, &cnt) in challenge.jobs_per_product.iter().enumerate() { for _ in 0..cnt { job_products.push(p); } }
        let job_ops_len: Vec<usize> = job_products.iter().map(|&p| challenge.product_processing_times[p].len()).collect();
        let job_total_work: Vec<f64> = job_products.iter().map(|&p| {
            challenge.product_processing_times[p].iter().map(|op| op.values().sum::<u32>() as f64 / op.len().max(1) as f64).sum()
        }).collect();
        run_greedy_rule_fs(challenge, &job_products, &job_ops_len, &job_total_work)
    }

    fn run_greedy_rule_fs(challenge: &Challenge, job_products: &[usize], job_ops_len: &[usize], job_total_work: &[f64]) -> Result<(Solution, u32)> {
        let num_jobs = challenge.num_jobs; let num_machines = challenge.num_machines;
        let mut job_next_op = vec![0usize; num_jobs]; let mut job_ready = vec![0u32; num_jobs];
        let mut machine_avail = vec![0u32; num_machines];
        let mut job_schedule: Vec<Vec<(usize, u32)>> = job_ops_len.iter().map(|&len| Vec::with_capacity(len)).collect();
        let mut job_work_left = job_total_work.to_vec();
        let mut remaining = job_ops_len.iter().sum::<usize>(); let mut time = 0u32;
        while remaining > 0 {
            let mut did_work = false;
            for m in 0..num_machines {
                if machine_avail[m] > time { continue; }
                let mut best_job: Option<usize> = None; let mut best_priority = f64::NEG_INFINITY;
                for j in 0..num_jobs {
                    if job_next_op[j] >= job_ops_len[j] || job_ready[j] > time { continue; }
                    let product = job_products[j]; let op_idx = job_next_op[j];
                    let op_times = &challenge.product_processing_times[product][op_idx];
                    let pt = match op_times.get(&m) { Some(&v) => v, None => continue };
                    let earliest = op_times.iter().map(|(&mm, &ppt)| time.max(machine_avail[mm]) + ppt).min().unwrap_or(u32::MAX);
                    if time + pt != earliest { continue; }
                    let priority = job_work_left[j];
                    if best_job.is_none() || priority > best_priority { best_priority = priority; best_job = Some(j); }
                }
                if let Some(j) = best_job {
                    let product = job_products[j]; let op_idx = job_next_op[j];
                    let op_times = &challenge.product_processing_times[product][op_idx];
                    let pt = op_times[&m]; let avg_pt = op_times.values().sum::<u32>() as f64 / op_times.len().max(1) as f64;
                    let st = time.max(machine_avail[m]); let end = st + pt;
                    job_schedule[j].push((m, st)); job_next_op[j] += 1; job_ready[j] = end; machine_avail[m] = end;
                    job_work_left[j] -= avg_pt; if job_work_left[j] < 0.0 { job_work_left[j] = 0.0; }
                    remaining -= 1; did_work = true;
                }
            }
            if remaining == 0 { break; }
            if !did_work {
                let mut next = u32::MAX;
                for &t in &machine_avail { if t > time && t < next { next = t; } }
                for j in 0..num_jobs { if job_next_op[j] < job_ops_len[j] && job_ready[j] > time && job_ready[j] < next { next = job_ready[j]; } }
                if next == u32::MAX { return Err(anyhow!("Greedy stuck")); }
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
        for j in 0..n { if a[j] <= b[j] { front.push((a[j], j)); } else { back.push((b[j], j)); } }
        front.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
        back.sort_unstable_by(|x, y| y.0.cmp(&x.0).then_with(|| x.1.cmp(&y.1)));
        let mut ord = Vec::with_capacity(n);
        for &(_, j) in &front { ord.push(j); }
        for &(_, j) in &back { ord.push(j); }
        ord
    }

    fn palmer_order(pt: &[Vec<u32>]) -> Vec<usize> {
        let n = pt.len(); let m = pt.first().map(|r| r.len()).unwrap_or(0);
        let mut jobs: Vec<(i64, usize)> = Vec::with_capacity(n);
        if m == 0 { return (0..n).collect(); }
        let mm = m as i64;
        for j in 0..n {
            let row = &pt[j]; let mut s: i64 = 0;
            for k in 0..m { let w = mm - 2 * (k as i64) - 1; s += w * (row[k] as i64); }
            jobs.push((s, j));
        }
        jobs.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        jobs.into_iter().map(|x| x.1).collect()
    }

    fn cds_orders(pt: &[Vec<u32>]) -> Vec<Vec<usize>> {
        let n = pt.len(); if n == 0 { return vec![]; }
        let m = pt[0].len(); if m <= 1 { return vec![(0..n).collect()]; }
        let mut totals = vec![0u32; n]; let mut prefix = vec![vec![0u32; m + 1]; n];
        for j in 0..n { let row = &pt[j]; let mut s = 0u32; prefix[j][0] = 0; for k in 0..m { s = s.saturating_add(row[k]); prefix[j][k+1] = s; } totals[j] = s; }
        let mut res: Vec<Vec<usize>> = Vec::with_capacity(m - 1);
        let mut a = vec![0u32; n]; let mut b = vec![0u32; n];
        for k in 1..m { for j in 0..n { a[j] = prefix[j][k]; b[j] = totals[j].saturating_sub(prefix[j][k]); } res.push(johnson_order_from_ab(&a, &b)); }
        res
    }

    fn route_is_unique(route: &[usize], num_machines: usize) -> bool {
        if route.is_empty() { return false; }
        let mut seen = vec![false; num_machines.max(1)];
        for &m in route { if m >= seen.len() || seen[m] { return false; } seen[m] = true; }
        true
    }

    #[derive(Default, Clone)]
    struct TaillardInsBuf { f: Vec<u32>, b: Vec<u32>, e: Vec<u32>, comp: Vec<u32> }
    impl TaillardInsBuf {
        fn ensure(&mut self, len: usize, m: usize) {
            let need = (len + 1) * m;
            if self.f.len() < need { self.f.resize(need, 0); }
            if self.b.len() < need { self.b.resize(need, 0); }
            if self.e.len() < m { self.e.resize(m, 0); }
            if self.comp.len() < m { self.comp.resize(m, 0); }
        }
    }

    thread_local! { static TL_TAILLARD: RefCell<TaillardInsBuf> = RefCell::new(TaillardInsBuf::default()); }

    fn taillard_best_insert_pos(seq: &[usize], job: usize, pt: &[Vec<u32>], m: usize, buf: &mut TaillardInsBuf) -> (usize, u32) {
        let l = seq.len(); if m == 0 { return (0, 0); }
        buf.ensure(l, m);
        let f = &mut buf.f; let b = &mut buf.b; let e = &mut buf.e;
        for k in 0..m { f[k] = 0; }
        for t in 1..=l { let jj = seq[t-1]; let row = &pt[jj]; let base = t*m; let prev = (t-1)*m; f[base] = f[prev].saturating_add(row[0]); for k in 1..m { f[base+k] = f[base+k-1].max(f[prev+k]).saturating_add(row[k]); } }
        let base_l = l * m;
        for k in 0..m { b[base_l + k] = 0; }
        for t in (0..l).rev() { let jj = seq[t]; let row = &pt[jj]; let base = t*m; let next = (t+1)*m; b[base+(m-1)] = b[next+(m-1)].saturating_add(row[m-1]); if m >= 2 { for kk in 0..(m-1) { let k = (m-2)-kk; b[base+k] = b[base+k+1].max(b[next+k]).saturating_add(row[k]); } } }
        let prow = &pt[job]; let mut best_pos = 0usize; let mut best_mk = u32::MAX;
        for pos in 0..=l {
            let fb = pos * m; e[0] = f[fb].saturating_add(prow[0]);
            for k in 1..m { e[k] = e[k-1].max(f[fb+k]).saturating_add(prow[k]); }
            let mut mk = 0u32; for k in 0..m { mk = mk.max(e[k].saturating_add(b[fb+k])); }
            if mk < best_mk { best_mk = mk; best_pos = pos; }
        }
        (best_pos, best_mk)
    }

    fn improve_perm_seq_taillard(seq: &mut Vec<usize>, pt: &[Vec<u32>], rounds: usize, buf: &mut TaillardInsBuf) {
        let m = pt.first().map(|r| r.len()).unwrap_or(0);
        if seq.len() <= 2 || m == 0 { return; }
        buf.ensure(seq.len(), m);
        let mut cur_mk = flow_makespan(seq, pt, &mut buf.comp[..m]);
        for _ in 0..rounds {
            let mut improved_any = false;
            for i0 in 0..seq.len() { let job = seq.remove(i0); let (pos, mk) = taillard_best_insert_pos(seq, job, pt, m, buf); seq.insert(pos, job); if mk < cur_mk { cur_mk = mk; improved_any = true; } }
            if !improved_any { break; }
        }
    }

    fn neh_build_seq(order: &[usize], route: &[usize], pt: &[Vec<u32>], num_machines: usize) -> Vec<usize> {
        let unique = route_is_unique(route, num_machines);
        if unique {
            let m = route.len(); if m == 0 { return vec![]; }
            return TL_TAILLARD.with(|cell| {
                let mut buf = cell.borrow_mut(); let mut seq: Vec<usize> = Vec::with_capacity(order.len());
                for &j in order { if seq.is_empty() { seq.push(j); continue; } let (pos, _mk) = taillard_best_insert_pos(&seq, j, pt, m, &mut buf); seq.insert(pos, j); }
                seq
            });
        }
        let mut seq: Vec<usize> = Vec::with_capacity(order.len()); let mut tmp: Vec<usize> = Vec::with_capacity(order.len()); let mut mready = vec![0u32; num_machines];
        for &j in order {
            if seq.is_empty() { seq.push(j); continue; }
            let mut best_mk = u32::MAX; let mut best_pos = 0usize;
            for pos in 0..=seq.len() { tmp.clear(); tmp.extend_from_slice(&seq[..pos]); tmp.push(j); tmp.extend_from_slice(&seq[pos..]); let mk = reentrant_makespan(&tmp, route, pt, &mut mready); if mk < best_mk { best_mk = mk; best_pos = pos; } }
            seq.insert(best_pos, j);
        }
        seq
    }

    fn fs_improve_reentrant_seq(seq: &mut Vec<usize>, route: &[usize], pt: &[Vec<u32>], num_machines: usize) {
        if seq.len() <= 2 || route.is_empty() { return; }
        if route_is_unique(route, num_machines) {
            TL_TAILLARD.with(|cell| { let mut buf = cell.borrow_mut(); improve_perm_seq_taillard(seq, pt, 8, &mut buf); });
            return;
        }
        let mut mready = vec![0u32; num_machines]; let mut cur_mk = reentrant_makespan(seq, route, pt, &mut mready);
        for _ in 0..8usize {
            let mut improved_any = false;
            for i0 in 0..seq.len() {
                let j = seq.remove(i0); let mut best_mk = u32::MAX; let mut best_pos = 0usize;
                for pos in 0..=seq.len() { seq.insert(pos, j); let mk = reentrant_makespan(seq, route, pt, &mut mready); if mk < best_mk { best_mk = mk; best_pos = pos; } seq.remove(pos); }
                seq.insert(best_pos, j); if best_mk < cur_mk { cur_mk = best_mk; improved_any = true; }
            }
            if !improved_any { break; }
        }
    }

    fn build_perm_solution_from_seq(seq: &[usize], route: &[usize], pt: &[Vec<u32>], num_jobs: usize, num_machines: usize) -> Solution {
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

    fn order_from_solution_first_op_start(sol: &Solution, num_jobs: usize) -> Vec<usize> {
        let mut v: Vec<(u32, usize)> = Vec::with_capacity(num_jobs);
        for j in 0..num_jobs { if let Some(t) = sol.job_schedule.get(j).and_then(|ops| ops.first()).map(|x| x.1) { v.push((t, j)); } }
        v.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let mut seen = vec![false; num_jobs]; let mut ord: Vec<usize> = Vec::with_capacity(num_jobs);
        for &(_, j) in &v { if j < num_jobs && !seen[j] { seen[j] = true; ord.push(j); } }
        for j in 0..num_jobs { if !seen[j] { ord.push(j); } }
        ord
    }

    fn neh_best_sequence(pre: &Pre, num_jobs: usize, num_machines: usize) -> Result<Vec<usize>> {
        let route = pre.flow_route.as_ref().ok_or_else(|| anyhow!("No flow route"))?;
        let pt = pre.flow_pt_by_job.as_ref().ok_or_else(|| anyhow!("No flow pt"))?;
        let ops = route.len(); if ops == 0 || pt.len() != num_jobs { return Err(anyhow!("Invalid flow data")); }
        let mut candidates: Vec<Vec<usize>> = Vec::new();
        { let mut jobs: Vec<usize> = (0..num_jobs).collect(); jobs.sort_unstable_by(|&a, &b| { let sa: u32 = pt[a].iter().copied().sum(); let sb: u32 = pt[b].iter().copied().sum(); sb.cmp(&sa).then_with(|| a.cmp(&b)) }); candidates.push(jobs); }
        candidates.push(palmer_order(pt));
        for o in cds_orders(pt) { if o.len() == num_jobs { candidates.push(o); } }
        let unique = route_is_unique(route, num_machines); let mut best_seq: Vec<usize> = (0..num_jobs).collect(); let mut best_mk: u32 = u32::MAX;
        if unique {
            TL_TAILLARD.with(|cell| { let mut buf = cell.borrow_mut(); let m = ops;
                for ord in candidates.iter() { if ord.len() != num_jobs { continue; } let mut seq = neh_build_seq(ord, route, pt, num_machines); improve_perm_seq_taillard(&mut seq, pt, 8, &mut buf); buf.ensure(seq.len(), m); let mk = flow_makespan(&seq, pt, &mut buf.comp[..m]); if mk < best_mk { best_mk = mk; best_seq = seq; } }
            });
            return Ok(best_seq);
        }
        let mut mready = vec![0u32; num_machines];
        for ord in candidates.iter() { if ord.len() != num_jobs { continue; } let mut seq = neh_build_seq(ord, route, pt, num_machines); fs_improve_reentrant_seq(&mut seq, route, pt, num_machines); let mk = reentrant_makespan(&seq, route, pt, &mut mready); if mk < best_mk { best_mk = mk; best_seq = seq; } }
        Ok(best_seq)
    }

    fn iterated_greedy_search(init: &[usize], pt: &[Vec<u32>], iters: usize, d: usize, rng: &mut SmallRng) -> Vec<usize> {
        let n = init.len(); if n <= 2 { return init.to_vec(); }
        let m = pt.first().map(|r| r.len()).unwrap_or(0); if m == 0 { return init.to_vec(); }
        let mut buf = TaillardInsBuf::default(); buf.ensure(n, m);
        let mut cur = init.to_vec(); let mut best = cur.clone();
        let mut cur_mk = flow_makespan(&cur, pt, &mut buf.comp[..m]); let mut best_mk = cur_mk;
        let mut temp = (cur_mk as f64) * 0.10 + 1.0;
        let dd = d.clamp(2, n.saturating_sub(1)); let its = iters.max(1);
        let mut idxs: Vec<usize> = Vec::with_capacity(dd); let mut removed: Vec<usize> = Vec::with_capacity(dd);
        for _ in 0..its {
            idxs.clear();
            while idxs.len() < dd { let x = rng.gen_range(0..n); if !idxs.iter().any(|&y| y == x) { idxs.push(x); } }
            idxs.sort_unstable(); removed.clear();
            let mut partial = cur.clone();
            for &ix in idxs.iter().rev() { if ix < partial.len() { removed.push(partial.remove(ix)); } }
            removed.shuffle(rng);
            for &j in &removed { let (pos, _mk) = taillard_best_insert_pos(&partial, j, pt, m, &mut buf); partial.insert(pos, j); }
            let mut cand = partial; let mut cand_mk = flow_makespan(&cand, pt, &mut buf.comp[..m]);
            if cand.len() >= 2 { for i in 0..(cand.len()-1) { cand.swap(i, i+1); let mk2 = flow_makespan(&cand, pt, &mut buf.comp[..m]); if mk2 <= cand_mk { cand_mk = mk2; } else { cand.swap(i, i+1); } } }
            if cand_mk < best_mk { best_mk = cand_mk; best = cand.clone(); }
            if cand_mk <= cur_mk { cur = cand; cur_mk = cand_mk; }
            else { let delta = (cand_mk - cur_mk) as f64; let prob = (-delta / temp).exp(); if rng.gen::<f64>() < prob { cur = cand; cur_mk = cand_mk; } }
            temp = (temp * 0.995).max(1.0);
        }
        best
    }

    fn strict_makespan(challenge: &Challenge, pre: &Pre, rank: &[usize]) -> Result<u32> {
        let route = pre.flow_route.as_ref().ok_or_else(|| anyhow!("flow_route missing"))?;
        let num_jobs = challenge.num_jobs; let num_machines = challenge.num_machines;
        let mut job_next_op = vec![0usize; num_jobs]; let mut job_ready = vec![0u32; num_jobs]; let mut machine_avail = vec![0u32; num_machines];
        let mut remaining_ops = pre.total_ops;
        let mut future: Vec<BinaryHeap<Reverse<(u32, usize, usize)>>> = (0..num_machines).map(|_| BinaryHeap::new()).collect();
        let mut avail: Vec<BinaryHeap<Reverse<(usize, usize)>>> = (0..num_machines).map(|_| BinaryHeap::new()).collect();
        for job in 0..num_jobs { if pre.job_ops_len[job] == 0 { continue; } let m = route[0]; future[m].push(Reverse((0u32, rank[job], job))); }
        let mut next_time: Vec<Option<u32>> = vec![None; num_machines];
        let mut machine_events: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();
        let compute_next_time = |m: usize, machine_avail: &Vec<u32>, future: &Vec<BinaryHeap<Reverse<(u32, usize, usize)>>>, avail: &Vec<BinaryHeap<Reverse<(usize, usize)>>>| -> Option<u32> {
            if !avail[m].is_empty() { return Some(machine_avail[m]); }
            if let Some(Reverse((release, _, _))) = future[m].peek().copied() { return Some(machine_avail[m].max(release)); }
            None
        };
        for m in 0..num_machines { let t = compute_next_time(m, &machine_avail, &future, &avail); next_time[m] = t; if let Some(tt) = t { machine_events.push(Reverse((tt, m))); } }
        let mut makespan = 0u32;
        while remaining_ops > 0 {
            let Reverse((t, m)) = machine_events.pop().ok_or_else(|| anyhow!("stalled"))?;
            if next_time[m] != Some(t) || machine_avail[m] > t { continue; }
            while let Some(Reverse((release, _, job))) = future[m].peek().copied() { if release > t { break; } future[m].pop(); avail[m].push(Reverse((rank[job], job))); }
            let Some(Reverse((_, job))) = avail[m].pop() else {
                let nt = compute_next_time(m, &machine_avail, &future, &avail); next_time[m] = nt;
                if let Some(tt) = nt { machine_events.push(Reverse((tt, m))); } continue;
            };
            let op_idx = job_next_op[job];
            if op_idx >= pre.job_ops_len[job] { return Err(anyhow!("job complete but popped")); }
            if route[op_idx] != m { return Err(anyhow!("route mismatch")); }
            let start = t.max(job_ready[job]).max(machine_avail[m]);
            if start != t { avail[m].push(Reverse((rank[job], job))); let nt = compute_next_time(m, &machine_avail, &future, &avail); next_time[m] = nt; if let Some(tt) = nt { machine_events.push(Reverse((tt, m))); } continue; }
            let product = pre.job_products[job];
            let ptv = *challenge.product_processing_times[product][op_idx].get(&m).ok_or_else(|| anyhow!("missing pt"))?;
            let end = start.saturating_add(ptv);
            job_next_op[job] += 1; job_ready[job] = end; machine_avail[m] = end; remaining_ops -= 1; makespan = makespan.max(end);
            if job_next_op[job] < pre.job_ops_len[job] { let next_op = job_next_op[job]; let m2 = route[next_op]; future[m2].push(Reverse((end, rank[job], job))); let nt2 = compute_next_time(m2, &machine_avail, &future, &avail); next_time[m2] = nt2; if let Some(tt) = nt2 { machine_events.push(Reverse((tt, m2))); } }
            let nt = compute_next_time(m, &machine_avail, &future, &avail); next_time[m] = nt; if let Some(tt) = nt { machine_events.push(Reverse((tt, m))); }
        }
        Ok(makespan)
    }

    fn strict_simulate(challenge: &Challenge, pre: &Pre, rank: &[usize]) -> Result<(Solution, u32)> {
        let route = pre.flow_route.as_ref().ok_or_else(|| anyhow!("flow_route missing"))?;
        let num_jobs = challenge.num_jobs; let num_machines = challenge.num_machines;
        let mut job_next_op = vec![0usize; num_jobs]; let mut job_ready = vec![0u32; num_jobs]; let mut machine_avail = vec![0u32; num_machines];
        let mut job_schedule: Vec<Vec<(usize, u32)>> = pre.job_ops_len.iter().map(|&len| Vec::with_capacity(len)).collect();
        let mut remaining_ops = pre.total_ops;
        let mut future: Vec<BinaryHeap<Reverse<(u32, usize, usize)>>> = (0..num_machines).map(|_| BinaryHeap::new()).collect();
        let mut avail: Vec<BinaryHeap<Reverse<(usize, usize)>>> = (0..num_machines).map(|_| BinaryHeap::new()).collect();
        for job in 0..num_jobs { if pre.job_ops_len[job] == 0 { continue; } let m = route[0]; future[m].push(Reverse((0u32, rank[job], job))); }
        let mut next_time: Vec<Option<u32>> = vec![None; num_machines];
        let mut machine_events: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();
        let compute_next_time = |m: usize, machine_avail: &Vec<u32>, future: &Vec<BinaryHeap<Reverse<(u32, usize, usize)>>>, avail: &Vec<BinaryHeap<Reverse<(usize, usize)>>>| -> Option<u32> {
            if !avail[m].is_empty() { return Some(machine_avail[m]); }
            if let Some(Reverse((release, _, _))) = future[m].peek().copied() { return Some(machine_avail[m].max(release)); }
            None
        };
        for m in 0..num_machines { let t = compute_next_time(m, &machine_avail, &future, &avail); next_time[m] = t; if let Some(tt) = t { machine_events.push(Reverse((tt, m))); } }
        let mut makespan = 0u32;
        while remaining_ops > 0 {
            let Reverse((t, m)) = machine_events.pop().ok_or_else(|| anyhow!("stalled"))?;
            if next_time[m] != Some(t) || machine_avail[m] > t { continue; }
            while let Some(Reverse((release, _, job))) = future[m].peek().copied() { if release > t { break; } future[m].pop(); avail[m].push(Reverse((rank[job], job))); }
            let Some(Reverse((_, job))) = avail[m].pop() else {
                let nt = compute_next_time(m, &machine_avail, &future, &avail); next_time[m] = nt;
                if let Some(tt) = nt { machine_events.push(Reverse((tt, m))); } continue;
            };
            let op_idx = job_next_op[job];
            if op_idx >= pre.job_ops_len[job] { return Err(anyhow!("job complete")); }
            if route[op_idx] != m { return Err(anyhow!("route mismatch")); }
            let start = t.max(job_ready[job]).max(machine_avail[m]);
            if start != t { avail[m].push(Reverse((rank[job], job))); let nt = compute_next_time(m, &machine_avail, &future, &avail); next_time[m] = nt; if let Some(tt) = nt { machine_events.push(Reverse((tt, m))); } continue; }
            let product = pre.job_products[job];
            let ptv = *challenge.product_processing_times[product][op_idx].get(&m).ok_or_else(|| anyhow!("missing pt"))?;
            let end = start.saturating_add(ptv);
            job_schedule[job].push((m, start)); job_next_op[job] += 1; job_ready[job] = end; machine_avail[m] = end; remaining_ops -= 1; makespan = makespan.max(end);
            if job_next_op[job] < pre.job_ops_len[job] { let next_op = job_next_op[job]; let m2 = route[next_op]; future[m2].push(Reverse((end, rank[job], job))); let nt2 = compute_next_time(m2, &machine_avail, &future, &avail); next_time[m2] = nt2; if let Some(tt) = nt2 { machine_events.push(Reverse((tt, m2))); } }
            let nt = compute_next_time(m, &machine_avail, &future, &avail); next_time[m] = nt; if let Some(tt) = nt { machine_events.push(Reverse((tt, m))); }
        }
        Ok((Solution { job_schedule }, makespan))
    }

    fn strict_best_by_order_search(challenge: &Challenge, pre: &Pre, passes: usize) -> Result<(Solution, u32)> {
        if pre.flow_route.is_none() || pre.flex_avg > 1.25 { return Err(anyhow!("not strict-like")); }
        let n = challenge.num_jobs;
        let pt_stage: Vec<Vec<u32>> = if let Some(pt) = pre.flow_pt_by_job.as_ref() { pt.clone() } else {
            let mut tmp = vec![vec![0u32; pre.max_ops.max(1)]; n];
            for j in 0..n { let p = pre.job_products[j]; let len = pre.job_ops_len[j]; for k in 0..len { tmp[j][k] = pre.product_ops[p][k].machines.first().map(|x| x.1).unwrap_or(0); } tmp[j].truncate(len); }
            tmp
        };
        let mut cand_orders: Vec<Vec<usize>> = Vec::new();
        { let mut lpt: Vec<usize> = (0..n).collect(); lpt.sort_unstable_by(|&a, &b| { let sa: u32 = pt_stage[a].iter().copied().sum(); let sb: u32 = pt_stage[b].iter().copied().sum(); sb.cmp(&sa).then_with(|| a.cmp(&b)) }); cand_orders.push(lpt); }
        { let mut spt: Vec<usize> = (0..n).collect(); spt.sort_unstable_by(|&a, &b| { let sa: u32 = pt_stage[a].iter().copied().sum(); let sb: u32 = pt_stage[b].iter().copied().sum(); sa.cmp(&sb).then_with(|| a.cmp(&b)) }); cand_orders.push(spt); }
        cand_orders.push(palmer_order(&pt_stage));
        for o in cds_orders(&pt_stage) { if o.len() == n { cand_orders.push(o); } }
        { let mut seed = challenge.seed; seed[0] ^= 0x3C; let mut rng = SmallRng::from_seed(seed); for _ in 0..100usize { let mut r: Vec<usize> = (0..n).collect(); r.shuffle(&mut rng); cand_orders.push(r); } }
        let mut rank = vec![0usize; n]; let mut best_mk = u32::MAX; let mut best_order: Vec<usize> = (0..n).collect();
        for ord in cand_orders.iter() { if ord.len() != n { continue; } for (pos, &j) in ord.iter().enumerate() { rank[j] = pos; } let mk = strict_makespan(challenge, pre, &rank)?; if mk < best_mk { best_mk = mk; best_order.clone_from(ord); } }
        let max_passes = passes.max(1).min(6);
        let mut cand_order: Vec<usize> = vec![0usize; n];
        for _ in 0..max_passes.min(2) {
            let mut improved = false;
            for i in 0..n {
                let job = best_order[i]; let mut best_pos = i; let mut best_local_mk = best_mk;
                for pos in 0..n {
                    if pos == i { continue; }
                    if pos < i { cand_order[..pos].copy_from_slice(&best_order[..pos]); cand_order[pos] = job; cand_order[pos+1..=i].copy_from_slice(&best_order[pos..i]); cand_order[i+1..].copy_from_slice(&best_order[i+1..]); }
                    else { cand_order[..i].copy_from_slice(&best_order[..i]); cand_order[i..pos].copy_from_slice(&best_order[i+1..=pos]); cand_order[pos] = job; cand_order[pos+1..].copy_from_slice(&best_order[pos+1..]); }
                    for (p, &jj) in cand_order.iter().enumerate() { rank[jj] = p; }
                    let mk = strict_makespan(challenge, pre, &rank)?;
                    if mk < best_local_mk { best_local_mk = mk; best_pos = pos; }
                }
                if best_local_mk < best_mk { best_mk = best_local_mk; if best_pos < i { best_order[best_pos..=i].rotate_right(1); } else if best_pos > i { best_order[i..=best_pos].rotate_left(1); } improved = true; }
            }
            if !improved { break; }
        }
        let mut order = best_order.clone();
        for (pos, &j) in order.iter().enumerate() { rank[j] = pos; }
        for _ in 0..max_passes {
            let mut improved = false;
            for i in 0..(n.saturating_sub(1)) {
                order.swap(i, i+1); rank[order[i]] = i; rank[order[i+1]] = i+1;
                let mk = strict_makespan(challenge, pre, &rank)?;
                if mk < best_mk { best_mk = mk; improved = true; best_order = order.clone(); }
                else { order.swap(i, i+1); rank[order[i]] = i; rank[order[i+1]] = i+1; }
            }
            if !improved { break; }
        }
        order = best_order.clone();
        for (pos, &j) in order.iter().enumerate() { rank[j] = pos; }
        { let mut seed = challenge.seed; seed[0] ^= 0xA5; let mut rng = SmallRng::from_seed(seed); let swap_budget = (n * 8).clamp(160, 600); for _ in 0..swap_budget { let i = rng.gen_range(0..n); let j = rng.gen_range(0..n); if i == j { continue; } order.swap(i, j); rank[order[i]] = i; rank[order[j]] = j; let mk = strict_makespan(challenge, pre, &rank)?; if mk < best_mk { best_mk = mk; best_order = order.clone(); } else { order.swap(i, j); rank[order[i]] = i; rank[order[j]] = j; } } }
        order = best_order.clone(); for (pos, &j) in order.iter().enumerate() { rank[j] = pos; }
        if n >= 2 { let max_seg = 5usize.min(n); for _ in 0..2 { let mut improved = false; for seg_len in 2..=max_seg { for start in 0..=(n-seg_len) { order[start..start+seg_len].reverse(); for k in start..start+seg_len { rank[order[k]] = k; } let mk = strict_makespan(challenge, pre, &rank)?; if mk < best_mk { best_mk = mk; best_order = order.clone(); improved = true; } else { order[start..start+seg_len].reverse(); for k in start..start+seg_len { rank[order[k]] = k; } } } } if !improved { break; } } }
        for (pos, &j) in best_order.iter().enumerate() { rank[j] = pos; }
        let (best_sol, mk2) = strict_simulate(challenge, pre, &rank)?;
        Ok((best_sol, if mk2 != best_mk { mk2 } else { best_mk }))
    }

    #[allow(dead_code)]
    fn lns_disj(
        pre: &Pre,
        challenge: &Challenge,
        base_sol: &Solution,
        iters: usize,
    ) -> Result<Option<(Solution, u32)>> {
        let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
        let mut buf = EvalBuf::new(ds.n);
        let Some(init_eval) = eval_disj(&ds, &mut buf) else { return Ok(None) };
        let mut best_mk = init_eval.0;
        let mut best_ds = ds.clone();
        let mut cur_eval = init_eval;

        let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15)
            ^ (best_mk as u64).wrapping_shl(16) ^ (ds.n as u64);

        for _ in 0..iters {
            pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);

            let machines_with_jobs: Vec<usize> = (0..ds.num_machines)
                .filter(|&m| ds.machine_seq[m].len() >= 2)
                .collect();
            if machines_with_jobs.is_empty() { break; }
            let m = machines_with_jobs[(pseed as usize) % machines_with_jobs.len()];
            let seq_len = ds.machine_seq[m].len();

            pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
            let move_type = (pseed as usize) % 3;

            match move_type {
                0 => {
                    if seq_len < 2 { continue; }
                    let max_seg = seq_len.min(6);
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    let seg_len = 2 + (pseed as usize) % (max_seg - 1);
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    let start = (pseed as usize) % (seq_len - seg_len + 1);
                    ds.machine_seq[m][start..start+seg_len].reverse();
                    if let Some(ev) = eval_disj(&ds, &mut buf) {
                        if ev.0 < best_mk { best_mk = ev.0; best_ds = ds.clone(); cur_eval = ev; }
                        else if ev.0 <= cur_eval.0 { cur_eval = ev; }
                        else { ds.machine_seq[m][start..start+seg_len].reverse(); }
                    } else { ds.machine_seq[m][start..start+seg_len].reverse(); }
                }
                1 => {
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    let from = (pseed as usize) % seq_len;
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    let to = (pseed as usize) % seq_len;
                    if from == to { continue; }
                    let job = ds.machine_seq[m].remove(from);
                    let ins = to.min(ds.machine_seq[m].len());
                    ds.machine_seq[m].insert(ins, job);
                    if let Some(ev) = eval_disj(&ds, &mut buf) {
                        if ev.0 < best_mk { best_mk = ev.0; best_ds = ds.clone(); cur_eval = ev; }
                        else if ev.0 <= cur_eval.0 { cur_eval = ev; }
                        else { ds.machine_seq[m].remove(ins); ds.machine_seq[m].insert(from, job); }
                    } else { ds.machine_seq[m].remove(ins); ds.machine_seq[m].insert(from, job); }
                }
                _ => {
                    if seq_len < 3 { continue; }
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    let i = (pseed as usize) % seq_len;
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    let j = (pseed as usize) % seq_len;
                    if i == j { continue; }
                    ds.machine_seq[m].swap(i, j);
                    if let Some(ev) = eval_disj(&ds, &mut buf) {
                        if ev.0 < best_mk { best_mk = ev.0; best_ds = ds.clone(); cur_eval = ev; }
                        else if ev.0 <= cur_eval.0 { cur_eval = ev; }
                        else { ds.machine_seq[m].swap(i, j); }
                    } else { ds.machine_seq[m].swap(i, j); }
                }
            }
        }

        if best_mk >= init_eval.0 { return Ok(None); }
        ds = best_ds;
        let Some((mk_final, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
        let sol = disj_to_solution(pre, &ds, &buf.start)?;
        Ok(Some((sol, mk_final)))
    }

    pub fn solve(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        pre: &Pre,
        greedy_sol: Solution,
        greedy_mk: u32,
        effort: &EffortConfig,
    ) -> Result<()> {
        let mut best_sol = greedy_sol;
        let mut best_mk = greedy_mk;
        let mut top_solutions: Vec<(Solution, u32)> = Vec::new();
        push_top_solutions(&mut top_solutions, &best_sol, best_mk, 5);

        let mut strict_sol: Option<(Solution, u32)> = None;
        if pre.flow_route.is_some() && pre.flex_avg <= 1.25 {
            if let Ok((sol, mk)) = strict_best_by_order_search(challenge, pre, 6) {
                strict_sol = Some((sol.clone(), mk));
                if mk <= best_mk { best_mk = mk; best_sol = sol; save_solution(&best_sol)?; push_top_solutions(&mut top_solutions, &best_sol, best_mk, 5); }
            }
        }

        if let (Some(route), Some(pt)) = (&pre.flow_route, &pre.flow_pt_by_job) {
            if false {
            if let Ok(neh_seq) = neh_best_sequence(pre, challenge.num_jobs, challenge.num_machines) {
                let perm_sol = build_perm_solution_from_seq(&neh_seq, route, pt, challenge.num_jobs, challenge.num_machines);
                if let Ok(mk) = challenge.evaluate_makespan(&perm_sol) {
                    if mk <= best_mk { best_mk = mk; best_sol = perm_sol.clone(); save_solution(&best_sol)?; }
                    push_top_solutions(&mut top_solutions, &perm_sol, mk, 5);
                }
                if false && pre.flex_avg <= 1.25 {
                    let mut rank = vec![challenge.num_jobs; challenge.num_jobs];
                    for (pos, &j) in neh_seq.iter().enumerate() { if j < challenge.num_jobs { rank[j] = pos; } }
                    if let Ok((ssol, _)) = strict_simulate(challenge, pre, &rank) {
                        if let Ok(mk) = challenge.evaluate_makespan(&ssol) {
                            if mk <= best_mk { best_mk = mk; best_sol = ssol.clone(); save_solution(&best_sol)?; } push_top_solutions(&mut top_solutions, &ssol, mk, 5);
                        }
                    }
                }
                let unique = route_is_unique(route, challenge.num_machines);
                if false && unique && !neh_seq.is_empty() && route.len() == pt[neh_seq[0]].len() {
                    let mut starts: Vec<Vec<usize>> = Vec::new();
                    starts.push(neh_seq.clone());
                    if let Some((s, _mk)) = &strict_sol { starts.push(order_from_solution_first_op_start(s, challenge.num_jobs)); }
                    starts.push(order_from_solution_first_op_start(&best_sol, challenge.num_jobs));
                    let mut uniq: Vec<Vec<usize>> = Vec::new();
                    for ord in starts { if ord.len() != challenge.num_jobs { continue; } let mut ok = true; for u in &uniq { if *u == ord { ok = false; break; } } if ok { uniq.push(ord); } if uniq.len() >= 3 { break; } }
                    let per = 50usize; let d = 4usize;
                    let mut best_ig_seq = neh_seq;
                    let mut seed = challenge.seed; seed[0] ^= 0x6B; let mut rng = SmallRng::from_seed(seed);
                    TL_TAILLARD.with(|cell| {
                        let mut buf = cell.borrow_mut(); let m = route.len(); buf.ensure(best_ig_seq.len(), m);
                        let mk0 = flow_makespan(&best_ig_seq, pt, &mut buf.comp[..m]); let mut best_ig_mk = mk0;
                        for start_seq in uniq.iter() { let cand_seq = iterated_greedy_search(start_seq, pt, per, d, &mut rng); buf.ensure(cand_seq.len(), m); let mk = flow_makespan(&cand_seq, pt, &mut buf.comp[..m]); if mk < best_ig_mk { best_ig_mk = mk; best_ig_seq = cand_seq; } }
                    });
                    let ig_perm_sol = build_perm_solution_from_seq(&best_ig_seq, route, pt, challenge.num_jobs, challenge.num_machines);
                    if let Ok(mk) = challenge.evaluate_makespan(&ig_perm_sol) { if mk <= best_mk { best_mk = mk; best_sol = ig_perm_sol.clone(); save_solution(&best_sol)?; } push_top_solutions(&mut top_solutions, &ig_perm_sol, mk, 5); }
                }
            } else if let Ok(sol) = { let route = route; let pt = pt; let seq = neh_best_sequence(pre, challenge.num_jobs, challenge.num_machines); seq.map(|s| build_perm_solution_from_seq(&s, route, pt, challenge.num_jobs, challenge.num_machines)) } {
                if let Ok(mk) = challenge.evaluate_makespan(&sol) { if mk <= best_mk { best_mk = mk; best_sol = sol.clone(); save_solution(&best_sol)?; } push_top_solutions(&mut top_solutions, &sol, mk, 5); }
            }
            } // end if false
        }

        let flow_is_reentrant = pre.flow_route.is_some()
            && !route_is_unique(pre.flow_route.as_deref().unwrap_or(&[]), challenge.num_machines)
            && pre.flex_avg <= 1.25;

        if flow_is_reentrant {
            let mut seed = challenge.seed; seed[0] ^= 0xF1; let mut rng = SmallRng::from_seed(seed);
            let grasp_rules = [Rule::BnHeavy, Rule::MostWork, Rule::EndTight, Rule::ShortestProc, Rule::LeastFlex, Rule::CriticalPath, Rule::Regret, Rule::EarliestStart, Rule::MachineBalance, Rule::SlackRatio, Rule::BackwardCritical, Rule::WeightedCompletion];
            let num_restarts = effort.num_restarts;
            for r in 0..num_restarts {
                let rule = if r < grasp_rules.len() { grasp_rules[r] } else { grasp_rules[rng.gen_range(0..grasp_rules.len())] };
                let k = if r < grasp_rules.len() { 0 } else { rng.gen_range(2..=5) };
                if let Ok((sol, mk)) = construct_solution_conflict(challenge, pre, rule, k, Some(best_mk.saturating_add(best_mk / 20)), &mut rng, None, None, None, 0.0) {
                    if mk < best_mk { best_mk = mk; best_sol = sol.clone(); save_solution(&best_sol)?; }
                    push_top_solutions(&mut top_solutions, &sol, mk, 15);
                }
            }
        }

        let is_strict_perm = route_is_unique(pre.flow_route.as_deref().unwrap_or(&[]), challenge.num_machines) && pre.flex_avg <= 1.25;

        if !is_strict_perm {
            let ls_runs = top_solutions.len().min(15);
            let perturb_cycles = 16usize;
            for i in 0..ls_runs {
                let base_sol = &top_solutions[i].0;
                if let Ok(Some((sol2, mk2))) = critical_block_move_local_search_ex(pre, challenge, base_sol, 5, 64, perturb_cycles) {
                    if mk2 < best_mk { best_mk = mk2; best_sol = sol2.clone(); save_solution(&best_sol)?; }
                    push_top_solutions(&mut top_solutions, &sol2, mk2, 15);
                }
            }
        } else {
            let extra_iters = effort.flow_shop_iters / 2;
            if let (Some(route), Some(pt)) = (&pre.flow_route, &pre.flow_pt_by_job) {
                let unique = route_is_unique(route, challenge.num_machines);
                if unique && !pt.is_empty() {
                    let mut seed = challenge.seed; seed[0] ^= 0xD4; let mut rng = SmallRng::from_seed(seed);
                    let m = route.len();
                    TL_TAILLARD.with(|cell| {
                        let mut buf = cell.borrow_mut();
                        for i in 0..top_solutions.len().min(5) {
                            let start_ord = order_from_solution_first_op_start(&top_solutions[i].0, challenge.num_jobs);
                            if start_ord.len() != challenge.num_jobs { continue; }
                            let cand_seq = iterated_greedy_search(&start_ord, pt, (extra_iters / 5).min(300), 4, &mut rng);
                            buf.ensure(cand_seq.len(), m);
                            let mk = flow_makespan(&cand_seq, pt, &mut buf.comp[..m]);
                            if mk < best_mk {
                                best_mk = mk;
                                let sol = build_perm_solution_from_seq(&cand_seq, route, pt, challenge.num_jobs, challenge.num_machines);
                                best_sol = sol; let _ = save_solution(&best_sol);
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
    use rand::{rngs::SmallRng, Rng, SeedableRng, seq::SliceRandom};
    use tig_challenges::job_scheduling::*;
    use std::collections::HashMap;

    use super::types::*;
    use super::infra::*;

    // Sigmoid-normalized construction for HFS — mirrors av5's construct_solution_conflict_mode
    // Uses x/(1+x) for all terms + multiplicative interactions between terms
    fn construct_hfs_sig(
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
            .job_ops_len.iter().map(|&len| Vec::with_capacity(len)).collect();

        let mut remaining_ops = pre.total_ops;
        let mut time = 0u32;

        let mut demand: Vec<u16> = vec![0u16; num_machines];
        let mut raw_by_machine: Vec<Vec<RawCand>> =
            (0..num_machines).map(|_| Vec::with_capacity(12)).collect();
        let mut idle_machines: Vec<usize> = Vec::with_capacity(num_machines);

        let chaotic_like = pre.chaotic_like;
        let mut machine_work: Vec<u64> = if chaotic_like { vec![0u64; num_machines] } else { vec![] };
        let mut sum_work: u64 = 0;

        let avg_op_min_scale = pre.avg_op_min.max(1.0);
        let horizon_scale = pre.horizon.max(1.0);
        let time_scale_sc = pre.time_scale.max(1.0);
        let bn_focus_u = { let x = pre.bn_focus; if x <= 0.0 { 0.0 } else { x / (1.0 + x) } };

        // sigmoid helper — captured by closure for zero-cost inlining
        let sig = |x: f64| -> f64 { if x <= 0.0 { 0.0 } else { x / (1.0 + x) } };

        while remaining_ops > 0 {
            loop {
                idle_machines.clear();
                for m in 0..num_machines {
                    if machine_avail[m] <= time { idle_machines.push(m); }
                }
                if idle_machines.is_empty() { break; }

                for &m in &idle_machines { demand[m] = 0; raw_by_machine[m].clear(); }

                let progress = 1.0 - (remaining_ops as f64) / (pre.total_ops as f64).max(1.0);
                let cap_per_machine = if k == 0 { 12usize } else { (k + 6).min(12) };

                // Phased k-decay (same as infra.rs — tested +3.6% on fjsp_high)
                let ek = if k > 1 {
                    if chaotic_like {
                        if progress > 0.80 { 1usize } else if progress > 0.60 { ((k + 1) / 2).max(1) } else { k }
                    } else if pre.flex_avg > 1.5 {
                        if progress > 0.90 { 1usize } else if progress > 0.75 { ((k + 1) / 2).max(1) } else { k }
                    } else { k }
                } else { k };

                let prog_gate = sig(progress);

                for job in 0..num_jobs {
                    let op_idx = job_next_op[job];
                    if op_idx >= pre.job_ops_len[job] || job_ready_time[job] > time { continue; }
                    let product = pre.job_products[job];
                    let op = &pre.product_ops[product][op_idx];
                    if op.flex == 0 || op.machines.is_empty() || op.min_pt >= INF { continue; }

                    let (best_end, second_end, best_cnt_total, best_cnt_idle) =
                        best_second_and_counts(time, &machine_avail, op);
                    if best_end >= INF || best_cnt_idle == 0 { continue; }

                    let ops_rem = pre.job_ops_len[job] - op_idx;
                    let jb = job_bias.map(|v| v[job]).unwrap_or(0.0);
                    let flex_inv = 1.0 / (op.flex as f64).max(1.0);
                    let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
                    let regret = if second_end >= INF { pre.avg_op_min * 2.6 } else { (second_end - best_end) as f64 };
                    let regn = (regret / avg_op_min_scale).clamp(0.0, 6.0);
                    let rigidity = (0.60 * flex_inv + 0.40 * scarcity_urg).clamp(0.0, 2.5);

                    let rem_min = pre.product_suf_min[product][op_idx] as f64;
                    let rem_avg = pre.product_suf_avg[product][op_idx];
                    let rem_bn = pre.product_suf_bn[product][op_idx];
                    let density_n = ((rem_min / (ops_rem as f64).max(1.0)) / avg_op_min_scale).clamp(0.0, 4.0);
                    let rem_min_n = rem_min / horizon_scale;
                    let rem_avg_n = rem_avg / pre.max_job_avg_work.max(1e-9);
                    let bn_n = rem_bn / pre.max_job_bn.max(1e-9);
                    let next_min_n = (pre.product_next_min[product][op_idx] as f64) / horizon_scale;
                    let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
                    let next_term_raw = (0.55 * next_min_n + 0.45 * next_flex_inv)
                        * (1.0 + 0.30 * density_n * pre.high_flex);
                    let flow_term = pre.flow_w * pre.job_flow_pref[job] * (0.65 + 0.70 * (1.0 - progress));
                    let slack_u = slack_urgency(pre, target_mk, time, product, op_idx);

                    // Sigmoid normalization (per-job terms)
                    let rem_min_u = sig(rem_min_n);
                    let rem_avg_u = sig(rem_avg_n);
                    let bn_u = sig(bn_n);
                    let reg_u = sig(regn);
                    let dens_u = sig(density_n);
                    let next_u = sig(next_term_raw);
                    let flex_u = sig(flex_inv * pre.flex_factor.max(0.0));
                    let sat_scarcity = sig(scarcity_urg);
                    let scarce_slack = scarcity_urg * slack_u;
                    let scarce_reg = scarcity_urg * reg_u;
                    let base_bias0 = jb + flow_term;
                    let route_gain = (0.70 + 0.80 * (1.0 - progress)).clamp(0.70, 1.40);

                    for &(m, pt) in &op.machines {
                        if machine_avail[m] > time { continue; }
                        let end = time.saturating_add(pt);
                        if end != best_end { continue; }

                        demand[m] = demand[m].saturating_add(1);

                        let mp = machine_penalty.map(|v| v[m]).unwrap_or(0.0);
                        let jitter = if k > 0 { rng.gen::<f64>() * 1e-9 } else { 0.0 };
                        let load_n = machine_load[m] / pre.avg_machine_load.max(1e-9);
                        let proc_n = (pt as f64) / avg_op_min_scale;
                        let mpen = mp.clamp(0.0, 1.0);
                        let end_n = (best_end as f64) / time_scale_sc;
                        let pop_pen = if chaotic_like && op.flex >= 2 {
                            let pop = pre.machine_best_pop[m];
                            (0.07 + 0.15 * (1.0 - progress)).clamp(0.05, 0.24) * pop * pre.flex_factor
                        } else { 0.0 };
                        let load_u = sig(load_n);
                        let proc_u = sig(proc_n);
                        let mpen_u = sig(mpen);
                        let end_u = sig(end_n);
                        let base_bias = base_bias0 + jitter;
                        let route_term = if route_w > 0.0 && op.flex >= 2 {
                            route_w * route_gain * route_pref_bonus_lite(route_pref, product, op_idx, m)
                        } else { 0.0 };

                        let base = match rule {
                            Rule::CriticalPath => {
                                let chain = rem_min_u * (1.0 + next_u);
                                let urgent = scarce_slack * (1.0 + scarce_reg * prog_gate);
                                chain + urgent + base_bias - end_u - pop_pen + route_term
                            }
                            Rule::MostWork => {
                                let work = rem_avg_u * (1.0 + dens_u);
                                work * (1.0 + load_u) + base_bias - end_u - pop_pen + route_term
                            }
                            Rule::LeastFlex => {
                                let rigid = flex_u * (1.0 + sat_scarcity);
                                rigid + rem_min_u + next_u + base_bias - end_u - pop_pen + route_term
                            }
                            Rule::ShortestProc => {
                                (0.0 - proc_u) + rem_min_u * (1.0 + next_u) + sat_scarcity
                                    + base_bias - end_u - pop_pen + route_term
                            }
                            Rule::Regret => {
                                let rf = reg_u * (1.0 + sat_scarcity) * (1.0 + prog_gate);
                                rf + rem_min_u + next_u + base_bias - end_u - pop_pen + route_term
                            }
                            Rule::EndTight => {
                                let tight = scarce_slack * (1.0 + scarce_reg);
                                let chain = rem_min_u * (1.0 + prog_gate) * (1.0 + next_u);
                                let penal = end_u * (1.0 + prog_gate) + proc_u + mpen_u * pre.flex_factor;
                                chain + tight + base_bias - penal - pop_pen + route_term
                            }
                            Rule::BnHeavy => {
                                let bn_focus = bn_u * (1.0 + dens_u) * (1.0 + bn_focus_u);
                                let chain = rem_min_u * (1.0 + next_u);
                                let penal = end_u + proc_u + load_u * pre.flex_factor + mpen_u * pre.flex_factor;
                                bn_focus + chain + scarce_slack + reg_u + flex_u + base_bias - penal - pop_pen + route_term
                            }
                            Rule::Adaptive => {
                                let js = pre.jobshopness;
                                if js >= 1.0 - js {
                                    let hard = reg_u * (1.0 + scarce_reg) + flex_u + rem_min_u * (1.0 + next_u);
                                    hard + base_bias - (end_u + mpen_u * pre.flex_factor) - pop_pen + route_term
                                } else {
                                    let flow = rem_avg_u * (1.0 + dens_u) + (0.0 - proc_u) + slack_u;
                                    flow + base_bias - (end_u + load_u * pre.flex_factor) - pop_pen + route_term
                                }
                            }
                            Rule::FlexBalance => {
                                let flexible = flex_u * (1.0 + sat_scarcity);
                                let chain = (rem_avg_u + rem_min_u) * (1.0 + next_u);
                                let penal = end_u + load_u * pre.flex_factor + mpen_u * (1.0 + pre.flex_factor);
                                flexible + chain + base_bias - penal - pop_pen + route_term
                            }
                        };

                        push_top_k_raw(
                            &mut raw_by_machine[m],
                            RawCand { job, machine: m, pt, base_score: base, rigidity, reg_n: regn },
                            cap_per_machine,
                        );
                    }
                }

                // Conflict boost + candidate selection (same as infra.rs)
                let denom = (idle_machines.len() as f64).max(1.0);
                let (conflict_w, conflict_scale) = if chaotic_like {
                    (-(0.05 + 0.08 * (1.0 - progress)).clamp(0.04, 0.14),
                     (0.95 + 0.20 * pre.flex_factor).clamp(0.90, 1.20))
                } else {
                    ((0.09 + 0.26 * pre.jobshopness + 0.11 * pre.high_flex + 0.16 * (1.0 - progress)).clamp(0.05, 0.45),
                     (0.90 + 0.40 * pre.flex_factor).clamp(0.85, 1.75))
                };
                let (bal_w, avg_work) = if chaotic_like {
                    ((0.030 + 0.070 * (1.0 - progress)).clamp(0.025, 0.11),
                     (sum_work as f64) / (num_machines as f64).max(1.0))
                } else { (0.0, 0.0) };

                let mut best: Option<Cand> = None;
                let mut top: Vec<Cand> = if ek > 0 { Vec::with_capacity(ek) } else { Vec::new() };

                for &m in &idle_machines {
                    let dem = demand[m] as f64;
                    if dem <= 0.0 || raw_by_machine[m].is_empty() { continue; }
                    let dem_n = ((dem - 1.0) / denom).clamp(0.0, 2.5);
                    let bal_pen = if chaotic_like && bal_w > 0.0 {
                        let denomw = (avg_work + (pre.avg_op_min * 3.0).max(1.0)).max(1.0);
                        let r = (machine_work[m] as f64) / denomw;
                        -bal_w * (r / (r + 1.0)).clamp(0.0, 1.0)
                    } else { 0.0 };
                    for rc in &raw_by_machine[m] {
                        let rig = rc.rigidity.clamp(0.0, 2.5);
                        let regc = rc.reg_n.clamp(0.0, 4.5);
                        let mut boost = conflict_w * conflict_scale * dem_n * (1.15 * rig + 0.85 * regc);
                        if chaotic_like { boost = boost.max(-0.26); }
                        let c = Cand { job: rc.job, machine: rc.machine, pt: rc.pt,
                                       score: rc.base_score + boost + bal_pen };
                        if ek == 0 {
                            if best.map_or(true, |bb| c.score > bb.score) { best = Some(c); }
                        } else {
                            push_top_k(&mut top, c, ek);
                        }
                    }
                }

                let chosen = if ek == 0 {
                    match best { Some(c) => c, None => break }
                } else {
                    if top.is_empty() { break; }
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
                if machine_avail[machine] > time || end_check != best_end_now { break; }

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
                if remaining_ops == 0 { break; }
            }

            if remaining_ops == 0 { break; }

            let mut next_time: Option<u32> = None;
            for &t in &machine_avail {
                if t > time { next_time = Some(next_time.map_or(t, |b| b.min(t))); }
            }
            for j in 0..num_jobs {
                let op_idx = job_next_op[j];
                if op_idx < pre.job_ops_len[j] && job_ready_time[j] > time {
                    let t = job_ready_time[j];
                    next_time = Some(next_time.map_or(t, |b| b.min(t)));
                }
            }
            time = next_time.ok_or_else(|| anyhow!("Stalled"))?;
        }

        let mk = machine_avail.into_iter().max().unwrap_or(0);
        Ok((Solution { job_schedule }, mk))
    }

    fn maybe_plateau_local_search(
        pre: &Pre,
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        best_solution: &mut Option<Solution>,
        best_makespan: &mut u32,
        top_solutions: &mut Vec<(Solution, u32)>,
        learned_jb: &mut Option<Vec<f64>>,
        learned_mp: &mut Option<Vec<f64>>,
        learned_rp: &mut Option<RoutePrefLite>,
        learn_updates_left: &mut usize,
        stuck: usize,
    ) -> Result<bool> {
        let (p1, p2, p3) = match stuck {
            50 => (34, 57, 13),
            100 => (38, 63, 14),
            150 => (44, 73, 14),
            200 => (50, 82, 15),
            250 => (55, 88, 16),
            _ => return Ok(false),
        };

        let base_sol = match best_solution.as_ref() {
            Some(s) => s,
            None => return Ok(false),
        };

        if let Some((sol2, mk2)) = critical_block_move_local_search_ex(pre, challenge, base_sol, p1, p2, p3)? {
            if mk2 < *best_makespan {
                *best_makespan = mk2;
                *best_solution = Some(sol2.clone());
                save_solution(&sol2)?;

                if *learn_updates_left > 0 {
                    *learned_jb = Some(job_bias_from_solution(pre, &sol2)?);
                    *learned_mp = Some(machine_penalty_from_solution(pre, &sol2, challenge.num_machines)?);
                    *learned_rp = Some(route_pref_from_solution_lite(pre, &sol2, challenge)?);
                    *learn_updates_left -= 1;
                }

                push_top_solutions(top_solutions, &sol2, mk2, 25);
                return Ok(true);
            }
            push_top_solutions(top_solutions, &sol2, mk2, 25);
        }

        Ok(false)
    }

    pub fn solve(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        pre: &Pre,
        greedy_sol: Solution,
        greedy_mk: u32,
        effort: &EffortConfig,
    ) -> Result<()> {
        let mut rng = SmallRng::from_seed(challenge.seed);

        let rules: Vec<Rule> = vec![
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

        let mut best_makespan = greedy_mk;
        let mut best_solution: Option<Solution> = Some(greedy_sol);
        let mut top_solutions: Vec<(Solution, u32)> = Vec::new();

        let target_margin: u32 =
            ((pre.avg_op_min * (0.9 + 0.9 * pre.high_flex + 0.6 * pre.jobshopness)).max(1.0)) as u32;

        let route_w_base: f64 =
            (0.040 + 0.10 * pre.high_flex + 0.08 * pre.jobshopness).clamp(0.04, 0.22);

        if pre.flow_route.is_some() && pre.flow_pt_by_job.is_some() {
            let (sol, mk) = neh_reentrant_flow_solution(pre, challenge.num_jobs, challenge.num_machines)?;
            if mk < best_makespan {
                best_makespan = mk;
                best_solution = Some(sol.clone());
                save_solution(&sol)?;
            }
            push_top_solutions(&mut top_solutions, &sol, mk, 25);
        }

        let mut ranked: Vec<(Rule, u32, Solution)> = Vec::with_capacity(rules.len());
        for &rule in &rules {
            let (sol, mk) = construct_hfs_sig(
                challenge,
                pre,
                rule,
                0,
                None,
                &mut rng,
                None,
                None,
                None,
                0.0,
            )?;
            if mk < best_makespan {
                best_makespan = mk;
                best_solution = Some(sol.clone());
                save_solution(&sol)?;
            }
            push_top_solutions(&mut top_solutions, &sol, mk, 25);
            ranked.push((rule, mk, sol));
        }
        ranked.sort_by_key(|x| x.1);
        let r0 = ranked[0].0;
        let r1 = ranked.get(1).map(|x| x.0).unwrap_or(r0);
        let r2 = ranked.get(2).map(|x| x.0).unwrap_or(r1);

        let mut rule_best: Vec<u32> = vec![u32::MAX; 9];
        let mut rule_tries: Vec<u32> = vec![0u32; 9];
        for (rr, mk, _) in &ranked {
            let idx = rule_idx(*rr);
            rule_best[idx] = rule_best[idx].min(*mk);
            rule_tries[idx] = rule_tries[idx].saturating_add(1);
        }

        // Elite pool (av5-style: genotypic diversity + cross-mixing)
        let elite_cap: usize = 8;
        let mut elite_jb: Vec<Vec<f64>> = Vec::with_capacity(elite_cap);
        let mut elite_mp: Vec<Vec<f64>> = Vec::with_capacity(elite_cap);
        let mut elite_rp: Vec<RoutePrefLite> = Vec::with_capacity(elite_cap);
        let mut elite_mk: Vec<u32> = Vec::with_capacity(elite_cap);

        // Initialize elite pool from top 3 ranked solutions + greedy
        for i in 0..ranked.len().min(3) {
            let sol = &ranked[i].2; let mk = ranked[i].1;
            elite_jb.push(job_bias_from_solution(pre, sol)?);
            elite_mp.push(machine_penalty_from_solution(pre, sol, challenge.num_machines)?);
            elite_rp.push(route_pref_from_solution_lite(pre, sol, challenge)?);
            elite_mk.push(mk);
        }

        // Learned from single best (backwards compatibility)
        let base = &ranked[0].2;
        let mut learned_jb = Some(job_bias_from_solution(pre, base)?);
        let mut learned_mp = Some(machine_penalty_from_solution(pre, base, challenge.num_machines)?);
        let mut learned_rp = Some(route_pref_from_solution_lite(pre, base, challenge)?);
        let mut learn_updates_left = 8usize;

        let num_restarts = (effort.num_restarts * 5) / 4;
        let k_hi = 6usize;
        let mut stuck: usize = 0;

        for r in 0..num_restarts {
            let late = r >= (num_restarts * 2) / 3;

            let (k_min, k_max) = if stuck > 170 {
                (4usize, 6usize)
            } else if stuck > 90 {
                (3usize, 6usize)
            } else if stuck > 35 {
                (2usize, 6usize)
            } else {
                (2usize, 4usize)
            };

            let rule = if r < 35 {
                let u: f64 = rng.gen();
                if u < 0.12 {
                    Rule::FlexBalance
                } else if u < 0.50 {
                    r0
                } else if u < 0.75 {
                    r1
                } else if u < 0.90 {
                    r2
                } else {
                    rules[rng.gen_range(0..rules.len())]
                }
            } else {
                choose_rule_bandit(
                    &mut rng,
                    &rules,
                    &rule_best,
                    &rule_tries,
                    best_makespan,
                    target_margin,
                    stuck,
                    false,
                    late,
                )
            };

            let k = if k_max <= k_min {
                k_min
            } else {
                rng.gen_range(k_min..=k_max)
            }
            .min(k_hi);

            let learn_base =
                (0.08 + 0.22 * pre.jobshopness + 0.18 * pre.high_flex).clamp(0.05, 0.42);
            let learn_boost =
                (1.0 + 0.35 * ((stuck as f64) / 120.0).clamp(0.0, 1.0)).clamp(1.0, 1.35);
            let learn_p = (learn_base * learn_boost).clamp(0.0, 0.60);

            let target = if best_makespan < (u32::MAX / 2) {
                Some(best_makespan.saturating_add(target_margin))
            } else {
                None
            };

            let elite_boost = ((stuck as f64) / 140.0).clamp(0.0, 1.0);
            // Elite pool cross-mixing (av5-style): pick jb/mp/rp from potentially different elites
            let learn_base2 = (0.09 + 0.24 * pre.jobshopness + 0.20 * pre.high_flex).clamp(0.06, 0.44);
            let learn_boost2 = (1.0 + 0.38 * ((stuck as f64) / 120.0).clamp(0.0, 1.0)).clamp(1.0, 1.38);
            let use_elite_pool = !elite_mk.is_empty() && r > 10 && rng.gen::<f64>() < (learn_base2 * learn_boost2).clamp(0.0, 0.65);

            let use_elite_top = !use_elite_pool && !top_solutions.is_empty() && r > 25 && {
                let elite_base = (0.05 + 0.10 * pre.high_flex + 0.06 * pre.jobshopness).clamp(0.02, 0.22);
                let elite_p = (elite_base + 0.28 * elite_boost + if late { 0.05 } else { 0.0 }).clamp(0.0, 0.45);
                rng.gen::<f64>() < elite_p
            };

            let use_learn = !use_elite_pool && !use_elite_top
                && learned_jb.is_some() && learned_mp.is_some() && learned_rp.is_some()
                && rng.gen::<f64>() < learn_p;

            let (sol, mk) = if use_elite_pool {
                let n = elite_mk.len();
                let pick_elite = |rng: &mut SmallRng| -> usize {
                    if n <= 1 { return 0; }
                    let a = rng.gen_range(0..n); let b = rng.gen_range(0..n);
                    if elite_mk[a] <= elite_mk[b] { a } else { b }
                };
                let base_idx = pick_elite(&mut rng);
                let mix_p = (0.055 + 0.10 * pre.high_flex + 0.09 * pre.jobshopness + 0.16 * elite_boost).clamp(0.05, 0.40);
                let mp_idx = if n > 1 && rng.gen::<f64>() < mix_p { pick_elite(&mut rng) } else { base_idx };
                let rp_idx = if n > 1 && rng.gen::<f64>() < mix_p { pick_elite(&mut rng) } else { base_idx };
                let jitter = (0.80 + 0.70 * rng.gen::<f64>()).clamp(0.65, 1.55);
                let route_w = (route_w_base * jitter).clamp(route_w_base * 0.55, 0.45);
                construct_hfs_sig(
                    challenge, pre, rule, k, target, &mut rng,
                    Some(&elite_jb[base_idx]), Some(&elite_mp[mp_idx]), Some(&elite_rp[rp_idx]), route_w,
                )?
            } else if use_elite_top {
                let elite_n = top_solutions.len().min(6).max(1);
                let t: f64 = rng.gen(); let u: f64 = rng.gen();
                let pick = (((t * u) * elite_n as f64) as usize).min(elite_n - 1);
                let elite_sol = &top_solutions[pick].0;
                let jb = job_bias_from_solution(pre, elite_sol)?;
                let mp = machine_penalty_from_solution(pre, elite_sol, challenge.num_machines)?;
                let rp = route_pref_from_solution_lite(pre, elite_sol, challenge)?;
                let route_w = (route_w_base * (1.10 + 0.30 * elite_boost)).clamp(route_w_base, 0.45);
                construct_hfs_sig(
                    challenge, pre, rule, k, target, &mut rng,
                    Some(&jb), Some(&mp), Some(&rp), route_w,
                )?
            } else if use_learn {
                construct_hfs_sig(
                    challenge, pre, rule, k, target, &mut rng,
                    learned_jb.as_deref(), learned_mp.as_deref(), learned_rp.as_ref(), route_w_base,
                )?
            } else {
                construct_hfs_sig(
                    challenge, pre, rule, k, target, &mut rng,
                    None, None, None, 0.0,
                )?
            };

            let ridx = rule_idx(rule);
            rule_tries[ridx] = rule_tries[ridx].saturating_add(1);
            rule_best[ridx] = rule_best[ridx].min(mk);

            // Inline LS (av5-style): fire on improvements with moderate probability
            // Uses lighter params than post-processing CBMLS to preserve construction budget
            let flex01 = (pre.high_flex + pre.jobshopness).clamp(0.0, 1.0);
            let near_best = mk <= best_makespan.saturating_add((target_margin / 3).max(1));
            let do_inline_ls = if mk < best_makespan {
                late || stuck > 20 || flex01 >= 0.12 || rng.gen::<f64>() < 0.55
            } else if near_best && stuck > 140 {
                rng.gen::<f64>() < 0.04
            } else {
                false
            };
            let (mut sol, mut mk) = (sol, mk);
            if do_inline_ls {
                let bump = if flex01 > 0.60 { 1.0f64 } else { 0.0f64 };
                let p1 = (34.0 + 6.0 * bump) as usize;
                let p2 = (55.0 + 10.0 * bump) as usize;
                let p3 = 12usize;
                if let Some((sol2, mk2)) = critical_block_move_local_search_ex(pre, challenge, &sol, p1, p2, p3)? {
                    if mk2 < mk { sol = sol2; mk = mk2; }
                }
            }

            if mk < best_makespan {
                best_makespan = mk;
                best_solution = Some(sol.clone());
                save_solution(&sol)?;
                stuck = 0;

                if learn_updates_left > 0 {
                    learned_jb = Some(job_bias_from_solution(pre, &sol)?);
                    learned_mp = Some(machine_penalty_from_solution(pre, &sol, challenge.num_machines)?);
                    learned_rp = Some(route_pref_from_solution_lite(pre, &sol, challenge)?);
                    learn_updates_left -= 1;
                }
                // Update elite pool with improved solution
                let jb = job_bias_from_solution(pre, &sol)?;
                let mp = machine_penalty_from_solution(pre, &sol, challenge.num_machines)?;
                let rp = route_pref_from_solution_lite(pre, &sol, challenge)?;
                elite_jb.push(jb); elite_mp.push(mp); elite_rp.push(rp); elite_mk.push(mk);
                // Prune: keep top elite_cap by makespan
                if elite_mk.len() > elite_cap {
                    let mut order: Vec<usize> = (0..elite_mk.len()).collect();
                    order.sort_unstable_by_key(|&i| elite_mk[i]);
                    let keep: Vec<usize> = order.into_iter().take(elite_cap).collect();
                    elite_jb = keep.iter().map(|&i| elite_jb[i].clone()).collect();
                    elite_mp = keep.iter().map(|&i| elite_mp[i].clone()).collect();
                    elite_rp = keep.iter().map(|&i| elite_rp[i].clone()).collect();
                    elite_mk = keep.iter().map(|&i| elite_mk[i]).collect();
                }
            } else {
                stuck = stuck.saturating_add(1);
                // Occasionally add near-best solutions to pool for diversity
                if mk <= best_makespan.saturating_add(target_margin / 2) && rng.gen::<f64>() < 0.08 {
                    let jb = job_bias_from_solution(pre, &sol)?;
                    let mp = machine_penalty_from_solution(pre, &sol, challenge.num_machines)?;
                    let rp = route_pref_from_solution_lite(pre, &sol, challenge)?;
                    elite_jb.push(jb); elite_mp.push(mp); elite_rp.push(rp); elite_mk.push(mk);
                    if elite_mk.len() > elite_cap {
                        let mut order: Vec<usize> = (0..elite_mk.len()).collect();
                        order.sort_unstable_by_key(|&i| elite_mk[i]);
                        let keep: Vec<usize> = order.into_iter().take(elite_cap).collect();
                        elite_jb = keep.iter().map(|&i| elite_jb[i].clone()).collect();
                        elite_mp = keep.iter().map(|&i| elite_mp[i].clone()).collect();
                        elite_rp = keep.iter().map(|&i| elite_rp[i].clone()).collect();
                        elite_mk = keep.iter().map(|&i| elite_mk[i]).collect();
                    }
                }
            }

            push_top_solutions(&mut top_solutions, &sol, mk, 25);

            if maybe_plateau_local_search(
                pre,
                challenge,
                save_solution,
                &mut best_solution,
                &mut best_makespan,
                &mut top_solutions,
                &mut learned_jb,
                &mut learned_mp,
                &mut learned_rp,
                &mut learn_updates_left,
                stuck,
            )? {
                stuck = 0;
            }
        }

        let route_w_ls: f64 = (route_w_base * 1.40).clamp(route_w_base, 0.40);

        let mut refine_results: Vec<(Solution, u32)> = Vec::new();
        for (base_sol, _) in top_solutions.iter() {
            let jb = job_bias_from_solution(pre, base_sol)?;
            let mp = machine_penalty_from_solution(pre, base_sol, challenge.num_machines)?;
            let rp = Some(route_pref_from_solution_lite(pre, base_sol, challenge)?);

            let target_ls = if best_makespan < (u32::MAX / 2) {
                Some(best_makespan.saturating_add(target_margin / 2))
            } else {
                None
            };

            for attempt in 0..15 {
                let rule = match attempt {
                    0 => r0,
                    1 => Rule::Adaptive,
                    2 => Rule::BnHeavy,
                    3 => Rule::EndTight,
                    4 => Rule::Regret,
                    5 => Rule::CriticalPath,
                    6 => Rule::LeastFlex,
                    7 => Rule::MostWork,
                    8 => Rule::FlexBalance,
                    _ => r1,
                };

                let k = match attempt % 4 {
                    0 => 2,
                    1 => 3,
                    2 => 4,
                    _ => 2,
                }
                .min(k_hi);

                let (sol, mk) = construct_hfs_sig(
                    challenge,
                    pre,
                    rule,
                    k,
                    target_ls,
                    &mut rng,
                    Some(&jb),
                    Some(&mp),
                    rp.as_ref(),
                    route_w_ls,
                )?;

                if mk < best_makespan {
                    best_makespan = mk;
                    best_solution = Some(sol.clone());
                    save_solution(&sol)?;
                }
                refine_results.push((sol, mk));
            }
        }
        for (sol, mk) in refine_results {
            push_top_solutions(&mut top_solutions, &sol, mk, 25);
        }

        let ls_runs = top_solutions.len().min(15);
        let ls_iters = (effort.hybrid_flow_shop_iters / 65).max(30);
        let ls_cands = (effort.hybrid_flow_shop_iters / 40).max(48);
        let ls_perturb = (effort.hybrid_flow_shop_iters / 200).max(8);
        for i in 0..ls_runs {
            let base_sol = &top_solutions[i].0;
            if let Some((sol2, mk2)) =
                critical_block_move_local_search_ex(pre, challenge, base_sol, ls_iters, ls_cands, ls_perturb)?
            {
                if mk2 < best_makespan {
                    best_makespan = mk2;
                    best_solution = Some(sol2.clone());
                    save_solution(&sol2)?;
                }
                push_top_solutions(&mut top_solutions, &sol2, mk2, 25);
            }
        }

        // ─── Extra post-processing (job_seven additions) ─────────────────────────

        // Pass A: Schrage resequencing on top 8
        for i in 0..top_solutions.len().min(8) {
            let bs = top_solutions[i].0.clone();
            if let Ok(Some((s, m))) = schrage_pass(pre, challenge, &bs, 8) {
                if m < best_makespan { best_makespan = m; best_solution = Some(s.clone()); save_solution(&s)?; }
                push_top_solutions(&mut top_solutions, &s, m, 25);
            }
        }

        // Pass B: CP Descent BF on top 6
        for i in 0..top_solutions.len().min(6) {
            let bs = top_solutions[i].0.clone();
            if let Ok(Some((s, m))) = cp_descent_bf(pre, challenge, &bs, 6) {
                if m < best_makespan { best_makespan = m; best_solution = Some(s.clone()); save_solution(&s)?; }
                push_top_solutions(&mut top_solutions, &s, m, 25);
            }
        }

        // Pass C: Schrage again after machine reassignments
        if let Some(ref sol) = best_solution {
            if let Ok(Some((s, m))) = schrage_pass(pre, challenge, sol, 4) {
                if m < best_makespan { best_makespan = m; best_solution = Some(s.clone()); save_solution(&s)?; }
            }
        }

        // Pass D: Final CBMLS
        if let Some(ref sol) = best_solution {
            if let Some((s, m)) = critical_block_move_local_search_ex(pre, challenge, sol, ls_iters, ls_cands, ls_perturb)? {
                if m < best_makespan { best_makespan = m; best_solution = Some(s.clone()); save_solution(&s)?; }
            }
        }

        // Pass E: Greedy reassign
        if let Some(ref sol) = best_solution {
            if let Some((s, m)) = greedy_reassign_pass(pre, challenge, sol)? {
                if m < best_makespan { best_makespan = m; best_solution = Some(s.clone()); save_solution(&s)?; }
            }
        }

        // Pass F: Final CP Descent BF
        if let Some(ref sol) = best_solution {
            if let Ok(Some((s, m))) = cp_descent_bf(pre, challenge, sol, 6) {
                if m < best_makespan { best_makespan = m; best_solution = Some(s.clone()); save_solution(&s)?; }
            }
        }

        // Pass G: Window exhaustive on best
        if let Some(ref sol) = best_solution {
            if let Ok(Some((s, m))) = window_exhaustive(pre, challenge, sol, &[4, 3, 2]) {
                if m < best_makespan { best_makespan = m; best_solution = Some(s.clone()); save_solution(&s)?; }
            }
        }

        // Phase H: Multi-seed Tabu-ILS with O(1) move estimators
        // Replaces expensive SES-based H1/H2/LNS phases (bottleneck at O(n²) per SES call).
        // Uses tabu_ils_hfs (already proven on T44/T46) on top-5 diverse pool seeds.
        {
            let fuel = effort.hybrid_flow_shop_iters.max(800);
            let num_seeds = 5.min(top_solutions.len()).max(if best_solution.is_some() { 1 } else { 0 });
            if num_seeds > 0 {
                let tenure_base = ((pre.total_ops as f64).sqrt() as usize).clamp(6, 16);
                let stagn_limit = 120usize;
                let tabu_fuel_per_seed = ((fuel * 3 / 4) / num_seeds).max(80);

                // Build seed list: best_solution first, then diverse pool
                let mut seeds: Vec<(Solution, u32)> = Vec::with_capacity(num_seeds);
                if let Some(ref s) = best_solution {
                    seeds.push((s.clone(), best_makespan));
                }
                for (sol, mk) in top_solutions.iter().take(num_seeds.saturating_sub(seeds.len())) {
                    seeds.push((sol.clone(), *mk));
                }

                for (seed_sol, _) in &seeds {
                    match tabu_ils_hfs(
                        pre, challenge,
                        seed_sol,
                        best_makespan,
                        tabu_fuel_per_seed,
                        tenure_base,
                        stagn_limit,
                        &mut rng,
                        save_solution,
                    ) {
                        Ok((new_sol, new_mk)) => {
                            if new_mk < best_makespan {
                                best_makespan = new_mk;
                                best_solution = Some(new_sol.clone());
                                save_solution(&new_sol)?;
                            }
                            push_top_solutions(&mut top_solutions, &new_sol, new_mk, 20);
                        }
                        Err(_) => {}
                    }
                }

                // Final deep single-seed run on the overall best
                let final_fuel = (fuel / 2).max(200);
                if let Some(best_sol_clone) = best_solution.clone() {
                    match tabu_ils_hfs(
                        pre, challenge,
                        &best_sol_clone,
                        best_makespan,
                        final_fuel,
                        tenure_base,
                        stagn_limit / 2,
                        &mut rng,
                        save_solution,
                    ) {
                        Ok((final_sol, final_mk)) => {
                            if final_mk < best_makespan {
                                best_makespan = final_mk;
                                best_solution = Some(final_sol);
                            }
                        }
                        Err(_) => {}
                    }
                }
            }
        }

        // Multi-start Pass TS: N1 tabu on best + diverse elite solutions
        {
            let ts_budget = 50_000usize;
            let ts_tenure = ((pre.total_ops as f64).sqrt() as usize).max(5).min(20);
            let mut ts_pool: Vec<Solution> = Vec::with_capacity(3);
            let mut seen_mks: Vec<u32> = Vec::with_capacity(3);
            if let Some(ref s) = best_solution {
                ts_pool.push(s.clone());
                seen_mks.push(best_makespan);
            }
            for (pool_sol, pool_mk) in top_solutions.iter().take(5) {
                if ts_pool.len() >= 3 { break; }
                if !seen_mks.contains(pool_mk) {
                    seen_mks.push(*pool_mk);
                    ts_pool.push(pool_sol.clone());
                }
            }
            for ts_base in &ts_pool {
                if let Ok(Some((ts_sol, ts_mk))) = super::job_shop::tabu_search_phase(pre, challenge, ts_base, ts_budget, ts_tenure) {
                    if ts_mk < best_makespan {
                        best_makespan = ts_mk;
                        best_solution = Some(ts_sol.clone());
                        save_solution(&ts_sol)?;
                    }
                }
            }
        }

        // SCD-SR: Stage-Coordinate Descent with Schrage-Recreation
        // Ruins ALL ops at one stage, rebuilds with ECT+tail heuristic, exact eval_disj.
        // Escapes assignment basins that N1 swaps cannot reach.
        if let Some(ref base) = best_solution.clone() {
            if let Some((scd_sol, scd_mk)) = scd_sr_pass(pre, challenge, base, best_makespan, 50) {
                if scd_mk < best_makespan {
                    best_makespan = scd_mk;
                    best_solution = Some(scd_sol.clone());
                    save_solution(&scd_sol)?;
                    // Final TS polish on SCD-SR result
                    let ts_budget = 30_000usize;
                    let ts_tenure = ((pre.total_ops as f64).sqrt() as usize).max(5).min(20);
                    if let Ok(Some((ts_sol, ts_mk))) = super::job_shop::tabu_search_phase(pre, challenge, &scd_sol, ts_budget, ts_tenure) {
                        if ts_mk < best_makespan {
                            best_makespan = ts_mk;
                            best_solution = Some(ts_sol.clone());
                            save_solution(&ts_sol)?;
                        }
                    }
                }
            }
        }

        // Phase I: ARR-IG-FI seeding (1k iters — pure seed for CPDT-ILS)
        if let Some(ref base) = best_solution.clone() {
            let (ils_sol, ils_mk) = fast_ils_hfs(pre, challenge, base, best_makespan, &mut rng, save_solution);
            if ils_mk < best_makespan {
                best_makespan = ils_mk;
                best_solution = Some(ils_sol.clone());
            }
            // Phase J: CPDT-ILS — disjunctive tabu with O(1) swap + reassign
            let (tabu_sol, tabu_mk) = tabu_ils_hfs(
                pre, challenge, &ils_sol, ils_mk.min(best_makespan),
                40_000, 8, 800,
                &mut rng, save_solution,
            )?;
            if tabu_mk < best_makespan {
                best_makespan = tabu_mk;
                best_solution = Some(tabu_sol);
            }
        }

        // Multi-start SCD-SR from elite pool (3 restarts, 50 sweeps, ~75s budget)
        {
            let mut pool_starts: Vec<(Solution, u32)> = Vec::new();
            if let Some(ref s) = best_solution { pool_starts.push((s.clone(), best_makespan)); }
            for (sol, mk) in top_solutions.iter().take(5) {
                if pool_starts.len() >= 3 { break; }
                if !pool_starts.iter().any(|(_, m)| *m == *mk) {
                    pool_starts.push((sol.clone(), *mk));
                }
            }
            for (start_sol, start_mk) in pool_starts {
                if let Some((scd_sol, scd_mk)) = scd_sr_pass(pre, challenge, &start_sol, start_mk, 50) {
                    if scd_mk < best_makespan {
                        best_makespan = scd_mk;
                        best_solution = Some(scd_sol.clone());
                        save_solution(&scd_sol)?;
                    }
                }
            }
        }

        if let Some(sol) = best_solution { save_solution(&sol)?; }
        let _ = best_makespan;
        Ok(())
    }

    // --------------------------------------------------------------------------
    // SCD-SR: Stage-Coordinate Descent with Schrage-Recreation
    // Ruins ALL ops at one stage, rebuilds via ECT+tail heuristic, exact eval.
    // --------------------------------------------------------------------------

    fn scd_sr_pass(
        pre: &Pre,
        challenge: &Challenge,
        base_sol: &Solution,
        base_mk: u32,
        max_sweeps: usize,
    ) -> Option<(Solution, u32)> {
        let num_jobs = challenge.num_jobs;
        let num_machines = challenge.num_machines;
        let max_stages = pre.max_ops;

        let mut best = base_sol.clone();
        let mut best_mk = base_mk;

        for _sweep in 0..max_sweeps {
            let mut improved_sweep = false;

            for s in 0..max_stages {
                // Skip stage if no job has an op here
                if (0..num_jobs).all(|j| s >= pre.job_ops_len[j]) { break; }

                let mut scratch = best.clone();

                // 1. Release times at stage s: end of stage s-1 for each job
                let mut releases = vec![0u32; num_jobs];
                for j in 0..num_jobs {
                    if s == 0 || s >= pre.job_ops_len[j] { continue; }
                    let (prev_m, prev_start) = scratch.job_schedule[j][s - 1];
                    let product = pre.job_products[j];
                    let prev_pt = pt_from_op(&pre.product_ops[product][s - 1], prev_m).unwrap_or(0);
                    releases[j] = prev_start + prev_pt;
                }

                // 2. Tail LBs: min remaining time from stage s+1 onward
                let mut tails = vec![0u32; num_jobs];
                for j in 0..num_jobs {
                    if s >= pre.job_ops_len[j] { continue; }
                    let product = pre.job_products[j];
                    if s + 1 < pre.product_suf_min[product].len() {
                        tails[j] = pre.product_suf_min[product][s + 1];
                    }
                }

                // 3. Sort jobs at stage s: ascending release, descending tail (Schrage)
                let mut stage_jobs: Vec<usize> = (0..num_jobs)
                    .filter(|&j| s < pre.job_ops_len[j])
                    .collect();
                stage_jobs.sort_unstable_by(|&a, &b| {
                    releases[a].cmp(&releases[b])
                        .then_with(|| tails[b].cmp(&tails[a]))
                });

                // 4. Build machine usages from stages 0..s-1 (fixed)
                let mut machine_usages: Vec<Vec<(u32, u32)>> = vec![Vec::new(); num_machines];
                for j in 0..num_jobs {
                    for t in 0..s.min(pre.job_ops_len[j]) {
                        let (m, st) = scratch.job_schedule[j][t];
                        let product = pre.job_products[j];
                        let pt = pt_from_op(&pre.product_ops[product][t], m).unwrap_or(0);
                        if pt > 0 { insert_usage(&mut machine_usages[m], st, st + pt); }
                    }
                }

                // 5. RECREATE: ECT+tail assignment for stage s
                for &j in &stage_jobs {
                    let product = pre.job_products[j];
                    let op_info = &pre.product_ops[product][s];
                    let rel = releases[j];
                    let tail = tails[j];

                    let mut best_m = usize::MAX;
                    let mut best_start = 0u32;
                    let mut best_score = u32::MAX;

                    for &(m, pt) in &op_info.machines {
                        let start = find_earliest_gap(&machine_usages[m], rel, pt);
                        let score = (start + pt).saturating_add(tail);
                        if score < best_score {
                            best_score = score;
                            best_start = start;
                            best_m = m;
                        }
                    }

                    if best_m == usize::MAX { continue; }
                    let pt = pt_from_op(op_info, best_m).unwrap_or(0);
                    insert_usage(&mut machine_usages[best_m], best_start, best_start + pt);
                    scratch.job_schedule[j][s] = (best_m, best_start);
                }

                // 6. Propagate: recompute start times for stages s+1..max_stages-1
                // Keep machine assignments fixed, recompute timing using accumulated usages.
                // machine_usages now contains stages 0..s; extend for stages s+1..
                for t in (s + 1)..max_stages {
                    if (0..num_jobs).all(|j| t >= pre.job_ops_len[j]) { break; }

                    // Collect jobs at stage t with release times from stage t-1
                    let mut machine_queues: Vec<Vec<(u32, usize)>> = vec![Vec::new(); num_machines];
                    for j in 0..num_jobs {
                        if t >= pre.job_ops_len[j] { continue; }
                        let (prev_m, prev_start) = scratch.job_schedule[j][t - 1];
                        let product = pre.job_products[j];
                        let prev_pt = pt_from_op(&pre.product_ops[product][t - 1], prev_m).unwrap_or(0);
                        let rel = prev_start + prev_pt;
                        let (m, _) = scratch.job_schedule[j][t];
                        machine_queues[m].push((rel, j));
                    }

                    // For each machine: sort by release, schedule via gap-filling
                    for m in 0..num_machines {
                        if machine_queues[m].is_empty() { continue; }
                        machine_queues[m].sort_unstable();
                        for &(rel, j) in &machine_queues[m] {
                            let product = pre.job_products[j];
                            let pt = pt_from_op(&pre.product_ops[product][t], m).unwrap_or(0);
                            let start = find_earliest_gap(&machine_usages[m], rel, pt);
                            insert_usage(&mut machine_usages[m], start, start + pt);
                            scratch.job_schedule[j][t] = (m, start);
                        }
                    }
                }

                // 7. EXACT EVAL via eval_disj
                if let Some(mk) = eval_solution_makespan(pre, challenge, &scratch) {
                    if mk < best_mk {
                        best_mk = mk;
                        best = scratch;
                        improved_sweep = true;
                    }
                }
            }

            if !improved_sweep { break; }
        }

        if best_mk < base_mk { Some((best, best_mk)) } else { None }
    }

    // ============================================================================
    // CPDT-ILS: Critical-Path Disjunctive Tabu-ILS with Machine Reassignment
    // Ported from fjsp_medium.rs — O(1) estimated swap + reassign on disj. graph
    // ============================================================================

    #[inline]
    fn estimate_swap_hfs(
        u: usize, v: usize,
        heads: &[u32], tails: &[u32], pt: &[u32],
        job_pred: &[usize], job_succ: &[usize],
        machine_pred: &[usize], machine_succ: &[usize],
    ) -> u32 {
        let mp_u = machine_pred[u];
        let ms_v = machine_succ[v];
        let jp_v = job_pred[v];
        let jp_u = job_pred[u];
        let js_u = job_succ[u];
        let js_v = job_succ[v];

        let r_mp_u = if mp_u != NONE_USIZE { heads[mp_u].saturating_add(pt[mp_u]) } else { 0 };
        let r_jp_v = if jp_v != NONE_USIZE { heads[jp_v].saturating_add(pt[jp_v]) } else { 0 };
        let new_r_v = r_jp_v.max(r_mp_u);

        let r_jp_u = if jp_u != NONE_USIZE { heads[jp_u].saturating_add(pt[jp_u]) } else { 0 };
        let new_r_u = r_jp_u.max(new_r_v.saturating_add(pt[v]));

        let q_js_u = if js_u != NONE_USIZE { pt[js_u].saturating_add(tails[js_u]) } else { 0 };
        let q_ms_v = if ms_v != NONE_USIZE { pt[ms_v].saturating_add(tails[ms_v]) } else { 0 };
        let new_q_u = q_js_u.max(q_ms_v);

        let q_js_v = if js_v != NONE_USIZE { pt[js_v].saturating_add(tails[js_v]) } else { 0 };
        let new_q_v = q_js_v.max(pt[u].saturating_add(new_q_u));

        (new_r_v.saturating_add(pt[v]).saturating_add(new_q_v))
            .max(new_r_u.saturating_add(pt[u]).saturating_add(new_q_u))
    }

    #[inline]
    fn estimate_reassign_hfs(
        ds: &DisjSchedule,
        heads: &[u32], tails: &[u32],
        node: usize, new_machine: usize, new_pt: u32, insert_pos: usize,
        job_pred: &[usize], machine_pred: &[usize], machine_succ: &[usize],
    ) -> u32 {
        let jp = job_pred[node];
        let js = ds.job_succ[node];
        let old_mp = machine_pred[node];
        let old_ms = machine_succ[node];

        let jp_end = if jp != NONE_USIZE { heads[jp].saturating_add(ds.node_pt[jp]) } else { 0 };

        let new_seq = &ds.machine_seq[new_machine];
        let new_mp_end = if insert_pos > 0 && !new_seq.is_empty() {
            let pred = new_seq[(insert_pos - 1).min(new_seq.len() - 1)];
            heads[pred].saturating_add(ds.node_pt[pred])
        } else { 0 };

        let new_start = jp_end.max(new_mp_end);
        let new_end = new_start.saturating_add(new_pt);

        let js_tail = if js != NONE_USIZE { ds.node_pt[js].saturating_add(tails[js]) } else { 0 };
        let new_ms_tail = if insert_pos < new_seq.len() {
            let succ = new_seq[insert_pos];
            ds.node_pt[succ].saturating_add(tails[succ])
        } else { 0 };

        let node_path = new_end.saturating_add(js_tail.max(new_ms_tail));

        let old_reconnect = if old_mp != NONE_USIZE && old_ms != NONE_USIZE {
            let old_mp_end = heads[old_mp].saturating_add(ds.node_pt[old_mp]);
            old_mp_end.saturating_add(ds.node_pt[old_ms]).saturating_add(tails[old_ms])
        } else { 0 };

        node_path.max(old_reconnect)
    }

    fn find_insert_positions_hfs(
        ds: &DisjSchedule,
        starts: &[u32],
        node: usize,
        new_machine: usize,
        job_pred: &[usize],
    ) -> Vec<usize> {
        let seq = &ds.machine_seq[new_machine];
        let len = seq.len();
        if len == 0 { return vec![0]; }

        let jp = job_pred[node];
        let job_pred_end = if jp != NONE_USIZE { starts[jp].saturating_add(ds.node_pt[jp]) } else { 0 };
        let cur_start = starts[node];

        let mut pos_after_jp = len;
        for (i, &nd) in seq.iter().enumerate() {
            if starts[nd] > job_pred_end { pos_after_jp = i; break; }
        }

        let mut pos_by_cur = len;
        for (i, &nd) in seq.iter().enumerate() {
            if starts[nd] >= cur_start { pos_by_cur = i; break; }
        }

        let mut out: Vec<usize> = Vec::with_capacity(6);
        {
            let push = |v: &mut Vec<usize>, p: usize| {
                if p <= len && !v.contains(&p) { v.push(p); }
            };
            push(&mut out, pos_after_jp);
            push(&mut out, pos_after_jp.saturating_sub(1));
            push(&mut out, pos_by_cur);
            push(&mut out, pos_by_cur.saturating_sub(1));
            push(&mut out, 0);
            push(&mut out, len);
        }
        if out.is_empty() { out.push(len); }
        if out.len() > 6 { out.truncate(6); }
        out
    }

    enum MoveHfs {
        Swap { machine: usize, pos: usize },
        Reassign { node: usize, new_machine: usize, new_pt: u32, insert_pos: usize },
    }

    fn tabu_ils_hfs(
        pre: &Pre,
        challenge: &Challenge,
        seed_sol: &Solution,
        seed_mk: u32,
        max_iters: usize,
        tenure_base: usize,
        stagnation_limit: usize,
        rng: &mut SmallRng,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
    ) -> Result<(Solution, u32)> {
        let Ok(mut ds) = build_disj_from_solution(pre, challenge, seed_sol) else {
            return Ok((seed_sol.clone(), seed_mk));
        };
        let mut buf = EvalBuf::new(ds.n);
        let n = ds.n;
        let Some((initial_mk, _)) = eval_disj(&ds, &mut buf) else {
            return Ok((seed_sol.clone(), seed_mk));
        };

        let mut best_mk = initial_mk.min(seed_mk);
        let mut best_ds = ds.clone();
        let mut cur_mk = initial_mk;

        let tenure = tenure_base.max(5);
        let tenure_delta = (tenure / 3).max(2);
        let kick_threshold = (stagnation_limit * 2 / 3).max(50);

        let mut tabu_swap: HashMap<(usize, usize), usize> = HashMap::with_capacity(tenure * 8);
        let mut tabu_reassign: HashMap<(usize, usize), usize> = HashMap::with_capacity(tenure * 4);

        // Job predecessor array (constant — job chains don't change)
        let mut job_pred_node = vec![NONE_USIZE; n];
        for j in 0..ds.num_jobs {
            let base = ds.job_offsets[j];
            let end = ds.job_offsets[j + 1];
            for k in (base + 1)..end { job_pred_node[k] = k - 1; }
        }

        let mut no_improve = 0usize;
        let mut kicks_left = 5usize;
        let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15)
            ^ (initial_mk as u64).wrapping_shl(16) ^ (n as u64).wrapping_mul(0x517CC1B727220A95);

        let mut machine_pred_node = vec![NONE_USIZE; n];
        let mut crit = vec![false; n];

        for iter in 0..max_iters {
            if no_improve >= stagnation_limit {
                if kicks_left == 0 { break; }
                ds = best_ds.clone();
                let Some((mk, _)) = eval_disj(&ds, &mut buf) else { break };
                cur_mk = mk;
                no_improve = 0;
                kicks_left -= 1;
                tabu_swap.clear();
                tabu_reassign.clear();
                continue;
            }

            // Periodic kick on stagnation plateau
            if no_improve > 0 && no_improve % kick_threshold == 0 && kicks_left > 0 {
                let Some((_, kick_mk_node)) = eval_disj(&ds, &mut buf) else { break };
                crit.fill(false);
                let mut u = kick_mk_node;
                while u != NONE_USIZE { crit[u] = true; u = buf.best_pred[u]; }
                let mut kick_swaps: Vec<(usize, usize)> = Vec::new();
                for m in 0..ds.num_machines {
                    if ds.machine_seq[m].len() <= 1 { continue; }
                    for i in 0..(ds.machine_seq[m].len() - 1) {
                        if crit[ds.machine_seq[m][i]] || crit[ds.machine_seq[m][i + 1]] {
                            kick_swaps.push((m, i));
                        }
                    }
                }
                if !kick_swaps.is_empty() {
                    for _ in 0..3 {
                        pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                        let idx = (pseed as usize) % kick_swaps.len();
                        let (m, pos) = kick_swaps[idx];
                        if pos + 1 < ds.machine_seq[m].len() { ds.machine_seq[m].swap(pos, pos + 1); }
                    }
                }
                kicks_left -= 1;
                continue;
            }

            let Some((mk_now, mk_node)) = eval_disj(&ds, &mut buf) else { break };
            cur_mk = mk_now;

            if cur_mk < best_mk {
                best_mk = cur_mk;
                best_ds = ds.clone();
                no_improve = 0;
                if let Ok(s) = disj_to_solution(pre, &ds, &buf.start) {
                    let _ = save_solution(&s);
                }
            } else {
                no_improve += 1;
            }

            let tails = compute_tails_pulsar(&ds, &buf);

            machine_pred_node.fill(NONE_USIZE);
            for seq in &ds.machine_seq {
                for i in 1..seq.len() { machine_pred_node[seq[i]] = seq[i - 1]; }
            }

            crit.fill(false);
            let mut u = mk_node;
            while u != NONE_USIZE { crit[u] = true; u = buf.best_pred[u]; }

            let mut best_move: Option<MoveHfs> = None;
            let mut best_move_mk = u32::MAX;
            let mut fallback_move: Option<MoveHfs> = None;
            let mut fallback_mk = u32::MAX;

            // N1: swap adjacent pairs in critical blocks
            for m in 0..ds.num_machines {
                if ds.machine_seq[m].len() <= 1 { continue; }
                let mut i = 0;
                while i < ds.machine_seq[m].len() {
                    if !crit[ds.machine_seq[m][i]] { i += 1; continue; }
                    let bstart = i;
                    let mut bend = i;
                    while bend + 1 < ds.machine_seq[m].len() {
                        let x = ds.machine_seq[m][bend];
                        let y = ds.machine_seq[m][bend + 1];
                        if !crit[y] { break; }
                        let end_x = buf.start[x].saturating_add(ds.node_pt[x]);
                        if buf.start[y] != end_x { break; }
                        bend += 1;
                    }
                    if bend > bstart {
                        let block_len = bend - bstart + 1;
                        let swap_positions = if block_len >= 3 { [bstart, bend - 1] } else { [bstart, NONE_USIZE] };
                        let num_swaps = if block_len >= 3 { 2 } else { 1 };
                        for si in 0..num_swaps {
                            let pos = swap_positions[si];
                            if pos == NONE_USIZE || pos + 1 >= ds.machine_seq[m].len() { continue; }
                            let node_u = ds.machine_seq[m][pos];
                            let node_v = ds.machine_seq[m][pos + 1];
                            let est_mk = estimate_swap_hfs(
                                node_u, node_v, &buf.start, &tails, &ds.node_pt,
                                &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ,
                            );
                            let key = (node_u.min(node_v), node_u.max(node_v));
                            let is_tabu = tabu_swap.get(&key).map_or(false, |&exp| iter < exp);
                            let aspiration = est_mk < best_mk;
                            if (!is_tabu || aspiration) && est_mk < best_move_mk {
                                best_move_mk = est_mk;
                                best_move = Some(MoveHfs::Swap { machine: m, pos });
                            }
                            if est_mk < fallback_mk {
                                fallback_mk = est_mk;
                                fallback_move = Some(MoveHfs::Swap { machine: m, pos });
                            }
                        }
                    }
                    i = bend + 1;
                }
            }

            // N2: machine reassignment of critical nodes (every 3rd iter)
            if iter % 3 == 0 {
                for node in 0..n {
                    if !crit[node] { continue; }
                    let job = ds.node_job[node];
                    let op_idx = ds.node_op[node];
                    let product = pre.job_products[job];
                    if op_idx >= pre.product_ops[product].len() { continue; }
                    let op_info = &pre.product_ops[product][op_idx];
                    if op_info.machines.len() <= 1 { continue; }
                    let cur_machine = ds.node_machine[node];

                    for &(new_m, new_pt) in &op_info.machines {
                        if new_m == cur_machine { continue; }
                        let key = (node, new_m);
                        let is_tabu = tabu_reassign.get(&key).map_or(false, |&exp| iter < exp);
                        let positions = find_insert_positions_hfs(&ds, &buf.start, node, new_m, &job_pred_node);
                        for insert_pos in positions {
                            let est_mk = estimate_reassign_hfs(
                                &ds, &buf.start, &tails,
                                node, new_m, new_pt, insert_pos,
                                &job_pred_node, &machine_pred_node, &buf.machine_succ,
                            );
                            let aspiration = est_mk < best_mk;
                            if (!is_tabu || aspiration) && est_mk < best_move_mk {
                                best_move_mk = est_mk;
                                best_move = Some(MoveHfs::Reassign { node, new_machine: new_m, new_pt, insert_pos });
                            }
                            if est_mk < fallback_mk {
                                fallback_mk = est_mk;
                                fallback_move = Some(MoveHfs::Reassign { node, new_machine: new_m, new_pt, insert_pos });
                            }
                        }
                    }
                }
            }

            match best_move.or(fallback_move) {
                Some(MoveHfs::Swap { machine: m, pos }) => {
                    let node_a = ds.machine_seq[m][pos];
                    let node_b = ds.machine_seq[m][pos + 1];
                    ds.machine_seq[m].swap(pos, pos + 1);
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    let offset = (pseed % ((2 * tenure_delta + 1) as u64)) as usize;
                    let this_tenure = (tenure + offset).saturating_sub(tenure_delta);
                    tabu_swap.insert((node_a.min(node_b), node_a.max(node_b)), iter + this_tenure);
                }
                Some(MoveHfs::Reassign { node, new_machine, new_pt, insert_pos }) => {
                    let old_machine = ds.node_machine[node];
                    if let Some(op) = ds.machine_seq[old_machine].iter().position(|&x| x == node) {
                        ds.machine_seq[old_machine].remove(op);
                    }
                    let ins = insert_pos.min(ds.machine_seq[new_machine].len());
                    ds.machine_seq[new_machine].insert(ins, node);
                    ds.node_machine[node] = new_machine;
                    ds.node_pt[node] = new_pt;
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    let offset = (pseed % ((2 * tenure_delta + 1) as u64)) as usize;
                    let this_tenure = (tenure + offset).saturating_sub(tenure_delta / 2);
                    tabu_reassign.insert((node, old_machine), iter + this_tenure);
                }
                None => break,
            }
            let _ = cur_mk;
        }

        // Final: evaluate best_ds
        let Some((_, _)) = eval_disj(&best_ds, &mut buf) else {
            return Ok((seed_sol.clone(), seed_mk));
        };
        match disj_to_solution(pre, &best_ds, &buf.start) {
            Ok(s) => Ok((s, best_mk)),
            Err(_) => Ok((seed_sol.clone(), seed_mk)),
        }
    }

    // --------------------------------------------------------------------------
    // Multi-resolution permutation ILS
    // Fast SA on job permutations (eval_hfs_perm_makespan) for inner loop.
    // Exact decode (decode_hfs_gap_filling) every 50 iters to validate.
    // SCD-SR polish triggered only on exact improvements. Budget: ~43s.
    // --------------------------------------------------------------------------

    fn fast_ils_hfs(
        pre: &Pre,
        challenge: &Challenge,
        base_sol: &Solution,
        base_mk: u32,
        rng: &mut SmallRng,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
    ) -> (Solution, u32) {
        let num_jobs = challenge.num_jobs;
        let num_machines = challenge.num_machines;

        // ARR-IG-FI: seeding only — FAST_ITERS=1k (~0.5s/nonce), main search is tabu_ils_hfs
        const FAST_ITERS: usize = 1_000;
        const DECODE_EVERY: usize = 50;
        const STAG_LIMIT: usize = 2_000;
        const FI_PASS_LIMIT: usize = 1;
        const ELITE_CAP: usize = 5;

        let mut seq = extract_sequence(base_sol, num_jobs);
        let mut machine_ready = vec![0u32; num_machines];

        let mut cur_mk = eval_hfs_perm_makespan(&seq, pre, &mut machine_ready);
        let mut best_fast_mk = cur_mk;
        let mut best_seq = seq.clone();
        let mut best_exact_sol = base_sol.clone();
        let mut best_exact_mk = base_mk;

        // Elite pool: (permutation, fast_makespan)
        let mut elite: Vec<(Vec<usize>, u32)> = Vec::with_capacity(ELITE_CAP);
        elite.push((seq.clone(), cur_mk));

        let mut stag_count: usize = 0;

        // SA base temperature: 4% of base_mk (was 1% in i73 — higher for more exploration)
        let t0 = base_mk as f64 * 0.04;

        for iter in 0..FAST_ITERS {
            // =================================================================
            // 1. NON-CONTIGUOUS DESTRUCTION (d random distinct indices)
            // =================================================================
            let d = 2usize.saturating_add(stag_count / 800).min(4);

            // Partial Fisher-Yates: pick d distinct indices from seq
            let mut idx_scratch: Vec<usize> = (0..num_jobs).collect();
            for i in 0..d {
                let j = rng.gen_range(i..num_jobs);
                idx_scratch.swap(i, j);
            }

            // Collect job values in random order (shuffled)
            let mut removed: Vec<usize> = idx_scratch[..d].iter().map(|&i| seq[i]).collect();
            removed.shuffle(rng);

            // Build new_seq by removing in descending index order (preserves index validity)
            let mut new_seq = seq.clone();
            let mut sorted_idx = idx_scratch[..d].to_vec();
            sorted_idx.sort_unstable_by(|a, b| b.cmp(a));
            for &idx in &sorted_idx {
                new_seq.remove(idx);
            }

            // =================================================================
            // 2. BIASED RANDOMIZED RECONSTRUCTION (RCL + roulette)
            // =================================================================
            let t_rec = (cur_mk as f64) * 0.012; // 1.2% of makespan for roulette temperature

            for &job in &removed {
                let m = new_seq.len();
                let k = (m / 4 + 1).min(5).max(2); // RCL size: 2..5

                // Evaluate all insertion positions
                let mut cands: Vec<(usize, u32)> = Vec::with_capacity(m + 1);
                for pos in 0..=m {
                    new_seq.insert(pos, job);
                    let mk = eval_hfs_perm_makespan(&new_seq, pre, &mut machine_ready);
                    cands.push((pos, mk));
                    new_seq.remove(pos);
                }
                cands.sort_by_key(|&(_, mk)| mk);

                let best_ins = cands[0].1;
                let top = &cands[..k.min(cands.len())];

                // Roulette weights: exp(-(mk - best) / t_rec)
                let mut weights: Vec<f64> = Vec::with_capacity(top.len());
                let mut sum_w = 0.0f64;
                for &(_, mk) in top {
                    let w = (-(mk.saturating_sub(best_ins) as f64) / t_rec.max(1.0)).exp();
                    weights.push(w);
                    sum_w += w;
                }

                let chosen_pos = if sum_w > 0.0 {
                    let mut r = rng.gen::<f64>() * sum_w;
                    let mut pos = top[0].0;
                    for (i, &w) in weights.iter().enumerate() {
                        r -= w;
                        if r <= 0.0 {
                            pos = top[i].0;
                            break;
                        }
                    }
                    pos
                } else {
                    top[0].0
                };

                new_seq.insert(chosen_pos, job);
            }

            // =================================================================
            // 3. POST-RECONSTRUCTION FIRST-IMPROVEMENT LOCAL SEARCH (1 pass)
            // =================================================================
            {
                let mut fi_pass = 0;
                let mut improved = true;
                while improved && fi_pass < FI_PASS_LIMIT {
                    improved = false;
                    fi_pass += 1;
                    let mut i = 0;
                    while i + 1 < new_seq.len() {
                        let mk_before = eval_hfs_perm_makespan(&new_seq, pre, &mut machine_ready);
                        new_seq.swap(i, i + 1);
                        let mk_after = eval_hfs_perm_makespan(&new_seq, pre, &mut machine_ready);
                        if mk_after < mk_before {
                            improved = true;
                        } else {
                            new_seq.swap(i, i + 1); // undo
                        }
                        i += 1;
                    }
                }
            }

            let new_mk = eval_hfs_perm_makespan(&new_seq, pre, &mut machine_ready);

            // =================================================================
            // 4. SA ACCEPTANCE with linear cooling + stagnation reheat
            // =================================================================
            let progress = iter as f64 / FAST_ITERS as f64;
            let mut temp = t0 * (1.0 - progress * 0.95).max(0.05);
            if stag_count > STAG_LIMIT / 2 {
                temp = temp.max(t0 * 0.30); // reheat floor on long stagnation
            }

            let accept = if new_mk <= cur_mk {
                true
            } else {
                let delta = new_mk.saturating_sub(cur_mk) as f64;
                rng.gen::<f64>() < (-delta / temp.max(1e-6)).exp()
            };

            let mut global_improved = false;
            if accept {
                seq = new_seq;
                cur_mk = new_mk;
                if cur_mk < best_fast_mk {
                    best_fast_mk = cur_mk;
                    best_seq = seq.clone();
                    global_improved = true;
                }
                // Update elite pool (deduplicate by makespan value)
                if elite.iter().all(|(_, m)| *m != cur_mk) {
                    elite.push((seq.clone(), cur_mk));
                    elite.sort_by_key(|(_, m)| *m);
                    if elite.len() > ELITE_CAP {
                        elite.truncate(ELITE_CAP);
                    }
                }
            }

            if global_improved {
                stag_count = 0;
            } else {
                stag_count += 1;
            }

            // =================================================================
            // 5. ELITE RESTART on stagnation
            // =================================================================
            if stag_count >= STAG_LIMIT {
                if !elite.is_empty() && rng.gen_bool(0.6) {
                    let eidx = rng.gen_range(0..elite.len());
                    seq = elite[eidx].0.clone();
                    cur_mk = elite[eidx].1;
                } else {
                    seq = best_seq.clone();
                    cur_mk = best_fast_mk;
                }
                // Strong random perturbation: 4 random remove-and-reinserts
                for _ in 0..4 {
                    let idx = rng.gen_range(0..seq.len());
                    let job = seq.remove(idx);
                    let pos = rng.gen_range(0..=seq.len());
                    seq.insert(pos, job);
                }
                cur_mk = eval_hfs_perm_makespan(&seq, pre, &mut machine_ready);
                stag_count = 0;
            }

            // =================================================================
            // 6. PERIODIC EXACT DECODE CHECKPOINT
            // =================================================================
            if iter % DECODE_EVERY == 0 || iter == FAST_ITERS - 1 {
                // Decode current best permutation with gap-filling decoder
                let (sol_best, exact_best) = decode_hfs_gap_filling(&best_seq, pre, challenge);
                if exact_best < best_exact_mk {
                    best_exact_mk = exact_best;
                    best_exact_sol = sol_best;
                    let _ = save_solution(&best_exact_sol);
                }
                // Also decode current seq
                let (sol_cur, exact_cur) = decode_hfs_gap_filling(&seq, pre, challenge);
                if exact_cur < best_exact_mk {
                    best_exact_mk = exact_cur;
                    best_exact_sol = sol_cur;
                    let _ = save_solution(&best_exact_sol);
                }
            }
        }

        (best_exact_sol, best_exact_mk)
    }

    // --------------------------------------------------------------------------
    // SBS-IG: Stage-by-Stage gap-filling decoder + sum_ct tiebreaker
    // --------------------------------------------------------------------------

    struct SbsWorkspace {
        ready_times: Vec<u32>,
        seq_rank: Vec<u32>,
        order: Vec<usize>,
        usages: Vec<Vec<(u32, u32)>>,
    }

    impl SbsWorkspace {
        fn new(num_jobs: usize, num_machines: usize) -> Self {
            Self {
                ready_times: vec![0; num_jobs],
                seq_rank: vec![0; num_jobs],
                order: Vec::with_capacity(num_jobs),
                usages: vec![Vec::with_capacity(num_jobs); num_machines],
            }
        }

        fn clear(&mut self) {
            self.ready_times.fill(0);
            self.seq_rank.fill(0);
            self.order.clear();
            for u in self.usages.iter_mut() { u.clear(); }
        }
    }

    fn eval_sbs_makespan(seq: &[usize], pre: &Pre, challenge: &Challenge, wk: &mut SbsWorkspace) -> (u32, u32) {
        wk.clear();
        for (idx, &j) in seq.iter().enumerate() { wk.seq_rank[j] = idx as u32; }
        wk.order.extend_from_slice(seq);
        if wk.order.is_empty() { return (0, 0); }

        for stage in 0..pre.max_ops {
            // Filter to jobs that have an op at this stage
            wk.order.retain(|&j| stage < pre.job_ops_len[j]);
            if wk.order.is_empty() { break; }

            // Sort by ready_time then seq_rank (stable priority from IG permutation)
            wk.order.sort_unstable_by(|&a, &b| {
                wk.ready_times[a].cmp(&wk.ready_times[b])
                    .then_with(|| wk.seq_rank[a].cmp(&wk.seq_rank[b]))
            });

            for &j in &wk.order {
                let product = pre.job_products[j];
                let op_info = &pre.product_ops[product][stage];
                let r_time = wk.ready_times[j];

                let mut best_m = 0usize;
                let mut best_end = u32::MAX;
                let mut best_start = 0u32;

                for &(m, pt) in &op_info.machines {
                    let st = find_earliest_gap(&wk.usages[m], r_time, pt);
                    let end = st + pt;
                    if end < best_end {
                        best_end = end;
                        best_start = st;
                        best_m = m;
                    }
                }
                if best_end < u32::MAX {
                    insert_usage(&mut wk.usages[best_m], best_start, best_end);
                    wk.ready_times[j] = best_end;
                }
            }

            // Restore full sequence for next stage
            wk.order.clear();
            wk.order.extend_from_slice(seq);
        }

        let mut mk = 0u32;
        let mut sum_ct = 0u64;
        for &j in seq {
            let ct = wk.ready_times[j];
            if ct > mk { mk = ct; }
            sum_ct += ct as u64;
        }
        (mk, sum_ct as u32)
    }

    fn build_sbs_solution(seq: &[usize], pre: &Pre, challenge: &Challenge) -> (Solution, u32) {
        let num_jobs = challenge.num_jobs;
        let num_machines = challenge.num_machines;
        let mut usages: Vec<Vec<(u32, u32)>> = vec![Vec::with_capacity(num_jobs); num_machines];
        let mut ready_times = vec![0u32; num_jobs];
        let mut seq_rank = vec![0u32; num_jobs];
        let mut job_schedule: Vec<Vec<(usize, u32)>> = vec![Vec::with_capacity(pre.max_ops); num_jobs];

        for (idx, &j) in seq.iter().enumerate() { seq_rank[j] = idx as u32; }
        let mut order: Vec<usize> = seq.to_vec();

        for stage in 0..pre.max_ops {
            order.retain(|&j| stage < pre.job_ops_len[j]);
            if order.is_empty() { break; }

            order.sort_unstable_by(|&a, &b| {
                ready_times[a].cmp(&ready_times[b])
                    .then_with(|| seq_rank[a].cmp(&seq_rank[b]))
            });

            for &j in &order {
                let product = pre.job_products[j];
                let op_info = &pre.product_ops[product][stage];
                let r_time = ready_times[j];

                let mut best_m = 0usize;
                let mut best_end = u32::MAX;
                let mut best_start = 0u32;

                for &(m, pt) in &op_info.machines {
                    let st = find_earliest_gap(&usages[m], r_time, pt);
                    let end = st + pt;
                    if end < best_end {
                        best_end = end;
                        best_start = st;
                        best_m = m;
                    }
                }
                if best_end < u32::MAX {
                    insert_usage(&mut usages[best_m], best_start, best_end);
                    job_schedule[j].push((best_m, best_start));
                    ready_times[j] = best_end;
                }
            }

            order.clear();
            order.extend_from_slice(seq);
        }

        let mk = ready_times.iter().copied().max().unwrap_or(0);
        (Solution { job_schedule }, mk)
    }

    #[allow(dead_code)]
    fn sbs_iterated_greedy_hfs(
        pre: &Pre,
        challenge: &Challenge,
        initial_seq: Vec<usize>,
        max_iters: usize,
        rng: &mut SmallRng,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        global_best_mk: &mut u32,
    ) -> Option<(Solution, u32)> {
        let num_jobs = challenge.num_jobs;
        if num_jobs <= 1 { return None; }
        let mut wk = SbsWorkspace::new(num_jobs, challenge.num_machines);

        let mut current_seq = initial_seq;
        let (mut current_mk, _) = eval_sbs_makespan(&current_seq, pre, challenge, &mut wk);
        let mut best_seq = current_seq.clone();
        let mut best_mk = current_mk;

        let d = 5usize;
        let temp = 0.5f64;
        let mut removed: Vec<usize> = Vec::with_capacity(d);
        let mut iters = 0u64;

        loop {
            if iters >= max_iters as u64 { break; }
            iters += 1;

            // Destruction
            let mut new_seq = current_seq.clone();
            removed.clear();
            for _ in 0..d {
                if new_seq.is_empty() { break; }
                let idx = rng.gen_range(0..new_seq.len());
                removed.push(new_seq.remove(idx));
            }

            // Greedy re-insertion using (mk, sum_ct) objective
            for &job in &removed {
                let mut best_pos = 0usize;
                let mut best_obj = (u32::MAX, u32::MAX);
                for pos in 0..=new_seq.len() {
                    new_seq.insert(pos, job);
                    let (mk, sct) = eval_sbs_makespan(&new_seq, pre, challenge, &mut wk);
                    if mk < best_obj.0 || (mk == best_obj.0 && sct < best_obj.1) {
                        best_obj = (mk, sct);
                        best_pos = pos;
                    }
                    new_seq.remove(pos);
                }
                new_seq.insert(best_pos, job);
            }

            let (new_mk, _) = eval_sbs_makespan(&new_seq, pre, challenge, &mut wk);

            if new_mk < best_mk {
                best_mk = new_mk;
                best_seq = new_seq.clone();
                if new_mk < *global_best_mk {
                    *global_best_mk = new_mk;
                    let (sol, _) = build_sbs_solution(&best_seq, pre, challenge);
                    let _ = save_solution(&sol);
                }
            }

            if new_mk <= current_mk {
                current_seq = new_seq;
                current_mk = new_mk;
            } else {
                let delta = (new_mk - current_mk) as f64;
                let p = (-delta / temp).exp();
                if rng.gen::<f64>() < p {
                    current_seq = new_seq;
                    current_mk = new_mk;
                }
            }
        }

        Some(build_sbs_solution(&best_seq, pre, challenge))
    }

    // --------------------------------------------------------------------------
    // Iterated Greedy with ECT Gap-Filling decoder
    // Permutation-based search adapted for HFS: decoder does dynamic machine routing.
    // --------------------------------------------------------------------------

    #[inline]
    fn find_earliest_gap(usage: &[(u32, u32)], ready_time: u32, pt: u32) -> u32 {
        let mut s = ready_time;
        for &(start, end) in usage {
            if s + pt <= start {
                return s;
            }
            if s < end {
                s = end;
            }
        }
        s
    }

    #[inline]
    fn insert_usage(usage: &mut Vec<(u32, u32)>, start: u32, end: u32) {
        let pos = usage.binary_search_by_key(&start, |&(s, _)| s).unwrap_or_else(|e| e);
        usage.insert(pos, (start, end));
    }

    pub fn decode_hfs_gap_filling(
        seq: &[usize],
        pre: &Pre,
        challenge: &Challenge,
    ) -> (Solution, u32) {
        let mut usage = vec![Vec::<(u32, u32)>::with_capacity(challenge.num_jobs); challenge.num_machines];
        let mut schedule: Vec<Vec<(usize, u32)>> = vec![Vec::with_capacity(pre.max_ops); challenge.num_jobs];
        let mut makespan = 0u32;

        for &j in seq {
            let mut t = 0u32;
            let prod = pre.job_products[j];
            for op_idx in 0..pre.job_ops_len[j] {
                let eligible = &pre.product_ops[prod][op_idx].machines;
                let mut best_m = 0usize;
                let mut best_start = 0u32;
                let mut best_end = u32::MAX;
                for &(m, pt) in eligible {
                    let s = find_earliest_gap(&usage[m], t, pt);
                    let e = s + pt;
                    if e < best_end {
                        best_end = e;
                        best_m = m;
                        best_start = s;
                    }
                }
                insert_usage(&mut usage[best_m], best_start, best_end);
                schedule[j].push((best_m, best_start));
                t = best_end;
            }
            if t > makespan {
                makespan = t;
            }
        }
        (Solution { job_schedule: schedule }, makespan)
    }

    pub fn extract_sequence(sol: &Solution, num_jobs: usize) -> Vec<usize> {
        let mut starts: Vec<(usize, u32)> = Vec::with_capacity(num_jobs);
        for j in 0..num_jobs {
            let st = sol.job_schedule[j].first().map_or(0u32, |&(_, s)| s);
            starts.push((j, st));
        }
        starts.sort_unstable_by_key(|&(_, st)| st);
        starts.into_iter().map(|(j, _)| j).collect()
    }

    pub fn iterated_greedy_hfs(
        initial_seq: Vec<usize>,
        pre: &Pre,
        challenge: &Challenge,
        iters: usize,
        rng: &mut SmallRng,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        global_best_mk: &mut u32,
    ) -> Result<(Solution, u32)> {
        let mut current_seq = initial_seq.clone();
        let (mut current_sol, mut current_mk) = decode_hfs_gap_filling(&current_seq, pre, challenge);

        let mut best_sol = current_sol.clone();
        let mut best_mk = current_mk;

        let d = 4usize;
        let t_base = 2.0f64;

        for _ in 0..iters {
            let mut temp_seq = current_seq.clone();
            let mut removed: Vec<usize> = Vec::with_capacity(d);

            for _ in 0..d {
                if temp_seq.is_empty() { break; }
                let idx = rng.gen_range(0..temp_seq.len());
                removed.push(temp_seq.remove(idx));
            }

            for &job in &removed {
                let mut best_ins_mk = {
                    let mut s0 = temp_seq.clone();
                    s0.insert(0, job);
                    decode_hfs_gap_filling(&s0, pre, challenge).1
                };
                let mut best_pos = 0usize;
                for pos in 1..=temp_seq.len() {
                    let mut test_seq = temp_seq.clone();
                    test_seq.insert(pos, job);
                    let (_, mk) = decode_hfs_gap_filling(&test_seq, pre, challenge);
                    if mk < best_ins_mk {
                        best_ins_mk = mk;
                        best_pos = pos;
                    }
                }
                temp_seq.insert(best_pos, job);
            }

            let (new_sol, new_mk) = decode_hfs_gap_filling(&temp_seq, pre, challenge);

            if new_mk < best_mk {
                best_mk = new_mk;
                best_sol = new_sol.clone();
                if best_mk < *global_best_mk {
                    *global_best_mk = best_mk;
                    let _ = save_solution(&best_sol);
                }
            }

            if new_mk <= current_mk {
                current_mk = new_mk;
                current_seq = temp_seq;
                current_sol = new_sol;
            } else {
                let delta = (new_mk - current_mk) as f64;
                let p = (-delta / t_base).exp();
                if rng.gen::<f64>() < p {
                    current_mk = new_mk;
                    current_seq = temp_seq;
                    current_sol = new_sol;
                }
            }
        }
        let _ = current_sol;
        Ok((best_sol, best_mk))
    }

    // --------------------------------------------------------------------------
    // Fast Permutation IG with allocation-free ECT decoder
    // Evaluates permutations without allocating any Vec in the inner loop.
    // 50,000+ iterations/nonce vs ~200 for the gap-filling version.
    // --------------------------------------------------------------------------

    #[inline(always)]
    fn eval_hfs_perm_makespan(seq: &[usize], pre: &Pre, machine_ready: &mut [u32]) -> u32 {
        machine_ready.fill(0);
        let mut makespan = 0u32;
        for &j in seq {
            let p = pre.job_products[j];
            let ops = &pre.product_ops[p];
            let mut prev_ready = 0u32;
            for op in ops {
                let mut best_end = u32::MAX;
                let mut best_m = 0usize;
                for &(m, pt) in &op.machines {
                    let start = prev_ready.max(machine_ready[m]);
                    let end = start.saturating_add(pt);
                    if end < best_end {
                        best_end = end;
                        best_m = m;
                    }
                }
                if best_end < u32::MAX {
                    machine_ready[best_m] = best_end;
                    prev_ready = best_end;
                }
            }
            if prev_ready > makespan {
                makespan = prev_ready;
            }
        }
        makespan
    }

    fn decode_perm_to_solution(seq: &[usize], pre: &Pre, num_machines: usize) -> (Solution, u32) {
        let mut machine_ready = vec![0u32; num_machines];
        let mut job_schedule: Vec<Vec<(usize, u32)>> = seq.iter().map(|_| Vec::with_capacity(pre.max_ops)).collect();
        let mut makespan = 0u32;
        for &j in seq {
            let p = pre.job_products[j];
            let mut prev = 0u32;
            for op in &pre.product_ops[p] {
                let mut best_end = u32::MAX;
                let mut best_m = 0usize;
                let mut best_st = 0u32;
                for &(m, pt) in &op.machines {
                    let st = prev.max(machine_ready[m]);
                    let end = st.saturating_add(pt);
                    if end < best_end {
                        best_end = end;
                        best_m = m;
                        best_st = st;
                    }
                }
                if best_end < u32::MAX {
                    job_schedule[j].push((best_m, best_st));
                    machine_ready[best_m] = best_end;
                    prev = best_end;
                }
            }
            if prev > makespan {
                makespan = prev;
            }
        }
        (Solution { job_schedule }, makespan)
    }

    #[allow(dead_code)]
    fn fast_permutation_ig(
        pre: &Pre,
        challenge: &Challenge,
        base_sol: &Solution,
        max_iters: usize,
        rng: &mut SmallRng,
    ) -> Option<(Solution, u32)> {
        let num_jobs = challenge.num_jobs;
        if num_jobs <= 1 { return None; }
        let num_machines = challenge.num_machines;

        let mut current_seq = extract_sequence(base_sol, num_jobs);
        let mut eval_buf = vec![0u32; num_machines];

        let mut current_mk = eval_hfs_perm_makespan(&current_seq, pre, &mut eval_buf);
        let base_mk = current_mk;

        let mut best_seq = current_seq.clone();
        let mut best_mk = current_mk;

        let d = 4usize;
        let mut temp_seq: Vec<usize> = Vec::with_capacity(num_jobs);
        let mut removed: Vec<usize> = Vec::with_capacity(d);

        let t_start = 5.0f64;
        let mut t = t_start;
        let alpha = 0.9995f64;

        let mut iters = 0u64;
        loop {
            if iters >= max_iters as u64 {
                break;
            }
            iters += 1;

            // Destruction
            temp_seq.clear();
            temp_seq.extend_from_slice(&current_seq);
            removed.clear();
            for _ in 0..d {
                if temp_seq.is_empty() { break; }
                let idx = rng.gen_range(0..temp_seq.len());
                removed.push(temp_seq.remove(idx));
            }

            // Greedy construction — best insertion position for each removed job
            for &job in &removed {
                let mut best_ins_mk = u32::MAX;
                let mut best_ins_pos = 0usize;
                for pos in 0..=temp_seq.len() {
                    temp_seq.insert(pos, job);
                    let mk = eval_hfs_perm_makespan(&temp_seq, pre, &mut eval_buf);
                    if mk < best_ins_mk {
                        best_ins_mk = mk;
                        best_ins_pos = pos;
                    }
                    temp_seq.remove(pos);
                }
                temp_seq.insert(best_ins_pos, job);
            }

            let new_mk = eval_hfs_perm_makespan(&temp_seq, pre, &mut eval_buf);

            if new_mk < best_mk {
                best_mk = new_mk;
                best_seq.copy_from_slice(&temp_seq);
                current_seq.copy_from_slice(&temp_seq);
                current_mk = new_mk;
            } else if new_mk <= current_mk {
                current_seq.copy_from_slice(&temp_seq);
                current_mk = new_mk;
            } else {
                let delta = (new_mk - current_mk) as f64;
                if rng.gen::<f64>() < (-delta / t).exp() {
                    current_seq.copy_from_slice(&temp_seq);
                    current_mk = new_mk;
                }
            }
            t *= alpha;
        }

        if best_mk < base_mk {
            let (sol, mk) = decode_perm_to_solution(&best_seq, pre, num_machines);
            Some((sol, mk))
        } else {
            None
        }
    }

    // --------------------------------------------------------------------------
    // Dynamic Memetic IG with gap-filling decoder + Order Crossover
    // Dynamic d: starts at 4, escalates to 8/12 when stuck, recenters at 150
    // OX crossover every 150 iters to bridge separate local optima
    // --------------------------------------------------------------------------

    fn update_ig_pool(pool: &mut Vec<(Vec<usize>, u32)>, seq: Vec<usize>, mk: u32) {
        if !pool.iter().any(|(_, m)| *m == mk) {
            pool.push((seq, mk));
            pool.sort_unstable_by_key(|(_, m)| *m);
            if pool.len() > 4 { pool.truncate(4); }
        }
    }

    fn order_crossover(p1: &[usize], p2: &[usize], rng: &mut SmallRng) -> Vec<usize> {
        let n = p1.len();
        if n == 0 { return Vec::new(); }
        let mut child = vec![usize::MAX; n];
        let mut start = rng.gen_range(0..n);
        let mut end = rng.gen_range(0..n);
        if start > end { let tmp = start; start = end; end = tmp; }
        let mut in_child = vec![false; n];
        for i in start..=end {
            child[i] = p1[i];
            if p1[i] < n { in_child[p1[i]] = true; }
        }
        let mut p2_idx = (end + 1) % n;
        let mut child_idx = (end + 1) % n;
        for _ in 0..n {
            let job = p2[p2_idx];
            if job < n && !in_child[job] {
                child[child_idx] = job;
                in_child[job] = true;
                child_idx = (child_idx + 1) % n;
            }
            p2_idx = (p2_idx + 1) % n;
        }
        child
    }

    #[allow(dead_code)]
    fn dynamic_memetic_ig(
        pre: &Pre,
        challenge: &Challenge,
        elite_pool: &mut Vec<(Vec<usize>, u32)>,
        max_iters: usize,
        rng: &mut SmallRng,
        save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
        global_best_mk: &mut u32,
    ) -> Option<(Vec<usize>, u32)> {
        if elite_pool.is_empty() { return None; }
        let num_jobs = challenge.num_jobs;

        let mut current_seq = elite_pool[0].0.clone();
        let mut current_mk = elite_pool[0].1;
        let mut best_seq = current_seq.clone();
        let mut best_mk = current_mk;

        let mut d = 4usize;
        let mut stuck_iters = 0usize;
        let mut loop_counter = 0u64;
        let temp_scale = pre.avg_op_min * 2.0;

        let mut temp_seq: Vec<usize> = Vec::with_capacity(num_jobs);
        let mut removed: Vec<usize> = Vec::with_capacity(16);

        loop {
            if loop_counter >= max_iters as u64 { break; }
            loop_counter += 1;

            // Periodic OX crossover
            if loop_counter % 150 == 0 && elite_pool.len() > 1 {
                let p2_idx = rng.gen_range(1..elite_pool.len());
                let ox_seq = order_crossover(&elite_pool[0].0, &elite_pool[p2_idx].0, rng);
                let (ox_sol, ox_mk) = decode_hfs_gap_filling(&ox_seq, pre, challenge);
                if ox_mk < best_mk {
                    best_mk = ox_mk;
                    best_seq = ox_seq.clone();
                    let _ = save_solution(&ox_sol);
                    *global_best_mk = (*global_best_mk).min(ox_mk);
                    update_ig_pool(elite_pool, ox_seq.clone(), ox_mk);
                }
                current_seq = ox_seq;
                current_mk = ox_mk;
                d = 4;
                continue;
            }

            // Destruction
            temp_seq.clear();
            temp_seq.extend_from_slice(&current_seq);
            removed.clear();
            for _ in 0..d {
                if temp_seq.is_empty() { break; }
                let idx = rng.gen_range(0..temp_seq.len());
                removed.push(temp_seq.remove(idx));
            }

            // Construction with gap-filling decoder
            for &job in &removed {
                let mut best_ins_mk = u32::MAX;
                let mut best_ins_pos = 0usize;
                for pos in 0..=temp_seq.len() {
                    temp_seq.insert(pos, job);
                    let (_, mk) = decode_hfs_gap_filling(&temp_seq, pre, challenge);
                    if mk < best_ins_mk {
                        best_ins_mk = mk;
                        best_ins_pos = pos;
                    }
                    temp_seq.remove(pos);
                }
                temp_seq.insert(best_ins_pos, job);
            }

            let (trial_sol, trial_mk) = decode_hfs_gap_filling(&temp_seq, pre, challenge);

            // Acceptance + dynamic d escalation
            if trial_mk < best_mk {
                best_mk = trial_mk;
                best_seq.clear();
                best_seq.extend_from_slice(&temp_seq);
                let _ = save_solution(&trial_sol);
                *global_best_mk = (*global_best_mk).min(best_mk);
                update_ig_pool(elite_pool, temp_seq.clone(), trial_mk);
                current_seq.clear();
                current_seq.extend_from_slice(&temp_seq);
                current_mk = trial_mk;
                stuck_iters = 0;
                d = 4;
            } else {
                stuck_iters += 1;
                let delta = (trial_mk as f64) - (current_mk as f64);
                let prob = (-delta / temp_scale.max(1.0)).exp();
                if rng.gen::<f64>() < prob {
                    current_seq.clear();
                    current_seq.extend_from_slice(&temp_seq);
                    current_mk = trial_mk;
                }
                d = if stuck_iters > 150 {
                    current_seq.clear(); current_seq.extend_from_slice(&best_seq);
                    current_mk = best_mk; stuck_iters = 0; 4
                } else if stuck_iters > 100 { 12 }
                else if stuck_iters > 50 { 8 }
                else { 4 };
            }
        }

        if best_mk < elite_pool.first().map(|(_, m)| *m).unwrap_or(u32::MAX) {
            Some((best_seq, best_mk))
        } else {
            Some((best_seq, best_mk))
        }
    }

}

pub mod job_shop {
    use anyhow::{anyhow, Result};
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    use std::collections::HashMap;
    use tig_challenges::job_scheduling::*;
    use super::types::*;
    use super::infra::*;


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
    fn score_candidate(
        pre: &Pre, rule: Rule, job: usize, product: usize, op_idx: usize,
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
        match rule {
            Rule::CriticalPath => { let next_term = next_w_base*0.30*next_term_raw; let slack_term = slack_w*slack_u*slack_reg_boost; (1.03*rem_min_n)+(0.10*ops_n)+(0.24*scarcity_urg)+(0.20*pre.flex_factor)*flex_inv+next_term+0.10*slack_term-(0.70*end_n)-pop_pen+(0.45*job_bias)+flow_term+route_term+jitter }
            Rule::MostWork => { let next_term = next_w_base*0.25*next_term_raw; (1.00*rem_avg_n)+(0.12*ops_n)+(0.18*scarcity_urg)+(0.15*pre.flex_factor)*flex_inv+next_term-(0.62*end_n)-pop_pen+(0.45*job_bias)+flow_term+route_term+jitter }
            Rule::LeastFlex => { let next_term = next_w_base*0.20*next_term_raw; (1.00*flex_inv)+(0.28*rem_min_n)+(0.22*scarcity_urg)+next_term-(0.55*end_n)-pop_pen+(0.35*job_bias)+flow_term+route_term+jitter }
            Rule::ShortestProc => { let next_term = next_w_base*0.20*next_term_raw; (-1.00*proc_n)+(0.25*rem_min_n)+(0.12*scarcity_urg)+next_term-(0.20*end_n)-pop_pen+(0.25*job_bias)+flow_term+route_term+jitter }
            Rule::Regret => { let next_term = next_w_base*0.25*next_term_raw; (1.05*reg_n)+(0.55*rem_min_n)+(0.22*scarcity_urg)+next_term-(0.68*end_n)-pop_pen+(0.35*job_bias)+flow_term+route_term+jitter }
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
            let mk = rule_best[rule_idx(r)]; let t = rule_tries[rule_idx(r)].max(1) as f64;
            let delta = mk.saturating_sub(best_seen) as f64; let exploit = (-delta/scale).exp(); let explore = (1.0/t).sqrt();
            let mut ww = (1.0-explore_mix)*exploit+explore_mix*explore; ww = ww.max(1e-6);
            if chaos_like { ww = ww.powf(0.70); } else if late_phase { ww = ww.powf(1.18); }
            w[i] = ww;
        }
        let mut sum = 0.0; for &ww in &w { sum += ww.max(0.0); }
        if !(sum > 0.0) { return rules[rng.gen_range(0..rules.len())]; }
        let mut r = rng.gen::<f64>() * sum;
        for (i, &ww) in w.iter().enumerate() { r -= ww.max(0.0); if r <= 0.0 { return rules[i]; } }
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
        let mut idle_machines: Vec<usize> = Vec::with_capacity(num_machines);
        let chaotic_like = pre.chaotic_like;
        let mut machine_work: Vec<u64> = if chaotic_like { vec![0u64; num_machines] } else { vec![] };
        let mut sum_work: u64 = 0;
        while remaining_ops > 0 {
            loop {
                idle_machines.clear();
                for m in 0..num_machines { if machine_avail[m] <= time { idle_machines.push(m); } }
                if idle_machines.is_empty() { break; }
                for &m in &idle_machines { demand[m] = 0; raw_by_machine[m].clear(); }
                let progress = 1.0 - (remaining_ops as f64) / (pre.total_ops as f64).max(1.0);
                let cap_per_machine = if k == 0 { 12usize } else { (k+6).min(12) };
                for job in 0..num_jobs {
                    let op_idx = job_next_op[job];
                    if op_idx >= pre.job_ops_len[job] || job_ready_time[job] > time { continue; }
                    let product = pre.job_products[job]; let op = &pre.product_ops[product][op_idx];
                    if op.flex == 0 || op.machines.is_empty() || op.min_pt >= INF { continue; }
                    let (best_end, second_end, best_cnt_total, best_cnt_idle) = best_second_and_counts(time, &machine_avail, op);
                    if best_end >= INF || best_cnt_idle == 0 { continue; }
                    let ops_rem = pre.job_ops_len[job] - op_idx; let jb = job_bias.map(|v| v[job]).unwrap_or(0.0);
                    let flex_inv = 1.0/(op.flex as f64).max(1.0); let scarcity_urg = 1.0/(best_cnt_total as f64).max(1.0);
                    let regret = if second_end >= INF { pre.avg_op_min*2.6 } else { (second_end-best_end) as f64 };
                    let regn = (regret/pre.avg_op_min.max(1.0)).clamp(0.0,6.0); let rigidity = (0.60*flex_inv+0.40*scarcity_urg).clamp(0.0,2.5);
                    for &(m, pt) in &op.machines {
                        if machine_avail[m] > time { continue; }
                        let end = time.saturating_add(pt); if end != best_end { continue; }
                        demand[m] = demand[m].saturating_add(1);
                        let mp = machine_penalty.map(|v| v[m]).unwrap_or(0.0); let jitter = if k > 0 { rng.gen::<f64>()*1e-9 } else { 0.0 };
                        let base = score_candidate(pre, rule, job, product, op_idx, ops_rem, op, m, pt, time, target_mk, best_end, second_end, best_cnt_total, progress, jb, mp, machine_load[m], route_pref, route_w, jitter);
                        push_top_k_raw(&mut raw_by_machine[m], RawCand { job, machine: m, pt, base_score: base, rigidity, reg_n: regn }, cap_per_machine);
                    }
                }
                let denom = (idle_machines.len() as f64).max(1.0);
                let (conflict_w, conflict_scale) = if chaotic_like { (-(0.05+0.08*(1.0-progress)).clamp(0.04,0.14), (0.95+0.20*pre.flex_factor).clamp(0.90,1.20)) } else { ((0.09+0.26*pre.jobshopness+0.11*pre.high_flex+0.16*(1.0-progress)).clamp(0.05,0.45), (0.90+0.40*pre.flex_factor).clamp(0.85,1.75)) };
                let (bal_w, avg_work) = if chaotic_like { ((0.030+0.070*(1.0-progress)).clamp(0.025,0.11), (sum_work as f64)/(num_machines as f64).max(1.0)) } else { (0.0, 0.0) };
                let mut best: Option<Cand> = None; let mut top: Vec<Cand> = if k > 0 { Vec::with_capacity(k) } else { Vec::new() };
                for &m in &idle_machines {
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
                if chaotic_like { machine_work[machine]=machine_work[machine].saturating_add(pt as u64); sum_work=sum_work.saturating_add(pt as u64); }
                if op.min_pt < INF && op.flex > 0 && !op.machines.is_empty() { let delta=(op.min_pt as f64)/(op.flex as f64).max(1.0); if delta>0.0 { for &(mm,_) in &op.machines { let v=machine_load[mm]-delta; machine_load[mm]=if v>0.0{v}else{0.0}; } } }
                if remaining_ops == 0 { break; }
            }
            if remaining_ops == 0 { break; }
            let mut next_time: Option<u32> = None;
            for &t in &machine_avail { if t > time { next_time = Some(next_time.map_or(t, |b| b.min(t))); } }
            for j in 0..num_jobs { let op_idx=job_next_op[j]; if op_idx < pre.job_ops_len[j] && job_ready_time[j] > time { let t=job_ready_time[j]; next_time=Some(next_time.map_or(t,|b|b.min(t))); } }
            time = next_time.ok_or_else(|| anyhow!("Stalled"))?;
        }
        let mk = machine_avail.into_iter().max().unwrap_or(0);
        Ok((Solution { job_schedule }, mk))
    }

    pub fn tabu_search_phase(pre: &Pre, challenge: &Challenge, base_sol: &Solution, max_iterations: usize, tenure_base: usize) -> Result<Option<(Solution, u32)>> {
        let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
        let mut buf = EvalBuf::new(ds.n); let n = ds.n;
        let Some(init_eval) = eval_disj(&ds, &mut buf) else { return Ok(None) };
        let initial_mk = init_eval.0; let mut best_global_mk = initial_mk; let mut best_global_ds = ds.clone();
        let tenure = tenure_base.max(5); let tenure_delta = (tenure/3).max(2); let max_no_improve = (max_iterations/2).max(60);
        let mut tabu: HashMap<(usize,usize),usize> = HashMap::with_capacity(tenure*8);
        let mut crit = vec![false; n]; let mut no_improve = 0usize;
        let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (initial_mk as u64).wrapping_shl(16) ^ (n as u64).wrapping_mul(0x517CC1B727220A95);
        let mut tail = vec![0u32; n]; let mut back_deg = vec![0u16; n]; let mut back_stack: Vec<usize> = Vec::with_capacity(n);
        let mut machine_pred_node = vec![NONE_USIZE; n]; let mut job_pred_node = vec![NONE_USIZE; n];
        for j in 0..ds.num_jobs { let base = ds.job_offsets[j]; let end = ds.job_offsets[j+1]; for k in (base+1)..end { job_pred_node[k] = k-1; } }
        let kick_threshold = (max_no_improve*2/3).max(40); let mut kicks_left = 5usize;
        for iter in 0..max_iterations {
            if no_improve >= max_no_improve {
                if kicks_left == 0 { break; }
                ds = best_global_ds.clone(); no_improve = 0; kicks_left -= 1; tabu.clear(); continue;
            }
            if no_improve > 0 && no_improve % kick_threshold == 0 && kicks_left > 0 {
                let Some((_, kick_mk_node)) = eval_disj(&ds, &mut buf) else { break };
                crit.fill(false); let mut u = kick_mk_node; while u != NONE_USIZE { crit[u]=true; u=buf.best_pred[u]; }
                let mut kick_swaps: Vec<(usize,usize)> = Vec::new();
                for m in 0..ds.num_machines { if ds.machine_seq[m].len() <= 1 { continue; } for i in 0..(ds.machine_seq[m].len()-1) { if crit[ds.machine_seq[m][i]] || crit[ds.machine_seq[m][i+1]] { kick_swaps.push((m,i)); } } }
                if !kick_swaps.is_empty() {
                    let num_kicks = (3 + (no_improve / kick_threshold)).min(5);
                    for _ in 0..num_kicks { pseed^=pseed.wrapping_shl(13); pseed^=pseed.wrapping_shr(7); pseed^=pseed.wrapping_shl(17); let idx=(pseed as usize)%kick_swaps.len(); let (m,pos)=kick_swaps[idx]; if pos+1<ds.machine_seq[m].len() { ds.machine_seq[m].swap(pos,pos+1); } }
                }
                kicks_left -= 1; continue;
            }
            let Some((cur_mk, mk_node)) = eval_disj(&ds, &mut buf) else { break };
            if iter > 0 { if cur_mk < best_global_mk { best_global_mk = cur_mk; best_global_ds = ds.clone(); no_improve = 0; } else { no_improve += 1; } }
            machine_pred_node.fill(NONE_USIZE);
            for seq in &ds.machine_seq { for i in 1..seq.len() { machine_pred_node[seq[i]] = seq[i-1]; } }
            tail.fill(0); back_deg.fill(0);
            for i in 0..n { if ds.job_succ[i] != NONE_USIZE { back_deg[i] += 1; } if buf.machine_succ[i] != NONE_USIZE { back_deg[i] += 1; } }
            back_stack.clear(); for i in 0..n { if back_deg[i] == 0 { back_stack.push(i); } }
            while let Some(nd) = back_stack.pop() {
                let contrib = ds.node_pt[nd].saturating_add(tail[nd]);
                let jp = job_pred_node[nd]; if jp != NONE_USIZE { if contrib > tail[jp] { tail[jp] = contrib; } back_deg[jp] = back_deg[jp].saturating_sub(1); if back_deg[jp] == 0 { back_stack.push(jp); } }
                let mp = machine_pred_node[nd]; if mp != NONE_USIZE { if contrib > tail[mp] { tail[mp] = contrib; } back_deg[mp] = back_deg[mp].saturating_sub(1); if back_deg[mp] == 0 { back_stack.push(mp); } }
            }
            crit.fill(false); let mut u = mk_node; while u != NONE_USIZE { crit[u]=true; u=buf.best_pred[u]; }
            let mut best_move: Option<(usize,usize,u32)> = None; let mut best_move_mk = u32::MAX;
            let mut fallback_move: Option<(usize,usize,u32)> = None; let mut fallback_mk = u32::MAX;
            for m in 0..ds.num_machines {
                if ds.machine_seq[m].len() <= 1 { continue; }
                let mut blocks: Vec<(usize,usize)> = Vec::new(); let mut i = 0;
                while i < ds.machine_seq[m].len() {
                    if !crit[ds.machine_seq[m][i]] { i+=1; continue; }
                    let bstart=i; let mut bend=i;
                    while bend+1 < ds.machine_seq[m].len() { let x=ds.machine_seq[m][bend]; let y=ds.machine_seq[m][bend+1]; if !crit[y] { break; } let end_x=buf.start[x].saturating_add(ds.node_pt[x]); if buf.start[y]!=end_x { break; } bend+=1; }
                    if bend > bstart { blocks.push((bstart,bend)); }
                    i = bend+1;
                }
                for &(bstart,bend) in &blocks {
                    let block_len = bend-bstart+1;
                    let mut swap_positions = [bstart,NONE_USIZE]; let num_swaps = if block_len>=3 { swap_positions[1]=bend-1; 2 } else { 1 };
                    for si in 0..num_swaps {
                        let pos=swap_positions[si]; if pos+1>=ds.machine_seq[m].len() { continue; }
                        let node_u=ds.machine_seq[m][pos]; let node_v=ds.machine_seq[m][pos+1];
                        let est_mk = estimate_swap_mk(node_u, node_v, &buf.start, &tail, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ);
                        let key=(node_u.min(node_v),node_u.max(node_v)); let is_tabu=tabu.get(&key).map_or(false,|&exp| iter<exp); let aspiration=est_mk<best_global_mk;
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
                    pseed^=pseed.wrapping_shl(13); pseed^=pseed.wrapping_shr(7); pseed^=pseed.wrapping_shl(17);
                    let offset=(pseed%((2*tenure_delta+1) as u64)) as usize;
                    let progress=(iter as f64)/(max_iterations as f64); let late_bonus=if progress>0.6{((progress-0.6)*10.0) as usize}else{0};
                    let this_tenure=(tenure+offset+late_bonus).saturating_sub(tenure_delta);
                    let key=(node_a.min(node_b),node_a.max(node_b)); tabu.insert(key,iter+this_tenure);
                }
                None => break,
            }
        }
        let Some((final_mk,_)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
        if final_mk < best_global_mk { best_global_mk = final_mk; best_global_ds = ds.clone(); }
        if best_global_mk >= initial_mk { return Ok(None); }
        ds = best_global_ds;
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
            for base_sol in &bn_pool {
                let mut ds = match build_disj_from_solution(pre, challenge, base_sol) { Ok(d) => d, Err(_) => continue };
                let mut buf = EvalBuf::new(ds.n);
                let Some((mut cur_mk, _)) = eval_disj(&ds, &mut buf) else { continue };
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
                    let mut found_improvement = true;
                    while found_improvement {
                        found_improvement = false;
                        'swap_loop: for i in 0..seq_cap.saturating_sub(1) {
                            for j in (i+1)..seq_cap {
                                if j >= ds.machine_seq[m].len() { break; }
                                ds.machine_seq[m].swap(i, j);
                                if let Some((new_mk, _)) = eval_disj(&ds, &mut buf) {
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
                    if let Some((mk_bn, _)) = eval_disj(&ds, &mut buf) {
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

                let (sol_a, mk_a) = mem_pop[ia].clone();
                let (sol_b, mk_b) = mem_pop[ib].clone();

                let ds_a = match build_disj_from_solution(pre, challenge, &sol_a) { Ok(d) => d, Err(_) => { gen_no_improve += 1; continue; } };
                let ds_b = match build_disj_from_solution(pre, challenge, &sol_b) { Ok(d) => d, Err(_) => { gen_no_improve += 1; continue; } };

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

                let mut child_buf = EvalBuf::new(child_ds.n);
                let child_mk = match eval_disj(&child_ds, &mut child_buf) {
                    Some((mk, _)) => mk,
                    None => { gen_no_improve += 1; continue; }
                };
                let child_sol = match disj_to_solution(pre, &child_ds, &child_buf.start) {
                    Ok(s) => s,
                    Err(_) => { gen_no_improve += 1; continue; }
                };

                let (ls_sol, ls_mk) = match critical_block_move_local_search_ex(pre, challenge, &child_sol, 4, 250, 100) {
                    Ok(Some((s, mk))) => (s, mk),
                    _ => (child_sol, child_mk),
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
    // TIG's UI uses the pattern `tig_challenges::job_scheduling` to automatically detect your algorithm's challenge
    use anyhow::Result;
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    use tig_challenges::job_scheduling::*;
    use std::collections::HashMap;

    use super::types::*;
    use super::infra::*;

    // ============================================================================
    // TABU SEARCH — integrated swap + reassign on disjunctive graph
    // ============================================================================

    /// O(1) estimate of makespan after swapping adjacent nodes u, v on same machine.
    /// u is at pos, v is at pos+1. After swap: v first, then u.
    #[inline]
    fn estimate_swap_fm(
        u: usize, v: usize,
        heads: &[u32], tails: &[u32], pt: &[u32],
        job_pred: &[usize], job_succ: &[usize],
        machine_pred: &[usize], machine_succ: &[usize],
    ) -> u32 {
        let mp_u = machine_pred[u];
        let ms_v = machine_succ[v];
        let jp_v = job_pred[v];
        let jp_u = job_pred[u];
        let js_u = job_succ[u];
        let js_v = job_succ[v];

        let r_mp_u = if mp_u != NONE_USIZE { heads[mp_u].saturating_add(pt[mp_u]) } else { 0 };
        let r_jp_v = if jp_v != NONE_USIZE { heads[jp_v].saturating_add(pt[jp_v]) } else { 0 };
        let new_r_v = r_jp_v.max(r_mp_u);

        let r_jp_u = if jp_u != NONE_USIZE { heads[jp_u].saturating_add(pt[jp_u]) } else { 0 };
        let new_r_u = r_jp_u.max(new_r_v.saturating_add(pt[v]));

        let q_js_u = if js_u != NONE_USIZE { pt[js_u].saturating_add(tails[js_u]) } else { 0 };
        let q_ms_v = if ms_v != NONE_USIZE { pt[ms_v].saturating_add(tails[ms_v]) } else { 0 };
        let new_q_u = q_js_u.max(q_ms_v);

        let q_js_v = if js_v != NONE_USIZE { pt[js_v].saturating_add(tails[js_v]) } else { 0 };
        let new_q_v = q_js_v.max(pt[u].saturating_add(new_q_u));

        (new_r_v.saturating_add(pt[v]).saturating_add(new_q_v))
            .max(new_r_u.saturating_add(pt[u]).saturating_add(new_q_u))
    }

    /// O(1) estimate of makespan after reassigning `node` to `new_machine` at `insert_pos`.
    #[inline]
    fn estimate_reassign_fm(
        ds: &DisjSchedule,
        heads: &[u32], tails: &[u32],
        node: usize, new_machine: usize, new_pt: u32, insert_pos: usize,
        job_pred: &[usize], machine_pred: &[usize], machine_succ: &[usize],
    ) -> u32 {
        let jp = job_pred[node];
        let js = ds.job_succ[node];
        let old_mp = machine_pred[node];
        let old_ms = machine_succ[node];

        let jp_end = if jp != NONE_USIZE { heads[jp].saturating_add(ds.node_pt[jp]) } else { 0 };

        let new_seq = &ds.machine_seq[new_machine];
        let new_mp_end = if insert_pos > 0 && !new_seq.is_empty() {
            let pred = new_seq[(insert_pos - 1).min(new_seq.len() - 1)];
            heads[pred].saturating_add(ds.node_pt[pred])
        } else { 0 };

        let new_start = jp_end.max(new_mp_end);
        let new_end = new_start.saturating_add(new_pt);

        let js_tail = if js != NONE_USIZE { ds.node_pt[js].saturating_add(tails[js]) } else { 0 };
        let new_ms_tail = if insert_pos < new_seq.len() {
            let succ = new_seq[insert_pos];
            ds.node_pt[succ].saturating_add(tails[succ])
        } else { 0 };

        let node_path = new_end.saturating_add(js_tail.max(new_ms_tail));

        let old_reconnect = if old_mp != NONE_USIZE && old_ms != NONE_USIZE {
            let old_mp_end = heads[old_mp].saturating_add(ds.node_pt[old_mp]);
            old_mp_end.saturating_add(ds.node_pt[old_ms]).saturating_add(tails[old_ms])
        } else { 0 };

        node_path.max(old_reconnect)
    }

    /// Find candidate insertion positions for `node` on `new_machine`.
    fn find_insert_positions_fm(
        ds: &DisjSchedule,
        starts: &[u32],
        node: usize,
        new_machine: usize,
        job_pred: &[usize],
    ) -> Vec<usize> {
        let seq = &ds.machine_seq[new_machine];
        let len = seq.len();
        if len == 0 { return vec![0]; }

        let jp = job_pred[node];
        let job_pred_end = if jp != NONE_USIZE { starts[jp].saturating_add(ds.node_pt[jp]) } else { 0 };
        let cur_start = starts[node];

        let mut pos_after_jp = len;
        for (i, &nd) in seq.iter().enumerate() {
            if starts[nd] > job_pred_end { pos_after_jp = i; break; }
        }
        let pos_after_jp = pos_after_jp.min(len);

        let mut pos_by_cur = len;
        for (i, &nd) in seq.iter().enumerate() {
            if starts[nd] >= cur_start { pos_by_cur = i; break; }
        }
        let pos_by_cur = pos_by_cur.min(len);

        let mut out: Vec<usize> = Vec::with_capacity(6);
        let push = |v: &mut Vec<usize>, p: usize| {
            if p <= len && !v.contains(&p) { v.push(p); }
        };
        push(&mut out, pos_after_jp);
        push(&mut out, pos_after_jp.saturating_sub(1));
        push(&mut out, pos_by_cur);
        push(&mut out, pos_by_cur.saturating_sub(1));
        push(&mut out, 0);
        push(&mut out, len);

        if out.is_empty() { out.push(len); }
        if out.len() > 6 { out.truncate(6); }
        out
    }

    enum MoveTypeFm {
        Swap { machine: usize, pos: usize },
        Reassign { node: usize, new_machine: usize, new_pt: u32, insert_pos: usize },
    }

    /// Tabu search with integrated swap + reassign moves on disjunctive graph.
    /// Combined neighborhood is key for FJSP medium quality.
    pub fn tabu_search_fjsp_medium(
        pre: &Pre, challenge: &Challenge, base_sol: &Solution,
        max_iterations: usize, tenure_base: usize,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        global_best: &mut u32,
    ) -> Result<Option<(Solution, u32)>> {
        let Ok(mut ds) = build_disj_from_solution(pre, challenge, base_sol) else { return Ok(None) };
        let mut buf = EvalBuf::new(ds.n);
        let n = ds.n;
        let Some((initial_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };

        let mut best_global_mk = initial_mk;
        let mut best_global_ds = ds.clone();

        let tenure = tenure_base.max(5);
        let tenure_delta = (tenure / 3).max(2);
        let max_no_improve = (max_iterations / 2).max(60);
        let kick_threshold = (max_no_improve * 2 / 3).max(50);

        let mut tabu_swap: HashMap<(usize, usize), usize> = HashMap::with_capacity(tenure * 8);
        let mut tabu_reassign: HashMap<(usize, usize), usize> = HashMap::with_capacity(tenure * 4);

        // Precompute job_pred_node (constant - job chains don't change)
        let mut job_pred_node = vec![NONE_USIZE; n];
        for j in 0..ds.num_jobs {
            let base = ds.job_offsets[j];
            let end = ds.job_offsets[j + 1];
            for k in (base + 1)..end { job_pred_node[k] = k - 1; }
        }

        let mut no_improve = 0usize;
        let mut kicks_left = 3usize;
        let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15)
            ^ (initial_mk as u64).wrapping_shl(16) ^ (n as u64).wrapping_mul(0x517CC1B727220A95);

        let mut machine_pred_node = vec![NONE_USIZE; n];
        let mut crit = vec![false; n];

        for iter in 0..max_iterations {
            if no_improve >= max_no_improve {
                if kicks_left == 0 { break; }
                ds = best_global_ds.clone();
                no_improve = 0;
                kicks_left -= 1;
                tabu_swap.clear();
                tabu_reassign.clear();
                continue;
            }

            // Periodic kick: perturb on CP swaps
            if no_improve > 0 && no_improve % kick_threshold == 0 && kicks_left > 0 {
                let Some((_, kick_mk_node)) = eval_disj(&ds, &mut buf) else { break };
                crit.fill(false);
                let mut u = kick_mk_node;
                while u != NONE_USIZE { crit[u] = true; u = buf.best_pred[u]; }
                let mut kick_swaps: Vec<(usize, usize)> = Vec::new();
                for m in 0..ds.num_machines {
                    if ds.machine_seq[m].len() <= 1 { continue; }
                    for i in 0..(ds.machine_seq[m].len() - 1) {
                        if crit[ds.machine_seq[m][i]] && crit[ds.machine_seq[m][i + 1]] {
                            kick_swaps.push((m, i));
                        }
                    }
                }
                if !kick_swaps.is_empty() {
                    for _ in 0..2 {
                        pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                        let idx = (pseed as usize) % kick_swaps.len();
                        let (m, pos) = kick_swaps[idx];
                        if pos + 1 < ds.machine_seq[m].len() { ds.machine_seq[m].swap(pos, pos + 1); }
                    }
                }
                kicks_left -= 1;
                continue;
            }

            let Some((cur_mk, mk_node)) = eval_disj(&ds, &mut buf) else { break };
            if iter > 0 {
                if cur_mk < best_global_mk {
                    best_global_mk = cur_mk;
                    best_global_ds = ds.clone();
                    no_improve = 0;
                    if cur_mk < *global_best {
                        *global_best = cur_mk;
                        if let Ok(s) = disj_to_solution(pre, &ds, &buf.start) {
                            let _ = save_solution(&s);
                        }
                    }
                } else {
                    no_improve += 1;
                }
            }

            let tails = compute_tails_pulsar(&ds, &buf);

            machine_pred_node.fill(NONE_USIZE);
            for seq in &ds.machine_seq {
                for i in 1..seq.len() { machine_pred_node[seq[i]] = seq[i - 1]; }
            }

            crit.fill(false);
            let mut u = mk_node;
            while u != NONE_USIZE { crit[u] = true; u = buf.best_pred[u]; }

            let mut best_move: Option<MoveTypeFm> = None;
            let mut best_move_mk = u32::MAX;
            let mut fallback_move: Option<MoveTypeFm> = None;
            let mut fallback_mk = u32::MAX;

            // Swap moves on critical blocks (N1 neighborhood)
            for m in 0..ds.num_machines {
                if ds.machine_seq[m].len() <= 1 { continue; }
                let mut i = 0;
                while i < ds.machine_seq[m].len() {
                    if !crit[ds.machine_seq[m][i]] { i += 1; continue; }
                    let bstart = i;
                    let mut bend = i;
                    while bend + 1 < ds.machine_seq[m].len() {
                        let x = ds.machine_seq[m][bend];
                        let y = ds.machine_seq[m][bend + 1];
                        if !crit[y] { break; }
                        let end_x = buf.start[x].saturating_add(ds.node_pt[x]);
                        if buf.start[y] != end_x { break; }
                        bend += 1;
                    }
                    if bend > bstart {
                        let block_len = bend - bstart + 1;
                        let swap_positions = if block_len >= 3 { [bstart, bend - 1] } else { [bstart, NONE_USIZE] };
                        let num_swaps = if block_len >= 3 { 2 } else { 1 };
                        for si in 0..num_swaps {
                            let pos = swap_positions[si];
                            if pos == NONE_USIZE || pos + 1 >= ds.machine_seq[m].len() { continue; }
                            let node_u = ds.machine_seq[m][pos];
                            let node_v = ds.machine_seq[m][pos + 1];
                            let est_mk = estimate_swap_fm(
                                node_u, node_v, &buf.start, &tails, &ds.node_pt,
                                &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ,
                            );
                            let key = (node_u.min(node_v), node_u.max(node_v));
                            let is_tabu = tabu_swap.get(&key).map_or(false, |&exp| iter < exp);
                            let aspiration = est_mk < best_global_mk;
                            if (!is_tabu || aspiration) && est_mk < best_move_mk {
                                best_move_mk = est_mk;
                                best_move = Some(MoveTypeFm::Swap { machine: m, pos });
                            }
                            if est_mk < fallback_mk {
                                fallback_mk = est_mk;
                                fallback_move = Some(MoveTypeFm::Swap { machine: m, pos });
                            }
                        }
                    }
                    i = bend + 1;
                }
            }

            // Reassign moves on critical nodes (every 3rd iteration)
            if iter % 3 == 0 {
                for node in 0..n {
                    if !crit[node] { continue; }
                    let job = ds.node_job[node];
                    let op_idx = ds.node_op[node];
                    let product = pre.job_products[job];
                    if op_idx >= pre.product_ops[product].len() { continue; }
                    let op_info = &pre.product_ops[product][op_idx];
                    if op_info.machines.len() <= 1 { continue; }
                    let cur_machine = ds.node_machine[node];

                    for &(new_m, new_pt) in &op_info.machines {
                        if new_m == cur_machine { continue; }
                        let key = (node, new_m);
                        let is_tabu = tabu_reassign.get(&key).map_or(false, |&exp| iter < exp);
                        let positions = find_insert_positions_fm(&ds, &buf.start, node, new_m, &job_pred_node);
                        for insert_pos in positions {
                            let est_mk = estimate_reassign_fm(
                                &ds, &buf.start, &tails,
                                node, new_m, new_pt, insert_pos,
                                &job_pred_node, &machine_pred_node, &buf.machine_succ,
                            );
                            let aspiration = est_mk < best_global_mk;
                            if (!is_tabu || aspiration) && est_mk < best_move_mk {
                                best_move_mk = est_mk;
                                best_move = Some(MoveTypeFm::Reassign { node, new_machine: new_m, new_pt, insert_pos });
                            }
                            if est_mk < fallback_mk {
                                fallback_mk = est_mk;
                                fallback_move = Some(MoveTypeFm::Reassign { node, new_machine: new_m, new_pt, insert_pos });
                            }
                        }
                    }
                }
            }

            match best_move.or(fallback_move) {
                Some(MoveTypeFm::Swap { machine: m, pos }) => {
                    let node_a = ds.machine_seq[m][pos];
                    let node_b = ds.machine_seq[m][pos + 1];
                    ds.machine_seq[m].swap(pos, pos + 1);
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    let offset = (pseed % ((2 * tenure_delta + 1) as u64)) as usize;
                    let progress = (iter as f64) / (max_iterations as f64);
                    let late_bonus = if progress > 0.6 { ((progress - 0.6) * 10.0) as usize } else { 0 };
                    let this_tenure = (tenure + offset + late_bonus).saturating_sub(tenure_delta);
                    tabu_swap.insert((node_a.min(node_b), node_a.max(node_b)), iter + this_tenure);
                }
                Some(MoveTypeFm::Reassign { node, new_machine, new_pt, insert_pos }) => {
                    let old_machine = ds.node_machine[node];
                    if let Some(op) = ds.machine_seq[old_machine].iter().position(|&x| x == node) {
                        ds.machine_seq[old_machine].remove(op);
                    }
                    let ins = insert_pos.min(ds.machine_seq[new_machine].len());
                    ds.machine_seq[new_machine].insert(ins, node);
                    ds.node_machine[node] = new_machine;
                    ds.node_pt[node] = new_pt;
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    let offset = (pseed % ((2 * tenure_delta + 1) as u64)) as usize;
                    let this_tenure = (tenure + offset).saturating_sub(tenure_delta / 2);
                    tabu_reassign.insert((node, old_machine), iter + this_tenure);
                }
                None => break,
            }
        }

        // Final evaluation
        let Some((final_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
        if final_mk < best_global_mk { best_global_mk = final_mk; best_global_ds = ds; }
        if best_global_mk >= initial_mk { return Ok(None); }

        let Some((_, _)) = eval_disj(&best_global_ds, &mut buf) else { return Ok(None) };
        let sol = disj_to_solution(pre, &best_global_ds, &buf.start)?;
        Ok(Some((sol, best_global_mk)))
    }

    /// CP-targeted ruin: reassigns flexible CP nodes to non-current machines
    /// and swaps CP nodes in the temporal sequence, then decodes via forward scheduler.
    pub fn targeted_cp_kick(
        pre: &Pre,
        challenge: &Challenge,
        base_sol: &Solution,
        d_reassign: usize,
        d_swap: usize,
        rng: &mut SmallRng,
    ) -> Result<Solution> {
        use rand::seq::SliceRandom;
        let ds = build_disj_from_solution(pre, challenge, base_sol)?;
        let mut buf = EvalBuf::new(ds.n);
        let (mk, _mk_node) = match eval_disj(&ds, &mut buf) {
            Some(v) => v,
            None => return Ok(base_sol.clone()),
        };
        let tails = compute_tails_pulsar(&ds, &buf);

        // Build temporal sequence and machine assignments
        let mut m_assign: Vec<Vec<usize>> = vec![Vec::new(); challenge.num_jobs];
        let mut ops_by_start: Vec<(u32, usize, usize)> = Vec::with_capacity(ds.n);
        for j in 0..challenge.num_jobs {
            for (k, &(m, _)) in base_sol.job_schedule[j].iter().enumerate() {
                m_assign[j].push(m);
                let node = ds.job_offsets[j] + k;
                ops_by_start.push((buf.start[node], j, k));
            }
        }
        ops_by_start.sort_unstable_by_key(|x| x.0);
        let mut seq: Vec<usize> = ops_by_start.iter().map(|x| x.1).collect();

        // Identify flexible CP nodes
        let mut flex_cp: Vec<(usize, usize)> = Vec::new();
        let mut cp_indices: Vec<usize> = Vec::new();
        for (idx, &(st, j, k)) in ops_by_start.iter().enumerate() {
            let node = ds.job_offsets[j] + k;
            let pt = ds.node_pt[node];
            let tail = tails[node];
            if st + pt + tail == mk {
                cp_indices.push(idx);
                let prod = pre.job_products[j];
                if pre.product_ops[prod][k].machines.len() > 1 {
                    flex_cp.push((j, k));
                }
            }
        }

        // Reassign flexible CP nodes to alternative machines
        flex_cp.shuffle(rng);
        let num_reassign = d_reassign.min(flex_cp.len());
        for i in 0..num_reassign {
            let (j, k) = flex_cp[i];
            let prod = pre.job_products[j];
            let cur_m = m_assign[j][k];
            let alts: Vec<(usize, u32)> = pre.product_ops[prod][k].machines.iter()
                .filter(|&&(m, _)| m != cur_m).copied().collect();
            if !alts.is_empty() {
                let &(new_m, _) = alts.choose(rng).unwrap();
                m_assign[j][k] = new_m;
            }
        }

        // Swap CP positions in sequence
        cp_indices.shuffle(rng);
        let mut swaps_done = 0;
        for &idx in &cp_indices {
            if swaps_done >= d_swap { break; }
            if idx + 1 < seq.len() && seq[idx] != seq[idx + 1] {
                seq.swap(idx, idx + 1);
                swaps_done += 1;
            }
        }

        // Forward decode with forced machine assignments
        let mut next_op = vec![0usize; challenge.num_jobs];
        let mut mready = vec![0u32; challenge.num_machines];
        let mut jready = vec![0u32; challenge.num_jobs];
        let mut new_job_schedule: Vec<Vec<(usize, u32)>> = vec![Vec::new(); challenge.num_jobs];

        for &j in &seq {
            let k = next_op[j];
            if k >= pre.job_ops_len[j] { continue; }
            next_op[j] += 1;
            let m = m_assign[j][k];
            let prod = pre.job_products[j];
            let pt = pt_from_op(&pre.product_ops[prod][k], m).unwrap_or(1);
            let st = jready[j].max(mready[m]);
            new_job_schedule[j].push((m, st));
            let end = st + pt;
            jready[j] = end;
            mready[m] = end;
        }

        Ok(Solution { job_schedule: new_job_schedule })
    }

    pub fn solve(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
        pre: &Pre,
        greedy_sol: Solution,
        greedy_mk: u32,
        effort: &EffortConfig,
    ) -> Result<()> {
        let mut rng = SmallRng::from_seed(challenge.seed);
        let mut best_mk = greedy_mk;
        let mut top_solutions: Vec<(Solution, u32)> = Vec::with_capacity(25);
        push_top_solutions(&mut top_solutions, &greedy_sol, greedy_mk, 20);

        let allow_flex_balance = pre.high_flex > 0.60 && pre.jobshopness > 0.38;
        let mut rules: Vec<Rule> = vec![
            Rule::Adaptive, Rule::BnHeavy, Rule::EndTight, Rule::CriticalPath,
            Rule::MostWork, Rule::LeastFlex, Rule::Regret, Rule::ShortestProc,
        ];
        if allow_flex_balance { rules.push(Rule::FlexBalance); }

        let target_margin = ((pre.avg_op_min * (0.9 + 0.9 * pre.high_flex + 0.6 * pre.jobshopness)).max(1.0)) as u32;
        let route_w_base = if pre.chaotic_like {
            0.0
        } else {
            (0.050 + 0.12 * pre.high_flex + 0.10 * pre.jobshopness + 0.08 / pre.flex_avg.max(1.0)).clamp(0.04, 0.28)
        };

        // Phase 1: Initial construction with all rules
        let mut ranked: Vec<(Rule, u32, Solution)> = Vec::with_capacity(rules.len());
        for &rule in &rules {
            let (sol, mk) = construct_solution_conflict(challenge, pre, rule, 0, None, &mut rng, None, None, None, 0.0)?;
            if mk < best_mk { best_mk = mk; save_solution(&sol)?; }
            push_top_solutions(&mut top_solutions, &sol, mk, 20);
            ranked.push((rule, mk, sol));
        }
        ranked.sort_by_key(|x| x.1);

        let r0 = ranked[0].0;
        let r1 = ranked.get(1).map(|x| x.0).unwrap_or(r0);
        let r2 = ranked.get(2).map(|x| x.0).unwrap_or(r1);

        let mut rule_best = vec![u32::MAX; 10];
        let mut rule_tries = vec![0u32; 10];
        for (rr, mk, _) in &ranked {
            let idx = rule_idx(*rr);
            rule_best[idx] = rule_best[idx].min(*mk);
            rule_tries[idx] = rule_tries[idx].saturating_add(1);
        }

        // Learn from initial top solutions
        let base_sol = &top_solutions[0].0;
        let mut learned_jb = Some(job_bias_from_solution(pre, base_sol)?);
        let mut learned_mp = Some(machine_penalty_from_solution(pre, base_sol, challenge.num_machines)?);
        let mut learned_rp = if route_w_base > 0.0 {
            Some(route_pref_from_solution_lite(pre, base_sol, challenge)?)
        } else { None };
        let mut learn_updates_left = 12usize;

        // Phase 2: Main construction loop with bandit rule selection
        let num_restarts = effort.fjsp_medium_iters;
        let k_hi = if pre.flex_avg > 4.0 { 5 } else { 6 };
        let mut stuck = 0usize;

        for r in 0..num_restarts {
            let late = r >= (num_restarts * 2) / 3;
            let (k_min, k_max) = if stuck > 170 { (4, 6usize.min(k_hi)) }
                else if stuck > 90 { (3, 6usize.min(k_hi)) }
                else if stuck > 35 { (2, k_hi) }
                else { (2, k_hi.min(4)) };
            let rule = if r < 35 {
                let u: f64 = rng.gen();
                if allow_flex_balance && pre.high_flex > 0.82 && u < 0.10 { Rule::FlexBalance }
                else if u < 0.52 { r0 } else if u < 0.80 { r1 } else if u < 0.92 { r2 }
                else { rules[rng.gen_range(0..rules.len())] }
            } else {
                choose_rule_bandit(&mut rng, &rules, &rule_best, &rule_tries, best_mk, target_margin, stuck, pre.chaotic_like, late)
            };
            let k = if k_max <= k_min { k_min } else { rng.gen_range(k_min..=k_max) };
            let learn_base = if pre.chaotic_like { 0.0 } else {
                (0.08 + 0.22 * pre.jobshopness + 0.18 * pre.high_flex).clamp(0.05, 0.42)
            };
            let learn_boost = (1.0 + 0.35 * ((stuck as f64) / 120.0).clamp(0.0, 1.0)).clamp(1.0, 1.35);
            let learn_p = (learn_base * learn_boost).clamp(0.0, 0.60);
            let use_learn = learned_jb.is_some() && learned_mp.is_some()
                && rng.gen::<f64>() < learn_p
                && (route_w_base == 0.0 || learned_rp.is_some());
            let target = if best_mk < u32::MAX / 2 { Some(best_mk.saturating_add(target_margin)) } else { None };

            let (sol, mk) = if use_learn {
                construct_solution_conflict(challenge, pre, rule, k, target, &mut rng,
                    learned_jb.as_deref(), learned_mp.as_deref(), learned_rp.as_ref(), route_w_base)?
            } else {
                construct_solution_conflict(challenge, pre, rule, k, target, &mut rng, None, None, None, 0.0)?
            };

            let ridx = rule_idx(rule);
            rule_tries[ridx] = rule_tries[ridx].saturating_add(1);
            rule_best[ridx] = rule_best[ridx].min(mk);

            if mk < best_mk { best_mk = mk; save_solution(&sol)?; stuck = 0; } else { stuck = stuck.saturating_add(1); }
            push_top_solutions(&mut top_solutions, &sol, mk, 20);

            // Periodic elite learning update
            if learn_updates_left > 0 && !pre.chaotic_like && !top_solutions.is_empty() {
                let refresh = (r > 0 && r % 35 == 0) || stuck == 90 || stuck == 170;
                if refresh {
                    let rep_sol = &top_solutions[top_solutions.len().min(10) / 2].0;
                    learned_jb = Some(job_bias_from_solution(pre, rep_sol)?);
                    learned_mp = Some(machine_penalty_from_solution(pre, rep_sol, challenge.num_machines)?);
                    if route_w_base > 0.0 {
                        learned_rp = Some(route_pref_from_solution_lite(pre, rep_sol, challenge)?);
                    }
                    learn_updates_left -= 1;
                }
            }
        }

        // Phase 3: Per-elite refinement
        let route_w_ls = if route_w_base > 0.0 { (route_w_base * 1.40).clamp(route_w_base, 0.40) } else { 0.0 };
        let refine_n = top_solutions.len().min(10);
        let mut refine_buf: Vec<(Solution, u32)> = Vec::new();
        for i in 0..refine_n {
            let base = top_solutions[i].0.clone();
            let jb = job_bias_from_solution(pre, &base)?;
            let mp = machine_penalty_from_solution(pre, &base, challenge.num_machines)?;
            let rp = if route_w_ls > 0.0 { Some(route_pref_from_solution_lite(pre, &base, challenge)?) } else { None };
            let target_ls = if best_mk < u32::MAX / 2 { Some(best_mk.saturating_add(target_margin / 2)) } else { None };
            for attempt in 0..8 {
                let rule = match attempt {
                    0 => r0, 1 => Rule::Adaptive, 2 => Rule::BnHeavy, 3 => Rule::EndTight,
                    4 => Rule::Regret, 5 => Rule::LeastFlex, 6 => Rule::MostWork, _ => r1,
                };
                let k = match attempt % 4 { 0 => 2, 1 => 3, 2 => 4, _ => 2 }.min(k_hi);
                let (sol, mk) = construct_solution_conflict(challenge, pre, rule, k, target_ls, &mut rng,
                    Some(&jb), Some(&mp), rp.as_ref(), if rp.is_some() { route_w_ls } else { 0.0 })?;
                if mk < best_mk { best_mk = mk; save_solution(&sol)?; }
                refine_buf.push((sol, mk));
            }
        }
        for (sol, mk) in refine_buf { push_top_solutions(&mut top_solutions, &sol, mk, 20); }

        // Phase 4: Tabu search with combined swap + reassign on top solutions (reduced to 4)
        let ts_n = top_solutions.len().min(4);
        let ts_iters = 600usize;
        let ts_tenure = ((pre.total_ops as f64).sqrt() as usize).clamp(5, 12);
        for i in 0..ts_n {
            let base_sol = top_solutions[i].0.clone();
            if let Some((sol, mk)) = tabu_search_fjsp_medium(
                pre, challenge, &base_sol, ts_iters, ts_tenure, save_solution, &mut best_mk,
            )? {
                push_top_solutions(&mut top_solutions, &sol, mk, 20);
            }
        }

        // Phase 5: CBMLS on top solutions (reduced passes)
        let cb_passes = 3;
        let cb_iters = (pre.total_ops / 8).max(30).min(80);
        let cb_no_improve = cb_iters / 2;
        let cb_top_n = top_solutions.len().min(8);
        for ci in 0..cb_top_n {
            let base_sol = top_solutions[ci].0.clone();
            if let Ok(Some((cb_sol, cb_mk))) = critical_block_move_local_search_ex(pre, challenge, &base_sol, cb_passes, cb_iters, cb_no_improve) {
                if cb_mk < best_mk { best_mk = cb_mk; save_solution(&cb_sol)?; }
                push_top_solutions(&mut top_solutions, &cb_sol, cb_mk, 20);
            }
        }
        if let Ok(Some((bmr_sol, bmr_mk))) = bottleneck_machine_relief_pass(pre, challenge, &top_solutions[0].0, 20) {
            if bmr_mk < best_mk { best_mk = bmr_mk; save_solution(&bmr_sol)?; }
            push_top_solutions(&mut top_solutions, &bmr_sol, bmr_mk, 20);
        }

        // Phase 6: ILS loop skipped (i7: measuring ILS cost — go directly to fuel_loop)
        let ils_rounds = 0usize;
        let ils_max_no_improve = (ils_rounds * 3) / 4 + 3;
        let mut ils_best_mk = best_mk;
        let mut ils_no_improve = 0usize;
        let mut op_weights = vec![10.0f64; 5]; // strategy 4 = targeted CP ejection

        for ils_r in 0..ils_rounds {
            if ils_no_improve >= ils_max_no_improve { break; }
            let mut ds = build_disj_from_solution(pre, challenge, &top_solutions[0].0)?;
            let mut buf = EvalBuf::new(ds.n);
            let Some((_, mk_node)) = eval_disj(&ds, &mut buf) else { continue };
            let n = ds.n;
            let mut pseed: u64 = (ils_r as u64).wrapping_mul(0x517CC1B727220A95)
                .wrapping_add(ils_best_mk as u64)
                .wrapping_add(challenge.seed[0] as u64)
                .wrapping_add((ils_r as u64).wrapping_mul(0xDEADBEEF));
            let k_perturb = (3 + ils_r / 3).min(8);

            let total_weight: f64 = op_weights.iter().sum();
            let mut choice_val = rng.gen::<f64>() * total_weight;
            let mut strategy = 3usize;
            for (i, &weight) in op_weights.iter().enumerate() {
                if choice_val < weight { strategy = i; break; }
                choice_val -= weight;
            }

            if strategy == 0 {
                let mut crit_nodes: Vec<usize> = Vec::with_capacity(64);
                let mut u = mk_node;
                while u != NONE_USIZE { crit_nodes.push(u); u = buf.best_pred[u]; }
                let mut perturbed = 0; let mut attempts = 0;
                while perturbed < k_perturb && attempts < crit_nodes.len() * 4 {
                    attempts += 1;
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    if crit_nodes.is_empty() { break; }
                    let node = crit_nodes[(pseed as usize) % crit_nodes.len()];
                    let job = ds.node_job[node]; let op_idx = ds.node_op[node];
                    let product = pre.job_products[job];
                    let op_info = &pre.product_ops[product][op_idx];
                    if op_info.machines.len() <= 1 { continue; }
                    let cur_m = ds.node_machine[node];
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    let (new_m, new_pt) = op_info.machines[(pseed as usize) % op_info.machines.len()];
                    if new_m == cur_m { continue; }
                    let old_pos = match ds.machine_seq[cur_m].iter().position(|&x| x == node) { Some(p) => p, None => continue };
                    ds.machine_seq[cur_m].remove(old_pos);
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
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    let cur_seq_len = ds.machine_seq[worst_m].len();
                    if cur_seq_len == 0 { break; }
                    let node = ds.machine_seq[worst_m][(pseed as usize) % cur_seq_len];
                    let job = ds.node_job[node]; let op_idx = ds.node_op[node];
                    let product = pre.job_products[job];
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
                let k_re = k_perturb / 2;
                let mut perturbed = 0; let mut attempts = 0;
                while perturbed < k_re && attempts < crit_nodes.len() * 3 {
                    attempts += 1;
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    if crit_nodes.is_empty() { break; }
                    let node = crit_nodes[(pseed as usize) % crit_nodes.len()];
                    let job = ds.node_job[node]; let op_idx = ds.node_op[node];
                    let product = pre.job_products[job];
                    let op_info = &pre.product_ops[product][op_idx];
                    if op_info.machines.len() <= 1 { continue; }
                    let cur_m = ds.node_machine[node];
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    let (new_m, new_pt) = op_info.machines[(pseed as usize) % op_info.machines.len()];
                    if new_m == cur_m { continue; }
                    let old_pos = match ds.machine_seq[cur_m].iter().position(|&x| x == node) { Some(p) => p, None => continue };
                    ds.machine_seq[cur_m].remove(old_pos);
                    ds.node_machine[node] = new_m; ds.node_pt[node] = new_pt;
                    let cur_start = buf.start[node];
                    let mut ins_pos = ds.machine_seq[new_m].len();
                    for (ki, &nd) in ds.machine_seq[new_m].iter().enumerate() { if buf.start[nd] >= cur_start { ins_pos = ki; break; } }
                    ds.machine_seq[new_m].insert(ins_pos, node);
                    perturbed += 1;
                }
                let k_sw = k_perturb - k_re;
                let mut swapped = 0;
                for _ in 0..(k_sw * 4) {
                    if swapped >= k_sw || crit_machines.is_empty() { break; }
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    let m = crit_machines[(pseed as usize) % crit_machines.len()];
                    if ds.machine_seq[m].len() < 2 { continue; }
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    let pos = (pseed as usize) % (ds.machine_seq[m].len() - 1);
                    ds.machine_seq[m].swap(pos, pos + 1);
                    swapped += 1;
                }
            } else if strategy == 3 {
                let mut swapped = 0; let mut attempts = 0;
                while swapped < k_perturb && attempts < 100 {
                    attempts += 1;
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    let m = (pseed as usize) % ds.num_machines;
                    if ds.machine_seq[m].len() < 2 { continue; }
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    let pos = (pseed as usize) % (ds.machine_seq[m].len() - 1);
                    ds.machine_seq[m].swap(pos, pos + 1);
                    swapped += 1;
                }
            } else {
                // strategy 4: targeted CP ejection from bottleneck machine
                let mut crit_nodes: Vec<usize> = Vec::with_capacity(64);
                let mut u = mk_node;
                while u != NONE_USIZE { crit_nodes.push(u); u = buf.best_pred[u]; }
                if crit_nodes.is_empty() { ils_no_improve += 1; continue; }
                let mut machine_crit_count = vec![0usize; ds.num_machines];
                for &nd in &crit_nodes { machine_crit_count[ds.node_machine[nd]] += 1; }
                let target_m = machine_crit_count.iter().enumerate()
                    .max_by_key(|&(_, &c)| c).map(|(m, _)| m).unwrap_or(0);
                let candidates: Vec<usize> = crit_nodes.iter().cloned()
                    .filter(|&nd| {
                        if ds.node_machine[nd] != target_m { return false; }
                        let job = ds.node_job[nd]; let op_idx = ds.node_op[nd];
                        let product = pre.job_products[job];
                        pre.product_ops[product][op_idx].machines.len() > 1
                    })
                    .collect();
                if candidates.is_empty() { ils_no_improve += 1; continue; }
                let ejections = candidates.len().min(2);
                let mut perturbed = 0;
                for i in 0..ejections {
                    let node = candidates[i];
                    let job = ds.node_job[node]; let op_idx = ds.node_op[node];
                    let product = pre.job_products[job];
                    let op_info = &pre.product_ops[product][op_idx];
                    let cur_m = ds.node_machine[node];
                    let best_alt = op_info.machines.iter().filter(|&&(m,_)| m != cur_m)
                        .min_by_key(|&&(_,pt)| pt);
                    if let Some(&(new_m, new_pt)) = best_alt {
                        let old_pos = match ds.machine_seq[cur_m].iter().position(|&x| x == node) { Some(p) => p, None => continue };
                        ds.machine_seq[cur_m].remove(old_pos);
                        ds.node_machine[node] = new_m; ds.node_pt[node] = new_pt;
                        let cur_start = buf.start[node];
                        let mut ins_pos = ds.machine_seq[new_m].len();
                        for (ki, &nd) in ds.machine_seq[new_m].iter().enumerate() { if buf.start[nd] >= cur_start { ins_pos = ki; break; } }
                        ds.machine_seq[new_m].insert(ins_pos, node);
                        perturbed += 1;
                    }
                }
                if perturbed == 0 { ils_no_improve += 1; continue; }
            }

            let Some((_, _)) = eval_disj(&ds, &mut buf) else { ils_no_improve += 1; continue };
            let perturbed_sol = match disj_to_solution(pre, &ds, &buf.start) {
                Ok(s) => s, Err(_) => { ils_no_improve += 1; continue; }
            };
            let after_gr = match greedy_reassign_pass(pre, challenge, &perturbed_sol)? {
                Some((s, mk)) => (s, mk),
                None => {
                    if let Some((pmk, _)) = eval_disj(&ds, &mut buf) { (perturbed_sol.clone(), pmk) }
                    else { ils_no_improve += 1; continue; }
                }
            };
            let (candidate_sol, candidate_mk) = match critical_block_move_local_search_ex(pre, challenge, &after_gr.0, cb_passes, cb_iters, cb_no_improve) {
                Ok(Some((ls_sol, ls_mk))) => if ls_mk < after_gr.1 { (ls_sol, ls_mk) } else { after_gr },
                _ => after_gr,
            };

            if candidate_mk < ils_best_mk {
                let reward = 1.0 + (ils_best_mk - candidate_mk) as f64 / ils_best_mk as f64;
                op_weights[strategy] = (op_weights[strategy] * 0.8 + reward * 2.0).clamp(1.0, 50.0);
                ils_best_mk = candidate_mk;
                if candidate_mk < best_mk { best_mk = candidate_mk; save_solution(&candidate_sol)?; }
                push_top_solutions(&mut top_solutions, &candidate_sol, candidate_mk, 20);
                ils_no_improve = 0;
            } else {
                op_weights[strategy] = (op_weights[strategy] * 0.95).max(1.0);
                ils_no_improve += 1;
            }
        }

        // Final greedy reassign
        if let Ok(Some((s, m))) = greedy_reassign_pass(pre, challenge, &top_solutions[0].0) {
            if m < best_mk { best_mk = m; save_solution(&s)?; }
        }
        // Q-recovery: Schrage resequence on top elites before fuel loop
        for i in 0..top_solutions.len().min(5) {
            let base = top_solutions[i].0.clone();
            if let Ok(Some((s, m))) = schrage_pass(pre, challenge, &base, 6) {
                if m < best_mk { best_mk = m; save_solution(&s)?; }
                push_top_solutions(&mut top_solutions, &s, m, 20);
            }
        }

        // Fuel-based CP-Kick + Tabu loop (cycle-push: 65 outer to hit 407k ms target)
        // Phase4 cut to 4×600 + Phase5 cut to 3 passes saves ~38k ms; 65 outer×1350 = 88.7k steps
        {
            let ts_tenure = ((pre.total_ops as f64).sqrt() as usize).clamp(5, 12);
            let max_outer_iters = 150usize;
            let mut tb_no_improve = 0usize;
            for loop_iter in 0..max_outer_iters {
                if top_solutions.is_empty() { break; }

                // Pick base solution: cycle through pool
                let pool_size = top_solutions.len().min(12);
                let base_idx = loop_iter % pool_size;
                let base_sol = top_solutions[base_idx].0.clone();

                // Phase-based adaptive kick+ts_iters (L2c+L2b hybrid, threshold=100):
                // productive phase: fine kicks + full tabu budget; stagnated phase: heavy kick + reduced budget
                let (kick_reassigns, kick_swaps, ts_iters) = if tb_no_improve < 100 {
                    (rng.gen_range(2usize..=3), rng.gen_range(1usize..=2), (effort.fjsp_medium_iters / 2).max(60))
                } else {
                    (rng.gen_range(3usize..=5), rng.gen_range(2usize..=4), (effort.fjsp_medium_iters * 3 / 8).max(60))
                };
                let perturbed = match targeted_cp_kick(pre, challenge, &base_sol, kick_reassigns, kick_swaps, &mut rng) {
                    Ok(s) => s,
                    Err(_) => base_sol.clone(),
                };

                // Tabu search on perturbed solution
                let (cand_sol, cand_mk) = match tabu_search_fjsp_medium(
                    pre, challenge, &perturbed, ts_iters, ts_tenure, save_solution, &mut best_mk,
                )? {
                    Some((s, m)) => (s, m),
                    None => {
                        // Fallback: CBMLS on perturbed if tabu fails
                        match critical_block_move_local_search_ex(pre, challenge, &perturbed, cb_passes, cb_iters, cb_no_improve) {
                            Ok(Some((ls_sol, ls_mk))) => (ls_sol, ls_mk),
                            _ => (perturbed, u32::MAX),
                        }
                    }
                };

                if cand_mk < best_mk {
                    best_mk = cand_mk;
                    save_solution(&cand_sol)?;
                    tb_no_improve = 0;
                } else {
                    tb_no_improve += 1;
                }
                if cand_mk < u32::MAX {
                    push_top_solutions(&mut top_solutions, &cand_sol, cand_mk, 20);
                }
                if tb_no_improve > 40 { break; }
            }
        }

        // Final Q-recovery pass: Schrage + greedy_reassign on best solutions
        for i in 0..top_solutions.len().min(3) {
            let base = top_solutions[i].0.clone();
            if let Ok(Some((s, m))) = schrage_pass(pre, challenge, &base, 8) {
                if m < best_mk { best_mk = m; save_solution(&s)?; }
            }
        }
        if let Ok(Some((s, m))) = greedy_reassign_pass(pre, challenge, &top_solutions[0].0) {
            if m < best_mk { let _ = save_solution(&s); }
        }

        Ok(())
    }

}

pub mod fjsp_high {
    use anyhow::{anyhow, Result};
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    use tig_challenges::job_scheduling::*;

    use super::types::*;
    use super::infra::*;
    use super::fjsp_medium;

    const PERM_2: [[usize; 2]; 2] = [[0, 1], [1, 0]];
    const PERM_3: [[usize; 3]; 6] = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    const PERM_4: [[usize; 4]; 24] = [
        [0,1,2,3],[0,1,3,2],[0,2,1,3],[0,2,3,1],[0,3,1,2],[0,3,2,1],
        [1,0,2,3],[1,0,3,2],[1,2,0,3],[1,2,3,0],[1,3,0,2],[1,3,2,0],
        [2,0,1,3],[2,0,3,1],[2,1,0,3],[2,1,3,0],[2,3,0,1],[2,3,1,0],
        [3,0,1,2],[3,0,2,1],[3,1,0,2],[3,1,2,0],[3,2,0,1],[3,2,1,0],
    ];

    #[inline(always)]
    fn local_rule_idx(r: Rule) -> usize {
        match r {
            Rule::Adaptive=>0, Rule::BnHeavy=>1, Rule::EndTight=>2, Rule::CriticalPath=>3,
            Rule::MostWork=>4, Rule::LeastFlex=>5, Rule::Regret=>6, Rule::ShortestProc=>7, Rule::FlexBalance=>8,
        }
    }

    // ============================================================================
    // ELITE DIVERSITY POOL
    // ============================================================================

    fn solution_machine_signature(pre: &Pre, challenge: &Challenge, sol: &Solution) -> Vec<u16> {
        let mut sig: Vec<u16> = Vec::with_capacity(pre.total_ops);
        for job in 0..challenge.num_jobs {
            let lim = pre.job_ops_len[job].min(sol.job_schedule[job].len());
            for op_idx in 0..pre.job_ops_len[job] {
                sig.push(if op_idx < lim { sol.job_schedule[job][op_idx].0 as u16 } else { u16::MAX });
            }
        }
        sig
    }

    #[inline(always)]
    fn hamming_distance_sig(a: &[u16], b: &[u16]) -> u32 {
        let mut d = 0u32;
        for i in 0..a.len().min(b.len()) { if a[i] != b[i] { d += 1; } }
        d
    }

    fn push_top_solutions_diverse(
        pre: &Pre, challenge: &Challenge, pool: &mut Vec<(Solution, u32, Vec<u16>)>, sol: &Solution, mk: u32, cap: usize,
    ) {
        let sig = solution_machine_signature(pre, challenge, sol);
        pool.push((sol.clone(), mk, sig));
        while pool.len() > cap {
            let len = pool.len();
            let mut min_nn = vec![u32::MAX; len];
            for i in 0..len { for j in (i+1)..len {
                let d = hamming_distance_sig(&pool[i].2, &pool[j].2);
                if d < min_nn[i] { min_nn[i] = d; }
                if d < min_nn[j] { min_nn[j] = d; }
            }}
            let mut worst_mk = 0u32;
            for i in 0..len { if pool[i].1 > worst_mk { worst_mk = pool[i].1; } }
            let mut drop_idx = 0; let mut drop_min_nn = u32::MAX;
            for i in 0..len { if pool[i].1 == worst_mk && min_nn[i] < drop_min_nn { drop_idx = i; drop_min_nn = min_nn[i]; } }
            pool.swap_remove(drop_idx);
        }
        pool.sort_unstable_by_key(|x| x.1);
    }

    fn consensus_learning_from_elites(
        pre: &Pre, challenge: &Challenge, elites: &[(Solution, u32, Vec<u16>)],
    ) -> Result<(Vec<f64>, Vec<f64>, RoutePrefLite)> {
        if elites.is_empty() { return Err(anyhow!("empty")); }
        let denom = elites.len() as f64;
        let mut jb_sum = vec![0.0f64; challenge.num_jobs];
        for (sol, _, _) in elites { let jb = job_bias_from_solution(pre, sol)?; for j in 0..challenge.num_jobs { jb_sum[j] += jb[j]; } }
        for j in 0..challenge.num_jobs { jb_sum[j] /= denom; }

        let mut bn_cnt = vec![0u32; challenge.num_machines];
        let mut m_end = vec![0u32; challenge.num_machines];
        for (sol, mk, _) in elites {
            m_end.fill(0);
            for job in 0..challenge.num_jobs {
                let prod = pre.job_products[job]; if prod >= pre.product_ops.len() { continue; }
                let sched = &sol.job_schedule[job]; let lim = sched.len().min(pre.product_ops[prod].len());
                for op_idx in 0..lim {
                    let (m, st) = sched[op_idx]; if m >= challenge.num_machines { continue; }
                    let pt = pre.product_ops[prod][op_idx].machines.iter().find(|&&(mm,_)| mm==m).map(|&(_,p)|p).unwrap_or(0);
                    let end = st.saturating_add(pt); if end > m_end[m] { m_end[m] = end; }
                }
            }
            for m in 0..challenge.num_machines { if m_end[m] == *mk { bn_cnt[m] += 1; } }
        }
        let mut mp = vec![0.0f64; challenge.num_machines];
        for m in 0..challenge.num_machines { mp[m] = (bn_cnt[m] as f64) / denom; }

        // Learn route_pref from ALL elites (majority vote per product×op)
        let mut rp = route_pref_from_solution_lite(pre, &elites[0].0, challenge)?;
        let num_products = pre.product_ops.len();
        let mut counts: Vec<Vec<Vec<u32>>> = pre.product_ops.iter()
            .map(|ops| ops.iter().map(|_| vec![0u32; challenge.num_machines]).collect())
            .collect();
        for (sol, _, _) in elites {
            for job in 0..challenge.num_jobs {
                let product = pre.job_products[job];
                if product >= num_products { continue; }
                let sched = &sol.job_schedule[job];
                let ops_len = counts[product].len();
                for (op_idx, (m, _)) in sched.iter().enumerate() {
                    if op_idx >= ops_len { break; }
                    if *m < challenge.num_machines { counts[product][op_idx][*m] = counts[product][op_idx][*m].saturating_add(1); }
                }
            }
        }
        for product in 0..num_products {
            let ops_len = counts[product].len();
            if product >= rp.len() { break; }
            for op_idx in 0..ops_len {
                if op_idx >= rp[product].len() { break; }
                let row = &counts[product][op_idx];
                let total: u32 = row.iter().sum();
                if total == 0 { continue; }
                let mut best_m = 0usize; let mut best_c = 0u32;
                let mut second_m = 0usize; let mut second_c = 0u32;
                for (m, &c) in row.iter().enumerate() {
                    if c > best_c { second_m = best_m; second_c = best_c; best_m = m; best_c = c; }
                    else if m != best_m && c > second_c { second_m = m; second_c = c; }
                }
                if second_c == 0 { second_m = best_m; }
                let total64 = total as u64;
                rp[product][op_idx].best_m = best_m.min(255) as u8;
                rp[product][op_idx].best_w = ((best_c as u64) * 255 / total64) as u8;
                rp[product][op_idx].second_m = second_m.min(255) as u8;
                rp[product][op_idx].second_w = ((second_c as u64) * 255 / total64) as u8;
            }
        }
        Ok((jb_sum, mp, rp))
    }

    // PULSAR functions (schrage_pass, slack_ejection_schrage) are in infra.rs

    // ============================================================================
    // WINDOW EXHAUSTIVE — Local permutation search on bottleneck machine (kept as complement)
    // ============================================================================
    fn cp_window_exhaustive(
        pre: &Pre, challenge: &Challenge, base_sol: &Solution, max_iters: usize,
    ) -> Result<Option<(Solution, u32)>> {
        let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
        let mut buf = EvalBuf::new(ds.n);
        let Some((mut cur_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None); };
        let init_mk = cur_mk; let n = ds.n;

        for _ in 0..max_iters {
            let mut on_cp = vec![false; n];
            for nd in 0..n { if buf.start[nd].saturating_add(ds.node_pt[nd]) == cur_mk { on_cp[nd] = true; } }
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_unstable_by_key(|&a| std::cmp::Reverse(buf.start[a]));
            for &nd in &order {
                if !on_cp[nd] { continue; }
                let seq = &ds.machine_seq[ds.node_machine[nd]];
                if let Some(pos) = seq.iter().position(|&x| x == nd) {
                    if pos > 0 { let p = seq[pos-1]; if buf.start[p].saturating_add(ds.node_pt[p]) == buf.start[nd] { on_cp[p] = true; } }
                }
            }

            let mut mc = vec![0usize; ds.num_machines];
            for nd in 0..n { if on_cp[nd] { mc[ds.node_machine[nd]] += 1; } }
            let bn_m = mc.iter().enumerate().max_by_key(|&(_,&c)| c).map(|(m,_)| m).unwrap_or(0);
            let mut found = false;

            'ws: for &wsize in &[4usize, 3, 2] {
                let slen = ds.machine_seq[bn_m].len(); if slen < wsize { continue; }
                for start in 0..=(slen - wsize) {
                    if !ds.machine_seq[bn_m][start..start+wsize].iter().any(|&nd| on_cp[nd]) { continue; }
                    let mut orig = [0usize; 4];
                    for i in 0..wsize { orig[i] = ds.machine_seq[bn_m][start+i]; }
                    let mut best_pmk = cur_mk; let mut best_p: Option<&[usize]> = None;
                    let pc = if wsize==4 { 24 } else if wsize==3 { 6 } else { 2 };
                    for pi in 0..pc {
                        let p = if wsize==4 { &PERM_4[pi][..] } else if wsize==3 { &PERM_3[pi][..] } else { &PERM_2[pi][..] };
                        for i in 0..wsize { ds.machine_seq[bn_m][start+i] = orig[p[i]]; }
                        if let Some((tmk, _)) = eval_disj(&ds, &mut buf) { if tmk < best_pmk { best_pmk = tmk; best_p = Some(p); } }
                    }
                    if let Some(p) = best_p {
                        for i in 0..wsize { ds.machine_seq[bn_m][start+i] = orig[p[i]]; }
                        cur_mk = best_pmk; let _ = eval_disj(&ds, &mut buf); found = true; break 'ws;
                    } else { for i in 0..wsize { ds.machine_seq[bn_m][start+i] = orig[i]; } }
                }
            }
            if !found { break; }
        }
        if cur_mk < init_mk { Ok(Some((disj_to_solution(pre, &ds, &buf.start)?, cur_mk))) } else { Ok(None) }
    }

    // ============================================================================
    // ITERATIVE CP DESCENT — smart insertion + pair-wise CP moves
    // ============================================================================
    fn iterative_cp_descent(
        pre: &Pre, challenge: &Challenge, base_sol: &Solution, max_iters: usize,
    ) -> Result<Option<(Solution, u32)>> {
        let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
        let mut buf = EvalBuf::new(ds.n);
        let Some((mut current_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None); };
        let initial_mk = current_mk;
        let n = ds.n;

        let mut job_op_to_node: Vec<Vec<usize>> = vec![vec![]; challenge.num_jobs];
        for nd in 0..n {
            let job = ds.node_job[nd]; let op_idx = ds.node_op[nd];
            if op_idx >= job_op_to_node[job].len() { job_op_to_node[job].resize(op_idx + 1, usize::MAX); }
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
                let job = ds.node_job[nd]; let op_idx = ds.node_op[nd];
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
                    if pos > 0 { let pred = seq[pos - 1]; if finish[pred] == buf.start[nd] { on_cp[pred] = true; } }
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

            // Single-node CP moves with smart insertion
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

            // Pair-wise CP moves: try coordinated moves of consecutive CP node pairs
            if cp_flex.len() >= 2 {
                let pairs: Vec<(usize, usize)> = cp_flex.windows(2).map(|w| (w[0], w[1])).collect();

                let cand_positions = |ds: &DisjSchedule, buf: &EvalBuf, m: usize, nd: usize| -> Vec<usize> {
                    let cur_start = buf.start[nd];
                    let tlen = ds.machine_seq[m].len();
                    let mut spos = tlen;
                    for (ki, &nx) in ds.machine_seq[m].iter().enumerate() {
                        if buf.start[nx] >= cur_start { spos = ki; break; }
                    }
                    let mut positions: Vec<usize> = Vec::with_capacity(3);
                    for &p in &[spos, spos.saturating_sub(1), tlen] {
                        if p <= tlen && !positions.contains(&p) { positions.push(p); }
                    }
                    positions
                };

                for (nd1, nd2) in pairs {
                    let job1 = ds.node_job[nd1]; let op1 = ds.node_op[nd1];
                    let job2 = ds.node_job[nd2]; let op2 = ds.node_op[nd2];
                    let prod1 = pre.job_products[job1]; let prod2 = pre.job_products[job2];
                    if op1 >= pre.product_ops[prod1].len() || op2 >= pre.product_ops[prod2].len() { continue; }
                    let op_info1 = &pre.product_ops[prod1][op1];
                    let op_info2 = &pre.product_ops[prod2][op2];
                    if op_info1.machines.len() <= 1 && op_info2.machines.len() <= 1 { continue; }

                    let cur_m1 = ds.node_machine[nd1]; let cur_pt1 = ds.node_pt[nd1];
                    let cur_m2 = ds.node_machine[nd2]; let cur_pt2 = ds.node_pt[nd2];
                    let mut opts1: Vec<(usize, u32)> = Vec::new();
                    opts1.push((cur_m1, cur_pt1));
                    opts1.extend(op_info1.machines.iter().copied().filter(|&(m, _)| m != cur_m1).take(4));
                    let mut opts2: Vec<(usize, u32)> = Vec::new();
                    opts2.push((cur_m2, cur_pt2));
                    opts2.extend(op_info2.machines.iter().copied().filter(|&(m, _)| m != cur_m2).take(4));

                    let pos1 = match ds.machine_seq[cur_m1].iter().position(|&x| x == nd1) { Some(p)=>p, None=>continue };
                    let pos2 = match ds.machine_seq[cur_m2].iter().position(|&x| x == nd2) { Some(p)=>p, None=>continue };

                    // Remove both nodes
                    if cur_m1 == cur_m2 {
                        if pos1 > pos2 { ds.machine_seq[cur_m1].remove(pos1); ds.machine_seq[cur_m2].remove(pos2); }
                        else { ds.machine_seq[cur_m2].remove(pos2); ds.machine_seq[cur_m1].remove(pos1); }
                    } else {
                        ds.machine_seq[cur_m1].remove(pos1); ds.machine_seq[cur_m2].remove(pos2);
                    }

                    let mut best_mk_pair = current_mk;
                    let mut best_config: Option<(usize, u32, usize, usize, u32, usize, u8)> = None;

                    for &(m1, pt1) in &opts1 {
                        for &(m2, pt2) in &opts2 {
                            if m1 != m2 {
                                ds.node_machine[nd1] = m1; ds.node_pt[nd1] = pt1;
                                ds.node_machine[nd2] = m2; ds.node_pt[nd2] = pt2;
                                let p1s = cand_positions(&ds, &buf, m1, nd1);
                                let p2s = cand_positions(&ds, &buf, m2, nd2);
                                for &p1i in &p1s {
                                    ds.machine_seq[m1].insert(p1i, nd1);
                                    for &p2i in &p2s {
                                        ds.machine_seq[m2].insert(p2i, nd2);
                                        if let Some((tmk, _)) = eval_disj(&ds, &mut buf) {
                                            if tmk < best_mk_pair { best_mk_pair = tmk; best_config = Some((m1,pt1,p1i,m2,pt2,p2i,0)); }
                                        }
                                        ds.machine_seq[m2].remove(p2i);
                                    }
                                    ds.machine_seq[m1].remove(p1i);
                                }
                            } else {
                                let m = m1;
                                ds.node_machine[nd1] = m; ds.node_pt[nd1] = pt1;
                                ds.node_machine[nd2] = m; ds.node_pt[nd2] = pt2;
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
                                                    if order_flag == 0 { best_config = Some((m,pt1,pai,m,pt2,pbi,0)); }
                                                    else { best_config = Some((m,pt1,pbi,m,pt2,pai,1)); }
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
                            ds.node_machine[nd1] = bm1; ds.node_pt[nd1] = bpt1;
                            ds.node_machine[nd2] = bm2; ds.node_pt[nd2] = bpt2;
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

                    // Restore both nodes to original positions
                    ds.node_machine[nd1] = cur_m1; ds.node_pt[nd1] = cur_pt1;
                    ds.node_machine[nd2] = cur_m2; ds.node_pt[nd2] = cur_pt2;
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

    pub fn solve(
        challenge: &Challenge, save_solution: &dyn Fn(&Solution) -> Result<()>,
        pre: &Pre, greedy_sol: Solution, greedy_mk: u32, effort: &EffortConfig,
    ) -> Result<()> {
        let mut rng = SmallRng::from_seed(challenge.seed);
        let mut best_mk = greedy_mk; let mut best_sol = greedy_sol.clone();
        let mut pool: Vec<(Solution, u32, Vec<u16>)> = Vec::with_capacity(30);
        push_top_solutions_diverse(pre, challenge, &mut pool, &greedy_sol, greedy_mk, 25);

        let rules = [Rule::Adaptive, Rule::BnHeavy, Rule::EndTight, Rule::CriticalPath,
            Rule::MostWork, Rule::LeastFlex, Rule::Regret, Rule::ShortestProc, Rule::FlexBalance];
        let mut ranked: Vec<(Rule, u32, Solution)> = Vec::with_capacity(9);
        for &rule in &rules {
            let (sol, mk) = construct_solution_conflict(challenge, pre, rule, 0, None, &mut rng, None, None, None, 0.0)?;
            if mk < best_mk { best_mk = mk; best_sol = sol.clone(); save_solution(&sol)?; }
            push_top_solutions_diverse(pre, challenge, &mut pool, &sol, mk, 25);
            ranked.push((rule, mk, sol));
        }
        ranked.sort_by_key(|x| x.1);
        let r0 = ranked[0].0; let r1 = ranked.get(1).map(|x|x.0).unwrap_or(r0); let r2 = ranked.get(2).map(|x|x.0).unwrap_or(r1);
        let mut rb = vec![u32::MAX; 10]; let mut rt = vec![0u32; 10];
        for (rr, mk, _) in &ranked { let i = local_rule_idx(*rr); rb[i] = rb[i].min(*mk); rt[i] += 1; }

        let (jb0, mp0, rp0) = consensus_learning_from_elites(pre, challenge, &pool)?;
        let mut ljb = Some(jb0); let mut lmp = Some(mp0); let mut lrp = Some(rp0);
        let mut lu = 10usize;
        let tm: u32 = ((pre.avg_op_min * (0.9 + 0.9*pre.high_flex + 0.6*pre.jobshopness)).max(1.0)) as u32;
        let rw: f64 = (0.040 + 0.10*pre.high_flex + 0.08*pre.jobshopness).clamp(0.04, 0.22);
        let kh = if pre.flex_avg > 8.0 { 8 } else if pre.flex_avg > 6.5 { 7 } else { 6 };
        let mut stuck = 0usize;

        // PHASE 1: 2000 restarts. Phases 2-8 pool ops dominate runtime (522s at min(20));
        // cutting pool ops to min(5) frees ~420s, so Phase 1 can be small.
        let num_restarts = effort.fjsp_high_iters.max(2_000);
        for r in 0..num_restarts {
            let p = r as f64 / num_restarts as f64; let late = p >= 0.66;
            let (kn, kx) = if stuck>170 {(4,6usize.min(kh))} else if stuck>90 {(3,6usize.min(kh))} else if stuck>35 {(2,6usize.min(kh))} else {(2,4usize.min(kh))};
            let (kn, kx) = if p >= 0.80 { (1,1) } else if p >= 0.60 { let kx2=(kx/2).max(1); (kn.min(kx2), kx2) } else { (kn, kx) };
            let rule = if r < 35 { let u: f64 = rng.gen(); if u<0.12{Rule::FlexBalance} else if u<0.50{r0} else if u<0.75{r1} else if u<0.90{r2} else {rules[rng.gen_range(0..9)]} }
                else { choose_rule_bandit(&mut rng, &rules.to_vec(), &rb, &rt, best_mk, tm, stuck, false, late) };
            let k = if kx<=kn { kn } else { rng.gen_range(kn..=kx) };
            let lp = (0.08 + 0.22*pre.jobshopness + 0.18*pre.high_flex).clamp(0.05, 0.42) * (1.0 + 0.35*((stuck as f64)/120.0).clamp(0.0,1.0));
            let ul = ljb.is_some() && lmp.is_some() && rng.gen::<f64>() < lp && lrp.is_some();
            let tgt = if best_mk < u32::MAX/2 { Some(best_mk.saturating_add(tm)) } else { None };
            let (sol, mk) = if ul { construct_solution_conflict(challenge, pre, rule, k, tgt, &mut rng, ljb.as_deref(), lmp.as_deref(), lrp.as_ref(), rw)? }
                else { construct_solution_conflict(challenge, pre, rule, k, tgt, &mut rng, None, None, None, 0.0)? };
            let ri = local_rule_idx(rule); rt[ri] += 1; rb[ri] = rb[ri].min(mk);
            push_top_solutions_diverse(pre, challenge, &mut pool, &sol, mk, 25);
            if mk < best_mk {
                best_mk = mk; best_sol = sol.clone(); save_solution(&sol)?; stuck = 0;
                // INLINE improvement: Schrage resequencing + Window Exhaustive after each new best
                // (Schrage optimizes sequence within each CP machine, window exhaustive permutes windows)
                let mut cur_sol = sol.clone(); let mut cur_mk = mk;
                if let Ok(Some((s2, m2))) = schrage_pass(pre, challenge, &cur_sol, 2) {
                    if m2 < cur_mk { cur_mk = m2; cur_sol = s2; }
                }
                if let Ok(Some((s3, m3))) = cp_window_exhaustive(pre, challenge, &cur_sol, 2) {
                    if m3 < cur_mk { cur_mk = m3; cur_sol = s3; }
                }
                if cur_mk < best_mk { best_mk = cur_mk; best_sol = cur_sol.clone(); save_solution(&cur_sol)?; }
                push_top_solutions_diverse(pre, challenge, &mut pool, &cur_sol, cur_mk, 25);
                // Update consensus from improved pool (improves next construction bias)
                if lu > 0 { if let Ok((j,m,rp)) = consensus_learning_from_elites(pre, challenge, &pool) { ljb=Some(j); lmp=Some(m); lrp=Some(rp); lu-=1; } }
            } else { stuck += 1; }
        }

        // PHASE 2: Refinement 10 elites × 10
        let rwl = (rw*1.40).clamp(rw, 0.40);
        let er = pool.len().min(10);
        for i in 0..er {
            let bs = &pool[i].0; let jb = job_bias_from_solution(pre, bs)?; let mp = machine_penalty_from_solution(pre, bs, challenge.num_machines)?;
            let rp = Some(route_pref_from_solution_lite(pre, bs, challenge)?);
            let tl = if best_mk < u32::MAX/2 { Some(best_mk.saturating_add(tm/2)) } else { None };
            for a in 0..10 {
                let rule = match a { 0=>r0,1=>Rule::Adaptive,2=>Rule::BnHeavy,3=>Rule::EndTight,4=>Rule::Regret,5=>Rule::CriticalPath,6=>Rule::LeastFlex,7=>Rule::MostWork,8=>Rule::FlexBalance,_=>r1 };
                let k = match a%4 { 0=>2,1=>3,2=>4,_=>2 }.min(kh);
                let (sol, mk) = construct_solution_conflict(challenge, pre, rule, k, tl, &mut rng, Some(&jb), Some(&mp), rp.as_ref(), rwl)?;
                if mk < best_mk { best_mk = mk; best_sol = sol.clone(); save_solution(&sol)?; }
                push_top_solutions_diverse(pre, challenge, &mut pool, &sol, mk, 25);
            }
        }

        // PHASE 3: CBMLS-ex short bursts on 5 elites (was 20; cut to fit 279s budget)
        let lr = pool.len().min(5);
        for i in 0..lr {
            let bs = pool[i].0.clone();
            if let Some((s, m)) = critical_block_move_local_search_ex(pre, challenge, &bs, 8, 128, 24)? {
                if m < best_mk { best_mk = m; best_sol = s.clone(); save_solution(&s)?; }
                push_top_solutions_diverse(pre, challenge, &mut pool, &s, m, 25);
            }
        }
        if let Ok(Some((s, m))) = greedy_reassign_pass(pre, challenge, &best_sol) {
            if m < best_mk { best_mk = m; best_sol = s.clone(); save_solution(&s)?; }
        }

        // PHASE 4: PULSAR — Schrage resequencing on top 5 elites (was 20)
        let sr = pool.len().min(5);
        for i in 0..sr {
            let bs = pool[i].0.clone();
            if let Ok(Some((s, m))) = schrage_pass(pre, challenge, &bs, 10) {
                if m < best_mk { best_mk = m; best_sol = s.clone(); save_solution(&s)?; }
                push_top_solutions_diverse(pre, challenge, &mut pool, &s, m, 25);
            }
        }

        // PHASE 5: Iterative CP Descent on top 5 elites (was 20; iters 8→5)
        let ejr = pool.len().min(5);
        for i in 0..ejr {
            let bs = pool[i].0.clone();
            if let Ok(Some((s, m))) = iterative_cp_descent(pre, challenge, &bs, 5) {
                if m < best_mk { best_mk = m; best_sol = s.clone(); save_solution(&s)?; }
                push_top_solutions_diverse(pre, challenge, &mut pool, &s, m, 25);
                if let Ok(Some((s2, m2))) = greedy_reassign_pass(pre, challenge, &s) {
                    if m2 < best_mk { best_mk = m2; best_sol = s2.clone(); save_solution(&s2)?; }
                    push_top_solutions_diverse(pre, challenge, &mut pool, &s2, m2, 25);
                }
            }
        }

        // PHASE 6: Window Exhaustive on top 5 elites (was 15; iters 6→3)
        let wr = pool.len().min(5);
        for i in 0..wr {
            let bs = pool[i].0.clone();
            if let Ok(Some((s, m))) = cp_window_exhaustive(pre, challenge, &bs, 3) {
                if m < best_mk { best_mk = m; best_sol = s.clone(); save_solution(&s)?; }
                push_top_solutions_diverse(pre, challenge, &mut pool, &s, m, 25);
                if let Ok(Some((s2, m2))) = iterative_cp_descent(pre, challenge, &s, 5) {
                    if m2 < best_mk { best_mk = m2; best_sol = s2.clone(); save_solution(&s2)?; }
                    push_top_solutions_diverse(pre, challenge, &mut pool, &s2, m2, 25);
                }
            }
        }

        // PHASE 7: CBMLS polish
        if let Some((s, m)) = critical_block_move_local_search_ex(pre, challenge, &best_sol, 10, 80, 14)? {
            if m < best_mk { best_mk = m; best_sol = s.clone(); save_solution(&s)?; }
        }

        // PHASE 8: Greedy reassign polish
        for _ in 0..2 {
            if let Ok(Some((s, m))) = greedy_reassign_pass(pre, challenge, &best_sol) {
                if m < best_mk { best_mk = m; best_sol = s.clone(); save_solution(&s)?; }
            }
        }

        // PHASE 9: Fuel-based tabu intensification loop (NO wall-time).
        // T44 budget ~279s; Phase 1-8 consume ~60s. Remaining ~200s for tabu.
        // Each tabu call (5000 iters) ≈ ~250ms → 260 outer iters ≈ 65s.
        let tabu_tenure = ((pre.total_ops as f64).sqrt() as usize).clamp(5, 14);
        let tabu_iters_per_seed = 5_000usize;
        let tabu_outer_max = 60usize;
        let mut tabu_no_improve = 0usize;
        for outer in 0..tabu_outer_max {
            if pool.is_empty() { break; }
            let pool_size = pool.len().min(20);
            let base_idx = outer % pool_size;
            let base_candidate = pool[base_idx].0.clone();

            // CP-targeted kick every 4th outer iter to escape basins
            let perturbed = if outer % 4 == 0 {
                match fjsp_medium::targeted_cp_kick(pre, challenge, &base_candidate, 2, 3, &mut rng) {
                    Ok(s) => s,
                    Err(_) => base_candidate.clone(),
                }
            } else {
                base_candidate.clone()
            };

            let (cand_sol, cand_mk) = match fjsp_medium::tabu_search_fjsp_medium(
                pre, challenge, &perturbed, tabu_iters_per_seed, tabu_tenure, save_solution, &mut best_mk
            )? {
                Some((s, m)) => (s, m),
                None => {
                    tabu_no_improve += 1;
                    if tabu_no_improve > 100 { break; }
                    continue;
                }
            };

            if cand_mk < best_mk {
                best_mk = cand_mk; best_sol = cand_sol.clone(); save_solution(&cand_sol)?;
                tabu_no_improve = 0;
            } else {
                tabu_no_improve += 1;
            }
            push_top_solutions_diverse(pre, challenge, &mut pool, &cand_sol, cand_mk, 25);
            if tabu_no_improve > 100 { break; }
        }

        // PHASE 10: Final CP descent polish on global best
        if let Ok(Some((s, m))) = iterative_cp_descent(pre, challenge, &best_sol, 30) {
            if m < best_mk { best_mk = m; best_sol = s.clone(); save_solution(&s)?; }
        }
        // Final greedy reassign
        if let Ok(Some((s, m))) = greedy_reassign_pass(pre, challenge, &best_sol) {
            if m < best_mk { let _ = save_solution(&s); }
        }

        Ok(())
    }

}

pub mod our_search {
    use anyhow::Result;
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    use tig_challenges::job_scheduling::*;

    const NONE: usize = usize::MAX;
    const INF32: u32 = u32::MAX / 4;

    struct PD {
        nj: usize,
        nm: usize,
        n: usize,
        job_product: Vec<usize>,
        job_ops_len: Vec<usize>,
        job_offset: Vec<usize>,
        op_elig: Vec<Vec<(usize, u32)>>,
        op_min_pt: Vec<u32>,
        op_flex: Vec<usize>,
        flex_avg: f64,
        machine_load0: Vec<f64>,
        op_pt_on: Vec<Vec<u32>>,
        op_avg_pt: Vec<f64>,
    }

    impl PD {
        fn new(c: &Challenge) -> Self {
            let nj = c.num_jobs;
            let nm = c.num_machines;

            let mut job_product = Vec::with_capacity(nj);
            for (p, &cnt) in c.jobs_per_product.iter().enumerate() {
                for _ in 0..cnt { job_product.push(p); }
            }

            let mut job_ops_len = Vec::with_capacity(nj);
            let mut job_offset = Vec::with_capacity(nj);
            let mut n = 0usize;
            for j in 0..nj {
                job_offset.push(n);
                let ops = c.product_processing_times[job_product[j]].len();
                job_ops_len.push(ops);
                n += ops;
            }

            let mut op_elig = Vec::with_capacity(n);
            let mut op_min_pt = Vec::with_capacity(n);
            let mut op_flex = Vec::with_capacity(n);
            let mut total_flex = 0usize;
            let mut machine_load0 = vec![0.0f64; nm];

            for j in 0..nj {
                let p = job_product[j];
                for op in &c.product_processing_times[p] {
                    let mut el: Vec<(usize, u32)> = op.iter().map(|(&m, &t)| (m, t)).collect();
                    el.sort_unstable_by_key(|&(_, t)| t);
                    let min_pt = el.first().map(|&(_, t)| t).unwrap_or(INF32);
                    let flex = el.len();
                    total_flex += flex;
                    if flex > 0 && min_pt < INF32 {
                        let delta = min_pt as f64 / flex as f64;
                        for &(m, _) in &el { machine_load0[m] += delta; }
                    }
                    op_min_pt.push(min_pt);
                    op_flex.push(flex);
                    op_elig.push(el);
                }
            }

            let mut op_pt_on = vec![vec![0u32; nm]; n];
            let mut op_avg_pt = vec![0.0f64; n];
            for o in 0..n {
                for &(m, pt) in &op_elig[o] { op_pt_on[o][m] = pt; }
                if !op_elig[o].is_empty() {
                    let sum: u32 = op_elig[o].iter().map(|&(_, t)| t).sum();
                    op_avg_pt[o] = sum as f64 / op_elig[o].len() as f64;
                }
            }

            let flex_avg = if n > 0 { total_flex as f64 / n as f64 } else { 1.0 };
            PD { nj, nm, n, job_product, job_ops_len, job_offset, op_elig, op_min_pt, op_flex, flex_avg, machine_load0, op_pt_on, op_avg_pt }
        }

        #[inline] fn op_id(&self, j: usize, k: usize) -> usize { self.job_offset[j] + k }
    }

    #[derive(Clone)]
    struct DG {
        n: usize,
        nm: usize,
        nj: usize,
        node_pt: Vec<u32>,
        node_machine: Vec<usize>,
        node_job: Vec<usize>,
        node_op: Vec<usize>,
        job_offset: Vec<usize>,
        job_ops_len: Vec<usize>,
        machine_seq: Vec<Vec<usize>>,
    }

    struct Buf {
        indeg: Vec<u16>,
        start: Vec<u32>,
        pred: Vec<usize>,
        msucc: Vec<usize>,
        stack: Vec<usize>,
        tail: Vec<u32>,
        crit: Vec<bool>,
        mpred: Vec<usize>,
        jpred: Vec<usize>,
        saved_crit: Vec<bool>,
        saved_start: Vec<u32>,
    }

    impl Buf {
        fn new(n: usize) -> Self {
            Buf {
                indeg: vec![0; n], start: vec![0; n], pred: vec![NONE; n],
                msucc: vec![NONE; n], stack: Vec::with_capacity(n),
                tail: vec![0; n], crit: vec![false; n],
                mpred: vec![NONE; n], jpred: vec![NONE; n],
                saved_crit: vec![false; n], saved_start: vec![0; n],
            }
        }
    }

    impl DG {
        fn from_solution(pd: &PD, c: &Challenge, sol: &Solution) -> Result<Self> {
            let n = pd.n;
            let nm = pd.nm;
            let nj = pd.nj;
            let mut node_pt = vec![0u32; n];
            let mut node_machine = vec![0usize; n];
            let mut node_job = vec![0usize; n];
            let mut node_op = vec![0usize; n];
            let mut machine_ops: Vec<Vec<(u32, usize)>> = vec![vec![]; nm];

            for j in 0..nj {
                let p = pd.job_product[j];
                for k in 0..pd.job_ops_len[j] {
                    let o = pd.op_id(j, k);
                    let (m, st) = sol.job_schedule[j][k];
                    let pt = c.product_processing_times[p][k].get(&m).copied().unwrap_or(1);
                    node_pt[o] = pt; node_machine[o] = m; node_job[o] = j; node_op[o] = k;
                    machine_ops[m].push((st, o));
                }
            }
            let mut machine_seq = vec![vec![]; nm];
            for m in 0..nm {
                machine_ops[m].sort_unstable_by_key(|&(st, _)| st);
                machine_seq[m] = machine_ops[m].iter().map(|&(_, o)| o).collect();
            }
            Ok(DG { n, nm, nj, node_pt, node_machine, node_job, node_op,
                     job_offset: pd.job_offset.clone(), job_ops_len: pd.job_ops_len.clone(), machine_seq })
        }

        fn eval(&self, buf: &mut Buf) -> Option<(u32, usize)> {
            let n = self.n;
            for i in 0..n { buf.indeg[i] = 0; buf.start[i] = 0; buf.pred[i] = NONE; buf.msucc[i] = NONE; }

            for j in 0..self.nj {
                let base = self.job_offset[j];
                for k in 1..self.job_ops_len[j] { buf.indeg[base + k] += 1; }
            }
            for m in 0..self.nm {
                let seq = &self.machine_seq[m];
                for i in 1..seq.len() { buf.indeg[seq[i]] += 1; }
                for i in 0..seq.len().saturating_sub(1) { buf.msucc[seq[i]] = seq[i + 1]; }
            }

            buf.stack.clear();
            for o in 0..n { if buf.indeg[o] == 0 { buf.stack.push(o); } }

            let mut qi = 0;
            while qi < buf.stack.len() {
                let o = buf.stack[qi]; qi += 1;
                let end = buf.start[o] + self.node_pt[o];
                let j = self.node_job[o];
                let k = self.node_op[o];
                if k + 1 < self.job_ops_len[j] {
                    let ns = self.job_offset[j] + k + 1;
                    if end > buf.start[ns] { buf.start[ns] = end; buf.pred[ns] = o; }
                    buf.indeg[ns] -= 1;
                    if buf.indeg[ns] == 0 { buf.stack.push(ns); }
                }
                let ms = buf.msucc[o];
                if ms != NONE {
                    if end > buf.start[ms] { buf.start[ms] = end; buf.pred[ms] = o; }
                    buf.indeg[ms] -= 1;
                    if buf.indeg[ms] == 0 { buf.stack.push(ms); }
                }
            }
            if qi != n { return None; }

            let mut mk = 0u32;
            let mut mk_node = 0;
            for o in 0..n {
                let end = buf.start[o] + self.node_pt[o];
                if end > mk { mk = end; mk_node = o; }
            }
            Some((mk, mk_node))
        }

        fn compute_tails_and_preds(&self, buf: &mut Buf) {
            let n = self.n;
            for i in 0..n { buf.tail[i] = 0; buf.mpred[i] = NONE; buf.jpred[i] = NONE; }

            for seq in &self.machine_seq {
                for i in 1..seq.len() { buf.mpred[seq[i]] = seq[i - 1]; }
            }
            for j in 0..self.nj {
                let base = self.job_offset[j];
                for k in 1..self.job_ops_len[j] { buf.jpred[base + k] = base + k - 1; }
            }

            for &o in buf.stack.iter().rev() {
                let contrib = self.node_pt[o] + buf.tail[o];
                let j = self.node_job[o];
                let k = self.node_op[o];
                if k > 0 {
                    let jp = self.job_offset[j] + k - 1;
                    if contrib > buf.tail[jp] { buf.tail[jp] = contrib; }
                }
                if buf.mpred[o] != NONE {
                    let mp = buf.mpred[o];
                    if contrib > buf.tail[mp] { buf.tail[mp] = contrib; }
                }
            }
        }

        fn mark_critical(&self, buf: &mut Buf, mk: u32, mk_node: usize) {
            for o in 0..self.n {
                buf.crit[o] = buf.start[o] + self.node_pt[o] + buf.tail[o] == mk;
            }
            let mut u = mk_node;
            while u != NONE { buf.crit[u] = true; u = buf.pred[u]; }
        }

        fn to_solution(&self, pd: &PD, buf: &Buf) -> Solution {
            let mut js = Vec::with_capacity(pd.nj);
            for j in 0..pd.nj {
                let mut ops = Vec::with_capacity(pd.job_ops_len[j]);
                for k in 0..pd.job_ops_len[j] {
                    let o = pd.op_id(j, k);
                    ops.push((self.node_machine[o], buf.start[o]));
                }
                js.push(ops);
            }
            Solution { job_schedule: js }
        }
    }

    #[inline]
    fn estimate_insert_mk(
        op: usize, old_m: usize, old_idx: usize,
        target_m: usize, ins_pos: usize,
        new_pt: u32, buf: &Buf, dg: &DG,
    ) -> u32 {
        let j = dg.node_job[op];
        let k = dg.node_op[op];
        let jp = if k > 0 { dg.job_offset[j] + k - 1 } else { NONE };
        let js = if k + 1 < dg.job_ops_len[j] { dg.job_offset[j] + k + 1 } else { NONE };
        let r_from_job = if jp != NONE { buf.start[jp] + dg.node_pt[jp] } else { 0 };
        let q_from_job = if js != NONE { dg.node_pt[js] + buf.tail[js] } else { 0 };

        let (mp, ms, old_mp, old_ms);

        if old_m == target_m {
            let seq = &dg.machine_seq[old_m];
            let slen = seq.len();
            let cleaned_len = slen - 1;
            mp = if ins_pos == 0 { NONE } else {
                let ci = ins_pos - 1;
                if ci < old_idx { seq[ci] } else { seq[ci + 1] }
            };
            ms = if ins_pos >= cleaned_len { NONE } else {
                let ci = ins_pos;
                if ci < old_idx { seq[ci] } else { seq[ci + 1] }
            };
            old_mp = if old_idx > 0 { seq[old_idx - 1] } else { NONE };
            old_ms = if old_idx + 1 < slen { seq[old_idx + 1] } else { NONE };
        } else {
            let tseq = &dg.machine_seq[target_m];
            mp = if ins_pos > 0 { tseq[ins_pos - 1] } else { NONE };
            ms = if ins_pos < tseq.len() { tseq[ins_pos] } else { NONE };
            let oseq = &dg.machine_seq[old_m];
            old_mp = if old_idx > 0 { oseq[old_idx - 1] } else { NONE };
            old_ms = if old_idx + 1 < oseq.len() { oseq[old_idx + 1] } else { NONE };
        }

        let r_mach = if mp != NONE { buf.start[mp] + dg.node_pt[mp] } else { 0 };
        let new_r_op = r_from_job.max(r_mach);
        let q_mach = if ms != NONE { dg.node_pt[ms] + buf.tail[ms] } else { 0 };
        let new_q_op = q_from_job.max(q_mach);
        let path_op = new_r_op + new_pt + new_q_op;

        let path_gap = if old_ms != NONE {
            let jj = dg.node_job[old_ms];
            let kk = dg.node_op[old_ms];
            let jpp = if kk > 0 { dg.job_offset[jj] + kk - 1 } else { NONE };
            let r_j = if jpp != NONE { buf.start[jpp] + dg.node_pt[jpp] } else { 0 };
            let r_m = if old_mp != NONE { buf.start[old_mp] + dg.node_pt[old_mp] } else { 0 };
            r_j.max(r_m) + dg.node_pt[old_ms] + buf.tail[old_ms]
        } else { 0 };

        let path_ms = if ms != NONE {
            let jj = dg.node_job[ms];
            let kk = dg.node_op[ms];
            let jpp = if kk > 0 { dg.job_offset[jj] + kk - 1 } else { NONE };
            let r_j = if jpp != NONE { buf.start[jpp] + dg.node_pt[jpp] } else { 0 };
            r_j.max(new_r_op + new_pt) + dg.node_pt[ms] + buf.tail[ms]
        } else { 0 };

        path_op.max(path_gap).max(path_ms)
    }

    struct CB { m: usize, s: usize, e: usize }

    fn find_blocks(dg: &DG, buf: &Buf) -> Vec<CB> {
        let mut blocks = Vec::new();
        for m in 0..dg.nm {
            let seq = &dg.machine_seq[m];
            let mut i = 0;
            while i < seq.len() {
                if !buf.crit[seq[i]] { i += 1; continue; }
                let bs = i;
                let mut be = i;
                while be + 1 < seq.len() {
                    let x = seq[be]; let y = seq[be + 1];
                    if !buf.crit[y] { break; }
                    if buf.start[y] != buf.start[x] + dg.node_pt[x] { break; }
                    be += 1;
                }
                if be > bs { blocks.push(CB { m, s: bs, e: be }); }
                i = be + 1;
            }
        }
        blocks
    }

    #[inline]
    fn estimate_swap_mk(m: usize, pos: usize, buf: &Buf, dg: &DG) -> u32 {
        let seq = &dg.machine_seq[m];
        let o1 = seq[pos];
        let o2 = seq[pos + 1];

        let mp = if pos > 0 { seq[pos - 1] } else { NONE };
        let ms = if pos + 2 < seq.len() { seq[pos + 2] } else { NONE };

        let end_mp = if mp != NONE { buf.start[mp] + dg.node_pt[mp] } else { 0 };

        let j2 = dg.node_job[o2]; let k2 = dg.node_op[o2];
        let jp2 = if k2 > 0 { dg.job_offset[j2] + k2 - 1 } else { NONE };
        let end_jp2 = if jp2 != NONE { buf.start[jp2] + dg.node_pt[jp2] } else { 0 };
        let r_o2 = end_mp.max(end_jp2);

        let j1 = dg.node_job[o1]; let k1 = dg.node_op[o1];
        let jp1 = if k1 > 0 { dg.job_offset[j1] + k1 - 1 } else { NONE };
        let end_jp1 = if jp1 != NONE { buf.start[jp1] + dg.node_pt[jp1] } else { 0 };
        let r_o1 = (r_o2 + dg.node_pt[o2]).max(end_jp1);

        let js1 = if k1 + 1 < dg.job_ops_len[j1] { dg.job_offset[j1] + k1 + 1 } else { NONE };
        let q_j1 = if js1 != NONE { dg.node_pt[js1] + buf.tail[js1] } else { 0 };
        let q_ms = if ms != NONE { dg.node_pt[ms] + buf.tail[ms] } else { 0 };
        let q_o1 = q_j1.max(q_ms);

        let js2 = if k2 + 1 < dg.job_ops_len[j2] { dg.job_offset[j2] + k2 + 1 } else { NONE };
        let q_j2 = if js2 != NONE { dg.node_pt[js2] + buf.tail[js2] } else { 0 };
        let q_o2 = q_j2.max(dg.node_pt[o1] + q_o1);

        let path1 = r_o2 + dg.node_pt[o2] + q_o2;
        let path2 = r_o1 + dg.node_pt[o1] + q_o1;

        let path3 = if ms != NONE {
            let jms = dg.node_job[ms]; let kms = dg.node_op[ms];
            let jpms = if kms > 0 { dg.job_offset[jms] + kms - 1 } else { NONE };
            let end_jpms = if jpms != NONE { buf.start[jpms] + dg.node_pt[jpms] } else { 0 };
            end_jpms.max(r_o1 + dg.node_pt[o1]) + dg.node_pt[ms] + buf.tail[ms]
        } else { 0 };

        path1.max(path2).max(path3)
    }

    struct Cand {
        kind: u8,
        op: usize,
        m: usize,
        old_idx: usize,
        ins_pos: usize,
        new_m: usize,
        new_pt: u32,
        est: u32,
    }

    fn hybrid_search(
        dg: &mut DG, pd: &PD, buf: &mut Buf,
        max_iter: usize, high_flex: bool, pseed: &mut u64,
        save: &dyn Fn(&Solution) -> Result<()>,
        global_best: &mut u32,
        perturb_str: usize, noi_limit: usize,
        verify_k: usize,
    ) -> Result<(u32, DG)> {
        let Some((mk, mk_node)) = dg.eval(buf) else { return Ok((u32::MAX, dg.clone())); };
        dg.compute_tails_and_preds(buf);
        dg.mark_critical(buf, mk, mk_node);

        let mut best_mk = mk;

        let mut best_dg = dg.clone();
        let mut tabu = vec![0usize; pd.n];
        let tenure = if high_flex { 4 + (pd.n as f64).sqrt() as usize / 2 }
                     else { 7 + (pd.n as f64).sqrt() as usize };
        let mut no_improve = 0usize;
        let mut cands: Vec<Cand> = Vec::with_capacity(256);

        for iter in 0..max_iter {
            buf.saved_crit.copy_from_slice(&buf.crit);
            buf.saved_start.copy_from_slice(&buf.start);

            let blocks = find_blocks(dg, buf);
            cands.clear();

            for block in &blocks {
                let m = block.m;
                let seq = &dg.machine_seq[m];
                let bs = block.s;
                let be = block.e;
                if be <= bs { continue; }

                let est = estimate_swap_mk(m, bs, buf, dg);
                cands.push(Cand { kind: 0, op: 0, m, old_idx: bs, ins_pos: bs + 1, new_m: m, new_pt: 0, est });

                if be - bs > 1 {
                    let est = estimate_swap_mk(m, be - 1, buf, dg);
                    cands.push(Cand { kind: 0, op: 0, m, old_idx: be - 1, ins_pos: be, new_m: m, new_pt: 0, est });
                }

                let first_op = seq[bs];
                let last_op = seq[be];

                for target in (bs + 1)..=be {
                    let est = estimate_insert_mk(first_op, m, bs, m, target, dg.node_pt[first_op], buf, dg);
                    cands.push(Cand { kind: 1, op: first_op, m, old_idx: bs, ins_pos: target, new_m: m, new_pt: dg.node_pt[first_op], est });
                }

                for target in bs..be {
                    let est = estimate_insert_mk(last_op, m, be, m, target, dg.node_pt[last_op], buf, dg);
                    cands.push(Cand { kind: 1, op: last_op, m, old_idx: be, ins_pos: target, new_m: m, new_pt: dg.node_pt[last_op], est });
                }

                for pos in (bs + 1)..be {
                    let op = seq[pos];
                    let pt = dg.node_pt[op];
                    let est1 = estimate_insert_mk(op, m, pos, m, bs, pt, buf, dg);
                    cands.push(Cand { kind: 1, op, m, old_idx: pos, ins_pos: bs, new_m: m, new_pt: pt, est: est1 });
                    let est2 = estimate_insert_mk(op, m, pos, m, be, pt, buf, dg);
                    cands.push(Cand { kind: 1, op, m, old_idx: pos, ins_pos: be, new_m: m, new_pt: pt, est: est2 });
                }
            }

            if high_flex {
                for o in 0..pd.n {
                    if !buf.saved_crit[o] || pd.op_flex[o] < 2 { continue; }
                    let old_m = dg.node_machine[o];
                    let old_idx = match dg.machine_seq[old_m].iter().position(|&x| x == o) {
                        Some(i) => i, None => continue,
                    };

                    for &(new_m, new_pt) in &pd.op_elig[o] {
                        if new_m == old_m { continue; }
                        let tseq = &dg.machine_seq[new_m];
                        for ip in 0..=tseq.len() {
                            let est = estimate_insert_mk(o, old_m, old_idx, new_m, ip, new_pt, buf, dg);
                            cands.push(Cand { kind: 2, op: o, m: old_m, old_idx, ins_pos: ip, new_m, new_pt, est });
                        }
                    }
                }
            }

            if cands.is_empty() {
                no_improve += 1;
                if no_improve > noi_limit {
                    dg.clone_from(&best_dg);
                    perturb(dg, pd, buf, perturb_str, pseed);
                    if let Some((mk, mkn)) = dg.eval(buf) {
                        dg.compute_tails_and_preds(buf);
                        dg.mark_critical(buf, mk, mkn);
                    }
                    no_improve = 0;
                    for t in tabu.iter_mut() { *t = 0; }
                }
                continue;
            }

            let k = verify_k.min(cands.len());
            if k > 0 && cands.len() > k {
                cands.select_nth_unstable_by_key(k - 1, |c| c.est);
            }

            let mut bmk = u32::MAX;
            let mut best_ci: Option<usize> = None;

            for ci in 0..k {
                let c = &cands[ci];
                let (is_tabu, mk_val) = match c.kind {
                    0 => {
                        let o1 = dg.machine_seq[c.m][c.old_idx];
                        let o2 = dg.machine_seq[c.m][c.ins_pos];
                        let tb = tabu[o1] > iter || tabu[o2] > iter;
                        dg.machine_seq[c.m].swap(c.old_idx, c.ins_pos);
                        let mk = dg.eval(buf).map(|(v, _)| v).unwrap_or(u32::MAX);
                        dg.machine_seq[c.m].swap(c.old_idx, c.ins_pos);
                        (tb, mk)
                    },
                    1 => {
                        let tb = tabu[c.op] > iter;
                        dg.machine_seq[c.m].remove(c.old_idx);
                        let ins = c.ins_pos.min(dg.machine_seq[c.m].len());
                        dg.machine_seq[c.m].insert(ins, c.op);
                        let mk = dg.eval(buf).map(|(v, _)| v).unwrap_or(u32::MAX);
                        dg.machine_seq[c.m].remove(ins);
                        dg.machine_seq[c.m].insert(c.old_idx, c.op);
                        (tb, mk)
                    },
                    _ => {
                        let tb = tabu[c.op] > iter;
                        let saved_pt = dg.node_pt[c.op];
                        let saved_machine = dg.node_machine[c.op];
                        dg.machine_seq[c.m].remove(c.old_idx);
                        dg.node_machine[c.op] = c.new_m;
                        dg.node_pt[c.op] = c.new_pt;
                        let ins = c.ins_pos.min(dg.machine_seq[c.new_m].len());
                        dg.machine_seq[c.new_m].insert(ins, c.op);
                        let mk = dg.eval(buf).map(|(v, _)| v).unwrap_or(u32::MAX);
                        dg.machine_seq[c.new_m].remove(ins);
                        dg.node_machine[c.op] = saved_machine;
                        dg.node_pt[c.op] = saved_pt;
                        dg.machine_seq[c.m].insert(c.old_idx, c.op);
                        (tb, mk)
                    },
                };

                if mk_val < bmk && (!is_tabu || mk_val < best_mk) {
                    bmk = mk_val;
                    best_ci = Some(ci);
                }
            }

            if let Some(ci) = best_ci {
                let c = &cands[ci];
                match c.kind {
                    0 => {
                        let o1 = dg.machine_seq[c.m][c.old_idx];
                        let o2 = dg.machine_seq[c.m][c.ins_pos];
                        dg.machine_seq[c.m].swap(c.old_idx, c.ins_pos);
                        tabu[o1] = iter + tenure;
                        tabu[o2] = iter + tenure;
                    },
                    1 => {
                        dg.machine_seq[c.m].remove(c.old_idx);
                        let ins = c.ins_pos.min(dg.machine_seq[c.m].len());
                        dg.machine_seq[c.m].insert(ins, c.op);
                        tabu[c.op] = iter + tenure;
                    },
                    _ => {
                        dg.machine_seq[c.m].remove(c.old_idx);
                        dg.node_machine[c.op] = c.new_m;
                        dg.node_pt[c.op] = c.new_pt;
                        let ins = c.ins_pos.min(dg.machine_seq[c.new_m].len());
                        dg.machine_seq[c.new_m].insert(ins, c.op);
                        tabu[c.op] = iter + tenure;
                    },
                }

                if let Some((mk, mkn)) = dg.eval(buf) {
                    dg.compute_tails_and_preds(buf);
                    dg.mark_critical(buf, mk, mkn);
                    if mk < best_mk {
                        best_mk = mk;
                        best_dg.clone_from(dg);
                        if mk < *global_best {
                            *global_best = mk;
                            let s = dg.to_solution(pd, buf);
                            save(&s)?;
                        }
                        no_improve = 0;
                    } else {
                        no_improve += 1;
                    }
                } else {
                    dg.clone_from(&best_dg);
                    if let Some((mk, mkn)) = dg.eval(buf) {

                        dg.compute_tails_and_preds(buf);
                        dg.mark_critical(buf, mk, mkn);
                    }
                    no_improve += 1;
                }
            } else {
                no_improve += 1;
            }

            if no_improve > noi_limit {
                dg.clone_from(&best_dg);
                perturb(dg, pd, buf, perturb_str, pseed);
                if let Some((mk, mkn)) = dg.eval(buf) {
                    dg.compute_tails_and_preds(buf);
                    dg.mark_critical(buf, mk, mkn);
                }
                no_improve = 0;
                for t in tabu.iter_mut() { *t = 0; }
            }
        }

        Ok((best_mk, best_dg))
    }

    fn perturb(dg: &mut DG, pd: &PD, buf: &mut Buf, strength: usize, pseed: &mut u64) {
        let Some((mk, mkn)) = dg.eval(buf) else { return; };
        dg.compute_tails_and_preds(buf);
        dg.mark_critical(buf, mk, mkn);
        let blocks = find_blocks(dg, buf);
        if blocks.is_empty() { return; }

        let crit_flex: Vec<usize> = (0..pd.n)
            .filter(|&o| buf.crit[o] && pd.op_flex[o] >= 2)
            .collect();

        for _ in 0..strength {
            *pseed ^= (*pseed).wrapping_shl(13);
            *pseed ^= (*pseed).wrapping_shr(7);
            *pseed ^= (*pseed).wrapping_shl(17);

            if !crit_flex.is_empty() && (*pseed % 3) == 0 {
                let ci = (*pseed as usize / 3) % crit_flex.len();
                let node = crit_flex[ci];
                let old_m = dg.node_machine[node];
                *pseed ^= (*pseed).wrapping_shl(13);
                *pseed ^= (*pseed).wrapping_shr(7);
                *pseed ^= (*pseed).wrapping_shl(17);
                let ei = (*pseed as usize) % pd.op_elig[node].len();
                let (new_m, new_pt) = pd.op_elig[node][ei];
                if new_m != old_m {
                    if let Some(from_pos) = dg.machine_seq[old_m].iter().position(|&x| x == node) {
                        dg.machine_seq[old_m].remove(from_pos);
                        dg.node_machine[node] = new_m;
                        dg.node_pt[node] = new_pt;
                        let j = dg.node_job[node];
                        let k = dg.node_op[node];
                        let jpred_end = if k > 0 {
                            let jp = dg.job_offset[j] + k - 1;
                            buf.start[jp] + dg.node_pt[jp]
                        } else { 0 };
                        let near = dg.machine_seq[new_m]
                            .partition_point(|&x| buf.start[x] < jpred_end);
                        let ip = near.min(dg.machine_seq[new_m].len());
                        dg.machine_seq[new_m].insert(ip, node);
                    }
                }
            } else {
                let bi = (*pseed as usize) % blocks.len();
                let block = &blocks[bi];
                let blen = block.e - block.s;
                if blen < 1 { continue; }
                *pseed ^= (*pseed).wrapping_shl(13);
                *pseed ^= (*pseed).wrapping_shr(7);
                *pseed ^= (*pseed).wrapping_shl(17);
                let pos = block.s + (*pseed as usize) % blen;
                dg.machine_seq[block.m].swap(pos, pos + 1);
            }
        }
    }

    fn construct_mc(pd: &PD, rule: usize, rng: &mut SmallRng, top_k: usize) -> (Solution, u32) {
        let nj = pd.nj;
        let nm = pd.nm;
        let mut job_next = vec![0usize; nj];
        let mut job_ready = vec![0u32; nj];
        let mut m_avail = vec![0u32; nm];
        let mut schedule: Vec<Vec<(usize, u32)>> = pd.job_ops_len.iter().map(|&l| Vec::with_capacity(l)).collect();
        let mut remaining = pd.n;
        let mut makespan = 0u32;

        let mut job_rem_work: Vec<f64> = (0..nj).map(|j| {
            (0..pd.job_ops_len[j]).map(|k| {
                let o = pd.op_id(j, k);
                pd.op_avg_pt[o] * 0.7 + pd.op_min_pt[o] as f64 * 0.3
            }).sum()
        }).collect();

        let mut time = 0u32;
        let use_random = top_k > 1;
        let mut cands: Vec<(f64, usize, u32, usize)> = Vec::with_capacity(nj);
        let mut m_order: Vec<usize> = (0..nm).collect();

        while remaining > 0 {
            let mut scheduled_any = false;


            if use_random {
                for i in (1..nm).rev() {
                    let j = rng.gen_range(0..=i);
                    m_order.swap(i, j);
                }
            }

            for &m in &m_order {
                if m_avail[m] > time { continue; }

                cands.clear();

                for j in 0..nj {
                    let k = job_next[j];
                    if k >= pd.job_ops_len[j] { continue; }
                    if job_ready[j] > time { continue; }

                    let o = pd.op_id(j, k);
                    let pt_on_m = pd.op_pt_on[o][m];
                    if pt_on_m == 0 { continue; }


                    let end_on_m = time + pt_on_m;
                    let mut earliest_end = u32::MAX;
                    for &(em, ept) in &pd.op_elig[o] {
                        let eend = time.max(m_avail[em]) + ept;
                        if eend < earliest_end { earliest_end = eend; }
                    }
                    if end_on_m != earliest_end { continue; }

                    let ops_rem = (pd.job_ops_len[j] - k) as f64;
                    let priority = match rule % 5 {
                        0 => job_rem_work[j],
                        1 => ops_rem,
                        2 => -(pd.op_flex[o] as f64),
                        3 => -(pt_on_m as f64),
                        _ => pt_on_m as f64,
                    };

                    cands.push((priority, j, pt_on_m, pd.op_flex[o]));
                }

                if cands.is_empty() { continue; }

                let (j, pt) = if !use_random || cands.len() <= 1 {

                    let mut bi = 0usize;
                    for ci in 1..cands.len() {
                        let (cp, cj, cpt, cflex) = cands[ci];
                        let (bp, bj, bpt, bflex) = cands[bi];
                        let eps = 1e-9;
                        if cp > bp + eps
                            || ((cp - bp).abs() <= eps && cpt < bpt)
                            || ((cp - bp).abs() <= eps && cpt == bpt && cflex < bflex)
                            || ((cp - bp).abs() <= eps && cpt == bpt && cflex == bflex && cj < bj) {
                            bi = ci;
                        }
                    }
                    (cands[bi].1, cands[bi].2)
                } else {
                    cands.sort_unstable_by(|a, b| {
                        let ord = b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal);
                        if ord != std::cmp::Ordering::Equal { return ord; }
                        let ord = a.2.cmp(&b.2);
                        if ord != std::cmp::Ordering::Equal { return ord; }
                        let ord = a.3.cmp(&b.3);
                        if ord != std::cmp::Ordering::Equal { return ord; }
                        a.1.cmp(&b.1)
                    });
                    let pick = rng.gen_range(0..top_k.min(cands.len()));
                    (cands[pick].1, cands[pick].2)
                };

                let start = time;
                let end = start + pt;
                schedule[j].push((m, start));
                m_avail[m] = end;
                job_ready[j] = end;
                if end > makespan { makespan = end; }
                let oid = pd.op_id(j, job_next[j]);
                job_rem_work[j] -= pd.op_avg_pt[oid] * 0.7 + pd.op_min_pt[oid] as f64 * 0.3;
                if job_rem_work[j] < 0.0 { job_rem_work[j] = 0.0; }
                job_next[j] += 1;
                remaining -= 1;
                scheduled_any = true;
            }

            if remaining == 0 { break; }

            let mut next_time = u32::MAX;
            for mm in 0..nm { if m_avail[mm] > time && m_avail[mm] < next_time { next_time = m_avail[mm]; } }
            for j in 0..nj {
                if job_next[j] < pd.job_ops_len[j] && job_ready[j] > time && job_ready[j] < next_time {
                    next_time = job_ready[j];
                }
            }
            if next_time == u32::MAX || next_time == time {
                if !scheduled_any { break; }
                continue;
            }
            time = next_time;
        }

        let mk = if remaining > 0 { INF32 } else { makespan };
        (Solution { job_schedule: schedule }, mk)
    }

    fn construct(pd: &PD, rule: usize, rng: &mut SmallRng, top_k: usize) -> (Solution, u32) {
        let nj = pd.nj;
        let nm = pd.nm;
        let mut job_next_op = vec![0usize; nj];
        let mut job_ready = vec![0u32; nj];
        let mut m_avail = vec![0u32; nm];
        let mut schedule: Vec<Vec<(usize, u32)>> = pd.job_ops_len.iter().map(|&l| Vec::with_capacity(l)).collect();
        let mut remaining = pd.n;
        let mut makespan = 0u32;

        let mut job_rem_work: Vec<f64> = (0..nj).map(|j| {
            (0..pd.job_ops_len[j]).map(|k| pd.op_min_pt[pd.op_id(j, k)] as f64).sum()
        }).collect();
        let avg_load = pd.machine_load0.iter().sum::<f64>() / nm.max(1) as f64;

        let mut cands: Vec<(f64, usize, usize, u32)> = Vec::with_capacity(nj);

        while remaining > 0 {
            cands.clear();
            for j in 0..nj {
                let k = job_next_op[j];
                if k >= pd.job_ops_len[j] { continue; }
                let o = pd.op_id(j, k);
                let jr = job_ready[j];

                let mut best_end = u32::MAX;
                let mut best_m = 0usize;
                let mut best_pt = 0u32;
                let mut second_end = u32::MAX;

                for &(m, pt) in &pd.op_elig[o] {
                    let start = jr.max(m_avail[m]);
                    let end = start + pt;
                    if end < best_end {
                        second_end = best_end;
                        best_end = end; best_m = m; best_pt = pt;
                    } else if end < second_end {
                        second_end = end;
                    }
                }
                if best_end == u32::MAX { continue; }

                let ops_rem = (pd.job_ops_len[j] - k) as f64;
                let flex_inv = 1.0 / (pd.op_flex[o] as f64).max(1.0);
                let rem_work_n = job_rem_work[j] / avg_load.max(1.0);
                let end_n = best_end as f64 / avg_load.max(1.0);

                let score = match rule % 7 {
                    0 => rem_work_n + 0.15 * ops_rem - 0.7 * end_n,
                    1 => ops_rem - 0.5 * end_n,
                    2 => flex_inv + 0.4 * rem_work_n - 0.3 * end_n,
                    3 => -(best_end as f64),
                    4 => -(best_pt as f64) + 0.3 * rem_work_n,
                    5 => {
                        let regret = if second_end < INF32 {
                            (second_end - best_end) as f64 / avg_load.max(1.0)
                        } else {
                            0.0
                        };
                        regret + 0.5 * rem_work_n - 0.3 * end_n
                    }
                    _ => rem_work_n + 0.3 * flex_inv
                        - 0.5 * (pd.machine_load0[best_m] / avg_load.max(1.0))
                        - 0.4 * end_n,
                };

                cands.push((score, j, best_m, best_pt));
            }

            if cands.is_empty() { break; }

            cands.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let pick = if top_k > 1 && cands.len() > 1 { rng.gen_range(0..top_k.min(cands.len())) } else { 0 };
            let (_, j, m, pt) = cands[pick];

            let start = job_ready[j].max(m_avail[m]);
            let end = start + pt;
            schedule[j].push((m, start));
            m_avail[m] = end;
            job_ready[j] = end;
            if end > makespan { makespan = end; }
            let oid = pd.op_id(j, job_next_op[j]);
            job_rem_work[j] -= pd.op_min_pt[oid] as f64;
            if job_rem_work[j] < 0.0 { job_rem_work[j] = 0.0; }
            job_next_op[j] += 1;
            remaining -= 1;
        }

        let mk = if remaining > 0 { INF32 } else { makespan };
        (Solution { job_schedule: schedule }, mk)
    }

    fn insert_top(top: &mut Vec<(Solution, u32)>, sol: Solution, mk: u32, max_size: usize) {
        let pos = top.binary_search_by_key(&mk, |(_, m)| *m).unwrap_or_else(|e| e);
        if pos < max_size {
            top.insert(pos, (sol, mk));
            if top.len() > max_size { top.truncate(max_size); }
        }
    }

    pub fn solve_our(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> Result<()>,
    ) -> Result<()> {
        let pd = PD::new(challenge);
        let mut rng = SmallRng::from_seed(challenge.seed);
        let high_flex = pd.flex_avg > 1.5;


        let max_ops = *pd.job_ops_len.iter().max().unwrap_or(&0);
        let min_ops = *pd.job_ops_len.iter().min().unwrap_or(&0);
        let uniform_routing = max_ops == min_ops;
        let (num_restarts, keep_top, search_iters, final_iters, perturb_str, noi_limit, verify_k) =
            if pd.flex_avg > 5.0 {
                (2000, 5, 5000, 12000, 5, 60, 25)
            } else if high_flex && !uniform_routing {
                // FjspMedium — reduced for speed, still good quality
                (2000, 6, 5000, 12000, 4, 80, 20)
            } else if high_flex {
                // HybridFlowShop path (rarely used — most HFS goes to dedicated solver)
                (1500, 8, 3000, 8000, 4, 100, 15)
            } else {
                (1500, 8, 4000, 10000, 4, 100, 12)
            };


        let mut top: Vec<(Solution, u32)> = Vec::new();
        let mut best_mk = u32::MAX;

        for rule in 0..5 {
            let (sol, mk) = construct_mc(&pd, rule, &mut rng, 1);
            if mk < best_mk && mk < INF32 { best_mk = mk; save_solution(&sol)?; }
            insert_top(&mut top, sol, mk, keep_top);
        }
        for rule in 0..7 {
            let (sol, mk) = construct(&pd, rule, &mut rng, 1);
            if mk < best_mk && mk < INF32 { best_mk = mk; save_solution(&sol)?; }
            insert_top(&mut top, sol, mk, keep_top);
        }
        for i in 0..num_restarts {
            let k = rng.gen_range(2..=5);
            let (sol, mk) = if i % 2 == 0 {
                construct_mc(&pd, rng.gen_range(0..5), &mut rng, k)
            } else {
                construct(&pd, rng.gen_range(0..7), &mut rng, k)
            };
            if mk < best_mk && mk < INF32 { best_mk = mk; save_solution(&sol)?; }
            insert_top(&mut top, sol, mk, keep_top);
        }


        let mut buf = Buf::new(pd.n);
        let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15)
            ^ (best_mk as u64).wrapping_shl(16) ^ (pd.n as u64);
        let mut global_best = best_mk;
        let mut best_dg: Option<DG> = None;

        for i in 0..top.len() {
            let mut dg = match DG::from_solution(&pd, challenge, &top[i].0) {
                Ok(d) => d, Err(_) => continue,
            };
            let iters = if i < 3 { search_iters } else { search_iters / 2 };
            let (mk, rdg) = hybrid_search(
                &mut dg, &pd, &mut buf, iters, high_flex, &mut pseed,
                save_solution, &mut global_best, perturb_str, noi_limit, verify_k,
            )?;
            if best_dg.is_none() || mk < best_mk {
                best_mk = mk;
                best_dg = Some(rdg);
            }
        }


        if let Some(ref bdg) = best_dg {
            let mut dg = bdg.clone();
            let (mk3, dg3) = hybrid_search(
                &mut dg, &pd, &mut buf, final_iters, high_flex, &mut pseed,
                save_solution, &mut global_best, perturb_str, noi_limit, verify_k,
            )?;
            if mk3 < best_mk { best_mk = mk3; best_dg = Some(dg3); }
        }

        Ok(())
    }

}