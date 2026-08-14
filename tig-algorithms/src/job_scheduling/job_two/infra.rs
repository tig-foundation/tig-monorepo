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
            let mut top: Vec<Cand> = if k > 0 { Vec::with_capacity(k) } else { Vec::new() };

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
