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

    let mut best_sim = 0usize;
    let mut best_idx = NONE_USIZE;
    for (i, (s2, _mk2)) in top.iter().enumerate() {
        let sig2 = signature(s2);
        let sim = similarity(&sig_new, &sig2);
        if sim > best_sim {
            best_sim = sim;
            best_idx = i;
        }
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

    let mut sigs: Vec<Vec<usize>> = Vec::with_capacity(top.len());
    for (s, _) in top.iter() {
        sigs.push(signature(s));
    }

    let mut evict_idx = 0usize;
    let mut evict_crowd = 0usize;
    for i in 0..top.len() {
        let mut crowd = 0usize;
        for j in 0..top.len() {
            if i == j {
                continue;
            }
            let sim = similarity(&sigs[i], &sigs[j]);
            if sim > crowd {
                crowd = sim;
            }
        }
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

    for _iter in 0..max_iters {
        crit.fill(false);
        let mut u = cur_eval.1;
        while u != NONE_USIZE {
            crit[u] = true;
            u = buf.best_pred[u];
        }

        let impact_proxy = |u: usize| -> u32 {
            let end_u = buf.start[u].saturating_add(ds.node_pt[u]);
            let js = ds.job_succ[u];
            let ms = buf.machine_succ[u];

            let js_st = if js != NONE_USIZE { buf.start[js] } else { end_u };
            let ms_st = if ms != NONE_USIZE { buf.start[ms] } else { end_u };

            let gap_job = js_st.saturating_sub(end_u);
            let gap_mach = ms_st.saturating_sub(end_u);

            end_u.saturating_add(gap_job.saturating_add(gap_mach))
        };

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
                    for &sh in &[1usize, 2, max_shift] {
                        if sh == 0 || sh > max_shift {
                            continue;
                        }
                        let from = bstart;
                        let to_after = bstart + sh;
                        if from < seq.len() && to_after <= seq.len() {
                            let tgt_idx = (bstart + sh).min(seq.len() - 1);
                            let sc = impact_proxy(seq[from]).max(impact_proxy(seq[tgt_idx]));
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
                                top_cands,
                            );
                        }
                        let from2 = bend;
                        let to_after2 = bend - sh;
                        let tgt_idx2 = (bend - sh).min(seq.len().saturating_sub(1));
                        let sc2 = impact_proxy(seq[from2]).max(impact_proxy(seq[tgt_idx2]));
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
                            top_cands,
                        );
                    }

                    if bstart > 0 {
                        let sc = impact_proxy(seq[bstart - 1]).max(impact_proxy(seq[bstart]));
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
                            top_cands,
                        );
                    }
                    if bend + 1 < seq.len() {
                        let sc = impact_proxy(seq[bend]).max(impact_proxy(seq[bend + 1]));
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
                            top_cands,
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
                        let sc = impact_proxy(seq[lo]).max(impact_proxy(seq[hi]));
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
                            top_cands,
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
            if cand.kind == 0 {
                let m = cand.m_from;
                if m >= ds.num_machines || cand.from >= ds.machine_seq[m].len() {
                    continue;
                }
                let new_idx = apply_insert_fs(&mut ds.machine_seq[m], cand.from, cand.to);
                if let Some((mk2, _)) = eval_disj(ds, buf) {
                    if mk2 < best_mk {
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
                    if mk2 < best_mk {
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
                    if mk2 < best_mk {
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