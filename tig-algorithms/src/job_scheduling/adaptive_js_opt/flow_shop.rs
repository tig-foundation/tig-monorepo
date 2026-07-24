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

struct AdaptiveBoost {
    boost_strength: f64,
    ema_delta: f64,
    n_samples: usize,
}

impl AdaptiveBoost {
    fn new(_pre: &Pre) -> Self {
        AdaptiveBoost { boost_strength: 1.0, ema_delta: 0.0, n_samples: 0 }
    }

    fn compute_base(pre: &Pre, _progress: f64) -> (f64, f64) {
        let _ = _progress; 
        let base = 0.12 + 0.08 * pre.jobshopness + 0.10 * pre.avg_machine_scarcity;
        let conflict_w = base.clamp(0.05, 0.45);
        let conflict_scale = (0.85 + 0.45 * pre.flex_factor).clamp(0.8, 1.8);
        (conflict_w, conflict_scale)
    }

    fn update_from_test(&mut self, mk_boost: u32, mk_no_boost: u32) {
        let delta = (mk_no_boost as f64 - mk_boost as f64) / mk_boost.max(1) as f64;
        let lr = 0.05;
        self.ema_delta += lr * (delta - self.ema_delta);
        self.boost_strength = (1.0 + 0.1 * self.ema_delta).clamp(0.5, 2.0);
        self.n_samples += 1;
    }
}

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
    _target_mk: Option<u32>,
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
    let rem_min_n = rem_min / pre.horizon.max(1.0);
    let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
    let density_n = if pre.high_flex != 0.0 || rule == Rule::BackwardCritical {
        ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0)
    } else {
        0.0
    };
    let next_min = pre.product_next_min[product][op_idx] as f64;
    let next_min_n = next_min / pre.horizon.max(1.0);
    let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
    let p2 = progress * progress;
    let next_w_base = 0.12 + p2 * 0.28;
    let next_term_raw =
        (0.55 * next_min_n + 0.45 * next_flex_inv) * (1.0 + 0.30 * density_n * pre.high_flex);
    let pop_pen = if pre.chaotic_like && op.flex >= 2 {
        let pop = pre.machine_best_pop[machine];
        (0.07 + 0.15 * (1.0 - progress)).clamp(0.05, 0.24) * pop * pre.flex_factor
    } else {
        0.0
    };

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

    match rule {
        Rule::BnHeavy => {
            let rem_bn = pre.product_suf_bn[product][op_idx];
            let ops_n = (ops_rem as f64) / (pre.max_ops as f64).max(1.0);
            let end_n = (best_end as f64) / pre.time_scale.max(1.0);
            let proc_n = (pt as f64) / pre.avg_op_min.max(1.0);
            let regret = if second_end >= INF {
                pre.avg_op_min * 2.6
            } else {
                (second_end - best_end) as f64
            };
            let reg_n = (regret / pre.avg_op_min.max(1.0)).clamp(0.0, 6.0);
            let js = pre.jobshopness;
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
            let rem_avg = pre.product_suf_avg[product][op_idx];
            let ops_n = (ops_rem as f64) / (pre.max_ops as f64).max(1.0);
            let end_n = (best_end as f64) / pre.time_scale.max(1.0);
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
            let ops_n = (ops_rem as f64) / (pre.max_ops as f64).max(1.0);
            let end_n = (best_end as f64) / pre.time_scale.max(1.0);
            let proc_n = (pt as f64) / pre.avg_op_min.max(1.0);
            let regret = if second_end >= INF {
                pre.avg_op_min * 2.6
            } else {
                (second_end - best_end) as f64
            };
            let reg_n = (regret / pre.avg_op_min.max(1.0)).clamp(0.0, 6.0);
            let js = pre.jobshopness;
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
            let end_n = (best_end as f64) / pre.time_scale.max(1.0);
            let proc_n = (pt as f64) / pre.avg_op_min.max(1.0);
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
            let flex_f = (op.flex as f64).max(1.0);
            let flex_inv = 1.0 / flex_f;
            let end_n = (best_end as f64) / pre.time_scale.max(1.0);
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
            let ops_n = (ops_rem as f64) / (pre.max_ops as f64).max(1.0);
            let end_n = (best_end as f64) / pre.time_scale.max(1.0);
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
            let end_n = (best_end as f64) / pre.time_scale.max(1.0);
            let regret = if second_end >= INF {
                pre.avg_op_min * 2.6
            } else {
                (second_end - best_end) as f64
            };
            let reg_n = (regret / pre.avg_op_min.max(1.0)).clamp(0.0, 6.0);
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
            let proc_n = (pt as f64) / pre.avg_op_min.max(1.0);
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
            let end_n = (best_end as f64) / pre.time_scale.max(1.0);
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
            let end_n = (best_end as f64) / pre.time_scale.max(1.0);
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
            let end_n = (best_end as f64) / pre.time_scale.max(1.0);
            let bn_suf = pre.product_suf_bn[product][op_idx] as f64 / pre.max_job_bn.max(1e-9);
            let next_term = next_w_base * 0.30 * next_term_raw;
            (1.15 * bn_suf)
                + (0.45 * density_n)
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
            let rem_avg = pre.product_suf_avg[product][op_idx];
            let end_n = (best_end as f64) / pre.time_scale.max(1.0);
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
    adaptive_boost: &mut AdaptiveBoost,
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
    let mut best_idle_alts: Vec<(usize, u32)> = Vec::with_capacity(num_machines);
    let mut ready_jobs: Vec<u64> = vec![0u64; (num_jobs + 63) / 64];
    let mut future_jobs: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();

    for job in 0..num_jobs {
        if pre.job_ops_len[job] > 0 {
            ready_jobs[job >> 6] |= 1u64 << (job & 63);
        }
    }

    while remaining_ops > 0 {
        while let Some(Reverse((release, job))) = future_jobs.peek().copied() {
            if release > time {
                break;
            }
            future_jobs.pop();
            if job_next_op[job] < pre.job_ops_len[job] && job_ready_time[job] == release {
                ready_jobs[job >> 6] |= 1u64 << (job & 63);
            }
        }

        idle_machines.clear();
        for m in 0..num_machines {
            if machine_avail[m] <= time {
                idle_machines.push(m);
            }
        }

        loop {
            if idle_machines.is_empty() {
                break;
            }
            for &m in &idle_machines {
                demand[m] = 0;
                raw_by_machine[m].clear();
            }
            let progress = 1.0 - (remaining_ops as f64) / (pre.total_ops as f64).max(1.0);
            let cap_per_machine = if k == 0 { 12usize } else { (k + 6).min(12) };

            for word_idx in 0..ready_jobs.len() {
                let mut ready = ready_jobs[word_idx];
                while ready != 0 {
                    let bit = ready.trailing_zeros() as usize;
                    ready &= ready - 1;
                    let job = (word_idx << 6) + bit;
                let op_idx = job_next_op[job];
                if op_idx >= pre.job_ops_len[job] {
                    continue;
                }
                let product = pre.job_products[job];
                let op = &pre.product_ops[product][op_idx];
                if op.flex == 0 || op.machines.is_empty() || op.min_pt >= INF {
                    continue;
                }
                let mut best_end = INF;
                let mut second_end = INF;
                let mut best_cnt_total = 0usize;
                best_idle_alts.clear();
                for &(m, pt) in &op.machines {
                    let idle = machine_avail[m] <= time;
                    let end = time.max(machine_avail[m]).saturating_add(pt);
                    if end < best_end {
                        second_end = best_end;
                        best_end = end;
                        best_cnt_total = 1;
                        best_idle_alts.clear();
                        if idle {
                            best_idle_alts.push((m, pt));
                        }
                    } else if end == best_end {
                        best_cnt_total += 1;
                        if idle {
                            best_idle_alts.push((m, pt));
                        }
                    } else if end < second_end {
                        second_end = end;
                    }
                }
                if best_cnt_total > 1 {
                    second_end = best_end;
                }
                if best_end >= INF || best_idle_alts.is_empty() {
                    continue;
                }
                let best_cnt_total = best_cnt_total.max(1);
                let ops_rem = pre.job_ops_len[job] - op_idx;
                let jb = job_bias.map(|v| v[job]).unwrap_or(0.0);
                let regret = if second_end >= INF {
                    pre.avg_op_min * 2.6
                } else {
                    (second_end - best_end) as f64
                };
                let regn = (regret / pre.avg_op_min.max(1.0)).clamp(0.0, 6.0);
                let rem_min = pre.product_suf_min[product][op_idx] as f64;
                let cp_pen = if let Some(tmk) = target_mk {
                    let excess = (best_end as f64 + rem_min - tmk as f64).max(0.0);
                    (excess / pre.avg_op_min.max(1.0)).clamp(0.0, 8.0)
                } else {
                    0.0
                };

                for &(m, pt) in &best_idle_alts {
                    demand[m] = demand[m].saturating_add(1);
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
                        0.0,
                        if rule == Rule::MachineBalance { machine_load[m] } else { 0.0 },
                        route_pref,
                        route_w,
                        jitter,
                    );
                    push_top_k_raw(
                        &mut raw_by_machine[m],
                        RawCand { job, machine: m, pt, base_score: base, rigidity: cp_pen, reg_n: regn },
                        cap_per_machine,
                    );
                }
                }
            }

            let denom = (idle_machines.len() as f64).max(1.0);
            let (conflict_w, conflict_scale) = AdaptiveBoost::compute_base(pre, progress);

            let mut best: Option<Cand> = None;
            let mut top: Vec<Cand> = if k > 0 { Vec::with_capacity(k) } else { Vec::new() };

            for &m in &idle_machines {
                let dem = demand[m] as f64;
                if dem <= 0.0 || raw_by_machine[m].is_empty() {
                    continue;
                }
                let dem_n = ((dem - 1.0) / denom).clamp(0.0, 2.5);
                for rc in &raw_by_machine[m] {
                    let cp = rc.rigidity.clamp(0.0, 8.0);
                    let regc = rc.reg_n.clamp(0.0, 4.5);
                    let boost = conflict_w * conflict_scale * dem_n * (0.85 * regc + 0.35 * cp) * adaptive_boost.boost_strength;
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
            ready_jobs[job >> 6] &= !(1u64 << (job & 63));
            let op = &pre.product_ops[product][op_idx];
            let end_time = time.saturating_add(pt);
            job_schedule[job].push((machine, time));
            job_next_op[job] += 1;
            job_ready_time[job] = end_time;
            machine_avail[machine] = end_time;
            if end_time > time {
                let pos = idle_machines.binary_search(&machine).unwrap();
                idle_machines.remove(pos);
            }
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
            if job_next_op[job] < pre.job_ops_len[job] {
                if end_time <= time {
                    ready_jobs[job >> 6] |= 1u64 << (job & 63);
                } else {
                    future_jobs.push(Reverse((end_time, job)));
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
        if let Some(Reverse((t, _))) = future_jobs.peek().copied() {
            if t > time {
                next_time = Some(next_time.map_or(t, |b| b.min(t)));
            }
        }
        time = next_time.ok_or_else(|| anyhow!("Stalled: no next event"))?;
    }
    let mk = machine_avail.into_iter().max().unwrap_or(0);
    Ok((Solution { job_schedule }, mk))
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
    let len = top.len();
    if len == k && top[len - 1].base_score >= c.base_score {
        return;
    }
    let mut pos = len;
    while pos > 0 && top[pos - 1].base_score < c.base_score {
        pos -= 1;
    }
    if pos >= k {
        return;
    }
    if len < k {
        top.reserve(1);
        unsafe {
            let ptr = top.as_mut_ptr();
            std::ptr::copy(ptr.add(pos), ptr.add(pos + 1), len - pos);
            std::ptr::write(ptr.add(pos), c);
            top.set_len(len + 1);
        }
    } else {
        unsafe {
            let ptr = top.as_mut_ptr();
            std::ptr::drop_in_place(ptr.add(len - 1));
            std::ptr::copy(ptr.add(pos), ptr.add(pos + 1), len - pos - 1);
            std::ptr::write(ptr.add(pos), c);
        }
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
    
    if new_crowd < evict_crowd || (new_crowd <= evict_crowd + 1 && mk < top[evict_idx].1) {
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
    let mut machine_run_starts: Vec<Vec<usize>> = vec![Vec::new(); num_machines];
    let mut last_run_job = vec![NONE_USIZE; num_machines];
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
            if last_run_job[m] != job {
                machine_run_starts[m].push(per_machine[m].len());
                last_run_job[m] = job;
            }
            per_machine[m].push((st, id));
        }
    }
    let mut machine_seq: Vec<Vec<usize>> = Vec::with_capacity(num_machines);
    for m in 0..num_machines {
        let entries = &mut per_machine[m];
        let runs = &machine_run_starts[m];
        if entries.len() <= 1 || runs.len() <= 1 {
            machine_seq.push(entries.iter().map(|&(_, id)| id).collect());
            continue;
        }

        let mut multi_nodes = 0usize;
        let mut singleton_nodes = 0usize;
        for r in 0..runs.len() {
            let end = if r + 1 < runs.len() {
                runs[r + 1]
            } else {
                entries.len()
            };
            let len = end - runs[r];
            if len > 1 {
                multi_nodes += len;
            } else {
                singleton_nodes += 1;
            }
        }

        if multi_nodes > singleton_nodes {
            let mut heap: BinaryHeap<Reverse<(u32, usize, usize, usize)>> =
                BinaryHeap::with_capacity(runs.len());
            for r in 0..runs.len() {
                let pos = runs[r];
                let end = if r + 1 < runs.len() {
                    runs[r + 1]
                } else {
                    entries.len()
                };
                let (st, id) = entries[pos];
                heap.push(Reverse((st, id, pos, end)));
            }
            let mut seq = Vec::with_capacity(entries.len());
            while let Some(Reverse((_, id, pos, end))) = heap.pop() {
                seq.push(id);
                let next = pos + 1;
                if next < end {
                    let (st, next_id) = entries[next];
                    heap.push(Reverse((st, next_id, next, end)));
                }
            }
            machine_seq.push(seq);
        } else {
            entries.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            machine_seq.push(entries.iter().map(|&(_, id)| id).collect());
        }
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

#[inline]
fn rebuild_eval_machine_state_into(
    ds: &DisjSchedule,
    machine_succ: &mut [usize],
    indeg: &mut [u16],
    indeg_job: &[u16],
) {
    indeg.clone_from_slice(indeg_job);
    machine_succ.fill(NONE_USIZE);
    for seq in &ds.machine_seq {
        if seq.len() <= 1 {
            continue;
        }
        for i in 0..(seq.len() - 1) {
            let u = seq[i];
            let v = seq[i + 1];
            machine_succ[u] = v;
            indeg[v] = indeg[v].saturating_add(1);
        }
    }
}

#[inline]
fn patch_eval_machine_state_one_machine(
    ds: &DisjSchedule,
    buf: &mut EvalBuf,
    indeg_base: &mut [u16],
    machine: usize,
) {
    if machine >= ds.num_machines {
        return;
    }
    let seq = &ds.machine_seq[machine];
    let len = seq.len();
    if len == 0 {
        return;
    }
    for i in 0..(len - 1) {
        let u = seq[i];
        let new_v = seq[i + 1];
        let old_v = buf.machine_succ[u];
        if old_v != new_v {
            if old_v != NONE_USIZE {
                indeg_base[old_v] = indeg_base[old_v].saturating_sub(1);
            }
            indeg_base[new_v] = indeg_base[new_v].saturating_add(1);
            buf.machine_succ[u] = new_v;
        }
    }
    let u = seq[len - 1];
    let old_v = buf.machine_succ[u];
    if old_v != NONE_USIZE {
        indeg_base[old_v] = indeg_base[old_v].saturating_sub(1);
        buf.machine_succ[u] = NONE_USIZE;
    }
}

#[inline]
fn eval_disj_prepared(ds: &DisjSchedule, buf: &mut EvalBuf) -> Option<(u32, usize)> {
    let n = ds.n;
    buf.start.fill(0);
    buf.best_pred.fill(NONE_USIZE);
    buf.stack.clear();
    for job in 0..ds.num_jobs {
        let root = ds.job_offsets[job];
        if root < ds.job_offsets[job + 1] && buf.indeg[root] == 0 {
            buf.stack.push(root);
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
fn eval_disj_stateful(ds: &DisjSchedule, buf: &mut EvalBuf, indeg_base: &[u16]) -> Option<(u32, usize)> {
    buf.indeg.clone_from_slice(indeg_base);
    eval_disj_prepared(ds, buf)
}

fn eval_disj(ds: &DisjSchedule, buf: &mut EvalBuf) -> Option<(u32, usize)> {
    rebuild_eval_machine_state_into(ds, &mut buf.machine_succ, &mut buf.indeg, &ds.indeg_job);
    eval_disj_prepared(ds, buf)
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
    let _ = (old_arcs, new_arcs, tmp_seq);
    let m = cand.m_from;
    if m >= ds.num_machines {
        return false;
    }
    let seq = &ds.machine_seq[m];
    let len = seq.len();
    if len <= 1 {
        return false;
    }

    let hits = |left: usize, right: usize| -> bool {
        let key = arc_key_fs(m, left, right);
        for i in 0..tabu_keys.len() {
            if tabu_keys[i] == key && tabu_until[i] > step {
                return true;
            }
        }
        false
    };

    match cand.kind {
        0 => {
            let from = cand.from;
            if from >= len {
                return false;
            }
            let t = cand.to.min(len - 1);
            if t == from {
                return false;
            }
            let x = seq[from];
            if t < from {
                if t > 0 && hits(seq[t - 1], seq[t]) {
                    return true;
                }
                if from > 0 && hits(seq[from - 1], x) {
                    return true;
                }
                if from + 1 < len && hits(x, seq[from + 1]) {
                    return true;
                }
            } else {
                if from > 0 && hits(seq[from - 1], x) {
                    return true;
                }
                if from + 1 < len && hits(x, seq[from + 1]) {
                    return true;
                }
                if t + 1 < len && hits(seq[t], seq[t + 1]) {
                    return true;
                }
            }
            false
        }
        2 => {
            let from = cand.from;
            if from + 1 >= len {
                return false;
            }
            if from > 0 && hits(seq[from - 1], seq[from]) {
                return true;
            }
            if hits(seq[from], seq[from + 1]) {
                return true;
            }
            if from + 2 < len && hits(seq[from + 1], seq[from + 2]) {
                return true;
            }
            false
        }
        3 => {
            let from = cand.from;
            let to = cand.to;
            if from >= len || to >= len || from + 1 >= to {
                return false;
            }
            if from > 0 && hits(seq[from - 1], seq[from]) {
                return true;
            }
            if hits(seq[from], seq[from + 1]) {
                return true;
            }
            if hits(seq[to - 1], seq[to]) {
                return true;
            }
            if to + 1 < len && hits(seq[to], seq[to + 1]) {
                return true;
            }
            false
        }
        _ => false,
    }
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
    if tabu_keys.is_empty() || before_seq.len() < 3 || after_seq.len() <= 1 {
        return;
    }

    let kind = before_seq[0] as u8;
    let from = before_seq[1];
    let to = before_seq[2];
    let len = after_seq.len();
    let t = to.min(len - 1);
    let mut old_pos = [0usize; 4];
    let mut new_pos = [0usize; 4];
    let mut old_count = 0usize;
    let mut new_count = 0usize;

    {
        let mut add_old = |p: usize| {
            if p + 1 < len {
                old_pos[old_count] = p;
                old_count += 1;
            }
        };
        let mut add_new = |p: usize| {
            if p + 1 < len {
                new_pos[new_count] = p;
                new_count += 1;
            }
        };

        match kind {
            0 => {
                if from >= len || t == from {
                    return;
                }
                if t < from {
                    if t > 0 {
                        add_old(t - 1);
                        add_new(t - 1);
                    }
                    add_old(from - 1);
                    add_old(from);
                    add_new(t);
                    add_new(from);
                } else {
                    if from > 0 {
                        add_old(from - 1);
                        add_new(from - 1);
                    }
                    add_old(from);
                    add_old(t);
                    add_new(t - 1);
                    add_new(t);
                }
            }
            2 => {
                if from + 1 >= len {
                    return;
                }
                if from > 0 {
                    add_old(from - 1);
                    add_new(from - 1);
                }
                add_old(from);
                add_new(from);
                add_old(from + 1);
                add_new(from + 1);
            }
            3 => {
                if from >= len || to >= len || from + 1 >= to {
                    return;
                }
                if from > 0 {
                    add_old(from - 1);
                    add_new(from - 1);
                }
                add_old(from);
                add_new(from);
                add_old(to - 1);
                add_new(to - 1);
                add_old(to);
                add_new(to);
            }
            _ => return,
        }
    }

    let old_node = |i: usize| -> usize {
        match kind {
            0 => {
                if t < from {
                    if i < t || i > from {
                        after_seq[i]
                    } else if i == from {
                        after_seq[t]
                    } else {
                        after_seq[i + 1]
                    }
                } else if i < from || i > t {
                    after_seq[i]
                } else if i == from {
                    after_seq[t]
                } else {
                    after_seq[i - 1]
                }
            }
            2 => {
                if i == from {
                    after_seq[from + 1]
                } else if i == from + 1 {
                    after_seq[from]
                } else {
                    after_seq[i]
                }
            }
            3 => {
                if i == from {
                    after_seq[to]
                } else if i == to {
                    after_seq[from]
                } else {
                    after_seq[i]
                }
            }
            _ => after_seq[i],
        }
    };

    old_arcs.clear();
    for &p in &old_pos[..old_count] {
        let key = arc_key_fs(machine, old_node(p), old_node(p + 1));
        if !old_arcs.iter().any(|&k| k == key) {
            old_arcs.push(key);
        }
    }

    new_pos[..new_count].sort_unstable();
    new_arcs.clear();
    for i in 0..new_count {
        if i > 0 && new_pos[i] == new_pos[i - 1] {
            continue;
        }
        let p = new_pos[i];
        let key = arc_key_fs(machine, after_seq[p], after_seq[p + 1]);
        if !old_arcs.iter().any(|&k| k == key) {
            new_arcs.push(key);
        }
    }

    for &key in new_arcs.iter() {
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
    let mut global_best_mk = cur_eval.0;
    let mut global_best_machine_seq = ds.machine_seq.clone();
    let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ (initial_mk as u64).wrapping_shl(16)
        ^ (ds.n as u64);
    let mut perturbed_states: std::collections::HashMap<u64, Vec<Vec<Vec<usize>>>> =
        std::collections::HashMap::new();
    for _cycle in 0..perturb_cycles {
        ds.machine_seq.clone_from(&global_best_machine_seq);
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
        let mut state_key = 1469598103934665603u64 ^ (ds.machine_seq.len() as u64);
        for seq in &ds.machine_seq {
            state_key ^= seq.len() as u64;
            state_key = state_key.wrapping_mul(1099511628211u64);
            for &node in seq {
                state_key ^= (node as u64).wrapping_add(1);
                state_key = state_key.wrapping_mul(1099511628211u64);
            }
        }
        let repeated = perturbed_states
            .get(&state_key)
            .map_or(false, |states| states.iter().any(|state| state == &ds.machine_seq));
        if repeated {
            continue;
        }
        perturbed_states
            .entry(state_key)
            .or_default()
            .push(ds.machine_seq.clone());
        match eval_disj(&ds, &mut buf) {
            Some(x) => cur_eval = x,
            None => continue,
        }
        descent_phase(&mut ds, &mut buf, &mut crit, pre, &mut cur_eval, max_iters, top_cands);
        if cur_eval.0 < global_best_mk {
            global_best_mk = cur_eval.0;
            global_best_machine_seq.clone_from(&ds.machine_seq);
            perturbed_states.clear();
        }
    }
    if global_best_mk >= initial_mk {
        return Ok(None);
    }
    ds.machine_seq = global_best_machine_seq;
    let Some((mk_final, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let sol = disj_to_solution(pre, &ds, &buf.start)?;
    Ok(Some((sol, mk_final)))
}

#[inline(always)]
fn swap_vec_u32(a: &mut Vec<u32>, b: &mut Vec<u32>) {
    unsafe {
        std::ptr::swap(a, b);
    }
}

#[inline(always)]
fn swap_vec_usize(a: &mut Vec<usize>, b: &mut Vec<usize>) {
    unsafe {
        std::ptr::swap(a, b);
    }
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

    let mut aspiration: u32 = 0;

    let hlen = max_iters.max(1);
    let mut delta_hist: Vec<u32> = vec![0u32; hlen];
    let mut delta_sorted: Vec<u32> = Vec::with_capacity(hlen);
    let mut dhpos: usize = 0;
    let mut dhfill: usize = 0;

    let mut tenure = 2usize;
    let tabu_cap = 48usize;
    let mut tabu_keys: Vec<u64> = vec![0u64; tabu_cap];
    let mut tabu_until: Vec<usize> = vec![0usize; tabu_cap];
    let mut tabu_pos: usize = 0;
    let mut arc_old: Vec<u64> = Vec::with_capacity(4);
    let mut arc_new: Vec<u64> = Vec::with_capacity(4);
    let mut tmp_seq: Vec<usize> = Vec::new();
    let n = ds.n;
    let mut chain: Vec<usize> = Vec::with_capacity(n);
    let mut crit_tail = vec![0u32; n];
    let mut crit_rank = vec![NONE_USIZE; n];
    let mut machine_pos = vec![NONE_USIZE; n];
    let mut crit_blocks: Vec<(usize, usize, usize)> = Vec::new();
    let mut indeg_base = vec![0u16; n];
    rebuild_eval_machine_state_into(ds, &mut buf.machine_succ, &mut indeg_base, &ds.indeg_job);
    let mut trial_memo: Vec<(MoveCand, Option<(u32, usize)>)> = Vec::with_capacity(48);
    let mut best_trial_start = vec![0u32; n];
    let mut best_trial_pred = vec![NONE_USIZE; n];
    let mut surrogate_value = vec![0u64; n];
    let mut surrogate_epoch = vec![NONE_USIZE; n];
   
    let mut recent_makespans: Vec<u32> = vec![0u32; 50];
    let mut recent_fill: usize = 0;
    let mut recent_idx: usize = 0;
    let mut non_improving_count: usize = 0;

    for iter_ix in 0..max_iters {
        trial_memo.clear();
        crit.fill(false);
        let mut u = cur_eval.1;
        while u != NONE_USIZE {
            crit[u] = true;
            u = buf.best_pred[u];
        }

        let prescreen_k = top_cands.min(48).max(8);

        chain.clear();
        let mut z = cur_eval.1;
        while z != NONE_USIZE {
            chain.push(z);
            z = buf.best_pred[z];
        }
        chain.reverse();

        crit_tail.fill(0);
        crit_rank.fill(NONE_USIZE);
        let mut carry = 0u32;
        for idx in (0..chain.len()).rev() {
            let v = chain[idx];
            carry = ds.node_pt[v].saturating_add(carry);
            crit_tail[v] = carry;
            crit_rank[v] = idx;
        }

        machine_pos.fill(NONE_USIZE);
        crit_blocks.clear();
        for m in 0..ds.num_machines {
            let seq = &ds.machine_seq[m];
            let len = seq.len();
            let mut i = 0usize;
            while i < len {
                let a = seq[i];
                machine_pos[a] = i;
                if !crit[a] {
                    i += 1;
                    continue;
                }
                let bstart = i;
                let mut bend = i;
                while bend + 1 < len {
                    let y = seq[bend + 1];
                    machine_pos[y] = bend + 1;
                    if !crit[y] {
                        break;
                    }
                    let x = seq[bend];
                    let end_x = buf.start[x].saturating_add(ds.node_pt[x]);
                    if buf.start[y] != end_x {
                        break;
                    }
                    bend += 1;
                }
                if bend > bstart {
                    crit_blocks.push((m, bstart, bend));
                }
                i = bend + 1;
            }
        }

        let cycle_detected = recent_fill > 0
            && recent_makespans[..recent_fill].iter().any(|&x| x == cur_mk);
        if cycle_detected {
            tenure = 12;
        }

        let mut op_surrogate = |u: usize| -> u64 {
            if surrogate_epoch[u] == iter_ix {
                return surrogate_value[u];
            }
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

            let value = (end_u as u64) * 7
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
                + if near_chain { (ptu as u64) * 2 } else { 0 };
            surrogate_value[u] = value;
            surrogate_epoch[u] = iter_ix;
            value
        };

        let mut pair_surrogate = |u: usize, v: usize| -> u32 {
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
        for &(m, bstart, bend) in &crit_blocks {
            let seq = &ds.machine_seq[m];
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

        if cands.is_empty() {
            break;
        }

        let mut best_cand: Option<MoveCand> = None;
        let mut best_mk = u32::MAX;
        let mut best_trial_eval: Option<(u32, usize)> = None;

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

            let mut evaluated_now = false;
            let trial_result = if let Some((_, result)) = trial_memo.iter().find(|(seen, _)| {
                seen.kind == cand.kind
                    && seen.m_from == cand.m_from
                    && seen.from == cand.from
                    && seen.to == cand.to
            }) {
                *result
            } else {
                evaluated_now = true;
                let result = if cand.kind == 0 {
                    let m = cand.m_from;
                    if m >= ds.num_machines || cand.from >= ds.machine_seq[m].len() {
                        None
                    } else {
                        let new_idx = apply_insert_fs(&mut ds.machine_seq[m], cand.from, cand.to);
                        patch_eval_machine_state_one_machine(ds, buf, &mut indeg_base, m);
                        let result = eval_disj_stateful(ds, buf, &indeg_base);
                        let _ = apply_insert_fs(&mut ds.machine_seq[m], new_idx, cand.from);
                        patch_eval_machine_state_one_machine(ds, buf, &mut indeg_base, m);
                        result
                    }
                } else if cand.kind == 2 {
                    let m = cand.m_from;
                    if m >= ds.num_machines || cand.from + 1 >= ds.machine_seq[m].len() {
                        None
                    } else {
                        ds.machine_seq[m].swap(cand.from, cand.from + 1);
                        patch_eval_machine_state_one_machine(ds, buf, &mut indeg_base, m);
                        let result = eval_disj_stateful(ds, buf, &indeg_base);
                        ds.machine_seq[m].swap(cand.from, cand.from + 1);
                        patch_eval_machine_state_one_machine(ds, buf, &mut indeg_base, m);
                        result
                    }
                } else if cand.kind == 3 {
                    let m = cand.m_from;
                    if m >= ds.num_machines {
                        None
                    } else {
                        let len = ds.machine_seq[m].len();
                        if cand.from >= len || cand.to >= len || cand.from + 1 >= cand.to {
                            None
                        } else {
                            ds.machine_seq[m].swap(cand.from, cand.to);
                            patch_eval_machine_state_one_machine(ds, buf, &mut indeg_base, m);
                            let result = eval_disj_stateful(ds, buf, &indeg_base);
                            ds.machine_seq[m].swap(cand.from, cand.to);
                            patch_eval_machine_state_one_machine(ds, buf, &mut indeg_base, m);
                            result
                        }
                    }
                } else {
                    None
                };
                trial_memo.push((*cand, result));
                result
            };

            if let Some(eval2) = trial_result {
                let mk2 = eval2.0;
                if mk2 < best_mk && (!cand_tabu || mk2 < best_seen.saturating_add(aspiration)) {
                    best_mk = mk2;
                    best_cand = Some(*cand);
                    best_trial_eval = Some(eval2);
                    if evaluated_now {
                        swap_vec_u32(&mut best_trial_start, &mut buf.start);
                        swap_vec_usize(&mut best_trial_pred, &mut buf.best_pred);
                    }
                }
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
        if dhfill < hlen {
            delta_hist[dhpos] = d;
            let pos = delta_sorted.binary_search(&d).unwrap_or_else(|p| p);
            delta_sorted.insert(pos, d);
            dhfill += 1;
        } else {
            let old = delta_hist[dhpos];
            if let Ok(pos) = delta_sorted.binary_search(&old) {
                delta_sorted.remove(pos);
            }
            delta_hist[dhpos] = d;
            let pos = delta_sorted.binary_search(&d).unwrap_or_else(|p| p);
            delta_sorted.insert(pos, d);
        }
        dhpos += 1;
        if dhpos >= hlen {
            dhpos = 0;
        }

        let band = if dhfill == 0 {
            0
        } else {
            delta_sorted[dhfill >> 1]
        };
        let rrt_limit = best_seen.saturating_add(band);

        let mut accepted = false;
        let mut global_improved = false;
        if bc.kind == 0 {
            let m = bc.m_from;
            let before_seq = [bc.kind as usize, bc.from, bc.to];
            let new_idx = apply_insert_fs(&mut ds.machine_seq[m], bc.from, bc.to);
            patch_eval_machine_state_one_machine(ds, buf, &mut indeg_base, m);
            let next_eval = best_trial_eval.unwrap();
            swap_vec_u32(&mut best_trial_start, &mut buf.start);
            swap_vec_usize(&mut best_trial_pred, &mut buf.best_pred);
            let next_mk = next_eval.0;
            if next_mk < prev_mk || next_mk <= rrt_limit {
                *cur_eval = next_eval;
                cur_mk = next_mk;
                if next_mk < prev_mk {
                    improved = true;
                }
                if next_mk < best_seen {
                    best_seen = next_mk;
                    global_improved = true;
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
                patch_eval_machine_state_one_machine(ds, buf, &mut indeg_base, m);
            }
        } else if bc.kind == 2 {
            let m = bc.m_from;
            if m < ds.num_machines && bc.from + 1 < ds.machine_seq[m].len() {
                let before_seq = [bc.kind as usize, bc.from, bc.to];
                ds.machine_seq[m].swap(bc.from, bc.from + 1);
                patch_eval_machine_state_one_machine(ds, buf, &mut indeg_base, m);
                let next_eval = best_trial_eval.unwrap();
                swap_vec_u32(&mut best_trial_start, &mut buf.start);
                swap_vec_usize(&mut best_trial_pred, &mut buf.best_pred);
                let next_mk = next_eval.0;
                if next_mk < prev_mk || next_mk <= rrt_limit {
                    *cur_eval = next_eval;
                    cur_mk = next_mk;
                    if next_mk < prev_mk {
                        improved = true;
                    }
                    if next_mk < best_seen {
                        best_seen = next_mk;
                        global_improved = true;
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
                    patch_eval_machine_state_one_machine(ds, buf, &mut indeg_base, m);
                }
            }
        } else if bc.kind == 3 {
            let m = bc.m_from;
            if m < ds.num_machines {
                let len = ds.machine_seq[m].len();
                if bc.from < len && bc.to < len && bc.from + 1 < bc.to {
                    let before_seq = [bc.kind as usize, bc.from, bc.to];
                    ds.machine_seq[m].swap(bc.from, bc.to);
                    patch_eval_machine_state_one_machine(ds, buf, &mut indeg_base, m);
                    let next_eval = best_trial_eval.unwrap();
                    swap_vec_u32(&mut best_trial_start, &mut buf.start);
                    swap_vec_usize(&mut best_trial_pred, &mut buf.best_pred);
                    let next_mk = next_eval.0;
                    if next_mk < prev_mk || next_mk <= rrt_limit {
                        *cur_eval = next_eval;
                        cur_mk = next_mk;
                        if next_mk < prev_mk {
                            improved = true;
                        }
                        if next_mk < best_seen {
                            best_seen = next_mk;
                            global_improved = true;
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
                        patch_eval_machine_state_one_machine(ds, buf, &mut indeg_base, m);
                    }
                }
            }
        }

        if !accepted {
            aspiration = aspiration.saturating_add(1);
            if aspiration > 5 {
                aspiration = 5;
            }
        } else {
            aspiration = 0;

            if global_improved {
                tenure = 2;
                non_improving_count = 0;
            } else {
                non_improving_count += 1;
                tenure = (2 + non_improving_count).min(12);
            }

            if recent_fill < recent_makespans.len() {
                recent_makespans[recent_fill] = cur_mk;
                recent_fill += 1;
            } else {
                recent_makespans[recent_idx] = cur_mk;
                recent_idx = (recent_idx + 1) % recent_makespans.len();
            }
        }
    }

    improved
}

#[inline]
fn apply_insert_fs(seq: &mut Vec<usize>, from: usize, to_after_removal: usize) -> usize {
    let len = seq.len();
    if len == 0 || from >= len {
        return from.min(len.saturating_sub(1));
    }
    let t = to_after_removal.min(len - 1);
    if t < from {
        seq[t..=from].rotate_right(1);
    } else if t > from {
        seq[from..=t].rotate_left(1);
    }
    t
}

#[inline]
fn push_top_k_move_fs(top: &mut Vec<MoveCand>, c: MoveCand, k: usize) {
    if k == 0 {
        return;
    }
    let len = top.len();
    if len == k && top[len - 1].score >= c.score {
        return;
    }
    let mut pos = len;
    while pos > 0 && top[pos - 1].score < c.score {
        pos -= 1;
    }
    if pos >= k {
        return;
    }
    if len < k {
        top.reserve(1);
        unsafe {
            let ptr = top.as_mut_ptr();
            std::ptr::copy(ptr.add(pos), ptr.add(pos + 1), len - pos);
            std::ptr::write(ptr.add(pos), c);
            top.set_len(len + 1);
        }
    } else {
        unsafe {
            let ptr = top.as_mut_ptr();
            std::ptr::drop_in_place(ptr.add(len - 1));
            std::ptr::copy(ptr.add(pos), ptr.add(pos + 1), len - pos - 1);
            std::ptr::write(ptr.add(pos), c);
        }
    }
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

fn neh_build_seq_taillard(
    order: &[usize],
    pt: &[Vec<u32>],
    m: usize,
    buf: &mut TaillardInsBuf,
) -> Vec<usize> {
    if m == 0 {
        return vec![];
    }
    let mut seq: Vec<usize> = Vec::with_capacity(order.len());
    for &j in order {
        if seq.is_empty() {
            seq.push(j);
            continue;
        }
        let (pos, _mk) = taillard_best_insert_pos(&seq, j, pt, m, buf);
        seq.insert(pos, j);
    }
    seq
}

fn neh_build_seq(order: &[usize], route: &[usize], pt: &[Vec<u32>], num_machines: usize) -> Vec<usize> {
    let unique = route_is_unique(route, num_machines);
    if unique {
        let m = route.len();
        return TL_TAILLARD.with(|cell| {
            let mut buf = cell.borrow_mut();
            neh_build_seq_taillard(order, pt, m, &mut buf)
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
                let mut seq = neh_build_seq_taillard(ord, pt, m, &mut buf);
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

#[inline]
fn flowshop_adjacent_swap_polish(
    seq: &mut [usize],
    pt: &[Vec<u32>],
    cur_mk: u32,
    buf: &mut TaillardInsBuf,
) -> u32 {
    let n = seq.len();
    let m = pt.first().map(|r| r.len()).unwrap_or(0);
    if n <= 1 || m == 0 {
        return cur_mk;
    }

    buf.ensure(n, m);
    let tail = &mut buf.b;
    let base_n = n * m;
    for k in 0..m {
        tail[base_n + k] = 0;
    }
    for t in (0..n).rev() {
        let row = &pt[seq[t]];
        let base = t * m;
        let next = (t + 1) * m;
        tail[base + (m - 1)] = tail[next + (m - 1)].saturating_add(row[m - 1]);
        if m >= 2 {
            for kk in 0..(m - 1) {
                let k = (m - 2) - kk;
                tail[base + k] = tail[base + k + 1]
                    .max(tail[next + k])
                    .saturating_add(row[k]);
            }
        }
    }

    let prefix = &mut buf.comp[..m];
    let first_after = &mut buf.e[..m];
    let swap_after = &mut buf.f[..m];
    prefix.fill(0);
    let mut best = cur_mk;

    for i in 0..(n - 1) {
        let ja = seq[i];
        let jb = seq[i + 1];

        let row_b = &pt[jb];
        first_after[0] = prefix[0].saturating_add(row_b[0]);
        for k in 1..m {
            first_after[k] = first_after[k - 1]
                .max(prefix[k])
                .saturating_add(row_b[k]);
        }

        let row_a = &pt[ja];
        swap_after[0] = first_after[0].saturating_add(row_a[0]);
        for k in 1..m {
            swap_after[k] = swap_after[k - 1]
                .max(first_after[k])
                .saturating_add(row_a[k]);
        }

        let base = (i + 2) * m;
        let mut mk2 = 0u32;
        for k in 0..m {
            mk2 = mk2.max(swap_after[k].saturating_add(tail[base + k]));
        }

        if mk2 <= best {
            seq.swap(i, i + 1);
            best = mk2;
            prefix.copy_from_slice(first_after);
        } else {
            prefix[0] = prefix[0].saturating_add(row_a[0]);
            for k in 1..m {
                let v = prefix[k].max(prefix[k - 1]).saturating_add(row_a[k]);
                prefix[k] = v;
            }
        }
    }

    best
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
    let mut partial: Vec<usize> = Vec::with_capacity(n);
    let mut remove_mark: Vec<u32> = vec![0u32; n];
    let mut mark_epoch: u32 = 1;
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
        partial.clear();
        if mark_epoch == u32::MAX {
            remove_mark.fill(0);
            mark_epoch = 1;
        }
        let epoch = mark_epoch;
        mark_epoch += 1;
        for &ix in &idxs {
            remove_mark[ix] = epoch;
        }
        for &ix in idxs.iter().rev() {
            removed.push(cur[ix]);
        }
        for (pos, &job) in cur.iter().enumerate() {
            if remove_mark[pos] != epoch {
                partial.push(job);
            }
        }
        removed.shuffle(rng);
        for &j in &removed {
            let (pos, _mk) = taillard_best_insert_pos(&partial, j, pt, m, &mut buf);
            partial.insert(pos, j);
        }
        let mut cand_mk = flow_makespan(&partial, pt, &mut buf.comp[..m]);
        if partial.len() >= 2 {
            cand_mk = flowshop_adjacent_swap_polish(&mut partial, pt, cand_mk, &mut buf);
        }
        if cand_mk < best_mk {
            best_mk = cand_mk;
            best.clear();
            best.extend_from_slice(&partial);
        }
        if cand_mk <= cur_mk {
            cur.clear();
            cur.extend_from_slice(&partial);
            cur_mk = cand_mk;
        } else {
            let delta = (cand_mk - cur_mk) as f64;
            let prob = (-delta / temp).exp();
            if rng.gen::<f64>() < prob {
                cur.clear();
                cur.extend_from_slice(&partial);
                cur_mk = cand_mk;
            }
        }
        temp = (temp * 0.995).max(1.0);
    }
    best
}

fn strict_run<const BUILD_SOL: bool>(
    challenge: &Challenge,
    pre: &Pre,
    rank: &[usize],
) -> Result<(u32, Vec<Vec<(usize, u32)>>)> {
    let route = pre.flow_route.as_ref().ok_or_else(|| anyhow!("flow_route missing"))?;
    let pt_by_job = pre.flow_pt_by_job.as_ref();
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;
    let mut job_next_op = vec![0usize; num_jobs];
    let mut job_ready = vec![0u32; num_jobs];
    let mut machine_avail = vec![0u32; num_machines];
    let mut job_schedule: Vec<Vec<(usize, u32)>> = if BUILD_SOL {
        pre.job_ops_len
            .iter()
            .map(|&len| Vec::with_capacity(len))
            .collect()
    } else {
        Vec::new()
    };
    let mut remaining_ops = pre.total_ops;
    let mut future: Vec<BinaryHeap<Reverse<(u32, usize, usize)>>> =
        (0..num_machines).map(|_| BinaryHeap::new()).collect();
    let mut avail: Vec<BinaryHeap<Reverse<(usize, usize)>>> =
        (0..num_machines).map(|_| BinaryHeap::new()).collect();
    let mut initially_available: Vec<Reverse<(usize, usize)>> =
        Vec::with_capacity(num_jobs);
    for job in 0..num_jobs {
        if pre.job_ops_len[job] > 0 {
            initially_available.push(Reverse((rank[job], job)));
        }
    }
    if !initially_available.is_empty() {
        let m = route[0];
        avail[m] = BinaryHeap::from(initially_available);
    }
    let mut next_time: Vec<Option<u32>> = vec![None; num_machines];
    let mut machine_events: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();
    let compute_next_time = |m: usize,
                             machine_avail: &[u32],
                             future: &[BinaryHeap<Reverse<(u32, usize, usize)>>],
                             avail: &[BinaryHeap<Reverse<(usize, usize)>>]|
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
            return Err(anyhow!(if BUILD_SOL {
                "job complete"
            } else {
                "job complete but popped"
            }));
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
        let ptv = if let Some(v) = pt_by_job
            .and_then(|pt| pt.get(job))
            .and_then(|row| row.get(op_idx))
            .copied()
        {
            v
        } else {
            let product = pre.job_products[job];
            *challenge.product_processing_times[product][op_idx]
                .get(&m)
                .ok_or_else(|| anyhow!("missing pt"))?
        };
        let end = start.saturating_add(ptv);
        if BUILD_SOL {
            job_schedule[job].push((m, start));
        }
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
    Ok((makespan, job_schedule))
}

fn strict_makespan(challenge: &Challenge, pre: &Pre, rank: &[usize]) -> Result<u32> {
    strict_run::<false>(challenge, pre, rank).map(|x| x.0)
}

fn strict_simulate(challenge: &Challenge, pre: &Pre, rank: &[usize]) -> Result<(Solution, u32)> {
    strict_run::<true>(challenge, pre, rank)
        .map(|(makespan, job_schedule)| (Solution { job_schedule }, makespan))
}

#[inline]
fn order_fingerprint(order: &[usize]) -> u64 {
    let mut h = 1469598103934665603u64 ^ (order.len() as u64);
    for &j in order {
        h ^= (j as u64).wrapping_add(1);
        h = h.wrapping_mul(1099511628211u64);
    }
    h
}

#[inline]
fn strict_makespan_cached(
    challenge: &Challenge,
    pre: &Pre,
    order: &[usize],
    rank: &mut [usize],
    eval_cache: &mut std::collections::HashMap<u64, (Vec<usize>, u32)>,
    cache_cap: usize,
) -> Result<u32> {
    let key = order_fingerprint(order);
    if let Some((cached_order, mk)) = eval_cache.get(&key) {
        if cached_order.as_slice() == order {
            return Ok(*mk);
        }
    }
    for (pos, &j) in order.iter().enumerate() {
        rank[j] = pos;
    }
    let mk = strict_makespan(challenge, pre, rank)?;
    if cache_cap > 0 {
        if !eval_cache.contains_key(&key) && eval_cache.len() >= cache_cap {
            eval_cache.clear();
        }
        eval_cache.insert(key, (order.to_vec(), mk));
    }
    Ok(mk)
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
    let cache_cap = n.saturating_mul(cand_orders.len().max(1)).max(1);
    let mut eval_cache: std::collections::HashMap<u64, (Vec<usize>, u32)> =
        std::collections::HashMap::with_capacity(cand_orders.len().max(1));
    let mut rank = vec![0usize; n];
    let mut best_mk = u32::MAX;
    let mut best_order: Vec<usize> = (0..n).collect();
    for ord in cand_orders.iter() {
        if ord.len() != n {
            continue;
        }
        let mk = strict_makespan_cached(
            challenge,
            pre,
            ord,
            &mut rank,
            &mut eval_cache,
            cache_cap,
        )?;
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
                let mk = strict_makespan_cached(
                    challenge,
                    pre,
                    &cand_order,
                    &mut rank,
                    &mut eval_cache,
                    cache_cap,
                )?;
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

                    let mk = strict_makespan_cached(
                        challenge,
                        pre,
                        &cand_order,
                        &mut rank,
                        &mut eval_cache,
                        cache_cap,
                    )?;
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
        let swap_budget = (n * 12).clamp(200, 800);
        for _ in 0..swap_budget {
            let i = rng.gen_range(0..n);
            let j = rng.gen_range(0..n);
            if i == j {
                continue;
            }
            order.swap(i, j);
            rank[order[i]] = i;
            rank[order[j]] = j;
            let mk = strict_makespan_cached(
                challenge,
                pre,
                &order,
                &mut rank,
                &mut eval_cache,
                cache_cap,
            )?;
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
                    let mk = strict_makespan_cached(
                        challenge,
                        pre,
                        &order,
                        &mut rank,
                        &mut eval_cache,
                        cache_cap,
                    )?;
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

fn job_bias_from_eval(
    ds: &DisjSchedule,
    buf: &EvalBuf,
    mk_node: usize,
    num_jobs: usize,
) -> Vec<f64> {
    let mut job_bias = vec![0.0f64; num_jobs];
    let mut u = mk_node;
    while u != NONE_USIZE {
        let job = ds.node_job[u];
        job_bias[job] += ds.node_pt[u] as f64;
        u = buf.best_pred[u];
    }
    let max_bias = job_bias.iter().cloned().fold(0.0f64, f64::max);
    if max_bias > 0.0 {
        let scale = 5.0 / max_bias;
        for b in &mut job_bias {
            *b *= scale;
        }
    }
    job_bias
}

fn job_bias_from_solution(pre: &Pre, challenge: &Challenge, sol: &Solution, _mk: u32) -> Option<Vec<f64>> {
    let ds = build_disj_from_solution(pre, challenge, sol).ok()?;
    let mut buf = EvalBuf::new(ds.n);
    let (_, mk_node) = eval_disj(&ds, &mut buf)?;
    Some(job_bias_from_eval(&ds, &buf, mk_node, challenge.num_jobs))
}

fn validate_solution_strict(challenge: &Challenge, sol: &Solution) -> Result<u32> {
    if sol.job_schedule.len() != challenge.num_jobs {
        return Err(anyhow!("invalid job count"));
    }

    let mut job_products = Vec::with_capacity(challenge.num_jobs);
    for (p, &cnt) in challenge.jobs_per_product.iter().enumerate() {
        for _ in 0..cnt {
            job_products.push(p);
        }
    }
    if job_products.len() != challenge.num_jobs {
        return Err(anyhow!("invalid product expansion"));
    }

    let mut machine_intervals: Vec<Vec<(u32, u32)>> =
        (0..challenge.num_machines).map(|_| Vec::new()).collect();
    let mut makespan = 0u32;

    for job in 0..challenge.num_jobs {
        let product = job_products[job];
        let ops = &challenge.product_processing_times[product];
        let sched = &sol.job_schedule[job];
        if sched.len() != ops.len() {
            return Err(anyhow!("invalid operation count"));
        }

        let mut job_ready = 0u32;
        for (op_idx, &(machine, start)) in sched.iter().enumerate() {
            if machine >= challenge.num_machines {
                return Err(anyhow!("invalid machine"));
            }
            if start < job_ready {
                return Err(anyhow!("job precedence violation"));
            }
            let pt = *ops[op_idx]
                .get(&machine)
                .ok_or_else(|| anyhow!("operation cannot run on machine"))?;
            let end = start.saturating_add(pt);
            if end < start {
                return Err(anyhow!("time overflow"));
            }
            job_ready = end;
            makespan = makespan.max(end);
            machine_intervals[machine].push((start, end));
        }
    }

    for intervals in &mut machine_intervals {
        intervals.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        for i in 1..intervals.len() {
            if intervals[i].0 < intervals[i - 1].1 {
                return Err(anyhow!("machine overlap"));
            }
        }
    }

    Ok(makespan)
}

fn commit_valid_best(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    best_sol: &mut Solution,
    best_mk: &mut u32,
    sol: Solution,
    candidate_mk: u32,
    allow_equal: bool,
) -> Result<bool> {
    let is_candidate_better = if allow_equal {
        candidate_mk <= *best_mk
    } else {
        candidate_mk < *best_mk
    };
    if !is_candidate_better {
        return Ok(false);
    }

    let Ok(strict_mk) = validate_solution_strict(challenge, &sol) else {
        return Ok(false);
    };
    let Ok(checked_mk) = challenge.evaluate_makespan(&sol) else {
        return Ok(false);
    };
    if checked_mk != strict_mk {
        return Ok(false);
    }
    let is_checked_better = if allow_equal {
        checked_mk <= *best_mk
    } else {
        checked_mk < *best_mk
    };
    if !is_checked_better {
        return Ok(false);
    }

    *best_mk = checked_mk;
    *best_sol = sol;
    save_solution(best_sol)?;
    Ok(true)
}

fn build_perturbation_base(
    pre: &Pre,
    challenge: &Challenge,
    sol: &Solution,
) -> Option<(DisjSchedule, Vec<bool>, Vec<usize>)> {
    let ds = build_disj_from_solution(pre, challenge, sol).ok()?;
    let mut buf = EvalBuf::new(ds.n);
    let (_, mk_node) = eval_disj(&ds, &mut buf)?;
    let mut crit = vec![false; ds.n];
    let mut u = mk_node;
    while u != NONE_USIZE {
        crit[u] = true;
        u = buf.best_pred[u];
    }
    let mut non_crit_machines = Vec::new();
    for m in 0..ds.num_machines {
        let seq = &ds.machine_seq[m];
        if seq.len() > 1 && seq.iter().all(|&id| !crit[id]) {
            non_crit_machines.push(m);
        }
    }
    Some((ds, crit, non_crit_machines))
}

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    pre: &Pre,
    _effort: &EffortConfig,
) -> Result<()> {
    let (greedy_sol, _) = super::infra_shared::run_simple_greedy_baseline(challenge)?;
    let greedy_mk = validate_solution_strict(challenge, &greedy_sol)?;
    let checked_greedy_mk = challenge.evaluate_makespan(&greedy_sol)?;
    if checked_greedy_mk != greedy_mk {
        return Err(anyhow!("greedy validation mismatch"));
    }
    save_solution(&greedy_sol)?;

    let mut best_sol = greedy_sol;
    let mut best_mk = greedy_mk;
    let mut top_solutions: Vec<(Solution, u32)> = Vec::new();
    push_top_solutions(&mut top_solutions, &best_sol, best_mk, 5);

    let run_strict_search = pre.flow_route.is_some() && pre.flex_avg <= 1.25;
    let run_neh_search = pre.flow_route.is_some() && pre.flow_pt_by_job.is_some();
    let strict_search_result = if run_strict_search {
        strict_best_by_order_search(challenge, pre, 6).ok()
    } else {
        None
    };
    let neh_search_result = if run_neh_search {
        neh_best_sequence(pre, challenge.num_jobs, challenge.num_machines).ok()
    } else {
        None
    };

    let mut strict_sol: Option<(Solution, u32)> = None;
    if pre.flow_route.is_some() && pre.flex_avg <= 1.25 {
        if let Some((sol, mk)) = strict_search_result {
            if let Ok(checked_mk) = challenge.evaluate_makespan(&sol) {
                strict_sol = Some((sol.clone(), checked_mk));
                if commit_valid_best(challenge, save_solution, &mut best_sol, &mut best_mk, sol, mk, true)? {
                    push_top_solutions(&mut top_solutions, &best_sol, best_mk, 5);
                }
            }
        }
    }

    if let (Some(route), Some(pt)) = (&pre.flow_route, &pre.flow_pt_by_job) {
        if let Some(neh_seq) = neh_search_result {
            let perm_sol = build_perm_solution_from_seq(
                &neh_seq,
                route,
                pt,
                challenge.num_jobs,
                challenge.num_machines,
            );
            if let Ok(mk) = challenge.evaluate_makespan(&perm_sol) {
                commit_valid_best(challenge, save_solution, &mut best_sol, &mut best_mk, perm_sol.clone(), mk, true)?;
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
                        commit_valid_best(challenge, save_solution, &mut best_sol, &mut best_mk, ssol.clone(), mk, true)?;
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
                    commit_valid_best(challenge, save_solution, &mut best_sol, &mut best_mk, ig_perm_sol.clone(), mk, true)?;
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
                commit_valid_best(challenge, save_solution, &mut best_sol, &mut best_mk, sol.clone(), mk, true)?;
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
        let mut adaptive_boost = AdaptiveBoost::new(pre);
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
        let mut job_bias: Option<Vec<f64>> = None;

        let num_restarts = 600usize;
        for r in 0..num_restarts {
            let do_test = (r % 45 == 0) && r > 0;
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
            let mut candidate_bias: Option<Vec<f64>> = None;
            if let Ok((mut sol, mut mk)) = construct_solution_conflict(
                challenge,
                pre,
                rule,
                k,
                Some(best_mk.saturating_add(best_mk / 20)),
                &mut rng,
                &mut adaptive_boost,
                job_bias.as_deref(),
                None,
                None,
                0.0,
            ) {
                if mk <= best_mk.saturating_add(best_mk / 20) {
                    if let Ok(mut ds) = build_disj_from_solution(pre, challenge, &sol) {
                        let mut buf = EvalBuf::new(ds.n);
                        if let Some((initial_mk, mk_node)) = eval_disj(&ds, &mut buf) {
                            candidate_bias = Some(job_bias_from_eval(
                                &ds,
                                &buf,
                                mk_node,
                                challenge.num_jobs,
                            ));
                            let mut crit = vec![false; ds.n];
                            let mut cur_eval = (initial_mk, mk_node);
                            let improved = descent_phase(&mut ds, &mut buf, &mut crit, pre, &mut cur_eval, 1, 20);
                            if improved {
                                let (new_mk, new_mk_node) = cur_eval;
                                if new_mk < mk {
                                    if let Ok(polished_sol) = disj_to_solution(pre, &ds, &buf.start) {
                                        candidate_bias = Some(job_bias_from_eval(
                                            &ds,
                                            &buf,
                                            new_mk_node,
                                            challenge.num_jobs,
                                        ));
                                        sol = polished_sol;
                                        mk = new_mk;
                                    }
                                }
                            }
                        }
                    }
                }
                if mk < prev_best {
                    improves[ridx] = improves[ridx].saturating_add(1);
                    delta_sum[ridx] = delta_sum[ridx].saturating_add((prev_best - mk) as u64);
                }

                if let Ok(checked_mk) = challenge.evaluate_makespan(&sol) {
                    commit_valid_best(challenge, save_solution, &mut best_sol, &mut best_mk, sol.clone(), mk, false)?;
                    push_top_solutions(&mut top_solutions, &sol, checked_mk, 15);
                    mk = checked_mk;
                } else {
                    continue;
                }

                if mk <= best_mk.saturating_mul(11) / 10 {
                    if let Some(bias) = candidate_bias
                        .take()
                        .or_else(|| job_bias_from_solution(pre, challenge, &sol, mk))
                    {
                        job_bias = Some(bias);
                    }
                }

                if do_test {
                    let saved_strength = adaptive_boost.boost_strength;
                    adaptive_boost.boost_strength = 0.0;
                    if let Ok((_, mk_no_boost)) = construct_solution_conflict(
                        challenge,
                        pre,
                        rule,
                        k,
                        Some(best_mk.saturating_add(best_mk / 20)),
                        &mut rng,
                        &mut adaptive_boost,
                        job_bias.as_deref(),
                        None,
                        None,
                        0.0,
                    ) {
                        adaptive_boost.boost_strength = saved_strength;
                        adaptive_boost.update_from_test(mk, mk_no_boost);
                    } else {
                        adaptive_boost.boost_strength = saved_strength;
                    }
                }
            }
        }
    }

    let is_strict_perm = route_is_unique(
        pre.flow_route.as_deref().unwrap_or(&[]),
        challenge.num_machines,
    ) && pre.flex_avg <= 1.25;

    if !is_strict_perm {
        let ls_runs = top_solutions.len().min(15);
        let perturb_cycles = ((pre.total_ops / 40) as usize).clamp(10, 40);
        for i in 0..ls_runs {
            let base_sol = &top_solutions[i].0;
            if let Ok(Some((sol2, mk2))) =
                critical_block_move_local_search_ex(pre, challenge, base_sol, 7, 80, perturb_cycles)
            {
                if let Ok(checked_mk) = challenge.evaluate_makespan(&sol2) {
                    commit_valid_best(challenge, save_solution, &mut best_sol, &mut best_mk, sol2.clone(), mk2, false)?;
                    push_top_solutions(&mut top_solutions, &sol2, checked_mk, 15);
                }
            }
        }
        {
            let mut seed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15);
            let mut perturb_base = build_perturbation_base(pre, challenge, &best_sol);
            let mut perturb_work = perturb_base
                .as_ref()
                .map(|(ds, _, _)| (ds.clone(), EvalBuf::new(ds.n)));
            for _ in 0..20 {
                let Some((base_ds, _base_crit, non_crit_machines)) = perturb_base.as_ref() else {
                    continue;
                };
                if non_crit_machines.is_empty() {
                    continue;
                }
                seed ^= seed.wrapping_shl(13);
                seed ^= seed.wrapping_shr(7);
                seed ^= seed.wrapping_shl(17);
                let m_idx = (seed as usize) % non_crit_machines.len();
                let m = non_crit_machines[m_idx];
                let Some((ds, buf)) = perturb_work.as_mut() else { continue };
                ds.machine_seq.clone_from(&base_ds.machine_seq);
                let len = ds.machine_seq[m].len();
                let pos = (seed.wrapping_shl(21) as usize) % (len - 1);
                ds.machine_seq[m].swap(pos, pos + 1);
                let Some((_, _)) = eval_disj(ds, buf) else { continue };
                let Ok(perturbed_sol) = disj_to_solution(pre, ds, &buf.start) else { continue };
                if let Ok(Some((sol2, mk2))) =
                    critical_block_move_local_search_ex(pre, challenge, &perturbed_sol, 5, 80, ((pre.total_ops / 160) as usize).clamp(1, 5))
                {
                    if let Ok(checked_mk) = challenge.evaluate_makespan(&sol2) {
                        if commit_valid_best(challenge, save_solution, &mut best_sol, &mut best_mk, sol2, mk2, false)? {
                            push_top_solutions(&mut top_solutions, &best_sol, checked_mk, 15);
                            perturb_base = build_perturbation_base(pre, challenge, &best_sol);
                            perturb_work = perturb_base
                                .as_ref()
                                .map(|(ds, _, _)| (ds.clone(), EvalBuf::new(ds.n)));
                        }
                    }
                }
            }
        }
    } else {
        let extra_iters = 1600usize;
        if let (Some(route), Some(pt)) = (&pre.flow_route, &pre.flow_pt_by_job) {
            let unique = route_is_unique(route, challenge.num_machines);
            if unique && !pt.is_empty() {
                let mut seed = challenge.seed;
                seed[0] ^= 0xD4;
                let mut rng = SmallRng::from_seed(seed);
                let m = route.len();
                let initial_best_mk = best_mk;
                let mut pending_ig_commits: Vec<(Solution, u32)> = Vec::new();
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

                    let best_ig_seq_start = order_from_solution_first_op_start(&best_sol, challenge.num_jobs);
                    let mut best_ig_seq = best_ig_seq_start.clone();
                    let mut best_ig_mk = best_mk;
                    let mut stagnation = 0usize;
                    let mut perturbation_attempts = 0usize;
                    let perturb_max = 3usize;

                    for &i in &picked {
                        let perturb_mode = stagnation >= 2 && perturbation_attempts < perturb_max && best_ig_seq.len() == challenge.num_jobs;
                        let start_ord = if perturb_mode {
                            let ratio = (initial_best_mk - best_ig_mk) as f64 / initial_best_mk.max(1) as f64;
                            let d_perturb = ((challenge.num_jobs as f64 * (0.08 - 0.05 * ratio)).max(2.0).min(6.0)) as usize;
                            let mut seq = best_ig_seq.clone();
                            let mut indices: Vec<usize> = (0..seq.len()).collect();
                            indices.shuffle(&mut rng);
                            let mut removed = Vec::with_capacity(d_perturb);
                            for &idx in indices.iter().take(d_perturb) {
                                removed.push(seq[idx]);
                            }
                            let mut to_remove: Vec<usize> = indices.iter().take(d_perturb).cloned().collect();
                            to_remove.sort_unstable_by(|a,b| b.cmp(a));
                            for idx in to_remove {
                                seq.remove(idx);
                            }
                            for job in removed {
                                let pos = rng.gen_range(0..=seq.len());
                                seq.insert(pos, job);
                            }
                            seq
                        } else {
                            order_from_solution_first_op_start(&top_solutions[i].0, challenge.num_jobs)
                        };
                        if start_ord.len() != challenge.num_jobs {
                            continue;
                        }
                        let cand_seq = iterated_greedy_search(&start_ord, pt, extra_iters / 5, 4, &mut rng);
                        buf.ensure(cand_seq.len(), m);
                        let mk = flow_makespan(&cand_seq, pt, &mut buf.comp[..m]);
                        if mk < best_ig_mk {
                            best_ig_mk = mk;
                            best_ig_seq = cand_seq.clone();
                            stagnation = 0;
                            if mk < best_mk {
                                let sol = build_perm_solution_from_seq(
                                    &cand_seq,
                                    route,
                                    pt,
                                    challenge.num_jobs,
                                    challenge.num_machines,
                                );
                                pending_ig_commits.push((sol, mk));
                            }
                        } else {
                            stagnation += 1;
                            if perturb_mode {
                                perturbation_attempts += 1;
                                if perturbation_attempts >= perturb_max {
                                    stagnation = 0; 
                                }
                            }
                        }
                    }
                });
                for (sol, mk) in pending_ig_commits {
                    let _ = commit_valid_best(
                        challenge,
                        save_solution,
                        &mut best_sol,
                        &mut best_mk,
                        sol,
                        mk,
                        false,
                    );
                }
            }
        }
    }

    Ok(())
}