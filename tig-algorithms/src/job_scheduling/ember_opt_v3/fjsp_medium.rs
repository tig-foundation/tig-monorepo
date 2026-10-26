#![allow(dead_code)]
use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng};
use std::collections::HashMap;
use tig_challenges::job_scheduling::*;
use super::types::*;
use super::infra_shared::*;

extern "C" {
    static __fuel_remaining: u64;
}

#[inline]
fn fuel_left() -> u64 {
    unsafe { core::ptr::read_volatile(&__fuel_remaining) }
}

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
    let r = rp[product][op_idx]; let mu = machine as u16;
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
    let tie_gate = if op.flex < 2 {
        0.0
    } else if best_cnt_total >= 2 {
        1.0
    } else if second_end < INF {
        let window = (0.30 * pre.avg_op_min + 0.20 * pt as f64).max(1.0);
        (1.0 - (second_end.saturating_sub(best_end) as f64) / window).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let bottleneck_risk = (
        0.36 * (dynamic_load / pre.avg_machine_load.max(1e-9)).clamp(0.0, 2.0) * 0.5
        + 0.34 * (pre.machine_scarcity[machine] / pre.avg_machine_scarcity.max(1e-9)).clamp(0.0, 2.0) * 0.5
        + 0.30 * mpen
    ).clamp(0.0, 1.0);
    let tie_risk_penalty = if tie_gate > 0.0 {
        let saturated_risk = bottleneck_risk * bottleneck_risk / (0.30 + bottleneck_risk);
        tie_gate * saturated_risk * (0.075 + 0.055 * pre.jobshopness + 0.035 * pre.high_flex)
    } else {
        0.0
    };
    let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop=pre.machine_best_pop[machine]; (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 } + tie_risk_penalty;
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
    let mut weights = [0.0f64; 9];
    for (i, &r) in rules.iter().enumerate() {
        let mk=rule_best[rule_idx(r)]; let t=rule_tries[rule_idx(r)].max(1) as f64;
        let delta=mk.saturating_sub(best_seen) as f64; let exploit=(-delta/scale).exp(); let explore=(1.0/t).sqrt();
        let mut ww=(1.0-explore_mix)*exploit+explore_mix*explore; ww=ww.max(1e-6);
        if chaos_like{ww=ww.powf(0.70);}else if late_phase{ww=ww.powf(1.18);}
        weights[i]=ww.max(0.0);
        sum+=weights[i];
    }
    if !(sum>0.0) { return rules[rng.gen_range(0..rules.len())]; }
    let mut r=rng.gen::<f64>()*sum;
    for (i, &rule) in rules.iter().enumerate() {
        r-=weights[i];
        if r<=0.0 { return rule; }
    }
    rules[rules.len()-1]
}

fn contention_choice_fm(pre: &Pre, candidates: &[Cand], job_next_op: &[usize]) -> usize {
    let cap = candidates.len().min(12);
    if cap <= 1 || (pre.high_flex <= 0.25 && pre.jobshopness <= 0.35) {
        return 0;
    }

    let mut best_score = candidates[0].score;
    for cand in candidates.iter().take(cap).skip(1) {
        if cand.score > best_score {
            best_score = cand.score;
        }
    }
    let score_band = 0.045 + 0.035 * best_score.abs();
    let mut chosen = 0usize;
    let mut chosen_pressure = 0usize;
    let mut chosen_score = f64::NEG_INFINITY;

    for i in 0..cap {
        let cand = candidates[i];
        if best_score - cand.score > score_band {
            continue;
        }
        let op_idx = job_next_op[cand.job];
        if op_idx >= pre.job_ops_len[cand.job] {
            continue;
        }
        let product = pre.job_products[cand.job];
        let op = &pre.product_ops[product][op_idx];
        let mut pressure = 0usize;

        for other in candidates.iter().take(cap) {
            if other.job == cand.job {
                continue;
            }
            let other_op_idx = job_next_op[other.job];
            if other_op_idx >= pre.job_ops_len[other.job] {
                continue;
            }
            let other_product = pre.job_products[other.job];
            let other_op = &pre.product_ops[other_product][other_op_idx];
            let mut overlap = 0usize;
            for &(machine, _) in &op.machines {
                if other_op.machines.iter().any(|&(other_machine, _)| other_machine == machine) {
                    overlap += 1;
                }
            }
            if overlap == 0 {
                continue;
            }
            let other_flex = other_op.machines.len().max(1);
            pressure = pressure.saturating_add(overlap.saturating_mul(24) / other_flex);
            if other_op.machines.iter().any(|&(machine, _)| machine == cand.machine) {
                pressure = pressure.saturating_add(6);
            }
        }
        pressure = pressure.saturating_mul(8) / op.machines.len().max(1);

        if pressure > chosen_pressure
            || (pressure == chosen_pressure
                && (cand.score > chosen_score
                    || (cand.score == chosen_score
                        && (cand.job < candidates[chosen].job
                            || (cand.job == candidates[chosen].job
                                && cand.machine < candidates[chosen].machine)))))
        {
            chosen = i;
            chosen_pressure = pressure;
            chosen_score = cand.score;
        }
    }
    chosen
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
    let mut idle_count = num_machines;

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
            if idle_count == 0 || ready_jobs.is_empty() {
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

            let mut top: Vec<Cand> = Vec::with_capacity(12);

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
                    push_top_k(&mut top, c, 12);
                }
            }

            if top.is_empty() {
                break;
            }
            let chosen = top[contention_choice_fm(pre, &top, &job_next_op)];

            let job = chosen.job;
            let machine = chosen.machine;
            let pt = chosen.pt;

            let product = pre.job_products[job];
            let op_idx = job_next_op[job];
            let op = &pre.product_ops[product][op_idx];

            if let Ok(pos) = ready_jobs.binary_search(&job) {
                ready_jobs.remove(pos);
            }

            let end_time = time.saturating_add(pt);
            job_schedule[job].push((machine, time));
            job_next_op[job] += 1;
            job_ready_time[job] = end_time;
            machine_avail[machine] = end_time;
            if end_time > time {
                idle_count = idle_count.saturating_sub(1);
            }
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
        idle_count = machine_avail.iter().filter(|&&t| t <= time).count();

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
    let mut idle_count = num_machines;

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
            if idle_count == 0 || ready_jobs.is_empty() {
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
            let sampled = choose_from_top_weighted(rng, &top);
            let contention_idx = contention_choice_fm(pre, &top, &job_next_op);
            let contention = top[contention_idx];
            let sampled_gap = (top[0].score - sampled.score).max(0.0);
            let contention_gap = (top[0].score - contention.score).max(0.0);
            let score_band = 0.045 + 0.035 * top[0].score.abs();
            let chosen = if contention_gap <= score_band
                && sampled_gap <= score_band
                && rng.gen::<f64>() < 0.60
            {
                contention
            } else {
                sampled
            };

            let job = chosen.job;
            let machine = chosen.machine;
            let pt = chosen.pt;

            let product = pre.job_products[job];
            let op_idx = job_next_op[job];
            let op = &pre.product_ops[product][op_idx];

            if let Ok(pos) = ready_jobs.binary_search(&job) {
                ready_jobs.remove(pos);
            }

            let end_time = time.saturating_add(pt);
            job_schedule[job].push((machine, time));
            job_next_op[job] += 1;
            job_ready_time[job] = end_time;
            machine_avail[machine] = end_time;
            if end_time > time {
                idle_count = idle_count.saturating_sub(1);
            }
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
        idle_count = machine_avail.iter().filter(|&&t| t <= time).count();

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
    let mut pos_of_node = vec![0usize; ds.n];

    for seq in &ds.machine_seq {
        for (pos, &nd) in seq.iter().enumerate() {
            pos_of_node[nd] = pos;
        }
    }

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
        'node_loop: for &node in crit_nodes.iter().take(5) {
            let job = ds.node_job[node]; let op_idx = ds.node_op[node]; let product = pre.job_products[job];
            let op_info = &pre.product_ops[product][op_idx];
            if op_info.machines.len() <= 1 { continue; }
            let cur_machine = ds.node_machine[node]; let cur_pt = ds.node_pt[node];
            let old_pos0 = pos_of_node[node];
            if old_pos0 >= ds.machine_seq[cur_machine].len() || ds.machine_seq[cur_machine][old_pos0] != node { continue; }
            let mut best_mk = current_mk; let mut best_m = cur_machine; let mut best_pt = cur_pt; let mut best_pos = 0usize;
            for &(new_m, new_pt) in &op_info.machines {
                if new_m == cur_machine { continue; }
                let old_pos = pos_of_node[node];
                if old_pos >= ds.machine_seq[cur_machine].len() || ds.machine_seq[cur_machine][old_pos] != node { continue; }
                ds.machine_seq[cur_machine].remove(old_pos);
                for idx in old_pos..ds.machine_seq[cur_machine].len() {
                    let nd = ds.machine_seq[cur_machine][idx];
                    pos_of_node[nd] = idx;
                }
                ds.node_machine[node] = new_m; ds.node_pt[node] = new_pt;
                let target_len = ds.machine_seq[new_m].len();
                for pos in 0..=target_len {
                    ds.machine_seq[new_m].insert(pos, node);
                    for idx in pos..ds.machine_seq[new_m].len() {
                        let nd = ds.machine_seq[new_m][idx];
                        pos_of_node[nd] = idx;
                    }
                    if let Some((test_mk, _)) = eval_disj(&ds, &mut buf) {
                        if test_mk < best_mk { best_mk = test_mk; best_m = new_m; best_pt = new_pt; best_pos = pos; }
                    }
                    ds.machine_seq[new_m].remove(pos);
                    for idx in pos..ds.machine_seq[new_m].len() {
                        let nd = ds.machine_seq[new_m][idx];
                        pos_of_node[nd] = idx;
                    }
                    buf_matches_current = false;
                }
                ds.machine_seq[cur_machine].insert(old_pos, node);
                for idx in old_pos..ds.machine_seq[cur_machine].len() {
                    let nd = ds.machine_seq[cur_machine][idx];
                    pos_of_node[nd] = idx;
                }
                ds.node_machine[node] = cur_machine; ds.node_pt[node] = cur_pt;
            }
            if best_m != cur_machine {
                let old_pos = pos_of_node[node];
                if old_pos >= ds.machine_seq[cur_machine].len() || ds.machine_seq[cur_machine][old_pos] != node { continue; }
                ds.machine_seq[cur_machine].remove(old_pos);
                for idx in old_pos..ds.machine_seq[cur_machine].len() {
                    let nd = ds.machine_seq[cur_machine][idx];
                    pos_of_node[nd] = idx;
                }
                let ins = best_pos.min(ds.machine_seq[best_m].len());
                ds.machine_seq[best_m].insert(ins, node);
                for idx in ins..ds.machine_seq[best_m].len() {
                    let nd = ds.machine_seq[best_m][idx];
                    pos_of_node[nd] = idx;
                }
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

fn unified_reassign_pass(pre: &Pre, challenge: &Challenge, base_sol: &Solution) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n);
    let Some((current_mk, mk_node)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let current_start = buf.start.clone();
    let n = ds.n;
    let num_machines = challenge.num_machines;
    if num_machines <= 1 {
        return Ok(None);
    }

    let mut machine_loads = vec![0u64; num_machines];
    for nd in 0..n {
        let m = ds.node_machine[nd];
        machine_loads[m] = machine_loads[m].saturating_add(ds.node_pt[nd] as u64);
    }
    let total_load: u64 = machine_loads.iter().copied().sum();

    let mut crit = vec![false; n];
    let mut crit_rank = vec![usize::MAX; n];
    let mut crit_nodes: Vec<usize> = Vec::with_capacity(128);
    let mut u = mk_node;
    let mut rank = 0usize;
    while u != NONE_USIZE {
        crit[u] = true;
        crit_rank[u] = rank;
        crit_nodes.push(u);
        u = buf.best_pred[u];
        rank += 1;
    }

    let mut crit_count_by_machine = vec![0usize; num_machines];
    let mut crit_pt_by_machine = vec![0u64; num_machines];
    for &nd in &crit_nodes {
        let m = ds.node_machine[nd];
        crit_count_by_machine[m] += 1;
        crit_pt_by_machine[m] = crit_pt_by_machine[m].saturating_add(ds.node_pt[nd] as u64);
    }

    let mut pos_of_node = vec![0usize; n];
    for seq in &ds.machine_seq {
        for (pos, &nd) in seq.iter().enumerate() {
            pos_of_node[nd] = pos;
        }
    }

    let source_limit = if num_machines <= 4 {
        num_machines
    } else if pre.high_flex > 0.55 || pre.jobshopness > 0.45 {
        5usize.min(num_machines)
    } else {
        4usize.min(num_machines)
    };
    let target_limit = if num_machines <= 5 {
        num_machines - 1
    } else if pre.high_flex > 0.55 || pre.jobshopness > 0.45 {
        5usize.min(num_machines - 1)
    } else {
        4usize.min(num_machines - 1)
    };
    let per_source_node_cap = if pre.high_flex > 0.60 || pre.jobshopness > 0.50 { 7usize } else { 5usize };
    let per_pair_node_cap = if pre.high_flex > 0.60 || pre.jobshopness > 0.50 { 4usize } else { 3usize };

    let mut source_order: Vec<usize> = (0..num_machines)
        .filter(|&m| {
            !ds.machine_seq[m].is_empty()
                && (crit_count_by_machine[m] > 0
                    || machine_loads[m].saturating_mul(num_machines as u64) >= total_load)
        })
        .collect();
    if source_order.is_empty() {
        return Ok(None);
    }
    source_order.sort_unstable_by(|&a, &b| {
        let score_a = crit_pt_by_machine[a].saturating_mul(3).saturating_add(machine_loads[a]);
        let score_b = crit_pt_by_machine[b].saturating_mul(3).saturating_add(machine_loads[b]);
        score_b
            .cmp(&score_a)
            .then_with(|| crit_count_by_machine[b].cmp(&crit_count_by_machine[a]))
            .then_with(|| machine_loads[b].cmp(&machine_loads[a]))
            .then_with(|| a.cmp(&b))
    });

    let mut best_global_move: Option<(usize, usize, u32, usize, usize)> = None;
    let mut best_global_mk = current_mk;

    for &source_m in source_order.iter().take(source_limit) {
        let source_seq = ds.machine_seq[source_m].clone();
        if source_seq.is_empty() {
            continue;
        }

        let mut source_nodes: Vec<usize> = Vec::with_capacity(per_source_node_cap * 2);
        for &node in &source_seq {
            let job = ds.node_job[node];
            let op_idx = ds.node_op[node];
            let product = pre.job_products[job];
            if pre.product_ops[product][op_idx].machines.len() <= 1 {
                continue;
            }
            if crit[node] && !source_nodes.contains(&node) {
                source_nodes.push(node);
            }
        }
        let tail_start = source_seq.len().saturating_sub(per_source_node_cap);
        for &node in &source_seq[tail_start..] {
            let job = ds.node_job[node];
            let op_idx = ds.node_op[node];
            let product = pre.job_products[job];
            if pre.product_ops[product][op_idx].machines.len() <= 1 {
                continue;
            }
            if !source_nodes.contains(&node) {
                source_nodes.push(node);
            }
        }
        if source_nodes.is_empty() {
            continue;
        }

        source_nodes.sort_unstable_by(|&a, &b| {
            crit[b]
                .cmp(&crit[a])
                .then_with(|| crit_rank[a].cmp(&crit_rank[b]))
                .then_with(|| ds.node_pt[b].cmp(&ds.node_pt[a]))
                .then_with(|| pos_of_node[b].cmp(&pos_of_node[a]))
                .then_with(|| a.cmp(&b))
        });
        if source_nodes.len() > per_source_node_cap {
            source_nodes.truncate(per_source_node_cap);
        }

        let mut target_buckets: Vec<Vec<(usize, u32)>> = vec![Vec::new(); num_machines];
        for &node in &source_nodes {
            let job = ds.node_job[node];
            let op_idx = ds.node_op[node];
            let product = pre.job_products[job];
            let op_info = &pre.product_ops[product][op_idx];
            for &(new_m, new_pt) in &op_info.machines {
                if new_m != source_m {
                    target_buckets[new_m].push((node, new_pt));
                }
            }
        }

        let mut target_order: Vec<usize> = (0..num_machines)
            .filter(|&m| m != source_m && !target_buckets[m].is_empty())
            .collect();
        if target_order.is_empty() {
            continue;
        }
        target_order.sort_unstable_by(|&a, &b| {
            machine_loads[a]
                .cmp(&machine_loads[b])
                .then_with(|| crit_pt_by_machine[a].cmp(&crit_pt_by_machine[b]))
                .then_with(|| target_buckets[b].len().cmp(&target_buckets[a].len()))
                .then_with(|| a.cmp(&b))
        });

        for &target_m in target_order.iter().take(target_limit) {
            let bucket = &mut target_buckets[target_m];
            bucket.sort_unstable_by(|&(node_a, pt_a), &(node_b, pt_b)| {
                crit[node_b]
                    .cmp(&crit[node_a])
                    .then_with(|| crit_rank[node_a].cmp(&crit_rank[node_b]))
                    .then_with(|| ds.node_pt[node_b].cmp(&ds.node_pt[node_a]))
                    .then_with(|| pt_a.cmp(&pt_b))
                    .then_with(|| node_a.cmp(&node_b))
            });

            for &(node, new_pt) in bucket.iter().take(per_pair_node_cap) {
                let old_pos = pos_of_node[node];
                if old_pos >= ds.machine_seq[source_m].len() || ds.machine_seq[source_m][old_pos] != node {
                    continue;
                }

                let cur_pt = ds.node_pt[node];
                let job = ds.node_job[node];
                let jp = if node > ds.job_offsets[job] { node - 1 } else { NONE_USIZE };
                let jp_end = if jp != NONE_USIZE {
                    current_start[jp].saturating_add(ds.node_pt[jp])
                } else {
                    0u32
                };

                let base_ins = {
                    let seq = &ds.machine_seq[target_m];
                    match seq.binary_search_by(|&nd| current_start[nd].cmp(&jp_end)) {
                        Ok(pos) | Err(pos) => pos,
                    }
                    .min(seq.len())
                };
                let target_len = ds.machine_seq[target_m].len();
                let pos_candidates = [
                    base_ins,
                    base_ins.saturating_sub(1),
                    base_ins.saturating_add(1).min(target_len),
                    target_len,
                ];

                ds.machine_seq[source_m].remove(old_pos);
                for idx in old_pos..ds.machine_seq[source_m].len() {
                    let nd = ds.machine_seq[source_m][idx];
                    pos_of_node[nd] = idx;
                }
                ds.node_machine[node] = target_m;
                ds.node_pt[node] = new_pt;

                let mut tested_positions = [usize::MAX; 4];
                let mut tested_len = 0usize;
                for &cand_ins in &pos_candidates {
                    let ins = cand_ins.min(ds.machine_seq[target_m].len());
                    if tested_positions[..tested_len].contains(&ins) {
                        continue;
                    }
                    tested_positions[tested_len] = ins;
                    tested_len += 1;

                    ds.machine_seq[target_m].insert(ins, node);
                    for idx in ins..ds.machine_seq[target_m].len() {
                        let nd = ds.machine_seq[target_m][idx];
                        pos_of_node[nd] = idx;
                    }

                    if let Some((test_mk, _)) = eval_disj(&ds, &mut buf) {
                        if test_mk < best_global_mk {
                            best_global_mk = test_mk;
                            best_global_move = Some((node, target_m, new_pt, source_m, ins));
                        }
                    }

                    ds.machine_seq[target_m].remove(ins);
                    for idx in ins..ds.machine_seq[target_m].len() {
                        let nd = ds.machine_seq[target_m][idx];
                        pos_of_node[nd] = idx;
                    }
                }

                ds.machine_seq[source_m].insert(old_pos, node);
                for idx in old_pos..ds.machine_seq[source_m].len() {
                    let nd = ds.machine_seq[source_m][idx];
                    pos_of_node[nd] = idx;
                }
                ds.node_machine[node] = source_m;
                ds.node_pt[node] = cur_pt;
            }
        }
    }

    if let Some((node, new_m, new_pt, cur_m, ins)) = best_global_move {
        let old_pos = pos_of_node[node];
        if old_pos >= ds.machine_seq[cur_m].len() || ds.machine_seq[cur_m][old_pos] != node {
            return Ok(None);
        }
        ds.machine_seq[cur_m].remove(old_pos);
        for idx in old_pos..ds.machine_seq[cur_m].len() {
            let nd = ds.machine_seq[cur_m][idx];
            pos_of_node[nd] = idx;
        }
        let ins = ins.min(ds.machine_seq[new_m].len());
        ds.machine_seq[new_m].insert(ins, node);
        for idx in ins..ds.machine_seq[new_m].len() {
            let nd = ds.machine_seq[new_m][idx];
            pos_of_node[nd] = idx;
        }
        ds.node_machine[node] = new_m;
        ds.node_pt[node] = new_pt;

        let Some((final_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
        if final_mk < current_mk {
            let sol = disj_to_solution(pre, &ds, &buf.start)?;
            Ok(Some((sol, final_mk)))
        } else {
            Ok(None)
        }
    } else {
        Ok(None)
    }
}

fn greedy_reassign_pass(pre: &Pre, challenge: &Challenge, base_sol: &Solution) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n);
    let Some((mut current_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let initial_mk = current_mk;
    let n = ds.n;
    let num_machines = challenge.num_machines;
    let max_passes = 3;
    let mut improved = true;
    let mut passes = 0;

    let mut pos_of_node = vec![0usize; n];
    for seq in &ds.machine_seq {
        for (pos, &nd) in seq.iter().enumerate() {
            pos_of_node[nd] = pos;
        }
    }

    while improved && passes < max_passes {
        improved = false;
        passes += 1;

        let Some((mk, _mk_node)) = eval_disj(&ds, &mut buf) else { break };
        current_mk = mk;

        let mut machine_loads = vec![0u64; num_machines];
        for nd in 0..n {
            let m = ds.node_machine[nd];
            machine_loads[m] = machine_loads[m].saturating_add(ds.node_pt[nd] as u64);
        }

        let mut machine_order: Vec<usize> = (0..num_machines).collect();
        machine_order.sort_unstable_by(|&a, &b| machine_loads[b].cmp(&machine_loads[a]));

        let top_machine_limit = if num_machines <= 4 {
            num_machines
        } else if pre.high_flex > 0.55 || pre.jobshopness > 0.45 {
            4usize.min(num_machines)
        } else {
            3usize.min(num_machines)
        };
        let critical_cap = if pre.high_flex > 0.65 || pre.jobshopness > 0.55 { 16usize } else { 10usize };
        let per_machine_cap = if pre.high_flex > 0.55 { 6usize } else { 4usize };

        let mut top_machine = vec![false; num_machines];
        for &m in machine_order.iter().take(top_machine_limit) {
            top_machine[m] = true;
        }

        let corridor_tails = compute_tails_pulsar(&ds, &buf);
        let corridor_tol = ((pre.avg_op_min * (1.0 + pre.jobshopness)).round() as u32).max(1);
        let crit_nodes = critical_corridor_fm(
            &ds,
            &buf,
            &corridor_tails,
            current_mk,
            corridor_tol,
            critical_cap,
        );
        let mut crit = vec![false; n];
        let mut crit_rank = vec![usize::MAX; n];
        for (rank, &node) in crit_nodes.iter().enumerate() {
            crit[node] = true;
            crit_rank[node] = rank;
        }

        let mut candidate_nodes: Vec<usize> = Vec::with_capacity(critical_cap + top_machine_limit * per_machine_cap);
        let mut seen = vec![false; n];
        for &node in &crit_nodes {
            if !seen[node] {
                seen[node] = true;
                candidate_nodes.push(node);
            }
        }
        for &m in machine_order.iter().take(top_machine_limit) {
            let seq = &ds.machine_seq[m];
            if seq.is_empty() { continue; }
            let start = seq.len().saturating_sub(per_machine_cap);
            for &node in &seq[start..] {
                if !seen[node] {
                    seen[node] = true;
                    candidate_nodes.push(node);
                }
            }
        }

        candidate_nodes.sort_unstable_by(|&a, &b| {
            let ma = ds.node_machine[a];
            let mb = ds.node_machine[b];
            let job_a = ds.node_job[a];
            let job_b = ds.node_job[b];
            let op_idx_a = ds.node_op[a];
            let op_idx_b = ds.node_op[b];
            let product_a = pre.job_products[job_a];
            let product_b = pre.job_products[job_b];
            let flex_a = pre.product_ops[product_a][op_idx_a].machines.len();
            let flex_b = pre.product_ops[product_b][op_idx_b].machines.len();

            crit[b]
                .cmp(&crit[a])
                .then_with(|| crit_rank[a].cmp(&crit_rank[b]))
                .then_with(|| machine_loads[mb].cmp(&machine_loads[ma]))
                .then_with(|| flex_b.cmp(&flex_a))
                .then_with(|| ds.node_pt[b].cmp(&ds.node_pt[a]))
                .then_with(|| a.cmp(&b))
        });

        'search_move: for &node in &candidate_nodes {
            let job = ds.node_job[node];
            let op_idx = ds.node_op[node];
            let product = pre.job_products[job];
            let op_info = &pre.product_ops[product][op_idx];
            if op_info.machines.len() <= 1 { continue; }

            let cur_machine = ds.node_machine[node];
            if !crit[node] && !top_machine[cur_machine] { continue; }
            let cur_pt = ds.node_pt[node];
            let old_pos = pos_of_node[node];
            if old_pos >= ds.machine_seq[cur_machine].len() || ds.machine_seq[cur_machine][old_pos] != node { continue; }

            let jp = if node > ds.job_offsets[job] { node - 1 } else { NONE_USIZE };
            let jp_end = if jp != NONE_USIZE { buf.start[jp].saturating_add(ds.node_pt[jp]) } else { 0u32 };

            let mut alt_machines: Vec<(u64, usize, u32)> = op_info.machines.iter()
                .filter(|&&(am, _)| am != cur_machine)
                .map(|&(am, apt)| (machine_loads[am].saturating_add(apt as u64), am, apt))
                .collect();
            if alt_machines.is_empty() { continue; }
            alt_machines.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.2.cmp(&b.2)));

            for &(_, new_m, new_pt) in alt_machines.iter().take(3) {
                let base_ins = {
                    let seq = &ds.machine_seq[new_m];
                    match seq.binary_search_by(|&nd| buf.start[nd].cmp(&jp_end)) {
                        Ok(pos) | Err(pos) => pos,
                    }.min(seq.len())
                };
                let seq_len = ds.machine_seq[new_m].len();
                let pos_candidates = [
                    base_ins,
                    base_ins.saturating_sub(1),
                    base_ins.saturating_add(1).min(seq_len),
                    seq_len,
                ];

                ds.machine_seq[cur_machine].remove(old_pos);
                for idx in old_pos..ds.machine_seq[cur_machine].len() {
                    let nd = ds.machine_seq[cur_machine][idx];
                    pos_of_node[nd] = idx;
                }
                ds.node_machine[node] = new_m;
                ds.node_pt[node] = new_pt;

                let mut tested_positions = [usize::MAX; 4];
                let mut tested_len = 0usize;
                for &cand_ins in &pos_candidates {
                    let ins = cand_ins.min(ds.machine_seq[new_m].len());
                    if tested_positions[..tested_len].contains(&ins) { continue; }
                    tested_positions[tested_len] = ins;
                    tested_len += 1;

                    ds.machine_seq[new_m].insert(ins, node);
                    for idx in ins..ds.machine_seq[new_m].len() {
                        let nd = ds.machine_seq[new_m][idx];
                        pos_of_node[nd] = idx;
                    }

                    if let Some((test_mk, _)) = eval_disj(&ds, &mut buf) {
                        if test_mk < current_mk {
                            current_mk = test_mk;
                            improved = true;
                            break 'search_move;
                        }
                    }

                    ds.machine_seq[new_m].remove(ins);
                    for idx in ins..ds.machine_seq[new_m].len() {
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
    }

    if current_mk < initial_mk {
        let Some((mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
        let sol = disj_to_solution(pre, &ds, &buf.start)?;
        Ok(Some((sol, mk)))
    } else {
        Ok(None)
    }
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
enum MoveType { Swap{machine:usize,pos:usize}, Reassign{node:usize,new_machine:usize,new_pt:u32,insert_pos:usize} }

#[allow(dead_code)]
fn tabu_search_hybrid(pre: &Pre, challenge: &Challenge, base_sol: &Solution, max_iterations: usize, tenure_base: usize) -> Result<Option<(Solution, u32)>> {
    let mut ds=build_disj_from_solution(pre,challenge,base_sol)?; let mut buf=EvalBuf::new(ds.n); let n=ds.n;
    let Some(init_eval)=eval_disj(&ds,&mut buf) else{return Ok(None)};
    let initial_mk=init_eval.0; let mut best_global_mk=initial_mk; let mut best_global_ds=ds.clone();
    let effective_iterations=max_iterations.saturating_sub(max_iterations/10).max(1);
    let tenure=tenure_base.max(5); let tenure_delta=(tenure/3).max(2); let max_no_improve=(effective_iterations/2).max(60);
    let mut tabu_swap: HashMap<(usize,usize),usize>=HashMap::with_capacity(tenure*8);
    let mut tabu_reassign: HashMap<(usize,usize),usize>=HashMap::with_capacity(tenure*4);
    let mut crit=vec![false;n]; let mut no_improve=0usize;
    let mut pseed: u64=(challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15)^(initial_mk as u64).wrapping_shl(16)^(n as u64).wrapping_mul(0x517CC1B727220A95);
    let mut tail=vec![0u32;n]; let mut back_deg=vec![0u16;n]; let mut back_stack: Vec<usize>=Vec::with_capacity(n);
    let mut machine_pred_node=vec![NONE_USIZE;n]; let mut job_pred_node=vec![NONE_USIZE;n]; let mut pos_of_node=vec![0usize;n];
    for j in 0..ds.num_jobs{let base=ds.job_offsets[j];let end=ds.job_offsets[j+1];for k in (base+1)..end{job_pred_node[k]=k-1;}}
    let kick_threshold=(max_no_improve*2/3).max(50); let mut kicks_left=3usize;
    let beam_adm_cap=if pre.high_flex>0.58||pre.jobshopness>0.48{3usize}else{2usize};
    let beam_fallback_cap=1usize;
    let reassign_pos_cap=if pre.high_flex>0.60||pre.jobshopness>0.50{4usize}else{3usize};
    let mut reassign_positions = [0usize; 6];
    for iter in 0..effective_iterations {
        let no_impr_dyn=((max_no_improve as f64)*(1.0-0.4*(iter as f64/effective_iterations as f64).clamp(0.0,1.0))).max(10.0) as usize;
        if no_improve>=no_impr_dyn{if kicks_left==0{break;}ds=best_global_ds.clone();no_improve=0;kicks_left-=1;tabu_swap.clear();tabu_reassign.clear();continue;}
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
        for seq in &ds.machine_seq{for (i,&nd) in seq.iter().enumerate(){pos_of_node[nd]=i;if i>0{machine_pred_node[nd]=seq[i-1];}}}
        tail.fill(0); back_deg.fill(0);
        for i in 0..n{if ds.job_succ[i]!=NONE_USIZE{back_deg[i]+=1;}if buf.machine_succ[i]!=NONE_USIZE{back_deg[i]+=1;}}
        back_stack.clear(); for i in 0..n{if back_deg[i]==0{back_stack.push(i);}}
        while let Some(nd)=back_stack.pop(){let contrib=ds.node_pt[nd].saturating_add(tail[nd]);let jp=job_pred_node[nd];if jp!=NONE_USIZE{if contrib>tail[jp]{tail[jp]=contrib;}back_deg[jp]=back_deg[jp].saturating_sub(1);if back_deg[jp]==0{back_stack.push(jp);}}let mp=machine_pred_node[nd];if mp!=NONE_USIZE{if contrib>tail[mp]{tail[mp]=contrib;}back_deg[mp]=back_deg[mp].saturating_sub(1);if back_deg[mp]==0{back_stack.push(mp);}}}
        crit.fill(false); let mut u=mk_node; while u!=NONE_USIZE{crit[u]=true;u=buf.best_pred[u];}
        let corridor_tol = ((pre.avg_op_min * (1.0 + 1.5 * pre.jobshopness)).round() as u32).max(1);
        let corridor_nodes = critical_corridor_fm(&ds, &buf, &tail, cur_mk, corridor_tol, 48);
        let mut reassign_corridor = vec![false; n];
        for node in corridor_nodes {
            reassign_corridor[node] = true;
        }
        let reassign_freq=if no_improve>(max_no_improve/3).max(15)||pre.high_flex>0.62||pre.jobshopness>0.52{3usize}else{4usize};
        let use_exact_validation=!(iter<10&&no_improve==0&&pre.high_flex<0.45&&pre.jobshopness<0.35);
        let mut admissible_beam: Vec<(u32,MoveType)>=Vec::with_capacity(beam_adm_cap);
        let mut fallback_beam: Vec<(u32,MoveType)>=Vec::with_capacity(beam_fallback_cap);
        let mut swap_beam: Vec<(u32,MoveType)>=Vec::with_capacity(2);
        let mut reassign_beam: Vec<(u32,MoveType)>=Vec::with_capacity(2);
        let push_beam=|beam: &mut Vec<(u32,MoveType)>, cap: usize, est: u32, mv: MoveType| {
            let pos=beam.iter().position(|&(b,_)| est<b).unwrap_or(beam.len());
            if pos<cap{beam.insert(pos,(est,mv));if beam.len()>cap{beam.pop();}}
            else if beam.len()<cap{beam.push((est,mv));}
        };
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
                    let mv=MoveType::Swap{machine:m,pos};
                    let key=(node_u.min(node_v),node_u.max(node_v)); let is_tabu=tabu_swap.get(&key).map_or(false,|&exp|iter<exp); let aspiration=est_mk<best_global_mk;
                    if !is_tabu||aspiration{push_beam(&mut admissible_beam,beam_adm_cap,est_mk,mv);push_beam(&mut swap_beam,2,est_mk,mv);}else{push_beam(&mut fallback_beam,beam_fallback_cap,est_mk,mv);}
                }
            }
        }
        if iter%reassign_freq==0 {
            for node in 0..n {
                if !reassign_corridor[node]{continue;}
                let job=ds.node_job[node]; let op_idx=ds.node_op[node]; let product=pre.job_products[job];
                let op_info=&pre.product_ops[product][op_idx]; if op_info.machines.len()<=1{continue;}
                let cur_machine=ds.node_machine[node];
                for &(new_m,new_pt) in &op_info.machines {
                    if new_m==cur_machine{continue;}
                    let key=(node,new_m); let is_tabu=tabu_reassign.get(&key).map_or(false,|&exp|iter<exp);
                    let positions_len=find_candidate_insert_positions_fm(&ds,&buf.start,node,new_m,new_pt,&job_pred_node,&mut reassign_positions);
                    for &insert_pos in reassign_positions[..positions_len.min(reassign_pos_cap)].iter() {
                        let est_mk=estimate_reassign_mk_fm(&ds,&buf.start,&tail,node,new_m,new_pt,insert_pos,&job_pred_node,&machine_pred_node,&buf.machine_succ);
                        let mv=MoveType::Reassign{node,new_machine:new_m,new_pt,insert_pos};
                        let aspiration=est_mk<best_global_mk;
                        if !is_tabu||aspiration{push_beam(&mut admissible_beam,beam_adm_cap,est_mk,mv);push_beam(&mut reassign_beam,2,est_mk,mv);}else{push_beam(&mut fallback_beam,beam_fallback_cap,est_mk,mv);}
                    }
                }
            }
        }
        let chosen=if use_exact_validation{
            let primary=admissible_beam.first().copied().or_else(||fallback_beam.first().copied());
            let mut best_choice: Option<MoveType>=None;
            let mut best_choice_mk=u32::MAX;
            let mut eval_candidate=|mv: MoveType| -> Option<u32> {
                match mv {
                    MoveType::Swap{machine:m,pos} => {
                        if pos+1>=ds.machine_seq[m].len(){return None;}
                        ds.machine_seq[m].swap(pos,pos+1);
                        let out=eval_disj(&ds,&mut buf).map(|(mk,_)| mk);
                        ds.machine_seq[m].swap(pos,pos+1);
                        out
                    }
                    MoveType::Reassign{node,new_machine,new_pt,insert_pos} => {
                        let old_machine=ds.node_machine[node]; let old_pt=ds.node_pt[node]; let old_pos=pos_of_node[node];
                        if old_pos>=ds.machine_seq[old_machine].len()||ds.machine_seq[old_machine][old_pos]!=node{return None;}
                        ds.machine_seq[old_machine].remove(old_pos);
                        let ins=insert_pos.min(ds.machine_seq[new_machine].len());
                        ds.machine_seq[new_machine].insert(ins,node); ds.node_machine[node]=new_machine; ds.node_pt[node]=new_pt;
                        let out=eval_disj(&ds,&mut buf).map(|(mk,_)| mk);
                        ds.machine_seq[new_machine].remove(ins);
                        ds.machine_seq[old_machine].insert(old_pos,node); ds.node_machine[node]=old_machine; ds.node_pt[node]=old_pt;
                        out
                    }
                }
            };
            if let Some((_,mv))=primary {
                let primary_mk=eval_candidate(mv);
                if let Some(mk)=primary_mk {
                    best_choice=Some(mv);
                    best_choice_mk=mk;
                }
                let stagnating=no_improve>(max_no_improve/6).max(8);
                let need_alternatives=primary_mk.map_or(true,|mk|mk>=cur_mk)||stagnating;
                if need_alternatives {
                    let primary_is_swap=matches!(mv,MoveType::Swap{..});
                    let same_type=if primary_is_swap{swap_beam.iter().skip(1).copied().next()}else{reassign_beam.iter().skip(1).copied().next()};
                    let other_type=if primary_is_swap{reassign_beam.first().copied()}else{swap_beam.first().copied()};
                    for (_,alt) in [same_type,other_type].into_iter().flatten() {
                        if let Some(mk)=eval_candidate(alt) {
                            if mk<best_choice_mk {
                                best_choice_mk=mk;
                                best_choice=Some(alt);
                            }
                        }
                    }
                }
            }
            best_choice.map(|mv|(mv,best_choice_mk))
        }else{
            admissible_beam.first().copied().or_else(||fallback_beam.first().copied()).map(|(mk,mv)|(mv,mk))
        };
        match chosen {
            Some((MoveType::Swap{machine:m,pos},_)) => {
                let node_a=ds.machine_seq[m][pos]; let node_b=ds.machine_seq[m][pos+1]; ds.machine_seq[m].swap(pos,pos+1);
                pseed^=pseed.wrapping_shl(13);pseed^=pseed.wrapping_shr(7);pseed^=pseed.wrapping_shl(17);
                let offset=(pseed%((2*tenure_delta+1) as u64)) as usize; let progress=(iter as f64)/(effective_iterations as f64); let late_bonus=if progress>0.6{((progress-0.6)*10.0) as usize}else{0};
                let this_tenure=(tenure+offset+late_bonus).saturating_sub(tenure_delta);
                tabu_swap.insert((node_a.min(node_b),node_a.max(node_b)),iter+this_tenure);
            }
            Some((MoveType::Reassign{node,new_machine,new_pt,insert_pos},_)) => {
                let old_machine=ds.node_machine[node];
                let mut old_pos=pos_of_node[node];
                if old_pos>=ds.machine_seq[old_machine].len()||ds.machine_seq[old_machine][old_pos]!=node{
                    let Some(p)=ds.machine_seq[old_machine].iter().position(|&x|x==node) else{break};
                    old_pos=p;
                }
                ds.machine_seq[old_machine].remove(old_pos);
                let ins=insert_pos.min(ds.machine_seq[new_machine].len());
                ds.machine_seq[new_machine].insert(ins,node); ds.node_machine[node]=new_machine; ds.node_pt[node]=new_pt;
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

fn compute_tails_pulsar(ds: &DisjSchedule, buf: &EvalBuf) -> Vec<u32> {
    let n = ds.n;
    let mut tails = vec![0u32; n];
    let mut out_deg = vec![0u8; n];
    let mut machine_pred = vec![NONE_USIZE; n];

    for nd in 0..n {
        if ds.job_succ[nd] != NONE_USIZE {
            out_deg[nd] += 1;
        }
        if buf.machine_succ[nd] != NONE_USIZE {
            out_deg[nd] += 1;
        }
    }
    for seq in &ds.machine_seq {
        for i in 1..seq.len() {
            machine_pred[seq[i]] = seq[i - 1];
        }
    }

    let mut stack: Vec<usize> = Vec::with_capacity(n);
    for nd in 0..n {
        if out_deg[nd] == 0 {
            stack.push(nd);
        }
    }

    while let Some(nd) = stack.pop() {
        let contrib = ds.node_pt[nd].saturating_add(tails[nd]);
        let job = ds.node_job[nd];
        let jp = if nd > ds.job_offsets[job] { nd - 1 } else { NONE_USIZE };
        if jp != NONE_USIZE {
            if contrib > tails[jp] {
                tails[jp] = contrib;
            }
            out_deg[jp] = out_deg[jp].saturating_sub(1);
            if out_deg[jp] == 0 {
                stack.push(jp);
            }
        }
        let mp = machine_pred[nd];
        if mp != NONE_USIZE {
            if contrib > tails[mp] {
                tails[mp] = contrib;
            }
            out_deg[mp] = out_deg[mp].saturating_sub(1);
            if out_deg[mp] == 0 {
                stack.push(mp);
            }
        }
    }

    tails
}

#[allow(dead_code)]
fn critical_machine_shadow_prices_fm(
    pre: &Pre,
    challenge: &Challenge,
    elites: &[(Solution, u32)],
) -> Result<Vec<f64>> {
    let mut prices = vec![0.0f64; challenge.num_machines];
    let mut sampled = 0usize;

    for (sol, _) in elites.iter().take(8) {
        let ds = build_disj_from_solution(pre, challenge, sol)?;
        let mut buf = EvalBuf::new(ds.n);
        let Some((makespan, mk_node)) = eval_disj(&ds, &mut buf) else {
            continue;
        };
        let tails = compute_tails_pulsar(&ds, &buf);
        let tolerance = ((pre.avg_op_min * (0.70 + pre.jobshopness)).round() as u32).max(1);
        let corridor = critical_corridor_fm(&ds, &buf, &tails, makespan, tolerance, 40);
        let mut exact = vec![false; ds.n];
        let mut node = mk_node;
        let mut path_seen = 0usize;
        while node != NONE_USIZE && path_seen < 40 {
            exact[node] = true;
            node = buf.best_pred[node];
            path_seen += 1;
        }

        for node in corridor {
            let machine = ds.node_machine[node];
            if machine < prices.len() {
                let weight = if exact[node] { 1.0 } else { 0.28 };
                prices[machine] += weight;
            }
        }
        sampled += 1;
    }

    if sampled > 0 {
        let scale = sampled as f64;
        for price in &mut prices {
            *price = (*price / scale).clamp(0.0, 1.50);
        }
    }
    Ok(prices)
}

fn critical_corridor_fm(
    ds: &DisjSchedule,
    buf: &EvalBuf,
    tails: &[u32],
    makespan: u32,
    tolerance: u32,
    cap: usize,
) -> Vec<usize> {
    let mut corridor = Vec::with_capacity(cap);
    if cap == 0 {
        return corridor;
    }

    let mut exact = vec![false; ds.n];
    let mut u = buf
        .start
        .iter()
        .enumerate()
        .max_by_key(|&(nd, &st)| st.saturating_add(ds.node_pt[nd]))
        .map(|(nd, _)| nd)
        .unwrap_or(NONE_USIZE);
    while u != NONE_USIZE && corridor.len() < cap {
        if !exact[u] {
            exact[u] = true;
            corridor.push(u);
        }
        u = buf.best_pred[u];
    }
    if corridor.len() >= cap {
        return corridor;
    }

    let mut represented_machine = vec![false; ds.num_machines];
    let mut represented_job = vec![false; ds.num_jobs];
    for &nd in &corridor {
        represented_machine[ds.node_machine[nd]] = true;
        represented_job[ds.node_job[nd]] = true;
    }

    let mut order: Vec<usize> = (0..ds.n).collect();
    order.sort_unstable_by(|&a, &b| {
        let ca = buf.start[a]
            .saturating_add(ds.node_pt[a])
            .saturating_add(tails[a]);
        let cb = buf.start[b]
            .saturating_add(ds.node_pt[b])
            .saturating_add(tails[b]);
        cb.cmp(&ca).then_with(|| a.cmp(&b))
    });

    for &nd in &order {
        if corridor.len() >= cap {
            break;
        }
        let completion = buf.start[nd]
            .saturating_add(ds.node_pt[nd])
            .saturating_add(tails[nd]);
        if exact[nd] || completion.saturating_add(tolerance) < makespan {
            continue;
        }
        let m = ds.node_machine[nd];
        let j = ds.node_job[nd];
        if !represented_machine[m] || !represented_job[j] {
            represented_machine[m] = true;
            represented_job[j] = true;
            corridor.push(nd);
        }
    }

    for &nd in &order {
        if corridor.len() >= cap {
            break;
        }
        let completion = buf.start[nd]
            .saturating_add(ds.node_pt[nd])
            .saturating_add(tails[nd]);
        if !exact[nd] && completion.saturating_add(tolerance) >= makespan {
            corridor.push(nd);
        }
    }
    corridor
}

fn release_dates_excl_machine(ds: &DisjSchedule, excl_m: usize) -> Vec<u32> {
    let n = ds.n;
    let mut indeg = ds.indeg_job.clone();
    let mut msucc = vec![NONE_USIZE; n];
    for m in 0..ds.num_machines {
        if m == excl_m {
            continue;
        }
        let seq = &ds.machine_seq[m];
        for i in 0..seq.len().saturating_sub(1) {
            let u = seq[i];
            let v = seq[i + 1];
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
            if start[js] < eu {
                start[js] = eu;
            }
            indeg[js] = indeg[js].saturating_sub(1);
            if indeg[js] == 0 {
                stack.push(js);
            }
        }
        let ms = msucc[u];
        if ms != NONE_USIZE {
            if start[ms] < eu {
                start[ms] = eu;
            }
            indeg[ms] = indeg[ms].saturating_sub(1);
            if indeg[ms] == 0 {
                stack.push(ms);
            }
        }
    }
    start
}

fn schrage_seq_pulsar(nodes: &[usize], r: &[u32], p: &[u32], q: &[u32]) -> Vec<usize> {
    let m = nodes.len();
    if m <= 1 {
        return nodes.to_vec();
    }
    let mut by_r: Vec<usize> = nodes.to_vec();
    by_r.sort_unstable_by_key(|&nd| r[nd]);
    let mut result = Vec::with_capacity(m);
    let mut t = r[by_r[0]];
    let mut i = 0usize;
    let mut avail: Vec<usize> = Vec::with_capacity(m);
    while result.len() < m {
        while i < by_r.len() && r[by_r[i]] <= t {
            avail.push(by_r[i]);
            i += 1;
        }
        if avail.is_empty() {
            if i < by_r.len() {
                t = r[by_r[i]];
                continue;
            }
            break;
        }
        let bp = avail
            .iter()
            .enumerate()
            .max_by_key(|&(_, &nd)| q[nd])
            .map(|(p2, _)| p2)
            .unwrap_or(0);
        let chosen = avail.swap_remove(bp);
        result.push(chosen);
        t = t.saturating_add(p[chosen]);
    }
    result
}

fn schrage_pass(
    pre: &Pre,
    challenge: &Challenge,
    base_sol: &Solution,
    max_iters: usize,
) -> Result<Option<(Solution, u32)>> {
    #[derive(Clone, Copy)]
    struct MachineWindow {
        m: usize,
        ws: usize,
        we: usize,
        run_len: usize,
        crit_cnt: usize,
    }

    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n);
    let Some((mut cur_mk, _)) = eval_disj(&ds, &mut buf) else {
        return Ok(None);
    };
    let init_mk = cur_mk;
    const HALO: usize = 1;
    const MAX_WINDOWS_PER_ITER: usize = 4;

    for _ in 0..max_iters {
        let Some((mk_now, mk_node)) = eval_disj(&ds, &mut buf) else {
            break;
        };
        cur_mk = mk_now;
        let tails = compute_tails_pulsar(&ds, &buf);

        let mut crit = vec![false; ds.n];
        let mut u = mk_node;
        while u != NONE_USIZE {
            crit[u] = true;
            u = buf.best_pred[u];
        }

        let mut candidates: Vec<MachineWindow> = Vec::new();
        for m in 0..ds.num_machines {
            let seq = &ds.machine_seq[m];
            if seq.len() < 3 {
                continue;
            }

            let mut best_local: Option<MachineWindow> = None;
            let mut i = 0usize;
            while i < seq.len() {
                if !crit[seq[i]] {
                    i += 1;
                    continue;
                }
                let start = i;
                while i + 1 < seq.len() && crit[seq[i + 1]] {
                    i += 1;
                }
                let end = i;
                let run_len = end - start + 1;
                if run_len >= 2 {
                    let ws = start.saturating_sub(HALO);
                    let we = (end + HALO).min(seq.len() - 1);
                    let mut crit_cnt = 0usize;
                    for &nd in &seq[ws..=we] {
                        if crit[nd] {
                            crit_cnt += 1;
                        }
                    }
                    let cand = MachineWindow { m, ws, we, run_len, crit_cnt };
                    let replace = match best_local {
                        None => true,
                        Some(best) => {
                            cand.run_len > best.run_len
                                || (cand.run_len == best.run_len
                                    && (cand.crit_cnt > best.crit_cnt
                                        || (cand.crit_cnt == best.crit_cnt
                                            && (cand.we - cand.ws) < (best.we - best.ws))))
                        }
                    };
                    if replace {
                        best_local = Some(cand);
                    }
                }
                i += 1;
            }

            if let Some(cand) = best_local {
                candidates.push(cand);
            }
        }

        if candidates.is_empty() {
            break;
        }

        candidates.sort_unstable_by(|a, b| {
            b.run_len
                .cmp(&a.run_len)
                .then_with(|| b.crit_cnt.cmp(&a.crit_cnt))
                .then_with(|| (a.we - a.ws).cmp(&(b.we - b.ws)))
                .then_with(|| a.m.cmp(&b.m))
        });

        let mut improved = false;
        for cand in candidates.into_iter().take(MAX_WINDOWS_PER_ITER) {
            let m = cand.m;
            let old_seq = ds.machine_seq[m].clone();
            let window_nodes = old_seq[cand.ws..=cand.we].to_vec();
            if window_nodes.len() <= 2 {
                continue;
            }

            let r_excl = release_dates_excl_machine(&ds, m);
            let new_window = schrage_seq_pulsar(&window_nodes, &r_excl, &ds.node_pt, &tails);
            if new_window == window_nodes {
                continue;
            }

            let mut new_seq = old_seq.clone();
            for (off, &nd) in new_window.iter().enumerate() {
                new_seq[cand.ws + off] = nd;
            }
            ds.machine_seq[m] = new_seq;

            if let Some((new_mk, _)) = eval_disj(&ds, &mut buf) {
                if new_mk < cur_mk {
                    cur_mk = new_mk;
                    improved = true;
                    break;
                }
            }

            ds.machine_seq[m] = old_seq;
            let _ = eval_disj(&ds, &mut buf);
        }

        if !improved {
            break;
        }
    }
    if cur_mk < init_mk {
        Ok(Some((disj_to_solution(pre, &ds, &buf.start)?, cur_mk)))
    } else {
        Ok(None)
    }
}

#[allow(dead_code)]
fn find_candidate_insert_positions_fm(
    ds: &DisjSchedule,
    starts: &[u32],
    node: usize,
    new_machine: usize,
    _new_pt: u32,
    job_pred: &[usize],
    out: &mut [usize; 6],
) -> usize {
    let seq = &ds.machine_seq[new_machine];
    let len = seq.len();
    if len == 0 {
        out[0] = 0;
        return 1;
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

    let mut out_len = 0usize;
    #[inline]
    fn push_uniq(out: &mut [usize; 6], out_len: &mut usize, p: usize, len: usize) {
        if p <= len && !out[..*out_len].contains(&p) {
            out[*out_len] = p;
            *out_len += 1;
        }
    }

    push_uniq(out, &mut out_len, pos_after_jp, len);
    push_uniq(out, &mut out_len, pos_after_jp.saturating_sub(1), len);

    push_uniq(out, &mut out_len, pos_by_cur, len);
    push_uniq(out, &mut out_len, pos_by_cur.saturating_sub(1), len);

    push_uniq(out, &mut out_len, 0, len);
    push_uniq(out, &mut out_len, len, len);

    if out_len == 0 {
        out[0] = len;
        out_len = 1;
    }
    out_len.min(5)
}

#[allow(dead_code)]
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

fn critical_path_mg(
    ds: &DisjSchedule,
    heads: &[u32],
    qend: &[u32],
    ms: &[usize],
    makespan: u32,
    path: &mut Vec<usize>,
) {
    let crit = |o: usize| heads[o].saturating_add(qend[o]) == makespan;
    let mut start = NONE_USIZE;
    for j in 0..ds.num_jobs {
        let o = ds.job_offsets[j];
        if o < ds.n && crit(o) && heads[o] == 0 {
            start = o;
            break;
        }
    }
    if start == NONE_USIZE {
        let mut best = u32::MAX;
        for o in 0..ds.n {
            if crit(o) && heads[o] < best {
                best = heads[o];
                start = o;
            }
        }
    }
    path.clear();
    if start == NONE_USIZE {
        return;
    }
    let mut cur = start;
    loop {
        path.push(cur);
        let end = heads[cur].saturating_add(ds.node_pt[cur]);
        let mut next = NONE_USIZE;
        let mv = ms[cur];
        if mv != NONE_USIZE && crit(mv) && heads[mv] == end {
            next = mv;
        }
        if next == NONE_USIZE {
            let js = ds.job_succ[cur];
            if js != NONE_USIZE && crit(js) && heads[js] == end {
                next = js;
            }
        }
        if next == NONE_USIZE {
            break;
        }
        cur = next;
    }
}

fn n5_moves_full_mg(
    ds: &DisjSchedule,
    heads: &[u32],
    qend: &[u32],
    makespan: u32,
    moves: &mut Vec<(usize, usize)>,
) {
    let crit = |o: usize| heads[o].saturating_add(qend[o]) == makespan;
    moves.clear();
    for m in 0..ds.num_machines {
        let seq = &ds.machine_seq[m];
        let mut pos = 0usize;
        while pos < seq.len() {
            if !crit(seq[pos]) {
                pos += 1;
                continue;
            }
            let run_start = pos;
            let mut run_end = pos;
            while run_end + 1 < seq.len() {
                let node = seq[run_end + 1];
                let prev = seq[run_end];
                if crit(node) && heads[node] == heads[prev].saturating_add(ds.node_pt[prev]) {
                    run_end += 1;
                } else {
                    break;
                }
            }
            let block_len = run_end - run_start + 1;
            if block_len >= 2 {
                moves.push((m, run_start));
                if block_len >= 3 {
                    moves.push((m, run_end - 1));
                }
            }
            pos = run_end + 1;
        }
    }
}

fn n5_moves_path_mg(
    ds: &DisjSchedule,
    path: &[usize],
    mpos: &[usize],
    moves: &mut Vec<(usize, usize)>,
    blk: &mut Vec<(u32, u32)>,
) {
    blk.clear();
    moves.clear();
    let mut s0 = 0usize;
    for i in 1..path.len() {
        let o = path[i];
        let prev = path[i - 1];
        if !(ds.node_machine[o] == ds.node_machine[prev] && mpos[o] == mpos[prev] + 1) {
            blk.push((s0 as u32, i as u32));
            s0 = i;
        }
    }
    if !path.is_empty() {
        blk.push((s0 as u32, path.len() as u32));
    }
    let nb = blk.len();
    for bi in 0..nb {
        let bs = blk[bi].0 as usize;
        let be = blk[bi].1 as usize;
        if be - bs < 2 {
            continue;
        }
        let m = ds.node_machine[path[bs]];
        if bi != 0 {
            moves.push((m, mpos[path[bs]]));
        }
        if bi != nb - 1 {
            let p = mpos[path[be - 2]];
            if !(bi != 0 && p == mpos[path[bs]]) {
                moves.push((m, p));
            }
        }
    }
}

fn best_machine_move_mg_fm(
    ds: &DisjSchedule,
    pre: &Pre,
    path: &[usize],
    heads: &[u32],
    qend: &[u32],
    mp: &[usize],
    best_real: u32,
    tabu: &HashMap<(usize, usize), usize>,
    it: usize,
    tabu_hi: &[usize],
    sweep_cap: usize,
    w: &mut [u32],
    stamp: &mut [u32],
    gen: &mut u32,
    gap: &mut [u32],
) -> Option<(usize, usize, usize, u32, u32)> {
    *gen = gen.wrapping_add(1);
    let cur = *gen;
    for k in 0..ds.num_machines {
        let mut a = 0u32;
        let mut lo = u32::MAX;
        for &o in &ds.machine_seq[k] {
            let s = a.saturating_add(qend[o]);
            if s < lo {
                lo = s;
            }
            a = heads[o].saturating_add(ds.node_pt[o]);
        }
        if a < lo {
            lo = a;
        }
        gap[k] = lo;
    }
    let mut best: Option<(usize, usize, usize, u32, u32)> = None;
    let mut seen = 0usize;
    for &v in path {
        let job = ds.node_job[v];
        let oi = ds.node_op[v];
        let product = pre.job_products[job];
        if oi >= pre.product_ops[product].len() {
            continue;
        }
        let elig = &pre.product_ops[product][oi].machines;
        if elig.len() < 2 {
            continue;
        }
        seen += 1;
        if seen > sweep_cap {
            break;
        }
        let m_cur = ds.node_machine[v];
        let rpj = if ds.job_offsets[job] < v { heads[v - 1].saturating_add(ds.node_pt[v - 1]) } else { 0 };
        let qsj = if ds.job_succ[v] != NONE_USIZE { qend[ds.job_succ[v]] } else { 0 };
        let is_tabu = if tabu_hi[v] <= it { false } else { tabu.get(&(v, usize::MAX)).map_or(false, |&e| e > it) };
        for &(k, pk) in elig {
            if k == m_cur {
                continue;
            }
            if let Some((_, _, _, _, bl)) = best {
                if rpj.saturating_add(pk).saturating_add(qsj) >= bl || pk.saturating_add(gap[k]) >= bl {
                    continue;
                }
            }
            let seq = &ds.machine_seq[k];
            let len = seq.len();
            let left = seq.partition_point(|&o| heads[o].saturating_add(ds.node_pt[o]) <= rpj);
            let right = seq.partition_point(|&o| qend[o] > qsj);
            if left > right {
                continue;
            }
            for pos in left..=right {
                let l = if pos > left && pos < right {
                    let o = seq[pos];
                    if stamp[o] != cur {
                        let p = mp[o];
                        w[o] = if p != NONE_USIZE {
                            heads[p].saturating_add(ds.node_pt[p]).saturating_add(qend[o])
                        } else {
                            qend[o]
                        };
                        stamp[o] = cur;
                    }
                    pk.saturating_add(w[o])
                } else {
                    let hv = if pos > 0 {
                        heads[seq[pos - 1]].saturating_add(ds.node_pt[seq[pos - 1]]).max(rpj)
                    } else {
                        rpj
                    };
                    let tv = if pos < len { qend[seq[pos]].max(qsj) } else { qsj };
                    hv.saturating_add(pk).saturating_add(tv)
                };
                let allowed = !is_tabu || l < best_real;
                if allowed && best.map_or(true, |(_, _, _, _, bl)| l < bl) {
                    best = Some((v, k, pos, pk, l));
                }
            }
        }
    }
    best
}

fn pt_on_machine(pre: &Pre, ds: &DisjSchedule, node: usize, machine: usize) -> Option<u32> {
    let product = pre.job_products[ds.node_job[node]];
    let oi = ds.node_op[node];
    let op = pre.product_ops.get(product)?.get(oi)?;
    op.machines.iter().find(|&&(m, _)| m == machine).map(|&(_, d)| d)
}

fn best_machine_swap_mg_fm(
    ds: &DisjSchedule,
    pre: &Pre,
    path: &[usize],
    heads: &[u32],
    qend: &[u32],
    mpos: &[usize],
    best_real: u32,
    tabu: &HashMap<(usize, usize), usize>,
    it: usize,
    tabu_hi: &[usize],
    sweep_cap: usize,
) -> Option<(usize, usize, u32)> {
    let mut best: Option<(usize, usize, u32)> = None;
    let mut seen = 0usize;
    for &v in path {
        let job = ds.node_job[v];
        let oi = ds.node_op[v];
        let product = pre.job_products[job];
        if oi >= pre.product_ops[product].len() {
            continue;
        }
        let elig = &pre.product_ops[product][oi].machines;
        if elig.len() < 2 {
            continue;
        }
        seen += 1;
        if seen > sweep_cap {
            break;
        }
        let mv = ds.node_machine[v];
        let pv = mpos[v];
        let rpv = if ds.job_offsets[job] < v {
            heads[v - 1].saturating_add(ds.node_pt[v - 1])
        } else {
            0
        };
        let qsv = if ds.job_succ[v] != NONE_USIZE { qend[ds.job_succ[v]] } else { 0 };
        let v_tabu = if tabu_hi[v] <= it {
            false
        } else {
            tabu.get(&(v, usize::MAX)).map_or(false, |&e| e > it)
        };
        let sm = &ds.machine_seq[mv];
        let hu_base = if pv > 0 {
            heads[sm[pv - 1]].saturating_add(ds.node_pt[sm[pv - 1]])
        } else {
            0
        };
        let tu_base = if pv + 1 < sm.len() { qend[sm[pv + 1]] } else { 0 };
        for &(k, pvk) in elig {
            if k == mv {
                continue;
            }
            if let Some((_, _, bl)) = best {
                if rpv.saturating_add(pvk).saturating_add(qsv) >= bl {
                    continue;
                }
            }
            let seq = &ds.machine_seq[k];
            let len = seq.len();
            if len == 0 {
                continue;
            }
            let left = seq.partition_point(|&o| heads[o].saturating_add(ds.node_pt[o]) <= rpv);
            let right = seq.partition_point(|&o| qend[o] > qsv);
            if left > right {
                continue;
            }
            let hi = if right < len { right } else { len - 1 };
            if left > hi {
                continue;
            }
            for pos in left..=hi {
                let u = seq[pos];
                let hv = if pos > 0 {
                    heads[seq[pos - 1]].saturating_add(ds.node_pt[seq[pos - 1]]).max(rpv)
                } else {
                    rpv
                };
                let tv = if pos + 1 < len {
                    qend[seq[pos + 1]].max(qsv)
                } else {
                    qsv
                };
                let l_v = hv.saturating_add(pvk).saturating_add(tv);
                if let Some((_, _, bl)) = best {
                    if l_v >= bl {
                        continue;
                    }
                }
                let Some(pumv) = pt_on_machine(pre, ds, u, mv) else {
                    continue;
                };
                let ju = ds.node_job[u];
                let rpu = if ds.job_offsets[ju] < u {
                    heads[u - 1].saturating_add(ds.node_pt[u - 1])
                } else {
                    0
                };
                let qsu = if ds.job_succ[u] != NONE_USIZE { qend[ds.job_succ[u]] } else { 0 };
                let hu = hu_base.max(rpu);
                let tu = tu_base.max(qsu);
                let l_u = hu.saturating_add(pumv).saturating_add(tu);
                let l = l_v.max(l_u);
                if let Some((_, _, bl)) = best {
                    if l >= bl {
                        continue;
                    }
                }
                let u_tabu = if tabu_hi[u] <= it {
                    false
                } else {
                    tabu.get(&(u, usize::MAX)).map_or(false, |&e| e > it)
                };
                if (v_tabu || u_tabu) && l >= best_real {
                    continue;
                }
                best = Some((v, u, l));
            }
        }
    }
    best
}

fn kick_critical_mg(ds: &mut DisjSchedule, buf: &mut EvalBuf, strength: usize, rng: &mut SmallRng) {
    let Some((cur, _)) = eval_disj(ds, buf) else { return };
    let tails = compute_tails_pulsar(ds, buf);
    let n = ds.n;
    let mut qend = vec![0u32; n];
    let mut ms = vec![NONE_USIZE; n];
    for o in 0..n {
        qend[o] = ds.node_pt[o].saturating_add(tails[o]);
    }
    for seq in &ds.machine_seq {
        for i in 0..seq.len().saturating_sub(1) {
            ms[seq[i]] = seq[i + 1];
        }
    }
    let mut path = Vec::new();
    critical_path_mg(ds, &buf.start, &qend, &ms, cur, &mut path);
    let mut cands: Vec<(usize, usize)> = Vec::new();
    for &u in &path {
        let m = ds.node_machine[u];
        if let Some(pos) = ds.machine_seq[m].iter().position(|&x| x == u) {
            if pos > 0 {
                cands.push((m, pos - 1));
            }
            if pos + 1 < ds.machine_seq[m].len() {
                cands.push((m, pos));
            }
        }
    }
    cands.sort_unstable();
    cands.dedup();
    if cands.is_empty() {
        return;
    }
    let mut applied = 0usize;
    let mut attempts = 0usize;
    while applied < strength && attempts < strength * 5 {
        attempts += 1;
        let (m, pos) = cands[rng.gen_range(0..cands.len())];
        if pos + 1 >= ds.machine_seq[m].len() {
            continue;
        }
        ds.machine_seq[m].swap(pos, pos + 1);
        if eval_disj(ds, buf).is_none() {
            ds.machine_seq[m].swap(pos, pos + 1);
        } else {
            applied += 1;
        }
    }
}

#[inline]
fn tails_from_topo(ds: &DisjSchedule, buf: &EvalBuf, tails: &mut [u32], qend: &mut [u32]) {
    for &u in buf.topo.iter().rev() {
        let mut t = 0u32;
        let js = ds.job_succ[u];
        if js != NONE_USIZE {
            t = t.max(qend[js]);
        }
        let ms = buf.machine_succ[u];
        if ms != NONE_USIZE {
            t = t.max(qend[ms]);
        }
        tails[u] = t;
        qend[u] = ds.node_pt[u].saturating_add(t);
    }
}

#[inline]
fn repair_machine_arcs(seq: &[usize], mp: &mut [usize], mpos: &mut [usize]) {
    let mut prev = NONE_USIZE;
    for (i, &o) in seq.iter().enumerate() {
        mpos[o] = i;
        mp[o] = prev;
        prev = o;
    }
}

#[derive(Clone, Copy)]
enum IncrOut {
    Mk(u32),
    Cycle,
    Bad,
}

#[derive(Clone, Copy)]
enum ArcPk {
    Done,
    Cycle,
    Bad,
}

struct IncrTopo {
    ord: Vec<u32>,
    oat: Vec<u32>,
    markf: Vec<u32>,
    markb: Vec<u32>,
    dirty: Vec<u32>,
    stamp: u32,
    dstamp: u32,
    stk: Vec<u32>,
    df: Vec<u32>,
    db: Vec<u32>,
    pool: Vec<u32>,
    chg: Vec<u32>,
}

impl IncrTopo {
    fn new(n: usize) -> Self {
        Self {
            ord: vec![0; n],
            oat: vec![0; n],
            markf: vec![0; n],
            markb: vec![0; n],
            dirty: vec![0; n],
            stamp: 0,
            dstamp: 0,
            stk: vec![0; n],
            df: vec![0; n],
            db: vec![0; n],
            pool: vec![0; n],
            chg: Vec::with_capacity(n),
        }
    }

    fn sync(&mut self, topo: &[usize]) {
        for (rank, &node) in topo.iter().enumerate() {
            if node < self.ord.len() {
                self.ord[node] = rank as u32;
                self.oat[rank] = node as u32;
            }
        }
    }

    fn bump(&mut self) -> u32 {
        if self.stamp == u32::MAX {
            self.markf.fill(0);
            self.markb.fill(0);
            self.stamp = 1;
        } else {
            self.stamp += 1;
        }
        self.stamp
    }

    fn pk_arc(&mut self, x: usize, y: usize, job_succ: &[usize], machine_succ: &[usize], job_pred: &[usize], machine_pred: &[usize]) -> ArcPk {
        let n = self.ord.len();
        if x >= n || y >= n || x == y {
            return ArcPk::Bad;
        }
        let ox = self.ord[x];
        let oy = self.ord[y];
        if ox == oy {
            return ArcPk::Bad;
        }
        if ox < oy {
            return ArcPk::Done;
        }
        let lb = oy;
        let ub = ox;
        let stamp = self.bump();
        let mut dfn = 0usize;
        let mut sl = 0usize;
        self.markf[y] = stamp;
        self.stk[sl] = y as u32;
        sl += 1;
        while sl > 0 {
            sl -= 1;
            let v = self.stk[sl] as usize;
            if dfn >= n {
                return ArcPk::Bad;
            }
            self.df[dfn] = v as u32;
            dfn += 1;
            for &w in [job_succ[v], machine_succ[v]].iter() {
                if w == NONE_USIZE || w >= n {
                    continue;
                }
                if self.ord[w] == ub {
                    return ArcPk::Cycle;
                }
                if self.markf[w] != stamp && self.ord[w] < ub {
                    if sl >= n {
                        return ArcPk::Bad;
                    }
                    self.markf[w] = stamp;
                    self.stk[sl] = w as u32;
                    sl += 1;
                }
            }
        }
        let mut dbn = 0usize;
        sl = 0;
        if self.markf[x] == stamp {
            return ArcPk::Bad;
        }
        self.markb[x] = stamp;
        self.stk[sl] = x as u32;
        sl += 1;
        while sl > 0 {
            sl -= 1;
            let v = self.stk[sl] as usize;
            if self.markf[v] == stamp {
                return ArcPk::Bad;
            }
            if dbn >= n {
                return ArcPk::Bad;
            }
            self.db[dbn] = v as u32;
            dbn += 1;
            for &w in [job_pred[v], machine_pred[v]].iter() {
                if w == NONE_USIZE || w >= n {
                    continue;
                }
                if self.markb[w] != stamp && self.ord[w] > lb {
                    if sl >= n {
                        return ArcPk::Bad;
                    }
                    self.markb[w] = stamp;
                    self.stk[sl] = w as u32;
                    sl += 1;
                }
            }
        }
        self.df[..dfn].sort_unstable_by_key(|&o| self.ord[o as usize]);
        self.db[..dbn].sort_unstable_by_key(|&o| self.ord[o as usize]);
        let mut pn = 0usize;
        let mut i = 0usize;
        let mut j = 0usize;
        while i < dbn || j < dfn {
            let take_b = if j >= dfn {
                true
            } else if i >= dbn {
                false
            } else {
                self.ord[self.db[i] as usize] < self.ord[self.df[j] as usize]
            };
            if pn >= self.pool.len() {
                return ArcPk::Bad;
            }
            if take_b {
                self.pool[pn] = self.ord[self.db[i] as usize];
                i += 1;
            } else {
                self.pool[pn] = self.ord[self.df[j] as usize];
                j += 1;
            }
            pn += 1;
        }
        for k in 0..pn {
            let o = if k < dbn { self.db[k] } else { self.df[k - dbn] } as usize;
            let r = self.pool[k];
            self.ord[o] = r;
            self.oat[r as usize] = o as u32;
            self.chg.push(o as u32);
        }
        ArcPk::Done
    }

    fn repair_machines(
        &mut self,
        seqs: &[Vec<usize>],
        machines: &[usize],
        job_succ: &[usize],
        machine_succ: &[usize],
        job_pred: &[usize],
        machine_pred: &[usize],
    ) -> ArcPk {
        let n = self.ord.len();
        let mut inserts = 0usize;
        for _round in 0..4 {
            let mut pending = false;
            for &m in machines {
                if m >= seqs.len() {
                    continue;
                }
                let len = seqs[m].len();
                let mut i = 0usize;
                while i + 1 < len {
                    let x = seqs[m][i];
                    let y = seqs[m][i + 1];
                    if x >= n || y >= n {
                        return ArcPk::Bad;
                    }
                    if self.ord[x] >= self.ord[y] {
                        inserts += 1;
                        if inserts > 16 {
                            return ArcPk::Bad;
                        }
                        match self.pk_arc(x, y, job_succ, machine_succ, job_pred, machine_pred) {
                            ArcPk::Done => {}
                            other => return other,
                        }
                        pending = true;
                        i = 0;
                        continue;
                    }
                    i += 1;
                }
            }
            if !pending {
                return self.edges_ok(job_succ, machine_succ, job_pred, machine_pred);
            }
        }
        ArcPk::Bad
    }

    fn edges_ok(&self, job_succ: &[usize], machine_succ: &[usize], job_pred: &[usize], machine_pred: &[usize]) -> ArcPk {
        let n = self.ord.len();
        for &ow in &self.chg {
            let o = ow as usize;
            if o >= n {
                return ArcPk::Bad;
            }
            let r = self.ord[o] as usize;
            if r >= n || self.oat[r] as usize != o {
                return ArcPk::Bad;
            }
            let js = job_succ[o];
            if js != NONE_USIZE && (js >= n || (self.ord[js] as usize) <= r) {
                return ArcPk::Bad;
            }
            let ms = machine_succ[o];
            if ms != NONE_USIZE && (ms >= n || (self.ord[ms] as usize) <= r) {
                return ArcPk::Bad;
            }
            let jp = job_pred[o];
            if jp != NONE_USIZE && (jp >= n || (self.ord[jp] as usize) >= r) {
                return ArcPk::Bad;
            }
            let mp = machine_pred[o];
            if mp != NONE_USIZE && (mp >= n || (self.ord[mp] as usize) >= r) {
                return ArcPk::Bad;
            }
        }
        ArcPk::Done
    }
}

fn sync_machine_succ(seq: &[usize], machine_succ: &mut [usize]) {
    let mut prev = NONE_USIZE;
    for &v in seq {
        if prev != NONE_USIZE {
            machine_succ[prev] = v;
        }
        prev = v;
    }
    if prev != NONE_USIZE {
        machine_succ[prev] = NONE_USIZE;
    }
}

fn incr_eval_move(
    incr: &mut IncrTopo,
    ds: &DisjSchedule,
    buf: &mut EvalBuf,
    job_pred: &[usize],
    machine_pred: &[usize],
    machines: &[usize],
    pt_changed: &[usize],
) -> IncrOut {
    let n = ds.n;
    if buf.topo.len() != n || incr.ord.len() != n {
        return IncrOut::Bad;
    }
    incr.chg.clear();
    for &m in machines {
        if m < ds.machine_seq.len() {
            sync_machine_succ(&ds.machine_seq[m], &mut buf.machine_succ);
        }
    }
    match incr.repair_machines(&ds.machine_seq, machines, &ds.job_succ, &buf.machine_succ, job_pred, machine_pred) {
        ArcPk::Done => {}
        ArcPk::Cycle => return IncrOut::Cycle,
        ArcPk::Bad => return IncrOut::Bad,
    }
    if incr.dstamp == u32::MAX {
        incr.dirty.fill(0);
        incr.dstamp = 1;
    } else {
        incr.dstamp += 1;
    }
    let stamp = incr.dstamp;
    let mut lo = n;
    let mut seed = |incr: &mut IncrTopo, node: usize| {
        if node >= n {
            return;
        }
        let rank = incr.ord[node] as usize;
        if rank < n && incr.dirty[rank] != stamp {
            incr.dirty[rank] = stamp;
            if rank < lo {
                lo = rank;
            }
        }
    };
    for &m in machines {
        if m < ds.machine_seq.len() {
            for &node in &ds.machine_seq[m] {
                seed(incr, node);
            }
        }
    }
    for &node in pt_changed {
        if node >= n {
            continue;
        }
        seed(incr, node);
        seed(incr, ds.job_succ[node]);
        seed(incr, buf.machine_succ[node]);
    }
    if lo < n {
        for rank in lo..n {
            if incr.dirty[rank] != stamp {
                continue;
            }
            let o = incr.oat[rank] as usize;
            if o >= n || incr.ord[o] as usize != rank {
                return IncrOut::Bad;
            }
            let mut h = 0u32;
            let jp = job_pred[o];
            if jp != NONE_USIZE && jp < n {
                if (incr.ord[jp] as usize) >= rank {
                    return IncrOut::Bad;
                }
                h = h.max(buf.start[jp].saturating_add(ds.node_pt[jp]));
            }
            let pred = machine_pred[o];
            if pred != NONE_USIZE && pred < n {
                if (incr.ord[pred] as usize) >= rank {
                    return IncrOut::Bad;
                }
                h = h.max(buf.start[pred].saturating_add(ds.node_pt[pred]));
            }
            if h != buf.start[o] {
                buf.start[o] = h;
                let js = ds.job_succ[o];
                if js != NONE_USIZE && js < n {
                    let r = incr.ord[js] as usize;
                    if r <= rank {
                        return IncrOut::Bad;
                    }
                    incr.dirty[r] = stamp;
                }
                let ms = buf.machine_succ[o];
                if ms != NONE_USIZE && ms < n {
                    let r = incr.ord[ms] as usize;
                    if r <= rank {
                        return IncrOut::Bad;
                    }
                    incr.dirty[r] = stamp;
                }
            }
        }
        for rank in lo..n {
            if incr.dirty[rank] != stamp {
                continue;
            }
            let o = incr.oat[rank] as usize;
            if o >= n {
                return IncrOut::Bad;
            }
            let mut h = 0u32;
            let jp = job_pred[o];
            if jp != NONE_USIZE && jp < n {
                h = h.max(buf.start[jp].saturating_add(ds.node_pt[jp]));
            }
            let pred = machine_pred[o];
            if pred != NONE_USIZE && pred < n {
                h = h.max(buf.start[pred].saturating_add(ds.node_pt[pred]));
            }
            if h != buf.start[o] {
                return IncrOut::Bad;
            }
        }
    }
    buf.topo.clear();
    for &node in &incr.oat {
        buf.topo.push(node as usize);
    }
    if buf.topo.len() != n {
        return IncrOut::Bad;
    }
    let mut mk = 0u32;
    for o in 0..n {
        let end = buf.start[o].saturating_add(ds.node_pt[o]);
        if end > mk {
            mk = end;
        }
    }
    IncrOut::Mk(mk)
}

pub fn tabu_run_mg(
    pre: &Pre,
    challenge: &Challenge,
    seed_sol: &Solution,
    seed_mk: u32,
    max_iter: usize,
    tenure: usize,
    full: bool,
    max_kicks: usize,
    rng: &mut SmallRng,
    pre_kick: usize,
    mg_if_no_n5: bool,
    patience: usize,
    exchange: bool,
    fuel_floor: u64,
) -> Result<Option<(Solution, u32)>> {
    let Ok(mut ds) = build_disj_from_solution(pre, challenge, seed_sol) else {
        return Ok(None);
    };
    let mut buf = EvalBuf::new(ds.n);
    let Some((mut cur_real, _)) = eval_disj_topo(&ds, &mut buf) else {
        return Ok(None);
    };
    let n = ds.n;
    let mut best_ds = ds.clone();
    let mut best_real = cur_real.min(seed_mk);
    if pre_kick > 0 {
        kick_critical_mg(&mut ds, &mut buf, pre_kick, rng);
        if let Some((r, _)) = eval_disj_topo(&ds, &mut buf) {
            cur_real = r;
            if cur_real < best_real {
                best_real = cur_real;
                best_ds = ds.clone();
            }
        } else {
            ds = best_ds.clone();
            if let Some((r, _)) = eval_disj_topo(&ds, &mut buf) {
                cur_real = r;
            } else {
                return Ok(None);
            }
        }
    }
    let mut incr = if exchange { Some(IncrTopo::new(n)) } else { None };
    if let Some(incr) = incr.as_mut() {
        incr.sync(&buf.topo);
    }
    let mut tails = vec![0u32; n];
    let mut tabu: HashMap<(usize, usize), usize> = HashMap::with_capacity(tenure * 8);
    let mut tabu_hi = vec![0usize; n];
    let mut no_improve = 0usize;
    let max_no_improve = if max_kicks > 0 { patience } else { 8 * tenure + 400 };
    let sweep_cap = if pre.product_ops.len() >= 27 { 60usize } else { 80usize };
    let mut kicks_left = max_kicks;
    let mut mg_w = vec![0u32; n];
    let mut mg_stamp = vec![0u32; n];
    let mut mg_gen = 0u32;
    let mut mg_gap = vec![0u32; ds.num_machines];
    let mut path = Vec::with_capacity(n);
    let mut moves = Vec::with_capacity(n);
    let mut blk = Vec::with_capacity(n);
    let mut qend = vec![0u32; n];
    let mut mp = vec![NONE_USIZE; n];
    let mut mpos = vec![0usize; n];
    let mut job_pred = vec![NONE_USIZE; n];
    for j in 0..ds.num_jobs {
        let base = ds.job_offsets[j];
        let end = ds.job_offsets[j + 1];
        for k in (base + 1)..end {
            job_pred[k] = k - 1;
        }
    }
    for m in 0..ds.num_machines {
        repair_machine_arcs(&ds.machine_seq[m], &mut mp, &mut mpos);
    }

    for it in 0..max_iter {
        if fuel_floor > 0 && fuel_left() <= fuel_floor {
            break;
        }
        tails_from_topo(&ds, &buf, &mut tails, &mut qend);
        critical_path_mg(&ds, &buf.start, &qend, &buf.machine_succ, cur_real, &mut path);
        if path.len() < 2 {
            break;
        }
        if full {
            n5_moves_full_mg(&ds, &buf.start, &qend, cur_real, &mut moves);
        } else {
            n5_moves_path_mg(&ds, &path, &mpos, &mut moves, &mut blk);
        }
        if moves.is_empty() && !mg_if_no_n5 {
            break;
        }
        let mut best_swap: Option<(usize, usize)> = None;
        let mut best_est = u32::MAX;
        let mut best_pair = (0usize, 0usize);
        let mut fb_swap: Option<(usize, usize)> = None;
        let mut fb_est = u32::MAX;
        let mut fb_pair = (0usize, 0usize);
        for &(m, p) in &moves {
            if p + 1 >= ds.machine_seq[m].len() {
                continue;
            }
            let a = ds.machine_seq[m][p];
            let b = ds.machine_seq[m][p + 1];
            let pair = if a < b { (a, b) } else { (b, a) };
            let est = estimate_swap_mk_fm(
                a, b, &buf.start, &tails, &ds.node_pt, &job_pred, &ds.job_succ, &mp, &buf.machine_succ,
            );
            let is_tabu = if tabu_hi[a] <= it || tabu_hi[b] <= it {
                false
            } else {
                tabu.get(&pair).map_or(false, |&exp| exp > it)
            };
            if (!is_tabu || est < best_real) && est < best_est {
                best_est = est;
                best_swap = Some((m, p));
                best_pair = pair;
            }
            if est < fb_est {
                fb_est = est;
                fb_swap = Some((m, p));
                fb_pair = pair;
            }
        }
        let mm = best_machine_move_mg_fm(
            &ds, pre, &path, &buf.start, &qend, &mp, best_real, &tabu, it, &tabu_hi,
            sweep_cap, &mut mg_w, &mut mg_stamp, &mut mg_gen, &mut mg_gap,
        );
        let mm_l = mm.map_or(u32::MAX, |(_, _, _, _, l)| l);
        let xs = if exchange && best_est >= cur_real && mm_l >= cur_real {
            best_machine_swap_mg_fm(
                &ds, pre, &path, &buf.start, &qend, &mpos, best_real, &tabu, it, &tabu_hi, sweep_cap,
            )
        } else {
            None
        };
        let xs_l = xs.map_or(u32::MAX, |(_, _, l)| l);
        let mut mg_revert: Option<(usize, usize, usize, u32)> = None;
        let mut swap_revert: Option<(usize, usize)> = None;
        let mut xs_revert: Option<(usize, usize, usize, u32, usize, usize, usize, u32)> = None;
        let xs_pick = if xs_l < cur_real { xs } else { None };
        let xs_pick = xs_pick.filter(|&(v, u, _)| {
            let mv = ds.node_machine[v];
            let k = ds.node_machine[u];
            let pv = mpos[v];
            let pu = mpos[u];
            mv != k
                && pv < ds.machine_seq[mv].len()
                && pu < ds.machine_seq[k].len()
                && ds.machine_seq[mv][pv] == v
                && ds.machine_seq[k][pu] == u
                && pt_on_machine(pre, &ds, v, k).is_some()
                && pt_on_machine(pre, &ds, u, mv).is_some()
        });
        if let Some((v, u, _)) = xs_pick {
            let mv = ds.node_machine[v];
            let k = ds.node_machine[u];
            let pv = mpos[v];
            let pu = mpos[u];
            let dv_old = ds.node_pt[v];
            let du_old = ds.node_pt[u];
            let dv_new = pt_on_machine(pre, &ds, v, k).unwrap();
            let du_new = pt_on_machine(pre, &ds, u, mv).unwrap();
            ds.machine_seq[mv].remove(pv);
            ds.machine_seq[k].remove(pu);
            ds.machine_seq[k].insert(pu, v);
            ds.machine_seq[mv].insert(pv, u);
            ds.node_machine[v] = k;
            ds.node_pt[v] = dv_new;
            ds.node_machine[u] = mv;
            ds.node_pt[u] = du_new;
            repair_machine_arcs(&ds.machine_seq[mv], &mut mp, &mut mpos);
            repair_machine_arcs(&ds.machine_seq[k], &mut mp, &mut mpos);
            tabu.insert((v, usize::MAX), it + tenure);
            tabu.insert((u, usize::MAX), it + tenure);
            if it + tenure > tabu_hi[v] {
                tabu_hi[v] = it + tenure;
            }
            if it + tenure > tabu_hi[u] {
                tabu_hi[u] = it + tenure;
            }
            xs_revert = Some((v, mv, pv, dv_old, u, k, pu, du_old));
        } else {
        if best_swap.is_none() && mm.is_none() {
            if let Some((m, p)) = fb_swap {
                best_swap = Some((m, p));
                best_est = fb_est;
                best_pair = fb_pair;
            } else {
                break;
            }
        }
        if mm_l < best_est {
            let (v, k, pos, new_pt, _) = mm.unwrap();
            let m_old = ds.node_machine[v];
            let Some(p_old) = ds.machine_seq[m_old].iter().position(|&x| x == v) else {
                break;
            };
            let old_dur = ds.node_pt[v];
            ds.machine_seq[m_old].remove(p_old);
            ds.node_machine[v] = k;
            ds.node_pt[v] = new_pt;
            let ins = pos.min(ds.machine_seq[k].len());
            ds.machine_seq[k].insert(ins, v);
            repair_machine_arcs(&ds.machine_seq[m_old], &mut mp, &mut mpos);
            repair_machine_arcs(&ds.machine_seq[k], &mut mp, &mut mpos);
            tabu.insert((v, usize::MAX), it + tenure);
            if it + tenure > tabu_hi[v] {
                tabu_hi[v] = it + tenure;
            }
            mg_revert = Some((v, m_old, p_old, old_dur));
        } else if let Some((m, p)) = best_swap {
            ds.machine_seq[m].swap(p, p + 1);
            repair_machine_arcs(&ds.machine_seq[m], &mut mp, &mut mpos);
            tabu.insert(best_pair, it + tenure);
            if it + tenure > tabu_hi[best_pair.0] {
                tabu_hi[best_pair.0] = it + tenure;
            }
            if it + tenure > tabu_hi[best_pair.1] {
                tabu_hi[best_pair.1] = it + tenure;
            }
            swap_revert = Some((m, p));
        } else {
            break;
        }
        }

        let undo_move = |ds: &mut DisjSchedule, mp: &mut [usize], mpos: &mut [usize]| {
            if let Some((v, mv, pv, dv_old, u, k, pu, du_old)) = xs_revert {
                ds.machine_seq[mv].remove(pv);
                ds.machine_seq[k].remove(pu);
                ds.machine_seq[k].insert(pu, u);
                ds.machine_seq[mv].insert(pv, v);
                ds.node_machine[v] = mv;
                ds.node_pt[v] = dv_old;
                ds.node_machine[u] = k;
                ds.node_pt[u] = du_old;
                repair_machine_arcs(&ds.machine_seq[mv], mp, mpos);
                repair_machine_arcs(&ds.machine_seq[k], mp, mpos);
            } else if let Some((v, m_old, p_old, old_dur)) = mg_revert {
                    let k = ds.node_machine[v];
                    if let Some(pn) = ds.machine_seq[k].iter().position(|&x| x == v) {
                        ds.machine_seq[k].remove(pn);
                    }
                    ds.node_machine[v] = m_old;
                    ds.node_pt[v] = old_dur;
                    let old_len = ds.machine_seq[m_old].len();
                    ds.machine_seq[m_old].insert(p_old.min(old_len), v);
                repair_machine_arcs(&ds.machine_seq[k], mp, mpos);
                repair_machine_arcs(&ds.machine_seq[m_old], mp, mpos);
                } else if let Some((m, p)) = swap_revert {
                    ds.machine_seq[m].swap(p, p + 1);
                repair_machine_arcs(&ds.machine_seq[m], mp, mpos);
            }
        };
        let mut fast_mk = None;
        let mut undone = false;
        if exchange {
            let mut machines = [0usize; 2];
            let mut nm = 0usize;
            let mut pt_changed = [NONE_USIZE; 2];
            let mut npt = 0usize;
            if let Some((v, mv, _, _, u, k, _, _)) = xs_revert {
                machines[0] = mv;
                machines[1] = k;
                nm = 2;
                pt_changed[0] = v;
                pt_changed[1] = u;
                npt = 2;
            } else if let Some((v, m_old, _, _)) = mg_revert {
                machines[0] = m_old;
                machines[1] = ds.node_machine[v];
                nm = if machines[0] == machines[1] { 1 } else { 2 };
                pt_changed[0] = v;
                npt = 1;
            } else if let Some((m, _)) = swap_revert {
                machines[0] = m;
                nm = 1;
            }
            if let Some(incr) = incr.as_mut() {
                match incr_eval_move(incr, &ds, &mut buf, &job_pred, &mp, &machines[..nm], &pt_changed[..npt]) {
                    IncrOut::Mk(r) => fast_mk = Some(r),
                    IncrOut::Cycle => {
                        undo_move(&mut ds, &mut mp, &mut mpos);
                        undone = true;
                    }
                    IncrOut::Bad => {}
                }
            }
        }
        cur_real = if let Some(r) = fast_mk {
            r
        } else {
            match eval_disj_topo(&ds, &mut buf) {
                Some((r, _)) => {
                    if let Some(incr) = incr.as_mut() {
                        incr.sync(&buf.topo);
                    }
                    r
                }
                None => {
                    if !undone {
                        undo_move(&mut ds, &mut mp, &mut mpos);
                    }
                    match eval_disj_topo(&ds, &mut buf) {
                        Some((r, _)) => {
                            if let Some(incr) = incr.as_mut() {
                                incr.sync(&buf.topo);
                            }
                            r
                        }
                    None => break,
                    }
                }
            }
        };
        if cur_real < best_real {
            best_real = cur_real;
            best_ds = ds.clone();
            no_improve = 0;
        } else {
            no_improve += 1;
            if no_improve >= max_no_improve {
                if kicks_left == 0 {
                    break;
                }
                ds = best_ds.clone();
                kick_critical_mg(&mut ds, &mut buf, 3 + (max_kicks - kicks_left), rng);
                cur_real = match eval_disj_topo(&ds, &mut buf) {
                    Some((r, _)) => {
                        if let Some(incr) = incr.as_mut() {
                            incr.sync(&buf.topo);
                        }
                        r
                    }
                    None => break,
                };
                for m in 0..ds.num_machines {
                    repair_machine_arcs(&ds.machine_seq[m], &mut mp, &mut mpos);
                }
                tabu.clear();
                tabu_hi.fill(0);
                no_improve = 0;
                kicks_left -= 1;
            }
        }
    }

    let Some((_, _)) = eval_disj(&best_ds, &mut buf) else {
        return Ok(None);
    };
    match disj_to_solution(pre, &best_ds, &buf.start) {
        Ok(s) => {
            if let Ok(mk) = challenge.evaluate_makespan(&s) {
                if mk < seed_mk {
                    Ok(Some((s, mk)))
                } else {
                    Ok(None)
                }
            } else {
                Ok(None)
            }
        }
        Err(_) => Ok(None),
    }
}

fn adaptive_budget(
    base_ts_iters: usize,
    base_cb_passes: usize,
    base_cb_iters: usize,
    base_alns_rounds: usize,
    base_ils_rounds: usize,
    base_final_ils_rounds: usize,
    global_no_improve: usize,
) -> (usize, usize, usize, usize, usize, usize, usize) {
    let factor = (global_no_improve as f64 / 80.0).min(1.8);
    let scale_up = 1.0 + 0.4 * factor;
    let scale_down = 1.0 / (1.0 + 0.25 * factor).max(0.6);
    let ts_iters = (base_ts_iters as f64 * scale_up).round() as usize;
    let cb_passes = (base_cb_passes as f64 * scale_up).round() as usize;
    let cb_iters = (base_cb_iters as f64 * scale_up).round() as usize;
    let alns_rounds = (base_alns_rounds as f64 * scale_down).round() as usize;
    let ils_rounds = (base_ils_rounds as f64 * scale_up).round() as usize;
    let final_ils_rounds = (base_final_ils_rounds as f64 * scale_down).round() as usize;
    let max_outer_iters = (65.0 * scale_down).round() as usize;
    (ts_iters, cb_passes, cb_iters, alns_rounds, ils_rounds, final_ils_rounds, max_outer_iters)
}


#[allow(dead_code)]
struct MediumPoolSeed {
    sol: Solution,
    mk: u32,
    deep: bool,
    machine: Vec<usize>,
    pos: Vec<usize>,
}

#[allow(dead_code)]
fn assignment_pos(sol: &Solution) -> (Vec<usize>, Vec<usize>) {
    let mut ops: Vec<(usize, u32, usize)> = Vec::new();
    let mut flat = 0usize;
    for job in &sol.job_schedule {
        for &(machine, start) in job {
            ops.push((machine, start, flat));
            flat += 1;
        }
    }
    let mut machine = vec![0usize; flat];
    let mut pos = vec![0usize; flat];
    let width = ops.iter().map(|op| op.0).max().unwrap_or(0).saturating_add(1);
    let mut by_machine: Vec<Vec<(u32, usize)>> = vec![Vec::new(); width];
    for &(m, start, i) in &ops {
        machine[i] = m;
        by_machine[m].push((start, i));
    }
    for seq in &mut by_machine {
        seq.sort_by_key(|item| item.0);
        for (p, &(_, i)) in seq.iter().enumerate() {
            pos[i] = p;
        }
    }
    (machine, pos)
}

#[allow(dead_code)]
fn schedule_distance(a_m: &[usize], a_p: &[usize], b_m: &[usize], b_p: &[usize]) -> usize {
    let n = a_m.len().min(b_m.len()).min(a_p.len()).min(b_p.len());
    let mut d = 0usize;
    for i in 0..n {
        if a_m[i] != b_m[i] {
            d += 2;
        }
        if a_p[i] != b_p[i] {
            d += 1;
        }
    }
    d
}

#[allow(dead_code)]
fn pool_consider(pool: &mut Vec<MediumPoolSeed>, sol: &Solution, mk: u32, cap: usize, min_dist: usize) {
    let (machine, pos) = assignment_pos(sol);
    let mut nearest = usize::MAX;
    let mut nd = usize::MAX;
    for (i, member) in pool.iter().enumerate() {
        let dd = schedule_distance(&member.machine, &member.pos, &machine, &pos);
        if dd < nd {
            nd = dd;
            nearest = i;
        }
    }
    if nearest != usize::MAX && nd < min_dist {
        if mk < pool[nearest].mk {
            pool[nearest] = MediumPoolSeed { sol: sol.clone(), mk, deep: false, machine, pos };
        }
        return;
    }
    if pool.len() < cap {
        pool.push(MediumPoolSeed { sol: sol.clone(), mk, deep: false, machine, pos });
        return;
    }
    let mut worst = 0usize;
    for i in 1..pool.len() {
        if pool[i].mk > pool[worst].mk {
            worst = i;
        }
    }
    if mk < pool[worst].mk {
        pool[worst] = MediumPoolSeed { sol: sol.clone(), mk, deep: false, machine, pos };
    }
}

#[allow(dead_code)]
fn basin_aware_elite_indices(
    pre: &Pre,
    challenge: &Challenge,
    pool: &[(Solution, u32)],
    incumbent: &Solution,
    incumbent_mk: u32,
    max_additional: usize,
) -> Result<Vec<usize>> {
    fn descriptor(
        pre: &Pre,
        challenge: &Challenge,
        sol: &Solution,
    ) -> Result<Option<(u32, u64, u64, u16)>> {
        let ds = build_disj_from_solution(pre, challenge, sol)?;
        let mut buf = EvalBuf::new(ds.n);
        let Some((mk, mk_node)) = eval_disj(&ds, &mut buf) else {
            return Ok(None);
        };

        let sample_stride = (pre.total_ops.saturating_add(31) / 32).max(1);
        let mut assignment_hash = 0u64;
        let mut flat = 0usize;
        for schedule in &sol.job_schedule {
            for &(machine, _) in schedule {
                if flat % sample_stride == 0 {
                    let value = (flat as u64)
                        .wrapping_mul(0x9E3779B97F4A7C15)
                        ^ (machine as u64).wrapping_mul(0xBF58476D1CE4E5B9);
                    assignment_hash ^= value.rotate_left((flat % 63) as u32);
                }
                flat = flat.saturating_add(1);
            }
        }

        let mut critical_machine_hash = 0u64;
        let mut critical_count = 0u32;
        let mut flexible_count = 0u32;
        let mut node = mk_node;
        while node != NONE_USIZE {
            let machine = ds.node_machine[node];
            critical_machine_hash ^= ((machine as u64).wrapping_add(1))
                .wrapping_mul(0x94D049BB133111EB)
                .rotate_left((critical_count % 63) + 1);
            let job = ds.node_job[node];
            let product = pre.job_products[job];
            let op_idx = ds.node_op[node];
            if pre.product_ops[product][op_idx].machines.len() > 1 {
                flexible_count = flexible_count.saturating_add(1);
            }
            critical_count = critical_count.saturating_add(1);
            node = buf.best_pred[node];
        }
        let flexibility_ratio = if critical_count == 0 {
            0
        } else {
            ((flexible_count.saturating_mul(256)) / critical_count).min(256) as u16
        };
        Ok(Some((mk, critical_machine_hash, assignment_hash, flexibility_ratio)))
    }

    let Some(incumbent_desc) = descriptor(pre, challenge, incumbent)? else {
        return Ok(Vec::new());
    };
    let quality_gap = ((pre.avg_op_min * 10.0).round() as u32)
        .max(20)
        .min(incumbent_mk / 10 + 1);
    let candidate_cap = pool.len().min(16);
    let mut candidates: Vec<(usize, (u32, u64, u64, u16))> = Vec::with_capacity(candidate_cap);
    for (idx, (sol, mk)) in pool.iter().take(candidate_cap).enumerate() {
        if *mk > incumbent_mk.saturating_add(quality_gap) {
            continue;
        }
        if let Some(desc) = descriptor(pre, challenge, sol)? {
            candidates.push((idx, desc));
        }
    }

    let mut selected: Vec<(u32, u64, u64, u16)> = vec![incumbent_desc];
    let mut result = Vec::with_capacity(max_additional);
    while result.len() < max_additional {
        let mut chosen: Option<(usize, (u32, u64, u64, u16), u32)> = None;
        for &(idx, desc) in &candidates {
            if result.contains(&idx) {
                continue;
            }
            let mut nearest = u32::MAX;
            for &other in &selected {
                let mk_part = desc.0.abs_diff(other.0).saturating_mul(8) / quality_gap.max(1);
                let machine_part = (desc.1 ^ other.1).count_ones();
                let assignment_part = (desc.2 ^ other.2).count_ones();
                let flex_part = desc.3.abs_diff(other.3) as u32 / 8;
                let distance = mk_part
                    .saturating_add(machine_part)
                    .saturating_add(assignment_part)
                    .saturating_add(flex_part);
                nearest = nearest.min(distance);
            }
            let replace = chosen.map_or(true, |(best_idx, best_desc, best_distance)| {
                nearest > best_distance
                    || (nearest == best_distance
                        && (desc.0 < best_desc.0
                            || (desc.0 == best_desc.0 && idx < best_idx)))
            });
            if replace {
                chosen = Some((idx, desc, nearest));
            }
        }
        let Some((idx, desc, _)) = chosen else {
            break;
        };
        result.push(idx);
        selected.push(desc);
    }
    Ok(result)
}

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _pre: &Pre,
    _effort: &EffortConfig,
) -> Result<()> {
    super::ember_medium::solve(challenge, save_solution)
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
        let mut vote_counts = vec![0usize; num_ops.saturating_mul(num_machines)];
        for (sol, _mk) in top_solutions.iter().take(pool_size) {
            if sol.job_schedule.len() <= job { continue; }
            let job_sched = &sol.job_schedule[job];
            for op_idx in 0..num_ops.min(job_sched.len()) {
                let (machine, _) = job_sched[op_idx];
                if machine < num_machines {
                    vote_counts[op_idx * num_machines + machine] += 1;
                }
            }
        }
        let product = pre.job_products[job];
        let mut choices: Vec<usize> = Vec::with_capacity(num_ops);
        for op_idx in 0..num_ops {
            let op_info = &pre.product_ops[product][op_idx];
            let mut best_machine = op_info.machines.first().map(|&(m, _)| m).unwrap_or(0);
            let mut best_votes = 0usize;
            let base = op_idx * num_machines;
            for &(m, _) in &op_info.machines {
                let cnt = if m < num_machines { vote_counts[base + m] } else { 0 };
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

fn apply_unified_reassign(
    pre: &Pre,
    challenge: &Challenge,
    start_sol: &Solution,
    start_mk: u32,
    best_mk: &mut u32,
    best_sol: &mut Option<Solution>,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    max_iters: usize,
) -> Result<()> {
    let mut current_sol = start_sol.clone();
    let mut current_mk = start_mk;

    let meaningful_abs = ((0.35 * pre.avg_op_min).round() as u32).max(2);
    let strong_abs = ((0.80 * pre.avg_op_min).round() as u32).max(4);

    let meaningful_gain = |before: u32, after: u32| -> bool {
        if after >= before { return false; }
        let gain = (before - after) as u64;
        gain >= meaningful_abs as u64 || gain.saturating_mul(1000) >= before as u64
    };
    let strong_gain = |before: u32, after: u32| -> bool {
        if after >= before { return false; }
        let gain = (before - after) as u64;
        gain >= strong_abs as u64 || gain.saturating_mul(500) >= before as u64
    };

    if max_iters > 0 {
        let before1 = current_mk;
        if let Some((s1, m1)) = unified_reassign_pass(pre, challenge, &current_sol)? {
            if m1 < before1 {
                current_sol = s1;
                current_mk = m1;

                if max_iters > 1 && meaningful_gain(before1, m1) {
                    let before2 = current_mk;
                    if let Some((s2, m2)) = unified_reassign_pass(pre, challenge, &current_sol)? {
                        if m2 < before2 {
                            current_sol = s2;
                            current_mk = m2;

                            if max_iters > 2 && (pre.high_flex > 0.58 || strong_gain(before2, m2)) {
                                let before3 = current_mk;
                                if let Some((s3, m3)) = unified_reassign_pass(pre, challenge, &current_sol)? {
                                    if m3 < before3 {
                                        current_sol = s3;
                                        current_mk = m3;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if current_mk < *best_mk {
        *best_mk = current_mk;
        *best_sol = Some(current_sol.clone());
        save_solution(&current_sol)?;
    }
    Ok(())
}