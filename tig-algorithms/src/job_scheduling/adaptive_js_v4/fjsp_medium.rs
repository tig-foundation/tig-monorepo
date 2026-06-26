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
            let mut top: Vec<Cand> = if k > 0 { Vec::with_capacity(k) } else { Vec::new() };

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

                for &(m, pt) in &op.machines {
                    if machine_avail[m] > time {
                        continue;
                    }
                    let end = time.saturating_add(pt);
                    if end != best_end {
                        continue;
                    }

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

                    let bal_pen = if chaotic_like && bal_w > 0.0 {
                        let denomw = (avg_work + (pre.avg_op_min * 3.0).max(1.0)).max(1.0);
                        let r = (machine_work[m] as f64) / denomw;
                        let done_n = (r / (r + 1.0)).clamp(0.0, 1.0);
                        -bal_w * done_n
                    } else {
                        0.0
                    };

                    let c = Cand { job, machine: m, pt, score: base + bal_pen };
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
        time = next_time.ok_or_else(|| anyhow!("Stalled"))?;
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
    let Some((mut current_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let initial_mk = current_mk;
    let mut improved = true;
    let mut passes = 0;
    let max_passes = 5;
    while improved && passes < max_passes {
        improved = false;
        passes += 1;
        let Some((_, mk_node)) = eval_disj(&ds, &mut buf) else { break };
        let mut crit_nodes: Vec<usize> = Vec::with_capacity(128);
        let mut u = mk_node;
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
                current_mk = best_mk; improved = true;
                continue 'node_loop;
            }
        }
    }
    if current_mk >= initial_mk { return Ok(None); }
    let Some((_, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let sol = disj_to_solution(pre, &ds, &buf.start)?;
    Ok(Some((sol, current_mk)))
}

fn greedy_reassign_pass(pre: &Pre, challenge: &Challenge, base_sol: &Solution) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n);
    let n = ds.n;

    let Some((mut current_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let initial_mk = current_mk;

    let mut passes = 0usize;
    let max_passes = 3;

    while passes < max_passes {
        passes += 1;

        let Some((cur_mk, _)) = eval_disj(&ds, &mut buf) else { break };
        current_mk = cur_mk;

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

                let old_pos = match ds.machine_seq[cur_machine].iter().position(|&x| x == node) {
                    Some(p) => p,
                    None => continue,
                };

                ds.machine_seq[cur_machine].remove(old_pos);
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
                    if let Some((test_mk, _)) = eval_disj(&ds, &mut buf) {
                        if test_mk < best_pass_mk {
                            best_pass_mk = test_mk;
                            best_pass_move = Some((node, new_m, new_pt, pos));
                        }
                    }
                    ds.machine_seq[new_m].remove(pos);
                }

                ds.machine_seq[cur_machine].insert(old_pos, node);
                ds.node_machine[node] = cur_machine;
                ds.node_pt[node] = cur_pt;
            }
        }

        let Some((node, best_m, best_pt, best_ins_pos)) = best_pass_move else { break };
        if best_pass_mk >= current_mk {
            break;
        }

        let cur_machine = ds.node_machine[node];
        let old_pos = ds.machine_seq[cur_machine].iter().position(|&x| x == node).unwrap();
        ds.machine_seq[cur_machine].remove(old_pos);
        let ins = best_ins_pos.min(ds.machine_seq[best_m].len());
        ds.machine_seq[best_m].insert(ins, node);
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
    if let Some(ref sol)=best_solution.clone(){if let Some((improved_sol,improved_mk))=greedy_reassign_pass(pre,challenge,sol)?{if improved_mk<best_makespan{best_makespan=improved_mk;best_solution=Some(improved_sol.clone());save_solution(&improved_sol)?;}}}

    if let Some(ref sol) = best_solution.clone() {
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

    if let Some(ref sol) = best_solution.clone() {
        if let Ok(Some((cb_sol, cb_mk))) = critical_block_move_local_search_ex(pre, challenge, sol, cb_passes, cb_iters, cb_no_improve) {
            if cb_mk < best_makespan {
                best_makespan = cb_mk;
                best_solution = Some(cb_sol.clone());
                save_solution(&cb_sol)?;
            }
        }
    }

    if let Some(ref sol) = best_solution.clone() {
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
        let Some(ref base) = ils_best_sol.clone() else { break };
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

    if let Some(ref sol) = best_solution.clone() {
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
            let Some(ref base) = alns_sa_sol.clone() else { break };
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

    if let Some(ref sol) = best_solution.clone() {
        if let Ok(Some((bmr_sol, bmr_mk))) = bottleneck_machine_relief_pass(pre, challenge, sol, 20) {
            if bmr_mk < best_makespan {
                best_makespan = bmr_mk;
                best_solution = Some(bmr_sol.clone());
                save_solution(&bmr_sol)?;
            }
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

    if let Some(ref sol) = best_solution.clone() {
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

    if let Some(ref sol) = best_solution.clone() {
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
            let Some(ref base) = final_ils_best_sol.clone() else { break };
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
    if let Some(ref sol) = best_solution.clone() {
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

    for iter in 0..max_iters {
        let mut machine_loads = vec![0u64; num_machines];
        for nd in 0..n {
            let m = ds.node_machine[nd];
            machine_loads[m] = machine_loads[m].saturating_add(ds.node_pt[nd] as u64);
        }

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
            let old_pos = ds.machine_seq[cur_machine].iter().position(|&x| x == best_node).unwrap();
            ds.machine_seq[cur_machine].remove(old_pos);
            let ins = best_ins_pos.min(ds.machine_seq[best_new_m].len());
            ds.machine_seq[best_new_m].insert(ins, best_node);
            ds.node_machine[best_node] = best_new_m;
            ds.node_pt[best_node] = best_new_pt;
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