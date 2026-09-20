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

#[inline(always)]
fn progress_mode_js(progress: f64) -> u8 {
    if progress < 0.34 { 0 } else if progress < 0.72 { 1 } else { 2 }
}

#[inline(always)]
fn js_cached_scar_n(pre: &Pre, machine: usize) -> f64 {
    pre.machine_scarcity.get(machine).copied().unwrap_or(0.0) / pre.avg_machine_scarcity.max(1e-9)
}

#[inline(always)]
fn js_cached_pop(pre: &Pre, machine: usize) -> f64 {
    pre.machine_best_pop.get(machine).copied().unwrap_or(0.0)
}

#[inline(always)]
fn js_cached_penalty(_machine: usize, machine_penalty: f64) -> f64 {
    machine_penalty.clamp(0.0, 1.0)
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
    let route_term = if route_w > 0.0 && op.flex >= 2 {
        let route_gain = (0.70 + 0.80*(1.0-progress)).clamp(0.70, 1.40);
        route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine)
    } else { 0.0 };
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
    let route_term = if route_w > 0.0 && op.flex >= 2 {
        let route_gain = (0.70 + 0.80*(1.0-progress)).clamp(0.70, 1.40);
        route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine)
    } else { 0.0 };
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
    let route_term = if route_w > 0.0 && op.flex >= 2 {
        let route_gain = (0.70 + 0.80*(1.0-progress)).clamp(0.70, 1.40);
        route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine)
    } else { 0.0 };
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
    let route_term = if route_w > 0.0 && op.flex >= 2 {
        let route_gain = (0.70 + 0.80*(1.0-progress)).clamp(0.70, 1.40);
        route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine)
    } else { 0.0 };
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
    let route_term = if route_w > 0.0 && op.flex >= 2 {
        let route_gain = (0.70 + 0.80*(1.0-progress)).clamp(0.70, 1.40);
        route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine)
    } else { 0.0 };
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
    let (cp_w, phase_reg_w, next_w, end_w, flow_gain, slack_w, proc_w, flex_w, scarce_w, scarcity_w, mpen_w) = match phase {
        0 => (
            1.00 + 0.18*js,
            0.32 + 0.28*js,
            0.17*(0.80 + 0.40*js),
            0.92 + 0.08*pre.high_flex,
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
    let route_term = if route_w > 0.0 && op.flex >= 2 {
        let route_gain = match phase { 0 => 1.24, 1 => 0.98, _ => 0.74 };
        route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine)
    } else { 0.0 };
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
    let (bn_w, end_w, phase_reg_w, load_w, density_w, next_w, flow_gain, slack_w, proc_w, mpen_w, scarcity_w) = match phase {
        0 => (
            (0.44 + 0.26*js)*pre.bn_focus,
            0.58,
            0.28 + 0.18*js,
            if pre.hi_flex { -0.45 } else { 0.65 + 0.18*js } * pre.flex_factor,
            0.12,
            0.16*(0.70 + 0.50*js),
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
    let route_term = if route_w > 0.0 && op.flex >= 2 {
        let route_gain = match phase { 0 => 1.26, 1 => 0.98, _ => 0.72 };
        route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine)
    } else { 0.0 };
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
    let (bn_w, end_w, phase_reg_w, load_w, density_w, next_w, flow_gain, slack_w, proc_w, mpen_w, scarcity_w) = match phase {
        0 => (
            (0.20 + 0.18*js)*pre.bn_focus,
            0.86*fl + 0.72*js + 0.12*pre.high_flex,
            0.24*fl + 0.46*js,
            load_sign*(0.60*fl + 0.92*js)*pre.flex_factor,
            0.05*fl + 0.12*js,
            0.16*(0.65*fl + 1.20*js),
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
    let route_term = if route_w > 0.0 && op.flex >= 2 {
        let route_gain = match phase { 0 => 1.28, 1 => 0.98, _ => 0.72 };
        route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine)
    } else { 0.0 };
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
    let route_term = if route_w > 0.0 && op.flex >= 2 {
        let route_gain = (0.70 + 0.80*(1.0-progress)).clamp(0.70, 1.40);
        route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine)
    } else { 0.0 };
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
    let beta = 1.0 + 3.0 * sharp;
    let n = top.len();
    let mut weights = [0.0f64; 16];
    let mut sum = 0.0;
    for i in 0..n {
        let w = flat + sharp * (-beta * (i as f64)).exp();
        weights[i] = w;
        sum += w;
    }
    let mut r = rng.gen::<f64>() * sum;
    for i in 0..n {
        r -= weights[i];
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
    let mut job_bias_cur: Vec<f64> = if USE_JB { vec![0.0; num_jobs] } else { Vec::new() };
    let job_bias = if USE_JB { job_bias.unwrap_or(&[]) } else { &[][..] };
    let machine_penalty = if USE_MP { machine_penalty.unwrap_or(&[]) } else { &[][..] };
    let route_pref = if USE_ROUTE { route_pref } else { None };
    let route_w = if USE_ROUTE { route_w } else { 0.0 };
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
                        if USE_JB { job_bias_cur[job] = job_bias.get(job).copied().unwrap_or(0.0); }
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
                    let product = job_product_cur[job];
                    let op = &pre.product_ops[product][op_idx];
                    let ops_rem = job_ops_rem[job];
                    let jb = if USE_JB { job_bias_cur[job] } else { 0.0 };
                    let tie_idle = op.flex >= 2 && job_best_cnt_idle[job] > 1;
                    let tie_frac = if tie_idle {
                        ((job_best_cnt_idle[job].saturating_sub(1) as f64) / (op.flex as f64).max(1.0)).clamp(0.0, 1.0)
                    } else { 0.0 };
                    let mp_eff = if USE_MP {
                        if tie_frac > 0.0 {
                            (machine_penalty.get(m).copied().unwrap_or(0.0) * (0.72 + 0.20*progress) + (0.10 + 0.16*(1.0-progress))*tie_frac).clamp(0.0, 1.0)
                        } else {
                            machine_penalty.get(m).copied().unwrap_or(0.0)
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
            let product = job_product_cur[job]; let op_idx = job_next_op[job]; let op = &pre.product_ops[product][op_idx];
            let best_end_now = job_best_end[job];
            let end_check = time.max(machine_avail[machine]).saturating_add(pt);
            if machine_avail[machine] > time || end_check != best_end_now { break; }
            let end_time = time.saturating_add(pt);
            job_schedule[job].push((machine, time)); job_next_op[job]+=1; job_ready_time[job]=end_time; machine_avail[machine]=end_time; remaining_ops-=1;
            job_gen[job] = job_gen[job].wrapping_add(1);
            let pos = idle_pos[machine];
            if pos != NONE_USIZE {
                let Some(last) = idle_machines.pop() else { return Err(anyhow!("Idle-machine state inconsistent")); };
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

#[inline]
fn take_adj_move(
    allowed: bool,
    adj_mk: u32,
    cand: (usize, usize, usize, u32),
    best_key: &mut u32,
    best: &mut Option<(usize, usize, usize, u32)>,
    tie_n: &mut u64,
    tseed: &mut u64,
) {
    if !allowed { return; }
    if adj_mk < *best_key {
        *best_key = adj_mk;
        *best = Some(cand);
        *tie_n = 1;
    } else if adj_mk == *best_key {
        *tie_n = tie_n.saturating_add(1);
        *tseed ^= tseed.wrapping_shl(13);
        *tseed ^= tseed.wrapping_shr(7);
        *tseed ^= tseed.wrapping_shl(17);
        if (*tseed % (*tie_n).max(1)) == 0 {
            *best = Some(cand);
        }
    }
}

fn consider_n5_block(
    m: usize,
    run_start: usize,
    run_end: usize,
    seq: &[usize],
    iter: usize,
    use_insertions: bool,
    best_global_mk: u32,
    diversify: f64,
    diversify_unit: u32,
    heads: &[u32],
    tails: &[u32],
    pt: &[u32],
    job_pred: &[usize],
    job_succ: &[usize],
    machine_pred: &[usize],
    machine_succ: &[usize],
    node_local: &[usize],
    pair_offsets: &[usize],
    tabu_expiry: &[usize],
    machine_last_pick: &[usize],
    best_move_key: &mut u32,
    best_move: &mut Option<(usize, usize, usize, u32)>,
    best_tie_n: &mut u64,
    relaxed_key: &mut u32,
    relaxed_move: &mut Option<(usize, usize, usize, u32)>,
    relaxed_tie_n: &mut u64,
    tseed: &mut u64,
) {
    let block_len = run_end - run_start + 1;
    if block_len < 2 { return; }
    let max_dist = if use_insertions { 4usize.min(block_len - 1) } else { 1 };
    for d in 1..=max_dist {
        let pos = run_start;
        let to = run_start + d;
        let node_u = seq[pos];
        let node_v = seq[to];
        let est_mk = if d == 1 {
            estimate_swap_mk(node_u, node_v, heads, tails, pt, job_pred, job_succ, machine_pred, machine_succ)
        } else {
            estimate_block_insertion_backward(pos, to, seq, heads, tails, pt, job_pred, job_succ, machine_pred, machine_succ)
        };
        let lu = node_local[node_u];
        let lv = node_local[node_v];
        let (a, b) = if lu < lv { (lu, lv) } else { (lv, lu) };
        let tabu_idx = pair_offsets[m] + b * (b - 1) / 2 + a;
        let is_tabu = tabu_expiry[tabu_idx] > iter;
        let aspiration = est_mk < best_global_mk;
        let age = iter.saturating_sub(machine_last_pick[m]).min(9) as u32;
        let block_bonus = block_len.saturating_sub(2).min(3) as u32;
        let div_bonus = if diversify > 0.0 {
            ((diversify_unit as f64) * diversify * (0.35 * (age as f64) + 0.75 * (block_bonus as f64)) / 3.0) as u32
        } else { 0 };
        let adj_mk = est_mk.saturating_sub(div_bonus);
        take_adj_move(!is_tabu || aspiration, adj_mk, (m, pos, to, est_mk), best_move_key, best_move, best_tie_n, tseed);
        take_adj_move(true, adj_mk, (m, pos, to, est_mk), relaxed_key, relaxed_move, relaxed_tie_n, tseed);

        if block_len > 2 {
            let pos = run_end;
            let to = run_end - d;
            let node_u = seq[pos];
            let node_v = seq[to];
            let est_mk = if d == 1 {
                estimate_swap_mk(node_v, node_u, heads, tails, pt, job_pred, job_succ, machine_pred, machine_succ)
            } else {
                estimate_block_insertion_forward(to, pos, seq, heads, tails, pt, job_pred, job_succ, machine_pred, machine_succ)
            };
            let lu = node_local[node_u];
            let lv = node_local[node_v];
            let (a, b) = if lu < lv { (lu, lv) } else { (lv, lu) };
            let tabu_idx = pair_offsets[m] + b * (b - 1) / 2 + a;
            let is_tabu = tabu_expiry[tabu_idx] > iter;
            let aspiration = est_mk < best_global_mk;
            let age = iter.saturating_sub(machine_last_pick[m]).min(9) as u32;
            let block_bonus = block_len.saturating_sub(2).min(3) as u32;
            let div_bonus = if diversify > 0.0 {
                ((diversify_unit as f64) * diversify * (0.35 * (age as f64) + 0.75 * (block_bonus as f64)) / 3.0) as u32
            } else { 0 };
            let adj_mk = est_mk.saturating_sub(div_bonus);
            take_adj_move(!is_tabu || aspiration, adj_mk, (m, pos, to, est_mk), best_move_key, best_move, best_tie_n, tseed);
            take_adj_move(true, adj_mk, (m, pos, to, est_mk), relaxed_key, relaxed_move, relaxed_tie_n, tseed);
        }
    }
}

#[allow(dead_code)]
fn sbp_try_reorder_machine(
    ds: &mut DisjSchedule,
    buf: &mut EvalBuf,
    m: usize,
    cur_mk: u32,
    b: &mut SbpBuf,
) -> Option<(u32, usize)> {
    let k = ds.machine_seq[m].len();
    if k < 3 { return None; }
    if !sbp_heads_tails_excl(ds, m, b) { return None; }
    b.nodes.clear();
    b.nodes.extend_from_slice(&ds.machine_seq[m]);
    b.r.clear();
    b.p.clear();
    b.q.clear();
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
    let val = sbp_carlier(&r, &p, &q, b)?;
    if val >= cur_val { return None; }
    b.saved_seq.clear();
    b.saved_seq.extend_from_slice(&ds.machine_seq[m]);
    for pos in 0..k { ds.machine_seq[m][pos] = b.nodes[b.best_order[pos]]; }
    match eval_disj(ds, buf) {
        Some((new_mk, new_node)) if new_mk < cur_mk => Some((new_mk, new_node)),
        _ => {
            ds.machine_seq[m].clear();
            ds.machine_seq[m].extend_from_slice(&b.saved_seq);
            None
        }
    }
}

fn restore_tabu_best(
    ds: &mut DisjSchedule,
    best_seq: &[Vec<usize>],
    current_pos: &mut [usize],
    node_machine: &mut [usize],
    machine_pred_node: &mut [usize],
    buf: &mut EvalBuf,
) -> Option<(u32, usize)> {
    for (dst, src) in ds.machine_seq.iter_mut().zip(best_seq.iter()) {
        dst.clear();
        dst.extend_from_slice(src);
    }
    rebuild_machine_pred_nodes(ds, machine_pred_node);
    for m in 0..ds.num_machines {
        for (i, &node) in ds.machine_seq[m].iter().enumerate() {
            current_pos[node] = i;
            node_machine[node] = m;
        }
    }
    eval_disj(ds, buf)
}

fn tabu_search_phase(pre: &Pre, challenge: &Challenge, base_sol: &Solution, max_iterations: usize, tenure_base: usize) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n); let n = ds.n;
    let Some((initial_mk, mut mk_node)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let mut cur_mk = initial_mk; let mut best_global_mk = initial_mk; let mut best_global_machine_seq = ds.machine_seq.clone();
    let tenure = tenure_base.max(5); let tenure_delta = (tenure/3).max(2);
    let max_no_improve = (max_iterations / 2).max(60);
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
    let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (initial_mk as u64).wrapping_shl(16) ^ (n as u64).wrapping_mul(0x517CC1B727220A95);
    let mut tseed: u64 = ((challenge.seed[0] as u64).wrapping_mul(0xD1B54A32D192ED03)
        ^ (initial_mk as u64).wrapping_mul(0xA24BAED4963EE407)
        ^ (n as u64).wrapping_shl(32)) | 1;
    let mut tail = vec![0u32; n]; let mut back_deg = vec![0u16; n]; let mut back_stack: Vec<usize> = Vec::with_capacity(n);
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
    let kick_step = (max_no_improve / 4).max(1); let diversify_threshold = (max_no_improve/3).max(20); let diversify_unit = (pre.avg_op_min * (0.10 + 0.16*pre.jobshopness + 0.06*pre.high_flex)).max(1.0) as u32; let mut kicks_left = 4usize;
    for iter in 0..max_iterations {
        if no_improve >= max_no_improve {
            if kicks_left == 0 { break; }
            no_improve = 0; kicks_left -= 1; tabu_expiry.fill(0);
            let Some((mk, node)) = restore_tabu_best(&mut ds, &best_global_machine_seq, &mut current_pos, &mut node_machine, &mut machine_pred_node, &mut buf) else { break };
            cur_mk = mk; mk_node = node;
            continue;
        }
        let kick_ord = if kick_step > 0 { no_improve / kick_step } else { 0 };
        if no_improve > 0 && no_improve % kick_step == 0 && kicks_left > 0 && kick_ord <= 4 {
            let use_base_first = kick_ord % 2 == 1;
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
                let num_kicks = (2 + 1).min(4);
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
            no_improve += 1;
            rebuild_machine_pred_nodes(&ds, &mut machine_pred_node);
            match eval_disj(&ds, &mut buf) {
                Some((mk, node)) => { cur_mk = mk; mk_node = node; }
                None => {
                    match restore_tabu_best(&mut ds, &best_global_machine_seq, &mut current_pos, &mut node_machine, &mut machine_pred_node, &mut buf) {
                        Some((mk, node)) => { cur_mk = mk; mk_node = node; }
                        None => break,
                    }
                }
            }
            continue;
        }
        if iter > 0 { if cur_mk < best_global_mk { best_global_mk = cur_mk; best_global_machine_seq.clone_from(&ds.machine_seq); rebuild_machine_pos_map(&best_global_machine_seq, &mut best_global_pos); no_improve = 0; } else { no_improve += 1; } }

        tail.fill(0);
        back_deg.copy_from_slice(&job_back_deg);
        for i in 0..n { if buf.machine_succ[i] != NONE_USIZE { back_deg[i] += 1; } }
        back_stack.clear(); for i in 0..n { if back_deg[i] == 0 { back_stack.push(i); } }
        while let Some(nd) = back_stack.pop() {
            let contrib = ds.node_pt[nd].saturating_add(tail[nd]);
            let jp = job_pred_node[nd]; if jp != NONE_USIZE { if contrib > tail[jp] { tail[jp] = contrib; } back_deg[jp] = back_deg[jp].saturating_sub(1); if back_deg[jp] == 0 { back_stack.push(jp); } }
            let mp = machine_pred_node[nd]; if mp != NONE_USIZE { if contrib > tail[mp] { tail[mp] = contrib; } back_deg[mp] = back_deg[mp].saturating_sub(1); if back_deg[mp] == 0 { back_stack.push(mp); } }
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
        crit_pos_machines.sort_unstable();
        let diversify = ((no_improve.saturating_sub(diversify_threshold) as f64) / (max_no_improve.saturating_sub(diversify_threshold).max(1) as f64)).clamp(0.0, 1.0);
        let mut best_move: Option<(usize,usize,usize,u32)> = None; let mut best_move_key = u32::MAX;
        let mut relaxed_move: Option<(usize,usize,usize,u32)> = None; let mut relaxed_key = u32::MAX;
        let mut best_tie_n = 0u64; let mut relaxed_tie_n = 0u64;
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
                        consider_n5_block(
                            m, run_start, run_end, seq, iter, n <= 5000 || no_improve >= diversify_threshold / 2, best_global_mk, diversify, diversify_unit,
                            &buf.start, &tail, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ,
                            &node_local, &pair_offsets, &tabu_expiry, &machine_last_pick,
                            &mut best_move_key, &mut best_move, &mut best_tie_n,
                            &mut relaxed_key, &mut relaxed_move, &mut relaxed_tie_n, &mut tseed,
                        );
                    }
                    run_start = pos;
                    run_end = pos;
                }
                prev_pos = pos;
                prev_node = node;
            }
            if run_end > run_start {
                consider_n5_block(
                    m, run_start, run_end, seq, iter, n <= 5000 || no_improve >= diversify_threshold / 2, best_global_mk, diversify, diversify_unit,
                    &buf.start, &tail, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ,
                    &node_local, &pair_offsets, &tabu_expiry, &machine_last_pick,
                    &mut best_move_key, &mut best_move, &mut best_tie_n,
                    &mut relaxed_key, &mut relaxed_move, &mut relaxed_tie_n, &mut tseed,
                );
            }
        }
        let chosen = best_move.or(relaxed_move);
        match chosen {
            Some((m,from,to,_est)) => {
                let node_a=ds.machine_seq[m][from]; let node_b=ds.machine_seq[m][to];
                relocate_machine_seq(&mut ds.machine_seq[m], from, to);
                let (lo, hi) = if from < to { (from, to) } else { (to, from) };
                for i in lo..=hi { current_pos[ds.machine_seq[m][i]] = i; }
                machine_last_pick[m] = iter;
                let seq = &ds.machine_seq[m];
                let prev = if lo > 0 { seq[lo - 1] } else { NONE_USIZE };
                machine_pred_node[seq[lo]] = prev;
                for i in (lo + 1)..=hi { machine_pred_node[seq[i]] = seq[i - 1]; }
                if hi + 1 < seq.len() { machine_pred_node[seq[hi + 1]] = seq[hi]; }
                pseed^=pseed.wrapping_shl(13); pseed^=pseed.wrapping_shr(7); pseed^=pseed.wrapping_shl(17);
                let offset=(pseed%((2*tenure_delta+1) as u64)) as usize;
                let progress=(iter as f64)/(max_iterations as f64); let late_bonus=if progress>0.6{((progress-0.6)*10.0) as usize}else{0};
                let this_tenure=(tenure+offset+late_bonus).saturating_sub(tenure_delta);
                let la = node_local[node_a]; let lb = node_local[node_b];
                let (a, b) = if la < lb { (la, lb) } else { (lb, la) };
                let tabu_idx = pair_offsets[m] + b * (b - 1) / 2 + a;
                tabu_expiry[tabu_idx] = iter + this_tenure;
                match eval_disj(&ds, &mut buf) {
                    Some((mk, node)) => { cur_mk = mk; mk_node = node; }
                    None => {
                        match restore_tabu_best(&mut ds, &best_global_machine_seq, &mut current_pos, &mut node_machine, &mut machine_pred_node, &mut buf) {
                            Some((mk, node)) => { cur_mk = mk; mk_node = node; }
                            None => break,
                        }
                    }
                }
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
fn estimate_block_insertion_forward(
    run_start: usize, run_end: usize, seq: &[usize], 
    heads: &[u32], tails: &[u32], pt: &[u32], 
    job_pred: &[usize], job_succ: &[usize], 
    machine_pred: &[usize], machine_succ: &[usize]
) -> u32 {
    let v = seq[run_end];
    let u = seq[run_start];
    let w = seq[run_end - 1];
    let jp_v = job_pred[v];
    let r_jp_v = if jp_v != NONE_USIZE { heads[jp_v].saturating_add(pt[jp_v]) } else { 0 };
    let mp_u = machine_pred[u];
    let r_mp_u = if mp_u != NONE_USIZE { heads[mp_u].saturating_add(pt[mp_u]) } else { 0 };
    let new_r_v = r_jp_v.max(r_mp_u);
    let js_v = job_succ[v];
    let q_js_v = if js_v != NONE_USIZE { pt[js_v].saturating_add(tails[js_v]) } else { 0 };
    let jp_u = job_pred[u];
    let r_jp_u = if jp_u != NONE_USIZE { heads[jp_u].saturating_add(pt[jp_u]) } else { 0 };
    let new_r_u = r_jp_u.max(new_r_v.saturating_add(pt[v]));
    let js_u = job_succ[u];
    let q_js_u = if js_u != NONE_USIZE { pt[js_u].saturating_add(tails[js_u]) } else { 0 };
    let js_w = job_succ[w];
    let q_js_w = if js_w != NONE_USIZE { pt[js_w].saturating_add(tails[js_w]) } else { 0 };
    let ms_v = machine_succ[v];
    let q_ms_v = if ms_v != NONE_USIZE { pt[ms_v].saturating_add(tails[ms_v]) } else { 0 };
    let new_q_w = q_js_w.max(q_ms_v);
    let mut pt_sum = 0;
    for i in run_start..=run_end { pt_sum += pt[seq[i]]; }
    let bound_v = new_r_v.saturating_add(pt[v]).saturating_add(q_js_v);
    let bound_block = new_r_v.saturating_add(pt_sum).saturating_add(new_q_w);
    let bound_u = new_r_u.saturating_add(pt[u]).saturating_add(q_js_u);
    bound_v.max(bound_block).max(bound_u)
}

#[inline]
fn estimate_block_insertion_backward(
    run_start: usize, run_end: usize, seq: &[usize], 
    heads: &[u32], tails: &[u32], pt: &[u32], 
    job_pred: &[usize], job_succ: &[usize], 
    machine_pred: &[usize], machine_succ: &[usize]
) -> u32 {
    let u = seq[run_start];
    let x = seq[run_start + 1];
    let v = seq[run_end];
    let jp_x = job_pred[x];
    let r_jp_x = if jp_x != NONE_USIZE { heads[jp_x].saturating_add(pt[jp_x]) } else { 0 };
    let mp_u = machine_pred[u];
    let r_mp_u = if mp_u != NONE_USIZE { heads[mp_u].saturating_add(pt[mp_u]) } else { 0 };
    let new_r_x = r_jp_x.max(r_mp_u);
    let jp_u = job_pred[u];
    let r_jp_u = if jp_u != NONE_USIZE { heads[jp_u].saturating_add(pt[jp_u]) } else { 0 };
    let js_u = job_succ[u];
    let q_js_u = if js_u != NONE_USIZE { pt[js_u].saturating_add(tails[js_u]) } else { 0 };
    let ms_v = machine_succ[v];
    let q_ms_v = if ms_v != NONE_USIZE { pt[ms_v].saturating_add(tails[ms_v]) } else { 0 };
    let new_q_u = q_js_u.max(q_ms_v);
    let mut pt_sum = 0;
    for i in run_start..=run_end { pt_sum += pt[seq[i]]; }
    let jp_v = job_pred[v];
    let r_jp_v = if jp_v != NONE_USIZE { heads[jp_v].saturating_add(pt[jp_v]) } else { 0 };
    let js_v = job_succ[v];
    let q_js_v = if js_v != NONE_USIZE { pt[js_v].saturating_add(tails[js_v]) } else { 0 };
    let new_q_v = q_js_v.max(pt[u].saturating_add(new_q_u));
    let bound_u = r_jp_u.saturating_add(pt[u]).saturating_add(new_q_u);
    let bound_block = new_r_x.saturating_add(pt_sum).saturating_add(new_q_u);
    let bound_v = r_jp_v.saturating_add(pt[v]).saturating_add(new_q_v);
    bound_u.max(bound_block).max(bound_v)
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
fn relocate_machine_seq(seq: &mut [usize], from: usize, to: usize) {
    if from < to {
        seq[from..=to].rotate_left(1);
    } else if to < from {
        seq[to..=from].rotate_right(1);
    }
}

#[allow(dead_code)]
#[inline]
fn inherit_machine_consensus_order(
    child_seq: &mut Vec<usize>,
    better_seq: &[usize],
    _worse_seq: &[usize],
    elite_machine_seqs: &[&[usize]],
    pos_buf: &mut [usize],
    cross_votes: &mut Vec<u8>,
) {
    let l = better_seq.len();
    if l <= 1 || l != _worse_seq.len() {
        if child_seq.len() != l {
            child_seq.clear();
            child_seq.extend_from_slice(better_seq);
        }
        return;
    }
    if child_seq.len() != l {
        child_seq.clear();
        child_seq.extend_from_slice(better_seq);
    }

    for (idx, &node) in better_seq.iter().enumerate() {
        pos_buf[node] = idx;
    }

    let mat_size = l.saturating_mul(l);
    if cross_votes.len() < mat_size {
        cross_votes.resize(mat_size, 0);
    }
    cross_votes[..mat_size].fill(0);

    for &eseq in elite_machine_seqs {
        for i in 0..eseq.len() {
            let u = eseq[i];
            let u_idx = pos_buf[u];
            if u_idx == NONE_USIZE { continue; }
            for j in (i + 1)..eseq.len() {
                let v = eseq[j];
                let v_idx = pos_buf[v];
                if v_idx != NONE_USIZE {
                    cross_votes[u_idx * l + v_idx] += 1;
                }
            }
        }
    }

    for i in 1..l {
        let mut j = i;
        while j > 0 {
            let u = child_seq[j - 1];
            let v = child_seq[j];
            let u_idx = pos_buf[u];
            let v_idx = pos_buf[v];
            
            let u_before_v = cross_votes[u_idx * l + v_idx];
            let v_before_u = cross_votes[v_idx * l + u_idx];
            
            if v_before_u > u_before_v {
                child_seq.swap(j - 1, j);
                j -= 1;
            } else {
                break;
            }
        }
    }

    for &node in better_seq {
        pos_buf[node] = NONE_USIZE;
    }
}

#[inline]
fn invert_job_bias_guidance(jb: &[f64]) -> Vec<f64> {
    if jb.is_empty() { return Vec::new(); }
    let mean = jb.iter().copied().sum::<f64>() / (jb.len() as f64);
    let mut anti = Vec::with_capacity(jb.len());
    for &v in jb {
        let diff = mean - v;
        anti.push((diff * 0.75 * (1.0 + diff.abs())).clamp(-3.0, 3.0));
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

#[allow(dead_code)]
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
    let mut test_buf_b: Option<EvalBuf> = None;
    let mut cur_mk = initial_mk;
    let mut tail = vec![0u32; n];
    let mut back_deg = vec![0u16; n];
    let mut back_stack: Vec<usize> = Vec::with_capacity(n);
    let mut machine_pred_node = vec![NONE_USIZE; n];
    let mut job_pred_node = vec![NONE_USIZE; n];
    let mut job_back_deg = vec![0u16; n];
    let mut moves: Vec<(u32,u16,usize,usize,usize)> = Vec::with_capacity(64);
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
    for i in 0..n {
        if ds.job_succ[i] != NONE_USIZE { job_back_deg[i] += 1; }
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
        back_deg.copy_from_slice(&job_back_deg);
        for i in 0..n {
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
                        let max_dist = 4usize.min(block_len - 1);
                        for d in 1..=max_dist {
                            let to_fwd = run_start + d;
                            let est_bwd = if d == 1 {
                                estimate_swap_mk(seq[run_start], seq[to_fwd], &buf.start, &tail, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ)
                            } else {
                                estimate_block_insertion_backward(run_start, to_fwd, seq, &buf.start, &tail, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ)
                            };
                            if est_bwd < cur_mk { moves.push((est_bwd, d as u16, m, run_start, to_fwd)); }
                            
                            if block_len > 2 {
                                let to_bwd = run_end - d;
                                let est_fwd = if d == 1 {
                                    estimate_swap_mk(seq[to_bwd], seq[run_end], &buf.start, &tail, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ)
                                } else {
                                    estimate_block_insertion_forward(to_bwd, run_end, seq, &buf.start, &tail, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ)
                                };
                                if est_fwd < cur_mk { moves.push((est_fwd, d as u16, m, run_end, to_bwd)); }
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
                let max_dist = 4usize.min(block_len - 1);
                for d in 1..=max_dist {
                    let to_fwd = run_start + d;
                    let est_bwd = if d == 1 {
                        estimate_swap_mk(seq[run_start], seq[to_fwd], &buf.start, &tail, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ)
                    } else {
                        estimate_block_insertion_backward(run_start, to_fwd, seq, &buf.start, &tail, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ)
                    };
                    if est_bwd < cur_mk { moves.push((est_bwd, d as u16, m, run_start, to_fwd)); }
                    
                    if block_len > 2 {
                        let to_bwd = run_end - d;
                        let est_fwd = if d == 1 {
                            estimate_swap_mk(seq[to_bwd], seq[run_end], &buf.start, &tail, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ)
                        } else {
                            estimate_block_insertion_forward(to_bwd, run_end, seq, &buf.start, &tail, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ)
                        };
                        if est_fwd < cur_mk { moves.push((est_fwd, d as u16, m, run_end, to_bwd)); }
                    }
                }
            }
        }

        if moves.is_empty() { break; }
        if moves.len() > 1 {
            moves.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)).then_with(|| a.2.cmp(&b.2)).then_with(|| a.3.cmp(&b.3)).then_with(|| a.4.cmp(&b.4)));
        }

        let eval_cap = moves.len().min(2);
        let mut best_idx = NONE_USIZE;
        let mut best_actual_mk = cur_mk;
        let mut best_actual_node = NONE_USIZE;
        let mut tested = [(NONE_USIZE, NONE_USIZE, NONE_USIZE); 2];

        for idx in 0..eval_cap {
            let (_, _, m, from, to) = moves[idx];
            tested[idx] = (m, from, to);
            {
                let seq = &mut ds.machine_seq[m];
                relocate_machine_seq(seq, from, to);
            }
            let res = if idx == 0 {
                eval_disj(ds, &mut test_buf_a)
            } else {
                let tb = test_buf_b.get_or_insert_with(|| EvalBuf::new(n));
                eval_disj(ds, tb)
            };
            if let Some((new_mk, new_node)) = res {
                if new_mk < best_actual_mk {
                    best_actual_mk = new_mk;
                    best_actual_node = new_node;
                    best_idx = idx;
                }
            }
            {
                let seq = &mut ds.machine_seq[m];
                relocate_machine_seq(seq, to, from);
            }
        }

        if best_idx != NONE_USIZE {
            let (m, from, to) = tested[best_idx];
            {
                let seq = &mut ds.machine_seq[m];
                relocate_machine_seq(seq, from, to);
                let (lo, hi) = if from < to { (from, to) } else { (to, from) };
                for i in lo..=hi { current_pos[seq[i]] = i; }
            }
            cur_mk = best_actual_mk;
            mk_node = best_actual_node;
            if best_idx == 0 {
                core::mem::swap(buf, &mut test_buf_a);
            } else {
                core::mem::swap(buf, test_buf_b.as_mut().unwrap());
            }
            stalled = 0;
        } else {
            stalled += 1;
            if stalled >= stall_cap { break; }
        }
    }

    if cur_mk < initial_mk { Some(cur_mk) } else { None }
}

#[allow(dead_code)]
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

        let mut best_m = NONE_USIZE;
        let mut max_crit_ops = 0;
        for &m in &crit_pos_machines {
            let count = crit_pos_by_machine[m].len();
            if count > max_crit_ops && ds.machine_seq[m].len() > 1 {
                max_crit_ops = count;
                best_m = m;
            }
        }

        let Some(m) = (if best_m != NONE_USIZE { Some(best_m) } else { None }) else { break; };

        let old_seq = ds.machine_seq[m].clone();
        let mut extracted: Vec<usize> = old_seq.clone();
        
        extracted.sort_unstable_by_key(|&node| {
            let pred_node = if ds.node_op[node] > 0 { node - 1 } else { NONE_USIZE };
            let r_jp = if pred_node != NONE_USIZE { buf.start[pred_node].saturating_add(ds.node_pt[pred_node]) } else { 0 };
            (r_jp, node)
        });
        
        ds.machine_seq[m] = extracted;

        match eval_disj(&ds, &mut buf) {
            Some((new_mk, new_node)) => {
                cur_mk = new_mk;
                mk_node = new_node;
                moved += 1;
            }
            None => {
                ds.machine_seq[m] = old_seq;
                let Some((restored_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
                cur_mk = restored_mk;
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
const SBP_REOPT_ROUNDS: usize = 2;

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
    let (start_mk, mk_node) = eval_disj(&ds, &mut buf)?;
    let mut cur_mk = start_mk;
    let mut b = SbpBuf::new(ds.n);

    let num_machines = challenge.num_machines.min(ds.machine_seq.len());
    let mut m_order: Vec<usize> = (0..num_machines).filter(|&m| ds.machine_seq[m].len() >= 3).collect();
    let mut load: Vec<u64> = vec![0u64; num_machines];
    let mut critical_load: Vec<u64> = vec![0u64; num_machines];
    for m in 0..num_machines {
        for &nd in &ds.machine_seq[m] { load[m] = load[m].saturating_add(ds.node_pt[nd] as u64); }
    }
    let mut node = mk_node;
    while node != NONE_USIZE {
        let m = ds.node_machine[node];
        if m < num_machines {
            critical_load[m] = critical_load[m].saturating_add(ds.node_pt[node] as u64);
        }
        node = buf.best_pred[node];
    }
    m_order.sort_by(|&x, &y| {
        critical_load[y].cmp(&critical_load[x])
            .then_with(|| load[y].cmp(&load[x]))
            .then(x.cmp(&y))
    });
    let machine_cap = if ds.n <= 3000 { 8 } else if ds.n <= 8000 { 6 } else { 4 };
    m_order.truncate(machine_cap);
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

fn apply_sbp_incumbent(
    pre: &Pre,
    challenge: &Challenge,
    best_solution: &mut Option<Solution>,
    best_makespan: &mut u32,
    top_solutions: &mut Vec<(Solution, u32)>,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<bool> {
    let Some(cur) = best_solution.as_ref() else { return Ok(false) };
    let Some((sbp_sol, sbp_mk)) = sbp_reoptimize(pre, challenge, cur, SBP_REOPT_ROUNDS) else {
        return Ok(false);
    };
    if sbp_mk >= *best_makespan { return Ok(false); }
    *best_makespan = sbp_mk;
    *best_solution = Some(sbp_sol.clone());
    save_solution(&sbp_sol)?;
    push_top_solutions(top_solutions, &sbp_sol, sbp_mk, 20);
    Ok(true)
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

fn pick_competitive_alternative_seed(
    pre: &Pre,
    num_machines: usize,
    top: &[(Solution, u32)],
) -> Option<usize> {
    if top.len() < 2 { return None; }
    let best_mk = top[0].1;
    let quality_band = ((best_mk as f64) * (0.008 + 0.012 * pre.high_flex + 0.008 * pre.jobshopness))
        .max(pre.avg_op_min * 2.0) as u32;
    let keys: Vec<Vec<u32>> = top.iter().map(|(sol, _)| ts_seed_key(pre, num_machines, sol)).collect();
    let min_distance = ((keys[0].len() / 50).max(4)) as u32;
    let mut chosen: Option<usize> = None;
    let mut chosen_distance = 0u32;
    for i in 1..top.len() {
        if top[i].1 > best_mk.saturating_add(quality_band) { continue; }
        let distance = ts_seed_key_distance(&keys[0], &keys[i]);
        if distance < min_distance { continue; }
        let better = match chosen {
            None => true,
            Some(previous) => {
                distance > chosen_distance
                    || (distance == chosen_distance && top[i].1 < top[previous].1)
            }
        };
        if better {
            chosen = Some(i);
            chosen_distance = distance;
        }
    }
    chosen
}

fn pick_diverse_ts_seeds(
    pre: &Pre,
    num_machines: usize,
    top: &[(Solution, u32)],
    n_best: usize,
    n_diverse: usize,
) -> Vec<usize> {
    let pool_len = top.len();
    if pool_len == 0 { return Vec::new(); }
    let n_best = n_best.min(pool_len);
    let mut picked: Vec<usize> = (0..n_best).collect();
    let want = (n_best + n_diverse).min(pool_len);
    if picked.len() >= want { return picked; }
    let keys: Vec<Vec<u32>> = top.iter().map(|(s, _)| ts_seed_key(pre, num_machines, s)).collect();
    let mut taken = vec![false; pool_len];
    for &i in &picked { taken[i] = true; }
    let mut min_d = vec![u32::MAX; pool_len];
    for i in 0..pool_len {
        if taken[i] { continue; }
        for &p in &picked {
            let d = ts_seed_key_distance(&keys[p], &keys[i]);
            if d < min_d[i] { min_d[i] = d; }
        }
    }
    while picked.len() < want {
        let mut best: Option<usize> = None;
        let mut best_d = 0u32;
        for i in 0..pool_len {
            if taken[i] { continue; }
            if best.is_none() || min_d[i] > best_d {
                best = Some(i);
                best_d = min_d[i];
            }
        }
        let Some(b) = best else { break };
        picked.push(b);
        taken[b] = true;
        for i in 0..pool_len {
            if taken[i] { continue; }
            let d = ts_seed_key_distance(&keys[b], &keys[i]);
            if d < min_d[i] { min_d[i] = d; }
        }
    }
    picked
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

    let ts_iters = effort.job_shop_iters;
    let ts_tenure = ((pre.total_ops as f64).sqrt() as usize * (100 + (pre.load_cv * 60.0) as usize) / 100).clamp(8, 24);
    let mut ts_seed_idx = pick_diverse_ts_seeds(pre, challenge.num_machines, &top_solutions, 2, 3);
    if pre.total_ops <= 5000 {
        if let Some(idx) = pick_competitive_alternative_seed(pre, challenge.num_machines, &top_solutions) {
            if !ts_seed_idx.contains(&idx) {
                ts_seed_idx.push(idx);
            }
        }
    }
    {
        let mut ts_results: Vec<(Solution, u32)> = Vec::new();
        for &idx in &ts_seed_idx {
            let res = {
                let base_sol = &top_solutions[idx].0;
                tabu_search_phase(pre, challenge, base_sol, ts_iters, ts_tenure)?
            };
            if let Some((sol2, mk2)) = res {
                if mk2 < best_makespan { best_makespan=mk2; best_solution=Some(sol2.clone()); save_solution(&sol2)?; }
                ts_results.push((sol2, mk2));
            }
        }
        let mut sbp_seeds: Vec<(Solution, u32)> = Vec::new();
        for cycles in [2usize, 5] {
            if let Some((seed_sol, seed_mk)) = sbp_construct_seed(pre, challenge, &greedy_sol, true, cycles) {
                if seed_mk < best_makespan {
                    best_makespan = seed_mk;
                    best_solution = Some(seed_sol.clone());
                    save_solution(&seed_sol)?;
                }
                push_top_solutions(&mut top_solutions, &seed_sol, seed_mk, 20);
                if let Some((_, prev_mk)) = sbp_seeds.last() {
                    if seed_mk == *prev_mk {
                        let k0 = ts_seed_key(pre, challenge.num_machines, &sbp_seeds[0].0);
                        let k1 = ts_seed_key(pre, challenge.num_machines, &seed_sol);
                        if ts_seed_key_distance(&k0, &k1) == 0 { continue; }
                    }
                }
                sbp_seeds.push((seed_sol, seed_mk));
            }
        }
        let sbp_iters = ts_iters.saturating_mul(4);
        for (seed_sol, _) in &sbp_seeds {
            if let Some((sol2, mk2)) = tabu_search_phase(pre, challenge, seed_sol, sbp_iters, ts_tenure)? {
                if mk2 < best_makespan { best_makespan=mk2; best_solution=Some(sol2.clone()); save_solution(&sol2)?; }
                ts_results.push((sol2, mk2));
            }
        }
        for (sol2, mk2) in ts_results {
            push_top_solutions(&mut top_solutions, &sol2, mk2, 20);
        }
    }
    apply_sbp_incumbent(pre, challenge, &mut best_solution, &mut best_makespan, &mut top_solutions, save_solution)?;

    if let Some(final_best) = best_solution.as_ref() {
        if let Some((sol4, mk4)) = tabu_search_phase(pre, challenge, final_best, ts_iters, ts_tenure)? {
            if mk4 < best_makespan {
                best_makespan = mk4;
                best_solution = Some(sol4.clone());
                save_solution(&sol4)?;
            }
        }
    }
    if apply_sbp_incumbent(pre, challenge, &mut best_solution, &mut best_makespan, &mut top_solutions, save_solution)? {
        if let Some(cur) = best_solution.as_ref() {
            if let Some((sol5, mk5)) = tabu_search_phase(pre, challenge, cur, ts_iters / 3, ts_tenure)? {
                if mk5 < best_makespan {
                    best_makespan = mk5;
                    best_solution = Some(sol5.clone());
                    save_solution(&sol5)?;
                    push_top_solutions(&mut top_solutions, &sol5, mk5, 20);
                }
            }
        }
        apply_sbp_incumbent(pre, challenge, &mut best_solution, &mut best_makespan, &mut top_solutions, save_solution)?;
    }

    if pre.total_ops <= 3000 && challenge.num_machines <= 80 {
        if let Some(cur) = best_solution.as_ref() {
            if let Some((sol6, mk6)) = tabu_search_phase(pre, challenge, cur, ts_iters.saturating_mul(8), ts_tenure)? {
                if mk6 < best_makespan {
                    best_makespan = mk6;
                    best_solution = Some(sol6.clone());
                    save_solution(&sol6)?;
                    push_top_solutions(&mut top_solutions, &sol6, mk6, 20);
                }
            }
        }
        apply_sbp_incumbent(pre, challenge, &mut best_solution, &mut best_makespan, &mut top_solutions, save_solution)?;
    }

    if let Some(sol) = best_solution { save_solution(&sol)?; }
    Ok(())
}