use super::types::*;

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
