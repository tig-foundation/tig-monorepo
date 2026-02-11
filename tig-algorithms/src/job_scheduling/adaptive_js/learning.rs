use anyhow::{anyhow, Result};
use tig_challenges::job_scheduling::*;
use super::types::*;
use super::helpers::pt_from_op;

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
