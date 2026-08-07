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
                    // E45: filled by the memo pass at the end of `build_pre`, once
                    // `avg_op_min` is final.
                    flex_inv: 0.0,
                    pt_n: Vec::new(),
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
                // E45: filled by the memo pass at the end of `build_pre`, once
                // `avg_op_min` is final.
                flex_inv: 0.0,
                pt_n: Vec::new(),
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

    // -----------------------------------------------------------------------
    // E45 (`fm_fuel_lean`) memo tables -- see the comment block on `Pre`.
    //
    // BIT-IDENTICAL CONTRACT.  Every expression below is a verbatim copy of the one
    // `fjsp_medium::score_candidate` evaluates per candidate, with the SAME operand order
    // and the SAME association -- IEEE-754 f64 ops are deterministic functions of their
    // inputs, so evaluating each one once per nonce instead of ~7.5e8 times per nonce
    // yields the same bits.  Do not "simplify" any of these (in particular do NOT turn a
    // division into a reciprocal multiply: `x/c != x*(1/c)` in f64).
    //
    // The denominators are spelled out to mirror the double-`.max()` in the reader: the
    // struct stores e.g. `max_job_avg_work.max(1.0)` and `score_candidate` then applies
    // `.max(1e-9)` to it.  `.max()` is idempotent, but both are written out so the
    // correspondence is checkable line-by-line.
    // -----------------------------------------------------------------------
    let pre_max_ops = max_ops.max(1);
    let pre_max_job_avg_work = max_job_avg_work.max(1.0);
    let pre_max_job_bn = max_job_bn.max(1e-9);

    let horizon_d = horizon.max(1.0);
    let max_job_avg_work_d = pre_max_job_avg_work.max(1e-9);
    let max_job_bn_d = pre_max_job_bn.max(1e-9);
    let max_ops_d = (pre_max_ops as f64).max(1.0);
    let avg_machine_scarcity_d = avg_machine_scarcity.max(1e-9);
    let avg_op_min_d = avg_op_min.max(1.0);

    // Per-op memos: `flex_inv` and the per-(op, machine) `proc_n`.
    for ops in product_ops.iter_mut() {
        for oi in ops.iter_mut() {
            oi.flex_inv = 1.0 / (oi.flex as f64).max(1.0);
            let mut ptn: Vec<f64> = Vec::with_capacity(oi.machines.len());
            for &(_, pt) in oi.machines.iter() {
                ptn.push((pt as f64) / avg_op_min_d);
            }
            oi.pt_n = ptn;
        }
    }

    let n_prod = product_ops.len();
    let mut fm_rem_min_n: Vec<Vec<f64>> = Vec::with_capacity(n_prod);
    let mut fm_rem_avg_n: Vec<Vec<f64>> = Vec::with_capacity(n_prod);
    let mut fm_bn_n: Vec<Vec<f64>> = Vec::with_capacity(n_prod);
    let mut fm_ops_n: Vec<Vec<f64>> = Vec::with_capacity(n_prod);
    let mut fm_density_n: Vec<Vec<f64>> = Vec::with_capacity(n_prod);
    let mut fm_next_term_raw: Vec<Vec<f64>> = Vec::with_capacity(n_prod);

    for p in 0..n_prod {
        // `job_ops_len[job] == product_ops[job_products[job]].len()` (built above), so the
        // `ops_rem = job_ops_len[job] - op_idx` that `score_candidate` uses is a function of
        // (product, op_idx) alone.  That is what makes `fm_ops_n` / `fm_density_n` tabulable.
        let n_ops = product_ops[p].len();
        let mut v_rem_min: Vec<f64> = Vec::with_capacity(n_ops + 1);
        let mut v_rem_avg: Vec<f64> = Vec::with_capacity(n_ops + 1);
        let mut v_bn: Vec<f64> = Vec::with_capacity(n_ops + 1);
        let mut v_ops: Vec<f64> = Vec::with_capacity(n_ops + 1);
        let mut v_density: Vec<f64> = Vec::with_capacity(n_ops + 1);
        let mut v_next: Vec<f64> = Vec::with_capacity(n_ops + 1);

        // `product_suf_*` / `product_next_*` are all length `n_ops + 1`; mirror that so a
        // stale index can never read out of bounds.
        for o in 0..=n_ops {
            let rem_min = product_suf_min[p][o] as f64;
            let ops_rem = n_ops - o;

            let rem_min_n = rem_min / horizon_d;
            let rem_avg_n = product_suf_avg[p][o] / max_job_avg_work_d;
            let bn_n = product_suf_bn[p][o] / max_job_bn_d;
            let ops_n = (ops_rem as f64) / max_ops_d;
            let density_n =
                ((rem_min / (ops_rem as f64).max(1.0)) / avg_op_min_d).clamp(0.0, 4.0);

            let next_min_n = (product_next_min[p][o] as f64) / horizon_d;
            let next_flex_inv = product_next_flex_inv[p][o];
            let next_term_raw =
                (0.55 * next_min_n + 0.45 * next_flex_inv) * (1.0 + 0.30 * density_n * high_flex);

            v_rem_min.push(rem_min_n);
            v_rem_avg.push(rem_avg_n);
            v_bn.push(bn_n);
            v_ops.push(ops_n);
            v_density.push(density_n);
            v_next.push(next_term_raw);
        }

        fm_rem_min_n.push(v_rem_min);
        fm_rem_avg_n.push(v_rem_avg);
        fm_bn_n.push(v_bn);
        fm_ops_n.push(v_ops);
        fm_density_n.push(v_density);
        fm_next_term_raw.push(v_next);
    }

    let fm_machine_scar_n: Vec<f64> = machine_scarcity
        .iter()
        .map(|&s| s / avg_machine_scarcity_d)
        .collect();

    // `best_cnt_total` is `cnt_best.max(1)` from `best_second_and_counts`, so it is in
    // `1..=op.machines.len()` and `op.machines.len() <= num_machines`.  Size `+ 2` so that
    // index 0 and the inclusive upper end are both in range without an extra guard.
    let fm_scarcity_urg: Vec<f64> = (0..(num_machines + 2))
        .map(|c| 1.0 / (c as f64).max(1.0))
        .collect();

    // `flow_term` is `pre.flow_w * pre.job_flow_pref[job] * (0.65 + 0.70*(1.0 - progress))`.
    // `*` is left-associative, so folding the leading pair is exact.
    let fm_flow_wj: Vec<f64> = job_flow_pref.iter().map(|&f| flow_w * f).collect();

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
        // E45 memo tables (read only under `fm_fuel_lean`).
        fm_rem_min_n,
        fm_rem_avg_n,
        fm_bn_n,
        fm_ops_n,
        fm_density_n,
        fm_next_term_raw,
        fm_machine_scar_n,
        fm_scarcity_urg,
        fm_flow_wj,
    })
}
