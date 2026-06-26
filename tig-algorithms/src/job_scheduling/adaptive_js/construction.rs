use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng};
use tig_challenges::job_scheduling::*;
use super::types::*;
use super::helpers::*;
use super::scoring::*;

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
