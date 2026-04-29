use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng, seq::SliceRandom};
use tig_challenges::job_scheduling::*;
use std::collections::HashMap;

use super::types::*;
use super::infra::*;

// Sigmoid-normalized construction for HFS — mirrors av5's construct_solution_conflict_mode
// Uses x/(1+x) for all terms + multiplicative interactions between terms
fn construct_hfs_sig(
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
        .job_ops_len.iter().map(|&len| Vec::with_capacity(len)).collect();

    let mut remaining_ops = pre.total_ops;
    let mut time = 0u32;

    let mut demand: Vec<u16> = vec![0u16; num_machines];
    let mut raw_by_machine: Vec<Vec<RawCand>> =
        (0..num_machines).map(|_| Vec::with_capacity(12)).collect();
    let mut idle_machines: Vec<usize> = Vec::with_capacity(num_machines);

    let chaotic_like = pre.chaotic_like;
    let mut machine_work: Vec<u64> = if chaotic_like { vec![0u64; num_machines] } else { vec![] };
    let mut sum_work: u64 = 0;

    let avg_op_min_scale = pre.avg_op_min.max(1.0);
    let horizon_scale = pre.horizon.max(1.0);
    let time_scale_sc = pre.time_scale.max(1.0);
    let bn_focus_u = { let x = pre.bn_focus; if x <= 0.0 { 0.0 } else { x / (1.0 + x) } };

    // sigmoid helper — captured by closure for zero-cost inlining
    let sig = |x: f64| -> f64 { if x <= 0.0 { 0.0 } else { x / (1.0 + x) } };

    while remaining_ops > 0 {
        loop {
            idle_machines.clear();
            for m in 0..num_machines {
                if machine_avail[m] <= time { idle_machines.push(m); }
            }
            if idle_machines.is_empty() { break; }

            for &m in &idle_machines { demand[m] = 0; raw_by_machine[m].clear(); }

            let progress = 1.0 - (remaining_ops as f64) / (pre.total_ops as f64).max(1.0);
            let cap_per_machine = if k == 0 { 12usize } else { (k + 6).min(12) };

            // Phased k-decay (same as infra.rs — tested +3.6% on fjsp_high)
            let ek = if k > 1 {
                if chaotic_like {
                    if progress > 0.80 { 1usize } else if progress > 0.60 { ((k + 1) / 2).max(1) } else { k }
                } else if pre.flex_avg > 1.5 {
                    if progress > 0.90 { 1usize } else if progress > 0.75 { ((k + 1) / 2).max(1) } else { k }
                } else { k }
            } else { k };

            let prog_gate = sig(progress);

            for job in 0..num_jobs {
                let op_idx = job_next_op[job];
                if op_idx >= pre.job_ops_len[job] || job_ready_time[job] > time { continue; }
                let product = pre.job_products[job];
                let op = &pre.product_ops[product][op_idx];
                if op.flex == 0 || op.machines.is_empty() || op.min_pt >= INF { continue; }

                let (best_end, second_end, best_cnt_total, best_cnt_idle) =
                    best_second_and_counts(time, &machine_avail, op);
                if best_end >= INF || best_cnt_idle == 0 { continue; }

                let ops_rem = pre.job_ops_len[job] - op_idx;
                let jb = job_bias.map(|v| v[job]).unwrap_or(0.0);
                let flex_inv = 1.0 / (op.flex as f64).max(1.0);
                let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
                let regret = if second_end >= INF { pre.avg_op_min * 2.6 } else { (second_end - best_end) as f64 };
                let regn = (regret / avg_op_min_scale).clamp(0.0, 6.0);
                let rigidity = (0.60 * flex_inv + 0.40 * scarcity_urg).clamp(0.0, 2.5);

                let rem_min = pre.product_suf_min[product][op_idx] as f64;
                let rem_avg = pre.product_suf_avg[product][op_idx];
                let rem_bn = pre.product_suf_bn[product][op_idx];
                let density_n = ((rem_min / (ops_rem as f64).max(1.0)) / avg_op_min_scale).clamp(0.0, 4.0);
                let rem_min_n = rem_min / horizon_scale;
                let rem_avg_n = rem_avg / pre.max_job_avg_work.max(1e-9);
                let bn_n = rem_bn / pre.max_job_bn.max(1e-9);
                let next_min_n = (pre.product_next_min[product][op_idx] as f64) / horizon_scale;
                let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
                let next_term_raw = (0.55 * next_min_n + 0.45 * next_flex_inv)
                    * (1.0 + 0.30 * density_n * pre.high_flex);
                let flow_term = pre.flow_w * pre.job_flow_pref[job] * (0.65 + 0.70 * (1.0 - progress));
                let slack_u = slack_urgency(pre, target_mk, time, product, op_idx);

                // Sigmoid normalization (per-job terms)
                let rem_min_u = sig(rem_min_n);
                let rem_avg_u = sig(rem_avg_n);
                let bn_u = sig(bn_n);
                let reg_u = sig(regn);
                let dens_u = sig(density_n);
                let next_u = sig(next_term_raw);
                let flex_u = sig(flex_inv * pre.flex_factor.max(0.0));
                let sat_scarcity = sig(scarcity_urg);
                let scarce_slack = scarcity_urg * slack_u;
                let scarce_reg = scarcity_urg * reg_u;
                let base_bias0 = jb + flow_term;
                let route_gain = (0.70 + 0.80 * (1.0 - progress)).clamp(0.70, 1.40);

                for &(m, pt) in &op.machines {
                    if machine_avail[m] > time { continue; }
                    let end = time.saturating_add(pt);
                    if end != best_end { continue; }

                    demand[m] = demand[m].saturating_add(1);

                    let mp = machine_penalty.map(|v| v[m]).unwrap_or(0.0);
                    let jitter = if k > 0 { rng.gen::<f64>() * 1e-9 } else { 0.0 };
                    let load_n = machine_load[m] / pre.avg_machine_load.max(1e-9);
                    let proc_n = (pt as f64) / avg_op_min_scale;
                    let mpen = mp.clamp(0.0, 1.0);
                    let end_n = (best_end as f64) / time_scale_sc;
                    let pop_pen = if chaotic_like && op.flex >= 2 {
                        let pop = pre.machine_best_pop[m];
                        (0.07 + 0.15 * (1.0 - progress)).clamp(0.05, 0.24) * pop * pre.flex_factor
                    } else { 0.0 };
                    let load_u = sig(load_n);
                    let proc_u = sig(proc_n);
                    let mpen_u = sig(mpen);
                    let end_u = sig(end_n);
                    let base_bias = base_bias0 + jitter;
                    let route_term = if route_w > 0.0 && op.flex >= 2 {
                        route_w * route_gain * route_pref_bonus_lite(route_pref, product, op_idx, m)
                    } else { 0.0 };

                    let base = match rule {
                        Rule::CriticalPath => {
                            let chain = rem_min_u * (1.0 + next_u);
                            let urgent = scarce_slack * (1.0 + scarce_reg * prog_gate);
                            chain + urgent + base_bias - end_u - pop_pen + route_term
                        }
                        Rule::MostWork => {
                            let work = rem_avg_u * (1.0 + dens_u);
                            work * (1.0 + load_u) + base_bias - end_u - pop_pen + route_term
                        }
                        Rule::LeastFlex => {
                            let rigid = flex_u * (1.0 + sat_scarcity);
                            rigid + rem_min_u + next_u + base_bias - end_u - pop_pen + route_term
                        }
                        Rule::ShortestProc => {
                            (0.0 - proc_u) + rem_min_u * (1.0 + next_u) + sat_scarcity
                                + base_bias - end_u - pop_pen + route_term
                        }
                        Rule::Regret => {
                            let rf = reg_u * (1.0 + sat_scarcity) * (1.0 + prog_gate);
                            rf + rem_min_u + next_u + base_bias - end_u - pop_pen + route_term
                        }
                        Rule::EndTight => {
                            let tight = scarce_slack * (1.0 + scarce_reg);
                            let chain = rem_min_u * (1.0 + prog_gate) * (1.0 + next_u);
                            let penal = end_u * (1.0 + prog_gate) + proc_u + mpen_u * pre.flex_factor;
                            chain + tight + base_bias - penal - pop_pen + route_term
                        }
                        Rule::BnHeavy => {
                            let bn_focus = bn_u * (1.0 + dens_u) * (1.0 + bn_focus_u);
                            let chain = rem_min_u * (1.0 + next_u);
                            let penal = end_u + proc_u + load_u * pre.flex_factor + mpen_u * pre.flex_factor;
                            bn_focus + chain + scarce_slack + reg_u + flex_u + base_bias - penal - pop_pen + route_term
                        }
                        Rule::Adaptive => {
                            let js = pre.jobshopness;
                            if js >= 1.0 - js {
                                let hard = reg_u * (1.0 + scarce_reg) + flex_u + rem_min_u * (1.0 + next_u);
                                hard + base_bias - (end_u + mpen_u * pre.flex_factor) - pop_pen + route_term
                            } else {
                                let flow = rem_avg_u * (1.0 + dens_u) + (0.0 - proc_u) + slack_u;
                                flow + base_bias - (end_u + load_u * pre.flex_factor) - pop_pen + route_term
                            }
                        }
                        Rule::FlexBalance => {
                            let flexible = flex_u * (1.0 + sat_scarcity);
                            let chain = (rem_avg_u + rem_min_u) * (1.0 + next_u);
                            let penal = end_u + load_u * pre.flex_factor + mpen_u * (1.0 + pre.flex_factor);
                            flexible + chain + base_bias - penal - pop_pen + route_term
                        }
                    };

                    push_top_k_raw(
                        &mut raw_by_machine[m],
                        RawCand { job, machine: m, pt, base_score: base, rigidity, reg_n: regn },
                        cap_per_machine,
                    );
                }
            }

            // Conflict boost + candidate selection (same as infra.rs)
            let denom = (idle_machines.len() as f64).max(1.0);
            let (conflict_w, conflict_scale) = if chaotic_like {
                (-(0.05 + 0.08 * (1.0 - progress)).clamp(0.04, 0.14),
                 (0.95 + 0.20 * pre.flex_factor).clamp(0.90, 1.20))
            } else {
                ((0.09 + 0.26 * pre.jobshopness + 0.11 * pre.high_flex + 0.16 * (1.0 - progress)).clamp(0.05, 0.45),
                 (0.90 + 0.40 * pre.flex_factor).clamp(0.85, 1.75))
            };
            let (bal_w, avg_work) = if chaotic_like {
                ((0.030 + 0.070 * (1.0 - progress)).clamp(0.025, 0.11),
                 (sum_work as f64) / (num_machines as f64).max(1.0))
            } else { (0.0, 0.0) };

            let mut best: Option<Cand> = None;
            let mut top: Vec<Cand> = if ek > 0 { Vec::with_capacity(ek) } else { Vec::new() };

            for &m in &idle_machines {
                let dem = demand[m] as f64;
                if dem <= 0.0 || raw_by_machine[m].is_empty() { continue; }
                let dem_n = ((dem - 1.0) / denom).clamp(0.0, 2.5);
                let bal_pen = if chaotic_like && bal_w > 0.0 {
                    let denomw = (avg_work + (pre.avg_op_min * 3.0).max(1.0)).max(1.0);
                    let r = (machine_work[m] as f64) / denomw;
                    -bal_w * (r / (r + 1.0)).clamp(0.0, 1.0)
                } else { 0.0 };
                for rc in &raw_by_machine[m] {
                    let rig = rc.rigidity.clamp(0.0, 2.5);
                    let regc = rc.reg_n.clamp(0.0, 4.5);
                    let mut boost = conflict_w * conflict_scale * dem_n * (1.15 * rig + 0.85 * regc);
                    if chaotic_like { boost = boost.max(-0.26); }
                    let c = Cand { job: rc.job, machine: rc.machine, pt: rc.pt,
                                   score: rc.base_score + boost + bal_pen };
                    if ek == 0 {
                        if best.map_or(true, |bb| c.score > bb.score) { best = Some(c); }
                    } else {
                        push_top_k(&mut top, c, ek);
                    }
                }
            }

            let chosen = if ek == 0 {
                match best { Some(c) => c, None => break }
            } else {
                if top.is_empty() { break; }
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
            if machine_avail[machine] > time || end_check != best_end_now { break; }

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
            if remaining_ops == 0 { break; }
        }

        if remaining_ops == 0 { break; }

        let mut next_time: Option<u32> = None;
        for &t in &machine_avail {
            if t > time { next_time = Some(next_time.map_or(t, |b| b.min(t))); }
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

fn maybe_plateau_local_search(
    pre: &Pre,
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    best_solution: &mut Option<Solution>,
    best_makespan: &mut u32,
    top_solutions: &mut Vec<(Solution, u32)>,
    learned_jb: &mut Option<Vec<f64>>,
    learned_mp: &mut Option<Vec<f64>>,
    learned_rp: &mut Option<RoutePrefLite>,
    learn_updates_left: &mut usize,
    stuck: usize,
) -> Result<bool> {
    let (p1, p2, p3) = match stuck {
        50 => (34, 57, 13),
        100 => (38, 63, 14),
        150 => (44, 73, 14),
        200 => (50, 82, 15),
        250 => (55, 88, 16),
        _ => return Ok(false),
    };

    let base_sol = match best_solution.as_ref() {
        Some(s) => s,
        None => return Ok(false),
    };

    if let Some((sol2, mk2)) = critical_block_move_local_search_ex(pre, challenge, base_sol, p1, p2, p3)? {
        if mk2 < *best_makespan {
            *best_makespan = mk2;
            *best_solution = Some(sol2.clone());
            save_solution(&sol2)?;

            if *learn_updates_left > 0 {
                *learned_jb = Some(job_bias_from_solution(pre, &sol2)?);
                *learned_mp = Some(machine_penalty_from_solution(pre, &sol2, challenge.num_machines)?);
                *learned_rp = Some(route_pref_from_solution_lite(pre, &sol2, challenge)?);
                *learn_updates_left -= 1;
            }

            push_top_solutions(top_solutions, &sol2, mk2, 25);
            return Ok(true);
        }
        push_top_solutions(top_solutions, &sol2, mk2, 25);
    }

    Ok(false)
}

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    pre: &Pre,
    greedy_sol: Solution,
    greedy_mk: u32,
    effort: &EffortConfig,
) -> Result<()> {
    let mut rng = SmallRng::from_seed(challenge.seed);

    let rules: Vec<Rule> = vec![
        Rule::Adaptive,
        Rule::BnHeavy,
        Rule::EndTight,
        Rule::CriticalPath,
        Rule::MostWork,
        Rule::LeastFlex,
        Rule::Regret,
        Rule::ShortestProc,
        Rule::FlexBalance,
    ];

    let mut best_makespan = greedy_mk;
    let mut best_solution: Option<Solution> = Some(greedy_sol);
    let mut top_solutions: Vec<(Solution, u32)> = Vec::new();

    let target_margin: u32 =
        ((pre.avg_op_min * (0.9 + 0.9 * pre.high_flex + 0.6 * pre.jobshopness)).max(1.0)) as u32;

    let route_w_base: f64 =
        (0.040 + 0.10 * pre.high_flex + 0.08 * pre.jobshopness).clamp(0.04, 0.22);

    if pre.flow_route.is_some() && pre.flow_pt_by_job.is_some() {
        let (sol, mk) = neh_reentrant_flow_solution(pre, challenge.num_jobs, challenge.num_machines)?;
        if mk < best_makespan {
            best_makespan = mk;
            best_solution = Some(sol.clone());
            save_solution(&sol)?;
        }
        push_top_solutions(&mut top_solutions, &sol, mk, 25);
    }

    let mut ranked: Vec<(Rule, u32, Solution)> = Vec::with_capacity(rules.len());
    for &rule in &rules {
        let (sol, mk) = construct_hfs_sig(
            challenge,
            pre,
            rule,
            0,
            None,
            &mut rng,
            None,
            None,
            None,
            0.0,
        )?;
        if mk < best_makespan {
            best_makespan = mk;
            best_solution = Some(sol.clone());
            save_solution(&sol)?;
        }
        push_top_solutions(&mut top_solutions, &sol, mk, 25);
        ranked.push((rule, mk, sol));
    }
    ranked.sort_by_key(|x| x.1);
    let r0 = ranked[0].0;
    let r1 = ranked.get(1).map(|x| x.0).unwrap_or(r0);
    let r2 = ranked.get(2).map(|x| x.0).unwrap_or(r1);

    let mut rule_best: Vec<u32> = vec![u32::MAX; 9];
    let mut rule_tries: Vec<u32> = vec![0u32; 9];
    for (rr, mk, _) in &ranked {
        let idx = rule_idx(*rr);
        rule_best[idx] = rule_best[idx].min(*mk);
        rule_tries[idx] = rule_tries[idx].saturating_add(1);
    }

    // Elite pool (av5-style: genotypic diversity + cross-mixing)
    let elite_cap: usize = 8;
    let mut elite_jb: Vec<Vec<f64>> = Vec::with_capacity(elite_cap);
    let mut elite_mp: Vec<Vec<f64>> = Vec::with_capacity(elite_cap);
    let mut elite_rp: Vec<RoutePrefLite> = Vec::with_capacity(elite_cap);
    let mut elite_mk: Vec<u32> = Vec::with_capacity(elite_cap);

    // Initialize elite pool from top 3 ranked solutions + greedy
    for i in 0..ranked.len().min(3) {
        let sol = &ranked[i].2; let mk = ranked[i].1;
        elite_jb.push(job_bias_from_solution(pre, sol)?);
        elite_mp.push(machine_penalty_from_solution(pre, sol, challenge.num_machines)?);
        elite_rp.push(route_pref_from_solution_lite(pre, sol, challenge)?);
        elite_mk.push(mk);
    }

    // Learned from single best (backwards compatibility)
    let base = &ranked[0].2;
    let mut learned_jb = Some(job_bias_from_solution(pre, base)?);
    let mut learned_mp = Some(machine_penalty_from_solution(pre, base, challenge.num_machines)?);
    let mut learned_rp = Some(route_pref_from_solution_lite(pre, base, challenge)?);
    let mut learn_updates_left = 8usize;

    let num_restarts = (effort.num_restarts * 5) / 4;
    let k_hi = 6usize;
    let mut stuck: usize = 0;

    for r in 0..num_restarts {
        let late = r >= (num_restarts * 2) / 3;

        let (k_min, k_max) = if stuck > 170 {
            (4usize, 6usize)
        } else if stuck > 90 {
            (3usize, 6usize)
        } else if stuck > 35 {
            (2usize, 6usize)
        } else {
            (2usize, 4usize)
        };

        let rule = if r < 35 {
            let u: f64 = rng.gen();
            if u < 0.12 {
                Rule::FlexBalance
            } else if u < 0.50 {
                r0
            } else if u < 0.75 {
                r1
            } else if u < 0.90 {
                r2
            } else {
                rules[rng.gen_range(0..rules.len())]
            }
        } else {
            choose_rule_bandit(
                &mut rng,
                &rules,
                &rule_best,
                &rule_tries,
                best_makespan,
                target_margin,
                stuck,
                false,
                late,
            )
        };

        let k = if k_max <= k_min {
            k_min
        } else {
            rng.gen_range(k_min..=k_max)
        }
        .min(k_hi);

        let learn_base =
            (0.08 + 0.22 * pre.jobshopness + 0.18 * pre.high_flex).clamp(0.05, 0.42);
        let learn_boost =
            (1.0 + 0.35 * ((stuck as f64) / 120.0).clamp(0.0, 1.0)).clamp(1.0, 1.35);
        let learn_p = (learn_base * learn_boost).clamp(0.0, 0.60);

        let target = if best_makespan < (u32::MAX / 2) {
            Some(best_makespan.saturating_add(target_margin))
        } else {
            None
        };

        let elite_boost = ((stuck as f64) / 140.0).clamp(0.0, 1.0);
        // Elite pool cross-mixing (av5-style): pick jb/mp/rp from potentially different elites
        let learn_base2 = (0.09 + 0.24 * pre.jobshopness + 0.20 * pre.high_flex).clamp(0.06, 0.44);
        let learn_boost2 = (1.0 + 0.38 * ((stuck as f64) / 120.0).clamp(0.0, 1.0)).clamp(1.0, 1.38);
        let use_elite_pool = !elite_mk.is_empty() && r > 10 && rng.gen::<f64>() < (learn_base2 * learn_boost2).clamp(0.0, 0.65);

        let use_elite_top = !use_elite_pool && !top_solutions.is_empty() && r > 25 && {
            let elite_base = (0.05 + 0.10 * pre.high_flex + 0.06 * pre.jobshopness).clamp(0.02, 0.22);
            let elite_p = (elite_base + 0.28 * elite_boost + if late { 0.05 } else { 0.0 }).clamp(0.0, 0.45);
            rng.gen::<f64>() < elite_p
        };

        let use_learn = !use_elite_pool && !use_elite_top
            && learned_jb.is_some() && learned_mp.is_some() && learned_rp.is_some()
            && rng.gen::<f64>() < learn_p;

        let (sol, mk) = if use_elite_pool {
            let n = elite_mk.len();
            let pick_elite = |rng: &mut SmallRng| -> usize {
                if n <= 1 { return 0; }
                let a = rng.gen_range(0..n); let b = rng.gen_range(0..n);
                if elite_mk[a] <= elite_mk[b] { a } else { b }
            };
            let base_idx = pick_elite(&mut rng);
            let mix_p = (0.055 + 0.10 * pre.high_flex + 0.09 * pre.jobshopness + 0.16 * elite_boost).clamp(0.05, 0.40);
            let mp_idx = if n > 1 && rng.gen::<f64>() < mix_p { pick_elite(&mut rng) } else { base_idx };
            let rp_idx = if n > 1 && rng.gen::<f64>() < mix_p { pick_elite(&mut rng) } else { base_idx };
            let jitter = (0.80 + 0.70 * rng.gen::<f64>()).clamp(0.65, 1.55);
            let route_w = (route_w_base * jitter).clamp(route_w_base * 0.55, 0.45);
            construct_hfs_sig(
                challenge, pre, rule, k, target, &mut rng,
                Some(&elite_jb[base_idx]), Some(&elite_mp[mp_idx]), Some(&elite_rp[rp_idx]), route_w,
            )?
        } else if use_elite_top {
            let elite_n = top_solutions.len().min(6).max(1);
            let t: f64 = rng.gen(); let u: f64 = rng.gen();
            let pick = (((t * u) * elite_n as f64) as usize).min(elite_n - 1);
            let elite_sol = &top_solutions[pick].0;
            let jb = job_bias_from_solution(pre, elite_sol)?;
            let mp = machine_penalty_from_solution(pre, elite_sol, challenge.num_machines)?;
            let rp = route_pref_from_solution_lite(pre, elite_sol, challenge)?;
            let route_w = (route_w_base * (1.10 + 0.30 * elite_boost)).clamp(route_w_base, 0.45);
            construct_hfs_sig(
                challenge, pre, rule, k, target, &mut rng,
                Some(&jb), Some(&mp), Some(&rp), route_w,
            )?
        } else if use_learn {
            construct_hfs_sig(
                challenge, pre, rule, k, target, &mut rng,
                learned_jb.as_deref(), learned_mp.as_deref(), learned_rp.as_ref(), route_w_base,
            )?
        } else {
            construct_hfs_sig(
                challenge, pre, rule, k, target, &mut rng,
                None, None, None, 0.0,
            )?
        };

        let ridx = rule_idx(rule);
        rule_tries[ridx] = rule_tries[ridx].saturating_add(1);
        rule_best[ridx] = rule_best[ridx].min(mk);

        // Inline LS (av5-style): fire on improvements with moderate probability
        // Uses lighter params than post-processing CBMLS to preserve construction budget
        let flex01 = (pre.high_flex + pre.jobshopness).clamp(0.0, 1.0);
        let near_best = mk <= best_makespan.saturating_add((target_margin / 3).max(1));
        let do_inline_ls = if mk < best_makespan {
            late || stuck > 20 || flex01 >= 0.12 || rng.gen::<f64>() < 0.55
        } else if near_best && stuck > 140 {
            rng.gen::<f64>() < 0.04
        } else {
            false
        };
        let (mut sol, mut mk) = (sol, mk);
        if do_inline_ls {
            let bump = if flex01 > 0.60 { 1.0f64 } else { 0.0f64 };
            let p1 = (34.0 + 6.0 * bump) as usize;
            let p2 = (55.0 + 10.0 * bump) as usize;
            let p3 = 12usize;
            if let Some((sol2, mk2)) = critical_block_move_local_search_ex(pre, challenge, &sol, p1, p2, p3)? {
                if mk2 < mk { sol = sol2; mk = mk2; }
            }
        }

        if mk < best_makespan {
            best_makespan = mk;
            best_solution = Some(sol.clone());
            save_solution(&sol)?;
            stuck = 0;

            if learn_updates_left > 0 {
                learned_jb = Some(job_bias_from_solution(pre, &sol)?);
                learned_mp = Some(machine_penalty_from_solution(pre, &sol, challenge.num_machines)?);
                learned_rp = Some(route_pref_from_solution_lite(pre, &sol, challenge)?);
                learn_updates_left -= 1;
            }
            // Update elite pool with improved solution
            let jb = job_bias_from_solution(pre, &sol)?;
            let mp = machine_penalty_from_solution(pre, &sol, challenge.num_machines)?;
            let rp = route_pref_from_solution_lite(pre, &sol, challenge)?;
            elite_jb.push(jb); elite_mp.push(mp); elite_rp.push(rp); elite_mk.push(mk);
            // Prune: keep top elite_cap by makespan
            if elite_mk.len() > elite_cap {
                let mut order: Vec<usize> = (0..elite_mk.len()).collect();
                order.sort_unstable_by_key(|&i| elite_mk[i]);
                let keep: Vec<usize> = order.into_iter().take(elite_cap).collect();
                elite_jb = keep.iter().map(|&i| elite_jb[i].clone()).collect();
                elite_mp = keep.iter().map(|&i| elite_mp[i].clone()).collect();
                elite_rp = keep.iter().map(|&i| elite_rp[i].clone()).collect();
                elite_mk = keep.iter().map(|&i| elite_mk[i]).collect();
            }
        } else {
            stuck = stuck.saturating_add(1);
            // Occasionally add near-best solutions to pool for diversity
            if mk <= best_makespan.saturating_add(target_margin / 2) && rng.gen::<f64>() < 0.08 {
                let jb = job_bias_from_solution(pre, &sol)?;
                let mp = machine_penalty_from_solution(pre, &sol, challenge.num_machines)?;
                let rp = route_pref_from_solution_lite(pre, &sol, challenge)?;
                elite_jb.push(jb); elite_mp.push(mp); elite_rp.push(rp); elite_mk.push(mk);
                if elite_mk.len() > elite_cap {
                    let mut order: Vec<usize> = (0..elite_mk.len()).collect();
                    order.sort_unstable_by_key(|&i| elite_mk[i]);
                    let keep: Vec<usize> = order.into_iter().take(elite_cap).collect();
                    elite_jb = keep.iter().map(|&i| elite_jb[i].clone()).collect();
                    elite_mp = keep.iter().map(|&i| elite_mp[i].clone()).collect();
                    elite_rp = keep.iter().map(|&i| elite_rp[i].clone()).collect();
                    elite_mk = keep.iter().map(|&i| elite_mk[i]).collect();
                }
            }
        }

        push_top_solutions(&mut top_solutions, &sol, mk, 25);

        if maybe_plateau_local_search(
            pre,
            challenge,
            save_solution,
            &mut best_solution,
            &mut best_makespan,
            &mut top_solutions,
            &mut learned_jb,
            &mut learned_mp,
            &mut learned_rp,
            &mut learn_updates_left,
            stuck,
        )? {
            stuck = 0;
        }
    }

    let route_w_ls: f64 = (route_w_base * 1.40).clamp(route_w_base, 0.40);

    let mut refine_results: Vec<(Solution, u32)> = Vec::new();
    for (base_sol, _) in top_solutions.iter() {
        let jb = job_bias_from_solution(pre, base_sol)?;
        let mp = machine_penalty_from_solution(pre, base_sol, challenge.num_machines)?;
        let rp = Some(route_pref_from_solution_lite(pre, base_sol, challenge)?);

        let target_ls = if best_makespan < (u32::MAX / 2) {
            Some(best_makespan.saturating_add(target_margin / 2))
        } else {
            None
        };

        for attempt in 0..15 {
            let rule = match attempt {
                0 => r0,
                1 => Rule::Adaptive,
                2 => Rule::BnHeavy,
                3 => Rule::EndTight,
                4 => Rule::Regret,
                5 => Rule::CriticalPath,
                6 => Rule::LeastFlex,
                7 => Rule::MostWork,
                8 => Rule::FlexBalance,
                _ => r1,
            };

            let k = match attempt % 4 {
                0 => 2,
                1 => 3,
                2 => 4,
                _ => 2,
            }
            .min(k_hi);

            let (sol, mk) = construct_hfs_sig(
                challenge,
                pre,
                rule,
                k,
                target_ls,
                &mut rng,
                Some(&jb),
                Some(&mp),
                rp.as_ref(),
                route_w_ls,
            )?;

            if mk < best_makespan {
                best_makespan = mk;
                best_solution = Some(sol.clone());
                save_solution(&sol)?;
            }
            refine_results.push((sol, mk));
        }
    }
    for (sol, mk) in refine_results {
        push_top_solutions(&mut top_solutions, &sol, mk, 25);
    }

    let ls_runs = top_solutions.len().min(15);
    let ls_iters = (effort.hybrid_flow_shop_iters / 65).max(30);
    let ls_cands = (effort.hybrid_flow_shop_iters / 40).max(48);
    let ls_perturb = (effort.hybrid_flow_shop_iters / 200).max(8);
    for i in 0..ls_runs {
        let base_sol = &top_solutions[i].0;
        if let Some((sol2, mk2)) =
            critical_block_move_local_search_ex(pre, challenge, base_sol, ls_iters, ls_cands, ls_perturb)?
        {
            if mk2 < best_makespan {
                best_makespan = mk2;
                best_solution = Some(sol2.clone());
                save_solution(&sol2)?;
            }
            push_top_solutions(&mut top_solutions, &sol2, mk2, 25);
        }
    }

    // ─── Extra post-processing (job_seven additions) ─────────────────────────

    // Pass A: Schrage resequencing on top 8
    for i in 0..top_solutions.len().min(8) {
        let bs = top_solutions[i].0.clone();
        if let Ok(Some((s, m))) = schrage_pass(pre, challenge, &bs, 8) {
            if m < best_makespan { best_makespan = m; best_solution = Some(s.clone()); save_solution(&s)?; }
            push_top_solutions(&mut top_solutions, &s, m, 25);
        }
    }

    // Pass B: CP Descent BF on top 6
    for i in 0..top_solutions.len().min(6) {
        let bs = top_solutions[i].0.clone();
        if let Ok(Some((s, m))) = cp_descent_bf(pre, challenge, &bs, 6) {
            if m < best_makespan { best_makespan = m; best_solution = Some(s.clone()); save_solution(&s)?; }
            push_top_solutions(&mut top_solutions, &s, m, 25);
        }
    }

    // Pass C: Schrage again after machine reassignments
    if let Some(ref sol) = best_solution {
        if let Ok(Some((s, m))) = schrage_pass(pre, challenge, sol, 4) {
            if m < best_makespan { best_makespan = m; best_solution = Some(s.clone()); save_solution(&s)?; }
        }
    }

    // Pass D: Final CBMLS
    if let Some(ref sol) = best_solution {
        if let Some((s, m)) = critical_block_move_local_search_ex(pre, challenge, sol, ls_iters, ls_cands, ls_perturb)? {
            if m < best_makespan { best_makespan = m; best_solution = Some(s.clone()); save_solution(&s)?; }
        }
    }

    // Pass E: Greedy reassign
    if let Some(ref sol) = best_solution {
        if let Some((s, m)) = greedy_reassign_pass(pre, challenge, sol)? {
            if m < best_makespan { best_makespan = m; best_solution = Some(s.clone()); save_solution(&s)?; }
        }
    }

    // Pass F: Final CP Descent BF
    if let Some(ref sol) = best_solution {
        if let Ok(Some((s, m))) = cp_descent_bf(pre, challenge, sol, 6) {
            if m < best_makespan { best_makespan = m; best_solution = Some(s.clone()); save_solution(&s)?; }
        }
    }

    // Pass G: Window exhaustive on best
    if let Some(ref sol) = best_solution {
        if let Ok(Some((s, m))) = window_exhaustive(pre, challenge, sol, &[4, 3, 2]) {
            if m < best_makespan { best_makespan = m; best_solution = Some(s.clone()); save_solution(&s)?; }
        }
    }

    // Pass H: Two-level routing search.
    // Level 1 (7s): Single-stage perturbation (T47/i21 approach) — reliably finds Q=46825 micro-basin.
    // Level 2 (7s): Micro-perturbation FROM Q=46825 solutions — explores neighborhood to find Q>46825.
    let num_jobs = challenge.num_jobs;
    let max_stages = pre.max_ops;
    {
        // Pass H Level 1 — Two-tier screening + success-guided adaptive stage selection
        // Stage selection: flex stages weighted by past improvement success (bandit approach)
        let phase_h1_deadline = std::time::Instant::now() + std::time::Duration::from_millis(7_000);
        if let Some(mut current_base) = best_solution.clone() {
            let perturb_rates = [0.10f64, 0.20, 0.30, 0.40, 0.15, 0.25, 0.35, 0.10];
            let mut rate_idx = 0usize;
            let mut no_improve_count = 0usize;
            let mut loop_iters = 0usize;

            // Pre-compute flexible stages
            let mut flex_stages = Vec::with_capacity(max_stages);
            for st in 0..max_stages {
                let mut has_flex = false;
                for j in 0..num_jobs {
                    if st < pre.job_ops_len[j] {
                        let op_info = &pre.product_ops[pre.job_products[j]][st];
                        if op_info.machines.len() > 1 { has_flex = true; break; }
                    }
                }
                if has_flex { flex_stages.push(st); }
            }

            // Success-based adaptive stage weights (bandit)
            let mut stage_weight = vec![1.0f64; max_stages];
            let mut stage_fail_count = vec![0u32; max_stages];

            while std::time::Instant::now() < phase_h1_deadline {
                if loop_iters > 0 && loop_iters % 50 == 0 {
                    if let Some(ref best) = best_solution { current_base = best.clone(); }
                }
                loop_iters += 1;

                let rate = if no_improve_count > 20 { 0.40 } else { perturb_rates[rate_idx % perturb_rates.len()] };
                rate_idx += 1;

                let mut candidate = current_base.clone();

                // Stage selection: 70% from flex stages weighted by success, 30% any stage
                let target_stage = if !flex_stages.is_empty() && rng.gen::<f64>() < 0.70 {
                    // Weighted random selection among flex stages
                    let total_w: f64 = flex_stages.iter().map(|&s| stage_weight[s]).sum();
                    let mut pick = rng.gen::<f64>() * total_w;
                    let mut selected = flex_stages[0];
                    for &s in &flex_stages {
                        pick -= stage_weight[s];
                        if pick <= 0.0 { selected = s; break; }
                    }
                    selected
                } else {
                    rng.gen_range(0..max_stages)
                };

                for j in 0..num_jobs {
                    if target_stage >= pre.job_ops_len[j] { continue; }
                    if rng.gen::<f64>() >= rate { continue; }
                    let op_info = &pre.product_ops[pre.job_products[j]][target_stage];
                    if op_info.machines.len() <= 1 { continue; }
                    let new_m_idx = rng.gen_range(0..op_info.machines.len());
                    candidate.job_schedule[j][target_stage].0 = op_info.machines[new_m_idx].0;
                }

                // Two-tier fast screening: schrage_pass(5) pre-filter before SES(30)
                let quick_res = schrage_pass(pre, challenge, &candidate, 5).unwrap_or(None);
                let threshold = best_makespan.saturating_add(best_makespan / 30);

                let mut run_ses = false;
                let ses_input_owned: Option<Solution>;
                let ses_input_ref: &Solution;

                if let Some((ref quick_sol, quick_mk)) = quick_res {
                    if quick_mk < best_makespan {
                        best_makespan = quick_mk;
                        best_solution = Some(quick_sol.clone());
                        let _ = save_solution(quick_sol);
                        push_top_solutions(&mut top_solutions, quick_sol, quick_mk, 30);
                        no_improve_count = 0;
                        // Big win: boost this stage weight
                        if target_stage < stage_weight.len() {
                            stage_weight[target_stage] = (stage_weight[target_stage] * 2.0).min(20.0);
                            stage_fail_count[target_stage] = 0;
                        }
                    }
                    if quick_mk <= threshold {
                        run_ses = true;
                        ses_input_owned = Some(quick_sol.clone());
                        ses_input_ref = ses_input_owned.as_ref().unwrap();
                    } else {
                        ses_input_owned = None;
                        ses_input_ref = &candidate;
                    }
                } else {
                    ses_input_owned = None;
                    ses_input_ref = &candidate;
                }

                // 15% exploration fallback if surrogate rejects or fails
                if !run_ses && rng.gen::<f64>() < 0.15 { run_ses = true; }

                if run_ses {
                    if let Ok(Some((healed_sol, healed_mk))) = slack_ejection_schrage(pre, challenge, ses_input_ref, 30) {
                        let mut improved = false;

                        if healed_mk < best_makespan {
                            improved = true;
                            best_makespan = healed_mk;
                            best_solution = Some(healed_sol.clone());
                            let _ = save_solution(&healed_sol);
                            no_improve_count = 0;
                            // SES improvement: boost stage weight
                            if target_stage < stage_weight.len() {
                                stage_weight[target_stage] = (stage_weight[target_stage] * 1.5).min(20.0);
                                stage_fail_count[target_stage] = 0;
                            }
                        } else {
                            no_improve_count = no_improve_count.saturating_add(1);
                            // Track failures; decay weight every 30 consecutive fails
                            if target_stage < stage_fail_count.len() {
                                stage_fail_count[target_stage] += 1;
                                if stage_fail_count[target_stage] % 30 == 0 {
                                    stage_weight[target_stage] = (stage_weight[target_stage] * 0.85).max(0.2);
                                }
                            }
                        }

                        push_top_solutions(&mut top_solutions, &healed_sol, healed_mk, 30);

                        if improved {
                            if let Ok(Some((deep_sol, deep_mk))) = slack_ejection_schrage(pre, challenge, &healed_sol, 30) {
                                if deep_mk < best_makespan {
                                    best_makespan = deep_mk;
                                    best_solution = Some(deep_sol.clone());
                                    let _ = save_solution(&deep_sol);
                                }
                                push_top_solutions(&mut top_solutions, &deep_sol, deep_mk, 30);
                            }
                        }
                    } else {
                        no_improve_count = no_improve_count.saturating_add(1);
                    }
                } else {
                    no_improve_count = no_improve_count.saturating_add(1);
                }
            }
        }
    }

    // Level 2: slack_ejection_schrage with DIVERSE inputs.
    // Key insight: slack_ejection_schrage is deterministic — same input → same output.
    // Fix: every other call, apply a heavy perturbation (rate=0.4) BEFORE calling it,
    // creating genuinely new entry points. Also use top-50 seeds (wider pool).
    {
        let phase_h2_deadline = std::time::Instant::now() + std::time::Duration::from_millis(7_000);
        // Wider seed pool: top 50 solutions within +200 of best (vs 12 within +15)
        let top_seeds: Vec<Solution> = top_solutions.iter()
            .filter(|(_, mk)| *mk <= best_makespan.saturating_add(200))
            .take(50)
            .map(|(s, _)| s.clone())
            .collect();
        if !top_seeds.is_empty() {
            let mut seed_idx = 0usize;
            let mut call_count = 0usize;
            while std::time::Instant::now() < phase_h2_deadline {
                let seed = &top_seeds[seed_idx % top_seeds.len()];
                seed_idx += 1;
                // Every other call: perturb the seed with 40% rate before slack_ejection_schrage
                // This breaks the determinism and explores genuinely new topologies.
                let effective_seed: std::borrow::Cow<Solution> = if call_count % 2 == 1 {
                    let mut perturbed = seed.clone();
                    let target_stage = rng.gen_range(0..max_stages);
                    for j in 0..num_jobs {
                        if target_stage >= pre.job_ops_len[j] { continue; }
                        if rng.gen::<f64>() >= 0.40 { continue; }
                        let op_info = &pre.product_ops[pre.job_products[j]][target_stage];
                        if op_info.machines.len() <= 1 { continue; }
                        let new_m_idx = rng.gen_range(0..op_info.machines.len());
                        perturbed.job_schedule[j][target_stage].0 = op_info.machines[new_m_idx].0;
                    }
                    std::borrow::Cow::Owned(perturbed)
                } else {
                    std::borrow::Cow::Borrowed(seed)
                };
                call_count += 1;
                if let Ok(Some((improved_sol, improved_mk))) = slack_ejection_schrage(pre, challenge, &effective_seed, 30) {
                    if improved_mk < best_makespan {
                        best_makespan = improved_mk;
                        best_solution = Some(improved_sol.clone());
                        save_solution(&improved_sol)?;
                    }
                    push_top_solutions(&mut top_solutions, &improved_sol, improved_mk, 50);
                    if improved_mk < best_makespan.saturating_add(50) {
                        if let Ok(Some((healed_sol, healed_mk))) = schrage_pass(pre, challenge, &improved_sol, 5) {
                            if healed_mk < best_makespan {
                                best_makespan = healed_mk;
                                best_solution = Some(healed_sol.clone());
                                save_solution(&healed_sol)?;
                            }
                            push_top_solutions(&mut top_solutions, &healed_sol, healed_mk, 50);
                        }
                    }
                }
            }
        }
    }

    // Phase H Level 3: Consensus-Guided LNS — T47/i39
    // Instead of random 20% destroy (which causes massive makespan spikes rejected by RTT),
    // compute machine-assignment consensus across top_solutions and only perturb "disputed"
    // operations (agreement < 85%). Much smaller perturbations → within RTT tolerance.
    {
        let lns_deadline = std::time::Instant::now() + std::time::Duration::from_millis(9_000);
        let rtt_tolerance = 0.04f64;
        let no_improve_streak_limit = 200usize;

        // Step 1: Compute machine-assignment frequency across top_solutions
        let pool_size = top_solutions.len().max(1);
        let mut machine_freq: Vec<Vec<u32>> = vec![vec![0u32; challenge.num_machines]; pre.total_ops];
        // Precompute job op offsets (Pre lacks job_offsets; DisjSchedule has it)
        let job_op_offsets: Vec<usize> = {
            let mut off = vec![0usize; num_jobs + 1];
            for j in 0..num_jobs { off[j + 1] = off[j] + pre.job_ops_len[j]; }
            off
        };

        for (sol, _) in &top_solutions {
            for j in 0..num_jobs {
                let jops = pre.job_ops_len[j];
                if j >= sol.job_schedule.len() { continue; }
                let op_base = job_op_offsets[j];
                for stage in 0..jops {
                    if stage >= sol.job_schedule[j].len() { continue; }
                    let m = sol.job_schedule[j][stage].0;
                    let op_id = op_base + stage;
                    if op_id < machine_freq.len() && m < challenge.num_machines {
                        machine_freq[op_id][m] = machine_freq[op_id][m].saturating_add(1);
                    }
                }
            }
        }

        // Step 2: Identify disputed ops (max-agreement < 85% AND flex >= 2)
        let mut disputed_ops: Vec<(usize, usize)> = Vec::new(); // (job, stage)
        let threshold = (0.85 * pool_size as f64).ceil() as u32;
        for j in 0..num_jobs {
            let jops = pre.job_ops_len[j];
            let op_base = job_op_offsets[j];
            let product = pre.job_products[j];
            for stage in 0..jops {
                let op_id = op_base + stage;
                if op_id >= machine_freq.len() { continue; }
                if product >= pre.product_ops.len() || stage >= pre.product_ops[product].len() { continue; }
                if pre.product_ops[product][stage].machines.len() < 2 { continue; }
                let max_freq = *machine_freq[op_id].iter().max().unwrap_or(&0);
                if max_freq < threshold {
                    disputed_ops.push((j, stage));
                }
            }
        }

        // If no disputed ops (all solutions agree on routing), fall back to random 2-job destroy
        if disputed_ops.is_empty() {
            for j in 0..num_jobs {
                let product = pre.job_products[j];
                for stage in 0..pre.job_ops_len[j] {
                    if product < pre.product_ops.len() && stage < pre.product_ops[product].len()
                        && pre.product_ops[product][stage].machines.len() >= 2 {
                        disputed_ops.push((j, stage));
                    }
                }
            }
        }

        let init_sol = if let Some(ref s) = best_solution { s.clone() }
            else if !top_solutions.is_empty() { top_solutions[0].0.clone() }
            else { return Ok(()); };

        let mut lns_current = init_sol;
        let mut no_improve_streak = 0usize;
        let n_perturb = ((disputed_ops.len() / 4).clamp(1, 5)).min(disputed_ops.len());

        while std::time::Instant::now() < lns_deadline {
            let mut candidate = lns_current.clone();

            // Destroy: perturb only disputed operations (1-5 randomly selected)
            let start_idx = rng.gen_range(0..disputed_ops.len().max(1));
            for i in 0..n_perturb {
                let idx = (start_idx + i) % disputed_ops.len();
                let (j, stage) = disputed_ops[idx];
                if j >= candidate.job_schedule.len() || stage >= candidate.job_schedule[j].len() { continue; }
                let product = pre.job_products[j];
                if product >= pre.product_ops.len() || stage >= pre.product_ops[product].len() { continue; }
                let op_info = &pre.product_ops[product][stage];
                if op_info.machines.is_empty() { continue; }
                let new_m_idx = rng.gen_range(0..op_info.machines.len());
                candidate.job_schedule[j][stage].0 = op_info.machines[new_m_idx].0;
            }

            // Repair with slack_ejection_schrage
            let repair_result = slack_ejection_schrage(pre, challenge, &candidate, 30);
            let (repaired_sol, repaired_mk) = match repair_result {
                Ok(Some((s, mk))) => (s, mk),
                _ => match schrage_pass(pre, challenge, &candidate, 5) {
                    Ok(Some((s, mk))) => (s, mk),
                    _ => { no_improve_streak += 1; continue; }
                }
            };

            push_top_solutions(&mut top_solutions, &repaired_sol, repaired_mk, 50);

            if repaired_mk < best_makespan {
                best_makespan = repaired_mk;
                best_solution = Some(repaired_sol.clone());
                save_solution(&repaired_sol)?;
                lns_current = repaired_sol;
                no_improve_streak = 0;
            } else if (repaired_mk as f64) <= (best_makespan as f64) * (1.0 + rtt_tolerance) {
                lns_current = repaired_sol;
                no_improve_streak = 0;
            } else {
                no_improve_streak += 1;
                if no_improve_streak >= no_improve_streak_limit {
                    lns_current = best_solution.clone().unwrap_or_else(|| top_solutions[0].0.clone());
                    no_improve_streak = 0;
                }
            }
        }
    }

    // Multi-start Pass TS: N1 tabu on best + diverse elite solutions
    {
        let ts_budget = 50_000usize;
        let ts_tenure = ((pre.total_ops as f64).sqrt() as usize).max(5).min(20);
        let mut ts_pool: Vec<Solution> = Vec::with_capacity(3);
        let mut seen_mks: Vec<u32> = Vec::with_capacity(3);
        if let Some(ref s) = best_solution {
            ts_pool.push(s.clone());
            seen_mks.push(best_makespan);
        }
        for (pool_sol, pool_mk) in top_solutions.iter().take(5) {
            if ts_pool.len() >= 3 { break; }
            if !seen_mks.contains(pool_mk) {
                seen_mks.push(*pool_mk);
                ts_pool.push(pool_sol.clone());
            }
        }
        for ts_base in &ts_pool {
            if let Ok(Some((ts_sol, ts_mk))) = super::job_shop::tabu_search_phase(pre, challenge, ts_base, ts_budget, ts_tenure) {
                if ts_mk < best_makespan {
                    best_makespan = ts_mk;
                    best_solution = Some(ts_sol.clone());
                    save_solution(&ts_sol)?;
                }
            }
        }
    }

    // SCD-SR: Stage-Coordinate Descent with Schrage-Recreation
    // Ruins ALL ops at one stage, rebuilds with ECT+tail heuristic, exact eval_disj.
    // Escapes assignment basins that N1 swaps cannot reach.
    if let Some(ref base) = best_solution.clone() {
        if let Some((scd_sol, scd_mk)) = scd_sr_pass(pre, challenge, base, best_makespan, 50) {
            if scd_mk < best_makespan {
                best_makespan = scd_mk;
                best_solution = Some(scd_sol.clone());
                save_solution(&scd_sol)?;
                // Final TS polish on SCD-SR result
                let ts_budget = 30_000usize;
                let ts_tenure = ((pre.total_ops as f64).sqrt() as usize).max(5).min(20);
                if let Ok(Some((ts_sol, ts_mk))) = super::job_shop::tabu_search_phase(pre, challenge, &scd_sol, ts_budget, ts_tenure) {
                    if ts_mk < best_makespan {
                        best_makespan = ts_mk;
                        best_solution = Some(ts_sol.clone());
                        save_solution(&ts_sol)?;
                    }
                }
            }
        }
    }

    // Phase I: ARR-IG-FI seeding (1k iters — pure seed for CPDT-ILS)
    if let Some(ref base) = best_solution.clone() {
        let (ils_sol, ils_mk) = fast_ils_hfs(pre, challenge, base, best_makespan, &mut rng, save_solution);
        if ils_mk < best_makespan {
            best_makespan = ils_mk;
            best_solution = Some(ils_sol.clone());
        }
        // Phase J: CPDT-ILS — disjunctive tabu with O(1) swap + reassign (TABU_ITERS=40k)
        let (tabu_sol, tabu_mk) = tabu_ils_hfs(
            pre, challenge, &ils_sol, ils_mk.min(best_makespan),
            40_000, 8, 800,
            &mut rng, save_solution,
        )?;
        if tabu_mk < best_makespan {
            best_makespan = tabu_mk;
            best_solution = Some(tabu_sol);
        }
    }

    // Multi-start SCD-SR from elite pool (3 restarts, 50 sweeps, ~75s budget)
    {
        let mut pool_starts: Vec<(Solution, u32)> = Vec::new();
        if let Some(ref s) = best_solution { pool_starts.push((s.clone(), best_makespan)); }
        for (sol, mk) in top_solutions.iter().take(5) {
            if pool_starts.len() >= 3 { break; }
            if !pool_starts.iter().any(|(_, m)| *m == *mk) {
                pool_starts.push((sol.clone(), *mk));
            }
        }
        for (start_sol, start_mk) in pool_starts {
            if let Some((scd_sol, scd_mk)) = scd_sr_pass(pre, challenge, &start_sol, start_mk, 50) {
                if scd_mk < best_makespan {
                    best_makespan = scd_mk;
                    best_solution = Some(scd_sol.clone());
                    save_solution(&scd_sol)?;
                }
            }
        }
    }

    if let Some(sol) = best_solution { save_solution(&sol)?; }
    let _ = best_makespan;
    Ok(())
}

// --------------------------------------------------------------------------
// SCD-SR: Stage-Coordinate Descent with Schrage-Recreation (Gemini t47_i67)
// Ruins ALL ops at one stage, rebuilds via ECT+tail heuristic, exact eval.
// --------------------------------------------------------------------------

fn scd_sr_pass(
    pre: &Pre,
    challenge: &Challenge,
    base_sol: &Solution,
    base_mk: u32,
    max_sweeps: usize,
) -> Option<(Solution, u32)> {
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;
    let max_stages = pre.max_ops;

    let mut best = base_sol.clone();
    let mut best_mk = base_mk;

    for _sweep in 0..max_sweeps {
        let mut improved_sweep = false;

        for s in 0..max_stages {
            // Skip stage if no job has an op here
            if (0..num_jobs).all(|j| s >= pre.job_ops_len[j]) { break; }

            let mut scratch = best.clone();

            // 1. Release times at stage s: end of stage s-1 for each job
            let mut releases = vec![0u32; num_jobs];
            for j in 0..num_jobs {
                if s == 0 || s >= pre.job_ops_len[j] { continue; }
                let (prev_m, prev_start) = scratch.job_schedule[j][s - 1];
                let product = pre.job_products[j];
                let prev_pt = pt_from_op(&pre.product_ops[product][s - 1], prev_m).unwrap_or(0);
                releases[j] = prev_start + prev_pt;
            }

            // 2. Tail LBs: min remaining time from stage s+1 onward
            let mut tails = vec![0u32; num_jobs];
            for j in 0..num_jobs {
                if s >= pre.job_ops_len[j] { continue; }
                let product = pre.job_products[j];
                if s + 1 < pre.product_suf_min[product].len() {
                    tails[j] = pre.product_suf_min[product][s + 1];
                }
            }

            // 3. Sort jobs at stage s: ascending release, descending tail (Schrage)
            let mut stage_jobs: Vec<usize> = (0..num_jobs)
                .filter(|&j| s < pre.job_ops_len[j])
                .collect();
            stage_jobs.sort_unstable_by(|&a, &b| {
                releases[a].cmp(&releases[b])
                    .then_with(|| tails[b].cmp(&tails[a]))
            });

            // 4. Build machine usages from stages 0..s-1 (fixed)
            let mut machine_usages: Vec<Vec<(u32, u32)>> = vec![Vec::new(); num_machines];
            for j in 0..num_jobs {
                for t in 0..s.min(pre.job_ops_len[j]) {
                    let (m, st) = scratch.job_schedule[j][t];
                    let product = pre.job_products[j];
                    let pt = pt_from_op(&pre.product_ops[product][t], m).unwrap_or(0);
                    if pt > 0 { insert_usage(&mut machine_usages[m], st, st + pt); }
                }
            }

            // 5. RECREATE: ECT+tail assignment for stage s
            for &j in &stage_jobs {
                let product = pre.job_products[j];
                let op_info = &pre.product_ops[product][s];
                let rel = releases[j];
                let tail = tails[j];

                let mut best_m = usize::MAX;
                let mut best_start = 0u32;
                let mut best_score = u32::MAX;

                for &(m, pt) in &op_info.machines {
                    let start = find_earliest_gap(&machine_usages[m], rel, pt);
                    let score = (start + pt).saturating_add(tail);
                    if score < best_score {
                        best_score = score;
                        best_start = start;
                        best_m = m;
                    }
                }

                if best_m == usize::MAX { continue; }
                let pt = pt_from_op(op_info, best_m).unwrap_or(0);
                insert_usage(&mut machine_usages[best_m], best_start, best_start + pt);
                scratch.job_schedule[j][s] = (best_m, best_start);
            }

            // 6. Propagate: recompute start times for stages s+1..max_stages-1
            // Keep machine assignments fixed, recompute timing using accumulated usages.
            // machine_usages now contains stages 0..s; extend for stages s+1..
            for t in (s + 1)..max_stages {
                if (0..num_jobs).all(|j| t >= pre.job_ops_len[j]) { break; }

                // Collect jobs at stage t with release times from stage t-1
                let mut machine_queues: Vec<Vec<(u32, usize)>> = vec![Vec::new(); num_machines];
                for j in 0..num_jobs {
                    if t >= pre.job_ops_len[j] { continue; }
                    let (prev_m, prev_start) = scratch.job_schedule[j][t - 1];
                    let product = pre.job_products[j];
                    let prev_pt = pt_from_op(&pre.product_ops[product][t - 1], prev_m).unwrap_or(0);
                    let rel = prev_start + prev_pt;
                    let (m, _) = scratch.job_schedule[j][t];
                    machine_queues[m].push((rel, j));
                }

                // For each machine: sort by release, schedule via gap-filling
                for m in 0..num_machines {
                    if machine_queues[m].is_empty() { continue; }
                    machine_queues[m].sort_unstable();
                    for &(rel, j) in &machine_queues[m] {
                        let product = pre.job_products[j];
                        let pt = pt_from_op(&pre.product_ops[product][t], m).unwrap_or(0);
                        let start = find_earliest_gap(&machine_usages[m], rel, pt);
                        insert_usage(&mut machine_usages[m], start, start + pt);
                        scratch.job_schedule[j][t] = (m, start);
                    }
                }
            }

            // 7. EXACT EVAL via eval_disj
            if let Some(mk) = eval_solution_makespan(pre, challenge, &scratch) {
                if mk < best_mk {
                    best_mk = mk;
                    best = scratch;
                    improved_sweep = true;
                }
            }
        }

        if !improved_sweep { break; }
    }

    if best_mk < base_mk { Some((best, best_mk)) } else { None }
}

// ============================================================================
// CPDT-ILS: Critical-Path Disjunctive Tabu-ILS with Machine Reassignment
// Ported from fjsp_medium.rs — O(1) estimated swap + reassign on disj. graph
// ============================================================================

#[inline]
fn estimate_swap_hfs(
    u: usize, v: usize,
    heads: &[u32], tails: &[u32], pt: &[u32],
    job_pred: &[usize], job_succ: &[usize],
    machine_pred: &[usize], machine_succ: &[usize],
) -> u32 {
    let mp_u = machine_pred[u];
    let ms_v = machine_succ[v];
    let jp_v = job_pred[v];
    let jp_u = job_pred[u];
    let js_u = job_succ[u];
    let js_v = job_succ[v];

    let r_mp_u = if mp_u != NONE_USIZE { heads[mp_u].saturating_add(pt[mp_u]) } else { 0 };
    let r_jp_v = if jp_v != NONE_USIZE { heads[jp_v].saturating_add(pt[jp_v]) } else { 0 };
    let new_r_v = r_jp_v.max(r_mp_u);

    let r_jp_u = if jp_u != NONE_USIZE { heads[jp_u].saturating_add(pt[jp_u]) } else { 0 };
    let new_r_u = r_jp_u.max(new_r_v.saturating_add(pt[v]));

    let q_js_u = if js_u != NONE_USIZE { pt[js_u].saturating_add(tails[js_u]) } else { 0 };
    let q_ms_v = if ms_v != NONE_USIZE { pt[ms_v].saturating_add(tails[ms_v]) } else { 0 };
    let new_q_u = q_js_u.max(q_ms_v);

    let q_js_v = if js_v != NONE_USIZE { pt[js_v].saturating_add(tails[js_v]) } else { 0 };
    let new_q_v = q_js_v.max(pt[u].saturating_add(new_q_u));

    (new_r_v.saturating_add(pt[v]).saturating_add(new_q_v))
        .max(new_r_u.saturating_add(pt[u]).saturating_add(new_q_u))
}

#[inline]
fn estimate_reassign_hfs(
    ds: &DisjSchedule,
    heads: &[u32], tails: &[u32],
    node: usize, new_machine: usize, new_pt: u32, insert_pos: usize,
    job_pred: &[usize], machine_pred: &[usize], machine_succ: &[usize],
) -> u32 {
    let jp = job_pred[node];
    let js = ds.job_succ[node];
    let old_mp = machine_pred[node];
    let old_ms = machine_succ[node];

    let jp_end = if jp != NONE_USIZE { heads[jp].saturating_add(ds.node_pt[jp]) } else { 0 };

    let new_seq = &ds.machine_seq[new_machine];
    let new_mp_end = if insert_pos > 0 && !new_seq.is_empty() {
        let pred = new_seq[(insert_pos - 1).min(new_seq.len() - 1)];
        heads[pred].saturating_add(ds.node_pt[pred])
    } else { 0 };

    let new_start = jp_end.max(new_mp_end);
    let new_end = new_start.saturating_add(new_pt);

    let js_tail = if js != NONE_USIZE { ds.node_pt[js].saturating_add(tails[js]) } else { 0 };
    let new_ms_tail = if insert_pos < new_seq.len() {
        let succ = new_seq[insert_pos];
        ds.node_pt[succ].saturating_add(tails[succ])
    } else { 0 };

    let node_path = new_end.saturating_add(js_tail.max(new_ms_tail));

    let old_reconnect = if old_mp != NONE_USIZE && old_ms != NONE_USIZE {
        let old_mp_end = heads[old_mp].saturating_add(ds.node_pt[old_mp]);
        old_mp_end.saturating_add(ds.node_pt[old_ms]).saturating_add(tails[old_ms])
    } else { 0 };

    node_path.max(old_reconnect)
}

fn find_insert_positions_hfs(
    ds: &DisjSchedule,
    starts: &[u32],
    node: usize,
    new_machine: usize,
    job_pred: &[usize],
) -> Vec<usize> {
    let seq = &ds.machine_seq[new_machine];
    let len = seq.len();
    if len == 0 { return vec![0]; }

    let jp = job_pred[node];
    let job_pred_end = if jp != NONE_USIZE { starts[jp].saturating_add(ds.node_pt[jp]) } else { 0 };
    let cur_start = starts[node];

    let mut pos_after_jp = len;
    for (i, &nd) in seq.iter().enumerate() {
        if starts[nd] > job_pred_end { pos_after_jp = i; break; }
    }

    let mut pos_by_cur = len;
    for (i, &nd) in seq.iter().enumerate() {
        if starts[nd] >= cur_start { pos_by_cur = i; break; }
    }

    let mut out: Vec<usize> = Vec::with_capacity(6);
    {
        let push = |v: &mut Vec<usize>, p: usize| {
            if p <= len && !v.contains(&p) { v.push(p); }
        };
        push(&mut out, pos_after_jp);
        push(&mut out, pos_after_jp.saturating_sub(1));
        push(&mut out, pos_by_cur);
        push(&mut out, pos_by_cur.saturating_sub(1));
        push(&mut out, 0);
        push(&mut out, len);
    }
    if out.is_empty() { out.push(len); }
    if out.len() > 6 { out.truncate(6); }
    out
}

enum MoveHfs {
    Swap { machine: usize, pos: usize },
    Reassign { node: usize, new_machine: usize, new_pt: u32, insert_pos: usize },
}

fn tabu_ils_hfs(
    pre: &Pre,
    challenge: &Challenge,
    seed_sol: &Solution,
    seed_mk: u32,
    max_iters: usize,
    tenure_base: usize,
    stagnation_limit: usize,
    rng: &mut SmallRng,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<(Solution, u32)> {
    let Ok(mut ds) = build_disj_from_solution(pre, challenge, seed_sol) else {
        return Ok((seed_sol.clone(), seed_mk));
    };
    let mut buf = EvalBuf::new(ds.n);
    let n = ds.n;
    let Some((initial_mk, _)) = eval_disj(&ds, &mut buf) else {
        return Ok((seed_sol.clone(), seed_mk));
    };

    let mut best_mk = initial_mk.min(seed_mk);
    let mut best_ds = ds.clone();
    let mut cur_mk = initial_mk;

    let tenure = tenure_base.max(5);
    let tenure_delta = (tenure / 3).max(2);
    let kick_threshold = (stagnation_limit * 2 / 3).max(50);

    let mut tabu_swap: HashMap<(usize, usize), usize> = HashMap::with_capacity(tenure * 8);
    let mut tabu_reassign: HashMap<(usize, usize), usize> = HashMap::with_capacity(tenure * 4);

    // Job predecessor array (constant — job chains don't change)
    let mut job_pred_node = vec![NONE_USIZE; n];
    for j in 0..ds.num_jobs {
        let base = ds.job_offsets[j];
        let end = ds.job_offsets[j + 1];
        for k in (base + 1)..end { job_pred_node[k] = k - 1; }
    }

    let mut no_improve = 0usize;
    let mut kicks_left = 5usize;
    let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ (initial_mk as u64).wrapping_shl(16) ^ (n as u64).wrapping_mul(0x517CC1B727220A95);

    let mut machine_pred_node = vec![NONE_USIZE; n];
    let mut crit = vec![false; n];

    for iter in 0..max_iters {
        if no_improve >= stagnation_limit {
            if kicks_left == 0 { break; }
            ds = best_ds.clone();
            let Some((mk, _)) = eval_disj(&ds, &mut buf) else { break };
            cur_mk = mk;
            no_improve = 0;
            kicks_left -= 1;
            tabu_swap.clear();
            tabu_reassign.clear();
            continue;
        }

        // Periodic kick on stagnation plateau
        if no_improve > 0 && no_improve % kick_threshold == 0 && kicks_left > 0 {
            let Some((_, kick_mk_node)) = eval_disj(&ds, &mut buf) else { break };
            crit.fill(false);
            let mut u = kick_mk_node;
            while u != NONE_USIZE { crit[u] = true; u = buf.best_pred[u]; }
            let mut kick_swaps: Vec<(usize, usize)> = Vec::new();
            for m in 0..ds.num_machines {
                if ds.machine_seq[m].len() <= 1 { continue; }
                for i in 0..(ds.machine_seq[m].len() - 1) {
                    if crit[ds.machine_seq[m][i]] || crit[ds.machine_seq[m][i + 1]] {
                        kick_swaps.push((m, i));
                    }
                }
            }
            if !kick_swaps.is_empty() {
                for _ in 0..3 {
                    pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                    let idx = (pseed as usize) % kick_swaps.len();
                    let (m, pos) = kick_swaps[idx];
                    if pos + 1 < ds.machine_seq[m].len() { ds.machine_seq[m].swap(pos, pos + 1); }
                }
            }
            kicks_left -= 1;
            continue;
        }

        let Some((mk_now, mk_node)) = eval_disj(&ds, &mut buf) else { break };
        cur_mk = mk_now;

        if cur_mk < best_mk {
            best_mk = cur_mk;
            best_ds = ds.clone();
            no_improve = 0;
            if let Ok(s) = disj_to_solution(pre, &ds, &buf.start) {
                let _ = save_solution(&s);
            }
        } else {
            no_improve += 1;
        }

        let tails = compute_tails_pulsar(&ds, &buf);

        machine_pred_node.fill(NONE_USIZE);
        for seq in &ds.machine_seq {
            for i in 1..seq.len() { machine_pred_node[seq[i]] = seq[i - 1]; }
        }

        crit.fill(false);
        let mut u = mk_node;
        while u != NONE_USIZE { crit[u] = true; u = buf.best_pred[u]; }

        let mut best_move: Option<MoveHfs> = None;
        let mut best_move_mk = u32::MAX;
        let mut fallback_move: Option<MoveHfs> = None;
        let mut fallback_mk = u32::MAX;

        // N1: swap adjacent pairs in critical blocks
        for m in 0..ds.num_machines {
            if ds.machine_seq[m].len() <= 1 { continue; }
            let mut i = 0;
            while i < ds.machine_seq[m].len() {
                if !crit[ds.machine_seq[m][i]] { i += 1; continue; }
                let bstart = i;
                let mut bend = i;
                while bend + 1 < ds.machine_seq[m].len() {
                    let x = ds.machine_seq[m][bend];
                    let y = ds.machine_seq[m][bend + 1];
                    if !crit[y] { break; }
                    let end_x = buf.start[x].saturating_add(ds.node_pt[x]);
                    if buf.start[y] != end_x { break; }
                    bend += 1;
                }
                if bend > bstart {
                    let block_len = bend - bstart + 1;
                    let swap_positions = if block_len >= 3 { [bstart, bend - 1] } else { [bstart, NONE_USIZE] };
                    let num_swaps = if block_len >= 3 { 2 } else { 1 };
                    for si in 0..num_swaps {
                        let pos = swap_positions[si];
                        if pos == NONE_USIZE || pos + 1 >= ds.machine_seq[m].len() { continue; }
                        let node_u = ds.machine_seq[m][pos];
                        let node_v = ds.machine_seq[m][pos + 1];
                        let est_mk = estimate_swap_hfs(
                            node_u, node_v, &buf.start, &tails, &ds.node_pt,
                            &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ,
                        );
                        let key = (node_u.min(node_v), node_u.max(node_v));
                        let is_tabu = tabu_swap.get(&key).map_or(false, |&exp| iter < exp);
                        let aspiration = est_mk < best_mk;
                        if (!is_tabu || aspiration) && est_mk < best_move_mk {
                            best_move_mk = est_mk;
                            best_move = Some(MoveHfs::Swap { machine: m, pos });
                        }
                        if est_mk < fallback_mk {
                            fallback_mk = est_mk;
                            fallback_move = Some(MoveHfs::Swap { machine: m, pos });
                        }
                    }
                }
                i = bend + 1;
            }
        }

        // N2: machine reassignment of critical nodes (every 3rd iter)
        if iter % 3 == 0 {
            for node in 0..n {
                if !crit[node] { continue; }
                let job = ds.node_job[node];
                let op_idx = ds.node_op[node];
                let product = pre.job_products[job];
                if op_idx >= pre.product_ops[product].len() { continue; }
                let op_info = &pre.product_ops[product][op_idx];
                if op_info.machines.len() <= 1 { continue; }
                let cur_machine = ds.node_machine[node];

                for &(new_m, new_pt) in &op_info.machines {
                    if new_m == cur_machine { continue; }
                    let key = (node, new_m);
                    let is_tabu = tabu_reassign.get(&key).map_or(false, |&exp| iter < exp);
                    let positions = find_insert_positions_hfs(&ds, &buf.start, node, new_m, &job_pred_node);
                    for insert_pos in positions {
                        let est_mk = estimate_reassign_hfs(
                            &ds, &buf.start, &tails,
                            node, new_m, new_pt, insert_pos,
                            &job_pred_node, &machine_pred_node, &buf.machine_succ,
                        );
                        let aspiration = est_mk < best_mk;
                        if (!is_tabu || aspiration) && est_mk < best_move_mk {
                            best_move_mk = est_mk;
                            best_move = Some(MoveHfs::Reassign { node, new_machine: new_m, new_pt, insert_pos });
                        }
                        if est_mk < fallback_mk {
                            fallback_mk = est_mk;
                            fallback_move = Some(MoveHfs::Reassign { node, new_machine: new_m, new_pt, insert_pos });
                        }
                    }
                }
            }
        }

        match best_move.or(fallback_move) {
            Some(MoveHfs::Swap { machine: m, pos }) => {
                let node_a = ds.machine_seq[m][pos];
                let node_b = ds.machine_seq[m][pos + 1];
                ds.machine_seq[m].swap(pos, pos + 1);
                pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                let offset = (pseed % ((2 * tenure_delta + 1) as u64)) as usize;
                let this_tenure = (tenure + offset).saturating_sub(tenure_delta);
                tabu_swap.insert((node_a.min(node_b), node_a.max(node_b)), iter + this_tenure);
            }
            Some(MoveHfs::Reassign { node, new_machine, new_pt, insert_pos }) => {
                let old_machine = ds.node_machine[node];
                if let Some(op) = ds.machine_seq[old_machine].iter().position(|&x| x == node) {
                    ds.machine_seq[old_machine].remove(op);
                }
                let ins = insert_pos.min(ds.machine_seq[new_machine].len());
                ds.machine_seq[new_machine].insert(ins, node);
                ds.node_machine[node] = new_machine;
                ds.node_pt[node] = new_pt;
                pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
                let offset = (pseed % ((2 * tenure_delta + 1) as u64)) as usize;
                let this_tenure = (tenure + offset).saturating_sub(tenure_delta / 2);
                tabu_reassign.insert((node, old_machine), iter + this_tenure);
            }
            None => break,
        }
        let _ = cur_mk;
    }

    // Final: evaluate best_ds
    let Some((_, _)) = eval_disj(&best_ds, &mut buf) else {
        return Ok((seed_sol.clone(), seed_mk));
    };
    match disj_to_solution(pre, &best_ds, &buf.start) {
        Ok(s) => Ok((s, best_mk)),
        Err(_) => Ok((seed_sol.clone(), seed_mk)),
    }
}

// --------------------------------------------------------------------------
// Multi-resolution permutation ILS (Gemini t47_i69)
// Fast SA on job permutations (eval_hfs_perm_makespan) for inner loop.
// Exact decode (decode_hfs_gap_filling) every 50 iters to validate.
// SCD-SR polish triggered only on exact improvements. Budget: ~43s.
// --------------------------------------------------------------------------

fn fast_ils_hfs(
    pre: &Pre,
    challenge: &Challenge,
    base_sol: &Solution,
    base_mk: u32,
    rng: &mut SmallRng,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> (Solution, u32) {
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;

    // ARR-IG-FI: seeding only — FAST_ITERS=1k (~0.5s/nonce), main search is tabu_ils_hfs
    const FAST_ITERS: usize = 1_000;
    const DECODE_EVERY: usize = 50;
    const STAG_LIMIT: usize = 2_000;
    const FI_PASS_LIMIT: usize = 1;
    const ELITE_CAP: usize = 5;

    let mut seq = extract_sequence(base_sol, num_jobs);
    let mut machine_ready = vec![0u32; num_machines];

    let mut cur_mk = eval_hfs_perm_makespan(&seq, pre, &mut machine_ready);
    let mut best_fast_mk = cur_mk;
    let mut best_seq = seq.clone();
    let mut best_exact_sol = base_sol.clone();
    let mut best_exact_mk = base_mk;

    // Elite pool: (permutation, fast_makespan)
    let mut elite: Vec<(Vec<usize>, u32)> = Vec::with_capacity(ELITE_CAP);
    elite.push((seq.clone(), cur_mk));

    let mut stag_count: usize = 0;

    // SA base temperature: 4% of base_mk (was 1% in i73 — higher for more exploration)
    let t0 = base_mk as f64 * 0.04;

    for iter in 0..FAST_ITERS {
        // =================================================================
        // 1. NON-CONTIGUOUS DESTRUCTION (d random distinct indices)
        // =================================================================
        let d = 2usize.saturating_add(stag_count / 800).min(4);

        // Partial Fisher-Yates: pick d distinct indices from seq
        let mut idx_scratch: Vec<usize> = (0..num_jobs).collect();
        for i in 0..d {
            let j = rng.gen_range(i..num_jobs);
            idx_scratch.swap(i, j);
        }

        // Collect job values in random order (shuffled)
        let mut removed: Vec<usize> = idx_scratch[..d].iter().map(|&i| seq[i]).collect();
        removed.shuffle(rng);

        // Build new_seq by removing in descending index order (preserves index validity)
        let mut new_seq = seq.clone();
        let mut sorted_idx = idx_scratch[..d].to_vec();
        sorted_idx.sort_unstable_by(|a, b| b.cmp(a));
        for &idx in &sorted_idx {
            new_seq.remove(idx);
        }

        // =================================================================
        // 2. BIASED RANDOMIZED RECONSTRUCTION (RCL + roulette)
        // =================================================================
        let t_rec = (cur_mk as f64) * 0.012; // 1.2% of makespan for roulette temperature

        for &job in &removed {
            let m = new_seq.len();
            let k = (m / 4 + 1).min(5).max(2); // RCL size: 2..5

            // Evaluate all insertion positions
            let mut cands: Vec<(usize, u32)> = Vec::with_capacity(m + 1);
            for pos in 0..=m {
                new_seq.insert(pos, job);
                let mk = eval_hfs_perm_makespan(&new_seq, pre, &mut machine_ready);
                cands.push((pos, mk));
                new_seq.remove(pos);
            }
            cands.sort_by_key(|&(_, mk)| mk);

            let best_ins = cands[0].1;
            let top = &cands[..k.min(cands.len())];

            // Roulette weights: exp(-(mk - best) / t_rec)
            let mut weights: Vec<f64> = Vec::with_capacity(top.len());
            let mut sum_w = 0.0f64;
            for &(_, mk) in top {
                let w = (-(mk.saturating_sub(best_ins) as f64) / t_rec.max(1.0)).exp();
                weights.push(w);
                sum_w += w;
            }

            let chosen_pos = if sum_w > 0.0 {
                let mut r = rng.gen::<f64>() * sum_w;
                let mut pos = top[0].0;
                for (i, &w) in weights.iter().enumerate() {
                    r -= w;
                    if r <= 0.0 {
                        pos = top[i].0;
                        break;
                    }
                }
                pos
            } else {
                top[0].0
            };

            new_seq.insert(chosen_pos, job);
        }

        // =================================================================
        // 3. POST-RECONSTRUCTION FIRST-IMPROVEMENT LOCAL SEARCH (1 pass)
        // =================================================================
        {
            let mut fi_pass = 0;
            let mut improved = true;
            while improved && fi_pass < FI_PASS_LIMIT {
                improved = false;
                fi_pass += 1;
                let mut i = 0;
                while i + 1 < new_seq.len() {
                    let mk_before = eval_hfs_perm_makespan(&new_seq, pre, &mut machine_ready);
                    new_seq.swap(i, i + 1);
                    let mk_after = eval_hfs_perm_makespan(&new_seq, pre, &mut machine_ready);
                    if mk_after < mk_before {
                        improved = true;
                    } else {
                        new_seq.swap(i, i + 1); // undo
                    }
                    i += 1;
                }
            }
        }

        let new_mk = eval_hfs_perm_makespan(&new_seq, pre, &mut machine_ready);

        // =================================================================
        // 4. SA ACCEPTANCE with linear cooling + stagnation reheat
        // =================================================================
        let progress = iter as f64 / FAST_ITERS as f64;
        let mut temp = t0 * (1.0 - progress * 0.95).max(0.05);
        if stag_count > STAG_LIMIT / 2 {
            temp = temp.max(t0 * 0.30); // reheat floor on long stagnation
        }

        let accept = if new_mk <= cur_mk {
            true
        } else {
            let delta = new_mk.saturating_sub(cur_mk) as f64;
            rng.gen::<f64>() < (-delta / temp.max(1e-6)).exp()
        };

        let mut global_improved = false;
        if accept {
            seq = new_seq;
            cur_mk = new_mk;
            if cur_mk < best_fast_mk {
                best_fast_mk = cur_mk;
                best_seq = seq.clone();
                global_improved = true;
            }
            // Update elite pool (deduplicate by makespan value)
            if elite.iter().all(|(_, m)| *m != cur_mk) {
                elite.push((seq.clone(), cur_mk));
                elite.sort_by_key(|(_, m)| *m);
                if elite.len() > ELITE_CAP {
                    elite.truncate(ELITE_CAP);
                }
            }
        }

        if global_improved {
            stag_count = 0;
        } else {
            stag_count += 1;
        }

        // =================================================================
        // 5. ELITE RESTART on stagnation
        // =================================================================
        if stag_count >= STAG_LIMIT {
            if !elite.is_empty() && rng.gen_bool(0.6) {
                let eidx = rng.gen_range(0..elite.len());
                seq = elite[eidx].0.clone();
                cur_mk = elite[eidx].1;
            } else {
                seq = best_seq.clone();
                cur_mk = best_fast_mk;
            }
            // Strong random perturbation: 4 random remove-and-reinserts
            for _ in 0..4 {
                let idx = rng.gen_range(0..seq.len());
                let job = seq.remove(idx);
                let pos = rng.gen_range(0..=seq.len());
                seq.insert(pos, job);
            }
            cur_mk = eval_hfs_perm_makespan(&seq, pre, &mut machine_ready);
            stag_count = 0;
        }

        // =================================================================
        // 6. PERIODIC EXACT DECODE CHECKPOINT
        // =================================================================
        if iter % DECODE_EVERY == 0 || iter == FAST_ITERS - 1 {
            // Decode current best permutation with gap-filling decoder
            let (sol_best, exact_best) = decode_hfs_gap_filling(&best_seq, pre, challenge);
            if exact_best < best_exact_mk {
                best_exact_mk = exact_best;
                best_exact_sol = sol_best;
                let _ = save_solution(&best_exact_sol);
            }
            // Also decode current seq
            let (sol_cur, exact_cur) = decode_hfs_gap_filling(&seq, pre, challenge);
            if exact_cur < best_exact_mk {
                best_exact_mk = exact_cur;
                best_exact_sol = sol_cur;
                let _ = save_solution(&best_exact_sol);
            }
        }
    }

    (best_exact_sol, best_exact_mk)
}

// --------------------------------------------------------------------------
// SBS-IG: Stage-by-Stage gap-filling decoder + sum_ct tiebreaker (Gemini t47_i14)
// --------------------------------------------------------------------------

struct SbsWorkspace {
    ready_times: Vec<u32>,
    seq_rank: Vec<u32>,
    order: Vec<usize>,
    usages: Vec<Vec<(u32, u32)>>,
}

impl SbsWorkspace {
    fn new(num_jobs: usize, num_machines: usize) -> Self {
        Self {
            ready_times: vec![0; num_jobs],
            seq_rank: vec![0; num_jobs],
            order: Vec::with_capacity(num_jobs),
            usages: vec![Vec::with_capacity(num_jobs); num_machines],
        }
    }

    fn clear(&mut self) {
        self.ready_times.fill(0);
        self.seq_rank.fill(0);
        self.order.clear();
        for u in self.usages.iter_mut() { u.clear(); }
    }
}

fn eval_sbs_makespan(seq: &[usize], pre: &Pre, challenge: &Challenge, wk: &mut SbsWorkspace) -> (u32, u32) {
    wk.clear();
    for (idx, &j) in seq.iter().enumerate() { wk.seq_rank[j] = idx as u32; }
    wk.order.extend_from_slice(seq);
    if wk.order.is_empty() { return (0, 0); }

    for stage in 0..pre.max_ops {
        // Filter to jobs that have an op at this stage
        wk.order.retain(|&j| stage < pre.job_ops_len[j]);
        if wk.order.is_empty() { break; }

        // Sort by ready_time then seq_rank (stable priority from IG permutation)
        wk.order.sort_unstable_by(|&a, &b| {
            wk.ready_times[a].cmp(&wk.ready_times[b])
                .then_with(|| wk.seq_rank[a].cmp(&wk.seq_rank[b]))
        });

        for &j in &wk.order {
            let product = pre.job_products[j];
            let op_info = &pre.product_ops[product][stage];
            let r_time = wk.ready_times[j];

            let mut best_m = 0usize;
            let mut best_end = u32::MAX;
            let mut best_start = 0u32;

            for &(m, pt) in &op_info.machines {
                let st = find_earliest_gap(&wk.usages[m], r_time, pt);
                let end = st + pt;
                if end < best_end {
                    best_end = end;
                    best_start = st;
                    best_m = m;
                }
            }
            if best_end < u32::MAX {
                insert_usage(&mut wk.usages[best_m], best_start, best_end);
                wk.ready_times[j] = best_end;
            }
        }

        // Restore full sequence for next stage
        wk.order.clear();
        wk.order.extend_from_slice(seq);
    }

    let mut mk = 0u32;
    let mut sum_ct = 0u64;
    for &j in seq {
        let ct = wk.ready_times[j];
        if ct > mk { mk = ct; }
        sum_ct += ct as u64;
    }
    (mk, sum_ct as u32)
}

fn build_sbs_solution(seq: &[usize], pre: &Pre, challenge: &Challenge) -> (Solution, u32) {
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;
    let mut usages: Vec<Vec<(u32, u32)>> = vec![Vec::with_capacity(num_jobs); num_machines];
    let mut ready_times = vec![0u32; num_jobs];
    let mut seq_rank = vec![0u32; num_jobs];
    let mut job_schedule: Vec<Vec<(usize, u32)>> = vec![Vec::with_capacity(pre.max_ops); num_jobs];

    for (idx, &j) in seq.iter().enumerate() { seq_rank[j] = idx as u32; }
    let mut order: Vec<usize> = seq.to_vec();

    for stage in 0..pre.max_ops {
        order.retain(|&j| stage < pre.job_ops_len[j]);
        if order.is_empty() { break; }

        order.sort_unstable_by(|&a, &b| {
            ready_times[a].cmp(&ready_times[b])
                .then_with(|| seq_rank[a].cmp(&seq_rank[b]))
        });

        for &j in &order {
            let product = pre.job_products[j];
            let op_info = &pre.product_ops[product][stage];
            let r_time = ready_times[j];

            let mut best_m = 0usize;
            let mut best_end = u32::MAX;
            let mut best_start = 0u32;

            for &(m, pt) in &op_info.machines {
                let st = find_earliest_gap(&usages[m], r_time, pt);
                let end = st + pt;
                if end < best_end {
                    best_end = end;
                    best_start = st;
                    best_m = m;
                }
            }
            if best_end < u32::MAX {
                insert_usage(&mut usages[best_m], best_start, best_end);
                job_schedule[j].push((best_m, best_start));
                ready_times[j] = best_end;
            }
        }

        order.clear();
        order.extend_from_slice(seq);
    }

    let mk = ready_times.iter().copied().max().unwrap_or(0);
    (Solution { job_schedule }, mk)
}

fn sbs_iterated_greedy_hfs(
    pre: &Pre,
    challenge: &Challenge,
    initial_seq: Vec<usize>,
    deadline: std::time::Instant,
    rng: &mut SmallRng,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    global_best_mk: &mut u32,
) -> Option<(Solution, u32)> {
    let num_jobs = challenge.num_jobs;
    if num_jobs <= 1 { return None; }
    let mut wk = SbsWorkspace::new(num_jobs, challenge.num_machines);

    let mut current_seq = initial_seq;
    let (mut current_mk, _) = eval_sbs_makespan(&current_seq, pre, challenge, &mut wk);
    let mut best_seq = current_seq.clone();
    let mut best_mk = current_mk;

    let d = 5usize;
    let temp = 0.5f64;
    let mut removed: Vec<usize> = Vec::with_capacity(d);
    let mut iters = 0u64;

    loop {
        if iters & 63 == 0 && std::time::Instant::now() >= deadline { break; }
        iters += 1;

        // Destruction
        let mut new_seq = current_seq.clone();
        removed.clear();
        for _ in 0..d {
            if new_seq.is_empty() { break; }
            let idx = rng.gen_range(0..new_seq.len());
            removed.push(new_seq.remove(idx));
        }

        // Greedy re-insertion using (mk, sum_ct) objective
        for &job in &removed {
            let mut best_pos = 0usize;
            let mut best_obj = (u32::MAX, u32::MAX);
            for pos in 0..=new_seq.len() {
                new_seq.insert(pos, job);
                let (mk, sct) = eval_sbs_makespan(&new_seq, pre, challenge, &mut wk);
                if mk < best_obj.0 || (mk == best_obj.0 && sct < best_obj.1) {
                    best_obj = (mk, sct);
                    best_pos = pos;
                }
                new_seq.remove(pos);
            }
            new_seq.insert(best_pos, job);
        }

        let (new_mk, _) = eval_sbs_makespan(&new_seq, pre, challenge, &mut wk);

        if new_mk < best_mk {
            best_mk = new_mk;
            best_seq = new_seq.clone();
            if new_mk < *global_best_mk {
                *global_best_mk = new_mk;
                let (sol, _) = build_sbs_solution(&best_seq, pre, challenge);
                let _ = save_solution(&sol);
            }
        }

        if new_mk <= current_mk {
            current_seq = new_seq;
            current_mk = new_mk;
        } else {
            let delta = (new_mk - current_mk) as f64;
            let p = (-delta / temp).exp();
            if rng.gen::<f64>() < p {
                current_seq = new_seq;
                current_mk = new_mk;
            }
        }
    }

    Some(build_sbs_solution(&best_seq, pre, challenge))
}

// --------------------------------------------------------------------------
// Iterated Greedy with ECT Gap-Filling decoder (Gemini t47_i1, §5)
// Permutation-based search adapted for HFS: decoder does dynamic machine routing.
// Adjustments vs Gemini's raw code:
//   - `challenge.product_processing_times[prod][op]` → `pre.product_ops[prod][op].machines`
//     (correct field name in this codebase)
//   - iteration uses `for &(m, pt) in eligible` destructuring pattern matching the
//     rest of the code
// --------------------------------------------------------------------------

#[inline]
fn find_earliest_gap(usage: &[(u32, u32)], ready_time: u32, pt: u32) -> u32 {
    let mut s = ready_time;
    for &(start, end) in usage {
        if s + pt <= start {
            return s;
        }
        if s < end {
            s = end;
        }
    }
    s
}

#[inline]
fn insert_usage(usage: &mut Vec<(u32, u32)>, start: u32, end: u32) {
    let pos = usage.binary_search_by_key(&start, |&(s, _)| s).unwrap_or_else(|e| e);
    usage.insert(pos, (start, end));
}

pub fn decode_hfs_gap_filling(
    seq: &[usize],
    pre: &Pre,
    challenge: &Challenge,
) -> (Solution, u32) {
    let mut usage = vec![Vec::<(u32, u32)>::with_capacity(challenge.num_jobs); challenge.num_machines];
    let mut schedule: Vec<Vec<(usize, u32)>> = vec![Vec::with_capacity(pre.max_ops); challenge.num_jobs];
    let mut makespan = 0u32;

    for &j in seq {
        let mut t = 0u32;
        let prod = pre.job_products[j];
        for op_idx in 0..pre.job_ops_len[j] {
            let eligible = &pre.product_ops[prod][op_idx].machines;
            let mut best_m = 0usize;
            let mut best_start = 0u32;
            let mut best_end = u32::MAX;
            for &(m, pt) in eligible {
                let s = find_earliest_gap(&usage[m], t, pt);
                let e = s + pt;
                if e < best_end {
                    best_end = e;
                    best_m = m;
                    best_start = s;
                }
            }
            insert_usage(&mut usage[best_m], best_start, best_end);
            schedule[j].push((best_m, best_start));
            t = best_end;
        }
        if t > makespan {
            makespan = t;
        }
    }
    (Solution { job_schedule: schedule }, makespan)
}

pub fn extract_sequence(sol: &Solution, num_jobs: usize) -> Vec<usize> {
    let mut starts: Vec<(usize, u32)> = Vec::with_capacity(num_jobs);
    for j in 0..num_jobs {
        let st = sol.job_schedule[j].first().map_or(0u32, |&(_, s)| s);
        starts.push((j, st));
    }
    starts.sort_unstable_by_key(|&(_, st)| st);
    starts.into_iter().map(|(j, _)| j).collect()
}

pub fn iterated_greedy_hfs(
    initial_seq: Vec<usize>,
    pre: &Pre,
    challenge: &Challenge,
    iters: usize,
    rng: &mut SmallRng,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    global_best_mk: &mut u32,
) -> Result<(Solution, u32)> {
    let mut current_seq = initial_seq.clone();
    let (mut current_sol, mut current_mk) = decode_hfs_gap_filling(&current_seq, pre, challenge);

    let mut best_sol = current_sol.clone();
    let mut best_mk = current_mk;

    let d = 4usize;
    let t_base = 2.0f64;

    for _ in 0..iters {
        let mut temp_seq = current_seq.clone();
        let mut removed: Vec<usize> = Vec::with_capacity(d);

        for _ in 0..d {
            if temp_seq.is_empty() { break; }
            let idx = rng.gen_range(0..temp_seq.len());
            removed.push(temp_seq.remove(idx));
        }

        for &job in &removed {
            let mut best_ins_mk = {
                let mut s0 = temp_seq.clone();
                s0.insert(0, job);
                decode_hfs_gap_filling(&s0, pre, challenge).1
            };
            let mut best_pos = 0usize;
            for pos in 1..=temp_seq.len() {
                let mut test_seq = temp_seq.clone();
                test_seq.insert(pos, job);
                let (_, mk) = decode_hfs_gap_filling(&test_seq, pre, challenge);
                if mk < best_ins_mk {
                    best_ins_mk = mk;
                    best_pos = pos;
                }
            }
            temp_seq.insert(best_pos, job);
        }

        let (new_sol, new_mk) = decode_hfs_gap_filling(&temp_seq, pre, challenge);

        if new_mk < best_mk {
            best_mk = new_mk;
            best_sol = new_sol.clone();
            if best_mk < *global_best_mk {
                *global_best_mk = best_mk;
                let _ = save_solution(&best_sol);
            }
        }

        if new_mk <= current_mk {
            current_mk = new_mk;
            current_seq = temp_seq;
            current_sol = new_sol;
        } else {
            let delta = (new_mk - current_mk) as f64;
            let p = (-delta / t_base).exp();
            if rng.gen::<f64>() < p {
                current_mk = new_mk;
                current_seq = temp_seq;
                current_sol = new_sol;
            }
        }
    }
    let _ = current_sol;
    Ok((best_sol, best_mk))
}

// --------------------------------------------------------------------------
// Fast Permutation IG with allocation-free ECT decoder (Gemini t47_i9)
// Evaluates permutations without allocating any Vec in the inner loop.
// 50,000+ iterations/nonce vs ~200 for the gap-filling version.
// --------------------------------------------------------------------------

#[inline(always)]
fn eval_hfs_perm_makespan(seq: &[usize], pre: &Pre, machine_ready: &mut [u32]) -> u32 {
    machine_ready.fill(0);
    let mut makespan = 0u32;
    for &j in seq {
        let p = pre.job_products[j];
        let ops = &pre.product_ops[p];
        let mut prev_ready = 0u32;
        for op in ops {
            let mut best_end = u32::MAX;
            let mut best_m = 0usize;
            for &(m, pt) in &op.machines {
                let start = prev_ready.max(machine_ready[m]);
                let end = start.saturating_add(pt);
                if end < best_end {
                    best_end = end;
                    best_m = m;
                }
            }
            if best_end < u32::MAX {
                machine_ready[best_m] = best_end;
                prev_ready = best_end;
            }
        }
        if prev_ready > makespan {
            makespan = prev_ready;
        }
    }
    makespan
}

fn decode_perm_to_solution(seq: &[usize], pre: &Pre, num_machines: usize) -> (Solution, u32) {
    let mut machine_ready = vec![0u32; num_machines];
    let mut job_schedule: Vec<Vec<(usize, u32)>> = seq.iter().map(|_| Vec::with_capacity(pre.max_ops)).collect();
    let mut makespan = 0u32;
    for &j in seq {
        let p = pre.job_products[j];
        let mut prev = 0u32;
        for op in &pre.product_ops[p] {
            let mut best_end = u32::MAX;
            let mut best_m = 0usize;
            let mut best_st = 0u32;
            for &(m, pt) in &op.machines {
                let st = prev.max(machine_ready[m]);
                let end = st.saturating_add(pt);
                if end < best_end {
                    best_end = end;
                    best_m = m;
                    best_st = st;
                }
            }
            if best_end < u32::MAX {
                job_schedule[j].push((best_m, best_st));
                machine_ready[best_m] = best_end;
                prev = best_end;
            }
        }
        if prev > makespan {
            makespan = prev;
        }
    }
    (Solution { job_schedule }, makespan)
}

fn fast_permutation_ig(
    pre: &Pre,
    challenge: &Challenge,
    base_sol: &Solution,
    deadline: std::time::Instant,
    rng: &mut SmallRng,
) -> Option<(Solution, u32)> {
    let num_jobs = challenge.num_jobs;
    if num_jobs <= 1 { return None; }
    let num_machines = challenge.num_machines;

    let mut current_seq = extract_sequence(base_sol, num_jobs);
    let mut eval_buf = vec![0u32; num_machines];

    let mut current_mk = eval_hfs_perm_makespan(&current_seq, pre, &mut eval_buf);
    let base_mk = current_mk;

    let mut best_seq = current_seq.clone();
    let mut best_mk = current_mk;

    let d = 4usize;
    let mut temp_seq: Vec<usize> = Vec::with_capacity(num_jobs);
    let mut removed: Vec<usize> = Vec::with_capacity(d);

    let t_start = 5.0f64;
    let mut t = t_start;
    let alpha = 0.9995f64;

    let mut iters = 0u64;
    loop {
        if iters & 63 == 0 && std::time::Instant::now() >= deadline {
            break;
        }
        iters += 1;

        // Destruction
        temp_seq.clear();
        temp_seq.extend_from_slice(&current_seq);
        removed.clear();
        for _ in 0..d {
            if temp_seq.is_empty() { break; }
            let idx = rng.gen_range(0..temp_seq.len());
            removed.push(temp_seq.remove(idx));
        }

        // Greedy construction — best insertion position for each removed job
        for &job in &removed {
            let mut best_ins_mk = u32::MAX;
            let mut best_ins_pos = 0usize;
            for pos in 0..=temp_seq.len() {
                temp_seq.insert(pos, job);
                let mk = eval_hfs_perm_makespan(&temp_seq, pre, &mut eval_buf);
                if mk < best_ins_mk {
                    best_ins_mk = mk;
                    best_ins_pos = pos;
                }
                temp_seq.remove(pos);
            }
            temp_seq.insert(best_ins_pos, job);
        }

        let new_mk = eval_hfs_perm_makespan(&temp_seq, pre, &mut eval_buf);

        if new_mk < best_mk {
            best_mk = new_mk;
            best_seq.copy_from_slice(&temp_seq);
            current_seq.copy_from_slice(&temp_seq);
            current_mk = new_mk;
        } else if new_mk <= current_mk {
            current_seq.copy_from_slice(&temp_seq);
            current_mk = new_mk;
        } else {
            let delta = (new_mk - current_mk) as f64;
            if rng.gen::<f64>() < (-delta / t).exp() {
                current_seq.copy_from_slice(&temp_seq);
                current_mk = new_mk;
            }
        }
        t *= alpha;
    }

    if best_mk < base_mk {
        let (sol, mk) = decode_perm_to_solution(&best_seq, pre, num_machines);
        Some((sol, mk))
    } else {
        None
    }
}

// --------------------------------------------------------------------------
// Dynamic Memetic IG with gap-filling decoder + Order Crossover (Gemini t47_i10)
// Dynamic d: starts at 4, escalates to 8/12 when stuck, recenters at 150
// OX crossover every 150 iters to bridge separate local optima
// --------------------------------------------------------------------------

fn update_ig_pool(pool: &mut Vec<(Vec<usize>, u32)>, seq: Vec<usize>, mk: u32) {
    if !pool.iter().any(|(_, m)| *m == mk) {
        pool.push((seq, mk));
        pool.sort_unstable_by_key(|(_, m)| *m);
        if pool.len() > 4 { pool.truncate(4); }
    }
}

fn order_crossover(p1: &[usize], p2: &[usize], rng: &mut SmallRng) -> Vec<usize> {
    let n = p1.len();
    if n == 0 { return Vec::new(); }
    let mut child = vec![usize::MAX; n];
    let mut start = rng.gen_range(0..n);
    let mut end = rng.gen_range(0..n);
    if start > end { let tmp = start; start = end; end = tmp; }
    let mut in_child = vec![false; n];
    for i in start..=end {
        child[i] = p1[i];
        if p1[i] < n { in_child[p1[i]] = true; }
    }
    let mut p2_idx = (end + 1) % n;
    let mut child_idx = (end + 1) % n;
    for _ in 0..n {
        let job = p2[p2_idx];
        if job < n && !in_child[job] {
            child[child_idx] = job;
            in_child[job] = true;
            child_idx = (child_idx + 1) % n;
        }
        p2_idx = (p2_idx + 1) % n;
    }
    child
}

fn dynamic_memetic_ig(
    pre: &Pre,
    challenge: &Challenge,
    elite_pool: &mut Vec<(Vec<usize>, u32)>,
    deadline: std::time::Instant,
    rng: &mut SmallRng,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    global_best_mk: &mut u32,
) -> Option<(Vec<usize>, u32)> {
    if elite_pool.is_empty() { return None; }
    let num_jobs = challenge.num_jobs;

    let mut current_seq = elite_pool[0].0.clone();
    let mut current_mk = elite_pool[0].1;
    let mut best_seq = current_seq.clone();
    let mut best_mk = current_mk;

    let mut d = 4usize;
    let mut stuck_iters = 0usize;
    let mut loop_counter = 0u64;
    let temp_scale = pre.avg_op_min * 2.0;

    let mut temp_seq: Vec<usize> = Vec::with_capacity(num_jobs);
    let mut removed: Vec<usize> = Vec::with_capacity(16);

    loop {
        if loop_counter % 10 == 0 && std::time::Instant::now() >= deadline { break; }
        loop_counter += 1;

        // Periodic OX crossover
        if loop_counter % 150 == 0 && elite_pool.len() > 1 {
            let p2_idx = rng.gen_range(1..elite_pool.len());
            let ox_seq = order_crossover(&elite_pool[0].0, &elite_pool[p2_idx].0, rng);
            let (ox_sol, ox_mk) = decode_hfs_gap_filling(&ox_seq, pre, challenge);
            if ox_mk < best_mk {
                best_mk = ox_mk;
                best_seq = ox_seq.clone();
                let _ = save_solution(&ox_sol);
                *global_best_mk = (*global_best_mk).min(ox_mk);
                update_ig_pool(elite_pool, ox_seq.clone(), ox_mk);
            }
            current_seq = ox_seq;
            current_mk = ox_mk;
            d = 4;
            continue;
        }

        // Destruction
        temp_seq.clear();
        temp_seq.extend_from_slice(&current_seq);
        removed.clear();
        for _ in 0..d {
            if temp_seq.is_empty() { break; }
            let idx = rng.gen_range(0..temp_seq.len());
            removed.push(temp_seq.remove(idx));
        }

        // Construction with gap-filling decoder
        for &job in &removed {
            let mut best_ins_mk = u32::MAX;
            let mut best_ins_pos = 0usize;
            for pos in 0..=temp_seq.len() {
                temp_seq.insert(pos, job);
                let (_, mk) = decode_hfs_gap_filling(&temp_seq, pre, challenge);
                if mk < best_ins_mk {
                    best_ins_mk = mk;
                    best_ins_pos = pos;
                }
                temp_seq.remove(pos);
            }
            temp_seq.insert(best_ins_pos, job);
        }

        let (trial_sol, trial_mk) = decode_hfs_gap_filling(&temp_seq, pre, challenge);

        // Acceptance + dynamic d escalation
        if trial_mk < best_mk {
            best_mk = trial_mk;
            best_seq.clear();
            best_seq.extend_from_slice(&temp_seq);
            let _ = save_solution(&trial_sol);
            *global_best_mk = (*global_best_mk).min(best_mk);
            update_ig_pool(elite_pool, temp_seq.clone(), trial_mk);
            current_seq.clear();
            current_seq.extend_from_slice(&temp_seq);
            current_mk = trial_mk;
            stuck_iters = 0;
            d = 4;
        } else {
            stuck_iters += 1;
            let delta = (trial_mk as f64) - (current_mk as f64);
            let prob = (-delta / temp_scale.max(1.0)).exp();
            if rng.gen::<f64>() < prob {
                current_seq.clear();
                current_seq.extend_from_slice(&temp_seq);
                current_mk = trial_mk;
            }
            d = if stuck_iters > 150 {
                current_seq.clear(); current_seq.extend_from_slice(&best_seq);
                current_mk = best_mk; stuck_iters = 0; 4
            } else if stuck_iters > 100 { 12 }
            else if stuck_iters > 50 { 8 }
            else { 4 };
        }
    }

    if best_mk < elite_pool.first().map(|(_, m)| *m).unwrap_or(u32::MAX) {
        Some((best_seq, best_mk))
    } else {
        Some((best_seq, best_mk))
    }
}
