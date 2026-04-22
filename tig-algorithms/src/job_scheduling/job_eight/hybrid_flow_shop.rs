use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::job_scheduling::*;

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

    if let Some(sol) = best_solution { save_solution(&sol)?; }
    let _ = best_makespan;
    Ok(())
}
