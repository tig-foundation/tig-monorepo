use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use tig_challenges::job_scheduling::*;

use super::types::*;
use super::construction::{construct_solution_conflict, neh_reentrant_flow_solution};
use super::learning::{job_bias_from_solution, machine_penalty_from_solution, route_pref_from_solution_lite};
use super::local_search::critical_block_move_local_search;
use super::rules::{choose_rule_bandit, rule_idx};
use super::helpers::push_top_solutions;

fn strict_makespan(challenge: &Challenge, pre: &Pre, rank: &[usize]) -> Result<u32> {
    let route = pre
        .strict_route
        .as_ref()
        .ok_or_else(|| anyhow!("strict_route missing"))?;
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;

    let mut job_next_op = vec![0usize; num_jobs];
    let mut job_ready = vec![0u32; num_jobs];
    let mut machine_avail = vec![0u32; num_machines];

    let mut remaining_ops = pre.total_ops;

    let mut future: Vec<BinaryHeap<Reverse<(u32, usize, usize)>>> = (0..num_machines)
        .map(|_| BinaryHeap::new())
        .collect();
    let mut avail: Vec<BinaryHeap<Reverse<(usize, usize)>>> = (0..num_machines)
        .map(|_| BinaryHeap::new())
        .collect();

    for job in 0..num_jobs {
        if pre.job_ops_len[job] == 0 {
            continue;
        }
        if route.is_empty() {
            return Err(anyhow!("strict_route empty"));
        }
        let op_idx = 0usize;
        if op_idx >= route.len() {
            return Err(anyhow!("op_idx out of strict route bounds"));
        }
        let m = route[op_idx];
        if m >= num_machines {
            return Err(anyhow!("strict_route machine out of bounds"));
        }
        future[m].push(Reverse((0u32, rank[job], job)));
    }

    let mut next_time: Vec<Option<u32>> = vec![None; num_machines];
    let mut machine_events: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();

    let compute_next_time = |m: usize,
                            machine_avail: &Vec<u32>,
                            future: &Vec<BinaryHeap<Reverse<(u32, usize, usize)>>>,
                            avail: &Vec<BinaryHeap<Reverse<(usize, usize)>>>|
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
        let Reverse((t, m)) = machine_events
            .pop()
            .ok_or_else(|| anyhow!("stalled in strict simulate (no machine events)"))?;

        if next_time[m] != Some(t) {
            continue;
        }
        if machine_avail[m] > t {
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
            return Err(anyhow!("job popped but already complete"));
        }
        if op_idx >= route.len() {
            return Err(anyhow!("op_idx out of strict route bounds"));
        }
        if route[op_idx] != m {
            return Err(anyhow!("route mismatch in strict simulate"));
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

        let product = pre.job_products[job];
        let pt = *challenge.product_processing_times[product][op_idx]
            .get(&m)
            .ok_or_else(|| anyhow!("missing pt in strict simulate"))?;
        let end = start.saturating_add(pt);

        job_next_op[job] += 1;
        job_ready[job] = end;
        machine_avail[m] = end;
        remaining_ops -= 1;
        makespan = makespan.max(end);

        if job_next_op[job] < pre.job_ops_len[job] {
            let next_op = job_next_op[job];
            if next_op >= route.len() {
                return Err(anyhow!("op_idx out of strict route bounds"));
            }
            let m2 = route[next_op];
            if m2 >= num_machines {
                return Err(anyhow!("strict_route machine out of bounds"));
            }
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

    Ok(makespan)
}

fn strict_simulate(challenge: &Challenge, pre: &Pre, rank: &[usize]) -> Result<(Solution, u32)> {
    let route = pre
        .strict_route
        .as_ref()
        .ok_or_else(|| anyhow!("strict_route missing"))?;
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;

    let mut job_next_op = vec![0usize; num_jobs];
    let mut job_ready = vec![0u32; num_jobs];
    let mut machine_avail = vec![0u32; num_machines];

    let mut job_schedule: Vec<Vec<(usize, u32)>> = pre
        .job_ops_len
        .iter()
        .map(|&len| Vec::with_capacity(len))
        .collect();

    let mut remaining_ops = pre.total_ops;

    let mut future: Vec<BinaryHeap<Reverse<(u32, usize, usize)>>> = (0..num_machines)
        .map(|_| BinaryHeap::new())
        .collect();
    let mut avail: Vec<BinaryHeap<Reverse<(usize, usize)>>> = (0..num_machines)
        .map(|_| BinaryHeap::new())
        .collect();

    for job in 0..num_jobs {
        if pre.job_ops_len[job] == 0 {
            continue;
        }
        if route.is_empty() {
            return Err(anyhow!("strict_route empty"));
        }
        let op_idx = 0usize;
        if op_idx >= route.len() {
            return Err(anyhow!("op_idx out of strict route bounds"));
        }
        let m = route[op_idx];
        if m >= num_machines {
            return Err(anyhow!("strict_route machine out of bounds"));
        }
        future[m].push(Reverse((0u32, rank[job], job)));
    }

    let mut next_time: Vec<Option<u32>> = vec![None; num_machines];
    let mut machine_events: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();

    let compute_next_time = |m: usize,
                            machine_avail: &Vec<u32>,
                            future: &Vec<BinaryHeap<Reverse<(u32, usize, usize)>>>,
                            avail: &Vec<BinaryHeap<Reverse<(usize, usize)>>>|
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
        let Reverse((t, m)) = machine_events
            .pop()
            .ok_or_else(|| anyhow!("stalled in strict simulate (no machine events)"))?;

        if next_time[m] != Some(t) {
            continue;
        }
        if machine_avail[m] > t {
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
            return Err(anyhow!("job popped but already complete"));
        }
        if op_idx >= route.len() {
            return Err(anyhow!("op_idx out of strict route bounds"));
        }
        if route[op_idx] != m {
            return Err(anyhow!("route mismatch in strict simulate"));
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

        let product = pre.job_products[job];
        let pt = *challenge.product_processing_times[product][op_idx]
            .get(&m)
            .ok_or_else(|| anyhow!("missing pt in strict simulate"))?;
        let end = start.saturating_add(pt);

        job_schedule[job].push((m, start));
        job_next_op[job] += 1;
        job_ready[job] = end;
        machine_avail[m] = end;
        remaining_ops -= 1;
        makespan = makespan.max(end);

        if job_next_op[job] < pre.job_ops_len[job] {
            let next_op = job_next_op[job];
            if next_op >= route.len() {
                return Err(anyhow!("op_idx out of strict route bounds"));
            }
            let m2 = route[next_op];
            if m2 >= num_machines {
                return Err(anyhow!("strict_route machine out of bounds"));
            }
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

    Ok((Solution { job_schedule }, makespan))
}

fn strict_best_by_order_search(challenge: &Challenge, pre: &Pre, passes: usize) -> Result<(Solution, u32)> {
    if pre.strict_route.is_none() || pre.flex_avg > 1.25 {
        return Err(anyhow!("not strict-like"));
    }
    let n = challenge.num_jobs;
    let mut order: Vec<usize> = (0..n).collect();

    let mut job_work: Vec<u32> = vec![0u32; n];
    for j in 0..n {
        let p = pre.job_products[j];
        let mut sum = 0u32;
        for op_idx in 0..pre.job_ops_len[j] {
            let op = &pre.product_ops[p][op_idx];
            if op.machines.is_empty() {
                continue;
            }
            sum = sum.saturating_add(op.machines[0].1);
        }
        job_work[j] = sum;
    }
    order.sort_unstable_by(|&a, &b| job_work[b].cmp(&job_work[a]).then_with(|| a.cmp(&b)));

    let mut rank = vec![0usize; n];
    for (pos, &j) in order.iter().enumerate() {
        rank[j] = pos;
    }

    let mut best_mk = strict_makespan(challenge, pre, &rank)?;
    let mut best_order = order.clone();

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

                for (p, &jj) in cand_order.iter().enumerate() {
                    rank[jj] = p;
                }

                let mk = strict_makespan(challenge, pre, &rank)?;
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
    
    order = best_order.clone();
    for (pos, &j) in order.iter().enumerate() {
        rank[j] = pos;
    }

    for _ in 0..max_passes {
        let mut improved = false;
        for i in 0..(n.saturating_sub(1)) {
            order.swap(i, i + 1);
            rank[order[i]] = i;
            rank[order[i + 1]] = i + 1;

            let mk = strict_makespan(challenge, pre, &rank)?;
            if mk < best_mk {
                best_mk = mk;
                improved = true;
                best_order = order.clone();
            } else {
                order.swap(i, i + 1);
                rank[order[i]] = i;
                rank[order[i + 1]] = i + 1;
            }
        }
        if !improved {
            break;
        }
    }
    
    order = best_order.clone();
    for (pos, &j) in order.iter().enumerate() {
        rank[j] = pos;
    }
    let mut seed = challenge.seed;
    seed[0] ^= 0xA5;
    let mut rng = SmallRng::from_seed(seed);
    let swap_budget = (n * 8).clamp(160, 600);
    for _ in 0..swap_budget {
        let i = rng.gen_range(0..n);
        let j = rng.gen_range(0..n);
        if i == j {
            continue;
        }
        order.swap(i, j);
        rank[order[i]] = i;
        rank[order[j]] = j;

        let mk = strict_makespan(challenge, pre, &rank)?;
        if mk < best_mk {
            best_mk = mk;
            best_order = order.clone();
        } else {
            order.swap(i, j);
            rank[order[i]] = i;
            rank[order[j]] = j;
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
                    let mk = strict_makespan(challenge, pre, &rank)?;
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
    order = best_order.clone();
    for (pos, &j) in order.iter().enumerate() {
        rank[j] = pos;
    }
    for _ in 0..2 {
        let mut improved = false;
        for i in 0..(n.saturating_sub(1)) {
            order.swap(i, i + 1);
            rank[order[i]] = i;
            rank[order[i + 1]] = i + 1;
            let mk = strict_makespan(challenge, pre, &rank)?;
            if mk < best_mk {
                best_mk = mk;
                improved = true;
                best_order = order.clone();
            } else {
                order.swap(i, i + 1);
                rank[order[i]] = i;
                rank[order[i + 1]] = i + 1;
            }
        }
        if !improved {
            break;
        }
    }
    for (pos, &j) in best_order.iter().enumerate() {
        rank[j] = pos;
    }
    let (best_sol, mk2) = strict_simulate(challenge, pre, &rank)?;
    if mk2 != best_mk {
        best_mk = mk2;
    }
    Ok((best_sol, best_mk))
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

    let allow_flex_balance = pre.high_flex > 0.60 && pre.jobshopness > 0.38;

    let mut rules: Vec<Rule> = vec![
        Rule::Adaptive,
        Rule::BnHeavy,
        Rule::EndTight,
        Rule::CriticalPath,
        Rule::MostWork,
        Rule::LeastFlex,
        Rule::Regret,
        Rule::ShortestProc,
    ];
    if allow_flex_balance {
        rules.push(Rule::FlexBalance);
    }

    let mut best_makespan = greedy_mk;
    let mut best_solution: Option<Solution> = Some(greedy_sol);
    let mut top_solutions: Vec<(Solution, u32)> = Vec::new();

    let target_margin: u32 =
        ((pre.avg_op_min * (0.9 + 0.9 * pre.high_flex + 0.6 * pre.jobshopness)).max(1.0)) as u32;

    let route_w_base: f64 = if pre.chaotic_like {
        0.0
    } else {
        (0.040 + 0.10 * pre.high_flex + 0.08 * pre.jobshopness).clamp(0.04, 0.22)
    };

    if pre.strict_route.is_some() && pre.flex_avg <= 1.15 && pre.jobshopness <= 0.25 {
        if let Ok((sol, mk)) = strict_best_by_order_search(challenge, pre, 3) {
            if mk < best_makespan {
                best_makespan = mk;
                best_solution = Some(sol.clone());
                save_solution(&sol)?;
            }
            push_top_solutions(&mut top_solutions, sol, mk, 15);
        }
    }

    if pre.flow_route.is_some() && pre.flow_pt_by_job.is_some() {
        let (sol, mk) =
            neh_reentrant_flow_solution(&pre, challenge.num_jobs, challenge.num_machines)?;
        if mk < best_makespan {
            best_makespan = mk;
            best_solution = Some(sol.clone());
            save_solution(&sol)?;
        }
        push_top_solutions(&mut top_solutions, sol, mk, 15);
    }

    let mut ranked: Vec<(Rule, u32, Solution)> = Vec::with_capacity(rules.len());
    for &rule in &rules {
        let (sol, mk) = construct_solution_conflict(
            challenge, &pre, rule, 0, None, &mut rng, None, None, None, 0.0,
        )?;
        if mk < best_makespan {
            best_makespan = mk;
            best_solution = Some(sol.clone());
            save_solution(&sol)?;
        }
        push_top_solutions(&mut top_solutions, sol.clone(), mk, 15);
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

    let base = &ranked[0].2;
    let mut learned_jb = Some(job_bias_from_solution(&pre, base)?);
    let mut learned_mp =
        Some(machine_penalty_from_solution(&pre, base, challenge.num_machines)?);
    let mut learned_rp = if route_w_base > 0.0 {
        Some(route_pref_from_solution_lite(&pre, base, challenge)?)
    } else {
        None
    };
    let mut learn_updates_left = 4usize;

    let num_restarts = effort.num_restarts;

    let mut k_hi = if pre.flex_avg > 8.0 {
        6
    } else if pre.flex_avg > 6.5 {
        4
    } else if pre.flex_avg > 4.0 {
        5
    } else {
        6
    };
    if pre.jobshopness > 0.60 && k_hi < 6 {
        k_hi += 1;
    }
    k_hi = k_hi.min(6).max(2);

    let mut stuck: usize = 0;

    for r in 0..num_restarts {
        let late = r >= (num_restarts * 2) / 3;

        let (k_min, k_max) = if stuck > 170 {
            (4usize, 6usize.min(k_hi))
        } else if stuck > 90 {
            (3usize, 6usize.min(k_hi.max(4)))
        } else if stuck > 35 {
            (2usize, k_hi)
        } else {
            (2usize, k_hi.min(4))
        };

        let rule = if r < 35 {
            let u: f64 = rng.gen();
            if allow_flex_balance && pre.high_flex > 0.82 && u < 0.10 {
                Rule::FlexBalance
            } else if u < 0.52 {
                r0
            } else if u < 0.80 {
                r1
            } else if u < 0.92 {
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
                pre.chaotic_like,
                late,
            )
        };

        let k = if k_max <= k_min {
            k_min
        } else {
            rng.gen_range(k_min..=k_max)
        };

        let learn_base = if pre.chaotic_like {
            0.0
        } else {
            (0.08 + 0.22 * pre.jobshopness + 0.18 * pre.high_flex).clamp(0.05, 0.42)
        };
        let learn_boost =
            (1.0 + 0.35 * ((stuck as f64) / 120.0).clamp(0.0, 1.0)).clamp(1.0, 1.35);
        let learn_p = (learn_base * learn_boost).clamp(0.0, 0.60);

        let use_learn = learned_jb.is_some()
            && learned_mp.is_some()
            && rng.gen::<f64>() < learn_p
            && (route_w_base == 0.0 || learned_rp.is_some());

        let target = if best_makespan < (u32::MAX / 2) {
            Some(best_makespan.saturating_add(target_margin))
        } else {
            None
        };

        let (sol, mk) = if use_learn {
            construct_solution_conflict(
                challenge,
                &pre,
                rule,
                k,
                target,
                &mut rng,
                learned_jb.as_deref(),
                learned_mp.as_deref(),
                learned_rp.as_ref(),
                route_w_base,
            )?
        } else {
            construct_solution_conflict(
                challenge, &pre, rule, k, target, &mut rng, None, None, None, 0.0,
            )?
        };

        let ridx = rule_idx(rule);
        rule_tries[ridx] = rule_tries[ridx].saturating_add(1);
        rule_best[ridx] = rule_best[ridx].min(mk);

        if mk < best_makespan {
            best_makespan = mk;
            best_solution = Some(sol.clone());
            save_solution(&sol)?;

            stuck = 0;

            if learn_updates_left > 0 && !pre.chaotic_like {
                learned_jb = Some(job_bias_from_solution(&pre, &sol)?);
                learned_mp = Some(machine_penalty_from_solution(
                    &pre,
                    &sol,
                    challenge.num_machines,
                )?);
                if route_w_base > 0.0 {
                    learned_rp =
                        Some(route_pref_from_solution_lite(&pre, &sol, challenge)?);
                }
                learn_updates_left -= 1;
            }
        } else {
            stuck = stuck.saturating_add(1);
        }

        push_top_solutions(&mut top_solutions, sol, mk, 15);
    }

    let route_w_ls: f64 = if route_w_base > 0.0 {
        (route_w_base * 1.40).clamp(route_w_base, 0.40)
    } else {
        0.0
    };

    for (base_sol, _) in top_solutions.iter() {
        let jb = job_bias_from_solution(&pre, base_sol)?;
        let mp = machine_penalty_from_solution(&pre, base_sol, challenge.num_machines)?;
        let rp = if route_w_ls > 0.0 {
            Some(route_pref_from_solution_lite(&pre, base_sol, challenge)?)
        } else {
            None
        };

        let target_ls = if best_makespan < (u32::MAX / 2) {
            Some(best_makespan.saturating_add(target_margin / 2))
        } else {
            None
        };

        for attempt in 0..10 {
            let rule = if pre.chaotic_like {
                match attempt % 4 {
                    0 => Rule::Adaptive,
                    1 => Rule::ShortestProc,
                    2 => Rule::MostWork,
                    _ => Rule::Regret,
                }
            } else {
                match attempt {
                    0 => r0,
                    1 => Rule::Adaptive,
                    2 => Rule::BnHeavy,
                    3 => Rule::EndTight,
                    4 => Rule::Regret,
                    5 => Rule::CriticalPath,
                    6 => Rule::LeastFlex,
                    7 => Rule::MostWork,
                    8 => {
                        if allow_flex_balance {
                            Rule::FlexBalance
                        } else {
                            r1
                        }
                    }
                    _ => r1,
                }
            };

            let k = match attempt % 4 {
                0 => 2,
                1 => 3,
                2 => 4,
                _ => 2,
            }
            .min(k_hi);

            let (sol, mk) = construct_solution_conflict(
                challenge,
                &pre,
                rule,
                k,
                target_ls,
                &mut rng,
                Some(&jb),
                Some(&mp),
                rp.as_ref(),
                if rp.is_some() { route_w_ls } else { 0.0 },
            )?;

            if mk < best_makespan {
                best_makespan = mk;
                best_solution = Some(sol.clone());
                save_solution(&sol)?;
            }
        }
    }

    let ls_runs = if pre.jobshopness > 0.55 || pre.high_flex > 0.55 {
        6usize
    } else if pre.flow_like > 0.90 {
        3usize
    } else {
        4usize
    }
    .min(top_solutions.len());

    let max_iters = if pre.jobshopness > 0.60 {
        18usize
    } else if pre.high_flex > 0.75 && pre.jobshopness < 0.35 {
        10usize
    } else {
        14usize
    };

    let top_cands = if pre.jobshopness > 0.55 { 36usize } else { 28usize };

    for i in 0..ls_runs {
        let base_sol = &top_solutions[i].0;
        if let Some((sol2, mk2)) =
            critical_block_move_local_search(&pre, challenge, base_sol, max_iters, top_cands)?
        {
            if mk2 < best_makespan {
                best_makespan = mk2;
                best_solution = Some(sol2.clone());
                save_solution(&sol2)?;
            }
            push_top_solutions(&mut top_solutions, sol2, mk2, 15);
        }
    }

    if let Some(sol) = best_solution {
        save_solution(&sol)?;
    }
    Ok(())
}
