use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::collections::HashMap;
use tig_challenges::job_scheduling::*;

use super::types::*;
use super::infra::*;

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

    if pre.flow_route.is_some() && pre.flow_pt_by_job.is_some() {
        let (sol, mk) =
            neh_reentrant_flow_solution(pre, challenge.num_jobs, challenge.num_machines)?;
        if mk < best_makespan {
            best_makespan = mk;
            best_solution = Some(sol.clone());
            save_solution(&sol)?;
        }
        push_top_solutions(&mut top_solutions, &sol, mk, 15);
    }

    let mut ranked: Vec<(Rule, u32, Solution)> = Vec::with_capacity(rules.len());
    for &rule in &rules {
        let (sol, mk) = construct_solution_conflict(
            challenge, pre, rule, 0, None, &mut rng, None, None, None, 0.0,
        )?;
        if mk < best_makespan {
            best_makespan = mk;
            best_solution = Some(sol.clone());
            save_solution(&sol)?;
        }
        push_top_solutions(&mut top_solutions, &sol, mk, 15);
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
    let mut learned_jb = Some(job_bias_from_solution(pre, base)?);
    let mut learned_mp =
        Some(machine_penalty_from_solution(pre, base, challenge.num_machines)?);
    let mut learned_rp = if route_w_base > 0.0 {
        Some(route_pref_from_solution_lite(pre, base, challenge)?)
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
                pre,
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
                challenge, pre, rule, k, target, &mut rng, None, None, None, 0.0,
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
                learned_jb = Some(job_bias_from_solution(pre, &sol)?);
                learned_mp = Some(machine_penalty_from_solution(
                    pre,
                    &sol,
                    challenge.num_machines,
                )?);
                if route_w_base > 0.0 {
                    learned_rp =
                        Some(route_pref_from_solution_lite(pre, &sol, challenge)?);
                }
                learn_updates_left -= 1;
            }
        } else {
            stuck = stuck.saturating_add(1);
        }

        push_top_solutions(&mut top_solutions, &sol, mk, 15);
    }

    let route_w_ls: f64 = if route_w_base > 0.0 {
        (route_w_base * 1.40).clamp(route_w_base, 0.40)
    } else {
        0.0
    };

    let mut refine_results: Vec<(Solution, u32)> = Vec::new();
    for (base_sol, _) in top_solutions.iter() {
        let jb = job_bias_from_solution(pre, base_sol)?;
        let mp = machine_penalty_from_solution(pre, base_sol, challenge.num_machines)?;
        let rp = if route_w_ls > 0.0 {
            Some(route_pref_from_solution_lite(pre, base_sol, challenge)?)
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
                pre,
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
            refine_results.push((sol, mk));
        }
    }
    for (sol, mk) in refine_results {
        push_top_solutions(&mut top_solutions, &sol, mk, 15);
    }

    let ts_starts = top_solutions.len().min(15);
    let ts_iters = effort.fjsp_medium_iters;
    let ts_tenure = ((pre.total_ops as f64).sqrt() as usize).clamp(5, 12);

    for i in 0..ts_starts {
        let base_sol = &top_solutions[i].0;
        if let Some((sol2, mk2)) = tabu_search_hybrid(pre, challenge, base_sol, ts_iters, ts_tenure)? {
            if mk2 < best_makespan {
                best_makespan = mk2;
                best_solution = Some(sol2.clone());
                save_solution(&sol2)?;
            }
        }
        let _ = best_makespan;
    }

    if let Some(ref sol) = best_solution {
        if let Some((improved_sol, improved_mk)) = greedy_reassign_pass(pre, challenge, sol)? {
            if improved_mk < best_makespan {
                let _ = improved_mk; 
                best_solution = Some(improved_sol.clone());
                save_solution(&improved_sol)?;
            }
        }
    }

    if let Some(sol) = best_solution {
        save_solution(&sol)?;
    }
    Ok(())
}

fn greedy_reassign_pass(
    pre: &Pre,
    challenge: &Challenge,
    base_sol: &Solution,
) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n);
    let n = ds.n;

    let Some((mut current_mk, _)) = eval_disj(&ds, &mut buf) else {
        return Ok(None);
    };
    let initial_mk = current_mk;

    let mut improved = true;
    let mut passes = 0;
    let max_passes = 3;

    while improved && passes < max_passes {
        improved = false;
        passes += 1;

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

            let mut best_m = cur_machine;
            let mut best_pt = cur_pt;
            let mut best_mk = current_mk;

            for &(new_m, new_pt) in &op_info.machines {
                if new_m == cur_machine {
                    continue;
                }

                let old_pos = ds.machine_seq[cur_machine].iter().position(|&x| x == node);
                if old_pos.is_none() {
                    continue;
                }
                let old_pos = old_pos.unwrap();

                ds.machine_seq[cur_machine].remove(old_pos);
                
                ds.machine_seq[new_m].push(node);
                ds.node_machine[node] = new_m;
                ds.node_pt[node] = new_pt;

                if let Some((test_mk, _)) = eval_disj(&ds, &mut buf) {
                    if test_mk < best_mk {
                        best_mk = test_mk;
                        best_m = new_m;
                        best_pt = new_pt;
                    }
                }

                ds.machine_seq[new_m].pop();
                ds.machine_seq[cur_machine].insert(old_pos, node);
                ds.node_machine[node] = cur_machine;
                ds.node_pt[node] = cur_pt;
            }

            if best_m != cur_machine {
                let old_pos = ds.machine_seq[cur_machine].iter().position(|&x| x == node).unwrap();
                ds.machine_seq[cur_machine].remove(old_pos);
                ds.machine_seq[best_m].push(node);
                ds.node_machine[node] = best_m;
                ds.node_pt[node] = best_pt;
                current_mk = best_mk;
                improved = true;
            }
        }
    }

    if current_mk >= initial_mk {
        return Ok(None);
    }

    let Some((_, _)) = eval_disj(&ds, &mut buf) else {
        return Ok(None);
    };
    let sol = disj_to_solution(pre, &ds, &buf.start)?;
    Ok(Some((sol, current_mk)))
}

enum MoveType {
    Swap { machine: usize, pos: usize },
    Reassign { node: usize, new_machine: usize, new_pt: u32, insert_pos: usize },
}

fn tabu_search_hybrid(
    pre: &Pre,
    challenge: &Challenge,
    base_sol: &Solution,
    max_iterations: usize,
    tenure_base: usize,
) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n);
    let n = ds.n;

    let Some(init_eval) = eval_disj(&ds, &mut buf) else {
        return Ok(None);
    };
    let initial_mk = init_eval.0;
    let mut best_global_mk = initial_mk;
    let mut best_global_ds = ds.clone();

    let tenure = tenure_base.max(5);
    let tenure_delta = (tenure / 3).max(2);
    let max_no_improve = (max_iterations / 2).max(60);

    let mut tabu_swap: HashMap<(usize, usize), usize> = HashMap::with_capacity(tenure * 8);
    let mut tabu_reassign: HashMap<(usize, usize), usize> = HashMap::with_capacity(tenure * 4);
    let mut crit = vec![false; n];
    let mut no_improve = 0usize;

    let mut pseed: u64 = (challenge.seed[0] as u64)
        .wrapping_mul(0x9E3779B97F4A7C15)
        ^ (initial_mk as u64).wrapping_shl(16)
        ^ (n as u64).wrapping_mul(0x517CC1B727220A95);

    let mut tail = vec![0u32; n];
    let mut back_deg = vec![0u16; n];
    let mut back_stack: Vec<usize> = Vec::with_capacity(n);
    let mut machine_pred_node = vec![NONE_USIZE; n];

    let mut job_pred_node = vec![NONE_USIZE; n];
    for j in 0..ds.num_jobs {
        let base = ds.job_offsets[j];
        let end = ds.job_offsets[j + 1];
        for k in (base + 1)..end {
            job_pred_node[k] = k - 1;
        }
    }

    let kick_threshold = (max_no_improve * 2 / 3).max(50);
    let mut kicks_left = 3usize;

    for iter in 0..max_iterations {
        if no_improve >= max_no_improve {
            if kicks_left == 0 {
                break;
            }
            ds = best_global_ds.clone();
            no_improve = 0;
            kicks_left -= 1;
            tabu_swap.clear();
            tabu_reassign.clear();
            continue;
        }

        if no_improve > 0 && no_improve % kick_threshold == 0 && kicks_left > 0 {
            let Some((_, kick_mk_node)) = eval_disj(&ds, &mut buf) else {
                break;
            };
            crit.fill(false);
            let mut u = kick_mk_node;
            while u != NONE_USIZE {
                crit[u] = true;
                u = buf.best_pred[u];
            }

            let mut kick_swaps: Vec<(usize, usize)> = Vec::new();
            for m in 0..ds.num_machines {
                if ds.machine_seq[m].len() <= 1 {
                    continue;
                }
                for i in 0..(ds.machine_seq[m].len() - 1) {
                    if crit[ds.machine_seq[m][i]] && crit[ds.machine_seq[m][i + 1]] {
                        kick_swaps.push((m, i));
                    }
                }
            }

            if !kick_swaps.is_empty() {
                for _ in 0..2 {
                    pseed ^= pseed.wrapping_shl(13);
                    pseed ^= pseed.wrapping_shr(7);
                    pseed ^= pseed.wrapping_shl(17);
                    let idx = (pseed as usize) % kick_swaps.len();
                    let (m, pos) = kick_swaps[idx];
                    if pos + 1 < ds.machine_seq[m].len() {
                        ds.machine_seq[m].swap(pos, pos + 1);
                    }
                }
            }
            kicks_left -= 1;
            continue;
        }

        let Some((cur_mk, mk_node)) = eval_disj(&ds, &mut buf) else {
            break;
        };

        if iter > 0 {
            if cur_mk < best_global_mk {
                best_global_mk = cur_mk;
                best_global_ds = ds.clone();
                no_improve = 0;
            } else {
                no_improve += 1;
            }
        }

        machine_pred_node.fill(NONE_USIZE);
        for seq in &ds.machine_seq {
            for i in 1..seq.len() {
                machine_pred_node[seq[i]] = seq[i - 1];
            }
        }

        tail.fill(0);
        back_deg.fill(0);
        for i in 0..n {
            if ds.job_succ[i] != NONE_USIZE {
                back_deg[i] += 1;
            }
            if buf.machine_succ[i] != NONE_USIZE {
                back_deg[i] += 1;
            }
        }
        back_stack.clear();
        for i in 0..n {
            if back_deg[i] == 0 {
                back_stack.push(i);
            }
        }
        while let Some(nd) = back_stack.pop() {
            let contrib = ds.node_pt[nd].saturating_add(tail[nd]);
            let jp = job_pred_node[nd];
            if jp != NONE_USIZE {
                if contrib > tail[jp] {
                    tail[jp] = contrib;
                }
                back_deg[jp] = back_deg[jp].saturating_sub(1);
                if back_deg[jp] == 0 {
                    back_stack.push(jp);
                }
            }
            let mp = machine_pred_node[nd];
            if mp != NONE_USIZE {
                if contrib > tail[mp] {
                    tail[mp] = contrib;
                }
                back_deg[mp] = back_deg[mp].saturating_sub(1);
                if back_deg[mp] == 0 {
                    back_stack.push(mp);
                }
            }
        }

        crit.fill(false);
        let mut u = mk_node;
        while u != NONE_USIZE {
            crit[u] = true;
            u = buf.best_pred[u];
        }

        let mut best_move: Option<(MoveType, u32)> = None;
        let mut best_move_mk = u32::MAX;
        let mut fallback_move: Option<(MoveType, u32)> = None;
        let mut fallback_mk = u32::MAX;

        for m in 0..ds.num_machines {
            if ds.machine_seq[m].len() <= 1 {
                continue;
            }

            let mut blocks: Vec<(usize, usize)> = Vec::new();
            let mut i = 0;
            while i < ds.machine_seq[m].len() {
                if !crit[ds.machine_seq[m][i]] {
                    i += 1;
                    continue;
                }

                let bstart = i;
                let mut bend = i;
                while bend + 1 < ds.machine_seq[m].len() {
                    let x = ds.machine_seq[m][bend];
                    let y = ds.machine_seq[m][bend + 1];
                    if !crit[y] {
                        break;
                    }
                    let end_x = buf.start[x].saturating_add(ds.node_pt[x]);
                    if buf.start[y] != end_x {
                        break;
                    }
                    bend += 1;
                }

                if bend > bstart {
                    blocks.push((bstart, bend));
                }
                i = bend + 1;
            }

            for &(bstart, bend) in &blocks {
                let block_len = bend - bstart + 1;

                let mut swap_positions = [bstart, NONE_USIZE];
                let num_swaps = if block_len >= 3 {
                    swap_positions[1] = bend - 1;
                    2
                } else {
                    1
                };

                for si in 0..num_swaps {
                    let pos = swap_positions[si];
                    if pos + 1 >= ds.machine_seq[m].len() {
                        continue;
                    }
                    let node_u = ds.machine_seq[m][pos];
                    let node_v = ds.machine_seq[m][pos + 1];

                    let est_mk = estimate_swap_mk(
                        node_u,
                        node_v,
                        &buf.start,
                        &tail,
                        &ds.node_pt,
                        &job_pred_node,
                        &ds.job_succ,
                        &machine_pred_node,
                        &buf.machine_succ,
                    );

                    let key = (node_u.min(node_v), node_u.max(node_v));
                    let is_tabu = tabu_swap.get(&key).map_or(false, |&exp| iter < exp);
                    let aspiration = est_mk < best_global_mk;

                    if (!is_tabu || aspiration) && est_mk < best_move_mk {
                        best_move_mk = est_mk;
                        best_move = Some((MoveType::Swap { machine: m, pos }, est_mk));
                    }

                    if est_mk < fallback_mk {
                        fallback_mk = est_mk;
                        fallback_move = Some((MoveType::Swap { machine: m, pos }, est_mk));
                    }
                }
            }
        }

        let reassign_freq = 3;
        if iter % reassign_freq == 0 {
            for node in 0..n {
                if !crit[node] {
                    continue;
                }
                let job = ds.node_job[node];
                let op_idx = ds.node_op[node];
                let product = pre.job_products[job];
                let op_info = &pre.product_ops[product][op_idx];

                if op_info.machines.len() <= 1 {
                    continue;
                }

                let cur_machine = ds.node_machine[node];

                for &(new_m, new_pt) in &op_info.machines {
                    if new_m == cur_machine {
                        continue;
                    }

                    let key = (node, new_m);
                    let is_tabu = tabu_reassign.get(&key).map_or(false, |&exp| iter < exp);

                    let positions = find_candidate_insert_positions(&ds, &buf.start, node, new_m, new_pt, &job_pred_node);

                    for insert_pos in positions {
                        let est_mk = estimate_reassign_mk(
                            &ds, &buf.start, &tail, node, new_m, new_pt,
                            insert_pos, &job_pred_node, &machine_pred_node, &buf.machine_succ,
                        );

                        let aspiration = est_mk < best_global_mk;

                        if (!is_tabu || aspiration) && est_mk < best_move_mk {
                            best_move_mk = est_mk;
                            best_move = Some((MoveType::Reassign { node, new_machine: new_m, new_pt, insert_pos }, est_mk));
                        }

                        if est_mk < fallback_mk {
                            fallback_mk = est_mk;
                            fallback_move = Some((MoveType::Reassign { node, new_machine: new_m, new_pt, insert_pos }, est_mk));
                        }
                    }
                }
            }
        }

        let chosen = best_move.or(fallback_move);

        match chosen {
            Some((MoveType::Swap { machine: m, pos }, _)) => {
                let node_a = ds.machine_seq[m][pos];
                let node_b = ds.machine_seq[m][pos + 1];
                ds.machine_seq[m].swap(pos, pos + 1);

                pseed ^= pseed.wrapping_shl(13);
                pseed ^= pseed.wrapping_shr(7);
                pseed ^= pseed.wrapping_shl(17);
                let offset = (pseed % ((2 * tenure_delta + 1) as u64)) as usize;
                let progress = (iter as f64) / (max_iterations as f64);
                let late_bonus = if progress > 0.6 { ((progress - 0.6) * 10.0) as usize } else { 0 };
                let this_tenure = (tenure + offset + late_bonus).saturating_sub(tenure_delta);

                let key = (node_a.min(node_b), node_a.max(node_b));
                tabu_swap.insert(key, iter + this_tenure);
            }
            Some((MoveType::Reassign { node, new_machine, new_pt, insert_pos }, _)) => {
                let old_machine = ds.node_machine[node];

                let old_pos = ds.machine_seq[old_machine].iter().position(|&x| x == node);
                if let Some(op) = old_pos {
                    ds.machine_seq[old_machine].remove(op);
                }

                ds.machine_seq[new_machine].insert(insert_pos, node);
                ds.node_machine[node] = new_machine;
                ds.node_pt[node] = new_pt;

                pseed ^= pseed.wrapping_shl(13);
                pseed ^= pseed.wrapping_shr(7);
                pseed ^= pseed.wrapping_shl(17);
                let offset = (pseed % ((2 * tenure_delta + 1) as u64)) as usize;
                let this_tenure = (tenure + offset).saturating_sub(tenure_delta / 2);

                tabu_reassign.insert((node, old_machine), iter + this_tenure);
            }
            None => break,
        }
    }

    let Some((final_mk, _)) = eval_disj(&ds, &mut buf) else {
        return Ok(None);
    };
    if final_mk < best_global_mk {
        best_global_mk = final_mk;
        best_global_ds = ds.clone();
    }

    if best_global_mk >= initial_mk {
        return Ok(None);
    }

    ds = best_global_ds;
    let Some((_, _)) = eval_disj(&ds, &mut buf) else {
        return Ok(None);
    };
    let sol = disj_to_solution(pre, &ds, &buf.start)?;
    Ok(Some((sol, best_global_mk)))
}

fn find_candidate_insert_positions(
    ds: &DisjSchedule,
    starts: &[u32],
    node: usize,
    new_machine: usize,
    new_pt: u32,
    job_pred: &[usize],
) -> Vec<usize> {
    let seq = &ds.machine_seq[new_machine];
    if seq.is_empty() {
        return vec![0];
    }

    let jp = job_pred[node];
    let job_pred_end = if jp != NONE_USIZE {
        starts[jp].saturating_add(ds.node_pt[jp])
    } else {
        0
    };

    let mut candidates: Vec<(usize, u32)> = Vec::with_capacity(seq.len() + 1);

    for pos in 0..=seq.len() {
        let machine_pred_end = if pos > 0 {
            let pred = seq[pos - 1];
            starts[pred].saturating_add(ds.node_pt[pred])
        } else {
            0
        };

        let my_start = job_pred_end.max(machine_pred_end);
        let my_end = my_start.saturating_add(new_pt);

        candidates.push((pos, my_end));
    }

    candidates.sort_by_key(|&(_, end)| end);

    let max_candidates = 5;
    candidates.into_iter()
        .take(max_candidates)
        .map(|(pos, _)| pos)
        .collect()
}

fn estimate_reassign_mk(
    ds: &DisjSchedule,
    heads: &[u32],
    tails: &[u32],
    node: usize,
    new_machine: usize,
    new_pt: u32,
    insert_pos: usize,
    job_pred: &[usize],
    machine_pred: &[usize],
    machine_succ: &[usize],
) -> u32 {
    let jp = job_pred[node];
    let js = ds.job_succ[node];
    let old_mp = machine_pred[node];
    let old_ms = machine_succ[node];

    let jp_end = if jp != NONE_USIZE { heads[jp].saturating_add(ds.node_pt[jp]) } else { 0 };

    let new_seq = &ds.machine_seq[new_machine];
    let new_mp_end = if insert_pos > 0 && !new_seq.is_empty() {
        let pred = new_seq[insert_pos.min(new_seq.len()) - 1];
        heads[pred].saturating_add(ds.node_pt[pred])
    } else {
        0
    };

    let new_start = jp_end.max(new_mp_end);
    let new_end = new_start.saturating_add(new_pt);

    let js_tail = if js != NONE_USIZE { ds.node_pt[js].saturating_add(tails[js]) } else { 0 };
    let new_ms_tail = if insert_pos < new_seq.len() {
        let succ = new_seq[insert_pos];
        ds.node_pt[succ].saturating_add(tails[succ])
    } else {
        0
    };

    let node_path = new_end.saturating_add(js_tail.max(new_ms_tail));

    let old_reconnect = if old_mp != NONE_USIZE && old_ms != NONE_USIZE {
        let old_mp_end = heads[old_mp].saturating_add(ds.node_pt[old_mp]);
        let old_ms_start = old_mp_end;
        old_ms_start.saturating_add(ds.node_pt[old_ms]).saturating_add(tails[old_ms])
    } else {
        0
    };

    node_path.max(old_reconnect)
}

#[inline]
fn estimate_swap_mk(
    u: usize,
    v: usize,
    heads: &[u32],
    tails: &[u32],
    pt: &[u32],
    job_pred: &[usize],
    job_succ: &[usize],
    machine_pred: &[usize],
    machine_succ: &[usize],
) -> u32 {
    let mp_u = machine_pred[u];
    let ms_v = machine_succ[v];
    let jp_v = job_pred[v];
    let jp_u = job_pred[u];
    let js_u = job_succ[u];
    let js_v = job_succ[v];

    let r_jp_v = if jp_v != NONE_USIZE { heads[jp_v].saturating_add(pt[jp_v]) } else { 0 };
    let r_mp_u = if mp_u != NONE_USIZE { heads[mp_u].saturating_add(pt[mp_u]) } else { 0 };
    let new_r_v = r_jp_v.max(r_mp_u);

    let r_jp_u = if jp_u != NONE_USIZE { heads[jp_u].saturating_add(pt[jp_u]) } else { 0 };
    let new_r_u = r_jp_u.max(new_r_v.saturating_add(pt[v]));

    let q_js_u = if js_u != NONE_USIZE { pt[js_u].saturating_add(tails[js_u]) } else { 0 };
    let q_ms_v = if ms_v != NONE_USIZE { pt[ms_v].saturating_add(tails[ms_v]) } else { 0 };
    let new_q_u = q_js_u.max(q_ms_v);

    let q_js_v = if js_v != NONE_USIZE { pt[js_v].saturating_add(tails[js_v]) } else { 0 };
    let new_q_v = q_js_v.max(pt[u].saturating_add(new_q_u));

    let path_v = new_r_v.saturating_add(pt[v]).saturating_add(new_q_v);
    let path_u = new_r_u.saturating_add(pt[u]).saturating_add(new_q_u);

    path_v.max(path_u)
}
