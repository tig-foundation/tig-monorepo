use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::job_scheduling::*;

use super::types::*;
use super::infra::*;

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

            *learned_jb = Some(job_bias_from_solution(pre, &sol2)?);
            *learned_mp = Some(machine_penalty_from_solution(pre, &sol2, challenge.num_machines)?);
            *learned_rp = Some(route_pref_from_solution_lite(pre, &sol2, challenge)?);

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
        let (sol, mk) = construct_solution_conflict(
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

        let elite_ok = !top_solutions.is_empty() && r > 25;
        let elite_boost = ((stuck as f64) / 140.0).clamp(0.0, 1.0);
        let elite_base =
            (0.05 + 0.10 * pre.high_flex + 0.06 * pre.jobshopness).clamp(0.02, 0.22);
        let elite_p = (elite_base + 0.28 * elite_boost + if late { 0.05 } else { 0.0 }).clamp(0.0, 0.45);

        let use_elite = elite_ok && rng.gen::<f64>() < elite_p;

        let use_learn = !use_elite
            && learned_jb.is_some()
            && learned_mp.is_some()
            && learned_rp.is_some()
            && rng.gen::<f64>() < learn_p;

        let (sol, mk) = if use_elite {
            let elite_n = top_solutions.len().min(6).max(1);
            let t: f64 = rng.gen();
            let u: f64 = rng.gen();
            let pick = (((t * u) * elite_n as f64) as usize).min(elite_n - 1);
            let elite_sol = &top_solutions[pick].0;

            let jb = job_bias_from_solution(pre, elite_sol)?;
            let mp = machine_penalty_from_solution(pre, elite_sol, challenge.num_machines)?;
            let rp = route_pref_from_solution_lite(pre, elite_sol, challenge)?;
            let route_w = (route_w_base * (1.10 + 0.30 * elite_boost)).clamp(route_w_base, 0.45);

            construct_solution_conflict(
                challenge,
                pre,
                rule,
                k,
                target,
                &mut rng,
                Some(&jb),
                Some(&mp),
                Some(&rp),
                route_w,
            )?
        } else if use_learn {
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
                challenge,
                pre,
                rule,
                k,
                target,
                &mut rng,
                None,
                None,
                None,
                0.0,
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

            learned_jb = Some(job_bias_from_solution(pre, &sol)?);
            learned_mp = Some(machine_penalty_from_solution(pre, &sol, challenge.num_machines)?);
            learned_rp = Some(route_pref_from_solution_lite(pre, &sol, challenge)?);
        } else {
            stuck = stuck.saturating_add(1);
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

    // Phase 1: Concentrated LS (fewer runs, budget for final pass)
    let ls_runs = top_solutions.len().min(5);
    let ls_iters = (effort.hybrid_flow_shop_iters / 50).max(40);
    let ls_cands = (effort.hybrid_flow_shop_iters / 25).max(80);
    let ls_perturb = (effort.hybrid_flow_shop_iters / 120).max(12);
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

    // Phase 2: Escape — try LS from a non-best archived solution (diversification)
    if top_solutions.len() > 2 {
        let escape_idx = top_solutions.len().min(5).max(2) - 1;
        let escape_sol = &top_solutions[escape_idx].0;
        if let Some((sol3, mk3)) =
            critical_block_move_local_search_ex(pre, challenge, escape_sol, ls_iters, ls_cands, ls_perturb)?
        {
            if mk3 < best_makespan {
                best_makespan = mk3;
                best_solution = Some(sol3.clone());
                save_solution(&sol3)?;
            }
        }
    }

    // Phase 3: Greedy reassign → LS → reassign chain (each feeds the next)
    if let Some(ref sol) = best_solution {
        if let Some((rsol, rmk)) = greedy_reassign_pass(pre, challenge, sol)? {
            if rmk < best_makespan {
                best_makespan = rmk;
                best_solution = Some(rsol.clone());
                save_solution(&rsol)?;
            }
        }
    }

    if let Some(ref sol) = best_solution {
        if let Some((sol4, mk4)) =
            critical_block_move_local_search_ex(pre, challenge, sol, ls_iters * 2, ls_cands, ls_perturb * 2)?
        {
            if mk4 < best_makespan {
                best_makespan = mk4;
                best_solution = Some(sol4.clone());
                save_solution(&sol4)?;
            }
        }
    }

    // Final reassign after LS
    if let Some(ref sol) = best_solution {
        if let Some((rsol2, rmk2)) = greedy_reassign_pass(pre, challenge, sol)? {
            if rmk2 < best_makespan {
                best_makespan = rmk2;
                best_solution = Some(rsol2.clone());
                save_solution(&rsol2)?;
            }
        }
    }

    // LNS post-processing on best
    if let Some(ref sol) = best_solution {
        if let Some((lsol, lmk)) = lns_disj_post(pre, challenge, sol, 500)? {
            if lmk < best_makespan {
                best_makespan = lmk;
                best_solution = Some(lsol.clone());
                save_solution(&lsol)?;
            }
        }
    }

    if let Some(sol) = best_solution {
        save_solution(&sol)?;
    }
    Ok(())
}
