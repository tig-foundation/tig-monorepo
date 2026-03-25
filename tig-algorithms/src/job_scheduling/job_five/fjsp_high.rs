use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
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

    let route_w_base: f64 = (0.040 + 0.10 * pre.high_flex + 0.08 * pre.jobshopness).clamp(0.04, 0.22);

    if pre.flow_route.is_some() && pre.flow_pt_by_job.is_some() {
        let (sol, mk) =
            neh_reentrant_flow_solution(pre, challenge.num_jobs, challenge.num_machines)?;
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
            challenge, pre, rule, 0, None, &mut rng, None, None, None, 0.0,
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
    let mut learned_mp =
        Some(machine_penalty_from_solution(pre, base, challenge.num_machines)?);
    let mut learned_rp = Some(route_pref_from_solution_lite(pre, base, challenge)?);
    let mut learn_updates_left = 10usize;

    let num_restarts = (effort.num_restarts * 5) / 4;

    let k_hi = if pre.flex_avg > 8.0 {
        8
    } else if pre.flex_avg > 6.5 {
        7
    } else {
        6
    };

    let mut stuck: usize = 0;

    for r in 0..num_restarts {
        let late = r >= (num_restarts * 2) / 3;

        let (k_min, k_max) = if stuck > 170 {
            (4usize, 6usize.min(k_hi))
        } else if stuck > 90 {
            (3usize, 6usize.min(k_hi))
        } else if stuck > 35 {
            (2usize, 6usize.min(k_hi))
        } else {
            (2usize, 4usize.min(k_hi))
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
        };

        let learn_base = (0.08 + 0.22 * pre.jobshopness + 0.18 * pre.high_flex).clamp(0.05, 0.42);
        let learn_boost =
            (1.0 + 0.35 * ((stuck as f64) / 120.0).clamp(0.0, 1.0)).clamp(1.0, 1.35);
        let learn_p = (learn_base * learn_boost).clamp(0.0, 0.60);

        let use_learn = learned_jb.is_some()
            && learned_mp.is_some()
            && rng.gen::<f64>() < learn_p
            && learned_rp.is_some();

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

            if learn_updates_left > 0 {
                learned_jb = Some(job_bias_from_solution(pre, &sol)?);
                learned_mp = Some(machine_penalty_from_solution(
                    pre,
                    &sol,
                    challenge.num_machines,
                )?);
                learned_rp = Some(route_pref_from_solution_lite(pre, &sol, challenge)?);
                learn_updates_left -= 1;
            }
        } else {
            stuck = stuck.saturating_add(1);
        }

        push_top_solutions(&mut top_solutions, &sol, mk, 25);
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

    // Phase 1: Broad LS on diverse top solutions (fewer runs, same budget)
    let ls_runs = top_solutions.len().min(6);
    let ls_iters = (effort.fjsp_high_iters / 250).max(10);
    let ls_cands = (effort.fjsp_high_iters / 20).max(96);
    let ls_perturb = (effort.fjsp_high_iters / 80).max(20);

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

    // Phase 2: Greedy reassign
    if let Some(ref sol) = best_solution {
        if let Some((improved_sol, improved_mk)) = greedy_reassign_pass(pre, challenge, sol)? {
            if improved_mk < best_makespan {
                best_makespan = improved_mk;
                best_solution = Some(improved_sol.clone());
                save_solution(&improved_sol)?;
            }
        }
    }

    // Phase 3: Final LS on the absolute best solution (reuse saved budget)
    if let Some(ref sol) = best_solution {
        if let Some((sol3, mk3)) =
            critical_block_move_local_search_ex(pre, challenge, sol, ls_iters, ls_cands, ls_perturb)?
        {
            if mk3 < best_makespan {
                best_makespan = mk3;
                best_solution = Some(sol3.clone());
                save_solution(&sol3)?;
            }
        }
    }

    // Phase 4: Final reassign after LS (chain: reassign→LS→reassign)
    if let Some(ref sol) = best_solution {
        if let Some((rsol, rmk)) = greedy_reassign_pass(pre, challenge, sol)? {
            if rmk < best_makespan {
                best_makespan = rmk;
                best_solution = Some(rsol.clone());
                save_solution(&rsol)?;
            }
        }
    }

    // Phase 5: LNS post-processing (light)
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
