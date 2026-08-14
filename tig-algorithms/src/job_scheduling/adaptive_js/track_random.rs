use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::job_scheduling::*;

use super::types::*;
use super::construction::{construct_solution_conflict, neh_reentrant_flow_solution};
use super::learning::{job_bias_from_solution, machine_penalty_from_solution, route_pref_from_solution_lite};
use super::local_search::critical_block_move_local_search;
use super::rules::{choose_rule_bandit, rule_idx};
use super::helpers::push_top_solutions;

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

    let mut refine_results: Vec<(Solution, u32)> = Vec::new();
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
            refine_results.push((sol, mk));
        }
    }
    for (sol, mk) in refine_results {
        push_top_solutions(&mut top_solutions, sol, mk, 15);
    }

    let ls_runs = top_solutions.len().min(12);

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
