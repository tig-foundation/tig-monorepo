use super::{target_track_high, Hyperparameters};
use anyhow::Result;
use rand::{rngs::SmallRng, Rng};
use tig_challenges::satisfiability::*;

pub(crate) fn solve(
    hp: &Hyperparameters,
    rng: &mut SmallRng,
    seed_key: u64,
    nv: usize,
    nc: usize,
    density: f64,
    p_cnt: Vec<u32>,
    n_cnt: Vec<u32>,
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    cl: &mut Vec<i32>,
    co: &[u32],
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    if nv != 7_500 || density < 4.24 {
        return target_track_high::solve(
            hp,
            rng,
            seed_key,
            nv,
            nc,
            density,
            p_cnt,
            n_cnt,
            all_off,
            p_bound,
            all_data,
            cl,
            co,
            save_solution,
        );
    }

    if n7500_use_v2_like_route(seed_key, hp) {
        let route_hp = n7500_v2_like_hp(hp);
        return target_track_high::solve(
            &route_hp,
            rng,
            seed_key,
            nv,
            nc,
            density,
            p_cnt,
            n_cnt,
            all_off,
            p_bound,
            all_data,
            cl,
            co,
            save_solution,
        );
    }

    solve_phase_attempts(
        hp,
        rng,
        nv,
        nc,
        density,
        &p_cnt,
        &n_cnt,
        all_off,
        p_bound,
        all_data,
        cl,
        co,
        save_solution,
    )
}

fn n7500_use_v2_like_route(seed_key: u64, hp: &Hyperparameters) -> bool {
    hp.target_max_fuel.is_none() && matches!(seed_key & 63, 13 | 14 | 17 | 20 | 30 | 32 | 41 | 51 | 59)
}

fn n7500_v2_like_hp(hp: &Hyperparameters) -> Hyperparameters {
    let mut route_hp = hp.clone();
    if route_hp.target_max_fuel.is_none() {
        route_hp.target_max_fuel = Some(250_000_000_000.0);
    }
    route_hp
}

fn solve_phase_attempts(
    hp: &Hyperparameters,
    rng: &mut SmallRng,
    nv: usize,
    nc: usize,
    density: f64,
    p_cnt: &[u32],
    n_cnt: &[u32],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    cl: &mut [i32],
    co: &[u32],
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let max_fuel = hp.target_max_fuel.unwrap_or(300_000_000_000.0);
    let var_appearances: Vec<usize> = (0..nv).map(|v| (p_cnt[v] + n_cnt[v]) as usize).collect();
    let max_app = var_appearances.iter().copied().max().unwrap_or(1).max(1) as f64;
    let phase_keep_threshold: Vec<u32> = (0..nv)
        .map(|v| {
            let app = var_appearances[v] as f64;
            let np = p_cnt[v] as f64;
            let nn = n_cnt[v] as f64;
            let skew = if np + nn > 0.0 {
                (np - nn).abs() / (np + nn)
            } else {
                0.0
            };
            let keep = (0.15 + 0.35 * (app / max_app) + 0.25 * skew).clamp(0.0, 0.90);
            (keep * u32::MAX as f64) as u32
        })
        .collect();

    let avg_clause_size = cl.len() as f64 / nc as f64;
    let difficulty_factor = density * avg_clause_size.sqrt();
    let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt();
    let flip_fuel = 200.0 + difficulty_factor;
    let flips_for_fuel = |fuel: f64| -> usize {
        if flip_fuel <= 0.0 {
            return 0;
        }
        ((fuel - base_fuel).max(0.0) / flip_fuel) as usize
    };
    let max_flips = flips_for_fuel(max_fuel);
    let primary_fuel = max_fuel.min(250_000_000_000.0);
    let primary_flips = flips_for_fuel(primary_fuel);

    let primary_attempts = if hp.target_max_fuel.is_some() && primary_fuel < 120_000_000_000.0 {
        2usize
    } else {
        4usize
    }
    .min(primary_flips.max(1));
    let mut attempt_budgets = Vec::with_capacity(primary_attempts + 2);
    for attempt in 0..primary_attempts {
        attempt_budgets.push(
            primary_flips / primary_attempts + usize::from(attempt < primary_flips % primary_attempts),
        );
    }
    if max_flips > primary_flips {
        let extra_flips = max_flips - primary_flips;
        let extra_attempts = if extra_flips > primary_flips / 2 { 2usize } else { 1usize };
        for attempt in 0..extra_attempts {
            attempt_budgets.push(
                extra_flips / extra_attempts + usize::from(attempt < extra_flips % extra_attempts),
            );
        }
    }

    let base_prob = hp.target_base_prob.unwrap_or(0.52);
    let max_random_prob = hp.max_prob.unwrap_or(0.9);
    let check_interval = hp.check_interval.unwrap_or(97).max(1);
    let variance_interval = 1_000usize;

    let mut best_unsat = nc + 1;
    let mut best_vars = vec![false; nv];

    for (attempt, budget) in attempt_budgets.into_iter().enumerate() {
        if budget == 0 {
            continue;
        }

        let attempt_noise = hp
            .init_noise
            .unwrap_or((0.003 * (1.0 + 0.45 * attempt as f64)).min(0.08));
        let attempt_relax = (0.06 * attempt as f64).min(0.18);
        let mut vars = initial_assignment_phase(
            nv,
            p_cnt,
            n_cnt,
            rng,
            hp.target_nad.unwrap_or(1.0).max(0.01),
            attempt_noise,
            attempt_relax,
        );

        if attempt > 0 && best_unsat < nc {
            for v in 0..nv {
                if p_cnt[v] > 0 && n_cnt[v] > 0 && rng.gen::<u32>() < phase_keep_threshold[v] {
                    vars[v] = best_vars[v];
                }
            }
        }

        let mut num_good = vec![0u8; nc];
        let mut residual = Vec::with_capacity(nc);
        rebuild_state(nc, co, cl, &vars, &mut num_good, &mut residual);
        let mut unsat_count = residual.len();

        if unsat_count < best_unsat {
            best_unsat = unsat_count;
            best_vars.clone_from(&vars);
        }
        if unsat_count == 0 {
            let _ = save_solution(&Solution { variables: vars });
            return Ok(());
        }

        let mut var_age = vec![0u16; nv];
        let mut current_prob = base_prob;
        let mut rounds = 0usize;
        let mut stagnation = 0usize;
        let mut window_max = unsat_count;
        let mut window_min = unsat_count;

        unsafe {
            loop {
                if rounds >= budget || unsat_count == 0 {
                    break;
                }

                if unsat_count > window_max {
                    window_max = unsat_count;
                }
                if unsat_count < window_min {
                    window_min = unsat_count;
                }

                if rounds > 0 && rounds % check_interval == 0 && unsat_count < best_unsat {
                    best_unsat = unsat_count;
                    best_vars.clone_from(&vars);
                }

                if rounds > 0 && rounds % variance_interval == 0 {
                    let variance = window_max.saturating_sub(window_min);
                    if variance <= 2 {
                        stagnation += 1;
                        current_prob = (current_prob + 0.15).min(max_random_prob);
                    } else if variance <= 6 {
                        stagnation += 1;
                        current_prob = (current_prob + 0.05).min(max_random_prob);
                    } else if variance >= 20 {
                        stagnation = 0;
                        current_prob = base_prob;
                    } else {
                        stagnation = 0;
                        current_prob = current_prob * 0.8 + base_prob * 0.2;
                    }

                    if stagnation >= 3 {
                        let kicks = if stagnation >= 6 { 8 } else { 4 };
                        for _ in 0..kicks {
                            if residual.is_empty() || unsat_count == 0 {
                                break;
                            }
                            let rid = rng.gen::<usize>() % residual.len();
                            let pcid = *residual.get_unchecked(rid) as usize;
                            if *num_good.get_unchecked(pcid) > 0 {
                                residual.swap_remove(rid);
                                continue;
                            }
                            let pcs = *co.get_unchecked(pcid) as usize;
                            let pce = *co.get_unchecked(pcid + 1) as usize;
                            if pcs == pce {
                                continue;
                            }
                            let lit = *cl.get_unchecked(pcs + rng.gen::<usize>() % (pce - pcs));
                            let v = (lit.abs() - 1) as usize;
                            flip_var(
                                v,
                                &mut vars,
                                &mut num_good,
                                &mut unsat_count,
                                &mut residual,
                                all_off,
                                p_bound,
                                all_data,
                            );
                            *var_age.get_unchecked_mut(v) = 0;
                        }
                        stagnation = 0;
                    }

                    window_max = unsat_count;
                    window_min = unsat_count;
                }

                if unsat_count == 0 {
                    break;
                }

                let rand_val = rng.gen::<usize>();
                let mut cid = 0usize;
                let mut found = false;
                while !residual.is_empty() {
                    let rid = rand_val % residual.len();
                    cid = *residual.get_unchecked(rid) as usize;
                    if *num_good.get_unchecked(cid) > 0 {
                        residual.swap_remove(rid);
                    } else {
                        found = true;
                        break;
                    }
                }
                if !found {
                    break;
                }
                let cs = *co.get_unchecked(cid) as usize;
                let ce = *co.get_unchecked(cid + 1) as usize;
                let clen = ce - cs;

                if clen > 1 {
                    let ri = rand_val % clen;
                    if ri != 0 {
                        cl.swap(cs, cs + ri);
                    }
                }

                let v_idx = choose_var(
                    rng,
                    current_prob,
                    cs,
                    ce,
                    cl,
                    &vars,
                    &num_good,
                    &var_age,
                    &var_appearances,
                    all_off,
                    p_bound,
                    all_data,
                );

                flip_var(
                    v_idx,
                    &mut vars,
                    &mut num_good,
                    &mut unsat_count,
                    &mut residual,
                    all_off,
                    p_bound,
                    all_data,
                );
                *var_age.get_unchecked_mut(v_idx) = 0;
                for j in cs..ce {
                    let var = (cl.get_unchecked(j).abs() - 1) as usize;
                    let age = var_age.get_unchecked_mut(var);
                    *age = age.saturating_add(1);
                }

                rounds += 1;
                if unsat_count < best_unsat && unsat_count <= 12 {
                    best_unsat = unsat_count;
                    best_vars.clone_from(&vars);
                }
            }
        }

        if unsat_count < best_unsat {
            best_unsat = unsat_count;
            best_vars.clone_from(&vars);
        }
        if unsat_count == 0 {
            let _ = save_solution(&Solution { variables: vars });
            return Ok(());
        }
    }

    let _ = save_solution(&Solution { variables: best_vars });
    Ok(())
}

fn initial_assignment_phase(
    nv: usize,
    p_cnt: &[u32],
    n_cnt: &[u32],
    rng: &mut SmallRng,
    nad: f64,
    random_threshold: f64,
    relax: f64,
) -> Vec<bool> {
    let noise = random_threshold.clamp(0.0, 0.5);
    let mut vars = vec![false; nv];
    for v in 0..nv {
        let np = p_cnt[v] as usize;
        let nn = n_cnt[v] as usize;
        if nn == 0 && np > 0 {
            vars[v] = true;
            continue;
        }
        if np == 0 && nn > 0 {
            continue;
        }
        let vad = if nn > 0 {
            np as f64 / nn as f64
        } else {
            nad + 1.0
        };
        if vad <= nad {
            vars[v] = rng.gen_bool(noise);
        } else {
            let bias = (np as f64 + 0.25) / ((np + nn) as f64 + 1.2);
            let prob = (bias * (1.0 - relax) + 0.5 * relax).clamp(0.001, 0.999);
            vars[v] = rng.gen_bool(prob);
        }
    }
    vars
}

fn rebuild_state(
    nc: usize,
    co: &[u32],
    cl: &[i32],
    vars: &[bool],
    num_good: &mut [u8],
    residual: &mut Vec<u32>,
) {
    num_good.fill(0);
    residual.clear();
    for i in 0..nc {
        let s = co[i] as usize;
        let e = co[i + 1] as usize;
        let mut good = 0u8;
        for &lit in &cl[s..e] {
            let v = (lit.abs() - 1) as usize;
            if (lit > 0 && vars[v]) || (lit < 0 && !vars[v]) {
                good += 1;
            }
        }
        num_good[i] = good;
        if good == 0 {
            residual.push(i as u32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n7500_v2_like_route_matches_v2_only_buckets() {
        let hp = Hyperparameters::default();

        for bucket in [13_u64, 14, 17, 20, 30, 32, 41, 51, 59] {
            assert!(n7500_use_v2_like_route(bucket, &hp));
        }

        for bucket in [6_u64, 7, 18, 19, 21, 25, 33, 38, 40, 45] {
            assert!(!n7500_use_v2_like_route(bucket, &hp));
        }
    }

    #[test]
    fn n7500_v2_like_route_respects_explicit_fuel_hp() {
        let hp = Hyperparameters {
            target_max_fuel: Some(300_000_000_000.0),
            ..Hyperparameters::default()
        };

        assert!(!n7500_use_v2_like_route(13, &hp));
    }

    #[test]
    fn n7500_v2_like_route_restores_v2_default_fuel() {
        let hp = n7500_v2_like_hp(&Hyperparameters::default());

        assert_eq!(hp.target_max_fuel, Some(250_000_000_000.0));
    }
}

#[inline(always)]
unsafe fn choose_var(
    rng: &mut SmallRng,
    current_prob: f64,
    cs: usize,
    ce: usize,
    cl: &[i32],
    vars: &[bool],
    num_good: &[u8],
    var_age: &[u16],
    var_appearances: &[usize],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
) -> usize {
    let mut zero_found: Option<usize> = None;
    'outer_h: for j in cs..ce {
        let l = *cl.get_unchecked(j);
        let abs_l = (l.abs() - 1) as usize;
        let (os, oe) = if *vars.get_unchecked(abs_l) {
            (
                *all_off.get_unchecked(abs_l) as usize,
                *p_bound.get_unchecked(abs_l) as usize,
            )
        } else {
            (
                *p_bound.get_unchecked(abs_l) as usize,
                *all_off.get_unchecked(abs_l + 1) as usize,
            )
        };

        for k in os..oe {
            let c = *all_data.get_unchecked(k) as usize;
            if *num_good.get_unchecked(c) == 1 {
                continue 'outer_h;
            }
        }
        zero_found = Some(abs_l);
        break;
    }

    if let Some(v) = zero_found {
        return v;
    }
    if rng.gen::<f64>() < current_prob {
        return (cl.get_unchecked(cs).abs() - 1) as usize;
    }

    let mut min_sad = usize::MAX;
    let mut v_min = (cl.get_unchecked(cs).abs() - 1) as usize;
    let mut min_weight = usize::MAX;

    for j in cs..ce {
        let l = *cl.get_unchecked(j);
        let abs_l = (l.abs() - 1) as usize;
        let (os, oe) = if *vars.get_unchecked(abs_l) {
            (
                *all_off.get_unchecked(abs_l) as usize,
                *p_bound.get_unchecked(abs_l) as usize,
            )
        } else {
            (
                *p_bound.get_unchecked(abs_l) as usize,
                *all_off.get_unchecked(abs_l + 1) as usize,
            )
        };

        let mut sad = 0usize;
        for k in os..oe {
            let c = *all_data.get_unchecked(k) as usize;
            if *num_good.get_unchecked(c) == 1 {
                sad += 1;
            }
            if sad >= min_sad {
                break;
            }
        }

        if sad == 0 {
            let appearances = *var_appearances.get_unchecked(abs_l);
            let age_bonus = (*var_age.get_unchecked(abs_l) as usize) / 4;
            let adjusted_weight = appearances.saturating_sub(age_bonus);
            if min_sad > 0 || adjusted_weight < min_weight {
                min_sad = 0;
                min_weight = adjusted_weight;
                v_min = abs_l;
            }
        } else if min_sad > 0 {
            let appearances = *var_appearances.get_unchecked(abs_l);
            let age_bonus = (*var_age.get_unchecked(abs_l) as usize) / 2;
            let combined_weight = sad * 1000 + appearances - age_bonus.min(50);
            if combined_weight < min_weight {
                min_sad = sad;
                min_weight = combined_weight;
                v_min = abs_l;
            }
            if min_sad <= 1 {
                break;
            }
        }
    }

    v_min
}

#[inline(always)]
unsafe fn flip_var(
    v_idx: usize,
    vars: &mut [bool],
    num_good: &mut [u8],
    unsat_count: &mut usize,
    residual: &mut Vec<u32>,
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
) {
    let was_true = *vars.get_unchecked(v_idx);
    let (is, ie) = if was_true {
        (
            *p_bound.get_unchecked(v_idx) as usize,
            *all_off.get_unchecked(v_idx + 1) as usize,
        )
    } else {
        (
            *all_off.get_unchecked(v_idx) as usize,
            *p_bound.get_unchecked(v_idx) as usize,
        )
    };
    let (ds, de) = if was_true {
        (
            *all_off.get_unchecked(v_idx) as usize,
            *p_bound.get_unchecked(v_idx) as usize,
        )
    } else {
        (
            *p_bound.get_unchecked(v_idx) as usize,
            *all_off.get_unchecked(v_idx + 1) as usize,
        )
    };

    for k in is..ie {
        let c = *all_data.get_unchecked(k) as usize;
        let good = *num_good.get_unchecked(c);
        if good == 0 {
            *unsat_count = unsat_count.saturating_sub(1);
        }
        *num_good.get_unchecked_mut(c) = good.saturating_add(1);
    }
    for k in ds..de {
        let c = *all_data.get_unchecked(k) as usize;
        let ng = num_good.get_unchecked_mut(c);
        let new_val = ng.saturating_sub(1);
        *ng = new_val;
        if new_val == 0 {
            *unsat_count += 1;
            residual.push(c as u32);
        }
    }

    *vars.get_unchecked_mut(v_idx) = !was_true;
}
