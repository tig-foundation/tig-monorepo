use super::Hyperparameters;
use anyhow::Result;
use rand::{rngs::SmallRng, Rng};
use tig_challenges::satisfiability::*;

#[inline(always)]
fn initial_residual_capacity(nc: usize) -> usize {
    (nc / 4).saturating_add(16)
}

#[inline(always)]
fn prob_cutoff_u64(prob: f64) -> u64 {
    (prob.max(0.0).min(1.0) * (u64::MAX as f64)) as u64
}

#[inline(always)]
fn lit_var_index(lit: i32) -> usize {
    if lit > 0 {
        lit as usize - 1
    } else {
        (-lit) as usize - 1
    }
}

#[inline(always)]
unsafe fn bump_mid_var_age(var: usize, var_age: &mut [u8]) {
    let age = var_age.get_unchecked_mut(var);
    *age = age.saturating_add(1);
}

#[inline(always)]
unsafe fn bump_mid_clause_var_ages(cs: usize, ce: usize, cl: &[i32], var_age: &mut [u8]) {
    for j in cs..ce {
        let lit = *cl.get_unchecked(j);
        bump_mid_var_age(lit_var_index(lit), var_age);
    }
}

#[inline(always)]
unsafe fn bump_mid_clause_var_ages3(cs: usize, cl: &[i32], var_age: &mut [u8]) {
    debug_assert!(cs + 2 < cl.len());
    bump_mid_var_age(lit_var_index(*cl.get_unchecked(cs)), var_age);
    bump_mid_var_age(lit_var_index(*cl.get_unchecked(cs + 1)), var_age);
    bump_mid_var_age(lit_var_index(*cl.get_unchecked(cs + 2)), var_age);
}

#[inline(always)]
fn select_weighted3_f64_accum_ge(
    threshold: f64,
    count: usize,
    w0: f64,
    v0: usize,
    w1: f64,
    v1: usize,
    w2: f64,
    v2: usize,
) -> usize {
    let mut accum = w0;
    if accum >= threshold {
        return v0;
    }
    if count > 1 {
        accum += w1;
        if accum >= threshold {
            return v1;
        }
        if count > 2 {
            accum += w2;
            if accum >= threshold {
                return v2;
            }
        }
    }
    v0
}

fn build_mid_appearances_if_needed(
    enabled: bool,
    nv: usize,
    p_cnt: &[u32],
    n_cnt: &[u32],
) -> Vec<u8> {
    if enabled {
        let mut appearances = Vec::with_capacity(nv);
        for v in 0..nv {
            appearances.push(((p_cnt[v] + n_cnt[v]) as usize).min(255) as u8);
        }
        appearances
    } else {
        Vec::new()
    }
}

fn build_mid_var_age_if_needed(enabled: bool, nv: usize) -> Vec<u8> {
    if enabled {
        vec![0u8; nv]
    } else {
        Vec::new()
    }
}

fn build_mid_probsat_break_weights(avg_clause_size: f64, p_cnt: &[u32], n_cnt: &[u32]) -> Vec<f64> {
    let limit = max_pair_occurrence_bound(p_cnt, n_cnt).min(255);
    let cb: f64 = if avg_clause_size > 4.5 {
        3.5
    } else if avg_clause_size > 3.5 {
        2.85
    } else {
        2.06
    };

    let mut weights = Vec::with_capacity(limit + 1);
    for i in 0..=limit {
        weights.push(cb.powf(-(i as f64)));
    }
    weights
}

fn max_pair_occurrence_bound(p_cnt: &[u32], n_cnt: &[u32]) -> usize {
    let len = p_cnt.len().min(n_cnt.len());
    let mut max_occ = 0usize;
    for i in 0..len {
        max_occ = max_occ.max(p_cnt[i] as usize).max(n_cnt[i] as usize);
    }
    max_occ
}

#[inline(always)]
unsafe fn mid_probsat_break_count(
    lit: i32,
    num_good: &[u8],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
) -> (usize, usize) {
    let abs_l = lit_var_index(lit);
    let (os, oe) = if lit > 0 {
        (
            *p_bound.get_unchecked(abs_l) as usize,
            *all_off.get_unchecked(abs_l + 1) as usize,
        )
    } else {
        (
            *all_off.get_unchecked(abs_l) as usize,
            *p_bound.get_unchecked(abs_l) as usize,
        )
    };

    let mut sad = 0usize;
    for k in os..oe {
        let c = *all_data.get_unchecked(k) as usize;
        if (*num_good.get_unchecked(c >> 2) >> ((c & 3) << 1)) & 3 == 1 {
            sad += 1;
        }
    }
    (abs_l, sad)
}

#[inline(always)]
unsafe fn choose_mid_probsat_var_generic(
    cs: usize,
    ce: usize,
    cl: &[i32],
    num_good: &[u8],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    probsat_break: &[f64],
    probsat_break_limit: usize,
    rand_val: usize,
    rng: &mut SmallRng,
) -> usize {
    let mut zero0: usize = 0;
    let mut zero1: usize = 0;
    let mut zero2: usize = 0;
    let mut zero_cnt: usize = 0;
    let mut total_weight = 0.0;
    let mut w0 = 0.0f64;
    let mut w1 = 0.0f64;
    let mut w2 = 0.0f64;
    let mut v0 = 0usize;
    let mut v1 = 0usize;
    let mut v2 = 0usize;
    let limit = (ce - cs).min(3);

    for idx in 0..limit {
        let l = *cl.get_unchecked(cs + idx);
        let (abs_l, sad) = mid_probsat_break_count(l, num_good, all_off, p_bound, all_data);

        if sad == 0 {
            match zero_cnt {
                0 => zero0 = abs_l,
                1 => zero1 = abs_l,
                _ => zero2 = abs_l,
            }
            zero_cnt += 1;
            continue;
        }

        if zero_cnt == 0 {
            let weight = *probsat_break.get_unchecked(sad.min(probsat_break_limit));
            match idx {
                0 => {
                    w0 = weight;
                    v0 = abs_l;
                }
                1 => {
                    w1 = weight;
                    v1 = abs_l;
                }
                _ => {
                    w2 = weight;
                    v2 = abs_l;
                }
            }
            total_weight += weight;
        }
    }

    if zero_cnt > 0 {
        match rand_val % zero_cnt {
            0 => zero0,
            1 => zero1,
            _ => zero2,
        }
    } else {
        let threshold = rng.gen::<f64>() * total_weight;
        select_weighted3_f64_accum_ge(threshold, limit, w0, v0, w1, v1, w2, v2)
    }
}

#[inline(always)]
unsafe fn choose_mid_probsat_var3(
    cs: usize,
    cl: &[i32],
    num_good: &[u8],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    probsat_break: &[f64],
    probsat_break_limit: usize,
    rand_val: usize,
    rng: &mut SmallRng,
) -> usize {
    let (v0, sad0) =
        mid_probsat_break_count(*cl.get_unchecked(cs), num_good, all_off, p_bound, all_data);
    let mut zero0: usize = 0;
    let mut zero1: usize = 0;
    let mut zero2: usize = 0;
    let mut zero_cnt: usize = 0;
    let mut w0 = 0.0f64;
    let mut w1 = 0.0f64;
    let mut w2 = 0.0f64;

    if sad0 == 0 {
        zero0 = v0;
        zero_cnt = 1;
    } else {
        w0 = *probsat_break.get_unchecked(sad0.min(probsat_break_limit));
    }

    let (v1, sad1) = mid_probsat_break_count(
        *cl.get_unchecked(cs + 1),
        num_good,
        all_off,
        p_bound,
        all_data,
    );
    if sad1 == 0 {
        match zero_cnt {
            0 => zero0 = v1,
            1 => zero1 = v1,
            _ => zero2 = v1,
        }
        zero_cnt += 1;
    } else if zero_cnt == 0 {
        w1 = *probsat_break.get_unchecked(sad1.min(probsat_break_limit));
    }

    let (v2, sad2) = mid_probsat_break_count(
        *cl.get_unchecked(cs + 2),
        num_good,
        all_off,
        p_bound,
        all_data,
    );
    if sad2 == 0 {
        match zero_cnt {
            0 => zero0 = v2,
            1 => zero1 = v2,
            _ => zero2 = v2,
        }
        zero_cnt += 1;
    } else if zero_cnt == 0 {
        w2 = *probsat_break.get_unchecked(sad2.min(probsat_break_limit));
    }

    if zero_cnt > 0 {
        match rand_val % zero_cnt {
            0 => zero0,
            1 => zero1,
            _ => zero2,
        }
    } else {
        select_weighted3_f64_accum_ge(rng.gen::<f64>() * (w0 + w1 + w2), 3, w0, v0, w1, v1, w2, v2)
    }
}

#[inline(always)]
unsafe fn mid_non_probsat_break_count(
    lit: i32,
    num_good: &[u8],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    stop_at: usize,
) -> (usize, usize) {
    let abs_l = lit_var_index(lit);
    let (os, oe) = if lit > 0 {
        (
            *p_bound.get_unchecked(abs_l) as usize,
            *all_off.get_unchecked(abs_l + 1) as usize,
        )
    } else {
        (
            *all_off.get_unchecked(abs_l) as usize,
            *p_bound.get_unchecked(abs_l) as usize,
        )
    };
    let mut sad = 0usize;
    for k in os..oe {
        let c = *all_data.get_unchecked(k) as usize;
        if (*num_good.get_unchecked(c >> 2) >> ((c & 3) << 1)) & 3 == 1 {
            sad += 1;
        }
        if sad >= stop_at {
            break;
        }
    }
    (abs_l, sad)
}

#[inline(always)]
unsafe fn choose_mid_non_probsat_var_generic(
    cs: usize,
    ce: usize,
    cl: &[i32],
    num_good: &[u8],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    appearances: &[u8],
    var_age: &[u8],
    current_prob_cutoff: u64,
    rand_val: usize,
) -> usize {
    let mut zero0: usize = 0;
    let mut zero1: usize = 0;
    let mut zero2: usize = 0;
    let mut zero_cnt: usize = 0;
    for j in cs..ce {
        let l = *cl.get_unchecked(j);
        let (abs_l, sad) = mid_non_probsat_break_count(l, num_good, all_off, p_bound, all_data, 1);
        if sad == 0 {
            match zero_cnt {
                0 => zero0 = abs_l,
                1 => zero1 = abs_l,
                _ => zero2 = abs_l,
            }
            zero_cnt += 1;
        }
    }

    if zero_cnt > 0 {
        return match rand_val % zero_cnt {
            0 => zero0,
            1 => zero1,
            _ => zero2,
        };
    }

    if (rand_val as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) <= current_prob_cutoff {
        return lit_var_index(*cl.get_unchecked(cs));
    }

    let mut min_sad = usize::MAX;
    let mut v_min = lit_var_index(*cl.get_unchecked(cs));
    let mut min_weight = usize::MAX;
    for j in cs..ce {
        let l = *cl.get_unchecked(j);
        let (abs_l, sad) =
            mid_non_probsat_break_count(l, num_good, all_off, p_bound, all_data, min_sad);
        consider_mid_non_probsat_weighted_candidate(
            abs_l,
            sad,
            appearances,
            var_age,
            &mut min_sad,
            &mut v_min,
            &mut min_weight,
        );
        if min_sad <= 1 {
            break;
        }
    }
    v_min
}

#[inline(always)]
unsafe fn consider_mid_non_probsat_weighted_candidate(
    abs_l: usize,
    sad: usize,
    appearances: &[u8],
    var_age: &[u8],
    min_sad: &mut usize,
    v_min: &mut usize,
    min_weight: &mut usize,
) {
    if sad == 0 {
        let app = *appearances.get_unchecked(abs_l) as usize;
        let age_bonus = (*var_age.get_unchecked(abs_l) as usize) / 4;
        let adjusted_weight = app.saturating_sub(age_bonus);
        if *min_sad > 0 || adjusted_weight < *min_weight {
            *min_sad = 0;
            *min_weight = adjusted_weight;
            *v_min = abs_l;
        }
    } else if *min_sad > 0 {
        let app = *appearances.get_unchecked(abs_l) as usize;
        let age_bonus = (*var_age.get_unchecked(abs_l) as usize) / 2;
        let combined_weight = sad * sad * 1024 + app - age_bonus.min(50);
        if combined_weight < *min_weight {
            *min_sad = sad;
            *min_weight = combined_weight;
            *v_min = abs_l;
        }
    }
}

#[inline(always)]
unsafe fn choose_mid_non_probsat_var3(
    cs: usize,
    cl: &[i32],
    num_good: &[u8],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    appearances: &[u8],
    var_age: &[u8],
    current_prob_cutoff: u64,
    rand_val: usize,
) -> usize {
    let l0 = *cl.get_unchecked(cs);
    let l1 = *cl.get_unchecked(cs + 1);
    let l2 = *cl.get_unchecked(cs + 2);
    let (v0, sad0) = mid_non_probsat_break_count(l0, num_good, all_off, p_bound, all_data, 1);
    let (v1, sad1) = mid_non_probsat_break_count(l1, num_good, all_off, p_bound, all_data, 1);
    let (v2, sad2) = mid_non_probsat_break_count(l2, num_good, all_off, p_bound, all_data, 1);

    let mut zero0 = 0usize;
    let mut zero1 = 0usize;
    let mut zero2 = 0usize;
    let mut zero_cnt = 0usize;
    if sad0 == 0 {
        zero0 = v0;
        zero_cnt = 1;
    }
    if sad1 == 0 {
        match zero_cnt {
            0 => zero0 = v1,
            1 => zero1 = v1,
            _ => zero2 = v1,
        }
        zero_cnt += 1;
    }
    if sad2 == 0 {
        match zero_cnt {
            0 => zero0 = v2,
            1 => zero1 = v2,
            _ => zero2 = v2,
        }
        zero_cnt += 1;
    }
    if zero_cnt > 0 {
        return match rand_val % zero_cnt {
            0 => zero0,
            1 => zero1,
            _ => zero2,
        };
    }

    if (rand_val as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) <= current_prob_cutoff {
        return v0;
    }

    let mut min_sad = usize::MAX;
    let mut v_min = v0;
    let mut min_weight = usize::MAX;
    let (_, sad0) = mid_non_probsat_break_count(l0, num_good, all_off, p_bound, all_data, min_sad);
    consider_mid_non_probsat_weighted_candidate(
        v0,
        sad0,
        appearances,
        var_age,
        &mut min_sad,
        &mut v_min,
        &mut min_weight,
    );
    if min_sad > 1 {
        let (_, sad1) =
            mid_non_probsat_break_count(l1, num_good, all_off, p_bound, all_data, min_sad);
        consider_mid_non_probsat_weighted_candidate(
            v1,
            sad1,
            appearances,
            var_age,
            &mut min_sad,
            &mut v_min,
            &mut min_weight,
        );
        if min_sad > 1 {
            let (_, sad2) =
                mid_non_probsat_break_count(l2, num_good, all_off, p_bound, all_data, min_sad);
            consider_mid_non_probsat_weighted_candidate(
                v2,
                sad2,
                appearances,
                var_age,
                &mut min_sad,
                &mut v_min,
                &mut min_weight,
            );
        }
    }
    v_min
}

#[inline(always)]
unsafe fn remove_mid_unsat_exact(unsat_list: &mut Vec<u32>, unsat_pos: &mut [u32], c: usize) {
    let pos = *unsat_pos.get_unchecked(c) as usize;
    let last_idx = unsat_list.len() - 1;
    if pos != last_idx {
        let last_c = *unsat_list.get_unchecked(last_idx) as usize;
        *unsat_list.get_unchecked_mut(pos) = last_c as u32;
        *unsat_pos.get_unchecked_mut(last_c) = pos as u32;
    }
    *unsat_pos.get_unchecked_mut(c) = u32::MAX;
    unsat_list.set_len(last_idx);
}

pub fn solve(
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
    all_three_clauses: bool,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let nvf = nv as f64;
    let default_fuel = match hp.hw_profile.as_deref() {
        Some("zen5") => 147_000_000_000.0,
        Some("zen4") => 149_000_000_000.0,
        Some("zen5c") => 150_000_000_000.0,
        _ => 149_000_000_000.0,
    };
    let max_fuel = hp.target_max_fuel.unwrap_or(default_fuel);
    let avg_clause_size = cl.len() as f64 / nc as f64;
    let difficulty_factor = density * avg_clause_size.sqrt();
    let scale_factor = if nv > 25000 { 1.5 } else { 1.0 };
    let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt() * scale_factor;
    let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
    let remaining = (max_fuel - base_fuel).max(0.0);
    let max_flips = if flip_fuel > 0.0 {
        (remaining / flip_fuel) as usize
    } else {
        0
    };
    let default_tail_cut_fuel = match hp.hw_profile.as_deref() {
        Some("zen5") => Some(138_000_000_000.0),
        Some("zen4") => Some(140_000_000_000.0),
        _ => None,
    };
    let tail_cut_round = hp
        .target_tail_cut_fuel
        .or(default_tail_cut_fuel)
        .filter(|fuel| *fuel > base_fuel && *fuel < max_fuel && flip_fuel > 0.0)
        .map(|fuel| ((fuel - base_fuel) / flip_fuel) as usize)
        .filter(|round| *round < max_flips);
    let default_tail_cut_unsat_threshold = match hp.hw_profile.as_deref() {
        Some("zen5") => 16,
        Some("zen4") => 16,
        _ => usize::MAX,
    };
    let tail_cut_unsat_threshold = hp
        .target_tail_cut_unsat_threshold
        .unwrap_or(default_tail_cut_unsat_threshold);
    let default_tail_cut_best_unsat_threshold = match hp.hw_profile.as_deref() {
        Some("zen5") => 8,
        Some("zen4") => 8,
        _ => usize::MAX,
    };
    let tail_cut_best_unsat_threshold = hp
        .target_tail_cut_best_unsat_threshold
        .unwrap_or(default_tail_cut_best_unsat_threshold);

    let use_probsat_pick = nv == 100_000 && (4.19..4.21).contains(&density);
    debug_assert_eq!(
        all_three_clauses,
        mid_clause_offsets_are_three(nc, co, cl.len())
    );

    let mut vars = initial_assignment_mid(nv, density, &p_cnt, &n_cnt, rng, hp, seed_key, None);
    let appearances = build_mid_appearances_if_needed(!use_probsat_pick, nv, &p_cnt, &n_cnt);

    let ng_len = (nc + 3) >> 2;
    let mut num_good = vec![0u8; ng_len];
    let mut unsat_list: Vec<u32> = Vec::with_capacity(initial_residual_capacity(nc));
    let mut unsat_pos = vec![u32::MAX; nc];
    rebuild_mid_state_fresh(
        nc,
        co,
        cl,
        &vars,
        &mut num_good,
        &mut unsat_list,
        &mut unsat_pos,
        all_three_clauses,
    );

    if unsat_list.is_empty() {
        let _ = save_solution(&Solution { variables: vars });
        return Ok(());
    }

    let base_prob = if use_probsat_pick {
        0.0
    } else {
        hp.target_base_prob
            .unwrap_or(0.45 + 0.1 * (density / 5.0).min(1.0))
    };
    let mut current_prob = base_prob;
    let mut current_prob_cutoff = prob_cutoff_u64(current_prob);

    let large_problem_scale = ((nvf - 25000.0) / 35000.0).max(0.0).min(1.0);
    let base_interval = 60.0 - 30.0 * large_problem_scale;
    let min_interval = 25.0 - 10.0 * large_problem_scale;
    let density_s = 1.0 / (1.0 + (-(density - 4.0) / 0.5).exp());
    let density_factor = 1.0 + 0.2 * density_s;
    let check_interval = hp.check_interval.unwrap_or(
        (base_interval * density_factor * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval)
            as usize,
    );
    let max_random_prob = if use_probsat_pick {
        0.0
    } else {
        hp.max_prob.unwrap_or(0.9)
    };
    let prob_adjustment_factor = 0.03;
    let smoothing_factor = 0.8;
    let progress_threshold = 0.15 + 0.05 * (density / 3.0).min(1.0);

    let size_scale = 1.0 / (1.0 + (-(nvf - 30000.0) / 7000.0).exp());
    let perturbation_flips = hp
        .perturbation_flips
        .unwrap_or(1 + (2.0 * size_scale) as usize);
    let stagnation_limit = hp
        .stagnation_limit
        .unwrap_or(2 + (2.0 * (1.0 - (density / 5.0).min(1.0))) as usize);

    let mut last_check_residual = unsat_list.len();
    let mut stagnation = 0usize;
    let mut var_age = build_mid_var_age_if_needed(!use_probsat_pick, nv);
    let mut countdown = check_interval;
    let mut rounds = 0usize;
    let mut best_unsat_seen = unsat_list.len();
    let initial_unsat = unsat_list.len();
    let trace_4200 = hp.target_trace_4200.unwrap_or(false) && use_probsat_pick;
    let mut best_unsat_round = 0usize;
    let mut trace_stagnation_ticks = 0usize;
    let mut trace_kick_flips = 0usize;
    let mut trace_tail_cut_triggered = false;
    let mid_best_restart_limit = mid_best_restart_limit_for(hp, use_probsat_pick);
    let mid_best_restart_enabled = mid_best_restart_limit > 0;
    let mid_best_restart_max_unsat = hp.target_mid_best_restart_max_unsat.unwrap_or(24);
    let mid_last_mile_budget = mid_last_mile_budget_for(hp);
    let mid_best_restart_interval = hp
        .target_mid_best_restart_interval
        .or(hp.restart_interval)
        .unwrap_or(8_000_000)
        .max(check_interval.saturating_mul(4).max(1));
    let mid_best_restart_noise_divisor = hp
        .target_mid_best_restart_noise_divisor
        .or(hp.phase_noise_divisor)
        .unwrap_or(1024)
        .max(1);
    let mut best_snapshot_unsat = unsat_list.len();
    let mut best_vars = initial_mid_best_vars(
        mid_best_restart_enabled,
        best_snapshot_unsat,
        mid_best_restart_max_unsat,
        &vars,
    );
    let mut rounds_since_best_snapshot = 0usize;
    let mut mid_best_restart_count = 0usize;
    let mut mid_last_mile_start_unsat = usize::MAX;
    let mut mid_last_mile_end_unsat = usize::MAX;
    let mut mid_last_mile_flips = 0usize;
    let probsat_break = if use_probsat_pick {
        build_mid_probsat_break_weights(avg_clause_size, &p_cnt, &n_cnt)
    } else {
        Vec::new()
    };
    let probsat_break_limit = probsat_break.len().saturating_sub(1);

    macro_rules! run_search {
        () => {{
            unsafe {
                loop {
                    if unsat_list.is_empty() || rounds >= max_flips {
                        break;
                    }
                    if mid_best_restart_count < mid_best_restart_limit
                        && best_snapshot_unsat <= mid_best_restart_max_unsat
                        && rounds_since_best_snapshot >= mid_best_restart_interval
                    {
                        mid_best_restart_count += 1;
                        vars.clone_from(&best_vars);
                        let flips = if nv == 0 {
                            0
                        } else {
                            (nv / mid_best_restart_noise_divisor + 32).min(nv).max(1)
                        };
                        for _ in 0..flips {
                            let v = rng.gen::<usize>() % nv;
                            *vars.get_unchecked_mut(v) = !*vars.get_unchecked(v);
                        }
                        rebuild_mid_state(
                            nc,
                            co,
                            cl,
                            &vars,
                            &mut num_good,
                            &mut unsat_list,
                            &mut unsat_pos,
                            all_three_clauses,
                        );
                        if !use_probsat_pick {
                            var_age.fill(0);
                        }
                        last_check_residual = unsat_list.len();
                        if !use_probsat_pick {
                            current_prob = base_prob;
                            current_prob_cutoff = prob_cutoff_u64(current_prob);
                        }
                        countdown = check_interval;
                        stagnation = 0;
                        rounds_since_best_snapshot = 0;
                        let cur_unsat = unsat_list.len();
                        if cur_unsat < best_unsat_seen {
                            best_unsat_seen = cur_unsat;
                            best_unsat_round = rounds;
                        }
                        if cur_unsat < best_snapshot_unsat
                            && cur_unsat <= mid_best_restart_max_unsat
                        {
                            best_snapshot_unsat = cur_unsat;
                            if cur_unsat > 0 {
                                best_vars.clone_from(&vars);
                            }
                        }
                        continue;
                    }
                    if let Some(cut_round) = tail_cut_round {
                        if rounds >= cut_round {
                            let cur_unsat = unsat_list.len();
                            if cur_unsat < best_unsat_seen {
                                best_unsat_seen = cur_unsat;
                                best_unsat_round = rounds;
                            }
                            if mid_best_restart_enabled
                                && cur_unsat < best_snapshot_unsat
                                && cur_unsat <= mid_best_restart_max_unsat
                            {
                                best_snapshot_unsat = cur_unsat;
                                if cur_unsat > 0 {
                                    best_vars.clone_from(&vars);
                                }
                                rounds_since_best_snapshot = 0;
                            }
                            if cur_unsat > tail_cut_unsat_threshold
                                && best_unsat_seen > tail_cut_best_unsat_threshold
                            {
                                trace_tail_cut_triggered = true;
                                break;
                            }
                        }
                    }

                    countdown -= 1;
                    if countdown == 0 {
                        countdown = check_interval;
                        let cur_residual = unsat_list.len();
                        if tail_cut_round.is_some() && cur_residual < best_unsat_seen {
                            best_unsat_seen = cur_residual;
                            best_unsat_round = rounds;
                        }
                        if mid_best_restart_enabled
                            && cur_residual < best_snapshot_unsat
                            && cur_residual <= mid_best_restart_max_unsat
                        {
                            best_snapshot_unsat = cur_residual;
                            if cur_residual > 0 {
                                best_vars.clone_from(&vars);
                            }
                            rounds_since_best_snapshot = 0;
                        }
                        let progress = last_check_residual as i64 - cur_residual as i64;

                        if progress <= 0 {
                            stagnation += 1;
                            trace_stagnation_ticks += 1;
                            if !use_probsat_pick {
                                let prob_adjustment = prob_adjustment_factor
                                    * (-progress as f64 / last_check_residual.max(1) as f64)
                                        .min(1.0);
                                current_prob =
                                    (current_prob + prob_adjustment).min(max_random_prob);
                            }

                            if stagnation >= stagnation_limit {
                                let kicks = if stagnation >= 5 {
                                    (perturbation_flips * 12).min(100)
                                } else if stagnation >= 4 {
                                    (perturbation_flips * 6).min(50)
                                } else if stagnation >= 3 {
                                    (perturbation_flips * 3).min(20)
                                } else {
                                    (perturbation_flips + 2).min(10)
                                };
                                trace_kick_flips += kicks;

                                for _ in 0..kicks {
                                    if unsat_list.is_empty() {
                                        break;
                                    }
                                    let rid = rng.gen::<usize>() % unsat_list.len();
                                    let pcid = *unsat_list.get_unchecked(rid) as usize;

                                    let (pcs, pce) =
                                        mid_clause_bounds_unchecked(pcid, co, all_three_clauses);
                                    if pcs == pce {
                                        continue;
                                    }
                                    let lit =
                                        *cl.get_unchecked(pcs + rng.gen::<usize>() % (pce - pcs));
                                    let v = lit_var_index(lit);

                                    let was_true = *vars.get_unchecked(v);
                                    let (is, ie) = if was_true {
                                        (
                                            *p_bound.get_unchecked(v) as usize,
                                            *all_off.get_unchecked(v + 1) as usize,
                                        )
                                    } else {
                                        (
                                            *all_off.get_unchecked(v) as usize,
                                            *p_bound.get_unchecked(v) as usize,
                                        )
                                    };
                                    let (ds, de) = if was_true {
                                        (
                                            *all_off.get_unchecked(v) as usize,
                                            *p_bound.get_unchecked(v) as usize,
                                        )
                                    } else {
                                        (
                                            *p_bound.get_unchecked(v) as usize,
                                            *all_off.get_unchecked(v + 1) as usize,
                                        )
                                    };

                                    for k in is..ie {
                                        let c = *all_data.get_unchecked(k) as usize;
                                        let shift = (c & 3) << 1;
                                        let byte_idx = c >> 2;
                                        let old = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
                                        *num_good.get_unchecked_mut(byte_idx) += 1u8 << shift;
                                        if old == 0 {
                                            remove_mid_unsat_exact(
                                                &mut unsat_list,
                                                &mut unsat_pos,
                                                c,
                                            );
                                        }
                                    }

                                    for k in ds..de {
                                        let c = *all_data.get_unchecked(k) as usize;
                                        let shift = (c & 3) << 1;
                                        let byte_idx = c >> 2;
                                        let ng_before =
                                            (*num_good.get_unchecked(byte_idx) >> shift) & 3;
                                        *num_good.get_unchecked_mut(byte_idx) -= 1u8 << shift;
                                        if ng_before == 1 {
                                            *unsat_pos.get_unchecked_mut(c) =
                                                unsat_list.len() as u32;
                                            unsat_list.push(c as u32);
                                        }
                                    }
                                    *vars.get_unchecked_mut(v) = !was_true;
                                    if !use_probsat_pick {
                                        *var_age.get_unchecked_mut(v) = 0;
                                    }
                                }
                                stagnation = 0;
                            }
                        } else if !use_probsat_pick {
                            stagnation = 0;
                            let progress_ratio =
                                progress as f64 / last_check_residual.max(1) as f64;
                            if progress_ratio > progress_threshold {
                                current_prob = base_prob;
                            } else {
                                current_prob = current_prob * smoothing_factor
                                    + base_prob * (1.0 - smoothing_factor);
                            }
                        } else {
                            stagnation = 0;
                        }

                        if !use_probsat_pick {
                            current_prob_cutoff = prob_cutoff_u64(current_prob);
                        }
                        last_check_residual = unsat_list.len();
                    }

                    let rand_val = rng.gen::<usize>();

                    if unsat_list.is_empty() {
                        break;
                    }
                    let cid = *unsat_list.get_unchecked(rand_val % unsat_list.len()) as usize;

                    let (cs, ce) = mid_clause_bounds_unchecked(cid, co, all_three_clauses);
                    let clen = ce - cs;

                    if clen > 1 {
                        let ri = rand_val % clen;
                        if ri != 0 {
                            cl.swap(cs, cs + ri);
                        }
                    }

                    let v_idx = if use_probsat_pick {
                        if all_three_clauses {
                            choose_mid_probsat_var3(
                                cs,
                                cl,
                                &num_good,
                                all_off,
                                p_bound,
                                all_data,
                                &probsat_break,
                                probsat_break_limit,
                                rand_val,
                                rng,
                            )
                        } else {
                            choose_mid_probsat_var_generic(
                                cs,
                                ce,
                                cl,
                                &num_good,
                                all_off,
                                p_bound,
                                all_data,
                                &probsat_break,
                                probsat_break_limit,
                                rand_val,
                                rng,
                            )
                        }
                    } else if all_three_clauses {
                        choose_mid_non_probsat_var3(
                            cs,
                            cl,
                            &num_good,
                            all_off,
                            p_bound,
                            all_data,
                            &appearances,
                            &var_age,
                            current_prob_cutoff,
                            rand_val,
                        )
                    } else {
                        choose_mid_non_probsat_var_generic(
                            cs,
                            ce,
                            cl,
                            &num_good,
                            all_off,
                            p_bound,
                            all_data,
                            &appearances,
                            &var_age,
                            current_prob_cutoff,
                            rand_val,
                        )
                    };

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
                        let shift = (c & 3) << 1;
                        let byte_idx = c >> 2;
                        let old = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
                        *num_good.get_unchecked_mut(byte_idx) += 1u8 << shift;
                        if old == 0 {
                            remove_mid_unsat_exact(&mut unsat_list, &mut unsat_pos, c);
                        }
                    }

                    for k in ds..de {
                        let c = *all_data.get_unchecked(k) as usize;
                        let shift = (c & 3) << 1;
                        let byte_idx = c >> 2;
                        let ng_before = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
                        *num_good.get_unchecked_mut(byte_idx) -= 1u8 << shift;
                        if ng_before == 1 {
                            *unsat_pos.get_unchecked_mut(c) = unsat_list.len() as u32;
                            unsat_list.push(c as u32);
                        }
                    }
                    *vars.get_unchecked_mut(v_idx) = !was_true;
                    if !use_probsat_pick {
                        *var_age.get_unchecked_mut(v_idx) = 0;
                        if all_three_clauses {
                            bump_mid_clause_var_ages3(cs, cl, &mut var_age);
                        } else {
                            bump_mid_clause_var_ages(cs, ce, cl, &mut var_age);
                        }
                    }
                    rounds += 1;
                    let cur_unsat = unsat_list.len();
                    if cur_unsat < best_unsat_seen {
                        best_unsat_seen = cur_unsat;
                        best_unsat_round = rounds;
                    }
                    if mid_best_restart_enabled
                        && cur_unsat < best_snapshot_unsat
                        && cur_unsat <= mid_best_restart_max_unsat
                    {
                        best_snapshot_unsat = cur_unsat;
                        if cur_unsat > 0 {
                            best_vars.clone_from(&vars);
                        }
                        rounds_since_best_snapshot = 0;
                    } else if mid_best_restart_enabled
                        && best_snapshot_unsat <= mid_best_restart_max_unsat
                    {
                        rounds_since_best_snapshot = rounds_since_best_snapshot.saturating_add(1);
                    }
                }
            }
        }};
    }

    run_search!();

    if should_restore_mid_best_snapshot(
        mid_best_restart_enabled,
        best_snapshot_unsat,
        unsat_list.len(),
    ) {
        vars = best_vars;
        rebuild_mid_state(
            nc,
            co,
            cl,
            &vars,
            &mut num_good,
            &mut unsat_list,
            &mut unsat_pos,
            all_three_clauses,
        );
    }

    let mut final_unsat_for_trace = unsat_list.len();
    if mid_last_mile_repair_enabled(hp, use_probsat_pick, unsat_list.len()) {
        mid_last_mile_start_unsat = unsat_list.len();
        let before_sat = nc.saturating_sub(unsat_list.len());
        let (after_sat, used_flips) = refine_mid_last_mile_solution_from_mid_state(
            nv,
            nc,
            co,
            cl,
            all_off,
            p_bound,
            all_data,
            &mut vars,
            mid_last_mile_budget,
            &num_good,
            &unsat_list,
            unsat_list.len(),
            all_three_clauses,
        );
        mid_last_mile_flips = used_flips;
        debug_assert!(after_sat >= before_sat);
        if after_sat > before_sat {
            final_unsat_for_trace = repair_end_unsat(nc, after_sat);
        }
        mid_last_mile_end_unsat = final_unsat_for_trace;
    }

    if trace_4200 {
        eprintln!(
            "c001_r4200_trace_diag seed_key={} max_fuel={:.0} base_fuel={:.3} flip_fuel={:.6} max_flips={} tail_cut_round={} tail_cut_triggered={} initial_unsat={} final_unsat={} best_unsat={} best_round={} best_snapshot_unsat={} mid_best_restarts={} mid_last_mile_start_unsat={} mid_last_mile_end_unsat={} mid_last_mile_flips={} rounds={} stagnation_ticks={} kick_flips={} solved={}",
            seed_key,
            max_fuel,
            base_fuel,
            flip_fuel,
            max_flips,
            tail_cut_round
                .map(|round| round.to_string())
                .unwrap_or_else(|| "none".to_string()),
            trace_tail_cut_triggered,
            initial_unsat,
            final_unsat_for_trace,
            best_unsat_seen,
            best_unsat_round,
            best_snapshot_unsat,
            mid_best_restart_count,
            if mid_last_mile_start_unsat == usize::MAX {
                "none".to_string()
            } else {
                mid_last_mile_start_unsat.to_string()
            },
            if mid_last_mile_end_unsat == usize::MAX {
                "none".to_string()
            } else {
                mid_last_mile_end_unsat.to_string()
            },
            mid_last_mile_flips,
            rounds,
            trace_stagnation_ticks,
            trace_kick_flips,
            final_unsat_for_trace == 0
        );
    }
    let _ = save_solution(&Solution { variables: vars });
    Ok(())
}

fn initial_assignment_mid(
    nv: usize,
    density: f64,
    p_cnt: &[u32],
    n_cnt: &[u32],
    rng: &mut SmallRng,
    hp: &Hyperparameters,
    _seed_key: u64,
    init_noise_override: Option<f64>,
) -> Vec<bool> {
    let nvf = nv as f64;
    let nad = hp.target_nad.unwrap_or(1.0).max(0.01);
    let default_random_threshold = 0.003 + 0.007 / (1.0 + (-(nvf - 30000.0) / 8000.0).exp());
    let random_threshold = init_noise_override
        .or(hp.init_noise)
        .unwrap_or(default_random_threshold)
        .clamp(0.0, 0.5);
    let steep = 0.35 / (1.0 + (density - 4.18).max(0.0) * 12.0);
    let mut vars = Vec::with_capacity(nv);
    for v in 0..nv {
        let np = p_cnt[v] as f64;
        let nn = n_cnt[v] as f64;
        if nn == 0.0 && np > 0.0 {
            vars.push(true);
            continue;
        }
        if np == 0.0 {
            vars.push(false);
            continue;
        }
        let vad = np / nn;
        let bias_prob = (np + 0.25) / (np + nn + 1.2);
        let s = 1.0 / (1.0 + (-(vad - nad) / steep).exp());
        let prob = (random_threshold * (1.0 - s) + bias_prob * s)
            .max(0.0)
            .min(1.0);
        vars.push(rng.gen_bool(prob));
    }
    debug_assert_eq!(vars.len(), nv);
    vars
}

fn mid_best_restart_limit_for(hp: &Hyperparameters, use_probsat_pick: bool) -> usize {
    if !use_probsat_pick || !hp.target_mid_best_restart.unwrap_or(true) {
        return 0;
    }

    hp.target_mid_best_restart_limit.unwrap_or(8).min(64)
}

fn initial_mid_best_vars(
    mid_best_restart_enabled: bool,
    initial_unsat: usize,
    max_unsat: usize,
    vars: &[bool],
) -> Vec<bool> {
    if mid_best_restart_enabled && initial_unsat > 0 && initial_unsat <= max_unsat {
        vars.to_vec()
    } else {
        Vec::new()
    }
}

fn should_restore_mid_best_snapshot(
    mid_best_restart_enabled: bool,
    best_snapshot_unsat: usize,
    current_unsat: usize,
) -> bool {
    mid_best_restart_enabled && current_unsat > 0 && best_snapshot_unsat < current_unsat
}

fn mid_last_mile_repair_enabled(
    hp: &Hyperparameters,
    use_probsat_pick: bool,
    unsat: usize,
) -> bool {
    use_probsat_pick
        && hp.target_mid_last_mile_repair.unwrap_or(true)
        && mid_last_mile_budget_for(hp) > 0
        && unsat > 0
        && unsat <= hp.target_mid_last_mile_max_unsat.unwrap_or(24)
}

fn mid_last_mile_budget_for(hp: &Hyperparameters) -> usize {
    hp.target_mid_last_mile_budget.unwrap_or(512).min(20_000)
}

#[inline(always)]
fn repair_end_unsat(nc: usize, after_sat: usize) -> usize {
    nc.saturating_sub(after_sat)
}

#[inline(always)]
fn last_mile_unsat_capacity(nc: usize) -> usize {
    nc.min(64)
}

#[inline(always)]
fn last_mile_clause_var_capacity(unsat_len: usize) -> usize {
    unsat_len.saturating_mul(8).min(256)
}

const REPAIR_VAR_STACK_CAPACITY: usize = 256;
const REPAIR_SINGLE_CLAUSE_STACK_CAPACITY: usize = 4;

struct RepairVarBuffer {
    stack: [(usize, u8); REPAIR_VAR_STACK_CAPACITY],
    len: usize,
    overflow: Vec<(usize, u8)>,
}

impl RepairVarBuffer {
    fn new() -> Self {
        Self {
            stack: [(0, 0); REPAIR_VAR_STACK_CAPACITY],
            len: 0,
            overflow: Vec::new(),
        }
    }

    fn push_or_mark_polarity(&mut self, v: usize, want: u8) {
        if self.len > 0 {
            let last = &mut self.stack[self.len - 1];
            if last.0 == v {
                if last.1 != want {
                    last.1 = 3;
                }
                return;
            }
        }
        for i in 0..self.len.saturating_sub(1) {
            if self.stack[i].0 == v {
                if self.stack[i].1 != want {
                    self.stack[i].1 = 3;
                }
                return;
            }
        }
        if !self.overflow.is_empty() {
            let last_idx = self.overflow.len() - 1;
            let last = &mut self.overflow[last_idx];
            if last.0 == v {
                if last.1 != want {
                    last.1 = 3;
                }
                return;
            }
            if let Some((_, polarity)) = self.overflow[..last_idx]
                .iter_mut()
                .find(|(seen_v, _)| *seen_v == v)
            {
                if *polarity != want {
                    *polarity = 3;
                }
                return;
            }
        }
        if self.len < REPAIR_VAR_STACK_CAPACITY {
            self.stack[self.len] = (v, want);
            self.len += 1;
        } else {
            self.overflow.push((v, want));
        }
    }

    fn stack_slice(&self) -> &[(usize, u8)] {
        &self.stack[..self.len]
    }

    fn overflow_slice(&self) -> &[(usize, u8)] {
        &self.overflow
    }
}

struct RepairCandidateBuffer {
    stack: [usize; REPAIR_VAR_STACK_CAPACITY],
    len: usize,
    overflow: Vec<usize>,
}

impl RepairCandidateBuffer {
    fn new() -> Self {
        Self {
            stack: [0; REPAIR_VAR_STACK_CAPACITY],
            len: 0,
            overflow: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.len = 0;
        self.overflow.clear();
    }

    fn push_unique(&mut self, v: usize) {
        if self.len > 0 && self.stack[self.len - 1] == v {
            return;
        }
        for &seen_v in &self.stack[..self.len.saturating_sub(1)] {
            if seen_v == v {
                return;
            }
        }
        if !self.overflow.is_empty() {
            let last_idx = self.overflow.len() - 1;
            if self.overflow[last_idx] == v {
                return;
            }
            if self.overflow[..last_idx].iter().any(|&seen_v| seen_v == v) {
                return;
            }
        }
        if self.len < REPAIR_VAR_STACK_CAPACITY {
            self.stack[self.len] = v;
            self.len += 1;
        } else {
            self.overflow.push(v);
        }
    }

    fn stack_slice(&self) -> &[usize] {
        &self.stack[..self.len]
    }

    fn overflow_slice(&self) -> &[usize] {
        &self.overflow
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SingleUnsatRepairChoice {
    UseGeneral,
    Flip(usize),
    Stop,
}

#[inline(always)]
fn repair_seen_contains(
    seen: &[usize; REPAIR_SINGLE_CLAUSE_STACK_CAPACITY],
    len: usize,
    v: usize,
) -> bool {
    match len {
        0 => false,
        1 => seen[0] == v,
        2 => seen[0] == v || seen[1] == v,
        3 => seen[0] == v || seen[1] == v || seen[2] == v,
        4 => seen[0] == v || seen[1] == v || seen[2] == v || seen[3] == v,
        _ => false,
    }
}

#[inline(always)]
fn repair_touched_position(
    touched: &[(usize, u8); REPAIR_SINGLE_CLAUSE_STACK_CAPACITY],
    len: usize,
    v: usize,
) -> Option<usize> {
    match len {
        0 => None,
        1 => {
            if touched[0].0 == v {
                Some(0)
            } else {
                None
            }
        }
        2 => {
            if touched[0].0 == v {
                Some(0)
            } else if touched[1].0 == v {
                Some(1)
            } else {
                None
            }
        }
        3 => {
            if touched[0].0 == v {
                Some(0)
            } else if touched[1].0 == v {
                Some(1)
            } else if touched[2].0 == v {
                Some(2)
            } else {
                None
            }
        }
        4 => {
            if touched[0].0 == v {
                Some(0)
            } else if touched[1].0 == v {
                Some(1)
            } else if touched[2].0 == v {
                Some(2)
            } else if touched[3].0 == v {
                Some(3)
            } else {
                None
            }
        }
        _ => None,
    }
}

#[inline(always)]
fn apply_safe_purity_touched_entry(
    entry: (usize, u8),
    vars: &mut [bool],
    counts: &mut RepairCounts<'_>,
    unsat: &mut Vec<u32>,
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    used: &mut usize,
    budget: usize,
) -> bool {
    if *used >= budget {
        return true;
    }
    let (v, polarity) = entry;
    let want = match polarity {
        1 => true,
        2 => false,
        _ => return false,
    };
    if vars[v] != want && safe_to_flip(v, vars, counts, all_off, p_bound, all_data) {
        flip_repair_var(v, vars, counts, unsat, all_off, p_bound, all_data);
        *used += 1;
        if unsat.is_empty() {
            return true;
        }
    }
    false
}

enum RepairCountBase<'a> {
    Dense(&'a [u8]),
    Packed(&'a [u8]),
}

struct RepairCounts<'a> {
    base: RepairCountBase<'a>,
    overrides: Vec<RepairCountOverride>,
}

#[derive(Clone, Copy)]
struct RepairCountOverride {
    clause: u32,
    value: u8,
}

impl<'a> RepairCounts<'a> {
    fn from_dense(base: &'a [u8]) -> Self {
        Self::from_dense_with_capacity(base, 0)
    }

    fn from_dense_with_capacity(base: &'a [u8], override_capacity: usize) -> Self {
        Self {
            base: RepairCountBase::Dense(base),
            overrides: Vec::with_capacity(override_capacity),
        }
    }

    fn from_packed(base: &'a [u8], override_capacity: usize) -> Self {
        Self {
            base: RepairCountBase::Packed(base),
            overrides: Vec::with_capacity(override_capacity),
        }
    }

    fn find_override(&self, c: usize) -> Result<usize, usize> {
        let override_len = self.overrides.len();
        if override_len == 0 {
            return Err(0);
        }
        let key = c as u32;
        let first = self.overrides[0].clause;
        if key <= first {
            return if key == first { Ok(0) } else { Err(0) };
        }
        if override_len == 1 {
            return Err(1);
        }
        let last_pos = override_len - 1;
        let last = self.overrides[last_pos].clause;
        if key >= last {
            return if key == last {
                Ok(last_pos)
            } else {
                Err(override_len)
            };
        }
        if override_len <= 4 {
            for (pos, entry) in self.overrides[1..last_pos].iter().enumerate() {
                if key == entry.clause {
                    return Ok(pos + 1);
                }
                if key < entry.clause {
                    return Err(pos + 1);
                }
            }
            return Err(last_pos);
        }
        self.overrides
            .binary_search_by_key(&key, |entry| entry.clause)
    }

    fn base_get(&self, c: usize) -> u8 {
        match self.base {
            RepairCountBase::Dense(base) => base[c],
            RepairCountBase::Packed(base) => packed_num_good_at(base, c),
        }
    }

    fn get(&self, c: usize) -> u8 {
        let override_len = self.overrides.len();
        if override_len == 0 {
            return self.base_get(c);
        }
        let key = c as u32;
        let first = self.overrides.first().unwrap();
        if override_len == 1 {
            return if key == first.clause {
                first.value
            } else {
                self.base_get(c)
            };
        }
        if key < first.clause {
            return self.base_get(c);
        }
        if key == first.clause {
            return first.value;
        }
        let last = self.overrides.last().unwrap();
        if key > last.clause {
            return self.base_get(c);
        }
        if key == last.clause {
            return last.value;
        }
        if override_len <= 4 {
            if override_len > 2 {
                let entry = self.overrides[1];
                if key <= entry.clause {
                    return if key == entry.clause {
                        entry.value
                    } else {
                        self.base_get(c)
                    };
                }
                if override_len > 3 {
                    let entry = self.overrides[2];
                    if key == entry.clause {
                        return entry.value;
                    }
                }
            }
            return self.base_get(c);
        }
        if let Ok(pos) = self.find_override(c) {
            return self.overrides[pos].value;
        }
        self.base_get(c)
    }

    #[inline(always)]
    fn count_occurrences_eq(
        &self,
        all_data: &[u32],
        start: usize,
        end: usize,
        target: u8,
    ) -> usize {
        let mut count = 0usize;
        if self.overrides.is_empty() {
            match self.base {
                RepairCountBase::Dense(base) => {
                    for &clause in &all_data[start..end] {
                        if base[clause as usize] == target {
                            count += 1;
                        }
                    }
                }
                RepairCountBase::Packed(base) => {
                    for &clause in &all_data[start..end] {
                        if packed_num_good_at(base, clause as usize) == target {
                            count += 1;
                        }
                    }
                }
            }
        } else {
            for &clause in &all_data[start..end] {
                if self.get(clause as usize) == target {
                    count += 1;
                }
            }
        }
        count
    }

    #[inline(always)]
    fn all_occurrences_gt(
        &self,
        all_data: &[u32],
        start: usize,
        end: usize,
        threshold: u8,
    ) -> bool {
        if self.overrides.is_empty() {
            match self.base {
                RepairCountBase::Dense(base) => {
                    for &clause in &all_data[start..end] {
                        if base[clause as usize] <= threshold {
                            return false;
                        }
                    }
                }
                RepairCountBase::Packed(base) => {
                    for &clause in &all_data[start..end] {
                        if packed_num_good_at(base, clause as usize) <= threshold {
                            return false;
                        }
                    }
                }
            }
        } else {
            for &clause in &all_data[start..end] {
                if self.get(clause as usize) <= threshold {
                    return false;
                }
            }
        }
        true
    }

    fn set(&mut self, c: usize, value: u8) {
        let key = c as u32;
        let base = self.base_get(c);
        if value == base {
            if let Ok(pos) = self.find_override(c) {
                self.overrides.remove(pos);
            }
            return;
        }
        if self.overrides.is_empty() {
            self.overrides
                .push(RepairCountOverride { clause: key, value });
            return;
        }
        if let Some(last) = self.overrides.last_mut() {
            if last.clause == key {
                last.value = value;
                return;
            }
            if last.clause < key {
                self.overrides
                    .push(RepairCountOverride { clause: key, value });
                return;
            }
        }
        if self.overrides.len() == 1 {
            self.overrides
                .insert(0, RepairCountOverride { clause: key, value });
            return;
        }
        match self.find_override(c) {
            Ok(pos) => self.overrides[pos].value = value,
            Err(pos) => self
                .overrides
                .insert(pos, RepairCountOverride { clause: key, value }),
        }
    }

    fn increment(&mut self, c: usize) -> u8 {
        let key = c as u32;
        let base = self.base_get(c);
        if self.overrides.is_empty() {
            let old = base;
            let value = old.saturating_add(1);
            if value != old {
                self.overrides
                    .push(RepairCountOverride { clause: key, value });
            }
            return old;
        }
        if let Some(last) = self.overrides.last_mut() {
            if last.clause == key {
                let old = last.value;
                let value = old.saturating_add(1);
                if value == base {
                    self.overrides.pop();
                } else {
                    last.value = value;
                }
                return old;
            }
            if last.clause < key {
                let old = base;
                let value = old.saturating_add(1);
                if value != old {
                    self.overrides
                        .push(RepairCountOverride { clause: key, value });
                }
                return old;
            }
        }
        if self.overrides.len() == 1 {
            let old = base;
            let value = old.saturating_add(1);
            if value != old {
                self.overrides
                    .insert(0, RepairCountOverride { clause: key, value });
            }
            return old;
        }
        match self.find_override(c) {
            Ok(pos) => {
                let old = self.overrides[pos].value;
                let value = old.saturating_add(1);
                if value == base {
                    self.overrides.remove(pos);
                } else {
                    self.overrides[pos].value = value;
                }
                old
            }
            Err(pos) => {
                let old = base;
                let value = old.saturating_add(1);
                if value != old {
                    self.overrides
                        .insert(pos, RepairCountOverride { clause: key, value });
                }
                old
            }
        }
    }

    fn decrement(&mut self, c: usize) -> u8 {
        let key = c as u32;
        let base = self.base_get(c);
        if self.overrides.is_empty() {
            let old = base;
            let new_value = old.saturating_sub(1);
            if new_value != old {
                self.overrides.push(RepairCountOverride {
                    clause: key,
                    value: new_value,
                });
            }
            return new_value;
        }
        if let Some(last) = self.overrides.last_mut() {
            if last.clause == key {
                let new_value = last.value.saturating_sub(1);
                if new_value == base {
                    self.overrides.pop();
                } else {
                    last.value = new_value;
                }
                return new_value;
            }
            if last.clause < key {
                let old = base;
                let new_value = old.saturating_sub(1);
                if new_value != old {
                    self.overrides.push(RepairCountOverride {
                        clause: key,
                        value: new_value,
                    });
                }
                return new_value;
            }
        }
        if self.overrides.len() == 1 {
            let old = base;
            let new_value = old.saturating_sub(1);
            if new_value != old {
                self.overrides.insert(
                    0,
                    RepairCountOverride {
                        clause: key,
                        value: new_value,
                    },
                );
            }
            return new_value;
        }
        match self.find_override(c) {
            Ok(pos) => {
                let new_value = self.overrides[pos].value.saturating_sub(1);
                if new_value == base {
                    self.overrides.remove(pos);
                } else {
                    self.overrides[pos].value = new_value;
                }
                new_value
            }
            Err(pos) => {
                let old = base;
                let new_value = old.saturating_sub(1);
                if new_value != old {
                    self.overrides.insert(
                        pos,
                        RepairCountOverride {
                            clause: key,
                            value: new_value,
                        },
                    );
                }
                new_value
            }
        }
    }
}

#[inline(always)]
fn packed_num_good_at(packed_num_good: &[u8], c: usize) -> u8 {
    let shift = (c & 3) << 1;
    (packed_num_good[c >> 2] >> shift) & 3
}

#[inline(always)]
fn packed_num_good_byte_has_zero(byte: u8) -> bool {
    ((byte | (byte >> 1)) & 0x55) != 0x55
}

fn refine_mid_last_mile_solution(
    _nv: usize,
    nc: usize,
    co: &[u32],
    cl: &[i32],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    vars: &mut [bool],
    budget: usize,
) -> (usize, usize) {
    let all_three_clauses = mid_clause_offsets_are_three(nc, co, cl.len());
    if budget == 0 {
        return (count_satisfied_mid(nc, co, cl, vars, all_three_clauses), 0);
    }

    let mut num_good = vec![0u8; nc];
    let mut unsat = Vec::with_capacity(last_mile_unsat_capacity(nc));
    rebuild_repair_state(
        nc,
        co,
        cl,
        vars,
        &mut num_good,
        &mut unsat,
        all_three_clauses,
    );
    if unsat.is_empty() {
        return (nc, 0);
    }
    let mut counts = RepairCounts::from_dense_with_capacity(
        &num_good,
        last_mile_clause_var_capacity(unsat.len().max(1)),
    );
    finish_mid_last_mile_repair(
        co,
        cl,
        all_off,
        p_bound,
        all_data,
        vars,
        budget,
        &mut counts,
        &mut unsat,
        nc,
        all_three_clauses,
    )
}

fn refine_mid_last_mile_solution_from_mid_state(
    _nv: usize,
    nc: usize,
    co: &[u32],
    cl: &[i32],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    vars: &mut [bool],
    budget: usize,
    packed_num_good: &[u8],
    current_unsat: &[u32],
    expected_unsat: usize,
    all_three_clauses: bool,
) -> (usize, usize) {
    if budget == 0 {
        return (count_satisfied_mid(nc, co, cl, vars, all_three_clauses), 0);
    }

    let mut unsat = Vec::with_capacity(last_mile_unsat_capacity(
        current_unsat.len().max(expected_unsat).max(1),
    ));
    rebuild_repair_unsat_from_mid_tail(nc, packed_num_good, current_unsat, &mut unsat);
    if unsat.is_empty() {
        return (nc, 0);
    }
    let actual_unsat = unsat.len();
    let mut counts = RepairCounts::from_packed(
        packed_num_good,
        last_mile_clause_var_capacity(actual_unsat.max(1)),
    );
    finish_mid_last_mile_repair(
        co,
        cl,
        all_off,
        p_bound,
        all_data,
        vars,
        budget,
        &mut counts,
        &mut unsat,
        nc,
        all_three_clauses,
    )
}

fn finish_mid_last_mile_repair(
    co: &[u32],
    cl: &[i32],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    vars: &mut [bool],
    budget: usize,
    counts: &mut RepairCounts<'_>,
    unsat: &mut Vec<u32>,
    nc: usize,
    all_three_clauses: bool,
) -> (usize, usize) {
    if unsat.is_empty() {
        return (nc, 0);
    }

    let mut used = safe_purity_repair(
        co,
        cl,
        all_off,
        p_bound,
        all_data,
        vars,
        counts,
        unsat,
        budget,
        all_three_clauses,
    );
    if !unsat.is_empty() && used < budget {
        used += greedy_net_repair(
            co,
            cl,
            all_off,
            p_bound,
            all_data,
            vars,
            counts,
            unsat,
            budget - used,
            all_three_clauses,
        );
    }

    (nc - unsat.len(), used)
}

fn count_satisfied_mid(
    nc: usize,
    co: &[u32],
    cl: &[i32],
    vars: &[bool],
    all_three_clauses: bool,
) -> usize {
    let mut sat = 0usize;
    debug_assert!(!all_three_clauses || mid_clause_offsets_are_three(nc, co, cl.len()));
    if all_three_clauses {
        debug_assert!(co.len() > nc);
        debug_assert_eq!(co[0], 0);
        debug_assert_eq!(co[nc] as usize, cl.len());
        debug_assert!((0..=nc).all(|i| co[i] as usize == i * 3));

        for c in 0..nc {
            if mid_clause3_is_satisfied(c * 3, cl, vars) {
                sat += 1;
            }
        }
        return sat;
    }

    for c in 0..nc {
        let s = co[c] as usize;
        let e = co[c + 1] as usize;
        if mid_clause_is_satisfied(s, e, cl, vars) {
            sat += 1;
        }
    }
    sat
}

#[inline(always)]
fn mid_clause_offsets_are_three(nc: usize, co: &[u32], cl_len: usize) -> bool {
    if co.len() <= nc || cl_len != nc.saturating_mul(3) {
        return false;
    }
    for (i, &off) in co.iter().take(nc + 1).enumerate() {
        if off as usize != i * 3 {
            return false;
        }
    }
    true
}

#[inline(always)]
unsafe fn mid_clause_bounds_unchecked(
    cid: usize,
    co: &[u32],
    all_three_clauses: bool,
) -> (usize, usize) {
    if all_three_clauses {
        let s = cid * 3;
        (s, s + 3)
    } else {
        (
            *co.get_unchecked(cid) as usize,
            *co.get_unchecked(cid + 1) as usize,
        )
    }
}

#[inline(always)]
fn mid_repair_clause_bounds(cid: usize, co: &[u32], all_three_clauses: bool) -> (usize, usize) {
    if all_three_clauses {
        let s = cid * 3;
        (s, s + 3)
    } else {
        (co[cid] as usize, co[cid + 1] as usize)
    }
}

#[inline(always)]
fn mid_lit_is_satisfied(lit: i32, vars: &[bool]) -> bool {
    (lit > 0 && vars[lit_var_index(lit)]) || (lit < 0 && !vars[lit_var_index(lit)])
}

#[inline(always)]
fn mid_clause3_is_satisfied(s: usize, cl: &[i32], vars: &[bool]) -> bool {
    mid_lit_is_satisfied(cl[s], vars)
        || mid_lit_is_satisfied(cl[s + 1], vars)
        || mid_lit_is_satisfied(cl[s + 2], vars)
}

#[inline(always)]
fn mid_lit_good(lit: i32, vars: &[bool]) -> u8 {
    mid_lit_is_satisfied(lit, vars) as u8
}

#[inline(always)]
fn mid_clause_is_satisfied(s: usize, e: usize, cl: &[i32], vars: &[bool]) -> bool {
    match e - s {
        1 => mid_lit_is_satisfied(cl[s], vars),
        2 => mid_lit_is_satisfied(cl[s], vars) || mid_lit_is_satisfied(cl[s + 1], vars),
        3 => {
            mid_lit_is_satisfied(cl[s], vars)
                || mid_lit_is_satisfied(cl[s + 1], vars)
                || mid_lit_is_satisfied(cl[s + 2], vars)
        }
        _ => cl[s..e].iter().any(|&lit| mid_lit_is_satisfied(lit, vars)),
    }
}

fn rebuild_repair_state(
    nc: usize,
    co: &[u32],
    cl: &[i32],
    vars: &[bool],
    num_good: &mut [u8],
    unsat: &mut Vec<u32>,
    all_three_clauses: bool,
) {
    debug_assert!(num_good.len() >= nc);
    unsat.clear();
    debug_assert!(!all_three_clauses || mid_clause_offsets_are_three(nc, co, cl.len()));
    if all_three_clauses {
        debug_assert!(co.len() > nc);
        debug_assert_eq!(co[0], 0);
        debug_assert_eq!(co[nc] as usize, cl.len());
        debug_assert!((0..=nc).all(|i| co[i] as usize == i * 3));

        for c in 0..nc {
            let good = repair_clause3_good_count(c * 3, cl, vars);
            num_good[c] = good;
            if good == 0 {
                unsat.push(c as u32);
            }
        }
        return;
    }

    for c in 0..nc {
        let s = co[c] as usize;
        let e = co[c + 1] as usize;
        let good = repair_clause_good_count(s, e, cl, vars);
        num_good[c] = good;
        if good == 0 {
            unsat.push(c as u32);
        }
    }
}

#[inline(always)]
fn repair_clause3_good_count(s: usize, cl: &[i32], vars: &[bool]) -> u8 {
    repair_lit_good(cl[s], vars)
        + repair_lit_good(cl[s + 1], vars)
        + repair_lit_good(cl[s + 2], vars)
}

#[inline(always)]
fn repair_lit_good(lit: i32, vars: &[bool]) -> u8 {
    mid_lit_good(lit, vars)
}

#[inline(always)]
fn repair_clause_good_count(s: usize, e: usize, cl: &[i32], vars: &[bool]) -> u8 {
    match e - s {
        1 => repair_lit_good(cl[s], vars),
        2 => repair_lit_good(cl[s], vars) + repair_lit_good(cl[s + 1], vars),
        3 => {
            repair_lit_good(cl[s], vars)
                + repair_lit_good(cl[s + 1], vars)
                + repair_lit_good(cl[s + 2], vars)
        }
        _ => {
            let mut good = 0u8;
            for &lit in &cl[s..e] {
                let v = lit_var_index(lit);
                if (lit > 0 && vars[v]) || (lit < 0 && !vars[v]) {
                    good = good.saturating_add(1);
                }
            }
            good
        }
    }
}

fn rebuild_repair_unsat_from_mid_state(nc: usize, packed_num_good: &[u8], unsat: &mut Vec<u32>) {
    unsat.clear();
    let byte_count = (nc + 3) >> 2;
    for byte_idx in 0..byte_count {
        let byte = packed_num_good[byte_idx];
        if !packed_num_good_byte_has_zero(byte) {
            continue;
        }
        let base = byte_idx << 2;
        for lane in 0..4 {
            let c = base + lane;
            if c >= nc {
                break;
            }
            if ((byte >> (lane << 1)) & 3) == 0 {
                unsat.push(c as u32);
            }
        }
    }
}

fn rebuild_repair_unsat_from_mid_tail(
    nc: usize,
    packed_num_good: &[u8],
    current_unsat: &[u32],
    unsat: &mut Vec<u32>,
) {
    unsat.clear();
    for &cid in current_unsat {
        let c = cid as usize;
        debug_assert!(c < nc);
        if c < nc && packed_num_good_at(packed_num_good, c) == 0 {
            unsat.push(cid);
        }
    }

    debug_assert!({
        let mut scanned = Vec::with_capacity(last_mile_unsat_capacity(nc));
        rebuild_repair_unsat_from_mid_state(nc, packed_num_good, &mut scanned);
        scanned.as_slice() == unsat.as_slice()
    });
}

fn safe_purity_repair(
    co: &[u32],
    cl: &[i32],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    vars: &mut [bool],
    counts: &mut RepairCounts<'_>,
    unsat: &mut Vec<u32>,
    budget: usize,
    all_three_clauses: bool,
) -> usize {
    if unsat.len() == 1 {
        if let Some(used) = safe_purity_repair_single_unsat(
            unsat[0] as usize,
            co,
            cl,
            all_off,
            p_bound,
            all_data,
            vars,
            counts,
            unsat,
            budget,
            all_three_clauses,
        ) {
            return used;
        }
    }

    let mut touched = RepairVarBuffer::new();
    for &cid in unsat.iter() {
        let c = cid as usize;
        let (s, e) = mid_repair_clause_bounds(c, co, all_three_clauses);
        for &lit in &cl[s..e] {
            let v = lit_var_index(lit);
            let want = if lit > 0 { 1 } else { 2 };
            touched.push_or_mark_polarity(v, want);
        }
    }

    let mut used = 0usize;
    for &entry in touched.stack_slice() {
        if apply_safe_purity_touched_entry(
            entry, vars, counts, unsat, all_off, p_bound, all_data, &mut used, budget,
        ) {
            return used;
        }
    }
    if !touched.overflow_slice().is_empty() {
        for &entry in touched.overflow_slice() {
            if apply_safe_purity_touched_entry(
                entry, vars, counts, unsat, all_off, p_bound, all_data, &mut used, budget,
            ) {
                return used;
            }
        }
    }
    used
}

fn safe_purity_repair_single_unsat(
    c: usize,
    co: &[u32],
    cl: &[i32],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    vars: &mut [bool],
    counts: &mut RepairCounts<'_>,
    unsat: &mut Vec<u32>,
    budget: usize,
    all_three_clauses: bool,
) -> Option<usize> {
    let (s, e) = mid_repair_clause_bounds(c, co, all_three_clauses);
    let mut touched = [(usize::MAX, 0u8); REPAIR_SINGLE_CLAUSE_STACK_CAPACITY];
    let mut touched_len = 0usize;

    for &lit in &cl[s..e] {
        let v = lit_var_index(lit);
        let want = if lit > 0 { 1 } else { 2 };
        if let Some(i) = repair_touched_position(&touched, touched_len, v) {
            if touched[i].1 != want {
                touched[i].1 = 3;
            }
            continue;
        }
        if touched_len == touched.len() {
            return None;
        }
        touched[touched_len] = (v, want);
        touched_len += 1;
    }

    let mut used = 0usize;
    if touched_len > 0
        && apply_safe_purity_touched_entry(
            touched[0], vars, counts, unsat, all_off, p_bound, all_data, &mut used, budget,
        )
    {
        return Some(used);
    }
    if touched_len > 1
        && apply_safe_purity_touched_entry(
            touched[1], vars, counts, unsat, all_off, p_bound, all_data, &mut used, budget,
        )
    {
        return Some(used);
    }
    if touched_len > 2
        && apply_safe_purity_touched_entry(
            touched[2], vars, counts, unsat, all_off, p_bound, all_data, &mut used, budget,
        )
    {
        return Some(used);
    }
    if touched_len > 3 {
        apply_safe_purity_touched_entry(
            touched[3], vars, counts, unsat, all_off, p_bound, all_data, &mut used, budget,
        );
    }
    Some(used)
}

fn safe_to_flip(
    v: usize,
    vars: &[bool],
    counts: &RepairCounts<'_>,
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
) -> bool {
    let was_true = vars[v];
    let (dec_s, dec_e) = if was_true {
        (all_off[v] as usize, p_bound[v] as usize)
    } else {
        (p_bound[v] as usize, all_off[v + 1] as usize)
    };
    counts.all_occurrences_gt(all_data, dec_s, dec_e, 1)
}

#[inline(always)]
fn consider_greedy_repair_candidate(
    v: usize,
    vars: &[bool],
    counts: &RepairCounts<'_>,
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    best_v: &mut usize,
    best_net: &mut i32,
    best_make: &mut usize,
    best_break: &mut usize,
) {
    let (make, brk) = flip_make_break(v, vars, counts, all_off, p_bound, all_data);
    let net = make as i32 - brk as i32;
    if net > *best_net || (net == *best_net && net > 0 && (make > *best_make || brk < *best_break))
    {
        *best_net = net;
        *best_make = make;
        *best_break = brk;
        *best_v = v;
    }
}

fn greedy_net_repair(
    co: &[u32],
    cl: &[i32],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    vars: &mut [bool],
    counts: &mut RepairCounts<'_>,
    unsat: &mut Vec<u32>,
    budget: usize,
    all_three_clauses: bool,
) -> usize {
    let mut used = 0usize;
    let mut candidates = RepairCandidateBuffer::new();
    while used < budget && !unsat.is_empty() {
        if unsat.len() == 1 {
            match select_single_unsat_candidate(
                unsat[0] as usize,
                co,
                cl,
                vars,
                counts,
                all_off,
                p_bound,
                all_data,
                all_three_clauses,
            ) {
                SingleUnsatRepairChoice::Flip(v) => {
                    flip_repair_var(v, vars, counts, unsat, all_off, p_bound, all_data);
                    used += 1;
                    continue;
                }
                SingleUnsatRepairChoice::Stop => break,
                SingleUnsatRepairChoice::UseGeneral => {}
            }
        }

        candidates.clear();
        for &cid in unsat.iter() {
            let c = cid as usize;
            let (s, e) = mid_repair_clause_bounds(c, co, all_three_clauses);
            for &lit in &cl[s..e] {
                let v = lit_var_index(lit);
                candidates.push_unique(v);
            }
        }

        let mut best_v = usize::MAX;
        let mut best_net = 0i32;
        let mut best_make = 0usize;
        let mut best_break = usize::MAX;
        for &v in candidates.stack_slice() {
            consider_greedy_repair_candidate(
                v,
                vars,
                counts,
                all_off,
                p_bound,
                all_data,
                &mut best_v,
                &mut best_net,
                &mut best_make,
                &mut best_break,
            );
        }
        if !candidates.overflow_slice().is_empty() {
            for &v in candidates.overflow_slice() {
                consider_greedy_repair_candidate(
                    v,
                    vars,
                    counts,
                    all_off,
                    p_bound,
                    all_data,
                    &mut best_v,
                    &mut best_net,
                    &mut best_make,
                    &mut best_break,
                );
            }
        }

        if best_v == usize::MAX || best_net <= 0 {
            break;
        }

        flip_repair_var(best_v, vars, counts, unsat, all_off, p_bound, all_data);
        used += 1;
    }
    used
}

fn select_single_unsat_candidate(
    c: usize,
    co: &[u32],
    cl: &[i32],
    vars: &[bool],
    counts: &RepairCounts<'_>,
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    all_three_clauses: bool,
) -> SingleUnsatRepairChoice {
    let (s, e) = mid_repair_clause_bounds(c, co, all_three_clauses);
    let mut seen = [usize::MAX; REPAIR_SINGLE_CLAUSE_STACK_CAPACITY];
    let mut seen_len = 0usize;
    let mut best_v = usize::MAX;
    let mut best_net = 0i32;
    let mut best_make = 0usize;
    let mut best_break = usize::MAX;

    for &lit in &cl[s..e] {
        let v = lit_var_index(lit);
        if repair_seen_contains(&seen, seen_len, v) {
            continue;
        }
        if seen_len == seen.len() {
            return SingleUnsatRepairChoice::UseGeneral;
        }
        seen[seen_len] = v;
        seen_len += 1;

        let (make, brk) = flip_make_break(v, vars, counts, all_off, p_bound, all_data);
        let net = make as i32 - brk as i32;
        if net > best_net || (net == best_net && net > 0 && (make > best_make || brk < best_break))
        {
            best_net = net;
            best_make = make;
            best_break = brk;
            best_v = v;
        }
    }

    if best_v == usize::MAX || best_net <= 0 {
        SingleUnsatRepairChoice::Stop
    } else {
        SingleUnsatRepairChoice::Flip(best_v)
    }
}

fn flip_make_break(
    v: usize,
    vars: &[bool],
    counts: &RepairCounts<'_>,
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
) -> (usize, usize) {
    let was_true = vars[v];
    let (inc_s, inc_e, dec_s, dec_e) = if was_true {
        (
            p_bound[v] as usize,
            all_off[v + 1] as usize,
            all_off[v] as usize,
            p_bound[v] as usize,
        )
    } else {
        (
            all_off[v] as usize,
            p_bound[v] as usize,
            p_bound[v] as usize,
            all_off[v + 1] as usize,
        )
    };

    let make = counts.count_occurrences_eq(all_data, inc_s, inc_e, 0);
    let brk = counts.count_occurrences_eq(all_data, dec_s, dec_e, 1);
    (make, brk)
}

fn flip_repair_var(
    v: usize,
    vars: &mut [bool],
    counts: &mut RepairCounts<'_>,
    unsat: &mut Vec<u32>,
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
) {
    let was_true = vars[v];
    let (inc_s, inc_e, dec_s, dec_e) = if was_true {
        (
            p_bound[v] as usize,
            all_off[v + 1] as usize,
            all_off[v] as usize,
            p_bound[v] as usize,
        )
    } else {
        (
            all_off[v] as usize,
            p_bound[v] as usize,
            p_bound[v] as usize,
            all_off[v + 1] as usize,
        )
    };

    for k in inc_s..inc_e {
        let c = all_data[k] as usize;
        let old = counts.increment(c);
        if old == 0 {
            remove_repair_unsat(unsat, c);
        }
    }
    for k in dec_s..dec_e {
        let c = all_data[k] as usize;
        let new_count = counts.decrement(c);
        if new_count == 0 {
            push_new_repair_unsat(unsat, c);
        }
    }
    vars[v] = !was_true;
}

fn remove_repair_unsat(unsat: &mut Vec<u32>, c: usize) {
    let len = unsat.len();
    if len == 0 {
        return;
    }
    if len == 1 {
        if unsat[0] as usize == c {
            unsat.clear();
        }
        return;
    }
    if unsat[len - 1] as usize == c {
        unsat.pop();
        return;
    }
    if unsat[0] as usize == c {
        unsat.swap_remove(0);
        return;
    }
    if len == 2 {
        return;
    }
    if len == 3 {
        if unsat[1] as usize == c {
            unsat.swap_remove(1);
        }
        return;
    }
    if len == 4 {
        if unsat[1] as usize == c {
            unsat.swap_remove(1);
        } else if unsat[2] as usize == c {
            unsat.swap_remove(2);
        }
        return;
    }
    if let Some(pos) = unsat[1..len - 1].iter().position(|&cid| cid as usize == c) {
        unsat.swap_remove(pos + 1);
    }
}

#[inline(always)]
fn push_new_repair_unsat(unsat: &mut Vec<u32>, c: usize) {
    debug_assert!(!unsat.iter().any(|&cid| cid as usize == c));
    unsat.push(c as u32);
}

fn add_repair_unsat(unsat: &mut Vec<u32>, c: usize) {
    if unsat.is_empty() {
        unsat.push(c as u32);
        return;
    }
    if !unsat.iter().any(|&cid| cid as usize == c) {
        unsat.push(c as u32);
    }
}

#[inline(always)]
fn mid_clause_good_count(s: usize, e: usize, cl: &[i32], vars: &[bool]) -> u8 {
    match e - s {
        1 => mid_lit_good(cl[s], vars),
        2 => mid_lit_good(cl[s], vars) + mid_lit_good(cl[s + 1], vars),
        3 => {
            mid_lit_good(cl[s], vars)
                + mid_lit_good(cl[s + 1], vars)
                + mid_lit_good(cl[s + 2], vars)
        }
        _ => {
            let mut good = 0u8;
            for &lit in &cl[s..e] {
                let v = lit_var_index(lit);
                if (lit > 0 && vars[v]) || (lit < 0 && !vars[v]) {
                    good += 1;
                }
            }
            good
        }
    }
}

#[inline(always)]
fn mid_clause3_good_count(s: usize, cl: &[i32], vars: &[bool]) -> u8 {
    mid_lit_good(cl[s], vars) + mid_lit_good(cl[s + 1], vars) + mid_lit_good(cl[s + 2], vars)
}

fn clear_exact_mid_unsat_positions(nc: usize, unsat_list: &[u32], unsat_pos: &mut [u32]) {
    debug_assert!(unsat_pos.len() >= nc);
    debug_assert!(unsat_list
        .iter()
        .enumerate()
        .all(|(idx, &cid)| { unsat_pos.get(cid as usize).copied() == Some(idx as u32) }));
    for &cid in unsat_list {
        unsat_pos[cid as usize] = u32::MAX;
    }
    debug_assert!(unsat_pos[..nc].iter().all(|&pos| pos == u32::MAX));
}

fn rebuild_mid_state_impl<const CLEAR_STATE: bool>(
    nc: usize,
    co: &[u32],
    cl: &[i32],
    vars: &[bool],
    num_good: &mut [u8],
    unsat_list: &mut Vec<u32>,
    unsat_pos: &mut [u32],
    all_three_clauses: bool,
) {
    let byte_count = (nc + 3) >> 2;
    debug_assert!(num_good.len() >= byte_count);
    debug_assert!(unsat_pos.len() >= nc);
    if CLEAR_STATE {
        clear_exact_mid_unsat_positions(nc, unsat_list, unsat_pos);
        unsat_list.clear();
    } else {
        debug_assert!(num_good[..byte_count].iter().all(|&count| count == 0));
        debug_assert!(unsat_list.is_empty());
        debug_assert!(unsat_pos[..nc].iter().all(|&pos| pos == u32::MAX));
    }

    debug_assert!(!all_three_clauses || mid_clause_offsets_are_three(nc, co, cl.len()));
    if all_three_clauses {
        debug_assert!(co.len() > nc);
        debug_assert_eq!(co[0], 0);
        debug_assert_eq!(co[nc] as usize, cl.len());
        debug_assert!((0..=nc).all(|i| co[i] as usize == i * 3));

        let full_bytes = nc >> 2;
        for byte_idx in 0..full_bytes {
            let base = byte_idx << 2;
            let s = base * 3;
            let g0 = mid_clause3_good_count(s, cl, vars);
            let g1 = mid_clause3_good_count(s + 3, cl, vars);
            let g2 = mid_clause3_good_count(s + 6, cl, vars);
            let g3 = mid_clause3_good_count(s + 9, cl, vars);
            num_good[byte_idx] = g0 | (g1 << 2) | (g2 << 4) | (g3 << 6);

            if g0 == 0 {
                unsat_pos[base] = unsat_list.len() as u32;
                unsat_list.push(base as u32);
            }
            if g1 == 0 {
                unsat_pos[base + 1] = unsat_list.len() as u32;
                unsat_list.push((base + 1) as u32);
            }
            if g2 == 0 {
                unsat_pos[base + 2] = unsat_list.len() as u32;
                unsat_list.push((base + 2) as u32);
            }
            if g3 == 0 {
                unsat_pos[base + 3] = unsat_list.len() as u32;
                unsat_list.push((base + 3) as u32);
            }
        }

        let mut i = full_bytes << 2;
        if i < nc {
            let byte_idx = i >> 2;
            let mut packed = 0u8;
            while i < nc {
                let good = mid_clause3_good_count(i * 3, cl, vars);
                packed |= good << ((i & 3) << 1);
                if good == 0 {
                    unsat_pos[i] = unsat_list.len() as u32;
                    unsat_list.push(i as u32);
                }
                i += 1;
            }
            num_good[byte_idx] = packed;
        }
        return;
    }

    let full_bytes = nc >> 2;
    for byte_idx in 0..full_bytes {
        let base = byte_idx << 2;
        let g0 = mid_clause_good_count(co[base] as usize, co[base + 1] as usize, cl, vars);
        let g1 = mid_clause_good_count(co[base + 1] as usize, co[base + 2] as usize, cl, vars);
        let g2 = mid_clause_good_count(co[base + 2] as usize, co[base + 3] as usize, cl, vars);
        let g3 = mid_clause_good_count(co[base + 3] as usize, co[base + 4] as usize, cl, vars);
        num_good[byte_idx] = g0.min(3) | (g1.min(3) << 2) | (g2.min(3) << 4) | (g3.min(3) << 6);

        if g0 == 0 {
            unsat_pos[base] = unsat_list.len() as u32;
            unsat_list.push(base as u32);
        }
        if g1 == 0 {
            unsat_pos[base + 1] = unsat_list.len() as u32;
            unsat_list.push((base + 1) as u32);
        }
        if g2 == 0 {
            unsat_pos[base + 2] = unsat_list.len() as u32;
            unsat_list.push((base + 2) as u32);
        }
        if g3 == 0 {
            unsat_pos[base + 3] = unsat_list.len() as u32;
            unsat_list.push((base + 3) as u32);
        }
    }

    let mut i = full_bytes << 2;
    if i < nc {
        let byte_idx = i >> 2;
        let mut packed = 0u8;
        while i < nc {
            let good = mid_clause_good_count(co[i] as usize, co[i + 1] as usize, cl, vars);
            packed |= good.min(3) << ((i & 3) << 1);
            if good == 0 {
                unsat_pos[i] = unsat_list.len() as u32;
                unsat_list.push(i as u32);
            }
            i += 1;
        }
        num_good[byte_idx] = packed;
    }
}

fn rebuild_mid_state(
    nc: usize,
    co: &[u32],
    cl: &[i32],
    vars: &[bool],
    num_good: &mut [u8],
    unsat_list: &mut Vec<u32>,
    unsat_pos: &mut [u32],
    all_three_clauses: bool,
) {
    rebuild_mid_state_impl::<true>(
        nc,
        co,
        cl,
        vars,
        num_good,
        unsat_list,
        unsat_pos,
        all_three_clauses,
    );
}

fn rebuild_mid_state_fresh(
    nc: usize,
    co: &[u32],
    cl: &[i32],
    vars: &[bool],
    num_good: &mut [u8],
    unsat_list: &mut Vec<u32>,
    unsat_pos: &mut [u32],
    all_three_clauses: bool,
) {
    rebuild_mid_state_impl::<false>(
        nc,
        co,
        cl,
        vars,
        num_good,
        unsat_list,
        unsat_pos,
        all_three_clauses,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn mid_lit_var_index_matches_abs_index_for_valid_literals() {
        for lit in [-128_i32, -7, -1, 1, 7, 128] {
            assert_eq!(lit_var_index(lit), (lit.abs() - 1) as usize);
        }
    }

    #[test]
    fn mid_var_appearances_only_builds_when_needed() {
        let p_cnt = vec![1_u32, 4, 0];
        let n_cnt = vec![2_u32, 1, 300];

        assert!(build_mid_appearances_if_needed(false, 100_000, &p_cnt, &n_cnt).is_empty());
        assert_eq!(
            build_mid_appearances_if_needed(true, 3, &p_cnt, &n_cnt),
            vec![3_u8, 5, 255]
        );
    }

    #[test]
    fn mid_var_age_only_builds_when_needed() {
        assert!(build_mid_var_age_if_needed(false, 100_000).is_empty());
        assert_eq!(build_mid_var_age_if_needed(true, 4), vec![0_u8; 4]);
    }

    #[test]
    fn bump_mid_clause_var_ages3_matches_generic_loop() {
        let cl = [99, 1, -2, 2, 100];
        let mut generic_age = [5_u8, u8::MAX - 1, 17];
        let mut fast_age = generic_age;

        unsafe {
            bump_mid_clause_var_ages(1, 4, &cl, &mut generic_age);
            bump_mid_clause_var_ages3(1, &cl, &mut fast_age);
        }

        assert_eq!(fast_age, generic_age);
        assert_eq!(fast_age, [6, u8::MAX, 17]);
    }

    #[test]
    fn mid_probsat_break_weights_stop_at_occurrence_bound() {
        let p_cnt = vec![0_u32, 3, 9, 2];
        let n_cnt = vec![1_u32, 5, 4, 8];
        let cb: f64 = 2.85;

        let weights = build_mid_probsat_break_weights(4.2, &p_cnt, &n_cnt);

        assert_eq!(weights.len(), 10);
        for (i, &weight) in weights.iter().enumerate() {
            assert_eq!(weight, cb.powf(-(i as f64)));
        }
    }

    #[test]
    fn mid_pair_occurrence_bound_matches_zip_max_reference() {
        assert_eq!(
            max_pair_occurrence_bound(&[0_u32, 3, 9, 2], &[1_u32, 5, 4, 8]),
            9
        );
        assert_eq!(
            max_pair_occurrence_bound(&[0_u32, 300], &[1_u32, 5, 4]),
            300
        );
        assert_eq!(max_pair_occurrence_bound(&[0_u32], &[1_u32, 500]), 1);
        assert_eq!(max_pair_occurrence_bound(&[], &[9_u32]), 0);
    }

    #[test]
    fn mid_weighted3_selector_matches_array_reference() {
        fn reference(threshold: f64, count: usize, weights: [f64; 3], vars: [usize; 3]) -> usize {
            let mut accum = 0.0;
            let mut selected = vars[0];
            for idx in 0..count {
                accum += weights[idx];
                if accum >= threshold {
                    selected = vars[idx];
                    break;
                }
            }
            selected
        }

        let weights = [0.10, 0.35, 0.55];
        let vars = [7, 11, 13];
        for &(count, threshold) in &[
            (1, 0.0),
            (1, 0.11),
            (2, 0.10),
            (2, 0.45),
            (3, 0.46),
            (3, 1.01),
        ] {
            assert_eq!(
                select_weighted3_f64_accum_ge(
                    threshold, count, weights[0], vars[0], weights[1], vars[1], weights[2],
                    vars[2],
                ),
                reference(threshold, count, weights, vars)
            );
        }
    }

    fn mid_probsat_occurrence_fixture(sads: [usize; 3]) -> (Vec<u8>, Vec<u32>, Vec<u32>, Vec<u32>) {
        let mut all_off = vec![0_u32; 4];
        let mut p_bound = vec![0_u32; 3];
        let mut all_data = Vec::new();

        all_off[0] = 0;
        p_bound[0] = 0;
        for _ in 0..sads[0] {
            all_data.push(all_data.len() as u32);
        }
        all_off[1] = all_data.len() as u32;

        for _ in 0..sads[1] {
            all_data.push(all_data.len() as u32);
        }
        p_bound[1] = all_data.len() as u32;
        all_off[2] = all_data.len() as u32;

        p_bound[2] = all_data.len() as u32;
        for _ in 0..sads[2] {
            all_data.push(all_data.len() as u32);
        }
        all_off[3] = all_data.len() as u32;

        let mut num_good = vec![0_u8; (all_data.len() + 3) / 4];
        for c in 0..all_data.len() {
            num_good[c >> 2] |= 1_u8 << ((c & 3) << 1);
        }

        (num_good, all_off, p_bound, all_data)
    }

    #[test]
    fn mid_probsat_var3_matches_generic_and_rng_consumption() {
        let cl = vec![1, -2, 3];
        let probsat_break = vec![1.0_f64, 0.5, 0.25, 0.125, 0.0625];
        let probsat_break_limit = probsat_break.len() - 1;
        let cases = [
            [1_usize, 2, 3],
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 2],
            [2, 0, 0],
            [5, 1, 3],
        ];

        for sads in cases {
            let (num_good, all_off, p_bound, all_data) = mid_probsat_occurrence_fixture(sads);
            for rand_val in 0usize..8 {
                for seed in 0_u64..16 {
                    let mut generic_rng = SmallRng::seed_from_u64(seed);
                    let mut fast_rng = SmallRng::seed_from_u64(seed);
                    let generic = unsafe {
                        choose_mid_probsat_var_generic(
                            0,
                            3,
                            &cl,
                            &num_good,
                            &all_off,
                            &p_bound,
                            &all_data,
                            &probsat_break,
                            probsat_break_limit,
                            rand_val,
                            &mut generic_rng,
                        )
                    };
                    let fast = unsafe {
                        choose_mid_probsat_var3(
                            0,
                            &cl,
                            &num_good,
                            &all_off,
                            &p_bound,
                            &all_data,
                            &probsat_break,
                            probsat_break_limit,
                            rand_val,
                            &mut fast_rng,
                        )
                    };

                    assert_eq!(
                        fast, generic,
                        "sads={sads:?} rand_val={rand_val} seed={seed}"
                    );
                    assert_eq!(
                        fast_rng.gen::<u64>(),
                        generic_rng.gen::<u64>(),
                        "rng mismatch for sads={sads:?} rand_val={rand_val} seed={seed}"
                    );
                }
            }
        }
    }

    #[test]
    fn mid_non_probsat_var3_matches_generic_reference() {
        let cl = vec![99, 1, -2, 3, 100];
        let appearances = vec![100_u8, 110, 120];
        let var_age = vec![1_u8, 8, 12];
        let cases = [
            [0_usize, 1, 2],
            [1, 0, 2],
            [1, 2, 0],
            [1, 1, 2],
            [2, 3, 4],
            [5, 1, 3],
            [3, 2, 1],
        ];
        let cutoffs = [0_u64, u64::MAX / 2, u64::MAX];

        for sads in cases {
            let (num_good, all_off, p_bound, all_data) = mid_probsat_occurrence_fixture(sads);
            for current_prob_cutoff in cutoffs {
                for rand_val in 0usize..16 {
                    let generic = unsafe {
                        choose_mid_non_probsat_var_generic(
                            1,
                            4,
                            &cl,
                            &num_good,
                            &all_off,
                            &p_bound,
                            &all_data,
                            &appearances,
                            &var_age,
                            current_prob_cutoff,
                            rand_val,
                        )
                    };
                    let direct = unsafe {
                        choose_mid_non_probsat_var3(
                            1,
                            &cl,
                            &num_good,
                            &all_off,
                            &p_bound,
                            &all_data,
                            &appearances,
                            &var_age,
                            current_prob_cutoff,
                            rand_val,
                        )
                    };

                    assert_eq!(
                        direct, generic,
                        "sads={sads:?} cutoff={current_prob_cutoff} rand_val={rand_val}"
                    );
                }
            }
        }
    }

    #[test]
    fn initial_assignment_mid_matches_legacy_rng_path() {
        let p_cnt = vec![10_u32, 0, 4, 1, 260];
        let n_cnt = vec![0_u32, 3, 2, 1, 10];
        let density = 4.20;
        let hp = Hyperparameters::default();
        let init_noise = Some(0.017);
        let mut direct_rng = SmallRng::seed_from_u64(17);
        let mut legacy_rng = SmallRng::seed_from_u64(17);

        let vars = initial_assignment_mid(
            5,
            density,
            &p_cnt,
            &n_cnt,
            &mut direct_rng,
            &hp,
            123,
            init_noise,
        );

        let mut legacy_vars = vec![false; 5];
        let nad = hp.target_nad.unwrap_or(1.0).max(0.01);
        let random_threshold = init_noise.unwrap();
        let steep = 0.35 / (1.0 + (density - 4.18_f64).max(0.0) * 12.0);
        for v in 0..5 {
            let np = p_cnt[v] as f64;
            let nn = n_cnt[v] as f64;
            if nn == 0.0 && np > 0.0 {
                legacy_vars[v] = true;
                continue;
            }
            if np == 0.0 {
                continue;
            }
            let vad = np / nn;
            let bias_prob = (np + 0.25) / (np + nn + 1.2);
            let s = 1.0 / (1.0 + (-(vad - nad) / steep).exp());
            let prob = (random_threshold * (1.0 - s) + bias_prob * s)
                .max(0.0)
                .min(1.0);
            legacy_vars[v] = legacy_rng.gen_bool(prob);
        }

        assert_eq!(vars, legacy_vars);
        assert_eq!(direct_rng.gen::<u64>(), legacy_rng.gen::<u64>());
    }

    #[test]
    fn mid_best_restart_defaults_only_for_4200_route() {
        assert_eq!(
            mid_best_restart_limit_for(&Hyperparameters::default(), true),
            8
        );
        assert_eq!(
            mid_best_restart_limit_for(
                &Hyperparameters {
                    target_mid_best_restart: Some(true),
                    ..Hyperparameters::default()
                },
                false,
            ),
            0
        );
    }

    #[test]
    fn mid_best_restart_can_be_disabled_or_capped() {
        assert_eq!(
            mid_best_restart_limit_for(
                &Hyperparameters {
                    target_mid_best_restart: Some(false),
                    target_mid_best_restart_limit: Some(12),
                    ..Hyperparameters::default()
                },
                true,
            ),
            0
        );
        assert_eq!(
            mid_best_restart_limit_for(
                &Hyperparameters {
                    target_mid_best_restart_limit: Some(0),
                    ..Hyperparameters::default()
                },
                true,
            ),
            0
        );
        assert_eq!(
            mid_best_restart_limit_for(
                &Hyperparameters {
                    target_mid_best_restart_limit: Some(200),
                    ..Hyperparameters::default()
                },
                true,
            ),
            64
        );
    }

    #[test]
    fn mid_best_snapshot_restore_obeys_restart_hp_boundary() {
        assert!(should_restore_mid_best_snapshot(true, 4, 8));
        assert!(!should_restore_mid_best_snapshot(false, 4, 8));
        assert!(!should_restore_mid_best_snapshot(true, 8, 8));
        assert!(!should_restore_mid_best_snapshot(true, 0, 0));
    }

    #[test]
    fn mid_best_snapshot_storage_is_lazy_when_restart_disabled() {
        let vars = vec![true, false, true];

        assert!(initial_mid_best_vars(false, 4, 8, &vars).is_empty());
        assert!(initial_mid_best_vars(true, 32, 8, &vars).is_empty());
        assert!(initial_mid_best_vars(true, 0, 8, &vars).is_empty());

        let mut snapshot = initial_mid_best_vars(true, 4, 8, &vars);
        assert_eq!(snapshot, vars);
        snapshot[0] = false;
        assert_eq!(vars[0], true);
    }

    #[test]
    fn mid_last_mile_repair_is_track_scoped_and_disableable() {
        let hp = Hyperparameters::default();
        assert_eq!(mid_last_mile_budget_for(&hp), 512);
        assert!(mid_last_mile_repair_enabled(&hp, true, 12));
        assert!(!mid_last_mile_repair_enabled(&hp, false, 12));
        assert!(!mid_last_mile_repair_enabled(&hp, true, 0));
        assert!(!mid_last_mile_repair_enabled(
            &Hyperparameters {
                target_mid_last_mile_repair: Some(false),
                ..Hyperparameters::default()
            },
            true,
            12,
        ));
        assert!(!mid_last_mile_repair_enabled(
            &Hyperparameters {
                target_mid_last_mile_max_unsat: Some(4),
                ..Hyperparameters::default()
            },
            true,
            5,
        ));
        assert!(!mid_last_mile_repair_enabled(
            &Hyperparameters {
                target_mid_last_mile_budget: Some(0),
                ..Hyperparameters::default()
            },
            true,
            4,
        ));
        assert_eq!(
            mid_last_mile_budget_for(&Hyperparameters {
                target_mid_last_mile_budget: Some(50_000),
                ..Hyperparameters::default()
            }),
            20_000,
        );
    }

    #[test]
    fn repair_end_unsat_uses_saturating_satisfied_count() {
        assert_eq!(repair_end_unsat(100, 97), 3);
        assert_eq!(repair_end_unsat(100, 100), 0);
        assert_eq!(repair_end_unsat(100, 101), 0);
    }

    #[test]
    fn last_mile_unsat_capacity_stays_tail_sized() {
        assert_eq!(last_mile_unsat_capacity(0), 0);
        assert_eq!(last_mile_unsat_capacity(24), 24);
        assert_eq!(last_mile_unsat_capacity(1_000_000), 64);
    }

    #[test]
    fn last_mile_clause_var_capacity_stays_tail_sized() {
        assert_eq!(last_mile_clause_var_capacity(0), 0);
        assert_eq!(last_mile_clause_var_capacity(24), 192);
        assert_eq!(last_mile_clause_var_capacity(1_000_000), 256);
    }

    #[test]
    fn remove_mid_unsat_exact_handles_moved_and_last_entries() {
        let mut unsat = vec![2_u32, 5, 7];
        let mut unsat_pos = vec![u32::MAX; 8];
        unsat_pos[2] = 0;
        unsat_pos[5] = 1;
        unsat_pos[7] = 2;

        unsafe {
            remove_mid_unsat_exact(&mut unsat, &mut unsat_pos, 5);
        }

        assert_eq!(unsat, vec![2_u32, 7]);
        assert_eq!(unsat_pos[2], 0);
        assert_eq!(unsat_pos[5], u32::MAX);
        assert_eq!(unsat_pos[7], 1);

        unsafe {
            remove_mid_unsat_exact(&mut unsat, &mut unsat_pos, 7);
        }

        assert_eq!(unsat, vec![2_u32]);
        assert_eq!(unsat_pos[2], 0);
        assert_eq!(unsat_pos[5], u32::MAX);
        assert_eq!(unsat_pos[7], u32::MAX);
    }

    #[test]
    fn repair_var_buffer_preserves_order_and_conflict_polarity() {
        let mut touched = RepairVarBuffer::new();

        touched.push_or_mark_polarity(5, 1);
        touched.push_or_mark_polarity(2, 2);
        touched.push_or_mark_polarity(5, 2);
        touched.push_or_mark_polarity(2, 2);

        let values = touched
            .stack_slice()
            .iter()
            .chain(touched.overflow_slice().iter())
            .copied()
            .collect::<Vec<_>>();

        assert_eq!(values, vec![(5, 3), (2, 2)]);
        assert!(touched.overflow_slice().is_empty());
    }

    #[test]
    fn repair_var_buffer_overflow_keeps_insertion_order() {
        let mut touched = RepairVarBuffer::new();

        for v in 0..(REPAIR_VAR_STACK_CAPACITY + 2) {
            touched.push_or_mark_polarity(v, 1);
        }
        touched.push_or_mark_polarity(REPAIR_VAR_STACK_CAPACITY + 1, 2);

        let values = touched
            .stack_slice()
            .iter()
            .chain(touched.overflow_slice().iter())
            .copied()
            .collect::<Vec<_>>();

        assert_eq!(values.len(), REPAIR_VAR_STACK_CAPACITY + 2);
        assert_eq!(values[0], (0, 1));
        assert_eq!(
            values[REPAIR_VAR_STACK_CAPACITY - 1],
            (REPAIR_VAR_STACK_CAPACITY - 1, 1),
        );
        assert_eq!(
            values[REPAIR_VAR_STACK_CAPACITY],
            (REPAIR_VAR_STACK_CAPACITY, 1),
        );
        assert_eq!(
            values[REPAIR_VAR_STACK_CAPACITY + 1],
            (REPAIR_VAR_STACK_CAPACITY + 1, 3),
        );
    }

    #[test]
    fn repair_candidate_buffer_preserves_order_and_uniqueness() {
        let mut candidates = RepairCandidateBuffer::new();

        candidates.push_unique(5);
        candidates.push_unique(2);
        candidates.push_unique(5);
        candidates.push_unique(9);

        let values = candidates
            .stack_slice()
            .iter()
            .chain(candidates.overflow_slice().iter())
            .copied()
            .collect::<Vec<_>>();

        assert_eq!(values, vec![5, 2, 9]);
        assert!(candidates.overflow_slice().is_empty());
    }

    #[test]
    fn repair_buffers_last_entry_fast_paths_keep_semantics() {
        let mut touched = RepairVarBuffer::new();
        touched.push_or_mark_polarity(4, 1);
        touched.push_or_mark_polarity(4, 2);

        assert_eq!(touched.stack_slice(), &[(4, 3)]);
        assert!(touched.overflow_slice().is_empty());

        let mut candidates = RepairCandidateBuffer::new();
        candidates.push_unique(8);
        candidates.push_unique(8);
        candidates.push_unique(2);

        assert_eq!(candidates.stack_slice(), &[8, 2]);
        assert!(candidates.overflow_slice().is_empty());
    }

    #[test]
    fn repair_buffers_overflow_last_entry_fast_paths_keep_semantics() {
        let mut touched = RepairVarBuffer::new();
        for v in 0..(REPAIR_VAR_STACK_CAPACITY + 3) {
            touched.push_or_mark_polarity(v, 1);
        }
        touched.push_or_mark_polarity(REPAIR_VAR_STACK_CAPACITY + 2, 2);
        touched.push_or_mark_polarity(REPAIR_VAR_STACK_CAPACITY, 2);

        let values = touched
            .stack_slice()
            .iter()
            .chain(touched.overflow_slice().iter())
            .copied()
            .collect::<Vec<_>>();

        assert_eq!(values.len(), REPAIR_VAR_STACK_CAPACITY + 3);
        assert_eq!(
            values[REPAIR_VAR_STACK_CAPACITY],
            (REPAIR_VAR_STACK_CAPACITY, 3)
        );
        assert_eq!(
            values[REPAIR_VAR_STACK_CAPACITY + 2],
            (REPAIR_VAR_STACK_CAPACITY + 2, 3)
        );

        let mut candidates = RepairCandidateBuffer::new();
        for v in 0..(REPAIR_VAR_STACK_CAPACITY + 3) {
            candidates.push_unique(v);
        }
        candidates.push_unique(REPAIR_VAR_STACK_CAPACITY + 2);
        candidates.push_unique(REPAIR_VAR_STACK_CAPACITY);

        let candidate_values = candidates
            .stack_slice()
            .iter()
            .chain(candidates.overflow_slice().iter())
            .copied()
            .collect::<Vec<_>>();

        assert_eq!(candidate_values.len(), REPAIR_VAR_STACK_CAPACITY + 3);
        assert_eq!(
            candidate_values[REPAIR_VAR_STACK_CAPACITY],
            REPAIR_VAR_STACK_CAPACITY
        );
        assert_eq!(
            candidate_values[REPAIR_VAR_STACK_CAPACITY + 2],
            REPAIR_VAR_STACK_CAPACITY + 2
        );
    }

    #[test]
    fn repair_candidate_buffer_overflow_and_clear_keep_order() {
        let mut candidates = RepairCandidateBuffer::new();

        for v in 0..(REPAIR_VAR_STACK_CAPACITY + 2) {
            candidates.push_unique(v);
        }
        candidates.push_unique(REPAIR_VAR_STACK_CAPACITY + 1);

        let values = candidates
            .stack_slice()
            .iter()
            .chain(candidates.overflow_slice().iter())
            .copied()
            .collect::<Vec<_>>();

        assert_eq!(values.len(), REPAIR_VAR_STACK_CAPACITY + 2);
        assert_eq!(values[0], 0);
        assert_eq!(
            values[REPAIR_VAR_STACK_CAPACITY - 1],
            REPAIR_VAR_STACK_CAPACITY - 1
        );
        assert_eq!(values[REPAIR_VAR_STACK_CAPACITY], REPAIR_VAR_STACK_CAPACITY);
        assert_eq!(
            values[REPAIR_VAR_STACK_CAPACITY + 1],
            REPAIR_VAR_STACK_CAPACITY + 1,
        );

        candidates.clear();
        candidates.push_unique(7);

        assert_eq!(candidates.stack_slice(), &[7]);
        assert!(candidates.overflow_slice().is_empty());
    }

    #[test]
    fn mid_last_mile_zero_budget_is_noop() {
        let co = [0_u32, 1, 3];
        let cl = [1, -1, 2];
        let all_off = [0_u32, 2, 3];
        let p_bound = [1_u32, 3];
        let all_data = [0_u32, 1, 1];
        let mut vars = [false, true];

        let (sat, flips) = refine_mid_last_mile_solution(
            2, 2, &co, &cl, &all_off, &p_bound, &all_data, &mut vars, 0,
        );

        assert_eq!(sat, 1);
        assert_eq!(flips, 0);
        assert_eq!(vars, [false, true]);
    }

    #[test]
    fn mid_last_mile_no_positive_move_leaves_assignment_unchanged() {
        let co = [0_u32, 1, 2];
        let cl = [1, -1];
        let all_off = [0_u32, 2];
        let p_bound = [1_u32];
        let all_data = [0_u32, 1];
        let mut vars = [false];

        let (sat, flips) = refine_mid_last_mile_solution(
            1, 2, &co, &cl, &all_off, &p_bound, &all_data, &mut vars, 8,
        );

        assert_eq!(sat, 1);
        assert_eq!(flips, 0);
        assert_eq!(vars, [false]);
    }

    #[test]
    fn mid_last_mile_purity_repair_keeps_satisfied_clauses() {
        let co = [0_u32, 1, 3];
        let cl = [1, -1, 2];
        let all_off = [0_u32, 2, 3];
        let p_bound = [1_u32, 3];
        let all_data = [0_u32, 1, 1];
        let mut vars = [false, true];

        let (sat, flips) = refine_mid_last_mile_solution(
            2, 2, &co, &cl, &all_off, &p_bound, &all_data, &mut vars, 8,
        );

        assert_eq!(sat, 2);
        assert_eq!(flips, 1);
        assert_eq!(vars, [true, true]);
    }

    #[test]
    fn mid_last_mile_fast_state_matches_scan_rebuild() {
        let co = [0_u32, 1, 3];
        let cl = [1, -1, 2];
        let all_off = [0_u32, 2, 3];
        let p_bound = [1_u32, 3];
        let all_data = [0_u32, 1, 1];
        let start_vars = [false, true];
        let mut packed_num_good = vec![0u8; (2 + 3) >> 2];
        let mut mid_unsat = Vec::new();
        let mut mid_unsat_pos = vec![u32::MAX; 2];
        rebuild_mid_state(
            2,
            &co,
            &cl,
            &start_vars,
            &mut packed_num_good,
            &mut mid_unsat,
            &mut mid_unsat_pos,
            false,
        );
        assert_eq!(mid_unsat, vec![0]);

        let mut fast_vars = start_vars;
        let mut scan_vars = start_vars;
        let fast = refine_mid_last_mile_solution_from_mid_state(
            2,
            2,
            &co,
            &cl,
            &all_off,
            &p_bound,
            &all_data,
            &mut fast_vars,
            8,
            &packed_num_good,
            &mid_unsat,
            mid_unsat.len(),
            false,
        );
        let scan = refine_mid_last_mile_solution(
            2,
            2,
            &co,
            &cl,
            &all_off,
            &p_bound,
            &all_data,
            &mut scan_vars,
            8,
        );

        assert_eq!(fast, scan);
        assert_eq!(fast_vars, scan_vars);
        assert_eq!(fast, (2, 1));
    }

    #[test]
    fn mid_last_mile_from_mid_state_returns_before_overlay_when_already_solved() {
        let co = [0_u32, 1, 2];
        let cl = [1, -2];
        let all_off = [0_u32, 1, 2];
        let p_bound = [1_u32, 2];
        let all_data = [0_u32, 1];
        let mut vars = [true, false];
        let packed_num_good = [1_u8 | (1 << 2)];

        let result = refine_mid_last_mile_solution_from_mid_state(
            2,
            2,
            &co,
            &cl,
            &all_off,
            &p_bound,
            &all_data,
            &mut vars,
            8,
            &packed_num_good,
            &[],
            4,
            false,
        );

        assert_eq!(result, (2, 0));
        assert_eq!(vars, [true, false]);
    }

    #[test]
    fn rebuild_repair_unsat_from_mid_tail_matches_full_scan() {
        let packed = [
            1_u8 | (0 << 2) | (2 << 4) | (3 << 6),
            0_u8 | (1 << 2) | (2 << 4),
        ];
        let current_unsat = [1_u32, 4];
        let mut from_tail = Vec::new();
        let mut from_scan = Vec::new();

        rebuild_repair_unsat_from_mid_tail(7, &packed, &current_unsat, &mut from_tail);
        rebuild_repair_unsat_from_mid_state(7, &packed, &mut from_scan);

        assert_eq!(from_tail, from_scan);
        assert_eq!(from_tail, vec![1, 4]);
    }

    #[test]
    fn rebuild_mid_state_fresh_matches_clearing_rebuild_on_clean_sidecars() {
        let co = [0_u32, 2, 4, 5];
        let cl = [1, -2, -1, 2, 3];
        let vars = [true, true, false];

        let mut fresh_num_good = vec![0_u8; (3 + 3) >> 2];
        let mut fresh_unsat = Vec::new();
        let mut fresh_pos = vec![u32::MAX; 3];
        rebuild_mid_state_fresh(
            3,
            &co,
            &cl,
            &vars,
            &mut fresh_num_good,
            &mut fresh_unsat,
            &mut fresh_pos,
            false,
        );

        let mut clearing_num_good = vec![0xff_u8; (3 + 3) >> 2];
        let mut clearing_unsat = vec![0_u32, 1];
        let mut clearing_pos = vec![0_u32, 1, u32::MAX];
        rebuild_mid_state(
            3,
            &co,
            &cl,
            &vars,
            &mut clearing_num_good,
            &mut clearing_unsat,
            &mut clearing_pos,
            false,
        );

        assert_eq!(fresh_num_good, clearing_num_good);
        assert_eq!(fresh_num_good, vec![1 | (1 << 2)]);
        assert_eq!(fresh_unsat, clearing_unsat);
        assert_eq!(fresh_unsat, vec![2]);
        assert_eq!(fresh_pos, clearing_pos);
        assert_eq!(fresh_pos, vec![u32::MAX, u32::MAX, 0]);
    }

    #[test]
    fn clear_exact_mid_unsat_positions_resets_live_entries() {
        let old_unsat = vec![2_u32, 5, 7];
        let mut old_pos = vec![u32::MAX; 9];
        old_pos[2] = 0;
        old_pos[5] = 1;
        old_pos[7] = 2;

        clear_exact_mid_unsat_positions(9, &old_unsat, &mut old_pos);

        assert!(old_pos.iter().all(|&pos| pos == u32::MAX));
    }

    #[test]
    fn rebuild_mid_state_four_clause_packing_matches_legacy_order() {
        fn legacy_rebuild_mid_state(
            nc: usize,
            co: &[u32],
            cl: &[i32],
            vars: &[bool],
        ) -> (Vec<u8>, Vec<u32>, Vec<u32>) {
            let mut num_good = vec![0_u8; (nc + 3) >> 2];
            let mut unsat = Vec::new();
            let mut unsat_pos = vec![u32::MAX; nc];
            for i in 0..nc {
                let mut good = 0u8;
                for &lit in &cl[co[i] as usize..co[i + 1] as usize] {
                    let v = lit_var_index(lit);
                    if (lit > 0 && vars[v]) || (lit < 0 && !vars[v]) {
                        good += 1;
                    }
                }
                num_good[i >> 2] |= good.min(3) << ((i & 3) << 1);
                if good == 0 {
                    unsat_pos[i] = unsat.len() as u32;
                    unsat.push(i as u32);
                }
            }
            (num_good, unsat, unsat_pos)
        }

        let co = [0_u32, 3, 6, 9, 12, 14, 17, 20];
        let cl = [
            1, -2, 3, -1, 2, -3, 1, 2, -3, -1, -2, -3, 1, -2, -1, 2, 3, 1, -2, -3,
        ];
        let vars = [true, true, false];
        let expected = legacy_rebuild_mid_state(7, &co, &cl, &vars);
        let mut actual_good = vec![0xff_u8; (7 + 3) >> 2];
        let mut actual_unsat = vec![1_u32, 4];
        let mut actual_pos = vec![u32::MAX; 7];
        actual_pos[1] = 0;
        actual_pos[4] = 1;

        rebuild_mid_state(
            7,
            &co,
            &cl,
            &vars,
            &mut actual_good,
            &mut actual_unsat,
            &mut actual_pos,
            false,
        );

        assert_eq!(actual_good, expected.0);
        assert_eq!(actual_unsat, expected.1);
        assert_eq!(actual_pos, expected.2);
    }

    #[test]
    fn rebuild_mid_state_all_three_fast_path_matches_generic_reference() {
        let nc = 7;
        let co = [0_u32, 3, 6, 9, 12, 15, 18, 21];
        let cl = [
            1, 2, 3, -1, -2, -3, 1, -2, 4, -1, 2, -4, 3, 4, -2, -3, -4, 2, 1, -3, -4,
        ];
        let vars = [true, false, true, false];

        let mut expected_good = vec![0_u8; (nc + 3) >> 2];
        let mut expected_unsat = Vec::new();
        let mut expected_pos = vec![u32::MAX; nc];
        for c in 0..nc {
            let good = mid_clause_good_count(co[c] as usize, co[c + 1] as usize, &cl, &vars);
            expected_good[c >> 2] |= good.min(3) << ((c & 3) << 1);
            if good == 0 {
                expected_pos[c] = expected_unsat.len() as u32;
                expected_unsat.push(c as u32);
            }
        }

        let mut actual_good = vec![0_u8; (nc + 3) >> 2];
        let mut actual_unsat = Vec::new();
        let mut actual_pos = vec![u32::MAX; nc];
        rebuild_mid_state_fresh(
            nc,
            &co,
            &cl,
            &vars,
            &mut actual_good,
            &mut actual_unsat,
            &mut actual_pos,
            true,
        );

        assert_eq!(actual_good, expected_good);
        assert_eq!(actual_unsat, expected_unsat);
        assert_eq!(actual_pos, expected_pos);
    }

    #[test]
    fn rebuild_mid_state_rejects_average_three_mixed_offsets() {
        let nc = 3;
        let co = [0_u32, 2, 5, 9];
        let cl = [1, -2, -1, 2, -3, 1, 2, 3, -4];
        let vars = [true, true, false, true];
        assert!(!mid_clause_offsets_are_three(nc, &co, cl.len()));

        let mut expected_good = vec![0_u8; (nc + 3) >> 2];
        let mut expected_unsat = Vec::new();
        let mut expected_pos = vec![u32::MAX; nc];
        for c in 0..nc {
            let good = mid_clause_good_count(co[c] as usize, co[c + 1] as usize, &cl, &vars);
            expected_good[c >> 2] |= good.min(3) << ((c & 3) << 1);
            if good == 0 {
                expected_pos[c] = expected_unsat.len() as u32;
                expected_unsat.push(c as u32);
            }
        }

        let mut actual_good = vec![0_u8; (nc + 3) >> 2];
        let mut actual_unsat = Vec::new();
        let mut actual_pos = vec![u32::MAX; nc];
        rebuild_mid_state_fresh(
            nc,
            &co,
            &cl,
            &vars,
            &mut actual_good,
            &mut actual_unsat,
            &mut actual_pos,
            false,
        );

        assert_eq!(actual_good, expected_good);
        assert_eq!(actual_unsat, expected_unsat);
        assert_eq!(actual_pos, expected_pos);
    }

    #[test]
    fn packed_num_good_byte_zero_detection_matches_2bit_counts() {
        assert!(!packed_num_good_byte_has_zero(0x55));
        assert!(!packed_num_good_byte_has_zero(0xaa));
        assert!(!packed_num_good_byte_has_zero(0xff));
        assert!(packed_num_good_byte_has_zero(0x54));
        assert!(packed_num_good_byte_has_zero(0x00));
    }

    #[test]
    fn rebuild_repair_unsat_from_mid_state_skips_nonzero_bytes() {
        let packed = [
            1_u8 | (0 << 2) | (2 << 4) | (3 << 6),
            1_u8 | (1 << 2) | (0 << 4) | (0 << 6),
        ];
        let mut unsat = Vec::new();

        rebuild_repair_unsat_from_mid_state(7, &packed, &mut unsat);

        assert_eq!(unsat, vec![1, 6]);
    }

    #[test]
    fn rebuild_repair_state_overwrites_existing_counts_without_prefill() {
        let co = [0_u32, 2, 4, 5];
        let cl = [1, -2, -1, 2, 3];
        let vars = [true, true, false];
        let mut num_good = vec![9_u8; 3];
        let mut unsat = vec![99_u32];

        rebuild_repair_state(3, &co, &cl, &vars, &mut num_good, &mut unsat, false);

        assert_eq!(num_good, vec![1, 1, 0]);
        assert_eq!(unsat, vec![2]);
    }

    #[test]
    fn rebuild_repair_state_all_three_fast_path_matches_generic_reference() {
        let nc = 7;
        let co = [0_u32, 3, 6, 9, 12, 15, 18, 21];
        let cl = [
            1, 2, 3, -1, -2, -3, 1, -2, 4, -1, 2, -4, 3, 4, -2, -3, -4, 2, 1, -3, -4,
        ];
        let vars = [true, false, true, false];

        let mut expected_good = vec![0_u8; nc];
        let mut expected_unsat = Vec::new();
        for c in 0..nc {
            let good = repair_clause_good_count(co[c] as usize, co[c + 1] as usize, &cl, &vars);
            expected_good[c] = good;
            if good == 0 {
                expected_unsat.push(c as u32);
            }
        }

        let mut actual_good = vec![9_u8; nc];
        let mut actual_unsat = vec![99_u32];
        rebuild_repair_state(
            nc,
            &co,
            &cl,
            &vars,
            &mut actual_good,
            &mut actual_unsat,
            true,
        );

        assert_eq!(actual_good, expected_good);
        assert_eq!(actual_unsat, expected_unsat);
    }

    #[test]
    fn repair_clause_good_count_fast_paths_match_standard_clauses() {
        let cl = [1, -2, 3, -1, 2, -3];
        let vars = [true, true, false];

        assert_eq!(repair_clause_good_count(0, 1, &cl, &vars), 1);
        assert_eq!(repair_clause_good_count(0, 2, &cl, &vars), 1);
        assert_eq!(repair_clause_good_count(0, 3, &cl, &vars), 1);
        assert_eq!(repair_clause_good_count(3, 6, &cl, &vars), 2);
    }

    #[test]
    fn repair_clause_good_count_long_clause_keeps_saturating_semantics() {
        let cl: Vec<i32> = (1..=260).collect();
        let vars = vec![true; 260];

        assert_eq!(repair_clause_good_count(0, cl.len(), &cl, &vars), u8::MAX);
    }

    #[test]
    fn mid_clause_is_satisfied_fast_paths_match_generic_reference() {
        let cl = [1, -2, 3, -1, 2, -3, 1, -2, 3, -4];
        let vars = [true, true, false, true];

        for (s, e) in [(0, 0), (0, 1), (0, 2), (0, 3), (3, 6), (6, 10)] {
            let expected = cl[s..e].iter().any(|&lit| {
                let v = lit_var_index(lit);
                (lit > 0 && vars[v]) || (lit < 0 && !vars[v])
            });
            assert_eq!(mid_clause_is_satisfied(s, e, &cl, &vars), expected);
        }
    }

    #[test]
    fn count_satisfied_mid_counts_mixed_clause_lengths() {
        let co = [0_u32, 0, 1, 3, 6, 10];
        let cl = [1, -2, 3, -1, 2, -3, 1, -2, 3, -4];
        let vars = [true, true, false, true];

        assert_eq!(count_satisfied_mid(5, &co, &cl, &vars, false), 3);
    }

    #[test]
    fn count_satisfied_mid_all_three_fast_path_matches_generic_reference() {
        let nc = 7;
        let co = [0_u32, 3, 6, 9, 12, 15, 18, 21];
        let cl = [
            1, 2, 3, -1, -2, -3, 1, -2, 4, -1, 2, -4, 3, 4, -2, -3, -4, 2, 1, -3, -4,
        ];
        let vars = [true, false, true, false];
        let mut expected = 0usize;
        for c in 0..nc {
            if cl[co[c] as usize..co[c + 1] as usize]
                .iter()
                .any(|&lit| mid_lit_is_satisfied(lit, &vars))
            {
                expected += 1;
            }
        }

        assert_eq!(count_satisfied_mid(nc, &co, &cl, &vars, true), expected);
    }

    #[test]
    fn mid_clause_bounds_all_three_fast_path_matches_offsets() {
        let co = [0_u32, 3, 6, 9, 12];
        assert!(mid_clause_offsets_are_three(4, &co, 12));
        for cid in 0..4 {
            assert_eq!(
                unsafe { mid_clause_bounds_unchecked(cid, &co, true) },
                (co[cid] as usize, co[cid + 1] as usize)
            );
        }
    }

    #[test]
    fn count_satisfied_mid_rejects_average_three_mixed_offsets() {
        let nc = 3;
        let co = [0_u32, 2, 5, 9];
        let cl = [1, -2, -1, 2, -3, 1, 2, 3, -4];
        let vars = [true, true, false, true];
        assert!(!mid_clause_offsets_are_three(nc, &co, cl.len()));
        assert_eq!(
            unsafe { mid_clause_bounds_unchecked(1, &co, false) },
            (2, 5)
        );

        let mut expected = 0usize;
        for c in 0..nc {
            if cl[co[c] as usize..co[c + 1] as usize]
                .iter()
                .any(|&lit| mid_lit_is_satisfied(lit, &vars))
            {
                expected += 1;
            }
        }

        assert_eq!(count_satisfied_mid(nc, &co, &cl, &vars, false), expected);
    }

    #[test]
    fn mid_clause_good_count_fast_paths_match_generic_reference() {
        let cl = [1, -2, 3, -1, 2, -3, 1, -2, 3, -4];
        let vars = [true, true, false, true];

        for (s, e) in [(0, 0), (0, 1), (0, 2), (0, 3), (3, 6), (6, 10)] {
            let mut expected = 0u8;
            for &lit in &cl[s..e] {
                let v = lit_var_index(lit);
                if (lit > 0 && vars[v]) || (lit < 0 && !vars[v]) {
                    expected += 1;
                }
            }
            assert_eq!(mid_clause_good_count(s, e, &cl, &vars), expected);
        }
    }

    #[test]
    fn repair_unsat_sparse_set_keeps_unique_members() {
        let mut unsat = vec![3_u32, 8, 13];

        add_repair_unsat(&mut unsat, 8);
        assert_eq!(unsat, vec![3, 8, 13]);

        add_repair_unsat(&mut unsat, 21);
        assert_eq!(unsat, vec![3, 8, 13, 21]);

        remove_repair_unsat(&mut unsat, 8);
        assert_eq!(unsat.len(), 3);
        assert!(!unsat.contains(&8));
        assert!(unsat.contains(&3));
        assert!(unsat.contains(&13));
        assert!(unsat.contains(&21));

        remove_repair_unsat(&mut unsat, 999);
        assert_eq!(unsat.len(), 3);
    }

    #[test]
    fn repair_unsat_singleton_fast_paths_keep_set_semantics() {
        let mut unsat = Vec::new();

        add_repair_unsat(&mut unsat, 5);
        assert_eq!(unsat, vec![5]);

        add_repair_unsat(&mut unsat, 5);
        assert_eq!(unsat, vec![5]);

        remove_repair_unsat(&mut unsat, 7);
        assert_eq!(unsat, vec![5]);

        remove_repair_unsat(&mut unsat, 5);
        assert!(unsat.is_empty());
    }

    #[test]
    fn repair_unsat_endpoint_fast_paths_keep_unordered_set_semantics() {
        let mut unsat = vec![3_u32, 8, 13, 21];

        remove_repair_unsat(&mut unsat, 21);
        assert_eq!(unsat, vec![3, 8, 13]);

        remove_repair_unsat(&mut unsat, 3);
        assert_eq!(unsat.len(), 2);
        assert!(!unsat.contains(&3));
        assert!(unsat.contains(&8));
        assert!(unsat.contains(&13));

        let before = unsat.clone();
        remove_repair_unsat(&mut unsat, 999);
        assert_eq!(unsat, before);
    }

    #[test]
    fn repair_unsat_short_and_middle_fast_paths_keep_set_semantics() {
        let mut len_two = vec![7_u32, 11];
        remove_repair_unsat(&mut len_two, 99);
        assert_eq!(len_two, vec![7, 11]);

        let mut len_three = vec![7_u32, 11, 13];
        remove_repair_unsat(&mut len_three, 11);
        assert_eq!(len_three.len(), 2);
        assert!(len_three.contains(&7));
        assert!(!len_three.contains(&11));
        assert!(len_three.contains(&13));

        let mut middle = vec![3_u32, 5, 7, 9, 11];
        remove_repair_unsat(&mut middle, 7);
        assert_eq!(middle.len(), 4);
        assert!(middle.contains(&3));
        assert!(middle.contains(&5));
        assert!(!middle.contains(&7));
        assert!(middle.contains(&9));
        assert!(middle.contains(&11));
    }

    #[test]
    fn repair_unsat_len_four_middle_fast_paths_keep_set_semantics() {
        let mut first_middle = vec![3_u32, 5, 7, 9];
        remove_repair_unsat(&mut first_middle, 5);
        assert_eq!(first_middle.len(), 3);
        assert!(first_middle.contains(&3));
        assert!(!first_middle.contains(&5));
        assert!(first_middle.contains(&7));
        assert!(first_middle.contains(&9));

        let mut second_middle = vec![3_u32, 5, 7, 9];
        remove_repair_unsat(&mut second_middle, 7);
        assert_eq!(second_middle.len(), 3);
        assert!(second_middle.contains(&3));
        assert!(second_middle.contains(&5));
        assert!(!second_middle.contains(&7));
        assert!(second_middle.contains(&9));
    }

    #[test]
    fn flip_repair_var_pushes_new_unsat_once_for_duplicate_occurrence() {
        let all_off = [0_u32, 2];
        let p_bound = [2_u32];
        let all_data = [0_u32, 0];
        let num_good = [2_u8];
        let mut counts = RepairCounts::from_dense(&num_good);
        let mut vars = [true];
        let mut unsat = Vec::<u32>::new();

        flip_repair_var(
            0,
            &mut vars,
            &mut counts,
            &mut unsat,
            &all_off,
            &p_bound,
            &all_data,
        );

        assert_eq!(vars, [false]);
        assert_eq!(counts.get(0), 0);
        assert_eq!(unsat, vec![0]);
    }

    #[test]
    fn repair_counts_packed_overlay_preserves_base_counts() {
        let packed = [0_u8 | (1 << 2) | (2 << 4) | (3 << 6)];
        let mut counts = RepairCounts::from_packed(&packed, 2);

        assert_eq!(counts.get(0), 0);
        assert_eq!(counts.get(1), 1);
        assert_eq!(counts.get(2), 2);
        assert_eq!(counts.get(3), 3);

        counts.set(1, 0);
        counts.set(3, 2);

        assert_eq!(counts.get(0), 0);
        assert_eq!(counts.get(1), 0);
        assert_eq!(counts.get(2), 2);
        assert_eq!(counts.get(3), 2);
        assert_eq!(packed[0], 0_u8 | (1 << 2) | (2 << 4) | (3 << 6));
    }

    #[test]
    fn repair_counts_empty_overlay_uses_base_until_first_set() {
        let packed = [1_u8 | (2 << 2) | (3 << 4) | (1 << 6)];
        let mut counts = RepairCounts::from_packed(&packed, 1);

        assert_eq!(counts.get(0), 1);
        assert_eq!(counts.get(1), 2);
        assert_eq!(counts.get(2), 3);
        assert!(counts.overrides.is_empty());

        counts.set(2, 0);

        assert_eq!(counts.overrides.len(), 1);
        assert_eq!(counts.overrides[0].clause, 2);
        assert_eq!(counts.overrides[0].value, 0);
        assert_eq!(counts.get(0), 1);
        assert_eq!(counts.get(2), 0);
        assert_eq!(counts.get(3), 1);
    }

    #[test]
    fn repair_counts_single_override_get_matches_base_lookup() {
        let packed = [1_u8 | (2 << 2) | (3 << 4) | (1 << 6)];
        let mut counts = RepairCounts::from_packed(&packed, 1);

        counts.set(2, 0);

        assert_eq!(counts.overrides.len(), 1);
        assert_eq!(counts.get(0), 1);
        assert_eq!(counts.get(2), 0);
        assert_eq!(counts.get(3), 1);
    }

    #[test]
    fn repair_counts_single_override_prepend_fast_path_keeps_order() {
        let packed = [1_u8 | (2 << 2) | (3 << 4) | (1 << 6)];
        let mut counts = RepairCounts::from_packed(&packed, 2);

        counts.set(3, 0);
        counts.set(1, 0);

        assert_eq!(
            counts
                .overrides
                .iter()
                .map(|entry| (entry.clause, entry.value))
                .collect::<Vec<_>>(),
            vec![(1, 0), (3, 0)]
        );
    }

    #[test]
    fn repair_counts_dense_overlay_can_preallocate_capacity() {
        let num_good = vec![1_u8, 0, 2];

        let plain = RepairCounts::from_dense(&num_good);
        assert_eq!(plain.overrides.capacity(), 0);

        let preallocated = RepairCounts::from_dense_with_capacity(&num_good, 8);
        assert!(preallocated.overrides.capacity() >= 8);
        assert_eq!(preallocated.get(1), 0);
    }

    #[test]
    fn repair_counts_drop_base_equal_overrides() {
        let packed = [0_u8 | (1 << 2) | (2 << 4) | (3 << 6)];
        let mut counts = RepairCounts::from_packed(&packed, 4);

        counts.set(1, 1);
        assert!(counts.overrides.is_empty());

        counts.set(1, 0);
        counts.set(2, 3);
        assert_eq!(
            counts
                .overrides
                .iter()
                .map(|entry| entry.clause)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );

        counts.set(1, 1);
        assert_eq!(
            counts
                .overrides
                .iter()
                .map(|entry| entry.clause)
                .collect::<Vec<_>>(),
            vec![2]
        );
        assert_eq!(counts.get(1), 1);
        assert_eq!(counts.get(2), 3);
    }

    #[test]
    fn repair_counts_get_uses_sorted_overlay_boundaries() {
        let packed = [0_u8 | (1 << 2) | (2 << 4) | (3 << 6), 1_u8 | (2 << 2)];
        let mut counts = RepairCounts::from_packed(&packed, 4);

        counts.set(1, 0);
        counts.set(4, 3);

        assert_eq!(counts.get(0), 0);
        assert_eq!(counts.get(1), 0);
        assert_eq!(counts.get(2), 2);
        assert_eq!(counts.get(4), 3);
        assert_eq!(counts.get(5), 2);
    }

    #[test]
    fn repair_counts_get_small_overlay_checks_interior_entries() {
        let packed = [0_u8 | (1 << 2) | (2 << 4) | (3 << 6), 1_u8 | (2 << 2)];
        let mut counts = RepairCounts::from_packed(&packed, 4);

        counts.set(1, 0);
        counts.set(3, 2);
        counts.set(5, 3);

        assert_eq!(
            counts
                .overrides
                .iter()
                .map(|entry| entry.clause)
                .collect::<Vec<_>>(),
            vec![1, 3, 5]
        );
        assert_eq!(counts.get(2), 2);
        assert_eq!(counts.get(3), 2);
        assert_eq!(counts.get(4), 1);
    }

    #[test]
    fn repair_counts_count_occurrences_eq_matches_get_reference() {
        let dense = [0_u8, 1, 2, 1, 0, 3];
        let packed = [0_u8 | (1 << 2) | (2 << 4) | (1 << 6), 0_u8 | (3 << 2)];
        let all_data = [0_u32, 1, 2, 3, 4, 5, 0, 3];

        for target in 0..=3 {
            let dense_counts = RepairCounts::from_dense(&dense);
            let dense_ref = all_data
                .iter()
                .filter(|&&clause| dense_counts.get(clause as usize) == target)
                .count();
            assert_eq!(
                dense_counts.count_occurrences_eq(&all_data, 0, all_data.len(), target),
                dense_ref
            );

            let packed_counts = RepairCounts::from_packed(&packed, 0);
            let packed_ref = all_data
                .iter()
                .filter(|&&clause| packed_counts.get(clause as usize) == target)
                .count();
            assert_eq!(
                packed_counts.count_occurrences_eq(&all_data, 0, all_data.len(), target),
                packed_ref
            );

            let mut overlay_counts = RepairCounts::from_packed(&packed, 2);
            overlay_counts.set(2, 0);
            overlay_counts.set(5, 1);
            let overlay_ref = all_data
                .iter()
                .filter(|&&clause| overlay_counts.get(clause as usize) == target)
                .count();
            assert_eq!(
                overlay_counts.count_occurrences_eq(&all_data, 0, all_data.len(), target),
                overlay_ref
            );
        }
    }

    #[test]
    fn repair_counts_find_override_small_overlay_returns_sorted_positions() {
        let base = [1_u8; 8];
        let mut counts = RepairCounts::from_dense_with_capacity(&base, 4);

        assert_eq!(counts.find_override(0), Err(0));

        counts.set(1, 2);
        counts.set(3, 2);
        counts.set(6, 2);

        assert_eq!(counts.find_override(0), Err(0));
        assert_eq!(counts.find_override(1), Ok(0));
        assert_eq!(counts.find_override(2), Err(1));
        assert_eq!(counts.find_override(3), Ok(1));
        assert_eq!(counts.find_override(5), Err(2));
        assert_eq!(counts.find_override(6), Ok(2));
        assert_eq!(counts.find_override(7), Err(3));
    }

    #[test]
    fn repair_counts_find_override_large_overlay_fast_bounds_match_sorted_positions() {
        let base = [1_u8; 16];
        let mut counts = RepairCounts::from_dense_with_capacity(&base, 8);
        for clause in [2_usize, 4, 6, 8, 10] {
            counts.set(clause, 2);
        }

        assert_eq!(counts.find_override(1), Err(0));
        assert_eq!(counts.find_override(2), Ok(0));
        assert_eq!(counts.find_override(5), Err(2));
        assert_eq!(counts.find_override(8), Ok(3));
        assert_eq!(counts.find_override(10), Ok(4));
        assert_eq!(counts.find_override(15), Err(5));
    }

    #[test]
    fn repair_counts_get_small_overlay_direct_checks_match_reference() {
        let base = [1_u8; 10];
        let mut counts = RepairCounts::from_dense_with_capacity(&base, 4);
        for clause in [1_usize, 3, 5, 7] {
            counts.set(clause, 2);
        }

        for clause in 0..base.len() {
            let expected = counts
                .overrides
                .iter()
                .find(|entry| entry.clause as usize == clause)
                .map(|entry| entry.value)
                .unwrap_or(base[clause]);
            assert_eq!(counts.get(clause), expected);
        }
    }

    #[test]
    fn repair_counts_all_occurrences_gt_matches_get_reference() {
        let dense = [2_u8, 1, 3, 2, 0, 3];
        let packed = [2_u8 | (1 << 2) | (3 << 4) | (2 << 6), 0_u8 | (3 << 2)];
        let all_data = [0_u32, 2, 3, 5];
        let mixed_data = [0_u32, 1, 2, 3, 4, 5];

        for threshold in 0..=2 {
            let dense_counts = RepairCounts::from_dense(&dense);
            let dense_ref = all_data
                .iter()
                .all(|&clause| dense_counts.get(clause as usize) > threshold);
            assert_eq!(
                dense_counts.count_occurrences_eq(&all_data, 0, all_data.len(), 0),
                0
            );
            assert_eq!(
                dense_counts.all_occurrences_gt(&all_data, 0, all_data.len(), threshold),
                dense_ref
            );

            let packed_counts = RepairCounts::from_packed(&packed, 0);
            let packed_ref = mixed_data
                .iter()
                .all(|&clause| packed_counts.get(clause as usize) > threshold);
            assert_eq!(
                packed_counts.all_occurrences_gt(&mixed_data, 0, mixed_data.len(), threshold),
                packed_ref
            );

            let mut overlay_counts = RepairCounts::from_packed(&packed, 2);
            overlay_counts.set(1, 3);
            overlay_counts.set(4, 2);
            let overlay_ref = mixed_data
                .iter()
                .all(|&clause| overlay_counts.get(clause as usize) > threshold);
            assert_eq!(
                overlay_counts.all_occurrences_gt(&mixed_data, 0, mixed_data.len(), threshold),
                overlay_ref
            );
        }
    }

    #[test]
    fn repair_counts_override_insertions_stay_sorted() {
        let packed = [0_u8 | (1 << 2) | (2 << 4) | (3 << 6)];
        let mut counts = RepairCounts::from_packed(&packed, 3);

        counts.set(3, 1);
        counts.set(1, 0);
        counts.set(2, 3);
        counts.set(3, 2);

        let clauses = counts
            .overrides
            .iter()
            .map(|entry| entry.clause)
            .collect::<Vec<_>>();
        let values = counts
            .overrides
            .iter()
            .map(|entry| entry.value)
            .collect::<Vec<_>>();

        assert_eq!(clauses, vec![1, 2, 3]);
        assert_eq!(values, vec![0, 3, 2]);
        assert_eq!(counts.get(0), 0);
        assert_eq!(counts.get(1), 0);
        assert_eq!(counts.get(2), 3);
        assert_eq!(counts.get(3), 2);
    }

    #[test]
    fn repair_counts_append_fast_path_preserves_sorted_overrides() {
        let packed = [1_u8 | (1 << 2) | (1 << 4) | (1 << 6)];
        let mut counts = RepairCounts::from_packed(&packed, 4);

        counts.set(0, 2);
        counts.set(2, 3);
        counts.set(3, 0);
        counts.set(3, 2);
        counts.set(1, 0);

        let clauses = counts
            .overrides
            .iter()
            .map(|entry| entry.clause)
            .collect::<Vec<_>>();
        let values = counts
            .overrides
            .iter()
            .map(|entry| entry.value)
            .collect::<Vec<_>>();

        assert_eq!(clauses, vec![0, 1, 2, 3]);
        assert_eq!(values, vec![2, 0, 3, 2]);
        assert_eq!(counts.get(0), 2);
        assert_eq!(counts.get(1), 0);
        assert_eq!(counts.get(2), 3);
        assert_eq!(counts.get(3), 2);
    }

    #[test]
    fn repair_counts_increment_decrement_preserve_sorted_overrides() {
        let packed = [0_u8 | (1 << 2) | (2 << 4) | (3 << 6)];
        let mut counts = RepairCounts::from_packed(&packed, 4);

        assert_eq!(counts.increment(2), 2);
        assert_eq!(counts.get(2), 3);
        assert_eq!(counts.decrement(2), 2);
        assert_eq!(counts.get(2), 2);

        assert_eq!(counts.increment(1), 1);
        assert_eq!(counts.decrement(3), 2);

        let clauses = counts
            .overrides
            .iter()
            .map(|entry| entry.clause)
            .collect::<Vec<_>>();
        let values = counts
            .overrides
            .iter()
            .map(|entry| entry.value)
            .collect::<Vec<_>>();

        assert_eq!(clauses, vec![1, 3]);
        assert_eq!(values, vec![2, 2]);
        assert_eq!(counts.get(0), 0);
        assert_eq!(counts.get(1), 2);
        assert_eq!(counts.get(2), 2);
        assert_eq!(counts.get(3), 2);
    }

    #[test]
    fn repair_seen_contains_matches_slice_reference() {
        let seen = [7_usize, 3, 11, 5];

        for len in 0..=REPAIR_SINGLE_CLAUSE_STACK_CAPACITY {
            for v in [3_usize, 5, 7, 11, 99] {
                assert_eq!(
                    repair_seen_contains(&seen, len, v),
                    seen[..len].contains(&v)
                );
            }
        }
    }

    #[test]
    fn repair_touched_position_matches_slice_reference() {
        let touched = [(7_usize, 1_u8), (3, 2), (11, 3), (5, 1)];

        for len in 0..=REPAIR_SINGLE_CLAUSE_STACK_CAPACITY {
            for v in [3_usize, 5, 7, 11, 99] {
                let expected = touched[..len].iter().position(|(seen_v, _)| *seen_v == v);
                assert_eq!(repair_touched_position(&touched, len, v), expected);
            }
        }
    }

    #[test]
    fn single_unsat_candidate_skips_duplicate_vars_and_keeps_best_choice() {
        let co = [0_u32, 3, 4, 5];
        let cl = [3, 3, 1, -1, -2];
        let all_off = [0_u32, 2, 3, 5];
        let p_bound = [1_u32, 2, 5];
        let all_data = [0_u32, 1, 2, 0, 0];
        let vars = [false, false, false];
        let num_good = [0_u8, 1, 1];
        let counts = RepairCounts::from_dense(&num_good);

        let choice = select_single_unsat_candidate(
            0, &co, &cl, &vars, &counts, &all_off, &p_bound, &all_data, false,
        );

        assert_eq!(choice, SingleUnsatRepairChoice::Flip(2));
    }

    #[test]
    fn single_unsat_candidate_falls_back_for_wide_clause() {
        let co = [0_u32, 5];
        let cl = [1, 2, 3, 4, 5];
        let all_off = [0_u32, 1, 2, 3, 4, 5];
        let p_bound = [1_u32, 2, 3, 4, 5];
        let all_data = [0_u32, 0, 0, 0, 0];
        let vars = [false, false, false, false, false];
        let num_good = [0_u8];
        let counts = RepairCounts::from_dense(&num_good);

        let choice = select_single_unsat_candidate(
            0, &co, &cl, &vars, &counts, &all_off, &p_bound, &all_data, false,
        );

        assert_eq!(choice, SingleUnsatRepairChoice::UseGeneral);
    }

    #[test]
    fn single_unsat_purity_repair_applies_safe_literal() {
        let co = [0_u32, 1];
        let cl = [1];
        let all_off = [0_u32, 1];
        let p_bound = [1_u32];
        let all_data = [0_u32];
        let mut vars = [false];
        let num_good = [0_u8];
        let mut counts = RepairCounts::from_dense(&num_good);
        let mut unsat = vec![0_u32];

        let used = safe_purity_repair_single_unsat(
            0,
            &co,
            &cl,
            &all_off,
            &p_bound,
            &all_data,
            &mut vars,
            &mut counts,
            &mut unsat,
            8,
            false,
        );

        assert_eq!(used, Some(1));
        assert_eq!(vars, [true]);
        assert!(unsat.is_empty());
        assert_eq!(counts.get(0), 1);
    }

    #[test]
    fn single_unsat_purity_repair_keeps_touched_order_after_unsafe_skip() {
        let co = [0_u32, 3, 4];
        let cl = [1, 2, 3, -1];
        let all_off = [0_u32, 2, 3, 4];
        let p_bound = [1_u32, 3, 4];
        let all_data = [0_u32, 1, 0, 0];
        let mut vars = [false, false, false];
        let num_good = [0_u8, 1];
        let mut counts = RepairCounts::from_dense(&num_good);
        let mut unsat = vec![0_u32];

        let used = safe_purity_repair_single_unsat(
            0,
            &co,
            &cl,
            &all_off,
            &p_bound,
            &all_data,
            &mut vars,
            &mut counts,
            &mut unsat,
            8,
            false,
        );

        assert_eq!(used, Some(1));
        assert_eq!(vars, [false, true, false]);
        assert!(unsat.is_empty());
        assert_eq!(counts.get(0), 1);
        assert_eq!(counts.get(1), 1);
    }

    #[test]
    fn single_unsat_purity_repair_falls_back_for_wide_clause_without_mutating() {
        let co = [0_u32, 5];
        let cl = [1, 2, 3, 4, 5];
        let all_off = [0_u32, 1, 2, 3, 4, 5];
        let p_bound = [1_u32, 2, 3, 4, 5];
        let all_data = [0_u32, 0, 0, 0, 0];
        let mut vars = [false, false, false, false, false];
        let num_good = [0_u8];
        let mut counts = RepairCounts::from_dense(&num_good);
        let mut unsat = vec![0_u32];

        let used = safe_purity_repair_single_unsat(
            0,
            &co,
            &cl,
            &all_off,
            &p_bound,
            &all_data,
            &mut vars,
            &mut counts,
            &mut unsat,
            8,
            false,
        );

        assert_eq!(used, None);
        assert_eq!(vars, [false, false, false, false, false]);
        assert_eq!(unsat, vec![0]);
        assert_eq!(counts.get(0), 0);
    }

    #[test]
    fn mid_last_mile_purity_repair_skips_conflicting_polarity() {
        let co = [0_u32, 1, 2];
        let cl = [1, -1];
        let all_off = [0_u32, 2];
        let p_bound = [1_u32];
        let all_data = [0_u32, 1];
        let mut vars = [false];

        let mut num_good = vec![0u8; 2];
        let mut unsat = vec![0, 1];
        num_good[1] = 2;
        let mut counts = RepairCounts::from_dense(&num_good);

        let flips = safe_purity_repair(
            &co,
            &cl,
            &all_off,
            &p_bound,
            &all_data,
            &mut vars,
            &mut counts,
            &mut unsat,
            8,
            false,
        );

        assert_eq!(flips, 0);
        assert_eq!(vars, [false]);
        assert_eq!(unsat, vec![0, 1]);
    }

    #[test]
    fn mid_last_mile_purity_repair_preserves_stack_before_overflow_order() {
        let nv = REPAIR_VAR_STACK_CAPACITY + 1;
        let mut co = Vec::with_capacity(REPAIR_VAR_STACK_CAPACITY + 2);
        let mut cl = Vec::with_capacity(nv + REPAIR_VAR_STACK_CAPACITY);
        co.push(0);
        for v in 1..=nv {
            cl.push(v as i32);
        }
        co.push(cl.len() as u32);
        for v in 1..=REPAIR_VAR_STACK_CAPACITY {
            cl.push(-(v as i32));
            co.push(cl.len() as u32);
        }

        let mut all_off = vec![0_u32; nv + 1];
        let mut p_bound = vec![0_u32; nv];
        let mut all_data = Vec::with_capacity(nv + REPAIR_VAR_STACK_CAPACITY);
        for v in 0..nv {
            all_off[v] = all_data.len() as u32;
            all_data.push(0);
            p_bound[v] = all_data.len() as u32;
            if v < REPAIR_VAR_STACK_CAPACITY {
                all_data.push((v + 1) as u32);
            }
        }
        all_off[nv] = all_data.len() as u32;

        let mut vars = vec![false; nv];
        let mut num_good = vec![1_u8; REPAIR_VAR_STACK_CAPACITY + 1];
        num_good[0] = 0;
        let mut counts = RepairCounts::from_dense(&num_good);
        let mut unsat = vec![0_u32];

        let flips = safe_purity_repair(
            &co,
            &cl,
            &all_off,
            &p_bound,
            &all_data,
            &mut vars,
            &mut counts,
            &mut unsat,
            8,
            false,
        );

        assert_eq!(flips, 1);
        assert!(vars[..REPAIR_VAR_STACK_CAPACITY].iter().all(|&v| !v));
        assert!(vars[REPAIR_VAR_STACK_CAPACITY]);
        assert!(unsat.is_empty());
        assert_eq!(counts.get(0), 1);
        for c in 1..=REPAIR_VAR_STACK_CAPACITY {
            assert_eq!(counts.get(c), 1);
        }
    }

    #[test]
    fn mid_last_mile_net_repair_applies_positive_gain_flip() {
        let co = [0_u32, 1, 3, 4];
        let cl = [1, 1, 2, -2];
        let all_off = [0_u32, 2, 4];
        let p_bound = [2_u32, 3];
        let all_data = [0_u32, 1, 1, 2];
        let mut vars = [false, false];

        let (sat, flips) = refine_mid_last_mile_solution(
            2, 3, &co, &cl, &all_off, &p_bound, &all_data, &mut vars, 8,
        );

        assert_eq!(sat, 3);
        assert_eq!(flips, 1);
        assert_eq!(vars, [true, false]);
    }

    #[test]
    fn mid_last_mile_net_repair_preserves_stack_before_overflow_tie_order() {
        let nv = REPAIR_VAR_STACK_CAPACITY + 1;
        let co = [0_u32, (REPAIR_VAR_STACK_CAPACITY + 1) as u32];
        let mut cl = Vec::with_capacity(nv);
        let mut all_off = vec![0_u32; nv + 1];
        let mut p_bound = vec![0_u32; nv];
        let mut all_data = Vec::with_capacity(nv);
        for v in 0..nv {
            cl.push((v + 1) as i32);
            all_off[v] = all_data.len() as u32;
            all_data.push(0);
            p_bound[v] = all_data.len() as u32;
        }
        all_off[nv] = all_data.len() as u32;

        let mut vars = vec![false; nv];
        let num_good = [0_u8];
        let mut counts = RepairCounts::from_dense(&num_good);
        let mut unsat = vec![0_u32];

        let flips = greedy_net_repair(
            &co,
            &cl,
            &all_off,
            &p_bound,
            &all_data,
            &mut vars,
            &mut counts,
            &mut unsat,
            1,
            false,
        );

        assert_eq!(flips, 1);
        assert!(vars[0]);
        assert!(vars[1..].iter().all(|&v| !v));
        assert!(unsat.is_empty());
        assert_eq!(counts.get(0), 1);
    }
}
