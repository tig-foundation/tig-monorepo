use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::satisfiability::{Challenge, Solution};

use super::{
    formula::{is_lit_sat, Formula},
    satisfies_original, target_state, Hyperparameters,
};

const UNSAT_NONE: u32 = u32::MAX;
const DEFAULT_TARGET_MAX_FUEL: f64 = 140_000_000_000.0;
const REINIT_STAGNATION: usize = 2_000_000;
const REINIT_MIN_UNSAT: usize = 10;
const MAX_REINITS: usize = 5;
const PROBS_BREAK: [u32; 16] = [
    2535, 551, 233, 127, 80, 55, 41, 30, 24, 19, 16, 13, 11, 9, 8, 7,
];

#[inline(always)]
fn select_weighted3_u32(
    rand_val: u32,
    total_weight: u32,
    cnt: usize,
    w0: u32,
    v0: usize,
    w1: u32,
    v1: usize,
    w2: u32,
    v2: usize,
) -> usize {
    let mut r = rand_val % total_weight.max(1);
    if r < w0 {
        return v0;
    }
    if cnt > 1 {
        r = r.saturating_sub(w0);
        if r < w1 {
            return v1;
        }
        if cnt > 2 {
            r = r.saturating_sub(w1);
            if r < w2 {
                return v2;
            }
        }
    }
    v0
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
fn target_clause3_common(a: i32, b: i32, c: i32) -> bool {
    a != 0 && b != 0 && c != 0 && b != a && b != -a && c != a && c != -a && c != b && c != -b
}

fn restore_target_clause_order(clauses: &[Vec<i32>], cl: &mut [i32]) {
    let mut out = 0usize;
    for orig in clauses {
        if orig.len() < 3 {
            continue;
        }
        let (a, b, c) = (orig[0], orig[1], orig[2]);
        if target_clause3_common(a, b, c) {
            debug_assert!(out + 3 <= cl.len());
            cl[out] = a;
            cl[out + 1] = b;
            cl[out + 2] = c;
            out += 3;
            continue;
        }
        if a == -b || a == -c || b == -c {
            continue;
        }

        debug_assert!(out < cl.len());
        cl[out] = a;
        out += 1;
        if b != a {
            debug_assert!(out < cl.len());
            cl[out] = b;
            out += 1;
        }
        if c != a && c != b {
            debug_assert!(out < cl.len());
            cl[out] = c;
            out += 1;
        }
    }
    debug_assert_eq!(out, cl.len());
}

#[inline(always)]
fn count_target_lit(lit: i32, p_cnt: &mut [u32], n_cnt: &mut [u32]) {
    let v = lit_var_index(lit);
    if lit > 0 {
        p_cnt[v] += 1;
    } else {
        n_cnt[v] += 1;
    }
}

#[inline(always)]
fn write_target_occ(
    lit: i32,
    clause_id: u32,
    all_data: &mut [u32],
    p_pos: &mut [u32],
    n_pos: &mut [u32],
) {
    let v = lit_var_index(lit);
    if lit > 0 {
        all_data[p_pos[v] as usize] = clause_id;
        p_pos[v] += 1;
    } else {
        all_data[n_pos[v] as usize] = clause_id;
        n_pos[v] += 1;
    }
}

fn seed_target_occ_cursors(all_off: &[u32], p_bound: &[u32], p_cnt: &mut [u32], n_cnt: &mut [u32]) {
    for v in 0..p_cnt.len() {
        p_cnt[v] = all_off[v];
        n_cnt[v] = p_bound[v];
    }
}

fn restore_target_occ_counts(
    all_off: &[u32],
    p_bound: &[u32],
    p_cnt: &mut [u32],
    n_cnt: &mut [u32],
) {
    for v in 0..p_cnt.len() {
        p_cnt[v] = p_bound[v] - all_off[v];
        n_cnt[v] = all_off[v + 1] - p_bound[v];
    }
}

struct TargetOcc {
    all_off: Vec<u32>,
    p_bound: Vec<u32>,
    all_data: Vec<u32>,
}

pub(crate) fn is_c001_target(challenge: &Challenge) -> bool {
    if challenge.num_variables < 4_500 || challenge.num_variables > 120_000 {
        return false;
    }
    let ratio_x100 = challenge.clauses.len() * 100 / challenge.num_variables.max(1);
    (400..=450).contains(&ratio_x100)
}

pub(crate) fn solve_c001_target(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hp: &Hyperparameters,
) -> Result<()> {
    let nv = challenge.num_variables;
    let public_clause_count = challenge.clauses.len();

    let seed_key = u64::from_le_bytes(challenge.seed[..8].try_into().unwrap());
    let mut rng = SmallRng::seed_from_u64(seed_key);
    let mut p_cnt = vec![0u32; nv];
    let mut n_cnt = vec![0u32; nv];
    let mut good_clauses = 0u32;

    for orig in &challenge.clauses {
        if orig.len() < 3 {
            continue;
        }
        let (a, b, c) = (orig[0], orig[1], orig[2]);
        if target_clause3_common(a, b, c) {
            good_clauses += 1;
            count_target_lit(a, &mut p_cnt, &mut n_cnt);
            count_target_lit(b, &mut p_cnt, &mut n_cnt);
            count_target_lit(c, &mut p_cnt, &mut n_cnt);
            continue;
        }
        if a == -b || a == -c || b == -c {
            continue;
        }
        good_clauses += 1;

        count_target_lit(a, &mut p_cnt, &mut n_cnt);
        if b != a {
            count_target_lit(b, &mut p_cnt, &mut n_cnt);
        }
        if c != a && c != b {
            count_target_lit(c, &mut p_cnt, &mut n_cnt);
        }
    }

    let nc = good_clauses as usize;
    if nc == 0 {
        let _ = save_solution(&Solution {
            variables: vec![false; nv],
        });
        return Ok(());
    }

    let mut all_off = vec![0u32; nv + 1];
    let mut p_bound = vec![0u32; nv];
    for v in 0..nv {
        let p_start = all_off[v];
        let n_start = p_start + p_cnt[v];
        p_bound[v] = n_start;
        all_off[v + 1] = n_start + n_cnt[v];
    }
    let total_entries = all_off[nv] as usize;
    let mut all_data = vec![0u32; total_entries];
    let mut cl = Vec::with_capacity(nc * 3);
    let mut co = Vec::with_capacity(nc + 1);
    let mut retained_clauses_all_three = true;
    co.push(0u32);

    {
        seed_target_occ_cursors(&all_off, &p_bound, &mut p_cnt, &mut n_cnt);

        let mut ci = 0u32;
        for orig in &challenge.clauses {
            if orig.len() < 3 {
                continue;
            }
            let (a, b, c) = (orig[0], orig[1], orig[2]);
            if target_clause3_common(a, b, c) {
                write_target_occ(a, ci, &mut all_data, &mut p_cnt, &mut n_cnt);
                write_target_occ(b, ci, &mut all_data, &mut p_cnt, &mut n_cnt);
                write_target_occ(c, ci, &mut all_data, &mut p_cnt, &mut n_cnt);
                cl.push(a);
                cl.push(b);
                cl.push(c);
                co.push(cl.len() as u32);
                ci += 1;
                continue;
            }
            if a == -b || a == -c || b == -c {
                continue;
            }

            let before_len = cl.len();
            write_target_occ(a, ci, &mut all_data, &mut p_cnt, &mut n_cnt);
            if b != a {
                write_target_occ(b, ci, &mut all_data, &mut p_cnt, &mut n_cnt);
            }
            if c != a && c != b {
                write_target_occ(c, ci, &mut all_data, &mut p_cnt, &mut n_cnt);
            }

            cl.push(a);
            if b != a {
                cl.push(b);
            }
            if c != a && c != b {
                cl.push(c);
            }
            retained_clauses_all_three &= cl.len() - before_len == 3;
            co.push(cl.len() as u32);
            ci += 1;
        }

        restore_target_occ_counts(&all_off, &p_bound, &mut p_cnt, &mut n_cnt);
    }
    debug_assert_eq!(
        retained_clauses_all_three,
        target_raw_clause_offsets_are_three(nc, &co, cl.len())
    );

    let density = nc as f64 / nv.max(1) as f64;
    let track = super::track_dispatch::classify_by_shape(nv, public_clause_count);
    match track {
        super::track_dispatch::C001Track::N5000R4267 => {
            return super::track_n5000_r4267::solve(
                hp,
                &mut rng,
                seed_key,
                nv,
                nc,
                density,
                p_cnt,
                n_cnt,
                &all_off,
                &p_bound,
                &all_data,
                &mut cl,
                &co,
                retained_clauses_all_three,
                save_solution,
            );
        }
        super::track_dispatch::C001Track::N7500R4267 => {
            return super::track_n7500_r4267::solve(
                hp,
                &mut rng,
                seed_key,
                nv,
                nc,
                density,
                p_cnt,
                n_cnt,
                &all_off,
                &p_bound,
                &all_data,
                &mut cl,
                &co,
                retained_clauses_all_three,
                save_solution,
            );
        }
        super::track_dispatch::C001Track::N10000R4267 => {
            return super::track_n10000_r4267::solve(
                hp,
                &mut rng,
                seed_key,
                nv,
                nc,
                density,
                p_cnt,
                n_cnt,
                &all_off,
                &p_bound,
                &all_data,
                &mut cl,
                &co,
                retained_clauses_all_three,
                save_solution,
            );
        }
        _ => {}
    }
    if (nv == 7_500 || nv == 10_000) && density >= 4.24 {
        return super::target_track_high::solve(
            hp,
            &mut rng,
            seed_key,
            nv,
            nc,
            density,
            p_cnt,
            n_cnt,
            &all_off,
            &p_bound,
            &all_data,
            &mut cl,
            &co,
            retained_clauses_all_three,
            save_solution,
        );
    }
    if nv > 10_000 {
        if density < 4.18 {
            if track == super::track_dispatch::C001Track::N100000R4150 {
                return super::track_n100000_r4150::solve(
                    hp,
                    &mut rng,
                    seed_key,
                    nv,
                    nc,
                    density,
                    p_cnt,
                    n_cnt,
                    &all_off,
                    &p_bound,
                    &all_data,
                    &mut cl,
                    &co,
                    retained_clauses_all_three,
                    save_solution,
                );
            }
            return super::target_track_low::solve(
                hp,
                &mut rng,
                nv,
                nc,
                density,
                p_cnt,
                n_cnt,
                &all_off,
                &p_bound,
                &all_data,
                &mut cl,
                &co,
                retained_clauses_all_three,
                save_solution,
            );
        }
        if density < 4.25 {
            if track == super::track_dispatch::C001Track::N100000R4200 {
                return super::track_n100000_r4200::solve(
                    hp,
                    &mut rng,
                    seed_key,
                    nv,
                    nc,
                    density,
                    p_cnt,
                    n_cnt,
                    &all_off,
                    &p_bound,
                    &all_data,
                    &mut cl,
                    &co,
                    retained_clauses_all_three,
                    save_solution,
                );
            }
            return super::target_track_mid::solve(
                hp,
                &mut rng,
                seed_key,
                nv,
                nc,
                density,
                p_cnt,
                n_cnt,
                &all_off,
                &p_bound,
                &all_data,
                &mut cl,
                &co,
                retained_clauses_all_three,
                save_solution,
            );
        }
        return super::target_track_high::solve(
            hp,
            &mut rng,
            seed_key,
            nv,
            nc,
            density,
            p_cnt,
            n_cnt,
            &all_off,
            &p_bound,
            &all_data,
            &mut cl,
            &co,
            retained_clauses_all_three,
            save_solution,
        );
    }
    let all_raw_clauses_are_three = retained_clauses_all_three;
    let avg_clause_size = cl.len() as f64 / nc.max(1) as f64;
    let difficulty_factor = density * avg_clause_size.sqrt();
    let scale_factor = if nv > 25_000 { 1.5 } else { 1.0 };
    let default_max_fuel = match track {
        super::track_dispatch::C001Track::N5000R4267 => {
            super::track_n5000_r4267::default_max_fuel(hp)
        }
        _ => match hp.hw_profile.as_deref() {
            Some("zen4") if nv == 7_500 && density >= 4.24 => 115_000_000_000.0,
            _ => DEFAULT_TARGET_MAX_FUEL,
        },
    };
    let max_fuel = hp.target_max_fuel.unwrap_or(default_max_fuel);
    let base_fuel = (2_000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt() * scale_factor;
    let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
    let remaining = (max_fuel - base_fuel).max(0.0);
    let max_flips = if flip_fuel > 0.0 {
        (remaining / flip_fuel) as usize
    } else {
        0
    };
    let active_init_noise_override = None;
    let target_nad_override = None;

    let mut vars = Vec::with_capacity(nv);
    target_initial_assignment_counts_into(
        &mut vars,
        nv,
        &p_cnt,
        &n_cnt,
        &mut rng,
        hp,
        seed_key,
        active_init_noise_override,
        target_nad_override,
    );
    let mut num_good = vec![0u8; nc];
    let mut break_score = vec![0u16; nv];
    let mut make_score = vec![0u16; nv];
    let mut sat_xor = vec![0u32; nc];
    let mut unsat_list: Vec<u32> = Vec::with_capacity(super::initial_residual_capacity(nc));
    let mut unsat_pos = vec![u32::MAX; nc];

    target_state::rebuild_u8_exact_with_make_fresh(
        nc,
        &co,
        &cl,
        &vars,
        &mut num_good,
        &mut unsat_list,
        &mut unsat_pos,
        &mut break_score,
        &mut sat_xor,
        &mut make_score,
        all_raw_clauses_are_three,
    );

    if unsat_list.is_empty() {
        let _ = save_solution(&Solution { variables: vars });
        return Ok(());
    }

    let large_problem_scale = ((nv as f64 - 25_000.0) / 35_000.0).clamp(0.0, 1.0);
    let base_interval = 60.0 - 30.0 * large_problem_scale;
    let min_interval = if large_problem_scale > 0.0 {
        15.0
    } else {
        25.0
    };
    let density_factor = if density > 4.0 { 1.2 } else { 1.0 };
    let check_interval = hp
        .check_interval
        .unwrap_or(
            (base_interval * density_factor * (1.0 + (density / 3.0).ln().max(0.0)))
                .max(min_interval) as usize,
        )
        .max(1);

    let mut last_check_unsat = unsat_list.len();
    let mut rounds = 0usize;
    let mut stagnation = 0usize;
    let stagnation_limit = hp.stagnation_limit.unwrap_or(3);
    let reinit_stagnation = if nv >= 10_000 {
        REINIT_STAGNATION / 2
    } else {
        REINIT_STAGNATION
    };
    let max_reinits = if nv >= 10_000 {
        MAX_REINITS * 3
    } else {
        MAX_REINITS
    };
    let mut best_unsat = unsat_list.len();
    let mut best_vars = vars.clone();
    let mut stagnation_count = 0usize;
    let mut reinit_count = 0usize;
    let disable_make_score = hp.disable_make_score.unwrap_or(false);
    let mut check_countdown = check_interval;
    let mut check_due = false;

    unsafe {
        loop {
            if rounds >= max_flips || unsat_list.is_empty() {
                break;
            }

            if stagnation_count >= reinit_stagnation
                && best_unsat >= REINIT_MIN_UNSAT
                && reinit_count < max_reinits
            {
                reinit_count += 1;
                target_initial_assignment_counts_into(
                    &mut vars,
                    nv,
                    &p_cnt,
                    &n_cnt,
                    &mut rng,
                    hp,
                    seed_key,
                    active_init_noise_override,
                    target_nad_override,
                );
                target_state::rebuild_u8_exact_with_make(
                    nc,
                    &co,
                    &cl,
                    &vars,
                    &mut num_good,
                    &mut unsat_list,
                    &mut unsat_pos,
                    &mut break_score,
                    &mut sat_xor,
                    &mut make_score,
                    all_raw_clauses_are_three,
                );

                best_unsat = unsat_list.len();
                if best_unsat > 0 {
                    best_vars.copy_from_slice(&vars);
                }
                last_check_unsat = unsat_list.len();
                stagnation_count = 0;
            }

            if check_due {
                check_due = false;
                let progress = last_check_unsat as i64 - unsat_list.len() as i64;
                if progress <= 0 {
                    stagnation += 1;
                    if stagnation >= stagnation_limit {
                        let kicks = if stagnation >= 8 { 6 } else { 3 };
                        for _ in 0..kicks {
                            if unsat_list.is_empty() {
                                break;
                            }

                            let rid = rng.gen::<usize>() % unsat_list.len();
                            let cid = *unsat_list.get_unchecked(rid) as usize;
                            let (cs, ce) = target_raw_clause_bounds_unchecked(
                                cid,
                                &co,
                                all_raw_clauses_are_three,
                            );
                            if cs == ce {
                                continue;
                            }
                            let lit = *cl.get_unchecked(cs + rng.gen::<usize>() % (ce - cs));
                            let v = lit_var_index(lit);

                            target_state::flip_u8_exact_with_make(
                                v,
                                &mut vars,
                                &mut num_good,
                                &mut sat_xor,
                                &mut break_score,
                                &mut make_score,
                                &mut unsat_list,
                                &mut unsat_pos,
                                &co,
                                &cl,
                                &all_off,
                                &p_bound,
                                &all_data,
                                all_raw_clauses_are_three,
                            );
                        }
                        stagnation = 0;
                    }
                } else {
                    stagnation = 0;
                }
                last_check_unsat = unsat_list.len();
            }

            if unsat_list.is_empty() {
                break;
            }

            let rand_val = rng.gen::<usize>();
            let cid = {
                let uc = unsat_list.len();
                let i = uniform_clause_sample_index(rand_val, uc);
                *unsat_list.get_unchecked(i) as usize
            };

            let (cs, ce) = target_raw_clause_bounds_unchecked(cid, &co, all_raw_clauses_are_three);
            let clen = ce - cs;
            if clen > 1 {
                let ri = rand_val % clen;
                if ri != 0 {
                    cl.swap(cs, cs + ri);
                }
            }

            let chosen_v = if all_raw_clauses_are_three {
                choose_target_raw_var3(
                    cs,
                    &cl,
                    &break_score,
                    &make_score,
                    disable_make_score,
                    rand_val,
                )
            } else {
                choose_target_raw_var_generic(
                    cs,
                    ce,
                    &cl,
                    &break_score,
                    &make_score,
                    disable_make_score,
                    rand_val,
                )
            };

            target_state::flip_u8_exact_with_make(
                chosen_v,
                &mut vars,
                &mut num_good,
                &mut sat_xor,
                &mut break_score,
                &mut make_score,
                &mut unsat_list,
                &mut unsat_pos,
                &co,
                &cl,
                &all_off,
                &p_bound,
                &all_data,
                all_raw_clauses_are_three,
            );
            rounds += 1;
            check_due = super::advance_interval_due(&mut check_countdown, check_interval);

            let cur_unsat = unsat_list.len();
            if cur_unsat < best_unsat {
                best_unsat = cur_unsat;
                if cur_unsat > 0 {
                    best_vars.copy_from_slice(&vars);
                }
                stagnation_count = 0;
            } else {
                stagnation_count += 1;
            }
        }
    }

    let final_vars = if unsat_list.is_empty() {
        vars
    } else {
        best_vars
    };
    let _ = save_solution(&Solution {
        variables: final_vars,
    });
    Ok(())
}

fn target_initial_assignment_counts(
    nv: usize,
    p_cnt: &[u32],
    n_cnt: &[u32],
    rng: &mut SmallRng,
    hp: &Hyperparameters,
    seed_key: u64,
    init_noise_override: Option<f64>,
    target_nad_override: Option<f64>,
) -> Vec<bool> {
    let mut vars = Vec::with_capacity(nv);
    target_initial_assignment_counts_into(
        &mut vars,
        nv,
        p_cnt,
        n_cnt,
        rng,
        hp,
        seed_key,
        init_noise_override,
        target_nad_override,
    );
    vars
}

fn target_initial_assignment_counts_into(
    vars: &mut Vec<bool>,
    nv: usize,
    p_cnt: &[u32],
    n_cnt: &[u32],
    rng: &mut SmallRng,
    hp: &Hyperparameters,
    seed_key: u64,
    init_noise_override: Option<f64>,
    target_nad_override: Option<f64>,
) {
    let nad = target_nad_override
        .or(hp.target_nad)
        .unwrap_or(1.0)
        .max(0.01);
    let random_threshold = target_init_noise(nv, seed_key, hp, init_noise_override);
    vars.clear();
    for v in 0..nv {
        let np = p_cnt[v] as usize;
        let nn = n_cnt[v] as usize;
        if nn == 0 && np > 0 {
            vars.push(true);
            continue;
        }
        if np == 0 && nn > 0 {
            vars.push(false);
            continue;
        }
        let vad = if nn > 0 {
            np as f64 / nn as f64
        } else {
            nad + 1.0
        };
        if vad <= nad {
            vars.push(rng.gen_bool(random_threshold));
        } else {
            let prob = ((np as f64 + 0.25) / ((np + nn) as f64 + 1.2)).clamp(0.001, 0.999);
            vars.push(rng.gen_bool(prob));
        }
    }
}

fn target_init_noise(
    nv: usize,
    _seed_key: u64,
    hp: &Hyperparameters,
    init_noise_override: Option<f64>,
) -> f64 {
    let default_noise = if nv >= 30_000 { 0.01 } else { 0.003 };
    init_noise_override
        .or(hp.init_noise)
        .unwrap_or(default_noise)
        .clamp(0.0, 0.5)
}

#[inline(always)]
fn uniform_clause_sample_index(rand_val: usize, unsat_len: usize) -> usize {
    (rand_val as u32 as usize) % unsat_len
}

#[allow(dead_code)]
fn rebuild_flat_target_state(
    nc: usize,
    co: &[u32],
    cl: &[i32],
    vars: &[bool],
    num_good: &mut [u8],
    unsat_list: &mut Vec<u32>,
    unsat_pos: &mut [u32],
) {
    clear_target_unsat_positions(unsat_list, unsat_pos);
    unsat_list.clear();
    for c in 0..nc {
        let s = co[c] as usize;
        let e = co[c + 1] as usize;
        let mut good = 0u8;
        for &lit in &cl[s..e] {
            let v = lit_var_index(lit);
            if (lit > 0 && vars[v]) || (lit < 0 && !vars[v]) {
                good += 1;
            }
        }
        num_good[c] = good;
        if good == 0 {
            unsat_pos[c] = unsat_list.len() as u32;
            unsat_list.push(c as u32);
        }
    }
}

#[allow(dead_code)]
fn solve_c001_target_legacy(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    formula: &Formula,
    hp: &Hyperparameters,
) -> Result<()> {
    let mut rng =
        SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));
    let mut clause_lits = formula.cl.clone();
    let occ = build_target_occ(formula);
    let mut vars = Vec::with_capacity(formula.nv);
    target_initial_assignment_into(&mut vars, formula, &mut rng, hp);
    let _ = save_solution(&Solution {
        variables: vars.clone(),
    });

    let mut num_good = vec![0u8; formula.nc];
    let mut unsat_list = Vec::with_capacity(formula.nc / 8 + 8);
    let mut unsat_pos = vec![UNSAT_NONE; formula.nc];
    rebuild_target_state(
        formula,
        &vars,
        &mut num_good,
        &mut unsat_list,
        &mut unsat_pos,
    );

    if unsat_list.is_empty() {
        return save_target_solution(challenge, save_solution, vars);
    }

    let density = formula.nc as f64 / formula.nv.max(1) as f64;
    let avg_clause_size = formula.cl.len() as f64 / formula.nc.max(1) as f64;
    let large_problem_scale = ((formula.nv as f64 - 25_000.0) / 35_000.0).clamp(0.0, 1.0);
    let base_interval = 60.0 - 30.0 * large_problem_scale;
    let min_interval = if large_problem_scale > 0.0 {
        15.0
    } else {
        25.0
    };
    let density_factor = if density > 4.0 { 1.2 } else { 1.0 };
    let check_interval = hp
        .check_interval
        .unwrap_or(
            (base_interval * density_factor * (1.0 + (density / 3.0).ln().max(0.0)))
                .max(min_interval) as usize,
        )
        .max(1);
    let mut last_check_unsat = unsat_list.len();
    let difficulty_factor = density * avg_clause_size.sqrt();
    let base_fuel = (2_000.0 + 100.0 * difficulty_factor) * (formula.nv as f64).sqrt();
    let flip_fuel = 200.0 + difficulty_factor;
    let max_fuel = hp.target_max_fuel.unwrap_or(DEFAULT_TARGET_MAX_FUEL);
    let max_rounds = ((max_fuel - base_fuel).max(1.0) / flip_fuel) as usize;
    let stagnation_limit = hp.stagnation_limit.unwrap_or(3);
    let mut stagnation = 0usize;
    let mut best_unsat = unsat_list.len();
    let mut best_vars = vars.clone();
    let mut stagnation_count = 0usize;
    let mut reinit_count = 0usize;
    let mut check_countdown = check_interval;
    let mut check_due = false;

    for _round in 0..max_rounds {
        if unsat_list.is_empty() {
            return save_target_solution(challenge, save_solution, vars);
        }

        if stagnation_count >= REINIT_STAGNATION
            && best_unsat >= REINIT_MIN_UNSAT
            && reinit_count < MAX_REINITS
        {
            target_initial_assignment_into(&mut vars, formula, &mut rng, hp);
            rebuild_target_state(
                formula,
                &vars,
                &mut num_good,
                &mut unsat_list,
                &mut unsat_pos,
            );
            best_unsat = unsat_list.len();
            if best_unsat > 0 {
                best_vars.clone_from(&vars);
            }
            last_check_unsat = unsat_list.len();
            stagnation_count = 0;
            reinit_count += 1;
        }

        if check_due {
            check_due = false;
            let progress = last_check_unsat as i64 - unsat_list.len() as i64;
            if progress <= 0 {
                stagnation += 1;
                if stagnation >= stagnation_limit {
                    let kicks = if stagnation >= 8 { 6 } else { 3 };
                    perturb_target(
                        formula,
                        &occ,
                        &mut num_good,
                        &mut unsat_list,
                        &mut unsat_pos,
                        &mut vars,
                        &mut rng,
                        kicks,
                    );
                    stagnation = 0;
                }
            } else {
                stagnation = 0;
            }
            last_check_unsat = unsat_list.len();
        }

        if unsat_list.is_empty() {
            return save_target_solution(challenge, save_solution, vars);
        }

        let rand_val = rng.gen::<usize>();
        let c = choose_target_clause(&unsat_list, rand_val);

        let v = choose_target_var(
            formula,
            &occ,
            &mut clause_lits,
            &num_good,
            &vars,
            c,
            rand_val,
        );
        flip_target_var(
            &occ,
            &mut num_good,
            &mut unsat_list,
            &mut unsat_pos,
            &mut vars,
            v,
        );
        check_due = super::advance_interval_due(&mut check_countdown, check_interval);

        let cur_unsat = unsat_list.len();
        if cur_unsat < best_unsat {
            best_unsat = cur_unsat;
            if cur_unsat > 0 {
                best_vars.clone_from(&vars);
            }
            stagnation_count = 0;
        } else {
            stagnation_count += 1;
        }
    }

    if unsat_list.is_empty() {
        save_target_solution(challenge, save_solution, vars)
    } else {
        let _ = save_solution(&Solution {
            variables: best_vars,
        });
        Ok(())
    }
}

fn target_initial_assignment(
    formula: &Formula,
    rng: &mut SmallRng,
    hp: &Hyperparameters,
) -> Vec<bool> {
    let mut vars = Vec::with_capacity(formula.nv);
    target_initial_assignment_into(&mut vars, formula, rng, hp);
    vars
}

fn target_initial_assignment_into(
    vars: &mut Vec<bool>,
    formula: &Formula,
    rng: &mut SmallRng,
    hp: &Hyperparameters,
) {
    let nad = hp.target_nad.unwrap_or(1.0).max(0.01);
    let low_side_noise = hp.init_noise.unwrap_or(0.003).clamp(0.0, 0.5);
    vars.clear();
    for v in 0..formula.nv {
        let p = formula.pos_occ_len(v);
        let n = formula.neg_occ_len(v);
        if n == 0 && p > 0 {
            vars.push(true);
            continue;
        }
        if p == 0 && n > 0 {
            vars.push(false);
            continue;
        }
        let vad = if n > 0 {
            p as f64 / n as f64
        } else {
            nad + 1.0
        };

        let value = if vad <= nad {
            rng.gen::<f64>() < low_side_noise
        } else {
            let prob = ((p as f64 + 0.25) / ((p + n) as f64 + 1.2)).clamp(0.001, 0.999);
            rng.gen_bool(prob)
        };
        vars.push(value);
    }
}

#[inline(always)]
fn choose_target_clause(unsat_list: &[u32], rand_val: usize) -> usize {
    debug_assert!(!unsat_list.is_empty());
    unsat_list[rand_val % unsat_list.len()] as usize
}

#[inline(always)]
fn target_raw_clause_offsets_are_three(nc: usize, co: &[u32], cl_len: usize) -> bool {
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
unsafe fn target_raw_clause_bounds_unchecked(
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
unsafe fn target_raw_zero_make(v: usize, make_score: &[u16], disable_make_score: bool) -> usize {
    if disable_make_score {
        1
    } else {
        (*make_score.get_unchecked(v) as usize).max(1)
    }
}

#[inline(always)]
unsafe fn target_raw_weight(
    v: usize,
    break_score: &[u16],
    make_score: &[u16],
    disable_make_score: bool,
) -> u32 {
    let sad = *break_score.get_unchecked(v) as usize;
    let mk = if disable_make_score {
        1u32
    } else {
        (*make_score.get_unchecked(v) as u32).max(1).min(9)
    };
    let b_idx = sad.min(15);
    (*PROBS_BREAK.get_unchecked(b_idx)).saturating_mul(mk)
}

#[inline(always)]
unsafe fn consider_target_raw_zero_candidate(
    v: usize,
    break_score: &[u16],
    make_score: &[u16],
    disable_make_score: bool,
    rand_val: usize,
    zero_cnt: &mut usize,
    zero_best: &mut usize,
    zero_best_make: &mut usize,
) {
    if *break_score.get_unchecked(v) != 0 {
        return;
    }

    let mk = target_raw_zero_make(v, make_score, disable_make_score);
    *zero_cnt += 1;
    if mk > *zero_best_make || (mk == *zero_best_make && (rand_val >> *zero_cnt) & 1 == 0) {
        *zero_best_make = mk;
        *zero_best = v;
    }
}

#[inline(always)]
unsafe fn choose_target_raw_var_generic(
    cs: usize,
    ce: usize,
    cl: &[i32],
    break_score: &[u16],
    make_score: &[u16],
    disable_make_score: bool,
    rand_val: usize,
) -> usize {
    let mut zero_cnt = 0usize;
    let mut zero_best = 0usize;
    let mut zero_best_make = 0usize;
    for j in cs..ce {
        let lit = *cl.get_unchecked(j);
        let v = lit_var_index(lit);
        consider_target_raw_zero_candidate(
            v,
            break_score,
            make_score,
            disable_make_score,
            rand_val,
            &mut zero_cnt,
            &mut zero_best,
            &mut zero_best_make,
        );
    }
    if zero_cnt > 0 {
        return zero_best;
    }

    let mut pw0 = 0u32;
    let mut pw1 = 0u32;
    let mut pw2 = 0u32;
    let mut pv0 = 0usize;
    let mut pv1 = 0usize;
    let mut pv2 = 0usize;
    let mut pw_cnt = 0usize;
    let mut total_pw = 0u32;
    for j in cs..ce {
        let lit = *cl.get_unchecked(j);
        let v = lit_var_index(lit);
        let pw = target_raw_weight(v, break_score, make_score, disable_make_score);
        match pw_cnt {
            0 => {
                pw0 = pw;
                pv0 = v;
            }
            1 => {
                pw1 = pw;
                pv1 = v;
            }
            _ => {
                pw2 = pw;
                pv2 = v;
            }
        }
        total_pw = total_pw.saturating_add(pw);
        pw_cnt += 1;
    }

    select_weighted3_u32(
        rand_val as u32,
        total_pw,
        pw_cnt,
        pw0,
        pv0,
        pw1,
        pv1,
        pw2,
        pv2,
    )
}

#[inline(always)]
unsafe fn choose_target_raw_var3(
    cs: usize,
    cl: &[i32],
    break_score: &[u16],
    make_score: &[u16],
    disable_make_score: bool,
    rand_val: usize,
) -> usize {
    debug_assert!(cs + 2 < cl.len());
    let v0 = lit_var_index(*cl.get_unchecked(cs));
    let v1 = lit_var_index(*cl.get_unchecked(cs + 1));
    let v2 = lit_var_index(*cl.get_unchecked(cs + 2));

    let mut zero_cnt = 0usize;
    let mut zero_best = 0usize;
    let mut zero_best_make = 0usize;
    consider_target_raw_zero_candidate(
        v0,
        break_score,
        make_score,
        disable_make_score,
        rand_val,
        &mut zero_cnt,
        &mut zero_best,
        &mut zero_best_make,
    );
    consider_target_raw_zero_candidate(
        v1,
        break_score,
        make_score,
        disable_make_score,
        rand_val,
        &mut zero_cnt,
        &mut zero_best,
        &mut zero_best_make,
    );
    consider_target_raw_zero_candidate(
        v2,
        break_score,
        make_score,
        disable_make_score,
        rand_val,
        &mut zero_cnt,
        &mut zero_best,
        &mut zero_best_make,
    );
    if zero_cnt > 0 {
        return zero_best;
    }

    let w0 = target_raw_weight(v0, break_score, make_score, disable_make_score);
    let w1 = target_raw_weight(v1, break_score, make_score, disable_make_score);
    let w2 = target_raw_weight(v2, break_score, make_score, disable_make_score);
    let total = w0.saturating_add(w1).saturating_add(w2);
    select_weighted3_u32(rand_val as u32, total, 3, w0, v0, w1, v1, w2, v2)
}

fn choose_target_var(
    formula: &Formula,
    occ: &TargetOcc,
    clause_lits: &mut [i32],
    num_good: &[u8],
    vars: &[bool],
    c: usize,
    rand_val: usize,
) -> usize {
    if formula.all_clauses_are_three {
        debug_assert_eq!(formula.cl.len(), formula.nc * 3);
        let s = c * 3;
        let first = rand_val % 3;
        if first != 0 {
            clause_lits.swap(s, s + first);
        }

        let v0 = lit_var_index(unsafe { *clause_lits.get_unchecked(s) });
        let v1 = lit_var_index(unsafe { *clause_lits.get_unchecked(s + 1) });
        let v2 = lit_var_index(unsafe { *clause_lits.get_unchecked(s + 2) });

        let mut zero0 = 0usize;
        let mut zero1 = 0usize;
        let mut zero2 = 0usize;
        let mut zero_cnt = 0usize;
        if break_count_is_zero(occ, num_good, vars, v0) {
            zero0 = v0;
            zero_cnt = 1;
        }
        if break_count_is_zero(occ, num_good, vars, v1) {
            match zero_cnt {
                0 => zero0 = v1,
                1 => zero1 = v1,
                _ => zero2 = v1,
            }
            zero_cnt += 1;
        }
        if break_count_is_zero(occ, num_good, vars, v2) {
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

        let br0 = break_count(occ, num_good, vars, v0).min(15);
        let br1 = break_count(occ, num_good, vars, v1).min(15);
        let br2 = break_count(occ, num_good, vars, v2).min(15);
        let w0 = PROBS_BREAK[br0];
        let w1 = PROBS_BREAK[br1];
        let w2 = PROBS_BREAK[br2];
        return select_weighted3_u32(rand_val as u32, w0 + w1 + w2, 3, w0, v0, w1, v1, w2, v2);
    }

    let (s, len) = target_clause_bounds(formula, c);
    let e = s + len;
    debug_assert!(len > 0);

    let first = rand_val % len;
    if len > 1 && first != 0 {
        clause_lits.swap(s, s + first);
    }

    let mut zero0 = 0usize;
    let mut zero1 = 0usize;
    let mut zero2 = 0usize;
    let mut zero_cnt = 0usize;
    for i in s..e {
        let lit = unsafe { *clause_lits.get_unchecked(i) };
        let v = lit_var_index(lit);
        if break_count_is_zero(occ, num_good, vars, v) {
            match zero_cnt {
                0 => zero0 = v,
                1 => zero1 = v,
                _ => zero2 = v,
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

    let mut w0 = 0u32;
    let mut w1 = 0u32;
    let mut w2 = 0u32;
    let mut v0 = 0usize;
    let mut v1 = 0usize;
    let mut v2 = 0usize;
    let mut total_weight = 0u32;
    let mut cnt = 0usize;
    for i in s..e {
        let v = unsafe { lit_var_index(*clause_lits.get_unchecked(i)) };
        let br = break_count(occ, num_good, vars, v).min(15);
        let weight = PROBS_BREAK[br];
        match cnt {
            0 => {
                w0 = weight;
                v0 = v;
            }
            1 => {
                w1 = weight;
                v1 = v;
            }
            _ => {
                w2 = weight;
                v2 = v;
            }
        }
        total_weight += weight;
        cnt += 1;
    }
    select_weighted3_u32(rand_val as u32, total_weight, cnt, w0, v0, w1, v1, w2, v2)
}

#[inline(always)]
fn break_count_is_zero(occ: &TargetOcc, num_good: &[u8], vars: &[bool], v: usize) -> bool {
    let (s, e) = break_occ_range(occ, vars, v);
    for k in s..e {
        unsafe {
            let c = *occ.all_data.get_unchecked(k) as usize;
            if *num_good.get_unchecked(c) == 1 {
                return false;
            }
        }
    }
    true
}

fn break_count(occ: &TargetOcc, num_good: &[u8], vars: &[bool], v: usize) -> usize {
    let mut br = 0usize;
    let (s, e) = break_occ_range(occ, vars, v);
    for k in s..e {
        unsafe {
            let c = *occ.all_data.get_unchecked(k) as usize;
            if *num_good.get_unchecked(c) == 1 {
                br += 1;
            }
        }
    }
    br
}

#[inline(always)]
fn break_occ_range(occ: &TargetOcc, vars: &[bool], v: usize) -> (usize, usize) {
    if unsafe { *vars.get_unchecked(v) } {
        (unsafe { *occ.all_off.get_unchecked(v) as usize }, unsafe {
            *occ.p_bound.get_unchecked(v) as usize
        })
    } else {
        (unsafe { *occ.p_bound.get_unchecked(v) as usize }, unsafe {
            *occ.all_off.get_unchecked(v + 1) as usize
        })
    }
}

fn flip_target_var(
    occ: &TargetOcc,
    num_good: &mut [u8],
    unsat_list: &mut Vec<u32>,
    unsat_pos: &mut [u32],
    vars: &mut [bool],
    v: usize,
) {
    let was_true = vars[v];
    let (inc_s, inc_e, dec_s, dec_e) = flip_occ_ranges(occ, v, was_true);

    for k in inc_s..inc_e {
        unsafe {
            let c = *occ.all_data.get_unchecked(k) as usize;
            let good = num_good.get_unchecked_mut(c);
            if *good == 0 {
                remove_unsat(unsat_list, unsat_pos, c);
            }
            *good += 1;
        }
    }

    for k in dec_s..dec_e {
        unsafe {
            let c = *occ.all_data.get_unchecked(k) as usize;
            let good = num_good.get_unchecked_mut(c);
            debug_assert!(*good > 0);
            *good -= 1;
            if *good == 0 {
                add_unsat(unsat_list, unsat_pos, c);
            }
        }
    }

    vars[v] = !was_true;
}

#[inline(always)]
fn flip_occ_ranges(occ: &TargetOcc, v: usize, was_true: bool) -> (usize, usize, usize, usize) {
    unsafe {
        let all_s = *occ.all_off.get_unchecked(v) as usize;
        let mid = *occ.p_bound.get_unchecked(v) as usize;
        let all_e = *occ.all_off.get_unchecked(v + 1) as usize;
        if was_true {
            (mid, all_e, all_s, mid)
        } else {
            (all_s, mid, mid, all_e)
        }
    }
}

fn perturb_target(
    formula: &Formula,
    occ: &TargetOcc,
    num_good: &mut [u8],
    unsat_list: &mut Vec<u32>,
    unsat_pos: &mut [u32],
    vars: &mut [bool],
    rng: &mut SmallRng,
    flips: usize,
) {
    for _ in 0..flips {
        if unsat_list.is_empty() {
            return;
        }
        let c = unsat_list[rng.gen::<usize>() % unsat_list.len()] as usize;
        let (s, len) = target_clause_bounds(formula, c);
        if len == 0 {
            continue;
        }
        let lit = formula.cl[s + (rng.gen::<usize>() % len)];
        flip_target_var(
            occ,
            num_good,
            unsat_list,
            unsat_pos,
            vars,
            lit_var_index(lit),
        );
    }
}

#[inline(always)]
fn target_clause_bounds(formula: &Formula, c: usize) -> (usize, usize) {
    if formula.all_clauses_are_three {
        debug_assert_eq!(formula.cl.len(), formula.nc * 3);
        (c * 3, 3)
    } else {
        let s = formula.co[c] as usize;
        (s, formula.co[c + 1] as usize - s)
    }
}

fn build_target_occ(formula: &Formula) -> TargetOcc {
    let mut all_off = vec![0u32; formula.nv + 1];
    let mut p_bound = vec![0u32; formula.nv];
    for v in 0..formula.nv {
        let p = formula.pos_occ_len(v) as u32;
        let n = formula.neg_occ_len(v) as u32;
        all_off[v + 1] = all_off[v] + p + n;
        p_bound[v] = all_off[v] + p;
    }

    let mut all_data = vec![0u32; all_off[formula.nv] as usize];
    for v in 0..formula.nv {
        let ps = all_off[v] as usize;
        let ns = p_bound[v] as usize;
        let pos = formula.pos_occ(v);
        let neg = formula.neg_occ(v);
        all_data[ps..ps + pos.len()].copy_from_slice(pos);
        all_data[ns..ns + neg.len()].copy_from_slice(neg);
    }

    TargetOcc {
        all_off,
        p_bound,
        all_data,
    }
}

fn rebuild_target_state(
    formula: &Formula,
    vars: &[bool],
    num_good: &mut [u8],
    unsat_list: &mut Vec<u32>,
    unsat_pos: &mut [u32],
) {
    clear_target_unsat_positions(unsat_list, unsat_pos);
    unsat_list.clear();
    if formula.all_clauses_are_three {
        debug_assert_eq!(formula.cl.len(), formula.nc * 3);
        for c in 0..formula.nc {
            let good = clause3_sat_count(&formula.cl, c * 3, vars);
            num_good[c] = good;
            if good == 0 {
                unsat_pos[c] = unsat_list.len() as u32;
                unsat_list.push(c as u32);
            }
        }
        return;
    }

    for c in 0..formula.nc {
        let good = clause_sat_count(formula, c, vars);
        num_good[c] = good;
        if good == 0 {
            unsat_pos[c] = unsat_list.len() as u32;
            unsat_list.push(c as u32);
        }
    }
}

fn clear_target_unsat_positions(unsat_list: &[u32], unsat_pos: &mut [u32]) {
    debug_assert!(unsat_list
        .iter()
        .enumerate()
        .all(|(idx, &cid)| unsat_pos.get(cid as usize).copied() == Some(idx as u32)));
    for &cid in unsat_list {
        unsat_pos[cid as usize] = UNSAT_NONE;
    }
    debug_assert!(unsat_pos.iter().all(|&pos| pos == UNSAT_NONE));
}

#[inline(always)]
fn add_unsat(unsat_list: &mut Vec<u32>, unsat_pos: &mut [u32], c: usize) {
    if unsat_pos[c] == UNSAT_NONE {
        unsat_pos[c] = unsat_list.len() as u32;
        unsat_list.push(c as u32);
    }
}

#[inline(always)]
fn remove_unsat(unsat_list: &mut Vec<u32>, unsat_pos: &mut [u32], c: usize) {
    let pos = unsat_pos[c];
    if pos == UNSAT_NONE {
        return;
    }
    let pos = pos as usize;
    let last = unsat_list.pop().expect("unsat list position without item");
    if pos < unsat_list.len() {
        unsat_list[pos] = last;
        unsat_pos[last as usize] = pos as u32;
    }
    unsat_pos[c] = UNSAT_NONE;
}

fn save_target_solution(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    vars: Vec<bool>,
) -> Result<()> {
    if satisfies_original(challenge, &vars) {
        save_solution(&Solution { variables: vars })?;
    }
    Ok(())
}

#[inline(always)]
fn clause3_sat_count(cl: &[i32], s: usize, vars: &[bool]) -> u8 {
    is_lit_sat(cl[s], vars) as u8
        + is_lit_sat(cl[s + 1], vars) as u8
        + is_lit_sat(cl[s + 2], vars) as u8
}

fn clause_sat_count(formula: &Formula, c: usize, vars: &[bool]) -> u8 {
    let mut cnt = 0u8;
    for i in formula.co[c] as usize..formula.co[c + 1] as usize {
        if is_lit_sat(formula.cl[i], vars) {
            cnt += 1;
        }
    }
    cnt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_walk_lit_var_index_matches_abs_index_for_valid_literals() {
        for lit in [-128_i32, -7, -1, 1, 7, 128] {
            assert_eq!(lit_var_index(lit), (lit.abs() - 1) as usize);
        }
    }

    #[test]
    fn target_clause3_common_accepts_only_distinct_non_opposite_nonzero_literals() {
        assert!(target_clause3_common(1, -2, 3));
        assert!(!target_clause3_common(1, 1, 3));
        assert!(!target_clause3_common(1, -1, 3));
        assert!(!target_clause3_common(1, 2, -2));
        assert!(!target_clause3_common(0, 2, 3));
    }

    #[test]
    fn uniform_clause_sample_matches_old_all_one_weight_tournament() {
        for unsat_len in [1usize, 2, 3, 17, 257] {
            for rand_val in [
                0usize,
                1,
                7,
                0x0000_0001_0000_0000usize,
                0x89ab_cdef_0123_4567usize,
            ] {
                let old_i1 = (rand_val as u32 as usize) % unsat_len;
                let old_i2 = (rand_val >> 32) % unsat_len;
                let old_selected = if 1u8 >= 1u8 { old_i1 } else { old_i2 };

                assert_eq!(
                    uniform_clause_sample_index(rand_val, unsat_len),
                    old_selected
                );
            }
        }
    }

    #[test]
    fn target_initial_assignment_counts_into_matches_allocating_path() {
        let hp = Hyperparameters::default();
        let p_cnt = vec![3u32, 0, 1, 5, 2, 0, 10, 4];
        let n_cnt = vec![0u32, 4, 4, 1, 2, 0, 2, 8];
        let seed_key = 0x1357_9bdf_2468_ace0;
        let init_noise_override = Some(0.017);
        let target_nad_override = Some(1.1);

        let mut rng_expected = SmallRng::seed_from_u64(0xfeed_face_cafe_beef);
        let expected = target_initial_assignment_counts(
            p_cnt.len(),
            &p_cnt,
            &n_cnt,
            &mut rng_expected,
            &hp,
            seed_key,
            init_noise_override,
            target_nad_override,
        );

        let mut rng_actual = SmallRng::seed_from_u64(0xfeed_face_cafe_beef);
        let mut actual = Vec::with_capacity(32);
        actual.extend([true, false, true, true, false]);
        let reused_capacity = actual.capacity();
        target_initial_assignment_counts_into(
            &mut actual,
            p_cnt.len(),
            &p_cnt,
            &n_cnt,
            &mut rng_actual,
            &hp,
            seed_key,
            init_noise_override,
            target_nad_override,
        );

        assert_eq!(actual, expected);
        assert_eq!(actual.len(), p_cnt.len());
        assert_eq!(actual.capacity(), reused_capacity);
    }

    #[test]
    fn target_initial_assignment_into_matches_allocating_path() {
        let hp = Hyperparameters {
            init_noise: Some(0.031),
            target_nad: Some(1.2),
            ..Default::default()
        };
        let formula = Formula::from_raw(
            8,
            &[
                vec![1, 2, -3],
                vec![-2, 4, 5],
                vec![3, -4, -5],
                vec![-6, 7, 8],
                vec![6, -7, -8],
            ],
        );

        let mut rng_expected = SmallRng::seed_from_u64(0x0123_4567_89ab_cdef);
        let expected = target_initial_assignment(&formula, &mut rng_expected, &hp);

        let mut rng_actual = SmallRng::seed_from_u64(0x0123_4567_89ab_cdef);
        let mut actual = Vec::with_capacity(32);
        actual.extend([false, true, false, true]);
        let reused_capacity = actual.capacity();
        target_initial_assignment_into(&mut actual, &formula, &mut rng_actual, &hp);

        assert_eq!(actual, expected);
        assert_eq!(actual.len(), formula.nv);
        assert_eq!(actual.capacity(), reused_capacity);
    }

    #[test]
    fn target_clause_bounds_all_three_fast_path_matches_offsets() {
        let formula = Formula::from_raw(
            4,
            &[
                vec![1, -2, 3],
                vec![-1, 2, -3],
                vec![1, 3, -4],
                vec![-1, -3, 4],
            ],
        );
        assert!(formula.all_clauses_are_three);

        for c in 0..formula.nc {
            let (s, len) = target_clause_bounds(&formula, c);
            assert_eq!(s, formula.co[c] as usize);
            assert_eq!(len, formula.co[c + 1] as usize - formula.co[c] as usize);
        }
    }

    #[test]
    fn target_raw_clause_bounds_all_three_fast_path_matches_offsets() {
        let co = [0_u32, 3, 6, 9, 12];
        assert!(target_raw_clause_offsets_are_three(4, &co, 12));
        for cid in 0..4 {
            assert_eq!(
                unsafe { target_raw_clause_bounds_unchecked(cid, &co, true) },
                (co[cid] as usize, co[cid + 1] as usize)
            );
        }
    }

    #[test]
    fn target_raw_clause_offsets_reject_average_three_mixed_lengths() {
        let co = [0_u32, 2, 5, 9];
        assert!(!target_raw_clause_offsets_are_three(3, &co, 9));
        assert_eq!(
            unsafe { target_raw_clause_bounds_unchecked(1, &co, false) },
            (2, 5)
        );
    }

    #[test]
    fn target_clause_bounds_mixed_lengths_use_offsets() {
        let formula = Formula::from_raw(3, &[vec![1, 2, 3], vec![1, 1, -2], vec![-1, 2]]);
        assert!(!formula.all_clauses_are_three);

        for c in 0..formula.nc {
            let (s, len) = target_clause_bounds(&formula, c);
            assert_eq!(s, formula.co[c] as usize);
            assert_eq!(len, formula.co[c + 1] as usize - formula.co[c] as usize);
        }
    }

    #[test]
    fn restore_target_clause_order_rebuilds_initial_flat_order() {
        let clauses = vec![
            vec![1, -2, 3],
            vec![4, 4, -5],
            vec![6, -6, 7],
            vec![8, 9, 10, -8],
            vec![11, 12, 11],
        ];
        let expected = vec![1, -2, 3, 4, -5, 8, 9, 10, 11, 12];
        let mut cl = expected.clone();
        cl.swap(0, 2);
        cl.swap(5, 9);

        restore_target_clause_order(&clauses, &mut cl);

        assert_eq!(cl, expected);
    }

    #[test]
    fn target_rebuild_state_sparse_clears_previous_unsat_positions() {
        let formula = Formula::from_raw(3, &[vec![1], vec![2], vec![-1, -2], vec![3]]);
        let vars = vec![true, false, true];
        let mut num_good = vec![0u8; formula.nc];
        let mut unsat_list = vec![0_u32, 3];
        let mut unsat_pos = vec![0_u32, UNSAT_NONE, UNSAT_NONE, 1];

        rebuild_target_state(
            &formula,
            &vars,
            &mut num_good,
            &mut unsat_list,
            &mut unsat_pos,
        );

        assert_eq!(num_good, vec![1, 0, 1, 1]);
        assert_eq!(unsat_list, vec![1]);
        assert_eq!(unsat_pos, vec![UNSAT_NONE, 0, UNSAT_NONE, UNSAT_NONE]);
    }

    #[test]
    fn target_rebuild_state_all_three_fast_path_matches_reference() {
        let formula = Formula::from_raw(
            4,
            &[
                vec![1, -2, 3],
                vec![-1, 2, -3],
                vec![1, 3, -4],
                vec![-1, -3, 4],
            ],
        );
        assert!(formula.all_clauses_are_three);
        let vars = vec![true, false, false, false];

        let expected: Vec<u8> = (0..formula.nc)
            .map(|c| clause_sat_count(&formula, c, &vars))
            .collect();
        let expected_unsat: Vec<u32> = expected
            .iter()
            .enumerate()
            .filter_map(|(c, &good)| (good == 0).then_some(c as u32))
            .collect();

        let mut num_good = vec![9u8; formula.nc];
        let mut unsat_list = vec![0_u32, 2];
        let mut unsat_pos = vec![0_u32, UNSAT_NONE, 1, UNSAT_NONE];

        rebuild_target_state(
            &formula,
            &vars,
            &mut num_good,
            &mut unsat_list,
            &mut unsat_pos,
        );

        assert_eq!(num_good, expected);
        assert_eq!(unsat_list, expected_unsat);
        for c in 0..formula.nc {
            let expected_pos = expected_unsat
                .iter()
                .position(|&cid| cid as usize == c)
                .map_or(UNSAT_NONE, |idx| idx as u32);
            assert_eq!(unsat_pos[c], expected_pos);
        }
    }

    #[test]
    fn choose_target_var_zero_break_keeps_rand_mod_candidate_order() {
        let formula = Formula::from_raw(3, &[vec![1, 2, 3]]);
        let occ = build_target_occ(&formula);
        let vars = vec![false, false, false];
        let num_good = vec![0u8; formula.nc];

        for rand_val in 0usize..6 {
            let mut clause_lits = formula.cl.clone();
            let first = rand_val % 3;
            let mut expected_order = [0usize, 1, 2];
            expected_order.swap(0, first);
            let expected = expected_order[rand_val % 3];

            assert_eq!(
                choose_target_var(
                    &formula,
                    &occ,
                    &mut clause_lits,
                    &num_good,
                    &vars,
                    0,
                    rand_val,
                ),
                expected
            );
        }
    }

    #[test]
    fn choose_target_var_all_three_direct_path_matches_generic_reference() {
        let formula = Formula::from_raw(
            4,
            &[
                vec![1, 2, 3],
                vec![-1, 2, 4],
                vec![1, -3, -4],
                vec![-2, 3, 4],
            ],
        );
        assert!(formula.all_clauses_are_three);
        let occ = build_target_occ(&formula);
        let vars = vec![false, true, false, true];

        let mut generic_formula = formula.clone();
        generic_formula.all_clauses_are_three = false;

        for num_good in [vec![0_u8; formula.nc], vec![1_u8; formula.nc]] {
            for c in 0..formula.nc {
                for rand_val in [
                    0usize,
                    1,
                    2,
                    3,
                    7,
                    0x0123_4567_89ab_cdefusize,
                    0xfedc_ba98_7654_3210usize,
                ] {
                    let mut direct_lits = formula.cl.clone();
                    let mut generic_lits = formula.cl.clone();

                    let direct = choose_target_var(
                        &formula,
                        &occ,
                        &mut direct_lits,
                        &num_good,
                        &vars,
                        c,
                        rand_val,
                    );
                    let generic = choose_target_var(
                        &generic_formula,
                        &occ,
                        &mut generic_lits,
                        &num_good,
                        &vars,
                        c,
                        rand_val,
                    );

                    assert_eq!(direct, generic);
                    assert_eq!(direct_lits, generic_lits);
                }
            }
        }
    }

    #[test]
    fn choose_target_raw_var3_matches_generic_reference() {
        let cl = [99, 1, -2, 3, 100];
        let rand_values = [
            0usize,
            1,
            2,
            3,
            7,
            0x0123_4567_89ab_cdefusize,
            0xfedc_ba98_7654_3210usize,
        ];
        let cases = [
            ([0_u16, 0, 0], [1_u16, 5, 9]),
            ([0_u16, 2, 0], [4_u16, 9, 4]),
            ([3_u16, 1, 5], [2_u16, 8, 6]),
        ];

        for disable_make_score in [false, true] {
            for (break_score, make_score) in cases {
                for rand_val in rand_values {
                    let generic = unsafe {
                        choose_target_raw_var_generic(
                            1,
                            4,
                            &cl,
                            &break_score,
                            &make_score,
                            disable_make_score,
                            rand_val,
                        )
                    };
                    let direct = unsafe {
                        choose_target_raw_var3(
                            1,
                            &cl,
                            &break_score,
                            &make_score,
                            disable_make_score,
                            rand_val,
                        )
                    };
                    assert_eq!(
                        direct, generic,
                        "disable_make_score={disable_make_score} break_score={break_score:?} make_score={make_score:?} rand_val={rand_val}"
                    );
                }
            }
        }
    }

    #[test]
    fn weighted3_local_slots_match_array_reference() {
        fn reference(rand_val: u32, total: u32, cnt: usize, weights: [u32; 3]) -> usize {
            let vars = [11usize, 22, 33];
            let mut r = rand_val % total.max(1);
            let mut chosen = vars[0];
            for i in 0..cnt {
                if r < weights[i] {
                    chosen = vars[i];
                    break;
                }
                r = r.saturating_sub(weights[i]);
            }
            chosen
        }

        for weights in [[0_u32, 0, 0], [7, 0, 5], [2, 3, 4], [2535, 551, 233]] {
            for cnt in 1usize..=3 {
                let total = weights[..cnt]
                    .iter()
                    .fold(0u32, |acc, &w| acc.saturating_add(w));
                for rand_val in 0u32..64 {
                    assert_eq!(
                        select_weighted3_u32(
                            rand_val, total, cnt, weights[0], 11, weights[1], 22, weights[2], 33,
                        ),
                        reference(rand_val, total, cnt, weights)
                    );
                }
            }
        }
    }

    #[test]
    fn target_common_count_and_occ_helpers_preserve_sign_buckets() {
        let mut p_cnt = vec![0u32; 3];
        let mut n_cnt = vec![0u32; 3];
        count_target_lit(1, &mut p_cnt, &mut n_cnt);
        count_target_lit(-2, &mut p_cnt, &mut n_cnt);
        count_target_lit(3, &mut p_cnt, &mut n_cnt);

        assert_eq!(p_cnt, vec![1, 0, 1]);
        assert_eq!(n_cnt, vec![0, 1, 0]);

        let mut all_data = vec![0u32; 3];
        let mut p_pos = vec![0u32, 1, 2];
        let mut n_pos = vec![0u32, 1, 2];
        write_target_occ(1, 7, &mut all_data, &mut p_pos, &mut n_pos);
        write_target_occ(-2, 8, &mut all_data, &mut p_pos, &mut n_pos);
        write_target_occ(3, 9, &mut all_data, &mut p_pos, &mut n_pos);

        assert_eq!(all_data, vec![7, 8, 9]);
        assert_eq!(p_pos, vec![1, 1, 3]);
        assert_eq!(n_pos, vec![0, 2, 2]);
    }

    #[test]
    fn target_occ_cursor_reuse_restores_original_counts() {
        let all_off = vec![0u32, 3, 5, 8];
        let p_bound = vec![2u32, 4, 7];
        let mut p_cnt = vec![2u32, 1, 2];
        let mut n_cnt = vec![1u32, 1, 1];

        seed_target_occ_cursors(&all_off, &p_bound, &mut p_cnt, &mut n_cnt);
        assert_eq!(p_cnt, vec![0, 3, 5]);
        assert_eq!(n_cnt, vec![2, 4, 7]);

        p_cnt[0] += 2;
        n_cnt[0] += 1;
        p_cnt[1] += 1;
        n_cnt[1] += 1;
        p_cnt[2] += 2;
        n_cnt[2] += 1;

        restore_target_occ_counts(&all_off, &p_bound, &mut p_cnt, &mut n_cnt);
        assert_eq!(p_cnt, vec![2, 1, 2]);
        assert_eq!(n_cnt, vec![1, 1, 1]);
    }

    #[test]
    fn target_init_noise_is_seed_independent_without_an_override() {
        let hp = Hyperparameters::default();

        assert_eq!(target_init_noise(5_000, 54, &hp, None), 0.003);
        assert_eq!(target_init_noise(5_000, 321, &hp, None), 0.003);
    }
}
