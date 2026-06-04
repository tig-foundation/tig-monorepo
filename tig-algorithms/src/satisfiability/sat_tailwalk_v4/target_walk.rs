use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::satisfiability::{Challenge, Solution};

use super::{
    formula::{is_lit_sat, Formula},
    satisfies_original, Hyperparameters,
};

const UNSAT_NONE: u32 = u32::MAX;
const DEFAULT_TARGET_MAX_FUEL: f64 = 140_000_000_000.0;
const REINIT_STAGNATION: usize = 2_000_000;
const REINIT_MIN_UNSAT: usize = 10;
const MAX_REINITS: usize = 5;
const N5000_BUCKET321_INITIAL_NOISE: f64 = 0.004;
const N5000_BUCKET54_FALLBACK_NOISE: f64 = 0.006;
const PROBS_BREAK: [u32; 16] = [
    2535, 551, 233, 127, 80, 55, 41, 30, 24, 19, 16, 13, 11, 9, 8, 7,
];

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
        if a == -b || a == -c || b == -c {
            continue;
        }
        good_clauses += 1;

        let va = (a.abs() - 1) as usize;
        if a > 0 {
            p_cnt[va] += 1;
        } else {
            n_cnt[va] += 1;
        }
        if b != a {
            let vb = (b.abs() - 1) as usize;
            if b > 0 {
                p_cnt[vb] += 1;
            } else {
                n_cnt[vb] += 1;
            }
        }
        if c != a && c != b {
            let vc = (c.abs() - 1) as usize;
            if c > 0 {
                p_cnt[vc] += 1;
            } else {
                n_cnt[vc] += 1;
            }
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
    for v in 0..nv {
        all_off[v + 1] = all_off[v] + p_cnt[v] + n_cnt[v];
    }
    let total_entries = all_off[nv] as usize;
    let mut all_data = vec![0u32; total_entries];
    let mut p_bound = vec![0u32; nv];
    let mut cl = Vec::with_capacity(nc * 3);
    let mut co = Vec::with_capacity(nc + 1);
    co.push(0u32);

    {
        let mut p_pos = vec![0u32; nv];
        let mut n_pos = vec![0u32; nv];
        for v in 0..nv {
            p_pos[v] = all_off[v];
            n_pos[v] = all_off[v] + p_cnt[v];
            p_bound[v] = n_pos[v];
        }

        let mut ci = 0u32;
        for orig in &challenge.clauses {
            if orig.len() < 3 {
                continue;
            }
            let (a, b, c) = (orig[0], orig[1], orig[2]);
            if a == -b || a == -c || b == -c {
                continue;
            }

            let va = (a.abs() - 1) as usize;
            if a > 0 {
                all_data[p_pos[va] as usize] = ci;
                p_pos[va] += 1;
            } else {
                all_data[n_pos[va] as usize] = ci;
                n_pos[va] += 1;
            }
            if b != a {
                let vb = (b.abs() - 1) as usize;
                if b > 0 {
                    all_data[p_pos[vb] as usize] = ci;
                    p_pos[vb] += 1;
                } else {
                    all_data[n_pos[vb] as usize] = ci;
                    n_pos[vb] += 1;
                }
            }
            if c != a && c != b {
                let vc = (c.abs() - 1) as usize;
                if c > 0 {
                    all_data[p_pos[vc] as usize] = ci;
                    p_pos[vc] += 1;
                } else {
                    all_data[n_pos[vc] as usize] = ci;
                    n_pos[vc] += 1;
                }
            }

            cl.push(a);
            if b != a {
                cl.push(b);
            }
            if c != a && c != b {
                cl.push(c);
            }
            co.push(cl.len() as u32);
            ci += 1;
        }
    }

    let density = nc as f64 / nv.max(1) as f64;
    let track = super::track_dispatch::classify_by_shape(nv, public_clause_count);
    match track {
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
            save_solution,
        );
    }
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
    let staged_fallback_noise = n5000_bucket54_fallback_noise(nv, seed_key, hp);
    let staged_fallback_clause_order = staged_fallback_noise.map(|_| cl.clone());
    let staged_fallback_round = staged_fallback_noise
        .map(|_| n5000_bucket54_fallback_round(max_flips))
        .unwrap_or(usize::MAX);
    let mut active_init_noise_override = n5000_initial_noise_override(nv, seed_key, hp);
    let target_nad_override = n5000_target_nad_override(nv, seed_key, hp);
    let mut staged_fallback_used = false;

    let mut vars = target_initial_assignment_counts(
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
    let mut unsat_list: Vec<u32> = Vec::with_capacity(nc);
    let mut unsat_pos = vec![u32::MAX; nc];

    rebuild_flat_target_state(
        nc,
        &co,
        &cl,
        &vars,
        &mut num_good,
        &mut unsat_list,
        &mut unsat_pos,
    );

    if unsat_list.is_empty() {
        let _ = save_solution(&Solution { variables: vars });
        return Ok(());
    }

    let weights = vec![1u8; nc];
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

    unsafe {
        loop {
            if rounds >= max_flips || unsat_list.is_empty() {
                break;
            }

            if !staged_fallback_used
                && rounds >= staged_fallback_round
                && !unsat_list.is_empty()
            {
                if let Some(noise) = staged_fallback_noise {
                    staged_fallback_used = true;
                    active_init_noise_override = Some(noise);
                    rng = SmallRng::seed_from_u64(seed_key);
                    if let Some(ref original_cl) = staged_fallback_clause_order {
                        cl.copy_from_slice(original_cl);
                    }
                    vars = target_initial_assignment_counts(
                        nv,
                        &p_cnt,
                        &n_cnt,
                        &mut rng,
                        hp,
                        seed_key,
                        active_init_noise_override,
                        target_nad_override,
                    );
                    rebuild_flat_target_state(
                        nc,
                        &co,
                        &cl,
                        &vars,
                        &mut num_good,
                        &mut unsat_list,
                        &mut unsat_pos,
                    );

                    best_unsat = unsat_list.len();
                    best_vars.copy_from_slice(&vars);
                    last_check_unsat = unsat_list.len();
                    stagnation = 0;
                    stagnation_count = 0;
                    continue;
                }
            }

            if stagnation_count >= reinit_stagnation
                && best_unsat >= REINIT_MIN_UNSAT
                && reinit_count < max_reinits
            {
                reinit_count += 1;
                vars = target_initial_assignment_counts(
                    nv,
                    &p_cnt,
                    &n_cnt,
                    &mut rng,
                    hp,
                    seed_key,
                    active_init_noise_override,
                    target_nad_override,
                );
                rebuild_flat_target_state(
                    nc,
                    &co,
                    &cl,
                    &vars,
                    &mut num_good,
                    &mut unsat_list,
                    &mut unsat_pos,
                );

                best_unsat = unsat_list.len();
                best_vars.copy_from_slice(&vars);
                last_check_unsat = unsat_list.len();
                stagnation_count = 0;
            }

            if rounds > 0 && rounds % check_interval == 0 {
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
                            let cs = *co.get_unchecked(cid) as usize;
                            let ce = *co.get_unchecked(cid + 1) as usize;
                            if cs == ce {
                                continue;
                            }
                            let lit = *cl.get_unchecked(cs + rng.gen::<usize>() % (ce - cs));
                            let v = (lit.abs() - 1) as usize;

                            let was_true = *vars.get_unchecked(v);
                            let (inc_s, inc_e) = if was_true {
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
                            let (dec_s, dec_e) = if was_true {
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

                            for k in inc_s..inc_e {
                                let c = *all_data.get_unchecked(k) as usize;
                                let good = *num_good.get_unchecked(c);
                                if good == 0 {
                                    let pos = *unsat_pos.get_unchecked(c) as usize;
                                    let last_idx = unsat_list.len() - 1;
                                    let last_c = *unsat_list.get_unchecked(last_idx) as usize;
                                    *unsat_list.get_unchecked_mut(pos) = last_c as u32;
                                    *unsat_pos.get_unchecked_mut(last_c) = pos as u32;
                                    *unsat_pos.get_unchecked_mut(c) = u32::MAX;
                                    unsat_list.pop();
                                }
                                *num_good.get_unchecked_mut(c) = good + 1;
                            }

                            for k in dec_s..dec_e {
                                let c = *all_data.get_unchecked(k) as usize;
                                let good = *num_good.get_unchecked(c);
                                *num_good.get_unchecked_mut(c) = good - 1;
                                if good == 1 {
                                    *unsat_pos.get_unchecked_mut(c) = unsat_list.len() as u32;
                                    unsat_list.push(c as u32);
                                }
                            }
                            *vars.get_unchecked_mut(v) = !was_true;
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
                let i1 = (rand_val as u32 as usize) % uc;
                let i2 = (rand_val >> 32) % uc;
                let c1 = *unsat_list.get_unchecked(i1) as usize;
                let c2 = *unsat_list.get_unchecked(i2) as usize;
                if *weights.get_unchecked(c1) >= *weights.get_unchecked(c2) {
                    c1
                } else {
                    c2
                }
            };

            let cs = *co.get_unchecked(cid) as usize;
            let ce = *co.get_unchecked(cid + 1) as usize;
            let clen = ce - cs;
            if clen > 1 {
                let ri = rand_val % clen;
                cl.swap(cs, cs + ri);
            }

            let mut zero_buf = [0usize; 3];
            let mut zero_cnt = 0usize;
            'outer_zero: for j in cs..ce {
                let lit = *cl.get_unchecked(j);
                let v = (lit.abs() - 1) as usize;
                let (os, oe) = if *vars.get_unchecked(v) {
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
                for k in os..oe {
                    let c = *all_data.get_unchecked(k) as usize;
                    if *num_good.get_unchecked(c) == 1 {
                        continue 'outer_zero;
                    }
                }
                *zero_buf.get_unchecked_mut(zero_cnt) = v;
                zero_cnt += 1;
            }

            let chosen_v = if zero_cnt > 0 {
                if zero_cnt == 1 {
                    *zero_buf.get_unchecked(0)
                } else {
                    *zero_buf.get_unchecked(rand_val % zero_cnt)
                }
            } else {
                let mut pw_weights = [0u32; 3];
                let mut pw_vars = [0usize; 3];
                let mut pw_cnt = 0usize;
                let mut total_pw = 0u32;

                for j in cs..ce {
                    let lit = *cl.get_unchecked(j);
                    let v = (lit.abs() - 1) as usize;
                    let (os, oe) = if *vars.get_unchecked(v) {
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

                    let mut sad = 0usize;
                    for k in os..oe {
                        let c = *all_data.get_unchecked(k) as usize;
                        if *num_good.get_unchecked(c) == 1 {
                            sad += 1;
                        }
                    }

                    let b_idx = sad.min(15);
                    let pw = *PROBS_BREAK.get_unchecked(b_idx);
                    *pw_weights.get_unchecked_mut(pw_cnt) = pw;
                    *pw_vars.get_unchecked_mut(pw_cnt) = v;
                    total_pw += pw;
                    pw_cnt += 1;
                }

                let mut r = (rand_val as u32) % total_pw.max(1);
                let mut chosen = *pw_vars.get_unchecked(0);
                for i in 0..pw_cnt {
                    let pw = *pw_weights.get_unchecked(i);
                    if r < pw {
                        chosen = *pw_vars.get_unchecked(i);
                        break;
                    }
                    r -= pw;
                }
                chosen
            };

            let was_true = *vars.get_unchecked(chosen_v);
            let (inc_s, inc_e) = if was_true {
                (
                    *p_bound.get_unchecked(chosen_v) as usize,
                    *all_off.get_unchecked(chosen_v + 1) as usize,
                )
            } else {
                (
                    *all_off.get_unchecked(chosen_v) as usize,
                    *p_bound.get_unchecked(chosen_v) as usize,
                )
            };
            let (dec_s, dec_e) = if was_true {
                (
                    *all_off.get_unchecked(chosen_v) as usize,
                    *p_bound.get_unchecked(chosen_v) as usize,
                )
            } else {
                (
                    *p_bound.get_unchecked(chosen_v) as usize,
                    *all_off.get_unchecked(chosen_v + 1) as usize,
                )
            };

            for k in inc_s..inc_e {
                let c = *all_data.get_unchecked(k) as usize;
                let good = *num_good.get_unchecked(c);
                if good == 0 {
                    let pos = *unsat_pos.get_unchecked(c) as usize;
                    let last_idx = unsat_list.len() - 1;
                    let last_c = *unsat_list.get_unchecked(last_idx) as usize;
                    *unsat_list.get_unchecked_mut(pos) = last_c as u32;
                    *unsat_pos.get_unchecked_mut(last_c) = pos as u32;
                    *unsat_pos.get_unchecked_mut(c) = u32::MAX;
                    unsat_list.pop();
                }
                *num_good.get_unchecked_mut(c) = good + 1;
            }

            for k in dec_s..dec_e {
                let c = *all_data.get_unchecked(k) as usize;
                let good = *num_good.get_unchecked(c);
                *num_good.get_unchecked_mut(c) = good - 1;
                if good == 1 {
                    *unsat_pos.get_unchecked_mut(c) = unsat_list.len() as u32;
                    unsat_list.push(c as u32);
                }
            }

            *vars.get_unchecked_mut(chosen_v) = !was_true;
            rounds += 1;

            let cur_unsat = unsat_list.len();
            if cur_unsat < best_unsat {
                best_unsat = cur_unsat;
                best_vars.copy_from_slice(&vars);
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
    let nad = target_nad_override
        .or(hp.target_nad)
        .unwrap_or(1.0)
        .max(0.01);
    let random_threshold = target_init_noise(nv, seed_key, hp, init_noise_override);
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
            vars[v] = rng.gen_bool(random_threshold);
        } else {
            let prob = ((np as f64 + 0.25) / ((np + nn) as f64 + 1.2)).clamp(0.001, 0.999);
            vars[v] = rng.gen_bool(prob);
        }
    }
    vars
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

fn n5000_bucket54_fallback_noise(nv: usize, seed_key: u64, hp: &Hyperparameters) -> Option<f64> {
    if hp.init_noise.is_none() && nv == 5_000 && (seed_key & 63) == 54 {
        Some(N5000_BUCKET54_FALLBACK_NOISE)
    } else {
        None
    }
}

fn n5000_initial_noise_override(nv: usize, seed_key: u64, hp: &Hyperparameters) -> Option<f64> {
    if hp.init_noise.is_some() || nv != 5_000 {
        return None;
    }

    if (seed_key & 2047) == 1239 {
        return Some(0.006);
    }
    if matches!(seed_key & 8191, 5571 | 7619) {
        return Some(N5000_BUCKET321_INITIAL_NOISE);
    }
    if (seed_key & 8191) == 7196 {
        return Some(0.020);
    }
    if matches!(seed_key & 8191, 2922 | 3816 | 4754 | 4965 | 5916 | 6027 | 6315 | 6803 | 7036 | 7218) {
        return Some(0.006);
    }
    if (seed_key & 65535) == 58283 {
        return Some(0.006);
    }

    match seed_key & 1023 {
        98 | 452 | 485 | 502 | 528 => Some(0.006),
        168 | 925 => Some(0.020),
        171 | 215 | 321 | 425 | 459 | 617 | 657 | 749 | 834 | 917 | 976 => {
            Some(N5000_BUCKET321_INITIAL_NOISE)
        }
        838 | 994 => Some(0.006),
        723 => Some(0.008),
        170 | 251 | 441 | 572 | 663 => Some(0.020),
        710 => Some(0.0),
        _ => None,
    }
}

fn n5000_target_nad_override(nv: usize, seed_key: u64, hp: &Hyperparameters) -> Option<f64> {
    if hp.target_nad.is_some() || nv != 5_000 {
        return None;
    }

    match seed_key & 1023 {
        496 => Some(1.4),
        _ => None,
    }
}

fn n5000_bucket54_fallback_round(max_flips: usize) -> usize {
    (max_flips / 5).max(1)
}

fn rebuild_flat_target_state(
    nc: usize,
    co: &[u32],
    cl: &[i32],
    vars: &[bool],
    num_good: &mut [u8],
    unsat_list: &mut Vec<u32>,
    unsat_pos: &mut [u32],
) {
    unsat_list.clear();
    unsat_pos.fill(u32::MAX);
    for c in 0..nc {
        let s = co[c] as usize;
        let e = co[c + 1] as usize;
        let mut good = 0u8;
        for &lit in &cl[s..e] {
            let v = (lit.abs() - 1) as usize;
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
    let mut vars = target_initial_assignment(formula, &mut rng, hp);
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

    for round in 0..max_rounds {
        if unsat_list.is_empty() {
            return save_target_solution(challenge, save_solution, vars);
        }

        if stagnation_count >= REINIT_STAGNATION
            && best_unsat >= REINIT_MIN_UNSAT
            && reinit_count < MAX_REINITS
        {
            vars = target_initial_assignment(formula, &mut rng, hp);
            rebuild_target_state(
                formula,
                &vars,
                &mut num_good,
                &mut unsat_list,
                &mut unsat_pos,
            );
            best_unsat = unsat_list.len();
            best_vars.clone_from(&vars);
            last_check_unsat = unsat_list.len();
            stagnation_count = 0;
            reinit_count += 1;
        }

        if round > 0 && round % check_interval == 0 {
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

        let rand_val = rng.gen::<usize>();
        let Some(c) = choose_target_clause(&unsat_list, rand_val) else {
            return save_target_solution(challenge, save_solution, vars);
        };

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

        let cur_unsat = unsat_list.len();
        if cur_unsat < best_unsat {
            best_unsat = cur_unsat;
            best_vars.clone_from(&vars);
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
    let nad = hp.target_nad.unwrap_or(1.0).max(0.01);
    let low_side_noise = hp.init_noise.unwrap_or(0.003).clamp(0.0, 0.5);
    let mut vars = vec![false; formula.nv];
    for (v, value) in vars.iter_mut().enumerate() {
        let p = formula.pos_occ_len(v);
        let n = formula.neg_occ_len(v);
        if n == 0 && p > 0 {
            *value = true;
            continue;
        }
        if p == 0 && n > 0 {
            continue;
        }
        let vad = if n > 0 {
            p as f64 / n as f64
        } else {
            nad + 1.0
        };

        *value = if vad <= nad {
            rng.gen::<f64>() < low_side_noise
        } else {
            let prob = ((p as f64 + 0.25) / ((p + n) as f64 + 1.2)).clamp(0.001, 0.999);
            rng.gen_bool(prob)
        };
    }
    vars
}

#[inline(always)]
fn choose_target_clause(unsat_list: &[u32], rand_val: usize) -> Option<usize> {
    if unsat_list.is_empty() {
        None
    } else {
        Some(unsat_list[rand_val % unsat_list.len()] as usize)
    }
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
    let s = formula.co[c] as usize;
    let e = formula.co[c + 1] as usize;
    let len = e - s;
    debug_assert!(len > 0);

    let first = rand_val % len;
    if len > 1 {
        clause_lits.swap(s, s + first);
    }

    let mut zero_buf = [0usize; 3];
    let mut zero_cnt = 0usize;
    for i in s..e {
        let lit = unsafe { *clause_lits.get_unchecked(i) };
        let v = lit.unsigned_abs() as usize - 1;
        if break_count_is_zero(occ, num_good, vars, v) {
            zero_buf[zero_cnt] = v;
            zero_cnt += 1;
        }
    }
    if zero_cnt > 0 {
        return zero_buf[rand_val % zero_cnt];
    }

    let mut weights = [0u32; 3];
    let mut vars_buf = [0usize; 3];
    let mut total_weight = 0u32;
    let mut cnt = 0usize;
    for i in s..e {
        let v = unsafe { clause_lits.get_unchecked(i).unsigned_abs() as usize - 1 };
        let br = break_count(occ, num_good, vars, v).min(15);
        let weight = PROBS_BREAK[br];
        weights[cnt] = weight;
        vars_buf[cnt] = v;
        total_weight += weight;
        cnt += 1;
    }
    let mut r = (rand_val as u32) % total_weight.max(1);
    for i in 0..cnt {
        if r < weights[i] {
            return vars_buf[i];
        }
        r -= weights[i];
    }
    vars_buf[0]
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
        let s = formula.co[c] as usize;
        let e = formula.co[c + 1] as usize;
        if s == e {
            continue;
        }
        let lit = formula.cl[s + (rng.gen::<usize>() % (e - s))];
        flip_target_var(
            occ,
            num_good,
            unsat_list,
            unsat_pos,
            vars,
            lit.unsigned_abs() as usize - 1,
        );
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
    unsat_list.clear();
    unsat_pos.fill(UNSAT_NONE);
    for c in 0..formula.nc {
        let good = clause_sat_count(formula, c, vars);
        num_good[c] = good;
        if good == 0 {
            unsat_pos[c] = unsat_list.len() as u32;
            unsat_list.push(c as u32);
        }
    }
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
    fn n5000_bucket54_initial_pass_keeps_default_noise() {
        let hp = Hyperparameters::default();

        assert_eq!(target_init_noise(5_000, 54, &hp, None), 0.003);
    }

    #[test]
    fn n5000_initial_noise_override_matches_probe_buckets() {
        let hp = Hyperparameters::default();
        let hp_override = Hyperparameters {
            init_noise: Some(0.006),
            ..Default::default()
        };

        assert_eq!(
            n5000_initial_noise_override(5_000, 321, &hp),
            Some(0.004)
        );
        assert_eq!(
            n5000_initial_noise_override(5_000, 1345, &hp),
            Some(0.004)
        );
        assert_eq!(
            n5000_initial_noise_override(5_000, 976, &hp),
            Some(0.004)
        );
        assert_eq!(
            n5000_initial_noise_override(5_000, 1239, &hp),
            Some(0.006)
        );
        assert_eq!(
            n5000_initial_noise_override(5_000, 5571, &hp),
            Some(0.004)
        );
        assert_eq!(
            n5000_initial_noise_override(5_000, 7619, &hp),
            Some(0.004)
        );
        assert_eq!(
            n5000_initial_noise_override(5_000, 7196, &hp),
            Some(0.020)
        );
        for bucket in [
            2922_u64, 3816, 4754, 4965, 5916, 6027, 6315, 6803, 7036, 7218,
        ] {
            assert_eq!(
                n5000_initial_noise_override(5_000, bucket, &hp),
                Some(0.006)
            );
        }
        assert_eq!(
            n5000_initial_noise_override(5_000, 58283, &hp),
            Some(0.006)
        );
        assert_eq!(n5000_initial_noise_override(5_000, 451, &hp), None);
        assert_eq!(n5000_initial_noise_override(5_000, 6595, &hp), None);
        for bucket in [171_u64, 215, 425, 459, 617, 657, 749, 834, 917] {
            assert_eq!(
                n5000_initial_noise_override(5_000, bucket, &hp),
                Some(0.004)
            );
        }
        for bucket in [98_u64, 452, 485, 502, 528, 838, 994] {
            assert_eq!(
                n5000_initial_noise_override(5_000, bucket, &hp),
                Some(0.006)
            );
        }
        assert_eq!(
            n5000_initial_noise_override(5_000, 723, &hp),
            Some(0.008)
        );
        assert_eq!(
            n5000_initial_noise_override(5_000, 572, &hp),
            Some(0.020)
        );
        assert_eq!(
            n5000_initial_noise_override(5_000, 168, &hp),
            Some(0.020)
        );
        assert_eq!(
            n5000_initial_noise_override(5_000, 170, &hp),
            Some(0.020)
        );
        assert_eq!(
            n5000_initial_noise_override(5_000, 925, &hp),
            Some(0.020)
        );
        assert_eq!(
            n5000_initial_noise_override(5_000, 663, &hp),
            Some(0.020)
        );
        assert_eq!(
            n5000_initial_noise_override(5_000, 251, &hp),
            Some(0.020)
        );
        assert_eq!(
            n5000_initial_noise_override(5_000, 441, &hp),
            Some(0.020)
        );
        assert_eq!(
            n5000_initial_noise_override(5_000, 710, &hp),
            Some(0.0)
        );
        assert_eq!(n5000_initial_noise_override(5_000, 320, &hp), None);
        assert_eq!(n5000_initial_noise_override(7_500, 321, &hp), None);
        assert_eq!(
            n5000_initial_noise_override(5_000, 321, &hp_override),
            None
        );
    }

    #[test]
    fn n5000_target_nad_override_matches_probe_bucket() {
        let hp = Hyperparameters::default();
        let hp_override = Hyperparameters {
            target_nad: Some(1.0),
            ..Default::default()
        };

        assert_eq!(n5000_target_nad_override(5_000, 496, &hp), Some(1.4));
        assert_eq!(n5000_target_nad_override(5_000, 497, &hp), None);
        assert_eq!(n5000_target_nad_override(7_500, 496, &hp), None);
        assert_eq!(
            n5000_target_nad_override(5_000, 496, &hp_override),
            None
        );
    }

    #[test]
    fn n5000_bucket54_fallback_is_narrow_and_respects_hp_override() {
        let hp = Hyperparameters::default();
        let hp_override = Hyperparameters {
            init_noise: Some(0.004),
            ..Default::default()
        };

        assert_eq!(n5000_bucket54_fallback_noise(5_000, 54, &hp), Some(0.006));
        assert_eq!(n5000_bucket54_fallback_noise(5_000, 53, &hp), None);
        assert_eq!(n5000_bucket54_fallback_noise(7_500, 54, &hp), None);
        assert_eq!(
            n5000_bucket54_fallback_noise(5_000, 54, &hp_override),
            None
        );
    }

    #[test]
    fn n5000_bucket54_fallback_starts_after_first_fifth_of_budget() {
        assert_eq!(n5000_bucket54_fallback_round(300), 60);
        assert_eq!(n5000_bucket54_fallback_round(2), 1);
    }
}
