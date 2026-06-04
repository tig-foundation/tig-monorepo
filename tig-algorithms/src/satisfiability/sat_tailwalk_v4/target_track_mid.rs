use super::Hyperparameters;
use anyhow::Result;
use rand::{rngs::SmallRng, Rng};
use tig_challenges::satisfiability::*;

#[inline(always)]
fn prob_cutoff_u64(prob: f64) -> u64 {
    (prob.max(0.0).min(1.0) * (u64::MAX as f64)) as u64
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

    let mut vars = initial_assignment_mid(nv, density, &p_cnt, &n_cnt, rng, hp, seed_key, None);

    let appearances: Vec<u8> = (0..nv)
        .map(|v| ((p_cnt[v] + n_cnt[v]) as usize).min(255) as u8)
        .collect();

    let ng_len = (nc + 3) >> 2;
    let mut num_good = vec![0u8; ng_len];
    let mut unsat_list: Vec<u32> = Vec::with_capacity(nc);
    let mut unsat_pos = vec![u32::MAX; nc];
    rebuild_mid_state(
        nc,
        co,
        cl,
        &vars,
        &mut num_good,
        &mut unsat_list,
        &mut unsat_pos,
    );

    if unsat_list.is_empty() {
        let _ = save_solution(&Solution { variables: vars });
        return Ok(());
    }

    let base_prob = hp
        .target_base_prob
        .unwrap_or(0.45 + 0.1 * (density / 5.0).min(1.0));
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
    let max_random_prob = hp.max_prob.unwrap_or(0.9);
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
    let mut var_age = vec![0u8; nv];
    let mut countdown = check_interval;
    let mut rounds = 0usize;
    let mut best_unsat_seen = unsat_list.len();
    let use_probsat_pick = nv == 100_000 && (4.19..4.21).contains(&density);
    let mut probsat_break = [0.0f64; 256];
    if use_probsat_pick {
        let cb: f64 = if avg_clause_size > 4.5 {
            3.5
        } else if avg_clause_size > 3.5 {
            2.85
        } else {
            2.06
        };
        for (i, w) in probsat_break.iter_mut().enumerate() {
            *w = cb.powf(-(i as f64));
        }
    }

    macro_rules! run_search {
        () => {{
            unsafe {
                loop {
                    if unsat_list.is_empty() || rounds >= max_flips {
                        break;
                    }
                    if let Some(cut_round) = tail_cut_round {
                        if rounds >= cut_round {
                            let cur_unsat = unsat_list.len();
                            if cur_unsat < best_unsat_seen {
                                best_unsat_seen = cur_unsat;
                            }
                            if cur_unsat > tail_cut_unsat_threshold
                                && best_unsat_seen > tail_cut_best_unsat_threshold
                            {
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
                        }
                        let progress = last_check_residual as i64 - cur_residual as i64;
                        let progress_ratio = progress as f64 / last_check_residual.max(1) as f64;

                        if progress <= 0 {
                            stagnation += 1;
                            let prob_adjustment = prob_adjustment_factor
                                * (-progress as f64 / last_check_residual.max(1) as f64).min(1.0);
                            current_prob = (current_prob + prob_adjustment).min(max_random_prob);

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

                                for _ in 0..kicks {
                                    if unsat_list.is_empty() {
                                        break;
                                    }
                                    let rid = rng.gen::<usize>() % unsat_list.len();
                                    let pcid = *unsat_list.get_unchecked(rid) as usize;

                                    let pcs = *co.get_unchecked(pcid) as usize;
                                    let pce = *co.get_unchecked(pcid + 1) as usize;
                                    if pcs == pce {
                                        continue;
                                    }
                                    let lit =
                                        *cl.get_unchecked(pcs + rng.gen::<usize>() % (pce - pcs));
                                    let v = (lit.abs() - 1) as usize;

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
                                        let old =
                                            (*num_good.get_unchecked(byte_idx) >> shift) & 3;
                                        *num_good.get_unchecked_mut(byte_idx) += 1u8 << shift;
                                        if old == 0 {
                                            let pos = *unsat_pos.get_unchecked(c) as usize;
                                            let last_idx = unsat_list.len() - 1;
                                            let last_c =
                                                *unsat_list.get_unchecked(last_idx) as usize;
                                            *unsat_list.get_unchecked_mut(pos) = last_c as u32;
                                            *unsat_pos.get_unchecked_mut(last_c) = pos as u32;
                                            *unsat_pos.get_unchecked_mut(c) = u32::MAX;
                                            unsat_list.set_len(last_idx);
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
                                    *var_age.get_unchecked_mut(v) = 0;
                                }
                                stagnation = 0;
                            }
                        } else if progress_ratio > progress_threshold {
                            stagnation = 0;
                            current_prob = base_prob;
                        } else {
                            stagnation = 0;
                            current_prob =
                                current_prob * smoothing_factor + base_prob * (1.0 - smoothing_factor);
                        }

                        current_prob_cutoff = prob_cutoff_u64(current_prob);
                        last_check_residual = unsat_list.len();
                    }

                    let rand_val = rng.gen::<usize>();

                    if unsat_list.is_empty() {
                        break;
                    }
                    let cid = *unsat_list.get_unchecked(rand_val % unsat_list.len()) as usize;

                    let cs = *co.get_unchecked(cid) as usize;
                    let ce = *co.get_unchecked(cid + 1) as usize;
                    let clen = ce - cs;

                    if clen > 1 {
                        let ri = rand_val % clen;
                        cl.swap(cs, cs + ri);
                    }

                    let v_idx = if use_probsat_pick {
                        let mut zero_buf: [usize; 3] = [0; 3];
                        let mut zero_cnt: usize = 0;
                        let mut total_weight = 0.0;
                        let mut weights = [0.0f64; 8];
                        let limit = (ce - cs).min(weights.len());

                        for idx in 0..limit {
                            let l = *cl.get_unchecked(cs + idx);
                            let abs_l = (l.abs() - 1) as usize;
                            let (os, oe) = if l > 0 {
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

                            if sad == 0 {
                                *zero_buf.get_unchecked_mut(zero_cnt) = abs_l;
                                zero_cnt += 1;
                            } else {
                                let weight = *probsat_break.get_unchecked(sad.min(255));
                                *weights.get_unchecked_mut(idx) = weight;
                                total_weight += weight;
                            }
                        }

                        if zero_cnt > 0 {
                            if zero_cnt == 1 {
                                *zero_buf.get_unchecked(0)
                            } else {
                                *zero_buf.get_unchecked(rand_val % zero_cnt)
                            }
                        } else {
                            let threshold = rng.gen::<f64>() * total_weight;
                            let mut accum = 0.0;
                            let mut selected = (cl.get_unchecked(cs).abs() - 1) as usize;
                            for idx in 0..limit {
                                accum += *weights.get_unchecked(idx);
                                if accum >= threshold {
                                    selected = (cl.get_unchecked(cs + idx).abs() - 1) as usize;
                                    break;
                                }
                            }
                            selected
                        }
                    } else {
                        let mut zero_buf: [usize; 3] = [0; 3];
                        let mut zero_cnt: usize = 0;
                        'outer: for j in cs..ce {
                            let l = *cl.get_unchecked(j);
                            let abs_l = (l.abs() - 1) as usize;
                            let (os, oe) = if l > 0 {
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
                            for k in os..oe {
                                let c = *all_data.get_unchecked(k) as usize;
                                if (*num_good.get_unchecked(c >> 2) >> ((c & 3) << 1)) & 3 == 1 {
                                    continue 'outer;
                                }
                            }
                            *zero_buf.get_unchecked_mut(zero_cnt) = abs_l;
                            zero_cnt += 1;
                        }

                        if zero_cnt > 0 {
                            if zero_cnt == 1 {
                                *zero_buf.get_unchecked(0)
                            } else {
                                *zero_buf.get_unchecked(rand_val % zero_cnt)
                            }
                        } else if (rand_val as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            <= current_prob_cutoff
                        {
                            (cl.get_unchecked(cs).abs() - 1) as usize
                        } else {
                            let mut min_sad = usize::MAX;
                            let mut v_min = (cl.get_unchecked(cs).abs() - 1) as usize;
                            let mut min_weight = usize::MAX;

                            for j in cs..ce {
                                let l = *cl.get_unchecked(j);
                                let abs_l = (l.abs() - 1) as usize;
                                let (os, oe) = if l > 0 {
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
                                    if sad >= min_sad {
                                        break;
                                    }
                                }

                                if sad == 0 {
                                    let app = *appearances.get_unchecked(abs_l) as usize;
                                    let age_bonus = (*var_age.get_unchecked(abs_l) as usize) / 4;
                                    let adjusted_weight = app.saturating_sub(age_bonus);
                                    if min_sad > 0 || adjusted_weight < min_weight {
                                        min_sad = 0;
                                        min_weight = adjusted_weight;
                                        v_min = abs_l;
                                    }
                                } else if min_sad > 0 {
                                    let app = *appearances.get_unchecked(abs_l) as usize;
                                    let age_bonus = (*var_age.get_unchecked(abs_l) as usize) / 2;
                                    let combined_weight = sad * sad * 1024 + app - age_bonus.min(50);
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
                            let pos = *unsat_pos.get_unchecked(c) as usize;
                            let last_idx = unsat_list.len() - 1;
                            let last_c = *unsat_list.get_unchecked(last_idx) as usize;
                            *unsat_list.get_unchecked_mut(pos) = last_c as u32;
                            *unsat_pos.get_unchecked_mut(last_c) = pos as u32;
                            *unsat_pos.get_unchecked_mut(c) = u32::MAX;
                            unsat_list.set_len(last_idx);
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
                    *var_age.get_unchecked_mut(v_idx) = 0;
                    for j in cs..ce {
                        let l = *cl.get_unchecked(j);
                        let var = (l.abs() - 1) as usize;
                        let age = var_age.get_unchecked_mut(var);
                        *age = age.saturating_add(1);
                    }
                    rounds += 1;
                }
            }
        }};
    }

    run_search!();

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
    seed_key: u64,
    init_noise_override: Option<f64>,
) -> Vec<bool> {
    let nvf = nv as f64;
    let nad = hp.target_nad.unwrap_or(1.0).max(0.01);
    let default_random_threshold = 0.003 + 0.007 / (1.0 + (-(nvf - 30000.0) / 8000.0).exp());
    let portfolio_random_threshold = if hp.init_noise.is_none()
        && nv == 100_000
        && (4.19..4.21).contains(&density)
        && hp.target_max_fuel.unwrap_or(0.0) >= 400_000_000_000.0
        && hp.target_tail_cut_fuel == Some(0.0)
    {
        if (seed_key & 31) == 19 {
            0.043
        } else if (seed_key & 63) == 4 {
            0.042
        } else if (seed_key & 1) == 1 {
            0.042
        } else {
            default_random_threshold
        }
    } else {
        default_random_threshold
    };
    let random_threshold = init_noise_override
        .or(hp.init_noise)
        .unwrap_or(portfolio_random_threshold)
        .clamp(0.0, 0.5);
    let steep = 0.35 / (1.0 + (density - 4.18).max(0.0) * 12.0);
    let mut vars = vec![false; nv];
    for v in 0..nv {
        let np = p_cnt[v] as f64;
        let nn = n_cnt[v] as f64;
        if nn == 0.0 && np > 0.0 {
            vars[v] = true;
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
        vars[v] = rng.gen_bool(prob);
    }
    vars
}

fn rebuild_mid_state(
    nc: usize,
    co: &[u32],
    cl: &[i32],
    vars: &[bool],
    num_good: &mut [u8],
    unsat_list: &mut Vec<u32>,
    unsat_pos: &mut [u32],
) {
    num_good.fill(0);
    unsat_list.clear();
    unsat_pos.fill(u32::MAX);
    for i in 0..nc {
        let s = co[i] as usize;
        let e = co[i + 1] as usize;
        let shift = (i & 3) << 1;
        let byte_idx = i >> 2;
        let mut good = 0u8;
        for &lit in &cl[s..e] {
            let v = (lit.abs() - 1) as usize;
            if (lit > 0 && vars[v]) || (lit < 0 && !vars[v]) {
                good += 1;
            }
        }
        num_good[byte_idx] |= good.min(3) << shift;
        if good == 0 {
            unsat_pos[i] = unsat_list.len() as u32;
            unsat_list.push(i as u32);
        }
    }
}
