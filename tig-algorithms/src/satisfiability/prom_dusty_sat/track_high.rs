use anyhow::Result;
use rand::{rngs::SmallRng, Rng};
use super::*;
use super::Hyperparameters;

pub fn solve(
    hp: &Option<Hyperparameters>,
    rng: &mut SmallRng,
    nv: usize, nc: usize, density: f64,
    p_cnt: Vec<u32>, n_cnt: Vec<u32>,
    all_off: &[u32], p_bound: &[u32],
    all_data: &[u32],
    cl: &mut Vec<i32>, co: &[u32],
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    // ... (code remains the same until the initialization of variables)

    let mut vars = vec![false; nv];
    for v in 0..nv {
        let np = p_cnt[v] as usize;
        let nn = n_cnt[v] as usize;
        if nn == 0 && np > 0 { vars[v] = true; continue; }
        if np == 0 && nn > 0 { continue; }
        let vad = if nn > 0 { np as f64 / nn as f64 } else { 1.0 + 1.0 };
        if vad <= 1.0f64 {
            vars[v] = rng.gen_bool(0.5);
        } else {
            let prob = (np as f64 + 0.25) / ((np + nn) as f64 + 1.2);
            vars[v] = rng.gen_bool(prob);
        }
    }

    let mut survey_bias = vec![0.0f64; nv];
    let mut clause_weights = vec![0.0f64; nc];
    let mut var_surveys = vec![vec![0.0f64; 3]; nv]; 

    for i in 0..nc {
        let s = co[i] as usize;
        let e = co[i + 1] as usize;
        let mut pos_count = 0;
        let mut neg_count = 0;
        for j in s..e {
            let l = cl[j];
            let v = (l.abs() - 1) as usize;
            if l > 0 {
                pos_count += 1;
                var_surveys[v][0] += 1.0f64; 
            } else {
                neg_count += 1;
                var_surveys[v][1] += 1.0f64; 
            }
        }
        clause_weights[i] = 1.0f64 / (pos_count as f64 + neg_count as f64);
        for j in s..e {
            let l = cl[j];
            let v = (l.abs() - 1) as usize;
            var_surveys[v][2] += clause_weights[i]; 
        }
    }

    for v in 0..nv {
        let pos_survey = var_surveys[v][0];
        let neg_survey = var_surveys[v][1];
        let total_survey = var_surveys[v][2];
        survey_bias[v] = (pos_survey - neg_survey) / total_survey.max(1.0f64);
    }

    let mut iterations = 0;
    let max_iterations = 1000; 
    let damping_factor = 0.9f64; 

    while iterations < max_iterations {
        for i in 0..nc {
            let s = co[i] as usize;
            let e = co[i + 1] as usize;
            let mut prod_pos = 1.0f64;
            let mut prod_neg = 1.0f64;
            for j in s..e {
                let l = cl[j];
                let v = (l.abs() - 1) as usize;
                if l > 0 {
                    prod_pos *= 1.0f64 - survey_bias[v].max(0.0f64).min(1.0f64);
                } else {
                    prod_neg *= 1.0f64 + survey_bias[v].max(-1.0f64).min(1.0f64);
                }
            }
            clause_weights[i] = 1.0f64 / (1.0f64 - prod_pos * prod_neg).max(1e-10f64);
        }

        for v in 0..nv {
            let mut new_pos_survey = 0.0f64;
            let mut new_neg_survey = 0.0f64;
            let mut new_total_survey = 0.0f64;
            for k in all_off[v] as usize..p_bound[v] as usize {
                let c = all_data[k] as usize;
                new_pos_survey += clause_weights[c];
                new_total_survey += clause_weights[c];
            }
            for k in p_bound[v] as usize..all_off[v + 1] as usize {
                let c = all_data[k] as usize;
                new_neg_survey += clause_weights[c];
                new_total_survey += clause_weights[c];
            }
            var_surveys[v][0] = damping_factor * var_surveys[v][0] + (1.0f64 - damping_factor) * new_pos_survey;
            var_surveys[v][1] = damping_factor * var_surveys[v][1] + (1.0f64 - damping_factor) * new_neg_survey;
            var_surveys[v][2] = damping_factor * var_surveys[v][2] + (1.0f64 - damping_factor) * new_total_survey;

            let pos_survey = var_surveys[v][0];
            let neg_survey = var_surveys[v][1];
            let total_survey = var_surveys[v][2];
            survey_bias[v] = (pos_survey - neg_survey) / total_survey.max(1.0f64);
        }

        iterations += 1;
    }

    for v in 0..nv {
        if survey_bias[v] > 0.0f64 {
            vars[v] = true;
        } else if survey_bias[v] < 0.0f64 {
            vars[v] = false;
        } else {
            vars[v] = rng.gen_bool(0.5);
        }
    }

    let var_appearances: Vec<usize> = (0..nv)
        .map(|v| (p_cnt[v] + n_cnt[v]) as usize)
        .collect();

    let avg_clause_size = cl.len() as f64 / nc as f64;
    let difficulty_factor = density * avg_clause_size.sqrt();
    let scale_factor = if nv > 25000 { 1.5f64 } else { 1.0f64 };
    let base_fuel = (2000.0f64 + 100.0f64 * difficulty_factor) * (nv as f64).sqrt() * scale_factor;
    let flip_fuel = (200.0f64 + difficulty_factor) / scale_factor;
    let default_fuel = if nv >= 10000 { 125_000_000_000.0f64 } else { 250_000_000_000.0f64 };
    let max_fuel = hp.as_ref().and_then(|h| h.max_fuel_high).unwrap_or(default_fuel);

    let remaining = (max_fuel - base_fuel).max(0.0f64);
    let max_flips = if flip_fuel > 0.0f64 { (remaining / flip_fuel) as usize } else { 0 };

    let nad = 1.0f64;
    let random_threshold = if nv >= 30000 { 0.01f64 } else { 0.003f64 };
    let mut num_good = vec![0u8; nc];
    for i in 0..nc {
        let s = co[i] as usize;
        let e = co[i + 1] as usize;
        for j in s..e {
            let l = cl[j];
            let v = (l.abs() - 1) as usize;
            if (l > 0 && vars[v]) || (l < 0 && !vars[v]) {
                num_good[i] += 1;
            }
        }
    }

    let mut residual: Vec<u32> = Vec::with_capacity(nc);
    for i in 0..nc {
        if num_good[i] == 0 {
            residual.push(i as u32);
        }
    }

    if residual.is_empty() {
        let _ = save_solution(&Solution { variables: vars });
        return Ok(());
    }

    let base_prob: f64 = hp.as_ref().and_then(|h| h.base_prob).unwrap_or(0.52);
    let mut current_prob = base_prob;
    let max_random_prob: f64 = hp.as_ref().and_then(|h| h.max_prob).unwrap_or(0.9);
    let prob_adjustment_factor: f64 = 0.025;
    let smoothing_factor: f64 = 0.8;

    let large_problem_scale = ((nv as f64 - 25000.0f64) / 35000.0f64).max(0.0f64).min(1.0f64);
    let base_interval = 60.0f64 - 30.0f64 * large_problem_scale;
    let min_interval = if large_problem_scale > 0.0f64 { 15.0f64 } else { 25.0f64 };
    let density_factor_ci = if density > 4.0f64 { 1.2f64 } else { 1.0f64 };
    let check_interval = hp.as_ref().and_then(|h| h.check_interval)
        .unwrap_or((base_interval * density_factor_ci * (1.0f64 + (density / 3.0f64).ln().max(0.0f64))).max(min_interval) as usize);

    let mut last_check_residual = residual.len();
    let mut var_age = vec![0u16; nv];
    let mut rounds = 0usize;
    let mut stagnation = 0usize;

    let _probs_break: [u32; 16] = [2535, 551, 233, 127, 80, 55, 41, 30, 24, 19, 16, 13, 11, 9, 8, 7];

    unsafe {
        loop {
            if rounds >= max_flips { break; }
            if residual.is_empty() { break; }

            if rounds % check_interval == 0 && rounds > 0 {
                let progress = last_check_residual as i64 - residual.len() as i64;
                let progress_ratio = progress as f64 / last_check_residual.max(1) as f64;
                let progress_threshold = 0.15f64 + 0.05f64 * (density / 3.0f64).min(1.0f64);

                if progress <= 0 {
                    stagnation += 1;
                    let prob_adjustment = prob_adjustment_factor * (-progress as f64 / last_check_residual.max(1) as f64).min(1.0f64);
                    current_prob = (current_prob + prob_adjustment).min(max_random_prob);

                    if stagnation >= 4 {
                        let kicks = if stagnation >= 8 { 6 } else { 3 };
                        for _ in 0..kicks {
                            if residual.is_empty() { break; }
                            let rid = rng.gen::<usize>() % residual.len();
                            let pcid = *residual.get_unchecked(rid) as usize;
                            if *num_good.get_unchecked(pcid) > 0 {
                                residual.swap_remove(rid);
                                continue;
                            }
                            let pcs = *co.get_unchecked(pcid) as usize;
                            let pce = *co.get_unchecked(pcid + 1) as usize;
                            if pcs == pce { continue; }
                            let lit = *cl.get_unchecked(pcs + rng.gen::<usize>() % (pce - pcs));
                            let v = (lit.abs() - 1) as usize;

                            let was_true = *vars.get_unchecked(v);
                            let (is, ie) = if was_true {
                                (*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
                            } else {
                                (*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
                            };
                            let (ds, de) = if was_true {
                                (*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
                            } else {
                                (*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
                            };

                            for k in is..ie {
                                let c = *all_data.get_unchecked(k) as usize;
                                *num_good.get_unchecked_mut(c) = num_good.get_unchecked(c).saturating_add(1);
                            }
                            for k in ds..de {
                                let c = *all_data.get_unchecked(k) as usize;
                                let ng = num_good.get_unchecked_mut(c);
                                let new_val = ng.saturating_sub(1);
                                *ng = new_val;
                                if new_val == 0 {
                                    residual.push(c as u32);
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
                    current_prob = current_prob * smoothing_factor + base_prob * (1.0f64 - smoothing_factor);
                }

                last_check_residual = residual.len();
            }

            if residual.is_empty() { break; }

            let rand_val = rng.gen::<usize>();
            let mut cid = 0usize;
            while !residual.is_empty() {
                let id = rand_val % residual.len();
                cid = *residual.get_unchecked(id) as usize;
                if *num_good.get_unchecked(cid) > 0 {
                    residual.swap_remove(id);
                } else {
                    break;
                }
            }
            if residual.is_empty() { break; }

            let cs = *co.get_unchecked(cid) as usize;
            let ce = *co.get_unchecked(cid + 1) as usize;
            let clen = ce - cs;

            if clen > 1 {
                let ri = rand_val % clen;
                cl.swap(cs, cs + ri);
            }

            let mut zero_found: Option<usize> = None;
            'outer_h: for j in cs..ce {
                let l = *cl.get_unchecked(j);
                let abs_l = (l.abs() - 1) as usize;
                let (os, oe) = if *vars.get_unchecked(abs_l) {
                    (*all_off.get_unchecked(abs_l) as usize, *p_bound.get_unchecked(abs_l) as usize)
                } else {
                    (*p_bound.get_unchecked(abs_l) as usize, *all_off.get_unchecked(abs_l + 1) as usize)
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

            let v_idx = if let Some(v) = zero_found {
                v
            } else if rng.gen::<f64>() < current_prob {
                (cl.get_unchecked(cs).abs() - 1) as usize
            } else {
                let mut min_sad = usize::MAX;
                let mut v_min = (cl.get_unchecked(cs).abs() - 1) as usize;
                let mut min_weight = usize::MAX;

                for j in cs..ce {
                    let l = *cl.get_unchecked(j);
                    let abs_l = (l.abs() - 1) as usize;
                    let (os, oe) = if *vars.get_unchecked(abs_l) {
                        (*all_off.get_unchecked(abs_l) as usize, *p_bound.get_unchecked(abs_l) as usize)
                    } else {
                        (*p_bound.get_unchecked(abs_l) as usize, *all_off.get_unchecked(abs_l + 1) as usize)
                    };

                    let mut sad = 0usize;
                    for k in os..oe {
                        let c = *all_data.get_unchecked(k) as usize;
                        if *num_good.get_unchecked(c) == 1 {
                            sad += 1;
                        }
                        if sad >= min_sad { break; }
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
                        if min_sad <= 1 { break; }
                    }
                }
                v_min
            };

            let was_true = *vars.get_unchecked(v_idx);
            let (is, ie) = if was_true {
                (*p_bound.get_unchecked(v_idx) as usize, *all_off.get_unchecked(v_idx + 1) as usize)
            } else {
                (*all_off.get_unchecked(v_idx) as usize, *p_bound.get_unchecked(v_idx) as usize)
            };
            let (ds, de) = if was_true {
                (*all_off.get_unchecked(v_idx) as usize, *p_bound.get_unchecked(v_idx) as usize)
            } else {
                (*p_bound.get_unchecked(v_idx) as usize, *all_off.get_unchecked(v_idx + 1) as usize)
            };

            for k in is..ie {
                let c = *all_data.get_unchecked(k) as usize;
                *num_good.get_unchecked_mut(c) = num_good.get_unchecked(c).saturating_add(1);
            }
            for k in ds..de {
                let c = *all_data.get_unchecked(k) as usize;
                let ng = num_good.get_unchecked_mut(c);
                let new_val = ng.saturating_sub(1);
                *ng = new_val;
                if new_val == 0 {
                    residual.push(c as u32);
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

    let _ = save_solution(&Solution { variables: vars });
    Ok(())
}