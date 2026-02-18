// TIG's UI uses the pattern `tig_challenges::satisfiability` to automatically detect your algorithm's challenge
use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub base_prob: Option<f64>,
    pub max_prob: Option<f64>,
    pub check_interval: Option<usize>,
    pub stagnation_limit: Option<usize>,
    pub perturbation_flips: Option<usize>,
}

pub fn help() {
    println!("SAT turbo - WalkSAT with dynamic probability and stagnation escape");
    println!("Hyperparameters (all optional, defaults computed from problem density):");
    println!("  base_prob: base random walk probability (default ~0.53 for density 4.15)");
    println!("  max_prob: max random walk probability during stagnation (default 0.9)");
    println!("  check_interval: flips between stagnation checks (default ~44)");
    println!("  stagnation_limit: consecutive non-progress checks before perturbation (default 2)");
    println!("  perturbation_flips: base number of random flips per perturbation (default 3)");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let hp: Option<Hyperparameters> = hyperparameters.as_ref().and_then(|m| {
        serde_json::from_value(Value::Object(m.clone())).ok()
    });

    let nv = challenge.num_variables;
    let _ = save_solution(&Solution { variables: vec![false; nv] });
    let mut rng = SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));

    let mut p_cnt = vec![0u32; nv];
    let mut n_cnt = vec![0u32; nv];
    let mut good_clauses = 0u32;

    for orig in &challenge.clauses {
        let (a, b, c) = (orig[0], orig[1], orig[2]);
        if a == -b || a == -c || b == -c { continue; }
        good_clauses += 1;
        let va = (a.abs() - 1) as usize;
        if a > 0 { p_cnt[va] += 1; } else { n_cnt[va] += 1; }
        if b != a {
            let vb = (b.abs() - 1) as usize;
            if b > 0 { p_cnt[vb] += 1; } else { n_cnt[vb] += 1; }
        }
        if c != a && c != b {
            let vc = (c.abs() - 1) as usize;
            if c > 0 { p_cnt[vc] += 1; } else { n_cnt[vc] += 1; }
        }
    }

    let nc = good_clauses as usize;

    let mut p_off = vec![0u32; nv + 1];
    let mut n_off = vec![0u32; nv + 1];
    for v in 0..nv {
        p_off[v + 1] = p_off[v] + p_cnt[v];
        n_off[v + 1] = n_off[v] + n_cnt[v];
    }
    let mut p_data = vec![0u32; p_off[nv] as usize];
    let mut n_data = vec![0u32; n_off[nv] as usize];

    {
        let mut p_pos = p_off[..nv].to_vec();
        let mut n_pos = n_off[..nv].to_vec();
        let mut ci = 0u32;

        for orig in &challenge.clauses {
            let (a, b, c) = (orig[0], orig[1], orig[2]);
            if a == -b || a == -c || b == -c { continue; }

            let va = (a.abs() - 1) as usize;
            if a > 0 { p_data[p_pos[va] as usize] = ci; p_pos[va] += 1; }
            else { n_data[n_pos[va] as usize] = ci; n_pos[va] += 1; }

            if b != a {
                let vb = (b.abs() - 1) as usize;
                if b > 0 { p_data[p_pos[vb] as usize] = ci; p_pos[vb] += 1; }
                else { n_data[n_pos[vb] as usize] = ci; n_pos[vb] += 1; }
            }
            if c != a && c != b {
                let vc = (c.abs() - 1) as usize;
                if c > 0 { p_data[p_pos[vc] as usize] = ci; p_pos[vc] += 1; }
                else { n_data[n_pos[vc] as usize] = ci; n_pos[vc] += 1; }
            }
            ci += 1;
        }
    }

    let mut cl = Vec::with_capacity(nc * 3);
    let mut co = Vec::with_capacity(nc + 1);
    co.push(0u32);

    for orig in &challenge.clauses {
        let (a, b, c) = (orig[0], orig[1], orig[2]);
        if a == -b || a == -c || b == -c { continue; }
        cl.push(a);
        if b != a { cl.push(b); }
        if c != a && c != b { cl.push(c); }
        co.push(cl.len() as u32);
    }

    let density = nc as f64 / nv as f64;
    let nvf = nv as f64;

    let mut vars = vec![false; nv];
    let nad = 1.0;
    let random_threshold = 0.003 + 0.007 / (1.0 + (-(nvf - 30000.0) / 8000.0).exp());
    let steep = 0.35 / (1.0 + (density - 4.18).max(0.0) * 12.0);
    for v in 0..nv {
        let np = p_cnt[v] as f64;
        let nn = n_cnt[v] as f64;
        if nn == 0.0 && np > 0.0 { vars[v] = true; continue; }
        if np == 0.0 { continue; }
        let vad = np / nn;
        let bias_prob = (np + 0.25) / (np + nn + 1.2);
        let s = 1.0 / (1.0 + (-(vad - nad) / steep).exp());
        let prob = (random_threshold * (1.0 - s) + bias_prob * s).max(0.0).min(1.0);
        vars[v] = rng.gen_bool(prob);
    }

    let appearances: Vec<u8> = (0..nv).map(|v| {
        ((p_cnt[v] + n_cnt[v]) as usize).min(255) as u8
    }).collect();
    drop(p_cnt);
    drop(n_cnt);

    let ng_len = (nc + 3) >> 2;
    let mut num_good = vec![0u8; ng_len];

    for i in 0..nc {
        let s = co[i] as usize;
        let e = co[i + 1] as usize;
        let shift = (i & 3) << 1;
        let byte_idx = i >> 2;
        for j in s..e {
            let l = cl[j];
            let v = (l.abs() - 1) as usize;
            if (l > 0 && vars[v]) || (l < 0 && !vars[v]) {
                num_good[byte_idx] += 1u8 << shift;
            }
        }
    }

    let mut residual: Vec<u32> = Vec::with_capacity(nc);
    for i in 0..nc {
        if (num_good[i >> 2] >> ((i & 3) << 1)) & 3 == 0 {
            residual.push(i as u32);
        }
    }

    if residual.is_empty() {
        let _ = save_solution(&Solution { variables: vars });
        return Ok(());
    }

    let base_prob = hp.as_ref().and_then(|h| h.base_prob)
        .unwrap_or(0.45 + 0.1 * (density / 5.0).min(1.0));
    let mut current_prob = base_prob;

    let large_problem_scale = ((nvf - 25000.0) / 35000.0).max(0.0).min(1.0);
    let base_interval = 60.0 - 30.0 * large_problem_scale;
    let min_interval = 25.0 - 10.0 * large_problem_scale;
    let density_s = 1.0 / (1.0 + (-(density - 4.0) / 0.5).exp());
    let density_factor = 1.0 + 0.2 * density_s;
    let check_interval = hp.as_ref().and_then(|h| h.check_interval)
        .unwrap_or((base_interval * density_factor * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize);
    let max_random_prob = hp.as_ref().and_then(|h| h.max_prob).unwrap_or(0.9);
    let prob_adjustment_factor = 0.03;
    let smoothing_factor = 0.8;
    let progress_threshold = 0.15 + 0.05 * (density / 3.0).min(1.0);

    let size_scale = 1.0 / (1.0 + (-(nvf - 30000.0) / 7000.0).exp());
    let perturbation_flips = hp.as_ref().and_then(|h| h.perturbation_flips)
        .unwrap_or(1 + (2.0 * size_scale) as usize);
    let stagnation_limit = hp.as_ref().and_then(|h| h.stagnation_limit)
        .unwrap_or(2 + (2.0 * (1.0 - (density / 5.0).min(1.0))) as usize);

    let mut last_check_residual = residual.len();
    let mut stagnation = 0usize;
    let mut var_age = vec![0u8; nv];
    let mut countdown = check_interval;

    unsafe {
        loop {
            if residual.is_empty() { break; }

            countdown -= 1;
            if countdown == 0 {
                countdown = check_interval;
                let progress = last_check_residual as i64 - residual.len() as i64;
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
                            if residual.is_empty() { break; }
                            let rid = rng.gen::<usize>() % residual.len();
                            let pcid = *residual.get_unchecked(rid) as usize;
                            let ng_val = (*num_good.get_unchecked(pcid >> 2) >> ((pcid & 3) << 1)) & 3;
                            if ng_val > 0 {
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
                                (*n_off.get_unchecked(v), *n_off.get_unchecked(v + 1))
                            } else {
                                (*p_off.get_unchecked(v), *p_off.get_unchecked(v + 1))
                            };
                            let ia = if was_true { &n_data } else { &p_data };
                            let (ds, de) = if was_true {
                                (*p_off.get_unchecked(v), *p_off.get_unchecked(v + 1))
                            } else {
                                (*n_off.get_unchecked(v), *n_off.get_unchecked(v + 1))
                            };
                            let da = if was_true { &p_data } else { &n_data };

                            for k in is..ie {
                                let c = *ia.get_unchecked(k as usize) as usize;
                                *num_good.get_unchecked_mut(c >> 2) += 1u8 << ((c & 3) << 1);
                            }
                            for k in ds..de {
                                let c = *da.get_unchecked(k as usize) as usize;
                                let shift = (c & 3) << 1;
                                let byte_idx = c >> 2;
                                let ng_before = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
                                *num_good.get_unchecked_mut(byte_idx) -= 1u8 << shift;
                                if ng_before == 1 {
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
                    current_prob = current_prob * smoothing_factor + base_prob * (1.0 - smoothing_factor);
                }

                last_check_residual = residual.len();
            }

            let rand_val = rng.gen::<usize>();
            let mut cid = 0usize;
            let mut found = false;
            while !residual.is_empty() {
                let id = rand_val % residual.len();
                let candidate = *residual.get_unchecked(id) as usize;
                let ng_val = (*num_good.get_unchecked(candidate >> 2) >> ((candidate & 3) << 1)) & 3;
                if ng_val > 0 {
                    residual.swap_remove(id);
                } else {
                    cid = candidate;
                    found = true;
                    break;
                }
            }
            if !found { break; }

            let cs = *co.get_unchecked(cid) as usize;
            let ce = *co.get_unchecked(cid + 1) as usize;
            let clen = ce - cs;

            if clen > 1 {
                let ri = rand_val % clen;
                cl.swap(cs, cs + ri);
            }

            let mut zero_found = None;
            'outer: for j in cs..ce {
                let l = *cl.get_unchecked(j);
                let abs_l = (l.abs() - 1) as usize;
                let (os, oe) = if l > 0 {
                    (*n_off.get_unchecked(abs_l), *n_off.get_unchecked(abs_l + 1))
                } else {
                    (*p_off.get_unchecked(abs_l), *p_off.get_unchecked(abs_l + 1))
                };
                let arr = if l > 0 { &n_data } else { &p_data };
                for k in os..oe {
                    let c = *arr.get_unchecked(k as usize) as usize;
                    if (*num_good.get_unchecked(c >> 2) >> ((c & 3) << 1)) & 3 == 1 {
                        continue 'outer;
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
                    let (os, oe) = if l > 0 {
                        (*n_off.get_unchecked(abs_l), *n_off.get_unchecked(abs_l + 1))
                    } else {
                        (*p_off.get_unchecked(abs_l), *p_off.get_unchecked(abs_l + 1))
                    };
                    let arr = if l > 0 { &n_data } else { &p_data };
                    let mut sad = 0usize;
                    for k in os..oe {
                        let c = *arr.get_unchecked(k as usize) as usize;
                        if (*num_good.get_unchecked(c >> 2) >> ((c & 3) << 1)) & 3 == 1 {
                            sad += 1;
                        }
                        if sad >= min_sad { break; }
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
                        let combined_weight = sad * sad * 256 + app - age_bonus.min(50);
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
            let (is, ie, ia) = if was_true {
                (*n_off.get_unchecked(v_idx), *n_off.get_unchecked(v_idx + 1), &n_data)
            } else {
                (*p_off.get_unchecked(v_idx), *p_off.get_unchecked(v_idx + 1), &p_data)
            };
            let (ds, de, da) = if was_true {
                (*p_off.get_unchecked(v_idx), *p_off.get_unchecked(v_idx + 1), &p_data)
            } else {
                (*n_off.get_unchecked(v_idx), *n_off.get_unchecked(v_idx + 1), &n_data)
            };
            for k in is..ie {
                let c = *ia.get_unchecked(k as usize) as usize;
                *num_good.get_unchecked_mut(c >> 2) += 1u8 << ((c & 3) << 1);
            }
            for k in ds..de {
                let c = *da.get_unchecked(k as usize) as usize;
                let shift = (c & 3) << 1;
                let byte_idx = c >> 2;
                let ng_before = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
                *num_good.get_unchecked_mut(byte_idx) -= 1u8 << shift;
                if ng_before == 1 {
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
        }
    }

    let _ = save_solution(&Solution { variables: vars });
    Ok(())
}
