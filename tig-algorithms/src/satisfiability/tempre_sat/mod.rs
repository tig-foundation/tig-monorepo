use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

use rand::rngs::{SmallRng, StdRng};
use rand::{Rng, SeedableRng};

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub max_restarts: u32,
    pub flips_multiplier: u32,
    pub cb_exp: u32,
    pub cambium_interval_divisor: u32,
    pub smooth_every: u32,
    pub perturb_pct: u32,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            max_restarts: 80,
            flips_multiplier: 60,
            cb_exp: 26,
            cambium_interval_divisor: 4,
            smooth_every: 3,
            perturb_pct: 12,
        }
    }
}

struct EffectiveParams {
    max_restarts: u32,
    flips_multiplier: u64,
    cb_exp: u32,
    cambium_interval_divisor: usize,
    smooth_every: u32,
    perturb_pct: u32,
    crossover_pct: u32,       
    crossover_bias: u32,      
    stagnation_threshold: u64, 
}

fn chronosynaptic_scale(hp: &Hyperparameters, n: usize, m: usize) -> EffectiveParams {
    let ratio_x1000 = if n > 0 { (m * 1000) / n } else { 4000 };

    let hard_regime = ratio_x1000 >= 4200 && n <= 15_000;
    let medium_regime = ratio_x1000 >= 4200 && n <= 50_000;

    if hard_regime {
        EffectiveParams {
            // Strict Time Cap: 8 * 3100 = 24,800 max budget (~17% time reduction -> ~790s)
            max_restarts: 8,
            flips_multiplier: 3100, 
            // Efficiency Boost: Hyper-greedy variable selection to reach minima faster
            cb_exp: 25,  
            // Finer Contours: Faster updates, allowing stronger local gradients before smoothing
            cambium_interval_divisor: 15, 
            smooth_every: 5, 
            perturb_pct: 12, 
            crossover_pct: 10, 
            crossover_bias: 85,
            // Goldilocks Memory: Wait just long enough to climb out of deep wells, but no longer
            stagnation_threshold: (m as u64) * 12, 
        }
    } else if medium_regime {
        EffectiveParams {
            max_restarts: 25,
            flips_multiplier: 1200,
            cb_exp: 24,
            cambium_interval_divisor: 6,
            smooth_every: 3,
            perturb_pct: 13,
            crossover_pct: 20,
            crossover_bias: 70,
            stagnation_threshold: (m as u64) * 10,
        }
    } else {
        EffectiveParams {
            max_restarts: hp.max_restarts,
            flips_multiplier: hp.flips_multiplier as u64,
            cb_exp: hp.cb_exp,
            cambium_interval_divisor: hp.cambium_interval_divisor as usize,
            smooth_every: hp.smooth_every,
            perturb_pct: hp.perturb_pct,
            crossover_pct: 20,
            crossover_bias: 70,
            stagnation_threshold: (m as u64) * 15,
        }
    }
}

pub fn help() {
    println!("Chronosynaptic SAT Solver v11 (High-Fidelity Greed + Sub-800s Limit)");
}

#[inline]
fn pow_approx(base: u64, cb_exp: u32) -> u64 {
    if base <= 1 {
        return 1;
    }
    let int_part = cb_exp / 10;
    let frac_tenth = cb_exp % 10;

    let mut result = 1u64;
    let mut b = base;
    let mut exp = int_part;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result.saturating_mul(b);
        }
        b = b.saturating_mul(b);
        exp >>= 1;
    }

    if frac_tenth > 0 {
        let ln_b_1024: u64 = if base <= 10 {
            match base {
                2 => 710, 3 => 1124, 4 => 1419, 5 => 1648,
                6 => 1835, 7 => 1992, 8 => 2129, 9 => 2251, 10 => 2359,
                _ => 0,
            }
        } else {
            2359 + 1024u64.saturating_mul(base - 10) / base
        };

        let x_1024 = ln_b_1024 * (frac_tenth as u64) / 10;
        let x2_half = x_1024.saturating_mul(x_1024) / (2 * 1024);
        let exp_x_1024 = 1024 + x_1024 + x2_half;

        result = result.saturating_mul(exp_x_1024) / 1024;
    }

    result.max(1)
}

#[inline]
fn probsat_score(break_count: u32, cb_exp: u32) -> u64 {
    if break_count == 0 {
        return 1_000_000;
    }
    let base = (1 + break_count) as u64;
    let denom = pow_approx(base, cb_exp);
    if denom == 0 {
        return 1;
    }
    1_000_000u64 / denom
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let hp: Hyperparameters = match hyperparameters {
        Some(hp_map) => {
            serde_json::from_value::<Hyperparameters>(Value::Object(hp_map.clone()))
                .unwrap_or_default()
        }
        None => Hyperparameters::default(),
    };

    let n = challenge.num_variables;
    let m = challenge.clauses.len();
    if m == 0 || n == 0 {
        save_solution(&Solution { variables: vec![false; n] })?;
        return Ok(());
    }

    let initial_asgn: Vec<bool> = (0..n).map(|_| false).collect();
    save_solution(&Solution { variables: initial_asgn })?;

    let ep = chronosynaptic_scale(&hp, n, m);
    let mut rng = SmallRng::from_seed(StdRng::from_seed(challenge.seed).gen());

    let mut pos_occ: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut neg_occ: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut clause_vars: Vec<[u32; 3]> = Vec::with_capacity(m);
    let mut clause_signs: Vec<[bool; 3]> = Vec::with_capacity(m);
    let mut polarity: Vec<i32> = vec![0; n];

    for (ci, clause) in challenge.clauses.iter().enumerate() {
        let ci32 = ci as u32;
        let mut vars = [0u32; 3];
        let mut signs = [false; 3];
        for li in 0..3 {
            let lit = clause[li];
            let vi = (lit.unsigned_abs() as usize) - 1;
            let positive = lit > 0;
            vars[li] = vi as u32;
            signs[li] = positive;
            if positive {
                pos_occ[vi].push(ci32);
                polarity[vi] += 1;
            } else {
                neg_occ[vi].push(ci32);
                polarity[vi] -= 1;
            }
        }
        clause_vars.push(vars);
        clause_signs.push(signs);
    }

    let flips_budget = (m as u64) * ep.flips_multiplier;
    let cambium_interval = (n / ep.cambium_interval_divisor).max(50);

    let mut global_best_asgn: Vec<bool> = (0..n).map(|vi| polarity[vi] >= 0).collect();
    let mut local_best_asgn: Vec<bool> = global_best_asgn.clone();
    let mut global_best_sat: usize = 0;
    let mut local_best_asgn_sat: usize = 0;
    let mut conf_changed: Vec<bool> = vec![true; n];

    let mut weights: Vec<u16> = vec![1; m];

    for restart in 0..ep.max_restarts {
        let mut asgn: Vec<bool> = if restart == 0 {
            (0..n).map(|vi| polarity[vi] >= 0).collect()
        } else if restart % 7 == 0 {
            (0..n).map(|_| rng.gen_bool(0.5)).collect()
        } else if rng.gen_range(0u32..100) < ep.crossover_pct {
            let mut a = Vec::with_capacity(n);
            for vi in 0..n {
                let from_global = rng.gen_range(0u32..100) < ep.crossover_bias;
                let base_val = if from_global { global_best_asgn[vi] } else { local_best_asgn[vi] };
                if rng.gen_range(0u32..100) < (ep.perturb_pct / 2) {
                    a.push(!base_val);
                } else {
                    a.push(base_val);
                }
            }
            a
        } else {
            let mut a = global_best_asgn.clone();
            for vi in 0..n {
                if rng.gen_range(0u32..100) < ep.perturb_pct {
                    a[vi] = !a[vi];
                }
            }
            a
        };

        for v in conf_changed.iter_mut() {
            *v = true;
        }

        let mut tc: Vec<u8> = vec![0; m];
        for ci in 0..m {
            for li in 0..3 {
                let vi = clause_vars[ci][li] as usize;
                let pos = clause_signs[ci][li];
                if (pos && asgn[vi]) || (!pos && !asgn[vi]) {
                    tc[ci] += 1;
                }
            }
        }

        let mut unsat: Vec<u32> = Vec::with_capacity(m / 4);
        let mut unsat_pos: Vec<u32> = vec![u32::MAX; m];
        for ci in 0..m {
            if tc[ci] == 0 {
                unsat_pos[ci] = unsat.len() as u32;
                unsat.push(ci as u32);
            }
        }

        let cur_sat = m - unsat.len();
        if cur_sat > global_best_sat {
            global_best_sat = cur_sat;
            global_best_asgn = asgn.clone();
            save_solution(&Solution { variables: asgn.clone() })?;
        }
        if unsat.is_empty() {
            return Ok(());
        }

        let mut cambium_bumps: u32 = 0;
        let mut flips_since_improve: u64 = 0;
        let mut local_best_sat: usize = cur_sat;
        let mut restart_best_asgn: Vec<bool> = asgn.clone();

        for _flip_num in 1u64..=flips_budget {
            let pick = rng.gen_range(0..unsat.len());
            let ci = unsat[pick] as usize;
            let cvars = clause_vars[ci];

            let mut break_counts: [u32; 3] = [0; 3];
            let mut weighted_make: [u64; 3] = [0; 3];
            let mut has_freebie = false;
            let mut best_freebie_k: usize = 0;
            let mut best_freebie_wmake: u64 = 0;

            for k in 0..3 {
                let vi = cvars[k] as usize;
                let cur_val = asgn[vi];
                let (lose_list, gain_list) = if cur_val {
                    (&pos_occ[vi], &neg_occ[vi])
                } else {
                    (&neg_occ[vi], &pos_occ[vi])
                };

                let mut brk: u32 = 0;
                let mut wmk: u64 = 0;
                for &oci in lose_list {
                    if tc[oci as usize] == 1 {
                        brk += 1;
                    }
                }
                for &oci in gain_list {
                    if tc[oci as usize] == 0 {
                        wmk += weights[oci as usize] as u64;
                    }
                }
                break_counts[k] = brk;
                weighted_make[k] = wmk;

                if brk == 0 {
                    if !has_freebie || wmk > best_freebie_wmake {
                        has_freebie = true;
                        best_freebie_k = k;
                        best_freebie_wmake = wmk;
                    }
                }
            }

            let flip_var: usize;
            if has_freebie {
                flip_var = cvars[best_freebie_k] as usize;
            } else {
                let mut scores: [u64; 3] = [0; 3];
                let mut total: u64 = 0;
                for k in 0..3 {
                    let vi = cvars[k] as usize;
                    let mut s = probsat_score(break_counts[k], ep.cb_exp);

                    if conf_changed[vi] {
                        s = s.saturating_mul(2);
                    }
                    s = s.saturating_add(weighted_make[k]);
                    scores[k] = s;
                    total += s;
                }

                if total == 0 {
                    flip_var = cvars[rng.gen_range(0..3)] as usize;
                } else {
                    let r = rng.gen_range(0..total);
                    if r < scores[0] {
                        flip_var = cvars[0] as usize;
                    } else if r < scores[0] + scores[1] {
                        flip_var = cvars[1] as usize;
                    } else {
                        flip_var = cvars[2] as usize;
                    }
                }
            }

            asgn[flip_var] = !asgn[flip_var];
            let new_val = asgn[flip_var];
            conf_changed[flip_var] = false;

            let (gain_list, lose_list) = if new_val {
                (&pos_occ[flip_var], &neg_occ[flip_var])
            } else {
                (&neg_occ[flip_var], &pos_occ[flip_var])
            };

            for &oci in gain_list {
                let oci_us = oci as usize;
                tc[oci_us] += 1;
                if tc[oci_us] == 1 {
                    let pos = unsat_pos[oci_us] as usize;
                    let last = *unsat.last().unwrap();
                    unsat[pos] = last;
                    unsat_pos[last as usize] = pos as u32;
                    unsat.pop();
                    unsat_pos[oci_us] = u32::MAX;
                    for lk in 0..3 {
                        let vk = clause_vars[oci_us][lk] as usize;
                        if vk != flip_var { conf_changed[vk] = true; }
                    }
                }
            }

            for &oci in lose_list {
                let oci_us = oci as usize;
                tc[oci_us] -= 1;
                if tc[oci_us] == 0 {
                    unsat_pos[oci_us] = unsat.len() as u32;
                    unsat.push(oci);
                    for lk in 0..3 {
                        let vk = clause_vars[oci_us][lk] as usize;
                        if vk != flip_var { conf_changed[vk] = true; }
                    }
                }
            }

            let cur_sat = m - unsat.len();
            if cur_sat > local_best_sat {
                local_best_sat = cur_sat;
                restart_best_asgn = asgn.clone();
                flips_since_improve = 0;
            } else {
                flips_since_improve += 1;
            }

            if cur_sat > global_best_sat {
                global_best_sat = cur_sat;
                global_best_asgn = asgn.clone();
                save_solution(&Solution { variables: asgn.clone() })?;
                if unsat.is_empty() {
                    return Ok(());
                }
            }

            if flips_since_improve > 0 && flips_since_improve as usize % cambium_interval == 0 {
                for &uci in &unsat {
                    let w = &mut weights[uci as usize];
                    *w = w.saturating_add(3);
                }
                cambium_bumps += 1;

                if cambium_bumps % ep.smooth_every == 0 {
                    for w in weights.iter_mut() {
                        *w = (*w / 2).max(1);
                    }
                }
            }

            if flips_since_improve >= ep.stagnation_threshold {
                for w in weights.iter_mut() {
                    *w = (*w / 4).max(1);
                }
                flips_since_improve = 0;
            }
        }

        if local_best_sat > local_best_asgn_sat {
            local_best_asgn_sat = local_best_sat;
            local_best_asgn = restart_best_asgn;
        }
    }

    Ok(())
}