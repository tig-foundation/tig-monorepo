mod formula;
mod hw;
mod simd;
mod solver;
mod target_track_high;
mod target_track_low;
mod target_track_mid;
mod target_walk;
pub mod track_dispatch;
mod track_n100000_r4150;
mod track_n100000_r4200;
mod track_n10000_r4267;
mod track_n5000_r4267;
mod track_n7500_r4267;

use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

use formula::Formula;
use hw::{profile_params, select_hw_profile};
use solver::{
    bump_unsat_weights, choose_unsat_clause, choose_var_from_clause, flip_var, init_state,
    rescale_clause_weights, verify_invariants,
};

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Hyperparameters {
    pub hw_profile: Option<String>,
    pub base_prob: Option<f64>,
    pub max_prob: Option<f64>,
    pub check_interval: Option<usize>,
    pub stagnation_limit: Option<usize>,
    pub perturbation_flips: Option<usize>,
    pub restart_interval: Option<usize>,
    pub max_flips_multiplier: Option<f64>,
    pub verify_invariants: Option<bool>,
    pub disable_make_score: Option<bool>,
    pub disable_clause_weights: Option<bool>,
    pub make_mult: Option<i32>,
    pub break_mult: Option<i32>,
    pub weight_update_interval: Option<usize>,
    pub max_clause_weight: Option<u16>,
    pub clause_pick_samples: Option<usize>,
    pub phase_restart_prob: Option<f64>,
    pub phase_noise_divisor: Option<usize>,
    pub age_shift: Option<u32>,
    pub age_cap: Option<i32>,
    pub init_mode: Option<String>,
    pub init_noise: Option<f64>,
    pub target_fast_path: Option<bool>,
    pub target_max_fuel: Option<f64>,
    pub target_base_prob: Option<f64>,
    pub target_nad: Option<f64>,
    pub target_tail_cut_fuel: Option<f64>,
    pub target_tail_cut_unsat_threshold: Option<usize>,
    pub target_tail_cut_best_unsat_threshold: Option<usize>,
}

pub fn help() {
    println!("sat_tailwalk_v4: track-routed tail-aware local-search SAT solver for TIG c001.");
    println!("Core active features:");
    println!("  - flat clause storage with u32 CSR occurrences");
    println!("  - incremental make_score and break_score sidecars");
    println!("  - adaptive unsatisfied-clause weighting");
    println!("  - sparse unsat set with O(1) add/remove");
    println!("  - broad hardware profile defaults for Zen 4, Zen 5, and Zen 5c CPUs");
    println!("Hyperparameters:");
    println!("  hw_profile: auto | zen4 | zen5 | zen5c | generic_avx512 | generic");
    println!("  verify_invariants: bool, debug only, default false");
    println!("  base_prob, max_prob: generic fallback solver noise controls");
    println!("  make_mult, break_mult: weighted score coefficients");
    println!("  disable_make_score: bool, debug isolation only, default false");
    println!("  disable_clause_weights: bool, debug isolation only, default false");
    println!("  clause_pick_samples: usize 1..16; unsat clause sampling");
    println!("  weight_update_interval, max_clause_weight: adaptive clause weighting controls");
    println!("  phase_restart_prob, phase_noise_divisor: phase-saving restart controls");
    println!("  age_shift, age_cap: variable age bonus controls");
    println!("  init_mode: auto | random | occurrence | majority; default auto");
    println!("  target_fast_path: bool, default true on c001 dense 3-SAT tracks");
    println!("  target_max_fuel, target_base_prob, target_nad, init_noise: c001 target fast-path controls; route-specific defaults");
    println!("  target_tail_cut_fuel, target_tail_cut_unsat_threshold, target_tail_cut_best_unsat_threshold: experimental mid-route tail cutoff controls");
    println!("Recommended:");
    println!("  Zen 4 class: {{\"hw_profile\":\"zen4\"}}");
    println!("  Zen 5 class: {{\"hw_profile\":\"zen5\"}}");
    println!("  Zen 5c class: {{\"hw_profile\":\"zen5c\"}}");
    println!("Note: SIMD is currently a runtime label/helper only; the main flip loop is scalar.");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let hp: Hyperparameters = hyperparameters
        .as_ref()
        .and_then(|m| serde_json::from_value(Value::Object(m.clone())).ok())
        .unwrap_or_default();

    if hp.target_fast_path.unwrap_or(true) && target_walk::is_c001_target(challenge) {
        return target_walk::solve_c001_target(challenge, save_solution, &hp);
    }

    let formula = Formula::from_challenge(challenge);
    let profile = select_hw_profile(&hp);
    let params = profile_params(profile, formula.nv, formula.nc, &hp);

    let mut rng = SmallRng::from_seed(challenge.seed);
    let mut vars = initial_assignment(&formula, &mut rng, &hp);
    let _ = save_solution(&Solution {
        variables: vars.clone(),
    });

    let track_unsat_age = params.clause_pick_samples > 1;
    let mut state = init_state(&formula, &vars, track_unsat_age);
    if save_if_solved(
        challenge,
        save_solution,
        &formula,
        &state,
        &vars,
        hp.verify_invariants.unwrap_or(false),
    )? {
        return Ok(());
    }

    let large = formula.nv >= 30_000 || formula.nc >= 100_000;
    let default_multiplier = if large { 35_000.0 } else { 80_000.0 };
    let max_flips = ((formula.nv.max(1) as f64)
        * hp.max_flips_multiplier.unwrap_or(default_multiplier))
    .max((formula.nc.max(1) * 64) as f64) as usize;

    let mut current_prob = params.base_prob;
    let mut noise_threshold = prob_to_u64(current_prob);
    let mut last_unsat = state.unsat_len();
    let mut stagnation = 0usize;
    let mut best_unsat = state.unsat_len();
    let mut best_vars = vars.clone();
    let mut flips_since_best = 0usize;
    let density = formula.nc as f64 / formula.nv.max(1) as f64;
    let hard_small = formula.nv <= 10_000 && density >= 4.24;
    let default_restart_interval = if hard_small {
        100_000_000
    } else if formula.nv <= 10_000 {
        1_000_000
    } else {
        8_000_000
    };
    let restart_interval = hp
        .restart_interval
        .unwrap_or(default_restart_interval)
        .max(1);
    let max_restarts = (max_flips / restart_interval).clamp(1, 10_000);
    let mut restarts = 0usize;

    for round in 0..max_flips {
        if save_if_solved(
            challenge,
            save_solution,
            &formula,
            &state,
            &vars,
            hp.verify_invariants.unwrap_or(false),
        )? {
            return Ok(());
        }

        let cur_unsat = state.unsat_len();
        if cur_unsat < best_unsat {
            best_unsat = cur_unsat;
            best_vars.clone_from(&vars);
            flips_since_best = 0;
        } else {
            flips_since_best += 1;
        }

        if flips_since_best >= restart_interval && best_unsat > 0 && restarts < max_restarts {
            vars = restart_assignment(
                &formula,
                &mut rng,
                &best_vars,
                &hp,
                params.phase_restart_prob,
                params.phase_noise_divisor,
            );
            state = init_state(&formula, &vars, track_unsat_age);
            last_unsat = state.unsat_len();
            flips_since_best = 0;
            restarts += 1;
            continue;
        }

        if params.check_interval > 0 && round > 0 && round % params.check_interval == 0 {
            if hp.verify_invariants.unwrap_or(false) {
                verify_invariants(&formula, &state, &vars);
            }

            let now = state.unsat_len();
            if now >= last_unsat {
                stagnation += 1;
                current_prob = (current_prob * 1.35 + 0.002).min(params.max_prob);
                noise_threshold = prob_to_u64(current_prob);
                if stagnation >= params.stagnation_limit {
                    perturb(
                        &formula,
                        &mut state,
                        &mut vars,
                        &mut rng,
                        params.perturbation_flips,
                    );
                    stagnation = 0;
                }
            } else {
                stagnation = 0;
                current_prob =
                    (current_prob * 0.75 + params.base_prob * 0.25).max(params.base_prob);
                noise_threshold = prob_to_u64(current_prob);
            }
            last_unsat = now;
        }

        if params.use_clause_weights
            && params.weight_update_interval > 0
            && round > 0
            && round % params.weight_update_interval == 0
            && state.unsat_len() > 0
        {
            let should_rescale = bump_unsat_weights(&formula, &mut state, params.max_clause_weight);
            if should_rescale {
                rescale_clause_weights(&formula, &mut state, &vars);
            }
        }

        let c = choose_unsat_clause(&state, &mut rng, params.clause_pick_samples);
        let v = choose_var_from_clause(
            &formula,
            &state,
            c,
            &mut rng,
            noise_threshold,
            params.make_mult,
            params.break_mult,
            params.use_make_score,
            params.use_clause_weights,
            params.age_shift,
            params.age_cap,
        );
        flip_var(&formula, &mut state, &mut vars, v);
    }

    if hp.verify_invariants.unwrap_or(false) {
        verify_invariants(&formula, &state, &vars);
    }
    if !save_if_solved(
        challenge,
        save_solution,
        &formula,
        &state,
        &vars,
        hp.verify_invariants.unwrap_or(false),
    )? {
        let _ = save_solution(&Solution { variables: vars });
    }
    Ok(())
}

fn save_if_solved(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    formula: &Formula,
    state: &solver::State,
    vars: &[bool],
    verify_state: bool,
) -> Result<bool> {
    if state.unsat_len() != 0 {
        return Ok(false);
    }

    if satisfies_original(challenge, vars) {
        save_solution(&Solution {
            variables: vars.to_vec(),
        })?;
        return Ok(true);
    }

    if verify_state {
        verify_invariants(formula, state, vars);
    }
    anyhow::bail!("sat_tailwalk_v4 normalized solution failed original challenge validation")
}

#[inline(always)]
fn satisfies_original(challenge: &Challenge, vars: &[bool]) -> bool {
    if vars.len() != challenge.num_variables {
        return false;
    }

    challenge.clauses.iter().all(|clause| {
        clause.iter().any(|&lit| {
            if lit == 0 {
                return false;
            }
            let v = lit.unsigned_abs() as usize - 1;
            if v >= vars.len() {
                return false;
            }
            if lit > 0 {
                vars[v]
            } else {
                !vars[v]
            }
        })
    })
}

fn prob_to_u64(p: f64) -> u64 {
    let p = p.clamp(0.0, 1.0);
    (p * u64::MAX as f64) as u64
}

fn initial_assignment(formula: &Formula, rng: &mut SmallRng, hp: &Hyperparameters) -> Vec<bool> {
    let mut vars = vec![false; formula.nv];
    let density = formula.nc as f64 / formula.nv.max(1) as f64;
    let random_mixed = density >= 4.24;
    let mode = hp.init_mode.as_deref().unwrap_or("auto");
    let init_noise = hp.init_noise.unwrap_or(0.003).clamp(0.0, 1.0);
    for (v, value) in vars.iter_mut().enumerate() {
        let p = formula.pos_occ_len(v);
        let n = formula.neg_occ_len(v);
        *value = match mode {
            "random" => rng.gen_bool(0.5),
            "occurrence" | "occ" => occurrence_assignment(p, n, rng),
            "majority" => {
                if rng.gen_bool(init_noise) {
                    rng.gen_bool(0.5)
                } else {
                    p > n
                }
            }
            _ => {
                if n == 0 && p > 0 {
                    true
                } else if p == 0 {
                    false
                } else if random_mixed {
                    rng.gen_bool(0.5)
                } else {
                    occurrence_assignment(p, n, rng)
                }
            }
        };
    }
    vars
}

fn restart_assignment(
    formula: &Formula,
    rng: &mut SmallRng,
    best_vars: &[bool],
    hp: &Hyperparameters,
    phase_restart_prob: f64,
    phase_noise_divisor: usize,
) -> Vec<bool> {
    if formula.nv == 0 || best_vars.len() != formula.nv {
        return initial_assignment(formula, rng, hp);
    }

    if rng.gen_bool(phase_restart_prob.clamp(0.0, 1.0)) {
        let mut vars = best_vars.to_vec();
        let flips = (formula.nv / phase_noise_divisor.max(1) + 32).min(formula.nv);
        for _ in 0..flips {
            let v = rng.gen::<usize>() % formula.nv;
            vars[v] = !vars[v];
        }
        vars
    } else {
        initial_assignment(formula, rng, hp)
    }
}

fn occurrence_assignment(p: usize, n: usize, rng: &mut SmallRng) -> bool {
    if n == 0 && p > 0 {
        true
    } else if p == 0 {
        false
    } else {
        let prob = ((p as f64 + 0.25) / ((p + n) as f64 + 0.5)).clamp(0.02, 0.98);
        rng.gen_bool(prob)
    }
}

fn perturb(
    formula: &Formula,
    state: &mut solver::State,
    vars: &mut [bool],
    rng: &mut SmallRng,
    flips: usize,
) {
    for _ in 0..flips {
        if state.unsat_len() == 0 {
            return;
        }
        let c = choose_unsat_clause(state, rng, 1);
        let s = formula.co[c] as usize;
        let e = formula.co[c + 1] as usize;
        if s == e {
            continue;
        }
        let lit = formula.cl[s + (rng.gen::<usize>() % (e - s))];
        flip_var(formula, state, vars, lit.unsigned_abs() as usize - 1);
    }
}
