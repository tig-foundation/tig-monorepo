mod formula;
mod hw;
mod imp_v4_track1;
mod imp_v4_track3;
mod simd;
mod solver;
mod target_state;
mod target_track_high;
mod target_track_high_breakscore;
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

#[inline(always)]
fn lit_var_index(lit: i32) -> usize {
    if lit > 0 {
        lit as usize - 1
    } else {
        (-lit) as usize - 1
    }
}

#[inline(always)]
pub(crate) fn initial_residual_capacity(nc: usize) -> usize {
    (nc / 4).saturating_add(16)
}

#[inline(always)]
pub(crate) fn advance_interval_due(countdown: &mut usize, interval: usize) -> bool {
    debug_assert!(interval > 0);
    if interval == 0 {
        return false;
    }
    *countdown -= 1;
    if *countdown == 0 {
        *countdown = interval;
        true
    } else {
        false
    }
}

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
    pub max_fuel_high: Option<f64>,
    pub max_fuel_low: Option<f64>,
    pub target_base_prob: Option<f64>,
    pub target_nad: Option<f64>,
    pub target_greedy_init: Option<bool>,
    pub target_tail_cut_fuel: Option<f64>,
    pub target_tail_cut_unsat_threshold: Option<usize>,
    pub target_tail_cut_best_unsat_threshold: Option<usize>,
    pub target_tail_extend_fuel: Option<f64>,
    pub target_tail_extend_max_unsat: Option<usize>,
    pub target_n7500_best_tail_fuel: Option<f64>,
    pub target_n7500_best_tail_max_unsat: Option<usize>,
    pub target_n10000_best_tail_fuel: Option<f64>,
    pub target_n10000_best_tail_max_unsat: Option<usize>,
    pub target_mid_best_restart: Option<bool>,
    pub target_mid_best_restart_max_unsat: Option<usize>,
    pub target_mid_best_restart_interval: Option<usize>,
    pub target_mid_best_restart_noise_divisor: Option<usize>,
    pub target_mid_best_restart_limit: Option<usize>,
    pub target_mid_last_mile_repair: Option<bool>,
    pub target_mid_last_mile_max_unsat: Option<usize>,
    pub target_mid_last_mile_budget: Option<usize>,
    pub target_residual_compaction_factor: Option<usize>,
    pub target_residual_compaction_min_gap: Option<usize>,
    pub target_n7500_route: Option<String>,
    pub target_trace_n7500: Option<bool>,
    pub target_trace_5000: Option<bool>,
    pub target_trace_10000: Option<bool>,
    pub target_trace_4200: Option<bool>,
    pub target_trace_interval: Option<usize>,
}

pub fn help() {
    println!(
        "sat_tailwalk_v5_p46_progress_gates: progress-gated C001 best-state tails without seed-bucket route selection."
    );
    println!("Core active features:");
    println!("  - flat clause storage with u32 CSR occurrences");
    println!("  - incremental break_score and target-path make_score sidecars");
    println!("  - adaptive unsatisfied-clause weighting");
    println!("  - sparse unsat set with O(1) add/remove");
    println!(
        "  - quality-biased make-vs-break candidate scoring for dense C001 tracks
  - broad hardware profile defaults for Zen 4, Zen 5, and Zen 5c CPUs"
    );
    println!("  - experimental 100000/4200 best-state restart and last-mile repair");
    println!("  - experimental 7500/4267 lazy residual compaction");
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
    println!("  target_max_fuel, max_fuel_high, max_fuel_low, target_base_prob, target_nad, init_noise, target_greedy_init: c001 target fast-path controls; route-specific defaults");
    println!("  target_tail_cut_fuel, target_tail_cut_unsat_threshold, target_tail_cut_best_unsat_threshold: experimental mid-route tail cutoff controls");
    println!("  target_tail_extend_fuel, target_tail_extend_max_unsat: experimental 5000/4267 best-state tail continuation controls; set fuel to 0 to disable");
    println!("  target_n7500_best_tail_fuel, target_n7500_best_tail_max_unsat: experimental 7500/4267 post-portfolio best-state tail; disabled unless fuel is explicitly positive");
    println!("  target_n10000_best_tail_fuel, target_n10000_best_tail_max_unsat: experimental 10000/4267 best-state tail replacing legacy automatic budget growth; disabled unless fuel is explicitly positive");
    println!("  target_mid_best_restart, target_mid_best_restart_max_unsat, target_mid_best_restart_interval, target_mid_best_restart_noise_divisor, target_mid_best_restart_limit: experimental 100000/4200 best-state restart controls");
    println!("  target_mid_last_mile_repair, target_mid_last_mile_max_unsat, target_mid_last_mile_budget: experimental 100000/4200 near-solution post-search repair controls");
    println!("  target_residual_compaction_factor, target_residual_compaction_min_gap: experimental 7500/4267 lazy residual compaction controls; factor 0 disables");
    println!("  target_n7500_route: phase | high; diagnostic parameter probe only, default phase");
    println!("  target_trace_n7500, target_trace_5000, target_trace_10000, target_trace_4200, target_trace_interval: diagnostic only, default off; emits target-route counters");
    println!("Selective HP routing:");
    println!("  - max_fuel_high is accepted for compatibility but is not mapped into 5000/4267, 7500/4267, or 10000/4267 target_max_fuel");
    println!(
        "  - 100000/4150 keeps internal defaults unless target_max_fuel is explicitly supplied"
    );
    println!("  - 100000/4200 accepts target_max_fuel or max_fuel_low for quality-recovery probes");
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
    let hp = effective_hyperparameters(challenge, hyperparameters);

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
    let verify_invariants_enabled = hp.verify_invariants.unwrap_or(false);
    if save_if_solved(
        challenge,
        save_solution,
        &formula,
        &state,
        &vars,
        verify_invariants_enabled,
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
            verify_invariants_enabled,
        )? {
            return Ok(());
        }

        let cur_unsat = state.unsat_len();
        if cur_unsat < best_unsat {
            best_unsat = cur_unsat;
            if cur_unsat > 0 {
                best_vars.clone_from(&vars);
            }
            flips_since_best = 0;
        } else {
            flips_since_best += 1;
        }

        if flips_since_best >= restart_interval && best_unsat > 0 && restarts < max_restarts {
            restart_assignment_into(
                &formula,
                &mut rng,
                &mut vars,
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
            if verify_invariants_enabled {
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

    if verify_invariants_enabled {
        verify_invariants(&formula, &state, &vars);
    }
    if !save_if_solved(
        challenge,
        save_solution,
        &formula,
        &state,
        &vars,
        verify_invariants_enabled,
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
    anyhow::bail!(
        "sat_tailwalk_v5_p43_mid_best_restart normalized solution failed original challenge validation"
    )
}

fn effective_hyperparameters(
    challenge: &Challenge,
    hyperparameters: &Option<Map<String, Value>>,
) -> Hyperparameters {
    effective_hyperparameters_for_shape(
        challenge.num_variables,
        challenge.clauses.len(),
        hyperparameters,
    )
}

fn effective_hyperparameters_for_shape(
    num_variables: usize,
    num_clauses: usize,
    hyperparameters: &Option<Map<String, Value>>,
) -> Hyperparameters {
    let mut parsed: Hyperparameters = hyperparameters
        .as_ref()
        .and_then(|m| serde_json::from_value(Value::Object(m.clone())).ok())
        .unwrap_or_default();

    let track = track_dispatch::classify_by_shape(num_variables, num_clauses);
    if parsed.target_max_fuel.is_none() {
        parsed.target_max_fuel = match track {
            track_dispatch::C001Track::N100000R4200 => parsed.max_fuel_low,
            _ => None,
        };
    }

    if track == track_dispatch::C001Track::N100000R4150 && parsed.target_max_fuel.is_none() {
        Hyperparameters::default()
    } else {
        parsed
    }
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
            let v = lit_var_index(lit);
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
    let mut vars = Vec::with_capacity(formula.nv);
    initial_assignment_into(formula, rng, hp, &mut vars);
    vars
}

fn initial_assignment_into(
    formula: &Formula,
    rng: &mut SmallRng,
    hp: &Hyperparameters,
    vars: &mut Vec<bool>,
) {
    vars.clear();
    let density = formula.nc as f64 / formula.nv.max(1) as f64;
    let random_mixed = density >= 4.24;
    let mode = hp.init_mode.as_deref().unwrap_or("auto");
    let init_noise = hp.init_noise.unwrap_or(0.003).clamp(0.0, 1.0);
    for v in 0..formula.nv {
        let p = formula.pos_occ_len(v);
        let n = formula.neg_occ_len(v);
        let value = match mode {
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
        vars.push(value);
    }
    debug_assert_eq!(vars.len(), formula.nv);
}

fn restart_assignment_into(
    formula: &Formula,
    rng: &mut SmallRng,
    vars: &mut Vec<bool>,
    best_vars: &[bool],
    hp: &Hyperparameters,
    phase_restart_prob: f64,
    phase_noise_divisor: usize,
) {
    if formula.nv == 0 || best_vars.len() != formula.nv {
        initial_assignment_into(formula, rng, hp, vars);
        return;
    }

    if rng.gen_bool(phase_restart_prob.clamp(0.0, 1.0)) {
        if vars.len() != formula.nv {
            vars.resize(formula.nv, false);
        }
        vars.clone_from_slice(best_vars);
        let flips = (formula.nv / phase_noise_divisor.max(1) + 32).min(formula.nv);
        for _ in 0..flips {
            let v = rng.gen::<usize>() % formula.nv;
            vars[v] = !vars[v];
        }
    } else {
        initial_assignment_into(formula, rng, hp, vars);
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
        flip_var(formula, state, vars, lit_var_index(lit));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_track_wrappers_do_not_route_on_seed_buckets_or_salts() {
        for source in [
            include_str!("track_n5000_r4267.rs"),
            include_str!("track_n7500_r4267.rs"),
            include_str!("track_n100000_r4150.rs"),
            include_str!("track_n100000_r4200.rs"),
        ] {
            assert!(!source.contains("seed_key &"));
            assert!(!source.contains("seed_key ^"));
        }
    }

    #[test]
    fn active_search_modules_do_not_branch_on_seed_buckets() {
        for source in [
            include_str!("track_n10000_r4267.rs"),
            include_str!("target_track_high.rs"),
            include_str!("target_track_high_breakscore.rs"),
            include_str!("target_track_mid.rs"),
            include_str!("target_walk.rs"),
        ] {
            assert!(!source.contains("seed_key &"));
        }
    }

    #[test]
    fn fallback_lit_var_index_matches_unsigned_abs_index_for_valid_literals() {
        for lit in [-128_i32, -7, -1, 1, 7, 128] {
            assert_eq!(lit_var_index(lit), lit.unsigned_abs() as usize - 1);
        }
    }

    #[test]
    fn shared_initial_residual_capacity_is_quarter_clause_estimate() {
        assert_eq!(initial_residual_capacity(0), 16);
        assert_eq!(initial_residual_capacity(3), 16);
        assert_eq!(initial_residual_capacity(100), 41);
        assert_eq!(initial_residual_capacity(400_000), 100_016);
    }

    #[test]
    fn high_fuel_alias_does_not_expand_tail_budget_tracks() {
        let mut supplied = Map::new();
        supplied.insert("target_fast_path".to_string(), Value::Bool(false));
        supplied.insert("max_fuel_high".to_string(), Value::from(123_000_000_000.0));

        let hp = effective_hyperparameters_for_shape(5_000, 21_335, &Some(supplied));

        assert_eq!(hp.target_fast_path, Some(false));
        assert_eq!(hp.max_fuel_high, Some(123_000_000_000.0));
        assert_eq!(hp.target_max_fuel, None);
    }

    #[test]
    fn explicit_target_max_fuel_still_controls_tail_budget_tracks() {
        let mut supplied = Map::new();
        supplied.insert(
            "target_max_fuel".to_string(),
            Value::from(180_000_000_000.0),
        );
        supplied.insert("max_fuel_high".to_string(), Value::from(250_000_000_000.0));

        let hp = effective_hyperparameters_for_shape(7_500, 32_002, &Some(supplied));

        assert_eq!(hp.target_max_fuel, Some(180_000_000_000.0));
    }

    #[test]
    fn accepts_4200_low_fuel_alias_for_quality_probe() {
        let mut supplied = Map::new();
        supplied.insert("target_fast_path".to_string(), Value::Bool(false));
        supplied.insert("max_fuel_low".to_string(), Value::from(260_000_000_000.0));

        let hp = effective_hyperparameters_for_shape(100_000, 420_000, &Some(supplied));

        assert_eq!(hp.target_fast_path, Some(false));
        assert_eq!(hp.target_max_fuel, Some(260_000_000_000.0));
    }

    #[test]
    fn protects_high_4150_when_hp_has_no_quality_edge() {
        let mut supplied = Map::new();
        supplied.insert("target_fast_path".to_string(), Value::Bool(false));

        let hp = effective_hyperparameters_for_shape(100_000, 415_000, &Some(supplied));

        assert!(hp.target_fast_path.is_none());
    }

    #[test]
    fn trace_flags_are_default_off() {
        let hp = Hyperparameters::default();

        assert!(!hp.target_trace_n7500.unwrap_or(false));
        assert!(!hp.target_trace_5000.unwrap_or(false));
        assert!(!hp.target_trace_10000.unwrap_or(false));
        assert!(!hp.target_trace_4200.unwrap_or(false));
    }

    #[test]
    fn mid_best_restart_controls_are_open_hp() {
        let mut supplied = Map::new();
        supplied.insert("target_mid_best_restart".to_string(), Value::Bool(false));
        supplied.insert(
            "target_mid_best_restart_max_unsat".to_string(),
            Value::from(12usize),
        );
        supplied.insert(
            "target_mid_best_restart_interval".to_string(),
            Value::from(1_000_000usize),
        );
        supplied.insert(
            "target_mid_best_restart_noise_divisor".to_string(),
            Value::from(2048usize),
        );
        supplied.insert(
            "target_mid_best_restart_limit".to_string(),
            Value::from(5usize),
        );
        supplied.insert(
            "target_mid_last_mile_repair".to_string(),
            Value::Bool(false),
        );
        supplied.insert(
            "target_mid_last_mile_max_unsat".to_string(),
            Value::from(16usize),
        );
        supplied.insert(
            "target_mid_last_mile_budget".to_string(),
            Value::from(256usize),
        );
        supplied.insert(
            "target_residual_compaction_factor".to_string(),
            Value::from(4usize),
        );
        supplied.insert(
            "target_residual_compaction_min_gap".to_string(),
            Value::from(128usize),
        );

        let hp = effective_hyperparameters_for_shape(100_000, 420_000, &Some(supplied));

        assert_eq!(hp.target_mid_best_restart, Some(false));
        assert_eq!(hp.target_mid_best_restart_max_unsat, Some(12));
        assert_eq!(hp.target_mid_best_restart_interval, Some(1_000_000));
        assert_eq!(hp.target_mid_best_restart_noise_divisor, Some(2048));
        assert_eq!(hp.target_mid_best_restart_limit, Some(5));
        assert_eq!(hp.target_mid_last_mile_repair, Some(false));
        assert_eq!(hp.target_mid_last_mile_max_unsat, Some(16));
        assert_eq!(hp.target_mid_last_mile_budget, Some(256));
        assert_eq!(hp.target_residual_compaction_factor, Some(4));
        assert_eq!(hp.target_residual_compaction_min_gap, Some(128));
    }

    #[test]
    fn fallback_initial_assignment_direct_fill_matches_legacy_rng_path() {
        fn legacy_initial_assignment(
            formula: &Formula,
            rng: &mut SmallRng,
            hp: &Hyperparameters,
        ) -> Vec<bool> {
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

        let formula = Formula::from_raw(
            6,
            &[
                vec![1, -2, 3],
                vec![-1, 2, -3],
                vec![4, -5, 6],
                vec![-4, 5, -6],
                vec![1, 4, -6],
                vec![-2, -5, 6],
            ],
        );
        let hps = [
            Hyperparameters::default(),
            Hyperparameters {
                init_mode: Some("random".to_string()),
                ..Default::default()
            },
            Hyperparameters {
                init_mode: Some("majority".to_string()),
                init_noise: Some(0.37),
                ..Default::default()
            },
            Hyperparameters {
                init_mode: Some("occurrence".to_string()),
                ..Default::default()
            },
        ];

        for (idx, hp) in hps.iter().enumerate() {
            let seed = 0x9e37_79b9_7f4a_7c15_u64 ^ idx as u64;
            let mut direct_rng = SmallRng::seed_from_u64(seed);
            let mut legacy_rng = SmallRng::seed_from_u64(seed);
            let mut direct = Vec::with_capacity(32);
            direct.extend([true, false, true]);

            initial_assignment_into(&formula, &mut direct_rng, hp, &mut direct);
            let legacy = legacy_initial_assignment(&formula, &mut legacy_rng, hp);

            assert_eq!(direct, legacy);
            assert_eq!(direct.len(), formula.nv);
            assert_eq!(direct_rng.gen::<u64>(), legacy_rng.gen::<u64>());
        }
    }

    #[test]
    fn fallback_restart_assignment_reuses_existing_buffer_for_phase_restart() {
        let formula = Formula::from_raw(4, &[vec![1, -2, 3]]);
        let hp = Hyperparameters::default();
        let best_vars = vec![true, false, true, false];
        let mut vars = vec![false; formula.nv];
        let original_ptr = vars.as_ptr();

        let mut expected = best_vars.clone();
        let mut expected_rng = SmallRng::seed_from_u64(77);
        if expected_rng.gen_bool(1.0) {
            let flips = (formula.nv / 1 + 32).min(formula.nv);
            for _ in 0..flips {
                let v = expected_rng.gen::<usize>() % formula.nv;
                expected[v] = !expected[v];
            }
        }

        let mut rng = SmallRng::seed_from_u64(77);
        restart_assignment_into(&formula, &mut rng, &mut vars, &best_vars, &hp, 1.0, 1);

        assert_eq!(vars, expected);
        assert_eq!(vars.as_ptr(), original_ptr);
    }

    #[test]
    fn interval_countdown_matches_modulo_schedule() {
        for interval in [1usize, 2, 3, 17, 64] {
            let mut countdown = interval;
            let mut due_next = false;
            let mut due_rounds = Vec::new();
            for rounds in 0..200usize {
                if due_next {
                    due_rounds.push(rounds);
                    due_next = false;
                }
                due_next = advance_interval_due(&mut countdown, interval);
            }

            let expected: Vec<usize> = (1..200).filter(|round| round % interval == 0).collect();
            assert_eq!(due_rounds, expected, "interval {interval}");
        }
    }
}
