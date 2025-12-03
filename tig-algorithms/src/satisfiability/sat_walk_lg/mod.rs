// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use crate::{seeded_hasher, HashMap, HashSet};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::convert::TryInto;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
            pub max_flips: usize,
            pub early_limit: usize,
            pub noise: f64,
            pub early_factor: f64,
}

pub fn help() {
    println!("sat_walk_lg: Best hyperparameters to use per track:");
    println!("  track: num_variables=5000,clauses_to_variables_ratio=4267, best hyperparameters: null");
    println!("  track: num_variables=7500,clauses_to_variables_ratio=4267, best hyperparameters: '{{\"max_flips\":75000000, \"early_limit\":300000}}'");
    println!("  track: num_variables=10000,clauses_to_variables_ratio=4267, best hyperparameters: '{{\"max_flips\":100000000, \"early_limit\":400000}}'");
    println!("  track: num_variables=100000,clauses_to_variables_ratio=4100, best hyperparameters: '{{\"max_flips\":120000000, \"early_limit\":4000000, \"noise\":0.50, \"early_factor\":0.9}}'");
    println!("  track: num_variables=100000,clauses_to_variables_ratio=4150, best hyperparameters: '{{\"max_flips\":150000000, \"early_limit\":4000000, \"noise\":0.52, \"early_factor\":0.9}}'");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let seed = u64::from_le_bytes(challenge.seed[..8].try_into().unwrap());
    let mut rng = SmallRng::seed_from_u64(seed);

    let density = if challenge.num_variables > 0 {
        challenge.clauses.len() as f64 / challenge.num_variables as f64
    } else {
        0.0
    };

    if density >= 4.3 {
        solve_with_walksat_ga(challenge, &mut rng, save_solution, hyperparameters)
    } else {
        solve_with_walksat_vav(challenge, &mut rng, save_solution, hyperparameters)
    }
}

/* ------------------------------------------------------------------------- */
/* Core data structures                                                      */
/* ------------------------------------------------------------------------- */

#[derive(Clone)]
struct SatInstance {
    num_variables: usize,
    clauses: Vec<Vec<i32>>,
}

#[inline]
fn extract_usize(v: &Value) -> Option<usize> {
    if let Some(u) = v.as_u64() {
        Some(u as usize)
    } else if let Some(f) = v.as_f64() {
        if f.is_finite() && f >= 0.0 {
            Some(f as usize)
        } else {
            None
        }
    } else {
        None
    }
}

fn get_max_flips(
    hyperparameters: &Option<Map<String, Value>>,
    default: usize,
) -> usize {
    hyperparameters
        .as_ref()
        .and_then(|h| {
            h.get("max_flips")
                .or_else(|| h.get("max_fuel"))
                .or_else(|| h.get("max_fuel_high"))
                .or_else(|| h.get("max_fuel_low"))
        })
        .and_then(extract_usize)
        .unwrap_or(default.max(1))
}

fn get_early_limit(
    hyperparameters: &Option<Map<String, Value>>,
    default: usize,
) -> usize {
    hyperparameters
        .as_ref()
        .and_then(|h| h.get("early_limit"))
        .and_then(extract_usize)
        .unwrap_or(default)
}

fn get_noise(
    hyperparameters: &Option<Map<String, Value>>,
    default: f64,
) -> f64 {
    hyperparameters
        .as_ref()
        .and_then(|h| h.get("noise"))
        .and_then(|v| v.as_f64())
        .unwrap_or(default)
}

fn get_early_factor(
    hyperparameters: &Option<Map<String, Value>>,
    default: f64,
) -> f64 {
    hyperparameters
        .as_ref()
        .and_then(|h| h.get("early_factor"))
        .and_then(|v| v.as_f64())
        .unwrap_or(default)
}

/* ------------------------------------------------------------------------- */
/* Preprocessing: tautology removal + iterative unit propagation             */
/* ------------------------------------------------------------------------- */

/// Returns (simplified instance, fixed assignments, contradiction_flag)
fn preprocess_instance(
    challenge: &Challenge,
) -> (SatInstance, Vec<Option<bool>>, bool) {
    let num_variables = challenge.num_variables;
    let mut clauses = challenge.clauses.clone();

    let hasher = seeded_hasher(&challenge.seed);

    // Remove tautological clauses and duplicate literals.
    let mut i = 0;
    while i < clauses.len() {
        let clause = &mut clauses[i];

        if clause.len() > 1 {
            let mut seen = HashSet::with_hasher(hasher.clone());
            let mut j = 0;
            let mut tautology = false;

            while j < clause.len() {
                let lit = clause[j];
                if seen.contains(&-lit) {
                    tautology = true;
                    break;
                }
                if !seen.insert(lit) {
                    // duplicate literal -> remove it
                    clause.swap_remove(j);
                } else {
                    j += 1;
                }
            }

            if tautology {
                clauses.swap_remove(i);
                continue;
            }
        }

        i += 1;
    }

    let mut fixed = vec![None; num_variables];

    // Standard iterative unit propagation, keeping indices in original space.
    loop {
        let mut changed = false;
        let mut new_clauses: Vec<Vec<i32>> = Vec::with_capacity(clauses.len());

        for clause in clauses.into_iter() {
            let mut new_clause: Vec<i32> = Vec::with_capacity(clause.len());
            let mut satisfied = false;

            for lit in clause {
                let var_idx = (lit.abs() as usize).saturating_sub(1);
                if var_idx >= num_variables {
                    continue;
                }

                if let Some(val) = fixed[var_idx] {
                    // Literal already fixed.
                    if (lit > 0 && val) || (lit < 0 && !val) {
                        // Clause already satisfied.
                        satisfied = true;
                        break;
                    } else {
                        // Literal false, drop it.
                        continue;
                    }
                } else {
                    new_clause.push(lit);
                }
            }

            if satisfied {
                // Drop satisfied clause.
                continue;
            }

            match new_clause.len() {
                0 => {
                    // Empty clause -> contradiction discovered.
                    return (
                        SatInstance {
                            num_variables,
                            clauses: Vec::new(),
                        },
                        fixed,
                        true,
                    );
                }
                1 => {
                    // New unit clause.
                    let lit = new_clause[0];
                    let var_idx = (lit.abs() as usize).saturating_sub(1);
                    if var_idx >= num_variables {
                        continue;
                    }
                    let val = lit > 0;
                    match fixed[var_idx] {
                        Some(prev) if prev != val => {
                            // Conflict on unit propagation.
                            return (
                                SatInstance {
                                    num_variables,
                                    clauses: Vec::new(),
                                },
                                fixed,
                                true,
                            );
                        }
                        Some(_) => { /* already consistent */ }
                        None => {
                            fixed[var_idx] = Some(val);
                            changed = true;
                        }
                    }
                }
                _ => {
                    new_clauses.push(new_clause);
                }
            }
        }

        clauses = new_clauses;

        if !changed {
            break;
        }
    }

    (
        SatInstance {
            num_variables,
            clauses,
        },
        fixed,
        false,
    )
}

/* ------------------------------------------------------------------------- */
/* Utility helpers                                                           */
/* ------------------------------------------------------------------------- */

fn build_occurrence_lists(
    instance: &SatInstance,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut pos = vec![Vec::<usize>::new(); instance.num_variables];
    let mut neg = vec![Vec::<usize>::new(); instance.num_variables];

    for (cid, clause) in instance.clauses.iter().enumerate() {
        for &lit in clause {
            let var = (lit.abs() as usize).saturating_sub(1);
            if var >= instance.num_variables {
                continue;
            }
            if lit > 0 {
                pos[var].push(cid);
            } else {
                neg[var].push(cid);
            }
        }
    }

    (pos, neg)
}

fn compute_literal_counts(
    instance: &SatInstance,
) -> (Vec<usize>, Vec<usize>) {
    let mut pos = vec![0usize; instance.num_variables];
    let mut neg = vec![0usize; instance.num_variables];

    for clause in &instance.clauses {
        for &lit in clause {
            let var = (lit.abs() as usize).saturating_sub(1);
            if var >= instance.num_variables {
                continue;
            }
            if lit > 0 {
                pos[var] += 1;
            } else {
                neg[var] += 1;
            }
        }
    }

    (pos, neg)
}

/// Count the number of unsatisfied clauses under `assignment`.
fn count_unsat(instance: &SatInstance, assignment: &[bool]) -> usize {
    let mut unsat = 0usize;
    for clause in &instance.clauses {
        let mut satisfied = false;
        for &lit in clause {
            let var_idx = (lit.abs() as usize).saturating_sub(1);
            if var_idx >= assignment.len() {
                continue;
            }
            let val = assignment[var_idx];
            if (lit > 0 && val) || (lit < 0 && !val) {
                satisfied = true;
                break;
            }
        }
        if !satisfied {
            unsat += 1;
        }
    }
    unsat
}

/// Fill in a full assignment, respecting fixed units and optionally
/// a "base" assignment for the free variables. Any still-unset bits
/// are chosen randomly.
fn finalize_assignment(
    num_variables: usize,
    fixed: &[Option<bool>],
    base: Option<&[bool]>,
    rng: &mut SmallRng,
) -> Vec<bool> {
    let mut vars = vec![false; num_variables];

    for v in 0..num_variables {
        if v < fixed.len() {
            if let Some(val) = fixed[v] {
                vars[v] = val;
                continue;
            }
        }
        if let Some(base_ass) = base {
            if v < base_ass.len() {
                vars[v] = base_ass[v];
                continue;
            }
        }
        vars[v] = rng.gen();
    }

    vars
}

/* ------------------------------------------------------------------------- */
/* Allocation strategy (Vad / Vav)                                           */
/* ------------------------------------------------------------------------- */

/// Generate an initial assignment using the allocation strategy Vav.
/// This captures the spirit of WalkSATvav. :contentReference[oaicite:1]{index=1}
fn random_assignment_allocation(
    instance: &SatInstance,
    fixed: &[Option<bool>],
    pos_counts: &[usize],
    neg_counts: &[usize],
    rng: &mut SmallRng,
    pad: f64,
    nad: f64,
) -> Vec<bool> {
    let n = instance.num_variables;
    let mut assignment = vec![false; n];

    for v in 0..n {
        if v < fixed.len() {
            if let Some(val) = fixed[v] {
                assignment[v] = val;
                continue;
            }
        }

        let p = *pos_counts.get(v).unwrap_or(&0) as f64;
        let nneg = *neg_counts.get(v).unwrap_or(&0) as f64;

        if p == 0.0 && nneg == 0.0 {
            // Completely unused variable -> pure random.
            assignment[v] = rng.gen();
            continue;
        }

        // Variable allocation degree: bias by ratio of positive to negative
        // occurrences, with +1 smoothing to avoid division by zero.
        let vad = (p + 1.0) / (nneg + 1.0);

        // Variable allocation value (Vav): if vad stronger than thresholds,
        // fix to 1 or 0; otherwise random, following Def.2 in the paper. :contentReference[oaicite:2]{index=2}
        if vad >= pad {
            assignment[v] = true;
        } else if vad <= nad {
            assignment[v] = false;
        } else {
            assignment[v] = rng.gen();
        }
    }

    assignment
}

/* ------------------------------------------------------------------------- */
/* GA for WalkSATga                                                          */
/* ------------------------------------------------------------------------- */

fn tournament_select(
    fitness: &[usize],
    rng: &mut SmallRng,
) -> usize {
    let len = fitness.len();
    let k = 3.min(len);
    let mut best_idx = rng.gen_range(0..len);
    let mut best_fit = fitness[best_idx];

    for _ in 1..k {
        let idx = rng.gen_range(0..len);
        if fitness[idx] > best_fit {
            best_fit = fitness[idx];
            best_idx = idx;
        }
    }

    best_idx
}

fn crossover(
    p1: &[bool],
    p2: &[bool],
    fixed: &[Option<bool>],
    rng: &mut SmallRng,
) -> Vec<bool> {
    let n = p1.len().min(p2.len()).min(fixed.len());
    let mut child = vec![false; n];

    for i in 0..n {
        if let Some(val) = fixed[i] {
            child[i] = val;
        } else {
            child[i] = if rng.gen_bool(0.5) { p1[i] } else { p2[i] };
        }
    }

    child
}

fn mutate(
    indiv: &mut [bool],
    fixed: &[Option<bool>],
    mutation_rate: f64,
    rng: &mut SmallRng,
) {
    for (i, bit) in indiv.iter_mut().enumerate() {
        if i < fixed.len() && fixed[i].is_some() {
            continue;
        }
        if rng.gen::<f64>() < mutation_rate {
            *bit = !*bit;
        }
    }
}

/// GA component of WalkSATga: returns (best_assignment, best_unsat, solved?)
fn run_ga(
    instance: &SatInstance,
    fixed: &[Option<bool>],
    pos_counts: &[usize],
    neg_counts: &[usize],
    rng: &mut SmallRng,
) -> (Vec<bool>, usize, bool) {
    let n = instance.num_variables;
    if n == 0 || instance.clauses.is_empty() {
        let assignment = finalize_assignment(n, fixed, None, rng);
        return (assignment, 0, true);
    }

    let nv = n as f64;
    let pop_size = if nv <= 1_000.0 {
        40
    } else if nv <= 10_000.0 {
        25
    } else {
        15
    };

    let max_generations = if nv <= 1_000.0 {
        30
    } else if nv <= 10_000.0 {
        20
    } else {
        10
    };

    let mutation_rate = (1.0 / nv.max(32.0)).min(0.2);

    // Parameters from the allocation strategy (best empirically in the paper). :contentReference[oaicite:3]{index=3}
    let pad = 1.8;
    let nad = 0.56;

    let mut population: Vec<Vec<bool>> = Vec::with_capacity(pop_size);
    let mut fitness: Vec<usize> = vec![0; pop_size];

    // Initial population using allocation-biased assignments.
    for i in 0..pop_size {
        let indiv = random_assignment_allocation(instance, fixed, pos_counts, neg_counts, rng, pad, nad);
        let unsat = count_unsat(instance, &indiv);
        let fit = instance.clauses.len().saturating_sub(unsat);
        population.push(indiv);
        fitness[i] = fit;
    }

    let mut best_idx = 0usize;
    for i in 1..pop_size {
        if fitness[i] > fitness[best_idx] {
            best_idx = i;
        }
    }

    let mut best_assignment = population[best_idx].clone();
    let mut best_unsat = instance.clauses.len().saturating_sub(fitness[best_idx]);

    if best_unsat == 0 {
        return (best_assignment, 0, true);
    }

    for _ in 0..max_generations {
        let mut new_population: Vec<Vec<bool>> = Vec::with_capacity(pop_size);

        for _ in 0..pop_size {
            let p1_idx = tournament_select(&fitness, rng);
            let p2_idx = tournament_select(&fitness, rng);
            let mut child = crossover(&population[p1_idx], &population[p2_idx], fixed, rng);
            mutate(&mut child, fixed, mutation_rate, rng);
            new_population.push(child);
        }

        population = new_population;

        best_idx = 0;
        for (i, indiv) in population.iter().enumerate() {
            let unsat = count_unsat(instance, indiv);
            let fit = instance.clauses.len().saturating_sub(unsat);
            fitness[i] = fit;

            if fit > fitness[best_idx] {
                best_idx = i;
            }

            if unsat < best_unsat {
                best_unsat = unsat;
                best_assignment = indiv.clone();
                if best_unsat == 0 {
                    return (best_assignment, 0, true);
                }
            }
        }
    }

    (best_assignment, best_unsat, best_unsat == 0)
}

/* ------------------------------------------------------------------------- */
/* Ant Colony Component (IAS-style) for WalkSATga                            */
/* ------------------------------------------------------------------------- */

fn run_aco(
    instance: &SatInstance,
    fixed: &[Option<bool>],
    pos_counts: &[usize],
    neg_counts: &[usize],
    ga_best: &[bool],
    rng: &mut SmallRng,
) -> (Vec<bool>, usize, bool) {
    let n = instance.num_variables;
    if n == 0 || instance.clauses.is_empty() {
        let assignment = finalize_assignment(n, fixed, Some(ga_best), rng);
        return (assignment, 0, true);
    }

    let nv = n as f64;
    let num_ants = if nv <= 1_000.0 {
        30
    } else if nv <= 10_000.0 {
        20
    } else {
        10
    };

    let max_generations = if nv <= 1_000.0 {
        20
    } else if nv <= 10_000.0 {
        15
    } else {
        8
    };

    let alpha = 2.0; // pheromone influence
    let beta = 1.0;  // heuristic influence
    let evap = 0.2;  // pheromone evaporation
    let q = 2.0;     // pheromone deposit scaling
    let pher_min = 0.05;
    let pher_max = 50.0;

    let mut pher_true = vec![1.0f64; n];
    let mut pher_false = vec![1.0f64; n];

    // Heuristic information based on literal frequencies. :contentReference[oaicite:4]{index=4}
    let mut eta_true = vec![0.5f64; n];
    let mut eta_false = vec![0.5f64; n];

    for v in 0..n {
        let p = pos_counts.get(v).copied().unwrap_or(0) as f64;
        let ng = neg_counts.get(v).copied().unwrap_or(0) as f64;
        let total = p + ng;
        if total > 0.0 {
            eta_true[v] = (p + 1.0) / (total + 2.0);
            eta_false[v] = (ng + 1.0) / (total + 2.0);
        } else {
            eta_true[v] = 0.5;
            eta_false[v] = 0.5;
        }
    }

    // Seed pheromones with GA best individual & fixed units.
    for v in 0..n {
        if v < fixed.len() {
            if let Some(val) = fixed[v] {
                if val {
                    pher_true[v] += 3.0;
                } else {
                    pher_false[v] += 3.0;
                }
            }
        }
        if v < ga_best.len() {
            if ga_best[v] {
                pher_true[v] += 2.0;
            } else {
                pher_false[v] += 2.0;
            }
        }
    }

    let mut global_best_assignment = ga_best.to_vec();
    let mut global_best_unsat = count_unsat(instance, &global_best_assignment);

    if global_best_unsat == 0 {
        return (global_best_assignment, 0, true);
    }

    let m = instance.clauses.len();

    for _ in 0..max_generations {
        for _ant in 0..num_ants {
            let mut ass = vec![false; n];

            for v in 0..n {
                if v < fixed.len() {
                    if let Some(val) = fixed[v] {
                        ass[v] = val;
                        continue;
                    }
                }

                let tau_t = pher_true[v].powf(alpha) * eta_true[v].powf(beta);
                let tau_f = pher_false[v].powf(alpha) * eta_false[v].powf(beta);
                let sum = tau_t + tau_f;

                let prob_true = if sum <= 0.0 {
                    0.5
                } else {
                    tau_t / sum
                };

                ass[v] = rng.gen::<f64>() < prob_true;
            }

            let unsat = count_unsat(instance, &ass);
            if unsat < global_best_unsat {
                global_best_unsat = unsat;
                global_best_assignment = ass.clone();
            }

            if global_best_unsat == 0 {
                return (global_best_assignment, 0, true);
            }
        }

        // Pheromone update based on global best.
        let delta = q * (m as f64 / ((global_best_unsat + 1) as f64));

        for v in 0..n {
            pher_true[v] = (1.0 - evap) * pher_true[v];
            pher_false[v] = (1.0 - evap) * pher_false[v];

            if global_best_assignment[v] {
                pher_true[v] += delta;
            } else {
                pher_false[v] += delta;
            }

            if pher_true[v] < pher_min {
                pher_true[v] = pher_min;
            } else if pher_true[v] > pher_max {
                pher_true[v] = pher_max;
            }

            if pher_false[v] < pher_min {
                pher_false[v] = pher_min;
            } else if pher_false[v] > pher_max {
                pher_false[v] = pher_max;
            }
        }
    }

    (global_best_assignment, global_best_unsat, global_best_unsat == 0)
}

/* ------------------------------------------------------------------------- */
/* WalkSAT core (used by both vav and ga variants)                           */
/* ------------------------------------------------------------------------- */

fn pick_unsat_clause(
    unsat_list: &mut Vec<usize>,
    is_unsat: &[bool],
    rng: &mut SmallRng,
) -> Option<usize> {
    while !unsat_list.is_empty() {
        let idx = rng.gen_range(0..unsat_list.len());
        let cid = unsat_list[idx];
        if cid < is_unsat.len() && is_unsat[cid] {
            return Some(cid);
        } else {
            unsat_list.swap_remove(idx);
        }
    }
    None
}

fn compute_break(
    var: usize,
    assignment: &[bool],
    sat_counts: &[u32],
    pos_occ: &[Vec<usize>],
    neg_occ: &[Vec<usize>],
) -> u32 {
    let mut br = 0u32;
    if var >= assignment.len() {
        return 0;
    }

    if assignment[var] {
        // x is true: positive literals satisfied; those clauses might become unsat.
        for &cid in &pos_occ[var] {
            if sat_counts[cid] == 1 {
                br += 1;
            }
        }
    } else {
        // x is false: ¬x literals satisfied; those clauses might become unsat.
        for &cid in &neg_occ[var] {
            if sat_counts[cid] == 1 {
                br += 1;
            }
        }
    }
    br
}

/// WalkSAT-style local search with noise, using a WalkSATvav / WalkSATga-style
/// variable selection rule, and an early-exit "probably unsat" heuristic.
///
/// Returns (best_assignment, best_unsat, solved?).
fn walksat_search(
    instance: &SatInstance,
    fixed: &[Option<bool>],
    pos_occ: &[Vec<usize>],
    neg_occ: &[Vec<usize>],
    initial: Vec<bool>,
    rng: &mut SmallRng,
    max_flips: usize,
    noise: f64,
    early_factor: f64,
    early_limit: usize,
) -> (Vec<bool>, usize, bool) {
    let n = instance.num_variables;
    let m = instance.clauses.len();

    if n == 0 || m == 0 {
        return (initial, 0, true);
    }

    let mut global_best_assignment = initial.clone();
    let mut global_best_unsat = count_unsat(instance, &global_best_assignment);

    if global_best_unsat == 0 {
        return (global_best_assignment, 0, true);
    }

    // Number of restarts is modest; we rely on the strong initialization heuristics.
    let max_restarts = 5usize;
    let mut flips_used = 0usize;
    let mut solved = false;
    let mut model = global_best_assignment.clone();

    for restart in 0..max_restarts {
        let mut assignment = if restart == 0 {
            initial.clone()
        } else {
            // New start from allocation strategy to avoid cycling.
            let (pos_counts, neg_counts) = compute_literal_counts(instance);
            random_assignment_allocation(
                instance,
                fixed,
                &pos_counts,
                &neg_counts,
                rng,
                1.8,
                0.56,
            )
        };

        let mut sat_counts = vec![0u32; m];
        let mut is_unsat = vec![false; m];
        let mut unsat_list = Vec::<usize>::new();

        for (cid, clause) in instance.clauses.iter().enumerate() {
            let mut count = 0u32;
        
            for &lit in clause {
                let var_idx = (lit.abs() as usize).saturating_sub(1);
                if var_idx >= assignment.len() {
                    continue;
                }
        
                // Respect fixed values, otherwise use the current assignment.
                let val = if let Some(fixed_val) = fixed.get(var_idx).copied().flatten() {
                    fixed_val
                } else {
                    assignment[var_idx]
                };
        
                if (lit > 0 && val) || (lit < 0 && !val) {
                    count = count.saturating_add(1);
                }
            }
        
            sat_counts[cid] = count;
            if count == 0 {
                is_unsat[cid] = true;
                unsat_list.push(cid);
            } else {
                is_unsat[cid] = false;
            }
        }

        let mut curr_unsat = is_unsat.iter().filter(|&&b| b).count();
        if curr_unsat < global_best_unsat {
            global_best_unsat = curr_unsat;
            global_best_assignment = assignment.clone();
        }
        if curr_unsat == 0 {
            solved = true;
            model = assignment;
            break;
        }

        let mut last_improve_flip = flips_used;

        while flips_used < max_flips {
            if curr_unsat == 0 {
                solved = true;
                model = assignment.clone();
                break;
            }

            // Heuristic "probably unsat / hopeless" early exit:
            // if we've spent a large fraction of the flip budget with no improvement,
            // and still have many unsatisfied clauses, give up on search.
            let spent_fraction = flips_used as f64 / (max_flips as f64);
            if spent_fraction > early_factor
                && flips_used - last_improve_flip > early_limit
                && global_best_unsat > 0
            {
                // Probably unsat (or at least hopeless under this budget).
                break;
            }

            flips_used += 1;

            let cid = match pick_unsat_clause(&mut unsat_list, &is_unsat, rng) {
                Some(c) => c,
                None => {
                    // All clauses currently satisfied.
                    curr_unsat = 0;
                    solved = true;
                    model = assignment.clone();
                    break;
                }
            };

            let clause = &instance.clauses[cid];

            // Choose variable according to WalkSAT rule:
            //   - any variable with break==0 is preferred,
            //   - else, with prob p pick random variable,
            //   - otherwise pick variable with minimum break.
            let mut vars = Vec::<usize>::new();
            let mut zero_break_vars = Vec::<usize>::new();
            let mut min_break = u32::MAX;

            for &lit in clause {
                let var = (lit.abs() as usize).saturating_sub(1);
                if var >= n {
                    continue;
                }
                if var < fixed.len() && fixed[var].is_some() {
                    // Fixed by unit propagation; do not flip.
                    continue;
                }

                let br = compute_break(var, &assignment, &sat_counts, pos_occ, neg_occ);
                if br == 0 {
                    zero_break_vars.push(var);
                }
                if br < min_break {
                    min_break = br;
                }
                vars.push(var);
            }

            if vars.is_empty() {
                // Clause involving only fixed variables is unsatisfied -> unsat under our preprocessing assumptions.
                break;
            }

            let chosen_var = if !zero_break_vars.is_empty() {
                zero_break_vars[rng.gen_range(0..zero_break_vars.len())]
            } else if rng.gen::<f64>() < noise {
                vars[rng.gen_range(0..vars.len())]
            } else {
                let mut best_candidates = Vec::<usize>::new();
                for &v in &vars {
                    let br = compute_break(v, &assignment, &sat_counts, pos_occ, neg_occ);
                    if br == min_break {
                        best_candidates.push(v);
                    }
                }
                best_candidates[rng.gen_range(0..best_candidates.len())]
            };

            // Flip chosen_var and update sat_counts / unsat_list incrementally.
            let old_val = assignment[chosen_var];
            let (set_true_list, set_false_list) = if old_val {
                // true -> false
                (&neg_occ[chosen_var], &pos_occ[chosen_var])
            } else {
                // false -> true
                (&pos_occ[chosen_var], &neg_occ[chosen_var])
            };

            // Literals that are currently satisfied and will become false.
            for &c2 in set_false_list {
                if sat_counts[c2] == 1 {
                    // Clause will become unsatisfied.
                    sat_counts[c2] = 0;
                    if !is_unsat[c2] {
                        is_unsat[c2] = true;
                        unsat_list.push(c2);
                        curr_unsat += 1;
                    }
                } else if sat_counts[c2] > 1 {
                    sat_counts[c2] -= 1;
                }
            }

            // Literals that are currently false and will become satisfied.
            for &c2 in set_true_list {
                if sat_counts[c2] == 0 {
                    sat_counts[c2] = 1;
                    if is_unsat[c2] {
                        is_unsat[c2] = false;
                        if curr_unsat > 0 {
                            curr_unsat -= 1;
                        }
                    }
                } else {
                    sat_counts[c2] += 1;
                }
            }

            assignment[chosen_var] = !old_val;

            if curr_unsat < global_best_unsat {
                global_best_unsat = curr_unsat;
                global_best_assignment = assignment.clone();
                last_improve_flip = flips_used;
                if global_best_unsat == 0 {
                    solved = true;
                    model = global_best_assignment.clone();
                    break;
                }
            }
        }

        if solved || flips_used >= max_flips {
            break;
        }
    }

    if solved {
        (model, 0, true)
    } else {
        (global_best_assignment, global_best_unsat, false)
    }
}

/* ------------------------------------------------------------------------- */
/* WalkSATvav solver                                                         */
/* ------------------------------------------------------------------------- */

fn solve_with_walksat_vav(
    challenge: &Challenge,
    rng: &mut SmallRng,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let (instance, fixed, contrad) = preprocess_instance(challenge);

    // If preprocessing already found a contradiction, we still must return an
    // assignment (random consistent with fixed units if any).
    if contrad {
        let variables = finalize_assignment(
            challenge.num_variables,
            &fixed,
            None,
            rng,
        );
        save_solution(&Solution { variables })?;
        return Ok(());
    }

    // Trivially satisfiable after preprocessing.
    if instance.clauses.is_empty() {
        let variables = finalize_assignment(
            challenge.num_variables,
            &fixed,
            None,
            rng,
        );
        save_solution(&Solution { variables })?;
        return Ok(());
    }

    let (pos_counts, neg_counts) = compute_literal_counts(&instance);
    let (pos_occ, neg_occ) = build_occurrence_lists(&instance);

    // Flip budget: scaled by problem size.
    let nv = instance.num_variables as f64;
    let m = instance.clauses.len() as f64;
    let default_flips = ((nv.sqrt() * m) * 30.0) as usize;
    let max_flips = get_max_flips(hyperparameters, default_flips);

    // Extract hyperparameters with defaults
    let noise = get_noise(hyperparameters, 0.55);
    let early_factor = get_early_factor(hyperparameters, 0.8);
    let default_early_limit = ((instance.num_variables as f64) * 20.0) as usize;
    let early_limit = get_early_limit(hyperparameters, default_early_limit);

    // Allocation-based initial assignment (WalkSATvav). :contentReference[oaicite:5]{index=5}
    let initial = random_assignment_allocation(
        &instance,
        &fixed,
        &pos_counts,
        &neg_counts,
        rng,
        1.8,
        0.56,
    );

    let (best_assignment, _best_unsat, _solved) = walksat_search(
        &instance,
        &fixed,
        &pos_occ,
        &neg_occ,
        initial,
        rng,
        max_flips,
        noise,
        early_factor,
        early_limit,
    );

    // Always return a full assignment (either model or best/ random).
    let variables =
        finalize_assignment(challenge.num_variables, &fixed, Some(&best_assignment), rng);
    save_solution(&Solution { variables })?;
    Ok(())
}

/* ------------------------------------------------------------------------- */
/* WalkSATga solver (GA + ACO + WalkSAT)                                     */
/* ------------------------------------------------------------------------- */

fn solve_with_walksat_ga(
    challenge: &Challenge,
    rng: &mut SmallRng,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let (instance, fixed, contrad) = preprocess_instance(challenge);

    if contrad {
        let variables = finalize_assignment(
            challenge.num_variables,
            &fixed,
            None,
            rng,
        );
        save_solution(&Solution { variables })?;
        return Ok(());
    }

    if instance.clauses.is_empty() {
        let variables = finalize_assignment(
            challenge.num_variables,
            &fixed,
            None,
            rng,
        );
        save_solution(&Solution { variables })?;
        return Ok(());
    }

    let (pos_counts, neg_counts) = compute_literal_counts(&instance);
    let (pos_occ, neg_occ) = build_occurrence_lists(&instance);

    let nv = instance.num_variables as f64;
    let m = instance.clauses.len() as f64;
    let default_flips = ((nv.sqrt() * m) * 6.0) as usize;
    let max_flips = get_max_flips(hyperparameters, default_flips);

    // Extract hyperparameters with defaults (a bit noisier for dense hard instances)
    let noise = get_noise(hyperparameters, 0.6);
    let early_factor = get_early_factor(hyperparameters, 0.7);
    let default_early_limit = ((instance.num_variables as f64) * 20.0) as usize;
    let early_limit = get_early_limit(hyperparameters, default_early_limit);

    // 1) Global search with improved GA.
    let (ga_best, ga_unsat, ga_solved) =
        run_ga(&instance, &fixed, &pos_counts, &neg_counts, rng);
    if ga_solved && ga_unsat == 0 {
        let variables =
            finalize_assignment(challenge.num_variables, &fixed, Some(&ga_best), rng);
        save_solution(&Solution { variables })?;
        return Ok(());
    }

    // 2) Ant-colony style IAS heuristic, seeded by GA best. :contentReference[oaicite:6]{index=6}
    let (aco_best, aco_unsat, aco_solved) =
        run_aco(&instance, &fixed, &pos_counts, &neg_counts, &ga_best, rng);

    if aco_solved && aco_unsat == 0 {
        let variables =
            finalize_assignment(challenge.num_variables, &fixed, Some(&aco_best), rng);
        save_solution(&Solution { variables })?;
        return Ok(());
    }

    // Choose the better of GA / ACO as the WalkSAT starting point.
    let (start, start_unsat) = if aco_unsat <= ga_unsat {
        (aco_best, aco_unsat)
    } else {
        (ga_best, ga_unsat)
    };

    // 3) Local search refinement with WalkSAT-core.
    let (walk_best, walk_unsat, walk_solved) = walksat_search(
        &instance,
        &fixed,
        &pos_occ,
        &neg_occ,
        start.clone(),
        rng,
        max_flips,
        noise,
        early_factor,
        early_limit,
    );

    // Choose the best assignment we saw anywhere (GA, ACO, WalkSAT).
    let mut best_assignment = start;
    let mut best_unsat = start_unsat;

    if walk_unsat < best_unsat {
        best_unsat = walk_unsat;
        best_assignment = walk_best;
    }

    // (Optionally we could also compare with ga_best directly, but start already
    // captures the min of GA/ACO.)

    if walk_solved && best_unsat == 0 {
        let variables =
            finalize_assignment(challenge.num_variables, &fixed, Some(&best_assignment), rng);
        save_solution(&Solution { variables })?;
        return Ok(());
    }

    // Early “probably unsat” exit or fuel exhaustion: still return a full assignment.
    let variables =
        finalize_assignment(challenge.num_variables, &fixed, Some(&best_assignment), rng);
    save_solution(&Solution { variables })?;
    Ok(())
}
