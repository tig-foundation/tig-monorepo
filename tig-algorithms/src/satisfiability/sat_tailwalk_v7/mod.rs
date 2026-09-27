// sat_tailwalk_v7 derivative/composite source; constituent origins and retained copyrights are listed in README.md.
mod donor_kernel;
mod engine_iv;
mod track1_cached;
mod engine_b;
mod engine_spr;

mod track1;
mod track2;
mod track3;
mod track4;
mod track5;
mod track_fallback;
use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

#[derive(Clone, Serialize, Deserialize)]
pub struct Hyperparameters {
    pub base_prob: Option<f64>,
    pub max_prob: Option<f64>,
    pub check_interval: Option<usize>,
    pub stagnation_limit: Option<usize>,
    pub perturbation_flips: Option<usize>,
    pub max_fuel_high: Option<f64>,
    pub max_fuel_low: Option<f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SolverRoute {
    Track1,
    Track2,
    Track3,
    Track4,
    Track5,
    Fallback,
}

fn select_route(
    raw_num_variables: usize,
    raw_num_clauses: usize,
    num_variables: usize,
    density: f64,
) -> SolverRoute {
    if (raw_num_variables, raw_num_clauses) == (100_000, 420_000) {
        return SolverRoute::Track5;
    }
    if density >= 4.25 {
        if num_variables <= 5000 {
            return SolverRoute::Track1;
        }
        if num_variables <= 7500 {
            return SolverRoute::Track2;
        }
        return SolverRoute::Track3;
    }
    if density < 4.18 {
        return SolverRoute::Track4;
    }
    SolverRoute::Fallback
}

pub fn help() {
    println!("sat_tailwalk_v7: use {{}} or null for all five standard satisfiability tracks; see README.md for route budgets and source attribution");
}

#[path = "a115_reference.rs"]
mod reference_a115;

/// Per-track composite derived from v6 and the public solvers acknowledged in README.md.
/// Alternate routes forward complete SAT assignments; failed alternates retain the original fallback seed.
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    if (challenge.num_variables, challenge.clauses.len()) == (100000, 420000) {
        // Accept only a complete SAT assignment from the SP portfolio. If it
        // fails, enter the unchanged R5 path below with a fresh original RNG.
        let solved = std::cell::Cell::new(false);
        let accept_sat = |solution: &Solution| {
            if solution.variables.len() == challenge.num_variables
                && challenge.clauses.iter().all(|clause| clause.iter().any(|&lit| {
                    solution.variables[(lit.abs() - 1) as usize] == (lit > 0)
                }))
            {
                save_solution(solution)?;
                solved.set(true);
            }
            Ok(())
        };
        engine_b::solve(challenge, &accept_sat, hyperparameters)?;
        if solved.get() { return Ok(()); }
    }
    if (challenge.num_variables, challenge.clauses.len()) == (5000, 21335) {
        // The complement and original R5 each seed their own RNG. Forward
        // complete SAT results only, otherwise use the unchanged R5 below.
        let solved = std::cell::Cell::new(false);
        let accept_sat = |solution: &Solution| {
            if solution.variables.len() == challenge.num_variables
                && challenge.clauses.iter().all(|clause| clause.iter().any(|&lit| {
                    solution.variables[(lit.abs()-1) as usize] == (lit > 0)
                }))
            {
                save_solution(solution)?;
                solved.set(true);
            }
            Ok(())
        };
        let budget = hyperparameters.as_ref()
            .and_then(|m| m.get("t4_fuel_b")).and_then(Value::as_u64)
            .unwrap_or(50_000_000_000);
        let mut complement_hp = Map::new();
        complement_hp.insert("max_fuel_high".into(), Value::from(budget));
        engine_iv::solve(challenge, &accept_sat, &Some(complement_hp))?;
        if solved.get() { return Ok(()); }
    }
    if (challenge.num_variables, challenge.clauses.len()) == (5000, 21335) {
        // t4_fuel_b configures only the preceding complement. The original
        // a115 fallback ignores it. Other keys retain a115's complete parser,
        // diagnostics and defaults, including whole-object parse failures.
        let default_fallback = hyperparameters.as_ref()
            .map_or(true, |m| m.keys().all(|key| key == "t4_fuel_b"));
        if default_fallback {
            return track1_cached::solve(challenge, save_solution);
        }
        return reference_a115::solve_challenge(challenge, save_solution, hyperparameters);
    }
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
    let density = nc as f64 / nv as f64;
    let route = select_route(
        challenge.num_variables,
        challenge.clauses.len(),
        nv,
        density,
    );

    if route == SolverRoute::Track2 {
        return track2::solve(challenge, &hp, save_solution);
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
            let (a, b, c) = (orig[0], orig[1], orig[2]);
            if a == -b || a == -c || b == -c { continue; }
            let va = (a.abs() - 1) as usize;
            if a > 0 { all_data[p_pos[va] as usize] = ci; p_pos[va] += 1; }
            else { all_data[n_pos[va] as usize] = ci; n_pos[va] += 1; }
            if b != a {
                let vb = (b.abs() - 1) as usize;
                if b > 0 { all_data[p_pos[vb] as usize] = ci; p_pos[vb] += 1; }
                else { all_data[n_pos[vb] as usize] = ci; n_pos[vb] += 1; }
            }
            if c != a && c != b {
                let vc = (c.abs() - 1) as usize;
                if c > 0 { all_data[p_pos[vc] as usize] = ci; p_pos[vc] += 1; }
                else { all_data[n_pos[vc] as usize] = ci; n_pos[vc] += 1; }
            }
            cl.push(a);
            if b != a { cl.push(b); }
            if c != a && c != b { cl.push(c); }
            co.push(cl.len() as u32);
            ci += 1;
        }
    }

    match route {
        SolverRoute::Track1 => {
            track1::solve(
                &hp, &mut rng, nv, nc, density, p_cnt, n_cnt, &all_off,
                &p_bound, &all_data, &mut cl, &co, save_solution,
            )
        }
        SolverRoute::Track2 => {
            unreachable!("Track2 returns before the shared index build")
        }
        SolverRoute::Track3 => {
            track3::solve(
                &hp, &mut rng, nv, nc, density, p_cnt, n_cnt, &all_off,
                &p_bound, &all_data, &mut cl, &co, save_solution,
            )
        }
        SolverRoute::Track4 => {
            track4::solve(
                &hp, &mut rng, nv, nc, density, p_cnt, n_cnt, &all_off,
                &p_bound, &all_data, &mut cl, &co, save_solution,
            )
        }
        SolverRoute::Track5 => {
            track5::solve(
                &hp, &mut rng, nv, nc, density, p_cnt, n_cnt, &all_off,
                &p_bound, &all_data, &mut cl, &co, save_solution,
            )
        }
        SolverRoute::Fallback => {
            track_fallback::solve(
                &hp, &mut rng, nv, nc, density, p_cnt, n_cnt, &all_off,
                &p_bound, &all_data, &mut cl, &co, save_solution,
            )
        }
    }
}

#[repr(C, align(16))]
struct SearchClause {
    literals: [i32; 3],
    len: u32,
}

fn pack_search_clauses(flat: &[i32], offsets: &[u32]) -> Vec<SearchClause> {
    offsets.windows(2).map(|pair| {
        let slice = &flat[pair[0] as usize..pair[1] as usize];
        let mut literals = [0i32; 3];
        literals[..slice.len()].copy_from_slice(slice);
        SearchClause { literals, len: slice.len() as u32 }
    }).collect()
}

// Normalization produces one to three distinct literals. Keep the generic case
// explicit so this helper also retains modulo semantics for other lengths.
#[inline(always)]
fn short_clause_index(value: usize, len: usize) -> usize {
    match len {
        1 => 0,
        2 => value & 1,
        3 => value % 3,
        _ => value % len,
    }
}
