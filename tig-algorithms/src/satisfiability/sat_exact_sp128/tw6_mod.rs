
#[path = "tw6_track2.rs"]
mod track2;
use anyhow::{anyhow, Result};
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

#[allow(dead_code)]
pub fn help() {
    println!(
        "sat_tailwalk_v6: use {{\"max_fuel_high\":175000000000}} for n_vars=10000, ratio=4267; use null for other tracks"
    );
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

    Err(anyhow!(
        "sat_exact: tw6 route {:?} is not used by this composite (only Track2)",
        route
    ))
}
