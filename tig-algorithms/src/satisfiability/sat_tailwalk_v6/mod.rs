
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
