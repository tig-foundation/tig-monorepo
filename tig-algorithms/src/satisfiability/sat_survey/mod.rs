// One solver per track. Tracks 1-3 are the scored 4267-ratio engines.
// Tracks 4-5 are the 100000-variable 4150 and 4200 engines.

pub mod common;
pub mod track_t4;
pub mod track_t5;

mod donor_kernel;
mod list_edit;
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
mod a115_engine_e;
mod a115_engine_ag;

use anyhow::Result;
use rand::{rngs::SmallRng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Hparams {
    #[serde(default)]
    pub base_prob: Option<f64>,
    #[serde(default)]
    pub max_prob: Option<f64>,
    #[serde(default)]
    pub check_interval: Option<usize>,
    #[serde(default)]
    pub stagnation_limit: Option<usize>,
    #[serde(default)]
    pub perturbation_flips: Option<usize>,
    #[serde(default)]
    pub max_fuel_high: Option<f64>,
    #[serde(default)]
    pub max_fuel_low: Option<f64>,
    #[serde(default)]
    pub max_reinits: Option<usize>,
    #[serde(default)]
    pub pf_arms: Option<usize>,
    #[serde(default)]
    pub pf_arm0: Option<usize>,
    #[serde(default)]
    pub sp_on: Option<bool>,
    #[serde(default)]
    pub sp_f: Option<f64>,
    #[serde(default)]
    pub sp_eps: Option<f64>,
    #[serde(default)]
    pub sp_triv: Option<f64>,
    #[serde(default)]
    pub sp_edge_budget: Option<f64>,
    #[serde(default)]
    pub spr_on: Option<bool>,
    #[serde(default)]
    pub spr_target: Option<f64>,
    #[serde(default)]
    pub spr_minit: Option<f64>,
    #[serde(default)]
    pub spr_maxit: Option<f64>,
    #[serde(default)]
    pub spr_rate: Option<f64>,
    #[serde(default)]
    pub spr_gmax: Option<f64>,
    #[serde(default)]
    pub spr_lam: Option<f64>,
    #[serde(default)]
    pub spr_damp: Option<f64>,
    #[serde(default)]
    pub spr_check: Option<f64>,
    #[serde(default)]
    pub spr_patience: Option<f64>,
    #[serde(default)]
    pub spr_budget: Option<f64>,
    #[serde(default)]
    pub spr_flip_frac: Option<f64>,
    #[serde(default)]
    pub cb_break: Option<f64>,
}

impl Hparams {
    pub fn for_t1() -> Self {
        Self::default()
    }

    pub fn for_t2() -> Self {
        let mut h = Self::default();
        h.max_fuel_high = Some(18_000_000_000.0);
        h
    }

    pub fn for_t3() -> Self {
        let mut h = Self::default();
        h.max_fuel_high = Some(10_000_000_000.0);
        h
    }

    pub fn for_t4() -> Self {
        Self::default()
    }

    pub fn for_t5() -> Self {
        let mut h = Self::default();
        h.stagnation_limit = Some(3);
        h.sp_on = Some(true);
        h.sp_f = Some(0.01);
        h.sp_eps = Some(5e-3);
        h.sp_triv = Some(1e-2);
        h.sp_edge_budget = Some(3e9);
        h.spr_on = Some(true);
        h.spr_target = Some(300.0);
        h.spr_minit = Some(0.0);
        h.spr_maxit = Some(1500.0);
        h.spr_rate = Some(1e-3);
        h.spr_gmax = Some(0.5);
        h.spr_lam = Some(0.1);
        h.spr_damp = Some(0.3);
        h.spr_check = Some(10.0);
        h.spr_patience = Some(150.0);
        h.spr_budget = Some(3e9);
        h.spr_flip_frac = Some(0.15);
        h.cb_break = Some(2.5);
        h
    }

    pub fn from_map(m: &Map<String, Value>) -> Option<Self> {
        serde_json::from_value::<Self>(Value::Object(m.clone())).ok()
    }

    pub fn merge_user(mut self, user: Option<&Map<String, Value>>) -> Self {
        if let Some(m) = user {
            if let Some(u) = Self::from_map(m) {
                if u.base_prob.is_some() { self.base_prob = u.base_prob; }
                if u.max_prob.is_some() { self.max_prob = u.max_prob; }
                if u.check_interval.is_some() { self.check_interval = u.check_interval; }
                if u.stagnation_limit.is_some() { self.stagnation_limit = u.stagnation_limit; }
                if u.perturbation_flips.is_some() { self.perturbation_flips = u.perturbation_flips; }
                if u.max_fuel_high.is_some() { self.max_fuel_high = u.max_fuel_high; }
                if u.max_fuel_low.is_some() { self.max_fuel_low = u.max_fuel_low; }
                if u.max_reinits.is_some() { self.max_reinits = u.max_reinits; }
                if u.pf_arms.is_some() { self.pf_arms = u.pf_arms; }
                if u.pf_arm0.is_some() { self.pf_arm0 = u.pf_arm0; }
                if u.sp_on.is_some() { self.sp_on = u.sp_on; }
                if u.sp_f.is_some() { self.sp_f = u.sp_f; }
                if u.sp_eps.is_some() { self.sp_eps = u.sp_eps; }
                if u.sp_triv.is_some() { self.sp_triv = u.sp_triv; }
                if u.sp_edge_budget.is_some() { self.sp_edge_budget = u.sp_edge_budget; }
                if u.spr_on.is_some() { self.spr_on = u.spr_on; }
                if u.spr_target.is_some() { self.spr_target = u.spr_target; }
                if u.spr_minit.is_some() { self.spr_minit = u.spr_minit; }
                if u.spr_maxit.is_some() { self.spr_maxit = u.spr_maxit; }
                if u.spr_rate.is_some() { self.spr_rate = u.spr_rate; }
                if u.spr_gmax.is_some() { self.spr_gmax = u.spr_gmax; }
                if u.spr_lam.is_some() { self.spr_lam = u.spr_lam; }
                if u.spr_damp.is_some() { self.spr_damp = u.spr_damp; }
                if u.spr_check.is_some() { self.spr_check = u.spr_check; }
                if u.spr_patience.is_some() { self.spr_patience = u.spr_patience; }
                if u.spr_budget.is_some() { self.spr_budget = u.spr_budget; }
                if u.spr_flip_frac.is_some() { self.spr_flip_frac = u.spr_flip_frac; }
                if u.cb_break.is_some() { self.cb_break = u.cb_break; }
            }
        }
        self
    }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let nv = challenge.num_variables;
    let nc_total = challenge.clauses.len();
    let user = hyperparameters.as_ref();
    match (nv, nc_total) {
        (5000, 21335) | (7500, 32002) | (10000, 42670) => {
            solve_fast(challenge, save_solution, hyperparameters)
        }
        (100000, 415000) => track_t4::solve(challenge, save_solution, &Hparams::for_t4().merge_user(user)),
        (100000, 420000) => track_t5::solve(challenge, save_solution, &Hparams::for_t5().merge_user(user)),
        _ => {
            let density = nc_total as f64 / (nv.max(1)) as f64;
            if nv <= 30000 {
                solve_fast(challenge, save_solution, hyperparameters)
            } else if density < 4.175 {
                track_t4::solve(challenge, save_solution, &Hparams::for_t4().merge_user(user))
            } else {
                track_t5::solve(challenge, save_solution, &Hparams::for_t5().merge_user(user))
            }
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct Hyperparameters {
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

fn solve_fast(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    if (challenge.num_variables, challenge.clauses.len()) == (100000, 420000) {
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
        let default_fallback = hyperparameters.as_ref()
            .map_or(true, |m| m.keys().all(|key| key == "t4_fuel_b"));
        if default_fallback {
            return track1_cached::solve(challenge, save_solution);
        }
        return a115_engine_ag::solve(challenge, save_solution, hyperparameters);
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

#[inline(always)]
fn short_clause_index(value: usize, len: usize) -> usize {
    match len {
        1 => 0,
        2 => value & 1,
        3 => value % 3,
        _ => value % len,
    }
}

pub fn help() {
    println!("sat_survey - per-track SAT solver");
    println!("  t1 n_vars=5000   ratio=4267");
    println!("  t2 n_vars=7500   ratio=4267");
    println!("  t3 n_vars=10000  ratio=4267");
    println!("  t4 n_vars=100000 ratio=4150");
    println!("  t5 n_vars=100000 ratio=4200");
    println!("null hyperparameters are the scored defaults; max_fuel_high and max_fuel_low override the track budget");
}
