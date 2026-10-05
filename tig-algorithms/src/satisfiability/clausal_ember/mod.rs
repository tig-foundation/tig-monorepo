// clausal_ember — per-track SAT solver.
// Dispatch by instance size to a dedicated stochastic-local-search engine
// (adaptive-noise WalkSAT/probSAT flip loop, bounded restarts, stagnation
// perturbation); the largest tracks add a survey-propagation seeding stage.
// A per-track fuel budget bounds the work. All tuning is read in from_map;
// defaults reproduce the best measured operating point of each track.

pub mod common;
pub mod track_t1;
pub mod track_t2;
pub mod track_t3;
pub mod track_t4;
pub mod track_t5;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Hparams {
    pub base_prob: Option<f64>,
    pub max_prob: Option<f64>,
    pub check_interval: Option<usize>,
    pub stagnation_limit: Option<usize>,
    pub perturbation_flips: Option<usize>,
    pub max_fuel_high: Option<f64>,
    pub max_fuel_low: Option<f64>,
    pub max_reinits: Option<usize>,
    pub sp_on: Option<bool>,
    pub sp_f: Option<f64>,
    pub sp_eps: Option<f64>,
    pub sp_triv: Option<f64>,
    pub sp_edge_budget: Option<f64>,
}

impl Hparams {
    pub fn for_t1() -> Self {
        let mut h = Self::default();
        h.max_reinits = Some(15);
        h
    }
    pub fn for_t2() -> Self {
        let mut h = Self::default();
        h.max_fuel_high = Some(180_000_000_000.0);
        h
    }
    pub fn for_t3() -> Self {
        let mut h = Self::default();
        h.max_fuel_high = Some(180_000_000_000.0);
        h
    }
    pub fn for_t4() -> Self { Self::default() }
    pub fn for_t5() -> Self {
        let mut h = Self::default();
        h.stagnation_limit = Some(3);
        h.sp_on = Some(true);
        h.sp_f = Some(0.01);
        h.sp_eps = Some(1e-3);
        h.sp_triv = Some(1e-2);
        h.sp_edge_budget = Some(3e9);
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
                if u.sp_on.is_some() { self.sp_on = u.sp_on; }
                if u.sp_f.is_some() { self.sp_f = u.sp_f; }
                if u.sp_eps.is_some() { self.sp_eps = u.sp_eps; }
                if u.sp_triv.is_some() { self.sp_triv = u.sp_triv; }
                if u.sp_edge_budget.is_some() { self.sp_edge_budget = u.sp_edge_budget; }
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
        (5000, 21335) => track_t1::solve(challenge, save_solution, &Hparams::for_t1().merge_user(user)),
        (7500, 32002) => track_t2::solve(challenge, save_solution, &Hparams::for_t2().merge_user(user)),
        (10000, 42670) => track_t3::solve(challenge, save_solution, &Hparams::for_t3().merge_user(user)),
        (100000, 415000) => track_t4::solve(challenge, save_solution, &Hparams::for_t4().merge_user(user)),
        (100000, 420000) => track_t5::solve(challenge, save_solution, &Hparams::for_t5().merge_user(user)),
        _ => {
            let density = nc_total as f64 / (nv.max(1)) as f64;
            if nv <= 6250 {
                track_t1::solve(challenge, save_solution, &Hparams::for_t1().merge_user(user))
            } else if nv <= 8750 {
                track_t2::solve(challenge, save_solution, &Hparams::for_t2().merge_user(user))
            } else if nv <= 30000 {
                track_t3::solve(challenge, save_solution, &Hparams::for_t3().merge_user(user))
            } else if density < 4.175 {
                track_t4::solve(challenge, save_solution, &Hparams::for_t4().merge_user(user))
            } else {
                track_t5::solve(challenge, save_solution, &Hparams::for_t5().merge_user(user))
            }
        }
    }
}

pub fn help() {
    println!("clausal_ember - per-track SAT solver (5 tracks, 1 engine per track)");
    println!("tracks: (nv,nc) = (5000,21335) (7500,32002) (10000,42670) (100000,415000) (100000,420000)");
    println!("hyperparameters (all optional, all read, all overridable):");
    println!("  base_prob          f64   - walk probability floor           [t2,t4]");
    println!("  max_prob           f64   - adaptive noise ceiling           [t2,t4]");
    println!("  check_interval     usize - flips between stagnation checks  [t1,t2,t3,t4,t5]");
    println!("  stagnation_limit   usize - checks without progress before perturbation [t1,t4,t5]");
    println!("  perturbation_flips usize - flips injected on perturbation   [t4,t5]");
    println!("  max_fuel_high      f64   - fuel budget, dense/small tracks  [t1,t2,t3]");
    println!("  max_fuel_low       f64   - fuel budget, large tracks        [t4,t5]");
    println!("  max_reinits        usize - bounded restarts                 [t1]");
    println!("  sp_on              bool  - survey-propagation seeding on/off       [t5]");
    println!("  sp_f               f64   - fraction of vars fixed per decim. step  [t5]");
    println!("  sp_eps             f64   - survey convergence tolerance            [t5]");
    println!("  sp_triv            f64   - trivial-survey (paramagnetic) cutoff    [t5]");
    println!("  sp_edge_budget     f64   - cap on edge-sweeps spent in seeding     [t5]");
    println!("defaults reproduce the best measured operating point of each track with no hp_json.");
}
