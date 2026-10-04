// flipwave - satisfiability solver. One fuel-bounded stochastic local-search engine per
// track (dispatch by size), shared CSR preprocessing in `common`. Search diversity comes
// from deterministic trajectory portfolios and message-passing seeding; randomisation is
// seeded only from public instance content. Tuning read in from_map; defaults ship.

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
    pub pf_arms: Option<usize>,
    pub pf_arm0: Option<usize>,
    pub sp_on: Option<bool>,
    pub sp_f: Option<f64>,
    pub sp_eps: Option<f64>,
    pub sp_triv: Option<f64>,
    pub sp_edge_budget: Option<f64>,
    pub spr_on: Option<bool>,
    pub spr_target: Option<f64>,
    pub spr_minit: Option<f64>,
    pub spr_maxit: Option<f64>,
    pub spr_rate: Option<f64>,
    pub spr_gmax: Option<f64>,
    pub spr_lam: Option<f64>,
    pub spr_damp: Option<f64>,
    pub spr_check: Option<f64>,
    pub spr_patience: Option<f64>,
    pub spr_budget: Option<f64>,
    pub spr_flip_frac: Option<f64>,
    pub cb_break: Option<f64>,
}

impl Hparams {
    pub fn for_t1() -> Self {
        let mut h = Self::default();
        h.max_reinits = Some(15);
        h.max_fuel_high = Some(125_000_000_000.0);
        h.pf_arms = Some(4);
        h
    }
    pub fn for_t2() -> Self {
        let mut h = Self::default();
        h.max_fuel_high = Some(750_000_000_000.0);
        h.pf_arms = Some(1);
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
    println!("flipwave - per-track SAT solver (5 tracks, 1 engine per track)");
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
    println!("  pf_arms            usize - sequential trajectory portfolio: number of arms [t1,t2,t3]");
    println!("  pf_arm0            usize - index of the first arm (salt offset)            [t1,t2,t3]");
    println!("  NB: on t1/t2/t3 `max_fuel_high` is the budget of EACH arm.");
    println!("  sp_on              bool  - survey-propagation seeding on/off       [t5]");
    println!("  sp_f               f64   - fraction of vars fixed per decim. step  [t5]");
    println!("  sp_eps             f64   - survey convergence tolerance            [t5]");
    println!("  sp_triv            f64   - trivial-survey (paramagnetic) cutoff    [t5]");
    println!("  sp_edge_budget     f64   - cap on edge-sweeps spent in seeding     [t5]");
    println!("  spr_on             bool  - reinforced message-passing seeding (default); false = survey decimation [t5]");
    println!("  spr_target         f64   - stop seeding at this many unsatisfied clauses [t5]");
    println!("  spr_minit          f64   - minimum iterations before the target may fire [t5]");
    println!("  spr_maxit          f64   - iteration cap                                 [t5]");
    println!("  spr_rate           f64   - reinforcement ramp: gamma = min(rate*it, gmax) [t5]");
    println!("  spr_gmax           f64   - reinforcement ceiling                         [t5]");
    println!("  spr_lam            f64   - field smoothing                               [t5]");
    println!("  spr_damp           f64   - survey damping                                [t5]");
    println!("  spr_check          f64   - iterations between residual checks             [t5]");
    println!("  spr_patience       f64   - checks without improvement before giving up    [t5]");
    println!("  spr_budget         f64   - cap on edge-sweeps spent in seeding            [t5]");
    println!("  spr_flip_frac      f64   - share of the flip budget given to the seeded phase   [t5]");
    println!("  cb_break           f64   - probSAT break exponent, f(b) = cb^-b           [t5]");
    println!("defaults reproduce the best measured operating point of each track with no hp_json.");
}
