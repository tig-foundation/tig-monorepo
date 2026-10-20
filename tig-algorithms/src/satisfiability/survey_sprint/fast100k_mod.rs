// fast100k_dev — development package: only the two n_vars=100000 tracks (lab build, not for submission).
#[path = "fast100k.rs"]
mod fast100k;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Hp {
    pub spr_rate: Option<f64>,
    pub spr_gmax: Option<f64>,
    pub spr_lam: Option<f64>,
    pub spr_damp: Option<f64>,
    pub spr_check: Option<usize>,
    pub spr_patience: Option<usize>,
    pub spr_maxit: Option<usize>,
    pub spr_target: Option<usize>,
    pub cb: Option<f64>,
    pub flip_cap0: Option<f64>,
    pub flip_cap1: Option<f64>,
    pub walk_restart: Option<f64>,
    pub spr_async: Option<bool>,
    pub spr_resync: Option<usize>,
    pub lab_trace: Option<bool>,
}

fn apply(cfg: &mut fast100k::Cfg, h: &Hp) {
    if let Some(v) = h.spr_rate { cfg.rate = v as f32; }
    if let Some(v) = h.spr_gmax { cfg.gmax = v as f32; }
    if let Some(v) = h.spr_lam { cfg.lam = v as f32; }
    if let Some(v) = h.spr_damp { cfg.damp = v as f32; }
    if let Some(v) = h.spr_check { cfg.check = v; }
    if let Some(v) = h.spr_patience { cfg.patience = v; }
    if let Some(v) = h.spr_maxit { cfg.maxit = v; }
    if let Some(v) = h.spr_target { cfg.target = v; }
    if let Some(v) = h.cb { cfg.cb = v; }
    if let Some(v) = h.flip_cap0 { cfg.flip_cap0 = v as u64; }
    if let Some(v) = h.flip_cap1 { cfg.flip_cap1 = v as u64; }
    if let Some(v) = h.walk_restart { cfg.restart = v as u64; }
    if let Some(v) = h.spr_async { cfg.asyncu = v; }
    if let Some(v) = h.spr_resync { cfg.resync = v; }
    if let Some(v) = h.lab_trace { cfg.trace = v; }
}

pub fn help() {
    println!("survey_sprint - n_vars=100000 routes for n_vars=100000 (ratio 4150 / 4200): SP reinforcement seed + probSAT");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let nv = challenge.num_variables;
    let nc = challenge.clauses.len();
    let (mut cfg, track) = match (nv, nc) {
        (100000, 415000) => (fast100k::Cfg::for_4150(), 4150u32),
        (100000, 420000) => (fast100k::Cfg::for_4200(), 4200u32),
        _ => return Err(anyhow!("survey_sprint: fast100k handles only the n_vars=100000 tracks (got {} vars, {} clauses)", nv, nc)),
    };
    if let Some(m) = hyperparameters {
        if let Ok(h) = serde_json::from_value::<Hp>(Value::Object(m.clone())) { apply(&mut cfg, &h); }
    }
    fast100k::solve(challenge, save_solution, &cfg, track)
}
