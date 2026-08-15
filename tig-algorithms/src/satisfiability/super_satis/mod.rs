mod weighted;
mod track1;
mod track2;
mod track3;
mod track4;
mod track5;

use anyhow::Result;
use rand::{rngs::SmallRng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

#[derive(Serialize, Deserialize, Clone)]
pub struct Hyperparameters {
    pub base_prob: Option<f64>,
    pub max_prob: Option<f64>,
    pub check_interval: Option<usize>,
    pub stagnation_limit: Option<usize>,
    pub perturbation_flips: Option<usize>,
    pub max_fuel_high: Option<f64>,
    pub max_fuel_low: Option<f64>,
}

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
}

impl Hparams {
    pub fn for_t1() -> Self {
        // Track3 / engine A: 315B clears the 7th-nonce fuel cliff (180B was under-fueled).
        let mut h = Self::default();
        h.max_fuel_high = Some(315_000_000_000.0);
        h
    }
    pub fn for_t3() -> Self { Self::default() }
    pub fn for_t4() -> Self {
        let mut h = Self::default();
        h.max_reinits = Some(15);
        h
    }
    pub fn for_t5() -> Self { Self::default() }
    pub fn for_t38() -> Self {
        let mut h = Self::default();
        h.stagnation_limit = Some(3);
        h
    }

    pub(crate) fn merge_user(mut self, user: Option<&Map<String, Value>>) -> Self {
        if let Some(m) = user {
            if let Ok(u) = serde_json::from_value::<Hparams>(Value::Object(m.clone())) {
                if u.base_prob.is_some() { self.base_prob = u.base_prob; }
                if u.max_prob.is_some() { self.max_prob = u.max_prob; }
                if u.check_interval.is_some() { self.check_interval = u.check_interval; }
                if u.stagnation_limit.is_some() { self.stagnation_limit = u.stagnation_limit; }
                if u.perturbation_flips.is_some() { self.perturbation_flips = u.perturbation_flips; }
                if u.max_fuel_high.is_some() { self.max_fuel_high = u.max_fuel_high; }
                if u.max_fuel_low.is_some() { self.max_fuel_low = u.max_fuel_low; }
                if u.max_reinits.is_some() { self.max_reinits = u.max_reinits; }
            }
        }
        self
    }
}

pub(crate) struct Prepared {
    pub rng: SmallRng,
    pub nv: usize,
    pub nc: usize,
    pub density: f64,
    pub p_cnt: Vec<u32>,
    pub n_cnt: Vec<u32>,
    pub all_off: Vec<u32>,
    pub p_bound: Vec<u32>,
    pub all_data: Vec<u32>,
    pub cl: Vec<i32>,
    pub co: Vec<u32>,
}

#[inline(always)]
pub(crate) fn preprocess(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Prepared {
    let nv = challenge.num_variables;
    let _ = save_solution(&Solution { variables: vec![false; nv] });
    let rng = SmallRng::seed_from_u64(u64::from_le_bytes(
        challenge.seed[..8].try_into().unwrap(),
    ));

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

    let density = nc as f64 / nv as f64;
    Prepared { rng, nv, nc, density, p_cnt, n_cnt, all_off, p_bound, all_data, cl, co }
}

pub fn help() {
    println!("Tig is my favourite challenge!!!");
}

/// Bake track3 (engine A) defaults into the JSON HP map.
/// `max_fuel_high=315B` clears the 7th-nonce fuel cliff (180B ⇒ 6/32).
fn track3_hyperparameters(hyperparameters: &Option<Map<String, Value>>) -> Option<Map<String, Value>> {
    let mut map = hyperparameters.clone().unwrap_or_default();
    map.insert("max_fuel_high".to_string(), Value::from(315_000_000_000.0));
    Some(map)
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
    let nc = challenge.clauses.len();
    if nv == 0 || nc == 0 {
        save_solution(&Solution { variables: vec![false; nv] })?;
        return Ok(());
    }

    let ratio = (nc as f64 / nv as f64 * 1000.0).round() as u32;
    match (nv, ratio) {
        (5000, 4267) => return track1::solve(challenge, save_solution),
        (7500, 4267) => return track2::solve(challenge, &hp, save_solution),
        (10000, 4267) => {
            let hp3 = track3_hyperparameters(hyperparameters);
            return track3::solve(challenge, save_solution, &hp3);
        }
        (100000, 4150) => return track4::solve(challenge, &hp, save_solution),
        (100000, 4200) => {
            let hp5 = Hparams::for_t38().merge_user(hyperparameters.as_ref());
            return track5::solve(challenge, save_solution, &hp5);
        }
        _ => {}
    }

    let density = nc as f64 / nv as f64;
    if density >= 4.25 {
        if nv <= 5000 {
            return track1::solve(challenge, save_solution);
        }
        if nv <= 7500 {
            return track2::solve(challenge, &hp, save_solution);
        }
        let hp3 = track3_hyperparameters(hyperparameters);
        return track3::solve(challenge, save_solution, &hp3);
    }
    if density < 4.18 {
        return track4::solve(challenge, &hp, save_solution);
    }
    let hp5 = Hparams::for_t38().merge_user(hyperparameters.as_ref());
    track5::solve(challenge, save_solution, &hp5)
}
