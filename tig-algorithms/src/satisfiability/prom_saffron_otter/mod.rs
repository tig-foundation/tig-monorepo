use tig_challenges::satisfiability::*;
use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

mod track_high;
mod track_t4;
mod track_low;

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

pub fn help() {
}

fn dpll_solve(nv: usize, cl: &[i32], co: &[u32], budget: &mut u64) -> Option<Option<Vec<bool>>> {
    let mut assignment = vec![-1i8; nv];
    match dpll_recursive(&mut assignment, cl, co, nv, budget) {
        Some(true) => Some(Some(assignment.into_iter().map(|v| v == 1).collect())),
        Some(false) => Some(None),
        None => None,
    }
}

fn dpll_recursive(
    assignment: &mut [i8],
    cl: &[i32],
    co: &[u32],
    nv: usize,
    budget: &mut u64,
) -> Option<bool> {
    if *budget == 0 {
        return None;
    }
    *budget -= 1;
    let nc = co.len() - 1;
    // Unit propagation
    loop {
        let mut changed = false;
        for i in 0..nc {
            let s = co[i] as usize;
            let e = co[i + 1] as usize;
            let mut unassigned_cnt = 0usize;
            let mut last_unassigned_lit = 0i32;
            let mut clause_sat = false;
            for j in s..e {
                let lit = cl[j];
                let v = (lit.abs() - 1) as usize;
                let val = assignment[v];
                if val == -1 {
                    unassigned_cnt += 1;
                    last_unassigned_lit = lit;
                } else {
                    let lit_true = if lit > 0 { val == 1 } else { val == 0 };
                    if lit_true {
                        clause_sat = true;
                        break;
                    }
                }
            }
            if clause_sat {
                continue;
            }
            if unassigned_cnt == 0 {
                return Some(false);
            }
            if unassigned_cnt == 1 {
                let v = (last_unassigned_lit.abs() - 1) as usize;
                let desired = if last_unassigned_lit > 0 { 1i8 } else { 0i8 };
                if assignment[v] == -1 {
                    assignment[v] = desired;
                    changed = true;
                } else if assignment[v] != desired {
                    return Some(false);
                }
            }
        }
        if !changed {
            break;
        }
    }

    let mut pos = vec![0u32; nv];
    let mut neg = vec![0u32; nv];
    let mut all_sat = true;
    for i in 0..nc {
        let s = co[i] as usize;
        let e = co[i + 1] as usize;
        let mut clause_sat = false;
        let mut unassigned_cnt = 0usize;
        for j in s..e {
            let lit = cl[j];
            let v = (lit.abs() - 1) as usize;
            let val = assignment[v];
            if val == -1 {
                unassigned_cnt += 1;
            } else {
                let lit_true = if lit > 0 { val == 1 } else { val == 0 };
                if lit_true {
                    clause_sat = true;
                    break;
                }
            }
        }
        if clause_sat {
            continue;
        }
        if unassigned_cnt == 0 {
            return Some(false);
        }
        all_sat = false;
        for j in s..e {
            let lit = cl[j];
            let v = (lit.abs() - 1) as usize;
            if assignment[v] == -1 {
                if lit > 0 {
                    pos[v] += 1;
                } else {
                    neg[v] += 1;
                }
            }
        }
    }
    if all_sat {
        return Some(true);
    }

    let mut changed = false;
    for v in 0..nv {
        if assignment[v] == -1 {
            if pos[v] > 0 && neg[v] == 0 {
                assignment[v] = 1;
                changed = true;
            } else if neg[v] > 0 && pos[v] == 0 {
                assignment[v] = 0;
                changed = true;
            }
        }
    }
    if changed {
        return dpll_recursive(assignment, cl, co, nv, budget);
    }

    let mut best_v = 0usize;
    let mut best_score = 0u32;
    let mut found = false;
    for v in 0..nv {
        if assignment[v] == -1 {
            let score = pos[v] + neg[v];
            if !found || score > best_score {
                found = true;
                best_v = v;
                best_score = score;
            }
        }
    }
    if !found {
        return Some(true);
    }

    assignment[best_v] = 1;
    match dpll_recursive(assignment, cl, co, nv, budget) {
        Some(true) => return Some(true),
        None => return None,
        Some(false) => {}
    }
    assignment[best_v] = 0;
    match dpll_recursive(assignment, cl, co, nv, budget) {
        Some(true) => return Some(true),
        None => return None,
        Some(false) => {}
    }
    assignment[best_v] = -1;
    Some(false)
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
        if a == -b || a == -c || b == -c {
            continue;
        }
        good_clauses += 1;
        let va = (a.abs() - 1) as usize;
        if a > 0 {
            p_cnt[va] += 1;
        } else {
            n_cnt[va] += 1;
        }
        if b != a {
            let vb = (b.abs() - 1) as usize;
            if b > 0 {
                p_cnt[vb] += 1;
            } else {
                n_cnt[vb] += 1;
            }
        }
        if c != a && c != b {
            let vc = (c.abs() - 1) as usize;
            if c > 0 {
                p_cnt[vc] += 1;
            } else {
                n_cnt[vc] += 1;
            }
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
            if a == -b || a == -c || b == -c {
                continue;
            }
            let va = (a.abs() - 1) as usize;
            if a > 0 {
                all_data[p_pos[va] as usize] = ci;
                p_pos[va] += 1;
            } else {
                all_data[n_pos[va] as usize] = ci;
                n_pos[va] += 1;
            }
            if b != a {
                let vb = (b.abs() - 1) as usize;
                if b > 0 {
                    all_data[p_pos[vb] as usize] = ci;
                    p_pos[vb] += 1;
                } else {
                    all_data[n_pos[vb] as usize] = ci;
                    n_pos[vb] += 1;
                }
            }
            if c != a && c != b {
                let vc = (c.abs() - 1) as usize;
                if c > 0 {
                    all_data[p_pos[vc] as usize] = ci;
                    p_pos[vc] += 1;
                } else {
                    all_data[n_pos[vc] as usize] = ci;
                    n_pos[vc] += 1;
                }
            }
            cl.push(a);
            if b != a {
                cl.push(b);
            }
            if c != a && c != b {
                cl.push(c);
            }
            co.push(cl.len() as u32);
            ci += 1;
        }
    }

    let density = nc as f64 / nv as f64;

    if nv <= 200 {
        let mut dpll_budget: u64 = 2_000_000;
        match dpll_solve(nv, &cl, &co, &mut dpll_budget) {
            Some(Some(assignment)) => {
                let _ = save_solution(&Solution { variables: assignment });
                return Ok(());
            }
            Some(None) => {
                return Ok(());
            }
            None => {
                return track_t4::solve(
                    &hp, &mut rng, nv, nc, density, p_cnt, n_cnt, &all_off, &p_bound, &all_data, &mut cl,
                    &co, save_solution,
                );
            }
        }
    }

    if density >= 4.25 {
        if nv <= 5000 {
            return track_t4::solve(
                &hp, &mut rng, nv, nc, density, p_cnt, n_cnt, &all_off, &p_bound, &all_data, &mut cl,
                &co, save_solution,
            );
        }
        return track_high::solve(
            &hp, &mut rng, nv, nc, density, p_cnt, n_cnt, &all_off, &p_bound, &all_data, &mut cl,
            &co, save_solution,
        );
    }

    track_low::solve(
        &hp, &mut rng, nv, nc, density, p_cnt, n_cnt, &all_off, &p_bound, &all_data, &mut cl,
        &co, save_solution,
    )
}