// SAT solver engine F — T5 (nv=7500, nc=32002, α=4.267 = seuil SAT-UNSAT exact)
//
// médiane n=3 same-host 341 540 ms (metal-cpu-07) — DOMINANT-2-AXES vs le SOTA
//
// dominé sur LES DEUX axes par toute la lignée v6→v9. Le gap n'est pas un
// réglage : c'est une autre famille de moteur (WalkSAT break-only + warm-restart
// PERTURB_K vs probSAT-poly v3).
//
//   moteur == v6 SOTA : restart warm (RESTART_PERIOD=80M, REINIT_MIN_UNSAT=30,
//                       PERTURB_K=100 Fisher-Yates partiel), WALK_P=0.52,
//                       init polarity-biased clamp(0.2,0.8), SmallRng seed challenge
//   i20 branchless break-count (byte-identique)
//   i21 hoist de `new_val` hors des 2 boucles RMW (byte-identique)
//   ⇒ composite i22 = −8.6 % temps vs v6 à Q strictement identique.
//
// que 2 `rng.gen_range` supplémentaires par flip suffisent à détruire les 3
// nonces SAT de t5 (α=4.267 = seuil exact, trajectoire stochastique critique).
// Seules adaptations : `super::Prepared` / `super::preprocess` / `super::Hparams`
// sont internalisés ici pour que le fichier soit auto-suffisant dans l'archi
// `engine_*` de sat_hybrid, et un wrapper `solve()` parse la Map JSON des HP.
//
// = None ⇒ 155B (`unwrap_or`), soit max_flips = (155e9 − 50e6)/200 = 774 750 000.
// On ne bake DONC RIEN côté mod.rs : toute déviation de fuel change le nombre de
// flips et donc l'issue des nonces borderline (Hindsight `3d11b544` : −25 % de Q
// sur t1 en baissant le fuel). `target_max_fuel=200B` de l'ancien engine_c est un
// HP d'un AUTRE moteur (formule de fuel différente) — il ne se transpose pas.
use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Hparams {
    pub max_fuel_high: Option<f64>,
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

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let hp: Hparams = hyperparameters
        .as_ref()
        .and_then(|m| serde_json::from_value::<Hparams>(Value::Object(m.clone())).ok())
        .unwrap_or_default();
    solve_t5(challenge, save_solution, &hp)
}

fn solve_t5(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hp: &Hparams,
) -> Result<()> {
    let Prepared {
        mut rng,
        nv,
        nc,
        density: _density,
        p_cnt,
        n_cnt,
        all_off,
        p_bound,
        all_data,
        mut cl,
        co,
    } = preprocess(challenge, save_solution);

    let max_fuel = hp.max_fuel_high.unwrap_or(155_000_000_000.0);
    let base_fuel = 50_000_000.0;
    let flip_fuel = 200.0;
    let max_flips = if max_fuel > base_fuel {
        ((max_fuel - base_fuel) / flip_fuel) as u64
    } else {
        10_000_000
    };

    const WALK_P: f64 = 0.52;
    const RESTART_PERIOD: u64 = 80_000_000;
    const REINIT_MIN_UNSAT: usize = 30;
    const PERTURB_K: usize = 100;

    let mut assignment = vec![false; nv];
    let mut best_assignment = vec![false; nv];
    let mut true_lit_count = vec![0u8; nc];
    let mut unsat_clauses: Vec<usize> = Vec::with_capacity(nc);
    let mut clause_pos_in_unsat = vec![usize::MAX; nc];

    let mut perturb_indices: Vec<usize> = (0..nv).collect();

    for v in 0..nv {
        let pc = p_cnt[v] as f64;
        let nc_v = n_cnt[v] as f64;
        let total = pc + nc_v;
        if total == 0.0 {
            assignment[v] = rng.gen_bool(0.5);
        } else {
            assignment[v] = rng.gen_bool((pc / total).clamp(0.2, 0.8));
        }
    }

    for c in 0..nc {
        let off = co[c] as usize;
        let end = co[c + 1] as usize;
        let mut trues = 0u8;
        for i in off..end {
            let lit = cl[i];
            let v = (lit.abs() - 1) as usize;
            if assignment[v] == (lit > 0) { trues += 1; }
        }
        true_lit_count[c] = trues;
        if trues == 0 {
            clause_pos_in_unsat[c] = unsat_clauses.len();
            unsat_clauses.push(c);
        }
    }

    let mut best_unsat = unsat_clauses.len();
    best_assignment.copy_from_slice(&assignment);
    let _ = save_solution(&Solution { variables: best_assignment.clone() });

    let mut period_best_unsat = best_unsat;
    let mut var_age = vec![0u8; nv];
    let mut global_flips: u64 = 0;

    unsafe {
        loop {
            if global_flips >= max_flips { break; }
            if unsat_clauses.is_empty() { break; }

            if global_flips > 0 && global_flips % RESTART_PERIOD == 0 {
                if period_best_unsat >= REINIT_MIN_UNSAT {

                    // Warm restart (v6 SOTA operator): restart from the best assignment
                    // and perturb only PERTURB_K vars via partial Fisher-Yates, instead of a
                    // cold full-random reinit that discards all progress. Reconverges the
                    // calibrated nonces in far fewer flips -> less fuel burned -> less time.
                    assignment.copy_from_slice(&best_assignment);

                    for i in 0..PERTURB_K {
                        let j = rng.gen_range(i..nv);
                        perturb_indices.swap(i, j);
                        let v = *perturb_indices.get_unchecked(i);
                        let cur = *assignment.get_unchecked(v);
                        *assignment.get_unchecked_mut(v) = !cur;
                    }

                    true_lit_count.fill(0);
                    unsat_clauses.clear();
                    clause_pos_in_unsat.fill(usize::MAX);
                    for c in 0..nc {
                        let off = *co.get_unchecked(c) as usize;
                        let end = *co.get_unchecked(c + 1) as usize;
                        let mut trues = 0u8;
                        for i in off..end {
                            let lit = *cl.get_unchecked(i);
                            let v = (lit.abs() - 1) as usize;
                            if *assignment.get_unchecked(v) == (lit > 0) { trues += 1; }
                        }
                        *true_lit_count.get_unchecked_mut(c) = trues;
                        if trues == 0 {
                            *clause_pos_in_unsat.get_unchecked_mut(c) = unsat_clauses.len();
                            unsat_clauses.push(c);
                        }
                    }

                    var_age.fill(0);

                    let cur = unsat_clauses.len();
                    if cur < best_unsat {
                        best_unsat = cur;
                        best_assignment.copy_from_slice(&assignment);
                        let _ = save_solution(&Solution { variables: best_assignment.clone() });
                    }
                }
                period_best_unsat = unsat_clauses.len();
            }

            let cur_unsat = unsat_clauses.len();
            if cur_unsat < period_best_unsat { period_best_unsat = cur_unsat; }

            global_flips += 1;

            let r_idx = rng.gen_range(0..unsat_clauses.len());
            let c = *unsat_clauses.get_unchecked(r_idx);
            let off = *co.get_unchecked(c) as usize;
            let end = *co.get_unchecked(c + 1) as usize;
            let len = end - off;

            if len > 1 {
                let ri = (global_flips as usize) % len;
                cl.swap(off, off + ri);
            }

            let mut picked_v = usize::MAX;
            let mut vars = [0usize; 3];
            let mut breaks = [0u8; 3];
            let mut nvars = 0;

            for i in off..end {
                let lit = *cl.get_unchecked(i);
                let v = (lit.abs() - 1) as usize;
                *vars.get_unchecked_mut(nvars) = v;

                let val = *assignment.get_unchecked(v);
                let (start, stop) = if val {
                    (*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
                } else {
                    (*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
                };

                let mut b = 0u8;
                for k in start..stop {
                    b += (*true_lit_count.get_unchecked(*all_data.get_unchecked(k) as usize) == 1) as u8;
                }
                *breaks.get_unchecked_mut(nvars) = b;
                if b == 0 {
                    picked_v = v;
                    break;
                }
                nvars += 1;
            }

            if picked_v == usize::MAX {
                if nvars == 0 {
                    if let Some(&lit) = cl.get(off) {
                        picked_v = (lit.abs() - 1) as usize;
                    }
                } else if rng.gen_bool(WALK_P) {
                    picked_v = *vars.get_unchecked(rng.gen_range(0..nvars));
                } else {
                    let mut min_b = u8::MAX;
                    let mut best_idx = 0;
                    for i in 0..nvars {
                        let b = *breaks.get_unchecked(i);
                        let vi = *vars.get_unchecked(i);
                        let vb = *vars.get_unchecked(best_idx);
                        if b < min_b || (b == min_b && *var_age.get_unchecked(vi) < *var_age.get_unchecked(vb)) {
                            min_b = b;
                            best_idx = i;
                        }
                    }
                    picked_v = *vars.get_unchecked(best_idx);
                }
            }

            if picked_v == usize::MAX { continue; }

            let new_val = !*assignment.get_unchecked(picked_v);
            *assignment.get_unchecked_mut(picked_v) = new_val;

            // i21 (loop-unswitching / branch hoist, dir 015550Z): `new_val` is invariant for
            // the entire flip, yet the original code tested `if new_val { incr } else { decr }`
            // on EVERY occurrence (~18 clauses/flip * 2 loops). Hoist the test OUT of each loop.
            // BYTE-IDENTICAL: same true_lit_count writes, same unsat_clauses ops, same order ->
            // identical trajectory -> Q=93750 guaranteed. Composes over i20 branchless break-count.
            let p_start = *all_off.get_unchecked(picked_v) as usize;
            let p_end = *p_bound.get_unchecked(picked_v) as usize;
            if new_val {
                for k in p_start..p_end {
                    let c_idx = *all_data.get_unchecked(k) as usize;
                    let trues = *true_lit_count.get_unchecked(c_idx);
                    *true_lit_count.get_unchecked_mut(c_idx) = trues + 1;
                    if trues == 0 {
                        let last_c = unsat_clauses.pop().unwrap_unchecked();
                        let pos = *clause_pos_in_unsat.get_unchecked(c_idx);
                        *clause_pos_in_unsat.get_unchecked_mut(c_idx) = usize::MAX;
                        if last_c != c_idx {
                            *unsat_clauses.get_unchecked_mut(pos) = last_c;
                            *clause_pos_in_unsat.get_unchecked_mut(last_c) = pos;
                        }
                    }
                }
            } else {
                for k in p_start..p_end {
                    let c_idx = *all_data.get_unchecked(k) as usize;
                    let trues = *true_lit_count.get_unchecked(c_idx);
                    *true_lit_count.get_unchecked_mut(c_idx) = trues - 1;
                    if trues == 1 {
                        *clause_pos_in_unsat.get_unchecked_mut(c_idx) = unsat_clauses.len();
                        unsat_clauses.push(c_idx);
                    }
                }
            }

            let n_start = *p_bound.get_unchecked(picked_v) as usize;
            let n_end = *all_off.get_unchecked(picked_v + 1) as usize;
            if !new_val {
                for k in n_start..n_end {
                    let c_idx = *all_data.get_unchecked(k) as usize;
                    let trues = *true_lit_count.get_unchecked(c_idx);
                    *true_lit_count.get_unchecked_mut(c_idx) = trues + 1;
                    if trues == 0 {
                        let last_c = unsat_clauses.pop().unwrap_unchecked();
                        let pos = *clause_pos_in_unsat.get_unchecked(c_idx);
                        *clause_pos_in_unsat.get_unchecked_mut(c_idx) = usize::MAX;
                        if last_c != c_idx {
                            *unsat_clauses.get_unchecked_mut(pos) = last_c;
                            *clause_pos_in_unsat.get_unchecked_mut(last_c) = pos;
                        }
                    }
                }
            } else {
                for k in n_start..n_end {
                    let c_idx = *all_data.get_unchecked(k) as usize;
                    let trues = *true_lit_count.get_unchecked(c_idx);
                    *true_lit_count.get_unchecked_mut(c_idx) = trues - 1;
                    if trues == 1 {
                        *clause_pos_in_unsat.get_unchecked_mut(c_idx) = unsat_clauses.len();
                        unsat_clauses.push(c_idx);
                    }
                }
            }

            for i in off..end {
                let lit = *cl.get_unchecked(i);
                let v = (lit.abs() - 1) as usize;
                let a = *var_age.get_unchecked(v);
                *var_age.get_unchecked_mut(v) = a.saturating_add(1);
            }

            let cur_unsat = unsat_clauses.len();
            if cur_unsat < period_best_unsat { period_best_unsat = cur_unsat; }
            if cur_unsat < best_unsat {
                best_unsat = cur_unsat;
                best_assignment.copy_from_slice(&assignment);
                let _ = save_solution(&Solution { variables: best_assignment.clone() });
                if cur_unsat == 0 { break; }
            }
        }
    }

    let _ = save_solution(&Solution { variables: best_assignment });
    Ok(())
}
