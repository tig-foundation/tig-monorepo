use tig_challenges::satisfiability::*;
use serde_json::Map;
use serde_json::Value;

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

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let hp: Option<Hyperparameters> = hyperparameters.as_ref().and_then(|m| {
        serde_json::from_value(Value::Object(m.clone())).ok()
    });
    let nv = challenge.num_variables;
    let _ = save_solution(&Solution {
        variables: vec![false; nv],
    });
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
    // === SCHONING'S RANDOM WALK PRE-SOLVER ===
    // We'll use a bounded random walk: 3 * nv steps max.
    // Start with a random initial assignment.
    let mut vars = vec![false; nv];
    for v in 0..nv {
        vars[v] = rng.gen_bool(0.5);
    }
    // Build initial num_good and residual
    let mut num_good = vec![0u8; nc];
    let mut residual: Vec<u32> = Vec::with_capacity(nc);
    for i in 0..nc {
        let s = co[i] as usize;
        let e = co[i + 1] as usize;
        for j in s..e {
            let l = cl[j];
            let v = (l.abs() - 1) as usize;
            if (l > 0 && vars[v]) || (l < 0 && !vars[v]) {
                num_good[i] += 1;
            }
        }
        if num_good[i] == 0 {
            residual.push(i as u32);
        }
    }
    // If already solved, save and return.
    if residual.is_empty() {
        let _ = save_solution(&Solution {
            variables: vars,
        });
        return Ok(());
    }
    // Schoning walk: at most 3 * nv steps
    let max_steps = 3 * nv;
    let mut steps = 0;
    // We'll use the same CSR structures for incremental updates.
    unsafe {
        while steps < max_steps && !residual.is_empty() {
            // Pick a random unsatisfied clause
            let rid = rng.gen::<usize>() % residual.len();
            let cid = *residual.get_unchecked(rid) as usize;
            // Pick a random literal in this clause
            let cs = *co.get_unchecked(cid) as usize;
            let ce = *co.get_unchecked(cid + 1) as usize;
            let lit_idx = rng.gen::<usize>() % (ce - cs);
            let lit = *cl.get_unchecked(cs + lit_idx);
            let v = (lit.abs() - 1) as usize;
            // Flip the variable
            let was_true = *vars.get_unchecked(v);
            // Update clause satisfactions: for each clause containing v, update num_good
            let (is, ie) = if was_true {
                (*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
            } else {
                (*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
            };
            let (ds, de) = if was_true {
                (*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
            } else {
                (*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
            };
            // For clauses where v was contributing positively: they lose one good literal
            for k in is..ie {
                let c = *all_data.get_unchecked(k) as usize;
                *num_good.get_unchecked_mut(c) = num_good.get_unchecked(c).saturating_sub(1);
            }
            // For clauses where v was contributing negatively: they gain one good literal
            for k in ds..de {
                let c = *all_data.get_unchecked(k) as usize;
                *num_good.get_unchecked_mut(c) = num_good.get_unchecked(c).saturating_add(1);
            }
            // Update residual: for clauses that became satisfied or unsatisfied
            for k in is..ie {
                let c = *all_data.get_unchecked(k) as usize;
                if *num_good.get_unchecked(c) == 0 {
                    residual.push(c as u32);
                }
            }
            for k in ds..de {
                let c = *all_data.get_unchecked(k) as usize;
                if *num_good.get_unchecked(c) == 1 {
                    // Remove from residual if present
                    let pos = residual.iter().position(|&x| x == c as u32);
                    if let Some(p) = pos {
                        residual.swap_remove(p);
                    }
                }
            }
            // Flip the variable
            *vars.get_unchecked_mut(v) = !was_true;
            steps += 1;
            // Rebuild residual if empty? We don't need to — we update it incrementally.
            // But we must check if we have any unsatisfied clauses left.
            // We'll check at top of loop.
        }
    }
    // If Schoning walk found a solution, save and return.
    if residual.is_empty() {
        let _ = save_solution(&Solution {
            variables: vars,
        });
        return Ok(());
    }
    // === FALL THROUGH TO DENSITY-BASED TRACKS ===
    if density >= 4.25 {
        if nv <= 5000 {
            return track_t4::solve(&hp, &mut rng, nv, nc, density, p_cnt, n_cnt, &all_off, &p_bound, &all_data, &mut cl, &co, save_solution);
        }
        return track_high::solve(&hp, &mut rng, nv, nc, density, p_cnt, n_cnt, &all_off, &p_bound, &all_data, &mut cl, &co, save_solution);
    }
    track_low::solve(&hp, &mut rng, nv, nc, density, p_cnt, n_cnt, &all_off, &p_bound, &all_data, &mut cl, &co, save_solution)
}

mod track_t4;
mod track_high;
mod track_low;