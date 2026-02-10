use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::convert::TryInto;
use tig_challenges::satisfiability::*;

pub fn solve_low_density_impl(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    pad: f32,
    nad: f32,
) -> anyhow::Result<()> {
    let _ = save_solution(&Solution {
        variables: vec![false; challenge.num_variables],
    });
    let mut rng = SmallRng::seed_from_u64(u64::from_le_bytes(
        challenge.seed[..8].try_into().unwrap(),
    ));

    let num_variables = challenge.num_variables;
    let mut clauses = challenge.clauses.clone();
    let mut i = clauses.len();
    while i > 0 {
        i -= 1;
        let clause = &mut clauses[i];

        if clause.len() > 1 {
            let mut j = 1;
            while j < clause.len() {
                if clause[..j].contains(&clause[j]) {
                    clause.swap_remove(j);
                } else {
                    j += 1;
                }
            }
        }

        let mut is_tautology = false;
        for &lit in clause.iter() {
            if clause.contains(&-lit) {
                is_tautology = true;
                break;
            }
        }

        if is_tautology {
            clauses.swap_remove(i);
        }
    }

    let mut p_single = vec![false; num_variables];
    let mut n_single = vec![false; num_variables];
    let mut clauses_ = clauses;
    clauses = Vec::with_capacity(clauses_.len());
    let mut dead = false;

    while !dead {
        let mut done = true;
        for c in &clauses_ {
            let mut c_: Vec<i32> = Vec::with_capacity(c.len());
            let mut skip = false;
            for (ii, l) in c.iter().enumerate() {
                let v = (l.abs() as usize) - 1;
                if (p_single[v] && *l > 0) || (n_single[v] && *l < 0) || c[(ii + 1)..].contains(&-l)
                {
                    skip = true;
                    break;
                } else if p_single[v] || n_single[v] || c[(ii + 1)..].contains(l) {
                    done = false;
                    continue;
                } else {
                    c_.push(*l);
                }
            }
            if skip {
                done = false;
                continue;
            }
            match c_[..] {
                [l] => {
                    done = false;
                    let v = (l.abs() as usize) - 1;
                    if l > 0 {
                        if n_single[v] {
                            dead = true;
                            break;
                        } else {
                            p_single[v] = true;
                        }
                    } else if p_single[v] {
                        dead = true;
                        break;
                    } else {
                        n_single[v] = true;
                    }
                }
                [] => {
                    dead = true;
                    break;
                }
                _ => clauses.push(c_),
            }
        }
        if done {
            break;
        } else {
            clauses_ = clauses;
            clauses = Vec::with_capacity(clauses_.len());
        }
    }

    if dead {
        return Ok(());
    }

    let num_clauses = clauses.len();

    if num_clauses == 0 {
        let mut variables = vec![false; num_variables];
        for v in 0..num_variables {
            if p_single[v] {
                variables[v] = true;
            } else if n_single[v] {
                variables[v] = false;
            }
        }
        let _ = save_solution(&Solution { variables });
        return Ok(());
    }

    let mut p_cnt = vec![0u32; num_variables];
    let mut n_cnt = vec![0u32; num_variables];
    
    for c in clauses.iter() {
        for &l in c.iter() {
            let v = (l.abs() as usize) - 1;
            if l > 0 {
                p_cnt[v] += 1;
            } else {
                n_cnt[v] += 1;
            }
        }
    }
    
    let mut p_off = vec![0u32; num_variables + 1];
    let mut n_off = vec![0u32; num_variables + 1];
    for v in 0..num_variables {
        p_off[v + 1] = p_off[v] + p_cnt[v];
        n_off[v + 1] = n_off[v] + n_cnt[v];
    }
    
    let mut p_flat = vec![0u32; p_off[num_variables] as usize];
    let mut n_flat = vec![0u32; n_off[num_variables] as usize];
    
    let mut p_cur = p_off[..num_variables].to_vec();
    let mut n_cur = n_off[..num_variables].to_vec();
    
    for (cid, c) in clauses.iter().enumerate() {
        let cid_u32 = cid as u32;
        for &l in c.iter() {
            let v = (l.abs() as usize) - 1;
            if l > 0 {
                let idx = p_cur[v] as usize;
                p_flat[idx] = cid_u32;
                p_cur[v] += 1;
            } else {
                let idx = n_cur[v] as usize;
                n_flat[idx] = cid_u32;
                n_cur[v] += 1;
            }
        }
    }
    
    #[inline(always)]
    fn occ_slice<'a>(flat: &'a [u32], off: &[u32], v: usize) -> &'a [u32] {
        let s = off[v] as usize;
        let e = off[v + 1] as usize;
        &flat[s..e]
    }

    let mut variables = vec![false; num_variables];
    for v in 0..num_variables {
        let num_p = occ_slice(&p_flat, &p_off, v).len();
        let num_n = occ_slice(&n_flat, &n_off, v).len();

        let vad = if num_n == 0 {
            pad + 1.0
        } else {
            num_p as f32 / num_n as f32
        };

        if vad > pad {
            variables[v] = true;
        } else if vad < nad {
            variables[v] = false;
        } else {
            variables[v] = rng.gen_bool(0.5);
        }
    }

    let mut num_good_so_far: Vec<u8> = vec![0; num_clauses];
    for (i, c) in clauses.iter().enumerate() {
        for &l in c {
            let var = (l.abs() as usize) - 1;
            if l > 0 && variables[var] {
                num_good_so_far[i] += 1
            } else if l < 0 && !variables[var] {
                num_good_so_far[i] += 1
            }
        }
    }

    let mut residual_ = Vec::with_capacity(num_clauses);

    for (i, &num_good) in num_good_so_far.iter().enumerate() {
        if num_good == 0 {
            residual_.push(i);
        }
    }

    let current_prob = 0.52;

    unsafe {
        loop {
            if !residual_.is_empty() {
                let rand_val = rng.gen::<usize>();

                let mut i = residual_.len() - 1;
                while !residual_.is_empty() {
                    let id = rand_val % residual_.len();
                    i = residual_[id];
                    if num_good_so_far[i] > 0 {
                        residual_.swap_remove(id);
                    } else {
                        break;
                    }
                }
                if residual_.is_empty() {
                    break;
                }

                let c = clauses.get_unchecked_mut(i);

                if c.len() > 1 {
                    let random_index = rand_val % c.len();
                    c.swap(0, random_index);
                }

                let mut zero_found = None;
                'outer: for &l in c.iter() {
                    let abs_l = (l.abs() as usize) - 1;
                    let clauses_to_check = if *variables.get_unchecked(abs_l) {
                        occ_slice(&p_flat, &p_off, abs_l)
                    } else {
                        occ_slice(&n_flat, &n_off, abs_l)
                    };

                    for &c in clauses_to_check {
                        if *num_good_so_far.get_unchecked(c as usize) == 1 {
                            continue 'outer;
                        }
                    }
                    zero_found = Some(abs_l);
                    break;
                }

                let v = if let Some(abs_l) = zero_found {
                    abs_l
                } else if rand_val < (current_prob * (usize::MAX as f64)) as usize {
                    (c[0].abs() as usize) - 1
                } else {
                    let mut min_sad = usize::MAX;
                    let mut v_min_sad = (c[0].abs() as usize) - 1;
                    let mut min_weight = usize::MAX;

                    for &l in c.iter() {
                        let abs_l = (l.abs() as usize) - 1;
                        let clauses_to_check = if *variables.get_unchecked(abs_l) {
                            occ_slice(&p_flat, &p_off, abs_l)
                        } else {
                            occ_slice(&n_flat, &n_off, abs_l)
                        };

                        let mut sad = 0;

                        for &c_idx in clauses_to_check {
                            if *num_good_so_far.get_unchecked(c_idx as usize) == 1 {
                                sad += 1;
                            }
                            if sad >= min_sad {
                                break;
                            }
                        }

                        let appearances = occ_slice(&p_flat, &p_off, abs_l).len() + occ_slice(&n_flat, &n_off, abs_l).len();
                        let combined_weight = sad * 1000 + appearances;

                        if combined_weight < min_weight {
                            min_sad = sad;
                            min_weight = combined_weight;
                            v_min_sad = abs_l;
                        }

                        if min_sad <= 1 {
                            break;
                        }
                    }
                    v_min_sad
                };

                let was_true = *variables.get_unchecked(v);
                let clauses_to_decrement = if was_true {
                    occ_slice(&p_flat, &p_off, v)
                } else {
                    occ_slice(&n_flat, &n_off, v)
                };
                let clauses_to_increment = if was_true {
                    occ_slice(&n_flat, &n_off, v)
                } else {
                    occ_slice(&p_flat, &p_off, v)
                };

                for &cid in clauses_to_increment {
                    let num_good = num_good_so_far.get_unchecked_mut(cid as usize);
                    *num_good += 1;
                }

                for &cid in clauses_to_decrement {
                    let num_good = num_good_so_far.get_unchecked_mut(cid as usize);
                    *num_good -= 1;
                    if *num_good == 0 {
                        residual_.push(cid as usize);
                    }
                }

                *variables.get_unchecked_mut(v) = !was_true;
            } else {
                break;
            }
        }
    }

    for v in 0..num_variables {
        if p_single[v] {
            variables[v] = true;
        } else if n_single[v] {
            variables[v] = false;
        }
    }

    let _ = save_solution(&Solution { variables });
    Ok(())
}
