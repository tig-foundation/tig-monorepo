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

    #[inline(always)]
    fn var_index_from_lit(l: i32) -> usize {
        let a = if l < 0 { -l } else { l };
        (a as usize) - 1
    }

    let mut seen_pos = vec![0u32; num_variables];
    let mut seen_neg = vec![0u32; num_variables];
    let mut stamp: u32 = 1;

    let mut i = clauses.len();
    while i > 0 {
        i -= 1;
        let clause = &mut clauses[i];

        if clause.len() > 1 {
            if stamp == u32::MAX {
                seen_pos.fill(0);
                seen_neg.fill(0);
                stamp = 1;
            }
            stamp += 1;

            let l0 = clause[0];
            let v0 = var_index_from_lit(l0);
            if l0 > 0 {
                seen_pos[v0] = stamp;
            } else {
                seen_neg[v0] = stamp;
            }

            let mut is_tautology = false;

            let mut j = 1;
            while j < clause.len() {
                let l = clause[j];
                let v = var_index_from_lit(l);

                if l > 0 {
                    if seen_pos[v] == stamp {
                        clause.swap_remove(j);
                        continue;
                    }
                    if seen_neg[v] == stamp {
                        is_tautology = true;
                        break;
                    }
                    seen_pos[v] = stamp;
                } else {
                    if seen_neg[v] == stamp {
                        clause.swap_remove(j);
                        continue;
                    }
                    if seen_pos[v] == stamp {
                        is_tautology = true;
                        break;
                    }
                    seen_neg[v] = stamp;
                }

                j += 1;
            }

            if is_tautology {
                clauses.swap_remove(i);
            }
        }
    }

    let mut p_single = vec![false; num_variables];
    let mut n_single = vec![false; num_variables];
    let mut clauses_ = clauses;
    clauses = Vec::with_capacity(clauses_.len());
    let mut dead = false;

    let mut c_scratch: Vec<i32> = Vec::new();

    while !dead {
        let mut done = true;
        for c in &clauses_ {
            c_scratch.clear();
            c_scratch.reserve(c.len());

            let mut skip = false;
            for (ii, l) in c.iter().enumerate() {
                let v = var_index_from_lit(*l);
                if (p_single[v] && *l > 0) || (n_single[v] && *l < 0) || c[(ii + 1)..].contains(&-l)
                {
                    skip = true;
                    break;
                } else if p_single[v] || n_single[v] || c[(ii + 1)..].contains(l) {
                    done = false;
                    continue;
                } else {
                    c_scratch.push(*l);
                }
            }
            if skip {
                done = false;
                continue;
            }

            if c_scratch.len() == 1 {
                done = false;
                let l = c_scratch[0];
                let v = var_index_from_lit(l);
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
            } else if c_scratch.is_empty() {
                dead = true;
                break;
            } else {
                let mut c_out: Vec<i32> = Vec::with_capacity(c_scratch.len());
                c_out.extend_from_slice(&c_scratch);
                clauses.push(c_out);
            }
        }
        if done {
            break;
        } else {
            clauses_.clear();
            clauses_.append(&mut clauses);
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
            let v = var_index_from_lit(l);
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

    let total_lits: usize = clauses.iter().map(|c| c.len()).sum();
    let mut cl: Vec<i32> = Vec::with_capacity(total_lits);
    let mut co: Vec<u32> = Vec::with_capacity(num_clauses + 1);
    co.push(0);

    for (cid, c) in clauses.iter().enumerate() {
        let cid_u32 = cid as u32;
        for &l in c.iter() {
            let v = var_index_from_lit(l);
            if l > 0 {
                let idx = p_cur[v] as usize;
                p_flat[idx] = cid_u32;
                p_cur[v] += 1;
            } else {
                let idx = n_cur[v] as usize;
                n_flat[idx] = cid_u32;
                n_cur[v] += 1;
            }
            cl.push(l);
        }
        co.push(cl.len() as u32);
    }

    drop(clauses);

    let mut variables = vec![false; num_variables];
    for v in 0..num_variables {
        let num_p = (p_off[v + 1] - p_off[v]) as usize;
        let num_n = (n_off[v + 1] - n_off[v]) as usize;

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

    let ng_len = (num_clauses + 3) >> 2;
    let mut num_good_so_far: Vec<u8> = vec![0u8; ng_len];

    for i in 0..num_clauses {
        let s = co[i] as usize;
        let e = co[i + 1] as usize;
        let shift = (i & 3) << 1;
        let byte_idx = i >> 2;
        for j in s..e {
            let l = cl[j];
            let var = var_index_from_lit(l);
            if (l > 0) == variables[var] {
                num_good_so_far[byte_idx] += 1u8 << shift;
            }
        }
    }

    let mut residual_: Vec<usize> = Vec::with_capacity(num_clauses);
    for i in 0..num_clauses {
        if (num_good_so_far[i >> 2] >> ((i & 3) << 1)) & 3 == 0 {
            residual_.push(i);
        }
    }

    let current_prob: f64 = 0.52;

    unsafe {
        loop {
            if !residual_.is_empty() {
                let rand_val = rng.gen::<usize>();

                while !residual_.is_empty() {
                    let id = rand_val % residual_.len();
                    let candidate = *residual_.get_unchecked(id);
                    if (num_good_so_far.get_unchecked(candidate >> 2) >> ((candidate & 3) << 1)) & 3 > 0 {
                        residual_.swap_remove(id);
                    } else {
                        break;
                    }
                }
                if residual_.is_empty() {
                    break;
                }

                let i = *residual_.get_unchecked(rand_val % residual_.len());

                let cs = *co.get_unchecked(i) as usize;
                let ce = *co.get_unchecked(i + 1) as usize;
                let clen = ce - cs;

                if clen > 1 {
                    let random_index = rand_val % clen;
                    cl.swap(cs, cs + random_index);
                }

                let mut zero_found = None;
                'outer: for j in cs..ce {
                    let l = *cl.get_unchecked(j);
                    let abs_l = var_index_from_lit(l);

                    let p_s = *p_off.get_unchecked(abs_l) as usize;
                    let p_e = *p_off.get_unchecked(abs_l + 1) as usize;
                    let n_s = *n_off.get_unchecked(abs_l) as usize;
                    let n_e = *n_off.get_unchecked(abs_l + 1) as usize;

                    let clauses_to_check = if *variables.get_unchecked(abs_l) {
                        p_flat.get_unchecked(p_s..p_e)
                    } else {
                        n_flat.get_unchecked(n_s..n_e)
                    };

                    for &cid_u32 in clauses_to_check {
                        let c = cid_u32 as usize;
                        if (*num_good_so_far.get_unchecked(c >> 2) >> ((c & 3) << 1)) & 3 == 1 {
                            continue 'outer;
                        }
                    }
                    zero_found = Some(abs_l);
                    break;
                }

                let v = if let Some(abs_l) = zero_found {
                    abs_l
                } else if (rand_val as f64) < (current_prob * (usize::MAX as f64)) {
                    var_index_from_lit(*cl.get_unchecked(cs))
                } else {
                    let mut min_sad = usize::MAX;
                    let mut v_min_sad = var_index_from_lit(*cl.get_unchecked(cs));
                    let mut min_weight = usize::MAX;

                    for j in cs..ce {
                        let l = *cl.get_unchecked(j);
                        let abs_l = var_index_from_lit(l);

                        let p_s = *p_off.get_unchecked(abs_l) as usize;
                        let p_e = *p_off.get_unchecked(abs_l + 1) as usize;
                        let n_s = *n_off.get_unchecked(abs_l) as usize;
                        let n_e = *n_off.get_unchecked(abs_l + 1) as usize;

                        let clauses_to_check = if *variables.get_unchecked(abs_l) {
                            p_flat.get_unchecked(p_s..p_e)
                        } else {
                            n_flat.get_unchecked(n_s..n_e)
                        };

                        let mut sad: usize = 0;

                        for &cid_u32 in clauses_to_check {
                            let c = cid_u32 as usize;
                            if (*num_good_so_far.get_unchecked(c >> 2) >> ((c & 3) << 1)) & 3 == 1 {
                                sad += 1;
                            }
                            if sad >= min_sad {
                                break;
                            }
                        }

                        let appearances = (p_e - p_s) + (n_e - n_s);
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

                let (dec_flat, dec_off, inc_flat, inc_off) = if was_true {
                    (&p_flat, &p_off, &n_flat, &n_off)
                } else {
                    (&n_flat, &n_off, &p_flat, &p_off)
                };

                let s_inc = *inc_off.get_unchecked(v) as usize;
                let e_inc = *inc_off.get_unchecked(v + 1) as usize;
                let clauses_to_increment = inc_flat.get_unchecked(s_inc..e_inc);

                let s_dec = *dec_off.get_unchecked(v) as usize;
                let e_dec = *dec_off.get_unchecked(v + 1) as usize;
                let clauses_to_decrement = dec_flat.get_unchecked(s_dec..e_dec);

                let ng_ptr = num_good_so_far.as_mut_ptr();

                for &cid_u32 in clauses_to_increment {
                    let cu = cid_u32 as usize;
                    let shift = (cu & 3) << 1;
                    let byte_idx = cu >> 2;
                    *ng_ptr.add(byte_idx) += 1u8 << shift;
                }

                for &cid_u32 in clauses_to_decrement {
                    let cu = cid_u32 as usize;
                    let shift = (cu & 3) << 1;
                    let byte_idx = cu >> 2;
                    let p = ng_ptr.add(byte_idx);
                    let before = (*p >> shift) & 3;
                    *p -= 1u8 << shift;
                    if before == 1 {
                        residual_.push(cu);
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
