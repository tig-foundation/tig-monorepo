const ABSENT: u32 = u32::MAX;

#[inline(always)]
fn lit_var_index(lit: i32) -> usize {
    if lit > 0 {
        lit as usize - 1
    } else {
        (-lit) as usize - 1
    }
}

#[inline(always)]
fn lit_good_xor(lit: i32, vars: &[bool]) -> (u8, u32) {
    let v = lit_var_index(lit);
    if (lit > 0 && vars[v]) || (lit < 0 && !vars[v]) {
        (1, v as u32)
    } else {
        (0, 0)
    }
}

#[inline(always)]
fn clause_good_xor_count(s: usize, e: usize, cl: &[i32], vars: &[bool]) -> (u8, u32) {
    match e - s {
        1 => lit_good_xor(cl[s], vars),
        2 => {
            let (g0, x0) = lit_good_xor(cl[s], vars);
            let (g1, x1) = lit_good_xor(cl[s + 1], vars);
            (g0 + g1, x0 ^ x1)
        }
        3 => {
            let (g0, x0) = lit_good_xor(cl[s], vars);
            let (g1, x1) = lit_good_xor(cl[s + 1], vars);
            let (g2, x2) = lit_good_xor(cl[s + 2], vars);
            (g0 + g1 + g2, x0 ^ x1 ^ x2)
        }
        _ => {
            let mut good = 0u8;
            let mut xor = 0u32;
            for &lit in &cl[s..e] {
                let (g, x) = lit_good_xor(lit, vars);
                good += g;
                xor ^= x;
            }
            (good, xor)
        }
    }
}

#[inline(always)]
fn natural_three_clause_layout(nc: usize, co: &[u32], cl: &[i32]) -> bool {
    if cl.len() != nc.saturating_mul(3) {
        return false;
    }
    debug_assert!(co.len() > nc);
    debug_assert_eq!(co[0], 0);
    debug_assert_eq!(co[nc] as usize, cl.len());
    debug_assert!((0..=nc).all(|i| co[i] as usize == i * 3));
    true
}

#[inline(always)]
fn clause3_good_xor_count(s: usize, cl: &[i32], vars: &[bool]) -> (u8, u32) {
    let (g0, x0) = lit_good_xor(cl[s], vars);
    let (g1, x1) = lit_good_xor(cl[s + 1], vars);
    let (g2, x2) = lit_good_xor(cl[s + 2], vars);
    (g0 + g1 + g2, x0 ^ x1 ^ x2)
}

#[inline(always)]
fn clause3_good_count(s: usize, cl: &[i32], vars: &[bool]) -> u8 {
    let (g0, _) = lit_good_xor(cl[s], vars);
    let (g1, _) = lit_good_xor(cl[s + 1], vars);
    let (g2, _) = lit_good_xor(cl[s + 2], vars);
    g0 + g1 + g2
}

#[inline(always)]
unsafe fn inc_break(score: &mut [u16], v: usize) {
    debug_assert!(*score.get_unchecked(v) < u16::MAX);
    *score.get_unchecked_mut(v) += 1;
}

#[inline(always)]
unsafe fn dec_break(score: &mut [u16], v: usize) {
    debug_assert!(*score.get_unchecked(v) > 0);
    *score.get_unchecked_mut(v) -= 1;
}

#[inline(always)]
unsafe fn inc_make(score: &mut [u16], v: usize) {
    debug_assert!(*score.get_unchecked(v) < u16::MAX);
    *score.get_unchecked_mut(v) += 1;
}

#[inline(always)]
unsafe fn dec_make(score: &mut [u16], v: usize) {
    debug_assert!(*score.get_unchecked(v) > 0);
    *score.get_unchecked_mut(v) -= 1;
}

#[inline(always)]
unsafe fn add_make_for_clause(co: &[u32], cl: &[i32], make_score: &mut [u16], c: usize) {
    let s = *co.get_unchecked(c) as usize;
    let e = *co.get_unchecked(c + 1) as usize;
    match e - s {
        1 => add_make_lit(make_score, *cl.get_unchecked(s)),
        2 => {
            add_make_lit(make_score, *cl.get_unchecked(s));
            add_make_lit(make_score, *cl.get_unchecked(s + 1));
        }
        3 => {
            add_make_lit(make_score, *cl.get_unchecked(s));
            add_make_lit(make_score, *cl.get_unchecked(s + 1));
            add_make_lit(make_score, *cl.get_unchecked(s + 2));
        }
        _ => {
            for i in s..e {
                add_make_lit(make_score, *cl.get_unchecked(i));
            }
        }
    }
}

#[inline(always)]
unsafe fn remove_make_for_clause(co: &[u32], cl: &[i32], make_score: &mut [u16], c: usize) {
    let s = *co.get_unchecked(c) as usize;
    let e = *co.get_unchecked(c + 1) as usize;
    match e - s {
        1 => remove_make_lit(make_score, *cl.get_unchecked(s)),
        2 => {
            remove_make_lit(make_score, *cl.get_unchecked(s));
            remove_make_lit(make_score, *cl.get_unchecked(s + 1));
        }
        3 => {
            remove_make_lit(make_score, *cl.get_unchecked(s));
            remove_make_lit(make_score, *cl.get_unchecked(s + 1));
            remove_make_lit(make_score, *cl.get_unchecked(s + 2));
        }
        _ => {
            for i in s..e {
                remove_make_lit(make_score, *cl.get_unchecked(i));
            }
        }
    }
}

#[inline(always)]
unsafe fn add_make_for_clause3(cl: &[i32], make_score: &mut [u16], s: usize) {
    add_make_lit(make_score, *cl.get_unchecked(s));
    add_make_lit(make_score, *cl.get_unchecked(s + 1));
    add_make_lit(make_score, *cl.get_unchecked(s + 2));
}

#[inline(always)]
unsafe fn remove_make_for_clause3(cl: &[i32], make_score: &mut [u16], s: usize) {
    remove_make_lit(make_score, *cl.get_unchecked(s));
    remove_make_lit(make_score, *cl.get_unchecked(s + 1));
    remove_make_lit(make_score, *cl.get_unchecked(s + 2));
}

#[inline(always)]
unsafe fn add_make_lit(make_score: &mut [u16], lit: i32) {
    inc_make(make_score, lit_var_index(lit));
}

#[inline(always)]
unsafe fn remove_make_lit(make_score: &mut [u16], lit: i32) {
    dec_make(make_score, lit_var_index(lit));
}

#[inline(always)]
pub(crate) unsafe fn remove_unsat_exact(unsat: &mut Vec<u32>, unsat_pos: &mut [u32], c: usize) {
    let pos = *unsat_pos.get_unchecked(c);
    if pos == ABSENT {
        return;
    }
    let pos = pos as usize;
    let last_idx = unsat.len() - 1;
    if pos != last_idx {
        let last_c = *unsat.get_unchecked(last_idx) as usize;
        *unsat.get_unchecked_mut(pos) = last_c as u32;
        *unsat_pos.get_unchecked_mut(last_c) = pos as u32;
    }
    unsat.set_len(last_idx);
    *unsat_pos.get_unchecked_mut(c) = ABSENT;
}

#[inline(always)]
unsafe fn add_unsat_exact(unsat: &mut Vec<u32>, unsat_pos: &mut [u32], c: usize) {
    if *unsat_pos.get_unchecked(c) == ABSENT {
        *unsat_pos.get_unchecked_mut(c) = unsat.len() as u32;
        unsat.push(c as u32);
    }
}

fn clear_exact_unsat_positions(unsat: &[u32], unsat_pos: &mut [u32]) {
    debug_assert!(unsat
        .iter()
        .enumerate()
        .all(|(idx, &cid)| unsat_pos.get(cid as usize).copied() == Some(idx as u32)));
    for &cid in unsat {
        unsat_pos[cid as usize] = ABSENT;
    }
    debug_assert!(unsat_pos.iter().all(|&pos| pos == ABSENT));
}

fn clear_exact_make_scores(
    co: &[u32],
    cl: &[i32],
    make_score: &mut [u16],
    unsat: &[u32],
    all_three_clauses: bool,
) {
    if unsat.len().saturating_mul(8) <= make_score.len() {
        unsafe {
            for &cid in unsat {
                if all_three_clauses {
                    remove_make_for_clause3(cl, make_score, cid as usize * 3);
                } else {
                    remove_make_for_clause(co, cl, make_score, cid as usize);
                }
            }
        }
    } else {
        make_score.fill(0);
    }
    debug_assert!(make_score.iter().all(|&score| score == 0));
}

fn rebuild_u8_exact_impl<const CLEAR_SIDE_CARS: bool>(
    nc: usize,
    co: &[u32],
    cl: &[i32],
    vars: &[bool],
    num_good: &mut [u8],
    unsat: &mut Vec<u32>,
    unsat_pos: &mut [u32],
    break_score: &mut [u16],
    sat_xor: &mut [u32],
    all_three_clauses: bool,
) {
    debug_assert!(num_good.len() >= nc);
    debug_assert!(sat_xor.len() >= nc);
    debug_assert!(!all_three_clauses || natural_three_clause_layout(nc, co, cl));
    if CLEAR_SIDE_CARS {
        clear_exact_unsat_positions(unsat, unsat_pos);
        unsat.clear();
        break_score.fill(0);
    } else {
        debug_assert!(unsat.is_empty());
        debug_assert!(unsat_pos[..nc].iter().all(|&pos| pos == ABSENT));
    }

    if all_three_clauses {
        for c in 0..nc {
            let (good, xor) = clause3_good_xor_count(c * 3, cl, vars);
            num_good[c] = good;
            sat_xor[c] = xor;
            if good == 0 {
                unsat_pos[c] = unsat.len() as u32;
                unsat.push(c as u32);
            } else if good == 1 {
                debug_assert!((xor as usize) < break_score.len());
                break_score[xor as usize] += 1;
            }
        }
        return;
    }

    for c in 0..nc {
        let s = co[c] as usize;
        let e = co[c + 1] as usize;
        let (good, xor) = clause_good_xor_count(s, e, cl, vars);
        num_good[c] = good;
        sat_xor[c] = xor;
        if good == 0 {
            unsat_pos[c] = unsat.len() as u32;
            unsat.push(c as u32);
        } else if good == 1 {
            // Safe for C001: expected per-variable break count is small; debug catches malformed inputs.
            debug_assert!((xor as usize) < break_score.len());
            break_score[xor as usize] += 1;
        }
    }
}

fn rebuild_u8_exact_with_make_impl<const CLEAR_SIDE_CARS: bool>(
    nc: usize,
    co: &[u32],
    cl: &[i32],
    vars: &[bool],
    num_good: &mut [u8],
    unsat: &mut Vec<u32>,
    unsat_pos: &mut [u32],
    break_score: &mut [u16],
    sat_xor: &mut [u32],
    make_score: &mut [u16],
    all_three_clauses: bool,
) {
    debug_assert!(num_good.len() >= nc);
    debug_assert!(sat_xor.len() >= nc);
    debug_assert!(!all_three_clauses || natural_three_clause_layout(nc, co, cl));
    debug_assert!(make_score.iter().all(|&score| score == 0));
    if CLEAR_SIDE_CARS {
        clear_exact_unsat_positions(unsat, unsat_pos);
        unsat.clear();
        break_score.fill(0);
    } else {
        debug_assert!(unsat.is_empty());
        debug_assert!(unsat_pos[..nc].iter().all(|&pos| pos == ABSENT));
    }

    if all_three_clauses {
        for c in 0..nc {
            let s = c * 3;
            let (good, xor) = clause3_good_xor_count(s, cl, vars);
            num_good[c] = good;
            sat_xor[c] = xor;
            if good == 0 {
                unsat_pos[c] = unsat.len() as u32;
                unsat.push(c as u32);
                unsafe {
                    add_make_for_clause3(cl, make_score, s);
                }
            } else if good == 1 {
                debug_assert!((xor as usize) < break_score.len());
                break_score[xor as usize] += 1;
            }
        }
        return;
    }

    for c in 0..nc {
        let s = co[c] as usize;
        let e = co[c + 1] as usize;
        let (good, xor) = clause_good_xor_count(s, e, cl, vars);
        num_good[c] = good;
        sat_xor[c] = xor;
        if good == 0 {
            unsat_pos[c] = unsat.len() as u32;
            unsat.push(c as u32);
            unsafe {
                add_make_for_clause(co, cl, make_score, c);
            }
        } else if good == 1 {
            debug_assert!((xor as usize) < break_score.len());
            break_score[xor as usize] += 1;
        }
    }
}

pub(crate) fn rebuild_u8_exact(
    nc: usize,
    co: &[u32],
    cl: &[i32],
    vars: &[bool],
    num_good: &mut [u8],
    unsat: &mut Vec<u32>,
    unsat_pos: &mut [u32],
    break_score: &mut [u16],
    sat_xor: &mut [u32],
) {
    let all_three_clauses = natural_three_clause_layout(nc, co, cl);
    rebuild_u8_exact_impl::<true>(
        nc,
        co,
        cl,
        vars,
        num_good,
        unsat,
        unsat_pos,
        break_score,
        sat_xor,
        all_three_clauses,
    );
}

pub(crate) fn rebuild_u8_exact_with_make(
    nc: usize,
    co: &[u32],
    cl: &[i32],
    vars: &[bool],
    num_good: &mut [u8],
    unsat: &mut Vec<u32>,
    unsat_pos: &mut [u32],
    break_score: &mut [u16],
    sat_xor: &mut [u32],
    make_score: &mut [u16],
    all_three_clauses: bool,
) {
    debug_assert!(!all_three_clauses || natural_three_clause_layout(nc, co, cl));
    clear_exact_make_scores(co, cl, make_score, unsat, all_three_clauses);
    rebuild_u8_exact_with_make_impl::<true>(
        nc,
        co,
        cl,
        vars,
        num_good,
        unsat,
        unsat_pos,
        break_score,
        sat_xor,
        make_score,
        all_three_clauses,
    );
}

pub(crate) fn rebuild_u8_exact_with_make_fresh(
    nc: usize,
    co: &[u32],
    cl: &[i32],
    vars: &[bool],
    num_good: &mut [u8],
    unsat: &mut Vec<u32>,
    unsat_pos: &mut [u32],
    break_score: &mut [u16],
    sat_xor: &mut [u32],
    make_score: &mut [u16],
    all_three_clauses: bool,
) {
    debug_assert!(!all_three_clauses || natural_three_clause_layout(nc, co, cl));
    debug_assert!(make_score.iter().all(|&score| score == 0));
    rebuild_u8_exact_with_make_impl::<false>(
        nc,
        co,
        cl,
        vars,
        num_good,
        unsat,
        unsat_pos,
        break_score,
        sat_xor,
        make_score,
        all_three_clauses,
    );
}

pub(crate) fn rebuild_u8_lazy_make_count(
    nc: usize,
    co: &[u32],
    cl: &[i32],
    vars: &[bool],
    num_good: &mut [u8],
    residual: &mut Vec<u32>,
    make_score: &mut [u16],
) {
    debug_assert!(num_good.len() >= nc);
    residual.clear();
    make_score.fill(0);

    if natural_three_clause_layout(nc, co, cl) {
        for c in 0..nc {
            let good = clause3_good_count(c * 3, cl, vars);
            num_good[c] = good;
            if good == 0 {
                residual.push(c as u32);
                unsafe {
                    add_make_for_clause3(cl, make_score, c * 3);
                }
            }
        }
        return;
    }

    for c in 0..nc {
        let s = co[c] as usize;
        let e = co[c + 1] as usize;
        let mut good = 0u8;
        for &lit in &cl[s..e] {
            let v = lit_var_index(lit);
            if (lit > 0 && vars[v]) || (lit < 0 && !vars[v]) {
                good += 1;
            }
        }
        num_good[c] = good;
        if good == 0 {
            residual.push(c as u32);
            unsafe {
                add_make_for_clause(co, cl, make_score, c);
            }
        }
    }
}

pub(crate) fn rebuild_u8_lazy(
    nc: usize,
    co: &[u32],
    cl: &[i32],
    vars: &[bool],
    num_good: &mut [u8],
    residual: &mut Vec<u32>,
    break_score: &mut [u16],
    sat_xor: &mut [u32],
) {
    debug_assert!(num_good.len() >= nc);
    debug_assert!(sat_xor.len() >= nc);
    residual.clear();
    break_score.fill(0);

    if natural_three_clause_layout(nc, co, cl) {
        for c in 0..nc {
            let (good, xor) = clause3_good_xor_count(c * 3, cl, vars);
            num_good[c] = good;
            sat_xor[c] = xor;
            if good == 0 {
                residual.push(c as u32);
            } else if good == 1 {
                debug_assert!((xor as usize) < break_score.len());
                break_score[xor as usize] += 1;
            }
        }
        return;
    }

    for c in 0..nc {
        let s = co[c] as usize;
        let e = co[c + 1] as usize;
        let (good, xor) = clause_good_xor_count(s, e, cl, vars);
        num_good[c] = good;
        sat_xor[c] = xor;
        if good == 0 {
            residual.push(c as u32);
        } else if good == 1 {
            debug_assert!((xor as usize) < break_score.len());
            break_score[xor as usize] += 1;
        }
    }
}

pub(crate) fn rebuild_packed_exact(
    nc: usize,
    co: &[u32],
    cl: &[i32],
    vars: &[bool],
    num_good: &mut [u8],
    unsat: &mut Vec<u32>,
    unsat_pos: &mut [u32],
    break_score: &mut [u16],
    sat_xor: &mut [u32],
) {
    debug_assert!(sat_xor.len() >= nc);
    let byte_count = (nc + 3) >> 2;
    debug_assert!(num_good.len() >= byte_count);
    unsat.clear();
    unsat_pos.fill(ABSENT);
    break_score.fill(0);

    if natural_three_clause_layout(nc, co, cl) {
        let full_bytes = nc >> 2;
        for byte_idx in 0..full_bytes {
            let base = byte_idx << 2;
            let s = base * 3;
            let (g0, x0) = clause3_good_xor_count(s, cl, vars);
            let (g1, x1) = clause3_good_xor_count(s + 3, cl, vars);
            let (g2, x2) = clause3_good_xor_count(s + 6, cl, vars);
            let (g3, x3) = clause3_good_xor_count(s + 9, cl, vars);
            num_good[byte_idx] = g0 | (g1 << 2) | (g2 << 4) | (g3 << 6);
            sat_xor[base] = x0;
            sat_xor[base + 1] = x1;
            sat_xor[base + 2] = x2;
            sat_xor[base + 3] = x3;
            if g0 == 0 {
                unsat_pos[base] = unsat.len() as u32;
                unsat.push(base as u32);
            } else if g0 == 1 {
                debug_assert!((x0 as usize) < break_score.len());
                break_score[x0 as usize] += 1;
            }
            if g1 == 0 {
                unsat_pos[base + 1] = unsat.len() as u32;
                unsat.push((base + 1) as u32);
            } else if g1 == 1 {
                debug_assert!((x1 as usize) < break_score.len());
                break_score[x1 as usize] += 1;
            }
            if g2 == 0 {
                unsat_pos[base + 2] = unsat.len() as u32;
                unsat.push((base + 2) as u32);
            } else if g2 == 1 {
                debug_assert!((x2 as usize) < break_score.len());
                break_score[x2 as usize] += 1;
            }
            if g3 == 0 {
                unsat_pos[base + 3] = unsat.len() as u32;
                unsat.push((base + 3) as u32);
            } else if g3 == 1 {
                debug_assert!((x3 as usize) < break_score.len());
                break_score[x3 as usize] += 1;
            }
        }

        let mut c = full_bytes << 2;
        if c < nc {
            let byte_idx = c >> 2;
            let mut packed = 0u8;
            while c < nc {
                let (good, xor) = clause3_good_xor_count(c * 3, cl, vars);
                packed |= good << ((c & 3) << 1);
                sat_xor[c] = xor;
                if good == 0 {
                    unsat_pos[c] = unsat.len() as u32;
                    unsat.push(c as u32);
                } else if good == 1 {
                    debug_assert!((xor as usize) < break_score.len());
                    break_score[xor as usize] += 1;
                }
                c += 1;
            }
            num_good[byte_idx] = packed;
        }
        if num_good.len() > byte_count {
            num_good[byte_count..].fill(0);
        }
        return;
    }

    num_good.fill(0);
    for c in 0..nc {
        let s = co[c] as usize;
        let e = co[c + 1] as usize;
        let (good, xor) = clause_good_xor_count(s, e, cl, vars);
        num_good[c >> 2] |= good.min(3) << ((c & 3) << 1);
        sat_xor[c] = xor;
        if good == 0 {
            unsat_pos[c] = unsat.len() as u32;
            unsat.push(c as u32);
        } else if good == 1 {
            debug_assert!((xor as usize) < break_score.len());
            break_score[xor as usize] += 1;
        }
    }
}

pub(crate) fn rebuild_packed_lazy(
    nc: usize,
    co: &[u32],
    cl: &[i32],
    vars: &[bool],
    num_good: &mut [u8],
    residual: &mut Vec<u32>,
    break_score: &mut [u16],
    sat_xor: &mut [u32],
) {
    debug_assert!(sat_xor.len() >= nc);
    let byte_count = (nc + 3) >> 2;
    debug_assert!(num_good.len() >= byte_count);
    residual.clear();
    break_score.fill(0);

    if natural_three_clause_layout(nc, co, cl) {
        let full_bytes = nc >> 2;
        for byte_idx in 0..full_bytes {
            let base = byte_idx << 2;
            let s = base * 3;
            let (g0, x0) = clause3_good_xor_count(s, cl, vars);
            let (g1, x1) = clause3_good_xor_count(s + 3, cl, vars);
            let (g2, x2) = clause3_good_xor_count(s + 6, cl, vars);
            let (g3, x3) = clause3_good_xor_count(s + 9, cl, vars);
            num_good[byte_idx] = g0 | (g1 << 2) | (g2 << 4) | (g3 << 6);
            sat_xor[base] = x0;
            sat_xor[base + 1] = x1;
            sat_xor[base + 2] = x2;
            sat_xor[base + 3] = x3;
            if g0 == 0 {
                residual.push(base as u32);
            } else if g0 == 1 {
                debug_assert!((x0 as usize) < break_score.len());
                break_score[x0 as usize] += 1;
            }
            if g1 == 0 {
                residual.push((base + 1) as u32);
            } else if g1 == 1 {
                debug_assert!((x1 as usize) < break_score.len());
                break_score[x1 as usize] += 1;
            }
            if g2 == 0 {
                residual.push((base + 2) as u32);
            } else if g2 == 1 {
                debug_assert!((x2 as usize) < break_score.len());
                break_score[x2 as usize] += 1;
            }
            if g3 == 0 {
                residual.push((base + 3) as u32);
            } else if g3 == 1 {
                debug_assert!((x3 as usize) < break_score.len());
                break_score[x3 as usize] += 1;
            }
        }

        let mut c = full_bytes << 2;
        if c < nc {
            let byte_idx = c >> 2;
            let mut packed = 0u8;
            while c < nc {
                let (good, xor) = clause3_good_xor_count(c * 3, cl, vars);
                packed |= good << ((c & 3) << 1);
                sat_xor[c] = xor;
                if good == 0 {
                    residual.push(c as u32);
                } else if good == 1 {
                    debug_assert!((xor as usize) < break_score.len());
                    break_score[xor as usize] += 1;
                }
                c += 1;
            }
            num_good[byte_idx] = packed;
        }
        if num_good.len() > byte_count {
            num_good[byte_count..].fill(0);
        }
        return;
    }

    num_good.fill(0);
    for c in 0..nc {
        let s = co[c] as usize;
        let e = co[c + 1] as usize;
        let (good, xor) = clause_good_xor_count(s, e, cl, vars);
        num_good[c >> 2] |= good.min(3) << ((c & 3) << 1);
        sat_xor[c] = xor;
        if good == 0 {
            residual.push(c as u32);
        } else if good == 1 {
            debug_assert!((xor as usize) < break_score.len());
            break_score[xor as usize] += 1;
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn flip_u8_exact(
    v: usize,
    vars: &mut [bool],
    num_good: &mut [u8],
    sat_xor: &mut [u32],
    break_score: &mut [u16],
    unsat: &mut Vec<u32>,
    unsat_pos: &mut [u32],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
) {
    let was_true = *vars.get_unchecked(v);
    let (inc_s, inc_e, dec_s, dec_e) = if was_true {
        (
            *p_bound.get_unchecked(v) as usize,
            *all_off.get_unchecked(v + 1) as usize,
            *all_off.get_unchecked(v) as usize,
            *p_bound.get_unchecked(v) as usize,
        )
    } else {
        (
            *all_off.get_unchecked(v) as usize,
            *p_bound.get_unchecked(v) as usize,
            *p_bound.get_unchecked(v) as usize,
            *all_off.get_unchecked(v + 1) as usize,
        )
    };
    let vu = v as u32;

    for k in inc_s..inc_e {
        let c = *all_data.get_unchecked(k) as usize;
        let old = *num_good.get_unchecked(c);
        debug_assert!(old <= 2);
        if old == 1 {
            dec_break(break_score, *sat_xor.get_unchecked(c) as usize);
        }
        *num_good.get_unchecked_mut(c) = old + 1;
        *sat_xor.get_unchecked_mut(c) ^= vu;
        if old == 0 {
            remove_unsat_exact(unsat, unsat_pos, c);
            inc_break(break_score, v);
        }
    }

    for k in dec_s..dec_e {
        let c = *all_data.get_unchecked(k) as usize;
        let old = *num_good.get_unchecked(c);
        debug_assert!(old >= 1);
        if old == 1 {
            dec_break(break_score, v);
        }
        *num_good.get_unchecked_mut(c) = old - 1;
        *sat_xor.get_unchecked_mut(c) ^= vu;
        if old == 1 {
            add_unsat_exact(unsat, unsat_pos, c);
        } else if old == 2 {
            inc_break(break_score, *sat_xor.get_unchecked(c) as usize);
        }
    }

    *vars.get_unchecked_mut(v) = !was_true;
}

#[inline(always)]
pub(crate) unsafe fn flip_u8_exact_with_make(
    v: usize,
    vars: &mut [bool],
    num_good: &mut [u8],
    sat_xor: &mut [u32],
    break_score: &mut [u16],
    make_score: &mut [u16],
    unsat: &mut Vec<u32>,
    unsat_pos: &mut [u32],
    co: &[u32],
    cl: &[i32],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    all_three_clauses: bool,
) {
    debug_assert!(!all_three_clauses || natural_three_clause_layout(num_good.len(), co, cl));
    let was_true = *vars.get_unchecked(v);
    let (inc_s, inc_e, dec_s, dec_e) = if was_true {
        (
            *p_bound.get_unchecked(v) as usize,
            *all_off.get_unchecked(v + 1) as usize,
            *all_off.get_unchecked(v) as usize,
            *p_bound.get_unchecked(v) as usize,
        )
    } else {
        (
            *all_off.get_unchecked(v) as usize,
            *p_bound.get_unchecked(v) as usize,
            *p_bound.get_unchecked(v) as usize,
            *all_off.get_unchecked(v + 1) as usize,
        )
    };
    let vu = v as u32;

    for k in inc_s..inc_e {
        let c = *all_data.get_unchecked(k) as usize;
        let old = *num_good.get_unchecked(c);
        debug_assert!(old <= 2);
        if old == 1 {
            dec_break(break_score, *sat_xor.get_unchecked(c) as usize);
        }
        if old == 0 {
            if all_three_clauses {
                remove_make_for_clause3(cl, make_score, c * 3);
            } else {
                remove_make_for_clause(co, cl, make_score, c);
            }
        }
        *num_good.get_unchecked_mut(c) = old + 1;
        *sat_xor.get_unchecked_mut(c) ^= vu;
        if old == 0 {
            remove_unsat_exact(unsat, unsat_pos, c);
            inc_break(break_score, v);
        }
    }

    for k in dec_s..dec_e {
        let c = *all_data.get_unchecked(k) as usize;
        let old = *num_good.get_unchecked(c);
        debug_assert!(old >= 1);
        if old == 1 {
            dec_break(break_score, v);
        }
        *num_good.get_unchecked_mut(c) = old - 1;
        *sat_xor.get_unchecked_mut(c) ^= vu;
        if old == 1 {
            add_unsat_exact(unsat, unsat_pos, c);
            if all_three_clauses {
                add_make_for_clause3(cl, make_score, c * 3);
            } else {
                add_make_for_clause(co, cl, make_score, c);
            }
        } else if old == 2 {
            inc_break(break_score, *sat_xor.get_unchecked(c) as usize);
        }
    }

    *vars.get_unchecked_mut(v) = !was_true;
}

#[inline(always)]
pub(crate) unsafe fn flip_u8_lazy_count_with_make(
    v: usize,
    vars: &mut [bool],
    num_good: &mut [u8],
    make_score: &mut [u16],
    unsat_count: &mut usize,
    residual: &mut Vec<u32>,
    co: &[u32],
    cl: &[i32],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
) {
    let was_true = *vars.get_unchecked(v);
    let (inc_s, inc_e, dec_s, dec_e) = if was_true {
        (
            *p_bound.get_unchecked(v) as usize,
            *all_off.get_unchecked(v + 1) as usize,
            *all_off.get_unchecked(v) as usize,
            *p_bound.get_unchecked(v) as usize,
        )
    } else {
        (
            *all_off.get_unchecked(v) as usize,
            *p_bound.get_unchecked(v) as usize,
            *p_bound.get_unchecked(v) as usize,
            *all_off.get_unchecked(v + 1) as usize,
        )
    };

    for k in inc_s..inc_e {
        let c = *all_data.get_unchecked(k) as usize;
        let old = *num_good.get_unchecked(c);
        debug_assert!(old < 3);
        if old == 0 {
            debug_assert!(*unsat_count > 0);
            *unsat_count -= 1;
            remove_make_for_clause(co, cl, make_score, c);
        }
        *num_good.get_unchecked_mut(c) = old + 1;
    }

    for k in dec_s..dec_e {
        let c = *all_data.get_unchecked(k) as usize;
        let old = *num_good.get_unchecked(c);
        debug_assert!(old > 0);
        let new_val = old - 1;
        *num_good.get_unchecked_mut(c) = new_val;
        if new_val == 0 {
            *unsat_count += 1;
            residual.push(c as u32);
            add_make_for_clause(co, cl, make_score, c);
        }
    }

    *vars.get_unchecked_mut(v) = !was_true;
}

#[inline(always)]
pub(crate) unsafe fn flip_u8_lazy_count(
    v: usize,
    vars: &mut [bool],
    num_good: &mut [u8],
    sat_xor: &mut [u32],
    break_score: &mut [u16],
    unsat_count: &mut usize,
    residual: &mut Vec<u32>,
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
) {
    let was_true = *vars.get_unchecked(v);
    let (inc_s, inc_e, dec_s, dec_e) = if was_true {
        (
            *p_bound.get_unchecked(v) as usize,
            *all_off.get_unchecked(v + 1) as usize,
            *all_off.get_unchecked(v) as usize,
            *p_bound.get_unchecked(v) as usize,
        )
    } else {
        (
            *all_off.get_unchecked(v) as usize,
            *p_bound.get_unchecked(v) as usize,
            *p_bound.get_unchecked(v) as usize,
            *all_off.get_unchecked(v + 1) as usize,
        )
    };
    let vu = v as u32;

    for k in inc_s..inc_e {
        let c = *all_data.get_unchecked(k) as usize;
        let old = *num_good.get_unchecked(c);
        debug_assert!(old <= 2);
        if old == 1 {
            dec_break(break_score, *sat_xor.get_unchecked(c) as usize);
        }
        *num_good.get_unchecked_mut(c) = old + 1;
        *sat_xor.get_unchecked_mut(c) ^= vu;
        if old == 0 {
            debug_assert!(*unsat_count > 0);
            *unsat_count -= 1;
            inc_break(break_score, v);
        }
    }

    for k in dec_s..dec_e {
        let c = *all_data.get_unchecked(k) as usize;
        let old = *num_good.get_unchecked(c);
        debug_assert!(old >= 1);
        if old == 1 {
            dec_break(break_score, v);
        }
        *num_good.get_unchecked_mut(c) = old - 1;
        *sat_xor.get_unchecked_mut(c) ^= vu;
        if old == 1 {
            *unsat_count += 1;
            residual.push(c as u32);
        } else if old == 2 {
            inc_break(break_score, *sat_xor.get_unchecked(c) as usize);
        }
    }

    *vars.get_unchecked_mut(v) = !was_true;
}

#[inline(always)]
pub(crate) unsafe fn flip_packed_exact(
    v: usize,
    vars: &mut [bool],
    num_good: &mut [u8],
    sat_xor: &mut [u32],
    break_score: &mut [u16],
    unsat: &mut Vec<u32>,
    unsat_pos: &mut [u32],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
) {
    let was_true = *vars.get_unchecked(v);
    let (inc_s, inc_e, dec_s, dec_e) = if was_true {
        (
            *p_bound.get_unchecked(v) as usize,
            *all_off.get_unchecked(v + 1) as usize,
            *all_off.get_unchecked(v) as usize,
            *p_bound.get_unchecked(v) as usize,
        )
    } else {
        (
            *all_off.get_unchecked(v) as usize,
            *p_bound.get_unchecked(v) as usize,
            *p_bound.get_unchecked(v) as usize,
            *all_off.get_unchecked(v + 1) as usize,
        )
    };
    let vu = v as u32;

    for k in inc_s..inc_e {
        let c = *all_data.get_unchecked(k) as usize;
        let shift = (c & 3) << 1;
        let byte_idx = c >> 2;
        let old = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
        debug_assert!(old <= 2);
        if old == 1 {
            dec_break(break_score, *sat_xor.get_unchecked(c) as usize);
        }
        *num_good.get_unchecked_mut(byte_idx) += 1u8 << shift;
        *sat_xor.get_unchecked_mut(c) ^= vu;
        if old == 0 {
            remove_unsat_exact(unsat, unsat_pos, c);
            inc_break(break_score, v);
        }
    }

    for k in dec_s..dec_e {
        let c = *all_data.get_unchecked(k) as usize;
        let shift = (c & 3) << 1;
        let byte_idx = c >> 2;
        let old = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
        debug_assert!(old >= 1);
        if old == 1 {
            dec_break(break_score, v);
        }
        *num_good.get_unchecked_mut(byte_idx) -= 1u8 << shift;
        *sat_xor.get_unchecked_mut(c) ^= vu;
        if old == 1 {
            add_unsat_exact(unsat, unsat_pos, c);
        } else if old == 2 {
            inc_break(break_score, *sat_xor.get_unchecked(c) as usize);
        }
    }

    *vars.get_unchecked_mut(v) = !was_true;
}

#[inline(always)]
pub(crate) unsafe fn flip_packed_lazy(
    v: usize,
    vars: &mut [bool],
    num_good: &mut [u8],
    sat_xor: &mut [u32],
    break_score: &mut [u16],
    residual: &mut Vec<u32>,
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
) {
    let was_true = *vars.get_unchecked(v);
    let (inc_s, inc_e, dec_s, dec_e) = if was_true {
        (
            *p_bound.get_unchecked(v) as usize,
            *all_off.get_unchecked(v + 1) as usize,
            *all_off.get_unchecked(v) as usize,
            *p_bound.get_unchecked(v) as usize,
        )
    } else {
        (
            *all_off.get_unchecked(v) as usize,
            *p_bound.get_unchecked(v) as usize,
            *p_bound.get_unchecked(v) as usize,
            *all_off.get_unchecked(v + 1) as usize,
        )
    };
    let vu = v as u32;

    for k in inc_s..inc_e {
        let c = *all_data.get_unchecked(k) as usize;
        let shift = (c & 3) << 1;
        let byte_idx = c >> 2;
        let old = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
        debug_assert!(old <= 2);
        if old == 1 {
            dec_break(break_score, *sat_xor.get_unchecked(c) as usize);
        }
        *num_good.get_unchecked_mut(byte_idx) += 1u8 << shift;
        *sat_xor.get_unchecked_mut(c) ^= vu;
        if old == 0 {
            inc_break(break_score, v);
        }
    }

    for k in dec_s..dec_e {
        let c = *all_data.get_unchecked(k) as usize;
        let shift = (c & 3) << 1;
        let byte_idx = c >> 2;
        let old = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
        debug_assert!(old >= 1);
        if old == 1 {
            dec_break(break_score, v);
        }
        *num_good.get_unchecked_mut(byte_idx) -= 1u8 << shift;
        *sat_xor.get_unchecked_mut(c) ^= vu;
        if old == 1 {
            residual.push(c as u32);
        } else if old == 2 {
            inc_break(break_score, *sat_xor.get_unchecked(c) as usize);
        }
    }

    *vars.get_unchecked_mut(v) = !was_true;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_state_lit_var_index_matches_unsigned_abs_index_for_valid_literals() {
        for lit in [-128_i32, -7, -1, 1, 7, 128] {
            assert_eq!(lit_var_index(lit), lit.unsigned_abs() as usize - 1);
        }
    }

    #[test]
    fn clause_good_xor_count_matches_generic_reference() {
        let cl = [1, -2, 3, -1, 2, 4, -5, 5, -4, 3, 3, -3, 3];
        let vars = [true, false, true, false, true];

        for (s, e, expected_good, expected_xor) in [
            (0, 0, 0_u8, 0_u32),
            (0, 1, 1, 0),
            (3, 5, 0, 0),
            (0, 3, 3_u8, 0_u32 ^ 1 ^ 2),
            (5, 10, 3, 4 ^ 3 ^ 2),
            (10, 13, 2, 2 ^ 2),
        ] {
            let mut generic_good = 0_u8;
            let mut generic_xor = 0_u32;
            for &lit in &cl[s..e] {
                let (good, xor) = lit_good_xor(lit, &vars);
                generic_good += good;
                generic_xor ^= xor;
            }

            assert_eq!(generic_good, expected_good);
            assert_eq!(generic_xor, expected_xor);
            assert_eq!(
                clause_good_xor_count(s, e, &cl, &vars),
                (expected_good, expected_xor)
            );
        }
    }

    #[test]
    fn make_score_clause_updates_match_generic_reference_for_short_clauses() {
        let co = [0_u32, 1, 3, 6, 10];
        let cl = [1, -2, 2, -1, 2, -3, 1, 4, -4, 5];
        let mut expected = vec![0_u16; 5];

        for c in 0..co.len() - 1 {
            for &lit in &cl[co[c] as usize..co[c + 1] as usize] {
                expected[lit_var_index(lit)] += 1;
            }
        }

        let mut make_score = vec![0_u16; 5];
        unsafe {
            for c in 0..co.len() - 1 {
                add_make_for_clause(&co, &cl, &mut make_score, c);
            }
        }
        assert_eq!(make_score, expected);

        unsafe {
            for c in 0..co.len() - 1 {
                remove_make_for_clause(&co, &cl, &mut make_score, c);
            }
        }
        assert_eq!(make_score, vec![0_u16; 5]);
    }

    #[test]
    fn remove_unsat_exact_handles_last_entry_without_moving() {
        let mut unsat = vec![2_u32, 5, 7];
        let mut unsat_pos = vec![ABSENT; 8];
        unsat_pos[2] = 0;
        unsat_pos[5] = 1;
        unsat_pos[7] = 2;

        unsafe {
            remove_unsat_exact(&mut unsat, &mut unsat_pos, 7);
        }

        assert_eq!(unsat, vec![2_u32, 5]);
        assert_eq!(unsat_pos[2], 0);
        assert_eq!(unsat_pos[5], 1);
        assert_eq!(unsat_pos[7], ABSENT);

        unsafe {
            remove_unsat_exact(&mut unsat, &mut unsat_pos, 2);
        }

        assert_eq!(unsat, vec![5_u32]);
        assert_eq!(unsat_pos[2], ABSENT);
        assert_eq!(unsat_pos[5], 0);
        assert_eq!(unsat_pos[7], ABSENT);
    }

    #[test]
    fn u8_rebuild_paths_overwrite_existing_counts_without_prefill() {
        let nc = 3;
        let co = [0_u32, 2, 4, 5];
        let cl = [1, -2, -1, 2, 3];
        let vars = [true, true, false];

        let mut num_good = vec![9_u8; nc];
        let mut unsat = vec![0_u32, 1];
        let mut unsat_pos = vec![0_u32, 1, ABSENT];
        let mut break_score = vec![5_u16; 3];
        let mut sat_xor = vec![8_u32; nc];
        rebuild_u8_exact(
            nc,
            &co,
            &cl,
            &vars,
            &mut num_good,
            &mut unsat,
            &mut unsat_pos,
            &mut break_score,
            &mut sat_xor,
        );
        assert_eq!(num_good, vec![1, 1, 0]);
        assert_eq!(unsat, vec![2]);
        assert_eq!(unsat_pos, vec![ABSENT, ABSENT, 0]);
        assert_eq!(break_score, vec![1, 1, 0]);
        assert_eq!(sat_xor, vec![0, 1, 0]);

        let mut num_good = vec![9_u8; nc];
        let mut residual = vec![99_u32];
        let mut make_score = vec![5_u16; 3];
        rebuild_u8_lazy_make_count(
            nc,
            &co,
            &cl,
            &vars,
            &mut num_good,
            &mut residual,
            &mut make_score,
        );
        assert_eq!(num_good, vec![1, 1, 0]);
        assert_eq!(residual, vec![2]);
        assert_eq!(make_score, vec![0, 0, 1]);

        let mut num_good = vec![9_u8; nc];
        let mut residual = vec![99_u32];
        let mut break_score = vec![5_u16; 3];
        let mut sat_xor = vec![8_u32; nc];
        rebuild_u8_lazy(
            nc,
            &co,
            &cl,
            &vars,
            &mut num_good,
            &mut residual,
            &mut break_score,
            &mut sat_xor,
        );
        assert_eq!(num_good, vec![1, 1, 0]);
        assert_eq!(residual, vec![2]);
        assert_eq!(break_score, vec![1, 1, 0]);
        assert_eq!(sat_xor, vec![0, 1, 0]);
    }

    #[test]
    fn u8_rebuild_all_three_fast_paths_match_generic_reference() {
        let nc = 7;
        let co = [0_u32, 3, 6, 9, 12, 15, 18, 21];
        let cl = [
            1, 2, 3, -1, -2, -3, 1, -2, 4, -1, 2, -4, 3, 4, -2, -3, -4, 2, 1, -3, -4,
        ];
        let vars = [true, false, true, false];

        let mut expected_num_good = vec![0_u8; nc];
        let mut expected_unsat = Vec::new();
        let mut expected_unsat_pos = vec![ABSENT; nc];
        let mut expected_residual = Vec::new();
        let mut expected_break = vec![0_u16; vars.len()];
        let mut expected_xor = vec![0_u32; nc];
        let mut expected_make = vec![0_u16; vars.len()];
        let mut expected_packed = vec![0_u8; (nc + 3) >> 2];

        for c in 0..nc {
            let (good, xor) = clause_good_xor_count(co[c] as usize, co[c + 1] as usize, &cl, &vars);
            expected_num_good[c] = good;
            expected_xor[c] = xor;
            expected_packed[c >> 2] |= good.min(3) << ((c & 3) << 1);
            if good == 0 {
                expected_unsat_pos[c] = expected_unsat.len() as u32;
                expected_unsat.push(c as u32);
                expected_residual.push(c as u32);
                unsafe {
                    add_make_for_clause(&co, &cl, &mut expected_make, c);
                }
            } else if good == 1 {
                expected_break[xor as usize] += 1;
            }
        }

        let mut num_good = vec![9_u8; nc];
        let mut unsat = vec![0_u32, 1];
        let mut unsat_pos = vec![0_u32, 1, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT];
        let mut break_score = vec![5_u16; vars.len()];
        let mut sat_xor = vec![8_u32; nc];
        rebuild_u8_exact(
            nc,
            &co,
            &cl,
            &vars,
            &mut num_good,
            &mut unsat,
            &mut unsat_pos,
            &mut break_score,
            &mut sat_xor,
        );
        assert_eq!(num_good, expected_num_good);
        assert_eq!(unsat, expected_unsat);
        assert_eq!(unsat_pos, expected_unsat_pos);
        assert_eq!(break_score, expected_break);
        assert_eq!(sat_xor, expected_xor);

        let mut num_good = vec![9_u8; nc];
        let mut residual = vec![99_u32];
        let mut make_score = vec![5_u16; vars.len()];
        rebuild_u8_lazy_make_count(
            nc,
            &co,
            &cl,
            &vars,
            &mut num_good,
            &mut residual,
            &mut make_score,
        );
        assert_eq!(num_good, expected_num_good);
        assert_eq!(residual, expected_residual);
        assert_eq!(make_score, expected_make);

        let mut num_good = vec![9_u8; nc];
        let mut residual = vec![99_u32];
        let mut break_score = vec![5_u16; vars.len()];
        let mut sat_xor = vec![8_u32; nc];
        rebuild_u8_lazy(
            nc,
            &co,
            &cl,
            &vars,
            &mut num_good,
            &mut residual,
            &mut break_score,
            &mut sat_xor,
        );
        assert_eq!(num_good, expected_num_good);
        assert_eq!(residual, expected_residual);
        assert_eq!(break_score, expected_break);
        assert_eq!(sat_xor, expected_xor);

        let mut packed = vec![0xff_u8; (nc + 3) >> 2];
        let mut unsat = vec![99_u32];
        let mut unsat_pos = vec![9_u32; nc];
        let mut break_score = vec![5_u16; vars.len()];
        let mut sat_xor = vec![8_u32; nc];
        rebuild_packed_exact(
            nc,
            &co,
            &cl,
            &vars,
            &mut packed,
            &mut unsat,
            &mut unsat_pos,
            &mut break_score,
            &mut sat_xor,
        );
        assert_eq!(packed, expected_packed);
        assert_eq!(unsat, expected_unsat);
        assert_eq!(unsat_pos, expected_unsat_pos);
        assert_eq!(break_score, expected_break);
        assert_eq!(sat_xor, expected_xor);

        let mut packed = vec![0xff_u8; (nc + 3) >> 2];
        let mut residual = vec![99_u32];
        let mut break_score = vec![5_u16; vars.len()];
        let mut sat_xor = vec![8_u32; nc];
        rebuild_packed_lazy(
            nc,
            &co,
            &cl,
            &vars,
            &mut packed,
            &mut residual,
            &mut break_score,
            &mut sat_xor,
        );
        assert_eq!(packed, expected_packed);
        assert_eq!(residual, expected_residual);
        assert_eq!(break_score, expected_break);
        assert_eq!(sat_xor, expected_xor);
    }

    #[test]
    fn fresh_exact_with_make_rebuild_skips_clean_sidecar_prefills() {
        let nc = 3;
        let co = [0_u32, 2, 4, 5];
        let cl = [1, -2, -1, 2, 3];
        let vars = [true, true, false];
        let mut num_good = vec![9_u8; nc];
        let mut unsat = Vec::new();
        let mut unsat_pos = vec![ABSENT; nc];
        let mut break_score = vec![0_u16; 3];
        let mut sat_xor = vec![8_u32; nc];
        let mut make_score = vec![0_u16; 3];

        rebuild_u8_exact_with_make_fresh(
            nc,
            &co,
            &cl,
            &vars,
            &mut num_good,
            &mut unsat,
            &mut unsat_pos,
            &mut break_score,
            &mut sat_xor,
            &mut make_score,
            false,
        );

        assert_eq!(num_good, vec![1, 1, 0]);
        assert_eq!(unsat, vec![2]);
        assert_eq!(unsat_pos, vec![ABSENT, ABSENT, 0]);
        assert_eq!(break_score, vec![1, 1, 0]);
        assert_eq!(sat_xor, vec![0, 1, 0]);
        assert_eq!(make_score, vec![0, 0, 1]);
    }

    #[test]
    fn exact_with_make_rebuild_sparse_clears_old_unsat_make_scores() {
        let nc = 3;
        let co = [0_u32, 2, 4, 5];
        let cl = [1, -2, -1, 2, 3];
        let vars = [true, true, false, false, false, false, false, false];
        let mut num_good = vec![9_u8; nc];
        let mut unsat = vec![0_u32, 1];
        let mut unsat_pos = vec![0_u32, 1, ABSENT];
        let mut break_score = vec![5_u16; 8];
        let mut sat_xor = vec![8_u32; nc];
        let mut make_score = vec![2_u16, 2, 0, 0, 0, 0, 0, 0];

        rebuild_u8_exact_with_make(
            nc,
            &co,
            &cl,
            &vars,
            &mut num_good,
            &mut unsat,
            &mut unsat_pos,
            &mut break_score,
            &mut sat_xor,
            &mut make_score,
            false,
        );

        assert_eq!(num_good, vec![1, 1, 0]);
        assert_eq!(unsat, vec![2]);
        assert_eq!(unsat_pos, vec![ABSENT, ABSENT, 0]);
        assert_eq!(break_score, vec![1, 1, 0, 0, 0, 0, 0, 0]);
        assert_eq!(sat_xor, vec![0, 1, 0]);
        assert_eq!(make_score, vec![0, 0, 1, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn exact_with_make_rebuild_all_three_make_score_fast_paths_match_expected() {
        let nc = 2;
        let co = [0_u32, 3, 6];
        let cl = [1, 2, 3, -1, -2, -3];
        let mut vars = vec![false; 16];
        let mut num_good = vec![9_u8; nc];
        let mut unsat = Vec::new();
        let mut unsat_pos = vec![ABSENT; nc];
        let mut break_score = vec![0_u16; vars.len()];
        let mut sat_xor = vec![8_u32; nc];
        let mut make_score = vec![0_u16; vars.len()];

        rebuild_u8_exact_with_make_fresh(
            nc,
            &co,
            &cl,
            &vars,
            &mut num_good,
            &mut unsat,
            &mut unsat_pos,
            &mut break_score,
            &mut sat_xor,
            &mut make_score,
            true,
        );

        let mut expected_make = vec![0_u16; vars.len()];
        expected_make[0] = 1;
        expected_make[1] = 1;
        expected_make[2] = 1;
        assert_eq!(num_good, vec![0, 3]);
        assert_eq!(unsat, vec![0]);
        assert_eq!(unsat_pos, vec![0, ABSENT]);
        assert_eq!(break_score, vec![0; vars.len()]);
        assert_eq!(sat_xor, vec![0, 3]);
        assert_eq!(make_score, expected_make);

        vars[0] = true;
        vars[1] = true;
        vars[2] = true;
        rebuild_u8_exact_with_make(
            nc,
            &co,
            &cl,
            &vars,
            &mut num_good,
            &mut unsat,
            &mut unsat_pos,
            &mut break_score,
            &mut sat_xor,
            &mut make_score,
            true,
        );

        assert_eq!(num_good, vec![3, 0]);
        assert_eq!(unsat, vec![1]);
        assert_eq!(unsat_pos, vec![ABSENT, 0]);
        assert_eq!(break_score, vec![0; vars.len()]);
        assert_eq!(sat_xor, vec![3, 0]);
        assert_eq!(make_score, expected_make);
    }

    #[test]
    fn exact_with_make_fused_rebuild_matches_two_pass_reference() {
        fn two_pass_reference(
            nc: usize,
            co: &[u32],
            cl: &[i32],
            vars: &[bool],
            all_three_clauses: bool,
        ) -> (Vec<u8>, Vec<u32>, Vec<u32>, Vec<u16>, Vec<u32>, Vec<u16>) {
            let mut num_good = vec![9_u8; nc];
            let mut unsat = Vec::new();
            let mut unsat_pos = vec![ABSENT; nc];
            let mut break_score = vec![0_u16; vars.len()];
            let mut sat_xor = vec![8_u32; nc];
            let mut make_score = vec![0_u16; vars.len()];
            rebuild_u8_exact_impl::<false>(
                nc,
                co,
                cl,
                vars,
                &mut num_good,
                &mut unsat,
                &mut unsat_pos,
                &mut break_score,
                &mut sat_xor,
                all_three_clauses,
            );
            unsafe {
                for &cid in unsat.iter() {
                    if all_three_clauses {
                        add_make_for_clause3(cl, &mut make_score, cid as usize * 3);
                    } else {
                        add_make_for_clause(co, cl, &mut make_score, cid as usize);
                    }
                }
            }
            (num_good, unsat, unsat_pos, break_score, sat_xor, make_score)
        }

        fn fused_fresh(
            nc: usize,
            co: &[u32],
            cl: &[i32],
            vars: &[bool],
            all_three_clauses: bool,
        ) -> (Vec<u8>, Vec<u32>, Vec<u32>, Vec<u16>, Vec<u32>, Vec<u16>) {
            let mut num_good = vec![9_u8; nc];
            let mut unsat = Vec::new();
            let mut unsat_pos = vec![ABSENT; nc];
            let mut break_score = vec![0_u16; vars.len()];
            let mut sat_xor = vec![8_u32; nc];
            let mut make_score = vec![0_u16; vars.len()];
            rebuild_u8_exact_with_make_fresh(
                nc,
                co,
                cl,
                vars,
                &mut num_good,
                &mut unsat,
                &mut unsat_pos,
                &mut break_score,
                &mut sat_xor,
                &mut make_score,
                all_three_clauses,
            );
            (num_good, unsat, unsat_pos, break_score, sat_xor, make_score)
        }

        let three_nc = 4;
        let three_co = [0_u32, 3, 6, 9, 12];
        let three_cl = [1, 2, 3, -1, -2, -3, 1, -2, 4, -1, 2, -4];
        let three_vars = [false, false, false, true];
        assert_eq!(
            fused_fresh(three_nc, &three_co, &three_cl, &three_vars, true),
            two_pass_reference(three_nc, &three_co, &three_cl, &three_vars, true)
        );

        let mixed_nc = 4;
        let mixed_co = [0_u32, 1, 3, 6, 10];
        let mixed_cl = [1, -1, 2, 3, -4, 2, -1, -2, -3, 4];
        let mixed_vars = [false, false, false, true];
        assert_eq!(
            fused_fresh(mixed_nc, &mixed_co, &mixed_cl, &mixed_vars, false),
            two_pass_reference(mixed_nc, &mixed_co, &mixed_cl, &mixed_vars, false)
        );
    }

    #[test]
    fn exact_with_make_rebuild_mixed_average_three_layout_uses_guarded_generic_path() {
        let nc = 2;
        let co = [0_u32, 2, 6];
        let cl = [1, -2, 3, -4, 5, -6];
        let mut vars = vec![false, true, false, false, false, false];
        let mut num_good = vec![9_u8; nc];
        let mut unsat = Vec::new();
        let mut unsat_pos = vec![ABSENT; nc];
        let mut break_score = vec![0_u16; vars.len()];
        let mut sat_xor = vec![8_u32; nc];
        let mut make_score = vec![0_u16; vars.len()];

        rebuild_u8_exact_with_make_fresh(
            nc,
            &co,
            &cl,
            &vars,
            &mut num_good,
            &mut unsat,
            &mut unsat_pos,
            &mut break_score,
            &mut sat_xor,
            &mut make_score,
            false,
        );

        assert_eq!(num_good, vec![0, 2]);
        assert_eq!(unsat, vec![0]);
        assert_eq!(unsat_pos, vec![0, ABSENT]);
        assert_eq!(break_score, vec![0; vars.len()]);
        assert_eq!(sat_xor, vec![0, 6]);
        assert_eq!(make_score, vec![1, 1, 0, 0, 0, 0]);

        vars = vec![true, false, false, true, false, true];
        rebuild_u8_exact_with_make(
            nc,
            &co,
            &cl,
            &vars,
            &mut num_good,
            &mut unsat,
            &mut unsat_pos,
            &mut break_score,
            &mut sat_xor,
            &mut make_score,
            false,
        );

        assert_eq!(num_good, vec![2, 0]);
        assert_eq!(unsat, vec![1]);
        assert_eq!(unsat_pos, vec![ABSENT, 0]);
        assert_eq!(break_score, vec![0; vars.len()]);
        assert_eq!(sat_xor, vec![1, 0]);
        assert_eq!(make_score, vec![0, 0, 1, 1, 1, 1]);
    }

    #[test]
    fn flip_exact_with_make_all_three_fast_path_matches_generic_reference() {
        fn build_occ(nv: usize, co: &[u32], cl: &[i32]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
            let mut p_cnt = vec![0_u32; nv];
            let mut n_cnt = vec![0_u32; nv];
            for c in 0..co.len() - 1 {
                for &lit in &cl[co[c] as usize..co[c + 1] as usize] {
                    let v = lit_var_index(lit);
                    if lit > 0 {
                        p_cnt[v] += 1;
                    } else {
                        n_cnt[v] += 1;
                    }
                }
            }

            let mut all_off = vec![0_u32; nv + 1];
            let mut p_bound = vec![0_u32; nv];
            for v in 0..nv {
                p_bound[v] = all_off[v] + p_cnt[v];
                all_off[v + 1] = p_bound[v] + n_cnt[v];
            }

            let mut p_pos = all_off[..nv].to_vec();
            let mut n_pos = p_bound.clone();
            let mut all_data = vec![0_u32; all_off[nv] as usize];
            for c in 0..co.len() - 1 {
                for &lit in &cl[co[c] as usize..co[c + 1] as usize] {
                    let v = lit_var_index(lit);
                    if lit > 0 {
                        let pos = p_pos[v] as usize;
                        all_data[pos] = c as u32;
                        p_pos[v] += 1;
                    } else {
                        let pos = n_pos[v] as usize;
                        all_data[pos] = c as u32;
                        n_pos[v] += 1;
                    }
                }
            }

            (all_off, p_bound, all_data)
        }

        let nc = 2;
        let co = [0_u32, 3, 6];
        let cl = [1, 2, 3, -1, -2, -3];
        let vars = vec![false, false, false];
        let (all_off, p_bound, all_data) = build_occ(vars.len(), &co, &cl);

        let mut direct_vars = vars.clone();
        let mut direct_num_good = vec![0_u8; nc];
        let mut direct_unsat = Vec::new();
        let mut direct_unsat_pos = vec![ABSENT; nc];
        let mut direct_break = vec![0_u16; vars.len()];
        let mut direct_xor = vec![0_u32; nc];
        let mut direct_make = vec![0_u16; vars.len()];
        rebuild_u8_exact_with_make_fresh(
            nc,
            &co,
            &cl,
            &direct_vars,
            &mut direct_num_good,
            &mut direct_unsat,
            &mut direct_unsat_pos,
            &mut direct_break,
            &mut direct_xor,
            &mut direct_make,
            true,
        );

        let mut generic_vars = direct_vars.clone();
        let mut generic_num_good = direct_num_good.clone();
        let mut generic_unsat = direct_unsat.clone();
        let mut generic_unsat_pos = direct_unsat_pos.clone();
        let mut generic_break = direct_break.clone();
        let mut generic_xor = direct_xor.clone();
        let mut generic_make = direct_make.clone();

        for _ in 0..2 {
            unsafe {
                flip_u8_exact_with_make(
                    0,
                    &mut direct_vars,
                    &mut direct_num_good,
                    &mut direct_xor,
                    &mut direct_break,
                    &mut direct_make,
                    &mut direct_unsat,
                    &mut direct_unsat_pos,
                    &co,
                    &cl,
                    &all_off,
                    &p_bound,
                    &all_data,
                    true,
                );
                flip_u8_exact_with_make(
                    0,
                    &mut generic_vars,
                    &mut generic_num_good,
                    &mut generic_xor,
                    &mut generic_break,
                    &mut generic_make,
                    &mut generic_unsat,
                    &mut generic_unsat_pos,
                    &co,
                    &cl,
                    &all_off,
                    &p_bound,
                    &all_data,
                    false,
                );
            }

            assert_eq!(direct_vars, generic_vars);
            assert_eq!(direct_num_good, generic_num_good);
            assert_eq!(direct_xor, generic_xor);
            assert_eq!(direct_break, generic_break);
            assert_eq!(direct_make, generic_make);
            assert_eq!(direct_unsat, generic_unsat);
            assert_eq!(direct_unsat_pos, generic_unsat_pos);
        }
    }

    #[test]
    fn packed_rebuild_paths_overwrite_sat_xor_without_prefill() {
        let nc = 3;
        let co = [0_u32, 2, 4, 5];
        let cl = [1, -2, -1, 2, 3];
        let vars = [true, true, false];

        let mut num_good = vec![0xff_u8; (nc + 3) >> 2];
        let mut unsat = vec![99_u32];
        let mut unsat_pos = vec![7_u32; nc];
        let mut break_score = vec![5_u16; 3];
        let mut sat_xor = vec![8_u32; nc];
        rebuild_packed_exact(
            nc,
            &co,
            &cl,
            &vars,
            &mut num_good,
            &mut unsat,
            &mut unsat_pos,
            &mut break_score,
            &mut sat_xor,
        );
        assert_eq!(num_good, vec![1 | (1 << 2)]);
        assert_eq!(unsat, vec![2]);
        assert_eq!(unsat_pos, vec![ABSENT, ABSENT, 0]);
        assert_eq!(break_score, vec![1, 1, 0]);
        assert_eq!(sat_xor, vec![0, 1, 0]);

        let mut num_good = vec![0xff_u8; (nc + 3) >> 2];
        let mut residual = vec![99_u32];
        let mut break_score = vec![5_u16; 3];
        let mut sat_xor = vec![8_u32; nc];
        rebuild_packed_lazy(
            nc,
            &co,
            &cl,
            &vars,
            &mut num_good,
            &mut residual,
            &mut break_score,
            &mut sat_xor,
        );
        assert_eq!(num_good, vec![1 | (1 << 2)]);
        assert_eq!(residual, vec![2]);
        assert_eq!(break_score, vec![1, 1, 0]);
        assert_eq!(sat_xor, vec![0, 1, 0]);
    }
}
