use rand::{rngs::SmallRng, Rng};

#[inline(always)]
pub unsafe fn apply_flip(
    v: usize,
    variables: &mut [bool],
    num_good: &mut [u8],
    residual: &mut Vec<usize>,
    p_clauses: &[Vec<usize>],
    n_clauses: &[Vec<usize>],
) {
    let was_true = *variables.get_unchecked(v);
    let dec = if was_true {
        p_clauses.get_unchecked(v)
    } else {
        n_clauses.get_unchecked(v)
    };
    let inc = if was_true {
        n_clauses.get_unchecked(v)
    } else {
        p_clauses.get_unchecked(v)
    };

    for &cid in inc.iter() {
        *num_good.get_unchecked_mut(cid) = num_good.get_unchecked(cid).wrapping_add(1);
    }
    for &cid in dec.iter() {
        let ng = num_good.get_unchecked(cid).wrapping_sub(1);
        *num_good.get_unchecked_mut(cid) = ng;
        if ng == 0 {
            residual.push(cid);
        }
    }
    *variables.get_unchecked_mut(v) = !was_true;
}

#[inline(always)]
pub fn clause_sat_with_one_flip(clause: &[i32], vars: &[bool], a: usize) -> bool {
    for &lit in clause.iter() {
        let v = (lit.abs() as usize) - 1;
        let mut val = unsafe { *vars.get_unchecked(v) };
        if v == a {
            val = !val;
        }
        if (lit > 0 && val) || (lit < 0 && !val) {
            return true;
        }
    }
    false
}

#[inline(always)]
pub fn clause_sat_with_two_flips(clause: &[i32], vars: &[bool], a: usize, b: usize) -> bool {
    for &lit in clause.iter() {
        let v = (lit.abs() as usize) - 1;
        let mut val = unsafe { *vars.get_unchecked(v) };
        if v == a || v == b {
            val = !val;
        }
        if (lit > 0 && val) || (lit < 0 && !val) {
            return true;
        }
    }
    false
}

#[inline(always)]
pub fn clause_sat_with_mask(clause: &[i32], vars: &[bool], pos: &[i16], mask: u32) -> bool {
    for &lit in clause.iter() {
        let v = (lit.abs() as usize) - 1;
        let mut val = unsafe { *vars.get_unchecked(v) };
        let p = unsafe { *pos.get_unchecked(v) };
        if p >= 0 && ((mask >> (p as u32)) & 1) != 0 {
            val = !val;
        }
        if (lit > 0 && val) || (lit < 0 && !val) {
            return true;
        }
    }
    false
}

#[inline(always)]
pub fn verify_all(clauses: &[Vec<i32>], vars: &[bool]) -> bool {
    for c in clauses.iter() {
        let mut sat = false;
        for &lit in c.iter() {
            let v = (lit.abs() as usize) - 1;
            let val = unsafe { *vars.get_unchecked(v) };
            if (lit > 0 && val) || (lit < 0 && !val) {
                sat = true;
                break;
            }
        }
        if !sat {
            return false;
        }
    }
    true
}

#[inline(always)]
pub fn count_unsat(clauses: &[Vec<i32>], vars: &[bool]) -> usize {
    let mut u = 0usize;
    for c in clauses.iter() {
        let mut sat = false;
        for &lit in c.iter() {
            let v = (lit.abs() as usize) - 1;
            let val = unsafe { *vars.get_unchecked(v) };
            if (lit > 0 && val) || (lit < 0 && !val) {
                sat = true;
                break;
            }
        }
        if !sat {
            u += 1;
        }
    }
    u
}

#[inline(always)]
pub fn init_from_pref(rng: &mut SmallRng, pref: &[i8], flip_one_in: u32, variables: &mut [bool]) {
    if flip_one_in == 0 {
        for (i, &b) in pref.iter().enumerate() {
            variables[i] = match b {
                2 => true,
                -2 => false,
                1 => true,
                -1 => false,
                _ => rng.gen(),
            };
        }
    } else {
        for (i, &b) in pref.iter().enumerate() {
            let mut val = match b {
                2 => true,
                -2 => false,
                1 => true,
                -1 => false,
                _ => rng.gen(),
            };
            if b.abs() != 2 && (rng.gen::<u32>() % flip_one_in) == 0 {
                val = !val;
            }
            variables[i] = val;
        }
    }
}

#[inline(always)]
pub fn recompute_state(
    clauses: &[Vec<i32>],
    variables: &[bool],
    num_good: &mut [u8],
    residual: &mut Vec<usize>,
) {
    residual.clear();
    for (ci, c) in clauses.iter().enumerate() {
        let mut g = 0u8;
        for &l in c.iter() {
            let var = (l.abs() as usize) - 1;
            let val = unsafe { *variables.get_unchecked(var) };
            if (l > 0 && val) || (l < 0 && !val) {
                g = g.wrapping_add(1);
            }
        }
        num_good[ci] = g;
        if g == 0 {
            residual.push(ci);
        }
    }
}
