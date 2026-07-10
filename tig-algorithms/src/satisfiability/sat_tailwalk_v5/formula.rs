use tig_challenges::satisfiability::Challenge;

#[inline(always)]
fn lit_var_index(lit: i32) -> usize {
    if lit > 0 {
        lit as usize - 1
    } else {
        (-lit) as usize - 1
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SmallClause {
    pub(crate) lits: [i32; 3],
    pub(crate) len: u8,
}

#[derive(Clone, Debug)]
pub(crate) struct Formula {
    pub(crate) nv: usize,
    pub(crate) nc: usize,
    pub(crate) cl: Vec<i32>,
    pub(crate) co: Vec<u32>,
    pub(crate) all_clauses_are_three: bool,
    p_off: Vec<u32>,
    n_off: Vec<u32>,
    p_items: Vec<u32>,
    n_items: Vec<u32>,
}

impl Formula {
    pub(crate) fn from_challenge(challenge: &Challenge) -> Self {
        Self::from_raw(challenge.num_variables, &challenge.clauses)
    }

    pub(crate) fn from_raw(nv: usize, clauses: &[Vec<i32>]) -> Self {
        let mut cl = Vec::with_capacity(clauses.len() * 3);
        let mut co = Vec::with_capacity(clauses.len() + 1);
        let mut p_counts = vec![0u32; nv];
        let mut n_counts = vec![0u32; nv];
        let mut all_clauses_are_three = true;
        co.push(0);

        for raw in clauses {
            let before_len = cl.len();
            if raw.len() >= 3 {
                if append_normalized_clause3(
                    raw[0],
                    raw[1],
                    raw[2],
                    &mut cl,
                    &mut p_counts,
                    &mut n_counts,
                ) {
                    all_clauses_are_three &= cl.len() - before_len == 3;
                    co.push(cl.len() as u32);
                }
            } else if let Some(clause) = normalize_clause_generic(raw) {
                all_clauses_are_three &= clause.len == 3;
                for &lit in &clause.lits[..clause.len as usize] {
                    let v = lit_var_index(lit);
                    if lit > 0 {
                        p_counts[v] += 1;
                    } else {
                        n_counts[v] += 1;
                    }
                    cl.push(lit);
                }
                co.push(cl.len() as u32);
            }
        }

        let nc = co.len() - 1;
        all_clauses_are_three &= nc > 0 || cl.is_empty();

        let mut p_off = vec![0u32; nv + 1];
        let mut n_off = vec![0u32; nv + 1];
        for v in 0..nv {
            let p_count = p_counts[v];
            let n_count = n_counts[v];
            p_counts[v] = p_off[v];
            n_counts[v] = n_off[v];
            p_off[v + 1] = p_off[v] + p_count;
            n_off[v + 1] = n_off[v] + n_count;
        }

        let mut p_items = vec![0u32; p_off[nv] as usize];
        let mut n_items = vec![0u32; n_off[nv] as usize];
        let mut p_write = p_counts;
        let mut n_write = n_counts;

        for c in 0..nc {
            for i in co[c] as usize..co[c + 1] as usize {
                let lit = cl[i];
                let v = lit_var_index(lit);
                if lit > 0 {
                    p_items[p_write[v] as usize] = c as u32;
                    p_write[v] += 1;
                } else {
                    n_items[n_write[v] as usize] = c as u32;
                    n_write[v] += 1;
                }
            }
        }

        Self {
            nv,
            nc,
            cl,
            co,
            all_clauses_are_three,
            p_off,
            n_off,
            p_items,
            n_items,
        }
    }

    #[inline(always)]
    pub(crate) fn pos_occ(&self, v: usize) -> &[u32] {
        csr_slice(&self.p_off, &self.p_items, v)
    }

    #[inline(always)]
    pub(crate) fn neg_occ(&self, v: usize) -> &[u32] {
        csr_slice(&self.n_off, &self.n_items, v)
    }

    #[inline(always)]
    pub(crate) fn pos_occ_len(&self, v: usize) -> usize {
        (self.p_off[v + 1] - self.p_off[v]) as usize
    }

    #[inline(always)]
    pub(crate) fn neg_occ_len(&self, v: usize) -> usize {
        (self.n_off[v + 1] - self.n_off[v]) as usize
    }
}

#[inline(always)]
fn append_counted_lit(lit: i32, cl: &mut Vec<i32>, p_counts: &mut [u32], n_counts: &mut [u32]) {
    let v = lit_var_index(lit);
    if lit > 0 {
        p_counts[v] += 1;
    } else {
        n_counts[v] += 1;
    }
    cl.push(lit);
}

#[inline(always)]
fn append_normalized_clause3(
    a: i32,
    b: i32,
    c: i32,
    cl: &mut Vec<i32>,
    p_counts: &mut [u32],
    n_counts: &mut [u32],
) -> bool {
    if a != 0 && b != 0 && c != 0 && b != a && b != -a && c != a && c != -a && c != b && c != -b {
        append_counted_lit(a, cl, p_counts, n_counts);
        append_counted_lit(b, cl, p_counts, n_counts);
        append_counted_lit(c, cl, p_counts, n_counts);
        return true;
    }

    let mut first = 0i32;
    let mut second = 0i32;
    let mut third = 0i32;
    let mut len = 0u8;

    if a != 0 {
        first = a;
        len = 1;
    }

    if b != 0 {
        if len > 0 {
            if b == -first {
                return false;
            }
            if b != first {
                second = b;
                len = 2;
            }
        } else {
            first = b;
            len = 1;
        }
    }

    if c != 0 {
        if len > 0 {
            if c == -first {
                return false;
            }
            if c != first {
                if len > 1 {
                    if c == -second {
                        return false;
                    }
                    if c != second {
                        third = c;
                        len = 3;
                    }
                } else {
                    second = c;
                    len = 2;
                }
            }
        } else {
            first = c;
            len = 1;
        }
    }

    match len {
        0 => false,
        1 => {
            append_counted_lit(first, cl, p_counts, n_counts);
            true
        }
        2 => {
            append_counted_lit(first, cl, p_counts, n_counts);
            append_counted_lit(second, cl, p_counts, n_counts);
            true
        }
        3 => {
            append_counted_lit(first, cl, p_counts, n_counts);
            append_counted_lit(second, cl, p_counts, n_counts);
            append_counted_lit(third, cl, p_counts, n_counts);
            true
        }
        _ => unreachable!(),
    }
}

pub(crate) fn normalize_clause(raw: &[i32]) -> Option<SmallClause> {
    if raw.len() >= 3 {
        return normalize_clause3(raw[0], raw[1], raw[2]);
    }

    normalize_clause_generic(raw)
}

#[inline(always)]
fn normalize_clause3(a: i32, b: i32, c: i32) -> Option<SmallClause> {
    let Some((lits, len)) = normalize_clause3_lits(a, b, c) else {
        return None;
    };

    Some(SmallClause { lits, len })
}

#[inline(always)]
fn normalize_clause3_lits(a: i32, b: i32, c: i32) -> Option<([i32; 3], u8)> {
    let mut out = [0i32; 3];
    let mut len = 0u8;

    if a != 0 {
        out[0] = a;
        len = 1;
    }

    if b != 0 {
        if len > 0 {
            let first = out[0];
            if b == -first {
                return None;
            }
            if b != first {
                out[len as usize] = b;
                len += 1;
            }
        } else {
            out[0] = b;
            len = 1;
        }
    }

    if c != 0 {
        if len > 0 {
            let first = out[0];
            if c == -first {
                return None;
            }
            if c != first {
                if len > 1 {
                    let second = out[1];
                    if c == -second {
                        return None;
                    }
                    if c != second {
                        out[2] = c;
                        len = 3;
                    }
                } else {
                    out[1] = c;
                    len = 2;
                }
            }
        } else {
            out[0] = c;
            len = 1;
        }
    }

    if len == 0 {
        None
    } else {
        Some((out, len))
    }
}

fn normalize_clause_generic(raw: &[i32]) -> Option<SmallClause> {
    let mut out = [0i32; 3];
    let mut len = 0usize;

    'lit_loop: for &lit in raw.iter().take(3) {
        if lit == 0 {
            continue;
        }
        for &prev in &out[..len] {
            if prev == lit {
                continue 'lit_loop;
            }
            if prev == -lit {
                return None;
            }
        }
        out[len] = lit;
        len += 1;
    }

    if len == 0 {
        None
    } else {
        Some(SmallClause {
            lits: out,
            len: len as u8,
        })
    }
}

#[inline(always)]
fn csr_slice<'a>(off: &[u32], items: &'a [u32], v: usize) -> &'a [u32] {
    let s = off[v] as usize;
    let e = off[v + 1] as usize;
    &items[s..e]
}

#[inline(always)]
pub(crate) fn is_lit_sat(lit: i32, vars: &[bool]) -> bool {
    let v = lit_var_index(lit);
    if lit > 0 {
        vars[v]
    } else {
        !vars[v]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formula_lit_var_index_matches_unsigned_abs_index_for_valid_literals() {
        for lit in [-128_i32, -7, -1, 1, 7, 128] {
            assert_eq!(lit_var_index(lit), lit.unsigned_abs() as usize - 1);
        }
    }

    #[test]
    fn formula_occurrence_lists_preserve_active_clause_order() {
        let clauses = vec![
            vec![1, -2, 3],
            vec![-1, 2, 3],
            vec![1, 2, -3],
            vec![1, 1, -2],
            vec![2, -2, 3],
            vec![0, 0, 0],
            vec![0, 2, 2],
            vec![1, 2, 3, -1],
        ];
        let formula = Formula::from_raw(3, &clauses);

        assert_eq!(formula.nc, 6);
        assert_eq!(formula.co, vec![0, 3, 6, 9, 11, 12, 15]);
        assert_eq!(
            formula.cl,
            vec![1, -2, 3, -1, 2, 3, 1, 2, -3, 1, -2, 2, 1, 2, 3]
        );
        assert_eq!(formula.pos_occ(0), &[0, 2, 3, 5]);
        assert_eq!(formula.neg_occ(0), &[1]);
        assert_eq!(formula.pos_occ(1), &[1, 2, 4, 5]);
        assert_eq!(formula.neg_occ(1), &[0, 3]);
        assert_eq!(formula.pos_occ(2), &[0, 1, 5]);
        assert_eq!(formula.neg_occ(2), &[2]);
    }

    #[test]
    fn formula_tracks_all_retained_three_literal_clauses() {
        let all_three = Formula::from_raw(4, &[vec![1, -2, 3], vec![-1, 2, -3], vec![1, 3, -4]]);
        assert!(all_three.all_clauses_are_three);

        let shortened = Formula::from_raw(3, &[vec![1, 2, 3], vec![1, 1, -2]]);
        assert!(!shortened.all_clauses_are_three);
    }

    #[test]
    fn normalize_clause_fast_path_matches_generic_reference() {
        let cases = [
            vec![1, 2, 3],
            vec![1, 1, -2],
            vec![1, -1, 2],
            vec![0, 0, 0],
            vec![0, 2, 2],
            vec![1, 0, -1],
            vec![1, 2, 3, -1],
            vec![1, -1],
            vec![0, 2],
            vec![0],
            vec![],
        ];

        for raw in cases {
            assert_eq!(normalize_clause(&raw), normalize_clause_reference(&raw));
        }
    }

    #[test]
    fn append_clause3_common_path_writes_literals_and_counts_in_order() {
        let mut cl = Vec::new();
        let mut p_counts = vec![0u32; 3];
        let mut n_counts = vec![0u32; 3];

        assert!(append_normalized_clause3(
            1,
            -2,
            3,
            &mut cl,
            &mut p_counts,
            &mut n_counts
        ));

        assert_eq!(cl, vec![1, -2, 3]);
        assert_eq!(p_counts, vec![1, 0, 1]);
        assert_eq!(n_counts, vec![0, 1, 0]);
    }

    fn normalize_clause_reference(raw: &[i32]) -> Option<SmallClause> {
        let mut out = [0i32; 3];
        let mut len = 0usize;

        'lit_loop: for &lit in raw.iter().take(3) {
            if lit == 0 {
                continue;
            }
            for &prev in &out[..len] {
                if prev == lit {
                    continue 'lit_loop;
                }
                if prev == -lit {
                    return None;
                }
            }
            out[len] = lit;
            len += 1;
        }

        if len == 0 {
            None
        } else {
            Some(SmallClause {
                lits: out,
                len: len as u8,
            })
        }
    }
}
