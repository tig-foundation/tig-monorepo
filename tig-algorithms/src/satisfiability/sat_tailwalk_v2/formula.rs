use tig_challenges::satisfiability::Challenge;

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
        let mut active = Vec::with_capacity(clauses.len());
        for raw in clauses {
            if let Some(c) = normalize_clause(raw) {
                active.push(c);
            }
        }

        let nc = active.len();
        let mut cl = Vec::with_capacity(nc * 3);
        let mut co = Vec::with_capacity(nc + 1);
        let mut p_counts = vec![0u32; nv];
        let mut n_counts = vec![0u32; nv];
        co.push(0);

        for clause in &active {
            for &lit in &clause.lits[..clause.len as usize] {
                let v = lit.unsigned_abs() as usize - 1;
                if lit > 0 {
                    p_counts[v] += 1;
                } else {
                    n_counts[v] += 1;
                }
                cl.push(lit);
            }
            co.push(cl.len() as u32);
        }

        let mut p_off = vec![0u32; nv + 1];
        let mut n_off = vec![0u32; nv + 1];
        for v in 0..nv {
            p_off[v + 1] = p_off[v] + p_counts[v];
            n_off[v + 1] = n_off[v] + n_counts[v];
        }

        let mut p_items = vec![0u32; p_off[nv] as usize];
        let mut n_items = vec![0u32; n_off[nv] as usize];
        let mut p_write = p_off[..nv].to_vec();
        let mut n_write = n_off[..nv].to_vec();

        for c in 0..nc {
            for i in co[c] as usize..co[c + 1] as usize {
                let lit = cl[i];
                let v = lit.unsigned_abs() as usize - 1;
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

pub(crate) fn normalize_clause(raw: &[i32]) -> Option<SmallClause> {
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
    let v = lit.unsigned_abs() as usize - 1;
    if lit > 0 {
        vars[v]
    } else {
        !vars[v]
    }
}
