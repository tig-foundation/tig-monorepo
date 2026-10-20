// fast100k — engine for the two n_vars=100000 tracks of c001 (ratio 4150 and 4200).
//
// Pipeline (both tracks): CSR preprocessing -> survey-propagation "reinforcement" seed
// (Chavas, Furtlehner, Mezard, Zecchina, J. Stat. Mech. 2005 P11016; same update equations and
// constants as c001_a123/a127 `engine_spr` by Fred and as flipwave by NVX) stopped as soon as the
// current assignment leaves at most `target` unsatisfied clauses -> probSAT local search
// (break-only exponential distribution, Balint & Schoening, SAT 2012) -> on failure a second seed
// with another RNG stream, then an uncapped probSAT walk from a polarity-majority assignment.
//
// Implementation notes (this is where the time goes):
//  * SP kernel: f32, fully unrolled three-literal clause body, scatter unrolled by four, one division
//    per clause for the three cavity ratios, `get_unchecked` in the hot loops; the per-variable pass
//    also prepares the literal products of the next iteration.
//  * walk: u8 true-literal counters, per-literal occurrence CSR, swap-remove unsatisfied list,
//    Lemire range reduction instead of modulo, break values computed on demand (non-caching).
use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::satisfiability::*;

#[derive(Clone, Copy, Debug)]
pub struct Cfg {
    pub rate: f32,
    pub gmax: f32,
    pub lam: f32,
    pub damp: f32,
    pub check: usize,
    pub patience: usize,
    pub maxit: usize,
    pub target: usize,
    pub cb: f64,
    pub flip_cap0: u64,
    pub flip_cap1: u64,
    pub restart: u64,
    pub asyncu: bool,
    pub resync: usize,
    pub trace: bool,
}

impl Cfg {
    pub fn for_4200() -> Self {
        Cfg { rate: 0.0017, gmax: 0.5, lam: 0.1, damp: 0.0, check: 10, patience: 150, maxit: 1500, target: 200,
              cb: 2.8, flip_cap0: 60_000_000, flip_cap1: 200_000_000, restart: 0, asyncu: true, resync: 16, trace: false }
    }
    pub fn for_4150() -> Self {
        Cfg { rate: 0.012, gmax: 0.5, lam: 0.1, damp: 0.0, check: 5, patience: 150, maxit: 1500, target: 200,
              cb: 2.8, flip_cap0: 60_000_000, flip_cap1: 200_000_000, restart: 0, asyncu: true, resync: 16, trace: false }
    }
}

const FLOOR: f32 = 1e-6;

pub(crate) struct Formula {
    nv: usize,
    nc: usize,
    lit: Vec<u32>,     // encoded literal (v << 1 | positive) per edge, clause-contiguous
    co: Vec<u32>,      // nc + 1 clause offsets into `lit`
    occ: Vec<u32>,     // clause ids grouped by literal
    occ_off: Vec<u32>, // 2 nv + 1
}

#[inline(always)]
fn enc(l: i32) -> u32 {
    ((l.unsigned_abs() - 1) << 1) | (l > 0) as u32
}

/// Distinct encoded literals of a clause, or 0 literals if the clause is a tautology.
#[inline(always)]
fn norm(cl: &[i32]) -> ([u32; 3], usize) {
    let mut out = [0u32; 3];
    let mut k = 0usize;
    for i in 0..cl.len() {
        let a = cl[i];
        for j in (i + 1)..cl.len() {
            if cl[j] == -a {
                return (out, 0);
            }
        }
        let e = enc(a);
        if !out[..k].contains(&e) && k < 3 {
            out[k] = e;
            k += 1;
        }
    }
    (out, k)
}

fn build(ch: &Challenge) -> Formula {
    let nv = ch.num_variables;
    let mut occ_cnt = vec![0u32; 2 * nv];
    let mut nc = 0usize;
    let mut ne = 0usize;
    for cl in &ch.clauses {
        let (ls, k) = norm(cl);
        if k == 0 { continue; }
        for q in 0..k { occ_cnt[ls[q] as usize] += 1; }
        nc += 1;
        ne += k;
    }
    let mut occ_off = vec![0u32; 2 * nv + 1];
    for l in 0..2 * nv { occ_off[l + 1] = occ_off[l] + occ_cnt[l]; }
    let mut fill: Vec<u32> = occ_off[..2 * nv].to_vec();
    let mut lit: Vec<u32> = Vec::with_capacity(ne);
    let mut co: Vec<u32> = Vec::with_capacity(nc + 1);
    let mut occ = vec![0u32; ne];
    co.push(0);
    let mut ci = 0u32;
    for cl in &ch.clauses {
        let (ls, k) = norm(cl);
        if k == 0 { continue; }
        for q in 0..k {
            let l = ls[q] as usize;
            lit.push(ls[q]);
            let p = fill[l] as usize;
            occ[p] = ci;
            fill[l] += 1;
        }
        co.push(lit.len() as u32);
        ci += 1;
    }
    Formula { nv, nc, lit, co, occ, occ_off }
}

/// Survey-propagation reinforcement seed. Returns (assignment, unsatisfied clauses of that assignment, iterations).
fn sp_seed(f: &Formula, seed: u64, cfg: &Cfg) -> (Vec<bool>, usize, usize) {
    let nv = f.nv;
    let nc = f.nc;
    let lit = &f.lit[..];
    let co = &f.co[..];
    let ne = lit.len();
    let mut s = seed | 1;
    let mut eta: Vec<f32> = Vec::with_capacity(ne);
    for _ in 0..ne {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        eta.push(((s >> 40) as f32) * (1.0 / 16_777_216.0));
    }
    let mut p = vec![1f32; 2 * nv];
    let mut hf = vec![0f32; 2 * nv];
    let mut x = vec![false; nv];
    let mut best_x = vec![false; nv];
    let mut best_u = usize::MAX;
    let mut best_it = 0usize;
    let mut it = 0usize;
    let damp = cfg.damp;
    let nd = 1.0 - damp;
    let lam = cfg.lam;
    let nl = 1.0 - lam;
    let check = cfg.check.max(1);
    while it < cfg.maxit {
        it += 1;
        let gamma = (cfg.rate * it as f32).min(cfg.gmax);
        let full = !cfg.asyncu || it == 1 || (it - 1) % cfg.resync.max(1) == 0;
        unsafe {
            if full {
            if cfg.asyncu { for v in 0..nv { *p.get_unchecked_mut(2 * v) = 1.0 - *hf.get_unchecked(2 * v); *p.get_unchecked_mut(2 * v + 1) = 1.0 - *hf.get_unchecked(2 * v + 1); } }
            // products per literal: p[l] = (1 - h_l) * prod_{e in occ(l)} (1 - eta_e)   (p holds 1 - h_l on entry)
            let m4 = ne & !3usize;
            let mut e = 0usize;
            while e < m4 {
                let o0 = (1.0 - *eta.get_unchecked(e)).max(FLOOR);
                *p.get_unchecked_mut(*lit.get_unchecked(e) as usize) *= o0;
                let o1 = (1.0 - *eta.get_unchecked(e + 1)).max(FLOOR);
                *p.get_unchecked_mut(*lit.get_unchecked(e + 1) as usize) *= o1;
                let o2 = (1.0 - *eta.get_unchecked(e + 2)).max(FLOOR);
                *p.get_unchecked_mut(*lit.get_unchecked(e + 2) as usize) *= o2;
                let o3 = (1.0 - *eta.get_unchecked(e + 3)).max(FLOOR);
                *p.get_unchecked_mut(*lit.get_unchecked(e + 3) as usize) *= o3;
                e += 4;
            }
            while e < ne {
                let o = (1.0 - *eta.get_unchecked(e)).max(FLOOR);
                *p.get_unchecked_mut(*lit.get_unchecked(e) as usize) *= o;
                e += 1;
            }
            }
            let au = cfg.asyncu;
            // clause pass: cavity ratios and new surveys (fully unrolled for 3-literal clauses)
            for a in 0..nc {
                let cs = *co.get_unchecked(a) as usize;
                let ce = *co.get_unchecked(a + 1) as usize;
                if ce - cs == 3 {
                    let l0 = *lit.get_unchecked(cs) as usize;
                    let l1 = *lit.get_unchecked(cs + 1) as usize;
                    let l2 = *lit.get_unchecked(cs + 2) as usize;
                    let t0 = *eta.get_unchecked(cs);
                    let t1 = *eta.get_unchecked(cs + 1);
                    let t2 = *eta.get_unchecked(cs + 2);
                    let av0 = *p.get_unchecked(l0) / (1.0 - t0).max(FLOOR);
                    let av1 = *p.get_unchecked(l1) / (1.0 - t1).max(FLOOR);
                    let av2 = *p.get_unchecked(l2) / (1.0 - t2).max(FLOOR);
                    let bv0 = *p.get_unchecked(l0 ^ 1);
                    let bv1 = *p.get_unchecked(l1 ^ 1);
                    let bv2 = *p.get_unchecked(l2 ^ 1);
                    let u0 = (1.0 - bv0) * av0;
                    let u1 = (1.0 - bv1) * av1;
                    let u2 = (1.0 - bv2) * av2;
                    let d0 = u0 + (1.0 - av0) * bv0 + av0 * bv0;
                    let d1 = u1 + (1.0 - av1) * bv1 + av1 * bv1;
                    let d2 = u2 + (1.0 - av2) * bv2 + av2 * bv2;
                    let dd = d0 * d1 * d2;
                    let (n0, n1, n2) = if dd > 1e-30 {
                        let id = 1.0 / dd;
                        ((u1 * u2 * d0 * id).min(1.0), (u0 * u2 * d1 * id).min(1.0), (u0 * u1 * d2 * id).min(1.0))
                    } else {
                        let r0 = if d0 > 0.0 { u0 / d0 } else { 0.0 };
                        let r1 = if d1 > 0.0 { u1 / d1 } else { 0.0 };
                        let r2 = if d2 > 0.0 { u2 / d2 } else { 0.0 };
                        ((r1 * r2).min(1.0), (r0 * r2).min(1.0), (r0 * r1).min(1.0))
                    };
                    let w0 = damp * t0 + nd * n0;
                    let w1 = damp * t1 + nd * n1;
                    let w2 = damp * t2 + nd * n2;
                    *eta.get_unchecked_mut(cs) = w0;
                    *eta.get_unchecked_mut(cs + 1) = w1;
                    *eta.get_unchecked_mut(cs + 2) = w2;
                    if au {
                        *p.get_unchecked_mut(l0) = av0 * (1.0 - w0).max(FLOOR);
                        *p.get_unchecked_mut(l1) = av1 * (1.0 - w1).max(FLOOR);
                        *p.get_unchecked_mut(l2) = av2 * (1.0 - w2).max(FLOOR);
                    }
                } else {
                    let k = ce - cs;
                    let mut r = [0f32; 3];
                    for q in 0..k {
                        let e = cs + q;
                        let l = *lit.get_unchecked(e) as usize;
                        let av = *p.get_unchecked(l) / (1.0 - *eta.get_unchecked(e)).max(FLOOR);
                        let bv = *p.get_unchecked(l ^ 1);
                        let u = (1.0 - bv) * av;
                        let dn = u + (1.0 - av) * bv + av * bv;
                        r[q] = if dn > 0.0 { u / dn } else { 0.0 };
                    }
                    for q in 0..k {
                        let mut prod = 1f32;
                        for j in 0..k { if j != q { prod *= r[j]; } }
                        let e = cs + q;
                        let told = *eta.get_unchecked(e);
                        let w = damp * told + nd * prod.min(1.0);
                        *eta.get_unchecked_mut(e) = w;
                        if au {
                            let l = *lit.get_unchecked(e) as usize;
                            let pl = p.get_unchecked_mut(l);
                            *pl = (*pl / (1.0 - told).max(FLOOR)) * (1.0 - w).max(FLOOR);
                        }
                    }
                }
            }
            // variable pass: marginals, reinforcement fields, next products
            for v in 0..nv {
                let pm = *p.get_unchecked(2 * v);
                let pp = *p.get_unchecked(2 * v + 1);
                let pip = (1.0 - pp) * pm;
                let pim = (1.0 - pm) * pp;
                let den = pip + pim + pp * pm;
                let idn = if den > 0.0 { 1.0 / den } else { 0.0 };
                let wp = pip * idn;
                let wm = pim * idn;
                *x.get_unchecked_mut(v) = wp > wm;
                let hp = nl * *hf.get_unchecked(2 * v + 1) + lam * gamma * wp;
                let hm = nl * *hf.get_unchecked(2 * v) + lam * gamma * wm;
                let hp_old = *hf.get_unchecked(2 * v + 1);
                let hm_old = *hf.get_unchecked(2 * v);
                *hf.get_unchecked_mut(2 * v + 1) = hp;
                *hf.get_unchecked_mut(2 * v) = hm;
                if au {
                    *p.get_unchecked_mut(2 * v + 1) = pp * ((1.0 - hp) / (1.0 - hp_old));
                    *p.get_unchecked_mut(2 * v) = pm * ((1.0 - hm) / (1.0 - hm_old));
                } else {
                    *p.get_unchecked_mut(2 * v + 1) = 1.0 - hp;
                    *p.get_unchecked_mut(2 * v) = 1.0 - hm;
                }
            }
        }
        if it % check == 0 {
            let u = count_unsat(f, &x);
            if u < best_u {
                best_u = u;
                best_it = it;
                best_x.copy_from_slice(&x);
                if u == 0 { break; }
            } else if it - best_it >= cfg.patience {
                break;
            }
            if best_u <= cfg.target { break; }
        }
    }
    (best_x, best_u, it)
}

fn count_unsat(f: &Formula, x: &[bool]) -> usize {
    let lit = &f.lit[..];
    let co = &f.co[..];
    let mut u = 0usize;
    unsafe {
        for a in 0..f.nc {
            let cs = *co.get_unchecked(a) as usize;
            let ce = *co.get_unchecked(a + 1) as usize;
            if ce - cs == 3 {
                let l0 = *lit.get_unchecked(cs) as usize;
                let l1 = *lit.get_unchecked(cs + 1) as usize;
                let l2 = *lit.get_unchecked(cs + 2) as usize;
                let s0 = ((l0 & 1) == 1) == *x.get_unchecked(l0 >> 1);
                let s1 = ((l1 & 1) == 1) == *x.get_unchecked(l1 >> 1);
                let s2 = ((l2 & 1) == 1) == *x.get_unchecked(l2 >> 1);
                u += (!(s0 | s1 | s2)) as usize;
            } else {
                let mut ok = false;
                for e in cs..ce {
                    let l = *lit.get_unchecked(e) as usize;
                    if ((l & 1) == 1) == *x.get_unchecked(l >> 1) { ok = true; break; }
                }
                if !ok { u += 1; }
            }
        }
    }
    u
}

/// probSAT (break-only, exponential cb^-break). Returns (solved, flips).
fn walk(f: &Formula, vars: &mut [bool], rng: &mut SmallRng, cb: f64, cap: u64, restart: u64) -> (bool, u64) {
    let nc = f.nc;
    let start: Vec<bool> = if restart > 0 { vars.to_vec() } else { Vec::new() };
    let mut gap = restart;
    let mut best_n = usize::MAX;
    let mut best_at = 0u64;
    let lit = &f.lit[..];
    let co = &f.co[..];
    let occ = &f.occ[..];
    let occ_off = &f.occ_off[..];
    let mut ng = vec![0u8; nc];
    let mut unsat: Vec<u32> = Vec::with_capacity(nc);
    let mut pos = vec![u32::MAX; nc];
    for a in 0..nc {
        let mut t = 0u8;
        for e in co[a] as usize..co[a + 1] as usize {
            let l = lit[e];
            if ((l & 1) == 1) == vars[(l >> 1) as usize] { t += 1; }
        }
        ng[a] = t;
        if t == 0 { pos[a] = unsat.len() as u32; unsat.push(a as u32); }
    }
    let mut tab = [0f64; 64];
    for b in 0..64 { tab[b] = cb.powi(-(b as i32)); }
    let mut flips = 0u64;
    unsafe {
        loop {
            let n = unsat.len();
            if n == 0 { return (true, flips); }
            if flips >= cap { return (false, flips); }
            if restart > 0 {
                if n < best_n { best_n = n; best_at = flips; }
                else if flips - best_at > gap {
                    // stagnation: back to the seed with the next RNG draws (a different trajectory), growing gap
                    vars.copy_from_slice(&start);
                    unsat.clear();
                    for a in 0..nc {
                        let mut t = 0u8;
                        for e in co[a] as usize..co[a + 1] as usize {
                            let l = lit[e];
                            if ((l & 1) == 1) == vars[(l >> 1) as usize] { t += 1; }
                        }
                        *ng.get_unchecked_mut(a) = t;
                        if t == 0 { *pos.get_unchecked_mut(a) = unsat.len() as u32; unsat.push(a as u32); } else { *pos.get_unchecked_mut(a) = u32::MAX; }
                    }
                    best_n = unsat.len(); best_at = flips; gap += gap / 2;
                    continue;
                }
            }
            let r = rng.gen::<u64>();
            let a = *unsat.get_unchecked((((r >> 32) * n as u64) >> 32) as usize) as usize;
            let cs = *co.get_unchecked(a) as usize;
            let ce = *co.get_unchecked(a + 1) as usize;
            let k = ce - cs;
            let mut brk = [0u32; 3];
            let mut zeros = 0u32;
            let mut zmask = 0u32;
            for q in 0..k {
                let t = (*lit.get_unchecked(cs + q) ^ 1) as usize; // the literal currently true
                let os = *occ_off.get_unchecked(t) as usize;
                let oe = *occ_off.get_unchecked(t + 1) as usize;
                let mut b = 0u32;
                for j in os..oe {
                    let c = *occ.get_unchecked(j) as usize;
                    b += (*ng.get_unchecked(c) == 1) as u32;
                }
                brk[q] = b;
                if b == 0 { zeros += 1; zmask |= 1 << q; }
            }
            let q = if zeros > 0 {
                if zeros == 1 {
                    zmask.trailing_zeros() as usize
                } else {
                    let mut idx = ((r & 0xffff) * zeros as u64) >> 16;
                    let mut sel = 0usize;
                    for qq in 0..k {
                        if zmask & (1 << qq) != 0 {
                            if idx == 0 { sel = qq; break; }
                            idx -= 1;
                        }
                    }
                    sel
                }
            } else {
                let mut w = [0f64; 3];
                let mut sum = 0f64;
                for qq in 0..k {
                    let wq = *tab.get_unchecked((brk[qq] as usize).min(63));
                    w[qq] = wq;
                    sum += wq;
                }
                let thr = rng.gen::<f64>() * sum;
                let mut acc = 0f64;
                let mut sel = k - 1;
                for qq in 0..k {
                    acc += w[qq];
                    if acc >= thr { sel = qq; break; }
                }
                sel
            };
            let l = *lit.get_unchecked(cs + q) as usize; // becomes true
            let t = l ^ 1;                                // becomes false
            let v = l >> 1;
            for j in *occ_off.get_unchecked(l) as usize..*occ_off.get_unchecked(l + 1) as usize {
                let c = *occ.get_unchecked(j) as usize;
                let g = *ng.get_unchecked(c);
                *ng.get_unchecked_mut(c) = g + 1;
                if g == 0 {
                    let pz = *pos.get_unchecked(c) as usize;
                    let last = *unsat.get_unchecked(unsat.len() - 1);
                    unsat.set_len(unsat.len() - 1);
                    if pz < unsat.len() {
                        *unsat.get_unchecked_mut(pz) = last;
                        *pos.get_unchecked_mut(last as usize) = pz as u32;
                    }
                    *pos.get_unchecked_mut(c) = u32::MAX;
                }
            }
            for j in *occ_off.get_unchecked(t) as usize..*occ_off.get_unchecked(t + 1) as usize {
                let c = *occ.get_unchecked(j) as usize;
                let g = *ng.get_unchecked(c);
                *ng.get_unchecked_mut(c) = g - 1;
                if g == 1 {
                    *pos.get_unchecked_mut(c) = unsat.len() as u32;
                    let ul = unsat.len();
                    *unsat.get_unchecked_mut(ul) = c as u32;
                    unsat.set_len(ul + 1);
                }
            }
            *vars.get_unchecked_mut(v) = !*vars.get_unchecked(v);
            flips += 1;
        }
    }
}

fn majority_init(f: &Formula, rng: &mut SmallRng) -> Vec<bool> {
    let nv = f.nv;
    let mut x = vec![false; nv];
    for v in 0..nv {
        let np = f.occ_off[2 * v + 2] - f.occ_off[2 * v + 1];
        let nn = f.occ_off[2 * v + 1] - f.occ_off[2 * v];
        x[v] = if np != nn { np > nn } else { rng.gen::<bool>() };
    }
    x
}

fn verify(ch: &Challenge, x: &[bool]) -> bool {
    if x.len() != ch.num_variables { return false; }
    for cl in &ch.clauses {
        let mut ok = false;
        for &l in cl.iter() {
            if (l > 0) == x[(l.unsigned_abs() - 1) as usize] { ok = true; break; }
        }
        if !ok { return false; }
    }
    true
}

pub fn solve(ch: &Challenge, save_solution: &dyn Fn(&Solution) -> Result<()>, cfg: &Cfg, track: u32) -> Result<()> {
    let f = build(ch);
    let seed0 = u64::from_le_bytes(ch.seed[..8].try_into().unwrap());
    let mut rng = SmallRng::seed_from_u64(seed0 ^ 0xA5A5_5A5A_0C00_1100);
    for phase in 0..2u64 {
        // phase 1: another RNG stream and a more conservative reinforcement schedule (a seed that stalled
        // at the nominal rate must not stall again for the same reason)
        let mut c1 = *cfg;
        if phase == 1 { c1.rate = cfg.rate * 0.75; c1.target = cfg.target + cfg.target / 2; c1.patience = cfg.patience * 2; }
        let cfg = &c1;
        let (mut x, u, it) = sp_seed(&f, seed0.wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(phase + 1)), cfg);
        let cap = if phase == 0 { cfg.flip_cap0 } else { cfg.flip_cap1 };
        let (ok, flips) = walk(&f, &mut x, &mut rng, cfg.cb, cap, cfg.restart);
        if ok && verify(ch, &x) {
            save_solution(&Solution { variables: x })?;
            return Ok(());
        }
    }
    let mut x = majority_init(&f, &mut rng);
    let (ok, _flips) = walk(&f, &mut x, &mut rng, cfg.cb, u64::MAX, 0);
    if ok && verify(ch, &x) {
        save_solution(&Solution { variables: x })?;
    }
    Ok(())
}
