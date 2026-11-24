use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

#[derive(Clone, Debug)]
pub struct P {
    pub sp: bool,
    pub g0: f32,
    pub rate: f32,
    pub gmax: f32,
    pub lam: f32,
    pub damp: f32,
    pub maxit: usize,
    pub check: usize,
    pub stop_u: usize,
    pub refresh: usize,
    pub cb: f64,
    pub max_flips: u64,
    pub attempts: usize,
    pub giveup: Vec<(u64, usize)>,
}

impl P {
    pub fn for_instance(nv: usize, m: usize) -> P {
        let base = P { sp: true, g0: 0.06, rate: 0.001, gmax: 0.5, lam: 0.1, damp: 0.0, maxit: 1000, check: 10,
                       stop_u: 550, refresh: 10, cb: 3.0, max_flips: 30_000_000, attempts: 4, giveup: Vec::new() };
        let ratio = m as f64 / nv.max(1) as f64;
        if nv >= 50_000 && ratio < 4.175 {
            P { g0: 0.12, rate: 0.003, stop_u: 250, ..base }
        } else if nv >= 50_000 {
            P { g0: 0.07, rate: 0.0008, stop_u: 250, ..base }
        } else {
            let (giveup, max_flips) = if nv <= 6000 {
                (vec![(1 << 26, 4), (1 << 29, 1)], 4_000_000_000)
            } else if nv <= 8500 {
                (vec![(1 << 25, 8), (1 << 26, 6), (1 << 28, 2)], 3_000_000_000)
            } else {
                (vec![(1 << 26, 14), (1 << 28, 6), (1 << 29, 2), (1 << 30, 1)], 4_000_000_000)
            };
            P { sp: false, cb: 2.2, max_flips, attempts: 1, giveup, ..base }
        }
    }
}

#[derive(Default, Debug)]
pub struct Stats {
    pub attempts: usize,
    pub sp_it: usize,
    pub sp_u: usize,
    pub flips: u64,
}

pub struct Inst {
    pub nv: usize,
    pub nvx: usize,
    pub nc: usize,
    pub lits: Vec<u32>,
    pub deg: Vec<u32>,
    pub off: Vec<u32>,
    pub occ: Vec<u32>,
}

pub fn build(ch: &Challenge) -> Inst {
    let nv = ch.num_variables;
    let m = ch.clauses.len();
    let mut lits = vec![0u32; 3 * m + 3];
    let mut special: Vec<usize> = Vec::new();
    let mut nc = 0usize;
    unsafe {
        let lp = lits.as_mut_ptr();
        for (i, c) in ch.clauses.iter().enumerate() {
            let (a, b, d) = (*c.get_unchecked(0), *c.get_unchecked(1), *c.get_unchecked(2));
            let ea = (((a.abs() - 1) as u32) << 1) | (a > 0) as u32;
            let eb = (((b.abs() - 1) as u32) << 1) | (b > 0) as u32;
            let ed = (((d.abs() - 1) as u32) << 1) | (d > 0) as u32;
            *lp.add(3 * nc) = ea;
            *lp.add(3 * nc + 1) = eb;
            *lp.add(3 * nc + 2) = ed;
            let same = ((ea >> 1) == (eb >> 1)) | ((ea >> 1) == (ed >> 1)) | ((eb >> 1) == (ed >> 1));
            nc += (!same) as usize;
            if same { special.push(i); }
        }
    }
    lits.truncate(3 * nc);
    let mut nvx = nv;
    for &i in &special {
        let mut l = [0u32; 3];
        let mut k = 0;
        let mut taut = false;
        for &x in &ch.clauses[i] {
            let e = (((x.abs() - 1) as u32) << 1) | (x > 0) as u32;
            if l[..k].contains(&e) { continue; }
            if l[..k].contains(&(e ^ 1)) { taut = true; break; }
            l[k] = e;
            k += 1;
        }
        if taut { continue; }
        while k < 3 { l[k] = ((nvx as u32) << 1) | 1; nvx += 1; k += 1; }
        lits.extend_from_slice(&l);
    }
    let nc = lits.len() / 3;
    let ne = 3 * nc;
    let mut deg = vec![0u32; 2 * nvx];
    let mut off = vec![0u32; 2 * nvx + 1];
    unsafe {
        let lp = lits.as_ptr();
        let dp = deg.as_mut_ptr();
        for e in 0..ne { *dp.add(*lp.add(e) as usize) += 1; }
        for l in 0..2 * nvx { off[l + 1] = off[l] + ((*dp.add(l) + 3) & !3); }
    }
    let mut occ = vec![nc as u32; off[2 * nvx] as usize];
    unsafe {
        let lp = lits.as_ptr();
        let mut pos = off.clone();
        let pp = pos.as_mut_ptr();
        let oc = occ.as_mut_ptr();
        for c in 0..nc {
            for q in 0..3 {
                let l = *lp.add(3 * c + q) as usize;
                let k = *pp.add(l);
                *oc.add(k as usize) = c as u32;
                *pp.add(l) = k + 1;
            }
        }
    }
    Inst { nv, nvx, nc, lits, deg, off, occ }
}

#[inline(always)]
unsafe fn g<T: Copy>(v: &[T], i: usize) -> T { *v.get_unchecked(i) }
#[inline(always)]
unsafe fn gm<T>(v: &mut [T], i: usize) -> &mut T { v.get_unchecked_mut(i) }

pub fn unsat_count(inst: &Inst, x: &[u8]) -> usize {
    let mut u = 0usize;
    let lp = inst.lits.as_ptr();
    let xp = x.as_ptr();
    #[inline(always)]
    unsafe fn one(lp: *const u32, xp: *const u8, c: usize) -> usize {
        let l0 = *lp.add(3 * c);
        let l1 = *lp.add(3 * c + 1);
        let l2 = *lp.add(3 * c + 2);
        ((*xp.add((l0 >> 1) as usize) ^ (l0 & 1) as u8)
            & (*xp.add((l1 >> 1) as usize) ^ (l1 & 1) as u8)
            & (*xp.add((l2 >> 1) as usize) ^ (l2 & 1) as u8)) as usize
    }
    unsafe {
        let mut c = 0;
        #[inline(always)]
        unsafe fn one4(lp: *const u32, xp: *const u8, c: usize) -> usize {
            one(lp, xp, c) + one(lp, xp, c + 1) + one(lp, xp, c + 2) + one(lp, xp, c + 3)
        }
        while c + 16 <= inst.nc {
            u += one4(lp, xp, c) + one4(lp, xp, c + 4) + one4(lp, xp, c + 8) + one4(lp, xp, c + 12);
            c += 16;
        }
        while c < inst.nc { u += one(lp, xp, c); c += 1; }
    }
    u
}

fn rand_om(ne: usize, rng: &mut SmallRng) -> Vec<f32> {
    let mut om = vec![0f32; ne];
    for ch in om.chunks_exact_mut(4) {
        for h in 0..2 {
            let r = rng.gen::<u64>();
            ch[2 * h] = 1.0 - ((r >> 40) as f32) * (1.0 / 16777216.0) * 0.999;
            ch[2 * h + 1] = 1.0 - (((r >> 8) & 0xffffff) as f32) * (1.0 / 16777216.0) * 0.999;
        }
    }
    om
}

pub fn spr(inst: &Inst, p: &P, rng: &mut SmallRng, st: &mut Stats) -> Vec<u8> {
    let (nv, nvx, nc) = (inst.nv, inst.nvx, inst.nc);
    let nc4 = (nc + 3) & !3;
    let mut ls = vec![(nvx as u32) << 1; 3 * nc4];
    unsafe {
        let lsm = ls.as_mut_ptr();
        let ilp = inst.lits.as_ptr();
        let mut c = 0;
        while c + 4 <= nc {
            for k in 0..4 { for q in 0..3 { *lsm.add(q * nc4 + c + k) = *ilp.add(3 * (c + k) + q); } }
            c += 4;
        }
        while c < nc {
            for q in 0..3 { *lsm.add(q * nc4 + c) = *ilp.add(3 * c + q); }
            c += 1;
        }
    }
    let mut om = rand_om(3 * nc4, rng);
    let mut pr = vec![1f32; 2 * nvx + 2];
    let mut hp = vec![0f32; nvx + 1];
    let mut hm = vec![0f32; nvx + 1];
    for dv in nv..nvx { hm[dv] = 1.0; }
    let mut x = vec![0u8; nvx];
    let mut best = vec![0u8; nvx];
    let mut best_u = usize::MAX;
    let mut best_it = 0;
    let (mut prev_u, mut prev_it, mut next_check) = (usize::MAX, 0, p.check);
    let mut it = 0;
    let d = p.damp;
    let d1 = 1.0 - d;
    let lam1 = 1.0 - p.lam;
    let prp = pr.as_mut_ptr();
    let omp = om.as_mut_ptr();
    let lsp = ls.as_ptr();
    let hpp = hp.as_mut_ptr();
    let hmp = hm.as_mut_ptr();
    let xp = x.as_mut_ptr();
    let ne = 3 * nc4;
    unsafe fn refresh(prp: *mut f32, hpp: *const f32, hmp: *const f32, omp: *const f32, lsp: *const u32, nvx: usize, ne: usize) {
        let mut v = 0;
        while v + 4 <= nvx + 1 {
            for k in 0..4 {
                *prp.add(2 * (v + k)) = 1.0 - *hmp.add(v + k);
                *prp.add(2 * (v + k) + 1) = 1.0 - *hpp.add(v + k);
            }
            v += 4;
        }
        while v <= nvx {
            *prp.add(2 * v) = 1.0 - *hmp.add(v);
            *prp.add(2 * v + 1) = 1.0 - *hpp.add(v);
            v += 1;
        }
        #[inline(always)]
        unsafe fn mul4(prp: *mut f32, omp: *const f32, lsp: *const u32, e: usize) {
            for k in 0..4 { *prp.add(*lsp.add(e + k) as usize) *= *omp.add(e + k); }
        }
        let mut e = 0;
        while e + 16 <= ne {
            mul4(prp, omp, lsp, e);
            mul4(prp, omp, lsp, e + 4);
            mul4(prp, omp, lsp, e + 8);
            mul4(prp, omp, lsp, e + 12);
            e += 16;
        }
        while e < ne {
            mul4(prp, omp, lsp, e);
            e += 4;
        }
    }
    unsafe {
        refresh(prp, hpp, hmp, omp, lsp, nvx, ne);
        while it < p.maxit {
            it += 1;
            let gamma = (p.g0 + p.rate * it as f32).min(p.gmax);
            #[inline(always)]
            unsafe fn blk(lsp: *const u32, prp: *mut f32, omp: *mut f32, nc4: usize, c: usize, d: f32, d1: f32) {
                let mut l = [[0usize; 4]; 3];
                let mut pp = [[0f32; 4]; 3];
                let mut bv = [[0f32; 4]; 3];
                let mut o = [[0f32; 4]; 3];
                for q in 0..3 {
                    for k in 0..4 {
                        let li = *lsp.add(q * nc4 + c + k) as usize;
                        l[q][k] = li;
                        pp[q][k] = *prp.add(li);
                        bv[q][k] = *prp.add(li ^ 1);
                        o[q][k] = *omp.add(q * nc4 + c + k);
                    }
                }
                let mut r = [[0f32; 4]; 3];
                for q in 0..3 {
                    for k in 0..4 {
                        let n = (1.0 - bv[q][k]) * pp[q][k];
                        r[q][k] = n / fmax(n + bv[q][k] * o[q][k], 1e-30);
                    }
                }
                let mut pe = [[0f32; 4]; 3];
                for k in 0..4 {
                    pe[0][k] = r[1][k] * r[2][k];
                    pe[1][k] = r[0][k] * r[2][k];
                    pe[2][k] = r[0][k] * r[1][k];
                }
                let mut ratio = [[0f32; 4]; 3];
                for q in 0..3 {
                    for k in 0..4 {
                        let on = fmax(d * o[q][k] + d1 * (1.0 - pe[q][k]), 1e-6);
                        *omp.add(q * nc4 + c + k) = on;
                        ratio[q][k] = on / o[q][k];
                    }
                }
                let ratio = std::hint::black_box(ratio);
                for q in 0..3 {
                    for k in 0..4 { *prp.add(l[q][k]) *= ratio[q][k]; }
                }
            }
            let mut c = 0;
            while c + 16 <= nc4 {
                blk(lsp, prp, omp, nc4, c, d, d1);
                blk(lsp, prp, omp, nc4, c + 4, d, d1);
                blk(lsp, prp, omp, nc4, c + 8, d, d1);
                blk(lsp, prp, omp, nc4, c + 12, d, d1);
                c += 16;
            }
            while c < nc4 { blk(lsp, prp, omp, nc4, c, d, d1); c += 4; }
            let lg = p.lam * gamma;
            #[inline(always)]
            unsafe fn var(prp: *mut f32, hpp: *mut f32, hmp: *mut f32, xp: *mut u8, v: usize, lam1: f32, lg: f32) {
                let pm = *prp.add(2 * v);
                let pp = *prp.add(2 * v + 1);
                let pip = (1.0 - pp) * pm;
                let pim = (1.0 - pm) * pp;
                let inv = 1.0 / fmax(pip + pim + pp * pm, 1e-30);
                let wp = pip * inv;
                let wm = pim * inv;
                let o1 = *hpp.add(v);
                let o0 = *hmp.add(v);
                let h1 = lam1 * o1 + lg * wp;
                let h0 = lam1 * o0 + lg * wm;
                *hpp.add(v) = h1;
                *hmp.add(v) = h0;
                *xp.add(v) = (wp > wm) as u8;
                *prp.add(2 * v) = pm * ((1.0 - h0) / (1.0 - o0));
                *prp.add(2 * v + 1) = pp * ((1.0 - h1) / (1.0 - o1));
            }
            let mut v = 0;
            #[inline(always)]
            unsafe fn var4(prp: *mut f32, hpp: *mut f32, hmp: *mut f32, xp: *mut u8, v: usize, lam1: f32, lg: f32) {
                for k in 0..4 { var(prp, hpp, hmp, xp, v + k, lam1, lg); }
            }
            while v + 16 <= nv {
                var4(prp, hpp, hmp, xp, v, lam1, lg);
                var4(prp, hpp, hmp, xp, v + 4, lam1, lg);
                var4(prp, hpp, hmp, xp, v + 8, lam1, lg);
                var4(prp, hpp, hmp, xp, v + 12, lam1, lg);
                v += 16;
            }
            while v < nv { var(prp, hpp, hmp, xp, v, lam1, lg); v += 1; }
            if it % p.refresh == 0 { refresh(prp, hpp, hmp, omp, lsp, nvx, ne); }
            if it == next_check {
                let u = unsat_count(inst, &x);
                let mut iv = p.check;
                if prev_it > 0 && u > p.stop_u && u < prev_u {
                    iv = ((u - p.stop_u) * (it - prev_it) / (prev_u - u)).clamp(1, p.check);
                }
                prev_u = u;
                prev_it = it;
                next_check = it + iv;
                if u < best_u {
                    best_u = u;
                    best_it = it;
                    best.copy_from_slice(&x);
                    if u == 0 { break; }
                } else if it - best_it >= 150 { break; }
                if best_u <= p.stop_u { break; }
            }
        }
    }
    st.sp_it += it;
    st.sp_u = best_u;
    best
}

#[inline(always)]
fn fmax(a: f32, b: f32) -> f32 { if a > b { a } else { b } }

#[inline(always)]
fn range(r: u32, n: usize) -> usize { ((r as u64 * n as u64) >> 32) as usize }

const SENT: u32 = 0x40 << 24;

pub fn sls(inst: &Inst, p: &P, x: &mut [u8], rng: &mut SmallRng, st: &mut Stats) -> bool {
    let (nvx, nc) = (inst.nvx, inst.nc);
    let mask = nvx.next_power_of_two() - 1;
    let poff = &inst.off[..];
    let mut cs = vec![0u32; nc + 1];
    cs[nc] = SENT;
    let mut brk = vec![0u32; mask + 1];
    let mut ul = vec![0u32; nc + 1];
    let mut pos = vec![0u32; nc + 1];
    let mut len = 0usize;
    let lits = &inst.lits[..];
    let mut wt = [0u32; 64];
    for b in 0..63 { wt[b] = ((1u64 << 22) as f64 * (1.0 + b as f64).powf(-p.cb)).max(1.0) as u32; }
    let mut flips = 0u64;
    unsafe {
        for c in 0..nc {
            let mut s = 0u32;
            for q in 0..3 {
                let l = g(lits, 3 * c + q);
                let t = g(x, (l >> 1) as usize) as u32 ^ (l & 1) ^ 1;
                s = (s + (t << 24)) ^ (t.wrapping_neg() & (l >> 1));
            }
            *gm(&mut cs, c) = s;
            *gm(&mut brk, (s & 0xffffff) as usize) += (s >> 24 == 1) as u32;
            *gm(&mut ul, len) = c as u32;
            *gm(&mut pos, c) = len as u32;
            len += (s == 0) as usize;
        }
        for dv in inst.nv..nvx { *gm(&mut brk, dv) = 63; }
        let csp = cs.as_mut_ptr();
        let bp = brk.as_mut_ptr();
        let ulp = ul.as_mut_ptr();
        let psp = pos.as_mut_ptr();
        let op = inst.occ.as_ptr();
        let ncu = nc as u32;
        let mut rl = rng.clone();
        let mut minlen = usize::MAX;
        let mut gi = 0;
        loop {
            let seg = p.max_flips.min(p.giveup.get(gi).map_or(u64::MAX, |g| g.0));
            while (len > 0) & (flips < seg) {
                *csp.add(nc) = SENT;
                let r = rl.gen::<u64>();
                let c = *ulp.add(range(r as u32, len)) as usize;
                let l0 = g(lits, 3 * c);
                let l1 = g(lits, 3 * c + 1);
                let l2 = g(lits, 3 * c + 2);
                let w0 = g(&wt, (*bp.add((l0 >> 1) as usize) as usize).min(63));
                let w1 = g(&wt, (*bp.add((l1 >> 1) as usize) as usize).min(63));
                let w2 = g(&wt, (*bp.add((l2 >> 1) as usize) as usize).min(63));
                let t = (((r >> 32) as u32 as u64 * (w0 + w1 + w2) as u64) >> 32) as u32;
                let s1 = (t >= w0) as u32;
                let s2 = (t >= w0 + w1) as u32;
                let l = l0 ^ ((l0 ^ l1) & s1.wrapping_neg()) ^ ((l1 ^ l2) & s2.wrapping_neg());
                let v = (l >> 1) as usize;
                let xv = g(x, v) as u32;
                *gm(x, v) = (xv ^ 1) as u8;
                let tl = (v << 1) | (xv ^ 1) as usize;
                let fl = tl ^ 1;
                let vv = v as u32;
                let mut bn = 0u32;
                let mut k = g(poff, tl) as usize;
                let ke = g(poff, tl + 1) as usize;
                while k < ke {
                    for j in 0..4 {
                        let cc = *op.add(k + j);
                        let s = *csp.add(cc as usize);
                        let n = s >> 24;
                        bn += (n == 0) as u32;
                        *bp.add((s & 0xffffff) as usize) -= (n == 1) as u32;
                        *csp.add(cc as usize) = (s + (1 << 24)) ^ vv;
                        let m = (n.wrapping_sub(1) as i32 >> 31) as u32;
                        len -= (m & 1) as usize;
                        let last = *ulp.add(len);
                        let pi = *psp.add((ncu ^ ((cc ^ ncu) & m)) as usize);
                        *ulp.add(len ^ ((pi as usize ^ len) & m as usize)) = last;
                        *psp.add((ncu ^ ((last ^ ncu) & m)) as usize) = pi;
                    }
                    k += 4;
                }
                let mut k = g(poff, fl) as usize;
                let ke = g(poff, fl + 1) as usize;
                while k < ke {
                    for j in 0..4 {
                        let cc = *op.add(k + j);
                        let s = (*csp.add(cc as usize) - (1 << 24)) ^ vv;
                        *csp.add(cc as usize) = s;
                        *bp.add((s & 0xffffff) as usize) += (s >> 24 == 1) as u32;
                        let add = (s == 0) as u32;
                        *ulp.add(len) = cc;
                        *psp.add((ncu ^ ((cc ^ ncu) & add.wrapping_neg())) as usize) = len as u32;
                        len += add as usize;
                    }
                    k += 4;
                }
                *bp.add(v) = bn;
                flips += 1;
                minlen = minlen.min(len);
            }
            if (len == 0) | (flips >= p.max_flips) { break; }
            if p.giveup.get(gi).map_or(false, |g| g.0 == flips) {
                if minlen > p.giveup[gi].1 { break; }
                gi += 1;
            }
        }
        *rng = rl;
    }
    st.flips += flips;
    len == 0
}

pub fn run(ch: &Challenge, p: &P, seed: u64) -> (Vec<bool>, bool, Stats) {
    let mut st = Stats::default();
    let inst = build(ch);
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut x = vec![0u8; inst.nvx];
    let mut ok = false;
    for a in 0..p.attempts {
        st.attempts = a + 1;
        if p.sp {
            x = spr(&inst, p, &mut rng, &mut st);
        } else {
            for v in 0..inst.nv {
                let pc = inst.deg[2 * v + 1];
                let nc = inst.deg[2 * v];
                x[v] = (if pc == nc { rng.gen() } else { pc > nc }) as u8;
            }
        }
        for d in inst.nv..inst.nvx { x[d] = 0; }
        if sls(&inst, p, &mut x, &mut rng, &mut st) { ok = true; break; }
    }
    (x[..inst.nv].iter().map(|&b| b == 1).collect(), ok, st)
}

pub fn help() {
    println!("Send null on every track.");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let p = P::for_instance(challenge.num_variables, challenge.clauses.len());
    let seed = u64::from_le_bytes(challenge.seed[..8].try_into().unwrap());
    let (x, _, _) = run(challenge, &p, seed);
    save_solution(&Solution { variables: x })
}
