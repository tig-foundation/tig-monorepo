use anyhow::Result;
use rand::{Rng, RngCore, SeedableRng};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

// Copyright 2018 Developers of the Rand project (MIT OR Apache-2.0).
// Xoshiro256++ transition and byte filling follow rand 0.8.5 / rand_core
// 0.6.4 (Rand project, MIT OR Apache-2.0). Keep SeedableRng's DEFAULT PCG32
// seed_from_u64: SmallRng does not forward the inner SplitMix64 override.
#[derive(Clone,Debug)]
pub struct PortableRng64 { s:[u64;4] }
// Retain the established native64 hot path; standardize other targets on its
// exact stream. Tests compare PortableRng64 itself, never the alias to itself.
#[cfg(target_pointer_width="64")]
pub type Rng64=rand::rngs::SmallRng;
#[cfg(not(target_pointer_width="64"))]
pub type Rng64=PortableRng64;
impl SeedableRng for PortableRng64 {
    type Seed=[u8;32];
    fn from_seed(seed:Self::Seed)->Self {
        let mut s=std::array::from_fn(|i|u64::from_le_bytes(seed[8*i..8*i+8].try_into().unwrap()));
        if s==[0;4] {
            let mut state=0u64;
            for v in &mut s {
                state=state.wrapping_add(0x9e3779b97f4a7c15);let mut z=state;
                z=(z^(z>>30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z=(z^(z>>27)).wrapping_mul(0x94d049bb133111eb);*v=z^(z>>31);
            }
        }
        Self{s}
    }
}
impl RngCore for PortableRng64 {
    #[inline(always)]fn next_u32(&mut self)->u32{(self.next_u64()>>32)as u32}
    #[inline(always)]fn next_u64(&mut self)->u64{
        let out=self.s[0].wrapping_add(self.s[3]).rotate_left(23).wrapping_add(self.s[0]);
        let t=self.s[1]<<17;self.s[2]^=self.s[0];self.s[3]^=self.s[1];
        self.s[1]^=self.s[2];self.s[0]^=self.s[3];self.s[2]^=t;self.s[3]=self.s[3].rotate_left(45);out
    }
    fn fill_bytes(&mut self,dest:&mut[u8]){
        let mut left=dest;
        while left.len()>=8{let(a,b)=left.split_at_mut(8);a.copy_from_slice(&self.next_u64().to_le_bytes());left=b;}
        let n=left.len();if n>4{left.copy_from_slice(&self.next_u64().to_le_bytes()[..n]);}
        else if n>0{left.copy_from_slice(&self.next_u32().to_le_bytes()[..n]);}
    }
    fn try_fill_bytes(&mut self,dest:&mut[u8])->std::result::Result<(),rand::Error>{self.fill_bytes(dest);Ok(())}
}

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
    pub giveup: Vec<(u64, usize, u64, usize)>,
    pub stall: usize,
}

impl P {
    pub fn for_instance(nv: usize, m: usize) -> P {
        let base = P { sp: true, g0: 0.06, rate: 0.001, gmax: 0.5, lam: 0.1, damp: 0.0, maxit: 1000, check: 10,
                       stop_u: 550, refresh: 10, cb: 3.0, max_flips: 30_000_000, attempts: 4, giveup: Vec::new(), stall: 150 };
        let ratio = m as f64 / nv.max(1) as f64;
        if nv >= 50_000 && ratio < 4.175 {
            P { g0: 0.12, rate: 0.002, lam: 0.3, stop_u: 250, ..base }
        } else if nv >= 50_000 {
            P { g0: 0.07, rate: 0.0008, stop_u: 250, stall: 40, ..base }
        } else {
            let (giveup, max_flips) = if nv <= 6000 {
                (vec![(1 << 25, 9, 0, 0), (1 << 26, 4, 0, 0), (1 << 29, 1, 0, 0)], 4_000_000_000)
            } else if nv <= 8500 {
                (vec![(1 << 21, usize::MAX, 0, 0), (1 << 23, 27, 1 << 21, 47), (1 << 25, 8, 0, 0), (1 << 26, 6, 0, 0),
                      (1 << 28, 2, 0, 0)], 3_000_000_000)
            } else {
                (vec![(1 << 22, 75, 0, 0), (1 << 23, 51, 0, 0), (1 << 26, 14, 0, 0), (1 << 28, 6, 0, 0), (1 << 29, 2, 0, 0),
                      (1 << 30, 1, 0, 0)], 4_000_000_000)
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

fn rand_om(ne: usize, rng: &mut Rng64) -> Vec<f32> {
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

pub fn spr(inst: &Inst, p: &P, rng: &mut Rng64, st: &mut Stats) -> Vec<u8> {
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
                } else if it - best_it >= p.stall { break; }
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

pub trait Count: Copy {
    fn of(v: u32) -> Self;
    fn get(self) -> usize;
    fn weight(self, wt: &[u32;256]) -> u32;
    fn inc(&mut self, v: u32);
    fn dec(&mut self, v: u32);
}

impl Count for u32 {
    #[inline(always)]
    fn weight(self, wt: &[u32;256]) -> u32 { unsafe{*wt.get_unchecked((self as usize).min(63))} }
    #[inline(always)]
    fn of(v: u32) -> u32 { v }
    #[inline(always)]
    fn get(self) -> usize { self as usize }
    #[inline(always)]
    fn inc(&mut self, v: u32) { *self += v; }
    #[inline(always)]
    fn dec(&mut self, v: u32) { *self -= v; }
}

impl Count for u8 {
    #[inline(always)]
    fn weight(self, wt: &[u32;256]) -> u32 { unsafe{*wt.get_unchecked(self as usize)} }
    #[inline(always)]
    fn of(v: u32) -> u8 { v as u8 }
    #[inline(always)]
    fn get(self) -> usize { self as usize }
    #[inline(always)]
    fn inc(&mut self, v: u32) { *self += v as u8; }
    #[inline(always)]
    fn dec(&mut self, v: u32) { *self -= v as u8; }
}

struct SlsState<B> {
    cs:Vec<u32>,brk:Vec<B>,ul:Vec<u32>,pos:Vec<u32>,len:usize,
    flips:u64,minlen:usize,best:[usize;64],gi:usize,rng:Rng64,
}
pub fn sls<B: Count>(inst: &Inst, p: &P, x: &mut [u8], rng: &mut Rng64, st: &mut Stats) -> bool {
    sls_state::<B>(inst,p,x,rng,st,None,None)
}
fn sls_state<B:Count>(inst:&Inst,p:&P,x:&mut[u8],rng:&mut Rng64,st:&mut Stats,resume:Option<SlsState<B>>,checkpoint:Option<&mut Option<SlsState<B>>>)->bool {
    sls_state_capture::<B,false>(inst,p,x,rng,st,resume,checkpoint,0,&mut Near::default())
}
#[derive(Debug)]pub struct Near{pub u:usize,pub x:Vec<u8>}
impl Default for Near{fn default()->Self{Self{u:usize::MAX,x:Vec::new()}}}
fn sls_state_capture<B:Count,const CAPTURE:bool>(inst:&Inst,p:&P,x:&mut[u8],rng:&mut Rng64,st:&mut Stats,resume:Option<SlsState<B>>,checkpoint:Option<&mut Option<SlsState<B>>>,record_max:usize,near:&mut Near)->bool {
    let (nvx, nc) = (inst.nvx, inst.nc);
    let mask = nvx.next_power_of_two() - 1;
    let poff = &inst.off[..];
    let resumed=resume.is_some();
    let mut state=resume.unwrap_or_else(||SlsState{cs:vec![0;nc+1],brk:vec![B::of(0);mask+1],ul:vec![0;nc+1],pos:vec![0;nc+1],len:0,flips:0,minlen:usize::MAX,best:[usize::MAX;64],gi:0,rng:rng.clone()});
    let SlsState{mut cs,mut brk,mut ul,mut pos,mut len,mut flips,mut minlen,mut best,mut gi,rng:mut rl}=state;
    cs[nc] = SENT;
    let lits = &inst.lits[..];
    let mut wt = [0u32; 256];
    for b in 0..63 { wt[b] = ((1u64 << 22) as f64 * (1.0 + b as f64).powf(-p.cb)).max(1.0) as u32; }
    let start_flips=flips;
    unsafe {
      if !resumed {
        for c in 0..nc {
            let mut s = 0u32;
            for q in 0..3 {
                let l = g(lits, 3 * c + q);
                let t = g(x, (l >> 1) as usize) as u32 ^ (l & 1) ^ 1;
                s = (s + (t << 24)) ^ (t.wrapping_neg() & (l >> 1));
            }
            *gm(&mut cs, c) = s;
            gm(&mut brk, (s & 0xffffff) as usize).inc((s >> 24 == 1) as u32);
            *gm(&mut ul, len) = c as u32;
            *gm(&mut pos, c) = len as u32;
            len += (s == 0) as usize;
        }
        for dv in inst.nv..nvx { *gm(&mut brk, dv) = B::of(63); }
      }
        let csp = cs.as_mut_ptr();
        let bp = brk.as_mut_ptr();
        let ulp = ul.as_mut_ptr();
        let psp = pos.as_mut_ptr();
        let op = inst.occ.as_ptr();
        let ncu = nc as u32;
        let mut stop_len=if CAPTURE{record_max.min(near.u.saturating_sub(1))}else{0};
        loop {
            let seg = p.max_flips.min(p.giveup.get(gi).map_or(u64::MAX, |g| g.0));
            loop {
            while (len > stop_len) & (flips < seg) {
                *csp.add(nc) = SENT;
                let r = rl.gen::<u64>();
                let c = *ulp.add(range(r as u32, len)) as usize;
                let l0 = g(lits, 3 * c);
                let l1 = g(lits, 3 * c + 1);
                let l2 = g(lits, 3 * c + 2);
                let w0 = (*bp.add((l0 >> 1) as usize)).weight(&wt);
                let w1 = (*bp.add((l1 >> 1) as usize)).weight(&wt);
                let w2 = (*bp.add((l2 >> 1) as usize)).weight(&wt);
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
                        (*bp.add((s & 0xffffff) as usize)).dec((n == 1) as u32);
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
                        (*bp.add((s & 0xffffff) as usize)).inc((s >> 24 == 1) as u32);
                        let add = (s == 0) as u32;
                        *ulp.add(len) = cc;
                        *psp.add((ncu ^ ((cc ^ ncu) & add.wrapping_neg())) as usize) = len as u32;
                        len += add as usize;
                    }
                    k += 4;
                }
                *bp.add(v) = B::of(bn);
                flips += 1;
                minlen = minlen.min(len);
            }
            // Capture outside the flip kernel, then resume the same segment.
            // In particular, do not run giveup checks at a capture-only boundary.
            if CAPTURE && len>0 && len<=stop_len {
                near.u=len;near.x=x[..inst.nv].to_vec();stop_len=len-1;continue;
            }
            break;
            }
            if (len == 0) | (flips >= p.max_flips) { break; }
            if p.giveup.get(gi).map_or(false, |g| g.0 == flips) {
                let (_, lim, f1, lim1) = p.giveup[gi];
                best[flips.trailing_zeros() as usize] = minlen;
                if minlen > lim && (f1 == 0 || best[f1.trailing_zeros() as usize] > lim1) { break; }
                gi += 1;
            }
        }
        *rng = rl.clone();
    }
    st.flips += flips-start_flips;
    if let Some(slot)=checkpoint { *slot=Some(SlsState{cs,brk,ul,pos,len,flips,minlen,best,gi,rng:rl}); }
    len == 0
}

/// Exact restart prefix, followed only on failure by the saved attempt-zero
/// continuation. Clause-state and unsatisfied-list order are retained as well
/// as x/RNG, so this resumes rather than reconstructs a different basin.
pub fn run_restarts_resume(ch:&Challenge,p:&P,seed:u64,total:u64,first_cap:u64,resume_cap:u64,min_limit:usize)->(Vec<bool>,bool,Stats){
    let(x,ok,st,_)=run_resume_capture::<false>(ch,p,seed,total,first_cap,resume_cap,min_limit,0);(x,ok,st)
}
pub fn run_restarts_resume_near(ch:&Challenge,p:&P,seed:u64,total:u64,first_cap:u64,resume_cap:u64,min_limit:usize,record_max:usize)->(Vec<bool>,bool,Stats,Near){
    run_resume_capture::<true>(ch,p,seed,total,first_cap,resume_cap,min_limit,record_max)
}
fn run_resume_capture<const CAPTURE:bool>(ch:&Challenge,p:&P,seed:u64,total:u64,first_cap:u64,resume_cap:u64,min_limit:usize,record_max:usize)->(Vec<bool>,bool,Stats,Near){
    assert!(!p.sp && ch.num_variables<=10000);
    let inst=build(ch);let mut st=Stats::default();let mut rng=Rng64::seed_from_u64(seed);let mut x=vec![0u8;inst.nvx];
    let mut saved=None;let mut saved_x=Vec::new();let mut ok=false;
    let mut near=Near::default();
    for a in 0..p.attempts {
        if st.flips>=total{break;}
        st.attempts=a+1;
        for v in 0..inst.nv{let pc=inst.deg[2*v+1];let nc=inst.deg[2*v];x[v]=(if a>0 || pc==nc{rng.gen()}else{pc>nc})as u8;}
        for v in inst.nv..inst.nvx{x[v]=0;}
        let mut q=p.clone();q.max_flips=q.max_flips.min(total-st.flips);
        let solved=if a==0 {
            let solved=sls_state_capture::<u8,CAPTURE>(&inst,&q,&mut x,&mut rng,&mut st,None,Some(&mut saved),record_max,&mut near);
            if saved.as_ref().map_or(false,|s|!solved && s.flips==first_cap && s.minlen<=min_limit){saved_x=x.clone();}else{saved=None;}
            solved
        }else{sls_state_capture::<u8,CAPTURE>(&inst,&q,&mut x,&mut rng,&mut st,None,None,record_max,&mut near)};
        if solved{ok=true;break;}
    }
    if !ok {if let Some(state)=saved{
        x=saved_x;let mut q=p.clone();q.max_flips=resume_cap;
        ok=sls_state_capture::<u8,CAPTURE>(&inst,&q,&mut x,&mut rng,&mut st,Some(state),None,record_max,&mut near);
    }}
    (x[..inst.nv].iter().map(|&v|v==1).collect(),ok,st,near)
}

/// Opt-in small-track snapshot path. Large SP tracks continue to use run().
pub fn run_near(ch:&Challenge,p:&P,seed:u64,record_max:usize)->(Vec<bool>,bool,Stats,Near){
    assert!(!p.sp && ch.num_variables<=10000);
    let inst=build(ch);let mut st=Stats::default();let mut rng=Rng64::seed_from_u64(seed);let mut x=vec![0;inst.nvx];let mut near=Near::default();let mut ok=false;
    for a in 0..p.attempts{
        st.attempts=a+1;
        for v in 0..inst.nv{let pc=inst.deg[2*v+1];let nc=inst.deg[2*v];x[v]=(if pc==nc{rng.gen()}else{pc>nc})as u8;}
        for v in inst.nv..inst.nvx{x[v]=0;}
        if sls_state_capture::<u8,true>(&inst,p,&mut x,&mut rng,&mut st,None,None,record_max,&mut near){ok=true;break;}
    }
    (x[..inst.nv].iter().map(|&v|v==1).collect(),ok,st,near)
}

pub fn run(ch: &Challenge, p: &P, seed: u64) -> (Vec<bool>, bool, Stats) {
    run_inner(ch,p,seed,None)
}

/// Small-track restart diagnostic/policy. Attempt zero retains the exact old
/// initialization and RNG prefix; later attempts draw a uniform assignment.
pub fn run_restarts(ch:&Challenge,p:&P,seed:u64,total:u64)->(Vec<bool>,bool,Stats){
    assert!(!p.sp && ch.num_variables<=10_000);
    run_inner(ch,p,seed,Some(total))
}
fn run_inner(ch:&Challenge,p:&P,seed:u64,total:Option<u64>)->(Vec<bool>,bool,Stats){
    let mut st = Stats::default();
    let inst = build(ch);
    let mut rng = Rng64::seed_from_u64(seed);
    let mut x = vec![0u8; inst.nvx];
    let mut ok = false;
    for a in 0..p.attempts {
        if total.map_or(false,|cap|st.flips>=cap){break;}
        st.attempts = a + 1;
        if p.sp {
            x = spr(&inst, p, &mut rng, &mut st);
        } else {
            for v in 0..inst.nv {
                let pc = inst.deg[2 * v + 1];
                let nc = inst.deg[2 * v];
                x[v] = (if (a>0 && total.is_some()) || pc == nc { rng.gen() } else { pc > nc }) as u8;
            }
        }
        for d in inst.nv..inst.nvx { x[d] = 0; }
        let mut bounded;
        let attempt_p=if let Some(cap)=total {bounded=p.clone();bounded.max_flips=bounded.max_flips.min(cap-st.flips);&bounded}else{p};
        let solved = finish_attempt(&inst, attempt_p, &mut x, &mut rng, &mut st);
        if solved { ok = true; break; }
    }
    (x[..inst.nv].iter().map(|&b| b == 1).collect(), ok, st)
}

pub fn help() {
    println!("Send null on every track.");
}

fn finish_attempt(inst:&Inst,p:&P,x:&mut[u8],rng:&mut Rng64,st:&mut Stats)->bool {
    if p.sp {st.sp_u==0 || sls::<u32>(inst,p,x,rng,st)}else{sls::<u8>(inst,p,x,rng,st)}
}

#[cfg(test)]mod cycle8_tests {
    use super::*;
    fn contradiction()->Challenge {
        let mut clauses:Vec<Vec<i32>>=(0..8).map(|m|(0..3).map(|v|if m&(1<<v)==0{v+1}else{-(v+1)}).collect()).collect();
        for i in 0..48{clauses.push(vec![4+i%16,4+(i+1)%16,4+(i+2)%16]);}
        Challenge{seed:[0;32],num_variables:19,clauses}
    }
    #[test]fn resumed_state_matches_uninterrupted_trajectory(){
        let ch=contradiction();let inst=build(&ch);let mut p=P::for_instance(7500,32002);p.max_flips=10000;p.giveup=vec![(4096,usize::MAX,0,0)];
        let mut x=vec![0;inst.nvx];let mut y=x.clone();let mut r=Rng64::seed_from_u64(8);let mut s=r.clone();let mut a=Stats::default();let mut b=Stats::default();
        assert!(!sls::<u8>(&inst,&p,&mut x,&mut r,&mut a));
        let mut q=p.clone();q.max_flips=3000;let mut saved=None;
        assert!(!sls_state::<u8>(&inst,&q,&mut y,&mut s,&mut b,None,Some(&mut saved)));
        assert!(!sls_state(&inst,&p,&mut y,&mut s,&mut b,saved,None));
        assert_eq!(x,y);assert_eq!(a.flips,10000);assert_eq!(a.flips,b.flips);
        for _ in 0..100{assert_eq!(r.next_u64(),s.next_u64());}
    }
    #[test]fn resume_wrapper_preserves_failed_restart_prefix(){
        let ch=contradiction();let mut p=P::for_instance(7500,32002);p.max_flips=200;p.attempts=4;p.giveup.clear();
        for seed in 0..4{
            let(a,ok,st)=run_restarts(&ch,&p,seed,500);
            let(b,ok2,st2)=run_restarts_resume(&ch,&p,seed,500,200,1000,0);
            assert!(!ok && !ok2);assert_eq!(a,b);assert_eq!(st.flips,st2.flips);assert_eq!(st.attempts,st2.attempts);
        }
    }
    #[test]fn all_u8_weights_match_old_clamp(){
        for cb in [2.2,3.0] {
            let mut old=[0u32;64];for b in 0..63{old[b]=((1u64<<22)as f64*(1.0+b as f64).powf(-cb)).max(1.0)as u32;}
            let mut wide=[0u32;256];wide[..64].copy_from_slice(&old);
            for b in 0..=255u32{assert_eq!((b as u8).weight(&wide),old[(b as usize).min(63)]);assert_eq!(b.weight(&wide),old[(b as usize).min(63)]);}
            assert_eq!(u32::MAX.weight(&wide),0);
        }
    }
    #[test]fn sp_solved_bypass_preserves_assignment_flips_and_rng(){
        let ch=Challenge{seed:[0;32],num_variables:64,clauses:(0..64).map(|i|vec![1+i,1+(i+1)%64,1+(i+2)%64]).collect()};
        let inst=build(&ch);let mut p=P::for_instance(50000,200000);p.max_flips=32;
        let mut x=vec![1;inst.nvx];let mut y=x.clone();assert_eq!(unsat_count(&inst,&x),0);
        let mut r=Rng64::seed_from_u64(19);let mut s=r.clone();let mut a=Stats::default();let mut b=Stats::default();a.flips=7;b.flips=7;
        assert!(sls::<u32>(&inst,&p,&mut x,&mut r,&mut a));assert!(finish_attempt(&inst,&p,&mut y,&mut s,&mut b));
        assert_eq!(x,y);assert_eq!(a.flips,b.flips);assert_eq!(a.flips,7);
        for _ in 0..100{assert_eq!(r.next_u64(),s.next_u64());}
    }
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

pub fn build_fast(ch: &Challenge) -> Inst {
    let nv = ch.num_variables;
    let m = ch.clauses.len();
    let mut lits:Vec<u32> = Vec::with_capacity(3 * m + 3);
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
    unsafe { lits.set_len(3 * nc); }
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
    let total=off[2*nvx]as usize; let mut occ:Vec<u32> = Vec::with_capacity(total);
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
        for l in 0..2*nvx {for k in pos[l]as usize..off[l+1]as usize{oc.add(k).write((nc+(k&3))as u32);}}
        occ.set_len(total);
    }
    Inst { nv, nvx, nc, lits, deg, off, occ }
}
// C21 opt-in staged SIMD operations. SSE2 is baseline on x86_64. Every
// arithmetic operation has the same scalar order; no reciprocal approximation.
macro_rules! c21_vop {($name:ident,$intr:ident,$expr:expr)=>{
 #[inline(always)]fn $name(a:[f32;4],b:[f32;4])->[f32;4]{
  #[cfg(target_arch="x86_64")]unsafe{use core::arch::x86_64::*;let mut out=[0.;4];_mm_storeu_ps(out.as_mut_ptr(),$intr(_mm_loadu_ps(a.as_ptr()),_mm_loadu_ps(b.as_ptr())));out}
  #[cfg(not(target_arch="x86_64"))]{std::array::from_fn(|k|($expr)(a[k],b[k]))}
 }
}}
c21_vop!(c21_add,_mm_add_ps,|a,b|a+b);
c21_vop!(c21_sub,_mm_sub_ps,|a,b|a-b);
c21_vop!(c21_mul,_mm_mul_ps,|a,b|a*b);
c21_vop!(c21_div,_mm_div_ps,|a,b|a/b);
c21_vop!(c21_max,_mm_max_ps,|a,b|fmax(a,b));

#[inline(always)]unsafe fn c21_var4_simd(prp:*mut f32,hpp:*mut f32,hmp:*mut f32,xp:*mut u8,v:usize,lam1:f32,lg:f32){
 let pm=std::array::from_fn(|k|*prp.add(2*(v+k)));let pp=std::array::from_fn(|k|*prp.add(2*(v+k)+1));
 let o1=std::array::from_fn(|k|*hpp.add(v+k));let o0=std::array::from_fn(|k|*hmp.add(v+k));let one=[1.;4];
 let pip=c21_mul(c21_sub(one,pp),pm);let pim=c21_mul(c21_sub(one,pm),pp);
 let inv=c21_div(one,c21_max(c21_add(c21_add(pip,pim),c21_mul(pp,pm)),[1e-30;4]));
 let wp=c21_mul(pip,inv);let wm=c21_mul(pim,inv);
 let h1=c21_add(c21_mul([lam1;4],o1),c21_mul([lg;4],wp));let h0=c21_add(c21_mul([lam1;4],o0),c21_mul([lg;4],wm));
 let nm=c21_mul(pm,c21_div(c21_sub(one,h0),c21_sub(one,o0)));let np=c21_mul(pp,c21_div(c21_sub(one,h1),c21_sub(one,o1)));
 std::ptr::copy_nonoverlapping(h1.as_ptr(),hpp.add(v),4);std::ptr::copy_nonoverlapping(h0.as_ptr(),hmp.add(v),4);
 for k in 0..4{*xp.add(v+k)=(wp[k]>wm[k])as u8;*prp.add(2*(v+k))=nm[k];*prp.add(2*(v+k)+1)=np[k];}
}

#[inline(always)]unsafe fn c21_blk_simd<const DAMP_ZERO:bool>(lsp:*const u32,prp:*mut f32,omp:*mut f32,nc4:usize,c:usize,d:f32,d1:f32){
 let mut l=[[0usize;4];3];let mut pp=[[0.;4];3];let mut bv=[[0.;4];3];let mut o=[[0.;4];3];
 for q in 0..3{for k in 0..4{let li=*lsp.add(q*nc4+c+k)as usize;l[q][k]=li;pp[q][k]=*prp.add(li);bv[q][k]=*prp.add(li^1);o[q][k]=*omp.add(q*nc4+c+k);}}
 let mut r=[[0.;4];3];for q in 0..3{let n=c21_mul(c21_sub([1.;4],bv[q]),pp[q]);r[q]=c21_div(n,c21_max(c21_add(n,c21_mul(bv[q],o[q])),[1e-30;4]));}
 let pe=[c21_mul(r[1],r[2]),c21_mul(r[0],r[2]),c21_mul(r[0],r[1])];let mut on=[[0.;4];3];let mut ratio=[[0.;4];3];
 for q in 0..3{let n=c21_sub([1.;4],pe[q]);on[q]=c21_max(if DAMP_ZERO{n}else{c21_add(c21_mul([d;4],o[q]),c21_mul([d1;4],n))},[1e-6;4]);ratio[q]=c21_div(on[q],o[q]);}
 for q in 0..3{std::ptr::copy_nonoverlapping(on[q].as_ptr(),omp.add(q*nc4+c),4);}
 // Repeated variables across clauses must retain this exact product order.
 for q in 0..3{for k in 0..4{*prp.add(l[q][k])*=ratio[q][k];}}
}

fn c21_initial_state<B:Count>(inst:&Inst,x:&[u8],rng:&Rng64,mask:usize)->SlsState<B>{
 let nc=inst.nc;let mut cs:Vec<u32>=Vec::with_capacity(nc+4);let mut pos:Vec<u32>=Vec::with_capacity(nc+4);
 // ul[len] is read even when a removal mask is zero; retain defined zero tail.
 let mut ul=vec![0u32;nc+4];let mut brk=vec![B::of(0);mask+1];let mut len=0;
 unsafe{for c in 0..nc{let mut s=0u32;for q in 0..3{let l=*inst.lits.get_unchecked(3*c+q);let t=(*x.get_unchecked((l>>1)as usize)as u32)^(l&1)^1;s=(s+(t<<24))^(t.wrapping_neg()&(l>>1));}cs.as_mut_ptr().add(c).write(s);brk[(s&0xffffff)as usize].inc((s>>24==1)as u32);ul[len]=c as u32;pos.as_mut_ptr().add(c).write(len as u32);len+=(s==0)as usize;}
  for k in 0..4{cs.as_mut_ptr().add(nc+k).write(SENT);pos.as_mut_ptr().add(nc+k).write(0);}cs.set_len(nc+4);pos.set_len(nc+4);
 }
 for dv in inst.nv..inst.nvx{brk[dv]=B::of(63);}
 SlsState{cs,brk,ul,pos,len,flips:0,minlen:usize::MAX,best:[usize::MAX;64],gi:0,rng:rng.clone()}
}

unsafe fn c21_blk_scalar(lsp: *const u32, prp: *mut f32, omp: *mut f32, nc4: usize, c: usize, d: f32, d1: f32) {
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
                for q in 0..3 {
                    for k in 0..4 { *prp.add(l[q][k]) *= ratio[q][k]; }
                }
            }
unsafe fn c21_var_scalar(prp: *mut f32, hpp: *mut f32, hmp: *mut f32, xp: *mut u8, v: usize, lam1: f32, lg: f32) {
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

pub fn spr_simd(inst: &Inst, p: &P, rng: &mut Rng64, st: &mut Stats, uninit_ls:bool) -> Vec<u8> {
    let (nv, nvx, nc) = (inst.nv, inst.nvx, inst.nc);
    let nc4 = (nc + 3) & !3;
    let mut ls = if uninit_ls { Vec::with_capacity(3*nc4) } else {vec![(nvx as u32)<<1;3*nc4]};
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
    if uninit_ls {unsafe{let dst=ls.as_mut_ptr(); for q in 0..3{for c in nc..nc4{dst.add(q*nc4+c).write((nvx as u32)<<1);}}ls.set_len(3*nc4);}}
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
            unsafe fn blk(lsp:*const u32,prp:*mut f32,omp:*mut f32,nc4:usize,c:usize,d:f32,d1:f32){if d==0.0{c21_blk_simd::<true>(lsp,prp,omp,nc4,c,d,d1)}else{c21_blk_simd::<false>(lsp,prp,omp,nc4,c,d,d1)}}
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
                c21_var4_simd(prp,hpp,hmp,xp,v,lam1,lg);
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
                } else if it - best_it >= p.stall { break; }
                if best_u <= p.stop_u { break; }
            }
        }
    }
    st.sp_it += it;
    st.sp_u = best_u;
    best
}

fn sls_fast_capture<B:Count,const CAPTURE:bool>(inst:&Inst,p:&P,x:&mut[u8],rng:&mut Rng64,st:&mut Stats,resume:Option<SlsState<B>>,checkpoint:Option<&mut Option<SlsState<B>>>,record_max:usize,near:&mut Near)->bool {
    let (nvx, nc) = (inst.nvx, inst.nc);
    let mask = nvx.next_power_of_two() - 1;
    let poff = &inst.off[..];
    let resumed=resume.is_some();
    let mut state=resume.unwrap_or_else(||c21_initial_state::<B>(inst,x,rng,mask));
    let SlsState{mut cs,mut brk,mut ul,mut pos,mut len,mut flips,mut minlen,mut best,mut gi,rng:mut rl}=state;
    cs[nc..nc+4].fill(SENT);
    let lits = &inst.lits[..];
    let mut wt = [0u32; 256];
    for b in 0..63 { wt[b] = ((1u64 << 22) as f64 * (1.0 + b as f64).powf(-p.cb)).max(1.0) as u32; }
    let start_flips=flips;
    unsafe {
        let csp = cs.as_mut_ptr();
        let bp = brk.as_mut_ptr();
        let ulp = ul.as_mut_ptr();
        let psp = pos.as_mut_ptr();
        let op = inst.occ.as_ptr();
        let ncu = nc as u32;
        let mut stop_len=if CAPTURE{record_max.min(near.u.saturating_sub(1))}else{0};
        loop {
            let seg = p.max_flips.min(p.giveup.get(gi).map_or(u64::MAX, |g| g.0));
            loop {
            while (len > stop_len) & (flips < seg) {
                (csp.add(nc)as *mut [u32;4]).write([SENT;4]);
                let r = rl.gen::<u64>();
                let c = *ulp.add(range(r as u32, len)) as usize;
                let l0 = g(lits, 3 * c);
                let l1 = g(lits, 3 * c + 1);
                let l2 = g(lits, 3 * c + 2);
                let w0 = (*bp.add((l0 >> 1) as usize)).weight(&wt);
                let w1 = (*bp.add((l1 >> 1) as usize)).weight(&wt);
                let w2 = (*bp.add((l2 >> 1) as usize)).weight(&wt);
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
                        (*bp.add((s & 0xffffff) as usize)).dec((n == 1) as u32);
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
                        (*bp.add((s & 0xffffff) as usize)).inc((s >> 24 == 1) as u32);
                        let add = (s == 0) as u32;
                        *ulp.add(len) = cc;
                        *psp.add((ncu ^ ((cc ^ ncu) & add.wrapping_neg())) as usize) = len as u32;
                        len += add as usize;
                    }
                    k += 4;
                }
                *bp.add(v) = B::of(bn);
                flips += 1;
                minlen = minlen.min(len);
            }
            // Capture outside the flip kernel, then resume the same segment.
            // In particular, do not run giveup checks at a capture-only boundary.
            if CAPTURE && len>0 && len<=stop_len {
                near.u=len;near.x=x[..inst.nv].to_vec();stop_len=len-1;continue;
            }
            break;
            }
            if (len == 0) | (flips >= p.max_flips) { break; }
            if p.giveup.get(gi).map_or(false, |g| g.0 == flips) {
                let (_, lim, f1, lim1) = p.giveup[gi];
                best[flips.trailing_zeros() as usize] = minlen;
                if minlen > lim && (f1 == 0 || best[f1.trailing_zeros() as usize] > lim1) { break; }
                gi += 1;
            }
        }
        *rng = rl.clone();
    }
    st.flips += flips-start_flips;
    if let Some(slot)=checkpoint { *slot=Some(SlsState{cs,brk,ul,pos,len,flips,minlen,best,gi,rng:rl}); }
    len == 0
}

pub fn run_near_fast(ch:&Challenge,p:&P,seed:u64,record_max:usize)->(Vec<bool>,bool,Stats,Near){
    assert!(!p.sp && ch.num_variables<=10000);
    let inst=build_fast(ch);let mut st=Stats::default();let mut rng=Rng64::seed_from_u64(seed);let mut x=vec![0;inst.nvx];let mut near=Near::default();let mut ok=false;
    for a in 0..p.attempts{
        st.attempts=a+1;
        for v in 0..inst.nv{let pc=inst.deg[2*v+1];let nc=inst.deg[2*v];x[v]=(if pc==nc{rng.gen()}else{pc>nc})as u8;}
        for v in inst.nv..inst.nvx{x[v]=0;}
        if sls_fast_capture::<u8,true>(&inst,p,&mut x,&mut rng,&mut st,None,None,record_max,&mut near){ok=true;break;}
    }
    (x[..inst.nv].iter().map(|&v|v==1).collect(),ok,st,near)
}

fn run_resume_fast<const CAPTURE:bool>(ch:&Challenge,p:&P,seed:u64,total:u64,first_cap:u64,resume_cap:u64,min_limit:usize,record_max:usize)->(Vec<bool>,bool,Stats,Near){
    assert!(!p.sp && ch.num_variables<=10000);
    let inst=build_fast(ch);let mut st=Stats::default();let mut rng=Rng64::seed_from_u64(seed);let mut x=vec![0u8;inst.nvx];
    let mut saved=None;let mut saved_x=Vec::new();let mut ok=false;
    let mut near=Near::default();
    for a in 0..p.attempts {
        if st.flips>=total{break;}
        st.attempts=a+1;
        for v in 0..inst.nv{let pc=inst.deg[2*v+1];let nc=inst.deg[2*v];x[v]=(if a>0 || pc==nc{rng.gen()}else{pc>nc})as u8;}
        for v in inst.nv..inst.nvx{x[v]=0;}
        let mut q=p.clone();q.max_flips=q.max_flips.min(total-st.flips);
        let solved=if a==0 {
            let solved=sls_fast_capture::<u8,CAPTURE>(&inst,&q,&mut x,&mut rng,&mut st,None,Some(&mut saved),record_max,&mut near);
            if saved.as_ref().map_or(false,|s|!solved && s.flips==first_cap && s.minlen<=min_limit){saved_x=x.clone();}else{saved=None;}
            solved
        }else{sls_fast_capture::<u8,CAPTURE>(&inst,&q,&mut x,&mut rng,&mut st,None,None,record_max,&mut near)};
        if solved{ok=true;break;}
    }
    if !ok {if let Some(state)=saved{
        x=saved_x;let mut q=p.clone();q.max_flips=resume_cap;
        ok=sls_fast_capture::<u8,CAPTURE>(&inst,&q,&mut x,&mut rng,&mut st,Some(state),None,record_max,&mut near);
    }}
    (x[..inst.nv].iter().map(|&v|v==1).collect(),ok,st,near)
}

fn run_inner_fast(ch:&Challenge,p:&P,seed:u64,total:Option<u64>,fast_build:bool)->(Vec<bool>,bool,Stats){
    let mut st = Stats::default();
    let inst = if fast_build{build_fast(ch)}else{build(ch)};
    let mut rng = Rng64::seed_from_u64(seed);
    let mut x = vec![0u8; inst.nvx];
    let mut ok = false;
    for a in 0..p.attempts {
        if total.map_or(false,|cap|st.flips>=cap){break;}
        st.attempts = a + 1;
        if p.sp {
            x = spr_simd(&inst, p, &mut rng, &mut st,fast_build);
        } else {
            for v in 0..inst.nv {
                let pc = inst.deg[2 * v + 1];
                let nc = inst.deg[2 * v];
                x[v] = (if (a>0 && total.is_some()) || pc == nc { rng.gen() } else { pc > nc }) as u8;
            }
        }
        for d in inst.nv..inst.nvx { x[d] = 0; }
        let mut bounded;
        let attempt_p=if let Some(cap)=total {bounded=p.clone();bounded.max_flips=bounded.max_flips.min(cap-st.flips);&bounded}else{p};
        let solved = if p.sp && st.sp_u==0 {true}else if fast_build{sls_fast_capture::<u32,false>(&inst,attempt_p,&mut x,&mut rng,&mut st,None,None,0,&mut Near::default())}else{sls::<u32>(&inst,attempt_p,&mut x,&mut rng,&mut st)};
        if solved { ok = true; break; }
    }
    (x[..inst.nv].iter().map(|&b| b == 1).collect(), ok, st)
}

pub fn run_spr_simd(ch:&Challenge,p:&P,seed:u64)->(Vec<bool>,bool,Stats){run_inner_fast(ch,p,seed,None,false)}
pub fn run_fast(ch:&Challenge,p:&P,seed:u64)->(Vec<bool>,bool,Stats){run_inner_fast(ch,p,seed,None,true)}
pub fn run_restarts_resume_near_fast(ch:&Challenge,p:&P,seed:u64,total:u64,first_cap:u64,resume_cap:u64,min_limit:usize,record_max:usize)->(Vec<bool>,bool,Stats,Near){run_resume_fast::<true>(ch,p,seed,total,first_cap,resume_cap,min_limit,record_max)}

#[cfg(test)]mod c21_tests {
 use super::*;
 fn bits(a:&[f32],b:&[f32]){assert_eq!(a.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),b.iter().map(|x|x.to_bits()).collect::<Vec<_>>());}
 fn random_ch(seed:u64,n:usize)->Challenge{let mut r=Rng64::seed_from_u64(seed);let mut clauses:Vec<Vec<i32>>=(0..(n*4267/1000)).map(|_|(0..3).map(|_|{let v=r.gen_range(1..=n)as i32;if r.gen::<bool>(){v}else{-v}}).collect()).collect();clauses.extend([vec![1,1,2],vec![1,-1,3],vec![2,2,2]]);Challenge{seed:[0;32],num_variables:n,clauses}}
 #[test]fn simd_max_matches_scalar_including_nan_and_signed_zero(){let a=[0.,-0.,f32::NAN,-1.,1.,f32::NEG_INFINITY,f32::INFINITY];for &x in &a{for &y in &a{bits(&c21_max([x;4],[y;4]),&[fmax(x,y);4]);}}}
 #[test]fn staged_variable_arithmetic_has_zero_ulp_error(){let mut rng=Rng64::seed_from_u64(91);for _ in 0..1000{let mut pr:Vec<f32>=(0..24).map(|_|rng.gen_range(0.000001..1.0)).collect();let mut hp:Vec<f32>=(0..12).map(|_|rng.gen_range(0.0..0.9)).collect();let mut hm=hp.clone();let mut x=vec![0;12];let(mut a,mut b,mut c,mut d)=(pr.clone(),hp.clone(),hm.clone(),x.clone());unsafe{for k in 0..4{c21_var_scalar(pr.as_mut_ptr(),hp.as_mut_ptr(),hm.as_mut_ptr(),x.as_mut_ptr(),2+k,0.7,0.11);}c21_var4_simd(a.as_mut_ptr(),b.as_mut_ptr(),c.as_mut_ptr(),d.as_mut_ptr(),2,0.7,0.11);}bits(&pr,&a);bits(&hp,&b);bits(&hm,&c);assert_eq!(x,d);}}
 #[test]fn staged_clause_arithmetic_and_product_order_have_zero_ulp_error(){let mut rng=Rng64::seed_from_u64(92);for damp in [0.,0.2]{for _ in 0..1000{let l:Vec<u32>=(0..24).map(|_|rng.gen_range(0..16)).collect();let mut pr:Vec<f32>=(0..16).map(|_|rng.gen_range(0.000001..1.0)).collect();let mut om:Vec<f32>=(0..24).map(|_|rng.gen_range(0.000001..1.0)).collect();let(mut a,mut b)=(pr.clone(),om.clone());unsafe{c21_blk_scalar(l.as_ptr(),pr.as_mut_ptr(),om.as_mut_ptr(),8,0,damp,1.-damp);if damp==0.{c21_blk_simd::<true>(l.as_ptr(),a.as_mut_ptr(),b.as_mut_ptr(),8,0,damp,1.-damp);}else{c21_blk_simd::<false>(l.as_ptr(),a.as_mut_ptr(),b.as_mut_ptr(),8,0,damp,1.-damp);}}bits(&pr,&a);bits(&om,&b);}}}
 #[test]fn fast_build_normalization_occurrences_and_four_padding_lanes(){for seed in 0..12{let ch=random_ch(seed,127);let a=build(&ch);let b=build_fast(&ch);assert_eq!((a.nv,a.nvx,a.nc),(b.nv,b.nvx,b.nc));assert_eq!(a.lits,b.lits);assert_eq!(a.deg,b.deg);assert_eq!(a.off,b.off);for l in 0..2*a.nvx{let start=a.off[l]as usize;let end=start+a.deg[l]as usize;assert_eq!(&a.occ[start..end],&b.occ[start..end]);for k in end..a.off[l+1]as usize{assert_eq!(b.occ[k],(a.nc+(k&3))as u32);}}}}
 #[test]fn full_sp_paths_preserve_rng_iterations_residual_and_assignment(){for seed in 0..8{let ch=random_ch(seed,257);let a=build(&ch);let b=build_fast(&ch);let mut p=P::for_instance(100000,420000);p.maxit=80;p.check=1;p.stop_u=0;p.stall=40;for damp in [0.,0.2]{p.damp=damp;let mut rng=Rng64::seed_from_u64(seed);let mut st=Stats::default();let x=spr(&a,&p,&mut rng,&mut st);for uninit in [false,true]{let mut r=Rng64::seed_from_u64(seed);let mut s=Stats::default();let y=spr_simd(if uninit{&b}else{&a},&p,&mut r,&mut s,uninit);assert_eq!(x,y);assert_eq!((st.sp_it,st.sp_u),(s.sp_it,s.sp_u));let mut rr=rng.clone();for _ in 0..10{assert_eq!(rr.next_u64(),r.next_u64());}}}}}
 fn states(a:&SlsState<u8>,b:&SlsState<u8>,nc:usize){assert_eq!(&a.cs[..nc],&b.cs[..nc]);assert_eq!(&a.pos[..nc],&b.pos[..nc]);assert_eq!(&a.ul[..a.len],&b.ul[..b.len]);assert_eq!(a.brk,b.brk);assert_eq!((a.len,a.flips,a.minlen,a.gi,a.best),(b.len,b.flips,b.minlen,b.gi,b.best));let(mut r,mut s)=(a.rng.clone(),b.rng.clone());for _ in 0..20{assert_eq!(r.next_u64(),s.next_u64());}}
 #[test]fn sentinel_kernel_full_checkpoint_and_resumed_state_are_exact(){for seed in 0..10{let ch=random_ch(seed,127);let a=build(&ch);let b=build_fast(&ch);let mut p=P::for_instance(7500,32002);p.max_flips=20000;p.giveup=vec![(4096,usize::MAX,0,0),(16384,usize::MAX,0,0)];let mut x=vec![0;a.nvx];let mut y=x.clone();let(mut r,mut s)=(Rng64::seed_from_u64(seed),Rng64::seed_from_u64(seed));let(mut st,mut ts)=(Stats::default(),Stats::default());let(mut na,mut nb)=(Near::default(),Near::default());let(mut sa,mut sb)=(None,None);let ok=sls_state_capture::<u8,true>(&a,&p,&mut x,&mut r,&mut st,None,Some(&mut sa),3,&mut na);let ok2=sls_fast_capture::<u8,true>(&b,&p,&mut y,&mut s,&mut ts,None,Some(&mut sb),3,&mut nb);assert_eq!(ok,ok2);assert_eq!(x,y);assert_eq!(na.u,nb.u);assert_eq!(na.x,nb.x);states(sa.as_ref().unwrap(),sb.as_ref().unwrap(),a.nc);if !ok{p.max_flips=50000;let(mut ca,mut cb)=(None,None);let ok=sls_state_capture::<u8,true>(&a,&p,&mut x,&mut r,&mut st,sa,Some(&mut ca),3,&mut na);let ok2=sls_fast_capture::<u8,true>(&b,&p,&mut y,&mut s,&mut ts,sb,Some(&mut cb),3,&mut nb);assert_eq!(ok,ok2);assert_eq!(x,y);assert_eq!((st.flips,na.u),(ts.flips,nb.u));assert_eq!(na.x,nb.x);states(ca.as_ref().unwrap(),cb.as_ref().unwrap(),a.nc);}}}
 #[test]fn near_and_restart_resume_wrappers_preserve_full_search(){for seed in 0..6{let ch=random_ch(seed,191);let mut p=P::for_instance(7500,32002);p.attempts=4;p.max_flips=8000;p.giveup=vec![(4096,usize::MAX,0,0)];let(a,o,s,n)=run_near(&ch,&p,seed,3);let(b,q,t,m)=run_near_fast(&ch,&p,seed,3);assert_eq!((a,o,s.flips,n.u,n.x),(b,q,t.flips,m.u,m.x));let(a,o,s,n)=run_restarts_resume_near(&ch,&p,seed,24000,8000,40000,usize::MAX,3);let(b,q,t,m)=run_restarts_resume_near_fast(&ch,&p,seed,24000,8000,40000,usize::MAX,3);assert_eq!((a,o,s.flips,s.attempts,n.u,n.x),(b,q,t.flips,t.attempts,m.u,m.x));}}
}

#[inline(always)]unsafe fn c25_var4_simd(prp:*mut f32,hpp:*mut f32,hmp:*mut f32,xp:*mut u8,v:usize,lam1:f32,lg:f32){
 let (pm,pp):([f32;4],[f32;4]);
 #[cfg(target_arch="x86_64")]{use core::arch::x86_64::*;let lo=_mm_loadu_ps(prp.add(2*v));let hi=_mm_loadu_ps(prp.add(2*v+4));let mut m=[0.;4];let mut p=[0.;4];_mm_storeu_ps(m.as_mut_ptr(),_mm_shuffle_ps::<0x88>(lo,hi));_mm_storeu_ps(p.as_mut_ptr(),_mm_shuffle_ps::<0xdd>(lo,hi));pm=m;pp=p;}
 #[cfg(not(target_arch="x86_64"))]{pm=std::array::from_fn(|k|*prp.add(2*(v+k)));pp=std::array::from_fn(|k|*prp.add(2*(v+k)+1));}
 let o1=std::array::from_fn(|k|*hpp.add(v+k));let o0=std::array::from_fn(|k|*hmp.add(v+k));let one=[1.;4];
 let pip=c21_mul(c21_sub(one,pp),pm);let pim=c21_mul(c21_sub(one,pm),pp);
 let inv=c21_div(one,c21_max(c21_add(c21_add(pip,pim),c21_mul(pp,pm)),[1e-30;4]));
 let wp=c21_mul(pip,inv);let wm=c21_mul(pim,inv);
 let h1=c21_add(c21_mul([lam1;4],o1),c21_mul([lg;4],wp));let h0=c21_add(c21_mul([lam1;4],o0),c21_mul([lg;4],wm));
 let nm=c21_mul(pm,c21_div(c21_sub(one,h0),c21_sub(one,o0)));let np=c21_mul(pp,c21_div(c21_sub(one,h1),c21_sub(one,o1)));
 std::ptr::copy_nonoverlapping(h1.as_ptr(),hpp.add(v),4);std::ptr::copy_nonoverlapping(h0.as_ptr(),hmp.add(v),4);
 for k in 0..4{*xp.add(v+k)=(wp[k]>wm[k])as u8;}
 #[cfg(target_arch="x86_64")]{use core::arch::x86_64::*;let m=_mm_loadu_ps(nm.as_ptr());let p=_mm_loadu_ps(np.as_ptr());_mm_storeu_ps(prp.add(2*v),_mm_unpacklo_ps(m,p));_mm_storeu_ps(prp.add(2*v+4),_mm_unpackhi_ps(m,p));}
 #[cfg(not(target_arch="x86_64"))]{for k in 0..4{*prp.add(2*(v+k))=nm[k];*prp.add(2*(v+k)+1)=np[k];}}
}
#[inline(always)]unsafe fn c25_blk_simd<const DAMP_ZERO:bool>(lsp:*const u32,prp:*mut f32,omp:*mut f32,nc4:usize,c:usize,d:f32,d1:f32){
 let mut l=[[0usize;4];3];let mut pp=[[0.;4];3];let mut bv=[[0.;4];3];let mut o=[[0.;4];3];
 for q in 0..3{for k in 0..4{let li=*lsp.add(q*nc4+c+k)as usize;l[q][k]=li;pp[q][k]=*prp.add(li);bv[q][k]=*prp.add(li^1);}}
 for q in 0..3{
  #[cfg(target_arch="x86_64")]{use core::arch::x86_64::*;_mm_storeu_ps(o[q].as_mut_ptr(),_mm_loadu_ps(omp.add(q*nc4+c)));}
  #[cfg(not(target_arch="x86_64"))]{std::ptr::copy_nonoverlapping(omp.add(q*nc4+c),o[q].as_mut_ptr(),4);}
 } let mut r=[[0.;4];3];for q in 0..3{let n=c21_mul(c21_sub([1.;4],bv[q]),pp[q]);r[q]=c21_div(n,c21_max(c21_add(n,c21_mul(bv[q],o[q])),[1e-30;4]));}
 let pe=[c21_mul(r[1],r[2]),c21_mul(r[0],r[2]),c21_mul(r[0],r[1])];let mut on=[[0.;4];3];let mut ratio=[[0.;4];3];
 for q in 0..3{let n=c21_sub([1.;4],pe[q]);on[q]=c21_max(if DAMP_ZERO{n}else{c21_add(c21_mul([d;4],o[q]),c21_mul([d1;4],n))},[1e-6;4]);ratio[q]=c21_div(on[q],o[q]);}
 for q in 0..3{std::ptr::copy_nonoverlapping(on[q].as_ptr(),omp.add(q*nc4+c),4);}
 // Repeated variables across clauses must retain this exact product order.
 for q in 0..3{for k in 0..4{*prp.add(l[q][k])*=ratio[q][k];}}
}
fn c25_make_ls(inst:&Inst,uninit_ls:bool)->Vec<u32>{let nvx=inst.nvx;let nc=inst.nc;let nc4=(nc+3)&!3;
    let mut ls = if uninit_ls { Vec::with_capacity(3*nc4) } else {vec![(nvx as u32)<<1;3*nc4]};
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
    if uninit_ls {unsafe{let dst=ls.as_mut_ptr(); for q in 0..3{for c in nc..nc4{dst.add(q*nc4+c).write((nvx as u32)<<1);}}ls.set_len(3*nc4);}}
    ls
}
fn c25_spr_with_ls(inst: &Inst, p: &P, rng: &mut Rng64, st: &mut Stats, ls:&[u32],work:&mut C25SpWork) -> Vec<u8> {
    let (nv, nvx, nc) = (inst.nv, inst.nvx, inst.nc);
    let nc4 = (nc + 3) & !3;
    let om=&mut work.om;if om.len()!=3*nc4{*om=c25_rand_om(3*nc4,rng);}else{for ch in om.chunks_exact_mut(4){for h in 0..2{let r=rng.gen::<u64>();ch[2*h]=1.0-((r>>40)as f32)*(1.0/16777216.0)*0.999;ch[2*h+1]=1.0-(((r>>8)&0xffffff)as f32)*(1.0/16777216.0)*0.999;}}}
    let pr=&mut work.pr;let hp=&mut work.hp;let hm=&mut work.hm;let x=&mut work.x;
    hp.fill(0.0);hm.fill(0.0);x.fill(0);for dv in nv..nvx{hm[dv]=1.0;}
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
            #[cfg(target_arch="x86_64")]{use core::arch::x86_64::*;let one=_mm_set1_ps(1.0);let m=_mm_sub_ps(one,_mm_loadu_ps(hmp.add(v)));let p=_mm_sub_ps(one,_mm_loadu_ps(hpp.add(v)));_mm_storeu_ps(prp.add(2*v),_mm_unpacklo_ps(m,p));_mm_storeu_ps(prp.add(2*v+4),_mm_unpackhi_ps(m,p));}
            #[cfg(not(target_arch="x86_64"))]{for k in 0..4{*prp.add(2*(v+k))=1.0-*hmp.add(v+k);*prp.add(2*(v+k)+1)=1.0-*hpp.add(v+k);}}
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
            unsafe fn blk(lsp:*const u32,prp:*mut f32,omp:*mut f32,nc4:usize,c:usize,d:f32,d1:f32){if d==0.0{c25_blk_simd::<true>(lsp,prp,omp,nc4,c,d,d1)}else{c25_blk_simd::<false>(lsp,prp,omp,nc4,c,d,d1)}}
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
                c25_var4_simd(prp,hpp,hmp,xp,v,lam1,lg);
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
                    best.copy_from_slice(x);
                    if u == 0 { break; }
                } else if it - best_it >= p.stall { break; }
                if best_u <= p.stop_u { break; }
            }
        }
    }
    st.sp_it += it;
    st.sp_u = best_u;
    best
}
pub fn c25_spr_simd(inst:&Inst,p:&P,rng:&mut Rng64,st:&mut Stats,uninit_ls:bool)->Vec<u8>{let ls=c25_make_ls(inst,uninit_ls);let mut w=C25SpWork::new(inst);c25_spr_with_ls(inst,p,rng,st,&ls,&mut w)}
fn c25_run_inner_fast(ch:&Challenge,p:&P,seed:u64,total:Option<u64>,fast_build:bool)->(Vec<bool>,bool,Stats){
    let mut st = Stats::default();
    let mut inst = if fast_build && p.sp{c25_build_clauses(ch)}else if fast_build{build_fast(ch)}else{build(ch)};
    let mut rng = Rng64::seed_from_u64(seed);
    let mut x = vec![0u8; inst.nvx];
    let mut ok = false;
    let ls=if p.sp{Some(c25_make_ls(&inst,fast_build))}else{None};
    let mut work=if p.sp{Some(C25SpWork::new(&inst))}else{None};
    for a in 0..p.attempts {
        if total.map_or(false,|cap|st.flips>=cap){break;}
        st.attempts = a + 1;
        if p.sp {
            x = c25_spr_with_ls(&inst, p, &mut rng, &mut st,ls.as_ref().unwrap(),work.as_mut().unwrap());
        } else {
            for v in 0..inst.nv {
                let pc = inst.deg[2 * v + 1];
                let nc = inst.deg[2 * v];
                x[v] = (if (a>0 && total.is_some()) || pc == nc { rng.gen() } else { pc > nc }) as u8;
            }
        }
        for d in inst.nv..inst.nvx { x[d] = 0; }
        let mut bounded;
        let attempt_p=if let Some(cap)=total {bounded=p.clone();bounded.max_flips=bounded.max_flips.min(cap-st.flips);&bounded}else{p};
        if fast_build && p.sp && st.sp_u!=0 {c25_finish_occ(&mut inst);}
        let solved = if p.sp && st.sp_u==0 {true}else if fast_build{sls_fast_capture::<u32,false>(&inst,attempt_p,&mut x,&mut rng,&mut st,None,None,0,&mut Near::default())}else{sls::<u32>(&inst,attempt_p,&mut x,&mut rng,&mut st)};
        if solved { ok = true; break; }
    }
    (x[..inst.nv].iter().map(|&b| b == 1).collect(), ok, st)
}

pub struct UniformTail{inst:Inst,state:SlsState<u8>,x:Vec<u8>}
pub fn resume_uniform_tail(mut tail:UniformTail,p:&P,record_max:usize,reset:bool)->(Vec<bool>,bool,u64,Near){
 assert!(p.max_flips>=tail.state.flips);
 // The replacement schedule has different indices; do not reuse the old gi.
 tail.state.gi=p.giveup.partition_point(|g|g.0<tail.state.flips);
 if reset{tail.state.minlen=tail.state.len;} let mut rng=tail.state.rng.clone();let mut st=Stats::default();let mut near=Near::default();
 let ok=sls_fast_capture::<u8,true>(&tail.inst,p,&mut tail.x,&mut rng,&mut st,Some(tail.state),None,record_max,&mut near);
 (tail.x[..tail.inst.nv].iter().map(|&v|v==1).collect(),ok,st.flips,near)
}
pub fn run_restarts_resume_with_uniform_tail(ch:&Challenge,p:&P,seed:u64,total:u64,first_cap:u64,resume_cap:u64,min_limit:usize,record_max:usize,u_limit:usize)->(Vec<bool>,bool,Stats,Near,Option<UniformTail>){
    assert!(!p.sp && ch.num_variables<=10000);
    let inst=build_fast(ch);let mut st=Stats::default();let mut rng=Rng64::seed_from_u64(seed);let mut x=vec![0u8;inst.nvx];
    let mut saved=None;let mut saved_x=Vec::new();let mut ok=false;let mut uniform:Option<SlsState<u8>>=None;let mut uniform_x=Vec::new();
    let mut near=Near::default();
    for a in 0..p.attempts {
        if st.flips>=total{break;}
        st.attempts=a+1;
        for v in 0..inst.nv{let pc=inst.deg[2*v+1];let nc=inst.deg[2*v];x[v]=(if a>0 || pc==nc{rng.gen()}else{pc>nc})as u8;}
        for v in inst.nv..inst.nvx{x[v]=0;}
        let mut q=p.clone();q.max_flips=q.max_flips.min(total-st.flips);
        let solved=if a==0 {
            let solved=sls_fast_capture::<u8,true>(&inst,&q,&mut x,&mut rng,&mut st,None,Some(&mut saved),record_max,&mut near);
            if saved.as_ref().map_or(false,|s|!solved && s.flips==first_cap && s.minlen<=min_limit){saved_x=x.clone();}else{saved=None;}
            solved
        }else{
            let mut slot=None;let solved=sls_fast_capture::<u8,true>(&inst,&q,&mut x,&mut rng,&mut st,None,Some(&mut slot),record_max,&mut near);
            if !solved{if let Some(state)=slot{if state.flips==first_cap&&state.minlen<=u_limit&&uniform.as_ref().map_or(true,|old|state.minlen<old.minlen){uniform=Some(state);uniform_x=x.clone();}}}solved
        };
        if solved{ok=true;break;}
    }
    if !ok {if let Some(state)=saved{
        x=saved_x;let mut q=p.clone();q.max_flips=resume_cap;
        ok=sls_fast_capture::<u8,true>(&inst,&q,&mut x,&mut rng,&mut st,Some(state),None,record_max,&mut near);
    }}
    let result=x[..inst.nv].iter().map(|&v|v==1).collect();let tail=if ok{None}else{uniform.map(|state|UniformTail{inst,state,x:uniform_x})};(result,ok,st,near,tail)
}


impl UniformTail{pub fn initial_minlen(&self)->usize{self.state.minlen}}



pub fn c25_run_fast(ch:&Challenge,p:&P,seed:u64)->(Vec<bool>,bool,Stats){c25_run_inner_fast(ch,p,seed,None,true)}
fn c25_rand_om(ne:usize,rng:&mut Rng64)->Vec<f32>{assert_eq!(ne%4,0);let mut om=Vec::<f32>::with_capacity(ne);unsafe{let p=om.as_mut_ptr();for i in (0..ne).step_by(4){for h in 0..2{let r=rng.gen::<u64>();p.add(i+2*h).write(1.0-((r>>40)as f32)*(1.0/16777216.0)*0.999);p.add(i+2*h+1).write(1.0-(((r>>8)&0xffffff)as f32)*(1.0/16777216.0)*0.999);}}om.set_len(ne);}om}
fn c25_build_clauses(ch: &Challenge) -> Inst {
    let nv = ch.num_variables;
    let m = ch.clauses.len();
    let mut lits:Vec<u32> = Vec::with_capacity(3 * m + 3);
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
    unsafe { lits.set_len(3 * nc); }
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
    Inst {nv,nvx,nc,lits,deg:Vec::new(),off:Vec::new(),occ:Vec::new()}
}
fn c25_finish_occ(inst:&mut Inst){if !inst.off.is_empty(){return;} let nvx=inst.nvx;let nc=inst.nc;let lits=&inst.lits;
    let ne = 3 * nc;
    let mut deg = vec![0u32; 2 * nvx];
    let mut off = vec![0u32; 2 * nvx + 1];
    unsafe {
        let lp = lits.as_ptr();
        let dp = deg.as_mut_ptr();
        for e in 0..ne { *dp.add(*lp.add(e) as usize) += 1; }
        for l in 0..2 * nvx { off[l + 1] = off[l] + ((*dp.add(l) + 3) & !3); }
    }
    let total=off[2*nvx]as usize; let mut occ:Vec<u32> = Vec::with_capacity(total);
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
        for l in 0..2*nvx {for k in pos[l]as usize..off[l+1]as usize{oc.add(k).write((nc+(k&3))as u32);}}
        occ.set_len(total);
    }
    inst.deg=deg;inst.off=off;inst.occ=occ;
}
struct C25SpWork{om:Vec<f32>,pr:Vec<f32>,hp:Vec<f32>,hm:Vec<f32>,x:Vec<u8>}
impl C25SpWork{fn new(inst:&Inst)->Self{let n=inst.nvx;Self{om:Vec::new(),pr:vec![1.0;2*n+2],hp:vec![0.0;n+1],hm:vec![0.0;n+1],x:vec![0;n]}}}
#[cfg(test)]mod c25_lazy_tests{use super::*;
#[test]fn random_messages_match_bits_and_rng(){for n in [0,4,16,400]{let(mut a,mut b)=(Rng64::seed_from_u64(41),Rng64::seed_from_u64(41));let(x,y)=(rand_om(n,&mut a),c25_rand_om(n,&mut b));assert_eq!(x.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),y.iter().map(|v|v.to_bits()).collect::<Vec<_>>());assert_eq!(a.next_u64(),b.next_u64());}}
#[test]fn lazy_occurrences_match_eager(){let ch=Challenge{seed:[0;32],num_variables:7,clauses:vec![vec![1,2,3],vec![1,1,2],vec![2,-2,3],vec![-3,4,5]]};let a=build_fast(&ch);let mut b=c25_build_clauses(&ch);assert_eq!(a.lits,b.lits);assert!(b.occ.is_empty());c25_finish_occ(&mut b);assert_eq!(a.deg,b.deg);assert_eq!(a.off,b.off);assert_eq!(a.occ,b.occ);}}
#[cfg(test)]mod c25_tail_tests{
 use super::*;
 fn forced()->Challenge{Challenge{seed:[0;32],num_variables:31,clauses:(0..8).map(|m|(0..3).map(|v|if m&(1<<v)==0{v+1}else{-(v+1)}).collect()).collect()}}
 #[test]fn passive_capture_keeps_failed_prefix_and_a0_resume_exact(){
  let ch=forced();let mut p=P::for_instance(7500,32002);p.attempts=4;p.max_flips=32;p.giveup=vec![(16,usize::MAX,0,0),(32,usize::MAX,0,0)];
  for seed in 0..16{for lim in[0,6]{let(a,o,s,n)=run_restarts_resume_near_fast(&ch,&p,seed,128,32,128,lim,2);let(b,q,t,m,tail)=run_restarts_resume_with_uniform_tail(&ch,&p,seed,128,32,128,lim,2,6);assert_eq!((a,o,s.flips,s.attempts,n.u,n.x),(b,q,t.flips,t.attempts,m.u,m.x));assert!(tail.is_some());}}
 }
 #[test]fn solved_prefix_drops_uniform_tail_without_changing_assignment(){
  let ch=Challenge{seed:[0;32],num_variables:31,clauses:vec![vec![1,2,3],vec![-1,2,4]]};let mut p=P::for_instance(7500,32002);p.attempts=4;p.max_flips=1000;
  for seed in 0..8{let(a,o,s,n)=run_restarts_resume_near_fast(&ch,&p,seed,4000,1000,2000,6,2);let(b,q,t,m,tail)=run_restarts_resume_with_uniform_tail(&ch,&p,seed,4000,1000,2000,6,2,6);assert!(o&&q);assert!(tail.is_none());assert_eq!((a,s.flips,s.attempts,n.u,n.x),(b,t.flips,t.attempts,m.u,m.x));}
 }
 #[test]fn replacement_tail_schedule_reindexes_gi_and_stops_at_first_new_gate(){
  let ch=forced();let mut p=P::for_instance(7500,32002);p.attempts=2;p.max_flips=32;p.giveup=vec![(16,usize::MAX,0,0),(32,usize::MAX,0,0)];
  let(_,ok,_,_,tail)=run_restarts_resume_with_uniform_tail(&ch,&p,0,64,32,64,6,2,6);assert!(!ok);let tail=tail.unwrap();assert_eq!((tail.state.flips,tail.state.gi),(32,1));
  let mut q=p.clone();q.max_flips=128;q.giveup=vec![(64,0,0,0)];let(_,ok,extra,_)=resume_uniform_tail(tail,&q,2,false);assert!(!ok);assert_eq!(extra,32);
 }

 #[test]fn non_power_of_two_checkpoint_stops_after_exact_16m(){let ch=forced();let mut p=P::for_instance(7500,32002);p.attempts=2;p.max_flips=32;p.giveup=vec![];let(_,ok,_,_,tail)=run_restarts_resume_with_uniform_tail(&ch,&p,0,64,32,64,6,2,6);assert!(!ok);let mut tail=tail.unwrap();tail.state.flips=1<<25;tail.state.gi=0;let mut q=p.clone();q.max_flips=(1<<25)+64_000_000;q.giveup=vec![((1<<25)+16_000_000,0,0,0)];let(_,ok,extra,_)=resume_uniform_tail(tail,&q,2,true);assert!(!ok);assert_eq!(extra,16_000_000);}
}
#[cfg(test)]mod c25_workspace_tests{use super::*;
 #[test]fn reused_workspace_preserves_four_attempts_and_rng(){let ch=Challenge{seed:[0;32],num_variables:71,clauses:(0..300).map(|i|vec![1+(i%71)as i32,-(1+((i+13)%71)as i32),1+((i+27)%71)as i32]).collect()};let inst=build_fast(&ch);let mut p=P::for_instance(100000,420000);p.maxit=25;p.check=1;p.stop_u=0;let ls=c25_make_ls(&inst,true);let mut work=C25SpWork::new(&inst);let(mut a,mut b)=(Rng64::seed_from_u64(55),Rng64::seed_from_u64(55));for _ in 0..4{let(mut s,mut u)=(Stats::default(),Stats::default());let x=spr_simd(&inst,&p,&mut a,&mut s,true);let y=c25_spr_with_ls(&inst,&p,&mut b,&mut u,&ls,&mut work);assert_eq!(x,y);assert_eq!((s.sp_it,s.sp_u),(u.sp_it,u.sp_u));let(mut aa,mut bb)=(a.clone(),b.clone());for _ in 0..4{assert_eq!(aa.next_u64(),bb.next_u64());}}}
 #[test]fn u32_sls_keeps_flip_and_rng_parity(){let ch=Challenge{seed:[0;32],num_variables:31,clauses:(0..8).map(|m|(0..3).map(|v|if m&(1<<v)==0{v+1}else{-(v+1)}).collect()).collect()};let inst=build_fast(&ch);let mut p=P::for_instance(7500,32002);p.max_flips=20000;p.giveup=vec![];let(mut x,mut y)=(vec![0;inst.nvx],vec![0;inst.nvx]);let(mut a,mut b)=(Rng64::seed_from_u64(9),Rng64::seed_from_u64(9));let(mut s,mut u)=(Stats::default(),Stats::default());let(mut na,mut nb)=(Near::default(),Near::default());let ra=sls_fast_capture::<u32,true>(&inst,&p,&mut x,&mut a,&mut s,None,None,3,&mut na);let rb=sls_fast_capture::<u32,true>(&inst,&p,&mut y,&mut b,&mut u,None,None,3,&mut nb);assert_eq!((ra,x,s.flips,na.u,na.x,a.next_u64()),(rb,y,u.flips,nb.u,nb.x,b.next_u64()));}
 #[test]fn tail_checkpoint_stops_at_24m(){let ch=Challenge{seed:[0;32],num_variables:31,clauses:(0..4).flat_map(|g|(0..8).map(move |m|(0..3).map(|v|{let n=g*3+v+1;if m&(1<<v)==0{n}else{-n}}).collect())).collect()};let mut p=P::for_instance(7500,32002);p.attempts=2;p.max_flips=32;p.giveup=vec![];let(_,ok,_,_,tail)=run_restarts_resume_with_uniform_tail(&ch,&p,0,64,32,64,6,2,6);assert!(!ok);let mut tail=tail.unwrap();tail.state.flips=1<<25;let mut q=p.clone();q.max_flips=(1<<25)+64_000_000;q.giveup=vec![((1<<25)+24_000_000,3,0,0)];let(_,ok,extra,_)=resume_uniform_tail(tail,&q,2,true);assert!(!ok);assert_eq!(extra,24_000_000);}
}
#[cfg(test)]mod c25_parity_tests {
 use super::*;
 fn bits(a:&[f32],b:&[f32]){assert_eq!(a.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),b.iter().map(|x|x.to_bits()).collect::<Vec<_>>());}
 fn random_ch(seed:u64,n:usize)->Challenge{let mut r=Rng64::seed_from_u64(seed);let mut clauses:Vec<Vec<i32>>=(0..(n*4267/1000)).map(|_|(0..3).map(|_|{let v=r.gen_range(1..=n)as i32;if r.gen::<bool>(){v}else{-v}}).collect()).collect();clauses.extend([vec![1,1,2],vec![1,-1,3],vec![2,2,2]]);Challenge{seed:[0;32],num_variables:n,clauses}}
 #[test]fn simd_max_matches_scalar_including_nan_and_signed_zero(){let a=[0.,-0.,f32::NAN,-1.,1.,f32::NEG_INFINITY,f32::INFINITY];for &x in &a{for &y in &a{bits(&c21_max([x;4],[y;4]),&[fmax(x,y);4]);}}}
 #[test]fn staged_variable_arithmetic_has_zero_ulp_error(){let mut rng=Rng64::seed_from_u64(91);for _ in 0..1000{let mut pr:Vec<f32>=(0..24).map(|_|rng.gen_range(0.000001..1.0)).collect();let mut hp:Vec<f32>=(0..12).map(|_|rng.gen_range(0.0..0.9)).collect();let mut hm=hp.clone();let mut x=vec![0;12];let(mut a,mut b,mut c,mut d)=(pr.clone(),hp.clone(),hm.clone(),x.clone());unsafe{for k in 0..4{c21_var_scalar(pr.as_mut_ptr(),hp.as_mut_ptr(),hm.as_mut_ptr(),x.as_mut_ptr(),2+k,0.7,0.11);}c25_var4_simd(a.as_mut_ptr(),b.as_mut_ptr(),c.as_mut_ptr(),d.as_mut_ptr(),2,0.7,0.11);}bits(&pr,&a);bits(&hp,&b);bits(&hm,&c);assert_eq!(x,d);}}
 #[test]fn staged_clause_arithmetic_and_product_order_have_zero_ulp_error(){let mut rng=Rng64::seed_from_u64(92);for damp in [0.,0.2]{for _ in 0..1000{let l:Vec<u32>=(0..24).map(|_|rng.gen_range(0..16)).collect();let mut pr:Vec<f32>=(0..16).map(|_|rng.gen_range(0.000001..1.0)).collect();let mut om:Vec<f32>=(0..24).map(|_|rng.gen_range(0.000001..1.0)).collect();let(mut a,mut b)=(pr.clone(),om.clone());unsafe{c21_blk_scalar(l.as_ptr(),pr.as_mut_ptr(),om.as_mut_ptr(),8,0,damp,1.-damp);if damp==0.{c25_blk_simd::<true>(l.as_ptr(),a.as_mut_ptr(),b.as_mut_ptr(),8,0,damp,1.-damp);}else{c25_blk_simd::<false>(l.as_ptr(),a.as_mut_ptr(),b.as_mut_ptr(),8,0,damp,1.-damp);}}bits(&pr,&a);bits(&om,&b);}}}
 #[test]fn fast_build_normalization_occurrences_and_four_padding_lanes(){for seed in 0..12{let ch=random_ch(seed,127);let a=build(&ch);let b=build_fast(&ch);assert_eq!((a.nv,a.nvx,a.nc),(b.nv,b.nvx,b.nc));assert_eq!(a.lits,b.lits);assert_eq!(a.deg,b.deg);assert_eq!(a.off,b.off);for l in 0..2*a.nvx{let start=a.off[l]as usize;let end=start+a.deg[l]as usize;assert_eq!(&a.occ[start..end],&b.occ[start..end]);for k in end..a.off[l+1]as usize{assert_eq!(b.occ[k],(a.nc+(k&3))as u32);}}}}
 #[test]fn full_sp_paths_preserve_rng_iterations_residual_and_assignment(){for seed in 0..8{let ch=random_ch(seed,257);let a=build(&ch);let b=build_fast(&ch);let mut p=P::for_instance(100000,420000);p.maxit=80;p.check=1;p.stop_u=0;p.stall=40;for damp in [0.,0.2]{p.damp=damp;let mut rng=Rng64::seed_from_u64(seed);let mut st=Stats::default();let x=spr(&a,&p,&mut rng,&mut st);for uninit in [false,true]{let mut r=Rng64::seed_from_u64(seed);let mut s=Stats::default();let y=c25_spr_simd(if uninit{&b}else{&a},&p,&mut r,&mut s,uninit);assert_eq!(x,y);assert_eq!((st.sp_it,st.sp_u),(s.sp_it,s.sp_u));let mut rr=rng.clone();for _ in 0..10{assert_eq!(rr.next_u64(),r.next_u64());}}}}}
 fn states(a:&SlsState<u8>,b:&SlsState<u8>,nc:usize){assert_eq!(&a.cs[..nc],&b.cs[..nc]);assert_eq!(&a.pos[..nc],&b.pos[..nc]);assert_eq!(&a.ul[..a.len],&b.ul[..b.len]);assert_eq!(a.brk,b.brk);assert_eq!((a.len,a.flips,a.minlen,a.gi,a.best),(b.len,b.flips,b.minlen,b.gi,b.best));let(mut r,mut s)=(a.rng.clone(),b.rng.clone());for _ in 0..20{assert_eq!(r.next_u64(),s.next_u64());}}
 #[test]fn sentinel_kernel_full_checkpoint_and_resumed_state_are_exact(){for seed in 0..10{let ch=random_ch(seed,127);let a=build(&ch);let b=build_fast(&ch);let mut p=P::for_instance(7500,32002);p.max_flips=20000;p.giveup=vec![(4096,usize::MAX,0,0),(16384,usize::MAX,0,0)];let mut x=vec![0;a.nvx];let mut y=x.clone();let(mut r,mut s)=(Rng64::seed_from_u64(seed),Rng64::seed_from_u64(seed));let(mut st,mut ts)=(Stats::default(),Stats::default());let(mut na,mut nb)=(Near::default(),Near::default());let(mut sa,mut sb)=(None,None);let ok=sls_state_capture::<u8,true>(&a,&p,&mut x,&mut r,&mut st,None,Some(&mut sa),3,&mut na);let ok2=sls_fast_capture::<u8,true>(&b,&p,&mut y,&mut s,&mut ts,None,Some(&mut sb),3,&mut nb);assert_eq!(ok,ok2);assert_eq!(x,y);assert_eq!(na.u,nb.u);assert_eq!(na.x,nb.x);states(sa.as_ref().unwrap(),sb.as_ref().unwrap(),a.nc);if !ok{p.max_flips=50000;let(mut ca,mut cb)=(None,None);let ok=sls_state_capture::<u8,true>(&a,&p,&mut x,&mut r,&mut st,sa,Some(&mut ca),3,&mut na);let ok2=sls_fast_capture::<u8,true>(&b,&p,&mut y,&mut s,&mut ts,sb,Some(&mut cb),3,&mut nb);assert_eq!(ok,ok2);assert_eq!(x,y);assert_eq!((st.flips,na.u),(ts.flips,nb.u));assert_eq!(na.x,nb.x);states(ca.as_ref().unwrap(),cb.as_ref().unwrap(),a.nc);}}}
 #[test]fn near_and_restart_resume_wrappers_preserve_full_search(){for seed in 0..6{let ch=random_ch(seed,191);let mut p=P::for_instance(7500,32002);p.attempts=4;p.max_flips=8000;p.giveup=vec![(4096,usize::MAX,0,0)];let(a,o,s,n)=run_near(&ch,&p,seed,3);let(b,q,t,m)=run_near_fast(&ch,&p,seed,3);assert_eq!((a,o,s.flips,n.u,n.x),(b,q,t.flips,m.u,m.x));let(a,o,s,n)=run_restarts_resume_near(&ch,&p,seed,24000,8000,40000,usize::MAX,3);let(b,q,t,m)=run_restarts_resume_near_fast(&ch,&p,seed,24000,8000,40000,usize::MAX,3);assert_eq!((a,o,s.flips,s.attempts,n.u,n.x),(b,q,t.flips,t.attempts,m.u,m.x));}}
}
pub fn run_near_fast_minlen(ch:&Challenge,p:&P,seed:u64,record_max:usize)->(Vec<bool>,bool,Stats,Near,usize){
    assert!(!p.sp && ch.num_variables<=10000);
    let inst=build_fast(ch);let mut st=Stats::default();let mut rng=Rng64::seed_from_u64(seed);let mut x=vec![0;inst.nvx];let mut near=Near::default();let mut ok=false;let mut slot=None;
    for a in 0..p.attempts{
        st.attempts=a+1;
        for v in 0..inst.nv{let pc=inst.deg[2*v+1];let nc=inst.deg[2*v];x[v]=(if pc==nc{rng.gen()}else{pc>nc})as u8;}
        for v in inst.nv..inst.nvx{x[v]=0;}
        if sls_fast_capture::<u8,true>(&inst,p,&mut x,&mut rng,&mut st,None,Some(&mut slot),record_max,&mut near){ok=true;break;}
    }
    (x[..inst.nv].iter().map(|&v|v==1).collect(),ok,st,near,if ok{0}else{slot.as_ref().map_or(usize::MAX,|s|s.minlen)})
}
#[cfg(test)]mod c26_near_tests{use super::*;
 #[test]fn minlen_capture_preserves_trajectory_and_reports_final_state(){
  let ch=Challenge{seed:[0;32],num_variables:31,clauses:(0..8).map(|m|(0..3).map(|v|if m&(1<<v)==0{v+1}else{-(v+1)}).collect()).collect()};
  let mut p=P::for_instance(5000,21335);p.attempts=1;p.max_flips=5000;p.giveup=vec![];
  for seed in 0..8{let(a,ok,s,n)=run_near_fast(&ch,&p,seed,1);let(b,q,t,m,minlen)=run_near_fast_minlen(&ch,&p,seed,1);assert_eq!((a,ok,s.flips,n.u,n.x),(b,q,t.flips,m.u,m.x));assert_eq!(minlen,1);}
 }
}


fn c27_unsat_ls(ls:&[u32],nc:usize,nc4:usize,x:&[u8])->usize{
 #[inline(always)]unsafe fn one(lp:*const u32,xp:*const u8,nc4:usize,c:usize)->usize{let a=*lp.add(c);let b=*lp.add(nc4+c);let d=*lp.add(2*nc4+c);((*xp.add((a>>1)as usize)^((a&1)as u8))&(*xp.add((b>>1)as usize)^((b&1)as u8))&(*xp.add((d>>1)as usize)^((d&1)as u8)))as usize}
 #[inline(always)]unsafe fn four(lp:*const u32,xp:*const u8,nc4:usize,c:usize)->usize{one(lp,xp,nc4,c)+one(lp,xp,nc4,c+1)+one(lp,xp,nc4,c+2)+one(lp,xp,nc4,c+3)}
 let(mut u,mut c)=(0,0);unsafe{let(lp,xp)=(ls.as_ptr(),x.as_ptr());while c+16<=nc{u+=four(lp,xp,nc4,c)+four(lp,xp,nc4,c+4)+four(lp,xp,nc4,c+8)+four(lp,xp,nc4,c+12);c+=16;}while c<nc{u+=one(lp,xp,nc4,c);c+=1;}}u
}
#[inline(always)]unsafe fn c27_blk<const ARM:usize>(lsp:*const u32,prp:*mut f32,omp:*mut f32,nc4:usize,c:usize,d:f32,d1:f32){
 #[cfg(target_arch="x86_64")]{use core::arch::x86_64::{_mm_prefetch,_MM_HINT_T0};if c+12<=nc4{let from=if ARM==1{0}else{1};for q in from..3{for k in 0..4{let li=*lsp.add(q*nc4+c+8+k) as usize;_mm_prefetch(prp.add(li)as *const i8,_MM_HINT_T0);}}}}
 if d==0.0{c25_blk_simd::<true>(lsp,prp,omp,nc4,c,d,d1)}else{c25_blk_simd::<false>(lsp,prp,omp,nc4,c,d,d1)}
}
#[derive(Debug,Clone,Copy,PartialEq,Eq)]pub enum C27Status{Malformed,NeedsFallback,Unsolved,Solved}
fn c27_input_status(ch:&Challenge)->Option<C27Status>{let mut long=false;for c in &ch.clauses{if c.is_empty()||c.iter().any(|l|{let v=l.unsigned_abs()as usize;v==0||v>ch.num_variables}){return Some(C27Status::Malformed);}long|=c.len()>3;}if long{Some(C27Status::NeedsFallback)}else{None}}
fn c27_build<const ARM:usize>(ch:&Challenge)->Result<Inst,C27Status>{
 if let Some(s)=c27_input_status(ch){return Err(s);}let mut inst=if ch.clauses.iter().all(|c|c.len()==3){c25_build_clauses(ch)}else{let mut padded=ch.clone();for c in &mut padded.clauses{while c.len()<3{c.push(*c.last().unwrap());}}c25_build_clauses(&padded)};if ARM==1{return Ok(inst);}
 for c in inst.lits.chunks_exact_mut(3){c.sort_unstable();}
 let count=if ARM==2{(inst.nvx>>9)+2}else{((inst.nvx>>7)+2)*64};let mut hist=vec![0usize;count];
 let key=|a:u32,b:u32|if ARM==2{(a>>10)as usize}else{((a>>8)as usize)*64+(((b>>12)as usize)&63)};
 for c in inst.lits.chunks_exact(3){hist[key(c[0],c[1])]+=1;}
 let mut pos=vec![0usize;count];let mut sum=0;for i in 0..count{pos[i]=sum;sum+=hist[i];}let mut sorted=vec![0u32;inst.lits.len()];
 for c in inst.lits.chunks_exact(3){let k=key(c[0],c[1]);let j=pos[k];sorted[3*j..3*j+3].copy_from_slice(c);pos[k]+=1;}inst.lits=sorted;Ok(inst)
}
pub fn c27_solve<const ARM:usize>(ch:&Challenge,save:&dyn Fn(&tig_challenges::satisfiability::Solution)->anyhow::Result<()>)->anyhow::Result<C27Status>{
 let mut inst=match c27_build::<ARM>(ch){Ok(i)=>i,Err(s)=>return Ok(s)};let mut p=P::for_instance(ch.num_variables,ch.clauses.len());if ARM!=1 && ch.num_variables==100000 && ch.clauses.len()==420000{p.stall=15;}
 let seed=u64::from_le_bytes(ch.seed[..8].try_into().unwrap());let mut rng=Rng64::seed_from_u64(seed);let mut st=Stats::default();let ls=c25_make_ls(&inst,true);let mut work=C25SpWork::new(&inst);
 for a in 0..p.attempts{st.attempts=a+1;let mut x=c27_spr_with_ls::<ARM>(&inst,&p,&mut rng,&mut st,&ls,&mut work);for v in inst.nv..inst.nvx{x[v]=0;}
  if st.sp_u!=0{c25_finish_occ(&mut inst);}let ok=if st.sp_u==0{true}else{sls_fast_capture::<u32,false>(&inst,&p,&mut x,&mut rng,&mut st,None,None,0,&mut Near::default())};
  if ok && unsat_count(&inst,&x)==0{let sol=tig_challenges::satisfiability::Solution{variables:x[..inst.nv].iter().map(|&v|v==1).collect()};save(&sol)?;return Ok(C27Status::Solved);}
 }Ok(C27Status::Unsolved)
}

fn c27_spr_with_ls<const ARM:usize>(inst: &Inst, p: &P, rng: &mut Rng64, st: &mut Stats, ls:&[u32],work:&mut C25SpWork) -> Vec<u8> {
    let (nv, nvx, nc) = (inst.nv, inst.nvx, inst.nc);
    let nc4 = (nc + 3) & !3;
    let om=&mut work.om;if om.len()!=3*nc4{*om=c25_rand_om(3*nc4,rng);}else{for ch in om.chunks_exact_mut(4){for h in 0..2{let r=rng.gen::<u64>();ch[2*h]=1.0-((r>>40)as f32)*(1.0/16777216.0)*0.999;ch[2*h+1]=1.0-(((r>>8)&0xffffff)as f32)*(1.0/16777216.0)*0.999;}}}
    let pr=&mut work.pr;let hp=&mut work.hp;let hm=&mut work.hm;let x=&mut work.x;
    hp.fill(0.0);hm.fill(0.0);x.fill(0);for dv in nv..nvx{hm[dv]=1.0;}
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
            #[cfg(target_arch="x86_64")]{use core::arch::x86_64::*;let one=_mm_set1_ps(1.0);let m=_mm_sub_ps(one,_mm_loadu_ps(hmp.add(v)));let p=_mm_sub_ps(one,_mm_loadu_ps(hpp.add(v)));_mm_storeu_ps(prp.add(2*v),_mm_unpacklo_ps(m,p));_mm_storeu_ps(prp.add(2*v+4),_mm_unpackhi_ps(m,p));}
            #[cfg(not(target_arch="x86_64"))]{for k in 0..4{*prp.add(2*(v+k))=1.0-*hmp.add(v+k);*prp.add(2*(v+k)+1)=1.0-*hpp.add(v+k);}}
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

            let mut c = 0;
            while c + 16 <= nc4 {
                c27_blk::<ARM>(lsp, prp, omp, nc4, c, d, d1);
                c27_blk::<ARM>(lsp, prp, omp, nc4, c + 4, d, d1);
                c27_blk::<ARM>(lsp, prp, omp, nc4, c + 8, d, d1);
                c27_blk::<ARM>(lsp, prp, omp, nc4, c + 12, d, d1);
                c += 16;
            }
            while c < nc4 { c27_blk::<ARM>(lsp, prp, omp, nc4, c, d, d1); c += 4; }
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
                c25_var4_simd(prp,hpp,hmp,xp,v,lam1,lg);
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
                let u = c27_unsat_ls(ls,nc,nc4,&x);
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
                    best.copy_from_slice(x);
                    if u == 0 { break; }
                } else if it - best_it >= p.stall { break; }
                if best_u <= p.stop_u { break; }
            }
        }
    }
    st.sp_it += it;
    st.sp_u = best_u;
    best
}

#[cfg(test)]
fn c27_spr_scalar<const ARM:usize>(inst: &Inst, p: &P, rng: &mut Rng64, st: &mut Stats, ls:&[u32],work:&mut C25SpWork) -> Vec<u8> {
    let (nv, nvx, nc) = (inst.nv, inst.nvx, inst.nc);
    let nc4 = (nc + 3) & !3;
    let om=&mut work.om;if om.len()!=3*nc4{*om=c25_rand_om(3*nc4,rng);}else{for ch in om.chunks_exact_mut(4){for h in 0..2{let r=rng.gen::<u64>();ch[2*h]=1.0-((r>>40)as f32)*(1.0/16777216.0)*0.999;ch[2*h+1]=1.0-(((r>>8)&0xffffff)as f32)*(1.0/16777216.0)*0.999;}}}
    let pr=&mut work.pr;let hp=&mut work.hp;let hm=&mut work.hm;let x=&mut work.x;
    hp.fill(0.0);hm.fill(0.0);x.fill(0);for dv in nv..nvx{hm[dv]=1.0;}
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
            for k in 0..4{*prp.add(2*(v+k))=1.0-*hmp.add(v+k);*prp.add(2*(v+k)+1)=1.0-*hpp.add(v+k);}
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

            let mut c = 0;
            while c + 16 <= nc4 {
                c21_blk_scalar(lsp, prp, omp, nc4, c, d, d1);
                c21_blk_scalar(lsp, prp, omp, nc4, c + 4, d, d1);
                c21_blk_scalar(lsp, prp, omp, nc4, c + 8, d, d1);
                c21_blk_scalar(lsp, prp, omp, nc4, c + 12, d, d1);
                c += 16;
            }
            while c < nc4 { c21_blk_scalar(lsp, prp, omp, nc4, c, d, d1); c += 4; }
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
                for k in 0..4{c21_var_scalar(prp,hpp,hmp,xp,v+k,lam1,lg);}
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
                let u = c27_unsat_ls(ls,nc,nc4,&x);
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
                    best.copy_from_slice(x);
                    if u == 0 { break; }
                } else if it - best_it >= p.stall { break; }
                if best_u <= p.stop_u { break; }
            }
        }
    }
    st.sp_it += it;
    st.sp_u = best_u;
    best
}

#[cfg(test)]mod c27_tests{use super::*;use rand::RngCore;
 fn ch(clauses:Vec<Vec<i32>>)->Challenge{Challenge{seed:[3;32],num_variables:7,clauses}}
 fn check<const ARM:usize>(c:&Challenge){let inst=c27_build::<ARM>(c).unwrap();let ls=c25_make_ls(&inst,true);let nc4=(inst.nc+3)&!3;for mask in 0..(1usize<<c.num_variables){let mut x=vec![0u8;inst.nvx];for i in 0..inst.nv{x[i]=((mask>>i)&1)as u8;}let raw_ok=c.clauses.iter().all(|cl|cl.iter().any(|&l|(x[l.unsigned_abs() as usize-1]!=0)==(l>0)));assert_eq!(unsat_count(&inst,&x)==0,raw_ok);assert_eq!(c27_unsat_ls(&ls,inst.nc,nc4,&x),unsat_count(&inst,&x));}}
 #[test]fn normalized_fast_accept_matches_raw_formula_for_all_assignments(){let c=ch(vec![vec![1],vec![-1,2],vec![1,-1,2],vec![2,2,3],vec![4,4,4],vec![-1,2,3],vec![4,-5,6]]);check::<1>(&c);check::<2>(&c);check::<3>(&c);let cube=ch((0..8).map(|m|(0..3).map(|v|if m&(1<<v)==0{v+1}else{-(v+1)}).collect()).collect());check::<1>(&cube);check::<2>(&cube);check::<3>(&cube);}
 #[test]fn invalid_input_never_calls_fast_accept(){for bad in [vec![],vec![0,1,2],vec![8,1,2],vec![-8,1,2],vec![i32::MIN,1,2]]{let c=ch(vec![bad]);for f in [c27_solve::<1> as fn(&Challenge,&dyn Fn(&tig_challenges::satisfiability::Solution)->anyhow::Result<()>)->anyhow::Result<C27Status>,c27_solve::<2>,c27_solve::<3>]{assert_eq!(f(&c,&|_|panic!("invalid callback")).unwrap(),C27Status::Malformed);}}let c=ch(vec![vec![1,2,3,4]]);assert_eq!(c27_solve::<1>(&c,&|_|panic!("unexpected callback")).unwrap(),C27Status::NeedsFallback);let c=ch(vec![vec![1,2,3,4],vec![0,1,2]]);assert_eq!(c27_solve::<1>(&c,&|_|panic!("unexpected callback")).unwrap(),C27Status::Malformed);}
 #[test]fn spatial_sse_and_scalar_sp_are_bit_exact(){let c=ch(vec![vec![1,2,3],vec![-1,3,4],vec![4,4,5],vec![1,-1,7],vec![-2,-5,6],vec![6,6,6]]);for arm in [1,2,3]{let inst=match arm{1=>c27_build::<1>(&c),2=>c27_build::<2>(&c),_=>c27_build::<3>(&c)}.unwrap();let ls=c25_make_ls(&inst,true);for damp in [0.,0.2]{let mut p=P::for_instance(100000,420000);p.damp=damp;p.maxit=20;p.check=1;p.stop_u=0;let(mut r,mut q)=(Rng64::seed_from_u64(42),Rng64::seed_from_u64(42));let(mut s,mut z)=(Stats::default(),Stats::default());let(mut a,mut b)=(C25SpWork::new(&inst),C25SpWork::new(&inst));for _ in 0..4{let x=match arm{1=>c27_spr_with_ls::<1>(&inst,&p,&mut r,&mut s,&ls,&mut a),2=>c27_spr_with_ls::<2>(&inst,&p,&mut r,&mut s,&ls,&mut a),_=>c27_spr_with_ls::<3>(&inst,&p,&mut r,&mut s,&ls,&mut a)};let y=match arm{1=>c27_spr_scalar::<1>(&inst,&p,&mut q,&mut z,&ls,&mut b),2=>c27_spr_scalar::<2>(&inst,&p,&mut q,&mut z,&ls,&mut b),_=>c27_spr_scalar::<3>(&inst,&p,&mut q,&mut z,&ls,&mut b)};assert_eq!(x,y);assert_eq!((s.sp_it,s.sp_u),(z.sp_it,z.sp_u));for(a,b)in [(&a.om,&b.om),(&a.pr,&b.pr),(&a.hp,&b.hp),(&a.hm,&b.hm)]{assert_eq!(a.iter().map(|f|f.to_bits()).collect::<Vec<_>>(),b.iter().map(|f|f.to_bits()).collect::<Vec<_>>());}let(mut rr,mut qq)=(r.clone(),q.clone());assert_eq!(rr.next_u64(),qq.next_u64());}}}}
 #[test]fn unsorted_arm_matches_c26_sp_and_rng(){let c=ch(vec![vec![1,2,3],vec![-1,3,4],vec![4,4,5],vec![1,-1,7],vec![-2,-5,6]]);let inst=c27_build::<1>(&c).unwrap();let ls=c25_make_ls(&inst,true);let mut p=P::for_instance(100000,420000);p.maxit=20;p.check=1;p.stop_u=0;let(mut a,mut b)=(Rng64::seed_from_u64(7),Rng64::seed_from_u64(7));let(mut x,mut y)=(Stats::default(),Stats::default());let(mut wa,mut wb)=(C25SpWork::new(&inst),C25SpWork::new(&inst));let r=c25_spr_with_ls(&inst,&p,&mut a,&mut x,&ls,&mut wa);let s=c27_spr_with_ls::<1>(&inst,&p,&mut b,&mut y,&ls,&mut wb);assert_eq!((r,x.sp_it,x.sp_u,a.next_u64()),(s,y.sp_it,y.sp_u,b.next_u64()));}
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn c31_sls_fast_avx2_bmi<B:Count,const CAPTURE:bool,const PREFETCH:bool>(inst:&Inst,p:&P,x:&mut[u8],rng:&mut Rng64,st:&mut Stats,resume:Option<SlsState<B>>,checkpoint:Option<&mut Option<SlsState<B>>>,record_max:usize,near:&mut Near)->bool {
    let (nvx, nc) = (inst.nvx, inst.nc);
    let mask = nvx.next_power_of_two() - 1;
    let poff = &inst.off[..];
    let resumed=resume.is_some();
    let mut state=resume.unwrap_or_else(||c21_initial_state::<B>(inst,x,rng,mask));
    let SlsState{mut cs,mut brk,mut ul,mut pos,mut len,mut flips,mut minlen,mut best,mut gi,rng:mut rl}=state;
    cs[nc..nc+4].fill(SENT);
    let lits = &inst.lits[..];
    let mut wt = [0u32; 256];
    for b in 0..63 { wt[b] = ((1u64 << 22) as f64 * (1.0 + b as f64).powf(-p.cb)).max(1.0) as u32; }
    let start_flips=flips;
    unsafe {
        let csp = cs.as_mut_ptr();
        let bp = brk.as_mut_ptr();
        let ulp = ul.as_mut_ptr();
        let psp = pos.as_mut_ptr();
        let op = inst.occ.as_ptr();
        let ncu = nc as u32;
        let mut stop_len=if CAPTURE{record_max.min(near.u.saturating_sub(1))}else{0};
        loop {
            let seg = p.max_flips.min(p.giveup.get(gi).map_or(u64::MAX, |g| g.0));
            loop {
            while (len > stop_len) & (flips < seg) {
                (csp.add(nc)as *mut [u32;4]).write([SENT;4]);
                let r = rl.gen::<u64>();
                let c = *ulp.add(range(r as u32, len)) as usize;
                let l0 = g(lits, 3 * c);
                let l1 = g(lits, 3 * c + 1);
                let l2 = g(lits, 3 * c + 2);
                let w0 = (*bp.add((l0 >> 1) as usize)).weight(&wt);
                let w1 = (*bp.add((l1 >> 1) as usize)).weight(&wt);
                let w2 = (*bp.add((l2 >> 1) as usize)).weight(&wt);
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
                if PREFETCH { let start=g(poff,fl) as usize; let end=g(poff,fl+1) as usize; if start<end { for j in 0..4 { std::arch::x86_64::_mm_prefetch(csp.add(*op.add(start+j) as usize) as *const i8, std::arch::x86_64::_MM_HINT_T0); } } }
                while k < ke {
                    if PREFETCH && k+4<ke { for j in 0..4 { std::arch::x86_64::_mm_prefetch(csp.add(*op.add(k+4+j) as usize) as *const i8, std::arch::x86_64::_MM_HINT_T0); } }
                    for j in 0..4 {
                        let cc = *op.add(k + j);
                        let s = *csp.add(cc as usize);
                        let n = s >> 24;
                        bn += (n == 0) as u32;
                        (*bp.add((s & 0xffffff) as usize)).dec((n == 1) as u32);
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
                    if PREFETCH && k+4<ke { for j in 0..4 { std::arch::x86_64::_mm_prefetch(csp.add(*op.add(k+4+j) as usize) as *const i8, std::arch::x86_64::_MM_HINT_T0); } }
                    for j in 0..4 {
                        let cc = *op.add(k + j);
                        let s = (*csp.add(cc as usize) - (1 << 24)) ^ vv;
                        *csp.add(cc as usize) = s;
                        (*bp.add((s & 0xffffff) as usize)).inc((s >> 24 == 1) as u32);
                        let add = (s == 0) as u32;
                        *ulp.add(len) = cc;
                        *psp.add((ncu ^ ((cc ^ ncu) & add.wrapping_neg())) as usize) = len as u32;
                        len += add as usize;
                    }
                    k += 4;
                }
                *bp.add(v) = B::of(bn);
                flips += 1;
                minlen = minlen.min(len);
            }
            // Capture outside the flip kernel, then resume the same segment.
            // In particular, do not run giveup checks at a capture-only boundary.
            if CAPTURE && len>0 && len<=stop_len {
                near.u=len;near.x=x[..inst.nv].to_vec();stop_len=len-1;continue;
            }
            break;
            }
            if (len == 0) | (flips >= p.max_flips) { break; }
            if p.giveup.get(gi).map_or(false, |g| g.0 == flips) {
                let (_, lim, f1, lim1) = p.giveup[gi];
                best[flips.trailing_zeros() as usize] = minlen;
                if minlen > lim && (f1 == 0 || best[f1.trailing_zeros() as usize] > lim1) { break; }
                gi += 1;
            }
        }
        *rng = rl.clone();
    }
    st.flips += flips-start_flips;
    if let Some(slot)=checkpoint { *slot=Some(SlsState{cs,brk,ul,pos,len,flips,minlen,best,gi,rng:rl}); }
    len == 0
}
fn c31_sls_fast_dispatch<B:Count,const CAPTURE:bool,const PREFETCH:bool>(inst:&Inst,p:&P,x:&mut[u8],rng:&mut Rng64,st:&mut Stats,resume:Option<SlsState<B>>,checkpoint:Option<&mut Option<SlsState<B>>>,record_max:usize,near:&mut Near)->bool {
 #[cfg(target_arch="x86_64")]
 if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("bmi1") && std::arch::is_x86_feature_detected!("bmi2") && std::arch::is_x86_feature_detected!("lzcnt") && std::arch::is_x86_feature_detected!("popcnt") {
  return unsafe { c31_sls_fast_avx2_bmi::<B,CAPTURE,PREFETCH>(inst,p,x,rng,st,resume,checkpoint,record_max,near) };
 }
 sls_fast_capture::<B,CAPTURE>(inst,p,x,rng,st,resume,checkpoint,record_max,near)
}
pub fn c31_run_near_fast_minlen<const PREFETCH:bool>(ch:&Challenge,p:&P,seed:u64,record_max:usize)->(Vec<bool>,bool,Stats,Near,usize){
    assert!(!p.sp && ch.num_variables<=10000);
    let inst=build_fast(ch);let mut st=Stats::default();let mut rng=Rng64::seed_from_u64(seed);let mut x=vec![0;inst.nvx];let mut near=Near::default();let mut ok=false;let mut slot=None;
    for a in 0..p.attempts{
        st.attempts=a+1;
        for v in 0..inst.nv{let pc=inst.deg[2*v+1];let nc=inst.deg[2*v];x[v]=(if pc==nc{rng.gen()}else{pc>nc})as u8;}
        for v in inst.nv..inst.nvx{x[v]=0;}
        if c31_sls_fast_dispatch::<u8,true,PREFETCH>(&inst,p,&mut x,&mut rng,&mut st,None,Some(&mut slot),record_max,&mut near){ok=true;break;}
    }
    (x[..inst.nv].iter().map(|&v|v==1).collect(),ok,st,near,if ok{0}else{slot.as_ref().map_or(usize::MAX,|s|s.minlen)})
}
pub fn c31_resume_uniform_tail<const PREFETCH:bool>(mut tail:UniformTail,p:&P,record_max:usize,reset:bool)->(Vec<bool>,bool,u64,Near){
 assert!(p.max_flips>=tail.state.flips);
 // The replacement schedule has different indices; do not reuse the old gi.
 tail.state.gi=p.giveup.partition_point(|g|g.0<tail.state.flips);
 if reset{tail.state.minlen=tail.state.len;} let mut rng=tail.state.rng.clone();let mut st=Stats::default();let mut near=Near::default();
 let ok=c31_sls_fast_dispatch::<u8,true,PREFETCH>(&tail.inst,p,&mut tail.x,&mut rng,&mut st,Some(tail.state),None,record_max,&mut near);
 (tail.x[..tail.inst.nv].iter().map(|&v|v==1).collect(),ok,st.flips,near)
}
pub fn c31_run_restarts_resume_with_uniform_tail_minlen<const PREFETCH:bool>(ch:&Challenge,p:&P,seed:u64,total:u64,first_cap:u64,resume_cap:u64,min_limit:usize,record_max:usize,u_limit:usize)->(Vec<bool>,bool,Stats,Near,Option<UniformTail>,usize){
    assert!(!p.sp && ch.num_variables<=10000);
    let inst=build_fast(ch);let mut st=Stats::default();let mut rng=Rng64::seed_from_u64(seed);let mut x=vec![0u8;inst.nvx];
    let mut saved=None;let mut saved_x=Vec::new();let mut ok=false;let mut uniform:Option<SlsState<u8>>=None;let mut uniform_x=Vec::new();
    let mut near=Near::default();let mut best_minlen=usize::MAX;
    for a in 0..p.attempts {
        if st.flips>=total{break;}
        st.attempts=a+1;
        for v in 0..inst.nv{let pc=inst.deg[2*v+1];let nc=inst.deg[2*v];x[v]=(if a>0 || pc==nc{rng.gen()}else{pc>nc})as u8;}
        for v in inst.nv..inst.nvx{x[v]=0;}
        let mut q=p.clone();q.max_flips=q.max_flips.min(total-st.flips);
        let solved=if a==0 {
            let solved=c31_sls_fast_dispatch::<u8,true,PREFETCH>(&inst,&q,&mut x,&mut rng,&mut st,None,Some(&mut saved),record_max,&mut near);
            best_minlen=best_minlen.min(saved.as_ref().unwrap().minlen);
            if saved.as_ref().map_or(false,|s|!solved && s.flips==first_cap && s.minlen<=min_limit){saved_x=x.clone();}else{saved=None;}
            solved
        }else{
            let mut slot=None;let solved=c31_sls_fast_dispatch::<u8,true,PREFETCH>(&inst,&q,&mut x,&mut rng,&mut st,None,Some(&mut slot),record_max,&mut near);
            if !solved{if let Some(state)=slot{best_minlen=best_minlen.min(state.minlen);if state.flips==first_cap&&state.minlen<=u_limit&&uniform.as_ref().map_or(true,|old|state.minlen<old.minlen){uniform=Some(state);uniform_x=x.clone();}}}solved
        };
        if solved{ok=true;break;}
    }
    if !ok {if let Some(mut state)=saved{
        x=saved_x;let mut q=p.clone();q.max_flips=resume_cap;state.gi=q.giveup.partition_point(|g|g.0<state.flips);
        let mut resumed=None;ok=c31_sls_fast_dispatch::<u8,true,PREFETCH>(&inst,&q,&mut x,&mut rng,&mut st,Some(state),Some(&mut resumed),record_max,&mut near);best_minlen=best_minlen.min(resumed.as_ref().unwrap().minlen);
    }}
    let result=x[..inst.nv].iter().map(|&v|v==1).collect();let tail=if ok{None}else{uniform.map(|state|UniformTail{inst,state,x:uniform_x})};(result,ok,st,near,tail,if ok{0}else{best_minlen})
}
#[cfg(test)] mod c31_kernel_tests {
 use super::*; use rand::{RngCore,SeedableRng};
 fn fixture()->Challenge{Challenge{seed:[0;32],num_variables:3,clauses:(0..8).map(|m|(0..3).map(|v|if m&(1<<v)==0{v+1}else{-(v+1)}).collect()).collect()}}
 fn one(mode:usize)->(Vec<u8>,Stats,Near,SlsState<u8>,u64){
  let inst=build_fast(&fixture()); let mut p=P::for_instance(5000,21335);p.sp=false;p.attempts=1;p.max_flips=4096;p.giveup=vec![(1024,usize::MAX,0,0),(2048,usize::MAX,0,0)];
  let mut x=vec![0;inst.nvx];let mut rng=Rng64::seed_from_u64(42);let mut st=Stats::default();let mut near=Near::default();let mut slot=None;
  for end in [4096,8192]{p.max_flips=end;let old=slot.take();let ok=match mode{0=>sls_fast_capture::<u8,true>(&inst,&p,&mut x,&mut rng,&mut st,old,Some(&mut slot),3,&mut near),1=>c31_sls_fast_dispatch::<u8,true,true>(&inst,&p,&mut x,&mut rng,&mut st,old,Some(&mut slot),3,&mut near),2=>unsafe{c31_sls_fast_avx2_bmi::<u8,true,false>(&inst,&p,&mut x,&mut rng,&mut st,old,Some(&mut slot),3,&mut near)},_=>unsafe{c31_sls_fast_avx2_bmi::<u8,true,true>(&inst,&p,&mut x,&mut rng,&mut st,old,Some(&mut slot),3,&mut near)}};assert!(!ok);}
  let r=rng.next_u64();(x,st,near,slot.unwrap(),r)
 }
 #[test] #[cfg(target_arch="x86_64")] fn initial_and_resumed_states_match_scalar(){
  if !(std::arch::is_x86_feature_detected!("avx2")&&std::arch::is_x86_feature_detected!("bmi1")&&std::arch::is_x86_feature_detected!("bmi2")&&std::arch::is_x86_feature_detected!("lzcnt")&&std::arch::is_x86_feature_detected!("popcnt")){return;}
  let (x,a,n,s,r)=one(0);for mode in 1..=3 {let(y,b,m,t,q)=one(mode);assert_eq!(x,y);assert_eq!((a.flips,n.u,&n.x,r),(b.flips,m.u,&m.x,q));assert_eq!((&s.cs,&s.brk,&s.ul,&s.pos,s.len,s.flips,s.minlen,s.gi,&s.best),(&t.cs,&t.brk,&t.ul,&t.pos,t.len,t.flips,t.minlen,t.gi,&t.best));let(mut sr,mut tr)=(s.rng.clone(),t.rng.clone());for _ in 0..8{assert_eq!(sr.next_u64(),tr.next_u64());}}
 }
 #[test] fn best_minlen_and_uniform_tail_preserve_scalar_result(){
  let ch=fixture();let mut p=P::for_instance(7500,32002);p.sp=false;p.attempts=4;p.max_flips=4096;p.giveup=vec![(1024,usize::MAX,0,0),(2048,usize::MAX,0,0)];
  for seed in 0..5 {let(a,ok,s,n,tail)=run_restarts_resume_with_uniform_tail(&ch,&p,seed,16384,4096,8192,usize::MAX,2,usize::MAX);let(b,pass,t,m,other,best)=c31_run_restarts_resume_with_uniform_tail_minlen::<true>(&ch,&p,seed,16384,4096,8192,usize::MAX,2,usize::MAX);assert_eq!((a,ok,s.flips,s.attempts,n.u,n.x),(b,pass,t.flips,t.attempts,m.u,m.x));assert_eq!(best,1);let mut q=p.clone();q.max_flips=8192;q.giveup=vec![(6144,usize::MAX,0,0)];let(x,u,f,n)=resume_uniform_tail(tail.unwrap(),&q,2,true);let(y,v,g,m)=c31_resume_uniform_tail::<true>(other.unwrap(),&q,2,true);assert_eq!((x,u,f,n.u,n.x),(y,v,g,m.u,m.x));}
 }
}

#[cfg(test)]mod c31_resume_partition_test{
 use super::*;use rand::SeedableRng;
 #[test]fn reindex_resumes_at_the_new_next_checkpoint(){
  let ch=Challenge{seed:[0;32],num_variables:3,clauses:(0..8).map(|m|(0..3).map(|v|if m&(1<<v)==0{v+1}else{-(v+1)}).collect()).collect()};
  let inst=build_fast(&ch);let mut p=P::for_instance(7500,32002);p.sp=false;p.max_flips=6;p.giveup=vec![(4,usize::MAX,0,0),(8,0,0,0)];
  let mut x=vec![0;inst.nvx];let mut rng=Rng64::seed_from_u64(42);let mut st=Stats::default();let mut near=Near::default();let mut slot=None;
  assert!(!c31_sls_fast_dispatch::<u8,true,true>(&inst,&p,&mut x,&mut rng,&mut st,None,Some(&mut slot),0,&mut near));
  let mut state=slot.unwrap();assert_eq!(state.flips,6);state.gi=0;p.max_flips=100;state.gi=p.giveup.partition_point(|g|g.0<state.flips);assert_eq!(state.gi,1);
  let mut resumed=None;assert!(!c31_sls_fast_dispatch::<u8,true,true>(&inst,&p,&mut x,&mut rng,&mut st,Some(state),Some(&mut resumed),0,&mut near));assert_eq!(resumed.unwrap().flips,8);
 }
}

#[cfg(target_arch="x86_64")]fn c32_has_avx2()->bool{std::arch::is_x86_feature_detected!("avx2")&&std::arch::is_x86_feature_detected!("bmi1")&&std::arch::is_x86_feature_detected!("bmi2")&&std::arch::is_x86_feature_detected!("lzcnt")&&std::arch::is_x86_feature_detected!("popcnt")}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline]
unsafe fn c32_var8_avx2(prp:*mut f32,hpp:*mut f32,hmp:*mut f32,xp:*mut u8,v:usize,lam1:f32,lg:f32){
 use core::arch::x86_64::*;
 let lo0=_mm_loadu_ps(prp.add(2*v));let hi0=_mm_loadu_ps(prp.add(2*v+4));let lo1=_mm_loadu_ps(prp.add(2*v+8));let hi1=_mm_loadu_ps(prp.add(2*v+12));
 let m_lo=_mm_shuffle_ps::<0x88>(lo0,hi0);let p_lo=_mm_shuffle_ps::<0xdd>(lo0,hi0);let m_hi=_mm_shuffle_ps::<0x88>(lo1,hi1);let p_hi=_mm_shuffle_ps::<0xdd>(lo1,hi1);
 let pm=_mm256_insertf128_ps::<1>(_mm256_castps128_ps256(m_lo),m_hi);let pp=_mm256_insertf128_ps::<1>(_mm256_castps128_ps256(p_lo),p_hi);
 let o1=_mm256_loadu_ps(hpp.add(v));let o0=_mm256_loadu_ps(hmp.add(v));let one=_mm256_set1_ps(1.0);let eps=_mm256_set1_ps(1e-30);let vl=_mm256_set1_ps(lam1);let vg=_mm256_set1_ps(lg);
 let pip=_mm256_mul_ps(_mm256_sub_ps(one,pp),pm);let pim=_mm256_mul_ps(_mm256_sub_ps(one,pm),pp);let denom=_mm256_max_ps(_mm256_add_ps(_mm256_add_ps(pip,pim),_mm256_mul_ps(pp,pm)),eps);
 let inv=_mm256_div_ps(one,denom);let wp=_mm256_mul_ps(pip,inv);let wm=_mm256_mul_ps(pim,inv);
 let h1=_mm256_add_ps(_mm256_mul_ps(vl,o1),_mm256_mul_ps(vg,wp));let h0=_mm256_add_ps(_mm256_mul_ps(vl,o0),_mm256_mul_ps(vg,wm));
 let nm=_mm256_mul_ps(pm,_mm256_div_ps(_mm256_sub_ps(one,h0),_mm256_sub_ps(one,o0)));let np=_mm256_mul_ps(pp,_mm256_div_ps(_mm256_sub_ps(one,h1),_mm256_sub_ps(one,o1)));
 _mm256_storeu_ps(hpp.add(v),h1);_mm256_storeu_ps(hmp.add(v),h0);let mask=_mm256_movemask_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(wp,wm))as u32;
 (xp.add(v)as *mut u64).write_unaligned(_pdep_u64(mask as u64,0x0101_0101_0101_0101u64).to_le());
 let ml=_mm256_castps256_ps128(nm);let mh=_mm256_extractf128_ps::<1>(nm);let pl=_mm256_castps256_ps128(np);let ph=_mm256_extractf128_ps::<1>(np);
 _mm_storeu_ps(prp.add(2*v),_mm_unpacklo_ps(ml,pl));_mm_storeu_ps(prp.add(2*v+4),_mm_unpackhi_ps(ml,pl));_mm_storeu_ps(prp.add(2*v+8),_mm_unpacklo_ps(mh,ph));_mm_storeu_ps(prp.add(2*v+12),_mm_unpackhi_ps(mh,ph));
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn c32_spr_avx2<const ARM:usize>(inst: &Inst, p: &P, rng: &mut Rng64, st: &mut Stats, ls:&[u32],work:&mut C25SpWork) -> Vec<u8> {
    let (nv, nvx, nc) = (inst.nv, inst.nvx, inst.nc);
    let nc4 = (nc + 3) & !3;
    let om=&mut work.om;if om.len()!=3*nc4{*om=c25_rand_om(3*nc4,rng);}else{for ch in om.chunks_exact_mut(4){for h in 0..2{let r=rng.gen::<u64>();ch[2*h]=1.0-((r>>40)as f32)*(1.0/16777216.0)*0.999;ch[2*h+1]=1.0-(((r>>8)&0xffffff)as f32)*(1.0/16777216.0)*0.999;}}}
    let pr=&mut work.pr;let hp=&mut work.hp;let hm=&mut work.hm;let x=&mut work.x;
    hp.fill(0.0);hm.fill(0.0);x.fill(0);for dv in nv..nvx{hm[dv]=1.0;}
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
            #[cfg(target_arch="x86_64")]{use core::arch::x86_64::*;let one=_mm_set1_ps(1.0);let m=_mm_sub_ps(one,_mm_loadu_ps(hmp.add(v)));let p=_mm_sub_ps(one,_mm_loadu_ps(hpp.add(v)));_mm_storeu_ps(prp.add(2*v),_mm_unpacklo_ps(m,p));_mm_storeu_ps(prp.add(2*v+4),_mm_unpackhi_ps(m,p));}
            #[cfg(not(target_arch="x86_64"))]{for k in 0..4{*prp.add(2*(v+k))=1.0-*hmp.add(v+k);*prp.add(2*(v+k)+1)=1.0-*hpp.add(v+k);}}
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

            let mut c = 0;
            while c + 16 <= nc4 {
                c27_blk::<ARM>(lsp, prp, omp, nc4, c, d, d1);
                c27_blk::<ARM>(lsp, prp, omp, nc4, c + 4, d, d1);
                c27_blk::<ARM>(lsp, prp, omp, nc4, c + 8, d, d1);
                c27_blk::<ARM>(lsp, prp, omp, nc4, c + 12, d, d1);
                c += 16;
            }
            while c < nc4 { c27_blk::<ARM>(lsp, prp, omp, nc4, c, d, d1); c += 4; }
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
                c25_var4_simd(prp,hpp,hmp,xp,v,lam1,lg);
            }
            while v + 16 <= nv { c32_var8_avx2(prp,hpp,hmp,xp,v,lam1,lg);c32_var8_avx2(prp,hpp,hmp,xp,v+8,lam1,lg);v+=16; }
            while v + 8 <= nv { c32_var8_avx2(prp,hpp,hmp,xp,v,lam1,lg);v+=8; }
            while v + 4 <= nv { var4(prp,hpp,hmp,xp,v,lam1,lg);v+=4; }
            while v < nv { var(prp,hpp,hmp,xp,v,lam1,lg);v+=1; }
            if it % p.refresh == 0 { refresh(prp, hpp, hmp, omp, lsp, nvx, ne); }
            if it == next_check {
                let u = c27_unsat_ls(ls,nc,nc4,&x);
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
                    best.copy_from_slice(x);
                    if u == 0 { break; }
                } else if it - best_it >= p.stall { break; }
                if best_u <= p.stop_u { break; }
            }
        }
    }
    st.sp_it += it;
    st.sp_u = best_u;
    best
}
pub fn c32_solve_large<const ARM:usize,const AVX2_SP:bool>(ch:&Challenge,save:&dyn Fn(&tig_challenges::satisfiability::Solution)->anyhow::Result<()>)->anyhow::Result<C27Status>{
 let mut inst=match c27_build::<ARM>(ch){Ok(i)=>i,Err(s)=>return Ok(s)};let mut p=P::for_instance(ch.num_variables,ch.clauses.len());if ARM!=1 && ch.num_variables==100000 && ch.clauses.len()==420000{p.stall=15;}
 let seed=u64::from_le_bytes(ch.seed[..8].try_into().unwrap());let mut rng=Rng64::seed_from_u64(seed);let mut st=Stats::default();let ls=c25_make_ls(&inst,true);let mut work=C25SpWork::new(&inst);
 for a in 0..p.attempts{st.attempts=a+1;let mut x={
  #[cfg(target_arch="x86_64")]
  {if AVX2_SP && c32_has_avx2(){unsafe{c32_spr_avx2::<ARM>(&inst,&p,&mut rng,&mut st,&ls,&mut work)}}else{c27_spr_with_ls::<ARM>(&inst,&p,&mut rng,&mut st,&ls,&mut work)}}
  #[cfg(not(target_arch="x86_64"))]
  {c27_spr_with_ls::<ARM>(&inst,&p,&mut rng,&mut st,&ls,&mut work)}
 };for v in inst.nv..inst.nvx{x[v]=0;}
  if st.sp_u!=0{c25_finish_occ(&mut inst);}let ok=if st.sp_u==0{true}else{c31_sls_fast_dispatch::<u32,false,true>(&inst,&p,&mut x,&mut rng,&mut st,None,None,0,&mut Near::default())};
  if ok && unsat_count(&inst,&x)==0{let sol=tig_challenges::satisfiability::Solution{variables:x[..inst.nv].iter().map(|&v|v==1).collect()};save(&sol)?;return Ok(C27Status::Solved);}
 }Ok(C27Status::Unsolved)
}

#[cfg(all(test,target_arch="x86_64"))]mod c32_sp_tests{
 use super::*;use rand::{Rng,RngCore,SeedableRng};
 fn bits(a:&[f32],b:&[f32]){assert_eq!(a.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),b.iter().map(|x|x.to_bits()).collect::<Vec<_>>());}
 #[test]fn var8_matches_scalar_and_sse(){if !c32_has_avx2(){return;}let mut r=Rng64::seed_from_u64(100);for _ in 0..1000{let mut a:Vec<f32>=(0..24).map(|_|r.gen_range(0.000001..1.0)).collect();let mut h:Vec<f32>=(0..12).map(|_|r.gen_range(0.0..0.9)).collect();let mut m=h.clone();let mut x=vec![0;12];let(mut b,mut i,mut j,mut y)=(a.clone(),h.clone(),m.clone(),x.clone());let(mut c,mut k,mut l,mut z)=(a.clone(),h.clone(),m.clone(),x.clone());unsafe{c32_var8_avx2(a.as_mut_ptr(),h.as_mut_ptr(),m.as_mut_ptr(),x.as_mut_ptr(),2,0.7,0.11);for v in 2..10{c21_var_scalar(b.as_mut_ptr(),i.as_mut_ptr(),j.as_mut_ptr(),y.as_mut_ptr(),v,0.7,0.11);}c25_var4_simd(c.as_mut_ptr(),k.as_mut_ptr(),l.as_mut_ptr(),z.as_mut_ptr(),2,0.7,0.11);c25_var4_simd(c.as_mut_ptr(),k.as_mut_ptr(),l.as_mut_ptr(),z.as_mut_ptr(),6,0.7,0.11);}bits(&a,&b);bits(&a,&c);bits(&h,&i);bits(&h,&k);bits(&m,&j);bits(&m,&l);assert_eq!(x,y);assert_eq!(x,z);}}
 #[test]fn full_sp_preserves_float_bits_rng_and_solution(){if !c32_has_avx2(){return;}let mut rr=Rng64::seed_from_u64(21);let n=257usize;let clauses:Vec<Vec<i32>>=(0..1100).map(|_|(0..3).map(|_|{let v=rr.gen_range(1..=n)as i32;if rr.gen::<bool>(){v}else{-v}}).collect()).collect();let ch=Challenge{seed:[0;32],num_variables:n,clauses};for arm in 1..=3{let inst=match arm{1=>c27_build::<1>(&ch),2=>c27_build::<2>(&ch),_=>c27_build::<3>(&ch)}.unwrap();let ls=c25_make_ls(&inst,true);for damp in [0.0,0.2]{let mut p=P::for_instance(100000,420000);p.damp=damp;p.maxit=20;p.check=1;p.stop_u=0;let(mut a,mut b,mut c)=(C25SpWork::new(&inst),C25SpWork::new(&inst),C25SpWork::new(&inst));let(mut r,mut s,mut u)=(Rng64::seed_from_u64(42),Rng64::seed_from_u64(42),Rng64::seed_from_u64(42));let(mut x,mut y,mut z)=(Stats::default(),Stats::default(),Stats::default());let (aa,bb,cc)=unsafe{match arm{1=>(c32_spr_avx2::<1>(&inst,&p,&mut r,&mut x,&ls,&mut a),c27_spr_with_ls::<1>(&inst,&p,&mut s,&mut y,&ls,&mut b),c27_spr_scalar::<1>(&inst,&p,&mut u,&mut z,&ls,&mut c)),2=>(c32_spr_avx2::<2>(&inst,&p,&mut r,&mut x,&ls,&mut a),c27_spr_with_ls::<2>(&inst,&p,&mut s,&mut y,&ls,&mut b),c27_spr_scalar::<2>(&inst,&p,&mut u,&mut z,&ls,&mut c)),_=>(c32_spr_avx2::<3>(&inst,&p,&mut r,&mut x,&ls,&mut a),c27_spr_with_ls::<3>(&inst,&p,&mut s,&mut y,&ls,&mut b),c27_spr_scalar::<3>(&inst,&p,&mut u,&mut z,&ls,&mut c))}};assert_eq!(aa,bb);assert_eq!(aa,cc);assert_eq!((x.sp_it,x.sp_u),(y.sp_it,y.sp_u));assert_eq!((x.sp_it,x.sp_u),(z.sp_it,z.sp_u));for(aa,bb,cc)in[(&a.om,&b.om,&c.om),(&a.pr,&b.pr,&c.pr),(&a.hp,&b.hp,&c.hp),(&a.hm,&b.hm,&c.hm)]{bits(aa,bb);bits(aa,cc);}assert_eq!(a.x,b.x);assert_eq!(a.x,c.x);for _ in 0..8{let q=r.next_u64();assert_eq!(q,s.next_u64());assert_eq!(q,u.next_u64());}}}}
 #[test]fn u32_walksat_matches_scalar(){if !c32_has_avx2(){return;}let ch=Challenge{seed:[0;32],num_variables:3,clauses:(0..8).map(|m|(0..3).map(|v|if m&(1<<v)==0{v+1}else{-(v+1)}).collect()).collect()};let inst=build_fast(&ch);let mut p=P::for_instance(5000,21335);p.sp=false;p.max_flips=4096;p.giveup=vec![(1024,usize::MAX,0,0)];let run=|mode|{let mut x=vec![0;inst.nvx];let mut r=Rng64::seed_from_u64(42);let mut st=Stats::default();let mut near=Near::default();let mut slot=None;let ok=match mode{0=>sls_fast_capture::<u32,false>(&inst,&p,&mut x,&mut r,&mut st,None,Some(&mut slot),0,&mut near),1=>c31_sls_fast_dispatch::<u32,false,true>(&inst,&p,&mut x,&mut r,&mut st,None,Some(&mut slot),0,&mut near),2=>unsafe{c31_sls_fast_avx2_bmi::<u32,false,false>(&inst,&p,&mut x,&mut r,&mut st,None,Some(&mut slot),0,&mut near)},_=>unsafe{c31_sls_fast_avx2_bmi::<u32,false,true>(&inst,&p,&mut x,&mut r,&mut st,None,Some(&mut slot),0,&mut near)}};(ok,x,st.flips,slot.unwrap(),r.next_u64())};let(ok,x,f,q,r)=run(0);for mode in 1..=3{let(pass,y,g,z,s)=run(mode);assert_eq!((ok,&x,f,&q.cs,&q.brk,&q.ul,&q.pos),(pass,&y,g,&z.cs,&z.brk,&z.ul,&z.pos));assert_eq!((q.len,q.flips,q.minlen,q.gi,&q.best,r),(z.len,z.flips,z.minlen,z.gi,&z.best,s));}}
}
