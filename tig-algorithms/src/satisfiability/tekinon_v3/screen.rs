use super::Inst;
use rand::{rngs::SmallRng, Rng};

const TINY: f64 = 1e-15;

struct Sp<'a> {
    lits: &'a [u32],
    live: Vec<u8>,
    off: Vec<u32>,
    occ: Vec<u32>,
    eta: Vec<f64>,
    prod: Vec<f64>,
    zc: Vec<u32>,
}

impl<'a> Sp<'a> {
    fn recompute(&mut self) {
        for l in 0..self.prod.len() {
            let mut p = 1.0;
            let mut z = 0;
            for k in self.off[l]..self.off[l + 1] {
                let f = 1.0 - self.eta[self.occ[k as usize] as usize];
                if f < TINY { z += 1 } else { p *= f }
            }
            self.prod[l] = p;
            self.zc[l] = z;
        }
    }

    #[inline(always)]
    fn full(&self, l: usize) -> f64 { if self.zc[l] > 0 { 0.0 } else { self.prod[l] } }

    #[inline(always)]
    fn without(&self, l: usize, e: usize) -> f64 {
        let f = 1.0 - self.eta[e];
        if f < TINY {
            if self.zc[l] > 1 { 0.0 } else { self.prod[l] }
        } else if self.zc[l] > 0 {
            0.0
        } else {
            self.prod[l] / f
        }
    }

    #[inline(always)]
    fn ab(&self, e: usize) -> (f64, f64) {
        let l = self.lits[e] as usize;
        (self.without(l, e), self.full(l ^ 1))
    }

    #[inline(always)]
    fn set(&mut self, e: usize, nw: f64) -> f64 {
        let old = self.eta[e];
        if old == nw { return 0.0; }
        let l = self.lits[e] as usize;
        let fo = 1.0 - old;
        if fo < TINY { self.zc[l] -= 1 } else { self.prod[l] /= fo }
        let fnw = 1.0 - nw;
        if fnw < TINY { self.zc[l] += 1 } else { self.prod[l] *= fnw }
        self.eta[e] = nw;
        (nw - old).abs()
    }

    fn sweep(&mut self) -> f64 {
        let mut dmax = 0.0f64;
        for c in 0..self.lits.len() / 3 {
            let e0 = 3 * c;
            let mut r = [1.0f64; 3];
            for q in 0..3 {
                if self.live[e0 + q] == 1 {
                    let (a, b) = self.ab(e0 + q);
                    let z = a + b - a * b;
                    r[q] = if z <= 0.0 { 0.0 } else { (1.0 - b) * a / z };
                }
            }
            let nw = [r[1] * r[2], r[0] * r[2], r[0] * r[1]];
            for q in 0..3 {
                if self.live[e0 + q] == 1 {
                    let d = self.set(e0 + q, nw[q]);
                    if d > dmax { dmax = d; }
                }
            }
        }
        dmax
    }

    fn complexity(&self, nv: usize) -> f64 {
        let mut s = 0.0;
        for c in 0..self.lits.len() / 3 {
            let mut tot = 1.0;
            let mut pu = 1.0;
            for q in 0..3 {
                let e = 3 * c + q;
                if self.live[e] == 0 { continue; }
                let (a, b) = self.ab(e);
                tot *= a + b - a * b;
                pu *= (1.0 - b) * a;
            }
            s += (tot - pu).max(1e-300).ln();
        }
        for v in 0..nv {
            let n = (self.off[2 * v + 2] - self.off[2 * v]) as f64;
            let pp = self.full(2 * v + 1);
            let pm = self.full(2 * v);
            s -= (n - 1.0) * (pp + pm - pp * pm).max(1e-300).ln();
        }
        s / nv as f64
    }
}

pub fn complexity(inst: &Inst, rng: &mut SmallRng) -> Option<f64> {
    let lits = &inst.lits[..];
    let nl = 2 * inst.nvx;
    let mut deg = vec![0u32; nl];
    for &l in lits { deg[l as usize] += 1; }
    let mut off = vec![0u32; nl + 1];
    for l in 0..nl { off[l + 1] = off[l] + deg[l]; }
    let mut pos = off.clone();
    let mut occ = vec![0u32; lits.len()];
    for (e, &l) in lits.iter().enumerate() {
        occ[pos[l as usize] as usize] = e as u32;
        pos[l as usize] += 1;
    }
    let live: Vec<u8> = lits.iter().map(|&l| ((l >> 1) < inst.nv as u32) as u8).collect();
    let eta: Vec<f64> = live.iter().map(|&b| if b == 1 { rng.gen::<f64>() } else { 0.0 }).collect();
    let mut sp = Sp { lits, live, off, occ, eta, prod: vec![1.0; nl], zc: vec![0; nl] };
    for it in 0..300 {
        if it % 16 == 0 { sp.recompute(); }
        if sp.sweep() < 0.01 { return Some(sp.complexity(inst.nv)); }
    }
    None
}
