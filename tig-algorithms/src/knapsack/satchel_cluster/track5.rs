use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::cmp::Reverse;
use tig_challenges::knapsack::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub window_k: Option<usize>,
    pub ils_rounds: Option<usize>,
    pub core_half_dp: Option<usize>,
    pub back_to_best: Option<usize>,
    pub sparse_contrib: Option<usize>,
}

struct Csr {
    starts: Vec<u32>,
    idx: Vec<u16>,
    val: Vec<u16>,
}

impl Csr {
    fn build(ch: &Challenge) -> Option<Self> {
        let n = ch.num_items;
        if n == 0 || n > u16::MAX as usize { return None; }
        let mut symmetric = true;
        if fast(1 << 23) {
            let mut probe: Vec<usize> = Vec::with_capacity(n / 31 + 1);
            let mut i = 0usize;
            while i < n { probe.push(i); i += 31; }
            'symd: for &i in &probe {
                let row = unsafe { ch.interaction_values.get_unchecked(i) };
                if unsafe { *row.get_unchecked(i) } != 0 { symmetric = false; break 'symd; }
            }
            if symmetric {
                'symr: for j in 0..n {
                    let rowj = unsafe { ch.interaction_values.get_unchecked(j) };
                    for &i in &probe {
                        let a = unsafe { *ch.interaction_values.get_unchecked(i).get_unchecked(j) };
                        let b = unsafe { *rowj.get_unchecked(i) };
                        if a != b { symmetric = false; break 'symr; }
                    }
                }
            }
        } else {
        'sym: for i in (0..n).step_by(31) {
            let row = unsafe { ch.interaction_values.get_unchecked(i) };
            if unsafe { *row.get_unchecked(i) } != 0 { symmetric = false; break 'sym; }
            for j in 0..n {
                let a = unsafe { *row.get_unchecked(j) };
                let b = unsafe { *ch.interaction_values.get_unchecked(j).get_unchecked(i) };
                if a != b { symmetric = false; break 'sym; }
            }
        }
        }
        let mut starts: Vec<u32> = Vec::with_capacity(n + 1);
        let mut nnz: usize = 0;
        starts.push(0);
        let pack = fast(1 << 24) && symmetric;
        let mut packed: Vec<u32> = Vec::new();
        let mut ucnt: Vec<u32> = Vec::new();
        if pack { ucnt = vec![0u32; n]; }
        if symmetric {
            let mut deg: Vec<u32> = vec![0; n];
            let mut bad: u32 = 0;
            for i in 0..n {
                let row = unsafe { ch.interaction_values.get_unchecked(i) };
                if unsafe { *row.get_unchecked(i) } != 0 { return None; }
                let mut di = 0u32;
                if pack {
                    packed.reserve(n - i + 1);
                    let base = packed.len();
                    let mut k = 0usize;
                    unsafe {
                        let rp = row.as_ptr();
                        let dp = deg.as_mut_ptr();
                        let pp = packed.as_mut_ptr().add(base);
                        let mut j = i + 1;
                        while j + 4 <= n {
                            let v0 = *rp.add(j);
                            let v1 = *rp.add(j + 1);
                            let v2 = *rp.add(j + 2);
                            let v3 = *rp.add(j + 3);
                            let n0 = (v0 != 0) as u32;
                            let n1 = (v1 != 0) as u32;
                            let n2 = (v2 != 0) as u32;
                            let n3 = (v3 != 0) as u32;
                            di += n0 + n1 + n2 + n3;
                            *dp.add(j) += n0;
                            *dp.add(j + 1) += n1;
                            *dp.add(j + 2) += n2;
                            *dp.add(j + 3) += n3;
                            bad |= (n0 * (v0 as u32 > u16::MAX as u32) as u32)
                                 | (n1 * (v1 as u32 > u16::MAX as u32) as u32)
                                 | (n2 * (v2 as u32 > u16::MAX as u32) as u32)
                                 | (n3 * (v3 as u32 > u16::MAX as u32) as u32);
                            pp.add(k).write(((j as u32) << 16) | (v0 as u32 & 0xffff)); k += n0 as usize;
                            pp.add(k).write((((j + 1) as u32) << 16) | (v1 as u32 & 0xffff)); k += n1 as usize;
                            pp.add(k).write((((j + 2) as u32) << 16) | (v2 as u32 & 0xffff)); k += n2 as usize;
                            pp.add(k).write((((j + 3) as u32) << 16) | (v3 as u32 & 0xffff)); k += n3 as usize;
                            j += 4;
                        }
                        while j < n {
                            let v = *rp.add(j);
                            let nz = (v != 0) as u32;
                            di += nz;
                            *dp.add(j) += nz;
                            bad |= nz * (v as u32 > u16::MAX as u32) as u32;
                            pp.add(k).write(((j as u32) << 16) | (v as u32 & 0xffff)); k += nz as usize;
                            j += 1;
                        }
                        packed.set_len(base + k);
                    }
                    ucnt[i] = k as u32;
                    deg[i] += di;
                    continue;
                }
                unsafe {
                    let rp = row.as_ptr();
                    let dp = deg.as_mut_ptr();
                    let mut j = i + 1;
                    while j + 4 <= n {
                        let v0 = *rp.add(j);
                        let v1 = *rp.add(j + 1);
                        let v2 = *rp.add(j + 2);
                        let v3 = *rp.add(j + 3);
                        let n0 = (v0 != 0) as u32;
                        let n1 = (v1 != 0) as u32;
                        let n2 = (v2 != 0) as u32;
                        let n3 = (v3 != 0) as u32;
                        di += n0 + n1 + n2 + n3;
                        *dp.add(j) += n0;
                        *dp.add(j + 1) += n1;
                        *dp.add(j + 2) += n2;
                        *dp.add(j + 3) += n3;
                        bad |= (n0 * (v0 as u32 > u16::MAX as u32) as u32)
                             | (n1 * (v1 as u32 > u16::MAX as u32) as u32)
                             | (n2 * (v2 as u32 > u16::MAX as u32) as u32)
                             | (n3 * (v3 as u32 > u16::MAX as u32) as u32);
                        j += 4;
                    }
                    while j < n {
                        let v = *rp.add(j);
                        let nz = (v != 0) as u32;
                        di += nz;
                        *dp.add(j) += nz;
                        bad |= nz * (v as u32 > u16::MAX as u32) as u32;
                        j += 1;
                    }
                }
                deg[i] += di;
            }
            if bad != 0 { return None; }
            for i in 0..n {
                nnz += deg[i] as usize;
                if nnz > u32::MAX as usize { return None; }
                starts.push(nnz as u32);
            }
        } else {
        for i in 0..n {
            let row = unsafe { ch.interaction_values.get_unchecked(i) };
            let mut c = 0usize;
            for j in 0..n {
                let v = unsafe { *row.get_unchecked(j) };
                if v != 0 {
                    if v < 0 || v > u16::MAX as i32 { return None; }
                    c += 1;
                }
            }
            nnz += c;
            if nnz > u32::MAX as usize { return None; }
            starts.push(nnz as u32);
        }
        }
        if nnz * 2 > n.saturating_mul(n) { return None; }
        let mut idx: Vec<u16> = vec![0; nnz];
        let mut val: Vec<u16> = vec![0; nnz];
        if pack {
            let mut cur: Vec<u32> = starts[..n].to_vec();
            let mut p = 0usize;
            for i in 0..n {
                let c = ucnt[i] as usize;
                for t in 0..c {
                    let e = unsafe { *packed.get_unchecked(p + t) };
                    let j = (e >> 16) as usize;
                    let v = (e & 0xffff) as u16;
                    let pi = cur[i] as usize;
                    idx[pi] = j as u16; val[pi] = v; cur[i] += 1;
                    let pj = cur[j] as usize;
                    idx[pj] = i as u16; val[pj] = v; cur[j] += 1;
                }
                p += c;
            }
        } else if symmetric {
            let mut cur: Vec<u32> = starts[..n].to_vec();
            for i in 0..n {
                let row = unsafe { ch.interaction_values.get_unchecked(i) };
                for j in (i + 1)..n {
                    let v = unsafe { *row.get_unchecked(j) };
                    if v != 0 {
                        let pi = cur[i] as usize;
                        idx[pi] = j as u16; val[pi] = v as u16; cur[i] += 1;
                        let pj = cur[j] as usize;
                        idx[pj] = i as u16; val[pj] = v as u16; cur[j] += 1;
                    }
                }
            }
        } else {
            let mut w = 0usize;
            for i in 0..n {
                let row = unsafe { ch.interaction_values.get_unchecked(i) };
                for j in 0..n {
                    let v = unsafe { *row.get_unchecked(j) };
                    if v != 0 { idx[w] = j as u16; val[w] = v as u16; w += 1; }
                }
            }
        }
        Some(Self { starts, idx, val })
    }
}

#[derive(Clone, Copy)]
struct Rng { state: u64 }
impl Rng {
    fn from_seed(seed: &[u8; 32]) -> Self {
        let mut s: u64 = 0x9E3779B97F4A7C15;
        for (i, &b) in seed.iter().enumerate() {
            s ^= (b as u64) << ((i & 7) * 8);
            s = s.rotate_left(7).wrapping_mul(0xBF58476D1CE4E5B9);
        }
        if s == 0 { s = 1; }
        Self { state: s }
    }
    #[inline] fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 7;
        x ^= x >> 9;
        x ^= x << 8;
        self.state = x;
        x
    }
    #[inline] fn next_u32(&mut self) -> u32 { (self.next_u64() >> 32) as u32 }
    #[inline] fn next_f64(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    #[inline] fn next_usize(&mut self, bound: usize) -> usize {
        if bound == 0 { return 0; }
        (self.next_u64() % bound as u64) as usize
    }
}

struct State<'a> {
    ch: &'a Challenge,
    csr: Option<&'a Csr>,
    selected_bit: Vec<bool>,
    contrib: Vec<i32>,
    total_value: i64,
    total_weight: u32,
    solution_hash: u64,
    dp_cache: Vec<i64>,
    choose_cache: Vec<u8>,
}

impl<'a> State<'a> {
    fn new_empty(ch: &'a Challenge) -> Self { Self::new_empty_with(ch, None) }

    fn new_empty_with(ch: &'a Challenge, csr: Option<&'a Csr>) -> Self {
        let n = ch.num_items;
        let mut contrib = vec![0i32; n];
        for i in 0..n { contrib[i] = ch.values[i] as i32; }
        Self {
            ch,
            csr,
            selected_bit: vec![false; n],
            contrib,
            total_value: 0,
            total_weight: 0,
            solution_hash: 0,
            dp_cache: Vec::new(),
            choose_cache: Vec::new(),
        }
    }

    #[inline(always)] fn slack(&self) -> u32 { self.ch.max_weight - self.total_weight }

    #[inline(always)]
    fn add_item(&mut self, i: usize) {
        self.total_value += self.contrib[i] as i64;
        self.total_weight += self.ch.weights[i];
        if let Some(csr) = self.csr {
            let s = unsafe { *csr.starts.get_unchecked(i) } as usize;
            let e = unsafe { *csr.starts.get_unchecked(i + 1) } as usize;
            let contrib_ptr = self.contrib.as_mut_ptr();
            unsafe {
                let ip = csr.idx.as_ptr();
                let vp = csr.val.as_ptr();
                let mut k = s;
                while k + 8 <= e {
                    let t0 = *ip.add(k) as usize;
                    let t1 = *ip.add(k + 1) as usize;
                    let t2 = *ip.add(k + 2) as usize;
                    let t3 = *ip.add(k + 3) as usize;
                    let t4 = *ip.add(k + 4) as usize;
                    let t5 = *ip.add(k + 5) as usize;
                    let t6 = *ip.add(k + 6) as usize;
                    let t7 = *ip.add(k + 7) as usize;
                    let c0 = contrib_ptr.add(t0);
                    let c1 = contrib_ptr.add(t1);
                    let c2 = contrib_ptr.add(t2);
                    let c3 = contrib_ptr.add(t3);
                    let c4 = contrib_ptr.add(t4);
                    let c5 = contrib_ptr.add(t5);
                    let c6 = contrib_ptr.add(t6);
                    let c7 = contrib_ptr.add(t7);
                    *c0 = (*c0).wrapping_add(*vp.add(k) as i32);
                    *c1 = (*c1).wrapping_add(*vp.add(k + 1) as i32);
                    *c2 = (*c2).wrapping_add(*vp.add(k + 2) as i32);
                    *c3 = (*c3).wrapping_add(*vp.add(k + 3) as i32);
                    *c4 = (*c4).wrapping_add(*vp.add(k + 4) as i32);
                    *c5 = (*c5).wrapping_add(*vp.add(k + 5) as i32);
                    *c6 = (*c6).wrapping_add(*vp.add(k + 6) as i32);
                    *c7 = (*c7).wrapping_add(*vp.add(k + 7) as i32);
                    k += 8;
                }
                while k + 4 <= e {
                    let t0 = *ip.add(k) as usize;
                    let t1 = *ip.add(k + 1) as usize;
                    let t2 = *ip.add(k + 2) as usize;
                    let t3 = *ip.add(k + 3) as usize;
                    let c0 = contrib_ptr.add(t0);
                    let c1 = contrib_ptr.add(t1);
                    let c2 = contrib_ptr.add(t2);
                    let c3 = contrib_ptr.add(t3);
                    *c0 = (*c0).wrapping_add(*vp.add(k) as i32);
                    *c1 = (*c1).wrapping_add(*vp.add(k + 1) as i32);
                    *c2 = (*c2).wrapping_add(*vp.add(k + 2) as i32);
                    *c3 = (*c3).wrapping_add(*vp.add(k + 3) as i32);
                    k += 4;
                }
                while k < e {
                    let t = *ip.add(k) as usize;
                    let cp = contrib_ptr.add(t);
                    *cp = (*cp).wrapping_add(*vp.add(k) as i32);
                    k += 1;
                }
            }
            self.solution_hash ^= zobrist_item(i);
            self.selected_bit[i] = true;
            return;
        }
        let n = self.ch.num_items;
        let row_ptr = unsafe { self.ch.interaction_values.get_unchecked(i).as_ptr() };
        let contrib_ptr = self.contrib.as_mut_ptr();
        unsafe {
            let mut k = 0usize;
            while k + 8 <= n {
                let cp = contrib_ptr.add(k);
                let rp = row_ptr.add(k);
                *cp.add(0) = (*cp.add(0)).wrapping_add(*rp.add(0));
                *cp.add(1) = (*cp.add(1)).wrapping_add(*rp.add(1));
                *cp.add(2) = (*cp.add(2)).wrapping_add(*rp.add(2));
                *cp.add(3) = (*cp.add(3)).wrapping_add(*rp.add(3));
                *cp.add(4) = (*cp.add(4)).wrapping_add(*rp.add(4));
                *cp.add(5) = (*cp.add(5)).wrapping_add(*rp.add(5));
                *cp.add(6) = (*cp.add(6)).wrapping_add(*rp.add(6));
                *cp.add(7) = (*cp.add(7)).wrapping_add(*rp.add(7));
                k += 8;
            }
            while k < n {
                let ck = contrib_ptr.add(k);
                *ck = (*ck).wrapping_add(*row_ptr.add(k));
                k += 1;
            }
        }
        self.solution_hash ^= zobrist_item(i);
        self.selected_bit[i] = true;
    }

    #[inline(always)]
    fn remove_item(&mut self, j: usize) {
        self.total_value -= self.contrib[j] as i64;
        self.total_weight -= self.ch.weights[j];
        if let Some(csr) = self.csr {
            let s = unsafe { *csr.starts.get_unchecked(j) } as usize;
            let e = unsafe { *csr.starts.get_unchecked(j + 1) } as usize;
            let contrib_ptr = self.contrib.as_mut_ptr();
            unsafe {
                let ip = csr.idx.as_ptr();
                let vp = csr.val.as_ptr();
                let mut k = s;
                while k + 8 <= e {
                    let t0 = *ip.add(k) as usize;
                    let t1 = *ip.add(k + 1) as usize;
                    let t2 = *ip.add(k + 2) as usize;
                    let t3 = *ip.add(k + 3) as usize;
                    let t4 = *ip.add(k + 4) as usize;
                    let t5 = *ip.add(k + 5) as usize;
                    let t6 = *ip.add(k + 6) as usize;
                    let t7 = *ip.add(k + 7) as usize;
                    let c0 = contrib_ptr.add(t0);
                    let c1 = contrib_ptr.add(t1);
                    let c2 = contrib_ptr.add(t2);
                    let c3 = contrib_ptr.add(t3);
                    let c4 = contrib_ptr.add(t4);
                    let c5 = contrib_ptr.add(t5);
                    let c6 = contrib_ptr.add(t6);
                    let c7 = contrib_ptr.add(t7);
                    *c0 = (*c0).wrapping_sub(*vp.add(k) as i32);
                    *c1 = (*c1).wrapping_sub(*vp.add(k + 1) as i32);
                    *c2 = (*c2).wrapping_sub(*vp.add(k + 2) as i32);
                    *c3 = (*c3).wrapping_sub(*vp.add(k + 3) as i32);
                    *c4 = (*c4).wrapping_sub(*vp.add(k + 4) as i32);
                    *c5 = (*c5).wrapping_sub(*vp.add(k + 5) as i32);
                    *c6 = (*c6).wrapping_sub(*vp.add(k + 6) as i32);
                    *c7 = (*c7).wrapping_sub(*vp.add(k + 7) as i32);
                    k += 8;
                }
                while k + 4 <= e {
                    let t0 = *ip.add(k) as usize;
                    let t1 = *ip.add(k + 1) as usize;
                    let t2 = *ip.add(k + 2) as usize;
                    let t3 = *ip.add(k + 3) as usize;
                    let c0 = contrib_ptr.add(t0);
                    let c1 = contrib_ptr.add(t1);
                    let c2 = contrib_ptr.add(t2);
                    let c3 = contrib_ptr.add(t3);
                    *c0 = (*c0).wrapping_sub(*vp.add(k) as i32);
                    *c1 = (*c1).wrapping_sub(*vp.add(k + 1) as i32);
                    *c2 = (*c2).wrapping_sub(*vp.add(k + 2) as i32);
                    *c3 = (*c3).wrapping_sub(*vp.add(k + 3) as i32);
                    k += 4;
                }
                while k < e {
                    let t = *ip.add(k) as usize;
                    let cp = contrib_ptr.add(t);
                    *cp = (*cp).wrapping_sub(*vp.add(k) as i32);
                    k += 1;
                }
            }
            self.solution_hash ^= zobrist_item(j);
            self.selected_bit[j] = false;
            return;
        }
        let n = self.ch.num_items;
        let row_ptr = unsafe { self.ch.interaction_values.get_unchecked(j).as_ptr() };
        let contrib_ptr = self.contrib.as_mut_ptr();
        unsafe {
            let mut k = 0usize;
            while k + 8 <= n {
                let cp = contrib_ptr.add(k);
                let rp = row_ptr.add(k);
                *cp.add(0) = (*cp.add(0)).wrapping_sub(*rp.add(0));
                *cp.add(1) = (*cp.add(1)).wrapping_sub(*rp.add(1));
                *cp.add(2) = (*cp.add(2)).wrapping_sub(*rp.add(2));
                *cp.add(3) = (*cp.add(3)).wrapping_sub(*rp.add(3));
                *cp.add(4) = (*cp.add(4)).wrapping_sub(*rp.add(4));
                *cp.add(5) = (*cp.add(5)).wrapping_sub(*rp.add(5));
                *cp.add(6) = (*cp.add(6)).wrapping_sub(*rp.add(6));
                *cp.add(7) = (*cp.add(7)).wrapping_sub(*rp.add(7));
                k += 8;
            }
            while k < n {
                let ck = contrib_ptr.add(k);
                *ck = (*ck).wrapping_sub(*row_ptr.add(k));
                k += 1;
            }
        }
        self.solution_hash ^= zobrist_item(j);
        self.selected_bit[j] = false;
    }

    #[inline(always)]
    fn replace_item(&mut self, rm: usize, cand: usize) {
        self.remove_item(rm);
        self.add_item(cand);
    }

    fn selected_items(&self) -> Vec<usize> {
        let n = self.ch.num_items;
        let mut out: Vec<usize> = Vec::with_capacity(n);
        unsafe {
            let op = out.as_mut_ptr();
            let bp = self.selected_bit.as_ptr();
            let mut k = 0usize;
            let mut i = 0usize;
            while i + 4 <= n {
                let b0 = *bp.add(i); op.add(k).write(i); k += b0 as usize;
                let b1 = *bp.add(i + 1); op.add(k).write(i + 1); k += b1 as usize;
                let b2 = *bp.add(i + 2); op.add(k).write(i + 2); k += b2 as usize;
                let b3 = *bp.add(i + 3); op.add(k).write(i + 3); k += b3 as usize;
                i += 4;
            }
            while i < n { let b = *bp.add(i); op.add(k).write(i); k += b as usize; i += 1; }
            out.set_len(k);
        }
        out
    }

    fn clone_solution(&self) -> SolState {
        SolState {
            bits: self.selected_bit.clone(),
            contrib: self.contrib.clone(),
            value: self.total_value,
            weight: self.total_weight,
            solution_hash: self.solution_hash,
        }
    }

    fn restore_solution(&mut self, sol: &SolState) {
        self.selected_bit.clone_from(&sol.bits);
        self.contrib.clone_from(&sol.contrib);
        self.total_value = sol.value;
        self.total_weight = sol.weight;
        self.solution_hash = sol.solution_hash;
    }
}

#[derive(Clone)]
struct SolState {
    bits: Vec<bool>,
    contrib: Vec<i32>,
    value: i64,
    weight: u32,
    solution_hash: u64,
}

#[inline(always)]
fn zobrist_item(i: usize) -> u64 {
    let mut h: u64 = 0x517CC1B727220A95;
    h ^= (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
    h = h.rotate_left(17).wrapping_mul(0xBF58476D1CE4E5B9);
    h
}

struct TabuTable {
    slots: [(u64, u16); 512],
}

impl TabuTable {
    fn new() -> Self {
        Self { slots: [(0, 0); 512] }
    }

    #[inline(always)]
    fn start(h: u64) -> usize {
        let mut x = h;
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51afd7ed558ccd);
        x ^= x >> 33;
        (x as usize) & 511
    }

    #[inline(always)]
    fn contains(&self, h: u64) -> bool {
        let mut p = Self::start(h);
        loop {
            let (key, cnt) = self.slots[p];
            if cnt == 0 { return false; }
            if cnt != u16::MAX && key == h { return true; }
            p = (p + 1) & 511;
        }
    }

    #[inline(always)]
    fn insert(&mut self, h: u64) {
        let mut p = Self::start(h);
        let mut first_tomb = usize::MAX;
        loop {
            let (key, cnt) = self.slots[p];
            if cnt == 0 {
                let q = if first_tomb == usize::MAX { p } else { first_tomb };
                self.slots[q] = (h, 1);
                return;
            }
            if cnt == u16::MAX {
                if first_tomb == usize::MAX { first_tomb = p; }
            } else if key == h {
                self.slots[p].1 = cnt + 1;
                return;
            }
            p = (p + 1) & 511;
        }
    }

    #[inline(always)]
    fn remove(&mut self, h: u64) {
        let mut p = Self::start(h);
        loop {
            let (key, cnt) = self.slots[p];
            if cnt == 0 { return; }
            if cnt != u16::MAX && key == h {
                if cnt > 1 { self.slots[p].1 = cnt - 1; }
                else { self.slots[p].1 = u16::MAX; }
                return;
            }
            p = (p + 1) & 511;
        }
    }
}

fn set_all_selected(state: &mut State, total_interactions: &[i64]) {
    let n = state.ch.num_items;
    let mut tv: i64 = 0;
    let mut tw: u32 = 0;
    let mut zhash: u64 = 0;
    for i in 0..n {
        state.selected_bit[i] = true;
        state.contrib[i] = state.ch.values[i] as i32 + total_interactions[i] as i32;
        tv += state.ch.values[i] as i64;
        tw += state.ch.weights[i];
        zhash ^= zobrist_item(i);
    }
    tv += total_interactions.iter().sum::<i64>() / 2;
    state.total_value = tv;
    state.total_weight = tw;
    state.solution_hash = zhash;
}

#[inline(always)]
fn dw(x: i64, w: i64) -> i64 {
    x / w
}
fn build_greedy_density_from_all(state: &mut State, total_interactions: &[i64]) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    set_all_selected(state, total_interactions);
    if fast(1 << 21) {
        if let Some(mut queue) = exact_density::Density::new(&state.ch.weights, &state.selected_bit) {
            while state.total_weight > cap {
                let worst = queue.pop(&state.contrib, 10, false).unwrap();
                state.remove_item(worst);
            }
        }
    }
    while state.total_weight > cap {
        let mut worst = 0;
        let mut worst_s = i64::MAX;
        for i in 0..n {
            if (unsafe { *state.selected_bit.get_unchecked(i) }) {
                let c = (unsafe { *state.contrib.get_unchecked(i) }) as i64;
                let w = ((unsafe { *state.ch.weights.get_unchecked(i) }) as i64).max(1);
                let s = dw(c * 1000, w);
                if s < worst_s { worst_s = s; worst = i; }
            }
        }
        state.remove_item(worst);
    }
    for _ in 0..2 {
        let mut by_density: Vec<usize> = (0..n).collect();
        let contrib = &state.contrib;
        let weights = &state.ch.weights;
        const LDIV: [i64; 11] = [2520, 2520, 1260, 840, 630, 504, 420, 360, 315, 280, 252];
        let mut end = n;
        if fast(1 << 21) && usize::BITS == 64 && n <= u16::MAX as usize
            && MINW.load(std::sync::atomic::Ordering::Relaxed) >= 1 && exact_sort::compatible() {
            by_density.clear(); by_density.resize(n, 0);
            exact_sort::pack_density(&mut by_density, &contrib[..n], &weights[..n]);
            end = exact_sort::capacity_prefix(&mut by_density,
                |a, b| ((a as i64) >> 16) > ((b as i64) >> 16),
                |key| unsafe { *weights.get_unchecked(key & 0xffff) }, cap, 0);
            for item in &mut by_density { *item &= 0xffff; }
        } else {
        let mut dkey = vec![0i64; n];
        for i in 0..n { dkey[i] = contrib[i] as i64 * LDIV[(weights[i] as usize).clamp(1, 10)]; }
        by_density.sort_unstable_by(|&a, &b| unsafe {
                dkey.get_unchecked(b).cmp(dkey.get_unchecked(a))
            });
        }
        let mut target = vec![false; n];
        let mut rem = cap;
        for &i in &by_density[..end] {
            if (unsafe { *state.ch.weights.get_unchecked(i) }) <= rem { target[i] = true; rem -= (unsafe { *state.ch.weights.get_unchecked(i) }); }
        }
        let mut to_rm = Vec::new();
        let mut to_add = Vec::new();
        for i in 0..n {
            if (unsafe { *state.selected_bit.get_unchecked(i) }) && !target[i] { to_rm.push(i); }
            if !(unsafe { *state.selected_bit.get_unchecked(i) }) && target[i] { to_add.push(i); }
        }
        if to_rm.is_empty() && to_add.is_empty() { break; }
        for &r in &to_rm { state.remove_item(r); }
        for &a in &to_add { state.add_item(a); }
    }
}

fn build_greedy_value(state: &mut State) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&i| std::cmp::Reverse((unsafe { *state.ch.values.get_unchecked(i) })));
    for &i in &order {
        if state.total_weight + (unsafe { *state.ch.weights.get_unchecked(i) }) <= cap { state.add_item(i); }
    }
}

fn build_greedy_hub(state: &mut State, total_interactions: &[i64]) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let mut hub_scores: Vec<(usize, i64)> = (0..n).map(|i| (i, total_interactions[i])).collect();
    hub_scores.sort_unstable_by_key(|&(_, s)| std::cmp::Reverse(s));
    for &(i, _) in &hub_scores {
        if state.total_weight + (unsafe { *state.ch.weights.get_unchecked(i) }) <= cap { state.add_item(i); }
    }
}

fn build_greedy_synergy_weight(state: &mut State, total_interactions: &[i64]) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let nm1 = (n as i64 - 1).max(1);
    let mut scores: Vec<(usize, i64)> = (0..n).map(|i| {
        let avg_syn = total_interactions[i] / nm1;
        let w = ((unsafe { *state.ch.weights.get_unchecked(i) }) as i64).max(1);
        (i, ((unsafe { *state.ch.values.get_unchecked(i) }) as i64 + avg_syn) * 100 / w)
    }).collect();
    scores.sort_unstable_by_key(|&(_, s)| std::cmp::Reverse(s));
    for &(i, _) in &scores {
        if state.total_weight + (unsafe { *state.ch.weights.get_unchecked(i) }) <= cap { state.add_item(i); }
    }
}

fn construct_forward_queue(state: &mut State, mode: usize, rng: &mut Rng) -> bool {
    let n = state.ch.num_items;
    if n == 0 || n > u16::MAX as usize || state.csr.is_none() { return false; }
    if MINW.load(std::sync::atomic::Ordering::Relaxed) < 1 { return false; }
    let mut wmax = 0u32;
    for &w in state.ch.weights.iter() { if w > wmax { wmax = w; } }
    if wmax > 10 { return false; }

    let mut counts = [0u32; 11];
    for i in 0..n {
        counts[state.ch.weights[i] as usize] += (!unsafe { *state.selected_bit.get_unchecked(i) }) as u32;
    }
    let mut capacity = state.slack();
    let mut additions = 0u64;
    for w in 1..=10 {
        let take = counts[w].min(capacity / w as u32);
        capacity -= take * w as u32;
        additions += take as u64;
    }
    let current = state.contrib.iter().copied().max().unwrap_or(0).max(0) as u64;
    let max_edge = MAXEDGE.load(std::sync::atomic::Ordering::Relaxed) as u64;
    let upper = current.saturating_add(additions.saturating_mul(max_edge));
    if upper.saturating_mul(1000).saturating_add(31) > u32::MAX as u64 { return false; }

    match mode {
        2 => construct_forward_q::<2>(state, rng),
        3 => construct_forward_q::<3>(state, rng),
        4 => construct_forward_q::<4>(state, rng),
        5..=usize::MAX => construct_forward_q::<5>(state, rng),
        _ => construct_forward_q::<0>(state, rng),
    }
    true
}

fn construct_forward_q<const MODE: usize>(state: &mut State, rng: &mut Rng) {
    use exact_construct::{Jumps, Queue};
    thread_local! { static JUMPS: Jumps = Jumps::new(); }
    let n = state.ch.num_items;
    #[inline(always)]
    fn score_of<const MODE: usize>(c: i32, w: u32) -> i64 {
        if MODE == 2 { return c as i64; }
        let d = exact_construct::positive_density_narrow(c, w);
        if MODE == 3 { d + w as i64 * 3 } else { d }
    }
    JUMPS.with(|jumps| {
    let mut queue = Queue::new(n);
    {
        let slack0 = state.slack();
        for i in 0..n {
            let c = unsafe { *state.contrib.get_unchecked(i) };
            let w = unsafe { *state.ch.weights.get_unchecked(i) };
            if !unsafe { *state.selected_bit.get_unchecked(i) } && w <= slack0 && c > 0 {
                queue.set(i, score_of::<MODE>(c, w));
            }
        }
    }
    let mut finalists: Vec<(usize, i64)> = Vec::new();
    let g2 = fast(1 << 36);
    let g3 = fast(1 << 37);
    let mut flips: Vec<u32> = if g2 { vec![0u32; n + 1] } else { Vec::new() };
    while state.slack() != 0 && queue.len() != 0 {
        let pick = if MODE != 4 && MODE != 5 {
            queue.best().unwrap()
        } else {
            let mask: u64 = if MODE == 5 { 0x7f } else { 0x1f };
            if g3 { queue.finalists_fast(mask, &mut finalists); } else { queue.finalists(mask, &mut finalists); }
            let mut drawn = 0usize;
            let mut best: Option<usize> = None;
            let mut second: Option<usize> = None;
            let mut best_score = i64::MIN;
            let mut second_score = i64::MIN;
            for &(i, base) in &finalists {
                let rank = queue.rank(i);
                jumps.advance(&mut rng.state, rank - drawn);
                let value = base + (rng.next_u32() as u64 & mask) as i64;
                drawn = rank + 1;
                if value > best_score {
                    second = best; second_score = best_score;
                    best = Some(i); best_score = value;
                } else if value > second_score {
                    second = Some(i); second_score = value;
                }
            }
            jumps.advance(&mut rng.state, queue.len() - drawn);
            if let Some(sec) = second {
                if rng.next_u32() & (if MODE == 5 { 1 } else { 3 }) == 0 { sec } else { best.unwrap() }
            } else { best.unwrap() }
        };
        let old_slack = state.slack();
        let new_slack = old_slack - unsafe { *state.ch.weights.get_unchecked(pick) };
        if new_slack == 0 { state.add_item(pick); break; }
        state.total_value += unsafe { *state.contrib.get_unchecked(pick) } as i64;
        state.total_weight += unsafe { *state.ch.weights.get_unchecked(pick) };
        unsafe { *state.selected_bit.get_unchecked_mut(pick) = true; }
        state.solution_hash ^= zobrist_item(pick);
        queue.remove(pick);
        if new_slack < 10 {
            for i in 0..n {
                let w = unsafe { *state.ch.weights.get_unchecked(i) };
                if w > new_slack && w <= old_slack { queue.remove(i); }
            }
        }
        let csr = state.csr.unwrap();
        let start = unsafe { *csr.starts.get_unchecked(pick) } as usize;
        let end = unsafe { *csr.starts.get_unchecked(pick + 1) } as usize;
        let ids = csr.idx.as_ptr();
        let vals = csr.val.as_ptr();
        let selected = state.selected_bit.as_ptr();
        let weights = state.ch.weights.as_ptr();
        let contributions = state.contrib.as_mut_ptr();
        macro_rules! update {
            ($p:expr, $check_weight:expr) => {{
                unsafe {
                    let i = *ids.add($p) as usize;
                    let cp = contributions.add(i);
                    *cp = (*cp).wrapping_add(*vals.add($p) as i32);
                    let c = *cp;
                    let w = *weights.add(i);
                    let active = (!*selected.add(i)) & (!$check_weight || w <= new_slack) & (c > 0);
                    let value = score_of::<MODE>(c.max(1), w);
                    queue.assign_small(i, value as u32, active);
                }
            }};
        }
        if g2 {
            let fp = flips.as_mut_ptr();
            let mut nf = 0usize;
            macro_rules! update2 {
                ($p:expr, $check_weight:expr) => {{
                    unsafe {
                        let i = *ids.add($p) as usize;
                        let cp = contributions.add(i);
                        *cp = (*cp).wrapping_add(*vals.add($p) as i32);
                        let c = *cp;
                        let w = *weights.add(i);
                        let active = (!*selected.add(i)) & (!$check_weight || w <= new_slack) & (c > 0);
                        let value = score_of::<MODE>(c.max(1), w);
                        queue.assign_small_defer(i, value as u32, active, fp, &mut nf);
                    }
                }};
            }
            if new_slack >= 10 {
                let mut p = start;
                let full = start + (end - start) / 8 * 8;
                while p < full {
                    update2!(p, false); update2!(p + 1, false); update2!(p + 2, false); update2!(p + 3, false);
                    update2!(p + 4, false); update2!(p + 5, false); update2!(p + 6, false); update2!(p + 7, false);
                    p += 8;
                }
                while p < end { update2!(p, false); p += 1; }
            } else {
                for p in start..end { update2!(p, true); }
            }
            queue.apply_flips(&flips[..nf]);
        } else if new_slack >= 10 {
            let mut p = start;
            let full = start + (end - start) / 8 * 8;
            while p < full {
                update!(p, false); update!(p + 1, false); update!(p + 2, false); update!(p + 3, false);
                update!(p + 4, false); update!(p + 5, false); update!(p + 6, false); update!(p + 7, false);
                p += 8;
            }
            while p < end { update!(p, false); p += 1; }
        } else {
            for p in start..end { update!(p, true); }
        }
    }
    })
}

fn construct_forward_incremental(state: &mut State, mode: usize, rng: &mut Rng) {
    if fast(1 << 14) && construct_forward_queue(state, mode, rng) { return; }
    let n = state.ch.num_items;
    match mode {
        2 => loop {
            let slack = state.slack();
            if slack == 0 { break; }
            let mut best_i: Option<usize> = None;
            let mut best_s: i64 = i64::MIN;
            for i in 0..n {
                if unsafe { *state.selected_bit.get_unchecked(i) } { continue; }
                if unsafe { *state.ch.weights.get_unchecked(i) } > slack { continue; }
                let s = unsafe { *state.contrib.get_unchecked(i) } as i64;
                if s <= 0 { continue; }
                if s > best_s {
                    best_s = s;
                    best_i = Some(i);
                }
            }
            if let Some(i) = best_i { state.add_item(i); } else { break; }
        },
        3 => loop {
            let slack = state.slack();
            if slack == 0 { break; }
            let mut best_i: Option<usize> = None;
            let mut best_s: i64 = i64::MIN;
            for i in 0..n {
                if unsafe { *state.selected_bit.get_unchecked(i) } { continue; }
                if unsafe { *state.ch.weights.get_unchecked(i) } > slack { continue; }
                let c = unsafe { *state.contrib.get_unchecked(i) } as i64;
                if c <= 0 { continue; }
                let w = (unsafe { *state.ch.weights.get_unchecked(i) } as i64).max(1);
                let s = dw(c * 1000, w) + ((unsafe { *state.ch.weights.get_unchecked(i) }) as i64) * 3;
                if s > best_s {
                    best_s = s;
                    best_i = Some(i);
                }
            }
            if let Some(i) = best_i { state.add_item(i); } else { break; }
        },
        4 => {
            let mut cand: Vec<usize> = Vec::with_capacity(n);
            for i in 0..n {
                if !unsafe { *state.selected_bit.get_unchecked(i) } { cand.push(i); }
            }
            loop {
                let slack = state.slack();
                if slack == 0 { break; }
                let mut best_i: Option<usize> = None;
                let mut best_p: usize = 0;
                let mut best_s: i64 = i64::MIN;
                let mut second_i: Option<usize> = None;
                let mut second_p: usize = 0;
                let mut second_s: i64 = i64::MIN;
                const DMAG: [u128; 11] = [
                    18446744073709551617, 18446744073709551617, 9223372036854775809,
                    6148914691236517206, 4611686018427387905, 3689348814741910324,
                    3074457345618258603, 2635249153387078803, 2305843009213693953,
                    2049638230412172402, 1844674407370955162];
                if fast(2048) {
                for (p, &i) in cand.iter().enumerate() {
                    if unsafe { *state.ch.weights.get_unchecked(i) } > slack { continue; }
                    let c = unsafe { *state.contrib.get_unchecked(i) } as i64;
                    if c <= 0 { continue; }
                    let wi = (unsafe { *state.ch.weights.get_unchecked(i) } as usize).clamp(1, 10);
                    let s = ((((c * 1000) as u128).wrapping_mul(unsafe { *DMAG.get_unchecked(wi) })) >> 64) as i64
                        + (rng.next_u32() & 0x1F) as i64;
                    if s > best_s {
                        second_s = best_s;
                        second_i = best_i;
                        second_p = best_p;
                        best_s = s;
                        best_i = Some(i);
                        best_p = p;
                    } else if s > second_s {
                        second_s = s;
                        second_i = Some(i);
                        second_p = p;
                    }
                }
                } else {
                for (p, &i) in cand.iter().enumerate() {
                    if unsafe { *state.ch.weights.get_unchecked(i) } > slack { continue; }
                    let c = unsafe { *state.contrib.get_unchecked(i) } as i64;
                    if c <= 0 { continue; }
                    let w = (unsafe { *state.ch.weights.get_unchecked(i) } as i64).max(1);
                    let s = dw(c * 1000, w) + (rng.next_u32() & 0x1F) as i64;
                    if s > best_s {
                        second_s = best_s;
                        second_i = best_i;
                        second_p = best_p;
                        best_s = s;
                        best_i = Some(i);
                        best_p = p;
                    } else if s > second_s {
                        second_s = s;
                        second_i = Some(i);
                        second_p = p;
                    }
                }
                }
                let (pick, pick_p) = if second_i.is_some() {
                    if (rng.next_u32() & 3) == 0 { (second_i, second_p) } else { (best_i, best_p) }
                } else {
                    (best_i, best_p)
                };
                if let Some(i) = pick { state.add_item(i); cand.remove(pick_p); } else { break; }
            }
        },
        5.. => loop {
            let slack = state.slack();
            if slack == 0 { break; }
            let mut best_i: Option<usize> = None;
            let mut best_s: i64 = i64::MIN;
            let mut second_i: Option<usize> = None;
            let mut second_s: i64 = i64::MIN;
            for i in 0..n {
                if unsafe { *state.selected_bit.get_unchecked(i) } { continue; }
                if unsafe { *state.ch.weights.get_unchecked(i) } > slack { continue; }
                let c = unsafe { *state.contrib.get_unchecked(i) } as i64;
                if c <= 0 { continue; }
                let w = (unsafe { *state.ch.weights.get_unchecked(i) } as i64).max(1);
                let s = dw(c * 1000, w) + (rng.next_u32() & 0x7F) as i64;
                if s > best_s {
                    second_s = best_s;
                    second_i = best_i;
                    best_s = s;
                    best_i = Some(i);
                } else if s > second_s {
                    second_s = s;
                    second_i = Some(i);
                }
            }
            let pick = if second_i.is_some() {
                if (rng.next_u32() & 1) == 0 { second_i } else { best_i }
            } else {
                best_i
            };
            if let Some(i) = pick { state.add_item(i); } else { break; }
        },
        _ => loop {
            let slack = state.slack();
            if slack == 0 { break; }
            let mut best_i: Option<usize> = None;
            let mut best_s: i64 = i64::MIN;
            for i in 0..n {
                if unsafe { *state.selected_bit.get_unchecked(i) } { continue; }
                if unsafe { *state.ch.weights.get_unchecked(i) } > slack { continue; }
                let c = unsafe { *state.contrib.get_unchecked(i) } as i64;
                if c <= 0 { continue; }
                let w = (unsafe { *state.ch.weights.get_unchecked(i) } as i64).max(1);
                let s = dw(c * 1000, w);
                if s > best_s {
                    best_s = s;
                    best_i = Some(i);
                }
            }
            if let Some(i) = best_i { state.add_item(i); } else { break; }
        },
    }
}

fn build_synergy_supernodes(state: &mut State, total_interactions: &[i64], rng: &mut Rng) {
    let g5 = fast(1 << 39) && state.total_weight == 0 && state.solution_hash == 0
        && state.selected_bit.iter().all(|&b| !b);
    if g5 {
        if let Some(pre) = SUPER_PREFIX.with(|c| c.borrow().clone()) {
            state.restore_solution(&pre);
            construct_forward_incremental(state, 4, rng);
            return;
        }
    }
    let n = state.ch.num_items;
    let cap = state.ch.max_weight as u64;
    let mut seeds: Vec<(usize, i64)> = Vec::with_capacity(24);
    for i in 0..n {
        let score = total_interactions[i];
        let pos = seeds.iter().position(|&(_, prior)| score > prior)
            .unwrap_or(seeds.len());
        if pos < 24 {
            seeds.insert(pos, (i, score));
            if seeds.len() > 24 { seeds.pop(); }
        }
    }

    let mut assigned = vec![false; n];
    let mut components: Vec<(Vec<usize>, u64, i64)> = Vec::new();
    for &(seed, _) in &seeds {
        if assigned[seed] || (unsafe { *state.ch.weights.get_unchecked(seed) }) as u64 > cap { continue; }
        let mut neighbors: Vec<(usize, i32)> = Vec::with_capacity(4);
        for item in 0..n {
            if item == seed || assigned[item] { continue; }
            let interaction = (unsafe { *state.ch.interaction_values.get_unchecked(seed).get_unchecked(item) });
            if interaction <= 0 { continue; }
            let pos = neighbors.iter().position(|&(_, prior)| interaction > prior)
                .unwrap_or(neighbors.len());
            if pos < 4 {
                neighbors.insert(pos, (item, interaction));
                if neighbors.len() > 4 { neighbors.pop(); }
            }
        }
        if neighbors.is_empty() { continue; }

        let mut items = vec![seed];
        let mut weight = (unsafe { *state.ch.weights.get_unchecked(seed) }) as u64;
        for &(item, _) in &neighbors {
            let item_weight = (unsafe { *state.ch.weights.get_unchecked(item) }) as u64;
            if weight + item_weight <= cap {
                items.push(item);
                weight += item_weight;
            }
        }
        if items.len() < 2 { continue; }

        let mut internal_value = 0i64;
        for ai in 0..items.len() {
            let item = items[ai];
            internal_value += (unsafe { *state.ch.values.get_unchecked(item) }) as i64;
            for bi in 0..ai {
                internal_value += state.ch.interaction_values[item][items[bi]] as i64;
            }
        }
        for &item in &items { assigned[item] = true; }
        if internal_value > 0 {
            components.push((items, weight, internal_value));
        }
    }

    components.sort_unstable_by(|a, b| {
        ((b.2 as i128) * (a.1 as i128)).cmp(&((a.2 as i128) * (b.1 as i128)))
    });
    for (items, weight, _) in components {
        if state.total_weight as u64 + weight > cap { continue; }
        let mut delta = 0i64;
        for ai in 0..items.len() {
            let item = items[ai];
            delta += (unsafe { *state.contrib.get_unchecked(item) }) as i64;
            for bi in 0..ai {
                delta += state.ch.interaction_values[item][items[bi]] as i64;
            }
        }
        if delta > 0 {
            for item in items { state.add_item(item); }
        }
    }
    if g5 { SUPER_PREFIX.with(|c| *c.borrow_mut() = Some(state.clone_solution())); }
    construct_forward_incremental(state, 4, rng);
}

fn build_hub_pair_kth(state: &mut State, k: usize) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let mut pairs: Vec<(i32, usize, usize)> = Vec::new();
    for i in 0..n {
        for j in (i+1)..n {
            if (unsafe { *state.ch.weights.get_unchecked(i) }) + (unsafe { *state.ch.weights.get_unchecked(j) }) <= cap {
                pairs.push(((unsafe { *state.ch.interaction_values.get_unchecked(i).get_unchecked(j) }), i, j));
            }
        }
    }
    pairs.sort_unstable_by_key(|&(s, _, _)| std::cmp::Reverse(s));
    let mut used = Vec::new();
    let mut count = 0;
    for &(_, pi, pj) in &pairs {
        if used.contains(&pi) || used.contains(&pj) { continue; }
        if count == k {
            state.add_item(pi);
            state.add_item(pj);
            break;
        }
        used.push(pi);
        used.push(pj);
        count += 1;
    }
    loop {
        let slack = state.slack();
        if slack == 0 { break; }
        let mut best_i: Option<usize> = None;
        let mut best_s: i64 = 0;
        for i in 0..n {
            if (unsafe { *state.selected_bit.get_unchecked(i) }) { continue; }
            if (unsafe { *state.ch.weights.get_unchecked(i) }) > slack { continue; }
            let c = (unsafe { *state.contrib.get_unchecked(i) }) as i64;
            if c <= 0 { continue; }
            let w = ((unsafe { *state.ch.weights.get_unchecked(i) }) as i64).max(1);
            let s = dw(c * 1000, w);
            if s > best_s { best_s = s; best_i = Some(i); }
        }
        if let Some(i) = best_i { state.add_item(i); } else { break; }
    }
}

static MAXW: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static MAXEDGE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static FASTM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(63);
static MINW: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
#[inline(always)] fn fast(bit: usize) -> bool { FASTM.load(std::sync::atomic::Ordering::Relaxed) & bit != 0 }

thread_local! {
    static VND_PATHS: std::cell::RefCell<exact_paths::Paths> =
        const { std::cell::RefCell::new(exact_paths::Paths::new_const()) };
}
thread_local! {
    static RAND_MEMBER: std::cell::RefCell<Option<SolState>> = const { std::cell::RefCell::new(None) };
    static SUPER_PREFIX: std::cell::RefCell<Option<SolState>> = const { std::cell::RefCell::new(None) };
}
thread_local! {
    static DP_MEMO: std::cell::RefCell<Vec<Option<(usize, SolState, SolState)>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

fn dp_refinement_hp(state: &mut State, core_half: usize) {
    if !fast(1 << 15) { dp_refinement_uncached(state, core_half); return; }
    let slot = if fast(1 << 47) {
        ((state.solution_hash ^ (core_half as u64).wrapping_mul(0x9e3779b97f4a7c15))
            .wrapping_mul(0xbf58476d1ce4e5b9) >> 55) as usize
    } else {
        exact_memo::address(&state.selected_bit, core_half)
    };
    let hit = DP_MEMO.with(|cell| {
        let cache = cell.borrow();
        if let Some(Some((parameter, input, output))) = cache.get(slot) {
            if *parameter == core_half
                && input.value == state.total_value
                && input.weight == state.total_weight
                && input.solution_hash == state.solution_hash
                && input.bits == state.selected_bit
                && input.contrib == state.contrib
            {
                state.restore_solution(output);
                return true;
            }
        }
        false
    });
    if hit { return; }
    let input = state.clone_solution();
    dp_refinement_uncached(state, core_half);
    let output = state.clone_solution();
    DP_MEMO.with(|cell| {
        let mut cache = cell.borrow_mut();
        if cache.is_empty() { cache.resize_with(512, || None); }
        cache[slot] = Some((core_half, input, output));
    });
}

fn dp_refinement_uncached(state: &mut State, core_half: usize) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let contrib = &state.contrib;
    let weights = &state.ch.weights;

    const LDIV: [i64; 11] = [2520, 2520, 1260, 840, 630, 504, 420, 360, 315, 280, 252];
    let mut by_density: Vec<usize> = Vec::with_capacity(n);
    let mut dkey: Vec<i64> = Vec::with_capacity(n);
    let mut min_w = u32::MAX;
    if fast(1 << 16) && usize::BITS == 64 && n <= u16::MAX as usize
        && MINW.load(std::sync::atomic::Ordering::Relaxed) >= 1
        && exact_sort::compatible() {
        by_density.resize(n, 0);
        exact_sort::pack_density(&mut by_density, &contrib[..n], &weights[..n]);
        exact_sort::capacity_core(
            &mut by_density,
            |a, b| ((a as i64) >> 16) > ((b as i64) >> 16),
            |key| unsafe { *weights.get_unchecked(key & 0xffff) },
            cap,
            core_half,
            MAXW.load(std::sync::atomic::Ordering::Relaxed),
        );
        for item in &mut by_density { *item &= 0xffff; }
        min_w = MINW.load(std::sync::atomic::Ordering::Relaxed);
    } else if fast(16) {
        unsafe {
            let bp = by_density.as_mut_ptr();
            let dp = dkey.as_mut_ptr();
            let cp = contrib.as_ptr();
            let wp = weights.as_ptr();
            let mut i = 0usize;
            while i + 4 <= n {
                let w0 = *wp.add(i);
                let w1 = *wp.add(i + 1);
                let w2 = *wp.add(i + 2);
                let w3 = *wp.add(i + 3);
                bp.add(i).write(i);
                bp.add(i + 1).write(i + 1);
                bp.add(i + 2).write(i + 2);
                bp.add(i + 3).write(i + 3);
                dp.add(i).write(*cp.add(i) as i64 * *LDIV.get_unchecked((w0 as usize).clamp(1, 10)));
                dp.add(i + 1).write(*cp.add(i + 1) as i64 * *LDIV.get_unchecked((w1 as usize).clamp(1, 10)));
                dp.add(i + 2).write(*cp.add(i + 2) as i64 * *LDIV.get_unchecked((w2 as usize).clamp(1, 10)));
                dp.add(i + 3).write(*cp.add(i + 3) as i64 * *LDIV.get_unchecked((w3 as usize).clamp(1, 10)));
                min_w = min_w.min(w0).min(w1).min(w2).min(w3);
                i += 4;
            }
            while i < n {
                let w0 = *wp.add(i);
                bp.add(i).write(i);
                dp.add(i).write(*cp.add(i) as i64 * *LDIV.get_unchecked((w0 as usize).clamp(1, 10)));
                min_w = min_w.min(w0);
                i += 1;
            }
            by_density.set_len(n);
            dkey.set_len(n);
        }
    } else {
        for i in 0..n { by_density.push(i); }
        for i in 0..n { dkey.push(contrib[i] as i64 * LDIV[(weights[i] as usize).clamp(1, 10)]); }
        min_w = 0;
    }
    if !dkey.is_empty() {
        by_density.sort_unstable_by(|&a, &b| unsafe {
                    dkey.get_unchecked(b).cmp(dkey.get_unchecked(a))
                });
    }
    let mut idx_last_inserted = 0usize;
    let mut idx_first_rejected = n;
    let mut rem = cap;
    let early = fast(16) && min_w >= 1;
    let mut start_idx = 0usize;
    if fast(1 << 45) {
        let bp = by_density.as_ptr();
        let wp = weights.as_ptr();
        let mut t = 0usize;
        while t + 8 <= n {
            let s8 = unsafe {
                *wp.add(*bp.add(t)) + *wp.add(*bp.add(t + 1)) + *wp.add(*bp.add(t + 2))
                    + *wp.add(*bp.add(t + 3)) + *wp.add(*bp.add(t + 4)) + *wp.add(*bp.add(t + 5))
                    + *wp.add(*bp.add(t + 6)) + *wp.add(*bp.add(t + 7))
            };
            if s8 >= rem { break; }
            rem -= s8;
            idx_last_inserted = t + 7;
            t += 8;
        }
        start_idx = t;
    }
    for (idx, &i) in by_density.iter().enumerate().skip(start_idx) {
        let w = weights[i];
        if w <= rem {
            rem -= w;
            idx_last_inserted = idx;
            if early && rem == 0 {
                if idx_first_rejected == n && idx + 1 < n { idx_first_rejected = idx + 1; }
                break;
            }
        } else if idx_first_rejected == n {
            idx_first_rejected = idx;
            if early && rem == 0 { break; }
        }
    }

    let left = idx_first_rejected.saturating_sub(core_half + 1);
    let right = (idx_last_inserted + core_half + 1).min(n);
    let locked: Vec<usize> = if fast(1 << 29) { Vec::new() } else { by_density[..left].to_vec() };
    let core: Vec<usize> = by_density[left..right].to_vec();

    let used_locked: u64 = by_density[..left].iter().map(|&i| weights[i] as u64).sum();
    let rem_cap = (cap as u64).saturating_sub(used_locked) as usize;
    let myk = core.len();
    if myk == 0 || rem_cap == 0 { return; }

    let mut total_core_weight: usize = 0;
    let mut total_pos_weight: usize = 0;
    let mut all_pos_fit = true;
    for &it in &core {
        let wt = weights[it] as usize;
        total_core_weight += wt;
        if contrib[it] > 0 {
            total_pos_weight += wt;
            if total_pos_weight > rem_cap { all_pos_fit = false; }
        }
    }

    let mut picks: Vec<usize> = Vec::with_capacity(core.len());
    if all_pos_fit {
        for &it in &core { if contrib[it] > 0 { picks.push(it); } }
        return finish_sel(state, &by_density, left, &picks, &locked);
    }
    {
        let myw = rem_cap.min(total_core_weight);
        let dp_size = myw + 1;
        let mut exact_dp_rows = Vec::<i32>::new();
        if fast(1 << 17) {
            if let Some(w) = exact_dp::fill(&core, weights, contrib, myw,
                                            &mut state.choose_cache, &mut exact_dp_rows) {
                let mut w_star = w;
                for t in (0..myk).rev() {
                    let it = core[t];
                    let wt = weights[it] as usize;
                    if wt <= w_star && state.choose_cache[t * dp_size + w_star] == 1 {
                        picks.push(it);
                        w_star -= wt;
                    }
                }
                return finish_sel(state, &by_density, left, &picks, &locked);
            }
        }
        let choose_size = myk * dp_size;
        if state.dp_cache.len() < dp_size { state.dp_cache.resize(dp_size, i64::MIN / 4); }
        if state.choose_cache.len() < choose_size { state.choose_cache.resize(choose_size, 0); }
        let init_val = i64::MIN / 4;
        for v in &mut state.dp_cache[..dp_size] { *v = init_val; }
        state.dp_cache[0] = 0;
        state.choose_cache[..choose_size].fill(0);

        let mut w_hi: usize = 0;
        for (t, &it) in core.iter().enumerate() {
            let wt = weights[it] as usize;
            if wt > myw { continue; }
            let val = contrib[it] as i64;
            let new_hi = (w_hi + wt).min(myw);
            for w in (wt..=new_hi).rev() {
                let cand = state.dp_cache[w - wt] + val;
                if cand > state.dp_cache[w] {
                    state.dp_cache[w] = cand;
                    state.choose_cache[t * dp_size + w] = 1;
                }
            }
            w_hi = new_hi;
        }

        let mut w_star = (0..=myw).max_by_key(|&w| state.dp_cache[w]).unwrap_or(0);
        for t in (0..myk).rev() {
            let it = core[t];
            let wt = weights[it] as usize;
            if wt <= w_star && state.choose_cache[t * dp_size + w_star] == 1 {
                picks.push(it);
                w_star -= wt;
            }
        }
    }

    finish_sel(state, &by_density, left, &picks, &locked)
}

#[inline]
fn finish_sel(state: &mut State, by_density: &[usize], left: usize, picks: &[usize], locked: &[usize]) {
    if fast(1 << 29) {
        let n = state.ch.num_items;
        let mut mask = vec![0u8; n];
        let mut to_add: Vec<u32> = Vec::with_capacity(left + picks.len() + 1);
        unsafe {
            let mp = mask.as_mut_ptr();
            let lp = by_density.as_ptr();
            let mut t = 0usize;
            while t + 4 <= left {
                *mp.add(*lp.add(t)) = 1;
                *mp.add(*lp.add(t + 1)) = 1;
                *mp.add(*lp.add(t + 2)) = 1;
                *mp.add(*lp.add(t + 3)) = 1;
                t += 4;
            }
            while t < left { *mp.add(*lp.add(t)) = 1; t += 1; }
            for &i in picks { *mp.add(i) = 1; }
        }
        finish_dp_mask(state, &mask, &mut to_add)
    } else {
        let mut sel: Vec<usize> = locked.to_vec();
        sel.extend_from_slice(picks);
        sel.sort_unstable();
        finish_dp(state, &sel)
    }
}

fn finish_dp_mask(state: &mut State, mask: &[u8], to_add: &mut Vec<u32>) {
    let n = state.ch.num_items;
    if fast(1 << 44) {
        let mut to_rm: Vec<u32> = Vec::with_capacity(n + 1);
        to_add.clear();
        to_add.reserve(n + 1);
        let (mut ka, mut kr) = (0usize, 0usize);
        unsafe {
            let ap = to_add.as_mut_ptr();
            let rp = to_rm.as_mut_ptr();
            let sp = state.selected_bit.as_ptr();
            let mp = mask.as_ptr();
            macro_rules! slot { ($i:expr) => {{
                let i = $i;
                let sel = *sp.add(i);
                let m = *mp.add(i) != 0;
                ap.add(ka).write(i as u32);
                ka += (m & !sel) as usize;
                rp.add(kr).write(i as u32);
                kr += (sel & !m) as usize;
            }}; }
            let mut i = 0usize;
            while i + 8 <= n {
                slot!(i); slot!(i + 1); slot!(i + 2); slot!(i + 3);
                slot!(i + 4); slot!(i + 5); slot!(i + 6); slot!(i + 7);
                i += 8;
            }
            while i < n { slot!(i); i += 1; }
            to_add.set_len(ka);
            to_rm.set_len(kr);
        }
        for t in 0..kr { state.remove_item(unsafe { *to_rm.get_unchecked(t) } as usize); }
        for t in 0..ka { state.add_item(unsafe { *to_add.get_unchecked(t) } as usize); }
        return;
    }
    let mut k = 0usize;
    unsafe {
        let ap = to_add.as_mut_ptr();
        for i in 0..n {
            let sel = *state.selected_bit.get_unchecked(i);
            let m = *mask.get_unchecked(i) != 0;
            ap.add(k).write(i as u32);
            k += (m & !sel) as usize;
            if sel & !m { state.remove_item(i); }
        }
        to_add.set_len(k);
    }
    for t in 0..k { state.add_item(unsafe { *to_add.get_unchecked(t) } as usize); }
}

fn finish_dp(state: &mut State, target_sel: &[usize]) {
    let n = state.ch.num_items;
    let mut j = 0;
    let m = target_sel.len();
    if m > 0 {
        let tp = target_sel.as_ptr();
        for i in 0..n {
            let jj = if j < m { j } else { m - 1 };
            let in_target = (j < m) & ((unsafe { *tp.add(jj) }) == i);
            j += in_target as usize;
            if (unsafe { *state.selected_bit.get_unchecked(i) }) && !in_target {
                state.remove_item(i);
            }
        }
    } else {
        for i in 0..n {
            if unsafe { *state.selected_bit.get_unchecked(i) } { state.remove_item(i); }
        }
    }
    for &i in target_sel {
        if !(unsafe { *state.selected_bit.get_unchecked(i) }) {
            state.add_item(i);
        }
    }
}

fn apply_best_add(state: &mut State, unselected: &[usize]) -> bool {
    let slack = state.slack();
    if slack == 0 { return false; }
    let mut best_i: Option<usize> = None;
    let mut best_d: i32 = 0;
    for &i in unselected {
        if (unsafe { *state.ch.weights.get_unchecked(i) }) > slack { continue; }
        let d = (unsafe { *state.contrib.get_unchecked(i) });
        if d > best_d { best_d = d; best_i = Some(i); }
    }
    if let Some(i) = best_i { state.add_item(i); true } else { false }
}

fn apply_best_swap_1_1(state: &mut State, selected: &[usize], unselected: &[usize]) -> bool {
    let slack = state.slack();
    let mut best: Option<(usize, usize, i32)> = None;
    for &rm in selected {
        let w_rm = (unsafe { *state.ch.weights.get_unchecked(rm) });
        let max_w = w_rm + slack;
        for &cand in unselected {
            let wc = (unsafe { *state.ch.weights.get_unchecked(cand) });
            if wc > max_w { continue; }
            let delta = (unsafe { *state.contrib.get_unchecked(cand) }) - (unsafe { *state.contrib.get_unchecked(rm) })
                - (unsafe { *state.ch.interaction_values.get_unchecked(cand).get_unchecked(rm) });
            if delta > 0 && best.map_or(true, |(_, _, bd)| delta > bd) {
                best = Some((cand, rm, delta));
            }
        }
    }
    if let Some((cand, rm, _)) = best { state.replace_item(rm, cand); true } else { false }
}

fn apply_pair_add(state: &mut State, unselected: &[usize]) -> bool {
    let slack = state.slack();
    if slack < 2 { return false; }
    let m = unselected.len();

    let mut best_delta: i64 = 0;
    let mut best_pair: Option<(usize, usize)> = None;
    for ai in 0..m {
        let a = unselected[ai];
        let wa = (unsafe { *state.ch.weights.get_unchecked(a) });
        if wa >= slack { continue; }
        let ca = (unsafe { *state.contrib.get_unchecked(a) }) as i64;
        for bi in (ai+1)..m {
            let b = unselected[bi];
            let wb = (unsafe { *state.ch.weights.get_unchecked(b) });
            if wb >= slack || wa + wb > slack { continue; }
            let delta = ca + (unsafe { *state.contrib.get_unchecked(b) }) as i64 + (unsafe { *state.ch.interaction_values.get_unchecked(a).get_unchecked(b) }) as i64;
            if delta > best_delta {
                best_delta = delta;
                best_pair = Some((a, b));
            }
        }
    }
    if let Some((a, b)) = best_pair {
        state.add_item(a);
        state.add_item(b);
        true
    } else { false }
}

fn apply_chain_move(state: &mut State) -> bool {
    let n = state.ch.num_items;
    let sel: Vec<usize> = (0..n).filter(|&i| (unsafe { *state.selected_bit.get_unchecked(i) })).collect();
    let unsel: Vec<usize> = (0..n).filter(|&i| !(unsafe { *state.selected_bit.get_unchecked(i) })).collect();
    let cap = state.ch.max_weight;

    let mut best_delta: i64 = 0;
    let mut best_move: Option<(usize, usize, usize)> = None;

    for &rm in &sel {
        let w_rm = (unsafe { *state.ch.weights.get_unchecked(rm) }) as i64;
        let c_rm = (unsafe { *state.contrib.get_unchecked(rm) }) as i64;
        let budget = state.slack() as i64 + w_rm;

        for ui in 0..unsel.len() {
            let a1 = unsel[ui];
            let w_a1 = (unsafe { *state.ch.weights.get_unchecked(a1) }) as i64;
            if w_a1 >= budget { continue; }
            let c_a1 = (unsafe { *state.contrib.get_unchecked(a1) }) as i64 - (unsafe { *state.ch.interaction_values.get_unchecked(a1).get_unchecked(rm) }) as i64;

            for uj in (ui+1)..unsel.len() {
                let a2 = unsel[uj];
                let w_a2 = (unsafe { *state.ch.weights.get_unchecked(a2) }) as i64;
                if w_a1 + w_a2 > budget { continue; }

                let c_a2 = (unsafe { *state.contrib.get_unchecked(a2) }) as i64 - (unsafe { *state.ch.interaction_values.get_unchecked(a2).get_unchecked(rm) }) as i64;
                let syn = (unsafe { *state.ch.interaction_values.get_unchecked(a1).get_unchecked(a2) }) as i64;
                let delta = c_a1 + c_a2 + syn - c_rm;

                if delta > best_delta {
                    let new_w = state.total_weight as i64 - w_rm + w_a1 + w_a2;
                    if new_w <= cap as i64 {
                        best_delta = delta;
                        best_move = Some((rm, a1, a2));
                    }
                }
            }
        }
    }

    if let Some((rm, a1, a2)) = best_move {
        state.remove_item(rm);
        state.add_item(a1);
        state.add_item(a2);
        true
    } else { false }
}

fn apply_reverse_chain(state: &mut State) -> bool {
    let n = state.ch.num_items;
    let sel: Vec<usize> = (0..n).filter(|&i| (unsafe { *state.selected_bit.get_unchecked(i) })).collect();
    let unsel: Vec<usize> = (0..n).filter(|&i| !(unsafe { *state.selected_bit.get_unchecked(i) })).collect();
    let cap = state.ch.max_weight;

    let mut best_delta: i64 = 0;
    let mut best_move: Option<(usize, usize, usize)> = None;

    for &add in &unsel {
        let w_add = (unsafe { *state.ch.weights.get_unchecked(add) }) as i64;
        let c_add = (unsafe { *state.contrib.get_unchecked(add) }) as i64;

        for si in 0..sel.len() {
            let r1 = sel[si];
            let w_r1 = (unsafe { *state.ch.weights.get_unchecked(r1) }) as i64;
            let c_r1 = (unsafe { *state.contrib.get_unchecked(r1) }) as i64;
            let c_add_r1 = (unsafe { *state.ch.interaction_values.get_unchecked(add).get_unchecked(r1) }) as i64;

            for sj in (si+1)..sel.len() {
                let r2 = sel[sj];
                let w_r2 = (unsafe { *state.ch.weights.get_unchecked(r2) }) as i64;
                let freed = w_r1 + w_r2;
                let new_w = state.total_weight as i64 - freed + w_add;
                if new_w > cap as i64 || new_w < 0 { continue; }

                let c_r2 = (unsafe { *state.contrib.get_unchecked(r2) }) as i64;
                let syn_r1_r2 = (unsafe { *state.ch.interaction_values.get_unchecked(r1).get_unchecked(r2) }) as i64;
                let c_add_r2 = (unsafe { *state.ch.interaction_values.get_unchecked(add).get_unchecked(r2) }) as i64;

                let lost = c_r1 + c_r2 - syn_r1_r2;
                let gained = c_add - c_add_r1 - c_add_r2;
                let delta = gained - lost;

                if delta > best_delta {
                    best_delta = delta;
                    best_move = Some((r1, r2, add));
                }
            }
        }
    }

    if let Some((r1, r2, add)) = best_move {
        state.remove_item(r1);
        state.remove_item(r2);
        state.add_item(add);
        true
    } else { false }
}

fn apply_swap_2_2_bounded(state: &mut State, k: usize) -> bool {
    let n = state.ch.num_items;
    let mut sel_ranked: Vec<(usize, i32)> = (0..n)
        .filter(|&i| (unsafe { *state.selected_bit.get_unchecked(i) }))
        .map(|i| (i, (unsafe { *state.contrib.get_unchecked(i) })))
        .collect();
    sel_ranked.sort_unstable_by_key(|&(_, c)| c);
    sel_ranked.truncate(k);

    let mut unsel_ranked: Vec<(usize, i32)> = (0..n)
        .filter(|&i| !(unsafe { *state.selected_bit.get_unchecked(i) }))
        .map(|i| (i, (unsafe { *state.contrib.get_unchecked(i) })))
        .collect();
    unsel_ranked.sort_unstable_by_key(|&(_, c)| std::cmp::Reverse(c));
    unsel_ranked.truncate(k);

    let cap = state.ch.max_weight;
    let mut best_delta: i64 = 0;
    let mut best_move: Option<(usize, usize, usize, usize)> = None;

    for si in 0..sel_ranked.len() {
        let r1 = sel_ranked[si].0;
        let w_r1 = (unsafe { *state.ch.weights.get_unchecked(r1) }) as i64;
        let c_r1 = (unsafe { *state.contrib.get_unchecked(r1) }) as i64;
        for sj in (si+1)..sel_ranked.len() {
            let r2 = sel_ranked[sj].0;
            let w_r2 = (unsafe { *state.ch.weights.get_unchecked(r2) }) as i64;
            let c_r2 = (unsafe { *state.contrib.get_unchecked(r2) }) as i64;
            let freed_weight = w_r1 + w_r2;
            let removed_syn = (unsafe { *state.ch.interaction_values.get_unchecked(r1).get_unchecked(r2) }) as i64;
            let lost = c_r1 + c_r2 - removed_syn;
            let budget = state.slack() as i64 + freed_weight;

            for ui in 0..unsel_ranked.len() {
                let a1 = unsel_ranked[ui].0;
                let w_a1 = (unsafe { *state.ch.weights.get_unchecked(a1) }) as i64;
                if w_a1 > budget { continue; }
                let c_a1 = (unsafe { *state.contrib.get_unchecked(a1) }) as i64
                    - (unsafe { *state.ch.interaction_values.get_unchecked(a1).get_unchecked(r1) }) as i64
                    - (unsafe { *state.ch.interaction_values.get_unchecked(a1).get_unchecked(r2) }) as i64;
                for uj in (ui+1)..unsel_ranked.len() {
                    let a2 = unsel_ranked[uj].0;
                    let w_a2 = (unsafe { *state.ch.weights.get_unchecked(a2) }) as i64;
                    if w_a1 + w_a2 > budget { continue; }
                    let c_a2 = (unsafe { *state.contrib.get_unchecked(a2) }) as i64
                        - (unsafe { *state.ch.interaction_values.get_unchecked(a2).get_unchecked(r1) }) as i64
                        - (unsafe { *state.ch.interaction_values.get_unchecked(a2).get_unchecked(r2) }) as i64;
                    let added_syn = (unsafe { *state.ch.interaction_values.get_unchecked(a1).get_unchecked(a2) }) as i64;
                    let delta = c_a1 + c_a2 + added_syn - lost;
                    if delta > best_delta {
                        let new_weight = state.total_weight as i64 - freed_weight + w_a1 + w_a2;
                        if new_weight <= cap as i64 {
                            best_delta = delta;
                            best_move = Some((r1, r2, a1, a2));
                        }
                    }
                }
            }
        }
    }
    if let Some((r1, r2, a1, a2)) = best_move {
        state.remove_item(r1);
        state.remove_item(r2);
        state.add_item(a1);
        state.add_item(a2);
        true
    } else { false }
}

fn local_search_vnd_fast(state: &mut State) {
    let n = state.ch.num_items;
    let mut selected_buf: Vec<usize> = Vec::with_capacity(n);
    let mut unselected_buf: Vec<usize> = Vec::with_capacity(n);
    for _ in 0..80 {
        selected_buf.clear();
        unselected_buf.clear();
        for i in 0..n {
            if (unsafe { *state.selected_bit.get_unchecked(i) }) { selected_buf.push(i); } else { unselected_buf.push(i); }
        }
        if apply_best_add(state, &unselected_buf) { continue; }
        if apply_best_swap_1_1(state, &selected_buf, &unselected_buf) { continue; }
        break;
    }
}

fn ils_vnd(state: &mut State) {
    local_search_vnd_fast(state);
}

fn local_search_vnd_heavy(state: &mut State) {
    let n = state.ch.num_items;
    let mut selected_buf: Vec<usize> = Vec::with_capacity(n);
    let mut unselected_buf: Vec<usize> = Vec::with_capacity(n);
    for _ in 0..297 {
        selected_buf.clear();
        unselected_buf.clear();
        for i in 0..n {
            if (unsafe { *state.selected_bit.get_unchecked(i) }) { selected_buf.push(i); } else { unselected_buf.push(i); }
        }
        if apply_best_add(state, &unselected_buf) { continue; }
        if apply_best_swap_1_1(state, &selected_buf, &unselected_buf) { continue; }
        if apply_pair_add(state, &unselected_buf) { continue; }
        if apply_swap_2_2_bounded(state, 25) { continue; }
        if apply_chain_move(state) { continue; }
        if apply_reverse_chain(state) { continue; }
        break;
    }
}

fn crossover_frequency(population: &[SolState], ch: &Challenge, rng: &mut Rng) -> Vec<bool> {
    let n = ch.num_items;
    let pop_size = population.len();
    let mut freq = vec![0usize; n];
    for sol in population {
        for i in 0..n { if sol.bits[i] { freq[i] += 1; } }
    }
    let threshold = (pop_size * 3) / 4;
    let mut child_bits = vec![false; n];
    let mut child_weight: u32 = 0;
    let mut consensus: Vec<usize> = Vec::new();
    let mut exploratory: Vec<usize> = Vec::new();
    for i in 0..n {
        if freq[i] > threshold { consensus.push(i); }
        else if freq[i] > 0 { exploratory.push(i); }
    }
    for &i in &consensus {
        if child_weight + (unsafe { *ch.weights.get_unchecked(i) }) <= ch.max_weight {
            child_bits[i] = true;
            child_weight += (unsafe { *ch.weights.get_unchecked(i) });
        }
    }
    for &i in &exploratory {
        if rng.next_u32() % 2 == 0 && child_weight + (unsafe { *ch.weights.get_unchecked(i) }) <= ch.max_weight {
            child_bits[i] = true;
            child_weight += (unsafe { *ch.weights.get_unchecked(i) });
        }
    }
    child_bits
}

fn crossover_uniform(sol_a: &SolState, sol_b: &SolState, ch: &Challenge, rng: &mut Rng) -> Vec<bool> {
    let n = ch.num_items;
    let mut bits = vec![false; n];
    let mut weight: u32 = 0;
    for i in 0..n {
        if sol_a.bits[i] && sol_b.bits[i] {
            if weight + (unsafe { *ch.weights.get_unchecked(i) }) <= ch.max_weight {
                bits[i] = true;
                weight += (unsafe { *ch.weights.get_unchecked(i) });
            }
        }
    }
    for i in 0..n {
        if bits[i] { continue; }
        if sol_a.bits[i] || sol_b.bits[i] {
            if rng.next_u32() % 2 == 0 && weight + (unsafe { *ch.weights.get_unchecked(i) }) <= ch.max_weight {
                bits[i] = true;
                weight += (unsafe { *ch.weights.get_unchecked(i) });
            }
        }
    }
    bits
}

fn crossover_linkage(state: &mut State, sol_a: &SolState, sol_b: &SolState) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let mut shared: Vec<usize> = Vec::new();
    let mut inherited: Vec<(usize, i64)> = Vec::new();

    for i in 0..n {
        if sol_a.bits[i] && sol_b.bits[i] {
            shared.push(i);
        } else if sol_a.bits[i] || sol_b.bits[i] {
            let weight = ((unsafe { *state.ch.weights.get_unchecked(i) }) as i64).max(1);
            inherited.push((i, (unsafe { *state.ch.values.get_unchecked(i) }) as i64 * 1000 / weight));
        }
    }

    let from_parent = fast(1 << 41) && sol_a.weight <= cap && state.total_weight == 0
        && state.solution_hash == 0 && state.selected_bit.iter().all(|&b| !b)
        && sol_a.bits.len() == n && sol_b.bits.len() == n;
    if from_parent {
        state.restore_solution(sol_a);
        for i in 0..n {
            if sol_a.bits[i] && !sol_b.bits[i] { state.remove_item(i); }
        }
    } else {
    shared.sort_unstable_by(|&a, &b| {
        let va = (unsafe { *state.ch.values.get_unchecked(a) }) as i64;
        let vb = (unsafe { *state.ch.values.get_unchecked(b) }) as i64;
        let wa = ((unsafe { *state.ch.weights.get_unchecked(a) }) as i64).max(1);
        let wb = ((unsafe { *state.ch.weights.get_unchecked(b) }) as i64).max(1);
        (vb * wa).cmp(&(va * wb))
    });
    for i in shared {
        if (unsafe { *state.ch.weights.get_unchecked(i) }) <= state.slack() {
            state.add_item(i);
        }
    }
    }

    inherited.sort_unstable_by_key(|&(_, score)| Reverse(score));
    inherited.truncate(56);

    let mut edges: Vec<(usize, usize, i64)> = Vec::with_capacity(40);
    for ai in 0..inherited.len() {
        let a = inherited[ai].0;
        for bi in (ai + 1)..inherited.len() {
            let b = inherited[bi].0;
            let together_a = sol_a.bits[a] && sol_a.bits[b];
            let together_b = sol_b.bits[a] && sol_b.bits[b];
            if !together_a && !together_b {
                continue;
            }
            let interaction = (unsafe { *state.ch.interaction_values.get_unchecked(a).get_unchecked(b) }) as i64;
            if interaction <= 0 {
                continue;
            }
            let weight = ((unsafe { *state.ch.weights.get_unchecked(a) }) as i64 + (unsafe { *state.ch.weights.get_unchecked(b) }) as i64).max(1);
            let score = ((unsafe { *state.ch.values.get_unchecked(a) }) as i64 + (unsafe { *state.ch.values.get_unchecked(b) }) as i64 + interaction) * 1000 / weight;
            let pos = edges.iter().position(|&(_, _, prior)| score > prior).unwrap_or(edges.len());
            if pos < 40 {
                edges.insert(pos, (a, b, score));
                if edges.len() > 40 {
                    edges.pop();
                }
            }
        }
    }

    for (a, b, _) in edges {
        if (unsafe { *state.selected_bit.get_unchecked(a) }) || (unsafe { *state.selected_bit.get_unchecked(b) }) {
            continue;
        }
        let pair_weight = (unsafe { *state.ch.weights.get_unchecked(a) }) as u64 + (unsafe { *state.ch.weights.get_unchecked(b) }) as u64;
        if pair_weight <= state.slack() as u64 {
            state.add_item(a);
            state.add_item(b);
        }
    }

    let fill4 = fast(1 << 43) && MINW.load(std::sync::atomic::Ordering::Relaxed) >= 1
        && MAXW.load(std::sync::atomic::Ordering::Relaxed) <= 10;
    loop {
        let slack = state.slack();
        let mut best: Option<(usize, i64)> = None;
        if fill4 {
            let sb = state.selected_bit.as_ptr();
            let wp = state.ch.weights.as_ptr();
            let cp = state.contrib.as_ptr();
            macro_rules! key { ($i:expr) => {{
                let (sel, w, c) = unsafe { (*sb.add($i), *wp.add($i), *cp.add($i)) };
                let el = (!sel) & (w <= slack) & (c > 0);
                let sc = exact_construct::positive_density(c.max(1), w.clamp(1, 10));
                if el { sc } else { i64::MIN }
            }}; }
            let mut bi = usize::MAX;
            let mut bsc = i64::MIN;
            let mut i = 0usize;
            while i + 4 <= n {
                let k0 = key!(i);
                let k1 = key!(i + 1);
                let k2 = key!(i + 2);
                let k3 = key!(i + 3);
                if (k0 > bsc) | (k1 > bsc) | (k2 > bsc) | (k3 > bsc) {
                    if k0 > bsc { bsc = k0; bi = i; }
                    if k1 > bsc { bsc = k1; bi = i + 1; }
                    if k2 > bsc { bsc = k2; bi = i + 2; }
                    if k3 > bsc { bsc = k3; bi = i + 3; }
                }
                i += 4;
            }
            while i < n {
                let k = key!(i);
                if k > bsc { bsc = k; bi = i; }
                i += 1;
            }
            if bi != usize::MAX { best = Some((bi, bsc)); }
        } else {
        for i in 0..n {
            if (unsafe { *state.selected_bit.get_unchecked(i) }) || (unsafe { *state.ch.weights.get_unchecked(i) }) > slack {
                continue;
            }
            let weight = ((unsafe { *state.ch.weights.get_unchecked(i) }) as i64).max(1);
            let score = (unsafe { *state.contrib.get_unchecked(i) }) as i64 * 1000 / weight;
            if (unsafe { *state.contrib.get_unchecked(i) }) > 0 && best.map_or(true, |(_, prior)| score > prior) {
                best = Some((i, score));
            }
        }
        }
        if let Some((i, _)) = best {
            state.add_item(i);
        } else {
            break;
        }
    }

    let _ = cap;
}

fn crossover_residual_modules(
    state: &mut State,
    backbone: &SolState,
    donor: &SolState,
) {
    let n = state.ch.num_items;
    let target_weight = backbone.weight as u64 * 82 / 100;
    let mut shared: Vec<usize> = Vec::new();
    let mut private: Vec<usize> = Vec::new();
    for item in 0..n {
        if backbone.bits[item] {
            if donor.bits[item] {
                shared.push(item);
            } else {
                private.push(item);
            }
        }
    }
    let density_cmp = |a: &usize, b: &usize| {
        let wa = (state.ch.weights[*a] as i64).max(1);
        let wb = (state.ch.weights[*b] as i64).max(1);
        (backbone.contrib[*b] as i64 * wa).cmp(&(backbone.contrib[*a] as i64 * wb))
    };
    let from_parent = fast(1 << 42) && backbone.weight <= state.ch.max_weight
        && state.total_weight == 0 && state.solution_hash == 0
        && state.selected_bit.iter().all(|&b| !b) && backbone.bits.len() == n;
    if !from_parent { shared.sort_unstable_by(density_cmp); }
    private.sort_unstable_by(density_cmp);

    let mut protected = vec![false; n];
    if from_parent {
        state.restore_solution(backbone);
        for &item in &private { state.remove_item(item); }
        for &item in &shared { protected[item] = true; }
    } else {
    for item in shared {
        state.add_item(item);
        protected[item] = true;
    }
    }
    for item in private {
        if state.total_weight as u64 >= target_weight {
            break;
        }
        state.add_item(item);
        protected[item] = true;
    }

    let mut donor_items: Vec<usize> = (0..n)
        .filter(|&item| donor.bits[item] && !backbone.bits[item])
        .collect();
    donor_items.sort_unstable_by(|&a, &b| {
        let wa = ((unsafe { *state.ch.weights.get_unchecked(a) }) as i64).max(1);
        let wb = ((unsafe { *state.ch.weights.get_unchecked(b) }) as i64).max(1);
        (donor.contrib[b] as i64 * wa).cmp(&(donor.contrib[a] as i64 * wb))
    });
    donor_items.truncate(48);

    let mut modules: Vec<(Vec<usize>, i64)> = Vec::with_capacity(96);
    for &item in &donor_items {
        let weight = ((unsafe { *state.ch.weights.get_unchecked(item) }) as i64).max(1);
        modules.push((vec![item], donor.contrib[item] as i64 * 1000 / weight));
    }
    for ai in 0..donor_items.len() {
        let a = donor_items[ai];
        for bi in (ai + 1)..donor_items.len() {
            let b = donor_items[bi];
            let interaction = (unsafe { *state.ch.interaction_values.get_unchecked(a).get_unchecked(b) }) as i64;
            if interaction <= 0 {
                continue;
            }
            let weight = ((unsafe { *state.ch.weights.get_unchecked(a) }) as i64 + (unsafe { *state.ch.weights.get_unchecked(b) }) as i64).max(1);
            let score = (donor.contrib[a] as i64 + donor.contrib[b] as i64 + interaction) * 1000 / weight;
            modules.push((vec![a, b], score));
        }
    }
    modules.sort_unstable_by_key(|(_, score)| Reverse(*score));
    modules.truncate(56);

    let mut weak: Vec<usize> = (0..n).filter(|&item| protected[item]).collect();
    weak.sort_unstable_by_key(|&item| (unsafe { *state.contrib.get_unchecked(item) }));
    weak.truncate(12);

    for _ in 0..48 {
        let mut best: Option<(usize, i64, Option<usize>)> = None;
        for (module_idx, (items, _)) in modules.iter().enumerate() {
            if items.iter().any(|&item| (unsafe { *state.selected_bit.get_unchecked(item) })) {
                continue;
            }
            let module_weight: u64 = items.iter().map(|&item| (unsafe { *state.ch.weights.get_unchecked(item) }) as u64).sum();
            let mut gain = 0i64;
            for &item in items {
                gain += (unsafe { *state.contrib.get_unchecked(item) }) as i64;
            }
            if items.len() == 2 {
                gain += state.ch.interaction_values[items[0]][items[1]] as i64;
            }
            if state.total_weight as u64 + module_weight <= state.ch.max_weight as u64 {
                if gain > 0 && best.map_or(true, |(_, prior, _)| gain > prior) {
                    best = Some((module_idx, gain, None));
                }
                continue;
            }
            for &removed in &weak {
                if !(unsafe { *state.selected_bit.get_unchecked(removed) })
                    || state.total_weight as u64 + module_weight
                        > state.ch.max_weight as u64 + (unsafe { *state.ch.weights.get_unchecked(removed) }) as u64
                {
                    continue;
                }
                let mut replacement_gain = gain - (unsafe { *state.contrib.get_unchecked(removed) }) as i64;
                for &item in items {
                    replacement_gain -= (unsafe { *state.ch.interaction_values.get_unchecked(item).get_unchecked(removed) }) as i64;
                }
                if replacement_gain > 0
                    && best.map_or(true, |(_, prior, _)| replacement_gain > prior)
                {
                    best = Some((module_idx, replacement_gain, Some(removed)));
                }
            }
        }
        let Some((module_idx, _, removed)) = best else { break; };
        if let Some(item) = removed {
            state.remove_item(item);
        }
        let items = modules[module_idx].0.clone();
        for item in items {
            state.add_item(item);
        }
    }
}

fn set_state_from_bits(state: &mut State, bits: &[bool]) {
    let n = state.ch.num_items;
    for i in (0..n).rev() {
        if (unsafe { *state.selected_bit.get_unchecked(i) }) && !bits[i] { state.remove_item(i); }
    }
    for i in 0..n {
        if bits[i] && !(unsafe { *state.selected_bit.get_unchecked(i) }) { state.add_item(i); }
    }
}

fn local_search_vnd_windowed(state: &mut State, window_k: usize) {
    let n = state.ch.num_items;
    let mut unused_r: Vec<(usize, i64, i64)> = Vec::with_capacity(n);
    let mut used_r: Vec<(usize, i64, i64)> = Vec::with_capacity(n);
    let mut best_unused: Vec<usize> = Vec::with_capacity(window_k.min(n));
    let mut worst_used: Vec<usize> = Vec::with_capacity(window_k.min(n));
    for _ in 0..80 {
        unused_r.clear();
        used_r.clear();
        best_unused.clear();
        worst_used.clear();
        for i in 0..n {
            let c = (unsafe { *state.contrib.get_unchecked(i) }) as i64;
            let w = ((unsafe { *state.ch.weights.get_unchecked(i) }) as i64).max(1);
            if (unsafe { *state.selected_bit.get_unchecked(i) }) { used_r.push((i, c, w)); } else { unused_r.push((i, c, w)); }
        }
        let ku = window_k.min(unused_r.len());
        if ku > 0 && ku < unused_r.len() {
            unused_r.select_nth_unstable_by(ku - 1, |a, b| {
                ((b.1 as i128) * (a.2 as i128)).cmp(&((a.1 as i128) * (b.2 as i128)))
            });
        }
        let ks = window_k.min(used_r.len());
        if ks > 0 && ks < used_r.len() {
            used_r.select_nth_unstable_by(ks - 1, |a, b| {
                ((a.1 as i128) * (b.2 as i128)).cmp(&((b.1 as i128) * (a.2 as i128)))
            });
        }
        for x in &unused_r[..ku] { best_unused.push(x.0); }
        for x in &used_r[..ks] { worst_used.push(x.0); }
        let improved = false;

        let slack = state.slack();
        if slack > 0 {
            let mut ba: Option<(usize, i32)> = None;
            for &c in &best_unused {
                if (unsafe { *state.ch.weights.get_unchecked(c) }) > slack { continue; }
                let d = (unsafe { *state.contrib.get_unchecked(c) });
                if d > 0 && ba.map_or(true, |(_, bd)| d > bd) { ba = Some((c, d)); }
            }
            if let Some((c, _)) = ba { state.add_item(c); continue; }
        }

        {
            let mut bs: Option<(usize, usize, i32)> = None;
            let mut cmax = [i32::MIN; 11];
            if fast(1 << 34) {
                for &c in &best_unused {
                    let cw = ((unsafe { *state.ch.weights.get_unchecked(c) }) as usize).clamp(1, 10);
                    let v = unsafe { *state.contrib.get_unchecked(c) };
                    let slot = unsafe { cmax.get_unchecked_mut(cw) };
                    *slot = if v > *slot { v } else { *slot };
                }
            } else {
            for &c in &best_unused {
                let cw = ((unsafe { *state.ch.weights.get_unchecked(c) }) as usize).clamp(1, 10);
                if (unsafe { *state.contrib.get_unchecked(c) }) > cmax[cw] { cmax[cw] = (unsafe { *state.contrib.get_unchecked(c) }); }
            }
            }
            for w in 1..=10 { if cmax[w - 1] > cmax[w] { cmax[w] = cmax[w - 1]; } }
            let mut rmin = [i32::MAX; 12];
            if fast(1 << 34) {
                for &rm in &worst_used {
                    let rw = ((unsafe { *state.ch.weights.get_unchecked(rm) }) as usize).clamp(1, 10);
                    let v = unsafe { *state.contrib.get_unchecked(rm) };
                    let slot = unsafe { rmin.get_unchecked_mut(rw) };
                    *slot = if v < *slot { v } else { *slot };
                }
            } else {
            for &rm in &worst_used {
                let rw = ((unsafe { *state.ch.weights.get_unchecked(rm) }) as usize).clamp(1, 10);
                if (unsafe { *state.contrib.get_unchecked(rm) }) < rmin[rw] { rmin[rw] = (unsafe { *state.contrib.get_unchecked(rm) }); }
            }
            }
            for w in (1..=10).rev() { if rmin[w + 1] < rmin[w] { rmin[w] = rmin[w + 1]; } }
            let slack_c = state.slack() as i32;
            let mut cand_live: Vec<usize> = Vec::with_capacity(best_unused.len());
            for &c in &best_unused {
                let wc = (unsafe { *state.ch.weights.get_unchecked(c) }) as i32;
                let lo = (wc - slack_c).max(1).min(10) as usize;
                if (unsafe { *state.contrib.get_unchecked(c) }) > rmin[lo] { cand_live.push(c); }
            }
            for &rm in &worst_used {
                let max_w = (unsafe { *state.ch.weights.get_unchecked(rm) }) + state.slack();
                if cmax[(max_w as usize).clamp(1, 10)] <= (unsafe { *state.contrib.get_unchecked(rm) }) { continue; }
                let row_rm = unsafe { state.ch.interaction_values.get_unchecked(rm) };
                for &c in &cand_live {
                    if (unsafe { *state.ch.weights.get_unchecked(c) }) > max_w { continue; }
                    let d = (unsafe { *state.contrib.get_unchecked(c) }) - (unsafe { *state.contrib.get_unchecked(rm) }) - unsafe { *row_rm.get_unchecked(c) };
                    if d > 0 && bs.map_or(true, |(_, _, bd)| d > bd) { bs = Some((c, rm, d)); }
                }
            }
            if let Some((c, rm, _)) = bs { state.replace_item(rm, c); continue; }
        }

        let slack = state.slack();
        if slack >= 2 {
            let fits: Vec<usize> = best_unused.iter().copied().filter(|&i| (unsafe { *state.ch.weights.get_unchecked(i) }) < slack).collect();
            let m = fits.len();
            if m >= 2 {
                let mut bp: Option<(usize, usize, i64)> = None;
                for ai in 0..m {
                    let a = fits[ai];
                    let wa = (unsafe { *state.ch.weights.get_unchecked(a) });
                    let ca = (unsafe { *state.contrib.get_unchecked(a) }) as i64;
                    for bi in (ai+1)..m {
                        let b = fits[bi];
                        if wa + (unsafe { *state.ch.weights.get_unchecked(b) }) > slack { continue; }
                        let d = ca + (unsafe { *state.contrib.get_unchecked(b) }) as i64 + (unsafe { *state.ch.interaction_values.get_unchecked(a).get_unchecked(b) }) as i64;
                        if d > 0 && bp.map_or(true, |(_, _, bd)| d > bd) { bp = Some((a, b, d)); }
                    }
                }
                if let Some((a, b, _)) = bp { state.add_item(a); state.add_item(b); continue; }
            }
        }

        if !improved { break; }
    }
}

fn local_search_vnd_windowed_deep(state: &mut State, window_k: usize) {
    let n = state.ch.num_items;
    let mut unused_r: Vec<(usize, i64, i64)> = Vec::with_capacity(n);
    let mut used_r: Vec<(usize, i64, i64)> = Vec::with_capacity(n);
    let mut best_unused: Vec<usize> = Vec::with_capacity(window_k.min(n));
    let mut worst_used: Vec<usize> = Vec::with_capacity(window_k.min(n));
    for _ in 0..120 {
        unused_r.clear();
        used_r.clear();
        best_unused.clear();
        worst_used.clear();
        for i in 0..n {
            let c = (unsafe { *state.contrib.get_unchecked(i) }) as i64;
            let w = ((unsafe { *state.ch.weights.get_unchecked(i) }) as i64).max(1);
            if (unsafe { *state.selected_bit.get_unchecked(i) }) { used_r.push((i, c, w)); } else { unused_r.push((i, c, w)); }
        }
        let ku = window_k.min(unused_r.len());
        if ku > 0 && ku < unused_r.len() {
            unused_r.select_nth_unstable_by(ku - 1, |a, b| {
                ((b.1 as i128) * (a.2 as i128)).cmp(&((a.1 as i128) * (b.2 as i128)))
            });
        }
        let ks = window_k.min(used_r.len());
        if ks > 0 && ks < used_r.len() {
            used_r.select_nth_unstable_by(ks - 1, |a, b| {
                ((a.1 as i128) * (b.2 as i128)).cmp(&((b.1 as i128) * (a.2 as i128)))
            });
        }
        for x in &unused_r[..ku] { best_unused.push(x.0); }
        for x in &used_r[..ks] { worst_used.push(x.0); }
        let improved = false;

        let slack = state.slack();
        if slack > 0 {
            let mut ba: Option<(usize, i32)> = None;
            for &c in &best_unused {
                if (unsafe { *state.ch.weights.get_unchecked(c) }) > slack { continue; }
                let d = (unsafe { *state.contrib.get_unchecked(c) });
                if d > 0 && ba.map_or(true, |(_, bd)| d > bd) { ba = Some((c, d)); }
            }
            if let Some((c, _)) = ba { state.add_item(c); continue; }
        }

        {
            let mut bs: Option<(usize, usize, i32)> = None;
            let mut cmax = [i32::MIN; 11];
            if fast(1 << 34) {
                for &c in &best_unused {
                    let cw = ((unsafe { *state.ch.weights.get_unchecked(c) }) as usize).clamp(1, 10);
                    let v = unsafe { *state.contrib.get_unchecked(c) };
                    let slot = unsafe { cmax.get_unchecked_mut(cw) };
                    *slot = if v > *slot { v } else { *slot };
                }
            } else {
            for &c in &best_unused {
                let cw = ((unsafe { *state.ch.weights.get_unchecked(c) }) as usize).clamp(1, 10);
                if (unsafe { *state.contrib.get_unchecked(c) }) > cmax[cw] { cmax[cw] = (unsafe { *state.contrib.get_unchecked(c) }); }
            }
            }
            for w in 1..=10 { if cmax[w - 1] > cmax[w] { cmax[w] = cmax[w - 1]; } }
            let mut rmin = [i32::MAX; 12];
            if fast(1 << 34) {
                for &rm in &worst_used {
                    let rw = ((unsafe { *state.ch.weights.get_unchecked(rm) }) as usize).clamp(1, 10);
                    let v = unsafe { *state.contrib.get_unchecked(rm) };
                    let slot = unsafe { rmin.get_unchecked_mut(rw) };
                    *slot = if v < *slot { v } else { *slot };
                }
            } else {
            for &rm in &worst_used {
                let rw = ((unsafe { *state.ch.weights.get_unchecked(rm) }) as usize).clamp(1, 10);
                if (unsafe { *state.contrib.get_unchecked(rm) }) < rmin[rw] { rmin[rw] = (unsafe { *state.contrib.get_unchecked(rm) }); }
            }
            }
            for w in (1..=10).rev() { if rmin[w + 1] < rmin[w] { rmin[w] = rmin[w + 1]; } }
            let slack_c = state.slack() as i32;
            let mut cand_live: Vec<usize> = Vec::with_capacity(best_unused.len());
            for &c in &best_unused {
                let wc = (unsafe { *state.ch.weights.get_unchecked(c) }) as i32;
                let lo = (wc - slack_c).max(1).min(10) as usize;
                if (unsafe { *state.contrib.get_unchecked(c) }) > rmin[lo] { cand_live.push(c); }
            }
            for &rm in &worst_used {
                let max_w = (unsafe { *state.ch.weights.get_unchecked(rm) }) + state.slack();
                if cmax[(max_w as usize).clamp(1, 10)] <= (unsafe { *state.contrib.get_unchecked(rm) }) { continue; }
                let row_rm = unsafe { state.ch.interaction_values.get_unchecked(rm) };
                for &c in &cand_live {
                    if (unsafe { *state.ch.weights.get_unchecked(c) }) > max_w { continue; }
                    let d = (unsafe { *state.contrib.get_unchecked(c) }) - (unsafe { *state.contrib.get_unchecked(rm) }) - unsafe { *row_rm.get_unchecked(c) };
                    if d > 0 && bs.map_or(true, |(_, _, bd)| d > bd) { bs = Some((c, rm, d)); }
                }
            }
            if let Some((c, rm, _)) = bs { state.replace_item(rm, c); continue; }
        }

        let slack = state.slack();
        if slack >= 2 {
            let fits: Vec<usize> = best_unused.iter().copied().filter(|&i| (unsafe { *state.ch.weights.get_unchecked(i) }) < slack).collect();
            let m = fits.len();
            if m >= 2 {
                let mut bp: Option<(usize, usize, i64)> = None;
                for ai in 0..m {
                    let a = fits[ai];
                    let wa = (unsafe { *state.ch.weights.get_unchecked(a) });
                    let ca = (unsafe { *state.contrib.get_unchecked(a) }) as i64;
                    for bi in (ai+1)..m {
                        let b = fits[bi];
                        if wa + (unsafe { *state.ch.weights.get_unchecked(b) }) > slack { continue; }
                        let d = ca + (unsafe { *state.contrib.get_unchecked(b) }) as i64 + (unsafe { *state.ch.interaction_values.get_unchecked(a).get_unchecked(b) }) as i64;
                        if d > 0 && bp.map_or(true, |(_, _, bd)| d > bd) { bp = Some((a, b, d)); }
                    }
                }
                if let Some((a, b, _)) = bp { state.add_item(a); state.add_item(b); continue; }
            }
        }

        {
            let cap = state.ch.max_weight;
            let mut bd = 0i64;
            let mut bm: Option<(usize, usize, usize)> = None;
            let ks = worst_used.len().min(80);
            let ku = best_unused.len().min(80);
            for si in 0..ks {
                let rm = worst_used[si];
                let c_rm = (unsafe { *state.contrib.get_unchecked(rm) }) as i64;
                let w_rm = (unsafe { *state.ch.weights.get_unchecked(rm) });
                let budget = state.slack() + w_rm;
                for ai in 0..ku {
                    let a1 = best_unused[ai];
                    let wa1 = (unsafe { *state.ch.weights.get_unchecked(a1) });
                    if wa1 >= budget { continue; }
                    let ca1_eff = (unsafe { *state.contrib.get_unchecked(a1) }) as i64 - (unsafe { *state.ch.interaction_values.get_unchecked(a1).get_unchecked(rm) }) as i64;
                    for bi in (ai+1)..ku {
                        let a2 = best_unused[bi];
                        let wa2 = (unsafe { *state.ch.weights.get_unchecked(a2) });
                        if wa1 + wa2 > budget { continue; }
                        let ca2_eff = (unsafe { *state.contrib.get_unchecked(a2) }) as i64 - (unsafe { *state.ch.interaction_values.get_unchecked(a2).get_unchecked(rm) }) as i64;
                        let syn = (unsafe { *state.ch.interaction_values.get_unchecked(a1).get_unchecked(a2) }) as i64;
                        let delta = ca1_eff + ca2_eff + syn - c_rm;
                        if delta > bd && state.total_weight - w_rm + wa1 + wa2 <= cap {
                            bd = delta; bm = Some((rm, a1, a2));
                        }
                    }
                }
            }
            if let Some((rm, a1, a2)) = bm {
                state.remove_item(rm); state.add_item(a1); state.add_item(a2);
                continue;
            }
        }

        {
            let cap = state.ch.max_weight;
            let mut bd = 0i64;
            let mut bm: Option<(usize, usize, usize)> = None;
            let ku = best_unused.len().min(80);
            let ks = worst_used.len().min(80);
            for ui in 0..ku {
                let add = best_unused[ui];
                let c_add = (unsafe { *state.contrib.get_unchecked(add) }) as i64;
                let w_add = (unsafe { *state.ch.weights.get_unchecked(add) });
                for si in 0..ks {
                    let r1 = worst_used[si];
                    let wr1 = (unsafe { *state.ch.weights.get_unchecked(r1) });
                    let cr1 = (unsafe { *state.contrib.get_unchecked(r1) }) as i64;
                    let c_add_r1 = (unsafe { *state.ch.interaction_values.get_unchecked(add).get_unchecked(r1) }) as i64;
                    for sj in (si+1)..ks {
                        let r2 = worst_used[sj];
                        let wr2 = (unsafe { *state.ch.weights.get_unchecked(r2) });
                        let new_w = state.total_weight + w_add - wr1 - wr2;
                        if new_w > cap { continue; }
                        let cr2 = (unsafe { *state.contrib.get_unchecked(r2) }) as i64;
                        let syn_r1_r2 = (unsafe { *state.ch.interaction_values.get_unchecked(r1).get_unchecked(r2) }) as i64;
                        let c_add_r2 = (unsafe { *state.ch.interaction_values.get_unchecked(add).get_unchecked(r2) }) as i64;
                        let lost = cr1 + cr2 - syn_r1_r2;
                        let gained = c_add - c_add_r1 - c_add_r2;
                        let delta = gained - lost;
                        if delta > bd { bd = delta; bm = Some((r1, r2, add)); }
                    }
                }
            }
            if let Some((r1, r2, add)) = bm {
                state.remove_item(r1); state.remove_item(r2); state.add_item(add);
                continue;
            }
        }

        {
            let cap = state.ch.max_weight;
            let mut bd = 0i64;
            let mut bm: Option<(usize, usize, usize, usize)> = None;
            let ks = worst_used.len().min(25);
            let ku = best_unused.len().min(25);
            for si in 0..ks {
                let r1 = worst_used[si];
                let wr1 = (unsafe { *state.ch.weights.get_unchecked(r1) });
                let cr1 = (unsafe { *state.contrib.get_unchecked(r1) }) as i64;
                for sj in (si+1)..ks {
                    let r2 = worst_used[sj];
                    let wr2 = (unsafe { *state.ch.weights.get_unchecked(r2) });
                    let cr2 = (unsafe { *state.contrib.get_unchecked(r2) }) as i64;
                    let syn_rm = (unsafe { *state.ch.interaction_values.get_unchecked(r1).get_unchecked(r2) }) as i64;
                    let lost = cr1 + cr2 - syn_rm;
                    let budget = state.slack() + wr1 + wr2;
                    for ui in 0..ku {
                        let a1 = best_unused[ui];
                        let wa1 = (unsafe { *state.ch.weights.get_unchecked(a1) });
                        if wa1 >= budget { continue; }
                        let ca1_eff = (unsafe { *state.contrib.get_unchecked(a1) }) as i64
                            - (unsafe { *state.ch.interaction_values.get_unchecked(a1).get_unchecked(r1) }) as i64
                            - (unsafe { *state.ch.interaction_values.get_unchecked(a1).get_unchecked(r2) }) as i64;
                        for uj in (ui+1)..ku {
                            let a2 = best_unused[uj];
                            let wa2 = (unsafe { *state.ch.weights.get_unchecked(a2) });
                            if wa1 + wa2 > budget { continue; }
                            let ca2_eff = (unsafe { *state.contrib.get_unchecked(a2) }) as i64
                                - (unsafe { *state.ch.interaction_values.get_unchecked(a2).get_unchecked(r1) }) as i64
                                - (unsafe { *state.ch.interaction_values.get_unchecked(a2).get_unchecked(r2) }) as i64;
                            let syn_add = (unsafe { *state.ch.interaction_values.get_unchecked(a1).get_unchecked(a2) }) as i64;
                            let delta = ca1_eff + ca2_eff + syn_add - lost;
                            if delta > bd && state.total_weight + wa1 + wa2 <= cap + wr1 + wr2 {
                                bd = delta; bm = Some((r1, r2, a1, a2));
                            }
                        }
                    }
                }
            }
            if let Some((r1, r2, a1, a2)) = bm {
                state.remove_item(r1); state.remove_item(r2);
                state.add_item(a1); state.add_item(a2);
                continue;
            }
        }

        if !improved { break; }
    }
}

fn perturb_by_strategy(state: &mut State, strength: usize, stall_count: usize, strategy: usize, rng: &mut Rng, hp: &Hparams, total_interactions: &[i64]) {
    let selected = state.selected_items();
    if selected.is_empty() { return; }
    let mut removal_candidates: Vec<(usize, i64)>;

    match strategy {
        0 => {
            removal_candidates = selected.iter().map(|&i| (i, (unsafe { *state.contrib.get_unchecked(i) }) as i64)).collect();
            removal_candidates.sort_unstable_by_key(|&(_, c)| c);
        },
        1 => {
            removal_candidates = selected.iter().map(|&i| (i, -((unsafe { *state.ch.weights.get_unchecked(i) }) as i64))).collect();
            removal_candidates.sort_unstable_by_key(|&(_, w)| w);
        },
        2 => {
            removal_candidates = selected.iter().map(|&i| {
                let syn = (unsafe { *state.contrib.get_unchecked(i) }) as i64 - (unsafe { *state.ch.values.get_unchecked(i) }) as i64;
                (i, syn)
            }).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        },
        3 => {
            removal_candidates = selected.iter().map(|&i| {
                let w = ((unsafe { *state.ch.weights.get_unchecked(i) }) as i64).max(1);
                (i, dw((unsafe { *state.contrib.get_unchecked(i) }) as i64 * 1000, w))
            }).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        },
        4 => {
            removal_candidates = selected.iter().map(|&i| {
                let w = ((unsafe { *state.ch.weights.get_unchecked(i) }) as i64).max(1);
                let density = dw((unsafe { *state.contrib.get_unchecked(i) }) as i64 * 100, w);
                (i, (unsafe { *state.ch.weights.get_unchecked(i) }) as i64 - density)
            }).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        },
        5 => {
            removal_candidates = selected.iter().map(|&i| {
                let w = ((unsafe { *state.ch.weights.get_unchecked(i) }) as i64).max(1);
                (i, ((unsafe { *state.contrib.get_unchecked(i) }) as i64 * 10000) / (w * w))
            }).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        },
        6 => {
            removal_candidates = selected.iter().map(|&i| (i, rng.next_u32() as i64)).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        },
        7 => {
            removal_candidates = selected.iter().map(|&i| {
                let anti = 2 * (unsafe { *state.contrib.get_unchecked(i) }) as i64 - total_interactions[i];
                let w = ((unsafe { *state.ch.weights.get_unchecked(i) }) as i64).max(1);
                (i, dw(anti * 1000, w))
            }).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        },
        8 => {
            removal_candidates = selected.iter().map(|&i| {
                let c = (unsafe { *state.contrib.get_unchecked(i) }) as i64;
                let potential = total_interactions[i];
                (i, c * 100 - potential)
            }).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        },
        _ => {
            removal_candidates = selected.iter().map(|&i| (i, -((unsafe { *state.contrib.get_unchecked(i) }) as i64))).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        }
    }

    let base_remove = (selected.len() / hp.perturb_base_frac).max(2);
    let adaptive_mult = 1 + (stall_count / 2);
    let n_remove = (base_remove * adaptive_mult).min(strength).min(selected.len() * 2 / hp.perturb_max_frac);
    for j in 0..n_remove {
        if j < removal_candidates.len() {
            state.remove_item(removal_candidates[j].0);
        }
    }
}

fn greedy_reconstruct(state: &mut State, strategy: usize, total_interactions: &[i64]) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let mut candidates: Vec<usize> = Vec::with_capacity(n);
    unsafe {
        let op = candidates.as_mut_ptr();
        let bp = state.selected_bit.as_ptr();
        let mut k = 0usize;
        let mut i = 0usize;
        while i + 4 <= n {
            let b0 = *bp.add(i); op.add(k).write(i); k += (!b0) as usize;
            let b1 = *bp.add(i + 1); op.add(k).write(i + 1); k += (!b1) as usize;
            let b2 = *bp.add(i + 2); op.add(k).write(i + 2); k += (!b2) as usize;
            let b3 = *bp.add(i + 3); op.add(k).write(i + 3); k += (!b3) as usize;
            i += 4;
        }
        while i < n { let b = *bp.add(i); op.add(k).write(i); k += (!b) as usize; i += 1; }
        candidates.set_len(k);
    }

    let gmode = strategy % 6;
    if fast(1 << 22) && MINW.load(std::sync::atomic::Ordering::Relaxed) >= 1
        && exact_sort::compatible() {
        let nd = (n as i64).max(1);
        let mut keys = vec![0i64; n];
        if fast(1 << 31) {
            let k4 = fast(1 << 48);
            macro_rules! kloop { ($e:expr) => {{
                let cn = candidates.len();
                let mut t = 0usize;
                if k4 {
                    let cp = candidates.as_ptr();
                    let kp = keys.as_mut_ptr();
                    macro_rules! one { ($t:expr) => {{
                        let i = unsafe { *cp.add($t) };
                        let c = (unsafe { *state.contrib.get_unchecked(i) }) as i64;
                        let w = ((unsafe { *state.ch.weights.get_unchecked(i) }) as i64).max(1);
                        let _ = w;
                        unsafe { *kp.add(i) = $e(i, c, w); }
                    }}; }
                    while t + 4 <= cn { one!(t); one!(t + 1); one!(t + 2); one!(t + 3); t += 4; }
                }
                for &i in &candidates[t..] {
                    let c = (unsafe { *state.contrib.get_unchecked(i) }) as i64;
                    let w = ((unsafe { *state.ch.weights.get_unchecked(i) }) as i64).max(1);
                    let _ = w;
                    unsafe { *keys.get_unchecked_mut(i) = $e(i, c, w); }
                }
            }}; }
            match gmode {
                0 | 1 => kloop!(|_i: usize, c: i64, _w: i64| -c),
                2 => kloop!(|i: usize, c: i64, _w: i64| -(total_interactions[i] + c / 10)),
                3 => kloop!(|_i: usize, c: i64, w: i64| -dw(c * 100, w)),
                4 => kloop!(|i: usize, c: i64, w: i64| -(dw((2 * c - total_interactions[i]) * 100, w) + c / 5)),
                _ => kloop!(|i: usize, c: i64, _w: i64| -(c + (total_interactions[i] / nd) * 3)),
            }
        } else {
        for &i in &candidates {
            let c = (unsafe { *state.contrib.get_unchecked(i) }) as i64;
            let w = ((unsafe { *state.ch.weights.get_unchecked(i) }) as i64).max(1);
            keys[i] = match gmode {
                0 => -c,
                1 => -c,
                2 => -(total_interactions[i] + c / 10),
                3 => -dw(c * 100, w),
                4 => -(dw((2 * c - total_interactions[i]) * 100, w) + c / 5),
                _ => -(c + (total_interactions[i] / nd) * 3),
            };
        }
        }
        let end = exact_sort::keyed_prefix(&mut candidates, &keys, &state.ch.weights,
                                           state.slack(), gmode == 1);
        for &i in &candidates[..end] {
            if state.total_weight + (unsafe { *state.ch.weights.get_unchecked(i) }) <= cap {
                state.add_item(i);
            }
        }
        return;
    }
    match strategy % 6 {
        0 => candidates.sort_unstable_by_key(|&i| -(unsafe { *state.contrib.get_unchecked(i) })),
        1 => candidates.sort_unstable_by(|&a, &b| {
            (unsafe { *state.ch.weights.get_unchecked(a) }).cmp(&(unsafe { *state.ch.weights.get_unchecked(b) }))
                .then((unsafe { *state.contrib.get_unchecked(b) }).cmp(&(unsafe { *state.contrib.get_unchecked(a) })))
        }),
        2 => candidates.sort_unstable_by_key(|&i| {
            -(total_interactions[i] + (unsafe { *state.contrib.get_unchecked(i) }) as i64 / 10)
        }),
        3 => {
            let mut keys = vec![0i64; n];
            for &i in &candidates {
                let w = ((unsafe { *state.ch.weights.get_unchecked(i) }) as i64).max(1);
                keys[i] = -dw((unsafe { *state.contrib.get_unchecked(i) }) as i64 * 100, w);
            }
            candidates.sort_unstable_by_key(|&i| keys[i]);
        },
        4 => {
            let mut keys = vec![0i64; n];
            for &i in &candidates {
                let w = ((unsafe { *state.ch.weights.get_unchecked(i) }) as i64).max(1);
                let c = (unsafe { *state.contrib.get_unchecked(i) }) as i64;
                let anti = 2 * c - total_interactions[i];
                keys[i] = -(dw(anti * 100, w) + c / 5);
            }
            candidates.sort_unstable_by_key(|&i| keys[i]);
        },
        _ => {
            let nd = (n as i64).max(1);
            let mut keys = vec![0i64; n];
            for &i in &candidates {
                let c = (unsafe { *state.contrib.get_unchecked(i) }) as i64;
                let potential = total_interactions[i] / nd;
                keys[i] = -(c + potential * 3);
            }
            candidates.sort_unstable_by_key(|&i| keys[i]);
        },
    }

    let early = fast(32) && MINW.load(std::sync::atomic::Ordering::Relaxed) >= 1;
    for &i in &candidates {
        if state.total_weight + (unsafe { *state.ch.weights.get_unchecked(i) }) <= cap {
            state.add_item(i);
            if early && state.total_weight == cap { break; }
        }
    }
}

struct TopNeighbors {
    vals: Vec<Vec<i32>>,
    friends: Vec<Vec<usize>>,
}
impl TopNeighbors {
    fn new(ch: &Challenge, k: usize, csr: Option<&Csr>) -> Self {
        let n = ch.num_items;
        let mut min_weight = u64::MAX;
        let mut max_weight = 0u64;
        for &weight in &ch.weights {
            let weight = (weight as u64).max(1);
            min_weight = min_weight.min(weight);
            max_weight = max_weight.max(weight);
        }
        let weight_aware = k >= 2 && min_weight != u64::MAX
            && max_weight > min_weight.saturating_mul(3) / 2;
        let abs_slots = if weight_aware { (k + 1) / 2 } else { k };
        let mut friends = vec![Vec::with_capacity(k); n];
        let mut vals: Vec<Vec<i32>> = vec![Vec::with_capacity(k); n];

        if fast(1 << 25) && csr.is_some() && weight_aware {
            let c = csr.unwrap();
            for i in 0..n {
                let mut absolute: Vec<(usize, i32, u64)> = Vec::with_capacity(abs_slots);
                let mut density: Vec<(usize, i32, u64)> = Vec::with_capacity(k);
                let wi = ((unsafe { *ch.weights.get_unchecked(i) }) as u64).max(1);
                let rs = unsafe { *c.starts.get_unchecked(i) } as usize;
                let re = unsafe { *c.starts.get_unchecked(i + 1) } as usize;
                let mut abs_thr: i32 = 0;
                let mut den_v: i32 = 0;
                let mut den_w: u64 = 1;
                let fastd = fast(1 << 32);
                for t in rs..re {
                    let j = (unsafe { *c.idx.get_unchecked(t) }) as usize;
                    let interaction = (unsafe { *c.val.get_unchecked(t) }) as i32;
                    let combined_weight = wi + ((unsafe { *ch.weights.get_unchecked(j) }) as u64).max(1);
                    if fastd {
                        let af = absolute.len() >= abs_slots;
                        let df = density.len() >= k;
                        let ap = !af | (interaction > abs_thr);
                        let dp = !df | ((interaction as u64) * den_w > (den_v as u64) * combined_weight);
                        if !(ap | dp) { continue; }
                    }
                    let al = absolute.len();
                    if al < abs_slots || interaction > unsafe { absolute.get_unchecked(al - 1).1 } {
                        let abs_pos = absolute.iter().position(|&(_, value, _)| value < interaction)
                            .unwrap_or(al);
                        if abs_pos < abs_slots {
                            absolute.insert(abs_pos, (j, interaction, combined_weight));
                            if absolute.len() > abs_slots { absolute.pop(); }
                        }
                    }
                    let dl = density.len();
                    let pass = if dl < k {
                        true
                    } else {
                        let last = unsafe { density.get_unchecked(dl - 1) };
                        (interaction as u64) * last.2 > (last.1 as u64) * combined_weight
                    };
                    if pass {
                        let density_pos = density.iter().position(|&(_, value, weight)| {
                            (interaction as u64) * weight > (value as u64) * combined_weight
                        }).unwrap_or(dl);
                        if density_pos < k {
                            density.insert(density_pos, (j, interaction, combined_weight));
                            if density.len() > k { density.pop(); }
                        }
                    }
                    if fastd {
                        if absolute.len() >= abs_slots {
                            abs_thr = unsafe { absolute.get_unchecked(abs_slots - 1).1 };
                        }
                        if density.len() >= k {
                            let l = unsafe { density.get_unchecked(k - 1) };
                            den_v = l.1; den_w = l.2;
                        }
                    }
                }
                let mut neighbors = Vec::with_capacity(k);
                let mut nvals: Vec<i32> = Vec::with_capacity(k);
                for (j, value, _) in absolute {
                    neighbors.push(j);
                    nvals.push(value);
                }
                for (j, value, _) in density {
                    if neighbors.len() >= k { break; }
                    if !neighbors.contains(&j) {
                        neighbors.push(j);
                        nvals.push(value);
                    }
                }
                friends[i] = neighbors;
                vals[i] = nvals;
            }
            return Self { friends, vals };
        }

        for i in 0..n {
            let mut absolute: Vec<(usize, i32, u64)> = Vec::with_capacity(abs_slots);
            let mut density: Vec<(usize, i32, u64)> = Vec::with_capacity(k);
            let wi = ((unsafe { *ch.weights.get_unchecked(i) }) as u64).max(1);

            let (rs, re) = match csr {
                Some(c) => (unsafe { *c.starts.get_unchecked(i) } as usize,
                            unsafe { *c.starts.get_unchecked(i + 1) } as usize),
                None => (0usize, n),
            };
            for t in rs..re {
                let j = match csr {
                    Some(c) => (unsafe { *c.idx.get_unchecked(t) }) as usize,
                    None => t,
                };
                if i == j { continue; }
                let interaction = match csr {
                    Some(c) => (unsafe { *c.val.get_unchecked(t) }) as i32,
                    None => unsafe { *ch.interaction_values.get_unchecked(i).get_unchecked(j) },
                };
                if interaction <= 0 { continue; }

                let combined_weight = wi + ((unsafe { *ch.weights.get_unchecked(j) }) as u64).max(1);
                if absolute.len() < abs_slots
                    || interaction > absolute[absolute.len() - 1].1
                {
                    let abs_pos = absolute.iter().position(|&(_, value, _)| value < interaction)
                        .unwrap_or(absolute.len());
                    if abs_pos < abs_slots {
                        absolute.insert(abs_pos, (j, interaction, combined_weight));
                        if absolute.len() > abs_slots { absolute.pop(); }
                    }
                }

                if weight_aware {
                    let dl = density.len();
                    if dl < k
                        || (interaction as u128) * (density[dl - 1].2 as u128)
                            > (density[dl - 1].1 as u128) * (combined_weight as u128)
                    {
                        let density_pos = density.iter().position(|&(_, value, weight)| {
                            (interaction as u128) * (weight as u128)
                                > (value as u128) * (combined_weight as u128)
                        }).unwrap_or(density.len());
                        if density_pos < k {
                            density.insert(density_pos, (j, interaction, combined_weight));
                            if density.len() > k { density.pop(); }
                        }
                    }
                }
            }

            let mut neighbors = Vec::with_capacity(k);
            let mut nvals: Vec<i32> = Vec::with_capacity(k);
            for (j, value, _) in absolute {
                neighbors.push(j);
                nvals.push(value);
            }
            if weight_aware {
                for (j, value, _) in density {
                    if neighbors.len() >= k { break; }
                    if !neighbors.contains(&j) {
                        neighbors.push(j);
                        nvals.push(value);
                    }
                }
            }
            friends[i] = neighbors;
            vals[i] = nvals;
        }
        Self { friends, vals }
    }
}

fn local_search_vnd_tsn(state: &mut State, tsn: &TopNeighbors) {
    let n = state.ch.num_items;
    let wk: usize = if n > 3000 { 200 } else { 300 };
    const LDIV: [i64; 11] = [2520, 2520, 1260, 840, 630, 504, 420, 360, 315, 280, 252];
    let mut unused_buf: Vec<(usize, i64)> = Vec::with_capacity(n);
    let mut used_buf: Vec<(usize, i64)> = Vec::with_capacity(n);
    let mut best_unused: Vec<usize> = Vec::with_capacity(wk);
    let mut worst_used: Vec<usize> = Vec::with_capacity(wk);
    let mut cand_live: Vec<usize> = Vec::with_capacity(wk);
    let mut cl_blk: Vec<i32> = Vec::with_capacity(wk / 16 + 2);
    let mut g4: Vec<i64> = Vec::with_capacity(wk);
    let mut g4_blk: Vec<i64> = Vec::with_capacity(wk / 16 + 2);
    let mut h5: Vec<i64> = Vec::with_capacity(wk);
    let mut u5_blk: Vec<i64> = Vec::with_capacity(wk / 16 + 2);
    let mut u5: Vec<i64> = Vec::with_capacity(wk);
    let mut fc4: Vec<(u32, u32, i64)> = Vec::with_capacity(wk * 16);
    let mut fc4_off: Vec<u32> = Vec::with_capacity(wk + 1);
    let mut fc5: Vec<(u32, u32, i64)> = Vec::with_capacity(wk * 16);
    let mut fc5_off: Vec<u32> = Vec::with_capacity(wk + 1);
    let mut mq: Vec<i64> = vec![i64::MIN; wk * 11];
    let mut mv: Vec<i64> = vec![i64::MIN; wk * 12];
    let mut ord4v: Vec<i64> = Vec::with_capacity(wk * 10);
    let mut ord4i: Vec<u32> = Vec::with_capacity(wk * 10);
    let mut pre4: Vec<u64> = Vec::with_capacity((wk + 1) * 8 * 10);
    let mut ord4_off = [0usize; 11];
    let mut pre4_off = [0usize; 11];
    let mut ord5v: Vec<i64> = Vec::with_capacity(wk * 10);
    let mut ord5i: Vec<u32> = Vec::with_capacity(wk * 10);
    let mut pre5: Vec<u64> = Vec::with_capacity((wk + 1) * 8 * 10);
    let mut ord5_off = [0usize; 11];
    let mut pre5_off = [0usize; 11];
    let mut tmp_ord: Vec<(i64, u32)> = Vec::with_capacity(wk);
    let ch_ref: &Challenge = state.ch;
    let mut windows = exact_windows::Windows::new(&ch_ref.weights);
    let use_win = fast(1 << 18) && windows.is_packed() && exact_sort::compatible();
    let use_paths = fast(1 << 20);
    let usefv = fast(1 << 26);
    let mut pending: Vec<(exact_paths::Snapshot, usize)> = Vec::new();
    for step in 0..80 {
        let step_hash = state.solution_hash;
        if use_paths {
            let hit = VND_PATHS.with(|c| c.borrow().lookup(wk * 4 + 1, step_hash, &state.selected_bit,
                &state.contrib, state.total_value, state.total_weight, 80 - step));
            if let Some((output, steps)) = hit {
                state.selected_bit.clone_from(&output.bits);
                state.contrib.clone_from(&output.contrib);
                state.total_value = output.value;
                state.total_weight = output.weight;
                state.solution_hash = output.hash;
                VND_PATHS.with(|c| c.borrow_mut().publish(wk * 4 + 1, pending, step + steps, output));
                return;
            }
        }
        unused_buf.clear();
        used_buf.clear();
        best_unused.clear();
        worst_used.clear();
        let (ku, ks);
        if use_win {
            windows.build(&state.contrib, &state.selected_bit, wk, &mut best_unused, &mut worst_used);
            ku = best_unused.len(); ks = worst_used.len();
        } else {
        if fast(1) {
            unsafe {
                let cp = state.contrib.as_ptr();
                let wp = state.ch.weights.as_ptr();
                let bp = state.selected_bit.as_ptr();
                let up = unused_buf.as_mut_ptr();
                let sp = used_buf.as_mut_ptr();
                let mut ui = 0usize;
                let mut si = 0usize;
                let mut i = 0usize;
                while i + 4 <= n {
                    let c0 = *cp.add(i) as i64;
                    let w0 = (*wp.add(i) as i64).max(1);
                    let k0 = c0 * *LDIV.get_unchecked((w0 as usize).min(10));
                    let s0 = *bp.add(i);
                    (if s0 { sp.add(si) } else { up.add(ui) }).write((i, k0));
                    ui += (!s0) as usize; si += s0 as usize;
                    let c1 = *cp.add(i + 1) as i64;
                    let w1 = (*wp.add(i + 1) as i64).max(1);
                    let k1 = c1 * *LDIV.get_unchecked((w1 as usize).min(10));
                    let s1 = *bp.add(i + 1);
                    (if s1 { sp.add(si) } else { up.add(ui) }).write((i + 1, k1));
                    ui += (!s1) as usize; si += s1 as usize;
                    let c2 = *cp.add(i + 2) as i64;
                    let w2 = (*wp.add(i + 2) as i64).max(1);
                    let k2 = c2 * *LDIV.get_unchecked((w2 as usize).min(10));
                    let s2 = *bp.add(i + 2);
                    (if s2 { sp.add(si) } else { up.add(ui) }).write((i + 2, k2));
                    ui += (!s2) as usize; si += s2 as usize;
                    let c3 = *cp.add(i + 3) as i64;
                    let w3 = (*wp.add(i + 3) as i64).max(1);
                    let k3 = c3 * *LDIV.get_unchecked((w3 as usize).min(10));
                    let s3 = *bp.add(i + 3);
                    (if s3 { sp.add(si) } else { up.add(ui) }).write((i + 3, k3));
                    ui += (!s3) as usize; si += s3 as usize;
                    i += 4;
                }
                while i < n {
                    let c0 = *cp.add(i) as i64;
                    let w0 = (*wp.add(i) as i64).max(1);
                    let k0 = c0 * *LDIV.get_unchecked((w0 as usize).min(10));
                    let s0 = *bp.add(i);
                    (if s0 { sp.add(si) } else { up.add(ui) }).write((i, k0));
                    ui += (!s0) as usize; si += s0 as usize;
                    i += 1;
                }
                unused_buf.set_len(ui);
                used_buf.set_len(si);
            }
        } else {
            for i in 0..n {
                let (c, w, sel) = unsafe {
                    (*state.contrib.get_unchecked(i) as i64,
                     (*state.ch.weights.get_unchecked(i) as i64).max(1),
                     *state.selected_bit.get_unchecked(i))
                };
                let key = c * unsafe { *LDIV.get_unchecked((w as usize).min(10)) };
                if sel { used_buf.push((i, key)); } else { unused_buf.push((i, key)); }
            }
        }
        ku = wk.min(unused_buf.len());
        if ku > 0 && ku < unused_buf.len() {
            unused_buf.select_nth_unstable_by(ku - 1, |a, b| b.1.cmp(&a.1));
        }
        ks = wk.min(used_buf.len());
        if ks > 0 && ks < used_buf.len() {
            used_buf.select_nth_unstable_by(ks - 1, |a, b| a.1.cmp(&b.1));
        }
        for t in &unused_buf[..ku] { best_unused.push(t.0); }
        for t in &used_buf[..ks] { worst_used.push(t.0); }
        }
        if use_paths {
            pending.push((exact_paths::Snapshot::new(&state.selected_bit, &state.contrib,
                state.total_value, state.total_weight, step_hash), step));
        }

        let slack = state.slack();
        if slack > 0 {
            let mut ba: Option<(usize, i32)> = None;
            for &c in &best_unused {
                if (unsafe { *state.ch.weights.get_unchecked(c) }) > slack { continue; }
                let d = (unsafe { *state.contrib.get_unchecked(c) });
                if d > 0 && ba.map_or(true, |(_, bd)| d > bd) { ba = Some((c, d)); }
            }
            if let Some((c, _)) = ba { state.add_item(c); continue; }
        }

        {
            let mut bs: Option<(usize, usize, i32)> = None;
            let mut cmax = [i32::MIN; 11];
            if fast(1 << 34) {
                for &c in &best_unused {
                    let cw = ((unsafe { *state.ch.weights.get_unchecked(c) }) as usize).clamp(1, 10);
                    let v = unsafe { *state.contrib.get_unchecked(c) };
                    let slot = unsafe { cmax.get_unchecked_mut(cw) };
                    *slot = if v > *slot { v } else { *slot };
                }
            } else {
            for &c in &best_unused {
                let cw = ((unsafe { *state.ch.weights.get_unchecked(c) }) as usize).clamp(1, 10);
                if (unsafe { *state.contrib.get_unchecked(c) }) > cmax[cw] { cmax[cw] = (unsafe { *state.contrib.get_unchecked(c) }); }
            }
            }
            for w in 1..=10 { if cmax[w - 1] > cmax[w] { cmax[w] = cmax[w - 1]; } }
            let mut rmin = [i32::MAX; 12];
            if fast(1 << 34) {
                for &rm in &worst_used {
                    let rw = ((unsafe { *state.ch.weights.get_unchecked(rm) }) as usize).clamp(1, 10);
                    let v = unsafe { *state.contrib.get_unchecked(rm) };
                    let slot = unsafe { rmin.get_unchecked_mut(rw) };
                    *slot = if v < *slot { v } else { *slot };
                }
            } else {
            for &rm in &worst_used {
                let rw = ((unsafe { *state.ch.weights.get_unchecked(rm) }) as usize).clamp(1, 10);
                if (unsafe { *state.contrib.get_unchecked(rm) }) < rmin[rw] { rmin[rw] = (unsafe { *state.contrib.get_unchecked(rm) }); }
            }
            }
            for w in (1..=10).rev() { if rmin[w + 1] < rmin[w] { rmin[w] = rmin[w + 1]; } }
            let slack_c = state.slack() as i32;
            cand_live.clear();
            if fast(1 << 34) {
                unsafe {
                    let cp = cand_live.as_mut_ptr();
                    let mut kk = 0usize;
                    for &c in &best_unused {
                        let wc = (*state.ch.weights.get_unchecked(c)) as i32;
                        let lo = (wc - slack_c).max(1).min(10) as usize;
                        cp.add(kk).write(c);
                        kk += ((*state.contrib.get_unchecked(c)) > *rmin.get_unchecked(lo)) as usize;
                    }
                    cand_live.set_len(kk);
                }
            } else {
            for &c in &best_unused {
                let wc = (unsafe { *state.ch.weights.get_unchecked(c) }) as i32;
                let lo = (wc - slack_c).max(1).min(10) as usize;
                if (unsafe { *state.contrib.get_unchecked(c) }) > rmin[lo] { cand_live.push(c); }
            }
            }
            const CB: usize = 16;
            if fast(2) {
                cl_blk.clear();
                let mut q = 0usize;
                while q < cand_live.len() {
                    let e = (q + CB).min(cand_live.len());
                    let mut mx = i32::MIN;
                    for t in q..e {
                        let v = unsafe { *state.contrib.get_unchecked(*cand_live.get_unchecked(t)) };
                        if v > mx { mx = v; }
                    }
                    cl_blk.push(mx);
                    q = e;
                }
            }
            for &rm in &worst_used {
                let max_w = (unsafe { *state.ch.weights.get_unchecked(rm) }) + state.slack();
                let crm_i = (unsafe { *state.contrib.get_unchecked(rm) });
                if fast(1024) {
                    let thr0 = bs.map_or(0i64, |(_, _, d)| d as i64);
                    if (cmax[(max_w as usize).clamp(1, 10)] as i64) - (crm_i as i64) <= thr0 { continue; }
                } else if cmax[(max_w as usize).clamp(1, 10)] <= crm_i { continue; }
                let row_rm = unsafe { state.ch.interaction_values.get_unchecked(rm) };
                if fast(2) {
                    let crm = (unsafe { *state.contrib.get_unchecked(rm) }) as i64;
                    let mut b = 0usize;
                    while b < cl_blk.len() {
                        let bmv = (unsafe { *cl_blk.get_unchecked(b) }) as i64;
                        let lo = b * CB;
                        let hi = (lo + CB).min(cand_live.len());
                        b += 1;
                        let thr = bs.map_or(0i64, |(_, _, d)| d as i64);
                        if bmv - crm <= thr { continue; }
                        if fast(1 << 33) {
                            let mut bd0 = bs.map_or(0i32, |(_, _, d)| d);
                            let crm_i = unsafe { *state.contrib.get_unchecked(rm) };
                            for t in lo..hi {
                                let c = unsafe { *cand_live.get_unchecked(t) };
                                let ok = (unsafe { *state.ch.weights.get_unchecked(c) }) <= max_w;
                                let d = (unsafe { *state.contrib.get_unchecked(c) }) - crm_i
                                    - unsafe { *row_rm.get_unchecked(c) };
                                if ok & (d > bd0) { bs = Some((c, rm, d)); bd0 = d; }
                            }
                        } else {
                        for t in lo..hi {
                            let c = unsafe { *cand_live.get_unchecked(t) };
                            if (unsafe { *state.ch.weights.get_unchecked(c) }) > max_w { continue; }
                            let d = (unsafe { *state.contrib.get_unchecked(c) }) - (unsafe { *state.contrib.get_unchecked(rm) }) - unsafe { *row_rm.get_unchecked(c) };
                            if d > 0 && bs.map_or(true, |(_, _, bd)| d > bd) { bs = Some((c, rm, d)); }
                        }
                        }
                    }
                } else {
                    for &c in &cand_live {
                        if (unsafe { *state.ch.weights.get_unchecked(c) }) > max_w { continue; }
                        let d = (unsafe { *state.contrib.get_unchecked(c) }) - (unsafe { *state.contrib.get_unchecked(rm) }) - unsafe { *row_rm.get_unchecked(c) };
                        if d > 0 && bs.map_or(true, |(_, _, bd)| d > bd) { bs = Some((c, rm, d)); }
                    }
                }
            }
            if let Some((c, rm, _)) = bs { state.replace_item(rm, c); continue; }
        }

        {
            let slack_i = state.slack() as i32;
            if slack_i >= 2 {
                let mut bd = 0i64;
                let mut bp = None;
                for &a1 in &best_unused {
                    let ca1 = (unsafe { *state.contrib.get_unchecked(a1) }) as i64;
                    if ca1 <= 0 && bd > 0 { break; }
                    let wa1 = (unsafe { *state.ch.weights.get_unchecked(a1) }) as i32;
                    if wa1 >= slack_i { continue; }
                    if usefv {
                        let fr = unsafe { tsn.friends.get_unchecked(a1) };
                        let fv = unsafe { tsn.vals.get_unchecked(a1) };
                        for (fi, &a2) in fr.iter().enumerate() {
                            if (unsafe { *state.selected_bit.get_unchecked(a2) }) || a1 == a2 { continue; }
                            if wa1 + ((unsafe { *state.ch.weights.get_unchecked(a2) }) as i32) <= slack_i {
                                let delta = ca1 + (unsafe { *state.contrib.get_unchecked(a2) }) as i64 + (unsafe { *fv.get_unchecked(fi) }) as i64;
                                if delta > bd { bd = delta; bp = Some((a1, a2)); }
                            }
                        }
                    } else {
                    for &a2 in (unsafe { tsn.friends.get_unchecked(a1) }) {
                        if (unsafe { *state.selected_bit.get_unchecked(a2) }) || a1 == a2 { continue; }
                        if wa1 + ((unsafe { *state.ch.weights.get_unchecked(a2) }) as i32) <= slack_i {
                            let delta = ca1 + (unsafe { *state.contrib.get_unchecked(a2) }) as i64 + (unsafe { *state.ch.interaction_values.get_unchecked(a1).get_unchecked(a2) }) as i64;
                            if delta > bd { bd = delta; bp = Some((a1, a2)); }
                        }
                    }
                    }
                }
                if let Some((a1, a2)) = bp { state.add_item(a1); state.add_item(a2); continue; }
            }
        }

        {
            let cap = state.ch.max_weight;
            let mut bd = 0i64;
            let mut bm = None;
            if fast(256) {
            fc4.clear(); fc4_off.clear(); g4.clear();
            for (ai, &a1) in best_unused.iter().enumerate() {
                fc4_off.push(fc4.len() as u32);
                let row_a1 = unsafe { state.ch.interaction_values.get_unchecked(a1) };
                let _ = &row_a1;
                let mut best = i64::MIN;
                let mb = ai * 11;
                if fast(4096) { for t in 0..11 { unsafe { *mq.get_unchecked_mut(mb + t) = i64::MIN; } } }
                if usefv {
                    let fr = unsafe { tsn.friends.get_unchecked(a1) };
                    let fv = unsafe { tsn.vals.get_unchecked(a1) };
                    for (fi, &a2) in fr.iter().enumerate() {
                        if (unsafe { *state.selected_bit.get_unchecked(a2) }) || a1 == a2 { continue; }
                        let q = (unsafe { *state.contrib.get_unchecked(a2) }) as i64
                            + (unsafe { *fv.get_unchecked(fi) }) as i64;
                        let wa2 = (unsafe { *state.ch.weights.get_unchecked(a2) });
                        fc4.push((a2 as u32, wa2, q));
                        if fast(4096) {
                            let sl = mb + (wa2 as usize).clamp(1, 10);
                            if q > unsafe { *mq.get_unchecked(sl) } { unsafe { *mq.get_unchecked_mut(sl) = q; } }
                        }
                        if q > best { best = q; }
                    }
                } else {
                for &a2 in (unsafe { tsn.friends.get_unchecked(a1) }) {
                    if (unsafe { *state.selected_bit.get_unchecked(a2) }) || a1 == a2 { continue; }
                    let q = (unsafe { *state.contrib.get_unchecked(a2) }) as i64
                        + (unsafe { *row_a1.get_unchecked(a2) }) as i64;
                    let wa2 = (unsafe { *state.ch.weights.get_unchecked(a2) });
                    fc4.push((a2 as u32, wa2, q));
                    if fast(4096) {
                        let sl = mb + (wa2 as usize).clamp(1, 10);
                        if q > unsafe { *mq.get_unchecked(sl) } { unsafe { *mq.get_unchecked_mut(sl) = q; } }
                    }
                    if q > best { best = q; }
                }
                }
                if fast(4096) { for t in 1..11 { if unsafe { *mq.get_unchecked(mb + t - 1) } > unsafe { *mq.get_unchecked(mb + t) } { unsafe { *mq.get_unchecked_mut(mb + t) = *mq.get_unchecked(mb + t - 1); } } } }
                g4.push(if best == i64::MIN { i64::MIN } else { (unsafe { *state.contrib.get_unchecked(a1) }) as i64 + best });
            }
            fc4_off.push(fc4.len() as u32);
            let mut g4max = i64::MIN;
            for &v in &g4 { if v > g4max { g4max = v; } }
            const GB: usize = 16;
            let words4 = (g4.len() + 63) / 64;
            let use_pref4 = fast(1 << 27) && fast(4096) && words4 <= 8
                && MINW.load(std::sync::atomic::Ordering::Relaxed) >= 1
                && MAXW.load(std::sync::atomic::Ordering::Relaxed) <= 10;
            let slack4 = state.slack();
            if use_pref4 {
                ord4v.clear(); ord4i.clear(); pre4.clear();
                for bi in 0..10usize {
                    ord4_off[bi] = ord4i.len();
                    pre4_off[bi] = pre4.len();
                    let budget = slack4 + bi as u32 + 1;
                    tmp_ord.clear();
                    for ai in 0..g4.len() {
                        if unsafe { *g4.get_unchecked(ai) } == i64::MIN { continue; }
                        let a1 = unsafe { *best_unused.get_unchecked(ai) };
                        let wa1 = unsafe { *state.ch.weights.get_unchecked(a1) };
                        if wa1 >= budget { continue; }
                        let t = ((budget - wa1) as usize).min(10);
                        let mqv = unsafe { *mq.get_unchecked(ai * 11 + t) };
                        if mqv == i64::MIN { continue; }
                        let ca1 = (unsafe { *state.contrib.get_unchecked(a1) }) as i64;
                        tmp_ord.push((ca1 + mqv, ai as u32));
                    }
                    tmp_ord.sort_unstable_by(|a, b| b.0.cmp(&a.0));
                    let base = pre4.len();
                    pre4.resize(base + (tmp_ord.len() + 1) * words4, 0u64);
                    for k in 0..tmp_ord.len() {
                        let (v, ai) = unsafe { *tmp_ord.get_unchecked(k) };
                        ord4v.push(v); ord4i.push(ai);
                        let src = base + k * words4;
                        let dst = src + words4;
                        for w in 0..words4 { unsafe { *pre4.get_unchecked_mut(dst + w) = *pre4.get_unchecked(src + w); } }
                        unsafe { *pre4.get_unchecked_mut(dst + ai as usize / 64) |= 1u64 << (ai & 63); }
                    }
                }
                ord4_off[10] = ord4i.len();
                pre4_off[10] = pre4.len();
            } else {
            g4_blk.clear();
            {
                let mut q = 0usize;
                while q < g4.len() {
                    let e = (q + GB).min(g4.len());
                    let mut mx = i64::MIN;
                    for t in q..e { let v = unsafe { *g4.get_unchecked(t) }; if v > mx { mx = v; } }
                    g4_blk.push(mx);
                    q = e;
                }
            }
            }
            for &rm in &worst_used {
                let c_rm = (unsafe { *state.contrib.get_unchecked(rm) }) as i64;
                if g4max == i64::MIN || g4max - c_rm <= bd { continue; }
                let w_rm = (unsafe { *state.ch.weights.get_unchecked(rm) });
                let budget = state.slack() + w_rm;
                let wbase = state.total_weight - w_rm;
                let row_rm = unsafe { state.ch.interaction_values.get_unchecked(rm) };
                macro_rules! elem4 {
                    ($ai:expr) => {{
                        let ai = $ai;
                        let gv = unsafe { *g4.get_unchecked(ai) };
                        if gv == i64::MIN || gv - c_rm <= bd { continue; }
                        let a1 = unsafe { *best_unused.get_unchecked(ai) };
                        let wa1 = (unsafe { *state.ch.weights.get_unchecked(a1) });
                        if wa1 >= budget { continue; }
                        let ca1 = (unsafe { *state.contrib.get_unchecked(a1) }) as i64;
                        if fast(4096) {
                            let t = ((budget - wa1) as usize).min(10);
                            let mqv = unsafe { *mq.get_unchecked(ai * 11 + t) };
                            if mqv == i64::MIN || ca1 + mqv - c_rm <= bd { continue; }
                        }
                        let base = ca1 - (unsafe { *row_rm.get_unchecked(a1) }) as i64 - c_rm;
                        let t0 = (unsafe { *fc4_off.get_unchecked(ai) }) as usize;
                        let t1 = (unsafe { *fc4_off.get_unchecked(ai + 1) }) as usize;
                        for t in t0..t1 {
                            let (a2u, wa2, q) = unsafe { *fc4.get_unchecked(t) };
                            if wa1 + wa2 > budget { continue; }
                            let a2 = a2u as usize;
                            let delta = base + q - (unsafe { *row_rm.get_unchecked(a2) }) as i64;
                            if delta > bd && wbase + wa1 + wa2 <= cap { bd = delta; bm = Some((rm, a1, a2)); }
                        }
                    }};
                }
                if use_pref4 {
                    let bi = (w_rm as usize) - 1;
                    let (os, oe) = (ord4_off[bi], ord4_off[bi + 1]);
                    let thr = bd + c_rm;
                    let mut lo2 = os;
                    let mut hi2 = oe;
                    while lo2 < hi2 {
                        let mid = lo2 + (hi2 - lo2) / 2;
                        if unsafe { *ord4v.get_unchecked(mid) } > thr { lo2 = mid + 1; } else { hi2 = mid; }
                    }
                    let m = lo2 - os;
                    if m == 0 { continue; }
                    let off = pre4_off[bi] + m * words4;
                    for w in 0..words4 {
                        let mut word = unsafe { *pre4.get_unchecked(off + w) };
                        while word != 0 {
                            let ai = w * 64 + word.trailing_zeros() as usize;
                            word &= word - 1;
                            elem4!(ai);
                        }
                    }
                } else {
                let mut b = 0usize;
                while b < g4_blk.len() {
                    let bmv = unsafe { *g4_blk.get_unchecked(b) };
                    let lo = b * GB;
                    let hi = (lo + GB).min(best_unused.len());
                    b += 1;
                    if bmv == i64::MIN || bmv - c_rm <= bd { continue; }
                    for ai in lo..hi {
                        elem4!(ai);
                    }
                }
                }
            }
            } else {
            g4.clear();
            for &a1 in &best_unused {
                let mut best = i64::MIN;
                for &a2 in (unsafe { tsn.friends.get_unchecked(a1) }) {
                    if (unsafe { *state.selected_bit.get_unchecked(a2) }) || a1 == a2 { continue; }
                    let v = (unsafe { *state.contrib.get_unchecked(a2) }) as i64
                        + (unsafe { *state.ch.interaction_values.get_unchecked(a1).get_unchecked(a2) }) as i64;
                    if v > best { best = v; }
                }
                g4.push(if best == i64::MIN { i64::MIN } else { (unsafe { *state.contrib.get_unchecked(a1) }) as i64 + best });
            }
            let mut g4max = i64::MIN;
            for &v in &g4 { if v > g4max { g4max = v; } }
            const GB: usize = 16;
            if fast(4) {
                g4_blk.clear();
                let mut q = 0usize;
                while q < g4.len() {
                    let e = (q + GB).min(g4.len());
                    let mut mx = i64::MIN;
                    for t in q..e { let v = unsafe { *g4.get_unchecked(t) }; if v > mx { mx = v; } }
                    g4_blk.push(mx);
                    q = e;
                }
            }
            for &rm in &worst_used {
                let c_rm = (unsafe { *state.contrib.get_unchecked(rm) }) as i64;
                if g4max == i64::MIN || g4max - c_rm <= bd { continue; }
                let w_rm = (unsafe { *state.ch.weights.get_unchecked(rm) });
                let budget = state.slack() + w_rm;
                if fast(4) {
                    let mut b = 0usize;
                    while b < g4_blk.len() {
                        let bmv = unsafe { *g4_blk.get_unchecked(b) };
                        let lo = b * GB;
                        let hi = (lo + GB).min(best_unused.len());
                        b += 1;
                        if bmv == i64::MIN || bmv - c_rm <= bd { continue; }
                        for ai in lo..hi {
                            let a1 = unsafe { *best_unused.get_unchecked(ai) };
                            if (unsafe { *g4.get_unchecked(ai) }) == i64::MIN || (unsafe { *g4.get_unchecked(ai) }) - c_rm <= bd { continue; }
                    let ca1 = (unsafe { *state.contrib.get_unchecked(a1) }) as i64;
                        let wa1 = (unsafe { *state.ch.weights.get_unchecked(a1) });
                        if wa1 >= budget { continue; }
                        let ca1_eff = ca1 - (unsafe { *state.ch.interaction_values.get_unchecked(a1).get_unchecked(rm) }) as i64;
                            for &a2 in (unsafe { tsn.friends.get_unchecked(a1) }) {
                            if (unsafe { *state.selected_bit.get_unchecked(a2) }) || a1 == a2 { continue; }
                            let wa2 = (unsafe { *state.ch.weights.get_unchecked(a2) });
                            if wa1 + wa2 > budget { continue; }
                            let delta = ca1_eff + (unsafe { *state.contrib.get_unchecked(a2) }) as i64
                                - (unsafe { *state.ch.interaction_values.get_unchecked(a2).get_unchecked(rm) }) as i64
                                + (unsafe { *state.ch.interaction_values.get_unchecked(a1).get_unchecked(a2) }) as i64 - c_rm;
                            if delta > bd && state.total_weight - w_rm + wa1 + wa2 <= cap {
                                bd = delta; bm = Some((rm, a1, a2));
                            }
                        }
                        }
                    }
                } else {
                for (ai, &a1) in best_unused.iter().enumerate() {
                    if g4[ai] == i64::MIN || g4[ai] - c_rm <= bd { continue; }
                    let ca1 = (unsafe { *state.contrib.get_unchecked(a1) }) as i64;
                    let wa1 = (unsafe { *state.ch.weights.get_unchecked(a1) });
                    if wa1 >= budget { continue; }
                    let ca1_eff = ca1 - (unsafe { *state.ch.interaction_values.get_unchecked(a1).get_unchecked(rm) }) as i64;
                    for &a2 in (unsafe { tsn.friends.get_unchecked(a1) }) {
                        if (unsafe { *state.selected_bit.get_unchecked(a2) }) || a1 == a2 { continue; }
                        let wa2 = (unsafe { *state.ch.weights.get_unchecked(a2) });
                        if wa1 + wa2 > budget { continue; }
                        let delta = ca1_eff + (unsafe { *state.contrib.get_unchecked(a2) }) as i64
                            - (unsafe { *state.ch.interaction_values.get_unchecked(a2).get_unchecked(rm) }) as i64
                            + (unsafe { *state.ch.interaction_values.get_unchecked(a1).get_unchecked(a2) }) as i64 - c_rm;
                        if delta > bd && state.total_weight - w_rm + wa1 + wa2 <= cap {
                            bd = delta; bm = Some((rm, a1, a2));
                        }
                    }
                }
                }
            }
            }
            if let Some((rm, a1, a2)) = bm {
                state.remove_item(rm); state.add_item(a1); state.add_item(a2); continue;
            }
        }

        {
            let cap = state.ch.max_weight;
            let mut bd = 0i64;
            let mut bm = None;
            if fast(256) {
            fc5.clear(); fc5_off.clear(); h5.clear();
            for (ri, &r1) in worst_used.iter().enumerate() {
                fc5_off.push(fc5.len() as u32);
                let row_r1 = unsafe { state.ch.interaction_values.get_unchecked(r1) };
                let _ = &row_r1;
                let mut best = i64::MIN;
                let nb = ri * 12;
                if fast(8192) { for t in 0..12 { unsafe { *mv.get_unchecked_mut(nb + t) = i64::MIN; } } }
                if usefv {
                    let fr = unsafe { tsn.friends.get_unchecked(r1) };
                    let fv = unsafe { tsn.vals.get_unchecked(r1) };
                    for (fi, &r2) in fr.iter().enumerate() {
                        if !(unsafe { *state.selected_bit.get_unchecked(r2) }) || r1 == r2 { continue; }
                        let v = (unsafe { *fv.get_unchecked(fi) }) as i64
                            - (unsafe { *state.contrib.get_unchecked(r2) }) as i64;
                        let wr2 = (unsafe { *state.ch.weights.get_unchecked(r2) });
                        fc5.push((r2 as u32, wr2, v));
                        if fast(8192) {
                            let sl = nb + (wr2 as usize).clamp(1, 10);
                            if v > unsafe { *mv.get_unchecked(sl) } { unsafe { *mv.get_unchecked_mut(sl) = v; } }
                        }
                        if v > best { best = v; }
                    }
                } else {
                for &r2 in (unsafe { tsn.friends.get_unchecked(r1) }) {
                    if !(unsafe { *state.selected_bit.get_unchecked(r2) }) || r1 == r2 { continue; }
                    let v = (unsafe { *row_r1.get_unchecked(r2) }) as i64
                        - (unsafe { *state.contrib.get_unchecked(r2) }) as i64;
                    let wr2 = (unsafe { *state.ch.weights.get_unchecked(r2) });
                    fc5.push((r2 as u32, wr2, v));
                    if fast(8192) {
                        let sl = nb + (wr2 as usize).clamp(1, 10);
                        if v > unsafe { *mv.get_unchecked(sl) } { unsafe { *mv.get_unchecked_mut(sl) = v; } }
                    }
                    if v > best { best = v; }
                }
                }
                if fast(8192) { for t in (0..10).rev() { if unsafe { *mv.get_unchecked(nb + t + 1) } > unsafe { *mv.get_unchecked(nb + t) } { unsafe { *mv.get_unchecked_mut(nb + t) = *mv.get_unchecked(nb + t + 1); } } } }
                h5.push(best);
            }
            fc5_off.push(fc5.len() as u32);
            u5.clear();
            let mut m5 = i64::MIN;
            for (ri, &r1) in worst_used.iter().enumerate() {
                if h5[ri] == i64::MIN { u5.push(i64::MIN); continue; }
                let v = h5[ri] - (unsafe { *state.contrib.get_unchecked(r1) }) as i64;
                u5.push(v);
                if v > m5 { m5 = v; }
            }
            const HB: usize = 16;
            let words5 = (u5.len() + 63) / 64;
            let use_pref5 = fast(1 << 28) && fast(8192) && words5 <= 8
                && MINW.load(std::sync::atomic::Ordering::Relaxed) >= 1
                && MAXW.load(std::sync::atomic::Ordering::Relaxed) <= 10;
            if use_pref5 {
                ord5v.clear(); ord5i.clear(); pre5.clear();
                for bi in 0..10usize {
                    ord5_off[bi] = ord5i.len();
                    pre5_off[bi] = pre5.len();
                    let need_b = (state.total_weight as i64) + bi as i64 + 1 - (cap as i64);
                    tmp_ord.clear();
                    for ri in 0..u5.len() {
                        if unsafe { *h5.get_unchecked(ri) } == i64::MIN { continue; }
                        let r1 = unsafe { *worst_used.get_unchecked(ri) };
                        let wr1 = unsafe { *state.ch.weights.get_unchecked(r1) };
                        let t = (need_b - wr1 as i64).max(0);
                        if t > 10 { continue; }
                        let mvv = unsafe { *mv.get_unchecked(ri * 12 + t as usize) };
                        if mvv == i64::MIN { continue; }
                        let cr1 = (unsafe { *state.contrib.get_unchecked(r1) }) as i64;
                        tmp_ord.push((mvv - cr1, ri as u32));
                    }
                    tmp_ord.sort_unstable_by(|a, b| b.0.cmp(&a.0));
                    let base = pre5.len();
                    pre5.resize(base + (tmp_ord.len() + 1) * words5, 0u64);
                    for k in 0..tmp_ord.len() {
                        let (v, ri) = unsafe { *tmp_ord.get_unchecked(k) };
                        ord5v.push(v); ord5i.push(ri);
                        let src = base + k * words5;
                        let dst = src + words5;
                        for w in 0..words5 { unsafe { *pre5.get_unchecked_mut(dst + w) = *pre5.get_unchecked(src + w); } }
                        unsafe { *pre5.get_unchecked_mut(dst + ri as usize / 64) |= 1u64 << (ri & 63); }
                    }
                }
                ord5_off[10] = ord5i.len();
                pre5_off[10] = pre5.len();
            } else {
            u5_blk.clear();
            {
                let mut q = 0usize;
                while q < u5.len() {
                    let e = (q + HB).min(u5.len());
                    let mut mx = i64::MIN;
                    for t in q..e { let v = unsafe { *u5.get_unchecked(t) }; if v > mx { mx = v; } }
                    u5_blk.push(mx);
                    q = e;
                }
            }
            }
            for &add in &best_unused {
                let c_add = (unsafe { *state.contrib.get_unchecked(add) }) as i64;
                let w_add = (unsafe { *state.ch.weights.get_unchecked(add) });
                if c_add <= 0 && bd > 0 { break; }
                if m5 == i64::MIN || c_add + m5 <= bd { continue; }
                let row_add = unsafe { state.ch.interaction_values.get_unchecked(add) };
                let wsum = state.total_weight + w_add;
                let need = (wsum as i64) - (cap as i64);
                macro_rules! elem5 {
                    ($ri:expr) => {{
                        let ri = $ri;
                        let hv = unsafe { *h5.get_unchecked(ri) };
                        let r1 = unsafe { *worst_used.get_unchecked(ri) };
                        let cr1 = (unsafe { *state.contrib.get_unchecked(r1) }) as i64;
                        if hv == i64::MIN || c_add - cr1 + hv <= bd { continue; }
                        let wr1 = (unsafe { *state.ch.weights.get_unchecked(r1) });
                        if fast(8192) {
                            let t = (need - wr1 as i64).max(0);
                            if t > 10 { continue; }
                            let mvv = unsafe { *mv.get_unchecked(ri * 12 + t as usize) };
                            if mvv == i64::MIN || c_add - cr1 + mvv <= bd { continue; }
                        }
                        let base = c_add - (unsafe { *row_add.get_unchecked(r1) }) as i64 - cr1;
                        let t0 = (unsafe { *fc5_off.get_unchecked(ri) }) as usize;
                        let t1 = (unsafe { *fc5_off.get_unchecked(ri + 1) }) as usize;
                        for t in t0..t1 {
                            let (r2u, wr2, v) = unsafe { *fc5.get_unchecked(t) };
                            if wsum <= cap + wr1 + wr2 {
                                let delta = base - (unsafe { *row_add.get_unchecked(r2u as usize) }) as i64 + v;
                                if delta > bd { bd = delta; bm = Some((r1, r2u as usize, add)); }
                            }
                        }
                    }};
                }
                if use_pref5 {
                    let bi = (w_add as usize) - 1;
                    let (os, oe) = (ord5_off[bi], ord5_off[bi + 1]);
                    let thr = bd - c_add;
                    let mut lo2 = os;
                    let mut hi2 = oe;
                    while lo2 < hi2 {
                        let mid = lo2 + (hi2 - lo2) / 2;
                        if unsafe { *ord5v.get_unchecked(mid) } > thr { lo2 = mid + 1; } else { hi2 = mid; }
                    }
                    let m = lo2 - os;
                    if m == 0 { continue; }
                    let off = pre5_off[bi] + m * words5;
                    for w in 0..words5 {
                        let mut word = unsafe { *pre5.get_unchecked(off + w) };
                        while word != 0 {
                            let ri = w * 64 + word.trailing_zeros() as usize;
                            word &= word - 1;
                            elem5!(ri);
                        }
                    }
                } else {
                let mut b = 0usize;
                while b < u5_blk.len() {
                    let bmv = unsafe { *u5_blk.get_unchecked(b) };
                    let lo = b * HB;
                    let hi = (lo + HB).min(worst_used.len());
                    b += 1;
                    if bmv == i64::MIN || c_add + bmv <= bd { continue; }
                    for ri in lo..hi {
                        elem5!(ri);
                    }
                }
                }
            }
            } else {
            h5.clear();
            for &r1 in &worst_used {
                let mut best = i64::MIN;
                for &r2 in (unsafe { tsn.friends.get_unchecked(r1) }) {
                    if !(unsafe { *state.selected_bit.get_unchecked(r2) }) || r1 == r2 { continue; }
                    let v = (unsafe { *state.ch.interaction_values.get_unchecked(r1).get_unchecked(r2) }) as i64
                        - (unsafe { *state.contrib.get_unchecked(r2) }) as i64;
                    if v > best { best = v; }
                }
                h5.push(best);
            }
            u5.clear();
            let mut m5 = i64::MIN;
            for (ri, &r1) in worst_used.iter().enumerate() {
                if h5[ri] == i64::MIN { u5.push(i64::MIN); continue; }
                let v = h5[ri] - (unsafe { *state.contrib.get_unchecked(r1) }) as i64;
                u5.push(v);
                if v > m5 { m5 = v; }
            }
            const HB: usize = 16;
            if fast(8) {
                u5_blk.clear();
                let mut q = 0usize;
                while q < u5.len() {
                    let e = (q + HB).min(u5.len());
                    let mut mx = i64::MIN;
                    for t in q..e { let v = unsafe { *u5.get_unchecked(t) }; if v > mx { mx = v; } }
                    u5_blk.push(mx);
                    q = e;
                }
            }
            for &add in &best_unused {
                let c_add = (unsafe { *state.contrib.get_unchecked(add) }) as i64;
                let w_add = (unsafe { *state.ch.weights.get_unchecked(add) });
                if c_add <= 0 && bd > 0 { break; }
                if m5 == i64::MIN || c_add + m5 <= bd { continue; }
                if fast(8) {
                    let mut b = 0usize;
                    while b < u5_blk.len() {
                        let bmv = unsafe { *u5_blk.get_unchecked(b) };
                        let lo = b * HB;
                        let hi = (lo + HB).min(worst_used.len());
                        b += 1;
                        if bmv == i64::MIN || c_add + bmv <= bd { continue; }
                        for ri in lo..hi {
                            let r1 = unsafe { *worst_used.get_unchecked(ri) };
                            let cr1 = (unsafe { *state.contrib.get_unchecked(r1) }) as i64;
                            if (unsafe { *h5.get_unchecked(ri) }) == i64::MIN || c_add - cr1 + (unsafe { *h5.get_unchecked(ri) }) <= bd { continue; }
                    let wr1 = (unsafe { *state.ch.weights.get_unchecked(r1) });
                        let c_add_r1 = (unsafe { *state.ch.interaction_values.get_unchecked(add).get_unchecked(r1) }) as i64;
                            for &r2 in (unsafe { tsn.friends.get_unchecked(r1) }) {
                            if !(unsafe { *state.selected_bit.get_unchecked(r2) }) || r1 == r2 { continue; }
                            let wr2 = (unsafe { *state.ch.weights.get_unchecked(r2) });
                            if state.total_weight + w_add <= cap + wr1 + wr2 {
                                let delta = c_add - c_add_r1
                                    - (unsafe { *state.ch.interaction_values.get_unchecked(add).get_unchecked(r2) }) as i64
                                    - cr1 - (unsafe { *state.contrib.get_unchecked(r2) }) as i64
                                    + (unsafe { *state.ch.interaction_values.get_unchecked(r1).get_unchecked(r2) }) as i64;
                                if delta > bd { bd = delta; bm = Some((r1, r2, add)); }
                            }
                        }
                        }
                    }
                } else {
                for (ri, &r1) in worst_used.iter().enumerate() {
                    let cr1 = (unsafe { *state.contrib.get_unchecked(r1) }) as i64;
                    if h5[ri] == i64::MIN || c_add - cr1 + h5[ri] <= bd { continue; }
                    let wr1 = (unsafe { *state.ch.weights.get_unchecked(r1) });
                    let c_add_r1 = (unsafe { *state.ch.interaction_values.get_unchecked(add).get_unchecked(r1) }) as i64;
                    for &r2 in (unsafe { tsn.friends.get_unchecked(r1) }) {
                        if !(unsafe { *state.selected_bit.get_unchecked(r2) }) || r1 == r2 { continue; }
                        let wr2 = (unsafe { *state.ch.weights.get_unchecked(r2) });
                        if state.total_weight + w_add <= cap + wr1 + wr2 {
                            let delta = c_add - c_add_r1
                                - (unsafe { *state.ch.interaction_values.get_unchecked(add).get_unchecked(r2) }) as i64
                                - cr1 - (unsafe { *state.contrib.get_unchecked(r2) }) as i64
                                + (unsafe { *state.ch.interaction_values.get_unchecked(r1).get_unchecked(r2) }) as i64;
                            if delta > bd { bd = delta; bm = Some((r1, r2, add)); }
                        }
                    }
                }
                }
            }
            }
            if let Some((r1, r2, add)) = bm {
                state.remove_item(r1); state.remove_item(r2); state.add_item(add); continue;
            }
        }

        {
            let cap = state.ch.max_weight;
            let mut bd = 0i64;
            let mut bm = None;
            let ks = worst_used.len().min(30);
            let ku = best_unused.len().min(30);
            if fast(256) {
            let mut g4max30 = i64::MIN;
            if fast(512) {
                for u in 0..ku { let v = unsafe { *g4.get_unchecked(u) }; if v > g4max30 { g4max30 = v; } }
            }
            for i in 0..ks {
                let r1 = unsafe { *worst_used.get_unchecked(i) };
                let cr1 = (unsafe { *state.contrib.get_unchecked(r1) }) as i64;
                let wr1 = (unsafe { *state.ch.weights.get_unchecked(r1) });
                let row_r1 = unsafe { state.ch.interaction_values.get_unchecked(r1) };
                for &r2 in (unsafe { tsn.friends.get_unchecked(r1) }) {
                    if !(unsafe { *state.selected_bit.get_unchecked(r2) }) || r1 == r2 { continue; }
                    let cr2 = (unsafe { *state.contrib.get_unchecked(r2) }) as i64;
                    let wr2 = (unsafe { *state.ch.weights.get_unchecked(r2) });
                    let lost = cr1 + cr2 - (unsafe { *row_r1.get_unchecked(r2) }) as i64;
                    if fast(512) && (g4max30 == i64::MIN || g4max30 - lost <= bd) { continue; }
                    let row_r2 = unsafe { state.ch.interaction_values.get_unchecked(r2) };
                    let budget = state.slack() + wr1 + wr2;
                    let wcap = cap + wr1 + wr2;
                    for u in 0..ku {
                        if fast(512) {
                            let gv = unsafe { *g4.get_unchecked(u) };
                            if gv == i64::MIN || gv - lost <= bd { continue; }
                        }
                        let a1 = unsafe { *best_unused.get_unchecked(u) };
                        let wa1 = (unsafe { *state.ch.weights.get_unchecked(a1) });
                        if wa1 >= budget { continue; }
                        let base = (unsafe { *state.contrib.get_unchecked(a1) }) as i64
                            - (unsafe { *row_r1.get_unchecked(a1) }) as i64
                            - (unsafe { *row_r2.get_unchecked(a1) }) as i64
                            - lost;
                        let t0 = (unsafe { *fc4_off.get_unchecked(u) }) as usize;
                        let t1 = (unsafe { *fc4_off.get_unchecked(u + 1) }) as usize;
                        for t in t0..t1 {
                            let (a2u, wa2, q) = unsafe { *fc4.get_unchecked(t) };
                            if wa1 + wa2 <= budget {
                                let a2 = a2u as usize;
                                let d = base + q
                                    - (unsafe { *row_r1.get_unchecked(a2) }) as i64
                                    - (unsafe { *row_r2.get_unchecked(a2) }) as i64;
                                if d > bd && state.total_weight + wa1 + wa2 <= wcap {
                                    bd = d; bm = Some((r1, r2, a1, a2));
                                }
                            }
                        }
                    }
                }
            }
            } else {
            for i in 0..ks {
                let r1 = worst_used[i];
                let cr1 = (unsafe { *state.contrib.get_unchecked(r1) }) as i64;
                let wr1 = (unsafe { *state.ch.weights.get_unchecked(r1) });
                for &r2 in (unsafe { tsn.friends.get_unchecked(r1) }) {
                    if !(unsafe { *state.selected_bit.get_unchecked(r2) }) || r1 == r2 { continue; }
                    let cr2 = (unsafe { *state.contrib.get_unchecked(r2) }) as i64;
                    let wr2 = (unsafe { *state.ch.weights.get_unchecked(r2) });
                    let lost = cr1 + cr2 - (unsafe { *state.ch.interaction_values.get_unchecked(r1).get_unchecked(r2) }) as i64;
                    let budget = state.slack() + wr1 + wr2;
                    for u in 0..ku {
                        let a1 = best_unused[u];
                        let wa1 = (unsafe { *state.ch.weights.get_unchecked(a1) });
                        if wa1 >= budget { continue; }
                        let ca1_eff = (unsafe { *state.contrib.get_unchecked(a1) }) as i64
                            - (unsafe { *state.ch.interaction_values.get_unchecked(a1).get_unchecked(r1) }) as i64
                            - (unsafe { *state.ch.interaction_values.get_unchecked(a1).get_unchecked(r2) }) as i64;
                        for &a2 in (unsafe { tsn.friends.get_unchecked(a1) }) {
                            if (unsafe { *state.selected_bit.get_unchecked(a2) }) || a1 == a2 { continue; }
                            let wa2 = (unsafe { *state.ch.weights.get_unchecked(a2) });
                            if wa1 + wa2 <= budget {
                                let gained = ca1_eff + (unsafe { *state.contrib.get_unchecked(a2) }) as i64
                                    - (unsafe { *state.ch.interaction_values.get_unchecked(a2).get_unchecked(r1) }) as i64
                                    - (unsafe { *state.ch.interaction_values.get_unchecked(a2).get_unchecked(r2) }) as i64
                                    + (unsafe { *state.ch.interaction_values.get_unchecked(a1).get_unchecked(a2) }) as i64;
                                if gained - lost > bd && state.total_weight + wa1 + wa2 <= cap + wr1 + wr2 {
                                    bd = gained - lost; bm = Some((r1, r2, a1, a2));
                                }
                            }
                        }
                    }
                }
            }
            }
            if let Some((r1, r2, a1, a2)) = bm {
                state.remove_item(r1); state.remove_item(r2);
                state.add_item(a1); state.add_item(a2); continue;
            }
        }

        if use_paths {
            let output = std::rc::Rc::new(exact_paths::Snapshot::new(&state.selected_bit,
                &state.contrib, state.total_value, state.total_weight, state.solution_hash));
            VND_PATHS.with(|c| c.borrow_mut().publish(wk * 4 + 1, pending, step + 1, output));
        }
        break;
    }
}

fn cluster_bomb_perturb(state: &mut State, tsn: &TopNeighbors, rng: &mut Rng, strength: usize) {
    let sel = state.selected_items();
    if sel.is_empty() { return; }
    let target = state.total_weight / (strength as u32).max(2);
    let mut freed = 0u32;
    let root = sel[rng.next_usize(sel.len())];
    state.remove_item(root);
    freed += (unsafe { *state.ch.weights.get_unchecked(root) });
    for &f in (unsafe { tsn.friends.get_unchecked(root) }) {
        if (unsafe { *state.selected_bit.get_unchecked(f) }) {
            state.remove_item(f);
            freed += (unsafe { *state.ch.weights.get_unchecked(f) });
            if freed >= target { break; }
        }
    }
    let slack = state.slack();
    if slack > 0 {
        let unsel: Vec<usize> = (0..state.ch.num_items)
            .filter(|&i| !(unsafe { *state.selected_bit.get_unchecked(i) }) && (unsafe { *state.ch.weights.get_unchecked(i) }) <= slack)
            .collect();
        if !unsel.is_empty() { state.add_item(unsel[rng.next_usize(unsel.len())]); }
    }
}

struct Hparams {
    n_random_starts: usize,
    n_crossover_gen: usize,
    ils_rounds: usize,
    ils_restart_interval: usize,
    perturb_base_frac: usize,
    perturb_max_frac: usize,
    n_full_restarts: usize,
    use_hub_pair: bool,
    use_heavy_polish: bool,
    window_k: usize,
    core_half_dp: usize,
    restart_stall_lim: usize,
    stall_cap: usize,
    back_to_best: usize,
    seed_every: usize,
    seed_ruin_div: usize,
    sparse_contrib: usize,
    fastmask: usize,
    stage_stop: usize,
}

impl Hparams {
    fn defaults() -> Self {
        Self {
            n_random_starts: 3, n_crossover_gen: 3, ils_rounds: 150,
            ils_restart_interval: 11, perturb_base_frac: 6,
            perturb_max_frac: 4, n_full_restarts: 24, use_hub_pair: false,
            use_heavy_polish: false, window_k: 179, core_half_dp: 41,
            restart_stall_lim: 20, stall_cap: 20, back_to_best: 1,
            seed_every: 1, seed_ruin_div: 24, sparse_contrib: 1, fastmask: 562949953421311, stage_stop: 0,
        }
    }

    fn from_map(h: &Option<Map<String, Value>>) -> Self {
        let mut p = Self::defaults();
        if let Some(m) = h {
            if let Some(v) = m.get("n_random_starts").and_then(|v| v.as_u64()) { p.n_random_starts = v as usize; }
            if let Some(v) = m.get("n_crossover_gen").and_then(|v| v.as_u64()) { p.n_crossover_gen = v as usize; }
            if let Some(v) = m.get("ils_rounds").and_then(|v| v.as_u64()) { p.ils_rounds = v as usize; }
            if let Some(v) = m.get("ils_restart_interval").and_then(|v| v.as_u64()) { p.ils_restart_interval = v as usize; }
            if let Some(v) = m.get("perturb_base_frac").and_then(|v| v.as_u64()) { p.perturb_base_frac = v as usize; }
            if let Some(v) = m.get("perturb_max_frac").and_then(|v| v.as_u64()) { p.perturb_max_frac = v as usize; }
            if let Some(v) = m.get("n_full_restarts").and_then(|v| v.as_u64()) { p.n_full_restarts = v as usize; }
            if let Some(v) = m.get("window_k").and_then(|v| v.as_u64()) { p.window_k = v as usize; }
            if let Some(v) = m.get("core_half_dp").and_then(|v| v.as_u64()) { p.core_half_dp = v as usize; }
            if let Some(v) = m.get("restart_stall_lim").and_then(|v| v.as_u64()) { p.restart_stall_lim = v as usize; }
            if let Some(v) = m.get("stall_cap").and_then(|v| v.as_u64()) { p.stall_cap = v as usize; }
            if let Some(v) = m.get("back_to_best").and_then(|v| v.as_u64()) { p.back_to_best = v as usize; }
            if let Some(v) = m.get("seed_every").and_then(|v| v.as_u64()) { p.seed_every = (v as usize).max(1); }
            if let Some(v) = m.get("seed_ruin_div").and_then(|v| v.as_u64()) { p.seed_ruin_div = (v as usize).max(1); }
            if let Some(v) = m.get("sparse_contrib").and_then(|v| v.as_u64()) { p.sparse_contrib = v as usize; }
            if let Some(v) = m.get("fastmask").and_then(|v| v.as_u64()) { p.fastmask = v as usize; }
            if let Some(v) = m.get("stage_stop").and_then(|v| v.as_u64()) { p.stage_stop = v as usize; }
        }
        p
    }
}

fn local_search_vnd_tsn_light(state: &mut State, tsn: &TopNeighbors) {
    let n = state.ch.num_items;
    let wk: usize = 189;
    let ch_ref: &Challenge = state.ch;
    let mut windows = exact_windows::Windows::new(&ch_ref.weights);
    let use_win = fast(1 << 18) && windows.is_packed() && exact_sort::compatible();
    let use_paths = fast(1 << 19);
    let usefv = fast(1 << 26);
    let mut pending: Vec<(exact_paths::Snapshot, usize)> = Vec::new();
    const LDIV: [i64; 11] = [2520, 2520, 1260, 840, 630, 504, 420, 360, 315, 280, 252];
    let mut unused_buf: Vec<(usize, i64)> = Vec::with_capacity(n);
    let mut used_buf: Vec<(usize, i64)> = Vec::with_capacity(n);
    let mut best_unused: Vec<usize> = Vec::with_capacity(wk);
    let mut worst_used: Vec<usize> = Vec::with_capacity(wk);
    let mut cand_live: Vec<usize> = Vec::with_capacity(wk);
    let mut cl_blk: Vec<i32> = Vec::with_capacity(wk / 16 + 2);
    for step in 0..40 {
        let step_hash = state.solution_hash;
        if use_paths {
            let hit = VND_PATHS.with(|c| c.borrow().lookup(wk * 2, step_hash, &state.selected_bit,
                &state.contrib, state.total_value, state.total_weight, 40 - step));
            if let Some((output, steps)) = hit {
                state.selected_bit.clone_from(&output.bits);
                state.contrib.clone_from(&output.contrib);
                state.total_value = output.value;
                state.total_weight = output.weight;
                state.solution_hash = output.hash;
                VND_PATHS.with(|c| c.borrow_mut().publish(wk * 2, pending, step + steps, output));
                return;
            }
        }
        unused_buf.clear();
        used_buf.clear();
        best_unused.clear();
        worst_used.clear();
        if use_win {
            windows.build(&state.contrib, &state.selected_bit, wk, &mut best_unused, &mut worst_used);
        } else {
        if fast(1) {
            unsafe {
                let cp = state.contrib.as_ptr();
                let wp = state.ch.weights.as_ptr();
                let bp = state.selected_bit.as_ptr();
                let up = unused_buf.as_mut_ptr();
                let sp = used_buf.as_mut_ptr();
                let mut ui = 0usize;
                let mut si = 0usize;
                let mut i = 0usize;
                while i + 4 <= n {
                    let c0 = *cp.add(i) as i64;
                    let w0 = (*wp.add(i) as i64).max(1);
                    let k0 = c0 * *LDIV.get_unchecked((w0 as usize).min(10));
                    let s0 = *bp.add(i);
                    (if s0 { sp.add(si) } else { up.add(ui) }).write((i, k0));
                    ui += (!s0) as usize; si += s0 as usize;
                    let c1 = *cp.add(i + 1) as i64;
                    let w1 = (*wp.add(i + 1) as i64).max(1);
                    let k1 = c1 * *LDIV.get_unchecked((w1 as usize).min(10));
                    let s1 = *bp.add(i + 1);
                    (if s1 { sp.add(si) } else { up.add(ui) }).write((i + 1, k1));
                    ui += (!s1) as usize; si += s1 as usize;
                    let c2 = *cp.add(i + 2) as i64;
                    let w2 = (*wp.add(i + 2) as i64).max(1);
                    let k2 = c2 * *LDIV.get_unchecked((w2 as usize).min(10));
                    let s2 = *bp.add(i + 2);
                    (if s2 { sp.add(si) } else { up.add(ui) }).write((i + 2, k2));
                    ui += (!s2) as usize; si += s2 as usize;
                    let c3 = *cp.add(i + 3) as i64;
                    let w3 = (*wp.add(i + 3) as i64).max(1);
                    let k3 = c3 * *LDIV.get_unchecked((w3 as usize).min(10));
                    let s3 = *bp.add(i + 3);
                    (if s3 { sp.add(si) } else { up.add(ui) }).write((i + 3, k3));
                    ui += (!s3) as usize; si += s3 as usize;
                    i += 4;
                }
                while i < n {
                    let c0 = *cp.add(i) as i64;
                    let w0 = (*wp.add(i) as i64).max(1);
                    let k0 = c0 * *LDIV.get_unchecked((w0 as usize).min(10));
                    let s0 = *bp.add(i);
                    (if s0 { sp.add(si) } else { up.add(ui) }).write((i, k0));
                    ui += (!s0) as usize; si += s0 as usize;
                    i += 1;
                }
                unused_buf.set_len(ui);
                used_buf.set_len(si);
            }
        } else {
            for i in 0..n {
                let (c, w, sel) = unsafe {
                    (*state.contrib.get_unchecked(i) as i64,
                     (*state.ch.weights.get_unchecked(i) as i64).max(1),
                     *state.selected_bit.get_unchecked(i))
                };
                let key = c * unsafe { *LDIV.get_unchecked((w as usize).min(10)) };
                if sel { used_buf.push((i, key)); } else { unused_buf.push((i, key)); }
            }
        }
        let ku = wk.min(unused_buf.len());
        if ku > 0 && ku < unused_buf.len() {
            unused_buf.select_nth_unstable_by(ku - 1, |a, b| b.1.cmp(&a.1));
        }
        let ks = wk.min(used_buf.len());
        if ks > 0 && ks < used_buf.len() {
            used_buf.select_nth_unstable_by(ks - 1, |a, b| a.1.cmp(&b.1));
        }
        for t in &unused_buf[..ku] { best_unused.push(t.0); }
        for t in &used_buf[..ks] { worst_used.push(t.0); }
        }

        let slack = state.slack();
        if slack > 0 {
            let mut ba: Option<(usize, i32)> = None;
            for &c in &best_unused {
                if (unsafe { *state.ch.weights.get_unchecked(c) }) > slack { continue; }
                let d = (unsafe { *state.contrib.get_unchecked(c) });
                if d > 0 && ba.map_or(true, |(_, bd)| d > bd) { ba = Some((c, d)); }
            }
            if let Some((c, _)) = ba { state.add_item(c); continue; }
        }

        {
            let mut bs: Option<(usize, usize, i32)> = None;
            let mut cmax = [i32::MIN; 11];
            if fast(1 << 34) {
                for &c in &best_unused {
                    let cw = ((unsafe { *state.ch.weights.get_unchecked(c) }) as usize).clamp(1, 10);
                    let v = unsafe { *state.contrib.get_unchecked(c) };
                    let slot = unsafe { cmax.get_unchecked_mut(cw) };
                    *slot = if v > *slot { v } else { *slot };
                }
            } else {
            for &c in &best_unused {
                let cw = ((unsafe { *state.ch.weights.get_unchecked(c) }) as usize).clamp(1, 10);
                if (unsafe { *state.contrib.get_unchecked(c) }) > cmax[cw] { cmax[cw] = (unsafe { *state.contrib.get_unchecked(c) }); }
            }
            }
            for w in 1..=10 { if cmax[w - 1] > cmax[w] { cmax[w] = cmax[w - 1]; } }
            let mut rmin = [i32::MAX; 12];
            if fast(1 << 34) {
                for &rm in &worst_used {
                    let rw = ((unsafe { *state.ch.weights.get_unchecked(rm) }) as usize).clamp(1, 10);
                    let v = unsafe { *state.contrib.get_unchecked(rm) };
                    let slot = unsafe { rmin.get_unchecked_mut(rw) };
                    *slot = if v < *slot { v } else { *slot };
                }
            } else {
            for &rm in &worst_used {
                let rw = ((unsafe { *state.ch.weights.get_unchecked(rm) }) as usize).clamp(1, 10);
                if (unsafe { *state.contrib.get_unchecked(rm) }) < rmin[rw] { rmin[rw] = (unsafe { *state.contrib.get_unchecked(rm) }); }
            }
            }
            for w in (1..=10).rev() { if rmin[w + 1] < rmin[w] { rmin[w] = rmin[w + 1]; } }
            let slack_c = state.slack() as i32;
            cand_live.clear();
            if fast(1 << 34) {
                unsafe {
                    let cp = cand_live.as_mut_ptr();
                    let mut kk = 0usize;
                    for &c in &best_unused {
                        let wc = (*state.ch.weights.get_unchecked(c)) as i32;
                        let lo = (wc - slack_c).max(1).min(10) as usize;
                        cp.add(kk).write(c);
                        kk += ((*state.contrib.get_unchecked(c)) > *rmin.get_unchecked(lo)) as usize;
                    }
                    cand_live.set_len(kk);
                }
            } else {
            for &c in &best_unused {
                let wc = (unsafe { *state.ch.weights.get_unchecked(c) }) as i32;
                let lo = (wc - slack_c).max(1).min(10) as usize;
                if (unsafe { *state.contrib.get_unchecked(c) }) > rmin[lo] { cand_live.push(c); }
            }
            }
            const CB: usize = 16;
            if fast(2) {
                cl_blk.clear();
                let mut q = 0usize;
                while q < cand_live.len() {
                    let e = (q + CB).min(cand_live.len());
                    let mut mx = i32::MIN;
                    for t in q..e {
                        let v = unsafe { *state.contrib.get_unchecked(*cand_live.get_unchecked(t)) };
                        if v > mx { mx = v; }
                    }
                    cl_blk.push(mx);
                    q = e;
                }
            }
            for &rm in &worst_used {
                let max_w = (unsafe { *state.ch.weights.get_unchecked(rm) }) + state.slack();
                if fast(1 << 40) {
                    let thr0 = bs.map_or(0i64, |(_, _, d)| d as i64);
                    if (cmax[(max_w as usize).clamp(1, 10)] as i64)
                        - (unsafe { *state.contrib.get_unchecked(rm) }) as i64 <= thr0 { continue; }
                } else if cmax[(max_w as usize).clamp(1, 10)] <= (unsafe { *state.contrib.get_unchecked(rm) }) { continue; }
                let row_rm = unsafe { state.ch.interaction_values.get_unchecked(rm) };
                if fast(2) {
                    let crm = (unsafe { *state.contrib.get_unchecked(rm) }) as i64;
                    let mut b = 0usize;
                    while b < cl_blk.len() {
                        let bmv = (unsafe { *cl_blk.get_unchecked(b) }) as i64;
                        let lo = b * CB;
                        let hi = (lo + CB).min(cand_live.len());
                        b += 1;
                        let thr = bs.map_or(0i64, |(_, _, d)| d as i64);
                        if bmv - crm <= thr { continue; }
                        if fast(1 << 33) {
                            let mut bd0 = bs.map_or(0i32, |(_, _, d)| d);
                            let crm_i = unsafe { *state.contrib.get_unchecked(rm) };
                            for t in lo..hi {
                                let c = unsafe { *cand_live.get_unchecked(t) };
                                let ok = (unsafe { *state.ch.weights.get_unchecked(c) }) <= max_w;
                                let d = (unsafe { *state.contrib.get_unchecked(c) }) - crm_i
                                    - unsafe { *row_rm.get_unchecked(c) };
                                if ok & (d > bd0) { bs = Some((c, rm, d)); bd0 = d; }
                            }
                        } else {
                        for t in lo..hi {
                            let c = unsafe { *cand_live.get_unchecked(t) };
                            if (unsafe { *state.ch.weights.get_unchecked(c) }) > max_w { continue; }
                            let d = (unsafe { *state.contrib.get_unchecked(c) }) - (unsafe { *state.contrib.get_unchecked(rm) }) - unsafe { *row_rm.get_unchecked(c) };
                            if d > 0 && bs.map_or(true, |(_, _, bd)| d > bd) { bs = Some((c, rm, d)); }
                        }
                        }
                    }
                } else {
                    for &c in &cand_live {
                        if (unsafe { *state.ch.weights.get_unchecked(c) }) > max_w { continue; }
                        let d = (unsafe { *state.contrib.get_unchecked(c) }) - (unsafe { *state.contrib.get_unchecked(rm) }) - unsafe { *row_rm.get_unchecked(c) };
                        if d > 0 && bs.map_or(true, |(_, _, bd)| d > bd) { bs = Some((c, rm, d)); }
                    }
                }
            }
            if let Some((c, rm, _)) = bs { state.replace_item(rm, c); continue; }
        }

        {
            let slack_i = state.slack() as i32;
            if slack_i >= 2 {
                let mut bd = 0i64;
                let mut bp = None;
                for &a1 in &best_unused {
                    let ca1 = (unsafe { *state.contrib.get_unchecked(a1) }) as i64;
                    if ca1 <= 0 && bd > 0 { break; }
                    let wa1 = (unsafe { *state.ch.weights.get_unchecked(a1) }) as i32;
                    if wa1 >= slack_i { continue; }
                    if usefv {
                        let fr = unsafe { tsn.friends.get_unchecked(a1) };
                        let fv = unsafe { tsn.vals.get_unchecked(a1) };
                        for (fi, &a2) in fr.iter().enumerate() {
                            if (unsafe { *state.selected_bit.get_unchecked(a2) }) || a1 == a2 { continue; }
                            if wa1 + ((unsafe { *state.ch.weights.get_unchecked(a2) }) as i32) <= slack_i {
                                let delta = ca1 + (unsafe { *state.contrib.get_unchecked(a2) }) as i64 + (unsafe { *fv.get_unchecked(fi) }) as i64;
                                if delta > bd { bd = delta; bp = Some((a1, a2)); }
                            }
                        }
                    } else {
                    for &a2 in (unsafe { tsn.friends.get_unchecked(a1) }) {
                        if (unsafe { *state.selected_bit.get_unchecked(a2) }) || a1 == a2 { continue; }
                        if wa1 + ((unsafe { *state.ch.weights.get_unchecked(a2) }) as i32) <= slack_i {
                            let delta = ca1 + (unsafe { *state.contrib.get_unchecked(a2) }) as i64 + (unsafe { *state.ch.interaction_values.get_unchecked(a1).get_unchecked(a2) }) as i64;
                            if delta > bd { bd = delta; bp = Some((a1, a2)); }
                        }
                    }
                    }
                }
                if let Some((a1, a2)) = bp { state.add_item(a1); state.add_item(a2); continue; }
            }
        }

        if use_paths {
            let output = std::rc::Rc::new(exact_paths::Snapshot::new(&state.selected_bit,
                &state.contrib, state.total_value, state.total_weight, state.solution_hash));
            VND_PATHS.with(|c| c.borrow_mut().publish(wk * 2, pending, step + 1, output));
        }
        break;
    }
}

fn vnd_v2(state: &mut State, hp: &Hparams, tsn: Option<&TopNeighbors>) {
    if let Some(t) = tsn {
        local_search_vnd_tsn(state, t);
    } else if hp.window_k < state.ch.num_items {
        local_search_vnd_windowed(state, hp.window_k);
    } else {
        ils_vnd(state);
    }
}

fn polish_v2(state: &mut State, hp: &Hparams, tsn: Option<&TopNeighbors>) {
    if let Some(t) = tsn {
        local_search_vnd_tsn(state, t);
    } else if hp.use_heavy_polish {
        local_search_vnd_heavy(state);
    } else {
        vnd_v2(state, hp, None);
    }
}

struct ReactivePerturbSchedule {
    rewards: [i64; 8],
    uses: [usize; 8],
    cursor: usize,
    polish_stalls: usize,
}

fn basin_signature(state: &State) -> u64 {
    let n = state.ch.num_items;
    let mut selected_count = 0usize;
    let mut sample: Vec<(i64, usize)> = Vec::with_capacity(24);
    if fast(1 << 35) && fast(1 << 30) {
        let mut thr: i64 = 0;
        let mut full = false;
        let sb = state.selected_bit.as_ptr();
        let wp = state.ch.weights.as_ptr();
        let cp = state.contrib.as_ptr();
        macro_rules! pred {
            ($i:expr) => {{
                let (sel, weight, q) = unsafe {
                    (*sb.add($i), (*wp.add($i) as i64).max(1), *cp.add($i) as i64 * 1000)
                };
                let s1 = (q >= 0) as i64;
                let over = q >= (thr + s1) * weight + (1 - s1);
                (sel as usize, sel & (!full | over))
            }};
        }
        macro_rules! one {
            ($i:expr) => {{
                let i = $i;
                let (_, hit) = pred!(i);
                if hit {
                    let weight = ((unsafe { *wp.add(i) }) as i64).max(1);
                    let q = (unsafe { *cp.add(i) }) as i64 * 1000;
                    let score = q / weight;
                    let pos = sample.iter().position(|&(prior, _)| score > prior)
                        .unwrap_or(sample.len());
                    if pos < 24 {
                        sample.insert(pos, (score, i));
                        if sample.len() > 24 { sample.pop(); }
                    }
                    full = sample.len() == 24;
                    thr = if full { sample[23].0 } else { 0 };
                }
            }};
        }
        let mut i = 0usize;
        let end = n / 8 * 8;
        while i < end {
            let (a0, h0) = pred!(i);
            let (a1, h1) = pred!(i + 1);
            let (a2, h2) = pred!(i + 2);
            let (a3, h3) = pred!(i + 3);
            let (a4, h4) = pred!(i + 4);
            let (a5, h5) = pred!(i + 5);
            let (a6, h6) = pred!(i + 6);
            let (a7, h7) = pred!(i + 7);
            selected_count += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
            if h0 | h1 | h2 | h3 | h4 | h5 | h6 | h7 {
                for j in i..i + 8 { one!(j); }
            }
            i += 8;
        }
        while i < n {
            let (a, _) = pred!(i);
            selected_count += a;
            one!(i);
            i += 1;
        }
    } else if fast(1 << 30) {
        let mut thr: i64 = 0;
        let mut full = false;
        for i in 0..n {
            let sel = unsafe { *state.selected_bit.get_unchecked(i) };
            selected_count += sel as usize;
            let weight = ((unsafe { *state.ch.weights.get_unchecked(i) }) as i64).max(1);
            let q = (unsafe { *state.contrib.get_unchecked(i) }) as i64 * 1000;
            let s1 = (q >= 0) as i64;
            let over = q >= (thr + s1) * weight + (1 - s1);
            if sel & (!full | over) {
                let score = q / weight;
                let pos = sample.iter().position(|&(prior, _)| score > prior)
                    .unwrap_or(sample.len());
                if pos < 24 {
                    sample.insert(pos, (score, i));
                    if sample.len() > 24 { sample.pop(); }
                }
                full = sample.len() == 24;
                thr = if full { sample[23].0 } else { 0 };
            }
        }
    } else {
    for i in 0..n {
        if !(unsafe { *state.selected_bit.get_unchecked(i) }) {
            continue;
        }
        selected_count += 1;
        let weight = ((unsafe { *state.ch.weights.get_unchecked(i) }) as i64).max(1);
        let score = (unsafe { *state.contrib.get_unchecked(i) }) as i64 * 1000 / weight;
        if sample.len() == 24 && score <= sample[23].0 { continue; }
        let pos = sample.iter().position(|&(prior, _)| score > prior)
            .unwrap_or(sample.len());
        if pos < 24 {
            sample.insert(pos, (score, i));
            if sample.len() > 24 {
                sample.pop();
            }
        }
    }
    }

    let mut edges: Vec<(i32, usize, usize)> = Vec::with_capacity(8);
    for ai in 0..sample.len() {
        let a = sample[ai].1;
        for bi in (ai + 1)..sample.len() {
            let b = sample[bi].1;
            let interaction = (unsafe { *state.ch.interaction_values.get_unchecked(a).get_unchecked(b) });
            if interaction <= 0 {
                continue;
            }
            let pos = edges.iter().position(|&(prior, _, _)| interaction > prior)
                .unwrap_or(edges.len());
            if pos < 8 {
                edges.insert(pos, (interaction, a.min(b), a.max(b)));
                if edges.len() > 8 {
                    edges.pop();
                }
            }
        }
    }

    let capacity_band = if state.ch.max_weight > 0 {
        (state.total_weight as u64 * 16 / state.ch.max_weight as u64).min(16)
    } else {
        0
    };
    let mut signature = (selected_count as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ capacity_band.wrapping_mul(0xBF58476D1CE4E5B9);
    for (interaction, a, b) in edges {
        let edge = ((a as u64) << 32) ^ b as u64
            ^ (interaction as u32 as u64).wrapping_mul(0x94D049BB133111EB);
        signature ^= edge.rotate_left(17);
        signature = signature.rotate_left(11).wrapping_mul(0x9E3779B185EBCA87);
    }
    signature
}

impl ReactivePerturbSchedule {
    fn new() -> Self {
        Self {
            rewards: [0; 8],
            uses: [0; 8],
            cursor: 0,
            polish_stalls: 0,
        }
    }

    fn choose(&mut self) -> usize {
        for offset in 0..8 {
            let strategy = (self.cursor + offset) % 8;
            if self.uses[strategy] == 0 {
                self.uses[strategy] = 1;
                self.cursor = (strategy + 1) % 8;
                return strategy;
            }
        }

        let mut best = self.cursor;
        for strategy in 0..8 {
            let lhs = self.rewards[strategy] as i128 * self.uses[best] as i128;
            let rhs = self.rewards[best] as i128 * self.uses[strategy] as i128;
            if lhs > rhs || (lhs == rhs && strategy == self.cursor) {
                best = strategy;
            }
        }
        self.uses[best] += 1;
        self.cursor = (best + 1) % 8;
        best
    }

    fn note_perturb(&mut self, strategy: usize, before: i64, after: i64, repeated: bool) {
        let gain = (after - before).max(0);
        self.rewards[strategy] = self.rewards[strategy].saturating_mul(3) / 4 + gain;
        if repeated {
            self.rewards[strategy] = self.rewards[strategy].saturating_mul(3) / 4;
        }
    }

    fn note_polish(&mut self, before: i64, after: i64) {
        if after > before {
            self.polish_stalls = 0;
        } else {
            self.polish_stalls += 1;
        }
    }
}

fn build_det_seeds(challenge: &Challenge, hp: &Hparams, tsn_ref: Option<&TopNeighbors>, total_interactions: &[i64], csr: Option<&Csr>) -> Vec<SolState> {
    let ch = hp.core_half_dp;
    let mut out = Vec::with_capacity(3);
    for variant in 0..3 {
        let mut st = State::new_empty_with(challenge, csr);
        match variant {
            0 => build_greedy_density_from_all(&mut st, total_interactions),
            1 => build_greedy_value(&mut st),
            _ => build_greedy_synergy_weight(&mut st, total_interactions),
        }
        dp_refinement_hp(&mut st, ch);
        polish_v2(&mut st, hp, tsn_ref);
        out.push(st.clone_solution());
    }
    out
}

fn run_one_instance(challenge: &Challenge, hp: &Hparams, rng_offset: usize, shared_tsn: Option<&TopNeighbors>, total_interactions: &[i64], det_seeds: Option<&[SolState]>, csr: Option<&Csr>) -> (Solution, i64) {
    run_one_instance_seeded(challenge, hp, rng_offset, shared_tsn, total_interactions, None, det_seeds, csr)
}

fn run_one_instance_seeded(challenge: &Challenge, hp: &Hparams, rng_offset: usize, shared_tsn: Option<&TopNeighbors>, total_interactions: &[i64], seed_sol: Option<&Solution>, det_seeds: Option<&[SolState]>, csr: Option<&Csr>) -> (Solution, i64) {
    let n = challenge.num_items;
    let mut rng = Rng::from_seed(&challenge.seed);
    for _ in 0..rng_offset * 100 { rng.next_u32(); }
    let ch = hp.core_half_dp;

    let tsn_opt: Option<TopNeighbors> = if shared_tsn.is_none() && n > 1200 {
        Some(TopNeighbors::new(challenge, 12, csr))
    } else { None };
    let tsn_ref = shared_tsn.or(tsn_opt.as_ref());

    let mut population: Vec<SolState> = Vec::with_capacity(16);

    if let Some(seed) = seed_sol {
        let mut st = State::new_empty_with(challenge, csr);
        for &i in &seed.items { st.add_item(i); }
        let sel = st.selected_items();
        let n_remove = (sel.len() / hp.seed_ruin_div).max(2).min(sel.len());
        let mut scored: Vec<(usize, i64)> = sel.iter().map(|&i| {
            let w = (challenge.weights[i] as i64).max(1);
            (i, dw(st.contrib[i] as i64 * 1000, w))
        }).collect();
        scored.sort_unstable_by_key(|&(_, s)| s);
        for j in 0..n_remove { st.remove_item(scored[j].0); }
        greedy_reconstruct(&mut st, rng_offset % 6, total_interactions);
        dp_refinement_hp(&mut st, ch);
        vnd_v2(&mut st, hp, tsn_ref);
        population.push(st.clone_solution());
    }

    let n_greedy = 4;
    let n_rand = if seed_sol.is_some() { hp.n_random_starts.saturating_sub(1) } else { hp.n_random_starts };
    for variant in 0..n_greedy {
        if variant < 3 {
            if let Some(ds) = det_seeds {
                population.push(ds[variant].clone());
                continue;
            }
        }
        let mut st = State::new_empty_with(challenge, csr);
        match variant {
            0 => build_greedy_density_from_all(&mut st, total_interactions),
            1 => build_greedy_value(&mut st),
            2 => build_greedy_synergy_weight(&mut st, total_interactions),
            3 => build_synergy_supernodes(&mut st, total_interactions, &mut rng),
            _ => build_greedy_hub(&mut st, total_interactions),
        }
        dp_refinement_hp(&mut st, ch);
        polish_v2(&mut st, hp, tsn_ref);
        population.push(st.clone_solution());
    }

    let ctor_is_noop = challenge.values.iter().all(|&v| v == 0);
    let mut rand_member: Option<SolState> = None;
    let g4 = fast(1 << 38) && ctor_is_noop;
    for mode in 4..(4 + n_rand) {
        if ctor_is_noop {
            if let Some(m0) = rand_member.as_ref() {
                population.push(m0.clone());
                continue;
            }
        }
        if g4 && mode == 4 {
            if let Some(m0) = RAND_MEMBER.with(|c| c.borrow().clone()) {
                population.push(m0.clone());
                rand_member = Some(m0);
                continue;
            }
        }
        let mut st = State::new_empty_with(challenge, csr);
        let m = if mode < 6 { mode } else { mode - 2 };
        construct_forward_incremental(&mut st, m, &mut rng);
        dp_refinement_hp(&mut st, ch);
        vnd_v2(&mut st, hp, tsn_ref);
        let sol = st.clone_solution();
        if ctor_is_noop { rand_member = Some(sol.clone()); }
        if g4 && mode == 4 { RAND_MEMBER.with(|c| *c.borrow_mut() = Some(sol.clone())); }
        population.push(sol);
    }

    if hp.use_hub_pair {
        for k in 0..4 {
            let mut st = State::new_empty_with(challenge, csr);
            build_hub_pair_kth(&mut st, k);
            dp_refinement_hp(&mut st, ch);
            vnd_v2(&mut st, hp, tsn_ref);
            population.push(st.clone_solution());
        }
    }

    population.sort_unstable_by_key(|s| std::cmp::Reverse(s.value));
    population.truncate(8);

    let mut state = State::new_empty_with(challenge, csr);
    let mut xo_stall = 0usize;
    for gen in 0..hp.n_crossover_gen {
        let best_before = population[0].value;
        let child_bits = crossover_frequency(&population, challenge, &mut rng);
        set_state_from_bits(&mut state, &child_bits);
        dp_refinement_hp(&mut state, ch);
        vnd_v2(&mut state, hp, tsn_ref);
        population.push(state.clone_solution());

        if population.len() >= 2 {
            let a = gen % population.len().min(4);
            let b = (gen + 1) % population.len().min(4);
            if a != b {
                let child_bits = crossover_uniform(&population[a], &population[b], challenge, &mut rng);
                set_state_from_bits(&mut state, &child_bits);
                dp_refinement_hp(&mut state, ch);
                vnd_v2(&mut state, hp, tsn_ref);
                population.push(state.clone_solution());

                let parent_a = population[a].clone();
                let parent_b = population[b].clone();
                state = State::new_empty_with(challenge, csr);
                if gen % 2 == 0 {
                    crossover_residual_modules(&mut state, &parent_a, &parent_b);
                } else {
                    crossover_residual_modules(&mut state, &parent_b, &parent_a);
                }
                dp_refinement_hp(&mut state, ch);
                vnd_v2(&mut state, hp, tsn_ref);
                population.push(state.clone_solution());

                state = State::new_empty_with(challenge, csr);
                crossover_linkage(&mut state, &parent_a, &parent_b);
                dp_refinement_hp(&mut state, ch);
                vnd_v2(&mut state, hp, tsn_ref);
                population.push(state.clone_solution());
            }
        }
        population.sort_unstable_by_key(|s| std::cmp::Reverse(s.value));
        population.truncate(8);
        if population[0].value <= best_before {
            xo_stall += 1;
            if xo_stall >= 3 { break; }
        } else {
            xo_stall = 0;
        }
    }

    state.restore_solution(&population[0]);
    let mut best_val = state.total_value;
    let mut best_sel: Vec<usize> = state.selected_items();
    let mut best_bit = state.selected_bit.clone();
    let mut best_snap = state.clone_solution();

    let mut tabu_hashes: Vec<u64> = Vec::with_capacity(128);
    tabu_hashes.push(state.solution_hash);
    let mut tabu_membership = TabuTable::new();
    tabu_membership.insert(state.solution_hash);
    let mut basin_signatures = [0u64; 32];
    basin_signatures[0] = basin_signature(&state);
    let mut basin_len = 1usize;

    let mut stall_count = 0;
    let max_stall = (hp.ils_rounds / 3).min(hp.stall_cap);
    let use_light_vnd = tsn_ref.is_some();
    let mut reactive_perturb = ReactivePerturbSchedule::new();
    let mut cnt = hp.ils_rounds;
    for round in 0..cnt {
        if stall_count >= max_stall { break; }
        let snap = state.clone_solution();

        dp_refinement_hp(&mut state, ch);
        if use_light_vnd {
            local_search_vnd_tsn_light(&mut state, tsn_ref.unwrap());
        } else {
            vnd_v2(&mut state, hp, tsn_ref);
        }

        reactive_perturb.note_polish(snap.value, state.total_value);
        if state.total_value > best_val {
            best_val = state.total_value;
            best_sel = state.selected_items();
            best_bit.clone_from(&state.selected_bit);
            if hp.back_to_best > 0 { best_snap = state.clone_solution(); }
            stall_count = 0;
        }

        if state.total_value <= snap.value {
            state.restore_solution(&snap);
            stall_count += 1;

            if hp.back_to_best > 0 && stall_count % hp.back_to_best == 0 {
                state.restore_solution(&best_snap);
            } else if hp.ils_restart_interval > 0 && stall_count > 0 && stall_count % hp.ils_restart_interval == 0 {
                let pi = (stall_count / hp.ils_restart_interval) % population.len();
                state.restore_solution(&population[pi]);
            }

            let use_bomb = tsn_ref.is_some() && round % 3 == 0;
            if use_bomb {
                let str_v = if stall_count > 15 { 3 } else { 5 };
                cluster_bomb_perturb(&mut state, tsn_ref.unwrap(), &mut rng, str_v);
            } else {
                let strategy = reactive_perturb.choose();
                let strength = 5 + round / 4;
                perturb_by_strategy(&mut state, strength, stall_count, strategy, &mut rng, hp, total_interactions);
            }
            let strategy = reactive_perturb.cursor.wrapping_add(7) % 8;
            greedy_reconstruct(&mut state, strategy % 6, total_interactions);

            let candidate_hash = state.solution_hash;
            let candidate_signature = basin_signature(&state);
            let aspirational = state.total_value > best_val;
            let mut common_with_best = 0usize;
            let mut candidate_count = 0usize;
            unsafe {
                let sp = state.selected_bit.as_ptr();
                let bb = best_bit.as_ptr();
                for i in 0..n {
                    let a = *sp.add(i);
                    let b = *bb.add(i);
                    candidate_count += a as usize;
                    common_with_best += (a & b) as usize;
                }
            }
            let novel_from_best = candidate_count > 0
                && common_with_best.saturating_mul(8) < candidate_count.saturating_mul(7);
            let repeated_basin = basin_signatures[..basin_len].contains(&candidate_signature);
            if !aspirational
                && ((tabu_membership.contains(candidate_hash) && !novel_from_best) || repeated_basin) {
                let alternate_strength = 7 + round / 3;
                perturb_by_strategy(
                    &mut state,
                    alternate_strength,
                    stall_count + 2,
                    (round + 5) % 9,
                    &mut rng,
                    hp,
                    total_interactions,
                );
                greedy_reconstruct(&mut state, (round + 3) % 6, total_interactions);
            }

            if use_light_vnd {
                local_search_vnd_tsn_light(&mut state, tsn_ref.unwrap());
            } else {
                vnd_v2(&mut state, hp, tsn_ref);
            }

            let h = state.solution_hash;
            if tabu_membership.contains(h) {
                if let Some(t) = tsn_ref {
                    cluster_bomb_perturb(&mut state, t, &mut rng, 2);
                } else {
                    let extra_strength = 10 + round / 3;
                    perturb_by_strategy(&mut state, extra_strength, stall_count + 3, 6, &mut rng, hp, total_interactions);
                }
                greedy_reconstruct(&mut state, 0, total_interactions);
                if use_light_vnd {
                    local_search_vnd_tsn_light(&mut state, tsn_ref.unwrap());
                } else {
                    vnd_v2(&mut state, hp, tsn_ref);
                }
            }
            let h2 = state.solution_hash;
            let signature2 = basin_signature(&state);
            reactive_perturb.note_perturb(
                strategy,
                snap.value,
                state.total_value,
                h2 == snap.solution_hash,
            );
            if basin_len < basin_signatures.len() {
                basin_signatures[basin_len] = signature2;
                basin_len += 1;
            } else {
                basin_signatures[round % basin_signatures.len()] = signature2;
            }
            if tabu_hashes.len() < 128 {
                tabu_hashes.push(h2);
                tabu_membership.insert(h2);
            } else {
                let pos = round % 128;
                tabu_membership.remove(tabu_hashes[pos]);
                tabu_hashes[pos] = h2;
                tabu_membership.insert(h2);
            }

            if state.total_value > best_val {
                best_val = state.total_value;
                best_sel = state.selected_items();
                best_bit.clone_from(&state.selected_bit);
                if hp.back_to_best > 0 { best_snap = state.clone_solution(); }
                stall_count = 0;
                cnt += 2;
            }
        } else {
            stall_count = 0;
            let h = state.solution_hash;
            let signature = basin_signature(&state);
            if basin_len < basin_signatures.len() {
                basin_signatures[basin_len] = signature;
                basin_len += 1;
            } else {
                basin_signatures[round % basin_signatures.len()] = signature;
            }
            if tabu_hashes.len() < 128 {
                tabu_hashes.push(h);
                tabu_membership.insert(h);
            }
        }
    }

    let mut final_state = State::new_empty_with(challenge, csr);
    for &i in &best_sel { final_state.add_item(i); }

    if let Some(t) = tsn_ref {
        loop {
            let v_before = final_state.total_value;
            local_search_vnd_tsn(&mut final_state, t);
            dp_refinement_hp(&mut final_state, ch);
            if final_state.total_value <= v_before { break; }
        }
    } else if hp.use_heavy_polish {
        loop {
            let v_before = final_state.total_value;
            local_search_vnd_heavy(&mut final_state);
            dp_refinement_hp(&mut final_state, ch);
            if final_state.total_value <= v_before { break; }
        }
    } else {
        loop {
            let before = final_state.clone_solution();
            local_search_vnd_windowed_deep(&mut final_state, hp.window_k);
            if reactive_perturb.polish_stalls < 3 {
                dp_refinement_hp(&mut final_state, ch);
                local_search_vnd_windowed_deep(&mut final_state, hp.window_k);
            }
            reactive_perturb.note_polish(before.value, final_state.total_value);
            if final_state.total_value <= before.value {
                final_state.restore_solution(&before);
                break;
            }
        }
    }

    if final_state.total_value > best_val {
        let v = final_state.total_value;
        (Solution { items: final_state.selected_items() }, v)
    } else {
        (Solution { items: best_sel }, best_val)
    }
}

fn eval_solution(ch: &Challenge, sol: &Solution) -> i64 {
    let mut val: i64 = 0;
    for &i in &sol.items {
        val += (unsafe { *ch.values.get_unchecked(i) }) as i64;
        for &j in &sol.items {
            if j > i { val += (unsafe { *ch.interaction_values.get_unchecked(i).get_unchecked(j) }) as i64; }
        }
    }
    val
}

fn path_relink(challenge: &Challenge, sol_a: &Solution, sol_b: &Solution, hp: &Hparams, csr: Option<&Csr>) -> Solution {
    let n = challenge.num_items;
    let mut in_a = vec![false; n];
    let mut in_b = vec![false; n];
    for &i in &sol_a.items { in_a[i] = true; }
    for &i in &sol_b.items { in_b[i] = true; }

    let mut state = State::new_empty_with(challenge, csr);
    for &i in &sol_a.items { state.add_item(i); }

    let mut to_add: Vec<usize> = (0..n).filter(|&i| in_b[i] && !in_a[i]).collect();
    let mut to_remove: Vec<usize> = (0..n).filter(|&i| in_a[i] && !in_b[i]).collect();

    let mut best_val = state.total_value;
    let mut best_bits = state.selected_bit.clone();
    let cap = challenge.max_weight;
    let total_moves = to_add.len() + to_remove.len();
    let checkpoint_interval = (total_moves / 4).max(3);
    let mut move_count = 0usize;

    while !to_add.is_empty() || !to_remove.is_empty() {
        let mut best_delta = i64::MIN;
        let mut best_action: Option<(bool, usize)> = None;

        for (idx, &item) in to_add.iter().enumerate() {
            if state.total_weight + challenge.weights[item] <= cap {
                let delta = (unsafe { *state.contrib.get_unchecked(item) }) as i64;
                if delta > best_delta { best_delta = delta; best_action = Some((true, idx)); }
            }
        }
        for (idx, &item) in to_remove.iter().enumerate() {
            let delta = -((unsafe { *state.contrib.get_unchecked(item) }) as i64);
            if delta > best_delta { best_delta = delta; best_action = Some((false, idx)); }
        }

        match best_action {
            Some((true, idx)) => {
                let item = to_add[idx];
                state.add_item(item);
                to_add.swap_remove(idx);
            }
            Some((false, idx)) => {
                let item = to_remove[idx];
                state.remove_item(item);
                to_remove.swap_remove(idx);
            }
            None => break,
        }
        move_count += 1;

        if state.total_weight <= cap && state.total_value > best_val {
            best_val = state.total_value;
            best_bits = state.selected_bit.clone();
        }

        if move_count % checkpoint_interval == 0 && state.total_weight <= cap {
            let mut tmp = State::new_empty_with(challenge, csr);
            for i in 0..n { if (unsafe { *state.selected_bit.get_unchecked(i) }) { tmp.add_item(i); } }
            local_search_vnd_fast(&mut tmp);
            dp_refinement_hp(&mut tmp, hp.core_half_dp);
            if tmp.total_value > best_val {
                best_val = tmp.total_value;
                best_bits = tmp.selected_bit.clone();
            }
        }
    }

    let mut final_state = State::new_empty_with(challenge, csr);
    for i in 0..n { if best_bits[i] { final_state.add_item(i); } }
    loop {
        let v_before = final_state.total_value;
        local_search_vnd_windowed_deep(&mut final_state, hp.window_k);
        dp_refinement_hp(&mut final_state, hp.core_half_dp);
        if final_state.total_value <= v_before { break; }
    }
    Solution { items: final_state.selected_items() }
}

fn build_frequency_biased(state: &mut State, freq: &[f64], rng: &mut Rng) {
    let n = state.ch.num_items;
    loop {
        let slack = state.slack();
        if slack == 0 { break; }
        let mut best_i: Option<usize> = None;
        let mut best_s: f64 = f64::MIN;
        for i in 0..n {
            if (unsafe { *state.selected_bit.get_unchecked(i) }) { continue; }
            if (unsafe { *state.ch.weights.get_unchecked(i) }) > slack { continue; }
            let c = (unsafe { *state.contrib.get_unchecked(i) }) as f64;
            if c <= 0.0 { continue; }
            let w = ((unsafe { *state.ch.weights.get_unchecked(i) }) as f64).max(1.0);
            let s = (c / w) * (1.0 + freq[i] * 2.0)
                + (rng.next_u32() & 0x3F) as f64 * 0.01;
            if s > best_s { best_s = s; best_i = Some(i); }
        }
        if let Some(i) = best_i { state.add_item(i); } else { break; }
    }
}

fn crossover_elite_frequency(elite: &[(Solution, i64)], ch: &Challenge, rng: &mut Rng) -> Vec<bool> {
    let n = ch.num_items;
    let mut freq = vec![0.0f64; n];
    let total = elite.len() as f64;
    for (sol, _) in elite {
        for &i in &sol.items { freq[i] += 1.0; }
    }
    let mut bits = vec![false; n];
    let mut weight: u32 = 0;
    let mut order: Vec<(usize, f64)> = (0..n).map(|i| {
        let p = freq[i] / total;
        let w = (unsafe { *ch.weights.get_unchecked(i) }) as f64;
        (i, p * 1000.0 + ((unsafe { *ch.values.get_unchecked(i) }) as f64) / w.max(1.0))
    }).collect();
    order.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for entry in &order {
        let i = entry.0;
        let p = freq[i] / total;
        let threshold = if p > 0.8 { 0.1 } else if p > 0.5 { 0.4 } else { 0.7 };
        if rng.next_f64() > threshold && weight + (unsafe { *ch.weights.get_unchecked(i) }) <= ch.max_weight {
            bits[i] = true;
            weight += (unsafe { *ch.weights.get_unchecked(i) });
        }
    }
    bits
}

pub struct Solver;

impl Solver {
    pub fn solve(
        challenge: &Challenge,
        _save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
        hyperparameters: &Option<Map<String, Value>>,
    ) -> Result<Option<Solution>> {
        let n = challenge.num_items;
        let hp = Hparams::from_map(hyperparameters);
        let n_restarts = hp.n_full_restarts.max(1);
        FASTM.store(hp.fastmask, std::sync::atomic::Ordering::Relaxed);
        if hp.stage_stop == 1 { return Ok(Some(Solution { items: vec![] })); }
        {
            let mut mw = u32::MAX;
            let mut xw = 0u32;
            for &w in challenge.weights.iter() { mw = mw.min(w); xw = xw.max(w); }
            MINW.store(mw, std::sync::atomic::Ordering::Relaxed);
            MAXW.store(xw, std::sync::atomic::Ordering::Relaxed);
            DP_MEMO.with(|c| c.borrow_mut().clear());
            RAND_MEMBER.with(|c| *c.borrow_mut() = None);
            SUPER_PREFIX.with(|c| *c.borrow_mut() = None);
            VND_PATHS.with(|c| c.borrow_mut().clear());
        }

        let csr_owned: Option<Csr> = if hp.sparse_contrib > 0 { Csr::build(challenge) } else { None };
        let csr: Option<&Csr> = csr_owned.as_ref();
        MAXEDGE.store(csr.map_or(0, |c| c.val.iter().copied().max().unwrap_or(0)) as u32,
                      std::sync::atomic::Ordering::Relaxed);

        let total_interactions: Vec<i64> = {
            let mut sums = vec![0i64; n];
            if let Some(c) = csr {
                for i in 0..n {
                    let s0 = unsafe { *c.starts.get_unchecked(i) } as usize;
                    let e0 = unsafe { *c.starts.get_unchecked(i + 1) } as usize;
                    let mut si: i64 = 0;
                    for k in s0..e0 { si += unsafe { *c.val.get_unchecked(k) } as i64; }
                    sums[i] = si;
                }
            } else {
                for i in 0..n {
                    let row = unsafe { challenge.interaction_values.get_unchecked(i) };
                    let mut si: i64 = 0;
                    for j in 0..n { si += unsafe { *row.get_unchecked(j) } as i64; }
                    sums[i] = si;
                }
            }
            sums
        };

        if n > 1200 {
            let tsn = TopNeighbors::new(challenge, 12, csr);
            let det_seeds = build_det_seeds(challenge, &hp, Some(&tsn), &total_interactions, csr);
            let mut best_sol: Option<Solution> = None;
            let mut best_quality: i64 = i64::MIN;
            let mut restart_stall = 0usize;
            for restart in 0..n_restarts {
                let (sol, val) = if restart >= 2 && restart % hp.seed_every == 0 && best_sol.is_some() {
                    run_one_instance_seeded(challenge, &hp, restart, Some(&tsn), &total_interactions, best_sol.as_ref(), Some(&det_seeds), csr)
                } else {
                    run_one_instance(challenge, &hp, restart, Some(&tsn), &total_interactions, Some(&det_seeds), csr)
                };
                if val > best_quality {
                    best_quality = val;
                    best_sol = Some(sol);
                    restart_stall = 0;
                } else {
                    restart_stall += 1;
                    if restart_stall >= hp.restart_stall_lim { break; }
                }
            }
            return Ok(best_sol);
        }

        let mut rng = Rng::from_seed(&challenge.seed);
        for _ in 0..9999 { rng.next_u32(); }

        let ch_dp = hp.core_half_dp;
        let mut elite: Vec<(Solution, i64)> = Vec::new();

        let n_phase1 = n_restarts.min(6).max(2);
        let mut restart_stall_1k = 0usize;
        for restart in 0..n_phase1 {
            let (sol, val) = run_one_instance(challenge, &hp, restart, None, &total_interactions, None, csr);
            let prev_best = elite.first().map(|e| e.1).unwrap_or(i64::MIN);
            elite.push((sol, val));
            elite.sort_by_key(|&(_, v)| std::cmp::Reverse(v));
            if val <= prev_best {
                restart_stall_1k += 1;
                if restart_stall_1k >= 3 && restart >= 2 { break; }
            } else {
                restart_stall_1k = 0;
            }
        }

        let mut freq = vec![0.0f64; n];
        let top_k = elite.len().min(4);
        for (sol, _) in &elite[..top_k] {
            for &i in &sol.items { freq[i] += 1.0 / top_k as f64; }
        }

        let n_phase3 = 4;
        for restart in 0..n_phase3 {
            let mut state = State::new_empty_with(challenge, csr);
            if restart % 3 == 0 {
                let bits = crossover_elite_frequency(&elite[..top_k.min(elite.len())], challenge, &mut rng);
                set_state_from_bits(&mut state, &bits);
            } else {
                build_frequency_biased(&mut state, &freq, &mut rng);
            }
            dp_refinement_hp(&mut state, ch_dp);
            local_search_vnd_windowed(&mut state, hp.window_k);

            let mut best_snap = state.clone_solution();
            let mut stall = 0usize;
            for round in 0..19 {
                if stall >= 7 { break; }
                let snap = state.clone_solution();
                let strategy = round % 8;
                let strength = 4 + round / 3;
                perturb_by_strategy(&mut state, strength, stall, strategy, &mut rng, &hp, &total_interactions);
                greedy_reconstruct(&mut state, strategy % 10, &total_interactions);
                local_search_vnd_windowed(&mut state, hp.window_k);
                dp_refinement_hp(&mut state, ch_dp);
                if state.total_value > best_snap.value {
                    best_snap = state.clone_solution();
                    stall = 0;
                } else {
                    state.restore_solution(&snap);
                    stall += 1;
                }
            }
            state.restore_solution(&best_snap);
            let val = state.total_value;
            elite.push((Solution { items: state.selected_items() }, val));
            elite.sort_by_key(|&(_, v)| std::cmp::Reverse(v));
            elite.truncate(8);

            freq.fill(0.0);
            let tk = elite.len().min(4);
            for (sol, _) in &elite[..tk] {
                for &i in &sol.items { freq[i] += 1.0 / tk as f64; }
            }
        }

        let n_relink = elite.len().min(3);
        for i in 0..n_relink {
            for j in (i+1)..n_relink {
                let sol_ab = path_relink(challenge, &elite[i].0, &elite[j].0, &hp, csr);
                let val_ab = eval_solution(challenge, &sol_ab);
                elite.push((sol_ab, val_ab));
                let sol_ba = path_relink(challenge, &elite[j].0, &elite[i].0, &hp, csr);
                let val_ba = eval_solution(challenge, &sol_ba);
                elite.push((sol_ba, val_ba));
            }
        }
        elite.sort_by_key(|&(_, v)| std::cmp::Reverse(v));

        {
            let (sol, val) = run_one_instance_seeded(challenge, &hp, 99, None, &total_interactions, Some(&elite[0].0), None, csr);
            elite.push((sol, val));
            elite.sort_by_key(|&(_, v)| std::cmp::Reverse(v));
            elite.truncate(8);
        }

        let deep_dp = 150;
        let mut best_val = i64::MIN;
        let mut best_sel = Vec::new();

        for idx in 0..elite.len().min(2) {
            let mut state = State::new_empty_with(challenge, csr);
            for &i in &elite[idx].0.items { state.add_item(i); }
            loop {
                let v_before = state.total_value;
                local_search_vnd_heavy(&mut state);
                dp_refinement_hp(&mut state, deep_dp);
                if state.total_value <= v_before { break; }
            }
            if state.total_value > best_val {
                best_val = state.total_value;
                best_sel = state.selected_items();
            }
        }

        let mut state = State::new_empty_with(challenge, csr);
        for &i in &best_sel { state.add_item(i); }

        for lns_round in 0..7 {
            let sel = state.selected_items();
            let pct = 20 + (lns_round % 4) * 8;
            let n_remove = sel.len() * pct / 100;
            let mut candidates: Vec<(usize, i64)> = sel.iter().map(|&i| {
                let score = match lns_round % 10 {
                    0 => (unsafe { *state.contrib.get_unchecked(i) }) as i64,
                    1 => -((unsafe { *state.ch.weights.get_unchecked(i) }) as i64),
                    2 => (unsafe { *state.contrib.get_unchecked(i) }) as i64 - (unsafe { *state.ch.values.get_unchecked(i) }) as i64,
                    3 => { let w = ((unsafe { *state.ch.weights.get_unchecked(i) }) as i64).max(1); dw((unsafe { *state.contrib.get_unchecked(i) }) as i64 * 1000, w) },
                    4 => { let w = ((unsafe { *state.ch.weights.get_unchecked(i) }) as i64).max(1); ((unsafe { *state.contrib.get_unchecked(i) }) as i64 * 10000) / (w * w) },
                    5 => -((unsafe { *state.contrib.get_unchecked(i) }) as i64),
                    6 => rng.next_u32() as i64,
                    7 => {
                        let anti = 2 * (unsafe { *state.contrib.get_unchecked(i) }) as i64 - total_interactions[i];
                        let w = ((unsafe { *state.ch.weights.get_unchecked(i) }) as i64).max(1);
                        dw(anti * 1000, w)
                    },
                    8 => {
                        (unsafe { *state.contrib.get_unchecked(i) }) as i64 * 100 - total_interactions[i]
                    },
                    _ => {
                        if rng.next_u32() % 2 == 0 { rng.next_u32() as i64 }
                        else { (unsafe { *state.contrib.get_unchecked(i) }) as i64 }
                    },
                };
                (i, score)
            }).collect();
            candidates.sort_unstable_by_key(|&(_, s)| s);
            for j in 0..n_remove.min(candidates.len()) {
                state.remove_item(candidates[j].0);
            }

            match lns_round % 5 {
                0 => greedy_reconstruct(&mut state, 0, &total_interactions),
                1 => greedy_reconstruct(&mut state, 3, &total_interactions),
                2 => {
                    let mut cands: Vec<usize> = (0..n).filter(|&i| !(unsafe { *state.selected_bit.get_unchecked(i) })).collect();
                    cands.sort_unstable_by(|&a, &b| {
                        let sa = (unsafe { *state.contrib.get_unchecked(a) }) as f64 / ((unsafe { *state.ch.weights.get_unchecked(a) }) as f64).max(1.0)
                            + freq[a] * 50.0;
                        let sb = (unsafe { *state.contrib.get_unchecked(b) }) as f64 / ((unsafe { *state.ch.weights.get_unchecked(b) }) as f64).max(1.0)
                            + freq[b] * 50.0;
                        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    for &i in &cands {
                        if state.total_weight + (unsafe { *state.ch.weights.get_unchecked(i) }) <= challenge.max_weight {
                            state.add_item(i);
                        }
                    }
                },
                _ => greedy_reconstruct(&mut state, 2, &total_interactions),
            }

            loop {
                let v_before = state.total_value;
                local_search_vnd_windowed_deep(&mut state, hp.window_k);
                dp_refinement_hp(&mut state, deep_dp);
                if state.total_value <= v_before { break; }
            }

            if state.total_value > best_val {
                best_val = state.total_value;
                best_sel = state.selected_items();
            }
            let mut rst = State::new_empty_with(challenge, csr);
            for &i in &best_sel { rst.add_item(i); }
            state.restore_solution(&rst.clone_solution());
        }

        elite.sort_by_key(|&(_, v)| std::cmp::Reverse(v));
        let tk = elite.len().min(6);
        let mut item_freq = vec![0usize; n];
        for (sol, _) in &elite[..tk] {
            for &i in &sol.items { item_freq[i] += 1; }
        }
        let always_in: Vec<usize> = (0..n).filter(|&i| item_freq[i] == tk).collect();
        let disputed: Vec<usize> = (0..n).filter(|&i| item_freq[i] > 0 && item_freq[i] < tk).collect();

        if disputed.len() > 0 && disputed.len() <= 199 {
            let mut state = State::new_empty_with(challenge, csr);
            for &i in &always_in {
                if state.total_weight + challenge.weights[i] <= challenge.max_weight {
                    state.add_item(i);
                }
            }
            let fixed_weight = state.total_weight;
            let rem_cap = (challenge.max_weight - fixed_weight) as usize;

            if rem_cap > 0 {
                let dk = disputed.len();
                let mut total_disp_weight: usize = 0;
                for &it in &disputed { total_disp_weight += challenge.weights[it] as usize; }
                let myw = rem_cap.min(total_disp_weight);
                let dp_size = myw + 1;

                if dp_size <= 1_999_999 {
                    let mut dp = vec![i64::MIN / 4; dp_size];
                    let mut choose = vec![0u8; dk * dp_size];
                    dp[0] = 0;
                    let mut w_hi: usize = 0;

                    for (t, &it) in disputed.iter().enumerate() {
                        let wt = challenge.weights[it] as usize;
                        if wt > myw { continue; }
                        let val = (unsafe { *state.contrib.get_unchecked(it) }) as i64;
                        let new_hi = (w_hi + wt).min(myw);
                        for w in (wt..=new_hi).rev() {
                            let cand = dp[w - wt] + val;
                            if cand > dp[w] {
                                dp[w] = cand;
                                choose[t * dp_size + w] = 1;
                            }
                        }
                        w_hi = new_hi;
                    }

                    let mut w_star = (0..=myw).max_by_key(|&w| dp[w]).unwrap_or(0);
                    let mut dp_selected = Vec::new();
                    for t in (0..dk).rev() {
                        let it = disputed[t];
                        let wt = challenge.weights[it] as usize;
                        if wt <= w_star && choose[t * dp_size + w_star] == 1 {
                            dp_selected.push(it);
                            w_star -= wt;
                        }
                    }

                    for &i in &dp_selected { state.add_item(i); }

                    loop {
                        let v_before = state.total_value;
                        local_search_vnd_heavy(&mut state);
                        dp_refinement_hp(&mut state, deep_dp);
                        if state.total_value <= v_before { break; }
                    }

                    if state.total_value > best_val {
                        best_sel = state.selected_items();
                    }
                }
            }
        }

        Ok(Some(Solution { items: best_sel }))
    }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    if let Some(solution) = Solver::solve(challenge, Some(save_solution), hyperparameters)? {
        let _ = save_solution(&solution);
    }
    Ok(())
}

pub fn help() {
    println!("satchel");
}

#[inline(always)]
pub fn solve(
    challenge: &Challenge,
    save: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hp: &Option<serde_json::Map<String, serde_json::Value>>,
) -> anyhow::Result<()> {
    solve_challenge(challenge, save, hp)
}

#[allow(dead_code)]
mod exact_sort {
#![allow(unused_imports, unused_parens, clippy::all)]
use tig_challenges::knapsack::*;

#[inline(always)]
fn compare_swap<F: Fn(usize, usize) -> bool>(v: &mut [usize], a: usize, b: usize, less: &F) {
    unsafe {
        let x = *v.get_unchecked(a);
        let y = *v.get_unchecked(b);
        let swap = less(y, x);
        *v.get_unchecked_mut(a) = if swap { y } else { x };
        *v.get_unchecked_mut(b) = if swap { x } else { y };
    }
}

pub(super) fn pack_density(out: &mut [usize], contrib: &[i32], weights: &[u32]) {
    const FACTOR: [i64; 11] = [2520, 2520, 1260, 840, 630, 504, 420, 360, 315, 280, 252];
    assert_eq!(out.len(), contrib.len());
    assert_eq!(out.len(), weights.len());
    let n = out.len();
    let mut i = 0;
    unsafe {
        let p = out.as_mut_ptr();
        let c = contrib.as_ptr();
        let w = weights.as_ptr();
        macro_rules! emit {
            ($i:expr) => {{
                let i = $i;
                let factor = FACTOR[(*w.add(i) as usize).clamp(1, 10)];
                *p.add(i) = (((*c.add(i) as i64 * factor) << 16) as usize) | i;
            }};
        }
        let full = n / 16 * 16;
        while i < full {
            emit!(i);
            emit!(i + 1);
            emit!(i + 2);
            emit!(i + 3);
            emit!(i + 4);
            emit!(i + 5);
            emit!(i + 6);
            emit!(i + 7);
            emit!(i + 8);
            emit!(i + 9);
            emit!(i + 10);
            emit!(i + 11);
            emit!(i + 12);
            emit!(i + 13);
            emit!(i + 14);
            emit!(i + 15);
            i += 16;
        }
        while i < n {
            emit!(i);
            i += 1;
        }
    }
}

pub(super) fn density_order(
    ids: &mut [usize],
    contrib: &[i32],
    weights: &[u32],
    small_weights: bool,
) {
    if usize::BITS == 64 && weights.len() <= u16::MAX as usize && small_weights {
        const F: [i64; 11] = [0, 2520, 1260, 840, 630, 504, 420, 360, 315, 280, 252];
        for item in ids.iter_mut() {
            let i = *item;
            *item = (((contrib[i] as i64 * F[weights[i] as usize]) << 16) as usize) | i;
        }
        if compatible() {
            sort(ids, |a, b| ((a as i64) >> 16) > ((b as i64) >> 16));
        } else {
            ids.sort_unstable_by(|&a, &b| ((b as i64) >> 16).cmp(&((a as i64) >> 16)));
        }
        for item in ids {
            *item &= 0xffff;
        }
    } else {
        ids.sort_unstable_by(|&a, &b| {
            ((contrib[b] as i64) * (weights[a] as i64).max(1))
                .cmp(&((contrib[a] as i64) * (weights[b] as i64).max(1)))
        });
    }
}

pub(super) fn keyed_prefix(
    ids: &mut [usize],
    keys: &[i64],
    weights: &[u32],
    capacity: u32,
    weight_first: bool,
) -> usize {
    if usize::BITS == 64 && weights.len() <= u16::MAX as usize {
        let mut fits = true;
        for item in ids.iter_mut() {
            let i = *item;
            let mut key = keys[i];
            if weight_first {
                fits &= weights[i] <= 1024 && (-(1i64 << 31)..=(1i64 << 31)).contains(&key);
                key = (weights[i] as i64)
                    .wrapping_mul(1i64 << 33)
                    .wrapping_add(key);
            }
            fits &= (-(1i64 << 47)..(1i64 << 47)).contains(&key);
            *item = ((key << 16) as usize) | i;
        }
        if fits {
            let end = capacity_prefix(
                ids,
                |a, b| ((a as i64) >> 16) < ((b as i64) >> 16),
                |key| weights[key & 0xffff],
                capacity,
                0,
            );
            for item in ids {
                *item &= 0xffff;
            }
            return end;
        }
        for item in ids.iter_mut() {
            *item &= 0xffff;
        }
    }
    capacity_prefix(
        ids,
        |a, b| {
            if weight_first {
                weights[a]
                    .cmp(&weights[b])
                    .then(keys[a].cmp(&keys[b]))
                    .is_lt()
            } else {
                keys[a] < keys[b]
            }
        },
        |i| weights[i],
        capacity,
        0,
    )
}

fn insertion<F: Fn(usize, usize) -> bool>(v: &mut [usize], start: usize, less: &F) {
    for i in start..v.len() {
        let x = v[i];
        let mut j = i;
        while j > 0 && less(x, v[j - 1]) {
            v[j] = v[j - 1];
            j -= 1;
        }
        v[j] = x;
    }
}

fn small<F: Fn(usize, usize) -> bool>(v: &mut [usize], less: &F) {
    let n = v.len();
    if n < 2 {
        return;
    }
    let middle = if n < 18 { n } else { n / 2 };
    for range in [0..middle, middle..n] {
        let part = &mut v[range];
        let sorted = if part.len() >= 13 {
            sort13_optimal(part, less);
            13
        } else if part.len() >= 9 {
            sort9_optimal(part, less);
            9
        } else {
            1
        };
        insertion(part, sorted, less);
    }
    if middle == n {
        return;
    }
    let mut scratch = [0usize; 32];
    let (mut a, mut b) = (0, middle);
    for dest in &mut scratch[..n] {
        if b == n || (a < middle && !less(v[b], v[a])) {
            *dest = v[a];
            a += 1;
        } else {
            *dest = v[b];
            b += 1;
        }
    }
    v.copy_from_slice(&scratch[..n]);
}

#[inline(always)]
fn median<F: Fn(usize, usize) -> bool>(
    v: &[usize],
    a: usize,
    b: usize,
    c: usize,
    less: &F,
) -> usize {
    let x = less(v[a], v[b]);
    let y = less(v[a], v[c]);
    if x == y {
        if less(v[b], v[c]) ^ x {
            c
        } else {
            b
        }
    } else {
        a
    }
}
fn median_rec<F: Fn(usize, usize) -> bool>(
    v: &[usize],
    mut a: usize,
    mut b: usize,
    mut c: usize,
    n: usize,
    less: &F,
) -> usize {
    if n * 8 >= 64 {
        let k = n / 8;
        a = median_rec(v, a, a + 4 * k, a + 7 * k, k, less);
        b = median_rec(v, b, b + 4 * k, b + 7 * k, k, less);
        c = median_rec(v, c, c + 4 * k, c + 7 * k, k, less);
    }
    median(v, a, b, c, less)
}
fn pivot<F: Fn(usize, usize) -> bool>(v: &[usize], less: &F) -> usize {
    let k = v.len() / 8;
    if v.len() < 64 {
        median(v, 0, k * 4, k * 7, less)
    } else {
        median_rec(v, 0, k * 4, k * 7, k, less)
    }
}

fn partition<F: Fn(usize, usize) -> bool, const EQUAL: bool>(
    v: &mut [usize],
    pivot: usize,
    less: &F,
) -> usize {
    v.swap(0, pivot);
    let value = v[0];
    let n = v.len() - 1;
    if n == 0 {
        return 0;
    }
    unsafe {
        let p = v.as_mut_ptr().add(1);
        let saved = *p;
        let mut count = 0usize;
        let mut gap = 0usize;
        let mut right = 1usize;
        macro_rules! step {
            ($x:expr,$to:expr) => {{
                let x = $x;
                let yes = if EQUAL {
                    !less(value, x)
                } else {
                    less(x, value)
                };
                *p.add(gap) = *p.add(count);
                *p.add(count) = x;
                gap = $to;
                count += yes as usize;
            }};
        }
        macro_rules! batch4 {
            () => {{
                let x0 = *p.add(right);
                let x1 = *p.add(right + 1);
                let x2 = *p.add(right + 2);
                let x3 = *p.add(right + 3);
                let y0 = if EQUAL {
                    !less(value, x0)
                } else {
                    less(x0, value)
                };
                let y1 = if EQUAL {
                    !less(value, x1)
                } else {
                    less(x1, value)
                };
                let y2 = if EQUAL {
                    !less(value, x2)
                } else {
                    less(x2, value)
                };
                let y3 = if EQUAL {
                    !less(value, x3)
                } else {
                    less(x3, value)
                };
                macro_rules! commit {
                    ($x:expr,$yes:expr) => {{
                        *p.add(gap) = *p.add(count);
                        *p.add(count) = $x;
                        gap = right;
                        count += $yes as usize;
                        right += 1;
                    }};
                }
                commit!(x0, y0);
                commit!(x1, y1);
                commit!(x2, y2);
                commit!(x3, y3);
            }};
        }
        while right + 8 <= n {
            batch4!();
            batch4!();
        }
        while right < n {
            step!(*p.add(right), right);
            right += 1;
        }
        step!(saved, 0);
        let _ = gap;
        v.swap(0, count);
        count
    }
}

fn heap<F: Fn(usize, usize) -> bool>(v: &mut [usize], less: &F) {
    let n = v.len();
    for i in (0..n + n / 2).rev() {
        let mut node = if i >= n {
            i - n
        } else {
            v.swap(0, i);
            0
        };
        let len = i.min(n);
        loop {
            let mut child = node * 2 + 1;
            if child >= len {
                break;
            }
            if child + 1 < len {
                child += less(v[child], v[child + 1]) as usize;
            }
            if !less(v[node], v[child]) {
                break;
            }
            v.swap(node, child);
            node = child;
        }
    }
}

fn quick<F, S>(
    mut v: &mut [usize],
    mut ancestor: Option<usize>,
    mut limit: u32,
    mut offset: usize,
    less: &F,
    sink: &mut S,
) -> bool
where
    F: Fn(usize, usize) -> bool,
    S: FnMut(&[usize], usize) -> bool,
{
    loop {
        if v.len() <= 32 {
            small(v, less);
            return sink(v, offset);
        }
        if limit == 0 {
            heap(v, less);
            return sink(v, offset);
        }
        limit -= 1;
        let pos = pivot(v, less);
        if ancestor.map_or(false, |a| !less(a, v[pos])) {
            let count = partition::<F, true>(v, pos, less) + 1;
            if !sink(&v[..count], offset) {
                return false;
            }
            offset += count;
            v = &mut v[count..];
            ancestor = None;
            continue;
        }
        let count = partition::<F, false>(v, pos, less);
        let (left, right) = v.split_at_mut(count);
        let (p, right) = right.split_at_mut(1);
        if !quick(left, ancestor, limit, offset, less, sink) {
            return false;
        }
        if !sink(p, offset + count) {
            return false;
        }
        ancestor = Some(p[0]);
        offset += count + 1;
        v = right;
    }
}

pub(super) fn sort<F: Fn(usize, usize) -> bool>(v: &mut [usize], less: F) {
    visit(v, &less, &mut |_, _| true);
}

fn visit<F, S>(v: &mut [usize], less: &F, sink: &mut S)
where
    F: Fn(usize, usize) -> bool,
    S: FnMut(&[usize], usize) -> bool,
{
    let n = v.len();
    if n <= 20 {
        insertion(v, 1, less);
        sink(v, 0);
        return;
    }
    let descending = less(v[1], v[0]);
    let mut run = 2;
    while run < n
        && if descending {
            less(v[run], v[run - 1])
        } else {
            !less(v[run], v[run - 1])
        }
    {
        run += 1;
    }
    if run == n {
        if descending {
            v.reverse();
        }
        sink(v, 0);
        return;
    }
    quick(v, None, 2 * (n | 1).ilog2(), 0, less, sink);
}

pub(super) fn capacity_prefix<F, W>(
    v: &mut [usize],
    less: F,
    weight: W,
    capacity: u32,
    pad: usize,
) -> usize
where
    F: Fn(usize, usize) -> bool,
    W: Fn(usize) -> u32,
{
    let n = v.len();
    let mut rem = capacity;
    let mut goal = n;
    if rem == 0 {
        goal = pad.min(n);
    }
    visit(v, &less, &mut |part, offset| {
        for (i, &item) in part.iter().enumerate() {
            let index = offset + i;
            if index >= goal {
                return false;
            }
            if rem > 0 {
                let w = weight(item);
                if w <= rem {
                    rem -= w;
                    if rem == 0 {
                        goal = (index + pad + 1).min(n);
                    }
                }
            }
        }
        offset + part.len() < goal
    });
    goal
}

pub(super) fn capacity_core<F, W>(
    v: &mut [usize],
    less: F,
    weight: W,
    capacity: u32,
    half: usize,
    max_weight: u32,
) where
    F: Fn(usize, usize) -> bool,
    W: Fn(usize) -> u32,
{
    let n = v.len();
    let mut scan = CoreScan {
        rem: capacity as u64,
        goal: if capacity == 0 {
            half.saturating_add(1).min(n)
        } else {
            n
        },
        half,
        reserve: (half as u64 + 1).saturating_mul(max_weight as u64),
        n,
    };
    if n <= 20 {
        insertion(v, 1, &less);
        return;
    }
    let descending = less(v[1], v[0]);
    let mut run = 2;
    while run < n
        && if descending {
            less(v[run], v[run - 1])
        } else {
            !less(v[run], v[run - 1])
        }
    {
        run += 1;
    }
    if run == n {
        if descending {
            v.reverse();
        }
        return;
    }
    core_quick(v, None, 2 * (n | 1).ilog2(), 0, &less, &weight, &mut scan);
}
struct CoreScan {
    rem: u64,
    goal: usize,
    half: usize,
    reserve: u64,
    n: usize,
}
impl CoreScan {
    fn consume<W: Fn(usize) -> u32>(&mut self, part: &[usize], offset: usize, weight: &W) -> bool {
        for (i, &item) in part.iter().enumerate() {
            let index = offset + i;
            if index >= self.goal {
                return false;
            }
            if self.rem > 0 {
                let w = weight(item) as u64;
                if w <= self.rem {
                    self.rem -= w;
                    if self.rem == 0 {
                        self.goal = (index + self.half + 1).min(self.n);
                    }
                }
            }
        }
        offset + part.len() < self.goal
    }
}
fn core_quick<F, W>(
    mut v: &mut [usize],
    mut ancestor: Option<usize>,
    mut limit: u32,
    mut offset: usize,
    less: &F,
    weight: &W,
    scan: &mut CoreScan,
) -> bool
where
    F: Fn(usize, usize) -> bool,
    W: Fn(usize) -> u32,
{
    loop {
        if offset >= scan.goal {
            return false;
        }
        if scan.rem >= scan.reserve
            && !v.is_empty()
            && offset + v.len() <= scan.n.saturating_sub(scan.half.saturating_add(1))
        {
            let total: u64 = v.iter().map(|&i| weight(i) as u64).sum();
            if total <= scan.rem - scan.reserve {
                scan.rem -= total;
                return true;
            }
        }
        if v.len() <= 32 {
            small(v, less);
            return scan.consume(v, offset, weight);
        }
        if limit == 0 {
            heap(v, less);
            return scan.consume(v, offset, weight);
        }
        limit -= 1;
        let pos = pivot(v, less);
        if ancestor.map_or(false, |a| !less(a, v[pos])) {
            let count = partition::<F, true>(v, pos, less) + 1;
            if !scan.consume(&v[..count], offset, weight) {
                return false;
            }
            offset += count;
            v = &mut v[count..];
            ancestor = None;
            continue;
        }
        let count = partition::<F, false>(v, pos, less);
        let (left, right) = v.split_at_mut(count);
        let (p, right) = right.split_at_mut(1);
        if !core_quick(left, ancestor, limit, offset, less, weight, scan) {
            return false;
        }
        if !scan.consume(p, offset + count, weight) {
            return false;
        }
        ancestor = Some(p[0]);
        offset += count + 1;
        v = right;
    }
}

fn sort9_optimal<F: Fn(usize, usize) -> bool>(v: &mut [usize], less: &F) {
    compare_swap(v, 0, 3, less);
    compare_swap(v, 1, 7, less);
    compare_swap(v, 2, 5, less);
    compare_swap(v, 4, 8, less);
    compare_swap(v, 0, 7, less);
    compare_swap(v, 2, 4, less);
    compare_swap(v, 3, 8, less);
    compare_swap(v, 5, 6, less);
    compare_swap(v, 0, 2, less);
    compare_swap(v, 1, 3, less);
    compare_swap(v, 4, 5, less);
    compare_swap(v, 7, 8, less);
    compare_swap(v, 1, 4, less);
    compare_swap(v, 3, 6, less);
    compare_swap(v, 5, 7, less);
    compare_swap(v, 0, 1, less);
    compare_swap(v, 2, 4, less);
    compare_swap(v, 3, 5, less);
    compare_swap(v, 6, 8, less);
    compare_swap(v, 2, 3, less);
    compare_swap(v, 4, 5, less);
    compare_swap(v, 6, 7, less);
    compare_swap(v, 1, 2, less);
    compare_swap(v, 3, 4, less);
    compare_swap(v, 5, 6, less);
}

fn sort13_optimal<F: Fn(usize, usize) -> bool>(v: &mut [usize], less: &F) {
    compare_swap(v, 0, 12, less);
    compare_swap(v, 1, 10, less);
    compare_swap(v, 2, 9, less);
    compare_swap(v, 3, 7, less);
    compare_swap(v, 5, 11, less);
    compare_swap(v, 6, 8, less);
    compare_swap(v, 1, 6, less);
    compare_swap(v, 2, 3, less);
    compare_swap(v, 4, 11, less);
    compare_swap(v, 7, 9, less);
    compare_swap(v, 8, 10, less);
    compare_swap(v, 0, 4, less);
    compare_swap(v, 1, 2, less);
    compare_swap(v, 3, 6, less);
    compare_swap(v, 7, 8, less);
    compare_swap(v, 9, 10, less);
    compare_swap(v, 11, 12, less);
    compare_swap(v, 4, 6, less);
    compare_swap(v, 5, 9, less);
    compare_swap(v, 8, 11, less);
    compare_swap(v, 10, 12, less);
    compare_swap(v, 0, 5, less);
    compare_swap(v, 3, 8, less);
    compare_swap(v, 4, 7, less);
    compare_swap(v, 6, 11, less);
    compare_swap(v, 9, 10, less);
    compare_swap(v, 0, 1, less);
    compare_swap(v, 2, 5, less);
    compare_swap(v, 6, 9, less);
    compare_swap(v, 7, 8, less);
    compare_swap(v, 10, 11, less);
    compare_swap(v, 1, 3, less);
    compare_swap(v, 2, 4, less);
    compare_swap(v, 5, 6, less);
    compare_swap(v, 9, 10, less);
    compare_swap(v, 1, 2, less);
    compare_swap(v, 3, 4, less);
    compare_swap(v, 5, 7, less);
    compare_swap(v, 6, 8, less);
    compare_swap(v, 2, 3, less);
    compare_swap(v, 4, 5, less);
    compare_swap(v, 6, 7, less);
    compare_swap(v, 8, 9, less);
    compare_swap(v, 3, 4, less);
    compare_swap(v, 5, 6, less);
}

fn compatible_uncached() -> bool {
        for n in [17usize, 31, 89, 257, 1001] {
            for distinct in [3usize, 19, 511] {
                let mut a: Vec<usize> = (0..n).collect();
                let key = |i: usize| i.wrapping_mul(7919).wrapping_add(i / 7) % distinct;
                let mut b = a.clone();
                a.sort_unstable_by(|&x, &y| key(y).cmp(&key(x)));
                sort(&mut b, |x, y| key(x) > key(y));
                if a != b {
                    return false;
                }
                for k in [0, n - 1, n / 3] {
                    let mut a: Vec<usize> = (0..n).collect();
                    let mut b = a.clone();
                    a.select_nth_unstable_by(k, |&x, &y| key(y).cmp(&key(x)));
                    select(&mut b, k, |x, y| key(x) > key(y));
                    if a != b {
                        return false;
                    }
                }
            }
        }
        true
    }

pub(super) fn compatible() -> bool {
    thread_local! { static CHECK: bool = compatible_uncached(); }
    CHECK.with(|c| *c)
}

pub(super) fn select<F: Fn(usize, usize) -> bool>(mut v: &mut [usize], mut index: usize, less: F) {
    assert!(index < v.len());
    if index == 0 || index == v.len() - 1 {
        let p = extreme(v, index != 0, &less);
        v.swap(p, index);
        return;
    }
    let mut limit = 16;
    let mut ancestor = None;
    loop {
        if v.len() <= 16 {
            insertion(v, 1, &less);
            return;
        }
        if limit == 0 {
            median_select(v, index, &less);
            return;
        }
        limit -= 1;
        let pos = pivot(v, &less);
        if ancestor.map_or(false, |a| !less(a, v[pos])) {
            let count = partition::<F, true>(v, pos, &less) + 1;
            if count > index {
                return;
            }
            v = &mut v[count..];
            index -= count;
            ancestor = None;
            continue;
        }
        let count = partition::<F, false>(v, pos, &less);
        if count == index {
            return;
        }
        if count < index {
            ancestor = Some(v[count]);
            v = &mut v[count + 1..];
            index -= count + 1;
        } else {
            v = &mut v[..count];
        }
    }
}
fn extreme<F: Fn(usize, usize) -> bool>(v: &[usize], max: bool, less: &F) -> usize {
    let mut chosen = 0;
    for i in 1..v.len() {
        if if max {
            less(v[chosen], v[i])
        } else {
            less(v[i], v[chosen])
        } {
            chosen = i;
        }
    }
    chosen
}
fn median_select<F: Fn(usize, usize) -> bool>(mut v: &mut [usize], mut index: usize, less: &F) {
    loop {
        if v.len() <= 16 {
            insertion(v, 1, less);
            return;
        }
        if index == 0 || index == v.len() - 1 {
            let p = extreme(v, index != 0, less);
            v.swap(p, index);
            return;
        }
        let frac = if v.len() <= 1024 {
            v.len() / 12
        } else if v.len() <= 128 * 1024 {
            v.len() / 64
        } else {
            v.len() / 1024
        };
        let middle = frac / 2;
        let lo = v.len() / 2 - middle;
        let hi = frac + lo;
        let gap = (v.len() - 9 * frac) / 4;
        let mut a = lo - 4 * frac - gap;
        let mut b = hi + gap;
        for i in lo..hi {
            ninther(
                v,
                [a, i - frac, b, a + 1, i, b + 1, a + 2, i + frac, b + 2],
                less,
            );
            a += 3;
            b += 3;
        }
        median_select(&mut v[lo..lo + frac], middle, less);
        let p = partition::<F, false>(v, lo + middle, less);
        if p == index {
            return;
        }
        if p > index {
            v = &mut v[..p];
        } else {
            v = &mut v[p + 1..];
            index -= p + 1;
        }
    }
}
fn median_index<F: Fn(usize, usize) -> bool>(
    v: &[usize],
    mut a: usize,
    b: usize,
    mut c: usize,
    less: &F,
) -> usize {
    if less(v[c], v[a]) {
        let __t = a; a = c; c = __t;
    }
    if less(v[c], v[b]) {
        return c;
    }
    if less(v[b], v[a]) {
        return a;
    }
    b
}
fn ninther<F: Fn(usize, usize) -> bool>(v: &mut [usize], positions: [usize; 9], less: &F) {
    let [a, mut b, c, mut d, e, mut f, g, mut h, i] = positions;
    b = median_index(v, a, b, c, less);
    h = median_index(v, g, h, i, less);
    if less(v[h], v[b]) {
        let __t = b; b = h; h = __t;
    }
    if less(v[f], v[d]) {
        let __t = d; d = f; f = __t;
    }
    if less(v[e], v[d]) {
    } else if less(v[f], v[e]) {
        d = f;
    } else {
        if less(v[e], v[b]) {
            v.swap(e, b);
        } else if less(v[h], v[e]) {
            v.swap(e, h);
        }
        return;
    }
    if less(v[d], v[b]) {
        d = b;
    } else if less(v[h], v[d]) {
        d = h;
    }
    v.swap(d, e);
}

}

#[allow(dead_code)]
mod exact_construct {
#![allow(unused_imports, unused_parens, clippy::all)]
use tig_challenges::knapsack::*;

#[inline(always)]
pub(super) fn positive_density(c: i32, w: u32) -> i64 {
    const M: [u64; 11] = [
        0,
        1u64 << 48,
        ((1u64 << 48) + 1) / 2,
        ((1u64 << 48) + 2) / 3,
        ((1u64 << 48) + 3) / 4,
        ((1u64 << 48) + 4) / 5,
        ((1u64 << 48) + 5) / 6,
        ((1u64 << 48) + 6) / 7,
        ((1u64 << 48) + 7) / 8,
        ((1u64 << 48) + 8) / 9,
        ((1u64 << 48) + 9) / 10,
    ];
    debug_assert!(c > 0 && (1..=10).contains(&w));
    ((((c as u32 as u64) * 1000) as u128 * M[w as usize] as u128) >> 48) as i64
}

#[inline(always)]
pub(super) fn positive_density_narrow(c: i32, w: u32) -> i64 {
    const N: u64 = 1000u64 << 30;
    const M: [u64; 11] = [
        0,
        N,
        (N + 1) / 2,
        (N + 2) / 3,
        (N + 3) / 4,
        (N + 4) / 5,
        (N + 5) / 6,
        (N + 6) / 7,
        (N + 7) / 8,
        (N + 8) / 9,
        (N + 9) / 10,
    ];
    debug_assert!(c > 0 && c < (1 << 24) && (1..=10).contains(&w));
    ((c as u32 as u64 * M[w as usize]) >> 30) as i64
}

pub(super) struct Queue {
    small: Vec<u32>,
    wide: Option<Vec<u64>>,
    maxima: Vec<u64>,
    dirty: Vec<bool>,
    bits: Vec<u64>,
    counts: Vec<i32>,
    n: usize,
    len: usize,
    qual: Vec<u32>,
}
impl Queue {
    pub(super) fn new(n: usize) -> Self {
        assert!(n <= u16::MAX as usize);
        let blocks = (n + 31) / 32;
        let words = (n + 63) / 64;
        Self {
            small: vec![0; blocks * 32],
            wide: None,
            maxima: vec![0; blocks],
            dirty: vec![false; blocks],
            bits: vec![0; words],
            counts: vec![0; words + 1],
            n,
            len: 0,
            qual: Vec::new(),
        }
    }
    #[inline]
    fn active(&mut self, id: usize, yes: bool) {
        let word = id / 64;
        self.bits[word] ^= 1u64 << (id & 63);
        let mut p = word + 1;
        let delta = if yes { 1 } else { -1 };
        while p < self.counts.len() {
            self.counts[p] += delta;
            p += p & p.wrapping_neg();
        }
    }
    #[inline]
    pub(super) fn rank(&self, id: usize) -> usize {
        let word = id / 64;
        let mask = (1u64 << (id & 63)) - 1;
        let mut rank = (self.bits[word] & mask).count_ones() as usize;
        let mut p = word;
        while p != 0 {
            rank += self.counts[p] as usize;
            p &= p - 1;
        }
        rank
    }
    #[inline]
    pub(super) fn set(&mut self, id: usize, score: i64) {
        debug_assert!(id < self.n && score >= 0 && score < (1i64 << 48));
        let code = score as u64 + 1;
        if code > u32::MAX as u64 && self.wide.is_none() {
            self.wide = Some(self.small.iter().map(|&v| v as u64).collect());
        }
        let fresh = unsafe {
            if let Some(wide) = self.wide.as_mut() {
                let p = wide.get_unchecked_mut(id);
                let fresh = *p == 0;
                *p = code;
                fresh
            } else {
                let p = self.small.get_unchecked_mut(id);
                let fresh = *p == 0;
                *p = code as u32;
                fresh
            }
        };
        if fresh {
            self.active(id, true);
            self.len += 1;
        }
        unsafe {
            *self.dirty.get_unchecked_mut(id / 32) = true;
        }
    }
    #[inline]
    pub(super) fn remove(&mut self, id: usize) {
        debug_assert!(id < self.n);
        let existed = unsafe {
            if let Some(wide) = self.wide.as_mut() {
                let p = wide.get_unchecked_mut(id);
                let existed = *p != 0;
                *p = 0;
                existed
            } else {
                let p = self.small.get_unchecked_mut(id);
                let existed = *p != 0;
                *p = 0;
                existed
            }
        };
        if existed {
            self.active(id, false);
            self.len -= 1;
            unsafe {
                *self.dirty.get_unchecked_mut(id / 32) = true;
            }
        }
    }
    #[inline]
    pub(super) fn assign(&mut self, id: usize, score: i64, active: bool) {
        let code = (score as u64 + 1) & 0u64.wrapping_sub(active as u64);
        if code > u32::MAX as u64 && self.wide.is_none() {
            self.wide = Some(self.small.iter().map(|&v| v as u64).collect());
        }
        let existed = unsafe {
            if let Some(wide) = self.wide.as_mut() {
                let p = wide.get_unchecked_mut(id);
                let old = *p != 0;
                *p = code;
                old
            } else {
                let p = self.small.get_unchecked_mut(id);
                let old = *p != 0;
                *p = code as u32;
                old
            }
        };
        if existed != active {
            self.active(id, active);
            if active {
                self.len += 1;
            } else {
                self.len -= 1;
            }
        }
        unsafe {
            *self.dirty.get_unchecked_mut(id / 32) = true;
        }
    }
    #[inline]
    pub(super) unsafe fn assign_small(&mut self, id: usize, score: u32, active: bool) {
        debug_assert!(id < self.n && self.wide.is_none() && score < u32::MAX);
        let code = (score + 1) & 0u32.wrapping_sub(active as u32);
        let p = self.small.get_unchecked_mut(id);
        let existed = *p != 0;
        *p = code;
        if existed != active {
            self.active(id, active);
            if active {
                self.len += 1;
            } else {
                self.len -= 1;
            }
        }
        *self.dirty.get_unchecked_mut(id / 32) = true;
    }
    #[inline(always)]
    pub(super) unsafe fn assign_small_defer(&mut self, id: usize, score: u32, active: bool, flips: *mut u32, nf: &mut usize) {
        let code = (score + 1) & 0u32.wrapping_sub(active as u32);
        let p = self.small.get_unchecked_mut(id);
        let existed = *p != 0;
        *p = code;
        *flips.add(*nf) = id as u32;
        *nf += (existed != active) as usize;
        *self.dirty.get_unchecked_mut(id / 32) = true;
    }
    pub(super) fn apply_flips(&mut self, flips: &[u32]) {
        for &id in flips {
            let id = id as usize;
            let yes = unsafe { *self.small.get_unchecked(id) } != 0;
            self.active(id, yes);
            if yes { self.len += 1; } else { self.len -= 1; }
        }
    }
    pub(super) fn finalists_fast(&mut self, bonus: u64, out: &mut Vec<(usize, i64)>) {
        if self.wide.is_some() { return self.finalists(bonus, out); }
        out.clear();
        if self.len == 0 {
            return;
        }
        let nb = self.maxima.len();
        if self.qual.len() < nb { self.qual.resize(nb, 0); }
        let mut first = 0u64;
        let mut second = 0u64;
        for block in 0..nb {
            let m = unsafe {
                self.small.get_unchecked(block * 32..block * 32 + 32).iter().copied().max().unwrap_or(0)
            } as u64;
            unsafe { *self.maxima.get_unchecked_mut(block) = m; }
            let gt = m > first;
            let sm = if m > second { m } else { second };
            second = if gt { first } else { sm };
            first = if gt { m } else { first };
        }
        let lower = second.saturating_sub(1).saturating_sub(bonus);
        let mut nq = 0usize;
        let qp = self.qual.as_mut_ptr();
        for block in 0..nb {
            let m = unsafe { *self.maxima.get_unchecked(block) };
            unsafe { *qp.add(nq) = block as u32; }
            nq += (m > lower) as usize;
        }
        for t in 0..nq {
            let block = unsafe { *qp.add(t) } as usize;
            for (p, &code) in self.small[block * 32..block * 32 + 32].iter().enumerate() {
                if code > 0 && code as u64 - 1 >= lower {
                    out.push((block * 32 + p, (code - 1) as i64));
                }
            }
        }
    }
    fn refresh(&mut self) {
        if let Some(wide) = self.wide.as_ref() {
            for block in 0..self.maxima.len() {
                if self.dirty[block] {
                    self.maxima[block] = wide[block * 32..block * 32 + 32]
                        .iter()
                        .copied()
                        .max()
                        .unwrap();
                    self.dirty[block] = false;
                }
            }
        } else {
            for block in 0..self.maxima.len() {
                if self.dirty[block] {
                    self.maxima[block] = self.small[block * 32..block * 32 + 32]
                        .iter()
                        .copied()
                        .max()
                        .unwrap() as u64;
                    self.dirty[block] = false;
                }
            }
        }
    }
    pub(super) fn len(&self) -> usize {
        self.len
    }
    pub(super) fn best(&mut self) -> Option<usize> {
        if self.len == 0 {
            return None;
        }
        self.refresh();
        let maximum = self.maxima.iter().copied().max().unwrap();
        let block = self.maxima.iter().position(|&v| v == maximum).unwrap();
        let pos = if let Some(wide) = self.wide.as_ref() {
            wide[block * 32..block * 32 + 32]
                .iter()
                .position(|&v| v == maximum)
                .unwrap()
        } else {
            self.small[block * 32..block * 32 + 32]
                .iter()
                .position(|&v| v as u64 == maximum)
                .unwrap()
        };
        Some(block * 32 + pos)
    }
    pub(super) fn finalists(&mut self, bonus: u64, out: &mut Vec<(usize, i64)>) {
        out.clear();
        if self.len == 0 {
            return;
        }
        self.refresh();
        let mut first = 0u64;
        let mut second = 0u64;
        for &key in &self.maxima {
            if key > first {
                second = first;
                first = key;
            } else {
                second = second.max(key);
            }
        }
        let lower = second.saturating_sub(1).saturating_sub(bonus);
        for (block, &maximum) in self.maxima.iter().enumerate() {
            if maximum == 0 || maximum - 1 < lower {
                continue;
            }
            if let Some(wide) = self.wide.as_ref() {
                for (p, &code) in wide[block * 32..block * 32 + 32].iter().enumerate() {
                    if code > 0 && code - 1 >= lower {
                        out.push((block * 32 + p, (code - 1) as i64));
                    }
                }
            } else {
                for (p, &code) in self.small[block * 32..block * 32 + 32].iter().enumerate() {
                    if code > 0 && code as u64 - 1 >= lower {
                        out.push((block * 32 + p, (code - 1) as i64));
                    }
                }
            }
        }
    }
}

pub(super) struct Jumps {
    tables: Vec<[[u64; 256]; 8]>,
}

impl Jumps {
    fn step(mut x: u64) -> u64 {
        x ^= x << 7;
        x ^= x >> 9;
        x ^= x << 8;
        x
    }

    pub(super) fn new() -> Self {
        let mut powers = [0u64; 64];
        for bit in 0..64 {
            powers[bit] = Self::step(1u64 << bit);
        }
        let mut tables = Vec::with_capacity(16);
        for _ in 0..16 {
            let mut table = [[0u64; 256]; 8];
            for byte in 0..8 {
                for value in 1usize..256 {
                    let rest = value & (value - 1);
                    table[byte][value] =
                        table[byte][rest] ^ powers[byte * 8 + value.trailing_zeros() as usize];
                }
            }
            let mut squared = [0u64; 64];
            for bit in 0..64 {
                squared[bit] = Self::apply(&table, powers[bit]);
            }
            tables.push(table);
            powers = squared;
        }
        Self { tables }
    }

    #[inline(always)]
    fn apply(t: &[[u64; 256]; 8], x: u64) -> u64 {
        t[0][(x & 255) as usize]
            ^ t[1][((x >> 8) & 255) as usize]
            ^ t[2][((x >> 16) & 255) as usize]
            ^ t[3][((x >> 24) & 255) as usize]
            ^ t[4][((x >> 32) & 255) as usize]
            ^ t[5][((x >> 40) & 255) as usize]
            ^ t[6][((x >> 48) & 255) as usize]
            ^ t[7][(x >> 56) as usize]
    }

    #[inline]
    pub(super) fn advance(&self, state: &mut u64, mut count: usize) {
        debug_assert!(count <= u16::MAX as usize);
        if count < 8 {
            for _ in 0..count {
                *state = Self::step(*state);
            }
            return;
        }
        while count != 0 {
            let power = count.trailing_zeros() as usize;
            *state = Self::apply(&self.tables[power], *state);
            count &= count - 1;
        }
    }
}

}

#[allow(dead_code)]
mod exact_density {
#![allow(unused_imports, unused_parens, clippy::all)]
use tig_challenges::knapsack::*;
pub(super) struct Density {
    groups: [Vec<usize>; 11],
    positions: Vec<usize>,
}
impl Density {
    pub(super) fn new(weights: &[u32], include: &[bool]) -> Option<Self> {
        if weights.iter().any(|&w| !(1..=10).contains(&w)) {
            return None;
        }
        if weights.len() > u32::MAX as usize {
            return None;
        }
        let mut positions = vec![0usize; weights.len()];
        let mut groups: [Vec<usize>; 11] = Default::default();
        for (i, &w) in weights.iter().enumerate() {
            if include[i] {
                positions[i] = groups[w as usize].len();
                groups[w as usize].push(i);
            }
        }
        Some(Self { groups, positions })
    }
    fn extreme<const MAX: bool>(ids: &[usize], contrib: &[i32]) -> u64 {
        let mut result = if MAX { 0u64 } else { u64::MAX };
        unsafe {
            let ip = ids.as_ptr();
            let cp = contrib.as_ptr();
            macro_rules! visit {
                ($p:expr) => {{
                    let id = *ip.add($p);
                    let c = (*cp.add(id) as u32) ^ 0x80000000;
                    let key = ((c as u64) << 32)
                        | if MAX {
                            (u32::MAX - id as u32) as u64
                        } else {
                            id as u64
                        };
                    result = if MAX {
                        result.max(key)
                    } else {
                        result.min(key)
                    };
                }};
            }
            let mut p = 0usize;
            let full = ids.len() / 8 * 8;
            while p < full {
                visit!(p);
                visit!(p + 1);
                visit!(p + 2);
                visit!(p + 3);
                visit!(p + 4);
                visit!(p + 5);
                visit!(p + 6);
                visit!(p + 7);
                p += 8;
            }
            while p < ids.len() {
                visit!(p);
                p += 1;
            }
        }
        result
    }
    pub(super) fn pop(&mut self, contrib: &[i32], slack: u32, maximum: bool) -> Option<usize> {
        let mut chosen = None;
        let mut best = if maximum { i64::MIN } else { i64::MAX };
        let mut best_id = usize::MAX;
        for w in 1..=10usize.min(slack as usize) {
            let group = &self.groups[w];
            if group.is_empty() {
                continue;
            }
            let key = if maximum {
                Self::extreme::<true>(group, contrib)
            } else {
                Self::extreme::<false>(group, contrib)
            };
            let value = ((key >> 32) as u32 ^ 0x80000000) as i32;
            let id = if maximum {
                u32::MAX - key as u32
            } else {
                key as u32
            } as usize;
            let pos = self.positions[id];
            if maximum && value <= 0 {
                continue;
            }
            let score = value as i64 * 1000 / w as i64;
            if (if maximum { score > best } else { score < best })
                || (score == best && id < best_id)
            {
                best = score;
                best_id = id;
                chosen = Some((w, pos));
            }
        }
        chosen.map(|(w, pos)| {
            let id = self.groups[w].swap_remove(pos);
            if pos < self.groups[w].len() {
                self.positions[self.groups[w][pos]] = pos;
            }
            id
        })
    }

    pub(super) fn pop_affinity(
        &mut self,
        contrib: &[i32],
        affinity: &[i32],
        slack: u32,
    ) -> Option<usize> {
        assert!(self.positions.len() <= u16::MAX as usize);
        assert_eq!(contrib.len(), self.positions.len());
        assert_eq!(affinity.len(), self.positions.len());
        let mut chosen = None;
        let mut best = -1i64;
        let mut best_id = usize::MAX;
        for w in 1..=10usize.min(slack as usize) {
            let ids = &self.groups[w];
            let mut maxima = [0u64; 4];
            unsafe {
                let ip = ids.as_ptr();
                let cp = contrib.as_ptr();
                let ap = affinity.as_ptr();
                macro_rules! visit {
                    ($p:expr,$lane:expr) => {{
                        let id = *ip.add($p);
                        let c = *cp.add(id);
                        let numerator = c as i64 * 20 + (*ap.add(id)).max(0) as i64 * 7;
                        let key = if c > 0 {
                            ((numerator as u64) << 16) | (0xffff - id) as u64
                        } else {
                            0
                        };
                        maxima[$lane] = maxima[$lane].max(key);
                    }};
                }
                let mut p = 0;
                let full = ids.len() / 4 * 4;
                while p < full {
                    visit!(p, 0);
                    visit!(p + 1, 1);
                    visit!(p + 2, 2);
                    visit!(p + 3, 3);
                    p += 4;
                }
                while p < ids.len() {
                    visit!(p, 0);
                    p += 1;
                }
            }
            let maximum = maxima.into_iter().max().unwrap();
            if maximum == 0 {
                continue;
            }
            let id = 0xffff - (maximum as usize & 0xffff);
            let score = (maximum >> 16) as i64 * 50 / w as i64;
            if score > best || (score == best && id < best_id) {
                best = score;
                best_id = id;
                chosen = Some((w, self.positions[id]));
            }
        }
        chosen.map(|(w, pos)| {
            let id = self.groups[w].swap_remove(pos);
            if pos < self.groups[w].len() {
                self.positions[self.groups[w][pos]] = pos;
            }
            id
        })
    }
}

pub(super) unsafe fn top_three_sum(row: &[i32], ids: &[usize], exclude: usize) -> i64 {
    let mut tops = [[0i32; 4]; 3];
    let mut p = 0;
    while p + 4 <= ids.len() {
        for lane in 0..4 {
            let id = *ids.get_unchecked(p + lane);
            let x = *row.get_unchecked(id) & 0i32.wrapping_sub((id != exclude) as i32);
            tops[2][lane] = tops[2][lane].max(x.min(tops[1][lane]));
            tops[1][lane] = tops[1][lane].max(x.min(tops[0][lane]));
            tops[0][lane] = tops[0][lane].max(x);
        }
        p += 4;
    }
    let mut a = 0;
    let mut b = 0;
    let mut c = 0;
    for x in tops.into_iter().flatten() {
        c = c.max(x.min(b));
        b = b.max(x.min(a));
        a = a.max(x);
    }
    for &id in &ids[p..] {
        let x = *row.get_unchecked(id) & 0i32.wrapping_sub((id != exclude) as i32);
        c = c.max(x.min(b));
        b = b.max(x.min(a));
        a = a.max(x);
    }
    a as i64 + b as i64 + c as i64
}

}

#[allow(dead_code)]
mod exact_dp {
#![allow(unused_imports, unused_parens, clippy::all)]
use tig_challenges::knapsack::*;
pub fn fill(
    core: &[usize],
    weights: &[u32],
    values: &[i32],
    capacity: usize,
    choices: &mut Vec<u8>,
    scratch: &mut Vec<i32>,
) -> Option<usize> {
    let magnitude: i64 = core.iter().map(|&i| (values[i] as i64).abs()).sum();
    if magnitude >= (1i64 << 29) {
        return None;
    }
    const INIT: i32 = -(1 << 30);
    let width = capacity + 1;
    scratch.resize(2 * width, INIT);
    scratch[..2 * width].fill(INIT);
    choices.resize(core.len() * width, 0);
    choices[..core.len() * width].fill(0);
    scratch[0] = 0;
    let mut row = 0;
    let mut hi = 0usize;
    for (t, &item) in core.iter().enumerate() {
        let weight = weights[item] as usize;
        if weight > capacity {
            continue;
        }
        let value = values[item];
        let next_hi = (hi + weight).min(capacity);
        let (first, second) = scratch[..2 * width].split_at_mut(width);
        let (src, dst) = if row == 0 {
            (first, second)
        } else {
            (second, first)
        };
        dst[..weight].copy_from_slice(&src[..weight]);
        let source = &src[..=next_hi - weight];
        let prior = &src[weight..=next_hi];
        let output = &mut dst[weight..=next_hi];
        let bits = &mut choices[t * width + weight..=t * width + next_hi];
        for (((out, bit), &old), &from) in output.iter_mut().zip(bits).zip(prior).zip(source) {
            let candidate = from + value;
            *bit = (candidate > old) as u8;
            *out = candidate.max(old);
        }
        hi = next_hi;
        row ^= 1;
    }
    let last = &scratch[row * width..(row + 1) * width];
    Some((0..=capacity).max_by_key(|&w| last[w]).unwrap_or(0))
}

pub(super) fn strict_density_solution(
    packed: &[usize], selected: &[bool], weights: &[u32], capacity: u32,
) -> bool {
    assert_eq!(packed.len(), selected.len());
    assert_eq!(packed.len(), weights.len());
    let mut lowest_selected = i64::MAX;
    let mut highest_unused = i64::MIN;
    let mut weight = 0u64;
    for ((&key, &chosen), &w) in packed.iter().zip(selected).zip(weights) {
        let density = (key as i64) >> 16;
        lowest_selected = lowest_selected.min(if chosen { density } else { i64::MAX });
        highest_unused = highest_unused.max(if chosen { i64::MIN } else { density });
        weight += w as u64 * chosen as u64;
    }
    weight == capacity as u64 && lowest_selected > 0 && lowest_selected > highest_unused
}

}

#[allow(dead_code)]
mod exact_memo {
#![allow(unused_imports, unused_parens, clippy::all)]
use tig_challenges::knapsack::*;
pub(super) fn fingerprint(bits: &[bool], parameter: usize) -> u64 {
    let mut hash = 0x517cc1b727220a95u64 ^ parameter as u64;
    let mut i = 0;
    while i + 8 <= bits.len() {
        let word = unsafe { (bits.as_ptr().add(i) as *const u64).read_unaligned() };
        hash = (hash ^ word).wrapping_mul(0x9e3779b97f4a7c15);
        i += 8;
    }
    for &bit in &bits[i..] {
        hash = (hash ^ bit as u64).wrapping_mul(0x9e3779b97f4a7c15);
    }
    hash ^= hash >> 32;
    hash
}

pub(super) fn address(bits: &[bool], parameter: usize) -> usize {
    fingerprint(bits, parameter) as usize & 511
}

}

#[allow(dead_code)]
mod exact_neighbors {
#![allow(unused_imports, unused_parens, clippy::all)]
use tig_challenges::knapsack::*;

#[inline]
fn offer(list: &mut Vec<u64>, slots: usize, cutoff: &mut u64, key: u64) {
    if key <= *cutoff {
        return;
    }
    let at = list.partition_point(|&old| old > key);
    list.insert(at, key);
    if list.len() > slots {
        list.pop();
    }
    if list.len() == slots {
        *cutoff = list[slots - 1];
    }
}

pub(super) fn build(
    weights: &[u32],
    matrix: &[Vec<i32>],
    row_ptr: &[u32],
    columns: &[u16],
    k: usize,
    weight_aware: bool,
) -> Option<Vec<Vec<usize>>> {
    let n = weights.len();
    if n > u16::MAX as usize || k == 0 || weights.iter().any(|&w| !(1..=10).contains(&w)) {
        return None;
    }
    const LCM: u64 = 232_792_560;
    const FACTOR: [u64; 21] = [
        0,
        LCM,
        LCM / 2,
        LCM / 3,
        LCM / 4,
        LCM / 5,
        LCM / 6,
        LCM / 7,
        LCM / 8,
        LCM / 9,
        LCM / 10,
        LCM / 11,
        LCM / 12,
        LCM / 13,
        LCM / 14,
        LCM / 15,
        LCM / 16,
        LCM / 17,
        LCM / 18,
        LCM / 19,
        LCM / 20,
    ];
    let absolute_slots = if weight_aware { (k + 1) / 2 } else { k };
    let mut friends = Vec::with_capacity(n);
    let mut absolute = Vec::with_capacity(absolute_slots.min(n) + 1);
    let mut density = Vec::with_capacity(k.min(n) + 1);
    for i in 0..n {
        absolute.clear();
        density.clear();
        let mut absolute_cutoff = 0;
        let mut density_cutoff = 0;
        let row = &matrix[i];
        for &column in &columns[row_ptr[i] as usize..row_ptr[i + 1] as usize] {
            let j = column as usize;
            let interaction = row[j];
            if i == j || interaction <= 0 {
                continue;
            }
            if interaction > 2_000_000 {
                return None;
            }
            let tie = (u16::MAX as usize - j) as u64;
            offer(
                &mut absolute,
                absolute_slots,
                &mut absolute_cutoff,
                ((interaction as u64) << 16) | tie,
            );
            if weight_aware {
                let scaled = interaction as u64 * FACTOR[(weights[i] + weights[j]) as usize];
                offer(&mut density, k, &mut density_cutoff, (scaled << 16) | tie);
            }
        }
        let mut neighbors = Vec::with_capacity(k.min(n));
        neighbors.extend(
            absolute
                .iter()
                .map(|&key| u16::MAX as usize - (key as usize & 0xffff)),
        );
        if weight_aware {
            for &key in &density {
                if neighbors.len() >= k {
                    break;
                }
                let j = u16::MAX as usize - (key as usize & 0xffff);
                if !neighbors.contains(&j) {
                    neighbors.push(j);
                }
            }
        }
        friends.push(neighbors);
    }
    Some(friends)
}

}

#[allow(dead_code)]
mod exact_pairs {
#![allow(unused_imports, unused_parens, clippy::all)]
use tig_challenges::knapsack::*;
#[derive(Clone, Copy, Debug)]
pub struct Pair {
    pub a: usize,
    pub b: usize,
    pub weight: i64,
    pub value: i64,
    pub ordinal: usize,
}

pub fn best_exchange<F: Fn(usize, usize) -> i64>(
    additions: &mut [Pair],
    removals: &mut [Pair],
    slack: i64,
    cross: F,
) -> Option<(Pair, Pair)> {
    if additions.is_empty() || removals.is_empty() {
        return None;
    }
    if additions.iter().all(|p| (0..=20).contains(&p.weight)) {
        return best_by_weight(additions, removals, slack, cross);
    }
    additions.sort_unstable_by(|a, b| b.value.cmp(&a.value).then(a.ordinal.cmp(&b.ordinal)));
    removals.sort_unstable_by(|a, b| a.value.cmp(&b.value).then(a.ordinal.cmp(&b.ordinal)));
    let mut best = 0i64;
    let mut best_order = (usize::MAX, usize::MAX);
    let mut movement = None;
    for &r in removals.iter() {
        if additions[0].value - r.value < best {
            break;
        }
        let budget = slack + r.weight;
        for &a in additions.iter() {
            let bound = a.value - r.value;
            if bound < best {
                break;
            }
            let order = (r.ordinal, a.ordinal);
            if bound <= 0 || (bound == best && order >= best_order) || a.weight > budget {
                continue;
            }
            let delta =
                bound - cross(a.a, r.a) - cross(a.a, r.b) - cross(a.b, r.a) - cross(a.b, r.b);
            if delta > 0 && (delta > best || (delta == best && order < best_order)) {
                best = delta;
                best_order = order;
                movement = Some((r, a));
            }
        }
    }
    movement
}

fn best_by_weight<F: Fn(usize, usize) -> i64>(
    additions: &mut [Pair],
    removals: &mut [Pair],
    slack: i64,
    cross: F,
) -> Option<(Pair, Pair)> {
    let mut minimum = [i64::MAX; 21];
    for r in removals.iter() {
        let budget = slack + r.weight;
        if budget >= 0 {
            let b = budget.min(20) as usize;
            minimum[b] = minimum[b].min(r.value);
        }
    }
    for w in (0..20).rev() {
        minimum[w] = minimum[w].min(minimum[w + 1]);
    }
    let mut ai: Vec<usize> = additions
        .iter()
        .enumerate()
        .filter(|(_, a)| a.value > minimum[a.weight as usize])
        .map(|(i, _)| i)
        .collect();
    if ai.is_empty() {
        return None;
    }
    ai.sort_unstable_by(|&i, &j| {
        let a = &additions[i];
        let b = &additions[j];
        a.weight
            .cmp(&b.weight)
            .then(b.value.cmp(&a.value))
            .then(a.ordinal.cmp(&b.ordinal))
    });
    let mut starts = [0usize; 22];
    for &i in &ai {
        starts[additions[i].weight as usize + 1] += 1;
    }
    for w in 1..22 {
        starts[w] += starts[w - 1];
    }
    let mut maxima = [i64::MIN; 21];
    let mut groups = [0usize; 21];
    let mut ng = 0;
    for w in 0..21 {
        if starts[w] != starts[w + 1] {
            maxima[w] = additions[ai[starts[w]]].value;
            groups[ng] = w;
            ng += 1;
        }
    }
    groups[..ng].sort_unstable_by_key(|&w| std::cmp::Reverse(maxima[w]));
    for w in 1..21 {
        maxima[w] = maxima[w].max(maxima[w - 1]);
    }
    let mut ri: Vec<usize> = removals
        .iter()
        .enumerate()
        .filter(|(_, r)| {
            let budget = slack + r.weight;
            budget >= 0 && maxima[budget.min(20) as usize] > r.value
        })
        .map(|(i, _)| i)
        .collect();
    ri.sort_unstable_by(|&i, &j| {
        removals[i]
            .value
            .cmp(&removals[j].value)
            .then(removals[i].ordinal.cmp(&removals[j].ordinal))
    });
    let mut best = 0i64;
    let mut order = (usize::MAX, usize::MAX);
    let mut movement = None;
    for rindex in ri {
        let r = removals[rindex];
        if maxima[20] - r.value < best {
            break;
        }
        let limit = (slack + r.weight).min(20) as usize;
        if maxima[limit] - r.value < best {
            continue;
        }
        for &w in &groups[..ng] {
            let indices = &ai[starts[w]..starts[w + 1]];
            if additions[indices[0]].value - r.value < best {
                break;
            }
            if w > limit {
                continue;
            }
            for &aindex in indices {
                let a = additions[aindex];
                let bound = a.value - r.value;
                if bound < best {
                    break;
                }
                let at = (r.ordinal, a.ordinal);
                if bound <= 0 || (bound == best && at >= order) {
                    continue;
                }
                let delta =
                    bound - cross(a.a, r.a) - cross(a.a, r.b) - cross(a.b, r.a) - cross(a.b, r.b);
                if delta > 0 && (delta > best || (delta == best && at < order)) {
                    best = delta;
                    order = at;
                    movement = Some((r, a));
                }
            }
        }
    }
    movement
}

}

#[allow(dead_code)]
mod exact_compound {
#![allow(unused_imports, unused_parens, clippy::all)]
use tig_challenges::knapsack::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct Group {
    pub ids: [usize; 3],
    pub len: usize,
    pub weight: i64,
    pub value: i64,
    pub ordinal: usize,
}

pub(super) fn groups<W, C, Q>(
    items: &[usize],
    count: usize,
    weight: W,
    value: C,
    q: Q,
    add: bool,
) -> Vec<Group>
where
    W: Fn(usize) -> i64,
    C: Fn(usize) -> i64,
    Q: Fn(usize, usize) -> i64,
{
    let mut out = Vec::new();
    for (i, &a) in items.iter().enumerate() {
        if count == 1 {
            out.push(Group {
                ids: [a, 0, 0],
                len: 1,
                weight: weight(a),
                value: value(a),
                ordinal: out.len(),
            });
            continue;
        }
        for (j, &b) in items.iter().enumerate().skip(i + 1) {
            let pair = value(a) + value(b) + if add { q(a, b) } else { -q(a, b) };
            let pair_weight = weight(a) + weight(b);
            if count == 2 {
                out.push(Group {
                    ids: [a, b, 0],
                    len: 2,
                    weight: pair_weight,
                    value: pair,
                    ordinal: out.len(),
                });
            } else {
                for &c in &items[j + 1..] {
                    let extra = q(a, c) + q(b, c);
                    out.push(Group {
                        ids: [a, b, c],
                        len: 3,
                        weight: pair_weight + weight(c),
                        value: pair + value(c) + if add { extra } else { -extra },
                        ordinal: out.len(),
                    });
                }
            }
        }
    }
    out
}

pub(super) fn best<Q>(
    adds: &[Group],
    removes: &[Group],
    slack: i64,
    add_outer: bool,
    stop_nonpositive: bool,
    q: Q,
) -> Option<(Group, Group)>
where
    Q: Fn(usize, usize) -> i64,
{
    if adds.is_empty() || removes.is_empty() {
        return None;
    }
    let mut best = 0i64;
    let mut order = (usize::MAX, usize::MAX);
    let mut result = None;
    let consider = |r: Group, a: Group| {
        let mut delta = a.value - r.value;
        for &i in &a.ids[..a.len] {
            for &j in &r.ids[..r.len] {
                delta -= q(i, j);
            }
        }
        let key = if add_outer {
            (a.ordinal, r.ordinal)
        } else {
            (r.ordinal, a.ordinal)
        };
        (delta, key)
    };
    if adds
        .iter()
        .chain(removes)
        .any(|g| !(0..=30).contains(&g.weight))
    {
        if add_outer {
            for &a in adds {
                if stop_nonpositive && a.value <= 0 && best > 0 {
                    break;
                }
                for &r in removes {
                    if a.weight > slack + r.weight {
                        continue;
                    }
                    let (delta, key) = consider(r, a);
                    if delta > 0 && (delta > best || (delta == best && key < order)) {
                        best = delta;
                        order = key;
                        result = Some((r, a));
                    }
                }
            }
        } else {
            for &r in removes {
                for &a in adds {
                    if a.weight > slack + r.weight {
                        continue;
                    }
                    let (delta, key) = consider(r, a);
                    if delta > 0 && (delta > best || (delta == best && key < order)) {
                        best = delta;
                        order = key;
                        result = Some((r, a));
                    }
                }
            }
        }
        return result;
    }
    let mut buckets: [Vec<usize>; 31] = Default::default();
    if add_outer {
        let mut maximum = [i64::MIN; 31];
        for a in adds {
            maximum[a.weight as usize] = maximum[a.weight as usize].max(a.value);
        }
        for w in 1..31 {
            maximum[w] = maximum[w].max(maximum[w - 1]);
        }
        for (i, r) in removes.iter().enumerate() {
            let budget = slack + r.weight;
            if budget >= 0 && maximum[budget.min(30) as usize] > r.value {
                buckets[r.weight as usize].push(i);
            }
        }
        for b in &mut buckets {
            b.sort_unstable_by_key(|&i| (removes[i].value, removes[i].ordinal));
        }
        let mut weights: Vec<usize> = (0..31).filter(|&w| !buckets[w].is_empty()).collect();
        weights.sort_unstable_by_key(|&w| removes[buckets[w][0]].value);
        for &a in adds {
            if stop_nonpositive && a.value <= 0 && best > 0 {
                break;
            }
            for &w in &weights {
                if a.value - removes[buckets[w][0]].value < best {
                    break;
                }
                if a.weight > slack + w as i64 {
                    continue;
                }
                for &i in &buckets[w] {
                    let r = removes[i];
                    if a.value - r.value < best {
                        break;
                    }
                    let (delta, key) = consider(r, a);
                    if delta > 0 && (delta > best || (delta == best && key < order)) {
                        best = delta;
                        order = key;
                        result = Some((r, a));
                    }
                }
            }
        }
    } else {
        let mut minimum = [i64::MAX; 31];
        for r in removes {
            minimum[r.weight as usize] = minimum[r.weight as usize].min(r.value);
        }
        for w in (0..30).rev() {
            minimum[w] = minimum[w].min(minimum[w + 1]);
        }
        for (i, a) in adds.iter().enumerate() {
            let need = (a.weight - slack).max(0);
            if need <= 30 && a.value > minimum[need as usize] {
                buckets[a.weight as usize].push(i);
            }
        }
        for b in &mut buckets {
            b.sort_unstable_by_key(|&i| (std::cmp::Reverse(adds[i].value), adds[i].ordinal));
        }
        let mut weights: Vec<usize> = (0..31).filter(|&w| !buckets[w].is_empty()).collect();
        weights.sort_unstable_by_key(|&w| std::cmp::Reverse(adds[buckets[w][0]].value));
        for &r in removes {
            for &w in &weights {
                if adds[buckets[w][0]].value - r.value < best {
                    break;
                }
                if w as i64 > slack + r.weight {
                    continue;
                }
                for &i in &buckets[w] {
                    let a = adds[i];
                    if a.value - r.value < best {
                        break;
                    }
                    let (delta, key) = consider(r, a);
                    if delta > 0 && (delta > best || (delta == best && key < order)) {
                        best = delta;
                        order = key;
                        result = Some((r, a));
                    }
                }
            }
        }
    }
    result
}

}

#[allow(dead_code)]
mod exact_single {
#![allow(unused_imports, unused_parens, clippy::all)]
use tig_challenges::knapsack::*;
pub(super) fn best<F: Fn(usize, usize) -> i32>(
    unused: &[usize],
    used: &[usize],
    weights: &[u32],
    contrib: &[i32],
    slack: u32,
    cross: F,
) -> Option<(usize, usize)> {
    if unused.len() > 256 {
        return best_heap(unused, used, weights, contrib, slack, cross);
    }
    if unused
        .iter()
        .chain(used)
        .any(|&i| weights[i] < 1 || weights[i] > 10 || (contrib[i] as i64).abs() >= (1i64 << 28))
    {
        let mut best = 0i32;
        let mut movement = None;
        for &r in used {
            for &a in unused {
                if weights[a] > weights[r] + slack {
                    continue;
                }
                let delta = contrib[a]
                    .wrapping_sub(contrib[r])
                    .wrapping_sub(cross(r, a));
                if delta > best {
                    best = delta;
                    movement = Some((a, r));
                }
            }
        }
        return movement;
    }
    let mut max_add = [i32::MIN; 11];
    let mut min_remove = [i32::MAX; 12];
    for &a in unused {
        max_add[weights[a] as usize] = max_add[weights[a] as usize].max(contrib[a]);
    }
    for &r in used {
        min_remove[weights[r] as usize] = min_remove[weights[r] as usize].min(contrib[r]);
    }
    for w in 1..=10 {
        max_add[w] = max_add[w].max(max_add[w - 1]);
    }
    for w in (1..=10).rev() {
        min_remove[w] = min_remove[w].min(min_remove[w + 1]);
    }
    if !used
        .iter()
        .any(|&r| max_add[(weights[r] as u64 + slack as u64).min(10) as usize] > contrib[r])
    {
        return None;
    }
    let mut storage = [0u64; 11 * 256];
    let mut counts = [0usize; 11];
    for (ordinal, &a) in unused.iter().enumerate() {
        let w = weights[a] as usize;
        let lower = weights[a].saturating_sub(slack).max(1) as usize;
        if contrib[a] <= min_remove[lower] {
            continue;
        }
        let c = ((contrib[a] as u32) ^ 0x80000000) as u64;
        let entry = (c << 32) | (u32::MAX as usize - ordinal) as u64;
        storage[w * 256 + counts[w]] = entry;
        counts[w] += 1;
    }
    for w in 1..=10 {
        let __lo = w * 256; let bucket = &mut storage[__lo..__lo + counts[w]];
        bucket.sort_unstable_by(|a, b| b.cmp(a));
    }
    let mut buckets: [&[u64]; 11] = [&[]; 11];
    for w in 0..11 {
        let __lo = w * 256;
        let __n = counts[w];
        buckets[w] = &storage[__lo..__lo + __n];
    }
    let buckets = buckets;
    let mut best = 0i64;
    let mut best_order = (usize::MAX, usize::MAX);
    let mut movement = None;
    for (ri, &r) in used.iter().enumerate() {
        let limit = (weights[r] as u64 + slack as u64).min(10) as usize;
        if max_add[limit] as i64 - contrib[r] as i64 <= best {
            continue;
        }
        for bucket in &buckets[1..=limit] {
            for &entry in bucket.iter() {
                let value = ((entry >> 32) as u32 ^ 0x80000000) as i32;
                let bound = value as i64 - contrib[r] as i64;
                if bound < best || bound <= 0 {
                    break;
                }
                let ai = u32::MAX as usize - entry as u32 as usize;
                let order = (ri, ai);
                if bound == best && order >= best_order {
                    continue;
                }
                let a = unused[ai];
                let delta = bound - cross(r, a) as i64;
                if delta > 0 && (delta > best || (delta == best && order < best_order)) {
                    best = delta;
                    best_order = order;
                    movement = Some((a, r));
                }
            }
        }
    }
    movement
}

pub(super) fn best_heap<F: Fn(usize, usize) -> i32>(
    unused: &[usize],
    used: &[usize],
    weights: &[u32],
    contrib: &[i32],
    slack: u32,
    cross: F,
) -> Option<(usize, usize)> {
    if unused
        .iter()
        .chain(used)
        .any(|&i| weights[i] < 1 || weights[i] > 10 || (contrib[i] as i64).abs() >= (1i64 << 28))
    {
        let mut best = 0i32;
        let mut movement = None;
        for &r in used {
            for &a in unused {
                if weights[a] > weights[r] + slack {
                    continue;
                }
                let delta = contrib[a]
                    .wrapping_sub(contrib[r])
                    .wrapping_sub(cross(r, a));
                if delta > best {
                    best = delta;
                    movement = Some((a, r));
                }
            }
        }
        return movement;
    }
    let mut max_add = [i32::MIN; 11];
    let mut min_remove = [i32::MAX; 12];
    for &a in unused {
        max_add[weights[a] as usize] = max_add[weights[a] as usize].max(contrib[a]);
    }
    for &r in used {
        min_remove[weights[r] as usize] = min_remove[weights[r] as usize].min(contrib[r]);
    }
    for w in 1..=10 {
        max_add[w] = max_add[w].max(max_add[w - 1]);
    }
    for w in (1..=10).rev() {
        min_remove[w] = min_remove[w].min(min_remove[w + 1]);
    }
    if !used
        .iter()
        .any(|&r| max_add[(weights[r] as u64 + slack as u64).min(10) as usize] > contrib[r])
    {
        return None;
    }
    let mut buckets: [Vec<u64>; 11] = Default::default();
    for (ordinal, &a) in unused.iter().enumerate() {
        let w = weights[a] as usize;
        let lower = weights[a].saturating_sub(slack).max(1) as usize;
        if contrib[a] <= min_remove[lower] {
            continue;
        }
        let c = ((contrib[a] as u32) ^ 0x80000000) as u64;
        buckets[w].push((c << 32) | (u32::MAX as usize - ordinal) as u64);
    }
    for bucket in &mut buckets {
        bucket.sort_unstable_by(|a, b| b.cmp(a));
    }
    let mut best = 0i64;
    let mut best_order = (usize::MAX, usize::MAX);
    let mut movement = None;
    for (ri, &r) in used.iter().enumerate() {
        let limit = (weights[r] as u64 + slack as u64).min(10) as usize;
        if max_add[limit] as i64 - contrib[r] as i64 <= best {
            continue;
        }
        for bucket in &buckets[1..=limit] {
            for &entry in bucket {
                let value = ((entry >> 32) as u32 ^ 0x80000000) as i32;
                let bound = value as i64 - contrib[r] as i64;
                if bound < best || bound <= 0 {
                    break;
                }
                let ai = u32::MAX as usize - entry as u32 as usize;
                let order = (ri, ai);
                if bound == best && order >= best_order {
                    continue;
                }
                let a = unused[ai];
                let delta = bound - cross(r, a) as i64;
                if delta > 0 && (delta > best || (delta == best && order < best_order)) {
                    best = delta;
                    best_order = order;
                    movement = Some((a, r));
                }
            }
        }
    }
    movement
}

pub(super) fn best_fused<F: Fn(usize, usize) -> i32>(
    unused: &[usize],
    used: &[usize],
    weights: &[u32],
    contrib: &[i32],
    slack: u32,
    cross: F,
) -> Option<(usize, usize)> {
    if unused.len() > 256 {
        return best_heap(unused, used, weights, contrib, slack, cross);
    }
    let mut max_add = [i32::MIN; 11];
    let mut min_remove = [i32::MAX; 12];
    for &a in unused {
        let w = weights[a];
        if !(1..=10).contains(&w) {
            return best_heap(unused, used, weights, contrib, slack, cross);
        }
        let c = contrib[a];
        if (c as i64).abs() >= (1i64 << 28) {
            return best_heap(unused, used, weights, contrib, slack, cross);
        }
        max_add[w as usize] = max_add[w as usize].max(c);
    }
    for &r in used {
        let w = weights[r];
        if !(1..=10).contains(&w) {
            return best_heap(unused, used, weights, contrib, slack, cross);
        }
        let c = contrib[r];
        if (c as i64).abs() >= (1i64 << 28) {
            return best_heap(unused, used, weights, contrib, slack, cross);
        }
        min_remove[w as usize] = min_remove[w as usize].min(c);
    }
    let weight = |i: usize| unsafe { *weights.get_unchecked(i) };
    let contribution = |i: usize| unsafe { *contrib.get_unchecked(i) };
    for w in 1..=10 {
        max_add[w] = max_add[w].max(max_add[w - 1]);
    }
    for w in (1..=10).rev() {
        min_remove[w] = min_remove[w].min(min_remove[w + 1]);
    }
    if !used
        .iter()
        .any(|&r| max_add[(weight(r) as u64 + slack as u64).min(10) as usize] > contribution(r))
    {
        return None;
    }
    let mut storage = [0u64; 11 * 256];
    let mut counts = [0usize; 11];
    for (ordinal, &a) in unused.iter().enumerate() {
        let w = weight(a) as usize;
        let lower = weight(a).saturating_sub(slack).max(1) as usize;
        if contribution(a) <= unsafe { *min_remove.get_unchecked(lower) } {
            continue;
        }
        let c = ((contribution(a) as u32) ^ 0x80000000) as u64;
        let entry = (c << 32) | (u32::MAX as usize - ordinal) as u64;
        let __i = unsafe { *counts.get_unchecked(w) };
        storage[w * 256 + __i] = entry;
        unsafe {
            *counts.get_unchecked_mut(w) += 1;
        }
    }
    for w in 1..=10 {
        let __lo = w * 256; let bucket = &mut storage[__lo..__lo + counts[w]];
        bucket.sort_unstable_by(|a, b| b.cmp(a));
    }
    let mut buckets: [&[u64]; 11] = [&[]; 11];
    for w in 0..11 {
        let __lo = w * 256;
        let __n = counts[w];
        buckets[w] = &storage[__lo..__lo + __n];
    }
    let buckets = buckets;
    let mut best = 0i64;
    let mut best_order = (usize::MAX, usize::MAX);
    let mut movement = None;
    for (ri, &r) in used.iter().enumerate() {
        let limit = (weight(r) as u64 + slack as u64).min(10) as usize;
        if max_add[limit] as i64 - contribution(r) as i64 <= best {
            continue;
        }
        for bucket in &buckets[1..=limit] {
            for &entry in bucket.iter() {
                let value = ((entry >> 32) as u32 ^ 0x80000000) as i32;
                let bound = value as i64 - contribution(r) as i64;
                if bound < best || bound <= 0 {
                    break;
                }
                let ai = u32::MAX as usize - entry as u32 as usize;
                let order = (ri, ai);
                if bound == best && order >= best_order {
                    continue;
                }
                let a = unsafe { *unused.get_unchecked(ai) };
                let delta = bound - cross(r, a) as i64;
                if delta > 0 && (delta > best || (delta == best && order < best_order)) {
                    best = delta;
                    best_order = order;
                    movement = Some((a, r));
                }
            }
        }
    }
    movement
}

}

#[allow(dead_code)]
mod exact_windows {
#![allow(unused_imports, unused_parens, clippy::all)]
use tig_challenges::knapsack::*;

const SCALE: [i64; 11] = [0, 2520, 1260, 840, 630, 504, 420, 360, 315, 280, 252];
type Original = (usize, i64, i64);

pub(super) struct Windows<'a> {
    weights: &'a [u32],
    packed: bool,
    unused: Vec<usize>,
    used: Vec<usize>,
    original_unused: Vec<Original>,
    original_used: Vec<Original>,
}

impl<'a> Windows<'a> {
    pub(super) fn new(weights: &'a [u32]) -> Self {
        let n = weights.len();
        let packed = usize::BITS == 64
            && n <= u16::MAX as usize
            && if super::fast(1 << 46) {
                super::MINW.load(std::sync::atomic::Ordering::Relaxed) >= 1
                    && super::MAXW.load(std::sync::atomic::Ordering::Relaxed) <= 10
            } else {
                weights.iter().all(|&w| (1..=10).contains(&w))
            };
        Self {
            weights,
            packed,
            unused: Vec::with_capacity(if packed { n } else { 0 }),
            used: Vec::with_capacity(if packed { n } else { 0 }),
            original_unused: Vec::with_capacity(if packed { 0 } else { n }),
            original_used: Vec::with_capacity(if packed { 0 } else { n }),
        }
    }

    pub(super) fn is_packed(&self) -> bool {
        self.packed
    }

    pub(super) fn build(
        &mut self,
        contrib: &[i32],
        selected: &[bool],
        k: usize,
        best_unused: &mut Vec<usize>,
        worst_used: &mut Vec<usize>,
    ) {
        let weights = self.weights;
        assert_eq!(weights.len(), contrib.len());
        assert_eq!(weights.len(), selected.len());
        best_unused.clear();
        worst_used.clear();
        if !self.packed {
            self.original_unused.clear();
            self.original_used.clear();
            for i in 0..weights.len() {
                let score = (i, contrib[i] as i64, (weights[i] as i64).max(1));
                if selected[i] {
                    self.original_used.push(score);
                } else {
                    self.original_unused.push(score);
                }
            }
            let ku = k.min(self.original_unused.len());
            let ks = k.min(self.original_used.len());
            if ku > 0 && ku < self.original_unused.len() {
                self.original_unused
                    .select_nth_unstable_by(ku - 1, |a, b| (b.1 * a.2).cmp(&(a.1 * b.2)));
            }
            if ks > 0 && ks < self.original_used.len() {
                self.original_used
                    .select_nth_unstable_by(ks - 1, |a, b| (a.1 * b.2).cmp(&(b.1 * a.2)));
            }
            best_unused.extend(self.original_unused[..ku].iter().map(|t| t.0));
            worst_used.extend(self.original_used[..ks].iter().map(|t| t.0));
            return;
        }

        self.unused.clear();
        self.used.clear();
        let mut nu = 0usize;
        let mut ns = 0usize;
        let unused = self.unused.as_mut_ptr();
        let used = self.used.as_mut_ptr();
        macro_rules! emit {
            ($i:expr) => {{
                let i = $i;
                let (c, factor, is_selected) = unsafe {
                    (
                        *contrib.get_unchecked(i) as i64,
                        *SCALE.get_unchecked(*weights.get_unchecked(i) as usize),
                        *selected.get_unchecked(i),
                    )
                };
                let entry = ((c * factor) << 16) as usize | i;
                unsafe {
                    unused.add(nu).write(entry);
                    used.add(ns).write(entry);
                }
                ns += is_selected as usize;
                nu += (!is_selected) as usize;
            }};
        }
        let mut i = 0usize;
        let end = weights.len() / 16 * 16;
        while i < end {
            emit!(i);
            emit!(i + 1);
            emit!(i + 2);
            emit!(i + 3);
            emit!(i + 4);
            emit!(i + 5);
            emit!(i + 6);
            emit!(i + 7);
            emit!(i + 8);
            emit!(i + 9);
            emit!(i + 10);
            emit!(i + 11);
            emit!(i + 12);
            emit!(i + 13);
            emit!(i + 14);
            emit!(i + 15);
            i += 16;
        }
        while i < weights.len() {
            emit!(i);
            i += 1;
        }
        unsafe {
            self.unused.set_len(nu);
            self.used.set_len(ns);
        }
        let ku = k.min(self.unused.len());
        let ks = k.min(self.used.len());
        if super::exact_sort::compatible() {
            if ku > 0 && ku < self.unused.len() {
                super::exact_sort::select(&mut self.unused, ku - 1, |a, b| {
                    ((a as i64) >> 16) > ((b as i64) >> 16)
                });
            }
            if ks > 0 && ks < self.used.len() {
                super::exact_sort::select(&mut self.used, ks - 1, |a, b| {
                    ((a as i64) >> 16) < ((b as i64) >> 16)
                });
            }
        } else {
            let ku = k.min(self.unused.len());
            let ks = k.min(self.used.len());
            if ku > 0 && ku < self.unused.len() {
                self.unused.select_nth_unstable_by(ku - 1, |&a, &b| {
                    ((b as i64) >> 16).cmp(&((a as i64) >> 16))
                });
            }
            if ks > 0 && ks < self.used.len() {
                self.used.select_nth_unstable_by(ks - 1, |&a, &b| {
                    ((a as i64) >> 16).cmp(&((b as i64) >> 16))
                });
            }
        }
        best_unused.extend(self.unused[..ku].iter().map(|&entry| entry & 0xffff));
        worst_used.extend(self.used[..ks].iter().map(|&entry| entry & 0xffff));
    }
}

}

#[allow(dead_code)]
mod exact_paths {
#![allow(unused_imports, unused_parens, clippy::all)]
use tig_challenges::knapsack::*;
use std::rc::Rc;
pub(super) struct Snapshot {
    pub bits: Vec<bool>,
    pub contrib: Vec<i32>,
    pub value: i64,
    pub weight: u32,
    pub hash: u64,
}
impl Snapshot {
    pub(super) fn new(bits: &[bool], contrib: &[i32], value: i64, weight: u32, hash: u64) -> Self {
        Self {
            bits: bits.to_vec(),
            contrib: contrib.to_vec(),
            value,
            weight,
            hash,
        }
    }
}
struct Entry {
    parameter: usize,
    input: Snapshot,
    output: Rc<Snapshot>,
    steps: usize,
}
pub(super) struct Paths {
    slots: Vec<Option<Entry>>,
}
impl Paths {
    pub(super) fn new() -> Self {
        Self { slots: Vec::new() }
    }
    pub(super) const fn new_const() -> Self {
        Self { slots: Vec::new() }
    }
    pub(super) fn clear(&mut self) {
        self.slots = Vec::new();
    }
    fn index(hash: u64, parameter: usize) -> usize {
        ((hash ^ (parameter as u64).wrapping_mul(0x9e3779b97f4a7c15))
            .wrapping_mul(0xbf58476d1ce4e5b9)
            >> 53) as usize
    }
    pub(super) fn lookup(
        &self,
        parameter: usize,
        hash: u64,
        bits: &[bool],
        contrib: &[i32],
        value: i64,
        weight: u32,
        remaining: usize,
    ) -> Option<(Rc<Snapshot>, usize)> {
        let entry = self.slots.get(Self::index(hash, parameter))?.as_ref()?;
        let s = &entry.input;
        if entry.parameter == parameter
            && entry.steps <= remaining
            && s.hash == hash
            && s.value == value
            && s.weight == weight
            && s.bits == bits
            && s.contrib == contrib
        {
            Some((Rc::clone(&entry.output), entry.steps))
        } else {
            None
        }
    }
    pub(super) fn publish(
        &mut self,
        parameter: usize,
        pending: Vec<(Snapshot, usize)>,
        end: usize,
        output: Rc<Snapshot>,
    ) {
        if self.slots.is_empty() {
            self.slots.resize_with(2048, || None);
        }
        for (input, start) in pending {
            let index = Self::index(input.hash, parameter);
            self.slots[index] = Some(Entry {
                parameter,
                input,
                output: Rc::clone(&output),
                steps: end - start,
            });
        }
    }
}

}

#[allow(dead_code)]
mod exact_updates {
#![allow(unused_imports, unused_parens, clippy::all)]
use tig_challenges::knapsack::*;
pub(super) fn apply(contrib: &mut [i32], q: &[Vec<i32>], ids: &[usize], add: bool) -> i64 {
    let n = contrib.len();
    let mut delta = 0i64;
    for group in ids.chunks(4) {
        for (p, &i) in group.iter().enumerate() {
            let mut c = contrib[i];
            for &prior in &group[..p] {
                c = if add {
                    c.wrapping_add(q[prior][i])
                } else {
                    c.wrapping_sub(q[prior][i])
                };
            }
            delta = if add {
                delta.wrapping_add(c as i64)
            } else {
                delta.wrapping_sub(c as i64)
            };
        }
        if group.len() == 4 {
            let a = q[group[0]].as_ptr();
            let b = q[group[1]].as_ptr();
            let c = q[group[2]].as_ptr();
            let d = q[group[3]].as_ptr();
            let out = contrib.as_mut_ptr();
            unsafe {
                if add {
                    for i in 0..n {
                        *out.add(i) = (*out.add(i))
                            .wrapping_add(*a.add(i))
                            .wrapping_add(*b.add(i))
                            .wrapping_add(*c.add(i))
                            .wrapping_add(*d.add(i));
                    }
                } else {
                    for i in 0..n {
                        *out.add(i) = (*out.add(i))
                            .wrapping_sub(*a.add(i))
                            .wrapping_sub(*b.add(i))
                            .wrapping_sub(*c.add(i))
                            .wrapping_sub(*d.add(i));
                    }
                }
            }
        } else {
            for &id in group {
                let row = &q[id];
                if add {
                    for (c, &v) in contrib.iter_mut().zip(row) {
                        *c = c.wrapping_add(v);
                    }
                } else {
                    for (c, &v) in contrib.iter_mut().zip(row) {
                        *c = c.wrapping_sub(v);
                    }
                }
            }
        }
    }
    delta
}
pub(super) fn differences(
    selected: &[bool],
    target: &[u8],
    removed: &mut Vec<usize>,
    added: &mut Vec<usize>,
) {
    assert_eq!(selected.len(), target.len());
    let n = selected.len();
    removed.clear();
    added.clear();
    removed.reserve(n);
    added.reserve(n);
    let mut nr = 0;
    let mut na = 0;
    unsafe {
        let rp = removed.as_mut_ptr();
        let ap = added.as_mut_ptr();
        let sp = selected.as_ptr();
        let tp = target.as_ptr();
        macro_rules! emit {
            ($i:expr) => {{
                let i = $i;
                let s = *sp.add(i);
                let t = *tp.add(i) != 0;
                rp.add(nr).write(i);
                ap.add(na).write(i);
                nr += (s && !t) as usize;
                na += (!s && t) as usize;
            }};
        }
        let mut i = 0;
        let full = n / 16 * 16;
        while i < full {
            emit!(i);
            emit!(i + 1);
            emit!(i + 2);
            emit!(i + 3);
            emit!(i + 4);
            emit!(i + 5);
            emit!(i + 6);
            emit!(i + 7);
            emit!(i + 8);
            emit!(i + 9);
            emit!(i + 10);
            emit!(i + 11);
            emit!(i + 12);
            emit!(i + 13);
            emit!(i + 14);
            emit!(i + 15);
            i += 16;
        }
        while i < n {
            emit!(i);
            i += 1;
        }
        removed.set_len(nr);
        added.set_len(na);
    }
}

pub(super) fn bool_bytes(bits: &[bool]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(bits.as_ptr().cast::<u8>(), bits.len()) }
}

pub(super) fn value_delta(
    contrib: &[i32],
    q: &[Vec<i32>],
    removed: &[usize],
    added: &[usize],
) -> i64 {
    let n = contrib.len();
    let mut r: Vec<i32> = removed.iter().map(|&i| contrib[i]).collect();
    let mut a: Vec<i32> = added.iter().map(|&i| contrib[i]).collect();
    for (p, &id) in removed.iter().enumerate() {
        let row = &q[id][..n];
        adjust::<false>(&mut r[p + 1..], &removed[p + 1..], row);
        adjust::<false>(&mut a, added, row);
    }
    for (p, &id) in added.iter().enumerate() {
        adjust::<true>(&mut a[p + 1..], &added[p + 1..], &q[id][..n]);
    }
    let removed_value: i64 = r.iter().map(|&c| c as i64).sum();
    let added_value: i64 = a.iter().map(|&c| c as i64).sum();
    added_value.wrapping_sub(removed_value)
}
fn adjust<const ADD: bool>(values: &mut [i32], ids: &[usize], row: &[i32]) {
    debug_assert_eq!(values.len(), ids.len());
    unsafe {
        let vp = values.as_mut_ptr();
        let ip = ids.as_ptr();
        let qp = row.as_ptr();
        macro_rules! visit {
            ($p:expr) => {{
                let p = $p;
                let old = *vp.add(p);
                let q = *qp.add(*ip.add(p));
                *vp.add(p) = if ADD {
                    old.wrapping_add(q)
                } else {
                    old.wrapping_sub(q)
                };
            }};
        }
        let mut p = 0;
        let full = ids.len() / 8 * 8;
        while p < full {
            visit!(p);
            visit!(p + 1);
            visit!(p + 2);
            visit!(p + 3);
            visit!(p + 4);
            visit!(p + 5);
            visit!(p + 6);
            visit!(p + 7);
            p += 8;
        }
        while p < ids.len() {
            visit!(p);
            p += 1;
        }
    }
}

}

#[allow(dead_code)]
mod exact_terminal {
#![allow(unused_imports, unused_parens, clippy::all)]
use tig_challenges::knapsack::*;
struct Proof {
    parameter: usize,
    hash: u64,
    bits: Vec<bool>,
    contrib: Vec<i32>,
    value: i64,
    weight: u32,
}
pub(super) struct Terminal {
    slots: Vec<Option<Proof>>,
}
impl Terminal {
    pub(super) fn new() -> Self {
        Self { slots: Vec::new() }
    }
    pub(super) fn clear(&mut self) {
        self.slots = Vec::new();
    }
    fn index(hash: u64, parameter: usize) -> usize {
        ((hash ^ (parameter as u64).wrapping_mul(0x9e3779b97f4a7c15))
            .wrapping_mul(0xbf58476d1ce4e5b9)
            >> 53) as usize
    }
    pub(super) fn contains(
        &self,
        parameter: usize,
        hash: u64,
        bits: &[bool],
        contrib: &[i32],
        value: i64,
        weight: u32,
    ) -> bool {
        if let Some(Some(p)) = self.slots.get(Self::index(hash, parameter)) {
            p.parameter == parameter
                && p.hash == hash
                && p.value == value
                && p.weight == weight
                && p.bits == bits
                && p.contrib == contrib
        } else {
            false
        }
    }
    pub(super) fn insert(
        &mut self,
        parameter: usize,
        hash: u64,
        bits: &[bool],
        contrib: &[i32],
        value: i64,
        weight: u32,
    ) {
        if self.slots.is_empty() {
            self.slots.resize_with(2048, || None);
        }
        let p = Proof {
            parameter,
            hash,
            bits: bits.to_vec(),
            contrib: contrib.to_vec(),
            value,
            weight,
        };
        self.slots[Self::index(hash, parameter)] = Some(p);
    }
}

}
