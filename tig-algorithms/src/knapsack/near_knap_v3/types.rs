use std::collections::BTreeMap;
use tig_challenges::knapsack::*;

pub const DIFF_LIM: usize = 9;
pub const CORE_HALF: usize = 25;
pub const N_IT_CONSTRUCT: usize = 2;

pub const MICRO_K: usize = 16;
pub const MICRO_RM_K: usize = 8;
pub const MICRO_ADD_K: usize = 8;

#[derive(Clone, Copy)]
pub struct Rng {
    pub state: u64,
}

impl Rng {
    pub fn from_seed(seed: &[u8; 32]) -> Self {
        let mut s: u64 = 0x9E3779B97F4A7C15;
        for (i, &b) in seed.iter().enumerate() {
            s ^= (b as u64) << ((i & 7) * 8);
            s = s.rotate_left(7).wrapping_mul(0xBF58476D1CE4E5B9);
        }
        if s == 0 {
            s = 1;
        }
        Self { state: s }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 7;
        x ^= x >> 9;
        x ^= x << 8;
        self.state = x;
        x
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
}

pub struct State<'a> {
    pub ch: &'a Challenge,
    pub selected_bit: Vec<bool>,
    pub contrib: Vec<i32>,
    pub support: Vec<u16>,
    pub total_interactions: &'a [i64],
    pub hubs_static: &'a [usize],
    pub neigh: Option<&'a Vec<Vec<(u16, i16)>>>,
    pub total_value: i64,
    pub total_weight: u32,
    pub window_locked: Vec<usize>,
    pub window_core: Vec<usize>,
    pub window_rejected: Vec<usize>,
    pub core_bins: Vec<(u32, Vec<usize>)>,
    pub usage: Vec<u16>,
    pub dp_cache: Vec<i64>,
    pub choose_cache: Vec<u8>,
    pub snap_bits: Vec<bool>,
    pub snap_contrib: Vec<i32>,
    pub snap_support: Vec<u16>,
}

impl<'a> State<'a> {
    pub fn new_empty(
        ch: &'a Challenge,
        total_interactions: &'a [i64],
        hubs_static: &'a [usize],
        neigh: Option<&'a Vec<Vec<(u16, i16)>>>,
    ) -> Self {
        let n = ch.num_items;
        let mut contrib = vec![0i32; n];
        for i in 0..n {
            contrib[i] = ch.values[i] as i32;
        }

        Self {
            ch,
            selected_bit: vec![false; n],
            contrib,
            support: vec![0u16; n],
            total_interactions,
            hubs_static,
            neigh,
            total_value: 0,
            total_weight: 0,
            window_locked: Vec::new(),
            window_core: Vec::new(),
            window_rejected: Vec::new(),
            core_bins: Vec::new(),
            usage: vec![0u16; n],
            dp_cache: Vec::new(),
            choose_cache: Vec::new(),
            snap_bits: vec![false; n],
            snap_contrib: vec![0i32; n],
            snap_support: vec![0u16; n],
        }
    }

    pub fn selected_items(&self) -> Vec<usize> {
        let n = self.ch.num_items;
        let mut out: Vec<usize> = Vec::new();
        out.reserve(n / 8);
        for i in 0..n {
            if self.selected_bit[i] {
                out.push(i);
            }
        }
        out
    }

    #[inline(always)]
    pub fn slack(&self) -> u32 {
        self.ch.max_weight - self.total_weight
    }

    #[inline(always)]
    pub fn add_item(&mut self, i: usize) {
        self.total_value += self.contrib[i] as i64;
        self.total_weight += self.ch.weights[i];

        let contrib_ptr = self.contrib.as_mut_ptr();
        let sup_ptr = self.support.as_mut_ptr();

        if let Some(ref neigh) = self.neigh {
            let row = unsafe { neigh.get_unchecked(i) };
            for &(k, v) in row.iter() {
                unsafe {
                    let kk = k as usize;
                    let ck = contrib_ptr.add(kk);
                    *ck = (*ck).wrapping_add(v as i32);
                    let sk = sup_ptr.add(kk);
                    *sk = (*sk).saturating_add(1);
                }
            }
        } else {
            let n = self.ch.num_items;
            let row_ptr = unsafe { self.ch.interaction_values.get_unchecked(i).as_ptr() };
            unsafe {
                for k in 0..n {
                    let v = *row_ptr.add(k);
                    let ck = contrib_ptr.add(k);
                    *ck = (*ck).wrapping_add(v);
                    if v != 0 {
                        let sk = sup_ptr.add(k);
                        *sk = (*sk).saturating_add(1);
                    }
                }
            }
        }

        self.selected_bit[i] = true;
    }

    #[inline(always)]
    pub fn remove_item(&mut self, j: usize) {
        self.total_value -= self.contrib[j] as i64;
        self.total_weight -= self.ch.weights[j];

        let contrib_ptr = self.contrib.as_mut_ptr();
        let sup_ptr = self.support.as_mut_ptr();

        if let Some(ref neigh) = self.neigh {
            let row = unsafe { neigh.get_unchecked(j) };
            for &(k, v) in row.iter() {
                unsafe {
                    let kk = k as usize;
                    let ck = contrib_ptr.add(kk);
                    *ck = (*ck).wrapping_sub(v as i32);
                    let sk = sup_ptr.add(kk);
                    *sk = (*sk).saturating_sub(1);
                }
            }
        } else {
            let n = self.ch.num_items;
            let row_ptr = unsafe { self.ch.interaction_values.get_unchecked(j).as_ptr() };
            unsafe {
                for k in 0..n {
                    let v = *row_ptr.add(k);
                    let ck = contrib_ptr.add(k);
                    *ck = (*ck).wrapping_sub(v);
                    if v != 0 {
                        let sk = sup_ptr.add(k);
                        *sk = (*sk).saturating_sub(1);
                    }
                }
            }
        }

        self.selected_bit[j] = false;
    }

    #[inline(always)]
    pub fn replace_item(&mut self, rm: usize, cand: usize) {
        self.remove_item(rm);
        self.add_item(cand);
    }

    pub fn restore_snapshot(&mut self, snap_value: i64, snap_weight: u32) {
        self.selected_bit.clone_from(&self.snap_bits);
        self.contrib.clone_from(&self.snap_contrib);
        self.support.clone_from(&self.snap_support);
        self.total_value = snap_value;
        self.total_weight = snap_weight;
    }
}

#[inline]
pub fn core_half_for(ch: &Challenge) -> usize {
    let n = ch.num_items;
    let team_est = (ch.max_weight as usize) / 6;

    let by_budget = if team_est <= 140 {
        70
    } else if team_est >= 1100 {
        130
    } else if team_est >= 900 {
        120
    } else if team_est >= 450 {
        90
    } else if team_est >= 200 {
        75
    } else {
        0
    };

    if by_budget != 0 {
        by_budget.min(150)
    } else if n <= 600 {
        60
    } else if n <= 1200 {
        40
    } else if n <= 2500 {
        CORE_HALF
    } else {
        40
    }
}

pub fn compute_total_interactions(ch: &Challenge) -> Vec<i64> {
    let n = ch.num_items;
    let mut sums = vec![0i64; n];
    let sums_ptr = sums.as_mut_ptr();

    for i in 0..n {
        let row_ptr = unsafe { ch.interaction_values.get_unchecked(i).as_ptr() };
        let mut si: i64 = 0;
        unsafe {
            for j in 0..i {
                let v = *row_ptr.add(j) as i64;
                si += v;
                *sums_ptr.add(j) += v;
            }
        }
        sums[i] += si;
    }
    sums
}

pub fn build_sparse_neighbors_and_totals(ch: &Challenge) -> (Vec<Vec<(u16, i16)>>, Vec<i64>) {
    let n = ch.num_items;
    let mut neigh: Vec<Vec<(u16, i16)>> = (0..n).map(|_| Vec::with_capacity(12)).collect();
    let mut totals: Vec<i64> = vec![0i64; n];

    for i in 0..n {
        let row_ptr = unsafe { ch.interaction_values.get_unchecked(i).as_ptr() };
        for j in 0..i {
            let val = unsafe { *row_ptr.add(j) };
            if val != 0 {
                let v16 = val as i16;
                neigh[i].push((j as u16, v16));
                neigh[j].push((i as u16, v16));
                let vv = val as i64;
                totals[i] += vv;
                totals[j] += vv;
            }
        }
    }

    for row in neigh.iter_mut() {
        row.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    }

    (neigh, totals)
}

pub fn set_windows_from_density(
    state: &mut State,
    by_density: &[usize],
    idx_first_rejected: usize,
    idx_last_inserted: usize,
) {
    let n = state.ch.num_items;
    let core_half = core_half_for(state.ch);

    let mut left = idx_first_rejected.saturating_sub(core_half + 1);
    let right = (idx_last_inserted + core_half + 1).min(n);
    if left > right {
        left = right;
    }

    state.window_locked = by_density[..left].to_vec();
    state.window_core = by_density[left..right].to_vec();
    state.window_rejected = by_density[right..].to_vec();

    let mut bins: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for &i in &state.window_core {
        bins.entry(state.ch.weights[i]).or_default().push(i);
    }
    state.core_bins = bins.into_iter().collect();
}

pub fn rebuild_windows(state: &mut State) {
    let n = state.ch.num_items;
    if n == 0 {
        return;
    }
    let cap = state.ch.max_weight;

    let mut by_density: Vec<usize> = (0..n).collect();
    let contrib = &state.contrib;
    by_density.sort_unstable_by(|&a, &b| {
        let na = contrib[a] as i64;
        let nb = contrib[b] as i64;
        let wa = state.ch.weights[a] as i64;
        let wb = state.ch.weights[b] as i64;
        (na * wb).cmp(&(nb * wa)).reverse()
    });

    let mut rem = cap;
    let mut idx_last_inserted = 0usize;
    let mut idx_first_rejected = n;
    for (idx, &i) in by_density.iter().enumerate() {
        let w = state.ch.weights[i];
        if w <= rem {
            rem -= w;
            idx_last_inserted = idx;
        } else if idx_first_rejected == n {
            idx_first_rejected = idx;
        }
    }
    set_windows_from_density(state, &by_density, idx_first_rejected, idx_last_inserted);
}
