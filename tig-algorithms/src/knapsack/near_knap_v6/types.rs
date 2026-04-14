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
    let weights = &state.ch.weights;
    let contrib = &state.contrib;
    let support = &state.support;
    let totals = state.total_interactions;
    let mut core_half = core_half_for(state.ch);

    if idx_first_rejected < n {
        let team_est = ((state.ch.max_weight as i64) / 6).clamp(18, 192);
        let inter_div = if state.neigh.is_some() {
            team_est + 12
        } else {
            team_est * 2 + 24
        };
        let lo = idx_first_rejected.saturating_sub(6);
        let hi = (idx_first_rejected + 6).min(n - 1);
        let frontier_i = by_density[idx_first_rejected];
        let frontier_prior =
            totals[frontier_i].max(0) / (inter_div * (1 + support[frontier_i].min(3) as i64));
        let frontier_score =
            (contrib[frontier_i] as i64) * 1024 / (weights[frontier_i].max(1) as i64)
                + frontier_prior;
        let gap_limit = frontier_score.abs() / 24 + 8;
        let frontier_w = weights[frontier_i];
        let mut near = 0usize;
        let mut same_weight = 0usize;

        for idx in lo..=hi {
            let i = by_density[idx];
            let prior = totals[i].max(0) / (inter_div * (1 + support[i].min(3) as i64));
            let score = (contrib[i] as i64) * 1024 / (weights[i].max(1) as i64) + prior;
            let diff = if score >= frontier_score {
                score - frontier_score
            } else {
                frontier_score - score
            };
            if diff <= gap_limit {
                near += 1;
            }
            if weights[i] == frontier_w {
                same_weight += 1;
            }
        }

        if near >= 6 {
            core_half = (core_half + core_half / 2 + 8).min(180);
        } else if near <= 2 {
            core_half = core_half.saturating_sub(core_half / 4);
        }
        if same_weight >= 4 && core_half < 96 {
            core_half = (core_half + 12).min(96);
        }
    }

    let mut left = idx_first_rejected.saturating_sub(core_half + 1);
    let right = (idx_last_inserted + core_half + 1).min(n);
    if left > right {
        left = right;
    }

    state.window_locked = by_density[..left].to_vec();
    state.window_core = by_density[left..right].to_vec();
    state.window_rejected = by_density[right..].to_vec();

    let use_interaction_prior = state.neigh.is_some() || n <= 1200;
    let team_est = ((state.ch.max_weight as i64) / 6).clamp(18, 192);
    let inter_div = if state.neigh.is_some() {
        team_est + 12
    } else {
        team_est * 2 + 24
    };

    let mut bins: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for &i in &state.window_core {
        bins.entry(weights[i]).or_default().push(i);
    }
    for items in bins.values_mut() {
        items.sort_unstable_by(|&a, &b| {
            let pa = if use_interaction_prior {
                totals[a].max(0) / (inter_div * (1 + support[a].min(3) as i64))
            } else {
                0
            };
            let pb = if use_interaction_prior {
                totals[b].max(0) / (inter_div * (1 + support[b].min(3) as i64))
            } else {
                0
            };
            let sa = (contrib[a] as i64) * 16 + pa;
            let sb = (contrib[b] as i64) * 16 + pb;
            sb.cmp(&sa)
                .then_with(|| totals[b].cmp(&totals[a]))
                .then_with(|| support[a].cmp(&support[b]))
                .then_with(|| a.cmp(&b))
        });
    }
    state.core_bins = bins.into_iter().collect();
}

pub fn rebuild_windows(state: &mut State) {
    let n = state.ch.num_items;
    if n == 0 {
        return;
    }
    let cap = state.ch.max_weight;
    let contrib = &state.contrib;
    let support = &state.support;
    let weights = &state.ch.weights;
    let totals = state.total_interactions;

    let use_interaction_prior = state.neigh.is_some() || n <= 1200;
    let team_est = ((cap as i64) / 6).clamp(18, 192);
    let inter_div = if state.neigh.is_some() {
        team_est + 12
    } else {
        team_est * 2 + 24
    };

    let mut shaped: Vec<i64> = vec![0; n];
    for i in 0..n {
        let prior = if use_interaction_prior {
            totals[i].max(0) / (inter_div * (1 + support[i].min(3) as i64))
        } else {
            0
        };
        shaped[i] = (contrib[i] as i64) * 64 + prior;
    }

    let mut by_density: Vec<usize> = (0..n).collect();
    by_density.sort_unstable_by(|&a, &b| {
        let lhs = (shaped[a] as i128) * (weights[b] as i128);
        let rhs = (shaped[b] as i128) * (weights[a] as i128);
        lhs.cmp(&rhs)
            .reverse()
            .then_with(|| {
                let lhs = (contrib[a] as i128) * (weights[b] as i128);
                let rhs = (contrib[b] as i128) * (weights[a] as i128);
                lhs.cmp(&rhs).reverse()
            })
            .then_with(|| totals[b].cmp(&totals[a]))
            .then_with(|| support[a].cmp(&support[b]))
            .then_with(|| weights[a].cmp(&weights[b]))
            .then_with(|| a.cmp(&b))
    });

    let mut rem = cap;
    let mut idx_last_inserted = 0usize;
    let mut idx_first_rejected = n;
    for (idx, &i) in by_density.iter().enumerate() {
        let w = weights[i];
        if w <= rem {
            rem -= w;
            idx_last_inserted = idx;
        } else if idx_first_rejected == n {
            idx_first_rejected = idx;
        }
    }
    set_windows_from_density(state, &by_density, idx_first_rejected, idx_last_inserted);
}