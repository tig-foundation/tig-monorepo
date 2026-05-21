use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;

use ils::run_one_instance;
use params::Params;

mod types {
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
}

mod params {
use serde_json::{Map, Value};

#[derive(Clone, Copy)]
pub struct Params {
    pub n_perturbation_rounds: usize,
    pub perturbation_strength_base: usize,
    pub extra_starts: usize,
    pub max_frontier_swaps_override: Option<usize>,
    pub dp_passes_multiplier: usize,
}

impl Params {
    pub fn initialize(h: &Option<Map<String, Value>>) -> Self {
        let mut p = Self {
            n_perturbation_rounds: 49,
            perturbation_strength_base: 3,
            extra_starts: 0,
            max_frontier_swaps_override: None,
            dp_passes_multiplier: 1,
        };
        if let Some(m) = h {
            if let Some(v) = m.get("n_perturbation_rounds").and_then(|v| v.as_u64()) {
                p.n_perturbation_rounds = v as usize;
            }
            if let Some(v) = m.get("perturbation_strength_base").and_then(|v| v.as_u64()) {
                p.perturbation_strength_base = v as usize;
            }
            if let Some(v) = m.get("extra_starts").and_then(|v| v.as_u64()) {
                p.extra_starts = v as usize;
            }
            if let Some(v) = m.get("max_frontier_swaps_override").and_then(|v| v.as_u64()) {
                p.max_frontier_swaps_override = Some(v as usize);
            }
            if let Some(v) = m.get("dp_passes_multiplier").and_then(|v| v.as_u64()) {
                p.dp_passes_multiplier = v as usize;
            }
        }
        p
    }
}
}

mod refinement {
use super::types::{State, MICRO_ADD_K, MICRO_K, MICRO_RM_K};
use tig_challenges::knapsack::Challenge;

fn integer_core_target(
    ch: &Challenge,
    locked: &[usize],
    core: &[usize],
    core_val: &[i32],
    dp_cache: &mut Vec<i64>,
    choose_cache: &mut Vec<u8>,
) -> Vec<usize> {
    let used_locked: u64 = locked.iter().map(|&i| ch.weights[i] as u64).sum();
    let rem_cap = (ch.max_weight as u64).saturating_sub(used_locked) as usize;

    let myk = core.len();
    if myk == 0 {
        let mut selected: Vec<usize> = locked.to_vec();
        selected.sort_unstable();
        return selected;
    }

    let mut total_core_weight: usize = 0;
    let mut total_pos_weight: usize = 0;
    let mut all_pos_fit = true;
    for (t, &it) in core.iter().enumerate() {
        let wt = ch.weights[it] as usize;
        total_core_weight += wt;
        if core_val[t] > 0 {
            total_pos_weight += wt;
            if total_pos_weight > rem_cap {
                all_pos_fit = false;
            }
        }
    }

    if rem_cap == 0 {
        let mut selected: Vec<usize> = locked.to_vec();
        for (t, &it) in core.iter().enumerate() {
            if ch.weights[it] == 0 && core_val[t] > 0 {
                selected.push(it);
            }
        }
        selected.sort_unstable();
        return selected;
    }

    if all_pos_fit {
        let mut selected: Vec<usize> = locked.to_vec();
        for (t, &it) in core.iter().enumerate() {
            if core_val[t] > 0 {
                selected.push(it);
            }
        }
        selected.sort_unstable();
        return selected;
    }

    let myw = rem_cap.min(total_core_weight);

    let dp_size = myw + 1;
    let choose_size = myk * dp_size;

    if dp_cache.len() < dp_size {
        dp_cache.resize(dp_size, i64::MIN / 4);
    }
    if choose_cache.len() < choose_size {
        choose_cache.resize(choose_size, 0);
    }

    for val in &mut dp_cache[0..dp_size] {
        *val = -1;
    }
    dp_cache[0] = 0;

    choose_cache[0..choose_size].fill(0);

    let mut w_hi: usize = 0;

    for (t, &it) in core.iter().enumerate() {
        let wt = ch.weights[it] as usize;
        if wt > myw {
            continue;
        }
        let val = core_val[t] as i64;
        let new_hi = (w_hi + wt).min(myw);
        for w in (wt..=new_hi).rev() {
            let prev = dp_cache[w - wt];
            if prev < 0 {
                continue;
            }
            let cand = prev + val;
            if cand > dp_cache[w] {
                dp_cache[w] = cand;
                choose_cache[t * dp_size + w] = 1;
            }
        }
        w_hi = new_hi;
    }

    let mut selected: Vec<usize> = locked.to_vec();
    let mut w_star = (0..=myw).max_by_key(|&w| dp_cache[w]).unwrap_or(0);
    for t in (0..myk).rev() {
        let it = core[t];
        let wt = ch.weights[it] as usize;
        if wt <= w_star && choose_cache[t * dp_size + w_star] == 1 {
            selected.push(it);
            w_star -= wt;
        }
    }
    selected.sort_unstable();
    selected
}

fn apply_dp_target_via_ops(state: &mut State, target_sel: &[usize]) {
    let n = state.ch.num_items;
    let mut to_remove: Vec<usize> = Vec::new();
    let mut to_add: Vec<usize> = Vec::new();
    let mut j = 0;
    let m = target_sel.len();

    for i in 0..n {
        let in_target = j < m && target_sel[j] == i;
        if in_target {
            j += 1;
        }
        if state.selected_bit[i] && !in_target {
            to_remove.push(i);
        } else if in_target && !state.selected_bit[i] {
            to_add.push(i);
        }
    }

    for &r in &to_remove {
        state.remove_item(r);
    }
    for &a in &to_add {
        state.add_item(a);
    }
}

pub fn dp_refinement(state: &mut State) {
    dp_refinement_x(state, 1);
}

pub fn dp_refinement_x(state: &mut State, passes_multiplier: usize) {
    let team_est = (state.ch.max_weight as usize) / 6;
    let k = state.window_core.len();
    let passes = (if team_est <= 140 {
        if k <= 80 { 5 } else { 4 }
    } else if team_est <= 180 {
        if k <= 70 { 4 } else { 3 }
    } else if k <= 160 {
        2
    } else {
        1
    }) * passes_multiplier;
    let n = state.ch.num_items;
    if n == 0 {
        return;
    }

    for _ in 0..passes {
        let mut core_val: Vec<i32> = Vec::with_capacity(k);
        if k != 0 {
            let mut sel_core_bit = vec![false; n];
            let mut sel_core: Vec<usize> = Vec::new();
            for &i in &state.window_core {
                if state.selected_bit[i] {
                    sel_core_bit[i] = true;
                    sel_core.push(i);
                }
            }

            if let Some(ref ng) = state.neigh {
                for &it in &state.window_core {
                    let mut sub: i32 = 0;
                    let row = unsafe { ng.get_unchecked(it) };
                    for &(kk, v) in row.iter() {
                        let j = kk as usize;
                        if sel_core_bit[j] {
                            sub += v as i32;
                        }
                    }

                    let mut v0 = state.contrib[it]
                        - if state.selected_bit[it] {
                            (sub * 2) / 3
                        } else {
                            sub / 3
                        };
                    if !state.selected_bit[it] {
                        v0 += (state.total_interactions[it] / 256) as i32
                            + (state.usage[it] as i32) * 14;
                    }
                    core_val.push(v0);
                }
            } else {
                for &it in &state.window_core {
                    let mut sub: i32 = 0;
                    for &j in &sel_core {
                        sub += state.ch.interaction_values[it][j];
                    }

                    let mut v0 = state.contrib[it]
                        - if state.selected_bit[it] {
                            (sub * 2) / 3
                        } else {
                            sub / 3
                        };
                    if !state.selected_bit[it] {
                        v0 += (state.total_interactions[it] / 256) as i32
                            + (state.usage[it] as i32) * 14;
                    }
                    core_val.push(v0);
                }
            }
        }

        let target = integer_core_target(
            state.ch,
            &state.window_locked,
            &state.window_core,
            &core_val,
            &mut state.dp_cache,
            &mut state.choose_cache,
        );
        apply_dp_target_via_ops(state, &target);
    }
}

pub fn micro_qkp_refinement(state: &mut State) {
    let n = state.ch.num_items;
    if n == 0 || state.window_core.is_empty() {
        return;
    }

    let team_est = (state.ch.max_weight as usize) / 6;
    let big_team = n >= 4500 && team_est >= 850;

    let micro_k: usize = if big_team {
        12
    } else if team_est <= 160 {
        18
    } else {
        MICRO_K
    };
    let rm_k: usize = if big_team {
        6
    } else if team_est <= 160 {
        9
    } else {
        MICRO_RM_K
    };
    let add_k: usize = if big_team {
        6
    } else if team_est <= 160 {
        9
    } else {
        MICRO_ADD_K
    };

    let mut sel: Vec<usize> = Vec::new();
    let mut unsel: Vec<usize> = Vec::new();

    for &i in &state.window_core {
        if state.selected_bit[i] {
            sel.push(i);
        } else {
            unsel.push(i);
        }
    }

    if let Some(ref ng) = state.neigh {
        let mut guides: Vec<usize> = Vec::new();
        for &i in &state.window_core {
            if state.selected_bit[i] {
                guides.push(i);
            }
        }
        guides.sort_unstable_by(|&a, &b| {
            state.support[b]
                .cmp(&state.support[a])
                .then_with(|| state.contrib[b].cmp(&state.contrib[a]))
                .then_with(|| b.cmp(&a))
        });

        let push_unsel = |v: &mut Vec<usize>, x: usize| {
            for &y in v.iter() {
                if y == x {
                    return;
                }
            }
            v.push(x);
        };

        let g = guides.len().min(if big_team { 4 } else { 6 });
        for t in 0..g {
            let vtx = guides[t];
            let row = unsafe { ng.get_unchecked(vtx) };
            let pref = row.len().min(if big_team { 16 } else { 24 });
            for u in 0..pref {
                let cand = row[u].0 as usize;
                if !state.selected_bit[cand] {
                    push_unsel(&mut unsel, cand);
                }
            }
        }

        let hub_take: usize = if big_team {
            5
        } else if team_est <= 160 {
            10
        } else {
            8
        };
        let mut added_hubs: usize = 0;
        let lim = state.hubs_static.len().min(192);
        for &h in state.hubs_static.iter().take(lim) {
            if added_hubs >= hub_take {
                break;
            }
            if state.selected_bit[h] {
                continue;
            }
            push_unsel(&mut unsel, h);
            added_hubs += 1;

            let row = unsafe { ng.get_unchecked(h) };
            let pref = row.len().min(16);
            for u in 0..pref {
                let cand = row[u].0 as usize;
                if !state.selected_bit[cand] {
                    push_unsel(&mut unsel, cand);
                }
            }
        }
    }

    let extra_r = state.window_rejected.len().min(if big_team { 12 } else { 24 });
    for &i in &state.window_rejected[..extra_r] {
        if !state.selected_bit[i] {
            unsel.push(i);
        }
    }

    let extra_l = state.window_locked.len().min(24);
    let start_l = state.window_locked.len().saturating_sub(extra_l);
    for &i in &state.window_locked[start_l..] {
        if state.selected_bit[i] {
            sel.push(i);
        }
    }

    let score = |st: &State, i: usize| -> i64 {
        let w = (st.ch.weights[i] as i64).max(1);
        let dens = (st.contrib[i] as i64 * 1000) / w;
        dens + (st.support[i] as i64) * 120 + (st.total_interactions[i] / 320)
    };

    sel.sort_unstable_by(|&a, &b| score(state, a).cmp(&score(state, b)).then_with(|| a.cmp(&b)));
    unsel.sort_unstable_by(|&a, &b| {
        score(state, b)
            .cmp(&score(state, a))
            .then_with(|| b.cmp(&a))
    });

    let mut cand: Vec<usize> = Vec::with_capacity(micro_k);
    let push_u = |v: &mut Vec<usize>, x: usize| {
        for &y in v.iter() {
            if y == x {
                return;
            }
        }
        v.push(x);
    };

    for &i in sel.iter().take(rm_k) {
        push_u(&mut cand, i);
        if cand.len() >= micro_k {
            break;
        }
    }
    for &i in unsel.iter().take(add_k) {
        push_u(&mut cand, i);
        if cand.len() >= micro_k {
            break;
        }
    }
    if cand.len() < 2 {
        return;
    }

    let k = cand.len();
    if k > 20 {
        return;
    }

    let mut sel_cand: Vec<usize> = Vec::new();
    let mut sel_cand_w: u32 = 0;
    for &it in &cand {
        if state.selected_bit[it] {
            sel_cand.push(it);
            sel_cand_w = sel_cand_w.saturating_add(state.ch.weights[it]);
        }
    }
    if state.total_weight < sel_cand_w {
        return;
    }

    let fixed_w = state.total_weight - sel_cand_w;
    if fixed_w > state.ch.max_weight {
        return;
    }
    let rem_cap: u32 = state.ch.max_weight - fixed_w;

    let mut w: Vec<u32> = vec![0; k];
    let mut base: Vec<i64> = vec![0; k];
    for t in 0..k {
        let it = cand[t];
        w[t] = state.ch.weights[it];
        let mut b = state.contrib[it] as i64;
        for &j in &sel_cand {
            b -= state.ch.interaction_values[it][j] as i64;
        }
        base[t] = b;
    }

    let mut inter: Vec<i64> = vec![0; k * k];
    for a in 0..k {
        let ia = cand[a];
        for b in 0..a {
            let ib = cand[b];
            let v = state.ch.interaction_values[ia][ib] as i64;
            inter[a * k + b] = v;
            inter[b * k + a] = v;
        }
    }

    let mut pos_sum: Vec<i64> = vec![0; k];
    for a in 0..k {
        let mut s: i64 = 0;
        for b in 0..k {
            if a == b {
                continue;
            }
            let v = inter[a * k + b];
            if v > 0 {
                s += v;
            }
        }
        pos_sum[a] = s;
    }

    let mut ub_item: Vec<i64> = vec![0; k];
    for t in 0..k {
        ub_item[t] = base[t] + pos_sum[t];
    }

    let mut order: Vec<usize> = (0..k).collect();
    order.sort_unstable_by(|&a, &b| ub_item[b].cmp(&ub_item[a]).then_with(|| w[a].cmp(&w[b])));

    let mut cand2: Vec<usize> = vec![0; k];
    let mut w2: Vec<u32> = vec![0; k];
    let mut base2: Vec<i64> = vec![0; k];
    let mut ub_item2: Vec<i64> = vec![0; k];
    for (new_idx, &old_idx) in order.iter().enumerate() {
        cand2[new_idx] = cand[old_idx];
        w2[new_idx] = w[old_idx];
        base2[new_idx] = base[old_idx];
        ub_item2[new_idx] = ub_item[old_idx];
    }

    let mut inter2: Vec<i64> = vec![0; k * k];
    for a in 0..k {
        let oa = order[a];
        for b in 0..k {
            let ob = order[b];
            inter2[a * k + b] = inter[oa * k + ob];
        }
    }

    let mut cur_mask: usize = 0;
    for t in 0..k {
        if state.selected_bit[cand2[t]] {
            cur_mask |= 1usize << t;
        }
    }

    let mut cur_w_sum: u32 = 0;
    let mut cur_v: i64 = 0;
    for i in 0..k {
        if ((cur_mask >> i) & 1) == 0 {
            continue;
        }
        cur_w_sum += w2[i];
        cur_v += base2[i];
        for j in 0..i {
            if ((cur_mask >> j) & 1) != 0 {
                cur_v += inter2[i * k + j];
            }
        }
    }
    if cur_w_sum > rem_cap {
        return;
    }

    let mut suffix_ub: Vec<i64> = vec![0; k + 1];
    for i in (0..k).rev() {
        let x = ub_item2[i];
        suffix_ub[i] = suffix_ub[i + 1] + if x > 0 { x } else { 0 };
    }

    let mut best_mask: usize = cur_mask;
    let mut best_val: i64 = cur_v;

    let mut stack: Vec<(usize, u32, i64, usize)> = Vec::new();
    stack.push((0, 0, 0, 0));

    while let Some((idx, cw, cv, mask)) = stack.pop() {
        let ub = cv + suffix_ub[idx];
        if ub <= best_val {
            continue;
        }
        if idx == k {
            if cv > best_val {
                best_val = cv;
                best_mask = mask;
            }
            continue;
        }

        stack.push((idx + 1, cw, cv, mask));

        let wt = w2[idx];
        let new_w = cw.saturating_add(wt);
        if new_w <= rem_cap {
            let mut delta = base2[idx];
            let mut pm = mask;
            while pm != 0 {
                let l = pm & pm.wrapping_neg();
                let j = l.trailing_zeros() as usize;
                delta += inter2[idx * k + j];
                pm ^= l;
            }
            stack.push((idx + 1, new_w, cv + delta, mask | (1usize << idx)));
        }
    }

    if best_mask == cur_mask {
        return;
    }

    let mut to_remove: Vec<usize> = Vec::new();
    let mut to_add: Vec<usize> = Vec::new();
    for t in 0..k {
        let it = cand2[t];
        let want = ((best_mask >> t) & 1) != 0;
        let have = state.selected_bit[it];
        if have && !want {
            to_remove.push(it);
        } else if !have && want {
            to_add.push(it);
        }
    }

    for &it in &to_remove {
        state.remove_item(it);
    }
    for &it in &to_add {
        if state.total_weight + state.ch.weights[it] <= state.ch.max_weight {
            state.add_item(it);
        }
    }
}
}

mod local_search {
use super::params::Params;
use super::refinement::micro_qkp_refinement;
use super::types::{rebuild_windows, State, DIFF_LIM};

fn apply_best_add_windowed(state: &mut State) -> bool {
    const SUP_ADD_BONUS: i64 = 40;

    let slack = state.slack();
    if slack == 0 {
        return false;
    }

    let mut best: Option<(usize, i64)> = None;

    for (bw, items) in &state.core_bins {
        if *bw > slack {
            break;
        }
        for &cand in items {
            if state.selected_bit[cand] {
                continue;
            }
            let delta = state.contrib[cand];
            if delta <= 0 {
                continue;
            }
            let s = (delta as i64) + (state.support[cand] as i64) * SUP_ADD_BONUS;
            if best.map_or(true, |(_, bs)| s > bs) {
                best = Some((cand, s));
            }
        }
    }

    if best.is_none() {
        let lim = if state.ch.num_items >= 2500 {
            state.window_rejected.len().min(384)
        } else {
            state.window_rejected.len()
        };

        for &cand in &state.window_rejected[..lim] {
            if state.selected_bit[cand] {
                continue;
            }
            let w = state.ch.weights[cand];
            if w > slack {
                continue;
            }
            let delta = state.contrib[cand];
            if delta <= 0 {
                continue;
            }
            let s = (delta as i64) + (state.support[cand] as i64) * SUP_ADD_BONUS;
            if best.map_or(true, |(_, bs)| s > bs) {
                best = Some((cand, s));
            }
        }
    }

    if let Some((cand, _)) = best {
        state.add_item(cand);
        true
    } else {
        false
    }
}

fn apply_best_add_neigh_global(state: &mut State) -> bool {
    const BETA_NUM: i64 = 3;
    const BETA_DEN: i64 = 20;
    const SUP_ADD_BONUS: i64 = 40;

    let slack = state.slack();
    if slack == 0 {
        return false;
    }

    let neigh = match state.neigh {
        Some(ng) => ng,
        None => return false,
    };

    let n = state.ch.num_items;
    if n == 0 {
        return false;
    }

    let edge_lim: usize = if n >= 4500 { 16000 } else if n >= 2500 { 12000 } else { 9000 };
    let node_lim: usize = if n >= 4500 { 56 } else if n >= 2500 { 64 } else { 72 };

    let start = (((state.total_value as u64) as usize)
        ^ ((state.total_weight as usize).wrapping_mul(911)))
        % n;
    let mut step = (n / 97).max(1);
    step |= 1;

    let mut best: Option<(usize, i64, i32)> = None;
    let mut scanned_edges: usize = 0;
    let mut scanned_nodes: usize = 0;

    let mut idx = start;
    let mut tries: usize = 0;

    while tries < n && scanned_nodes < node_lim && scanned_edges < edge_lim {
        if state.selected_bit[idx] {
            scanned_nodes += 1;
            let row = unsafe { neigh.get_unchecked(idx) };
            for &(cj, _vv) in row.iter() {
                scanned_edges += 1;
                if scanned_edges > edge_lim {
                    break;
                }

                let cand = cj as usize;
                if state.selected_bit[cand] {
                    continue;
                }

                let w_u = state.ch.weights[cand];
                if w_u == 0 || w_u > slack {
                    continue;
                }

                let delta = state.contrib[cand];
                if delta <= 0 {
                    continue;
                }

                let w = (w_u as i64).max(1);
                let c = delta as i64;
                let tot = state.total_interactions[cand];
                let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot);
                let s = (adj * 1000) / w + (state.support[cand] as i64) * SUP_ADD_BONUS;

                if best.map_or(true, |(_, bs, bd)| s > bs || (s == bs && delta > bd)) {
                    best = Some((cand, s, delta));
                }
            }
        }

        idx += step;
        if idx >= n {
            idx -= n;
        }
        tries += 1;
    }

    if let Some((cand, _, _)) = best {
        state.add_item(cand);
        true
    } else {
        false
    }
}

fn apply_best_replace12_windowed(state: &mut State, used: &[usize]) -> bool {
    let cap = state.ch.max_weight;
    if used.is_empty() {
        return false;
    }
    let slack0 = state.slack();

    let mut add_pool: Vec<usize> = Vec::with_capacity(state.window_core.len() + 64);
    add_pool.extend_from_slice(&state.window_core);
    let extra = state.window_rejected.len().min(64);
    add_pool.extend_from_slice(&state.window_rejected[..extra]);
    add_pool.retain(|&i| !state.selected_bit[i]);

    add_pool.sort_unstable_by(|&a, &b| {
        let wa = (state.ch.weights[a] as i64).max(1);
        let wb = (state.ch.weights[b] as i64).max(1);
        let sa = (state.contrib[a] as i64 * 1000) / wa + (state.support[a] as i64) * 80;
        let sb = (state.contrib[b] as i64 * 1000) / wb + (state.support[b] as i64) * 80;

        let pa = 2 * (state.contrib[a] as i64) - state.total_interactions[a];
        let pb = 2 * (state.contrib[b] as i64) - state.total_interactions[b];

        sb.cmp(&sa)
            .then_with(|| pb.cmp(&pa))
            .then_with(|| (state.support[b] as i64).cmp(&(state.support[a] as i64)))
            .then_with(|| b.cmp(&a))
    });
    if add_pool.len() > 28 {
        add_pool.truncate(28);
    }
    if add_pool.len() < 2 {
        return false;
    }

    let mut rm: Vec<usize> = used.to_vec();
    rm.sort_unstable_by(|&a, &b| {
        let wa = (state.ch.weights[a] as i64).max(1);
        let wb = (state.ch.weights[b] as i64).max(1);
        let sa = (state.contrib[a] as i64 * 1000) / wa + (state.support[a] as i64) * 120;
        let sb = (state.contrib[b] as i64 * 1000) / wb + (state.support[b] as i64) * 120;
        sa.cmp(&sb).then_with(|| a.cmp(&b))
    });
    if rm.len() > 16 {
        rm.truncate(16);
    }

    let mut best: Option<(usize, usize, usize, i64, i64, i64)> = None;

    for &r in &rm {
        if !state.selected_bit[r] {
            continue;
        }
        let wr = state.ch.weights[r];
        let avail = wr.saturating_add(slack0);

        for x in 0..add_pool.len() {
            let a = add_pool[x];
            let wa = state.ch.weights[a];
            if wa == 0 || wa > avail {
                continue;
            }
            for y in (x + 1)..add_pool.len() {
                let b = add_pool[y];
                let wb = state.ch.weights[b];
                if wb == 0 || wa + wb > avail {
                    continue;
                }

                let new_w = state
                    .total_weight
                    .saturating_sub(wr)
                    .saturating_add(wa)
                    .saturating_add(wb);
                if new_w > cap {
                    continue;
                }

                let delta = (state.contrib[a] as i64)
                    + (state.contrib[b] as i64)
                    - (state.contrib[r] as i64)
                    - (state.ch.interaction_values[a][r] as i64)
                    - (state.ch.interaction_values[b][r] as i64)
                    + (state.ch.interaction_values[a][b] as i64);

                if delta <= 0 {
                    continue;
                }

                let proxy = (2 * (state.contrib[a] as i64) - state.total_interactions[a])
                    + (2 * (state.contrib[b] as i64) - state.total_interactions[b]);
                let sup_sum = (state.support[a] as i64) + (state.support[b] as i64);

                if best.map_or(true, |(_, _, _, bd, bp, bs)| {
                    delta > bd || (delta == bd && (proxy > bp || (proxy == bp && sup_sum > bs)))
                }) {
                    best = Some((r, a, b, delta, proxy, sup_sum));
                }
            }
        }
    }

    if let Some((r, a, b, _, _, _)) = best {
        state.remove_item(r);
        state.add_item(a);
        state.add_item(b);
        true
    } else {
        false
    }
}

fn apply_best_replace21_windowed(state: &mut State, used: &[usize]) -> bool {
    let cap = state.ch.max_weight;
    if used.len() < 2 {
        return false;
    }

    let mut rm: Vec<usize> = used.to_vec();
    rm.sort_unstable_by(|&a, &b| {
        let ca = state.contrib[a] as i64;
        let cb = state.contrib[b] as i64;
        let wa = state.ch.weights[a] as i64;
        let wb = state.ch.weights[b] as i64;
        (ca * wb).cmp(&(cb * wa))
    });
    if rm.len() > 14 {
        rm.truncate(14);
    }

    let mut add_pool: Vec<usize> = Vec::with_capacity(state.window_core.len() + 48);
    add_pool.extend_from_slice(&state.window_core);
    let extra = state.window_rejected.len().min(48);
    add_pool.extend_from_slice(&state.window_rejected[..extra]);
    add_pool.retain(|&i| !state.selected_bit[i] && state.contrib[i] > 0);
    add_pool.sort_unstable_by(|&a, &b| {
        let ca = state.contrib[a] as i64;
        let cb = state.contrib[b] as i64;
        let wa = state.ch.weights[a] as i64;
        let wb = state.ch.weights[b] as i64;

        let ratio_cmp = (cb * wa).cmp(&(ca * wb));
        if ratio_cmp != core::cmp::Ordering::Equal {
            return ratio_cmp;
        }

        let mut min_a: i64 = i64::MAX;
        let mut min_b: i64 = i64::MAX;
        for &r in &rm {
            let va = state.ch.interaction_values[a][r] as i64;
            if va < min_a {
                min_a = va;
            }
            let vb = state.ch.interaction_values[b][r] as i64;
            if vb < min_b {
                min_b = vb;
            }
        }

        min_a.cmp(&min_b).then_with(|| b.cmp(&a))
    });
    if add_pool.len() > 28 {
        add_pool.truncate(28);
    }
    if add_pool.is_empty() {
        return false;
    }

    let mut best: Option<(usize, usize, usize, i64)> = None;

    for &cand in &add_pool {
        let wc = state.ch.weights[cand];
        if wc == 0 {
            continue;
        }

        for x in 0..rm.len() {
            let a = rm[x];
            let wa = state.ch.weights[a];
            for y in (x + 1)..rm.len() {
                let b = rm[y];
                let wb = state.ch.weights[b];

                let new_w = state
                    .total_weight
                    .saturating_sub(wa)
                    .saturating_sub(wb)
                    .saturating_add(wc);
                if new_w > cap {
                    continue;
                }

                let delta = (state.contrib[cand] as i64)
                    - (state.ch.interaction_values[cand][a] as i64)
                    - (state.ch.interaction_values[cand][b] as i64)
                    - (state.contrib[a] as i64)
                    - (state.contrib[b] as i64)
                    + (state.ch.interaction_values[a][b] as i64);

                if delta > 0 && best.map_or(true, |(_, _, _, bd)| delta > bd) {
                    best = Some((cand, a, b, delta));
                }
            }
        }
    }

    if let Some((cand, a, b, _)) = best {
        state.remove_item(a);
        state.remove_item(b);
        if !state.selected_bit[cand] && state.total_weight + state.ch.weights[cand] <= cap {
            state.add_item(cand);
        }
        true
    } else {
        false
    }
}

#[inline]
fn apply_best_swap_diff_reduce_windowed_cached(state: &mut State, used: &[usize]) -> bool {
    let mut best_imp: Option<(usize, usize, i32, u32, i64)> = None;
    let mut best_neu: Option<(usize, usize, u32, i64)> = None;

    for &rm in used {
        let w_rm = state.ch.weights[rm];
        if w_rm == 0 {
            continue;
        }
        let w_min = w_rm.saturating_sub(DIFF_LIM as u32);
        let sup_rm = state.support[rm] as i64;

        for (bw, items) in &state.core_bins {
            if *bw >= w_rm {
                break;
            }
            if *bw < w_min {
                continue;
            }
            let dw = w_rm - *bw;
            if dw == 0 {
                continue;
            }

            for &cand in items {
                if state.selected_bit[cand] {
                    continue;
                }

                let delta = state.contrib[cand]
                    - state.contrib[rm]
                    - state.ch.interaction_values[cand][rm];

                let sup_diff = (state.support[cand] as i64) - sup_rm;

                if delta > 0 {
                    let take = match best_imp {
                        None => true,
                        Some((_bc, _br, bd, bdw, bsd)) => {
                            delta > bd
                                || (delta == bd
                                    && (dw > bdw || (dw == bdw && sup_diff > bsd)))
                        }
                    };
                    if take {
                        best_imp = Some((cand, rm, delta, dw, sup_diff));
                    }
                } else if delta == 0 && sup_diff > 0 {
                    let take = match best_neu {
                        None => true,
                        Some((_bc, _br, bdw, bsd)) => dw > bdw || (dw == bdw && sup_diff > bsd),
                    };
                    if take {
                        best_neu = Some((cand, rm, dw, sup_diff));
                    }
                }
            }
        }
    }

    if let Some((cand, rm, _, _, _)) = best_imp {
        state.replace_item(rm, cand);
        true
    } else if let Some((cand, rm, _, _)) = best_neu {
        state.replace_item(rm, cand);
        true
    } else {
        false
    }
}

#[inline]
fn apply_best_swap_diff_increase_windowed_cached(state: &mut State, used: &[usize]) -> bool {
    let slack = state.slack();
    if slack == 0 {
        return false;
    }
    let mut best: Option<(usize, usize, f64)> = None;
    for &rm in used {
        let w_rm = state.ch.weights[rm];
        let max_dw = (DIFF_LIM as u32).min(slack);
        let w_max = w_rm.saturating_add(max_dw);
        for (bw, items) in &state.core_bins {
            if *bw <= w_rm {
                continue;
            }
            if *bw > w_max {
                break;
            }
            let dw = *bw - w_rm;
            if dw > slack {
                break;
            }
            for &cand in items {
                if state.selected_bit[cand] {
                    continue;
                }
                let delta = state.contrib[cand]
                    - state.contrib[rm]
                    - state.ch.interaction_values[cand][rm];
                if delta > 0 {
                    let ratio = (delta as f64) / (dw as f64);
                    if best.map_or(true, |(_, _, br)| ratio > br) {
                        best = Some((cand, rm, ratio));
                    }
                }
            }
        }
    }
    if let Some((cand, rm, _)) = best {
        state.replace_item(rm, cand);
        true
    } else {
        false
    }
}

fn apply_best_swap_neigh_any(state: &mut State, used: &[usize]) -> bool {
    if used.is_empty() {
        return false;
    }
    let cap = state.ch.max_weight;

    let neigh = match state.neigh {
        Some(ng) => ng,
        None => return false,
    };

    let mut best: Option<(usize, usize, i128, i64, i64, i64, i64)> = None;

    for &rm in used {
        if !state.selected_bit[rm] {
            continue;
        }
        let wrm = state.ch.weights[rm];
        let row = unsafe { neigh.get_unchecked(rm) };

        for &(cj, vv) in row.iter() {
            let cand = cj as usize;
            if state.selected_bit[cand] {
                continue;
            }
            let wc = state.ch.weights[cand];
            if wc == 0 {
                continue;
            }

            if (state.total_weight as u64) + (wc as u64) > (cap as u64) + (wrm as u64) {
                continue;
            }

            let vv_i64 = vv as i64;
            let delta_i64 = (state.contrib[cand] as i64) - (state.contrib[rm] as i64) - vv_i64;
            if delta_i64 <= 0 {
                continue;
            }

            let base_score: i128 = if wc == wrm {
                (delta_i64 as i128) * (1_000_000i128)
            } else if wc < wrm {
                (delta_i64 as i128) * (1000i128) + (wrm as i128 - wc as i128)
            } else {
                let dw = (wc - wrm) as i128;
                ((delta_i64 as i128) * (1000i128)) / dw.max(1)
            };

            let score: i128 = if vv_i64 > 0 {
                let v = vv_i64 as i128;
                let d = delta_i64 as i128;
                base_score - (v * v) / d.max(1)
            } else if vv_i64 < 0 {
                base_score + (-(vv_i64 as i128))
            } else {
                base_score
            };

            let bonus = if vv_i64 < 0 { -vv_i64 } else { 0 };
            let pen = if vv_i64 > 0 { vv_i64 } else { 0 };
            let sup = state.support[cand] as i64;

            if best.map_or(true, |(_, _, bs, bd, bb, bp, bsu)| {
                score > bs
                    || (score == bs
                        && (delta_i64 > bd
                            || (delta_i64 == bd
                                && (bonus > bb
                                    || (bonus == bb
                                        && (pen < bp || (pen == bp && sup > bsu)))))))
            }) {
                best = Some((cand, rm, score, delta_i64, bonus, pen, sup));
            }
        }
    }

    if let Some((cand, rm, _, _, _, _, _)) = best {
        state.replace_item(rm, cand);
        true
    } else {
        false
    }
}

fn apply_best_swap_frontier_global(state: &mut State, used: &[usize]) -> bool {
    let cap = state.ch.max_weight;
    if used.is_empty() {
        return false;
    }

    let neigh = match state.neigh {
        Some(ng) => ng,
        None => return false,
    };

    let n = state.ch.num_items;
    if n == 0 {
        return false;
    }

    const BETA_NUM: i64 = 3;
    const BETA_DEN: i64 = 20;
    const TOP_CAND: usize = 48;
    const TOP_RM: usize = 32;

    let mut rm: Vec<usize> = used.to_vec();
    rm.sort_unstable_by(|&a, &b| {
        let wa = (state.ch.weights[a] as i64).max(1);
        let wb = (state.ch.weights[b] as i64).max(1);
        let sa = (state.contrib[a] as i64 * 1000) / wa + (state.support[a] as i64) * 140;
        let sb = (state.contrib[b] as i64 * 1000) / wb + (state.support[b] as i64) * 140;
        sa.cmp(&sb).then_with(|| a.cmp(&b))
    });
    if rm.len() > TOP_RM {
        rm.truncate(TOP_RM);
    }

    let mut max_rm_w: u32 = 0;
    for &i in used {
        let w = state.ch.weights[i];
        if w > max_rm_w {
            max_rm_w = w;
        }
    }
    if max_rm_w == 0 {
        return false;
    }

    let slack0 = state.slack();
    let mut seen: Vec<u32> = vec![0u32; n];
    let stamp: u32 = 1;

    let mut cand_list: Vec<(i64, usize)> = Vec::with_capacity(TOP_CAND);

    #[inline]
    fn push_top(list: &mut Vec<(i64, usize)>, s: i64, idx: usize) {
        if list.len() < TOP_CAND {
            list.push((s, idx));
            return;
        }
        let mut worst_pos = 0usize;
        let mut worst_s = list[0].0;
        for t in 1..list.len() {
            if list[t].0 < worst_s {
                worst_s = list[t].0;
                worst_pos = t;
            }
        }
        if s > worst_s {
            list[worst_pos] = (s, idx);
        }
    }

    for &u in &state.window_core {
        if state.selected_bit[u] {
            let row = unsafe { neigh.get_unchecked(u) };
            for &(cj, _vv) in row.iter() {
                let cand = cj as usize;
                if cand >= n {
                    continue;
                }
                if state.selected_bit[cand] {
                    continue;
                }
                if seen[cand] == stamp {
                    continue;
                }
                seen[cand] = stamp;

                let wc = state.ch.weights[cand];
                if wc == 0 {
                    continue;
                }
                if wc > slack0.saturating_add(max_rm_w) {
                    continue;
                }
                let c = state.contrib[cand] as i64;
                if c <= 0 {
                    continue;
                }
                let tot = state.total_interactions[cand];
                let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot);
                let s = (adj * 1000) / (wc as i64).max(1) + (state.support[cand] as i64) * 60;
                push_top(&mut cand_list, s, cand);
            }
        } else {
            let cand = u;
            if cand >= n {
                continue;
            }
            if state.selected_bit[cand] {
                continue;
            }
            if seen[cand] == stamp {
                continue;
            }
            seen[cand] = stamp;

            let wc = state.ch.weights[cand];
            if wc == 0 {
                continue;
            }
            if wc > slack0.saturating_add(max_rm_w) {
                continue;
            }
            let c = state.contrib[cand] as i64;
            if c <= 0 {
                continue;
            }
            let tot = state.total_interactions[cand];
            let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot);
            let s = (adj * 1000) / (wc as i64).max(1) + (state.support[cand] as i64) * 60;
            push_top(&mut cand_list, s, cand);
        }
    }

    if cand_list.is_empty() {
        return false;
    }

    let mut best: Option<(usize, usize, i64, i64)> = None;

    for &(_s0, cand) in &cand_list {
        let wc = state.ch.weights[cand];
        if wc == 0 {
            continue;
        }

        for &r in &rm {
            if !state.selected_bit[r] {
                continue;
            }
            let wr = state.ch.weights[r];

            if (state.total_weight as u64) + (wc as u64) > (cap as u64) + (wr as u64) {
                continue;
            }

            let inter = state.ch.interaction_values[cand][r] as i64;
            let delta = (state.contrib[cand] as i64) - (state.contrib[r] as i64) - inter;
            if delta <= 0 {
                continue;
            }

            let score: i64 = if wc == wr {
                delta * 1_000_000
            } else if wc < wr {
                delta * 1000 + (wr as i64 - wc as i64)
            } else {
                let dw = (wc - wr) as i64;
                (delta * 1000) / dw.max(1)
            };

            if best.map_or(true, |(_, _, bs, bd)| score > bs || (score == bs && delta > bd)) {
                best = Some((cand, r, score, delta));
            }
        }
    }

    if let Some((cand, r, _, _)) = best {
        state.replace_item(r, cand);
        true
    } else {
        false
    }
}

fn apply_kick_refill_windowed(state: &mut State, used: &[usize]) -> bool {
    if used.is_empty() {
        return false;
    }

    let old_val = state.total_value;

    let mut kick: Option<usize> = None;
    let mut best_key: Option<(i64, i64, i64)> = None;

    for &i in used {
        if !state.selected_bit[i] {
            continue;
        }
        let c = state.contrib[i] as i64;
        let sup = state.support[i] as i64;
        let w = state.ch.weights[i] as i64;
        let key = (c, sup, -w);
        if best_key.map_or(true, |bk| key < bk) {
            best_key = Some(key);
            kick = Some(i);
        }
    }

    let r = match kick {
        Some(x) => x,
        None => return false,
    };

    state.remove_item(r);

    let mut added: Vec<usize> = Vec::new();

    loop {
        let slack = state.slack();
        if slack == 0 {
            break;
        }

        let mut best: Option<(usize, i64)> = None;

        for (bw, items) in &state.core_bins {
            if *bw > slack {
                break;
            }
            for &cand in items {
                if state.selected_bit[cand] {
                    continue;
                }
                let delta = state.contrib[cand];
                if delta <= 0 {
                    continue;
                }
                let s = (delta as i64) + (state.support[cand] as i64) * 40;
                if best.map_or(true, |(_, bs)| s > bs) {
                    best = Some((cand, s));
                }
            }
        }

        if best.is_none() {
            let lim = if state.ch.num_items >= 2500 {
                state.window_rejected.len().min(384)
            } else {
                state.window_rejected.len()
            };
            for &cand in &state.window_rejected[..lim] {
                if state.selected_bit[cand] {
                    continue;
                }
                let w = state.ch.weights[cand];
                if w > slack || w == 0 {
                    continue;
                }
                let delta = state.contrib[cand];
                if delta <= 0 {
                    continue;
                }
                let s = (delta as i64) + (state.support[cand] as i64) * 40;
                if best.map_or(true, |(_, bs)| s > bs) {
                    best = Some((cand, s));
                }
            }
        }

        let cand = match best {
            Some((c, _)) => c,
            None => break,
        };

        state.add_item(cand);
        added.push(cand);
    }

    if state.total_value > old_val {
        true
    } else {
        for &cand in added.iter().rev() {
            if state.selected_bit[cand] {
                state.remove_item(cand);
            }
        }
        if !state.selected_bit[r] {
            state.add_item(r);
        }
        false
    }
}

pub fn local_search_vnd(state: &mut State, params: &Params) {
    let mut iterations = 0;
    let n = state.ch.num_items;
    let max_iterations = if n >= 4500 {
        180
    } else if n >= 3000 {
        260
    } else if n >= 1000 {
        240
    } else {
        80
    };
    let mut used: Vec<usize> = Vec::new();
    let mut micro_used = false;

    let mut frontier_swap_tries: usize = 0;
    let max_frontier_swaps: usize = params.max_frontier_swaps_override.unwrap_or(if n >= 2500 {
        0
    } else if n >= 1500 {
        1
    } else {
        2
    });

    let mut dirty_window = false;
    let mut n_rebuilds = 0usize;
    let max_rebuilds: usize = if n >= 2500 { 1 } else { 2 };

    loop {
        iterations += 1;
        if iterations > max_iterations {
            break;
        }

        if apply_best_add_windowed(state) {
            dirty_window = true;
            continue;
        }

        if (iterations & 3) == 0 && apply_best_add_neigh_global(state) {
            dirty_window = true;
            continue;
        }

        used.clear();
        for &i in &state.window_core {
            if state.selected_bit[i] {
                used.push(i);
            }
        }

        let extra = state.window_locked.len().min(24);
        let start = state.window_locked.len().saturating_sub(extra);
        for &i in state.window_locked[start..].iter() {
            if state.selected_bit[i] {
                used.push(i);
            }
        }

        {
            const WORST_CAP: usize = 32;
            let mut worst: Vec<(i32, u32, usize)> = Vec::with_capacity(WORST_CAP);

            for i in 0..n {
                if !state.selected_bit[i] {
                    continue;
                }
                let c = state.contrib[i];
                let w = state.ch.weights[i];

                if worst.len() < WORST_CAP {
                    worst.push((c, w, i));
                    continue;
                }

                let mut best_pos = 0usize;
                let mut best_c = worst[0].0;
                let mut best_w = worst[0].1;

                for t in 1..worst.len() {
                    let (cc, ww, _) = worst[t];
                    if cc > best_c || (cc == best_c && ww < best_w) {
                        best_pos = t;
                        best_c = cc;
                        best_w = ww;
                    }
                }

                if c < best_c || (c == best_c && w > best_w) {
                    worst[best_pos] = (c, w, i);
                }
            }

            for &(_, _, idx) in &worst {
                used.push(idx);
            }

            used.sort_unstable();
            used.dedup();
        }

        if apply_best_swap_diff_reduce_windowed_cached(state, &used) {
            dirty_window = true;
            continue;
        }
        if apply_best_swap_diff_increase_windowed_cached(state, &used) {
            dirty_window = true;
            continue;
        }

        if apply_best_swap_neigh_any(state, &used) {
            dirty_window = true;
            continue;
        }

        if frontier_swap_tries < max_frontier_swaps {
            frontier_swap_tries += 1;
            if apply_best_swap_frontier_global(state, &used) {
                dirty_window = true;
                continue;
            }
        }

        if apply_best_replace12_windowed(state, &used) {
            dirty_window = true;
            if n_rebuilds < max_rebuilds {
                n_rebuilds += 1;
                dirty_window = false;
                rebuild_windows(state);
            }
            continue;
        }
        if apply_best_replace21_windowed(state, &used) {
            dirty_window = true;
            if n_rebuilds < max_rebuilds {
                n_rebuilds += 1;
                dirty_window = false;
                rebuild_windows(state);
            }
            continue;
        }

        if apply_kick_refill_windowed(state, &used) {
            dirty_window = true;
            if n_rebuilds < max_rebuilds {
                n_rebuilds += 1;
                dirty_window = false;
                rebuild_windows(state);
            }
            continue;
        }

        if dirty_window && n_rebuilds < max_rebuilds {
            n_rebuilds += 1;
            dirty_window = false;
            rebuild_windows(state);
            continue;
        }

        if !micro_used {
            micro_used = true;
            if dirty_window {
                rebuild_windows(state);
                dirty_window = false;
            }
            let old = state.total_value;
            micro_qkp_refinement(state);
            if state.total_value > old {
                rebuild_windows(state);
                continue;
            }
        }

        break;
    }
}
}

mod construct {
use super::types::{set_windows_from_density, State, N_IT_CONSTRUCT};
use super::types::Rng;

pub fn greedy_fill_with_beta(state: &mut State, rng: &mut Rng, noise_mask: u32, allow_seed: bool) {
    const BETA_NUM: i64 = 3;
    const BETA_DEN: i64 = 20;
    const HUB_K: usize = 12;
    const SUP_BONUS: i64 = 70;

    let n = state.ch.num_items;

    if n >= 2500 {
        if let Some(neigh) = state.neigh {
            const HUB_GLOBAL_K: usize = 64;
            const HUB_PAIR_K: usize = 12;

            let mut hubs_g: Vec<(i64, usize, u32)> = Vec::with_capacity(HUB_GLOBAL_K);
            let lim = state.hubs_static.len().min(256);
            for &i in state.hubs_static.iter().take(lim) {
                if state.selected_bit[i] {
                    continue;
                }
                let w_u = state.ch.weights[i];
                if w_u == 0 {
                    continue;
                }
                let w = w_u as i64;
                let tot_i = state.total_interactions[i];
                let mut ss = (tot_i * 1000) / w.max(1);
                if noise_mask != 0 {
                    ss += (rng.next_u32() & (noise_mask >> 1)) as i64;
                }
                hubs_g.push((ss, i, w_u));
            }
            hubs_g.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            if hubs_g.len() > HUB_GLOBAL_K {
                hubs_g.truncate(HUB_GLOBAL_K);
            }

            let mut in_frontier = vec![false; n];
            let mut frontier: Vec<usize> = Vec::with_capacity(n.min(4096));

            for i in 0..n {
                if !state.selected_bit[i] {
                    continue;
                }
                let row = unsafe { neigh.get_unchecked(i) };
                for &(k, _v) in row.iter() {
                    let u = k as usize;
                    if !state.selected_bit[u] && !in_frontier[u] {
                        in_frontier[u] = true;
                        frontier.push(u);
                    }
                }
            }

            let push_frontier_of =
                |st: &State, in_f: &mut Vec<bool>, fr: &mut Vec<usize>, v: usize| {
                    let row = unsafe { neigh.get_unchecked(v) };
                    for &(k, _vv) in row.iter() {
                        let u = k as usize;
                        if !st.selected_bit[u] && !in_f[u] {
                            in_f[u] = true;
                            fr.push(u);
                        }
                    }
                };

            loop {
                let slack = state.slack();
                if slack == 0 {
                    break;
                }

                let mut best_pos: Option<(usize, i64)> = None;

                let f_len = frontier.len();
                let scan_lim = if f_len > 2048 { 2048 } else { f_len };

                if f_len != 0 && f_len <= 2048 {
                    for &i in &frontier {
                        if state.selected_bit[i] {
                            continue;
                        }
                        let w_u = state.ch.weights[i];
                        if w_u == 0 || w_u > slack {
                            continue;
                        }
                        let c = state.contrib[i] as i64;
                        if c <= 0 {
                            continue;
                        }

                        let tot_i = state.total_interactions[i];
                        let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot_i);

                        let mut s = (adj * 1000) / (w_u as i64).max(1)
                            + (state.support[i] as i64) * SUP_BONUS;

                        if noise_mask != 0 {
                            s += (rng.next_u32() & noise_mask) as i64;
                        }
                        if best_pos.map_or(true, |(_, bs)| s > bs) {
                            best_pos = Some((i, s));
                        }
                    }
                } else if f_len > 2048 {
                    for _ in 0..scan_lim {
                        let idx = (rng.next_u32() as usize) % f_len;
                        let i = frontier[idx];

                        if state.selected_bit[i] {
                            continue;
                        }
                        let w_u = state.ch.weights[i];
                        if w_u == 0 || w_u > slack {
                            continue;
                        }
                        let c = state.contrib[i] as i64;
                        if c <= 0 {
                            continue;
                        }

                        let tot_i = state.total_interactions[i];
                        let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot_i);

                        let mut s = (adj * 1000) / (w_u as i64).max(1)
                            + (state.support[i] as i64) * SUP_BONUS;

                        if noise_mask != 0 {
                            s += (rng.next_u32() & noise_mask) as i64;
                        }
                        if best_pos.map_or(true, |(_, bs)| s > bs) {
                            best_pos = Some((i, s));
                        }
                    }
                }

                if let Some((i, _)) = best_pos {
                    state.add_item(i);
                    push_frontier_of(state, &mut in_frontier, &mut frontier, i);
                    continue;
                }

                let mut best_pair: Option<(usize, usize, i64)> = None;
                if slack >= 2 && !hubs_g.is_empty() {
                    let lim = hubs_g.len().min(HUB_PAIR_K);
                    for t in 0..lim {
                        let a = hubs_g[t].1;
                        let wa = hubs_g[t].2;
                        if state.selected_bit[a] || wa == 0 || wa >= slack {
                            continue;
                        }

                        let row = unsafe { neigh.get_unchecked(a) };
                        let pref = row.len().min(64);
                        for u in 0..pref {
                            let (bb, vv) = row[u];
                            let b = bb as usize;
                            if a == b || state.selected_bit[b] {
                                continue;
                            }
                            let wb = state.ch.weights[b];
                            if wb == 0 || wa + wb > slack {
                                continue;
                            }
                            let v = vv as i64;
                            if v <= 0 {
                                continue;
                            }

                            let delta = (state.contrib[a] as i64) + (state.contrib[b] as i64) + v;
                            if delta <= 0 {
                                continue;
                            }

                            let s = (delta * 1_000_000) / ((wa + wb) as i64).max(1);
                            if best_pair.map_or(true, |(_, _, bs)| s > bs) {
                                best_pair = Some((a, b, s));
                            }
                        }
                    }
                }

                if let Some((a, b, _)) = best_pair {
                    state.add_item(a);
                    push_frontier_of(state, &mut in_frontier, &mut frontier, a);
                    if state.slack() >= state.ch.weights[b] && !state.selected_bit[b] {
                        state.add_item(b);
                        push_frontier_of(state, &mut in_frontier, &mut frontier, b);
                    }
                    continue;
                }

                if allow_seed {
                    let mut best_seed: Option<(usize, i64)> = None;
                    for &(ss, i, _w) in &hubs_g {
                        if state.selected_bit[i] {
                            continue;
                        }
                        let wi = state.ch.weights[i];
                        if wi == 0 || wi > slack {
                            continue;
                        }
                        if best_seed.map_or(true, |(_, bs)| ss > bs) {
                            best_seed = Some((i, ss));
                        }
                    }

                    if let Some((i, _)) = best_seed {
                        state.add_item(i);
                        push_frontier_of(state, &mut in_frontier, &mut frontier, i);
                        continue;
                    }
                }

                break;
            }

            return;
        }
    }

    let mut hubs: Vec<(i64, usize, u32)> = Vec::with_capacity(HUB_K);

    loop {
        let slack = state.slack();
        if slack == 0 {
            break;
        }

        let mut best_pos: Option<(usize, i64)> = None;
        let mut best_seed: Option<(usize, i64)> = None;

        hubs.clear();
        let mut hubs_min_score: i64 = i64::MAX;
        let mut hubs_min_pos: usize = 0;

        for i in 0..n {
            if state.selected_bit[i] {
                continue;
            }
            let w_u = state.ch.weights[i];
            if w_u == 0 || w_u > slack {
                continue;
            }
            let w = w_u as i64;

            let c = state.contrib[i] as i64;
            let tot_i = state.total_interactions[i];

            if c > 0 {
                let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot_i);
                let mut s = (adj * 1000) / w + (state.support[i] as i64) * SUP_BONUS;
                if noise_mask != 0 {
                    s += (rng.next_u32() & noise_mask) as i64;
                }
                if best_pos.map_or(true, |(_, bs)| s > bs) {
                    best_pos = Some((i, s));
                }
            }

            let mut ss = (tot_i * 1000) / w;
            if noise_mask != 0 {
                ss += (rng.next_u32() & (noise_mask >> 1)) as i64;
            }
            if best_seed.map_or(true, |(_, bs)| ss > bs) {
                best_seed = Some((i, ss));
            }

            if slack >= 2 {
                if hubs.len() < HUB_K {
                    hubs.push((ss, i, w_u));
                    if ss < hubs_min_score {
                        hubs_min_score = ss;
                        hubs_min_pos = hubs.len() - 1;
                    }
                    if hubs.len() == 1 {
                        hubs_min_score = ss;
                        hubs_min_pos = 0;
                    }
                } else if ss > hubs_min_score {
                    hubs[hubs_min_pos] = (ss, i, w_u);
                    hubs_min_score = hubs[0].0;
                    hubs_min_pos = 0;
                    for t in 1..hubs.len() {
                        if hubs[t].0 < hubs_min_score {
                            hubs_min_score = hubs[t].0;
                            hubs_min_pos = t;
                        }
                    }
                }
            }
        }

        if let Some((i, _)) = best_pos {
            state.add_item(i);
            continue;
        }

        let mut best_pair: Option<(usize, usize, i64)> = None;
        if slack >= 2 && !hubs.is_empty() {
            for &(_hs, a, wa) in &hubs {
                if state.selected_bit[a] || wa == 0 || wa > slack {
                    continue;
                }
                if let Some(ref ng) = state.neigh {
                    let row = unsafe { ng.get_unchecked(a) };
                    for &(bb, vv) in row.iter() {
                        let b = bb as usize;
                        if a == b || state.selected_bit[b] {
                            continue;
                        }
                        let wb = state.ch.weights[b];
                        if wb == 0 || wa + wb > slack {
                            continue;
                        }
                        let v = vv as i64;
                        if v <= 0 {
                            continue;
                        }
                        let tot_a = state.total_interactions[a];
                        let tot_b = state.total_interactions[b];
                        let s = (v * 1_000_000) / ((wa + wb) as i64) + (tot_a + tot_b) / 2000;
                        if best_pair.map_or(true, |(_, _, bs)| s > bs) {
                            best_pair = Some((a, b, s));
                        }
                    }
                } else {
                    let row = &state.ch.interaction_values[a];
                    for b in 0..n {
                        if a == b || state.selected_bit[b] {
                            continue;
                        }
                        let wb = state.ch.weights[b];
                        if wb == 0 || wa + wb > slack {
                            continue;
                        }
                        let v = row[b] as i64;
                        if v <= 0 {
                            continue;
                        }
                        let tot_a = state.total_interactions[a];
                        let tot_b = state.total_interactions[b];
                        let s = (v * 1_000_000) / ((wa + wb) as i64) + (tot_a + tot_b) / 2000;
                        if best_pair.map_or(true, |(_, _, bs)| s > bs) {
                            best_pair = Some((a, b, s));
                        }
                    }
                }
            }
        }

        if let Some((a, b, _)) = best_pair {
            state.add_item(a);
            if state.slack() >= state.ch.weights[b] && !state.selected_bit[b] {
                state.add_item(b);
            }
            continue;
        }

        if allow_seed {
            if let Some((i, _)) = best_seed {
                state.add_item(i);
            } else {
                break;
            }
        } else {
            break;
        }
    }
}

pub fn construct_pair_seed_beta(state: &mut State, rng: &mut Rng) {
    let n = state.ch.num_items;
    if n == 0 {
        return;
    }

    if state.total_weight == 0 {
        let cap = state.ch.max_weight;
        let mut best_pair: Option<(usize, usize, i64)> = None;

        if let Some(ref ng) = state.neigh {
            for i in 0..n {
                let wi = state.ch.weights[i];
                if wi == 0 || wi > cap {
                    continue;
                }
                let row = unsafe { ng.get_unchecked(i) };
                for &(jj, vv) in row.iter() {
                    let j = jj as usize;
                    if j >= i {
                        continue;
                    }
                    let w = wi + state.ch.weights[j];
                    if w == 0 || w > cap {
                        continue;
                    }
                    let v = vv as i64;
                    if v <= 0 {
                        continue;
                    }
                    let ti = state.total_interactions[i];
                    let tj = state.total_interactions[j];

                    let noise: i64 =
                        (((i as i64) * 1315423911i64) ^ ((j as i64) * 2654435761i64)) & 0x3Fi64;

                    let s = (v * 1_000_000) / (w as i64) + (ti + tj) / 2000 + noise;
                    if best_pair.map_or(true, |(_, _, bs)| s > bs) {
                        best_pair = Some((i, j, s));
                    }
                }
            }
        } else {
            let samples = ((n as u32) * 16).min(16000) as usize;
            for _ in 0..samples {
                let i = (rng.next_u32() as usize) % n;
                let j = (rng.next_u32() as usize) % n;
                if i == j {
                    continue;
                }
                let w = state.ch.weights[i] + state.ch.weights[j];
                if w == 0 || w > cap {
                    continue;
                }
                let v = state.ch.interaction_values[i][j] as i64;
                if v <= 0 {
                    continue;
                }
                let ti = state.total_interactions[i];
                let tj = state.total_interactions[j];
                let s = (v * 1_000_000) / (w as i64) + (ti + tj) / 2000;
                if best_pair.map_or(true, |(_, _, bs)| s > bs) {
                    best_pair = Some((i, j, s));
                }
            }
        }

        if let Some((i, j, _)) = best_pair {
            state.add_item(i);
            if state.total_weight + state.ch.weights[j] <= cap && !state.selected_bit[j] {
                state.add_item(j);
            }
        }
    }

    greedy_fill_with_beta(state, rng, 0, true);
}

pub fn construct_frontier_cluster_grow(state: &mut State, rng: &mut Rng) {
    const BETA_NUM: i64 = 3;
    const BETA_DEN: i64 = 20;

    let n = state.ch.num_items;
    if n == 0 {
        return;
    }
    let cap = state.ch.max_weight;

    let mut in_frontier = vec![false; n];
    let mut frontier: Vec<usize> = Vec::with_capacity(n.min(4096));

    let push_frontier_of = |st: &State, in_f: &mut Vec<bool>, fr: &mut Vec<usize>, i: usize| {
        if let Some(ref ng) = st.neigh {
            let row = unsafe { ng.get_unchecked(i) };
            for &(k, _v) in row.iter() {
                let u = k as usize;
                if !st.selected_bit[u] && !in_f[u] {
                    in_f[u] = true;
                    fr.push(u);
                }
            }
        } else {
            let row = &st.ch.interaction_values[i];
            for u in 0..n {
                if row[u] != 0 && !st.selected_bit[u] && !in_f[u] {
                    in_f[u] = true;
                    fr.push(u);
                }
            }
        }
    };

    let add_seed = |st: &mut State,
                        in_f: &mut Vec<bool>,
                        fr: &mut Vec<usize>,
                        seed: usize| {
        if st.selected_bit[seed] {
            return;
        }
        if st.total_weight + st.ch.weights[seed] > cap {
            return;
        }
        st.add_item(seed);
        push_frontier_of(st, in_f, fr, seed);
    };

    let mut best_seed: Option<usize> = None;
    let mut best_s: i64 = i64::MIN;
    let samples = n.min(512).max(64);

    if n <= 1500 {
        for i in 0..n {
            let w = state.ch.weights[i] as i64;
            if w <= 0 || w as u32 > cap {
                continue;
            }
            let s = (state.total_interactions[i] * 1000) / w;
            if s > best_s {
                best_s = s;
                best_seed = Some(i);
            }
        }
    } else {
        for _ in 0..samples {
            let i = (rng.next_u32() as usize) % n;
            let w = state.ch.weights[i] as i64;
            if w <= 0 || w as u32 > cap {
                continue;
            }
            let s = (state.total_interactions[i] * 1000) / w;
            if s > best_s {
                best_s = s;
                best_seed = Some(i);
            }
        }
    }

    if let Some(i) = best_seed {
        add_seed(state, &mut in_frontier, &mut frontier, i);

        let mut best_j: Option<(usize, i64)> = None;
        if let Some(ref ng) = state.neigh {
            let row = unsafe { ng.get_unchecked(i) };
            for &(jj, vv) in row.iter() {
                let j = jj as usize;
                if j == i || state.selected_bit[j] {
                    continue;
                }
                let wsum = state.ch.weights[i] + state.ch.weights[j];
                if wsum == 0 || wsum > cap {
                    continue;
                }
                let v = vv as i64;
                if v <= 0 {
                    continue;
                }
                let s = (v * 1_000_000) / (wsum as i64) + state.total_interactions[j] / 2000;
                if best_j.map_or(true, |(_, bs)| s > bs) {
                    best_j = Some((j, s));
                }
            }
        } else {
            let row = &state.ch.interaction_values[i];
            for j in 0..n {
                if j == i || state.selected_bit[j] {
                    continue;
                }
                let wsum = state.ch.weights[i] + state.ch.weights[j];
                if wsum == 0 || wsum > cap {
                    continue;
                }
                let v = row[j] as i64;
                if v <= 0 {
                    continue;
                }
                let s = (v * 1_000_000) / (wsum as i64) + state.total_interactions[j] / 2000;
                if best_j.map_or(true, |(_, bs)| s > bs) {
                    best_j = Some((j, s));
                }
            }
        }
        if let Some((j, _)) = best_j {
            add_seed(state, &mut in_frontier, &mut frontier, j);
        }
    }

    let team_est = (cap as usize) / 6;
    let max_jumps: usize = if team_est >= 900 {
        10
    } else if team_est >= 450 {
        8
    } else if team_est >= 200 {
        6
    } else {
        3
    };
    let mut jumps_done: usize = 0;

    loop {
        let slack = state.slack();
        if slack == 0 {
            break;
        }

        let mut best_cand: Option<(usize, i64, i64)> = None;
        for &u in &frontier {
            if state.selected_bit[u] {
                continue;
            }
            let wu = state.ch.weights[u];
            if wu == 0 || wu > slack {
                continue;
            }

            let c = state.contrib[u] as i64;
            if c <= 0 {
                continue;
            }
            let tot_u = state.total_interactions[u];
            let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot_u);
            let s0 = (adj * 1000) / (wu as i64).max(1);
            let s = s0 + (rng.next_u32() & 0x0F) as i64;

            if best_cand.map_or(true, |(_, bs, _)| s > bs) {
                best_cand = Some((u, s, s0));
            }
        }

        let allow_early_jump = jumps_done < max_jumps && slack > cap / 5;
        let mut jump: Option<(usize, i64)> = None;
        let mut jump_s: i64 = i64::MIN;

        if allow_early_jump || best_cand.is_none() {
            for _ in 0..samples {
                let i = (rng.next_u32() as usize) % n;
                if state.selected_bit[i] {
                    continue;
                }
                let wi = state.ch.weights[i];
                if wi == 0 || wi >= slack {
                    continue;
                }

                if let Some(ref ng) = state.neigh {
                    let row = unsafe { ng.get_unchecked(i) };
                    let mut ok = false;
                    let lim = row.len().min(10);
                    for t in 0..lim {
                        let j = row[t].0 as usize;
                        if state.selected_bit[j] {
                            continue;
                        }
                        let wj = state.ch.weights[j];
                        if wj != 0 && wi + wj <= slack {
                            ok = true;
                            break;
                        }
                    }
                    if !ok {
                        continue;
                    }
                }

                let s = (state.total_interactions[i] * 1000) / (wi as i64).max(1);
                if s > jump_s {
                    jump_s = s;
                    jump = Some((i, s));
                }
            }
        }

        let do_jump = if let (Some((_u, _s, s0)), Some((_j, js))) = (best_cand, jump) {
            allow_early_jump && js > s0.saturating_mul(2)
        } else {
            best_cand.is_none() && jump.is_some()
        };

        if do_jump {
            if let Some((seed, _)) = jump {
                jumps_done += 1;
                add_seed(state, &mut in_frontier, &mut frontier, seed);

                let slack1 = state.slack();
                if state.selected_bit[seed] && slack1 >= 1 {
                    let mut best_nb: Option<(usize, i64)> = None;

                    if let Some(ref ng) = state.neigh {
                        let row = unsafe { ng.get_unchecked(seed) };
                        let pref = row.len().min(72);
                        for t in 0..pref {
                            let (jj, vv) = row[t];
                            let j = jj as usize;
                            if j == seed || state.selected_bit[j] {
                                continue;
                            }
                            let wj = state.ch.weights[j];
                            if wj == 0 || wj > slack1 {
                                continue;
                            }
                            let v = vv as i64;
                            if v <= 0 {
                                continue;
                            }
                            let wsum = (state.ch.weights[seed] + wj) as i64;
                            let s =
                                (v * 1_000_000) / wsum.max(1) + state.total_interactions[j] / 2000;
                            if best_nb.map_or(true, |(_, bs)| s > bs) {
                                best_nb = Some((j, s));
                            }
                        }
                    } else {
                        let row = &state.ch.interaction_values[seed];
                        for j in 0..n {
                            if j == seed || state.selected_bit[j] {
                                continue;
                            }
                            let wj = state.ch.weights[j];
                            if wj == 0 || wj > slack1 {
                                continue;
                            }
                            let v = row[j] as i64;
                            if v <= 0 {
                                continue;
                            }
                            let wsum = (state.ch.weights[seed] + wj) as i64;
                            let s =
                                (v * 1_000_000) / wsum.max(1) + state.total_interactions[j] / 2000;
                            if best_nb.map_or(true, |(_, bs)| s > bs) {
                                best_nb = Some((j, s));
                            }
                        }
                    }

                    if let Some((j, _)) = best_nb {
                        add_seed(state, &mut in_frontier, &mut frontier, j);
                    }
                }

                continue;
            }
        }

        if let Some((u, _s, _s0)) = best_cand {
            state.add_item(u);
            push_frontier_of(state, &mut in_frontier, &mut frontier, u);
            continue;
        }

        if let Some((seed, _)) = jump {
            add_seed(state, &mut in_frontier, &mut frontier, seed);
            continue;
        }

        break;
    }

    greedy_fill_with_beta(state, rng, 0, true);
}

pub fn construct_forward_incremental(state: &mut State, mode: usize, rng: &mut Rng) {
    let n = state.ch.num_items;

    if state.total_weight == 0 && n > 0 {
        let slack0 = state.slack();
        if slack0 > 0 {
            let tries = n.min(64);
            let samp = n.min(64);
            let mut best_seed: Option<usize> = None;
            let mut best_score: i64 = i64::MIN;

            for _ in 0..tries {
                let i = (rng.next_u32() as usize) % n;
                let wi = state.ch.weights[i];
                if wi == 0 || wi > slack0 {
                    continue;
                }

                let mut est: i64 = 0;
                for _ in 0..samp {
                    let j = (rng.next_u32() as usize) % n;
                    est += state.ch.interaction_values[i][j] as i64;
                }

                let score = (est * 1000) / (wi as i64);
                if score > best_score {
                    best_score = score;
                    best_seed = Some(i);
                }
            }

            if let Some(i) = best_seed {
                state.add_item(i);
            }
        }
    }

    loop {
        let slack = state.slack();
        if slack == 0 {
            break;
        }

        let mut best_i: Option<usize> = None;
        let mut best_s: i64 = i64::MIN;
        let mut second_i: Option<usize> = None;
        let mut second_s: i64 = i64::MIN;

        for i in 0..n {
            if state.selected_bit[i] {
                continue;
            }
            let w_u = state.ch.weights[i];
            if w_u > slack {
                continue;
            }
            let c = state.contrib[i] as i64;
            if c <= 0 {
                continue;
            }

            let w = (w_u as i64).max(1);
            let mut s = match mode {
                2 => c,
                3 => (c * 1000) / w + (w_u as i64) * 3,
                _ => (c * 1000) / w,
            };
            if mode >= 4 {
                let mask = if mode >= 5 { 0x7F } else { 0x1F };
                s += (rng.next_u32() & mask) as i64;
            }

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

        let pick = if mode >= 4 && second_i.is_some() {
            let m = if mode >= 5 { 1 } else { 3 };
            if (rng.next_u32() & m) == 0 {
                second_i
            } else {
                best_i
            }
        } else {
            best_i
        };

        if let Some(i) = pick {
            state.add_item(i);
        } else {
            break;
        }
    }

    if state.slack() >= 2 {
        let noise = if mode >= 4 { 0x1F } else { 0 };
        greedy_fill_with_beta(state, rng, noise, true);
    }
}

pub fn build_initial_solution(state: &mut State) {
    let n = state.ch.num_items;
    if n == 0 {
        return;
    }
    let cap = state.ch.max_weight;

    let mut sum_values: i64 = 0;
    let mut sum_w: u32 = 0;
    for i in 0..n {
        state.selected_bit[i] = true;

        state.support[i] = if let Some(ng) = state.neigh {
            unsafe { ng.get_unchecked(i).len().min(u16::MAX as usize) as u16 }
        } else {
            let mut c: u16 = 0;
            for &v in state.ch.interaction_values[i].iter() {
                if v != 0 && c < u16::MAX {
                    c += 1;
                }
            }
            c
        };

        let ti = state.total_interactions[i].min(i32::MAX as i64) as i32;
        state.contrib[i] = state.ch.values[i] as i32 + ti;
        sum_values += state.ch.values[i] as i64;
        sum_w += state.ch.weights[i];
    }
    state.total_weight = sum_w;
    let sum_inter: i64 = state.total_interactions.iter().sum();
    state.total_value = sum_values + sum_inter / 2;

    while state.total_weight > cap {
        let mut worst_item = 0;
        let mut worst_score = i64::MAX;

        for i in 0..n {
            if state.selected_bit[i] {
                let contrib = state.contrib[i] as i64;
                let weight = state.ch.weights[i] as i64;
                let score = if weight > 0 {
                    (contrib * 1000) / weight
                } else {
                    contrib * 1000
                };

                if score < worst_score {
                    worst_score = score;
                    worst_item = i;
                }
            }
        }

        state.remove_item(worst_item);
    }

    let mut idx_last_inserted = 0;
    let mut idx_first_rejected = n;
    let mut by_density: Vec<usize> = (0..n).collect();

    for _ in 0..=N_IT_CONSTRUCT {
        idx_last_inserted = 0;
        idx_first_rejected = n;
        let contrib = &state.contrib;
        by_density.sort_unstable_by(|&a, &b| {
            let na = contrib[a] as i64;
            let nb = contrib[b] as i64;
            let wa = state.ch.weights[a] as i64;
            let wb = state.ch.weights[b] as i64;
            (na * wb).cmp(&(nb * wa)).reverse()
        });

        let mut target_sel: Vec<usize> = Vec::with_capacity(n);
        let mut rem = cap;
        for (idx, &i) in by_density.iter().enumerate() {
            let w = state.ch.weights[i];
            if w <= rem {
                target_sel.push(i);
                rem -= w;
                idx_last_inserted = idx;
            } else if idx_first_rejected == n {
                idx_first_rejected = idx;
            }
        }

        let mut in_target = vec![false; n];
        for &i in &target_sel {
            in_target[i] = true;
        }
        let mut to_remove: Vec<usize> = Vec::new();
        let mut to_add: Vec<usize> = Vec::new();
        for i in 0..n {
            if state.selected_bit[i] && !in_target[i] {
                to_remove.push(i);
            }
            if !state.selected_bit[i] && in_target[i] {
                to_add.push(i);
            }
        }

        if to_remove.is_empty() && to_add.is_empty() {
            break;
        }

        for &r in &to_remove {
            state.remove_item(r);
        }
        for &a in &to_add {
            state.add_item(a);
        }
    }

    set_windows_from_density(state, &by_density, idx_first_rejected, idx_last_inserted);
}
}

mod ils {
use super::construct::{
    build_initial_solution, construct_forward_incremental, construct_frontier_cluster_grow,
    construct_pair_seed_beta, greedy_fill_with_beta,
};
use super::local_search::local_search_vnd;
use super::params::Params;
use super::refinement::{dp_refinement_x, micro_qkp_refinement};
use super::types::{
    build_sparse_neighbors_and_totals, compute_total_interactions, rebuild_windows, Rng, State,
};
use tig_challenges::knapsack::*;

thread_local! {
    static ACTIVE_FRONTIER: core::cell::RefCell<Vec<usize>> = core::cell::RefCell::new(Vec::new());
}

#[derive(Clone, Copy)]
struct TabuCtx {
    ptr: *mut u32,
    len: usize,
    iter: u32,
    tenure: u32,
    aspiration: i64,
    enabled: bool,
}

thread_local! {
    static TABU_CTX: core::cell::RefCell<TabuCtx> = core::cell::RefCell::new(TabuCtx {
        ptr: core::ptr::null_mut(),
        len: 0,
        iter: 0,
        tenure: 0,
        aspiration: i64::MIN,
        enabled: false,
    });
}

struct TabuGuard(TabuCtx);

impl TabuGuard {
    fn activate(ptr: *mut u32, len: usize, iter: u32, tenure: u32, aspiration: i64) -> Self {
        let prev = TABU_CTX.with(|t| {
            let mut ctx = t.borrow_mut();
            let prev = *ctx;
            let enabled = !ptr.is_null() && len > 0 && tenure > 0;
            *ctx = TabuCtx {
                ptr,
                len,
                iter,
                tenure,
                aspiration,
                enabled,
            };
            prev
        });
        TabuGuard(prev)
    }
}

impl Drop for TabuGuard {
    fn drop(&mut self) {
        TABU_CTX.with(|t| {
            *t.borrow_mut() = self.0;
        });
    }
}

#[inline]
fn tabu_active() -> bool {
    TABU_CTX.with(|t| t.borrow().enabled)
}

#[inline]
fn tabu_recent(i: usize) -> bool {
    TABU_CTX.with(|t| {
        let ctx = *t.borrow();
        if !ctx.enabled || i >= ctx.len {
            return false;
        }
        unsafe {
            let stamp = *ctx.ptr.add(i);
            ctx.iter.saturating_sub(stamp) <= ctx.tenure
        }
    })
}

#[inline]
fn tabu_aspiration_ok(state: &State, i: usize) -> bool {
    TABU_CTX.with(|t| {
        let ctx = *t.borrow();
        if !ctx.enabled {
            return true;
        }
        (state.total_value as i64) + (state.contrib[i] as i64) > ctx.aspiration
    })
}

#[inline]
fn tabu_mark(i: usize) {
    TABU_CTX.with(|t| {
        let ctx = *t.borrow();
        if !ctx.enabled || i >= ctx.len {
            return;
        }
        unsafe {
            *ctx.ptr.add(i) = ctx.iter;
        }
    });
}

fn frontier_reset(seed: Vec<usize>) {
    ACTIVE_FRONTIER.with(|f| {
        let mut v = f.borrow_mut();
        v.clear();
        for i in seed {
            if v.iter().any(|&x| x == i) {
                continue;
            }
            v.push(i);
            if v.len() > 64 {
                let drop = v.len() - 64;
                v.drain(0..drop);
            }
        }
    });
}

fn frontier_push(i: usize) {
    ACTIVE_FRONTIER.with(|f| {
        let mut v = f.borrow_mut();
        if v.iter().any(|&x| x == i) {
            return;
        }
        v.push(i);
        if v.len() > 64 {
            let drop = v.len() - 64;
            v.drain(0..drop);
        }
    });
}

fn frontier_clone() -> Vec<usize> {
    ACTIVE_FRONTIER.with(|f| f.borrow().clone())
}

fn seed_frontier_from_state(state: &State) {
    let mut seed: Vec<usize> = Vec::new();
    let k = state.hubs_static.len().min(32);
    for &i in state.hubs_static.iter().take(k) {
        if state.selected_bit[i] {
            seed.push(i);
        }
    }
    if seed.is_empty() {
        let mut cnt = 0usize;
        for i in 0..state.ch.num_items {
            if state.selected_bit[i] {
                seed.push(i);
                cnt += 1;
                if cnt >= 32 {
                    break;
                }
            }
        }
    }
    frontier_reset(seed);
}

fn restart_from_mutated_best<'a>(
    challenge: &'a Challenge,
    total_pre: &'a [i64],
    hubs_static: &'a [usize],
    neigh: Option<&'a Vec<Vec<(u16, i16)>>>,
    best_sel: &[usize],
    rng: &mut Rng,
    strategy: usize,
    strength: usize,
) -> State<'a> {
    let mut st = State::new_empty(challenge, total_pre, hubs_static, neigh);
    for &i in best_sel {
        if st.total_weight + challenge.weights[i] <= challenge.max_weight {
            st.add_item(i);
        }
    }

    let selected = st.selected_items();
    if selected.len() > 1 {
        let mut scored: Vec<(i64, i64, i64, i64, usize, u32)> = Vec::with_capacity(selected.len());

        for &i in &selected {
            let w = challenge.weights[i];
            if w == 0 {
                continue;
            }

            let base = challenge.values[i] as i64;
            let inter_part = (st.contrib[i] as i64) - base;

            let mut neg_mass: i64 = 0;
            if let Some(ng) = neigh {
                let row = unsafe { ng.get_unchecked(i) };
                for &(jj, vv) in row.iter() {
                    if vv < 0 {
                        let j = jj as usize;
                        if j < st.selected_bit.len() && st.selected_bit[j] {
                            neg_mass += vv as i64;
                        }
                    }
                }
            }

            let supp = st.support[i] as i64;
            let contrib = st.contrib[i] as i64;
            scored.push((neg_mass, supp, -inter_part, contrib, i, w));
        }

        scored.sort_unstable_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| a.3.cmp(&b.3))
                .then_with(|| a.4.cmp(&b.4))
        });

        let sel_len = selected.len();
        let mut k = (sel_len / 10).max(1);
        let k2 = strength.min((sel_len / 3).max(1));
        if k2 > k {
            k = k2;
        }
        if k >= sel_len {
            k = sel_len - 1;
        }

        let mut removed: Vec<usize> = Vec::with_capacity(k);
        for t in 0..scored.len().min(k) {
            let i = scored[t].4;
            if st.selected_bit[i] {
                st.remove_item(i);
                removed.push(i);
            }
        }
        if !removed.is_empty() {
            frontier_reset(removed);
        }
    }

    greedy_reconstruct(&mut st, rng, strategy);
    rebuild_windows(&mut st);
    micro_qkp_refinement(&mut st);
    st
}

fn one_one_swap_phase(state: &mut State) {
    let cap = state.ch.max_weight;

    loop {
        let mut improved = false;

        let mut focus: Vec<usize> = Vec::new();
        focus.reserve(state.hubs_static.len());
        for &i in state.hubs_static.iter() {
            if state.selected_bit[i] {
                focus.push(i);
            }
        }

        let selected = if focus.is_empty() {
            state.selected_items()
        } else {
            focus
        };

        for r in selected {
            if !state.selected_bit[r] {
                continue;
            }
            let wr = state.ch.weights[r];
            if wr == 0 {
                continue;
            }

            let loss = state.contrib[r];
            state.remove_item(r);

            let slack = cap - state.total_weight;

            let mut best_a: usize = usize::MAX;
            let mut best_gain: i32 = i32::MIN;
            let mut best_w: u32 = 0;

            if let Some(ref ng) = state.neigh {
                let row = unsafe { ng.get_unchecked(r) };
                let pref = row.len().min(32);
                for t in 0..pref {
                    let a = row[t].0 as usize;
                    if a == r || state.selected_bit[a] {
                        continue;
                    }
                    let wa = state.ch.weights[a];
                    if wa == 0 || wa > slack {
                        continue;
                    }
                    if tabu_active() && tabu_recent(a) && !tabu_aspiration_ok(state, a) {
                        continue;
                    }
                    let g = state.contrib[a];
                    if g > best_gain || (g == best_gain && wa < best_w) {
                        best_gain = g;
                        best_a = a;
                        best_w = wa;
                    }
                }
            }

            let hub_pref = state.hubs_static.len().min(32);
            for &a in state.hubs_static.iter().take(hub_pref) {
                if a == r || state.selected_bit[a] {
                    continue;
                }
                let wa = state.ch.weights[a];
                if wa == 0 || wa > slack {
                    continue;
                }
                if tabu_active() && tabu_recent(a) && !tabu_aspiration_ok(state, a) {
                    continue;
                }
                let g = state.contrib[a];
                if g > best_gain || (g == best_gain && wa < best_w) {
                    best_gain = g;
                    best_a = a;
                    best_w = wa;
                }
            }

            if best_a != usize::MAX && best_gain > loss {
                let wa = state.ch.weights[best_a];
                if wa != 0 && state.total_weight + wa <= cap && !state.selected_bit[best_a] {
                    state.add_item(best_a);
                    tabu_mark(best_a);
                    frontier_push(best_a);
                    improved = true;
                    break;
                }
            }

            if !state.selected_bit[r] && state.total_weight + wr <= cap {
                state.add_item(r);
                tabu_mark(r);
            } else if !state.selected_bit[r] {
                let slack2 = cap - state.total_weight;
                if slack2 > 0 {
                    let mut best_i: Option<usize>;
                    let mut best_c: i32;
                    let mut best_w2: u32;
                    let mut respect_tabu = tabu_active();
                    loop {
                        best_i = None;
                        best_c = 0;
                        best_w2 = 0;
                        for i in 0..state.ch.num_items {
                            if state.selected_bit[i] {
                                continue;
                            }
                            let w = state.ch.weights[i];
                            if w == 0 || w > slack2 {
                                continue;
                            }
                            if respect_tabu && tabu_recent(i) && !tabu_aspiration_ok(state, i) {
                                continue;
                            }
                            let c = state.contrib[i];
                            if c <= 0 {
                                continue;
                            }
                            if best_i.map_or(true, |_| c > best_c || (c == best_c && w < best_w2))
                            {
                                best_i = Some(i);
                                best_c = c;
                                best_w2 = w;
                            }
                        }
                        if best_i.is_some() || !respect_tabu {
                            break;
                        }
                        respect_tabu = false;
                    }
                    if let Some(i) = best_i {
                        state.add_item(i);
                        tabu_mark(i);
                    }
                }
            }
        }

        if !improved {
            break;
        }
    }
}

fn one_two_exchange_phase(state: &mut State) {
    if state.neigh.is_none() {
        return;
    }
    let cap = state.ch.max_weight;
    let n = state.ch.num_items;

    loop {
        let mut improved = false;

        let mut seeds: Vec<usize> = frontier_clone();
        if seeds.len() > 64 {
            let drop = seeds.len() - 64;
            seeds.drain(0..drop);
        }
        let hub_lim = state.hubs_static.len().min(32);
        for &h in state.hubs_static.iter().take(hub_lim) {
            seeds.push(h);
        }
        seeds.sort_unstable();
        seeds.dedup();

        let mut pairs: Vec<(usize, usize, i32, u32)> = Vec::new();
        if let Some(ref ng) = state.neigh {
            for &a0 in &seeds {
                if a0 >= n {
                    continue;
                }
                let row = unsafe { ng.get_unchecked(a0) };
                let pref = row.len().min(64);
                for t in 0..pref {
                    let b0 = row[t].0 as usize;
                    if b0 >= n || b0 == a0 {
                        continue;
                    }
                    let inter = row[t].1 as i32;
                    if inter <= 0 {
                        continue;
                    }
                    let wa = state.ch.weights[a0];
                    let wb = state.ch.weights[b0];
                    if wa == 0 || wb == 0 {
                        continue;
                    }
                    let (a, b) = if a0 < b0 { (a0, b0) } else { (b0, a0) };
                    pairs.push((a, b, inter, wa + wb));
                }
            }
        }
        if pairs.is_empty() {
            return;
        }

        pairs.sort_unstable_by(|x, y| {
            x.0.cmp(&y.0)
                .then_with(|| x.1.cmp(&y.1))
                .then_with(|| y.2.cmp(&x.2))
        });
        pairs.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

        let mut focus: Vec<usize> = Vec::new();
        focus.reserve(state.hubs_static.len());
        for &i in state.hubs_static.iter() {
            if state.selected_bit[i] {
                focus.push(i);
            }
        }

        let selected = if focus.is_empty() {
            state.selected_items()
        } else {
            focus
        };

        for r in selected {
            if !state.selected_bit[r] {
                continue;
            }
            let wr = state.ch.weights[r];
            if wr == 0 {
                continue;
            }

            let loss = state.contrib[r] as i64;
            state.remove_item(r);

            let slack = cap - state.total_weight;

            let mut best_a: usize = usize::MAX;
            let mut best_b: usize = usize::MAX;
            let mut best_delta: i64 = 0;

            for &(a, b, inter, wsum) in &pairs {
                if wsum > slack {
                    continue;
                }
                if state.selected_bit[a] || state.selected_bit[b] {
                    continue;
                }
                let delta =
                    (state.contrib[a] as i64) + (state.contrib[b] as i64) + (inter as i64) - loss;
                if delta > best_delta {
                    best_delta = delta;
                    best_a = a;
                    best_b = b;
                }
            }

            if best_a != usize::MAX && best_delta > 0 {
                let wa = state.ch.weights[best_a];
                let wb = state.ch.weights[best_b];
                if wa != 0
                    && wb != 0
                    && !state.selected_bit[best_a]
                    && !state.selected_bit[best_b]
                    && state.total_weight + wa + wb <= cap
                {
                    state.add_item(best_a);
                    state.add_item(best_b);
                    frontier_push(r);
                    frontier_push(best_a);
                    frontier_push(best_b);
                    improved = true;
                    break;
                } else {
                    state.add_item(r);
                }
            } else {
                state.add_item(r);
            }
        }

        if !improved {
            break;
        }
    }
}

pub fn perturb_by_strategy(
    state: &mut State,
    rng: &mut Rng,
    strength: usize,
    stall_count: usize,
    strategy: usize,
) {
    let selected = state.selected_items();
    if selected.is_empty() {
        return;
    }
    let cap = state.ch.max_weight;

    let mut target: Option<(usize, usize, u32)> = None;
    if stall_count > 0 && (strategy == 0 || strategy == 3 || strategy == 6) {
        if let Some(ref ng) = state.neigh {
            let lim = state.hubs_static.len().min(96);
            let extra = 32usize;
            let mut best: Option<(i64, usize, usize, u32)> = None;

            for &a in state.hubs_static.iter().take(lim) {
                if state.selected_bit[a] {
                    continue;
                }
                let wa = state.ch.weights[a];
                if wa == 0 || wa > cap {
                    continue;
                }

                let row = unsafe { ng.get_unchecked(a) };
                let pref = row.len().min(56);
                for t in 0..pref {
                    let (bb, vv) = row[t];
                    let b = bb as usize;
                    if b == a || state.selected_bit[b] {
                        continue;
                    }
                    let wb = state.ch.weights[b];
                    if wb == 0 || wa + wb > cap {
                        continue;
                    }
                    let v = vv as i64;
                    if v <= 0 {
                        continue;
                    }

                    let delta = (state.contrib[a] as i64) + (state.contrib[b] as i64) + v;
                    if delta <= 0 {
                        continue;
                    }

                    let mut s = (v * 1_000_000) / ((wa + wb) as i64).max(1);
                    s += delta * 40;
                    s += (state.total_interactions[a] + state.total_interactions[b]) / 2000;
                    s += (rng.next_u32() & 0x1F) as i64;
                    s += (state.usage[a] as i64 + state.usage[b] as i64) * 18;

                    if best.map_or(true, |(bs, _, _, _)| s > bs) {
                        best = Some((s, a, b, wa + wb));
                    }
                }
            }

            for _ in 0..extra {
                let a = (rng.next_u32() as usize) % state.ch.num_items;
                if state.selected_bit[a] {
                    continue;
                }
                let wa = state.ch.weights[a];
                if wa == 0 || wa > cap {
                    continue;
                }

                let row = unsafe { ng.get_unchecked(a) };
                let pref = row.len().min(48);
                for t in 0..pref {
                    let (bb, vv) = row[t];
                    let b = bb as usize;
                    if b == a || state.selected_bit[b] {
                        continue;
                    }
                    let wb = state.ch.weights[b];
                    if wb == 0 || wa + wb > cap {
                        continue;
                    }
                    let v = vv as i64;
                    if v <= 0 {
                        continue;
                    }

                    let delta = (state.contrib[a] as i64) + (state.contrib[b] as i64) + v;
                    if delta <= 0 {
                        continue;
                    }

                    let mut s = (v * 1_000_000) / ((wa + wb) as i64).max(1);
                    s += delta * 40;
                    s += (state.total_interactions[a] + state.total_interactions[b]) / 2000;
                    s += (rng.next_u32() & 0x1F) as i64;
                    s += (state.usage[a] as i64 + state.usage[b] as i64) * 18;

                    if best.map_or(true, |(bs, _, _, _)| s > bs) {
                        best = Some((s, a, b, wa + wb));
                    }
                }
            }

            if let Some((_s, a, b, w)) = best {
                target = Some((a, b, w));
            }
        }
    }

    let base_remove = (selected.len() / 10).max(1);
    let adaptive_mult = 1 + (stall_count / 2);
    let strength_scaled = strength + (selected.len() / 40);
    let n_remove = (base_remove * adaptive_mult)
        .min(
            ((selected.len() / if stall_count >= 4 { 12 } else { 16 }).max(1))
                .max(strength_scaled),
        )
        .min(selected.len() / 3);

    let (ta, tb, tw) = if let Some((a, b, w)) = target {
        (a, b, w)
    } else {
        (usize::MAX, usize::MAX, 0u32)
    };

    let slack0 = state.slack();
    let mut need_w: u32 = 0;
    if ta != usize::MAX {
        if tw > slack0 {
            need_w = tw - slack0;
        }
        need_w = need_w.saturating_add(((strength as u32) + (stall_count as u32)).min(10));
    } else if stall_count >= 3 && slack0 < 4 {
        need_w = 4 - slack0;
    }

    let mut respect_tabu_rm = tabu_active();
    let mut removal_candidates: Vec<(i64, usize, u32)> = Vec::with_capacity(selected.len());
    loop {
        removal_candidates.clear();
        for &i in &selected {
            let w = state.ch.weights[i];
            if w == 0 {
                continue;
            }
            if respect_tabu_rm && tabu_recent(i) {
                continue;
            }

            let mut s: i64 = match strategy {
                0 => state.contrib[i] as i64,
                1 => -(w as i64),
                2 => (state.contrib[i] - state.ch.values[i] as i32) as i64,
                3 => (state.contrib[i] as i64 * 1000) / (w as i64).max(1),
                4 => {
                    (state.contrib[i] as i64 * 1000) / (w as i64).max(1)
                        + (state.support[i] as i64) * 200
                }
                5 => {
                    (state.support[i] as i64) * 500
                        - (w as i64) * 220
                        + (state.contrib[i] as i64) / 50
                }
                _ => (state.contrib[i] as i64) - (state.usage[i] as i64) * 50,
            };

            if need_w >= 6 && strategy != 1 {
                let ww = (w as i64).max(1);
                s = (s * 1024) / ww - ww * 12;
            }
            if ta != usize::MAX {
                let ia = state.ch.interaction_values[i][ta] as i64;
                let ib = state.ch.interaction_values[i][tb] as i64;
                s += (ia + ib) * 4 + (state.usage[i] as i64) * 80;
            } else if stall_count >= 3 {
                s += (state.usage[i] as i64) * 30;
            }

            removal_candidates.push((s, i, w));
        }
        if !removal_candidates.is_empty() || !respect_tabu_rm {
            break;
        }
        respect_tabu_rm = false;
    }

    removal_candidates.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let mut freed: u32 = 0;
    let mut removed: usize = 0;

    let pool_lim: usize = if ta != usize::MAX { 96 } else { 64 };
    let mut pool: Vec<usize> = Vec::with_capacity(pool_lim);
    let mut cursor: usize = 0;
    let mut last_rm: usize = usize::MAX;

    while removed < n_remove || freed < need_w {
        while cursor < removal_candidates.len() && pool.len() < (pool_lim / 2).max(8) {
            let i = removal_candidates[cursor].1;
            cursor += 1;
            if !state.selected_bit[i] {
                continue;
            }
            let mut dup = false;
            for &x in &pool {
                if x == i {
                    dup = true;
                    break;
                }
            }
            if !dup {
                pool.push(i);
            }
        }

        if last_rm != usize::MAX && pool.len() < pool_lim {
            if let Some(ref ng) = state.neigh {
                let row = unsafe { ng.get_unchecked(last_rm) };
                let pref = row.len().min(32);
                for t in 0..pref {
                    let j = row[t].0 as usize;
                    if !state.selected_bit[j] {
                        continue;
                    }
                    if respect_tabu_rm && tabu_recent(j) {
                        continue;
                    }
                    let mut dup = false;
                    for &x in &pool {
                        if x == j {
                            dup = true;
                            break;
                        }
                    }
                    if !dup {
                        pool.push(j);
                        if pool.len() >= pool_lim {
                            break;
                        }
                    }
                }
            }
        }

        let mut best_pos: Option<usize> = None;
        let mut best_s: i64 = i64::MAX;

        for (pos, &i) in pool.iter().enumerate() {
            if !state.selected_bit[i] {
                continue;
            }
            let w = state.ch.weights[i];
            if w == 0 {
                continue;
            }

            let mut s: i64 = match strategy {
                0 => state.contrib[i] as i64,
                1 => -(w as i64),
                2 => (state.contrib[i] - state.ch.values[i] as i32) as i64,
                3 => (state.contrib[i] as i64 * 1000) / (w as i64).max(1),
                4 => {
                    (state.contrib[i] as i64 * 1000) / (w as i64).max(1)
                        + (state.support[i] as i64) * 200
                }
                5 => {
                    (state.support[i] as i64) * 500
                        - (w as i64) * 220
                        + (state.contrib[i] as i64) / 50
                }
                _ => (state.contrib[i] as i64) - (state.usage[i] as i64) * 50,
            };

            if need_w >= 6 && strategy != 1 {
                let ww = (w as i64).max(1);
                s = (s * 1024) / ww - ww * 12;
            }
            if ta != usize::MAX {
                let ia = state.ch.interaction_values[i][ta] as i64;
                let ib = state.ch.interaction_values[i][tb] as i64;
                s += (ia + ib) * 4 + (state.usage[i] as i64) * 80;
            } else if stall_count >= 3 {
                s += (state.usage[i] as i64) * 30;
            }

            if best_pos.map_or(true, |bp| s < best_s || (s == best_s && i < pool[bp])) {
                best_pos = Some(pos);
                best_s = s;
            }
        }

        let pos = match best_pos {
            Some(p) => p,
            None => {
                if cursor >= removal_candidates.len() {
                    break;
                }
                pool.clear();
                last_rm = usize::MAX;
                continue;
            }
        };

        let rm = pool.swap_remove(pos);
        let w = state.ch.weights[rm];
        if w != 0 && state.selected_bit[rm] {
            state.remove_item(rm);
            tabu_mark(rm);
            frontier_push(rm);
            freed = freed.saturating_add(w);
            removed += 1;
            last_rm = rm;
        }
        if pool.len() > pool_lim {
            pool.truncate(pool_lim);
        }
    }

    if ta != usize::MAX {
        let mut a = ta;
        let mut b = tb;
        if b != usize::MAX && state.ch.weights[b] < state.ch.weights[a] {
            let t = a;
            a = b;
            b = t;
        }

        let wa = state.ch.weights[a];
        if !state.selected_bit[a] && state.total_weight + wa <= cap {
            state.add_item(a);
            tabu_mark(a);
            frontier_push(a);
        }
        if b != usize::MAX {
            let wb = state.ch.weights[b];
            if !state.selected_bit[b] && state.total_weight + wb <= cap {
                state.add_item(b);
                tabu_mark(b);
                frontier_push(b);
            }
        }
    }
}

pub fn greedy_reconstruct(state: &mut State, rng: &mut Rng, strategy: usize) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let allow_zero = strategy == 2;

    loop {
        if state.total_weight >= cap {
            break;
        }
        let slack = cap - state.total_weight;

        let mut respect_tabu = tabu_active();
        let chosen: Option<usize> = loop {
            let mut best_i: Option<usize> = None;

            let mut best_w: u32 = 0;
            let mut best_c: i32 = 0;
            let mut best_supp: i64 = 0;
            let mut best_ti: i64 = 0;
            let mut best_adja: i64 = 0;
            let mut best_score_i64: i64 = 0;
            let mut best_score_i128: i128 = 0;

            let mut restricted_candidates: Vec<usize> = Vec::new();
            let use_restricted = state.neigh.is_some();
            if use_restricted {
                let frontier = frontier_clone();
                if let Some(ref ng) = state.neigh {
                    let f_lim = frontier.len().min(32);
                    for &f in frontier.iter().take(f_lim) {
                        if f >= n {
                            continue;
                        }
                        let row = unsafe { ng.get_unchecked(f) };
                        let pref = row.len().min(64);
                        for t in 0..pref {
                            restricted_candidates.push(row[t].0 as usize);
                        }
                    }
                    let hub_lim = state.hubs_static.len().min(32);
                    for &h in state.hubs_static.iter().take(hub_lim) {
                        restricted_candidates.push(h);
                    }
                    restricted_candidates.sort_unstable();
                    restricted_candidates.dedup();
                }
            }

            let mut scanned_any = false;

            let scan_iter = |i: usize,
                                 best_i: &mut Option<usize>,
                                 best_w: &mut u32,
                                 best_c: &mut i32,
                                 best_supp: &mut i64,
                                 best_ti: &mut i64,
                                 best_adja: &mut i64,
                                 best_score_i64: &mut i64,
                                 best_score_i128: &mut i128,
                                 scanned_any_ref: &mut bool| {
                if i >= n {
                    return;
                }
                if state.selected_bit[i] {
                    return;
                }
                let w = state.ch.weights[i];
                if w == 0 || w > slack {
                    return;
                }
                let c = state.contrib[i];
                if !allow_zero && c <= 0 {
                    return;
                }

                if respect_tabu && tabu_recent(i) && !tabu_aspiration_ok(state, i) {
                    return;
                }

                *scanned_any_ref = true;

                let ua = state.usage[i];

                match strategy {
                    0 => {
                        const BETA_NUM: i64 = 3;
                        const BETA_DEN: i64 = 20;

                        let wa = (w as i64).max(1);
                        let ca = c as i64;
                        let ta = state.total_interactions[i];
                        let sa = state.support[i] as i64;
                        let adja = ca * BETA_DEN + BETA_NUM * (2 * ca - ta);

                        let better = if let Some(bi) = *best_i {
                            let ub = state.usage[bi];
                            let wb = (*best_w as i64).max(1);
                            let lhs = (adja as i128) * (wb as i128);
                            let rhs = (*best_adja as i128) * (wa as i128);
                            if lhs != rhs {
                                lhs > rhs
                            } else if sa != *best_supp {
                                sa > *best_supp
                            } else if ta != *best_ti {
                                ta > *best_ti
                            } else if ca != *best_c as i64 {
                                ca > *best_c as i64
                            } else if ua != ub {
                                ua < ub
                            } else {
                                i < bi
                            }
                        } else {
                            true
                        };

                        if better {
                            *best_i = Some(i);
                            *best_w = w;
                            *best_c = c;
                            *best_supp = sa;
                            *best_ti = ta;
                            *best_adja = adja;
                        }
                    }
                    1 => {
                        let better = if let Some(bi) = *best_i {
                            let ub = state.usage[bi];
                            if w != *best_w {
                                w < *best_w
                            } else if c != *best_c {
                                c > *best_c
                            } else if ua != ub {
                                ua < ub
                            } else {
                                i < bi
                            }
                        } else {
                            true
                        };

                        if better {
                            *best_i = Some(i);
                            *best_w = w;
                            *best_c = c;
                        }
                    }
                    2 => {
                        let score = state.total_interactions[i] + (c as i64) * 10;
                        let better = if let Some(bi) = *best_i {
                            let ub = state.usage[bi];
                            if score != *best_score_i64 {
                                score > *best_score_i64
                            } else if ua != ub {
                                ua < ub
                            } else {
                                i < bi
                            }
                        } else {
                            true
                        };

                        if better {
                            *best_i = Some(i);
                            *best_w = w;
                            *best_c = c;
                            *best_score_i64 = score;
                        }
                    }
                    3 => {
                        let score = if w > 0 {
                            (c as i64 * 100) / (w as i64)
                        } else {
                            i64::MIN
                        };

                        let better = if let Some(bi) = *best_i {
                            let ub = state.usage[bi];
                            if score != *best_score_i64 {
                                score > *best_score_i64
                            } else if c != *best_c {
                                c > *best_c
                            } else if ua != ub {
                                ua < ub
                            } else {
                                i < bi
                            }
                        } else {
                            true
                        };

                        if better {
                            *best_i = Some(i);
                            *best_w = w;
                            *best_c = c;
                            *best_score_i64 = score;
                        }
                    }
                    4 => {
                        let ww = w as i128;
                        let score = (c as i128) * ww * ww / 100;

                        let better = if let Some(bi) = *best_i {
                            let ub = state.usage[bi];
                            if score != *best_score_i128 {
                                score > *best_score_i128
                            } else if c != *best_c {
                                c > *best_c
                            } else if ua != ub {
                                ua < ub
                            } else {
                                i < bi
                            }
                        } else {
                            true
                        };

                        if better {
                            *best_i = Some(i);
                            *best_w = w;
                            *best_c = c;
                            *best_score_i128 = score;
                        }
                    }
                    5 => {
                        let wa = (w as i64).max(1);
                        let ca = c as i64;
                        let sa = state.support[i] as i64;
                        let score = (ca * 1000) / wa + sa * 60 + state.total_interactions[i] / 500;

                        let better = if let Some(bi) = *best_i {
                            let ub = state.usage[bi];
                            if score != *best_score_i64 {
                                score > *best_score_i64
                            } else if ua != ub {
                                ua < ub
                            } else {
                                i > bi
                            }
                        } else {
                            true
                        };

                        if better {
                            *best_i = Some(i);
                            *best_w = w;
                            *best_c = c;
                            *best_score_i64 = score;
                        }
                    }
                    _ => {
                        let wa = (w as i128).max(1);
                        let ca = c as i128;
                        let base = (ca * 10000) / (wa * wa);
                        let penalty = (state.usage[i] as i128) * 50;
                        let score = base - penalty;

                        let better = if let Some(bi) = *best_i {
                            let ub = state.usage[bi];
                            if score != *best_score_i128 {
                                score > *best_score_i128
                            } else if ua != ub {
                                ua < ub
                            } else {
                                i < bi
                            }
                        } else {
                            true
                        };

                        if better {
                            *best_i = Some(i);
                            *best_w = w;
                            *best_c = c;
                            *best_score_i128 = score;
                        }
                    }
                }
            };

            if !restricted_candidates.is_empty() {
                for &i in &restricted_candidates {
                    scan_iter(
                        i,
                        &mut best_i,
                        &mut best_w,
                        &mut best_c,
                        &mut best_supp,
                        &mut best_ti,
                        &mut best_adja,
                        &mut best_score_i64,
                        &mut best_score_i128,
                        &mut scanned_any,
                    );
                }
            }

            let scanned_any_now = scanned_any;
            if best_i.is_none() && !scanned_any_now {
                for i in 0..n {
                    scan_iter(
                        i,
                        &mut best_i,
                        &mut best_w,
                        &mut best_c,
                        &mut best_supp,
                        &mut best_ti,
                        &mut best_adja,
                        &mut best_score_i64,
                        &mut best_score_i128,
                        &mut scanned_any,
                    );
                }
            } else if best_i.is_none() && restricted_candidates.len() < n {
                for i in 0..n {
                    scan_iter(
                        i,
                        &mut best_i,
                        &mut best_w,
                        &mut best_c,
                        &mut best_supp,
                        &mut best_ti,
                        &mut best_adja,
                        &mut best_score_i64,
                        &mut best_score_i128,
                        &mut scanned_any,
                    );
                }
            }

            if best_i.is_some() || !respect_tabu || !tabu_active() {
                break best_i;
            }
            respect_tabu = false;
        };

        let i = match chosen {
            Some(x) => x,
            None => break,
        };

        if !state.selected_bit[i] {
            let w = state.ch.weights[i];
            if w > 0 && state.total_weight + w <= cap {
                state.add_item(i);
                tabu_mark(i);
                frontier_push(i);
            } else {
                break;
            }
        } else {
            break;
        }
    }

    let slack = state.slack();
    if slack >= 2 {
        let noise = if strategy == 0 { 0 } else { 0x0F };
        greedy_fill_with_beta(state, rng, noise, true);
    }
}

fn component_exact_refine(state: &mut State) -> bool {
    let n = state.ch.num_items;
    const MAX_COMP: usize = 10;
    const N_SEEDS: usize = 10;

    let mut seeds: Vec<usize> = (0..n).filter(|&i| state.selected_bit[i]).collect();
    if seeds.len() < 2 {
        return false;
    }
    seeds.sort_unstable_by(|&a, &b| {
        let ta = state.total_interactions[a].unsigned_abs();
        let tb = state.total_interactions[b].unsigned_abs();
        tb.cmp(&ta)
    });
    seeds.truncate(N_SEEDS);

    let mut any_improved = false;

    for seed in seeds {

        if !state.selected_bit[seed] {
            continue;
        }

        let mut cluster: Vec<usize> = vec![seed];
        if let Some(ref neigh) = state.neigh {
            for &(k, _v) in neigh[seed].iter() {
                if cluster.len() >= MAX_COMP {
                    break;
                }
                cluster.push(k as usize);
            }
        } else {
            let row = &state.ch.interaction_values[seed];
            let mut nbrs: Vec<(i32, usize)> = (0..n)
                .filter(|&j| j != seed && row[j] != 0)
                .map(|j| (row[j].unsigned_abs() as i32, j))
                .collect();
            nbrs.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            for &(_, j) in nbrs.iter().take(MAX_COMP - 1) {
                cluster.push(j);
            }
        }

        let csize = cluster.len();
        if csize < 2 {
            continue;
        }

        let was_sel: Vec<bool> = cluster.iter().map(|&i| state.selected_bit[i]).collect();

        for k in 0..csize {
            if was_sel[k] {
                state.remove_item(cluster[k]);
            }
        }

        let budget = state.slack();
        let base_val: Vec<i64> = cluster.iter().map(|&i| state.contrib[i] as i64).collect();
        let weights: Vec<u32> = cluster.iter().map(|&i| state.ch.weights[i]).collect();

        let mut intra = [0i64; MAX_COMP * MAX_COMP];
        for k in 0..csize {
            for l in (k + 1)..csize {
                let v = state.ch.interaction_values[cluster[k]][cluster[l]] as i64;
                intra[k * MAX_COMP + l] = v;
                intra[l * MAX_COMP + k] = v;
            }
        }

        let mut current_mask = 0u32;
        for k in 0..csize {
            if was_sel[k] {
                current_mask |= 1u32 << k;
            }
        }

        let mut current_val = 0i64;
        for k in 0..csize {
            if current_mask & (1 << k) != 0 {
                current_val += base_val[k];
                for l in (k + 1)..csize {
                    if current_mask & (1 << l) != 0 {
                        current_val += intra[k * MAX_COMP + l];
                    }
                }
            }
        }

        let mut best_val_sub = current_val;
        let mut best_mask = current_mask;

        let max_mask = 1u32 << csize;
        for mask in 0u32..max_mask {

            let mut w = 0u32;
            let mut feasible = true;
            for k in 0..csize {
                if mask & (1 << k) != 0 {
                    w += weights[k];
                    if w > budget {
                        feasible = false;
                        break;
                    }
                }
            }
            if !feasible {
                continue;
            }

            let mut v = 0i64;
            for k in 0..csize {
                if mask & (1 << k) != 0 {
                    v += base_val[k];
                    for l in (k + 1)..csize {
                        if mask & (1 << l) != 0 {
                            v += intra[k * MAX_COMP + l];
                        }
                    }
                }
            }

            if v > best_val_sub {
                best_val_sub = v;
                best_mask = mask;
            }
        }

        for k in 0..csize {
            if best_mask & (1 << k) != 0 {
                if !state.selected_bit[cluster[k]] {
                    state.add_item(cluster[k]);
                }
            }
        }

        if best_mask != current_mask {
            any_improved = true;
        }
    }

    any_improved
}

pub fn run_one_instance(challenge: &Challenge, params: &Params) -> Solution {
    let n = challenge.num_items;
    let mut rng = Rng::from_seed(&challenge.seed);

    let (neigh_pre, total_pre): (Option<Vec<Vec<(u16, i16)>>>, Vec<i64>) = if n >= 900 {
        let (ng, tot) = build_sparse_neighbors_and_totals(challenge);
        (Some(ng), tot)
    } else {
        (None, compute_total_interactions(challenge))
    };

    let team_est = (challenge.max_weight as usize) / 6;

    let sample = n.min(96);
    let mut nz: u32 = 0;
    let mut tot: u32 = 0;
    for i in 0..sample {
        for j in 0..i {
            tot += 1;
            if challenge.interaction_values[i][j] != 0 {
                nz += 1;
            }
        }
    }
    let dens = if tot > 0 {
        (nz as f64) / (tot as f64)
    } else {
        1.0
    };
    let hard = dens < 0.10;

    let mut hubs_all: Vec<(i64, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        let w = challenge.weights[i] as i64;
        if w <= 0 {
            continue;
        }
        let s = (total_pre[i] * 1000) / w.max(1);
        hubs_all.push((s, i));
    }
    hubs_all.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let hubs_k: usize = if n >= 4500 {
        320
    } else if n >= 2500 {
        256
    } else {
        192
    };
    let hubs_static: Vec<usize> = hubs_all
        .into_iter()
        .take(hubs_k.min(n))
        .map(|(_, i)| i)
        .collect();

    let mut n_starts: usize = if n <= 600 {
        if hard { 4 } else { 3 }
    } else if n <= 1500 {
        if hard { 3 } else { 3 }
    } else if n >= 2500 {
        if hard { 4 } else { 3 }
    } else {
        2
    };

    if n <= 1500 && team_est >= 200 {
        n_starts = (n_starts + 1).min(4);
    }

    n_starts += params.extra_starts;

    let mut best: Option<State> = None;
    let mut second: Option<State> = None;

    for sid in 0..n_starts {
        let mut st = State::new_empty(challenge, &total_pre, &hubs_static, neigh_pre.as_ref());

        match sid {
            0 => build_initial_solution(&mut st),
            1 => {
                if n >= 2500 {
                    construct_frontier_cluster_grow(&mut st, &mut rng);
                } else {
                    construct_pair_seed_beta(&mut st, &mut rng);
                }
                rebuild_windows(&mut st);
            }
            2 => {
                if n >= 2500 {
                    construct_pair_seed_beta(&mut st, &mut rng);
                } else {
                    construct_forward_incremental(&mut st, 1, &mut rng);
                }
                rebuild_windows(&mut st);
            }
            _ => {
                let m = if hard { 5 } else { 4 };
                construct_forward_incremental(&mut st, m, &mut rng);
                rebuild_windows(&mut st);
            }
        }

        dp_refinement_x(&mut st, params.dp_passes_multiplier);
        rebuild_windows(&mut st);
        micro_qkp_refinement(&mut st);
        local_search_vnd(&mut st, params);

        if best
            .as_ref()
            .map_or(true, |b| st.total_value > b.total_value)
        {
            second = best;
            best = Some(st);
        } else if second
            .as_ref()
            .map_or(true, |b| st.total_value > b.total_value)
        {
            second = Some(st);
        }
    }

    if best.is_some() && second.is_some() {
        let base_val = best.as_ref().unwrap().total_value;
        let mut best_new: Option<State> = None;
        let mut best_new_val = base_val;

        {
            let mut hyb = State::new_empty(challenge, &total_pre, &hubs_static, neigh_pre.as_ref());
            {
                let b1 = best.as_ref().unwrap();
                let b2 = second.as_ref().unwrap();
                for i in 0..n {
                    if b1.selected_bit[i]
                        && b2.selected_bit[i]
                        && hyb.total_weight + challenge.weights[i] <= challenge.max_weight
                    {
                        hyb.add_item(i);
                    }
                }
            }
            greedy_fill_with_beta(&mut hyb, &mut rng, 0, true);
            rebuild_windows(&mut hyb);
            dp_refinement_x(&mut hyb, params.dp_passes_multiplier);
            rebuild_windows(&mut hyb);
            micro_qkp_refinement(&mut hyb);
            local_search_vnd(&mut hyb, params);

            if hyb.total_value > best_new_val {
                best_new_val = hyb.total_value;
                best_new = Some(hyb);
            }
        }

        let (inter_cnt, union_cnt) = {
            let b1 = best.as_ref().unwrap();
            let b2 = second.as_ref().unwrap();
            let mut inter_cnt = 0usize;
            let mut union_cnt = 0usize;
            for i in 0..n {
                let a = b1.selected_bit[i];
                let b = b2.selected_bit[i];
                if a || b {
                    union_cnt += 1;
                }
                if a && b {
                    inter_cnt += 1;
                }
            }
            (inter_cnt, union_cnt)
        };

        if union_cnt > 0 && (inter_cnt * 100) / union_cnt <= 85 {
            let mut hyb = State::new_empty(challenge, &total_pre, &hubs_static, neigh_pre.as_ref());
            {
                let b1 = best.as_ref().unwrap();
                let b2 = second.as_ref().unwrap();
                for i in 0..n {
                    if b1.selected_bit[i] || b2.selected_bit[i] {
                        hyb.add_item(i);
                    }
                }
            }

            if hyb.total_weight > challenge.max_weight {
                if n >= 2500 {
                    let mut sel: Vec<(i64, usize)> = Vec::new();
                    for i in 0..n {
                        if !hyb.selected_bit[i] {
                            continue;
                        }
                        let c = hyb.contrib[i] as i64;
                        let w = challenge.weights[i] as i64;
                        let s = if w > 0 { (c * 1000) / w } else { c * 1000 };
                        sel.push((s, i));
                    }
                    sel.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                    for &(_s, i) in &sel {
                        if hyb.total_weight <= challenge.max_weight {
                            break;
                        }
                        if hyb.selected_bit[i] {
                            hyb.remove_item(i);
                        }
                    }
                } else {
                    while hyb.total_weight > challenge.max_weight {
                        let mut worst_item: Option<usize> = None;
                        let mut worst_score: i64 = i64::MAX;
                        for i in 0..n {
                            if !hyb.selected_bit[i] {
                                continue;
                            }
                            let c = hyb.contrib[i] as i64;
                            let w = hyb.ch.weights[i] as i64;
                            let s = if w > 0 { (c * 1000) / w } else { c * 1000 };
                            if s < worst_score {
                                worst_score = s;
                                worst_item = Some(i);
                            }
                        }
                        if let Some(wi) = worst_item {
                            hyb.remove_item(wi);
                        } else {
                            break;
                        }
                    }
                }
            }

            greedy_fill_with_beta(&mut hyb, &mut rng, 0, true);
            rebuild_windows(&mut hyb);
            dp_refinement_x(&mut hyb, params.dp_passes_multiplier);
            rebuild_windows(&mut hyb);
            micro_qkp_refinement(&mut hyb);
            local_search_vnd(&mut hyb, params);

            if hyb.total_value > best_new_val {
                best_new = Some(hyb);
            }
        }

        if let Some(s) = best_new {
            best = Some(s);
        }
    }

    let mut state_int = best.unwrap();

    component_exact_refine(&mut state_int);
    rebuild_windows(&mut state_int);
    local_search_vnd(&mut state_int, params);

    let mut state_div = if let Some(s) = second {
        s
    } else {
        let mut st = State::new_empty(challenge, &total_pre, &hubs_static, neigh_pre.as_ref());
        for i in 0..n {
            if state_int.selected_bit[i] {
                st.add_item(i);
            }
        }
        st
    };

    if state_div.total_value > state_int.total_value {
        core::mem::swap(&mut state_int, &mut state_div);
    }

    let mut tabu_int: Vec<u32> = vec![0; n];
    let mut tabu_div: Vec<u32> = vec![0; n];

    seed_frontier_from_state(&state_int);

    let mut best_sel: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        if state_int.selected_bit[i] {
            best_sel.push(i);
        }
    }
    let mut best_val = state_int.total_value;
    if state_div.total_value > best_val {
        best_val = state_div.total_value;
        best_sel.clear();
        for i in 0..n {
            if state_div.selected_bit[i] {
                best_sel.push(i);
            }
        }
    }

    let mut stall_int = 0usize;
    let mut stall_div = 0usize;
    let mut stall_since_ecr = 0usize;
    let mut max_rounds = params.n_perturbation_rounds;

    if n <= 600 && hard {
        max_rounds = max_rounds.saturating_add(3);
    }

    if n <= 1200 {
        max_rounds = max_rounds.min(56);
    } else if n >= 4500 {
        max_rounds = max_rounds.min(if hard { 13 } else { 12 });
    } else if n >= 3000 {
        max_rounds = max_rounds.min(if hard { 15 } else { 14 });
    } else if n >= 2000 {
        max_rounds = max_rounds.min(16);
    }

    let mut dp_next_int = true;

    for perturbation_round in 0..max_rounds {
        let is_last_round = perturbation_round >= max_rounds - 1;
        let iter_u32 = perturbation_round as u32;

        {
            state_int.snap_bits.clone_from(&state_int.selected_bit);
            state_int.snap_contrib.clone_from(&state_int.contrib);
            state_int.snap_support.clone_from(&state_int.support);
            let prev_val = state_int.total_value;
            let prev_weight = state_int.total_weight;

            let apply_dp = !is_last_round && dp_next_int;
            let applied_dp_this_round = apply_dp;
            dp_next_int = false;
            if apply_dp {
                rebuild_windows(&mut state_int);
                dp_refinement_x(&mut state_int, params.dp_passes_multiplier);
                rebuild_windows(&mut state_int);
                micro_qkp_refinement(&mut state_int);
            }

            one_two_exchange_phase(&mut state_int);
            local_search_vnd(&mut state_int, params);

            if state_int.total_value > best_val {
                best_val = state_int.total_value;
                best_sel.clear();
                for i in 0..n {
                    if state_int.selected_bit[i] {
                        if state_int.usage[i] < u16::MAX {
                            state_int.usage[i] += 1;
                        }
                        best_sel.push(i);
                    }
                }
                stall_int = 0;
            }

            let mut accept_plateau = false;
            if state_int.total_value == prev_val {
                let mut cur_supp_sum: i64 = i64::default();
                let mut snap_supp_sum: i64 = i64::default();
                for i in 0..n {
                    if state_int.selected_bit[i] {
                        cur_supp_sum += state_int.support[i] as i64;
                    }
                    if state_int.snap_bits[i] {
                        snap_supp_sum += state_int.snap_support[i] as i64;
                    }
                }
                accept_plateau = cur_supp_sum > snap_supp_sum
                    || (cur_supp_sum == snap_supp_sum && state_int.total_weight < prev_weight);
            }

            if state_int.total_value < prev_val
                || (state_int.total_value == prev_val && !accept_plateau)
            {
                state_int.restore_snapshot(prev_val, prev_weight);

                if perturbation_round >= 7 && stall_int >= 8 {
                    let strategy = perturbation_round % 7;
                    let strength =
                        params.perturbation_strength_base + (perturbation_round as usize) / 2;
                    state_int = restart_from_mutated_best(
                        challenge,
                        &total_pre,
                        &hubs_static,
                        neigh_pre.as_ref(),
                        &best_sel,
                        &mut rng,
                        strategy,
                        strength,
                    );
                    seed_frontier_from_state(&state_int);
                    stall_int = 0;
                }

                if !is_last_round {
                    stall_int += 1;

                    let strategy = perturbation_round % 7;
                    let strength =
                        params.perturbation_strength_base + (perturbation_round as usize) / 2;
                    let tenure_u32 = strength as u32;

                    seed_frontier_from_state(&state_int);

                    let mut before = state_int.selected_items();
                    before.sort_unstable();

                    {
                        let _g = TabuGuard::activate(
                            tabu_int.as_mut_ptr(),
                            tabu_int.len(),
                            iter_u32,
                            tenure_u32,
                            best_val as i64,
                        );
                        perturb_by_strategy(&mut state_int, &mut rng, strength, stall_int, strategy);
                    }

                    for &i in &before {
                        if !state_int.selected_bit[i] {
                            tabu_int[i] = iter_u32;
                        }
                    }
                    let after = state_int.selected_items();
                    for i in after {
                        if before.binary_search(&i).is_err() {
                            tabu_int[i] = iter_u32;
                        }
                    }

                    {
                        let _g = TabuGuard::activate(
                            tabu_int.as_mut_ptr(),
                            tabu_int.len(),
                            iter_u32,
                            tenure_u32,
                            best_val as i64,
                        );
                        greedy_reconstruct(&mut state_int, &mut rng, strategy);

                        {
                            let slack = state_int.slack();
                            if slack > 0 {
                                let mut best_i: Option<usize>;
                                let mut best_c: i32;
                                let mut best_w: u32;
                                let mut respect_tabu = tabu_active();
                                loop {
                                    best_i = None;
                                    best_c = 0;
                                    best_w = 0;
                                    for i in 0..n {
                                        if state_int.selected_bit[i] {
                                            continue;
                                        }
                                        let w = challenge.weights[i];
                                        if w == 0 || w > slack {
                                            continue;
                                        }
                                        if respect_tabu
                                            && tabu_recent(i)
                                            && !tabu_aspiration_ok(&state_int, i)
                                        {
                                            continue;
                                        }
                                        let c = state_int.contrib[i];
                                        if c <= 0 {
                                            continue;
                                        }
                                        if best_i.map_or(true, |_| {
                                            c > best_c || (c == best_c && w < best_w)
                                        }) {
                                            best_i = Some(i);
                                            best_c = c;
                                            best_w = w;
                                        }
                                    }
                                    if best_i.is_some() || !respect_tabu {
                                        break;
                                    }
                                    respect_tabu = false;
                                }
                                if let Some(i) = best_i {
                                    state_int.add_item(i);
                                    tabu_mark(i);
                                    frontier_push(i);
                                }
                            }
                        }

                        one_one_swap_phase(&mut state_int);
                    }

                    rebuild_windows(&mut state_int);
                    dp_refinement_x(&mut state_int, params.dp_passes_multiplier);
                    rebuild_windows(&mut state_int);
                    micro_qkp_refinement(&mut state_int);
                    one_two_exchange_phase(&mut state_int);
                    local_search_vnd(&mut state_int, params);

                    stall_since_ecr += 1;
                    if stall_since_ecr >= 5 {
                        stall_since_ecr = 0;
                        if component_exact_refine(&mut state_int) {
                            rebuild_windows(&mut state_int);
                            local_search_vnd(&mut state_int, params);
                        }
                    }

                    if state_int.total_value > best_val {
                        stall_since_ecr = 0;
                        best_val = state_int.total_value;
                        best_sel.clear();
                        for i in 0..n {
                            if state_int.selected_bit[i] {
                                if state_int.usage[i] < u16::MAX {
                                    state_int.usage[i] += 1;
                                }
                                best_sel.push(i);
                            }
                        }
                        stall_int = 0;
                    }
                }
            } else if !applied_dp_this_round {
                dp_next_int = true;
            }
        }

        let freeze_div = !is_last_round
            && state_int.total_value >= state_div.total_value
            && stall_div > stall_int;

        if freeze_div {
            let strategy = perturbation_round % 7;
            let strength = params.perturbation_strength_base + (perturbation_round as usize) / 2;
            let tenure_u32 = strength as u32;
            let iter2_u32 = iter_u32.wrapping_add(stall_int as u32);

            seed_frontier_from_state(&state_int);

            let mut before = state_int.selected_items();
            before.sort_unstable();

            {
                let _g = TabuGuard::activate(
                    tabu_int.as_mut_ptr(),
                    tabu_int.len(),
                    iter2_u32,
                    tenure_u32,
                    best_val as i64,
                );
                perturb_by_strategy(&mut state_int, &mut rng, strength, stall_int, strategy);
            }

            for &i in &before {
                if !state_int.selected_bit[i] {
                    tabu_int[i] = iter2_u32;
                }
            }
            let after = state_int.selected_items();
            for i in after {
                if before.binary_search(&i).is_err() {
                    tabu_int[i] = iter2_u32;
                }
            }

            {
                let _g = TabuGuard::activate(
                    tabu_int.as_mut_ptr(),
                    tabu_int.len(),
                    iter2_u32,
                    tenure_u32,
                    best_val as i64,
                );
                greedy_reconstruct(&mut state_int, &mut rng, strategy);
                one_one_swap_phase(&mut state_int);
            }

            rebuild_windows(&mut state_int);
            dp_refinement_x(&mut state_int, params.dp_passes_multiplier);
            rebuild_windows(&mut state_int);
            micro_qkp_refinement(&mut state_int);
            one_two_exchange_phase(&mut state_int);
            local_search_vnd(&mut state_int, params);

            if state_int.total_value > best_val {
                best_val = state_int.total_value;
                best_sel.clear();
                for i in 0..n {
                    if state_int.selected_bit[i] {
                        if state_int.usage[i] < u16::MAX {
                            state_int.usage[i] += 1;
                        }
                        best_sel.push(i);
                    }
                }
                stall_int = 0;
            }
        } else {
            state_div.snap_bits.clone_from(&state_div.selected_bit);
            state_div.snap_contrib.clone_from(&state_div.contrib);
            state_div.snap_support.clone_from(&state_div.support);
            let prev_val = state_div.total_value;
            let prev_weight = state_div.total_weight;

            local_search_vnd(&mut state_div, params);

            if state_div.total_value > best_val {
                best_val = state_div.total_value;
                best_sel.clear();
                for i in 0..n {
                    if state_div.selected_bit[i] {
                        if state_div.usage[i] < u16::MAX {
                            state_div.usage[i] += 1;
                        }
                        best_sel.push(i);
                    }
                }
                stall_div = 0;
            }

            let mut accept_plateau = false;
            if state_div.total_value == prev_val {
                let mut cur_supp_sum: i64 = i64::default();
                let mut snap_supp_sum: i64 = i64::default();
                for i in 0..n {
                    if state_div.selected_bit[i] {
                        cur_supp_sum += state_div.support[i] as i64;
                    }
                    if state_div.snap_bits[i] {
                        snap_supp_sum += state_div.snap_support[i] as i64;
                    }
                }
                accept_plateau = cur_supp_sum > snap_supp_sum
                    || (cur_supp_sum == snap_supp_sum && state_div.total_weight < prev_weight);
            }

            if state_div.total_value < prev_val
                || (state_div.total_value == prev_val && !accept_plateau)
            {
                state_div.restore_snapshot(prev_val, prev_weight);

                if perturbation_round >= 7 && stall_div >= 8 {
                    let strategy = perturbation_round % 7;
                    let strength =
                        params.perturbation_strength_base + (perturbation_round as usize) / 2;
                    state_div = restart_from_mutated_best(
                        challenge,
                        &total_pre,
                        &hubs_static,
                        neigh_pre.as_ref(),
                        &best_sel,
                        &mut rng,
                        strategy,
                        strength,
                    );
                    seed_frontier_from_state(&state_div);
                    stall_div = 0;
                }

                if !is_last_round {
                    stall_div += 1;

                    let strategy = perturbation_round % 7;
                    let strength =
                        params.perturbation_strength_base + (perturbation_round as usize) / 2;
                    let tenure_u32 = strength as u32;

                    seed_frontier_from_state(&state_div);

                    let mut before = state_div.selected_items();
                    before.sort_unstable();

                    {
                        let _g = TabuGuard::activate(
                            tabu_div.as_mut_ptr(),
                            tabu_div.len(),
                            iter_u32,
                            tenure_u32,
                            best_val as i64,
                        );
                        perturb_by_strategy(&mut state_div, &mut rng, strength, stall_div, strategy);
                    }

                    for &i in &before {
                        if !state_div.selected_bit[i] {
                            tabu_div[i] = iter_u32;
                        }
                    }
                    let after = state_div.selected_items();
                    for i in after {
                        if before.binary_search(&i).is_err() {
                            tabu_div[i] = iter_u32;
                        }
                    }

                    {
                        let _g = TabuGuard::activate(
                            tabu_div.as_mut_ptr(),
                            tabu_div.len(),
                            iter_u32,
                            tenure_u32,
                            best_val as i64,
                        );
                        greedy_reconstruct(&mut state_div, &mut rng, strategy);
                    }
                    local_search_vnd(&mut state_div, params);

                    if state_div.total_value > best_val {
                        best_val = state_div.total_value;
                        best_sel.clear();
                        for i in 0..n {
                            if state_div.selected_bit[i] {
                                if state_div.usage[i] < u16::MAX {
                                    state_div.usage[i] += 1;
                                }
                                best_sel.push(i);
                            }
                        }
                        stall_div = 0;
                    }
                }
            }
        }

        if state_div.total_value > state_int.total_value {
            core::mem::swap(&mut state_int, &mut state_div);
            core::mem::swap(&mut stall_int, &mut stall_div);
            core::mem::swap(&mut tabu_int, &mut tabu_div);
            seed_frontier_from_state(&state_int);
            dp_next_int = true;
        }
    }

    orbital_search_fn(
        challenge,
        neigh_pre.as_ref(),
        &mut best_sel,
        &mut best_val,
    );

    orbital_2for2_exchange(challenge, neigh_pre.as_ref(), &mut best_sel, &mut best_val);

    Solution { items: best_sel }
}

fn orbital_search_fn(
    ch: &Challenge,
    neigh: Option<&Vec<Vec<(u16, i16)>>>,
    best_items: &mut Vec<usize>,
    best_val: &mut i64,
) {
    let n: usize = ch.num_items;
    let cap: i64 = ch.max_weight as i64;
    let mut sel: Vec<bool> = vec![false; n];
    for &idx in &*best_items { sel[idx] = true; }

    let mut iws: Vec<i64> = vec![0i64; n];
    orbital_compute_inter(n, &sel, neigh, ch, &mut iws);

    let mut pool: Vec<usize> = orbital_build_pool(n, &sel, ch, &iws, 60);
    let max_evals: usize = 500_000;
    let mut evals: usize = 0;
    let mut improved: bool = true;

    while improved && evals < max_evals {
        improved = false;
        let sel_idx: Vec<usize> = best_items.clone();
        let total_w: u32 = best_items.iter().map(|x| ch.weights[*x]).sum();
        for &ir in &sel_idx {
            let iw: u32 = ch.weights[ir];
            let avail: u32 = iw + (ch.max_weight - total_w);
            let mut bd: i64 = 0;
            let mut br: Vec<usize> = Vec::new();
            let rd: i64 = -(ch.values[ir] as i64) - iws[ir];

            for &c in &pool {
                if sel[c] { continue; }
                if ch.weights[c] > avail { continue; }
                let ic: i64 = orbital_gi(neigh, ch, c, ir);
                let d: i64 = rd + ch.values[c] as i64 + (iws[c] - ic);
                evals += 1;
                if d > bd { bd = d; br = vec![c]; }
                if evals >= max_evals { break; }
            }

            for p1 in 0..pool.len() {
                let c1: usize = pool[p1];
                if sel[c1] { continue; }
                if ch.weights[c1] > avail { continue; }
                for p2 in (p1 + 1)..pool.len() {
                    let c2: usize = pool[p2];
                    if sel[c2] { continue; }
                    if ch.weights[c1].checked_add(ch.weights[c2]).map_or(true, |s| s > avail) { continue; }
                    let d: i64 = rd
                        + ch.values[c1] as i64 + (iws[c1] - orbital_gi(neigh, ch, c1, ir))
                        + ch.values[c2] as i64 + (iws[c2] - orbital_gi(neigh, ch, c2, ir))
                        + orbital_gi(neigh, ch, c1, c2);
                    evals += 1;
                    if d > bd { bd = d; br = vec![c1, c2]; }
                    if evals >= max_evals { break; }
                }
                if evals >= max_evals { break; }
            }
            if bd > 0 {
                sel[ir] = false;
                best_items.retain(|&x| x != ir);
                *best_val += bd;
                for &c in &br { sel[c] = true; best_items.push(c); }
                orbital_compute_inter(n, &sel, neigh, ch, &mut iws);
                pool = orbital_build_pool(n, &sel, ch, &iws, 60);
                improved = true;
                break;
            }
            if evals >= max_evals { break; }
        }
    }
}

fn orbital_compute_inter(
    n: usize,
    sel: &[bool],
    neigh: Option<&Vec<Vec<(u16, i16)>>>,
    ch: &Challenge,
    inter: &mut [i64],
) {
    for i in 0..n { inter[i] = 0; }
    if let Some(ng) = neigh {
        for i in 0..n {
            if !sel[i] { continue; }
            for &(k, v) in ng[i].iter() {
                if sel[k as usize] { inter[i] += v as i64; }
            }
        }
    } else {
        for i in 0..n {
            if !sel[i] { continue; }
            for j in (i + 1)..n {
                if sel[j] {
                    let v: i64 = ch.interaction_values[i][j] as i64;
                    inter[i] += v;
                    inter[j] += v;
                }
            }
        }
    }
}

fn orbital_build_pool(
    n: usize,
    sel: &[bool],
    ch: &Challenge,
    iws: &[i64],
    sz: usize,
) -> Vec<usize> {
    let mut s: Vec<(i64, usize)> = Vec::new();
    for i in 0..n {
        if !sel[i] {
            let sc: i64 = ch.values[i] as i64 + iws[i];
            let w: i64 = ch.weights[i] as i64;
            let score: i64 = if w > 0 { (sc * 1000) / w } else { sc * 1000 };
            s.push((-score, i));
        }
    }
    s.sort_unstable_by_key(|x| (x.0, x.1));
    let mut result: Vec<usize> = Vec::with_capacity(sz.min(s.len()));
    for item in s.iter().take(sz) {
        result.push(item.1);
    }
    result
}

fn orbital_gi(
    neigh: Option<&Vec<Vec<(u16, i16)>>>,
    ch: &Challenge,
    a: usize,
    b: usize,
) -> i64 {
    if let Some(ng) = neigh {
        for &(k, v) in ng[a].iter() {
            if k as usize == b { return v as i64; }
        }
        return 0;
    }
    if a < b {
        ch.interaction_values[b][a] as i64
    } else {
        ch.interaction_values[a][b] as i64
    }
}

fn orbital_2for2_exchange(
    ch: &Challenge,
    neigh: Option<&Vec<Vec<(u16, i16)>>>,
    best_sel: &mut Vec<usize>,
    best_val: &mut i64,
) {
    let n = ch.num_items;
    let cap = ch.max_weight;
    const POOL_S: usize = 50;
    const POOL_U: usize = 60;
    const MAX_EVALS: usize = 150_000;

    let mut sel_bits: Vec<bool> = vec![false; n];
    for &i in &*best_sel {
        sel_bits[i] = true;
    }
    let mut total_w: u32 = best_sel.iter().map(|&i| ch.weights[i]).sum();

    let mut contrib: Vec<i64> = vec![0i64; n];
    for i in 0..n {
        if let Some(ng) = neigh {
            for &(j, v) in ng[i].iter() {
                if sel_bits[j as usize] {
                    contrib[i] += v as i64;
                }
            }
        }
    }

    let mut current_val = *best_val;
    let mut evals: usize = 0;
    let mut improved = true;

    while improved && evals < MAX_EVALS {
        improved = false;
        let slack = cap - total_w;

        let mut sel_pool: Vec<(i64, usize)> = (0..n)
            .filter(|&i| sel_bits[i] && ch.weights[i] > 0)
            .map(|i| {
                let w = ch.weights[i] as i64;
                ((ch.values[i] as i64 + contrib[i]) * 1000 / w, i)
            })
            .collect();
        sel_pool.sort_unstable_by_key(|x| x.0);

        let mut unsel_pool: Vec<(i64, usize)> = (0..n)
            .filter(|&i| !sel_bits[i] && ch.weights[i] > 0)
            .map(|i| {
                let w = ch.weights[i] as i64;
                ((ch.values[i] as i64 + contrib[i]) * 1000 / w, i)
            })
            .collect();
        unsel_pool.sort_unstable_by(|a, b| b.0.cmp(&a.0));

        let p_s = POOL_S.min(sel_pool.len());
        let p_u = POOL_U.min(unsel_pool.len());
        if p_s < 2 || p_u < 2 {
            break;
        }

        let psel: Vec<usize> = sel_pool[..p_s].iter().map(|x| x.1).collect();
        let punsel: Vec<usize> = unsel_pool[..p_u].iter().map(|x| x.1).collect();

        let mut inter_ss = vec![0i64; p_s * p_s];
        for i in 0..p_s {
            for j in (i + 1)..p_s {
                let v = orbital_gi(neigh, ch, psel[i], psel[j]);
                inter_ss[i * p_s + j] = v;
                inter_ss[j * p_s + i] = v;
            }
        }

        let mut inter_uu = vec![0i64; p_u * p_u];
        for u in 0..p_u {
            for v in (u + 1)..p_u {
                let iv = orbital_gi(neigh, ch, punsel[u], punsel[v]);
                inter_uu[u * p_u + v] = iv;
                inter_uu[v * p_u + u] = iv;
            }
        }

        let mut cross = vec![0i64; p_s * p_u];
        for i in 0..p_s {
            for u in 0..p_u {
                cross[i * p_u + u] = orbital_gi(neigh, ch, psel[i], punsel[u]);
            }
        }

        let val_s: Vec<i64> = psel.iter().map(|&i| ch.values[i] as i64).collect();
        let val_u: Vec<i64> = punsel.iter().map(|&i| ch.values[i] as i64).collect();
        let c_s: Vec<i64> = psel.iter().map(|&i| contrib[i]).collect();
        let c_u: Vec<i64> = punsel.iter().map(|&i| contrib[i]).collect();
        let w_s: Vec<u32> = psel.iter().map(|&i| ch.weights[i]).collect();
        let w_u: Vec<u32> = punsel.iter().map(|&i| ch.weights[i]).collect();

        let mut best_delta: i64 = 0;
        let mut best_quad: Option<(usize, usize, usize, usize)> = None;

        'outer: for i in 0..p_s {
            let freed1 = w_s[i];
            let loss_base = val_s[i] + c_s[i];

            for j in (i + 1)..p_s {
                let freed = freed1 + w_s[j];
                let budget = freed + slack;
                let loss = loss_base + val_s[j] + c_s[j] - inter_ss[i * p_s + j];

                for u in 0..p_u {
                    if w_u[u] > budget {
                        continue;
                    }
                    let g1 = val_u[u] + c_u[u]
                        - cross[i * p_u + u]
                        - cross[j * p_u + u];

                    for v in (u + 1)..p_u {
                        if w_u[u] + w_u[v] > budget {
                            continue;
                        }

                        evals += 1;

                        let gain = g1
                            + val_u[v]
                            + c_u[v]
                            + inter_uu[u * p_u + v]
                            - cross[i * p_u + v]
                            - cross[j * p_u + v];

                        let delta = gain - loss;
                        if delta > best_delta {
                            best_delta = delta;
                            best_quad = Some((i, j, u, v));
                        }

                        if evals >= MAX_EVALS {
                            break 'outer;
                        }
                    }
                }
            }
        }

        if let Some((i, j, u, v)) = best_quad {
            let r1 = psel[i];
            let r2 = psel[j];
            let a1 = punsel[u];
            let a2 = punsel[v];

            sel_bits[r1] = false;
            sel_bits[r2] = false;
            sel_bits[a1] = true;
            sel_bits[a2] = true;
            total_w = total_w - w_s[i] - w_s[j] + w_u[u] + w_u[v];
            current_val += best_delta;

            for k in 0..n {
                contrib[k] = 0;
            }
            for k in 0..n {
                if let Some(ng) = neigh {
                    for &(jj, vv) in ng[k].iter() {
                        if sel_bits[jj as usize] {
                            contrib[k] += vv as i64;
                        }
                    }
                }
            }

            if current_val > *best_val {
                *best_val = current_val;
                best_sel.clear();
                for k in 0..n {
                    if sel_bits[k] {
                        best_sel.push(k);
                    }
                }
            }

            improved = true;
        }
    }
}
}
pub struct Solver;

impl Solver {
    pub fn solve(
        challenge: &Challenge,
        _save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
        hyperparameters: &Option<Map<String, Value>>,
    ) -> Result<Option<Solution>> {
        let params = Params::initialize(hyperparameters);
        let solution = run_one_instance(challenge, &params);
        Ok(Some(solution))
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
    println!("Quadratic Knapsack - Multi-Start ILS with Hybrid Basin Discovery");
    println!("No hyperparameters. All settings are hardcoded to defaults.");
}
