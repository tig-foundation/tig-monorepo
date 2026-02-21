use anyhow::Result;
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use tig_challenges::knapsack::*;

// === Parameters ===

const DIFF_LIM: usize = 4;

struct Params {
    core_half: usize,
    n_starts: usize,
    max_rounds: usize,
    perturb_base: usize,
    max_vnd_iters: usize,
    use_multi_item: bool,
    use_2for2: bool,
}

impl Params {
    fn for_challenge(ch: &Challenge) -> Self {
        let n = ch.num_items;
        let total_weight: u32 = ch.weights.iter().sum();
        let budget_pct = if total_weight > 0 {
            (ch.max_weight as f64 * 100.0) / total_weight as f64
        } else { 10.0 };
        let high_budget = budget_pct >= 18.0;

        let mut p = if n <= 600 {
            Self { core_half: 70, n_starts: 5, max_rounds: 24, perturb_base: 3, max_vnd_iters: 120, use_multi_item: true, use_2for2: true }
        } else if n <= 1200 {
            Self { core_half: 50, n_starts: 4, max_rounds: 22, perturb_base: 3, max_vnd_iters: 400, use_multi_item: true, use_2for2: true }
        } else if n <= 2500 {
            Self { core_half: 35, n_starts: 3, max_rounds: 18, perturb_base: 3, max_vnd_iters: 380, use_multi_item: true, use_2for2: true }
        } else if n <= 4000 {
            Self { core_half: 32, n_starts: 2, max_rounds: 14, perturb_base: 4, max_vnd_iters: 340, use_multi_item: true, use_2for2: false }
        } else {
            Self { core_half: 35, n_starts: 1, max_rounds: 12, perturb_base: 4, max_vnd_iters: 260, use_multi_item: true, use_2for2: false }
        };

        if high_budget {
            if n <= 2000 {
                p.core_half = (p.core_half * 3 / 2).min(100);
            } else {
                p.core_half = (p.core_half * 5 / 2).min(120);
            }
        }

        p
    }
}

// === Simple RNG (deterministic from seed) ===

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

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 7;
        x ^= x >> 9;
        x ^= x << 8;
        self.state = x;
        x
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
}

// === State ===

pub struct State<'a> {
    ch: &'a Challenge,
    selected_bit: Vec<bool>,
    contrib: Vec<i32>,
    total_value: i64,
    total_weight: u32,
    // Windows
    window_locked: Vec<usize>,
    window_core: Vec<usize>,
    window_rejected: Vec<usize>,
    core_bins: Vec<(u32, Vec<usize>)>,
    // Caches
    dp_cache: Vec<i64>,
    choose_cache: Vec<u8>,
    snap_bits: Vec<bool>,
    snap_contrib: Vec<i32>,
}

impl<'a> State<'a> {
    fn new_empty(ch: &'a Challenge) -> Self {
        let n = ch.num_items;
        let mut contrib = vec![0i32; n];
        for i in 0..n {
            contrib[i] = ch.values[i] as i32;
        }
        Self {
            ch,
            selected_bit: vec![false; n],
            contrib,
            total_value: 0,
            total_weight: 0,
            window_locked: Vec::new(),
            window_core: Vec::new(),
            window_rejected: Vec::new(),
            core_bins: Vec::new(),
            dp_cache: Vec::new(),
            choose_cache: Vec::new(),
            snap_bits: vec![false; n],
            snap_contrib: vec![0i32; n],
        }
    }

    fn selected_items(&self) -> Vec<usize> {
        (0..self.ch.num_items).filter(|&i| self.selected_bit[i]).collect()
    }

    #[inline(always)]
    fn slack(&self) -> u32 {
        self.ch.max_weight - self.total_weight
    }

    #[inline(always)]
    fn add_item(&mut self, i: usize) {
        self.total_value += self.contrib[i] as i64;
        self.total_weight += self.ch.weights[i];
        let n = self.ch.num_items;
        let row_ptr = unsafe { self.ch.interaction_values.get_unchecked(i).as_ptr() };
        let contrib_ptr = self.contrib.as_mut_ptr();
        unsafe {
            for k in 0..n {
                let ck = contrib_ptr.add(k);
                *ck = (*ck).wrapping_add(*row_ptr.add(k));
            }
        }
        self.selected_bit[i] = true;
    }

    #[inline(always)]
    fn remove_item(&mut self, j: usize) {
        self.total_value -= self.contrib[j] as i64;
        self.total_weight -= self.ch.weights[j];
        let n = self.ch.num_items;
        let row_ptr = unsafe { self.ch.interaction_values.get_unchecked(j).as_ptr() };
        let contrib_ptr = self.contrib.as_mut_ptr();
        unsafe {
            for k in 0..n {
                let ck = contrib_ptr.add(k);
                *ck = (*ck).wrapping_sub(*row_ptr.add(k));
            }
        }
        self.selected_bit[j] = false;
    }

    #[inline(always)]
    fn replace_item(&mut self, rm: usize, cand: usize) {
        self.remove_item(rm);
        self.add_item(cand);
    }

    fn save_snapshot(&mut self) {
        self.snap_bits.clone_from(&self.selected_bit);
        self.snap_contrib.clone_from(&self.contrib);
    }

    fn restore_snapshot(&mut self, snap_value: i64, snap_weight: u32) {
        self.selected_bit.clone_from(&self.snap_bits);
        self.contrib.clone_from(&self.snap_contrib);
        self.total_value = snap_value;
        self.total_weight = snap_weight;
    }
}

// === Window Management ===

fn set_windows(state: &mut State, by_density: &[usize], idx_first_rej: usize, idx_last_ins: usize, core_half: usize) {
    let n = state.ch.num_items;
    let mut left = idx_first_rej.saturating_sub(core_half + 1);
    let right = (idx_last_ins + core_half + 1).min(n);
    if left > right { left = right; }

    state.window_locked = by_density[..left].to_vec();
    state.window_core = by_density[left..right].to_vec();
    state.window_rejected = by_density[right..].to_vec();

    let mut bins: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for &i in &state.window_core {
        bins.entry(state.ch.weights[i]).or_default().push(i);
    }
    state.core_bins = bins.into_iter().collect();
}

fn rebuild_windows(state: &mut State, core_half: usize) {
    let n = state.ch.num_items;
    if n == 0 { return; }
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
    let mut idx_last_ins = 0usize;
    let mut idx_first_rej = n;
    for (idx, &i) in by_density.iter().enumerate() {
        let w = state.ch.weights[i];
        if w <= rem {
            rem -= w;
            idx_last_ins = idx;
        } else if idx_first_rej == n {
            idx_first_rej = idx;
        }
    }
    set_windows(state, &by_density, idx_first_rej, idx_last_ins, core_half);
}

// === Construction ===

fn build_initial_solution(state: &mut State, core_half: usize) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;

    // Start with all items, remove worst by density
    for i in 0..n {
        state.add_item(i);
    }
    while state.total_weight > cap {
        let mut worst_item = 0;
        let mut worst_score = i64::MAX;
        for i in 0..n {
            if state.selected_bit[i] {
                let c = state.contrib[i] as i64;
                let w = (state.ch.weights[i] as i64).max(1);
                let score = (c * 1000) / w;
                if score < worst_score {
                    worst_score = score;
                    worst_item = i;
                }
            }
        }
        state.remove_item(worst_item);
    }

    // Iterative density-based refinement
    let mut by_density: Vec<usize> = (0..n).collect();
    let mut idx_last_ins = 0;
    let mut idx_first_rej = n;

    for _ in 0..3 {
        idx_last_ins = 0;
        idx_first_rej = n;
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
                idx_last_ins = idx;
            } else if idx_first_rej == n {
                idx_first_rej = idx;
            }
        }

        let mut in_target = vec![false; n];
        for &i in &target_sel { in_target[i] = true; }
        let mut to_remove = Vec::new();
        let mut to_add = Vec::new();
        for i in 0..n {
            if state.selected_bit[i] && !in_target[i] { to_remove.push(i); }
            if !state.selected_bit[i] && in_target[i] { to_add.push(i); }
        }
        if to_remove.is_empty() && to_add.is_empty() { break; }
        for &r in &to_remove { state.remove_item(r); }
        for &a in &to_add { state.add_item(a); }
    }

    set_windows(state, &by_density, idx_first_rej, idx_last_ins, core_half);
}

fn construct_forward(state: &mut State, mode: usize, rng: &mut Rng) {
    let n = state.ch.num_items;
    loop {
        let slack = state.slack();
        if slack == 0 { break; }

        let mut best_i: Option<usize> = None;
        let mut best_s: i64 = i64::MIN;
        let mut second_i: Option<usize> = None;
        let mut second_s: i64 = i64::MIN;

        for i in 0..n {
            if state.selected_bit[i] { continue; }
            let w_u = state.ch.weights[i];
            if w_u > slack { continue; }
            let c = state.contrib[i] as i64;
            if c <= 0 { continue; }

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
                second_s = best_s; second_i = best_i;
                best_s = s; best_i = Some(i);
            } else if s > second_s {
                second_s = s; second_i = Some(i);
            }
        }

        let pick = if mode >= 4 && second_i.is_some() {
            let m = if mode >= 5 { 1 } else { 3 };
            if (rng.next_u32() & m) == 0 { second_i } else { best_i }
        } else {
            best_i
        };

        if let Some(i) = pick { state.add_item(i); } else { break; }
    }
}

// === Core DP Refinement ===

fn dp_refinement(state: &mut State) {
    let ch = state.ch;
    let locked = &state.window_locked;
    let core = &state.window_core;

    let used_locked: u64 = locked.iter().map(|&i| ch.weights[i] as u64).sum();
    let rem_cap = (ch.max_weight as u64).saturating_sub(used_locked) as usize;

    let myk = core.len();
    if myk == 0 { return; }

    // Check if all positive-contrib items fit
    let mut total_pos_weight: usize = 0;
    let mut all_pos_fit = true;
    for &it in core {
        if state.contrib[it] > 0 {
            total_pos_weight += ch.weights[it] as usize;
            if total_pos_weight > rem_cap {
                all_pos_fit = false;
            }
        }
    }

    let target = if rem_cap == 0 {
        let mut sel: Vec<usize> = locked.to_vec();
        for &it in core {
            if ch.weights[it] == 0 && state.contrib[it] > 0 {
                sel.push(it);
            }
        }
        sel.sort_unstable();
        sel
    } else if all_pos_fit {
        let mut sel: Vec<usize> = locked.to_vec();
        for &it in core {
            if state.contrib[it] > 0 {
                sel.push(it);
            }
        }
        sel.sort_unstable();
        sel
    } else {
        // Full DP on core
        let mut total_core_w: usize = 0;
        for &it in core { total_core_w += ch.weights[it] as usize; }
        let myw = rem_cap.min(total_core_w);
        let dp_size = myw + 1;
        let choose_size = myk * dp_size;

        if state.dp_cache.len() < dp_size {
            state.dp_cache.resize(dp_size, i64::MIN / 4);
        }
        if state.choose_cache.len() < choose_size {
            state.choose_cache.resize(choose_size, 0);
        }

        let init = i64::MIN / 4;
        for v in &mut state.dp_cache[..dp_size] { *v = init; }
        state.dp_cache[0] = 0;
        state.choose_cache[..choose_size].fill(0);

        let mut w_hi: usize = 0;
        for (t, &it) in core.iter().enumerate() {
            let wt = ch.weights[it] as usize;
            if wt > myw { continue; }
            let val = state.contrib[it] as i64;
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

        let mut sel: Vec<usize> = locked.to_vec();
        let mut w_star = (0..=myw).max_by_key(|&w| state.dp_cache[w]).unwrap_or(0);
        for t in (0..myk).rev() {
            let it = core[t];
            let wt = ch.weights[it] as usize;
            if wt <= w_star && state.choose_cache[t * dp_size + w_star] == 1 {
                sel.push(it);
                w_star -= wt;
            }
        }
        sel.sort_unstable();
        sel
    };

    // Apply target via add/remove operations
    let n = ch.num_items;
    let mut to_remove = Vec::new();
    let mut to_add = Vec::new();
    let mut j = 0;
    let m = target.len();
    for i in 0..n {
        let in_target = j < m && target[j] == i;
        if in_target { j += 1; }
        if state.selected_bit[i] && !in_target { to_remove.push(i); }
        else if in_target && !state.selected_bit[i] { to_add.push(i); }
    }
    for &r in &to_remove { state.remove_item(r); }
    for &a in &to_add { state.add_item(a); }
}

// === Standard VND Moves ===

fn apply_best_add(state: &mut State) -> bool {
    let slack = state.slack();
    if slack == 0 { return false; }
    let mut best: Option<(usize, i32)> = None;

    for (bw, items) in &state.core_bins {
        if *bw > slack { break; }
        for &cand in items {
            if state.selected_bit[cand] { continue; }
            let d = state.contrib[cand];
            if d > 0 && best.map_or(true, |(_, bd)| d > bd) {
                best = Some((cand, d));
            }
        }
    }
    if best.is_none() {
        for &cand in &state.window_rejected {
            if state.selected_bit[cand] { continue; }
            let w = state.ch.weights[cand];
            if w <= slack {
                let d = state.contrib[cand];
                if d > 0 && best.map_or(true, |(_, bd)| d > bd) {
                    best = Some((cand, d));
                }
            }
        }
    }

    if let Some((cand, _)) = best { state.add_item(cand); true }
    else { false }
}

fn apply_best_swap_equal(state: &mut State, used: &[usize]) -> bool {
    let mut best: Option<(usize, usize, i32)> = None;
    for &rm in used {
        let w_rm = state.ch.weights[rm];
        if let Ok(idx) = state.core_bins.binary_search_by_key(&w_rm, |(w, _)| *w) {
            for &cand in &state.core_bins[idx].1 {
                if state.selected_bit[cand] { continue; }
                let d = state.contrib[cand] - state.contrib[rm]
                    - state.ch.interaction_values[cand][rm];
                if d > 0 && best.map_or(true, |(_, _, bd)| d > bd) {
                    best = Some((cand, rm, d));
                }
            }
        }
    }
    if let Some((cand, rm, _)) = best { state.replace_item(rm, cand); true }
    else { false }
}

fn apply_best_swap_reduce(state: &mut State, used: &[usize]) -> bool {
    let mut best: Option<(usize, usize, i32)> = None;
    for &rm in used {
        let w_rm = state.ch.weights[rm];
        if w_rm == 0 { continue; }
        let w_min = w_rm.saturating_sub(DIFF_LIM as u32);
        for (bw, items) in &state.core_bins {
            if *bw >= w_rm { break; }
            if *bw < w_min { continue; }
            for &cand in items {
                if state.selected_bit[cand] { continue; }
                let d = state.contrib[cand] - state.contrib[rm]
                    - state.ch.interaction_values[cand][rm];
                if d > 0 && best.map_or(true, |(_, _, bd)| d > bd) {
                    best = Some((cand, rm, d));
                }
            }
        }
    }
    if let Some((cand, rm, _)) = best { state.replace_item(rm, cand); true }
    else { false }
}

fn apply_best_swap_increase(state: &mut State, used: &[usize]) -> bool {
    let slack = state.slack();
    if slack == 0 { return false; }
    let mut best: Option<(usize, usize, f64)> = None;
    for &rm in used {
        let w_rm = state.ch.weights[rm];
        let max_dw = (DIFF_LIM as u32).min(slack);
        let w_max = w_rm.saturating_add(max_dw);
        for (bw, items) in &state.core_bins {
            if *bw <= w_rm { continue; }
            if *bw > w_max { break; }
            let dw = *bw - w_rm;
            if dw > slack { break; }
            for &cand in items {
                if state.selected_bit[cand] { continue; }
                let d = state.contrib[cand] - state.contrib[rm]
                    - state.ch.interaction_values[cand][rm];
                if d > 0 {
                    let ratio = (d as f64) / (dw as f64);
                    if best.map_or(true, |(_, _, br)| ratio > br) {
                        best = Some((cand, rm, ratio));
                    }
                }
            }
        }
    }
    if let Some((cand, rm, _)) = best { state.replace_item(rm, cand); true }
    else { false }
}

// === NEW: Multi-item VND moves ===

/// Try to remove 1 selected item and add 2 unselected items (1-for-2 swap)
/// Delta = contrib[c1] + contrib[c2] + interaction[c1][c2] - contrib[rm] - interaction[c1][rm] - interaction[c2][rm]
fn try_1for2(state: &mut State) -> bool {
    let slack = state.slack();

    // Get worst used items from core window
    let mut worst_used: Vec<usize> = Vec::new();
    for &i in &state.window_core {
        if state.selected_bit[i] { worst_used.push(i); }
    }
    // Sort by contribution density ascending (worst first)
    worst_used.sort_unstable_by(|&a, &b| {
        let da = (state.contrib[a] as i64) * (state.ch.weights[b] as i64).max(1);
        let db = (state.contrib[b] as i64) * (state.ch.weights[a] as i64).max(1);
        da.cmp(&db)
    });
    if worst_used.len() > 40 { worst_used.truncate(40); }

    // Get best unused items from core window
    let mut best_unused: Vec<usize> = Vec::new();
    for &i in &state.window_core {
        if !state.selected_bit[i] { best_unused.push(i); }
    }
    best_unused.sort_unstable_by(|&a, &b| {
        let da = (state.contrib[a] as i64) * (state.ch.weights[b] as i64).max(1);
        let db = (state.contrib[b] as i64) * (state.ch.weights[a] as i64).max(1);
        db.cmp(&da)
    });
    if best_unused.len() > 40 { best_unused.truncate(40); }

    let mut best_move: Option<(usize, usize, usize, i64)> = None;

    for &rm in &worst_used {
        let w_freed = slack + state.ch.weights[rm];
        let c_rm = state.contrib[rm] as i64;

        for ai in 0..best_unused.len() {
            let c1 = best_unused[ai];
            let w1 = state.ch.weights[c1];
            if w1 >= w_freed { continue; }
            let remaining = w_freed - w1;

            for aj in (ai + 1)..best_unused.len() {
                let c2 = best_unused[aj];
                let w2 = state.ch.weights[c2];
                if w2 > remaining { continue; }

                let delta = (state.contrib[c1] as i64)
                    + (state.contrib[c2] as i64)
                    + (state.ch.interaction_values[c1][c2] as i64)
                    - c_rm
                    - (state.ch.interaction_values[c1][rm] as i64)
                    - (state.ch.interaction_values[c2][rm] as i64);

                if delta > 0 && best_move.map_or(true, |(_, _, _, bd)| delta > bd) {
                    best_move = Some((rm, c1, c2, delta));
                }
            }
        }
    }

    if let Some((rm, c1, c2, _)) = best_move {
        state.remove_item(rm);
        state.add_item(c1);
        state.add_item(c2);
        true
    } else {
        false
    }
}

/// Try to remove 2 selected items and add 1 unselected item (2-for-1 swap)
/// Delta = contrib[cand] - contrib[rm1] - contrib[rm2] + interaction[rm1][rm2]
///         - interaction[cand][rm1] - interaction[cand][rm2]
fn try_2for1(state: &mut State) -> bool {
    // Get worst used items from core window
    let mut worst_used: Vec<usize> = Vec::new();
    for &i in &state.window_core {
        if state.selected_bit[i] { worst_used.push(i); }
    }
    worst_used.sort_unstable_by(|&a, &b| {
        let da = (state.contrib[a] as i64) * (state.ch.weights[b] as i64).max(1);
        let db = (state.contrib[b] as i64) * (state.ch.weights[a] as i64).max(1);
        da.cmp(&db)
    });
    if worst_used.len() > 30 { worst_used.truncate(30); }

    // Get best unused items
    let mut best_unused: Vec<usize> = Vec::new();
    for &i in &state.window_core {
        if !state.selected_bit[i] { best_unused.push(i); }
    }
    // Also check some rejected items (might be good candidates with freed weight)
    for &i in &state.window_rejected {
        if !state.selected_bit[i] {
            best_unused.push(i);
            if best_unused.len() > 50 { break; }
        }
    }
    best_unused.sort_unstable_by(|&a, &b| {
        let da = (state.contrib[a] as i64) * (state.ch.weights[b] as i64).max(1);
        let db = (state.contrib[b] as i64) * (state.ch.weights[a] as i64).max(1);
        db.cmp(&da)
    });
    if best_unused.len() > 40 { best_unused.truncate(40); }

    let slack = state.slack();
    let mut best_move: Option<(usize, usize, usize, i64)> = None;

    for ai in 0..worst_used.len() {
        let rm1 = worst_used[ai];
        let c_rm1 = state.contrib[rm1] as i64;

        for aj in (ai + 1)..worst_used.len() {
            let rm2 = worst_used[aj];
            let c_rm2 = state.contrib[rm2] as i64;
            let w_freed = slack + state.ch.weights[rm1] + state.ch.weights[rm2];
            let synergy_rm = state.ch.interaction_values[rm1][rm2] as i64;

            for &cand in &best_unused {
                let w_c = state.ch.weights[cand];
                if w_c > w_freed { continue; }

                let delta = (state.contrib[cand] as i64)
                    - c_rm1 - c_rm2
                    + synergy_rm
                    - (state.ch.interaction_values[cand][rm1] as i64)
                    - (state.ch.interaction_values[cand][rm2] as i64);

                if delta > 0 && best_move.map_or(true, |(_, _, _, bd)| delta > bd) {
                    best_move = Some((rm1, rm2, cand, delta));
                }
            }
        }
    }

    if let Some((rm1, rm2, cand, _)) = best_move {
        state.remove_item(rm1);
        state.remove_item(rm2);
        state.add_item(cand);
        true
    } else {
        false
    }
}

/// Try to swap 2 selected for 2 unselected (2-for-2 swap)
/// Delta = contrib[c1] + contrib[c2] + interaction[c1][c2]
///       - contrib[rm1] - contrib[rm2] + interaction[rm1][rm2]
///       - interaction[c1][rm1] - interaction[c1][rm2]
///       - interaction[c2][rm1] - interaction[c2][rm2]
fn try_2for2(state: &mut State) -> bool {
    let slack = state.slack();

    let mut worst_used: Vec<usize> = Vec::new();
    for &i in &state.window_core {
        if state.selected_bit[i] { worst_used.push(i); }
    }
    worst_used.sort_unstable_by(|&a, &b| {
        let da = (state.contrib[a] as i64) * (state.ch.weights[b] as i64).max(1);
        let db = (state.contrib[b] as i64) * (state.ch.weights[a] as i64).max(1);
        da.cmp(&db)
    });
    if worst_used.len() > 25 { worst_used.truncate(25); }

    let mut best_unused: Vec<usize> = Vec::new();
    for &i in &state.window_core {
        if !state.selected_bit[i] { best_unused.push(i); }
    }
    best_unused.sort_unstable_by(|&a, &b| {
        let da = (state.contrib[a] as i64) * (state.ch.weights[b] as i64).max(1);
        let db = (state.contrib[b] as i64) * (state.ch.weights[a] as i64).max(1);
        db.cmp(&da)
    });
    if best_unused.len() > 25 { best_unused.truncate(25); }

    let mut best_move: Option<(usize, usize, usize, usize, i64)> = None;

    for ri in 0..worst_used.len() {
        let rm1 = worst_used[ri];
        let c_rm1 = state.contrib[rm1] as i64;
        let w_rm1 = state.ch.weights[rm1];

        for rj in (ri + 1)..worst_used.len() {
            let rm2 = worst_used[rj];
            let c_rm2 = state.contrib[rm2] as i64;
            let w_rm2 = state.ch.weights[rm2];
            let w_freed = slack + w_rm1 + w_rm2;
            let syn_rm = state.ch.interaction_values[rm1][rm2] as i64;
            let base_cost = c_rm1 + c_rm2 - syn_rm;

            for ci in 0..best_unused.len() {
                let c1 = best_unused[ci];
                let w1 = state.ch.weights[c1];
                if w1 >= w_freed { continue; }
                let remaining = w_freed - w1;
                let cc1 = state.contrib[c1] as i64;
                let ic1_rm1 = state.ch.interaction_values[c1][rm1] as i64;
                let ic1_rm2 = state.ch.interaction_values[c1][rm2] as i64;

                for cj in (ci + 1)..best_unused.len() {
                    let c2 = best_unused[cj];
                    let w2 = state.ch.weights[c2];
                    if w2 > remaining { continue; }

                    let cc2 = state.contrib[c2] as i64;
                    let delta = cc1 + cc2
                        + (state.ch.interaction_values[c1][c2] as i64)
                        - base_cost
                        - ic1_rm1 - ic1_rm2
                        - (state.ch.interaction_values[c2][rm1] as i64)
                        - (state.ch.interaction_values[c2][rm2] as i64);

                    if delta > 0 && best_move.map_or(true, |(_, _, _, _, bd)| delta > bd) {
                        best_move = Some((rm1, rm2, c1, c2, delta));
                    }
                }
            }
        }
    }

    if let Some((rm1, rm2, c1, c2, _)) = best_move {
        state.remove_item(rm1);
        state.remove_item(rm2);
        state.add_item(c1);
        state.add_item(c2);
        true
    } else {
        false
    }
}

/// Try pair add: add 2 unselected items if both fit and combined value is positive
fn try_pair_add(state: &mut State) -> bool {
    let slack = state.slack();
    if slack <= 1 { return false; }

    let mut candidates: Vec<usize> = Vec::new();
    for &i in &state.window_core {
        if !state.selected_bit[i] && state.ch.weights[i] < slack {
            candidates.push(i);
        }
    }
    if candidates.len() > 50 { candidates.truncate(50); }

    let mut best: Option<(usize, usize, i64)> = None;
    for ai in 0..candidates.len() {
        let c1 = candidates[ai];
        let w1 = state.ch.weights[c1];
        let rem = slack - w1;
        let cc1 = state.contrib[c1] as i64;

        for aj in (ai + 1)..candidates.len() {
            let c2 = candidates[aj];
            if state.ch.weights[c2] > rem { continue; }

            let delta = cc1 + (state.contrib[c2] as i64)
                + (state.ch.interaction_values[c1][c2] as i64);

            if delta > 0 && best.map_or(true, |(_, _, bd)| delta > bd) {
                best = Some((c1, c2, delta));
            }
        }
    }

    if let Some((c1, c2, _)) = best {
        state.add_item(c1);
        state.add_item(c2);
        true
    } else {
        false
    }
}

// === Extended VND ===

fn extended_vnd(state: &mut State, params: &Params) {
    let mut iterations = 0;
    let max_iter = params.max_vnd_iters;

    loop {
        iterations += 1;
        if iterations > max_iter { break; }

        // Phase 1: Add moves
        if apply_best_add(state) { continue; }
        if try_pair_add(state) { continue; }

        // Phase 2: Single-item swaps
        let mut used: Vec<usize> = Vec::new();
        for &i in &state.window_core {
            if state.selected_bit[i] { used.push(i); }
        }

        if apply_best_swap_reduce(state, &used) { continue; }
        if apply_best_swap_equal(state, &used) { continue; }
        if apply_best_swap_increase(state, &used) { continue; }

        // Phase 3: Multi-item swaps (only when enabled)
        if params.use_multi_item {
            if try_1for2(state) { continue; }
            if try_2for1(state) { continue; }
            if params.use_2for2 {
                if try_2for2(state) { continue; }
            }
        }

        break;
    }
}

// === Polish Step ===

fn polish(state: &mut State) {
    let n = state.ch.num_items;
    let k = 30usize.min(n);

    // Build fresh sorted candidate lists
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by(|&a, &b| {
        let ra = (state.contrib[a] as i64) * (state.ch.weights[b] as i64).max(1);
        let rb = (state.contrib[b] as i64) * (state.ch.weights[a] as i64).max(1);
        rb.cmp(&ra)
    });

    let mut unused_top: Vec<usize> = Vec::new();
    for &i in &idx {
        if !state.selected_bit[i] {
            unused_top.push(i);
            if unused_top.len() >= k { break; }
        }
    }
    let mut used_worst: Vec<usize> = Vec::new();
    for &i in idx.iter().rev() {
        if state.selected_bit[i] {
            used_worst.push(i);
            if used_worst.len() >= k { break; }
        }
    }

    loop {
        let slack = state.slack();
        // Try add
        let mut best_add: Option<(usize, i32)> = None;
        if slack > 0 {
            for &i in &unused_top {
                if state.selected_bit[i] { continue; }
                if state.ch.weights[i] > slack { continue; }
                let d = state.contrib[i];
                if d > 0 && best_add.map_or(true, |(_, bd)| d > bd) {
                    best_add = Some((i, d));
                }
            }
        }
        if let Some((i, _)) = best_add { state.add_item(i); continue; }

        // Try equal-weight swap
        let mut best_swap: Option<(usize, usize, i32)> = None;
        for &rm in &used_worst {
            if !state.selected_bit[rm] { continue; }
            let w_rm = state.ch.weights[rm];
            for &cand in &unused_top {
                if state.selected_bit[cand] { continue; }
                if state.ch.weights[cand] != w_rm { continue; }
                let d = state.contrib[cand] - state.contrib[rm]
                    - state.ch.interaction_values[cand][rm];
                if d > 0 && best_swap.map_or(true, |(_, _, bd)| d > bd) {
                    best_swap = Some((cand, rm, d));
                }
            }
        }
        if let Some((cand, rm, _)) = best_swap { state.replace_item(rm, cand); continue; }

        // Try reduce-weight swap
        let mut best_red: Option<(usize, usize, i32)> = None;
        for &rm in &used_worst {
            if !state.selected_bit[rm] { continue; }
            let w_rm = state.ch.weights[rm];
            for &cand in &unused_top {
                if state.selected_bit[cand] { continue; }
                let w_c = state.ch.weights[cand];
                if w_c >= w_rm { continue; }
                let dw = w_rm - w_c;
                if dw as usize > DIFF_LIM { continue; }
                let d = state.contrib[cand] - state.contrib[rm]
                    - state.ch.interaction_values[cand][rm];
                if d > 0 && best_red.map_or(true, |(_, _, bd)| d > bd) {
                    best_red = Some((cand, rm, d));
                }
            }
        }
        if let Some((cand, rm, _)) = best_red { state.replace_item(rm, cand); continue; }

        // Try increase-weight swap
        if slack > 0 {
            let mut best_inc: Option<(usize, usize, i32)> = None;
            for &rm in &used_worst {
                if !state.selected_bit[rm] { continue; }
                let w_rm = state.ch.weights[rm];
                for &cand in &unused_top {
                    if state.selected_bit[cand] { continue; }
                    let w_c = state.ch.weights[cand];
                    if w_c <= w_rm { continue; }
                    let dw = w_c - w_rm;
                    if dw as usize > DIFF_LIM { continue; }
                    if dw > slack { continue; }
                    let d = state.contrib[cand] - state.contrib[rm]
                        - state.ch.interaction_values[cand][rm];
                    if d > 0 && best_inc.map_or(true, |(_, _, bd)| d > bd) {
                        best_inc = Some((cand, rm, d));
                    }
                }
            }
            if let Some((cand, rm, _)) = best_inc { state.replace_item(rm, cand); continue; }
        }

        break;
    }
}

// === Round0 Construction ===

fn build_round0_solution(state: &mut State, core_half: usize) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;

    // Compute total interaction potential for each item
    let mut scores: Vec<i64> = Vec::with_capacity(n);
    for i in 0..n {
        let mut s: i64 = state.ch.values[i] as i64;
        for j in 0..n {
            s += state.ch.interaction_values[i][j] as i64;
        }
        scores.push(s);
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        let sa = scores[a] * (state.ch.weights[b] as i64).max(1);
        let sb = scores[b] * (state.ch.weights[a] as i64).max(1);
        sb.cmp(&sa)
    });

    for &i in &order {
        if state.total_weight + state.ch.weights[i] <= cap {
            state.add_item(i);
        }
    }

    rebuild_windows(state, core_half);
}

// === Perturbation ===

fn perturb(state: &mut State, strength: usize, stall: usize, strategy: usize) {
    let selected = state.selected_items();
    if selected.is_empty() { return; }

    let mut removal_candidates: Vec<(usize, i64)>;

    match strategy % 6 {
        0 => {
            // Remove lowest contribution
            removal_candidates = selected.iter()
                .map(|&i| (i, state.contrib[i] as i64))
                .collect();
            removal_candidates.sort_unstable_by_key(|&(_, c)| c);
        }
        1 => {
            // Remove heaviest
            removal_candidates = selected.iter()
                .map(|&i| (i, -(state.ch.weights[i] as i64)))
                .collect();
            removal_candidates.sort_unstable_by_key(|&(_, w)| w);
        }
        2 => {
            // Remove lowest active synergy
            removal_candidates = selected.iter()
                .map(|&i| {
                    let synergy = state.contrib[i] - state.ch.values[i] as i32;
                    (i, synergy as i64)
                })
                .collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        }
        3 => {
            // Remove lowest density (contrib/weight)
            removal_candidates = selected.iter().map(|&i| {
                let w = (state.ch.weights[i] as i64).max(1);
                let score = (state.contrib[i] as i64 * 1000) / w;
                (i, score)
            }).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        }
        4 => {
            // Remove items with high weight and low density
            removal_candidates = selected.iter().map(|&i| {
                let w = (state.ch.weights[i] as i64).max(1);
                let density = (state.contrib[i] as i64 * 100) / w;
                let score = w * 10 - density;
                (i, -score)
            }).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        }
        _ => {
            // Remove items with worst contribution per interaction degree
            removal_candidates = selected.iter().map(|&i| {
                let mut degree: i32 = 0;
                for &j in &selected {
                    if j != i && state.ch.interaction_values[i][j] > 0 {
                        degree += 1;
                    }
                }
                let score = if degree > 0 {
                    (state.contrib[i] as i64 * 100) / (degree as i64)
                } else {
                    state.contrib[i] as i64 * 100
                };
                (i, score)
            }).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        }
    }

    let base_remove = (selected.len() / 10).max(1);
    let adaptive_mult = 1 + (stall / 2);
    let n_remove = (base_remove * adaptive_mult).min(strength).min(selected.len() / 3);

    for j in 0..n_remove.min(removal_candidates.len()) {
        state.remove_item(removal_candidates[j].0);
    }
}

fn greedy_reconstruct(state: &mut State, strategy: usize) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;

    let mut candidates: Vec<usize> = (0..n)
        .filter(|&i| !state.selected_bit[i])
        .collect();

    match strategy % 6 {
        0 => {
            candidates.sort_unstable_by_key(|&i| -state.contrib[i]);
        }
        1 => {
            candidates.sort_unstable_by(|&a, &b| {
                state.ch.weights[a].cmp(&state.ch.weights[b])
                    .then(state.contrib[b].cmp(&state.contrib[a]))
            });
        }
        2 => {
            candidates.sort_unstable_by_key(|&i| {
                let n_check = n.min(100);
                let syn: i64 = state.ch.interaction_values[i].iter()
                    .take(n_check)
                    .map(|&v| v as i64)
                    .sum();
                -(syn + state.contrib[i] as i64 / 10)
            });
        }
        3 | 4 => {
            candidates.sort_unstable_by_key(|&i| {
                let w = (state.ch.weights[i] as i64).max(1);
                -((state.contrib[i] as i64 * 100) / w)
            });
        }
        _ => {
            candidates.sort_unstable_by_key(|&i| {
                let w = (state.ch.weights[i] as i64).max(1);
                let c = state.contrib[i] as i64;
                -(c * w * w / 100)
            });
        }
    }

    for &i in &candidates {
        let w = state.ch.weights[i];
        if state.total_weight + w <= cap {
            state.add_item(i);
        }
    }
}

// === Hybrid Crossover ===

fn crossover_intersection<'a>(
    ch: &'a Challenge,
    s1: &State,
    s2: &State,
    params: &Params,
    rng: &mut Rng,
) -> State<'a> {
    let n = ch.num_items;
    let mut hyb = State::new_empty(ch);

    // Start with items in both solutions
    for i in 0..n {
        if s1.selected_bit[i] && s2.selected_bit[i]
            && hyb.total_weight + ch.weights[i] <= ch.max_weight
        {
            hyb.add_item(i);
        }
    }

    // Fill remaining capacity
    construct_forward(&mut hyb, 4, rng);
    rebuild_windows(&mut hyb, params.core_half);
    dp_refinement(&mut hyb);
    rebuild_windows(&mut hyb, params.core_half);
    extended_vnd(&mut hyb, params);
    polish(&mut hyb);
    hyb
}

fn crossover_union<'a>(
    ch: &'a Challenge,
    s1: &State,
    s2: &State,
    params: &Params,
    rng: &mut Rng,
) -> State<'a> {
    let n = ch.num_items;
    let mut hyb = State::new_empty(ch);

    // Start with all items from either solution
    for i in 0..n {
        if s1.selected_bit[i] || s2.selected_bit[i] {
            hyb.add_item(i);
        }
    }

    // Remove excess weight
    while hyb.total_weight > ch.max_weight {
        let mut worst: Option<usize> = None;
        let mut worst_score = i64::MAX;
        for i in 0..n {
            if !hyb.selected_bit[i] { continue; }
            let c = hyb.contrib[i] as i64;
            let w = (ch.weights[i] as i64).max(1);
            let s = (c * 1000) / w;
            if s < worst_score { worst_score = s; worst = Some(i); }
        }
        if let Some(wi) = worst { hyb.remove_item(wi); } else { break; }
    }

    construct_forward(&mut hyb, 4, rng);
    rebuild_windows(&mut hyb, params.core_half);
    dp_refinement(&mut hyb);
    rebuild_windows(&mut hyb, params.core_half);
    extended_vnd(&mut hyb, params);
    polish(&mut hyb);
    hyb
}

// === Main Solver ===

fn run_solver(challenge: &Challenge, params: &Params) -> Solution {
    let n = challenge.num_items;
    let mut rng = Rng::from_seed(&challenge.seed);

    // Multi-start construction
    let mut best: Option<State> = None;
    let mut second: Option<State> = None;

    for sid in 0..params.n_starts {
        let mut st = State::new_empty(challenge);

        match sid {
            0 => build_initial_solution(&mut st, params.core_half),
            1 => build_round0_solution(&mut st, params.core_half),
            2 => { construct_forward(&mut st, 2, &mut rng); rebuild_windows(&mut st, params.core_half); }
            3 => { construct_forward(&mut st, 3, &mut rng); rebuild_windows(&mut st, params.core_half); }
            _ => {
                construct_forward(&mut st, 5, &mut rng);
                rebuild_windows(&mut st, params.core_half);
            }
        }

        if n <= 2500 {
            dp_refinement(&mut st);
            rebuild_windows(&mut st, params.core_half);
        }
        extended_vnd(&mut st, params);
        polish(&mut st);

        if best.as_ref().map_or(true, |b| st.total_value > b.total_value) {
            second = best;
            best = Some(st);
        } else if second.as_ref().map_or(true, |b| st.total_value > b.total_value) {
            second = Some(st);
        }
    }

    // Hybrid crossover
    if n <= 1500 && best.is_some() && second.is_some() {
        let base_val = best.as_ref().unwrap().total_value;

        // Intersection crossover
        let hyb_int = crossover_intersection(
            challenge,
            best.as_ref().unwrap(),
            second.as_ref().unwrap(),
            params,
            &mut rng,
        );
        if hyb_int.total_value > base_val {
            second = best;
            best = Some(hyb_int);
        }

        // Union crossover (only if solutions differ enough)
        let (inter_cnt, union_cnt) = {
            let b1 = best.as_ref().unwrap();
            let b2 = second.as_ref().unwrap();
            let mut ic = 0usize;
            let mut uc = 0usize;
            for i in 0..n {
                if b1.selected_bit[i] || b2.selected_bit[i] { uc += 1; }
                if b1.selected_bit[i] && b2.selected_bit[i] { ic += 1; }
            }
            (ic, uc)
        };

        if union_cnt > 0 && (inter_cnt * 100) / union_cnt <= 85 {
            let hyb_union = crossover_union(
                challenge,
                best.as_ref().unwrap(),
                second.as_ref().unwrap(),
                params,
                &mut rng,
            );
            if hyb_union.total_value > best.as_ref().unwrap().total_value {
                best = Some(hyb_union);
            }
        }
    }

    let mut state = best.unwrap();

    // ILS perturbation loop
    let mut best_sel = state.selected_items();
    let mut best_val = state.total_value;
    let mut stall_count = 0usize;

    for round in 0..params.max_rounds {
        state.save_snapshot();
        let prev_val = state.total_value;
        let prev_weight = state.total_weight;

        // DP refinement (not every round for large n)
        let apply_dp = if n >= 4000 {
            round < 4 || (round % 3 == 0 && stall_count < 3)
        } else if n >= 2000 {
            round % 2 == 0 && stall_count < 4
        } else if n >= 1000 {
            stall_count < 5
        } else {
            true
        };

        if apply_dp {
            if n <= 2500 && (stall_count > 0 || (round & 3) == 0) {
                rebuild_windows(&mut state, params.core_half);
            }
            dp_refinement(&mut state);
        }
        extended_vnd(&mut state, &params);
        polish(&mut state);

        if state.total_value > best_val {
            best_val = state.total_value;
            best_sel = state.selected_items();
            stall_count = 0;
        }

        if state.total_value <= prev_val {
            state.restore_snapshot(prev_val, prev_weight);

            if round >= 7 && stall_count >= 6 {
                break;
            }
            if round >= params.max_rounds - 1 {
                break;
            }
            stall_count += 1;

            let strategy = round % 6;
            let strength = params.perturb_base + round / 2;
            perturb(&mut state, strength, stall_count, strategy);
            greedy_reconstruct(&mut state, strategy);
            if n <= 2500 {
                rebuild_windows(&mut state, params.core_half);
            }
            extended_vnd(&mut state, &params);
            polish(&mut state);

            if state.total_value > best_val {
                best_val = state.total_value;
                best_sel = state.selected_items();
                stall_count = 0;
            }
        }
    }

    Solution { items: best_sel }
}

// === Entry Point ===

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let params = Params::for_challenge(challenge);
    let solution = run_solver(challenge, &params);
    let _ = save_solution(&solution);
    Ok(())
}

pub fn help() {
    println!("Quadratic Knapsack - Extended VND with multi-item neighborhoods");
    println!();
    println!("Improvements over standard VND:");
    println!("  - 1-for-2 swaps: remove 1 item, add 2 (finds improvements 1-for-1 misses)");
    println!("  - 2-for-1 swaps: remove 2 items, add 1 (escape from dense local optima)");
    println!("  - Pair adds: add 2 items simultaneously if synergistic");
    println!("  - Multi-start with hybrid crossover");
    println!("  - Adaptive ILS perturbation with 6 strategies");
}
