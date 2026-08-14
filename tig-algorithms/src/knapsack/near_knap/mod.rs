use anyhow::{Result};
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Params {
    pub n_perturbation_rounds: usize,
    pub perturbation_strength_base: usize,
}

impl Params {
    pub fn initialize(h: &Option<Map<String, Value>>) -> Self {
        let mut p = Self {
            n_perturbation_rounds: 15,
            perturbation_strength_base: 3,
        };
        if let Some(m) = h {
            if let Some(v) = m.get("n_perturbation_rounds").and_then(|v| v.as_u64()) { p.n_perturbation_rounds = v as usize; }
            if let Some(v) = m.get("perturbation_strength_base").and_then(|v| v.as_u64()) { p.perturbation_strength_base = v as usize; }
        }
        p
    }
}

const DIFF_LIM: usize = 4;
const CORE_HALF: usize = 25;
const N_IT_CONSTRUCT: usize = 2;

#[inline]
fn core_half_for(n: usize) -> usize {
    if n <= 600 { 60 } else if n <= 1200 { 40 } else { CORE_HALF }
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
}

fn set_windows_from_density(
    state: &mut State,
    by_density: &[usize],
    idx_first_rejected: usize,
    idx_last_inserted: usize,
) {
    let n = state.ch.num_items;
    let core_half = core_half_for(n);

    let mut left = idx_first_rejected.saturating_sub(core_half + 1);
    let right = (idx_last_inserted + core_half + 1).min(n);
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

fn rebuild_windows(state: &mut State) {
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

fn construct_forward_incremental(state: &mut State, mode: usize, rng: &mut Rng) {
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

pub struct State<'a> {
    pub ch: &'a Challenge,
    pub selected_bit: Vec<bool>,
    pub contrib: Vec<i32>,
    pub total_value: i64,
    pub total_weight: u32,
    pub window_locked: Vec<usize>,
    pub window_core:   Vec<usize>,
    pub window_rejected: Vec<usize>,
    pub core_bins: Vec<(u32, Vec<usize>)>,    
    pub usage: Vec<u16>,
    pub dp_cache: Vec<i64>,
    pub choose_cache: Vec<u8>,    
    pub snap_bits: Vec<bool>,
    pub snap_contrib: Vec<i32>,
}

impl<'a> State<'a> {

    pub fn new_empty(ch: &'a Challenge) -> Self {
        let n = ch.num_items;
        let mut contrib = vec![0i32; n];
        for i in 0..n { contrib[i] = ch.values[i] as i32; }
        Self {
            ch,
            selected_bit: vec![false; n],
            contrib,
            total_value: 0,
            total_weight: 0,
            window_locked: Vec::new(),
            window_core:   Vec::new(),
            window_rejected:   Vec::new(),
            core_bins: Vec::new(),
            usage: vec![0u16; n],
            dp_cache: Vec::new(),
            choose_cache: Vec::new(),
            snap_bits: vec![false; n],
            snap_contrib: vec![0i32; n],
        }
    }

    pub fn selected_items(&self) -> Vec<usize> {
        (0..self.ch.num_items).filter(|&i| self.selected_bit[i]).collect()
    }

    #[inline(always)] pub fn slack(&self)    -> u32 { self.ch.max_weight - self.total_weight }

    #[inline(always)]
    pub fn add_item(&mut self, i: usize) {
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
    pub fn remove_item(&mut self, j: usize) {
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
    pub fn replace_item(&mut self, rm: usize, cand: usize) {
        self.remove_item(rm);
        self.add_item(cand);
    }

    pub fn restore_snapshot(
        &mut self,
        snap_value: i64,
        snap_weight: u32,
    ) {
        self.selected_bit.clone_from(&self.snap_bits);
        self.contrib.clone_from(&self.snap_contrib);
        self.total_value = snap_value;
        self.total_weight = snap_weight;
    }
}

pub fn build_initial_solution(state: &mut State) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;

    for i in 0..n {
        state.add_item(i);
    }
    
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

    let mut idx_last_inserted = 0 ;
    let mut idx_first_rejected = n ;
    let mut by_density: Vec<usize> = (0..n).collect();

    for _ in 0..=N_IT_CONSTRUCT {
        idx_last_inserted = 0 ;
        idx_first_rejected = n ;
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
        for &i in &target_sel { in_target[i] = true; }
        let mut to_remove: Vec<usize> = Vec::new();
        let mut to_add: Vec<usize> = Vec::new();
        for i in 0..n {
            if state.selected_bit[i] && !in_target[i] { to_remove.push(i); }
            if !state.selected_bit[i] && in_target[i] { to_add.push(i); }
        }

        if to_remove.is_empty() && to_add.is_empty() {
            break;
        }

        for &r in &to_remove { state.remove_item(r); }
        for &a in &to_add {
            state.add_item(a);
        }
    }

    set_windows_from_density(state, &by_density, idx_first_rejected, idx_last_inserted);
}

fn integer_core_target(
    ch: &Challenge,
    contrib: &[i32],
    locked: &[usize],
    core: &[usize],
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
    for &it in core {
        let wt = ch.weights[it] as usize;
        total_core_weight += wt;
        if contrib[it] > 0 {
            total_pos_weight += wt;
            if total_pos_weight > rem_cap {
                all_pos_fit = false;
            }
        }
    }

    if rem_cap == 0 {
        let mut selected: Vec<usize> = locked.to_vec();
        for &it in core {
            if ch.weights[it] == 0 && contrib[it] > 0 {
                selected.push(it);
            }
        }
        selected.sort_unstable();
        return selected;
    }

    if all_pos_fit {
        let mut selected: Vec<usize> = locked.to_vec();
        for &it in core {
            if contrib[it] > 0 {
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
    
    let init_val = i64::MIN / 4;
    for val in &mut dp_cache[0..dp_size] {
        *val = init_val;
    }
    dp_cache[0] = 0;
    
    choose_cache[0..choose_size].fill(0);
    
    let mut w_hi: usize = 0;

    for (t, &it) in core.iter().enumerate() {
        let wt = ch.weights[it] as usize;
        if wt > myw { continue; }
        let val = contrib[it] as i64;
        let new_hi = (w_hi + wt).min(myw);
        for w in (wt..=new_hi).rev() {
            let cand = dp_cache[w - wt] + val;
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
    let target = integer_core_target(
        state.ch,
        &state.contrib,
        &state.window_locked,
        &state.window_core,
        &mut state.dp_cache,
        &mut state.choose_cache,
    );
    apply_dp_target_via_ops(state, &target);
}

fn apply_best_add_windowed(state: &mut State) -> bool {
    let slack = state.slack();
    if slack == 0 { return false; }
    let mut best: Option<(usize, i32)> = None;    
    
    for (bw, items) in &state.core_bins {
        if *bw > slack { break; }
        for &cand in items {
            if state.selected_bit[cand] { continue; }
            let delta = state.contrib[cand];
            if delta > 0 && best.map_or(true, |(_, bd)| delta > bd) {
                best = Some((cand, delta));
            }
        }
    }    
    
    if best.is_none() {
        for &cand in &state.window_rejected {
            if state.selected_bit[cand] { continue; }
            let w = state.ch.weights[cand];
            if w <= slack {
                let delta = state.contrib[cand];
                if delta > 0 && best.map_or(true, |(_, bd)| delta > bd) {
                    best = Some((cand, delta));
                }
            }
        }
    }
    
    if let Some((cand, _)) = best { state.add_item(cand); true }
    else { false }
}

#[inline]
fn apply_best_swap11_equal_windowed_cached(state: &mut State, used: &[usize]) -> bool {
    let mut best: Option<(usize, usize, i32)> = None;
    for &rm in used {
        let w_rm = state.ch.weights[rm];
        if let Ok(idx) = state.core_bins.binary_search_by_key(&w_rm, |(w, _)| *w) {
            let items = &state.core_bins[idx].1;
            for &cand in items {
                if state.selected_bit[cand] { continue; }
                let delta = state.contrib[cand] - state.contrib[rm]
                    - state.ch.interaction_values[cand][rm];
                if delta > 0 && best.map_or(true, |(_, _, bd)| delta > bd) {
                    best = Some((cand, rm, delta));
                }
            }
        }
    }
    if let Some((cand, rm, _)) = best { state.replace_item(rm, cand); true }
    else { false }
}

#[inline]
fn apply_best_swap_diff_reduce_windowed_cached(state: &mut State, used: &[usize]) -> bool {
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
                let delta = state.contrib[cand] - state.contrib[rm]
                    - state.ch.interaction_values[cand][rm];
                if delta > 0 && best.map_or(true, |(_, _, bd)| delta > bd) {
                    best = Some((cand, rm, delta));
                }
            }
        }
    }
    if let Some((cand, rm, _)) = best { state.replace_item(rm, cand); true }
    else { false }
}

#[inline]
fn apply_best_swap_diff_increase_windowed_cached(state: &mut State, used: &[usize]) -> bool {
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
                let delta = state.contrib[cand] - state.contrib[rm]
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
    if let Some((cand, rm, _)) = best { state.replace_item(rm, cand); true }
    else { false }
}

pub fn local_search_vnd(state: &mut State) {
    let mut iterations = 0;
    let n = state.ch.num_items;    
    let max_iterations =
        if n >= 4500 { 220 }
        else if n >= 3000 { 320 }
        else if n >= 1000 { 350 }
        else { 80 };
    let mut used: Vec<usize> = Vec::new();
    
    loop {
        iterations += 1;
        if iterations > max_iterations { break; }
        
        if apply_best_add_windowed(state) { continue; }
        
        used.clear();
        for &i in &state.window_core {
            if state.selected_bit[i] {
                used.push(i);
            }
        }
        
        if apply_best_swap_diff_reduce_windowed_cached(state, &used) { continue; }
        if apply_best_swap11_equal_windowed_cached(state, &used) { continue; }
        if apply_best_swap_diff_increase_windowed_cached(state, &used) { continue; }
        
        break;
    }
}

fn perturb_by_strategy(state: &mut State, strength: usize, stall_count: usize, strategy: usize) {
    let selected = state.selected_items();
    let mut removal_candidates: Vec<(usize, i32)>;
    
    match strategy {
        0 => {
            removal_candidates = selected.iter()
                .map(|&i| (i, state.contrib[i]))
                .collect();
            removal_candidates.sort_unstable_by_key(|&(_, c)| c);
        },
        1 => {
            removal_candidates = selected.iter()
                .map(|&i| (i, -(state.ch.weights[i] as i32)))
                .collect();
            removal_candidates.sort_unstable_by_key(|&(_, w)| w);
        },
        2 => {            
            removal_candidates = selected
                .iter()
                .map(|&i| {
                    let active_synergy = state.contrib[i] - state.ch.values[i] as i32;
                    (i, active_synergy)
                })
                .collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        },
        3 => {
            removal_candidates = selected.iter().map(|&i| {
                let score = if state.ch.weights[i] > 0 {
                    (state.contrib[i] as i64 * 1000) / (state.ch.weights[i] as i64)
                } else {
                    state.contrib[i] as i64 * 1000
                };
                (i, -(score as i32))
            }).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        },
        4 => {
            removal_candidates = selected.iter().map(|&i| {
                let density = if state.ch.weights[i] > 0 {
                    (state.contrib[i] as i64 * 100) / (state.ch.weights[i] as i64)
                } else {
                    i64::MAX
                };
                let score = state.ch.weights[i] as i64 - density;
                (i, -(score as i32))
            }).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        },
        _ => {
            removal_candidates = selected.iter().map(|&i| {
                let usage_penalty = state.usage[i] as i32;
                let score = state.contrib[i] - usage_penalty;
                (i, score)
            }).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        }
    }
    
    let base_remove = (selected.len() / 10).max(1);
    let adaptive_mult = 1 + (stall_count / 2);
    let n_remove = (base_remove * adaptive_mult).min(strength).min(selected.len() / 3);
    for j in 0..n_remove {
        if j < removal_candidates.len() {
            state.remove_item(removal_candidates[j].0);
        }
    }
}

fn greedy_reconstruct(state: &mut State, strategy: usize) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    
    let mut candidates: Vec<usize> = (0..n)
        .filter(|&i| !state.selected_bit[i])
        .collect();
    
    match strategy {
        0 => {            
            candidates.sort_unstable_by_key(|&i| -state.contrib[i]);
        },
        1 => {            
            candidates.sort_unstable_by(|&a, &b| {
                state.ch.weights[a].cmp(&state.ch.weights[b])
                    .then(state.contrib[b].cmp(&state.contrib[a]))
            });
        },
        2 => {
            candidates.sort_unstable_by_key(|&i| {
                let total_synergy: i64 = state.ch.interaction_values[i].iter()
                    .take(n.min(100))
                    .map(|&v| v as i64)
                    .sum();
                -(total_synergy + state.contrib[i] as i64 / 10)
            });
        },
        3 => {
            candidates.sort_unstable_by_key(|&i| {
                let w = state.ch.weights[i] as i64;
                if w > 0 {
                    let eff = (state.contrib[i] as i64 * 100) / w;
                    -eff
                } else {
                    i64::MIN
                }
            });
        },
        4 => {
            candidates.sort_unstable_by_key(|&i| {
                let w = state.ch.weights[i] as i64;
                let c = state.contrib[i] as i64;
                -(c * w * w / 100)
            });
        },
        _ => {
            candidates.sort_unstable_by_key(|&i| {
                let w = (state.ch.weights[i] as i64).max(1);
                let base = (state.contrib[i] as i64 * 10000) / (w * w);
                let penalty = (state.usage[i] as i64) * 50;
                -(base - penalty)
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

fn run_one_instance(challenge: &Challenge, params: &Params) -> Solution {
    let n = challenge.num_items;
    let mut rng = Rng::from_seed(&challenge.seed);

    let sample = n.min(96);
    let mut nz: u32 = 0;
    let mut tot: u32 = 0;
    for i in 0..sample {
        for j in 0..i {
            tot += 1;
            if challenge.interaction_values[i][j] != 0 { nz += 1; }
        }
    }
    let dens = if tot > 0 { (nz as f64) / (tot as f64) } else { 1.0 };
    let hard = dens < 0.10;

    let n_starts: usize = if n <= 600 {
        if hard { 4 } else { 3 }
    } else if n <= 1500 {
        if hard { 3 } else { 2 }
    } else {
        1
    };

    let mut best: Option<State> = None;
    let mut second: Option<State> = None;

    for sid in 0..n_starts {
        let mut st = State::new_empty(challenge);

        match sid {
            0 => build_initial_solution(&mut st), 
            1 => { construct_forward_incremental(&mut st, 1, &mut rng); rebuild_windows(&mut st); }
            2 => { construct_forward_incremental(&mut st, 2, &mut rng); rebuild_windows(&mut st); }
            _ => {
                let m = if hard { 5 } else { 4 };
                construct_forward_incremental(&mut st, m, &mut rng);
                rebuild_windows(&mut st);
            }
        }

        if n <= 2500 {
            dp_refinement(&mut st);
            rebuild_windows(&mut st);
        }
        local_search_vnd(&mut st);

        if best.as_ref().map_or(true, |b| st.total_value > b.total_value) {
            second = best;
            best = Some(st);
        } else if second.as_ref().map_or(true, |b| st.total_value > b.total_value) {
            second = Some(st);
        }
    }

    if n <= 1500 && best.is_some() && second.is_some() {
        let base_val = best.as_ref().unwrap().total_value;
        let mut best_new: Option<State> = None;
        let mut best_new_val = base_val;

        {
            let mut hyb = State::new_empty(challenge);
            {
                let b1 = best.as_ref().unwrap();
                let b2 = second.as_ref().unwrap();
                for i in 0..n {
                    if b1.selected_bit[i] && b2.selected_bit[i]
                        && hyb.total_weight + challenge.weights[i] <= challenge.max_weight
                    {
                        hyb.add_item(i);
                    }
                }
            }
            construct_forward_incremental(&mut hyb, 4, &mut rng);
            rebuild_windows(&mut hyb);
            dp_refinement(&mut hyb);
            rebuild_windows(&mut hyb);
            local_search_vnd(&mut hyb);

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
                if a || b { union_cnt += 1; }
                if a && b { inter_cnt += 1; }
            }
            (inter_cnt, union_cnt)
        };

        if union_cnt > 0 && (inter_cnt * 100) / union_cnt <= 85 {
            let mut hyb = State::new_empty(challenge);
            {
                let b1 = best.as_ref().unwrap();
                let b2 = second.as_ref().unwrap();
                for i in 0..n {
                    if b1.selected_bit[i] || b2.selected_bit[i] {
                        hyb.add_item(i);
                    }
                }
            }

            while hyb.total_weight > challenge.max_weight {
                let mut worst_item: Option<usize> = None;
                let mut worst_score: i64 = i64::MAX;
                for i in 0..n {
                    if !hyb.selected_bit[i] { continue; }
                    let c = hyb.contrib[i] as i64;
                    let w = challenge.weights[i] as i64;
                    let s = if w > 0 { (c * 1000) / w } else { c * 1000 };
                    if s < worst_score { worst_score = s; worst_item = Some(i); }
                }
                if let Some(wi) = worst_item { hyb.remove_item(wi); } else { break; }
            }

            construct_forward_incremental(&mut hyb, 4, &mut rng);
            rebuild_windows(&mut hyb);
            dp_refinement(&mut hyb);
            rebuild_windows(&mut hyb);
            local_search_vnd(&mut hyb);

            if hyb.total_value > best_new_val {
                best_new_val = hyb.total_value;
                best_new = Some(hyb);
            }
        }

        if let Some(s) = best_new { best = Some(s); }
    }

    let mut state = best.unwrap();

    let mut best_sel: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        if state.selected_bit[i] {
            best_sel.push(i);
        }
    }
    let mut best_val = state.total_value;

    let mut stall_count = 0;
    let mut max_rounds = params.n_perturbation_rounds;

    if n <= 600 && hard {
        max_rounds = max_rounds.saturating_add(3);
    }

    if n >= 4500 {
        max_rounds = max_rounds.min(8);
    } else if n >= 3000 {
        max_rounds = max_rounds.min(11);
    } else if n >= 2000 {
        max_rounds = max_rounds.min(13);
    }
    
    for perturbation_round in 0..max_rounds {
        let is_last_round = perturbation_round >= max_rounds - 1;
        
        state.snap_bits.clone_from(&state.selected_bit);
        state.snap_contrib.clone_from(&state.contrib);
        let prev_val    = state.total_value;
        let prev_weight = state.total_weight;
        
        let apply_dp = !is_last_round && if n >= 4000 {
            perturbation_round < 3 || (perturbation_round % 4 == 0 && stall_count < 2)
        } else if n >= 2000 {
            perturbation_round % 2 == 0 && stall_count < 4
        } else if n >= 1000 {
            stall_count < 5
        } else {
            true
        };
        if apply_dp {
            if n <= 2500 && (stall_count > 0 || (perturbation_round & 3) == 0) {
                rebuild_windows(&mut state);
            }
            dp_refinement(&mut state);
        }
        local_search_vnd(&mut state);
        
        if state.total_value > best_val {
            best_val = state.total_value;
            best_sel.clear();
            for i in 0..n {
                if state.selected_bit[i] {
                    if state.usage[i] < u16::MAX {
                        state.usage[i] += 1;
                    }
                    best_sel.push(i);
                }
            }
            stall_count = 0;
        }
        
        if state.total_value <= prev_val {
            state.restore_snapshot(prev_val, prev_weight);
            
            if perturbation_round >= 7 && stall_count >= 6 {
                break;
            }
            
            if perturbation_round >= max_rounds - 1 {
                break;
            }
            stall_count += 1;
            
            let strategy = perturbation_round % 6;
            let strength =
                params.perturbation_strength_base + (perturbation_round as usize) / 2;
            perturb_by_strategy(&mut state, strength, stall_count, strategy);
            greedy_reconstruct(&mut state, strategy);
            if n <= 2500 {
                rebuild_windows(&mut state);
            }
            local_search_vnd(&mut state);

            if state.total_value > best_val {
                best_val = state.total_value;
                best_sel.clear();
                for i in 0..n {
                    if state.selected_bit[i] {
                        if state.usage[i] < u16::MAX {
                            state.usage[i] += 1;
                        }
                        best_sel.push(i);
                    }
                }
                stall_count = 0;
            }
        }
    }

    Solution { items: best_sel }
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

#[allow(dead_code)]
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
    println!("Quadratic Knapsack Problem - Multi-Start ILS with Hybrid Basin Discovery");
    println!();
    println!("Hyperparameters:");
    println!("  n_perturbation_rounds (default: 15)");
    println!("    Number of ILS perturbation-reconstruction cycles.");
    println!("    Higher values explore more solutions but increase runtime.");
    println!("    Recommended range: 10-20 for quality, 5-10 for speed.");
    println!();
    println!("  perturbation_strength_base (default: 3)");
    println!("    Base number of items removed during perturbation.");
    println!("    Actual removal scales with solution size and stall count.");
    println!("    Higher values = stronger diversification, lower = more focused search.");
    println!("    Recommended range: 2-5.");
}
