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

    let mut by_weight: Vec<usize> = (0..n).collect();
    by_weight.sort_unstable_by(|&a, &b| state.ch.weights[a].cmp(&state.ch.weights[b]));

    let cap = state.ch.max_weight;
    for &i in &by_weight {
        let w = state.ch.weights[i];
        if state.total_weight + w <= cap {
            state.add_item(i);
        } else {
            break;
        }
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

    let mut left  = (idx_first_rejected - CORE_HALF - 1).max(0);
    let right = (idx_last_inserted  + CORE_HALF + 1).min(n);
    if left > right { left = right; }
    state.window_locked   = by_density[..left].to_vec();
    state.window_core     = by_density[left..right].to_vec();
    state.window_rejected = by_density[right..].to_vec();

    let mut bins: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for &i in &state.window_core {
        let w = state.ch.weights[i];
        bins.entry(w).or_default().push(i);
    }
    state.core_bins = bins.into_iter().collect();
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
    let max_iterations = if n >= 3000 { 500 } else if n >= 1000 { 350 } else { 80 };
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
    let mut state = State::new_empty(challenge);
    
    build_initial_solution(&mut state);
    local_search_vnd(&mut state);

    let n = challenge.num_items;

    let mut best_sel: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        if state.selected_bit[i] {
            best_sel.push(i);
        }
    }
    let mut best_val = state.total_value;

    let mut stall_count = 0;
    let max_rounds = params.n_perturbation_rounds;
    
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
    println!("No help information available.");
}
