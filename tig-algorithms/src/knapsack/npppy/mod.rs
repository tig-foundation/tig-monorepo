use anyhow::{Result};
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Params {
    pub n_perturbation_rounds: usize,
    pub perturbation_strength_base: usize,
    pub beta: f32,
}

impl Params {
    pub fn initialize(h: &Option<Map<String, Value>>) -> Self {
        let mut p = Self {
            n_perturbation_rounds: 15,
            perturbation_strength_base: 3,
            beta: 0.0,
        };
        if let Some(m) = h {
            if let Some(v) = m.get("n_perturbation_rounds").and_then(|v| v.as_u64()) { p.n_perturbation_rounds = v as usize; }
            if let Some(v) = m.get("perturbation_strength_base").and_then(|v| v.as_u64()) { p.perturbation_strength_base = v as usize; }
            if let Some(v) = m.get("beta").and_then(|v| v.as_f64()) { p.beta = v as f32; }
        }
        p
    }
}

const DIFF_LIM: usize = 4;
const CORE_HALF: usize = 25;
const N_IT_CONSTRUCT: usize = 2;


const SI_CORE_MARGIN: usize = 3;








fn compute_specialitems_fast(challenge: &Challenge, n_lambda: usize, interaction_sums: &[i64]) -> Vec<usize> {
    let n = challenge.num_items;



    let mut items_with_ratio: Vec<(usize, f32)> = (0..n)
        .map(|i| {
            let w = challenge.weights[i] as f32;
            let deg = interaction_sums[i] as f32;
            let ratio = if w > 0.0 { deg / w } else { deg * 1000.0 };
            (i, ratio)
        })
        .collect();

    items_with_ratio.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));


    let mut specialitems = vec![0usize; n];
    let bucket_size = (n + n_lambda - 1) / n_lambda.max(1);

    for (rank, &(item, _)) in items_with_ratio.iter().enumerate() {

        let bucket = (n - 1 - rank) / bucket_size.max(1);
        specialitems[item] = bucket + 1;
    }

    specialitems
}


fn group_by_specialitem(specialitems: &[usize], n: usize) -> Vec<Vec<usize>> {
    let max_si = specialitems.iter().copied().max().unwrap_or(0);
    let mut si_groups: Vec<Vec<usize>> = vec![Vec::new(); max_si + 2];
    for i in 0..n {
        si_groups[specialitems[i]].push(i);
    }
    si_groups
}





fn set_windows_from_specialitems(
    state: &mut State,
    specialitems: &[usize],
    threshold_si: usize,
) {
    let n = state.ch.num_items;


    let si_margin = if n <= 600 {
        SI_CORE_MARGIN + 2
    } else if n <= 1200 {
        SI_CORE_MARGIN + 1
    } else {
        SI_CORE_MARGIN
    };

    let low_si = threshold_si.saturating_sub(si_margin);
    let high_si = threshold_si + si_margin;

    state.window_locked.clear();
    state.window_core.clear();
    state.window_rejected.clear();

    for i in 0..n {
        let si = specialitems[i];
        if si > high_si {

            state.window_locked.push(i);
        } else if si < low_si {

            state.window_rejected.push(i);
        } else {

            state.window_core.push(i);
        }
    }


    let mut core_sorted: Vec<(u32, usize)> = state.window_core.iter()
        .map(|&i| (state.ch.weights[i], i))
        .collect();
    core_sorted.sort_unstable_by_key(|&(w, _)| w);
    let mut bins: Vec<(u32, Vec<usize>)> = Vec::new();
    for (w, item) in core_sorted {
        if bins.last().map_or(true, |(bw, _)| *bw != w) {
            bins.push((w, vec![item]));
        } else {
            bins.last_mut().unwrap().1.push(item);
        }
    }
    state.core_bins = bins;
}



fn find_specialitem_threshold(
    specialitems: &[usize],
    weights: &[u32],
    budget: u32,
) -> usize {
    let max_si = specialitems.iter().cloned().max().unwrap_or(0);
    if max_si == 0 {
        return 0;
    }



    let mut weight_at_or_above: Vec<u32> = vec![0; max_si + 2];
    for (i, &si) in specialitems.iter().enumerate() {
        weight_at_or_above[si] += weights[i];
    }


    for si in (0..max_si).rev() {
        weight_at_or_above[si] += weight_at_or_above[si + 1];
    }


    let mut best_threshold = 0;
    let mut best_diff = u32::MAX;

    for threshold in 0..=max_si + 1 {
        let total_weight = if threshold <= max_si {
            weight_at_or_above[threshold]
        } else {
            0
        };

        let diff = if total_weight <= budget {
            budget - total_weight
        } else {
            total_weight - budget
        };

        if diff < best_diff {
            best_diff = diff;
            best_threshold = threshold;
        }
    }

    best_threshold
}




#[inline]
fn beta_adjusted_contrib(
    state: &State,
    item: usize,
    beta: f32,
    total_interactions: &[i64],
) -> f32 {
    let base_contrib = state.contrib[item] as f32;

    if beta <= 0.0 {
        return base_contrib;
    }


    let value = state.ch.values[item] as f32;
    let selected_interactions = base_contrib - value;


    let total = total_interactions[item] as f32;






    base_contrib + beta * (2.0 * selected_interactions - total)
}




fn construct_from_specialitems(state: &mut State, beta: f32, specialitems: &[usize], interaction_sums: &[i64]) {
    let budget = state.ch.max_weight;
    let si_groups = group_by_specialitem(specialitems, state.ch.num_items);


    let mut cumulative_weight = 0u32;
    let mut include_up_to = 0usize;
    for (level, group) in si_groups.iter().enumerate() {
        let group_weight: u32 = group.iter().map(|&i| state.ch.weights[i]).sum();
        if cumulative_weight + group_weight <= budget {
            cumulative_weight += group_weight;
            include_up_to = level + 1;
        } else {
            break;
        }
    }


    for level in 0..include_up_to {
        for &item in &si_groups[level] {
            if !state.selected_bit[item] {
                state.add_item(item);
            }
        }
    }


    greedy_fill_with_beta(state, beta, interaction_sums);
}



#[inline]
fn compute_total_interactions(challenge: &Challenge) -> Vec<i64> {
    let n = challenge.num_items;
    (0..n)
        .map(|i| challenge.interaction_values[i].iter().map(|&v| v as i64).sum())
        .collect()
}



fn greedy_fill_with_beta(state: &mut State, beta: f32, total_interactions: &[i64]) {
    let n = state.ch.num_items;
    let budget = state.ch.max_weight;


    let mut remaining: Vec<(usize, i64)> = (0..n)
        .filter(|&i| !state.selected_bit[i])
        .map(|i| {
            let adj = beta_adjusted_contrib(state, i, beta, total_interactions);
            let w = state.ch.weights[i] as i64;

            let score = if w > 0 { (adj * 1000.0) as i64 / w } else { (adj * 1000.0) as i64 };
            (i, score)
        })
        .collect();


    remaining.sort_unstable_by_key(|&(_, score)| std::cmp::Reverse(score));


    for (item, _) in remaining {
        let w = state.ch.weights[item];
        if state.total_weight + w <= budget && state.contrib[item] > 0 {
            state.add_item(item);
        }
    }
}





fn construct_from_specialitems_right(state: &mut State, beta: f32, specialitems: &[usize], interaction_sums: &[i64]) {
    let n = state.ch.num_items;
    let budget = state.ch.max_weight;
    let si_groups = group_by_specialitem(specialitems, n);


    let mut cumulative_weight = 0u32;
    let mut over_level = si_groups.len();
    for (level, group) in si_groups.iter().enumerate() {
        let group_weight: u32 = group.iter().map(|&i| state.ch.weights[i]).sum();
        cumulative_weight += group_weight;
        if cumulative_weight > budget {
            over_level = level + 1;
            break;
        }
    }


    if cumulative_weight <= budget {
        construct_from_specialitems(state, beta, specialitems, interaction_sums);
        return;
    }


    for level in 0..over_level {
        for &item in &si_groups[level] {
            if !state.selected_bit[item] {
                state.add_item(item);
            }
        }
    }



    let mut selected_with_score: Vec<(usize, i64)> = (0..n)
        .filter(|&i| state.selected_bit[i])
        .map(|i| {
            let adj = beta_adjusted_contrib(state, i, beta, interaction_sums);
            let w = state.ch.weights[i] as i64;
            let score = if w > 0 { (adj * 1000.0) as i64 / w } else { (adj * 1000.0) as i64 };
            (i, score)
        })
        .collect();


    selected_with_score.sort_unstable_by_key(|&(_, score)| score);


    for (item, _) in selected_with_score {
        if state.total_weight <= budget {
            break;
        }
        if state.selected_bit[item] {
            state.remove_item(item);
        }
    }


    greedy_fill_with_beta(state, beta, interaction_sums);
}

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

#[allow(dead_code)]
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

    let mut core_sorted: Vec<(u32, usize)> = state.window_core.iter()
        .map(|&i| (state.ch.weights[i], i))
        .collect();
    core_sorted.sort_unstable_by_key(|&(w, _)| w);
    let mut bins: Vec<(u32, Vec<usize>)> = Vec::new();
    for (w, item) in core_sorted {
        if bins.last().map_or(true, |(bw, _)| *bw != w) {
            bins.push((w, vec![item]));
        } else {
            bins.last_mut().unwrap().1.push(item);
        }
    }
    state.core_bins = bins;
}

fn rebuild_windows(state: &mut State) {
    let n = state.ch.num_items;
    if n == 0 { return; }
    let cap = state.ch.max_weight;



    let n_lambda = 50.min(n / 2).max(10);
    let specialitems = compute_specialitems_from_contrib(state.ch, &state.contrib, n_lambda);


    let threshold = find_specialitem_threshold(&specialitems, &state.ch.weights, cap);


    set_windows_from_specialitems(state, &specialitems, threshold);
}




fn compute_specialitems_from_contrib(
    challenge: &Challenge,
    contrib: &[i32],
    n_lambda: usize,
) -> Vec<usize> {
    let n = challenge.num_items;



    let mut items_with_ratio: Vec<(usize, f64)> = (0..n)
        .map(|i| {
            let w = challenge.weights[i] as f64;
            let c = contrib[i] as f64;
            let ratio = if w > 0.0 { c / w } else { c * 1000.0 };
            (i, ratio)
        })
        .collect();

    items_with_ratio.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));


    let mut specialitems = vec![0usize; n];
    let bucket_size = (n + n_lambda - 1) / n_lambda.max(1);

    for (rank, &(item, _)) in items_with_ratio.iter().enumerate() {

        let bucket = (n - 1 - rank) / bucket_size.max(1);
        specialitems[item] = bucket + 1;
    }

    specialitems
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


    let n_lambda = 50.min(n / 2).max(10);
    let specialitems = compute_specialitems_from_contrib(state.ch, &state.contrib, n_lambda);
    let threshold = find_specialitem_threshold(&specialitems, &state.ch.weights, cap);
    set_windows_from_specialitems(state, &specialitems, threshold);
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
    let n = state.ch.num_items;
    let n_selected = state.selected_bit.iter().filter(|&&b| b).count();
    let mut removal_candidates: Vec<(usize, i32)>;

    match strategy {
        0 => {
            removal_candidates = (0..n).filter(|&i| state.selected_bit[i])
                .map(|i| (i, state.contrib[i]))
                .collect();
            removal_candidates.sort_unstable_by_key(|&(_, c)| c);
        },
        1 => {
            removal_candidates = (0..n).filter(|&i| state.selected_bit[i])
                .map(|i| (i, -(state.ch.weights[i] as i32)))
                .collect();
            removal_candidates.sort_unstable_by_key(|&(_, w)| w);
        },
        2 => {
            removal_candidates = (0..n).filter(|&i| state.selected_bit[i])
                .map(|i| {
                    let active_synergy = state.contrib[i] - state.ch.values[i] as i32;
                    (i, active_synergy)
                })
                .collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        },
        3 => {
            removal_candidates = (0..n).filter(|&i| state.selected_bit[i]).map(|i| {
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
            removal_candidates = (0..n).filter(|&i| state.selected_bit[i]).map(|i| {
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
            removal_candidates = (0..n).filter(|&i| state.selected_bit[i]).map(|i| {
                let usage_penalty = state.usage[i] as i32;
                let score = state.contrib[i] - usage_penalty;
                (i, score)
            }).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        }
    }

    let base_remove = (n_selected / 10).max(1);
    let adaptive_mult = 1 + (stall_count / 2);
    let n_remove = (base_remove * adaptive_mult).min(strength).min(n_selected / 3);
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


    let interaction_sums = compute_total_interactions(challenge);
    let n_lambda = 50.min(n / 2).max(10);
    let specialitems_fast = compute_specialitems_fast(challenge, n_lambda, &interaction_sums);


    let n_starts: usize = if n <= 600 {
        if hard { 5 } else { 4 }
    } else if n <= 1500 {
        if hard { 4 } else { 3 }
    } else {
        3
    };

    let mut best: Option<State> = None;
    let mut second: Option<State> = None;

    for sid in 0..n_starts {
        let mut st = State::new_empty(challenge);

        match sid {
            0 => build_initial_solution(&mut st),
            1 => {

                construct_from_specialitems(&mut st, params.beta, &specialitems_fast, &interaction_sums);
                rebuild_windows(&mut st);
            }
            2 => {

                construct_from_specialitems_right(&mut st, params.beta, &specialitems_fast, &interaction_sums);
                rebuild_windows(&mut st);
            }
            3 => { construct_forward_incremental(&mut st, 1, &mut rng); rebuild_windows(&mut st); }
            4 => { construct_forward_incremental(&mut st, 2, &mut rng); rebuild_windows(&mut st); }
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
    println!();
    println!("  beta (default: 0.15)");
    println!("    Penalty factor for interactions with unselected items.");
    println!("    Helps avoid selecting items with 'phantom' synergy to items that won't fit.");
    println!("    0.0 = disabled, higher values = stronger penalty.");
    println!("    Recommended range: 0.0-0.3 (tuned for team-formation instances).");
}