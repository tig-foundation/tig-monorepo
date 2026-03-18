// TIG's UI uses the pattern `tig_challenges::knapsack` to automatically detect your algorithm's challenge
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
    selected_bit: Vec<bool>,
    contrib: Vec<i32>,
    total_value: i64,
    total_weight: u32,
    dp_cache: Vec<i64>,
    choose_cache: Vec<u8>,
}

impl<'a> State<'a> {
    fn new_empty(ch: &'a Challenge) -> Self {
        let n = ch.num_items;
        let mut contrib = vec![0i32; n];
        for i in 0..n { contrib[i] = ch.values[i] as i32; }
        Self {
            ch,
            selected_bit: vec![false; n],
            contrib,
            total_value: 0,
            total_weight: 0,
            dp_cache: Vec::new(),
            choose_cache: Vec::new(),
        }
    }

    #[inline(always)] fn slack(&self) -> u32 { self.ch.max_weight - self.total_weight }

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

    fn selected_items(&self) -> Vec<usize> {
        (0..self.ch.num_items).filter(|&i| self.selected_bit[i]).collect()
    }

    fn clone_solution(&self) -> SolState {
        SolState {
            bits: self.selected_bit.clone(),
            contrib: self.contrib.clone(),
            value: self.total_value,
            weight: self.total_weight,
        }
    }

    fn restore_solution(&mut self, sol: &SolState) {
        self.selected_bit.clone_from(&sol.bits);
        self.contrib.clone_from(&sol.contrib);
        self.total_value = sol.value;
        self.total_weight = sol.weight;
    }
}

#[derive(Clone)]
struct SolState {
    bits: Vec<bool>,
    contrib: Vec<i32>,
    value: i64,
    weight: u32,
}

fn build_greedy_density(state: &mut State) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    for i in 0..n { state.add_item(i); }
    while state.total_weight > cap {
        let mut worst = 0;
        let mut worst_s = i64::MAX;
        for i in 0..n {
            if state.selected_bit[i] {
                let c = state.contrib[i] as i64;
                let w = (state.ch.weights[i] as i64).max(1);
                let s = (c * 1000) / w;
                if s < worst_s { worst_s = s; worst = i; }
            }
        }
        state.remove_item(worst);
    }
    for _ in 0..2 {
        let mut by_density: Vec<usize> = (0..n).collect();
        let contrib = &state.contrib;
        let weights = &state.ch.weights;
        by_density.sort_unstable_by(|&a, &b| {
            let na = contrib[a] as i64;
            let nb = contrib[b] as i64;
            let wa = weights[a] as i64;
            let wb = weights[b] as i64;
            (na * wb).cmp(&(nb * wa)).reverse()
        });
        let mut target = vec![false; n];
        let mut rem = cap;
        for &i in &by_density {
            if state.ch.weights[i] <= rem { target[i] = true; rem -= state.ch.weights[i]; }
        }
        let mut to_rm = Vec::new();
        let mut to_add = Vec::new();
        for i in 0..n {
            if state.selected_bit[i] && !target[i] { to_rm.push(i); }
            if !state.selected_bit[i] && target[i] { to_add.push(i); }
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
    order.sort_unstable_by_key(|&i| std::cmp::Reverse(state.ch.values[i]));
    for &i in &order {
        if state.total_weight + state.ch.weights[i] <= cap { state.add_item(i); }
    }
}

fn build_greedy_hub(state: &mut State) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let mut hub_scores: Vec<(usize, i64)> = (0..n).map(|i| {
        let s: i64 = state.ch.interaction_values[i].iter().map(|&v| v as i64).sum();
        (i, s)
    }).collect();
    hub_scores.sort_unstable_by_key(|&(_, s)| std::cmp::Reverse(s));
    for &(i, _) in &hub_scores {
        if state.total_weight + state.ch.weights[i] <= cap { state.add_item(i); }
    }
}

fn build_greedy_synergy_weight(state: &mut State) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let mut scores: Vec<(usize, i64)> = (0..n).map(|i| {
        let avg_syn: i64 = if n > 1 {
            state.ch.interaction_values[i].iter().map(|&v| v as i64).sum::<i64>() / (n as i64 - 1)
        } else { 0 };
        let w = (state.ch.weights[i] as i64).max(1);
        (i, (state.ch.values[i] as i64 + avg_syn) * 100 / w)
    }).collect();
    scores.sort_unstable_by_key(|&(_, s)| std::cmp::Reverse(s));
    for &(i, _) in &scores {
        if state.total_weight + state.ch.weights[i] <= cap { state.add_item(i); }
    }
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
            if state.ch.weights[i] > slack { continue; }
            let c = state.contrib[i] as i64;
            if c <= 0 { continue; }
            let w = (state.ch.weights[i] as i64).max(1);
            let mut s = match mode {
                2 => c,
                3 => (c * 1000) / w + (state.ch.weights[i] as i64) * 3,
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
        } else { best_i };
        if let Some(i) = pick { state.add_item(i); } else { break; }
    }
}

fn build_hub_pair_kth(state: &mut State, k: usize) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let mut pairs: Vec<(i32, usize, usize)> = Vec::new();
    for i in 0..n {
        for j in (i+1)..n {
            if state.ch.weights[i] + state.ch.weights[j] <= cap {
                pairs.push((state.ch.interaction_values[i][j], i, j));
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
            if state.selected_bit[i] { continue; }
            if state.ch.weights[i] > slack { continue; }
            let c = state.contrib[i] as i64;
            if c <= 0 { continue; }
            let w = (state.ch.weights[i] as i64).max(1);
            let s = (c * 1000) / w;
            if s > best_s { best_s = s; best_i = Some(i); }
        }
        if let Some(i) = best_i { state.add_item(i); } else { break; }
    }
}

fn dp_refinement_hp(state: &mut State, core_half: usize) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let contrib = &state.contrib;
    let weights = &state.ch.weights;

    let mut by_density: Vec<usize> = (0..n).collect();
    by_density.sort_unstable_by(|&a, &b| {
        let na = contrib[a] as i64;
        let nb = contrib[b] as i64;
        let wa = weights[a] as i64;
        let wb = weights[b] as i64;
        (na * wb).cmp(&(nb * wa)).reverse()
    });
    let mut idx_last_inserted = 0usize;
    let mut idx_first_rejected = n;
    let mut rem = cap;
    for (idx, &i) in by_density.iter().enumerate() {
        let w = weights[i];
        if w <= rem { rem -= w; idx_last_inserted = idx; }
        else if idx_first_rejected == n { idx_first_rejected = idx; }
    }

    let left = idx_first_rejected.saturating_sub(core_half + 1);
    let right = (idx_last_inserted + core_half + 1).min(n);
    let locked: Vec<usize> = by_density[..left].to_vec();
    let core: Vec<usize> = by_density[left..right].to_vec();

    let used_locked: u64 = locked.iter().map(|&i| weights[i] as u64).sum();
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

    let target_sel = if all_pos_fit {
        let mut sel: Vec<usize> = locked.clone();
        for &it in &core { if contrib[it] > 0 { sel.push(it); } }
        sel.sort_unstable();
        sel
    } else {
        let myw = rem_cap.min(total_core_weight);
        let dp_size = myw + 1;
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

        let mut sel: Vec<usize> = locked.clone();
        let mut w_star = (0..=myw).max_by_key(|&w| state.dp_cache[w]).unwrap_or(0);
        for t in (0..myk).rev() {
            let it = core[t];
            let wt = weights[it] as usize;
            if wt <= w_star && state.choose_cache[t * dp_size + w_star] == 1 {
                sel.push(it);
                w_star -= wt;
            }
        }
        sel.sort_unstable();
        sel
    };

    let mut to_rm = Vec::new();
    let mut to_add = Vec::new();
    let mut j = 0;
    let m = target_sel.len();
    for i in 0..n {
        let in_target = j < m && target_sel[j] == i;
        if in_target { j += 1; }
        if state.selected_bit[i] && !in_target { to_rm.push(i); }
        else if in_target && !state.selected_bit[i] { to_add.push(i); }
    }
    for &r in &to_rm { state.remove_item(r); }
    for &a in &to_add { state.add_item(a); }
}

fn apply_best_add(state: &mut State) -> bool {
    let slack = state.slack();
    if slack == 0 { return false; }
    let n = state.ch.num_items;
    let mut best_i: Option<usize> = None;
    let mut best_d: i32 = 0;
    for i in 0..n {
        if state.selected_bit[i] { continue; }
        if state.ch.weights[i] > slack { continue; }
        let d = state.contrib[i];
        if d > best_d { best_d = d; best_i = Some(i); }
    }
    if let Some(i) = best_i { state.add_item(i); true } else { false }
}

fn apply_best_swap_1_1(state: &mut State, selected: &[usize]) -> bool {
    let n = state.ch.num_items;
    let slack = state.slack();
    let mut best: Option<(usize, usize, i32)> = None;
    for &rm in selected {
        let w_rm = state.ch.weights[rm];
        let max_w = w_rm + slack;
        for cand in 0..n {
            if state.selected_bit[cand] { continue; }
            let wc = state.ch.weights[cand];
            if wc > max_w { continue; }
            let delta = state.contrib[cand] - state.contrib[rm]
                - state.ch.interaction_values[cand][rm];
            if delta > 0 && best.map_or(true, |(_, _, bd)| delta > bd) {
                best = Some((cand, rm, delta));
            }
        }
    }
    if let Some((cand, rm, _)) = best { state.replace_item(rm, cand); true } else { false }
}

fn apply_pair_add(state: &mut State) -> bool {
    let slack = state.slack();
    if slack < 2 { return false; }
    let n = state.ch.num_items;
    let unsel: Vec<usize> = (0..n).filter(|&i| !state.selected_bit[i] && state.ch.weights[i] < slack).collect();
    let m = unsel.len();
    if m < 2 { return false; }

    let mut best_delta: i64 = 0;
    let mut best_pair: Option<(usize, usize)> = None;
    for ai in 0..m {
        let a = unsel[ai];
        let wa = state.ch.weights[a];
        let ca = state.contrib[a] as i64;
        for bi in (ai+1)..m {
            let b = unsel[bi];
            if wa + state.ch.weights[b] > slack { continue; }
            let delta = ca + state.contrib[b] as i64 + state.ch.interaction_values[a][b] as i64;
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
    let sel: Vec<usize> = (0..n).filter(|&i| state.selected_bit[i]).collect();
    let unsel: Vec<usize> = (0..n).filter(|&i| !state.selected_bit[i]).collect();
    let cap = state.ch.max_weight;

    let mut best_delta: i64 = 0;
    let mut best_move: Option<(usize, usize, usize)> = None;

    for &rm in &sel {
        let w_rm = state.ch.weights[rm] as i64;
        let c_rm = state.contrib[rm] as i64;
        let budget = state.slack() as i64 + w_rm;

        for ui in 0..unsel.len() {
            let a1 = unsel[ui];
            let w_a1 = state.ch.weights[a1] as i64;
            if w_a1 >= budget { continue; }
            let c_a1 = state.contrib[a1] as i64 - state.ch.interaction_values[a1][rm] as i64;

            for uj in (ui+1)..unsel.len() {
                let a2 = unsel[uj];
                let w_a2 = state.ch.weights[a2] as i64;
                if w_a1 + w_a2 > budget { continue; }

                let c_a2 = state.contrib[a2] as i64 - state.ch.interaction_values[a2][rm] as i64;
                let syn = state.ch.interaction_values[a1][a2] as i64;
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
    let sel: Vec<usize> = (0..n).filter(|&i| state.selected_bit[i]).collect();
    let unsel: Vec<usize> = (0..n).filter(|&i| !state.selected_bit[i]).collect();
    let cap = state.ch.max_weight;

    let mut best_delta: i64 = 0;
    let mut best_move: Option<(usize, usize, usize)> = None;

    for &add in &unsel {
        let w_add = state.ch.weights[add] as i64;
        let c_add = state.contrib[add] as i64;

        for si in 0..sel.len() {
            let r1 = sel[si];
            let w_r1 = state.ch.weights[r1] as i64;
            let c_r1 = state.contrib[r1] as i64;
            let c_add_r1 = state.ch.interaction_values[add][r1] as i64;

            for sj in (si+1)..sel.len() {
                let r2 = sel[sj];
                let w_r2 = state.ch.weights[r2] as i64;
                let freed = w_r1 + w_r2;
                let new_w = state.total_weight as i64 - freed + w_add;
                if new_w > cap as i64 || new_w < 0 { continue; }

                let c_r2 = state.contrib[r2] as i64;
                let syn_r1_r2 = state.ch.interaction_values[r1][r2] as i64;
                let c_add_r2 = state.ch.interaction_values[add][r2] as i64;

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
        .filter(|&i| state.selected_bit[i])
        .map(|i| (i, state.contrib[i]))
        .collect();
    sel_ranked.sort_unstable_by_key(|&(_, c)| c);
    sel_ranked.truncate(k);

    let mut unsel_ranked: Vec<(usize, i32)> = (0..n)
        .filter(|&i| !state.selected_bit[i])
        .map(|i| (i, state.contrib[i]))
        .collect();
    unsel_ranked.sort_unstable_by_key(|&(_, c)| std::cmp::Reverse(c));
    unsel_ranked.truncate(k);

    let cap = state.ch.max_weight;
    let mut best_delta: i64 = 0;
    let mut best_move: Option<(usize, usize, usize, usize)> = None;

    for si in 0..sel_ranked.len() {
        let r1 = sel_ranked[si].0;
        let w_r1 = state.ch.weights[r1] as i64;
        let c_r1 = state.contrib[r1] as i64;
        for sj in (si+1)..sel_ranked.len() {
            let r2 = sel_ranked[sj].0;
            let w_r2 = state.ch.weights[r2] as i64;
            let c_r2 = state.contrib[r2] as i64;
            let freed_weight = w_r1 + w_r2;
            let removed_syn = state.ch.interaction_values[r1][r2] as i64;
            let lost = c_r1 + c_r2 - removed_syn;
            let budget = state.slack() as i64 + freed_weight;

            for ui in 0..unsel_ranked.len() {
                let a1 = unsel_ranked[ui].0;
                let w_a1 = state.ch.weights[a1] as i64;
                if w_a1 > budget { continue; }
                let c_a1 = state.contrib[a1] as i64
                    - state.ch.interaction_values[a1][r1] as i64
                    - state.ch.interaction_values[a1][r2] as i64;
                for uj in (ui+1)..unsel_ranked.len() {
                    let a2 = unsel_ranked[uj].0;
                    let w_a2 = state.ch.weights[a2] as i64;
                    if w_a1 + w_a2 > budget { continue; }
                    let c_a2 = state.contrib[a2] as i64
                        - state.ch.interaction_values[a2][r1] as i64
                        - state.ch.interaction_values[a2][r2] as i64;
                    let added_syn = state.ch.interaction_values[a1][a2] as i64;
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
    for _ in 0..80 {
        if apply_best_add(state) { continue; }
        selected_buf.clear();
        for i in 0..n { if state.selected_bit[i] { selected_buf.push(i); } }
        if apply_best_swap_1_1(state, &selected_buf) { continue; }
        break;
    }
}

fn local_search_vnd_medium(state: &mut State, k: usize) {
    let n = state.ch.num_items;
    let mut selected_buf: Vec<usize> = Vec::with_capacity(n);
    for _ in 0..120 {
        if apply_best_add(state) { continue; }
        selected_buf.clear();
        for i in 0..n { if state.selected_bit[i] { selected_buf.push(i); } }
        if apply_best_swap_1_1(state, &selected_buf) { continue; }
        if apply_pair_add(state) { continue; }
        if apply_swap_2_2_bounded(state, k) { continue; }
        break;
    }
}

fn ils_vnd(state: &mut State, hp: &Hparams) {
    match hp.ils_vnd_level {
        0 => local_search_vnd_fast(state),
        1 => local_search_vnd_medium(state, hp.bounded_2_2_k),
        _ => local_search_vnd_heavy(state),
    }
}

fn local_search_vnd_heavy(state: &mut State) {
    let n = state.ch.num_items;
    let mut selected_buf: Vec<usize> = Vec::with_capacity(n);
    for _ in 0..300 {
        if apply_best_add(state) { continue; }
        selected_buf.clear();
        for i in 0..n { if state.selected_bit[i] { selected_buf.push(i); } }
        if apply_best_swap_1_1(state, &selected_buf) { continue; }
        if apply_pair_add(state) { continue; }
        if apply_swap_2_2_bounded(state, 25) { continue; }
        if apply_chain_move(state) { continue; }
        if apply_reverse_chain(state) { continue; }
        break;
    }
}

fn simulated_annealing(state: &mut State, rng: &mut Rng, n_rounds: usize, n_iter: usize) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;

    let mut sel: Vec<usize> = Vec::with_capacity(n);
    let mut unsel: Vec<usize> = Vec::with_capacity(n);
    let mut pos_in_sel = vec![0usize; n];
    let mut pos_in_unsel = vec![0usize; n];
    for i in 0..n {
        if state.selected_bit[i] {
            pos_in_sel[i] = sel.len();
            sel.push(i);
        } else {
            pos_in_unsel[i] = unsel.len();
            unsel.push(i);
        }
    }
    if sel.is_empty() || unsel.is_empty() { return; }

    let mut best_snap = state.clone_solution();

    let mut deltas: Vec<f64> = Vec::new();
    for _ in 0..100 {
        let rm = sel[rng.next_usize(sel.len())];
        let add = unsel[rng.next_usize(unsel.len())];
        let d = state.contrib[add] as f64 - state.contrib[rm] as f64
            - state.ch.interaction_values[add][rm] as f64;
        if d < 0.0 { deltas.push(-d); }
    }
    if deltas.is_empty() { return; }
    deltas.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let p75 = deltas[deltas.len() * 3 / 4];
    let t0 = p75 / 0.693;
    if t0 < 1.0 { return; }

    let alpha = 0.95f64;
    let mut temp = t0;

    for _ in 0..n_rounds {
        for _ in 0..n_iter {
            if sel.is_empty() || unsel.is_empty() { continue; }

            let coin = rng.next_u32() % 10;
            if coin < 8 {
                let si = rng.next_usize(sel.len());
                let ui = rng.next_usize(unsel.len());
                let rm = sel[si];
                let add = unsel[ui];
                let w_new = state.total_weight - state.ch.weights[rm] + state.ch.weights[add];
                if w_new > cap { continue; }
                let delta = state.contrib[add] as i64 - state.contrib[rm] as i64
                    - state.ch.interaction_values[add][rm] as i64;
                if delta > 0 || rng.next_f64() < (-delta as f64 / temp).exp() {
                    state.replace_item(rm, add);
                    let last_sel = *sel.last().unwrap();
                    sel[si] = last_sel;
                    pos_in_sel[last_sel] = si;
                    sel.pop();
                    pos_in_sel[rm] = 0;

                    let last_unsel = *unsel.last().unwrap();
                    unsel[ui] = last_unsel;
                    pos_in_unsel[last_unsel] = ui;
                    unsel.pop();
                    pos_in_unsel[add] = 0;

                    pos_in_sel[add] = sel.len();
                    sel.push(add);
                    pos_in_unsel[rm] = unsel.len();
                    unsel.push(rm);
                }
            } else if coin == 8 {
                let slack = state.slack();
                if slack == 0 { continue; }
                let ui = rng.next_usize(unsel.len());
                let add = unsel[ui];
                if state.ch.weights[add] > slack { continue; }
                let delta = state.contrib[add] as i64;
                if delta > 0 || rng.next_f64() < (-delta as f64 / temp).exp() {
                    state.add_item(add);
                    let last_unsel = *unsel.last().unwrap();
                    unsel[ui] = last_unsel;
                    pos_in_unsel[last_unsel] = ui;
                    unsel.pop();
                    pos_in_sel[add] = sel.len();
                    sel.push(add);
                }
            } else {
                let si = rng.next_usize(sel.len());
                let rm = sel[si];
                let delta = -(state.contrib[rm] as i64);
                if rng.next_f64() < (-delta as f64 / temp).exp() {
                    state.remove_item(rm);
                    let last_sel = *sel.last().unwrap();
                    sel[si] = last_sel;
                    pos_in_sel[last_sel] = si;
                    sel.pop();
                    pos_in_unsel[rm] = unsel.len();
                    unsel.push(rm);
                }
            }

            if state.total_value > best_snap.value {
                best_snap = state.clone_solution();
            }
        }
        temp *= alpha;
    }

    if best_snap.value > state.total_value {
        state.restore_solution(&best_snap);
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
        if child_weight + ch.weights[i] <= ch.max_weight {
            child_bits[i] = true;
            child_weight += ch.weights[i];
        }
    }
    for &i in &exploratory {
        if rng.next_u32() % 2 == 0 && child_weight + ch.weights[i] <= ch.max_weight {
            child_bits[i] = true;
            child_weight += ch.weights[i];
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
            if weight + ch.weights[i] <= ch.max_weight {
                bits[i] = true;
                weight += ch.weights[i];
            }
        }
    }
    for i in 0..n {
        if bits[i] { continue; }
        if sol_a.bits[i] || sol_b.bits[i] {
            if rng.next_u32() % 2 == 0 && weight + ch.weights[i] <= ch.max_weight {
                bits[i] = true;
                weight += ch.weights[i];
            }
        }
    }
    bits
}

fn set_state_from_bits(state: &mut State, bits: &[bool]) {
    let n = state.ch.num_items;
    for i in (0..n).rev() {
        if state.selected_bit[i] { state.remove_item(i); }
    }
    for i in 0..n {
        if bits[i] { state.add_item(i); }
    }
}

fn build_windows(state: &State, k: usize) -> (Vec<usize>, Vec<usize>) {
    let n = state.ch.num_items;
    let mut unused_r: Vec<(usize, f64)> = Vec::with_capacity(n);
    let mut used_r: Vec<(usize, f64)> = Vec::with_capacity(n);
    for i in 0..n {
        let r = state.contrib[i] as f64 / (state.ch.weights[i] as f64).max(1.0);
        if state.selected_bit[i] { used_r.push((i, r)); } else { unused_r.push((i, r)); }
    }
    let ku = k.min(unused_r.len());
    if ku > 0 && ku < unused_r.len() {
        unused_r.select_nth_unstable_by(ku - 1, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }
    let ks = k.min(used_r.len());
    if ks > 0 && ks < used_r.len() {
        used_r.select_nth_unstable_by(ks - 1, |a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    }
    (unused_r[..ku].iter().map(|x| x.0).collect(), used_r[..ks].iter().map(|x| x.0).collect())
}

fn local_search_vnd_windowed(state: &mut State, window_k: usize) {
    for _ in 0..80 {
        let (best_unused, worst_used) = build_windows(state, window_k);
        let improved = false;

        let slack = state.slack();
        if slack > 0 {
            let mut ba: Option<(usize, i32)> = None;
            for &c in &best_unused {
                if state.ch.weights[c] > slack { continue; }
                let d = state.contrib[c];
                if d > 0 && ba.map_or(true, |(_, bd)| d > bd) { ba = Some((c, d)); }
            }
            if let Some((c, _)) = ba { state.add_item(c); continue; }
        }

        {
            let mut bs: Option<(usize, usize, i32)> = None;
            for &rm in &worst_used {
                let max_w = state.ch.weights[rm] + state.slack();
                for &c in &best_unused {
                    if state.ch.weights[c] > max_w { continue; }
                    let d = state.contrib[c] - state.contrib[rm] - state.ch.interaction_values[c][rm];
                    if d > 0 && bs.map_or(true, |(_, _, bd)| d > bd) { bs = Some((c, rm, d)); }
                }
            }
            if let Some((c, rm, _)) = bs { state.replace_item(rm, c); continue; }
        }

        let slack = state.slack();
        if slack >= 2 {
            let fits: Vec<usize> = best_unused.iter().copied().filter(|&i| state.ch.weights[i] < slack).collect();
            let m = fits.len();
            if m >= 2 {
                let mut bp: Option<(usize, usize, i64)> = None;
                for ai in 0..m {
                    let a = fits[ai];
                    let wa = state.ch.weights[a];
                    let ca = state.contrib[a] as i64;
                    for bi in (ai+1)..m {
                        let b = fits[bi];
                        if wa + state.ch.weights[b] > slack { continue; }
                        let d = ca + state.contrib[b] as i64 + state.ch.interaction_values[a][b] as i64;
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
    for _ in 0..120 {
        let (best_unused, worst_used) = build_windows(state, window_k);
        let improved = false;

        let slack = state.slack();
        if slack > 0 {
            let mut ba: Option<(usize, i32)> = None;
            for &c in &best_unused {
                if state.ch.weights[c] > slack { continue; }
                let d = state.contrib[c];
                if d > 0 && ba.map_or(true, |(_, bd)| d > bd) { ba = Some((c, d)); }
            }
            if let Some((c, _)) = ba { state.add_item(c); continue; }
        }

        {
            let mut bs: Option<(usize, usize, i32)> = None;
            for &rm in &worst_used {
                let max_w = state.ch.weights[rm] + state.slack();
                for &c in &best_unused {
                    if state.ch.weights[c] > max_w { continue; }
                    let d = state.contrib[c] - state.contrib[rm] - state.ch.interaction_values[c][rm];
                    if d > 0 && bs.map_or(true, |(_, _, bd)| d > bd) { bs = Some((c, rm, d)); }
                }
            }
            if let Some((c, rm, _)) = bs { state.replace_item(rm, c); continue; }
        }

        let slack = state.slack();
        if slack >= 2 {
            let fits: Vec<usize> = best_unused.iter().copied().filter(|&i| state.ch.weights[i] < slack).collect();
            let m = fits.len();
            if m >= 2 {
                let mut bp: Option<(usize, usize, i64)> = None;
                for ai in 0..m {
                    let a = fits[ai];
                    let wa = state.ch.weights[a];
                    let ca = state.contrib[a] as i64;
                    for bi in (ai+1)..m {
                        let b = fits[bi];
                        if wa + state.ch.weights[b] > slack { continue; }
                        let d = ca + state.contrib[b] as i64 + state.ch.interaction_values[a][b] as i64;
                        if d > 0 && bp.map_or(true, |(_, _, bd)| d > bd) { bp = Some((a, b, d)); }
                    }
                }
                if let Some((a, b, _)) = bp { state.add_item(a); state.add_item(b); continue; }
            }
        }

        // chain_move: remove 1 from worst_used, add 2 from best_unused (k=80)
        {
            let cap = state.ch.max_weight;
            let mut bd = 0i64;
            let mut bm: Option<(usize, usize, usize)> = None;
            let ks = worst_used.len().min(80);
            let ku = best_unused.len().min(80);
            for si in 0..ks {
                let rm = worst_used[si];
                let c_rm = state.contrib[rm] as i64;
                let w_rm = state.ch.weights[rm];
                let budget = state.slack() + w_rm;
                for ai in 0..ku {
                    let a1 = best_unused[ai];
                    let wa1 = state.ch.weights[a1];
                    if wa1 >= budget { continue; }
                    let ca1_eff = state.contrib[a1] as i64 - state.ch.interaction_values[a1][rm] as i64;
                    for bi in (ai+1)..ku {
                        let a2 = best_unused[bi];
                        let wa2 = state.ch.weights[a2];
                        if wa1 + wa2 > budget { continue; }
                        let ca2_eff = state.contrib[a2] as i64 - state.ch.interaction_values[a2][rm] as i64;
                        let syn = state.ch.interaction_values[a1][a2] as i64;
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

        // reverse_chain: remove 2 from worst_used, add 1 from best_unused (k=80)
        {
            let cap = state.ch.max_weight;
            let mut bd = 0i64;
            let mut bm: Option<(usize, usize, usize)> = None;
            let ku = best_unused.len().min(80);
            let ks = worst_used.len().min(80);
            for ui in 0..ku {
                let add = best_unused[ui];
                let c_add = state.contrib[add] as i64;
                let w_add = state.ch.weights[add];
                for si in 0..ks {
                    let r1 = worst_used[si];
                    let wr1 = state.ch.weights[r1];
                    let cr1 = state.contrib[r1] as i64;
                    let c_add_r1 = state.ch.interaction_values[add][r1] as i64;
                    for sj in (si+1)..ks {
                        let r2 = worst_used[sj];
                        let wr2 = state.ch.weights[r2];
                        let new_w = state.total_weight + w_add - wr1 - wr2;
                        if new_w > cap { continue; }
                        let cr2 = state.contrib[r2] as i64;
                        let syn_r1_r2 = state.ch.interaction_values[r1][r2] as i64;
                        let c_add_r2 = state.ch.interaction_values[add][r2] as i64;
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

        // swap_2_2: remove 2 from worst_used, add 2 from best_unused (k=25)
        {
            let cap = state.ch.max_weight;
            let mut bd = 0i64;
            let mut bm: Option<(usize, usize, usize, usize)> = None;
            let ks = worst_used.len().min(25);
            let ku = best_unused.len().min(25);
            for si in 0..ks {
                let r1 = worst_used[si];
                let wr1 = state.ch.weights[r1];
                let cr1 = state.contrib[r1] as i64;
                for sj in (si+1)..ks {
                    let r2 = worst_used[sj];
                    let wr2 = state.ch.weights[r2];
                    let cr2 = state.contrib[r2] as i64;
                    let syn_rm = state.ch.interaction_values[r1][r2] as i64;
                    let lost = cr1 + cr2 - syn_rm;
                    let budget = state.slack() + wr1 + wr2;
                    for ui in 0..ku {
                        let a1 = best_unused[ui];
                        let wa1 = state.ch.weights[a1];
                        if wa1 >= budget { continue; }
                        let ca1_eff = state.contrib[a1] as i64
                            - state.ch.interaction_values[a1][r1] as i64
                            - state.ch.interaction_values[a1][r2] as i64;
                        for uj in (ui+1)..ku {
                            let a2 = best_unused[uj];
                            let wa2 = state.ch.weights[a2];
                            if wa1 + wa2 > budget { continue; }
                            let ca2_eff = state.contrib[a2] as i64
                                - state.ch.interaction_values[a2][r1] as i64
                                - state.ch.interaction_values[a2][r2] as i64;
                            let syn_add = state.ch.interaction_values[a1][a2] as i64;
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

fn perturb_by_strategy(state: &mut State, strength: usize, stall_count: usize, strategy: usize, rng: &mut Rng, hp: &Hparams) {
    let selected = state.selected_items();
    if selected.is_empty() { return; }
    let mut removal_candidates: Vec<(usize, i64)>;

    match strategy {
        0 => {
            removal_candidates = selected.iter().map(|&i| (i, state.contrib[i] as i64)).collect();
            removal_candidates.sort_unstable_by_key(|&(_, c)| c);
        },
        1 => {
            removal_candidates = selected.iter().map(|&i| (i, -(state.ch.weights[i] as i64))).collect();
            removal_candidates.sort_unstable_by_key(|&(_, w)| w);
        },
        2 => {
            removal_candidates = selected.iter().map(|&i| {
                let syn = state.contrib[i] as i64 - state.ch.values[i] as i64;
                (i, syn)
            }).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        },
        3 => {
            removal_candidates = selected.iter().map(|&i| {
                let w = (state.ch.weights[i] as i64).max(1);
                (i, (state.contrib[i] as i64 * 1000) / w)
            }).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        },
        4 => {
            removal_candidates = selected.iter().map(|&i| {
                let w = (state.ch.weights[i] as i64).max(1);
                let density = (state.contrib[i] as i64 * 100) / w;
                (i, state.ch.weights[i] as i64 - density)
            }).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        },
        5 => {
            removal_candidates = selected.iter().map(|&i| {
                let w = (state.ch.weights[i] as i64).max(1);
                (i, (state.contrib[i] as i64 * 10000) / (w * w))
            }).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        },
        6 => {
            removal_candidates = selected.iter().map(|&i| (i, rng.next_u32() as i64)).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        },
        _ => {
            removal_candidates = selected.iter().map(|&i| (i, -(state.contrib[i] as i64))).collect();
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

fn greedy_reconstruct(state: &mut State, strategy: usize) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let mut candidates: Vec<usize> = (0..n).filter(|&i| !state.selected_bit[i]).collect();

    match strategy % 4 {
        0 => candidates.sort_unstable_by_key(|&i| -state.contrib[i]),
        1 => candidates.sort_unstable_by(|&a, &b| {
            state.ch.weights[a].cmp(&state.ch.weights[b])
                .then(state.contrib[b].cmp(&state.contrib[a]))
        }),
        2 => candidates.sort_unstable_by_key(|&i| {
            let syn: i64 = state.ch.interaction_values[i].iter()
                .take(n.min(100)).map(|&v| v as i64).sum();
            -(syn + state.contrib[i] as i64 / 10)
        }),
        _ => candidates.sort_unstable_by_key(|&i| {
            let w = (state.ch.weights[i] as i64).max(1);
            -((state.contrib[i] as i64 * 100) / w)
        }),
    }

    for &i in &candidates {
        if state.total_weight + state.ch.weights[i] <= cap { state.add_item(i); }
    }
}

struct TopNeighbors {
    friends: Vec<Vec<usize>>,
}
impl TopNeighbors {
    fn new(ch: &Challenge, k: usize) -> Self {
        let n = ch.num_items;
        let mut friends = vec![Vec::with_capacity(k); n];
        for i in 0..n {
            let mut row: Vec<(usize, i32)> = Vec::new();
            for j in 0..n {
                if i != j && ch.interaction_values[i][j] > 0 {
                    row.push((j, ch.interaction_values[i][j]));
                }
            }
            row.sort_unstable_by_key(|x| Reverse(x.1));
            friends[i] = row.into_iter().take(k).map(|x| x.0).collect();
        }
        Self { friends }
    }
}

fn local_search_vnd_tsn(state: &mut State, tsn: &TopNeighbors) {
    let n = state.ch.num_items;
    let wk: usize = if n > 3000 { 250 } else { 400 };
    for _ in 0..80 {
        let (best_unused, worst_used) = build_windows(state, wk);

        let slack = state.slack();
        if slack > 0 {
            let mut ba: Option<(usize, i32)> = None;
            for &c in &best_unused {
                if state.ch.weights[c] > slack { continue; }
                let d = state.contrib[c];
                if d > 0 && ba.map_or(true, |(_, bd)| d > bd) { ba = Some((c, d)); }
            }
            if let Some((c, _)) = ba { state.add_item(c); continue; }
        }

        {
            let mut bs: Option<(usize, usize, i32)> = None;
            for &rm in &worst_used {
                let max_w = state.ch.weights[rm] + state.slack();
                for &c in &best_unused {
                    if state.ch.weights[c] > max_w { continue; }
                    let d = state.contrib[c] - state.contrib[rm] - state.ch.interaction_values[c][rm];
                    if d > 0 && bs.map_or(true, |(_, _, bd)| d > bd) { bs = Some((c, rm, d)); }
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
                    let ca1 = state.contrib[a1] as i64;
                    if ca1 <= 0 && bd > 0 { break; }
                    let wa1 = state.ch.weights[a1] as i32;
                    if wa1 >= slack_i { continue; }
                    for &a2 in &tsn.friends[a1] {
                        if state.selected_bit[a2] || a1 == a2 { continue; }
                        if wa1 + (state.ch.weights[a2] as i32) <= slack_i {
                            let delta = ca1 + state.contrib[a2] as i64 + state.ch.interaction_values[a1][a2] as i64;
                            if delta > bd { bd = delta; bp = Some((a1, a2)); }
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
            for &rm in &worst_used {
                let c_rm = state.contrib[rm] as i64;
                let w_rm = state.ch.weights[rm];
                let budget = state.slack() + w_rm;
                for &a1 in &best_unused {
                    let ca1 = state.contrib[a1] as i64;
                    let wa1 = state.ch.weights[a1];
                    if wa1 >= budget { continue; }
                    let ca1_eff = ca1 - state.ch.interaction_values[a1][rm] as i64;
                    for &a2 in &tsn.friends[a1] {
                        if state.selected_bit[a2] || a1 == a2 { continue; }
                        let wa2 = state.ch.weights[a2];
                        if wa1 + wa2 > budget { continue; }
                        let delta = ca1_eff + state.contrib[a2] as i64
                            - state.ch.interaction_values[a2][rm] as i64
                            + state.ch.interaction_values[a1][a2] as i64 - c_rm;
                        if delta > bd && state.total_weight - w_rm + wa1 + wa2 <= cap {
                            bd = delta; bm = Some((rm, a1, a2));
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
            for &add in &best_unused {
                let c_add = state.contrib[add] as i64;
                let w_add = state.ch.weights[add];
                if c_add <= 0 && bd > 0 { break; }
                for &r1 in &worst_used {
                    let cr1 = state.contrib[r1] as i64;
                    let wr1 = state.ch.weights[r1];
                    let c_add_r1 = state.ch.interaction_values[add][r1] as i64;
                    for &r2 in &tsn.friends[r1] {
                        if !state.selected_bit[r2] || r1 == r2 { continue; }
                        let wr2 = state.ch.weights[r2];
                        if state.total_weight + w_add <= cap + wr1 + wr2 {
                            let delta = c_add - c_add_r1
                                - state.ch.interaction_values[add][r2] as i64
                                - cr1 - state.contrib[r2] as i64
                                + state.ch.interaction_values[r1][r2] as i64;
                            if delta > bd { bd = delta; bm = Some((r1, r2, add)); }
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
            for i in 0..ks {
                let r1 = worst_used[i];
                let cr1 = state.contrib[r1] as i64;
                let wr1 = state.ch.weights[r1];
                for &r2 in &tsn.friends[r1] {
                    if !state.selected_bit[r2] || r1 == r2 { continue; }
                    let cr2 = state.contrib[r2] as i64;
                    let wr2 = state.ch.weights[r2];
                    let lost = cr1 + cr2 - state.ch.interaction_values[r1][r2] as i64;
                    let budget = state.slack() + wr1 + wr2;
                    for u in 0..ku {
                        let a1 = best_unused[u];
                        let wa1 = state.ch.weights[a1];
                        if wa1 >= budget { continue; }
                        let ca1_eff = state.contrib[a1] as i64
                            - state.ch.interaction_values[a1][r1] as i64
                            - state.ch.interaction_values[a1][r2] as i64;
                        for &a2 in &tsn.friends[a1] {
                            if state.selected_bit[a2] || a1 == a2 { continue; }
                            let wa2 = state.ch.weights[a2];
                            if wa1 + wa2 <= budget {
                                let gained = ca1_eff + state.contrib[a2] as i64
                                    - state.ch.interaction_values[a2][r1] as i64
                                    - state.ch.interaction_values[a2][r2] as i64
                                    + state.ch.interaction_values[a1][a2] as i64;
                                if gained - lost > bd && state.total_weight + wa1 + wa2 <= cap + wr1 + wr2 {
                                    bd = gained - lost; bm = Some((r1, r2, a1, a2));
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
    freed += state.ch.weights[root];
    for &f in &tsn.friends[root] {
        if state.selected_bit[f] {
            state.remove_item(f);
            freed += state.ch.weights[f];
            if freed >= target { break; }
        }
    }
    let slack = state.slack();
    if slack > 0 {
        let unsel: Vec<usize> = (0..state.ch.num_items)
            .filter(|&i| !state.selected_bit[i] && state.ch.weights[i] <= slack)
            .collect();
        if !unsel.is_empty() { state.add_item(unsel[rng.next_usize(unsel.len())]); }
    }
}

struct Hparams {
    n_random_starts: usize,
    n_crossover_gen: usize,
    sa_rounds: usize,
    sa_iter: usize,
    n_sa_members: usize,
    ils_rounds: usize,
    ils_restart_interval: usize,
    perturb_base_frac: usize,
    perturb_max_frac: usize,
    ils_vnd_level: usize,
    bounded_2_2_k: usize,
    n_full_restarts: usize,
    use_hub_pair: bool,
    use_heavy_polish: bool,
    window_k: usize,
    core_half_dp: usize,
}

impl Hparams {
    fn for_size(n: usize, budget: u32) -> Self {
        if n <= 1200 {
            if budget <= 5 {
                Self {
                    n_random_starts: 4, n_crossover_gen: 12, sa_rounds: 0,
                    sa_iter: 0, n_sa_members: 0, ils_rounds: 100,
                    ils_restart_interval: 12, perturb_base_frac: 4,
                    perturb_max_frac: 5, ils_vnd_level: 0, bounded_2_2_k: 10,
                    n_full_restarts: 60, use_hub_pair: true,
                    use_heavy_polish: false, window_k: 300, core_half_dp: 60,
                }
            } else if budget <= 10 {
                Self {
                    n_random_starts: 4, n_crossover_gen: 12, sa_rounds: 0,
                    sa_iter: 0, n_sa_members: 0, ils_rounds: 150,
                    ils_restart_interval: 12, perturb_base_frac: 4,
                    perturb_max_frac: 5, ils_vnd_level: 0, bounded_2_2_k: 10,
                    n_full_restarts: 30, use_hub_pair: true,
                    use_heavy_polish: false, window_k: 300, core_half_dp: 60,
                }
            } else {
                Self {
                    n_random_starts: 4, n_crossover_gen: 12, sa_rounds: 0,
                    sa_iter: 0, n_sa_members: 0, ils_rounds: 200,
                    ils_restart_interval: 12, perturb_base_frac: 4,
                    perturb_max_frac: 5, ils_vnd_level: 0, bounded_2_2_k: 10,
                    n_full_restarts: 50, use_hub_pair: true,
                    use_heavy_polish: false, window_k: 300, core_half_dp: 60,
                }
            }
        } else {
            Self {
                n_random_starts: 5, n_crossover_gen: 4, sa_rounds: 0,
                sa_iter: 0, n_sa_members: 0, ils_rounds: 160,
                ils_restart_interval: 15, perturb_base_frac: 6,
                perturb_max_frac: 4, ils_vnd_level: 0, bounded_2_2_k: 0,
                n_full_restarts: 1, use_hub_pair: false,
                use_heavy_polish: false, window_k: 200, core_half_dp: 50,
            }
        }
    }

    fn from_map(h: &Option<Map<String, Value>>, n: usize, budget: u32) -> Self {
        let mut p = Self::for_size(n, budget);
        if let Some(m) = h {
            if let Some(v) = m.get("n_random_starts").and_then(|v| v.as_u64()) { p.n_random_starts = v as usize; }
            if let Some(v) = m.get("n_crossover_gen").and_then(|v| v.as_u64()) { p.n_crossover_gen = v as usize; }
            if let Some(v) = m.get("sa_rounds").and_then(|v| v.as_u64()) { p.sa_rounds = v as usize; }
            if let Some(v) = m.get("sa_iter").and_then(|v| v.as_u64()) { p.sa_iter = v as usize; }
            if let Some(v) = m.get("n_sa_members").and_then(|v| v.as_u64()) { p.n_sa_members = v as usize; }
            if let Some(v) = m.get("ils_rounds").and_then(|v| v.as_u64()) { p.ils_rounds = v as usize; }
            if let Some(v) = m.get("ils_restart_interval").and_then(|v| v.as_u64()) { p.ils_restart_interval = v as usize; }
            if let Some(v) = m.get("perturb_base_frac").and_then(|v| v.as_u64()) { p.perturb_base_frac = v as usize; }
            if let Some(v) = m.get("perturb_max_frac").and_then(|v| v.as_u64()) { p.perturb_max_frac = v as usize; }
            if let Some(v) = m.get("ils_vnd_level").and_then(|v| v.as_u64()) { p.ils_vnd_level = v as usize; }
            if let Some(v) = m.get("bounded_2_2_k").and_then(|v| v.as_u64()) { p.bounded_2_2_k = v as usize; }
            if let Some(v) = m.get("n_full_restarts").and_then(|v| v.as_u64()) { p.n_full_restarts = v as usize; }
            if let Some(v) = m.get("window_k").and_then(|v| v.as_u64()) { p.window_k = v as usize; }
            if let Some(v) = m.get("core_half_dp").and_then(|v| v.as_u64()) { p.core_half_dp = v as usize; }
        }
        p
    }
}


fn vnd_v2(state: &mut State, hp: &Hparams, tsn: Option<&TopNeighbors>) {
    if let Some(t) = tsn {
        local_search_vnd_tsn(state, t);
    } else if hp.window_k < state.ch.num_items {
        local_search_vnd_windowed(state, hp.window_k);
    } else {
        ils_vnd(state, hp);
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

fn run_one_instance(challenge: &Challenge, hp: &Hparams, rng_offset: usize) -> Solution {
    let n = challenge.num_items;
    let mut rng = Rng::from_seed(&challenge.seed);
    for _ in 0..rng_offset * 100 { rng.next_u32(); }
    let ch = hp.core_half_dp;

    let tsn_opt: Option<TopNeighbors> = if n > 1200 {
        Some(TopNeighbors::new(challenge, 12))
    } else { None };
    let tsn_ref = tsn_opt.as_ref();

    let mut population: Vec<SolState> = Vec::with_capacity(16);

    let n_greedy = if n <= 1200 { 4 } else { 3 };
    for variant in 0..n_greedy {
        let mut st = State::new_empty(challenge);
        match variant {
            0 => build_greedy_density(&mut st),
            1 => build_greedy_value(&mut st),
            2 => build_greedy_synergy_weight(&mut st),
            _ => build_greedy_hub(&mut st),
        }
        dp_refinement_hp(&mut st, ch);
        polish_v2(&mut st, hp, tsn_ref);
        population.push(st.clone_solution());
    }

    for mode in 4..(4 + hp.n_random_starts) {
        let mut st = State::new_empty(challenge);
        let m = if mode < 6 { mode } else { mode - 2 };
        construct_forward_incremental(&mut st, m, &mut rng);
        dp_refinement_hp(&mut st, ch);
        vnd_v2(&mut st, hp, tsn_ref);
        population.push(st.clone_solution());
    }

    if hp.use_hub_pair {
        for k in 0..4 {
            let mut st = State::new_empty(challenge);
            build_hub_pair_kth(&mut st, k);
            dp_refinement_hp(&mut st, ch);
            vnd_v2(&mut st, hp, tsn_ref);
            population.push(st.clone_solution());
        }
    }

    population.sort_unstable_by_key(|s| std::cmp::Reverse(s.value));
    population.truncate(8);

    let mut state = State::new_empty(challenge);
    for gen in 0..hp.n_crossover_gen {
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
            }
        }
        population.sort_unstable_by_key(|s| std::cmp::Reverse(s.value));
        population.truncate(8);
    }

    if hp.sa_rounds > 0 {
        for pi in 0..hp.n_sa_members.min(population.len()) {
            state.restore_solution(&population[pi]);
            simulated_annealing(&mut state, &mut rng, hp.sa_rounds, hp.sa_iter);
            vnd_v2(&mut state, hp, tsn_ref);
            let sol = state.clone_solution();
            if sol.value > population[pi].value { population.push(sol); }
        }
        population.sort_unstable_by_key(|s| std::cmp::Reverse(s.value));
        population.truncate(8);
    }

    state.restore_solution(&population[0]);
    let mut best_val = state.total_value;
    let mut best_sel: Vec<usize> = state.selected_items();

    let mut tabu_hashes: Vec<u64> = Vec::with_capacity(128);
    let zobrist_table: Vec<u64> = (0..n).map(|i| {
        let mut h: u64 = 0x517CC1B727220A95;
        h ^= (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
        h = h.rotate_left(17).wrapping_mul(0xBF58476D1CE4E5B9);
        h
    }).collect();
    let compute_hash = |bits: &[bool]| -> u64 {
        let mut h: u64 = 0;
        for i in 0..n { if bits[i] { h ^= zobrist_table[i]; } }
        h
    };
    tabu_hashes.push(compute_hash(&state.selected_bit));

    let mut stall_count = 0;
    for round in 0..hp.ils_rounds {
        let snap = state.clone_solution();

        dp_refinement_hp(&mut state, ch);
        vnd_v2(&mut state, hp, tsn_ref);

        if state.total_value > best_val {
            best_val = state.total_value;
            best_sel = state.selected_items();
            stall_count = 0;
        }

        if state.total_value <= snap.value {
            state.restore_solution(&snap);
            stall_count += 1;

            if hp.ils_restart_interval > 0 && stall_count > 0 && stall_count % hp.ils_restart_interval == 0 {
                let pi = (stall_count / hp.ils_restart_interval) % population.len();
                state.restore_solution(&population[pi]);
            }

            let use_bomb = tsn_ref.is_some() && round % 3 == 0;
            if use_bomb {
                let str_v = if stall_count > 15 { 3 } else { 5 };
                cluster_bomb_perturb(&mut state, tsn_ref.unwrap(), &mut rng, str_v);
            } else {
                let strategy = round % 8;
                let strength = 5 + round / 4;
                perturb_by_strategy(&mut state, strength, stall_count, strategy, &mut rng, hp);
            }
            greedy_reconstruct(&mut state, (round % 8) % 4);
            vnd_v2(&mut state, hp, tsn_ref);

            let h = compute_hash(&state.selected_bit);
            if tabu_hashes.contains(&h) {
                if let Some(t) = tsn_ref {
                    cluster_bomb_perturb(&mut state, t, &mut rng, 2);
                } else {
                    let extra_strength = 10 + round / 3;
                    perturb_by_strategy(&mut state, extra_strength, stall_count + 3, 6, &mut rng, hp);
                }
                greedy_reconstruct(&mut state, 0);
                vnd_v2(&mut state, hp, tsn_ref);
            }
            let h2 = compute_hash(&state.selected_bit);
            if tabu_hashes.len() < 128 { tabu_hashes.push(h2); }
            else { tabu_hashes[round % 128] = h2; }

            if state.total_value > best_val {
                best_val = state.total_value;
                best_sel = state.selected_items();
                stall_count = 0;
            }
        } else {
            stall_count = 0;
            let h = compute_hash(&state.selected_bit);
            if tabu_hashes.len() < 128 { tabu_hashes.push(h); }
        }
    }

    let mut final_state = State::new_empty(challenge);
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
            let v_before = final_state.total_value;
            local_search_vnd_windowed_deep(&mut final_state, hp.window_k);
            dp_refinement_hp(&mut final_state, ch);
            if final_state.total_value <= v_before { break; }
        }
    }

    if final_state.total_value > best_val {
        Solution { items: final_state.selected_items() }
    } else {
        Solution { items: best_sel }
    }
}

fn eval_solution(ch: &Challenge, sol: &Solution) -> i64 {
    let mut val: i64 = 0;
    for &i in &sol.items {
        val += ch.values[i] as i64;
        for &j in &sol.items {
            if j > i { val += ch.interaction_values[i][j] as i64; }
        }
    }
    val
}

// Path relinking: walk from sol_a towards sol_b, evaluating intermediates
fn path_relink(challenge: &Challenge, sol_a: &Solution, sol_b: &Solution, hp: &Hparams) -> Solution {
    let n = challenge.num_items;
    let mut in_a = vec![false; n];
    let mut in_b = vec![false; n];
    for &i in &sol_a.items { in_a[i] = true; }
    for &i in &sol_b.items { in_b[i] = true; }

    let mut state = State::new_empty(challenge);
    for &i in &sol_a.items { state.add_item(i); }

    let mut to_add: Vec<usize> = (0..n).filter(|&i| in_b[i] && !in_a[i]).collect();
    let mut to_remove: Vec<usize> = (0..n).filter(|&i| in_a[i] && !in_b[i]).collect();

    let mut best_val = state.total_value;
    let mut best_bits = state.selected_bit.clone();
    let cap = challenge.max_weight;

    while !to_add.is_empty() || !to_remove.is_empty() {
        let mut best_delta = i64::MIN;
        let mut best_action: Option<(bool, usize)> = None;

        for (idx, &item) in to_add.iter().enumerate() {
            if state.total_weight + challenge.weights[item] <= cap {
                let delta = state.contrib[item] as i64;
                if delta > best_delta { best_delta = delta; best_action = Some((true, idx)); }
            }
        }
        for (idx, &item) in to_remove.iter().enumerate() {
            let delta = -(state.contrib[item] as i64);
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

        if state.total_weight <= cap && state.total_value > best_val {
            best_val = state.total_value;
            best_bits = state.selected_bit.clone();
        }
    }

    // Polish the best intermediate with deep VND + DP
    let mut final_state = State::new_empty(challenge);
    for i in 0..n { if best_bits[i] { final_state.add_item(i); } }
    loop {
        let v_before = final_state.total_value;
        local_search_vnd_windowed_deep(&mut final_state, hp.window_k);
        dp_refinement_hp(&mut final_state, hp.core_half_dp);
        if final_state.total_value <= v_before { break; }
    }
    Solution { items: final_state.selected_items() }
}

// Frequency-biased greedy construction: items frequent in elite pool get bonus
fn build_frequency_biased(state: &mut State, freq: &[f64], rng: &mut Rng) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    loop {
        let slack = state.slack();
        if slack == 0 { break; }
        let mut best_i: Option<usize> = None;
        let mut best_s: f64 = f64::MIN;
        for i in 0..n {
            if state.selected_bit[i] { continue; }
            if state.ch.weights[i] > slack { continue; }
            let c = state.contrib[i] as f64;
            if c <= 0.0 { continue; }
            let w = (state.ch.weights[i] as f64).max(1.0);
            // Density score boosted by frequency from elite solutions
            let s = (c / w) * (1.0 + freq[i] * 2.0)
                + (rng.next_u32() & 0x3F) as f64 * 0.01;
            if s > best_s { best_s = s; best_i = Some(i); }
        }
        if let Some(i) = best_i { state.add_item(i); } else { break; }
    }
}

// Elite crossover: use frequency info to build a child
fn crossover_elite_frequency(elite: &[(Solution, i64)], ch: &Challenge, rng: &mut Rng) -> Vec<bool> {
    let n = ch.num_items;
    let mut freq = vec![0.0f64; n];
    let total = elite.len() as f64;
    for (sol, _) in elite {
        for &i in &sol.items { freq[i] += 1.0; }
    }
    // Probability of including item = frequency / total, with randomization
    let mut bits = vec![false; n];
    let mut weight: u32 = 0;
    // First pass: include high-frequency items
    let mut order: Vec<(usize, f64)> = (0..n).map(|i| {
        let p = freq[i] / total;
        let w = ch.weights[i] as f64;
        (i, p * 1000.0 + (ch.values[i] as f64) / w.max(1.0))
    }).collect();
    order.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for &(i, score) in &order {
        let p = freq[i] / total;
        let threshold = if p > 0.8 { 0.1 } else if p > 0.5 { 0.4 } else { 0.7 };
        if rng.next_f64() > threshold && weight + ch.weights[i] <= ch.max_weight {
            bits[i] = true;
            weight += ch.weights[i];
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
        let sum_w: u64 = challenge.weights.iter().map(|&w| w as u64).sum();
        let budget_pct = if sum_w > 0 { ((challenge.max_weight as u64) * 100 / sum_w) as u32 } else { 10 };
        let hp = Hparams::from_map(hyperparameters, n, budget_pct);
        let n_restarts = hp.n_full_restarts.max(1);

        // For non-enhanced path (n>1200), use original approach
        if n > 1200 {
            let mut best_sol: Option<Solution> = None;
            let mut best_quality: i64 = i64::MIN;
            for restart in 0..n_restarts {
                let sol = run_one_instance(challenge, &hp, restart);
                let val = eval_solution(challenge, &sol);
                if val > best_quality { best_quality = val; best_sol = Some(sol); }
            }
            return Ok(best_sol);
        }

        // === ENHANCED SOLVER for n<=1200, b>10 ===
        let mut rng = Rng::from_seed(&challenge.seed);
        for _ in 0..9999 { rng.next_u32(); } // unique rng stream

        // Phase 1: Collect diverse initial solutions
        let n_phase1 = n_restarts.min(10).max(4);
        let mut elite: Vec<(Solution, i64)> = Vec::new();
        for restart in 0..n_phase1 {
            let sol = run_one_instance(challenge, &hp, restart);
            let val = eval_solution(challenge, &sol);
            elite.push((sol, val));
        }
        elite.sort_by_key(|&(_, v)| std::cmp::Reverse(v));

        // Phase 2: Compute item frequencies from top solutions
        let top_k = elite.len().min(6);
        let mut freq = vec![0.0f64; n];
        for (sol, _) in &elite[..top_k] {
            for &i in &sol.items { freq[i] += 1.0 / top_k as f64; }
        }

        // Phase 3: Frequency-biased restarts
        let n_phase3 = n_restarts.saturating_sub(n_phase1);
        let ch_dp = hp.core_half_dp;
        // b≤10: medium VND (add+swap+pair+2-2, no chain_move) in ILS, heavy for final polish
        // b>10: windowed VND light in ILS, windowed deep for final polish
        let mini_ils_rounds = if budget_pct <= 5 { 80 } else if budget_pct <= 10 { 60 } else { 50 };
        for restart in 0..n_phase3 {
            let mut state = State::new_empty(challenge);
            if restart % 3 == 0 {
                let bits = crossover_elite_frequency(&elite[..top_k.min(elite.len())], challenge, &mut rng);
                set_state_from_bits(&mut state, &bits);
            } else {
                build_frequency_biased(&mut state, &freq, &mut rng);
            }
            dp_refinement_hp(&mut state, ch_dp);
            if budget_pct <= 10 { local_search_vnd_medium(&mut state, 25); }
            else { local_search_vnd_windowed(&mut state, hp.window_k); }

            // Mini ILS
            for round in 0..mini_ils_rounds {
                let snap = state.clone_solution();
                let strategy = round % 8;
                let strength = 5 + round / 4;
                perturb_by_strategy(&mut state, strength, 0, strategy, &mut rng, &hp);
                greedy_reconstruct(&mut state, strategy % 4);
                if budget_pct <= 10 { local_search_vnd_medium(&mut state, 25); }
                else { local_search_vnd_windowed(&mut state, hp.window_k); }
                dp_refinement_hp(&mut state, ch_dp);
                if state.total_value <= snap.value {
                    state.restore_solution(&snap);
                }
            }
            // Deep polish: heavy VND for b≤10, windowed deep for b>10
            loop {
                let v_before = state.total_value;
                if budget_pct <= 10 { local_search_vnd_heavy(&mut state); }
                else { local_search_vnd_windowed_deep(&mut state, hp.window_k); }
                dp_refinement_hp(&mut state, ch_dp);
                if state.total_value <= v_before { break; }
            }
            let sol = Solution { items: state.selected_items() };
            let val = eval_solution(challenge, &sol);
            elite.push((sol, val));
            elite.sort_by_key(|&(_, v)| std::cmp::Reverse(v));
            elite.truncate(12);

            // Update frequencies with new elite
            freq.fill(0.0);
            let tk = elite.len().min(6);
            for (sol, _) in &elite[..tk] {
                for &i in &sol.items { freq[i] += 1.0 / tk as f64; }
            }
        }

        // Phase 4: Path relinking between top diverse solutions
        let n_relink = elite.len().min(4);
        for i in 0..n_relink {
            for j in (i+1)..n_relink {
                let sol_ab = path_relink(challenge, &elite[i].0, &elite[j].0, &hp);
                let val_ab = eval_solution(challenge, &sol_ab);
                elite.push((sol_ab, val_ab));
                let sol_ba = path_relink(challenge, &elite[j].0, &elite[i].0, &hp);
                let val_ba = eval_solution(challenge, &sol_ba);
                elite.push((sol_ba, val_ba));
            }
        }
        elite.sort_by_key(|&(_, v)| std::cmp::Reverse(v));

        // Phase 5: Heavy VND + wide DP + LNS on top elite solutions
        let deep_dp = 150; // wider DP core for final polish
        let mut best_val = i64::MIN;
        let mut best_sel = Vec::new();

        // Polish top 3 elite solutions with heavy VND + wide DP
        for idx in 0..elite.len().min(3) {
            let mut state = State::new_empty(challenge);
            for &i in &elite[idx].0.items { state.add_item(i); }
            // Heavy VND + wide DP loop
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

        // LNS: large destroy + diverse reconstruct + heavy VND, 8 cycles
        let mut state = State::new_empty(challenge);
        for &i in &best_sel { state.add_item(i); }

        for lns_round in 0..8 {
            let snap = state.clone_solution();

            // Large ruin: remove 20-45% of selected items
            let sel = state.selected_items();
            let pct = 20 + (lns_round % 4) * 8; // 20%, 28%, 36%, 44%
            let n_remove = sel.len() * pct / 100;
            let mut candidates: Vec<(usize, i64)> = sel.iter().map(|&i| {
                let score = match lns_round % 8 {
                    0 => state.contrib[i] as i64,
                    1 => -(state.ch.weights[i] as i64),
                    2 => state.contrib[i] as i64 - state.ch.values[i] as i64,
                    3 => { let w = (state.ch.weights[i] as i64).max(1); (state.contrib[i] as i64 * 1000) / w },
                    4 => { let w = (state.ch.weights[i] as i64).max(1); (state.contrib[i] as i64 * 10000) / (w * w) },
                    5 => -(state.contrib[i] as i64), // best contrib = force diversity
                    6 => rng.next_u32() as i64,
                    _ => { // mixed: half random, half worst
                        if rng.next_u32() % 2 == 0 { rng.next_u32() as i64 }
                        else { state.contrib[i] as i64 }
                    },
                };
                (i, score)
            }).collect();
            candidates.sort_unstable_by_key(|&(_, s)| s);
            for j in 0..n_remove.min(candidates.len()) {
                state.remove_item(candidates[j].0);
            }

            // Reconstruct with alternating strategies
            match lns_round % 4 {
                0 => greedy_reconstruct(&mut state, 0), // by contrib
                1 => greedy_reconstruct(&mut state, 3), // by density
                2 => {
                    // Frequency-biased
                    let mut cands: Vec<usize> = (0..n).filter(|&i| !state.selected_bit[i]).collect();
                    cands.sort_unstable_by(|&a, &b| {
                        let sa = state.contrib[a] as f64 / (state.ch.weights[a] as f64).max(1.0)
                            + freq[a] * 50.0;
                        let sb = state.contrib[b] as f64 / (state.ch.weights[b] as f64).max(1.0)
                            + freq[b] * 50.0;
                        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    for &i in &cands {
                        if state.total_weight + state.ch.weights[i] <= challenge.max_weight {
                            state.add_item(i);
                        }
                    }
                },
                _ => greedy_reconstruct(&mut state, 2), // synergy-biased
            }

            // Full polish: deep VND → wide DP → heavy VND → wide DP
            loop {
                let v_before = state.total_value;
                local_search_vnd_windowed_deep(&mut state, hp.window_k);
                dp_refinement_hp(&mut state, deep_dp);
                if state.total_value <= v_before { break; }
            }
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
            // Always restart from best for next LNS round
            let mut rst = State::new_empty(challenge);
            for &i in &best_sel { rst.add_item(i); }
            state.restore_solution(&rst.clone_solution());
        }

        // Phase 6: Exact DP on disputed items
        // Find items that differ between top elite solutions → solve exactly
        elite.sort_by_key(|&(_, v)| std::cmp::Reverse(v));
        let tk = elite.len().min(6);
        let mut item_freq = vec![0usize; n];
        for (sol, _) in &elite[..tk] {
            for &i in &sol.items { item_freq[i] += 1; }
        }
        // Consensus items: always or never in top solutions
        let always_in: Vec<usize> = (0..n).filter(|&i| item_freq[i] == tk).collect();
        let never_in: Vec<usize> = (0..n).filter(|&i| item_freq[i] == 0).collect();
        let disputed: Vec<usize> = (0..n).filter(|&i| item_freq[i] > 0 && item_freq[i] < tk).collect();

        if disputed.len() > 0 && disputed.len() <= 200 {
            // Fix always_in items, run DP over disputed items only
            let mut state = State::new_empty(challenge);
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

                if dp_size <= 2_000_000 {
                    let mut dp = vec![i64::MIN / 4; dp_size];
                    let mut choose = vec![0u8; dk * dp_size];
                    dp[0] = 0;
                    let mut w_hi: usize = 0;

                    for (t, &it) in disputed.iter().enumerate() {
                        let wt = challenge.weights[it] as usize;
                        if wt > myw { continue; }
                        let val = state.contrib[it] as i64;
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

                    // Traceback
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

                    // Build full solution: always_in + dp_selected
                    for &i in &dp_selected { state.add_item(i); }

                    // Heavy VND + DP polish
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
    println!("knap_quality_opt_v3");
}
