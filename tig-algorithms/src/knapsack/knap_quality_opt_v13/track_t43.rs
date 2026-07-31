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
    // i29 (L4, min_level=7 replace): SPARSE-ADJACENCY move acceleration. The TIG
    // team-formation instance interaction matrix is ~99.6% zero (Jaccard of disjoint
    // project-subsets — each of the n=5000 participants is assigned ONE of ~535
    // subsets, so nonzero interaction only with the ~10-50 co-subset partners; cf
    // tig-challenges knapsack generator). The dense `add_item`/`remove_item` loop
    // over all n entries spends ~99.6% of work on zero adds. When `adj` is non-null
    // (set ONLY under hp `use_incremental_windows`), moves iterate the precomputed
    // nonzero-partner list → O(nnz)≈30 vs O(n)=5000, BYTE-EXACT (skipped entries are
    // +0 no-ops). Null = dense sentinel path (byte-identical AND clean-timed). Raw
    // ptr avoids a 2nd lifetime param; sound because the borrowed TopNeighbors.adj
    // outlives every move on this state (built before, dropped after).
    adj: *const Vec<Vec<(usize, i32)>>,
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
            adj: std::ptr::null(),
        }
    }

    /// i29: attach the shared sparse nonzero-partner adjacency (from TopNeighbors)
    /// to route add/remove through the sparse path. Caller guarantees `adj` outlives
    /// all subsequent moves on this state (it is a per-challenge-constant structure
    /// built once and reused across every restart/pass).
    #[inline(always)]
    fn attach_adj(&mut self, adj: &Vec<Vec<(usize, i32)>>) {
        self.adj = adj as *const Vec<Vec<(usize, i32)>>;
    }

    #[inline(always)] fn slack(&self) -> u32 { self.ch.max_weight - self.total_weight }

    #[inline(always)]
    fn add_item(&mut self, i: usize) {
        self.total_value += self.contrib[i] as i64;
        self.total_weight += self.ch.weights[i];
        let contrib_ptr = self.contrib.as_mut_ptr();
        if self.adj.is_null() {
            // Dense sentinel path (byte-identical to the original; clean-timed).
            let n = self.ch.num_items;
            let row_ptr = unsafe { self.ch.interaction_values.get_unchecked(i).as_ptr() };
            unsafe {
                for k in 0..n {
                    let ck = contrib_ptr.add(k);
                    *ck = (*ck).wrapping_add(*row_ptr.add(k));
                }
            }
        } else {
            // Sparse path: touch ONLY the nonzero partners. Byte-exact — the omitted
            // entries have interaction_values[i][k]==0, i.e. `+0` no-ops. Diagonal
            // [i][i]==0 in the generator, so contrib[i] is (correctly) untouched.
            let adj_ref: &Vec<Vec<(usize, i32)>> = unsafe { &*self.adj };
            let row = &adj_ref[i];
            unsafe {
                for &(k, v) in row.iter() {
                    let ck = contrib_ptr.add(k);
                    *ck = (*ck).wrapping_add(v);
                }
            }
        }
        self.selected_bit[i] = true;
    }

    #[inline(always)]
    fn remove_item(&mut self, j: usize) {
        self.total_value -= self.contrib[j] as i64;
        self.total_weight -= self.ch.weights[j];
        let contrib_ptr = self.contrib.as_mut_ptr();
        if self.adj.is_null() {
            let n = self.ch.num_items;
            let row_ptr = unsafe { self.ch.interaction_values.get_unchecked(j).as_ptr() };
            unsafe {
                for k in 0..n {
                    let ck = contrib_ptr.add(k);
                    *ck = (*ck).wrapping_sub(*row_ptr.add(k));
                }
            }
        } else {
            let adj_ref: &Vec<Vec<(usize, i32)>> = unsafe { &*self.adj };
            let row = &adj_ref[j];
            unsafe {
                for &(k, v) in row.iter() {
                    let ck = contrib_ptr.add(k);
                    *ck = (*ck).wrapping_sub(v);
                }
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

fn set_all_selected(state: &mut State, total_interactions: &[i64]) {
    let n = state.ch.num_items;
    let mut tv: i64 = 0;
    let mut tw: u32 = 0;
    for i in 0..n {
        state.selected_bit[i] = true;
        state.contrib[i] = state.ch.values[i] as i32 + total_interactions[i] as i32;
        tv += state.ch.values[i] as i64;
        tw += state.ch.weights[i];
    }
    tv += total_interactions.iter().sum::<i64>() / 2;
    state.total_value = tv;
    state.total_weight = tw;
}

fn build_greedy_density_from_all(state: &mut State, total_interactions: &[i64]) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    set_all_selected(state, total_interactions);
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

fn build_greedy_hub(state: &mut State, total_interactions: &[i64]) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let mut hub_scores: Vec<(usize, i64)> = (0..n).map(|i| (i, total_interactions[i])).collect();
    hub_scores.sort_unstable_by_key(|&(_, s)| std::cmp::Reverse(s));
    for &(i, _) in &hub_scores {
        if state.total_weight + state.ch.weights[i] <= cap { state.add_item(i); }
    }
}

fn build_greedy_synergy_weight(state: &mut State, total_interactions: &[i64]) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let nm1 = (n as i64 - 1).max(1);
    let mut scores: Vec<(usize, i64)> = (0..n).map(|i| {
        let avg_syn = total_interactions[i] / nm1;
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

fn perturb_by_strategy(state: &mut State, strength: usize, stall_count: usize, strategy: usize, rng: &mut Rng, hp: &Hparams, total_interactions: &[i64]) {
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
        7 => {
            removal_candidates = selected.iter().map(|&i| {
                let anti = 2 * state.contrib[i] as i64 - total_interactions[i];
                let w = (state.ch.weights[i] as i64).max(1);
                (i, (anti * 1000) / w)
            }).collect();
            removal_candidates.sort_unstable_by_key(|&(_, s)| s);
        },
        8 => {
            removal_candidates = selected.iter().map(|&i| {
                let c = state.contrib[i] as i64;
                let potential = total_interactions[i];
                (i, c * 100 - potential)
            }).collect();
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

fn greedy_reconstruct(state: &mut State, strategy: usize, total_interactions: &[i64]) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let mut candidates: Vec<usize> = (0..n).filter(|&i| !state.selected_bit[i]).collect();

    match strategy % 6 {
        0 => candidates.sort_unstable_by_key(|&i| -state.contrib[i]),
        1 => candidates.sort_unstable_by(|&a, &b| {
            state.ch.weights[a].cmp(&state.ch.weights[b])
                .then(state.contrib[b].cmp(&state.contrib[a]))
        }),
        2 => candidates.sort_unstable_by_key(|&i| {
            -(total_interactions[i] + state.contrib[i] as i64 / 10)
        }),
        3 => candidates.sort_unstable_by_key(|&i| {
            let w = (state.ch.weights[i] as i64).max(1);
            -((state.contrib[i] as i64 * 100) / w)
        }),
        4 => candidates.sort_unstable_by_key(|&i| {
            let w = (state.ch.weights[i] as i64).max(1);
            let c = state.contrib[i] as i64;
            let anti = 2 * c - total_interactions[i];
            -((anti * 100) / w + c / 5)
        }),
        _ => candidates.sort_unstable_by_key(|&i| {
            let c = state.contrib[i] as i64;
            let potential = total_interactions[i] / (n as i64).max(1);
            -(c + potential * 3)
        }),
    }

    for &i in &candidates {
        if state.total_weight + state.ch.weights[i] <= cap { state.add_item(i); }
    }
}

struct TopNeighbors {
    friends: Vec<Vec<usize>>,
    // i29: full nonzero-partner adjacency (j, interaction_value) per item, reused for
    // sparse add/remove moves. Built from the SAME O(n^2) scan already done for
    // `friends` (near-zero extra cost). Order is arbitrary — move updates are a
    // commutative sum of wrapping i32 adds, so byte-exactness is order-independent.
    adj: Vec<Vec<(usize, i32)>>,
    // i30 (L8, min_level=8 replace): gate for the SPARSE MOVE-EVAL decomposition of
    // the O(wk^2) replace loop in local_search_vnd_tsn (the real n=5000 hotspot,
    // DISTINCT from i29's sparse add/remove move-UPDATE which touched a minor O(n)
    // term). false = dense sentinel (byte-identical, clean-timed). true = column rm
    // materialized once from the sparse partner list `adj[rm]` into an L1/L2 scratch,
    // turning a strided ~100MB matrix column access into a cache-resident read. The
    // interaction matrix is symmetric (tig-challenges knapsack generator:
    // I[i][j]=I[j][i]=jaccard), so adj[rm]'s entry for k IS interaction_values[c=k][rm]
    // => Q byte-exact either way.
    sparse_moveeval: bool,
    // i31 (PIVOT-P1): when true, local_search_vnd_tsn maintains the best_unused/
    // worst_used windows incrementally from a dirty-set candidate pool instead of a
    // full O(n) rescan + double select_nth every pass. See the `win_incr` Hparams doc.
    win_incr: bool,
    // i33 (L8, min_level=8 replace): gate for the SPARSE MOVE-EVAL decomposition of
    // the TWO double-swap loops (1-remove/2-add l.~1541 and 2-remove/1-add l.~1574) —
    // the untouched heavy hotspot on the LEAN bp1 base (i30 sparsified only the single
    // replace loop). Each double-swap shares a common outer index (rm resp. add) whose
    // interaction column is read wk*wk times as a strided ~100MB access. We materialize
    // that column ONCE per outer iter from the sparse partner list `adj[·]` into an
    // L1/L2 scratch, then read it cache-resident. Interaction symmetry
    // (I[i][j]=I[j][i]) makes col[k] == interaction_values[k][outer] EXACTLY, and the
    // I[a1][a2]/I[r1][r2] friend-pair terms are left as dense reads (already sparse via
    // friends[·], k~12) => same order, same tie-break, same deltas => Q byte-exact.
    // false = dense sentinel (byte-identical, clean-timed).
    sparse_swap: bool,
    // i34 (L8, min_level=8 replace): flat interaction matrix LAYOUT. The framework's
    // `interaction_values: Vec<Vec<i32>>` is JAGGED — each row is a separate heap
    // allocation, so the strided hot read `interaction_values[c][rm]` (fixed rm,
    // varying c across ~200 window items) chases a DIFFERENT scattered row pointer
    // every step → TLB + cache miss, invisible to the HW prefetcher (it cannot stream
    // across scattered row bases). This is DISTINCT from the sparse-access family
    // (i28/i29/i30/i33, all FALSIFIED ratio≈1.0): those tried to AVOID reads via
    // partner lists; the gather overhead cancelled the saving. Here we keep the SAME
    // dense reads but change the LAYOUT: `flat_iv[i*n + j] == interaction_values[i][j]`
    // in one contiguous 100MB Vec<i32> (Hindsight 89203e04/4c6f27f5 rel:5: flatten
    // Vec<Vec<i32>>→Vec<i32> stride=n for L1/L2 cache access during delta calc). A
    // constant-stride read into ONE contiguous buffer IS prefetchable → the miss/chase
    // per (c,rm) is removed. Byte-EXACT (identical values, order, tie-break). Built
    // ONCE (TopNeighbors::new, reused across all restarts/passes). false = jagged
    // sentinel (must reproduce lean bp1 53375 byte-exact for the ratio pair).
    use_flat_iv: bool,
    flat_iv: Vec<i32>,
    // i35 (L8, min_level=8 replace): NARROW the interaction element i32->i16. Jaccard
    // values = (inter/union*1000) as i32 ∈ [0,1000] (framework knapsack/mod.rs:121) so
    // `v as i16` is LOSSLESS / byte-exact. Halves the streamed working set 100MB->50MB on
    // the memory-bandwidth-bound move-eval loop (Hindsight 6fecabca: L2 cache+bandwidth
    // dominate large QKP; 38cd5741: shrink footprint for L2 residency). STACKS on flat_iv.
    use_flat_iv16: bool,
    flat_iv16: Vec<i16>,
    // i36 (L8, min_level=8 replace): SYMMETRIC-TRANSPOSE the strided reads of the hot
    // move-eval loops. The replace loop reads `iv(c, rm)` with rm FIXED (outer) and c
    // varying (inner over best_unused) => flat index c*n+rm jumps by n=5000 elems each
    // step, scattering ~200 reads across the whole 100MB matrix (200 distinct, cold
    // cache lines — the hardware prefetcher CANNOT hide them because best_unused is a
    // short SCATTERED list, unlike the dead i28/i29 sequential O(n) scan). The matrix is
    // SYMMETRIC (framework knapsack/mod.rs:122-123 I[i][j]=I[j][i]=jaccard), so reading
    // `iv(rm, c)` = flat index rm*n+c is BYTE-IDENTICAL yet scans WITHIN the single fixed
    // row rm (one contiguous 20KB region, L2-resident, prefetchable). This is the i30
    // column-materialization locality benefit with ZERO gather/dirty-reset overhead (i30
    // ratio 1.045 / i33 1.036 were killed by the gather cost). false = original strided
    // order (drift-guard sentinel = 53375). true = row-major transpose. (Hindsight
    // 9301482b: row-CSR access beats strided column-CSC despite scattered reads.)
    use_iv_transpose: bool,
    // i37 (L8, min_level=8 replace): separable-upper-bound SCREEN of the O(wk^2) replace
    // loop (the confirmed n=5000 hotspot). d = contrib[c] - contrib[rm] - I[c][rm] with
    // I[c][rm] >= 0 (Jaccard, framework knapsack/mod.rs:121) ⇒ contrib[c]-contrib[rm] is
    // an UPPER BOUND on d. When that bound cannot beat the current incumbent best_d, the
    // precise delta (whose iv_mv() read is the cache-missing 100MB-matrix access) is a
    // PROVEN non-improver ⇒ we `continue` and SKIP the iv read. BYTE-EXACT: pruned c has
    // d <= bound <= best_d so `d > bd` is false anyway — same winner, same tie-break, same
    // iteration order (no sort) ⇒ identical bs, Q=53375. Attacks WORK/bandwidth (removes
    // iv reads), orthogonal to and stacks on flat_iv. He 2024 two-mode "fast approx screen
    // + precise delta only if it could win" (Hindsight 4f4c9fdc). false = full dense eval
    // (drift-guard sentinel = 53375). true = screened.
    use_bb_screen: bool,
    // i41 (L8, min_level=8 replace): SORTED early-BREAK of the O(wk^2) replace loop —
    // the second half of He 2024 two-mode O(k*n) that i37 (bb_screen) left on the table.
    // bb_screen only SKIPS the iv read of a provable non-improver but still ITERATES all
    // ~wk=200 candidates per rm (it got only -3.7%: the loop overhead, not the read, is
    // the cost). Here we sort best_unused DESCENDING by contrib ONCE per pass, so in the
    // inner scan ub = contrib[c] - contrib[rm] is NON-INCREASING. The moment ub <= the
    // current incumbent best_d, EVERY remaining c (lower contrib) is a proven non-improver
    // (d <= ub <= best_d, iv >= 0) => we `break` the whole inner loop, turning O(wk) into
    // O(effective prefix). This finds the SAME global max d (never prunes a potential
    // winner) — only the tie-break WINNER among equal-max-d moves can differ from density
    // order (directive-tolerated: Q >= 53359 & 32/32 deterministic; sort is deterministic).
    // Isolated to the replace loop via a separate sorted buffer; the other VND operators
    // keep the original density-ordered best_unused untouched. false = no sort/break
    // (drift-guard sentinel). true = sorted early-break. (Hindsight 4f4c9fdc/2e8b6c3a He
    // 2024: candidate list ordered by score + early termination = the O(k*n) reduction.)
    use_screen_break: bool,
    // i50 (L8, min_level=8 replace): port the screen_break sort+separable-early-break
    // and the incremental-window (win_incr) machinery to `local_search_vnd_tsn_light`
    // — the REAL n=5000 ILS hotspot. The heavy `local_search_vnd_tsn` (where the i37/i41
    // screen_break and i31 win_incr live) only runs in COLD construction + final polish;
    // for n>1200 the productive ILS trajectory (l.~2704) descends via tsn_light, whose
    // dense O(wk^2) replace loop + full O(n) per-pass window rebuild NEVER received these
    // proven levers. Separate flags so the champion (use_screen_break:1) sentinel path
    // through tsn_light stays byte-identical (both default false). i50 = light-path port.
    use_light_screen_break: bool,
    use_light_win_incr: bool,
    // i51 (L8, min_level=8 replace): fused refonte of the tsn_light replace loop. Same
    // sort-desc + separable early-break as use_light_screen_break, but the surviving
    // move-eval reads I[rm][c] ROW-CONTIGUOUSLY (rm = outer/fixed = row base hoisted once,
    // c = inner index steps +1 in memory) instead of the sentinel's strided I[c][rm]
    // (c-outer, stride = n*4B = cache-thrash). Byte-exact by matrix symmetry
    // (I[rm][c] == I[c][rm]); only the per-read LATENCY of the not-yet-broken candidates
    // changes. Distinct mechanism from screen_break (which cuts the ITERATION COUNT).
    // Sentinel false = unchanged. Isolated to tsn_light (track-scope).
    use_light_fast_replace: bool,
    // i91 (L8, min_level=8 replace, REOPEN brique inédite /research-challenge 20/07):
    // r-flip derivative screening of the tsn_light 2-ADD pair loop — the LAST unscreened
    // move operator of the true n=5000 ILS hotspot (i55 code-audit: light VND = replace
    // [screened i50/i51] + 2-add [NOT screened, up to wk*tsn_k=2400 iv reads/pass]).
    // Theory (Alidaee-Wang 2024, arXiv 2407.21062, Thm 1 / Cor. 1): at a 1-flip local
    // optimum an improving r-flip S must satisfy sum_{i in S}|E(x_i)| < M where M bounds
    // the coupling term. Mapping: E == contrib (maintained incrementally), the pair move
    // delta = contrib[a1] + contrib[a2] + I[a1][a2] with I[a1][a2] <= phi_a1[a1]
    // (max interaction of a1, friends are sorted desc by interaction). So
    //   delta <= contrib[a1] + cu2max + phi_a1[a1]
    // where cu2max = max(0, max contrib over UNUSED items with weight < slack) — the only
    // legal a2 candidates (computed in ONE contiguous O(n) pass when the operator runs).
    // Mode 1: skip a1 when that bound <= bd. BYTE-EXACT: every pruned pair has
    // delta <= bd, and the update below is strict (`delta > bd`) ⇒ same bd sequence, same
    // heuristic break behavior, same winner. Mode 2 (Strategy 1 of the paper): iterate
    // a1 in contrib-DESC order with the MONOTONE global break contrib[a1]+cu2max+phi_max
    // <= bd and NO heuristic `ca1<=0 && bd>0` break ⇒ screened SUPERSET neighborhood
    // (Q may move either way; bench-gated Q >= 53359, not byte-exact by construction).
    // 0 = off (drift-guard sentinel, dense loop byte-identical).
    use_rflip_screen: u8,
    // i91: per-item max interaction phi_a1[i] = max_j I[i][j] (row max, free at
    // construction since rows are sorted desc) + global max phi_max. Only materialized
    // on the rflip path so the sentinel construction stays clean-timed.
    phi_a1: Vec<i32>,
    phi_max: i32,
    n: usize,
}
impl TopNeighbors {
    fn new(ch: &Challenge, k: usize, sparse_moveeval: bool, win_incr: bool, sparse_swap: bool, use_flat_iv: bool, use_flat_iv16: bool, use_iv_transpose: bool, use_bb_screen: bool, use_screen_break: bool, use_light_screen_break: bool, use_light_win_incr: bool, use_light_fast_replace: bool, use_rflip_screen: u8) -> Self {
        let n = ch.num_items;
        let mut friends = vec![Vec::with_capacity(k); n];
        let mut adj: Vec<Vec<(usize, i32)>> = vec![Vec::new(); n];
        // i91: per-item row-max interaction, materialized ONLY on the rflip path so the
        // sentinel construction is untouched. After the sort below, row[0].1 IS the max.
        let mut phi_a1: Vec<i32> = if use_rflip_screen > 0 { vec![0i32; n] } else { Vec::new() };
        let mut phi_max: i32 = 0;
        for i in 0..n {
            let mut row: Vec<(usize, i32)> = Vec::new();
            for j in 0..n {
                if i != j && ch.interaction_values[i][j] > 0 {
                    row.push((j, ch.interaction_values[i][j]));
                }
            }
            adj[i] = row.clone();
            row.sort_unstable_by_key(|x| Reverse(x.1));
            if use_rflip_screen > 0 {
                let ph = row.first().map(|x| x.1).unwrap_or(0);
                phi_a1[i] = ph;
                if ph > phi_max { phi_max = ph; }
            }
            friends[i] = row.into_iter().take(k).map(|x| x.0).collect();
        }
        // i34: materialize the flat row-major (stride=n) copy ONLY on the flat path so
        // the jagged sentinel is untouched / clean-timed. extend_from_slice appends each
        // length-n row contiguously ⇒ flat_iv[i*n+j] == interaction_values[i][j] exactly.
        let flat_iv = if use_flat_iv {
            let mut f: Vec<i32> = Vec::with_capacity(n * n);
            for i in 0..n { f.extend_from_slice(&ch.interaction_values[i]); }
            f
        } else { Vec::new() };
        // i35: contiguous i16 copy (half the bytes). Jaccard ∈ [0,1000] ⇒ `as i16` lossless.
        // Built ONCE (reused across all restarts/passes), only on the i16 path.
        let flat_iv16 = if use_flat_iv16 {
            let mut f: Vec<i16> = Vec::with_capacity(n * n);
            for i in 0..n { for &v in &ch.interaction_values[i] { f.push(v as i16); } }
            f
        } else { Vec::new() };
        Self { friends, adj, sparse_moveeval, win_incr, sparse_swap, use_flat_iv, flat_iv, use_flat_iv16, flat_iv16, use_iv_transpose, use_bb_screen, use_screen_break, use_light_screen_break, use_light_win_incr, use_light_fast_replace, use_rflip_screen, phi_a1, phi_max, n }
    }
    /// i34/i35: single hot-read accessor. i16 path = narrow contiguous prefetchable read
    /// (half the bandwidth); flat path = contiguous i32; jagged sentinel = byte-identical.
    #[inline(always)]
    fn iv(&self, ch: &Challenge, i: usize, j: usize) -> i32 {
        if self.use_flat_iv16 {
            // i35: read the narrow i16 copy, widen to i32. Byte-exact (values ≤ 1000).
            // SAFETY: i,j < n by construction; flat_iv16 has length n*n.
            unsafe { *self.flat_iv16.get_unchecked(i * self.n + j) as i32 }
        } else if self.use_flat_iv {
            // SAFETY: i,j < n by construction (same indices used to index the [n][n]
            // ch.interaction_values in the sentinel path); flat_iv has length n*n.
            unsafe { *self.flat_iv.get_unchecked(i * self.n + j) }
        } else {
            ch.interaction_values[i][j]
        }
    }
    /// i36: move-eval read of I[fixed][var] == I[var][fixed] (matrix symmetry). When
    /// use_iv_transpose, index row-major on the FIXED outer index (`iv(fixed, var)` =
    /// within-row contiguous scan of row `fixed`); else keep the original var-outer
    /// strided order (`iv(var, fixed)`) as the byte-exact drift-guard sentinel. Both
    /// return the identical value (symmetry) — only the memory access pattern differs.
    #[inline(always)]
    fn iv_mv(&self, ch: &Challenge, fixed: usize, var: usize) -> i32 {
        if self.use_iv_transpose { self.iv(ch, fixed, var) } else { self.iv(ch, var, fixed) }
    }
}

/// i31 (PIVOT-P1): record the items a just-applied move touched, plus their sparse
/// interaction partners (whose realized contrib shifted), into the dirty set so the
/// next incremental window rebuild re-evaluates them. adj[it] is the exact set of
/// items whose contrib changed when `it` was added/removed (the move-update already
/// adjusts state.contrib for these), so this captures every frontier candidate the
/// move could have promoted. Called ONLY on the win_incr path (guarded at each site)
/// so the sentinel stays clean-timed.
#[inline(always)]
fn mark_dirty(dirty: &mut Vec<usize>, tsn: &TopNeighbors, items: &[usize]) {
    for &it in items {
        dirty.push(it);
        for &(k, _) in &tsn.adj[it] { dirty.push(k); }
    }
}

fn local_search_vnd_tsn(state: &mut State, tsn: &TopNeighbors, two_ex_cap: usize) {
    let n = state.ch.num_items;
    let wk: usize = if n > 3000 { 200 } else { 300 };
    let mut unused_buf: Vec<(usize, i64, i64)> = Vec::with_capacity(n);
    let mut used_buf: Vec<(usize, i64, i64)> = Vec::with_capacity(n);
    let mut best_unused: Vec<usize> = Vec::with_capacity(wk);
    let mut worst_used: Vec<usize> = Vec::with_capacity(wk);
    // i41: sorted-by-contrib-desc copy of best_unused for the replace early-break. Kept
    // SEPARATE from best_unused so the other VND operators keep their density order.
    // Allocated ONLY on the screen_break path (sentinel untouched / clean-timed).
    let mut bu_sorted: Vec<usize> = if tsn.use_screen_break { Vec::with_capacity(wk) } else { Vec::new() };
    // i30: sparse move-eval scratch column (partners-only, dirty-reset per rm).
    // Allocated ONLY on the sparse path so the dense sentinel is untouched/clean-timed.
    let mut col: Vec<i32> = if tsn.sparse_moveeval { vec![0i32; n] } else { Vec::new() };
    // i33: sparse double-swap scratch column (per common-outer index, dirty-reset in
    // O(nnz)). Allocated ONLY on the sparse_swap path so the dense sentinel is untouched.
    let mut swc: Vec<i32> = if tsn.sparse_swap { vec![0i32; n] } else { Vec::new() };
    // i31 (PIVOT-P1): incremental dirty-set window scratch. Allocated ONLY on the
    // win_incr path so the sentinel (win_incr=false) is untouched / clean-timed and
    // byte-identical to the original full-rescan code below.
    let mut dirty: Vec<usize> = Vec::new();
    let mut in_pool: Vec<bool> = if tsn.win_incr { vec![false; n] } else { Vec::new() };
    let mut pass: usize = 0;
    for _ in 0..80 {
        // FULL rescan on the sentinel (win_incr=false), on pass 0, and periodically
        // (every 8th pass) to re-seed items that drifted into the frontier but were
        // never a dirty partner (bounds Q-drift). Otherwise rebuild the windows from
        // a SMALL candidate pool = prev windows + dirty items => no O(n) scan.
        let full_rebuild = !tsn.win_incr || pass == 0 || pass % 8 == 0;
        unused_buf.clear();
        used_buf.clear();
        if full_rebuild {
            for i in 0..n {
                let c = state.contrib[i] as i64;
                let w = (state.ch.weights[i] as i64).max(1);
                if state.selected_bit[i] { used_buf.push((i, c, w)); } else { unused_buf.push((i, c, w)); }
            }
        } else {
            // Candidate pool = prev best_unused ∪ prev worst_used ∪ dirty (deduped via
            // in_pool). contrib/weight/selected_bit are read FRESH (moves already keep
            // state.contrib current), so no stale-snapshot risk.
            for &i in best_unused.iter().chain(worst_used.iter()).chain(dirty.iter()) {
                if !in_pool[i] {
                    in_pool[i] = true;
                    let c = state.contrib[i] as i64;
                    let w = (state.ch.weights[i] as i64).max(1);
                    if state.selected_bit[i] { used_buf.push((i, c, w)); } else { unused_buf.push((i, c, w)); }
                }
            }
            for t in unused_buf.iter() { in_pool[t.0] = false; }
            for t in used_buf.iter() { in_pool[t.0] = false; }
        }
        dirty.clear();
        best_unused.clear();
        worst_used.clear();
        pass += 1;
        let ku = wk.min(unused_buf.len());
        if ku > 0 && ku < unused_buf.len() {
            unused_buf.select_nth_unstable_by(ku - 1, |a, b| (b.1 * a.2).cmp(&(a.1 * b.2)));
        }
        let ks = wk.min(used_buf.len());
        if ks > 0 && ks < used_buf.len() {
            used_buf.select_nth_unstable_by(ks - 1, |a, b| (a.1 * b.2).cmp(&(b.1 * a.2)));
        }
        for t in &unused_buf[..ku] { best_unused.push(t.0); }
        for t in &used_buf[..ks] { worst_used.push(t.0); }

        let slack = state.slack();
        if slack > 0 {
            let mut ba: Option<(usize, i32)> = None;
            for &c in &best_unused {
                if state.ch.weights[c] > slack { continue; }
                let d = state.contrib[c];
                if d > 0 && ba.map_or(true, |(_, bd)| d > bd) { ba = Some((c, d)); }
            }
            if let Some((c, _)) = ba {
                state.add_item(c);
                if tsn.win_incr { mark_dirty(&mut dirty, tsn, &[c]); }
                continue;
            }
        }

        {
            let mut bs: Option<(usize, usize, i32)> = None;
            if tsn.use_screen_break && !tsn.sparse_moveeval {
                // i41: SORTED early-BREAK path. Sort best_unused DESCENDING by contrib ONCE
                // this pass into bu_sorted (a separate buffer; best_unused keeps its density
                // order for the other operators). In the inner scan ub = contrib[c]-crm is
                // then NON-INCREASING, so once ub <= the incumbent best_d every remaining c
                // is a proven non-improver ⇒ break the whole inner loop (not just skip the
                // read). Finds the SAME global max d (never prunes a winner); only the
                // tie-break winner among equal-max-d moves can differ (directive-tolerated,
                // deterministic). O(wk^2) -> O(wk * effective prefix).
                bu_sorted.clear();
                bu_sorted.extend_from_slice(&best_unused);
                bu_sorted.sort_by(|&x, &y| state.contrib[y].cmp(&state.contrib[x]));
                for &rm in &worst_used {
                    let max_w = state.ch.weights[rm] + state.slack();
                    let crm = state.contrib[rm];
                    for &c in &bu_sorted {
                        let ub = state.contrib[c] - crm;
                        // Break BEFORE the weight filter: ub is non-increasing, so if the
                        // best achievable delta here cannot beat the incumbent, neither can
                        // any later (lower-contrib) c — regardless of their weights.
                        match bs { Some((_, _, bd)) => { if ub <= bd { break; } }, None => { if ub <= 0 { break; } } }
                        if state.ch.weights[c] > max_w { continue; }
                        let d = ub - tsn.iv_mv(state.ch, rm, c);
                        if d > 0 && bs.map_or(true, |(_, _, bd)| d > bd) { bs = Some((c, rm, d)); }
                    }
                }
            } else if !tsn.sparse_moveeval {
                // Dense sentinel path — byte-identical to the original, clean-timed.
                for &rm in &worst_used {
                    let max_w = state.ch.weights[rm] + state.slack();
                    let crm = state.contrib[rm];
                    for &c in &best_unused {
                        if state.ch.weights[c] > max_w { continue; }
                        // i37: separable upper-bound SCREEN. ub = contrib[c]-contrib[rm] >= d
                        // (iv >= 0). If ub cannot beat the incumbent, the precise delta is a
                        // proven non-improver ⇒ skip the cache-missing iv_mv read. Same order,
                        // same tie-break ⇒ BYTE-EXACT (pruned d <= ub <= best_d ⇒ d>bd false).
                        let ub = state.contrib[c] - crm;
                        if tsn.use_bb_screen {
                            match bs { Some((_, _, bd)) => { if ub <= bd { continue; } }, None => { if ub <= 0 { continue; } } }
                        }
                        let d = ub - tsn.iv_mv(state.ch, rm, c);
                        if d > 0 && bs.map_or(true, |(_, _, bd)| d > bd) { bs = Some((c, rm, d)); }
                    }
                }
            } else {
                // i30 (L8 replace): SPARSE MOVE-EVAL decomposition of the O(wk^2)
                // replace loop — the real n=5000 hotspot (mem c003-t43: replace
                // double-loop ~40000 evals/pass, each a strided column read
                // `interaction_values[c][rm]` into a ~100MB matrix => a cache miss
                // per (c,rm)). The delta d = contrib[c] - contrib[rm] - I[c][rm] is
                // SEPARABLE (He 2024, Hindsight 4f4c9fdc/2e8b6c3a two-mode scoring;
                // 9301482b: row-CSR access beats strided column-CSC). For each rm we
                // materialize column rm ONCE from its sparse partner list adj[rm]
                // (partners only; non-partners keep the pre-zeroed 0) into a small
                // L1/L2-resident `col`, then read col[c] in the inner loop. Interaction
                // symmetry (I[rm][k]==I[k][rm]) makes col[c] == interaction_values[c][rm]
                // EXACTLY for every c. Same rm-outer/c-inner order + same strict `>`
                // tie-break + identical d values => BYTE-EXACT winner (53375). col is
                // dirty-reset in O(nnz) so it stays all-zero between rm iterations.
                for &rm in &worst_used {
                    let max_w = state.ch.weights[rm] + state.slack();
                    let crm = state.contrib[rm];
                    let adj_rm = &tsn.adj[rm];
                    for &(k, v) in adj_rm.iter() { col[k] = v; }
                    for &c in &best_unused {
                        if state.ch.weights[c] > max_w { continue; }
                        let d = state.contrib[c] - crm - col[c];
                        if d > 0 && bs.map_or(true, |(_, _, bd)| d > bd) { bs = Some((c, rm, d)); }
                    }
                    for &(k, _) in adj_rm.iter() { col[k] = 0; }
                }
            }
            if let Some((c, rm, _)) = bs {
                state.replace_item(rm, c);
                if tsn.win_incr { mark_dirty(&mut dirty, tsn, &[rm, c]); }
                continue;
            }
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
                            let delta = ca1 + state.contrib[a2] as i64 + tsn.iv(state.ch, a1, a2) as i64;
                            if delta > bd { bd = delta; bp = Some((a1, a2)); }
                        }
                    }
                }
                if let Some((a1, a2)) = bp {
                    state.add_item(a1); state.add_item(a2);
                if tsn.win_incr { mark_dirty(&mut dirty, tsn, &[a1, a2]); }
                    continue;
                }
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
                // i33: materialize column rm once from its sparse partner list; swc[k]
                // == I[k][rm] == interaction_values[k][rm] (symmetry), non-partners = 0.
                if tsn.sparse_swap { for &(k, v) in tsn.adj[rm].iter() { swc[k] = v; } }
                for &a1 in &best_unused {
                    let ca1 = state.contrib[a1] as i64;
                    let wa1 = state.ch.weights[a1];
                    if wa1 >= budget { continue; }
                    let i_a1_rm = if tsn.sparse_swap { swc[a1] as i64 } else { tsn.iv_mv(state.ch, rm, a1) as i64 };
                    let ca1_eff = ca1 - i_a1_rm;
                    for &a2 in &tsn.friends[a1] {
                        if state.selected_bit[a2] || a1 == a2 { continue; }
                        let wa2 = state.ch.weights[a2];
                        if wa1 + wa2 > budget { continue; }
                        let i_a2_rm = if tsn.sparse_swap { swc[a2] as i64 } else { tsn.iv_mv(state.ch, rm, a2) as i64 };
                        let delta = ca1_eff + state.contrib[a2] as i64
                            - i_a2_rm
                            + tsn.iv(state.ch, a1, a2) as i64 - c_rm;
                        if delta > bd && state.total_weight - w_rm + wa1 + wa2 <= cap {
                            bd = delta; bm = Some((rm, a1, a2));
                        }
                    }
                }
                if tsn.sparse_swap { for &(k, _) in tsn.adj[rm].iter() { swc[k] = 0; } }
            }
            if let Some((rm, a1, a2)) = bm {
                state.remove_item(rm); state.add_item(a1); state.add_item(a2);
                if tsn.win_incr { mark_dirty(&mut dirty, tsn, &[rm, a1, a2]); }
                continue;
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
                // i33: materialize column add once; swc[k] == I[k][add] (symmetry).
                if tsn.sparse_swap { for &(k, v) in tsn.adj[add].iter() { swc[k] = v; } }
                for &r1 in &worst_used {
                    let cr1 = state.contrib[r1] as i64;
                    let wr1 = state.ch.weights[r1];
                    let c_add_r1 = if tsn.sparse_swap { swc[r1] as i64 } else { tsn.iv(state.ch, add, r1) as i64 };
                    for &r2 in &tsn.friends[r1] {
                        if !state.selected_bit[r2] || r1 == r2 { continue; }
                        let wr2 = state.ch.weights[r2];
                        if state.total_weight + w_add <= cap + wr1 + wr2 {
                            let i_add_r2 = if tsn.sparse_swap { swc[r2] as i64 } else { tsn.iv(state.ch, add, r2) as i64 };
                            let delta = c_add - c_add_r1
                                - i_add_r2
                                - cr1 - state.contrib[r2] as i64
                                + tsn.iv(state.ch, r1, r2) as i64;
                            if delta > bd { bd = delta; bm = Some((r1, r2, add)); }
                        }
                    }
                }
                if tsn.sparse_swap { for &(k, _) in tsn.adj[add].iter() { swc[k] = 0; } }
            }
            if let Some((r1, r2, add)) = bm {
                state.remove_item(r1); state.remove_item(r2); state.add_item(add);
                if tsn.win_incr { mark_dirty(&mut dirty, tsn, &[r1, r2, add]); }
                continue;
            }
        }

        {
            let cap = state.ch.max_weight;
            let mut bd = 0i64;
            let mut bm = None;
            let ks = worst_used.len().min(two_ex_cap);
            let ku = best_unused.len().min(two_ex_cap);
            for i in 0..ks {
                let r1 = worst_used[i];
                let cr1 = state.contrib[r1] as i64;
                let wr1 = state.ch.weights[r1];
                for &r2 in &tsn.friends[r1] {
                    if !state.selected_bit[r2] || r1 == r2 { continue; }
                    let cr2 = state.contrib[r2] as i64;
                    let wr2 = state.ch.weights[r2];
                    let lost = cr1 + cr2 - tsn.iv(state.ch, r1, r2) as i64;
                    let budget = state.slack() + wr1 + wr2;
                    for u in 0..ku {
                        let a1 = best_unused[u];
                        let wa1 = state.ch.weights[a1];
                        if wa1 >= budget { continue; }
                        let ca1_eff = state.contrib[a1] as i64
                            - tsn.iv(state.ch, a1, r1) as i64
                            - tsn.iv(state.ch, a1, r2) as i64;
                        for &a2 in &tsn.friends[a1] {
                            if state.selected_bit[a2] || a1 == a2 { continue; }
                            let wa2 = state.ch.weights[a2];
                            if wa1 + wa2 <= budget {
                                let gained = ca1_eff + state.contrib[a2] as i64
                                    - tsn.iv(state.ch, a2, r1) as i64
                                    - tsn.iv(state.ch, a2, r2) as i64
                                    + tsn.iv(state.ch, a1, a2) as i64;
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
                state.add_item(a1); state.add_item(a2);
                if tsn.win_incr { mark_dirty(&mut dirty, tsn, &[r1, r2, a1, a2]); }
                continue;
            }
        }

        break;
    }
}

/// i16 threshold-accepting move for the walk below. Feasible-by-construction:
/// every kind checks the weight budget before it is ever proposed, so accepting a
/// worsening move never breaks feasibility (no invalid-nonce risk).
enum TaMove {
    Add(usize),
    Drop(usize),
    Replace(usize, usize),
}

/// i16 NEW FAMILY — threshold-accepting acceptance walk (refine_mode 60/61/62).
///
/// Every prior t43 refine family is STRICT-IMPROVE (VND, seeded escape, PR, ALNS,
/// exact-core, linkage). They all saturate at 53327: the atomic <=2-move descent
/// is trapped in the same basin the 1-swap reaches (mem c003-t41-tabu2ex-plateau:
/// for dense QKP the 2-exchange basin == 1-swap basin — pivot to threshold-accept
/// / IRTS/TSBMA). This walk drives the SAME incremental delta-moves (add / drop /
/// 1-1 replace over density candidate lists) but ACCEPTS controlled worsening so it
/// can traverse the 2-opt plateau, then re-descends with the strict VND. A short
/// tabu tenure blocks immediate move reversal (anti-cycling). Three acceptance
/// families (R7, 3 distinct mechanisms):
///   accept_mode 0 = Threshold Accepting  (accept delta >= -T, T decays linearly)
///   accept_mode 1 = Great Deluge         (accept new_value >= level, level rises)
///   accept_mode 2 = Late-Acceptance HC   (accept new_value >= value L steps ago)
/// Fully deterministic (greedy max-delta move, index tie-break; no RNG, no
/// wall-clock). Keep-best: returns the best state ever visited, so starting from
/// the refine2 incumbent it can never regress below it.
fn threshold_accept_escape(
    ch: &Challenge,
    incumbent: &Solution,
    tsn: &TopNeighbors,
    two_ex_cap: usize,
    walk_iters: usize,
    accept_mode: usize,
    ta_ratio: f64,
    tenure: usize,
    lahc_len: usize,
) -> Solution {
    let n = ch.num_items;
    let mut st = State::new_empty(ch);
    for &i in &incumbent.items { st.add_item(i); }

    let mut best = st.clone_solution();
    let mut best_q = st.total_value;
    let base_q = best_q;
    if base_q <= 0 || walk_iters == 0 { return Solution { items: st.selected_items() }; }

    let mut tabu: Vec<usize> = vec![0; n]; // move index until which item i is frozen
    let ll = lahc_len.max(1);
    let mut lahc_hist: Vec<i64> = vec![base_q; ll];

    let wk: usize = if n > 3000 { 200 } else { 300 };
    let mut unused_buf: Vec<(usize, i64, i64)> = Vec::with_capacity(n);
    let mut used_buf: Vec<(usize, i64, i64)> = Vec::with_capacity(n);

    for w in 0..walk_iters {
        unused_buf.clear();
        used_buf.clear();
        for i in 0..n {
            let c = st.contrib[i] as i64;
            let ww = (ch.weights[i] as i64).max(1);
            if st.selected_bit[i] { used_buf.push((i, c, ww)); } else { unused_buf.push((i, c, ww)); }
        }
        let ku = wk.min(unused_buf.len());
        if ku > 0 && ku < unused_buf.len() {
            unused_buf.select_nth_unstable_by(ku - 1, |a, b| (b.1 * a.2).cmp(&(a.1 * b.2)));
        }
        let ks = wk.min(used_buf.len());
        if ks > 0 && ks < used_buf.len() {
            used_buf.select_nth_unstable_by(ks - 1, |a, b| (a.1 * b.2).cmp(&(b.1 * a.2)));
        }

        // Best feasible move (delta may be negative), skipping tabu-frozen items.
        let slack = st.slack();
        let mut best_delta = i64::MIN;
        let mut best_move: Option<TaMove> = None;
        for t in &unused_buf[..ku] {
            let c = t.0;
            if tabu[c] > w { continue; }
            if ch.weights[c] > slack { continue; }
            let d = st.contrib[c] as i64;
            if d > best_delta { best_delta = d; best_move = Some(TaMove::Add(c)); }
        }
        for t in &used_buf[..ks] {
            let r = t.0;
            if tabu[r] > w { continue; }
            let d = -(st.contrib[r] as i64);
            if d > best_delta { best_delta = d; best_move = Some(TaMove::Drop(r)); }
        }
        for tr in &used_buf[..ks] {
            let rm = tr.0;
            if tabu[rm] > w { continue; }
            let max_w = ch.weights[rm] + slack;
            let crm = st.contrib[rm] as i64;
            for tc in &unused_buf[..ku] {
                let c = tc.0;
                if tabu[c] > w { continue; }
                if ch.weights[c] > max_w { continue; }
                let d = st.contrib[c] as i64 - crm - ch.interaction_values[c][rm] as i64;
                if d > best_delta { best_delta = d; best_move = Some(TaMove::Replace(rm, c)); }
            }
        }

        let mv = match best_move { Some(m) => m, None => break };
        let new_q = st.total_value + best_delta;

        let accept = match accept_mode {
            0 => {
                let frac = 1.0 - (w as f64) / (walk_iters as f64);
                let thr = (base_q as f64 * ta_ratio * frac) as i64;
                best_delta >= -thr
            }
            1 => {
                let frac = 1.0 - (w as f64) / (walk_iters as f64);
                let level = (base_q as f64 * (1.0 - ta_ratio * frac)) as i64;
                new_q >= level
            }
            _ => {
                let idx = w % ll;
                new_q >= lahc_hist[idx] || best_delta > 0
            }
        };

        if !accept {
            if accept_mode == 2 {
                // LAHC keeps walking; record current (unchanged) value in history.
                let idx = w % ll;
                if st.total_value > lahc_hist[idx] { lahc_hist[idx] = st.total_value; }
                continue;
            }
            // TA / GD: threshold only tightens from here, so the max-delta move stays
            // rejected -> converged. Stop the walk.
            break;
        }

        match mv {
            TaMove::Add(c) => { st.add_item(c); tabu[c] = w + tenure; }
            TaMove::Drop(r) => { st.remove_item(r); tabu[r] = w + tenure; }
            TaMove::Replace(rm, c) => {
                st.remove_item(rm); st.add_item(c);
                tabu[rm] = w + tenure; tabu[c] = w + tenure;
            }
        }
        if st.total_value > best_q { best_q = st.total_value; best = st.clone_solution(); }
        if accept_mode == 2 {
            let idx = w % ll;
            if st.total_value > lahc_hist[idx] { lahc_hist[idx] = st.total_value; }
        }
    }

    // Intensify from the best visited state with the strict multi-neighborhood VND.
    st.restore_solution(&best);
    local_search_vnd_tsn(&mut st, tsn, two_ex_cap);
    if st.total_value > best_q { best = st.clone_solution(); }
    st.restore_solution(&best);
    Solution { items: st.selected_items() }
}

/// i11 F_E helper: return a copy of `sol` with the lowest-density fraction of its
/// items stripped, sized by the shell index r in [0,k). r=0 returns the incumbent
/// untouched (so the first escape reproduces F2's behaviour and keep-best can never
/// regress below refine_mode=2). Density ranking mirrors the ruin used inside
/// run_one_instance_seeded (contrib*1000/weight, ascending). Deterministic.
fn strip_low_density(ch: &Challenge, sol: &Solution, r: usize, k: usize) -> Solution {
    if r == 0 || sol.items.is_empty() { return sol.clone(); }
    let mut st = State::new_empty(ch);
    for &i in &sol.items { st.add_item(i); }
    let mut scored: Vec<(usize, i64)> = sol.items.iter().map(|&i| {
        let w = (ch.weights[i] as i64).max(1);
        (i, (st.contrib[i] as i64 * 1000) / w)
    }).collect();
    scored.sort_unstable_by_key(|&(_, s)| s);
    let frac = (r as f64) / ((k + 2) as f64);
    let n_strip = (((sol.items.len() as f64) * frac) as usize).min(sol.items.len());
    let keep: Vec<usize> = scored.iter().skip(n_strip).map(|&(i, _)| i).collect();
    Solution { items: keep }
}

/// i13 (min_level=8, combine): STRATEGIC OSCILLATION over the full instance.
///
/// Memory 14741428: on t43 the feasible VND neighborhoods are EXHAUSTED — width
/// (i9 tsn_k/two_ex_cap widening REGRESSED), count (i10 basin-breadth <= champion)
/// and order (i11 F1/F2 compositions tie 53327) all saturate at the same attractor,
/// and only INFEASIBLE-navigation or global reconstruction can escape it (ALNS
/// reconstruction already failed, rebuilding the same attractor). The feasible VND
/// is bounded to atomic <=2-in / 2-out swaps, so any path to a better basin that
/// must transit MORE than 2 items over capacity is unreachable to it.
///
/// Strategic oscillation (Glover) crosses the feasibility boundary deliberately:
/// a CONSTRUCTIVE phase greedily packs the highest-marginal unselected items past
/// capacity up to cap*(1+beta), then a DESTRUCTIVE phase sheds the lowest marginal
/// -density items below cap*(1-beta). The best FEASIBLE solution touched anywhere on
/// the trajectory is tracked, and the best touch-down is re-polished by the exact
/// -delta VND. keep-best on the feasible incumbent => the returned Solution is ALWAYS
/// feasible (never below `seed`), so downside is bounded and validity is preserved.
/// Deterministic (no RNG, greedy by contrib): c003 needs 32/32.
fn strategic_oscillation(
    challenge: &Challenge,
    seed: &Solution,
    hp: &Hparams,
    tsn: &TopNeighbors,
    cycles: usize,
    beta0: f64,
    expand: bool,
) -> Solution {
    let n = challenge.num_items;
    let cap = challenge.max_weight as i64;
    let mut st = State::new_empty(challenge);
    for &i in &seed.items { st.add_item(i); }
    // Best FEASIBLE solution seen on the whole oscillation trajectory.
    let mut best_bits = st.selected_bit.clone();
    let mut best_val = st.total_value;

    let mut beta = beta0;
    for _cyc in 0..cycles {
        let over_limit = ((cap as f64) * (1.0 + beta)) as i64;
        let under_limit = ((cap as f64) * (1.0 - beta)) as i64;

        // CONSTRUCTIVE: pack past capacity by highest marginal contribution.
        loop {
            if (st.total_weight as i64) >= over_limit { break; }
            let mut best_i: Option<usize> = None;
            let mut best_c: i32 = 0;
            for i in 0..n {
                if st.selected_bit[i] { continue; }
                if (st.total_weight as i64) + (challenge.weights[i] as i64) > over_limit { continue; }
                let c = st.contrib[i];
                if c > best_c { best_c = c; best_i = Some(i); }
            }
            match best_i {
                Some(i) => {
                    st.add_item(i);
                    if (st.total_weight as i64) <= cap && st.total_value > best_val {
                        best_val = st.total_value;
                        best_bits.clone_from(&st.selected_bit);
                    }
                }
                None => break,
            }
        }

        // DESTRUCTIVE: shed lowest marginal-density items below cap*(1-beta),
        // snapshotting every feasible touch-down along the descent.
        loop {
            if (st.total_weight as i64) <= cap && st.total_value > best_val {
                best_val = st.total_value;
                best_bits.clone_from(&st.selected_bit);
            }
            if (st.total_weight as i64) <= under_limit { break; }
            let mut worst_i: Option<usize> = None;
            let mut worst_score: i64 = i64::MAX;
            for i in 0..n {
                if !st.selected_bit[i] { continue; }
                let w = (challenge.weights[i] as i64).max(1);
                let s = (st.contrib[i] as i64 * 1000) / w;
                if s < worst_score { worst_score = s; worst_i = Some(i); }
            }
            match worst_i {
                Some(i) => st.remove_item(i),
                None => break,
            }
        }

        if expand { beta += beta0; }
    }

    // Re-polish the best feasible touch-down with the exact-delta VND (keep-best).
    let mut pst = State::new_empty(challenge);
    for i in 0..n { if best_bits[i] { pst.add_item(i); } }
    if (pst.total_weight as i64) <= cap {
        local_search_vnd_tsn(&mut pst, tsn, hp.two_ex_cap);
        if (pst.total_weight as i64) <= cap && pst.total_value > best_val {
            best_val = pst.total_value;
            best_bits.clone_from(&pst.selected_bit);
        }
    }

    Solution { items: (0..n).filter(|&i| best_bits[i]).collect() }
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

#[derive(Clone, Copy)]
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
    // i10 (min_level=8): expose the previously-hardcoded throttle constants of the
    // active large-n LS `local_search_vnd_tsn` so the 2-out-2-in exchange candidate
    // list can be widened on the rc_mode5-reduced instance (Zhou-Hao 307b936f /
    // 656dba47: 2-exchange finds configs the pruned amis-12/cap-30 version misses).
    // Defaults reproduce the champion byte-exact (tsn_k=12, two_ex_cap=30).
    tsn_k: usize,
    two_ex_cap: usize,
    // i29 (L4, min_level=7): gate for sparse-adjacency move acceleration. false =
    // dense move loop (byte-identical sentinel, clean bp1 timing). true = O(nnz)
    // sparse moves on the ~0.4%-dense TIG interaction matrix. Q byte-exact either way.
    use_incr: bool,
    // i30 (L8, min_level=8 replace): gate for the SPARSE MOVE-EVAL decomposition of
    // the O(wk^2) replace loop in local_search_vnd_tsn. false = dense sentinel loop
    // (byte-identical, clean bp1 timing). true = column-rm materialized from adj[rm].
    // Q byte-exact either way (interaction symmetric). Threaded into TopNeighbors.
    use_sparse_me: bool,
    // i33 (L8, min_level=8 replace): gate for the SPARSE decomposition of the TWO
    // double-swap loops (1rm/2add + 2rm/1add). Distinct from use_sparse_me (single
    // replace loop only). false = dense sentinel. See TopNeighbors.sparse_swap.
    use_sparse_swap: bool,
    // i31 (L8, min_level=8 replace, PIVOT-P1): gate for INCREMENTAL DIRTY-SET WINDOW
    // maintenance in local_search_vnd_tsn. i28/i29/i30 all falsified the sparse
    // interaction-ACCESS family (hardware prefetch hides the strided jagged-Vec reads
    // => ratio ~1.0). The last untouched per-op cost center is the head-of-loop O(n)
    // rescan + TWO select_nth_unstable(~4700) rebuilds of best_unused/worst_used, run
    // ~158K times (80 passes x 180 ils x 11 restarts x refine). false = full O(n)
    // rescan every pass (byte-exact 53375 sentinel). true = after pass 0, rebuild the
    // windows from a small candidate POOL (prev windows + dirty items whose contrib
    // flipped) => no full O(n) scan. Q-tolerant (candidate restriction, moves still
    // feasibility-check => never invalid; periodic full rebuild bounds drift).
    win_incr: bool,
    // i34 (L8, min_level=8 replace): flatten the jagged Vec<Vec<i32>> interaction matrix
    // to a contiguous row-major Vec<i32> (stride=n) for the hot local_search_vnd_tsn
    // reads. Byte-EXACT layout-only change (Hindsight 89203e04). false = jagged sentinel
    // (must reproduce lean bp1 53375). true = flat prefetchable reads. See TopNeighbors.
    use_flat_iv: bool,
    // i35: narrow-precision gate (i32->i16 flat matrix). Default false. See TopNeighbors.
    use_flat_iv16: bool,
    // i36: symmetric-transpose the strided move-eval reads to row-major within-row scans
    // (byte-exact via matrix symmetry). Default false = strided sentinel. See TopNeighbors.
    use_iv_transpose: bool,
    // i37: separable-upper-bound screen of the replace loop (skip provably-losing iv reads,
    // byte-exact). Default false = full dense eval. See TopNeighbors.
    use_bb_screen: bool,
    // i41: sorted early-break of the replace loop (removes iterations, not just reads).
    // Default false = no sort/break. See TopNeighbors.
    use_screen_break: bool,
    // i50: light-VND ports of screen_break + win_incr (see TopNeighbors). Default false.
    use_light_screen_break: bool,
    use_light_win_incr: bool,
    // i51: fused light-VND replace loop with row-contiguous move-eval reads. Default false
    // = unchanged. See TopNeighbors::use_light_fast_replace.
    use_light_fast_replace: bool,
    // i91: r-flip screening of the tsn_light 2-add pair loop (arXiv 2407.21062).
    // 0 = off (sentinel byte-exact), 1 = byte-exact Corollary-1 skip, 2 = Strategy-1
    // contrib-desc monotone break (widened neighborhood). See TopNeighbors::use_rflip_screen.
    use_rflip_screen: u8,
}

impl Hparams {
    fn for_size(n: usize, budget: u32) -> Self {
        if n <= 1200 {
            if budget <= 5 {
                Self {
                    n_random_starts: 3, n_crossover_gen: 8, sa_rounds: 0,
                    sa_iter: 0, n_sa_members: 0, ils_rounds: 60,
                    ils_restart_interval: 10, perturb_base_frac: 4,
                    perturb_max_frac: 5, ils_vnd_level: 0, bounded_2_2_k: 10,
                    n_full_restarts: 5, use_hub_pair: true,
                    use_heavy_polish: false, window_k: 200, core_half_dp: 40,
                    tsn_k: 12, two_ex_cap: 30, use_incr: false, use_sparse_me: false, use_sparse_swap: false, win_incr: false, use_flat_iv: false, use_flat_iv16: false, use_iv_transpose: false, use_bb_screen: false, use_screen_break: false, use_light_screen_break: false, use_light_win_incr: false, use_light_fast_replace: false, use_rflip_screen: 0,
                }
            } else if budget <= 10 {
                Self {
                    n_random_starts: 3, n_crossover_gen: 8, sa_rounds: 0,
                    sa_iter: 0, n_sa_members: 0, ils_rounds: 100,
                    ils_restart_interval: 10, perturb_base_frac: 4,
                    perturb_max_frac: 5, ils_vnd_level: 0, bounded_2_2_k: 10,
                    n_full_restarts: 5, use_hub_pair: true,
                    use_heavy_polish: false, window_k: 200, core_half_dp: 50,
                    tsn_k: 12, two_ex_cap: 30, use_incr: false, use_sparse_me: false, use_sparse_swap: false, win_incr: false, use_flat_iv: false, use_flat_iv16: false, use_iv_transpose: false, use_bb_screen: false, use_screen_break: false, use_light_screen_break: false, use_light_win_incr: false, use_light_fast_replace: false, use_rflip_screen: 0,
                }
            } else {
                Self {
                    n_random_starts: 3, n_crossover_gen: 8, sa_rounds: 0,
                    sa_iter: 0, n_sa_members: 0, ils_rounds: 120,
                    ils_restart_interval: 10, perturb_base_frac: 4,
                    perturb_max_frac: 5, ils_vnd_level: 0, bounded_2_2_k: 10,
                    n_full_restarts: 5, use_hub_pair: true,
                    use_heavy_polish: false, window_k: 200, core_half_dp: 50,
                    tsn_k: 12, two_ex_cap: 30, use_incr: false, use_sparse_me: false, use_sparse_swap: false, win_incr: false, use_flat_iv: false, use_flat_iv16: false, use_iv_transpose: false, use_bb_screen: false, use_screen_break: false, use_light_screen_break: false, use_light_win_incr: false, use_light_fast_replace: false, use_rflip_screen: 0,
                }
            }
        } else {
            Self {
                n_random_starts: 4, n_crossover_gen: 3, sa_rounds: 0,
                sa_iter: 0, n_sa_members: 0, ils_rounds: 50,
                ils_restart_interval: 12, perturb_base_frac: 6,
                perturb_max_frac: 4, ils_vnd_level: 0, bounded_2_2_k: 0,
                n_full_restarts: 16, use_hub_pair: false,
                use_heavy_polish: false, window_k: 180, core_half_dp: 40,
                tsn_k: 12, two_ex_cap: 30, use_incr: false, use_sparse_me: false, use_sparse_swap: false, win_incr: false, use_flat_iv: false, use_flat_iv16: false, use_iv_transpose: false, use_bb_screen: false, use_screen_break: false, use_light_screen_break: false, use_light_win_incr: false, use_light_fast_replace: false, use_rflip_screen: 0,
            }
        }
    }

    fn from_map(h: &Option<Map<String, Value>>, n: usize, budget: u32) -> Self {
        let mut p = Self::for_size(n, budget);
        if let Some(m) = h {
            // i21 COMBINE (min_level=8, source_algo91): transplant the SOTA
            // prom_pulse_marlin_v2 PROVEN t43 deep-intensification schedule
            // (track_elites 53359@83s, same shared marlin knob vocabulary) as a
            // philosophy — many shallow restarts (n_full_restarts=16, ils_rounds=50,
            // window_k=180, core_half_dp=40, n_crossover_gen=3) — and every prior
            // t43 iter built rc/refine ON TOP of that weak base. base_profile=1
            // swaps in the deep-intensification budget allocation (fewer, deeper
            // restarts) which the champion transfer from t42 already proved KEPT
            // on t43 (+0.566%, mem 7ba4993f); our rc/refine layer then runs on top.
            // base_profile absent/0 => byte-identical to the champion path (sentinel
            // {rc_mode:5,rc_fraction:0.30,refine_mode:5} still reproduces 53327).
            // Explicit per-knob overrides below still win (applied after).
            if m.get("base_profile").and_then(|v| v.as_u64()) == Some(1) {
                p.window_k = 270;
                p.ils_rounds = 180;
                p.core_half_dp = 52;
                p.n_crossover_gen = 13;
                p.n_full_restarts = 11;
                p.n_random_starts = 2;
                p.bounded_2_2_k = 14;
                p.sa_rounds = 1;
                p.sa_iter = 0;
                p.n_sa_members = 0;
                p.ils_vnd_level = 0;
                p.perturb_base_frac = 4;
                p.perturb_max_frac = 7;
                p.ils_restart_interval = 17;
            }
            // i22 COMBINE (min_level=8): base_profile=2 = budget-calibrated
            // INTERMEDIATE deep-intensification. base_profile=1 (i21) reached
            // Q=53361 (> SOTA 53359) but at 148.5s = 47s over the 101530ms gate
            // (ARM_C job23918). The over-budget deep schedule is a known failure
            // mode (Hindsight e80d79cf: i45 revoked 131s>85s). Time drivers are
            // the RECOMBINATION generations (n_crossover_gen 3->13) and the DP
            // core (core_half_dp 40->52), NOT window_k (i62/0b02dcef: window_k
            // cut = NO time gain) nor ils_rounds (i43/43d49b5c: stall-capped,
            // cheap). So base_profile=2 keeps the deep character (wide window_k,
            // deep ils, fewer/deeper restarts) but SURGICALLY cuts the two
            // proven time-drivers back toward default to reclaim the ~47s while
            // retaining base_profile=1's +34 Q. Pairs with rc_fraction over-lock
            // (smaller reduced instance -> deep LS runs faster; refine5 F2
            // seeded-escape re-admits mis-locked items -> Q-safe). Explicit
            // per-knob overrides below still win.
            if m.get("base_profile").and_then(|v| v.as_u64()) == Some(2) {
                p.window_k = 270;          // keep deep window (not a time driver, i62)
                p.ils_rounds = 160;        // near-full (stall-capped, cheap, i43)
                p.core_half_dp = 44;       // 52 -> 44: trim DP core (time driver)
                p.n_crossover_gen = 5;     // 13 -> 5: cut recombination gens (main time driver)
                p.n_full_restarts = 11;    // keep fewer/deeper restarts
                p.n_random_starts = 2;
                p.bounded_2_2_k = 14;
                p.sa_rounds = 1;
                p.sa_iter = 0;
                p.n_sa_members = 0;
                p.ils_vnd_level = 0;
                p.perturb_base_frac = 4;
                p.perturb_max_frac = 7;
                p.ils_restart_interval = 17;
            }
            // i28 COMBINE (min_level=8, L4 pivot): base_profile=3 = ILS-PRESERVING
            // reclaim. bp1 (i21) = 53361 > SOTA 53359 but @148.5s (over 101530ms gate).
            // bp2 (i22) reclaimed time by cutting THREE knobs vs bp1: n_crossover_gen
            // 13->5, core_half_dp 52->44, AND ils_rounds 180->160 -> 53357 (-4 Q). The
            // directive's literal L4 (cut only the two "time drivers" n_crossover_gen +
            // core_half_dp) is ALREADY bp2 modulo the ils cut, so it merely reproduces the
            // 53357 record. Thesis (Hindsight 43d49b5c: ils_rounds 50->35 = NO-OP on time,
            // stall-capped; 9aea666a: ils_rounds 50->45 = time-inert but Q slightly DOWN):
            // ils_rounds is PRODUCTIVE depth yet time-cheap -> bp2 threw away bp1's +4 Q by
            // over-cutting the cheap knob. bp3 keeps bp2's TWO true time-driver cuts
            // (n_crossover_gen=5, core_half_dp=44) but RESTORES ils_rounds=180 -> recover
            // Q toward 53361 at ~zero extra time. Everything else identical to bp1/bp2.
            // Explicit per-knob overrides below still win (arms probe the reclaim point).
            if m.get("base_profile").and_then(|v| v.as_u64()) == Some(3) {
                p.window_k = 270;          // keep deep window (not a time driver, i62)
                p.ils_rounds = 180;        // RESTORED 160->180: productive + time-cheap (43d49b5c/9aea666a)
                p.core_half_dp = 44;       // keep bp2 DP-core cut (time driver)
                p.n_crossover_gen = 5;     // keep bp2 recombination cut (main time driver)
                p.n_full_restarts = 11;
                p.n_random_starts = 2;
                p.bounded_2_2_k = 14;
                p.sa_rounds = 1;
                p.sa_iter = 0;
                p.n_sa_members = 0;
                p.ils_vnd_level = 0;
                p.perturb_base_frac = 4;
                p.perturb_max_frac = 7;
                p.ils_restart_interval = 17;
            }
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
            if let Some(v) = m.get("tsn_k").and_then(|v| v.as_u64()) { p.tsn_k = v as usize; }
            if let Some(v) = m.get("two_ex_cap").and_then(|v| v.as_u64()) { p.two_ex_cap = v as usize; }
            // i29: sparse-move gate. Accepts bool or 0/1. Default false = dense sentinel.
            if let Some(v) = m.get("use_incremental_windows") {
                p.use_incr = v.as_bool().unwrap_or_else(|| v.as_u64().map(|x| x != 0).unwrap_or(false));
            }
            // i30: sparse move-eval gate. Accepts bool or 0/1. Default false = dense
            // sentinel (must reproduce bp1 53375 byte-exact for the ratio pair).
            if let Some(v) = m.get("use_sparse_moveeval") {
                p.use_sparse_me = v.as_bool().unwrap_or_else(|| v.as_u64().map(|x| x != 0).unwrap_or(false));
            }
            // i33: sparse double-swap gate. Accepts bool or 0/1. Default false = dense
            // sentinel (must reproduce lean bp1 53375 byte-exact for the ratio pair).
            if let Some(v) = m.get("use_sparse_swap") {
                p.use_sparse_swap = v.as_bool().unwrap_or_else(|| v.as_u64().map(|x| x != 0).unwrap_or(false));
            }
            // i31 (PIVOT-P1): incremental dirty-set window gate. Accepts bool or 0/1.
            // Default false = full O(n) rescan (must reproduce bp1 53375 byte-exact).
            if let Some(v) = m.get("use_win_incr") {
                p.win_incr = v.as_bool().unwrap_or_else(|| v.as_u64().map(|x| x != 0).unwrap_or(false));
            }
            // i34: flat interaction-matrix layout gate. Accepts bool or 0/1. Default
            // false = jagged sentinel (must reproduce lean bp1 53375 byte-exact for the
            // ratio pair). true = contiguous flat Vec<i32> stride=n hot reads.
            if let Some(v) = m.get("use_flat_iv") {
                p.use_flat_iv = v.as_bool().unwrap_or_else(|| v.as_u64().map(|x| x != 0).unwrap_or(false));
            }
            // i35: narrow i16 interaction-matrix gate. Accepts bool or 0/1. Default false.
            // true = contiguous i16 flat matrix (50MB) hot reads (byte-exact, values≤1000).
            if let Some(v) = m.get("use_flat_iv16") {
                p.use_flat_iv16 = v.as_bool().unwrap_or_else(|| v.as_u64().map(|x| x != 0).unwrap_or(false));
            }
            if let Some(v) = m.get("use_iv_transpose") {
                p.use_iv_transpose = v.as_bool().unwrap_or_else(|| v.as_u64().map(|x| x != 0).unwrap_or(false));
            }
            if let Some(v) = m.get("use_bb_screen") {
                p.use_bb_screen = v.as_bool().unwrap_or_else(|| v.as_u64().map(|x| x != 0).unwrap_or(false));
            }
            // i41: sorted early-break of the replace loop. Accepts bool or 0/1. Default false.
            // true = sort best_unused desc by contrib + break inner scan on ub<=incumbent.
            if let Some(v) = m.get("use_screen_break") {
                p.use_screen_break = v.as_bool().unwrap_or_else(|| v.as_u64().map(|x| x != 0).unwrap_or(false));
            }
            // i50: light-path ports (default false => champion sentinel byte-exact).
            if let Some(v) = m.get("use_light_screen_break") {
                p.use_light_screen_break = v.as_bool().unwrap_or_else(|| v.as_u64().map(|x| x != 0).unwrap_or(false));
            }
            if let Some(v) = m.get("use_light_win_incr") {
                p.use_light_win_incr = v.as_bool().unwrap_or_else(|| v.as_u64().map(|x| x != 0).unwrap_or(false));
            }
            if let Some(v) = m.get("use_rflip_screen") { p.use_rflip_screen = v.as_u64().unwrap_or(0) as u8; }
            if let Some(v) = m.get("use_light_fast_replace") {
                p.use_light_fast_replace = v.as_bool().unwrap_or_else(|| v.as_u64().map(|x| x != 0).unwrap_or(false));
            }
        }
        p
    }
}

/// i23 (min_level=8, replace): DECOUPLE refine-depth from base-depth.
///
/// Decisive i21 data (ver4162): the pure deep base `{rc0,refine0,base_profile:1}`
/// = 53110 @ 84s (≈ prom_pulse SOTA's 83s), but `{rc5,0.30,refine5,base_profile:1}`
/// = 53361 @ 148.5s. Our rc/refine chain adds +251 Q but +64.5s ON THE DEEP BASE.
/// The sink is that refine_mode=5 (F1 re-anchor → F2 escape) re-runs the FULL deep
/// `large_n_restarts` a SECOND time (line ~3391) plus deep seeded escapes — an
/// "equal-time" second deep restart pass (Hindsight 64be75db: equal-time strategies
/// fail; put 70 % of the budget in the productive LS, not a duplicated deep restart;
/// e687a4f6: only ~25-30 % should go to restart diversification). The re-anchor's
/// VALUE is re-REDUCING the frontier on the refined incumbent (cheap), not a deep
/// re-solve. So: keep the INITIAL solve deep (Q source), run the refine re-solves on
/// a SHALLOW schedule (window_k / core_half_dp / tsn UNCHANGED — same LS depth per
/// pass — only the restart/crossover/ils COUNTS are cut) to reclaim the ~47s.
///
/// `level=0` (refine_shallow absent) returns hp unchanged ⇒ the champion / sentinel
/// path `{rc5,0.30,refine5}` stays byte-exact (53327 drift-guard).
#[inline(always)]
fn shallow_refine_hp(hp: &Hparams, level: u64) -> Hparams {
    let mut r = *hp;
    match level {
        // levels 1/2 (i22, REJECTED): ALSO cut ils_rounds (160->60/40). That is the
        // BUG — ils_rounds is the productive per-pass VND *depth* (Q-source), not
        // diversification. bp2+refine_shallow=1 collapsed to 53042 (< the bare deep
        // base 53110). Kept only for reproducibility of i22's null.
        1 => { r.n_full_restarts = 3; r.n_crossover_gen = 3; r.ils_rounds = 60; }
        2 => { r.n_full_restarts = 2; r.n_crossover_gen = 2; r.ils_rounds = 40; }
        // i24 (min_level=8, REPLACE) — DIVERSIFICATION-ONLY shallow schedule. The
        // refine re-solves (F1 re-anchor `large_n_restarts` + F2 seeded escape) each
        // re-run the FULL n_full_restarts=16 outer restart pass = the +47s duplicated
        // deep-restart sink (comment ~l.1958). But the deep base already did that
        // diversification; the re-anchor's VALUE is re-REDUCING the frontier on the
        // refined incumbent (cheap), NOT a second deep multi-restart (Hindsight
        // 64be75db: 70% of the budget to the productive LS, not a duplicated deep
        // restart; e687a4f6: only 25-30% to restart diversification). So cut ONLY the
        // restart/crossover/random-start COUNTS and KEEP the per-pass VND depth
        // (ils_rounds / window_k / core_half_dp / tsn_k UNTOUCHED). Each re-anchor
        // restart still descends to full local depth — just far fewer of them.
        // Distinct from i22: SAME search depth, fewer diversification restarts.
        // level 3 = aggressive (nfr 16->2, ncg->1, nrs->2); level 4 = moderate.
        3 => { r.n_full_restarts = 2; r.n_crossover_gen = 1; r.n_random_starts = 2; }
        4 => { r.n_full_restarts = 4; r.n_crossover_gen = 2; r.n_random_starts = 3; }
        // i25 (min_level=8, REPLACE): level 5 = ULTRA-minimal restart count — F1
        // re-anchor becomes essentially "re-reduce the frontier + one deep-depth
        // restart polish", NOT a diversification pass. The deep base already spent
        // the restart-diversification budget; Hindsight dec86d81 caps restart
        // diversification at 15-30% of budget, so the F1 duplicate deep multi-restart
        // is pure waste. ils_rounds/window_k/core_half_dp/tsn_k UNTOUCHED ⇒ that
        // single restart still descends to FULL local depth. Frees F1's time to be
        // reinvested into the productive F2 seeded escape (Hindsight b1080370:
        // stripping ILS depth from a seeded restart cost -57Q → keep escape depth,
        // add escape passes instead).
        5 => { r.n_full_restarts = 1; r.n_crossover_gen = 0; r.n_random_starts = 1; }
        _ => {}
    }
    r
}

fn local_search_vnd_tsn_light(state: &mut State, tsn: &TopNeighbors) {
    let n = state.ch.num_items;
    let wk: usize = if n > 3000 { 200 } else { 300 };
    let mut unused_buf: Vec<(usize, i64, i64)> = Vec::with_capacity(n);
    let mut used_buf: Vec<(usize, i64, i64)> = Vec::with_capacity(n);
    let mut best_unused: Vec<usize> = Vec::with_capacity(wk);
    let mut worst_used: Vec<usize> = Vec::with_capacity(wk);
    // i50: screen_break scratch (light port of i41). Allocated ONLY on the light-break
    // path so the sentinel (use_light_screen_break=false) is byte-identical / clean-timed.
    let mut bu_sorted: Vec<usize> = if tsn.use_light_screen_break { Vec::with_capacity(wk) } else { Vec::new() };
    // i91: mode-2 scratch (contrib-desc order for the 2-add outer loop). Allocated ONLY
    // on the rflip mode-2 path so sentinel and mode-1 are untouched.
    let mut bu2: Vec<usize> = if tsn.use_rflip_screen == 2 { Vec::with_capacity(wk) } else { Vec::new() };
    // i50: incremental dirty-set window scratch (light port of i31 win_incr). Allocated
    // ONLY on the win_incr path so the sentinel (use_light_win_incr=false) is untouched.
    let mut dirty: Vec<usize> = Vec::new();
    let mut in_pool: Vec<bool> = if tsn.use_light_win_incr { vec![false; n] } else { Vec::new() };
    let mut pass: usize = 0;
    for _ in 0..40 {
        // i50: on the sentinel (use_light_win_incr=false) this is ALWAYS a full O(n)
        // rescan == the original code. On the win_incr path, rebuild the windows from a
        // SMALL candidate pool (prev windows ∪ dirty) except on pass 0 and every 8th pass
        // (full re-seed to bound Q-drift). Mirrors the proven heavy-VND machinery.
        let full_rebuild = !tsn.use_light_win_incr || pass == 0 || pass % 8 == 0;
        unused_buf.clear();
        used_buf.clear();
        if full_rebuild {
            for i in 0..n {
                let c = state.contrib[i] as i64;
                let w = (state.ch.weights[i] as i64).max(1);
                if state.selected_bit[i] { used_buf.push((i, c, w)); } else { unused_buf.push((i, c, w)); }
            }
        } else {
            for &i in best_unused.iter().chain(worst_used.iter()).chain(dirty.iter()) {
                if !in_pool[i] {
                    in_pool[i] = true;
                    let c = state.contrib[i] as i64;
                    let w = (state.ch.weights[i] as i64).max(1);
                    if state.selected_bit[i] { used_buf.push((i, c, w)); } else { unused_buf.push((i, c, w)); }
                }
            }
            for t in unused_buf.iter() { in_pool[t.0] = false; }
            for t in used_buf.iter() { in_pool[t.0] = false; }
        }
        dirty.clear();
        best_unused.clear();
        worst_used.clear();
        pass += 1;
        let ku = wk.min(unused_buf.len());
        if ku > 0 && ku < unused_buf.len() {
            unused_buf.select_nth_unstable_by(ku - 1, |a, b| (b.1 * a.2).cmp(&(a.1 * b.2)));
        }
        let ks = wk.min(used_buf.len());
        if ks > 0 && ks < used_buf.len() {
            used_buf.select_nth_unstable_by(ks - 1, |a, b| (a.1 * b.2).cmp(&(b.1 * a.2)));
        }
        for t in &unused_buf[..ku] { best_unused.push(t.0); }
        for t in &used_buf[..ks] { worst_used.push(t.0); }

        let slack = state.slack();
        if slack > 0 {
            let mut ba: Option<(usize, i32)> = None;
            for &c in &best_unused {
                if state.ch.weights[c] > slack { continue; }
                let d = state.contrib[c];
                if d > 0 && ba.map_or(true, |(_, bd)| d > bd) { ba = Some((c, d)); }
            }
            if let Some((c, _)) = ba {
                state.add_item(c);
                if tsn.use_light_win_incr { mark_dirty(&mut dirty, tsn, &[c]); }
                continue;
            }
        }

        {
            let mut bs: Option<(usize, usize, i32)> = None;
            if tsn.use_light_fast_replace {
                // i51: fused sort+early-break WITH row-contiguous move-eval reads.
                // Identical sort-desc-by-contrib + separable early-break as
                // use_light_screen_break (same order, same deltas, same tie-break =>
                // byte-exact), but the surviving `iv` reads walk row `rm` CONTIGUOUSLY:
                // rm is the fixed outer index (its flat row base `rm*n` is hoisted once)
                // and c steps +1 in memory (stride 4B, prefetchable) instead of the
                // screen_break path's I[c][rm] (c-outer => stride n*4B cache-thrash).
                // I[rm][c] == I[c][rm] by matrix symmetry, so Q is identical.
                bu_sorted.clear();
                bu_sorted.extend_from_slice(&best_unused);
                bu_sorted.sort_by(|&x, &y| state.contrib[y].cmp(&state.contrib[x]));
                let use_flat = tsn.use_flat_iv;
                let nrow = tsn.n;
                for &rm in &worst_used {
                    let max_w = state.ch.weights[rm] + state.slack();
                    let crm = state.contrib[rm];
                    // hoist the contiguous row of I once per rm.
                    let flat_row: &[i32] = if use_flat { &tsn.flat_iv[rm * nrow..(rm + 1) * nrow] } else { &[] };
                    for &c in &bu_sorted {
                        let ub = state.contrib[c] - crm;
                        match bs { Some((_, _, bd)) => { if ub <= bd { break; } }, None => { if ub <= 0 { break; } } }
                        if state.ch.weights[c] > max_w { continue; }
                        // contiguous read I[rm][c] == I[c][rm] (symmetry); byte-exact delta.
                        let ivrc = if use_flat { unsafe { *flat_row.get_unchecked(c) } } else { state.ch.interaction_values[rm][c] };
                        let d = ub - ivrc;
                        if d > 0 && bs.map_or(true, |(_, _, bd)| d > bd) { bs = Some((c, rm, d)); }
                    }
                }
            } else if tsn.use_light_screen_break {
                // i50: sorted early-BREAK port of i41 onto the light-VND replace loop —
                // the REAL n=5000 ILS hotspot. Sort best_unused DESC by contrib into
                // bu_sorted; ub = contrib[c]-crm is then non-increasing so once ub <= the
                // incumbent best_d every remaining c is a proven non-improver ⇒ break.
                // iv_mv(rm,c) == iv(c,rm) (matrix symmetry) so the delta is identical to
                // the dense path — only the tie-break winner among equal-max-d moves can
                // differ (same directive tolerance as the heavy screen_break).
                bu_sorted.clear();
                bu_sorted.extend_from_slice(&best_unused);
                bu_sorted.sort_by(|&x, &y| state.contrib[y].cmp(&state.contrib[x]));
                for &rm in &worst_used {
                    let max_w = state.ch.weights[rm] + state.slack();
                    let crm = state.contrib[rm];
                    for &c in &bu_sorted {
                        let ub = state.contrib[c] - crm;
                        match bs { Some((_, _, bd)) => { if ub <= bd { break; } }, None => { if ub <= 0 { break; } } }
                        if state.ch.weights[c] > max_w { continue; }
                        let d = ub - tsn.iv_mv(state.ch, rm, c);
                        if d > 0 && bs.map_or(true, |(_, _, bd)| d > bd) { bs = Some((c, rm, d)); }
                    }
                }
            } else {
                for &rm in &worst_used {
                    let max_w = state.ch.weights[rm] + state.slack();
                    for &c in &best_unused {
                        if state.ch.weights[c] > max_w { continue; }
                        let d = state.contrib[c] - state.contrib[rm] - tsn.iv(state.ch, c, rm);
                        if d > 0 && bs.map_or(true, |(_, _, bd)| d > bd) { bs = Some((c, rm, d)); }
                    }
                }
            }
            if let Some((c, rm, _)) = bs {
                state.replace_item(rm, c);
                if tsn.use_light_win_incr { mark_dirty(&mut dirty, tsn, &[c, rm]); }
                continue;
            }
        }

        {
            let slack_i = state.slack() as i32;
            if slack_i >= 2 {
                let mut bd = 0i64;
                let mut bp = None;
                // i91: cu2max = max(0, max contrib over unused items with weight < slack)
                // — an upper bound on contrib[a2] for every LEGAL a2 (wa2 <= slack-wa1 <
                // slack). One contiguous O(n) scan, only on the rflip paths. NOTE: a2
                // ranges over friends[a1] (possibly OUTSIDE the window), so the in-window
                // add-exhaustion argument alone does NOT bound ca2 — this scan does.
                let cu2max: i64 = if tsn.use_rflip_screen > 0 {
                    let mut m = 0i64;
                    for i in 0..state.ch.num_items {
                        if !state.selected_bit[i] && (state.ch.weights[i] as i32) < slack_i {
                            let c = state.contrib[i] as i64;
                            if c > m { m = c; }
                        }
                    }
                    m
                } else { 0 };
                if tsn.use_rflip_screen == 1 {
                    // i91 mode 1: Corollary-1 skip, byte-exact. Same iteration order and
                    // same heuristic break as the sentinel; ONLY provable non-improvers
                    // (delta <= ca1 + cu2max + phi_a1 <= bd, strict `> bd` update) are
                    // skipped, so bd/bp evolve identically => identical move, identical Q.
                    for &a1 in &best_unused {
                        let ca1 = state.contrib[a1] as i64;
                        if ca1 <= 0 && bd > 0 { break; }
                        if ca1 + cu2max + (tsn.phi_a1[a1] as i64) <= bd { continue; }
                        let wa1 = state.ch.weights[a1] as i32;
                        if wa1 >= slack_i { continue; }
                        for &a2 in &tsn.friends[a1] {
                            if state.selected_bit[a2] || a1 == a2 { continue; }
                            if wa1 + (state.ch.weights[a2] as i32) <= slack_i {
                                let delta = ca1 + state.contrib[a2] as i64 + tsn.iv(state.ch, a1, a2) as i64;
                                if delta > bd { bd = delta; bp = Some((a1, a2)); }
                            }
                        }
                    }
                } else if tsn.use_rflip_screen == 2 {
                    // i91 mode 2 (Strategy 1): contrib-DESC outer order => the bound
                    // ca1 + cu2max + phi_max is non-increasing => MONOTONE break. The
                    // sentinel's heuristic `ca1<=0 && bd>0` break is dropped, so the
                    // screened neighborhood is a SUPERSET of the sentinel's (never
                    // prunes an improving pair; may FIND pairs the sentinel missed).
                    // Q is bench-gated (>= 53359), not byte-exact by construction.
                    bu2.clear();
                    bu2.extend_from_slice(&best_unused);
                    bu2.sort_by(|&x, &y| state.contrib[y].cmp(&state.contrib[x]));
                    let phimax = tsn.phi_max as i64;
                    for &a1 in &bu2 {
                        let ca1 = state.contrib[a1] as i64;
                        if ca1 + cu2max + phimax <= bd { break; }
                        if ca1 + cu2max + (tsn.phi_a1[a1] as i64) <= bd { continue; }
                        let wa1 = state.ch.weights[a1] as i32;
                        if wa1 >= slack_i { continue; }
                        for &a2 in &tsn.friends[a1] {
                            if state.selected_bit[a2] || a1 == a2 { continue; }
                            if wa1 + (state.ch.weights[a2] as i32) <= slack_i {
                                let delta = ca1 + state.contrib[a2] as i64 + tsn.iv(state.ch, a1, a2) as i64;
                                if delta > bd { bd = delta; bp = Some((a1, a2)); }
                            }
                        }
                    }
                } else {
                    for &a1 in &best_unused {
                        let ca1 = state.contrib[a1] as i64;
                        if ca1 <= 0 && bd > 0 { break; }
                        let wa1 = state.ch.weights[a1] as i32;
                        if wa1 >= slack_i { continue; }
                        for &a2 in &tsn.friends[a1] {
                            if state.selected_bit[a2] || a1 == a2 { continue; }
                            if wa1 + (state.ch.weights[a2] as i32) <= slack_i {
                                let delta = ca1 + state.contrib[a2] as i64 + tsn.iv(state.ch, a1, a2) as i64;
                                if delta > bd { bd = delta; bp = Some((a1, a2)); }
                            }
                        }
                    }
                }
                if let Some((a1, a2)) = bp {
                    state.add_item(a1); state.add_item(a2);
                    if tsn.use_light_win_incr { mark_dirty(&mut dirty, tsn, &[a1, a2]); }
                    continue;
                }
            }
        }

        break;
    }
}

fn vnd_v2(state: &mut State, hp: &Hparams, tsn: Option<&TopNeighbors>) {
    if let Some(t) = tsn {
        local_search_vnd_tsn(state, t, hp.two_ex_cap);
    } else if hp.window_k < state.ch.num_items {
        local_search_vnd_windowed(state, hp.window_k);
    } else {
        ils_vnd(state, hp);
    }
}

fn polish_v2(state: &mut State, hp: &Hparams, tsn: Option<&TopNeighbors>) {
    if let Some(t) = tsn {
        local_search_vnd_tsn(state, t, hp.two_ex_cap);
    } else if hp.use_heavy_polish {
        local_search_vnd_heavy(state);
    } else {
        vnd_v2(state, hp, None);
    }
}

fn run_one_instance(challenge: &Challenge, hp: &Hparams, rng_offset: usize, shared_tsn: Option<&TopNeighbors>, total_interactions: &[i64]) -> (Solution, i64) {
    run_one_instance_seeded(challenge, hp, rng_offset, shared_tsn, total_interactions, None)
}

fn run_one_instance_seeded(challenge: &Challenge, hp: &Hparams, rng_offset: usize, shared_tsn: Option<&TopNeighbors>, total_interactions: &[i64], seed_sol: Option<&Solution>) -> (Solution, i64) {
    let n = challenge.num_items;
    let mut rng = Rng::from_seed(&challenge.seed);
    for _ in 0..rng_offset * 100 { rng.next_u32(); }
    let ch = hp.core_half_dp;

    let tsn_opt: Option<TopNeighbors> = if shared_tsn.is_none() && n > 1200 {
        Some(TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen))
    } else { None };
    let tsn_ref = shared_tsn.or(tsn_opt.as_ref());

    // i29: route every state built in this (hot) per-instance solver through the
    // sparse-move path when enabled. `tsn_ref` is Option<&TopNeighbors> (Copy), so
    // the closure captures a pointer copy; TopNeighbors.adj outlives all these states.
    let attach = |st: &mut State| {
        if hp.use_incr {
            if let Some(t) = tsn_ref { st.attach_adj(&t.adj); }
        }
    };

    let mut population: Vec<SolState> = Vec::with_capacity(16);

    if let Some(seed) = seed_sol {
        let mut st = State::new_empty(challenge);
        attach(&mut st);
        for &i in &seed.items { st.add_item(i); }
        let sel = st.selected_items();
        let n_remove = (sel.len() / 8).max(2).min(sel.len());
        let mut scored: Vec<(usize, i64)> = sel.iter().map(|&i| {
            let w = (challenge.weights[i] as i64).max(1);
            (i, (st.contrib[i] as i64 * 1000) / w)
        }).collect();
        scored.sort_unstable_by_key(|&(_, s)| s);
        for j in 0..n_remove { st.remove_item(scored[j].0); }
        greedy_reconstruct(&mut st, rng_offset % 6, total_interactions);
        dp_refinement_hp(&mut st, ch);
        vnd_v2(&mut st, hp, tsn_ref);
        population.push(st.clone_solution());
    }

    let n_greedy = if n <= 1200 { 4 } else { 3 };
    let n_rand = if seed_sol.is_some() { hp.n_random_starts.saturating_sub(1) } else { hp.n_random_starts };
    for variant in 0..n_greedy {
        let mut st = State::new_empty(challenge);
        attach(&mut st);
        match variant {
            0 => if n > 1200 { build_greedy_density_from_all(&mut st, total_interactions) } else { build_greedy_density(&mut st) },
            1 => build_greedy_value(&mut st),
            2 => build_greedy_synergy_weight(&mut st, total_interactions),
            _ => build_greedy_hub(&mut st, total_interactions),
        }
        dp_refinement_hp(&mut st, ch);
        polish_v2(&mut st, hp, tsn_ref);
        population.push(st.clone_solution());
    }

    for mode in 4..(4 + n_rand) {
        let mut st = State::new_empty(challenge);
        attach(&mut st);
        let m = if mode < 6 { mode } else { mode - 2 };
        construct_forward_incremental(&mut st, m, &mut rng);
        dp_refinement_hp(&mut st, ch);
        vnd_v2(&mut st, hp, tsn_ref);
        population.push(st.clone_solution());
    }

    if hp.use_hub_pair {
        for k in 0..4 {
            let mut st = State::new_empty(challenge);
            attach(&mut st);
            build_hub_pair_kth(&mut st, k);
            dp_refinement_hp(&mut st, ch);
            vnd_v2(&mut st, hp, tsn_ref);
            population.push(st.clone_solution());
        }
    }

    population.sort_unstable_by_key(|s| std::cmp::Reverse(s.value));
    population.truncate(8);

    let mut state = State::new_empty(challenge);
    attach(&mut state);
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
    let max_stall = if n > 1200 { (hp.ils_rounds / 3).min(8) } else { (hp.ils_rounds * 2 / 3).min(15) };
    let use_light_vnd = n > 1200 && tsn_ref.is_some();
    for round in 0..hp.ils_rounds {
        if stall_count >= max_stall { break; }
        let snap = state.clone_solution();

        dp_refinement_hp(&mut state, ch);
        if use_light_vnd {
            local_search_vnd_tsn_light(&mut state, tsn_ref.unwrap());
        } else {
            vnd_v2(&mut state, hp, tsn_ref);
        }

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
                perturb_by_strategy(&mut state, strength, stall_count, strategy, &mut rng, hp, total_interactions);
            }
            greedy_reconstruct(&mut state, (round % 8) % 10, total_interactions);
            if use_light_vnd {
                local_search_vnd_tsn_light(&mut state, tsn_ref.unwrap());
            } else {
                vnd_v2(&mut state, hp, tsn_ref);
            }

            let h = compute_hash(&state.selected_bit);
            if tabu_hashes.contains(&h) {
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
            local_search_vnd_tsn(&mut final_state, t, hp.two_ex_cap);
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
        let v = final_state.total_value;
        (Solution { items: final_state.selected_items() }, v)
    } else {
        (Solution { items: best_sel }, best_val)
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
    let total_moves = to_add.len() + to_remove.len();
    let checkpoint_interval = (total_moves / 4).max(3);
    let mut move_count = 0usize;

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
        move_count += 1;

        if state.total_weight <= cap && state.total_value > best_val {
            best_val = state.total_value;
            best_bits = state.selected_bit.clone();
        }

        if move_count % checkpoint_interval == 0 && state.total_weight <= cap {
            let mut tmp = State::new_empty(challenge);
            for i in 0..n { if state.selected_bit[i] { tmp.add_item(i); } }
            local_search_vnd_fast(&mut tmp);
            dp_refinement_hp(&mut tmp, hp.core_half_dp);
            if tmp.total_value > best_val {
                best_val = tmp.total_value;
                best_bits = tmp.selected_bit.clone();
            }
        }
    }

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

/// i14 (min_level=8, combine): EXACT RESTRICTED-CORE polish.
///
/// The active VND `local_search_vnd_tsn` is a best-improvement descent over atomic
/// <=2-in / <=2-out moves: every accepted move has delta>0, so any net-positive
/// COORDINATED k-flip (k>2, or one that must transit an intermediate loss) is
/// unreachable — the exact barrier the t43 meta-constat blames for the stuck +32.
/// Width (i9 tsn_k/two_ex_cap), count (i10 basins), composition-order (i11) and
/// oscillation (i13) all failed to cross it. Kernelisation crosses it EXACTLY:
/// freeze all but the ~K most UNCERTAIN items (worst-margin selected + best-margin
/// unselected) and solve the resulting K-item QKP sub-problem to OPTIMALITY by
/// exhaustive enumeration within the residual capacity, keep-best, iterated.
/// Hindsight 8c12c4e2: exact QKP is trivial for K<=~1500 (here K<=18);
/// c6dfea33: a QKP local optimum need not be a subset local optimum, so an exact
/// core solve reaches combinations atomic <=2-moves cannot. Deterministic (core
/// picked by contrib ranking, no RNG); feasible by construction (subsets are
/// enumerated only within residual capacity, and eval_feasible re-guards keep-best).
fn exact_core_polish(
    challenge: &Challenge,
    seed: &Solution,
    core_k_sel: usize,
    core_k_uns: usize,
    core_cap: usize,
    rounds: usize,
) -> Solution {
    let n = challenge.num_items;
    let mut best = seed.clone();
    let mut best_q = eval_feasible(challenge, &best);

    for _ in 0..rounds {
        // Rebuild a State from the current incumbent (contrib[i] = value_i + sum
        // over selected j of iv[i][j]).
        let mut st = State::new_empty(challenge);
        for &i in &best.items { st.add_item(i); }

        // Selected ranked by ASCENDING margin (cheapest to drop), unselected by
        // DESCENDING margin (best to add).
        let mut sel: Vec<usize> = (0..n).filter(|&i| st.selected_bit[i]).collect();
        let mut uns: Vec<usize> = (0..n).filter(|&i| !st.selected_bit[i]).collect();
        sel.sort_unstable_by_key(|&i| st.contrib[i]);
        uns.sort_unstable_by_key(|&i| -(st.contrib[i] as i64));
        sel.truncate(core_k_sel);
        uns.truncate(core_k_uns);

        let mut core: Vec<usize> = Vec::with_capacity(core_k_sel + core_k_uns);
        core.extend_from_slice(&sel);
        core.extend_from_slice(&uns);
        if core.len() > core_cap { core.truncate(core_cap); }
        let k = core.len();
        if k == 0 { break; }

        // Free the SELECTED core items so st.contrib[i] becomes the exact linear
        // coefficient of item i in the core sub-problem (value_i + interactions
        // with the FROZEN selected set only).
        for &i in &core { if st.selected_bit[i] { st.remove_item(i); } }
        let residual = st.slack() as u64;

        let lin: Vec<i64> = core.iter().map(|&i| st.contrib[i] as i64).collect();
        let wt: Vec<u64> = core.iter().map(|&i| challenge.weights[i] as u64).collect();
        let mut pw = vec![0i64; k * k];
        for a in 0..k {
            let ra = &challenge.interaction_values[core[a]];
            for b in (a + 1)..k {
                pw[a * k + b] = ra[core[b]] as i64;
            }
        }

        // Exhaustive optimum over the 2^k core subsets within residual capacity.
        let mut best_mask: u32 = 0;
        let mut best_val: i64 = i64::MIN;
        let full: u32 = 1u32 << k;
        for mask in 0..full {
            let mut w: u64 = 0;
            let mut m = mask;
            while m != 0 { w += wt[m.trailing_zeros() as usize]; m &= m - 1; }
            if w > residual { continue; }
            let mut val: i64 = 0;
            let mut m1 = mask;
            while m1 != 0 {
                let a = m1.trailing_zeros() as usize;
                val += lin[a];
                let mut m2 = m1 & (m1 - 1);
                while m2 != 0 {
                    val += pw[a * k + m2.trailing_zeros() as usize];
                    m2 &= m2 - 1;
                }
                m1 &= m1 - 1;
            }
            if val > best_val { best_val = val; best_mask = mask; }
        }

        // Reconstruct: frozen-selected + chosen core items.
        let mut items: Vec<usize> = st.selected_items();
        for b in 0..k { if best_mask & (1u32 << b) != 0 { items.push(core[b]); } }
        let cand = Solution { items };
        let q = eval_feasible(challenge, &cand);
        if q > best_q { best_q = q; best = cand; } else { break; }
    }
    best
}

/// i20 (F_X POOL-DISAGREEMENT exact core, min_level=8 combine).
///
/// The margin-ranked exact core (i13/i14 `exact_core_polish`) and the atomic
/// <=2-move VND both tie 53327: they kernelize the exact 2^k solve by the SINGLE
/// incumbent's local margin, so the block only ever re-optimizes items the champion
/// is already locally unsure about — inside the SAME basin. The restart `pool`
/// (`large_n_restarts`) holds structurally DISTINCT basins; the items whose
/// membership DISAGREES across those basins are the genuine >=3-item coordinated-move
/// frontier a margin core cannot see (mem `5c6b51ae`: distinct basins are the only
/// proven Q-mover on t43; `c6dfea33`: a QKP local optimum need not be a subset local
/// optimum — an exact block solve can co-locate a coordinated flip the descent misses).
/// Kernelize the exact core on POOL DISAGREEMENT (items selected in some basins,
/// absent in others, ranked by MAX disagreement `|2*freq - m|` ascending, tie-broken
/// by heavier weight = bigger coordinated move), then exact-solve that <=core_cap
/// block seeded from the champion. Keep-best => never below `seed` (downside bounded,
/// deterministic, 32/32 preserved).
fn pool_disagreement_core_polish(
    challenge: &Challenge,
    seed: &Solution,
    basins: &[Vec<bool>],
    core_cap: usize,
    rounds: usize,
) -> Solution {
    let n = challenge.num_items;
    let m = basins.len();
    let mut best = seed.clone();
    let mut best_q = eval_feasible(challenge, &best);
    if m == 0 || core_cap == 0 { return best; }

    // Disagreement frequency across the pool basins.
    let mut freq = vec![0u32; n];
    for b in basins {
        for i in 0..n { if b[i] { freq[i] += 1; } }
    }
    // Swing items: selected in some basins, absent in others. Closest to a 50/50
    // split first (smallest |2*freq - m|); tie-break by heavier weight.
    let mm = m as i64;
    let mut swing: Vec<usize> = (0..n)
        .filter(|&i| freq[i] > 0 && (freq[i] as usize) < m)
        .collect();
    swing.sort_unstable_by(|&a, &b| {
        let da = (2 * freq[a] as i64 - mm).abs();
        let db = (2 * freq[b] as i64 - mm).abs();
        da.cmp(&db).then(challenge.weights[b].cmp(&challenge.weights[a]))
    });
    if swing.is_empty() { return best; }
    // 2^k enumeration => keep k <= 20 (matches exact_core_polish budget usage).
    let core: Vec<usize> = swing.into_iter().take(core_cap.min(20)).collect();
    let k = core.len();

    for _ in 0..rounds {
        // State from the current incumbent; free the SELECTED core items so
        // contrib[i] is the exact linear coefficient vs the FROZEN complement.
        let mut st = State::new_empty(challenge);
        for &i in &best.items { st.add_item(i); }
        for &i in &core { if st.selected_bit[i] { st.remove_item(i); } }
        let residual = st.slack() as u64;

        let lin: Vec<i64> = core.iter().map(|&i| st.contrib[i] as i64).collect();
        let wt: Vec<u64> = core.iter().map(|&i| challenge.weights[i] as u64).collect();
        let mut pw = vec![0i64; k * k];
        for a in 0..k {
            let ra = &challenge.interaction_values[core[a]];
            for b in (a + 1)..k {
                pw[a * k + b] = ra[core[b]] as i64;
            }
        }

        // Exhaustive optimum over the 2^k core subsets within residual capacity.
        let mut best_mask: u32 = 0;
        let mut best_val: i64 = i64::MIN;
        let full: u32 = 1u32 << k;
        for mask in 0..full {
            let mut w: u64 = 0;
            let mut mm2 = mask;
            while mm2 != 0 { w += wt[mm2.trailing_zeros() as usize]; mm2 &= mm2 - 1; }
            if w > residual { continue; }
            let mut val: i64 = 0;
            let mut m1 = mask;
            while m1 != 0 {
                let a = m1.trailing_zeros() as usize;
                val += lin[a];
                let mut m3 = m1 & (m1 - 1);
                while m3 != 0 {
                    val += pw[a * k + m3.trailing_zeros() as usize];
                    m3 &= m3 - 1;
                }
                m1 &= m1 - 1;
            }
            if val > best_val { best_val = val; best_mask = mask; }
        }

        let mut items: Vec<usize> = st.selected_items();
        for b in 0..k { if best_mask & (1u32 << b) != 0 { items.push(core[b]); } }
        let cand = Solution { items };
        let q = eval_feasible(challenge, &cand);
        if q > best_q { best_q = q; best = cand; } else { break; }
    }
    best
}

/// Build full-space selection bitsets for the restart `pool` (reduced-space
/// solutions mapped through `survivors`) plus an extra incumbent basin. Shared by
/// the refine 130/131/132 arms so the disagreement frontier is computed identically.
fn build_pool_basins(
    challenge: &Challenge,
    pool: &[(Solution, i64)],
    survivors: &[usize],
    extra: &Solution,
) -> Vec<Vec<bool>> {
    let n = challenge.num_items;
    let mut basins: Vec<Vec<bool>> = Vec::with_capacity(pool.len() + 1);
    for (s, _) in pool {
        let mut bits = vec![false; n];
        for &kk in &s.items { bits[survivors[kk]] = true; }
        basins.push(bits);
    }
    let mut bits = vec![false; n];
    for &i in &extra.items { bits[i] = true; }
    basins.push(bits);
    basins
}

/// i18 (F_R/F_S) — POTENTIAL-GUIDED exact core polish. Identical exact 2^k core
/// solve as `exact_core_polish` (i14), but the UNSELECTED half of the core is
/// chosen by the surrogate-UB POTENTIAL `total_interactions[i] / w_i` (the code's
/// own provable marginal upper bound, l.3707), NOT by the item's CURRENT `contrib`.
/// Rationale (53327-plateau diagnosis): an item in the 53359 optimum looks
/// marginal in OUR configuration (low current contrib -> i14/i15 never place it in
/// the core), yet its surrogate potential is high; ranking by potential surfaces
/// exactly those items so the exact solve can co-locate an "eject-low-margin /
/// admit-high-potential" swap. The SELECTED half is still the lowest current-margin
/// items (cheapest to free capacity). `band_mode`: 0 = globally highest potential;
/// 1 = potential CLOSEST to the marginal selected item's potential (frontier band,
/// Hindsight a56c42e6). Keep-best => never below the seed.
fn potential_core_polish(
    challenge: &Challenge,
    seed: &Solution,
    total_interactions: &[i64],
    core_k_sel: usize,
    core_k_uns: usize,
    core_cap: usize,
    rounds: usize,
    band_mode: usize,
) -> Solution {
    let n = challenge.num_items;
    let pot = |i: usize| -> f64 {
        total_interactions[i] as f64 / (challenge.weights[i] as f64).max(1.0)
    };
    let mut best = seed.clone();
    let mut best_q = eval_feasible(challenge, &best);

    for _ in 0..rounds {
        let mut st = State::new_empty(challenge);
        for &i in &best.items { st.add_item(i); }

        // Selected core = lowest CURRENT margin (cheapest to eject to free capacity).
        let mut sel: Vec<usize> = (0..n).filter(|&i| st.selected_bit[i]).collect();
        sel.sort_unstable_by_key(|&i| st.contrib[i]);
        sel.truncate(core_k_sel);

        // Unselected core = surrogate POTENTIAL selection (the i18 lever).
        let mut uns: Vec<usize> = (0..n).filter(|&i| !st.selected_bit[i]).collect();
        if band_mode == 1 {
            // Frontier band: potential closest to the marginal selected potential
            // (the item at the inclusion boundary the density anchor was least sure
            // about). Threshold = min potential among the currently selected items.
            let thresh = (0..n).filter(|&i| st.selected_bit[i])
                .map(|i| pot(i))
                .fold(f64::INFINITY, f64::min);
            uns.sort_by(|&a, &b| {
                (pot(a) - thresh).abs().partial_cmp(&(pot(b) - thresh).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            // Globally highest potential (currently-excluded, high-UB items).
            uns.sort_by(|&a, &b| pot(b).partial_cmp(&pot(a)).unwrap_or(std::cmp::Ordering::Equal));
        }
        uns.truncate(core_k_uns);

        let mut core: Vec<usize> = Vec::with_capacity(core_k_sel + core_k_uns);
        core.extend_from_slice(&sel);
        core.extend_from_slice(&uns);
        if core.len() > core_cap { core.truncate(core_cap); }
        let k = core.len();
        if k == 0 { break; }

        for &i in &core { if st.selected_bit[i] { st.remove_item(i); } }
        let residual = st.slack() as u64;

        let lin: Vec<i64> = core.iter().map(|&i| st.contrib[i] as i64).collect();
        let wt: Vec<u64> = core.iter().map(|&i| challenge.weights[i] as u64).collect();
        let mut pw = vec![0i64; k * k];
        for a in 0..k {
            let ra = &challenge.interaction_values[core[a]];
            for b in (a + 1)..k {
                pw[a * k + b] = ra[core[b]] as i64;
            }
        }

        let mut best_mask: u32 = 0;
        let mut best_val: i64 = i64::MIN;
        let full: u32 = 1u32 << k;
        for mask in 0..full {
            let mut w: u64 = 0;
            let mut m = mask;
            while m != 0 { w += wt[m.trailing_zeros() as usize]; m &= m - 1; }
            if w > residual { continue; }
            let mut val: i64 = 0;
            let mut m1 = mask;
            while m1 != 0 {
                let a = m1.trailing_zeros() as usize;
                val += lin[a];
                let mut m2 = m1 & (m1 - 1);
                while m2 != 0 {
                    val += pw[a * k + m2.trailing_zeros() as usize];
                    m2 &= m2 - 1;
                }
                m1 &= m1 - 1;
            }
            if val > best_val { best_val = val; best_mask = mask; }
        }

        let mut items: Vec<usize> = st.selected_items();
        for b in 0..k { if best_mask & (1u32 << b) != 0 { items.push(core[b]); } }
        let cand = Solution { items };
        let q = eval_feasible(challenge, &cand);
        if q > best_q { best_q = q; best = cand; } else { break; }
    }
    best
}

/// i18 (F_T) — DIRECTED POTENTIAL KICK. A compound perturbation: eject the `k`
/// lowest current-margin SELECTED items and inject the highest surrogate-POTENTIAL
/// EXCLUDED items that fit the freed capacity (greedy, cap-checked -> always
/// i19 F_U/F_V/F_W — FIXPOINT DP<->VND DEEP-CONVERGENCE. Reconstruct a State from
/// `seed`, then drive `dp_refinement_hp` (exact linearized core-window refill) and
/// `vnd_v2` (multi-neighborhood LS incl. the 2-out-2-in exchange) as a coupled pair
/// to a FIXPOINT: each `dp_refinement_hp` re-derives contrib at the *current*
/// selection, so a coordinated multi-item cluster whose members are individually
/// low-margin (but jointly profitable) can enter across successive re-linearizations
/// — a trajectory a single DP+VND pass (the engine's only mode) cannot follow.
/// Keep-best snapshot => the return is never worse than `seed`. Deterministic (no
/// RNG, no wall-clock, fixed round cap). Track-local helper (isolation: only
/// track_t43.rs uses it).
fn fixpoint_dp_vnd_polish(
    challenge: &Challenge,
    seed: &Solution,
    hp: &Hparams,
    tsn: &TopNeighbors,
    max_rounds: usize,
) -> Solution {
    let mut st = State::new_empty(challenge);
    for &i in &seed.items {
        if !st.selected_bit[i] { st.add_item(i); }
    }
    let ch = hp.core_half_dp;
    let mut best_val = st.total_value;
    let mut best_items = st.selected_items();
    let mut prev = i64::MIN;
    for _ in 0..max_rounds {
        dp_refinement_hp(&mut st, ch);
        vnd_v2(&mut st, hp, Some(tsn));
        let v = st.total_value;
        if v > best_val {
            best_val = v;
            best_items = st.selected_items();
        }
        // Fixpoint reached: this round produced no net improvement over the last.
        if v <= prev { break; }
        prev = v;
    }
    Solution { items: best_items }
}

/// feasible). Unlike the atomic <=2-move VND or the <=20-item exact core, this
/// forces a coordinated k-out/k-in swap into the region the density anchor
/// under-weights, then hands the perturbed feasible solution back to the seeded ILS
/// for repair. Injection principle is inverted vs ALNS ruin-recreate (i8/refine6,
/// DEAD, which refilled by CURRENT margin). Deterministic (fixed ranking, greedy
/// cap-fill). Returns a feasible full-instance solution.
fn directed_potential_kick(
    challenge: &Challenge,
    seed: &Solution,
    total_interactions: &[i64],
    k: usize,
) -> Solution {
    let n = challenge.num_items;
    let pot = |i: usize| -> f64 {
        total_interactions[i] as f64 / (challenge.weights[i] as f64).max(1.0)
    };
    let mut st = State::new_empty(challenge);
    for &i in &seed.items { st.add_item(i); }

    // Eject the k lowest current-margin selected items.
    let mut sel: Vec<usize> = (0..n).filter(|&i| st.selected_bit[i]).collect();
    sel.sort_unstable_by_key(|&i| st.contrib[i]);
    for &i in sel.iter().take(k) { st.remove_item(i); }

    // Inject the highest-potential excluded items that fit the freed capacity.
    let cap = challenge.max_weight;
    let mut uns: Vec<usize> = (0..n).filter(|&i| !st.selected_bit[i]).collect();
    uns.sort_by(|&a, &b| pot(b).partial_cmp(&pot(a)).unwrap_or(std::cmp::Ordering::Equal));
    let mut injected = 0usize;
    for &i in &uns {
        if injected >= k { break; }
        if st.total_weight + challenge.weights[i] <= cap {
            st.add_item(i);
            injected += 1;
        }
    }
    Solution { items: st.selected_items() }
}

/// i15 — LINKAGE-CLUSTER exact core (family F_P). Distinct from `exact_core_polish`
/// (i14) in its CORE-SELECTION principle: i14 ranks unselected candidates by their
/// INDIVIDUAL margin (`contrib`), so a group of items that are each marginal on their
/// own but jointly strong (large positive mutual `interaction_values`) is never placed
/// in the core together — exactly the coordinated k-flip the atomic <=2-move VND also
/// misses. Here the unselected half of the core is grown as a positive-interaction
/// CLUSTER (Hindsight `76e4f8c4`: exploit the interaction matrix as a linkage graph;
/// `3b1f5482`: intra-cluster synergy; `559b50cb`: cluster contribution), so the exact
/// 2^k enumeration can admit the whole mutually-reinforcing block at once. Keep-best.
fn linkage_cluster_core(
    challenge: &Challenge,
    seed: &Solution,
    core_k_sel: usize,
    core_k_uns: usize,
    core_cap: usize,
    rounds: usize,
) -> Solution {
    let n = challenge.num_items;
    let mut best = seed.clone();
    let mut best_q = eval_feasible(challenge, &best);

    for _ in 0..rounds {
        let mut st = State::new_empty(challenge);
        for &i in &best.items { st.add_item(i); }

        // Selected core = lowest individual margin (cheapest to eject to make room).
        let mut sel: Vec<usize> = (0..n).filter(|&i| st.selected_bit[i]).collect();
        sel.sort_unstable_by_key(|&i| st.contrib[i]);
        sel.truncate(core_k_sel);

        // Unselected pool = the highest-margin boundary items (reasonable additions).
        let mut uns: Vec<usize> = (0..n).filter(|&i| !st.selected_bit[i]).collect();
        uns.sort_unstable_by_key(|&i| -(st.contrib[i] as i64));
        let pool_p = (core_k_uns * 5).min(uns.len());
        let pool = &uns[..pool_p];

        // Greedy grow a positive-interaction cluster from the pool: repeatedly add the
        // item maximizing (own margin) + (sum of POSITIVE interactions with the cluster
        // already chosen). This surfaces the mutually-reinforcing block, not the top-k
        // individually-best items (which i14 already tries and saturates).
        let mut chosen: Vec<usize> = Vec::with_capacity(core_k_uns);
        let mut picked = vec![false; n];
        while chosen.len() < core_k_uns && chosen.len() < pool.len() {
            let mut best_i: isize = -1;
            let mut best_s: i64 = i64::MIN;
            for &i in pool {
                if picked[i] { continue; }
                let mut s = st.contrib[i] as i64;
                let ri = &challenge.interaction_values[i];
                for &j in &chosen {
                    let v = ri[j] as i64;
                    if v > 0 { s += v; }
                }
                if s > best_s { best_s = s; best_i = i as isize; }
            }
            if best_i < 0 { break; }
            let bi = best_i as usize;
            picked[bi] = true;
            chosen.push(bi);
        }

        let mut core: Vec<usize> = Vec::with_capacity(core_k_sel + chosen.len());
        core.extend_from_slice(&sel);
        core.extend_from_slice(&chosen);
        if core.len() > core_cap { core.truncate(core_cap); }
        let k = core.len();
        if k == 0 { break; }

        // Free the SELECTED core items so st.contrib[i] is the exact linear coefficient
        // of item i against the FROZEN selected set (identical accounting to i14).
        for &i in &core { if st.selected_bit[i] { st.remove_item(i); } }
        let residual = st.slack() as u64;

        let lin: Vec<i64> = core.iter().map(|&i| st.contrib[i] as i64).collect();
        let wt: Vec<u64> = core.iter().map(|&i| challenge.weights[i] as u64).collect();
        let mut pw = vec![0i64; k * k];
        for a in 0..k {
            let ra = &challenge.interaction_values[core[a]];
            for b in (a + 1)..k {
                pw[a * k + b] = ra[core[b]] as i64;
            }
        }

        let mut best_mask: u32 = 0;
        let mut best_val: i64 = i64::MIN;
        let full: u32 = 1u32 << k;
        for mask in 0..full {
            let mut w: u64 = 0;
            let mut m = mask;
            while m != 0 { w += wt[m.trailing_zeros() as usize]; m &= m - 1; }
            if w > residual { continue; }
            let mut val: i64 = 0;
            let mut m1 = mask;
            while m1 != 0 {
                let a = m1.trailing_zeros() as usize;
                val += lin[a];
                let mut m2 = m1 & (m1 - 1);
                while m2 != 0 {
                    val += pw[a * k + m2.trailing_zeros() as usize];
                    m2 &= m2 - 1;
                }
                m1 &= m1 - 1;
            }
            if val > best_val { best_val = val; best_mask = mask; }
        }

        let mut items: Vec<usize> = st.selected_items();
        for b in 0..k { if best_mask & (1u32 << b) != 0 { items.push(core[b]); } }
        let cand = Solution { items };
        let q = eval_feasible(challenge, &cand);
        if q > best_q { best_q = q; best = cand; } else { break; }
    }
    best
}

/// i15 — LINKAGE EJECTION CHAIN (family F_Q). A polynomial-cost coordinated k-flip
/// (Glover ejection chain / Lin-Kernighan style) that reaches configurations the
/// atomic <=2-in/<=2-out VND cannot: inject a strongly-coupled UNSELECTED item even
/// when it does not fit, then coordinate-eject the currently-selected items with the
/// LOWEST marginal contribution (weakest links w.r.t. the enriched set) until feasible,
/// then greedily refill any positive-margin items that now fit — tracking the best
/// FEASIBLE touch-point along the whole chain (keep-best). Deterministic: the chain
/// entry item is varied by `round` (round-th best unselected). Because `contrib`
/// already embeds all interactions with the live selection, both the injection and the
/// ejection are interaction-aware (linkage-guided). Never returns below `seed`.
fn ejection_chain_linkage(
    challenge: &Challenge,
    seed: &Solution,
    max_steps: usize,
    rounds: usize,
) -> Solution {
    let n = challenge.num_items;
    let cap = challenge.max_weight as i64;
    let mut best = seed.clone();
    let mut best_q = eval_feasible(challenge, &best);

    for round in 0..rounds {
        // Fresh feasible incumbent each round.
        let mut st = State::new_empty(challenge);
        for &i in &best.items { st.add_item(i); }

        // Chain entry: the round-th best unselected item by margin (distinct trajectory
        // per round). If it already fits it is a plain add; if not, it forces the chain.
        let mut uns: Vec<usize> = (0..n).filter(|&i| !st.selected_bit[i]).collect();
        uns.sort_unstable_by_key(|&i| -(st.contrib[i] as i64));
        if round >= uns.len() { break; }
        let entry = uns[round];
        if st.contrib[entry] <= 0 { break; } // no promising entry left this round
        st.add_item(entry); // may exceed capacity (transient infeasibility)

        // Coordinate-eject weakest-margin selected items (never the entry) to restore
        // feasibility. Use i64 weight arithmetic so we never call slack() while over cap.
        let mut steps = 0usize;
        while (st.total_weight as i64) > cap && steps < max_steps {
            let mut victim: isize = -1;
            let mut victim_c: i32 = i32::MAX;
            for i in 0..n {
                if i == entry || !st.selected_bit[i] { continue; }
                if st.contrib[i] < victim_c { victim_c = st.contrib[i]; victim = i as isize; }
            }
            if victim < 0 { break; }
            st.remove_item(victim as usize);
            steps += 1;
        }

        // If feasible now, record the touch-point.
        if (st.total_weight as i64) <= cap {
            let q = st.total_value;
            if q > best_q {
                best_q = q;
                best = Solution { items: st.selected_items() };
            }
            // Greedily refill any positive-margin unselected item that now fits.
            loop {
                let slack = cap - st.total_weight as i64;
                if slack <= 0 { break; }
                let mut add: isize = -1;
                let mut add_c: i32 = 0;
                for i in 0..n {
                    if st.selected_bit[i] { continue; }
                    if (challenge.weights[i] as i64) <= slack && st.contrib[i] > add_c {
                        add_c = st.contrib[i];
                        add = i as isize;
                    }
                }
                if add < 0 { break; }
                st.add_item(add as usize);
                let q2 = st.total_value;
                if q2 > best_q {
                    best_q = q2;
                    best = Solution { items: st.selected_items() };
                }
            }
        }
    }
    best
}

#[inline]
fn eval_feasible(ch: &Challenge, sol: &Solution) -> i64 {
    let mut w: u64 = 0;
    for &i in &sol.items { w += ch.weights[i] as u64; }
    if w > ch.max_weight as u64 { return i64::MIN; }
    eval_solution(ch, sol)
}

/// Variable-fixation reduce-then-intensify (I2PLS / Pisinger style).
///
/// The n>1200 path only runs a bare restart loop; the rich elite/crossover/
/// path-relink/LNS pipeline (`solve_rich_small`) is gated to n<=1200 and is dead
/// code at n=5000. This helper shrinks the 5000-item instance to a disputed
/// `core` (<= `core_cap` <= 1200) by fixing variables the search agrees on, then
/// re-optimises the core as a reduced sub-Challenge routed through that rich
/// pipeline, and reconstructs the global solution.
///
/// `fix_mode`: 1 = restart-consensus (unanimous vote), 2 = density-band
/// (Pisinger reduced-cost flavour), 3 = hub-connectivity. `locked_in` is always a
/// subset of the incumbent `best`, so the reconstructed solution is feasible by
/// construction. Returns None (=> keep incumbent) when no useful reduction fits.
fn fixation_intensify(
    challenge: &Challenge,
    hyperparameters: &Option<Map<String, Value>>,
    pool: &[(Solution, i64)],
    best: &Solution,
    total_interactions: &[i64],
    fix_mode: usize,
    core_cap: usize,
) -> Option<Solution> {
    let n = challenge.num_items;
    let mut in_best = vec![false; n];
    for &i in &best.items { in_best[i] = true; }

    // Marginal contribution of each item given the incumbent (values are 0 here,
    // so contrib[i] = sum over selected j of iv[j][i]).
    let mut contrib = vec![0i64; n];
    {
        let mut st = State::new_empty(challenge);
        for &i in &best.items { st.add_item(i); }
        for i in 0..n { contrib[i] = st.contrib[i] as i64; }
    }

    let mut locked_in: Vec<usize> = Vec::new();
    let mut core: Vec<usize> = Vec::new();

    match fix_mode {
        // 1: restart-consensus — items taken by every top-pool elite are locked in,
        // items disputed among the elites form the core, items taken by none drop.
        1 => {
            let mut sorted: Vec<&(Solution, i64)> = pool.iter().collect();
            sorted.sort_by_key(|&&(_, v)| std::cmp::Reverse(v));
            let tk = sorted.len().min(6).max(1);
            let mut freq = vec![0usize; n];
            for &&(ref s, _) in &sorted[..tk] {
                for &i in &s.items { freq[i] += 1; }
            }
            for i in 0..n {
                if freq[i] == tk && in_best[i] { locked_in.push(i); }
                else if freq[i] > 0 { core.push(i); }
            }
        }
        // 2: density-band — the top half of the incumbent by marginal density is
        // locked in; its weak half plus every non-incumbent item with a positive
        // marginal gain form the disputed band.
        2 => {
            let mut dens: Vec<(usize, f64)> = best.items.iter().map(|&i| {
                let w = (challenge.weights[i] as f64).max(1.0);
                (i, contrib[i] as f64 / w)
            }).collect();
            dens.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let keep = dens.len() / 2;
            for (rank, &(i, _)) in dens.iter().enumerate() {
                if rank < keep { locked_in.push(i); } else { core.push(i); }
            }
            for i in 0..n {
                if !in_best[i] && contrib[i] > 0 { core.push(i); }
            }
        }
        // 3 (default): hub-connectivity — high total-interaction incumbent items are
        // structurally sticky and locked in; the incumbent's low-connectivity half
        // plus non-incumbent items above median connectivity form the core.
        _ => {
            let mut hubs: Vec<(usize, i64)> = best.items.iter().map(|&i| (i, total_interactions[i])).collect();
            hubs.sort_by_key(|&(_, v)| std::cmp::Reverse(v));
            let keep = hubs.len() / 2;
            for (rank, &(i, _)) in hubs.iter().enumerate() {
                if rank < keep { locked_in.push(i); } else { core.push(i); }
            }
            let thr = {
                let mut allh: Vec<i64> = (0..n).map(|i| total_interactions[i]).collect();
                allh.sort_unstable();
                allh[(n / 2).min(n.saturating_sub(1))]
            };
            for i in 0..n {
                if !in_best[i] && total_interactions[i] >= thr { core.push(i); }
            }
        }
    }

    if core.len() < 40 { return None; }
    if core.len() > core_cap {
        // Keep the `core_cap` most ambiguous items (smallest |margin| first);
        // overflow is resolved deterministically (kept only if in the incumbent).
        core.sort_by_key(|&i| contrib[i].abs());
        let overflow: Vec<usize> = core.split_off(core_cap);
        for i in overflow {
            if in_best[i] { locked_in.push(i); }
        }
    }
    let m = core.len();

    let mut locked_w: u64 = 0;
    for &i in &locked_in { locked_w += challenge.weights[i] as u64; }
    if locked_w >= challenge.max_weight as u64 { return None; }
    let red_max_weight = (challenge.max_weight as u64 - locked_w) as u32;

    // Build the reduced sub-Challenge over the core items. The interaction matrix
    // is symmetric (Jaccard) so the sub-matrix is exact; cross-interactions with
    // the locked-in set are folded into each core item's linear value (>= 0).
    let mut red_weights = vec![0u32; m];
    let mut red_values = vec![0u32; m];
    let mut red_iv: Vec<Vec<i32>> = vec![vec![0i32; m]; m];
    for a in 0..m {
        let ga = core[a];
        red_weights[a] = challenge.weights[ga];
        let row = &challenge.interaction_values[ga];
        let mut fold: i64 = 0;
        for &l in &locked_in { fold += row[l] as i64; }
        red_values[a] = fold.max(0).min(i32::MAX as i64) as u32;
        for b in (a + 1)..m {
            let iv = row[core[b]];
            red_iv[a][b] = iv;
            red_iv[b][a] = iv;
        }
    }

    let reduced = Challenge {
        seed: challenge.seed,
        num_items: m,
        weights: red_weights,
        values: red_values,
        interaction_values: red_iv,
        max_weight: red_max_weight,
    };

    let mut red_ti = vec![0i64; m];
    for a in 0..m {
        let mut s: i64 = 0;
        for b in 0..m { s += reduced.interaction_values[a][b] as i64; }
        red_ti[a] = s;
    }

    let red_sum_w: u64 = reduced.weights.iter().map(|&w| w as u64).sum();
    let red_budget = if red_sum_w > 0 { ((reduced.max_weight as u64) * 100 / red_sum_w) as u32 } else { 25 };
    let red_hp = Hparams::from_map(hyperparameters, m, red_budget);

    let core_sol = solve_rich_small(&reduced, red_hp, red_ti);

    let mut items = locked_in;
    for &k in &core_sol.items { items.push(core[k]); }
    Some(Solution { items })
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

        let total_interactions: Vec<i64> = {
            let mut sums = vec![0i64; n];
            for i in 0..n {
                let row = unsafe { challenge.interaction_values.get_unchecked(i) };
                let mut si: i64 = 0;
                for j in 0..n { si += row[j] as i64; }
                sums[i] = si;
            }
            sums
        };

        if n > 1200 {
            // L6 STRUCTURAL (i5): HEAD reduced-cost / surrogate lock-out.
            // Distinct from the DEAD consensus fixing (i3/i4 = same-attractor):
            // here we shrink the 5000-item instance FIRST, using a surrogate UPPER
            // BOUND on each item's marginal contribution (total_interactions[i] is a
            // valid UB since all q_ij >= 0), priced against its weight. We LOCK OUT
            // ONLY (never lock in) the weight-inefficient tail, so every survivor
            // solution is feasible on the full instance by construction (no
            // feasibility-tolerance risk). The SAME restart engine then refines the
            // smaller survivor set more densely per VND/DP pass at equal work HP.
            // rc_mode=0 (default) => byte-exact original path (sentinel).
            let rc_mode: usize = hyperparameters.as_ref()
                .and_then(|m| m.get("rc_mode")).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let rc_fraction: f64 = hyperparameters.as_ref()
                .and_then(|m| m.get("rc_fraction")).and_then(|v| v.as_f64()).unwrap_or(0.30);
            // i7 (min_level=7): reinvest the budget freed by the reduction by
            // CHANGING THE SEARCH SPACE (mem 14741428: same-space LS is exhausted;
            // only space-changing reconstruction escapes the attractor). The mode5
            // reduction anchors on a weak greedy incumbent and then returns without
            // using the ~44s of freed budget. refine_mode dispatches 3 distinct
            // space-changing families on top of the mode5 base:
            //   0 = sentinel (single reduce+refine, byte-exact i6 champion 53292);
            //   1 = iterated re-anchor (re-derive frontier from the REFINED
            //       incumbent, re-reduce, refine again — CHILS-style, mem a2ef186a);
            //   2 = full-instance seeded escape (seed the full 5000-item engine with
            //       the reduced solution so wrongly-locked items can re-enter);
            //   3 = union-of-anchors reduction (density ∪ raw-synergy frontier bands).
            let refine_mode: usize = hyperparameters.as_ref()
                .and_then(|m| m.get("refine_mode")).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            // i11: basin-count for the breadth families (refine_mode 10/11/12).
            // Structural knob = number of DISTINCT basins explored (the one Q lever
            // confirmed on t43, Hindsight 5c6b51ae), NOT a blind depth HP. Baked per
            // mode; overridable to size the reinvested budget.
            let n_escapes: Option<usize> = hyperparameters.as_ref()
                .and_then(|m| m.get("n_escapes")).and_then(|v| v.as_u64()).map(|v| v as usize);

            // i23 (min_level=8, replace): the refine chain's re-solves (F1 re-anchor
            // large_n_restarts + F2 seeded escapes) run on `refine_hp` = a SHALLOW
            // schedule so the deep base is spent ONCE (initial solve = Q source) and
            // the re-anchor stays cheap. `refine_shallow` absent/0 ⇒ refine_hp == hp
            // ⇒ champion/sentinel path byte-exact (53327). Only used by refine_mode=5.
            let refine_shallow: u64 = hyperparameters.as_ref()
                .and_then(|m| m.get("refine_shallow")).and_then(|v| v.as_u64()).unwrap_or(0);
            // i25: `refine_shallow` is now only the DEFAULT for the two decoupled phase
            // levels below (the shared `refine_hp` binding it used to build is superseded
            // by `anchor_hp`/`escape_hp`).

            // i25 (min_level=8, REPLACE): DECOUPLE the two refine-chain phases of
            // refine_mode=5. i23/i24 forced F1 re-anchor AND F2 seeded escape to share
            // ONE schedule (`refine_hp`); but their roles differ (dec86d81 / b1080370):
            //   • F1 re-anchor's VALUE is re-REDUCING the frontier (cheap) — its deep
            //     multi-restart re-solve is a DUPLICATE of the base diversification
            //     (waste). → drive it with `anchor_level` (default = refine_shallow).
            //   • F2 seeded escape is the PRODUCTIVE Q source; stripping its ILS depth
            //     cost -57Q (b1080370). → keep it deep (`escape_level`, default =
            //     refine_shallow) and REINVEST F1's reclaimed time by running MORE
            //     escape passes (`escape_passes`, default 2 = i24 byte-exact).
            // All three absent ⇒ anchor_hp==escape_hp==refine_hp & passes=2 ⇒ the
            // sentinel {rc5,0.30,refine5} and every i24 arm stay byte-exact.
            let anchor_level: u64 = hyperparameters.as_ref()
                .and_then(|m| m.get("anchor_level")).and_then(|v| v.as_u64()).unwrap_or(refine_shallow);
            let escape_level: u64 = hyperparameters.as_ref()
                .and_then(|m| m.get("escape_level")).and_then(|v| v.as_u64()).unwrap_or(refine_shallow);
            let escape_passes: usize = hyperparameters.as_ref()
                .and_then(|m| m.get("escape_passes")).and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(2);
            // i39 (min_level=8, REPLACE of the F2-escape brick): run the productive
            // F2 seeded escape (refine_mode=5) on a LIGHTLY-REDUCED instance instead
            // of the full n=5000. The base solve + F1 re-anchor already run on the
            // rc-reduced core, but the F2 escape is the SOLE place still paying full-n
            // ILS cost (`run_one_instance_seeded(challenge, ...)`). i28-i38 attacked the
            // BASE VND loop (prefetch-saturated, dead) or the BASE reduction (i38 rc0.6,
            // null); nobody reduced the escape instance. We build the escape instance
            // once, ANCHORED on `fb` (reduced_cost_reduce_anchored guarantees fb members
            // are never locked out => feasible seed, and the top locked-out re-entry band
            // is retained), so each escape ILS pass runs on ~(1-escape_rc_fraction)*n items
            // => cheaper deep escape. `escape_rc_fraction` absent/0 => full-instance escape
            // => champion byte-exact (53375). keep-best is evaluated on the FULL challenge
            // and the seed is fb => Q >= fb regardless (downside-bounded time lever).
            let escape_rc_fraction: f64 = hyperparameters.as_ref()
                .and_then(|m| m.get("escape_rc_fraction")).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let anchor_hp = shallow_refine_hp(&hp, anchor_level);
            let escape_hp = shallow_refine_hp(&hp, escape_level);

            // i26 (min_level=8, COMBINE src_algo91 prom_pulse_marlin_v2): gate the
            // ported consensus core-DP finisher (fn consensus_core_dp). Diagnostic
            // ARM_A {base_profile:1,rc0,refine0}=53110 runs the SAME deep schedule as
            // SOTA prom_pulse (53359) at the SAME ~84s yet loses -249 Q = a
            // NON-FAITHFUL PORT: prom_pulse FINISHES every large-n restart pool with a
            // consensus core-DP; our refacto left that finisher only in solve_rich_small
            // (n<=1200 / fixation path), so the rc0 and rc5 large-n pools never get it.
            // 0/absent => the finisher is skipped => champion/sentinel byte-exact.
            let pool_core_dp: u64 = hyperparameters.as_ref()
                .and_then(|m| m.get("pool_core_dp")).and_then(|v| v.as_u64()).unwrap_or(0);
            // i27 (min_level=7, brick-refonte of consensus_core_dp): expose the two
            // hardcoded internal constants of the finisher so its EXACT-solve scope can
            // be enlarged. Sentinel = 0 => keep the ported-verbatim defaults (6 / 200),
            // so a base run WITHOUT these keys stays byte-exact with i26 V3 (53357).
            //  - elite_vote_k: size of the freq-vote elite pool (Np). Larger Np suppresses
            //    incorrect `always_in` fixing (Hindsight 2867c803/5ec07a6f) => more swing
            //    items fall into `disputed` and reach the exact DP.
            //  - core_disputed_cap: cap on the `disputed` core passed to the weight-indexed
            //    exact DP. Exact QKP solvers hold up to ~1500 items (Hindsight 32a55354);
            //    the DP is indexed on residual capacity so a modestly larger core is cheap.
            let elite_vote_k: usize = hyperparameters.as_ref()
                .and_then(|m| m.get("elite_vote_k")).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let core_disputed_cap: usize = hyperparameters.as_ref()
                .and_then(|m| m.get("core_disputed_cap")).and_then(|v| v.as_u64()).unwrap_or(0) as usize;

            if rc_mode != 0 {
                let reduction = if refine_mode == 3 || refine_mode == 70 || refine_mode == 71 {
                    // i17 REDUCTION-SPACE ENRICHMENT (untouched axis): the survivor set is
                    // the UNION of the density and synergy frontier bands, so an item is
                    // locked out only if BOTH anchors discard it. Admits frontier items the
                    // single density anchor over-fixes (Kernel search 0cc77a59 enlarge the
                    // candidate set; Cover-Relax 024aec40 fix only the most certain, keep a
                    // band). Every downstream family so far (i9-i16) searched the SAME tight
                    // density sub-space -> same 53327 attractor; enrich the space instead.
                    reduced_cost_reduce_union(challenge, &total_interactions, rc_mode, rc_fraction)
                } else if refine_mode == 72 {
                    // i17: single SYNERGY anchor (raw-interaction frontier band) in place of
                    // the density anchor -> a differently-shaped reduced sub-space favouring
                    // high-synergy heavy items that density ranking pushes below the frontier.
                    let syn = build_synergy_incumbent(challenge, &total_interactions);
                    reduced_cost_reduce_anchored(
                        challenge, &total_interactions, rc_mode, rc_fraction, Some(&syn),
                    )
                } else {
                    reduced_cost_reduce(challenge, &total_interactions, rc_mode, rc_fraction)
                };
                if let Some((reduced, survivors, red_ti)) = reduction {
                    let (mut best_sol, pool) = large_n_restarts(&reduced, &hp, &red_ti);
                    // i26: ported consensus core-DP finisher over the REDUCED restart
                    // pool (indices are survivor-local; result maps back via `survivors`
                    // just like best_sol). Only fires when pool_core_dp>0.
                    if pool_core_dp > 0 {
                        if let Some(bs) = best_sol.clone() {
                            if let Some(imp) = consensus_core_dp(&reduced, &pool, &bs, &red_ti, elite_vote_k, core_disputed_cap) {
                                best_sol = Some(imp);
                            }
                        }
                    }
                    let mut full_best: Option<Solution> = best_sol.map(|s| Solution {
                        items: s.items.iter().map(|&k| survivors[k]).collect(),
                    });

                    match refine_mode {
                        1 => {
                            // F1 iterated re-anchor: re-reduce anchored on the
                            // refined incumbent, refine again, keep the better.
                            if let Some(fb) = full_best.clone() {
                                let mut mask = vec![false; challenge.num_items];
                                for &i in &fb.items { mask[i] = true; }
                                if let Some((r2, surv2, ti2)) = reduced_cost_reduce_anchored(
                                    challenge, &total_interactions, rc_mode, rc_fraction, Some(&mask),
                                ) {
                                    let (bs2, _) = large_n_restarts(&r2, &hp, &ti2);
                                    if let Some(s2) = bs2 {
                                        let cand = Solution {
                                            items: s2.items.iter().map(|&k| surv2[k]).collect(),
                                        };
                                        if eval_feasible(challenge, &cand) > eval_feasible(challenge, &fb) {
                                            full_best = Some(cand);
                                        }
                                    }
                                }
                            }
                        }
                        2 => {
                            // F2 full-instance seeded escape: a few seeded restarts on
                            // the FULL instance let locked-out items re-enter. Bounded
                            // to 2 restarts to stay well inside the 101530ms budget.
                            if let Some(fb) = full_best.clone() {
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let mut cand_best = fb.clone();
                                let mut cand_q = eval_feasible(challenge, &fb);
                                for r in 0..2usize {
                                    let (sol, val) = run_one_instance_seeded(
                                        challenge, &hp, 1000 + r, Some(&tsn),
                                        &total_interactions, Some(&fb),
                                    );
                                    if val > cand_q { cand_q = val; cand_best = sol; }
                                }
                                full_best = Some(cand_best);
                            }
                        }
                        4 => {
                            // F_A RECOMBINATION (i8 combine): path-relink the reduced
                            // champion against the best Hamming-distant restart basin.
                            // PR is a QKP crossover on binary vectors (Hindsight
                            // 3e86f62a) that explores trajectories ILS restarts miss
                            // (f1299b29); the t43 lesson 95824a48 explicitly recommended
                            // trying L6 Path Relinking here but it was never reached by
                            // the large-n path (the proven path_relink brick lived only
                            // in the n<=1200 elite-pool loop). Guides drawn from `pool`,
                            // never below the champion (downside bounded).
                            if let Some(fb) = full_best.clone() {
                                let mut fb_bits = vec![false; challenge.num_items];
                                for &i in &fb.items { fb_bits[i] = true; }
                                let mut guide: Option<Solution> = None;
                                let mut guide_q = i64::MIN;
                                for (s, v) in &pool {
                                    let cand: Vec<usize> =
                                        s.items.iter().map(|&k| survivors[k]).collect();
                                    let mut cb = vec![false; challenge.num_items];
                                    for &i in &cand { cb[i] = true; }
                                    let mut ham = 0usize;
                                    for i in 0..challenge.num_items {
                                        if cb[i] != fb_bits[i] { ham += 1; }
                                    }
                                    if ham > 8 && *v > guide_q {
                                        guide_q = *v;
                                        guide = Some(Solution { items: cand });
                                    }
                                }
                                if let Some(g) = guide {
                                    let mut best = fb.clone();
                                    let mut bq = eval_feasible(challenge, &fb);
                                    for cand in [
                                        path_relink(challenge, &fb, &g, &hp),
                                        path_relink(challenge, &g, &fb, &hp),
                                    ] {
                                        let q = eval_feasible(challenge, &cand);
                                        if q > bq { bq = q; best = cand; }
                                    }
                                    full_best = Some(best);
                                }
                            }
                        }
                        5 => {
                            // F_B CHAINED ESCAPE (i8): compose the two i7 levers proved
                            // individually positive — F1 iterated re-anchor lifts the
                            // incumbent (re-derive the frontier from the refined
                            // solution), then F2 full-instance seeded escape from THAT
                            // lets wrongly-locked items re-enter. Sequential, keep-best.
                            if let Some(fb0) = full_best.clone() {
                                let mut fb = fb0;
                                // F1 re-anchor
                                let mut mask = vec![false; challenge.num_items];
                                for &i in &fb.items { mask[i] = true; }
                                if let Some((r2, surv2, ti2)) = reduced_cost_reduce_anchored(
                                    challenge, &total_interactions, rc_mode, rc_fraction, Some(&mask),
                                ) {
                                    // i25: re-anchor re-solve on `anchor_hp` (default =
                                    // refine_hp). anchor_level=5 makes F1 a near-pure
                                    // re-reduce (1 full-depth restart) — its deep multi-
                                    // restart was a duplicate of the base diversification.
                                    let (bs2, _) = large_n_restarts(&r2, &anchor_hp, &ti2);
                                    if let Some(s2) = bs2 {
                                        let cand = Solution {
                                            items: s2.items.iter().map(|&k| surv2[k]).collect(),
                                        };
                                        if eval_feasible(challenge, &cand) > eval_feasible(challenge, &fb) {
                                            fb = cand;
                                        }
                                    }
                                }
                                // F2 escape seeded from the re-anchored incumbent
                                let mut cand_best = fb.clone();
                                let mut cand_q = eval_feasible(challenge, &fb);
                                // i39: build the reduced escape instance ONCE, anchored on
                                // fb (fb members never locked out); reused across all passes.
                                // Returns None (=> fall back to full escape) if the reduction
                                // is trivial (n-m<100) — safe. `inv` maps full->survivor-local.
                                let esc_reduced: Option<(Challenge, Vec<usize>, Vec<i64>, Solution)> =
                                    if escape_rc_fraction > 0.0 {
                                        let mut emask = vec![false; challenge.num_items];
                                        for &i in &fb.items { emask[i] = true; }
                                        reduced_cost_reduce_anchored(
                                            challenge, &total_interactions, rc_mode,
                                            escape_rc_fraction, Some(&emask),
                                        ).map(|(rc, surv, ti_e)| {
                                            let mut inv = vec![usize::MAX; challenge.num_items];
                                            for (k, &g) in surv.iter().enumerate() { inv[g] = k; }
                                            let fb_local = Solution {
                                                items: fb.items.iter().map(|&i| inv[i]).collect(),
                                            };
                                            (rc, surv, ti_e, fb_local)
                                        })
                                    } else { None };
                                if let Some((ref rc, ref surv, ref ti_e, ref fb_local)) = esc_reduced {
                                    // Deep escape on the reduced instance. tsn is rebuilt on
                                    // the reduced instance ONCE and shared across passes.
                                    let tsn_e = TopNeighbors::new(rc, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                    for r in 0..escape_passes {
                                        let (sol_l, _v) = run_one_instance_seeded(
                                            rc, &escape_hp, 1000 + r, Some(&tsn_e),
                                            ti_e, Some(fb_local),
                                        );
                                        // Map survivor-local back to full, evaluate on the
                                        // FULL challenge (val_l is survivor-local — not
                                        // comparable to cand_q which is full-instance Q).
                                        let sol_f = Solution {
                                            items: sol_l.items.iter().map(|&k| surv[k]).collect(),
                                        };
                                        let vf = eval_feasible(challenge, &sol_f);
                                        if vf > cand_q { cand_q = vf; cand_best = sol_f; }
                                    }
                                } else {
                                    // i25 sentinel path (escape_rc_fraction=0 OR trivial
                                    // reduction): full-instance escape — byte-exact champion.
                                    // F2 escape on `escape_hp` (default = refine_hp, KEEP its
                                    // ILS depth — b1080370: stripping it cost -57Q) and run
                                    // `escape_passes` (default 2 = i24 byte-exact) distinct-seed passes.
                                    let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                    for r in 0..escape_passes {
                                        let (sol, val) = run_one_instance_seeded(
                                            challenge, &escape_hp, 1000 + r, Some(&tsn),
                                            &total_interactions, Some(&fb),
                                        );
                                        if val > cand_q { cand_q = val; cand_best = sol; }
                                    }
                                }
                                full_best = Some(cand_best);
                            }
                        }
                        6 => {
                            // F_C ALNS RUIN-AND-RECREATE (i8): destroy ~20% least-
                            // connected items in the champion + greedy repair by best
                            // marginal contribution, then VND/DP polish (Hindsight
                            // ba100407: 3-8% expected on large-n t43). A single directed
                            // destroy-repair kernel — distinct from restart escape (F2)
                            // and recombination (F_A). Keep-best => never below champion.
                            if let Some(fb) = full_best.clone() {
                                let improved = alns_ruin_recreate(challenge, &fb, &hp);
                                if eval_feasible(challenge, &improved) > eval_feasible(challenge, &fb) {
                                    full_best = Some(improved);
                                }
                            }
                        }
                        10 => {
                            // F_D BASIN-BREADTH-BY-COUNT (i11). The Q lever on t43 is
                            // exploring ADDITIONAL basins, not incumbent-lift tricks
                            // (Hindsight 5c6b51ae: the +49 once credited to two-mode
                            // scoring was in fact one extra restart basin; two-mode was
                            // Q-neutral). F2 (refine_mode=2) spends only 2 seeded escapes
                            // and leaves ~38s of budget under 101530ms — here we reinvest
                            // that freed budget into k seeded escapes from the reduced
                            // champion incumbent, each with a distinct RNG trajectory.
                            // AUGMENT (keep-best), never REPLACE (i2 dead) -> superset of
                            // F2 => Q >= refine2 (53312) by construction.
                            if let Some(fb) = full_best.clone() {
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let mut cand_best = fb.clone();
                                let mut cand_q = eval_feasible(challenge, &fb);
                                let k = n_escapes.unwrap_or(12);
                                for r in 0..k {
                                    let (sol, val) = run_one_instance_seeded(
                                        challenge, &hp, 1000 + r, Some(&tsn),
                                        &total_interactions, Some(&fb),
                                    );
                                    if val > cand_q { cand_q = val; cand_best = sol; }
                                }
                                full_best = Some(cand_best);
                            }
                        }
                        11 => {
                            // F_E BASIN-BREADTH-BY-RADIUS (i11). Distinct mechanism from
                            // F_D: instead of varying only the RNG, vary the LAUNCH
                            // DISTANCE. Before each escape strip an increasing fraction of
                            // the incumbent's lowest-density items so successive escapes
                            // start from progressively farther shells (near->far basin
                            // diversification, Hindsight 2b7fba86 escape-basin / 3b1f5482
                            // inter-cluster explore). r=0 launches from the untouched
                            // incumbent (== F2's first escape) => keep-best still >= 53312.
                            if let Some(fb) = full_best.clone() {
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let mut cand_best = fb.clone();
                                let mut cand_q = eval_feasible(challenge, &fb);
                                let k = n_escapes.unwrap_or(8);
                                for r in 0..k {
                                    let seed_r = strip_low_density(challenge, &fb, r, k);
                                    let (sol, val) = run_one_instance_seeded(
                                        challenge, &hp, 2000 + r, Some(&tsn),
                                        &total_interactions, Some(&seed_r),
                                    );
                                    if val > cand_q { cand_q = val; cand_best = sol; }
                                }
                                full_best = Some(cand_best);
                            }
                        }
                        12 => {
                            // F_F POOL-BASIN MULTI-START (i11). Distinct again: launch the
                            // escapes from the top-M DISTINCT restart basins already in
                            // `pool` (mapped reduced->full) rather than all from the single
                            // champion incumbent. These are genuinely Hamming-distant start
                            // points (the same pool refine4 draws PR guides from), so the
                            // seeded ILS explores structurally different regions. The
                            // champion incumbent is always seed #0 => keep-best >= 53312.
                            if let Some(fb) = full_best.clone() {
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let mut cand_best = fb.clone();
                                let mut cand_q = eval_feasible(challenge, &fb);
                                let k = n_escapes.unwrap_or(8);
                                let mut pool_sorted: Vec<&(Solution, i64)> = pool.iter().collect();
                                pool_sorted.sort_unstable_by_key(|&&(_, v)| std::cmp::Reverse(v));
                                let mut seeds: Vec<Solution> = vec![fb.clone()];
                                for &(s, _) in pool_sorted.iter().take(k) {
                                    let full: Vec<usize> =
                                        s.items.iter().map(|&kk| survivors[kk]).collect();
                                    seeds.push(Solution { items: full });
                                }
                                for (r, seed) in seeds.iter().enumerate() {
                                    let (sol, val) = run_one_instance_seeded(
                                        challenge, &hp, 3000 + r, Some(&tsn),
                                        &total_interactions, Some(seed),
                                    );
                                    if val > cand_q { cand_q = val; cand_best = sol; }
                                }
                                full_best = Some(cand_best);
                            }
                        }
                        40 => {
                            // F_M EXACT RESTRICTED-CORE polish (i14). Base = F2 seeded
                            // escape (refine2 = 53312 @ ~63s, 38s margin under 101530ms):
                            // the escape re-admits locked-out items, then the exact core
                            // solve reaches the coordinated k-flips the atomic <=2-move VND
                            // cannot (meta-constat: the +32 is trapped behind that barrier;
                            // Hindsight c6dfea33 a QKP local-opt != subset local-opt).
                            // Iterated, keep-best => Q >= refine2 (53312) by construction.
                            if let Some(fb) = full_best.clone() {
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let mut cand_best = fb.clone();
                                let mut cand_q = eval_feasible(challenge, &fb);
                                for r in 0..2usize {
                                    let (sol, val) = run_one_instance_seeded(
                                        challenge, &hp, 1000 + r, Some(&tsn),
                                        &total_interactions, Some(&fb),
                                    );
                                    if val > cand_q { cand_q = val; cand_best = sol; }
                                }
                                let polished = exact_core_polish(challenge, &cand_best, 12, 12, 18, 8);
                                if eval_feasible(challenge, &polished) > cand_q {
                                    cand_best = polished;
                                }
                                full_best = Some(cand_best);
                            }
                        }
                        41 => {
                            // F_N EXACT CORE — MAX-REACH (i14). Same F2 base; wider core
                            // (cap 20, 2^20 exact) with fewer re-center rounds — tests
                            // reach-per-solve vs re-centering (distinct structural knob
                            // from F_M). Keep-best => Q >= 53312.
                            if let Some(fb) = full_best.clone() {
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let mut cand_best = fb.clone();
                                let mut cand_q = eval_feasible(challenge, &fb);
                                for r in 0..2usize {
                                    let (sol, val) = run_one_instance_seeded(
                                        challenge, &hp, 1000 + r, Some(&tsn),
                                        &total_interactions, Some(&fb),
                                    );
                                    if val > cand_q { cand_q = val; cand_best = sol; }
                                }
                                let polished = exact_core_polish(challenge, &cand_best, 14, 14, 20, 4);
                                if eval_feasible(challenge, &polished) > cand_q {
                                    cand_best = polished;
                                }
                                full_best = Some(cand_best);
                            }
                        }
                        42 => {
                            // F_O EXACT/STOCHASTIC INTERLEAVE (i14). polish -> one F2 escape
                            // seeded from the polished incumbent (re-admit items the exact
                            // core froze out) -> polish again. Alternates the deterministic
                            // exact core with the stochastic escape so each feeds the other.
                            // Keep-best throughout => Q >= 53312.
                            if let Some(fb) = full_best.clone() {
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let mut cand_best = fb.clone();
                                let mut cand_q = eval_feasible(challenge, &fb);
                                for r in 0..2usize {
                                    let (sol, val) = run_one_instance_seeded(
                                        challenge, &hp, 1000 + r, Some(&tsn),
                                        &total_interactions, Some(&fb),
                                    );
                                    if val > cand_q { cand_q = val; cand_best = sol; }
                                }
                                let p1 = exact_core_polish(challenge, &cand_best, 12, 12, 18, 4);
                                let q1 = eval_feasible(challenge, &p1);
                                if q1 > cand_q { cand_q = q1; cand_best = p1; }
                                let (esc, ev) = run_one_instance_seeded(
                                    challenge, &hp, 1500, Some(&tsn),
                                    &total_interactions, Some(&cand_best),
                                );
                                if ev > cand_q { cand_q = ev; cand_best = esc; }
                                let p2 = exact_core_polish(challenge, &cand_best, 12, 12, 18, 4);
                                if eval_feasible(challenge, &p2) > cand_q { cand_best = p2; }
                                full_best = Some(cand_best);
                            }
                        }
                        20 => {
                            // F_G REVERSED CHAIN (i12): escape FIRST (F2) to re-admit
                            // wrongly-locked items on the full instance, THEN re-anchor
                            // (F1) the reduced-cost frontier on the ENRICHED incumbent.
                            // refine5 anchors the re-reduction on the PRE-escape solution;
                            // here the reduction sees the escaped (richer) solution => a
                            // structurally different reduced sub-space (I2PLS explore-then-
                            // re-reduce, Hindsight ff1defda). refine5 (F1->F2)=53327 sat;
                            // this tests composition ORDER, not more/wider basins. Keep-best
                            // (>= reduced champion >= 53312). Budget: base ~57s + 2 escapes
                            // ~6s + 1 re-anchor ~37s ~= 100s < 101530ms.
                            if let Some(fb0) = full_best.clone() {
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let mut fb = fb0.clone();
                                let mut fq = eval_feasible(challenge, &fb0);
                                for r in 0..2usize {
                                    let (sol, val) = run_one_instance_seeded(
                                        challenge, &hp, 1000 + r, Some(&tsn),
                                        &total_interactions, Some(&fb0),
                                    );
                                    if val > fq { fq = val; fb = sol; }
                                }
                                let mut mask = vec![false; challenge.num_items];
                                for &i in &fb.items { mask[i] = true; }
                                if let Some((r2, surv2, ti2)) = reduced_cost_reduce_anchored(
                                    challenge, &total_interactions, rc_mode, rc_fraction, Some(&mask),
                                ) {
                                    let (bs2, _) = large_n_restarts(&r2, &hp, &ti2);
                                    if let Some(s2) = bs2 {
                                        let cand = Solution {
                                            items: s2.items.iter().map(|&k| surv2[k]).collect(),
                                        };
                                        if eval_feasible(challenge, &cand) > fq {
                                            fb = cand;
                                        }
                                    }
                                }
                                full_best = Some(fb);
                            }
                        }
                        21 => {
                            // F_H CHAIN + POOL-DIVERSE ESCAPE (i12 combine): compose the
                            // proven F1 re-anchor (refine5's winning prefix) with i11's
                            // pool-basin escape diversity (refine12) on the escape seeds.
                            // refine5 escapes twice from ONE incumbent (=53327); refine12
                            // diversifies seeds but WITHOUT the re-anchor (=53314). Stacking
                            // both: re-anchor lifts the frontier, then the 2 escapes launch
                            // from the incumbent AND one Hamming-distant pool basin => tests
                            // whether the two partial gains ADD (Hindsight 5c6b51ae basins +
                            // a2ef186a CHILS re-anchor). Same 2-escape budget as refine5 =>
                            // ~100s. Keep-best (>= 53312).
                            if let Some(fb0) = full_best.clone() {
                                let mut fb = fb0.clone();
                                let mut mask = vec![false; challenge.num_items];
                                for &i in &fb.items { mask[i] = true; }
                                if let Some((r2, surv2, ti2)) = reduced_cost_reduce_anchored(
                                    challenge, &total_interactions, rc_mode, rc_fraction, Some(&mask),
                                ) {
                                    let (bs2, _) = large_n_restarts(&r2, &hp, &ti2);
                                    if let Some(s2) = bs2 {
                                        let cand = Solution {
                                            items: s2.items.iter().map(|&k| surv2[k]).collect(),
                                        };
                                        if eval_feasible(challenge, &cand) > eval_feasible(challenge, &fb) {
                                            fb = cand;
                                        }
                                    }
                                }
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let mut cand_best = fb.clone();
                                let mut cand_q = eval_feasible(challenge, &fb);
                                let mut pool_sorted: Vec<&(Solution, i64)> = pool.iter().collect();
                                pool_sorted.sort_unstable_by_key(|&&(_, v)| std::cmp::Reverse(v));
                                let mut seeds: Vec<Solution> = vec![fb.clone()];
                                for &(s, _) in pool_sorted.iter().take(1) {
                                    let full: Vec<usize> =
                                        s.items.iter().map(|&kk| survivors[kk]).collect();
                                    seeds.push(Solution { items: full });
                                }
                                for (r, seed) in seeds.iter().enumerate() {
                                    let (sol, val) = run_one_instance_seeded(
                                        challenge, &hp, 1000 + r, Some(&tsn),
                                        &total_interactions, Some(seed),
                                    );
                                    if val > cand_q { cand_q = val; cand_best = sol; }
                                }
                                full_best = Some(cand_best);
                            }
                        }
                        22 => {
                            // F_I ALTERNATION SANDWICH (i12): F2 escape -> F1 re-anchor on
                            // the enriched incumbent -> F2 escape. A 3-phase I2PLS cycle
                            // (ff1defda: iterate Explore/Escape). refine5 does F1->F2 once;
                            // refine20 does F2->F1; this alternates so the re-anchor sits
                            // BETWEEN two escapes -> the frontier is re-derived from an
                            // escaped solution AND re-polished by a trailing escape. Only ONE
                            // re-anchor (the expensive stage) => ~57+3+37+3 ~= 100s. Keep-best.
                            if let Some(fb0) = full_best.clone() {
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let mut fb = fb0.clone();
                                let mut fq = eval_feasible(challenge, &fb0);
                                {
                                    let (sol, val) = run_one_instance_seeded(
                                        challenge, &hp, 1000, Some(&tsn),
                                        &total_interactions, Some(&fb0),
                                    );
                                    if val > fq { fq = val; fb = sol; }
                                }
                                let mut mask = vec![false; challenge.num_items];
                                for &i in &fb.items { mask[i] = true; }
                                if let Some((r2, surv2, ti2)) = reduced_cost_reduce_anchored(
                                    challenge, &total_interactions, rc_mode, rc_fraction, Some(&mask),
                                ) {
                                    let (bs2, _) = large_n_restarts(&r2, &hp, &ti2);
                                    if let Some(s2) = bs2 {
                                        let cand = Solution {
                                            items: s2.items.iter().map(|&k| surv2[k]).collect(),
                                        };
                                        let cq = eval_feasible(challenge, &cand);
                                        if cq > fq { fq = cq; fb = cand; }
                                    }
                                }
                                {
                                    let (sol, val) = run_one_instance_seeded(
                                        challenge, &hp, 1001, Some(&tsn),
                                        &total_interactions, Some(&fb),
                                    );
                                    if val > fq { fb = sol; }
                                }
                                full_best = Some(fb);
                            }
                        }
                        30 => {
                            // F_J STRATEGIC OSCILLATION — fixed span (i13). The feasible
                            // VND attractor is exhausted (i9 width / i10 count / i11 order
                            // all saturate 53327); mem 14741428 says only infeasible
                            // navigation escapes it. Cross the capacity boundary from the
                            // reduced champion incumbent: pack past cap then shed, tracking
                            // the best feasible touch-down + a final exact-delta VND polish.
                            // keep-best => never below the reduced champion.
                            if let Some(fb) = full_best.clone() {
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let osc = strategic_oscillation(challenge, &fb, &hp, &tsn, 6, 0.05, false);
                                if eval_feasible(challenge, &osc) > eval_feasible(challenge, &fb) {
                                    full_best = Some(osc);
                                }
                            }
                        }
                        31 => {
                            // F_K F1 RE-ANCHOR + STRATEGIC OSCILLATION (i13). Lift the
                            // incumbent to F1 quality (refine5's proven winning prefix),
                            // then use boundary-crossing oscillation IN PLACE OF the F2
                            // seeded escape — tests infeasible navigation as a stronger
                            // final escape stage than the feasible seeded restart. keep-best.
                            if let Some(fb0) = full_best.clone() {
                                let mut fb = fb0.clone();
                                let mut mask = vec![false; challenge.num_items];
                                for &i in &fb.items { mask[i] = true; }
                                if let Some((r2, surv2, ti2)) = reduced_cost_reduce_anchored(
                                    challenge, &total_interactions, rc_mode, rc_fraction, Some(&mask),
                                ) {
                                    let (bs2, _) = large_n_restarts(&r2, &hp, &ti2);
                                    if let Some(s2) = bs2 {
                                        let cand = Solution {
                                            items: s2.items.iter().map(|&k| surv2[k]).collect(),
                                        };
                                        if eval_feasible(challenge, &cand) > eval_feasible(challenge, &fb) {
                                            fb = cand;
                                        }
                                    }
                                }
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let osc = strategic_oscillation(challenge, &fb, &hp, &tsn, 6, 0.05, false);
                                full_best = Some(
                                    if eval_feasible(challenge, &osc) > eval_feasible(challenge, &fb) {
                                        osc
                                    } else {
                                        fb
                                    },
                                );
                            }
                        }
                        32 => {
                            // F_L EXPANDING-SPAN STRATEGIC OSCILLATION (i13, Glover classic).
                            // beta grows each cycle so successive excursions probe
                            // progressively deeper into the infeasible region, connecting
                            // farther feasible basins than a fixed span. keep-best.
                            if let Some(fb) = full_best.clone() {
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let osc = strategic_oscillation(challenge, &fb, &hp, &tsn, 5, 0.02, true);
                                if eval_feasible(challenge, &osc) > eval_feasible(challenge, &fb) {
                                    full_best = Some(osc);
                                }
                            }
                        }
                        50 => {
                            // F_P LINKAGE-CLUSTER EXACT CORE (i15). Same F2 base as i14's
                            // margin core (refine2 = 53312 @ ~63s, 38s margin), but the
                            // unselected half of the exact core is grown as a POSITIVE-
                            // INTERACTION cluster (linkage graph, Hindsight 76e4f8c4/
                            // 3b1f5482) rather than by individual margin — the exact 2^k
                            // enumeration can then admit a mutually-reinforcing block that
                            // i14's margin core never places together. Iterated, keep-best
                            // => Q >= refine2 (53312) by construction.
                            if let Some(fb) = full_best.clone() {
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let mut cand_best = fb.clone();
                                let mut cand_q = eval_feasible(challenge, &fb);
                                for r in 0..2usize {
                                    let (sol, val) = run_one_instance_seeded(
                                        challenge, &hp, 1000 + r, Some(&tsn),
                                        &total_interactions, Some(&fb),
                                    );
                                    if val > cand_q { cand_q = val; cand_best = sol; }
                                }
                                let polished = linkage_cluster_core(challenge, &cand_best, 12, 12, 18, 8);
                                if eval_feasible(challenge, &polished) > cand_q {
                                    cand_best = polished;
                                }
                                full_best = Some(cand_best);
                            }
                        }
                        51 => {
                            // F_Q LINKAGE EJECTION CHAIN (i15). Polynomial coordinated
                            // k-flip (Glover/LK): inject a strongly-coupled unselected item
                            // past capacity, coordinate-eject the weakest-margin selected
                            // items to restore feasibility, refill — the exact move class the
                            // atomic <=2-in/<=2-out VND cannot reach (meta-constat: the +32 is
                            // trapped behind coordinated k-flips). Distinct from i14 (no
                            // exhaustive enum) and from strategic oscillation (targeted
                            // linkage-guided chain, not a global capacity sweep). Keep-best.
                            if let Some(fb) = full_best.clone() {
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let mut cand_best = fb.clone();
                                let mut cand_q = eval_feasible(challenge, &fb);
                                for r in 0..2usize {
                                    let (sol, val) = run_one_instance_seeded(
                                        challenge, &hp, 1000 + r, Some(&tsn),
                                        &total_interactions, Some(&fb),
                                    );
                                    if val > cand_q { cand_q = val; cand_best = sol; }
                                }
                                let chained = ejection_chain_linkage(challenge, &cand_best, 12, 16);
                                if eval_feasible(challenge, &chained) > cand_q {
                                    cand_best = chained;
                                }
                                full_best = Some(cand_best);
                            }
                        }
                        52 => {
                            // F_R LINKAGE INTERLEAVE (i15). Alternate the two i15 primitives
                            // so each feeds the other: chain (opens coordinated moves) ->
                            // cluster core (exact-solves the freed block) -> chain again.
                            // Keep-best throughout => Q >= refine2 (53312).
                            if let Some(fb) = full_best.clone() {
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let mut cand_best = fb.clone();
                                let mut cand_q = eval_feasible(challenge, &fb);
                                for r in 0..2usize {
                                    let (sol, val) = run_one_instance_seeded(
                                        challenge, &hp, 1000 + r, Some(&tsn),
                                        &total_interactions, Some(&fb),
                                    );
                                    if val > cand_q { cand_q = val; cand_best = sol; }
                                }
                                let c1 = ejection_chain_linkage(challenge, &cand_best, 12, 12);
                                let q1 = eval_feasible(challenge, &c1);
                                if q1 > cand_q { cand_q = q1; cand_best = c1; }
                                let p1 = linkage_cluster_core(challenge, &cand_best, 12, 12, 18, 6);
                                let qp = eval_feasible(challenge, &p1);
                                if qp > cand_q { cand_q = qp; cand_best = p1; }
                                let c2 = ejection_chain_linkage(challenge, &cand_best, 12, 12);
                                if eval_feasible(challenge, &c2) > cand_q { cand_best = c2; }
                                full_best = Some(cand_best);
                            }
                        }
                        60 | 61 | 62 => {
                            // i16 NEW FAMILY — threshold-accepting acceptance walk on the
                            // refine2 base (F2 seeded escape = 53312 @ ~63s, 38s margin).
                            // All prior refines are strict-improve and saturate at 53327;
                            // this reinvests the freed budget in a controlled-worsening
                            // walk (accept moves the strict VND rejects) to cross the
                            // 2-opt plateau (mem c003-t41-tabu2ex-plateau -> IRTS/TSBMA),
                            // then re-descends with the strict VND. accept_mode = 60->TA,
                            // 61->Great-Deluge, 62->LAHC. Keep-best => Q >= 53312.
                            if let Some(fb) = full_best.clone() {
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let mut cand_best = fb.clone();
                                let mut cand_q = eval_feasible(challenge, &fb);
                                for r in 0..2usize {
                                    let (sol, val) = run_one_instance_seeded(
                                        challenge, &hp, 1000 + r, Some(&tsn),
                                        &total_interactions, Some(&fb),
                                    );
                                    if val > cand_q { cand_q = val; cand_best = sol; }
                                }
                                let accept_mode = refine_mode - 60;
                                let walk_iters: usize = hyperparameters.as_ref()
                                    .and_then(|m| m.get("walk_iters")).and_then(|v| v.as_u64())
                                    .unwrap_or(3000) as usize;
                                let ta_ratio: f64 = hyperparameters.as_ref()
                                    .and_then(|m| m.get("ta_ratio")).and_then(|v| v.as_f64())
                                    .unwrap_or(0.010);
                                let tenure: usize = hyperparameters.as_ref()
                                    .and_then(|m| m.get("ta_tenure")).and_then(|v| v.as_u64())
                                    .unwrap_or(7) as usize;
                                let lahc_len: usize = hyperparameters.as_ref()
                                    .and_then(|m| m.get("lahc_len")).and_then(|v| v.as_u64())
                                    .unwrap_or(50) as usize;
                                let walked = threshold_accept_escape(
                                    challenge, &cand_best, &tsn, hp.two_ex_cap,
                                    walk_iters, accept_mode, ta_ratio, tenure, lahc_len,
                                );
                                if eval_feasible(challenge, &walked) > cand_q {
                                    cand_best = walked;
                                }
                                full_best = Some(cand_best);
                            }
                        }
                        70 => {
                            // i17-A: ENRICHED (union) reduction + the PROVEN champion chain
                            // F1 iterated re-anchor -> F2 seeded escape (refine5 downstream).
                            // The initial reduction is the density∪synergy union (set above);
                            // the richer starting basin then flows through the exact champion
                            // refinement. Strongest test of the reduction-space lever. Keep-best.
                            if let Some(fb0) = full_best.clone() {
                                let mut fb = fb0;
                                // F1 re-anchor on the refined incumbent.
                                let mut mask = vec![false; challenge.num_items];
                                for &i in &fb.items { mask[i] = true; }
                                if let Some((r2, surv2, ti2)) = reduced_cost_reduce_anchored(
                                    challenge, &total_interactions, rc_mode, rc_fraction, Some(&mask),
                                ) {
                                    let (bs2, _) = large_n_restarts(&r2, &hp, &ti2);
                                    if let Some(s2) = bs2 {
                                        let cand = Solution {
                                            items: s2.items.iter().map(|&k| surv2[k]).collect(),
                                        };
                                        if eval_feasible(challenge, &cand) > eval_feasible(challenge, &fb) {
                                            fb = cand;
                                        }
                                    }
                                }
                                // F2 escape seeded from the re-anchored incumbent.
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let mut cand_best = fb.clone();
                                let mut cand_q = eval_feasible(challenge, &fb);
                                for r in 0..2usize {
                                    let (sol, val) = run_one_instance_seeded(
                                        challenge, &hp, 1000 + r, Some(&tsn),
                                        &total_interactions, Some(&fb),
                                    );
                                    if val > cand_q { cand_q = val; cand_best = sol; }
                                }
                                full_best = Some(cand_best);
                            }
                        }
                        71 | 72 => {
                            // i17-B/C: ENRICHED reduction (71 = union anchor, 72 = synergy
                            // anchor; both set in the reduction dispatch above) + F2-only
                            // escape (refine2's ~38s-margin path absorbs the larger enriched
                            // instance). Isolates the reduction-enrichment lever from the F1
                            // re-anchor: any gain here is attributable purely to the space.
                            if let Some(fb) = full_best.clone() {
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let mut cand_best = fb.clone();
                                let mut cand_q = eval_feasible(challenge, &fb);
                                for r in 0..2usize {
                                    let (sol, val) = run_one_instance_seeded(
                                        challenge, &hp, 1000 + r, Some(&tsn),
                                        &total_interactions, Some(&fb),
                                    );
                                    if val > cand_q { cand_q = val; cand_best = sol; }
                                }
                                full_best = Some(cand_best);
                            }
                        }
                        80 => {
                            // F_R POTENTIAL-CORE (i18). Same F2 base as i14 (refine2 =
                            // 53312 @ ~63s, 38s margin), but the EXACT-CORE candidate
                            // selection is picked by surrogate-UB POTENTIAL
                            // (total_interactions[i]/w_i) instead of the item's CURRENT
                            // margin. Diagnosis of the 53327 plateau: i14/i15 rank
                            // unselected core candidates by `contrib` (their marginal in
                            // OUR configuration) — but the item belonging to the 53359
                            // optimum looks marginal NOW (that is why it is excluded); it
                            // only pays off jointly. Its surrogate UB (total_interactions[i]
                            // = sum_j q_ij, the code's own provable marginal upper bound,
                            // l.3707) is HIGH even when its current contrib is low. Ranking
                            // the unselected core by that potential + ejecting the lowest
                            // current-margin selected items lets the exact 2^k solve place
                            // the "eject-marginal / admit-high-potential" swap in ONE core
                            // that a current-margin core never co-locates. Hindsight
                            // a56c42e6 (frontier band r(x)≈0 = where optima live),
                            // 4a581dc0 (LP/surrogate fixing over a core). Keep-best =>
                            // Q >= refine2 (53312) by construction.
                            if let Some(fb) = full_best.clone() {
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let mut cand_best = fb.clone();
                                let mut cand_q = eval_feasible(challenge, &fb);
                                for r in 0..2usize {
                                    let (sol, val) = run_one_instance_seeded(
                                        challenge, &hp, 1000 + r, Some(&tsn),
                                        &total_interactions, Some(&fb),
                                    );
                                    if val > cand_q { cand_q = val; cand_best = sol; }
                                }
                                let polished = potential_core_polish(
                                    challenge, &cand_best, &total_interactions,
                                    14, 14, 20, 6, 0,
                                );
                                if eval_feasible(challenge, &polished) > cand_q {
                                    cand_best = polished;
                                }
                                full_best = Some(cand_best);
                            }
                        }
                        81 => {
                            // F_S FRONTIER-BAND CORE (i18). Distinct from 80 by the SHAPE
                            // of the potential band: 80 takes the GLOBALLY highest-potential
                            // excluded items; 81 takes the excluded items whose surrogate
                            // potential is CLOSEST to the marginal selected item's potential
                            // — the genuine "could-go-either-way" frontier (a56c42e6:
                            // r(x)≈0). These are the items the density anchor was least sure
                            // about, exactly the boundary where a single-anchor reduction
                            // over/under-fixes. Same F2 base + keep-best (>= 53312).
                            if let Some(fb) = full_best.clone() {
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let mut cand_best = fb.clone();
                                let mut cand_q = eval_feasible(challenge, &fb);
                                for r in 0..2usize {
                                    let (sol, val) = run_one_instance_seeded(
                                        challenge, &hp, 1000 + r, Some(&tsn),
                                        &total_interactions, Some(&fb),
                                    );
                                    if val > cand_q { cand_q = val; cand_best = sol; }
                                }
                                let polished = potential_core_polish(
                                    challenge, &cand_best, &total_interactions,
                                    14, 14, 20, 6, 1,
                                );
                                if eval_feasible(challenge, &polished) > cand_q {
                                    cand_best = polished;
                                }
                                full_best = Some(cand_best);
                            }
                        }
                        82 => {
                            // F_T DIRECTED POTENTIAL KICK (i18). Attacks REACHABILITY, not a
                            // tiny exact window: from the F2 incumbent, forcibly eject the k
                            // lowest current-margin selected items AND inject the k highest
                            // surrogate-POTENTIAL excluded items that fit the freed capacity,
                            // in ONE compound move the atomic <=2-move VND (and the <=20-item
                            // exact core) structurally cannot reach — then re-run the seeded
                            // ILS/VND to repair around the injected high-potential items
                            // (MILS learning-driven perturbation, Hindsight 690f54b3). This
                            // is NOT ALNS ruin-recreate (i8 refine6, DEAD): that destroyed
                            // LEAST-CONNECTED items and greedy-refilled by CURRENT margin;
                            // here the injection principle is inverted (highest surrogate
                            // POTENTIAL, the exact excluded items current-margin repair never
                            // re-admits). Sweep the kick radius k, keep-best (>= 53312).
                            if let Some(fb) = full_best.clone() {
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let mut cand_best = fb.clone();
                                let mut cand_q = eval_feasible(challenge, &fb);
                                for r in 0..2usize {
                                    let (sol, val) = run_one_instance_seeded(
                                        challenge, &hp, 1000 + r, Some(&tsn),
                                        &total_interactions, Some(&fb),
                                    );
                                    if val > cand_q { cand_q = val; cand_best = sol; }
                                }
                                for (idx, k_kick) in [4usize, 8, 12].iter().enumerate() {
                                    let kicked = directed_potential_kick(
                                        challenge, &cand_best, &total_interactions, *k_kick,
                                    );
                                    let (rep, rv) = run_one_instance_seeded(
                                        challenge, &hp, 4000 + idx, Some(&tsn),
                                        &total_interactions, Some(&kicked),
                                    );
                                    if rv > cand_q { cand_q = rv; cand_best = rep; }
                                }
                                full_best = Some(cand_best);
                            }
                        }
                        90 => {
                            // F_U FIXPOINT DP<->VND DEEP-CONVERGENCE (i19). The engine
                            // calls dp_refinement_hp then vnd_v2 exactly ONCE per
                            // construction/crossover (l.2020/2035/2044/2068) and never
                            // drives the pair to a FIXPOINT on the FINAL incumbent. Each
                            // dp_refinement_hp re-linearizes contrib at the *current*
                            // selection, so iterating it against vnd_v2 lets a coordinated
                            // multi-item cluster assemble across passes (the +32 profile:
                            // low current margin, high synergy potential — i18 note) that a
                            // single DP+VND pass structurally cannot reach. Base = F2 seeded
                            // escape (refine2 = 53312 @ ~63s) which re-admits locked-out
                            // items; the ~38s free budget funds the convergence loop (mem
                            // 64be75db: reinvest freed budget in deep multi-neighborhood LS
                            // to convergence). NOT an HP sweep of core_half_dp (min_level=8
                            // veto) and NOT ALNS ruin-recreate (i8 refine6 DEAD: greedy
                            // repair, single pass) — this is an iterated exact-linearization
                            // fixpoint. Keep-best => Q >= refine2 (53312) by construction.
                            if let Some(fb) = full_best.clone() {
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let mut cand_best = fb.clone();
                                let mut cand_q = eval_feasible(challenge, &fb);
                                for r in 0..2usize {
                                    let (sol, val) = run_one_instance_seeded(
                                        challenge, &hp, 1000 + r, Some(&tsn),
                                        &total_interactions, Some(&fb),
                                    );
                                    if val > cand_q { cand_q = val; cand_best = sol; }
                                }
                                let polished =
                                    fixpoint_dp_vnd_polish(challenge, &cand_best, &hp, &tsn, 12);
                                if eval_feasible(challenge, &polished) > cand_q {
                                    cand_best = polished;
                                }
                                full_best = Some(cand_best);
                            }
                        }
                        91 => {
                            // F_V ESCAPE<->FIXPOINT ALTERNATION (i19). Distinct composition:
                            // alternate a full-instance seeded escape (re-admit locked-out
                            // items) with a DP<->VND fixpoint convergence, feeding each
                            // converged incumbent back as the launch point of the next
                            // escape. The escape opens the door (new items enter), the
                            // fixpoint then deeply re-optimizes AROUND them by successive
                            // re-linearization — a longer trajectory than F2's flat 2
                            // escapes. Keep-best over all cycles => Q >= 53312.
                            if let Some(fb) = full_best.clone() {
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let mut cur = fb.clone();
                                let mut cand_best = fb.clone();
                                let mut cand_q = eval_feasible(challenge, &fb);
                                for cyc in 0..3usize {
                                    let (sol, val) = run_one_instance_seeded(
                                        challenge, &hp, 1000 + cyc, Some(&tsn),
                                        &total_interactions, Some(&cur),
                                    );
                                    let esc = if val > eval_feasible(challenge, &cur) {
                                        sol
                                    } else {
                                        cur.clone()
                                    };
                                    let pol =
                                        fixpoint_dp_vnd_polish(challenge, &esc, &hp, &tsn, 10);
                                    let pq = eval_feasible(challenge, &pol);
                                    if pq > cand_q { cand_q = pq; cand_best = pol.clone(); }
                                    cur = pol;
                                }
                                full_best = Some(cand_best);
                            }
                        }
                        92 => {
                            // F_W FIXPOINT OVER DISTINCT BASINS (i19). Apply the deep
                            // DP<->VND fixpoint not only to the champion incumbent but also
                            // to the top Hamming-distant restart basins already in `pool`
                            // (the same genuinely-distant starts refine4 draws PR guides
                            // from). Depth-of-convergence x breadth-of-basin: a coordinated
                            // cluster may only be reachable from a structurally different
                            // basin. Champion incumbent is always a candidate => keep-best
                            // >= 53312.
                            if let Some(fb) = full_best.clone() {
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let mut cand_best = fb.clone();
                                let mut cand_q = eval_feasible(challenge, &fb);
                                let pol0 = fixpoint_dp_vnd_polish(challenge, &fb, &hp, &tsn, 12);
                                let pq0 = eval_feasible(challenge, &pol0);
                                if pq0 > cand_q { cand_q = pq0; cand_best = pol0; }
                                let mut pool_sorted: Vec<&(Solution, i64)> = pool.iter().collect();
                                pool_sorted.sort_unstable_by_key(|&&(_, v)| std::cmp::Reverse(v));
                                for &(s, _) in pool_sorted.iter().take(3) {
                                    let full: Vec<usize> =
                                        s.items.iter().map(|&kk| survivors[kk]).collect();
                                    let seed = Solution { items: full };
                                    let pol =
                                        fixpoint_dp_vnd_polish(challenge, &seed, &hp, &tsn, 10);
                                    let q = eval_feasible(challenge, &pol);
                                    if q > cand_q { cand_q = q; cand_best = pol; }
                                }
                                full_best = Some(cand_best);
                            }
                        }
                        130 => {
                            // F_X POOL-DISAGREEMENT exact core (i20). F2 escape first
                            // (re-admit locked-out items, same 2 seeded restarts as
                            // refine2), then exact-solve the block of items the restart
                            // basins DISAGREE on — the >=3-item coordinated-move frontier
                            // the margin core (i13/i14) and the <=2-move VND both miss.
                            // Base F2 leaves ~38s under 101530ms for the 2^20 core solve.
                            // Keep-best => >= 53312.
                            if let Some(fb0) = full_best.clone() {
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let mut fb = fb0.clone();
                                let mut fq = eval_feasible(challenge, &fb0);
                                for r in 0..2usize {
                                    let (sol, val) = run_one_instance_seeded(
                                        challenge, &hp, 1000 + r, Some(&tsn),
                                        &total_interactions, Some(&fb0),
                                    );
                                    if val > fq { fq = val; fb = sol; }
                                }
                                let basins =
                                    build_pool_basins(challenge, &pool, &survivors, &fb);
                                let cap = n_escapes.unwrap_or(20);
                                let polished = pool_disagreement_core_polish(
                                    challenge, &fb, &basins, cap, 8,
                                );
                                if eval_feasible(challenge, &polished) > fq {
                                    full_best = Some(polished);
                                } else {
                                    full_best = Some(fb);
                                }
                            }
                        }
                        131 => {
                            // F_Y CHAMPION-CHAIN + DISAGREEMENT (i20 combine). Stack the
                            // champion refine5 winning prefix (F1 re-anchor -> F2 escape)
                            // with the pool-disagreement exact core: the re-anchor lifts
                            // the frontier, the escape enriches it, THEN the exact block
                            // solve co-locates the coordinated flip on the enriched basins.
                            // Tests whether the +32 needs the champion's lift BEFORE the
                            // disagreement solve. Keep-best throughout => >= 53312.
                            if let Some(fb0) = full_best.clone() {
                                let mut fb = fb0;
                                // F1 re-anchor.
                                let mut mask = vec![false; challenge.num_items];
                                for &i in &fb.items { mask[i] = true; }
                                if let Some((r2, surv2, ti2)) = reduced_cost_reduce_anchored(
                                    challenge, &total_interactions, rc_mode, rc_fraction, Some(&mask),
                                ) {
                                    let (bs2, _) = large_n_restarts(&r2, &hp, &ti2);
                                    if let Some(s2) = bs2 {
                                        let cand = Solution {
                                            items: s2.items.iter().map(|&k| surv2[k]).collect(),
                                        };
                                        if eval_feasible(challenge, &cand) > eval_feasible(challenge, &fb) {
                                            fb = cand;
                                        }
                                    }
                                }
                                // F2 escape from the re-anchored incumbent.
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let mut cand_best = fb.clone();
                                let mut cand_q = eval_feasible(challenge, &fb);
                                for r in 0..2usize {
                                    let (sol, val) = run_one_instance_seeded(
                                        challenge, &hp, 1000 + r, Some(&tsn),
                                        &total_interactions, Some(&fb),
                                    );
                                    if val > cand_q { cand_q = val; cand_best = sol; }
                                }
                                // Pool-disagreement exact core on the enriched incumbent.
                                let basins =
                                    build_pool_basins(challenge, &pool, &survivors, &cand_best);
                                let cap = n_escapes.unwrap_or(20);
                                let polished = pool_disagreement_core_polish(
                                    challenge, &cand_best, &basins, cap, 8,
                                );
                                if eval_feasible(challenge, &polished) > cand_q {
                                    full_best = Some(polished);
                                } else {
                                    full_best = Some(cand_best);
                                }
                            }
                        }
                        132 => {
                            // F_Z VND-CONVERGED DISAGREEMENT (i20). Raw restart basins are
                            // partially-converged; their disagreement is noisy. First drive
                            // the top pool basins to a LOCAL OPTIMUM with the active VND, so
                            // the disagreement set reflects competing OPTIMA (the true basin
                            // frontier) rather than transient restart noise, then exact-solve
                            // it. Distinct kernelization input from 130. Keep-best => >= 53312.
                            if let Some(fb0) = full_best.clone() {
                                let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
                                let mut fb = fb0.clone();
                                let mut fq = eval_feasible(challenge, &fb0);
                                for r in 0..2usize {
                                    let (sol, val) = run_one_instance_seeded(
                                        challenge, &hp, 1000 + r, Some(&tsn),
                                        &total_interactions, Some(&fb0),
                                    );
                                    if val > fq { fq = val; fb = sol; }
                                }
                                // VND-converge the top-4 pool basins (full space).
                                let mut pool_sorted: Vec<&(Solution, i64)> = pool.iter().collect();
                                pool_sorted.sort_unstable_by_key(|&&(_, v)| std::cmp::Reverse(v));
                                let mut conv: Vec<Vec<bool>> = Vec::new();
                                for &(s, _) in pool_sorted.iter().take(4) {
                                    let mut st = State::new_empty(challenge);
                                    for &kk in &s.items { st.add_item(survivors[kk]); }
                                    local_search_vnd_tsn(&mut st, &tsn, hp.two_ex_cap);
                                    conv.push(st.selected_bit.clone());
                                }
                                // Add the escaped incumbent as a basin.
                                let mut bits = vec![false; challenge.num_items];
                                for &i in &fb.items { bits[i] = true; }
                                conv.push(bits);
                                let cap = n_escapes.unwrap_or(20);
                                let polished = pool_disagreement_core_polish(
                                    challenge, &fb, &conv, cap, 8,
                                );
                                if eval_feasible(challenge, &polished) > fq {
                                    full_best = Some(polished);
                                } else {
                                    full_best = Some(fb);
                                }
                            }
                        }
                        _ => {}
                    }
                    return Ok(full_best);
                }
                // Reduction not applicable (too little / too much removed) -> fall
                // through to the baseline path below.
            }

            let (mut best_sol, pool) = large_n_restarts(challenge, &hp, &total_interactions);
            // i26: ported consensus core-DP finisher over the FULL-instance restart pool
            // — this is the faithful reconstruction of prom_pulse's monolithic solve
            // (ARM_A {bp1,rc0,refine0}: 53110 -> target ~53359). Only fires when
            // pool_core_dp>0; byte-exact otherwise.
            if pool_core_dp > 0 {
                if let Some(bs) = best_sol.clone() {
                    if let Some(imp) = consensus_core_dp(challenge, &pool, &bs, &total_interactions, elite_vote_k, core_disputed_cap) {
                        best_sol = Some(imp);
                    }
                }
            }

            // Variable-fixation reduce-then-intensify (I2PLS/Pisinger).
            // fix_mode read from hp_json: 0 = sentinel (OFF, reproduces baseline),
            // 1 = restart-consensus, 2 = density-band, 3 = hub-connectivity.
            let fix_mode: usize = hyperparameters.as_ref()
                .and_then(|m| m.get("fix_mode")).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let core_cap: usize = hyperparameters.as_ref()
                .and_then(|m| m.get("core_cap")).and_then(|v| v.as_u64()).unwrap_or(1100) as usize;

            if fix_mode == 0 || pool.is_empty() || best_sol.is_none() {
                return Ok(best_sol);
            }
            let best_ref = best_sol.as_ref().unwrap();
            if let Some(cand) = fixation_intensify(
                challenge, hyperparameters, &pool, best_ref, &total_interactions, fix_mode, core_cap,
            ) {
                if eval_feasible(challenge, &cand) > eval_feasible(challenge, best_ref) {
                    return Ok(Some(cand));
                }
            }
            return Ok(best_sol);
        }

        return Ok(Some(solve_rich_small(challenge, hp, total_interactions)));
    }
}

/// i26 (min_level=8, COMBINE src_algo91 prom_pulse_marlin_v2): the MISSING consensus
/// core-DP finisher, ported verbatim from prom_pulse's monolithic n>1200 `solve`
/// (code_raw/mod.rs:3272-3388). Diagnostic ARM_A `{base_profile:1,rc0,refine0}`=53110
/// runs the identical deep schedule as SOTA prom_pulse (53359) at the same ~84s yet
/// loses -249 Q — a NON-FAITHFUL PORT (cf mem c006-t27-gap-is-nonfaithful-port). Root
/// cause: prom_pulse FINISHES every restart pool with a consensus core-DP; our refacto
/// factored the restarts into `large_n_restarts` but kept the finisher ONLY inside
/// `solve_rich_small` (n<=1200 / fixation-reduced path), so the rc_mode0 full pool and
/// the rc_mode5 reduced pool never receive it.
///
/// Mechanism: freq-vote the top-6 elites of `pool`; items in ALL 6 (`always_in`) are
/// locked; items in some-but-not-all (`disputed`) are re-optimised by an exact 0-1 DP
/// (truncated to the top-200 by disagreement+best_bonus+hub when >200, seeding the
/// out-of-core items of `best` so nothing is lost), then a VND+DP polish. Returns the
/// improved solution (same index space as `challenge`/`pool`) or None if no strict
/// gain. Never regresses (starts from `best`, keep-if-better).
fn consensus_core_dp(
    challenge: &Challenge,
    pool: &[(Solution, i64)],
    best: &Solution,
    total_interactions: &[i64],
    elite_vote_k: usize,
    core_disputed_cap: usize,
) -> Option<Solution> {
    let n = challenge.num_items;
    let deep_dp = 150usize;
    if pool.is_empty() { return None; }

    // i27: 0 => ported-verbatim defaults (byte-exact with i26 V3).
    let vote_k = if elite_vote_k == 0 { 6 } else { elite_vote_k };
    let core_cap = if core_disputed_cap == 0 { 200 } else { core_disputed_cap };

    let mut elite: Vec<&(Solution, i64)> = pool.iter().collect();
    elite.sort_by_key(|&&(_, v)| std::cmp::Reverse(v));
    let tk = elite.len().min(vote_k);
    if tk == 0 { return None; }
    let mut item_freq = vec![0usize; n];
    for e in elite[..tk].iter() {
        for &i in &e.0.items { item_freq[i] += 1; }
    }
    let always_in: Vec<usize> = (0..n).filter(|&i| item_freq[i] == tk).collect();
    let disputed: Vec<usize> = (0..n).filter(|&i| item_freq[i] > 0 && item_freq[i] < tk).collect();
    if disputed.is_empty() { return None; }

    // Baseline value on the SAME scale as State::total_value (avoid eval_solution offset).
    let best_sel = &best.items;
    let mut best_state = State::new_empty(challenge);
    for &i in best_sel {
        if best_state.total_weight + challenge.weights[i] <= challenge.max_weight {
            best_state.add_item(i);
        }
    }
    let orig_val = best_state.total_value;
    let mut best_val = orig_val;
    let mut out_sel = best_sel.clone();

    // top-200 truncation by (disagreement + best_bonus)*1e4 + hub (prom_pulse 3282-3298).
    let mut core_disputed = disputed.clone();
    let use_best_outside_core = core_disputed.len() > core_cap;
    if use_best_outside_core {
        let mut in_best = vec![false; n];
        for &i in best_sel { in_best[i] = true; }
        core_disputed.sort_unstable_by_key(|&i| {
            let disagreement = item_freq[i] * (tk - item_freq[i]);
            let best_bonus = if in_best[i] { tk } else { 0 };
            let w = (challenge.weights[i] as i64).max(1);
            let hub = total_interactions[i] / w;
            std::cmp::Reverse((disagreement + best_bonus) as i64 * 10_000 + hub)
        });
        core_disputed.truncate(core_cap);
        core_disputed.sort_unstable();
    }
    let mut in_core = vec![false; n];
    for &i in &core_disputed { in_core[i] = true; }

    let mut state = State::new_empty(challenge);
    for &i in &always_in {
        if state.total_weight + challenge.weights[i] <= challenge.max_weight {
            state.add_item(i);
        }
    }
    if use_best_outside_core {
        for &i in best_sel {
            if !in_core[i]
                && !state.selected_bit[i]
                && state.total_weight + challenge.weights[i] <= challenge.max_weight
            {
                state.add_item(i);
            }
        }
    }
    let fixed_weight = state.total_weight;
    let rem_cap = (challenge.max_weight - fixed_weight) as usize;
    if rem_cap == 0 { return None; }

    let dk = core_disputed.len();
    let mut total_disp_weight: usize = 0;
    for &it in &core_disputed { total_disp_weight += challenge.weights[it] as usize; }
    let myw = rem_cap.min(total_disp_weight);
    let dp_size = myw + 1;
    if dp_size > 2_000_000 { return None; }

    let mut dp = vec![i64::MIN / 4; dp_size];
    let mut choose = vec![0u8; dk * dp_size];
    dp[0] = 0;
    let mut w_hi: usize = 0;
    for (t, &it) in core_disputed.iter().enumerate() {
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

    let mut w_star = (0..=myw).max_by_key(|&w| dp[w]).unwrap_or(0);
    let mut dp_selected = Vec::new();
    for t in (0..dk).rev() {
        let it = core_disputed[t];
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
        best_val = state.total_value;
        out_sel = state.selected_items();
    }

    if best_val > orig_val { Some(Solution { items: out_sel }) } else { None }
}

/// Bare large-n (n>1200) restart driver, factored out so it can run on BOTH the
/// full instance and the reduced survivor sub-Challenge produced by the head
/// reduced-cost lock-out (i5). Byte-identical logic to the previous inline loop:
/// same `n_restarts`, same seeded/unseeded alternation, same stall break — so the
/// rc_mode=0 sentinel reproduces the baseline path exactly.
fn large_n_restarts(
    challenge: &Challenge,
    hp: &Hparams,
    total_interactions: &[i64],
) -> (Option<Solution>, Vec<(Solution, i64)>) {
    let n_restarts = hp.n_full_restarts.max(1);
    let tsn = TopNeighbors::new(challenge, hp.tsn_k, hp.use_sparse_me, hp.win_incr, hp.use_sparse_swap, hp.use_flat_iv, hp.use_flat_iv16, hp.use_iv_transpose, hp.use_bb_screen, hp.use_screen_break, hp.use_light_screen_break, hp.use_light_win_incr, hp.use_light_fast_replace, hp.use_rflip_screen);
    let mut best_sol: Option<Solution> = None;
    let mut best_quality: i64 = i64::MIN;
    let mut restart_stall = 0usize;
    let mut pool: Vec<(Solution, i64)> = Vec::new();
    for restart in 0..n_restarts {
        let (sol, val) = if restart >= 2 && restart % 2 == 0 && best_sol.is_some() {
            run_one_instance_seeded(challenge, hp, restart, Some(&tsn), total_interactions, best_sol.as_ref())
        } else {
            run_one_instance(challenge, hp, restart, Some(&tsn), total_interactions)
        };
        pool.push((sol.clone(), val));
        if val > best_quality {
            best_quality = val;
            best_sol = Some(sol);
            restart_stall = 0;
        } else {
            restart_stall += 1;
            if restart_stall >= 5 { break; }
        }
    }
    (best_sol, pool)
}

/// F_C (i8) ALNS ruin-and-recreate for the large-n refine (Hindsight ba100407:
/// destroy ~20% least-connected items + greedy repair, 3-8% expected on large-n
/// t43). Remove the selected items with the LOWEST realized contribution (least
/// coupled to the rest of the solution), greedily re-fill the freed capacity by
/// best marginal contribution, then VND/DP polish. Deterministic (fixed 20% cut,
/// deterministic greedy), feasibility-safe (cap-checked adds, only positive
/// marginals inserted). Reuses the proven VND/DP polish bricks.
fn alns_ruin_recreate(challenge: &Challenge, sol: &Solution, hp: &Hparams) -> Solution {
    let n = challenge.num_items;
    let mut state = State::new_empty(challenge);
    for &i in &sol.items { state.add_item(i); }
    // destroy: 20% of selected items with the lowest realized contribution.
    let mut sel: Vec<usize> = sol.items.clone();
    sel.sort_by_key(|&i| state.contrib[i] as i64);
    let k = ((sel.len() as f64) * 0.20) as usize;
    for &i in sel.iter().take(k) { state.remove_item(i); }
    // recreate: greedy best positive marginal contribution among fitting items.
    let cap = challenge.max_weight;
    loop {
        let mut best_i = usize::MAX;
        let mut best_c = 0i64;
        for i in 0..n {
            if state.selected_bit[i] { continue; }
            if state.total_weight + challenge.weights[i] > cap { continue; }
            let c = state.contrib[i] as i64;
            if c > best_c { best_c = c; best_i = i; }
        }
        if best_i == usize::MAX { break; }
        state.add_item(best_i);
    }
    // polish with the proven windowed VND + core DP refinement.
    loop {
        let v_before = state.total_value;
        local_search_vnd_windowed_deep(&mut state, hp.window_k);
        dp_refinement_hp(&mut state, hp.core_half_dp);
        if state.total_value <= v_before { break; }
    }
    Solution { items: state.selected_items() }
}

/// Head-of-pipeline reduced-cost / surrogate lock-out (L6 structural, i5).
///
/// c003 QKP values are 0 => all worth is pairwise synergy q_ij >= 0. Therefore
/// `total_interactions[i] = sum_j q_ij` is a valid UPPER BOUND on item i's marginal
/// contribution in ANY solution (contrib_i = sum_{j selected} q_ij <= sum_all q_ij).
/// An item whose best-case synergy, priced against its weight, is dominated is
/// provably weight-inefficient and can be fixed to 0.
///
/// This is DISTINCT from the dead consensus fixing (i3/i4):
///  * scoring = intrinsic surrogate UB (incumbent-INDEPENDENT), not incumbent contrib;
///  * action = LOCK-OUT ONLY (never lock in) => survivors always feasible on full inst;
///  * a greedy density incumbent forms a frontier band that is NEVER locked out.
///
/// rc_mode: 1 = optimistic-density rank (bottom `rc_fraction` by ti_i/w_i),
///          2 = Lagrangian reduced cost r_i = ti_i - lambda*w_i < 0 (auto lambda),
///          3 = exact-safe (isolated ti==0 / over-budget) + mild density rank.
///          4 = STACK: exact-safe filter (ti==0 || w>cap, provably fixable to 0,
///              mem 5e889c83) UNCONDITIONAL + aggressive density tail (mode1 rank).
///          5 = pairwise-frontier scoring: rank the non-incumbent tail by REALIZED
///              synergy against the greedy frontier band only, sum_{j in incumbent}
///              q_ij / w_i, instead of the diffuse all-j total-interaction UB.
///              Concentrates lock-out on items whose synergy is with weak items
///              (mem 3ca65b92 pairwise interactions, mem 95824a48 positive-interaction
///              with selected). Frontier members are still never locked out.
///          6 = mode4 (+) mode5: exact-safe filter + pairwise-frontier ranking.
/// Modes 4/5/6 (i6 combine) preserve the LOCK-OUT-ONLY invariant (incumbent frontier
/// band is never removed -> every survivor solution stays feasible on the full
/// instance, no feasibility-tolerance risk), and never touch rc_mode 0/1/2/3.
/// Returns None when the reduction would be trivial or excessive.
fn reduced_cost_reduce(
    challenge: &Challenge,
    total_interactions: &[i64],
    rc_mode: usize,
    rc_fraction: f64,
) -> Option<(Challenge, Vec<usize>, Vec<i64>)> {
    // Single-anchor: the frontier band is the greedy DENSITY incumbent (i5/i6).
    reduced_cost_reduce_anchored(challenge, total_interactions, rc_mode, rc_fraction, None)
}

/// i7 (min_level=7 combine): reduce anchored on an EXTERNAL incumbent (frontier
/// band). `incumbent_override=None` reproduces the i6 greedy-density anchor
/// byte-exact (sentinel). Passing a refined solution's mask re-anchors the
/// lock-out on a better frontier (F1 iterated re-anchor); the union path (F3)
/// combines two anchors. Feasibility invariant is preserved: any incumbent
/// member is never locked out, so every survivor solution stays feasible on the
/// full instance.
fn reduced_cost_reduce_anchored(
    challenge: &Challenge,
    total_interactions: &[i64],
    rc_mode: usize,
    rc_fraction: f64,
    incumbent_override: Option<&[bool]>,
) -> Option<(Challenge, Vec<usize>, Vec<i64>)> {
    let in_incumbent = match incumbent_override {
        Some(m) => m.to_vec(),
        None => build_density_incumbent(challenge, total_interactions),
    };
    let locked_out = compute_locked_out(challenge, total_interactions, rc_mode, rc_fraction, &in_incumbent);
    build_reduced_from_survivors(challenge, &locked_out)
}

/// i7 (F3 union-of-anchors): survivor set = UNION over two differently-shaped
/// anchor incumbents (greedy density + greedy raw-synergy). An item is locked out
/// only if BOTH anchors lock it, so the reduced instance keeps the union of both
/// frontier bands — a richer, less density-biased search space (mem 14741428:
/// escape the attractor by changing the space, not by more same-space LS).
fn reduced_cost_reduce_union(
    challenge: &Challenge,
    total_interactions: &[i64],
    rc_mode: usize,
    rc_fraction: f64,
) -> Option<(Challenge, Vec<usize>, Vec<i64>)> {
    let inc1 = build_density_incumbent(challenge, total_interactions);
    let l1 = compute_locked_out(challenge, total_interactions, rc_mode, rc_fraction, &inc1);
    let inc2 = build_synergy_incumbent(challenge, total_interactions);
    let l2 = compute_locked_out(challenge, total_interactions, rc_mode, rc_fraction, &inc2);
    let n = challenge.num_items;
    let locked_out: Vec<bool> = (0..n).map(|i| l1[i] && l2[i]).collect();
    build_reduced_from_survivors(challenge, &locked_out)
}

/// Greedy DENSITY incumbent (ti_i/w_i desc, cap-fill). Its members form the
/// frontier band that is never locked out. Byte-exact to the i6 inline path.
fn build_density_incumbent(challenge: &Challenge, total_interactions: &[i64]) -> Vec<bool> {
    let n = challenge.num_items;
    let cap = challenge.max_weight as u64;
    let density = |i: usize| -> f64 {
        total_interactions[i] as f64 / (challenge.weights[i] as f64).max(1.0)
    };
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| density(b).partial_cmp(&density(a)).unwrap_or(std::cmp::Ordering::Equal));
    let mut in_incumbent = vec![false; n];
    let mut used: u64 = 0;
    for &i in &order {
        let w = challenge.weights[i] as u64;
        if used + w <= cap { in_incumbent[i] = true; used += w; }
    }
    in_incumbent
}

/// Greedy RAW-SYNERGY incumbent (total_interactions[i] desc, cap-fill) — a second
/// anchor whose frontier band differs from the density one (favours high-synergy
/// heavy items). Used only by the F3 union reduction.
fn build_synergy_incumbent(challenge: &Challenge, total_interactions: &[i64]) -> Vec<bool> {
    let n = challenge.num_items;
    let cap = challenge.max_weight as u64;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| total_interactions[b].cmp(&total_interactions[a]));
    let mut in_incumbent = vec![false; n];
    let mut used: u64 = 0;
    for &i in &order {
        let w = challenge.weights[i] as u64;
        if used + w <= cap { in_incumbent[i] = true; used += w; }
    }
    in_incumbent
}

/// Given a frontier-band incumbent, compute the lock-out mask for `rc_mode`.
/// Match arms are byte-identical to the i6 inline logic.
fn compute_locked_out(
    challenge: &Challenge,
    total_interactions: &[i64],
    rc_mode: usize,
    rc_fraction: f64,
    in_incumbent: &[bool],
) -> Vec<bool> {
    let n = challenge.num_items;
    let cap = challenge.max_weight as u64;
    let density = |i: usize| -> f64 {
        total_interactions[i] as f64 / (challenge.weights[i] as f64).max(1.0)
    };
    let used: u64 = (0..n).filter(|&i| in_incumbent[i]).map(|i| challenge.weights[i] as u64).sum();
    let lambda = {
        let mut ti_sel: i64 = 0;
        for i in 0..n { if in_incumbent[i] { ti_sel += total_interactions[i]; } }
        if used > 0 { ti_sel as f64 / used as f64 } else { 0.0 }
    };

    let mut locked_out = vec![false; n];
    match rc_mode {
        2 => {
            for i in 0..n {
                if in_incumbent[i] { continue; }
                let r = total_interactions[i] as f64 - lambda * (challenge.weights[i] as f64);
                if r < 0.0 { locked_out[i] = true; }
            }
        }
        3 => {
            for i in 0..n {
                if in_incumbent[i] { continue; }
                if total_interactions[i] == 0 || challenge.weights[i] as u64 > cap {
                    locked_out[i] = true;
                }
            }
            let mut cand: Vec<usize> = (0..n).filter(|&i| !in_incumbent[i] && !locked_out[i]).collect();
            cand.sort_by(|&a, &b| density(a).partial_cmp(&density(b)).unwrap_or(std::cmp::Ordering::Equal));
            let k = ((cand.len() as f64) * rc_fraction * 0.5) as usize;
            for &i in cand.iter().take(k) { locked_out[i] = true; }
        }
        4 => {
            // STACK (i6): exact-safe fixables (isolated ti==0 / over-budget) locked
            // UNCONDITIONALLY, then the aggressive mode1 density tail on the rest.
            for i in 0..n {
                if in_incumbent[i] { continue; }
                if total_interactions[i] == 0 || challenge.weights[i] as u64 > cap {
                    locked_out[i] = true;
                }
            }
            let mut cand: Vec<usize> = (0..n).filter(|&i| !in_incumbent[i] && !locked_out[i]).collect();
            cand.sort_by(|&a, &b| density(a).partial_cmp(&density(b)).unwrap_or(std::cmp::Ordering::Equal));
            let k = ((cand.len() as f64) * rc_fraction) as usize;
            for &i in cand.iter().take(k) { locked_out[i] = true; }
        }
        5 => {
            // PAIRWISE-FRONTIER (i6): score the non-incumbent tail by realized synergy
            // against the greedy frontier band only, not the diffuse all-j UB.
            let inc_idx: Vec<usize> = (0..n).filter(|&i| in_incumbent[i]).collect();
            let fscore = |i: usize| -> f64 {
                let row = &challenge.interaction_values[i];
                let mut s: i64 = 0;
                for &j in &inc_idx { s += row[j] as i64; }
                s as f64 / (challenge.weights[i] as f64).max(1.0)
            };
            let cand: Vec<usize> = (0..n).filter(|&i| !in_incumbent[i]).collect();
            let sc: Vec<f64> = cand.iter().map(|&i| fscore(i)).collect();
            let mut idx: Vec<usize> = (0..cand.len()).collect();
            idx.sort_by(|&a, &b| sc[a].partial_cmp(&sc[b]).unwrap_or(std::cmp::Ordering::Equal));
            let k = ((cand.len() as f64) * rc_fraction) as usize;
            for &t in idx.iter().take(k) { locked_out[cand[t]] = true; }
        }
        6 => {
            // STACK + PAIRWISE (i6): exact-safe filter, then pairwise-frontier rank
            // on the remaining non-incumbent tail.
            for i in 0..n {
                if in_incumbent[i] { continue; }
                if total_interactions[i] == 0 || challenge.weights[i] as u64 > cap {
                    locked_out[i] = true;
                }
            }
            let inc_idx: Vec<usize> = (0..n).filter(|&i| in_incumbent[i]).collect();
            let fscore = |i: usize| -> f64 {
                let row = &challenge.interaction_values[i];
                let mut s: i64 = 0;
                for &j in &inc_idx { s += row[j] as i64; }
                s as f64 / (challenge.weights[i] as f64).max(1.0)
            };
            let cand: Vec<usize> = (0..n).filter(|&i| !in_incumbent[i] && !locked_out[i]).collect();
            let sc: Vec<f64> = cand.iter().map(|&i| fscore(i)).collect();
            let mut idx: Vec<usize> = (0..cand.len()).collect();
            idx.sort_by(|&a, &b| sc[a].partial_cmp(&sc[b]).unwrap_or(std::cmp::Ordering::Equal));
            let k = ((cand.len() as f64) * rc_fraction) as usize;
            for &t in idx.iter().take(k) { locked_out[cand[t]] = true; }
        }
        7 => {
            // VALUE-AWARE PAIRWISE-FRONTIER (i38, min_level=8 replace). mode5's
            // frontier-synergy score `Σ_{j∈inc} I[i][j] / w_i` OMITS the linear
            // value term `values[i]`, yet the QKP objective is
            // `total_value = Σ values[i] + Σ_pairs I[i][j]` (l.193 set_all_selected,
            // contrib[i] = values[i] + total_interactions[i]). Ranking the lock-out
            // tail by interaction synergy ALONE can wrongly lock a high-linear-value
            // item (or keep a low-value one), so the reduced instance either drops
            // productive items or wastes core slots. Here the score is the TRUE
            // marginal-contribution density `(values[i] + Σ_{j∈inc} I[i][j]) / w_i`,
            // consistent with the objective and with the value-aware constructions
            // already used elsewhere (build_greedy_synergy_weight l.318, value-sort
            // l.295). More accurate separation ⇒ same rc_fraction keeps better
            // survivors (+Q, fast corner), OR a larger rc_fraction stays Q-safe
            // (smaller/faster reduced instance under budget). Hindsight 4f4c9fdc
            // (two-mode scoring), 122c1281 (accurate variable reduction). mode5
            // remains byte-exact (untouched) ⇒ {rc_mode:5} is the drift-guard sentinel.
            let inc_idx: Vec<usize> = (0..n).filter(|&i| in_incumbent[i]).collect();
            let fscore = |i: usize| -> f64 {
                let row = &challenge.interaction_values[i];
                let mut s: i64 = challenge.values[i] as i64;
                for &j in &inc_idx { s += row[j] as i64; }
                s as f64 / (challenge.weights[i] as f64).max(1.0)
            };
            let cand: Vec<usize> = (0..n).filter(|&i| !in_incumbent[i]).collect();
            let sc: Vec<f64> = cand.iter().map(|&i| fscore(i)).collect();
            let mut idx: Vec<usize> = (0..cand.len()).collect();
            idx.sort_by(|&a, &b| sc[a].partial_cmp(&sc[b]).unwrap_or(std::cmp::Ordering::Equal));
            let k = ((cand.len() as f64) * rc_fraction) as usize;
            for &t in idx.iter().take(k) { locked_out[cand[t]] = true; }
        }
        8 => {
            // ADAPTIVE VALUE-AWARE SAFE REDUCTION (i40, min_level=8 replace).
            // mode7 (i38) made the lock-out score objective-coherent (value-aware
            // marginal density `(values[i] + Σ_{j∈inc} I[i][j]) / w_i`) but still
            // locks a FIXED `rc_fraction` regardless of the score distribution, so
            // it can never shrink the reduced instance beyond that hand-set fraction
            // => i38 reproduced 53375 at the SAME instance size and time (job 24290:
            // 167s == sentinel, 0 time gain). The un-exploited TIME lever is to let
            // the reduction SELF-CALIBRATE: an item whose value-aware marginal
            // density is below the WEAKEST incumbent-frontier item (`theta`) is
            // frontier-dominated — it scores worse per unit weight than an item we
            // already chose to keep — so it is very unlikely to belong to an
            // improving solution and can be locked. This locks MORE when the
            // frontier is sharp (smaller/faster reduced core) and less when flat
            // (Q-safe), with NO tuned fraction. Q is protected three ways:
            //   (a) the value-aware, objective-consistent ranking (Hindsight 559b50cb:
            //       "prune items whose UB on contribution is below the current best
            //       solution's average per-item value"; a56c42e6: reduced-cost r(x)<<0
            //       => locked-out);
            //   (b) the full-instance F2 escape (escape_rc_fraction=0) can re-admit
            //       any wrongly-locked item — the reduction is not final;
            //   (c) downside-bounded keep-best on the full challenge.
            // `rc_fraction` is reused ONLY as a LOWER bound on the lock count (the
            // mode7-equivalent floor), so V1 is never a LARGER instance / slower than
            // mode7 — worst case it degenerates to mode7. mode5/7 untouched =>
            // {rc_mode:5} remains the byte-exact 53375 drift-guard sentinel.
            // Hindsight: 122c1281 (accurate variable reduction), 4f4c9fdc (two-mode
            // separable screen), 775aef23 (fewer distinct items => less compute time).
            let inc_idx: Vec<usize> = (0..n).filter(|&i| in_incumbent[i]).collect();
            let vscore = |i: usize| -> f64 {
                let row = &challenge.interaction_values[i];
                let mut s: i64 = challenge.values[i] as i64;
                for &j in &inc_idx { s += row[j] as i64; }
                s as f64 / (challenge.weights[i] as f64).max(1.0)
            };
            // theta = weakest incumbent-frontier value-aware marginal density.
            let mut theta = f64::INFINITY;
            for &i in &inc_idx {
                let sc_i = vscore(i);
                if sc_i < theta { theta = sc_i; }
            }
            let cand: Vec<usize> = (0..n).filter(|&i| !in_incumbent[i]).collect();
            let sc: Vec<f64> = cand.iter().map(|&i| vscore(i)).collect();
            // Fraction FLOOR (mode7-equivalent lock set): lock the bottom `k` by score.
            let mut idx: Vec<usize> = (0..cand.len()).collect();
            idx.sort_by(|&a, &b| sc[a].partial_cmp(&sc[b]).unwrap_or(std::cmp::Ordering::Equal));
            let k_floor = ((cand.len() as f64) * rc_fraction) as usize;
            for &t in idx.iter().take(k_floor) { locked_out[cand[t]] = true; }
            // Adaptive: additionally lock every candidate frontier-dominated by theta.
            for (t, &i) in cand.iter().enumerate() {
                if sc[t] < theta { locked_out[i] = true; }
            }
        }
        _ => {
            let mut cand: Vec<usize> = (0..n).filter(|&i| !in_incumbent[i]).collect();
            cand.sort_by(|&a, &b| density(a).partial_cmp(&density(b)).unwrap_or(std::cmp::Ordering::Equal));
            let k = ((cand.len() as f64) * rc_fraction) as usize;
            for &i in cand.iter().take(k) { locked_out[i] = true; }
        }
    }
    locked_out
}

/// Build the reduced sub-Challenge over the survivor items (those NOT locked out).
/// Byte-identical to the i6 inline build. Returns None when the reduction would be
/// trivial or excessive.
fn build_reduced_from_survivors(
    challenge: &Challenge,
    locked_out: &[bool],
) -> Option<(Challenge, Vec<usize>, Vec<i64>)> {
    let n = challenge.num_items;
    let survivors: Vec<usize> = (0..n).filter(|&i| !locked_out[i]).collect();
    let m = survivors.len();
    // Need a meaningful reduction, and never over-shrink the search space.
    if m < 200 || m >= n || (n - m) < 100 { return None; }

    let mut red_weights = vec![0u32; m];
    let mut red_values = vec![0u32; m];
    let mut red_iv: Vec<Vec<i32>> = vec![vec![0i32; m]; m];
    for a in 0..m {
        let ga = survivors[a];
        red_weights[a] = challenge.weights[ga];
        red_values[a] = challenge.values[ga];
        let row = &challenge.interaction_values[ga];
        for b in (a + 1)..m {
            let iv = row[survivors[b]];
            red_iv[a][b] = iv;
            red_iv[b][a] = iv;
        }
    }
    let reduced = Challenge {
        seed: challenge.seed,
        num_items: m,
        weights: red_weights,
        values: red_values,
        interaction_values: red_iv,
        max_weight: challenge.max_weight,
    };
    let mut red_ti = vec![0i64; m];
    for a in 0..m {
        let mut s = 0i64;
        for b in 0..m { s += reduced.interaction_values[a][b] as i64; }
        red_ti[a] = s;
    }
    Some((reduced, survivors, red_ti))
}

/// Rich small-instance pipeline: multi-restart elite pool + frequency crossover +
/// path-relinking + deep LNS + consensus core-DP. Native regime is n<=1200; it is
/// also invoked on the reduced sub-Challenge produced by `fixation_intensify` so a
/// large-n (n=5000) instance can be intensified after variable fixation.
fn solve_rich_small(challenge: &Challenge, hp: Hparams, total_interactions: Vec<i64>) -> Solution {
    let n = challenge.num_items;
    let n_restarts = hp.n_full_restarts.max(1);
    let mut rng = Rng::from_seed(&challenge.seed);
    for _ in 0..9999 { rng.next_u32(); }

    let ch_dp = hp.core_half_dp;
    let mut elite: Vec<(Solution, i64)> = Vec::new();

        let n_phase1 = n_restarts.min(6).max(2);
        let mut restart_stall_1k = 0usize;
        for restart in 0..n_phase1 {
            let (sol, val) = run_one_instance(challenge, &hp, restart, None, &total_interactions);
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
            let mut state = State::new_empty(challenge);
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
            for round in 0..20 {
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
                let sol_ab = path_relink(challenge, &elite[i].0, &elite[j].0, &hp);
                let val_ab = eval_solution(challenge, &sol_ab);
                elite.push((sol_ab, val_ab));
                let sol_ba = path_relink(challenge, &elite[j].0, &elite[i].0, &hp);
                let val_ba = eval_solution(challenge, &sol_ba);
                elite.push((sol_ba, val_ba));
            }
        }
        elite.sort_by_key(|&(_, v)| std::cmp::Reverse(v));

        {
            let (sol, val) = run_one_instance_seeded(challenge, &hp, 99, None, &total_interactions, Some(&elite[0].0));
            elite.push((sol, val));
            elite.sort_by_key(|&(_, v)| std::cmp::Reverse(v));
            elite.truncate(8);
        }

        let deep_dp = 150;
        let mut best_val = i64::MIN;
        let mut best_sel = Vec::new();

        for idx in 0..elite.len().min(2) {
            let mut state = State::new_empty(challenge);
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

        let mut state = State::new_empty(challenge);
        for &i in &best_sel { state.add_item(i); }

        for lns_round in 0..6 {
            let snap = state.clone_solution();

            let sel = state.selected_items();
            let pct = 20 + (lns_round % 4) * 8;
            let n_remove = sel.len() * pct / 100;
            let mut candidates: Vec<(usize, i64)> = sel.iter().map(|&i| {
                let score = match lns_round % 10 {
                    0 => state.contrib[i] as i64,
                    1 => -(state.ch.weights[i] as i64),
                    2 => state.contrib[i] as i64 - state.ch.values[i] as i64,
                    3 => { let w = (state.ch.weights[i] as i64).max(1); (state.contrib[i] as i64 * 1000) / w },
                    4 => { let w = (state.ch.weights[i] as i64).max(1); (state.contrib[i] as i64 * 10000) / (w * w) },
                    5 => -(state.contrib[i] as i64),
                    6 => rng.next_u32() as i64,
                    7 => {
                        let anti = 2 * state.contrib[i] as i64 - total_interactions[i];
                        let w = (state.ch.weights[i] as i64).max(1);
                        (anti * 1000) / w
                    },
                    8 => {
                        state.contrib[i] as i64 * 100 - total_interactions[i]
                    },
                    _ => {
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

            match lns_round % 4 {
                0 => greedy_reconstruct(&mut state, 0, &total_interactions),
                1 => greedy_reconstruct(&mut state, 3, &total_interactions),
                2 => {
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
            let mut rst = State::new_empty(challenge);
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
        let never_in: Vec<usize> = (0..n).filter(|&i| item_freq[i] == 0).collect();
        let disputed: Vec<usize> = (0..n).filter(|&i| item_freq[i] > 0 && item_freq[i] < tk).collect();

        if disputed.len() > 0 && disputed.len() <= 200 {
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
                        best_val = state.total_value;
                        best_sel = state.selected_items();
                    }
                }
            }
        }

    Solution { items: best_sel }
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
    println!("the base solver");
}

#[inline(always)]
pub fn solve(
    challenge: &Challenge,
    save: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hp: &Option<serde_json::Map<String, serde_json::Value>>,
) -> anyhow::Result<()> {
    solve_challenge(challenge, save, hp)
}
