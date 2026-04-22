use anyhow::Result;
use serde_json::{Map, Value};
use std::cmp::Reverse;
use tig_challenges::knapsack::*;

struct Hparams {
    window_k: usize,
    core_dp: usize,
    ils_rounds: usize,
    perturb_base: usize,
    n_starts: usize,
}

impl Hparams {
    fn for_size(n: usize, budget: u32) -> Self {
        if n <= 1200 {
            if budget <= 5 {
                Self { window_k: 200, core_dp: 50, ils_rounds: 400, perturb_base: 4, n_starts: 6 }
            } else if budget <= 10 {
                Self { window_k: 200, core_dp: 50, ils_rounds: 400, perturb_base: 4, n_starts: 6 }
            } else {
                Self { window_k: 200, core_dp: 50, ils_rounds: 300, perturb_base: 5, n_starts: 4 }
            }
        } else {
            Self { window_k: 100, core_dp: 25, ils_rounds: 100, perturb_base: 8, n_starts: 3 }
        }
    }
    fn from_map(h: &Option<Map<String, Value>>, n: usize, b: u32) -> Self {
        let mut p = Self::for_size(n, b);
        if let Some(m) = h {
            if let Some(v) = m.get("window_k").and_then(|v| v.as_u64()) { p.window_k = v as usize; }
            if let Some(v) = m.get("core_dp").and_then(|v| v.as_u64()) { p.core_dp = v as usize; }
            if let Some(v) = m.get("ils_rounds").and_then(|v| v.as_u64()) { p.ils_rounds = v as usize; }
            if let Some(v) = m.get("perturb_base").and_then(|v| v.as_u64()) { p.perturb_base = v as usize; }
            if let Some(v) = m.get("n_starts").and_then(|v| v.as_u64()) { p.n_starts = v as usize; }
        }
        p
    }
}

#[derive(Clone, Copy)]
struct Rng { state: u64 }
impl Rng {
    fn from_seed(seed: &[u8; 32]) -> Self {
        let mut s: u64 = 0x9E3779B97F4A7C15;
        for (i, &b) in seed.iter().enumerate() { s ^= (b as u64) << ((i & 7) * 8); s = s.rotate_left(7).wrapping_mul(0xBF58476D1CE4E5B9); }
        if s == 0 { s = 1; } Self { state: s }
    }
    #[inline(always)] fn next_u64(&mut self) -> u64 { let mut x = self.state; x ^= x << 7; x ^= x >> 9; x ^= x << 8; self.state = x; x }
    #[inline(always)] fn next_u32(&mut self) -> u32 { (self.next_u64() >> 32) as u32 }
    #[inline(always)] fn next_usize(&mut self, bound: usize) -> usize { if bound == 0 { return 0; } (self.next_u64() % bound as u64) as usize }
}

// -------------------------------------------------------------------------
// 1. FAST SPARSE ENGINE O(|E|)
// -------------------------------------------------------------------------
struct SparseAdj {
    offsets: Vec<usize>,
    neighbors: Vec<u16>,
    weights: Vec<i16>,
    total_inter: Vec<i32>,
}
impl SparseAdj {
    fn new(ch: &Challenge) -> Self {
        let n = ch.num_items;
        let mut offsets = Vec::with_capacity(n + 1);
        let mut neighbors = Vec::with_capacity(n * 100);
        let mut weights = Vec::with_capacity(n * 100);
        let mut total_inter = vec![0i32; n];
        let mut current_offset = 0;
        for i in 0..n {
            offsets.push(current_offset);
            let row = &ch.interaction_values[i];
            let mut syn = 0;
            for j in 0..n {
                if i != j {
                    let w = row[j];
                    if w > 0 {
                        neighbors.push(j as u16);
                        weights.push(w as i16);
                        syn += w as i32;
                        current_offset += 1;
                    }
                }
            }
            total_inter[i] = syn;
        }
        offsets.push(current_offset);
        Self { offsets, neighbors, weights, total_inter }
    }
}

// -------------------------------------------------------------------------
// 2. ZERO-ALLOCATION VECTORIZED STATE MANAGER
// -------------------------------------------------------------------------
struct State<'a> {
    ch: &'a Challenge,
    selected_bit: Vec<bool>,
    contrib: Vec<i32>,
    total_value: i64,
    total_weight: u32,
    dp_cache: Vec<i64>,
    choose_cache: Vec<u8>,
    adj: &'a SparseAdj,
    by_density: Vec<usize>,
    to_add: Vec<usize>,
    to_rm: Vec<usize>,
    u_r: Vec<(usize, f64)>,
    s_r: Vec<(usize, f64)>,
    best_u: Vec<usize>,
    worst_s: Vec<usize>,
}

impl<'a> State<'a> {
    fn new_empty(ch: &'a Challenge, adj: &'a SparseAdj) -> Self {
        let n = ch.num_items;
        let mut contrib = vec![0i32; n];
        for i in 0..n { contrib[i] = ch.values[i] as i32; }
        Self {
            ch, selected_bit: vec![false; n], contrib,
            total_value: 0, total_weight: 0,
            dp_cache: Vec::new(), choose_cache: Vec::new(), adj,
            by_density: (0..n).collect(),
            to_add: Vec::with_capacity(n),
            to_rm: Vec::with_capacity(n),
            u_r: Vec::with_capacity(n),
            s_r: Vec::with_capacity(n),
            best_u: Vec::with_capacity(n),
            worst_s: Vec::with_capacity(n),
        }
    }

    #[inline(always)] fn slack(&self) -> u32 { self.ch.max_weight - self.total_weight }

    #[inline(always)]
    fn add_item(&mut self, i: usize) {
        self.total_value += self.contrib[i] as i64;
        self.total_weight += self.ch.weights[i];
        let start = self.adj.offsets[i]; let end = self.adj.offsets[i + 1];
        let c_ptr = self.contrib.as_mut_ptr();
        unsafe {
            for idx in start..end {
                let nb = *self.adj.neighbors.get_unchecked(idx) as usize;
                let w = *self.adj.weights.get_unchecked(idx) as i32;
                let ptr = c_ptr.add(nb); *ptr = (*ptr).wrapping_add(w);
            }
        }
        self.selected_bit[i] = true;
    }

    #[inline(always)]
    fn remove_item(&mut self, j: usize) {
        self.total_value -= self.contrib[j] as i64;
        self.total_weight -= self.ch.weights[j];
        let start = self.adj.offsets[j]; let end = self.adj.offsets[j + 1];
        let c_ptr = self.contrib.as_mut_ptr();
        unsafe {
            for idx in start..end {
                let nb = *self.adj.neighbors.get_unchecked(idx) as usize;
                let w = *self.adj.weights.get_unchecked(idx) as i32;
                let ptr = c_ptr.add(nb); *ptr = (*ptr).wrapping_sub(w);
            }
        }
        self.selected_bit[j] = false;
    }

    #[inline(always)] fn replace_item(&mut self, rm: usize, cand: usize) { self.remove_item(rm); self.add_item(cand); }

    fn selected_items(&self) -> Vec<usize> {
        (0..self.ch.num_items).filter(|&i| self.selected_bit[i]).collect()
    }

    fn clone_solution(&self) -> SolState {
        let mut items = Vec::with_capacity(self.ch.num_items / 2);
        for i in 0..self.ch.num_items { if self.selected_bit[i] { items.push(i); } }
        SolState { bits: self.selected_bit.clone(), value: self.total_value, items }
    }

    fn restore_solution(&mut self, sol: &SolState) {
        self.to_rm.clear();
        for i in 0..self.ch.num_items { if self.selected_bit[i] && !sol.bits[i] { self.to_rm.push(i); } }
        let mut removes = vec![]; core::mem::swap(&mut removes, &mut self.to_rm);
        for &i in &removes { self.remove_item(i); }
        core::mem::swap(&mut removes, &mut self.to_rm);
        for &i in &sol.items { if !self.selected_bit[i] { self.add_item(i); } }
    }
}
#[derive(Clone)] struct SolState { bits: Vec<bool>, value: i64, items: Vec<usize> }

// -------------------------------------------------------------------------
// 3. ZERO-ALLOCATION MATHEMATICAL PRUNING NEIGHBORHOODS
// -------------------------------------------------------------------------
fn build_windows(state: &mut State, k: usize) {
    let n = state.ch.num_items;
    state.u_r.clear(); state.s_r.clear();
    for i in 0..n {
        let w = (state.ch.weights[i] as f64).max(1.0);
        let r = state.contrib[i] as f64 * 1000.0 / w;
        if state.selected_bit[i] { state.s_r.push((i, r)); } else { state.u_r.push((i, r)); }
    }
    let ku = k.min(state.u_r.len());
    if ku > 0 && ku < state.u_r.len() { state.u_r.select_nth_unstable_by(ku - 1, |a, b| b.1.partial_cmp(&a.1).unwrap()); }
    let ks = k.min(state.s_r.len());
    if ks > 0 && ks < state.s_r.len() { state.s_r.select_nth_unstable_by(ks - 1, |a, b| a.1.partial_cmp(&b.1).unwrap()); }

    state.best_u.clear(); state.worst_s.clear();
    for i in 0..ku { state.best_u.push(state.u_r[i].0); }
    for i in 0..ks { state.worst_s.push(state.s_r[i].0); }

    let contrib = &state.contrib;
    state.best_u.sort_unstable_by_key(|&i| Reverse(contrib[i]));
    state.worst_s.sort_unstable_by_key(|&i| contrib[i]);
}

fn apply_best_add(state: &mut State, unsel: &[usize]) -> bool {
    let slack = state.slack(); if slack == 0 { return false; }
    let (mut best_i, mut best_d) = (None, 0);
    for &cand in unsel {
        if state.contrib[cand] <= best_d { break; }
        if state.ch.weights[cand] <= slack { best_d = state.contrib[cand]; best_i = Some(cand); }
    }
    if let Some(i) = best_i { state.add_item(i); true } else { false }
}

fn apply_best_swap_1_1(state: &mut State, sel: &[usize], unsel: &[usize]) -> bool {
    let slack = state.slack() as i32; let mut best_delta = 0; let mut best_move = None;
    let max_ca = if !unsel.is_empty() { state.contrib[unsel[0]] } else { 0 };
    let ks = sel.len().min(80);
    let ku = unsel.len().min(120);
    for idx_rm in 0..ks {
        let rm = sel[idx_rm];
        let c_rm = state.contrib[rm]; let max_w = (state.ch.weights[rm] as i32) + slack;
        if max_ca - c_rm <= best_delta { break; }
        for idx_add in 0..ku {
            let add = unsel[idx_add];
            let c_add = state.contrib[add]; if c_add - c_rm <= best_delta { break; }
            if (state.ch.weights[add] as i32) <= max_w {
                let delta = c_add - c_rm - state.ch.interaction_values[add][rm];
                if delta > best_delta { best_delta = delta; best_move = Some((add, rm)); }
            }
        }
    }
    if let Some((add, rm)) = best_move { state.replace_item(rm, add); true } else { false }
}

fn apply_pair_add(state: &mut State, unsel: &[usize]) -> bool {
    let slack = state.slack() as i32; if slack < 2 { return false; }
    let ku = unsel.len().min(60);
    if ku < 2 { return false; }
    let (mut best_delta, mut best_pair) = (0i64, None);
    for ai in 0..ku {
        let a = unsel[ai]; let ca = state.contrib[a] as i64; let wa = state.ch.weights[a] as i32;
        let max_cb = if ai + 1 < ku { state.contrib[unsel[ai+1]] as i64 } else { 0 };
        if ca + max_cb + 1000 <= best_delta { break; }
        for bi in (ai+1)..ku {
            let b = unsel[bi]; let cb = state.contrib[b] as i64;
            if ca + cb + 1000 <= best_delta { break; }
            if wa + state.ch.weights[b] as i32 <= slack {
                let delta = ca + cb + state.ch.interaction_values[a][b] as i64;
                if delta > best_delta { best_delta = delta; best_pair = Some((a, b)); }
            }
        }
    }
    if let Some((a, b)) = best_pair { state.add_item(a); state.add_item(b); true } else { false }
}

fn apply_chain_move(state: &mut State, sel: &[usize], unsel: &[usize]) -> bool {
    let slack = state.slack() as i32; let mut best_delta = 0i64; let mut best_move = None;
    let cap = state.ch.max_weight;
    let ks = sel.len().min(20);
    let ku = unsel.len().min(40);
    if ks == 0 || ku < 2 { return false; }
    let max_ca1 = state.contrib[unsel[0]] as i64;
    let max_ca2 = state.contrib[unsel[1]] as i64;
    for idx_rm in 0..ks {
        let rm = sel[idx_rm];
        let c_rm = state.contrib[rm] as i64; let w_rm = state.ch.weights[rm] as i32;
        if max_ca1 + max_ca2 + 1000 - c_rm <= best_delta { continue; }
        let budget = slack + w_rm;
        for i in 0..ku {
            let a1 = unsel[i]; let c_a1 = state.contrib[a1] as i64; let w_a1 = state.ch.weights[a1] as i32;
            if w_a1 >= budget { continue; }
            let max_ca2_local = unsel.get(i+1).map_or(0, |&x| state.contrib[x] as i64);
            if c_a1 + max_ca2_local + 1000 - c_rm <= best_delta { break; }
            let ca1_eff = c_a1 - state.ch.interaction_values[a1][rm] as i64;
            for j in (i+1)..ku {
                let a2 = unsel[j]; let c_a2 = state.contrib[a2] as i64;
                if c_a1 + c_a2 + 1000 - c_rm <= best_delta { break; }
                let w_a2 = state.ch.weights[a2] as i32; if w_a1 + w_a2 > budget { continue; }
                let delta = ca1_eff + c_a2 - state.ch.interaction_values[a2][rm] as i64 + state.ch.interaction_values[a1][a2] as i64 - c_rm;
                if delta > best_delta {
                    if state.total_weight as i32 - w_rm + w_a1 + w_a2 <= cap as i32 {
                        best_delta = delta; best_move = Some((rm, a1, a2));
                    }
                }
            }
        }
    }
    if let Some((rm, a1, a2)) = best_move { state.remove_item(rm); state.add_item(a1); state.add_item(a2); true } else { false }
}

fn apply_reverse_chain(state: &mut State, sel: &[usize], unsel: &[usize]) -> bool {
    let ks = sel.len().min(40);
    let ku = unsel.len().min(20);
    if ks < 2 || ku == 0 { return false; }
    let slack = state.slack() as i32; let mut best_delta = 0i64; let mut best_move = None;
    let cap = state.ch.max_weight;
    let min_cr1 = state.contrib[sel[0]] as i64;
    let min_cr2 = state.contrib[sel[1]] as i64;
    for idx_add in 0..ku {
        let add = unsel[idx_add];
        let c_add = state.contrib[add] as i64; let w_add = state.ch.weights[add] as i32;
        if c_add + 1000 - min_cr1 - min_cr2 <= best_delta { continue; }
        for i in 0..ks {
            let r1 = sel[i]; let c_r1 = state.contrib[r1] as i64; let w_r1 = state.ch.weights[r1] as i32;
            let min_cr2_local = sel.get(i+1).map_or(0, |&x| state.contrib[x] as i64);
            if c_add + 1000 - c_r1 - min_cr2_local <= best_delta { break; }
            let c_add_r1 = state.ch.interaction_values[add][r1] as i64;
            for j in (i+1)..ks {
                let r2 = sel[j]; let c_r2 = state.contrib[r2] as i64; let w_r2 = state.ch.weights[r2] as i32;
                if c_add + 1000 - c_r1 - c_r2 <= best_delta { break; }
                let freed = w_r1 + w_r2;
                if slack + freed >= w_add {
                    let delta = c_add - c_add_r1 - state.ch.interaction_values[add][r2] as i64 - c_r1 - c_r2 + state.ch.interaction_values[r1][r2] as i64;
                    if delta > best_delta && state.total_weight as i32 - freed + w_add <= cap as i32 {
                        best_delta = delta; best_move = Some((r1, r2, add));
                    }
                }
            }
        }
    }
    if let Some((r1, r2, add)) = best_move { state.remove_item(r1); state.remove_item(r2); state.add_item(add); true } else { false }
}

fn apply_swap_2_2_bounded(state: &mut State, b_u: &[usize], w_s: &[usize], limit: usize) -> bool {
    let ku = b_u.len().min(limit); let ks = w_s.len().min(limit);
    if ks < 2 || ku < 2 { return false; }
    let cap = state.ch.max_weight; let mut best_delta = 0i64; let mut best_move = None;
    let max_ca1 = state.contrib[b_u[0]] as i64; let max_ca2 = if ku > 1 { state.contrib[b_u[1]] as i64 } else { 0 };
    for i in 0..ks {
        let r1 = w_s[i]; let c_r1 = state.contrib[r1] as i64; let w_r1 = state.ch.weights[r1] as i32;
        for j in (i+1)..ks {
            let r2 = w_s[j]; let c_r2 = state.contrib[r2] as i64; let w_r2 = state.ch.weights[r2] as i32;
            let lost = c_r1 + c_r2 - state.ch.interaction_values[r1][r2] as i64;
            if max_ca1 + max_ca2 + 1000 - lost <= best_delta { continue; }
            let freed = w_r1 + w_r2; let budget = state.slack() as i32 + freed;
            for u in 0..ku {
                let a1 = b_u[u]; let c_a1 = state.contrib[a1] as i64; let w_a1 = state.ch.weights[a1] as i32;
                if w_a1 > budget { continue; }
                let max_ca2_local = b_u.get(u+1).map_or(0, |&x| state.contrib[x] as i64);
                if c_a1 + max_ca2_local + 1000 - lost <= best_delta { break; }
                let c_a1_eff = c_a1 - state.ch.interaction_values[a1][r1] as i64 - state.ch.interaction_values[a1][r2] as i64;
                for v in (u+1)..ku {
                    let a2 = b_u[v]; let c_a2 = state.contrib[a2] as i64;
                    if c_a1 + c_a2 + 1000 - lost <= best_delta { break; }
                    let w_a2 = state.ch.weights[a2] as i32; if w_a1 + w_a2 > budget { continue; }
                    let delta = c_a1_eff + c_a2 - state.ch.interaction_values[a2][r1] as i64 - state.ch.interaction_values[a2][r2] as i64 + state.ch.interaction_values[a1][a2] as i64 - lost;
                    if delta > best_delta && state.total_weight as i32 - freed + w_a1 + w_a2 <= cap as i32 {
                        best_delta = delta; best_move = Some((r1, r2, a1, a2));
                    }
                }
            }
        }
    }
    if let Some((r1, r2, a1, a2)) = best_move { state.remove_item(r1); state.remove_item(r2); state.add_item(a1); state.add_item(a2); true } else { false }
}

fn local_search_vnd_windowed(state: &mut State, window_k: usize) {
    let max_iters = if state.ch.num_items > 1200 { 60 } else { 100 };
    for _ in 0..max_iters {
        build_windows(state, window_k);
        let mut b_u = vec![]; let mut w_s = vec![];
        core::mem::swap(&mut b_u, &mut state.best_u);
        core::mem::swap(&mut w_s, &mut state.worst_s);
        let mut improved = false;
        if apply_best_add(state, &b_u) { improved = true; }
        else if apply_best_swap_1_1(state, &w_s, &b_u) { improved = true; }
        else if apply_pair_add(state, &b_u) { improved = true; }
        else if apply_chain_move(state, &w_s, &b_u) { improved = true; }
        else if apply_reverse_chain(state, &w_s, &b_u) { improved = true; }
        else if apply_swap_2_2_bounded(state, &b_u, &w_s, 20) { improved = true; }
        core::mem::swap(&mut b_u, &mut state.best_u);
        core::mem::swap(&mut w_s, &mut state.worst_s);
        if !improved { break; }
    }
}

// -------------------------------------------------------------------------
// 4. ZERO-ALLOCATION DP REFINEMENT
// -------------------------------------------------------------------------
fn dp_refinement_hp(state: &mut State, core_half: usize) {
    let n = state.ch.num_items; let cap = state.ch.max_weight;
    let contrib = &state.contrib;
    let weights = &state.ch.weights;
    state.by_density.sort_unstable_by(|&a, &b| (contrib[a] as i64 * weights[b] as i64).cmp(&(contrib[b] as i64 * weights[a] as i64)).reverse());
    let (mut idx_l, mut idx_f, mut rem) = (0, n, cap);
    for (idx, &i) in state.by_density.iter().enumerate() { let w = state.ch.weights[i]; if w <= rem { rem -= w; idx_l = idx; } else if idx_f == n { idx_f = idx; } }
    let left = idx_f.saturating_sub(core_half + 1); let right = (idx_l + core_half + 1).min(n);
    let locked_weight: u32 = state.by_density[..left].iter().map(|&i| state.ch.weights[i]).sum();
    let rem_cap = cap.saturating_sub(locked_weight) as usize; let myk = right - left;
    if myk == 0 || rem_cap == 0 { return; }
    let (mut tc_w, mut tp_w, mut all_pos) = (0, 0, true);
    for &it in &state.by_density[left..right] { let wt = state.ch.weights[it] as usize; tc_w += wt; if state.contrib[it] > 0 { tp_w += wt; if tp_w > rem_cap { all_pos = false; } } }
    state.to_add.clear(); state.to_rm.clear();
    if all_pos {
        for &it in &state.by_density[left..right] {
            if state.contrib[it] > 0 { if !state.selected_bit[it] { state.to_add.push(it); } }
            else { if state.selected_bit[it] { state.to_rm.push(it); } }
        }
    } else {
        let myw = rem_cap.min(tc_w); let dp_size = myw + 1; let choose_size = myk * dp_size;
        if dp_size * myk > 300_000 { return; }
        if state.dp_cache.len() < dp_size { state.dp_cache.resize(dp_size, i64::MIN / 4); }
        if state.choose_cache.len() < choose_size { state.choose_cache.resize(choose_size, 0); }
        state.dp_cache[..dp_size].fill(i64::MIN / 4); state.dp_cache[0] = 0; state.choose_cache[..choose_size].fill(0);
        let mut w_hi = 0;
        for t in 0..myk {
            let it = state.by_density[left + t];
            let wt = state.ch.weights[it] as usize; if wt > myw { continue; }
            let val = state.contrib[it] as i64; let new_hi = (w_hi + wt).min(myw);
            for w in (wt..=new_hi).rev() {
                let prev = state.dp_cache[w - wt]; if prev < i64::MIN / 8 { continue; }
                let cand = prev + val;
                if cand > state.dp_cache[w] { state.dp_cache[w] = cand; state.choose_cache[t * dp_size + w] = 1; }
            }
            w_hi = new_hi;
        }
        let mut w_s = (0..=myw).max_by_key(|&w| state.dp_cache[w]).unwrap_or(0);
        let mut is_in_dp = vec![false; myk];
        for t in (0..myk).rev() {
            let it = state.by_density[left + t]; let wt = state.ch.weights[it] as usize;
            if wt <= w_s && state.choose_cache[t * dp_size + w_s] == 1 { is_in_dp[t] = true; w_s -= wt; }
        }
        for t in 0..myk {
            let it = state.by_density[left + t];
            if is_in_dp[t] { if !state.selected_bit[it] { state.to_add.push(it); } }
            else { if state.selected_bit[it] { state.to_rm.push(it); } }
        }
    }
    for &it in &state.by_density[..left] { if !state.selected_bit[it] { state.to_add.push(it); } }
    for &it in &state.by_density[right..] { if state.selected_bit[it] { state.to_rm.push(it); } }
    let mut rms = vec![]; let mut adds = vec![];
    core::mem::swap(&mut rms, &mut state.to_rm);
    core::mem::swap(&mut adds, &mut state.to_add);
    for r in &rms { state.remove_item(*r); }
    for a in &adds { state.add_item(*a); }
    core::mem::swap(&mut rms, &mut state.to_rm);
    core::mem::swap(&mut adds, &mut state.to_add);
}

// -------------------------------------------------------------------------
// 5. ILS PERTURBATION
// -------------------------------------------------------------------------
fn cluster_bomb(state: &mut State, rng: &mut Rng, strength: usize) {
    state.to_rm.clear();
    for i in 0..state.ch.num_items { if state.selected_bit[i] { state.to_rm.push(i); } }
    if state.to_rm.is_empty() { return; }
    let target = state.total_weight / (strength as u32).max(2); let mut freed = 0;
    let root = state.to_rm[rng.next_usize(state.to_rm.len())];
    state.remove_item(root); freed += state.ch.weights[root];
    let start = state.adj.offsets[root]; let end = state.adj.offsets[root + 1];
    unsafe {
        for idx in start..end {
            let nb = *state.adj.neighbors.get_unchecked(idx) as usize;
            if state.selected_bit[nb] {
                state.remove_item(nb); freed += state.ch.weights[nb];
                if freed >= target { break; }
            }
        }
    }
}

fn perturb_by_strategy(state: &mut State, strength: usize, stall: usize, stgy: usize, rng: &mut Rng, perturb_base: usize, total_inter: &[i32]) {
    state.to_rm.clear();
    for i in 0..state.ch.num_items { if state.selected_bit[i] { state.to_rm.push(i); } }
    if state.to_rm.is_empty() { return; }
    let mut cands = vec![];
    core::mem::swap(&mut cands, &mut state.to_rm);
    let contrib = &state.contrib;
    let weights = &state.ch.weights;
    let values = &state.ch.values;
    if stgy % 8 == 6 {
        for i in (1..cands.len()).rev() { let j = rng.next_usize(i + 1); cands.swap(i, j); }
    } else {
        cands.sort_unstable_by_key(|&i| {
            match stgy % 8 {
                0 => contrib[i],
                1 => -(weights[i] as i32),
                2 => contrib[i] - values[i] as i32,
                3 => (contrib[i] * 1000) / (weights[i] as i32).max(1),
                4 => weights[i] as i32 - ((contrib[i] * 100) / (weights[i] as i32).max(1)),
                5 => (contrib[i] * 10000) / ((weights[i] as i32).pow(2)).max(1),
                _ => -(contrib[i] + total_inter[i] / 10),
            }
        });
    }
    let n_rm = ((cands.len() / perturb_base).max(2) * (1 + stall / 2)).min(strength).min(cands.len() * 2 / 5);
    for j in 0..n_rm { if j < cands.len() { state.remove_item(cands[j]); } }
    core::mem::swap(&mut cands, &mut state.to_rm);
}

fn greedy_reconstruct(state: &mut State, strategy: usize, total_inter: &[i32]) {
    let cap = state.ch.max_weight;
    state.to_add.clear();
    for i in 0..state.ch.num_items { if !state.selected_bit[i] { state.to_add.push(i); } }
    let mut cands = vec![];
    core::mem::swap(&mut cands, &mut state.to_add);
    let contrib = &state.contrib;
    let weights = &state.ch.weights;
    match strategy % 4 {
        0 => cands.sort_unstable_by_key(|&i| Reverse(contrib[i])),
        1 => cands.sort_unstable_by(|&a, &b| weights[a].cmp(&weights[b]).then(contrib[b].cmp(&contrib[a]))),
        2 => cands.sort_unstable_by_key(|&i| Reverse(total_inter[i] + contrib[i] * 5)),
        _ => cands.sort_unstable_by_key(|&i| Reverse((contrib[i] * 100) / (weights[i] as i32).max(1))),
    }
    for j in 0..cands.len() {
        let i = cands[j];
        if state.total_weight + weights[i] <= cap { state.add_item(i); }
    }
    core::mem::swap(&mut cands, &mut state.to_add);
}

// -------------------------------------------------------------------------
// 6. MAIN SOLVER
// -------------------------------------------------------------------------
pub fn solve(
    ch: &Challenge,
    save: &dyn Fn(&Solution) -> Result<()>,
    hp_map: &Option<Map<String, Value>>,
) -> Result<()> {
    let n = ch.num_items;
    let sum_w: u64 = ch.weights.iter().map(|&w| w as u64).sum();
    let budget_pct = if sum_w > 0 { ((ch.max_weight as u64) * 100 / sum_w) as u32 } else { 10 };
    let hp = Hparams::from_map(hp_map, n, budget_pct);
    let adj = SparseAdj::new(ch);
    let mut rng = Rng::from_seed(&ch.seed);
    let mut best_val = i64::MIN;
    let mut best_sol: Option<Solution> = None;

    // --- PHASE 1: DIVERSE STARTS ---
    let mut pop = Vec::with_capacity(16);
    for mode in 0..hp.n_starts {
        let mut st = State::new_empty(ch, &adj);
        let actual_mode = if mode == 0 && (n > 1200 || budget_pct <= 5) { 1 } else { mode % 4 };
        match actual_mode {
            0 => {
                for i in 0..n { st.add_item(i); }
                while st.total_weight > ch.max_weight {
                    let (mut worst, mut worst_s) = (0, i32::MAX);
                    for i in 0..n { if st.selected_bit[i] { let s = (st.contrib[i] * 1000) / (ch.weights[i] as i32).max(1); if s < worst_s { worst_s = s; worst = i; } } }
                    st.remove_item(worst);
                }
            }
            1 => {
                let mut cands: Vec<usize> = (0..n).collect(); cands.sort_unstable_by_key(|&i| Reverse(adj.total_inter[i]));
                for &i in &cands { if st.total_weight + ch.weights[i] <= ch.max_weight { st.add_item(i); } }
            }
            2 => {
                loop {
                    let slack = st.slack(); if slack == 0 { break; }
                    let (mut bi, mut bs) = (None, i32::MIN);
                    for i in 0..n {
                        if st.selected_bit[i] || ch.weights[i] > slack { continue; }
                        let s = (st.contrib[i] * 1000) / (ch.weights[i] as i32).max(1) + (rng.next_u32() & 0x3F) as i32;
                        if s > bs { bs = s; bi = Some(i); }
                    }
                    if let Some(i) = bi { st.add_item(i); } else { break; }
                }
            }
            _ => {
                let mut bp = None; let mut bs = 0;
                for i in 0..n {
                    unsafe {
                        for idx in adj.offsets[i]..adj.offsets[i+1] {
                            let j = *adj.neighbors.as_ptr().add(idx) as usize;
                            if i < j && ch.weights[i] + ch.weights[j] <= ch.max_weight {
                                let w = *adj.weights.as_ptr().add(idx) as i32; if w > bs { bs = w; bp = Some((i, j)); }
                            }
                        }
                    }
                }
                if let Some((i, j)) = bp { st.add_item(i); st.add_item(j); }
                loop {
                    let slack = st.slack(); if slack == 0 { break; }
                    let (mut bi, mut bs) = (None, i32::MIN);
                    for i in 0..n {
                        if st.selected_bit[i] || ch.weights[i] > slack { continue; }
                        let s = (st.contrib[i] * 1000) / (ch.weights[i] as i32).max(1);
                        if s > bs { bs = s; bi = Some(i); }
                    }
                    if let Some(i) = bi { st.add_item(i); } else { break; }
                }
            }
        }
        dp_refinement_hp(&mut st, hp.core_dp);
        local_search_vnd_windowed(&mut st, hp.window_k);
        if st.total_value > best_val {
            best_val = st.total_value;
            best_sol = Some(Solution { items: st.selected_items() });
        }
        pop.push(st.clone_solution());
    }

    // Save after phase 1 (anti-timeout guarantee)
    if let Some(ref sol) = best_sol { let _ = save(sol); }

    pop.sort_unstable_by_key(|s| Reverse(s.value)); pop.truncate(8);

    let mut tabu = Vec::with_capacity(128);
    let zob: Vec<u64> = (0..n).map(|i| { let mut h = 0x517CC1B727220A95u64; h ^= (i as u64).wrapping_mul(0x9E3779B97F4A7C15); h.rotate_left(17).wrapping_mul(0xBF58476D1CE4E5B9) }).collect();
    let chash = |b: &[bool]| -> u64 { let mut h: u64 = 0; for i in 0..n { if b[i] { h ^= zob[i]; } } h };

    // --- PHASE 2: ILS LOOP ---
    let mut st = State::new_empty(ch, &adj);
    st.restore_solution(&pop[0]);
    let mut curr_bval = st.total_value;
    let mut curr_bsel = st.selected_items();
    let mut stall = 0usize;
    tabu.push(chash(&st.selected_bit));

    for rnd in 0..hp.ils_rounds {
        let snap = st.clone_solution();
        dp_refinement_hp(&mut st, hp.core_dp);
        local_search_vnd_windowed(&mut st, hp.window_k);
        if st.total_value > curr_bval {
            curr_bval = st.total_value;
            curr_bsel = st.selected_items();
            stall = 0;
            if curr_bval > best_val {
                best_val = curr_bval;
                best_sol = Some(Solution { items: curr_bsel.clone() });
                let _ = save(best_sol.as_ref().unwrap());
            }
        } else if st.total_value > snap.value {
            stall += 1;
        } else {
            stall += 1;
            st.restore_solution(&snap);
            if stall > 0 && stall % 15 == 0 { st.restore_solution(&pop[(stall / 15) % pop.len()]); }
            let strength = hp.perturb_base + rnd / 10;
            let s = rnd % 8;
            if rnd % 3 == 0 { cluster_bomb(&mut st, &mut rng, strength); }
            else { perturb_by_strategy(&mut st, strength, stall, s, &mut rng, hp.perturb_base, &adj.total_inter); }
            greedy_reconstruct(&mut st, s % 4, &adj.total_inter);
            let h = chash(&st.selected_bit);
            if tabu.contains(&h) {
                cluster_bomb(&mut st, &mut rng, 2);
                greedy_reconstruct(&mut st, 0, &adj.total_inter);
            }
            let h2 = chash(&st.selected_bit);
            if tabu.len() < 128 { tabu.push(h2); } else { tabu[rnd % 128] = h2; }
        }
    }

    // --- PHASE 3: FINAL POLISH ---
    for i in (0..n).rev() { if st.selected_bit[i] { st.remove_item(i); } }
    for &i in &curr_bsel { st.add_item(i); }
    loop {
        let bef = st.total_value;
        local_search_vnd_windowed(&mut st, hp.window_k.max(250));
        dp_refinement_hp(&mut st, hp.core_dp + 20);
        if st.total_value <= bef { break; }
    }
    if st.total_value > best_val {
        best_sol = Some(Solution { items: st.selected_items() });
        let _ = save(best_sol.as_ref().unwrap());
    }

    Ok(())
}

