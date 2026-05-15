use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::cmp::Reverse;
use tig_challenges::knapsack::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub window_k: Option<usize>,
    pub core_dp: Option<usize>,
    pub ils_rounds: Option<usize>,
    pub perturb_base: Option<usize>,
    pub n_starts: Option<usize>,
    pub sa_steps: Option<usize>,
    pub sa_moves: Option<usize>,
    pub sa_t0: Option<f64>,
    pub sa_decay: Option<f64>,
}

struct Hparams {
    window_k: usize,
    core_dp: usize,
    ils_rounds: usize,
    perturb_base: usize,
    n_starts: usize,
    sa_steps: usize,
    sa_moves: usize,
    sa_t0: f64,
    sa_decay: f64,
}

impl Hparams {
    fn for_size(n: usize, _budget: u32) -> Self {
        if n <= 1200 {
            Self { window_k: 200, core_dp: 60, ils_rounds: 400, perturb_base: 4, n_starts: 4,
                   sa_steps: 300, sa_moves: 40000, sa_t0: 5000.0, sa_decay: 0.99 }
        } else {
            Self { window_k: 100, core_dp: 50, ils_rounds: 200, perturb_base: 6, n_starts: 3,
                   sa_steps: 300, sa_moves: 50000, sa_t0: 5000.0, sa_decay: 0.99 }
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
            if let Some(v) = m.get("sa_steps").and_then(|v| v.as_u64()) { p.sa_steps = v as usize; }
            if let Some(v) = m.get("sa_moves").and_then(|v| v.as_u64()) { p.sa_moves = v as usize; }
            if let Some(v) = m.get("sa_t0").and_then(|v| v.as_f64()) { p.sa_t0 = v; }
            if let Some(v) = m.get("sa_decay").and_then(|v| v.as_f64()) { p.sa_decay = v; }
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
    add_pairs: Vec<(usize, usize, i64, u32)>,
    rm_pairs: Vec<(usize, usize, i64, u32)>,
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
            add_pairs: Vec::with_capacity(80 * 80),
            rm_pairs: Vec::with_capacity(80 * 80),
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
        let mut removes = vec![]; removes.append(&mut self.to_rm);
        for &i in &removes { self.remove_item(i); }
        self.to_rm.append(&mut removes);
        for &i in &sol.items { if !self.selected_bit[i] { self.add_item(i); } }
    }
}
#[derive(Clone)] struct SolState { bits: Vec<bool>, value: i64, items: Vec<usize> }

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
        let row_rm = &state.ch.interaction_values[rm];
        for idx_add in 0..ku {
            let add = unsel[idx_add];
            let c_add = state.contrib[add]; if c_add - c_rm <= best_delta { break; }
            if (state.ch.weights[add] as i32) <= max_w {
                let delta = c_add - c_rm - row_rm[add];
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
    let slack = state.slack() as u32; let cap = state.ch.max_weight;
    let ks = sel.len().min(40); let ku = unsel.len().min(80);
    if ks == 0 || ku < 2 { return false; }

    state.add_pairs.clear();
    for u in 0..ku {
        let a1 = unsel[u]; let ca1 = state.contrib[a1] as i64; let wa1 = state.ch.weights[a1] as u32;
        for v in (u+1)..ku {
            let a2 = unsel[v];
            let exact_gain = ca1 + state.contrib[a2] as i64 + state.ch.interaction_values[a1][a2] as i64;
            state.add_pairs.push((a1, a2, exact_gain, wa1 + state.ch.weights[a2] as u32));
        }
    }
    state.add_pairs.sort_unstable_by_key(|p| Reverse(p.2));

    let mut best_delta: i64 = 0; let mut best_move = None;

    for idx_rm in 0..ks {
        let rm = sel[idx_rm]; let c_rm = state.contrib[rm] as i64; let w_rm = state.ch.weights[rm] as u32;
        let budget = slack + w_rm;
        if state.add_pairs.is_empty() || state.add_pairs[0].2 - c_rm <= best_delta { continue; }

        let row_rm = &state.ch.interaction_values[rm];
        for p_add in &state.add_pairs {
            if p_add.2 - c_rm <= best_delta { break; }
            if p_add.3 > budget { continue; }

            let delta = p_add.2 - c_rm
                - row_rm[p_add.0] as i64
                - row_rm[p_add.1] as i64;

            if delta > best_delta && state.total_weight + p_add.3 - w_rm <= cap {
                best_delta = delta; best_move = Some((rm, p_add.0, p_add.1));
            }
        }
    }
    if let Some((rm, a1, a2)) = best_move { state.remove_item(rm); state.add_item(a1); state.add_item(a2); true } else { false }
}

fn apply_reverse_chain(state: &mut State, sel: &[usize], unsel: &[usize]) -> bool {
    let ks = sel.len().min(80); let ku = unsel.len().min(40);
    if ks < 2 || ku == 0 { return false; }
    let slack = state.slack() as u32; let cap = state.ch.max_weight;

    state.rm_pairs.clear();
    for i in 0..ks {
        let r1 = sel[i]; let cr1 = state.contrib[r1] as i64; let wr1 = state.ch.weights[r1] as u32;
        for j in (i+1)..ks {
            let r2 = sel[j];
            let exact_loss = cr1 + state.contrib[r2] as i64 - state.ch.interaction_values[r1][r2] as i64;
            state.rm_pairs.push((r1, r2, exact_loss, wr1 + state.ch.weights[r2] as u32));
        }
    }
    state.rm_pairs.sort_unstable_by_key(|p| p.2);

    let mut best_delta: i64 = 0; let mut best_move = None;

    for idx_add in 0..ku {
        let add = unsel[idx_add]; let c_add = state.contrib[add] as i64; let w_add = state.ch.weights[add] as u32;
        if state.rm_pairs.is_empty() || c_add - state.rm_pairs[0].2 <= best_delta { continue; }

        let row_add = &state.ch.interaction_values[add];
        for p_rm in &state.rm_pairs {
            if c_add - p_rm.2 <= best_delta { break; }
            let freed = p_rm.3;
            if slack + freed < w_add { continue; }

            let delta = c_add - p_rm.2
                - row_add[p_rm.0] as i64
                - row_add[p_rm.1] as i64;

            if delta > best_delta && state.total_weight - freed + w_add <= cap {
                best_delta = delta; best_move = Some((p_rm.0, p_rm.1, add));
            }
        }
    }
    if let Some((r1, r2, add)) = best_move { state.remove_item(r1); state.remove_item(r2); state.add_item(add); true } else { false }
}

fn apply_swap_2_2_bounded(state: &mut State, b_u: &[usize], w_s: &[usize], limit: usize) -> bool {
    let ku = b_u.len().min(limit); let ks = w_s.len().min(limit);
    if ks < 2 || ku < 2 { return false; }
    let cap = state.ch.max_weight; let slack = state.slack() as u32;

    state.add_pairs.clear();
    for u in 0..ku {
        let a1 = b_u[u]; let ca1 = state.contrib[a1] as i64; let wa1 = state.ch.weights[a1] as u32;
        for v in (u+1)..ku {
            let a2 = b_u[v];
            let exact_gain = ca1 + state.contrib[a2] as i64 + state.ch.interaction_values[a1][a2] as i64;
            state.add_pairs.push((a1, a2, exact_gain, wa1 + state.ch.weights[a2] as u32));
        }
    }
    state.add_pairs.sort_unstable_by_key(|p| Reverse(p.2));

    state.rm_pairs.clear();
    for i in 0..ks {
        let r1 = w_s[i]; let cr1 = state.contrib[r1] as i64; let wr1 = state.ch.weights[r1] as u32;
        for j in (i+1)..ks {
            let r2 = w_s[j];
            let exact_loss = cr1 + state.contrib[r2] as i64 - state.ch.interaction_values[r1][r2] as i64;
            state.rm_pairs.push((r1, r2, exact_loss, wr1 + state.ch.weights[r2] as u32));
        }
    }
    state.rm_pairs.sort_unstable_by_key(|p| p.2);

    let mut best_delta: i64 = 0; let mut best_move = None;

    for p_rm in &state.rm_pairs {
        let budget = slack + p_rm.3;
        if state.add_pairs.is_empty() || state.add_pairs[0].2 - p_rm.2 <= best_delta { break; }

        let row_r1 = &state.ch.interaction_values[p_rm.0];
        let row_r2 = &state.ch.interaction_values[p_rm.1];
        for p_add in &state.add_pairs {
            if p_add.2 - p_rm.2 <= best_delta { break; }
            if p_add.3 > budget { continue; }

            let delta = p_add.2 - p_rm.2
                - row_r1[p_add.0] as i64
                - row_r1[p_add.1] as i64
                - row_r2[p_add.0] as i64
                - row_r2[p_add.1] as i64;

            if delta > best_delta && state.total_weight - p_rm.3 + p_add.3 <= cap {
                best_delta = delta;
                best_move = Some((p_rm.0, p_rm.1, p_add.0, p_add.1));
            }
        }
    }
    if let Some((r1, r2, a1, a2)) = best_move { state.remove_item(r1); state.remove_item(r2); state.add_item(a1); state.add_item(a2); true } else { false }
}

fn local_search_vnd_windowed(state: &mut State, window_k: usize, stall: usize) {
    let max_iters = if state.ch.num_items > 1200 {
        if stall == 0 { 60 } else { 25 }
    } else {
        if stall == 0 { 100 } else { 40 }
    };
    for _ in 0..max_iters {
        build_windows(state, window_k);
        let mut b_u = vec![]; let mut w_s = vec![];
        b_u.append(&mut state.best_u);
        w_s.append(&mut state.worst_s);
        let mut improved = false;
        if apply_best_add(state, &b_u) { improved = true; }
        else if apply_best_swap_1_1(state, &w_s, &b_u) { improved = true; }
        else if apply_pair_add(state, &b_u) { improved = true; }
        else if apply_chain_move(state, &w_s, &b_u) { improved = true; }
        else if apply_reverse_chain(state, &w_s, &b_u) { improved = true; }
        else if apply_swap_2_2_bounded(state, &b_u, &w_s, 40) { improved = true; }
        state.best_u.append(&mut b_u);
        state.worst_s.append(&mut w_s);
        if !improved { break; }
    }
}

/// Interaction-aware DP Refinement + Core Tabu Search
/// Treats the core as a localized QKP problem and refines it via Tabu Search
fn dp_refinement_hp(state: &mut State, core_half: usize, rng: &mut Rng) {
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
    rms.append(&mut state.to_rm);
    adds.append(&mut state.to_add);
    for r in &rms { state.remove_item(*r); }
    for a in &adds { state.add_item(*a); }
    state.to_rm.append(&mut rms);
    state.to_add.append(&mut adds);

    let mut core_items = Vec::with_capacity(right.saturating_sub(left));
    core_items.extend_from_slice(&state.by_density[left..right]);
    if core_items.is_empty() { return; }
    
    let iters = core_items.len() * 100;
    let mut best_val = state.total_value;
    let mut best_bits = vec![false; core_items.len()];
    for (idx, &it) in core_items.iter().enumerate() { best_bits[idx] = state.selected_bit[it]; }
    let mut tabu = vec![0usize; core_items.len()];
    let mut tabu_stall = 0;

    for step in 1..=iters {
        let slack = state.slack();
        let mut best_delta = i64::MIN;
        let mut best_move = None;
        let mut found_improving = false;

        for (i, &it) in core_items.iter().enumerate() {
            if state.selected_bit[it] {
                let delta = -(state.contrib[it] as i64);
                if delta > best_delta && (step > tabu[i] || state.total_value + delta > best_val) {
                    best_delta = delta; best_move = Some((Some(i), None));
                    if state.total_value + delta > best_val { found_improving = true; break; }
                }
            } else if state.ch.weights[it] <= slack {
                let delta = state.contrib[it] as i64;
                if delta > best_delta && (step > tabu[i] || state.total_value + delta > best_val) {
                    best_delta = delta; best_move = Some((None, Some(i)));
                    if state.total_value + delta > best_val { found_improving = true; break; }
                }
            }
        }

        if !found_improving {
            for (i, &it_rm) in core_items.iter().enumerate() {
                if !state.selected_bit[it_rm] { continue; }
                let w_rm = state.ch.weights[it_rm];
                let c_rm = state.contrib[it_rm] as i64;
                let row_rm = &state.ch.interaction_values[it_rm];
                
                for (j, &it_add) in core_items.iter().enumerate() {
                    if state.selected_bit[it_add] { continue; }
                    let w_add = state.ch.weights[it_add];
                    if w_add <= slack + w_rm {
                        let delta = state.contrib[it_add] as i64 - c_rm - row_rm[it_add] as i64;
                        if delta > best_delta && (step > tabu[i] && step > tabu[j] || state.total_value + delta > best_val) {
                            best_delta = delta; best_move = Some((Some(i), Some(j)));
                            if state.total_value + delta > best_val { found_improving = true; break; }
                        }
                    }
                }
                if found_improving { break; }
            }
        }

        if let Some(mv) = best_move {
            match mv {
                (Some(i), None) => { state.remove_item(core_items[i]); tabu[i] = step + rng.next_usize(4) + 3; }
                (None, Some(j)) => { state.add_item(core_items[j]); tabu[j] = step + rng.next_usize(4) + 3; }
                (Some(i), Some(j)) => { state.replace_item(core_items[i], core_items[j]); tabu[i] = step + rng.next_usize(4) + 4; tabu[j] = step + rng.next_usize(4) + 4; }
                _ => {}
            }
            if state.total_value > best_val {
                best_val = state.total_value;
                for (idx, &it) in core_items.iter().enumerate() { best_bits[idx] = state.selected_bit[it]; }
                tabu_stall = 0;
            } else {
                tabu_stall += 1;
                if tabu_stall > 100 { break; }
            }
        } else { break; }
    }

    state.to_rm.clear(); state.to_add.clear();
    for (idx, &it) in core_items.iter().enumerate() {
        if state.selected_bit[it] && !best_bits[idx] { state.to_rm.push(it); }
        else if !state.selected_bit[it] && best_bits[idx] { state.to_add.push(it); }
    }
    let mut trm = vec![]; trm.append(&mut state.to_rm);
    let mut tadd = vec![]; tadd.append(&mut state.to_add);
    for r in &trm { state.remove_item(*r); }
    for a in &tadd { state.add_item(*a); }
    state.to_rm.append(&mut trm);
    state.to_add.append(&mut tadd);
}

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
    cands.append(&mut state.to_rm);
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
    state.to_rm.append(&mut cands);
}

fn path_relink(state: &mut State, target_items: &[usize], max_steps: usize) -> SolState {
    let n = state.ch.num_items;
    let mut target_in = vec![false; n];
    for &i in target_items { target_in[i] = true; }

    let mut to_add: Vec<usize> = (0..n)
        .filter(|&i| target_in[i] && !state.selected_bit[i])
        .collect();
    let mut to_remove: Vec<usize> = (0..n)
        .filter(|&i| state.selected_bit[i] && !target_in[i])
        .collect();
    to_add.sort_unstable_by_key(|&i| Reverse(state.contrib[i]));
    to_remove.sort_unstable_by_key(|&i| state.contrib[i]);

    let max_w = state.ch.max_weight;
    let mut best = state.clone_solution();
    let mut a = 0usize;
    let mut r = 0usize;
    let mut steps = 0usize;

    while steps < max_steps && (a < to_add.len() || r < to_remove.len()) {
        if a < to_add.len() {
            let cand = to_add[a];
            while state.total_weight + state.ch.weights[cand] > max_w && r < to_remove.len() {
                let rm = to_remove[r];
                if state.selected_bit[rm] { state.remove_item(rm); }
                r += 1;
            }
            if state.total_weight + state.ch.weights[cand] <= max_w {
                state.add_item(cand);
                a += 1;
            } else {
                a += 1;
                continue;
            }
        } else if r < to_remove.len() {
            let rm = to_remove[r];
            if state.selected_bit[rm] { state.remove_item(rm); }
            r += 1;
        }
        steps += 1;
        if state.total_value > best.value { best = state.clone_solution(); }
    }
    best
}

fn greedy_reconstruct(state: &mut State, strategy: usize, total_inter: &[i32]) {
    let cap = state.ch.max_weight;
    state.to_add.clear();
    for i in 0..state.ch.num_items { if !state.selected_bit[i] { state.to_add.push(i); } }
    let mut cands = vec![];
    cands.append(&mut state.to_add);
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
    state.to_add.append(&mut cands);
}

struct SaListState<'a, 'b> {
    inner: &'b mut State<'a>,
    sel_list: Vec<u32>,
    unsel_list: Vec<u32>,
    pos_in_sel: Vec<i32>,
    pos_in_unsel: Vec<i32>,
}

impl<'a, 'b> SaListState<'a, 'b> {
    fn from_state(st: &'b mut State<'a>) -> Self {
        let n = st.ch.num_items;
        let mut pos_in_sel = vec![-1i32; n];
        let mut pos_in_unsel = vec![-1i32; n];
        let mut sel_list = Vec::with_capacity(n);
        let mut unsel_list = Vec::with_capacity(n);
        for i in 0..n {
            if st.selected_bit[i] {
                pos_in_sel[i] = sel_list.len() as i32;
                sel_list.push(i as u32);
            } else {
                pos_in_unsel[i] = unsel_list.len() as i32;
                unsel_list.push(i as u32);
            }
        }
        Self { inner: st, sel_list, unsel_list, pos_in_sel, pos_in_unsel }
    }

    #[inline(always)]
    fn add(&mut self, i: usize) {
        let pos = self.pos_in_unsel[i] as usize;
        let last = self.unsel_list.len() - 1;
        let last_item = self.unsel_list[last] as usize;
        self.unsel_list.swap(pos, last);
        self.pos_in_unsel[last_item] = pos as i32;
        self.unsel_list.pop();
        self.pos_in_unsel[i] = -1;
        self.pos_in_sel[i] = self.sel_list.len() as i32;
        self.sel_list.push(i as u32);
        self.inner.add_item(i);
    }

    #[inline(always)]
    fn remove(&mut self, i: usize) {
        let pos = self.pos_in_sel[i] as usize;
        let last = self.sel_list.len() - 1;
        let last_item = self.sel_list[last] as usize;
        self.sel_list.swap(pos, last);
        self.pos_in_sel[last_item] = pos as i32;
        self.sel_list.pop();
        self.pos_in_sel[i] = -1;
        self.pos_in_unsel[i] = self.unsel_list.len() as i32;
        self.unsel_list.push(i as u32);
        self.inner.remove_item(i);
    }

    fn restore_from(&mut self, sol: &SolState) {
        let n = self.inner.ch.num_items;
        let mut to_rm: Vec<usize> = Vec::new();
        for i in 0..n { if self.inner.selected_bit[i] && !sol.bits[i] { to_rm.push(i); } }
        for &i in &to_rm { self.remove(i); }
        let mut to_add: Vec<usize> = Vec::new();
        for &i in &sol.items { if !self.inner.selected_bit[i] { to_add.push(i); } }
        for &i in &to_add { self.add(i); }
    }
}

#[inline(always)]
fn linear_accept(delta: i64, temp: f64, rng: &mut Rng) -> bool {
    if delta >= 0 { return true; }
    let prob = 1.0 + (delta as f64) / temp;
    if prob <= 0.0 { return false; }
    (rng.next_u32() as f64) / (u32::MAX as f64 + 1.0) < prob
}

fn run_fast_sa(st: &mut State, rng: &mut Rng, temp_steps: usize, moves_per_step: usize, temp_initial: f64, temp_decay: f64) -> SolState {
    let mut sa = SaListState::from_state(st);
    let mut best = sa.inner.clone_solution();
    let mut temp = temp_initial;
    let mut reheat = 0usize;
    for _step in 0..temp_steps {
        let mut accepted: u64 = 0;
        let mut attempted: u64 = 0;
        for _ in 0..moves_per_step {
            attempted += 1;
            if sa.sel_list.is_empty() || sa.unsel_list.is_empty() { continue; }
            let sel_len = sa.sel_list.len();
            let rm_idx = if rng.next_u32() & 3 != 0 {
                let a = rng.next_usize(sel_len); let b = rng.next_usize(sel_len); let c = rng.next_usize(sel_len);
                let mut best = a;
                if sa.inner.contrib[sa.sel_list[b] as usize] < sa.inner.contrib[sa.sel_list[best] as usize] { best = b; }
                if sa.inner.contrib[sa.sel_list[c] as usize] < sa.inner.contrib[sa.sel_list[best] as usize] { best = c; }
                best
            } else { rng.next_usize(sel_len) };
            
            let unsel_len = sa.unsel_list.len();
            let add_idx = if rng.next_u32() & 3 != 0 {
                let a = rng.next_usize(unsel_len); let b = rng.next_usize(unsel_len); let c = rng.next_usize(unsel_len);
                let mut best = a;
                if sa.inner.contrib[sa.unsel_list[b] as usize] > sa.inner.contrib[sa.unsel_list[best] as usize] { best = b; }
                if sa.inner.contrib[sa.unsel_list[c] as usize] > sa.inner.contrib[sa.unsel_list[best] as usize] { best = c; }
                best
            } else { rng.next_usize(unsel_len) };
            
            let move_type = rng.next_u32() & 7;
            if move_type < 5 {
                let rm = sa.sel_list[rm_idx] as usize;
                let add = sa.unsel_list[add_idx] as usize;
                if sa.inner.total_weight + sa.inner.ch.weights[add] - sa.inner.ch.weights[rm] > sa.inner.ch.max_weight { continue; }
                let delta = sa.inner.contrib[add] as i64 - sa.inner.contrib[rm] as i64 - sa.inner.ch.interaction_values[rm][add] as i64;
                if linear_accept(delta, temp, rng) {
                    sa.remove(rm);
                    sa.add(add);
                    accepted += 1;
                    if sa.inner.total_value > best.value { best = sa.inner.clone_solution(); }
                }
            } else if move_type < 7 {
                let add = sa.unsel_list[add_idx] as usize;
                if sa.inner.total_weight + sa.inner.ch.weights[add] <= sa.inner.ch.max_weight {
                    let delta = sa.inner.contrib[add] as i64;
                    if linear_accept(delta, temp, rng) {
                        sa.add(add);
                        accepted += 1;
                        if sa.inner.total_value > best.value { best = sa.inner.clone_solution(); }
                    }
                }
            } else {
                let rm = sa.sel_list[rm_idx] as usize;
                let delta = -(sa.inner.contrib[rm] as i64);
                if linear_accept(delta, temp, rng) {
                    sa.remove(rm);
                    accepted += 1;
                }
            }
        }
        let ar = if attempted > 0 { accepted as f64 / attempted as f64 } else { 1.0 };
        if ar < 0.001 && reheat < 3 {
            reheat += 1;
            sa.restore_from(&best);
            temp = temp_initial * 0.5f64.powi(reheat as i32);
        } else { temp *= temp_decay; }
    }
    sa.restore_from(&best);
    best
}

fn solve_t43_fisa(ch: &Challenge, save: &dyn Fn(&Solution) -> Result<()>, hp_map: &Option<Map<String, Value>>) -> Result<()> {
    let n = ch.num_items;
    let sum_w: u64 = ch.weights.iter().map(|&w| w as u64).sum();
    let budget_pct = if sum_w > 0 { ((ch.max_weight as u64) * 100 / sum_w) as u32 } else { 25 };
    let hp = Hparams::from_map(hp_map, n, budget_pct);
    let adj = SparseAdj::new(ch);
    let mut rng = Rng::from_seed(&ch.seed);
    let mut best_val: i64 = i64::MIN;
    let mut best_sol: Option<Solution> = None;
    let total_edges = adj.offsets[n];
    let avg_deg = if n > 0 { (total_edges as f64) / (n as f64) } else { 100.0 };
    let base_moves = if avg_deg > 1.0 { ((hp.sa_moves as f64 * 100.0 / avg_deg).ceil() as usize).clamp(4_000, hp.sa_moves) } else { hp.sa_moves };
    let fast_moves = (base_moves as f64 * 2.5) as usize;
    for mode in 0..hp.n_starts.max(1) {
        let mut st = State::new_empty(ch, &adj);
        match mode {
            0 => {

                let mut cands: Vec<usize> = (0..n).collect();
                cands.sort_unstable_by(|&a, &b| {
                    let da = (adj.total_inter[a] as i64 * 1000) / (ch.weights[a] as i64).max(1);
                    let db = (adj.total_inter[b] as i64 * 1000) / (ch.weights[b] as i64).max(1);
                    db.cmp(&da)
                });
                for &i in &cands { if st.total_weight + ch.weights[i] <= ch.max_weight { st.add_item(i); } }
            }
            1 => {

                let mut best_pair_val = 0i64;
                let mut best_pair = (0usize, 0usize);

                for i in 0..n {
                    let start = adj.offsets[i];
                    let end = adj.offsets[i + 1];
                    unsafe {
                        for idx in start..end {
                            let j = *adj.neighbors.get_unchecked(idx) as usize;
                            if i >= j { continue; }
                            let w = *adj.weights.get_unchecked(idx) as i64;
                            let vi = ch.values[i] as i64;
                            let vj = ch.values[j] as i64;
                            let total = vi + vj + w;
                            if ch.weights[i] + ch.weights[j] <= ch.max_weight && total > best_pair_val {
                                best_pair_val = total;
                                best_pair = (i, j);
                            }
                        }
                    }
                }
                st.add_item(best_pair.0);
                st.add_item(best_pair.1);

                for _ in 0..15 {
                    let slack = st.slack(); if slack == 0 { break; }
                    let (mut bi, mut bs) = (None, -1.0);
                    for i in 0..n {
                        if st.selected_bit[i] || ch.weights[i] > slack { continue; }
                        let inter_only = st.contrib[i] - ch.values[i] as i32;
                        let s = inter_only as f64 / (ch.weights[i] as f64).max(1.0);
                        if s > bs { bs = s; bi = Some(i); }
                    }
                    if let Some(i) = bi { st.add_item(i); } else { break; }
                }

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
            2 => {
                for i in 0..n { st.add_item(i); }
                while st.total_weight > ch.max_weight {
                    let (mut worst, mut worst_s) = (0, i64::MAX);
                    for i in 0..n { 
                        if st.selected_bit[i] { 
                            let s = (st.contrib[i] as i64 * 1000) / (ch.weights[i] as i64).max(1); 
                            if s < worst_s { worst_s = s; worst = i; } 
                        } 
                    }
                    st.remove_item(worst);
                }
            }
            _ => {

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
        }
        dp_refinement_hp(&mut st, hp.core_dp, &mut rng);
        let mut curr_best = run_fast_sa(&mut st, &mut rng, hp.sa_steps, fast_moves, hp.sa_t0, hp.sa_decay);
        st.restore_solution(&curr_best);
        
        if curr_best.value > best_val {
            best_val = curr_best.value;
            best_sol = Some(Solution { items: curr_best.items.clone() });
            let _ = save(best_sol.as_ref().unwrap());
        }
        
        let mut stall = 0;
        let ils_rounds = hp.ils_rounds;
        for rnd in 0..ils_rounds {
            for _ in 0..3 {
                let bef_vnd = st.total_value;
                local_search_vnd_windowed(&mut st, hp.window_k.max(150), stall);
                dp_refinement_hp(&mut st, hp.core_dp + 10, &mut rng);
                if st.total_value <= bef_vnd { break; }
            }
            if st.total_value > curr_best.value {
                curr_best = st.clone_solution();
                stall = 0;
                if st.total_value > best_val {
                    best_val = st.total_value;
                    best_sol = Some(Solution { items: st.selected_items() });
                    let _ = save(best_sol.as_ref().unwrap());
                }
            } else {
                stall += 1;
                st.restore_solution(&curr_best);
            }
            
            let strength = hp.perturb_base + stall / 4;
            let s = rnd % 8;
            if rnd % 4 == 0 { cluster_bomb(&mut st, &mut rng, strength); }
            else { perturb_by_strategy(&mut st, strength, stall, s, &mut rng, hp.perturb_base, &adj.total_inter); }
            greedy_reconstruct(&mut st, s % 4, &adj.total_inter);
            
            if stall > 4 && rnd % 2 == 0 {
                let _ = run_fast_sa(&mut st, &mut rng, hp.sa_steps, fast_moves / 2, hp.sa_t0 * 0.5, hp.sa_decay);
            }
        }
    }
    if let Some(ref sol) = best_sol { let _ = save(sol); }
    Ok(())
}
pub fn solve_challenge(
    ch: &Challenge,
    save: &dyn Fn(&Solution) -> Result<()>,
    hp_map: &Option<Map<String, Value>>,
) -> Result<()> {
    solve_t43_fisa(ch, save, hp_map)
}

#[allow(dead_code)]
fn solve_challenge_legacy(
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
        dp_refinement_hp(&mut st, hp.core_dp, &mut rng);
        local_search_vnd_windowed(&mut st, hp.window_k, 0);
        if st.total_value > best_val {
            best_val = st.total_value;
            best_sol = Some(Solution { items: st.selected_items() });
        }
        pop.push(st.clone_solution());
    }

    if let Some(ref sol) = best_sol { let _ = save(sol); }

    pop.sort_unstable_by_key(|s| Reverse(s.value)); pop.truncate(8);

    let mut tabu = Vec::with_capacity(128);
    let zob: Vec<u64> = (0..n).map(|i| { let mut h = 0x517CC1B727220A95u64; h ^= (i as u64).wrapping_mul(0x9E3779B97F4A7C15); h.rotate_left(17).wrapping_mul(0xBF58476D1CE4E5B9) }).collect();
    let chash = |b: &[bool]| -> u64 { let mut h: u64 = 0; for i in 0..n { if b[i] { h ^= zob[i]; } } h };

    let mut st = State::new_empty(ch, &adj);
    st.restore_solution(&pop[0]);
    let mut curr_bval = st.total_value;
    let mut curr_bsel = st.selected_items();
    let mut stall = 0usize;
    tabu.push(chash(&st.selected_bit));

    for rnd in 0..hp.ils_rounds {
        let snap = st.clone_solution();
        dp_refinement_hp(&mut st, hp.core_dp, &mut rng);
        local_search_vnd_windowed(&mut st, hp.window_k, stall);
        if st.total_value > curr_bval {
            if stall > 0 {
                local_search_vnd_windowed(&mut st, hp.window_k, 0);
            }
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

    for i in (0..n).rev() { if st.selected_bit[i] { st.remove_item(i); } }
    for &i in &curr_bsel { st.add_item(i); }

    let elites = pop.len().min(3);
    for k in 1..elites {
        for i in (0..n).rev() { if st.selected_bit[i] { st.remove_item(i); } }
        for &i in &curr_bsel { st.add_item(i); }
        let intermediate = path_relink(&mut st, &pop[k].items, 50);
        st.restore_solution(&intermediate);
        dp_refinement_hp(&mut st, hp.core_dp + 10, &mut rng);
        local_search_vnd_windowed(&mut st, hp.window_k, 0);
        if st.total_value > curr_bval {
            curr_bval = st.total_value;
            curr_bsel = st.selected_items();
        }
    }

    for i in (0..n).rev() { if st.selected_bit[i] { st.remove_item(i); } }
    for &i in &curr_bsel { st.add_item(i); }
    loop {
        let bef = st.total_value;
        local_search_vnd_windowed(&mut st, hp.window_k.max(250), 0);
        dp_refinement_hp(&mut st, hp.core_dp + 20, &mut rng);
        if st.total_value <= bef { break; }
    }
    if st.total_value > best_val {
        best_sol = Some(Solution { items: st.selected_items() });
        let _ = save(best_sol.as_ref().unwrap());
    }

    Ok(())
}

#[allow(dead_code)]
pub fn help() {
    println!("knap_quality_opt_v10.0: Unified SA+VND T43 solver with Clique Construction, 3-Tournament SA, deeper Tabu Search");
}