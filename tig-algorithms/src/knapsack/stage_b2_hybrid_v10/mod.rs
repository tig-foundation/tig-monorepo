use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use std::cmp::Ordering;
use tig_challenges::knapsack::*;

pub mod base_evo9;
pub mod supreme_core;
pub mod track_t39;
pub mod track_t40;
pub mod track_t41;
pub mod track_t42;
pub mod track_t43;

/// Deterministic 64-bit Xorshift PRNG
#[derive(Clone, Copy, Debug)]
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    #[inline]
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x8a5cd789635d2dff } else { seed };
        Self { state }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[inline]
    pub fn gen_range(&mut self, min: u64, max: u64) -> u64 {
        if max <= min {
            return min;
        }
        let span = max - min + 1;
        min + (self.next_u64() % span)
    }
}

#[inline]
pub fn fold_seed_u64(seed: &[u8; 32]) -> u64 {
    let w0 = u64::from_le_bytes(seed[0..8].try_into().unwrap());
    let w1 = u64::from_le_bytes(seed[8..16].try_into().unwrap());
    let w2 = u64::from_le_bytes(seed[16..24].try_into().unwrap());
    let w3 = u64::from_le_bytes(seed[24..32].try_into().unwrap());
    w0 ^ w1.rotate_left(13) ^ w2.rotate_left(29) ^ w3.rotate_left(47)
}

/// Fast incremental state for Quadratic Knapsack with flat row-major interaction matrix
#[derive(Clone, Debug)]
pub struct FlatState {
    pub n: usize,
    pub capacity: i64,
    pub selected: Vec<u8>,
    pub delta: Vec<i64>,
    pub total_value: i64,
    pub total_weight: i64,
}

impl FlatState {
    pub fn new_empty(ch: &Challenge) -> Self {
        let n = ch.num_items;
        let delta: Vec<i64> = ch.values.iter().map(|&v| v as i64).collect();
        Self {
            n,
            capacity: ch.max_weight as i64,
            selected: vec![0u8; n],
            delta,
            total_value: 0,
            total_weight: 0,
        }
    }

    pub fn from_items(ch: &Challenge, q_flat: &[i64], items: &[usize]) -> Self {
        let n = ch.num_items;
        let mut selected = vec![0u8; n];
        let mut total_weight: i64 = 0;
        for &idx in items {
            if idx < n && selected[idx] == 0 {
                selected[idx] = 1;
                total_weight += ch.weights[idx] as i64;
            }
        }
        let sel_indices: Vec<usize> = (0..n).filter(|&i| selected[i] == 1).collect();
        let mut delta = vec![0i64; n];
        let mut total_value: i64 = 0;
        for k in 0..n {
            let mut d = ch.values[k] as i64;
            let row_offset = k * n;
            for &j in &sel_indices {
                if j != k {
                    d += q_flat[row_offset + j];
                }
            }
            delta[k] = d;
            if selected[k] == 1 {
                total_value += ch.values[k] as i64;
                for &j in &sel_indices {
                    if j > k {
                        total_value += q_flat[row_offset + j];
                    }
                }
            }
        }
        Self {
            n,
            capacity: ch.max_weight as i64,
            selected,
            delta,
            total_value,
            total_weight,
        }
    }

    #[inline]
    pub fn slack(&self) -> i64 {
        self.capacity - self.total_weight
    }

    #[inline]
    pub fn add_item(&mut self, ch: &Challenge, q_flat: &[i64], idx: usize) {
        debug_assert_eq!(self.selected[idx], 0);
        self.selected[idx] = 1;
        self.total_weight += ch.weights[idx] as i64;
        self.total_value += self.delta[idx];
        let n = self.n;
        let row_offset = idx * n;
        for k in 0..n {
            if k != idx {
                self.delta[k] += q_flat[row_offset + k];
            }
        }
    }

    #[inline]
    pub fn remove_item(&mut self, ch: &Challenge, q_flat: &[i64], idx: usize) {
        debug_assert_eq!(self.selected[idx], 1);
        self.selected[idx] = 0;
        self.total_weight -= ch.weights[idx] as i64;
        self.total_value -= self.delta[idx];
        let n = self.n;
        let row_offset = idx * n;
        for k in 0..n {
            if k != idx {
                self.delta[k] -= q_flat[row_offset + k];
            }
        }
    }

    pub fn to_solution(&self) -> Solution {
        Solution {
            items: (0..self.n).filter(|&i| self.selected[i] == 1).collect(),
        }
    }

    pub fn zobrist_hash(&self) -> u64 {
        let mut h: u64 = 0;
        for i in 0..self.n {
            if self.selected[i] == 1 {
                let mut v: u64 = 0x517CC1B727220A95;
                v ^= (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
                v = v.rotate_left(17).wrapping_mul(0xBF58476D1CE4E5B9);
                h ^= v;
            }
        }
        h
    }
}

pub fn build_q_flat(ch: &Challenge) -> (Vec<i64>, Vec<i64>, i64) {
    let n = ch.num_items;
    let mut q_flat = vec![0i64; n * n];
    let mut sum_q = vec![0i64; n];
    let mut min_q: i64 = 0;
    for i in 0..n {
        let mut s: i64 = 0;
        let row_offset = i * n;
        for j in 0..n {
            if i != j {
                let v = ch.interaction_values[i][j] as i64;
                q_flat[row_offset + j] = v;
                s += v;
                if v < min_q {
                    min_q = v;
                }
            }
        }
        sum_q[i] = s;
    }
    (q_flat, sum_q, min_q)
}

/// Fast greedy 0-out-1-in and 0-out-2-in slack fill + full N x N 1-out-1-in local search
pub fn fast_1to1_and_slack_climb(state: &mut FlatState, ch: &Challenge, q_flat: &[i64], min_q: i64) {
    let n = state.n;
    loop {
        while state.slack() > 0 {
            let slack = state.slack();
            let mut best_k: Option<usize> = None;
            let mut best_gain: i64 = 0;
            for k in 0..n {
                if state.selected[k] == 0 && (ch.weights[k] as i64) <= slack {
                    let g = state.delta[k];
                    if g > best_gain {
                        best_gain = g;
                        best_k = Some(k);
                    }
                }
            }
            if let Some(k) = best_k {
                state.add_item(ch, q_flat, k);
            } else {
                break;
            }
        }

        let slack = state.slack();
        let mut best_swap: Option<(usize, usize)> = None;
        let mut best_gain: i64 = 0;

        for a in 0..n {
            if state.selected[a] == 0 {
                continue;
            }
            let w_a = ch.weights[a] as i64;
            let d_a = state.delta[a];
            let a_offset = a * n;
            let max_w_b = w_a + slack;
            for b in 0..n {
                if state.selected[b] == 1 {
                    continue;
                }
                if state.delta[b] - d_a - min_q <= best_gain {
                    continue;
                }
                let w_b = ch.weights[b] as i64;
                if w_b <= max_w_b {
                    let gain = state.delta[b] - d_a - q_flat[a_offset + b];
                    if gain > best_gain {
                        best_gain = gain;
                        best_swap = Some((a, b));
                    }
                }
            }
        }

        if let Some((a, b)) = best_swap {
            state.remove_item(ch, q_flat, a);
            state.add_item(ch, q_flat, b);
            continue;
        }
        break;
    }
}

/// Core-Window 0-1 Knapsack DP around the current marginal density break index.
/// Strictly monotonic: accepts the recombined core window (plus fast 1-for-1 repair) iff it strictly improves `state.total_value`.
pub fn integer_core_dp_refine(
    state: &mut FlatState,
    ch: &Challenge,
    q_flat: &[i64],
    min_q: i64,
    core_half_dp: usize,
) -> bool {
    let n = state.n;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        let da = (state.delta[a] as f64) / (ch.weights[a] as f64);
        let db = (state.delta[b] as f64) / (ch.weights[b] as f64);
        db.partial_cmp(&da)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let mut pref_w: i64 = 0;
    let mut break_idx: usize = order.len().saturating_sub(1);
    for (pos, &i) in order.iter().enumerate() {
        let w = ch.weights[i] as i64;
        if pref_w + w > state.capacity {
            break_idx = pos;
            break;
        }
        pref_w += w;
    }

    let left = break_idx.saturating_sub(core_half_dp);
    let right = (break_idx + core_half_dp + 1).min(n);
    let locked = &order[..left];
    let core = &order[left..right];

    let used_locked: i64 = locked.iter().map(|&i| ch.weights[i] as i64).sum();
    let rem_cap_full = (state.capacity - used_locked).max(0) as usize;

    let mut sum_core_w_full: usize = 0;
    for &it in core {
        sum_core_w_full += ch.weights[it] as usize;
    }
    let myw = rem_cap_full.min(sum_core_w_full);
    let myk = core.len();
    if myk == 0 || myw == 0 {
        return false;
    }

    let mut dp: Vec<i64> = vec![i64::MIN / 4; myw + 1];
    dp[0] = 0;
    let mut choose: Vec<u8> = vec![0u8; myk * (myw + 1)];
    let mut w_hi: usize = 0;

    for (t, &it) in core.iter().enumerate() {
        let wt = ch.weights[it] as usize;
        if wt == 0 || wt > myw {
            continue;
        }
        let val = state.delta[it];
        let new_hi = (w_hi + wt).min(myw);
        for w in (wt..=new_hi).rev() {
            let cand = dp[w - wt] + val;
            if cand > dp[w] {
                dp[w] = cand;
                choose[t * (myw + 1) + w] = 1;
            }
        }
        w_hi = new_hi;
    }

    let mut target_sel: Vec<usize> = locked.to_vec();
    let mut w_star = (0..=myw).max_by_key(|&w| dp[w]).unwrap_or(0);
    for t in (0..myk).rev() {
        let it = core[t];
        let wt = ch.weights[it] as usize;
        if wt <= w_star && wt > 0 && choose[t * (myw + 1) + w_star] == 1 {
            target_sel.push(it);
            w_star -= wt;
        }
    }

    // Check if target_sel differs from current state
    let mut differs = false;
    if target_sel.len() != (0..n).filter(|&i| state.selected[i] == 1).count() {
        differs = true;
    } else {
        for &i in &target_sel {
            if state.selected[i] == 0 {
                differs = true;
                break;
            }
        }
    }
    if !differs {
        return false;
    }

    let mut cand = state.clone();
    let mut in_target = vec![false; n];
    for &i in &target_sel {
        in_target[i] = true;
    }
    for i in 0..n {
        if cand.selected[i] == 1 && !in_target[i] {
            cand.remove_item(ch, q_flat, i);
        }
    }
    for &i in &target_sel {
        if cand.selected[i] == 0 && cand.total_weight + (ch.weights[i] as i64) <= cand.capacity {
            cand.add_item(ch, q_flat, i);
        }
    }
    fast_1to1_and_slack_climb(&mut cand, ch, q_flat, min_q);

    if cand.total_value > state.total_value && cand.total_weight <= state.capacity {
        *state = cand;
        true
    } else {
        false
    }
}

/// Builds deduplicated boundary windows of worst-selected and best-unselected items
/// combining marginal density (`delta[i] / weight[i]`) and raw marginal contribution (`delta[i]`).
fn build_boundary_windows(
    state: &FlatState,
    ch: &Challenge,
    k_density: usize,
    k_raw: usize,
) -> (Vec<usize>, Vec<usize>) {
    let n = state.n;
    let mut unsel_den: Vec<(usize, f64)> = Vec::with_capacity(n);
    let mut sel_den: Vec<(usize, f64)> = Vec::with_capacity(n);

    for i in 0..n {
        let d = state.delta[i] as f64;
        let w = ch.weights[i] as f64;
        let ratio = d / w;
        if state.selected[i] == 0 {
            unsel_den.push((i, ratio));
        } else {
            sel_den.push((i, ratio));
        }
    }

    unsel_den.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    sel_den.sort_unstable_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut elite_unsel: Vec<usize> = Vec::with_capacity(k_density + k_raw);
    let mut in_unsel = vec![false; n];
    for &(idx, _) in unsel_den.iter().take(k_density) {
        elite_unsel.push(idx);
        in_unsel[idx] = true;
    }

    let mut elite_sel: Vec<usize> = Vec::with_capacity(k_density + k_raw);
    let mut in_sel = vec![false; n];
    for &(idx, _) in sel_den.iter().take(k_density) {
        elite_sel.push(idx);
        in_sel[idx] = true;
    }

    if k_raw > 0 {
        let mut unsel_raw: Vec<(usize, i64)> = (0..n)
            .filter(|&i| state.selected[i] == 0 && !in_unsel[i])
            .map(|i| (i, state.delta[i]))
            .collect();
        unsel_raw.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        for &(idx, _) in unsel_raw.iter().take(k_raw) {
            elite_unsel.push(idx);
        }

        let mut sel_raw: Vec<(usize, i64)> = (0..n)
            .filter(|&i| state.selected[i] == 1 && !in_sel[i])
            .map(|i| (i, state.delta[i]))
            .collect();
        sel_raw.sort_unstable_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        for &(idx, _) in sel_raw.iter().take(k_raw) {
            elite_sel.push(idx);
        }
    }

    (elite_sel, elite_unsel)
}

#[derive(Clone, Copy)]
struct RemTuple {
    len: u8,
    items: [usize; 3],
    weight: i64,
    loss: i64,
}

#[derive(Clone, Copy)]
struct AddTuple {
    len: u8,
    items: [usize; 3],
    weight: i64,
    gain_base: i64,
}

/// Stage B2 Hybrid Polish Kernel (`TIG-EVO-10`):
/// Exhaustively converges across:
/// - 0-out-1-in, 0-out-2-in, and 0-out-3-in slack fills
/// - Full N x N 1-out-1-in swaps
/// - Unified branch-and-bound pruned (p-out, q-in) coupled multi-swaps for all 1 <= p, q <= 3:
///   (1,2), (2,1), (2,2), (1,3), (3,1), (2,3), (3,2), (3,3)
///   plus coupled residual slack fill evaluation
/// - Core-Window 0-1 Knapsack DP (`core_half_dp = 45` and `25`)
pub fn polish_stage_b2_hybrid(
    state: &mut FlatState,
    ch: &Challenge,
    q_flat: &[i64],
    min_q: i64,
    elite_k_12: usize,
    elite_k_22: usize,
    elite_k_33: usize,
    core_half_dp: usize,
) {
    let n = state.n;

    loop {
        // Phase 1: Fast 0-out-1-in slack fill + full N x N 1-out-1-in climb
        fast_1to1_and_slack_climb(state, ch, q_flat, min_q);

        // Phase 2: 0-out-2-in and 0-out-3-in slack fill when slack >= 2
        if state.slack() >= 2 {
            let slack = state.slack();
            let (_, top_unsel) = build_boundary_windows(state, ch, 50, 10);
            let mut best_add_multi: Option<(u8, [usize; 3])> = None;
            let mut best_gain: i64 = 0;

            let m = top_unsel.len();
            for i in 0..m {
                let u = top_unsel[i];
                let wu = ch.weights[u] as i64;
                if wu >= slack {
                    continue;
                }
                let du = state.delta[u];
                let u_off = u * n;
                for j in (i + 1)..m {
                    let v = top_unsel[j];
                    let wuv = wu + (ch.weights[v] as i64);
                    if wuv > slack {
                        continue;
                    }
                    let g_uv = du + state.delta[v] + q_flat[u_off + v];
                    if g_uv > best_gain {
                        best_gain = g_uv;
                        best_add_multi = Some((2, [u, v, 0]));
                    }
                    if wuv < slack && j < 28 {
                        let v_off = v * n;
                        for k in (j + 1)..m.min(28) {
                            let w_idx = top_unsel[k];
                            let w_tot = wuv + (ch.weights[w_idx] as i64);
                            if w_tot <= slack {
                                let g_uvw = g_uv
                                    + state.delta[w_idx]
                                    + q_flat[u_off + w_idx]
                                    + q_flat[v_off + w_idx];
                                if g_uvw > best_gain {
                                    best_gain = g_uvw;
                                    best_add_multi = Some((3, [u, v, w_idx]));
                                }
                            }
                        }
                    }
                }
            }
            if let Some((cnt, arr)) = best_add_multi {
                for idx in 0..(cnt as usize) {
                    state.add_item(ch, q_flat, arr[idx]);
                }
                continue;
            }
        }

        // Phase 3: Build boundary windows for Stage B2 (1,2), (2,1), (2,2), and higher-order (p,q)
        let (elite_sel, elite_unsel) = build_boundary_windows(state, ch, elite_k_12, 16);
        let slack = state.slack();

        // 3a. Evaluate (1-out-2-in) and (2-out-1-in) on full `elite_sel` x `elite_unsel`
        // including optional coupled 1-item residual slack fill!
        let best_fill_by_weight: [Option<(usize, i64)>; 11] = {
            let mut table: [Option<(usize, i64)>; 11] = [None; 11];
            for &u in &elite_unsel {
                let wu = (ch.weights[u] as usize).min(10);
                let du = state.delta[u];
                if du > 0 {
                    for cap_w in wu..=10 {
                        if table[cap_w].map_or(true, |(_, bd)| du > bd) {
                            table[cap_w] = Some((u, du));
                        }
                    }
                }
            }
            table
        };

        let mut best_move: Option<(u8, [usize; 3], u8, [usize; 3])> = None;
        let mut best_gain: i64 = 0;

        // (1-out, 2-in)
        for &a in &elite_sel {
            let w_a = ch.weights[a] as i64;
            let d_a = state.delta[a];
            let max_bc_w = slack + w_a;
            let a_off = a * n;
            for i_b in 0..elite_unsel.len() {
                let b = elite_unsel[i_b];
                let w_b = ch.weights[b] as i64;
                if w_b >= max_bc_w {
                    continue;
                }
                let d_b = state.delta[b];
                let q_ab = q_flat[a_off + b];
                let b_off = b * n;
                for i_c in (i_b + 1)..elite_unsel.len() {
                    let c = elite_unsel[i_c];
                    let w_bc = w_b + (ch.weights[c] as i64);
                    if w_bc <= max_bc_w {
                        let gain = d_b + state.delta[c] - d_a - q_ab - q_flat[a_off + c] + q_flat[b_off + c];
                        if gain > best_gain {
                            best_gain = gain;
                            best_move = Some((1, [a, 0, 0], 2, [b, c, 0]));
                        }
                    }
                }
            }
        }

        // (2-out, 1-in) + coupled slack-fill check
        for i_a1 in 0..elite_sel.len() {
            let a1 = elite_sel[i_a1];
            let w_a1 = ch.weights[a1] as i64;
            let d_a1 = state.delta[a1];
            let a1_off = a1 * n;
            for i_a2 in (i_a1 + 1)..elite_sel.len() {
                let a2 = elite_sel[i_a2];
                let w_rem = w_a1 + (ch.weights[a2] as i64);
                let max_b_w = slack + w_rem;
                let loss_a12 = d_a1 + state.delta[a2] - q_flat[a1_off + a2];
                let a2_off = a2 * n;
                for &b in &elite_unsel {
                    let w_b = ch.weights[b] as i64;
                    if w_b <= max_b_w {
                        let gain = state.delta[b] - loss_a12 - q_flat[a1_off + b] - q_flat[a2_off + b];
                        if gain > best_gain {
                            best_gain = gain;
                            best_move = Some((2, [a1, a2, 0], 1, [b, 0, 0]));
                        }
                        // Check if residual slack allows coupling a second unselected item `f`
                        let rem_s = (max_b_w - w_b).min(10);
                        if rem_s >= 1 {
                            if let Some((f, _)) = best_fill_by_weight[rem_s as usize] {
                                if f != b {
                                    let coupled_gain = gain + state.delta[f]
                                        + q_flat[b * n + f]
                                        - q_flat[a1_off + f]
                                        - q_flat[a2_off + f];
                                    if coupled_gain > best_gain {
                                        best_gain = coupled_gain;
                                        best_move = Some((2, [a1, a2, 0], 2, [b, f, 0]));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if let Some((r_len, r_arr, a_len, a_arr)) = best_move {
            for i in 0..(r_len as usize) {
                state.remove_item(ch, q_flat, r_arr[i]);
            }
            for i in 0..(a_len as usize) {
                state.add_item(ch, q_flat, a_arr[i]);
            }
            continue;
        }

        // Phase 4: Branch-and-bound pruned (2-out, 2-in) on `elite_k_22`
        let s_22 = elite_k_22.min(elite_sel.len());
        let u_22 = elite_k_22.min(elite_unsel.len());
        if s_22 >= 2 && u_22 >= 2 {
            let mut rem_2: Vec<RemTuple> = Vec::with_capacity(s_22 * (s_22 - 1) / 2);
            for i in 0..s_22 {
                let r1 = elite_sel[i];
                let w1 = ch.weights[r1] as i64;
                let d1 = state.delta[r1];
                let r1_off = r1 * n;
                for j in (i + 1)..s_22 {
                    let r2 = elite_sel[j];
                    let loss = d1 + state.delta[r2] - q_flat[r1_off + r2];
                    rem_2.push(RemTuple {
                        len: 2,
                        items: [r1, r2, 0],
                        weight: w1 + (ch.weights[r2] as i64),
                        loss,
                    });
                }
            }
            rem_2.sort_unstable_by_key(|r| r.loss);

            let mut add_2: Vec<AddTuple> = Vec::with_capacity(u_22 * (u_22 - 1) / 2);
            for i in 0..u_22 {
                let a1 = elite_unsel[i];
                let w1 = ch.weights[a1] as i64;
                let d1 = state.delta[a1];
                let a1_off = a1 * n;
                for j in (i + 1)..u_22 {
                    let a2 = elite_unsel[j];
                    let gain_base = d1 + state.delta[a2] + q_flat[a1_off + a2];
                    add_2.push(AddTuple {
                        len: 2,
                        items: [a1, a2, 0],
                        weight: w1 + (ch.weights[a2] as i64),
                        gain_base,
                    });
                }
            }
            add_2.sort_unstable_by(|a, b| b.gain_base.cmp(&a.gain_base));

            if !rem_2.is_empty() && !add_2.is_empty() {
                let max_add_gain = add_2[0].gain_base;
                let min_cross_22 = 4 * min_q;
                let mut best_22: Option<([usize; 3], [usize; 3])> = None;
                let mut best_22_gain: i64 = 0;

                for r in &rem_2 {
                    if max_add_gain - r.loss - min_cross_22 <= best_22_gain {
                        break;
                    }
                    let max_w = slack + r.weight;
                    let r0 = r.items[0];
                    let r1 = r.items[1];
                    let r0_off = r0 * n;
                    let r1_off = r1 * n;
                    for a in &add_2 {
                        if a.gain_base - r.loss - min_cross_22 <= best_22_gain {
                            break;
                        }
                        if a.weight <= max_w {
                            let a0 = a.items[0];
                            let a1 = a.items[1];
                            let cross = q_flat[r0_off + a0]
                                + q_flat[r0_off + a1]
                                + q_flat[r1_off + a0]
                                + q_flat[r1_off + a1];
                            let gain = a.gain_base - r.loss - cross;
                            if gain > best_22_gain {
                                best_22_gain = gain;
                                best_22 = Some((r.items, a.items));
                            }
                        }
                    }
                }

                if let Some((r_arr, a_arr)) = best_22 {
                    state.remove_item(ch, q_flat, r_arr[0]);
                    state.remove_item(ch, q_flat, r_arr[1]);
                    state.add_item(ch, q_flat, a_arr[0]);
                    state.add_item(ch, q_flat, a_arr[1]);
                    continue;
                }
            }
        }

        // Phase 5: Higher-order branch-and-bound pruned coupled swaps:
        // (1,3), (3,1), (2,3), (3,2), and (3,3) on `elite_k_33`
        let s_33 = elite_k_33.min(elite_sel.len());
        let u_33 = elite_k_33.min(elite_unsel.len());
        if s_33 >= 3 && u_33 >= 3 {
            let mut rem_1: Vec<RemTuple> = (0..s_33)
                .map(|i| {
                    let r0 = elite_sel[i];
                    RemTuple {
                        len: 1,
                        items: [r0, 0, 0],
                        weight: ch.weights[r0] as i64,
                        loss: state.delta[r0],
                    }
                })
                .collect();
            rem_1.sort_unstable_by_key(|r| r.loss);

            let mut rem_2: Vec<RemTuple> = Vec::with_capacity(s_33 * (s_33 - 1) / 2);
            let mut rem_3: Vec<RemTuple> = Vec::with_capacity(s_33 * (s_33 - 1) * (s_33 - 2) / 6);
            for i in 0..s_33 {
                let r0 = elite_sel[i];
                let w0 = ch.weights[r0] as i64;
                let d0 = state.delta[r0];
                let r0_off = r0 * n;
                for j in (i + 1)..s_33 {
                    let r1 = elite_sel[j];
                    let w01 = w0 + (ch.weights[r1] as i64);
                    let loss01 = d0 + state.delta[r1] - q_flat[r0_off + r1];
                    rem_2.push(RemTuple {
                        len: 2,
                        items: [r0, r1, 0],
                        weight: w01,
                        loss: loss01,
                    });
                    let r1_off = r1 * n;
                    for k in (j + 1)..s_33 {
                        let r2 = elite_sel[k];
                        let loss012 = loss01 + state.delta[r2] - q_flat[r0_off + r2] - q_flat[r1_off + r2];
                        rem_3.push(RemTuple {
                            len: 3,
                            items: [r0, r1, r2],
                            weight: w01 + (ch.weights[r2] as i64),
                            loss: loss012,
                        });
                    }
                }
            }
            rem_2.sort_unstable_by_key(|r| r.loss);
            rem_3.sort_unstable_by_key(|r| r.loss);

            let mut add_1: Vec<AddTuple> = (0..u_33)
                .map(|i| {
                    let a0 = elite_unsel[i];
                    AddTuple {
                        len: 1,
                        items: [a0, 0, 0],
                        weight: ch.weights[a0] as i64,
                        gain_base: state.delta[a0],
                    }
                })
                .collect();
            add_1.sort_unstable_by(|a, b| b.gain_base.cmp(&a.gain_base));

            let mut add_2: Vec<AddTuple> = Vec::with_capacity(u_33 * (u_33 - 1) / 2);
            let mut add_3: Vec<AddTuple> = Vec::with_capacity(u_33 * (u_33 - 1) * (u_33 - 2) / 6);
            for i in 0..u_33 {
                let a0 = elite_unsel[i];
                let w0 = ch.weights[a0] as i64;
                let d0 = state.delta[a0];
                let a0_off = a0 * n;
                for j in (i + 1)..u_33 {
                    let a1 = elite_unsel[j];
                    let w01 = w0 + (ch.weights[a1] as i64);
                    let g01 = d0 + state.delta[a1] + q_flat[a0_off + a1];
                    add_2.push(AddTuple {
                        len: 2,
                        items: [a0, a1, 0],
                        weight: w01,
                        gain_base: g01,
                    });
                    let a1_off = a1 * n;
                    for k in (j + 1)..u_33 {
                        let a2 = elite_unsel[k];
                        let g012 = g01 + state.delta[a2] + q_flat[a0_off + a2] + q_flat[a1_off + a2];
                        add_3.push(AddTuple {
                            len: 3,
                            items: [a0, a1, a2],
                            weight: w01 + (ch.weights[a2] as i64),
                            gain_base: g012,
                        });
                    }
                }
            }
            add_2.sort_unstable_by(|a, b| b.gain_base.cmp(&a.gain_base));
            add_3.sort_unstable_by(|a, b| b.gain_base.cmp(&a.gain_base));

            let mut best_ho: Option<(RemTuple, AddTuple)> = None;
            let mut best_ho_gain: i64 = 0;

            let pairs_to_check: [(&[RemTuple], &[AddTuple]); 5] = [
                (&rem_1, &add_3),
                (&rem_3, &add_1),
                (&rem_2, &add_3),
                (&rem_3, &add_2),
                (&rem_3[..rem_3.len().min(320)], &add_3[..add_3.len().min(320)]),
            ];

            for &(rems, adds) in &pairs_to_check {
                if rems.is_empty() || adds.is_empty() {
                    continue;
                }
                let max_add_g = adds[0].gain_base;
                let min_cross = (rems[0].len as i64) * (adds[0].len as i64) * min_q;
                for r in rems {
                    if max_add_g - r.loss - min_cross <= best_ho_gain {
                        break;
                    }
                    let max_w = slack + r.weight;
                    let r_len = r.len as usize;
                    for a in adds {
                        if a.gain_base - r.loss - min_cross <= best_ho_gain {
                            break;
                        }
                        if a.weight <= max_w {
                            let a_len = a.len as usize;
                            let mut cross: i64 = 0;
                            for ir in 0..r_len {
                                let roff = r.items[ir] * n;
                                for ia in 0..a_len {
                                    cross += q_flat[roff + a.items[ia]];
                                }
                            }
                            let gain = a.gain_base - r.loss - cross;
                            if gain > best_ho_gain {
                                best_ho_gain = gain;
                                best_ho = Some((*r, *a));
                            }
                        }
                    }
                }
            }

            if let Some((r, a)) = best_ho {
                for i in 0..(r.len as usize) {
                    state.remove_item(ch, q_flat, r.items[i]);
                }
                for i in 0..(a.len as usize) {
                    state.add_item(ch, q_flat, a.items[i]);
                }
                continue;
            }
        }

        // Phase 6: Core-Window 0-1 Knapsack DP (`core_half_dp` and `core_half_dp / 2`)
        if core_half_dp > 0 {
            if integer_core_dp_refine(state, ch, q_flat, min_q, core_half_dp) {
                continue;
            }
            let half_dp = (core_half_dp / 2).max(16);
            if half_dp != core_half_dp && integer_core_dp_refine(state, ch, q_flat, min_q, half_dp) {
                continue;
            }
        }

        break;
    }
}

/// Bidirectional Path Relinking + Intersection/Union Recombination across an elite pool of `FlatState`s.
pub fn recombine_and_relink_elites(
    best_state: &mut FlatState,
    elites: &[FlatState],
    ch: &Challenge,
    q_flat: &[i64],
    min_q: i64,
) {
    let n = ch.num_items;
    if elites.is_empty() {
        return;
    }

    // 1. Intersection + Greedy/DP Fill & Union + Worst-Density Drop across top distinct elites
    let limit = elites.len().min(5);
    for i in 0..limit {
        for j in (i + 1)..limit {
            let e1 = &elites[i];
            let e2 = &elites[j];
            let mut hamming = 0usize;
            for k in 0..n {
                if e1.selected[k] != e2.selected[k] {
                    hamming += 1;
                }
            }
            if hamming < 4 {
                continue;
            }

            // Intersection + greedy density fill
            let mut inter_st = FlatState::new_empty(ch);
            for k in 0..n {
                if e1.selected[k] == 1 && e2.selected[k] == 1 {
                    inter_st.add_item(ch, q_flat, k);
                }
            }
            while inter_st.slack() > 0 {
                let slack = inter_st.slack();
                let mut best_item: Option<(usize, f64)> = None;
                for k in 0..n {
                    if inter_st.selected[k] == 0
                        && (ch.weights[k] as i64) <= slack
                        && inter_st.delta[k] > 0
                    {
                        let r = (inter_st.delta[k] as f64) / (ch.weights[k] as f64);
                        if best_item.map_or(true, |(_, br)| r > br) {
                            best_item = Some((k, r));
                        }
                    }
                }
                if let Some((k, _)) = best_item {
                    inter_st.add_item(ch, q_flat, k);
                } else {
                    break;
                }
            }
            integer_core_dp_refine(&mut inter_st, ch, q_flat, min_q, 35);
            fast_1to1_and_slack_climb(&mut inter_st, ch, q_flat, min_q);
            if inter_st.total_value > best_state.total_value - 450 {
                polish_stage_b2_hybrid(&mut inter_st, ch, q_flat, min_q, 60, 48, 22, 40);
                if inter_st.total_value > best_state.total_value {
                    *best_state = inter_st;
                }
            }
        }
    }

    // 2. Bidirectional Path Relinking between `best_state` and each distinct elite
    let base_snapshot = best_state.clone();
    for guide in elites.iter().take(6) {
        for &(src, dst) in &[(&base_snapshot, guide), (guide, &base_snapshot)] {
            let mut to_add: Vec<usize> = (0..n)
                .filter(|&k| dst.selected[k] == 1 && src.selected[k] == 0)
                .collect();
            let mut to_rem: Vec<usize> = (0..n)
                .filter(|&k| src.selected[k] == 1 && dst.selected[k] == 0)
                .collect();

            let total_diff = to_add.len() + to_rem.len();
            if total_diff < 4 {
                continue;
            }

            let mut cur = src.clone();
            let mut best_along: Option<FlatState> = None;
            let mut best_along_val = best_state.total_value.saturating_sub(250);
            let step_check = (total_diff / 4).max(3);
            let mut steps = 0usize;

            while !to_add.is_empty() || !to_rem.is_empty() {
                let slack = cur.slack();
                let mut best_act: Option<(bool, usize, i64)> = None;

                for (idx, &item) in to_add.iter().enumerate() {
                    if (ch.weights[item] as i64) <= slack {
                        let d = cur.delta[item];
                        if best_act.map_or(true, |(_, _, bd)| d > bd) {
                            best_act = Some((true, idx, d));
                        }
                    }
                }
                for (idx, &item) in to_rem.iter().enumerate() {
                    let d = -cur.delta[item];
                    if best_act.map_or(true, |(_, _, bd)| d > bd) {
                        best_act = Some((false, idx, d));
                    }
                }

                match best_act {
                    Some((true, idx, _)) => {
                        let item = to_add.swap_remove(idx);
                        cur.add_item(ch, q_flat, item);
                    }
                    Some((false, idx, _)) => {
                        let item = to_rem.swap_remove(idx);
                        cur.remove_item(ch, q_flat, item);
                    }
                    None => break,
                }
                steps += 1;

                if cur.total_weight <= cur.capacity && cur.total_value > best_along_val {
                    best_along_val = cur.total_value;
                    best_along = Some(cur.clone());
                }

                if steps % step_check == 0 && cur.total_weight <= cur.capacity {
                    let mut probe = cur.clone();
                    fast_1to1_and_slack_climb(&mut probe, ch, q_flat, min_q);
                    if probe.total_value > best_along_val {
                        best_along_val = probe.total_value;
                        best_along = Some(probe);
                    }
                }
            }

            if let Some(mut cand) = best_along {
                polish_stage_b2_hybrid(&mut cand, ch, q_flat, min_q, 64, 50, 22, 42);
                if cand.total_value > best_state.total_value {
                    *best_state = cand;
                }
            }
        }
    }
}

/// Non-improving Boundary-Window Tabu Escape Burst + Stage B2 Polish
pub fn tabu_basin_escape(
    best_state: &mut FlatState,
    ch: &Challenge,
    q_flat: &[i64],
    min_q: i64,
    seed_u64: u64,
    num_bursts: usize,
    steps_per_burst: usize,
) {
    let n = best_state.n;
    let mut rng = Xorshift64::new(seed_u64 ^ 0xA5A5_5A5A_F0F0_0F0F);

    for burst in 0..num_bursts {
        let mut cur = best_state.clone();
        let mut tabu_until = vec![0usize; n];

        // Optional light strategic perturbation on burst >= 1
        if burst > 0 {
            let (elite_sel, elite_unsel) = build_boundary_windows(&cur, ch, 40, 10);
            let k_drop = (2 + (burst % 3)).min(elite_sel.len());
            for _ in 0..k_drop {
                let pick = (rng.next_u64() as usize) % elite_sel.len().min(18);
                let rm = elite_sel[pick];
                if cur.selected[rm] == 1 {
                    cur.remove_item(ch, q_flat, rm);
                    tabu_until[rm] = 6 + (burst % 4);
                }
            }
            for &u in &elite_unsel {
                if cur.selected[u] == 0
                    && (ch.weights[u] as i64) <= cur.slack()
                    && cur.delta[u] > 0
                {
                    cur.add_item(ch, q_flat, u);
                }
            }
        }

        let mut best_burst_state: Option<FlatState> = None;

        for step in 1..=steps_per_burst {
            while cur.slack() > 0 {
                let slack = cur.slack();
                let mut best_k: Option<usize> = None;
                let mut best_g: i64 = 0;
                for k in 0..n {
                    if cur.selected[k] == 0
                        && tabu_until[k] < step
                        && (ch.weights[k] as i64) <= slack
                        && cur.delta[k] > best_g
                    {
                        best_g = cur.delta[k];
                        best_k = Some(k);
                    }
                }
                if let Some(k) = best_k {
                    cur.add_item(ch, q_flat, k);
                } else {
                    break;
                }
            }

            if cur.total_value > best_state.total_value {
                *best_state = cur.clone();
                best_burst_state = Some(cur.clone());
            }

            let (elite_sel, elite_unsel) = build_boundary_windows(&cur, ch, 45, 10);
            let slack = cur.slack();
            let mut best_tabu_swap: Option<(usize, usize, i64)> = None;

            for &a in &elite_sel {
                let w_a = ch.weights[a] as i64;
                let d_a = cur.delta[a];
                let a_off = a * n;
                let max_wb = w_a + slack;
                for &b in &elite_unsel {
                    let w_b = ch.weights[b] as i64;
                    if w_b > max_wb {
                        continue;
                    }
                    let gain = cur.delta[b] - d_a - q_flat[a_off + b];
                    let is_tabu = tabu_until[a] >= step || tabu_until[b] >= step;
                    let aspiration = cur.total_value + gain > best_state.total_value;
                    if !is_tabu || aspiration {
                        if best_tabu_swap.map_or(true, |(_, _, bg)| gain > bg) {
                            best_tabu_swap = Some((a, b, gain));
                        }
                    }
                }
            }

            if let Some((a, b, _)) = best_tabu_swap {
                cur.remove_item(ch, q_flat, a);
                cur.add_item(ch, q_flat, b);
                let tenure = 5 + (rng.gen_range(0, 4) as usize);
                tabu_until[a] = step + tenure;
                tabu_until[b] = step + tenure;
                if cur.total_value > best_state.total_value {
                    *best_state = cur.clone();
                }
            } else {
                break;
            }
        }

        if let Some(mut st) = best_burst_state {
            polish_stage_b2_hybrid(&mut st, ch, q_flat, min_q, 65, 52, 24, 42);
            if st.total_value > best_state.total_value {
                *best_state = st;
            }
        } else {
            polish_stage_b2_hybrid(&mut cur, ch, q_flat, min_q, 55, 44, 20, 38);
            if cur.total_value > best_state.total_value {
                *best_state = cur;
            }
        }
    }
}

/// `TIG-EVO-10a` Standalone Solver:
/// Multi-warm-start Stage A (`power_w` in `[0.80, 0.72, 0.90, 1.00]` + `knap_supreme` Core-DP trajectory),
/// stripped of the artificial 5.0M fuel brake, with Core-Window DP, Boundary-Windowed VND,
/// Stage C Ruin-and-Recreate Kicks, Unified (p-out, q-in) Stage B2 Multi-Swap Polish,
/// and Bidirectional Path Relinking.
pub fn solve_core_dp_b2_standalone_elites(
    ch: &Challenge,
    q_flat: &[i64],
    sum_q: &[i64],
    min_q: i64,
) -> (FlatState, Vec<FlatState>) {
    let n = ch.num_items;
    let seed_u64 = fold_seed_u64(&ch.seed);
    let mut elites: Vec<FlatState> = Vec::with_capacity(8);
    let mut elite_hashes: Vec<u64> = Vec::with_capacity(8);

    let mut register_elite = |st: FlatState| {
        let h = st.zobrist_hash();
        if !elite_hashes.contains(&h) {
            elite_hashes.push(h);
            elites.push(st);
        }
    };

    // 1. Run `supreme_core` Core-DP + ILS trajectory and polish it with `polish_stage_b2_hybrid`
    let saved_sup = std::cell::RefCell::new(None);
    if supreme_core::solve_challenge(
        ch,
        &|sol| {
            *saved_sup.borrow_mut() = Some(sol.clone());
            Ok(())
        },
        &None,
    )
    .is_ok()
    {
        if let Some(sol_sup) = saved_sup.into_inner() {
            let mut st_sup = FlatState::from_items(ch, q_flat, &sol_sup.items);
            polish_stage_b2_hybrid(&mut st_sup, ch, q_flat, min_q, 65, 52, 24, 42);
            register_elite(st_sup);
        }
    }

    // 2. Multi-warm-start Stage A (`degree_weight_num = 1164`, `power_w` in `[0.80, 0.72, 0.90, 1.00]`)
    //    + Core-Window DP + Unbraked Stage B/C Tabu & Kicks + Stage B2 Hybrid Polish
    let warm_configs: [(i64, f64, usize); 4] = [
        (1164, 0.80, 10),
        (1164, 0.72, 6),
        (1164, 0.90, 6),
        (1024, 1.00, 6),
    ];

    for (cfg_idx, &(deg_w, power_w, max_kicks)) in warm_configs.iter().enumerate() {
        let mut scores: Vec<(usize, f64)> = (0..n)
            .map(|i| {
                let num = ((ch.values[i] as i64) * 1024 + deg_w * sum_q[i]) as f64;
                let den = (ch.weights[i] as f64).powf(power_w).max(1.0);
                (i, num / den)
            })
            .collect();
        scores.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        let mut st = FlatState::new_empty(ch);
        for &(idx, _) in &scores {
            if st.total_weight + (ch.weights[idx] as i64) <= st.capacity {
                st.add_item(ch, q_flat, idx);
            }
        }

        // Core-Window DP + Fast 1-for-1 + Stage B2 polish on initial warm start
        integer_core_dp_refine(&mut st, ch, q_flat, min_q, 35);
        polish_stage_b2_hybrid(&mut st, ch, q_flat, min_q, 60, 48, 22, 40);

        let mut best_cfg_st = st.clone();
        let mut kick_rng = Xorshift64::new(
            seed_u64 ^ 0x9E37_79B9_7F4A_7C15 ^ ((cfg_idx as u64).wrapping_mul(0x517CC1B727220A95)),
        );

        for kick in 0..max_kicks {
            let mut cur = best_cfg_st.clone();
            let sel_indices: Vec<usize> = (0..n).filter(|&i| cur.selected[i] == 1).collect();
            if sel_indices.is_empty() {
                break;
            }

            if kick % 2 == 0 {
                // Stage C Ruin-and-Recreate kick (`ruin_rate = 0.045`)
                let num_to_drop = ((sel_indices.len() as f64) * 0.045).ceil() as usize;
                for _ in 0..num_to_drop {
                    let pick = kick_rng.gen_range(0, (sel_indices.len() - 1) as u64) as usize;
                    let drop_item = sel_indices[pick];
                    if cur.selected[drop_item] == 1 {
                        cur.remove_item(ch, q_flat, drop_item);
                    }
                }
                let mut candidates: Vec<usize> = (0..n).filter(|&i| cur.selected[i] == 0).collect();
                for i in (1..candidates.len()).rev() {
                    let j = kick_rng.gen_range(0, i as u64) as usize;
                    candidates.swap(i, j);
                }
                for &cand in &candidates {
                    if cur.total_weight + (ch.weights[cand] as i64) <= cur.capacity
                        && cur.delta[cand] > 0
                    {
                        cur.add_item(ch, q_flat, cand);
                    }
                }
            } else {
                // Boundary worst-density drop + synergy rebuild
                let (elite_sel, elite_unsel) = build_boundary_windows(&cur, ch, 35, 8);
                let drop_cnt = (4 + (kick % 5)).min(elite_sel.len());
                for i in 0..drop_cnt {
                    let idx = elite_sel[(i + (kick_rng.next_u64() as usize)) % elite_sel.len().min(16)];
                    if cur.selected[idx] == 1 {
                        cur.remove_item(ch, q_flat, idx);
                    }
                }
                for &u in &elite_unsel {
                    if cur.selected[u] == 0
                        && (ch.weights[u] as i64) <= cur.slack()
                        && cur.delta[u] > 0
                    {
                        cur.add_item(ch, q_flat, u);
                    }
                }
            }

            integer_core_dp_refine(&mut cur, ch, q_flat, min_q, 35);
            fast_1to1_and_slack_climb(&mut cur, ch, q_flat, min_q);
            if cur.total_value >= best_cfg_st.total_value.saturating_sub(300) {
                polish_stage_b2_hybrid(&mut cur, ch, q_flat, min_q, 56, 44, 20, 38);
                if cur.total_value > best_cfg_st.total_value {
                    best_cfg_st = cur;
                }
            }
        }

        register_elite(best_cfg_st);
    }

    elites.sort_unstable_by(|a, b| b.total_value.cmp(&a.total_value));
    let mut best_state = elites
        .first()
        .cloned()
        .unwrap_or_else(|| FlatState::new_empty(ch));

    recombine_and_relink_elites(&mut best_state, &elites, ch, q_flat, min_q);
    tabu_basin_escape(&mut best_state, ch, q_flat, min_q, seed_u64, 4, 25);
    polish_stage_b2_hybrid(&mut best_state, ch, q_flat, min_q, 70, 56, 26, 45);

    (best_state, elites)
}

/// Public entry point for `TIG-EVO-10a` (Pure standalone Core-DP + Unbraked Stage B2 solver)
pub fn solve_core_dp_b2_standalone(challenge: &Challenge) -> Solution {
    let (q_flat, sum_q, min_q) = build_q_flat(challenge);
    let (best_state, _) = solve_core_dp_b2_standalone_elites(challenge, &q_flat, &sum_q, min_q);
    best_state.to_solution()
}

#[derive(Clone, Debug, Default)]
pub struct Hparams {}

impl Hparams {
    pub fn for_t39() -> Self {
        Self::default()
    }
    pub fn for_t40() -> Self {
        Self::default()
    }
    pub fn for_t41() -> Self {
        Self::default()
    }
    pub fn for_t42() -> Self {
        Self::default()
    }
    pub fn for_t43() -> Self {
        Self::default()
    }
}

/// Full Multi-Basin Hybrid Fallback (`TIG-EVO-10b`):
/// Combines standalone `TIG-EVO-10a` elite basins (`power_w = 0.80` Stage A + Core-DP + Unbraked Stage B2)
/// with top-6 polished restart candidates on N=1000 (`55W / 45T / 0L`, `+49.06` pts vs `knap_quality_opt_v11`).
pub fn solve_full_hybrid_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let n = challenge.num_items;
    if n == 0 || n > 20_000 {
        return Err(anyhow!("Invalid challenge num_items: {}", n));
    }

    let sum_w: u64 = challenge.weights.iter().map(|&w| w as u64).sum();
    let budget_pct = if sum_w > 0 {
        ((challenge.max_weight as u64) * 100 / sum_w) as u32
    } else {
        10
    };

    if n > 1200 {
        let saved = std::cell::RefCell::new(None);
        if budget_pct <= 17 {
            let _ = track_t42::solve(
                challenge,
                &|sol| {
                    *saved.borrow_mut() = Some(sol.clone());
                    save_solution(sol)
                },
                hyperparameters,
            );
        } else {
            let _ = track_t43::solve(
                challenge,
                &|sol| {
                    *saved.borrow_mut() = Some(sol.clone());
                    save_solution(sol)
                },
                hyperparameters,
            );
        }
        if let Some(s) = saved.into_inner() {
            let (q_flat, _sum_q, min_q) = build_q_flat(challenge);
            let mut st = FlatState::from_items(challenge, &q_flat, &s.items);
            if st.total_weight <= st.capacity {
                let prev_val = st.total_value;
                polish_stage_b2_hybrid(&mut st, challenge, &q_flat, min_q, 48, 40, 18, 32);
                if st.total_value > prev_val {
                    let sol = st.to_solution();
                    save_solution(&sol)?;
                }
            }
        }
        return Ok(());
    }

    let (q_flat, sum_q, min_q) = build_q_flat(challenge);
    let seed_u64 = fold_seed_u64(&challenge.seed);

    // Step 1: Run our standalone `TIG-EVO-10a` solver and checkpoint immediately
    let (mut best_state, mut elites) =
        solve_core_dp_b2_standalone_elites(challenge, &q_flat, &sum_q, min_q);
    let mut last_saved_val = best_state.total_value;
    save_solution(&best_state.to_solution())?;

    if n <= 600 {
        let saved_evo9 = std::cell::RefCell::new(None);
        if base_evo9::solve_challenge(
            challenge,
            &|sol| {
                *saved_evo9.borrow_mut() = Some(sol.clone());
                Ok(())
            },
            &None,
        )
        .is_ok()
        {
            if let Some(sol_evo9) = saved_evo9.into_inner() {
                let mut st_evo9 = FlatState::from_items(challenge, &q_flat, &sol_evo9.items);
                polish_stage_b2_hybrid(&mut st_evo9, challenge, &q_flat, min_q, 70, 56, 26, 45);
                if st_evo9.total_value > best_state.total_value {
                    best_state = st_evo9.clone();
                    if best_state.total_value > last_saved_val {
                        last_saved_val = best_state.total_value;
                        save_solution(&best_state.to_solution())?;
                    }
                }
                elites.push(st_evo9);
            }
        }
        elites.sort_unstable_by(|a, b| b.total_value.cmp(&a.total_value));
        recombine_and_relink_elites(&mut best_state, &elites, challenge, &q_flat, min_q);
        if best_state.total_value > last_saved_val {
            last_saved_val = best_state.total_value;
            save_solution(&best_state.to_solution())?;
        }
        tabu_basin_escape(&mut best_state, challenge, &q_flat, min_q, seed_u64, 5, 28);
        if best_state.total_value > last_saved_val {
            save_solution(&best_state.to_solution())?;
        }
        polish_stage_b2_hybrid(&mut best_state, challenge, &q_flat, min_q, 72, 58, 26, 45);
    } else {
        let mut raw_candidates: Vec<Solution> = Vec::new();
        if budget_pct <= 7 {
            let saved = std::cell::RefCell::new(None);
            let _ = track_t41::solve(
                challenge,
                &|sol| {
                    *saved.borrow_mut() = Some(sol.clone());
                    Ok(())
                },
                hyperparameters,
            );
            if let Some(s) = saved.into_inner() {
                raw_candidates.push(s);
            }
        } else if budget_pct <= 17 {
            raw_candidates = track_t39::collect_candidate_solutions(
                challenge,
                hyperparameters,
            );
        } else {
            raw_candidates = track_t40::collect_restart_solutions(
                challenge,
                hyperparameters,
            );
        }

        // Convert each candidate into a `FlatState`, guarantee baseline floor, and polish top candidates
        let mut cand_states: Vec<FlatState> = Vec::with_capacity(raw_candidates.len());
        for sol in &raw_candidates {
            let st = FlatState::from_items(challenge, &q_flat, &sol.items);
            if st.total_weight <= st.capacity {
                if st.total_value > best_state.total_value {
                    best_state = st.clone();
                }
                cand_states.push(st);
            }
        }
        if best_state.total_value > last_saved_val {
            last_saved_val = best_state.total_value;
            save_solution(&best_state.to_solution())?;
        }
        cand_states.sort_unstable_by(|a, b| b.total_value.cmp(&a.total_value));

        let mut seen_hashes: Vec<u64> = elites.iter().map(|e| e.zobrist_hash()).collect();
        for (idx, mut st) in cand_states.into_iter().enumerate() {
            let h = st.zobrist_hash();
            if seen_hashes.contains(&h) && idx > 0 {
                continue;
            }
            seen_hashes.push(h);

            // Polish top 6 restart/basin candidates with full Stage B2 (1<=p,q<=3) + Core-DP
            if idx < 6 {
                polish_stage_b2_hybrid(&mut st, challenge, &q_flat, min_q, 72, 58, 26, 45);
            } else {
                polish_stage_b2_hybrid(&mut st, challenge, &q_flat, min_q, 52, 40, 20, 35);
            }
            if st.total_value > best_state.total_value {
                best_state = st.clone();
                if best_state.total_value > last_saved_val {
                    last_saved_val = best_state.total_value;
                    save_solution(&best_state.to_solution())?;
                }
            }
            elites.push(st);
        }

        elites.sort_unstable_by(|a, b| b.total_value.cmp(&a.total_value));
        elites.truncate(8);

        // Cross-recombine & path-relink between the Core-DP+ILS restarts and our `TIG-EVO-10a` elites
        recombine_and_relink_elites(&mut best_state, &elites, challenge, &q_flat, min_q);
        if best_state.total_value > last_saved_val {
            last_saved_val = best_state.total_value;
            save_solution(&best_state.to_solution())?;
        }

        // Run Boundary-Window Tabu Escape Bursts + Stage B2 Hybrid Polish
        tabu_basin_escape(&mut best_state, challenge, &q_flat, min_q, seed_u64, 6, 30);
        if best_state.total_value > last_saved_val {
            save_solution(&best_state.to_solution())?;
        }
        polish_stage_b2_hybrid(&mut best_state, challenge, &q_flat, min_q, 76, 60, 28, 48);
    }

    let sol = best_state.to_solution();
    save_solution(&sol)?;
    Ok(())
}

/// Public entry point for `TIG-EVO-10` (`stage_b2_hybrid_v10`):
/// - For `N = 1,000` (`600 < n <= 1200`), defaults to `FP3` (`solve_fast_polish_single_relink`:
///   Single-Winner Stage B2 Polish + Top-2 Runner-Up Path Relink + 3-Burst Tabu Escape,
///   `+9.51%` time overhead vs `knap_quality_opt_v11`, `39W / 61T / 0L`, `+17.86` pts mean gain),
///   with `solve_full_hybrid_challenge` (`55W / 45T / 0L`, `+49.06` pts) retained as fallback.
/// - For `N <= 600` and `N > 1200`, delegates to `solve_full_hybrid_challenge`.
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let n = challenge.num_items;
    if n == 0 || n > 20_000 {
        return Err(anyhow!("Invalid challenge num_items: {}", n));
    }

    let use_full_hybrid = hyperparameters
        .as_ref()
        .and_then(|hp| hp.get("mode"))
        .and_then(|v| v.as_str())
        .map(|s| s == "full_hybrid")
        .unwrap_or(false);

    if n > 600 && n <= 1200 && !use_full_hybrid {
        if solve_fast_polish_single_relink(challenge, save_solution, hyperparameters).is_ok() {
            return Ok(());
        }
    }

    solve_full_hybrid_challenge(challenge, save_solution, hyperparameters)
}

/// Fast-Polish Variant 1: Restrict Stage B2 polish strictly to the single winning candidate
/// returned by `track_t39::solve` / `track_t40::solve` / `track_t41::solve` (no standalone preamble,
/// no multi-candidate polish, no Tabu bursts).
pub fn solve_fast_polish_single_b2_only(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let sum_w: u64 = challenge.weights.iter().map(|&w| w as u64).sum();
    let budget_pct = if sum_w > 0 {
        ((challenge.max_weight as u64) * 100 / sum_w) as u32
    } else {
        10
    };

    let saved = std::cell::RefCell::new(None);
    if budget_pct <= 7 {
        track_t41::solve(
            challenge,
            &|sol| {
                *saved.borrow_mut() = Some(sol.clone());
                save_solution(sol)
            },
            hyperparameters,
        )?;
    } else if budget_pct <= 17 {
        track_t39::solve(
            challenge,
            &|sol| {
                *saved.borrow_mut() = Some(sol.clone());
                save_solution(sol)
            },
            hyperparameters,
        )?;
    } else {
        track_t40::solve(
            challenge,
            &|sol| {
                *saved.borrow_mut() = Some(sol.clone());
                save_solution(sol)
            },
            hyperparameters,
        )?;
    }

    if let Some(s) = saved.into_inner() {
        let (q_flat, _sum_q, min_q) = build_q_flat(challenge);
        let mut st = FlatState::from_items(challenge, &q_flat, &s.items);
        let base_val = st.total_value;
        polish_stage_b2_hybrid(&mut st, challenge, &q_flat, min_q, 76, 60, 28, 48);
        if st.total_value > base_val {
            save_solution(&st.to_solution())?;
        }
    }
    Ok(())
}

/// Fast-Polish Variant 2: Restrict Stage B2 polish + 3-burst boundary Tabu escape strictly
/// to the single winning candidate from `track_t39::solve` / `track_t40::solve` / `track_t41::solve`.
pub fn solve_fast_polish_single_b2_tabu(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let sum_w: u64 = challenge.weights.iter().map(|&w| w as u64).sum();
    let budget_pct = if sum_w > 0 {
        ((challenge.max_weight as u64) * 100 / sum_w) as u32
    } else {
        10
    };

    let saved = std::cell::RefCell::new(None);
    if budget_pct <= 7 {
        track_t41::solve(
            challenge,
            &|sol| {
                *saved.borrow_mut() = Some(sol.clone());
                save_solution(sol)
            },
            hyperparameters,
        )?;
    } else if budget_pct <= 17 {
        track_t39::solve(
            challenge,
            &|sol| {
                *saved.borrow_mut() = Some(sol.clone());
                save_solution(sol)
            },
            hyperparameters,
        )?;
    } else {
        track_t40::solve(
            challenge,
            &|sol| {
                *saved.borrow_mut() = Some(sol.clone());
                save_solution(sol)
            },
            hyperparameters,
        )?;
    }

    if let Some(s) = saved.into_inner() {
        let (q_flat, _sum_q, min_q) = build_q_flat(challenge);
        let seed_u64 = fold_seed_u64(&challenge.seed);
        let mut st = FlatState::from_items(challenge, &q_flat, &s.items);
        let mut last_val = st.total_value;
        polish_stage_b2_hybrid(&mut st, challenge, &q_flat, min_q, 76, 60, 28, 48);
        if st.total_value > last_val {
            last_val = st.total_value;
            save_solution(&st.to_solution())?;
        }
        tabu_basin_escape(&mut st, challenge, &q_flat, min_q, seed_u64, 3, 24);
        polish_stage_b2_hybrid(&mut st, challenge, &q_flat, min_q, 76, 60, 28, 48);
        if st.total_value > last_val {
            save_solution(&st.to_solution())?;
        }
    }
    Ok(())
}

/// Fast-Polish Variant 3: Restrict initial Stage B2 polish strictly to the single winning candidate
/// from `T39`/`T40`, plus path-relink with the runner-up candidate already computed during `T39`/`T40`
/// and 3 Tabu escape bursts (no 294ms standalone preamble, no 6-candidate polish loop).
pub fn solve_fast_polish_single_relink(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let sum_w: u64 = challenge.weights.iter().map(|&w| w as u64).sum();
    let budget_pct = if sum_w > 0 {
        ((challenge.max_weight as u64) * 100 / sum_w) as u32
    } else {
        10
    };

    let mut raw_candidates: Vec<Solution> = Vec::new();
    if budget_pct <= 7 {
        let saved = std::cell::RefCell::new(None);
        let _ = track_t41::solve(
            challenge,
            &|sol| {
                *saved.borrow_mut() = Some(sol.clone());
                save_solution(sol)
            },
            hyperparameters,
        );
        if let Some(s) = saved.into_inner() {
            raw_candidates.push(s);
        }
    } else if budget_pct <= 17 {
        raw_candidates = track_t39::collect_candidate_solutions(challenge, hyperparameters);
    } else {
        raw_candidates = track_t40::collect_restart_solutions(challenge, hyperparameters);
    }

    let (q_flat, _sum_q, min_q) = build_q_flat(challenge);
    let seed_u64 = fold_seed_u64(&challenge.seed);
    let mut cand_states: Vec<FlatState> = raw_candidates
        .iter()
        .map(|sol| FlatState::from_items(challenge, &q_flat, &sol.items))
        .filter(|st| st.total_weight <= st.capacity)
        .collect();
    cand_states.sort_unstable_by(|a, b| b.total_value.cmp(&a.total_value));

    if cand_states.is_empty() {
        return Err(anyhow!("No feasible candidate returned"));
    }

    let mut best_state = cand_states[0].clone();
    let mut last_val = best_state.total_value;
    save_solution(&best_state.to_solution())?;

    // Restrict Stage B2 polish strictly to the single winning candidate (`cand_states[0]`)
    polish_stage_b2_hybrid(&mut best_state, challenge, &q_flat, min_q, 76, 60, 28, 48);
    if best_state.total_value > last_val {
        last_val = best_state.total_value;
        save_solution(&best_state.to_solution())?;
    }

    // Optional ultra-light path-relink with the top distinct runner-up already produced by T39/T40
    let mut top_pair = vec![best_state.clone()];
    let best_h = best_state.zobrist_hash();
    for st in cand_states.into_iter().skip(1) {
        if st.zobrist_hash() != best_h {
            top_pair.push(st);
            if top_pair.len() >= 3 {
                break;
            }
        }
    }
    if top_pair.len() >= 2 {
        recombine_and_relink_elites(&mut best_state, &top_pair, challenge, &q_flat, min_q);
        if best_state.total_value > last_val {
            last_val = best_state.total_value;
            save_solution(&best_state.to_solution())?;
        }
    }

    tabu_basin_escape(&mut best_state, challenge, &q_flat, min_q, seed_u64, 3, 24);
    polish_stage_b2_hybrid(&mut best_state, challenge, &q_flat, min_q, 76, 60, 28, 48);
    if best_state.total_value > last_val {
        save_solution(&best_state.to_solution())?;
    }
    Ok(())
}

pub fn help() {
    println!("Stage B2 Hybrid V10 (TIG-EVO-10): Core-Window DP + Unified (p-out, q-in) (1<=p,q<=3) Multi-Swap Polish + Cross-Basin Path Relinking");
}
