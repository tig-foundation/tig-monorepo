use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use std::cell::RefCell;
use std::cmp::Ordering;
use tig_challenges::knapsack::*;

// debt: the helper block below (Xorshift64 through tabu_basin_escape) is a copy of stage_b2_hybrid_v10/mod.rs.
// A fix made in one copy does not reach the other. Ceiling: retire with v10 from the active gauntlet.

pub mod track1;
pub mod track2;
pub mod track3;
pub mod track4;
pub mod track5;
pub mod track_t39;

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

pub const LDIV: [i64; 11] = [2520, 2520, 1260, 840, 630, 504, 420, 360, 315, 280, 252];

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
        let mut delta: Vec<i64> = ch.values.iter().map(|&v| v as i64).collect();
        for j in 0..n {
            if selected[j] == 1 {
                let row = &q_flat[j * n..(j + 1) * n];
                for (d, &q) in delta.iter_mut().zip(row.iter()) {
                    *d += q;
                }
                delta[j] -= row[j];
            }
        }
        let mut twice_total: i64 = 0;
        for k in 0..n {
            if selected[k] == 1 {
                twice_total += (ch.values[k] as i64) + delta[k];
            }
        }
        Self {
            n,
            capacity: ch.max_weight as i64,
            selected,
            delta,
            total_value: twice_total / 2,
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
        let row = &q_flat[idx * n..(idx + 1) * n];
        for (d, &q) in self.delta.iter_mut().zip(row.iter()) {
            *d += q;
        }
        self.delta[idx] -= row[idx];
    }

    #[inline]
    pub fn remove_item(&mut self, ch: &Challenge, q_flat: &[i64], idx: usize) {
        debug_assert_eq!(self.selected[idx], 1);
        self.selected[idx] = 0;
        self.total_weight -= ch.weights[idx] as i64;
        self.total_value -= self.delta[idx];
        let n = self.n;
        let row = &q_flat[idx * n..(idx + 1) * n];
        for (d, &q) in self.delta.iter_mut().zip(row.iter()) {
            *d -= q;
        }
        self.delta[idx] += row[idx];
    }

    #[inline]
    pub fn replace_item(&mut self, ch: &Challenge, q_flat: &[i64], rm: usize, add: usize) {
        debug_assert_eq!(self.selected[rm], 1);
        debug_assert_eq!(self.selected[add], 0);
        let n = self.n;
        let q_rm_add = q_flat[rm * n + add];
        self.selected[rm] = 0;
        self.selected[add] = 1;
        self.total_weight += (ch.weights[add] as i64) - (ch.weights[rm] as i64);
        self.total_value += self.delta[add] - self.delta[rm] - q_rm_add;
        let row_rm = &q_flat[rm * n..(rm + 1) * n];
        let row_add = &q_flat[add * n..(add + 1) * n];
        for ((d, &q_r), &q_a) in self.delta.iter_mut().zip(row_rm.iter()).zip(row_add.iter()) {
            *d += q_a - q_r;
        }
        self.delta[rm] += row_rm[rm];
        self.delta[add] -= row_add[add];
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
        let row_i = i * n;
        let iv_i = &ch.interaction_values[i];
        let dst = &mut q_flat[row_i..row_i + n];
        let mut s_i: i64 = 0;
        for j in 0..n {
            let v = iv_i[j] as i64;
            dst[j] = v;
            s_i += v;
            if v < min_q {
                min_q = v;
            }
        }
        sum_q[i] = s_i;
    }
    (q_flat, sum_q, min_q)
}

fn eval_sol_if_feasible(ch: &Challenge, sol: &Solution) -> Option<i64> {
    let n = ch.num_items;
    let mut seen = vec![0u8; n];
    let mut w: i64 = 0;
    let mut sel: Vec<usize> = Vec::with_capacity(sol.items.len());
    for &idx in &sol.items {
        if idx >= n || seen[idx] != 0 {
            continue;
        }
        seen[idx] = 1;
        w += ch.weights[idx] as i64;
        if w > ch.max_weight as i64 {
            return None;
        }
        sel.push(idx);
    }
    let mut total: i64 = 0;
    for (pos, &u) in sel.iter().enumerate() {
        total += ch.values[u] as i64;
        let row = &ch.interaction_values[u];
        for &v in &sel[pos + 1..] {
            total += row[v] as i64;
        }
    }
    Some(total)
}

thread_local! {
    static CLIMB_CAND_UNSEL: std::cell::RefCell<Vec<usize>> = std::cell::RefCell::new(Vec::with_capacity(128));
}

/// Fast greedy 0-out-1-in and 0-out-2-in slack fill + full N x N 1-out-1-in local search
pub fn fast_1to1_and_slack_climb(state: &mut FlatState, ch: &Challenge, q_flat: &[i64], min_q: i64) {
    let n = state.n;
    loop {
        while state.slack() > 0 {
            let slack = state.slack();
            let mut best_k = usize::MAX;
            let mut best_gain: i64 = 0;
            for k in 0..n {
                if state.selected[k] == 0 && (ch.weights[k] as i64) <= slack {
                    let g = state.delta[k];
                    if g > best_gain {
                        best_gain = g;
                        best_k = k;
                    }
                }
            }
            if best_k != usize::MAX {
                state.add_item(ch, q_flat, best_k);
            } else {
                break;
            }
        }

        let slack = state.slack();
        let mut best_swap_a = usize::MAX;
        let mut best_swap_b = usize::MAX;
        let mut best_gain: i64 = 0;

        let mut max_unsel_delta: i64 = i64::MIN;
        let mut min_sel_delta: i64 = i64::MAX;
        for i in 0..n {
            let d = state.delta[i];
            if state.selected[i] == 0 {
                if d > max_unsel_delta {
                    max_unsel_delta = d;
                }
            } else if d < min_sel_delta {
                min_sel_delta = d;
            }
        }
        if max_unsel_delta == i64::MIN || min_sel_delta == i64::MAX || max_unsel_delta - min_sel_delta - min_q <= 0 {
            break;
        }

        CLIMB_CAND_UNSEL.with(|cand_cell| {
            let mut cand_unsel = cand_cell.borrow_mut();
            cand_unsel.clear();
            let thr = min_sel_delta + min_q;
            for b in 0..n {
                if state.selected[b] == 0 && state.delta[b] > thr {
                    cand_unsel.push(b);
                }
            }

            for a in 0..n {
                if state.selected[a] == 0 {
                    continue;
                }
                let d_a = state.delta[a];
                if max_unsel_delta - d_a - min_q <= best_gain {
                    continue;
                }
                let w_a = ch.weights[a] as i64;
                let a_offset = a * n;
                let row_a = &q_flat[a_offset..a_offset + n];
                let max_w_b = w_a + slack;
                for &b in cand_unsel.iter() {
                    if state.delta[b] - d_a - min_q <= best_gain {
                        continue;
                    }
                    let w_b = ch.weights[b] as i64;
                    if w_b <= max_w_b {
                        let gain = state.delta[b] - d_a - row_a[b];
                        if gain > best_gain {
                            best_gain = gain;
                            best_swap_a = a;
                            best_swap_b = b;
                        }
                    }
                }
            }
        });

        if best_swap_a != usize::MAX {
            state.replace_item(ch, q_flat, best_swap_a, best_swap_b);
            continue;
        }
        break;
    }
}

struct CoreDpWorkspace {
    order: Vec<usize>,
    dp: Vec<i64>,
    choose: Vec<u8>,
    target_sel: Vec<usize>,
    in_target: Vec<bool>,
}

thread_local! {
    static CORE_DP_WS: std::cell::RefCell<CoreDpWorkspace> = std::cell::RefCell::new(CoreDpWorkspace {
        order: Vec::with_capacity(5000),
        dp: Vec::with_capacity(4096),
        choose: Vec::with_capacity(16384),
        target_sel: Vec::with_capacity(5000),
        in_target: Vec::with_capacity(5000),
    });
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

    CORE_DP_WS.with(|ws_cell| {
        let mut ws = ws_cell.borrow_mut();
        let CoreDpWorkspace { order, dp, choose, target_sel, in_target } = &mut *ws;
        order.clear();
        order.extend(0..n);
        order.sort_unstable_by(|&a, &b| {
            let wa = ch.weights[a] as usize;
            let wb = ch.weights[b] as usize;
            if wa >= 1 && wa <= 10 && wb >= 1 && wb <= 10 {
                let ka = state.delta[a] * LDIV[wa];
                let kb = state.delta[b] * LDIV[wb];
                kb.cmp(&ka).then_with(|| a.cmp(&b))
            } else {
                let da = (state.delta[a] as f64) / (ch.weights[a] as f64);
                let db = (state.delta[b] as f64) / (ch.weights[b] as f64);
                db.partial_cmp(&da)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.cmp(&b))
            }
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

        dp.clear();
        dp.resize(myw + 1, i64::MIN / 4);
        dp[0] = 0;
        choose.clear();
        choose.resize(myk * (myw + 1), 0u8);
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

        target_sel.clear();
        target_sel.extend_from_slice(locked);
        let mut w_star = (0..=myw).max_by_key(|&w| dp[w]).unwrap_or(0);
        for t in (0..myk).rev() {
            let it = core[t];
            let wt = ch.weights[it] as usize;
            if wt <= w_star && wt > 0 && choose[t * (myw + 1) + w_star] == 1 {
                target_sel.push(it);
                w_star -= wt;
            }
        }

        if target_sel.is_empty() {
            return false;
        }

        if in_target.len() < n {
            in_target.resize(n, false);
        } else {
            in_target[..n].fill(false);
        }
        for &i in target_sel.iter() {
            in_target[i] = true;
        }

        let mut differs = false;
        for i in 0..n {
            if (state.selected[i] == 1) != in_target[i] {
                differs = true;
                break;
            }
        }
        if !differs {
            return false;
        }

        let mut cand = state.clone();
        for i in 0..n {
            if cand.selected[i] == 1 && !in_target[i] {
                cand.remove_item(ch, q_flat, i);
            }
        }
        for &i in target_sel.iter() {
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
    })
}

/// Family 1 (`exact_quad_bnb`): Exact Quadratic Branch-and-Bound Core Lookahead on `FlatState`.
/// Selects the `k_total` (`<= 24`) items closest to the marginal density break (`k_total/2` worst-selected
/// and `k_total/2` best-unselected), locks the outside items, and solves the exact sub-QKP
/// $\max \sum_{a \in C} v^{\text{base}}_a y_a + \sum_{a < b \in C} Q_{ab} y_a y_b$ subject to
/// $\sum_{a \in C} w_a y_a \le W_{\text{rem}}$ using recursive Branch-and-Bound with positive-interaction
/// suffix upper-bound pruning (`node_cap = 35_000`), followed by `fast_1to1_and_slack_climb`.
pub fn exact_quad_bnb_refine(
    state: &mut FlatState,
    ch: &Challenge,
    q_flat: &[i64],
    min_q: i64,
    k_total: usize,
) -> bool {
    if k_total < 4 {
        return false;
    }
    let k_half = (k_total / 2).min(12);
    let (sel_win, unsel_win) = build_boundary_windows(state, ch, k_half, 2);
    let mut core: Vec<usize> = Vec::with_capacity(2 * k_half + 4);
    for &i in sel_win.iter().take(k_half) {
        core.push(i);
    }
    for &i in unsel_win.iter().take(k_half) {
        if !core.contains(&i) {
            core.push(i);
        }
    }
    let m = core.len();
    if m < 4 || m > 26 {
        return false;
    }

    let n = state.n;

    // Compute exact marginal value of each core item relative to fixed (selected \ core) items:
    // base_v[a] = state.delta[u] - sum_{v in core \cap selected, v != u} Q[u, v]
    let mut raw_v = vec![0i64; m];
    let mut raw_w = vec![0i64; m];
    let mut cur_core_w: i64 = 0;

    for (a, &u) in core.iter().enumerate() {
        let mut bv = state.delta[u];
        let u_off = u * n;
        for (b, &v) in core.iter().enumerate() {
            if a != b && state.selected[v] == 1 {
                bv -= q_flat[u_off + v];
            }
        }
        raw_v[a] = bv;
        raw_w[a] = ch.weights[u] as i64;
        if state.selected[u] == 1 {
            cur_core_w += raw_w[a];
        }
    }

    let w_rem = state.slack() + cur_core_w;
    if w_rem <= 0 {
        return false;
    }

    // Order core items by optimistic density: (raw_v[a] + sum_{b != a} max(0, Q[a,b]) / 2) / w[a]
    let mut opt_density = [0.0f64; 28];
    for a in 0..m {
        let ua = core[a];
        let mut pos_a = 0i64;
        let ua_off = ua * n;
        for &v in &core {
            if v != ua {
                let q = q_flat[ua_off + v];
                if q > 0 {
                    pos_a += q;
                }
            }
        }
        opt_density[a] = ((raw_v[a] + pos_a / 2) as f64) / (raw_w[a].max(1) as f64);
    }

    let mut order: Vec<usize> = (0..m).collect();
    order.sort_unstable_by(|&a, &b| {
        opt_density[b]
            .partial_cmp(&opt_density[a])
            .unwrap_or(Ordering::Equal)
            .then_with(|| core[a].cmp(&core[b]))
    });

    let ordered_items: Vec<usize> = order.iter().map(|&idx| core[idx]).collect();
    let base_v: Vec<i64> = order.iter().map(|&idx| raw_v[idx]).collect();
    let cw: Vec<i64> = order.iter().map(|&idx| raw_w[idx]).collect();
    let mut cq = vec![0i64; m * m];
    for i in 0..m {
        let ui = ordered_items[i];
        for j in 0..m {
            if i != j {
                let uj = ordered_items[j];
                cq[i * m + j] = q_flat[ui * n + uj];
            }
        }
    }

    // Compute current core objective and mask in ordered coordinates
    let mut init_mask: u32 = 0;
    let mut init_val: i64 = 0;
    for i in 0..m {
        let ui = ordered_items[i];
        if state.selected[ui] == 1 {
            init_mask |= 1u32 << i;
            init_val += base_v[i];
            for j in 0..i {
                if (init_mask & (1u32 << j)) != 0 {
                    init_val += cq[i * m + j];
                }
            }
        }
    }

    // Suffix upper bound: sum_{t >= idx} max(0, base_v[t] + sum_{s != t} max(0, cq[t, s]))
    let mut suffix_ub = vec![0i64; m + 1];
    for i in (0..m).rev() {
        let max_pos_row: i64 = (0..m).filter(|&j| j != i).map(|j| cq[i * m + j].max(0)).sum();
        let item_ub = (base_v[i] + max_pos_row).max(0);
        suffix_ub[i] = suffix_ub[i + 1] + item_ub;
    }

    struct BnbCtx<'a> {
        m: usize,
        w_rem: i64,
        base_v: &'a [i64],
        cw: &'a [i64],
        cq: &'a [i64],
        suffix_ub: &'a [i64],
        best_val: i64,
        best_mask: u32,
        nodes: usize,
        node_cap: usize,
    }

    fn dfs(ctx: &mut BnbCtx<'_>, idx: usize, cur_w: i64, cur_val: i64, cur_mask: u32) {
        if cur_val > ctx.best_val {
            ctx.best_val = cur_val;
            ctx.best_mask = cur_mask;
        }
        if idx >= ctx.m || ctx.nodes >= ctx.node_cap {
            return;
        }
        if cur_val + ctx.suffix_ub[idx] <= ctx.best_val {
            return;
        }
        ctx.nodes += 1;

        // Branch 1: include `idx` if weight fits
        let w_i = ctx.cw[idx];
        if cur_w + w_i <= ctx.w_rem {
            let mut gain = ctx.base_v[idx];
            let row_off = idx * ctx.m;
            let mut m_bits = cur_mask;
            while m_bits != 0 {
                let j = m_bits.trailing_zeros() as usize;
                gain += ctx.cq[row_off + j];
                m_bits &= m_bits - 1;
            }
            dfs(ctx, idx + 1, cur_w + w_i, cur_val + gain, cur_mask | (1u32 << idx));
        }

        // Branch 2: exclude `idx`
        if cur_val + ctx.suffix_ub[idx + 1] > ctx.best_val {
            dfs(ctx, idx + 1, cur_w, cur_val, cur_mask);
        }
    }

    let mut ctx = BnbCtx {
        m,
        w_rem,
        base_v: &base_v,
        cw: &cw,
        cq: &cq,
        suffix_ub: &suffix_ub,
        best_val: init_val,
        best_mask: init_mask,
        nodes: 0,
        node_cap: 35_000,
    };

    dfs(&mut ctx, 0, 0, 0, 0);

    if ctx.best_mask == init_mask {
        return false;
    }

    let mut cand = state.clone();
    for i in 0..m {
        let u = ordered_items[i];
        let want = (ctx.best_mask & (1u32 << i)) != 0;
        if cand.selected[u] == 1 && !want {
            cand.remove_item(ch, q_flat, u);
        }
    }
    for i in 0..m {
        let u = ordered_items[i];
        let want = (ctx.best_mask & (1u32 << i)) != 0;
        if cand.selected[u] == 0 && want && cand.total_weight + (ch.weights[u] as i64) <= cand.capacity {
            cand.add_item(ch, q_flat, u);
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

thread_local! {
    static BUCKET_BUFFERS: std::cell::RefCell<([Vec<RemTuple>; 21], [Vec<AddTuple>; 21])> = std::cell::RefCell::new(Default::default());
}

struct PolishStageBuffers {
    rem_1: Vec<RemTuple>,
    rem_2: Vec<RemTuple>,
    rem_3: Vec<RemTuple>,
    add_1: Vec<AddTuple>,
    add_2: Vec<AddTuple>,
    add_3: Vec<AddTuple>,
}

thread_local! {
    static POLISH_STAGE_BUFFERS: std::cell::RefCell<PolishStageBuffers> = std::cell::RefCell::new(PolishStageBuffers {
        rem_1: Vec::with_capacity(64),
        rem_2: Vec::with_capacity(2048),
        rem_3: Vec::with_capacity(4096),
        add_1: Vec::with_capacity(64),
        add_2: Vec::with_capacity(2048),
        add_3: Vec::with_capacity(4096),
    });
}

/// Family 2 (`bucket_compound`): Weight-Bucketed `O(W^2)` Wide-Window `(2-out, 2-in)` Exact Compound Lookahead.
/// Buckets all `O(K^2)` add-pairs and remove-pairs by pair weight (`2..=20`), sorting each weight bucket
/// descending (for adds) or ascending (for removes), and prunes entire weight buckets whenever the bucket
/// leader cannot beat `best_gain` even with `min_cross_22`. Allows `K = 80..110` at minimal overhead.
pub fn bucket_compound_22_refine(
    state: &mut FlatState,
    ch: &Challenge,
    q_flat: &[i64],
    min_q: i64,
    k_window: usize,
) -> bool {
    if k_window < 16 {
        return false;
    }
    let (elite_sel, elite_unsel) = build_boundary_windows(state, ch, k_window, 16);
    let s_len = elite_sel.len().min(k_window + 16);
    let u_len = elite_unsel.len().min(k_window + 16);
    if s_len < 2 || u_len < 2 {
        return false;
    }
    let n = state.n;
    let slack = state.slack();

    let mut best_pair: Option<([usize; 3], [usize; 3])> = None;

    BUCKET_BUFFERS.with(|cell| {
        let mut buffers = cell.borrow_mut();
        let (rem_buckets, add_buckets) = &mut *buffers;
        for b in rem_buckets.iter_mut() {
            b.clear();
        }
        for b in add_buckets.iter_mut() {
            b.clear();
        }

        let mut min_rem_loss = [i64::MAX / 4; 21];
        for i in 0..s_len {
            let r1 = elite_sel[i];
            let w1 = ch.weights[r1] as i64;
            let d1 = state.delta[r1];
            let r1_off = r1 * n;
            for j in (i + 1)..s_len {
                let r2 = elite_sel[j];
                let w = (w1 + (ch.weights[r2] as i64)).clamp(0, 20) as usize;
                let loss = d1 + state.delta[r2] - q_flat[r1_off + r2];
                if loss < min_rem_loss[w] {
                    min_rem_loss[w] = loss;
                }
                rem_buckets[w].push(RemTuple {
                    len: 2,
                    items: [r1, r2, 0],
                    weight: w1 + (ch.weights[r2] as i64),
                    loss,
                });
            }
        }
        for w in (0..20).rev() {
            min_rem_loss[w] = min_rem_loss[w].min(min_rem_loss[w + 1]);
        }
        for b in rem_buckets.iter_mut() {
            b.sort_unstable_by_key(|r| r.loss);
        }

        let min_cross_22 = 4 * min_q;
        for i in 0..u_len {
            let a1 = elite_unsel[i];
            let w1 = ch.weights[a1] as i64;
            let d1 = state.delta[a1];
            let a1_off = a1 * n;
            for j in (i + 1)..u_len {
                let a2 = elite_unsel[j];
                let w_raw = w1 + (ch.weights[a2] as i64);
                let need_w = (w_raw - slack).clamp(0, 20) as usize;
                let gain_base = d1 + state.delta[a2] + q_flat[a1_off + a2];
                if gain_base - min_rem_loss[need_w] - min_cross_22 <= 0 {
                    continue;
                }
                let w_idx = w_raw.clamp(0, 20) as usize;
                add_buckets[w_idx].push(AddTuple {
                    len: 2,
                    items: [a1, a2, 0],
                    weight: w_raw,
                    gain_base,
                });
            }
        }
        for b in add_buckets.iter_mut() {
            b.sort_unstable_by(|a, b| b.gain_base.cmp(&a.gain_base));
        }

        let mut active_rem_w = [0usize; 21];
        let mut num_active_rem = 0;
        for w in 0..=20 {
            if !rem_buckets[w].is_empty() {
                active_rem_w[num_active_rem] = w;
                num_active_rem += 1;
            }
        }

        let mut active_add_w = [0usize; 21];
        let mut num_active_add = 0;
        for w in 0..=20 {
            if !add_buckets[w].is_empty() {
                active_add_w[num_active_add] = w;
                num_active_add += 1;
            }
        }

        if num_active_rem == 0 || num_active_add == 0 {
            return;
        }

        let mut best_gain: i64 = 0;

        for &rw in &active_rem_w[..num_active_rem] {
            let r_bucket = &rem_buckets[rw];
            let min_r_loss = r_bucket[0].loss;
            for &aw in &active_add_w[..num_active_add] {
                let a_bucket = &add_buckets[aw];
                if a_bucket[0].gain_base - min_r_loss - min_cross_22 <= best_gain {
                    continue;
                }
                for r in r_bucket {
                    if a_bucket[0].gain_base - r.loss - min_cross_22 <= best_gain {
                        break;
                    }
                    let max_w = slack + r.weight;
                    if (aw as i64) > max_w && aw < 20 {
                        continue;
                    }
                    let r0 = r.items[0];
                    let r1 = r.items[1];
                    let r0_off = r0 * n;
                    let r1_off = r1 * n;
                    let row_r0 = &q_flat[r0_off..r0_off + n];
                    let row_r1 = &q_flat[r1_off..r1_off + n];
                    for a in a_bucket {
                        if a.gain_base - r.loss - min_cross_22 <= best_gain {
                            break;
                        }
                        if a.weight <= max_w {
                            let a0 = a.items[0];
                            let a1 = a.items[1];
                            let cross = row_r0[a0]
                                + row_r0[a1]
                                + row_r1[a0]
                                + row_r1[a1];
                            let gain = a.gain_base - r.loss - cross;
                            if gain > best_gain {
                                best_gain = gain;
                                best_pair = Some((r.items, a.items));
                            }
                        }
                    }
                }
            }
        }
    });

    if let Some((r_arr, a_arr)) = best_pair {
        state.remove_item(ch, q_flat, r_arr[0]);
        state.remove_item(ch, q_flat, r_arr[1]);
        state.add_item(ch, q_flat, a_arr[0]);
        state.add_item(ch, q_flat, a_arr[1]);
        fast_1to1_and_slack_climb(state, ch, q_flat, min_q);
        true
    } else {
        false
    }
}

struct BoundaryBuffers {
    unsel_den: Vec<(usize, i64)>,
    sel_den: Vec<(usize, i64)>,
    unsel_raw: Vec<(usize, i64)>,
    sel_raw: Vec<(usize, i64)>,
    in_sel: Vec<bool>,
    in_unsel: Vec<bool>,
}

thread_local! {
    static BOUNDARY_BUFFERS: std::cell::RefCell<BoundaryBuffers> = std::cell::RefCell::new(BoundaryBuffers {
        unsel_den: Vec::with_capacity(5000),
        sel_den: Vec::with_capacity(5000),
        unsel_raw: Vec::with_capacity(5000),
        sel_raw: Vec::with_capacity(5000),
        in_sel: Vec::with_capacity(5000),
        in_unsel: Vec::with_capacity(5000),
    });
}

/// Builds deduplicated boundary windows of worst-selected and best-unselected items
/// combining marginal density (`delta[i] / weight[i]`) and raw marginal contribution (`delta[i]`)
/// directly into caller-provided vectors.
fn build_boundary_windows_into(
    state: &FlatState,
    ch: &Challenge,
    k_density: usize,
    k_raw: usize,
    elite_sel: &mut Vec<usize>,
    elite_unsel: &mut Vec<usize>,
) {
    elite_sel.clear();
    elite_unsel.clear();
    let n = state.n;
    BOUNDARY_BUFFERS.with(|bufs| {
        let BoundaryBuffers {
            unsel_den,
            sel_den,
            unsel_raw,
            sel_raw,
            in_sel,
            in_unsel,
        } = &mut *bufs.borrow_mut();

        unsel_den.clear();
        sel_den.clear();

        for i in 0..n {
            let d = state.delta[i];
            let w = (ch.weights[i] as usize).clamp(1, 10);
            let key = d * LDIV[w];
            if state.selected[i] == 0 {
                unsel_den.push((i, key));
            } else {
                sel_den.push((i, key));
            }
        }

        let cmp_desc = |a: &(usize, i64), b: &(usize, i64)| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0));
        let cmp_asc = |a: &(usize, i64), b: &(usize, i64)| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0));

        let k_u = k_density.min(unsel_den.len());
        if k_u > 0 && k_u < unsel_den.len() {
            unsel_den.select_nth_unstable_by(k_u - 1, cmp_desc);
            unsel_den[..k_u].sort_unstable_by(cmp_desc);
        } else {
            unsel_den.sort_unstable_by(cmp_desc);
        }

        let k_s = k_density.min(sel_den.len());
        if k_s > 0 && k_s < sel_den.len() {
            sel_den.select_nth_unstable_by(k_s - 1, cmp_asc);
            sel_den[..k_s].sort_unstable_by(cmp_asc);
        } else {
            sel_den.sort_unstable_by(cmp_asc);
        }

        for &(idx, _) in unsel_den[..k_u].iter() {
            elite_unsel.push(idx);
        }

        for &(idx, _) in sel_den[..k_s].iter() {
            elite_sel.push(idx);
        }

        if k_raw > 0 {
            if in_unsel.len() < n {
                in_unsel.resize(n, false);
            }
            if in_sel.len() < n {
                in_sel.resize(n, false);
            }
            for &idx in elite_unsel.iter() {
                in_unsel[idx] = true;
            }
            for &idx in elite_sel.iter() {
                in_sel[idx] = true;
            }

            unsel_raw.clear();
            for i in 0..n {
                if state.selected[i] == 0 && !in_unsel[i] {
                    unsel_raw.push((i, state.delta[i]));
                }
            }
            let kru = k_raw.min(unsel_raw.len());
            if kru > 0 && kru < unsel_raw.len() {
                unsel_raw.select_nth_unstable_by(kru - 1, cmp_desc);
                unsel_raw[..kru].sort_unstable_by(cmp_desc);
            } else {
                unsel_raw.sort_unstable_by(cmp_desc);
            }
            for &(idx, _) in unsel_raw[..kru].iter() {
                elite_unsel.push(idx);
            }

            sel_raw.clear();
            for i in 0..n {
                if state.selected[i] == 1 && !in_sel[i] {
                    sel_raw.push((i, state.delta[i]));
                }
            }
            let krs = k_raw.min(sel_raw.len());
            if krs > 0 && krs < sel_raw.len() {
                sel_raw.select_nth_unstable_by(krs - 1, cmp_asc);
                sel_raw[..krs].sort_unstable_by(cmp_asc);
            } else {
                sel_raw.sort_unstable_by(cmp_asc);
            }
            for &(idx, _) in sel_raw[..krs].iter() {
                elite_sel.push(idx);
            }

            for &idx in &elite_unsel[..k_u] {
                in_unsel[idx] = false;
            }
            for &idx in &elite_sel[..k_s] {
                in_sel[idx] = false;
            }
        }
    });
}

fn build_boundary_windows(
    state: &FlatState,
    ch: &Challenge,
    k_density: usize,
    k_raw: usize,
) -> (Vec<usize>, Vec<usize>) {
    let mut sel = Vec::with_capacity(k_density + k_raw);
    let mut unsel = Vec::with_capacity(k_density + k_raw);
    build_boundary_windows_into(state, ch, k_density, k_raw, &mut sel, &mut unsel);
    (sel, unsel)
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
    let mut elite_sel = Vec::with_capacity(elite_k_12 + 16);
    let mut elite_unsel = Vec::with_capacity(elite_k_12 + 16);
    let mut top_sel_dummy = Vec::with_capacity(64);
    let mut top_unsel = Vec::with_capacity(64);

    loop {
        // Phase 1: Fast 0-out-1-in slack fill + full N x N 1-out-1-in climb
        fast_1to1_and_slack_climb(state, ch, q_flat, min_q);

        // Phase 2: 0-out-2-in and 0-out-3-in slack fill when slack >= 2
        if state.slack() >= 2 {
            let slack = state.slack();
            build_boundary_windows_into(state, ch, 50, 10, &mut top_sel_dummy, &mut top_unsel);
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
                let row_u = &q_flat[u_off..u_off + n];
                for j in (i + 1)..m {
                    let v = top_unsel[j];
                    let wuv = wu + (ch.weights[v] as i64);
                    if wuv > slack {
                        continue;
                    }
                    let g_uv = du + state.delta[v] + row_u[v];
                    if g_uv > best_gain {
                        best_gain = g_uv;
                        best_add_multi = Some((2, [u, v, 0]));
                    }
                    if wuv < slack && j < 28 {
                        let v_off = v * n;
                        let row_v = &q_flat[v_off..v_off + n];
                        for k in (j + 1)..m.min(28) {
                            let w_idx = top_unsel[k];
                            let w_tot = wuv + (ch.weights[w_idx] as i64);
                            if w_tot <= slack {
                                let g_uvw = g_uv
                                    + state.delta[w_idx]
                                    + row_u[w_idx]
                                    + row_v[w_idx];
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
        build_boundary_windows_into(state, ch, elite_k_12, 16, &mut elite_sel, &mut elite_unsel);
        let slack = state.slack();

        // 3a. Evaluate (1-out-2-in) and (2-out-1-in) on full `elite_sel` x `elite_unsel`
        // including optional coupled 1-item residual slack fill!
        let mut best_fill_by_weight: [(usize, i64); 11] = [(0, i64::MIN); 11];
        for &u in &elite_unsel {
            let wu = (ch.weights[u] as usize).min(10);
            let du = state.delta[u];
            if du > 0 {
                for cap_w in wu..=10 {
                    if du > best_fill_by_weight[cap_w].1 {
                        best_fill_by_weight[cap_w] = (u, du);
                    }
                }
            }
        }

        let mut best_move: Option<(u8, [usize; 3], u8, [usize; 3])> = None;
        let mut best_gain: i64 = 0;

        // (1-out, 2-in)
        for &a in &elite_sel {
            let w_a = ch.weights[a] as i64;
            let d_a = state.delta[a];
            let max_bc_w = slack + w_a;
            let a_off = a * n;
            let row_a = &q_flat[a_off..a_off + n];
            for i_b in 0..elite_unsel.len() {
                let b = elite_unsel[i_b];
                let w_b = ch.weights[b] as i64;
                if w_b >= max_bc_w {
                    continue;
                }
                let d_b = state.delta[b];
                let q_ab = row_a[b];
                let b_off = b * n;
                let row_b = &q_flat[b_off..b_off + n];
                let base_ab = d_b - d_a - q_ab;
                for i_c in (i_b + 1)..elite_unsel.len() {
                    let c = elite_unsel[i_c];
                    let w_bc = w_b + (ch.weights[c] as i64);
                    if w_bc <= max_bc_w {
                        let gain = base_ab + state.delta[c] - row_a[c] + row_b[c];
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
            let row_a1 = &q_flat[a1_off..a1_off + n];
            for i_a2 in (i_a1 + 1)..elite_sel.len() {
                let a2 = elite_sel[i_a2];
                let w_rem = w_a1 + (ch.weights[a2] as i64);
                let max_b_w = slack + w_rem;
                let a2_off = a2 * n;
                let row_a2 = &q_flat[a2_off..a2_off + n];
                let loss_a12 = d_a1 + state.delta[a2] - row_a1[a2];
                for &b in &elite_unsel {
                    let w_b = ch.weights[b] as i64;
                    if w_b <= max_b_w {
                        let gain = state.delta[b] - loss_a12 - row_a1[b] - row_a2[b];
                        if gain > best_gain {
                            best_gain = gain;
                            best_move = Some((2, [a1, a2, 0], 1, [b, 0, 0]));
                        }
                        // Check if residual slack allows coupling a second unselected item `f`
                        let rem_s = (max_b_w - w_b).min(10);
                        if rem_s >= 1 {
                            let (f, f_du) = best_fill_by_weight[rem_s as usize];
                            if f_du > i64::MIN && f != b {
                                let coupled_gain = gain + state.delta[f]
                                    + q_flat[b * n + f]
                                    - row_a1[f]
                                    - row_a2[f];
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
            let mut best_22: Option<([usize; 3], [usize; 3])> = None;

            POLISH_STAGE_BUFFERS.with(|buf_cell| {
                let mut bufs = buf_cell.borrow_mut();
                let PolishStageBuffers { rem_2, add_2, .. } = &mut *bufs;
                rem_2.clear();
                add_2.clear();

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
                    let mut best_22_gain: i64 = 0;

                    for r in rem_2.iter() {
                        if max_add_gain - r.loss - min_cross_22 <= best_22_gain {
                            break;
                        }
                        let max_w = slack + r.weight;
                        let r0 = r.items[0];
                        let r1 = r.items[1];
                        let r0_off = r0 * n;
                        let r1_off = r1 * n;
                        let row_r0 = &q_flat[r0_off..r0_off + n];
                        let row_r1 = &q_flat[r1_off..r1_off + n];
                        for a in add_2.iter() {
                            if a.gain_base - r.loss - min_cross_22 <= best_22_gain {
                                break;
                            }
                            if a.weight <= max_w {
                                let a0 = a.items[0];
                                let a1 = a.items[1];
                                let cross = row_r0[a0]
                                    + row_r0[a1]
                                    + row_r1[a0]
                                    + row_r1[a1];
                                let gain = a.gain_base - r.loss - cross;
                                if gain > best_22_gain {
                                    best_22_gain = gain;
                                    best_22 = Some((r.items, a.items));
                                }
                            }
                        }
                    }
                }
            });

            if let Some((r_arr, a_arr)) = best_22 {
                state.remove_item(ch, q_flat, r_arr[0]);
                state.remove_item(ch, q_flat, r_arr[1]);
                state.add_item(ch, q_flat, a_arr[0]);
                state.add_item(ch, q_flat, a_arr[1]);
                continue;
            }
        }

        // Phase 5: Higher-order branch-and-bound pruned coupled swaps:
        // (1,3), (3,1), (2,3), (3,2), and (3,3) on `elite_k_33`
        let s_33 = elite_k_33.min(elite_sel.len());
        let u_33 = elite_k_33.min(elite_unsel.len());
        if s_33 >= 3 && u_33 >= 3 {
            let mut best_ho: Option<(RemTuple, AddTuple)> = None;

            POLISH_STAGE_BUFFERS.with(|buf_cell| {
                let mut bufs = buf_cell.borrow_mut();
                let PolishStageBuffers { rem_1, rem_2, rem_3, add_1, add_2, add_3 } = &mut *bufs;

                rem_1.clear();
                rem_2.clear();
                rem_3.clear();
                add_1.clear();
                add_2.clear();
                add_3.clear();

                for i in 0..s_33 {
                    let r0 = elite_sel[i];
                    rem_1.push(RemTuple {
                        len: 1,
                        items: [r0, 0, 0],
                        weight: ch.weights[r0] as i64,
                        loss: state.delta[r0],
                    });
                }
                rem_1.sort_unstable_by_key(|r| r.loss);

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

                for i in 0..u_33 {
                    let a0 = elite_unsel[i];
                    add_1.push(AddTuple {
                        len: 1,
                        items: [a0, 0, 0],
                        weight: ch.weights[a0] as i64,
                        gain_base: state.delta[a0],
                    });
                }
                add_1.sort_unstable_by(|a, b| b.gain_base.cmp(&a.gain_base));

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

                let mut best_ho_gain: i64 = 0;

                let r3_len = rem_3.len().min(320);
                let a3_len = add_3.len().min(320);
                let pairs_to_check: [(&[RemTuple], &[AddTuple]); 5] = [
                    (rem_1.as_slice(), add_3.as_slice()),
                    (rem_3.as_slice(), add_1.as_slice()),
                    (rem_2.as_slice(), add_3.as_slice()),
                    (rem_3.as_slice(), add_2.as_slice()),
                    (&rem_3[..r3_len], &add_3[..a3_len]),
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
                        let r0_off = r.items[0] * n;
                        let row_r0 = &q_flat[r0_off..r0_off + n];
                        let row_r1 = if r_len > 1 {
                            let off = r.items[1] * n;
                            &q_flat[off..off + n]
                        } else {
                            &[]
                        };
                        let row_r2 = if r_len > 2 {
                            let off = r.items[2] * n;
                            &q_flat[off..off + n]
                        } else {
                            &[]
                        };
                        for a in adds {
                            if a.gain_base - r.loss - min_cross <= best_ho_gain {
                                break;
                            }
                            if a.weight <= max_w {
                                let a_len = a.len as usize;
                                let a0 = a.items[0];
                                let mut cross = row_r0[a0];
                                if a_len > 1 { cross += row_r0[a.items[1]]; }
                                if a_len > 2 { cross += row_r0[a.items[2]]; }
                                if r_len > 1 {
                                    cross += row_r1[a0];
                                    if a_len > 1 { cross += row_r1[a.items[1]]; }
                                    if a_len > 2 { cross += row_r1[a.items[2]]; }
                                }
                                if r_len > 2 {
                                    cross += row_r2[a0];
                                    if a_len > 1 { cross += row_r2[a.items[1]]; }
                                    if a_len > 2 { cross += row_r2[a.items[2]]; }
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
            });

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
    max_guides: usize,
) {
    let n = ch.num_items;
    if elites.is_empty() || max_guides == 0 {
        return;
    }

    // 1. Intersection + Greedy/DP Fill & Union + Worst-Density Drop across top distinct elites
    let limit = elites.len().min(max_guides.min(5));
    for i in 0..limit {
        for j in (i + 1)..limit {
            let e1 = &elites[i];
            let e2 = &elites[j];
            let mut only_e1: Vec<usize> = Vec::with_capacity(32);
            let mut hamming = 0usize;
            for k in 0..n {
                let s1 = e1.selected[k];
                let s2 = e2.selected[k];
                if s1 != s2 {
                    hamming += 1;
                    if s1 == 1 {
                        only_e1.push(k);
                    }
                }
            }
            if hamming < 4 {
                continue;
            }

            // Intersection + greedy density fill (clone e1 and remove e1 \ e2: ~15 removals vs ~450 additions)
            let mut inter_st = e1.clone();
            for &k in &only_e1 {
                inter_st.remove_item(ch, q_flat, k);
            }
            while inter_st.slack() > 0 {
                let slack = inter_st.slack();
                let mut best_item: Option<usize> = None;
                let mut best_key: i64 = 0;
                for k in 0..n {
                    if inter_st.selected[k] == 0
                        && (ch.weights[k] as i64) <= slack
                        && inter_st.delta[k] > 0
                    {
                        let w = (ch.weights[k] as usize).clamp(1, 10);
                        let key = inter_st.delta[k] * LDIV[w];
                        if key > best_key {
                            best_key = key;
                            best_item = Some(k);
                        }
                    }
                }
                if let Some(k) = best_item {
                    inter_st.add_item(ch, q_flat, k);
                } else {
                    break;
                }
            }
            integer_core_dp_refine(&mut inter_st, ch, q_flat, min_q, 35);
            fast_1to1_and_slack_climb(&mut inter_st, ch, q_flat, min_q);
            if inter_st.total_value > best_state.total_value - 450 {
                if n > 1000 {
                    polish_stage_b2_hybrid(&mut inter_st, ch, q_flat, min_q, 48, 38, 18, 30);
                } else {
                    polish_stage_b2_hybrid(&mut inter_st, ch, q_flat, min_q, 60, 48, 22, 40);
                }
                if inter_st.total_value > best_state.total_value {
                    *best_state = inter_st;
                }
            }
        }
    }

    // 2. Bidirectional Path Relinking between `best_state` and each distinct elite
    let base_snapshot = best_state.clone();
    for guide in elites.iter().take(max_guides) {
        // Velocity skip: elites[0] is the base snapshot itself, so relinking it to the base is a no-op.
        if guide.selected == base_snapshot.selected {
            continue;
        }
        // Single pass across 0..n to partition differences between base_snapshot and guide
        let mut diff_base_unsel: Vec<usize> = Vec::with_capacity(64); // 0 in base, 1 in guide
        let mut diff_base_sel: Vec<usize> = Vec::with_capacity(64);   // 1 in base, 0 in guide
        for k in 0..n {
            let b = base_snapshot.selected[k];
            let g = guide.selected[k];
            if b != g {
                if b == 0 {
                    diff_base_unsel.push(k);
                } else {
                    diff_base_sel.push(k);
                }
            }
        }

        let total_diff = diff_base_unsel.len() + diff_base_sel.len();
        if total_diff < 4 {
            continue;
        }

        let directions = [
            (&base_snapshot, diff_base_unsel.clone(), diff_base_sel.clone()),
            (guide, diff_base_sel, diff_base_unsel),
        ];
        for (src, mut to_add, mut to_rem) in directions {
            let mut cur = src.clone();
            let mut best_along: Option<FlatState> = None;
            let mut best_along_val = best_state.total_value.saturating_sub(250);
            let step_check = (total_diff / 4).max(3);
            let mut steps = 0usize;

            while !to_add.is_empty() || !to_rem.is_empty() {
                let slack = cur.slack();
                let mut best_act: Option<(bool, usize, i64)> = None;
                let mut best_act_val = i64::MIN;

                for (idx, &item) in to_add.iter().enumerate() {
                    if (ch.weights[item] as i64) <= slack {
                        let d = cur.delta[item];
                        if d > best_act_val {
                            best_act_val = d;
                            best_act = Some((true, idx, d));
                        }
                    }
                }
                for (idx, &item) in to_rem.iter().enumerate() {
                    let d = -cur.delta[item];
                    if d > best_act_val {
                        best_act_val = d;
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

                if steps > 2 && steps + 2 < total_diff && cur.total_weight <= cur.capacity && cur.total_value > best_along_val {
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
                if n > 1000 {
                    polish_stage_b2_hybrid(&mut cand, ch, q_flat, min_q, 48, 38, 18, 30);
                } else {
                    polish_stage_b2_hybrid(&mut cand, ch, q_flat, min_q, 64, 50, 22, 42);
                }
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
    let mut tabu_until = vec![0usize; n];

    for burst in 0..num_bursts {
        let mut cur = best_state.clone();
        tabu_until.fill(0);
        let mut elite_unsel_prev: Vec<usize> = Vec::new();

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
            elite_unsel_prev = elite_unsel;
        }

        let mut best_burst_state: Option<FlatState> = None;

        for step in 1..=steps_per_burst {
            while cur.slack() > 0 {
                let slack = cur.slack();
                let mut best_k: Option<usize> = None;
                let mut best_g: i64 = 0;
                for &k in &elite_unsel_prev {
                    if cur.selected[k] == 0
                        && tabu_until[k] < step
                        && (ch.weights[k] as i64) <= slack
                        && cur.delta[k] > best_g
                    {
                        best_g = cur.delta[k];
                        best_k = Some(k);
                    }
                }
                let has_tail_pos = elite_unsel_prev.len() >= 45 && cur.delta[*elite_unsel_prev.last().unwrap()] > 0;
                if best_k.is_none() && (elite_unsel_prev.is_empty() || has_tail_pos) {
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
            elite_unsel_prev = elite_unsel;
            let slack = cur.slack();
            let mut best_tabu_swap: Option<(usize, usize, i64)> = None;

            for &a in &elite_sel {
                let w_a = ch.weights[a] as i64;
                let d_a = cur.delta[a];
                let a_off = a * n;
                let max_wb = w_a + slack;
                for &b in &elite_unsel_prev {
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
            if n > 1000 {
                polish_stage_b2_hybrid(&mut st, ch, q_flat, min_q, 48, 38, 18, 30);
            } else {
                polish_stage_b2_hybrid(&mut st, ch, q_flat, min_q, 65, 52, 24, 42);
            }
            if st.total_value > best_state.total_value {
                *best_state = st;
            }
        } else {
            if n > 1000 {
                polish_stage_b2_hybrid(&mut cur, ch, q_flat, min_q, 48, 38, 18, 30);
            } else {
                polish_stage_b2_hybrid(&mut cur, ch, q_flat, min_q, 55, 44, 20, 38);
            }
            if cur.total_value > best_state.total_value {
                *best_state = cur;
            }
        }
    }
}

/// Iterated Local Search (ILS) Perturbation Operator for Tracks 42 & 43:
/// Ejects weakly coupled marginal items and inserts high-density unselected items.
pub fn perturb_solution(
    state: &mut FlatState,
    ch: &Challenge,
    q_flat: &[i64],
    k_eject: usize,
    rng: &mut Xorshift64,
    elite_sel: &mut Vec<usize>,
    elite_unsel: &mut Vec<usize>,
) {
    build_boundary_windows_into(state, ch, 40, 10, elite_sel, elite_unsel);
    if elite_sel.is_empty() {
        return;
    }
    let k_drop = k_eject.min(elite_sel.len());
    let pool_len = elite_sel.len().min(16 + k_drop * 2);
    let mut dropped = 0;
    let mut attempts = 0;
    while dropped < k_drop && attempts < 32 {
        attempts += 1;
        let r1 = (rng.next_u64() as usize) % pool_len;
        let r2 = (rng.next_u64() as usize) % pool_len;
        let pick = r1.min(r2);
        let rm = elite_sel[pick];
        if state.selected[rm] == 1 {
            state.remove_item(ch, q_flat, rm);
            dropped += 1;
        }
    }
    // Inject proportional items from elite_unsel that fit into slack via tournament selection
    let unsel_len = elite_unsel.len();
    if unsel_len > 0 {
        let max_inject = 1 + (rng.next_u64() as usize % 3);
        let mut injected = 0;
        let mut attempts = 0;
        let pool_u = unsel_len.min(16);
        while injected < max_inject && attempts < 24 && state.slack() > 0 {
            attempts += 1;
            let r1 = (rng.next_u64() as usize) % pool_u;
            let r2 = (rng.next_u64() as usize) % pool_u;
            let pick = r1.min(r2);
            let u = elite_unsel[pick];
            if state.selected[u] == 0 && (ch.weights[u] as i64) <= state.slack() && state.delta[u] >= -25 {
                state.add_item(ch, q_flat, u);
                injected += 1;
            }
        }
    }
}

/// Remove an entire positive-interaction module, rather than isolated boundary items.
fn amputate_synergy_clique(
    state: &mut FlatState, ch: &Challenge, q_flat: &[i64], k: usize,
    rng: &mut Xorshift64,
) -> Vec<usize> {
    let mut selected: Vec<usize> = (0..state.n).filter(|&i| state.selected[i] == 1).collect();
    if selected.is_empty() || k == 0 { return Vec::new(); }
    selected.sort_unstable_by(|&a, &b| {
        (state.delta[a] * ch.weights[b] as i64)
            .cmp(&(state.delta[b] * ch.weights[a] as i64)).then(a.cmp(&b))
    });
    let tier = (selected.len() * 2 / 3).max(1);
    let seed = selected[rng.next_u64() as usize % tier];
    let row = &q_flat[seed * state.n..(seed + 1) * state.n];
    selected.retain(|&j| j != seed && row[j] > 0);
    selected.sort_unstable_by(|&a, &b| row[b].cmp(&row[a]).then(a.cmp(&b)));
    selected.truncate(k.saturating_sub(1));
    selected.insert(0, seed);
    for &j in &selected { state.remove_item(ch, q_flat, j); }
    selected
}

fn perturb_synergy_clique(
    state: &mut FlatState, ch: &Challenge, q_flat: &[i64], k: usize,
    rng: &mut Xorshift64,
) {
    let removed = amputate_synergy_clique(state, ch, q_flat, k, rng);
    if removed.is_empty() { return; }
    // Rebuild from the surviving core. Keep the amputated module out of this
    // construction only; the subsequent unrestricted climb may recover it.
    let mut order: Vec<usize> = (0..state.n)
        .filter(|&i| state.selected[i] == 0 && !removed.contains(&i)).collect();
    order.sort_unstable_by(|&a, &b| {
        (state.delta[b] * ch.weights[a] as i64)
            .cmp(&(state.delta[a] * ch.weights[b] as i64)).then(a.cmp(&b))
    });
    let mut attempts = 0;
    while !order.is_empty() && attempts < k * 4 {
        attempts += 1;
        let pool = order.len().min(24);
        let a = rng.next_u64() as usize % pool;
        let b = rng.next_u64() as usize % pool;
        let pick = if state.delta[order[a]] * ch.weights[order[b]] as i64
            >= state.delta[order[b]] * ch.weights[order[a]] as i64 { a } else { b };
        let i = order.remove(pick);
        if ch.weights[i] as i64 <= state.slack() && state.delta[i] > 0 {
            state.add_item(ch, q_flat, i);
        }
    }
}

thread_local! {
    static ILS_ELITE_BUFFERS: std::cell::RefCell<(Vec<usize>, Vec<usize>)> = std::cell::RefCell::new((
        Vec::with_capacity(128),
        Vec::with_capacity(128),
    ));
}

/// Deep Iterated Local Search (ILS) Refinement Loop on Promising Basins:
/// Alternates between perturbation, fast 1-1/slack climb, Zobrist diversification,
/// and bounded Stage B2 polish with strict zero-loss preservation.
pub fn iterated_local_search_refine(
    best_state: &mut FlatState,
    ch: &Challenge,
    q_flat: &[i64],
    min_q: i64,
    seed_u64: u64,
    max_rounds: usize,
    is_t42: bool,
) -> bool {
    if max_rounds == 0 {
        return false;
    }
    let mut rng = Xorshift64::new(seed_u64 ^ 0x55AA_AA55_33CC_CC33);
    let mut cur = best_state.clone();
    let mut improved = false;

    // Track visited basins via Zobrist hash to prevent cycling
    let mut visited_hashes = [0u64; 32];
    let mut hash_cnt = 0;
    visited_hashes[0] = best_state.zobrist_hash();
    hash_cnt += 1;

    ILS_ELITE_BUFFERS.with(|bufs| {
        let (elite_sel, elite_unsel) = &mut *bufs.borrow_mut();
        let total_rounds = if is_t42 { max_rounds } else { max_rounds.saturating_add(1) };
        for round in 0..total_rounds {
            let clique = !is_t42 && round >= max_rounds;
            if round == max_rounds { cur = best_state.clone(); }
            // 1. Perturb cur
            let k_eject = if is_t42 { 4 + (round % 5) } else { 7 + (round % 4) * 2 };
            if clique {
                perturb_synergy_clique(&mut cur, ch, q_flat, k_eject * 2, &mut rng);
            } else {
                perturb_solution(&mut cur, ch, q_flat, k_eject, &mut rng, elite_sel, elite_unsel);
            }

        // 2. Fast 1-1 and slack climb + conditional core-DP slack resolution
        fast_1to1_and_slack_climb(&mut cur, ch, q_flat, min_q);
        if cur.slack() > 0 {
            integer_core_dp_refine(&mut cur, ch, q_flat, min_q, if is_t42 { 25 } else { 20 });
            fast_1to1_and_slack_climb(&mut cur, ch, q_flat, min_q);
        }

        // 3. Diversification check via Zobrist hash
        let mut h = cur.zobrist_hash();
        if visited_hashes[..hash_cnt].contains(&h) {
            // Re-perturb to escape cycling basin
            if clique {
                perturb_synergy_clique(&mut cur, ch, q_flat, k_eject * 3, &mut rng);
            } else {
                perturb_solution(&mut cur, ch, q_flat, k_eject + 3, &mut rng, elite_sel, elite_unsel);
            }
            fast_1to1_and_slack_climb(&mut cur, ch, q_flat, min_q);
            if cur.slack() > 0 {
                integer_core_dp_refine(&mut cur, ch, q_flat, min_q, if is_t42 { 25 } else { 20 });
                fast_1to1_and_slack_climb(&mut cur, ch, q_flat, min_q);
            }
            h = cur.zobrist_hash();
        }
        if hash_cnt < visited_hashes.len() {
            visited_hashes[hash_cnt] = h;
            hash_cnt += 1;
        }

        // 4. Selective Stage B2 polish if candidate is within striking distance
        let strike_dist = if is_t42 { 350 } else { 500 };
        if cur.total_value + strike_dist > best_state.total_value {
            let (k12, k22, dp) = if is_t42 { (32, 24, 20) } else { (24, 18, 16) };
            polish_stage_b2_hybrid(&mut cur, ch, q_flat, min_q, k12, k22, 0, dp);
            fast_1to1_and_slack_climb(&mut cur, ch, q_flat, min_q);
        }

        // 5. Zero-loss ratchet update, peak intensification & adaptive basin anchor
        let drift_limit = if is_t42 { 500 } else { 750 };
        if cur.total_value > best_state.total_value {
            // Intensify newly discovered peak
            let (k12_peak, k22_peak, dp_peak) = if is_t42 { (48, 36, 28) } else { (36, 28, 24) };
            polish_stage_b2_hybrid(&mut cur, ch, q_flat, min_q, k12_peak, k22_peak, 0, dp_peak);
            integer_core_dp_refine(&mut cur, ch, q_flat, min_q, if is_t42 { 35 } else { 28 });
            fast_1to1_and_slack_climb(&mut cur, ch, q_flat, min_q);
            *best_state = cur.clone();
            improved = true;
        } else if cur.total_value + drift_limit < best_state.total_value {
            // Re-anchor to incumbent if energy drift is too large
            cur = best_state.clone();
        }
    }
    });

    improved
}

/// Warm-start basins of `stage_b2_hybrid_v10` (Stage A + Core-DP + Ruin-and-Recreate kicks + Stage B2 polish),
/// without the `supreme_core` preamble. Takes the first `num_starts` of the 4 warm configs and returns the
/// distinct elites by zobrist hash (unsorted).
pub fn solve_multi_warm_b2_elites(
    ch: &Challenge,
    q_flat: &[i64],
    sum_q: &[i64],
    min_q: i64,
    num_starts: usize,
) -> Vec<FlatState> {
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

    // Multi-warm-start Stage A (`degree_weight_num = 1164`, `power_w` in `[0.80, 0.72, 0.90, 1.00]`)
    //    + Core-Window DP + Unbraked Stage B/C Tabu & Kicks + Stage B2 Hybrid Polish
    let warm_configs: [(i64, f64, usize); 4] = [
        (1164, 0.80, 10),
        (1164, 0.72, 6),
        (1164, 0.90, 6),
        (1024, 1.00, 6),
    ];

    let mut scores: Vec<(usize, f64)> = Vec::with_capacity(n);
    let mut sel_indices: Vec<usize> = Vec::with_capacity(n);
    let mut candidates: Vec<usize> = Vec::with_capacity(n);

    for (cfg_idx, &(deg_w, power_w, max_kicks)) in warm_configs.iter().take(num_starts).enumerate() {
        let max_kicks = if n > 1000 { max_kicks.min(2) } else { max_kicks };
        scores.clear();
        for i in 0..n {
            let num = ((ch.values[i] as i64) * 1024 + deg_w * sum_q[i]) as f64;
            let den = (ch.weights[i] as f64).powf(power_w).max(1.0);
            scores.push((i, num / den));
        }
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
            sel_indices.clear();
            for i in 0..n {
                if cur.selected[i] == 1 {
                    sel_indices.push(i);
                }
            }
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
                candidates.clear();
                for i in 0..n {
                    if cur.selected[i] == 0 {
                        candidates.push(i);
                    }
                }
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
    elites
}

/// Stage B2 polish arguments `(elite_k_12, elite_k_22, elite_k_33, core_half_dp)`.
type PolishArgs = (usize, usize, usize, usize);

/// Track entry point; the signature shared by track1..track5 and track_t39.
type TrackFn = fn(&Challenge, &dyn Fn(&Solution) -> Result<()>, &Option<Map<String, Value>>) -> Result<()>;

/// Runs a track and returns the last solution it saved. Tracks save internally, so the outer
/// `save_solution` is never passed through (the harness keeps the last save).
fn capture_track(track: TrackFn, ch: &Challenge, hp: &Option<Map<String, Value>>) -> Result<Option<Solution>> {
    let captured: RefCell<Option<Solution>> = RefCell::new(None);
    track(ch, &|sol: &Solution| {
        *captured.borrow_mut() = Some(sol.clone());
        Ok(())
    }, hp)?;
    Ok(captured.into_inner())
}

/// Track result as a FlatState, optional candidate: None when the track saved nothing or it is infeasible.
fn track_flat(track: TrackFn, ch: &Challenge, q_flat: &[i64], hp: &Option<Map<String, Value>>) -> Result<Option<FlatState>> {
    Ok(capture_track(track, ch, hp)?.and_then(|sol| flat_if_feasible(ch, q_flat, &sol)))
}

/// packrun's dispatch, mirrored exactly (`packrun/mod.rs` `solve_challenge`): other shapes are an error.
fn packrun_track(n: usize, budget_pct: u32) -> Result<TrackFn> {
    match (n, budget_pct) {
        (1000, b) if b <= 7 => Ok(track1::solve as TrackFn),
        (1000, b) if b <= 17 => Ok(track2::solve as TrackFn),
        (1000, _) => Ok(track3::solve as TrackFn),
        (5000, b) if b <= 17 => Ok(track4::solve as TrackFn),
        (5000, _) => Ok(track5::solve as TrackFn),
        _ => Err(anyhow!("unknown track config (num_items={}, budget_pct={})", n, budget_pct)),
    }
}

/// packrun's budget_pct, including its zero-weight guard.
fn budget_pct(ch: &Challenge) -> u32 {
    let sum_w: u64 = ch.weights.iter().map(|&w| w as u64).sum();
    if sum_w > 0 {
        ((ch.max_weight as u64) * 100 / sum_w) as u32
    } else {
        10
    }
}

fn hp_number(key: &str, v: u64) -> Option<Map<String, Value>> {
    let mut m = Map::new();
    m.insert(key.to_string(), Value::Number(serde_json::Number::from(v as i64)));
    Some(m)
}

/// A solution as a FlatState; None when it exceeds capacity, so it cannot enter the elite pool.
fn flat_if_feasible(ch: &Challenge, q_flat: &[i64], sol: &Solution) -> Option<FlatState> {
    let st = FlatState::from_items(ch, q_flat, &sol.items);
    if st.total_weight <= st.capacity {
        Some(st)
    } else {
        None
    }
}

/// Adds a candidate unless an identical basin (zobrist hash) is already in the pool.
fn push_distinct(pool: &mut Vec<FlatState>, st: FlatState) {
    let h = st.zobrist_hash();
    if !pool.iter().any(|e| e.zobrist_hash() == h) {
        pool.push(st);
    }
}

/// Polishes the `top_k` best distinct candidates and adds them to the elite pool.
/// Linearize around the preliminary best core, with a separately weighted
/// positive potential outside it. No full-row-sum lambda sweep is reused.
fn restricted_core_seeds(
    ch: &Challenge, q_flat: &[i64], min_q: i64, cands: &[FlatState],
) -> Vec<FlatState> {
    let Some(core) = cands.iter().max_by_key(|s| s.total_value) else { return Vec::new(); };
    let n = ch.num_items;
    let mut outside = vec![0i64; n];
    for i in 0..n {
        outside[i] = q_flat[i * n..(i + 1) * n].iter().enumerate()
            .filter(|&(j, _)| j != i && core.selected[j] == 0)
            .map(|(_, &q)| q.max(0)).sum();
    }
    let mut starts = Vec::new();
    let mut climbed: Vec<Vec<u8>> = Vec::new();
    // Two orthogonal potentials, scaled by twenty: (1,0), (.5,.1).
    for (alpha, beta) in [(20, 0), (10, 2)] {
        let potential: Vec<i64> = (0..n).map(|i| {
            let v = ch.values[i] as i64;
            20 * v + alpha * (core.delta[i] - v) + beta * outside[i]
        }).collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| {
            (potential[b] * ch.weights[a] as i64)
                .cmp(&(potential[a] * ch.weights[b] as i64)).then(a.cmp(&b))
        });
        let mut st = FlatState::new_empty(ch);
        for i in order {
            if potential[i] > 0 && ch.weights[i] as i64 <= st.slack() {
                st.add_item(ch, q_flat, i);
            }
        }
        fast_1to1_and_slack_climb(&mut st, ch, q_flat, min_q);
        if cands.iter().any(|s| s.selected == st.selected)
            || climbed.iter().any(|bits| *bits == st.selected) { continue; }
        climbed.push(st.selected.clone());
        polish_stage_b2_hybrid(&mut st, ch, q_flat, min_q, 16, 12, 0, 10);
        push_distinct(&mut starts, st);
    }
    starts
}

#[cfg(test)]
mod basin_tests {
    use super::*;

    fn challenge(values: Vec<u32>, capacity: u32, edges: &[(usize, usize, i32)]) -> Challenge {
        let n = values.len();
        let mut interaction_values = vec![vec![0; n]; n];
        for &(a, b, q) in edges {
            interaction_values[a][b] = q;
            interaction_values[b][a] = q;
        }
        Challenge { seed: [0; 32], num_items: n, weights: vec![1; n], values,
            interaction_values, max_weight: capacity }
    }

    fn assert_consistent(ch: &Challenge, q: &[i64], st: &FlatState) {
        let rebuilt = FlatState::from_items(ch, q, &st.to_solution().items);
        assert_eq!(st.total_value, rebuilt.total_value);
        assert_eq!(st.total_weight, rebuilt.total_weight);
        assert_eq!(st.delta, rebuilt.delta);
        assert!(st.total_weight <= ch.max_weight as i64);
    }

    #[test]
    fn clique_ejection_removes_positive_module_and_preserves_state() {
        let ch = challenge(vec![1, 1, 1, 1000, 2000], 5,
            &[(0, 1, 100), (0, 2, 90), (1, 2, 80), (0, 4, -5), (1, 4, -5), (2, 4, -5)]);
        let (q, _, _) = build_q_flat(&ch);
        for seed in 1..=12 {
            let mut st = FlatState::from_items(&ch, &q, &[0, 1, 2, 3, 4]);
            let removed = amputate_synergy_clique(&mut st, &ch, &q, 3, &mut Xorshift64::new(seed));
            assert_eq!(removed.len(), 3);
            assert!(removed.iter().all(|&i| i < 3));
            assert_eq!(&st.selected[3..], &[1, 1]);
            assert_consistent(&ch, &q, &st);
        }
    }

    #[test]
    fn restricted_core_finds_disjoint_synergy_pair() {
        let ch = challenge(vec![10, 10, 5, 5], 2, &[(0, 1, 1), (2, 3, 100)]);
        let (q, _, min_q) = build_q_flat(&ch);
        let incumbent = FlatState::from_items(&ch, &q, &[0, 1]);
        let starts = restricted_core_seeds(&ch, &q, min_q, &[incumbent]);
        assert_eq!(starts.iter().map(|s| s.total_value).max(), Some(110));
        for st in &starts { assert_consistent(&ch, &q, st); }
        for (i, a) in starts.iter().enumerate() {
            assert!(starts[..i].iter().all(|b| a.selected != b.selected));
        }
        assert!(restricted_core_seeds(&ch, &q, min_q, &[]).is_empty());
    }

    #[test]
    fn clique_perturbation_handles_empty_and_exhausted_selected_set() {
        let ch = challenge(vec![1, 1, 1], 2, &[(0, 1, 4)]);
        let (q, _, _) = build_q_flat(&ch);
        let mut rng = Xorshift64::new(7);
        let mut st = FlatState::new_empty(&ch);
        perturb_synergy_clique(&mut st, &ch, &q, 20, &mut rng);
        assert_consistent(&ch, &q, &st);
        st = FlatState::from_items(&ch, &q, &[0, 1]);
        perturb_synergy_clique(&mut st, &ch, &q, 20, &mut rng);
        assert_consistent(&ch, &q, &st);
    }
}

fn polish_candidates(cands: Vec<FlatState>, elites: &mut Vec<FlatState>, top_k: usize, polish: &dyn Fn(&mut FlatState)) {
    let mut distinct: Vec<FlatState> = Vec::new();
    for st in cands {
        push_distinct(&mut distinct, st);
    }
    distinct.sort_unstable_by(|a, b| b.total_value.cmp(&a.total_value));
    for mut st in distinct.into_iter().take(top_k) {
        polish(&mut st);
        push_distinct(elites, st);
    }
}

/// Runs one stage on `best` and keeps the result only if `total_value` did not drop.
fn keep_if_not_worse(best: &mut FlatState, stage: impl FnOnce(&mut FlatState)) {
    let before = best.clone();
    stage(best);
    if best.total_value < before.total_value {
        *best = before;
    }
}

/// Family 3 (`lookahead_relink`): 2-Step Lookahead Path Relinking + Inflection Probe.
/// When `lookahead_mode >= 1`:
/// - Also performs Union + Worst-Density Drop recombination across elite pairs (symmetric to Intersection + Fill).
/// - Along each bidirectional path relinking trajectory `(src -> dst)` and `(dst -> src)`, scores each candidate
///   move by a 2-step lookahead continuation:
///   - Adding `i in to_add` freeing `w_i <= slack`: single-step gain `delta[i]`, plus best follow-up `j in to_add`
///     if `w_i + w_j <= slack` (`delta[j] + Q[i,j]`).
///   - Removing `r in to_rem`: single-step loss `-delta[r]`, coupled with best immediate feasible `a in to_add`
///     that fits in `slack + w_r` (`delta[a] - delta[r] - Q[r,a]`).
/// - When `quad_bnb_k > 0`, runs `exact_quad_bnb_refine` at trajectory inflection points and on `best_along`.
pub fn recombine_and_relink_elites_lookahead(
    best_state: &mut FlatState,
    elites: &[FlatState],
    ch: &Challenge,
    q_flat: &[i64],
    min_q: i64,
    max_guides: usize,
    lookahead_mode: usize,
    quad_bnb_k: usize,
    bucket_compound_k: usize,
) {
    // Always run standard path relinking first so `lookahead_relink >= 1` is strictly monotonic over `lookahead_relink == 0`.
    recombine_and_relink_elites(best_state, elites, ch, q_flat, min_q, max_guides);
    if lookahead_mode == 0 && quad_bnb_k == 0 && bucket_compound_k == 0 {
        return;
    }
    let n = ch.num_items;
    if elites.is_empty() || max_guides == 0 {
        return;
    }

    let extra_polish = |st: &mut FlatState| {
        if bucket_compound_k > 0 {
            while bucket_compound_22_refine(st, ch, q_flat, min_q, bucket_compound_k) {}
        }
        if quad_bnb_k > 0 {
            let _ = exact_quad_bnb_refine(st, ch, q_flat, min_q, quad_bnb_k);
        }
    };

    // 1. Intersection + Greedy/DP Fill & Union + Worst-Density Drop across top distinct elites
    let limit = elites.len().min(max_guides.min(5));
    for i in 0..limit {
        for j in (i + 1)..limit {
            let e1 = &elites[i];
            let e2 = &elites[j];
            let mut only_e1: Vec<usize> = Vec::with_capacity(64);
            let mut only_e2: Vec<usize> = Vec::with_capacity(64);
            for k in 0..n {
                let s1 = e1.selected[k];
                let s2 = e2.selected[k];
                if s1 != s2 {
                    if s1 == 1 {
                        only_e1.push(k);
                    } else {
                        only_e2.push(k);
                    }
                }
            }
            let hamming = only_e1.len() + only_e2.len();
            if hamming < 4 {
                continue;
            }

            // 1a. Intersection + greedy density fill (clone e1 and remove only_e1: ~15 removals vs ~450 additions)
            let mut inter_st = e1.clone();
            for &k in &only_e1 {
                inter_st.remove_item(ch, q_flat, k);
            }
            while inter_st.slack() > 0 {
                let slack = inter_st.slack();
                let mut best_item: Option<usize> = None;
                let mut best_key: i64 = 0;
                for k in 0..n {
                    if inter_st.selected[k] == 0
                        && (ch.weights[k] as i64) <= slack
                        && inter_st.delta[k] > 0
                    {
                        let w = (ch.weights[k] as usize).clamp(1, 10);
                        let key = inter_st.delta[k] * LDIV[w];
                        if key > best_key {
                            best_key = key;
                            best_item = Some(k);
                        }
                    }
                }
                if let Some(k) = best_item {
                    inter_st.add_item(ch, q_flat, k);
                } else {
                    break;
                }
            }
            integer_core_dp_refine(&mut inter_st, ch, q_flat, min_q, 35);
            fast_1to1_and_slack_climb(&mut inter_st, ch, q_flat, min_q);
            extra_polish(&mut inter_st);
            if inter_st.total_value > best_state.total_value - 450 {
                polish_stage_b2_hybrid(&mut inter_st, ch, q_flat, min_q, 60, 48, 22, 40);
                extra_polish(&mut inter_st);
                if inter_st.total_value > best_state.total_value {
                    *best_state = inter_st;
                }
            }

            // 1b. Union-inspired Symmetric-Difference Drop-and-Fill when `lookahead_mode >= 1`
            if lookahead_mode >= 1 {
                let mut union_st = e1.clone();
                // Try injecting highest-synergy items from `e2 \ e1` by dropping worst-density items from `e1 \ e2`
                only_e1.sort_unstable_by(|&a, &b| {
                    let da = (union_st.delta[a] as f64) / (ch.weights[a] as f64);
                    let db = (union_st.delta[b] as f64) / (ch.weights[b] as f64);
                    da.partial_cmp(&db).unwrap_or(Ordering::Equal).then_with(|| a.cmp(&b))
                });
                let drop_half = (only_e1.len() / 2).clamp(2, 16);
                for &rm in only_e1.iter().take(drop_half) {
                    if union_st.selected[rm] == 1 {
                        union_st.remove_item(ch, q_flat, rm);
                    }
                }
                loop {
                    let slack = union_st.slack();
                    let mut best_u: Option<(usize, f64)> = None;
                    for &u in &only_e2 {
                        if union_st.selected[u] == 0 && (ch.weights[u] as i64) <= slack && union_st.delta[u] > 0 {
                            let r = (union_st.delta[u] as f64) / (ch.weights[u] as f64);
                            if best_u.map_or(true, |(_, br)| r > br) {
                                best_u = Some((u, r));
                            }
                        }
                    }
                    if let Some((u, _)) = best_u {
                        union_st.add_item(ch, q_flat, u);
                    } else {
                        break;
                    }
                }
                integer_core_dp_refine(&mut union_st, ch, q_flat, min_q, 35);
                fast_1to1_and_slack_climb(&mut union_st, ch, q_flat, min_q);
                extra_polish(&mut union_st);
                if union_st.total_value > best_state.total_value - 450 {
                    polish_stage_b2_hybrid(&mut union_st, ch, q_flat, min_q, 60, 48, 22, 40);
                    extra_polish(&mut union_st);
                    if union_st.total_value > best_state.total_value {
                        *best_state = union_st;
                    }
                }
            }
        }
    }

    // 2. Bidirectional Path Relinking with 2-Step Lookahead Continuation Scoring
    let base_snapshot = best_state.clone();
    for guide in elites.iter().take(max_guides) {
        // Velocity skip: elites[0] is the base snapshot itself, so relinking it to the base is a no-op.
        if guide.selected == base_snapshot.selected {
            continue;
        }
        // Single pass across 0..n to partition differences between base_snapshot and guide
        let mut diff_base_unsel: Vec<usize> = Vec::with_capacity(64); // 0 in base, 1 in guide
        let mut diff_base_sel: Vec<usize> = Vec::with_capacity(64);   // 1 in base, 0 in guide
        for k in 0..n {
            let b = base_snapshot.selected[k];
            let g = guide.selected[k];
            if b != g {
                if b == 0 {
                    diff_base_unsel.push(k);
                } else {
                    diff_base_sel.push(k);
                }
            }
        }

        let total_diff = diff_base_unsel.len() + diff_base_sel.len();
        if total_diff < 4 {
            continue;
        }

        let directions = [
            (&base_snapshot, diff_base_unsel.clone(), diff_base_sel.clone()),
            (guide, diff_base_sel, diff_base_unsel),
        ];
        for (src, mut to_add, mut to_rem) in directions {

            let mut cur = src.clone();
            let mut best_along: Option<FlatState> = None;
            let mut best_along_val = best_state.total_value.saturating_sub(250);
            let step_check = (total_diff / 4).max(3);
            let mut steps = 0usize;

            while !to_add.is_empty() || !to_rem.is_empty() {
                let slack = cur.slack();

                if lookahead_mode >= 1 {
                    // 2-step lookahead move evaluation:
                    // Option A: direct 1-out-1-in or 1-out-2-in coupled move along `(to_rem, to_add)`
                    // or single `to_add` / `to_rem` scored by 2-step continuation.
                    let mut best_act: Option<(u8, usize, usize, i64)> = None;
                    // action code: 1 = add(idx), 2 = rem(idx), 3 = swap(rem_idx, add_idx)
                    for (a_idx, &item) in to_add.iter().enumerate() {
                        let w_a = ch.weights[item] as i64;
                        if w_a <= slack {
                            let mut score = cur.delta[item];
                            let rem_slack = slack - w_a;
                            if rem_slack > 0 {
                                let a_off = item * n;
                                let mut best_next = 0i64;
                                for (b_idx, &next_u) in to_add.iter().enumerate() {
                                    if b_idx != a_idx && (ch.weights[next_u] as i64) <= rem_slack {
                                        let g2 = cur.delta[next_u] + q_flat[a_off + next_u];
                                        if g2 > best_next {
                                            best_next = g2;
                                        }
                                    }
                                }
                                score += best_next / 2;
                            }
                            if best_act.map_or(true, |(_, _, _, bs)| score > bs) {
                                best_act = Some((1, a_idx, 0, score));
                            }
                        }
                    }

                    for (r_idx, &rm_item) in to_rem.iter().enumerate() {
                        let w_r = ch.weights[rm_item] as i64;
                        let d_r = cur.delta[rm_item];
                        let r_off = rm_item * n;
                        let avail = slack + w_r;
                        let mut best_coupled: Option<(usize, i64)> = None;
                        for (a_idx, &add_item) in to_add.iter().enumerate() {
                            if (ch.weights[add_item] as i64) <= avail {
                                let pair_gain = cur.delta[add_item] - d_r - q_flat[r_off + add_item];
                                if best_coupled.map_or(true, |(_, bg)| pair_gain > bg) {
                                    best_coupled = Some((a_idx, pair_gain));
                                }
                            }
                        }
                        if let Some((a_idx, pair_gain)) = best_coupled {
                            // Score the 2-step coupled `(remove rm_item, add add_item)` move directly
                            if best_act.map_or(true, |(_, _, _, bs)| pair_gain > bs) {
                                best_act = Some((3, r_idx, a_idx, pair_gain));
                            }
                        } else {
                            let score = -d_r;
                            if best_act.map_or(true, |(_, _, _, bs)| score > bs) {
                                best_act = Some((2, r_idx, 0, score));
                            }
                        }
                    }

                    match best_act {
                        Some((1, a_idx, _, _)) => {
                            let item = to_add.swap_remove(a_idx);
                            cur.add_item(ch, q_flat, item);
                            steps += 1;
                        }
                        Some((2, r_idx, _, _)) => {
                            let item = to_rem.swap_remove(r_idx);
                            cur.remove_item(ch, q_flat, item);
                            steps += 1;
                        }
                        Some((3, r_idx, a_idx, _)) => {
                            let rm_item = to_rem.swap_remove(r_idx);
                            let add_item = to_add.swap_remove(a_idx);
                            cur.remove_item(ch, q_flat, rm_item);
                            cur.add_item(ch, q_flat, add_item);
                            steps += 2;
                        }
                        _ => break,
                    }
                } else {
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
                }

                if cur.total_weight <= cur.capacity && cur.total_value > best_along_val {
                    best_along_val = cur.total_value;
                    best_along = Some(cur.clone());
                }

                if steps % step_check == 0 && cur.total_weight <= cur.capacity {
                    let mut probe = cur.clone();
                    fast_1to1_and_slack_climb(&mut probe, ch, q_flat, min_q);
                    if quad_bnb_k > 0 {
                        let _ = exact_quad_bnb_refine(&mut probe, ch, q_flat, min_q, quad_bnb_k);
                    }
                    if probe.total_value > best_along_val {
                        best_along_val = probe.total_value;
                        best_along = Some(probe);
                    }
                }
            }

            if let Some(mut cand) = best_along {
                polish_stage_b2_hybrid(&mut cand, ch, q_flat, min_q, 64, 50, 22, 42);
                extra_polish(&mut cand);
                if cand.total_value > best_state.total_value {
                    *best_state = cand;
                }
            }
        }
    }
}

/// Family 4 (`spectral_basins`): 2-Hop Spectral Synergy ($Q^2$) + Positive-Interaction Hub Seed Basins.
/// Constructs up to 3 orthogonal seed basins that capture 2-hop quadratic cluster synergy:
/// - Basin 0: 2-hop spectral potential $S^{(2)}_i = \sum_{j} \max(0, Q_{ij}) \cdot (v_j + \text{pos\_q}_j) / w_j$.
/// - Basin 1: Positive-only quadratic hub density $(v_i + 1.25 \cdot \text{pos\_q}_i) / w_i^{0.85}$.
/// - Basin 2: Incremental quadratic tension greedy (iteratively picking the unselected item maximising
///   $(\text{delta}_i + 0.35 \cdot \text{pos\_q}_i) / w_i$).
pub fn solve_spectral_synergy_basins(
    ch: &Challenge,
    q_flat: &[i64],
    min_q: i64,
    num_basins: usize,
    quad_bnb_k: usize,
) -> Vec<FlatState> {
    let n = ch.num_items;
    if num_basins == 0 || n == 0 {
        return Vec::new();
    }
    let mut pos_q = vec![0i64; n];
    for i in 0..n {
        let row_off = i * n;
        let mut s = 0i64;
        for j in 0..n {
            let v = q_flat[row_off + j];
            if v > 0 {
                s += v;
            }
        }
        pos_q[i] = s;
    }

    // 1-hop density score per item: (v_j + pos_q[j] / 2) / w_j
    let hop1: Vec<f64> = (0..n)
        .map(|j| ((ch.values[j] as i64 + pos_q[j] / 2) as f64) / ((ch.weights[j] as f64).max(1.0)))
        .collect();

    let mut out: Vec<FlatState> = Vec::with_capacity(num_basins.min(3));

    for b_idx in 0..num_basins.min(3) {
        let mut st = FlatState::new_empty(ch);
        if b_idx == 0 {
            // 2-hop spectral synergy ordering
            let mut order: Vec<(usize, f64)> = (0..n)
                .map(|i| {
                    let row_off = i * n;
                    let mut hop2 = 0.0f64;
                    for j in 0..n {
                        let qij = q_flat[row_off + j];
                        if qij > 0 {
                            hop2 += (qij as f64) * hop1[j];
                        }
                    }
                    let w = (ch.weights[i] as f64).max(1.0);
                    let score = ((ch.values[i] as f64) * 50.0 + hop2) / w;
                    (i, score)
                })
                .collect();
            order.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            for &(idx, _) in &order {
                if st.total_weight + (ch.weights[idx] as i64) <= st.capacity {
                    st.add_item(ch, q_flat, idx);
                }
            }
        } else if b_idx == 1 {
            // Positive-only quadratic hub density
            let mut order: Vec<(usize, f64)> = (0..n)
                .map(|i| {
                    let num = (ch.values[i] as f64) * 1024.0 + 480.0 * (pos_q[i] as f64);
                    let den = (ch.weights[i] as f64).powf(0.85).max(1.0);
                    (i, num / den)
                })
                .collect();
            order.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            for &(idx, _) in &order {
                if st.total_weight + (ch.weights[idx] as i64) <= st.capacity {
                    st.add_item(ch, q_flat, idx);
                }
            }
        } else {
            // Tension-shifted adaptive greedy construction:
            // Starts from the single highest-pos_q hub and greedily adds items maximising
            // `(cur_delta[i] + tension * pos_q[i]) / w_i` where `tension` decays as capacity fills.
            while st.slack() > 0 {
                let slack = st.slack();
                let fill_ratio = (st.total_weight as f64) / (st.capacity.max(1) as f64);
                let tension = (1.0 - fill_ratio).max(0.0) * 0.25;
                let mut best_k: Option<(usize, f64)> = None;
                for i in 0..n {
                    if st.selected[i] == 0 && (ch.weights[i] as i64) <= slack {
                        let eff = (st.delta[i] as f64) + tension * (pos_q[i] as f64);
                        if eff <= 0.0 {
                            continue;
                        }
                        let s = eff / (ch.weights[i] as f64).max(1.0);
                        if best_k.map_or(true, |(_, bs)| s > bs) {
                            best_k = Some((i, s));
                        }
                    }
                }
                if let Some((k, _)) = best_k {
                    st.add_item(ch, q_flat, k);
                } else {
                    break;
                }
            }
        }

        integer_core_dp_refine(&mut st, ch, q_flat, min_q, 35);
        fast_1to1_and_slack_climb(&mut st, ch, q_flat, min_q);
        if quad_bnb_k > 0 {
            let _ = exact_quad_bnb_refine(&mut st, ch, q_flat, min_q, quad_bnb_k);
        }
        push_distinct(&mut out, st);
    }
    // 1b. Positive-interaction hub & tension-shifted basins
    out
}

const CB_SC: i64 = 64;
const CB_INF: i64 = i64::MAX / 8;

struct CbFlow {
    to: Vec<u32>,
    cap: Vec<i64>,
    head: Vec<i32>,
    nxt: Vec<i32>,
    level: Vec<i32>,
    it: Vec<i32>,
    q: Vec<u32>,
    stk: Vec<u32>,
    pe: Vec<i32>,
    tarc: Vec<i32>,
    pre: bool,
}

impl CbFlow {
    fn new() -> Self {
        CbFlow {
            to: Vec::new(),
            cap: Vec::new(),
            head: Vec::new(),
            nxt: Vec::new(),
            level: Vec::new(),
            it: Vec::new(),
            q: Vec::new(),
            stk: Vec::new(),
            pe: Vec::new(),
            tarc: Vec::new(),
            pre: true,
        }
    }
    fn reset(&mut self, n: usize) {
        self.to.clear();
        self.cap.clear();
        self.nxt.clear();
        self.head.clear();
        self.head.resize(n, -1);
        self.level.clear();
        self.level.resize(n, -1);
        self.it.clear();
        self.it.resize(n, -1);
        self.pe.clear();
        self.pe.resize(n, -1);
        self.tarc.clear();
        self.tarc.resize(n, -1);
    }
    #[inline]
    fn add(&mut self, u: usize, v: usize, c: i64) {
        let e = self.to.len();
        if v + 1 == self.tarc.len() {
            self.tarc[u] = e as i32;
        }
        self.to.push(v as u32);
        self.cap.push(c);
        self.nxt.push(self.head[u]);
        self.head[u] = e as i32;
        self.to.push(u as u32);
        self.cap.push(0);
        self.nxt.push(self.head[v]);
        self.head[v] = (e + 1) as i32;
    }
    fn bfs(&mut self, s: usize, t: usize) -> bool {
        for x in self.level.iter_mut() {
            *x = -1;
        }
        self.level[s] = 0;
        self.q.clear();
        self.q.push(s as u32);
        let mut qi = 0usize;
        while qi < self.q.len() {
            let u = self.q[qi] as usize;
            qi += 1;
            if self.level[t] >= 0 && self.level[u] >= self.level[t] {
                break;
            }
            let mut e = self.head[u];
            while e != -1 {
                let ei = e as usize;
                let v = self.to[ei] as usize;
                if self.cap[ei] > 0 && self.level[v] < 0 {
                    self.level[v] = self.level[u] + 1;
                    self.q.push(v as u32);
                }
                e = self.nxt[ei];
            }
        }
        self.level[t] >= 0
    }
    fn augment(&mut self, s: usize, t: usize) -> i64 {
        let mut total = 0i64;
        self.stk.clear();
        self.stk.push(s as u32);
        loop {
            let u = match self.stk.last() {
                Some(&u) => u as usize,
                None => return total,
            };
            if u == t {
                let mut f = CB_INF;
                let mut cut = 1usize;
                for k in 1..self.stk.len() {
                    let v = self.stk[k] as usize;
                    let c = self.cap[self.pe[v] as usize];
                    if c < f {
                        f = c;
                        cut = k;
                    }
                }
                for k in 1..self.stk.len() {
                    let v = self.stk[k] as usize;
                    let ei = self.pe[v] as usize;
                    self.cap[ei] -= f;
                    self.cap[ei ^ 1] += f;
                }
                total += f;
                self.stk.truncate(cut);
                continue;
            }
            let mut adv = false;
            while self.it[u] != -1 {
                let ei = self.it[u] as usize;
                let v = self.to[ei] as usize;
                if self.cap[ei] > 0 && self.level[v] == self.level[u] + 1 {
                    self.pe[v] = ei as i32;
                    self.stk.push(v as u32);
                    adv = true;
                    break;
                }
                self.it[u] = self.nxt[ei];
            }
            if !adv {
                self.level[u] = -1;
                self.stk.pop();
                if let Some(&p) = self.stk.last() {
                    let p = p as usize;
                    let ei = self.it[p] as usize;
                    self.it[p] = self.nxt[ei];
                }
            }
        }
    }
    fn maxflow(&mut self, s: usize, t: usize) -> i64 {
        let mut flow = 0i64;
        if self.pre {
            let mut e0 = self.head[s];
            while e0 != -1 {
                let ei = e0 as usize;
                if self.cap[ei] > 0 {
                    let a = self.to[ei] as usize;
                    let mut ea = self.head[a];
                    while ea != -1 {
                        let eai = ea as usize;
                        if self.cap[eai] > 0 {
                            let b = self.to[eai] as usize;
                            let tb = self.tarc[b];
                            if tb >= 0 {
                                let tbi = tb as usize;
                                let mut f = self.cap[ei].min(self.cap[eai]).min(self.cap[tbi]);
                                if f > 0 {
                                    self.cap[ei] -= f;
                                    self.cap[ei ^ 1] += f;
                                    self.cap[eai] -= f;
                                    self.cap[eai ^ 1] += f;
                                    self.cap[tbi] -= f;
                                    self.cap[tbi ^ 1] += f;
                                    flow += f;
                                    if self.cap[ei] == 0 {
                                        break;
                                    }
                                }
                            }
                        }
                        ea = self.nxt[eai];
                    }
                }
                e0 = self.nxt[ei];
            }
        }
        while self.bfs(s, t) {
            for i in 0..self.head.len() {
                self.it[i] = self.head[i];
            }
            flow += self.augment(s, t);
        }
        flow
    }
    fn source_side(&mut self, s: usize, out: &mut Vec<bool>) {
        out.clear();
        out.resize(self.head.len(), false);
        self.stk.clear();
        self.stk.push(s as u32);
        out[s] = true;
        while let Some(u) = self.stk.pop() {
            let mut e = self.head[u as usize];
            while e != -1 {
                let ei = e as usize;
                let v = self.to[ei] as usize;
                if self.cap[ei] > 0 && !out[v] {
                    out[v] = true;
                    self.stk.push(v as u32);
                }
                e = self.nxt[ei];
            }
        }
    }
}

struct ClosureBnb {
    k: usize,
    w: Vec<i64>,
    lin_cur: Vec<i64>,
    cap: i64,
    best_val: i64,
    best_sel: Vec<u8>,
    improved: bool,
    work: u64,
    work_cap: u64,
    lam_iters: usize,
    adj_s: Vec<u32>,
    adj_j: Vec<u32>,
    adj_q: Vec<i64>,
    pos: Vec<i32>,
    state: Vec<i8>,
    fl: CbFlow,
    idx: Vec<u32>,
    side: Vec<bool>,
}

impl ClosureBnb {
    fn closure(&mut self, lam: i64) -> (i64, i64) {
        let m = self.idx.len();
        let s = m;
        let t = m + 1;
        self.fl.reset(m + 2);
        let mut cst = 0i64;
        for a in 0..m {
            let ia = self.idx[a] as usize;
            let mut e = lam * self.w[ia] - CB_SC * self.lin_cur[ia];
            let (b0, b1) = (self.adj_s[ia] as usize, self.adj_s[ia + 1] as usize);
            for x in b0..b1 {
                let pb = self.pos[self.adj_j[x] as usize];
                if pb > a as i32 {
                    e -= CB_SC * self.adj_q[x];
                }
            }
            if e > 0 {
                self.fl.add(a, t, e);
            } else if e < 0 {
                cst += e;
                self.fl.add(s, a, -e);
            }
        }
        for a in 0..m {
            let ia = self.idx[a] as usize;
            let (b0, b1) = (self.adj_s[ia] as usize, self.adj_s[ia + 1] as usize);
            self.work += (b1 - b0) as u64;
            for x in b0..b1 {
                let pb = self.pos[self.adj_j[x] as usize];
                if pb > a as i32 {
                    self.fl.add(a, pb as usize, CB_SC * self.adj_q[x]);
                }
            }
        }
        let cut = self.fl.maxflow(s, t);
        self.work += self.fl.to.len() as u64;
        let mut side = Vec::new();
        self.fl.source_side(s, &mut side);
        side.truncate(m);
        let mut wt = 0i64;
        for a in 0..m {
            if side[a] {
                wt += self.w[self.idx[a] as usize];
            }
        }
        self.side = side;
        (-(cst + cut), wt)
    }

    fn try_incumbent(&mut self, base: i64) {
        let mut v = base;
        let m = self.idx.len();
        for a in 0..m {
            if !self.side[a] {
                continue;
            }
            let ia = self.idx[a] as usize;
            v += self.lin_cur[ia];
            let (b0, b1) = (self.adj_s[ia] as usize, self.adj_s[ia + 1] as usize);
            for x in b0..b1 {
                let pb = self.pos[self.adj_j[x] as usize];
                if pb > a as i32 && self.side[pb as usize] {
                    v += self.adj_q[x];
                }
            }
        }
        if v > self.best_val {
            self.best_val = v;
            self.improved = true;
            for i in 0..self.k {
                self.best_sel[i] = (self.state[i] == 1) as u8;
            }
            for a in 0..m {
                if self.side[a] {
                    self.best_sel[self.idx[a] as usize] = 1;
                }
            }
        }
    }

    fn dfs(&mut self, base: i64, c: i64, lam_in: i64, li: usize) {
        if self.work >= self.work_cap {
            return;
        }
        self.idx.clear();
        for i in 0..self.k {
            if self.state[i] == -1 {
                self.pos[i] = self.idx.len() as i32;
                self.idx.push(i as u32);
            } else {
                self.pos[i] = -1;
            }
        }
        self.work += self.k as u64;
        if self.idx.is_empty() {
            if base > self.best_val {
                self.side = Vec::new();
                self.try_incumbent(base);
            }
            return;
        }
        let thr = (self.best_val + 1) * CB_SC;
        let bsc = base * CB_SC;
        let mut n_cuts = 1usize;
        let (v, wt) = self.closure(lam_in);
        let mut bound = bsc + lam_in * c + v;
        if wt <= c {
            self.try_incumbent(base);
        }
        if bound < thr {
            return;
        }
        let mut best_lam = lam_in;
        let mut best_side = self.side.clone();
        let mut lo;
        let mut hi;
        if wt > c {
            lo = lam_in;
            let mut step = 1i64.max(lam_in / 16);
            loop {
                hi = lam_in + step;
                let (v2, w2) = self.closure(hi);
                n_cuts += 1;
                if self.work >= self.work_cap {
                    return;
                }
                let b2 = bsc + hi * c + v2;
                if w2 <= c {
                    self.try_incumbent(base);
                }
                if b2 < bound {
                    bound = b2;
                    best_lam = hi;
                    best_side = self.side.clone();
                }
                if bound < (self.best_val + 1) * CB_SC {
                    return;
                }
                if w2 <= c || n_cuts > li || step > (1i64 << 40) {
                    break;
                }
                lo = hi;
                step *= 2;
            }
        } else {
            hi = lam_in;
            lo = lam_in;
            if lam_in > 0 {
                let mut step = 1i64.max(lam_in / 16);
                loop {
                    lo = (lam_in - step).max(0);
                    let (v2, w2) = self.closure(lo);
                    n_cuts += 1;
                    if self.work >= self.work_cap {
                        return;
                    }
                    let b2 = bsc + lo * c + v2;
                    if w2 <= c {
                        self.try_incumbent(base);
                    }
                    if b2 < bound {
                        bound = b2;
                        best_lam = lo;
                        best_side = self.side.clone();
                    }
                    if bound < (self.best_val + 1) * CB_SC {
                        return;
                    }
                    if w2 > c || lo == 0 || n_cuts > li {
                        break;
                    }
                    hi = lo;
                    step *= 2;
                }
            }
        }
        while hi - lo > 1 && n_cuts <= li {
            let mid = lo + (hi - lo) / 2;
            let (v2, w2) = self.closure(mid);
            n_cuts += 1;
            if self.work >= self.work_cap {
                return;
            }
            let b2 = bsc + mid * c + v2;
            if w2 <= c {
                self.try_incumbent(base);
            }
            if b2 < bound {
                bound = b2;
                best_lam = mid;
                best_side = self.side.clone();
            }
            if bound < (self.best_val + 1) * CB_SC {
                return;
            }
            if w2 > c {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let mut bi = usize::MAX;
        let mut bw = -1i64;
        for a in 0..self.idx.len() {
            if best_side[a] {
                let i = self.idx[a] as usize;
                if self.w[i] > bw {
                    bw = self.w[i];
                    bi = i;
                }
            }
        }
        if bi == usize::MAX {
            for a in 0..self.idx.len() {
                let i = self.idx[a] as usize;
                if self.w[i] > bw {
                    bw = self.w[i];
                    bi = i;
                }
            }
        }
        if self.w[bi] <= c {
            let add = self.lin_cur[bi];
            self.state[bi] = 1;
            let (b0, b1) = (self.adj_s[bi] as usize, self.adj_s[bi + 1] as usize);
            for x in b0..b1 {
                let j = self.adj_j[x] as usize;
                self.lin_cur[j] += self.adj_q[x];
            }
            self.dfs(base + add, c - self.w[bi], best_lam, self.lam_iters);
            for x in b0..b1 {
                let j = self.adj_j[x] as usize;
                self.lin_cur[j] -= self.adj_q[x];
            }
        }
        self.state[bi] = 0;
        self.dfs(base, c, best_lam, self.lam_iters);
        self.state[bi] = -1;
    }

    fn bound_at(&mut self, base: i64, c: i64, lam: i64) -> i64 {
        if c < 0 {
            return i64::MIN / 4;
        }
        let (v, w) = self.closure(lam);
        if w <= c {
            self.try_incumbent(base);
        }
        base * CB_SC + lam * c + v
    }

    fn root_lambda(&mut self, c: i64) -> i64 {
        let mut lo = 0i64;
        let (v0, w0) = self.closure(0);
        if w0 <= c {
            self.try_incumbent(0);
            return 0;
        }
        let mut hi = 64i64;
        let mut best_lam = 0i64;
        let mut best_b = v0;
        for _ in 0..36 {
            let (vh, wh) = self.closure(hi);
            let b = hi * c + vh;
            if wh <= c {
                self.try_incumbent(0);
            }
            if b < best_b {
                best_b = b;
                best_lam = hi;
            }
            if wh <= c {
                break;
            }
            lo = hi;
            hi *= 2;
        }
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            let (vm, wm) = self.closure(mid);
            let b = mid * c + vm;
            if wm <= c {
                self.try_incumbent(0);
            }
            if b < best_b {
                best_b = b;
                best_lam = mid;
            }
            if wm > c {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        best_lam
    }

    fn root_fix(&mut self, mut base: i64, mut c: i64, lam: i64) -> (i64, i64, bool) {
        for pass in 0..3 {
            if self.work >= self.work_cap {
                break;
            }
            let (v0, w0) = self.closure(lam);
            if w0 <= c {
                self.try_incumbent(base);
            }
            let b0 = base * CB_SC + lam * c + v0;
            let thr = (self.best_val + 1) * CB_SC;
            if b0 < thr {
                return (base, c, true);
            }
            let side0 = self.side.clone();
            let idx0 = self.idx.clone();
            let mut n_fixed = 0usize;
            for a in 0..idx0.len() {
                if self.work >= self.work_cap {
                    break;
                }
                let i = idx0[a] as usize;
                if self.state[i] >= 0 {
                    continue;
                }
                if self.w[i] > c {
                    self.state[i] = 0;
                    n_fixed += 1;
                    continue;
                }
                if !side0[a] {
                    let add = self.lin_cur[i];
                    self.state[i] = 1;
                    let (s0, s1) = (self.adj_s[i] as usize, self.adj_s[i + 1] as usize);
                    for x in s0..s1 {
                        let j = self.adj_j[x] as usize;
                        self.lin_cur[j] += self.adj_q[x];
                    }
                    let b = self.bound_at(base + add, c - self.w[i], lam);
                    for x in s0..s1 {
                        let j = self.adj_j[x] as usize;
                        self.lin_cur[j] -= self.adj_q[x];
                    }
                    let thr = (self.best_val + 1) * CB_SC;
                    if b < thr {
                        self.state[i] = 0;
                        n_fixed += 1;
                    } else {
                        self.state[i] = -1;
                    }
                } else {
                    self.state[i] = 0;
                    let b = self.bound_at(base, c, lam);
                    let thr = (self.best_val + 1) * CB_SC;
                    if b < thr {
                        self.state[i] = 1;
                        base += self.lin_cur[i];
                        c -= self.w[i];
                        let (s0, s1) = (self.adj_s[i] as usize, self.adj_s[i + 1] as usize);
                        for x in s0..s1 {
                            let j = self.adj_j[x] as usize;
                            self.lin_cur[j] += self.adj_q[x];
                        }
                        n_fixed += 1;
                    } else {
                        self.state[i] = -1;
                    }
                }
            }
            if n_fixed == 0 || pass == 2 {
                break;
            }
        }
        (base, c, false)
    }
}

thread_local! {
    static CLOSURE_IN_CORE: std::cell::RefCell<Vec<bool>> = std::cell::RefCell::new(Vec::new());
}

/// Family 6 (`closure_bnb_k`): Max-Flow / Min-Cut Lagrangian Closure `CoreBnb` on `FlatState` + Elite Symmetric Difference.
/// Builds a core window of size `k_core` (`40..=110`) combining:
/// 1. All items in the symmetric difference between `state` and the top distinct `elites` (`sym_diff`), and
/// 2. The boundary items (`build_boundary_windows`) closest to the marginal density break.
/// Solves the exact sub-QKP over `core` using Dinic Max-Flow / Min-Cut closure bounds (`CbFlow`) with Lagrangian
/// weight multiplier $\lambda$ bisection (`ClosureBnb::root_lambda` + `root_fix` + `dfs`), followed by `fast_1to1_and_slack_climb`.
pub fn closure_core_bnb_refine(
    state: &mut FlatState,
    elites: &[FlatState],
    ch: &Challenge,
    q_flat: &[i64],
    min_q: i64,
    k_core: usize,
    work_cap: u64,
) -> bool {
    if k_core < 16 || work_cap == 0 {
        return false;
    }
    let n = state.n;
    let mut core: Vec<usize> = Vec::with_capacity(k_core + 16);

    CLOSURE_IN_CORE.with(|cell| {
        let mut in_core = cell.borrow_mut();
        if in_core.len() < n {
            in_core.resize(n, false);
        } else {
            in_core[..n].fill(false);
        }

        // 1. Prioritize items that differ across the top distinct elites (`sym_diff`)
        for e in elites.iter().take(4) {
            for u in 0..n {
                if e.selected[u] != state.selected[u] && !in_core[u] {
                    in_core[u] = true;
                    core.push(u);
                    if core.len() >= k_core {
                        break;
                    }
                }
            }
            if core.len() >= k_core {
                break;
            }
        }

        // 2. Fill remaining core slots up to `k_core` from the boundary windows around `state`'s density break
        if core.len() < k_core {
            let rem_half = ((k_core - core.len()) / 2).max(16);
            let (sel_win, unsel_win) = build_boundary_windows(state, ch, rem_half + 16, 8);
            let max_len = sel_win.len().max(unsel_win.len());
            for idx in 0..max_len {
                if core.len() >= k_core {
                    break;
                }
                if idx < sel_win.len() {
                    let u = sel_win[idx];
                    if !in_core[u] {
                        in_core[u] = true;
                        core.push(u);
                    }
                }
                if core.len() >= k_core {
                    break;
                }
                if idx < unsel_win.len() {
                    let u = unsel_win[idx];
                    if !in_core[u] {
                        in_core[u] = true;
                        core.push(u);
                    }
                }
            }
        }
    });

    let k = core.len();
    if k < 8 {
        return false;
    }

    // Compute exact linear value `lin_cur[a]` of each core item relative to fixed `selected \ core` items,
    // current core weight `cur_core_w`, and current core objective `init_core_val`.
    let mut lin_cur = vec![0i64; k];
    let mut w = vec![0i64; k];
    let mut cur_core_w = 0i64;
    let mut init_core_val = 0i64;

    for (a, &ua) in core.iter().enumerate() {
        let mut l = state.delta[ua];
        let ua_off = ua * n;
        for (b, &ub) in core.iter().enumerate() {
            if a != b && state.selected[ub] == 1 {
                l -= q_flat[ua_off + ub];
            }
        }
        lin_cur[a] = l;
        w[a] = ch.weights[ua] as i64;
        if state.selected[ua] == 1 {
            cur_core_w += w[a];
            init_core_val += l;
            for b in 0..a {
                let ub = core[b];
                if state.selected[ub] == 1 {
                    init_core_val += q_flat[ua_off + ub];
                }
            }
        }
    }

    let rem_cap = state.slack() + cur_core_w;
    if rem_cap <= 0 {
        return false;
    }

    let mut adj_s: Vec<u32> = Vec::with_capacity(k + 1);
    let mut adj_j: Vec<u32> = Vec::new();
    let mut adj_q: Vec<i64> = Vec::new();
    for a in 0..k {
        adj_s.push(adj_j.len() as u32);
        let ua_off = core[a] * n;
        for b in 0..k {
            if a != b {
                let qq = q_flat[ua_off + core[b]];
                if qq > 0 {
                    adj_j.push(b as u32);
                    adj_q.push(qq);
                }
            }
        }
    }
    adj_s.push(adj_j.len() as u32);

    let mut bb = ClosureBnb {
        k,
        w,
        lin_cur,
        cap: rem_cap,
        best_val: init_core_val,
        best_sel: vec![0u8; k],
        improved: false,
        work: (k * k) as u64,
        work_cap,
        lam_iters: 14,
        adj_s,
        adj_j,
        adj_q,
        pos: vec![-1; k],
        state: vec![-1; k],
        fl: CbFlow::new(),
        idx: Vec::new(),
        side: Vec::new(),
    };

    let root_lam = bb.root_lambda(rem_cap);
    if bb.work < bb.work_cap {
        let (base0, c0, pruned) = bb.root_fix(0, rem_cap, root_lam);
        if !pruned && bb.work < bb.work_cap {
            bb.dfs(base0, c0, root_lam.max(1), 24);
        }
    }
    if !bb.improved {
        return false;
    }

    let mut cand = state.clone();
    for a in 0..k {
        let u = core[a];
        if cand.selected[u] == 1 && bb.best_sel[a] == 0 {
            cand.remove_item(ch, q_flat, u);
        }
    }
    for a in 0..k {
        let u = core[a];
        if cand.selected[u] == 0 && bb.best_sel[a] != 0 && cand.total_weight + (ch.weights[u] as i64) <= cand.capacity {
            cand.add_item(ch, q_flat, u);
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

/// Deterministic switchboard read from `hyperparameters`; a missing key takes the default below.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwitchConfig {
    /// T41: 1 also runs track1 with satchel's single undivided CoreBnb pass (`core_ktgt` 0, `core_ksteps` 1, `core_kstart` 0).
    pub satchel_core: usize,
    /// Number of v10 warm basins (`solve_multi_warm_b2_elites`, at most 4) per track.
    pub warm_basins_t41: usize,
    pub warm_basins_t39: usize,
    pub warm_basins_t40: usize,
    pub warm_basins_t42: usize,
    pub warm_basins_t43: usize,
    /// T39: `n_perturbation_rounds` passed to `track_t39` (its own default is 49).
    pub t39_rounds: usize,
    /// 1 selects the frontier-bounded polish and tabu bursts in `frontier_bounds`; 0 restores v11's.
    pub fast_polish: usize,
    /// T42: 1 runs track4's `fused_dual` (both hub_pair trajectories in one call, shared seed groups) as packrun's
    /// track and takes its candidate buffer; 0 restores v12's second `use_hub_pair` 0 call.
    pub t42_fused: usize,
    /// T42: number of distinct track4 fused candidates (best first) entering Stage B2 polish and relinking.
    pub t42_elites: usize,
    /// T43: number of distinct track5 elite basins (best first) entering the Stage B2 polish and relinking.
    pub t43_elites: usize,
    /// T40: number of distinct track3 elite basins (`xr_pool` + `core_stage` `alt`) drained from `LAST_T40_CANDIDATES`.
    pub t40_elites: usize,
    /// T39: number of distinct track2 `cpool` CoreBnb solutions drained from `LAST_T39_CANDIDATES`.
    pub t39_cpool: usize,
    /// Stage B2 exact core DP half-window override (`0` = use `frontier_bounds`).
    pub b2_core_dp: usize,
    /// Stage B2 tabu burst count override (`99` = use `frontier_bounds`; `0` skips tabu escape).
    pub tabu_bursts_override: usize,
    /// Maximum distinct candidates polished in Step 3 (`usize::MAX` = all).
    pub polish_top_k: usize,
    /// Stage B2 `(1-out, 2-in)/(2-out, 1-in)` window width override (`0` = use `frontier_bounds`).
    pub b2_k12: usize,
    /// Stage B2 `(2-out, 2-in)` window width override (`0` = use `frontier_bounds`).
    pub b2_k22: usize,
    /// Stage B2 `(3-out, 3-in)` window width override (`0` = use `frontier_bounds`).
    pub b2_k33: usize,
    /// Maximum elite guides used in `recombine_and_relink_elites` (`0` skips relinking).
    pub relink_limit: usize,
    /// Family 1 (`exact_quad_bnb`): Exact quadratic Branch-and-Bound core window size `k_total <= 24` (`0` = disabled).
    pub quad_bnb_k: usize,
    /// Family 2 (`bucket_compound`): Weight-bucketed `O(W^2)` wide-window `(2-out, 2-in)` compound width (`0` = disabled).
    pub bucket_compound: usize,
    /// Family 3 (`lookahead_relink`): `1` enables 2-step lookahead path relinking + union-drop recombination (`0` = standard).
    pub lookahead_relink: usize,
    /// Family 4 (`spectral_basins`): Number of 2-hop spectral synergy ($Q^2$) seed basins (`0..=3`).
    pub spectral_basins: usize,
    /// Family 5 (`t42_dedup`): `1` shares DP/VND memos across T42 fused trajectories; `2..=4` also caps trajectory-0 restarts (`20/16/12`).
    pub t42_dedup: usize,
    /// Family 6 (`closure_bnb_k`): Max-Flow / Min-Cut Lagrangian Closure `CoreBnb` core window size `k_core` (`0` = disabled).
    pub closure_bnb_k: usize,
    /// Family 7 (`t39_adaptive`): Adaptive T39 escalation & T39-only tabu escape (`0` = disabled; `1` = T39 tabu escape; `2` = adaptive rounds escalation + T39 tabu; `3` = adaptive rounds escalation when track_t39(4) != packrun + T39 cpool + T39 tabu).
    pub t39_adaptive: usize,
    /// Family 8 (`t43_dedup`): T43 restart-stall capping (`1` = `22/10`, `2` = `16/8`, `3` = `12/6`, `4` = `24/12`, `5` = `32/20`) + CSR disputed-core exact 0-1 DP recombination across distinct `T43` basins (skipped for `5`).
    pub t43_dedup: usize,
    /// Deep ILS rounds on Track 42 (dense N=1000/5000, `0` = disabled).
    pub ils_rounds_t42: usize,
    /// Deep ILS rounds on Track 43 (N=5000 25% budget, `0` = disabled).
    pub ils_rounds_t43: usize,
}

impl SwitchConfig {
    pub const DEFAULT: SwitchConfig = SwitchConfig {
        satchel_core: 1,
        warm_basins_t41: 0,
        warm_basins_t39: 0,
        warm_basins_t40: 0,
        warm_basins_t42: 0,
        warm_basins_t43: 0,
        t39_rounds: 4,
        fast_polish: 1,
        t42_fused: 1,
        t42_elites: 4,
        t43_elites: 2,
        t40_elites: 0,
        t39_cpool: 0,
        b2_core_dp: 0,
        tabu_bursts_override: 0,
        polish_top_k: usize::MAX,
        b2_k12: 0,
        b2_k22: 0,
        b2_k33: 0,
        relink_limit: 6,
        quad_bnb_k: 0,
        bucket_compound: 0,
        lookahead_relink: 0,
        spectral_basins: 0,
        t42_dedup: 3,
        closure_bnb_k: 0,
        t39_adaptive: 2,
        t43_dedup: 5,
        ils_rounds_t42: 2,
        ils_rounds_t43: 2,
    };

    pub fn from_hyperparameters(hp: &Option<Map<String, Value>>) -> Self {
        let mut c = Self::DEFAULT;
        if let Some(m) = hp {
            let get = |k: &str, d: usize| m.get(k).and_then(|v| v.as_u64()).map_or(d, |v| v as usize);
            c.satchel_core = get("satchel_core", c.satchel_core);
            c.warm_basins_t41 = get("warm_basins_t41", c.warm_basins_t41);
            c.warm_basins_t39 = get("warm_basins_t39", c.warm_basins_t39);
            c.warm_basins_t40 = get("warm_basins_t40", c.warm_basins_t40);
            c.warm_basins_t42 = get("warm_basins_t42", c.warm_basins_t42);
            c.warm_basins_t43 = get("warm_basins_t43", c.warm_basins_t43);
            c.t39_rounds = get("t39_rounds", c.t39_rounds);
            c.fast_polish = get("fast_polish", c.fast_polish);
            c.t42_fused = get("t42_fused", c.t42_fused);
            c.t42_elites = get("t42_elites", c.t42_elites);
            c.t43_elites = get("t43_elites", c.t43_elites);
            c.t40_elites = get("t40_elites", c.t40_elites);
            c.t39_cpool = get("t39_cpool", c.t39_cpool);
            c.b2_core_dp = get("b2_core_dp", c.b2_core_dp);
            c.tabu_bursts_override = get("tabu_bursts_override", c.tabu_bursts_override);
            c.polish_top_k = get("polish_top_k", c.polish_top_k);
            c.b2_k12 = get("b2_k12", c.b2_k12);
            c.b2_k22 = get("b2_k22", c.b2_k22);
            c.b2_k33 = get("b2_k33", c.b2_k33);
            c.relink_limit = get("relink_limit", c.relink_limit);
            c.quad_bnb_k = get("quad_bnb_k", c.quad_bnb_k);
            c.bucket_compound = get("bucket_compound", c.bucket_compound);
            c.lookahead_relink = get("lookahead_relink", c.lookahead_relink);
            c.spectral_basins = get("spectral_basins", c.spectral_basins);
            c.t42_dedup = get("t42_dedup", c.t42_dedup);
            c.closure_bnb_k = get("closure_bnb_k", c.closure_bnb_k);
            c.t39_adaptive = get("t39_adaptive", c.t39_adaptive);
            c.t43_dedup = get("t43_dedup", c.t43_dedup);
            c.ils_rounds_t42 = get("ils_rounds_t42", c.ils_rounds_t42);
            c.ils_rounds_t43 = get("ils_rounds_t43", c.ils_rounds_t43);
        }
        c
    }
}

/// The caller's hyperparameters with `pairs` set on top (tracks ignore keys they do not parse).
fn hp_with(hp: &Option<Map<String, Value>>, pairs: &[(&str, u64)]) -> Option<Map<String, Value>> {
    let mut m = hp.clone().unwrap_or_default();
    for &(k, v) in pairs {
        m.insert(k.to_string(), Value::Number(serde_json::Number::from(v as i64)));
    }
    Some(m)
}

/// Frontier bounds per shape: (polish args, tabu bursts, tabu steps per burst).
fn frontier_bounds(n: usize, budget: u32, fast: bool) -> (PolishArgs, usize, usize) {
    match (n, budget, fast) {
        (1000, b, true) if b <= 7 => ((48, 38, 18, 30), 2, 18),
        (1000, _, true) => ((64, 52, 24, 40), 3, 22),
        (_, _, true) => ((48, 38, 18, 30), 2, 16),
        (1000, b, false) if b <= 17 => ((78, 62, 28, 48), 5, 28),
        (1000, _, false) => ((80, 64, 30, 50), 6, 30),
        (_, _, false) => ((52, 42, 20, 34), 3, 20),
    }
}

/// TIG-EVO-16 entry point. Extends `stage_b2_hybrid_v15` with:
/// 6. `closure_bnb_k`: Max-Flow / Min-Cut Lagrangian Closure `CoreBnb` (`closure_core_bnb_refine`) over `FlatState` + elite symmetric differences.
/// 7. `t39_adaptive`: Adaptive Track-39 escalation and selective T39 tabu escape (recovering OOS `T39.s14` `+66` and `T39.s20` `+7` without N=5000 wall-time penalty).
/// 8. `t43_dedup`: Track-43 (`N=5000`, `25%` budget) restart-stall capping + CSR disputed-core exact 0-1 DP recombination.
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let n = challenge.num_items;
    let budget = budget_pct(challenge);
    let sw = SwitchConfig::from_hyperparameters(hyperparameters);
    let t39 = n == 1000 && budget > 7 && budget <= 17;
    let t42 = n == 5000 && budget <= 17;
    let t42_fused = t42 && sw.t42_fused == 1;

    // Step 1: packrun's track (on T42 with `t42_fused`, track4 running both hub_pair trajectories, so the first save
    // is the better of packrun's and satchel's T42 trajectories). Its internal save is captured and saved now.
    let track = packrun_track(n, budget)?;
    let step1_hp = if t42_fused {
        hp_with(hyperparameters, &[("fused_dual", 1), ("t42_dedup", sw.t42_dedup as u64)])
    } else if n == 5000 && !t42 && sw.t43_dedup > 0 {
        hp_with(hyperparameters, &[("t43_dedup", sw.t43_dedup as u64)])
    } else {
        hyperparameters.clone()
    };
    let packrun_sol = capture_track(track, challenge, &step1_hp)?
        .ok_or_else(|| anyhow!("packrun track saved no solution"))?;
    // The track buffers are thread-local and drained here, before any other track runs on this thread.
    let t39_buffer = if t39 { track2::collect_last_candidates() } else { Vec::new() };
    let t40_buffer = if n == 1000 && budget > 17 { track3::collect_last_candidates() } else { Vec::new() };
    let t42_buffer = if t42_fused { track4::collect_last_candidates() } else { Vec::new() };
    let t43_buffer = if n == 5000 && !t42 { track5::collect_last_candidates() } else { Vec::new() };
    save_solution(&packrun_sol)?;
    // Velocity exit (t43_dedup 5): track5's capped basins already reach Z* on T43, so Steps 2 and 3 are skipped.
    if n == 5000 && !t42 && sw.t43_dedup == 5 && sw.ils_rounds_t43 == 0 {
        return Ok(());
    }
    // Velocity exit (T42 dedup 3+): track4 returned no buffered candidates (its q_4 == q_0 early return clears them).
    if t42_fused && sw.t42_dedup >= 3 && t42_buffer.is_empty() && sw.ils_rounds_t42 == 0 {
        return Ok(());
    }

    // Fast pre-Step-2 velocity exits on T41 and T39 BEFORE building the N x N `q_flat` matrix:
    if n == 1000 && budget <= 7 && sw.satchel_core == 1 && sw.warm_basins_t41 == 0 {
        let saved_val = eval_sol_if_feasible(challenge, &packrun_sol).unwrap_or(0);
        let satchel_hp = hp_with(hyperparameters, &[("core_ktgt", 0), ("core_ksteps", 1), ("core_kstart", 0)]);
        if let Ok(Some(sat_sol)) = capture_track(track1::solve as TrackFn, challenge, &satchel_hp) {
            if eval_sol_if_feasible(challenge, &sat_sol).unwrap_or(0) > saved_val {
                save_solution(&sat_sol)?;
            }
            return Ok(());
        }
    }
    let mut precomputed_t39_sols: Option<Vec<Solution>> = None;
    if t39 && sw.t39_rounds == 4 && sw.t39_adaptive == 0 && sw.t39_cpool == 0 && sw.warm_basins_t39 == 0 {
        let saved_val = eval_sol_if_feasible(challenge, &packrun_sol).unwrap_or(0);
        let t39_hp = hp_with(hyperparameters, &[("n_perturbation_rounds", 2)]);
        let probe_sols = track_t39::collect_candidate_solutions(challenge, &t39_hp);
        let probe_best = probe_sols
            .iter()
            .filter_map(|sol| eval_sol_if_feasible(challenge, sol))
            .max()
            .unwrap_or(saved_val);
        if probe_best <= saved_val {
            return Ok(());
        }
        precomputed_t39_sols = Some(probe_sols);
    }

    // Step 2: per-shape candidates next to packrun's solution.
    let (q_flat, sum_q, min_q) = build_q_flat(challenge);
    let seed_u64 = fold_seed_u64(&challenge.seed);
    let packrun_state = FlatState::from_items(challenge, &q_flat, &packrun_sol.items);
    let saved_value = packrun_state.total_value;
    let mut cands: Vec<FlatState> = vec![packrun_state];

    let warm_basins = match (n, budget) {
        // T41: satchel's schedule, a single undivided CoreBnb pass on the nominal window
        (1000, b) if b <= 7 => {
            if sw.satchel_core == 1 {
                let satchel_hp = hp_with(hyperparameters, &[("core_ktgt", 0), ("core_ksteps", 1), ("core_kstart", 0)]);
                cands.extend(track_flat(track1::solve as TrackFn, challenge, &q_flat, &satchel_hp).ok().flatten());
            }
            // Velocity exit (satchel_core 1): max(packrun, satchel's core pass) already reaches Z* on every T41 seed, so Step 3 is skipped.
            if sw.satchel_core == 1 && cands.len() == 2 {
                if cands[1].total_value > saved_value {
                    save_solution(&cands[1].to_solution())?;
                }
                return Ok(());
            }
            sw.warm_basins_t41
        }
        // T39: track2's internal cpool basins + track_t39's ILS candidates (+ adaptive escalation when `t39_adaptive >= 2`)
        (1000, b) if b <= 17 => {
            let cpool_take = if sw.t39_adaptive >= 3 { sw.t39_cpool.max(3) } else { sw.t39_cpool };
            if cpool_take > 0 {
                cands.extend(t39_buffer.iter().take(cpool_take).filter_map(|sol| flat_if_feasible(challenge, &q_flat, sol)));
            }
            let probe_sols = if let Some(sols) = precomputed_t39_sols.take() {
                sols
            } else {
                let eff_rounds = if sw.t39_rounds == 4 { 2 } else { sw.t39_rounds as u64 };
                let t39_hp = hp_with(hyperparameters, &[("n_perturbation_rounds", eff_rounds)]);
                track_t39::collect_candidate_solutions(challenge, &t39_hp)
            };
            let probe_states: Vec<FlatState> = probe_sols
                .iter()
                .filter_map(|sol| flat_if_feasible(challenge, &q_flat, sol))
                .collect();
            let probe_best = probe_states.iter().map(|s| s.total_value).max().unwrap_or(saved_value);
            cands.extend(probe_states);

            // Adaptive T39 escalation: if the fast `t39_rounds` probe strictly beat `packrun` (as on T39.s10 and OOS T39.s20)
            // or if `sw.t39_adaptive >= 4` (escalate whenever probe differs from `packrun`), run a deeper `track_t39` schedule!
            let should_escalate = match sw.t39_adaptive {
                2 | 3 => probe_best > saved_value,
                4 => probe_best != saved_value || cands.len() >= 2,
                _ => false,
            };
            if should_escalate && sw.t39_rounds < 42 {
                let deep_rounds = if sw.t39_adaptive == 3 { 49 } else { 36 };
                let deep_hp = hp_with(hyperparameters, &[("n_perturbation_rounds", deep_rounds)]);
                cands.extend(
                    track_t39::collect_candidate_solutions(challenge, &deep_hp)
                        .iter()
                        .filter_map(|sol| flat_if_feasible(challenge, &q_flat, sol)),
                );
            }
            // Velocity exit (t39_rounds 4): the probe does not beat packrun, so Step 3 is skipped.
            if sw.t39_rounds == 4 && probe_best <= saved_value {
                return Ok(());
            }
            sw.warm_basins_t39
        }
        // T40: track3's distinct xr_pool + core_stage alt basins
        (1000, _) => {
            if sw.t40_elites > 0 {
                cands.extend(t40_buffer.iter().take(sw.t40_elites).filter_map(|sol| flat_if_feasible(challenge, &q_flat, sol)));
            }
            sw.warm_basins_t40
        }
        // T42: the fused buffer, or v12's second track4 call with use_hub_pair 0 (satchel's hub_pair_n=0 trajectory)
        (5000, b) if b <= 17 => {
            if t42_fused {
                cands.extend(t42_buffer.iter().take(sw.t42_elites).filter_map(|sol| flat_if_feasible(challenge, &q_flat, sol)));
            } else {
                cands.extend(track_flat(track4::solve as TrackFn, challenge, &q_flat, &hp_number("use_hub_pair", 0)).ok().flatten());
            }
            sw.warm_basins_t42
        }
        // T43: track5's distinct elite basins, best first
        (5000, _) => {
            cands.extend(t43_buffer.iter().take(sw.t43_elites).filter_map(|sol| flat_if_feasible(challenge, &q_flat, sol)));
            sw.warm_basins_t43
        }
        _ => 0,
    };
    if warm_basins > 0 {
        cands.extend(solve_multi_warm_b2_elites(challenge, &q_flat, &sum_q, min_q, warm_basins.min(4)));
    }
    if sw.spectral_basins > 0 {
        cands.extend(solve_spectral_synergy_basins(challenge, &q_flat, min_q, sw.spectral_basins, sw.quad_bnb_k));
    }

    if n == 5000 && !t42 {
        let starts = restricted_core_seeds(challenge, &q_flat, min_q, &cands);
        for st in starts { push_distinct(&mut cands, st); }
    }

    // Step 3: frontier-bounded polish of every distinct candidate, relinking, closure CoreBnb, tabu escape, final polish.
    let (mut polish_args, mut bursts, steps) = frontier_bounds(n, budget, sw.fast_polish == 1);
    if sw.b2_k12 > 0 {
        polish_args.0 = sw.b2_k12;
    }
    if sw.b2_k22 > 0 {
        polish_args.1 = sw.b2_k22;
    }
    if sw.b2_k33 > 0 {
        polish_args.2 = sw.b2_k33;
    }
    if sw.b2_core_dp > 0 {
        polish_args.3 = sw.b2_core_dp;
    }
    if sw.tabu_bursts_override < 99 {
        bursts = sw.tabu_bursts_override;
    }
    // When `t39_adaptive >= 1`, enable fast 3-burst tabu escape specifically on T39 (N=1000, <15 ms/seed)
    // even when `tabu_bursts_override == 0` disables slow N=5000 tabu bursts!
    if t39 && sw.t39_adaptive >= 1 && bursts == 0 {
        bursts = 3;
    } else if t42 && bursts == 0 {
        bursts = 2;
    }
    let polish = |st: &mut FlatState| {
        polish_stage_b2_hybrid(st, challenge, &q_flat, min_q, polish_args.0, polish_args.1, polish_args.2, polish_args.3);
    };
    let noop_polish = |_st: &mut FlatState| {};
    let mut elites: Vec<FlatState> = Vec::new();
    if (t42 && sw.t42_dedup >= 3) || (n == 5000 && !t42 && sw.t43_dedup >= 5) {
        // On T42 and T43, candidates are already VND+DP polished
        polish_candidates(cands, &mut elites, sw.polish_top_k, &noop_polish);
    } else {
        polish_candidates(cands, &mut elites, sw.polish_top_k, &polish);
    }
    let mut best_state = elites.iter().max_by_key(|e| e.total_value).cloned().expect("packrun is always a candidate");
    let pre_relink_val = best_state.total_value;
    if elites.len() >= 2 && sw.relink_limit > 0 {
        let relink_quad_bnb_k = if sw.quad_bnb_k > 0 { sw.quad_bnb_k } else if n == 5000 { 24 } else { 0 };
        keep_if_not_worse(&mut best_state, |st| {
            recombine_and_relink_elites_lookahead(
                st,
                &elites,
                challenge,
                &q_flat,
                min_q,
                sw.relink_limit,
                sw.lookahead_relink,
                relink_quad_bnb_k,
                sw.bucket_compound,
            )
        });
    }
    if sw.closure_bnb_k > 0 {
        let work_cap = if n <= 1000 { 450_000 } else { 300_000 };
        keep_if_not_worse(&mut best_state, |st| {
            if closure_core_bnb_refine(st, &elites, challenge, &q_flat, min_q, sw.closure_bnb_k, work_cap) {
                polish(st);
            }
        });
    }
    if sw.bucket_compound > 0 {
        keep_if_not_worse(&mut best_state, |st| {
            while bucket_compound_22_refine(st, challenge, &q_flat, min_q, sw.bucket_compound) {
                polish(st);
            }
        });
    }
    if sw.quad_bnb_k > 0 {
        keep_if_not_worse(&mut best_state, |st| {
            while exact_quad_bnb_refine(st, challenge, &q_flat, min_q, sw.quad_bnb_k) {
                polish(st);
            }
        });
    }
    // On T42 with t42_dedup >= 3, tabu escape only improves when relinking moved best_state into a new basin.
    let run_tabu = bursts > 0 && (!t42 || sw.t42_dedup < 3 || best_state.total_value > pre_relink_val);
    if run_tabu {
        let pre_tabu_val = best_state.total_value;
        keep_if_not_worse(&mut best_state, |st| tabu_basin_escape(st, challenge, &q_flat, min_q, seed_u64, bursts, steps));
        if best_state.total_value > pre_tabu_val {
            keep_if_not_worse(&mut best_state, |st| polish(st));
        }
    }

    // Step 4: Deep Iterated Local Search (ILS) on Tracks 42 & 43
    let run_ils = (t42 && sw.ils_rounds_t42 > 0) || (n == 5000 && !t42 && sw.ils_rounds_t43 > 0);
    if run_ils {
        let rounds = if t42 { sw.ils_rounds_t42 } else { sw.ils_rounds_t43 };
        keep_if_not_worse(&mut best_state, |st| {
            iterated_local_search_refine(st, challenge, &q_flat, min_q, seed_u64, rounds, t42);
        });
    }

    if best_state.total_value > saved_value {
        save_solution(&best_state.to_solution())?;
    }
    Ok(())
}

pub fn help() {
    println!("axiom_pulsar: stage-b2 hybrid & deep ILS quadratic knapsack solver");
}
