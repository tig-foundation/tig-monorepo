use anyhow::Result;
use serde_json::{Map, Value};
use std::cmp::Ordering;
use tig_challenges::knapsack::*;

#[derive(Clone, Copy)]
struct Params {
    diff_lim: usize,
    core_half_dp: usize,
    core_half_ls: usize,
    n_maxils: usize,
    polish_k: usize,
}
impl Params {
    fn for_problem_size(num_items: usize) -> Self {
        let n_maxils = if num_items <= 600 {
            3
        } else if num_items <= 800 {
            4
        } else {
            5
        };

        Self {
            diff_lim: 4,
            core_half_dp: 30,
            core_half_ls: 35,
            n_maxils,
            polish_k: 10,
        }
    }
}
impl Default for Params {
    fn default() -> Self {
        Self::for_problem_size(400)
    }
}

fn centrality_boosted_scores(ch: &Challenge, out: &mut [i32]) {
    let n = ch.num_items;
    for i in 0..n {
        let mut pos_sum: i32 = 0;
        for &x in &ch.interaction_values[i] {
            if x > 0 {
                pos_sum += x;
            }
        }
        out[i] = ch.values[i] as i32 + pos_sum;
    }
}

struct State<'a> {
    ch: &'a Challenge,
    selected_bit: Vec<bool>,
    contrib: Vec<i32>,
    total_value: i64,
    total_weight: i64,
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
        }
    }
    fn selected_items(&self) -> Vec<usize> {
        (0..self.ch.num_items)
            .filter(|&i| self.selected_bit[i])
            .collect()
    }
    #[inline]
    fn capacity(&self) -> i64 {
        self.ch.max_weight as i64
    }
    #[inline]
    fn slack(&self) -> i64 {
        self.capacity() - self.total_weight
    }
    fn add_item(&mut self, i: usize) {
        self.selected_bit[i] = true;
        self.total_value += self.contrib[i] as i64;
        self.total_weight += self.ch.weights[i] as i64;
        let n = self.ch.num_items;
        for k in 0..n {
            self.contrib[k] += self.ch.interaction_values[k][i] as i32;
        }
    }
    fn remove_item(&mut self, j: usize) {
        self.total_value -= self.contrib[j] as i64;
        self.total_weight -= self.ch.weights[j] as i64;
        let n = self.ch.num_items;
        for k in 0..n {
            self.contrib[k] -= self.ch.interaction_values[k][j] as i32;
        }
        self.selected_bit[j] = false;
    }
    fn replace_item(&mut self, rm: usize, cand: usize) {
        let w_c = self.ch.weights[cand] as i64;
        if self.slack() >= w_c {
            self.add_item(cand);
            self.remove_item(rm);
        } else {
            self.remove_item(rm);
            self.add_item(cand);
        }
    }
    fn restore_snapshot(
        &mut self,
        snapshot_sel: &[usize],
        snapshot_contrib: &[i32],
        snap_value: i64,
        snap_weight: i64,
    ) {
        self.selected_bit.fill(false);
        for &i in snapshot_sel {
            self.selected_bit[i] = true;
        }
        self.contrib.clear();
        self.contrib.extend_from_slice(snapshot_contrib);
        self.total_value = snap_value;
        self.total_weight = snap_weight;
    }
    #[inline]
    fn remove_from_vec(v: &mut Vec<usize>, x: usize) {
        if let Some(pos) = v.iter().position(|&y| y == x) {
            v.swap_remove(pos);
        }
    }
}

fn savings_construction(state: &mut State, order_scores: &[i32]) {
    let n = state.ch.num_items;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        let da = (order_scores[a] as f64) / (state.ch.weights[a] as f64);
        let db = (order_scores[b] as f64) / (state.ch.weights[b] as f64);
        db.partial_cmp(&da).unwrap_or(Ordering::Equal)
    });

    let top_k = if n <= 600 {
        64
    } else if n <= 1200 {
        96
    } else if n <= 2000 {
        112
    } else {
        128
    };
    let pool: Vec<usize> = order.iter().copied().take(top_k.min(n)).collect();

    let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
    for ai in 0..pool.len() {
        let i = pool[ai];
        let wi = state.ch.weights[i] as i64;
        if wi <= 0 {
            continue;
        }
        let ci = state.contrib[i] as i64;
        for aj in (ai + 1)..pool.len() {
            let j = pool[aj];
            let wj = state.ch.weights[j] as i64;
            let wsum = wi + wj;
            if wsum <= 0 {
                continue;
            }
            let cj = state.contrib[j] as i64;
            let syn = state.ch.interaction_values[i][j] as i64;
            let gain = ci + cj + syn;
            if gain <= 0 {
                continue;
            }
            let ratio = (gain as f64) / (wsum as f64);
            let syn_density = (syn.max(0) as f64) / ((wsum as f64).sqrt() + 1.0);
            let score = ratio + 0.05 * syn_density;
            pairs.push((i, j, score));
        }
    }
    pairs.sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));

    let mut used = vec![false; n];
    for &(i, j, _) in &pairs {
        if used[i] || used[j] {
            continue;
        }
        let wi = state.ch.weights[i] as i64;
        let wj = state.ch.weights[j] as i64;
        if wi + wj <= state.slack() {
            state.add_item(i);
            state.add_item(j);
            used[i] = true;
            used[j] = true;
        }
    }

    for &i in &order {
        if state.selected_bit[i] {
            continue;
        }
        let w = state.ch.weights[i] as i64;
        if w <= state.slack() && (state.contrib[i] as i64) > 0 {
            state.add_item(i);
        }
    }
}

fn integer_core_target(ch: &Challenge, contrib: &[i32], core_half_dp: usize) -> Vec<usize> {
    let n = ch.num_items;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        let da = (contrib[a] as f64) / (ch.weights[a] as f64);
        let db = (contrib[b] as f64) / (ch.weights[b] as f64);
        db.partial_cmp(&da).unwrap_or(Ordering::Equal)
    });
    let mut pref_w: i64 = 0;
    let mut break_idx: usize = order.len().saturating_sub(1);
    for (pos, &i) in order.iter().enumerate() {
        let w = ch.weights[i] as i64;
        if pref_w + w > ch.max_weight as i64 {
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
    let rem_cap_full = ((ch.max_weight as i64) - used_locked).max(0) as usize;

    fn gcd(mut a: usize, mut b: usize) -> usize {
        while b != 0 {
            let t = a % b;
            a = b;
            b = t;
        }
        if a == 0 {
            1
        } else {
            a
        }
    }

    let mut g: usize = 0;
    let mut sum_core_w_full: usize = 0;
    for &it in core {
        let w = ch.weights[it] as usize;
        sum_core_w_full += w;
        g = if g == 0 { w } else { gcd(g, w) };
    }
    if g == 0 {
        g = 1;
    }

    let cap_bound_full = rem_cap_full.min(sum_core_w_full);
    let myw = cap_bound_full / g;
    let myk = core.len();

    let mut dp: Vec<i64> = vec![i64::MIN / 4; myw + 1];
    dp[0] = 0;
    let mut choose: Vec<u8> = vec![0u8; myk * (myw + 1)];
    let mut w_hi: usize = 0;
    for (t, &it) in core.iter().enumerate() {
        let wt = (ch.weights[it] as usize) / g;
        if wt == 0 || wt > myw {
            continue;
        }
        let val = contrib[it] as i64;
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
    let mut selected: Vec<usize> = locked.to_vec();
    let mut w_star = (0..=myw).max_by_key(|&w| dp[w]).unwrap_or(0);
    for t in (0..myk).rev() {
        let it = core[t];
        let wt = (ch.weights[it] as usize) / g;
        if wt <= w_star && wt > 0 && choose[t * (myw + 1) + w_star] == 1 {
            selected.push(it);
            w_star -= wt;
        }
    }
    selected.sort_unstable();
    selected
}

fn apply_dp_target_via_ops(state: &mut State, target_sel: &[usize]) {
    let n = state.ch.num_items;
    let mut in_target = vec![false; n];
    for &i in target_sel {
        in_target[i] = true;
    }
    let mut to_remove: Vec<usize> = Vec::new();
    for i in 0..n {
        if state.selected_bit[i] && !in_target[i] {
            to_remove.push(i);
        }
    }
    let mut to_add: Vec<usize> = Vec::new();
    for &i in target_sel {
        if !state.selected_bit[i] {
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

fn build_ls_windows(state: &State, core_half_ls: usize) -> (Vec<usize>, Vec<usize>) {
    let n = state.ch.num_items;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        let da = (state.contrib[a] as f64) / (state.ch.weights[a] as f64);
        let db = (state.contrib[b] as f64) / (state.ch.weights[b] as f64);
        db.partial_cmp(&da).unwrap_or(Ordering::Equal)
    });
    let mut best_unused = Vec::with_capacity(core_half_ls);
    for &i in &order {
        if !state.selected_bit[i] {
            best_unused.push(i);
            if best_unused.len() >= core_half_ls {
                break;
            }
        }
    }
    let mut worst_used = Vec::with_capacity(core_half_ls);
    for &i in order.iter().rev() {
        if state.selected_bit[i] {
            worst_used.push(i);
            if worst_used.len() >= core_half_ls {
                break;
            }
        }
    }
    (best_unused, worst_used)
}

fn apply_best_add_windowed(
    state: &mut State,
    best_unused: &mut Vec<usize>,
    worst_used: &mut Vec<usize>,
) -> bool {
    let slack = state.slack();
    if slack <= 0 {
        return false;
    }
    let mut best: Option<(usize, i64)> = None;
    for &cand in &*best_unused {
        let w = state.ch.weights[cand] as i64;
        if w > slack {
            continue;
        }
        let delta = state.contrib[cand] as i64;
        if delta > 0 && best.map_or(true, |(_, bd)| delta > bd) {
            best = Some((cand, delta));
        }
    }
    if let Some((cand, _)) = best {
        state.add_item(cand);
        State::remove_from_vec(best_unused, cand);
        worst_used.push(cand);
        true
    } else {
        false
    }
}

fn apply_best_swap11_equal_windowed(
    state: &mut State,
    best_unused: &mut Vec<usize>,
    worst_used: &mut Vec<usize>,
) -> bool {
    let mut best: Option<(usize, usize, i64)> = None;
    for &rm in &*worst_used {
        let w_rm = state.ch.weights[rm];
        for &cand in &*best_unused {
            if state.ch.weights[cand] != w_rm {
                continue;
            }
            let delta = (state.contrib[cand] as i64)
                - (state.contrib[rm] as i64)
                - (state.ch.interaction_values[cand][rm] as i64);
            if delta > 0 && best.map_or(true, |(_, _, bd)| delta > bd) {
                best = Some((cand, rm, delta));
            }
        }
    }
    if let Some((cand, rm, _)) = best {
        state.replace_item(rm, cand);
        State::remove_from_vec(worst_used, rm);
        best_unused.push(rm);
        State::remove_from_vec(best_unused, cand);
        worst_used.push(cand);
        true
    } else {
        false
    }
}

fn apply_best_swap_diff_reduce_windowed(
    state: &mut State,
    _params: &Params,
    best_unused: &mut Vec<usize>,
    worst_used: &mut Vec<usize>,
) -> bool {
    let mut best: Option<(usize, usize, i64)> = None;
    for &rm in &*worst_used {
        let w_rm = state.ch.weights[rm] as i64;
        for &cand in &*best_unused {
            let w_c = state.ch.weights[cand] as i64;
            if w_c >= w_rm {
                continue;
            }
            let delta = (state.contrib[cand] as i64)
                - (state.contrib[rm] as i64)
                - (state.ch.interaction_values[cand][rm] as i64);
            if delta > 0 && best.map_or(true, |(_, _, bd)| delta > bd) {
                best = Some((cand, rm, delta));
            }
        }
    }
    if let Some((cand, rm, _)) = best {
        state.replace_item(rm, cand);
        State::remove_from_vec(worst_used, rm);
        best_unused.push(rm);
        State::remove_from_vec(best_unused, cand);
        worst_used.push(cand);
        true
    } else {
        false
    }
}

fn apply_best_swap_diff_increase_windowed(
    state: &mut State,
    _params: &Params,
    best_unused: &mut Vec<usize>,
    worst_used: &mut Vec<usize>,
) -> bool {
    if state.slack() <= 0 {
        return false;
    }
    let mut best: Option<(usize, usize, f64)> = None;
    for &rm in &*worst_used {
        let w_rm = state.ch.weights[rm] as i64;
        for &cand in &*best_unused {
            let w_c = state.ch.weights[cand] as i64;
            if w_c <= w_rm {
                continue;
            }
            let dw = (w_c - w_rm) as i64;
            if state.slack() < dw {
                continue;
            }
            let delta = (state.contrib[cand] as i64)
                - (state.contrib[rm] as i64)
                - (state.ch.interaction_values[cand][rm] as i64);
            if delta > 0 {
                let ratio = (delta as f64) / (dw as f64);
                if best.map_or(true, |(_, _, br)| ratio > br) {
                    best = Some((cand, rm, ratio));
                }
            }
        }
    }
    if let Some((cand, rm, _)) = best {
        state.replace_item(rm, cand);
        State::remove_from_vec(worst_used, rm);
        best_unused.push(rm);
        State::remove_from_vec(best_unused, cand);
        worst_used.push(cand);
        true
    } else {
        false
    }
}

fn polish_once(state: &mut State, params: &Params) {
    let n = state.ch.num_items;
    let k = params.polish_k.min(n).max(16);
    let mut unsel: Vec<usize> = Vec::new();
    let mut sel: Vec<usize> = Vec::new();
    for i in 0..n {
        if state.selected_bit[i] {
            sel.push(i);
        } else {
            unsel.push(i);
        }
    }
    let mut unused_top: Vec<usize> = Vec::new();
    if !unsel.is_empty() {
        let kk = k.min(unsel.len());
        if kk < unsel.len() {
            let (left, _, _) = unsel.select_nth_unstable_by(kk - 1, |&a, &b| {
                let ra = (state.contrib[a] as f64) / (state.ch.weights[a] as f64);
                let rb = (state.contrib[b] as f64) / (state.ch.weights[b] as f64);
                rb.partial_cmp(&ra).unwrap_or(Ordering::Equal)
            });
            unused_top = left.to_vec();
            unused_top.sort_unstable_by(|&a, &b| {
                let ra = (state.contrib[a] as f64) / (state.ch.weights[a] as f64);
                let rb = (state.contrib[b] as f64) / (state.ch.weights[b] as f64);
                rb.partial_cmp(&ra).unwrap_or(Ordering::Equal)
            });
        } else {
            unused_top = unsel;
            unused_top.sort_unstable_by(|&a, &b| {
                let ra = (state.contrib[a] as f64) / (state.ch.weights[a] as f64);
                let rb = (state.contrib[b] as f64) / (state.ch.weights[b] as f64);
                rb.partial_cmp(&ra).unwrap_or(Ordering::Equal)
            });
        }
        if unused_top.len() > kk {
            unused_top.truncate(kk);
        }
    }
    let mut used_worst: Vec<usize> = Vec::new();
    if !sel.is_empty() {
        let kk = k.min(sel.len());
        if kk < sel.len() {
            let (left, _, _) = sel.select_nth_unstable_by(kk - 1, |&a, &b| {
                let ra = (state.contrib[a] as f64) / (state.ch.weights[a] as f64);
                let rb = (state.contrib[b] as f64) / (state.ch.weights[b] as f64);
                ra.partial_cmp(&rb).unwrap_or(Ordering::Equal)
            });
            used_worst = left.to_vec();
            used_worst.sort_unstable_by(|&a, &b| {
                let ra = (state.contrib[a] as f64) / (state.ch.weights[a] as f64);
                let rb = (state.contrib[b] as f64) / (state.ch.weights[b] as f64);
                ra.partial_cmp(&rb).unwrap_or(Ordering::Equal)
            });
        } else {
            used_worst = sel;
            used_worst.sort_unstable_by(|&a, &b| {
                let ra = (state.contrib[a] as f64) / (state.ch.weights[a] as f64);
                let rb = (state.contrib[b] as f64) / (state.ch.weights[b] as f64);
                ra.partial_cmp(&rb).unwrap_or(Ordering::Equal)
            });
        }
        if used_worst.len() > kk {
            used_worst.truncate(kk);
        }
    }

    let mut best_add: Option<(usize, i64)> = None;
    let slack0 = state.slack();
    if slack0 > 0 {
        for &i in &unused_top {
            let w = state.ch.weights[i] as i64;
            if w <= slack0 {
                let d = state.contrib[i] as i64;
                if d > 0 && best_add.map_or(true, |(_, bd)| d > bd) {
                    best_add = Some((i, d));
                }
            }
        }
    }
    if let Some((i, _)) = best_add {
        state.add_item(i);
        return;
    }

    let mut best_swap: Option<(usize, usize, i64)> = None;
    for &rm in &used_worst {
        for &cand in &unused_top {
            if state.ch.weights[cand] != state.ch.weights[rm] {
                continue;
            }
            let d = (state.contrib[cand] as i64)
                - (state.contrib[rm] as i64)
                - (state.ch.interaction_values[cand][rm] as i64);
            if d > 0 && best_swap.map_or(true, |(_, _, bd)| d > bd) {
                best_swap = Some((cand, rm, d));
            }
        }
    }
    if let Some((cand, rm, _)) = best_swap {
        state.replace_item(rm, cand);
        return;
    }

    let mut best_swap_red: Option<(usize, usize, i64)> = None;
    for &rm in &used_worst {
        let w_rm = state.ch.weights[rm] as i64;
        for &cand in &unused_top {
            let w_c = state.ch.weights[cand] as i64;
            if w_c >= w_rm {
                continue;
            }
            let dw = (w_rm - w_c) as usize;
            if dw == 0 || dw > params.diff_lim {
                continue;
            }
            let d = (state.contrib[cand] as i64)
                - (state.contrib[rm] as i64)
                - (state.ch.interaction_values[cand][rm] as i64);
            if d > 0 && best_swap_red.map_or(true, |(_, _, bd)| d > bd) {
                best_swap_red = Some((cand, rm, d));
            }
        }
    }
    if let Some((cand, rm, _)) = best_swap_red {
        state.replace_item(rm, cand);
        return;
    }

    if state.slack() > 0 {
        let mut best_swap_inc: Option<(usize, usize, f64, i64)> = None;
        for &rm in &used_worst {
            let w_rm = state.ch.weights[rm] as i64;
            for &cand in &unused_top {
                let w_c = state.ch.weights[cand] as i64;
                if w_c <= w_rm {
                    continue;
                }
                let dw = w_c - w_rm;
                if dw as usize > params.diff_lim {
                    continue;
                }
                if state.slack() < dw {
                    continue;
                }
                let d = (state.contrib[cand] as i64)
                    - (state.contrib[rm] as i64)
                    - (state.ch.interaction_values[cand][rm] as i64);
                if d > 0 {
                    let r = (d as f64) / (dw as f64);
                    if best_swap_inc.map_or(true, |(_, _, br, bd)| d > bd || (d == bd && r > br)) {
                        best_swap_inc = Some((cand, rm, r, d));
                    }
                }
            }
        }
        if let Some((cand, rm, _, _)) = best_swap_inc {
            state.replace_item(rm, cand);
            return;
        }
    }
}

fn large_neighborhood_destroy_repair(state: &mut State, params: &Params) {
    let sel = state.selected_items();
    let m = sel.len();
    if m == 0 {
        return;
    }

    let mut scored: Vec<(usize, f64)> = Vec::with_capacity(m);
    for &i in &sel {
        let ratio = (state.contrib[i] as f64) / (state.ch.weights[i] as f64);
        scored.push((i, ratio));
    }
    scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    let mut to_remove: Vec<usize> = Vec::new();
    let seed_cnt = (m / 12).max(1).min(6);
    for s in 0..seed_cnt.min(scored.len()) {
        to_remove.push(scored[s].0);
    }
    for &r in to_remove.clone().iter() {
        let mut best_p: Option<(usize, i32)> = None;
        for &p in &sel {
            if p == r {
                continue;
            }
            if to_remove.contains(&p) {
                continue;
            }
            let v = state.ch.interaction_values[r][p];
            if v < 0 && best_p.map_or(true, |(_, bv)| v < bv) {
                best_p = Some((p, v));
            }
        }
        if let Some((p, _)) = best_p {
            to_remove.push(p);
            if to_remove.len() >= 12 {
                break;
            }
        }
    }
    for &x in &to_remove {
        if state.selected_bit[x] {
            state.remove_item(x);
        }
    }

    let mut iter = 0;
    while state.slack() > 0 {
        iter += 1;
        let (mut best_unused, _) = build_ls_windows(state, params.core_half_ls);
        if best_unused.is_empty() {
            break;
        }
        let pool_len = best_unused.len().min(12);
        let slack = state.slack();

        let mut best_pair: Option<(usize, usize, f64)> = None;
        for ai in 0..pool_len {
            let i = best_unused[ai];
            let wi = state.ch.weights[i] as i64;
            if wi > slack {
                continue;
            }
            let ci = state.contrib[i] as i64;
            for aj in (ai + 1)..pool_len {
                let j = best_unused[aj];
                let wj = state.ch.weights[j] as i64;
                let wsum = wi + wj;
                if wsum > slack {
                    continue;
                }
                let cj = state.contrib[j] as i64;
                let syn = state.ch.interaction_values[i][j] as i64;
                let delta = ci + cj + syn;
                if delta <= 0 {
                    continue;
                }
                let ratio = (delta as f64) / (wsum as f64);
                if best_pair.map_or(true, |(_, _, br)| ratio > br) {
                    best_pair = Some((i, j, ratio));
                }
            }
        }

        let mut best_single: Option<(usize, f64)> = None;
        for ai in 0..pool_len {
            let i = best_unused[ai];
            let wi = state.ch.weights[i] as i64;
            if wi > slack {
                continue;
            }
            let ci = state.contrib[i] as i64;
            if ci <= 0 {
                continue;
            }
            let row = &state.ch.interaction_values[i];
            let mut t1 = 0i32;
            let mut t2 = 0i32;
            for aj in 0..pool_len {
                if aj == ai {
                    continue;
                }
                let j = best_unused[aj];
                let v = row[j];
                if v > t1 {
                    t2 = t1;
                    t1 = v;
                } else if v > t2 {
                    t2 = v;
                }
            }
            let pot = ((ci + ((t1 as i64 + t2 as i64) / 2)) as f64) / (wi as f64);
            if best_single.map_or(true, |(_, bs)| pot > bs) {
                best_single = Some((i, pot));
            }
        }

        let mut acted = false;
        if let Some((i, j, pr)) = best_pair {
            if let Some((si, sr)) = best_single {
                if pr >= sr {
                    state.add_item(i);
                    state.add_item(j);
                    acted = true;
                } else {
                    state.add_item(si);
                    acted = true;
                }
            } else {
                state.add_item(i);
                state.add_item(j);
                acted = true;
            }
        } else if let Some((si, _)) = best_single {
            state.add_item(si);
            acted = true;
        }
        if !acted || iter >= 20 {
            break;
        }
    }

    let (best_unused, _) = build_ls_windows(state, params.core_half_ls);
    for &cand in &best_unused {
        let w = state.ch.weights[cand] as i64;
        if w <= state.slack() && (state.contrib[cand] as i64) > 0 {
            state.add_item(cand);
        }
    }
}

fn ejection_chain_vnd(state: &mut State, params: &Params) {
    let mut _iter = 0usize;
    loop {
        _iter += 1;

        let (mut best_unused, mut worst_used) = build_ls_windows(state, params.core_half_ls);

        if apply_best_add_windowed(state, &mut best_unused, &mut worst_used) {
            continue;
        }

        if state.slack() > 0 {
            let mut best_pair: Option<(usize, usize, i64)> = None;
            let slack = state.slack();
            let bu_len = best_unused.len().min(24);
            for a_i in 0..bu_len {
                let i = best_unused[a_i];
                let wi = state.ch.weights[i] as i64;
                if wi > slack {
                    continue;
                }
                let ci = state.contrib[i] as i64;
                for a_j in (a_i + 1)..bu_len {
                    let j = best_unused[a_j];
                    let wj = state.ch.weights[j] as i64;
                    let wsum = wi + wj;
                    if wsum > slack {
                        continue;
                    }
                    let cj = state.contrib[j] as i64;
                    let syn = state.ch.interaction_values[i][j] as i64;
                    let delta = ci + cj + syn;
                    if delta > 0 && best_pair.map_or(true, |(_, _, bd)| delta > bd) {
                        best_pair = Some((i, j, delta));
                    }
                }
            }
            if let Some((i, j, _)) = best_pair {
                state.add_item(i);
                state.add_item(j);
                continue;
            }
        }

        let slack = state.slack();
        let pool_u = best_unused.len().min(12);
        let pool_r = worst_used.len().min(12);
        let mut best_triple: Option<(usize, usize, usize, i64)> = None;
        if pool_u >= 2 && pool_r >= 1 {
            for a_i in 0..pool_u {
                let i = best_unused[a_i];
                let wi = state.ch.weights[i] as i64;
                let ci = state.contrib[i] as i64;
                for a_j in (a_i + 1)..pool_u {
                    let j = best_unused[a_j];
                    let wj = state.ch.weights[j] as i64;
                    let cj = state.contrib[j] as i64;
                    let wsum = wi + wj;
                    if wsum <= slack {
                        continue;
                    }
                    let need = wsum - slack;
                    let syn = state.ch.interaction_values[i][j] as i64;
                    for r_idx in 0..pool_r {
                        let rm = worst_used[r_idx];
                        let w_rm = state.ch.weights[rm] as i64;
                        if w_rm < need {
                            continue;
                        }
                        let loss_rm = state.contrib[rm] as i64;
                        let penalty = (state.ch.interaction_values[i][rm] as i64)
                            + (state.ch.interaction_values[j][rm] as i64);
                        let delta = ci + cj + syn - loss_rm - penalty;
                        if delta > 0 && best_triple.map_or(true, |(_, _, _, bd)| delta > bd) {
                            best_triple = Some((i, j, rm, delta));
                        }
                    }
                }
            }
        }
        if let Some((i, j, rm, _)) = best_triple {
            state.remove_item(rm);
            state.add_item(i);
            state.add_item(j);
            continue;
        }

        if apply_best_swap_diff_reduce_windowed(state, params, &mut best_unused, &mut worst_used) {
            continue;
        }
        if apply_best_swap11_equal_windowed(state, &mut best_unused, &mut worst_used) {
            continue;
        }
        if apply_best_swap_diff_increase_windowed(state, params, &mut best_unused, &mut worst_used)
        {
            continue;
        }

        break;
    }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let n = challenge.num_items;
    let params = Params::for_problem_size(n);
    let mut build_scores = vec![0i32; n];
    centrality_boosted_scores(challenge, &mut build_scores);

    let mut state = State::new_empty(challenge);
    savings_construction(&mut state, &build_scores);
    ejection_chain_vnd(&mut state, &params);
    polish_once(&mut state, &params);

    for _it in 0..params.n_maxils {
        let prev_sel = state.selected_items();
        let prev_val = state.total_value;
        let prev_weight = state.total_weight;
        let prev_contrib = state.contrib.clone();

        let target = integer_core_target(challenge, &state.contrib, params.core_half_dp);
        apply_dp_target_via_ops(&mut state, &target);
        ejection_chain_vnd(&mut state, &params);
        polish_once(&mut state, &params);

        if state.total_value > prev_val {
            continue;
        }

        state.restore_snapshot(&prev_sel, &prev_contrib, prev_val, prev_weight);
        large_neighborhood_destroy_repair(&mut state, &params);
        ejection_chain_vnd(&mut state, &params);
        polish_once(&mut state, &params);

        if state.total_value <= prev_val {
            state.restore_snapshot(&prev_sel, &prev_contrib, prev_val, prev_weight);
            break;
        }
    }

    let mut items = state.selected_items();
    items.sort_unstable();
    let _ = save_solution(&Solution { items });
    Ok(())
}

pub fn help() {
    println!("No help information available.");
}
