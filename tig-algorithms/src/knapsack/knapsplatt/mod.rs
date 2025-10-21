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

#[inline]
fn weight_of(ch: &Challenge, items: &[usize]) -> i64 {
    items.iter().map(|&i| ch.weights[i] as i64).sum()
}

fn round0_scores(ch: &Challenge, out: &mut [i32]) {
    let n = ch.difficulty.num_items;
    for i in 0..n {
        let row_sum: i32 = ch.interaction_values[i].iter().sum();
        out[i] = ch.values[i] as i32 + row_sum;
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
        let n = ch.difficulty.num_items;
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
        (0..self.ch.difficulty.num_items)
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
        let n = self.ch.difficulty.num_items;
        for k in 0..n {
            self.contrib[k] += self.ch.interaction_values[k][i] as i32;
        }
    }
    fn remove_item(&mut self, j: usize) {
        self.total_value -= self.contrib[j] as i64;
        self.total_weight -= self.ch.weights[j] as i64;
        let n = self.ch.difficulty.num_items;
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
        snapshot_contrib: Vec<i32>,
        snap_value: i64,
    ) {
        self.selected_bit.fill(false);
        for &i in snapshot_sel {
            self.selected_bit[i] = true;
        }
        self.contrib = snapshot_contrib;
        self.total_value = snap_value;
        self.total_weight = weight_of(self.ch, snapshot_sel);
    }
    #[inline]
    fn remove_from_vec(v: &mut Vec<usize>, x: usize) {
        if let Some(pos) = v.iter().position(|&y| y == x) {
            v.swap_remove(pos);
        }
    }
}

fn build_initial_solution(state: &mut State, order_scores: &[i32]) {
    let n = state.ch.difficulty.num_items;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        let da = (order_scores[a] as f64) / (state.ch.weights[a] as f64);
        let db = (order_scores[b] as f64) / (state.ch.weights[b] as f64);
        db.partial_cmp(&da).unwrap_or(Ordering::Equal)
    });
    for &i in &order {
        let w = state.ch.weights[i] as i64;
        if state.total_weight + w <= state.capacity() {
            state.add_item(i);
        }
    }
}

fn integer_core_target(ch: &Challenge, contrib: &[i32], core_half_dp: usize) -> Vec<usize> {
    let n = ch.difficulty.num_items;
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
    let rem_cap = ((ch.max_weight as i64) - used_locked).max(0) as usize;
    let myw = rem_cap;
    let myk = core.len();
    let mut dp: Vec<i64> = vec![i64::MIN / 4; myw + 1];
    dp[0] = 0;
    let mut choose: Vec<u8> = vec![0u8; myk * (myw + 1)];
    let mut w_hi: usize = 0;
    for (t, &it) in core.iter().enumerate() {
        let wt = ch.weights[it] as usize;
        if wt > myw {
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
        let wt = ch.weights[it] as usize;
        if wt <= w_star && choose[t * (myw + 1) + w_star] == 1 {
            selected.push(it);
            w_star -= wt;
        }
    }
    selected.sort_unstable();
    selected
}

fn apply_dp_target_via_ops(state: &mut State, target_sel: &[usize]) {
    let n = state.ch.difficulty.num_items;
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
    let n = state.ch.difficulty.num_items;
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
    params: &Params,
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
            let dw = (w_rm - w_c) as usize;
            if dw == 0 || dw > params.diff_lim {
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
    params: &Params,
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
            if dw as usize > params.diff_lim {
                continue;
            }
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
    let n = state.ch.difficulty.num_items;
    let k = params.polish_k.min(n).max(16);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by(|&a, &b| {
        let ra = (state.contrib[a] as f64) / (state.ch.weights[a] as f64);
        let rb = (state.contrib[b] as f64) / (state.ch.weights[b] as f64);
        rb.partial_cmp(&ra).unwrap_or(Ordering::Equal)
    });
    let mut unused_top: Vec<usize> = Vec::new();
    for &i in &idx {
        if !state.selected_bit[i] {
            unused_top.push(i);
            if unused_top.len() >= k {
                break;
            }
        }
    }
    let mut used_worst: Vec<usize> = Vec::new();
    for &i in idx.iter().rev() {
        if state.selected_bit[i] {
            used_worst.push(i);
            if used_worst.len() >= k {
                break;
            }
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

fn strategic_perturb_and_rebuild(state: &mut State, params: &Params) {
    let sel = state.selected_items();
    let m = sel.len();
    if m == 0 {
        return;
    }
    let mut bad = sel.clone();
    bad.sort_unstable_by(|&a, &b| {
        let ra = (state.contrib[a] as f64) / (state.ch.weights[a] as f64);
        let rb = (state.contrib[b] as f64) / (state.ch.weights[b] as f64);
        ra.partial_cmp(&rb).unwrap_or(Ordering::Equal)
    });
    let mut rem = (m / 10).max(1);
    if rem > 10 {
        rem = 10;
    }
    rem = rem.min(bad.len());
    for i in 0..rem {
        let r = bad[i];
        if state.selected_bit[r] {
            state.remove_item(r);
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

fn local_search_vnd(state: &mut State, params: &Params) {
    let (mut best_unused, mut worst_used) = build_ls_windows(state, params.core_half_ls);
    loop {
        if apply_best_add_windowed(state, &mut best_unused, &mut worst_used) {
            continue;
        }

        if state.slack() > 0 {
            let mut best_pair: Option<(usize, usize, i64)> = None;
            let slack = state.slack();
            let bu_len = best_unused.len();
            for a_i in 0..bu_len {
                let i = best_unused[a_i];
                let wi = state.ch.weights[i] as i64;
                if wi >= slack {
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
                State::remove_from_vec(&mut best_unused, i);
                State::remove_from_vec(&mut best_unused, j);
                worst_used.push(i);
                worst_used.push(j);
                continue;
            }
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
    let n = challenge.difficulty.num_items;
    let params = Params::for_problem_size(n);
    let mut build_scores = vec![0i32; n];
    round0_scores(challenge, &mut build_scores);

    let mut state = State::new_empty(challenge);
    build_initial_solution(&mut state, &build_scores);
    local_search_vnd(&mut state, &params);
    polish_once(&mut state, &params);

    for _it in 0..params.n_maxils {
        let prev_sel = state.selected_items();
        let prev_val = state.total_value;
        let prev_contrib = state.contrib.clone();

        let target = integer_core_target(challenge, &state.contrib, params.core_half_dp);
        apply_dp_target_via_ops(&mut state, &target);
        local_search_vnd(&mut state, &params);
        polish_once(&mut state, &params);

        if state.total_value > prev_val {
            continue;
        }

        state.restore_snapshot(&prev_sel, prev_contrib.clone(), prev_val);
        strategic_perturb_and_rebuild(&mut state, &params);
        local_search_vnd(&mut state, &params);
        polish_once(&mut state, &params);

        if state.total_value <= prev_val {
            state.restore_snapshot(&prev_sel, prev_contrib, prev_val);
            break;
        }
    }

    let mut items = state.selected_items();
    items.sort_unstable();
    let _ = save_solution(&Solution { items });
    Ok(())
}
