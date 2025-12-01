use serde_json::{Map, Value};
use std::cmp::Ordering;
use tig_challenges::knapsack::*;

#[derive(Clone, Copy)]
struct Params {
    core_half_dp: usize,
    core_half_ls: usize,
    n_maxils: usize,
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
            core_half_dp: 30,
            core_half_ls: 35,
            n_maxils,
        }
    }
}
impl Default for Params {
    fn default() -> Self {
        Self::for_problem_size(400)  
    }
}

fn fast_interaction_scores(ch: &Challenge, out: &mut [i32]) {
    let n = ch.num_items;

    for i in 0..n {
        let mut pos_sum: i32 = 0;
        let mut strong_count = 0;
        let mut very_strong_count = 0;
        let mut best_synergy: i32 = 0;

        for &x in &ch.interaction_values[i] {
            if x > 0 {
                pos_sum += x;
                if x >= 7 {
                    very_strong_count += 1;
                }
                if x >= 5 {
                    strong_count += 1;
                }
                if x > best_synergy {
                    best_synergy = x;
                }
            }
        }

        let hub_bonus = if very_strong_count >= 8 {
            pos_sum / 6
        } else if strong_count >= 10 {
            pos_sum / 8
        } else if strong_count >= 6 {
            pos_sum / 12
        } else if strong_count >= 3 {
            pos_sum / 20
        } else {
            0
        };

        let synergy_quality = if best_synergy >= 9 {
            best_synergy * best_synergy / 8
        } else if best_synergy >= 8 {
            best_synergy * best_synergy / 10
        } else if best_synergy >= 7 {
            best_synergy * best_synergy / 14
        } else if best_synergy >= 6 {
            best_synergy * best_synergy / 20
        } else if best_synergy >= 5 {
            best_synergy / 2
        } else {
            0
        };

        let clique_depth_bonus = if very_strong_count >= 4 {
            very_strong_count * 2
        } else if very_strong_count >= 2 {
            very_strong_count
        } else {
            0
        };

        out[i] = ch.values[i] as i32 + pos_sum + hub_bonus + synergy_quality + clique_depth_bonus;
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

fn beam_search_construction(challenge: &Challenge, beam_width: usize) -> Vec<usize> {
    let n = challenge.num_items;
    let top_n = 600.min(n);
    
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        let ra = (challenge.values[a] as f64) / (challenge.weights[a] as f64);
        let rb = (challenge.values[b] as f64) / (challenge.weights[b] as f64);
        rb.partial_cmp(&ra).unwrap_or(Ordering::Equal)
    });
    order.truncate(top_n);
    
    let mut beams: Vec<(Vec<usize>, i64, i64)> = vec![(Vec::new(), 0, 0)];
    
    for &item in &order {
        let mut candidates: Vec<(Vec<usize>, i64, i64)> = Vec::new();
        
        for (beam, val, weight) in &beams {
            candidates.push((beam.clone(), *val, *weight));
            
            let item_weight = challenge.weights[item] as i64;
            let item_value = challenge.values[item] as i64;
            if weight + item_weight <= challenge.max_weight as i64 {
                let mut new_beam = beam.clone();
                new_beam.push(item);
                candidates.push((new_beam, val + item_value, weight + item_weight));
            }
        }
        
        candidates.sort_by(|a, b| b.1.cmp(&a.1));
        candidates.truncate(beam_width);
        beams = candidates;
    }
    
    beams.into_iter()
        .max_by_key(|(_, val, _)| *val)
        .map(|(beam, _, _)| beam)
        .unwrap_or_default()
}

fn build_initial_solution(state: &mut State, order_scores: &[i32]) {
    let n = state.ch.num_items;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        let da = (order_scores[a] as f64) / (state.ch.weights[a] as f64);
        let db = (order_scores[b] as f64) / (state.ch.weights[b] as f64);
        db.partial_cmp(&da).unwrap_or(Ordering::Equal)
    });

    let mut added = vec![false; n];
    let top_items = 25.min(n);

    for i in 0..top_items.min(15) {
        if added[order[i]] {
            continue;
        }
        let item_i = order[i];
        let w_i = state.ch.weights[item_i] as i64;

        let mut best_pair: Option<(usize, i64)> = None;
        for j in (i + 1)..top_items.min(i + 12) {
            if added[order[j]] {
                continue;
            }
            let item_j = order[j];
            let w_j = state.ch.weights[item_j] as i64;
            if w_i + w_j > state.capacity() {
                continue;
            }

            let synergy = state.ch.interaction_values[item_i][item_j] as i64;
            if synergy >= 5 {
                let quality = if synergy >= 8 {
                    synergy * synergy * synergy
                } else if synergy >= 7 {
                    synergy * synergy * 2
                } else {
                    synergy * synergy
                };
                if best_pair.map_or(true, |(_, q)| quality > q) {
                    best_pair = Some((item_j, quality));
                }
            }
        }

        if let Some((partner, _)) = best_pair {
            let w = w_i + state.ch.weights[partner] as i64;
            if state.total_weight + w <= state.capacity() {
                state.add_item(item_i);
                state.add_item(partner);
                added[item_i] = true;
                added[partner] = true;
            }
        }
    }

    for &i in &order {
        if added[i] {
            continue;
        }
        let w = state.ch.weights[i] as i64;
        if state.total_weight + w <= state.capacity() {
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

fn apply_best_swap_diff_increase_windowed(
    state: &mut State,
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

    let mut rem = (m * 12 / 100).max(1);
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

    let (mut best_unused, _) = build_ls_windows(state, params.core_half_ls);

    let mut added = Vec::new();
    for &cand in &best_unused {
        let w = state.ch.weights[cand] as i64;
        if w <= state.slack() && (state.contrib[cand] as i64) > 0 {
            state.add_item(cand);
            added.push(cand);
        }
    }

    for &item in &added {
        State::remove_from_vec(&mut best_unused, item);
    }

    if state.slack() > 0 && best_unused.len() >= 2 {
        let slack = state.slack();
        let limit = 10.min(best_unused.len());

        if best_unused.len() >= 3 && slack > 0 {
            let triple_limit = 8.min(best_unused.len());
            for i in 0..triple_limit {
                let i1 = best_unused[i];
                let w1 = state.ch.weights[i1] as i64;
                if w1 > slack {
                    continue;
                }

                for j in (i + 1)..triple_limit {
                    let i2 = best_unused[j];
                    let w2 = state.ch.weights[i2] as i64;
                    if w1 + w2 > slack {
                        continue;
                    }

                    for k in (j + 1)..triple_limit {
                        let i3 = best_unused[k];
                        let w3 = state.ch.weights[i3] as i64;
                        if w1 + w2 + w3 > slack {
                            continue;
                        }

                        let syn12 = state.ch.interaction_values[i1][i2] as i64;
                        let syn13 = state.ch.interaction_values[i1][i3] as i64;
                        let syn23 = state.ch.interaction_values[i2][i3] as i64;

                        let total_syn = syn12 + syn13 + syn23;
                        let min_edge = syn12.min(syn13).min(syn23);

                        let quality_score = if min_edge >= 8 && total_syn >= 27 {
                            total_syn * total_syn / 10
                        } else if min_edge >= 7 && total_syn >= 24 {
                            total_syn * total_syn / 15
                        } else if min_edge >= 6 && total_syn >= 21 {
                            total_syn * total_syn / 20
                        } else if min_edge >= 5 && total_syn >= 18 {
                            total_syn
                        } else {
                            0
                        };

                        if quality_score > 0 {
                            let val = (state.contrib[i1] as i64)
                                + (state.contrib[i2] as i64)
                                + (state.contrib[i3] as i64)
                                + total_syn;
                            if val > 0 {
                                state.add_item(i1);
                                state.add_item(i2);
                                state.add_item(i3);
        return;
    }
                        }
                    }
                }
            }
        }

        for i in 0..limit {
            let item1 = best_unused[i];
            let w1 = state.ch.weights[item1] as i64;
            if w1 > slack {
                continue;
            }

            for j in (i + 1)..limit {
                let item2 = best_unused[j];
                let w2 = state.ch.weights[item2] as i64;
                if w1 + w2 <= slack {
                    let c1 = state.contrib[item1] as i64;
                    let c2 = state.contrib[item2] as i64;
                    let syn = state.ch.interaction_values[item1][item2] as i64;
                    if c1 + c2 + syn > 0 {
                        state.add_item(item1);
                        state.add_item(item2);
                        return;
                    }
                }
            }
        }
    }
}

fn apply_best_swap_21(state: &mut State, best_unused: &[usize], worst_used: &[usize]) -> bool {
    let mut best: Option<(usize, usize, usize, i64)> = None;
    
    for i in 0..worst_used.len() {
        for j in (i+1)..worst_used.len() {
            let rm1 = worst_used[i];
            let rm2 = worst_used[j];
            let weight_freed = state.ch.weights[rm1] as i64 + state.ch.weights[rm2] as i64;
            let value_lost = state.contrib[rm1] as i64 + state.contrib[rm2] as i64
                - state.ch.interaction_values[rm1][rm2] as i64;
            
            for &cand in best_unused {
                if state.ch.weights[cand] as i64 <= weight_freed {
                    let value_gain = state.contrib[cand] as i64;
                    let delta = value_gain - value_lost;
                    if delta > 0 && best.map_or(true, |(_, _, _, d)| delta > d) {
                        best = Some((rm1, rm2, cand, delta));
                    }
                }
            }
        }
    }
    
    if let Some((rm1, rm2, cand, _)) = best {
        state.remove_item(rm1);
        state.remove_item(rm2);
        state.add_item(cand);
        return true;
    }
    false
}

fn apply_best_swap_12(state: &mut State, best_unused: &[usize], worst_used: &[usize]) -> bool {
    let mut best: Option<(usize, usize, usize, i64)> = None;
    
    for &rm in worst_used {
        let weight_freed = state.ch.weights[rm] as i64;
        let value_lost = state.contrib[rm] as i64;
        
        for i in 0..best_unused.len() {
            for j in (i+1)..best_unused.len() {
                let cand1 = best_unused[i];
                let cand2 = best_unused[j];
                let weight_needed = state.ch.weights[cand1] as i64 + state.ch.weights[cand2] as i64;
                
                if weight_needed <= weight_freed + state.slack() {
                    let value_gain = state.contrib[cand1] as i64 + state.contrib[cand2] as i64
                        + state.ch.interaction_values[cand1][cand2] as i64;
                    let delta = value_gain - value_lost;
                    if delta > 0 && best.map_or(true, |(_, _, _, d)| delta > d) {
                        best = Some((rm, cand1, cand2, delta));
                    }
                }
            }
        }
    }
    
    if let Some((rm, cand1, cand2, _)) = best {
        state.remove_item(rm);
        state.add_item(cand1);
        state.add_item(cand2);
        return true;
    }
    false
}

fn local_search_vnd(state: &mut State, params: &Params) {
    let (mut best_unused, mut worst_used) = build_ls_windows(state, params.core_half_ls);
    loop {
        if apply_best_swap11_equal_windowed(state, &mut best_unused, &mut worst_used) {
            continue;
        }
        if apply_best_swap_diff_increase_windowed(state, &mut best_unused, &mut worst_used) {
            continue;
        }
        break;
    }
}

fn local_search_extended_vnd(state: &mut State, params: &Params) {
    let (mut best_unused, mut worst_used) = build_ls_windows(state, params.core_half_ls);
    let mut iterations = 0;
    loop {
        if apply_best_swap11_equal_windowed(state, &mut best_unused, &mut worst_used) {
            iterations += 1;
            continue;
        }
        if apply_best_swap_diff_increase_windowed(state, &mut best_unused, &mut worst_used) {
            iterations += 1;
            continue;
        }        
        
        if iterations >= 3 {
            if apply_best_swap_21(state, &best_unused, &worst_used) {
                let (new_best_unused, new_worst_used) = build_ls_windows(state, params.core_half_ls);
                best_unused = new_best_unused;
                worst_used = new_worst_used;
                iterations = 0;
                continue;
            }
            if apply_best_swap_12(state, &best_unused, &worst_used) {
                let (new_best_unused, new_worst_used) = build_ls_windows(state, params.core_half_ls);
                best_unused = new_best_unused;
                worst_used = new_worst_used;
                iterations = 0;
                continue;
            }
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
    
    let use_extended_vnd = hyperparameters
        .as_ref()
        .and_then(|hp| hp.get("use_extended_vnd"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true); 
    
    let use_beam = hyperparameters
        .as_ref()
        .and_then(|hp| hp.get("use_beam_search"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true); 
    
    let mut state = State::new_empty(challenge);
    
    if use_beam {
        let beam_items = beam_search_construction(challenge, 10);
        for &item in &beam_items {
            state.add_item(item);
        }
    } else {
        let mut build_scores = vec![0i32; n];
        fast_interaction_scores(challenge, &mut build_scores);
        build_initial_solution(&mut state, &build_scores);
    }
    
    if use_extended_vnd {
        local_search_extended_vnd(&mut state, &params);
    } else {
        local_search_vnd(&mut state, &params);
    }

    for _it in 0..params.n_maxils {
        let prev_sel = state.selected_items();
        let prev_val = state.total_value;
        let prev_weight = state.total_weight;
        let prev_contrib = state.contrib.clone();

        let target = integer_core_target(challenge, &state.contrib, params.core_half_dp);
        apply_dp_target_via_ops(&mut state, &target);
        if use_extended_vnd {
            local_search_extended_vnd(&mut state, &params);
        } else {
            local_search_vnd(&mut state, &params);
        }

        if state.total_value > prev_val {
            continue;
        }

        state.restore_snapshot(&prev_sel, &prev_contrib, prev_val, prev_weight);
        strategic_perturb_and_rebuild(&mut state, &params);
        if use_extended_vnd {
            local_search_extended_vnd(&mut state, &params);
        } else {
            local_search_vnd(&mut state, &params);
        }

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
