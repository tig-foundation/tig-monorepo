use super::params::Params;
use super::refinement::micro_qkp_refinement;
use super::types::{rebuild_windows, State, DIFF_LIM};

fn apply_best_add_windowed(state: &mut State) -> bool {
    const SUP_ADD_BONUS: i64 = 40;

    let slack = state.slack();
    if slack == 0 {
        return false;
    }

    let mut best: Option<(usize, i64)> = None;

    for (bw, items) in &state.core_bins {
        if *bw > slack {
            break;
        }
        for &cand in items {
            if state.selected_bit[cand] {
                continue;
            }
            let delta = state.contrib[cand];
            if delta <= 0 {
                continue;
            }
            let s = (delta as i64) + (state.support[cand] as i64) * SUP_ADD_BONUS;
            if best.map_or(true, |(_, bs)| s > bs) {
                best = Some((cand, s));
            }
        }
    }

    if best.is_none() {
        let lim = if state.ch.num_items >= 2500 {
            state.window_rejected.len().min(384)
        } else {
            state.window_rejected.len()
        };

        for &cand in &state.window_rejected[..lim] {
            if state.selected_bit[cand] {
                continue;
            }
            let w = state.ch.weights[cand];
            if w > slack {
                continue;
            }
            let delta = state.contrib[cand];
            if delta <= 0 {
                continue;
            }
            let s = (delta as i64) + (state.support[cand] as i64) * SUP_ADD_BONUS;
            if best.map_or(true, |(_, bs)| s > bs) {
                best = Some((cand, s));
            }
        }
    }

    if let Some((cand, _)) = best {
        state.add_item(cand);
        true
    } else {
        false
    }
}

fn apply_best_add_neigh_global(state: &mut State) -> bool {
    const BETA_NUM: i64 = 3;
    const BETA_DEN: i64 = 20;
    const SUP_ADD_BONUS: i64 = 40;

    let slack = state.slack();
    if slack == 0 {
        return false;
    }

    let neigh = match state.neigh {
        Some(ng) => ng,
        None => return false,
    };

    let n = state.ch.num_items;
    if n == 0 {
        return false;
    }

    let edge_lim: usize = if n >= 4500 { 16000 } else if n >= 2500 { 12000 } else { 9000 };
    let node_lim: usize = if n >= 4500 { 56 } else if n >= 2500 { 64 } else { 72 };

    let start = (((state.total_value as u64) as usize)
        ^ ((state.total_weight as usize).wrapping_mul(911)))
        % n;
    let mut step = (n / 97).max(1);
    step |= 1;

    let mut best: Option<(usize, i64, i32)> = None;
    let mut scanned_edges: usize = 0;
    let mut scanned_nodes: usize = 0;

    let mut idx = start;
    let mut tries: usize = 0;

    while tries < n && scanned_nodes < node_lim && scanned_edges < edge_lim {
        if state.selected_bit[idx] {
            scanned_nodes += 1;
            let row = unsafe { neigh.get_unchecked(idx) };
            for &(cj, _vv) in row.iter() {
                scanned_edges += 1;
                if scanned_edges > edge_lim {
                    break;
                }

                let cand = cj as usize;
                if state.selected_bit[cand] {
                    continue;
                }

                let w_u = state.ch.weights[cand];
                if w_u == 0 || w_u > slack {
                    continue;
                }

                let delta = state.contrib[cand];
                if delta <= 0 {
                    continue;
                }

                let w = (w_u as i64).max(1);
                let c = delta as i64;
                let tot = state.total_interactions[cand];
                let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot);
                let s = (adj * 1000) / w + (state.support[cand] as i64) * SUP_ADD_BONUS;

                if best.map_or(true, |(_, bs, bd)| s > bs || (s == bs && delta > bd)) {
                    best = Some((cand, s, delta));
                }
            }
        }

        idx += step;
        if idx >= n {
            idx -= n;
        }
        tries += 1;
    }

    if let Some((cand, _, _)) = best {
        state.add_item(cand);
        true
    } else {
        false
    }
}

fn apply_best_replace12_windowed(state: &mut State, used: &[usize]) -> bool {
    let cap = state.ch.max_weight;
    if used.is_empty() {
        return false;
    }
    let slack0 = state.slack();

    let mut add_pool: Vec<usize> = Vec::with_capacity(state.window_core.len() + 64);
    add_pool.extend_from_slice(&state.window_core);
    let extra = state.window_rejected.len().min(64);
    add_pool.extend_from_slice(&state.window_rejected[..extra]);
    add_pool.retain(|&i| !state.selected_bit[i]);

    add_pool.sort_unstable_by(|&a, &b| {
        let wa = (state.ch.weights[a] as i64).max(1);
        let wb = (state.ch.weights[b] as i64).max(1);
        let sa = (state.contrib[a] as i64 * 1000) / wa + (state.support[a] as i64) * 80;
        let sb = (state.contrib[b] as i64 * 1000) / wb + (state.support[b] as i64) * 80;
        sb.cmp(&sa).then_with(|| b.cmp(&a))
    });
    if add_pool.len() > 28 {
        add_pool.truncate(28);
    }
    if add_pool.len() < 2 {
        return false;
    }

    let mut rm: Vec<usize> = used.to_vec();
    rm.sort_unstable_by(|&a, &b| {
        let wa = (state.ch.weights[a] as i64).max(1);
        let wb = (state.ch.weights[b] as i64).max(1);
        let sa = (state.contrib[a] as i64 * 1000) / wa + (state.support[a] as i64) * 120;
        let sb = (state.contrib[b] as i64 * 1000) / wb + (state.support[b] as i64) * 120;
        sa.cmp(&sb).then_with(|| a.cmp(&b))
    });
    if rm.len() > 16 {
        rm.truncate(16);
    }

    let mut best: Option<(usize, usize, usize, i64)> = None;

    for &r in &rm {
        if !state.selected_bit[r] {
            continue;
        }
        let wr = state.ch.weights[r];
        let avail = wr.saturating_add(slack0);

        for x in 0..add_pool.len() {
            let a = add_pool[x];
            let wa = state.ch.weights[a];
            if wa == 0 || wa > avail {
                continue;
            }
            for y in (x + 1)..add_pool.len() {
                let b = add_pool[y];
                let wb = state.ch.weights[b];
                if wb == 0 || wa + wb > avail {
                    continue;
                }

                let new_w = state
                    .total_weight
                    .saturating_sub(wr)
                    .saturating_add(wa)
                    .saturating_add(wb);
                if new_w > cap {
                    continue;
                }

                let delta = (state.contrib[a] as i64)
                    + (state.contrib[b] as i64)
                    - (state.contrib[r] as i64)
                    - (state.ch.interaction_values[a][r] as i64)
                    - (state.ch.interaction_values[b][r] as i64)
                    + (state.ch.interaction_values[a][b] as i64);

                if delta > 0 && best.map_or(true, |(_, _, _, bd)| delta > bd) {
                    best = Some((r, a, b, delta));
                }
            }
        }
    }

    if let Some((r, a, b, _)) = best {
        state.remove_item(r);
        state.add_item(a);
        state.add_item(b);
        true
    } else {
        false
    }
}

fn apply_best_replace21_windowed(state: &mut State, used: &[usize]) -> bool {
    let cap = state.ch.max_weight;
    if used.len() < 2 {
        return false;
    }

    let mut rm: Vec<usize> = used.to_vec();
    rm.sort_unstable_by(|&a, &b| {
        let ca = state.contrib[a] as i64;
        let cb = state.contrib[b] as i64;
        let wa = state.ch.weights[a] as i64;
        let wb = state.ch.weights[b] as i64;
        (ca * wb).cmp(&(cb * wa))
    });
    if rm.len() > 14 {
        rm.truncate(14);
    }

    let mut add_pool: Vec<usize> = Vec::with_capacity(state.window_core.len() + 48);
    add_pool.extend_from_slice(&state.window_core);
    let extra = state.window_rejected.len().min(48);
    add_pool.extend_from_slice(&state.window_rejected[..extra]);
    add_pool.retain(|&i| !state.selected_bit[i] && state.contrib[i] > 0);
    add_pool.sort_unstable_by(|&a, &b| {
        let ca = state.contrib[a] as i64;
        let cb = state.contrib[b] as i64;
        let wa = state.ch.weights[a] as i64;
        let wb = state.ch.weights[b] as i64;
        (cb * wa).cmp(&(ca * wb))
    });
    if add_pool.len() > 28 {
        add_pool.truncate(28);
    }
    if add_pool.is_empty() {
        return false;
    }

    let mut best: Option<(usize, usize, usize, i64)> = None;

    for &cand in &add_pool {
        let wc = state.ch.weights[cand];
        if wc == 0 {
            continue;
        }

        for x in 0..rm.len() {
            let a = rm[x];
            let wa = state.ch.weights[a];
            for y in (x + 1)..rm.len() {
                let b = rm[y];
                let wb = state.ch.weights[b];

                let new_w = state
                    .total_weight
                    .saturating_sub(wa)
                    .saturating_sub(wb)
                    .saturating_add(wc);
                if new_w > cap {
                    continue;
                }

                let delta = (state.contrib[cand] as i64)
                    - (state.ch.interaction_values[cand][a] as i64)
                    - (state.ch.interaction_values[cand][b] as i64)
                    - (state.contrib[a] as i64)
                    - (state.contrib[b] as i64)
                    + (state.ch.interaction_values[a][b] as i64);

                if delta > 0 && best.map_or(true, |(_, _, _, bd)| delta > bd) {
                    best = Some((cand, a, b, delta));
                }
            }
        }
    }

    if let Some((cand, a, b, _)) = best {
        state.remove_item(a);
        state.remove_item(b);
        if !state.selected_bit[cand] && state.total_weight + state.ch.weights[cand] <= cap {
            state.add_item(cand);
        }
        true
    } else {
        false
    }
}

#[inline]
fn apply_best_swap_diff_reduce_windowed_cached(state: &mut State, used: &[usize]) -> bool {
    let mut best: Option<(usize, usize, i32)> = None;
    for &rm in used {
        let w_rm = state.ch.weights[rm];
        if w_rm == 0 {
            continue;
        }
        let w_min = w_rm.saturating_sub(DIFF_LIM as u32);
        for (bw, items) in &state.core_bins {
            if *bw >= w_rm {
                break;
            }
            if *bw < w_min {
                continue;
            }
            for &cand in items {
                if state.selected_bit[cand] {
                    continue;
                }
                let delta = state.contrib[cand]
                    - state.contrib[rm]
                    - state.ch.interaction_values[cand][rm];
                if delta > 0 && best.map_or(true, |(_, _, bd)| delta > bd) {
                    best = Some((cand, rm, delta));
                }
            }
        }
    }
    if let Some((cand, rm, _)) = best {
        state.replace_item(rm, cand);
        true
    } else {
        false
    }
}

#[inline]
fn apply_best_swap_diff_increase_windowed_cached(state: &mut State, used: &[usize]) -> bool {
    let slack = state.slack();
    if slack == 0 {
        return false;
    }
    let mut best: Option<(usize, usize, f64)> = None;
    for &rm in used {
        let w_rm = state.ch.weights[rm];
        let max_dw = (DIFF_LIM as u32).min(slack);
        let w_max = w_rm.saturating_add(max_dw);
        for (bw, items) in &state.core_bins {
            if *bw <= w_rm {
                continue;
            }
            if *bw > w_max {
                break;
            }
            let dw = *bw - w_rm;
            if dw > slack {
                break;
            }
            for &cand in items {
                if state.selected_bit[cand] {
                    continue;
                }
                let delta = state.contrib[cand]
                    - state.contrib[rm]
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
    if let Some((cand, rm, _)) = best {
        state.replace_item(rm, cand);
        true
    } else {
        false
    }
}

fn apply_best_swap_neigh_any(state: &mut State, used: &[usize]) -> bool {
    if used.is_empty() {
        return false;
    }
    let cap = state.ch.max_weight;

    let neigh = match state.neigh {
        Some(ng) => ng,
        None => return false,
    };

    let mut best: Option<(usize, usize, i64, i64)> = None;

    for &rm in used {
        if !state.selected_bit[rm] {
            continue;
        }
        let wrm = state.ch.weights[rm];
        let row = unsafe { neigh.get_unchecked(rm) };

        for &(cj, vv) in row.iter() {
            let cand = cj as usize;
            if state.selected_bit[cand] {
                continue;
            }
            let wc = state.ch.weights[cand];
            if wc == 0 {
                continue;
            }

            if (state.total_weight as u64) + (wc as u64) > (cap as u64) + (wrm as u64) {
                continue;
            }

            let delta = (state.contrib[cand] as i64) - (state.contrib[rm] as i64) - (vv as i64);
            if delta <= 0 {
                continue;
            }

            let score: i64 = if wc == wrm {
                delta * 1_000_000
            } else if wc < wrm {
                delta * 1000 + (wrm as i64 - wc as i64)
            } else {
                let dw = (wc - wrm) as i64;
                (delta * 1000) / dw.max(1)
            };

            if best.map_or(true, |(_, _, bs, bd)| score > bs || (score == bs && delta > bd)) {
                best = Some((cand, rm, score, delta));
            }
        }
    }

    if let Some((cand, rm, _, _)) = best {
        state.replace_item(rm, cand);
        true
    } else {
        false
    }
}

fn apply_best_swap_frontier_global(state: &mut State, used: &[usize]) -> bool {
    let cap = state.ch.max_weight;
    if used.is_empty() {
        return false;
    }

    let neigh = match state.neigh {
        Some(ng) => ng,
        None => return false,
    };

    let n = state.ch.num_items;
    if n == 0 {
        return false;
    }

    const BETA_NUM: i64 = 3;
    const BETA_DEN: i64 = 20;
    const TOP_CAND: usize = 48;
    const TOP_RM: usize = 32;

    let mut rm: Vec<usize> = used.to_vec();
    rm.sort_unstable_by(|&a, &b| {
        let wa = (state.ch.weights[a] as i64).max(1);
        let wb = (state.ch.weights[b] as i64).max(1);
        let sa = (state.contrib[a] as i64 * 1000) / wa + (state.support[a] as i64) * 140;
        let sb = (state.contrib[b] as i64 * 1000) / wb + (state.support[b] as i64) * 140;
        sa.cmp(&sb).then_with(|| a.cmp(&b))
    });
    if rm.len() > TOP_RM {
        rm.truncate(TOP_RM);
    }

    let mut max_rm_w: u32 = 0;
    for &i in used {
        let w = state.ch.weights[i];
        if w > max_rm_w {
            max_rm_w = w;
        }
    }
    if max_rm_w == 0 {
        return false;
    }

    let slack0 = state.slack();

    let edge_lim: usize = if n >= 4500 { 22000 } else if n >= 2500 { 16000 } else { 9000 };
    let node_lim: usize = if n >= 4500 { 64 } else if n >= 2500 { 72 } else { 84 };

    let start = (((state.total_value as u64) as usize)
        ^ ((state.total_weight as usize).wrapping_mul(911)))
        % n;
    let mut step = (n / 97).max(1);
    step |= 1;

    let mut cand_list: Vec<(i64, usize)> = Vec::with_capacity(TOP_CAND);

    let push_top_unique = |list: &mut Vec<(i64, usize)>, s: i64, idx: usize| {
        for &(_, j) in list.iter() {
            if j == idx {
                return;
            }
        }
        if list.len() < TOP_CAND {
            list.push((s, idx));
            return;
        }
        let mut worst_pos = 0usize;
        let mut worst_s = list[0].0;
        for t in 1..list.len() {
            if list[t].0 < worst_s {
                worst_s = list[t].0;
                worst_pos = t;
            }
        }
        if s > worst_s {
            list[worst_pos] = (s, idx);
        }
    };

    let mut scanned_edges: usize = 0;
    let mut scanned_nodes: usize = 0;
    let mut idx = start;
    let mut tries: usize = 0;

    while tries < n && scanned_nodes < node_lim && scanned_edges < edge_lim {
        if state.selected_bit[idx] {
            scanned_nodes += 1;
            let row = unsafe { neigh.get_unchecked(idx) };
            for &(cj, _vv) in row.iter() {
                scanned_edges += 1;
                if scanned_edges > edge_lim {
                    break;
                }

                let cand = cj as usize;
                if state.selected_bit[cand] {
                    continue;
                }

                let wc = state.ch.weights[cand];
                if wc == 0 {
                    continue;
                }

                if wc > slack0.saturating_add(max_rm_w) {
                    continue;
                }

                let c = state.contrib[cand] as i64;
                if c <= 0 {
                    continue;
                }

                let tot = state.total_interactions[cand];
                let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot);
                let s = (adj * 1000) / (wc as i64).max(1) + (state.support[cand] as i64) * 60;

                push_top_unique(&mut cand_list, s, cand);
            }
        }

        idx += step;
        if idx >= n {
            idx -= n;
        }
        tries += 1;
    }

    if cand_list.is_empty() {
        return false;
    }

    let mut best: Option<(usize, usize, i64, i64)> = None;

    for &(_s0, cand) in &cand_list {
        let wc = state.ch.weights[cand];
        if wc == 0 {
            continue;
        }

        for &r in &rm {
            if !state.selected_bit[r] {
                continue;
            }
            let wr = state.ch.weights[r];

            if (state.total_weight as u64) + (wc as u64) > (cap as u64) + (wr as u64) {
                continue;
            }

            let inter = state.ch.interaction_values[cand][r] as i64;
            let delta = (state.contrib[cand] as i64) - (state.contrib[r] as i64) - inter;
            if delta <= 0 {
                continue;
            }

            let score: i64 = if wc == wr {
                delta * 1_000_000
            } else if wc < wr {
                delta * 1000 + (wr as i64 - wc as i64)
            } else {
                let dw = (wc - wr) as i64;
                (delta * 1000) / dw.max(1)
            };

            if best.map_or(true, |(_, _, bs, bd)| score > bs || (score == bs && delta > bd)) {
                best = Some((cand, r, score, delta));
            }
        }
    }

    if let Some((cand, r, _, _)) = best {
        state.replace_item(r, cand);
        true
    } else {
        false
    }
}

pub fn local_search_vnd(state: &mut State, params: &Params) {
    let mut iterations = 0;
    let n = state.ch.num_items;
    let max_iterations = if n >= 4500 {
        180
    } else if n >= 3000 {
        260
    } else if n >= 1000 {
        300
    } else {
        80
    };
    let mut used: Vec<usize> = Vec::new();
    let mut micro_used = false;

    let mut frontier_swap_tries: usize = 0;
    let max_frontier_swaps: usize = params.max_frontier_swaps_override.unwrap_or(
        if n >= 2500 { 0 } else if n >= 1500 { 1 } else { 2 }
    );

    let mut dirty_window = false;
    let mut n_rebuilds = 0usize;
    let max_rebuilds: usize = if n >= 2500 { 1 } else { 2 };

    loop {
        iterations += 1;
        if iterations > max_iterations {
            break;
        }

        if apply_best_add_windowed(state) {
            continue;
        }

        if (iterations & 3) == 0 && apply_best_add_neigh_global(state) {
            dirty_window = true;
            continue;
        }

        used.clear();
        for &i in &state.window_core {
            if state.selected_bit[i] {
                used.push(i);
            }
        }

        let extra = state.window_locked.len().min(24);
        let start = state.window_locked.len().saturating_sub(extra);
        for &i in state.window_locked[start..].iter() {
            if state.selected_bit[i] {
                used.push(i);
            }
        }

        if apply_best_swap_diff_reduce_windowed_cached(state, &used) {
            continue;
        }
        if apply_best_swap_diff_increase_windowed_cached(state, &used) {
            continue;
        }

        if apply_best_swap_neigh_any(state, &used) {
            dirty_window = true;
            continue;
        }

        if frontier_swap_tries < max_frontier_swaps {
            frontier_swap_tries += 1;
            if apply_best_swap_frontier_global(state, &used) {
                dirty_window = true;
                continue;
            }
        }

        if apply_best_replace12_windowed(state, &used) {
            continue;
        }
        if apply_best_replace21_windowed(state, &used) {
            continue;
        }

        if dirty_window && n_rebuilds < max_rebuilds {
            n_rebuilds += 1;
            dirty_window = false;
            rebuild_windows(state);
            continue;
        }

        if !micro_used {
            micro_used = true;
            if dirty_window {
                rebuild_windows(state);
                dirty_window = false;
            }
            let old = state.total_value;
            micro_qkp_refinement(state);
            if state.total_value > old {
                rebuild_windows(state);
                continue;
            }
        }

        break;
    }
}
