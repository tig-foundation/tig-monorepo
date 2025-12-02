use super::state::State;
use super::params::Params;

/// Collect currently used items inside the core window
fn used_in_core(state: &State) -> Vec<usize> {
    state.window_core.iter().copied().filter(|&i| state.selected_bit[i]).collect()
}

fn apply_best_add_windowed(state: &mut State) -> bool {
    let slack = state.slack();
    if slack == 0 { return false; }
    let mut best: Option<(usize, i32)> = None;
    for (bw, items) in &state.core_bins {
        if *bw > slack { break; }
        for &cand in items {
            if state.selected_bit[cand] { continue; }
            let delta = state.contrib[cand];
            if delta > 0 && best.map_or(true, |(_, bd)| delta > bd) {
                best = Some((cand, delta));
            }
        }
    }
    if let Some((cand, _)) = best { state.add_item(cand); true }
    else { false }
}

fn apply_best_swap11_equal_windowed(state: &mut State) -> bool {
    let used = used_in_core(state);
    let mut best: Option<(usize, usize, i32)> = None;
    for &rm in &used {
        let w_rm = state.ch.weights[rm];
        if let Ok(idx) = state.core_bins.binary_search_by_key(&w_rm, |(w, _)| *w) {
            let items = &state.core_bins[idx].1;
            for &cand in items {
                if state.selected_bit[cand] { continue; }
                let delta = state.contrib[cand] - state.contrib[rm]
                    - state.ch.interaction_values[cand][rm];
                if delta > 0 && best.map_or(true, |(_, _, bd)| delta > bd) {
                    best = Some((cand, rm, delta));
                }
            }
        }
    }
    if let Some((cand, rm, _)) = best { state.replace_item(rm, cand); true }
    else { false }
}

fn apply_best_swap_diff_reduce_windowed(state: &mut State, params: &Params) -> bool {
    let used = used_in_core(state);
    let mut best: Option<(usize, usize, i32)> = None;
    for &rm in &used {
        let w_rm = state.ch.weights[rm];
        if w_rm == 0 { continue; }
        // look at bins with weight in [w_rm - diff_lim, w_rm - 1]
        let w_min = w_rm.saturating_sub(params.diff_lim as u32);
        for (bw, items) in &state.core_bins {
            if *bw >= w_rm { break; }
            if *bw < w_min { continue; }
            for &cand in items {
                if state.selected_bit[cand] { continue; }
                let delta = state.contrib[cand] - state.contrib[rm]
                    - state.ch.interaction_values[cand][rm];
                if delta > 0 && best.map_or(true, |(_, _, bd)| delta > bd) {
                    best = Some((cand, rm, delta));
                }
            }
        }
    }
    if let Some((cand, rm, _)) = best { state.replace_item(rm, cand); true }
    else { false }
}

fn apply_best_swap_diff_increase_windowed(state: &mut State, params: &Params) -> bool {
    let slack = state.slack();
    if slack == 0 { return false; }
    let used = used_in_core(state);
    let mut best: Option<(usize, usize, f64)> = None;
    for &rm in &used {
        let w_rm = state.ch.weights[rm];
        // consider bins with weight in (w_rm, w_rm + min(diff_lim, slack)]
        let max_dw = (params.diff_lim as u32).min(slack);
        let w_max = w_rm.saturating_add(max_dw);
        for (bw, items) in &state.core_bins {
            if *bw <= w_rm { continue; }
            if *bw > w_max { break; }
            let dw = *bw - w_rm;
            if dw > slack { break; }
            for &cand in items {
                if state.selected_bit[cand] { continue; }
                let delta = state.contrib[cand] - state.contrib[rm]
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
    if let Some((cand, rm, _)) = best { state.replace_item(rm, cand); true }
    else { false }
}

pub fn local_search_vnd(state: &mut State, params: &Params) {
    loop {
        if apply_best_add_windowed(state) { continue; }
        if apply_best_swap_diff_reduce_windowed(state, params) { continue; }
        if apply_best_swap11_equal_windowed(state) { continue; }
        if apply_best_swap_diff_increase_windowed(state, params) { continue; }
        break;
    }
}
