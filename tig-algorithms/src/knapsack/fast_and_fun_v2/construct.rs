use super::state::State;
use std::collections::BTreeMap;

/// Two-stage construction:
///  (1) Greedy by ascending weight (fast)
///  (2) Iterate greedy by density (contrib[i]/w[i]) up to n_it_construct times
///      or until the selection no longer changes
pub fn build_initial_solution(state: &mut State, n_it_construct: usize, core_half: usize) {
    let n = state.ch.num_items;

    // ---- Stage 1: greedy by ascending weight (fast) ----
    let mut by_weight: Vec<usize> = (0..n).collect();
    by_weight.sort_unstable_by(|&a, &b| state.ch.weights[a].cmp(&state.ch.weights[b]));

    let cap = state.ch.max_weight;
    for &i in &by_weight {
        let w = state.ch.weights[i];
        if state.total_weight + w <= cap {
            state.add_item(i);
        } else {
            break; // later items are heavier → won't fit
        }
    }

    // ---- Stage 2: iterate greedy by density (contrib / weight) ----
    let mut idx_last_inserted = 0 ;
    let mut idx_first_rejected = n ;
    let mut by_density: Vec<usize> = (0..n).collect();

    for _ in 0..=n_it_construct {
        idx_last_inserted = 0 ;
        idx_first_rejected = n ;
        let contrib = &state.contrib;
        by_density.sort_unstable_by(|&a, &b| {
            let na = contrib[a] as i64;
            let nb = contrib[b] as i64;
            let wa = state.ch.weights[a] as i64;
            let wb = state.ch.weights[b] as i64;
            (na * wb).cmp(&(nb * wa)).reverse()
        });

        // Build target set greedily from this static density order
        let mut target_sel: Vec<usize> = Vec::with_capacity(n);
        let mut rem = cap;
        for (idx, &i) in by_density.iter().enumerate() {
            let w = state.ch.weights[i];
            if w <= rem {
                target_sel.push(i);
                rem -= w;
                idx_last_inserted = idx;
            } else if idx_first_rejected == n {
                idx_first_rejected = idx;
            }
        }

        // Minimal set update: remove items not in target, then add missing ones
        let mut in_target = vec![false; n];
        for &i in &target_sel { in_target[i] = true; }
        let mut to_remove: Vec<usize> = Vec::new();
        let mut to_add: Vec<usize> = Vec::new();
        for i in 0..n {
            if state.selected_bit[i] && !in_target[i] { to_remove.push(i); }
            if !state.selected_bit[i] && in_target[i] { to_add.push(i); }
        }

        // Stop if no change
        if to_remove.is_empty() && to_add.is_empty() {
            break;
        }

        for &r in &to_remove { state.remove_item(r); }
        for &a in &to_add {
            debug_assert!(state.total_weight + state.ch.weights[a] <= cap, "Capacity violation");
            state.add_item(a);
        }
    }

    // ---- Export (locked, core, rejected) from the final density order ----
    let mut left  = (idx_first_rejected - core_half - 1).max(0);
    let right = (idx_last_inserted  + core_half + 1).min(n);
    if left > right { left = right; }
    state.window_locked   = by_density[..left].to_vec();
    state.window_core     = by_density[left..right].to_vec();
    state.window_rejected = by_density[right..].to_vec();

    // ---- Build static bins by weight for items in the core ----
    // (ascending by weight, then keep the item lists as-is)
    let mut bins: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for &i in &state.window_core {
        let w = state.ch.weights[i];
        bins.entry(w).or_default().push(i);
    }
    state.core_bins = bins.into_iter().collect();
}
