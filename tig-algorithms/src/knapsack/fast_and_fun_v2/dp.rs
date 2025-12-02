use tig_challenges::knapsack::*;
use super::state::State;

fn integer_core_target(
    ch: &Challenge,
    contrib: &[i32],
    locked: &[usize],
    core: &[usize],
) -> Vec<usize> {

    let used_locked: u64 = locked.iter().map(|&i| ch.weights[i] as u64).sum();
    let rem_cap = (ch.max_weight as u64).saturating_sub(used_locked) as usize;

    let myw = rem_cap;
    let myk = core.len();
    let mut dp: Vec<i64> = vec![i64::MIN / 4; myw + 1];
    dp[0] = 0;
    let mut choose: Vec<u8> = vec![0u8; myk * (myw + 1)];
    let mut w_hi: usize = 0;

    for (t, &it) in core.iter().enumerate() {
        let wt = ch.weights[it] as usize;
        if wt > myw { continue; }
        let val = contrib[it] as i64;
        let new_hi = (w_hi + wt).min(myw);
        for w in (wt..=new_hi).rev() {
            let cand = dp[w - wt] + val;
            if cand > dp[w] { dp[w] = cand; choose[t * (myw + 1) + w] = 1; }
        }
        w_hi = new_hi;
    }

    // Reconstruct selected set: locked + DP picks from core
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
    let n = state.ch.num_items;
    let mut in_target = vec![false; n];
    for &i in target_sel { in_target[i] = true; }

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
    for &r in &to_remove { state.remove_item(r); }
    for &a in &to_add    { state.add_item(a); }
}

pub fn dp_refinement(state: &mut State) {
    let target = integer_core_target(
        state.ch,
        &state.contrib,
        &state.window_locked,
        &state.window_core,
    );
    apply_dp_target_via_ops(state, &target);
}
