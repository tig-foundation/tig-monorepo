use super::types::{State, MICRO_ADD_K, MICRO_K, MICRO_RM_K};
use tig_challenges::knapsack::Challenge;

fn integer_core_target(
    ch: &Challenge,
    locked: &[usize],
    core: &[usize],
    core_val: &[i32],
    dp_cache: &mut Vec<i64>,
    choose_cache: &mut Vec<u8>,
) -> Vec<usize> {
    let used_locked: u64 = locked.iter().map(|&i| ch.weights[i] as u64).sum();
    let rem_cap = (ch.max_weight as u64).saturating_sub(used_locked) as usize;

    let myk = core.len();
    if myk == 0 {
        let mut selected: Vec<usize> = locked.to_vec();
        selected.sort_unstable();
        return selected;
    }

    let mut total_core_weight: usize = 0;
    let mut total_pos_weight: usize = 0;
    let mut all_pos_fit = true;
    for (t, &it) in core.iter().enumerate() {
        let wt = ch.weights[it] as usize;
        total_core_weight += wt;
        if core_val[t] > 0 {
            total_pos_weight += wt;
            if total_pos_weight > rem_cap {
                all_pos_fit = false;
            }
        }
    }

    if rem_cap == 0 {
        let mut selected: Vec<usize> = locked.to_vec();
        for (t, &it) in core.iter().enumerate() {
            if ch.weights[it] == 0 && core_val[t] > 0 {
                selected.push(it);
            }
        }
        selected.sort_unstable();
        return selected;
    }

    if all_pos_fit {
        let mut selected: Vec<usize> = locked.to_vec();
        for (t, &it) in core.iter().enumerate() {
            if core_val[t] > 0 {
                selected.push(it);
            }
        }
        selected.sort_unstable();
        return selected;
    }

    let myw = rem_cap.min(total_core_weight);

    let dp_size = myw + 1;
    let choose_size = myk * dp_size;

    if dp_cache.len() < dp_size {
        dp_cache.resize(dp_size, i64::MIN / 4);
    }
    if choose_cache.len() < choose_size {
        choose_cache.resize(choose_size, 0);
    }

    for val in &mut dp_cache[0..dp_size] {
        *val = -1;
    }
    dp_cache[0] = 0;

    choose_cache[0..choose_size].fill(0);

    let mut w_hi: usize = 0;

    for (t, &it) in core.iter().enumerate() {
        let wt = ch.weights[it] as usize;
        if wt > myw {
            continue;
        }
        let val = core_val[t] as i64;
        let new_hi = (w_hi + wt).min(myw);
        for w in (wt..=new_hi).rev() {
            let prev = dp_cache[w - wt];
            if prev < 0 {
                continue;
            }
            let cand = prev + val;
            if cand > dp_cache[w] {
                dp_cache[w] = cand;
                choose_cache[t * dp_size + w] = 1;
            }
        }
        w_hi = new_hi;
    }

    let mut selected: Vec<usize> = locked.to_vec();
    let mut w_star = (0..=myw).max_by_key(|&w| dp_cache[w]).unwrap_or(0);
    for t in (0..myk).rev() {
        let it = core[t];
        let wt = ch.weights[it] as usize;
        if wt <= w_star && choose_cache[t * dp_size + w_star] == 1 {
            selected.push(it);
            w_star -= wt;
        }
    }
    selected.sort_unstable();
    selected
}

fn apply_dp_target_via_ops(state: &mut State, target_sel: &[usize]) {
    let n = state.ch.num_items;
    let mut to_remove: Vec<usize> = Vec::new();
    let mut to_add: Vec<usize> = Vec::new();
    let mut j = 0;
    let m = target_sel.len();

    for i in 0..n {
        let in_target = j < m && target_sel[j] == i;
        if in_target {
            j += 1;
        }
        if state.selected_bit[i] && !in_target {
            to_remove.push(i);
        } else if in_target && !state.selected_bit[i] {
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

pub fn dp_refinement(state: &mut State) {
    dp_refinement_x(state, 1);
}

pub fn dp_refinement_x(state: &mut State, passes_multiplier: usize) {
    let team_est = (state.ch.max_weight as usize) / 6;
    let k = state.window_core.len();
    let passes = (if team_est <= 140 {
        if k <= 80 { 5 } else { 4 }
    } else if team_est <= 180 {
        if k <= 70 { 4 } else { 3 }
    } else if k <= 160 {
        2
    } else {
        1
    }) * passes_multiplier;
    let n = state.ch.num_items;
    if n == 0 {
        return;
    }

    for _pass in 0..passes {
        let mut core_val: Vec<i32> = Vec::with_capacity(k);
        if k != 0 {
            let mut sel_core_bit = vec![false; n];
            let mut sel_core: Vec<usize> = Vec::new();
            for &i in &state.window_core {
                if state.selected_bit[i] {
                    sel_core_bit[i] = true;
                    sel_core.push(i);
                }
            }

            let pair_mode = k <= 96 || (state.neigh.is_some() && k <= 144);
            if pair_mode {
                let mut core_index = vec![usize::MAX; n];
                for (idx, &it) in state.window_core.iter().enumerate() {
                    core_index[it] = idx;
                }

                let mut sel_sum: Vec<i32> = vec![0; k];
                let mut pos_unsel: Vec<i32> = vec![0; k];
                let mut neg_sel: Vec<i32> = vec![0; k];
                let mut neg_all: Vec<i32> = vec![0; k];
                let mut top_pos1: Vec<i32> = vec![0; k];
                let mut top_pos2: Vec<i32> = vec![0; k];

                if let Some(ref ng) = state.neigh {
                    for (a, &it) in state.window_core.iter().enumerate() {
                        let row = unsafe { ng.get_unchecked(it) };
                        for &(kk, v) in row.iter() {
                            let j = kk as usize;
                            let b = core_index[j];
                            if b == usize::MAX || b == a {
                                continue;
                            }
                            let iv = v as i32;
                            if state.selected_bit[j] {
                                sel_sum[a] += iv;
                            }
                            if iv > 0 {
                                if !state.selected_bit[j] {
                                    pos_unsel[a] += iv;
                                }
                                if iv >= top_pos1[a] {
                                    top_pos2[a] = top_pos1[a];
                                    top_pos1[a] = iv;
                                } else if iv > top_pos2[a] {
                                    top_pos2[a] = iv;
                                }
                            } else if iv < 0 {
                                let mag = -iv;
                                neg_all[a] += mag;
                                if state.selected_bit[j] {
                                    neg_sel[a] += mag;
                                }
                            }
                        }
                    }
                } else {
                    for (a, &it) in state.window_core.iter().enumerate() {
                        for (b, &j) in state.window_core.iter().enumerate() {
                            if a == b {
                                continue;
                            }
                            let iv = state.ch.interaction_values[it][j];
                            if state.selected_bit[j] {
                                sel_sum[a] += iv;
                            }
                            if iv > 0 {
                                if !state.selected_bit[j] {
                                    pos_unsel[a] += iv;
                                }
                                if iv >= top_pos1[a] {
                                    top_pos2[a] = top_pos1[a];
                                    top_pos1[a] = iv;
                                } else if iv > top_pos2[a] {
                                    top_pos2[a] = iv;
                                }
                            } else if iv < 0 {
                                let mag = -iv;
                                neg_all[a] += mag;
                                if state.selected_bit[j] {
                                    neg_sel[a] += mag;
                                }
                            }
                        }
                    }
                }

                for (idx, &it) in state.window_core.iter().enumerate() {
                    let bonus = top_pos1[idx] / 4 + top_pos2[idx] / 6 + pos_unsel[idx] / 18;
                    let penalty = neg_sel[idx] / 7 + neg_all[idx] / 24;
                    let mut v0 = state.contrib[it]
                        - if state.selected_bit[it] {
                            (sel_sum[idx] * 2) / 3
                        } else {
                            sel_sum[idx] / 3
                        };

                    if state.selected_bit[it] {
                        v0 += bonus / 2 - penalty;
                    } else {
                        v0 += bonus - penalty / 2;
                        v0 += (state.total_interactions[it] / 256) as i32
                            + (state.usage[it] as i32) * 14;
                    }
                    core_val.push(v0);
                }
            } else if let Some(ref ng) = state.neigh {
                for &it in &state.window_core {
                    let mut sub: i32 = 0;
                    let row = unsafe { ng.get_unchecked(it) };
                    for &(kk, v) in row.iter() {
                        let j = kk as usize;
                        if sel_core_bit[j] {
                            sub += v as i32;
                        }
                    }

                    let mut v0 = state.contrib[it]
                        - if state.selected_bit[it] {
                            (sub * 2) / 3
                        } else {
                            sub / 3
                        };
                    if !state.selected_bit[it] {
                        v0 += (state.total_interactions[it] / 256) as i32
                            + (state.usage[it] as i32) * 14;
                    }
                    core_val.push(v0);
                }
            } else {
                for &it in &state.window_core {
                    let mut sub: i32 = 0;
                    for &j in &sel_core {
                        sub += state.ch.interaction_values[it][j];
                    }

                    let mut v0 = state.contrib[it]
                        - if state.selected_bit[it] {
                            (sub * 2) / 3
                        } else {
                            sub / 3
                        };
                    if !state.selected_bit[it] {
                        v0 += (state.total_interactions[it] / 256) as i32
                            + (state.usage[it] as i32) * 14;
                    }
                    core_val.push(v0);
                }
            }
        }

        let target = integer_core_target(
            state.ch,
            &state.window_locked,
            &state.window_core,
            &core_val,
            &mut state.dp_cache,
            &mut state.choose_cache,
        );
        apply_dp_target_via_ops(state, &target);
    }
}

pub fn micro_qkp_refinement(state: &mut State) {
    let n = state.ch.num_items;
    if n == 0 || state.window_core.is_empty() {
        return;
    }

    let team_est = (state.ch.max_weight as usize) / 6;
    let big_team = n >= 4500 && team_est >= 850;

    let exact_k_limit: usize = if big_team {
        12
    } else if team_est <= 140 {
        23
    } else if team_est <= 160 {
        21
    } else {
        MICRO_K.min(20)
    };
    let micro_k: usize = exact_k_limit;
    let rm_k: usize = if big_team {
        6
    } else if team_est <= 140 {
        11
    } else if team_est <= 160 {
        10
    } else {
        MICRO_RM_K.min(micro_k)
    };
    let add_k: usize = if big_team {
        6
    } else if team_est <= 140 {
        11
    } else if team_est <= 160 {
        10
    } else {
        MICRO_ADD_K.min(micro_k)
    };

    let cand: Vec<usize> = {
        let add_score = |i: usize| -> i64 {
            let w = (state.ch.weights[i] as i64).max(1);
            let dens = (state.contrib[i] as i64 * 1000) / w;
            dens
                + (state.support[i] as i64) * 140
                + (state.total_interactions[i] / 280)
                + (state.usage[i] as i64) * 10
        };
        let rem_score = |i: usize| -> i64 {
            let w = (state.ch.weights[i] as i64).max(1);
            let dens = (state.contrib[i] as i64 * 1000) / w;
            dens + (state.support[i] as i64) * 110 + (state.total_interactions[i] / 384)
        };
        let push_u = |v: &mut Vec<usize>, x: usize| {
            for &y in v.iter() {
                if y == x {
                    return;
                }
            }
            v.push(x);
        };

        let mut sel_pool: Vec<usize> = Vec::new();
        let mut unsel_pool: Vec<usize> = Vec::new();

        for &i in &state.window_core {
            if state.selected_bit[i] {
                push_u(&mut sel_pool, i);
            } else {
                push_u(&mut unsel_pool, i);
            }
        }

        let extra_r = state.window_rejected.len().min(if big_team { 12 } else { 24 });
        for &i in &state.window_rejected[..extra_r] {
            if !state.selected_bit[i] {
                push_u(&mut unsel_pool, i);
            }
        }

        let extra_l = state.window_locked.len().min(24);
        let start_l = state.window_locked.len().saturating_sub(extra_l);
        for &i in &state.window_locked[start_l..] {
            if state.selected_bit[i] {
                push_u(&mut sel_pool, i);
            }
        }

        let hub_take: usize = if big_team {
            5
        } else if team_est <= 160 {
            12
        } else {
            8
        };
        let lim = state.hubs_static.len().min(192);
        let mut added_hubs: usize = 0;
        for &h in state.hubs_static.iter().take(lim) {
            if added_hubs >= hub_take {
                break;
            }
            if !state.selected_bit[h] {
                let before = unsel_pool.len();
                push_u(&mut unsel_pool, h);
                if unsel_pool.len() > before {
                    added_hubs += 1;
                }
            }
        }

        if let Some(ref ng) = state.neigh {
            let mut guides = sel_pool.clone();
            guides.sort_unstable_by(|&a, &b| {
                state.support[b]
                    .cmp(&state.support[a])
                    .then_with(|| state.contrib[b].cmp(&state.contrib[a]))
                    .then_with(|| b.cmp(&a))
            });

            let mut anchors = unsel_pool.clone();
            anchors.sort_unstable_by(|&a, &b| {
                add_score(b).cmp(&add_score(a)).then_with(|| b.cmp(&a))
            });

            let mut weak_sel = sel_pool.clone();
            weak_sel.sort_unstable_by(|&a, &b| {
                rem_score(a).cmp(&rem_score(b)).then_with(|| a.cmp(&b))
            });

            for &vtx in guides.iter().take(if big_team { 4 } else { 6 }) {
                let row = unsafe { ng.get_unchecked(vtx) };
                for &(kk, _) in row.iter().take(if big_team { 16 } else { 24 }) {
                    let j = kk as usize;
                    if state.selected_bit[j] {
                        push_u(&mut sel_pool, j);
                    } else {
                        push_u(&mut unsel_pool, j);
                    }
                }
            }

            for &vtx in anchors.iter().take(if big_team { 4 } else if team_est <= 160 { 7 } else { 5 }) {
                let row = unsafe { ng.get_unchecked(vtx) };
                for &(kk, _) in row.iter().take(if big_team { 16 } else { 28 }) {
                    let j = kk as usize;
                    if state.selected_bit[j] {
                        push_u(&mut sel_pool, j);
                    } else {
                        push_u(&mut unsel_pool, j);
                    }
                }
            }

            for &vtx in weak_sel.iter().take(if big_team { 3 } else { 5 }) {
                let row = unsafe { ng.get_unchecked(vtx) };
                for &(kk, _) in row.iter().take(if big_team { 12 } else { 20 }) {
                    let j = kk as usize;
                    if !state.selected_bit[j] {
                        push_u(&mut unsel_pool, j);
                    }
                }
            }
        } else if state.window_core.len() <= 96 {
            let selected_core: Vec<usize> = state
                .window_core
                .iter()
                .copied()
                .filter(|&i| state.selected_bit[i])
                .collect();
            let unselected_core: Vec<usize> = state
                .window_core
                .iter()
                .copied()
                .filter(|&i| !state.selected_bit[i])
                .collect();

            let mut anchors = unselected_core.clone();
            anchors.sort_unstable_by(|&a, &b| {
                add_score(b).cmp(&add_score(a)).then_with(|| b.cmp(&a))
            });
            let mut weak_sel = selected_core.clone();
            weak_sel.sort_unstable_by(|&a, &b| {
                rem_score(a).cmp(&rem_score(b)).then_with(|| a.cmp(&b))
            });

            for &it in anchors.iter().take(4) {
                let mut best_sel: [(i32, usize); 2] = [(i32::MIN, usize::MAX); 2];
                let mut best_unsel: [(i32, usize); 2] = [(i32::MIN, usize::MAX); 2];
                for &j in &selected_core {
                    let v = state.ch.interaction_values[it][j];
                    if v > best_sel[0].0 {
                        best_sel[1] = best_sel[0];
                        best_sel[0] = (v, j);
                    } else if v > best_sel[1].0 {
                        best_sel[1] = (v, j);
                    }
                }
                for &j in &unselected_core {
                    if j == it {
                        continue;
                    }
                    let v = state.ch.interaction_values[it][j];
                    if v > best_unsel[0].0 {
                        best_unsel[1] = best_unsel[0];
                        best_unsel[0] = (v, j);
                    } else if v > best_unsel[1].0 {
                        best_unsel[1] = (v, j);
                    }
                }
                for &(_, j) in &best_sel {
                    if j != usize::MAX {
                        push_u(&mut sel_pool, j);
                    }
                }
                for &(_, j) in &best_unsel {
                    if j != usize::MAX {
                        push_u(&mut unsel_pool, j);
                    }
                }
            }

            for &it in weak_sel.iter().take(4) {
                let mut best_unsel: [(i32, usize); 3] = [(i32::MIN, usize::MAX); 3];
                for &j in &unselected_core {
                    let v = state.ch.interaction_values[it][j];
                    if v > best_unsel[0].0 {
                        best_unsel[2] = best_unsel[1];
                        best_unsel[1] = best_unsel[0];
                        best_unsel[0] = (v, j);
                    } else if v > best_unsel[1].0 {
                        best_unsel[2] = best_unsel[1];
                        best_unsel[1] = (v, j);
                    } else if v > best_unsel[2].0 {
                        best_unsel[2] = (v, j);
                    }
                }
                for &(_, j) in &best_unsel {
                    if j != usize::MAX {
                        push_u(&mut unsel_pool, j);
                    }
                }
            }
        }

        sel_pool.sort_unstable_by(|&a, &b| {
            rem_score(a).cmp(&rem_score(b)).then_with(|| a.cmp(&b))
        });
        unsel_pool.sort_unstable_by(|&a, &b| {
            add_score(b).cmp(&add_score(a)).then_with(|| b.cmp(&a))
        });

        let mut cand: Vec<usize> = Vec::with_capacity(micro_k);
        for &i in sel_pool.iter().take(rm_k) {
            push_u(&mut cand, i);
            if cand.len() >= micro_k {
                break;
            }
        }
        for &i in unsel_pool.iter().take(add_k) {
            push_u(&mut cand, i);
            if cand.len() >= micro_k {
                break;
            }
        }

        let mut ps = rm_k.min(sel_pool.len());
        let mut pu = add_k.min(unsel_pool.len());
        while cand.len() < micro_k && (ps < sel_pool.len() || pu < unsel_pool.len()) {
            let take_unsel = if ps >= sel_pool.len() {
                true
            } else if pu >= unsel_pool.len() {
                false
            } else {
                let mut sel_cnt = 0usize;
                let mut unsel_cnt = 0usize;
                for &x in &cand {
                    if state.selected_bit[x] {
                        sel_cnt += 1;
                    } else {
                        unsel_cnt += 1;
                    }
                }
                unsel_cnt <= sel_cnt
            };

            if take_unsel {
                push_u(&mut cand, unsel_pool[pu]);
                pu += 1;
            } else {
                push_u(&mut cand, sel_pool[ps]);
                ps += 1;
            }
        }

        cand
    };

    if cand.len() < 2 {
        return;
    }

    let k = cand.len();
    if k > exact_k_limit || k > usize::BITS as usize {
        return;
    }

    let mut sel_cand: Vec<usize> = Vec::new();
    let mut sel_cand_w: u32 = 0;
    for &it in &cand {
        if state.selected_bit[it] {
            sel_cand.push(it);
            sel_cand_w = sel_cand_w.saturating_add(state.ch.weights[it]);
        }
    }
    if state.total_weight < sel_cand_w {
        return;
    }

    let fixed_w = state.total_weight - sel_cand_w;
    if fixed_w > state.ch.max_weight {
        return;
    }
    let rem_cap = state.ch.max_weight - fixed_w;

    let mut w: Vec<u32> = vec![0; k];
    let mut base: Vec<i64> = vec![0; k];
    for t in 0..k {
        let it = cand[t];
        w[t] = state.ch.weights[it];
        let mut b = state.contrib[it] as i64;
        for &j in &sel_cand {
            b -= state.ch.interaction_values[it][j] as i64;
        }
        base[t] = b;
    }

    let mut inter: Vec<i64> = vec![0; k * k];
    for a in 0..k {
        let ia = cand[a];
        for b in 0..a {
            let ib = cand[b];
            let v = state.ch.interaction_values[ia][ib] as i64;
            inter[a * k + b] = v;
            inter[b * k + a] = v;
        }
    }

    let mut pos_sum: Vec<i64> = vec![0; k];
    for a in 0..k {
        let mut s = 0i64;
        for b in 0..k {
            if a != b {
                let v = inter[a * k + b];
                if v > 0 {
                    s += v;
                }
            }
        }
        pos_sum[a] = s;
    }

    let mut ub_item: Vec<i64> = vec![0; k];
    for t in 0..k {
        ub_item[t] = base[t] + pos_sum[t];
    }

    let mut order: Vec<usize> = (0..k).collect();
    order.sort_unstable_by(|&a, &b| {
        let lhs = (ub_item[b].max(0) as i128) * (w[a].max(1) as i128);
        let rhs = (ub_item[a].max(0) as i128) * (w[b].max(1) as i128);
        lhs.cmp(&rhs)
            .then_with(|| ub_item[b].cmp(&ub_item[a]))
            .then_with(|| w[a].cmp(&w[b]))
    });

    let mut cand2: Vec<usize> = vec![0; k];
    let mut w2: Vec<u32> = vec![0; k];
    let mut base2: Vec<i64> = vec![0; k];
    let mut ub_item2: Vec<i64> = vec![0; k];
    for (new_idx, &old_idx) in order.iter().enumerate() {
        cand2[new_idx] = cand[old_idx];
        w2[new_idx] = w[old_idx];
        base2[new_idx] = base[old_idx];
        ub_item2[new_idx] = ub_item[old_idx];
    }

    let mut inter2: Vec<i64> = vec![0; k * k];
    for a in 0..k {
        let oa = order[a];
        for b in 0..k {
            let ob = order[b];
            inter2[a * k + b] = inter[oa * k + ob];
        }
    }

    let mut future_pos_prefix: Vec<Vec<i64>> = Vec::with_capacity(k);
    let mut future_w_prefix: Vec<Vec<u32>> = Vec::with_capacity(k);
    for i in 0..k {
        let mut pos_vals: Vec<i64> = Vec::new();
        let mut fut_w: Vec<u32> = Vec::new();
        for j in (i + 1)..k {
            let v = inter2[i * k + j];
            if v > 0 {
                pos_vals.push(v);
            }
            fut_w.push(w2[j]);
        }
        pos_vals.sort_unstable_by(|a, b| b.cmp(a));
        fut_w.sort_unstable();

        let mut pos_pref: Vec<i64> = Vec::with_capacity(pos_vals.len() + 1);
        pos_pref.push(0);
        for v in pos_vals {
            let last = *pos_pref.last().unwrap();
            pos_pref.push(last + v);
        }

        let mut w_pref: Vec<u32> = Vec::with_capacity(fut_w.len() + 1);
        w_pref.push(0);
        for wt in fut_w {
            let last = *w_pref.last().unwrap();
            w_pref.push(last.saturating_add(wt));
        }

        future_pos_prefix.push(pos_pref);
        future_w_prefix.push(w_pref);
    }

    let eval_mask = |mask: usize| -> (u32, i64) {
        let mut sum_w = 0u32;
        let mut sum_v = 0i64;
        for i in 0..k {
            if ((mask >> i) & 1) == 0 {
                continue;
            }
            sum_w = sum_w.saturating_add(w2[i]);
            sum_v += base2[i];
            for j in 0..i {
                if ((mask >> j) & 1) != 0 {
                    sum_v += inter2[i * k + j];
                }
            }
        }
        (sum_w, sum_v)
    };

    let mut cur_mask: usize = 0;
    for t in 0..k {
        if state.selected_bit[cand2[t]] {
            cur_mask |= 1usize << t;
        }
    }

    let (cur_w_sum, cur_v) = eval_mask(cur_mask);
    if cur_w_sum > rem_cap {
        return;
    }

    if k >= 10 {
        let mut comp_id: Vec<usize> = vec![usize::MAX; k];
        let mut comp_cnt = 0usize;
        for s in 0..k {
            if comp_id[s] != usize::MAX {
                continue;
            }
            let mut stack_nodes: Vec<usize> = vec![s];
            comp_id[s] = comp_cnt;
            while let Some(a) = stack_nodes.pop() {
                for b in 0..k {
                    if comp_id[b] == usize::MAX && inter2[a * k + b] > 0 {
                        comp_id[b] = comp_cnt;
                        stack_nodes.push(b);
                    }
                }
            }
            comp_cnt += 1;
        }

        if comp_cnt >= 2 {
            let mut separable = true;
            'cross_check: for a in 0..k {
                for b in 0..a {
                    if comp_id[a] != comp_id[b] && inter2[a * k + b] != 0 {
                        separable = false;
                        break 'cross_check;
                    }
                }
            }

            if separable {
                let mut comps: Vec<Vec<usize>> = vec![Vec::new(); comp_cnt];
                let mut max_comp = 0usize;
                for i in 0..k {
                    comps[comp_id[i]].push(i);
                }
                for comp in &comps {
                    max_comp = max_comp.max(comp.len());
                }

                if max_comp <= 12 || (comp_cnt >= 3 && max_comp <= 14) {
                    let prune_states = |states: &mut Vec<(u32, i64, usize)>| {
                        states.sort_unstable_by(|a, b| {
                            a.0.cmp(&b.0)
                                .then_with(|| b.1.cmp(&a.1))
                                .then_with(|| a.2.cmp(&b.2))
                        });
                        let mut write = 0usize;
                        let mut best_seen = i64::MIN;
                        for idx in 0..states.len() {
                            let (sw, sv, sm) = states[idx];
                            if sv > best_seen {
                                states[write] = (sw, sv, sm);
                                write += 1;
                                best_seen = sv;
                            }
                        }
                        states.truncate(write);
                    };

                    let mut merged: Vec<(u32, i64, usize)> = vec![(0, 0, 0)];
                    let mut decomp_ok = true;

                    for comp in &comps {
                        let local_n = comp.len();
                        let mut frontier: Vec<(u32, i64, usize)> =
                            Vec::with_capacity(1usize << local_n);
                        for local_mask in 0..(1usize << local_n) {
                            let mut sw = 0u32;
                            let mut sv = 0i64;
                            let mut sm = 0usize;
                            for li in 0..local_n {
                                if ((local_mask >> li) & 1) == 0 {
                                    continue;
                                }
                                let gi = comp[li];
                                sw = sw.saturating_add(w2[gi]);
                                if sw > rem_cap {
                                    break;
                                }
                                sv += base2[gi];
                                sm |= 1usize << gi;
                                for lj in 0..li {
                                    if ((local_mask >> lj) & 1) != 0 {
                                        sv += inter2[gi * k + comp[lj]];
                                    }
                                }
                            }
                            if sw <= rem_cap {
                                frontier.push((sw, sv, sm));
                            }
                        }
                        prune_states(&mut frontier);

                        let mut next_merged: Vec<(u32, i64, usize)> =
                            Vec::with_capacity(merged.len().saturating_mul(frontier.len()));
                        for &(mw, mv, mm) in &merged {
                            for &(fw, fv, fm) in &frontier {
                                let nw = mw.saturating_add(fw);
                                if nw <= rem_cap {
                                    next_merged.push((nw, mv + fv, mm | fm));
                                }
                            }
                        }
                        if next_merged.is_empty() {
                            decomp_ok = false;
                            break;
                        }
                        prune_states(&mut next_merged);
                        merged = next_merged;
                    }

                    if decomp_ok {
                        let mut best_mask = cur_mask;
                        let mut best_val = cur_v;
                        for &(_, v, mask) in &merged {
                            if v > best_val {
                                best_val = v;
                                best_mask = mask;
                            }
                        }

                        if best_mask == cur_mask {
                            return;
                        }

                        let mut to_remove: Vec<usize> = Vec::new();
                        let mut to_add: Vec<usize> = Vec::new();
                        for t in 0..k {
                            let it = cand2[t];
                            let want = ((best_mask >> t) & 1) != 0;
                            let have = state.selected_bit[it];
                            if have && !want {
                                to_remove.push(it);
                            } else if !have && want {
                                to_add.push(it);
                            }
                        }

                        for &it in &to_remove {
                            state.remove_item(it);
                        }
                        for &it in &to_add {
                            if state.total_weight + state.ch.weights[it] <= state.ch.max_weight {
                                state.add_item(it);
                            }
                        }
                        return;
                    }
                }
            }
        }
    }

    let mut best_mask = cur_mask;
    let mut best_val = cur_v;

    let mut gain_items: Vec<(i64, u32)> = Vec::with_capacity(k);
    let mut exact_states: Vec<(u32, i64)> = Vec::with_capacity(256);
    let mut exact_next_states: Vec<(u32, i64)> = Vec::with_capacity(256);

    let mut bound_from = |idx: usize, cw: u32, cv: i64, mask: usize| -> i64 {
        let rem = rem_cap - cw;
        gain_items.clear();

        for i in idx..k {
            let mut pos_mask = 0i64;
            let mut pm = mask;
            while pm != 0 {
                let l = pm & pm.wrapping_neg();
                let j = l.trailing_zeros() as usize;
                let v = inter2[i * k + j];
                if v > 0 {
                    pos_mask += v;
                }
                pm ^= l;
            }

            let left = rem.saturating_sub(w2[i]);
            let w_pref = &future_w_prefix[i];
            let mut partner_cap = 0usize;
            while partner_cap + 1 < w_pref.len() && w_pref[partner_cap + 1] <= left {
                partner_cap += 1;
            }

            let pos_pref = &future_pos_prefix[i];
            let future_bonus = pos_pref[partner_cap.min(pos_pref.len() - 1)];
            let gain = base2[i] + pos_mask + future_bonus;
            if gain > 0 {
                gain_items.push((gain, w2[i]));
            }
        }

        let suffix_len = k - idx;
        if !gain_items.is_empty()
            && (gain_items.len() <= 10
                || (idx >= k / 2 && suffix_len <= 14 && gain_items.len() <= 14))
        {
            exact_states.clear();
            exact_states.push((0, 0));

            for &(gain, wt) in &gain_items {
                if wt != 0 && wt > rem {
                    continue;
                }

                exact_next_states.clear();
                for &(sw, sv) in &exact_states {
                    exact_next_states.push((sw, sv));
                    let nw = sw.saturating_add(wt);
                    if nw <= rem {
                        exact_next_states.push((nw, sv + gain));
                    }
                }

                exact_next_states.sort_unstable_by(|a, b| {
                    a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1))
                });

                exact_states.clear();
                let mut best_seen = i64::MIN;
                for &(sw, sv) in &exact_next_states {
                    if sv > best_seen {
                        exact_states.push((sw, sv));
                        best_seen = sv;
                    }
                }
            }

            return cv
                + exact_states
                    .iter()
                    .map(|&(_, sv)| sv)
                    .max()
                    .unwrap_or(0);
        }

        gain_items.sort_unstable_by(|a, b| {
            let lhs = (b.0 as i128) * (a.1.max(1) as i128);
            let rhs = (a.0 as i128) * (b.1.max(1) as i128);
            lhs.cmp(&rhs)
                .then_with(|| b.0.cmp(&a.0))
                .then_with(|| a.1.cmp(&b.1))
        });

        let mut ub = cv;
        let mut cap = rem;
        for &(gain, wt) in &gain_items {
            if wt == 0 {
                ub += gain;
            } else if wt <= cap {
                ub += gain;
                cap -= wt;
            } else {
                ub += gain * cap as i64 / wt as i64;
                break;
            }
        }
        ub
    };

    let mut warm_mask = cur_mask;
    let mut warm_w = cur_w_sum;
    let mut warm_v = cur_v;

    loop {
        let mut next_mask = warm_mask;
        let mut next_w = warm_w;
        let mut next_v = warm_v;
        for i in 0..k {
            if ((warm_mask >> i) & 1) == 0 {
                continue;
            }
            let cand_mask = warm_mask & !(1usize << i);
            let (cand_w, cand_v) = eval_mask(cand_mask);
            if cand_v > next_v
                || (cand_v == next_v && cand_w < next_w)
                || (cand_v == next_v && cand_w == next_w && cand_mask < next_mask)
            {
                next_mask = cand_mask;
                next_w = cand_w;
                next_v = cand_v;
            }
        }
        if next_v > warm_v || (next_v == warm_v && next_w < warm_w) {
            warm_mask = next_mask;
            warm_w = next_w;
            warm_v = next_v;
            if warm_v > best_val {
                best_val = warm_v;
                best_mask = warm_mask;
            }
        } else {
            break;
        }
    }

    for _ in 0..3 {
        let mut best_add = usize::MAX;
        let mut best_add_w = warm_w;
        let mut best_add_v = warm_v;
        for i in 0..k {
            if ((warm_mask >> i) & 1) != 0 {
                continue;
            }
            if warm_w.saturating_add(w2[i]) > rem_cap {
                continue;
            }
            let cand_mask = warm_mask | (1usize << i);
            let (cand_w, cand_v) = eval_mask(cand_mask);
            if cand_v <= warm_v {
                continue;
            }
            if best_add == usize::MAX
                || cand_v > best_add_v
                || (cand_v == best_add_v && cand_w < best_add_w)
                || (cand_v == best_add_v && cand_w == best_add_w && i < best_add)
            {
                best_add = i;
                best_add_w = cand_w;
                best_add_v = cand_v;
            }
        }
        if best_add == usize::MAX {
            break;
        }
        warm_mask |= 1usize << best_add;
        warm_w = best_add_w;
        warm_v = best_add_v;
        if warm_v > best_val {
            best_val = warm_v;
            best_mask = warm_mask;
        }
    }

    for _ in 0..2 {
        let mut next_mask = warm_mask;
        let mut next_w = warm_w;
        let mut next_v = warm_v;
        for rem_i in 0..k {
            if ((warm_mask >> rem_i) & 1) == 0 {
                continue;
            }
            for add_i in 0..k {
                if ((warm_mask >> add_i) & 1) != 0 {
                    continue;
                }
                let cand_mask = (warm_mask & !(1usize << rem_i)) | (1usize << add_i);
                let (cand_w, cand_v) = eval_mask(cand_mask);
                if cand_w > rem_cap {
                    continue;
                }
                if cand_v > next_v
                    || (cand_v == next_v && cand_w < next_w)
                    || (cand_v == next_v && cand_w == next_w && cand_mask < next_mask)
                {
                    next_mask = cand_mask;
                    next_w = cand_w;
                    next_v = cand_v;
                }
            }
        }
        if next_v > warm_v || (next_v == warm_v && next_w < warm_w) {
            warm_mask = next_mask;
            warm_w = next_w;
            warm_v = next_v;
            if warm_v > best_val {
                best_val = warm_v;
                best_mask = warm_mask;
            }

            let mut best_add = usize::MAX;
            let mut best_add_w = warm_w;
            let mut best_add_v = warm_v;
            for i in 0..k {
                if ((warm_mask >> i) & 1) != 0 {
                    continue;
                }
                if warm_w.saturating_add(w2[i]) > rem_cap {
                    continue;
                }
                let cand_mask = warm_mask | (1usize << i);
                let (cand_w, cand_v) = eval_mask(cand_mask);
                if cand_v <= warm_v {
                    continue;
                }
                if best_add == usize::MAX
                    || cand_v > best_add_v
                    || (cand_v == best_add_v && cand_w < best_add_w)
                    || (cand_v == best_add_v && cand_w == best_add_w && i < best_add)
                {
                    best_add = i;
                    best_add_w = cand_w;
                    best_add_v = cand_v;
                }
            }
            if best_add != usize::MAX {
                warm_mask |= 1usize << best_add;
                warm_w = best_add_w;
                warm_v = best_add_v;
                if warm_v > best_val {
                    best_val = warm_v;
                    best_mask = warm_mask;
                }
            }
        } else {
            break;
        }
    }

    let mut stack: Vec<(usize, u32, i64, usize)> = Vec::new();
    stack.push((0, 0, 0, 0));

    while let Some((idx, cw, cv, mask)) = stack.pop() {
        let ub = bound_from(idx, cw, cv, mask);
        if ub <= best_val {
            continue;
        }

        if idx == k {
            if cv > best_val {
                best_val = cv;
                best_mask = mask;
            }
            continue;
        }

        stack.push((idx + 1, cw, cv, mask));

        let wt = w2[idx];
        let new_w = cw.saturating_add(wt);
        if new_w <= rem_cap {
            let mut delta = base2[idx];
            let mut pm = mask;
            while pm != 0 {
                let l = pm & pm.wrapping_neg();
                let j = l.trailing_zeros() as usize;
                delta += inter2[idx * k + j];
                pm ^= l;
            }
            stack.push((idx + 1, new_w, cv + delta, mask | (1usize << idx)));
        }
    }

    if best_mask == cur_mask {
        return;
    }

    let mut to_remove: Vec<usize> = Vec::new();
    let mut to_add: Vec<usize> = Vec::new();
    for t in 0..k {
        let it = cand2[t];
        let want = ((best_mask >> t) & 1) != 0;
        let have = state.selected_bit[it];
        if have && !want {
            to_remove.push(it);
        } else if !have && want {
            to_add.push(it);
        }
    }

    for &it in &to_remove {
        state.remove_item(it);
    }
    for &it in &to_add {
        if state.total_weight + state.ch.weights[it] <= state.ch.max_weight {
            state.add_item(it);
        }
    }
}