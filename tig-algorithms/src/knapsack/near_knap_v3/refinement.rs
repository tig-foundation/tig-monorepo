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

    let mut locked_sel: Vec<usize> = Vec::with_capacity(state.window_locked.len());
    for &i in &state.window_locked {
        if state.selected_bit[i] {
            locked_sel.push(i);
        }
    }

    for _ in 0..passes {
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

            if let Some(ref ng) = state.neigh {
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
            &locked_sel,
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

    let micro_k: usize = if big_team {
        12
    } else if team_est <= 160 {
        18
    } else {
        MICRO_K
    };
    let rm_k: usize = if big_team {
        6
    } else if team_est <= 160 {
        9
    } else {
        MICRO_RM_K
    };
    let add_k: usize = if big_team {
        6
    } else if team_est <= 160 {
        9
    } else {
        MICRO_ADD_K
    };

    let mut sel: Vec<usize> = Vec::new();
    let mut unsel: Vec<usize> = Vec::new();

    for &i in &state.window_core {
        if state.selected_bit[i] {
            sel.push(i);
        } else {
            unsel.push(i);
        }
    }

    if let Some(ref ng) = state.neigh {
        let mut guides: Vec<usize> = Vec::new();
        for &i in &state.window_core {
            if state.selected_bit[i] {
                guides.push(i);
            }
        }
        guides.sort_unstable_by(|&a, &b| {
            state.support[b]
                .cmp(&state.support[a])
                .then_with(|| state.contrib[b].cmp(&state.contrib[a]))
                .then_with(|| b.cmp(&a))
        });

        let push_unsel = |v: &mut Vec<usize>, x: usize| {
            for &y in v.iter() {
                if y == x {
                    return;
                }
            }
            v.push(x);
        };

        let g = guides.len().min(if big_team { 4 } else { 6 });
        for t in 0..g {
            let vtx = guides[t];
            let row = unsafe { ng.get_unchecked(vtx) };
            let pref = row.len().min(if big_team { 16 } else { 24 });
            for u in 0..pref {
                let cand = row[u].0 as usize;
                if !state.selected_bit[cand] {
                    push_unsel(&mut unsel, cand);
                }
            }
        }

        let hub_take: usize = if big_team {
            5
        } else if team_est <= 160 {
            10
        } else {
            8
        };
        let mut added_hubs: usize = 0;
        let lim = state.hubs_static.len().min(192);
        for &h in state.hubs_static.iter().take(lim) {
            if added_hubs >= hub_take {
                break;
            }
            if state.selected_bit[h] {
                continue;
            }
            push_unsel(&mut unsel, h);
            added_hubs += 1;

            let row = unsafe { ng.get_unchecked(h) };
            let pref = row.len().min(16);
            for u in 0..pref {
                let cand = row[u].0 as usize;
                if !state.selected_bit[cand] {
                    push_unsel(&mut unsel, cand);
                }
            }
        }
    }

    let extra_r = state.window_rejected.len().min(if big_team { 12 } else { 24 });
    for &i in &state.window_rejected[..extra_r] {
        if !state.selected_bit[i] {
            unsel.push(i);
        }
    }

    let extra_l = state.window_locked.len().min(24);
    let start_l = state.window_locked.len().saturating_sub(extra_l);
    for &i in &state.window_locked[start_l..] {
        if state.selected_bit[i] {
            sel.push(i);
        }
    }

    let score = |st: &State, i: usize| -> i64 {
        let w = (st.ch.weights[i] as i64).max(1);
        let dens = (st.contrib[i] as i64 * 1000) / w;
        dens + (st.support[i] as i64) * 120 + (st.total_interactions[i] / 320)
    };

    sel.sort_unstable_by(|&a, &b| score(state, a).cmp(&score(state, b)).then_with(|| a.cmp(&b)));
    unsel.sort_unstable_by(|&a, &b| {
        score(state, b)
            .cmp(&score(state, a))
            .then_with(|| b.cmp(&a))
    });

    let mut cand: Vec<usize> = Vec::with_capacity(micro_k);
    let push_u = |v: &mut Vec<usize>, x: usize| {
        for &y in v.iter() {
            if y == x {
                return;
            }
        }
        v.push(x);
    };

    for &i in sel.iter().take(rm_k) {
        push_u(&mut cand, i);
        if cand.len() >= micro_k {
            break;
        }
    }
    for &i in unsel.iter().take(add_k) {
        push_u(&mut cand, i);
        if cand.len() >= micro_k {
            break;
        }
    }
    if cand.len() < 2 {
        return;
    }

    let k = cand.len();
    if k > 20 {
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
    let rem_cap: u32 = state.ch.max_weight - fixed_w;

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

    let mut cur_mask: usize = 0;
    for t in 0..k {
        if state.selected_bit[cand[t]] {
            cur_mask |= 1usize << t;
        }
    }

    let mut cur_w_sum: u32 = 0;
    let mut cur_v: i64 = 0;
    for i in 0..k {
        if ((cur_mask >> i) & 1) == 0 {
            continue;
        }
        cur_w_sum += w[i];
        cur_v += base[i];
        for j in 0..i {
            if ((cur_mask >> j) & 1) != 0 {
                cur_v += inter[i * k + j];
            }
        }
    }
    if cur_w_sum > rem_cap {
        return;
    }

    let mmax: usize = 1usize << k;
    let mut wmask: Vec<u32> = vec![0; mmax];
    let mut vmask: Vec<i64> = vec![0; mmax];

    let mut best_mask: usize = cur_mask;
    let mut best_val: i64 = cur_v;

    for mask in 1..mmax {
        let lsb = mask & mask.wrapping_neg();
        let bi = lsb.trailing_zeros() as usize;
        let prev = mask ^ lsb;

        let ww = wmask[prev].saturating_add(w[bi]);
        if ww > rem_cap {
            wmask[mask] = u32::MAX;
            vmask[mask] = i64::MIN / 4;
            continue;
        }

        let mut val = vmask[prev] + base[bi];

        let mut pm = prev;
        while pm != 0 {
            let l = pm & pm.wrapping_neg();
            let j = l.trailing_zeros() as usize;
            val += inter[bi * k + j];
            pm ^= l;
        }

        wmask[mask] = ww;
        vmask[mask] = val;

        if val > best_val {
            best_val = val;
            best_mask = mask;
        }
    }

    if best_mask == cur_mask {
        return;
    }

    let mut to_remove: Vec<usize> = Vec::new();
    let mut to_add: Vec<usize> = Vec::new();
    for t in 0..k {
        let it = cand[t];
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
