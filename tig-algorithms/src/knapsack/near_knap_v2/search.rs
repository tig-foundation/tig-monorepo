use tig_challenges::knapsack::*;
use super::params::{Params, DIFF_LIM, MICRO_K, MICRO_RM_K, MICRO_ADD_K};
use super::state::{State, rebuild_windows};

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
    let passes = if state.window_core.len() <= 160 { 2 } else { 1 };
    let n = state.ch.num_items;

    for _ in 0..passes {
        let mut core_val: Vec<i32> = Vec::with_capacity(state.window_core.len());
        if !state.window_core.is_empty() {
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
                    for &(k, v) in row.iter() {
                        let j = k as usize;
                        if sel_core_bit[j] {
                            sub += v as i32;
                        }
                    }
                    let mut v0 = state.contrib[it] - (sub / 2);
                    if !state.selected_bit[it] {
                        v0 += (state.total_interactions[it] / 320) as i32
                            + (state.usage[it] as i32) * 12;
                    }
                    core_val.push(v0);
                }
            } else {
                for &it in &state.window_core {
                    let mut sub: i32 = 0;
                    for &j in &sel_core {
                        sub += state.ch.interaction_values[it][j];
                    }
                    let mut v0 = state.contrib[it] - (sub / 2);
                    if !state.selected_bit[it] {
                        v0 += (state.total_interactions[it] / 320) as i32
                            + (state.usage[it] as i32) * 12;
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

        let hub_take: usize = if big_team { 5 } else if team_est <= 160 { 10 } else { 8 };
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
    unsel.sort_unstable_by(|&a, &b| score(state, b).cmp(&score(state, a)).then_with(|| b.cmp(&a)));

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

    let edge_lim: usize = if n >= 4500 { 14000 } else if n >= 2500 { 11000 } else { 7000 };
    let node_lim: usize = if n >= 4500 { 48 } else if n >= 2500 { 56 } else { 64 };

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
    let max_iterations = params.vnd_max_iterations(n);
    let mut used: Vec<usize> = Vec::new();
    let mut micro_used = false;

    let mut frontier_swap_tries: usize = 0;
    let max_frontier_swaps: usize = if n >= 2500 {
        0
    } else if n >= 1500 {
        1
    } else {
        2
    };

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

        if apply_best_add_neigh_global(state) {
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
