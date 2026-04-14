use super::refinement::micro_qkp_refinement;
use super::types::{rebuild_windows, State, DIFF_LIM};

fn apply_best_add_windowed(state: &mut State) -> bool {
    const SUP_ADD_BONUS: i64 = 40;

    let slack = state.slack();
    if slack == 0 {
        return false;
    }

    let mut best: Option<(usize, i64, i32, u32)> = None;

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
            if best.map_or(true, |(_, bs, _, _)| s > bs) {
                best = Some((cand, s, delta, slack - state.ch.weights[cand]));
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
            if best.map_or(true, |(_, bs, _, _)| s > bs) {
                best = Some((cand, s, delta, slack - w));
            }
        }
    }

    if let Some((cand, _, single_gain_i32, single_residual)) = best {
        let tight_slack_lim = if state.ch.num_items >= 3000 {
            (DIFF_LIM as u32).saturating_mul(3).saturating_add(1)
        } else {
            (DIFF_LIM as u32).saturating_mul(4).saturating_add(1)
        };
        let awkward_single = single_residual
            > if state.ch.num_items >= 3000 {
                (DIFF_LIM as u32).saturating_mul(2)
            } else {
                (DIFF_LIM as u32).saturating_mul(3)
            };

        if slack <= tight_slack_lim || awkward_single {
            if let Some((a, Some(b), pair_gain)) = best_refill_move_windowed(state, slack) {
                let wa = state.ch.weights[a];
                let wb = state.ch.weights[b];
                let pair_used = wa.saturating_add(wb);
                if pair_used <= slack {
                    let pair_residual = slack - pair_used;
                    let pair_near_lim = if slack <= tight_slack_lim {
                        DIFF_LIM as u32
                    } else {
                        (DIFF_LIM as u32).saturating_mul(2)
                    };
                    let single_gain = single_gain_i32 as i64;
                    let pair_sup = (state.support[a] as i64) + (state.support[b] as i64);
                    let single_sup = state.support[cand] as i64;

                    let take_pair = pair_residual <= pair_near_lim
                        && (pair_gain > single_gain
                            || (pair_gain == single_gain
                                && (pair_residual < single_residual
                                    || (pair_residual == single_residual
                                        && pair_sup > single_sup))));

                    if take_pair {
                        state.add_item(a);
                        if !state.selected_bit[b]
                            && state.total_weight + wb <= state.ch.max_weight
                        {
                            state.add_item(b);
                            return true;
                        }
                        if state.selected_bit[a] {
                            state.remove_item(a);
                        }
                    }
                }
            }
        }

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

    #[inline]
    fn push_worst(list: &mut Vec<(i64, usize)>, limit: usize, s: i64, idx: usize) {
        if list.len() < limit {
            list.push((s, idx));
            return;
        }
        let mut worst_pos = 0usize;
        let mut worst_s = list[0].0;
        for t in 1..list.len() {
            if list[t].0 > worst_s {
                worst_s = list[t].0;
                worst_pos = t;
            }
        }
        if s < worst_s {
            list[worst_pos] = (s, idx);
        }
    }

    #[inline]
    fn push_top(list: &mut Vec<(i64, usize, i32)>, limit: usize, s: i64, idx: usize, delta: i32) {
        if list.len() < limit {
            list.push((s, idx, delta));
            return;
        }
        let mut worst_pos = 0usize;
        let mut worst_s = list[0].0;
        let mut worst_d = list[0].2;
        for t in 1..list.len() {
            if list[t].0 < worst_s || (list[t].0 == worst_s && list[t].2 < worst_d) {
                worst_s = list[t].0;
                worst_d = list[t].2;
                worst_pos = t;
            }
        }
        if s > worst_s || (s == worst_s && delta > worst_d) {
            list[worst_pos] = (s, idx, delta);
        }
    }

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

    let anchor_cap: usize = if n >= 4500 { 12 } else if n >= 2500 { 16 } else { 24 };
    let frontier_cap: usize = if n >= 4500 { 40 } else if n >= 2500 { 56 } else { 72 };
    let rej_lim: usize = if n >= 4500 {
        state.window_rejected.len().min(72)
    } else if n >= 2500 {
        state.window_rejected.len().min(112)
    } else {
        state.window_rejected.len().min(176)
    };
    let per_anchor_scan: usize = if n >= 4500 { 80 } else if n >= 2500 { 112 } else { 144 };
    let per_anchor_take: usize = if n >= 4500 { 12 } else if n >= 2500 { 16 } else { 20 };

    let mut anchor_pool: Vec<(i64, usize)> = Vec::with_capacity(anchor_cap * 4 + 32);

    for &i in &state.window_core {
        if state.selected_bit[i] {
            let c = state.contrib[i] as i64;
            let w = (state.ch.weights[i] as i64).max(1);
            let anti = 2 * c - state.total_interactions[i];
            let s = (c * 1000) / w + anti / 8 + (state.support[i] as i64) * 48 - 768;
            anchor_pool.push((s, i));
        }
    }

    let extra = state.window_locked.len().min(24);
    let start = state.window_locked.len().saturating_sub(extra);
    for &i in state.window_locked[start..].iter() {
        if state.selected_bit[i] {
            let c = state.contrib[i] as i64;
            let w = (state.ch.weights[i] as i64).max(1);
            let anti = 2 * c - state.total_interactions[i];
            let s = (c * 1000) / w + anti / 8 + (state.support[i] as i64) * 48 - 384;
            anchor_pool.push((s, i));
        }
    }

    let mut worst_selected: Vec<(i64, usize)> = Vec::with_capacity(anchor_cap * 2);
    for i in 0..n {
        if !state.selected_bit[i] {
            continue;
        }
        let c = state.contrib[i] as i64;
        let w = (state.ch.weights[i] as i64).max(1);
        let anti = 2 * c - state.total_interactions[i];
        let s = (c * 1000) / w + anti / 8 + (state.support[i] as i64) * 48;
        push_worst(&mut worst_selected, anchor_cap * 2, s, i);
    }
    anchor_pool.extend_from_slice(&worst_selected);

    if !anchor_pool.is_empty() {
        anchor_pool.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    }

    let mut anchor_seen = vec![false; n];
    let mut anchors: Vec<usize> = Vec::with_capacity(anchor_cap);
    for &(_, idx) in &anchor_pool {
        if !anchor_seen[idx] {
            anchor_seen[idx] = true;
            anchors.push(idx);
            if anchors.len() >= anchor_cap {
                break;
            }
        }
    }

    let mut frontier: Vec<(i64, usize, i32)> = Vec::with_capacity(frontier_cap);
    if !anchors.is_empty() {
        let mut seen = vec![false; n];
        let mut frontier_ids: Vec<usize> = Vec::with_capacity(frontier_cap * 2);
        let mut hit_count = vec![0u16; n];
        let mut best_link = vec![i64::MIN; n];
        let mut pos_sum = vec![0i64; n];

        for &anchor in &anchors {
            let row = unsafe { neigh.get_unchecked(anchor) };
            let mut scanned = 0usize;
            let mut taken = 0usize;

            for &(cj, vv) in row.iter() {
                scanned += 1;
                if scanned > per_anchor_scan || taken >= per_anchor_take {
                    break;
                }

                let cand = cj as usize;
                if cand >= n || state.selected_bit[cand] {
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

                if !seen[cand] {
                    seen[cand] = true;
                    frontier_ids.push(cand);
                }

                taken += 1;
                if hit_count[cand] < u16::MAX {
                    hit_count[cand] += 1;
                }

                let vv_i = vv as i64;
                if vv_i > best_link[cand] {
                    best_link[cand] = vv_i;
                }
                if vv_i > 0 {
                    pos_sum[cand] += vv_i.min(256);
                }
            }
        }

        for &cand in &state.window_rejected[..rej_lim] {
            if state.selected_bit[cand] || seen[cand] {
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

            seen[cand] = true;
            frontier_ids.push(cand);
        }

        for cand in frontier_ids {
            let w_u = state.ch.weights[cand];
            let delta = state.contrib[cand];
            let w = (w_u as i64).max(1);
            let c = delta as i64;
            let tot = state.total_interactions[cand];
            let fit = (slack - w_u) as i64;
            let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot);
            let link = if best_link[cand] == i64::MIN {
                0
            } else {
                best_link[cand].max(0)
            };
            let link_bonus = link * 96
                + pos_sum[cand].min(256) * 16
                + (hit_count[cand].saturating_sub(1) as i64) * 192;
            let s = (adj * 1000) / w
                + (state.support[cand] as i64) * SUP_ADD_BONUS
                - fit
                + link_bonus;

            push_top(&mut frontier, frontier_cap, s, cand, delta);
        }
    }

    if !frontier.is_empty() {
        let mut best: Option<(usize, i64, i32, u32)> = None;
        for &(s, cand, delta) in &frontier {
            let w = state.ch.weights[cand];
            if best.map_or(true, |(_, bs, bd, bw)| {
                s > bs || (s == bs && (delta > bd || (delta == bd && w > bw)))
            }) {
                best = Some((cand, s, delta, w));
            }
        }

        if let Some((cand, _, _, _)) = best {
            state.add_item(cand);
            return true;
        }
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

#[inline]
fn best_refill_move_windowed(state: &State, slack: u32) -> Option<(usize, Option<usize>, i64)> {
    #[inline]
    fn push_top(list: &mut Vec<(i64, usize)>, limit: usize, s: i64, idx: usize) {
        if list.len() < limit {
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
    }

    #[inline]
    fn push_top_unique(list: &mut Vec<(i64, usize)>, limit: usize, s: i64, idx: usize) {
        for ent in list.iter_mut() {
            if ent.1 == idx {
                if s > ent.0 {
                    ent.0 = s;
                }
                return;
            }
        }
        push_top(list, limit, s, idx);
    }

    #[inline]
    fn maybe_push_pool(
        pool: &mut Vec<usize>,
        band_counts: &mut [usize; 6],
        limit: usize,
        band_cap: usize,
        denom: u64,
        weight: u32,
        cand: usize,
        force: bool,
    ) {
        if pool.len() >= limit {
            return;
        }
        for &x in pool.iter() {
            if x == cand {
                return;
            }
        }

        let band = ((((weight as u64) * 6) / denom).min(5u64)) as usize;
        if band_counts[band] >= band_cap && !force {
            return;
        }

        band_counts[band] = band_counts[band].saturating_add(1);
        pool.push(cand);
    }

    #[inline]
    fn consider_best(
        best: &mut Option<(usize, Option<usize>, i64, i64, i64, i64, u32)>,
        a: usize,
        b: Option<usize>,
        delta: i64,
        residual: i64,
        inter: i64,
        sup: i64,
        used_w: u32,
    ) {
        if delta <= 0 {
            return;
        }

        let take = match best {
            None => true,
            Some((_, _, bd, br, bi, bs, bw)) => {
                delta > *bd
                    || (delta == *bd
                        && (residual < *br
                            || (residual == *br
                                && (inter > *bi
                                    || (inter == *bi
                                        && (sup > *bs || (sup == *bs && used_w > *bw)))))))
            }
        };

        if take {
            *best = Some((a, b, delta, residual, inter, sup, used_w));
        }
    }

    #[inline]
    fn anchor_score(state: &State, slack: u32, idx: usize) -> i64 {
        let w = state.ch.weights[idx];
        let delta = state.contrib[idx] as i64;
        let anti = 2 * delta - state.total_interactions[idx];
        let fit = (slack - w) as i64;
        let near_bonus = if fit == 0 {
            2048
        } else if fit <= DIFF_LIM as i64 {
            512
        } else if fit <= (DIFF_LIM as i64) * 2 {
            128
        } else {
            0
        };

        delta * 128 + anti * 6 + (state.support[idx] as i64) * 80 - fit * 4 + near_bonus
    }

    let n = state.ch.num_items;
    let top_cand = if n >= 3000 { 18 } else if n >= 1500 { 24 } else { 30 };
    let base_lim = top_cand * 2;
    let fit_lim = top_cand;
    let dens_lim = top_cand;
    let rej_lim = if n >= 3000 {
        state.window_rejected.len().min(96)
    } else {
        state.window_rejected.len().min(160)
    };

    let mut raw_base: Vec<(i64, usize)> = Vec::with_capacity(base_lim);
    let mut raw_fit: Vec<(i64, usize)> = Vec::with_capacity(fit_lim);
    let mut raw_dens: Vec<(i64, usize)> = Vec::with_capacity(dens_lim);

    for (bw, items) in &state.core_bins {
        if *bw > slack {
            break;
        }
        for &cand in items {
            if state.selected_bit[cand] {
                continue;
            }
            let w = state.ch.weights[cand];
            if w == 0 || w > slack {
                continue;
            }
            let delta = state.contrib[cand] as i64;
            if delta <= 0 {
                continue;
            }

            let anti = 2 * delta - state.total_interactions[cand];
            let fit = (slack - w) as i64;
            let sup = state.support[cand] as i64;
            let dens = (delta * 1000) / (w as i64).max(1);
            let near_bonus = if fit == 0 {
                2048
            } else if fit <= DIFF_LIM as i64 {
                512
            } else {
                0
            };

            let base_s = delta * 128 + anti * 8 + sup * 72 - fit;
            let fit_s = delta * 96 + anti * 4 + sup * 48 - fit * 6 + near_bonus;
            let dens_s = dens * 4 + anti / 4 + sup * 96 - fit / 2;

            push_top(&mut raw_base, base_lim, base_s, cand);
            push_top(&mut raw_fit, fit_lim, fit_s, cand);
            push_top(&mut raw_dens, dens_lim, dens_s, cand);
        }
    }

    for &cand in &state.window_rejected[..rej_lim] {
        if state.selected_bit[cand] {
            continue;
        }
        let w = state.ch.weights[cand];
        if w == 0 || w > slack {
            continue;
        }
        let delta = state.contrib[cand] as i64;
        if delta <= 0 {
            continue;
        }

        let anti = 2 * delta - state.total_interactions[cand];
        let fit = (slack - w) as i64;
        let sup = state.support[cand] as i64;
        let dens = (delta * 1000) / (w as i64).max(1);
        let near_bonus = if fit == 0 {
            2048
        } else if fit <= DIFF_LIM as i64 {
            512
        } else {
            0
        };

        let base_s = delta * 128 + anti * 8 + sup * 72 - fit;
        let fit_s = delta * 96 + anti * 4 + sup * 48 - fit * 6 + near_bonus;
        let dens_s = dens * 4 + anti / 4 + sup * 96 - fit / 2;

        push_top(&mut raw_base, base_lim, base_s, cand);
        push_top(&mut raw_fit, fit_lim, fit_s, cand);
        push_top(&mut raw_dens, dens_lim, dens_s, cand);
    }

    if raw_base.is_empty() {
        return None;
    }

    raw_base.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    raw_fit.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    raw_dens.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));

    let mut pool: Vec<usize> = Vec::with_capacity(top_cand);
    let mut band_counts = [0usize; 6];
    let denom = (slack as u64).max(1) + 1;
    let band_cap = if n >= 3000 { 3 } else { 4 };

    for &(_s, cand) in &raw_base {
        maybe_push_pool(
            &mut pool,
            &mut band_counts,
            top_cand,
            band_cap,
            denom,
            state.ch.weights[cand],
            cand,
            false,
        );
        if pool.len() >= top_cand {
            break;
        }
    }

    for &(_s, cand) in &raw_fit {
        let force = pool.len() + 4 < top_cand;
        maybe_push_pool(
            &mut pool,
            &mut band_counts,
            top_cand,
            band_cap,
            denom,
            state.ch.weights[cand],
            cand,
            force,
        );
        if pool.len() >= top_cand {
            break;
        }
    }

    for &(_s, cand) in &raw_dens {
        let force = pool.len() + 2 < top_cand;
        maybe_push_pool(
            &mut pool,
            &mut band_counts,
            top_cand,
            band_cap,
            denom,
            state.ch.weights[cand],
            cand,
            force,
        );
        if pool.len() >= top_cand {
            break;
        }
    }

    if pool.is_empty() {
        return None;
    }
    if pool.len() < 2 {
        if let Some(&cand) = pool.first() {
            if state.contrib[cand] > 0 {
                return Some((cand, None, state.contrib[cand] as i64));
            }
        }
        return None;
    }

    let mut best: Option<(usize, Option<usize>, i64, i64, i64, i64, u32)> = None;

    for &a in &pool {
        let delta = state.contrib[a] as i64;
        if delta <= 0 {
            continue;
        }

        consider_best(
            &mut best,
            a,
            None,
            delta,
            (slack - state.ch.weights[a]) as i64,
            0,
            state.support[a] as i64,
            state.ch.weights[a],
        );
    }

    let mut pair_pool = pool.clone();
    pair_pool.sort_unstable_by(|&a, &b| {
        anchor_score(state, slack, b)
            .cmp(&anchor_score(state, slack, a))
            .then_with(|| (state.contrib[b] as i64).cmp(&(state.contrib[a] as i64)))
            .then_with(|| b.cmp(&a))
    });

    let exact_len = if n >= 3000 {
        if slack <= (DIFF_LIM as u32).saturating_mul(3) {
            pair_pool.len().min(14)
        } else {
            pair_pool.len().min(12)
        }
    } else if n >= 1500 {
        pair_pool.len().min(18)
    } else {
        pair_pool.len()
    };

    for x in 0..exact_len {
        let a = pair_pool[x];
        let wa = state.ch.weights[a];

        for y in (x + 1)..exact_len {
            let b = pair_pool[y];
            let wb = state.ch.weights[b];
            let used_w = wa.saturating_add(wb);
            if used_w > slack {
                continue;
            }

            let inter = state.ch.interaction_values[a][b] as i64;
            let delta = (state.contrib[a] as i64) + (state.contrib[b] as i64) + inter;
            let sup = (state.support[a] as i64) + (state.support[b] as i64);

            consider_best(
                &mut best,
                a,
                Some(b),
                delta,
                (slack - used_w) as i64,
                inter,
                sup,
                used_w,
            );
        }
    }

    let anchor_cap = if n >= 3000 { 4 } else if n >= 1500 { 6 } else { 8 };
    let partner_cap = if n >= 3000 { 8 } else if n >= 1500 { 10 } else { 12 };
    let neigh_scan = if n >= 3000 { 24 } else if n >= 1500 { 32 } else { 40 };
    let core_scan = if n >= 3000 { 48 } else if n >= 1500 { 64 } else { 80 };
    let partner_rej_lim = if n >= 3000 {
        state.window_rejected.len().min(72)
    } else {
        state.window_rejected.len().min(96)
    };
    let near_band = if slack <= (DIFF_LIM as u32).saturating_mul(4) {
        (DIFF_LIM as u32)
            .saturating_mul(if n >= 3000 { 2 } else { 3 })
            .saturating_add(1)
    } else {
        (DIFF_LIM as u32)
            .saturating_mul(if n >= 3000 { 1 } else { 2 })
            .saturating_add(1)
    };
    let neigh = state.neigh;
    let core_band = if neigh.is_some() {
        near_band
    } else {
        near_band.saturating_mul(2).saturating_add(1)
    };
    let pool_seed_len = exact_len.min(if n >= 3000 { 10 } else { 14 });

    for &a in pair_pool.iter().take(anchor_cap) {
        let wa = state.ch.weights[a];
        if wa == 0 || wa >= slack {
            continue;
        }

        let residual_u = slack - wa;
        let lower = residual_u.saturating_sub(core_band);
        let mut partners: Vec<(i64, usize)> = Vec::with_capacity(partner_cap * 2 + 8);

        for &cand in pair_pool.iter().take(pool_seed_len) {
            if cand == a || state.selected_bit[cand] {
                continue;
            }

            let wc = state.ch.weights[cand];
            if wc == 0 || wc > residual_u {
                continue;
            }

            let fit = (residual_u - wc) as i64;
            let inter = state.ch.interaction_values[a][cand] as i64;
            let delta = state.contrib[cand] as i64;
            let near_bonus = if fit == 0 {
                1536
            } else if fit <= DIFF_LIM as i64 {
                384
            } else {
                0
            };

            let s = delta * 96
                + inter * 128
                + (state.support[cand] as i64) * 56
                - fit * 4
                + near_bonus;
            push_top_unique(&mut partners, partner_cap * 2, s, cand);
        }

        if let Some(ng) = neigh {
            let row = unsafe { ng.get_unchecked(a) };
            let mut scanned = 0usize;
            for &(cj, vv) in row.iter() {
                scanned += 1;
                if scanned > neigh_scan {
                    break;
                }

                let cand = cj as usize;
                if cand >= n || cand == a || state.selected_bit[cand] {
                    continue;
                }

                let wc = state.ch.weights[cand];
                if wc == 0 || wc > residual_u {
                    continue;
                }

                let delta = state.contrib[cand] as i64;
                if delta <= 0 {
                    continue;
                }

                let fit = (residual_u - wc) as i64;
                let near_bonus = if fit == 0 {
                    2048
                } else if fit <= DIFF_LIM as i64 {
                    512
                } else if fit <= core_band as i64 {
                    128
                } else {
                    0
                };

                let s = delta * 96
                    + (vv as i64) * 160
                    + (state.support[cand] as i64) * 56
                    - fit * 6
                    + near_bonus;
                push_top_unique(&mut partners, partner_cap * 2, s, cand);
            }
        }

        let mut core_scanned = 0usize;
        for (bw, items) in &state.core_bins {
            if *bw > residual_u {
                break;
            }
            if *bw < lower {
                continue;
            }

            let gap = residual_u - *bw;
            let near_bonus = if gap == 0 {
                1024
            } else if gap <= DIFF_LIM as u32 {
                256
            } else {
                64
            };

            for &cand in items {
                if core_scanned >= core_scan {
                    break;
                }
                core_scanned += 1;

                if cand == a || state.selected_bit[cand] {
                    continue;
                }

                let wc = state.ch.weights[cand];
                if wc == 0 || wc > residual_u {
                    continue;
                }

                let delta = state.contrib[cand] as i64;
                if delta <= 0 {
                    continue;
                }

                let inter = state.ch.interaction_values[a][cand] as i64;
                let anti = 2 * delta - state.total_interactions[cand];
                let s = delta * 96
                    + inter * 128
                    + anti * 2
                    + (state.support[cand] as i64) * 56
                    + near_bonus
                    - (gap as i64) * 10;
                push_top_unique(&mut partners, partner_cap * 2, s, cand);
            }

            if core_scanned >= core_scan {
                break;
            }
        }

        for &cand in &state.window_rejected[..partner_rej_lim] {
            if cand == a || state.selected_bit[cand] {
                continue;
            }

            let wc = state.ch.weights[cand];
            if wc == 0 || wc > residual_u {
                continue;
            }

            let gap = residual_u - wc;
            if gap > core_band.saturating_mul(2) {
                continue;
            }

            let delta = state.contrib[cand] as i64;
            if delta <= 0 {
                continue;
            }

            let inter = state.ch.interaction_values[a][cand] as i64;
            let near_bonus = if gap == 0 {
                1536
            } else if gap <= DIFF_LIM as u32 {
                384
            } else {
                96
            };

            let s = delta * 96
                + inter * 128
                + (state.support[cand] as i64) * 56
                + near_bonus
                - (gap as i64) * 8;
            push_top_unique(&mut partners, partner_cap * 2, s, cand);
        }

        partners.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));

        for &(_s, b) in partners.iter().take(partner_cap) {
            let wb = state.ch.weights[b];
            let used_w = wa.saturating_add(wb);
            if used_w > slack {
                continue;
            }

            let inter = state.ch.interaction_values[a][b] as i64;
            let delta = (state.contrib[a] as i64) + (state.contrib[b] as i64) + inter;
            let sup = (state.support[a] as i64) + (state.support[b] as i64);

            consider_best(
                &mut best,
                a,
                Some(b),
                delta,
                (slack - used_w) as i64,
                inter,
                sup,
                used_w,
            );
        }
    }

    best.map(|(a, b, delta, _, _, _, _)| (a, b, delta))
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

        let pa = 2 * (state.contrib[a] as i64) - state.total_interactions[a];
        let pb = 2 * (state.contrib[b] as i64) - state.total_interactions[b];

        sb.cmp(&sa)
            .then_with(|| pb.cmp(&pa))
            .then_with(|| (state.support[b] as i64).cmp(&(state.support[a] as i64)))
            .then_with(|| b.cmp(&a))
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

    let mut best: Option<(usize, usize, usize, i64, i64, i64)> = None;

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

                if delta <= 0 {
                    continue;
                }

                let proxy = (2 * (state.contrib[a] as i64) - state.total_interactions[a])
                    + (2 * (state.contrib[b] as i64) - state.total_interactions[b]);
                let sup_sum = (state.support[a] as i64) + (state.support[b] as i64);

                if best.map_or(true, |(_, _, _, bd, bp, bs)| {
                    delta > bd || (delta == bd && (proxy > bp || (proxy == bp && sup_sum > bs)))
                }) {
                    best = Some((r, a, b, delta, proxy, sup_sum));
                }
            }
        }
    }

    if let Some((r, a, b, _, _, _)) = best {
        state.remove_item(r);
        state.add_item(a);
        state.add_item(b);
        true
    } else {
        false
    }
}

fn apply_best_replace21_windowed(state: &mut State, used: &[usize]) -> bool {
    #[derive(Clone)]
    struct BeamPlan {
        added: Vec<usize>,
        gain: i64,
        residual: u32,
        score: i128,
        depth: u8,
    }

    #[inline]
    fn push_top_pair(
        list: &mut Vec<(i64, usize, usize, i64, u32)>,
        limit: usize,
        score: i64,
        a: usize,
        b: usize,
        net: i64,
        wsum: u32,
    ) {
        if list.len() < limit {
            list.push((score, a, b, net, wsum));
            return;
        }

        let mut worst_pos = 0usize;
        let mut worst_score = list[0].0;
        let mut worst_net = list[0].3;
        let mut worst_w = list[0].4;

        for t in 1..list.len() {
            let cur = list[t];
            if cur.0 < worst_score
                || (cur.0 == worst_score
                    && (cur.3 > worst_net || (cur.3 == worst_net && cur.4 < worst_w)))
            {
                worst_pos = t;
                worst_score = cur.0;
                worst_net = cur.3;
                worst_w = cur.4;
            }
        }

        if score > worst_score
            || (score == worst_score && (net < worst_net || (net == worst_net && wsum > worst_w)))
        {
            list[worst_pos] = (score, a, b, net, wsum);
        }
    }

    #[inline]
    fn rollback_items(state: &mut State, items: &[usize]) {
        for &x in items.iter().rev() {
            if state.selected_bit[x] {
                state.remove_item(x);
            }
        }
    }

    #[inline]
    fn apply_items(state: &mut State, items: &[usize], cap: u32) -> bool {
        let mut added = 0usize;
        for &x in items {
            if state.selected_bit[x] || state.total_weight + state.ch.weights[x] > cap {
                rollback_items(state, &items[..added]);
                return false;
            }
            state.add_item(x);
            added += 1;
        }
        true
    }

    #[inline]
    fn plan_score(gain: i64, residual: u32, depth: u8, add_count: usize, tie: i64) -> i128 {
        (gain as i128) * 1_000_000i128
            - (residual as i128) * 1024i128
            + (depth as i128) * 4096i128
            + (add_count as i128) * 512i128
            + (tie as i128)
    }

    #[inline]
    fn maybe_take_best(
        best: &mut Option<(usize, usize, Vec<usize>, i64, u8, u32)>,
        a: usize,
        b: usize,
        added: &[usize],
        gain: i64,
        depth: u8,
        residual: u32,
    ) {
        if gain <= 0 {
            return;
        }

        let take = match best {
            None => true,
            Some((_, _, badded, bg, bd, br)) => {
                gain > *bg
                    || (gain == *bg
                        && (residual < *br
                            || (residual == *br
                                && (depth > *bd
                                    || (depth == *bd && added.len() > badded.len())))))
            }
        };

        if take {
            *best = Some((a, b, added.to_vec(), gain, depth, residual));
        }
    }

    #[inline]
    fn try_refill_once(
        state: &mut State,
        cap: u32,
    ) -> Option<(Vec<usize>, i64, u32, i64)> {
        if state.slack() == 0 {
            return None;
        }

        let before_val = state.total_value as i64;
        let mut added: Vec<usize> = Vec::with_capacity(2);
        let mut tie = 0i64;

        let mv = match best_refill_move_windowed(state, state.slack()) {
            Some(mv) => mv,
            None => return None,
        };

        let a = mv.0;
        if !state.selected_bit[a] && state.total_weight + state.ch.weights[a] <= cap {
            tie += (state.support[a] as i64) * 64 + (state.ch.weights[a] as i64);
            state.add_item(a);
            added.push(a);
        }

        if let Some(b) = mv.1 {
            if !state.selected_bit[b] && state.total_weight + state.ch.weights[b] <= cap {
                tie += (state.support[b] as i64) * 64 + (state.ch.weights[b] as i64);
                state.add_item(b);
                added.push(b);
            }
        }

        let gain = (state.total_value as i64) - before_val;
        if added.is_empty() || gain <= 0 {
            rollback_items(state, &added);
            None
        } else {
            Some((added, gain, state.slack(), tie + mv.2))
        }
    }

    let cap = state.ch.max_weight;
    if used.len() < 2 {
        return false;
    }

    let old_val = state.total_value as i64;
    let n = state.ch.num_items;
    let pair_cap: usize = if n >= 3000 { 24 } else if n >= 1500 { 32 } else { 40 };
    let add_cap: usize = if n >= 3000 { 22 } else if n >= 1500 { 26 } else { 30 };
    let extra_rej: usize = if n >= 3000 { 48 } else { 64 };
    let ext_pair_lim: usize = if n >= 3000 { 4 } else if n >= 1500 { 5 } else { 6 };
    let ext_cand_lim: usize = if n >= 3000 { 4 } else if n >= 1500 { 5 } else { 6 };

    let mut rm_items: Vec<usize> = Vec::with_capacity(used.len());
    for &i in used {
        if state.selected_bit[i] {
            rm_items.push(i);
        }
    }
    if rm_items.len() < 2 {
        return false;
    }

    let mut rm_pairs: Vec<(i64, usize, usize, i64, u32)> = Vec::with_capacity(pair_cap);
    for x in 0..rm_items.len() {
        let a = rm_items[x];
        let wa = state.ch.weights[a];
        let ca = state.contrib[a] as i64;
        let anti_a = 2 * ca - state.total_interactions[a];
        let sup_a = state.support[a] as i64;

        for y in (x + 1)..rm_items.len() {
            let b = rm_items[y];
            let wb = state.ch.weights[b];
            let cb = state.contrib[b] as i64;
            let inter_ab = state.ch.interaction_values[a][b] as i64;
            let wsum = wa.saturating_add(wb);
            let net = ca + cb - inter_ab;
            let anti = anti_a + (2 * cb - state.total_interactions[b]);
            let sup = sup_a + (state.support[b] as i64);
            let dens = (net * 1000) / (wsum as i64).max(1);
            let score = inter_ab * 128 - dens - net * 8 - anti / 8 - sup * 24 + (wsum as i64);

            push_top_pair(&mut rm_pairs, pair_cap, score, a, b, net, wsum);
        }
    }

    if !rm_pairs.is_empty() {
        rm_pairs.sort_unstable_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| a.3.cmp(&b.3))
                .then_with(|| b.4.cmp(&a.4))
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });
    }

    let mut ext_cands: Vec<usize> = Vec::with_capacity(ext_cand_lim * 2 + 4);

    if rm_pairs.len() >= 3 {
        let compat_lim = rm_pairs.len().min(if n >= 3000 { 10 } else { 14 });

        let mut seen = vec![false; n];
        let mut add_scored: Vec<(i64, usize, i64)> =
            Vec::with_capacity(state.window_core.len() + extra_rej);

        for &cand in &state.window_core {
            if state.selected_bit[cand] || seen[cand] || state.contrib[cand] <= 0 {
                continue;
            }
            seen[cand] = true;

            let wc = state.ch.weights[cand];
            if wc == 0 {
                continue;
            }

            let c = state.contrib[cand] as i64;
            let anti = 2 * c - state.total_interactions[cand];
            let base =
                (c * 1000) / (wc as i64).max(1) + anti / 8 + (state.support[cand] as i64) * 80;

            let mut best_proxy = i64::MIN;
            for t in 0..compat_lim {
                let (_, a, b, net, _) = rm_pairs[t];
                let proxy = c
                    - (state.ch.interaction_values[cand][a] as i64)
                    - (state.ch.interaction_values[cand][b] as i64)
                    - net;
                if proxy > best_proxy {
                    best_proxy = proxy;
                }
            }

            add_scored.push((base * 4 + best_proxy * 128, cand, best_proxy));
        }

        let extra = state.window_rejected.len().min(extra_rej);
        for &cand in &state.window_rejected[..extra] {
            if state.selected_bit[cand] || seen[cand] || state.contrib[cand] <= 0 {
                continue;
            }
            seen[cand] = true;

            let wc = state.ch.weights[cand];
            if wc == 0 {
                continue;
            }

            let c = state.contrib[cand] as i64;
            let anti = 2 * c - state.total_interactions[cand];
            let base =
                (c * 1000) / (wc as i64).max(1) + anti / 8 + (state.support[cand] as i64) * 80;

            let mut best_proxy = i64::MIN;
            for t in 0..compat_lim {
                let (_, a, b, net, _) = rm_pairs[t];
                let proxy = c
                    - (state.ch.interaction_values[cand][a] as i64)
                    - (state.ch.interaction_values[cand][b] as i64)
                    - net;
                if proxy > best_proxy {
                    best_proxy = proxy;
                }
            }

            add_scored.push((base * 4 + best_proxy * 128, cand, best_proxy));
        }

        add_scored.sort_unstable_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| b.2.cmp(&a.2))
                .then_with(|| b.1.cmp(&a.1))
        });
        if add_scored.len() > add_cap {
            add_scored.truncate(add_cap);
        }

        let mut best_pair_first: Option<(usize, usize, usize, i64, i64)> = None;

        for &(pair_score, a, b, _net, wsum) in &rm_pairs {
            let wa = state.ch.weights[a];
            let wb = state.ch.weights[b];

            for &(_cand_score, cand, proxy) in &add_scored {
                let wc = state.ch.weights[cand];
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
                if delta <= 0 {
                    continue;
                }

                let tie = ((wsum as i64) - (wc as i64)) * 256
                    + (state.support[cand] as i64) * 32
                    + pair_score / 8
                    + proxy;

                if best_pair_first
                    .map_or(true, |(_, _, _, bd, bt)| delta > bd || (delta == bd && tie > bt))
                {
                    best_pair_first = Some((cand, a, b, delta, tie));
                }
            }
        }

        if let Some((cand, a, b, _, _)) = best_pair_first {
            state.remove_item(a);
            state.remove_item(b);
            if !state.selected_bit[cand] && state.total_weight + state.ch.weights[cand] <= cap {
                state.add_item(cand);
            }
            return true;
        }

        for &(_cand_score, cand, _proxy) in add_scored.iter().take(ext_cand_lim) {
            ext_cands.push(cand);
        }
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

        let ratio_cmp = (cb * wa).cmp(&(ca * wb));
        if ratio_cmp != core::cmp::Ordering::Equal {
            return ratio_cmp;
        }

        let mut min_a: i64 = i64::MAX;
        let mut min_b: i64 = i64::MAX;
        for &r in &rm {
            let va = state.ch.interaction_values[a][r] as i64;
            if va < min_a {
                min_a = va;
            }
            let vb = state.ch.interaction_values[b][r] as i64;
            if vb < min_b {
                min_b = vb;
            }
        }

        min_a.cmp(&min_b).then_with(|| b.cmp(&a))
    });
    if add_pool.len() > 28 {
        add_pool.truncate(28);
    }

    if !add_pool.is_empty() {
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
            return true;
        }

        for &cand in add_pool.iter().take(ext_cand_lim) {
            let mut dup = false;
            for &x in &ext_cands {
                if x == cand {
                    dup = true;
                    break;
                }
            }
            if !dup {
                ext_cands.push(cand);
                if ext_cands.len() >= ext_cand_lim * 2 {
                    break;
                }
            }
        }
    }

    if rm_pairs.is_empty() {
        return false;
    }

    let tight_slack = state.slack()
        <= if n >= 3000 {
            (DIFF_LIM as u32).saturating_mul(2)
        } else {
            (DIFF_LIM as u32).saturating_mul(3)
        };
    let dense_used = used.len()
        >= if n >= 3000 {
            10
        } else if n >= 1500 {
            8
        } else {
            6
        };
    let beam_active = tight_slack || dense_used;
    let pair_take = rm_pairs
        .len()
        .min(if beam_active { ext_pair_lim.max(2) } else { 2 });
    let beam_width: usize = if beam_active {
        if n >= 3000 { 3 } else { 4 }
    } else {
        2
    };
    let explicit_seed_lim: usize = if beam_active {
        ext_cand_lim.min(3)
    } else {
        1
    };

    let mut best_plan: Option<(usize, usize, Vec<usize>, i64, u8, u32)> = None;

    for &(_, a, b, _net, _wsum) in rm_pairs.iter().take(pair_take) {
        if !state.selected_bit[a] || !state.selected_bit[b] {
            continue;
        }

        state.remove_item(a);
        state.remove_item(b);

        let base_gain = (state.total_value as i64) - old_val;
        let base_residual = state.slack();

        let mut beam: Vec<BeamPlan> = Vec::with_capacity(beam_width + explicit_seed_lim + 3);
        beam.push(BeamPlan {
            added: Vec::new(),
            gain: base_gain,
            residual: base_residual,
            score: plan_score(base_gain, base_residual, 0, 0, 0),
            depth: 0,
        });
        maybe_take_best(&mut best_plan, a, b, &[], base_gain, 0, base_residual);

        let mut pair_explicit: Vec<(i64, usize)> = Vec::with_capacity(ext_cands.len());
        for &cand in &ext_cands {
            if state.selected_bit[cand] {
                continue;
            }

            let wc = state.ch.weights[cand];
            if wc == 0 || state.total_weight + wc > cap {
                continue;
            }

            let c = state.contrib[cand] as i64;
            if c <= 0 {
                continue;
            }

            let anti = 2 * c - state.total_interactions[cand];
            let fit = (state.slack() - wc) as i64;
            let pen_a = (state.ch.interaction_values[cand][a] as i64).max(0);
            let pen_b = (state.ch.interaction_values[cand][b] as i64).max(0);
            let score = c * 128
                + anti * 4
                + (state.support[cand] as i64) * 64
                - fit * 4
                - (pen_a + pen_b) * 8;

            pair_explicit.push((score, cand));
        }

        if !pair_explicit.is_empty() {
            pair_explicit.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
            if pair_explicit.len() > explicit_seed_lim {
                pair_explicit.truncate(explicit_seed_lim);
            }

            for &(seed_tie, cand) in &pair_explicit {
                state.add_item(cand);

                let gain = (state.total_value as i64) - old_val;
                let residual = state.slack();
                let added = vec![cand];

                beam.push(BeamPlan {
                    added: added.clone(),
                    gain,
                    residual,
                    score: plan_score(gain, residual, 1, 1, seed_tie),
                    depth: 1,
                });
                maybe_take_best(&mut best_plan, a, b, &added, gain, 1, residual);

                state.remove_item(cand);
            }
        }

        if let Some((added, _step_gain, residual, tie)) = try_refill_once(state, cap) {
            let gain = (state.total_value as i64) - old_val;

            beam.push(BeamPlan {
                added: added.clone(),
                gain,
                residual,
                score: plan_score(gain, residual, 1, added.len(), tie),
                depth: 1,
            });
            maybe_take_best(&mut best_plan, a, b, &added, gain, 1, residual);

            rollback_items(state, &added);
        }

        beam.sort_unstable_by(|a, b| {
            b.score
                .cmp(&a.score)
                .then_with(|| b.gain.cmp(&a.gain))
                .then_with(|| a.residual.cmp(&b.residual))
                .then_with(|| b.added.len().cmp(&a.added.len()))
        });
        if beam.len() > beam_width {
            beam.truncate(beam_width);
        }

        let expand_depth = if beam_active { beam.len() } else { beam.len().min(1) };
        for plan in beam.iter().take(expand_depth) {
            if !apply_items(state, &plan.added, cap) {
                continue;
            }

            if let Some((added2, _step_gain2, residual2, tie2)) = try_refill_once(state, cap) {
                let mut full_added = plan.added.clone();
                full_added.extend_from_slice(&added2);
                let gain2 = (state.total_value as i64) - old_val;
                let depth2 = plan.depth.saturating_add(1);

                maybe_take_best(&mut best_plan, a, b, &full_added, gain2, depth2, residual2);

                rollback_items(state, &added2);
            }

            rollback_items(state, &plan.added);
        }

        state.add_item(a);
        state.add_item(b);
    }

    if let Some((a, b, added, _, _, _)) = best_plan {
        state.remove_item(a);
        state.remove_item(b);
        for x in added {
            if !state.selected_bit[x] && state.total_weight + state.ch.weights[x] <= cap {
                state.add_item(x);
            }
        }
        true
    } else {
        false
    }
}

#[inline]
fn apply_best_swap_diff_reduce_windowed_cached(state: &mut State, used: &[usize]) -> bool {
    #[inline]
    fn push_setup(
        list: &mut Vec<(usize, usize, u32, i64)>,
        limit: usize,
        cand: usize,
        rm: usize,
        dw: u32,
        sup_diff: i64,
    ) {
        if list.len() < limit {
            list.push((cand, rm, dw, sup_diff));
            return;
        }

        let mut worst_pos = 0usize;
        let mut worst_dw = list[0].2;
        let mut worst_sup = list[0].3;
        for t in 1..list.len() {
            let cur = list[t];
            if cur.2 < worst_dw || (cur.2 == worst_dw && cur.3 < worst_sup) {
                worst_pos = t;
                worst_dw = cur.2;
                worst_sup = cur.3;
            }
        }

        if dw > worst_dw || (dw == worst_dw && sup_diff > worst_sup) {
            list[worst_pos] = (cand, rm, dw, sup_diff);
        }
    }

    let mut best_imp: Option<(usize, usize, i32, u32, i64)> = None;
    let neutral_cap: usize = if state.ch.num_items >= 3000 { 8 } else { 12 };
    let mut neutral_setups: Vec<(usize, usize, u32, i64)> = Vec::with_capacity(neutral_cap);

    for &rm in used {
        let w_rm = state.ch.weights[rm];
        if w_rm == 0 {
            continue;
        }
        let w_min = w_rm.saturating_sub(DIFF_LIM as u32);
        let sup_rm = state.support[rm] as i64;

        for (bw, items) in &state.core_bins {
            if *bw >= w_rm {
                break;
            }
            if *bw < w_min {
                continue;
            }
            let dw = w_rm - *bw;
            if dw == 0 {
                continue;
            }

            for &cand in items {
                if state.selected_bit[cand] {
                    continue;
                }

                let delta = state.contrib[cand]
                    - state.contrib[rm]
                    - state.ch.interaction_values[cand][rm];

                let sup_diff = (state.support[cand] as i64) - sup_rm;

                if delta > 0 {
                    let take = match best_imp {
                        None => true,
                        Some((_bc, _br, bd, bdw, bsd)) => {
                            delta > bd
                                || (delta == bd
                                    && (dw > bdw || (dw == bdw && sup_diff > bsd)))
                        }
                    };
                    if take {
                        best_imp = Some((cand, rm, delta, dw, sup_diff));
                    }
                } else if delta == 0 {
                    push_setup(&mut neutral_setups, neutral_cap, cand, rm, dw, sup_diff);
                }
            }
        }
    }

    if let Some((cand, rm, _, _, _)) = best_imp {
        state.replace_item(rm, cand);
        return true;
    }

    let mut best_combo: Option<(usize, usize, usize, Option<usize>, i64, u32, i64)> = None;

    for &(cand, rm, dw, sup_diff) in &neutral_setups {
        if !state.selected_bit[rm] || state.selected_bit[cand] {
            continue;
        }

        state.replace_item(rm, cand);

        let refill = if state.slack() > 0 {
            best_refill_move_windowed(state, state.slack())
        } else {
            None
        };

        state.replace_item(cand, rm);

        if let Some((a, b, gain)) = refill {
            if gain <= 0 {
                continue;
            }

            let take = match best_combo {
                None => true,
                Some((_bc, _br, _ba, bb, bg, bdw, bsd)) => {
                    gain > bg
                        || (gain == bg
                            && (dw > bdw
                                || (dw == bdw
                                    && (sup_diff > bsd
                                        || (sup_diff == bsd
                                            && (b.is_some() as u8) > (bb.is_some() as u8))))))
                }
            };

            if take {
                best_combo = Some((cand, rm, a, b, gain, dw, sup_diff));
            }
        }
    }

    if let Some((cand, rm, a, b, _, _, _)) = best_combo {
        state.replace_item(rm, cand);
        if !state.selected_bit[a] && state.total_weight + state.ch.weights[a] <= state.ch.max_weight {
            state.add_item(a);
        }
        if let Some(b2) = b {
            if !state.selected_bit[b2]
                && state.total_weight + state.ch.weights[b2] <= state.ch.max_weight
            {
                state.add_item(b2);
            }
        }
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

    let mut best: Option<(usize, usize, i32, u32, u32, i64)> = None;

    for &rm in used {
        let w_rm = state.ch.weights[rm];
        let max_dw = (DIFF_LIM as u32).min(slack);
        let w_max = w_rm.saturating_add(max_dw);
        let sup_rm = state.support[rm] as i64;

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
            let residual = slack - dw;

            for &cand in items {
                if state.selected_bit[cand] {
                    continue;
                }

                let delta = state.contrib[cand]
                    - state.contrib[rm]
                    - state.ch.interaction_values[cand][rm];
                if delta <= 0 {
                    continue;
                }

                let sup_diff = (state.support[cand] as i64) - sup_rm;

                let take = match best {
                    None => true,
                    Some((_bc, _br, bd, bdw, bres, bsup)) => {
                        delta > bd
                            || (delta == bd
                                && (residual < bres
                                    || (residual == bres
                                        && (((delta as i128) * (bdw as i128))
                                            > ((bd as i128) * (dw as i128))
                                            || (((delta as i128) * (bdw as i128))
                                                == ((bd as i128) * (dw as i128))
                                                && (sup_diff > bsup
                                                    || (sup_diff == bsup && dw < bdw)))))))
                    }
                };

                if take {
                    best = Some((cand, rm, delta, dw, residual, sup_diff));
                }
            }
        }
    }

    if let Some((cand, rm, _, _, _, _)) = best {
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

    let n = state.ch.num_items;
    if n == 0 {
        return false;
    }

    let frontier_cap: usize = if n >= 4500 { 64 } else if n >= 2500 { 80 } else { 112 };
    let slack0 = state.slack();

    let mut rm_list: Vec<usize> = Vec::with_capacity(used.len());
    let mut cand_union: Vec<usize> = Vec::with_capacity(frontier_cap * 3);
    let mut seen = vec![false; n];
    let mut hits = vec![0u16; n];
    let mut best_proxy = vec![i64::MIN; n];
    let mut best_bonus = vec![0i64; n];

    for &rm in used {
        if !state.selected_bit[rm] {
            continue;
        }
        rm_list.push(rm);

        let rm_contrib = state.contrib[rm] as i64;
        let row = unsafe { neigh.get_unchecked(rm) };

        for &(cj, vv) in row.iter() {
            let cand = cj as usize;
            if cand >= n || state.selected_bit[cand] {
                continue;
            }

            if !seen[cand] {
                seen[cand] = true;
                cand_union.push(cand);
            }

            if hits[cand] < u16::MAX {
                hits[cand] += 1;
            }

            let proxy = (state.contrib[cand] as i64) - rm_contrib - (vv as i64);
            if proxy > best_proxy[cand] {
                best_proxy[cand] = proxy;
            }

            if vv < 0 {
                let relief = -(vv as i64);
                if relief > best_bonus[cand] {
                    best_bonus[cand] = relief;
                }
            }
        }
    }

    if rm_list.is_empty() || cand_union.is_empty() {
        return false;
    }

    cand_union.sort_unstable_by(|&a, &b| {
        let ca = state.contrib[a] as i64;
        let cb = state.contrib[b] as i64;
        let wa = (state.ch.weights[a] as i64).max(1);
        let wb = (state.ch.weights[b] as i64).max(1);
        let anti_a = 2 * ca - state.total_interactions[a];
        let anti_b = 2 * cb - state.total_interactions[b];

        let sa = best_proxy[a] * 4096
            + best_bonus[a] * 128
            + (hits[a] as i64) * 192
            + (ca * 1000) / wa
            + anti_a / 8
            + (state.support[a] as i64) * 80;
        let sb = best_proxy[b] * 4096
            + best_bonus[b] * 128
            + (hits[b] as i64) * 192
            + (cb * 1000) / wb
            + anti_b / 8
            + (state.support[b] as i64) * 80;

        sb.cmp(&sa)
            .then_with(|| best_proxy[b].cmp(&best_proxy[a]))
            .then_with(|| best_bonus[b].cmp(&best_bonus[a]))
            .then_with(|| hits[b].cmp(&hits[a]))
            .then_with(|| (state.support[b] as i64).cmp(&(state.support[a] as i64)))
            .then_with(|| b.cmp(&a))
    });

    if cand_union.len() > frontier_cap {
        cand_union.truncate(frontier_cap);
    }

    let mut best: Option<(usize, usize, i128, i64, usize, i64, i64, i64, i64)> = None;

    for &cand in &cand_union {
        let wc = state.ch.weights[cand];
        if wc == 0 {
            continue;
        }

        let cand_contrib = state.contrib[cand] as i64;
        let anti = 2 * cand_contrib - state.total_interactions[cand];
        let sup = state.support[cand] as i64;
        let hit = hits[cand] as i64;
        let bonus_seen = best_bonus[cand];

        let mut best_rm: Option<(usize, i64, i64, i64, i64)> = None;
        let mut improving_count: usize = 0;
        let mut compat_count: usize = 0;
        let mut conflict_count: usize = 0;
        let mut top1 = 0i64;
        let mut top2 = 0i64;
        let mut top3 = 0i64;
        let mut neg_relief1 = 0i64;
        let mut neg_relief2 = 0i64;

        for &rm in &rm_list {
            let wrm = state.ch.weights[rm];
            if (state.total_weight as u64) + (wc as u64) > (cap as u64) + (wrm as u64) {
                continue;
            }

            let inter = state.ch.interaction_values[cand][rm] as i64;
            if inter < 0 {
                let relief = -inter;
                conflict_count += 1;
                if relief > neg_relief1 {
                    neg_relief2 = neg_relief1;
                    neg_relief1 = relief;
                } else if relief > neg_relief2 {
                    neg_relief2 = relief;
                }
            }

            let delta = cand_contrib - (state.contrib[rm] as i64) - inter;
            if delta <= 0 {
                continue;
            }

            improving_count += 1;
            if inter <= 0 {
                compat_count += 1;
            }

            if delta > top1 {
                top3 = top2;
                top2 = top1;
                top1 = delta;
            } else if delta > top2 {
                top3 = top2;
                top2 = delta;
            } else if delta > top3 {
                top3 = delta;
            }

            let fit = (slack0 as i64) + (wrm as i64) - (wc as i64);
            let pen = if inter > 0 { inter } else { 0 };
            let rm_sup = state.support[rm] as i64;

            let take = match best_rm {
                None => true,
                Some((_br, bd, bfit, bpen, brsup)) => {
                    delta > bd
                        || (delta == bd
                            && (fit > bfit
                                || (fit == bfit
                                    && (pen < bpen || (pen == bpen && rm_sup < brsup)))))
                }
            };

            if take {
                best_rm = Some((rm, delta, fit, pen, rm_sup));
            }
        }

        if let Some((rm, delta, fit, pen, rm_sup)) = best_rm {
            let second_best = top2;
            let third_best = top3;

            let count_bonus = (improving_count.saturating_sub(1).min(4) as i64) * 16_384;
            let compat_bonus = (compat_count.min(4) as i64) * 6_144;
            let conflict_bonus = (conflict_count.min(4) as i64) * 2_048
                + neg_relief1.min(128) * 64
                + neg_relief2.min(64) * 16;
            let second_bonus = second_best.min(512) * 192;
            let third_bonus = third_best.min(256) * 64;
            let frontier_bonus = hit.min(6) * 2_048 + bonus_seen.min(128) * 32;
            let shape_bonus = sup * 128 + anti * 4 - pen * 16 - rm_sup * 4;

            let robust_bonus = count_bonus
                + compat_bonus
                + conflict_bonus
                + second_bonus
                + third_bonus
                + frontier_bonus
                + shape_bonus;

            let score =
                (delta as i128) * 1_000_000i128 + (robust_bonus as i128);

            if best.map_or(true, |(_, _, bs, bd, bcnt, bsecond, bfit, bsup, banti)| {
                score > bs
                    || (score == bs
                        && (delta > bd
                            || (delta == bd
                                && (improving_count > bcnt
                                    || (improving_count == bcnt
                                        && (second_best > bsecond
                                            || (second_best == bsecond
                                                && (fit > bfit
                                                    || (fit == bfit
                                                        && (sup > bsup
                                                            || (sup == bsup && anti > banti)))))))))))
            }) {
                best = Some((
                    cand,
                    rm,
                    score,
                    delta,
                    improving_count,
                    second_best,
                    fit,
                    sup,
                    anti,
                ));
            }
        }
    }

    if let Some((cand, rm, _, _, _, _, _, _, _)) = best {
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

    #[inline]
    fn push_top(list: &mut Vec<(i64, usize)>, s: i64, idx: usize) {
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
    }

    #[inline]
    fn push_local(
        list: &mut Vec<(i64, usize, u32, i64, i64, bool)>,
        limit: usize,
        s: i64,
        idx: usize,
        gap: u32,
        proxy: i64,
        link: i64,
        src_hit: bool,
    ) {
        for ent in list.iter_mut() {
            if ent.1 == idx {
                if s > ent.0
                    || (s == ent.0
                        && (proxy > ent.3 || (proxy == ent.3 && gap < ent.2)))
                {
                    *ent = (s, idx, gap, proxy, link, src_hit || ent.5);
                } else if src_hit {
                    ent.5 = true;
                }
                return;
            }
        }

        if list.len() < limit {
            list.push((s, idx, gap, proxy, link, src_hit));
            return;
        }

        let mut worst_pos = 0usize;
        let mut worst = list[0];
        for t in 1..list.len() {
            let cur = list[t];
            if cur.0 < worst.0
                || (cur.0 == worst.0
                    && (cur.3 < worst.3 || (cur.3 == worst.3 && cur.2 > worst.2)))
            {
                worst_pos = t;
                worst = cur;
            }
        }

        if s > worst.0 || (s == worst.0 && (proxy > worst.3 || (proxy == worst.3 && gap < worst.2)))
        {
            list[worst_pos] = (s, idx, gap, proxy, link, src_hit);
        }
    }

    #[inline]
    fn push_unique_rm(list: &mut Vec<usize>, idx: usize) {
        for &x in list.iter() {
            if x == idx {
                return;
            }
        }
        list.push(idx);
    }

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
    if rm.is_empty() {
        return false;
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
    let max_swap_w = slack0.saturating_add(max_rm_w);

    let target_cap: usize = if n >= 3000 { 4 } else if n >= 1500 { 5 } else { 6 };
    let frontier_band: u32 = if n >= 3000 {
        (DIFF_LIM as u32).saturating_mul(2).saturating_add(1)
    } else {
        (DIFF_LIM as u32).saturating_mul(3).saturating_add(1)
    };
    let anchor_cap: usize = if n >= 3000 { 24 } else if n >= 1500 { 28 } else { 32 };
    let neigh_scan: usize = if n >= 3000 { 24 } else if n >= 1500 { 32 } else { 40 };
    let per_target_core: usize = if n >= 3000 { 18 } else if n >= 1500 { 24 } else { 30 };
    let per_target_rej: usize = if n >= 3000 { 10 } else if n >= 1500 { 14 } else { 18 };
    let rej_lim: usize = if n >= 3000 {
        state.window_rejected.len().min(72)
    } else if n >= 1500 {
        state.window_rejected.len().min(96)
    } else {
        state.window_rejected.len().min(128)
    };
    let per_target_frontier_cap: usize = if n >= 3000 { 12 } else if n >= 1500 { 14 } else { 16 };
    let per_target_take: usize = if n >= 3000 { 6 } else if n >= 1500 { 7 } else { 8 };
    let per_target_neigh: usize = if n >= 3000 { 18 } else if n >= 1500 { 24 } else { 30 };
    let per_target_anchor_neigh: usize = if n >= 3000 { 10 } else if n >= 1500 { 12 } else { 16 };

    let mut target_rms: Vec<usize> = Vec::with_capacity(target_cap);
    for &r in &rm {
        let tw = slack0.saturating_add(state.ch.weights[r]);
        let mut dup = false;
        for &old_r in &target_rms {
            let old_tw = slack0.saturating_add(state.ch.weights[old_r]);
            let gap = if tw > old_tw { tw - old_tw } else { old_tw - tw };
            if gap <= DIFF_LIM as u32 {
                dup = true;
                break;
            }
        }
        if !dup {
            target_rms.push(r);
            if target_rms.len() >= target_cap {
                break;
            }
        }
    }
    if target_rms.is_empty() {
        return false;
    }
    target_rms.sort_unstable_by_key(|&r| slack0.saturating_add(state.ch.weights[r]));

    let use_per_target_frontiers = target_rms.len() >= 2;

    let mut seen = vec![false; n];
    let mut cand_ids: Vec<usize> = Vec::with_capacity(TOP_CAND * 4);
    let mut source_hits = vec![0u16; n];
    let mut target_hits = vec![0u16; n];
    let mut best_gap = vec![u32::MAX; n];
    let mut best_proxy = vec![i64::MIN; n];
    let mut best_link = vec![i64::MIN; n];
    let mut primary_target = vec![usize::MAX; n];
    let mut secondary_target = vec![usize::MAX; n];
    let mut primary_proxy = vec![i64::MIN; n];
    let mut secondary_proxy = vec![i64::MIN; n];

    let mut register_candidate =
        |cand: usize,
         gap: u32,
         proxy: i64,
         link: i64,
         src_hit: bool,
         tgt_hit: bool,
         target_rm: Option<usize>| {
            if cand >= n || state.selected_bit[cand] {
                return;
            }
            let wc = state.ch.weights[cand];
            if wc == 0 || wc > max_swap_w || state.contrib[cand] <= 0 {
                return;
            }

            if !seen[cand] {
                seen[cand] = true;
                cand_ids.push(cand);
            }
            if src_hit && source_hits[cand] < u16::MAX {
                source_hits[cand] += 1;
            }
            if tgt_hit && target_hits[cand] < u16::MAX {
                target_hits[cand] += 1;
            }
            if gap < best_gap[cand] {
                best_gap[cand] = gap;
            }
            if proxy > best_proxy[cand] {
                best_proxy[cand] = proxy;
            }
            if link > best_link[cand] {
                best_link[cand] = link;
            }

            if let Some(rm_t) = target_rm {
                if primary_target[cand] == rm_t {
                    if proxy > primary_proxy[cand] {
                        primary_proxy[cand] = proxy;
                    }
                } else if secondary_target[cand] == rm_t {
                    if proxy > secondary_proxy[cand] {
                        secondary_proxy[cand] = proxy;
                    }
                    if secondary_proxy[cand] > primary_proxy[cand] {
                        core::mem::swap(&mut primary_target[cand], &mut secondary_target[cand]);
                        core::mem::swap(&mut primary_proxy[cand], &mut secondary_proxy[cand]);
                    }
                } else if proxy > primary_proxy[cand] {
                    secondary_target[cand] = primary_target[cand];
                    secondary_proxy[cand] = primary_proxy[cand];
                    primary_target[cand] = rm_t;
                    primary_proxy[cand] = proxy;
                } else if proxy > secondary_proxy[cand] {
                    secondary_target[cand] = rm_t;
                    secondary_proxy[cand] = proxy;
                }
            }
        };

    let mut anchors: Vec<usize> = Vec::with_capacity(anchor_cap);

    for &u in &state.window_core {
        if state.selected_bit[u] {
            anchors.push(u);
            if anchors.len() >= anchor_cap {
                break;
            }
        }
    }

    let extra_locked = state.window_locked.len().min(anchor_cap);
    let start = state.window_locked.len().saturating_sub(extra_locked);
    for &u in state.window_locked[start..].iter() {
        if state.selected_bit[u] {
            let mut dup = false;
            for &old in &anchors {
                if old == u {
                    dup = true;
                    break;
                }
            }
            if !dup {
                anchors.push(u);
                if anchors.len() >= anchor_cap {
                    break;
                }
            }
        }
    }

    if use_per_target_frontiers {
        let mut local_stamp = vec![0u8; n];
        let mut stamp: u8 = 1;

        for &r in &target_rms {
            if !state.selected_bit[r] {
                continue;
            }

            let wr = state.ch.weights[r];
            let tw = slack0.saturating_add(wr);
            let lower = tw.saturating_sub(frontier_band);
            let upper = tw.saturating_add(frontier_band).min(max_swap_w);
            let mut local_front: Vec<(i64, usize, u32, i64, i64, bool)> =
                Vec::with_capacity(per_target_frontier_cap);

            let extra_anchor = {
                let mut best_anchor: Option<usize> = None;
                let mut best_gap_w = u32::MAX;
                for &a in &anchors {
                    if a == r {
                        continue;
                    }
                    let aw = state.ch.weights[a];
                    let gap = if aw > wr { aw - wr } else { wr - aw };
                    if gap < best_gap_w {
                        best_gap_w = gap;
                        best_anchor = Some(a);
                    }
                }
                best_anchor
            };

            let row = unsafe { neigh.get_unchecked(r) };
            for &(cj, vv) in row.iter().take(per_target_neigh) {
                let cand = cj as usize;
                if cand >= n || state.selected_bit[cand] || local_stamp[cand] == stamp {
                    continue;
                }

                let wc = state.ch.weights[cand];
                if wc == 0 || wc > max_swap_w || state.contrib[cand] <= 0 {
                    continue;
                }
                if (state.total_weight as u64) + (wc as u64) > (cap as u64) + (wr as u64) {
                    continue;
                }

                local_stamp[cand] = stamp;

                let gap = if wc > tw { wc - tw } else { tw - wc };
                let proxy = (state.contrib[cand] as i64)
                    - (state.contrib[r] as i64)
                    - (state.ch.interaction_values[cand][r] as i64);
                let c = state.contrib[cand] as i64;
                let tot = state.total_interactions[cand];
                let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot);
                let base = (adj * 1000) / (wc as i64).max(1) + (state.support[cand] as i64) * 64;
                let gap_bonus: i64 = if gap == 0 {
                    2048
                } else if gap <= DIFF_LIM as u32 {
                    640
                } else if gap <= frontier_band {
                    160
                } else {
                    0
                };
                let s = base * 4
                    + proxy.max(-96).min(768) * 96
                    + gap_bonus
                    + (vv as i64).max(0).min(128) * 24
                    - (gap as i64) * 16;

                push_local(
                    &mut local_front,
                    per_target_frontier_cap,
                    s,
                    cand,
                    gap,
                    proxy,
                    vv as i64,
                    true,
                );
            }

            if let Some(anchor) = extra_anchor {
                let row2 = unsafe { neigh.get_unchecked(anchor) };
                for &(cj, vv) in row2.iter().take(per_target_anchor_neigh) {
                    let cand = cj as usize;
                    if cand >= n || state.selected_bit[cand] || local_stamp[cand] == stamp {
                        continue;
                    }

                    let wc = state.ch.weights[cand];
                    if wc == 0 || wc > max_swap_w || state.contrib[cand] <= 0 {
                        continue;
                    }
                    if (state.total_weight as u64) + (wc as u64) > (cap as u64) + (wr as u64) {
                        continue;
                    }

                    local_stamp[cand] = stamp;

                    let gap = if wc > tw { wc - tw } else { tw - wc };
                    let proxy = (state.contrib[cand] as i64)
                        - (state.contrib[r] as i64)
                        - (state.ch.interaction_values[cand][r] as i64);
                    let c = state.contrib[cand] as i64;
                    let tot = state.total_interactions[cand];
                    let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot);
                    let base =
                        (adj * 1000) / (wc as i64).max(1) + (state.support[cand] as i64) * 60;
                    let gap_bonus: i64 = if gap == 0 {
                        1536
                    } else if gap <= DIFF_LIM as u32 {
                        384
                    } else {
                        0
                    };
                    let s = base * 4
                        + proxy.max(-96).min(768) * 96
                        + gap_bonus
                        + (vv as i64).max(0).min(128) * 16
                        - (gap as i64) * 16;

                    push_local(
                        &mut local_front,
                        per_target_frontier_cap,
                        s,
                        cand,
                        gap,
                        proxy,
                        vv as i64,
                        true,
                    );
                }
            }

            let mut core_taken = 0usize;
            for (bw, items) in &state.core_bins {
                if *bw > upper {
                    break;
                }
                if *bw < lower {
                    continue;
                }

                for &cand in items {
                    if core_taken >= per_target_core {
                        break;
                    }
                    if state.selected_bit[cand] || local_stamp[cand] == stamp {
                        continue;
                    }

                    let wc = state.ch.weights[cand];
                    if wc == 0 || wc > max_swap_w || state.contrib[cand] <= 0 {
                        continue;
                    }
                    if (state.total_weight as u64) + (wc as u64) > (cap as u64) + (wr as u64) {
                        continue;
                    }

                    local_stamp[cand] = stamp;
                    core_taken += 1;

                    let gap = if wc > tw { wc - tw } else { tw - wc };
                    let proxy = (state.contrib[cand] as i64)
                        - (state.contrib[r] as i64)
                        - (state.ch.interaction_values[cand][r] as i64);
                    let c = state.contrib[cand] as i64;
                    let tot = state.total_interactions[cand];
                    let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot);
                    let base =
                        (adj * 1000) / (wc as i64).max(1) + (state.support[cand] as i64) * 56;
                    let gap_bonus: i64 = if gap == 0 {
                        1792
                    } else if gap <= DIFF_LIM as u32 {
                        512
                    } else if gap <= frontier_band {
                        128
                    } else {
                        0
                    };
                    let s = base * 4 + proxy.max(-96).min(768) * 96 + gap_bonus - (gap as i64) * 16;

                    push_local(
                        &mut local_front,
                        per_target_frontier_cap,
                        s,
                        cand,
                        gap,
                        proxy,
                        0,
                        false,
                    );
                }

                if core_taken >= per_target_core {
                    break;
                }
            }

            let mut rej_taken = 0usize;
            for &cand in &state.window_rejected[..rej_lim] {
                if rej_taken >= per_target_rej {
                    break;
                }
                if state.selected_bit[cand] || local_stamp[cand] == stamp {
                    continue;
                }

                let wc = state.ch.weights[cand];
                if wc == 0 || wc > max_swap_w || state.contrib[cand] <= 0 {
                    continue;
                }
                if (state.total_weight as u64) + (wc as u64) > (cap as u64) + (wr as u64) {
                    continue;
                }

                let gap = if wc > tw { wc - tw } else { tw - wc };
                if gap > frontier_band.saturating_mul(2) {
                    continue;
                }

                local_stamp[cand] = stamp;
                rej_taken += 1;

                let proxy = (state.contrib[cand] as i64)
                    - (state.contrib[r] as i64)
                    - (state.ch.interaction_values[cand][r] as i64);
                let c = state.contrib[cand] as i64;
                let tot = state.total_interactions[cand];
                let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot);
                let base = (adj * 1000) / (wc as i64).max(1) + (state.support[cand] as i64) * 52;
                let gap_bonus: i64 = if gap == 0 {
                    1536
                } else if gap <= DIFF_LIM as u32 {
                    384
                } else {
                    96
                };
                let s = base * 4 + proxy.max(-96).min(768) * 96 + gap_bonus - (gap as i64) * 16;

                push_local(
                    &mut local_front,
                    per_target_frontier_cap,
                    s,
                    cand,
                    gap,
                    proxy,
                    0,
                    false,
                );
            }

            local_front.sort_unstable_by(|a, b| {
                b.0.cmp(&a.0)
                    .then_with(|| b.3.cmp(&a.3))
                    .then_with(|| a.2.cmp(&b.2))
                    .then_with(|| b.1.cmp(&a.1))
            });

            for &(_s, cand, gap, proxy, link, src_hit) in local_front.iter().take(per_target_take) {
                register_candidate(cand, gap, proxy, link, src_hit, true, Some(r));
            }

            stamp = stamp.wrapping_add(1);
            if stamp == 0 {
                local_stamp.fill(0);
                stamp = 1;
            }
        }
    }

    drop(register_candidate);

    let cand_ids_len = cand_ids.len();
    let thin_target_frontier = cand_ids_len
        < if use_per_target_frontiers {
            target_rms.len().saturating_mul(if n >= 3000 { 3 } else { 4 })
        } else {
            usize::MAX
        };

    if !use_per_target_frontiers || thin_target_frontier {
        let mut register_candidate =
            |cand: usize,
             gap: u32,
             proxy: i64,
             link: i64,
             src_hit: bool,
             tgt_hit: bool,
             target_rm: Option<usize>| {
                if cand >= n || state.selected_bit[cand] {
                    return;
                }
                let wc = state.ch.weights[cand];
                if wc == 0 || wc > max_swap_w || state.contrib[cand] <= 0 {
                    return;
                }

                if !seen[cand] {
                    seen[cand] = true;
                    cand_ids.push(cand);
                }
                if src_hit && source_hits[cand] < u16::MAX {
                    source_hits[cand] += 1;
                }
                if tgt_hit && target_hits[cand] < u16::MAX {
                    target_hits[cand] += 1;
                }
                if gap < best_gap[cand] {
                    best_gap[cand] = gap;
                }
                if proxy > best_proxy[cand] {
                    best_proxy[cand] = proxy;
                }
                if link > best_link[cand] {
                    best_link[cand] = link;
                }

                if let Some(rm_t) = target_rm {
                    if primary_target[cand] == rm_t {
                        if proxy > primary_proxy[cand] {
                            primary_proxy[cand] = proxy;
                        }
                    } else if secondary_target[cand] == rm_t {
                        if proxy > secondary_proxy[cand] {
                            secondary_proxy[cand] = proxy;
                        }
                        if secondary_proxy[cand] > primary_proxy[cand] {
                            core::mem::swap(&mut primary_target[cand], &mut secondary_target[cand]);
                            core::mem::swap(&mut primary_proxy[cand], &mut secondary_proxy[cand]);
                        }
                    } else if proxy > primary_proxy[cand] {
                        secondary_target[cand] = primary_target[cand];
                        secondary_proxy[cand] = primary_proxy[cand];
                        primary_target[cand] = rm_t;
                        primary_proxy[cand] = proxy;
                    } else if proxy > secondary_proxy[cand] {
                        secondary_target[cand] = rm_t;
                        secondary_proxy[cand] = proxy;
                    }
                }
            };

        for &u in &state.window_core {
            if state.selected_bit[u] {
                continue;
            }

            let wc = state.ch.weights[u];
            if wc == 0 || wc > max_swap_w || state.contrib[u] <= 0 {
                continue;
            }

            let mut gap_best = u32::MAX;
            let mut proxy_best = i64::MIN;
            let mut proxy_rm = usize::MAX;

            for &r in &target_rms {
                let wr = state.ch.weights[r];
                let tw = slack0.saturating_add(wr);
                let gap = if wc > tw { wc - tw } else { tw - wc };
                if gap < gap_best {
                    gap_best = gap;
                }
                if (state.total_weight as u64) + (wc as u64) <= (cap as u64) + (wr as u64) {
                    let proxy = (state.contrib[u] as i64)
                        - (state.contrib[r] as i64)
                        - (state.ch.interaction_values[u][r] as i64);
                    if proxy > proxy_best {
                        proxy_best = proxy;
                        proxy_rm = r;
                    }
                }
            }

            register_candidate(
                u,
                gap_best,
                proxy_best,
                0,
                true,
                false,
                if proxy_rm == usize::MAX { None } else { Some(proxy_rm) },
            );
        }

        for &anchor in &anchors {
            let row = unsafe { neigh.get_unchecked(anchor) };
            for &(cj, vv) in row.iter().take(neigh_scan) {
                let cand = cj as usize;
                if cand >= n || state.selected_bit[cand] {
                    continue;
                }

                let wc = state.ch.weights[cand];
                if wc == 0 || wc > max_swap_w || state.contrib[cand] <= 0 {
                    continue;
                }

                let mut gap_best = u32::MAX;
                let mut proxy_best = i64::MIN;
                let mut proxy_rm = usize::MAX;

                for &r in &target_rms {
                    let wr = state.ch.weights[r];
                    let tw = slack0.saturating_add(wr);
                    let gap = if wc > tw { wc - tw } else { tw - wc };
                    if gap < gap_best {
                        gap_best = gap;
                    }
                    if (state.total_weight as u64) + (wc as u64) <= (cap as u64) + (wr as u64) {
                        let proxy = (state.contrib[cand] as i64)
                            - (state.contrib[r] as i64)
                            - (state.ch.interaction_values[cand][r] as i64);
                        if proxy > proxy_best {
                            proxy_best = proxy;
                            proxy_rm = r;
                        }
                    }
                }

                register_candidate(
                    cand,
                    gap_best,
                    proxy_best,
                    vv as i64,
                    true,
                    false,
                    if proxy_rm == usize::MAX { None } else { Some(proxy_rm) },
                );
            }
        }

        for &r in &target_rms {
            let wr = state.ch.weights[r];
            let tw = slack0.saturating_add(wr);
            let lower = tw.saturating_sub(frontier_band);
            let upper = tw.saturating_add(frontier_band).min(max_swap_w);
            let mut core_taken = 0usize;

            for (bw, items) in &state.core_bins {
                if *bw > upper {
                    break;
                }
                if *bw < lower {
                    continue;
                }

                let gap = if *bw > tw { *bw - tw } else { tw - *bw };

                for &cand in items {
                    if core_taken >= per_target_core {
                        break;
                    }
                    if state.selected_bit[cand] {
                        continue;
                    }

                    let wc = state.ch.weights[cand];
                    if wc == 0 || wc > max_swap_w || state.contrib[cand] <= 0 {
                        continue;
                    }

                    let proxy = if (state.total_weight as u64) + (wc as u64)
                        <= (cap as u64) + (wr as u64)
                    {
                        (state.contrib[cand] as i64)
                            - (state.contrib[r] as i64)
                            - (state.ch.interaction_values[cand][r] as i64)
                    } else {
                        i64::MIN
                    };

                    register_candidate(cand, gap, proxy, 0, false, true, Some(r));
                    core_taken += 1;
                }
            }

            let mut rej_taken = 0usize;
            for &cand in &state.window_rejected[..rej_lim] {
                if rej_taken >= per_target_rej {
                    break;
                }
                if state.selected_bit[cand] {
                    continue;
                }

                let wc = state.ch.weights[cand];
                if wc == 0 || wc > max_swap_w || state.contrib[cand] <= 0 {
                    continue;
                }

                let gap = if wc > tw { wc - tw } else { tw - wc };
                if gap > frontier_band.saturating_mul(2) {
                    continue;
                }

                let proxy = if (state.total_weight as u64) + (wc as u64)
                    <= (cap as u64) + (wr as u64)
                {
                    (state.contrib[cand] as i64)
                        - (state.contrib[r] as i64)
                        - (state.ch.interaction_values[cand][r] as i64)
                } else {
                    i64::MIN
                };

                register_candidate(cand, gap, proxy, 0, false, true, Some(r));
                rej_taken += 1;
            }
        }

        drop(register_candidate);
    }

    if cand_ids.is_empty() {
        return false;
    }

    let use_related_eval = use_per_target_frontiers && !thin_target_frontier;

    let mut cand_list: Vec<(i64, usize)> = Vec::with_capacity(TOP_CAND);
    for cand in cand_ids {
        let wc = state.ch.weights[cand];
        let c = state.contrib[cand] as i64;
        let tot = state.total_interactions[cand];
        let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot);
        let base = (adj * 1000) / (wc as i64).max(1) + (state.support[cand] as i64) * 60;

        let gap = best_gap[cand];
        let gap_bonus: i64 = if gap == 0 {
            1536
        } else if gap <= DIFF_LIM as u32 {
            448
        } else if gap <= frontier_band {
            128
        } else {
            0
        };

        let proxy = best_proxy[cand].max(-256).min(512);
        let link = if best_link[cand] == i64::MIN {
            0
        } else {
            best_link[cand].max(0).min(128)
        };
        let related_bonus = ((primary_target[cand] != usize::MAX) as i64) * 192
            + ((secondary_target[cand] != usize::MAX) as i64) * 96;

        let s = base * 4
            + gap_bonus
            + (target_hits[cand].min(4) as i64) * 256
            + (source_hits[cand].min(4) as i64) * 128
            + proxy * 64
            + link * 24
            + related_bonus;

        push_top(&mut cand_list, s, cand);
    }

    if cand_list.is_empty() {
        return false;
    }

    let mut best: Option<(usize, usize, i128, i64, u32, i64)> = None;

    for &(_s0, cand) in &cand_list {
        let wc = state.ch.weights[cand];
        if wc == 0 {
            continue;
        }

        let proxy_c = best_proxy[cand].max(-256).min(512);
        let src_hits = source_hits[cand].min(4) as i64;
        let tgt_hits = target_hits[cand].min(4) as i64;

        let mut eval_rms: Vec<usize> = Vec::with_capacity(8);
        if use_related_eval {
            let p = primary_target[cand];
            if p != usize::MAX {
                push_unique_rm(&mut eval_rms, p);
                if let Some(pos) = target_rms.iter().position(|&x| x == p) {
                    if pos > 0 {
                        push_unique_rm(&mut eval_rms, target_rms[pos - 1]);
                    }
                    if pos + 1 < target_rms.len() {
                        push_unique_rm(&mut eval_rms, target_rms[pos + 1]);
                    }
                }
            }

            let s = secondary_target[cand];
            if s != usize::MAX {
                push_unique_rm(&mut eval_rms, s);
                if let Some(pos) = target_rms.iter().position(|&x| x == s) {
                    if pos > 0 {
                        push_unique_rm(&mut eval_rms, target_rms[pos - 1]);
                    }
                    if pos + 1 < target_rms.len() {
                        push_unique_rm(&mut eval_rms, target_rms[pos + 1]);
                    }
                }
            }

            if eval_rms.len() < 2 {
                for &r in rm.iter().take(4) {
                    push_unique_rm(&mut eval_rms, r);
                }
            }
        } else {
            eval_rms.extend_from_slice(&rm);
        }

        for &r in &eval_rms {
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

            let target_w = slack0.saturating_add(wr);
            let target_gap = if wc > target_w { wc - target_w } else { target_w - wc };

            let weight_bonus = if wc == wr {
                8192
            } else if wc < wr {
                ((wr - wc) as i64).min(256) * 24
            } else {
                -(((wc - wr) as i64).min(256) * 6)
            };

            let tie = weight_bonus
                - (target_gap as i64) * 32
                + proxy_c * 16
                + tgt_hits * 128
                + src_hits * 64;

            let score = (delta as i128) * 1_000_000i128 + (tie as i128);

            let take = match best {
                None => true,
                Some((_bc, _br, bs, bd, bg, bp)) => {
                    score > bs
                        || (score == bs
                            && (delta > bd
                                || (delta == bd
                                    && (target_gap < bg
                                        || (target_gap == bg && proxy_c > bp)))))
                }
            };

            if take {
                best = Some((cand, r, score, delta, target_gap, proxy_c));
            }
        }
    }

    if let Some((cand, r, _, _, _, _)) = best {
        state.replace_item(r, cand);
        true
    } else {
        false
    }
}

fn apply_best_remove_refill_windowed(state: &mut State, used: &[usize]) -> bool {
    #[inline]
    fn estimate_refill_signal(state: &State, slack: u32, sample_lim: usize) -> (i64, i64) {
        let mut best = i64::MIN;
        let mut hits = 0i64;

        for (bw, items) in &state.core_bins {
            if *bw > slack {
                break;
            }
            for &cand in items {
                if state.selected_bit[cand] {
                    continue;
                }
                let delta = state.contrib[cand] as i64;
                if delta <= 0 {
                    continue;
                }

                let fit = (slack - state.ch.weights[cand]) as i64;
                let s = delta * 64 + (state.support[cand] as i64) * 32 - fit * 2;
                if s > best {
                    best = s;
                }

                hits += 1;
                if hits as usize >= sample_lim {
                    return (best.max(0), hits);
                }
            }
        }

        let extra = state.window_rejected.len().min(sample_lim);
        for &cand in &state.window_rejected[..extra] {
            if state.selected_bit[cand] {
                continue;
            }

            let w = state.ch.weights[cand];
            if w == 0 || w > slack {
                continue;
            }

            let delta = state.contrib[cand] as i64;
            if delta <= 0 {
                continue;
            }

            let fit = (slack - w) as i64;
            let s = delta * 64 + (state.support[cand] as i64) * 32 - fit * 2;
            if s > best {
                best = s;
            }

            hits += 1;
            if hits as usize >= sample_lim {
                break;
            }
        }

        (best.max(0), hits)
    }

    if used.is_empty() {
        return false;
    }

    let n = state.ch.num_items;
    let slack0 = state.slack();
    let broad_slack_lim = if n >= 3000 {
        (DIFF_LIM as u32).saturating_mul(2)
    } else {
        (DIFF_LIM as u32).saturating_mul(3)
    };
    let relaxed_slack_lim = broad_slack_lim.saturating_mul(2);
    let dense_used = used.len() >= if n >= 3000 {
        10
    } else if n >= 1500 {
        8
    } else {
        6
    };

    if !(slack0 <= broad_slack_lim || (dense_used && slack0 <= relaxed_slack_lim)) {
        return false;
    }

    let old_val = state.total_value as i64;
    let trial_cap: usize = if n >= 3000 {
        2
    } else if n >= 1500 {
        3
    } else {
        4
    };
    let sample_lim: usize = if n >= 3000 {
        12
    } else if n >= 1500 {
        16
    } else {
        22
    };

    let mut rm: Vec<(i64, i64, usize)> = Vec::with_capacity(used.len());
    for &i in used {
        if !state.selected_bit[i] {
            continue;
        }

        let c = state.contrib[i] as i64;
        let w = (state.ch.weights[i] as i64).max(1);
        let sup = state.support[i] as i64;
        let anti = 2 * c - state.total_interactions[i];
        let slack_after = slack0.saturating_add(state.ch.weights[i]);
        let (refill_best, refill_hits) = estimate_refill_signal(state, slack_after, sample_lim);

        let band_bonus = if slack_after <= broad_slack_lim {
            256
        } else if slack_after <= relaxed_slack_lim {
            96
        } else {
            0
        };

        let weakness = (c * 1000) / w + anti / 16 + sup * 64 - w / 4;
        let score = weakness * 8 - refill_best - refill_hits * 64 - band_bonus;
        rm.push((score, -w, i));
    }

    if rm.is_empty() {
        return false;
    }

    rm.sort_unstable_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    if rm.len() > trial_cap {
        rm.truncate(trial_cap);
    }

    let mut best_plan: Option<(usize, Vec<usize>, i64, usize, usize)> = None;

    for &(_, _, r) in &rm {
        if !state.selected_bit[r] {
            continue;
        }

        state.remove_item(r);
        let mut added: Vec<usize> = Vec::with_capacity(4);
        let mut refill_steps = 0usize;

        let slack1 = state.slack();
        if slack1 > 0 {
            if let Some((a, b, gain1)) = best_refill_move_windowed(state, slack1) {
                if gain1 > 0 {
                    let before_value = state.total_value as i64;
                    let before_weight = state.total_weight;
                    let mut moved = false;

                    if !state.selected_bit[a]
                        && state.total_weight + state.ch.weights[a] <= state.ch.max_weight
                    {
                        state.add_item(a);
                        added.push(a);
                        moved = true;
                    }

                    if let Some(b2) = b {
                        if !state.selected_bit[b2]
                            && state.total_weight + state.ch.weights[b2] <= state.ch.max_weight
                        {
                            state.add_item(b2);
                            added.push(b2);
                            moved = true;
                        }
                    }

                    if moved
                        && state.total_weight != before_weight
                        && (state.total_value as i64) > before_value
                    {
                        refill_steps = 1;

                        let slack2 = state.slack();
                        if slack2 > 0 {
                            if let Some((c1, c2, gain2)) = best_refill_move_windowed(state, slack2) {
                                if gain2 > 0 {
                                    let before_value2 = state.total_value as i64;
                                    let before_weight2 = state.total_weight;
                                    let mut moved2 = false;

                                    if !state.selected_bit[c1]
                                        && state.total_weight + state.ch.weights[c1] <= state.ch.max_weight
                                    {
                                        state.add_item(c1);
                                        added.push(c1);
                                        moved2 = true;
                                    }

                                    if let Some(c2i) = c2 {
                                        if !state.selected_bit[c2i]
                                            && state.total_weight + state.ch.weights[c2i] <= state.ch.max_weight
                                        {
                                            state.add_item(c2i);
                                            added.push(c2i);
                                            moved2 = true;
                                        }
                                    }

                                    if moved2
                                        && state.total_weight != before_weight2
                                        && (state.total_value as i64) > before_value2
                                    {
                                        refill_steps = 2;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let gain = (state.total_value as i64) - old_val;
        let add_count = added.len();
        if gain > 0
            && best_plan.as_ref().map_or(true, |(_, _, bg, bs, ba)| {
                gain > *bg
                    || (gain == *bg
                        && (refill_steps > *bs || (refill_steps == *bs && add_count > *ba)))
            })
        {
            best_plan = Some((r, added.clone(), gain, refill_steps, add_count));
        }

        for &cand in added.iter().rev() {
            if state.selected_bit[cand] {
                state.remove_item(cand);
            }
        }
        if !state.selected_bit[r] {
            state.add_item(r);
        }
    }

    if let Some((r, added, _, _, _)) = best_plan {
        state.remove_item(r);
        for cand in added {
            if !state.selected_bit[cand]
                && state.total_weight + state.ch.weights[cand] <= state.ch.max_weight
            {
                state.add_item(cand);
            }
        }
        true
    } else {
        false
    }
}

fn apply_kick_refill_windowed(state: &mut State, used: &[usize]) -> bool {
    if used.is_empty() {
        return false;
    }

    let old_val = state.total_value;
    let trial_cap: usize = if state.ch.num_items >= 2500 { 2 } else { 3 };
    let mut kicks: Vec<(i64, i64, i64, usize)> = Vec::with_capacity(trial_cap);

    for &i in used {
        if !state.selected_bit[i] {
            continue;
        }

        let c = state.contrib[i] as i64;
        let w = (state.ch.weights[i] as i64).max(1);
        let sup = state.support[i] as i64;
        let anti = 2 * c - state.total_interactions[i];
        let adj = (c * 1000) / w + anti / 16 + sup * 72 - w / 4;

        if kicks.len() < trial_cap {
            kicks.push((adj, c, -w, i));
            continue;
        }

        let mut worst_pos = 0usize;
        let mut worst_key = kicks[0];
        for t in 1..kicks.len() {
            if kicks[t] > worst_key {
                worst_key = kicks[t];
                worst_pos = t;
            }
        }

        let key = (adj, c, -w, i);
        if key < worst_key {
            kicks[worst_pos] = key;
        }
    }

    if kicks.is_empty() {
        return false;
    }

    kicks.sort_unstable();

    let mut best_plan: Option<(usize, Vec<usize>, i64)> = None;

    for &(_, _, _, r) in &kicks {
        state.remove_item(r);

        let mut added: Vec<usize> = Vec::new();

        loop {
            let slack = state.slack();
            if slack == 0 {
                break;
            }

            let mv = match best_refill_move_windowed(state, slack) {
                Some(mv) => mv,
                None => break,
            };

            let a = mv.0;
            state.add_item(a);
            added.push(a);

            if let Some(b) = mv.1 {
                if !state.selected_bit[b]
                    && state.total_weight + state.ch.weights[b] <= state.ch.max_weight
                {
                    state.add_item(b);
                    added.push(b);
                }
            }
        }

        let gain = (state.total_value as i64) - (old_val as i64);
        if gain > 0 && best_plan.as_ref().map_or(true, |(_, _, bg)| gain > *bg) {
            best_plan = Some((r, added.clone(), gain));
        }

        for &cand in added.iter().rev() {
            if state.selected_bit[cand] {
                state.remove_item(cand);
            }
        }
        if !state.selected_bit[r] {
            state.add_item(r);
        }
    }

    if let Some((r, added, _)) = best_plan {
        state.remove_item(r);
        for cand in added {
            if !state.selected_bit[cand] && state.total_weight + state.ch.weights[cand] <= state.ch.max_weight {
                state.add_item(cand);
            }
        }
        true
    } else {
        false
    }
}

pub fn local_search_vnd(state: &mut State, max_frontier_swaps_override: Option<usize>) {
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
    let mut post_struct_micro_used = false;

    let mut frontier_swap_tries: usize = 0;
    let frontier_override = max_frontier_swaps_override;
    let max_frontier_swaps: usize = frontier_override.unwrap_or(if n >= 2500 {
        0
    } else if n >= 1500 {
        1
    } else {
        2
    });

    let mut dirty_window = false;
    let mut n_rebuilds = 0usize;
    let max_rebuilds: usize = if n >= 2500 { 1 } else { 2 };

    let mut add_neigh_fail: u8 = 0;
    let mut add_neigh_cooldown: u8 = 0;
    let mut add_neigh_hot: u8 = 0;

    let mut swap_neigh_fail: u8 = 0;
    let mut swap_neigh_cooldown: u8 = 0;

    let mut frontier_fail: u8 = 0;
    let mut frontier_cooldown: u8 = 0;
    let mut frontier_hot: u8 = 0;

    let mut remove_refill_fail: u8 = 0;
    let mut remove_refill_cooldown: u8 = 0;
    let mut remove_refill_hot: u8 = 0;

    let mut replace12_fail: u8 = 0;
    let mut replace12_cooldown: u8 = 0;

    let mut replace21_fail: u8 = 0;
    let mut replace21_cooldown: u8 = 0;
    let mut replace21_hot: u8 = 0;

    let mut kick_fail: u8 = 0;
    let mut kick_cooldown: u8 = 0;

    let short_skip: u8 = if n >= 3000 { 2 } else { 1 };
    let long_skip: u8 = if n >= 3000 { 3 } else { 2 };
    let retry_boost: u8 = if n >= 3000 { 2 } else { 3 };

    let decay = |x: &mut u8| {
        if *x > 0 {
            *x -= 1;
        }
    };
    let note_success = |fail: &mut u8, cooldown: &mut u8, hot: &mut u8, boost: u8| {
        *fail = 0;
        *cooldown = 0;
        if *hot < boost {
            *hot = boost;
        }
    };
    let note_success_plain = |fail: &mut u8, cooldown: &mut u8| {
        *fail = 0;
        *cooldown = 0;
    };
    let note_failure = |fail: &mut u8,
                        cooldown: &mut u8,
                        hot: &mut u8,
                        fail_lim: u8,
                        skip: u8| {
        *hot = 0;
        if *cooldown > 0 {
            return;
        }
        *fail = (*fail).saturating_add(1);
        if *fail >= fail_lim {
            *fail = 0;
            *cooldown = skip;
        }
    };
    let note_failure_plain = |fail: &mut u8, cooldown: &mut u8, fail_lim: u8, skip: u8| {
        if *cooldown > 0 {
            return;
        }
        *fail = (*fail).saturating_add(1);
        if *fail >= fail_lim {
            *fail = 0;
            *cooldown = skip;
        }
    };

    loop {
        iterations += 1;
        if iterations > max_iterations {
            break;
        }

        decay(&mut add_neigh_cooldown);
        decay(&mut add_neigh_hot);
        decay(&mut swap_neigh_cooldown);
        decay(&mut frontier_cooldown);
        decay(&mut frontier_hot);
        decay(&mut remove_refill_cooldown);
        decay(&mut remove_refill_hot);
        decay(&mut replace12_cooldown);
        decay(&mut replace21_cooldown);
        decay(&mut replace21_hot);
        decay(&mut kick_cooldown);

        if apply_best_add_windowed(state) {
            dirty_window = true;
            continue;
        }

        let add_neigh_due =
            add_neigh_cooldown == 0 && (add_neigh_hot > 0 || (iterations & 3) == 0);
        if add_neigh_due {
            if apply_best_add_neigh_global(state) {
                note_success(
                    &mut add_neigh_fail,
                    &mut add_neigh_cooldown,
                    &mut add_neigh_hot,
                    retry_boost,
                );
                dirty_window = true;
                continue;
            } else {
                note_failure(
                    &mut add_neigh_fail,
                    &mut add_neigh_cooldown,
                    &mut add_neigh_hot,
                    2,
                    short_skip,
                );
            }
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

        {
            const WORST_CAP: usize = 32;
            let mut weakest: Vec<(i64, i64, i64, usize)> = Vec::with_capacity(WORST_CAP);

            for i in 0..n {
                if !state.selected_bit[i] {
                    continue;
                }

                let c = state.contrib[i] as i64;
                let w = (state.ch.weights[i] as i64).max(1);
                let anti = 2 * c - state.total_interactions[i];
                let sup = state.support[i] as i64;
                let weakness = (c * 1000) / w + anti / 12 + sup * 64 - w / 6;
                let key = (weakness, anti, -w, i);

                if weakest.len() < WORST_CAP {
                    weakest.push(key);
                    continue;
                }

                let mut worst_pos = 0usize;
                let mut worst_key = weakest[0];
                for t in 1..weakest.len() {
                    if weakest[t] > worst_key {
                        worst_key = weakest[t];
                        worst_pos = t;
                    }
                }

                if key < worst_key {
                    weakest[worst_pos] = key;
                }
            }

            for &(_, _, _, idx) in &weakest {
                used.push(idx);
            }

            used.sort_unstable();
            used.dedup();
        }

        let tight_slack_lim = if n >= 2500 {
            (DIFF_LIM as u32).saturating_mul(2)
        } else {
            (DIFF_LIM as u32).saturating_mul(3)
        };
        let dense_used = used.len() >= if n >= 3000 {
            10
        } else if n >= 1500 {
            8
        } else {
            6
        };
        let decomp_first = state.slack() <= tight_slack_lim && dense_used;
        let structural_first = decomp_first || remove_refill_hot > 0 || replace21_hot > 0;

        if apply_best_swap_diff_reduce_windowed_cached(state, &used) {
            dirty_window = true;
            continue;
        }
        if apply_best_swap_diff_increase_windowed_cached(state, &used) {
            dirty_window = true;
            continue;
        }

        let mut tried_remove_refill = false;
        let mut tried_replace21 = false;
        let mut tried_frontier = false;

        if structural_first {
            let prefer_replace21 =
                (replace21_hot > remove_refill_hot || remove_refill_cooldown > 0)
                    && replace21_cooldown == 0;

            if prefer_replace21 {
                if replace21_cooldown == 0 {
                    tried_replace21 = true;
                    if apply_best_replace21_windowed(state, &used) {
                        note_success(
                            &mut replace21_fail,
                            &mut replace21_cooldown,
                            &mut replace21_hot,
                            retry_boost,
                        );
                        dirty_window = true;
                        if n_rebuilds < max_rebuilds {
                            n_rebuilds += 1;
                            dirty_window = false;
                            rebuild_windows(state);
                        }

                        let allow_post_struct_micro = !post_struct_micro_used
                            && (n <= 1600 || iterations * 2 >= max_iterations);
                        if allow_post_struct_micro {
                            post_struct_micro_used = true;
                            if dirty_window {
                                rebuild_windows(state);
                                dirty_window = false;
                            }
                            let old = state.total_value;
                            micro_qkp_refinement(state);
                            if state.total_value > old {
                                rebuild_windows(state);
                            }
                        }
                        continue;
                    } else {
                        note_failure(
                            &mut replace21_fail,
                            &mut replace21_cooldown,
                            &mut replace21_hot,
                            2,
                            short_skip,
                        );
                    }
                }

                if remove_refill_cooldown == 0 {
                    tried_remove_refill = true;
                    if apply_best_remove_refill_windowed(state, &used) {
                        note_success(
                            &mut remove_refill_fail,
                            &mut remove_refill_cooldown,
                            &mut remove_refill_hot,
                            retry_boost,
                        );
                        dirty_window = true;
                        if n_rebuilds < max_rebuilds {
                            n_rebuilds += 1;
                            dirty_window = false;
                            rebuild_windows(state);
                        }

                        let allow_post_struct_micro = !post_struct_micro_used
                            && (n <= 1600 || iterations * 2 >= max_iterations);
                        if allow_post_struct_micro {
                            post_struct_micro_used = true;
                            if dirty_window {
                                rebuild_windows(state);
                                dirty_window = false;
                            }
                            let old = state.total_value;
                            micro_qkp_refinement(state);
                            if state.total_value > old {
                                rebuild_windows(state);
                            }
                        }
                        continue;
                    } else {
                        note_failure(
                            &mut remove_refill_fail,
                            &mut remove_refill_cooldown,
                            &mut remove_refill_hot,
                            2,
                            short_skip,
                        );
                    }
                }
            } else {
                if remove_refill_cooldown == 0 {
                    tried_remove_refill = true;
                    if apply_best_remove_refill_windowed(state, &used) {
                        note_success(
                            &mut remove_refill_fail,
                            &mut remove_refill_cooldown,
                            &mut remove_refill_hot,
                            retry_boost,
                        );
                        dirty_window = true;
                        if n_rebuilds < max_rebuilds {
                            n_rebuilds += 1;
                            dirty_window = false;
                            rebuild_windows(state);
                        }

                        let allow_post_struct_micro = !post_struct_micro_used
                            && (n <= 1600 || iterations * 2 >= max_iterations);
                        if allow_post_struct_micro {
                            post_struct_micro_used = true;
                            if dirty_window {
                                rebuild_windows(state);
                                dirty_window = false;
                            }
                            let old = state.total_value;
                            micro_qkp_refinement(state);
                            if state.total_value > old {
                                rebuild_windows(state);
                            }
                        }
                        continue;
                    } else {
                        note_failure(
                            &mut remove_refill_fail,
                            &mut remove_refill_cooldown,
                            &mut remove_refill_hot,
                            2,
                            short_skip,
                        );
                    }
                }

                if replace21_cooldown == 0 {
                    tried_replace21 = true;
                    if apply_best_replace21_windowed(state, &used) {
                        note_success(
                            &mut replace21_fail,
                            &mut replace21_cooldown,
                            &mut replace21_hot,
                            retry_boost,
                        );
                        dirty_window = true;
                        if n_rebuilds < max_rebuilds {
                            n_rebuilds += 1;
                            dirty_window = false;
                            rebuild_windows(state);
                        }

                        let allow_post_struct_micro = !post_struct_micro_used
                            && (n <= 1600 || iterations * 2 >= max_iterations);
                        if allow_post_struct_micro {
                            post_struct_micro_used = true;
                            if dirty_window {
                                rebuild_windows(state);
                                dirty_window = false;
                            }
                            let old = state.total_value;
                            micro_qkp_refinement(state);
                            if state.total_value > old {
                                rebuild_windows(state);
                            }
                        }
                        continue;
                    } else {
                        note_failure(
                            &mut replace21_fail,
                            &mut replace21_cooldown,
                            &mut replace21_hot,
                            2,
                            short_skip,
                        );
                    }
                }
            }
        }

        let frontier_gate = frontier_swap_tries < max_frontier_swaps
            || (frontier_override.is_none()
                && frontier_swap_tries == 0
                && max_frontier_swaps == 0
                && state.slack() <= (DIFF_LIM as u32).saturating_mul(2)
                && used.len() >= if n >= 3000 { 10 } else { 8 });
        let frontier_due =
            frontier_cooldown == 0 && (frontier_hot > 0 || frontier_gate);

        if !structural_first && frontier_hot > 0 && frontier_due {
            tried_frontier = true;
            frontier_swap_tries += 1;
            if apply_best_swap_frontier_global(state, &used) {
                note_success(
                    &mut frontier_fail,
                    &mut frontier_cooldown,
                    &mut frontier_hot,
                    retry_boost,
                );
                dirty_window = true;
                continue;
            } else {
                note_failure(
                    &mut frontier_fail,
                    &mut frontier_cooldown,
                    &mut frontier_hot,
                    2,
                    long_skip,
                );
            }
        }

        if swap_neigh_cooldown == 0 {
            if apply_best_swap_neigh_any(state, &used) {
                note_success_plain(&mut swap_neigh_fail, &mut swap_neigh_cooldown);
                dirty_window = true;
                continue;
            } else {
                note_failure_plain(
                    &mut swap_neigh_fail,
                    &mut swap_neigh_cooldown,
                    2,
                    short_skip,
                );
            }
        }

        if !tried_frontier && frontier_due {
            frontier_swap_tries += 1;
            if apply_best_swap_frontier_global(state, &used) {
                note_success(
                    &mut frontier_fail,
                    &mut frontier_cooldown,
                    &mut frontier_hot,
                    retry_boost,
                );
                dirty_window = true;
                continue;
            } else {
                note_failure(
                    &mut frontier_fail,
                    &mut frontier_cooldown,
                    &mut frontier_hot,
                    2,
                    long_skip,
                );
            }
        }

        if !tried_remove_refill && remove_refill_cooldown == 0 {
            if apply_best_remove_refill_windowed(state, &used) {
                note_success(
                    &mut remove_refill_fail,
                    &mut remove_refill_cooldown,
                    &mut remove_refill_hot,
                    retry_boost,
                );
                dirty_window = true;
                if n_rebuilds < max_rebuilds {
                    n_rebuilds += 1;
                    dirty_window = false;
                    rebuild_windows(state);
                }

                let allow_post_struct_micro =
                    !post_struct_micro_used && (n <= 1600 || iterations * 2 >= max_iterations);
                if allow_post_struct_micro {
                    post_struct_micro_used = true;
                    if dirty_window {
                        rebuild_windows(state);
                        dirty_window = false;
                    }
                    let old = state.total_value;
                    micro_qkp_refinement(state);
                    if state.total_value > old {
                        rebuild_windows(state);
                    }
                }
                continue;
            } else {
                note_failure(
                    &mut remove_refill_fail,
                    &mut remove_refill_cooldown,
                    &mut remove_refill_hot,
                    2,
                    short_skip,
                );
            }
        }

        if replace12_cooldown == 0 {
            if apply_best_replace12_windowed(state, &used) {
                note_success_plain(&mut replace12_fail, &mut replace12_cooldown);
                dirty_window = true;
                if n_rebuilds < max_rebuilds {
                    n_rebuilds += 1;
                    dirty_window = false;
                    rebuild_windows(state);
                }

                let allow_post_struct_micro =
                    !post_struct_micro_used && (n <= 1600 || iterations * 2 >= max_iterations);
                if allow_post_struct_micro {
                    post_struct_micro_used = true;
                    if dirty_window {
                        rebuild_windows(state);
                        dirty_window = false;
                    }
                    let old = state.total_value;
                    micro_qkp_refinement(state);
                    if state.total_value > old {
                        rebuild_windows(state);
                    }
                }
                continue;
            } else {
                note_failure_plain(
                    &mut replace12_fail,
                    &mut replace12_cooldown,
                    2,
                    short_skip,
                );
            }
        }

        if !tried_replace21 && replace21_cooldown == 0 {
            if apply_best_replace21_windowed(state, &used) {
                note_success(
                    &mut replace21_fail,
                    &mut replace21_cooldown,
                    &mut replace21_hot,
                    retry_boost,
                );
                dirty_window = true;
                if n_rebuilds < max_rebuilds {
                    n_rebuilds += 1;
                    dirty_window = false;
                    rebuild_windows(state);
                }

                let allow_post_struct_micro =
                    !post_struct_micro_used && (n <= 1600 || iterations * 2 >= max_iterations);
                if allow_post_struct_micro {
                    post_struct_micro_used = true;
                    if dirty_window {
                        rebuild_windows(state);
                        dirty_window = false;
                    }
                    let old = state.total_value;
                    micro_qkp_refinement(state);
                    if state.total_value > old {
                        rebuild_windows(state);
                    }
                }
                continue;
            } else {
                note_failure(
                    &mut replace21_fail,
                    &mut replace21_cooldown,
                    &mut replace21_hot,
                    2,
                    short_skip,
                );
            }
        }

        if kick_cooldown == 0 {
            if apply_kick_refill_windowed(state, &used) {
                note_success_plain(&mut kick_fail, &mut kick_cooldown);
                dirty_window = true;
                if n_rebuilds < max_rebuilds {
                    n_rebuilds += 1;
                    dirty_window = false;
                    rebuild_windows(state);
                }
                continue;
            } else {
                note_failure_plain(&mut kick_fail, &mut kick_cooldown, 1, long_skip);
            }
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