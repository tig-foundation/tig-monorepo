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

        for &anchor in &anchors {
            let row = unsafe { neigh.get_unchecked(anchor) };
            let mut scanned = 0usize;
            let mut taken = 0usize;

            for &(cj, _vv) in row.iter() {
                scanned += 1;
                if scanned > per_anchor_scan || taken >= per_anchor_take {
                    break;
                }

                let cand = cj as usize;
                if cand >= n || state.selected_bit[cand] || seen[cand] {
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
                taken += 1;

                let w = (w_u as i64).max(1);
                let c = delta as i64;
                let tot = state.total_interactions[cand];
                let fit = (slack - w_u) as i64;
                let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot);
                let s = (adj * 1000) / w + (state.support[cand] as i64) * SUP_ADD_BONUS - fit;

                push_top(&mut frontier, frontier_cap, s, cand, delta);
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

            let w = (w_u as i64).max(1);
            let c = delta as i64;
            let tot = state.total_interactions[cand];
            let fit = (slack - w_u) as i64;
            let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot);
            let s = (adj * 1000) / w + (state.support[cand] as i64) * SUP_ADD_BONUS - fit;

            push_top(&mut frontier, frontier_cap, s, cand, delta);
        }
    }

    if frontier.len() >= 4 {
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

    let n = state.ch.num_items;
    let top_cand = if n >= 3000 { 18 } else if n >= 1500 { 24 } else { 30 };
    let raw_lim = top_cand * 3;
    let rej_lim = if n >= 3000 {
        state.window_rejected.len().min(96)
    } else {
        state.window_rejected.len().min(160)
    };

    let mut raw: Vec<(i64, usize)> = Vec::with_capacity(raw_lim);

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
            let fit = if slack >= w { (slack - w) as i64 } else { 0 };
            let s = delta * 128 + anti * 8 + (state.support[cand] as i64) * 72 - fit;
            push_top(&mut raw, raw_lim, s, cand);
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
        let fit = if slack >= w { (slack - w) as i64 } else { 0 };
        let s = delta * 128 + anti * 8 + (state.support[cand] as i64) * 72 - fit;
        push_top(&mut raw, raw_lim, s, cand);
    }

    if raw.is_empty() {
        return None;
    }

    raw.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));

    let mut pool: Vec<usize> = Vec::with_capacity(top_cand);
    let mut band_counts = [0usize; 6];
    let denom = (slack as u64).max(1) + 1;

    for &(_s, cand) in &raw {
        if pool.len() >= top_cand {
            break;
        }
        let mut seen = false;
        for &x in &pool {
            if x == cand {
                seen = true;
                break;
            }
        }
        if seen {
            continue;
        }

        let band = ((((state.ch.weights[cand] as u64) * 6) / denom).min(5u64)) as usize;
        if band_counts[band] >= 4 {
            continue;
        }
        band_counts[band] += 1;
        pool.push(cand);
    }

    if pool.len() < 2 {
        if let Some(&cand) = pool.first() {
            if state.contrib[cand] > 0 {
                return Some((cand, None, state.contrib[cand] as i64));
            }
        }
        return None;
    }

    let mut best: Option<(usize, Option<usize>, i64, i64, u32)> = None;

    for &a in &pool {
        let delta = state.contrib[a] as i64;
        if delta > 0 {
            let sup = state.support[a] as i64;
            let used_w = state.ch.weights[a];
            if best.map_or(true, |(_, _, bd, bs, bw)| {
                delta > bd || (delta == bd && (sup > bs || (sup == bs && used_w > bw)))
            }) {
                best = Some((a, None, delta, sup, used_w));
            }
        }
    }

    for x in 0..pool.len() {
        let a = pool[x];
        let wa = state.ch.weights[a];
        for y in (x + 1)..pool.len() {
            let b = pool[y];
            let wb = state.ch.weights[b];
            if wa.saturating_add(wb) > slack {
                continue;
            }

            let delta = (state.contrib[a] as i64)
                + (state.contrib[b] as i64)
                + (state.ch.interaction_values[a][b] as i64);
            if delta <= 0 {
                continue;
            }

            let sup = (state.support[a] as i64) + (state.support[b] as i64);
            let used_w = wa.saturating_add(wb);

            if best.map_or(true, |(_, _, bd, bs, bw)| {
                delta > bd || (delta == bd && (sup > bs || (sup == bs && used_w > bw)))
            }) {
                best = Some((a, Some(b), delta, sup, used_w));
            }
        }
    }

    best.map(|(a, b, delta, _, _)| (a, b, delta))
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

        if score > worst_score || (score == worst_score && (net < worst_net || (net == worst_net && wsum > worst_w))) {
            list[worst_pos] = (score, a, b, net, wsum);
        }
    }

    let cap = state.ch.max_weight;
    if used.len() < 2 {
        return false;
    }

    let n = state.ch.num_items;
    let pair_cap: usize = if n >= 3000 { 24 } else if n >= 1500 { 32 } else { 40 };
    let add_cap: usize = if n >= 3000 { 22 } else if n >= 1500 { 26 } else { 30 };
    let extra_rej: usize = if n >= 3000 { 48 } else { 64 };

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
            let base = (c * 1000) / (wc as i64).max(1) + anti / 8 + (state.support[cand] as i64) * 80;

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
            let base = (c * 1000) / (wc as i64).max(1) + anti / 8 + (state.support[cand] as i64) * 80;

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

                if best_pair_first.map_or(true, |(_, _, _, bd, bt)| delta > bd || (delta == bd && tie > bt)) {
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

    let mut best: Option<(usize, usize, i128, i64, i64, i64, i64, i64)> = None;

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

        for &rm in &rm_list {
            let wrm = state.ch.weights[rm];
            if (state.total_weight as u64) + (wc as u64) > (cap as u64) + (wrm as u64) {
                continue;
            }

            let inter = state.ch.interaction_values[cand][rm] as i64;
            let delta = cand_contrib - (state.contrib[rm] as i64) - inter;
            if delta <= 0 {
                continue;
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
            let score = (delta as i128) * 1_000_000_000i128
                + (fit as i128) * 4096
                + (hit as i128) * 512
                + (bonus_seen as i128) * 64
                + (sup as i128) * 256
                + (anti as i128) * 8
                - (pen as i128) * 16
                - (rm_sup as i128) * 4;

            if best.map_or(true, |(_, _, bs, bd, bfit, bhit, bsup, banti)| {
                score > bs
                    || (score == bs
                        && (delta > bd
                            || (delta == bd
                                && (fit > bfit
                                    || (fit == bfit
                                        && (hit > bhit
                                            || (hit == bhit
                                                && (sup > bsup
                                                    || (sup == bsup && anti > banti)))))))))
            }) {
                best = Some((cand, rm, score, delta, fit, hit, sup, anti));
            }
        }
    }

    if let Some((cand, rm, _, _, _, _, _, _)) = best {
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
    let mut seen: Vec<u32> = vec![0u32; n];
    let stamp: u32 = 1;

    let mut cand_list: Vec<(i64, usize)> = Vec::with_capacity(TOP_CAND);

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

    for &u in &state.window_core {
        if state.selected_bit[u] {
            let row = unsafe { neigh.get_unchecked(u) };
            for &(cj, _vv) in row.iter() {
                let cand = cj as usize;
                if cand >= n {
                    continue;
                }
                if state.selected_bit[cand] {
                    continue;
                }
                if seen[cand] == stamp {
                    continue;
                }
                seen[cand] = stamp;

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
                push_top(&mut cand_list, s, cand);
            }
        } else {
            let cand = u;
            if cand >= n {
                continue;
            }
            if state.selected_bit[cand] {
                continue;
            }
            if seen[cand] == stamp {
                continue;
            }
            seen[cand] = stamp;

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
            push_top(&mut cand_list, s, cand);
        }
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
    let max_frontier_swaps: usize = params.max_frontier_swaps_override.unwrap_or(if n >= 2500 {
        0
    } else if n >= 1500 {
        1
    } else {
        2
    });

    let mut dirty_window = false;
    let mut n_rebuilds = 0usize;
    let max_rebuilds: usize = if n >= 2500 { 1 } else { 2 };

    loop {
        iterations += 1;
        if iterations > max_iterations {
            break;
        }

        if apply_best_add_windowed(state) {
            dirty_window = true;
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

        {
            const WORST_CAP: usize = 32;
            let mut worst: Vec<(i32, u32, usize)> = Vec::with_capacity(WORST_CAP);

            for i in 0..n {
                if !state.selected_bit[i] {
                    continue;
                }
                let c = state.contrib[i];
                let w = state.ch.weights[i];

                if worst.len() < WORST_CAP {
                    worst.push((c, w, i));
                    continue;
                }

                let mut best_pos = 0usize;
                let mut best_c = worst[0].0;
                let mut best_w = worst[0].1;

                for t in 1..worst.len() {
                    let (cc, ww, _) = worst[t];
                    if cc > best_c || (cc == best_c && ww < best_w) {
                        best_pos = t;
                        best_c = cc;
                        best_w = ww;
                    }
                }

                if c < best_c || (c == best_c && w > best_w) {
                    worst[best_pos] = (c, w, i);
                }
            }

            for &(_, _, idx) in &worst {
                used.push(idx);
            }

            used.sort_unstable();
            used.dedup();
        }

        if apply_best_swap_diff_reduce_windowed_cached(state, &used) {
            dirty_window = true;
            continue;
        }
        if apply_best_swap_diff_increase_windowed_cached(state, &used) {
            dirty_window = true;
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
            dirty_window = true;
            if n_rebuilds < max_rebuilds {
                n_rebuilds += 1;
                dirty_window = false;
                rebuild_windows(state);
            }
            continue;
        }
        if apply_best_replace21_windowed(state, &used) {
            dirty_window = true;
            if n_rebuilds < max_rebuilds {
                n_rebuilds += 1;
                dirty_window = false;
                rebuild_windows(state);
            }
            continue;
        }

        if apply_kick_refill_windowed(state, &used) {
            dirty_window = true;
            if n_rebuilds < max_rebuilds {
                n_rebuilds += 1;
                dirty_window = false;
                rebuild_windows(state);
            }
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