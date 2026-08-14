use super::construct::{
    build_initial_solution, construct_forward_incremental, construct_frontier_cluster_grow,
    construct_pair_seed_beta, greedy_fill_with_beta,
};
use super::local_search::local_search_vnd;
use super::params::Params;
use super::refinement::{dp_refinement_x, micro_qkp_refinement};
use super::types::{
    build_sparse_neighbors_and_totals, compute_total_interactions, rebuild_windows, Rng, State,
};
use tig_challenges::knapsack::*;

pub fn perturb_by_strategy(
    state: &mut State,
    rng: &mut Rng,
    strength: usize,
    stall_count: usize,
    strategy: usize,
) {
    let selected = state.selected_items();
    if selected.is_empty() {
        return;
    }
    let cap = state.ch.max_weight;

    let mut target: Option<(usize, usize, u32)> = None;
    if stall_count > 0 && (strategy == 0 || strategy == 3 || strategy == 6) {
        if let Some(ref ng) = state.neigh {
            let lim = state.hubs_static.len().min(96);
            let extra = 32usize;
            let mut best: Option<(i64, usize, usize, u32)> = None;

            for &a in state.hubs_static.iter().take(lim) {
                if state.selected_bit[a] {
                    continue;
                }
                let wa = state.ch.weights[a];
                if wa == 0 || wa > cap {
                    continue;
                }

                let row = unsafe { ng.get_unchecked(a) };
                let pref = row.len().min(56);
                for t in 0..pref {
                    let (bb, vv) = row[t];
                    let b = bb as usize;
                    if b == a || state.selected_bit[b] {
                        continue;
                    }
                    let wb = state.ch.weights[b];
                    if wb == 0 || wa + wb > cap {
                        continue;
                    }
                    let v = vv as i64;
                    if v <= 0 {
                        continue;
                    }

                    let delta = (state.contrib[a] as i64) + (state.contrib[b] as i64) + v;
                    if delta <= 0 {
                        continue;
                    }

                    let mut s = (v * 1_000_000) / ((wa + wb) as i64).max(1);
                    s += delta * 40;
                    s += (state.total_interactions[a] + state.total_interactions[b]) / 2000;
                    s += (rng.next_u32() & 0x1F) as i64;
                    s += (state.usage[a] as i64 + state.usage[b] as i64) * 18;

                    if best.map_or(true, |(bs, _, _, _)| s > bs) {
                        best = Some((s, a, b, wa + wb));
                    }
                }
            }

            for _ in 0..extra {
                let a = (rng.next_u32() as usize) % state.ch.num_items;
                if state.selected_bit[a] {
                    continue;
                }
                let wa = state.ch.weights[a];
                if wa == 0 || wa > cap {
                    continue;
                }

                let row = unsafe { ng.get_unchecked(a) };
                let pref = row.len().min(48);
                for t in 0..pref {
                    let (bb, vv) = row[t];
                    let b = bb as usize;
                    if b == a || state.selected_bit[b] {
                        continue;
                    }
                    let wb = state.ch.weights[b];
                    if wb == 0 || wa + wb > cap {
                        continue;
                    }
                    let v = vv as i64;
                    if v <= 0 {
                        continue;
                    }

                    let delta = (state.contrib[a] as i64) + (state.contrib[b] as i64) + v;
                    if delta <= 0 {
                        continue;
                    }

                    let mut s = (v * 1_000_000) / ((wa + wb) as i64).max(1);
                    s += delta * 40;
                    s += (state.total_interactions[a] + state.total_interactions[b]) / 2000;
                    s += (rng.next_u32() & 0x1F) as i64;
                    s += (state.usage[a] as i64 + state.usage[b] as i64) * 18;

                    if best.map_or(true, |(bs, _, _, _)| s > bs) {
                        best = Some((s, a, b, wa + wb));
                    }
                }
            }

            if let Some((_s, a, b, w)) = best {
                target = Some((a, b, w));
            }
        }
    }

    let base_remove = (selected.len() / 10).max(1);
    let adaptive_mult = 1 + (stall_count / 2);
    let strength_scaled = strength + (selected.len() / 40);
    let n_remove = (base_remove * adaptive_mult)
        .min(
            ((selected.len() / if stall_count >= 4 { 12 } else { 16 }).max(1))
                .max(strength_scaled),
        )
        .min(selected.len() / 3);

    let (ta, tb, tw) = if let Some((a, b, w)) = target {
        (a, b, w)
    } else {
        (usize::MAX, usize::MAX, 0u32)
    };

    let slack0 = state.slack();
    let mut need_w: u32 = 0;
    if ta != usize::MAX {
        if tw > slack0 {
            need_w = tw - slack0;
        }
        need_w = need_w.saturating_add(((strength as u32) + (stall_count as u32)).min(10));
    } else if stall_count >= 3 && slack0 < 4 {
        need_w = 4 - slack0;
    }

    let mut removal_candidates: Vec<(i64, usize, u32)> = Vec::with_capacity(selected.len());
    for &i in &selected {
        let w = state.ch.weights[i];
        if w == 0 {
            continue;
        }

        let mut s: i64 = match strategy {
            0 => state.contrib[i] as i64,
            1 => -(w as i64),
            2 => (state.contrib[i] - state.ch.values[i] as i32) as i64,
            3 => (state.contrib[i] as i64 * 1000) / (w as i64).max(1),
            4 => {
                (state.contrib[i] as i64 * 1000) / (w as i64).max(1)
                    + (state.support[i] as i64) * 200
            }
            5 => (state.support[i] as i64) * 500 - (w as i64) * 220 + (state.contrib[i] as i64) / 50,
            _ => (state.contrib[i] as i64) - (state.usage[i] as i64) * 50,
        };

        if need_w >= 6 && strategy != 1 {
            let ww = (w as i64).max(1);
            s = (s * 1024) / ww - ww * 12;
        }
        if ta != usize::MAX {
            let ia = state.ch.interaction_values[i][ta] as i64;
            let ib = state.ch.interaction_values[i][tb] as i64;
            s += (ia + ib) * 4 + (state.usage[i] as i64) * 80;
        } else if stall_count >= 3 {
            s += (state.usage[i] as i64) * 30;
        }

        removal_candidates.push((s, i, w));
    }

    removal_candidates.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let mut freed: u32 = 0;
    let mut removed: usize = 0;

    let pool_lim: usize = if ta != usize::MAX { 96 } else { 64 };
    let mut pool: Vec<usize> = Vec::with_capacity(pool_lim);
    let mut cursor: usize = 0;
    let mut last_rm: usize = usize::MAX;

    while removed < n_remove || freed < need_w {
        while cursor < removal_candidates.len() && pool.len() < (pool_lim / 2).max(8) {
            let i = removal_candidates[cursor].1;
            cursor += 1;
            if !state.selected_bit[i] {
                continue;
            }
            let mut dup = false;
            for &x in &pool {
                if x == i {
                    dup = true;
                    break;
                }
            }
            if !dup {
                pool.push(i);
            }
        }

        if last_rm != usize::MAX && pool.len() < pool_lim {
            if let Some(ref ng) = state.neigh {
                let row = unsafe { ng.get_unchecked(last_rm) };
                let pref = row.len().min(32);
                for t in 0..pref {
                    let j = row[t].0 as usize;
                    if !state.selected_bit[j] {
                        continue;
                    }
                    let mut dup = false;
                    for &x in &pool {
                        if x == j {
                            dup = true;
                            break;
                        }
                    }
                    if !dup {
                        pool.push(j);
                        if pool.len() >= pool_lim {
                            break;
                        }
                    }
                }
            }
        }

        let mut best_pos: Option<usize> = None;
        let mut best_s: i64 = i64::MAX;

        for (pos, &i) in pool.iter().enumerate() {
            if !state.selected_bit[i] {
                continue;
            }
            let w = state.ch.weights[i];
            if w == 0 {
                continue;
            }

            let mut s: i64 = match strategy {
                0 => state.contrib[i] as i64,
                1 => -(w as i64),
                2 => (state.contrib[i] - state.ch.values[i] as i32) as i64,
                3 => (state.contrib[i] as i64 * 1000) / (w as i64).max(1),
                4 => {
                    (state.contrib[i] as i64 * 1000) / (w as i64).max(1)
                        + (state.support[i] as i64) * 200
                }
                5 => {
                    (state.support[i] as i64) * 500
                        - (w as i64) * 220
                        + (state.contrib[i] as i64) / 50
                }
                _ => (state.contrib[i] as i64) - (state.usage[i] as i64) * 50,
            };

            if need_w >= 6 && strategy != 1 {
                let ww = (w as i64).max(1);
                s = (s * 1024) / ww - ww * 12;
            }
            if ta != usize::MAX {
                let ia = state.ch.interaction_values[i][ta] as i64;
                let ib = state.ch.interaction_values[i][tb] as i64;
                s += (ia + ib) * 4 + (state.usage[i] as i64) * 80;
            } else if stall_count >= 3 {
                s += (state.usage[i] as i64) * 30;
            }

            if best_pos.map_or(true, |bp| s < best_s || (s == best_s && i < pool[bp])) {
                best_pos = Some(pos);
                best_s = s;
            }
        }

        let pos = match best_pos {
            Some(p) => p,
            None => {
                if cursor >= removal_candidates.len() {
                    break;
                }
                pool.clear();
                last_rm = usize::MAX;
                continue;
            }
        };

        let rm = pool.swap_remove(pos);
        let w = state.ch.weights[rm];
        if w != 0 && state.selected_bit[rm] {
            state.remove_item(rm);
            freed = freed.saturating_add(w);
            removed += 1;
            last_rm = rm;
        }
        if pool.len() > pool_lim {
            pool.truncate(pool_lim);
        }
    }

    if ta != usize::MAX {
        let mut a = ta;
        let mut b = tb;
        if b != usize::MAX && state.ch.weights[b] < state.ch.weights[a] {
            let t = a;
            a = b;
            b = t;
        }

        let wa = state.ch.weights[a];
        if !state.selected_bit[a] && state.total_weight + wa <= cap {
            state.add_item(a);
        }
        if b != usize::MAX {
            let wb = state.ch.weights[b];
            if !state.selected_bit[b] && state.total_weight + wb <= cap {
                state.add_item(b);
            }
        }
    }
}

pub fn greedy_reconstruct(state: &mut State, rng: &mut Rng, strategy: usize) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;

    let mut candidates: Vec<usize> = (0..n).filter(|&i| !state.selected_bit[i]).collect();
    if strategy != 2 {
        candidates.retain(|&i| state.contrib[i] > 0);
    }

    match strategy {
        0 => {
            const BETA_NUM: i64 = 3;
            const BETA_DEN: i64 = 20;

            candidates.sort_unstable_by(|&a, &b| {
                let wa = (state.ch.weights[a] as i64).max(1);
                let wb = (state.ch.weights[b] as i64).max(1);
                let ca = state.contrib[a] as i64;
                let cb = state.contrib[b] as i64;

                let ta = state.total_interactions[a];
                let tb = state.total_interactions[b];

                let adja = ca * BETA_DEN + BETA_NUM * (2 * ca - ta);
                let adjb = cb * BETA_DEN + BETA_NUM * (2 * cb - tb);

                let lhs = (adja as i128) * (wb as i128);
                let rhs = (adjb as i128) * (wa as i128);

                rhs.cmp(&lhs)
                    .then_with(|| state.support[b].cmp(&state.support[a]))
                    .then_with(|| tb.cmp(&ta))
                    .then_with(|| state.contrib[b].cmp(&state.contrib[a]))
            });
        }
        1 => {
            candidates.sort_unstable_by(|&a, &b| {
                state.ch.weights[a]
                    .cmp(&state.ch.weights[b])
                    .then(state.contrib[b].cmp(&state.contrib[a]))
            });
        }
        2 => {
            candidates.sort_unstable_by_key(|&i| {
                -(state.total_interactions[i] + (state.contrib[i] as i64) * 10)
            });
        }
        3 => {
            candidates.sort_unstable_by_key(|&i| {
                let w = state.ch.weights[i] as i64;
                if w > 0 {
                    let eff = (state.contrib[i] as i64 * 100) / w;
                    -eff
                } else {
                    i64::MIN
                }
            });
        }
        4 => {
            candidates.sort_unstable_by_key(|&i| {
                let w = state.ch.weights[i] as i64;
                let c = state.contrib[i] as i64;
                -(c * w * w / 100)
            });
        }
        5 => {
            candidates.sort_unstable_by(|&a, &b| {
                let sa = state.support[a] as i64;
                let sb = state.support[b] as i64;

                let wa = (state.ch.weights[a] as i64).max(1);
                let wb = (state.ch.weights[b] as i64).max(1);

                let ca = state.contrib[a] as i64;
                let cb = state.contrib[b] as i64;

                let da = (ca * 1000) / wa + sa * 60 + state.total_interactions[a] / 500;
                let db = (cb * 1000) / wb + sb * 60 + state.total_interactions[b] / 500;

                db.cmp(&da).then_with(|| b.cmp(&a))
            });
        }
        _ => {
            candidates.sort_unstable_by_key(|&i| {
                let w = (state.ch.weights[i] as i64).max(1);
                let base = (state.contrib[i] as i64 * 10000) / (w * w);
                let penalty = (state.usage[i] as i64) * 50;
                -(base - penalty)
            });
        }
    }

    let allow_zero = strategy == 2;

    let passes = if n >= 3000 { 1 } else { 2 };
    for _ in 0..passes {
        let mut added_any = false;
        for &i in &candidates {
            if state.total_weight >= cap {
                break;
            }
            if state.selected_bit[i] {
                continue;
            }
            let w = state.ch.weights[i];
            if state.total_weight + w <= cap && (allow_zero || state.contrib[i] > 0) {
                state.add_item(i);
                added_any = true;
            }
        }
        if !added_any {
            break;
        }
    }

    let slack = state.slack();
    if slack >= 2 {
        let noise = if strategy == 0 { 0 } else { 0x0F };
        greedy_fill_with_beta(state, rng, noise, true);
    }
}

pub fn run_one_instance(challenge: &Challenge, params: &Params) -> Solution {
    let n = challenge.num_items;
    let mut rng = Rng::from_seed(&challenge.seed);

    let (neigh_pre, total_pre): (Option<Vec<Vec<(u16, i16)>>>, Vec<i64>) = if n >= 900 {
        let (ng, tot) = build_sparse_neighbors_and_totals(challenge);
        (Some(ng), tot)
    } else {
        (None, compute_total_interactions(challenge))
    };

    let team_est = (challenge.max_weight as usize) / 6;

    let sample = n.min(96);
    let mut nz: u32 = 0;
    let mut tot: u32 = 0;
    for i in 0..sample {
        for j in 0..i {
            tot += 1;
            if challenge.interaction_values[i][j] != 0 {
                nz += 1;
            }
        }
    }
    let dens = if tot > 0 {
        (nz as f64) / (tot as f64)
    } else {
        1.0
    };
    let hard = dens < 0.10;

    let mut hubs_all: Vec<(i64, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        let w = challenge.weights[i] as i64;
        if w <= 0 {
            continue;
        }
        let s = (total_pre[i] * 1000) / w.max(1);
        hubs_all.push((s, i));
    }
    hubs_all.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let hubs_k: usize = if n >= 4500 { 320 } else if n >= 2500 { 256 } else { 192 };
    let hubs_static: Vec<usize> = hubs_all
        .into_iter()
        .take(hubs_k.min(n))
        .map(|(_, i)| i)
        .collect();

    let mut n_starts: usize = if n <= 600 {
        if hard { 4 } else { 3 }
    } else if n <= 1500 {
        if hard { 3 } else { 2 }
    } else if n >= 2500 {
        if hard { 4 } else { 3 }
    } else {
        2
    };

    if n <= 1500 && team_est >= 200 {
        n_starts = (n_starts + 1).min(4);
    }

    n_starts += params.extra_starts;

    let mut best: Option<State> = None;
    let mut second: Option<State> = None;

    for sid in 0..n_starts {
        let mut st = State::new_empty(challenge, &total_pre, &hubs_static, neigh_pre.as_ref());

        match sid {
            0 => build_initial_solution(&mut st),
            1 => {
                if n >= 2500 {
                    construct_frontier_cluster_grow(&mut st, &mut rng);
                } else {
                    construct_pair_seed_beta(&mut st, &mut rng);
                }
                rebuild_windows(&mut st);
            }
            2 => {
                if n >= 2500 {
                    construct_pair_seed_beta(&mut st, &mut rng);
                } else {
                    construct_forward_incremental(&mut st, 1, &mut rng);
                }
                rebuild_windows(&mut st);
            }
            _ => {
                let m = if hard { 5 } else { 4 };
                construct_forward_incremental(&mut st, m, &mut rng);
                rebuild_windows(&mut st);
            }
        }

        dp_refinement_x(&mut st, params.dp_passes_multiplier);
        rebuild_windows(&mut st);
        micro_qkp_refinement(&mut st);
        local_search_vnd(&mut st, params);

        if best
            .as_ref()
            .map_or(true, |b| st.total_value > b.total_value)
        {
            second = best;
            best = Some(st);
        } else if second
            .as_ref()
            .map_or(true, |b| st.total_value > b.total_value)
        {
            second = Some(st);
        }
    }

    if best.is_some() && second.is_some() {
        let base_val = best.as_ref().unwrap().total_value;
        let mut best_new: Option<State> = None;
        let mut best_new_val = base_val;

        {
            let mut hyb =
                State::new_empty(challenge, &total_pre, &hubs_static, neigh_pre.as_ref());
            {
                let b1 = best.as_ref().unwrap();
                let b2 = second.as_ref().unwrap();
                for i in 0..n {
                    if b1.selected_bit[i]
                        && b2.selected_bit[i]
                        && hyb.total_weight + challenge.weights[i] <= challenge.max_weight
                    {
                        hyb.add_item(i);
                    }
                }
            }
            greedy_fill_with_beta(&mut hyb, &mut rng, 0, true);
            rebuild_windows(&mut hyb);
            dp_refinement_x(&mut hyb, params.dp_passes_multiplier);
            rebuild_windows(&mut hyb);
            micro_qkp_refinement(&mut hyb);
            local_search_vnd(&mut hyb, params);

            if hyb.total_value > best_new_val {
                best_new_val = hyb.total_value;
                best_new = Some(hyb);
            }
        }

        let (inter_cnt, union_cnt) = {
            let b1 = best.as_ref().unwrap();
            let b2 = second.as_ref().unwrap();
            let mut inter_cnt = 0usize;
            let mut union_cnt = 0usize;
            for i in 0..n {
                let a = b1.selected_bit[i];
                let b = b2.selected_bit[i];
                if a || b {
                    union_cnt += 1;
                }
                if a && b {
                    inter_cnt += 1;
                }
            }
            (inter_cnt, union_cnt)
        };

        if union_cnt > 0 && (inter_cnt * 100) / union_cnt <= 85 {
            let mut hyb =
                State::new_empty(challenge, &total_pre, &hubs_static, neigh_pre.as_ref());
            {
                let b1 = best.as_ref().unwrap();
                let b2 = second.as_ref().unwrap();
                for i in 0..n {
                    if b1.selected_bit[i] || b2.selected_bit[i] {
                        hyb.add_item(i);
                    }
                }
            }

            if hyb.total_weight > challenge.max_weight {
                if n >= 2500 {
                    let mut sel: Vec<(i64, usize)> = Vec::new();
                    for i in 0..n {
                        if !hyb.selected_bit[i] {
                            continue;
                        }
                        let c = hyb.contrib[i] as i64;
                        let w = challenge.weights[i] as i64;
                        let s = if w > 0 { (c * 1000) / w } else { c * 1000 };
                        sel.push((s, i));
                    }
                    sel.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                    for &(_s, i) in &sel {
                        if hyb.total_weight <= challenge.max_weight {
                            break;
                        }
                        if hyb.selected_bit[i] {
                            hyb.remove_item(i);
                        }
                    }
                } else {
                    while hyb.total_weight > challenge.max_weight {
                        let mut worst_item: Option<usize> = None;
                        let mut worst_score: i64 = i64::MAX;
                        for i in 0..n {
                            if !hyb.selected_bit[i] {
                                continue;
                            }
                            let c = hyb.contrib[i] as i64;
                            let w = hyb.ch.weights[i] as i64;
                            let s = if w > 0 { (c * 1000) / w } else { c * 1000 };
                            if s < worst_score {
                                worst_score = s;
                                worst_item = Some(i);
                            }
                        }
                        if let Some(wi) = worst_item {
                            hyb.remove_item(wi);
                        } else {
                            break;
                        }
                    }
                }
            }

            greedy_fill_with_beta(&mut hyb, &mut rng, 0, true);
            rebuild_windows(&mut hyb);
            dp_refinement_x(&mut hyb, params.dp_passes_multiplier);
            rebuild_windows(&mut hyb);
            micro_qkp_refinement(&mut hyb);
            local_search_vnd(&mut hyb, params);

            if hyb.total_value > best_new_val {
                best_new = Some(hyb);
            }
        }

        if let Some(s) = best_new {
            best = Some(s);
        }
    }

    let mut state = best.unwrap();

    let mut best_sel: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        if state.selected_bit[i] {
            best_sel.push(i);
        }
    }
    let mut best_val = state.total_value;

    let mut stall_count = 0;
    let mut max_rounds = params.n_perturbation_rounds;

    if n <= 600 && hard {
        max_rounds = max_rounds.saturating_add(3);
    }

    if n >= 4500 {
        max_rounds = max_rounds.min(if hard { 13 } else { 12 });
    } else if n >= 3000 {
        max_rounds = max_rounds.min(if hard { 15 } else { 14 });
    } else if n >= 2000 {
        max_rounds = max_rounds.min(16);
    }

    for perturbation_round in 0..max_rounds {
        let is_last_round = perturbation_round >= max_rounds - 1;

        state.snap_bits.clone_from(&state.selected_bit);
        state.snap_contrib.clone_from(&state.contrib);
        state.snap_support.clone_from(&state.support);
        let prev_val = state.total_value;
        let prev_weight = state.total_weight;

        let apply_dp = !is_last_round
            && if n >= 4000 {
                perturbation_round < 3 || (perturbation_round % 4 == 0 && stall_count < 2)
            } else if n >= 2000 {
                perturbation_round % 2 == 0 && stall_count < 4
            } else if n >= 1000 {
                stall_count < 5
            } else {
                true
            };
        if apply_dp {
            rebuild_windows(&mut state);
            dp_refinement_x(&mut state, params.dp_passes_multiplier);
            rebuild_windows(&mut state);
            micro_qkp_refinement(&mut state);
        }
        local_search_vnd(&mut state, params);

        if state.total_value > best_val {
            best_val = state.total_value;
            best_sel.clear();
            for i in 0..n {
                if state.selected_bit[i] {
                    if state.usage[i] < u16::MAX {
                        state.usage[i] += 1;
                    }
                    best_sel.push(i);
                }
            }
            stall_count = 0;
        }

        if state.total_value <= prev_val {
            state.restore_snapshot(prev_val, prev_weight);

            if perturbation_round >= 7 && stall_count >= 8 {
                break;
            }

            if perturbation_round >= max_rounds - 1 {
                break;
            }
            stall_count += 1;

            let strategy = perturbation_round % 7;
            let strength =
                params.perturbation_strength_base + (perturbation_round as usize) / 2;
            perturb_by_strategy(&mut state, &mut rng, strength, stall_count, strategy);
            greedy_reconstruct(&mut state, &mut rng, strategy);
            rebuild_windows(&mut state);
            dp_refinement_x(&mut state, params.dp_passes_multiplier);
            rebuild_windows(&mut state);
            micro_qkp_refinement(&mut state);
            local_search_vnd(&mut state, params);

            if state.total_value > best_val {
                best_val = state.total_value;
                best_sel.clear();
                for i in 0..n {
                    if state.selected_bit[i] {
                        if state.usage[i] < u16::MAX {
                            state.usage[i] += 1;
                        }
                        best_sel.push(i);
                    }
                }
                stall_count = 0;
            }
        }
    }

    Solution { items: best_sel }
}
