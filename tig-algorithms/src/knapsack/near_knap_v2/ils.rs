use tig_challenges::knapsack::*;
use super::params::{Params, Rng};
use super::state::{
    State, build_sparse_neighbors_and_totals, compute_total_interactions, rebuild_windows,
};
use super::construct::{
    greedy_fill_with_beta, construct_pair_seed_beta, construct_frontier_cluster_grow,
    construct_forward_incremental,
};
use super::search::{dp_refinement, micro_qkp_refinement, local_search_vnd};

fn perturb_by_strategy(
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
    if stall_count > 0 && (strategy == 0 || strategy == 6) {
        let lim = state.hubs_static.len().min(64);
        let extra = 48usize;
        let mut best_a: Option<(i64, usize, u32)> = None;

        for &a in state.hubs_static.iter().take(lim) {
            if state.selected_bit[a] {
                continue;
            }
            let wa = state.ch.weights[a];
            if wa == 0 || wa > cap {
                continue;
            }
            let mut s = (state.total_interactions[a] * 1000) / (wa as i64).max(1);
            s += (state.contrib[a] as i64) * 10;
            s += (rng.next_u32() & 0x3F) as i64;
            if best_a.map_or(true, |(bs, _, _)| s > bs) {
                best_a = Some((s, a, wa));
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
            let mut s = (state.total_interactions[a] * 1000) / (wa as i64).max(1);
            s += (state.contrib[a] as i64) * 10;
            s += (rng.next_u32() & 0x3F) as i64;
            if best_a.map_or(true, |(bs, _, _)| s > bs) {
                best_a = Some((s, a, wa));
            }
        }

        if let Some((_sa, a, wa)) = best_a {
            if let Some(ref ng) = state.neigh {
                let row = unsafe { ng.get_unchecked(a) };
                let pref = row.len().min(64);
                let mut best_b: Option<(i64, usize, u32)> = None;

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
                    let mut s = (v * 1_000_000) / ((wa + wb) as i64).max(1);
                    s += delta * 40;
                    s += (rng.next_u32() & 0x1F) as i64;

                    if best_b.map_or(true, |(bs, _, _)| s > bs) {
                        best_b = Some((s, b, wb));
                    }
                }

                if let Some((_sb, b, wb)) = best_b {
                    target = Some((a, b, wa + wb));
                }
            }
        }
    }

    let base_remove = (selected.len() / 10).max(1);
    let adaptive_mult = 1 + (stall_count / 2);
    let strength_scaled = strength + (selected.len() / 40);
    let n_remove = (base_remove * adaptive_mult)
        .min(strength_scaled)
        .min(selected.len() / 3);

    let (ta, tb, tw) = if let Some((a, b, w)) = target {
        (a, b, w)
    } else {
        (usize::MAX, usize::MAX, 0u32)
    };

    let mut need_w: u32 = 0;
    if ta != usize::MAX {
        let slack = state.slack();
        if tw > slack {
            need_w = tw - slack;
        }
        need_w = need_w.saturating_add(((strength as u32) + (stall_count as u32)).min(10));
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
    for &(_s, i, w) in &removal_candidates {
        if removed >= n_remove && freed >= need_w {
            break;
        }
        if state.selected_bit[i] {
            state.remove_item(i);
            freed = freed.saturating_add(w);
            removed += 1;
        }
    }

    if ta != usize::MAX {
        let wa = state.ch.weights[ta];
        let wb = state.ch.weights[tb];
        if !state.selected_bit[ta] && state.total_weight + wa <= cap {
            state.add_item(ta);
        }
        if !state.selected_bit[tb] && state.total_weight + wb <= cap {
            state.add_item(tb);
        }
    }
}

fn greedy_reconstruct(state: &mut State, rng: &mut Rng, strategy: usize) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;

    let mut candidates: Vec<usize> = (0..n).filter(|&i| !state.selected_bit[i]).collect();

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
        let allow_seed = slack >= 6;
        greedy_fill_with_beta(state, rng, noise, allow_seed);
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

    let n_starts = params.n_starts(n, hard, team_est);

    let mut best: Option<State> = None;
    let mut second: Option<State> = None;

    for sid in 0..n_starts {
        let mut st = State::new_empty(
            challenge,
            &total_pre,
            &hubs_static,
            neigh_pre.as_ref(),
        );

        match sid {
            0 => super::state::build_initial_solution(&mut st),
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

        dp_refinement(&mut st);
        rebuild_windows(&mut st);
        micro_qkp_refinement(&mut st);
        local_search_vnd(&mut st, params);

        if best.as_ref().map_or(true, |b| st.total_value > b.total_value) {
            second = best;
            best = Some(st);
        } else if second.as_ref().map_or(true, |b| st.total_value > b.total_value) {
            second = Some(st);
        }
    }

    if best.is_some() && second.is_some() {
        let base_val = best.as_ref().unwrap().total_value;
        let mut best_new: Option<State> = None;
        let mut best_new_val = base_val;

        {
            let mut hyb = State::new_empty(
                challenge,
                &total_pre,
                &hubs_static,
                neigh_pre.as_ref(),
            );
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
            dp_refinement(&mut hyb);
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
            let mut hyb = State::new_empty(
                challenge,
                &total_pre,
                &hubs_static,
                neigh_pre.as_ref(),
            );
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
                            let w = challenge.weights[i] as i64;
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
            dp_refinement(&mut hyb);
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
    let mut max_rounds = params.n_perturbation_rounds(n);

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

    let stall_limit = params.stall_limit_effective();

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
            dp_refinement(&mut state);
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

            if perturbation_round >= 7 && stall_count >= stall_limit {
                break;
            }

            if perturbation_round >= max_rounds - 1 {
                break;
            }
            stall_count += 1;

            let strategy = perturbation_round % 7;
            let strength = params.perturbation_strength_base(n) + (perturbation_round as usize) / 2;
            perturb_by_strategy(&mut state, &mut rng, strength, stall_count, strategy);
            greedy_reconstruct(&mut state, &mut rng, strategy);
            rebuild_windows(&mut state);
            dp_refinement(&mut state);
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
