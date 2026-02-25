use super::types::{set_windows_from_density, State, N_IT_CONSTRUCT};
use super::types::Rng;

pub fn greedy_fill_with_beta(state: &mut State, rng: &mut Rng, noise_mask: u32, allow_seed: bool) {
    const BETA_NUM: i64 = 3;
    const BETA_DEN: i64 = 20;
    const HUB_K: usize = 12;
    const SUP_BONUS: i64 = 70;

    let n = state.ch.num_items;

    if n >= 2500 {
        if let Some(neigh) = state.neigh {
            const HUB_GLOBAL_K: usize = 64;
            const HUB_PAIR_K: usize = 12;

            let mut hubs_g: Vec<(i64, usize, u32)> = Vec::with_capacity(HUB_GLOBAL_K);
            let lim = state.hubs_static.len().min(256);
            for &i in state.hubs_static.iter().take(lim) {
                if state.selected_bit[i] {
                    continue;
                }
                let w_u = state.ch.weights[i];
                if w_u == 0 {
                    continue;
                }
                let w = w_u as i64;
                let tot_i = state.total_interactions[i];
                let mut ss = (tot_i * 1000) / w.max(1);
                if noise_mask != 0 {
                    ss += (rng.next_u32() & (noise_mask >> 1)) as i64;
                }
                hubs_g.push((ss, i, w_u));
            }
            hubs_g.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            if hubs_g.len() > HUB_GLOBAL_K {
                hubs_g.truncate(HUB_GLOBAL_K);
            }

            let mut in_frontier = vec![false; n];
            let mut frontier: Vec<usize> = Vec::with_capacity(n.min(4096));

            for i in 0..n {
                if !state.selected_bit[i] {
                    continue;
                }
                let row = unsafe { neigh.get_unchecked(i) };
                for &(k, _v) in row.iter() {
                    let u = k as usize;
                    if !state.selected_bit[u] && !in_frontier[u] {
                        in_frontier[u] = true;
                        frontier.push(u);
                    }
                }
            }

            let push_frontier_of =
                |st: &State, in_f: &mut Vec<bool>, fr: &mut Vec<usize>, v: usize| {
                    let row = unsafe { neigh.get_unchecked(v) };
                    for &(k, _vv) in row.iter() {
                        let u = k as usize;
                        if !st.selected_bit[u] && !in_f[u] {
                            in_f[u] = true;
                            fr.push(u);
                        }
                    }
                };

            loop {
                let slack = state.slack();
                if slack == 0 {
                    break;
                }

                let mut best_pos: Option<(usize, i64)> = None;

                let f_len = frontier.len();
                let scan_lim = if f_len > 2048 { 2048 } else { f_len };

                if f_len != 0 && f_len <= 2048 {
                    for &i in &frontier {
                        if state.selected_bit[i] {
                            continue;
                        }
                        let w_u = state.ch.weights[i];
                        if w_u == 0 || w_u > slack {
                            continue;
                        }
                        let c = state.contrib[i] as i64;
                        if c <= 0 {
                            continue;
                        }

                        let tot_i = state.total_interactions[i];
                        let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot_i);

                        let mut s = (adj * 1000) / (w_u as i64).max(1)
                            + (state.support[i] as i64) * SUP_BONUS;

                        if noise_mask != 0 {
                            s += (rng.next_u32() & noise_mask) as i64;
                        }
                        if best_pos.map_or(true, |(_, bs)| s > bs) {
                            best_pos = Some((i, s));
                        }
                    }
                } else if f_len > 2048 {
                    for _ in 0..scan_lim {
                        let idx = (rng.next_u32() as usize) % f_len;
                        let i = frontier[idx];

                        if state.selected_bit[i] {
                            continue;
                        }
                        let w_u = state.ch.weights[i];
                        if w_u == 0 || w_u > slack {
                            continue;
                        }
                        let c = state.contrib[i] as i64;
                        if c <= 0 {
                            continue;
                        }

                        let tot_i = state.total_interactions[i];
                        let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot_i);

                        let mut s = (adj * 1000) / (w_u as i64).max(1)
                            + (state.support[i] as i64) * SUP_BONUS;

                        if noise_mask != 0 {
                            s += (rng.next_u32() & noise_mask) as i64;
                        }
                        if best_pos.map_or(true, |(_, bs)| s > bs) {
                            best_pos = Some((i, s));
                        }
                    }
                }

                if let Some((i, _)) = best_pos {
                    state.add_item(i);
                    push_frontier_of(state, &mut in_frontier, &mut frontier, i);
                    continue;
                }

                let mut best_pair: Option<(usize, usize, i64)> = None;
                if slack >= 2 && !hubs_g.is_empty() {
                    let lim = hubs_g.len().min(HUB_PAIR_K);
                    for t in 0..lim {
                        let a = hubs_g[t].1;
                        let wa = hubs_g[t].2;
                        if state.selected_bit[a] || wa == 0 || wa >= slack {
                            continue;
                        }

                        let row = unsafe { neigh.get_unchecked(a) };
                        let pref = row.len().min(64);
                        for u in 0..pref {
                            let (bb, vv) = row[u];
                            let b = bb as usize;
                            if a == b || state.selected_bit[b] {
                                continue;
                            }
                            let wb = state.ch.weights[b];
                            if wb == 0 || wa + wb > slack {
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

                            let s = (delta * 1_000_000) / ((wa + wb) as i64).max(1);
                            if best_pair.map_or(true, |(_, _, bs)| s > bs) {
                                best_pair = Some((a, b, s));
                            }
                        }
                    }
                }

                if let Some((a, b, _)) = best_pair {
                    state.add_item(a);
                    push_frontier_of(state, &mut in_frontier, &mut frontier, a);
                    if state.slack() >= state.ch.weights[b] && !state.selected_bit[b] {
                        state.add_item(b);
                        push_frontier_of(state, &mut in_frontier, &mut frontier, b);
                    }
                    continue;
                }

                if allow_seed {
                    let mut best_seed: Option<(usize, i64)> = None;
                    for &(ss, i, _w) in &hubs_g {
                        if state.selected_bit[i] {
                            continue;
                        }
                        let wi = state.ch.weights[i];
                        if wi == 0 || wi > slack {
                            continue;
                        }
                        if best_seed.map_or(true, |(_, bs)| ss > bs) {
                            best_seed = Some((i, ss));
                        }
                    }

                    if let Some((i, _)) = best_seed {
                        state.add_item(i);
                        push_frontier_of(state, &mut in_frontier, &mut frontier, i);
                        continue;
                    }
                }

                break;
            }

            return;
        }
    }

    let mut hubs: Vec<(i64, usize, u32)> = Vec::with_capacity(HUB_K);

    loop {
        let slack = state.slack();
        if slack == 0 {
            break;
        }

        let mut best_pos: Option<(usize, i64)> = None;
        let mut best_seed: Option<(usize, i64)> = None;

        hubs.clear();
        let mut hubs_min_score: i64 = i64::MAX;
        let mut hubs_min_pos: usize = 0;

        for i in 0..n {
            if state.selected_bit[i] {
                continue;
            }
            let w_u = state.ch.weights[i];
            if w_u == 0 || w_u > slack {
                continue;
            }
            let w = w_u as i64;

            let c = state.contrib[i] as i64;
            let tot_i = state.total_interactions[i];

            if c > 0 {
                let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot_i);
                let mut s = (adj * 1000) / w + (state.support[i] as i64) * SUP_BONUS;
                if noise_mask != 0 {
                    s += (rng.next_u32() & noise_mask) as i64;
                }
                if best_pos.map_or(true, |(_, bs)| s > bs) {
                    best_pos = Some((i, s));
                }
            }

            let mut ss = (tot_i * 1000) / w;
            if noise_mask != 0 {
                ss += (rng.next_u32() & (noise_mask >> 1)) as i64;
            }
            if best_seed.map_or(true, |(_, bs)| ss > bs) {
                best_seed = Some((i, ss));
            }

            if slack >= 2 {
                if hubs.len() < HUB_K {
                    hubs.push((ss, i, w_u));
                    if ss < hubs_min_score {
                        hubs_min_score = ss;
                        hubs_min_pos = hubs.len() - 1;
                    }
                    if hubs.len() == 1 {
                        hubs_min_score = ss;
                        hubs_min_pos = 0;
                    }
                } else if ss > hubs_min_score {
                    hubs[hubs_min_pos] = (ss, i, w_u);
                    hubs_min_score = hubs[0].0;
                    hubs_min_pos = 0;
                    for t in 1..hubs.len() {
                        if hubs[t].0 < hubs_min_score {
                            hubs_min_score = hubs[t].0;
                            hubs_min_pos = t;
                        }
                    }
                }
            }
        }

        if let Some((i, _)) = best_pos {
            state.add_item(i);
            continue;
        }

        let mut best_pair: Option<(usize, usize, i64)> = None;
        if slack >= 2 && !hubs.is_empty() {
            for &(_hs, a, wa) in &hubs {
                if state.selected_bit[a] || wa == 0 || wa > slack {
                    continue;
                }
                if let Some(ref ng) = state.neigh {
                    let row = unsafe { ng.get_unchecked(a) };
                    for &(bb, vv) in row.iter() {
                        let b = bb as usize;
                        if a == b || state.selected_bit[b] {
                            continue;
                        }
                        let wb = state.ch.weights[b];
                        if wb == 0 || wa + wb > slack {
                            continue;
                        }
                        let v = vv as i64;
                        if v <= 0 {
                            continue;
                        }
                        let tot_a = state.total_interactions[a];
                        let tot_b = state.total_interactions[b];
                        let s = (v * 1_000_000) / ((wa + wb) as i64) + (tot_a + tot_b) / 2000;
                        if best_pair.map_or(true, |(_, _, bs)| s > bs) {
                            best_pair = Some((a, b, s));
                        }
                    }
                } else {
                    let row = &state.ch.interaction_values[a];
                    for b in 0..n {
                        if a == b || state.selected_bit[b] {
                            continue;
                        }
                        let wb = state.ch.weights[b];
                        if wb == 0 || wa + wb > slack {
                            continue;
                        }
                        let v = row[b] as i64;
                        if v <= 0 {
                            continue;
                        }
                        let tot_a = state.total_interactions[a];
                        let tot_b = state.total_interactions[b];
                        let s = (v * 1_000_000) / ((wa + wb) as i64) + (tot_a + tot_b) / 2000;
                        if best_pair.map_or(true, |(_, _, bs)| s > bs) {
                            best_pair = Some((a, b, s));
                        }
                    }
                }
            }
        }

        if let Some((a, b, _)) = best_pair {
            state.add_item(a);
            if state.slack() >= state.ch.weights[b] && !state.selected_bit[b] {
                state.add_item(b);
            }
            continue;
        }

        if allow_seed {
            if let Some((i, _)) = best_seed {
                state.add_item(i);
            } else {
                break;
            }
        } else {
            break;
        }
    }
}

pub fn construct_pair_seed_beta(state: &mut State, rng: &mut Rng) {
    let n = state.ch.num_items;
    if n == 0 {
        return;
    }

    if state.total_weight == 0 {
        let cap = state.ch.max_weight;
        let mut best_pair: Option<(usize, usize, i64)> = None;

        if let Some(ref ng) = state.neigh {
            for i in 0..n {
                let wi = state.ch.weights[i];
                if wi == 0 || wi > cap {
                    continue;
                }
                let row = unsafe { ng.get_unchecked(i) };
                for &(jj, vv) in row.iter() {
                    let j = jj as usize;
                    if j >= i {
                        continue;
                    }
                    let w = wi + state.ch.weights[j];
                    if w == 0 || w > cap {
                        continue;
                    }
                    let v = vv as i64;
                    if v <= 0 {
                        continue;
                    }
                    let ti = state.total_interactions[i];
                    let tj = state.total_interactions[j];

                    let noise: i64 =
                        (((i as i64) * 1315423911i64) ^ ((j as i64) * 2654435761i64)) & 0x3Fi64;

                    let s = (v * 1_000_000) / (w as i64) + (ti + tj) / 2000 + noise;
                    if best_pair.map_or(true, |(_, _, bs)| s > bs) {
                        best_pair = Some((i, j, s));
                    }
                }
            }
        } else {
            let samples = ((n as u32) * 16).min(16000) as usize;
            for _ in 0..samples {
                let i = (rng.next_u32() as usize) % n;
                let j = (rng.next_u32() as usize) % n;
                if i == j {
                    continue;
                }
                let w = state.ch.weights[i] + state.ch.weights[j];
                if w == 0 || w > cap {
                    continue;
                }
                let v = state.ch.interaction_values[i][j] as i64;
                if v <= 0 {
                    continue;
                }
                let ti = state.total_interactions[i];
                let tj = state.total_interactions[j];
                let s = (v * 1_000_000) / (w as i64) + (ti + tj) / 2000;
                if best_pair.map_or(true, |(_, _, bs)| s > bs) {
                    best_pair = Some((i, j, s));
                }
            }
        }

        if let Some((i, j, _)) = best_pair {
            state.add_item(i);
            if state.total_weight + state.ch.weights[j] <= cap && !state.selected_bit[j] {
                state.add_item(j);
            }
        }
    }

    greedy_fill_with_beta(state, rng, 0, true);
}

pub fn construct_frontier_cluster_grow(state: &mut State, rng: &mut Rng) {
    const BETA_NUM: i64 = 3;
    const BETA_DEN: i64 = 20;

    let n = state.ch.num_items;
    if n == 0 {
        return;
    }
    let cap = state.ch.max_weight;

    let mut in_frontier = vec![false; n];
    let mut frontier: Vec<usize> = Vec::with_capacity(n.min(4096));

    let push_frontier_of = |st: &State, in_f: &mut Vec<bool>, fr: &mut Vec<usize>, i: usize| {
        if let Some(ref ng) = st.neigh {
            let row = unsafe { ng.get_unchecked(i) };
            for &(k, _v) in row.iter() {
                let u = k as usize;
                if !st.selected_bit[u] && !in_f[u] {
                    in_f[u] = true;
                    fr.push(u);
                }
            }
        } else {
            let row = &st.ch.interaction_values[i];
            for u in 0..n {
                if row[u] != 0 && !st.selected_bit[u] && !in_f[u] {
                    in_f[u] = true;
                    fr.push(u);
                }
            }
        }
    };

    let add_seed = |st: &mut State,
                        in_f: &mut Vec<bool>,
                        fr: &mut Vec<usize>,
                        seed: usize| {
        if st.selected_bit[seed] {
            return;
        }
        if st.total_weight + st.ch.weights[seed] > cap {
            return;
        }
        st.add_item(seed);
        push_frontier_of(st, in_f, fr, seed);
    };

    let mut best_seed: Option<usize> = None;
    let mut best_s: i64 = i64::MIN;
    let samples = n.min(512).max(64);

    if n <= 1500 {
        for i in 0..n {
            let w = state.ch.weights[i] as i64;
            if w <= 0 || w as u32 > cap {
                continue;
            }
            let s = (state.total_interactions[i] * 1000) / w;
            if s > best_s {
                best_s = s;
                best_seed = Some(i);
            }
        }
    } else {
        for _ in 0..samples {
            let i = (rng.next_u32() as usize) % n;
            let w = state.ch.weights[i] as i64;
            if w <= 0 || w as u32 > cap {
                continue;
            }
            let s = (state.total_interactions[i] * 1000) / w;
            if s > best_s {
                best_s = s;
                best_seed = Some(i);
            }
        }
    }

    if let Some(i) = best_seed {
        add_seed(state, &mut in_frontier, &mut frontier, i);

        let mut best_j: Option<(usize, i64)> = None;
        if let Some(ref ng) = state.neigh {
            let row = unsafe { ng.get_unchecked(i) };
            for &(jj, vv) in row.iter() {
                let j = jj as usize;
                if j == i || state.selected_bit[j] {
                    continue;
                }
                let wsum = state.ch.weights[i] + state.ch.weights[j];
                if wsum == 0 || wsum > cap {
                    continue;
                }
                let v = vv as i64;
                if v <= 0 {
                    continue;
                }
                let s = (v * 1_000_000) / (wsum as i64) + state.total_interactions[j] / 2000;
                if best_j.map_or(true, |(_, bs)| s > bs) {
                    best_j = Some((j, s));
                }
            }
        } else {
            let row = &state.ch.interaction_values[i];
            for j in 0..n {
                if j == i || state.selected_bit[j] {
                    continue;
                }
                let wsum = state.ch.weights[i] + state.ch.weights[j];
                if wsum == 0 || wsum > cap {
                    continue;
                }
                let v = row[j] as i64;
                if v <= 0 {
                    continue;
                }
                let s = (v * 1_000_000) / (wsum as i64) + state.total_interactions[j] / 2000;
                if best_j.map_or(true, |(_, bs)| s > bs) {
                    best_j = Some((j, s));
                }
            }
        }
        if let Some((j, _)) = best_j {
            add_seed(state, &mut in_frontier, &mut frontier, j);
        }
    }

    let team_est = (cap as usize) / 6;
    let max_jumps: usize = if team_est >= 900 {
        10
    } else if team_est >= 450 {
        8
    } else if team_est >= 200 {
        6
    } else {
        3
    };
    let mut jumps_done: usize = 0;

    loop {
        let slack = state.slack();
        if slack == 0 {
            break;
        }

        let mut best_cand: Option<(usize, i64, i64)> = None;
        for &u in &frontier {
            if state.selected_bit[u] {
                continue;
            }
            let wu = state.ch.weights[u];
            if wu == 0 || wu > slack {
                continue;
            }

            let c = state.contrib[u] as i64;
            if c <= 0 {
                continue;
            }
            let tot_u = state.total_interactions[u];
            let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot_u);
            let s0 = (adj * 1000) / (wu as i64).max(1);
            let s = s0 + (rng.next_u32() & 0x0F) as i64;

            if best_cand.map_or(true, |(_, bs, _)| s > bs) {
                best_cand = Some((u, s, s0));
            }
        }

        let allow_early_jump = jumps_done < max_jumps && slack > cap / 5;
        let mut jump: Option<(usize, i64)> = None;
        let mut jump_s: i64 = i64::MIN;

        if allow_early_jump || best_cand.is_none() {
            for _ in 0..samples {
                let i = (rng.next_u32() as usize) % n;
                if state.selected_bit[i] {
                    continue;
                }
                let wi = state.ch.weights[i];
                if wi == 0 || wi >= slack {
                    continue;
                }

                if let Some(ref ng) = state.neigh {
                    let row = unsafe { ng.get_unchecked(i) };
                    let mut ok = false;
                    let lim = row.len().min(10);
                    for t in 0..lim {
                        let j = row[t].0 as usize;
                        if state.selected_bit[j] {
                            continue;
                        }
                        let wj = state.ch.weights[j];
                        if wj != 0 && wi + wj <= slack {
                            ok = true;
                            break;
                        }
                    }
                    if !ok {
                        continue;
                    }
                }

                let s = (state.total_interactions[i] * 1000) / (wi as i64).max(1);
                if s > jump_s {
                    jump_s = s;
                    jump = Some((i, s));
                }
            }
        }

        let do_jump = if let (Some((_u, _s, s0)), Some((_j, js))) = (best_cand, jump) {
            allow_early_jump && js > s0.saturating_mul(2)
        } else {
            best_cand.is_none() && jump.is_some()
        };

        if do_jump {
            if let Some((seed, _)) = jump {
                jumps_done += 1;
                add_seed(state, &mut in_frontier, &mut frontier, seed);

                let slack1 = state.slack();
                if state.selected_bit[seed] && slack1 >= 1 {
                    let mut best_nb: Option<(usize, i64)> = None;

                    if let Some(ref ng) = state.neigh {
                        let row = unsafe { ng.get_unchecked(seed) };
                        let pref = row.len().min(72);
                        for t in 0..pref {
                            let (jj, vv) = row[t];
                            let j = jj as usize;
                            if j == seed || state.selected_bit[j] {
                                continue;
                            }
                            let wj = state.ch.weights[j];
                            if wj == 0 || wj > slack1 {
                                continue;
                            }
                            let v = vv as i64;
                            if v <= 0 {
                                continue;
                            }
                            let wsum = (state.ch.weights[seed] + wj) as i64;
                            let s =
                                (v * 1_000_000) / wsum.max(1) + state.total_interactions[j] / 2000;
                            if best_nb.map_or(true, |(_, bs)| s > bs) {
                                best_nb = Some((j, s));
                            }
                        }
                    } else {
                        let row = &state.ch.interaction_values[seed];
                        for j in 0..n {
                            if j == seed || state.selected_bit[j] {
                                continue;
                            }
                            let wj = state.ch.weights[j];
                            if wj == 0 || wj > slack1 {
                                continue;
                            }
                            let v = row[j] as i64;
                            if v <= 0 {
                                continue;
                            }
                            let wsum = (state.ch.weights[seed] + wj) as i64;
                            let s =
                                (v * 1_000_000) / wsum.max(1) + state.total_interactions[j] / 2000;
                            if best_nb.map_or(true, |(_, bs)| s > bs) {
                                best_nb = Some((j, s));
                            }
                        }
                    }

                    if let Some((j, _)) = best_nb {
                        add_seed(state, &mut in_frontier, &mut frontier, j);
                    }
                }

                continue;
            }
        }

        if let Some((u, _s, _s0)) = best_cand {
            state.add_item(u);
            push_frontier_of(state, &mut in_frontier, &mut frontier, u);
            continue;
        }

        if let Some((seed, _)) = jump {
            add_seed(state, &mut in_frontier, &mut frontier, seed);
            continue;
        }

        break;
    }

    greedy_fill_with_beta(state, rng, 0, true);
}

pub fn construct_forward_incremental(state: &mut State, mode: usize, rng: &mut Rng) {
    let n = state.ch.num_items;

    if state.total_weight == 0 && n > 0 {
        let slack0 = state.slack();
        if slack0 > 0 {
            let tries = n.min(64);
            let samp = n.min(64);
            let mut best_seed: Option<usize> = None;
            let mut best_score: i64 = i64::MIN;

            for _ in 0..tries {
                let i = (rng.next_u32() as usize) % n;
                let wi = state.ch.weights[i];
                if wi == 0 || wi > slack0 {
                    continue;
                }

                let mut est: i64 = 0;
                for _ in 0..samp {
                    let j = (rng.next_u32() as usize) % n;
                    est += state.ch.interaction_values[i][j] as i64;
                }

                let score = (est * 1000) / (wi as i64);
                if score > best_score {
                    best_score = score;
                    best_seed = Some(i);
                }
            }

            if let Some(i) = best_seed {
                state.add_item(i);
            }
        }
    }

    loop {
        let slack = state.slack();
        if slack == 0 {
            break;
        }

        let mut best_i: Option<usize> = None;
        let mut best_s: i64 = i64::MIN;
        let mut second_i: Option<usize> = None;
        let mut second_s: i64 = i64::MIN;

        for i in 0..n {
            if state.selected_bit[i] {
                continue;
            }
            let w_u = state.ch.weights[i];
            if w_u > slack {
                continue;
            }
            let c = state.contrib[i] as i64;
            if c <= 0 {
                continue;
            }

            let w = (w_u as i64).max(1);
            let mut s = match mode {
                2 => c,
                3 => (c * 1000) / w + (w_u as i64) * 3,
                _ => (c * 1000) / w,
            };
            if mode >= 4 {
                let mask = if mode >= 5 { 0x7F } else { 0x1F };
                s += (rng.next_u32() & mask) as i64;
            }

            if s > best_s {
                second_s = best_s;
                second_i = best_i;
                best_s = s;
                best_i = Some(i);
            } else if s > second_s {
                second_s = s;
                second_i = Some(i);
            }
        }

        let pick = if mode >= 4 && second_i.is_some() {
            let m = if mode >= 5 { 1 } else { 3 };
            if (rng.next_u32() & m) == 0 {
                second_i
            } else {
                best_i
            }
        } else {
            best_i
        };

        if let Some(i) = pick {
            state.add_item(i);
        } else {
            break;
        }
    }

    if state.slack() >= 2 {
        let noise = if mode >= 4 { 0x1F } else { 0 };
        greedy_fill_with_beta(state, rng, noise, true);
    }
}

pub fn build_initial_solution(state: &mut State) {
    let n = state.ch.num_items;
    if n == 0 {
        return;
    }
    let cap = state.ch.max_weight;

    let mut sum_values: i64 = 0;
    let mut sum_w: u32 = 0;
    for i in 0..n {
        state.selected_bit[i] = true;

        state.support[i] = if let Some(ng) = state.neigh {
            unsafe { ng.get_unchecked(i).len().min(u16::MAX as usize) as u16 }
        } else {
            let mut c: u16 = 0;
            for &v in state.ch.interaction_values[i].iter() {
                if v != 0 && c < u16::MAX {
                    c += 1;
                }
            }
            c
        };

        let ti = state.total_interactions[i].min(i32::MAX as i64) as i32;
        state.contrib[i] = state.ch.values[i] as i32 + ti;
        sum_values += state.ch.values[i] as i64;
        sum_w += state.ch.weights[i];
    }
    state.total_weight = sum_w;
    let sum_inter: i64 = state.total_interactions.iter().sum();
    state.total_value = sum_values + sum_inter / 2;

    while state.total_weight > cap {
        let mut worst_item = 0;
        let mut worst_score = i64::MAX;

        for i in 0..n {
            if state.selected_bit[i] {
                let contrib = state.contrib[i] as i64;
                let weight = state.ch.weights[i] as i64;
                let score = if weight > 0 {
                    (contrib * 1000) / weight
                } else {
                    contrib * 1000
                };

                if score < worst_score {
                    worst_score = score;
                    worst_item = i;
                }
            }
        }

        state.remove_item(worst_item);
    }

    let mut idx_last_inserted = 0;
    let mut idx_first_rejected = n;
    let mut by_density: Vec<usize> = (0..n).collect();

    for _ in 0..=N_IT_CONSTRUCT {
        idx_last_inserted = 0;
        idx_first_rejected = n;
        let contrib = &state.contrib;
        by_density.sort_unstable_by(|&a, &b| {
            let na = contrib[a] as i64;
            let nb = contrib[b] as i64;
            let wa = state.ch.weights[a] as i64;
            let wb = state.ch.weights[b] as i64;
            (na * wb).cmp(&(nb * wa)).reverse()
        });

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

        let mut in_target = vec![false; n];
        for &i in &target_sel {
            in_target[i] = true;
        }
        let mut to_remove: Vec<usize> = Vec::new();
        let mut to_add: Vec<usize> = Vec::new();
        for i in 0..n {
            if state.selected_bit[i] && !in_target[i] {
                to_remove.push(i);
            }
            if !state.selected_bit[i] && in_target[i] {
                to_add.push(i);
            }
        }

        if to_remove.is_empty() && to_add.is_empty() {
            break;
        }

        for &r in &to_remove {
            state.remove_item(r);
        }
        for &a in &to_add {
            state.add_item(a);
        }
    }

    set_windows_from_density(state, &by_density, idx_first_rejected, idx_last_inserted);
}
