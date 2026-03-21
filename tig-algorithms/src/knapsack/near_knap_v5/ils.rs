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

thread_local! {
    static ACTIVE_FRONTIER: core::cell::RefCell<Vec<usize>> = core::cell::RefCell::new(Vec::new());
}

#[derive(Clone, Copy)]
struct TabuCtx {
    ptr: *mut u32,
    len: usize,
    iter: u32,
    tenure: u32,
    aspiration: i64,
    enabled: bool,
}

thread_local! {
    static TABU_CTX: core::cell::RefCell<TabuCtx> = core::cell::RefCell::new(TabuCtx {
        ptr: core::ptr::null_mut(),
        len: 0,
        iter: 0,
        tenure: 0,
        aspiration: i64::MIN,
        enabled: false,
    });
}

struct TabuGuard(TabuCtx);

impl TabuGuard {
    fn activate(ptr: *mut u32, len: usize, iter: u32, tenure: u32, aspiration: i64) -> Self {
        let prev = TABU_CTX.with(|t| {
            let mut ctx = t.borrow_mut();
            let prev = *ctx;
            let enabled = !ptr.is_null() && len > 0 && tenure > 0;
            *ctx = TabuCtx {
                ptr,
                len,
                iter,
                tenure,
                aspiration,
                enabled,
            };
            prev
        });
        TabuGuard(prev)
    }
}

impl Drop for TabuGuard {
    fn drop(&mut self) {
        TABU_CTX.with(|t| {
            *t.borrow_mut() = self.0;
        });
    }
}

#[inline]
fn tabu_active() -> bool {
    TABU_CTX.with(|t| t.borrow().enabled)
}

#[inline]
fn tabu_recent(i: usize) -> bool {
    TABU_CTX.with(|t| {
        let ctx = *t.borrow();
        if !ctx.enabled || i >= ctx.len {
            return false;
        }
        unsafe {
            let stamp = *ctx.ptr.add(i);
            ctx.iter.saturating_sub(stamp) <= ctx.tenure
        }
    })
}

#[inline]
fn tabu_aspiration_ok(state: &State, i: usize) -> bool {
    TABU_CTX.with(|t| {
        let ctx = *t.borrow();
        if !ctx.enabled {
            return true;
        }
        (state.total_value as i64) + (state.contrib[i] as i64) > ctx.aspiration
    })
}

#[inline]
fn tabu_mark(i: usize) {
    TABU_CTX.with(|t| {
        let ctx = *t.borrow();
        if !ctx.enabled || i >= ctx.len {
            return;
        }
        unsafe {
            *ctx.ptr.add(i) = ctx.iter;
        }
    });
}

fn frontier_reset(seed: Vec<usize>) {
    ACTIVE_FRONTIER.with(|f| {
        let mut v = f.borrow_mut();
        v.clear();
        for i in seed {
            if v.iter().any(|&x| x == i) {
                continue;
            }
            v.push(i);
            if v.len() > 64 {
                let drop = v.len() - 64;
                v.drain(0..drop);
            }
        }
    });
}

fn frontier_push(i: usize) {
    ACTIVE_FRONTIER.with(|f| {
        let mut v = f.borrow_mut();
        if v.iter().any(|&x| x == i) {
            return;
        }
        v.push(i);
        if v.len() > 64 {
            let drop = v.len() - 64;
            v.drain(0..drop);
        }
    });
}

fn frontier_clone() -> Vec<usize> {
    ACTIVE_FRONTIER.with(|f| f.borrow().clone())
}

fn seed_frontier_from_state(state: &State) {
    let mut seed: Vec<usize> = Vec::new();
    let k = state.hubs_static.len().min(32);
    for &i in state.hubs_static.iter().take(k) {
        if state.selected_bit[i] {
            seed.push(i);
        }
    }
    if seed.is_empty() {
        let mut cnt = 0usize;
        for i in 0..state.ch.num_items {
            if state.selected_bit[i] {
                seed.push(i);
                cnt += 1;
                if cnt >= 32 {
                    break;
                }
            }
        }
    }
    frontier_reset(seed);
}

fn restart_from_mutated_best<'a>(
    challenge: &'a Challenge,
    total_pre: &'a [i64],
    hubs_static: &'a [usize],
    neigh: Option<&'a Vec<Vec<(u16, i16)>>>,
    best_sel: &[usize],
    rng: &mut Rng,
    strategy: usize,
    strength: usize,
) -> State<'a> {
    let mut st = State::new_empty(challenge, total_pre, hubs_static, neigh);
    for &i in best_sel {
        if st.total_weight + challenge.weights[i] <= challenge.max_weight {
            st.add_item(i);
        }
    }

    let selected = st.selected_items();
    if selected.len() > 1 {
        let mut scored: Vec<(i64, i64, i64, i64, usize, u32)> = Vec::with_capacity(selected.len());

        for &i in &selected {
            let w = challenge.weights[i];
            if w == 0 {
                continue;
            }

            let base = challenge.values[i] as i64;
            let inter_part = (st.contrib[i] as i64) - base;

            let mut neg_mass: i64 = 0;
            if let Some(ng) = neigh {
                let row = unsafe { ng.get_unchecked(i) };
                for &(jj, vv) in row.iter() {
                    if vv < 0 {
                        let j = jj as usize;
                        if j < st.selected_bit.len() && st.selected_bit[j] {
                            neg_mass += vv as i64;
                        }
                    }
                }
            }

            let supp = st.support[i] as i64;
            let contrib = st.contrib[i] as i64;
            scored.push((neg_mass, supp, -inter_part, contrib, i, w));
        }

        scored.sort_unstable_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| a.3.cmp(&b.3))
                .then_with(|| a.4.cmp(&b.4))
        });

        let sel_len = selected.len();
        let mut k = (sel_len / 10).max(1);
        let k2 = strength.min((sel_len / 3).max(1));
        if k2 > k {
            k = k2;
        }
        if k >= sel_len {
            k = sel_len - 1;
        }

        let mut removed: Vec<usize> = Vec::with_capacity(k);
        for t in 0..scored.len().min(k) {
            let i = scored[t].4;
            if st.selected_bit[i] {
                st.remove_item(i);
                removed.push(i);
            }
        }
        if !removed.is_empty() {
            frontier_reset(removed);
        }
    }

    greedy_reconstruct(&mut st, rng, strategy);
    rebuild_windows(&mut st);
    micro_qkp_refinement(&mut st);
    st
}

fn one_one_swap_phase(state: &mut State) {
    let cap = state.ch.max_weight;

    loop {
        let mut improved = false;

        let mut focus: Vec<usize> = Vec::new();
        focus.reserve(state.hubs_static.len());
        for &i in state.hubs_static.iter() {
            if state.selected_bit[i] {
                focus.push(i);
            }
        }

        let selected = if focus.is_empty() {
            state.selected_items()
        } else {
            focus
        };

        for r in selected {
            if !state.selected_bit[r] {
                continue;
            }
            let wr = state.ch.weights[r];
            if wr == 0 {
                continue;
            }

            let loss = state.contrib[r];
            state.remove_item(r);

            let slack = cap - state.total_weight;

            let mut best_a: usize = usize::MAX;
            let mut best_gain: i32 = i32::MIN;
            let mut best_w: u32 = 0;

            if let Some(ref ng) = state.neigh {
                let row = unsafe { ng.get_unchecked(r) };
                let pref = row.len().min(32);
                for t in 0..pref {
                    let a = row[t].0 as usize;
                    if a == r || state.selected_bit[a] {
                        continue;
                    }
                    let wa = state.ch.weights[a];
                    if wa == 0 || wa > slack {
                        continue;
                    }
                    if tabu_active() && tabu_recent(a) && !tabu_aspiration_ok(state, a) {
                        continue;
                    }
                    let g = state.contrib[a];
                    if g > best_gain || (g == best_gain && wa < best_w) {
                        best_gain = g;
                        best_a = a;
                        best_w = wa;
                    }
                }
            }

            let hub_pref = state.hubs_static.len().min(32);
            for &a in state.hubs_static.iter().take(hub_pref) {
                if a == r || state.selected_bit[a] {
                    continue;
                }
                let wa = state.ch.weights[a];
                if wa == 0 || wa > slack {
                    continue;
                }
                if tabu_active() && tabu_recent(a) && !tabu_aspiration_ok(state, a) {
                    continue;
                }
                let g = state.contrib[a];
                if g > best_gain || (g == best_gain && wa < best_w) {
                    best_gain = g;
                    best_a = a;
                    best_w = wa;
                }
            }

            if best_a != usize::MAX && best_gain > loss {
                let wa = state.ch.weights[best_a];
                if wa != 0 && state.total_weight + wa <= cap && !state.selected_bit[best_a] {
                    state.add_item(best_a);
                    tabu_mark(best_a);
                    frontier_push(best_a);
                    improved = true;
                    break;
                }
            }

            if !state.selected_bit[r] && state.total_weight + wr <= cap {
                state.add_item(r);
                tabu_mark(r);
            } else if !state.selected_bit[r] {
                let slack2 = cap - state.total_weight;
                if slack2 > 0 {
                    let mut best_i: Option<usize>;
                    let mut best_c: i32;
                    let mut best_w2: u32;
                    let mut respect_tabu = tabu_active();
                    loop {
                        best_i = None;
                        best_c = 0;
                        best_w2 = 0;
                        for i in 0..state.ch.num_items {
                            if state.selected_bit[i] {
                                continue;
                            }
                            let w = state.ch.weights[i];
                            if w == 0 || w > slack2 {
                                continue;
                            }
                            if respect_tabu && tabu_recent(i) && !tabu_aspiration_ok(state, i) {
                                continue;
                            }
                            let c = state.contrib[i];
                            if c <= 0 {
                                continue;
                            }
                            if best_i.map_or(true, |_| c > best_c || (c == best_c && w < best_w2))
                            {
                                best_i = Some(i);
                                best_c = c;
                                best_w2 = w;
                            }
                        }
                        if best_i.is_some() || !respect_tabu {
                            break;
                        }
                        respect_tabu = false;
                    }
                    if let Some(i) = best_i {
                        state.add_item(i);
                        tabu_mark(i);
                    }
                }
            }
        }

        if !improved {
            break;
        }
    }
}

fn one_two_exchange_phase(state: &mut State) {
    if state.neigh.is_none() {
        return;
    }
    let cap = state.ch.max_weight;
    let n = state.ch.num_items;

    loop {
        let mut improved = false;

        let mut seeds: Vec<usize> = frontier_clone();
        if seeds.len() > 64 {
            let drop = seeds.len() - 64;
            seeds.drain(0..drop);
        }
        let hub_lim = state.hubs_static.len().min(32);
        for &h in state.hubs_static.iter().take(hub_lim) {
            seeds.push(h);
        }
        seeds.sort_unstable();
        seeds.dedup();

        let mut pairs: Vec<(usize, usize, i32, u32)> = Vec::new();
        if let Some(ref ng) = state.neigh {
            for &a0 in &seeds {
                if a0 >= n {
                    continue;
                }
                let row = unsafe { ng.get_unchecked(a0) };
                let pref = row.len().min(64);
                for t in 0..pref {
                    let b0 = row[t].0 as usize;
                    if b0 >= n || b0 == a0 {
                        continue;
                    }
                    let inter = row[t].1 as i32;
                    if inter <= 0 {
                        continue;
                    }
                    let wa = state.ch.weights[a0];
                    let wb = state.ch.weights[b0];
                    if wa == 0 || wb == 0 {
                        continue;
                    }
                    let (a, b) = if a0 < b0 { (a0, b0) } else { (b0, a0) };
                    pairs.push((a, b, inter, wa + wb));
                }
            }
        }
        if pairs.is_empty() {
            return;
        }

        pairs.sort_unstable_by(|x, y| {
            x.0.cmp(&y.0)
                .then_with(|| x.1.cmp(&y.1))
                .then_with(|| y.2.cmp(&x.2))
        });
        pairs.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

        let mut focus: Vec<usize> = Vec::new();
        focus.reserve(state.hubs_static.len());
        for &i in state.hubs_static.iter() {
            if state.selected_bit[i] {
                focus.push(i);
            }
        }

        let selected = if focus.is_empty() {
            state.selected_items()
        } else {
            focus
        };

        for r in selected {
            if !state.selected_bit[r] {
                continue;
            }
            let wr = state.ch.weights[r];
            if wr == 0 {
                continue;
            }

            let loss = state.contrib[r] as i64;
            state.remove_item(r);

            let slack = cap - state.total_weight;

            let mut best_a: usize = usize::MAX;
            let mut best_b: usize = usize::MAX;
            let mut best_delta: i64 = 0;

            for &(a, b, inter, wsum) in &pairs {
                if wsum > slack {
                    continue;
                }
                if state.selected_bit[a] || state.selected_bit[b] {
                    continue;
                }
                let delta =
                    (state.contrib[a] as i64) + (state.contrib[b] as i64) + (inter as i64) - loss;
                if delta > best_delta {
                    best_delta = delta;
                    best_a = a;
                    best_b = b;
                }
            }

            if best_a != usize::MAX && best_delta > 0 {
                let wa = state.ch.weights[best_a];
                let wb = state.ch.weights[best_b];
                if wa != 0
                    && wb != 0
                    && !state.selected_bit[best_a]
                    && !state.selected_bit[best_b]
                    && state.total_weight + wa + wb <= cap
                {
                    state.add_item(best_a);
                    state.add_item(best_b);
                    frontier_push(r);
                    frontier_push(best_a);
                    frontier_push(best_b);
                    improved = true;
                    break;
                } else {
                    state.add_item(r);
                }
            } else {
                state.add_item(r);
            }
        }

        if !improved {
            break;
        }
    }
}

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

    let mut respect_tabu_rm = tabu_active();
    let mut removal_candidates: Vec<(i64, usize, u32)> = Vec::with_capacity(selected.len());
    loop {
        removal_candidates.clear();
        for &i in &selected {
            let w = state.ch.weights[i];
            if w == 0 {
                continue;
            }
            if respect_tabu_rm && tabu_recent(i) {
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

            removal_candidates.push((s, i, w));
        }
        if !removal_candidates.is_empty() || !respect_tabu_rm {
            break;
        }
        respect_tabu_rm = false;
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
                    if respect_tabu_rm && tabu_recent(j) {
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
            tabu_mark(rm);
            frontier_push(rm);
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
            tabu_mark(a);
            frontier_push(a);
        }
        if b != usize::MAX {
            let wb = state.ch.weights[b];
            if !state.selected_bit[b] && state.total_weight + wb <= cap {
                state.add_item(b);
                tabu_mark(b);
                frontier_push(b);
            }
        }
    }
}

pub fn greedy_reconstruct(state: &mut State, rng: &mut Rng, strategy: usize) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let allow_zero = strategy == 2;

    loop {
        if state.total_weight >= cap {
            break;
        }
        let slack = cap - state.total_weight;

        let mut respect_tabu = tabu_active();
        let chosen: Option<usize> = loop {
            let mut best_i: Option<usize> = None;

            let mut best_w: u32 = 0;
            let mut best_c: i32 = 0;
            let mut best_supp: i64 = 0;
            let mut best_ti: i64 = 0;
            let mut best_adja: i64 = 0;
            let mut best_score_i64: i64 = 0;
            let mut best_score_i128: i128 = 0;

            let mut restricted_candidates: Vec<usize> = Vec::new();
            let use_restricted = state.neigh.is_some();
            if use_restricted {
                let frontier = frontier_clone();
                if let Some(ref ng) = state.neigh {
                    let f_lim = frontier.len().min(32);
                    for &f in frontier.iter().take(f_lim) {
                        if f >= n {
                            continue;
                        }
                        let row = unsafe { ng.get_unchecked(f) };
                        let pref = row.len().min(64);
                        for t in 0..pref {
                            restricted_candidates.push(row[t].0 as usize);
                        }
                    }
                    let hub_lim = state.hubs_static.len().min(32);
                    for &h in state.hubs_static.iter().take(hub_lim) {
                        restricted_candidates.push(h);
                    }
                    restricted_candidates.sort_unstable();
                    restricted_candidates.dedup();
                }
            }

            let mut scanned_any = false;

            let scan_iter = |i: usize,
                                 best_i: &mut Option<usize>,
                                 best_w: &mut u32,
                                 best_c: &mut i32,
                                 best_supp: &mut i64,
                                 best_ti: &mut i64,
                                 best_adja: &mut i64,
                                 best_score_i64: &mut i64,
                                 best_score_i128: &mut i128,
                                 scanned_any_ref: &mut bool| {
                if i >= n {
                    return;
                }
                if state.selected_bit[i] {
                    return;
                }
                let w = state.ch.weights[i];
                if w == 0 || w > slack {
                    return;
                }
                let c = state.contrib[i];
                if !allow_zero && c <= 0 {
                    return;
                }

                if respect_tabu && tabu_recent(i) && !tabu_aspiration_ok(state, i) {
                    return;
                }

                *scanned_any_ref = true;

                let ua = state.usage[i];

                match strategy {
                    0 => {
                        const BETA_NUM: i64 = 3;
                        const BETA_DEN: i64 = 20;

                        let wa = (w as i64).max(1);
                        let ca = c as i64;
                        let ta = state.total_interactions[i];
                        let sa = state.support[i] as i64;
                        let adja = ca * BETA_DEN + BETA_NUM * (2 * ca - ta);

                        let better = if let Some(bi) = *best_i {
                            let ub = state.usage[bi];
                            let wb = (*best_w as i64).max(1);
                            let lhs = (adja as i128) * (wb as i128);
                            let rhs = (*best_adja as i128) * (wa as i128);
                            if lhs != rhs {
                                lhs > rhs
                            } else if sa != *best_supp {
                                sa > *best_supp
                            } else if ta != *best_ti {
                                ta > *best_ti
                            } else if ca != *best_c as i64 {
                                ca > *best_c as i64
                            } else if ua != ub {
                                ua < ub
                            } else {
                                i < bi
                            }
                        } else {
                            true
                        };

                        if better {
                            *best_i = Some(i);
                            *best_w = w;
                            *best_c = c;
                            *best_supp = sa;
                            *best_ti = ta;
                            *best_adja = adja;
                        }
                    }
                    1 => {
                        let better = if let Some(bi) = *best_i {
                            let ub = state.usage[bi];
                            if w != *best_w {
                                w < *best_w
                            } else if c != *best_c {
                                c > *best_c
                            } else if ua != ub {
                                ua < ub
                            } else {
                                i < bi
                            }
                        } else {
                            true
                        };

                        if better {
                            *best_i = Some(i);
                            *best_w = w;
                            *best_c = c;
                        }
                    }
                    2 => {
                        let score = state.total_interactions[i] + (c as i64) * 10;
                        let better = if let Some(bi) = *best_i {
                            let ub = state.usage[bi];
                            if score != *best_score_i64 {
                                score > *best_score_i64
                            } else if ua != ub {
                                ua < ub
                            } else {
                                i < bi
                            }
                        } else {
                            true
                        };

                        if better {
                            *best_i = Some(i);
                            *best_w = w;
                            *best_c = c;
                            *best_score_i64 = score;
                        }
                    }
                    3 => {
                        let score = if w > 0 {
                            (c as i64 * 100) / (w as i64)
                        } else {
                            i64::MIN
                        };

                        let better = if let Some(bi) = *best_i {
                            let ub = state.usage[bi];
                            if score != *best_score_i64 {
                                score > *best_score_i64
                            } else if c != *best_c {
                                c > *best_c
                            } else if ua != ub {
                                ua < ub
                            } else {
                                i < bi
                            }
                        } else {
                            true
                        };

                        if better {
                            *best_i = Some(i);
                            *best_w = w;
                            *best_c = c;
                            *best_score_i64 = score;
                        }
                    }
                    4 => {
                        let ww = w as i128;
                        let score = (c as i128) * ww * ww / 100;

                        let better = if let Some(bi) = *best_i {
                            let ub = state.usage[bi];
                            if score != *best_score_i128 {
                                score > *best_score_i128
                            } else if c != *best_c {
                                c > *best_c
                            } else if ua != ub {
                                ua < ub
                            } else {
                                i < bi
                            }
                        } else {
                            true
                        };

                        if better {
                            *best_i = Some(i);
                            *best_w = w;
                            *best_c = c;
                            *best_score_i128 = score;
                        }
                    }
                    5 => {
                        let wa = (w as i64).max(1);
                        let ca = c as i64;
                        let sa = state.support[i] as i64;
                        let score = (ca * 1000) / wa + sa * 60 + state.total_interactions[i] / 500;

                        let better = if let Some(bi) = *best_i {
                            let ub = state.usage[bi];
                            if score != *best_score_i64 {
                                score > *best_score_i64
                            } else if ua != ub {
                                ua < ub
                            } else {
                                i > bi
                            }
                        } else {
                            true
                        };

                        if better {
                            *best_i = Some(i);
                            *best_w = w;
                            *best_c = c;
                            *best_score_i64 = score;
                        }
                    }
                    _ => {
                        let wa = (w as i128).max(1);
                        let ca = c as i128;
                        let base = (ca * 10000) / (wa * wa);
                        let penalty = (state.usage[i] as i128) * 50;
                        let score = base - penalty;

                        let better = if let Some(bi) = *best_i {
                            let ub = state.usage[bi];
                            if score != *best_score_i128 {
                                score > *best_score_i128
                            } else if ua != ub {
                                ua < ub
                            } else {
                                i < bi
                            }
                        } else {
                            true
                        };

                        if better {
                            *best_i = Some(i);
                            *best_w = w;
                            *best_c = c;
                            *best_score_i128 = score;
                        }
                    }
                }
            };

            if !restricted_candidates.is_empty() {
                for &i in &restricted_candidates {
                    scan_iter(
                        i,
                        &mut best_i,
                        &mut best_w,
                        &mut best_c,
                        &mut best_supp,
                        &mut best_ti,
                        &mut best_adja,
                        &mut best_score_i64,
                        &mut best_score_i128,
                        &mut scanned_any,
                    );
                }
            }

            let scanned_any_now = scanned_any;
            if best_i.is_none() && !scanned_any_now {
                for i in 0..n {
                    scan_iter(
                        i,
                        &mut best_i,
                        &mut best_w,
                        &mut best_c,
                        &mut best_supp,
                        &mut best_ti,
                        &mut best_adja,
                        &mut best_score_i64,
                        &mut best_score_i128,
                        &mut scanned_any,
                    );
                }
            } else if best_i.is_none() && restricted_candidates.len() < n {
                for i in 0..n {
                    scan_iter(
                        i,
                        &mut best_i,
                        &mut best_w,
                        &mut best_c,
                        &mut best_supp,
                        &mut best_ti,
                        &mut best_adja,
                        &mut best_score_i64,
                        &mut best_score_i128,
                        &mut scanned_any,
                    );
                }
            }

            if best_i.is_some() || !respect_tabu || !tabu_active() {
                break best_i;
            }
            respect_tabu = false;
        };

        let i = match chosen {
            Some(x) => x,
            None => break,
        };

        if !state.selected_bit[i] {
            let w = state.ch.weights[i];
            if w > 0 && state.total_weight + w <= cap {
                state.add_item(i);
                tabu_mark(i);
                frontier_push(i);
            } else {
                break;
            }
        } else {
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
    let hubs_k: usize = if n >= 4500 {
        320
    } else if n >= 2500 {
        256
    } else {
        192
    };
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
            let mut hyb = State::new_empty(challenge, &total_pre, &hubs_static, neigh_pre.as_ref());
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
            let mut hyb = State::new_empty(challenge, &total_pre, &hubs_static, neigh_pre.as_ref());
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

    let mut state_int = best.unwrap();
    let mut state_div = if let Some(s) = second {
        s
    } else {
        let mut st = State::new_empty(challenge, &total_pre, &hubs_static, neigh_pre.as_ref());
        for i in 0..n {
            if state_int.selected_bit[i] {
                st.add_item(i);
            }
        }
        st
    };

    if state_div.total_value > state_int.total_value {
        core::mem::swap(&mut state_int, &mut state_div);
    }

    let mut tabu_int: Vec<u32> = vec![0; n];
    let mut tabu_div: Vec<u32> = vec![0; n];

    seed_frontier_from_state(&state_int);

    let mut best_sel: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        if state_int.selected_bit[i] {
            best_sel.push(i);
        }
    }
    let mut best_val = state_int.total_value;
    if state_div.total_value > best_val {
        best_val = state_div.total_value;
        best_sel.clear();
        for i in 0..n {
            if state_div.selected_bit[i] {
                best_sel.push(i);
            }
        }
    }

    let mut stall_int = 0usize;
    let mut stall_div = 0usize;
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

    let mut dp_next_int = true;

    for perturbation_round in 0..max_rounds {
        let is_last_round = perturbation_round >= max_rounds - 1;
        let iter_u32 = perturbation_round as u32;

        {
            state_int.snap_bits.clone_from(&state_int.selected_bit);
            state_int.snap_contrib.clone_from(&state_int.contrib);
            state_int.snap_support.clone_from(&state_int.support);
            let prev_val = state_int.total_value;
            let prev_weight = state_int.total_weight;

            let apply_dp = !is_last_round && dp_next_int;
            let applied_dp_this_round = apply_dp;
            dp_next_int = false;
            if apply_dp {
                rebuild_windows(&mut state_int);
                dp_refinement_x(&mut state_int, params.dp_passes_multiplier);
                rebuild_windows(&mut state_int);
                micro_qkp_refinement(&mut state_int);
            }

            one_two_exchange_phase(&mut state_int);
            local_search_vnd(&mut state_int, params);

            if state_int.total_value > best_val {
                best_val = state_int.total_value;
                best_sel.clear();
                for i in 0..n {
                    if state_int.selected_bit[i] {
                        if state_int.usage[i] < u16::MAX {
                            state_int.usage[i] += 1;
                        }
                        best_sel.push(i);
                    }
                }
                stall_int = 0;
            }

            let mut accept_plateau = false;
            if state_int.total_value == prev_val {
                let mut cur_supp_sum: i64 = i64::default();
                let mut snap_supp_sum: i64 = i64::default();
                for i in 0..n {
                    if state_int.selected_bit[i] {
                        cur_supp_sum += state_int.support[i] as i64;
                    }
                    if state_int.snap_bits[i] {
                        snap_supp_sum += state_int.snap_support[i] as i64;
                    }
                }
                accept_plateau = cur_supp_sum > snap_supp_sum
                    || (cur_supp_sum == snap_supp_sum && state_int.total_weight < prev_weight);
            }

            if state_int.total_value < prev_val
                || (state_int.total_value == prev_val && !accept_plateau)
            {
                state_int.restore_snapshot(prev_val, prev_weight);

                if perturbation_round >= 7 && stall_int >= 8 {
                    let strategy = perturbation_round % 7;
                    let strength =
                        params.perturbation_strength_base + (perturbation_round as usize) / 2;
                    state_int = restart_from_mutated_best(
                        challenge,
                        &total_pre,
                        &hubs_static,
                        neigh_pre.as_ref(),
                        &best_sel,
                        &mut rng,
                        strategy,
                        strength,
                    );
                    seed_frontier_from_state(&state_int);
                    stall_int = 0;
                }

                if !is_last_round {
                    stall_int += 1;

                    let strategy = perturbation_round % 7;
                    let strength =
                        params.perturbation_strength_base + (perturbation_round as usize) / 2;
                    let tenure_u32 = strength as u32;

                    seed_frontier_from_state(&state_int);

                    let mut before = state_int.selected_items();
                    before.sort_unstable();

                    {
                        let _g = TabuGuard::activate(
                            tabu_int.as_mut_ptr(),
                            tabu_int.len(),
                            iter_u32,
                            tenure_u32,
                            best_val as i64,
                        );
                        perturb_by_strategy(&mut state_int, &mut rng, strength, stall_int, strategy);
                    }

                    for &i in &before {
                        if !state_int.selected_bit[i] {
                            tabu_int[i] = iter_u32;
                        }
                    }
                    let after = state_int.selected_items();
                    for i in after {
                        if before.binary_search(&i).is_err() {
                            tabu_int[i] = iter_u32;
                        }
                    }

                    {
                        let _g = TabuGuard::activate(
                            tabu_int.as_mut_ptr(),
                            tabu_int.len(),
                            iter_u32,
                            tenure_u32,
                            best_val as i64,
                        );
                        greedy_reconstruct(&mut state_int, &mut rng, strategy);

                        {
                            let slack = state_int.slack();
                            if slack > 0 {
                                let mut best_i: Option<usize>;
                                let mut best_c: i32;
                                let mut best_w: u32;
                                let mut respect_tabu = tabu_active();
                                loop {
                                    best_i = None;
                                    best_c = 0;
                                    best_w = 0;
                                    for i in 0..n {
                                        if state_int.selected_bit[i] {
                                            continue;
                                        }
                                        let w = challenge.weights[i];
                                        if w == 0 || w > slack {
                                            continue;
                                        }
                                        if respect_tabu
                                            && tabu_recent(i)
                                            && !tabu_aspiration_ok(&state_int, i)
                                        {
                                            continue;
                                        }
                                        let c = state_int.contrib[i];
                                        if c <= 0 {
                                            continue;
                                        }
                                        if best_i.map_or(true, |_| {
                                            c > best_c || (c == best_c && w < best_w)
                                        }) {
                                            best_i = Some(i);
                                            best_c = c;
                                            best_w = w;
                                        }
                                    }
                                    if best_i.is_some() || !respect_tabu {
                                        break;
                                    }
                                    respect_tabu = false;
                                }
                                if let Some(i) = best_i {
                                    state_int.add_item(i);
                                    tabu_mark(i);
                                    frontier_push(i);
                                }
                            }
                        }

                        one_one_swap_phase(&mut state_int);
                    }

                    rebuild_windows(&mut state_int);
                    dp_refinement_x(&mut state_int, params.dp_passes_multiplier);
                    rebuild_windows(&mut state_int);
                    micro_qkp_refinement(&mut state_int);
                    one_two_exchange_phase(&mut state_int);
                    local_search_vnd(&mut state_int, params);

                    if state_int.total_value > best_val {
                        best_val = state_int.total_value;
                        best_sel.clear();
                        for i in 0..n {
                            if state_int.selected_bit[i] {
                                if state_int.usage[i] < u16::MAX {
                                    state_int.usage[i] += 1;
                                }
                                best_sel.push(i);
                            }
                        }
                        stall_int = 0;
                    }
                }
            } else if !applied_dp_this_round {
                dp_next_int = true;
            }
        }

        let freeze_div = !is_last_round
            && state_int.total_value >= state_div.total_value
            && stall_div > stall_int;

        if freeze_div {
            let strategy = perturbation_round % 7;
            let strength = params.perturbation_strength_base + (perturbation_round as usize) / 2;
            let tenure_u32 = strength as u32;
            let iter2_u32 = iter_u32.wrapping_add(stall_int as u32);

            seed_frontier_from_state(&state_int);

            let mut before = state_int.selected_items();
            before.sort_unstable();

            {
                let _g = TabuGuard::activate(
                    tabu_int.as_mut_ptr(),
                    tabu_int.len(),
                    iter2_u32,
                    tenure_u32,
                    best_val as i64,
                );
                perturb_by_strategy(&mut state_int, &mut rng, strength, stall_int, strategy);
            }

            for &i in &before {
                if !state_int.selected_bit[i] {
                    tabu_int[i] = iter2_u32;
                }
            }
            let after = state_int.selected_items();
            for i in after {
                if before.binary_search(&i).is_err() {
                    tabu_int[i] = iter2_u32;
                }
            }

            {
                let _g = TabuGuard::activate(
                    tabu_int.as_mut_ptr(),
                    tabu_int.len(),
                    iter2_u32,
                    tenure_u32,
                    best_val as i64,
                );
                greedy_reconstruct(&mut state_int, &mut rng, strategy);
                one_one_swap_phase(&mut state_int);
            }

            rebuild_windows(&mut state_int);
            dp_refinement_x(&mut state_int, params.dp_passes_multiplier);
            rebuild_windows(&mut state_int);
            micro_qkp_refinement(&mut state_int);
            one_two_exchange_phase(&mut state_int);
            local_search_vnd(&mut state_int, params);

            if state_int.total_value > best_val {
                best_val = state_int.total_value;
                best_sel.clear();
                for i in 0..n {
                    if state_int.selected_bit[i] {
                        if state_int.usage[i] < u16::MAX {
                            state_int.usage[i] += 1;
                        }
                        best_sel.push(i);
                    }
                }
                stall_int = 0;
            }
        } else {
            state_div.snap_bits.clone_from(&state_div.selected_bit);
            state_div.snap_contrib.clone_from(&state_div.contrib);
            state_div.snap_support.clone_from(&state_div.support);
            let prev_val = state_div.total_value;
            let prev_weight = state_div.total_weight;

            local_search_vnd(&mut state_div, params);

            if state_div.total_value > best_val {
                best_val = state_div.total_value;
                best_sel.clear();
                for i in 0..n {
                    if state_div.selected_bit[i] {
                        if state_div.usage[i] < u16::MAX {
                            state_div.usage[i] += 1;
                        }
                        best_sel.push(i);
                    }
                }
                stall_div = 0;
            }

            let mut accept_plateau = false;
            if state_div.total_value == prev_val {
                let mut cur_supp_sum: i64 = i64::default();
                let mut snap_supp_sum: i64 = i64::default();
                for i in 0..n {
                    if state_div.selected_bit[i] {
                        cur_supp_sum += state_div.support[i] as i64;
                    }
                    if state_div.snap_bits[i] {
                        snap_supp_sum += state_div.snap_support[i] as i64;
                    }
                }
                accept_plateau = cur_supp_sum > snap_supp_sum
                    || (cur_supp_sum == snap_supp_sum && state_div.total_weight < prev_weight);
            }

            if state_div.total_value < prev_val
                || (state_div.total_value == prev_val && !accept_plateau)
            {
                state_div.restore_snapshot(prev_val, prev_weight);

                if perturbation_round >= 7 && stall_div >= 8 {
                    let strategy = perturbation_round % 7;
                    let strength =
                        params.perturbation_strength_base + (perturbation_round as usize) / 2;
                    state_div = restart_from_mutated_best(
                        challenge,
                        &total_pre,
                        &hubs_static,
                        neigh_pre.as_ref(),
                        &best_sel,
                        &mut rng,
                        strategy,
                        strength,
                    );
                    seed_frontier_from_state(&state_div);
                    stall_div = 0;
                }

                if !is_last_round {
                    stall_div += 1;

                    let strategy = perturbation_round % 7;
                    let strength =
                        params.perturbation_strength_base + (perturbation_round as usize) / 2;
                    let tenure_u32 = strength as u32;

                    seed_frontier_from_state(&state_div);

                    let mut before = state_div.selected_items();
                    before.sort_unstable();

                    {
                        let _g = TabuGuard::activate(
                            tabu_div.as_mut_ptr(),
                            tabu_div.len(),
                            iter_u32,
                            tenure_u32,
                            best_val as i64,
                        );
                        perturb_by_strategy(&mut state_div, &mut rng, strength, stall_div, strategy);
                    }

                    for &i in &before {
                        if !state_div.selected_bit[i] {
                            tabu_div[i] = iter_u32;
                        }
                    }
                    let after = state_div.selected_items();
                    for i in after {
                        if before.binary_search(&i).is_err() {
                            tabu_div[i] = iter_u32;
                        }
                    }

                    {
                        let _g = TabuGuard::activate(
                            tabu_div.as_mut_ptr(),
                            tabu_div.len(),
                            iter_u32,
                            tenure_u32,
                            best_val as i64,
                        );
                        greedy_reconstruct(&mut state_div, &mut rng, strategy);
                    }
                    local_search_vnd(&mut state_div, params);

                    if state_div.total_value > best_val {
                        best_val = state_div.total_value;
                        best_sel.clear();
                        for i in 0..n {
                            if state_div.selected_bit[i] {
                                if state_div.usage[i] < u16::MAX {
                                    state_div.usage[i] += 1;
                                }
                                best_sel.push(i);
                            }
                        }
                        stall_div = 0;
                    }
                }
            }
        }

        if state_div.total_value > state_int.total_value {
            core::mem::swap(&mut state_int, &mut state_div);
            core::mem::swap(&mut stall_int, &mut stall_div);
            core::mem::swap(&mut tabu_int, &mut tabu_div);
            seed_frontier_from_state(&state_int);
            dp_next_int = true;
        }
    }

    Solution { items: best_sel }
}