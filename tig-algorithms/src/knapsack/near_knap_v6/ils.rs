use super::construct::{
    build_initial_solution, construct_forward_incremental, construct_frontier_cluster_grow,
    construct_pair_seed_beta, greedy_fill_with_beta,
};
use super::local_search::local_search_vnd;
use super::refinement::{dp_refinement_x, micro_qkp_refinement};
use super::types::{
    build_sparse_neighbors_and_totals, compute_total_interactions, rebuild_windows, Rng, State,
};
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;

#[derive(Clone, Copy)]
pub struct Params {
    pub n_perturbation_rounds: usize,
    pub perturbation_strength_base: usize,
    pub extra_starts: usize,
    pub max_frontier_swaps_override: Option<usize>,
    pub dp_passes_multiplier: usize,
}

impl Params {
    pub fn initialize(_h: &Option<Map<String, Value>>) -> Self {
        Self {
            n_perturbation_rounds: 15,
            perturbation_strength_base: 3,
            extra_starts: 0,
            max_frontier_swaps_override: None,
            dp_passes_multiplier: 1,
        }
    }
}

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

fn build_state_from_selection<'a>(
    challenge: &'a Challenge,
    total_pre: &'a [i64],
    hubs_static: &'a [usize],
    neigh: Option<&'a Vec<Vec<(u16, i16)>>>,
    sel: &[usize],
) -> State<'a> {
    let mut st = State::new_empty(challenge, total_pre, hubs_static, neigh);
    for &i in sel {
        if i < challenge.num_items
            && !st.selected_bit[i]
            && st.total_weight + challenge.weights[i] <= challenge.max_weight
        {
            st.add_item(i);
        }
    }
    st
}

#[inline]
fn better_state(a: &State, b: &State) -> bool {
    a.total_value > b.total_value || (a.total_value == b.total_value && a.total_weight < b.total_weight)
}

#[derive(Clone, Copy)]
struct OperatorBundle {
    strategy: usize,
    repair_hint: u8,
}

const OPERATOR_BUNDLES: [OperatorBundle; 7] = [
    OperatorBundle {
        strategy: 0,
        repair_hint: 2,
    },
    OperatorBundle {
        strategy: 1,
        repair_hint: 0,
    },
    OperatorBundle {
        strategy: 2,
        repair_hint: 1,
    },
    OperatorBundle {
        strategy: 3,
        repair_hint: 2,
    },
    OperatorBundle {
        strategy: 4,
        repair_hint: 1,
    },
    OperatorBundle {
        strategy: 5,
        repair_hint: 0,
    },
    OperatorBundle {
        strategy: 6,
        repair_hint: 2,
    },
];

fn select_operator_bundle(
    scores: &[f64],
    pulls: &[u32],
    elite_hits: &[u32],
    stall: usize,
    hard: bool,
    neigh_present: bool,
    n: usize,
    rng: &mut Rng,
) -> usize {
    let total_pulls: u32 = pulls.iter().copied().sum();

    if total_pulls < OPERATOR_BUNDLES.len() as u32 {
        let offset = (rng.next_u32() as usize) % OPERATOR_BUNDLES.len();
        for step in 0..OPERATOR_BUNDLES.len() {
            let idx = (offset + step) % OPERATOR_BUNDLES.len();
            if pulls[idx] == 0 {
                return idx;
            }
        }
    }

    let eps: u32 = if stall >= 6 {
        10
    } else if stall >= 3 {
        6
    } else {
        4
    };
    if (rng.next_u32() & 63) < eps {
        return (rng.next_u32() as usize) % OPERATOR_BUNDLES.len();
    }

    let log_total = ((total_pulls + 1) as f64).ln().max(1.0);
    let mut best_idx = 0usize;
    let mut best_score = f64::NEG_INFINITY;

    for (idx, bundle) in OPERATOR_BUNDLES.iter().enumerate() {
        let p = pulls[idx].max(1) as f64;
        let explore = (log_total / p).sqrt();
        let elite_bonus = (elite_hits[idx] as f64) / p;

        let mut bias = 0.0f64;
        if neigh_present {
            if bundle.repair_hint >= 2 && (hard || stall >= 3) {
                bias += 0.22;
            }
        } else if bundle.repair_hint >= 2 {
            bias -= 0.12;
        }
        if n >= 3000 && bundle.repair_hint == 0 {
            bias += 0.06;
        }
        if stall == 0 && bundle.repair_hint >= 2 {
            bias -= 0.08;
        }

        let score = scores[idx] + 0.75 * explore + 0.18 * elite_bonus + bias;
        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }

    best_idx
}

fn update_operator_bundle(
    scores: &mut [f64],
    pulls: &mut [u32],
    elite_hits: &mut [u32],
    idx: usize,
    gain: i64,
    reached_elite: bool,
    strength: usize,
) {
    let bundle = OPERATOR_BUNDLES[idx];
    pulls[idx] = pulls[idx].saturating_add(1);
    if reached_elite {
        elite_hits[idx] = elite_hits[idx].saturating_add(1);
    }

    let effort = 1.0 + strength as f64 + (bundle.repair_hint as f64) * 1.35;
    let clipped_gain = gain.max(-512).min(4096) as f64;
    let reward = clipped_gain / (40.0 * effort)
        + if gain > 0 {
            0.35
        } else if gain == 0 {
            0.05
        } else {
            -0.20
        }
        + if reached_elite { 0.80 } else { 0.0 };

    scores[idx] = if pulls[idx] <= 1 {
        reward
    } else {
        scores[idx] * 0.82 + reward * 0.18
    };
}

fn snapshot_change_count(state: &State) -> usize {
    let mut diff = 0usize;
    for i in 0..state.ch.num_items {
        if state.selected_bit[i] != state.snap_bits[i] {
            diff += 1;
        }
    }
    diff
}

fn should_run_heavy_refinement(
    state: &State,
    best_val: i64,
    stall: usize,
    repair_hint: usize,
    n: usize,
    drift: usize,
) -> bool {
    if n <= 1400 {
        return true;
    }

    let gap = (best_val - state.total_value as i64).max(0);
    if gap <= 32 || state.slack() <= 3 {
        return true;
    }

    let drift_cap = if n >= 3000 {
        8
    } else if n >= 1800 {
        12
    } else {
        16
    };
    if drift <= drift_cap {
        return true;
    }

    if repair_hint >= 2 && stall <= 2 && gap <= 96 {
        return true;
    }

    stall >= 6 && gap <= if n >= 2800 { 80 } else { 112 }
}

fn structured_budget_boost<'a>(
    challenge: &'a Challenge,
    total_pre: &'a [i64],
    hubs_static: &'a [usize],
    neigh: Option<&'a Vec<Vec<(u16, i16)>>>,
    state: &mut State<'a>,
    elite_sel: &[usize],
    rng: &mut Rng,
    strategy: usize,
    pressure: usize,
    max_frontier_swaps_override: Option<usize>,
) {
    let n = challenge.num_items;

    if pressure >= 3 {
        one_one_swap_phase(state);
    }
    if state.neigh.is_some() && state.slack() >= 2 && pressure >= 2 {
        pair_repair_phase(state, pressure.max(2));
    }

    if !elite_sel.is_empty() {
        let try_relink = pressure >= 3 && (n <= 2400 || (n <= 3200 && state.slack() <= 8));
        if try_relink {
            let mut cur_sel = state.selected_items();
            if cur_sel.len() >= 4 {
                cur_sel.sort_unstable();
                if let Some(cand) = path_relink_between(
                    challenge,
                    total_pre,
                    hubs_static,
                    neigh,
                    &cur_sel,
                    elite_sel,
                    rng,
                    strategy,
                    pressure.min(6),
                    max_frontier_swaps_override,
                ) {
                    if better_state(&cand, state) {
                        *state = cand;
                        seed_frontier_from_state(state);
                    }
                }
            }
        }
    }

    one_two_exchange_phase(state);
}

fn optimize_recent_block(state: &mut State, block: &[usize], passes: usize) {
    if block.is_empty() {
        return;
    }

    let cap = state.ch.max_weight;
    let pass_cap = passes.max(1);

    for _ in 0..pass_cap {
        let mut any_change = false;

        loop {
            let slack = cap - state.total_weight;
            if slack == 0 {
                break;
            }

            let mut best_single: Option<(usize, i64, i64, u32)> = None;
            let mut best_pair: Option<(usize, usize, i64, i64, u32)> = None;

            for &a in block {
                if state.selected_bit[a] {
                    continue;
                }
                let wa = state.ch.weights[a];
                if wa == 0 || wa > slack {
                    continue;
                }

                let delta = state.contrib[a] as i64;
                if delta <= 0 {
                    continue;
                }

                let weight_pen = if slack <= 4 {
                    24i64
                } else if slack <= 10 {
                    10i64
                } else {
                    4i64
                };
                let score = delta * 256
                    + (state.support[a] as i64) * 32
                    + state.total_interactions[a] / 8000
                    - (state.usage[a] as i64) * 10
                    - (wa as i64) * weight_pen;

                if best_single.map_or(true, |(_, best_score, best_delta, best_w)| {
                    score > best_score
                        || (score == best_score
                            && (delta > best_delta || (delta == best_delta && wa < best_w)))
                }) {
                    best_single = Some((a, score, delta, wa));
                }
            }

            if slack >= 2 {
                for ai in 0..block.len() {
                    let a = block[ai];
                    if state.selected_bit[a] {
                        continue;
                    }
                    let wa = state.ch.weights[a];
                    if wa == 0 || wa >= slack {
                        continue;
                    }

                    for bi in (ai + 1)..block.len() {
                        let b = block[bi];
                        if state.selected_bit[b] {
                            continue;
                        }
                        let wb = state.ch.weights[b];
                        let wsum = wa + wb;
                        if wb == 0 || wsum > slack {
                            continue;
                        }

                        let inter = state.ch.interaction_values[a][b] as i64;
                        let delta = state.contrib[a] as i64 + state.contrib[b] as i64 + inter;
                        if delta <= 0 {
                            continue;
                        }

                        let weight_pen = if slack <= 4 {
                            32i64
                        } else if slack <= 10 {
                            14i64
                        } else {
                            6i64
                        };
                        let score = delta * 256
                            + inter * 160
                            + (state.support[a] as i64 + state.support[b] as i64) * 24
                            + (state.total_interactions[a] + state.total_interactions[b]) / 8000
                            - (state.usage[a] as i64 + state.usage[b] as i64) * 10
                            - (wsum as i64) * weight_pen;

                        if best_pair.map_or(true, |(_, _, best_score, best_delta, best_wsum)| {
                            score > best_score
                                || (score == best_score
                                    && (delta > best_delta
                                        || (delta == best_delta && wsum < best_wsum)))
                        }) {
                            best_pair = Some((a, b, score, delta, wsum));
                        }
                    }
                }
            }

            let mut added = false;
            if let Some((a, b, pair_score, pair_delta, _)) = best_pair {
                let take_pair = best_single.map_or(true, |(_, single_score, single_delta, _)| {
                    pair_score > single_score + 64
                        || (pair_score == single_score && pair_delta > single_delta)
                });
                if take_pair {
                    let wa = state.ch.weights[a];
                    let wb = state.ch.weights[b];
                    if !state.selected_bit[a]
                        && !state.selected_bit[b]
                        && wa != 0
                        && wb != 0
                        && state.total_weight + wa + wb <= cap
                    {
                        state.add_item(a);
                        state.add_item(b);
                        frontier_push(a);
                        frontier_push(b);
                        any_change = true;
                        added = true;
                    }
                }
            }

            if !added {
                if let Some((a, _, _, _)) = best_single {
                    let wa = state.ch.weights[a];
                    if !state.selected_bit[a] && wa != 0 && state.total_weight + wa <= cap {
                        state.add_item(a);
                        frontier_push(a);
                        any_change = true;
                        added = true;
                    }
                }
            }

            if !added {
                break;
            }
        }

        let selected_block: Vec<usize> = block
            .iter()
            .copied()
            .filter(|&i| state.selected_bit[i])
            .collect();
        let mut exchange_change = false;

        'exchange: for r in selected_block {
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
            let mut best_mode = 0u8;
            let mut best_a = usize::MAX;
            let mut best_b = usize::MAX;
            let mut best_score = i64::MIN;
            let mut best_delta = 0i64;
            let mut best_wsum = u32::MAX;

            for &a in block {
                if state.selected_bit[a] {
                    continue;
                }
                let wa = state.ch.weights[a];
                if wa == 0 || wa > slack {
                    continue;
                }

                let delta = state.contrib[a] as i64 - loss;
                if delta <= 0 {
                    continue;
                }

                let weight_pen = if slack <= 4 { 20i64 } else { 6i64 };
                let score = delta * 256
                    + (state.support[a] as i64) * 24
                    - (state.usage[a] as i64) * 8
                    - (wa as i64) * weight_pen;

                if score > best_score
                    || (score == best_score
                        && (delta > best_delta || (delta == best_delta && wa < best_wsum)))
                {
                    best_mode = 1;
                    best_a = a;
                    best_b = usize::MAX;
                    best_score = score;
                    best_delta = delta;
                    best_wsum = wa;
                }
            }

            if slack >= 2 {
                for ai in 0..block.len() {
                    let a = block[ai];
                    if state.selected_bit[a] {
                        continue;
                    }
                    let wa = state.ch.weights[a];
                    if wa == 0 || wa >= slack {
                        continue;
                    }

                    for bi in (ai + 1)..block.len() {
                        let b = block[bi];
                        if state.selected_bit[b] {
                            continue;
                        }
                        let wb = state.ch.weights[b];
                        let wsum = wa + wb;
                        if wb == 0 || wsum > slack {
                            continue;
                        }

                        let inter = state.ch.interaction_values[a][b] as i64;
                        let delta =
                            state.contrib[a] as i64 + state.contrib[b] as i64 + inter - loss;
                        if delta <= 0 {
                            continue;
                        }

                        let weight_pen = if slack <= 4 { 28i64 } else { 8i64 };
                        let score = delta * 256
                            + inter * 160
                            + (state.support[a] as i64 + state.support[b] as i64) * 20
                            - (state.usage[a] as i64 + state.usage[b] as i64) * 8
                            - (wsum as i64) * weight_pen;

                        if score > best_score
                            || (score == best_score
                                && (delta > best_delta
                                    || (delta == best_delta && wsum < best_wsum)))
                        {
                            best_mode = 2;
                            best_a = a;
                            best_b = b;
                            best_score = score;
                            best_delta = delta;
                            best_wsum = wsum;
                        }
                    }
                }
            }

            let mut accepted = false;
            if best_mode == 1 && best_a != usize::MAX {
                let wa = state.ch.weights[best_a];
                if !state.selected_bit[best_a] && wa != 0 && state.total_weight + wa <= cap {
                    state.add_item(best_a);
                    frontier_push(r);
                    frontier_push(best_a);
                    accepted = true;
                }
            } else if best_mode == 2 && best_a != usize::MAX && best_b != usize::MAX {
                let wa = state.ch.weights[best_a];
                let wb = state.ch.weights[best_b];
                if !state.selected_bit[best_a]
                    && !state.selected_bit[best_b]
                    && wa != 0
                    && wb != 0
                    && state.total_weight + wa + wb <= cap
                {
                    state.add_item(best_a);
                    state.add_item(best_b);
                    frontier_push(r);
                    frontier_push(best_a);
                    frontier_push(best_b);
                    accepted = true;
                }
            }

            if accepted {
                any_change = true;
                exchange_change = true;
                break 'exchange;
            }

            if !state.selected_bit[r] && state.total_weight + wr <= cap {
                state.add_item(r);
            }
        }

        if !exchange_change && !any_change {
            break;
        }
    }
}

fn recent_disagreement_block_search<'a>(
    challenge: &'a Challenge,
    total_pre: &'a [i64],
    hubs_static: &'a [usize],
    neigh: Option<&'a Vec<Vec<(u16, i16)>>>,
    state: &mut State<'a>,
    elite_sel: &[usize],
    recent_marks: &[u32],
    iter_u32: u32,
    stall: usize,
) -> bool {
    let n = challenge.num_items;
    if elite_sel.is_empty() || stall < 5 || recent_marks.len() != n {
        return false;
    }

    let recent_window = if stall >= 8 {
        4u32
    } else if stall >= 6 {
        3u32
    } else {
        2u32
    };
    let block_cap = if n >= 3000 {
        18usize
    } else if n >= 1800 {
        24usize
    } else {
        30usize
    };

    let mut elite_bit = vec![false; n];
    for &i in elite_sel {
        if i < n {
            elite_bit[i] = true;
        }
    }

    let mut frontier = frontier_clone();
    frontier.sort_unstable();
    frontier.dedup();

    let mut scored: Vec<(i64, usize)> = Vec::new();
    let mut recent_cnt = 0usize;
    let mut disagree_cnt = 0usize;

    for i in 0..n {
        let stamp = recent_marks[i];
        let recent = stamp > 0 && iter_u32.saturating_sub(stamp) <= recent_window;
        let disagree = state.selected_bit[i] != elite_bit[i];
        if !recent && !disagree {
            continue;
        }

        if recent {
            recent_cnt += 1;
        }
        if disagree {
            disagree_cnt += 1;
        }

        let mut score = 0i64;
        if recent {
            score += 900;
        }
        if disagree {
            score += 1100;
        }
        if recent && disagree {
            score += 500;
        }
        score += i64::from(state.contrib[i]).abs().min(512);
        score += (state.support[i] as i64) * 24;
        score += state.total_interactions[i] / 9000;
        if frontier.binary_search(&i).is_ok() {
            score += 96;
        }

        if state.selected_bit[i] {
            if state.contrib[i] <= 0 {
                score += 160;
            }
            score += (challenge.weights[i] as i64).min(48);
        } else if state.contrib[i] > 0 {
            score += (state.contrib[i] as i64).min(256);
        }

        scored.push((score, i));
    }

    if recent_cnt < 2 || disagree_cnt < 3 || scored.len() < 6 {
        return false;
    }

    scored.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    if scored.len() > block_cap {
        scored.truncate(block_cap);
    }

    let mut block: Vec<usize> = scored.into_iter().map(|(_, i)| i).collect();
    block.sort_unstable();
    block.dedup();
    if block.len() < 6 {
        return false;
    }

    let mut in_block = vec![false; n];
    for &i in &block {
        in_block[i] = true;
    }

    let current_sel = state.selected_items();
    let mut frozen_sel: Vec<usize> = Vec::with_capacity(current_sel.len());
    for &i in &current_sel {
        if !in_block[i] {
            frozen_sel.push(i);
        }
    }

    let base = build_state_from_selection(challenge, total_pre, hubs_static, neigh, &frozen_sel);
    let base_slack = base.slack();
    let mut elite_ranked: Vec<(i64, usize)> = Vec::new();
    for &i in &block {
        if !elite_bit[i] {
            continue;
        }
        let w = challenge.weights[i];
        if w == 0 {
            continue;
        }
        let weight_pen = if base_slack <= 4 { 20i64 } else { 6i64 };
        let score = (base.contrib[i] as i64) * 256
            + (base.support[i] as i64) * 32
            + base.total_interactions[i] / 8000
            - (base.usage[i] as i64) * 10
            - (w as i64) * weight_pen;
        elite_ranked.push((score, i));
    }
    elite_ranked.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

    let mut elite_seed = frozen_sel.clone();
    for (_, i) in elite_ranked {
        elite_seed.push(i);
    }

    let pass_cap = if block.len() <= 14 {
        5
    } else if block.len() <= 20 {
        4
    } else {
        3
    };

    let mut best_candidate =
        build_state_from_selection(challenge, total_pre, hubs_static, neigh, &current_sel);
    optimize_recent_block(&mut best_candidate, &block, pass_cap);

    let mut cand = build_state_from_selection(challenge, total_pre, hubs_static, neigh, &elite_seed);
    optimize_recent_block(&mut cand, &block, pass_cap);
    if better_state(&cand, &best_candidate) {
        best_candidate = cand;
    }

    let mut cand = build_state_from_selection(challenge, total_pre, hubs_static, neigh, &frozen_sel);
    optimize_recent_block(&mut cand, &block, pass_cap);
    if better_state(&cand, &best_candidate) {
        best_candidate = cand;
    }

    if better_state(&best_candidate, state) {
        *state = best_candidate;
        frontier_reset(block);
        true
    } else {
        false
    }
}

fn path_relink_between<'a>(
    challenge: &'a Challenge,
    total_pre: &'a [i64],
    hubs_static: &'a [usize],
    neigh: Option<&'a Vec<Vec<(u16, i16)>>>,
    source_sel: &[usize],
    target_sel: &[usize],
    rng: &mut Rng,
    strategy: usize,
    pressure: usize,
    max_frontier_swaps_override: Option<usize>,
) -> Option<State<'a>> {
    let n = challenge.num_items;
    if source_sel.is_empty() || target_sel.is_empty() {
        return None;
    }

    let mut target_bit = vec![false; n];
    for &i in target_sel {
        if i < n {
            target_bit[i] = true;
        }
    }
    let mut source_bit = vec![false; n];
    for &i in source_sel {
        if i < n {
            source_bit[i] = true;
        }
    }

    let mut add_pool: Vec<usize> = Vec::new();
    for &i in target_sel {
        if i < n && !source_bit[i] {
            add_pool.push(i);
        }
    }
    add_pool.sort_unstable();
    add_pool.dedup();

    let mut drop_pool: Vec<usize> = Vec::new();
    for &i in source_sel {
        if i < n && !target_bit[i] {
            drop_pool.push(i);
        }
    }
    drop_pool.sort_unstable();
    drop_pool.dedup();

    let dist = add_pool.len() + drop_pool.len();
    let diff_cap = if n >= 3000 {
        48
    } else if n >= 1500 {
        64
    } else {
        80
    };
    if dist < 4 || dist > diff_cap {
        return None;
    }

    let mut st = build_state_from_selection(challenge, total_pre, hubs_static, neigh, source_sel);
    seed_frontier_from_state(&st);

    let add_cap = if n >= 3000 { 24 } else { 32 };
    let drop_cap = if n >= 3000 { 24 } else { 32 };

    if add_pool.len() > add_cap {
        let weight_pen = if st.slack() <= 6 { 16 } else { 6 };
        let mut scored: Vec<(i64, usize)> = Vec::with_capacity(add_pool.len());
        for &i in &add_pool {
            let score = (st.contrib[i] as i64) * 256
                + (st.support[i] as i64) * 32
                + st.total_interactions[i] / 7000
                - (st.usage[i] as i64) * 12
                - (challenge.weights[i] as i64) * weight_pen;
            scored.push((score, i));
        }
        scored.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        add_pool = scored.into_iter().take(add_cap).map(|(_, i)| i).collect();
        add_pool.sort_unstable();
    }

    if drop_pool.len() > drop_cap {
        let mut scored: Vec<(i64, usize)> = Vec::with_capacity(drop_pool.len());
        for &i in &drop_pool {
            let score = (st.contrib[i] as i64) * 256
                + (st.support[i] as i64) * 24
                + (st.usage[i] as i64) * 6
                - (challenge.weights[i] as i64) * if pressure >= 4 { 24 } else { 12 };
            scored.push((score, i));
        }
        scored.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        drop_pool = scored.into_iter().take(drop_cap).map(|(_, i)| i).collect();
        drop_pool.sort_unstable();
    }

    let step_cap = dist.min(if pressure >= 6 { 18 } else { 12 }).max(4);
    let eval_every = if pressure >= 6 { 2 } else { 3 };

    let baseline_val = st.total_value;
    let baseline_weight = st.total_weight;
    let mut best_mid: Option<Vec<usize>> = None;
    let mut best_mid_val = baseline_val;
    let mut best_mid_weight = baseline_weight;

    for step in 0..step_cap {
        let slack = st.slack();

        let mut best_add = usize::MAX;
        let mut best_add_score = i64::MIN;
        let mut best_add_feasible = false;
        let mut best_add_c = i64::MIN;

        for &a in &add_pool {
            if st.selected_bit[a] {
                continue;
            }
            let w = challenge.weights[a];
            let c = st.contrib[a] as i64;
            let feasible = st.total_weight + w <= challenge.max_weight;
            let mut score = c * 256
                + (st.support[a] as i64) * 32
                + st.total_interactions[a] / 8000
                - (st.usage[a] as i64) * 10
                - (w as i64) * if slack <= 4 { 20 } else { 6 };
            if feasible {
                score += 160;
            }
            if c > 0 {
                score += 96;
            }

            if best_add == usize::MAX
                || (feasible && !best_add_feasible)
                || (feasible == best_add_feasible
                    && (score > best_add_score || (score == best_add_score && c > best_add_c)))
            {
                best_add = a;
                best_add_score = score;
                best_add_feasible = feasible;
                best_add_c = c;
            }
        }

        let allow_feasible_swaps = pressure >= 5 || !best_add_feasible;
        let mut best_swap_r = usize::MAX;
        let mut best_swap_a = usize::MAX;
        let mut best_swap_score = i64::MIN;

        if !add_pool.is_empty() && !drop_pool.is_empty() {
            for &a in &add_pool {
                if st.selected_bit[a] {
                    continue;
                }
                let wa = challenge.weights[a];
                if !allow_feasible_swaps && st.total_weight + wa <= challenge.max_weight {
                    continue;
                }
                let ca = st.contrib[a] as i64;
                for &r in &drop_pool {
                    if !st.selected_bit[r] {
                        continue;
                    }
                    let wr = challenge.weights[r];
                    if st.total_weight - wr + wa > challenge.max_weight {
                        continue;
                    }

                    let score = (ca - st.contrib[r] as i64) * 256
                        + ((st.support[a] as i64) - (st.support[r] as i64)) * 24
                        + (wr as i64 - wa as i64) * 12
                        - (st.usage[a] as i64) * 8;

                    if score > best_swap_score
                        || (score == best_swap_score
                            && (r < best_swap_r || (r == best_swap_r && a < best_swap_a)))
                    {
                        best_swap_r = r;
                        best_swap_a = a;
                        best_swap_score = score;
                    }
                }
            }
        }

        let mut moved = false;

        if best_add != usize::MAX && best_add_feasible && best_add_score > -192 {
            let w = challenge.weights[best_add];
            if !st.selected_bit[best_add] && st.total_weight + w <= challenge.max_weight {
                st.add_item(best_add);
                frontier_push(best_add);
                moved = true;
            }
        }

        if !moved && best_swap_r != usize::MAX && best_swap_a != usize::MAX {
            if best_swap_score > -640 || step + 1 < step_cap / 2 + 1 {
                let wa = challenge.weights[best_swap_a];
                if st.selected_bit[best_swap_r]
                    && st.total_weight - challenge.weights[best_swap_r] + wa <= challenge.max_weight
                {
                    st.remove_item(best_swap_r);
                    frontier_push(best_swap_r);
                    if !st.selected_bit[best_swap_a] && st.total_weight + wa <= challenge.max_weight
                    {
                        st.add_item(best_swap_a);
                        frontier_push(best_swap_a);
                    }
                    moved = true;
                }
            }
        }

        if !moved && !drop_pool.is_empty() {
            let mut need_w = 0u32;
            let mut top_score = i64::MIN;
            for &a in &add_pool {
                if st.selected_bit[a] {
                    continue;
                }
                let w = challenge.weights[a];
                let score = (st.contrib[a] as i64) * 128 - (w as i64) * 4;
                if score > top_score {
                    top_score = score;
                    need_w = w.saturating_sub(slack);
                }
            }

            let mut best_drop = usize::MAX;
            let mut best_drop_score = i64::MAX;
            for &r in &drop_pool {
                if !st.selected_bit[r] {
                    continue;
                }
                let w = challenge.weights[r];
                let mut score = (st.contrib[r] as i64) * 256
                    + (st.support[r] as i64) * 32
                    + (st.usage[r] as i64) * 8
                    - (w as i64) * if need_w > 0 { 32 } else { 12 };
                if need_w > 0 && w >= need_w {
                    score -= 96;
                }

                if score < best_drop_score || (score == best_drop_score && r < best_drop) {
                    best_drop = r;
                    best_drop_score = score;
                }
            }

            if best_drop != usize::MAX {
                st.remove_item(best_drop);
                frontier_push(best_drop);
                moved = true;
            }
        }

        if !moved {
            break;
        }

        if (step + 1) % eval_every == 0 || step + 1 == step_cap {
            let cur_sel = st.selected_items();
            let mut cand = build_state_from_selection(challenge, total_pre, hubs_static, neigh, &cur_sel);
            seed_frontier_from_state(&cand);
            pair_repair_phase(&mut cand, pressure.max(2));
            greedy_reconstruct(&mut cand, rng, strategy);
            if pressure >= 4 {
                one_one_swap_phase(&mut cand);
            }

            if cand.total_value > best_mid_val
                || (cand.total_value == best_mid_val && cand.total_weight < best_mid_weight)
            {
                best_mid_val = cand.total_value;
                best_mid_weight = cand.total_weight;
                best_mid = Some(cand.selected_items());
            }
        }
    }

    let best_sel_mid = match best_mid {
        Some(v) => v,
        None => return None,
    };

    let mut out = build_state_from_selection(challenge, total_pre, hubs_static, neigh, &best_sel_mid);
    seed_frontier_from_state(&out);
    pair_repair_phase(&mut out, pressure.max(2));
    greedy_reconstruct(&mut out, rng, strategy);
    if pressure >= 4 {
        one_one_swap_phase(&mut out);
    }
    rebuild_windows(&mut out);
    if pressure >= 5 && n <= 1800 {
        micro_qkp_refinement(&mut out);
    }
    one_two_exchange_phase(&mut out);

    if n <= 2500 || pressure >= 6 {
        let ls_cap = if n >= 3000 {
            Some(max_frontier_swaps_override.unwrap_or(8).min(8))
        } else if n >= 1500 {
            Some(max_frontier_swaps_override.unwrap_or(12).min(12))
        } else {
            max_frontier_swaps_override
        };
        local_search_vnd(&mut out, ls_cap);
    }

    if out.total_value > baseline_val
        || (out.total_value == baseline_val && out.total_weight < baseline_weight)
    {
        Some(out)
    } else {
        None
    }
}

fn path_relink_bidirectional<'a>(
    challenge: &'a Challenge,
    total_pre: &'a [i64],
    hubs_static: &'a [usize],
    neigh: Option<&'a Vec<Vec<(u16, i16)>>>,
    a_sel: &[usize],
    b_sel: &[usize],
    rng: &mut Rng,
    strategy: usize,
    pressure: usize,
    max_frontier_swaps_override: Option<usize>,
) -> Option<State<'a>> {
    let ab = path_relink_between(
        challenge,
        total_pre,
        hubs_static,
        neigh,
        a_sel,
        b_sel,
        rng,
        strategy,
        pressure,
        max_frontier_swaps_override,
    );
    let ba = path_relink_between(
        challenge,
        total_pre,
        hubs_static,
        neigh,
        b_sel,
        a_sel,
        rng,
        strategy,
        pressure,
        max_frontier_swaps_override,
    );

    match (ab, ba) {
        (Some(x), Some(y)) => {
            if better_state(&x, &y) {
                Some(x)
            } else {
                Some(y)
            }
        }
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }
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
            let mut frontier_seed = removed.clone();
            if let Some(ng) = neigh {
                let row_cap = if challenge.num_items >= 3000 { 10 } else { 16 };
                'outer: for &r in &removed {
                    let row = unsafe { ng.get_unchecked(r) };
                    let pref = row.len().min(row_cap);
                    for t in 0..pref {
                        let j = row[t].0 as usize;
                        if j >= challenge.num_items || st.selected_bit[j] {
                            continue;
                        }
                        if frontier_seed.iter().any(|&x| x == j) {
                            continue;
                        }
                        frontier_seed.push(j);
                        if frontier_seed.len() >= 64 {
                            break 'outer;
                        }
                    }
                }
            }
            frontier_reset(frontier_seed);
        }
    }

    elite_repair_phase(&mut st, best_sel, strength.max(3));
    pair_repair_phase(&mut st, strength.max(2));
    greedy_reconstruct(&mut st, rng, strategy);
    one_two_exchange_phase(&mut st);
    rebuild_windows(&mut st);
    micro_qkp_refinement(&mut st);
    st
}

fn pair_repair_phase(state: &mut State, stall_count: usize) {
    if state.neigh.is_none() || state.slack() < 2 {
        return;
    }

    let n = state.ch.num_items;
    let frontier = frontier_clone();
    let seed_lim = if n >= 3000 { 20 } else { 28 };
    let hub_lim = if n >= 3000 { 16 } else { 24 };
    let row_cap = if n >= 3000 {
        if stall_count >= 4 { 28 } else { 20 }
    } else if stall_count >= 4 {
        48
    } else {
        32
    };
    let max_adds = if stall_count >= 4 { 3 } else { 2 };

    let mut seeds: Vec<usize> = Vec::with_capacity(seed_lim + hub_lim);
    let f_take = frontier.len().min(seed_lim);
    for &i in frontier.iter().rev().take(f_take) {
        if i < n && !state.selected_bit[i] {
            seeds.push(i);
        }
    }
    for &h in state.hubs_static.iter().take(hub_lim) {
        if !state.selected_bit[h] {
            seeds.push(h);
        }
    }
    seeds.sort_unstable();
    seeds.dedup();

    if seeds.is_empty() {
        return;
    }

    for _ in 0..max_adds {
        let slack = state.slack();
        if slack < 2 {
            break;
        }

        let mut best_a = usize::MAX;
        let mut best_b = usize::MAX;
        let mut best_score = i64::MIN;
        let mut best_delta = 0i64;
        let mut best_wsum = u32::MAX;

        {
            let ng = match state.neigh.as_ref() {
                Some(ng) => ng,
                None => return,
            };

            for &a in &seeds {
                if a >= n || state.selected_bit[a] {
                    continue;
                }
                let wa = state.ch.weights[a];
                if wa == 0 || wa >= slack {
                    continue;
                }
                if tabu_active() && tabu_recent(a) && !tabu_aspiration_ok(state, a) {
                    continue;
                }

                let row = unsafe { ng.get_unchecked(a) };
                let pref = row.len().min(row_cap);
                for t in 0..pref {
                    let b = row[t].0 as usize;
                    if b >= n || b == a || state.selected_bit[b] {
                        continue;
                    }

                    let wb = state.ch.weights[b];
                    if wb == 0 || wa + wb > slack {
                        continue;
                    }
                    if tabu_active() && tabu_recent(b) && !tabu_aspiration_ok(state, b) {
                        continue;
                    }

                    let inter = row[t].1 as i64;
                    if inter <= 0 {
                        continue;
                    }

                    let delta = state.contrib[a] as i64 + state.contrib[b] as i64 + inter;
                    if delta <= 0 {
                        continue;
                    }

                    let supp = state.support[a] as i64 + state.support[b] as i64;
                    let ti = state.total_interactions[a] + state.total_interactions[b];
                    let usage_pen = state.usage[a] as i64 + state.usage[b] as i64;
                    let wsum = wa + wb;
                    let weight_pen = if slack <= 6 {
                        48
                    } else if slack <= 12 {
                        20
                    } else {
                        8
                    };

                    let mut score = delta * 512 + inter * 160 + supp * 32 + ti / 4000;
                    score -= (wsum as i64) * weight_pen;
                    score -= usage_pen * 12;

                    if score > best_score
                        || (score == best_score
                            && (delta > best_delta
                                || (delta == best_delta && wsum < best_wsum)))
                    {
                        best_a = a;
                        best_b = b;
                        best_score = score;
                        best_delta = delta;
                        best_wsum = wsum;
                    }
                }
            }
        }

        if best_a == usize::MAX || best_b == usize::MAX {
            break;
        }

        let wa = state.ch.weights[best_a];
        let wb = state.ch.weights[best_b];
        if wa == 0
            || wb == 0
            || state.selected_bit[best_a]
            || state.selected_bit[best_b]
            || state.total_weight + wa + wb > state.ch.max_weight
        {
            break;
        }

        state.add_item(best_a);
        state.add_item(best_b);
        tabu_mark(best_a);
        tabu_mark(best_b);
        frontier_push(best_a);
        frontier_push(best_b);
    }
}

fn elite_repair_phase(state: &mut State, elite: &[usize], stall_count: usize) {
    if elite.is_empty() || stall_count < 3 {
        return;
    }

    let n = state.ch.num_items;
    let mut frontier = frontier_clone();
    frontier.sort_unstable();
    frontier.dedup();

    let pool_cap = if n >= 3000 { 24 } else { 36 };
    let pair_row_cap = if n >= 3000 {
        if stall_count >= 6 { 24 } else { 16 }
    } else if stall_count >= 6 {
        40
    } else {
        28
    };
    let max_steps = if stall_count >= 6 { 3 } else { 2 };

    let slack0 = state.slack();
    let mut scored: Vec<(i64, usize)> = Vec::with_capacity(elite.len().min(pool_cap * 2));
    for &i in elite {
        if i >= n || state.selected_bit[i] {
            continue;
        }
        let w = state.ch.weights[i];
        if w == 0 || w > state.ch.max_weight {
            continue;
        }
        let c = state.contrib[i] as i64;
        if c <= 0 && stall_count < 5 {
            continue;
        }
        let supp = state.support[i] as i64;
        let frontier_bonus = if frontier.binary_search(&i).is_ok() { 320 } else { 0 };
        let weight_pen = if slack0 <= 6 { 16 } else { 4 };
        let score = c * 96
            + supp * 24
            + state.total_interactions[i] / 5000
            + frontier_bonus
            - (state.usage[i] as i64) * 12
            - (w as i64) * weight_pen;
        scored.push((score, i));
    }
    if scored.is_empty() {
        return;
    }
    scored.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

    let mut pool: Vec<usize> = scored.into_iter().take(pool_cap).map(|(_, i)| i).collect();
    pool.sort_unstable();
    pool.dedup();

    for _ in 0..max_steps {
        let slack = state.slack();
        if slack == 0 {
            break;
        }

        let mut best_a = usize::MAX;
        let mut best_b = usize::MAX;
        let mut best_pair_score = i64::MIN;
        let mut best_delta = 0i64;
        let mut best_wsum = u32::MAX;

        if slack >= 2 {
            if let Some(ref ng) = state.neigh {
                for &a in &pool {
                    if state.selected_bit[a] {
                        continue;
                    }
                    let wa = state.ch.weights[a];
                    if wa == 0 || wa >= slack {
                        continue;
                    }
                    if tabu_active() && tabu_recent(a) && !tabu_aspiration_ok(state, a) {
                        continue;
                    }

                    let row = unsafe { ng.get_unchecked(a) };
                    let pref = row.len().min(pair_row_cap);
                    for t in 0..pref {
                        let b = row[t].0 as usize;
                        if b == a || pool.binary_search(&b).is_err() || state.selected_bit[b] {
                            continue;
                        }

                        let wb = state.ch.weights[b];
                        let wsum = wa + wb;
                        if wb == 0 || wsum > slack {
                            continue;
                        }
                        if tabu_active() && tabu_recent(b) && !tabu_aspiration_ok(state, b) {
                            continue;
                        }

                        let inter = row[t].1 as i64;
                        if inter <= 0 {
                            continue;
                        }

                        let delta = state.contrib[a] as i64 + state.contrib[b] as i64 + inter;
                        if delta <= 0 {
                            continue;
                        }

                        let supp = state.support[a] as i64 + state.support[b] as i64;
                        let usage_pen = state.usage[a] as i64 + state.usage[b] as i64;
                        let frontier_bonus = if frontier.binary_search(&a).is_ok()
                            || frontier.binary_search(&b).is_ok()
                        {
                            96
                        } else {
                            0
                        };
                        let weight_pen = if slack <= 4 {
                            56
                        } else if slack <= 10 {
                            20
                        } else {
                            8
                        };

                        let score = delta * 512
                            + inter * 192
                            + supp * 24
                            + frontier_bonus
                            - (wsum as i64) * weight_pen
                            - usage_pen * 12;

                        if score > best_pair_score
                            || (score == best_pair_score
                                && (delta > best_delta
                                    || (delta == best_delta && wsum < best_wsum)))
                        {
                            best_a = a;
                            best_b = b;
                            best_pair_score = score;
                            best_delta = delta;
                            best_wsum = wsum;
                        }
                    }
                }
            }
        }

        if best_a != usize::MAX && best_b != usize::MAX {
            let wa = state.ch.weights[best_a];
            let wb = state.ch.weights[best_b];
            if wa != 0
                && wb != 0
                && !state.selected_bit[best_a]
                && !state.selected_bit[best_b]
                && state.total_weight + wa + wb <= state.ch.max_weight
            {
                state.add_item(best_a);
                state.add_item(best_b);
                tabu_mark(best_a);
                tabu_mark(best_b);
                frontier_push(best_a);
                frontier_push(best_b);
                continue;
            }
        }

        let mut best_i: Option<usize> = None;
        let mut best_score = i64::MIN;
        let mut best_c = i32::MIN;
        let mut best_w = u32::MAX;

        for &i in &pool {
            if state.selected_bit[i] {
                continue;
            }
            let w = state.ch.weights[i];
            if w == 0 || w > slack {
                continue;
            }
            if tabu_active() && tabu_recent(i) && !tabu_aspiration_ok(state, i) {
                continue;
            }

            let c = state.contrib[i];
            if c <= 0 {
                continue;
            }

            let frontier_bonus = if frontier.binary_search(&i).is_ok() { 96 } else { 0 };
            let score = (c as i64) * 256
                + (state.support[i] as i64) * 32
                + frontier_bonus
                + state.total_interactions[i] / 8000
                - (state.usage[i] as i64) * 10
                - (w as i64) * if slack <= 4 { 24 } else { 8 };

            if best_i.map_or(true, |bi| {
                score > best_score
                    || (score == best_score && (c > best_c || (c == best_c && w < best_w)))
                    || (score == best_score && c == best_c && w == best_w && i < bi)
            }) {
                best_i = Some(i);
                best_score = score;
                best_c = c;
                best_w = w;
            }
        }

        if let Some(i) = best_i {
            state.add_item(i);
            tabu_mark(i);
            frontier_push(i);
        } else {
            break;
        }
    }
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

        let mut frontier: Vec<usize> = frontier_clone();
        if frontier.len() > 64 {
            let drop = frontier.len() - 64;
            frontier.drain(0..drop);
        }
        frontier.sort_unstable();
        frontier.dedup();

        let mut seeds: Vec<usize> = frontier.clone();
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
                let wa = state.ch.weights[a0];
                if wa == 0 {
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
                    let wb = state.ch.weights[b0];
                    if wb == 0 {
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
        let dense_sel = selected.len() * 3 >= n;

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
            let mut best_score: i64 = i64::MIN;
            let mut best_wsum: u32 = u32::MAX;

            for &(a, b, inter, wsum) in &pairs {
                if wsum > slack {
                    continue;
                }
                if state.selected_bit[a] || state.selected_bit[b] {
                    continue;
                }

                let delta =
                    (state.contrib[a] as i64) + (state.contrib[b] as i64) + (inter as i64) - loss;
                if delta <= 0 {
                    continue;
                }

                let supp = state.support[a] as i64 + state.support[b] as i64;
                let usage_pen = state.usage[a] as i64 + state.usage[b] as i64;
                let frontier_bonus: i64 =
                    if frontier.binary_search(&a).is_ok() || frontier.binary_search(&b).is_ok() {
                        96
                    } else {
                        0
                    };
                let weight_pen: i64 = if slack <= 3 {
                    84
                } else if slack <= 8 || dense_sel {
                    28
                } else {
                    10
                };

                let score = delta * 512
                    + (inter as i64) * 160
                    + supp * 28
                    + frontier_bonus
                    - (wsum as i64) * weight_pen
                    - usage_pen * 10;

                if score > best_score
                    || (score == best_score
                        && (delta > best_delta
                            || (delta == best_delta && wsum < best_wsum)))
                {
                    best_score = score;
                    best_delta = delta;
                    best_a = a;
                    best_b = b;
                    best_wsum = wsum;
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
                    tabu_mark(best_a);
                    tabu_mark(best_b);
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
        seed_frontier_from_state(&st);
        one_two_exchange_phase(&mut st);
        local_search_vnd(&mut st, params.max_frontier_swaps_override);

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
            seed_frontier_from_state(&hyb);
            one_two_exchange_phase(&mut hyb);
            local_search_vnd(&mut hyb, params.max_frontier_swaps_override);

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
            seed_frontier_from_state(&hyb);
            one_two_exchange_phase(&mut hyb);
            local_search_vnd(&mut hyb, params.max_frontier_swaps_override);

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
    let mut relink_cooldown = 0usize;
    let mut bandit_scores = [0.0f64; OPERATOR_BUNDLES.len()];
    let mut bandit_pulls = [0u32; OPERATOR_BUNDLES.len()];
    let mut bandit_elite_hits = [0u32; OPERATOR_BUNDLES.len()];

    for perturbation_round in 0..max_rounds {
        if relink_cooldown > 0 {
            relink_cooldown -= 1;
        }
        let round_start_best = best_val;
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
            local_search_vnd(&mut state_int, params.max_frontier_swaps_override);

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

                let rescued = if perturbation_round >= 5 && stall_int >= 5 {
                    recent_disagreement_block_search(
                        challenge,
                        &total_pre,
                        &hubs_static,
                        neigh_pre.as_ref(),
                        &mut state_int,
                        &best_sel,
                        &tabu_int,
                        iter_u32,
                        stall_int,
                    )
                } else {
                    false
                };

                if rescued {
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
                    } else {
                        stall_int = stall_int.saturating_sub(2);
                    }
                    dp_next_int = true;
                } else if perturbation_round >= 7 && stall_int >= 8 {
                    let restart_idx = select_operator_bundle(
                        &bandit_scores,
                        &bandit_pulls,
                        &bandit_elite_hits,
                        stall_int + 2,
                        hard,
                        neigh_pre.is_some(),
                        n,
                        &mut rng,
                    );
                    let restart_bundle = OPERATOR_BUNDLES[restart_idx];
                    let strength =
                        params.perturbation_strength_base + (perturbation_round as usize) / 2;
                    state_int = restart_from_mutated_best(
                        challenge,
                        &total_pre,
                        &hubs_static,
                        neigh_pre.as_ref(),
                        &best_sel,
                        &mut rng,
                        restart_bundle.strategy,
                        strength,
                    );
                    seed_frontier_from_state(&state_int);
                    stall_int = 0;
                }

                if !is_last_round {
                    stall_int += 1;

                    let start_val = state_int.total_value;
                    let best_before_bundle = best_val;
                    let bundle_idx = select_operator_bundle(
                        &bandit_scores,
                        &bandit_pulls,
                        &bandit_elite_hits,
                        stall_int,
                        hard,
                        neigh_pre.is_some(),
                        n,
                        &mut rng,
                    );
                    let bundle = OPERATOR_BUNDLES[bundle_idx];
                    let strategy = bundle.strategy;
                    let repair_hint = bundle.repair_hint as usize;
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
                        let repair_pressure = stall_int + repair_hint;
                        if repair_hint > 0 || stall_int >= 3 {
                            elite_repair_phase(
                                &mut state_int,
                                &best_sel,
                                repair_pressure.max(3),
                            );
                        }
                        pair_repair_phase(&mut state_int, repair_pressure.max(1));
                        greedy_reconstruct(&mut state_int, &mut rng, strategy);

                        if repair_hint > 0 || stall_int >= 2 {
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

                        if repair_hint > 0 {
                            one_one_swap_phase(&mut state_int);
                        }
                    }

                    let drift = snapshot_change_count(&state_int);
                    let do_heavy = should_run_heavy_refinement(
                        &state_int,
                        best_val as i64,
                        stall_int,
                        repair_hint,
                        n,
                        drift,
                    );
                    if do_heavy {
                        rebuild_windows(&mut state_int);
                        if repair_hint > 0 || n <= 2200 || stall_int >= 3 || drift <= 10 {
                            dp_refinement_x(&mut state_int, params.dp_passes_multiplier);
                            rebuild_windows(&mut state_int);
                        }
                        if repair_hint > 0
                            || n <= 1800
                            || stall_int >= 2
                            || state_int.slack() <= 4
                        {
                            micro_qkp_refinement(&mut state_int);
                        }
                        one_two_exchange_phase(&mut state_int);
                    } else {
                        structured_budget_boost(
                            challenge,
                            &total_pre,
                            &hubs_static,
                            neigh_pre.as_ref(),
                            &mut state_int,
                            &best_sel,
                            &mut rng,
                            strategy,
                            stall_int + repair_hint + 1,
                            params.max_frontier_swaps_override,
                        );
                    }
                    local_search_vnd(&mut state_int, params.max_frontier_swaps_override);

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
                    let reached_elite = best_val > best_before_bundle;
                    update_operator_bundle(
                        &mut bandit_scores,
                        &mut bandit_pulls,
                        &mut bandit_elite_hits,
                        bundle_idx,
                        (state_int.total_value - start_val) as i64,
                        reached_elite,
                        strength + repair_hint,
                    );
                }
            } else if !applied_dp_this_round {
                dp_next_int = true;
            }
        }

        let freeze_div = !is_last_round
            && state_int.total_value >= state_div.total_value
            && stall_div > stall_int;

        if freeze_div {
            let start_val = state_int.total_value;
            let best_before_bundle = best_val;
            let bundle_idx = select_operator_bundle(
                &bandit_scores,
                &bandit_pulls,
                &bandit_elite_hits,
                stall_int.max(1),
                hard,
                neigh_pre.is_some(),
                n,
                &mut rng,
            );
            let bundle = OPERATOR_BUNDLES[bundle_idx];
            let strategy = bundle.strategy;
            let repair_hint = bundle.repair_hint as usize;
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
                let repair_pressure = stall_int + repair_hint;
                if repair_hint > 0 || stall_int >= 3 {
                    elite_repair_phase(
                        &mut state_int,
                        &best_sel,
                        repair_pressure.max(3),
                    );
                }
                pair_repair_phase(&mut state_int, repair_pressure.max(1));
                greedy_reconstruct(&mut state_int, &mut rng, strategy);
                if repair_hint > 0 || stall_int >= 2 {
                    one_one_swap_phase(&mut state_int);
                }
            }

            let drift = snapshot_change_count(&state_int);
            let do_heavy = should_run_heavy_refinement(
                &state_int,
                best_val as i64,
                stall_int,
                repair_hint,
                n,
                drift,
            );
            if do_heavy {
                rebuild_windows(&mut state_int);
                if repair_hint > 0 || n <= 2200 || stall_int >= 3 || drift <= 10 {
                    dp_refinement_x(&mut state_int, params.dp_passes_multiplier);
                    rebuild_windows(&mut state_int);
                }
                if repair_hint > 0 || n <= 1800 || stall_int >= 2 || state_int.slack() <= 4 {
                    micro_qkp_refinement(&mut state_int);
                }
                one_two_exchange_phase(&mut state_int);
            } else {
                structured_budget_boost(
                    challenge,
                    &total_pre,
                    &hubs_static,
                    neigh_pre.as_ref(),
                    &mut state_int,
                    &best_sel,
                    &mut rng,
                    strategy,
                    stall_int + repair_hint + 1,
                    params.max_frontier_swaps_override,
                );
            }
            local_search_vnd(&mut state_int, params.max_frontier_swaps_override);

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
            let reached_elite = best_val > best_before_bundle;
            update_operator_bundle(
                &mut bandit_scores,
                &mut bandit_pulls,
                &mut bandit_elite_hits,
                bundle_idx,
                (state_int.total_value - start_val) as i64,
                reached_elite,
                strength + repair_hint,
            );
        } else {
            state_div.snap_bits.clone_from(&state_div.selected_bit);
            state_div.snap_contrib.clone_from(&state_div.contrib);
            state_div.snap_support.clone_from(&state_div.support);
            let prev_val = state_div.total_value;
            let prev_weight = state_div.total_weight;

            local_search_vnd(&mut state_div, params.max_frontier_swaps_override);

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

                let rescued = if perturbation_round >= 5 && stall_div >= 5 {
                    recent_disagreement_block_search(
                        challenge,
                        &total_pre,
                        &hubs_static,
                        neigh_pre.as_ref(),
                        &mut state_div,
                        &best_sel,
                        &tabu_div,
                        iter_u32,
                        stall_div,
                    )
                } else {
                    false
                };

                if rescued {
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
                    } else {
                        stall_div = stall_div.saturating_sub(2);
                    }
                } else if perturbation_round >= 7 && stall_div >= 8 {
                    let restart_idx = select_operator_bundle(
                        &bandit_scores,
                        &bandit_pulls,
                        &bandit_elite_hits,
                        stall_div + 2,
                        hard,
                        neigh_pre.is_some(),
                        n,
                        &mut rng,
                    );
                    let restart_bundle = OPERATOR_BUNDLES[restart_idx];
                    let strength =
                        params.perturbation_strength_base + (perturbation_round as usize) / 2;
                    state_div = restart_from_mutated_best(
                        challenge,
                        &total_pre,
                        &hubs_static,
                        neigh_pre.as_ref(),
                        &best_sel,
                        &mut rng,
                        restart_bundle.strategy,
                        strength,
                    );
                    seed_frontier_from_state(&state_div);
                    stall_div = 0;
                }

                if !is_last_round {
                    stall_div += 1;

                    let start_val = state_div.total_value;
                    let best_before_bundle = best_val;
                    let bundle_idx = select_operator_bundle(
                        &bandit_scores,
                        &bandit_pulls,
                        &bandit_elite_hits,
                        stall_div,
                        hard,
                        neigh_pre.is_some(),
                        n,
                        &mut rng,
                    );
                    let bundle = OPERATOR_BUNDLES[bundle_idx];
                    let strategy = bundle.strategy;
                    let repair_hint = bundle.repair_hint as usize;
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
                        let repair_pressure = stall_div + repair_hint;
                        if repair_hint > 0 || stall_div >= 3 {
                            elite_repair_phase(
                                &mut state_div,
                                &best_sel,
                                repair_pressure.max(3),
                            );
                        }
                        pair_repair_phase(&mut state_div, repair_pressure.max(1));
                        greedy_reconstruct(&mut state_div, &mut rng, strategy);
                        if repair_hint > 0 || stall_div >= 2 {
                            one_one_swap_phase(&mut state_div);
                        }
                        if repair_hint >= 2
                            || (stall_div >= 4 && (n <= 2800 || state_div.slack() <= 6))
                        {
                            one_two_exchange_phase(&mut state_div);
                        }
                    }
                    let drift = snapshot_change_count(&state_div);
                    let do_heavy = should_run_heavy_refinement(
                        &state_div,
                        best_val as i64,
                        stall_div,
                        repair_hint,
                        n,
                        drift,
                    );
                    if do_heavy {
                        if (repair_hint > 0
                            && (stall_div >= 3 || n <= 2200 || state_div.slack() <= 6))
                            || (stall_div >= 5 && n <= 1800)
                        {
                            rebuild_windows(&mut state_div);
                            micro_qkp_refinement(&mut state_div);
                        }
                    } else {
                        structured_budget_boost(
                            challenge,
                            &total_pre,
                            &hubs_static,
                            neigh_pre.as_ref(),
                            &mut state_div,
                            &best_sel,
                            &mut rng,
                            strategy,
                            stall_div + repair_hint + 1,
                            params.max_frontier_swaps_override,
                        );
                    }
                    local_search_vnd(&mut state_div, params.max_frontier_swaps_override);

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
                    let reached_elite = best_val > best_before_bundle;
                    update_operator_bundle(
                        &mut bandit_scores,
                        &mut bandit_pulls,
                        &mut bandit_elite_hits,
                        bundle_idx,
                        (state_div.total_value - start_val) as i64,
                        reached_elite,
                        strength + repair_hint,
                    );
                }
            }
        }

        if !is_last_round
            && relink_cooldown == 0
            && ((stall_int >= 3 || stall_div >= 3) || best_val > round_start_best)
        {
            let relink_idx = select_operator_bundle(
                &bandit_scores,
                &bandit_pulls,
                &bandit_elite_hits,
                stall_int.max(stall_div).max(1),
                hard,
                neigh_pre.is_some(),
                n,
                &mut rng,
            );
            let strategy = OPERATOR_BUNDLES[relink_idx].strategy;
            let pressure = stall_int.max(stall_div) + if best_val > round_start_best { 2 } else { 0 };

            let int_sel = state_int.selected_items();
            let div_sel = state_div.selected_items();

            let mut relink_best = path_relink_bidirectional(
                challenge,
                &total_pre,
                &hubs_static,
                neigh_pre.as_ref(),
                &int_sel,
                &div_sel,
                &mut rng,
                strategy,
                pressure.max(2),
                params.max_frontier_swaps_override,
            );

            if stall_div >= 2 || best_val > round_start_best {
                if let Some(s) = path_relink_bidirectional(
                    challenge,
                    &total_pre,
                    &hubs_static,
                    neigh_pre.as_ref(),
                    &div_sel,
                    &best_sel,
                    &mut rng,
                    strategy,
                    pressure.max(3),
                    params.max_frontier_swaps_override,
                ) {
                    let take = relink_best.as_ref().map_or(true, |b| better_state(&s, b));
                    if take {
                        relink_best = Some(s);
                    }
                }
            }

            if stall_int >= 4 && state_int.total_value < best_val {
                if let Some(s) = path_relink_bidirectional(
                    challenge,
                    &total_pre,
                    &hubs_static,
                    neigh_pre.as_ref(),
                    &int_sel,
                    &best_sel,
                    &mut rng,
                    strategy,
                    pressure.max(3),
                    params.max_frontier_swaps_override,
                ) {
                    let take = relink_best.as_ref().map_or(true, |b| better_state(&s, b));
                    if take {
                        relink_best = Some(s);
                    }
                }
            }

            if let Some(s) = relink_best {
                let mut accepted = false;

                if better_state(&s, &state_int) {
                    let old_int = core::mem::replace(&mut state_int, s);
                    if better_state(&old_int, &state_div) {
                        state_div = old_int;
                    }
                    seed_frontier_from_state(&state_int);
                    stall_int = 0;
                    accepted = true;
                } else if better_state(&s, &state_div) {
                    state_div = s;
                    stall_div = 0;
                    accepted = true;
                }

                if accepted {
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
                    }
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
                    }
                    relink_cooldown = if n >= 2500 { 3 } else { 2 };
                } else {
                    relink_cooldown = 1;
                }
            } else if stall_int + stall_div >= 6 {
                relink_cooldown = 1;
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