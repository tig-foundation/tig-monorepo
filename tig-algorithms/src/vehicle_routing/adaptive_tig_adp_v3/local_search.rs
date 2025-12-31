// Optimized Lightweight Local Search with time limit + neighborhood pruning
use crate::utilities::IZS;
use crate::tig_adaptive::TIGState;
use std::time::Instant;

pub mod controller {
use crate::delta::DeltaTables;
use crate::tig_adaptive::TIGState;
use crate::adp::vfa::VFA;
use rand::prelude::thread_rng;
use std::time::Instant;

/// Helper: simulate arrival times on a candidate node ordering without mutating state.
pub fn simulate_time_feasible(nodes: &[usize], state: &TIGState, start_idx: usize) -> bool {
    if nodes.is_empty() { return true; }

    let n = nodes.len();

    // compute base time: arrival at start_idx-1
    let mut current_time = if start_idx == 0 {
        state.time
    } else if start_idx - 1 < state.arrival_times.len() {
        state.arrival_times[start_idx - 1]
    } else {
        // compute from scratch up to start_idx-1
        let mut t = state.time;
        for k in 0..start_idx {
            if k > 0 {
                let prev = nodes[k - 1];
                let cur = nodes[k];
                t += state.travel_time(prev, cur) + state.service[prev];
            }
            if t < state.tw_start[nodes[k]] {
                t = state.tw_start[nodes[k]];
            }
        }
        t
    };

    for i in start_idx..n {
        if i > 0 {
            let prev = nodes[i - 1];
            let cur = nodes[i];
            current_time += state.travel_time(prev, cur) + state.service[prev];
        }
        if current_time < state.tw_start[nodes[i]] {
            current_time = state.tw_start[nodes[i]];
        }
        if current_time > state.tw_end[nodes[i]] {
            return false;
        }
    }
    true
}

/// Top level local search controller that runs multiple operators until no improvement
pub fn local_search(state: &mut TIGState, tables: &DeltaTables) {
    // Safety: add time-based and consecutive-no-improvement caps to avoid pathological loops.
    let max_iters: usize = 1000;

    let time_limit_ms: u64 = 50;

    let max_no_improve: usize = 5;

    // verbose diagnostics
    let verbose: bool = false;

    // VFA-driven ordering: create a local VFA that holds DLT and can be updated
    let mut rng = thread_rng();
    let mut vfa = VFA::new(0.1, 0.95, 0.1);

    let start = Instant::now();
    let time_limit = std::time::Duration::from_millis(time_limit_ms);

    let mut improved = true;
    let mut iter: usize = 0;
    let mut consecutive_no_improve: usize = 0;
    while improved && iter < max_iters {
        // time check at iteration start
        if start.elapsed() > time_limit {
            if verbose { eprintln!("local_search: reached time limit ({} ms). Stopping.", time_limit_ms); }
            break;
        }
        iter += 1;
        improved = false;

        // Compute operator potential scores using precomputed delta tables as a cheap proxy
        // Lower (more negative) delta means higher potential improvement.
        let mut op_scores: Vec<(&str, f64)> = Vec::new();

        // relocate: find minimal delta_relocate entry
        let mut min_reloc = i32::MAX;
        for row in &tables.delta_relocate {
            for &v in row {
                if v < min_reloc { min_reloc = v; }
            }
        }
        op_scores.push(("relocate", - (min_reloc as f64)));

        // swap
        let mut min_swap = i32::MAX;
        for row in &tables.delta_swap {
            for &v in row {
                if v < min_swap { min_swap = v; }
            }
        }
        op_scores.push(("swap", - (min_swap as f64)));

        // two_opt
        let mut min_two = i32::MAX;
        for row in &tables.delta_two_opt {
            for &v in row {
                if v < min_two { min_two = v; }
            }
        }
        op_scores.push(("two_opt", - (min_two as f64)));

        // two_opt_star: approximate using two_opt potential
        op_scores.push(("two_opt_star", - (min_two as f64)));

        // or_opt: no precomputed table — use small constant baseline
        op_scores.push(("or_opt", 0.0));

        // ejection_chain: prefer when relocate potential exists
        op_scores.push(("ejection_chain", - (min_reloc as f64) * 0.5));
        // time-window repair: give it high priority when there's a violation
        let has_violation = state.arrival_times.iter().enumerate().any(|(i,&t)| {
            let node = state.route.nodes.get(i).copied().unwrap_or(usize::MAX);
            node != usize::MAX && t > state.tw_end[node]
        });
        if has_violation { op_scores.push(("time_repair", 1e6)); } else { op_scores.push(("time_repair", 0.0)); }

        // bias using VFA estimate (small weight)
        let vfa_val = vfa.estimate(state, &mut rng);
        for entry in op_scores.iter_mut() {
            entry.1 += 0.01 * vfa_val;
        }

        // sort operators by descending score
        op_scores.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Execute operators in computed order
        for &(name, _) in &op_scores {
            // check time before invoking potentially expensive operator
            if start.elapsed() > time_limit {
                if verbose { eprintln!("local_search: reached time limit ({} ms) during operators. Stopping.", time_limit_ms); }
                break;
            }

            let op_improved = match name {
                "relocate" => crate::local_search::relocate::try_relocate(state, tables),
                "swap" => crate::local_search::swap::try_swap(state, tables),
                "two_opt" => crate::local_search::two_opt::try_two_opt(state, tables),
                "two_opt_star" => crate::local_search::two_opt_star::try_two_opt_star_multi(&mut [state.clone()], tables),
                "ejection_chain" => crate::local_search::ejection_chain::try_ejection_chain(state, tables),
                "time_repair" => crate::local_search::time_window_repair::try_time_window_repair(state, tables),
                "or_opt" => crate::local_search::or_opt::try_or_opt(state, tables),
                _ => false,
            };

            if op_improved {
                improved = true;
                consecutive_no_improve = 0;
                // reinforce arcs in DLT for the improved state
                for w in state.route.nodes.windows(2) {
                    if w.len() == 2 { vfa.update_dlt(state, 1.0, w[0], w[1]); }
                }
                // break to recompute operator ordering after improvement
                break;
            }
        }

        if !improved {
            consecutive_no_improve += 1;
            if consecutive_no_improve >= max_no_improve {
                if verbose { eprintln!("local_search: reached {} consecutive no-improve iterations. Stopping.", max_no_improve); }
                break;
            }
        }
    }

    if iter >= max_iters {
        if verbose { eprintln!("local_search: reached max iterations ({}). Stopping to avoid hang.", max_iters); }
    }
}

/// Delegate functions to operator modules for external use
pub fn try_relocate(state: &mut TIGState, tables: &DeltaTables) -> bool { crate::local_search::relocate::try_relocate(state, tables) }
pub fn try_swap(state: &mut TIGState, tables: &DeltaTables) -> bool { crate::local_search::swap::try_swap(state, tables) }
pub fn try_two_opt(state: &mut TIGState, tables: &DeltaTables) -> bool { crate::local_search::two_opt::try_two_opt(state, tables) }
pub fn try_two_opt_star(state: &mut TIGState, tables: &DeltaTables) -> bool { crate::local_search::two_opt_star::try_two_opt_star_multi(&mut [state.clone()], tables) }
pub fn try_or_opt(state: &mut TIGState, tables: &DeltaTables) -> bool { crate::local_search::or_opt::try_or_opt(state, tables) }

pub fn try_two_opt_star_between(s1: &mut TIGState, s2: &mut TIGState, tables: &DeltaTables) -> bool {
    crate::local_search::two_opt_star::try_two_opt_star_between(s1, s2, tables)
}

pub fn try_two_opt_star_multi(states: &mut [TIGState], tables: &DeltaTables) -> bool {
    crate::local_search::two_opt_star::try_two_opt_star_multi(states, tables)
}
}
pub mod relocate {
use crate::delta::DeltaTables;
use crate::tig_adaptive::TIGState;
use std::time::Instant;

// per-operator verbosity
fn verbose() -> bool { false }

/// Try an optimized relocate; returns true if improved and applied.
pub fn try_relocate(state: &mut TIGState, tables: &DeltaTables) -> bool {
    let n = state.route.len();
    if n < 4 { return false; }

    let max_checks: usize = 100_000;
    let mut checks: usize = 0;
    let op_time_limit_ms: u64 = 10;
    let op_start = Instant::now();

    for i in 1..n-1 {
        let node = state.route.nodes[i];

        // Candidate insertion positions driven by nearest neighbors of `node`.
        // For each neighbor `nb`, if it's on the current route use its position as insertion target.
        let mut tried_positions: Vec<usize> = Vec::new();
        if node < tables.neighbors.len() {
            for &nb in &tables.neighbors[node] {
                if op_start.elapsed().as_millis() as u64 > op_time_limit_ms { if verbose() { eprintln!("relocate: operator time limit {} ms reached — aborting", op_time_limit_ms); } return false; }
                // position of neighbor in current route
                if nb < state.route.pos.len() {
                    let pos = state.route.pos[nb];
                    if pos == usize::MAX { continue; }
                    // consider insertion before and after neighbor
                    for &j in &[pos, pos + 1] {
                        if j == i || j == i + 1 { continue; }
                        if tried_positions.contains(&j) { continue; }
                        tried_positions.push(j);
                        checks += 1;
                        if checks > max_checks { if verbose() { eprintln!("relocate: reached max checks ({}) — aborting operator", max_checks); } return false; }
                        let delta = tables.delta_relocate(state, i, j);
                        if delta >= 0 { continue; }
                        let insert_pos = if j > i { j - 1 } else { j };
                        if state.simulate_relocate_feasible(i, insert_pos) {
                            let node = state.route.remove(i);
                            state.route.insert(insert_pos, node);
                            state.recompute_times_from(std::cmp::min(i, insert_pos));
                            return true;
                        }
                    }
                }
            }
        }

        // fallback: try a few random insertion positions if NN didn't find improvements
        for j in (1..n).take(8) {
            if op_start.elapsed().as_millis() as u64 > op_time_limit_ms { if verbose() { eprintln!("relocate: operator time limit {} ms reached — aborting", op_time_limit_ms); } return false; }
            if j == i || j == i+1 { continue; }
            checks += 1;
            if checks > max_checks { if verbose() { eprintln!("relocate: reached max checks ({}) — aborting operator", max_checks); } return false; }
            let delta = tables.delta_relocate(state, i, j);
            if delta >= 0 { continue; }
            let insert_pos = if j > i { j - 1 } else { j };
            if state.simulate_relocate_feasible(i, insert_pos) {
                let node = state.route.remove(i);
                state.route.insert(insert_pos, node);
                state.recompute_times_from(std::cmp::min(i, insert_pos));
                return true;
            }
        }
    }
    false
}

/// Apply a relocate (remove then insert) and recompute times.
pub fn apply_relocate(state: &mut TIGState, from: usize, to: usize) {
    let node = state.route.remove(from);
    state.route.insert(to, node);
    state.recompute_times_from(std::cmp::min(from, to));
}
}
pub mod swap {
use crate::delta::DeltaTables;
use crate::tig_adaptive::TIGState;
use std::time::Instant;

fn verbose() -> bool { false }

/// Try swap operator; returns true if an improving, feasible swap is applied.
pub fn try_swap(state: &mut TIGState, tables: &DeltaTables) -> bool {
    let n = state.route.len();
    if n < 4 { return false; }

    let max_checks: usize = 100_000;
    let mut checks: usize = 0;
    let op_time_limit_ms: u64 = 10;
    let op_start = Instant::now();

    for i in 1..n-2 {
        let a = state.route.nodes[i];
        // use neighbor list of a to find promising j positions
        if a < tables.neighbors.len() {
            for &nb in &tables.neighbors[a] {
                if nb >= state.route.pos.len() { continue; }
                let j = state.route.pos[nb];
                if j == usize::MAX { continue; }
                if j <= i { continue; }
                checks += 1;
                if checks > max_checks { if verbose() { eprintln!("swap: reached max checks ({}) — aborting operator", max_checks); } return false; }
                if op_start.elapsed().as_millis() as u64 > op_time_limit_ms { if verbose() { eprintln!("swap: operator time limit {} ms reached — aborting", op_time_limit_ms); } return false; }
                let delta = tables.delta_swap(state, i, j);
                if delta >= 0 { continue; }
                if state.simulate_swap_feasible(i, j) {
                    state.route.swap(i, j);
                    state.recompute_times_from(std::cmp::min(i, j));
                    return true;
                }
            }
        }
        // fallback: limited scan ahead of fixed neighborhood size
        for j in (i+1)..((i+1+8).min(n-1)) {
            checks += 1;
            if checks > max_checks { if verbose() { eprintln!("swap: reached max checks ({}) — aborting operator", max_checks); } return false; }
            if op_start.elapsed().as_millis() as u64 > op_time_limit_ms { if verbose() { eprintln!("swap: operator time limit {} ms reached — aborting", op_time_limit_ms); } return false; }
            let delta = tables.delta_swap(state, i, j);
            if delta >= 0 { continue; }
            if state.simulate_swap_feasible(i, j) {
                state.route.swap(i, j);
                state.recompute_times_from(std::cmp::min(i, j));
                return true;
            }
        }
    }
    false
}
}
pub mod two_opt {
use crate::delta::DeltaTables;
use crate::tig_adaptive::TIGState;
use std::time::Instant;

fn verbose() -> bool { false }

/// Try two-opt operator; returns true if an improving feasible two-opt is applied.
pub fn try_two_opt(state: &mut TIGState, tables: &DeltaTables) -> bool {
    let n = state.route.len();
    if n < 4 { return false; }

    let max_checks: usize = 100_000;
    let mut checks: usize = 0;
    let op_time_limit_ms: u64 = 10;
    let op_start = Instant::now();

    for i in 1..n-2 {
        // use neighbor list of node at i to find promising j
        let a = state.route.nodes[i];
        if a < tables.neighbors.len() {
            for &nb in &tables.neighbors[a] {
                if nb >= state.route.pos.len() { continue; }
                let j = state.route.pos[nb];
                if j == usize::MAX { continue; }
                if j <= i { continue; }
                checks += 1;
                if checks > max_checks { if verbose() { eprintln!("two_opt: reached max checks ({}) — aborting operator", max_checks); } return false; }
                if op_start.elapsed().as_millis() as u64 > op_time_limit_ms { if verbose() { eprintln!("two_opt: operator time limit {} ms reached — aborting", op_time_limit_ms); } return false; }
                let delta = tables.delta_two_opt(state, i, j);
                if delta >= 0 { continue; }
                if state.simulate_two_opt_feasible(i, j) {
                    state.route.reverse(i, j);
                    state.recompute_times_from(i);
                    return true;
                }
            }
        }
        // fallback limited scan
        for j in (i+1)..((i+1+8).min(n-1)) {
            checks += 1;
            if checks > max_checks { if verbose() { eprintln!("two_opt: reached max checks ({}) — aborting operator", max_checks); } return false; }
            if op_start.elapsed().as_millis() as u64 > op_time_limit_ms { if verbose() { eprintln!("two_opt: operator time limit {} ms reached — aborting", op_time_limit_ms); } return false; }
            let delta = tables.delta_two_opt(state, i, j);
            if delta >= 0 { continue; }
            if state.simulate_two_opt_feasible(i, j) {
                state.route.reverse(i, j);
                state.recompute_times_from(i);
                return true;
            }
        }
    }
    false
}

pub fn two_opt_apply_range(state: &mut TIGState, i: usize, j: usize) {
    state.route.reverse(i, j);
    state.recompute_times_from(i);
}
}
pub mod two_opt_star {
use crate::delta::DeltaTables;
use crate::tig_adaptive::TIGState;
use std::time::Instant;

fn verbose() -> bool { false }

// TODO: Consider flattening `TIGState.distance_matrix` into a single `Vec<i32>` with stride
// so this module can avoid nested indexing (`distance_matrix[a][b]`) and instead use
// a single multiplication `dist[a * n + b]` for better cache locality and performance.

/// Analytical inter-route two-opt* feasibility check without creating temporary TIGState.
fn simulate_two_opt_star_feasible(s1: &TIGState, s2: &TIGState, i: usize, j: usize) -> bool {
    let n1 = s1.route.len();
    let n2 = s2.route.len();
    if i == 0 || j == 0 || i >= n1 - 1 || j >= n2 - 1 { return false; }

    // capacity check
    let prefix1 = if i == 0 { 0 } else { s1.prefix_demands[i - 1] };
    let total2 = if s2.prefix_demands.is_empty() { 0 } else { *s2.prefix_demands.last().unwrap() };
    let suffix2 = total2 - if j == 0 { 0 } else { s2.prefix_demands[j - 1] };
    let new_load1 = prefix1 + suffix2;
    if new_load1 > s1.max_capacity { return false; }

    let prefix2 = if j == 0 { 0 } else { s2.prefix_demands[j - 1] };
    let total1 = if s1.prefix_demands.is_empty() { 0 } else { *s1.prefix_demands.last().unwrap() };
    let suffix1 = total1 - if i == 0 { 0 } else { s1.prefix_demands[i - 1] };
    let new_load2 = prefix2 + suffix1;
    if new_load2 > s2.max_capacity { return false; }

    // quick slack checks
    let prev_arrival1 = if i == 0 { s1.time } else { s1.arrival_times[i - 1] };
    let prev_node1 = s1.route.nodes[i - 1];
    let new_first1 = s2.route.nodes[j];
    let arrival_new_first1 = (prev_arrival1 + s1.travel_time(prev_node1, new_first1) + s1.service[prev_node1]).max(s1.tw_start[new_first1]);
    let orig_arrival_first1 = s1.arrival_times[i];
    let delta1 = arrival_new_first1 - orig_arrival_first1;
    let allowed1 = if i < s1.suffix_min_slack.len() { s1.suffix_min_slack[i] } else { i32::MAX };
    if delta1 > allowed1 { return false; }

    let prev_arrival2 = if j == 0 { s2.time } else { s2.arrival_times[j - 1] };
    let prev_node2 = s2.route.nodes[j - 1];
    let new_first2 = s1.route.nodes[i];
    let arrival_new_first2 = (prev_arrival2 + s2.travel_time(prev_node2, new_first2) + s2.service[prev_node2]).max(s2.tw_start[new_first2]);
    let orig_arrival_first2 = s2.arrival_times[j];
    let delta2 = arrival_new_first2 - orig_arrival_first2;
    let allowed2 = if j < s2.suffix_min_slack.len() { s2.suffix_min_slack[j] } else { i32::MAX };
    if delta2 > allowed2 { return false; }

    // full simulation fallback for route1
    let mut current_time = if i == 0 { s1.time } else { s1.arrival_times[i - 1] };
    let mut prev = s1.route.nodes[i - 1];
    for &nd in &s2.route.nodes[j..] {
        current_time += s1.travel_time(prev, nd) + s1.service[prev];
        if current_time < s1.tw_start[nd] { current_time = s1.tw_start[nd]; }
        if current_time > s1.tw_end[nd] { return false; }
        prev = nd;
    }

    let mut current_time2 = if j == 0 { s2.time } else { s2.arrival_times[j - 1] };
    let mut prev2 = s2.route.nodes[j - 1];
    for &nd in &s1.route.nodes[i..] {
        current_time2 += s2.travel_time(prev2, nd) + s2.service[prev2];
        if current_time2 < s2.tw_start[nd] { current_time2 = s2.tw_start[nd]; }
        if current_time2 > s2.tw_end[nd] { return false; }
        prev2 = nd;
    }

    true
}

/// Attempt an inter-route two-opt* between two TIGStates. Returns true if improved.
pub fn try_two_opt_star_between(s1: &mut TIGState, s2: &mut TIGState, tables: &DeltaTables) -> bool {
    let n1 = s1.route.len();
    let n2 = s2.route.len();
    if n1 < 3 || n2 < 3 { return false; }

    let max_checks: usize = 100_000;
    let mut checks: usize = 0;
    let op_time_limit_ms: u64 = 20;
    let op_start = Instant::now();

    fn route_cost(state: &TIGState) -> i64 {
        let mut cost: i64 = 0;
        for i in 0..state.route.len().saturating_sub(1) {
            let a = state.route.nodes[i];
            let b = state.route.nodes[i+1];
            cost += state.travel_time(a, b) as i64 + state.service[a] as i64;
        }
        cost
    }

    let base_cost = route_cost(s1) + route_cost(s2);

    for i in 1..(n1 - 1) {
        for j in 1..(n2 - 1) {
            checks += 1;
            if checks > max_checks { if verbose() { eprintln!("two_opt_star_between: reached max checks ({}) — aborting operator", max_checks); } return false; }
            if op_start.elapsed().as_millis() as u64 > op_time_limit_ms { if verbose() { eprintln!("two_opt_star_between: operator time limit {} ms reached — aborting", op_time_limit_ms); } return false; }
            let prev_a = s1.route.nodes[i - 1];
            let a_i = s1.route.nodes[i];
            let prev_b = s2.route.nodes[j - 1];
            let b_j = s2.route.nodes[j];

            let old_edges = (tables.travel_time(prev_a, a_i) + tables.travel_time(prev_b, b_j)) as i64;
            let new_edges = (tables.travel_time(prev_a, b_j) + tables.travel_time(prev_b, a_i)) as i64;
            let pre_delta = new_edges - old_edges;
            if pre_delta >= 0 { continue; }

            if !simulate_two_opt_star_feasible(s1, s2, i, j) {
                continue;
            }

            let mut new1: Vec<usize> = Vec::with_capacity(n1 - i + (n2 - j));
            new1.extend_from_slice(&s1.route.nodes[0..i]);
            new1.extend_from_slice(&s2.route.nodes[j..]);
            let mut new2: Vec<usize> = Vec::with_capacity(n2 - j + (n1 - i));
            new2.extend_from_slice(&s2.route.nodes[0..j]);
            new2.extend_from_slice(&s1.route.nodes[i..]);

            let mut t1_cost: i64 = 0;
            for k in 0..new1.len().saturating_sub(1) {
                let a = new1[k];
                let b = new1[k+1];
                t1_cost += s1.travel_time(a, b) as i64 + s1.service[a] as i64;
            }
            let mut t2_cost: i64 = 0;
            for k in 0..new2.len().saturating_sub(1) {
                let a = new2[k];
                let b = new2[k+1];
                t2_cost += s2.travel_time(a, b) as i64 + s2.service[a] as i64;
            }

            if t1_cost + t2_cost < base_cost {
                s1.route.nodes.truncate(i);
                s1.route.nodes.extend_from_slice(&s2.route.nodes[j..]);
                s1.arrival_times.truncate(i);
                s1.recompute_times_from(i);
                s1.recompute_load();

                s2.route.nodes.truncate(j);
                s2.route.nodes.extend_from_slice(&s1.route.nodes[i..]);
                s2.arrival_times.truncate(j);
                s2.recompute_times_from(j);
                s2.recompute_load();

                return true;
            }
        }
    }

    false
}

pub fn try_two_opt_star_multi(states: &mut [TIGState], tables: &DeltaTables) -> bool {
    let m = states.len();
    for a in 0..m {
        for b in (a+1)..m {
            let (left, right) = states.split_at_mut(b);
            let s_a = &mut left[a];
            let s_b = &mut right[0];
            if try_two_opt_star_between(s_a, s_b, tables) {
                return true;
            }
        }
    }
    false
}
}
pub mod or_opt {
use crate::delta::DeltaTables;
use crate::tig_adaptive::TIGState;
use std::time::Instant;

fn verbose() -> bool { false }

/// Or-opt operator (block relocation) sizes 1..=3
pub fn try_or_opt(state: &mut TIGState, tables: &DeltaTables) -> bool {
    let n = state.route.len();
    if n < 5 { return false; }

    let max_checks: usize = 100_000;
    let mut checks: usize = 0;
    let op_time_limit_ms: u64 = 15;
    let op_start = Instant::now();

    for len in 1..=3 {
        // try neighbor-driven insertion positions first
        for i in 1..(n - len) {
            // copy block out to avoid aliasing when mutating route
            let block: Vec<usize> = state.route.nodes[i..i+len].iter().copied().collect();
            let block_first = block[0];
            // name intermediate positions explicitly for clarity and correctness
            let block_second = if len >= 2 { Some(block[1]) } else { None };
            let block_third = if len == 3 { Some(block[2]) } else { None };
            let block_tail = match len {
                1 => block_first,
                2 => block_second.unwrap(),
                3 => block_third.unwrap(),
                _ => block[len - 1],
            };

            // try neighbor-driven insertion positions first (based on first node in block)
            if block_first < tables.neighbors.len() {
                for &nb in &tables.neighbors[block_first] {
                    if nb >= state.route.pos.len() { continue; }
                    let j = state.route.pos[nb];
                    // skip neighbor slots not present in this route
                    if j == usize::MAX { continue; }
                    if j >= i && j <= i + len { continue; }
                    checks += 1;
                    if checks > max_checks { if verbose() { eprintln!("or_opt: reached max checks ({}) — aborting operator", max_checks); } return false; }
                    if op_start.elapsed().as_millis() as u64 > op_time_limit_ms { if verbose() { eprintln!("or_opt: operator time limit {} ms reached — aborting", op_time_limit_ms); } return false; }

                    // compute delta quickly
                    let pred_block = state.route.nodes.get(i - 1).copied();
                    let succ_block = state.route.nodes.get(i + len).copied();
                    let pred_j = state.route.nodes.get(j.wrapping_sub(1)).copied();
                    let succ_j = state.route.nodes.get(j).copied();

                    let mut delta = 0i32;
                    if let (Some(pb), Some(sb)) = (pred_block, succ_block) {
                        delta -= tables.travel_time(pb, block_first);
                        delta -= tables.travel_time(block_tail, sb);
                        delta += tables.travel_time(pb, sb);
                    }
                    if let (Some(pj), Some(sj)) = (pred_j, succ_j) { delta -= tables.travel_time(pj, sj); }
                    if let Some(pj) = pred_j { delta += tables.travel_time(pj, block_first); }
                    if let Some(sj) = succ_j { delta += tables.travel_time(block_tail, sj); }

                    if delta >= 0 { continue; }

                    // simulate new order
                    let mut new_nodes: Vec<usize> = state.route.nodes.iter().copied().collect();
                    for _ in 0..len { new_nodes.remove(i); }
                    let insert_pos = if j > i { j - len } else { j };
                    for (k, &nd) in block.iter().enumerate() {
                        new_nodes.insert(insert_pos + k, nd);
                    }

                    let start_sim = std::cmp::min(i, insert_pos);
                    if crate::local_search::controller::simulate_time_feasible(&new_nodes, state, start_sim) {
                        for _ in 0..len { state.route.remove(i); }
                        for (k, &nd) in block.iter().enumerate() {
                            state.route.insert(insert_pos + k, nd);
                        }
                        state.recompute_times_from(start_sim);
                        return true;
                    }
                }
            }

            // fallback scan over all insertion positions
            for j in 1..n {
                checks += 1;
                if checks > max_checks { if verbose() { eprintln!("or_opt: reached max checks ({}) — aborting operator", max_checks); } return false; }
                if op_start.elapsed().as_millis() as u64 > op_time_limit_ms { if verbose() { eprintln!("or_opt: operator time limit {} ms reached — aborting", op_time_limit_ms); } return false; }
                if j >= i && j <= i + len { continue; }

                // compute delta quickly
                let pred_block = state.route.nodes.get(i - 1).copied();
                let succ_block = state.route.nodes.get(i + len).copied();
                let pred_j = state.route.nodes.get(j.wrapping_sub(1)).copied();
                let succ_j = state.route.nodes.get(j).copied();

                let mut delta = 0i32;
                if let (Some(pb), Some(sb)) = (pred_block, succ_block) {
                    delta -= tables.travel_time(pb, block_first);
                    delta -= tables.travel_time(block_tail, sb);
                    delta += tables.travel_time(pb, sb);
                }
                if let (Some(pj), Some(sj)) = (pred_j, succ_j) { delta -= tables.travel_time(pj, sj); }
                if let Some(pj) = pred_j { delta += tables.travel_time(pj, block_first); }
                if let Some(sj) = succ_j { delta += tables.travel_time(block_tail, sj); }

                if delta >= 0 { continue; }

                // simulate new order
                let mut new_nodes: Vec<usize> = state.route.nodes.iter().copied().collect();
                for _ in 0..len { new_nodes.remove(i); }
                let insert_pos = if j > i { j - len } else { j };
                for (k, &nd) in block.iter().enumerate() {
                    new_nodes.insert(insert_pos + k, nd);
                }

                let start_sim = std::cmp::min(i, insert_pos);
                if crate::local_search::controller::simulate_time_feasible(&new_nodes, state, start_sim) {
                    for _ in 0..len { state.route.remove(i); }
                    for (k, &nd) in block.iter().enumerate() {
                        state.route.insert(insert_pos + k, nd);
                    }
                    state.recompute_times_from(start_sim);
                    return true;
                }
            }
        }
    }
    false
}
}
pub mod ejection_chain {
use crate::delta::DeltaTables;
use crate::tig_adaptive::TIGState;
use std::time::Instant;

fn verbose() -> bool { false }

/// Simple depth-2 ejection-chain: pick seed node, insert at neighbor position,
/// then try to relocate the displaced node to one of its neighbor positions.
pub fn try_ejection_chain(state: &mut TIGState, tables: &DeltaTables) -> bool {
    let n = state.route.len();
    if n < 6 { return false; }

    let max_checks: usize = 100_000;
    let mut checks: usize = 0;
    let op_time_limit_ms: u64 = 20;
    let op_start = Instant::now();

    // helper: compute total travel cost of a node ordering (slice-accepting)
    let total_travel = |nodes: &[usize]| -> i64 {
        let mut sum: i64 = 0;
        for w in nodes.windows(2) {
            sum += tables.travel_time(w[0], w[1]) as i64;
        }
        sum
    };

    let orig_nodes = state.route.nodes.clone();
    let orig_cost = total_travel(&orig_nodes);

    // configurable depth: support depth-2 (default) and depth-3 when requested
    let depth: usize = 2usize;

    for i in 1..n - 1 {
        let a = state.route.nodes[i];
        if a >= tables.neighbors.len() { continue; }

        for &nb in &tables.neighbors[a] {
            if nb >= state.route.pos.len() { continue; }
            let j = state.route.pos[nb];
            // skip neighbor slots that are not present in this route (pos may be usize::MAX)
            if j == usize::MAX { continue; }
            if j == i { continue; }

            checks += 1;
            if checks > max_checks { if verbose() { eprintln!("ejection_chain: reached max checks ({}) — aborting operator", max_checks); } return false; }
            if op_start.elapsed().as_millis() as u64 > op_time_limit_ms { if verbose() { eprintln!("ejection_chain: operator time limit {} ms reached — aborting", op_time_limit_ms); } return false; }

            // simulate first relocate a -> j
            let mut stage1: Vec<usize> = orig_nodes.to_vec();
            if i >= stage1.len() { continue; }
            stage1.remove(i);
            let insert_pos1 = if j > i { j - 1 } else { j };
            let insert_pos1 = insert_pos1.min(stage1.len());
            stage1.insert(insert_pos1, a);

            // node displaced at original j position (pre-move)
            let b = orig_nodes[j];
            if b == a { continue; }
            if b >= tables.neighbors.len() { continue; }

            // try moving b to one of its neighbors (depth-2)
            for &kb in &tables.neighbors[b] {
                if kb >= state.route.pos.len() { continue; }
                let kpos_orig = state.route.pos[kb];
                // compute b's current index in stage1
                let b_idx = stage1.iter().position(|&x| x == b).unwrap_or(usize::MAX);
                if b_idx == usize::MAX { continue; }
                let mut stage2 = stage1.clone();
                if b_idx >= stage2.len() { continue; }
                stage2.remove(b_idx);
                let insert_pos2 = if kpos_orig > b_idx { kpos_orig - 1 } else { kpos_orig };
                let insert_pos2 = insert_pos2.min(stage2.len());
                stage2.insert(insert_pos2, b);
                // If depth>=3, try a short depth-3 chain by selecting nearby candidates around insert_pos2
                if depth >= 3 {
                    // consider a small window around insert_pos2 for the next displaced candidate
                    let win_start = insert_pos2.saturating_sub(1);
                    let win_end = (insert_pos2 + 1).min(stage2.len().saturating_sub(1));
                    for k in win_start..=win_end {
                        if k >= stage2.len() { continue; }
                        let c = stage2[k];
                        if c == a || c == b { continue; }
                        if c >= tables.neighbors.len() { continue; }

                        for &kc in &tables.neighbors[c] {
                            if kc >= state.route.pos.len() { continue; }
                            checks += 1;
                            if checks > max_checks { if verbose() { eprintln!("ejection_chain: reached max checks ({}) — aborting operator", max_checks); } return false; }
                            if op_start.elapsed().as_millis() as u64 > op_time_limit_ms { if verbose() { eprintln!("ejection_chain: operator time limit {} ms reached — aborting", op_time_limit_ms); } return false; }

                            let kpos = state.route.pos[kc];
                            let mut stage3 = stage2.clone();
                            // remove c's current position in stage3
                            if let Some(cpos) = stage3.iter().position(|&x| x == c) {
                                stage3.remove(cpos);
                            } else { continue; }
                            let ins_pos = if kpos > stage3.len() { stage3.len() } else { kpos };
                            stage3.insert(ins_pos, c);

                            let start_sim3 = std::cmp::min(i, std::cmp::min(insert_pos1, std::cmp::min(insert_pos2, ins_pos)));
                            if !crate::local_search::controller::simulate_time_feasible(&stage3, state, start_sim3) { continue; }

                            let new_cost = total_travel(&stage3);
                            if new_cost < orig_cost {
                                state.route.nodes = stage3.into();
                                for (idx, &nd) in state.route.nodes.iter().enumerate() { if nd < state.route.pos.len() { state.route.pos[nd] = idx; } }
                                state.recompute_times_from(start_sim3);
                                return true;
                            }
                        }
                    }
                }

                // quick feasibility check for depth-2 result
                let start_sim = std::cmp::min(i, std::cmp::min(insert_pos1, insert_pos2));
                if crate::local_search::controller::simulate_time_feasible(&stage2, state, start_sim) {
                    let new_cost = total_travel(&stage2);
                    if new_cost < orig_cost {
                        // apply to real state: convert Vec into SmallVec via Into
                        state.route.nodes = stage2.into();
                        // rebuild pos mapping
                        for (idx, &nd) in state.route.nodes.iter().enumerate() {
                            if nd < state.route.pos.len() { state.route.pos[nd] = idx; }
                        }
                        state.recompute_times_from(start_sim);
                        return true;
                    }
                }
            }
        }
    }

    false
}

/// Multi-route ejection chain: try depth-2 chains across pairs of routes in `states`.
pub fn try_ejection_chain_multi(states: &mut [TIGState], tables: &DeltaTables) -> bool {
    if states.len() < 2 { return false; }

    let max_checks: usize = 100_000;
    let mut checks: usize = 0;
    let op_time_limit_ms: u64 = 30;
    let op_start = Instant::now();

    // total travel for a route nodes slice
    let route_travel = |nodes: &[usize]| -> i64 {
        let mut sum = 0i64;
        for w in nodes.windows(2) { sum += tables.travel_time(w[0], w[1]) as i64; }
        sum
    };

    // iterate over ordered pairs of routes (r -> s)
    for r in 0..states.len() {
        for s in 0..states.len() {
            if r == s { continue; }
            let (left, right) = states.split_at_mut(s.max(r));
            // get mutable refs to both states
            let (st_r, st_s) = if r < s { (&mut left[r], &mut right[0]) } else { (&mut right[0], &mut left[s]) };

            let orig_nodes_r = st_r.route.nodes.clone();
            let orig_nodes_s = st_s.route.nodes.clone();
            let orig_cost = route_travel(&orig_nodes_r) + route_travel(&orig_nodes_s);

            let n_r = st_r.route.len();
            let n_s = st_s.route.len();
            if n_r < 2 || n_s < 2 { continue; }

            for i in 1..(n_r - 1) {
                let a = st_r.route.nodes[i];
                if a >= tables.neighbors.len() { continue; }

                for &nb in &tables.neighbors[a] {
                    if nb >= st_s.route.pos.len() { continue; }
                    let j = st_s.route.pos[nb];
                    if j == usize::MAX { continue; }

                    checks += 1;
                    if checks > max_checks { if verbose() { eprintln!("ejection_chain_multi: reached max checks ({}) — aborting", max_checks); } return false; }
                    if op_start.elapsed().as_millis() as u64 > op_time_limit_ms { if verbose() { eprintln!("ejection_chain_multi: time limit {} ms reached — aborting", op_time_limit_ms); } return false; }

                    // simulate moving a from r at i to s at j
                    let mut r_stage: Vec<usize> = orig_nodes_r.to_vec();
                    let mut s_stage: Vec<usize> = orig_nodes_s.to_vec();
                    if i >= r_stage.len() { continue; }
                    let a_node = r_stage.remove(i);
                    let insert_pos_s = if j > s_stage.len() { s_stage.len() } else { j };
                    s_stage.insert(insert_pos_s, a_node);

                    // displaced node in s at position insert_pos_s+1 (or at j if j < orig len)
                    let displaced_idx = if insert_pos_s + 1 < s_stage.len() { insert_pos_s + 1 } else { continue; };
                    let b = s_stage[displaced_idx];
                    if b >= tables.neighbors.len() { continue; }

                    // try relocating b to one of its neighbors in either route
                    for &kb in &tables.neighbors[b] {
                        let mut r_try = r_stage.clone();
                        let mut s_try = s_stage.clone();
                        // remove b from s_try
                        let b_pos_s = s_try.iter().position(|&x| x == b).unwrap_or(usize::MAX);
                        if b_pos_s == usize::MAX { continue; }
                        s_try.remove(b_pos_s);

                        // decide insertion target: position from kb in r_try or s_try depending on presence
                        let pos_in_r = st_r.route.pos.get(kb).copied().unwrap_or(usize::MAX);
                        let pos_in_s = st_s.route.pos.get(kb).copied().unwrap_or(usize::MAX);

                        // try insert into r_try if valid
                        if pos_in_r != usize::MAX {
                            let ins = pos_in_r.min(r_try.len());
                            r_try.insert(ins, b);
                        } else if pos_in_s != usize::MAX {
                            let ins = pos_in_s.min(s_try.len());
                            s_try.insert(ins, b);
                        } else {
                            continue;
                        }

                        // quick feasibility per-route
                        let start_sim_r = 0usize.max(i.saturating_sub(1));
                        let start_sim_s = 0usize;
                        if !crate::local_search::controller::simulate_time_feasible(&r_try, st_r, start_sim_r) { continue; }
                        if !crate::local_search::controller::simulate_time_feasible(&s_try, st_s, start_sim_s) { continue; }

                        let new_cost = route_travel(&r_try) + route_travel(&s_try);
                        if new_cost < orig_cost {
                            // apply changes
                            st_r.route.nodes = r_try.into();
                            st_s.route.nodes = s_try.into();
                            // rebuild pos arrays conservatively
                            for (idx, &nd) in st_r.route.nodes.iter().enumerate() { if nd < st_r.route.pos.len() { st_r.route.pos[nd] = idx; } }
                            for (idx, &nd) in st_s.route.nodes.iter().enumerate() { if nd < st_s.route.pos.len() { st_s.route.pos[nd] = idx; } }
                            st_r.recompute_times_from(i.saturating_sub(1));
                            st_s.recompute_times_from(0);
                            return true;
                        }
                    }
                }
            }
        }
    }

    false
}
}
pub mod time_window_repair {
use crate::delta::DeltaTables;
use crate::tig_adaptive::TIGState;
use std::time::Instant;

fn verbose() -> bool { false }

/// Try to repair time-window violations within a single route by relocating violating nodes
/// to insertion positions that restore feasibility. Uses neighbor lists first, then a bounded scan.
pub fn try_time_window_repair(state: &mut TIGState, tables: &DeltaTables) -> bool {
    let n = state.route.len();
    if n < 3 { return false; }

    let max_checks: usize = 100_000;
    let mut checks: usize = 0;
    let op_time_limit_ms: u64 = 20;
    let op_start = Instant::now();

    // find first violating index
    let viol_idx = state.arrival_times.iter().enumerate().find_map(|(i,&t)| {
        let node = state.route.nodes.get(i).copied().unwrap_or(usize::MAX);
        if node == usize::MAX { return None; }
        if t > state.tw_end[node] { Some(i) } else { None }
    });
    let i = match viol_idx { Some(v) => v, None => return false };

    let node = state.route.nodes[i];
    // try neighbor-driven insertion positions
    if node < tables.neighbors.len() {
        for &nb in &tables.neighbors[node] {
            if nb >= state.route.pos.len() { continue; }
            let j = state.route.pos[nb];
            if j == i { continue; }
            checks += 1;
            if checks > max_checks { if verbose() { eprintln!("time_repair: reached max checks ({})", max_checks); } return false; }
            if op_start.elapsed().as_millis() as u64 > op_time_limit_ms { if verbose() { eprintln!("time_repair: op time limit {} ms reached", op_time_limit_ms); } return false; }

            // simulate relocate i -> j
            let mut new_nodes: Vec<usize> = state.route.nodes.iter().copied().collect();
            if i >= new_nodes.len() { continue; }
            let nd = new_nodes.remove(i);
            let insert_pos = if j > i { j - 1 } else { j };
            let insert_pos = insert_pos.min(new_nodes.len());
            new_nodes.insert(insert_pos, nd);

            // quick delta prefilter: skip if relocate does not reduce travel cost
            let pred_from = if i == 0 { None } else { Some(state.route.nodes[i - 1]) };
            let succ_from = if i + 1 >= state.route.len() { None } else { Some(state.route.nodes[i + 1]) };
            let pred_to = if insert_pos == 0 { None } else { Some(new_nodes[insert_pos - 1]) };
            let succ_to = if insert_pos >= new_nodes.len() { None } else { Some(new_nodes[insert_pos]) };
            let mut delta = 0i32;
            if let Some(pf) = pred_from { delta -= tables.travel_time(pf, nd); }
            if let Some(sf) = succ_from { delta -= tables.travel_time(nd, sf); }
            if let (Some(pf), Some(sf)) = (pred_from, succ_from) { delta += tables.travel_time(pf, sf); }
            if let (Some(pj), Some(sj)) = (pred_to, succ_to) { delta -= tables.travel_time(pj, sj); }
            if let Some(pj) = pred_to { delta += tables.travel_time(pj, nd); }
            if let Some(sj) = succ_to { delta += tables.travel_time(nd, sj); }
            if delta >= 0 { continue; }

            let start_sim = std::cmp::min(i, insert_pos);
            if crate::local_search::controller::simulate_time_feasible(&new_nodes, state, start_sim) {
                // apply
                state.route.nodes = new_nodes.into();
                for (idx, &nd) in state.route.nodes.iter().enumerate() { if nd < state.route.pos.len() { state.route.pos[nd] = idx; } }
                state.recompute_times_from(start_sim);
                return true;
            }
        }
    }

    // fallback: bounded full scan of insertion positions
    for j in 1..n {
        checks += 1;
        if checks > max_checks { if verbose() { eprintln!("time_repair: reached max checks ({})", max_checks); } return false; }
        if op_start.elapsed().as_millis() as u64 > op_time_limit_ms { if verbose() { eprintln!("time_repair: op time limit {} ms reached", op_time_limit_ms); } return false; }
        if j == i { continue; }
        let mut new_nodes: Vec<usize> = state.route.nodes.iter().copied().collect();
        if i >= new_nodes.len() { continue; }
        let nd = new_nodes.remove(i);
        let insert_pos = if j > i { j - 1 } else { j };
        let insert_pos = insert_pos.min(new_nodes.len());
        new_nodes.insert(insert_pos, nd);
        let start_sim = std::cmp::min(i, insert_pos);
        if crate::local_search::controller::simulate_time_feasible(&new_nodes, state, start_sim) {
            state.route.nodes = new_nodes.into();
            for (idx, &nd) in state.route.nodes.iter().enumerate() { if nd < state.route.pos.len() { state.route.pos[nd] = idx; } }
            state.recompute_times_from(start_sim);
            return true;
        }
    }

    false
}

/// Multi-route repair: move violating nodes across routes trying to restore feasibility.
pub fn try_time_window_repair_multi(states: &mut [TIGState], tables: &DeltaTables) -> bool {
    if states.len() < 2 { return false; }
    let max_checks: usize = 100_000;
    let mut checks: usize = 0;
    let op_time_limit_ms: u64 = 30;
    let op_start = Instant::now();

    // find any violating node in any route
    for r in 0..states.len() {
        let n_r = states[r].route.len();
        for i in 0..n_r {
            let node = states[r].route.nodes[i];
            if states[r].arrival_times.get(i).copied().unwrap_or(0) > states[r].tw_end[node] {
                // try to move node to other routes
                for s in 0..states.len() {
                    if s == r { continue; }
                    let (left, right) = states.split_at_mut(s.max(r));
                    let (st_r, st_s) = if r < s { (&mut left[r], &mut right[0]) } else { (&mut right[0], &mut left[s]) };

                    // candidate insertion positions in st_s via neighbor list
                    if node < tables.neighbors.len() {
                        for &nb in &tables.neighbors[node] {
                            if nb >= st_s.route.pos.len() { continue; }
                            let j = st_s.route.pos[nb];
                            checks += 1;
                            if checks > max_checks { return false; }
                            if op_start.elapsed().as_millis() as u64 > op_time_limit_ms { return false; }

                            // simulate move r:i -> s:j
                            let mut r_nodes: Vec<usize> = st_r.route.nodes.iter().copied().collect();
                            let mut s_nodes: Vec<usize> = st_s.route.nodes.iter().copied().collect();
                            if i >= r_nodes.len() { continue; }
                            let nd = r_nodes.remove(i);
                            let insert_pos = if j > s_nodes.len() { s_nodes.len() } else { j };
                            s_nodes.insert(insert_pos, nd);

                            if !crate::local_search::controller::simulate_time_feasible(&r_nodes, st_r, i.saturating_sub(1)) { continue; }
                            if !crate::local_search::controller::simulate_time_feasible(&s_nodes, st_s, 0) { continue; }

                            // apply
                            st_r.route.nodes = r_nodes.into();
                            st_s.route.nodes = s_nodes.into();
                            for (idx, &nd) in st_r.route.nodes.iter().enumerate() { if nd < st_r.route.pos.len() { st_r.route.pos[nd] = idx; } }
                            for (idx, &nd) in st_s.route.nodes.iter().enumerate() { if nd < st_s.route.pos.len() { st_s.route.pos[nd] = idx; } }
                            st_r.recompute_times_from(i.saturating_sub(1));
                            st_s.recompute_times_from(0);
                            return true;
                        }
                    }
                }
            }
        }
    }

    false
}
}
pub mod batched_time_window_repair {
use crate::delta::DeltaTables;
use crate::tig_adaptive::TIGState;
use std::time::Instant;

#[allow(dead_code)]
fn verbose() -> bool { false }

/// Batched repair: collect violating nodes (up to a cap) then try coordinated reinsertion
/// across routes / positions. Conservative: limits batch size and neighbor candidates.
pub fn try_batched_time_window_repair(states: &mut [TIGState], tables: &DeltaTables) -> bool {
    if states.is_empty() { return false; }

    let max_checks: usize = 200_000;
    let mut checks: usize = 0;
    let op_time_limit_ms: u64 = 100;
    let op_start = Instant::now();

    // Collect violations (route_idx, pos)
    let mut violations: Vec<(usize, usize)> = Vec::new();
    for (r, st) in states.iter().enumerate() {
        for i in 0..st.route.len() {
            let node = st.route.nodes[i];
            if st.arrival_times.get(i).copied().unwrap_or(0) > st.tw_end[node] {
                violations.push((r, i));
            }
        }
    }
    if violations.is_empty() { return false; }

    // limit number of violating nodes we attempt to coordinate
    let max_viol = 6usize;
    if violations.len() > max_viol { violations.truncate(max_viol); }

    // batch sizes to try (1..=3)
    let max_batch = 3usize;

    // neighbor candidate cap per node
    let per_node_k = 6usize;

    // helper: simulate per-route feasibility after applying modified node lists
    let simulate_all = |rnodes: &Vec<Vec<usize>>, states_ref: &mut [TIGState]| -> bool {
        for (ri, nodes_vec) in rnodes.iter().enumerate() {
            // reuse existing state reference to check feasibility; provide start index 0 for safety
            if !crate::local_search::controller::simulate_time_feasible(nodes_vec, &states_ref[ri], 0) {
                return false;
            }
        }
        true
    };

    // Build current route node lists for easy mutation
    let route_nodes: Vec<Vec<usize>> = states.iter().map(|s| s.route.nodes.to_vec()).collect();

    // For each batch size
    for bsize in 1..=max_batch {
        // generate combinations of bsize indices from violations (cheap since violations small)
        let vlen = violations.len();
        if vlen < bsize { break; }

        // simple recursive comb generator
        fn gen_combs(cur: &mut Vec<usize>, start: usize, left: usize, out: &mut Vec<Vec<usize>>, n: usize) {
            if left == 0 { out.push(cur.clone()); return; }
            for i in start..=(n - left) {
                cur.push(i);
                gen_combs(cur, i + 1, left - 1, out, n);
                cur.pop();
            }
        }

        let mut combs: Vec<Vec<usize>> = Vec::new();
        gen_combs(&mut Vec::new(), 0, bsize, &mut combs, vlen);

        for comb in combs {
            if op_start.elapsed() > std::time::Duration::from_millis(op_time_limit_ms) { return false; }

            // build working copy of route_nodes
            let working = route_nodes.clone();

            // For each selected violation, produce candidate target positions from neighbor lists (cap per_node_k)
            let mut candidates_per_violation: Vec<Vec<(usize, usize)>> = Vec::new();
            for &vi in &comb {
                let (r_idx, pos_idx) = violations[vi];
                let node = states[r_idx].route.nodes[pos_idx];
                // candidates: (target_route_idx, insert_pos)
                let mut cand: Vec<(usize, usize)> = Vec::new();

                // neighbors global -> map into route positions in each route
                if node < tables.neighbors.len() {
                    for &nb in tables.neighbors[node].iter().take(per_node_k) {
                        // map neighbor to route and pos by checking all routes' pos array
                        for tr in 0..states.len() {
                            if nb < states[tr].route.pos.len() {
                                let p = states[tr].route.pos[nb];
                                if p != usize::MAX { cand.push((tr, p)); }
                            }
                        }
                    }
                }

                // always allow same-route insertions (scan small window around original pos)
                let win = 6usize;
                let rlen = working[r_idx].len();
                let start_w = pos_idx.saturating_sub(win);
                let end_w = (pos_idx + win).min(rlen.saturating_sub(1));
                for p in start_w..=end_w { if p != pos_idx { cand.push((r_idx, p)); } }

                // de-dup and cap candidates
                cand.sort_unstable(); cand.dedup();
                if cand.len() > per_node_k { cand.truncate(per_node_k); }
                if cand.is_empty() { cand.push((r_idx, pos_idx)); }
                candidates_per_violation.push(cand);
            }

            // Cartesian iterate over candidates (bounded: per_node_k^bsize) but with small bsize and per_node_k it's tractable
            let mut idxs = vec![0usize; candidates_per_violation.len()];
            loop {
                // early time/check guard
                checks += 1;
                if checks > max_checks { return false; }
                if op_start.elapsed() > std::time::Duration::from_millis(op_time_limit_ms) { return false; }

                // apply candidate assignments on working copy
                let mut working2 = working.clone();
                let mut valid = true;
                // remove nodes in reverse order per their original routes to avoid shifting indices unpredictably
                // collect removals per route
                let mut removals: Vec<Vec<usize>> = vec![Vec::new(); states.len()];
                for &vi in &comb { let (r_idx, pos_idx) = violations[vi]; removals[r_idx].push(pos_idx); }
                for r in 0..removals.len() { removals[r].sort_unstable_by(|a,b| b.cmp(a)); for &p in &removals[r] { if p < working2[r].len() { working2[r].remove(p); } else { valid = false; break; } } if !valid { break; } }
                if !valid { break; }

                // perform insertions according to current idxs
                for (k, &vi) in comb.iter().enumerate() {
                    let cand = &candidates_per_violation[k];
                    if idxs[k] >= cand.len() { valid = false; break; }
                    let (tr, pos) = cand[idxs[k]];
                    let node = states[violations[vi].0].route.nodes[violations[vi].1];
                    let ins = pos.min(working2[tr].len());
                    working2[tr].insert(ins, node);
                }
                if !valid { break; }

                // quick prefilter: approximate sum of relocate deltas for the batch using original positions
                let mut approx_delta_sum: i64 = 0;
                for (k, &vi) in comb.iter().enumerate() {
                    let (r_idx, pos_idx) = violations[vi];
                    let (tr, pos) = candidates_per_violation[k][idxs[k]];
                    let node = states[r_idx].route.nodes[pos_idx];
                    // compute local relocate delta on source route
                    let pred_from = if pos_idx == 0 { None } else { Some(states[r_idx].route.nodes[pos_idx - 1]) };
                    let succ_from = if pos_idx + 1 >= states[r_idx].route.len() { None } else { Some(states[r_idx].route.nodes[pos_idx + 1]) };
                    let pred_to = if pos == 0 { None } else { Some(states[tr].route.nodes[pos - 1]) };
                    let succ_to = if pos >= states[tr].route.len() { None } else { Some(states[tr].route.nodes[pos]) };
                    let mut d = 0i64;
                    if let Some(pf) = pred_from { d -= tables.travel_time(pf, node) as i64; }
                    if let Some(sf) = succ_from { d -= tables.travel_time(node, sf) as i64; }
                    if let (Some(pf), Some(sf)) = (pred_from, succ_from) { d += tables.travel_time(pf, sf) as i64; }
                    if let (Some(pj), Some(sj)) = (pred_to, succ_to) { d -= tables.travel_time(pj, sj) as i64; }
                    if let Some(pj) = pred_to { d += tables.travel_time(pj, node) as i64; }
                    if let Some(sj) = succ_to { d += tables.travel_time(node, sj) as i64; }
                    approx_delta_sum += d;
                }
                if approx_delta_sum >= 0 { 
                    // skip heavy simulate when approximate batch delta is not improving
                } else {
                    // simulate per-route feasibility
                    // We must provide per-route TIGState references; clone short-lived state refs
                    let mut states_clone_for_sim: Vec<TIGState> = states.iter().cloned().collect();
                    // replace nodes in clones with working2 and test
                    for (ri, nodes_vec) in working2.iter().enumerate() { states_clone_for_sim[ri].route.nodes = nodes_vec.clone().into(); }
                    if simulate_all(&working2, &mut states_clone_for_sim) {
                        // count remaining violations after application
                        let mut violations_after = 0usize;
                        for (_ri, st) in states_clone_for_sim.iter().enumerate() {
                            for ii in 0..st.route.len() {
                                let node = st.route.nodes[ii];
                                if st.arrival_times.get(ii).copied().unwrap_or(0) > st.tw_end[node] { violations_after += 1; }
                            }
                        }
                        if violations_after < violations.len() {
                            // apply to real states
                            for (ri, nodes_vec) in working2.into_iter().enumerate() {
                                states[ri].route.nodes = nodes_vec.into();
                                for (idx, &nd) in states[ri].route.nodes.iter().enumerate() { if nd < states[ri].route.pos.len() { states[ri].route.pos[nd] = idx; } }
                                states[ri].recompute_times_from(0);
                            }
                            return true;
                        }
                    }

                }
                // advance indices
                let mut carry = true;
                for t in 0..idxs.len() {
                    if carry {
                        idxs[t] += 1;
                        if idxs[t] >= candidates_per_violation[t].len() { idxs[t] = 0; carry = true; } else { carry = false; }
                    }
                }
                if carry { break; }
            }
        }
    }

    false
}
}

pub struct LocalSearch {
    pub time_limit_ms: u64,       // Phase 2: configurable time limit
    pub neighborhood_size: usize, // Phase 2: configurable neighborhood
    pub improvement_threshold: i64, // Phase 2: configurable threshold
}

impl Default for LocalSearch {
    fn default() -> Self {
        Self {
            time_limit_ms: 20,
            neighborhood_size: 8,
            improvement_threshold: -5,
        }
    }
}

impl LocalSearch {
    pub fn new(time_limit_ms: u64, neighborhood_size: usize) -> Self {
        Self {
            time_limit_ms,
            neighborhood_size,
            improvement_threshold: -5,
        }
    }

    pub fn optimize(state: &mut TIGState, izs: &IZS) {
        Self::default().optimize_with_config(state, izs)
    }

    /// Phase 2: Optimize with configurable parameters
    pub fn optimize_with_config(&self, state: &mut TIGState, izs: &IZS) {
        // maintain backward compatibility: run optimizer and ignore stats
        let _ = self.optimize_with_stats(state, izs);
    }

    /// New: run optimization and return stats for diagnostics
    pub fn optimize_with_stats(&self, state: &mut TIGState, _izs: &IZS) -> LocalSearchStats {
        let start = Instant::now();
        let n = state.route.len();
        let mut stats = LocalSearchStats::new();
        stats.initial_cost = state.total_cost();
        if n < 5 { stats.duration = start.elapsed(); stats.final_cost = stats.initial_cost; return stats; }

        // Build delta tables once per invocation to enable O(1) deltas and KNN pruning
        let tables = crate::delta::DeltaTables::from_state(state);

        loop {
            if start.elapsed().as_millis() as u64 > self.time_limit_ms { break; }
            let mut improved = false;

            if self.try_relocate(state, &tables, &start, &mut stats) { improved = true; }
            if improved { continue; }

            if self.try_swap(state, &tables, &start, &mut stats) { improved = true; }
            if improved { continue; }

            if self.try_2opt(state, &tables, &start, &mut stats) { improved = true; }
            if improved { continue; }

            if self.try_cross(state, &tables, &start, &mut stats) { improved = true; }
            if improved { continue; }

            if !improved { break; }
        }

        stats.duration = start.elapsed();
        stats.final_cost = state.total_cost();
        stats.feasible = state.is_feasible();
        stats
    }

    /// Phase 2: Optimize across multiple routes and apply inter-route two-opt*
    pub fn optimize_multi_with_config(&self, states: &mut [TIGState], _izs: &IZS) {
        // Default behavior delegates to cached-aware variant with no precomputed tables
        self.optimize_multi_with_config_cached(states, None, _izs);
    }

    /// Like `optimize_multi_with_config` but accepts an optional slice of precomputed `DeltaTables`
    /// (borrowed) matching `states`. If a given entry is `Some(&DeltaTables)`, that table will be
    /// used instead of recomputing the tables for that state.
    pub fn optimize_multi_with_config_cached(&self, states: &mut [TIGState], precomputed: Option<&[Option<&crate::delta::DeltaTables>]>, _izs: &IZS) {
        let start = std::time::Instant::now();
        let time_limit = std::time::Duration::from_millis(self.time_limit_ms);

        // Per-route intra-route improvement
        for (idx, st) in states.iter_mut().enumerate() {
            if start.elapsed() > time_limit { break; }

            // Check for precomputed table
            let mut used_pre = false;
            if let Some(pre) = precomputed {
                if idx < pre.len() {
                    if let Some(dt_ref) = pre[idx] {
                        crate::local_search::controller::local_search(st, dt_ref);
                        used_pre = true;
                    }
                }
            }

            if used_pre { continue; }

            // Fallback: build fresh tables and use them
            let tables = crate::delta::DeltaTables::from_state(st);
            crate::local_search::controller::local_search(st, &tables);
        }

        // Two-opt* phase uses a representative table (from first state) for prefiltering
        while start.elapsed() <= time_limit {
            let tables = if !states.is_empty() { crate::delta::DeltaTables::from_state(&states[0]) } else { break };
            let improved = crate::local_search::controller::try_two_opt_star_multi(states, &tables);
            if !improved { break; }
        }

        // Try inter-route ejection-chains using representative delta tables
        while start.elapsed() <= time_limit {
            let tables = if !states.is_empty() { crate::delta::DeltaTables::from_state(&states[0]) } else { break };
            let improved = crate::local_search::ejection_chain::try_ejection_chain_multi(states, &tables);
            if !improved { break; }
        }

        // After ejection-chain, try time-window repair across routes repeatedly
        while start.elapsed() <= time_limit {
            let tables = if !states.is_empty() { crate::delta::DeltaTables::from_state(&states[0]) } else { break };
            let improved = crate::local_search::time_window_repair::try_time_window_repair_multi(states, &tables);
            if !improved { break; }
        }

        // Try batched repair (coordinated reinsertions) if violations persist
        while start.elapsed() <= time_limit {
            let tables = if !states.is_empty() { crate::delta::DeltaTables::from_state(&states[0]) } else { break };
            let improved = crate::local_search::batched_time_window_repair::try_batched_time_window_repair(states, &tables);
            if !improved { break; }
        }
    }

    #[allow(dead_code)]
    fn relocate_cost_delta(state: &TIGState, from: usize, to: usize) -> i64 {
        state.delta_relocate(from, to) as i64
    }

    fn apply_relocate(state: &mut TIGState, from: usize, to: usize) {
        let node = state.route.remove(from);
        state.route.insert(to, node);
    }

    fn apply_swap(state: &mut TIGState, i: usize, j: usize) {
        state.route.swap(i, j);
    }

    fn apply_2opt(state: &mut TIGState, i: usize, k: usize) {
        // reverse the segment between (i+1) and k inclusive
        if i + 1 >= k { return; }
        state.route.reverse(i + 1, k);
    }

    fn apply_cross(state: &mut TIGState, i: usize, j: usize) {
        // cross-exchange implemented as reversing segment between i+1 and j
        if i + 1 >= j { return; }
        state.route.reverse(i + 1, j);
    }

    fn try_relocate(&self, state: &mut TIGState, tables: &crate::delta::DeltaTables, start: &Instant, stats: &mut LocalSearchStats) -> bool {
        let n = state.route.len();
        if n < 4 { return false; }

        // iterate over internal positions (skip depot at 0 and last)
        for i in 1..(n - 1) {
            if start.elapsed().as_millis() as u64 > self.time_limit_ms { return false; }

            let node = state.route.nodes[i];
            // Use KNN neighbors of the node as candidate insertion zones
            let mut considered = 0usize;
            if node < tables.neighbors.len() {
                for &nb in &tables.neighbors[node] {
                    if considered >= self.neighborhood_size { break; }
                    // map neighbor node to position in this route
                    if nb >= state.route.pos.len() { continue; }
                    let pos = state.route.pos[nb];
                    if pos == usize::MAX { continue; }
                    // normalize insertion index: insert after pos (i.e., at pos or pos+1)
                    let candidates = [pos, pos + 1];
                    for &to in &candidates {
                        if to == i || to == i - 1 { continue; }
                        if to > n { continue; }
                        let delta = tables.delta_relocate(state, i, to);
                        if delta < 0 {
                            // fast feasibility gating
                            if !state.simulate_relocate_feasible(i, to) { continue; }
                            Self::apply_relocate(state, i, to);
                            state.repair_times();
                            stats.moves += 1;
                            stats.last_operator = Some("relocate".to_string());
                            return true;
                        }
                    }
                    considered += 1;
                }
            } else {
                // fallback: scan local neighborhood positions
                let j_end = (i + self.neighborhood_size).min(n - 2);
                for j in (i + 1)..=j_end {
                    if i == j { continue; }
                    let delta = tables.delta_relocate(state, i, j);
                    if delta < 0 {
                        if !state.simulate_relocate_feasible(i, j) { continue; }
                        Self::apply_relocate(state, i, j);
                        state.repair_times();
                        stats.moves += 1;
                        stats.last_operator = Some("relocate".to_string());
                        return true;
                    }
                }
            }
        }
        false
    }

    fn try_swap(&self, state: &mut TIGState, tables: &crate::delta::DeltaTables, start: &Instant, stats: &mut LocalSearchStats) -> bool {
        let n = state.route.len();
        if n < 4 { return false; }

        for i in 1..(n - 1) {
            if start.elapsed().as_millis() as u64 > self.time_limit_ms { return false; }
            let node = state.route.nodes[i];
            let mut considered = 0usize;
            if node < tables.neighbors.len() {
                for &nb in &tables.neighbors[node] {
                    if considered >= self.neighborhood_size { break; }
                    if nb >= state.route.pos.len() { continue; }
                    let j = state.route.pos[nb];
                    if j == usize::MAX || j == i { continue; }
                    let delta = tables.delta_swap(state, i, j);
                    if delta < 0 {
                        if !state.simulate_swap_feasible(i, j) { continue; }
                        Self::apply_swap(state, i, j);
                        state.repair_times();
                        stats.moves += 1;
                        stats.last_operator = Some("swap".to_string());
                        return true;
                    }
                    considered += 1;
                }
            } else {
                for j in i + 1..n - 1 {
                    let delta = tables.delta_swap(state, i, j);
                    if delta < 0 {
                        if !state.simulate_swap_feasible(i, j) { continue; }
                        Self::apply_swap(state, i, j);
                        state.repair_times();
                        stats.moves += 1;
                        stats.last_operator = Some("swap".to_string());
                        return true;
                    }
                }
            }
        }
        false
    }

    fn try_2opt(&self, state: &mut TIGState, tables: &crate::delta::DeltaTables, start: &Instant, stats: &mut LocalSearchStats) -> bool {
        let n = state.route.len();
        if n < 4 { return false; }
        for i in 1..(n - 1) {
            if start.elapsed().as_millis() as u64 > self.time_limit_ms { return false; }
            let node = state.route.nodes[i];
            let mut considered = 0usize;
            if node < tables.neighbors.len() {
                for &nb in &tables.neighbors[node] {
                    if considered >= self.neighborhood_size { break; }
                    if nb >= state.route.pos.len() { continue; }
                    let k = state.route.pos[nb];
                    if k == usize::MAX || k <= i + 0 { continue; }
                    let delta = tables.delta_two_opt(state, i, k);
                    if delta < 0 {
                        if !state.simulate_two_opt_feasible(i, k) { continue; }
                        Self::apply_2opt(state, i, k);
                        state.repair_times();
                        stats.moves += 1;
                        stats.last_operator = Some("2opt".to_string());
                        return true;
                    }
                    considered += 1;
                }
            } else {
                for k in i + 1..n - 1 {
                    let delta = tables.delta_two_opt(state, i, k);
                    if delta < 0 {
                        if !state.simulate_two_opt_feasible(i, k) { continue; }
                        Self::apply_2opt(state, i, k);
                        state.repair_times();
                        stats.moves += 1;
                        stats.last_operator = Some("2opt".to_string());
                        return true;
                    }
                }
            }
        }
        false
    }

    fn try_cross(&self, state: &mut TIGState, tables: &crate::delta::DeltaTables, start: &Instant, stats: &mut LocalSearchStats) -> bool {
        let n = state.route.len();
        if n < 4 { return false; }
        for i in 1..(n - 1) {
            if start.elapsed().as_millis() as u64 > self.time_limit_ms { return false; }
            let node = state.route.nodes[i];
            let mut considered = 0usize;
            if node < tables.neighbors.len() {
                for &nb in &tables.neighbors[node] {
                    if considered >= self.neighborhood_size { break; }
                    if nb >= state.route.pos.len() { continue; }
                    let j = state.route.pos[nb];
                    if j == usize::MAX || j <= i { continue; }
                    let delta = tables.delta_two_opt(state, i, j);
                    if delta < 0 {
                        if !state.simulate_two_opt_feasible(i, j) { continue; }
                        Self::apply_cross(state, i, j);
                        state.repair_times();
                        stats.moves += 1;
                        stats.last_operator = Some("cross".to_string());
                        return true;
                    }
                    considered += 1;
                }
            } else {
                for j in i + 1..n - 1 {
                    let delta = tables.delta_two_opt(state, i, j);
                    if delta < 0 {
                        if !state.simulate_two_opt_feasible(i, j) { continue; }
                        Self::apply_cross(state, i, j);
                        state.repair_times();
                        stats.moves += 1;
                        stats.last_operator = Some("cross".to_string());
                        return true;
                    }
                }
            }
        }
        false
    }

}

/// Lightweight stats for diagnostics
pub struct LocalSearchStats {
    pub moves: usize,
    pub last_operator: Option<String>,
    pub duration: std::time::Duration,
    pub initial_cost: i64,
    pub final_cost: i64,
    pub feasible: bool,
}

impl LocalSearchStats {
    pub fn new() -> Self {
        Self { moves: 0, last_operator: None, duration: std::time::Duration::ZERO, initial_cost: 0, final_cost: 0, feasible: true }
    }
}
