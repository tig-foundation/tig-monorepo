use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;

#[inline(always)]
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    simple_solver::solve_challenge(challenge, save_solution, hyperparameters)
}

mod utils {

    #[inline(always)]
    pub fn calculate_route_demands(routes: &Vec<Vec<usize>>, demands: &Vec<i32>) -> Vec<i32> {
        let mut out = Vec::with_capacity(routes.len());
        unsafe {
            let dem = demands.as_slice();
            for route in routes {
                let len = route.len();
                if len <= 2 {
                    out.push(0);
                    continue;
                }
                let mut s = 0i32;
                let r = route.as_slice();
                let mut idx = 1usize;
                let end = len - 1;
                while idx < end {
                    let n = *r.get_unchecked(idx);
                    s += *dem.get_unchecked(n);
                    idx += 1;
                }
                out.push(s);
            }
        }
        out
    }

    #[inline(always)]
    pub fn find_best_insertion(
        route: &Vec<usize>,
        remaining_nodes: &[usize],
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> Option<(usize, usize)> {
        let lambda = 1;

        let mut best_c2: Option<i32> = None;
        let mut best: Option<(usize, usize)> = None;

        let dm = distance_matrix;
        let rt = &ready_times[..];
        let dt = &due_times[..];
        let len = route.len();

        unsafe {
            let row0 = dm.get_unchecked(0);
            for &insert_node in remaining_nodes {
                let base_c2 = lambda * *row0.get_unchecked(insert_node);
                let row_insert = dm.get_unchecked(insert_node);
                let ready_ins = *rt.get_unchecked(insert_node);
                let due_ins = *dt.get_unchecked(insert_node);

                let mut curr_time: i32 = 0;
                let mut curr_node: usize = 0;

                for pos in 1..len {
                    let next_node = *route.get_unchecked(pos);
                    let row_curr = dm.get_unchecked(curr_node);

                    let travel_to_insert = *row_curr.get_unchecked(insert_node);
                    let travel_to_next = *row_curr.get_unchecked(next_node);
                    let mut new_arrival_time = curr_time + travel_to_insert;
                    if new_arrival_time < ready_ins {
                        new_arrival_time = ready_ins;
                    }

                    if new_arrival_time > due_ins {
                        let mut tmp = curr_time + travel_to_next;
                        let r_next = *rt.get_unchecked(next_node);
                        if tmp < r_next {
                            tmp = r_next;
                        }
                        curr_time = tmp + service_time;
                        curr_node = next_node;
                        continue;
                    }

                    let c11 =
                        travel_to_insert + *row_insert.get_unchecked(next_node) - travel_to_next;

                    let c1 = -c11;
                    let c2 = base_c2 + c1;

                    let c2_is_better = match best_c2 {
                        None => true,
                        Some(x) => c2 > x,
                    };

                    let depart_after_insert = new_arrival_time + service_time;
                    let next_due_ok = if next_node == 0 {
                        depart_after_insert + *row_insert.get_unchecked(0) <= *dt.get_unchecked(0)
                    } else {
                        depart_after_insert + *row_insert.get_unchecked(next_node)
                            <= *dt.get_unchecked(next_node)
                    };

                    if c2_is_better
                        && next_due_ok
                        && is_feasible(
                            route,
                            distance_matrix,
                            service_time,
                            ready_times,
                            due_times,
                            insert_node,
                            depart_after_insert,
                            pos,
                        )
                    {
                        best_c2 = Some(c2);
                        best = Some((insert_node, pos));
                    }

                    let mut tmp = curr_time + travel_to_next;
                    let r_next2 = *rt.get_unchecked(next_node);
                    if tmp < r_next2 {
                        tmp = r_next2;
                    }
                    curr_time = tmp + service_time;
                    curr_node = next_node;
                }
            }
        }
        best
    }

    #[inline(always)]
    pub fn is_feasible(
        route: &Vec<usize>,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
        mut curr_node: usize,
        mut curr_time: i32,
        start_pos: usize,
    ) -> bool {
        let dm = distance_matrix;
        let rt = ready_times.as_slice();
        let dt = due_times.as_slice();
        let len = route.len();
        let mut pos = start_pos;
        unsafe {
            let r = route.as_slice();
            while pos < len {
                let next_node = *r.get_unchecked(pos);
                let row_curr = dm.get_unchecked(curr_node);
                curr_time += *row_curr.get_unchecked(next_node);
                if next_node != 0 {
                    if curr_time > *dt.get_unchecked(next_node) {
                        return false;
                    }
                    let ready = *rt.get_unchecked(next_node);
                    if curr_time < ready {
                        curr_time = ready;
                    }
                    curr_time += service_time;
                }
                curr_node = next_node;
                pos += 1;
            }
        }
        if curr_time > dt[0] {
            return false;
        }
        true
    }

    #[inline(always)]
    pub fn find_best_insertion_in_route(
        route: &Vec<usize>,
        node: usize,
        _demands: &Vec<i32>,
        _max_capacity: i32,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> Option<(usize, i32)> {
        let len = route.len();
        if len < 2 {
            return None;
        }
        let mut best_pos = None;
        let mut best_delta = i32::MAX;

        unsafe {
            let dm = distance_matrix;
            let row_node = dm.get_unchecked(node);
            let rt = ready_times.as_slice();
            let dt = due_times.as_slice();
            let r = route.as_slice();

            let mut curr_time: i32 = 0;
            let mut curr_node: usize = 0;

            let mut pos = 1usize;
            while pos < len {
                let next_node = *r.get_unchecked(pos);
                let row_curr = dm.get_unchecked(curr_node);

                let travel_to_next = *row_curr.get_unchecked(next_node);
                let travel_to_insert = *row_curr.get_unchecked(node);

                let mut new_arrival_time = curr_time + travel_to_insert;
                let ready_ins = *rt.get_unchecked(node);
                if new_arrival_time < ready_ins {
                    new_arrival_time = ready_ins;
                }

                if new_arrival_time <= *dt.get_unchecked(node) {
                    let mut next_due_ok = true;
                    if pos != len - 1 {
                        let depart = new_arrival_time + service_time;
                        let arr_next = depart + *row_node.get_unchecked(next_node);
                        if arr_next > *dt.get_unchecked(next_node) {
                            next_due_ok = false;
                        }
                    }
                    if next_due_ok {
                        let delta =
                            travel_to_insert + *row_node.get_unchecked(next_node) - travel_to_next;
                        if delta < best_delta {
                            let suffix_ok = if pos == len - 1 {
                                let departure = new_arrival_time + service_time;
                                let row_ins = dm.get_unchecked(node);
                                departure + *row_ins.get_unchecked(0) <= *dt.get_unchecked(0)
                            } else {
                                is_feasible(
                                    route,
                                    dm,
                                    service_time,
                                    ready_times,
                                    due_times,
                                    node,
                                    new_arrival_time + service_time,
                                    pos,
                                )
                            };
                            if suffix_ok {
                                best_delta = delta;
                                best_pos = Some(pos);
                            }
                        }
                    }
                }

                let mut tmp = curr_time + travel_to_next;
                let r_next = *rt.get_unchecked(next_node);
                if tmp < r_next {
                    tmp = r_next;
                }
                curr_time = tmp + service_time;
                curr_node = next_node;

                pos += 1;
            }
        }

        best_pos.map(|p| (p, best_delta))
    }

    #[inline(always)]
    pub fn find_best_insertion_pair_in_route(
        route: &Vec<usize>,
        node_a: usize,
        node_b: usize,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> Option<(usize, i32)> {
        if node_a == 0 || node_b == 0 || route.len() < 3 {
            return None;
        }
        let mut best_pos = None;
        let mut best_delta = i32::MAX;

        unsafe {
            let dm = distance_matrix;
            let r = route.as_slice();
            let row_a = dm.get_unchecked(node_a);
            let row_b = dm.get_unchecked(node_b);
            let rt = ready_times.as_slice();
            let dt = due_times.as_slice();
            let len = route.len();

            let mut curr_time: i32 = 0;
            let mut curr_node: usize = 0;

            let mut pos = 1usize;
            while pos < len {
                let next = *r.get_unchecked(pos);
                let row_prev = dm.get_unchecked(curr_node);

                let mut arr_a = curr_time + *row_prev.get_unchecked(node_a);
                let ready_a = *rt.get_unchecked(node_a);
                if arr_a < ready_a {
                    arr_a = ready_a;
                }
                if arr_a <= *dt.get_unchecked(node_a) {
                    let mut arr_b = arr_a + service_time + *row_a.get_unchecked(node_b);
                    let ready_b = *rt.get_unchecked(node_b);
                    if arr_b < ready_b {
                        arr_b = ready_b;
                    }
                    if arr_b <= *dt.get_unchecked(node_b) {
                        let delta = *row_prev.get_unchecked(node_a)
                            + *row_a.get_unchecked(node_b)
                            + *row_b.get_unchecked(next)
                            - *row_prev.get_unchecked(next);

                        if delta < best_delta {
                            let immediate_ok = if pos == len - 1 {
                                let back = *row_b.get_unchecked(0);
                                arr_b + service_time + back <= *dt.get_unchecked(0)
                            } else {
                                let arrival_next =
                                    arr_b + service_time + *row_b.get_unchecked(next);
                                arrival_next <= *dt.get_unchecked(next)
                            };

                            if immediate_ok {
                                if pos == len - 1 {
                                    best_delta = delta;
                                    best_pos = Some(pos);
                                } else if is_feasible(
                                    route,
                                    dm,
                                    service_time,
                                    ready_times,
                                    due_times,
                                    node_b,
                                    arr_b + service_time,
                                    pos,
                                ) {
                                    best_delta = delta;
                                    best_pos = Some(pos);
                                }
                            }
                        }
                    }
                }

                let mut tmp = curr_time + *row_prev.get_unchecked(next);
                let ready_next = *rt.get_unchecked(next);
                if tmp < ready_next {
                    tmp = ready_next;
                }
                curr_time = tmp + service_time;
                curr_node = next;

                pos += 1;
            }
        }

        best_pos.map(|p| (p, best_delta))
    }

    #[inline(always)]
    pub fn calculate_route_distance(route: &Vec<usize>, distance_matrix: &Vec<Vec<i32>>) -> i32 {
        let len = route.len();
        if len < 2 {
            return 0;
        }
        let mut distance: i32 = 0;
        let dm = distance_matrix;
        unsafe {
            let r = route.as_slice();
            let mut i = 0usize;
            while i + 1 < len {
                let a = *r.get_unchecked(i);
                let b = *r.get_unchecked(i + 1);
                let row_a = dm.get_unchecked(a);
                distance += *row_a.get_unchecked(b);
                i += 1;
            }
        }
        distance
    }

    #[inline(always)]
    pub fn apply_efficient_2opt(
        route: &Vec<usize>,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> Vec<usize> {
        if route.len() < 4 {
            return route.clone();
        }

        let mut best_route = route.clone();
        let mut improved = true;
        let mut iteration = 0;
        let max_iterations = if route.len() > 80 {
            28
        } else {
            (route.len() / 2).min(24)
        };

        while improved && iteration < max_iterations {
            improved = false;
            iteration += 1;

            let dm = distance_matrix;
            let len = best_route.len();
            let end = len - 1;
            for i in 1..end - 1 {
                let prev = best_route[i - 1];
                let a = best_route[i];
                let row_prev = unsafe { dm.get_unchecked(prev) };
                let row_a = unsafe { dm.get_unchecked(a) };
                for j in i + 2..end {
                    let b = best_route[j];
                    let next = best_route[j + 1];
                    let delta = unsafe {
                        let row_b = dm.get_unchecked(b);
                        (*row_prev.get_unchecked(b) + *row_a.get_unchecked(next))
                            - (*row_prev.get_unchecked(a) + *row_b.get_unchecked(next))
                    };
                    if delta >= 0 {
                        continue;
                    }

                    best_route[i..=j].reverse();

                    if is_route_time_feasible_fast(
                        &best_route,
                        dm,
                        service_time,
                        ready_times,
                        due_times,
                    ) {
                        improved = true;
                        break;
                    } else {
                        best_route[i..=j].reverse();
                    }
                }
                if improved {
                    break;
                }
            }
        }

        best_route
    }

    #[inline(always)]
    pub fn is_route_time_feasible_fast(
        route: &Vec<usize>,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> bool {
        let len = route.len();
        if len < 3 || route[0] != 0 || route[len - 1] != 0 {
            return false;
        }
        let dm = distance_matrix;
        let rt = &ready_times[..];
        let dt = &due_times[..];
        let mut curr_time: i32 = 0;
        let mut curr_node: usize = 0;

        if len > 2 {
            unsafe {
                let r = route.as_slice();
                for idx in 1..len - 1 {
                    let next_node = *r.get_unchecked(idx);
                    let row_curr = dm.get_unchecked(curr_node);
                    curr_time += *row_curr.get_unchecked(next_node);
                    if curr_time > *dt.get_unchecked(next_node) {
                        return false;
                    }
                    let rdy = *rt.get_unchecked(next_node);
                    if curr_time < rdy {
                        curr_time = rdy;
                    }
                    curr_time += service_time;
                    curr_node = next_node;
                }
            }
        }

        unsafe {
            let row_curr = dm.get_unchecked(curr_node);
            curr_time += *row_curr.get_unchecked(0);
            curr_time <= *dt.get_unchecked(0)
        }
    }

    #[inline(always)]
    pub fn is_route_time_feasible_strict(
        route: &Vec<usize>,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> bool {
        is_route_time_feasible_fast(route, distance_matrix, service_time, ready_times, due_times)
    }

    #[inline(always)]
    pub fn apply_size_filtered_local_search(
        route: &Vec<usize>,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> Vec<usize> {
        if route.len() <= 4 {
            return route.clone();
        }

        apply_smart_local_search(route, distance_matrix, service_time, ready_times, due_times)
    }

    #[inline(always)]
    pub fn apply_smart_local_search(
        route: &Vec<usize>,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> Vec<usize> {
        if route.len() <= 3 {
            return route.clone();
        }

        let mut current_route =
            apply_efficient_2opt(route, distance_matrix, service_time, ready_times, due_times);

        if route.len() > 6 {
            current_route = apply_limited_or_opt(
                &current_route,
                distance_matrix,
                service_time,
                ready_times,
                due_times,
            );
        }

        current_route
    }

    #[inline(always)]
    fn apply_limited_or_opt(
        route: &Vec<usize>,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> Vec<usize> {
        if route.len() < 4 {
            return route.clone();
        }

        let dm = distance_matrix;
        let mut best_route = route.clone();
        let mut best_distance = calculate_route_distance(&best_route, dm);
        let mut improved = true;
        let mut iteration = 0;
        let max_iterations = if best_route.len() > 100 { 4 } else { 6 };

        let mut candidate: Vec<usize> = Vec::with_capacity(best_route.len());

        while improved && iteration < max_iterations {
            improved = false;
            iteration += 1;

            let max_seg = if best_route.len() > 80 { 2 } else { 3 };
            for segment_size in 1..=max_seg {
                let len = best_route.len();
                for i in 1..len - segment_size {
                    if i + segment_size >= len - 1 {
                        continue;
                    }

                    let first = best_route[i];
                    let last = best_route[i + segment_size - 1];
                    let prev_before = best_route[i - 1];
                    let after_segment = best_route[i + segment_size];

                    for insert_pos in 1..len {
                        if insert_pos >= i && insert_pos <= i + segment_size {
                            continue;
                        }

                        let prev_ins = best_route[insert_pos - 1];
                        let succ_ins = best_route[insert_pos];

                        let (delta_remove_segment, delta_insert_segment) = unsafe {
                            let row_prev_before = dm.get_unchecked(prev_before);
                            let row_last = dm.get_unchecked(last);
                            let row_prev_ins = dm.get_unchecked(prev_ins);
                            (
                                *row_prev_before.get_unchecked(after_segment)
                                    - *row_prev_before.get_unchecked(first)
                                    - *row_last.get_unchecked(after_segment),
                                *row_prev_ins.get_unchecked(first)
                                    + *row_last.get_unchecked(succ_ins)
                                    - *row_prev_ins.get_unchecked(succ_ins),
                            )
                        };
                        if delta_remove_segment + delta_insert_segment >= 0 {
                            continue;
                        }

                        candidate.clear();
                        if insert_pos < i {
                            candidate.extend_from_slice(&best_route[..insert_pos]);
                            candidate.extend_from_slice(&best_route[i..i + segment_size]);
                            candidate.extend_from_slice(&best_route[insert_pos..i]);
                            candidate.extend_from_slice(&best_route[i + segment_size..]);
                        } else {
                            candidate.extend_from_slice(&best_route[..i]);
                            candidate.extend_from_slice(&best_route[i + segment_size..insert_pos]);
                            candidate.extend_from_slice(&best_route[i..i + segment_size]);
                            candidate.extend_from_slice(&best_route[insert_pos..]);
                        }

                        if candidate.len() >= 3
                            && candidate[0] == 0
                            && candidate[candidate.len() - 1] == 0
                            && is_route_time_feasible_fast(
                                &candidate,
                                dm,
                                service_time,
                                ready_times,
                                due_times,
                            )
                        {
                            let new_distance =
                                best_distance + delta_remove_segment + delta_insert_segment;
                            if new_distance < best_distance {
                                best_distance = new_distance;
                                best_route.clear();
                                best_route.extend_from_slice(&candidate);
                                candidate.clear();
                                improved = true;
                                break;
                            }
                        }
                    }
                    if improved {
                        break;
                    }
                }
                if improved {
                    break;
                }
            }
        }

        best_route
    }

    #[inline(always)]
    pub fn check_swap_time_feasibility(
        route: &Vec<usize>,
        pos: usize,
        new_node: usize,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> bool {
        if pos == 0 || pos >= route.len() - 1 {
            return false;
        }

        let prev_node = route[pos - 1];
        let next_node = route[pos + 1];

        let mut curr_time = 0i32;
        unsafe {
            let r = route.as_slice();
            let dm = distance_matrix;
            let rt = &ready_times[..];
            let dt = &due_times[..];

            let mut i = 1usize;
            while i < pos {
                let from = *r.get_unchecked(i - 1);
                let to = *r.get_unchecked(i);
                let row_from = dm.get_unchecked(from);
                curr_time += *row_from.get_unchecked(to);
                let rdy = *rt.get_unchecked(to);
                if curr_time < rdy {
                    curr_time = rdy;
                }
                curr_time += service_time;
                i += 1;
            }

            let row_prev = dm.get_unchecked(prev_node);
            curr_time += *row_prev.get_unchecked(new_node);
            if curr_time > *dt.get_unchecked(new_node) {
                return false;
            }

            let rdy_new = *rt.get_unchecked(new_node);
            if curr_time < rdy_new {
                curr_time = rdy_new;
            }
            curr_time += service_time;

            let row_new = dm.get_unchecked(new_node);
            curr_time += *row_new.get_unchecked(next_node);
            if curr_time > *dt.get_unchecked(next_node) {
                return false;
            }
        }

        true
    }

    #[inline(always)]
    pub fn try_two_opt_star(
        routes: &mut Vec<Vec<usize>>,
        route1_idx: usize,
        pos1: usize,
        route2_idx: usize,
        pos2: usize,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
        route_distances: &mut Vec<i32>,
        total_distance: &mut i32,
        route_demands: &mut Vec<i32>,
        demands: &Vec<i32>,
        max_capacity: i32,
    ) -> bool {
        if pos1 == 0
            || pos2 == 0
            || pos1 >= routes[route1_idx].len() - 1
            || pos2 >= routes[route2_idx].len() - 1
        {
            return false;
        }

        let r1 = &routes[route1_idx];
        let r2 = &routes[route2_idx];

        let prev1 = r1[pos1 - 1];
        let curr1 = r1[pos1];
        let prev2 = r2[pos2 - 1];
        let curr2 = r2[pos2];

        let dm = distance_matrix;
        let (old_dist, new_dist) = unsafe {
            let row_prev1 = dm.get_unchecked(prev1);
            let row_prev2 = dm.get_unchecked(prev2);
            (
                *row_prev1.get_unchecked(curr1) + *row_prev2.get_unchecked(curr2),
                *row_prev1.get_unchecked(curr2) + *row_prev2.get_unchecked(curr1),
            )
        };

        if new_dist >= old_dist {
            return false;
        }

        let (demand1, demand2) = {
            unsafe {
                let dem = demands.as_slice();
                let r1s = r1.as_slice();
                let r2s = r2.as_slice();
                let len1 = r1s.len();
                let len2 = r2s.len();

                let mut d1_prefix = 0i32;
                let mut i = 1usize;
                while i < pos1 {
                    let node = *r1s.get_unchecked(i);
                    d1_prefix += *dem.get_unchecked(node);
                    i += 1;
                }
                let mut d2_prefix = 0i32;
                let mut j = 1usize;
                while j < pos2 {
                    let node = *r2s.get_unchecked(j);
                    d2_prefix += *dem.get_unchecked(node);
                    j += 1;
                }

                let mut d1_tail = 0i32;
                i = pos1;
                while i + 1 < len1 {
                    let node = *r1s.get_unchecked(i);
                    d1_tail += *dem.get_unchecked(node);
                    i += 1;
                }
                let mut d2_tail = 0i32;
                j = pos2;
                while j + 1 < len2 {
                    let node = *r2s.get_unchecked(j);
                    d2_tail += *dem.get_unchecked(node);
                    j += 1;
                }

                (d1_prefix + d2_tail, d2_prefix + d1_tail)
            }
        };
        if demand1 > max_capacity || demand2 > max_capacity {
            return false;
        }

        let cap1 = pos1 + (r2.len() - pos2);
        let cap2 = pos2 + (r1.len() - pos1);
        let mut new_route1 = Vec::with_capacity(cap1);
        let mut new_route2 = Vec::with_capacity(cap2);

        new_route1.extend_from_slice(&r1[..pos1]);
        new_route1.extend_from_slice(&r2[pos2..]);

        new_route2.extend_from_slice(&r2[..pos2]);
        new_route2.extend_from_slice(&r1[pos1..]);

        if !is_route_time_feasible_fast(
            &new_route1,
            distance_matrix,
            service_time,
            ready_times,
            due_times,
        ) || !is_route_time_feasible_fast(
            &new_route2,
            distance_matrix,
            service_time,
            ready_times,
            due_times,
        ) {
            return false;
        }

        let old_total = route_distances[route1_idx] + route_distances[route2_idx];
        let new_dist1 = calculate_route_distance(&new_route1, distance_matrix);
        let new_dist2 = calculate_route_distance(&new_route2, distance_matrix);
        let new_total = new_dist1 + new_dist2;

        if new_total < old_total {
            routes[route1_idx] = new_route1;
            routes[route2_idx] = new_route2;
            route_distances[route1_idx] = new_dist1;
            route_distances[route2_idx] = new_dist2;
            route_demands[route1_idx] = demand1;
            route_demands[route2_idx] = demand2;
            *total_distance = *total_distance - old_total + new_total;
            return true;
        }

        false
    }

    #[inline(always)]
    pub fn try_2_2_cross_exchange(
        routes: &mut Vec<Vec<usize>>,
        route1_idx: usize,
        pos1: usize,
        route2_idx: usize,
        pos2: usize,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
        route_distances: &mut Vec<i32>,
        total_distance: &mut i32,
        route_demands: &mut Vec<i32>,
        demands: &Vec<i32>,
        max_capacity: i32,
    ) -> bool {
        let r1 = &routes[route1_idx];
        let r2 = &routes[route2_idx];

        if pos1 + 1 >= r1.len() - 1 || pos2 + 1 >= r2.len() - 1 {
            return false;
        }

        let node1a = r1[pos1];
        let node1b = r1[pos1 + 1];
        let node2a = r2[pos2];
        let node2b = r2[pos2 + 1];

        let dem = demands.as_slice();
        let demand_diff1 = unsafe {
            *dem.get_unchecked(node2a) + *dem.get_unchecked(node2b)
                - *dem.get_unchecked(node1a)
                - *dem.get_unchecked(node1b)
        };
        let demand_diff2 = unsafe {
            *dem.get_unchecked(node1a) + *dem.get_unchecked(node1b)
                - *dem.get_unchecked(node2a)
                - *dem.get_unchecked(node2b)
        };

        if route_demands[route1_idx] + demand_diff1 > max_capacity
            || route_demands[route2_idx] + demand_diff2 > max_capacity
        {
            return false;
        }

        let prev1 = r1[pos1 - 1];
        let next1 = r1[pos1 + 2];
        let prev2 = r2[pos2 - 1];
        let next2 = r2[pos2 + 2];
        let dm = distance_matrix;
        let (delta1, delta2) = unsafe {
            let row_prev1 = dm.get_unchecked(prev1);
            let row_prev2 = dm.get_unchecked(prev2);
            let row_n1a = dm.get_unchecked(node1a);
            let row_n1b = dm.get_unchecked(node1b);
            let row_n2a = dm.get_unchecked(node2a);
            let row_n2b = dm.get_unchecked(node2b);

            (
                *row_prev1.get_unchecked(node2a)
                    + *row_n2a.get_unchecked(node2b)
                    + *row_n2b.get_unchecked(next1)
                    - (*row_prev1.get_unchecked(node1a)
                        + *row_n1a.get_unchecked(node1b)
                        + *row_n1b.get_unchecked(next1)),
                *row_prev2.get_unchecked(node1a)
                    + *row_n1a.get_unchecked(node1b)
                    + *row_n1b.get_unchecked(next2)
                    - (*row_prev2.get_unchecked(node2a)
                        + *row_n2a.get_unchecked(node2b)
                        + *row_n2b.get_unchecked(next2)),
            )
        };

        if delta1 + delta2 >= 0 {
            return false;
        }

        let mut new_route1 = Vec::with_capacity(r1.len());
        let mut new_route2 = Vec::with_capacity(r2.len());

        new_route1.extend_from_slice(&r1[..pos1]);
        new_route1.push(node2a);
        new_route1.push(node2b);
        new_route1.extend_from_slice(&r1[pos1 + 2..]);

        new_route2.extend_from_slice(&r2[..pos2]);
        new_route2.push(node1a);
        new_route2.push(node1b);
        new_route2.extend_from_slice(&r2[pos2 + 2..]);

        if !is_route_time_feasible_fast(
            &new_route1,
            distance_matrix,
            service_time,
            ready_times,
            due_times,
        ) || !is_route_time_feasible_fast(
            &new_route2,
            distance_matrix,
            service_time,
            ready_times,
            due_times,
        ) {
            return false;
        }

        let old_total = route_distances[route1_idx] + route_distances[route2_idx];
        let new_dist1 = calculate_route_distance(&new_route1, distance_matrix);
        let new_dist2 = calculate_route_distance(&new_route2, distance_matrix);
        let new_total = new_dist1 + new_dist2;

        if new_total < old_total {
            routes[route1_idx] = new_route1;
            routes[route2_idx] = new_route2;
            route_distances[route1_idx] = new_dist1;
            route_distances[route2_idx] = new_dist2;
            route_demands[route1_idx] += demand_diff1;
            route_demands[route2_idx] += demand_diff2;
            *total_distance = *total_distance - old_total + new_total;
            return true;
        }

        false
    }

    #[inline(always)]
    pub fn try_1_2_cross_exchange(
        routes: &mut Vec<Vec<usize>>,
        route1_idx: usize,
        pos1: usize,
        route2_idx: usize,
        pos2: usize,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
        route_distances: &mut Vec<i32>,
        total_distance: &mut i32,
        route_demands: &mut Vec<i32>,
        demands: &Vec<i32>,
        max_capacity: i32,
    ) -> bool {
        let r1 = &routes[route1_idx];
        let r2 = &routes[route2_idx];

        if pos1 >= r1.len() - 1 || pos2 + 1 >= r2.len() - 1 {
            return false;
        }

        let node1 = r1[pos1];
        let node2a = r2[pos2];
        let node2b = r2[pos2 + 1];

        let dem = demands.as_slice();
        let demand_diff1 = unsafe {
            *dem.get_unchecked(node2a) + *dem.get_unchecked(node2b) - *dem.get_unchecked(node1)
        };
        let demand_diff2 = unsafe {
            *dem.get_unchecked(node1) - *dem.get_unchecked(node2a) - *dem.get_unchecked(node2b)
        };

        if route_demands[route1_idx] + demand_diff1 > max_capacity
            || route_demands[route2_idx] + demand_diff2 > max_capacity
        {
            return false;
        }

        let prev1 = r1[pos1 - 1];
        let next1 = r1[pos1 + 1];
        let prev2 = r2[pos2 - 1];
        let next2 = r2[pos2 + 2];
        let dm = distance_matrix;
        let (delta1, delta2) = unsafe {
            let row_prev1 = dm.get_unchecked(prev1);
            let row_prev2 = dm.get_unchecked(prev2);
            let row_n1 = dm.get_unchecked(node1);
            let row_n2a = dm.get_unchecked(node2a);
            let row_n2b = dm.get_unchecked(node2b);

            (
                *row_prev1.get_unchecked(node2a)
                    + *row_n2a.get_unchecked(node2b)
                    + *row_n2b.get_unchecked(next1)
                    - (*row_prev1.get_unchecked(node1) + *row_n1.get_unchecked(next1)),
                *row_prev2.get_unchecked(node1) + *row_n1.get_unchecked(next2)
                    - (*row_prev2.get_unchecked(node2a)
                        + *row_n2a.get_unchecked(node2b)
                        + *row_n2b.get_unchecked(next2)),
            )
        };

        if delta1 + delta2 >= 0 {
            return false;
        }

        let mut new_route1 = Vec::with_capacity(r1.len() + 1);
        let mut new_route2 = Vec::with_capacity(r2.len() - 1);

        new_route1.extend_from_slice(&r1[..pos1]);
        new_route1.push(node2a);
        new_route1.push(node2b);
        new_route1.extend_from_slice(&r1[pos1 + 1..]);

        new_route2.extend_from_slice(&r2[..pos2]);
        new_route2.push(node1);
        new_route2.extend_from_slice(&r2[pos2 + 2..]);

        if !is_route_time_feasible_fast(
            &new_route1,
            distance_matrix,
            service_time,
            ready_times,
            due_times,
        ) || !is_route_time_feasible_fast(
            &new_route2,
            distance_matrix,
            service_time,
            ready_times,
            due_times,
        ) {
            return false;
        }

        let old_total = route_distances[route1_idx] + route_distances[route2_idx];
        let new_dist1 = calculate_route_distance(&new_route1, distance_matrix);
        let new_dist2 = calculate_route_distance(&new_route2, distance_matrix);
        let new_total = new_dist1 + new_dist2;

        if new_total < old_total {
            routes[route1_idx] = new_route1;
            routes[route2_idx] = new_route2;
            route_distances[route1_idx] = new_dist1;
            route_distances[route2_idx] = new_dist2;
            route_demands[route1_idx] += demand_diff1;
            route_demands[route2_idx] += demand_diff2;
            *total_distance = *total_distance - old_total + new_total;
            return true;
        }

        false
    }

    #[inline(always)]
    pub fn try_1_3_cross_exchange(
        routes: &mut Vec<Vec<usize>>,
        route1_idx: usize,
        pos1: usize,
        route2_idx: usize,
        pos2: usize,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
        route_distances: &mut Vec<i32>,
        total_distance: &mut i32,
        route_demands: &mut Vec<i32>,
        demands: &Vec<i32>,
        max_capacity: i32,
    ) -> bool {
        let r1 = &routes[route1_idx];
        let r2 = &routes[route2_idx];

        if pos1 >= r1.len() - 1 || pos2 + 2 >= r2.len() - 1 {
            return false;
        }

        let node1 = r1[pos1];
        let node2a = r2[pos2];
        let node2b = r2[pos2 + 1];
        let node2c = r2[pos2 + 2];

        let dem = demands.as_slice();
        let demand_diff1 = unsafe {
            *dem.get_unchecked(node2a) + *dem.get_unchecked(node2b) + *dem.get_unchecked(node2c)
                - *dem.get_unchecked(node1)
        };
        let demand_diff2 = unsafe {
            *dem.get_unchecked(node1)
                - *dem.get_unchecked(node2a)
                - *dem.get_unchecked(node2b)
                - *dem.get_unchecked(node2c)
        };

        if route_demands[route1_idx] + demand_diff1 > max_capacity
            || route_demands[route2_idx] + demand_diff2 > max_capacity
        {
            return false;
        }

        let prev1 = r1[pos1 - 1];
        let next1 = r1[pos1 + 1];
        let prev2 = r2[pos2 - 1];
        let next2 = r2[pos2 + 3];
        let dm = distance_matrix;
        let (delta1, delta2) = unsafe {
            let row_prev1 = dm.get_unchecked(prev1);
            let row_prev2 = dm.get_unchecked(prev2);
            let row_n1 = dm.get_unchecked(node1);
            let row_n2a = dm.get_unchecked(node2a);
            let row_n2b = dm.get_unchecked(node2b);
            let row_n2c = dm.get_unchecked(node2c);

            (
                *row_prev1.get_unchecked(node2a)
                    + *row_n2a.get_unchecked(node2b)
                    + *row_n2b.get_unchecked(node2c)
                    + *row_n2c.get_unchecked(next1)
                    - (*row_prev1.get_unchecked(node1) + *row_n1.get_unchecked(next1)),
                *row_prev2.get_unchecked(node1) + *row_n1.get_unchecked(next2)
                    - (*row_prev2.get_unchecked(node2a)
                        + *row_n2a.get_unchecked(node2b)
                        + *row_n2b.get_unchecked(node2c)
                        + *row_n2c.get_unchecked(next2)),
            )
        };

        if delta1 + delta2 >= 0 {
            return false;
        }

        let mut new_route1 = Vec::with_capacity(r1.len() + 2);
        let mut new_route2 = Vec::with_capacity(r2.len() - 2);

        new_route1.extend_from_slice(&r1[..pos1]);
        new_route1.push(node2a);
        new_route1.push(node2b);
        new_route1.push(node2c);
        new_route1.extend_from_slice(&r1[pos1 + 1..]);

        new_route2.extend_from_slice(&r2[..pos2]);
        new_route2.push(node1);
        new_route2.extend_from_slice(&r2[pos2 + 3..]);

        if !is_route_time_feasible_fast(
            &new_route1,
            distance_matrix,
            service_time,
            ready_times,
            due_times,
        ) || !is_route_time_feasible_fast(
            &new_route2,
            distance_matrix,
            service_time,
            ready_times,
            due_times,
        ) {
            return false;
        }

        let old_total = route_distances[route1_idx] + route_distances[route2_idx];
        let new_dist1 = calculate_route_distance(&new_route1, distance_matrix);
        let new_dist2 = calculate_route_distance(&new_route2, distance_matrix);
        let new_total = new_dist1 + new_dist2;

        if new_total < old_total {
            routes[route1_idx] = new_route1;
            routes[route2_idx] = new_route2;
            route_distances[route1_idx] = new_dist1;
            route_distances[route2_idx] = new_dist2;
            route_demands[route1_idx] += demand_diff1;
            route_demands[route2_idx] += demand_diff2;
            *total_distance = *total_distance - old_total + new_total;
            return true;
        }

        false
    }

    #[inline(always)]
    pub fn try_2_3_cross_exchange(
        routes: &mut Vec<Vec<usize>>,
        route1_idx: usize,
        pos1: usize,
        route2_idx: usize,
        pos2: usize,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
        route_distances: &mut Vec<i32>,
        total_distance: &mut i32,
        route_demands: &mut Vec<i32>,
        demands: &Vec<i32>,
        max_capacity: i32,
    ) -> bool {
        let r1 = &routes[route1_idx];
        let r2 = &routes[route2_idx];

        if pos1 + 1 >= r1.len() - 1 || pos2 + 2 >= r2.len() - 1 {
            return false;
        }

        let node1a = r1[pos1];
        let node1b = r1[pos1 + 1];
        let node2a = r2[pos2];
        let node2b = r2[pos2 + 1];
        let node2c = r2[pos2 + 2];

        let dem = demands.as_slice();
        let demand_diff1 = unsafe {
            *dem.get_unchecked(node2a) + *dem.get_unchecked(node2b) + *dem.get_unchecked(node2c)
                - *dem.get_unchecked(node1a)
                - *dem.get_unchecked(node1b)
        };
        let demand_diff2 = unsafe {
            *dem.get_unchecked(node1a) + *dem.get_unchecked(node1b)
                - *dem.get_unchecked(node2a)
                - *dem.get_unchecked(node2b)
                - *dem.get_unchecked(node2c)
        };

        if route_demands[route1_idx] + demand_diff1 > max_capacity
            || route_demands[route2_idx] + demand_diff2 > max_capacity
        {
            return false;
        }

        let prev1 = r1[pos1 - 1];
        let next1 = r1[pos1 + 2];
        let prev2 = r2[pos2 - 1];
        let next2 = r2[pos2 + 3];
        let dm = distance_matrix;
        let (delta1, delta2) = unsafe {
            let row_prev1 = dm.get_unchecked(prev1);
            let row_prev2 = dm.get_unchecked(prev2);
            let row_n1a = dm.get_unchecked(node1a);
            let row_n1b = dm.get_unchecked(node1b);
            let row_n2a = dm.get_unchecked(node2a);
            let row_n2b = dm.get_unchecked(node2b);
            let row_n2c = dm.get_unchecked(node2c);

            (
                *row_prev1.get_unchecked(node2a)
                    + *row_n2a.get_unchecked(node2b)
                    + *row_n2b.get_unchecked(node2c)
                    + *row_n2c.get_unchecked(next1)
                    - (*row_prev1.get_unchecked(node1a)
                        + *row_n1a.get_unchecked(node1b)
                        + *row_n1b.get_unchecked(next1)),
                *row_prev2.get_unchecked(node1a)
                    + *row_n1a.get_unchecked(node1b)
                    + *row_n1b.get_unchecked(next2)
                    - (*row_prev2.get_unchecked(node2a)
                        + *row_n2a.get_unchecked(node2b)
                        + *row_n2b.get_unchecked(node2c)
                        + *row_n2c.get_unchecked(next2)),
            )
        };

        if delta1 + delta2 >= 0 {
            return false;
        }

        let mut new_route1 = Vec::with_capacity(r1.len() + 1);
        let mut new_route2 = Vec::with_capacity(r2.len() - 1);

        new_route1.extend_from_slice(&r1[..pos1]);
        new_route1.push(node2a);
        new_route1.push(node2b);
        new_route1.push(node2c);
        new_route1.extend_from_slice(&r1[pos1 + 2..]);

        new_route2.extend_from_slice(&r2[..pos2]);
        new_route2.push(node1a);
        new_route2.push(node1b);
        new_route2.extend_from_slice(&r2[pos2 + 3..]);

        if !is_route_time_feasible_fast(
            &new_route1,
            distance_matrix,
            service_time,
            ready_times,
            due_times,
        ) || !is_route_time_feasible_fast(
            &new_route2,
            distance_matrix,
            service_time,
            ready_times,
            due_times,
        ) {
            return false;
        }

        let old_total = route_distances[route1_idx] + route_distances[route2_idx];
        let new_dist1 = calculate_route_distance(&new_route1, distance_matrix);
        let new_dist2 = calculate_route_distance(&new_route2, distance_matrix);
        let new_total = new_dist1 + new_dist2;

        if new_total < old_total {
            routes[route1_idx] = new_route1;
            routes[route2_idx] = new_route2;
            route_distances[route1_idx] = new_dist1;
            route_distances[route2_idx] = new_dist2;
            route_demands[route1_idx] += demand_diff1;
            route_demands[route2_idx] += demand_diff2;
            *total_distance = *total_distance - old_total + new_total;
            return true;
        }

        false
    }

    #[inline(always)]
    pub fn update_node_positions_for_routes(
        node_positions: &mut Vec<(usize, usize)>,
        routes: &Vec<Vec<usize>>,
        route_indices: &[usize],
    ) {
        for &route_idx in route_indices {
            let route = &routes[route_idx];
            let end = route.len() - 1;
            if end <= 1 {
                continue;
            }
            unsafe {
                let slice = route.as_slice();
                let mut j = 1usize;
                while j < end {
                    let node = *slice.get_unchecked(j);
                    *node_positions.get_unchecked_mut(node) = (route_idx, j);
                    j += 1;
                }
            }
        }
    }
}

mod simple_solver {
    use super::utils::*;
    use super::*;

    #[inline(always)]
    pub fn solve_challenge(
        challenge: &Challenge,
        save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
        hyperparameters: &Option<Map<String, Value>>,
    ) -> anyhow::Result<()> {
        let num_nodes = challenge.difficulty.num_nodes;
        let max_capacity = challenge.max_capacity;
        let demands = &challenge.demands;
        let distance_matrix = &challenge.distance_matrix;
        let service_time = challenge.service_time;
        let ready_times = &challenge.ready_times;
        let due_times = &challenge.due_times;
        let mut routes: Vec<Vec<usize>> = Vec::with_capacity(num_nodes);
        let dem = demands.as_slice();

        let mut nodes: Vec<usize> = (1..num_nodes).collect();

        nodes.sort_by(|a, b| distance_matrix[0][*a].cmp(&distance_matrix[0][*b]));

        let mut remaining = vec![false; num_nodes];
        for &n in &nodes {
            remaining[n] = true;
        }

        while let Some(node) = nodes.pop() {
            if !remaining[node] {
                continue;
            }
            remaining[node] = false;
            let mut route = vec![0, node, 0];
            let mut route_demand = unsafe { *dem.get_unchecked(node) };

            let mut insertion_attempts = 0;
            let max_insertion_attempts = {
                let base_attempts = if (due_times[0] - ready_times[0]) < service_time * 10 {
                    3
                } else {
                    2
                };
                let avg_slack = {
                    let mut s: i64 = 0;
                    let mut c: i64 = 0;
                    for i in 1..num_nodes {
                        s += (due_times[i] - ready_times[i]) as i64;
                        c += 1;
                    }
                    if c > 0 {
                        (s / c) as i32
                    } else {
                        0
                    }
                };
                let tight = if avg_slack <= service_time * 4 { 2 } else { 0 };
                let scale_factor = (num_nodes as f32 / 1200.0).sqrt();
                ((base_attempts + tight) as f32 * scale_factor).round() as usize
            }
            .clamp(3, 8);
            let mut candidates: Vec<usize> = Vec::with_capacity(num_nodes);

            while insertion_attempts < max_insertion_attempts {
                candidates.clear();

                for n in 1..num_nodes {
                    if remaining[n]
                        && route_demand + unsafe { *dem.get_unchecked(n) } <= max_capacity
                    {
                        candidates.push(n);
                    }
                }

                if candidates.is_empty() {
                    break;
                }

                if let Some((best_node, best_pos)) = find_best_insertion(
                    &route,
                    &candidates,
                    distance_matrix,
                    service_time,
                    ready_times,
                    due_times,
                ) {
                    remaining[best_node] = false;
                    route_demand += unsafe { *dem.get_unchecked(best_node) };
                    route.insert(best_pos, best_node);
                    insertion_attempts = 0;
                } else {
                    insertion_attempts += 1;
                }
            }

            route = apply_size_filtered_local_search(
                &route,
                distance_matrix,
                service_time,
                ready_times,
                due_times,
            );
            if route.len() >= 3 && route[0] == 0 && route[route.len() - 1] == 0 {
                routes.push(route);
            }
        }

        {
            let mut leftover: Vec<usize> = Vec::new();
            for n in 1..num_nodes {
                if remaining[n] {
                    leftover.push(n);
                }
            }
            if !leftover.is_empty() && !routes.is_empty() {
                let mut route_demands_pre = utils::calculate_route_demands(&routes, demands);
                let mut iter_count = 0usize;
                let max_global_iters = leftover.len().saturating_mul(2).min(64);
                let mut progress = true;
                while progress && !leftover.is_empty() && iter_count < max_global_iters {
                    iter_count += 1;
                    progress = false;
                    for &n in &leftover {
                        let mut best_delta = i32::MAX;
                        let mut best_route_idx = 0usize;
                        let mut best_pos = 0usize;
                        for (r_idx, r) in routes.iter().enumerate() {
                            if route_demands_pre[r_idx] + unsafe { *dem.get_unchecked(n) }
                                > max_capacity
                            {
                                continue;
                            }
                            if let Some((pos, delta)) = find_best_insertion_in_route(
                                r,
                                n,
                                demands,
                                max_capacity,
                                distance_matrix,
                                service_time,
                                ready_times,
                                due_times,
                            ) {
                                if delta < best_delta {
                                    best_delta = delta;
                                    best_route_idx = r_idx;
                                    best_pos = pos;
                                }
                            }
                        }
                        if best_delta != i32::MAX {
                            routes[best_route_idx].insert(best_pos, n);
                            route_demands_pre[best_route_idx] += unsafe { *dem.get_unchecked(n) };
                            remaining[n] = false;
                            if let Some(idx) = leftover.iter().position(|&x| x == n) {
                                leftover.swap_remove(idx);
                            }
                            progress = true;
                            break;
                        }
                    }
                }
            }
        }
        for n in 1..num_nodes {
            if remaining[n] {
                routes.push(vec![0, n, 0]);
            }
        }

        routes = do_local_searches(
            num_nodes,
            max_capacity,
            demands,
            distance_matrix,
            &routes,
            service_time,
            ready_times,
            due_times,
        );

        let mut seen = vec![false; num_nodes];
        let mut duplication = false;
        for r in &routes {
            for &n in r[1..r.len() - 1].iter() {
                if seen[n] {
                    duplication = true;
                } else {
                    seen[n] = true;
                }
            }
        }
        let mut missing_any = false;
        for n in 1..num_nodes {
            if !seen[n] {
                missing_any = true;
                break;
            }
        }
        if duplication || missing_any {
            let mut fallback: Vec<Vec<usize>> = Vec::with_capacity(num_nodes - 1);
            for n in 1..num_nodes {
                fallback.push(vec![0, n, 0]);
            }
            routes = fallback;
        }

        let mut validated_routes: Vec<Vec<usize>> = Vec::with_capacity(routes.len());
        let mut need_fallback = false;

        for route in routes {
            if route.len() < 3 || route[0] != 0 || route[route.len() - 1] != 0 {
                need_fallback = true;
                break;
            }

            let time_ok = is_route_time_feasible_fast(
                &route,
                distance_matrix,
                service_time,
                ready_times,
                due_times,
            );

            let cap: i32 = route[1..route.len() - 1].iter().map(|&n| demands[n]).sum();
            let cap_ok = cap <= max_capacity;

            if time_ok && cap_ok {
                validated_routes.push(route);
                continue;
            }

            let mut local_repair_ok = true;
            for &node in route[1..route.len() - 1].iter() {
                if demands[node] > max_capacity {
                    local_repair_ok = false;
                    break;
                }
                let singleton = vec![0, node, 0];
                if !is_route_time_feasible_strict(
                    &singleton,
                    distance_matrix,
                    service_time,
                    ready_times,
                    due_times,
                ) {
                    local_repair_ok = false;
                    break;
                }
                validated_routes.push(singleton);
            }

            if !local_repair_ok {
                need_fallback = true;
                break;
            }
        }

        if !need_fallback {
            let served: usize = validated_routes
                .iter()
                .map(|r| if r.len() >= 2 { r.len() - 2 } else { 0 })
                .sum();
            if served != num_nodes - 1 {
                need_fallback = true;
            }
        }

        if need_fallback || validated_routes.is_empty() {
            let mut fallback: Vec<Vec<usize>> = Vec::with_capacity(num_nodes - 1);
            for n in 1..num_nodes {
                if demands[n] > max_capacity {
                    return Ok(());
                }
                let singleton = vec![0, n, 0];
                if !is_route_time_feasible_strict(
                    &singleton,
                    distance_matrix,
                    service_time,
                    ready_times,
                    due_times,
                ) {
                    return Ok(());
                }
                fallback.push(singleton);
            }
            let _ = save_solution(&Solution { routes: fallback });
            return Ok(());
        }

        let _ = save_solution(&Solution {
            routes: validated_routes,
        });
        return Ok(());
    }

    #[inline(always)]
    fn do_local_searches(
        num_nodes: usize,
        max_capacity: i32,
        demands: &Vec<i32>,
        distance_matrix: &Vec<Vec<i32>>,
        routes: &Vec<Vec<usize>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> Vec<Vec<usize>> {
        let mut best_routes = routes.clone();
        let dem = demands.as_slice();

        let mut route_distances: Vec<i32> = Vec::with_capacity(best_routes.len());
        let mut best_distance: i32 = 0;
        for r in &best_routes {
            let d = utils::calculate_route_distance(r, distance_matrix);
            route_distances.push(d);
            best_distance += d;
        }
        let mut improved = true;
        let mut iteration_count = 0;
        let improvement_threshold = (best_distance as f32 * 0.00016).max(1.0) as i32;
        let max_total_iterations = {
            let base_iterations = 140;
            let scale_factor = if num_nodes <= 1500 {
                0.95 + (num_nodes - 1000) as f32 / 2400.0
            } else {
                1.10 + (num_nodes - 1500) as f32 / 2600.0
            };
            ((base_iterations as f32 * scale_factor).round() as usize).clamp(130, 300)
        };

        let neighbor_count = {
            let base_neighbors = 24;
            let scale_factor = (num_nodes as f32 / 1500.0).powf(0.62);
            (base_neighbors as f32 * scale_factor).round() as usize
        }
        .clamp(24, 44);
        let pool = {
            let base_pool = 34;
            let scale_factor = (num_nodes as f32 / 1500.0).powf(0.70);
            (base_pool as f32 * scale_factor).round() as usize
        }
        .clamp(30, 80);

        let mut proximity_pairs: Vec<(i32, usize, usize)> =
            Vec::with_capacity(num_nodes * neighbor_count * 2);
        let mut pool_vec: Vec<(i32, usize)> = Vec::with_capacity(pool);
        let mut bests: Vec<(i32, usize)> = Vec::with_capacity(neighbor_count);
        for i in 1..num_nodes {
            pool_vec.clear();
            let mut max_idx: usize = 0;
            let mut max_d: i32 = i32::MIN;
            let di = &distance_matrix[i];
            unsafe {
                for j in (i + 1)..num_nodes {
                    let d = *di.get_unchecked(j);
                    if pool_vec.len() < pool {
                        pool_vec.push((d, j));
                        if d > max_d {
                            max_d = d;
                            max_idx = pool_vec.len() - 1;
                        }
                    } else if d < max_d {
                        pool_vec[max_idx] = (d, j);
                        let mut new_max_d = i32::MIN;
                        let mut new_max_idx = 0;
                        let pv = pool_vec.as_slice();
                        let mut idx = 0usize;
                        while idx < pv.len() {
                            let dd = (*pv.get_unchecked(idx)).0;
                            if dd > new_max_d {
                                new_max_d = dd;
                                new_max_idx = idx;
                            }
                            idx += 1;
                        }
                        max_d = new_max_d;
                        max_idx = new_max_idx;
                    }
                }
            }
            bests.clear();
            let ri = ready_times[i];
            let di_due = due_times[i];
            unsafe {
                let rt = ready_times.as_slice();
                let dt = due_times.as_slice();
                for &(_d, j) in pool_vec.iter() {
                    let time_ij = *di.get_unchecked(j);
                    let rj = *rt.get_unchecked(j);
                    let dj = *dt.get_unchecked(j);
                    let expr1 = (rj - time_ij - service_time - di_due).max(0)
                        + (ri + service_time + time_ij - dj).max(0);
                    let expr2 = (ri - time_ij - service_time - dj).max(0)
                        + (rj + service_time + time_ij - di_due).max(0);

                    let time_window_penalty = if expr1 < expr2 { expr1 } else { expr2 };
                    let wi = di_due - ri;
                    let wj = dj - rj;
                    let time_compatibility = wi.min(wj) as f32 / (wi.max(wj) + 1) as f32;
                    let time_bonus = (time_compatibility * 20.0) as i32;

                    let p = time_ij + time_window_penalty - time_bonus;
                    if bests.len() < neighbor_count {
                        let mut pos = bests.len();
                        while pos > 0 && p < bests[pos - 1].0 {
                            pos -= 1;
                        }
                        bests.insert(pos, (p, j));
                    } else if p < bests[bests.len() - 1].0 {
                        let mut pos = bests.len() - 1;
                        while pos > 0 && p < bests[pos - 1].0 {
                            pos -= 1;
                        }
                        bests.insert(pos, (p, j));
                        bests.pop();
                    }
                }
            }
            for (p, j) in &bests {
                proximity_pairs.push((*p, i, *j));
                proximity_pairs.push((*p, *j, i));
            }
        }

        let mut node_positions = vec![(0, 0); num_nodes];
        for (i, route) in best_routes.iter().enumerate() {
            if route.len() <= 2 {
                continue;
            }
            unsafe {
                let r = route.as_slice();
                let mut j = 1usize;
                let end = r.len() - 1;
                while j < end {
                    let node = *r.get_unchecked(j);
                    *node_positions.get_unchecked_mut(node) = (i, j);
                    j += 1;
                }
            }
        }

        let mut route_demands = calculate_route_demands(&best_routes, demands);
        let mut cand_r1: Vec<usize> = Vec::with_capacity(num_nodes);
        let mut cand_r2: Vec<usize> = Vec::with_capacity(num_nodes);
        let mut swap_r1: Vec<usize> = Vec::with_capacity(num_nodes);
        let mut swap_r2: Vec<usize> = Vec::with_capacity(num_nodes);
        while improved && iteration_count < max_total_iterations {
            improved = false;
            iteration_count += 1;

            for (_corr, node, node2) in &proximity_pairs {
                let node = *node;
                let node2 = *node2;
                let (route1_idx, pos1) = node_positions[node];
                let (route2_idx, pos2_base) = node_positions[node2];
                if route1_idx == route2_idx {
                    continue;
                }

                for pos2_offset in 0..=1 {
                    let pos2 = pos2_base + pos2_offset;

                    if pos2_offset == 0 {
                        let target_route_demand = route_demands[route2_idx];
                        if target_route_demand + unsafe { *dem.get_unchecked(node) } <= max_capacity
                        {
                            if let Some((best_pos, _delta_cost)) = find_best_insertion_in_route(
                                &best_routes[route2_idx],
                                node,
                                demands,
                                max_capacity,
                                distance_matrix,
                                service_time,
                                ready_times,
                                due_times,
                            ) {
                                if best_routes[route1_idx].len() > pos1
                                    && best_routes[route1_idx][pos1] == node
                                {
                                    let r1_ref = &best_routes[route1_idx];
                                    let r2_ref = &best_routes[route2_idx];
                                    let prev1 = r1_ref[pos1 - 1];
                                    let next1 = r1_ref[pos1 + 1];
                                    let prev2 = r2_ref[best_pos - 1];
                                    let next2 = r2_ref[best_pos];
                                    let dm = distance_matrix;
                                    let (delta_remove, delta_insert) = unsafe {
                                        let row_prev1 = dm.get_unchecked(prev1);
                                        let row_prev2 = dm.get_unchecked(prev2);
                                        let row_node = dm.get_unchecked(node);
                                        (
                                            *row_prev1.get_unchecked(next1)
                                                - *row_prev1.get_unchecked(node)
                                                - *row_node.get_unchecked(next1),
                                            *row_prev2.get_unchecked(node)
                                                + *row_node.get_unchecked(next2)
                                                - *row_prev2.get_unchecked(next2),
                                        )
                                    };
                                    if delta_remove + delta_insert >= 0 {
                                        continue;
                                    }

                                    if r1_ref.len() <= 3 {
                                        continue;
                                    }

                                    cand_r1.clear();
                                    cand_r2.clear();

                                    if cand_r1.capacity() < r1_ref.len() - 1 {
                                        cand_r1.reserve(r1_ref.len() - 1 - cand_r1.capacity());
                                    }
                                    if cand_r2.capacity() < r2_ref.len() + 1 {
                                        cand_r2.reserve(r2_ref.len() + 1 - cand_r2.capacity());
                                    }

                                    cand_r1.extend_from_slice(&r1_ref[..pos1]);
                                    cand_r1.extend_from_slice(&r1_ref[pos1 + 1..]);

                                    cand_r2.extend_from_slice(&r2_ref[..best_pos]);
                                    cand_r2.push(node);
                                    cand_r2.extend_from_slice(&r2_ref[best_pos..]);

                                    if is_route_time_feasible_fast(
                                        &cand_r1,
                                        distance_matrix,
                                        service_time,
                                        ready_times,
                                        due_times,
                                    ) && is_route_time_feasible_fast(
                                        &cand_r2,
                                        distance_matrix,
                                        service_time,
                                        ready_times,
                                        due_times,
                                    ) {
                                        let old_d1 = route_distances[route1_idx];
                                        let old_d2 = route_distances[route2_idx];
                                        let new_d1 = old_d1 + delta_remove;
                                        let new_d2 = old_d2 + delta_insert;
                                        let new_distance =
                                            best_distance - old_d1 - old_d2 + new_d1 + new_d2;
                                        if new_distance < best_distance {
                                            let improvement = best_distance - new_distance;
                                            best_distance = new_distance;

                                            best_routes[route1_idx].clear();
                                            best_routes[route1_idx].extend_from_slice(&cand_r1);
                                            best_routes[route2_idx].clear();
                                            best_routes[route2_idx].extend_from_slice(&cand_r2);

                                            route_distances[route1_idx] = new_d1;
                                            route_distances[route2_idx] = new_d2;
                                            route_demands[route1_idx] -=
                                                unsafe { *dem.get_unchecked(node) };
                                            route_demands[route2_idx] +=
                                                unsafe { *dem.get_unchecked(node) };

                                            update_node_positions_for_routes(
                                                &mut node_positions,
                                                &best_routes,
                                                &[route1_idx, route2_idx],
                                            );

                                            if improvement >= improvement_threshold {
                                                improved = true;
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if pos2 < best_routes[route2_idx].len()
                        && best_routes[route2_idx][pos2] != 0
                        && route_demands[route1_idx] - demands[node] + demands[node2]
                            <= max_capacity
                        && route_demands[route2_idx] - demands[node2] + demands[node]
                            <= max_capacity
                    {
                        if best_routes[route1_idx].len() > pos1
                            && best_routes[route1_idx][pos1] == node
                            && best_routes[route2_idx].len() > pos2
                            && best_routes[route2_idx][pos2] == node2
                        {
                            let r1_ref = &best_routes[route1_idx];
                            let r2_ref = &best_routes[route2_idx];
                            let prev1 = r1_ref[pos1 - 1];
                            let next1 = r1_ref[pos1 + 1];
                            let prev2 = r2_ref[pos2 - 1];
                            let next2 = r2_ref[pos2 + 1];
                            let dm = distance_matrix;
                            let (delta1, delta2) = unsafe {
                                let row_prev1 = dm.get_unchecked(prev1);
                                let row_prev2 = dm.get_unchecked(prev2);
                                let row_node = dm.get_unchecked(node);
                                let row_node2 = dm.get_unchecked(node2);
                                (
                                    *row_prev1.get_unchecked(node2)
                                        + *row_node2.get_unchecked(next1)
                                        - (*row_prev1.get_unchecked(node)
                                            + *row_node.get_unchecked(next1)),
                                    *row_prev2.get_unchecked(node) + *row_node.get_unchecked(next2)
                                        - (*row_prev2.get_unchecked(node2)
                                            + *row_node2.get_unchecked(next2)),
                                )
                            };

                            if delta1 + delta2 >= 0 {
                                continue;
                            }

                            let time_feasible_1 = check_swap_time_feasibility(
                                r1_ref,
                                pos1,
                                node2,
                                distance_matrix,
                                service_time,
                                ready_times,
                                due_times,
                            );
                            let time_feasible_2 = check_swap_time_feasibility(
                                r2_ref,
                                pos2,
                                node,
                                distance_matrix,
                                service_time,
                                ready_times,
                                due_times,
                            );
                            if !time_feasible_1 || !time_feasible_2 {
                                continue;
                            }

                            swap_r1.clear();
                            swap_r2.clear();
                            let r1_src = &best_routes[route1_idx];
                            let r2_src = &best_routes[route2_idx];
                            if swap_r1.capacity() < r1_src.len() {
                                swap_r1.reserve(r1_src.len() - swap_r1.capacity());
                            }
                            if swap_r2.capacity() < r2_src.len() {
                                swap_r2.reserve(r2_src.len() - swap_r2.capacity());
                            }
                            swap_r1.extend_from_slice(r1_src);
                            swap_r2.extend_from_slice(r2_src);

                            swap_r1[pos1] = node2;
                            swap_r2[pos2] = node;

                            if is_route_time_feasible_fast(
                                &swap_r1,
                                distance_matrix,
                                service_time,
                                ready_times,
                                due_times,
                            ) && is_route_time_feasible_fast(
                                &swap_r2,
                                distance_matrix,
                                service_time,
                                ready_times,
                                due_times,
                            ) {
                                let old_d1 = route_distances[route1_idx];
                                let old_d2 = route_distances[route2_idx];
                                let new_d1 = old_d1 + delta1;
                                let new_d2 = old_d2 + delta2;
                                let new_distance =
                                    best_distance - old_d1 - old_d2 + new_d1 + new_d2;
                                if new_distance < best_distance {
                                    let improvement = best_distance - new_distance;
                                    best_distance = new_distance;

                                    best_routes[route1_idx].clear();
                                    best_routes[route1_idx].extend_from_slice(&swap_r1);
                                    best_routes[route2_idx].clear();
                                    best_routes[route2_idx].extend_from_slice(&swap_r2);

                                    route_distances[route1_idx] = new_d1;
                                    route_distances[route2_idx] = new_d2;
                                    route_demands[route1_idx] = route_demands[route1_idx]
                                        - unsafe { *dem.get_unchecked(node) }
                                        + unsafe { *dem.get_unchecked(node2) };
                                    route_demands[route2_idx] = route_demands[route2_idx]
                                        - unsafe { *dem.get_unchecked(node2) }
                                        + unsafe { *dem.get_unchecked(node) };

                                    unsafe {
                                        *node_positions.get_unchecked_mut(node) =
                                            (route2_idx, pos2);
                                        *node_positions.get_unchecked_mut(node2) =
                                            (route1_idx, pos1);
                                    }

                                    if improvement >= improvement_threshold {
                                        improved = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if !improved {
                let max_pairs_per_iter: usize = if num_nodes <= 1500 {
                    50000
                } else if num_nodes <= 2500 {
                    90000
                } else {
                    140000
                };
                let mut checked_pairs: usize = 0;
                for (_corr, node, node2) in &proximity_pairs {
                    if checked_pairs >= max_pairs_per_iter {
                        break;
                    }
                    checked_pairs += 1;
                    let node = *node;
                    let node2 = *node2;
                    let (route1_idx, pos1) = node_positions[node];
                    let (route2_idx, _) = node_positions[node2];
                    if route1_idx == route2_idx {
                        continue;
                    }
                    let r1_ref = &best_routes[route1_idx];
                    if pos1 + 1 >= r1_ref.len() - 1 {
                        continue;
                    }
                    let node_b = r1_ref[pos1 + 1];
                    if node_b == 0 {
                        continue;
                    }
                    let add_dem =
                        unsafe { *dem.get_unchecked(node) } + unsafe { *dem.get_unchecked(node_b) };
                    if route_demands[route2_idx] + add_dem > max_capacity {
                        continue;
                    }
                    if let Some((best_pos2, delta_insert2)) = find_best_insertion_pair_in_route(
                        &best_routes[route2_idx],
                        node,
                        node_b,
                        distance_matrix,
                        service_time,
                        ready_times,
                        due_times,
                    ) {
                        let prev1 = r1_ref[pos1 - 1];
                        let next_after = r1_ref[pos1 + 2];
                        let dm = distance_matrix;
                        let delta_remove_pair = unsafe {
                            let row_prev1 = dm.get_unchecked(prev1);
                            let row_node = dm.get_unchecked(node);
                            let row_nodeb = dm.get_unchecked(node_b);
                            *row_prev1.get_unchecked(next_after)
                                - (*row_prev1.get_unchecked(node)
                                    + *row_node.get_unchecked(node_b)
                                    + *row_nodeb.get_unchecked(next_after))
                        };
                        if delta_remove_pair + delta_insert2 >= 0 {
                            continue;
                        }

                        cand_r1.clear();
                        cand_r2.clear();

                        let r2_ref = &best_routes[route2_idx];
                        if cand_r1.capacity() < r1_ref.len() - 2 {
                            cand_r1.reserve(r1_ref.len() - 2 - cand_r1.capacity());
                        }
                        if cand_r2.capacity() < r2_ref.len() + 2 {
                            cand_r2.reserve(r2_ref.len() + 2 - cand_r2.capacity());
                        }

                        cand_r1.extend_from_slice(&r1_ref[..pos1]);
                        cand_r1.extend_from_slice(&r1_ref[pos1 + 2..]);

                        cand_r2.extend_from_slice(&r2_ref[..best_pos2]);
                        cand_r2.push(node);
                        cand_r2.push(node_b);
                        cand_r2.extend_from_slice(&r2_ref[best_pos2..]);

                        if is_route_time_feasible_fast(
                            &cand_r1,
                            distance_matrix,
                            service_time,
                            ready_times,
                            due_times,
                        ) && is_route_time_feasible_fast(
                            &cand_r2,
                            distance_matrix,
                            service_time,
                            ready_times,
                            due_times,
                        ) {
                            let old_d1 = route_distances[route1_idx];
                            let old_d2 = route_distances[route2_idx];
                            let new_d1 = old_d1 + delta_remove_pair;
                            let new_d2 = old_d2 + delta_insert2;
                            let new_total = best_distance - old_d1 - old_d2 + new_d1 + new_d2;

                            if new_total < best_distance {
                                let improvement = best_distance - new_total;
                                best_distance = new_total;

                                best_routes[route1_idx].clear();
                                best_routes[route1_idx].extend_from_slice(&cand_r1);
                                best_routes[route2_idx].clear();
                                best_routes[route2_idx].extend_from_slice(&cand_r2);

                                route_distances[route1_idx] = new_d1;
                                route_distances[route2_idx] = new_d2;
                                route_demands[route1_idx] -= add_dem;
                                route_demands[route2_idx] += add_dem;

                                update_node_positions_for_routes(
                                    &mut node_positions,
                                    &best_routes,
                                    &[route1_idx, route2_idx],
                                );

                                if improvement >= improvement_threshold {
                                    improved = true;
                                    break;
                                } else {
                                    improved = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            if !improved {
                let max_pairs_per_iter: usize = if num_nodes <= 1500 {
                    50000
                } else if num_nodes <= 2500 {
                    90000
                } else {
                    140000
                };
                let mut checked_pairs: usize = 0;
                for (_corr, node, node2) in &proximity_pairs {
                    if checked_pairs >= max_pairs_per_iter {
                        break;
                    }
                    checked_pairs += 1;
                    let node = *node;
                    let node2 = *node2;
                    let (route1_idx, pos1) = node_positions[node];
                    let (route2_idx, pos2_base) = node_positions[node2];
                    if route1_idx == route2_idx {
                        continue;
                    }

                    for pos2_offset in 0..=1 {
                        let pos2 = pos2_base + pos2_offset;

                        if try_1_2_cross_exchange(
                            &mut best_routes,
                            route1_idx,
                            pos1,
                            route2_idx,
                            pos2,
                            distance_matrix,
                            service_time,
                            ready_times,
                            due_times,
                            &mut route_distances,
                            &mut best_distance,
                            &mut route_demands,
                            demands,
                            max_capacity,
                        ) {
                            update_node_positions_for_routes(
                                &mut node_positions,
                                &best_routes,
                                &[route1_idx, route2_idx],
                            );
                            improved = true;
                            break;
                        }

                        if try_1_2_cross_exchange(
                            &mut best_routes,
                            route2_idx,
                            pos2,
                            route1_idx,
                            pos1,
                            distance_matrix,
                            service_time,
                            ready_times,
                            due_times,
                            &mut route_distances,
                            &mut best_distance,
                            &mut route_demands,
                            demands,
                            max_capacity,
                        ) {
                            update_node_positions_for_routes(
                                &mut node_positions,
                                &best_routes,
                                &[route1_idx, route2_idx],
                            );
                            improved = true;
                            break;
                        }

                        if try_2_2_cross_exchange(
                            &mut best_routes,
                            route1_idx,
                            pos1,
                            route2_idx,
                            pos2,
                            distance_matrix,
                            service_time,
                            ready_times,
                            due_times,
                            &mut route_distances,
                            &mut best_distance,
                            &mut route_demands,
                            demands,
                            max_capacity,
                        ) {
                            update_node_positions_for_routes(
                                &mut node_positions,
                                &best_routes,
                                &[route1_idx, route2_idx],
                            );
                            improved = true;
                            break;
                        }

                        if try_two_opt_star(
                            &mut best_routes,
                            route1_idx,
                            pos1,
                            route2_idx,
                            pos2,
                            distance_matrix,
                            service_time,
                            ready_times,
                            due_times,
                            &mut route_distances,
                            &mut best_distance,
                            &mut route_demands,
                            demands,
                            max_capacity,
                        ) {
                            update_node_positions_for_routes(
                                &mut node_positions,
                                &best_routes,
                                &[route1_idx, route2_idx],
                            );
                            improved = true;
                            break;
                        }

                        if (*_corr as usize + node + node2) % 3 == 0 {
                            if try_1_3_cross_exchange(
                                &mut best_routes,
                                route1_idx,
                                pos1,
                                route2_idx,
                                pos2,
                                distance_matrix,
                                service_time,
                                ready_times,
                                due_times,
                                &mut route_distances,
                                &mut best_distance,
                                &mut route_demands,
                                demands,
                                max_capacity,
                            ) {
                                update_node_positions_for_routes(
                                    &mut node_positions,
                                    &best_routes,
                                    &[route1_idx, route2_idx],
                                );
                                improved = true;
                                break;
                            }

                            if try_1_3_cross_exchange(
                                &mut best_routes,
                                route2_idx,
                                pos2,
                                route1_idx,
                                pos1,
                                distance_matrix,
                                service_time,
                                ready_times,
                                due_times,
                                &mut route_distances,
                                &mut best_distance,
                                &mut route_demands,
                                demands,
                                max_capacity,
                            ) {
                                update_node_positions_for_routes(
                                    &mut node_positions,
                                    &best_routes,
                                    &[route1_idx, route2_idx],
                                );
                                improved = true;
                                break;
                            }

                            if try_2_3_cross_exchange(
                                &mut best_routes,
                                route1_idx,
                                pos1,
                                route2_idx,
                                pos2,
                                distance_matrix,
                                service_time,
                                ready_times,
                                due_times,
                                &mut route_distances,
                                &mut best_distance,
                                &mut route_demands,
                                demands,
                                max_capacity,
                            ) {
                                update_node_positions_for_routes(
                                    &mut node_positions,
                                    &best_routes,
                                    &[route1_idx, route2_idx],
                                );
                                improved = true;
                                break;
                            }

                            if try_2_3_cross_exchange(
                                &mut best_routes,
                                route2_idx,
                                pos2,
                                route1_idx,
                                pos1,
                                distance_matrix,
                                service_time,
                                ready_times,
                                due_times,
                                &mut route_distances,
                                &mut best_distance,
                                &mut route_demands,
                                demands,
                                max_capacity,
                            ) {
                                update_node_positions_for_routes(
                                    &mut node_positions,
                                    &best_routes,
                                    &[route1_idx, route2_idx],
                                );
                                improved = true;
                                break;
                            }
                        }
                    }
                }
            }

            if !improved {
                for route_idx in 0..best_routes.len() {
                    let improved_route = apply_size_filtered_local_search(
                        &best_routes[route_idx],
                        distance_matrix,
                        service_time,
                        ready_times,
                        due_times,
                    );

                    if improved_route != best_routes[route_idx]
                        && improved_route.len() >= 3
                        && improved_route[0] == 0
                        && improved_route[improved_route.len() - 1] == 0
                    {
                        let old_d = route_distances[route_idx];
                        let new_d =
                            utils::calculate_route_distance(&improved_route, distance_matrix);
                        let total_distance = best_distance - old_d + new_d;
                        if total_distance < best_distance {
                            let improvement = best_distance - total_distance;
                            best_distance = total_distance;

                            best_routes[route_idx] = improved_route;

                            route_distances[route_idx] = new_d;
                            update_node_positions_for_routes(
                                &mut node_positions,
                                &best_routes,
                                &[route_idx],
                            );

                            if improvement >= improvement_threshold {
                                improved = true;
                                break;
                            }
                        }
                    }
                }
            }
        }

        best_routes
            .retain(|route| route.len() >= 3 && route[0] == 0 && route[route.len() - 1] == 0);

        best_routes
    }
}
