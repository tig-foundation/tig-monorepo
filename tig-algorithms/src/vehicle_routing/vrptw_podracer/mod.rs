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
        for route in routes {
            let mut s = 0i32;
            let len = route.len();
            if len > 2 {
                for &n in &route[1..len - 1] {
                    s += demands[n];
                }
            }
            out.push(s);
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
        let alpha1 = 1;
        let alpha2 = 0;
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
                let mut curr_time: i32 = 0;
                let mut curr_node: usize = 0;

                for pos in 1..len {
                    let next_node = *route.get_unchecked(pos);
                    let row_curr = dm.get_unchecked(curr_node);

                    let travel_to_insert = *row_curr.get_unchecked(insert_node);
                    let travel_to_next = *row_curr.get_unchecked(next_node);
                    let mut new_arrival_time = curr_time + travel_to_insert;
                    let ready_ins = *rt.get_unchecked(insert_node);
                    if new_arrival_time < ready_ins {
                        new_arrival_time = ready_ins;
                    }

                    if new_arrival_time > *dt.get_unchecked(insert_node) {
                        let mut tmp = curr_time + travel_to_next;
                        let r_next = *rt.get_unchecked(next_node);
                        if tmp < r_next {
                            tmp = r_next;
                        }
                        curr_time = tmp + service_time;
                        curr_node = next_node;
                        continue;
                    }

                    let mut old_arrival_time = curr_time + travel_to_next;
                    let r_next = *rt.get_unchecked(next_node);
                    if old_arrival_time < r_next {
                        old_arrival_time = r_next;
                    }

                    let c11 =
                        travel_to_insert + *row_insert.get_unchecked(next_node) - travel_to_next;

                    let c12 = new_arrival_time - old_arrival_time;

                    let c1 = -(alpha1 * c11 + alpha2 * c12);
                    let c2 = base_c2 + c1;

                    let c2_is_better = match best_c2 {
                        None => true,
                        Some(x) => c2 > x,
                    };

                    if c2_is_better
                        && is_feasible(
                            route,
                            distance_matrix,
                            service_time,
                            ready_times,
                            due_times,
                            insert_node,
                            new_arrival_time + service_time,
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
        let rt = &ready_times[..];
        let dt = &due_times[..];
        let len = route.len();
        let mut pos = start_pos;
        while pos < len {
            let next_node = route[pos];
            let row_curr = &dm[curr_node];
            curr_time += row_curr[next_node];
            if next_node != 0 {
                if curr_time > dt[next_node] {
                    return false;
                }
                let r = rt[next_node];
                if curr_time < r {
                    curr_time = r;
                }
                curr_time += service_time;
            }
            curr_node = next_node;
            pos += 1;
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
        let mut best_pos = None;
        let mut best_delta = i32::MAX;

        unsafe {
            let row_node = distance_matrix.get_unchecked(node);
            let len = route.len();
            let r = route.as_slice();
            let mut pos = 1usize;
            while pos < len {
                let prev_node = *r.get_unchecked(pos - 1);
                let next_node = *r.get_unchecked(pos);
                let row_prev = distance_matrix.get_unchecked(prev_node);

                let delta = *row_prev.get_unchecked(node) + *row_node.get_unchecked(next_node)
                    - *row_prev.get_unchecked(next_node);

                if check_feasible_insertion(
                    route,
                    node,
                    pos,
                    distance_matrix,
                    service_time,
                    ready_times,
                    due_times,
                ) {
                    if delta < best_delta {
                        best_delta = delta;
                        best_pos = Some(pos);
                    }
                }
                pos += 1;
            }
        }

        best_pos.map(|pos| (pos, best_delta))
    }

    #[inline(always)]
    pub fn check_feasible_insertion(
        route: &Vec<usize>,
        insert_node: usize,
        insert_pos: usize,
        distance_matrix: &Vec<Vec<i32>>,
        service_time: i32,
        ready_times: &Vec<i32>,
        due_times: &Vec<i32>,
    ) -> bool {
        let dm = distance_matrix;
        let rt = &ready_times[..];
        let dt = &due_times[..];

        if insert_pos == route.len() - 1 {
            let last_node = route[insert_pos - 1];
            let arrival_time = if last_node == 0 {
                0
            } else {
                let mut time: i32 = 0;
                let mut node: usize = 0;
                unsafe {
                    let r = route.as_slice();
                    let mut idx = 0usize;
                    while idx < insert_pos {
                        let n = *r.get_unchecked(idx);
                        if n != 0 {
                            let row_prev = dm.get_unchecked(node);
                            time += *row_prev.get_unchecked(n);
                            let rdy = *rt.get_unchecked(n);
                            if time < rdy {
                                time = rdy;
                            }
                            if time > *dt.get_unchecked(n) {
                                return false;
                            }
                            time += service_time;
                            node = n;
                        }
                        idx += 1;
                    }
                }
                time
            };

            unsafe {
                let row_last = dm.get_unchecked(last_node);
                let new_arrival = arrival_time + *row_last.get_unchecked(insert_node);
                if new_arrival > *dt.get_unchecked(insert_node) {
                    return false;
                }

                let departure = new_arrival.max(*rt.get_unchecked(insert_node)) + service_time;
                let row_insert = dm.get_unchecked(insert_node);
                let final_arrival = departure + *row_insert.get_unchecked(0);

                return final_arrival <= *dt.get_unchecked(0);
            }
        }

        let mut curr_time: i32 = 0;
        let mut curr_node: usize = 0;

        unsafe {
            let r = route.as_slice();
            let mut idx = 0usize;
            while idx < insert_pos {
                let node = *r.get_unchecked(idx);
                if node != 0 {
                    let row_curr = dm.get_unchecked(curr_node);
                    curr_time += *row_curr.get_unchecked(node);

                    if curr_time > *dt.get_unchecked(node) {
                        return false;
                    }

                    let ready = *rt.get_unchecked(node);
                    if curr_time < ready {
                        curr_time = ready;
                    }
                    curr_time += service_time;
                    curr_node = node;
                }
                idx += 1;
            }

            let row_curr = dm.get_unchecked(curr_node);
            curr_time += *row_curr.get_unchecked(insert_node);
            if curr_time > *dt.get_unchecked(insert_node) {
                return false;
            }

            let ready_ins = *rt.get_unchecked(insert_node);
            if curr_time < ready_ins {
                curr_time = ready_ins;
            }
            curr_time += service_time;
            curr_node = insert_node;

            let len = route.len();
            let mut idx2 = insert_pos;
            while idx2 < len {
                let node = *r.get_unchecked(idx2);
                if node != 0 {
                    let row_curr2 = dm.get_unchecked(curr_node);
                    curr_time += *row_curr2.get_unchecked(node);

                    if curr_time > *dt.get_unchecked(node) {
                        return false;
                    }

                    let ready2 = *rt.get_unchecked(node);
                    if curr_time < ready2 {
                        curr_time = ready2;
                    }
                    curr_time += service_time;
                    curr_node = node;
                }
                idx2 += 1;
            }
        }

        true
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
            25
        } else {
            (route.len() / 2).min(20)
        };

        while improved && iteration < max_iterations {
            improved = false;
            iteration += 1;

            let dm = distance_matrix;
            let len = best_route.len();
            let end = len - 1;
            for i in 1..end - 1 {
                for j in i + 2..end {
                    let prev = best_route[i - 1];
                    let a = best_route[i];
                    let b = best_route[j];
                    let next = best_route[j + 1];
                    let row_prev = &dm[prev];
                    let row_a = &dm[a];
                    let row_b = &dm[b];
                    let delta = (row_prev[b] + row_a[next]) - (row_prev[a] + row_b[next]);
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
        let max_iterations = if best_route.len() > 100 { 2 } else { 3 };

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

                        let row_prev_before = &dm[prev_before];
                        let row_last = &dm[last];
                        let row_prev_ins = &dm[prev_ins];
                        let delta_remove_segment = row_prev_before[after_segment]
                            - row_prev_before[first]
                            - row_last[after_segment];
                        let delta_insert_segment =
                            row_prev_ins[first] + row_last[succ_ins] - row_prev_ins[succ_ins];
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
                                std::mem::swap(&mut best_route, &mut candidate);
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
    pub fn update_node_positions_for_routes(
        node_positions: &mut Vec<(usize, usize)>,
        routes: &Vec<Vec<usize>>,
        route_indices: &[usize],
    ) {
        for &route_idx in route_indices {
            let route = &routes[route_idx];
            let end = route.len() - 1;
            for (j, &node) in route[1..end].iter().enumerate() {
                node_positions[node] = (route_idx, j + 1);
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
        let num_nodes = challenge.num_nodes;
        let max_capacity = challenge.max_capacity;
        let demands = &challenge.demands;
        let distance_matrix = &challenge.distance_matrix;
        let service_time = challenge.service_time;
        let ready_times = &challenge.ready_times;
        let due_times = &challenge.due_times;
        let mut routes: Vec<Vec<usize>> = Vec::with_capacity(num_nodes);

        let mut nodes: Vec<usize> = (1..num_nodes).collect();

        let strategy_idx = (num_nodes + max_capacity as usize + service_time as usize) % 5;
        match strategy_idx {
            0 => nodes.sort_by(|a, b| distance_matrix[0][*a].cmp(&distance_matrix[0][*b])),
            1 => nodes.sort_by(|a, b| demands[*b].cmp(&demands[*a])),
            2 => nodes.sort_by(|a, b| ready_times[*a].cmp(&ready_times[*b])),
            3 => nodes.sort_by(|a, b| due_times[*a].cmp(&due_times[*b])),
            _ => nodes.sort_by(|a, b| {
                let urgency_a = due_times[*a] - ready_times[*a];
                let urgency_b = due_times[*b] - ready_times[*b];
                urgency_a.cmp(&urgency_b)
            }),
        }

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
            let mut route_demand = demands[node];

            let mut insertion_attempts = 0;
            let max_insertion_attempts = if num_nodes >= 2000 {
                4
            } else if num_nodes >= 1500 {
                3
            } else {
                2
            };
            let mut candidates: Vec<usize> = Vec::with_capacity(num_nodes);

            while insertion_attempts < max_insertion_attempts {
                candidates.clear();

                for n in 1..num_nodes {
                    if remaining[n] && route_demand + demands[n] <= max_capacity {
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
                    route_demand += demands[best_node];
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

        let mut route_distances: Vec<i32> = Vec::with_capacity(best_routes.len());
        let mut best_distance: i32 = 0;
        for r in &best_routes {
            let d = utils::calculate_route_distance(r, distance_matrix);
            route_distances.push(d);
            best_distance += d;
        }
        let mut improved = true;
        let mut iteration_count = 0;
        let max_total_iterations = if num_nodes >= 4000 {
            110
        } else if num_nodes >= 3500 {
            95
        } else if num_nodes >= 2500 {
            80
        } else if num_nodes >= 2000 {
            65
        } else if num_nodes >= 1500 {
            50
        } else {
            45
        };

        let neighbor_count = if num_nodes >= 4000 {
            9usize
        } else if num_nodes >= 3500 {
            8usize
        } else if num_nodes >= 2500 {
            7usize
        } else if num_nodes >= 2000 {
            6usize
        } else if num_nodes >= 1500 {
            4usize
        } else {
            3usize
        };
        let pool = if num_nodes >= 4000 {
            48usize
        } else if num_nodes >= 3500 {
            44usize
        } else if num_nodes >= 2500 {
            40usize
        } else if num_nodes >= 2000 {
            36usize
        } else if num_nodes >= 1500 {
            24usize
        } else {
            18usize
        };
        let mut proximity_pairs: Vec<(i32, usize, usize)> =
            Vec::with_capacity(num_nodes * neighbor_count);
        let mut pool_vec: Vec<(i32, usize)> = Vec::with_capacity(pool);
        let mut bests: Vec<(i32, usize)> = Vec::with_capacity(neighbor_count);
        for i in 1..num_nodes {
            pool_vec.clear();
            let mut max_idx: usize = 0;
            let mut max_d: i32 = i32::MIN;
            let di = &distance_matrix[i];
            for j in 1..num_nodes {
                if j == i {
                    continue;
                }
                let d = di[j];
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
                    for idx in 0..pool_vec.len() {
                        let dd = pool_vec[idx].0;
                        if dd > new_max_d {
                            new_max_d = dd;
                            new_max_idx = idx;
                        }
                    }
                    max_d = new_max_d;
                    max_idx = new_max_idx;
                }
            }
            bests.clear();
            let ri = ready_times[i];
            let di_due = due_times[i];
            for &(_d, j) in pool_vec.iter() {
                let time_ij = di[j];
                let rj = ready_times[j];
                let dj = due_times[j];
                let expr1 = (rj - time_ij - service_time - di_due).max(0)
                    + (ri + service_time + time_ij - dj).max(0);
                let expr2 = (ri - time_ij - service_time - dj).max(0)
                    + (rj + service_time + time_ij - di_due).max(0);
                let p = time_ij + if expr1 < expr2 { expr1 } else { expr2 };
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
            for (p, j) in &bests {
                proximity_pairs.push((*p, i, *j));
            }
        }

        let mut node_positions = vec![(0, 0); num_nodes];
        for (i, route) in best_routes.iter().enumerate() {
            for (j, &node) in route[1..route.len() - 1].iter().enumerate() {
                node_positions[node] = (i, j + 1);
            }
        }

        let mut route_demands = calculate_route_demands(&best_routes, demands);
        let mut cand_r1: Vec<usize> = Vec::with_capacity(64);
        let mut cand_r2: Vec<usize> = Vec::with_capacity(64);
        let mut swap_r1: Vec<usize> = Vec::with_capacity(64);
        let mut swap_r2: Vec<usize> = Vec::with_capacity(64);
        while improved && iteration_count < max_total_iterations {
            improved = false;
            iteration_count += 1;

            for (_corr, node, node2) in &proximity_pairs {
                let node = *node;
                let node2 = *node2;
                let (route1_idx, pos1) = node_positions[node];
                let (route2_idx, pos2) = node_positions[node2];
                if route1_idx == route2_idx {
                    continue;
                }

                let target_route_demand = route_demands[route2_idx];
                if target_route_demand + demands[node] <= max_capacity {
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
                            let row_prev1 = &distance_matrix[prev1];
                            let row_prev2 = &distance_matrix[prev2];
                            let row_node = &distance_matrix[node];
                            let delta_remove = row_prev1[next1] - row_prev1[node] - row_node[next1];
                            let delta_insert = row_prev2[node] + row_node[next2] - row_prev2[next2];
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

                            let r1 = apply_size_filtered_local_search(
                                &cand_r1,
                                distance_matrix,
                                service_time,
                                ready_times,
                                due_times,
                            );
                            let r2 = apply_size_filtered_local_search(
                                &cand_r2,
                                distance_matrix,
                                service_time,
                                ready_times,
                                due_times,
                            );

                            if r1.len() >= 3
                                && r1[0] == 0
                                && r1[r1.len() - 1] == 0
                                && r2.len() >= 3
                                && r2[0] == 0
                                && r2[r2.len() - 1] == 0
                            {
                                let old_d1 = route_distances[route1_idx];
                                let old_d2 = route_distances[route2_idx];
                                let new_d1 = utils::calculate_route_distance(&r1, distance_matrix);
                                let new_d2 = utils::calculate_route_distance(&r2, distance_matrix);
                                let new_distance =
                                    best_distance - old_d1 - old_d2 + new_d1 + new_d2;
                                if new_distance < best_distance {
                                    best_distance = new_distance;

                                    best_routes[route1_idx] = r1;
                                    best_routes[route2_idx] = r2;

                                    route_distances[route1_idx] = new_d1;
                                    route_distances[route2_idx] = new_d2;
                                    route_demands[route1_idx] -= demands[node];
                                    route_demands[route2_idx] += demands[node];

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

                if route_demands[route1_idx] - demands[node] + demands[node2] <= max_capacity
                    && route_demands[route2_idx] - demands[node2] + demands[node] <= max_capacity
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
                        let row_prev1 = &distance_matrix[prev1];
                        let row_prev2 = &distance_matrix[prev2];
                        let row_node = &distance_matrix[node];
                        let row_node2 = &distance_matrix[node2];
                        let delta1 =
                            row_prev1[node2] + row_node2[next1] - row_prev1[node] - row_node[next1];
                        let delta2 =
                            row_prev2[node] + row_node[next2] - row_prev2[node2] - row_node2[next2];
                        if delta1 + delta2 >= 0 {
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
                            let new_distance = best_distance - old_d1 - old_d2 + new_d1 + new_d2;
                            if new_distance < best_distance {
                                best_distance = new_distance;

                                let old_vec1 = std::mem::replace(
                                    &mut best_routes[route1_idx],
                                    std::mem::take(&mut swap_r1),
                                );
                                let old_vec2 = std::mem::replace(
                                    &mut best_routes[route2_idx],
                                    std::mem::take(&mut swap_r2),
                                );

                                route_distances[route1_idx] = new_d1;
                                route_distances[route2_idx] = new_d2;
                                route_demands[route1_idx] =
                                    route_demands[route1_idx] - demands[node] + demands[node2];
                                route_demands[route2_idx] =
                                    route_demands[route2_idx] - demands[node2] + demands[node];

                                update_node_positions_for_routes(
                                    &mut node_positions,
                                    &best_routes,
                                    &[route1_idx, route2_idx],
                                );

                                swap_r1 = old_vec1;
                                swap_r2 = old_vec2;

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
                            best_distance = total_distance;

                            best_routes[route_idx] = improved_route;

                            route_distances[route_idx] = new_d;
                            update_node_positions_for_routes(
                                &mut node_positions,
                                &best_routes,
                                &[route_idx],
                            );
                            improved = true;
                            break;
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

pub fn help() {
    println!("No help information available.");
}
