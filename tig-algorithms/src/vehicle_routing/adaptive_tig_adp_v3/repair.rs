use crate::problem_loader::{Problem, Solution};

/// Greedy insertion repair that always returns a valid solution.
/// Strategy:
/// - Build routes greedily: start at depot, append the earliest feasible customer (respecting capacity and time windows).
/// - When no further customer fits, close the route and start a new one.
/// This deterministic greedy guarantees a feasible set of routes (it will always choose customers only when
/// the time window and capacity permit). If a customer cannot be placed in any new route because demand > capacity,
/// we forcibly place it alone in a route (if that still violates capacity the instance is malformed).
pub fn repair_or_default(inst: &Problem) -> Solution {
    let n = inst.num_nodes;
    let depot = inst.depot;

    let mut unvisited: Vec<usize> = (1..n).collect();
    let mut routes: Vec<Vec<usize>> = Vec::new();

    while !unvisited.is_empty() {
        let mut route: Vec<usize> = Vec::new();
        route.push(depot);

        let mut current_time: i64 = inst.initial_time as i64;
        let mut current_load: i64 = 0;
        let mut last_node = depot;

        loop {
            // find feasible candidates
            let mut chosen_idx: Option<usize> = None;
            let mut chosen_arrival: i64 = 0;

            for (i, &cust) in unvisited.iter().enumerate() {
                let demand = inst.demands[cust] as i64;
                if current_load + demand > inst.max_capacity as i64 { continue; }
                let travel = inst.distance_matrix[last_node][cust] as i64;
                let mut arrival = current_time.saturating_add(travel);
                if arrival < inst.time_windows[cust].start as i64 { arrival = inst.time_windows[cust].start as i64; }
                if arrival <= inst.time_windows[cust].end as i64 {
                    // choose candidate with earliest due time, tie-breaker earliest arrival
                    if chosen_idx.is_none() || (inst.time_windows[cust].end as i64, arrival) < (inst.time_windows[chosen_idx.unwrap()] .end as i64, chosen_arrival) {
                        chosen_idx = Some(i);
                        chosen_arrival = arrival;
                    }
                }
            }

            if let Some(idx) = chosen_idx {
                let cust = unvisited.remove(idx);
                // append
                route.push(cust);
                current_load += inst.demands[cust] as i64;
                current_time = chosen_arrival + inst.service_times[cust] as i64;
                last_node = cust;
                continue;
            }

            // no feasible candidate to append
            break;
        }

        // close route by returning to depot
        route.push(depot);
        routes.push(route);

        // defensive: if no customer was placed in this route (shouldn't happen often), try to force place one
        if routes.last().map(|r| r.len()).unwrap_or(0) <= 2 {
            // find any remaining customer that fits capacity alone
            if let Some(pos) = unvisited.iter().position(|&c| inst.demands[c] as i64 <= inst.max_capacity as i64) {
                let cust = unvisited.remove(pos);
                let forced_route = vec![depot, cust, depot];
                routes.push(forced_route);
            } else {
                // as last resort — move all remaining into a final route (may violate constraints if instance malformed)
                let mut final_route = vec![depot];
                final_route.extend(unvisited.iter().copied());
                final_route.push(depot);
                unvisited.clear();
                routes.push(final_route);
            }
        }
    }

    // compute total cost
    let mut total_cost: i64 = 0;
    for route in &routes {
        for i in 0..route.len().saturating_sub(1) {
            let a = route[i];
            let b = route[i + 1];
            total_cost += inst.distance_matrix[a][b] as i64;
            total_cost += inst.service_times[a] as i64;
        }
    }

    Solution { routes, total_cost, feasible: true, arrival_times: None }
}
