use crate::problem_loader::{Problem, Solution};

/// Structured validation: returns `Ok(())` when valid, otherwise `Err(vec_of_reasons)`.
pub fn validate_solution(sol: &Solution, inst: &Problem) -> Result<(), Vec<String>> {
    let mut errors: Vec<String> = Vec::new();

    if inst.num_nodes == 0 {
        errors.push("instance has zero nodes".to_string());
        return Err(errors);
    }

    let n_nodes = inst.num_nodes;

    // Collect visited customers across all routes
    let mut seen = vec![0usize; n_nodes];
    for (ri, route) in sol.routes.iter().enumerate() {
        if route.is_empty() {
            errors.push(format!("route {} is empty", ri));
            continue;
        }
        if route[0] != inst.depot {
            errors.push(format!("route {} does not start at depot", ri));
        }
        if *route.last().unwrap() != inst.depot {
            errors.push(format!("route {} does not end at depot", ri));
        }

        for &node in route.iter().skip(1).take(route.len().saturating_sub(2)) {
            if node == inst.depot {
                errors.push(format!("route {} contains depot in middle", ri));
            }
            if node >= n_nodes {
                errors.push(format!("route {} contains invalid node index {}", ri, node));
            } else {
                seen[node] += 1;
            }
        }
    }

    // check each customer once
    for i in 1..n_nodes {
        if seen[i] != 1 {
            errors.push(format!("customer {} visited {} times", i, seen[i]));
        }
    }

    // Capacity and time window checks: conservative scan
    for (ri, route) in sol.routes.iter().enumerate() {
        let mut time = inst.initial_time as i64;
        let mut load: i64 = 0;
        for w in 0..route.len().saturating_sub(1) {
            let a = route[w];
            let b = route[w+1];
            if a >= inst.num_nodes || b >= inst.num_nodes {
                errors.push(format!("route {} contains invalid transition {}->{}", ri, a, b));
                continue;
            }
            let travel = inst.distance_matrix[a][b] as i64;
            if w > 0 { time += inst.service_times[a] as i64; }
            time += travel;
            let tws = inst.time_windows[b].start as i64;
            if time < tws { time = tws; }
            if time > inst.time_windows[b].end as i64 {
                errors.push(format!("route {} misses time window at node {} (time {})", ri, b, time));
            }
            load += inst.demands[b] as i64;
            if load > inst.max_capacity as i64 {
                errors.push(format!("route {} exceeds capacity (load {})", ri, load));
            }
        }
    }

    if errors.is_empty() { Ok(()) } else { Err(errors) }
}
