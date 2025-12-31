use crate::config::TrackConfig;
use crate::instance_gen;
use crate::problem_loader::Problem;
// no serde derives needed here

/// Alias requested in spec
pub type VrptwInstance = Problem;

/// Deterministic hash from (track_id, nonce_idx) -> u64 using djb2-like
pub fn deterministic_seed(track_id: &str, nonce: usize) -> u64 {
    let mut hash: u64 = 5381;
    for b in track_id.as_bytes() {
        hash = ((hash << 5).wrapping_add(hash)).wrapping_add(*b as u64);
    }
    hash = hash.wrapping_add(nonce as u64 * 0x9e3779b97f4a7c15u64);
    if hash == 0 { 1 } else { hash }
}

/// Generate a `Problem` (VRPTW) deterministically from track config and nonce.
pub fn generate_instance(
    track_id: &str,
    nonce_idx: usize,
    tracks: &std::collections::BTreeMap<String, TrackConfig>,
) -> VrptwInstance {
    let cfg = tracks.get(track_id).expect("track id must exist");
    let n_customers = cfg.num_customers;
    let seed = deterministic_seed(track_id, nonce_idx);

    // reuse existing generator: customers count = n_customers (does NOT include depot)
    let inst = instance_gen::generate_instance(n_customers, seed);

    // build distance matrix
    let n_nodes = n_customers + 1; // depot + customers
    let mut distance_matrix = vec![vec![0i32; n_nodes]; n_nodes];

    // coordinates: depot index 0, customers index 1..n
    let mut coords: Vec<(i32,i32)> = Vec::with_capacity(n_nodes);
    coords.push(inst.depot);
    coords.extend(inst.customers.iter().cloned());

    for i in 0..n_nodes {
        for j in 0..n_nodes {
            if i == j { distance_matrix[i][j] = 0; continue; }
            let a = coords[i];
            let b = coords[j];
            let dx = (a.0 - b.0) as f64;
            let dy = (a.1 - b.1) as f64;
            let d = (dx*dx + dy*dy).sqrt().round() as i32;
            distance_matrix[i][j] = d;
        }
    }

    // build time windows: prepend depot tw = (0, large)
    let mut tws: Vec<crate::problem_loader::TimeWindow> = Vec::with_capacity(n_nodes);
    tws.push(crate::problem_loader::TimeWindow { start: 0, end: 10_000_000 });
    for i in 0..n_customers {
        tws.push(crate::problem_loader::TimeWindow { start: inst.tw_start[i], end: inst.tw_end[i] });
    }

    let service_times = {
        let mut s = Vec::with_capacity(n_nodes);
        s.push(0);
        s.extend(inst.service_time.iter().cloned());
        s
    };

    let mut demands = Vec::with_capacity(n_nodes);
    demands.push(0);
    demands.extend(inst.demands.iter().cloned());

    Problem {
        name: format!("track_{}_nonce_{}", track_id, nonce_idx),
        num_nodes: n_nodes,
        depot: 0,
        max_capacity: 100,
        initial_time: 0,
        time_windows: tws,
        service_times,
        demands,
        distance_matrix,
        initial_route: None,
        config: None,
    }
}
