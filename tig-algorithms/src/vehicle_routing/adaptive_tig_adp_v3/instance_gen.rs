use rand::prelude::*;
use rand_distr::{Normal, Distribution};
use std::collections::HashSet;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instance {
    pub depot: (i32,i32),
    pub customers: Vec<(i32,i32)>,
    pub demands: Vec<i32>,
    pub service_time: Vec<i32>,
    pub tw_start: Vec<i32>,
    pub tw_end: Vec<i32>,
}

/// Generate an instance following the spec in the user's message.
/// - `n_customers` does NOT include the depot.
/// - `seed` for deterministic generation.
pub fn generate_instance(n_customers: usize, seed: u64) -> Instance {
    let mut rng = StdRng::seed_from_u64(seed);
    let grid_size = 1000i32;
    let depot = (500i32, 500i32);

    // number of cluster seeds S ~ UD[3,8]
    let s: usize = rng.gen_range(3..=8);

    // place S cluster seeds uniformly on grid, unique
    let mut used = HashSet::new();
    used.insert(depot);
    let mut seeds: Vec<(i32,i32)> = Vec::new();
    while seeds.len() < s {
        let x = rng.gen_range(0..=grid_size);
        let y = rng.gen_range(0..=grid_size);
        if used.insert((x,y)) {
            seeds.push((x,y));
        }
    }

    let mut customers: Vec<(i32,i32)> = Vec::with_capacity(n_customers);

    // We'll ensure uniqueness by checking `used`.
    let sd = 60.0_f64;
    let normal = Normal::new(0.0, sd).unwrap();

    // Track explicit assignment to cluster seeds: None means uniform/random
    let mut assigned_seed: Vec<Option<usize>> = Vec::with_capacity(n_customers);
    for _ in 0..n_customers {
        // decide cluster membership: 50% chance
        let clustered = rng.gen_bool(0.5);
        if clustered {
            // pick a random seed
            let seed_idx = seeds.choose(&mut rng).map(|s| seeds.iter().position(|x| x == s).unwrap()).unwrap();
            let (sx, sy) = seeds[seed_idx];
            // sample via truncated normal around seed until unique and in bounds
            let mut attempts = 0;
            loop {
                attempts += 1;
                let dx = normal.sample(&mut rng).round() as i32;
                let dy = normal.sample(&mut rng).round() as i32;
                let x = (sx + dx).clamp(0, grid_size);
                let y = (sy + dy).clamp(0, grid_size);
                if used.insert((x,y)) { customers.push((x,y)); assigned_seed.push(Some(seed_idx)); break; }
                if attempts > 1000 {
                    // fallback to uniform if clustering fails
                    let mut uu_attempt = 0;
                    loop {
                        uu_attempt += 1;
                        let ux = rng.gen_range(0..=grid_size);
                        let uy = rng.gen_range(0..=grid_size);
                        if used.insert((ux,uy)) { customers.push((ux,uy)); assigned_seed.push(None); break; }
                        if uu_attempt > 10000 { panic!("Unable to sample unique customer positions"); }
                    }
                    break;
                }
            }
        } else {
            // uniform random placement
            let mut attempts = 0;
            loop {
                attempts += 1;
                let x = rng.gen_range(0..=grid_size);
                let y = rng.gen_range(0..=grid_size);
                if used.insert((x,y)) { customers.push((x,y)); assigned_seed.push(None); break; }
                if attempts > 10000 { panic!("Unable to sample unique customer positions"); }
            }
        }
    }

    // demands UD[1,35]
    let demands: Vec<i32> = (0..n_customers).map(|_| rng.gen_range(1..=35)).collect();
    // service times si = 10
    let service_time: Vec<i32> = vec![10i32; n_customers];

    // compute distances (Euclidean) rounded to nearest integer when needed
    // compute d0i (depot to customers)
    let d0i: Vec<f64> = customers.iter().map(|&(x,y)| {
        let dx = (x - depot.0) as f64;
        let dy = (y - depot.1) as f64;
        (dx*dx + dy*dy).sqrt()
    }).collect();

    // dav: average customer distance derived from quarter-grid formula approx 0.5214 * (grid_size/2)
    let dav = 0.5214 * ((grid_size as f64) / 2.0);

    // rav: average route size approx 11.43 per spec
    let rav = 11.43f64;

    // compute depot due time l0 = d0iF + (si + dav) * rav
    let d0i_f = d0i.iter().cloned().fold(0./0., f64::max); // max
    let l0 = d0i_f + (10.0 + dav) * rav;

    // For each customer, draw due time uniformly from [d0i, l0 - d0i - si]
    let mut tw_end: Vec<i32> = Vec::with_capacity(n_customers);
    for (i, &d) in d0i.iter().enumerate() {
        let ub = (l0 - d - (service_time[i] as f64)).max(d);
        let ub_i = ub.floor() as i32;
        let lb_i = d.ceil() as i32;
        let chosen = if ub_i <= lb_i { lb_i } else { rng.gen_range(lb_i..=ub_i) };
        tw_end.push(chosen);
    }

    // For clustered customers, adjust due times toward their seed due time using explicit assignment bookkeeping
    // Compute per-seed average due time (seed_due) from the initially drawn tw_end values of assigned customers.
    let mut seed_accum: Vec<(i64, usize)> = vec![(0i64, 0usize); seeds.len()];
    for (i, assign) in assigned_seed.iter().enumerate() {
        if let Some(si) = assign {
            seed_accum[*si].0 += tw_end[i] as i64;
            seed_accum[*si].1 += 1;
        }
    }
    let mut seed_due: Vec<i32> = vec![0i32; seeds.len()];
    for (i, &(sum, cnt)) in seed_accum.iter().enumerate() {
        if cnt == 0 { seed_due[i] = 0; } else { seed_due[i] = ((sum as f64) / (cnt as f64)).round() as i32; }
    }

    // Now adjust each clustered customer's due time to average(original, seed_due)
    for (i, assign) in assigned_seed.iter().enumerate() {
        if let Some(si) = assign {
            let orig = tw_end[i] as f64;
            let s_due = seed_due[*si] as f64;
            let new_due = ((orig + s_due) / 2.0).round() as i32;
            // ensure within original bounds [d0i, l0 - d0i - si]
            let lb = d0i[i].ceil() as i32;
            let ub = (l0 - d0i[i] - (service_time[i] as f64)).floor() as i32;
            let clipped = new_due.clamp(lb, ub);
            tw_end[i] = clipped;
        }
    }

    // ready times: depot 0, for ~50% customers set ready = due - width with width ~ UD[10,60]
    let mut tw_start: Vec<i32> = vec![0i32; n_customers];
    for i in 0..n_customers {
        if rng.gen_bool(0.5) {
            let width = rng.gen_range(10..=60) as i32;
            tw_start[i] = (tw_end[i] - width).max(0);
        } else {
            tw_start[i] = 0;
        }
    }

    Instance { depot, customers, demands, service_time, tw_start, tw_end }
}
