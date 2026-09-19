// Starting partition inferred on the device from the hypergraph's connectivity: two levels of
// recursive bisection, then the constructed partition places each node inside its quarter.

use cudarc::driver::{safe::LaunchConfig, CudaModule, CudaSlice, CudaStream, PushKernelArg};
use std::sync::Arc;
use tig_challenges::hypergraph::Challenge;

// Bisection levels; the device kernels handle two sets per level.
const DEPTH: u32 = 2;

pub struct Params {
    pub iters: u32,
    pub slack_pm: usize,
    pub tau_tenths: u32,
}

impl Params {
    pub fn read(hp: &super::Hp) -> Params {
        Params {
            iters: hp.int("inf_iters", 1, 1024, 32) as u32,
            slack_pm: hp.int("inf_slack", 0, 100, 6) as usize,
            tau_tenths: hp.int("inf_tau", 1, 1000, 30) as u32,
        }
    }
}

fn hash64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

// Returns None when the instance is too small or a level cannot be split within the slack;
// the caller then keeps its constructed partition.
pub fn starting_partition(
    p: &Params,
    challenge: &Challenge,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    prior: &[i32],
) -> anyhow::Result<Option<Vec<i32>>> {
    let n = challenge.num_nodes as usize;
    let m = challenge.num_hyperedges as usize;
    let num_parts = challenge.num_parts;
    let nb = 1u32 << DEPTH;
    if n < 64 || m == 0 || num_parts < nb || prior.len() != n {
        return Ok(None);
    }
    let tau = p.tau_tenths as f32 / 10.0;
    let seed = u64::from_le_bytes([
        challenge.seed[0], challenge.seed[1], challenge.seed[2], challenge.seed[3],
        challenge.seed[4], challenge.seed[5], challenge.seed[6], challenge.seed[7],
    ]);

    let k_edge = module.load_function("inf_edge_sums")?;
    let k_deg = module.load_function("inf_node_deg")?;
    let k_step = module.load_function("inf_node_step")?;
    let k_red = module.load_function("inf_reduce2")?;
    let k_mu = module.load_function("inf_mu")?;
    let k_count = module.load_function("inf_count_below")?;
    let k_split = module.load_function("inf_split")?;

    let bs = 256u32;
    let node_cfg = LaunchConfig { grid_dim: ((n as u32 + bs - 1) / bs, 1, 1), block_dim: (bs, 1, 1), shared_mem_bytes: 0 };
    let edge_cfg = LaunchConfig { grid_dim: ((m as u32 + bs - 1) / bs, 1, 1), block_dim: (bs, 1, 1), shared_mem_bytes: 0 };
    let red_cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
    let one_cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };

    let mut set_host = vec![0i32; n];
    let mut d_set = stream.memcpy_stod(&set_host)?;
    let mut d_ya = stream.alloc_zeros::<f32>(n)?;
    let mut d_yb = stream.alloc_zeros::<f32>(n)?;
    let mut d_mu = stream.alloc_zeros::<f32>(2)?;
    let mut d_cnt = stream.alloc_zeros::<i32>(2 * m)?;
    let mut d_z = stream.alloc_zeros::<f32>(2 * m)?;
    let mut d_wdeg = stream.alloc_zeros::<f32>(n)?;
    let mut d_w = stream.alloc_zeros::<f32>(2)?;
    let mut d_sum = stream.alloc_zeros::<f32>(2)?;
    let mut d_count = stream.alloc_zeros::<i32>(1)?;
    let mut d_set2 = stream.memcpy_stod(&set_host)?;
    let zero2 = [0f32; 2];
    let zero1 = [0i32; 1];

    macro_rules! edge_sums {
        ($y:expr) => {{
            unsafe {
                stream.launch_builder(&k_edge)
                    .arg(&(m as i32)).arg(&challenge.d_hyperedge_offsets).arg(&challenge.d_hyperedge_nodes)
                    .arg(&d_set).arg($y).arg(&d_mu).arg(&mut d_cnt).arg(&mut d_z)
                    .launch(edge_cfg.clone())?;
            }
        }};
    }
    macro_rules! recentre {
        ($y:expr) => {{
            unsafe {
                stream.launch_builder(&k_red)
                    .arg(&(n as i32)).arg(&d_set).arg(&d_wdeg).arg($y).arg(&1i32).arg(&mut d_sum)
                    .launch(red_cfg.clone())?;
                stream.launch_builder(&k_mu).arg(&d_sum).arg(&d_w).arg(&mut d_mu).launch(one_cfg.clone())?;
            }
        }};
    }
    macro_rules! step {
        ($y_in:expr, $y_out:expr) => {{
            edge_sums!($y_in);
            unsafe {
                stream.launch_builder(&k_step)
                    .arg(&(n as i32)).arg(&challenge.d_node_offsets).arg(&challenge.d_node_hyperedges)
                    .arg(&d_set).arg(&d_cnt).arg(&d_z).arg(&d_mu).arg(&d_wdeg).arg($y_in).arg($y_out)
                    .launch(node_cfg.clone())?;
            }
            recentre!($y_out);
        }};
    }
    // Nodes of set `s` whose centred value is below `t`, or below `t2` with an index below `vt`.
    let count_below = |d_set: &CudaSlice<i32>, d_y: &CudaSlice<f32>, d_mu: &CudaSlice<f32>,
                       d_count: &mut CudaSlice<i32>, s: i32, t: f32, t2: f32, vt: i32| -> anyhow::Result<usize> {
        stream.memcpy_htod(&zero1, d_count)?;
        unsafe {
            stream.launch_builder(&k_count)
                .arg(&(n as i32)).arg(d_set).arg(d_y).arg(d_mu).arg(&s).arg(&t).arg(&t2).arg(&vt).arg(&mut *d_count)
                .launch(node_cfg.clone())?;
        }
        Ok(stream.memcpy_dtov(&*d_count)?[0] as usize)
    };

    for level in 0..DEPTH {
        let nsets = 1usize << level;
        let salt = 0x5EED_0000u64 | level as u64;
        let x0: Vec<f32> = (0..n as u64)
            .map(|v| {
                let h = hash64((v + 1).wrapping_mul(0x9E3779B97F4A7C15) ^ seed ^ salt);
                ((h >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect();
        stream.memcpy_htod(&x0, &mut d_ya)?;
        stream.memcpy_htod(&zero2, &mut d_mu)?;
        edge_sums!(&d_ya);
        unsafe {
            stream.launch_builder(&k_deg)
                .arg(&(n as i32)).arg(&challenge.d_node_offsets).arg(&challenge.d_node_hyperedges)
                .arg(&d_set).arg(&d_cnt).arg(&tau).arg(&mut d_wdeg)
                .launch(node_cfg.clone())?;
            stream.launch_builder(&k_red)
                .arg(&(n as i32)).arg(&d_set).arg(&d_wdeg).arg(&d_wdeg).arg(&0i32).arg(&mut d_w)
                .launch(red_cfg.clone())?;
        }
        recentre!(&d_ya);
        let mut a_is_cur = true;
        for _ in 0..p.iters {
            if a_is_cur {
                step!(&d_ya, &mut d_yb);
            } else {
                step!(&d_yb, &mut d_ya);
            }
            a_is_cur = !a_is_cur;
        }
        let d_cur: &CudaSlice<f32> = if a_is_cur { &d_ya } else { &d_yb };
        let mut sizes = vec![0usize; nsets];
        for v in 0..n {
            let s = set_host[v];
            if s >= 0 && (s as usize) < nsets {
                sizes[s as usize] += 1;
            }
        }
        for s in 0..nsets {
            let total = sizes[s];
            if total < 4 {
                return Ok(None);
            }
            let half = total / 2;
            let slack = (total * p.slack_pm / 1000).max(1);
            let lo = half - slack.min(half);
            let hi = (half + slack).min(total);
            let (mut t, mut t2, mut vt) = (0f32, 0f32, 0i32);
            let c0 = count_below(&d_set, d_cur, &d_mu, &mut d_count, s as i32, 0.0, 0.0, 0)?;
            if c0 < lo || c0 > hi {
                // Threshold bisection first, then an index split between two thresholds.
                let (mut a, mut b) = (-8f32, 8f32);
                let mut found = false;
                for _ in 0..64 {
                    let mid = 0.5 * (a + b);
                    let c = count_below(&d_set, d_cur, &d_mu, &mut d_count, s as i32, mid, mid, 0)?;
                    if c >= lo && c <= hi {
                        t = mid;
                        t2 = mid;
                        found = true;
                        break;
                    }
                    if c < lo { a = mid; } else { b = mid; }
                }
                if !found {
                    let (mut va, mut vb) = (0i32, n as i32);
                    for _ in 0..40 {
                        let vm = (va + vb) / 2;
                        let c = count_below(&d_set, d_cur, &d_mu, &mut d_count, s as i32, a, b, vm)?;
                        if c >= lo && c <= hi {
                            t = a;
                            t2 = b;
                            vt = vm;
                            found = true;
                            break;
                        }
                        if c < lo { va = vm; } else { vb = vm; }
                        if vb - va <= 1 { break; }
                    }
                }
                if !found {
                    return Ok(None);
                }
            }
            unsafe {
                stream.launch_builder(&k_split)
                    .arg(&(n as i32)).arg(&d_set).arg(d_cur).arg(&d_mu).arg(&(s as i32)).arg(&t).arg(&t2).arg(&vt).arg(&mut d_set2)
                    .launch(node_cfg.clone())?;
            }
        }
        set_host = stream.memcpy_dtov(&d_set2)?;
        stream.memcpy_htod(&set_host, &mut d_set)?;
    }

    let q = num_parts / nb;
    let mut part = vec![0i32; n];
    let mut count = vec![0u32; num_parts as usize];
    for v in 0..n {
        let b = set_host[v];
        if b < 0 || b as u32 >= nb {
            return Ok(None);
        }
        let within = prior[v].max(0) as u32 % q;
        let p = b as u32 * q + within;
        part[v] = p as i32;
        count[p as usize] += 1;
    }
    if count.iter().any(|&c| c < 1) {
        return Ok(None);
    }
    Ok(Some(part))
}
