use super::params::{Knobs, Track};
use super::{FILTER_THREADS, FINISH_THREADS, LCG_INC, SELECT_THREADS};
use cudarc::driver::{safe::LaunchConfig, CudaModule, CudaSlice, CudaStream, PushKernelArg};
use std::sync::Arc;
use tig_challenges::hypergraph::Challenge;

type Header = [u64; 32];
type Config = [u64; 17];

const STATE: usize = 0;
const ROUND: usize = 1;
const FULL_FLAGS: usize = 4;
const LISTED: usize = 6;
const INCREMENTAL: usize = 8;
const LIMIT: usize = 14;
const SLACK: usize = 15;
const PROB_NUM: usize = 16;
const PROB_DEN: usize = 17;
const GC: usize = 30;

const SUSPENDED: u64 = 1;
const DONE: u64 = 2;

const EDGES: usize = 3;
const COOL_ROUNDS: usize = 5;
const COOL_POW: usize = 6;
const COOL_PCT: usize = 7;
const MOVE_LIMIT: usize = 9;
const SLACK_SCALE: usize = 11;
const INCREMENTAL_FLAGS: usize = 16;

fn schedule(h: &mut Header, c: &Config) {
    let r = h[ROUND];
    let left = c[COOL_ROUNDS].saturating_sub(r);
    let (mut num, mut den) = (left, c[COOL_ROUNDS]);
    if c[COOL_POW] >= 2 {
        num = num.saturating_mul(left);
        den = den.saturating_mul(c[COOL_ROUNDS]);
    }
    if c[COOL_POW] == 3 {
        num = num.saturating_mul(left);
        den = den.saturating_mul(c[COOL_ROUNDS]);
    }
    den = den.saturating_mul(100) / c[COOL_PCT];
    h[PROB_NUM] = num;
    h[PROB_DEN] = den;
    h[PROB_DEN + 1] = den.saturating_mul(6);
    h[PROB_DEN + 2] = den.saturating_mul(11);
    h[PROB_DEN + 3] = den.saturating_mul(16);
    h[LIMIT] = if r < 50 {
        c[MOVE_LIMIT] / 2
    } else if r < 200 {
        c[MOVE_LIMIT] * 3 / 4
    } else {
        c[MOVE_LIMIT] / 4
    };
    h[SLACK] = (if r < 64 {
        8
    } else if r < 256 {
        4
    } else {
        2
    }) * c[SLACK_SCALE];
    h[INCREMENTAL] =
        u64::from(c[INCREMENTAL_FLAGS] != 0 && h[FULL_FLAGS] == 0 && h[LISTED] < c[EDGES]);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    challenge: &Challenge,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    track: &Track,
    knobs: &Knobs,
    batch: usize,
    gc: bool,
    cool_rounds: usize,
    cool_pow: u32,
    cool_pct: u64,
    pert_rounds: usize,
    slack_scale: usize,
    filt_grid: u32,
    flags_cfg: LaunchConfig,
    hlist_cfg: LaunchConfig,
    moves_cfg: LaunchConfig,
    large_ids: Option<&CudaSlice<i32>>,
    large_count: i32,
    order: &CudaSlice<i32>,
    hubs: i32,
    lanes4: i32,
    lanes2: i32,
    part: &mut CudaSlice<i32>,
    sizes: &mut CudaSlice<i32>,
    tabu: &mut CudaSlice<i32>,
    flags: &mut CudaSlice<u64>,
    priorities: &mut CudaSlice<i32>,
    block_max: &mut CudaSlice<i32>,
    stats: &mut CudaSlice<i32>,
    block_sel: &mut CudaSlice<i32>,
    filt_out: &mut CudaSlice<u64>,
    sort_buf: &mut CudaSlice<u64>,
    sorted: &mut CudaSlice<u64>,
    picks: &mut CudaSlice<u64>,
    host_part: &mut [i32],
    host_sizes: &mut [i32],
    host_tabu: &mut [i32],
    offsets: &[i32],
) -> anyhow::Result<()> {
    let n = challenge.num_nodes as usize;
    let np = challenge.num_parts as usize;
    let ne = challenge.num_hyperedges as usize;
    let capacity = challenge.max_part_size as i32;
    let c: Config = [
        n as u64,
        np as u64,
        capacity as u64,
        ne as u64,
        knobs.refinement_rounds as u64,
        cool_rounds as u64,
        cool_pow as u64,
        cool_pct,
        pert_rounds as u64,
        knobs.move_limit as u64,
        track.extra_window as u64,
        slack_scale as u64,
        knobs.tabu_tenure as u64,
        track.tabu_fail_mark_len as u64,
        track.tabu_fail_tenure as u64,
        knobs.max_stagnant_rounds as u64,
        u64::from(track.k_edge_flags_incr.is_some()),
    ];
    let mut h: Header = [0; 32];
    h[FULL_FLAGS] = 1;
    h[GC] = u64::from(gc);
    schedule(&mut h, &c);
    let dc = stream.memcpy_stod(&c)?;
    let mut dh = stream.memcpy_stod(&h)?;
    let mut pairs = stream.alloc_zeros::<i32>(2 * n)?;
    let mut list = stream.alloc_zeros::<i32>((offsets[n] as usize).max(1))?;
    let mut stamp = stream.alloc_zeros::<i32>(ne)?;
    let expand = match track.k_expand {
        Some(name) => Some(module.load_function(name)?),
        None => None,
    };
    let incr = match track.k_edge_flags_incr {
        Some(name) => Some(module.load_function(name)?),
        None => None,
    };
    let flag_kernel = module.load_function(track.k_edge_flags)?;
    let moves = module.load_function(track.k_moves)?;
    let stat = module.load_function("mica_filt_stats")?;
    let pick = module.load_function("mica_filt_pick4")?;
    let sort = module.load_function("mica_sortsel")?;
    let finish = module.load_function("grouped_finish")?;
    let resume = module.load_function("grouped_resume")?;
    let filt_cfg = LaunchConfig {
        grid_dim: (filt_grid, 1, 1),
        block_dim: (FILTER_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let sort_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (SELECT_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let finish_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (FINISH_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let one = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let ni = n as i32;
    let npi = np as i32;
    let nei = ne as i32;
    let gi = filt_grid as i32;
    let chunk = ((n as u32 + filt_grid - 1) / filt_grid) as i32;
    let extra = track.extra_window.min(i32::MAX as usize) as i32;
    stream.memcpy_htod(host_tabu, tabu)?;
    while h[STATE] != DONE {
        if h[STATE] == SUSPENDED {
            unsafe {
                stream
                    .launch_builder(&resume)
                    .arg(&mut dh)
                    .launch(one.clone())?;
            }
        }
        for _ in 0..batch {
            unsafe {
                if let (Some(expand), Some(incr)) = (expand.as_ref(), incr.as_ref()) {
                    stream
                        .launch_builder(expand)
                        .arg(&0i32)
                        .arg(&pairs)
                        .arg(&challenge.d_node_hyperedges)
                        .arg(&challenge.d_node_offsets)
                        .arg(&mut list)
                        .arg(&dh)
                        .launch(hlist_cfg.clone())?;
                    stream
                        .launch_builder(incr)
                        .arg(&0i32)
                        .arg(&list)
                        .arg(&0i32)
                        .arg(&mut stamp)
                        .arg(&nei)
                        .arg(&ni)
                        .arg(&challenge.d_hyperedge_nodes)
                        .arg(&challenge.d_hyperedge_offsets)
                        .arg(&*part)
                        .arg(&mut *flags)
                        .arg(&dh)
                        .launch(hlist_cfg.clone())?;
                }
                if !(gc && c[INCREMENTAL_FLAGS] != 0) {
                    if let Some(ids) = large_ids {
                        stream
                            .launch_builder(&flag_kernel)
                            .arg(&nei)
                            .arg(&ni)
                            .arg(&challenge.d_hyperedge_nodes)
                            .arg(&challenge.d_hyperedge_offsets)
                            .arg(&*part)
                            .arg(&mut *flags)
                            .arg(ids)
                            .arg(&large_count)
                            .arg(&dh)
                            .launch(flags_cfg.clone())?;
                    } else {
                        stream
                            .launch_builder(&flag_kernel)
                            .arg(&nei)
                            .arg(&ni)
                            .arg(&challenge.d_hyperedge_nodes)
                            .arg(&challenge.d_hyperedge_offsets)
                            .arg(&*part)
                            .arg(&mut *flags)
                            .arg(&dh)
                            .launch(flags_cfg.clone())?;
                    }
                }
                stream
                    .launch_builder(&moves)
                    .arg(&ni)
                    .arg(&npi)
                    .arg(&capacity)
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&challenge.d_node_offsets)
                    .arg(&*part)
                    .arg(&*sizes)
                    .arg(&*flags)
                    .arg(&mut *priorities)
                    .arg(order)
                    .arg(&hubs)
                    .arg(&lanes4)
                    .arg(&lanes2)
                    .arg(&dh)
                    .launch(moves_cfg.clone())?;
                stream
                    .launch_builder(&stat)
                    .arg(&*priorities)
                    .arg(&*tabu)
                    .arg(&ni)
                    .arg(&0i32)
                    .arg(&-3i32)
                    .arg(&mut *block_max)
                    .arg(&mut *stats)
                    .arg(&dh)
                    .launch(filt_cfg.clone())?;
                stream
                    .launch_builder(&pick)
                    .arg(&*priorities)
                    .arg(&*tabu)
                    .arg(&ni)
                    .arg(&0i32)
                    .arg(&-3i32)
                    .arg(&*block_max)
                    .arg(&*stats)
                    .arg(&0u64)
                    .arg(&LCG_INC)
                    .arg(&0u64)
                    .arg(&0u64)
                    .arg(&0u64)
                    .arg(&0u64)
                    .arg(&0u64)
                    .arg(&mut *filt_out)
                    .arg(&mut *block_sel)
                    .arg(&dh)
                    .launch(filt_cfg.clone())?;
                stream
                    .launch_builder(&sort)
                    .arg(&*filt_out)
                    .arg(&*block_sel)
                    .arg(&gi)
                    .arg(&chunk)
                    .arg(&mut *sort_buf)
                    .arg(&mut *sorted)
                    .arg(&*sizes)
                    .arg(&npi)
                    .arg(&capacity)
                    .arg(&0i32)
                    .arg(&0i32)
                    .arg(&extra)
                    .arg(&mut *picks)
                    .arg(&dh)
                    .launch(sort_cfg.clone())?;
                stream
                    .launch_builder(&finish)
                    .arg(&mut dh)
                    .arg(&dc)
                    .arg(&*picks)
                    .arg(&*sorted)
                    .arg(&mut *part)
                    .arg(&mut *sizes)
                    .arg(&mut *tabu)
                    .arg(&challenge.d_node_offsets)
                    .arg(&mut pairs)
                    .launch(finish_cfg.clone())?;
            }
        }
        stream.memcpy_dtoh(&dh, &mut h)?;
    }
    stream.memcpy_dtoh(&*part, host_part)?;
    stream.memcpy_dtoh(&*sizes, host_sizes)?;
    stream.memcpy_dtoh(&*tabu, host_tabu)?;
    Ok(())
}
