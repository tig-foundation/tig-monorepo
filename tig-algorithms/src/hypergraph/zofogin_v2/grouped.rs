use super::params::{Knobs, Track};
use super::{FILTER_THREADS, FINISH_THREADS, LCG_INC, SELECT_THREADS};
use cudarc::driver::{safe::LaunchConfig, CudaModule, CudaSlice, CudaStream, DevicePtr, PushKernelArg};
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
    start: u64,
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
    check_every: u64,
    check: &mut dyn FnMut(u64, &[i32]) -> anyhow::Result<bool>,
) -> anyhow::Result<bool> {
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
    h[ROUND] = start;
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
    let (addr_pairs, _hold_pairs) = pairs.device_ptr(stream);
    let (addr_node_edges, _hold_node_edges) = challenge.d_node_hyperedges.device_ptr(stream);
    let (addr_node_offsets, _hold_node_offsets) = challenge.d_node_offsets.device_ptr(stream);
    let (addr_list, _hold_list) = list.device_ptr(stream);
    let (addr_header, _hold_header) = dh.device_ptr(stream);
    let (addr_stamp, _hold_stamp) = stamp.device_ptr(stream);
    let (addr_edge_nodes, _hold_edge_nodes) = challenge.d_hyperedge_nodes.device_ptr(stream);
    let (addr_edge_offsets, _hold_edge_offsets) = challenge.d_hyperedge_offsets.device_ptr(stream);
    let (addr_part, _hold_part) = part.device_ptr(stream);
    let (addr_flags, _hold_flags) = flags.device_ptr(stream);
    let (addr_sizes, _hold_sizes) = sizes.device_ptr(stream);
    let (addr_priorities, _hold_priorities) = priorities.device_ptr(stream);
    let (addr_order, _hold_order) = order.device_ptr(stream);
    let (addr_tabu, _hold_tabu) = tabu.device_ptr(stream);
    let (addr_block_max, _hold_block_max) = block_max.device_ptr(stream);
    let (addr_stats, _hold_stats) = stats.device_ptr(stream);
    let (addr_filt_out, _hold_filt_out) = filt_out.device_ptr(stream);
    let (addr_block_sel, _hold_block_sel) = block_sel.device_ptr(stream);
    let (addr_sort_buf, _hold_sort_buf) = sort_buf.device_ptr(stream);
    let (addr_sorted, _hold_sorted) = sorted.device_ptr(stream);
    let (addr_picks, _hold_picks) = picks.device_ptr(stream);
    let (addr_config, _hold_config) = dc.device_ptr(stream);
    let large_addr = large_ids.map(|ids| ids.device_ptr(stream));
    stream.synchronize()?;
    stream.begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let captured = (|| -> anyhow::Result<()> {
        for _ in 0..batch {
            unsafe {
                if let (Some(expand), Some(incr)) = (expand.as_ref(), incr.as_ref()) {
                    stream
                        .launch_builder(expand)
                        .arg(&0i32)
                        .arg(&addr_pairs)
                        .arg(&addr_node_edges)
                        .arg(&addr_node_offsets)
                        .arg(&addr_list)
                        .arg(&addr_header)
                        .launch(hlist_cfg.clone())?;
                    stream
                        .launch_builder(incr)
                        .arg(&0i32)
                        .arg(&addr_list)
                        .arg(&0i32)
                        .arg(&addr_stamp)
                        .arg(&nei)
                        .arg(&ni)
                        .arg(&addr_edge_nodes)
                        .arg(&addr_edge_offsets)
                        .arg(&addr_part)
                        .arg(&addr_flags)
                        .arg(&addr_header)
                        .launch(hlist_cfg.clone())?;
                }
                if !(gc && c[INCREMENTAL_FLAGS] != 0) {
                    if let Some((ids, _)) = large_addr.as_ref() {
                        stream
                            .launch_builder(&flag_kernel)
                            .arg(&nei)
                            .arg(&ni)
                            .arg(&addr_edge_nodes)
                            .arg(&addr_edge_offsets)
                            .arg(&addr_part)
                            .arg(&addr_flags)
                            .arg(ids)
                            .arg(&large_count)
                            .arg(&addr_header)
                            .launch(flags_cfg.clone())?;
                    } else {
                        stream
                            .launch_builder(&flag_kernel)
                            .arg(&nei)
                            .arg(&ni)
                            .arg(&addr_edge_nodes)
                            .arg(&addr_edge_offsets)
                            .arg(&addr_part)
                            .arg(&addr_flags)
                            .arg(&addr_header)
                            .launch(flags_cfg.clone())?;
                    }
                }
                stream
                    .launch_builder(&moves)
                    .arg(&ni)
                    .arg(&npi)
                    .arg(&capacity)
                    .arg(&addr_node_edges)
                    .arg(&addr_node_offsets)
                    .arg(&addr_part)
                    .arg(&addr_sizes)
                    .arg(&addr_flags)
                    .arg(&addr_priorities)
                    .arg(&addr_order)
                    .arg(&hubs)
                    .arg(&lanes4)
                    .arg(&lanes2)
                    .arg(&addr_header)
                    .launch(moves_cfg.clone())?;
                stream
                    .launch_builder(&stat)
                    .arg(&addr_priorities)
                    .arg(&addr_tabu)
                    .arg(&ni)
                    .arg(&0i32)
                    .arg(&-3i32)
                    .arg(&addr_block_max)
                    .arg(&addr_stats)
                    .arg(&addr_header)
                    .launch(filt_cfg.clone())?;
                stream
                    .launch_builder(&pick)
                    .arg(&addr_priorities)
                    .arg(&addr_tabu)
                    .arg(&ni)
                    .arg(&0i32)
                    .arg(&-3i32)
                    .arg(&addr_block_max)
                    .arg(&addr_stats)
                    .arg(&0u64)
                    .arg(&LCG_INC)
                    .arg(&0u64)
                    .arg(&0u64)
                    .arg(&0u64)
                    .arg(&0u64)
                    .arg(&0u64)
                    .arg(&addr_filt_out)
                    .arg(&addr_block_sel)
                    .arg(&addr_header)
                    .launch(filt_cfg.clone())?;
                stream
                    .launch_builder(&sort)
                    .arg(&addr_filt_out)
                    .arg(&addr_block_sel)
                    .arg(&gi)
                    .arg(&chunk)
                    .arg(&addr_sort_buf)
                    .arg(&addr_sorted)
                    .arg(&addr_sizes)
                    .arg(&npi)
                    .arg(&capacity)
                    .arg(&0i32)
                    .arg(&0i32)
                    .arg(&extra)
                    .arg(&addr_picks)
                    .arg(&addr_header)
                    .launch(sort_cfg.clone())?;
                stream
                    .launch_builder(&finish)
                    .arg(&addr_header)
                    .arg(&addr_config)
                    .arg(&addr_picks)
                    .arg(&addr_sorted)
                    .arg(&addr_part)
                    .arg(&addr_sizes)
                    .arg(&addr_tabu)
                    .arg(&addr_node_offsets)
                    .arg(&addr_pairs)
                    .launch(finish_cfg.clone())?;
            }
        }
        Ok(())
    })();
    let graph_result = stream.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
    captured?;
    let graph = graph_result?.ok_or_else(|| anyhow::anyhow!("empty refinement graph"))?;
    let _completion = cudarc::driver::safe::SyncOnDrop::sync_stream(stream);
    let mut next_check = check_every;
    let mut pending: Option<(u64, Vec<i32>)> = None;
    while h[STATE] != DONE {
        if h[STATE] == SUSPENDED {
            unsafe {
                stream
                    .launch_builder(&resume)
                    .arg(&addr_header)
                    .launch(one.clone())?;
            }
        }
        graph.launch()?;
        if let Some((round, v)) = pending.take() {
            if check(round, &v)? {
                return Ok(true);
            }
        }
        stream.memcpy_dtoh(&dh, &mut h)?;
        if check_every > 0 && h[ROUND] >= next_check {
            next_check = (h[ROUND] / check_every + 1) * check_every;
            pending = Some((h[ROUND], stream.memcpy_dtov(&*part)?));
        }
    }
    stream.memcpy_dtoh(&*part, host_part)?;
    stream.memcpy_dtoh(&*sizes, host_sizes)?;
    stream.memcpy_dtoh(&*tabu, host_tabu)?;
    Ok(false)
}
