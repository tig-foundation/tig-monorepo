use super::params::{Knobs, Track};
use super::{FILTER_THREADS, FINISH_THREADS, LCG_INC, SELECT_THREADS};
use cudarc::driver::{
    result, safe::LaunchConfig, sys, CudaModule, CudaSlice, CudaStream, DevicePtr, DriverError,
    LaunchArgs, PinnedHostSlice, PushKernelArg,
};
use std::sync::Arc;
use tig_challenges::hypergraph::Challenge;

pub(super) trait GraphArgs<'a> {
    fn graph_input<T>(
        &mut self,
        buffer: &'a CudaSlice<T>,
        address: &'a sys::CUdeviceptr,
        capture: bool,
    ) -> &mut Self;

    fn graph_output<T>(
        &mut self,
        buffer: &'a mut CudaSlice<T>,
        address: &'a sys::CUdeviceptr,
        capture: bool,
    ) -> &mut Self;
}

impl<'a> GraphArgs<'a> for LaunchArgs<'a> {
    fn graph_input<T>(
        &mut self,
        buffer: &'a CudaSlice<T>,
        address: &'a sys::CUdeviceptr,
        capture: bool,
    ) -> &mut Self {
        if capture {
            self.arg(address)
        } else {
            self.arg(buffer)
        }
    }

    fn graph_output<T>(
        &mut self,
        buffer: &'a mut CudaSlice<T>,
        address: &'a sys::CUdeviceptr,
        capture: bool,
    ) -> &mut Self {
        if capture {
            self.arg(address)
        } else {
            self.arg(buffer)
        }
    }
}

pub(super) struct BatchGraph {
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
    stream: Arc<CudaStream>,
}

impl BatchGraph {
    pub(super) fn capture(
        stream: &Arc<CudaStream>,
        submit: impl FnOnce() -> anyhow::Result<()>,
    ) -> Option<Self> {
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .ok()?;
        let submitted = submit();
        let raw = unsafe { result::stream::end_capture(stream.cu_stream()) }.ok()?;
        if raw.is_null() {
            return None;
        }
        let mut graph = Self {
            graph: raw,
            exec: std::ptr::null_mut(),
            stream: stream.clone(),
        };
        submitted.ok()?;
        unsafe { sys::cuGraphInstantiateWithFlags(&mut graph.exec, graph.graph, 0) }
            .result()
            .ok()?;
        Some(graph)
    }

    pub(super) fn launch(&self) -> Result<(), DriverError> {
        self.stream.context().bind_to_thread()?;
        unsafe { result::graph::launch(self.exec, self.stream.cu_stream()) }
    }
}

impl Drop for BatchGraph {
    fn drop(&mut self) {
        let ctx = self.stream.context();
        ctx.record_err(ctx.bind_to_thread());
        if !self.exec.is_null() {
            ctx.record_err(unsafe { result::graph::exec_destroy(self.exec) });
        }
        ctx.record_err(unsafe { result::graph::destroy(self.graph) });
    }
}

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
    sort_mb: u32,
    flags_cfg: LaunchConfig,
    hlist_cfg: LaunchConfig,
    mut moves_cfg: LaunchConfig,
    scalar_blocks: u32,
    move_seq: bool,
    move_ilp: i64,
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
    h4: &mut CudaSlice<i32>,
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
        u64::from(track.counts || track.k_edge_flags_incr.is_some()),
    ];
    let mut h: Header = [0; 32];
    h[FULL_FLAGS] = 1;
    h[GC] = u64::from(gc);
    schedule(&mut h, &c);
    let dc = stream.memcpy_stod(&c)?;
    let mut dh = stream.memcpy_stod(&h)?;
    let cached_flags = track.counts;
    if track.mvc {
        moves_cfg.grid_dim.0 = moves_cfg.grid_dim.0 - scalar_blocks + (scalar_blocks + 3) / 4;
    }
    let mut pairs = stream.alloc_zeros::<i32>(if cached_flags { 3 * n } else { 2 * n })?;
    let mut counts = stream.alloc_zeros::<u32>(if cached_flags { 32 * ne } else { 2 })?;
    let mut list = stream.alloc_zeros::<i32>(if cached_flags {
        1
    } else {
        (offsets[n] as usize).max(1)
    })?;
    let mut stamp = stream.alloc_zeros::<i32>(if cached_flags { 1 } else { ne })?;
    let cached_count = if cached_flags {
        Some(module.load_function("update_edge_pin_counts_100k")?)
    } else {
        None
    };
    let cached_refresh = if cached_flags {
        Some(module.load_function("update_edge_flags_from_counts_100k")?)
    } else {
        None
    };
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
    let moves = if move_seq {
        moves
    } else {
        module.load_function(&format!("{}_ilp{}", track.k_moves, move_ilp))?
    };
    let stat = module.load_function(if track.stats64 {
        "mica_sx_stats"
    } else {
        "mica_filt_stats"
    })?;
    let pick = module.load_function(if sort_mb != 0 && track.pick64 {
        "mica_sx_pick4_mb"
    } else if sort_mb != 0 {
        "mica_filt_pick4_mb"
    } else if track.pick64 {
        "mica_sx_pick4"
    } else {
        "mica_filt_pick4"
    })?;
    let sort = module.load_function("mica_sortsel")?;
    let tiles = ((n + 2047) / 2048) as i32;
    let mut mb = if sort_mb != 0 {
        Some((
            module.load_function("mica_sortmb_pass")?,
            module.load_function("mica_sortmb_select")?,
            stream.alloc_zeros::<i32>(filt_grid as usize * 1024)?,
            stream.alloc_zeros::<i32>(tiles as usize * 3 * 256)?,
            stream.alloc_zeros::<i32>(tiles as usize * 64)?,
        ))
    } else {
        None
    };
    let finish = module.load_function(if cached_flags {
        "grouped_finish_counts_100k"
    } else {
        "grouped_finish"
    })?;
    let resume = module.load_function("grouped_resume")?;
    let mut counts_cfg = hlist_cfg.clone();
    counts_cfg.grid_dim.0 = counts_cfg.grid_dim.0.min(32);
    let stats_cfg = LaunchConfig {
        grid_dim: (filt_grid, 1, 1),
        block_dim: (if track.stats64 { 64 } else { FILTER_THREADS }, 1, 1),
        shared_mem_bytes: 0,
    };
    let pick_cfg = LaunchConfig {
        grid_dim: (filt_grid, 1, 1),
        block_dim: (if track.pick64 { 64 } else { FILTER_THREADS }, 1, 1),
        shared_mem_bytes: 0,
    };
    let sort_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (SELECT_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mb_cfg = LaunchConfig {
        grid_dim: (sort_mb, 1, 1),
        block_dim: (256, 1, 1),
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
    let addresses = [
        dh.device_ptr(stream).0,
        pairs.device_ptr(stream).0,
        challenge.d_node_hyperedges.device_ptr(stream).0,
        challenge.d_node_offsets.device_ptr(stream).0,
        challenge.d_hyperedge_nodes.device_ptr(stream).0,
        challenge.d_hyperedge_offsets.device_ptr(stream).0,
        part.device_ptr(stream).0,
        counts.device_ptr(stream).0,
        flags.device_ptr(stream).0,
        list.device_ptr(stream).0,
        stamp.device_ptr(stream).0,
        large_ids.map(|ids| ids.device_ptr(stream).0).unwrap_or(0),
        sizes.device_ptr(stream).0,
        priorities.device_ptr(stream).0,
        order.device_ptr(stream).0,
        tabu.device_ptr(stream).0,
        block_max.device_ptr(stream).0,
        stats.device_ptr(stream).0,
        filt_out.device_ptr(stream).0,
        block_sel.device_ptr(stream).0,
        h4.device_ptr(stream).0,
        sort_buf.device_ptr(stream).0,
        sorted.device_ptr(stream).0,
        picks.device_ptr(stream).0,
        dc.device_ptr(stream).0,
        mb.as_ref().map(|b| b.2.device_ptr(stream).0).unwrap_or(0),
        mb.as_ref().map(|b| b.3.device_ptr(stream).0).unwrap_or(0),
        mb.as_ref().map(|b| b.4.device_ptr(stream).0).unwrap_or(0),
    ];
    stream.context().check_err()?;
    let mut pin: [PinnedHostSlice<u64>; 2] = unsafe {
        [
            stream.context().alloc_pinned(h.len())?,
            stream.context().alloc_pinned(h.len())?,
        ]
    };
    let ev = [
        stream
            .context()
            .new_event(Some(sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))?,
        stream
            .context()
            .new_event(Some(sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))?,
    ];
    let mut slot = 0;
    let mut primed = false;
    let mut graph = None;
    let mut capture_attempted = false;
    while h[STATE] != DONE {
        if h[STATE] == SUSPENDED {
            unsafe {
                stream
                    .launch_builder(&resume)
                    .arg(&mut dh)
                    .launch(one.clone())?;
            }
        }
        {
            let mut submit = |capture| -> anyhow::Result<()> {
                for _ in 0..batch {
                    unsafe {
                        if let (Some(count), Some(refresh)) =
                            (cached_count.as_ref(), cached_refresh.as_ref())
                        {
                            stream
                                .launch_builder(count)
                                .graph_input(&dh, &addresses[0], capture)
                                .arg(&nei)
                                .arg(&ni)
                                .graph_input(&pairs, &addresses[1], capture)
                                .graph_input(&challenge.d_node_hyperedges, &addresses[2], capture)
                                .graph_input(&challenge.d_node_offsets, &addresses[3], capture)
                                .graph_input(&challenge.d_hyperedge_nodes, &addresses[4], capture)
                                .graph_input(&challenge.d_hyperedge_offsets, &addresses[5], capture)
                                .graph_input(&*part, &addresses[6], capture)
                                .graph_output(&mut counts, &addresses[7], capture)
                                .graph_output(&mut *flags, &addresses[8], capture)
                                .launch(counts_cfg.clone())?;
                            stream
                                .launch_builder(refresh)
                                .graph_input(&dh, &addresses[0], capture)
                                .arg(&nei)
                                .arg(&ni)
                                .graph_input(&pairs, &addresses[1], capture)
                                .graph_input(&challenge.d_node_hyperedges, &addresses[2], capture)
                                .graph_input(&challenge.d_node_offsets, &addresses[3], capture)
                                .graph_input(&*part, &addresses[6], capture)
                                .graph_output(&mut counts, &addresses[7], capture)
                                .graph_output(&mut *flags, &addresses[8], capture)
                                .launch(counts_cfg.clone())?;
                        } else if let (Some(expand), Some(incr)) = (expand.as_ref(), incr.as_ref())
                        {
                            stream
                                .launch_builder(expand)
                                .arg(&0i32)
                                .graph_input(&pairs, &addresses[1], capture)
                                .graph_input(&challenge.d_node_hyperedges, &addresses[2], capture)
                                .graph_input(&challenge.d_node_offsets, &addresses[3], capture)
                                .graph_output(&mut list, &addresses[9], capture)
                                .graph_input(&dh, &addresses[0], capture)
                                .launch(hlist_cfg.clone())?;
                            stream
                                .launch_builder(incr)
                                .arg(&0i32)
                                .graph_input(&list, &addresses[9], capture)
                                .arg(&0i32)
                                .graph_output(&mut stamp, &addresses[10], capture)
                                .arg(&nei)
                                .arg(&ni)
                                .graph_input(&challenge.d_hyperedge_nodes, &addresses[4], capture)
                                .graph_input(&challenge.d_hyperedge_offsets, &addresses[5], capture)
                                .graph_input(&*part, &addresses[6], capture)
                                .graph_output(&mut *flags, &addresses[8], capture)
                                .graph_input(&dh, &addresses[0], capture)
                                .launch(hlist_cfg.clone())?;
                        }
                        if !cached_flags && !(gc && c[INCREMENTAL_FLAGS] != 0) {
                            if let Some(ids) = large_ids {
                                stream
                                    .launch_builder(&flag_kernel)
                                    .arg(&nei)
                                    .arg(&ni)
                                    .graph_input(
                                        &challenge.d_hyperedge_nodes,
                                        &addresses[4],
                                        capture,
                                    )
                                    .graph_input(
                                        &challenge.d_hyperedge_offsets,
                                        &addresses[5],
                                        capture,
                                    )
                                    .graph_input(&*part, &addresses[6], capture)
                                    .graph_output(&mut *flags, &addresses[8], capture)
                                    .graph_input(ids, &addresses[11], capture)
                                    .arg(&large_count)
                                    .graph_input(&dh, &addresses[0], capture)
                                    .launch(flags_cfg.clone())?;
                            } else {
                                stream
                                    .launch_builder(&flag_kernel)
                                    .arg(&nei)
                                    .arg(&ni)
                                    .graph_input(
                                        &challenge.d_hyperedge_nodes,
                                        &addresses[4],
                                        capture,
                                    )
                                    .graph_input(
                                        &challenge.d_hyperedge_offsets,
                                        &addresses[5],
                                        capture,
                                    )
                                    .graph_input(&*part, &addresses[6], capture)
                                    .graph_output(&mut *flags, &addresses[8], capture)
                                    .graph_input(&dh, &addresses[0], capture)
                                    .launch(flags_cfg.clone())?;
                            }
                        }
                        stream
                            .launch_builder(&moves)
                            .arg(&ni)
                            .arg(&npi)
                            .arg(&capacity)
                            .graph_input(&challenge.d_node_hyperedges, &addresses[2], capture)
                            .graph_input(&challenge.d_node_offsets, &addresses[3], capture)
                            .graph_input(&*part, &addresses[6], capture)
                            .graph_input(&*sizes, &addresses[12], capture)
                            .graph_input(&*flags, &addresses[8], capture)
                            .graph_output(&mut *priorities, &addresses[13], capture)
                            .graph_input(order, &addresses[14], capture)
                            .arg(&hubs)
                            .arg(&lanes4)
                            .arg(&lanes2)
                            .graph_input(&dh, &addresses[0], capture)
                            .launch(moves_cfg.clone())?;
                        stream
                            .launch_builder(&stat)
                            .graph_input(&*priorities, &addresses[13], capture)
                            .graph_input(&*tabu, &addresses[15], capture)
                            .arg(&ni)
                            .arg(&0i32)
                            .arg(&-3i32)
                            .graph_output(&mut *block_max, &addresses[16], capture)
                            .graph_output(&mut *stats, &addresses[17], capture)
                            .graph_input(&dh, &addresses[0], capture)
                            .launch(stats_cfg.clone())?;
                        if let Some((pass, select, seg_hist, tile_hist, tgt_hist)) = mb.as_mut() {
                            stream
                                .launch_builder(&pick)
                                .graph_input(&*priorities, &addresses[13], capture)
                                .graph_input(&*tabu, &addresses[15], capture)
                                .arg(&ni)
                                .arg(&0i32)
                                .arg(&-3i32)
                                .graph_input(&*block_max, &addresses[16], capture)
                                .graph_input(&*stats, &addresses[17], capture)
                                .arg(&0u64)
                                .arg(&LCG_INC)
                                .arg(&0u64)
                                .arg(&0u64)
                                .arg(&0u64)
                                .arg(&0u64)
                                .arg(&0u64)
                                .graph_output(&mut *filt_out, &addresses[18], capture)
                                .graph_output(&mut *block_sel, &addresses[19], capture)
                                .graph_output(&mut *h4, &addresses[20], capture)
                                .graph_input(&dh, &addresses[0], capture)
                                .graph_output(&mut *seg_hist, &addresses[25], capture)
                                .graph_output(&mut *tile_hist, &addresses[26], capture)
                                .graph_output(&mut *tgt_hist, &addresses[27], capture)
                                .arg(&tiles)
                                .launch(pick_cfg.clone())?;
                            for p in 0..4i32 {
                                stream
                                    .launch_builder(pass)
                                    .graph_input(&*filt_out, &addresses[18], capture)
                                    .graph_input(&*block_sel, &addresses[19], capture)
                                    .arg(&gi)
                                    .arg(&chunk)
                                    .arg(&tiles)
                                    .graph_output(&mut *sort_buf, &addresses[21], capture)
                                    .graph_output(&mut *sorted, &addresses[22], capture)
                                    .graph_input(&*h4, &addresses[20], capture)
                                    .graph_input(&*seg_hist, &addresses[25], capture)
                                    .graph_output(&mut *tile_hist, &addresses[26], capture)
                                    .graph_output(&mut *tgt_hist, &addresses[27], capture)
                                    .arg(&p)
                                    .arg(&0i32)
                                    .arg(&0i32)
                                    .graph_input(&dh, &addresses[0], capture)
                                    .launch(mb_cfg.clone())?;
                            }
                            stream
                                .launch_builder(select)
                                .graph_input(&*block_sel, &addresses[19], capture)
                                .arg(&gi)
                                .graph_input(&*sorted, &addresses[22], capture)
                                .graph_input(&*sizes, &addresses[12], capture)
                                .arg(&npi)
                                .arg(&capacity)
                                .arg(&0i32)
                                .arg(&0i32)
                                .arg(&extra)
                                .graph_output(&mut *picks, &addresses[23], capture)
                                .graph_output(&mut *h4, &addresses[20], capture)
                                .graph_input(&*tgt_hist, &addresses[27], capture)
                                .graph_input(&dh, &addresses[0], capture)
                                .launch(mb_cfg.clone())?;
                        } else {
                            stream
                                .launch_builder(&pick)
                                .graph_input(&*priorities, &addresses[13], capture)
                                .graph_input(&*tabu, &addresses[15], capture)
                                .arg(&ni)
                                .arg(&0i32)
                                .arg(&-3i32)
                                .graph_input(&*block_max, &addresses[16], capture)
                                .graph_input(&*stats, &addresses[17], capture)
                                .arg(&0u64)
                                .arg(&LCG_INC)
                                .arg(&0u64)
                                .arg(&0u64)
                                .arg(&0u64)
                                .arg(&0u64)
                                .arg(&0u64)
                                .graph_output(&mut *filt_out, &addresses[18], capture)
                                .graph_output(&mut *block_sel, &addresses[19], capture)
                                .graph_output(&mut *h4, &addresses[20], capture)
                                .graph_input(&dh, &addresses[0], capture)
                                .launch(pick_cfg.clone())?;
                            stream
                                .launch_builder(&sort)
                                .graph_input(&*filt_out, &addresses[18], capture)
                                .graph_input(&*block_sel, &addresses[19], capture)
                                .arg(&gi)
                                .arg(&chunk)
                                .graph_output(&mut *sort_buf, &addresses[21], capture)
                                .graph_output(&mut *sorted, &addresses[22], capture)
                                .graph_input(&*sizes, &addresses[12], capture)
                                .arg(&npi)
                                .arg(&capacity)
                                .arg(&0i32)
                                .arg(&0i32)
                                .arg(&extra)
                                .graph_output(&mut *picks, &addresses[23], capture)
                                .graph_output(&mut *h4, &addresses[20], capture)
                                .graph_input(&dh, &addresses[0], capture)
                                .launch(sort_cfg.clone())?;
                        }
                        stream
                            .launch_builder(&finish)
                            .graph_output(&mut dh, &addresses[0], capture)
                            .graph_input(&dc, &addresses[24], capture)
                            .graph_input(&*picks, &addresses[23], capture)
                            .graph_input(&*sorted, &addresses[22], capture)
                            .graph_output(&mut *part, &addresses[6], capture)
                            .graph_output(&mut *sizes, &addresses[12], capture)
                            .graph_output(&mut *tabu, &addresses[15], capture)
                            .graph_input(&challenge.d_node_offsets, &addresses[3], capture)
                            .graph_output(&mut pairs, &addresses[1], capture)
                            .launch(finish_cfg.clone())?;
                    }
                }
                Ok(())
            };
            if !capture_attempted {
                capture_attempted = true;
                graph = BatchGraph::capture(stream, || submit(true));
            }
            if let Some(graph) = graph.as_ref() {
                graph.launch()?;
            } else {
                submit(false)?;
            }
        }
        stream.memcpy_dtoh(&dh, &mut pin[slot])?;
        ev[slot].record(stream)?;
        if primed {
            let previous = slot ^ 1;
            ev[previous].synchronize()?;
            let header = pin[previous].as_slice()?;
            debug_assert!(header[ROUND] >= h[ROUND]);
            h.copy_from_slice(header);
        }
        primed = true;
        slot ^= 1;
    }
    stream.memcpy_dtoh(&*part, host_part)?;
    stream.memcpy_dtoh(&*sizes, host_sizes)?;
    stream.memcpy_dtoh(&*tabu, host_tabu)?;
    Ok(())
}
