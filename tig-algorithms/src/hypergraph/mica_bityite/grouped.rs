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
const ROUNDS: usize = 4;
const COOL_ROUNDS: usize = 5;
const COOL_POW: usize = 6;
const COOL_PCT: usize = 7;
const PERT_ROUNDS: usize = 8;
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
    knobs: &mut Knobs,
    hp: &super::Hp<'_>,
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
    node_hedges: &[i32],
    tri100: &mut Option<super::track::Tri100>,
    saver: &mut super::Saver<'_>,
) -> anyhow::Result<bool> {
    let n = challenge.num_nodes as usize;
    let np = challenge.num_parts as usize;
    let ne = challenge.num_hyperedges as usize;
    let capacity = challenge.max_part_size as i32;
    let mut c: Config = [
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
    let mut dc = stream.memcpy_stod(&c)?;
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
    let barriers_per_round = if sort_mb == 0 { 7u64 } else { 11u64 };
    let persistent_name = match track.k_moves {
        "compute_refinement_moves_warp8_20k"
        | "compute_refinement_moves_warp8_50k"
        | "compute_refinement_moves_warp8_200k" => Some("persistent_rounds_zgm0"),
        "compute_refinement_moves_warp8_100k" => Some("persistent_rounds_zgm1"),
        _ => None,
    };
    let persistent = if knobs.pz
        && cached_flags
        && track.stats64
        && !track.pick64
        && np <= 64
        && counts_cfg.block_dim == (128, 1, 1)
        && moves_cfg.block_dim == (64, 1, 1)
    {
        persistent_name
            .and_then(|name| module.load_function(name).ok())
            .and_then(|kernel| {
                let ctx = stream.context();
                let sms = ctx
                    .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
                    .ok()?;
                let threads = ctx
                    .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
                    .ok()?;
                let major = ctx
                    .attribute(
                        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    )
                    .ok()?;
                let generations = 1 + barriers_per_round * u64::from(knobs.pz_rounds);
                if sms <= 0
                    || threads < 256
                    || major < 7
                    || generations * sms as u64 > u32::MAX as u64
                    || kernel
                        .occupancy_max_active_blocks_per_multiprocessor(256, 0, None)
                        .ok()?
                        < 1
                {
                    return None;
                }
                Some((
                    kernel,
                    LaunchConfig {
                        grid_dim: (sms as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                ))
            })
    } else {
        None
    };
    let mut barrier = if persistent.is_some() {
        Some(stream.alloc_zeros::<u32>(1)?)
    } else {
        None
    };
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
        if let (Some((kernel, cfg)), Some(counter)) = (persistent.as_ref(), barrier.as_mut()) {
            debug_assert!(
                (1 + barriers_per_round * u64::from(knobs.pz_rounds)) * u64::from(cfg.grid_dim.0)
                    <= u32::MAX as u64
            );
            stream.memset_zeros(counter)?;
            unsafe {
                stream
                    .launch_builder(kernel)
                    .arg(&mut dh)
                    .arg(&dc)
                    .arg(counter)
                    .arg(&knobs.pz_rounds)
                    .arg(&(counts_cfg.grid_dim.0 as i32))
                    .arg(&(moves_cfg.grid_dim.0 as i32))
                    .arg(&gi)
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&challenge.d_node_offsets)
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&mut *part)
                    .arg(&mut *sizes)
                    .arg(&mut *tabu)
                    .arg(&mut counts)
                    .arg(&mut *flags)
                    .arg(&mut *priorities)
                    .arg(order)
                    .arg(&hubs)
                    .arg(&lanes4)
                    .arg(&lanes2)
                    .arg(&mut pairs)
                    .arg(&mut *block_max)
                    .arg(&mut *stats)
                    .arg(&mut *block_sel)
                    .arg(&mut *h4)
                    .arg(&mut *filt_out)
                    .arg(&mut *sort_buf)
                    .arg(&mut *sorted)
                    .arg(&mut *picks)
                    .arg(&sort_mb)
                    .arg(&tiles)
                    .arg(&addresses[25])
                    .arg(&addresses[26])
                    .arg(&addresses[27])
                    .launch(cfg.clone())?;
            }
        } else {
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
        if let Some(tri) = tri100.as_mut() {
            tri.launched();
            if tri.finish_pending(
                false,
                &*part,
                challenge,
                stream,
                offsets,
                node_hedges,
                saver,
            )? {
                return Ok(true);
            }
            let launch_rounds = if persistent.is_some() {
                u64::from(knobs.pz_rounds)
            } else {
                batch as u64
            };
            if tri.due(h[ROUND] + 2 * launch_rounds) {
                ev[slot].synchronize()?;
                let round = pin[slot].as_slice()?[ROUND] as usize;
                debug_assert!((h[ROUND]..=h[ROUND] + 2 * launch_rounds).contains(&(round as u64)));
                if tri.check(round, &*part, challenge, module, stream, saver)? {
                    return Ok(true);
                }
            }
            if tri.reduce {
                ev[slot].synchronize()?;
                let mut current: Header = [0; 32];
                current.copy_from_slice(pin[slot].as_slice()?);
                if let Some((cool, pert)) =
                    tri.reduce_budget(current[ROUND] as usize, hp, track, knobs)?
                {
                    c[ROUNDS] = knobs.refinement_rounds as u64;
                    c[COOL_ROUNDS] = cool as u64;
                    c[PERT_ROUNDS] = pert as u64;
                    schedule(&mut current, &c);
                    if current[ROUND] >= c[ROUNDS] {
                        current[STATE] = DONE;
                    }
                    stream.synchronize()?;
                    stream.memcpy_htod(&c, &mut dc)?;
                    stream.memcpy_htod(&current, &mut dh)?;
                    stream.synchronize()?;
                    h = current;
                    primed = false;
                    continue;
                }
            }
        }
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
    if let Some(tri) = tri100.as_mut() {
        tri.round = h[ROUND] as usize;
    }
    Ok(false)
}

fn gl_a0(part: &[i32], sizes: &[i32], n: usize, np: usize) -> Option<&'static str> {
    if np == 0 || np > 64 || part.len() != n || sizes.len() != np {
        return Some("shape");
    }
    if sizes.iter().any(|&s| s <= 0) {
        return Some("empty");
    }
    if sizes.iter().map(|&s| i64::from(s)).sum::<i64>() != n as i64 {
        return Some("size_sum");
    }
    let mut counts = [0i32; 64];
    for &p in part {
        if p < 0 || p as usize >= np {
            return Some("part_id");
        }
        counts[p as usize] += 1;
    }
    if &counts[..np] != sizes {
        return Some("size_match");
    }
    None
}

pub(super) fn gl_a1(
    challenge: &Challenge,
    module: &Arc<CudaModule>,
    prop: &cudarc::runtime::sys::cudaDeviceProp,
    part: &[i32],
    sizes: &[i32],
) -> Result<LaunchConfig, &'static str> {
    let n = challenge.num_nodes as usize;
    if !(256..524288).contains(&n) || challenge.max_part_size > (i32::MAX - 8) as u32 {
        return Err("bounds");
    }
    debug_assert_eq!(gl_a0(&[0, 0, 0, 1], &[3, 1], 4, 2), None);
    debug_assert_eq!(gl_a0(&[0, 0, 0, 1], &[2, 2], 4, 2), Some("size_match"));
    if let Some(why) = gl_a0(part, sizes, n, challenge.num_parts as usize) {
        return Err(why);
    }
    if prop.maxThreadsPerBlock < 512
        || prop.major < 7
        || prop.multiProcessorCount <= 0
        || prop.sharedMemPerBlock < 40 * 1024
        || prop.sharedMemPerMultiprocessor < 80 * 1024
    {
        return Err("hardware");
    }
    let blocks = prop.multiProcessorCount as u32;
    if u64::from(blocks) * (11 * 128 + 2) > u64::from(u32::MAX) {
        return Err("barrier");
    }
    let kernel = module.load_function("gl_k1").map_err(|_| "kernel")?;
    if kernel
        .occupancy_max_active_blocks_per_multiprocessor(512, 0, None)
        .map_err(|_| "residency")?
        < 1
    {
        return Err("residency");
    }
    Ok(LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (512, 1, 1),
        shared_mem_bytes: 0,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gl_a2(
    challenge: &Challenge,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    cfg: LaunchConfig,
    start: usize,
    pass: u64,
    batch_size: usize,
    limit: usize,
    part: &mut CudaSlice<i32>,
    sizes: &mut CudaSlice<i32>,
    tabu: &mut CudaSlice<i32>,
    flags: &mut CudaSlice<u64>,
    priorities: &mut CudaSlice<i32>,
    host_part: &mut [i32],
    host_sizes: &mut [i32],
    host_tabu: &mut [i32],
    offsets: &[i32],
    edges: &[i32],
    hedge_sizes: &[i32],
    tri: &mut super::track::Gl1,
    saver: &mut super::Saver<'_>,
) -> anyhow::Result<(usize, i32, usize, u64, usize)> {
    let n = challenge.num_nodes as usize;
    let np = challenge.num_parts as usize;
    let ne = challenge.num_hyperedges as usize;
    let cap = challenge.max_part_size as i32;
    let (mut total, mut cool, mut pert) = (tri.total, tri.cool, tri.pert);
    anyhow::ensure!(
        offsets.len() == n + 1
            && hedge_sizes.len() == ne
            && offsets[0] == 0
            && offsets[n] as usize == edges.len()
            && offsets.windows(2).all(|p| p[0] >= 0 && p[0] <= p[1]),
        "invalid graph offsets",
    );
    let mut meta = Vec::with_capacity(ne);
    let mut words = 0usize;
    for &size in hedge_sizes {
        anyhow::ensure!(size >= 0, "invalid edge size");
        let format = if size <= 255 {
            0u32
        } else if size <= 65535 {
            1u32
        } else {
            2u32
        };
        anyhow::ensure!(words <= (u32::MAX >> 2) as usize, "count offset overflow");
        meta.push(((words as u32) << 2) | format);
        words += 16usize << format;
    }
    anyhow::ensure!(
        words <= i32::MAX as usize
            && limit <= i32::MAX as usize / 3
            && start <= total
            && total <= 500_000
            && cool <= 500_000
            && pert <= 500_000
            && (1..=2_048).contains(&batch_size)
            && n as u64 * batch_size.max(128) as u64 <= i32::MAX as u64
            && u64::from(cfg.grid_dim.0) * (11 * batch_size.max(128) as u64 + 2)
                <= u64::from(u32::MAX),
        "invalid loop bounds",
    );
    let items: Vec<(i32, i32, i32)> = (0..n)
        .map(|v| (v as i32, offsets[v], offsets[v + 1] - offsets[v]))
        .collect();
    let (tasks, task_count) = gl_a4(&items, 4 * cfg.grid_dim.0 as usize);
    debug_assert_eq!(tasks.len(), task_count as usize * 128);
    let d_tasks = stream.memcpy_stod(&tasks)?;
    let d_meta = stream.memcpy_stod(&meta)?;
    let mut d_counts = stream.alloc_zeros::<i32>(words.max(1))?;
    let mut rng = Vec::with_capacity(2 * (n + 1));
    let (mut a, mut c) = (1u64, 0u64);
    for _ in 0..=n {
        rng.push(a);
        rng.push(c);
        a = a.wrapping_mul(super::LCG_MUL);
        c = c.wrapping_mul(super::LCG_MUL).wrapping_add(LCG_INC);
    }
    let d_rng = stream.memcpy_stod(&rng)?;
    let mut d_barrier = stream.alloc_zeros::<u32>(1)?;
    let d_marks = stream.alloc_zeros::<i32>(1)?;
    let mut d_max = stream.alloc_zeros::<i32>(1)?;
    let tiles = ((n + 511) / 512) as i32;
    let mut d_eligible = stream.alloc_zeros::<i32>(cfg.grid_dim.0 as usize)?;
    let mut d_candidates = stream.alloc_zeros::<i32>(1 + 2 * n)?;
    let mut d_keys = stream.alloc_zeros::<u64>(n.max(64 * 4096))?;
    let mut d_buckets = stream.alloc_zeros::<i32>(128)?;
    let mut d_selected = stream.alloc_zeros::<u64>(n)?;
    let mut d_order = stream.alloc_zeros::<i32>(n)?;
    let mut d_control = stream.alloc_zeros::<i32>(16)?;
    let mut d_masks = stream.alloc_zeros::<u64>(65)?;
    let mut d_old = stream.alloc_zeros::<i32>(n)?;
    let mut d_data = stream.alloc_zeros::<i32>(577)?;
    let mut d_events = stream.alloc_zeros::<u64>(1024)?;
    let kernel = module.load_function("gl_k1")?;
    let mut den = (cool as u64).pow(3);
    anyhow::ensure!(den <= u64::MAX / 16, "schedule overflow");
    host_tabu.fill(0);
    stream.memcpy_htod(host_tabu, tabu)?;
    let (mut round, mut lots, mut accepted, mut perturb) = (start, 0usize, 0u64, 0usize);
    let mut tri_start = start;
    let (mut reason, mut stagnant, mut zero) = (0i32, 0i32, 0i32);
    let mut control = [0i32; 16];
    while round < total {
        let (batch, check_tri) = if pass == 0 {
            tri.batch(tri_start, round, batch_size)
        } else {
            ((total - round).min(batch_size), false)
        };
        anyhow::ensure!(
            batch > 0 && batch <= total - round && batch <= 2_048,
            "invalid loop batch"
        );
        let batch = batch as u64;
        stream.memcpy_htod(&[0u32], &mut d_barrier)?;
        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&(n as i32))
                .arg(&(np as i32))
                .arg(&cap)
                .arg(&challenge.d_hyperedge_nodes)
                .arg(&challenge.d_node_hyperedges)
                .arg(&mut *part)
                .arg(&mut *sizes)
                .arg(&d_tasks)
                .arg(&task_count)
                .arg(&mut *flags)
                .arg(&mut *priorities)
                .arg(&mut d_barrier)
                .arg(&0u32)
                .arg(&mut *tabu)
                .arg(&d_marks)
                .arg(&0i32)
                .arg(&0i32)
                .arg(&0i32)
                .arg(&0i32)
                .arg(&(round as i32))
                .arg(&pass)
                .arg(&batch)
                .arg(&den)
                .arg(&mut d_max)
                .arg(&mut d_eligible)
                .arg(&tiles)
                .arg(&mut d_candidates)
                .arg(&(total as i32))
                .arg(&(cool as i32))
                .arg(&3i32)
                .arg(&(limit as i32))
                .arg(&8i32)
                .arg(&4i32)
                .arg(&2i32)
                .arg(&12i32)
                .arg(&4096i32)
                .arg(&16i32)
                .arg(&(pert as i32))
                .arg(&0i32)
                .arg(&stagnant)
                .arg(&zero)
                .arg(&mut d_keys)
                .arg(&mut d_buckets)
                .arg(&mut d_selected)
                .arg(&mut d_order)
                .arg(&mut d_control)
                .arg(&d_rng)
                .arg(&challenge.d_node_offsets)
                .arg(&challenge.d_hyperedge_offsets)
                .arg(&(ne as i32))
                .arg(&mut d_counts)
                .arg(&mut d_masks)
                .arg(&mut d_old)
                .arg(&mut d_data)
                .arg(&mut d_events)
                .arg(&d_meta)
                .arg(&(words as i32))
                .launch(cfg.clone())?;
        }
        stream.memcpy_dtoh(&d_control, &mut control)?;
        stream.synchronize()?;
        reason = control[0];
        anyhow::ensure!(
            (0..=4).contains(&reason)
                && control[1] >= 0
                && control[1] as u64 <= batch
                && control[2] >= 0
                && control[2] as u64 <= 11 * batch + 2
                && control[3] >= 0
                && control[3] as u64 <= n as u64 * control[1] as u64
                && control[4] >= 0
                && control[5] >= 0,
            "invalid loop control",
        );
        let completed = control[1] as usize;
        anyhow::ensure!(
            reason != 0 || completed == batch as usize,
            "incomplete loop batch"
        );
        anyhow::ensure!(reason != 2 || completed != 0, "empty perturbation batch");
        round += completed;
        lots += 1;
        accepted += control[3] as u64;
        stagnant = control[4];
        zero = control[5];
        if pass == 0 && (check_tri || reason != 0) {
            tri.launched(round);
            if tri.finish_pending(false, challenge, stream, offsets, edges, saver)?
                || tri.check(round, part, challenge, stream, offsets, edges, saver)?
            {
                break;
            }
            if tri.reduce_budget() {
                total = tri.total;
                cool = tri.cool;
                pert = tri.pert;
                den = (cool as u64).pow(3);
                anyhow::ensure!(den <= u64::MAX / 16, "schedule overflow");
                if round >= total {
                    break;
                }
            }
        }
        match reason {
            0 => {}
            1 | 3 | 4 => break,
            2 => {
                stream.memcpy_dtoh(&*part, host_part)?;
                stream.memcpy_dtoh(&*sizes, host_sizes)?;
                stream.synchronize()?;
                anyhow::ensure!(
                    gl_a0(host_part, host_sizes, n, np).is_none(),
                    "invalid partition before perturbation",
                );
                super::track::perturb_random(
                    host_part,
                    host_sizes,
                    cap,
                    3,
                    987654321u64 + round.saturating_sub(1) as u64 * 123456789u64,
                );
                stream.memcpy_htod(host_part, part)?;
                stream.memcpy_htod(host_sizes, sizes)?;
                stagnant = 0;
                perturb += 1;
                tri_start = round;
            }
            _ => unreachable!(),
        }
    }
    stream.memcpy_dtoh(&*part, host_part)?;
    stream.memcpy_dtoh(&*sizes, host_sizes)?;
    stream.memcpy_dtoh(&*tabu, host_tabu)?;
    stream.synchronize()?;
    anyhow::ensure!(
        gl_a0(host_part, host_sizes, n, np).is_none(),
        "invalid loop output (reason {})",
        reason,
    );
    saver.save(&super::to_u32(host_part))?;
    if pass == 0 {
        tri.finish_pending(true, challenge, stream, offsets, edges, saver)?;
        tri.reduce_budget();
    }
    Ok((round, reason, lots, accepted, perturb))
}

#[inline]
fn gl_a3(sz: i32) -> usize {
    if sz <= 4 {
        0
    } else if sz <= 8 {
        1
    } else if sz <= 16 {
        2
    } else if sz <= 32 {
        3
    } else if sz <= 64 {
        4
    } else {
        5
    }
}

fn gl_a4(units: &[(i32, i32, i32)], warps: usize) -> (Vec<i32>, i32) {
    let mut by_class: Vec<Vec<(i32, i32, i32)>> = (0..6).map(|_| Vec::new()).collect();
    for &(id, st, sz) in units.iter() {
        by_class[gl_a3(sz)].push((id, st, sz));
    }
    let mut light: Vec<(usize, usize)> = Vec::new();
    let mut heavy: Vec<(usize, usize)> = Vec::new();
    for c in (0..6).rev() {
        let n = by_class[c].len();
        if n == 0 {
            continue;
        }
        let per_task = 32 / [1usize, 2, 4, 8, 16, 32][c];
        let ntask = (n + per_task - 1) / per_task;
        for t in 0..ntask {
            if c == 0 {
                light.push((c, t * per_task));
            } else {
                heavy.push((c, t * per_task));
            }
        }
    }
    let n_tasks = light.len() + heavy.len();
    let extra = if n_tasks > warps { n_tasks - warps } else { 0 };
    let mut order: Vec<(usize, usize)> = Vec::with_capacity(n_tasks);
    if extra > 0 && light.len() >= 2 * extra {
        let head = light.len() - extra;
        order.extend_from_slice(&light[head..]);
        order.extend_from_slice(&heavy);
        order.extend_from_slice(&light[..head]);
    } else {
        order.extend_from_slice(&heavy);
        order.extend_from_slice(&light);
    }
    let mut slots: Vec<i32> = Vec::with_capacity(n_tasks * 32 * 4);
    for &(c, first) in order.iter() {
        let n = by_class[c].len();
        for l in 0..32usize {
            let k = first + (l >> c);
            let (id, st, sz) = by_class[c][std::cmp::min(k, n - 1)];
            slots.push(id);
            slots.push(st);
            slots.push(sz);
            slots.push(c as i32);
        }
    }
    (slots, n_tasks as i32)
}

pub(super) fn gm_a1(
    challenge: &Challenge,
    module: &Arc<CudaModule>,
    prop: &cudarc::runtime::sys::cudaDeviceProp,
    part: &[i32],
    sizes: &[i32],
) -> Result<LaunchConfig, &'static str> {
    let n = challenge.num_nodes as usize;
    if !(256..524288).contains(&n) || challenge.max_part_size > (i32::MAX - 8) as u32 {
        return Err("bounds");
    }
    debug_assert_eq!(gm_a3(60_000, 1000, 0), 60_000);
    debug_assert_eq!(gm_a3(60_000, 100, 0), 6_000);
    debug_assert_eq!(gm_a3(500_000, 3000, 0), 1_500_000);
    debug_assert_eq!(gm_a3(60_000, 1000, 100_000_000_000), 11_250);
    debug_assert_eq!(gm_a3(50, 100, 1), 5);
    debug_assert_eq!(gl_a0(&[0, 0, 0, 1], &[3, 1], 4, 2), None);
    debug_assert_eq!(gl_a0(&[0, 0, 0, 1], &[2, 2], 4, 2), Some("size_match"));
    if let Some(why) = gl_a0(part, sizes, n, challenge.num_parts as usize) {
        return Err(why);
    }
    if prop.maxThreadsPerBlock < 512
        || prop.major < 7
        || prop.multiProcessorCount <= 0
        || prop.sharedMemPerBlock < 40 * 1024
        || prop.sharedMemPerMultiprocessor < 80 * 1024
    {
        return Err("hardware");
    }
    let blocks = prop.multiProcessorCount as u32;
    if u64::from(blocks) * (11 * 128 + 2) > u64::from(u32::MAX) {
        return Err("barrier");
    }
    let kernel = module.load_function("gm_k1").map_err(|_| "kernel")?;
    if kernel
        .occupancy_max_active_blocks_per_multiprocessor(512, 0, None)
        .map_err(|_| "residency")?
        < 1
    {
        return Err("residency");
    }
    Ok(LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (512, 1, 1),
        shared_mem_bytes: 0,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gm_a2(
    challenge: &Challenge,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    cfg: LaunchConfig,
    total: usize,
    batch_size: usize,
    limit: usize,
    part: &mut CudaSlice<i32>,
    sizes: &mut CudaSlice<i32>,
    tabu: &mut CudaSlice<i32>,
    priorities: &mut CudaSlice<i32>,
    host_part: &mut [i32],
    host_sizes: &mut [i32],
    host_tabu: &mut [i32],
    hedge_offsets: &[i32],
    saver: &mut super::Saver<'_>,
) -> anyhow::Result<(usize, i32, usize, usize)> {
    let n = challenge.num_nodes as usize;
    let np = challenge.num_parts as usize;
    let ne = challenge.num_hyperedges as usize;
    let cap = challenge.max_part_size as i32;
    let offsets = stream.memcpy_dtov(&challenge.d_node_offsets)?;
    stream.synchronize()?;
    anyhow::ensure!(
        offsets.len() == n + 1
            && hedge_offsets.len() == ne + 1
            && offsets[0] == 0
            && offsets[n] as usize == challenge.d_node_hyperedges.len()
            && offsets.windows(2).all(|p| p[0] >= 0 && p[0] <= p[1])
            && hedge_offsets[0] == 0
            && hedge_offsets[ne] as usize == challenge.d_hyperedge_nodes.len()
            && hedge_offsets.windows(2).all(|p| p[0] >= 0 && p[0] <= p[1]),
        "invalid graph offsets",
    );
    let mut meta = Vec::with_capacity(ne);
    let mut words = 0usize;
    for span in hedge_offsets.windows(2) {
        let size = span[1] - span[0];
        let format = if size <= 255 {
            0u32
        } else if size <= 65535 {
            1u32
        } else {
            2u32
        };
        anyhow::ensure!(words <= (u32::MAX >> 2) as usize, "count offset overflow");
        meta.push(((words as u32) << 2) | format);
        words += 16usize << format;
    }
    anyhow::ensure!(
        words <= i32::MAX as usize
            && limit <= i32::MAX as usize / 3
            && (1..=1_500_000).contains(&total)
            && (1..=1_500_000).contains(&batch_size)
            && n as u64 * batch_size as u64 <= i32::MAX as u64
            && u64::from(cfg.grid_dim.0) * (11 * batch_size as u64 + 2) <= u64::from(u32::MAX),
        "invalid loop bounds",
    );
    let items: Vec<(i32, i32, i32)> = (0..n)
        .map(|v| (v as i32, offsets[v], offsets[v + 1] - offsets[v]))
        .collect();
    let (tasks, task_count) = gl_a4(&items, 4 * cfg.grid_dim.0 as usize);
    debug_assert_eq!(tasks.len(), task_count as usize * 128);
    let d_tasks = stream.memcpy_stod(&tasks)?;
    let d_meta = stream.memcpy_stod(&meta)?;
    let mut d_counts = stream.alloc_zeros::<i32>(words.max(1))?;
    let mut rng = Vec::with_capacity(2 * (n + 1));
    let (mut a, mut c) = (1u64, 0u64);
    for _ in 0..=n {
        rng.push(a);
        rng.push(c);
        a = a.wrapping_mul(super::LCG_MUL);
        c = c.wrapping_mul(super::LCG_MUL).wrapping_add(LCG_INC);
    }
    let d_rng = stream.memcpy_stod(&rng)?;
    let mut d_barrier = stream.alloc_zeros::<u32>(1)?;
    let mut d_max = stream.alloc_zeros::<i32>(1)?;
    let tiles = ((n + 511) / 512) as i32;
    let mut d_eligible = stream.alloc_zeros::<i32>(cfg.grid_dim.0 as usize)?;
    let mut d_candidates = stream.alloc_zeros::<i32>(1 + 2 * n)?;
    let mut d_keys = stream.alloc_zeros::<u64>(n.max(64 * 4096))?;
    let mut d_buckets = stream.alloc_zeros::<i32>(128)?;
    let mut d_selected = stream.alloc_zeros::<u64>(n)?;
    let mut d_order = stream.alloc_zeros::<i32>(n)?;
    let mut d_control = stream.alloc_zeros::<i32>(16)?;
    let mut d_slots = stream.alloc_zeros::<u64>(cfg.grid_dim.0 as usize)?;
    let mut d_flags = stream.alloc_zeros::<u64>(2 * ne)?;
    let mut d_old = stream.alloc_zeros::<i32>(n)?;
    let mut d_data = stream.alloc_zeros::<i32>(577)?;
    let mut d_events = stream.alloc_zeros::<u64>(1024)?;
    let kernel = module.load_function("gm_k1")?;
    let den = (total as u64).pow(2);
    anyhow::ensure!(den <= u64::MAX / 16, "schedule overflow");
    host_tabu.fill(0);
    stream.memcpy_htod(host_tabu, tabu)?;
    let (mut round, mut lots, mut perturb) = (0usize, 0usize, 0usize);
    let (mut reason, mut stagnant, mut zero) = (0i32, 0i32, 0i32);
    let mut control = [0i32; 16];
    while round < total {
        let batch = (total - round).min(batch_size) as u64;
        stream.memcpy_htod(&[0u32], &mut d_barrier)?;
        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&(n as i32))
                .arg(&(np as i32))
                .arg(&cap)
                .arg(&challenge.d_hyperedge_nodes)
                .arg(&challenge.d_node_hyperedges)
                .arg(&mut *part)
                .arg(&mut *sizes)
                .arg(&d_tasks)
                .arg(&task_count)
                .arg(&mut d_flags)
                .arg(&mut *priorities)
                .arg(&mut d_barrier)
                .arg(&0u32)
                .arg(&mut *tabu)
                .arg(&(round as i32))
                .arg(&batch)
                .arg(&den)
                .arg(&mut d_max)
                .arg(&mut d_eligible)
                .arg(&tiles)
                .arg(&mut d_candidates)
                .arg(&(total as i32))
                .arg(&(total as i32))
                .arg(&2i32)
                .arg(&(limit as i32))
                .arg(&8i32)
                .arg(&4i32)
                .arg(&2i32)
                .arg(&8i32)
                .arg(&2048i32)
                .arg(&16i32)
                .arg(&(total as i32))
                .arg(&stagnant)
                .arg(&zero)
                .arg(&mut d_keys)
                .arg(&mut d_buckets)
                .arg(&mut d_selected)
                .arg(&mut d_order)
                .arg(&mut d_control)
                .arg(&d_rng)
                .arg(&challenge.d_node_offsets)
                .arg(&challenge.d_hyperedge_offsets)
                .arg(&(ne as i32))
                .arg(&mut d_counts)
                .arg(&mut d_slots)
                .arg(&mut d_old)
                .arg(&mut d_data)
                .arg(&mut d_events)
                .arg(&d_meta)
                .arg(&(words as i32))
                .launch(cfg.clone())?;
        }
        stream.memcpy_dtoh(&d_control, &mut control)?;
        stream.synchronize()?;
        reason = control[0];
        anyhow::ensure!(
            (0..=4).contains(&reason)
                && control[1] >= 0
                && control[1] as u64 <= batch
                && control[2] >= 0
                && control[2] as u64 <= 11 * batch + 2
                && control[3] >= 0
                && control[3] as u64 <= n as u64 * control[1] as u64
                && control[4] >= 0
                && control[5] >= 0,
            "invalid loop control",
        );
        let completed = control[1] as usize;
        anyhow::ensure!(
            reason != 0 || completed == batch as usize,
            "incomplete loop batch"
        );
        anyhow::ensure!(reason != 2 || completed != 0, "empty perturbation batch");
        round += completed;
        lots += 1;
        stagnant = control[4];
        zero = control[5];
        match reason {
            0 => {}
            1 | 3 | 4 => break,
            2 => {
                stream.memcpy_dtoh(&*part, host_part)?;
                stream.memcpy_dtoh(&*sizes, host_sizes)?;
                stream.synchronize()?;
                anyhow::ensure!(
                    gl_a0(host_part, host_sizes, n, np).is_none(),
                    "invalid partition before perturbation",
                );
                super::track::perturb_random(
                    host_part,
                    host_sizes,
                    cap,
                    3,
                    987654321u64 + round.saturating_sub(1) as u64 * 123456789u64,
                );
                stream.memcpy_htod(host_part, part)?;
                stream.memcpy_htod(host_sizes, sizes)?;
                stagnant = 0;
                perturb += 1;
            }
            _ => unreachable!(),
        }
    }
    stream.memcpy_dtoh(&*part, host_part)?;
    stream.memcpy_dtoh(&*sizes, host_sizes)?;
    stream.memcpy_dtoh(&*tabu, host_tabu)?;
    stream.synchronize()?;
    anyhow::ensure!(
        gl_a0(host_part, host_sizes, n, np).is_none(),
        "invalid loop output (reason {})",
        reason,
    );
    saver.save(&super::to_u32(host_part))?;
    Ok((round, reason, lots, perturb))
}

pub(super) fn gm_a3(rounds: usize, permille: usize, fuel: u64) -> usize {
    let total = rounds * permille / 1000;
    if fuel == 0 {
        total
    } else {
        let d = &super::params::D_10K.d;
        let usable = (fuel / 10) * 7;
        let affordable = usable.saturating_sub(d.fuel_fixed) / d.fuel_per_round;
        total.min(affordable.max(50) as usize)
    }
}
