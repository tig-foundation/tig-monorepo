use super::params::{Knobs, Track};
use super::{FILTER_THREADS, FINISH_THREADS, LCG_INC, SELECT_THREADS};
use cudarc::driver::{
    result, safe::LaunchConfig, sys, CudaModule, CudaSlice, CudaStream, DevicePtr, DriverError,
    LaunchArgs, PinnedHostSlice, PushKernelArg,
};
use std::sync::Arc;
use tig_challenges::hypergraph::Challenge;

pub(super) const GL_MAX_ROUNDS: usize = 500_000;

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
type Config = [u64; 18];

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
const COOL_DEN: usize = 17;

fn cooling_den(span: u64, power: u32, pct: u64) -> u64 {
    (u128::from(span).pow(power) * 100 / u128::from(pct)) as u64
}

fn extend_schedule(
    round: u64,
    before: [u64; 3],
    ext: u64,
    cap: u64,
    power: u32,
    pct: u64,
) -> ([u64; 3], u64, u64) {
    let old_den = cooling_den(before[1], power, pct);
    if round >= before[0] {
        return (before, old_den, 1_000);
    }
    let left = before[0] - round;
    let room = cap.saturating_sub(before[0]).min(left);
    let applied = ext.min(1_000 + room * 1_000 / left);
    let after = before.map(|bound| {
        if bound > round {
            round + ((bound - round) * applied + 999) / 1_000
        } else {
            bound
        }
    });
    let den = cooling_den(before[1] * applied / 1_000, power, pct);
    let old_num = u128::from(before[1].saturating_sub(round)).pow(power);
    let new_num = u128::from(after[1].saturating_sub(round)).pow(power);
    let lhs = new_num * u128::from(old_den);
    let rhs = old_num * u128::from(den);
    if lhs.abs_diff(rhs) * 200 > rhs {
        return (before, old_den, 1_000);
    }
    (after, den, applied)
}

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
    if c[COOL_DEN] != 0 {
        den = c[COOL_DEN];
    }
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
    hlist_cfg: LaunchConfig,
    mut moves_cfg: LaunchConfig,
    scalar_blocks: u32,
    move_seq: bool,
    move_ilp: i64,
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
        1,
        0,
    ];
    let mut h: Header = [0; 32];
    h[FULL_FLAGS] = 1;
    h[GC] = u64::from(gc);
    schedule(&mut h, &c);
    let mut dc = stream.memcpy_stod(&c)?;
    let mut dh = stream.memcpy_stod(&h)?;
    if track.mvc {
        moves_cfg.grid_dim.0 = moves_cfg.grid_dim.0 - scalar_blocks + (scalar_blocks + 3) / 4;
    }
    let mut pairs = stream.alloc_zeros::<i32>(3 * n)?;
    let mut counts = stream.alloc_zeros::<u32>(32 * ne)?;
    let cached_count = module.load_function("update_edge_pin_counts_100k")?;
    let cached_refresh = module.load_function("update_edge_flags_from_counts_100k")?;
    if let Some(name) = track.k_expand {
        module.load_function(name)?;
    }
    if let Some(name) = track.k_edge_flags_incr {
        module.load_function(name)?;
    }
    module.load_function(track.k_edge_flags)?;
    let moves_name = track.k_moves;
    let moves = module.load_function(moves_name)?;
    let moves = if move_seq {
        moves
    } else {
        module.load_function(&format!("{}_ilp{}", moves_name, move_ilp))?
    };
    let stat = module.load_function("mica_sx_stats")?;
    let pick = module.load_function(if sort_mb != 0 {
        "mica_filt_pick4_mb"
    } else {
        "mica_filt_pick4"
    })?;
    let sort = module.load_function("mica_sortsel")?;
    let finish = module.load_function("grouped_finish_counts_100k")?;
    let resume = module.load_function("grouped_resume")?;
    let mut counts_cfg = hlist_cfg.clone();
    counts_cfg.grid_dim.0 = counts_cfg.grid_dim.0.min(32);
    let stats_cfg = LaunchConfig {
        grid_dim: (filt_grid, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };
    let pick_cfg = LaunchConfig {
        grid_dim: (filt_grid, 1, 1),
        block_dim: (FILTER_THREADS, 1, 1),
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
    let lz = ne == 200_000 && knobs.pz && hp.int("lz", 0, 1, super::LZ) == 1;
    let persistent_name = match moves_name {
        "compute_refinement_moves_warp8_20k"
        | "compute_refinement_moves_warp8_50k" => {
            Some("persistent_rounds_zgm0")
        }
        "compute_refinement_moves_warp8_100k" => Some("persistent_rounds_zgm1"),
        "compute_refinement_moves_warp8_200k" => Some(if lz {
            "persistent_rounds_zgm0_mv200_fa_lka1_a184"
        } else {
            "persistent_rounds_zgm0_mv200_fa_lka1"
        }),
        _ => None,
    };
    let sp_sel = sort_mb > 0 && matches!(ne, 20_000 | 50_000)
        && hp.int("sp_sel", 0, 1, 1) == 1;
    let fa = ne == 200_000 && persistent_name.is_some();
    let barriers_per_round = (if sort_mb == 0 { 7u64 } else { 11u64 }) + u64::from(fa);
    let mut selected_name = None;
    let persistent = if knobs.pz
        && np <= 64
        && counts_cfg.block_dim == (128, 1, 1)
        && moves_cfg.block_dim == (64, 1, 1)
    {
        persistent_name.and_then(|name| {
            let attempts = if sp_sel { 2 } else { 1 };
            (0..attempts).find_map(|attempt| {
                let optimized = sp_sel && attempt == 0;
                let sel_name = if optimized { Some(format!("{}_sel", name)) } else { None };
                let name = sel_name.as_deref().unwrap_or(name);
                let kernel = module.load_function(name).ok()?;
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
                if ne == 200_000 {
                    selected_name = Some(name.to_owned());
                }
                Some((
                    kernel,
                    LaunchConfig {
                        grid_dim: (sms as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    optimized,
                ))
            })
        })
    } else {
        None
    };
    let fa = fa && persistent.is_some();
    let lz = selected_name.as_deref().map_or(false, |name| name.ends_with("_a184"));
    let sp_sel = sp_sel && persistent.as_ref().map_or(false, |p| p.2);
    let tile_size = if sp_sel { 256 } else { 2048 };
    let tiles = ((n + tile_size - 1) / tile_size) as i32;
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
    let finish_prepared = if fa {
        Some(stream.alloc_zeros::<i32>(1 + n + (n + 255) / 256)?)
    } else {
        None
    };
    let finish_prepared_ptr = finish_prepared.as_ref().map_or(0, |d| d.device_ptr(stream).0);
    let a184 = if lz {
        Some(stream.alloc_zeros::<u64>(4 * ne)?)
    } else {
        None
    };
    let a184_ptr = a184.as_ref().map_or(0, |d| d.device_ptr(stream).0);
    anyhow::ensure!(a184_ptr % 32 == 0, "28");
    let move_tasks = if ne == 200_000 && persistent.is_some() {
        anyhow::ensure!(
            offsets.len() == n + 1 && offsets[0] == 0
                && offsets[n] as usize == node_hedges.len()
                && offsets.windows(2).all(|p| p[0] >= 0 && p[0] <= p[1]),
            "1",
        );
        let warps = persistent.as_ref().unwrap().1.grid_dim.0 as usize * 8;
        let (tasks, count) = moves200_tasks(offsets, warps);
        Some((stream.memcpy_stod(&tasks)?, count))
    } else {
        None
    };
    let persistent_order = move_tasks.as_ref().map_or(order, |(tasks, _)| tasks);
    let persistent_moves_grid = move_tasks.as_ref()
        .map_or(moves_cfg.grid_dim.0 as i32, |(_, count)| *count);
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
        if let (Some((kernel, cfg, _)), Some(counter)) = (persistent.as_ref(), barrier.as_mut()) {
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
                    .arg(&persistent_moves_grid)
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
                    .arg(persistent_order)
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
                    .arg(&addresses[22])
                    .arg(&addresses[23])
                    .arg(&addresses[24])
                    .arg(&finish_prepared_ptr)
                    .arg(&a184_ptr)
                    .launch(cfg.clone())?;
            }
        } else {
            let mut submit = |capture| -> anyhow::Result<()> {
                for _ in 0..batch {
                    unsafe {
                        stream
                            .launch_builder(&cached_count)
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
                            .launch_builder(&cached_refresh)
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
                        stream
                            .launch_builder(&moves)
                            .arg(&ni)
                            .arg(&npi)
                            .arg(&capacity)
                            .graph_input(&challenge.d_node_hyperedges, &addresses[2], capture)
                            .graph_input(&challenge.d_node_offsets, &addresses[3], capture)
                            .graph_input(&*part, &addresses[6], capture)
                            .graph_input(&*sizes, &addresses[9], capture)
                            .graph_input(&*flags, &addresses[8], capture)
                            .graph_output(&mut *priorities, &addresses[10], capture)
                            .graph_input(order, &addresses[11], capture)
                            .arg(&hubs)
                            .arg(&lanes4)
                            .arg(&lanes2)
                            .graph_input(&dh, &addresses[0], capture)
                            .launch(moves_cfg.clone())?;
                        stream
                            .launch_builder(&stat)
                            .graph_input(&*priorities, &addresses[10], capture)
                            .graph_input(&*tabu, &addresses[12], capture)
                            .arg(&ni)
                            .arg(&0i32)
                            .arg(&-3i32)
                            .graph_output(&mut *block_max, &addresses[13], capture)
                            .graph_output(&mut *stats, &addresses[14], capture)
                            .graph_input(&dh, &addresses[0], capture)
                            .launch(stats_cfg.clone())?;
                        if let Some((pass, select, seg_hist, tile_hist, tgt_hist)) = mb.as_mut() {
                            stream
                                .launch_builder(&pick)
                                .graph_input(&*priorities, &addresses[10], capture)
                                .graph_input(&*tabu, &addresses[12], capture)
                                .arg(&ni)
                                .arg(&0i32)
                                .arg(&-3i32)
                                .graph_input(&*block_max, &addresses[13], capture)
                                .graph_input(&*stats, &addresses[14], capture)
                                .arg(&0u64)
                                .arg(&LCG_INC)
                                .arg(&0u64)
                                .arg(&0u64)
                                .arg(&0u64)
                                .arg(&0u64)
                                .arg(&0u64)
                                .graph_output(&mut *filt_out, &addresses[15], capture)
                                .graph_output(&mut *block_sel, &addresses[16], capture)
                                .graph_output(&mut *h4, &addresses[17], capture)
                                .graph_input(&dh, &addresses[0], capture)
                                .graph_output(&mut *seg_hist, &addresses[22], capture)
                                .graph_output(&mut *tile_hist, &addresses[23], capture)
                                .graph_output(&mut *tgt_hist, &addresses[24], capture)
                                .arg(&tiles)
                                .launch(pick_cfg.clone())?;
                            for p in 0..4i32 {
                                stream
                                    .launch_builder(pass)
                                    .graph_input(&*filt_out, &addresses[15], capture)
                                    .graph_input(&*block_sel, &addresses[16], capture)
                                    .arg(&gi)
                                    .arg(&chunk)
                                    .arg(&tiles)
                                    .graph_output(&mut *sort_buf, &addresses[18], capture)
                                    .graph_output(&mut *sorted, &addresses[19], capture)
                                    .graph_input(&*h4, &addresses[17], capture)
                                    .graph_input(&*seg_hist, &addresses[22], capture)
                                    .graph_output(&mut *tile_hist, &addresses[23], capture)
                                    .graph_output(&mut *tgt_hist, &addresses[24], capture)
                                    .arg(&p)
                                    .arg(&0i32)
                                    .arg(&0i32)
                                    .graph_input(&dh, &addresses[0], capture)
                                    .launch(mb_cfg.clone())?;
                            }
                            stream
                                .launch_builder(select)
                                .graph_input(&*block_sel, &addresses[16], capture)
                                .arg(&gi)
                                .graph_input(&*sorted, &addresses[19], capture)
                                .graph_input(&*sizes, &addresses[9], capture)
                                .arg(&npi)
                                .arg(&capacity)
                                .arg(&0i32)
                                .arg(&0i32)
                                .arg(&extra)
                                .graph_output(&mut *picks, &addresses[20], capture)
                                .graph_output(&mut *h4, &addresses[17], capture)
                                .graph_input(&*tgt_hist, &addresses[24], capture)
                                .graph_input(&dh, &addresses[0], capture)
                                .launch(mb_cfg.clone())?;
                        } else {
                            stream
                                .launch_builder(&pick)
                                .graph_input(&*priorities, &addresses[10], capture)
                                .graph_input(&*tabu, &addresses[12], capture)
                                .arg(&ni)
                                .arg(&0i32)
                                .arg(&-3i32)
                                .graph_input(&*block_max, &addresses[13], capture)
                                .graph_input(&*stats, &addresses[14], capture)
                                .arg(&0u64)
                                .arg(&LCG_INC)
                                .arg(&0u64)
                                .arg(&0u64)
                                .arg(&0u64)
                                .arg(&0u64)
                                .arg(&0u64)
                                .graph_output(&mut *filt_out, &addresses[15], capture)
                                .graph_output(&mut *block_sel, &addresses[16], capture)
                                .graph_output(&mut *h4, &addresses[17], capture)
                                .graph_input(&dh, &addresses[0], capture)
                                .launch(pick_cfg.clone())?;
                            stream
                                .launch_builder(&sort)
                                .graph_input(&*filt_out, &addresses[15], capture)
                                .graph_input(&*block_sel, &addresses[16], capture)
                                .arg(&gi)
                                .arg(&chunk)
                                .graph_output(&mut *sort_buf, &addresses[18], capture)
                                .graph_output(&mut *sorted, &addresses[19], capture)
                                .graph_input(&*sizes, &addresses[9], capture)
                                .arg(&npi)
                                .arg(&capacity)
                                .arg(&0i32)
                                .arg(&0i32)
                                .arg(&extra)
                                .graph_output(&mut *picks, &addresses[20], capture)
                                .graph_output(&mut *h4, &addresses[17], capture)
                                .graph_input(&dh, &addresses[0], capture)
                                .launch(sort_cfg.clone())?;
                        }
                        stream
                            .launch_builder(&finish)
                            .graph_output(&mut dh, &addresses[0], capture)
                            .graph_input(&dc, &addresses[21], capture)
                            .graph_input(&*picks, &addresses[20], capture)
                            .graph_input(&*sorted, &addresses[19], capture)
                            .graph_output(&mut *part, &addresses[6], capture)
                            .graph_output(&mut *sizes, &addresses[9], capture)
                            .graph_output(&mut *tabu, &addresses[12], capture)
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
                    tri.reduce_budget(hp, track, knobs)?
                {
                    c[ROUNDS] = knobs.refinement_rounds as u64;
                    c[COOL_ROUNDS] = cool as u64;
                    c[PERT_ROUNDS] = pert as u64;
                    c[COOL_DEN] = 0;
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
            if let Some(ext) = tri.take_extension() {
                ev[slot].synchronize()?;
                let mut current: Header = [0; 32];
                current.copy_from_slice(pin[slot].as_slice()?);
                let before = [c[ROUNDS], c[COOL_ROUNDS], c[PERT_ROUNDS]];
                let fuel = knobs.refinement_fuel;
                let cap = if fuel == 0 {
                    u64::MAX
                } else {
                    (((fuel / 10) * 7).saturating_sub(track.d.fuel_fixed)
                        / track.d.fuel_per_round).max(50)
                };
                let (after, den, applied) = extend_schedule(
                    current[ROUND], before, ext,
                    if current[STATE] == DONE { before[0] } else { cap },
                    c[COOL_POW] as u32, c[COOL_PCT],
                );
                if applied > 1_000 {
                    c[ROUNDS] = after[0];
                    c[COOL_ROUNDS] = after[1];
                    c[PERT_ROUNDS] = after[2];
                    c[COOL_DEN] = den;
                    schedule(&mut current, &c);
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
        return Some("f");
    }
    if sizes.iter().any(|&s| s <= 0) {
        return Some("v");
    }
    if sizes.iter().map(|&s| i64::from(s)).sum::<i64>() != n as i64 {
        return Some("u");
    }
    let mut counts = [0i32; 64];
    for &p in part {
        if p < 0 || p as usize >= np {
            return Some("p");
        }
        counts[p as usize] += 1;
    }
    if &counts[..np] != sizes {
        return Some("m");
    }
    None
}

pub(super) fn gl_a1(
    challenge: &Challenge,
    module: &Arc<CudaModule>,
    prop: &cudarc::runtime::sys::cudaDeviceProp,
    part: &[i32],
    sizes: &[i32],
    kernel_name: &str,
) -> Result<LaunchConfig, &'static str> {
    let n = challenge.num_nodes as usize;
    if !(256..524288).contains(&n) || challenge.max_part_size > (i32::MAX - 8) as u32 {
        return Err("b");
    }
    debug_assert_eq!(gl_a0(&[0, 0, 0, 1], &[3, 1], 4, 2), None);
    debug_assert_eq!(gl_a0(&[0, 0, 0, 1], &[2, 2], 4, 2), Some("m"));
    if let Some(why) = gl_a0(part, sizes, n, challenge.num_parts as usize) {
        return Err(why);
    }
    if prop.maxThreadsPerBlock < 512
        || prop.major < 7
        || prop.multiProcessorCount <= 0
        || prop.sharedMemPerBlock < 40 * 1024
        || prop.sharedMemPerMultiprocessor < 80 * 1024
    {
        return Err("h");
    }
    let blocks = prop.multiProcessorCount as u32;
    if u64::from(blocks) * (11 * 128 + 2) > u64::from(u32::MAX) {
        return Err("r");
    }
    let kernel = module.load_function(kernel_name).map_err(|_| "k")?;
    if kernel
        .occupancy_max_active_blocks_per_multiprocessor(512, 0, None)
        .map_err(|_| "s")?
        < 1
    {
        return Err("s");
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
    kernel_name: &str,
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
) -> anyhow::Result<()> {
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
            && total <= GL_MAX_ROUNDS
            && cool <= GL_MAX_ROUNDS
            && pert <= GL_MAX_ROUNDS
            && (1..=2_048).contains(&batch_size)
            && n as u64 * batch_size.max(128) as u64 <= i32::MAX as u64
            && u64::from(cfg.grid_dim.0) * (11 * batch_size.max(128) as u64 + 2)
                <= u64::from(u32::MAX),
        "2",
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
    let kernel = module.load_function(kernel_name)?;
    let mut den = (cool as u64).pow(3);
    anyhow::ensure!(den <= u64::MAX / 16, "3");
    host_tabu.fill(0);
    stream.memcpy_htod(host_tabu, tabu)?;
    let mut round = start;
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
            "4"
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
            "5",
        );
        let completed = control[1] as usize;
        anyhow::ensure!(
            reason != 0 || completed == batch as usize,
            "6"
        );
        anyhow::ensure!(reason != 2 || completed != 0, "7");
        round += completed;
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
                anyhow::ensure!(den <= u64::MAX / 16, "3");
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
                    "8",
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
        "9 ({})",
        reason,
    );
    saver.save(&super::to_u32(host_part))?;
    if pass == 0 {
        tri.finish_pending(true, challenge, stream, offsets, edges, saver)?;
        tri.reduce_budget();
    }
    Ok(())
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

fn moves200_tasks(offsets: &[i32], warps: usize) -> (Vec<i32>, i32) {
    let units: Vec<_> = offsets.windows(2).enumerate()
        .map(|(node, p)| (node as i32, p[0], p[1] - p[0])).collect();
    let (mut tasks, count) = gl_a4(&units, warps);
    let mut nodes = 0;
    for task in tasks.chunks_exact_mut(128) {
        let lanes = 1usize << task[3];
        nodes += 32 / lanes;
        for group in (1..32 / lanes).rev() {
            let first = group * lanes * 4;
            if task[first] == task[first - lanes * 4] {
                for slot in task[first..first + lanes * 4].chunks_exact_mut(4) {
                    slot[0] = -1;
                }
                nodes -= 1;
            }
        }
    }
    debug_assert_eq!(nodes, units.len());
    debug_assert_eq!(tasks.len(), count as usize * 128);
    (tasks, count)
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
        return Err("b");
    }
    debug_assert_eq!(gm_a3(60_000, 1000, 0), 60_000);
    debug_assert_eq!(gm_a3(60_000, 100, 0), 6_000);
    debug_assert_eq!(gm_a3(500_000, 3000, 0), 1_500_000);
    debug_assert_eq!(gm_a3(60_000, 1000, 100_000_000_000), 11_250);
    debug_assert_eq!(gm_a3(50, 100, 1), 5);
    debug_assert_eq!(gl_a0(&[0, 0, 0, 1], &[3, 1], 4, 2), None);
    debug_assert_eq!(gl_a0(&[0, 0, 0, 1], &[2, 2], 4, 2), Some("m"));
    if let Some(why) = gl_a0(part, sizes, n, challenge.num_parts as usize) {
        return Err(why);
    }
    if prop.maxThreadsPerBlock < 512
        || prop.major < 7
        || prop.multiProcessorCount <= 0
        || prop.sharedMemPerBlock < 40 * 1024
        || prop.sharedMemPerMultiprocessor < 80 * 1024
    {
        return Err("h");
    }
    let blocks = prop.multiProcessorCount as u32;
    if u64::from(blocks) * (11 * 128 + 2) > u64::from(u32::MAX) {
        return Err("r");
    }
    let kernel = module.load_function("gm_k1").map_err(|_| "k")?;
    if kernel
        .occupancy_max_active_blocks_per_multiprocessor(512, 0, None)
        .map_err(|_| "s")?
        < 1
    {
        return Err("s");
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
    start: usize,
    fuel: u64,
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
    tri: &mut Option<super::track::Tri100>,
) -> anyhow::Result<(usize, bool)> {
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
            && start < total
            && (1..=1_500_000).contains(&batch_size)
            && n as u64 * batch_size as u64 <= i32::MAX as u64
            && u64::from(cfg.grid_dim.0) * (11 * batch_size as u64 + 2) <= u64::from(u32::MAX),
        "2",
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
    let mut den = (total as u64).pow(2);
    let mut end = total;
    anyhow::ensure!(den <= u64::MAX / 16, "3");
    host_tabu.fill(0);
    stream.memcpy_htod(host_tabu, tabu)?;
    let mut round = start;
    let (mut reason, mut stagnant, mut zero) = (0i32, 0i32, 0i32);
    let mut control = [0i32; 16];
    while round < end {
        let batch = (end - round).min(batch_size) as u64;
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
                .arg(&(end as i32))
                .arg(&(end as i32))
                .arg(&2i32)
                .arg(&(limit as i32))
                .arg(&8i32)
                .arg(&4i32)
                .arg(&2i32)
                .arg(&8i32)
                .arg(&2048i32)
                .arg(&16i32)
                .arg(&(end as i32))
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
                .arg(&0i32)
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
            "5",
        );
        let completed = control[1] as usize;
        anyhow::ensure!(
            reason != 0 || completed == batch as usize,
            "6"
        );
        anyhow::ensure!(reason != 2 || completed != 0, "7");
        round += completed;
        stagnant = control[4];
        zero = control[5];
        if let Some(tri) = tri.as_mut() {
            if tri.check(round, &*part, challenge, module, stream, saver)? {
                return Ok((round, true));
            }
            if let Some(ext) = tri.take_extension() {
                let before = [total as u64; 3];
                let cap = if matches!(reason, 0 | 2) {
                    gm_a3(1_500_000, 1_000, fuel) as u64
                } else {
                    total as u64
                };
                let (after, next_den, _) =
                    extend_schedule(round as u64, before, ext, cap, 2, 100);
                end = after[0] as usize;
                den = next_den;
                anyhow::ensure!(den <= u64::MAX / 16, "3");
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
                    "8",
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
        "9 ({})",
        reason,
    );
    saver.save(&super::to_u32(host_part))?;
    Ok((round, false))
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

fn cg_score(st: &super::gain::State, part: &[super::gain::Block]) -> Option<u64> {
    if part.len() != st.n {
        return None;
    }
    let mut sizes = [0u64; 64];
    for (v, &b) in part.iter().enumerate() {
        if b as usize >= st.k {
            return None;
        }
        sizes[b as usize] += u64::from(st.weight[v]);
    }
    if sizes[..st.k].iter().any(|&s| s == 0 || s > u64::from(st.max_block_size)) {
        return None;
    }
    let mut score = 0u64;
    for e in 0..st.m {
        let mut mask = 0u64;
        for &v in &st.he_nodes[st.he_off[e] as usize..st.he_off[e + 1] as usize] {
            mask |= 1u64 << part[v as usize];
        }
        score += u64::from(mask.count_ones().saturating_sub(1));
    }
    Some(score)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_cg(
    challenge: &Challenge,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    track: &Track,
    knobs: &Knobs,
    hp: &super::Hp<'_>,
    cool_pow: u32,
    cool_pct: u64,
    slack_scale: usize,
    offsets: &[i32],
    nodes: &[i32],
    partition: &[u32],
    saver: &mut super::Saver<'_>,
) -> anyhow::Result<Vec<u32>> {
    use super::gain::{Block, State};
    let t = hp.int("cg_t", 2, 1000, 160) as usize;
    let limit = hp.int("cg_lv", 1, 12, 8) as usize;
    let rounds = hp.int("cg_r", 1, 500, 32) as usize;
    let n = challenge.num_nodes as usize;
    let m = challenge.num_hyperedges as usize;
    let k = challenge.num_parts as usize;
    anyhow::ensure!(
        n > 0 && n <= i32::MAX as usize - 2 && m <= i32::MAX as usize
            && (1..=64).contains(&k) && challenge.max_part_size <= i32::MAX as u32
            && offsets.len() == m + 1 && offsets[0] == 0
            && offsets[m] >= 0 && offsets[m] as usize == nodes.len()
            && offsets.windows(2).all(|x| x[0] >= 0 && x[0] <= x[1])
            && nodes.iter().all(|&v| v >= 0 && (v as usize) < n)
            && partition.len() == n && partition.iter().all(|&b| (b as usize) < k),
        "10",
    );
    let Some(initial) = State::new(
        n, m, k, offsets.iter().map(|&x| x as u32).collect(),
        nodes.iter().map(|&x| x as u32).collect(),
        partition.iter().map(|&x| x as Block).collect(), challenge.max_part_size, None,
    ) else {
        return Ok(partition.to_vec());
    };
    let Some(initial_score) = cg_score(&initial, &initial.part) else {
        return Ok(partition.to_vec());
    };
    let target = t * k;
    let cmax = (2 * ((n as u64 + target as u64 - 1) / target as u64))
        .max(2).min(u64::from(challenge.max_part_size)) as u32;
    let mut states = vec![initial];
    let mut maps = Vec::new();
    while maps.len() < limit && states.last().unwrap().n > target {
        let prev = states.last().unwrap();
        let contracted = if m == 100_000 {
            super::ml2::contract::<false>(prev, maps.len() as u64 + 1, cmax)
        } else {
            super::ml2::contract::<true>(prev, maps.len() as u64 + 1, cmax)
        };
        let Some((next, map)) = contracted else {
            break;
        };
        debug_assert!(next.n as u64 * 100 <= prev.n as u64 * 95);
        debug_assert_eq!(next.weight.iter().map(|&w| u64::from(w)).sum::<u64>(), n as u64);
        debug_assert!(map.iter().enumerate().all(|(v, &u)| next.part[u as usize] == prev.part[v]));
        debug_assert_eq!(cg_score(prev, &prev.part), cg_score(&next, &next.part));
        maps.push(map);
        states.push(next);
    }
    let mut part = states.last().unwrap().part.clone();
    let mut final_score = None;
    for index in (0..states.len()).rev() {
        let st = &states[index];
        let (result, best_round, km1_best) = cg_level(
            st, &part, module, stream, track, knobs, rounds, cool_pow, cool_pct, slack_scale,
        )?;
        let after = cg_score(st, &result);
        debug_assert_eq!(after, Some(km1_best));
        debug_assert!(cg_score(st, &part).map_or(false, |score| km1_best <= score));
        debug_assert!(best_round <= rounds as u32);
        debug_assert!(best_round != 0 || result == part);
        if after.is_none() {
            break;
        }
        if index == 0 {
            part = result;
            final_score = after;
        } else {
            part = maps[index - 1].iter().map(|&v| result[v as usize]).collect();
        }
    }
    let kept = final_score.map_or(false, |score| score < initial_score);
    if kept {
        let result: Vec<u32> = part.iter().map(|&b| u32::from(b)).collect();
        saver.save(&result)?;
        Ok(result)
    } else {
        Ok(partition.to_vec())
    }
}

#[allow(clippy::too_many_arguments)]
fn cg_level(
    st: &super::gain::State,
    partition: &[super::gain::Block],
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    track: &Track,
    knobs: &Knobs,
    rounds: usize,
    cool_pow: u32,
    cool_pct: u64,
    slack_scale: usize,
) -> anyhow::Result<(Vec<super::gain::Block>, u32, u64)> {
    if st.m == 0 {
        return Ok((partition.to_vec(), 0, 0));
    }
    let ni = st.n as i32;
    let nei = st.m as i32;
    let np = st.k as i32;
    let capacity = st.max_block_size as i32;
    let gi = track.filt_blocks.min(st.n as u32).max(1) as i32;
    let chunk = ((st.n + gi as usize - 1) / gi as usize) as i32;
    let extra = track.extra_window.min(st.n) as i32;
    let active_rounds = 0u64;
    let c: Config = [
        st.n as u64, st.k as u64, st.max_block_size as u64, st.m as u64,
        rounds as u64, active_rounds, cool_pow as u64, cool_pct, active_rounds,
        knobs.move_limit as u64, extra as u64, slack_scale as u64,
        knobs.tabu_tenure as u64, track.tabu_fail_mark_len as u64,
        track.tabu_fail_tenure as u64, knobs.max_stagnant_rounds as u64, 0,
        0,
    ];
    let mut h: Header = [0; 32];
    h[FULL_FLAGS] = 1;
    schedule(&mut h, &c);
    let dc = stream.memcpy_stod(&c)?;
    let dh = stream.memcpy_stod(&h)?;
    let nd_off: Vec<i32> = st.nd_off.iter().map(|&v| v as i32).collect();
    let d_nd_off = stream.memcpy_stod(&nd_off)?;
    let d_nd_edges = stream.memcpy_stod(if st.nd_edges.is_empty() { &[0u32][..] } else { &st.nd_edges[..] })?;
    let d_he_off = stream.memcpy_stod(&st.he_off)?;
    let d_he_nodes = stream.memcpy_stod(if st.he_nodes.is_empty() { &[0u32][..] } else { &st.he_nodes[..] })?;
    let d_weights = stream.memcpy_stod(&st.weight)?;
    let d_order = stream.memcpy_stod(&super::track::degree_order(&nd_off, st.n))?;
    let mut host_part = vec![0i32; st.n + 2];
    let mut sizes = vec![0i32; st.k];
    for (v, &b) in partition.iter().enumerate() {
        host_part[v] = i32::from(b);
        sizes[b as usize] += st.weight[v] as i32;
    }
    let d_part = stream.memcpy_stod(&host_part)?;
    let d_sizes = stream.memcpy_stod(&sizes)?;
    let d_flags = stream.alloc_zeros::<u64>((2 * st.m).max(2))?;
    let d_priorities = stream.alloc_zeros::<i32>(st.n)?;
    let d_tabu = stream.alloc_zeros::<i32>(st.n)?;
    let d_max = stream.alloc_zeros::<i32>(gi as usize)?;
    let d_stats = stream.alloc_zeros::<i32>(gi as usize * 5)?;
    let d_selected = stream.alloc_zeros::<i32>(gi as usize)?;
    let d_h4 = stream.alloc_zeros::<i32>(1024)?;
    let d_out = stream.alloc_zeros::<u64>(st.n + gi as usize)?;
    let d_buf = stream.alloc_zeros::<u64>(st.n)?;
    let d_sorted = stream.alloc_zeros::<u64>(st.n)?;
    let d_picks = stream.alloc_zeros::<u64>(st.n + 2)?;
    let d_pairs = stream.alloc_zeros::<i32>(1)?;
    let nb = ((st.m + 511) / 512) as i32;
    let d_conn = stream.alloc_zeros::<i32>(st.m)?;
    let d_sum = stream.alloc_zeros::<i32>(nb as usize)?;
    let d_best = stream.alloc_zeros::<i32>(st.n + st.k + 5)?;
    let a = [
        dh.device_ptr(stream).0, dc.device_ptr(stream).0,
        d_he_nodes.device_ptr(stream).0, d_he_off.device_ptr(stream).0,
        d_nd_edges.device_ptr(stream).0, d_nd_off.device_ptr(stream).0,
        d_part.device_ptr(stream).0, d_sizes.device_ptr(stream).0,
        d_weights.device_ptr(stream).0, d_order.device_ptr(stream).0,
        d_flags.device_ptr(stream).0, d_priorities.device_ptr(stream).0,
        d_tabu.device_ptr(stream).0, d_max.device_ptr(stream).0,
        d_stats.device_ptr(stream).0, d_selected.device_ptr(stream).0,
        d_h4.device_ptr(stream).0, d_out.device_ptr(stream).0,
        d_buf.device_ptr(stream).0, d_sorted.device_ptr(stream).0,
        d_picks.device_ptr(stream).0, d_pairs.device_ptr(stream).0,
        d_conn.device_ptr(stream).0, d_sum.device_ptr(stream).0,
        d_best.device_ptr(stream).0,
    ];
    stream.context().check_err()?;
    let flags = module.load_function("precompute_edge_flags_200k")?;
    let moves = module.load_function("compute_refinement_moves_warp8_w")?;
    let stats = module.load_function("mica_sx_stats")?;
    let pick = module.load_function("mica_filt_pick4")?;
    let sort = module.load_function("mica_sortsel_w")?;
    let finish = module.load_function("grouped_finish_w")?;
    let result = module.load_function("grouped_result_w")?;
    let connectivity = module.load_function("compute_connectivity_200k")?;
    let reduce = module.load_function("reduce_connectivity_sum_200k")?;
    let store = module.load_function("grouped_store_w")?;
    let launch = |blocks, threads| LaunchConfig {
        grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0,
    };
    let update = || -> anyhow::Result<()> {
        unsafe {
            stream.launch_builder(&connectivity)
                .arg(&nei).arg(&a[2]).arg(&a[3]).arg(&a[6]).arg(&a[22])
                .launch(launch((st.m as u32 + 127) / 128, 128))?;
            stream.launch_builder(&reduce)
                .arg(&nei).arg(&a[22]).arg(&a[23])
                .launch(LaunchConfig { shared_mem_bytes: 32, ..launch(nb as u32, 256) })?;
            stream.launch_builder(&store)
                .arg(&ni).arg(&np).arg(&nb).arg(&a[23]).arg(&a[6]).arg(&a[7])
                .arg(&a[24]).arg(&a[0])
                .launch(launch(1, 256))?;
        }
        Ok(())
    };
    update()?;
    let submit = || -> anyhow::Result<()> {
        unsafe {
            stream.launch_builder(&flags)
                .arg(&nei).arg(&ni).arg(&a[2]).arg(&a[3]).arg(&a[6]).arg(&a[10]).arg(&a[0])
                .launch(launch(((st.m as u32 + 127) / 128).max(1), 128))?;
            stream.launch_builder(&moves)
                .arg(&ni).arg(&np).arg(&capacity).arg(&a[4]).arg(&a[5])
                .arg(&a[6]).arg(&a[7]).arg(&a[10]).arg(&a[11]).arg(&a[9]).arg(&a[8]).arg(&a[0])
                .launch(launch((st.n as u32 + 7) / 8, 64))?;
            stream.launch_builder(&stats)
                .arg(&a[11]).arg(&a[12]).arg(&ni).arg(&0i32).arg(&-3i32)
                .arg(&a[13]).arg(&a[14]).arg(&a[0])
                .launch(launch(gi as u32, 64))?;
            stream.launch_builder(&pick)
                .arg(&a[11]).arg(&a[12]).arg(&ni).arg(&0i32).arg(&-3i32)
                .arg(&a[13]).arg(&a[14]).arg(&0u64).arg(&LCG_INC)
                .arg(&0u64).arg(&0u64).arg(&0u64).arg(&0u64).arg(&0u64)
                .arg(&a[17]).arg(&a[15]).arg(&a[16]).arg(&a[0])
                .launch(launch(gi as u32, FILTER_THREADS))?;
            stream.launch_builder(&sort)
                .arg(&a[17]).arg(&a[15]).arg(&gi).arg(&chunk).arg(&a[18]).arg(&a[19])
                .arg(&a[7]).arg(&np).arg(&capacity).arg(&0i32).arg(&0i32).arg(&extra)
                .arg(&a[20]).arg(&a[16]).arg(&a[0]).arg(&a[8])
                .launch(launch(1, SELECT_THREADS))?;
            stream.launch_builder(&finish)
                .arg(&a[0]).arg(&a[1]).arg(&a[20]).arg(&a[19]).arg(&a[6]).arg(&a[7])
                .arg(&a[12]).arg(&a[5]).arg(&a[21]).arg(&a[8])
                .launch(launch(1, FINISH_THREADS))?;
        }
        update()
    };
    let graph = BatchGraph::capture(stream, submit);
    for _ in 0..rounds {
        if let Some(graph) = &graph {
            graph.launch()?;
        } else {
            submit()?;
        }
    }
    unsafe {
        stream.launch_builder(&result).arg(&ni).arg(&a[24]).arg(&a[0])
            .launch(launch(1, 1))?;
    }
    let host = stream.memcpy_dtov(&d_best)?;
    debug_assert!({
        sizes.fill(0);
        host[..st.n].iter().enumerate().all(|(v, &b)| {
            if b < 0 || b >= np {
                return false;
            }
            sizes[b as usize] += st.weight[v] as i32;
            true
        }) && sizes[..] == host[st.n + 5..]
    });
    let best_round = host[st.n + 2] as u32;
    let km1_best = u64::from(host[st.n + 3] as u32) | (u64::from(host[st.n + 4] as u32) << 32);
    let part = host[..st.n].iter().map(|&b| {
        if b >= 0 && b < np { b as super::gain::Block } else { super::gain::Block::MAX }
    }).collect();
    Ok((part, best_round, km1_best))
}
