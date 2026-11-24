use super::grouped::{BatchGraph, GraphArgs};
use super::params::Knobs;
use super::{FINISH_THREADS, SELECT_THREADS};
use cudarc::driver::{
    safe::LaunchConfig, sys, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr,
    PushKernelArg,
};
use std::sync::Arc;
use tig_challenges::hypergraph::Challenge;

type Header = [u64; 32];
type Config = [u64; 7];

const STATE: usize = 0;
const ROUND: usize = 1;
const GLOBAL_ROUND: usize = 2;
const STAGNANT: usize = 3;
const PHASE_SLACK: usize = 3;
const PHASE_EXTRA: usize = 7;
const LIMIT: usize = 14;
const PHASE: usize = 15;

const RUNNING: u64 = 0;
const HOST_STEP: u64 = 1;
const DONE: u64 = 2;

const NODES: usize = 0;
const PARTS: usize = 1;
const CAPACITY: usize = 2;
const ROUNDS: usize = 3;
const MOVE_LIMIT: usize = 4;
const MAX_STAGNANT: usize = 6;

const PERTURB_AFTER: u64 = 5;
const PERTURB_MARGIN: u64 = 50;

pub(super) struct Persistent {
    kernel: CudaFunction,
    cfg: LaunchConfig,
    counter: CudaSlice<u32>,
    rounds: u32,
}

impl Persistent {
    pub fn new(
        module: &Arc<CudaModule>,
        stream: &Arc<CudaStream>,
        knobs: &Knobs,
        np: usize,
        flags_cfg: &LaunchConfig,
        moves_cfg: &LaunchConfig,
    ) -> anyhow::Result<Option<Self>> {
        if !knobs.pz
            || !(1..=64).contains(&np)
            || flags_cfg.block_dim != (128, 1, 1)
            || moves_cfg.block_dim != (64, 1, 1)
            || flags_cfg.grid_dim.0 == 0
            || moves_cfg.grid_dim.0 == 0
            || flags_cfg.grid_dim.1 != 1
            || flags_cfg.grid_dim.2 != 1
            || moves_cfg.grid_dim.1 != 1
            || moves_cfg.grid_dim.2 != 1
        {
            return Ok(None);
        }
        let checked = module
            .load_function("persistent_rounds_10k")
            .ok()
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
                let generations = 1 + 4 * u64::from(knobs.pz_rounds);
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
                Some((kernel, sms as u32))
            });
        match checked {
            Some((kernel, blocks)) => Ok(Some(Self {
                kernel,
                cfg: LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                counter: stream.alloc_zeros::<u32>(1)?,
                rounds: knobs.pz_rounds,
            })),
            None => Ok(None),
        }
    }
}

fn resume(h: &mut Header, c: &Config, executed: i32) -> bool {
    h[STATE] = RUNNING;
    if h[PHASE] != 0 {
        if executed == 0 || h[ROUND] >= c[ROUNDS] {
            h[STATE] = DONE;
        }
        return false;
    }
    let mut perturb = false;
    if executed > 0 {
        h[STAGNANT] = 0;
    } else {
        h[STAGNANT] += 1;
        if h[STAGNANT] >= PERTURB_AFTER && h[ROUND] - 1 < c[ROUNDS].saturating_sub(PERTURB_MARGIN) {
            h[STAGNANT] = 0;
            perturb = true;
        } else if h[STAGNANT] > c[MAX_STAGNANT] {
            h[STATE] = DONE;
        }
    }
    if h[STATE] != DONE && h[ROUND] >= c[ROUNDS] {
        h[STATE] = DONE;
    }
    perturb
}

pub(super) struct Session {
    h: Header,
    c: Config,
    batch: usize,
    graph: Option<BatchGraph>,
    capture_attempted: bool,
    dh: CudaSlice<u64>,
    dc: CudaSlice<u64>,
}
impl Session {
    pub fn new(
        stream: &Arc<CudaStream>,
        knobs: &Knobs,
        batch: usize,
        part: &[i32],
        sizes: &[i32],
        cap: i32,
    ) -> anyhow::Result<Self> {
        let c = [
            part.len() as u64,
            sizes.len() as u64,
            cap as u64,
            knobs.refinement_rounds as u64,
            knobs.move_limit as u64,
            knobs.tabu_tenure as u64,
            knobs.max_stagnant_rounds as u64,
        ];
        let mut h = [0; 32];
        h[LIMIT] = c[MOVE_LIMIT];
        Ok(Self {
            h,
            c,
            batch,
            graph: None,
            capture_attempted: false,
            dh: stream.memcpy_stod(&h)?,
            dc: stream.memcpy_stod(&c)?,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn new_phase(
        stream: &Arc<CudaStream>,
        knobs: &Knobs,
        batch: usize,
        part: &[i32],
        sizes: &[i32],
        cap: i32,
        rounds: usize,
        limit: usize,
        slack: usize,
        extra: usize,
        global: i32,
        phase: u64,
    ) -> anyhow::Result<Self> {
        let c = [
            part.len() as u64,
            sizes.len() as u64,
            cap as u64,
            rounds as u64,
            limit as u64,
            knobs.tabu_tenure as u64,
            0,
        ];
        let mut h = [0; 32];
        h[GLOBAL_ROUND] = global as u64;
        h[PHASE_SLACK] = slack as u64;
        h[PHASE_EXTRA] = extra as u64;
        h[LIMIT] = limit as u64;
        h[PHASE] = phase;
        if rounds == 0 {
            h[STATE] = DONE;
        }
        Ok(Self {
            h,
            c,
            batch,
            graph: None,
            capture_attempted: false,
            dh: stream.memcpy_stod(&h)?,
            dc: stream.memcpy_stod(&c)?,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn batch(
        &mut self,
        challenge: &Challenge,
        module: &Arc<CudaModule>,
        stream: &Arc<CudaStream>,
        persistent: Option<&mut Persistent>,
        hedge_cfg: &LaunchConfig,
        moves_cfg: &LaunchConfig,
        order: &CudaSlice<i32>,
        num_wide: i32,
        main_blocks: i32,
        num_hubs: i32,
        part: &mut CudaSlice<i32>,
        sizes: &mut CudaSlice<i32>,
        tabu: &mut CudaSlice<i32>,
        fa: &mut CudaSlice<u64>,
        fd: &mut CudaSlice<u64>,
        keys: &mut CudaSlice<i32>,
        buf: &mut CudaSlice<u64>,
        sorted: &mut CudaSlice<u64>,
        picks: &mut CudaSlice<u64>,
    ) -> anyhow::Result<()> {
        if let Some(p) = persistent {
            debug_assert!(
                (1 + 4 * u64::from(p.rounds)) * u64::from(p.cfg.grid_dim.0) <= u32::MAX as u64
            );
            debug_assert_eq!(
                moves_cfg.grid_dim.0 as i32,
                (num_wide + 1) / 2 + (num_hubs - num_wide + 7) / 8 + main_blocks
            );
            stream.memset_zeros(&mut p.counter)?;
            unsafe {
                stream
                    .launch_builder(&p.kernel)
                    .arg(&mut self.dh)
                    .arg(&self.dc)
                    .arg(&mut p.counter)
                    .arg(&p.rounds)
                    .arg(&(challenge.num_hyperedges as i32))
                    .arg(&(hedge_cfg.grid_dim.0 as i32))
                    .arg(&(moves_cfg.grid_dim.0 as i32))
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&challenge.d_node_offsets)
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&mut *part)
                    .arg(&mut *sizes)
                    .arg(&mut *tabu)
                    .arg(&mut *fa)
                    .arg(&mut *fd)
                    .arg(&mut *keys)
                    .arg(order)
                    .arg(&num_wide)
                    .arg(&main_blocks)
                    .arg(&num_hubs)
                    .arg(&mut *buf)
                    .arg(&mut *sorted)
                    .arg(&mut *picks)
                    .launch(p.cfg.clone())?;
            }
            stream.memcpy_dtoh(&self.dh, &mut self.h)?;
            return Ok(());
        }
        if let Some(graph) = self.graph.as_ref() {
            graph.launch()?;
            stream.memcpy_dtoh(&self.dh, &mut self.h)?;
            return Ok(());
        }
        let n = self.c[NODES] as i32;
        let np = self.c[PARTS] as i32;
        let cap = self.c[CAPACITY] as i32;
        let flags = module.load_function("precompute_edge_flags_10k")?;
        let moves = module.load_function("compute_refinement_moves_10k")?;
        let pick = module.load_function("mica_pick1")?;
        let finish = module.load_function("grouped_small_finish")?;
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
        let addresses = if !self.capture_attempted {
            [
                challenge.d_hyperedge_nodes.device_ptr(stream).0,
                challenge.d_hyperedge_offsets.device_ptr(stream).0,
                part.device_ptr(stream).0,
                fa.device_ptr(stream).0,
                fd.device_ptr(stream).0,
                self.dh.device_ptr(stream).0,
                challenge.d_node_hyperedges.device_ptr(stream).0,
                challenge.d_node_offsets.device_ptr(stream).0,
                sizes.device_ptr(stream).0,
                keys.device_ptr(stream).0,
                order.device_ptr(stream).0,
                tabu.device_ptr(stream).0,
                buf.device_ptr(stream).0,
                sorted.device_ptr(stream).0,
                picks.device_ptr(stream).0,
                self.dc.device_ptr(stream).0,
            ]
        } else {
            [0; 16]
        };
        stream.context().check_err()?;
        {
            let batch = self.batch;
            let dh = &mut self.dh;
            let dc = &self.dc;
            let mut submit = |capture| -> anyhow::Result<()> {
                for _ in 0..batch {
                    unsafe {
                        stream
                            .launch_builder(&flags)
                            .arg(&(challenge.num_hyperedges as i32))
                            .arg(&n)
                            .graph_input(&challenge.d_hyperedge_nodes, &addresses[0], capture)
                            .graph_input(&challenge.d_hyperedge_offsets, &addresses[1], capture)
                            .graph_input(&*part, &addresses[2], capture)
                            .graph_output(&mut *fa, &addresses[3], capture)
                            .graph_output(&mut *fd, &addresses[4], capture)
                            .graph_output(&mut *dh, &addresses[5], capture)
                            .launch(hedge_cfg.clone())?;
                        stream
                            .launch_builder(&moves)
                            .arg(&n)
                            .arg(&np)
                            .arg(&cap)
                            .graph_input(&challenge.d_node_hyperedges, &addresses[6], capture)
                            .graph_input(&challenge.d_node_offsets, &addresses[7], capture)
                            .graph_input(&*part, &addresses[2], capture)
                            .graph_input(&*sizes, &addresses[8], capture)
                            .graph_input(&*fa, &addresses[3], capture)
                            .graph_input(&*fd, &addresses[4], capture)
                            .graph_output(&mut *keys, &addresses[9], capture)
                            .arg(&num_wide)
                            .arg(&main_blocks)
                            .arg(&num_hubs)
                            .graph_input(order, &addresses[10], capture)
                            .graph_input(&*dh, &addresses[5], capture)
                            .launch(moves_cfg.clone())?;
                        stream
                            .launch_builder(&pick)
                            .graph_input(&*keys, &addresses[9], capture)
                            .graph_input(&*tabu, &addresses[11], capture)
                            .arg(&n)
                            .arg(&0i32)
                            .arg(&i32::MIN)
                            .graph_output(&mut *buf, &addresses[12], capture)
                            .graph_output(&mut *sorted, &addresses[13], capture)
                            .graph_input(&*sizes, &addresses[8], capture)
                            .arg(&np)
                            .arg(&cap)
                            .arg(&0i32)
                            .arg(&0i32)
                            .arg(&16384i32)
                            .graph_output(&mut *picks, &addresses[14], capture)
                            .graph_input(&*dh, &addresses[5], capture)
                            .launch(sort_cfg.clone())?;
                        stream
                            .launch_builder(&finish)
                            .graph_output(&mut *dh, &addresses[5], capture)
                            .graph_input(dc, &addresses[15], capture)
                            .graph_input(&*picks, &addresses[14], capture)
                            .graph_input(&*sorted, &addresses[13], capture)
                            .graph_output(&mut *part, &addresses[2], capture)
                            .graph_output(&mut *sizes, &addresses[8], capture)
                            .graph_output(&mut *tabu, &addresses[11], capture)
                            .launch(finish_cfg.clone())?;
                    }
                }
                Ok(())
            };
            if !self.capture_attempted {
                self.capture_attempted = true;
                self.graph = BatchGraph::capture(stream, || submit(true));
            }
            if let Some(graph) = self.graph.as_ref() {
                graph.launch()?;
            } else {
                submit(false)?;
            }
        }
        stream.memcpy_dtoh(&self.dh, &mut self.h)?;
        Ok(())
    }
    pub fn is_done(&self) -> bool {
        self.h[STATE] == DONE
    }
    pub fn wants_host_step(&self) -> bool {
        self.h[STATE] == HOST_STEP
    }
    pub fn round(&self) -> u64 {
        self.h[ROUND]
    }
    pub fn global_round(&self) -> i32 {
        self.h[GLOBAL_ROUND] as i32
    }
    pub fn after_hyperedge(&mut self, executed: i32) -> bool {
        resume(&mut self.h, &self.c, executed)
    }
    pub fn publish(&mut self, stream: &Arc<CudaStream>) -> anyhow::Result<()> {
        stream.memcpy_htod(&self.h, &mut self.dh)?;
        Ok(())
    }
}
