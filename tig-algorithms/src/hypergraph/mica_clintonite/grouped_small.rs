use super::params::Knobs;
use super::{FINISH_THREADS, SELECT_THREADS};
use cudarc::driver::{safe::LaunchConfig, CudaModule, CudaSlice, CudaStream, PushKernelArg};
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
        for _ in 0..self.batch {
            unsafe {
                stream
                    .launch_builder(&flags)
                    .arg(&(challenge.num_hyperedges as i32))
                    .arg(&n)
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&*part)
                    .arg(&mut *fa)
                    .arg(&mut *fd)
                    .arg(&mut self.dh)
                    .launch(hedge_cfg.clone())?;
                stream
                    .launch_builder(&moves)
                    .arg(&n)
                    .arg(&np)
                    .arg(&cap)
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&challenge.d_node_offsets)
                    .arg(&*part)
                    .arg(&*sizes)
                    .arg(&*fa)
                    .arg(&*fd)
                    .arg(&mut *keys)
                    .arg(&num_wide)
                    .arg(&main_blocks)
                    .arg(&num_hubs)
                    .arg(order)
                    .arg(&self.dh)
                    .launch(moves_cfg.clone())?;
                stream
                    .launch_builder(&pick)
                    .arg(&*keys)
                    .arg(&*tabu)
                    .arg(&n)
                    .arg(&0i32)
                    .arg(&i32::MIN)
                    .arg(&mut *buf)
                    .arg(&mut *sorted)
                    .arg(&*sizes)
                    .arg(&np)
                    .arg(&cap)
                    .arg(&0i32)
                    .arg(&0i32)
                    .arg(&16384i32)
                    .arg(&mut *picks)
                    .arg(&self.dh)
                    .launch(sort_cfg.clone())?;
                stream
                    .launch_builder(&finish)
                    .arg(&mut self.dh)
                    .arg(&self.dc)
                    .arg(&*picks)
                    .arg(&*sorted)
                    .arg(&mut *part)
                    .arg(&mut *sizes)
                    .arg(&mut *tabu)
                    .launch(finish_cfg.clone())?;
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
