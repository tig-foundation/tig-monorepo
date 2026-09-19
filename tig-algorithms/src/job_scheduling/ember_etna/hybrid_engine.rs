//! Hybrid flow-shop track: tabu search over swap, relocate and reassign moves on a disjunctive
//! graph with incremental longest paths, alternating between the greedy start and a constructed
//! start.
use anyhow::Result;
use std::cell::{Cell, RefCell};
use tig_challenges::job_scheduling::*;

use rand::{rngs::SmallRng, SeedableRng};

use self::graph::{Graph, GraphState};
use self::model::Model;
use self::tabu::{Budget, TabuCfg};

extern "C" {
    #[allow(non_upper_case_globals)]
    static __fuel_remaining: u64;
}

#[inline(always)]
fn fuel_remaining() -> u64 {
    unsafe { core::ptr::read_volatile(core::ptr::addr_of!(__fuel_remaining)) }
}

/// Fuel the construction and the tabu search may spend together after the greedy baseline.
const FUEL_CAP: u64 = 200_000_000_000;
/// Share of the remaining fuel kept in reserve (1 / RESERVE_DIV).
const RESERVE_DIV: u64 = 64;
/// Non-improving iterations before a restart from the best schedule.
const RESTART_AFTER: u32 = 150_000;
/// Base of the number of machine reassignment moves generated per iteration.
const ASSIGN_MOVES_BASE: usize = 32;
/// Consecutive restarts without any improvement in between before the search leaves its start.
const STERILE_RESTARTS: u32 = 1;
/// Constructions of the alternative start.
const ALT_CONSTRUCTIONS: usize = 1000;
/// Fuel below which the alternative start is not constructed.
const ALT_MIN_FUEL: u64 = 100_000_000_000;

/// A start of the tabu search: its best schedule and makespan.
struct Start {
    state: GraphState,
    best: u32,
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let greedy = ref_greedy::greedy_baseline(challenge)?;
    let floor_ms = challenge.evaluate_makespan(&greedy)?;
    save_solution(&greedy)?;

    let available = fuel_remaining();
    let reserve = available / RESERVE_DIV;
    let max_spend = available.saturating_sub(reserve).min(FUEL_CAP);
    let budget = Budget {
        floor: available.saturating_sub(max_spend),
    };

    let md = Model::build(challenge);
    let mut g = Graph::new(&md);
    if !g.load_schedule(&md, &greedy.job_schedule) {
        return Ok(());
    }
    let retimed = match g.evaluate() {
        Some(ms) => ms,
        None => return Ok(()),
    };
    let ceiling = retimed.min(floor_ms);
    let mut greedy_start = Start {
        state: GraphState::new(&md),
        best: ceiling,
    };
    g.snapshot(&mut greedy_start.state);
    if retimed < floor_ms {
        save_solution(&Solution {
            job_schedule: g.to_schedule(&md),
        })?;
    }

    let before_cal = fuel_remaining();
    let _ = g.evaluate();
    let per_eval = before_cal.saturating_sub(fuel_remaining()).max(1);

    let mut rng = SmallRng::from_seed(challenge.seed);
    // A failed save ends the search: the callback signals the fuel floor this way.
    let stop = Cell::new(false);
    let saved_ms = RefCell::new(ceiling);
    let mut on_improve = |gr: &Graph, ms: u32| -> bool {
        if ms < *saved_ms.borrow() {
            match save_solution(&Solution {
                job_schedule: gr.to_schedule(&md),
            }) {
                Ok(()) => *saved_ms.borrow_mut() = ms,
                Err(_) => stop.set(true),
            }
        }
        !stop.get()
    };

    let mut flex_ops = 0usize;
    let mut flex_opts = 0usize;
    for o in 0..md.n_ops {
        let k = md.n_opts(o);
        if k > 1 {
            flex_ops += 1;
            flex_opts += k;
        }
    }
    let avg_flex = (flex_opts / flex_ops.max(1)).max(1);
    let assign_moves = (ASSIGN_MOVES_BASE * 2 / avg_flex).max(4);

    let cfg = TabuCfg {
        max_assign_moves: assign_moves,
        restart_after: RESTART_AFTER,
        sterile_limit: STERILE_RESTARTS,
        any_restart: false,
    };

    // The constructed start is the best schedule the construction reports; improvements are also
    // saved.
    let alt_best: RefCell<Option<(u32, Solution)>> = RefCell::new(None);
    let sink = |s: &Solution| -> Result<()> {
        if let Ok(mk) = challenge.evaluate_makespan(s) {
            if alt_best
                .borrow()
                .as_ref()
                .map_or(true, |(best, _)| mk < *best)
            {
                *alt_best.borrow_mut() = Some((mk, s.clone()));
            }
            if mk < *saved_ms.borrow() {
                if let Err(e) = save_solution(s) {
                    stop.set(true);
                    return Err(e);
                }
                *saved_ms.borrow_mut() = mk;
            }
        }
        Ok(())
    };
    if available >= ALT_MIN_FUEL {
        let _ = super::hfs_construct::construct(challenge, &sink, ALT_CONSTRUCTIONS);
    }
    if stop.get() {
        return Ok(());
    }

    let (stopped, ms) = tabu::tabu_search(
        &md,
        &mut g,
        &mut greedy_start.state,
        greedy_start.best,
        &cfg,
        &mut rng,
        &budget,
        per_eval,
        &mut on_improve,
    );
    greedy_start.best = ms;
    if !stopped {
        return Ok(());
    }

    let Some((_, alt)) = alt_best.into_inner() else {
        // No constructed start: the greedy start keeps the whole budget and never leaves.
        let cfg_last = TabuCfg {
            sterile_limit: 0,
            ..cfg
        };
        g.restore_from(&greedy_start.state);
        let _ = tabu::tabu_search(
            &md,
            &mut g,
            &mut greedy_start.state,
            greedy_start.best,
            &cfg_last,
            &mut rng,
            &budget,
            per_eval,
            &mut on_improve,
        );
        return Ok(());
    };
    if !g.load_schedule(&md, &alt.job_schedule) {
        return Ok(());
    }
    let alt_ms = match g.evaluate() {
        Some(ms) => ms,
        None => return Ok(()),
    };
    let mut alt_start = Start {
        state: GraphState::new(&md),
        best: alt_ms,
    };
    g.snapshot(&mut alt_start.state);
    if !on_improve(&g, alt_ms) {
        return Ok(());
    }

    // Turns alternate between the two starts until the budget is spent: a turn ends at its first
    // restart, and the next start resumes from its best schedule after a kick.
    let cfg_turn = TabuCfg {
        sterile_limit: 1,
        any_restart: true,
        ..cfg
    };
    let mut starts = [alt_start, greedy_start];
    let mut cur = 0usize;
    loop {
        let start = &mut starts[cur];
        let (stopped, ms) = tabu::tabu_search(
            &md,
            &mut g,
            &mut start.state,
            start.best,
            &cfg_turn,
            &mut rng,
            &budget,
            per_eval,
            &mut on_improve,
        );
        start.best = ms;
        if !stopped {
            break;
        }
        cur = 1 - cur;
        let state = &starts[cur].state;
        g.restore_from(state);
        tabu::kick(&mut g, state, &mut rng);
    }

    Ok(())
}

pub mod model {
//! Flat instance model: operations numbered job by job, eligible (machine, duration) pairs per
//! operation.
use tig_challenges::job_scheduling::Challenge;

pub struct Model {
    pub n_jobs: usize,
    pub n_machines: usize,
    pub n_ops: usize,
    pub job_base: Vec<u32>,
    pub job_len: Vec<u32>,
    pub opt_lo: Vec<u32>,
    pub opt_hi: Vec<u32>,
    pub opts: Vec<(u32, u32)>,
}

impl Model {
    pub fn build(challenge: &Challenge) -> Model {
        let n_jobs = challenge.num_jobs;
        let n_machines = challenge.num_machines;

        let mut job_prod = Vec::with_capacity(n_jobs);
        for (product, count) in challenge.jobs_per_product.iter().enumerate() {
            for _ in 0..*count {
                job_prod.push(product);
            }
        }

        let prod_len: Vec<usize> = challenge
            .product_processing_times
            .iter()
            .map(|ops| ops.len())
            .collect();

        let mut prod_opts: Vec<Vec<Vec<(u32, u32)>>> = Vec::with_capacity(prod_len.len());
        for ops in challenge.product_processing_times.iter() {
            let mut per_pos = Vec::with_capacity(ops.len());
            for op in ops.iter() {
                let mut e: Vec<(u32, u32)> = op.iter().map(|(&m, &t)| (m as u32, t)).collect();
                e.sort_unstable();
                per_pos.push(e);
            }
            prod_opts.push(per_pos);
        }

        let mut job_base = Vec::with_capacity(n_jobs);
        let mut job_len = Vec::with_capacity(n_jobs);
        let mut n_ops = 0usize;
        let mut opt_lo = Vec::new();
        let mut opt_hi = Vec::new();
        let mut opts: Vec<(u32, u32)> = Vec::new();

        for j in 0..n_jobs {
            let p = job_prod[j];
            job_base.push(n_ops as u32);
            job_len.push(prod_len[p] as u32);
            for k in 0..prod_len[p] {
                n_ops += 1;
                opt_lo.push(opts.len() as u32);
                opts.extend_from_slice(&prod_opts[p][k]);
                opt_hi.push(opts.len() as u32);
            }
        }

        Model {
            n_jobs,
            n_machines,
            n_ops,
            job_base,
            job_len,
            opt_lo,
            opt_hi,
            opts,
        }
    }

    #[inline]
    pub fn dur_on(&self, o: usize, m: u32) -> Option<u32> {
        let lo = self.opt_lo[o] as usize;
        let hi = self.opt_hi[o] as usize;
        for i in lo..hi {
            if self.opts[i].0 == m {
                return Some(self.opts[i].1);
            }
        }
        None
    }

    #[inline]
    pub fn n_opts(&self, o: usize) -> usize {
        (self.opt_hi[o] - self.opt_lo[o]) as usize
    }
}
}

pub mod graph {
//! Disjunctive graph with incremental heads and tails and a Pearce-Kelly topological order.
use super::model::Model;

pub const NONE: i32 = -1;

// Records mirror the flat arrays; link lanes use RNONE for an absent neighbour.
const R_HEAD: usize = 0;
const R_DUR: usize = 1;
const R_QEND: usize = 2;
const R_JN: usize = 3;
const R_JP: usize = 4;
const R_MN: usize = 5;
const R_MP: usize = 6;
const R_ORD: usize = 7;
// Neighbour rank lanes must agree with ord and the current links.
const R_JN_ORD: usize = 8;
const R_MN_ORD: usize = 9;
const R_JP_ORD: usize = 10;
const R_MP_ORD: usize = 11;
// rec[n_ops] is the zero sentinel; flat arrays exclude it, and sent_rank() uses an unswept spare
// dbits word.
const R_JN_S: usize = 12;
const R_JP_S: usize = 13;
const R_MN_S: usize = 14;
const R_MP_S: usize = 15;
const RNONE: u32 = u32::MAX;

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Rec([u32; 16]);

pub struct Graph {
    pub n_ops: usize,
    pub n_mach: usize,
    pub jnext: Vec<i32>,
    pub jprev: Vec<i32>,
    pub mach: Vec<u32>,
    pub dur: Vec<u32>,
    pub mseq: Vec<Vec<u32>>,
    pub mpos: Vec<u32>,

    indeg: Vec<u8>,
    indeg_full: Vec<u8>,
    // Each operation is pushed once when its in-degree reaches zero; stack and topo have n_ops
    // slots.
    topo: Vec<u32>,
    stack: Vec<u32>,
    pub head: Vec<u32>,
    // qend[o] == tail[o] + dur[o].
    pub qend: Vec<u32>,
    mnx: Vec<i32>,
    mpv: Vec<i32>,
    rec: Vec<Rec>,
    pub makespan: u32,

    // ord[o] is the rank of o; oat[ord[o]] == o.
    ord: Vec<u32>,
    oat: Vec<u32>,
    markf: Vec<u32>,
    markb: Vec<u32>,
    pk_stamp: u32,
    pk_stk: Vec<u32>,
    pk_df: Vec<u32>,
    pk_db: Vec<u32>,
    pk_pool: Vec<u32>,
    // Forward and backward seed lists have independent lifetimes and deduplication stamps.
    seed: Vec<u32>,
    seed_n: usize,
    sdup: Vec<u32>,
    sstamp: u32,
    qseed: Vec<u32>,
    qseed_n: usize,
    qdup: Vec<u32>,
    qstamp: u32,
    dbits: Vec<u64>,
    mbase: Vec<u32>,
    critbits: Vec<u64>,
    cstamp: Vec<u32>,
    cstamp_cur: u32,
    cqueue: Vec<u32>,
    job_last: Vec<u32>,
    full: bool,
    full_q: bool,
    cyc: bool,
    // chg_full invalidates the whole mirror; otherwise chg lists writes from tracked sweeps only.
    pub chg: Vec<u32>,
    pub chg_n: usize,
    pub chg_full: bool,
}

impl Graph {
    pub fn new(md: &Model) -> Graph {
        let n = md.n_ops;
        let mut jnext = vec![NONE; n];
        let mut jprev = vec![NONE; n];
        for j in 0..md.n_jobs {
            let base = md.job_base[j] as usize;
            let len = md.job_len[j] as usize;
            for k in 0..len {
                let o = base + k;
                if k + 1 < len {
                    jnext[o] = (o + 1) as i32;
                }
                if k > 0 {
                    jprev[o] = (o - 1) as i32;
                }
            }
        }
        let mut job_last: Vec<u32> = Vec::with_capacity(md.n_jobs);
        for o in 0..n {
            if jnext[o] == NONE {
                job_last.push(o as u32);
            }
        }
        let mut indeg_full = vec![1u8; n];
        for o in 0..n {
            if jprev[o] != NONE {
                indeg_full[o] = 2;
            }
        }
        let sent = n as u32;
        let srank = (((n + 63) / 64) * 64) as u32;
        let mut rec: Vec<Rec> = Vec::with_capacity(n + 1);
        for o in 0..n {
            rec.push(Rec([
                0,
                0,
                0,
                jnext[o] as u32,
                jprev[o] as u32,
                RNONE,
                RNONE,
                0,
                0,
                srank,
                0,
                srank,
                if jnext[o] == NONE {
                    sent
                } else {
                    jnext[o] as u32
                },
                if jprev[o] == NONE {
                    sent
                } else {
                    jprev[o] as u32
                },
                sent,
                sent,
            ]));
        }
        rec.push(Rec([
            0, 0, 0, RNONE, RNONE, RNONE, RNONE, srank, srank, srank, srank, srank, sent, sent,
            sent, sent,
        ]));
        Graph {
            n_ops: n,
            n_mach: md.n_machines,
            jnext,
            jprev,
            mach: vec![0; n],
            dur: vec![0; n],
            mseq: vec![Vec::new(); md.n_machines],
            mpos: vec![0; n],
            indeg: vec![0; n],
            indeg_full,
            topo: vec![0; n],
            stack: vec![0; n],
            head: vec![0; n],
            qend: vec![0; n],
            mnx: vec![NONE; n],
            mpv: vec![NONE; n],
            rec,
            makespan: u32::MAX,
            ord: vec![0; n],
            oat: vec![0; n],
            markf: vec![0; n],
            markb: vec![0; n],
            pk_stamp: 1,
            pk_stk: vec![0; n + 1],
            pk_df: vec![0; n + 1],
            pk_db: vec![0; n + 1],
            pk_pool: vec![0; 2 * n + 2],
            seed: vec![0; n + 1],
            seed_n: 0,
            sdup: vec![0; n],
            sstamp: 1,
            qseed: vec![0; n + 1],
            qseed_n: 0,
            qdup: vec![0; n],
            qstamp: 1,
            dbits: vec![0; (n + 63) / 64 + 1],
            mbase: vec![0; md.n_machines + 1],
            critbits: vec![0; (n + 63) / 64],
            cstamp: vec![0; n],
            cstamp_cur: 0,
            cqueue: vec![0; n],
            job_last,
            full: true,
            full_q: true,
            cyc: false,
            chg: vec![0; 4 * n + 4],
            chg_n: 0,
            chg_full: true,
        }
    }

    #[inline(always)]
    fn sent_rank(&self) -> u32 {
        (((self.n_ops + 63) / 64) * 64) as u32
    }

    // SAFETY: accessor arguments come from machine sequences or non-NONE route links, hence o <
    // n_ops.
    #[inline(always)]
    fn rec_rend(&self, o: usize) -> u32 {
        let r = unsafe { self.rec.get_unchecked(o) }.0;
        r[R_HEAD] + r[R_DUR]
    }
    #[inline(always)]
    fn rec_qend(&self, o: usize) -> u32 {
        unsafe { self.rec.get_unchecked(o) }.0[R_QEND]
    }
    #[inline(always)]
    fn rec_rp(&self, x: i32) -> u32 {
        if x == NONE {
            0
        } else {
            self.rec_rend(x as usize)
        }
    }
    #[inline(always)]
    fn rec_qp(&self, x: i32) -> u32 {
        if x == NONE {
            0
        } else {
            self.rec_qend(x as usize)
        }
    }
    #[inline(always)]
    pub fn ins_rend(&self, o: usize) -> u32 {
        self.rec_rend(o)
    }
    #[inline(always)]
    pub fn ins_qend(&self, o: usize) -> u32 {
        self.rec_qend(o)
    }
    #[inline(always)]
    pub fn ins_rp(&self, x: i32) -> u32 {
        self.rec_rp(x)
    }
    #[inline(always)]
    pub fn ins_qp(&self, x: i32) -> u32 {
        self.rec_qp(x)
    }

    #[inline]
    pub fn mnext(&self, o: usize) -> i32 {
        self.mnx[o]
    }

    #[inline]
    pub fn mprev(&self, o: usize) -> i32 {
        self.mpv[o]
    }

    pub fn load_schedule(&mut self, md: &Model, sched: &[Vec<(usize, u32)>]) -> bool {
        self.full = true;
        self.full_q = true;
        self.cyc = false;
        if sched.len() != md.n_jobs {
            return false;
        }
        for s in self.mseq.iter_mut() {
            s.clear();
        }
        let mut tmp: Vec<Vec<(u32, u32)>> = vec![Vec::new(); self.n_mach];
        for j in 0..md.n_jobs {
            let base = md.job_base[j] as usize;
            let len = md.job_len[j] as usize;
            if sched[j].len() != len {
                return false;
            }
            for k in 0..len {
                let o = base + k;
                let (m, st) = sched[j][k];
                if m >= self.n_mach {
                    return false;
                }
                let d = match md.dur_on(o, m as u32) {
                    Some(d) => d,
                    None => return false,
                };
                self.mach[o] = m as u32;
                self.dur[o] = d;
                self.rec[o].0[R_DUR] = d;
                tmp[m].push((st, o as u32));
            }
        }
        for m in 0..self.n_mach {
            tmp[m].sort_unstable();
            for (i, &(_, o)) in tmp[m].iter().enumerate() {
                self.mseq[m].push(o);
                self.mpos[o as usize] = i as u32;
            }
        }
        self.relink_all();
        true
    }

    // Machine links match the predecessor and successor in mseq, or NONE at the ends.
    #[inline]
    fn relink(&mut self, m: usize, lo: usize, hi: usize) {
        let len = self.mseq[m].len();
        if len == 0 {
            return;
        }
        let hi = if hi >= len { len - 1 } else { hi };
        let mut i = lo;
        while i <= hi {
            let o = self.mseq[m][i] as usize;
            let nx = if i + 1 < len {
                self.mseq[m][i + 1] as i32
            } else {
                NONE
            };
            let pv = if i > 0 {
                self.mseq[m][i - 1] as i32
            } else {
                NONE
            };
            self.mnx[o] = nx;
            self.mpv[o] = pv;
            self.rec[o].0[R_MN] = nx as u32;
            self.rec[o].0[R_MP] = pv as u32;
            let srank = self.sent_rank();
            let sent = self.n_ops as u32;
            self.rec[o].0[R_MN_ORD] = if nx == NONE {
                srank
            } else {
                self.ord[nx as usize]
            };
            self.rec[o].0[R_MP_ORD] = if pv == NONE {
                srank
            } else {
                self.ord[pv as usize]
            };
            self.rec[o].0[R_MN_S] = if nx == NONE { sent } else { nx as u32 };
            self.rec[o].0[R_MP_S] = if pv == NONE { sent } else { pv as u32 };
            // Every changed machine neighbour must seed its operation, for both added and removed
            // arcs.
            self.touch(o);
            i += 1;
        }
    }

    fn relink_all(&mut self) {
        for m in 0..self.n_mach {
            let len = self.mseq[m].len();
            if len > 0 {
                self.relink(m, 0, len - 1);
            }
        }
    }

    // Checks machine links when debug_assertions is enabled.
    #[cfg(debug_assertions)]
    fn links_consistent(&self) -> bool {
        let n = self.n_ops;
        let mut mnx = vec![NONE; n];
        let mut mpv = vec![NONE; n];
        let mut seen = vec![false; n];
        for m in 0..self.n_mach {
            let s = &self.mseq[m];
            for i in 0..s.len() {
                let o = s[i] as usize;
                if seen[o] || self.mpos[o] as usize != i || self.mach[o] as usize != m {
                    return false;
                }
                seen[o] = true;
                mnx[o] = if i + 1 < s.len() {
                    s[i + 1] as i32
                } else {
                    NONE
                };
                mpv[o] = if i > 0 { s[i - 1] as i32 } else { NONE };
            }
        }
        seen.iter().all(|&b| b) && mnx == self.mnx && mpv == self.mpv
    }

    #[cfg(not(debug_assertions))]
    #[inline(always)]
    fn links_consistent(&self) -> bool {
        true
    }

    // Checks the record mirror when debug_assertions is enabled.
    #[cfg(debug_assertions)]
    fn debug_check_rec(&self) -> bool {
        let srank = self.sent_rank();
        let sent = self.n_ops as u32;
        for o in 0..self.n_ops {
            let r = &self.rec[o].0;
            if r[R_HEAD] != self.head[o]
                || r[R_DUR] != self.dur[o]
                || r[R_QEND] != self.qend[o]
                || r[R_JN] != self.jnext[o] as u32
                || r[R_JP] != self.jprev[o] as u32
                || r[R_MN] != self.mnx[o] as u32
                || r[R_MP] != self.mpv[o] as u32
                || r[R_ORD] != self.ord[o]
                || r[R_JN_ORD]
                    != (if self.jnext[o] == NONE {
                        srank
                    } else {
                        self.ord[self.jnext[o] as usize]
                    })
                || r[R_MN_ORD]
                    != (if self.mnx[o] == NONE {
                        srank
                    } else {
                        self.ord[self.mnx[o] as usize]
                    })
                || r[R_JP_ORD]
                    != (if self.jprev[o] == NONE {
                        srank
                    } else {
                        self.ord[self.jprev[o] as usize]
                    })
                || r[R_MP_ORD]
                    != (if self.mpv[o] == NONE {
                        srank
                    } else {
                        self.ord[self.mpv[o] as usize]
                    })
                || r[R_JN_S]
                    != (if self.jnext[o] == NONE {
                        sent
                    } else {
                        self.jnext[o] as u32
                    })
                || r[R_JP_S]
                    != (if self.jprev[o] == NONE {
                        sent
                    } else {
                        self.jprev[o] as u32
                    })
                || r[R_MN_S]
                    != (if self.mnx[o] == NONE {
                        sent
                    } else {
                        self.mnx[o] as u32
                    })
                || r[R_MP_S]
                    != (if self.mpv[o] == NONE {
                        sent
                    } else {
                        self.mpv[o] as u32
                    })
            {
                return false;
            }
        }
        true
    }

    #[cfg(not(debug_assertions))]
    #[inline(always)]
    fn debug_check_rec(&self) -> bool {
        true
    }

    pub fn evaluate(&mut self) -> Option<u32> {
        self.evaluate_impl::<false>()
    }

    // Tracked evaluation sets chg_full on a full pass.
    pub fn evaluate_tracked(&mut self) -> Option<u32> {
        self.evaluate_impl::<true>()
    }

    #[inline]
    fn evaluate_impl<const TRACK: bool>(&mut self) -> Option<u32> {
        if self.full {
            if TRACK {
                self.chg_full = true;
            }
            return self.evaluate_full();
        }
        if self.cyc {
            // Pending seeds survive a rejected move and are combined with the undo seeds.
            self.cyc = false;
            self.makespan = u32::MAX;
            return None;
        }
        self.eval_incr::<TRACK>()
    }

    fn evaluate_full(&mut self) -> Option<u32> {
        let n = self.n_ops;
        debug_assert!(self.links_consistent(), "mnx/mpv out of sync with mseq");
        self.head.fill(0);
        for r in self.rec.iter_mut() {
            r.0[R_HEAD] = 0;
        }
        self.indeg.copy_from_slice(&self.indeg_full);
        let mut sl = 0usize;
        for m in 0..self.n_mach {
            if self.mseq[m].is_empty() {
                continue;
            }
            let o = self.mseq[m][0] as usize;
            self.indeg[o] -= 1;
            if self.indeg[o] == 0 {
                self.stack[sl] = o as u32;
                sl += 1;
            }
        }
        let mut tl = 0usize;
        let mut ms = 0u32;
        while sl > 0 {
            sl -= 1;
            let o = self.stack[sl] as usize;
            self.topo[tl] = o as u32;
            tl += 1;
            let f = self.head[o] + self.dur[o];
            if f > ms {
                ms = f;
            }
            let jn = self.jnext[o];
            if jn != NONE {
                let jn = jn as usize;
                if f > self.head[jn] {
                    self.head[jn] = f;
                    self.rec[jn].0[R_HEAD] = f;
                }
                self.indeg[jn] -= 1;
                if self.indeg[jn] == 0 {
                    self.stack[sl] = jn as u32;
                    sl += 1;
                }
            }
            let mn = self.mnx[o];
            if mn != NONE {
                let mn = mn as usize;
                if f > self.head[mn] {
                    self.head[mn] = f;
                    self.rec[mn].0[R_HEAD] = f;
                }
                self.indeg[mn] -= 1;
                if self.indeg[mn] == 0 {
                    self.stack[sl] = mn as u32;
                    sl += 1;
                }
            }
        }
        if tl != n {
            self.makespan = u32::MAX;
            return None;
        }
        for i in 0..tl {
            let o = self.topo[i] as usize;
            self.ord[o] = i as u32;
            self.rec[o].0[R_ORD] = i as u32;
            self.oat[i] = o as u32;
        }
        let srank = self.sent_rank();
        for o in 0..n {
            let r = &mut self.rec[o].0;
            r[R_JN_ORD] = if r[R_JN] == RNONE {
                srank
            } else {
                self.ord[r[R_JN] as usize]
            };
            r[R_MN_ORD] = if r[R_MN] == RNONE {
                srank
            } else {
                self.ord[r[R_MN] as usize]
            };
            r[R_JP_ORD] = if r[R_JP] == RNONE {
                srank
            } else {
                self.ord[r[R_JP] as usize]
            };
            r[R_MP_ORD] = if r[R_MP] == RNONE {
                srank
            } else {
                self.ord[r[R_MP] as usize]
            };
        }
        debug_assert!(
            self.debug_check_rec(),
            "rec out of sync after evaluate_full"
        );
        self.full = false;
        self.full_q = true;
        self.cyc = false;
        self.seed_n = 0;
        self.sstamp = self.sstamp.wrapping_add(1);
        self.makespan = ms;
        Some(ms)
    }

    pub fn compute_tails(&mut self) {
        self.compute_tails_impl::<false>()
    }

    pub fn compute_tails_tracked(&mut self) {
        self.compute_tails_impl::<true>()
    }

    #[inline]
    fn compute_tails_impl<const TRACK: bool>(&mut self) {
        if self.full_q {
            if TRACK {
                self.chg_full = true;
            }
            self.tails_full();
            return;
        }
        if self.qseed_n == 0 {
            return;
        }
        let nw = (self.n_ops + 63) / 64;
        self.dbits[..nw].fill(0);
        let mut hi = 0usize;
        for k in 0..self.qseed_n {
            let o = self.qseed[k] as usize;
            let r = self.rec[o].0[R_ORD] as usize;
            self.dbits[r >> 6] |= 1u64 << (r & 63);
            if r > hi {
                hi = r;
            }
        }
        self.qseed_n = 0;
        Self::bump(&mut self.qstamp, &mut self.qdup);
        // SAFETY: oat holds operations; _S links index rec including its sentinel, and _ORD ranks
        // index allocated dbits words.
        let mut wi = (hi >> 6) as isize;
        while wi >= 0 {
            let w = self.dbits[wi as usize];
            if w == 0 {
                wi -= 1;
                continue;
            }
            let b = 63 - w.leading_zeros() as usize;
            self.dbits[wi as usize] = w & !(1u64 << b);
            let o = unsafe { *self.oat.get_unchecked(((wi as usize) << 6) | b) } as usize;
            let ro = unsafe { self.rec.get_unchecked(o) }.0;
            let t = unsafe { self.rec.get_unchecked(ro[R_JN_S] as usize) }.0[R_QEND]
                .max(unsafe { self.rec.get_unchecked(ro[R_MN_S] as usize) }.0[R_QEND]);
            let v = t + ro[R_DUR];
            if v != ro[R_QEND] {
                self.qend[o] = v;
                unsafe { self.rec.get_unchecked_mut(o) }.0[R_QEND] = v;
                if TRACK {
                    self.chg[self.chg_n] = o as u32;
                    self.chg_n += 1;
                }
                let r = ro[R_JP_ORD] as usize;
                unsafe {
                    *self.dbits.get_unchecked_mut(r >> 6) |= 1u64 << (r & 63);
                }
                let r = ro[R_MP_ORD] as usize;
                unsafe {
                    *self.dbits.get_unchecked_mut(r >> 6) |= 1u64 << (r & 63);
                }
            }
        }
        debug_assert!(
            self.debug_check_rec(),
            "rec out of sync after compute_tails"
        );
    }

    fn tails_full(&mut self) {
        let n = self.n_ops;
        let mut i = n;
        while i > 0 {
            i -= 1;
            let o = self.oat[i] as usize;
            let mut t = 0u32;
            let jn = self.jnext[o];
            if jn != NONE {
                let v = self.qend[jn as usize];
                if v > t {
                    t = v;
                }
            }
            let mn = self.mnx[o];
            if mn != NONE {
                let v = self.qend[mn as usize];
                if v > t {
                    t = v;
                }
            }
            let v = t + self.dur[o];
            self.qend[o] = v;
            self.rec[o].0[R_QEND] = v;
        }
        self.full_q = false;
        self.qseed_n = 0;
        Self::bump(&mut self.qstamp, &mut self.qdup);
    }

    #[inline]
    pub fn is_critical(&self, o: usize) -> bool {
        self.head[o] + self.qend[o] == self.makespan
    }

    pub fn critical_blocks_flex(
        &mut self,
        md: &Model,
        out: &mut Vec<(u32, u32, u32)>,
        flex: &mut Vec<u32>,
        want_flex: bool,
    ) {
        out.clear();
        flex.clear();
        if self.makespan == u32::MAX {
            self.blocks_by_scan(md, out, flex, want_flex);
            return;
        }
        if !self.mark_critical() {
            out.clear();
            flex.clear();
            self.blocks_by_scan(md, out, flex, want_flex);
            return;
        }
        self.blocks_from_bits(md, out, flex, want_flex);
    }

    #[inline]
    fn crit_cap(&self) -> usize {
        self.n_ops / 8
    }

    fn mark_critical(&mut self) -> bool {
        let mut acc = 0u32;
        for m in 0..self.n_mach {
            self.mbase[m] = acc;
            acc += self.mseq[m].len() as u32;
        }
        self.mbase[self.n_mach] = acc;
        let nw = (self.n_ops + 63) / 64;
        self.critbits[..nw].fill(0);

        self.cstamp_cur = self.cstamp_cur.wrapping_add(1);
        if self.cstamp_cur == 0 {
            self.cstamp.fill(0);
            self.cstamp_cur = 1;
        }
        let st = self.cstamp_cur;
        let ms = self.makespan;
        let cap = self.crit_cap();

        let mut ql = 0usize;
        for m in 0..self.n_mach {
            if self.mseq[m].is_empty() {
                continue;
            }
            let o = self.mseq[m][0] as usize;
            if self.jprev[o] == NONE && self.qend[o] == ms {
                self.cstamp[o] = st;
                self.cqueue[ql] = o as u32;
                ql += 1;
                let slot = self.mbase[m] as usize;
                self.critbits[slot >> 6] |= 1u64 << (slot & 63);
            }
        }

        // SAFETY: non-RNONE successors are operations; stamps enqueue each once, and machine slots
        // are below n_ops.
        let mut hd = 0usize;
        while hd < ql {
            let o = unsafe { *self.cqueue.get_unchecked(hd) } as usize;
            hd += 1;
            let ro = unsafe { self.rec.get_unchecked(o) }.0;
            let jn = ro[R_JN];
            if jn != RNONE {
                let s = jn as usize;
                let rs = unsafe { self.rec.get_unchecked(s) }.0;
                if unsafe { *self.cstamp.get_unchecked(s) } != st
                    && rs[R_HEAD] + rs[R_QEND] == ms
                {
                    if ql >= cap {
                        return false;
                    }
                    unsafe {
                        *self.cstamp.get_unchecked_mut(s) = st;
                    }
                    unsafe {
                        *self.cqueue.get_unchecked_mut(ql) = s as u32;
                    }
                    ql += 1;
                    let slot = unsafe {
                        *self
                            .mbase
                            .get_unchecked(*self.mach.get_unchecked(s) as usize)
                    } as usize
                        + unsafe { *self.mpos.get_unchecked(s) } as usize;
                    unsafe {
                        *self.critbits.get_unchecked_mut(slot >> 6) |= 1u64 << (slot & 63);
                    }
                }
            }
            let mn = ro[R_MN];
            if mn != RNONE {
                let s = mn as usize;
                let rs = unsafe { self.rec.get_unchecked(s) }.0;
                if unsafe { *self.cstamp.get_unchecked(s) } != st
                    && rs[R_HEAD] + rs[R_QEND] == ms
                {
                    if ql >= cap {
                        return false;
                    }
                    unsafe {
                        *self.cstamp.get_unchecked_mut(s) = st;
                    }
                    unsafe {
                        *self.cqueue.get_unchecked_mut(ql) = s as u32;
                    }
                    ql += 1;
                    let slot = unsafe {
                        *self
                            .mbase
                            .get_unchecked(*self.mach.get_unchecked(s) as usize)
                    } as usize
                        + unsafe { *self.mpos.get_unchecked(s) } as usize;
                    unsafe {
                        *self.critbits.get_unchecked_mut(slot >> 6) |= 1u64 << (slot & 63);
                    }
                }
            }
        }
        true
    }

    fn blocks_from_bits(
        &self,
        md: &Model,
        out: &mut Vec<(u32, u32, u32)>,
        flex: &mut Vec<u32>,
        want_flex: bool,
    ) {
        let nw = (self.n_ops + 63) / 64;
        let mut m = 0usize;
        let mut mend = self.mbase[1] as usize;
        let mut act = false;
        let (mut rm, mut ri, mut rj) = (0usize, 0usize, 0usize);
        let mut rend = 0usize; // operation at the end of the current block
        for wi in 0..nw {
            let mut w = self.critbits[wi];
            while w != 0 {
                let slot = (wi << 6) + w.trailing_zeros() as usize;
                w &= w - 1;
                while slot >= mend {
                    m += 1;
                    mend = self.mbase[m + 1] as usize;
                }
                // SAFETY: a set slot belongs to machine m and indexes its sequence; the stored
                // operation is below n_ops.
                let pos = slot - unsafe { *self.mbase.get_unchecked(m) } as usize;
                let o = unsafe { *self.mseq.get_unchecked(m).get_unchecked(pos) } as usize;
                let ro = unsafe { self.rec.get_unchecked(o) }.0;
                if act
                    && m == rm
                    && pos == rj + 1
                    && ro[R_HEAD]
                        == unsafe { self.rec.get_unchecked(rend) }.0[R_HEAD]
                            + unsafe { self.rec.get_unchecked(rend) }.0[R_DUR]
                {
                    if want_flex && md.n_opts(o) > 1 {
                        flex.push(o as u32);
                    }
                    rj = pos;
                    rend = o;
                } else {
                    if act && rj > ri {
                        out.push((rm as u32, ri as u32, rj as u32));
                    }
                    if want_flex && md.n_opts(o) > 1 {
                        flex.push(o as u32);
                    }
                    rm = m;
                    ri = pos;
                    rj = pos;
                    rend = o;
                    act = true;
                }
            }
        }
        if act && rj > ri {
            out.push((rm as u32, ri as u32, rj as u32));
        }
    }

    fn blocks_by_scan(
        &self,
        md: &Model,
        out: &mut Vec<(u32, u32, u32)>,
        flex: &mut Vec<u32>,
        want_flex: bool,
    ) {
        for m in 0..self.n_mach {
            let s = &self.mseq[m];
            let len = s.len();
            let mut i = 0usize;
            while i < len {
                let a = s[i] as usize;
                if !self.is_critical(a) {
                    i += 1;
                    continue;
                }
                if want_flex && md.n_opts(a) > 1 {
                    flex.push(a as u32);
                }
                let mut j = i;
                while j + 1 < len {
                    let u = s[j] as usize;
                    let v = s[j + 1] as usize;
                    if self.is_critical(v) && self.head[v] == self.head[u] + self.dur[u] {
                        if want_flex && md.n_opts(v) > 1 {
                            flex.push(v as u32);
                        }
                        j += 1;
                    } else {
                        break;
                    }
                }
                if j > i {
                    out.push((m as u32, i as u32, j as u32));
                }
                i = j + 1;
            }
        }
    }

    pub fn relocate_within(&mut self, o: usize, at: usize) -> usize {
        let m = self.mach[o] as usize;
        let p = self.mpos[o] as usize;
        if p == at {
            return p;
        }
        let v = self.mseq[m].remove(p);
        let at = at.min(self.mseq[m].len());
        self.mseq[m].insert(at, v);
        let lo = p.min(at);
        let hi = p.max(at);
        for i in lo..=hi {
            let x = self.mseq[m][i] as usize;
            self.mpos[x] = i as u32;
        }
        self.relink(m, lo.saturating_sub(1), hi + 1);
        self.pk_after_relocate(m, o, p);
        p
    }

    #[inline]
    pub fn swap_adjacent(&mut self, u: u32, v: u32) {
        let m = self.mach[u as usize] as usize;
        let pu = self.mpos[u as usize] as usize;
        let pv = self.mpos[v as usize] as usize;
        self.mseq[m].swap(pu, pv);
        self.mpos[u as usize] = pv as u32;
        self.mpos[v as usize] = pu as u32;
        let (lo, hi) = if pu < pv { (pu, pv) } else { (pv, pu) };
        self.relink(m, lo.saturating_sub(1), hi + 1);
        self.pk_after_span(m, lo, hi);
    }

    fn detach(&mut self, o: usize) {
        let m = self.mach[o] as usize;
        let p = self.mpos[o] as usize;
        self.mseq[m].remove(p);
        for i in p..self.mseq[m].len() {
            let x = self.mseq[m][i] as usize;
            self.mpos[x] = i as u32;
        }
        self.relink(m, p.saturating_sub(1), p);
        self.mnx[o] = NONE;
        self.mpv[o] = NONE;
        self.rec[o].0[R_MN] = RNONE;
        self.rec[o].0[R_MP] = RNONE;
        let srank = self.sent_rank();
        self.rec[o].0[R_MN_ORD] = srank;
        self.rec[o].0[R_MP_ORD] = srank;
        self.rec[o].0[R_MN_S] = self.n_ops as u32;
        self.rec[o].0[R_MP_S] = self.n_ops as u32;
        self.touch(o);
        let jn = self.jnext[o];
        if jn != NONE {
            self.touch(jn as usize);
        }
    }

    fn attach(&mut self, o: usize, m: u32, at: usize, dur: u32) {
        let mm = m as usize;
        let at = at.min(self.mseq[mm].len());
        self.mseq[mm].insert(at, o as u32);
        self.mach[o] = m;
        // A duration change must also seed the route successor, since head[o] excludes dur[o].
        let dchg = self.dur[o] != dur;
        self.dur[o] = dur;
        self.rec[o].0[R_DUR] = dur;
        for i in at..self.mseq[mm].len() {
            let x = self.mseq[mm][i] as usize;
            self.mpos[x] = i as u32;
        }
        self.relink(mm, at.saturating_sub(1), at + 1);
        if dchg {
            let jn = self.jnext[o];
            if jn != NONE {
                self.touch(jn as usize);
            }
        }
        self.pk_after_attach(mm, o);
    }

    pub fn reassign_at(&mut self, o: usize, m: u32, at: usize, dur: u32) -> (u32, usize, u32) {
        let old = (self.mach[o], self.mpos[o] as usize, self.dur[o]);
        self.detach(o);
        self.attach(o, m, at, dur);
        old
    }

    pub fn undo_reassign(&mut self, o: usize, old: (u32, usize, u32)) {
        self.detach(o);
        self.attach(o, old.0, old.1, old.2);
    }

    pub fn to_schedule(&self, md: &Model) -> Vec<Vec<(usize, u32)>> {
        let mut out: Vec<Vec<(usize, u32)>> = Vec::with_capacity(md.n_jobs);
        for j in 0..md.n_jobs {
            let base = md.job_base[j] as usize;
            let len = md.job_len[j] as usize;
            let mut v = Vec::with_capacity(len);
            for k in 0..len {
                let o = base + k;
                v.push((self.mach[o] as usize, self.head[o]));
            }
            out.push(v);
        }
        out
    }

    pub fn restore_from(&mut self, src: &GraphState) {
        self.mach.copy_from_slice(&src.mach);
        self.dur.copy_from_slice(&src.dur);
        self.mpos.copy_from_slice(&src.mpos);
        self.mnx.copy_from_slice(&src.mnx);
        self.mpv.copy_from_slice(&src.mpv);
        // Restore copies the machine state; the next evaluation rebuilds head, qend and ord.
        let sent = self.n_ops as u32;
        for o in 0..self.n_ops {
            let r = &mut self.rec[o].0;
            r[R_DUR] = src.dur[o];
            r[R_MN] = src.mnx[o] as u32;
            r[R_MP] = src.mpv[o] as u32;
            r[R_MN_S] = if src.mnx[o] == NONE {
                sent
            } else {
                src.mnx[o] as u32
            };
            r[R_MP_S] = if src.mpv[o] == NONE {
                sent
            } else {
                src.mpv[o] as u32
            };
        }
        for m in 0..self.n_mach {
            self.mseq[m].clear();
            self.mseq[m].extend_from_slice(&src.mseq[m]);
        }
        self.full = true;
        self.full_q = true;
        self.cyc = false;
    }

    pub fn snapshot(&self, dst: &mut GraphState) {
        dst.mach.copy_from_slice(&self.mach);
        dst.dur.copy_from_slice(&self.dur);
        dst.mpos.copy_from_slice(&self.mpos);
        dst.mnx.copy_from_slice(&self.mnx);
        dst.mpv.copy_from_slice(&self.mpv);
        for m in 0..self.n_mach {
            dst.mseq[m].clear();
            dst.mseq[m].extend_from_slice(&self.mseq[m]);
        }
    }
}

pub struct GraphState {
    pub mach: Vec<u32>,
    pub dur: Vec<u32>,
    pub mpos: Vec<u32>,
    pub mnx: Vec<i32>,
    pub mpv: Vec<i32>,
    pub mseq: Vec<Vec<u32>>,
}

impl GraphState {
    pub fn new(md: &Model) -> GraphState {
        GraphState {
            mach: vec![0; md.n_ops],
            dur: vec![0; md.n_ops],
            mpos: vec![0; md.n_ops],
            mnx: vec![NONE; md.n_ops],
            mpv: vec![NONE; md.n_ops],
            mseq: vec![Vec::new(); md.n_machines],
        }
    }
}

impl Graph {
    #[inline]
    fn bump(stamp: &mut u32, marks: &mut [u32]) {
        let v = stamp.wrapping_add(1);
        if v == 0 {
            for m in marks.iter_mut() {
                *m = 0;
            }
            *stamp = 1;
        } else {
            *stamp = v;
        }
    }

    // Seed overflow requires a full recomputation.
    #[inline]
    fn touch(&mut self, o: usize) {
        if self.full {
            return;
        }
        if self.sdup[o] != self.sstamp {
            self.sdup[o] = self.sstamp;
            if self.seed_n < self.n_ops {
                self.seed[self.seed_n] = o as u32;
                self.seed_n += 1;
            } else {
                self.full = true;
                return;
            }
        }
        if self.qdup[o] != self.qstamp {
            self.qdup[o] = self.qstamp;
            if self.qseed_n < self.n_ops {
                self.qseed[self.qseed_n] = o as u32;
                self.qseed_n += 1;
            } else {
                self.full_q = true;
            }
        }
    }

    fn eval_incr<const TRACK: bool>(&mut self) -> Option<u32> {
        let n = self.n_ops;
        if self.seed_n == 0 {
            return Some(self.makespan);
        }
        let nw = (n + 63) / 64;
        self.dbits[..nw].fill(0);
        let mut lo = usize::MAX;
        for k in 0..self.seed_n {
            let o = self.seed[k] as usize;
            let r = self.rec[o].0[R_ORD] as usize;
            self.dbits[r >> 6] |= 1u64 << (r & 63);
            if r < lo {
                lo = r;
            }
        }
        self.seed_n = 0;
        Self::bump(&mut self.sstamp, &mut self.sdup);
        let mut wi = lo >> 6;
        while wi < nw {
            let w = self.dbits[wi];
            if w == 0 {
                wi += 1;
                continue;
            }
            let b = w.trailing_zeros() as usize;
            self.dbits[wi] = w & !(1u64 << b);
            // SAFETY: oat holds operations; _S links index rec including its sentinel, and _ORD
            // ranks index allocated dbits words.
            let o = unsafe { *self.oat.get_unchecked((wi << 6) | b) } as usize;
            let ro = unsafe { self.rec.get_unchecked(o) }.0;
            let rp = unsafe { self.rec.get_unchecked(ro[R_JP_S] as usize) }.0;
            let rq = unsafe { self.rec.get_unchecked(ro[R_MP_S] as usize) }.0;
            let h = (rp[R_HEAD] + rp[R_DUR]).max(rq[R_HEAD] + rq[R_DUR]);
            // Duration changes seed successors separately because head[o] excludes dur[o].
            if h != ro[R_HEAD] {
                self.head[o] = h;
                unsafe { self.rec.get_unchecked_mut(o) }.0[R_HEAD] = h;
                if TRACK {
                    self.chg[self.chg_n] = o as u32;
                    self.chg_n += 1;
                }
                let r = ro[R_JN_ORD] as usize;
                unsafe {
                    *self.dbits.get_unchecked_mut(r >> 6) |= 1u64 << (r & 63);
                }
                let r = ro[R_MN_ORD] as usize;
                unsafe {
                    *self.dbits.get_unchecked_mut(r >> 6) |= 1u64 << (r & 63);
                }
            }
        }
        let mut ms = 0u32;
        for k in 0..self.job_last.len() {
            let o = self.job_last[k] as usize;
            let ro = self.rec[o].0;
            if ro[R_MN] == RNONE {
                let f = ro[R_HEAD] + ro[R_DUR];
                if f > ms {
                    ms = f;
                }
            }
        }
        self.makespan = ms;
        debug_assert!(self.debug_check_rec(), "rec out of sync after eval_incr");
        Some(ms)
    }

    // Pearce-Kelly requires the graph without the inserted arc to respect ord.
    #[inline]
    fn pk_arc(&mut self, x: i32, y: i32) {
        if x == NONE || y == NONE {
            return;
        }
        let (x, y) = (x as usize, y as usize);
        if self.pk_stamp == u32::MAX {
            for m in self.markf.iter_mut() {
                *m = 0;
            }
            for m in self.markb.iter_mut() {
                *m = 0;
            }
            self.pk_stamp = 0;
        }
        self.pk_stamp += 1;
        let cycle = pk_insert_arc(
            &mut self.ord,
            &mut self.oat,
            &self.jnext,
            &self.mnx,
            &self.jprev,
            &self.mpv,
            &mut self.markf,
            &mut self.markb,
            self.pk_stamp,
            &mut self.pk_stk,
            &mut self.pk_df,
            &mut self.pk_db,
            &mut self.pk_pool,
            &mut self.rec,
            x,
            y,
        );
        if cycle {
            self.cyc = true;
        }
    }

    fn pk_after_span(&mut self, m: usize, lo: usize, hi: usize) {
        if self.full {
            return;
        }
        let len = self.mseq[m].len();
        let mut prev = if lo > 0 {
            self.mseq[m][lo - 1] as i32
        } else {
            NONE
        };
        let mut i = lo;
        while i <= hi && i < len {
            let cur = self.mseq[m][i] as i32;
            self.pk_arc(prev, cur);
            prev = cur;
            i += 1;
        }
        if hi + 1 < len {
            let nxt = self.mseq[m][hi + 1] as i32;
            self.pk_arc(prev, nxt);
        }
    }

    fn pk_after_relocate(&mut self, m: usize, o: usize, p: usize) {
        if self.full {
            return;
        }
        let ln = self.mseq[m].len();
        let ap = self.mpos[o] as usize;
        let q = if ap > p { p } else { p + 1 };
        if q >= 1 && q < ln {
            let a = self.mseq[m][q - 1] as i32;
            let b = self.mseq[m][q] as i32;
            self.pk_arc(a, b);
        }
        if ap >= 1 {
            let c = self.mseq[m][ap - 1] as i32;
            self.pk_arc(c, o as i32);
        }
        if ap + 1 < ln {
            let d = self.mseq[m][ap + 1] as i32;
            self.pk_arc(o as i32, d);
        }
    }

    fn pk_after_attach(&mut self, m: usize, o: usize) {
        if self.full {
            return;
        }
        let ln = self.mseq[m].len();
        let ap = self.mpos[o] as usize;
        if ap >= 1 {
            let c = self.mseq[m][ap - 1] as i32;
            self.pk_arc(c, o as i32);
        }
        if ap + 1 < ln {
            let d = self.mseq[m][ap + 1] as i32;
            self.pk_arc(o as i32, d);
        }
    }
}

// Pearce-Kelly (2007); returns true on a cycle without modifying ord. Ranks form a permutation, so
// cone sorting has no equal keys.
#[allow(clippy::too_many_arguments)]
fn pk_insert_arc(
    ord: &mut [u32],
    oat: &mut [u32],
    jn: &[i32],
    mn: &[i32],
    jp: &[i32],
    mp: &[i32],
    markf: &mut [u32],
    markb: &mut [u32],
    stamp: u32,
    stk: &mut [u32],
    df: &mut [u32],
    db: &mut [u32],
    pool: &mut [u32],
    rec: &mut [Rec],
    x: usize,
    y: usize,
) -> bool {
    // SAFETY: nodes are below n_ops; marks admit each once per cone, within the allocated cone and
    // stack buffers.
    let (ox, oy) = unsafe { (*ord.get_unchecked(x), *ord.get_unchecked(y)) };
    if ox < oy {
        return false;
    }
    let lb = oy;
    let ub = ox;

    let mut dfn = 0usize;
    let mut sl = 0usize;
    unsafe {
        *markf.get_unchecked_mut(y) = stamp;
    }
    unsafe {
        *stk.get_unchecked_mut(sl) = y as u32;
    }
    sl += 1;
    while sl > 0 {
        sl -= 1;
        let v = unsafe { *stk.get_unchecked(sl) } as usize;
        unsafe {
            *df.get_unchecked_mut(dfn) = v as u32;
        }
        dfn += 1;
        let a = unsafe { *jn.get_unchecked(v) };
        if a != NONE {
            let w = a as usize;
            if (unsafe { *ord.get_unchecked(w) }) == ub {
                return true;
            }
            if unsafe { *markf.get_unchecked(w) } != stamp
                && (unsafe { *ord.get_unchecked(w) }) < ub
            {
                unsafe {
                    *markf.get_unchecked_mut(w) = stamp;
                }
                unsafe {
                    *stk.get_unchecked_mut(sl) = w as u32;
                }
                sl += 1;
            }
        }
        let b = unsafe { *mn.get_unchecked(v) };
        if b != NONE {
            let w = b as usize;
            if (unsafe { *ord.get_unchecked(w) }) == ub {
                return true;
            }
            if unsafe { *markf.get_unchecked(w) } != stamp
                && (unsafe { *ord.get_unchecked(w) }) < ub
            {
                unsafe {
                    *markf.get_unchecked_mut(w) = stamp;
                }
                unsafe {
                    *stk.get_unchecked_mut(sl) = w as u32;
                }
                sl += 1;
            }
        }
    }

    let mut dbn = 0usize;
    let mut sl = 0usize;
    unsafe {
        *markb.get_unchecked_mut(x) = stamp;
    }
    unsafe {
        *stk.get_unchecked_mut(sl) = x as u32;
    }
    sl += 1;
    while sl > 0 {
        sl -= 1;
        let v = unsafe { *stk.get_unchecked(sl) } as usize;
        unsafe {
            *db.get_unchecked_mut(dbn) = v as u32;
        }
        dbn += 1;
        let a = unsafe { *jp.get_unchecked(v) };
        if a != NONE {
            let w = a as usize;
            if unsafe { *markb.get_unchecked(w) } != stamp
                && (unsafe { *ord.get_unchecked(w) }) > lb
            {
                unsafe {
                    *markb.get_unchecked_mut(w) = stamp;
                }
                unsafe {
                    *stk.get_unchecked_mut(sl) = w as u32;
                }
                sl += 1;
            }
        }
        let b = unsafe { *mp.get_unchecked(v) };
        if b != NONE {
            let w = b as usize;
            if unsafe { *markb.get_unchecked(w) } != stamp
                && (unsafe { *ord.get_unchecked(w) }) > lb
            {
                unsafe {
                    *markb.get_unchecked_mut(w) = stamp;
                }
                unsafe {
                    *stk.get_unchecked_mut(sl) = w as u32;
                }
                sl += 1;
            }
        }
    }

    df[..dfn].sort_unstable_by_key(|&o| ord[o as usize]);
    db[..dbn].sort_unstable_by_key(|&o| ord[o as usize]);
    let mut pn = 0usize;
    let mut i = 0usize;
    let mut j = 0usize;
    while i < dbn || j < dfn {
        let take_b = if j >= dfn {
            true
        } else if i >= dbn {
            false
        } else {
            ord[db[i] as usize] < ord[df[j] as usize]
        };
        if take_b {
            pool[pn] = ord[db[i] as usize];
            i += 1;
        } else {
            pool[pn] = ord[df[j] as usize];
            j += 1;
        }
        pn += 1;
    }
    for k in 0..pn {
        let o = if k < dbn { db[k] } else { df[k - dbn] } as usize;
        let r = pool[k];
        unsafe {
            *ord.get_unchecked_mut(o) = r;
        }
        rec[o].0[R_ORD] = r;
        oat[r as usize] = o as u32;
    }
    for k in 0..pn {
        let o = if k < dbn { db[k] } else { df[k - dbn] } as usize;
        let r = pool[k];
        let a = jp[o];
        if a != NONE {
            rec[a as usize].0[R_JN_ORD] = r;
        }
        let a = mp[o];
        if a != NONE {
            rec[a as usize].0[R_MN_ORD] = r;
        }
        let a = jn[o];
        if a != NONE {
            rec[a as usize].0[R_JP_ORD] = r;
        }
        let a = mn[o];
        if a != NONE {
            rec[a as usize].0[R_MP_ORD] = r;
        }
    }
    false
}
}

pub mod tabu {
//! Fuel-bounded tabu search over critical-block moves.
use super::graph::{Graph, GraphState, NONE};
use super::model::Model;
use rand::{rngs::SmallRng, Rng};

/// Fuel level below which the search stops.
pub struct Budget {
    pub floor: u64,
}

impl Budget {
    #[inline]
    pub fn left(&self) -> u64 {
        super::fuel_remaining().saturating_sub(self.floor)
    }
    #[inline]
    pub fn ok(&self, margin: u64) -> bool {
        self.left() > margin
    }
}

/// Tabu tenure is drawn in TENURE_MIN..=TENURE_MIN + TENURE_SPAN.
const TENURE_MIN: usize = 8;
const TENURE_SPAN: usize = 8;
/// Random adjacent swaps applied by a perturbation.
const PERTURB_KICKS: usize = 8;
/// Best-estimate moves evaluated exactly before falling back to a perturbation.
const VERIFY_TRIES: usize = 4;

#[derive(Clone, Copy, PartialEq)]
enum Mv {
    Swap(u32, u32),
    Relocate(u32, u32),
    Assign(u32, u32, u32, u32),
}

#[derive(Clone, Copy)]
pub struct TabuCfg {
    pub max_assign_moves: usize,
    pub restart_after: u32,
    pub sterile_limit: u32,
    pub any_restart: bool,
}

#[inline]
fn estimate_swap(g: &Graph, u: u32, v: u32) -> u32 {
    let (ui, vi) = (u as usize, v as usize);
    let mp_u = g.mprev(ui);
    let ms_v = g.mnext(vi);
    let rv = g.ins_rp(g.jprev[vi]).max(g.ins_rp(mp_u));
    let ru = (rv + g.dur[vi]).max(g.ins_rp(g.jprev[ui]));
    let qu = g.ins_qp(g.jnext[ui]).max(g.ins_qp(ms_v));
    let qv = (qu + g.dur[ui]).max(g.ins_qp(g.jnext[vi]));
    (rv + g.dur[vi] + qv).max(ru + g.dur[ui] + qu)
}

#[inline]
fn best_insertion(g: &Graph, o: usize, m: u32, d: u32) -> (u32, u32) {
    // SAFETY: search bounds and endpoint guards keep sequence reads below n; sequence entries are
    // operations.
    let seq = &g.mseq[m as usize];
    let rj = g.ins_rp(g.jprev[o]);
    let qj = g.ins_qp(g.jnext[o]);
    let n = seq.len();
    let mut start = 0usize;
    if n >= 8 {
        let (mut lo, mut hi) = (0usize, n);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if g.ins_rend(unsafe { *seq.get_unchecked(mid) } as usize) >= rj {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        let piv = lo;
        let q = if piv < n {
            g.ins_qend(unsafe { *seq.get_unchecked(piv) } as usize)
                .max(qj)
        } else {
            qj
        };
        let est_piv = rj + d + q;
        let (mut lo, mut hi) = (0usize, piv);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if rj + d + g.ins_qend(unsafe { *seq.get_unchecked(mid) } as usize) > est_piv {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        start = lo;
    }
    let mut best_est = u32::MAX;
    let mut best_at = 0u32;
    let mut r = rj;
    if start > 0 {
        let v = g.ins_rend(unsafe { *seq.get_unchecked(start - 1) } as usize);
        if v > r {
            r = v;
        }
    }
    for at in start..=n {
        if at > start {
            let p = unsafe { *seq.get_unchecked(at - 1) } as usize;
            let v = g.ins_rend(p);
            if v > r {
                r = v;
            }
        }
        if r + d + qj >= best_est {
            break;
        }
        let q = if at < n {
            let sx = unsafe { *seq.get_unchecked(at) } as usize;
            let v = g.ins_qend(sx);
            if v > qj {
                v
            } else {
                qj
            }
        } else {
            qj
        };
        let est = r + d + q;
        if est < best_est {
            best_est = est;
            best_at = at as u32;
        }
    }
    (best_est, best_at)
}

pub fn tabu_search(
    md: &Model,
    g: &mut Graph,
    best_state: &mut GraphState,
    start_ms: u32,
    cfg: &TabuCfg,
    rng: &mut SmallRng,
    budget: &Budget,
    per_eval_hint: u64,
    on_improve: &mut dyn FnMut(&Graph, u32) -> bool,
) -> (bool, u32) {
    let mut best_ms = start_ms;
    let mut sterile: u32 = 0;
    let mut improved_since_restart = false;
    let mut stopped = false;

    let mut blocks: Vec<(u32, u32, u32)> = Vec::with_capacity(64);
    let mut moves: Vec<(u32, Mv)> = Vec::with_capacity(512);
    let mut crit_flex: Vec<u32> = Vec::with_capacity(256);
    let tenure_cap = TENURE_MIN + TENURE_SPAN + 2;
    let mut ring: Vec<(u8, u32, u32)> = Vec::with_capacity(tenure_cap);
    let mut ring_pos = 0usize;
    let mut tb_stamp: Vec<u32> = vec![0; g.n_ops];
    let mut tb_kind: Vec<u8> = vec![0; g.n_ops];
    let mut stamp: u32 = 0;

    let mut since_improve = 0u32;
    let mut iter_cost = per_eval_hint.saturating_mul(8).max(1);

    if g.evaluate().is_none() {
        return (false, best_ms);
    }

    loop {
        if !budget.ok(iter_cost.saturating_mul(2)) {
            break;
        }
        let before = super::fuel_remaining();

        g.compute_tails();

        g.critical_blocks_flex(md, &mut blocks, &mut crit_flex, cfg.max_assign_moves > 0);
        moves.clear();
        for &(m, i, j) in blocks.iter() {
            let m = m as usize;
            let (i, j) = (i as usize, j as usize);
            let seq = &g.mseq[m];
            let (u, v) = (seq[i], seq[i + 1]);
            if g.jnext[u as usize] != v as i32 {
                moves.push((estimate_swap(g, u, v), Mv::Swap(u, v)));
            }
            if j - i >= 2 {
                let (u, v) = (seq[j - 1], seq[j]);
                if g.jnext[u as usize] != v as i32 {
                    moves.push((estimate_swap(g, u, v), Mv::Swap(u, v)));
                }
                // Relocations of every block operation to either block end. SAFETY: t is in the
                // machine block i..=j; seq[t] is an operation below n_ops.
                let len = seq.len();
                let ri = g.ins_rp(if i > 0 { seq[i - 1] as i32 } else { NONE });
                let qi = g.ins_qp(seq[i] as i32);
                let rj = g.ins_rp(seq[j] as i32);
                let qjb = g.ins_qp(if j + 1 < len { seq[j + 1] as i32 } else { NONE });
                for t in i..=j {
                    let o = unsafe { *seq.get_unchecked(t) };
                    let oi = o as usize;
                    let ro = g.ins_rp(unsafe { *g.jprev.get_unchecked(oi) });
                    let qo = g.ins_qp(unsafe { *g.jnext.get_unchecked(oi) });
                    let d = unsafe { *g.dur.get_unchecked(oi) };
                    if t != i {
                        moves.push((ro.max(ri) + d + qo.max(qi), Mv::Relocate(o, i as u32)));
                    }
                    if t != j {
                        moves.push((ro.max(rj) + d + qo.max(qjb), Mv::Relocate(o, j as u32)));
                    }
                }
            }
        }

        if cfg.max_assign_moves > 0 && !crit_flex.is_empty() {
            let take = cfg.max_assign_moves.min(crit_flex.len());
            for i in 0..take {
                let j = i + rng.gen_range(0..(crit_flex.len() - i));
                crit_flex.swap(i, j);
                let o = crit_flex[i] as usize;
                let lo = md.opt_lo[o] as usize;
                let hi = md.opt_hi[o] as usize;
                for t in lo..hi {
                    let (m, d) = md.opts[t];
                    if m != g.mach[o] {
                        let (est, at) = best_insertion(g, o, m, d);
                        let mv = Mv::Assign(o as u32, m, d, at);
                        moves.push((est, mv));
                    }
                }
            }
        }

        if moves.is_empty() {
            perturb(g, best_state, rng, PERTURB_KICKS);
            since_improve = since_improve.saturating_add(1);
            continue;
        }

        stamp = stamp.wrapping_add(1);
        if stamp == 0 {
            for s in tb_stamp.iter_mut() {
                *s = 0;
            }
            stamp = 1;
        }
        for &(k, a, _) in ring.iter() {
            let a = a as usize;
            if tb_stamp[a] != stamp {
                tb_stamp[a] = stamp;
                tb_kind[a] = 0;
            }
            tb_kind[a] |= 1u8 << k;
        }

        let tries = VERIFY_TRIES.min(moves.len());
        let mut applied: Option<u32> = None;
        for t in 0..tries {
            let mut bi = usize::MAX;
            let mut bk = u32::MAX;
            for (dx, &(est, mv)) in moves[t..].iter().enumerate() {
                if est >= bk {
                    continue;
                }
                if est >= best_ms && is_tabu(&ring, &tb_stamp, &tb_kind, stamp, mv) {
                    continue;
                }
                bk = est;
                bi = t + dx;
            }
            if bi == usize::MAX {
                break;
            }
            moves.swap(t, bi);
            let mv = moves[t].1;
            let (ms, undo) = apply_and_eval(g, mv);
            if ms != u32::MAX {
                push_tabu(
                    &mut ring,
                    &mut ring_pos,
                    mv,
                    TENURE_MIN + rng.gen_range(0..=TENURE_SPAN),
                );
                applied = Some(ms);
                break;
            }
            undo_move(g, mv, undo);
        }

        match applied {
            Some(ms) => {
                if ms < best_ms {
                    best_ms = ms;
                    g.snapshot(best_state);
                    if !on_improve(g, best_ms) {
                        break;
                    }
                    since_improve = 0;
                    improved_since_restart = true;
                    sterile = 0;
                } else {
                    since_improve += 1;
                }
            }
            None => {
                let pms = perturb(g, best_state, rng, PERTURB_KICKS);
                if let Some(ms) = pms {
                    if ms < best_ms {
                        best_ms = ms;
                        // The saved state must match best_ms.
                        g.snapshot(best_state);
                        if !on_improve(g, best_ms) {
                            break;
                        }
                    }
                }
                since_improve = since_improve.saturating_add(1);
            }
        }

        if since_improve >= cfg.restart_after {
            if !improved_since_restart || cfg.any_restart {
                sterile += 1;
                if cfg.sterile_limit > 0 && sterile >= cfg.sterile_limit {
                    stopped = true;
                    break;
                }
            }
            improved_since_restart = false;
            g.restore_from(best_state);
            let pms = perturb(g, best_state, rng, PERTURB_KICKS);
            if let Some(ms) = pms {
                if ms < best_ms {
                    best_ms = ms;
                    g.snapshot(best_state);
                    if !on_improve(g, best_ms) {
                        break;
                    }
                }
            }
            ring.clear();
            ring_pos = 0;
            since_improve = 0;
        }

        let spent = before.saturating_sub(super::fuel_remaining());
        if spent > iter_cost {
            iter_cost = spent;
        }
    }

    g.restore_from(best_state);
    let _ = g.evaluate();
    (stopped, best_ms)
}

#[inline]
fn apply_and_eval(g: &mut Graph, mv: Mv) -> (u32, (u32, usize, u32)) {
    match mv {
        Mv::Swap(u, v) => {
            g.swap_adjacent(u, v);
            let ms = g.evaluate().unwrap_or(u32::MAX);
            (ms, (0, 0, 0))
        }
        Mv::Relocate(o, at) => {
            let old = g.relocate_within(o as usize, at as usize);
            let ms = g.evaluate().unwrap_or(u32::MAX);
            (ms, (0, old, 0))
        }
        Mv::Assign(o, m, d, at) => {
            let old = g.reassign_at(o as usize, m, at as usize, d);
            let ms = g.evaluate().unwrap_or(u32::MAX);
            (ms, old)
        }
    }
}

#[inline]
fn undo_move(g: &mut Graph, mv: Mv, undo: (u32, usize, u32)) {
    match mv {
        Mv::Swap(u, v) => g.swap_adjacent(v, u),
        Mv::Relocate(o, _) => {
            g.relocate_within(o as usize, undo.1);
        }
        Mv::Assign(o, _, _, _) => g.undo_reassign(o as usize, undo),
    }
}

#[inline]
fn tabu_key(mv: Mv) -> (u8, u32, u32) {
    match mv {
        Mv::Swap(u, v) => (0u8, u, v),
        Mv::Relocate(o, _) => (1u8, o, u32::MAX),
        Mv::Assign(o, _, _, _) => (2u8, o, u32::MAX),
    }
}

#[inline]
fn is_tabu(
    ring: &[(u8, u32, u32)],
    tb_stamp: &[u32],
    tb_kind: &[u8],
    stamp: u32,
    mv: Mv,
) -> bool {
    let key = tabu_key(mv);
    let a = key.1 as usize;
    if tb_stamp[a] != stamp || tb_kind[a] & (1u8 << key.0) == 0 {
        return false;
    }
    if key.0 != 0 {
        return true;
    }
    ring.iter().any(|&e| e == key)
}

#[inline]
fn push_tabu(ring: &mut Vec<(u8, u32, u32)>, pos: &mut usize, mv: Mv, tenure: usize) {
    let key = match mv {
        Mv::Swap(u, v) => (0u8, v, u),
        other => tabu_key(other),
    };
    let cap = tenure.max(1);
    if ring.len() < cap {
        ring.push(key);
    } else {
        if *pos >= ring.len() {
            *pos = 0;
        }
        ring[*pos] = key;
        *pos += 1;
    }
}

pub fn kick(g: &mut Graph, state: &GraphState, rng: &mut SmallRng) {
    let _ = perturb(g, state, rng, PERTURB_KICKS);
}

fn perturb(
    g: &mut Graph,
    best_state: &GraphState,
    rng: &mut SmallRng,
    kicks: usize,
) -> Option<u32> {
    for _ in 0..kicks {
        let m = rng.gen_range(0..g.n_mach);
        let len = g.mseq[m].len();
        if len < 2 {
            continue;
        }
        let i = rng.gen_range(0..len - 1);
        let u = g.mseq[m][i];
        let v = g.mseq[m][i + 1];
        if g.jnext[u as usize] == v as i32 {
            continue;
        }
        g.swap_adjacent(u, v);
        if g.evaluate().is_none() {
            g.swap_adjacent(v, u);
        }
    }
    match g.evaluate() {
        None => {
            g.restore_from(best_state);
            g.evaluate()
        }
        some => some,
    }
}
}

pub mod ref_greedy {
//! Dispatching-rule baseline.
use anyhow::{anyhow, Result};
use rand::seq::SliceRandom;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::HashMap;
use tig_challenges::job_scheduling::{Challenge, Solution};

/// Weight of the minimum processing time in the remaining-work estimate of a job.
const WORK_MIN_WEIGHT: f64 = 0.3;
/// Randomised dispatching runs after the five deterministic rules.
const RANDOM_RESTARTS: usize = 10;

fn average_processing_time(operation: &HashMap<usize, u32>) -> f64 {
    if operation.is_empty() {
        return 0.0;
    }
    let sum: u32 = operation.values().sum();
    sum as f64 / operation.len() as f64
}

fn min_processing_time(operation: &HashMap<usize, u32>) -> f64 {
    operation.values().copied().min().unwrap_or(0) as f64
}

fn earliest_end_time(
    time: u32,
    machine_available_time: &[u32],
    operation: &HashMap<usize, u32>,
) -> u32 {
    let mut earliest_end = u32::MAX;
    for (&machine_id, &proc_time) in operation.iter() {
        let start = time.max(machine_available_time[machine_id]);
        let end = start + proc_time;
        if end < earliest_end {
            earliest_end = end;
        }
    }
    earliest_end
}

#[derive(Clone, Copy)]
enum DispatchRule {
    MostWorkRemaining,
    MostOpsRemaining,
    LeastFlexibility,
    ShortestProcTime,
    LongestProcTime,
}

#[derive(Clone, Copy)]
struct Candidate {
    job: usize,
    priority: f64,
    machine_end: u32,
    proc_time: u32,
    flexibility: usize,
}

fn better_candidate(candidate: &Candidate, best: &Candidate, eps: f64) -> bool {
    if candidate.priority > best.priority + eps {
        return true;
    }
    if (candidate.priority - best.priority).abs() <= eps {
        if candidate.machine_end < best.machine_end {
            return true;
        }
        if candidate.machine_end == best.machine_end {
            if candidate.proc_time < best.proc_time {
                return true;
            }
            if candidate.proc_time == best.proc_time {
                if candidate.flexibility < best.flexibility {
                    return true;
                }
                if candidate.flexibility == best.flexibility && candidate.job < best.job {
                    return true;
                }
            }
        }
    }
    false
}

/// Event-driven dispatching: each idle machine takes, among the ready jobs that finish earliest on
/// it, the best by rule, or a random one of the top-k when a rng is supplied.
fn run_dispatch_rule(
    challenge: &Challenge,
    job_products: &[usize],
    product_work_times: &[Vec<f64>],
    job_ops_len: &[usize],
    job_total_work: &[f64],
    rule: DispatchRule,
    mut random_top_k: Option<(usize, &mut SmallRng)>,
) -> Result<(Vec<Vec<(usize, u32)>>, u32)> {
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;

    let mut job_next_op_idx = vec![0usize; num_jobs];
    let mut job_ready_time = vec![0u32; num_jobs];
    let mut machine_available_time = vec![0u32; num_machines];
    let mut job_schedule = job_ops_len
        .iter()
        .map(|&ops_len| Vec::with_capacity(ops_len))
        .collect::<Vec<_>>();
    let mut job_remaining_work = job_total_work.to_vec();

    let mut remaining_ops = job_ops_len.iter().sum::<usize>();
    let mut time = 0u32;
    let eps = 1e-9_f64;
    let mut candidates: Vec<Candidate> = Vec::new();

    while remaining_ops > 0 {
        let mut available_machines = (0..num_machines)
            .filter(|&m| machine_available_time[m] <= time)
            .collect::<Vec<usize>>();
        if let Some((_, ref mut rng)) = random_top_k {
            available_machines.shuffle(*rng);
        }

        let mut scheduled_any = false;
        for &machine in available_machines.iter() {
            let mut best_candidate: Option<Candidate> = None;
            candidates.clear();
            for job in 0..num_jobs {
                if job_next_op_idx[job] >= job_ops_len[job] {
                    continue;
                }
                if job_ready_time[job] > time {
                    continue;
                }
                let product = job_products[job];
                let op_idx = job_next_op_idx[job];
                let op_times = &challenge.product_processing_times[product][op_idx];
                let proc_time = match op_times.get(&machine) {
                    Some(&value) => value,
                    None => continue,
                };
                let earliest_end = earliest_end_time(time, &machine_available_time, op_times);
                let machine_end = time.max(machine_available_time[machine]) + proc_time;
                if machine_end != earliest_end {
                    continue;
                }
                let flexibility = op_times.len();
                let priority = match rule {
                    DispatchRule::MostWorkRemaining => job_remaining_work[job],
                    DispatchRule::MostOpsRemaining => {
                        (job_ops_len[job] - job_next_op_idx[job]) as f64
                    }
                    DispatchRule::LeastFlexibility => -(flexibility as f64),
                    DispatchRule::ShortestProcTime => -(proc_time as f64),
                    DispatchRule::LongestProcTime => proc_time as f64,
                };
                let candidate = Candidate {
                    job,
                    priority,
                    machine_end,
                    proc_time,
                    flexibility,
                };
                if random_top_k.is_some() {
                    candidates.push(candidate);
                } else if best_candidate
                    .as_ref()
                    .map_or(true, |best| better_candidate(&candidate, best, eps))
                {
                    best_candidate = Some(candidate);
                }
            }
            if let Some((top_k, ref mut rng)) = random_top_k {
                if !candidates.is_empty() {
                    candidates.sort_by(|a, b| {
                        let ord = b
                            .priority
                            .partial_cmp(&a.priority)
                            .unwrap_or(Ordering::Equal);
                        if ord != Ordering::Equal {
                            return ord;
                        }
                        let ord = a.machine_end.cmp(&b.machine_end);
                        if ord != Ordering::Equal {
                            return ord;
                        }
                        let ord = a.proc_time.cmp(&b.proc_time);
                        if ord != Ordering::Equal {
                            return ord;
                        }
                        let ord = a.flexibility.cmp(&b.flexibility);
                        if ord != Ordering::Equal {
                            return ord;
                        }
                        a.job.cmp(&b.job)
                    });
                    let k = top_k.min(candidates.len());
                    let pick = rng.gen_range(0..k);
                    best_candidate = Some(candidates[pick]);
                }
            }

            if let Some(candidate) = best_candidate {
                let job = candidate.job;
                let product = job_products[job];
                let op_idx = job_next_op_idx[job];
                let proc_time = candidate.proc_time;
                let start_time = time.max(machine_available_time[machine]);
                let end_time = start_time + proc_time;
                job_schedule[job].push((machine, start_time));
                job_next_op_idx[job] += 1;
                job_ready_time[job] = end_time;
                machine_available_time[machine] = end_time;
                job_remaining_work[job] -= product_work_times[product][op_idx];
                if job_remaining_work[job] < 0.0 {
                    job_remaining_work[job] = 0.0;
                }
                remaining_ops -= 1;
                scheduled_any = true;
            }
        }

        if remaining_ops == 0 {
            break;
        }

        let mut next_time: Option<u32> = None;
        for &t in machine_available_time.iter() {
            if t > time {
                next_time = Some(next_time.map_or(t, |best| best.min(t)));
            }
        }
        for job in 0..num_jobs {
            if job_next_op_idx[job] < job_ops_len[job] && job_ready_time[job] > time {
                let t = job_ready_time[job];
                next_time = Some(next_time.map_or(t, |best| best.min(t)));
            }
        }
        time = next_time.ok_or_else(|| {
            if scheduled_any {
                anyhow!("No next event time found while operations remain unscheduled")
            } else {
                anyhow!("No schedulable operations remain; dispatching rules stalled")
            }
        })?;
    }

    let makespan = job_ready_time.iter().copied().max().unwrap_or(0);
    Ok((job_schedule, makespan))
}

/// Best schedule of five dispatching rules and ten randomised top-k runs.
pub fn greedy_baseline(challenge: &Challenge) -> Result<Solution> {
    let num_jobs = challenge.num_jobs;
    let mut job_products = Vec::with_capacity(num_jobs);
    for (product, count) in challenge.jobs_per_product.iter().enumerate() {
        for _ in 0..*count {
            job_products.push(product);
        }
    }
    if job_products.len() != num_jobs {
        return Err(anyhow!(
            "Job count mismatch. Expected {}, got {}",
            num_jobs,
            job_products.len()
        ));
    }

    let mut product_work_times = Vec::with_capacity(challenge.product_processing_times.len());
    for product_ops in challenge.product_processing_times.iter() {
        let mut work_ops = Vec::with_capacity(product_ops.len());
        for op in product_ops.iter() {
            let avg = average_processing_time(op);
            let min = min_processing_time(op);
            work_ops.push(avg * (1.0 - WORK_MIN_WEIGHT) + min * WORK_MIN_WEIGHT);
        }
        product_work_times.push(work_ops);
    }

    let mut job_ops_len = Vec::with_capacity(num_jobs);
    let mut job_total_work: Vec<f64> = Vec::with_capacity(num_jobs);
    for &product in job_products.iter() {
        let work_ops = &product_work_times[product];
        job_ops_len.push(work_ops.len());
        job_total_work.push(work_ops.iter().sum());
    }

    let rules = [
        DispatchRule::MostWorkRemaining,
        DispatchRule::MostOpsRemaining,
        DispatchRule::LeastFlexibility,
        DispatchRule::ShortestProcTime,
        DispatchRule::LongestProcTime,
    ];

    let mut best: Option<(Vec<Vec<(usize, u32)>>, u32)> = None;
    for rule in rules.iter().copied() {
        let result = run_dispatch_rule(
            challenge,
            &job_products,
            &product_work_times,
            &job_ops_len,
            &job_total_work,
            rule,
            None,
        )?;
        if best.as_ref().map_or(true, |b| result.1 < b.1) {
            best = Some(result);
        }
    }
    let mut best = best.ok_or_else(|| anyhow!("No valid schedule produced"))?;

    let mut rng = SmallRng::from_seed(challenge.seed);
    for _ in 0..RANDOM_RESTARTS {
        let seed = rng.r#gen::<u64>();
        let rule = rules[rng.gen_range(0..rules.len())];
        let random_top_k = rng.gen_range(2..=5);
        let mut local_rng = SmallRng::seed_from_u64(seed);
        let result = run_dispatch_rule(
            challenge,
            &job_products,
            &product_work_times,
            &job_ops_len,
            &job_total_work,
            rule,
            Some((random_top_k, &mut local_rng)),
        )?;
        if result.1 < best.1 {
            best = result;
        }
    }

    Ok(Solution {
        job_schedule: best.0,
    })
}
}
