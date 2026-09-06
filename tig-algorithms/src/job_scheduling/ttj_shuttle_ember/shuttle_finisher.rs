// Public ember_stromboli/shuttle_bell-family graph-tabu engine, adapted only
// to accept the completed task_tree_j schedule as its warm start.
use anyhow::Result;
use std::cell::Cell;
use tig_challenges::job_scheduling::*;

use rand::{rngs::SmallRng, SeedableRng};

use self::graph::{Graph, GraphState};
use self::model::Model;
use self::tabu::{Budget, TabuCfg, TabuOut};

extern "C" {
    #[allow(non_upper_case_globals)]
    static __fuel_remaining: u64;
}

#[inline(always)]
fn fuel_remaining() -> u64 {
    unsafe { core::ptr::read_volatile(core::ptr::addr_of!(__fuel_remaining)) }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FinisherStats {
    pub final_makespan: u32,
    pub fuel_used: u64,
    pub iterations: u64,
    pub moves_seen: u64,
    pub moves_applied: u64,
    pub swap_applied: u64,
    pub relocate_applied: u64,
    pub assign_applied: u64,
    pub improvements: u64,
    pub swap_improvements: u64,
    pub relocate_improvements: u64,
    pub assign_improvements: u64,
    pub perturb_improvements: u64,
}

pub fn finish_from_schedule(
    challenge: &Challenge,
    start_solution: &Solution,
    start_makespan: u32,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    fuel_cap: u64,
) -> Result<FinisherStats> {
    let fuel_at_entry = fuel_remaining();
    let floor_fuel = fuel_at_entry.saturating_sub(fuel_cap);

    let md = Model::build(challenge);
    let mut g = Graph::new(&md);
    if !g.load_schedule(&md, &start_solution.job_schedule) {
        return Ok(FinisherStats {
            final_makespan: start_makespan,
            fuel_used: fuel_at_entry.saturating_sub(fuel_remaining()),
            ..FinisherStats::default()
        });
    }
    let retimed = match g.evaluate() {
        Some(ms) => ms,
        None => {
            return Ok(FinisherStats {
                final_makespan: start_makespan,
                fuel_used: fuel_at_entry.saturating_sub(fuel_remaining()),
                ..FinisherStats::default()
            })
        }
    };
    let ceiling = retimed.min(start_makespan);
    let mut best_state = GraphState::new(&md);
    g.snapshot(&mut best_state);
    if retimed < start_makespan {
        save_solution(&Solution {
            job_schedule: g.to_schedule(&md),
        })?;
    }

    let budget = Budget {
        remaining: &fuel_remaining,
        floor: floor_fuel,
    };

    let before_cal = fuel_remaining();
    let _ = g.evaluate();
    let per_eval = before_cal.saturating_sub(fuel_remaining()).max(1);

    let mut rng = SmallRng::from_seed(challenge.seed);
    let saved_ms = Cell::new(ceiling);
    let saved_improvements = Cell::new(if retimed < start_makespan { 1u64 } else { 0u64 });
    let mut on_improve = |gr: &Graph, ms: u32| {
        if ms >= saved_ms.get() || ms >= start_makespan {
            return;
        }
        if save_solution(&Solution {
            job_schedule: gr.to_schedule(&md),
        })
        .is_ok()
        {
            saved_ms.set(ms);
            saved_improvements.set(saved_improvements.get() + 1);
        }
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
    let avg_flex = if flex_ops > 0 {
        flex_opts / flex_ops
    } else {
        1
    };
    let assign_moves = if flex_ops == 0 {
        0
    } else {
        (32usize * 2 / avg_flex.max(1)).max(4)
    };

    // This engine is only reached when `max_flex > 1` (see the dispatcher in `mod.rs`), so
    // `flex_ops >= 1` and the instance is always flexible here: the two rigid arms of the former
    // four-way choice on (shape, single_route) were unreachable. The two remaining arms read
    // `restart_after_hybrid` and `restart_after`, both 100000 and neither supplied by the shipped
    // map, so the route probe that separated them decided nothing - and it cost three sorted key
    // scans per product. Measured 48 nonces out of 48: shape_flex=1 single_route=1
    // restart_after=100000.
    let cfg = TabuCfg {
        tenure_min: 8,
        tenure_span: 8,
        max_assign_moves: assign_moves,
        restart_after: 100_000,
        perturb_kicks: 8,
        use_n7: true,
        max_moves: 0,
        verify_tries: 4,
    };

    let out: TabuOut = tabu::tabu_search(
        &md,
        &mut g,
        &mut best_state,
        ceiling,
        &cfg,
        &mut rng,
        &budget,
        per_eval,
        &mut on_improve,
    );
    Ok(FinisherStats {
        final_makespan: saved_ms.get().min(out.best_ms),
        fuel_used: fuel_at_entry.saturating_sub(fuel_remaining()),
        iterations: out.iterations,
        moves_seen: out.moves_seen,
        moves_applied: out.moves_applied,
        swap_applied: out.swap_applied,
        relocate_applied: out.relocate_applied,
        assign_applied: out.assign_applied,
        improvements: saved_improvements.get(),
        swap_improvements: out.swap_improvements,
        relocate_improvements: out.relocate_improvements,
        assign_improvements: out.assign_improvements,
        perturb_improvements: out.perturb_improvements,
    })
}

pub mod model {
// Instance model for the hybrid track.
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
// Graph structures for the hybrid track.
//
// INCREMENTAL EVALUATION. `head`, `qend` and `makespan` are no longer recomputed from a full
// Kahn sort on every move: a topological order `ord`/`oat` is kept up to date by Pearce-Kelly,
// and only the zone the move dirties is rescanned. The full sort remains in `evaluate_full`,
// called on load and after a state restore.
//
// Six invariants carry the correctness. Each one was measured before being written, on the real
// hybrid route (on-chain instance, production cap 1e11, 4 draws); the counter that establishes it
// is named each time. Detail and raw output: `outils/portes-pk.md`.
//
// I1. `ord` is a valid topological order of the current graph at every successful evaluation.
//     Checked at gate 3: `bad = 0` over 1 613 059 evaluations and 515 734 real repairs.
//     Negative control (gate 4, one real repair in three skipped): `bad = sabo` exactly,
//     74 474 / 112 962 / 107 366 / 106 029.
//
// I2. A move adds at most three machine arcs, of which at most one violates the order.
//     Proved for the three operators (see `pk_arc`), and checked: counter `multi = 0` over
//     1 613 059 evaluations. Consequence: a single repair per move, no fixpoint loop, and no
//     undo journal at all -- when PK reports a cycle it has not yet written anything into `ord`,
//     and the engine's undo puts its arcs back through the same path.
//
// I3. The bridge arc left by a detach never violates the order: it spans `A -> o -> B`, hence
//     `ord[A] < ord[o] < ord[B]`. Checked: `nts[detach] = 0` over 900 230 observations.
//     It is therefore not submitted to PK, only to seeding.
//
// I4. PK misses no cycle that the Kahn sort sees: counter `miss = 0` over the same
//     1 613 059 evaluations. The acyclicity verdict therefore comes from PK, and Kahn leaves the
//     hot loop.
//
// I5. The makespan is reached on a sink (`jnext == NONE && mnx == NONE`). This is a property of
//     longest paths -- any successor finishes later -- and it was checked empirically by the
//     shadow FIFO of 22/08 (~440 000 controls). So only the `n_jobs` end-of-route operations are
//     swept instead of the 1650 operations.
//
// I6. The sweep starts from the smallest seeded rank and runs to the end: a node can only be
//     dirtied by a predecessor, hence one of strictly lower rank. The rank is read after the
//     order repair, which is what makes the invariant true despite the permutations PK applies.
//     Length measured in the PK order: 0.9896 times that of the Kahn order (4 draws), that is,
//     slightly shorter.
use super::model::Model;

pub const NONE: i32 = -1;

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
    // Constant in-degree: 1 assumed machine predecessor + 1 if the operation has a route
    // predecessor. `evaluate` copies it in one go (memcpy = 40 fuel) then decrements the
    // `n_mach` sequence heads, instead of rescanning the 1700 operations.
    indeg_full: Vec<u8>,
    // `topo` and `stack` are fixed-size buffers of `n_ops` entries with their length kept apart.
    // Each operation enters the stack at most once (it is only pushed when its in-degree drops to
    // zero), so `n_ops` slots always suffice. This avoids the capacity check and the pointer
    // reload that `Vec::push`/`Vec::pop` charge on every operation, without changing the LIFO
    // order one bit.
    topo: Vec<u32>,
    topo_len: usize,
    stack: Vec<u32>,
    pub head: Vec<u32>,
    // `qend[o] == tail[o] + dur[o]`. No consumer wants `tail` on its own: `qp()`,
    // `best_insertion` and `is_critical` all rebuilt the sum on every read. Storing it
    // pre-added removes two loads and two additions per operation in the backward pass, and
    // seven fuel per criticality test.
    pub qend: Vec<u32>,
    mnx: Vec<i32>,
    mpv: Vec<i32>,
    pub makespan: u32,

    /// Dynamic topological order: `ord[o]` = rank of the operation, `oat[r]` = operation of
    /// rank `r`. Kept by Pearce-Kelly on every arc added (I1), it replaces `topo` in the hot
    /// loop.
    ord: Vec<u32>,
    oat: Vec<u32>,
    // Pearce-Kelly buffers. Stamped marks, to avoid a clear on every call.
    markf: Vec<u32>,
    markb: Vec<u32>,
    pk_stamp: u32,
    pk_stk: Vec<u32>,
    pk_df: Vec<u32>,
    pk_db: Vec<u32>,
    pk_pool: Vec<u32>,
    /// Seeds of the forward sweep (`head`) and of the backward sweep (`qend`), with their
    /// dedup stamps. The two lists are emptied at different moments: `evaluate` is called once
    /// per candidate move, `compute_tails` once per iteration.
    seed: Vec<u32>,
    seed_n: usize,
    sdup: Vec<u32>,
    sstamp: u32,
    qseed: Vec<u32>,
    qseed_n: usize,
    qdup: Vec<u32>,
    qstamp: u32,
    /// Dirty marks of the current sweep.
    dmark: Vec<u32>,
    dstamp: u32,
    // Critical set by propagation (fusion 3): one bit per critical operation, indexed by the
    // slot `mbase[machine] + position`; `cqueue`/`cstamp` drive the BFS from the roots.
    mbase: Vec<u32>,
    critbits: Vec<u64>,
    cstamp: Vec<u32>,
    cstamp_cur: u32,
    cqueue: Vec<u32>,
    /// Operations with no route successor: the sinks are a subset of them (I5).
    job_last: Vec<u32>,
    /// Order and values to rebuild from scratch (load, restore, seed overflow).
    full: bool,
    full_q: bool,
    /// Pearce-Kelly has seen a cycle since the last evaluation (I4).
    cyc: bool,
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
            topo_len: 0,
            stack: vec![0; n],
            head: vec![0; n],
            qend: vec![0; n],
            mnx: vec![NONE; n],
            mpv: vec![NONE; n],
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
            dmark: vec![0; n],
            dstamp: 1,
            mbase: vec![0; md.n_machines + 1],
            critbits: vec![0; (n + 63) / 64],
            cstamp: vec![0; n],
            cstamp_cur: 0,
            cqueue: vec![0; n],
            job_last,
            full: true,
            full_q: true,
            cyc: false,
        }
    }

    // `#[inline(always)]` accessors: same arithmetic as before, bounds still checked.
    #[inline(always)]
    pub fn rend(&self, o: usize) -> u32 {
        self.head[o] + self.dur[o]
    }
    #[inline(always)]
    pub fn qend_of(&self, o: usize) -> u32 {
        self.qend[o]
    }
    #[inline(always)]
    pub fn rp(&self, x: i32) -> u32 {
        if x == NONE { 0 } else { self.rend(x as usize) }
    }
    #[inline(always)]
    pub fn qp(&self, x: i32) -> u32 {
        if x == NONE { 0 } else { self.qend_of(x as usize) }
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
        // Set before the `relink_all`: `touch` returns immediately once `full` is raised, so the
        // load does not fill the seed lists for nothing.
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

    /// Recomputes `mnx`/`mpv` on positions `lo..=hi` of machine `m`.
    ///
    /// The machine links are now kept up to date incrementally by the operators
    /// (`swap_adjacent`, `relocate_within`, `detach`, `attach`): only a few positions change per
    /// move, whereas the former `evaluate` rewrote all 1700 of them on every call. The invariant
    /// maintained is exactly the one that pass produced: for every position `i` of `mseq[m]`,
    /// `mnx[mseq[m][i]] == mseq[m][i+1]` (or NONE) and
    /// `mpv[mseq[m][i]] == mseq[m][i-1]` (or NONE).
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
            self.mnx[o] = if i + 1 < len {
                self.mseq[m][i + 1] as i32
            } else {
                NONE
            };
            self.mpv[o] = if i > 0 { self.mseq[m][i - 1] as i32 } else { NONE };
            // The relinked window is a superset of the operations whose machine neighbour
            // changed: seeding here covers the arcs added and the arcs removed, with no extra
            // scan. One seed too many costs a recomputation that returns the same value and does
            // not propagate; a missing seed would return a wrong makespan silently.
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

    /// Check of the link invariant, active in test builds only (zero cost in release).
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
                mnx[o] = if i + 1 < s.len() { s[i + 1] as i32 } else { NONE };
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

    /// Entry point unchanged for the engine: returns the makespan, or `None` on a cycle.
    pub fn evaluate(&mut self) -> Option<u32> {
        if self.full {
            return self.evaluate_full();
        }
        if self.cyc {
            // I4: Pearce-Kelly has already decided. The seeds are not empty -- the engine is
            // about to undo its move, and the undo will add its own; the union will be swept at
            // the next evaluation.
            self.cyc = false;
            self.makespan = u32::MAX;
            return None;
        }
        self.eval_incr()
    }

    /// Full Kahn sort: load, restore, seed overflow. Also rebuilds `ord`/`oat`, which start
    /// again from the pop order.
    fn evaluate_full(&mut self) -> Option<u32> {
        let n = self.n_ops;
        debug_assert!(self.links_consistent(), "mnx/mpv out of sync with mseq");
        // `mnx`/`mpv` are kept up to date by the operators: all that is left to flatten is the
        // state proper to one evaluation. `fill`/`copy_from_slice` compile to memset/memcpy,
        // capped at 40 fuel by the counting pass, where the equivalent loop over 1700 operations
        // cost about 97 000.
        self.head.fill(0);
        self.indeg.copy_from_slice(&self.indeg_full);
        // Only the operation at the head of a machine sequence has no machine predecessor.
        // The former loop pushed those roots in increasing machine order (only `i == 0` could
        // give `d == 0`): that order is reproduced identically, so the starting stack, the
        // topological order and everything that follows from it are unchanged.
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
                }
                self.indeg[mn] -= 1;
                if self.indeg[mn] == 0 {
                    self.stack[sl] = mn as u32;
                    sl += 1;
                }
            }
        }
        self.topo_len = tl;
        if tl != n {
            self.makespan = u32::MAX;
            return None;
        }
        for i in 0..tl {
            let o = self.topo[i] as usize;
            self.ord[o] = i as u32;
            self.oat[i] = o as u32;
        }
        self.full = false;
        self.full_q = true;
        self.cyc = false;
        self.seed_n = 0;
        self.sstamp = self.sstamp.wrapping_add(1);
        self.makespan = ms;
        Some(ms)
    }

    /// Incremental backward sweep: from the largest seeded rank down to zero. Exact mirror of
    /// `eval_incr` (I6 by symmetry: a node can only be dirtied by a successor, hence one of
    /// strictly higher rank).
    pub fn compute_tails(&mut self) {
        if self.full_q {
            self.tails_full();
            return;
        }
        if self.qseed_n == 0 {
            return;
        }
        Self::bump(&mut self.dstamp, &mut self.dmark);
        let ds = self.dstamp;
        let mut hi = 0usize;
        for k in 0..self.qseed_n {
            let o = self.qseed[k] as usize;
            self.dmark[o] = ds;
            let r = self.ord[o] as usize;
            if r > hi {
                hi = r;
            }
        }
        self.qseed_n = 0;
        Self::bump(&mut self.qstamp, &mut self.qdup);
        let mut i = hi + 1;
        while i > 0 {
            i -= 1;
            let o = self.oat[i] as usize;
            if self.dmark[o] != ds {
                continue;
            }
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
            if v != self.qend[o] {
                self.qend[o] = v;
                // `qend[o]` already contains `dur[o]`, so a duration change shows up here:
                // no separate seeding to do, unlike the forward sweep.
                let jp = self.jprev[o];
                if jp != NONE {
                    self.dmark[jp as usize] = ds;
                }
                let mp = self.mpv[o];
                if mp != NONE {
                    self.dmark[mp as usize] = ds;
                }
            }
        }
    }

    /// Full backward sweep, in `oat` order -- identical value for value to the former scan over
    /// `topo`, of which `oat` is the copy after `evaluate_full`.
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
            self.qend[o] = t + self.dur[o];
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
        // Invalid graph state: no meaningful critical path; fall back to the scan.
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

    /// Beyond this many critical operations the propagation costs more than the full scan
    /// (the measured break-even sits around 15% of the operations). The guard is a COST
    /// choice only: both paths return exactly the same thing, which the fingerprint harness
    /// verifies by forcing one then the other.
    #[inline]
    fn crit_cap(&self) -> usize {
        self.n_ops / 8
    }

    /// Sets one bit per critical operation, at slot `mbase[machine] + position`.
    /// Returns `false` when the critical set exceeds the cost guard.
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

        // Roots: an operation without predecessors heads both its machine sequence and its
        // route, so its head is zero and it is critical iff `qend == makespan`.
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

        let mut hd = 0usize;
        while hd < ql {
            let o = self.cqueue[hd] as usize;
            hd += 1;
            let jn = self.jnext[o];
            if jn != NONE {
                let s = jn as usize;
                if self.cstamp[s] != st && self.head[s] + self.qend[s] == ms {
                    if ql >= cap {
                        return false;
                    }
                    self.cstamp[s] = st;
                    self.cqueue[ql] = s as u32;
                    ql += 1;
                    let slot = self.mbase[self.mach[s] as usize] as usize + self.mpos[s] as usize;
                    self.critbits[slot >> 6] |= 1u64 << (slot & 63);
                }
            }
            let mn = self.mnx[o];
            if mn != NONE {
                let s = mn as usize;
                if self.cstamp[s] != st && self.head[s] + self.qend[s] == ms {
                    if ql >= cap {
                        return false;
                    }
                    self.cstamp[s] = st;
                    self.cqueue[ql] = s as u32;
                    ql += 1;
                    let slot = self.mbase[self.mach[s] as usize] as usize + self.mpos[s] as usize;
                    self.critbits[slot >> 6] |= 1u64 << (slot & 63);
                }
            }
        }
        true
    }

    /// Walks the occupied slots in increasing order - machine by machine, position by
    /// position - and reforms exactly the blocks and the `flex` list of the original scan.
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
                let pos = slot - self.mbase[m] as usize;
                let o = self.mseq[m][pos] as usize;
                if act
                    && m == rm
                    && pos == rj + 1
                    && self.head[o] == self.head[rend] + self.dur[rend]
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

    /// The original scan, kept word for word: it is the fallback path and the witness.
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
        // Beyond `hi` the positions get their original occupant back: `remove` then `insert`
        // only move the segment [lo, hi]. The former loop ran to the end of the sequence and
        // rewrote the same values there.
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
        // I3: the bridge arc left by the removal never violates the order, so it is not
        // submitted to Pearce-Kelly -- only to seeding, which `relink` has already done.
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
        // `head[o]` does not depend on `dur[o]`, but the end of `o` does: a duration change
        // must seed the route successor, which the relinked window does not cover.
        let dchg = self.dur[o] != dur;
        self.dur[o] = dur;
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
        // The machine links are part of the state: copying them (memcpy) avoids a `relink_all`
        // over 1700 operations on every restore.
        self.mnx.copy_from_slice(&src.mnx);
        self.mpv.copy_from_slice(&src.mpv);
        for m in 0..self.n_mach {
            self.mseq[m].clear();
            self.mseq[m].extend_from_slice(&src.mseq[m]);
        }
        // The restored state bears no relation to the current order: everything is to be redone.
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

// ===========================================================================================
// Incremental evaluation and dynamic topological order. See the six invariants at the top of the
// file; each one is cited at the point of the code where it applies.
// ===========================================================================================

impl Graph {
    /// Advances a stamp and flattens its mark array when it wraps.
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

    /// Seeds an operation for both sweeps. The lists are emptied at different moments, hence two
    /// dedup stamps.
    ///
    /// Overflow: rather than truncating -- which would return a wrong makespan silently -- we
    /// fall back to the full recomputation, which is always correct.
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

    /// Incremental forward sweep: from the smallest seeded rank to the end (I6).
    fn eval_incr(&mut self) -> Option<u32> {
        let n = self.n_ops;
        if self.seed_n == 0 {
            // No arc has moved since the last evaluation: `head` and the makespan are still
            // those of the current graph.
            return Some(self.makespan);
        }
        Self::bump(&mut self.dstamp, &mut self.dmark);
        let ds = self.dstamp;
        let mut lo = u32::MAX;
        for k in 0..self.seed_n {
            let o = self.seed[k] as usize;
            self.dmark[o] = ds;
            let r = self.ord[o];
            if r < lo {
                lo = r;
            }
        }
        self.seed_n = 0;
        Self::bump(&mut self.sstamp, &mut self.sdup);
        for r in lo as usize..n {
            let o = self.oat[r] as usize;
            if self.dmark[o] != ds {
                continue;
            }
            let mut h = 0u32;
            let jp = self.jprev[o];
            if jp != NONE {
                let p = jp as usize;
                let f = self.head[p] + self.dur[p];
                if f > h {
                    h = f;
                }
            }
            let mp = self.mpv[o];
            if mp != NONE {
                let p = mp as usize;
                let f = self.head[p] + self.dur[p];
                if f > h {
                    h = f;
                }
            }
            // We only propagate if the end of `o` changes. Since `head[o]` does not contain
            // `dur[o]`, a duration change is seeded separately, in `attach` and `detach`.
            if h != self.head[o] {
                self.head[o] = h;
                let jn = self.jnext[o];
                if jn != NONE {
                    self.dmark[jn as usize] = ds;
                }
                let mn = self.mnx[o];
                if mn != NONE {
                    self.dmark[mn as usize] = ds;
                }
            }
        }
        // I5: the maximum is reached on a sink, and a sink is an end-of-route operation with no
        // machine successor. We sweep `n_jobs` operations, not `n_ops`.
        let mut ms = 0u32;
        for k in 0..self.job_last.len() {
            let o = self.job_last[k] as usize;
            if self.mnx[o] == NONE {
                let f = self.head[o] + self.dur[o];
                if f > ms {
                    ms = f;
                }
            }
        }
        self.makespan = ms;
        Some(ms)
    }

    /// An arc added by an operator, submitted to Pearce-Kelly.
    ///
    /// I2: at most one arc per move violates the order, and the graph without that arc stays
    /// ordered by `ord` -- which is exactly the precondition of Pearce-Kelly.
    ///   - Swap `a->u->v->b`: arcs added `(a,v)`, `(v,u)`, `(u,b)`; since
    ///     `ord[a] < ord[u] < ord[v] < ord[b]`, only `(v,u)` violates.
    ///   - Relocation: the bridge spans `o` so it is satisfied, and of the two new neighbours
    ///     `C`, `D` only one is on the wrong side, `C` and `D` being linked to `o` by the chain.
    ///   - Reassignment: `C -> D` was an arc before the insertion so `ord[C] < ord[D]`; two
    ///     violations would require `ord[D] < ord[o] < ord[C]`.
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
            x,
            y,
        );
        if cycle {
            // I2: nothing has been written into `ord`, so there is nothing to undo.
            self.cyc = true;
        }
    }

    /// Arcs added by a swap of positions `lo`/`hi` on machine `m`.
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

    /// Arcs added by a move of `o` from position `p` to `mpos[o]`.
    fn pk_after_relocate(&mut self, m: usize, o: usize, p: usize) {
        if self.full {
            return;
        }
        let ln = self.mseq[m].len();
        let ap = self.mpos[o] as usize;
        // Bridge left by the removal: satisfied by construction (I3), but submitted anyway, the
        // trivial test rejects it in two reads and the code stays uniform.
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

    /// Arcs added by the insertion of `o` on machine `m`.
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

/// Inserts the arc `x -> y` into the order `ord`/`oat` (Pearce-Kelly, 2007).
///
/// Returns `true` if the arc closes a cycle. In that case `ord` has not been touched: detection
/// precedes any write, which is what makes an undo journal unnecessary (I2).
///
/// R4 -- determinism. The two cones are sorted by `sort_unstable_by_key` on the rank. The ranks
/// form a permutation of `0..n`, so two equal keys are impossible and the instability of the sort
/// is unreachable. This is a hard condition: the `runtime_signature` is replayed at verification,
/// and an unstable sort on equal keys would be enough to make it diverge from one machine to
/// another.
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
    x: usize,
    y: usize,
) -> bool {
    if ord[x] < ord[y] {
        return false;
    }
    let lb = ord[y];
    let ub = ord[x];

    // Forward cone from `y`, bounded by the rank of `x`. Reaching `x` closes a cycle.
    let mut dfn = 0usize;
    let mut sl = 0usize;
    markf[y] = stamp;
    stk[sl] = y as u32;
    sl += 1;
    while sl > 0 {
        sl -= 1;
        let v = stk[sl] as usize;
        df[dfn] = v as u32;
        dfn += 1;
        let a = jn[v];
        if a != NONE {
            let w = a as usize;
            if ord[w] == ub {
                return true;
            }
            if markf[w] != stamp && ord[w] < ub {
                markf[w] = stamp;
                stk[sl] = w as u32;
                sl += 1;
            }
        }
        let b = mn[v];
        if b != NONE {
            let w = b as usize;
            if ord[w] == ub {
                return true;
            }
            if markf[w] != stamp && ord[w] < ub {
                markf[w] = stamp;
                stk[sl] = w as u32;
                sl += 1;
            }
        }
    }

    // Backward cone from `x`, bounded by the rank of `y`.
    let mut dbn = 0usize;
    let mut sl = 0usize;
    markb[x] = stamp;
    stk[sl] = x as u32;
    sl += 1;
    while sl > 0 {
        sl -= 1;
        let v = stk[sl] as usize;
        db[dbn] = v as u32;
        dbn += 1;
        let a = jp[v];
        if a != NONE {
            let w = a as usize;
            if markb[w] != stamp && ord[w] > lb {
                markb[w] = stamp;
                stk[sl] = w as u32;
                sl += 1;
            }
        }
        let b = mp[v];
        if b != NONE {
            let w = b as usize;
            if markb[w] != stamp && ord[w] > lb {
                markb[w] = stamp;
                stk[sl] = w as u32;
                sl += 1;
            }
        }
    }

    // The two cones put back in increasing rank order, then the ranks merged: the backward cone
    // first, the forward one next. Operations outside both cones keep their rank.
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
        ord[o] = r;
        oat[r as usize] = o as u32;
    }
    false
}
}
pub mod tabu {
// Local search for the hybrid track.
use super::graph::{Graph, GraphState, NONE};
use super::model::Model;
use rand::{rngs::SmallRng, Rng};

pub struct Budget<'a> {
    pub remaining: &'a dyn Fn() -> u64,
    pub floor: u64,
}

impl<'a> Budget<'a> {
    #[inline]
    pub fn left(&self) -> u64 {
        (self.remaining)().saturating_sub(self.floor)
    }
    #[inline]
    pub fn ok(&self, margin: u64) -> bool {
        self.left() > margin
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Mv {
    Swap(u32, u32),
    Relocate(u32, u32),
    Assign(u32, u32, u32, u32),
}

pub struct TabuCfg {
    pub tenure_min: usize,
    /// Only the sum `tenure_min + tenure_span` matters here: the tabu ring saturates at 16 and
    /// never comes back down (`ring_mark = 16.00` on 8 nonces out of 8), so the random draw
    /// inside the span never shortens an actual tenure. Kept because it is part of the
    /// hyperparameter interface.
    pub tenure_span: usize,
    pub max_assign_moves: usize,
    pub restart_after: u32,
    pub perturb_kicks: usize,
    pub use_n7: bool,
    pub max_moves: usize,
    pub verify_tries: usize,
}

impl Default for TabuCfg {
    fn default() -> Self {
        TabuCfg {
            tenure_min: 8,
            tenure_span: 8,
            max_assign_moves: 8,
            restart_after: 2000,
            perturb_kicks: 8,
            use_n7: true,
            max_moves: 0,
            verify_tries: 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TabuOut {
    pub best_ms: u32,
    pub iterations: u64,
    pub moves_seen: u64,
    pub moves_applied: u64,
    pub swap_applied: u64,
    pub relocate_applied: u64,
    pub assign_applied: u64,
    pub swap_improvements: u64,
    pub relocate_improvements: u64,
    pub assign_improvements: u64,
    pub perturb_improvements: u64,
}

/// Estimate for a swap of two adjacent operations on one machine.
///
/// Replaces the former `estimate(g, mv)`, whose two other arms could not be reached:
///   . the `Assign` arm was dead code, and always had been. Assign moves are pushed with the
///     estimate returned by `best_insertion` (`moves.push((est, mv))`); the one from `estimate`
///     was never asked for them, and there was no other caller. Its formula, `head[o] + qend[o]`,
///     is in fact the makespan of the current schedule when `o` is critical - it does not
///     describe the move. Removing it therefore fixes a latent inconsistency: the day a caller
///     had asked for it, it would have received a number unrelated to the reassignment asked for.
///   . the `Relocate` arm was the only other one called, from the N7 generation; that generation
///     now computes the estimate in place, with its insertion neighbours hoisted out of the loop.
/// Only `Swap` was left: the `match` on the move disappears from the hot path.
#[inline]
fn estimate_swap(g: &Graph, u: u32, v: u32) -> u32 {
    let (ui, vi) = (u as usize, v as usize);
    let mp_u = g.mprev(ui);
    let ms_v = g.mnext(vi);
    let rv = g.rp(g.jprev[vi]).max(g.rp(mp_u));
    let ru = (rv + g.dur[vi]).max(g.rp(g.jprev[ui]));
    let qu = g.qp(g.jnext[ui]).max(g.qp(ms_v));
    let qv = (qu + g.dur[ui]).max(g.qp(g.jnext[vi]));
    (rv + g.dur[vi] + qv).max(ru + g.dur[ui] + qu)
}

#[inline]
fn best_insertion(g: &Graph, o: usize, m: u32, d: u32) -> (u32, u32) {
    let seq = &g.mseq[m as usize];
    let rj = g.rp(g.jprev[o]);
    let qj = g.qp(g.jnext[o]);
    let n = seq.len();
    let mut best_est = u32::MAX;
    let mut best_at = 0u32;
    let mut r = rj;
    for at in 0..=n {
        if at > 0 {
            let p = seq[at - 1] as usize;
            let v = g.rend(p);
            if v > r {
                r = v;
            }
        }
        if r + d + qj >= best_est {
            break;
        }
        let q = if at < n {
            let sx = seq[at] as usize;
            let v = g.qend_of(sx);
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
    on_improve: &mut dyn FnMut(&Graph, u32),
) -> TabuOut {
    let mut best_ms = start_ms;
    let mut iterations = 0u64;
    let mut moves_seen = 0u64;
    let mut moves_applied = 0u64;
    let mut swap_applied = 0u64;
    let mut relocate_applied = 0u64;
    let mut assign_applied = 0u64;
    let mut swap_improvements = 0u64;
    let mut relocate_improvements = 0u64;
    let mut assign_improvements = 0u64;
    let mut perturb_improvements = 0u64;

    let mut blocks: Vec<(u32, u32, u32)> = Vec::with_capacity(64);
    let mut moves: Vec<(u32, Mv)> = Vec::with_capacity(512);
    let mut crit_flex: Vec<u32> = Vec::with_capacity(256);
    let tenure_cap = cfg.tenure_min + cfg.tenure_span + 2;
    let mut ring: Vec<(u8, u32, u32)> = Vec::with_capacity(tenure_cap);
    let mut ring_pos = 0usize;
    let mut tb_stamp: Vec<u32> = vec![0; g.n_ops];
    let mut tb_kind: Vec<u8> = vec![0; g.n_ops];
    let mut stamp: u32 = 0;

    let mut since_improve = 0u32;
    let mut iter_cost = per_eval_hint.saturating_mul(8).max(1);

    if g.evaluate().is_none() {
        return TabuOut { best_ms, ..TabuOut::default() };
    }

    loop {
        if !budget.ok(iter_cost.saturating_mul(2)) {
            break;
        }
        let before = (budget.remaining)();

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
                if cfg.use_n7 {
                    // The insertion neighbours do not depend on `t`, and the former `estimate`
                    // rediscovered them on every move - on top of a reload of `mach[o]`, of
                    // `mpos[o]` and of the `mseq[m]` pointer.
                    //
                    // Target `i`: the move is pushed only if t != i, and t runs over i..=j so
                    // t > i; and `mpos[seq[t]] == t` (the machine links are kept up to date by
                    // the operators, and `links_consistent()` checks it at the start of every
                    // `evaluate`). So `at = i < cur = t`: it is always the first branch of the
                    // former `estimate`, which gives (pred, succ) = (seq[i-1] or NONE, seq[i]).
                    // Target `j`: pushed only if t != j, so t < j, so at = j >= cur: always the
                    // second branch, (pred, succ) = (seq[j], seq[j+1] or NONE).
                    // The four neighbours are therefore loop invariants, and their `rp`/`qp`
                    // with them. The order of the additions is preserved: (r + dur) + q.
                    let len = seq.len();
                    let ri = g.rp(if i > 0 { seq[i - 1] as i32 } else { NONE });
                    let qi = g.qp(seq[i] as i32);
                    let rj = g.rp(seq[j] as i32);
                    let qjb = g.qp(if j + 1 < len { seq[j + 1] as i32 } else { NONE });
                    for t in i..=j {
                        let o = seq[t];
                        let oi = o as usize;
                        let ro = g.rp(g.jprev[oi]);
                        let qo = g.qp(g.jnext[oi]);
                        let d = g.dur[oi];
                        if t != i {
                            moves.push((ro.max(ri) + d + qo.max(qi), Mv::Relocate(o, i as u32)));
                        }
                        if t != j {
                            moves.push((ro.max(rj) + d + qo.max(qjb), Mv::Relocate(o, j as u32)));
                        }
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
            perturb(g, best_state, rng, cfg.perturb_kicks.max(1));
            since_improve = since_improve.saturating_add(1);
            iterations += 1;
            continue;
        }
        moves_seen += moves.len() as u64;
        if cfg.max_moves > 0 && moves.len() > cfg.max_moves {
            moves.truncate(cfg.max_moves);
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

        let tries = cfg.verify_tries.max(1).min(moves.len());
        let mut applied: Option<(u32, Mv)> = None;
        for t in 0..tries {
            let mut bi = usize::MAX;
            let mut bk = u32::MAX;
            // Scan over a slice: `moves[t..]` carries its own length, so the per-candidate
            // bound check disappears. Same start, same order, same comparisons.
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
                    cfg.tenure_min + rng.gen_range(0..=cfg.tenure_span),
                );
                applied = Some((ms, mv));
                break;
            }
            undo_move(g, mv, undo);
        }

        match applied {
            Some((ms, mv)) => {
                moves_applied += 1;
                match mv {
                    Mv::Swap(..) => swap_applied += 1,
                    Mv::Relocate(..) => relocate_applied += 1,
                    Mv::Assign(..) => assign_applied += 1,
                }
                if ms < best_ms {
                    best_ms = ms;
                    g.snapshot(best_state);
                    on_improve(g, best_ms);
                    match mv {
                        Mv::Swap(..) => swap_improvements += 1,
                        Mv::Relocate(..) => relocate_improvements += 1,
                        Mv::Assign(..) => assign_improvements += 1,
                    }
                    since_improve = 0;
                } else {
                    since_improve += 1;
                }
            }
            None => {
                let pms = perturb(g, best_state, rng, cfg.perturb_kicks.max(1));
                if let Some(ms) = pms {
                    if ms < best_ms {
                        best_ms = ms;
                        on_improve(g, best_ms);
                        perturb_improvements += 1;
                        // Without this the record descends but the state kept for it does not:
                        // the next `restore_from` returns the search to a point worse than its own
                        // best, and the state this function returns no longer matches `best_ms`.
                        g.snapshot(best_state);
                    }
                }
                since_improve = since_improve.saturating_add(1);
            }
        }

        if since_improve >= cfg.restart_after {
            g.restore_from(best_state);
            let pms = perturb(g, best_state, rng, cfg.perturb_kicks.max(1));
            if let Some(ms) = pms {
                if ms < best_ms {
                    best_ms = ms;
                    on_improve(g, best_ms);
                    perturb_improvements += 1;
                    g.snapshot(best_state);
                }
            }
            ring.clear();
            ring_pos = 0;
            since_improve = 0;
        }

        let spent = before.saturating_sub((budget.remaining)());
        if spent > iter_cost {
            iter_cost = spent;
        }
        iterations += 1;
    }

    g.restore_from(best_state);
    let _ = g.evaluate();
    TabuOut {
        best_ms,
        iterations,
        moves_seen,
        moves_applied,
        swap_applied,
        relocate_applied,
        assign_applied,
        swap_improvements,
        relocate_improvements,
        assign_improvements,
        perturb_improvements,
    }
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

fn perturb(g: &mut Graph, best_state: &GraphState, rng: &mut SmallRng, kicks: usize) -> Option<u32> {
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
// Constructive baseline for the hybrid track.
use tig_challenges::job_scheduling::{Challenge, Solution};
use anyhow::{anyhow, Result};
use rand::seq::SliceRandom;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::HashMap;

const WORK_MIN_WEIGHT: f64 = 0.3;

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

struct ScheduleResult {
    job_schedule: Vec<Vec<(usize, u32)>>,
    makespan: u32,
}

struct RestartResult {
    makespan: u32,
    rule: DispatchRule,
    random_top_k: usize,
    seed: u64,
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

fn run_dispatch_rule(
    challenge: &Challenge,
    job_products: &[usize],
    product_work_times: &[Vec<f64>],
    job_ops_len: &[usize],
    job_total_work: &[f64],
    rule: DispatchRule,
    random_top_k: Option<usize>,
    rng: Option<&mut SmallRng>,
) -> Result<ScheduleResult> {
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
    let random_top_k = random_top_k.unwrap_or(0);
    let mut rng = rng;
    let use_random = random_top_k > 1 && rng.is_some();

    while remaining_ops > 0 {
        // `(0..num_machines).filter(..)` yields in increasing order, so the collected vector is
        // already sorted; the sort that used to follow was a no-op, and the random branch
        // shuffles the vector right after anyway.
        let mut available_machines = (0..num_machines)
            .filter(|&m| machine_available_time[m] <= time)
            .collect::<Vec<usize>>();
        if use_random {
            available_machines.shuffle(rng.as_mut().unwrap());
        }

        let mut scheduled_any = false;
        for &machine in available_machines.iter() {
            let mut best_candidate: Option<Candidate> = None;

            if use_random {
                let mut candidates: Vec<Candidate> = Vec::new();

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

                    candidates.push(Candidate {
                        job,
                        priority,
                        machine_end,
                        proc_time,
                        flexibility,
                    });
                }

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
                    let k = random_top_k.min(candidates.len());
                    let pick = rng.as_mut().unwrap().gen_range(0..k);
                    best_candidate = Some(candidates[pick]);
                }
            } else {
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

                    if best_candidate
                        .as_ref()
                        .map_or(true, |best| better_candidate(&candidate, best, eps))
                    {
                        best_candidate = Some(candidate);
                    }
                }
            }

            if let Some(candidate) = best_candidate {
                let job = candidate.job;
                let product = job_products[job];
                let op_idx = job_next_op_idx[job];
                let op_times = &challenge.product_processing_times[product][op_idx];
                let proc_time = op_times[&machine];

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
    Ok(ScheduleResult {
        job_schedule,
        makespan,
    })
}

pub fn solve_challenge_with_effort(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    effort: usize,
) -> Result<()> {
    let (random_restarts, top_k) = if effort == 0 {
        (10usize, 0usize)
    } else if effort == 1 {
        (200usize, 2usize)
    } else {
        let random_restarts = 200usize.saturating_add(50usize.saturating_mul(effort));
        let top_k = 2usize.saturating_mul(effort);
        (random_restarts, top_k)
    };
    let local_search_tries = 1usize.saturating_add(3usize.saturating_mul(effort));
    solve_challenge_with_params(
        challenge,
        save_solution,
        random_restarts,
        top_k,
        local_search_tries,
    )
}

fn solve_challenge_with_params(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    random_restarts: usize,
    top_k: usize,
    local_search_tries: usize,
) -> Result<()> {
    let save_best = |best: &ScheduleResult| -> Result<()> {
        save_solution(&Solution {
            job_schedule: best.job_schedule.clone(),
        })
    };
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
            let work = avg * (1.0 - WORK_MIN_WEIGHT) + min * WORK_MIN_WEIGHT;
            work_ops.push(work);
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

    let mut best_result: Option<ScheduleResult> = None;
    for rule in rules.iter().copied() {
        let result = run_dispatch_rule(
            challenge,
            &job_products,
            &product_work_times,
            &job_ops_len,
            &job_total_work,
            rule,
            None,
            None,
        )?;
        let is_better = best_result
            .as_ref()
            .map_or(true, |best| result.makespan < best.makespan);
        if is_better {
            best_result = Some(result);
            if let Some(b) = best_result.as_ref() {
                save_best(b)?;
            }
        }
    }

    let mut best_result = best_result.ok_or_else(|| anyhow!("No valid schedule produced"))?;
    save_best(&best_result)?;

    let mut top_restarts: Vec<RestartResult> = Vec::new();

    if random_restarts > 0 {
        let mut rng = SmallRng::from_seed(challenge.seed);
        for _ in 1..=random_restarts {
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
                Some(random_top_k),
                Some(&mut local_rng),
            )?;
            let makespan = result.makespan;
            let is_better = makespan < best_result.makespan;
            if is_better {
                best_result = result;
                save_best(&best_result)?;
            }

            if top_k > 0 {
                top_restarts.push(RestartResult {
                    makespan,
                    rule,
                    random_top_k,
                    seed,
                });
                top_restarts.sort_by(|a, b| a.makespan.cmp(&b.makespan));
                if top_restarts.len() > top_k {
                    top_restarts.pop();
                }
            }
        }
    }

    if !top_restarts.is_empty() {
        for restart in top_restarts.iter() {
            for attempt in 0..local_search_tries {
                let local_seed = restart.seed.wrapping_add(attempt as u64 + 1);
                let mut local_rng = SmallRng::seed_from_u64(local_seed);
                let local_k = match attempt % 3 {
                    0 => restart.random_top_k,
                    1 => restart.random_top_k.saturating_sub(1),
                    _ => restart.random_top_k.saturating_add(1),
                }
                .max(2);
                let result = run_dispatch_rule(
                    challenge,
                    &job_products,
                    &product_work_times,
                    &job_ops_len,
                    &job_total_work,
                    restart.rule,
                    Some(local_k),
                    Some(&mut local_rng),
                )?;
                if result.makespan < best_result.makespan {
                    best_result = result;
                    save_best(&best_result)?;
                }
            }
        }
    }

    save_solution(&Solution {
        job_schedule: best_result.job_schedule,
    })?;
    Ok(())
}
}
