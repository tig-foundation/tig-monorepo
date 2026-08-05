// Tabu search over the critical-block neighbourhood of the disjunctive graph
// (Nowicki-Smutnicki TSAB, with Balas-Vazacopoulos block insertion) *plus* a
// Mastrolilli-Gambardella machine-reassignment layer for the flexible tracks.
//
// Neighbourhood, all restricted to critical operations -- reversing or
// relocating anything off the critical path cannot shorten the makespan:
//   * reverse the first / last machine arc of a block            (N5)
//   * move any operation of a block to the block's front or back (block insertion)
//   * move a critical operation to another eligible machine, at the BEST
//     insertion position on that machine (Mastrolilli-Gambardella)
//
// Candidate moves are ranked by an O(1) head/tail estimate rather than by a
// full O(n_ops) longest-path evaluation. Only the chosen move is actually
// evaluated, so an iteration costs ~2 evaluations instead of ~64. Any move that
// would create a cycle is caught by that evaluation and discarded, so reentrant
// routes need no special case.

use crate::job_scheduling::shuttle_bell::graph::{Graph, GraphState, NONE};
use crate::job_scheduling::shuttle_bell::model::Model;
use rand::{rngs::SmallRng, Rng};

/// Fuel-aware stopping controller. `remaining()` reads the runtime's
/// `__fuel_remaining` counter; `floor` is the reserve we refuse to dip below.
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
    /// reverse the machine arc u -> v (u, v adjacent on their machine)
    Swap(u32, u32),
    /// relocate operation o to index `at` within its own machine sequence
    Relocate(u32, u32),
    /// move operation o to machine m (duration d there) at index `at`
    Assign(u32, u32, u32, u32),
}

pub struct TabuCfg {
    pub tenure_min: usize,
    pub tenure_span: usize,
    /// Max number of critical flexible operations sampled for reassignment
    /// moves each iteration (0 disables the layer, usize::MAX = all).
    pub max_assign_moves: usize,
    pub restart_after: u32,
    pub perturb_kicks: usize,
    pub use_n7: bool,
    /// Cap on candidate moves generated per iteration (0 = unlimited).
    pub max_moves: usize,
    /// How many of the best-estimated moves to try before giving up on the
    /// iteration (each try costs one evaluation; retries only happen when a
    /// move turns out to create a cycle).
    pub verify_tries: usize,
    /// Also consider reinserting a critical operation at the best position on
    /// its *own* machine (MG's within-machine relocation).
    pub self_reinsert: bool,
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
            self_reinsert: false,
        }
    }
}

pub struct TabuOut {
    pub best_ms: u32,
    pub iters: u64,
    pub moves_seen: u64,
}

#[inline]
fn rp(g: &Graph, x: i32) -> u32 {
    if x == NONE {
        0
    } else {
        let x = x as usize;
        g.head[x] + g.dur[x]
    }
}

#[inline]
fn qp(g: &Graph, x: i32) -> u32 {
    if x == NONE {
        0
    } else {
        let x = x as usize;
        g.tail[x] + g.dur[x]
    }
}

/// O(1) lower-bound estimate of the makespan after applying `mv`, computed from
/// the current heads and tails (Taillard's formula for the reversal case).
fn estimate(g: &Graph, mv: Mv) -> u32 {
    match mv {
        Mv::Swap(u, v) => {
            let (ui, vi) = (u as usize, v as usize);
            let mp_u = g.mprev(ui);
            let ms_v = g.mnext(vi);
            let rv = rp(g, g.jprev[vi]).max(rp(g, mp_u));
            let ru = (rv + g.dur[vi]).max(rp(g, g.jprev[ui]));
            let qu = qp(g, g.jnext[ui]).max(qp(g, ms_v));
            let qv = (qu + g.dur[ui]).max(qp(g, g.jnext[vi]));
            (rv + g.dur[vi] + qv).max(ru + g.dur[ui] + qu)
        }
        Mv::Relocate(o, at) => {
            let oi = o as usize;
            let m = g.mach[oi] as usize;
            let at = at as usize;
            let cur = g.mpos[oi] as usize;
            let seq = &g.mseq[m];
            let (pred, succ) = if at < cur {
                // o is inserted immediately before seq[at]
                let w0 = seq[at] as i32;
                let pred = if at > 0 { seq[at - 1] as i32 } else { NONE };
                (pred, w0)
            } else {
                // o is inserted immediately after seq[at]
                let w = seq[at] as i32;
                let succ = if at + 1 < seq.len() {
                    seq[at + 1] as i32
                } else {
                    NONE
                };
                (w, succ)
            };
            let r = rp(g, g.jprev[oi]).max(rp(g, pred));
            let q = qp(g, g.jnext[oi]).max(qp(g, succ));
            r + g.dur[oi] + q
        }
        Mv::Assign(o, _m, d, _at) => {
            // Assign moves carry their estimate from generation time; this arm
            // is only reached if someone re-estimates one. Recompute cheaply.
            let oi = o as usize;
            let _ = d;
            g.head[oi] + g.dur[oi] + g.tail[oi]
        }
    }
}

/// Scan every insertion position of operation `o` on machine `m` (assumed
/// different from `o`'s current machine, so `o` does not appear in `mseq[m]`)
/// and return the position minimising the head/tail estimate.
///
/// This is the Mastrolilli-Gambardella reassignment evaluation: `head[x]` is
/// non-decreasing and `tail[x]` non-increasing along a machine sequence, so a
/// single pass finds the best splice point exactly under the approximation
/// that the rest of the graph keeps its current heads and tails.
#[inline]
fn best_insertion(g: &Graph, o: usize, m: u32, d: u32) -> (u32, u32) {
    let seq = &g.mseq[m as usize];
    let rj = rp(g, g.jprev[o]);
    let qj = qp(g, g.jnext[o]);
    let n = seq.len();
    let mut best_est = u32::MAX;
    let mut best_at = 0u32;
    let mut r = rj;
    for at in 0..=n {
        if at > 0 {
            let p = seq[at - 1] as usize;
            let v = g.head[p] + g.dur[p];
            if v > r {
                r = v;
            }
        }
        // `head` is non-decreasing along a machine sequence, so `r` only grows
        // from here on, and every `q` is at least `qj`: once this bound reaches
        // the incumbent, no later position can beat it.
        if r + d + qj >= best_est {
            break;
        }
        let q = if at < n {
            let s = seq[at] as usize;
            let v = g.tail[s] + g.dur[s];
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

/// Same, but for reinserting `o` at a different position on its own machine.
/// `o` is still present in the sequence, so it is skipped while scanning.
#[inline]
fn best_self_insertion(g: &Graph, o: usize) -> (u32, u32) {
    let m = g.mach[o] as usize;
    let cur = g.mpos[o] as usize;
    let seq = &g.mseq[m];
    let n = seq.len();
    let d = g.dur[o];
    let rj = rp(g, g.jprev[o]);
    let qj = qp(g, g.jnext[o]);
    let mut best_est = u32::MAX;
    let mut best_at = cur as u32;
    // positions expressed in the *original* index space, as `relocate_within`
    // expects (remove at `cur`, insert at `at`).
    for at in 0..n {
        if at == cur {
            continue;
        }
        // predecessor / successor once o sits immediately before seq[at]
        // (at < cur) or immediately after seq[at] (at > cur).
        let (pred, succ) = if at < cur {
            let pred = if at > 0 { seq[at - 1] as i32 } else { NONE };
            (pred, seq[at] as i32)
        } else {
            let succ = if at + 1 < n { seq[at + 1] as i32 } else { NONE };
            (seq[at] as i32, succ)
        };
        let r = rj.max(rp(g, pred));
        let q = qj.max(qp(g, succ));
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
    let mut iters = 0u64;
    let mut moves_seen = 0u64;

    let mut blocks: Vec<(u32, u32, u32)> = Vec::with_capacity(64);
    let mut moves: Vec<(u32, Mv)> = Vec::with_capacity(512);
    let mut crit_flex: Vec<u32> = Vec::with_capacity(256);
    let tenure_cap = cfg.tenure_min + cfg.tenure_span + 2;
    let mut ring: Vec<(u8, u32, u32)> = Vec::with_capacity(tenure_cap);
    let mut ring_pos = 0usize;
    // Index of `ring` by operation, so the overwhelmingly common "this move is
    // not tabu" answer costs one load instead of a scan of the whole ring.
    // `tb_kind[o]` is a bitmask of the move kinds forbidden for `o`; it is only
    // trusted when `tb_stamp[o]` carries the current stamp, which avoids having
    // to clear either array. Swap keys carry a second operation, so a set bit 0
    // still has to be confirmed against the ring -- which is rare.
    let mut tb_stamp: Vec<u32> = vec![0; g.n_ops];
    let mut tb_kind: Vec<u8> = vec![0; g.n_ops];
    let mut stamp: u32 = 0;

    let mut since_improve = 0u32;
    let mut iter_cost = per_eval_hint.saturating_mul(8).max(1);

    // The graph must arrive here already evaluated.
    if g.evaluate().is_none() {
        return TabuOut {
            best_ms,
            iters,
            moves_seen,
        };
    }

    loop {
        if !budget.ok(iter_cost.saturating_mul(2)) {
            break;
        }
        let before = (budget.remaining)();

        g.compute_tails();

        // ---- build the neighbourhood ----
        // One pass produces both the critical blocks and the critical flexible
        // operations the reassignment layer samples from.
        g.critical_blocks_flex(md, &mut blocks, &mut crit_flex, cfg.max_assign_moves > 0);
        moves.clear();
        for &(m, i, j) in blocks.iter() {
            let m = m as usize;
            let (i, j) = (i as usize, j as usize);
            let (u, v) = (g.mseq[m][i], g.mseq[m][i + 1]);
            if g.jnext[u as usize] != v as i32 {
                let mv = Mv::Swap(u, v);
                moves.push((estimate(g, mv), mv));
            }
            if j - i >= 2 {
                let (u, v) = (g.mseq[m][j - 1], g.mseq[m][j]);
                if g.jnext[u as usize] != v as i32 {
                    let mv = Mv::Swap(u, v);
                    moves.push((estimate(g, mv), mv));
                }
                if cfg.use_n7 {
                    for t in i..=j {
                        let o = g.mseq[m][t];
                        if t != i {
                            let mv = Mv::Relocate(o, i as u32);
                            moves.push((estimate(g, mv), mv));
                        }
                        if t != j {
                            let mv = Mv::Relocate(o, j as u32);
                            moves.push((estimate(g, mv), mv));
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
                if cfg.self_reinsert && g.mseq[g.mach[o] as usize].len() > 1 {
                    let (est, at) = best_self_insertion(g, o);
                    if est != u32::MAX {
                        let mv = Mv::Relocate(o as u32, at);
                        moves.push((est, mv));
                    }
                }
            }
        }

        if moves.is_empty() {
            perturb(g, best_state, rng, cfg.perturb_kicks.max(1));
            since_improve = since_improve.saturating_add(1);
            iters += 1;
            continue;
        }
        moves_seen += moves.len() as u64;
        if cfg.max_moves > 0 && moves.len() > cfg.max_moves {
            moves.truncate(cfg.max_moves);
        }

        // ---- pick by estimate, verify by evaluation ----
        // Refresh the per-operation index of the tabu ring.
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
        let mut applied: Option<u32> = None;
        for t in 0..tries {
            let mut bi = usize::MAX;
            let mut bk = u32::MAX;
            for x in t..moves.len() {
                let (est, mv) = moves[x];
                // A move that cannot beat the incumbent candidate cannot be
                // selected whatever its tabu status, so the tabu test is only
                // paid for the few moves that are actually in contention.
                if est >= bk {
                    continue;
                }
                if est >= best_ms && is_tabu(&ring, &tb_stamp, &tb_kind, stamp, mv) {
                    continue;
                }
                bk = est;
                bi = x;
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
                    on_improve(g, best_ms);
                    since_improve = 0;
                } else {
                    since_improve += 1;
                }
            }
            None => {
                perturb(g, best_state, rng, cfg.perturb_kicks.max(1));
                since_improve = since_improve.saturating_add(1);
            }
        }

        if since_improve >= cfg.restart_after {
            g.restore_from(best_state);
            perturb(g, best_state, rng, cfg.perturb_kicks.max(1));
            ring.clear();
            ring_pos = 0;
            since_improve = 0;
        }

        iters += 1;
        let spent = before.saturating_sub((budget.remaining)());
        if spent > iter_cost {
            iter_cost = spent;
        }
    }

    g.restore_from(best_state);
    let _ = g.evaluate();

    TabuOut {
        best_ms,
        iters,
        moves_seen,
    }
}

/// Apply a move, evaluate, and return (makespan-or-MAX, undo token).
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
    // Relocate / Assign keys are fully determined by (kind, op); only Swap
    // carries a partner that still has to be matched.
    if key.0 != 0 {
        return true;
    }
    ring.iter().any(|&e| e == key)
}

#[inline]
fn push_tabu(ring: &mut Vec<(u8, u32, u32)>, pos: &mut usize, mv: Mv, tenure: usize) {
    // Forbid the inverse of the move we just made.
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

/// Random adjacent swaps on random machines, used to escape a basin.
fn perturb(g: &mut Graph, best_state: &GraphState, rng: &mut SmallRng, kicks: usize) {
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
    if g.evaluate().is_none() {
        g.restore_from(best_state);
        // Re-evaluate: `compute_tails` and the machine-link cache both rely on
        // the scratch state matching the current sequencing, and a failed
        // evaluation leaves a truncated topological order behind.
        let _ = g.evaluate();
    }
}
