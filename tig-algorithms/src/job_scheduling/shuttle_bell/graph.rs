// Disjunctive graph: machine sequences + longest-path evaluation, critical
// path extraction, and the critical-block neighbourhood.

use crate::job_scheduling::shuttle_bell::model::Model;

pub const NONE: i32 = -1;

pub struct Graph {
    pub n_ops: usize,
    pub n_mach: usize,
    /// operation -> next / previous operation of the same job (-1 = none)
    pub jnext: Vec<i32>,
    pub jprev: Vec<i32>,
    /// operation -> currently assigned machine, and its duration there
    pub mach: Vec<u32>,
    pub dur: Vec<u32>,
    /// machine -> ordered list of operations
    pub mseq: Vec<Vec<u32>>,
    /// operation -> its index inside `mseq[mach[o]]`
    pub mpos: Vec<u32>,

    // ---- evaluation scratch (never reallocated in the hot loop) ----
    indeg: Vec<u8>,
    topo: Vec<u32>,
    stack: Vec<u32>,
    pub head: Vec<u32>,
    pub tail: Vec<u32>,
    /// Cached machine successor / predecessor of every operation, rebuilt by
    /// `evaluate()` in the same pass that computes the in-degrees. Resolving a
    /// machine link through `mseq`/`mpos` costs four dependent loads; this costs
    /// one, and the two hottest loops in the search (the topological longest
    /// path and the tail pass) touch it once per operation.
    ///
    /// Valid exactly while the sequencing state is the one `evaluate()` last
    /// saw, which is the case everywhere it is read: moves are applied and
    /// evaluated together, and nothing between an evaluation and the next move
    /// mutates `mseq`.
    mnx: Vec<i32>,
    mpv: Vec<i32>,
    pub makespan: u32,
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
            topo: Vec::with_capacity(n),
            stack: Vec::with_capacity(n),
            head: vec![0; n],
            tail: vec![0; n],
            mnx: vec![NONE; n],
            mpv: vec![NONE; n],
            makespan: u32::MAX,
        }
    }

    /// Machine successor of `o`. Reads the cache rebuilt by `evaluate()`.
    #[inline]
    pub fn mnext(&self, o: usize) -> i32 {
        self.mnx[o]
    }

    /// Machine predecessor of `o`. Reads the cache rebuilt by `evaluate()`.
    #[inline]
    pub fn mprev(&self, o: usize) -> i32 {
        self.mpv[o]
    }

    /// Seed the machine sequences from an explicit schedule (used to import the
    /// greedy baseline). Operations on each machine are ordered by start time.
    /// Returns false if the schedule does not match the model.
    pub fn load_schedule(&mut self, md: &Model, sched: &[Vec<(usize, u32)>]) -> bool {
        if sched.len() != md.n_jobs {
            return false;
        }
        for s in self.mseq.iter_mut() {
            s.clear();
        }
        // (start_time, op) per machine, then sort
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
        true
    }

    /// Longest-path evaluation. Returns `None` if the current machine sequences
    /// create a cycle (which makes the move that produced them infeasible).
    pub fn evaluate(&mut self) -> Option<u32> {
        let n = self.n_ops;
        self.stack.clear();
        self.topo.clear();
        {
            // One fused pass over the machine sequences. Every operation lives
            // in exactly one sequence, so walking the sequences visits each of
            // them exactly once -- and doing it in sequence order gives the
            // machine links for free, which the old code paid four dependent
            // loads for on every lookup. This single pass replaces the previous
            // three (in-degrees, head clear, source seeding).
            let Graph {
                ref mseq,
                ref mut mnx,
                ref mut mpv,
                ref mut head,
                ref mut indeg,
                ref jprev,
                ref mut stack,
                ..
            } = *self;
            for s in mseq.iter() {
                let len = s.len();
                for i in 0..len {
                    let o = s[i] as usize;
                    mnx[o] = if i + 1 < len { s[i + 1] as i32 } else { NONE };
                    mpv[o] = if i > 0 { s[i - 1] as i32 } else { NONE };
                    head[o] = 0;
                    let mut d = 0u8;
                    if jprev[o] != NONE {
                        d += 1;
                    }
                    if i > 0 {
                        d += 1;
                    }
                    indeg[o] = d;
                    if d == 0 {
                        stack.push(o as u32);
                    }
                }
            }
        }
        let mut ms = 0u32;
        while let Some(o) = self.stack.pop() {
            let o = o as usize;
            self.topo.push(o as u32);
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
                    self.stack.push(jn as u32);
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
                    self.stack.push(mn as u32);
                }
            }
        }
        if self.topo.len() != n {
            self.makespan = u32::MAX;
            return None;
        }
        self.makespan = ms;
        Some(ms)
    }

    /// Tail lengths (longest path from the end of an operation to the sink).
    /// Requires a successful `evaluate()` first.
    ///
    /// Walking `topo` in reverse means every successor of `o` has already been
    /// written when `o` is reached, and after a successful `evaluate()` `topo`
    /// holds all `n_ops` operations -- so every entry of `tail` is overwritten
    /// here and the zeroing pass this used to start with was dead work.
    pub fn compute_tails(&mut self) {
        // Defensive: every caller is preceded by a successful evaluation, but if
        // one ever is not, `topo` is short and the operations missing from it
        // would keep whatever the previous iteration left behind.
        if self.topo.len() != self.n_ops {
            for o in 0..self.n_ops {
                self.tail[o] = 0;
            }
        }
        for i in (0..self.topo.len()).rev() {
            let o = self.topo[i] as usize;
            let mut t = 0u32;
            let jn = self.jnext[o];
            if jn != NONE {
                let jn = jn as usize;
                let v = self.tail[jn] + self.dur[jn];
                if v > t {
                    t = v;
                }
            }
            let mn = self.mnx[o];
            if mn != NONE {
                let mn = mn as usize;
                let v = self.tail[mn] + self.dur[mn];
                if v > t {
                    t = v;
                }
            }
            self.tail[o] = t;
        }
    }

    #[inline]
    pub fn is_critical(&self, o: usize) -> bool {
        self.head[o] + self.dur[o] + self.tail[o] == self.makespan
    }

    /// Collect the critical blocks: maximal runs of operations that are
    /// consecutive on one machine and linked by critical arcs. Emits
    /// `(machine, first_index, last_index)` with `last > first`.
    ///
    /// This is the Nowicki-Smutnicki restriction: reversing an arc that is not
    /// inside a critical block can never shorten the makespan, so this cuts the
    /// neighbourhood from O(n_ops) to a few dozen moves without losing anything.
    /// `flex` additionally receives every critical operation that has more than
    /// one eligible machine -- the input to the reassignment layer. That used to
    /// be a separate full pass over all operations; every operation appears in
    /// exactly one machine sequence, so the block scan already visits each of
    /// them exactly once and the test rides along for free.
    pub fn critical_blocks_flex(
        &self,
        md: &Model,
        out: &mut Vec<(u32, u32, u32)>,
        flex: &mut Vec<u32>,
        want_flex: bool,
    ) {
        out.clear();
        flex.clear();
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

    /// Move operation `o` to index `at` within its own machine sequence.
    /// Returns the previous index; calling it again with that index undoes it.
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
        for i in lo..self.mseq[m].len() {
            let x = self.mseq[m][i] as usize;
            self.mpos[x] = i as u32;
        }
        p
    }

    /// Swap two operations that are adjacent in their machine sequence.
    #[inline]
    pub fn swap_adjacent(&mut self, u: u32, v: u32) {
        let m = self.mach[u as usize] as usize;
        let pu = self.mpos[u as usize] as usize;
        let pv = self.mpos[v as usize] as usize;
        self.mseq[m].swap(pu, pv);
        self.mpos[u as usize] = pv as u32;
        self.mpos[v as usize] = pu as u32;
    }

    /// Remove operation `o` from its machine sequence.
    fn detach(&mut self, o: usize) {
        let m = self.mach[o] as usize;
        let p = self.mpos[o] as usize;
        self.mseq[m].remove(p);
        for i in p..self.mseq[m].len() {
            let x = self.mseq[m][i] as usize;
            self.mpos[x] = i as u32;
        }
    }

    /// Insert operation `o` into machine `m` at index `at`, setting its
    /// duration for that machine.
    fn attach(&mut self, o: usize, m: u32, at: usize, dur: u32) {
        let mm = m as usize;
        let at = at.min(self.mseq[mm].len());
        self.mseq[mm].insert(at, o as u32);
        self.mach[o] = m;
        self.dur[o] = dur;
        for i in at..self.mseq[mm].len() {
            let x = self.mseq[mm][i] as usize;
            self.mpos[x] = i as u32;
        }
    }

    /// Reassign `o` to machine `m`, splicing it in at index `at`.
    /// Returns the previous (machine, index, duration) so the move can be undone.
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

    /// Materialise the schedule in the format `Challenge::evaluate_makespan`
    /// expects. Requires a successful `evaluate()`.
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

    /// Copy the sequencing state (not the scratch buffers) from `src`.
    pub fn restore_from(&mut self, src: &GraphState) {
        self.mach.copy_from_slice(&src.mach);
        self.dur.copy_from_slice(&src.dur);
        self.mpos.copy_from_slice(&src.mpos);
        for m in 0..self.n_mach {
            self.mseq[m].clear();
            self.mseq[m].extend_from_slice(&src.mseq[m]);
        }
    }

    pub fn snapshot(&self, dst: &mut GraphState) {
        dst.mach.copy_from_slice(&self.mach);
        dst.dur.copy_from_slice(&self.dur);
        dst.mpos.copy_from_slice(&self.mpos);
        for m in 0..self.n_mach {
            dst.mseq[m].clear();
            dst.mseq[m].extend_from_slice(&self.mseq[m]);
        }
    }
}

/// A saved sequencing state.
pub struct GraphState {
    pub mach: Vec<u32>,
    pub dur: Vec<u32>,
    pub mpos: Vec<u32>,
    pub mseq: Vec<Vec<u32>>,
}

impl GraphState {
    pub fn new(md: &Model) -> GraphState {
        GraphState {
            mach: vec![0; md.n_ops],
            dur: vec![0; md.n_ops],
            mpos: vec![0; md.n_ops],
            mseq: vec![Vec::new(); md.n_machines],
        }
    }
}
