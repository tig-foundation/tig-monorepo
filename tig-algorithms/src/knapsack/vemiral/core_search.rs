use tig_challenges::knapsack::{Challenge,Solution};
    const CB_SC: i64 = 64;
    const CB_INF: i64 = i64::MAX / 8;

    struct CbFlow {
        to: Vec<u32>, cap: Vec<i64>, head: Vec<i32>, nxt: Vec<i32>,
        level: Vec<i32>, it: Vec<i32>, q: Vec<u32>, stk: Vec<u32>, pe: Vec<i32>,
        tarc: Vec<i32>, pre: bool,
    }

    impl CbFlow {
        fn new() -> Self {
            CbFlow { to: Vec::new(), cap: Vec::new(), head: Vec::new(), nxt: Vec::new(),
                     level: Vec::new(), it: Vec::new(), q: Vec::new(), stk: Vec::new(), pe: Vec::new(),
                     tarc: Vec::new(), pre: false }
        }
        fn reset(&mut self, n: usize) {
            self.to.clear(); self.cap.clear(); self.nxt.clear();
            self.head.clear(); self.head.resize(n, -1);
            self.level.clear(); self.level.resize(n, -1);
            self.it.clear(); self.it.resize(n, -1);
            self.pe.clear(); self.pe.resize(n, -1);
            self.tarc.clear(); self.tarc.resize(n, -1);
        }
        #[inline]
        fn add(&mut self, u: usize, v: usize, c: i64) {
            let e = self.to.len();
            if v + 1 == self.tarc.len() { self.tarc[u] = e as i32; }
            self.to.push(v as u32); self.cap.push(c); self.nxt.push(self.head[u]); self.head[u] = e as i32;
            self.to.push(u as u32); self.cap.push(0); self.nxt.push(self.head[v]); self.head[v] = (e + 1) as i32;
        }
        fn bfs(&mut self, s: usize, t: usize) -> bool {
            for x in self.level.iter_mut() { *x = -1; }
            self.level[s] = 0; self.q.clear(); self.q.push(s as u32);
            let mut qi = 0usize;
            while qi < self.q.len() {
                let u = self.q[qi] as usize; qi += 1;
                if self.level[t] >= 0 && self.level[u] >= self.level[t] { break; }
                let mut e = self.head[u];
                while e != -1 {
                    let ei = e as usize; let v = self.to[ei] as usize;
                    if self.cap[ei] > 0 && self.level[v] < 0 { self.level[v] = self.level[u] + 1; self.q.push(v as u32); }
                    e = self.nxt[ei];
                }
            }
            self.level[t] >= 0
        }
        fn augment(&mut self, s: usize, t: usize) -> i64 {
            let mut total = 0i64;
            self.stk.clear(); self.stk.push(s as u32);
            loop {
                let u = match self.stk.last() { Some(&u) => u as usize, None => return total };
                if u == t {
                    let mut f = CB_INF; let mut cut = 1usize;
                    for k in 1..self.stk.len() {
                        let v = self.stk[k] as usize; let c = self.cap[self.pe[v] as usize];
                        if c < f { f = c; cut = k; }
                    }
                    for k in 1..self.stk.len() { let v = self.stk[k] as usize; let ei = self.pe[v] as usize; self.cap[ei] -= f; self.cap[ei ^ 1] += f; }
                    total += f;
                    self.stk.truncate(cut);
                    continue;
                }
                let mut adv = false;
                while self.it[u] != -1 {
                    let ei = self.it[u] as usize; let v = self.to[ei] as usize;
                    if self.cap[ei] > 0 && self.level[v] == self.level[u] + 1 {
                        self.pe[v] = ei as i32; self.stk.push(v as u32); adv = true; break;
                    }
                    self.it[u] = self.nxt[ei];
                }
                if !adv {
                    self.level[u] = -1;
                    self.stk.pop();
                    if let Some(&p) = self.stk.last() { let p = p as usize; let ei = self.it[p] as usize; self.it[p] = self.nxt[ei]; }
                }
            }
        }
        fn maxflow(&mut self, s: usize, t: usize) -> i64 {
            let mut flow = 0i64;
            if self.pre {
                let mut e0 = self.head[s];
                while e0 != -1 {
                    let ei = e0 as usize;
                    if self.cap[ei] > 0 {
                        let a = self.to[ei] as usize;
                        let mut ea = self.head[a];
                        while ea != -1 {
                            let eai = ea as usize;
                            if self.cap[eai] > 0 {
                                let b = self.to[eai] as usize;
                                let tb = self.tarc[b];
                                if tb >= 0 {
                                    let tbi = tb as usize;
                                    let mut f = self.cap[ei];
                                    if self.cap[eai] < f { f = self.cap[eai]; }
                                    if self.cap[tbi] < f { f = self.cap[tbi]; }
                                    if f > 0 {
                                        self.cap[ei] -= f; self.cap[ei ^ 1] += f;
                                        self.cap[eai] -= f; self.cap[eai ^ 1] += f;
                                        self.cap[tbi] -= f; self.cap[tbi ^ 1] += f;
                                        flow += f;
                                        if self.cap[ei] == 0 { break; }
                                    }
                                }
                            }
                            ea = self.nxt[eai];
                        }
                    }
                    e0 = self.nxt[ei];
                }
            }
            while self.bfs(s, t) {
                for i in 0..self.head.len() { self.it[i] = self.head[i]; }
                flow += self.augment(s, t);
            }
            flow
        }
        fn source_side(&mut self, s: usize, out: &mut Vec<bool>) {
            out.clear(); out.resize(self.head.len(), false);
            self.stk.clear(); self.stk.push(s as u32); out[s] = true;
            while let Some(u) = self.stk.pop() {
                let mut e = self.head[u as usize];
                while e != -1 {
                    let ei = e as usize; let v = self.to[ei] as usize;
                    if self.cap[ei] > 0 && !out[v] { out[v] = true; self.stk.push(v as u32); }
                    e = self.nxt[ei];
                }
            }
        }
    }

    struct CutEng {
        nb_s: Vec<u32>, nb_j: Vec<u32>, nb_q: Vec<i64>, kk: Vec<i64>, degsum: u64, nint: u64,
        st: Vec<i8>, pin: Vec<i64>, pmax: Vec<i64>, qu: Vec<u32>, rid: Vec<i32>, rl: Vec<u32>,
        off: Vec<u32>, to: Vec<u32>, rev: Vec<u32>, cap: Vec<i64>, fill: Vec<u32>,
        level: Vec<i32>, it: Vec<u32>, q: Vec<u32>, stk: Vec<u32>, pe: Vec<u32>, reach: Vec<bool>,
        tarc: Vec<u32>, pre: bool,
    }

    impl CutEng {
        fn new() -> Self {
            CutEng { nb_s: Vec::new(), nb_j: Vec::new(), nb_q: Vec::new(), kk: Vec::new(), degsum: 0, nint: 0,
                     st: Vec::new(), pin: Vec::new(), pmax: Vec::new(), qu: Vec::new(), rid: Vec::new(), rl: Vec::new(),
                     off: Vec::new(), to: Vec::new(), rev: Vec::new(), cap: Vec::new(), fill: Vec::new(),
                     level: Vec::new(), it: Vec::new(), q: Vec::new(), stk: Vec::new(), pe: Vec::new(), reach: Vec::new(),
                     tarc: Vec::new(), pre: true }
        }

        fn flow(&mut self, nn: usize, s: usize, t: usize) {
            if self.pre {
                for e0 in self.off[s] as usize..self.off[s + 1] as usize {
                    let a = self.to[e0] as usize;
                    for ea in self.off[a] as usize..self.off[a + 1] as usize {
                        if self.cap[e0] == 0 { break; }
                        let b = self.to[ea] as usize;
                        if b >= s || self.cap[ea] == 0 { continue; }
                        let tb = self.tarc[b];
                        if tb == u32::MAX { continue; }
                        let tb = tb as usize;
                        let mut f = self.cap[e0];
                        if self.cap[ea] < f { f = self.cap[ea]; }
                        if self.cap[tb] < f { f = self.cap[tb]; }
                        if f > 0 {
                            self.cap[e0] -= f; let r0 = self.rev[e0] as usize; self.cap[r0] += f;
                            self.cap[ea] -= f; let ra = self.rev[ea] as usize; self.cap[ra] += f;
                            self.cap[tb] -= f; let rb = self.rev[tb] as usize; self.cap[rb] += f;
                        }
                    }
                }
            }
            loop {
                for x in self.level[..nn].iter_mut() { *x = -1; }
                self.level[s] = 0; self.q.clear(); self.q.push(s as u32);
                let mut qi = 0usize;
                while qi < self.q.len() {
                    let u = self.q[qi] as usize; qi += 1;
                    if self.level[t] >= 0 && self.level[u] >= self.level[t] { break; }
                    let lu = self.level[u] + 1;
                    for e in self.off[u] as usize..self.off[u + 1] as usize {
                        let v = self.to[e] as usize;
                        if self.cap[e] > 0 && self.level[v] < 0 { self.level[v] = lu; self.q.push(v as u32); }
                    }
                }
                if self.level[t] < 0 { break; }
                for i in 0..nn { self.it[i] = self.off[i]; }
                self.stk.clear(); self.stk.push(s as u32);
                loop {
                    let u = match self.stk.last() { Some(&u) => u as usize, None => break };
                    if u == t {
                        let mut f = i64::MAX; let mut cut = 1usize;
                        for k in 1..self.stk.len() {
                            let c = self.cap[self.pe[self.stk[k] as usize] as usize];
                            if c < f { f = c; cut = k; }
                        }
                        for k in 1..self.stk.len() {
                            let ei = self.pe[self.stk[k] as usize] as usize;
                            self.cap[ei] -= f; let r = self.rev[ei] as usize; self.cap[r] += f;
                        }
                        self.stk.truncate(cut);
                        continue;
                    }
                    let end = self.off[u + 1];
                    let lu = self.level[u] + 1;
                    let mut adv = false;
                    while self.it[u] < end {
                        let ei = self.it[u] as usize; let v = self.to[ei] as usize;
                        if self.cap[ei] > 0 && self.level[v] == lu {
                            self.pe[v] = ei as u32; self.stk.push(v as u32); adv = true; break;
                        }
                        self.it[u] += 1;
                    }
                    if !adv {
                        self.level[u] = -1;
                        self.stk.pop();
                        if let Some(&p) = self.stk.last() { self.it[p as usize] += 1; }
                    }
                }
            }
            for x in self.reach[..nn].iter_mut() { *x = false; }
            self.stk.clear(); self.stk.push(s as u32); self.reach[s] = true;
            while let Some(u) = self.stk.pop() {
                let u = u as usize;
                for e in self.off[u] as usize..self.off[u + 1] as usize {
                    let v = self.to[e] as usize;
                    if self.cap[e] > 0 && !self.reach[v] { self.reach[v] = true; self.stk.push(v as u32); }
                }
            }
        }
    }

    impl CoreBnb {
        fn reached_target(&self) -> bool {
            self.target_val > 0 && self.best_val >= self.target_val
        }
        fn threshold(&self) -> i64 {
            (self.best_val + 1).max(self.target_val)
        }
        fn prep(&mut self) {
            let m = self.idx.len();
            let ce = &mut self.ce;
            ce.nb_s.clear(); ce.nb_j.clear(); ce.nb_q.clear(); ce.kk.clear();
            let mut degsum = 0u64; let mut nint = 0u64;
            for a in 0..m {
                ce.nb_s.push(ce.nb_j.len() as u32);
                let ia = self.idx[a] as usize;
                let (b0, b1) = (self.adj_s[ia] as usize, self.adj_s[ia + 1] as usize);
                degsum += (b1 - b0) as u64;
                let mut up = 0i64;
                for x in b0..b1 {
                    let pb = self.pos[self.adj_j[x] as usize];
                    if pb >= 0 {
                        ce.nb_j.push(pb as u32); ce.nb_q.push(CB_SC * self.adj_q[x]);
                        if pb > a as i32 { up += self.adj_q[x]; nint += 1; }
                    }
                }
                ce.kk.push(CB_SC * (self.lin_cur[ia] + up));
            }
            ce.nb_s.push(ce.nb_j.len() as u32);
            ce.degsum = degsum; ce.nint = nint;
            self.prep_ok = true;
        }

        fn eval_side(&mut self, lam: i64) -> (i64, i64) {
            if !self.prep_ok { self.prep(); }
            let m = self.idx.len();
            let mut nterm = 0u64;
            for a in 0..m { if lam * self.w[self.idx[a] as usize] != self.ce.kk[a] { nterm += 1; } }
            self.work += self.ce.degsum + 2 * (self.ce.nint + nterm);
            let mut v = 0i64; let mut wt = 0i64;
            for a in 0..m {
                if !self.side[a] { continue; }
                let ia = self.idx[a] as usize;
                v += CB_SC * self.lin_cur[ia] - lam * self.w[ia]; wt += self.w[ia];
                for x in self.ce.nb_s[a] as usize..self.ce.nb_s[a + 1] as usize {
                    let b = self.ce.nb_j[x] as usize;
                    if b > a && self.side[b] { v += self.ce.nb_q[x]; }
                }
            }
            (v, wt)
        }

        fn closure_n(&mut self, lam: i64, lo: Option<&[bool]>, hi: Option<&[bool]>) -> (i64, i64) {
            if !self.prep_ok { self.prep(); }
            let m = self.idx.len();
            let ce = &mut self.ce;
            ce.st.clear(); ce.st.resize(m, -1);
            if let Some(l) = lo { for a in 0..m { if l[a] { ce.st[a] = 1; } } }
            if let Some(h) = hi { for a in 0..m { if !h[a] { ce.st[a] = 0; } } }
            ce.pin.clear(); ce.pin.resize(m, 0); ce.pmax.clear(); ce.pmax.resize(m, 0);
            ce.qu.clear();
            for a in 0..m {
                if ce.st[a] != -1 { continue; }
                let ia = self.idx[a] as usize;
                let mut pi = CB_SC * self.lin_cur[ia] - lam * self.w[ia];
                let mut pm = pi;
                for x in ce.nb_s[a] as usize..ce.nb_s[a + 1] as usize {
                    let b = ce.nb_j[x] as usize; let sb = ce.st[b];
                    if sb == 1 { pi += ce.nb_q[x]; pm += ce.nb_q[x]; } else if sb == -1 { pm += ce.nb_q[x]; }
                }
                ce.pin[a] = pi; ce.pmax[a] = pm;
                if pi > 0 || pm <= 0 { ce.qu.push(a as u32); }
            }
            let mut qh = 0usize;
            while qh < ce.qu.len() {
                let a = ce.qu[qh] as usize; qh += 1;
                if ce.st[a] != -1 { continue; }
                if ce.pin[a] > 0 {
                    ce.st[a] = 1;
                    for x in ce.nb_s[a] as usize..ce.nb_s[a + 1] as usize {
                        let b = ce.nb_j[x] as usize;
                        if ce.st[b] == -1 { ce.pin[b] += ce.nb_q[x]; if ce.pin[b] > 0 { ce.qu.push(b as u32); } }
                    }
                } else if ce.pmax[a] <= 0 {
                    ce.st[a] = 0;
                    for x in ce.nb_s[a] as usize..ce.nb_s[a + 1] as usize {
                        let b = ce.nb_j[x] as usize;
                        if ce.st[b] == -1 { ce.pmax[b] -= ce.nb_q[x]; if ce.pmax[b] <= 0 { ce.qu.push(b as u32); } }
                    }
                }
            }
            ce.rid.clear(); ce.rid.resize(m, -1); ce.rl.clear();
            for a in 0..m { if ce.st[a] == -1 { ce.rid[a] = ce.rl.len() as i32; ce.rl.push(a as u32); } }
            let r = ce.rl.len();
            if r > 0 {
                let s = r; let t = r + 1; let nn = r + 2;
                ce.fill.clear(); ce.fill.resize(nn + 1, 0);
                let mut nterm_s = 0u32; let mut nterm_t = 0u32;
                for x in 0..r {
                    let a = ce.rl[x] as usize;
                    let mut e = ce.pin[a];
                    for y in ce.nb_s[a] as usize..ce.nb_s[a + 1] as usize {
                        let b = ce.nb_j[y] as usize;
                        if b > a && ce.st[b] == -1 { e += ce.nb_q[y]; ce.fill[x] += 1; ce.fill[ce.rid[b] as usize] += 1; }
                    }
                    ce.pmax[a] = e;
                    if e > 0 { ce.fill[x] += 1; nterm_s += 1; } else if e < 0 { ce.fill[x] += 1; nterm_t += 1; }
                }
                ce.fill[s] = nterm_s; ce.fill[t] = nterm_t;
                ce.off.clear(); ce.off.resize(nn + 1, 0);
                for u in 0..nn { ce.off[u + 1] = ce.off[u] + ce.fill[u]; }
                let na = ce.off[nn] as usize;
                ce.tarc.clear(); ce.tarc.resize(r, u32::MAX);
                ce.to.clear(); ce.to.resize(na, 0); ce.rev.clear(); ce.rev.resize(na, 0); ce.cap.clear(); ce.cap.resize(na, 0);
                for u in 0..nn { ce.fill[u] = ce.off[u]; }
                for x in 0..r {
                    let a = ce.rl[x] as usize;
                    let e = ce.pmax[a];
                    if e > 0 {
                        let p = ce.fill[s] as usize; ce.fill[s] += 1; let q = ce.fill[x] as usize; ce.fill[x] += 1;
                        ce.to[p] = x as u32; ce.cap[p] = e; ce.rev[p] = q as u32;
                        ce.to[q] = s as u32; ce.cap[q] = 0; ce.rev[q] = p as u32;
                    } else if e < 0 {
                        let p = ce.fill[x] as usize; ce.fill[x] += 1; let q = ce.fill[t] as usize; ce.fill[t] += 1;
                        ce.to[p] = t as u32; ce.cap[p] = -e; ce.rev[p] = q as u32; ce.tarc[x] = p as u32;
                        ce.to[q] = x as u32; ce.cap[q] = 0; ce.rev[q] = p as u32;
                    }
                    for y in ce.nb_s[a] as usize..ce.nb_s[a + 1] as usize {
                        let b = ce.nb_j[y] as usize;
                        if b > a && ce.st[b] == -1 {
                            let xb = ce.rid[b] as usize;
                            let p = ce.fill[x] as usize; ce.fill[x] += 1; let q = ce.fill[xb] as usize; ce.fill[xb] += 1;
                            ce.to[p] = xb as u32; ce.cap[p] = ce.nb_q[y]; ce.rev[p] = q as u32;
                            ce.to[q] = x as u32; ce.cap[q] = 0; ce.rev[q] = p as u32;
                        }
                    }
                }
                if ce.level.len() < nn { ce.level.resize(nn, -1); ce.it.resize(nn, 0); ce.pe.resize(nn, 0); ce.reach.resize(nn, false); }
                ce.flow(nn, s, t);
                for x in 0..r { if ce.reach[x] { ce.st[ce.rl[x] as usize] = 1; } }
            }
            self.side.clear(); self.side.resize(m, false);
            for a in 0..m { self.side[a] = self.ce.st[a] == 1; }
            self.eval_side(lam)
        }
    }

    struct CoreBnb {
        k: usize,
        w: Vec<i64>,
        lin_cur: Vec<i64>,
        cap: i64,
        best_val: i64,
        target_val: i64,
        best_sel: Vec<u8>,
        improved: bool,
        work: u64,
        work_cap: u64,
        lam_iters: usize,
        initial_lam: i64,
        wcut: u64,
        adj_s: Vec<u32>, adj_j: Vec<u32>, adj_q: Vec<i64>,
        pos: Vec<i32>,
        state: Vec<i8>,
        fl: CbFlow,
        idx: Vec<u32>,
        side: Vec<bool>,
        ce: CutEng, prep_ok: bool, eng: usize,
    }

    impl CoreBnb {
        fn closure(&mut self, lam: i64) -> (i64, i64) {
            if self.eng == 2 { return self.closure_n(lam, None, None); }
            let m = self.idx.len();
            let s = m; let t = m + 1;
            self.fl.reset(m + 2);
            let mut cst = 0i64;
            for a in 0..m {
                let ia = self.idx[a] as usize;
                let mut e = lam * self.w[ia] - CB_SC * self.lin_cur[ia];
                let (b0, b1) = (self.adj_s[ia] as usize, self.adj_s[ia + 1] as usize);
                for x in b0..b1 {
                    let pb = self.pos[self.adj_j[x] as usize];
                    if pb > a as i32 { e -= CB_SC * self.adj_q[x]; }
                }
                if e > 0 { self.fl.add(a, t, e); } else if e < 0 { cst += e; self.fl.add(s, a, -e); }
            }
            for a in 0..m {
                let ia = self.idx[a] as usize;
                let (b0, b1) = (self.adj_s[ia] as usize, self.adj_s[ia + 1] as usize);
                self.work += (b1 - b0) as u64;
                for x in b0..b1 {
                    let pb = self.pos[self.adj_j[x] as usize];
                    if pb > a as i32 { self.fl.add(a, pb as usize, CB_SC * self.adj_q[x]); }
                }
            }
            let cut = self.fl.maxflow(s, t);
            self.work += self.fl.to.len() as u64;
            let mut side = Vec::new();
            self.fl.source_side(s, &mut side);
            side.truncate(m);
            let mut wt = 0i64;
            for a in 0..m { if side[a] { wt += self.w[self.idx[a] as usize]; } }
            self.side = side;
            (-(cst + cut), wt)
        }

        fn try_incumbent(&mut self, base: i64) {
            let mut v = base;
            let m = self.idx.len();
            for a in 0..m {
                if !self.side[a] { continue; }
                let ia = self.idx[a] as usize;
                v += self.lin_cur[ia];
                let (b0, b1) = (self.adj_s[ia] as usize, self.adj_s[ia + 1] as usize);
                for x in b0..b1 {
                    let pb = self.pos[self.adj_j[x] as usize];
                    if pb > a as i32 && self.side[pb as usize] { v += self.adj_q[x]; }
                }
            }
            if v > self.best_val {
                self.best_val = v;
                self.improved = true;
                for i in 0..self.k { self.best_sel[i] = (self.state[i] == 1) as u8; }
                for a in 0..m { if self.side[a] { self.best_sel[self.idx[a] as usize] = 1; } }
            }
        }

        fn rebuild_idx(&mut self) {
            self.prep_ok = false;
            self.idx.clear();
            for i in 0..self.k {
                if self.state[i] == -1 { self.pos[i] = self.idx.len() as i32; self.idx.push(i as u32); }
                else { self.pos[i] = -1; }
            }
        }

        fn bound_at(&mut self, base: i64, c: i64, lam: i64) -> i64 {
            self.rebuild_idx();
            if self.idx.is_empty() {
                if base > self.best_val { self.side = Vec::new(); self.try_incumbent(base); }
                return base * CB_SC;
            }
            let (v, wt) = self.closure(lam);
            if wt <= c { self.try_incumbent(base); }
            base * CB_SC + lam * c + v
        }

        fn root_lambda(&mut self, base: i64, c: i64) -> (i64, i64) {
            let mut best = i64::MAX;
            let mut bl = 0i64;
            let seed = self.initial_lam.max(1);
            let mut lo = 0i64;
            let mut hi = seed;
            self.rebuild_idx();
            let (v, wt) = self.closure(seed);
            if wt <= c { self.try_incumbent(base); }
            let bound = base * CB_SC + seed * c + v;
            if bound < best { best = bound; bl = seed; }
            if self.reached_target() || (self.target_val > 0 && best < self.target_val * CB_SC) || self.work >= self.work_cap { return (best, bl); }
            let mut step = (seed / 16).max(1);
            if wt > c {
                lo = seed;
                loop {
                    hi = lo.saturating_add(step);
                    self.rebuild_idx();
                    let (v, wt) = self.closure(hi);
                    if wt <= c { self.try_incumbent(base); }
                    let bound = base * CB_SC + hi * c + v;
                    if bound < best { best = bound; bl = hi; }
                    if self.reached_target() || (self.target_val > 0 && best < self.target_val * CB_SC) || self.work >= self.work_cap { return (best, bl); }
                    if wt <= c || hi > (1i64 << 40) { break; }
                    lo = hi;
                    step *= 2;
                }
            } else {
                loop {
                    lo = (hi - step).max(0);
                    self.rebuild_idx();
                    let (v, wt) = self.closure(lo);
                    if wt <= c { self.try_incumbent(base); }
                    let bound = base * CB_SC + lo * c + v;
                    if bound < best { best = bound; bl = lo; }
                    if self.reached_target() || (self.target_val > 0 && best < self.target_val * CB_SC) || self.work >= self.work_cap { return (best, bl); }
                    if wt > c || lo == 0 { break; }
                    hi = lo;
                    step *= 2;
                }
            }
            while hi - lo > 1 {
                let mid = lo + (hi - lo) / 2;
                self.rebuild_idx();
                let (v, wt) = self.closure(mid);
                if wt <= c { self.try_incumbent(base); }
                let bound = base * CB_SC + mid * c + v;
                if bound < best { best = bound; bl = mid; }
                if self.reached_target() || (self.target_val > 0 && best < self.target_val * CB_SC) { return (best, bl); }
                if wt > c { lo = mid; } else { hi = mid; }
                if self.work >= self.work_cap { break; }
            }
            (best, bl)
        }

        fn set_in(&mut self, i: usize) {
            self.state[i] = 1;
            let (b0, b1) = (self.adj_s[i] as usize, self.adj_s[i + 1] as usize);
            for x in b0..b1 { let j = self.adj_j[x] as usize; self.lin_cur[j] += self.adj_q[x]; }
        }
        fn unset_in(&mut self, i: usize) {
            let (b0, b1) = (self.adj_s[i] as usize, self.adj_s[i + 1] as usize);
            for x in b0..b1 { let j = self.adj_j[x] as usize; self.lin_cur[j] -= self.adj_q[x]; }
            self.state[i] = -1;
        }

        fn root_fix(&mut self, base: &mut i64, c: &mut i64, lam: i64, passes: usize) {
            for _ in 0..passes {
                let mut changed = false;
                for i in 0..self.k {
                    if self.state[i] != -1 { continue; }
                    if self.work >= self.work_cap { return; }
                    let thr = (self.best_val + 1) * CB_SC;
                    let out_ok = if self.w[i] > *c { true } else {
                        let add = self.lin_cur[i];
                        self.set_in(i);
                        let b = self.bound_at(*base + add, *c - self.w[i], lam);
                        self.unset_in(i);
                        b < (self.best_val + 1) * CB_SC
                    };
                    let _ = thr;
                    if out_ok { self.state[i] = 0; changed = true; continue; }
                    self.state[i] = 0;
                    let b = self.bound_at(*base, *c, lam);
                    self.state[i] = -1;
                    if b < (self.best_val + 1) * CB_SC {
                        let add = self.lin_cur[i];
                        self.set_in(i);
                        *base += add; *c -= self.w[i];
                        changed = true;
                    }
                }
                if !changed { break; }
            }
        }

        fn closure_k(&mut self, lam: i64, known: &Vec<(i64, Vec<bool>, u8)>) -> (i64, i64) {
            self.work += self.wcut;
            if self.eng != 2 { return self.closure(lam); }
            let m = self.idx.len();
            let mut lo: Vec<bool> = Vec::new(); let mut hi: Vec<bool> = Vec::new();
            for kv in known.iter() {
                if kv.2 == 1 && kv.0 == lam {
                    self.side.clear(); self.side.extend_from_slice(&kv.1);
                    return self.eval_side(lam);
                }
            }
            for kv in known.iter() {
                if (kv.2 == 1 && kv.0 > lam) || (kv.2 == 3 && kv.0 >= lam) {
                    if lo.is_empty() { lo.extend_from_slice(&kv.1); } else { for a in 0..m { lo[a] |= kv.1[a]; } }
                }
                if (kv.2 == 1 && kv.0 < lam) || (kv.2 == 2 && kv.0 <= lam) {
                    if hi.is_empty() { hi.extend_from_slice(&kv.1); } else { for a in 0..m { hi[a] &= kv.1[a]; } }
                }
            }
            let lo_o = if lo.is_empty() { None } else { Some(&lo[..]) };
            let hi_o = if hi.is_empty() { None } else { Some(&hi[..]) };
            self.closure_n(lam, lo_o, hi_o)
        }

        fn dfs(&mut self, base: i64, c: i64, lam_in: i64, li: usize, mut known: Vec<(i64, Vec<bool>, u8)>) {
            if self.work >= self.work_cap || self.reached_target() { return; }
            self.prep_ok = false;
            self.idx.clear();
            for i in 0..self.k {
                if self.state[i] == -1 { self.pos[i] = self.idx.len() as i32; self.idx.push(i as u32); }
                else { self.pos[i] = -1; }
            }
            self.work += self.k as u64;
            if self.idx.is_empty() {
                if base>self.best_val {self.side.clear();self.try_incumbent(base);}
                return;
            }
            let thr = self.threshold() * CB_SC;
            let bsc = base * CB_SC;
            let mut n_cuts = 1usize;
            let (v, wt) = self.closure_k(lam_in, &known);
            if self.eng == 2 { known.push((lam_in, self.side.clone(), 1)); }
            let mut bound = bsc + lam_in * c + v;
            if wt <= c { self.try_incumbent(base); }
            if self.reached_target() { return; }
            if bound < thr { return; }
            let mut best_lam = lam_in;
            let mut best_side = self.side.clone();
            let mut lo; let mut hi;
            if wt > c {
                lo = lam_in;
                let mut step = 1i64.max(lam_in / 16);
                loop {
                    hi = lam_in + step;
                    let (v2, w2) = self.closure_k(hi, &known); n_cuts += 1;
                    if self.eng == 2 { known.push((hi, self.side.clone(), 1)); }
                    if self.work >= self.work_cap || self.reached_target() { return; }
                    let b2 = bsc + hi * c + v2;
                    if w2 <= c { self.try_incumbent(base); }
                    if self.reached_target() { return; }
                    if b2 < bound { bound = b2; best_lam = hi; best_side = self.side.clone(); }
                    if bound < self.threshold() * CB_SC { return; }
                    if w2 <= c || n_cuts > li || step > (1i64 << 40) { break; }
                    lo = hi; step *= 2;
                }
            } else {
                hi = lam_in; lo = lam_in;
                if lam_in > 0 {
                    let mut step = 1i64.max(lam_in / 16);
                    loop {
                        lo = (lam_in - step).max(0);
                        let (v2, w2) = self.closure_k(lo, &known); n_cuts += 1;
                        if self.eng == 2 { known.push((lo, self.side.clone(), 1)); }
                        if self.work >= self.work_cap || self.reached_target() { return; }
                        let b2 = bsc + lo * c + v2;
                        if w2 <= c { self.try_incumbent(base); }
                    if self.reached_target() { return; }
                        if b2 < bound { bound = b2; best_lam = lo; best_side = self.side.clone(); }
                        if bound < self.threshold() * CB_SC { return; }
                        if w2 > c || lo == 0 || n_cuts > li { break; }
                        hi = lo; step *= 2;
                    }
                }
            }
            while hi - lo > 1 && n_cuts <= li {
                let mid = lo + (hi - lo) / 2;
                let (v2, w2) = self.closure_k(mid, &known); n_cuts += 1;
                if self.eng == 2 { known.push((mid, self.side.clone(), 1)); }
                if self.work >= self.work_cap || self.reached_target() { return; }
                let b2 = bsc + mid * c + v2;
                if w2 <= c { self.try_incumbent(base); }
                    if self.reached_target() { return; }
                if b2 < bound { bound = b2; best_lam = mid; best_side = self.side.clone(); }
                if bound < self.threshold() * CB_SC { return; }
                if w2 > c { lo = mid; } else { hi = mid; }
            }
            let _ = thr;
            let mut bi = usize::MAX; let mut bw = -1i64;
            for a in 0..self.idx.len() {
                if best_side[a] { let i = self.idx[a] as usize; if self.w[i] > bw { bw = self.w[i]; bi = i; } }
            }
            if bi == usize::MAX {
                for a in 0..self.idx.len() { let i = self.idx[a] as usize; if self.w[i] > bw { bw = self.w[i]; bi = i; } }
            }
            let mut kin: Vec<(i64, Vec<bool>, u8)> = Vec::new();
            let mut kout: Vec<(i64, Vec<bool>, u8)> = Vec::new();
            if self.eng == 2 {
                let mut pb = 0usize;
                for a in 0..self.idx.len() { if self.idx[a] as usize == bi { pb = a; } }
                let st = if known.len() > 12 { known.len() - 12 } else { 0 };
                for kv in known[st..].iter() {
                    let mut v = kv.1.clone(); let had = v.remove(pb);
                    if kv.2 == 1 {
                        kin.push((kv.0, v.clone(), if had { 1 } else { 3 }));
                        kout.push((kv.0, v, if had { 2 } else { 1 }));
                    } else if kv.2 == 3 { kin.push((kv.0, v, 3)); }
                    else { kout.push((kv.0, v, 2)); }
                }
            }
            if self.w[bi] <= c {
                let add = self.lin_cur[bi];
                self.state[bi] = 1;
                let (b0, b1) = (self.adj_s[bi] as usize, self.adj_s[bi + 1] as usize);
                for x in b0..b1 { let j = self.adj_j[x] as usize; self.lin_cur[j] += self.adj_q[x]; }
                self.dfs(base + add, c - self.w[bi], best_lam, self.lam_iters, kin);
                for x in b0..b1 { let j = self.adj_j[x] as usize; self.lin_cur[j] -= self.adj_q[x]; }
            }
            self.state[bi] = 0;
            self.dfs(base, c, best_lam, self.lam_iters, kout);
            self.state[bi] = -1;
        }
    }


fn improve_given(ch: &Challenge, sol: &Solution, k: usize, work: u64, mode: usize, chosen: Option<Vec<usize>>, target_value: i64, hint_lambda: Option<i64>) -> (Solution, u64, i64) {
    let n=ch.num_items;
    let mut bits=vec![false;n];
    let mut c:Vec<i64>=ch.values.iter().map(|&x|x as i64).collect();
    for &i in &sol.items { bits[i]=true; for j in 0..n { c[j]+=ch.interaction_values[i][j] as i64; } }
    let mut used:Vec<usize>=(0..n).filter(|&i|bits[i]).collect();
    let mut unused:Vec<usize>=(0..n).filter(|&i|!bits[i]).collect();
    used.sort_by(|&a,&b|(c[a]*ch.weights[b] as i64).cmp(&(c[b]*ch.weights[a] as i64)).then(a.cmp(&b)));
    unused.sort_by(|&a,&b|(c[b]*ch.weights[a] as i64).cmp(&(c[a]*ch.weights[b] as i64)).then(a.cmp(&b)));
    let mut core:Vec<usize>=if mode==1 || mode==2 || mode==40 || mode==41 || mode==42 {(0..n).collect()}else{used.iter().take(k/2).chain(unused.iter().take(k-k/2)).copied().collect()};
    if mode>=3 && mode<40{
        let threshold=[40,80,160,320][((mode-3)/8).min(3)] as i32;
        let rank=(mode-3)%8;
        let mut parent:Vec<usize>=(0..n).collect();
        fn root(p:&mut [usize],mut i:usize)->usize{while p[i]!=i{p[i]=p[p[i]];i=p[i];}i}
        for i in 0..n{for j in i+1..n{if ch.interaction_values[i][j]>=threshold{let a=root(&mut parent,i);let b=root(&mut parent,j);if a!=b{parent[a]=b;}}}}
        let mut groups=vec![Vec::new();n];
        for i in 0..n{let r=root(&mut parent,i);groups[r].push(i);}
        let lam=c[used[0]] as f64/ch.weights[used[0]] as f64;
        let mut ranked=Vec::new();
        for (g,ids) in groups.iter().enumerate(){
            if ids.len()<2 || ids.len()>96 {continue;}
            let add:Vec<usize>=ids.iter().filter(|&&i|!bits[i]).copied().collect();
            if add.len()<2 {continue;}
            let mut score=0.0;
            for (a,&i) in add.iter().enumerate(){score+=c[i] as f64-lam*ch.weights[i] as f64;for &j in &add[a+1..]{score+=ch.interaction_values[i][j] as f64;}}
            ranked.push((score,g));
        }
        ranked.sort_by(|a,b|b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        if rank<7 {if let Some(&(_,g))=ranked.get(rank){core.extend_from_slice(&groups[g]);}}
        else {for &(_,g) in ranked.iter().take(8){core.extend_from_slice(&groups[g]);}}
    }
    if let Some(given)=chosen{core=given;}
    core.sort_unstable();core.dedup();
    let k=core.len();
    let mut is_core=vec![false;n];for &i in &core {is_core[i]=true;}
    let fixed:Vec<usize>=sol.items.iter().filter(|&&i|!is_core[i]).copied().collect();
    let wf:i64=fixed.iter().map(|&i|ch.weights[i] as i64).sum();
    let mut vf=0i64;
    for (a,&i) in fixed.iter().enumerate(){vf+=ch.values[i] as i64;for &j in &fixed[a+1..]{vf+=ch.interaction_values[i][j] as i64;}}
    let value=ch.evaluate_total_value(sol).unwrap() as i64;
    let mut lin=Vec::new();let mut w=Vec::new();let mut adj_s=Vec::new();let mut adj_j=Vec::new();let mut adj_q=Vec::new();
    for (a,&i) in core.iter().enumerate(){
        lin.push(ch.values[i] as i64+fixed.iter().map(|&j|ch.interaction_values[i][j] as i64).sum::<i64>());
        w.push(ch.weights[i] as i64);adj_s.push(adj_j.len() as u32);
        for (b,&j) in core.iter().enumerate(){if b!=a && ch.interaction_values[i][j]>0 {adj_j.push(b as u32);adj_q.push(ch.interaction_values[i][j] as i64);}}
    }
    adj_s.push(adj_j.len() as u32);
    let mut bb=CoreBnb{k,w,lin_cur:lin,cap:ch.max_weight as i64-wf,best_val:value-vf,target_val:if target_value > 0 { target_value-vf } else { 0 },best_sel:core.iter().map(|&i|bits[i] as u8).collect(),improved:false,work:0,work_cap:work,lam_iters:4,initial_lam:used.first().map(|&i|CB_SC*c[i]/(ch.weights[i] as i64).max(1)).unwrap_or(CB_SC*(target_value-vf).max(value-vf).max(1)/(ch.max_weight as i64-wf).max(1)),wcut:0,adj_s,adj_j,adj_q,pos:vec![-1;k],state:vec![-1;k],fl:CbFlow::new(),idx:Vec::new(),side:Vec::new(),ce:CutEng::new(),prep_ok:false,eng:2};
    let cap=bb.cap;
    let (bound,lam)=if mode==42 {let lam=hint_lambda.unwrap().max(0);(bb.bound_at(0,cap,lam),lam)} else {bb.root_lambda(0,cap)};
    let gap=bound/CB_SC-(value-vf);
    let target_done=target_value > 0 && (bb.reached_target() || bound < bb.target_val * CB_SC);
    if !target_done && bb.work<work && mode!=2 && mode!=40 && mode!=41 && mode!=42 && gap>0 { bb.dfs(0,cap,lam,8,Vec::new()); }
    if mode==40 && !target_done {
        bb.eng=1;bb.rebuild_idx();let (lag,_)=bb.closure(lam);
        let bound2=lag+lam*cap;
        assert_eq!(bound2,bound,"cut engines disagree");
        if target_value > 0 {
            assert!(fixed.is_empty());
            let allowance=bound2-target_value*CB_SC;
            assert!(allowance>=0);
            let ds=widest(&bb.fl,k,false);let dt=widest(&bb.fl,k+1,true);
            let mut forced_in=Vec::new();let mut free=Vec::new();
            for a in 0..k {
                if ds[a]>allowance {forced_in.push(core[a]);}
                else if dt[a]>allowance {} else {free.push(core[a]);}
            }
            eprintln!("target fixing: in={}, free={}, slack={}",forced_in.len(),free.len(),allowance/CB_SC);
            let weight:u64=forced_in.iter().map(|&i|ch.weights[i] as u64).sum();
            if weight>ch.max_weight as u64 {return (sol.clone(),bb.work,gap);}
            let seed=Solution{items:forced_in};
            if ch.evaluate_total_value(&seed).unwrap() as i64>=target_value {return (seed,bb.work,gap);}
            let remaining=work.saturating_sub(bb.work);
            if remaining==0 {return (sol.clone(),bb.work,gap);}
            let (candidate,spent,_)=improve_given(ch,&seed,free.len(),remaining,0,Some(free),target_value,None);
            let out=if ch.evaluate_total_value(&candidate).unwrap() as i64>value {candidate}else{sol.clone()};
            return (out,bb.work.saturating_add(spent),gap);
        }
        let allowance=bound2-(value-vf)*CB_SC;
        let ds=widest(&bb.fl,k,false);let dt=widest(&bb.fl,k+1,true);
        let mut free=Vec::new();let mut forced=0;
        for a in 0..k {
            if ds[a]>allowance {assert!(bits[core[a]]);forced+=1;}
            else if dt[a]>allowance {assert!(!bits[core[a]]);forced+=1;}
            else {free.push(core[a]);}
        }
        eprintln!("global bound: gap={gap}, lambda={}, forced={forced}, free={}",lam as f64/CB_SC as f64,free.len());
        return improve_given(ch,sol,free.len(),work,0,Some(free),target_value,None);
    }
    let mut items=fixed;
    for (a,&i) in core.iter().enumerate(){if bb.best_sel[a]!=0 {items.push(i);}}
    let out=Solution{items};
    assert!(ch.evaluate_total_value(&out).unwrap() as i64>=value);
    (out,bb.work,gap)
}

pub fn improve_to_target(ch: &Challenge, sol: &Solution, work: u64, target_value: i64) -> (Solution, u64, i64) {
    if ch.evaluate_total_value(sol).unwrap() as i64 >= target_value {
        return (sol.clone(),0,0);
    }
    improve_given(ch,sol,ch.num_items,work,40,None,target_value,None)
}

fn widest(fl:&CbFlow,source:usize,reverse:bool)->Vec<i64>{
 let mut dist=vec![0i64;fl.head.len()];dist[source]=i64::MAX;
 let mut heap=std::collections::BinaryHeap::new();heap.push((i64::MAX,source));
 while let Some((d,u))=heap.pop(){
  if d!=dist[u]{continue;}
  let mut e=fl.head[u];
  while e>=0{let ei=e as usize;let v=fl.to[ei] as usize;let c=fl.cap[if reverse{ei^1}else{ei}];let nd=d.min(c);if nd>dist[v]{dist[v]=nd;heap.push((nd,v));}e=fl.nxt[ei];}
 }
 dist
}
