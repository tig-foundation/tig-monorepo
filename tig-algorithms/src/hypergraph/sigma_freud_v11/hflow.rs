use super::hrefine::{Block, State};

pub struct FlowParams {
    pub alpha: u32,
    pub max_region: usize,
    pub edge_cap: usize,
    pub min_shared: usize,
}

impl Default for FlowParams {
    fn default() -> Self {
        FlowParams {
            alpha: 4,
            max_region: 96,
            edge_cap: 48,
            min_shared: 1,
        }
    }
}

const INF: i32 = 1 << 29;

#[derive(Default)]
struct Dinic {
    head: Vec<i32>,
    nxt: Vec<i32>,
    to: Vec<u32>,
    cap: Vec<i32>,
    level: Vec<i32>,
    iter: Vec<i32>,
    queue: Vec<u32>,
}

impl Dinic {
    fn reset(&mut self, n: usize) {
        self.head.clear();
        self.head.resize(n, -1);
        self.nxt.clear();
        self.to.clear();
        self.cap.clear();
        self.level.clear();
        self.level.resize(n, -1);
        self.iter.clear();
        self.iter.resize(n, -1);
    }

    #[inline]
    fn add(&mut self, u: usize, v: usize, c: i32) {
        self.to.push(v as u32);
        self.cap.push(c);
        self.nxt.push(self.head[u]);
        self.head[u] = (self.to.len() - 1) as i32;
        self.to.push(u as u32);
        self.cap.push(0);
        self.nxt.push(self.head[v]);
        self.head[v] = (self.to.len() - 1) as i32;
    }

    fn bfs(&mut self, s: usize, t: usize) -> bool {
        self.level.iter_mut().for_each(|l| *l = -1);
        self.level[s] = 0;
        self.queue.clear();
        self.queue.push(s as u32);
        let mut qi = 0;
        while qi < self.queue.len() {
            let u = self.queue[qi] as usize;
            qi += 1;
            let mut e = self.head[u];
            while e != -1 {
                let ei = e as usize;
                let v = self.to[ei] as usize;
                if self.cap[ei] > 0 && self.level[v] < 0 {
                    self.level[v] = self.level[u] + 1;
                    self.queue.push(v as u32);
                }
                e = self.nxt[ei];
            }
        }
        self.level[t] >= 0
    }

    fn dfs(&mut self, u: usize, t: usize, f: i32) -> i32 {
        if u == t {
            return f;
        }
        while self.iter[u] != -1 {
            let ei = self.iter[u] as usize;
            let v = self.to[ei] as usize;
            if self.cap[ei] > 0 && self.level[v] == self.level[u] + 1 {
                let d = self.dfs(v, t, f.min(self.cap[ei]));
                if d > 0 {
                    self.cap[ei] -= d;
                    self.cap[ei ^ 1] += d;
                    return d;
                }
            }
            self.iter[u] = self.nxt[ei];
        }
        0
    }

    fn max_flow(&mut self, s: usize, t: usize, limit: i32) -> i32 {
        let mut flow = 0;
        while flow < limit && self.bfs(s, t) {
            self.iter.copy_from_slice(&self.head);
            loop {
                let f = self.dfs(s, t, limit - flow);
                if f == 0 {
                    break;
                }
                flow += f;
                if flow >= limit {
                    break;
                }
            }
        }
        flow
    }

    fn reach_from(&mut self, s: usize, mark: &mut [bool]) {
        mark.iter_mut().for_each(|m| *m = false);
        mark[s] = true;
        self.queue.clear();
        self.queue.push(s as u32);
        let mut qi = 0;
        while qi < self.queue.len() {
            let u = self.queue[qi] as usize;
            qi += 1;
            let mut e = self.head[u];
            while e != -1 {
                let ei = e as usize;
                let v = self.to[ei] as usize;
                if self.cap[ei] > 0 && !mark[v] {
                    mark[v] = true;
                    self.queue.push(v as u32);
                }
                e = self.nxt[ei];
            }
        }
    }

    fn reach_to(&mut self, t: usize, mark: &mut [bool]) {
        mark.iter_mut().for_each(|m| *m = false);
        mark[t] = true;
        self.queue.clear();
        self.queue.push(t as u32);
        let mut qi = 0;
        while qi < self.queue.len() {
            let u = self.queue[qi] as usize;
            qi += 1;
            let mut e = self.head[u];
            while e != -1 {
                let ei = e as usize;
                let v = self.to[ei] as usize;
                if self.cap[ei ^ 1] > 0 && !mark[v] {
                    mark[v] = true;
                    self.queue.push(v as u32);
                }
                e = self.nxt[ei];
            }
        }
    }
}

pub struct FlowScratch {
    din: Dinic,
    node_idx: Vec<i32>,
    edge_idx: Vec<i32>,
    seen: Vec<u32>,
    stamp: u32,
    region: Vec<u32>,
    edges: Vec<u32>,
    cut_list: Vec<u32>,
    mark: Vec<bool>,
    newblk: Vec<Block>,
    moved: Vec<(u32, Block)>,
}

impl FlowScratch {
    pub fn new(n: usize, m: usize) -> Self {
        FlowScratch {
            din: Dinic::default(),
            node_idx: vec![-1; n],
            edge_idx: vec![-1; m],
            seen: vec![0; n],
            stamp: 0,
            region: Vec::with_capacity(256),
            edges: Vec::with_capacity(1024),
            cut_list: Vec::with_capacity(256),
            mark: Vec::with_capacity(4096),
            newblk: Vec::with_capacity(256),
            moved: Vec::with_capacity(256),
        }
    }

    fn next_stamp(&mut self) -> u32 {
        if self.stamp == u32::MAX {
            self.seen.iter_mut().for_each(|s| *s = 0);
            self.stamp = 0;
        }
        self.stamp += 1;
        self.stamp
    }
}

fn grow(st: &State, side: Block, cut_list: &[u32], budget: usize, edge_cap: usize, ws: &mut FlowScratch, start: usize) {
    let tok = ws.stamp;
    let mut wsum = 0usize;
    'seed: for &e in cut_list {
        for &v in st.pins_of(e as usize) {
            let vu = v as usize;
            if st.part[vu] == side && ws.seen[vu] != tok {
                let w = st.wt[vu] as usize;
                if wsum + w > budget {
                    continue;
                }
                ws.seen[vu] = tok;
                ws.region.push(v);
                wsum += w;
                if wsum >= budget {
                    break 'seed;
                }
            }
        }
    }
    let mut qi = start;
    'bfs: while qi < ws.region.len() && wsum < budget {
        let v = ws.region[qi] as usize;
        qi += 1;
        for &e in st.edges_of(v) {
            let pins = st.pins_of(e as usize);
            if pins.len() > edge_cap {
                continue;
            }
            for &u in pins {
                let uu = u as usize;
                if st.part[uu] == side && ws.seen[uu] != tok {
                    let w = st.wt[uu] as usize;
                    if wsum + w > budget {
                        continue;
                    }
                    ws.seen[uu] = tok;
                    ws.region.push(u);
                    wsum += w;
                    if wsum >= budget {
                        break 'bfs;
                    }
                }
            }
        }
    }
}

enum Outcome {
    Gain(i32),
    Unbalanced,
    None,
}

fn cleanup(ws: &mut FlowScratch) {
    for &v in &ws.region {
        ws.node_idx[v as usize] = -1;
    }
    for &e in &ws.edges {
        ws.edge_idx[e as usize] = -1;
    }
}

fn assign_and_check(st: &State, a: Block, b: Block, on: Block, off: Block, ws: &mut FlowScratch) -> bool {
    let r = ws.region.len();
    ws.newblk.clear();
    let mut da = 0i64;
    let mut db = 0i64;
    for i in 0..r {
        let nb = if ws.mark[i] { on } else { off };
        ws.newblk.push(nb);
        let v = ws.region[i] as usize;
        let old = st.part[v];
        if old != nb {
            let w = st.wt[v] as i64;
            if nb == a {
                da += w;
                db -= w;
            } else {
                da -= w;
                db += w;
            }
        }
    }
    let lmax = st.max_block_size as i64;
    let wa = st.block_size[a as usize] as i64 + da;
    let wb = st.block_size[b as usize] as i64 + db;
    wa >= 1 && wa <= lmax && wb >= 1 && wb <= lmax
}

fn try_flow(
    st: &mut State,
    a: Block,
    b: Block,
    cut_list: &[u32],
    budget_a: usize,
    budget_b: usize,
    fp: &FlowParams,
    ws: &mut FlowScratch,
) -> Outcome {
    ws.region.clear();
    ws.edges.clear();
    ws.next_stamp();
    if budget_a > 0 {
        grow(st, a, cut_list, budget_a, fp.edge_cap, ws, 0);
    }
    let na = ws.region.len();
    if budget_b > 0 {
        grow(st, b, cut_list, budget_b, fp.edge_cap, ws, na);
    }
    let r = ws.region.len();
    if r == 0 {
        return Outcome::None;
    }
    for i in 0..r {
        ws.node_idx[ws.region[i] as usize] = i as i32;
    }
    for i in 0..r {
        let v = ws.region[i] as usize;
        for &e in st.edges_of(v) {
            if ws.edge_idx[e as usize] == -1 {
                ws.edge_idx[e as usize] = ws.edges.len() as i32;
                ws.edges.push(e);
            }
        }
    }
    let ne = ws.edges.len();
    let s = r + 2 * ne;
    let t = s + 1;
    ws.din.reset(t + 1);

    let bit_a = 1u64 << a;
    let bit_b = 1u64 << b;
    let mut cur_cut = 0i32;
    for j in 0..ne {
        let e = ws.edges[j] as usize;
        let ein = r + 2 * j;
        let eout = ein + 1;
        let mask = st.blocks_of(e);
        let mut src = false;
        let mut snk = false;
        for &v in st.pins_of(e) {
            let vu = v as usize;
            if ws.node_idx[vu] >= 0 {
                continue;
            }
            let p = st.part[vu];
            if p == a {
                src = true;
            } else if p == b {
                snk = true;
            }
        }
        if src && snk {
            continue;
        }
        if (mask & bit_a) != 0 && (mask & bit_b) != 0 {
            cur_cut += 1;
        }
        ws.din.add(ein, eout, 1);
        if src {
            ws.din.add(s, ein, INF);
        }
        if snk {
            ws.din.add(eout, t, INF);
        }
        for &v in st.pins_of(e) {
            let idx = ws.node_idx[v as usize];
            if idx >= 0 {
                ws.din.add(idx as usize, ein, INF);
                ws.din.add(eout, idx as usize, INF);
            }
        }
    }
    if cur_cut == 0 {
        cleanup(ws);
        return Outcome::None;
    }
    let f = ws.din.max_flow(s, t, cur_cut);
    if f >= cur_cut {
        cleanup(ws);
        return Outcome::None;
    }

    ws.mark.clear();
    ws.mark.resize(t + 1, false);
    ws.din.reach_from(s, &mut ws.mark);
    let mut balanced = assign_and_check(st, a, b, a, b, ws);
    if !balanced {
        ws.din.reach_to(t, &mut ws.mark);
        balanced = assign_and_check(st, a, b, b, a, ws);
    }
    if !balanced {
        cleanup(ws);
        return Outcome::Unbalanced;
    }

    ws.moved.clear();
    let mut gain = 0i32;
    for i in 0..r {
        let v = ws.region[i];
        let nb = ws.newblk[i];
        let old = st.part[v as usize];
        if old != nb {
            gain += st.apply(v as usize, nb);
            ws.moved.push((v, old));
        }
    }
    if gain <= 0 {
        for i in (0..ws.moved.len()).rev() {
            let (v, old) = ws.moved[i];
            st.apply(v as usize, old);
        }
        gain = 0;
    }
    cleanup(ws);
    Outcome::Gain(gain)
}

fn refine_pair(st: &mut State, a: Block, b: Block, list: &[u32], fp: &FlowParams, ws: &mut FlowScratch) -> i32 {
    let both = (1u64 << a) | (1u64 << b);
    ws.cut_list.clear();
    for &e in list {
        if st.blocks_of(e as usize) & both == both {
            ws.cut_list.push(e);
        }
    }
    if ws.cut_list.len() < fp.min_shared.max(1) {
        return 0;
    }
    let lmax = st.max_block_size as usize;
    let wa = st.block_size[a as usize] as usize;
    let wb = st.block_size[b as usize] as usize;
    let slack_a = (lmax - wb).min(wa.saturating_sub(1));
    let slack_b = (lmax - wa).min(wb.saturating_sub(1));
    if slack_a == 0 && slack_b == 0 {
        return 0;
    }
    let mut cut_list = Vec::new();
    cut_list.append(&mut ws.cut_list);
    let alphas = [fp.alpha.max(1) as usize, 1usize];
    let mut result = 0;
    for (ai, &alpha) in alphas.iter().enumerate() {
        if ai == 1 && alphas[0] == 1 {
            break;
        }
        let ba = (slack_a * alpha).min(fp.max_region);
        let bb = (slack_b * alpha).min(fp.max_region);
        match try_flow(st, a, b, &cut_list, ba, bb, fp, ws) {
            Outcome::Gain(g) => {
                result = g;
                break;
            }
            Outcome::Unbalanced => continue,
            Outcome::None => break,
        }
    }
    ws.cut_list = cut_list;
    result
}

pub fn flow_sweep(st: &mut State, fp: &FlowParams, ws: &mut FlowScratch, budget_ok: &dyn Fn() -> bool) -> i64 {
    let k = st.k;
    let mut lists: Vec<Vec<u32>> = (0..k * k).map(|_| Vec::new()).collect();
    let mut bits = [0u8; 64];
    for e in 0..st.m {
        let mask = st.blocks_of(e);
        if mask.count_ones() < 2 {
            continue;
        }
        let mut c = 0usize;
        let mut tmp = mask;
        while tmp != 0 {
            bits[c] = tmp.trailing_zeros() as u8;
            c += 1;
            tmp &= tmp - 1;
        }
        for i in 0..c {
            for j in i + 1..c {
                lists[bits[i] as usize * k + bits[j] as usize].push(e as u32);
            }
        }
    }
    let min_shared = fp.min_shared.max(1);
    let mut order: Vec<(u32, u32)> = lists
        .iter()
        .enumerate()
        .filter(|(_, l)| l.len() >= min_shared)
        .map(|(i, l)| (l.len() as u32, i as u32))
        .collect();
    order.sort_unstable_by(|x, y| y.0.cmp(&x.0).then(x.1.cmp(&y.1)));

    let mut total = 0i64;
    for (pi, &(_, idx)) in order.iter().enumerate() {
        if pi % 32 == 0 && !budget_ok() {
            break;
        }
        let mut list = Vec::new();
        list.append(&mut lists[idx as usize]);
        let a = (idx as usize / k) as Block;
        let b = (idx as usize % k) as Block;
        total += refine_pair(st, a, b, &list, fp, ws) as i64;
    }
    total
}
