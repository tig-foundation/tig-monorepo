use std::collections::BinaryHeap;

pub type Block = u8;

pub struct Params {
    pub fm_rounds: usize,
    pub fm_noimprove: usize,
    pub fm_deficit: i32,
    pub fm_max_steps: usize,
    pub fm_seeds: usize,
    pub fm_touch_cap: usize,
    pub hcm_max_size: usize,
    pub hcm_max_pins: u32,
    pub passes: usize,
    pub ils_iters: usize,
    pub ils_strength: usize,
    pub flow: bool,
    pub flow_rounds: usize,
    pub flow_params: super::hflow::FlowParams,
    pub flow_in_cycles: bool,
    pub vcycles: usize,
    pub ml_levels: usize,
    pub ml_max_cluster: u32,
    pub ml_patience: usize,
    pub ml_fine_full: bool,
    pub ml_coarse_full: bool,
    pub jet_rounds: usize,
    pub jet_tolerance: usize,
    pub jet_neg_pct: u32,
    pub jet_min_gain: i32,
    pub jet_stages: u32,
    pub jet_slack: i32,
    pub seed: u64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            fm_rounds: 4,
            fm_noimprove: 12,
            fm_deficit: 3,
            fm_max_steps: 400,
            fm_seeds: 20,
            fm_touch_cap: 48,
            hcm_max_size: 24,
            hcm_max_pins: 3,
            passes: 2,
            ils_iters: 0,
            ils_strength: 64,
            flow: true,
            flow_rounds: 3,
            flow_params: super::hflow::FlowParams::default(),
            flow_in_cycles: false,
            vcycles: 4,
            ml_levels: 3,
            ml_max_cluster: 0,
            ml_patience: 2,
            ml_fine_full: false,
            ml_coarse_full: true,
            jet_rounds: 0,
            jet_tolerance: 6,
            jet_neg_pct: 25,
            jet_min_gain: 0,
            jet_stages: 3,
            jet_slack: -1,
            seed: 0x5EED_5EED_1234_ABCD,
        }
    }
}

#[inline]
fn splitmix(z: &mut u64) -> u64 {
    *z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = *z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

pub struct State {
    pub n: usize,
    pub m: usize,
    pub k: usize,
    pub he_off: Vec<u32>,
    pub he_nodes: Vec<u32>,
    pub nd_off: Vec<u32>,
    pub nd_edges: Vec<u32>,
    pub part: Vec<Block>,
    pub wt: Vec<u32>,
    pub block_size: Vec<u32>,
    pub max_block_size: u32,
    pc: Vec<u16>,
    emask: Vec<u64>,
    pub cut: i64,
}

impl State {
    pub fn new(
        n: usize,
        m: usize,
        k: usize,
        he_off: Vec<u32>,
        he_nodes: Vec<u32>,
        part: Vec<Block>,
        wt: Vec<u32>,
        max_block_size: u32,
    ) -> Self {
        let mut deg = vec![0u32; n + 1];
        for &v in &he_nodes {
            deg[v as usize + 1] += 1;
        }
        for i in 0..n {
            deg[i + 1] += deg[i];
        }
        let nd_off = deg;
        let mut cur = nd_off.clone();
        let mut nd_edges = vec![0u32; he_nodes.len()];
        for e in 0..m {
            for i in he_off[e]..he_off[e + 1] {
                let v = he_nodes[i as usize] as usize;
                nd_edges[cur[v] as usize] = e as u32;
                cur[v] += 1;
            }
        }
        let mut st = State {
            n,
            m,
            k,
            he_off,
            he_nodes,
            nd_off,
            nd_edges,
            part,
            wt,
            block_size: vec![0; k],
            max_block_size,
            pc: vec![0u16; m * k],
            emask: vec![0u64; m],
            cut: 0,
        };
        st.rebuild();
        st
    }

    pub fn rebuild(&mut self) {
        self.pc.iter_mut().for_each(|x| *x = 0);
        self.block_size.iter_mut().for_each(|x| *x = 0);
        for v in 0..self.n {
            self.block_size[self.part[v] as usize] += self.wt[v];
        }
        let mut cut = 0i64;
        for e in 0..self.m {
            let mut mask = 0u64;
            let base = e * self.k;
            for i in self.he_off[e]..self.he_off[e + 1] {
                let b = self.part[self.he_nodes[i as usize] as usize] as usize;
                self.pc[base + b] += 1;
                mask |= 1u64 << b;
            }
            self.emask[e] = mask;
            cut += mask.count_ones() as i64 - 1;
        }
        self.cut = cut;
    }

    #[inline]
    pub fn edges_of(&self, v: usize) -> &[u32] {
        &self.nd_edges[self.nd_off[v] as usize..self.nd_off[v + 1] as usize]
    }

    #[inline]
    pub fn pins_of(&self, e: usize) -> &[u32] {
        &self.he_nodes[self.he_off[e] as usize..self.he_off[e + 1] as usize]
    }

    #[inline]
    pub fn lambda(&self, e: usize) -> u32 {
        self.emask[e].count_ones()
    }

    #[inline]
    pub fn blocks_of(&self, e: usize) -> u64 {
        self.emask[e]
    }

    #[inline]
    pub fn pin_count(&self, e: usize, b: Block) -> u16 {
        self.pc[e * self.k + b as usize]
    }

    #[inline]
    pub fn is_boundary(&self, v: usize) -> bool {
        self.edges_of(v).iter().any(|&e| self.emask[e as usize].count_ones() >= 2)
    }

    #[inline]
    pub fn fits(&self, to: Block, w: u32) -> bool {
        self.block_size[to as usize] + w <= self.max_block_size
    }

    #[inline]
    pub fn can_leave(&self, from: Block, w: u32) -> bool {
        self.block_size[from as usize] > w
    }

    pub fn best_move(&self, v: usize, cnt: &mut [i32]) -> Option<(i32, Block)> {
        let from = self.part[v];
        let w = self.wt[v];
        if !self.can_leave(from, w) {
            return None;
        }
        let edges = self.edges_of(v);
        let deg = edges.len() as i32;
        let mut adj = 0u64;
        let mut a = 0i32;
        for &e in edges {
            let e = e as usize;
            let mask = self.emask[e];
            adj |= mask;
            if self.pc[e * self.k + from as usize] == 1 {
                a += 1;
            }
        }
        adj &= !(1u64 << from);
        if adj == 0 {
            return None;
        }
        let mut tmp = adj;
        while tmp != 0 {
            let b = tmp.trailing_zeros() as usize;
            tmp &= tmp - 1;
            cnt[b] = 0;
        }
        for &e in edges {
            let mut mask = self.emask[e as usize] & adj;
            while mask != 0 {
                let b = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                cnt[b] += 1;
            }
        }
        let mut best: Option<(i32, Block)> = None;
        let mut tmp = adj;
        while tmp != 0 {
            let b = tmp.trailing_zeros() as usize;
            tmp &= tmp - 1;
            if !self.fits(b as Block, w) {
                continue;
            }
            let g = a - (deg - cnt[b]);
            let better = match best {
                None => true,
                Some((bg, bb)) => {
                    g > bg
                        || (g == bg
                            && (self.block_size[b] < self.block_size[bb as usize]
                                || (self.block_size[b] == self.block_size[bb as usize] && (b as Block) < bb)))
                }
            };
            if better {
                best = Some((g, b as Block));
            }
        }
        best
    }

    pub fn best_move_unconstrained(&self, v: usize, cnt: &mut [i32], cap: u32) -> Option<(i32, Block, i32)> {
        let from = self.part[v];
        let w = self.wt[v];
        if !self.can_leave(from, w) {
            return None;
        }
        let edges = self.edges_of(v);
        let deg = edges.len() as i32;
        let mut adj = 0u64;
        let mut a = 0i32;
        for &e in edges {
            let e = e as usize;
            adj |= self.emask[e];
            if self.pc[e * self.k + from as usize] == 1 {
                a += 1;
            }
        }
        adj &= !(1u64 << from);
        if adj == 0 {
            return None;
        }
        let mut tmp = adj;
        while tmp != 0 {
            let b = tmp.trailing_zeros() as usize;
            tmp &= tmp - 1;
            cnt[b] = 0;
        }
        for &e in edges {
            let mut mask = self.emask[e as usize] & adj;
            while mask != 0 {
                let b = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                cnt[b] += 1;
            }
        }
        let mut best: Option<(i32, Block)> = None;
        let mut tmp = adj;
        while tmp != 0 {
            let b = tmp.trailing_zeros() as usize;
            tmp &= tmp - 1;
            if cap != u32::MAX && self.block_size[b] + w > cap {
                continue;
            }
            let g = a - (deg - cnt[b]);
            let better = match best {
                None => true,
                Some((bg, bb)) => {
                    g > bg
                        || (g == bg
                            && (self.block_size[b] < self.block_size[bb as usize]
                                || (self.block_size[b] == self.block_size[bb as usize] && (b as Block) < bb)))
                }
            };
            if better {
                best = Some((g, b as Block));
            }
        }
        best.map(|(g, b)| (g, b, deg - a))
    }

    pub fn best_move_any(&self, v: usize, cnt: &mut [i32]) -> Option<(i32, Block)> {
        let from = self.part[v];
        let w = self.wt[v];
        if !self.can_leave(from, w) {
            return None;
        }
        let edges = self.edges_of(v);
        let deg = edges.len() as i32;
        for b in 0..self.k {
            cnt[b] = 0;
        }
        let mut a = 0i32;
        for &e in edges {
            let e = e as usize;
            if self.pc[e * self.k + from as usize] == 1 {
                a += 1;
            }
            let mut mask = self.emask[e];
            while mask != 0 {
                let b = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                cnt[b] += 1;
            }
        }
        let mut best: Option<(i32, Block)> = None;
        for b in 0..self.k {
            if b == from as usize || !self.fits(b as Block, w) {
                continue;
            }
            let g = a - (deg - cnt[b]);
            let better = match best {
                None => true,
                Some((bg, bb)) => {
                    g > bg
                        || (g == bg
                            && (self.block_size[b] < self.block_size[bb as usize]
                                || (self.block_size[b] == self.block_size[bb as usize] && (b as Block) < bb)))
                }
            };
            if better {
                best = Some((g, b as Block));
            }
        }
        best
    }

    pub fn move_gain(&self, v: usize, to: Block) -> i32 {
        let from = self.part[v];
        if from == to {
            return 0;
        }
        let k = self.k;
        let mut g = 0i32;
        for &e in self.edges_of(v) {
            let base = e as usize * k;
            if self.pc[base + from as usize] == 1 {
                g += 1;
            }
            if self.pc[base + to as usize] == 0 {
                g -= 1;
            }
        }
        g
    }

    pub fn apply(&mut self, v: usize, to: Block) -> i32 {
        let from = self.part[v];
        if from == to {
            return 0;
        }
        let mut g = 0i32;
        let k = self.k;
        let lo = self.nd_off[v] as usize;
        let hi = self.nd_off[v + 1] as usize;
        for idx in lo..hi {
            let e = self.nd_edges[idx] as usize;
            let base = e * k;
            let cf = self.pc[base + from as usize] - 1;
            self.pc[base + from as usize] = cf;
            if cf == 0 {
                self.emask[e] &= !(1u64 << from);
                g += 1;
            }
            let ct = self.pc[base + to as usize];
            if ct == 0 {
                self.emask[e] |= 1u64 << to;
                g -= 1;
            }
            self.pc[base + to as usize] = ct + 1;
        }
        self.part[v] = to;
        let w = self.wt[v];
        self.block_size[from as usize] -= w;
        self.block_size[to as usize] += w;
        self.cut -= g as i64;
        g
    }

    pub fn feasible(&self) -> bool {
        self.block_size.iter().all(|&s| s >= 1 && s <= self.max_block_size)
    }
}

struct MoveRec {
    node: u32,
    from: Block,
}

struct Scratch {
    cnt: Vec<i32>,
    log: Vec<MoveRec>,
    heap: BinaryHeap<(i32, u32, std::cmp::Reverse<u32>, Block)>,
    touched: Vec<u32>,
    seen: Vec<u32>,
    stamp: u32,
    moved: Vec<u32>,
}

impl Scratch {
    #[inline]
    fn next_stamp(&mut self) -> u32 {
        if self.stamp == u32::MAX {
            self.seen.iter_mut().for_each(|s| *s = 0);
            self.stamp = 0;
        }
        self.stamp += 1;
        self.stamp
    }
}

fn localized_fm(st: &mut State, seeds: &[u32], locked: &mut [bool], p: &Params, sc: &mut Scratch) -> i32 {
    sc.log.clear();
    sc.heap.clear();

    for &u in seeds {
        let u = u as usize;
        if locked[u] {
            continue;
        }
        if let Some((g, b)) = st.best_move(u, &mut sc.cnt) {
            let deg = st.edges_of(u).len() as u32;
            sc.heap.push((g, deg, std::cmp::Reverse(u as u32), b));
        }
    }

    let mut running = 0i32;
    let mut best = 0i32;
    let mut best_len = 0usize;
    let mut since_improve = 0usize;

    while let Some((g_stale, _d, std::cmp::Reverse(v), b_stale)) = sc.heap.pop() {
        let vu = v as usize;
        if locked[vu] {
            continue;
        }
        let (g_now, to) = match st.best_move(vu, &mut sc.cnt) {
            Some(x) => x,
            None => continue,
        };
        if g_now != g_stale || to != b_stale {
            let deg = st.edges_of(vu).len() as u32;
            sc.heap.push((g_now, deg, std::cmp::Reverse(v), to));
            continue;
        }
        if sc.log.is_empty() && g_now < 0 {
            continue;
        }
        let from = st.part[vu];
        let real = st.apply(vu, to);
        locked[vu] = true;
        sc.log.push(MoveRec { node: v, from });
        running += real;
        if running > best {
            best = running;
            best_len = sc.log.len();
            since_improve = 0;
        } else {
            since_improve += 1;
            if running < best - p.fm_deficit {
                break;
            }
        }

        let tok = sc.next_stamp();
        sc.touched.clear();
        let lo = st.nd_off[vu] as usize;
        let hi = st.nd_off[vu + 1] as usize;
        for idx in lo..hi {
            let e = st.nd_edges[idx] as usize;
            let elo = st.he_off[e] as usize;
            let ehi = st.he_off[e + 1] as usize;
            if ehi - elo > p.fm_touch_cap {
                continue;
            }
            for i in elo..ehi {
                let u = st.he_nodes[i];
                let uu = u as usize;
                if !locked[uu] && sc.seen[uu] != tok {
                    sc.seen[uu] = tok;
                    sc.touched.push(u);
                }
            }
        }
        for i in 0..sc.touched.len() {
            let u = sc.touched[i] as usize;
            if let Some((g, b)) = st.best_move(u, &mut sc.cnt) {
                let deg = st.edges_of(u).len() as u32;
                sc.heap.push((g, deg, std::cmp::Reverse(u as u32), b));
            }
        }

        if since_improve >= p.fm_noimprove || sc.log.len() >= p.fm_max_steps {
            break;
        }
    }

    while sc.log.len() > best_len {
        let mv = sc.log.pop().unwrap();
        st.apply(mv.node as usize, mv.from);
    }
    for mv in &sc.log {
        sc.moved.push(mv.node);
    }
    best
}

fn boundary_nodes(st: &State) -> Vec<u32> {
    let mut border: Vec<u32> = Vec::with_capacity(st.n / 2);
    for v in 0..st.n {
        if st.is_boundary(v) {
            border.push(v as u32);
        }
    }
    border
}

fn neighbourhood(st: &State, nodes: &[u32], touch_cap: usize, sc: &mut Scratch, out: &mut Vec<u32>) {
    out.clear();
    let tok = sc.next_stamp();
    for &v in nodes {
        let vu = v as usize;
        if sc.seen[vu] != tok {
            sc.seen[vu] = tok;
            if st.is_boundary(vu) {
                out.push(v);
            }
        }
        for &e in st.edges_of(vu) {
            let pins = st.pins_of(e as usize);
            if pins.len() > touch_cap {
                continue;
            }
            for &u in pins {
                let uu = u as usize;
                if sc.seen[uu] != tok {
                    sc.seen[uu] = tok;
                    if st.is_boundary(uu) {
                        out.push(u);
                    }
                }
            }
        }
    }
}

fn fm_round(
    st: &mut State,
    p: &Params,
    rng: &mut u64,
    locked: &mut [bool],
    sc: &mut Scratch,
    seeds: &[u32],
    active: &mut Vec<u32>,
) -> i64 {
    active.clear();
    if seeds.is_empty() {
        return 0;
    }
    let mut order: Vec<u32> = seeds.to_vec();
    for i in (1..order.len()).rev() {
        let j = (splitmix(rng) % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    locked.iter_mut().for_each(|f| *f = false);
    sc.moved.clear();
    let mut total = 0i64;
    for chunk in order.chunks(p.fm_seeds.max(1)) {
        total += localized_fm(st, chunk, locked, p, sc) as i64;
    }
    let mut moved = Vec::new();
    moved.append(&mut sc.moved);
    neighbourhood(st, &moved, p.fm_touch_cap, sc, active);
    sc.moved.append(&mut moved);
    total
}

fn fm_pass(
    st: &mut State,
    p: &Params,
    rng: &mut u64,
    locked: &mut [bool],
    sc: &mut Scratch,
    mut seeds: Vec<u32>,
    budget_ok: &dyn Fn() -> bool,
) -> i64 {
    let mut total = 0i64;
    let mut active: Vec<u32> = Vec::new();
    for _ in 0..p.fm_rounds.max(1) {
        if seeds.is_empty() || !budget_ok() {
            break;
        }
        let g = fm_round(st, p, rng, locked, sc, &seeds, &mut active);
        total += g;
        if g <= 0 {
            break;
        }
        (seeds, active) = (active, seeds);
    }
    total
}

fn compound_round(st: &mut State, p: &Params) -> i64 {
    let mut order: Vec<(u32, u32)> = Vec::new();
    for e in 0..st.m {
        let sz = (st.he_off[e + 1] - st.he_off[e]) as usize;
        if sz <= p.hcm_max_size && st.lambda(e) >= 2 {
            order.push((sz as u32, e as u32));
        }
    }
    order.sort_unstable();
    let mut total = 0i64;
    let mut moved: Vec<(u32, Block)> = Vec::with_capacity(8);
    for &(_, e) in &order {
        let e = e as usize;
        if st.lambda(e) < 2 {
            continue;
        }
        let mask = st.emask[e];
        let mut major: Block = 0;
        let mut major_cnt = 0u16;
        let mut tmp = mask;
        while tmp != 0 {
            let b = tmp.trailing_zeros() as Block;
            tmp &= tmp - 1;
            let c = st.pin_count(e, b);
            if c > major_cnt {
                major_cnt = c;
                major = b;
            }
        }
        let mut tmp = mask;
        while tmp != 0 {
            let b = tmp.trailing_zeros() as Block;
            tmp &= tmp - 1;
            if b == major {
                continue;
            }
            let c = st.pin_count(e, b);
            if c == 0 || c as u32 > p.hcm_max_pins {
                continue;
            }
            let lo = st.he_off[e] as usize;
            let hi = st.he_off[e + 1] as usize;
            let mut wsum = 0u32;
            for i in lo..hi {
                let v = st.he_nodes[i] as usize;
                if st.part[v] == b {
                    wsum += st.wt[v];
                }
            }
            if st.block_size[major as usize] + wsum > st.max_block_size {
                continue;
            }
            if st.block_size[b as usize] <= wsum {
                continue;
            }
            moved.clear();
            let mut gain = 0i32;
            for i in lo..hi {
                let v = st.he_nodes[i];
                if st.part[v as usize] == b {
                    gain += st.apply(v as usize, major);
                    moved.push((v, b));
                }
            }
            if gain <= 0 {
                for &(v, from) in moved.iter().rev() {
                    st.apply(v as usize, from);
                }
            } else {
                total += gain as i64;
                if st.lambda(e) < 2 {
                    break;
                }
            }
        }
    }
    total
}

fn perturb(st: &mut State, strength: usize, rng: &mut u64, sc: &mut Scratch) -> Vec<u32> {
    let mut pool: Vec<u32> = Vec::with_capacity(strength * 4);
    for _ in 0..16 {
        let e = (splitmix(rng) % st.m as u64) as usize;
        if st.lambda(e) < 2 {
            continue;
        }
        let tok = sc.next_stamp();
        for &v in st.pins_of(e) {
            let vu = v as usize;
            if sc.seen[vu] != tok {
                sc.seen[vu] = tok;
                pool.push(v);
            }
            for &f in st.edges_of(vu) {
                let pins = st.pins_of(f as usize);
                if pins.len() > 48 {
                    continue;
                }
                for &u in pins {
                    let uu = u as usize;
                    if sc.seen[uu] != tok {
                        sc.seen[uu] = tok;
                        pool.push(u);
                    }
                }
            }
        }
        if pool.len() >= strength {
            break;
        }
    }
    let mut moved: Vec<u32> = Vec::with_capacity(strength);
    let mut attempts = 0usize;
    while moved.len() < strength && attempts < strength * 20 {
        attempts += 1;
        let v = if pool.is_empty() {
            (splitmix(rng) % st.n as u64) as usize
        } else {
            pool[(splitmix(rng) % pool.len() as u64) as usize] as usize
        };
        let from = st.part[v];
        let w = st.wt[v];
        if !st.can_leave(from, w) {
            continue;
        }
        let mut adj = 0u64;
        for &e in st.edges_of(v) {
            adj |= st.emask[e as usize];
        }
        adj &= !(1u64 << from);
        if adj == 0 {
            continue;
        }
        let nb = adj.count_ones() as u64;
        let pick = (splitmix(rng) % nb) as u32;
        let mut tmp = adj;
        for _ in 0..pick {
            tmp &= tmp - 1;
        }
        let to = tmp.trailing_zeros() as Block;
        if !st.fits(to, w) {
            continue;
        }
        st.apply(v, to);
        moved.push(v as u32);
    }
    moved
}

pub struct Report {
    pub before: i64,
    pub after: i64,
}

fn vcycle(
    st: &mut State,
    p: &Params,
    rng: &mut u64,
    locked: &mut [bool],
    sc: &mut Scratch,
    fw: &mut super::hflow::FlowScratch,
    ms: &mut super::hmulti::Scratch,
    budget_ok: &dyn Fn() -> bool,
) -> i64 {
    let before = st.cut;
    let max_cluster = if p.ml_max_cluster > 0 {
        p.ml_max_cluster
    } else {
        (st.max_block_size / 8).max(2)
    };

    let part_before: Vec<Block> = st.part.clone();
    let mut maps: Vec<Vec<u32>> = Vec::new();
    let mut sts: Vec<State> = Vec::new();
    while sts.len() < p.ml_levels.max(1) {
        let next = {
            let cur: &State = match sts.last() {
                Some(s) => s,
                None => &*st,
            };
            super::hmulti::coarsen(cur, max_cluster, p.fm_touch_cap, rng, ms)
        };
        match next {
            Some(c) => {
                maps.push(c.map);
                sts.push(c.state);
            }
            None => break,
        }
    }
    if sts.is_empty() {
        return 0;
    }

    let mut next_seeds: Option<Vec<u32>> = None;
    for li in (0..sts.len()).rev() {
        if !budget_ok() {
            break;
        }
        {
            let cs = &mut sts[li];
            let seeds = match next_seeds.take() {
                Some(s) if !p.ml_coarse_full => s,
                _ => boundary_nodes(cs),
            };
            let _ = fm_pass(cs, p, rng, locked, sc, seeds, budget_ok);
            let _ = compound_round(cs, p);
        }
        if li == 0 {
            let map = &maps[0];
            let pc = &sts[0].part;
            for v in 0..st.n {
                st.part[v] = pc[map[v] as usize];
            }
            st.rebuild();
        } else {
            let (lo, hi) = sts.split_at_mut(li);
            let finer = &mut lo[li - 1];
            let pc = &hi[0].part;
            let map = &maps[li];
            let mut moved: Vec<u32> = Vec::new();
            for v in 0..finer.n {
                let nb = pc[map[v] as usize];
                if finer.part[v] != nb {
                    finer.part[v] = nb;
                    moved.push(v as u32);
                }
            }
            finer.rebuild();
            if !p.ml_coarse_full {
                let mut out: Vec<u32> = Vec::new();
                neighbourhood(finer, &moved, p.fm_touch_cap, sc, &mut out);
                next_seeds = Some(out);
            }
        }
    }
    if budget_ok() {
        let seeds = if p.ml_fine_full {
            boundary_nodes(st)
        } else {
            let mut moved: Vec<u32> = Vec::new();
            for v in 0..st.n {
                if st.part[v] != part_before[v] {
                    moved.push(v as u32);
                }
            }
            let mut out: Vec<u32> = Vec::new();
            neighbourhood(st, &moved, p.fm_touch_cap, sc, &mut out);
            out
        };
        let _ = fm_pass(st, p, rng, locked, sc, seeds, budget_ok);
        let _ = compound_round(st, p);
    }
    if p.flow && p.flow_in_cycles && budget_ok() {
        for _ in 0..p.flow_rounds.max(1) {
            if !budget_ok() {
                break;
            }
            let g = super::hflow::flow_sweep(st, &p.flow_params, fw, budget_ok);
            if g <= 0 {
                break;
            }
        }
    }
    before - st.cut
}

pub fn improve(
    n: usize,
    m: usize,
    k: usize,
    he_off: Vec<u32>,
    he_nodes: Vec<u32>,
    max_block_size: u32,
    partition: &mut [u32],
    p: &Params,
    budget_ok: &dyn Fn() -> bool,
) -> Report {
    let part: Vec<Block> = partition.iter().map(|&x| x as Block).collect();
    let mut st = State::new(n, m, k, he_off, he_nodes, part, vec![1u32; n], max_block_size);
    let before = st.cut;
    let mut best_cut = st.cut;
    let mut best_part = st.part.clone();
    let start_feasible = st.feasible();

    let mut rng = p.seed ^ (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (m as u64);
    let mut locked = vec![false; n];
    let mut sc = Scratch {
        cnt: vec![0i32; k],
        log: Vec::with_capacity(p.fm_max_steps + 1),
        heap: BinaryHeap::with_capacity(4096),
        touched: Vec::with_capacity(512),
        seen: vec![0u32; n],
        stamp: 0,
        moved: Vec::with_capacity(4096),
    };
    let mut fw = super::hflow::FlowScratch::new(n, m);
    let mut ms = super::hmulti::Scratch::new(n);

    macro_rules! keep {
        () => {{
            if st.cut < best_cut && st.feasible() {
                best_cut = st.cut;
                best_part.copy_from_slice(&st.part);
            }
        }};
    }
    macro_rules! jet_stage {
        () => {{
            if p.jet_rounds > 0 && budget_ok() {
                let part_before: Vec<Block> = st.part.clone();
                let jp = super::hjet::JetParams {
                    rounds: p.jet_rounds,
                    tolerance: p.jet_tolerance,
                    neg_pct: p.jet_neg_pct,
                    min_gain: p.jet_min_gain,
                    rebalance_iters: 8,
                    cap: if p.jet_slack < 0 { u32::MAX } else { st.max_block_size + p.jet_slack as u32 },
                };
                let _ = super::hjet::jet_refine(&mut st, &jp, budget_ok);
                keep!();
                if budget_ok() {
                    let mut moved: Vec<u32> = Vec::new();
                    for v in 0..n {
                        if st.part[v] != part_before[v] {
                            moved.push(v as u32);
                        }
                    }
                    if !moved.is_empty() {
                        let mut seeds: Vec<u32> = Vec::new();
                        neighbourhood(&st, &moved, p.fm_touch_cap, &mut sc, &mut seeds);
                        let _ = fm_pass(&mut st, p, &mut rng, &mut locked, &mut sc, seeds, budget_ok);
                        keep!();
                    }
                }
            }
        }};
    }

    for _pass in 0..p.passes.max(1) {
        if !budget_ok() {
            break;
        }
        let before_pass = st.cut;
        let seeds = boundary_nodes(&st);
        let _ = fm_pass(&mut st, p, &mut rng, &mut locked, &mut sc, seeds, budget_ok);
        keep!();
        if !budget_ok() {
            break;
        }
        let _ = compound_round(&mut st, p);
        keep!();
        if p.flow && p.flow_in_cycles {
            for _ in 0..p.flow_rounds.max(1) {
                if !budget_ok() {
                    break;
                }
                let g = super::hflow::flow_sweep(&mut st, &p.flow_params, &mut fw, budget_ok);
                keep!();
                if g <= 0 {
                    break;
                }
            }
        }
        if st.cut >= before_pass {
            break;
        }
    }

    if p.jet_stages & 1 != 0 {
        jet_stage!();
    }

    if p.vcycles > 0 {
        let mut stall = 0usize;
        for _ in 0..p.vcycles {
            if !budget_ok() {
                break;
            }
            let g = vcycle(&mut st, p, &mut rng, &mut locked, &mut sc, &mut fw, &mut ms, budget_ok);
            keep!();
            if g <= 0 {
                stall += 1;
                if stall >= p.ml_patience.max(1) {
                    break;
                }
            } else {
                stall = 0;
            }
        }
    }

    if p.jet_stages & 2 != 0 {
        jet_stage!();
    }

    if p.flow && !p.flow_in_cycles && budget_ok() {
        let part_before: Vec<Block> = st.part.clone();
        let mut g_flow = 0i64;
        for _ in 0..p.flow_rounds.max(1) {
            if !budget_ok() {
                break;
            }
            let g = super::hflow::flow_sweep(&mut st, &p.flow_params, &mut fw, budget_ok);
            g_flow += g;
            keep!();
            if g <= 0 {
                break;
            }
        }
        if g_flow > 0 && budget_ok() {
            let mut moved: Vec<u32> = Vec::new();
            for v in 0..n {
                if st.part[v] != part_before[v] {
                    moved.push(v as u32);
                }
            }
            let mut seeds: Vec<u32> = Vec::new();
            neighbourhood(&st, &moved, p.fm_touch_cap, &mut sc, &mut seeds);
            let _ = fm_pass(&mut st, p, &mut rng, &mut locked, &mut sc, seeds, budget_ok);
            keep!();
        }
    }

    if p.ils_iters > 0 && budget_ok() {
        if st.cut != best_cut {
            st.part.copy_from_slice(&best_part);
            st.rebuild();
        }
        let mut seeds: Vec<u32> = Vec::new();
        for _ in 0..p.ils_iters {
            if !budget_ok() {
                break;
            }
            let moved = perturb(&mut st, p.ils_strength, &mut rng, &mut sc);
            neighbourhood(&st, &moved, p.fm_touch_cap, &mut sc, &mut seeds);
            let mut s = Vec::new();
            s.append(&mut seeds);
            fm_pass(&mut st, p, &mut rng, &mut locked, &mut sc, s, budget_ok);
            compound_round(&mut st, p);
            if p.flow && budget_ok() {
                super::hflow::flow_sweep(&mut st, &p.flow_params, &mut fw, budget_ok);
            }
            if st.cut <= best_cut && st.feasible() {
                keep!();
                if st.cut == best_cut {
                    best_part.copy_from_slice(&st.part);
                }
            } else {
                st.part.copy_from_slice(&best_part);
                st.rebuild();
            }
        }
    }

    if best_cut < before || !start_feasible {
        for (dst, &b) in partition.iter_mut().zip(best_part.iter()) {
            *dst = b as u32;
        }
    }
    Report {
        before,
        after: best_cut,
    }
}
