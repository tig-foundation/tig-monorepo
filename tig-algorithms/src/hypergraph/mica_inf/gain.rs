// Partition state with incremental km1 gains: per-edge pin counts per block (u16, dense),
// per-node counts of incident edges touching each block, and an adjacent-block mask.
// Domain: 1 <= k <= 64 (mask), hyperedges of 2..=65535 pins (u16 counters, empty edges count 0),
// m * k pin counters allocated up front.

// Candidate scan of best_move_of: up to this many adjacent blocks, walk the mask bits.
const MASK_WALK_MAX_BLOCKS: u32 = 32;

pub type Node = u32;
pub type Edge = u32;
pub type Block = u16;

pub struct State {
    pub n: usize,
    pub m: usize,
    pub k: usize,
    pub he_off: Vec<u32>,
    pub he_nodes: Vec<Node>,
    pub nd_off: Vec<u32>,
    pub nd_edges: Vec<Edge>,
    pub part: Vec<Block>,
    pub weight: Vec<u32>,
    pub block_size: Vec<u32>,
    pub max_block_size: u32,
    // pin_count[e * k + b]: pins of edge e in block b; nnz[e]: blocks touched by e.
    pin_count: Vec<u16>,
    nnz: Vec<u32>,
    // b[v * k + blk]: incident edges of v with a pin in blk; pen[v]: incident edges where v is
    // not the only pin of its block.
    b: Vec<u32>,
    pen: Vec<u32>,
    adj: Vec<u64>,
    bs_max: u32,
    bs_max_cnt: u32,
    bs_viol: u32,
}

impl State {
    pub fn new(
        n: usize,
        m: usize,
        k: usize,
        he_off: Vec<u32>,
        he_nodes: Vec<Node>,
        part: Vec<Block>,
        max_block_size: u32,
        weight: Option<Vec<u32>>,
    ) -> Self {
        assert!(k <= 64, "adjacent-block mask is limited to 64 blocks, k={}", k);
        let mut deg = vec![0u32; n + 1];
        for &v in &he_nodes {
            deg[v as usize + 1] += 1;
        }
        for i in 0..n {
            deg[i + 1] += deg[i];
        }
        let nd_off = deg.clone();
        let mut cur = nd_off.clone();
        let mut nd_edges = vec![0u32; he_nodes.len()];
        for e in 0..m {
            for i in he_off[e]..he_off[e + 1] {
                let v = he_nodes[i as usize] as usize;
                nd_edges[cur[v] as usize] = e as Edge;
                cur[v] += 1;
            }
        }

        let mut pin_count = vec![0u16; m * k];
        let mut nnz = vec![0u32; m];
        for e in 0..m {
            for i in he_off[e]..he_off[e + 1] {
                let b = part[he_nodes[i as usize] as usize] as usize;
                let j = e * k + b;
                if pin_count[j] == 0 {
                    nnz[e] += 1;
                }
                pin_count[j] += 1;
            }
        }
        let weight = weight.unwrap_or_else(|| vec![1u32; n]);
        let mut block_size = vec![0u32; k];
        for (v, &b) in part.iter().enumerate() {
            block_size[b as usize] += weight[v];
        }

        let mut b = vec![0u32; n * k];
        let mut pen = vec![0u32; n];
        let mut blocks: Vec<usize> = Vec::with_capacity(k);
        for e in 0..m {
            let lo = he_off[e] as usize;
            let hi = he_off[e + 1] as usize;
            if lo == hi {
                continue;
            }
            let base = e * k;
            blocks.clear();
            for j in 0..k {
                if pin_count[base + j] > 0 {
                    blocks.push(j);
                }
            }
            for i in lo..hi {
                let u = he_nodes[i] as usize;
                for &blk in blocks.iter() {
                    b[u * k + blk] += 1;
                }
                if pin_count[base + part[u] as usize] > 1 {
                    pen[u] += 1;
                }
            }
        }

        let mut adj = vec![0u64; n];
        for v in 0..n {
            let mut msk = 0u64;
            for j in 0..k {
                if b[v * k + j] > 0 {
                    msk |= 1u64 << j;
                }
            }
            adj[v] = msk;
        }

        let mut bs_max = 0u32;
        let mut bs_max_cnt = 0u32;
        let mut bs_viol = 0u32;
        for &s in block_size.iter() {
            if s > bs_max {
                bs_max = s;
                bs_max_cnt = 1;
            } else if s == bs_max {
                bs_max_cnt += 1;
            }
            if s > max_block_size || s == 0 {
                bs_viol += 1;
            }
        }

        State {
            n,
            m,
            k,
            he_off,
            he_nodes,
            nd_off,
            nd_edges,
            part,
            weight,
            block_size,
            max_block_size,
            pin_count,
            nnz,
            b,
            pen,
            adj,
            bs_max,
            bs_max_cnt,
            bs_viol,
        }
    }

    #[inline]
    pub fn adj_mask(&self, v: Node) -> u64 {
        self.adj[v as usize]
    }

    #[inline]
    pub fn gain_cached(&self, v: Node, to: Block) -> i32 {
        if self.part[v as usize] == to {
            return 0;
        }
        self.b[v as usize * self.k + to as usize] as i32 - self.pen[v as usize] as i32
    }

    #[inline]
    pub fn connectivity(&self, e: Edge) -> u32 {
        self.nnz[e as usize].saturating_sub(1)
    }

    pub fn km1(&self) -> u64 {
        (0..self.m).map(|e| self.connectivity(e as Edge) as u64).sum()
    }

    pub fn adjacent_blocks(&self, v: Node, out: &mut Vec<Block>) {
        out.clear();
        let mut msk = self.adj[v as usize];
        while msk != 0 {
            out.push(msk.trailing_zeros() as Block);
            msk &= msk - 1;
        }
    }

    // Moves v to `to` and returns the km1 gain.
    pub fn apply(&mut self, v: Node, to: Block) -> i32 {
        let from = self.part[v as usize];
        if from == to {
            return 0;
        }
        let g = self.apply_edges(v, from, to);
        self.part[v as usize] = to;
        let w = self.weight[v as usize];
        self.bs_update(from, to, w);
        g
    }

    fn apply_edges(&mut self, v: Node, from: Block, to: Block) -> i32 {
        let k = self.k;
        let fb = from as usize;
        let tb = to as usize;
        let mut g = 0i32;
        let lo = self.nd_off[v as usize] as usize;
        let hi = self.nd_off[v as usize + 1] as usize;
        // Edges that already had a pin in the target block: the new pen[v].
        let mut np = 0u32;
        for idx in lo..hi {
            let e = self.nd_edges[idx] as usize;
            let base = e * k;
            let cf = self.pin_count[base + fb];
            if cf == 1 {
                g += 1;
                self.pin_count[base + fb] = 0;
                self.nnz[e] -= 1;
            } else if cf > 1 {
                self.pin_count[base + fb] = cf - 1;
            }
            let ct = self.pin_count[base + tb];
            if ct == 0 {
                g -= 1;
                self.nnz[e] += 1;
            } else {
                np += 1;
            }
            self.pin_count[base + tb] = ct + 1;
        }
        for idx in lo..hi {
            let e = self.nd_edges[idx] as usize;
            let base = e * k;
            let c_from = self.pin_count[base + fb] as u32;
            let c_to = self.pin_count[base + tb] as u32;
            let elo = self.he_off[e] as usize;
            let ehi = self.he_off[e + 1] as usize;
            if c_from == 0 {
                for i in elo..ehi {
                    let u = self.he_nodes[i] as usize;
                    let idx = u * k + fb;
                    self.b[idx] -= 1;
                    if self.b[idx] == 0 {
                        self.adj[u] &= !(1u64 << from);
                    }
                }
            } else if c_from == 1 {
                for i in elo..ehi {
                    let u = self.he_nodes[i] as usize;
                    if self.part[u] == from && u != v as usize {
                        self.pen[u] -= 1;
                    }
                }
            }
            if c_to == 1 {
                for i in elo..ehi {
                    let u = self.he_nodes[i] as usize;
                    let idx = u * k + tb;
                    if self.b[idx] == 0 {
                        self.adj[u] |= 1u64 << to;
                    }
                    self.b[idx] += 1;
                }
            } else if c_to == 2 {
                for i in elo..ehi {
                    let u = self.he_nodes[i] as usize;
                    if self.part[u] == to && u != v as usize {
                        self.pen[u] += 1;
                    }
                }
            }
        }
        self.pen[v as usize] = np;
        g
    }

    #[inline]
    fn bs_update(&mut self, from: Block, to: Block, w: u32) {
        let f = from as usize;
        let t = to as usize;
        let of = self.block_size[f];
        let ot = self.block_size[t];
        let nf = of - w;
        let nt = ot + w;
        self.block_size[f] = nf;
        self.block_size[t] = nt;

        let cap = self.max_block_size;
        let vf_o = of > cap || of == 0;
        let vf_n = nf > cap || nf == 0;
        if vf_o != vf_n {
            if vf_n { self.bs_viol += 1; } else { self.bs_viol -= 1; }
        }
        let vt_o = ot > cap || ot == 0;
        let vt_n = nt > cap || nt == 0;
        if vt_o != vt_n {
            if vt_n { self.bs_viol += 1; } else { self.bs_viol -= 1; }
        }

        if w == 0 {
            return;
        }
        if of == self.bs_max {
            self.bs_max_cnt -= 1;
        }
        if nt > self.bs_max {
            self.bs_max = nt;
            self.bs_max_cnt = 1;
        } else if nt == self.bs_max {
            self.bs_max_cnt += 1;
        }
        if self.bs_max_cnt == 0 {
            let mut mx = 0u32;
            let mut c = 0u32;
            for &s in self.block_size.iter() {
                if s > mx {
                    mx = s;
                    c = 1;
                } else if s == mx {
                    c += 1;
                }
            }
            self.bs_max = mx;
            self.bs_max_cnt = c;
        }
    }

    #[inline]
    pub fn max_bs(&self) -> u32 {
        self.bs_max
    }

    // Best feasible move of u: highest gain, then smallest target block, then lowest block index.
    #[inline]
    pub fn best_move_of(&self, u: Node) -> Option<(i32, Block)> {
        let fu = self.part[u as usize];
        if self.block_size[fu as usize] <= self.weight[u as usize] {
            return None;
        }
        let k = self.k;
        let fub = fu as usize;
        let w = self.weight[u as usize] as u64;
        let cap = self.max_block_size as u64;
        if w > cap {
            return None;
        }
        let lim = cap - w;
        let base = u as usize * k;
        let row = &self.b[base..base + k];
        let bs: &[u32] = &self.block_size[..k];
        let msk = self.adj_mask(u) & !(1u64 << fu);
        let mut best_key = 0u64;
        let mut best_b = 0u32;
        if msk.count_ones() <= MASK_WALK_MAX_BLOCKS {
            let mut m = msk;
            while m != 0 {
                let bi = m.trailing_zeros() as usize;
                m &= m - 1;
                let sz = bs[bi];
                if sz as u64 > lim {
                    continue;
                }
                let key = ((row[bi] as u64) << 32) | (!sz) as u64;
                if key > best_key {
                    best_key = key;
                    best_b = bi as u32;
                }
            }
        } else {
            let (rl, rr) = row.split_at(fub);
            let (bl, br) = bs.split_at(fub);
            let mut i = 0u32;
            for (&gv, &sz) in rl.iter().zip(bl.iter()) {
                let mut key = ((gv as u64) << 32) | (!sz) as u64;
                if sz as u64 > lim {
                    key = 0;
                }
                if key > best_key {
                    best_key = key;
                    best_b = i;
                }
                i += 1;
            }
            let mut i = fub as u32 + 1;
            for (&gv, &sz) in rr.iter().skip(1).zip(br.iter().skip(1)) {
                let mut key = ((gv as u64) << 32) | (!sz) as u64;
                if sz as u64 > lim {
                    key = 0;
                }
                if key > best_key {
                    best_key = key;
                    best_b = i;
                }
                i += 1;
            }
        }
        if best_key >> 32 == 0 {
            return None;
        }
        Some((
            (best_key >> 32) as u32 as i32 - self.pen[u as usize] as i32,
            best_b as Block,
        ))
    }

    #[inline]
    pub fn fits_node(&self, v: Node, to: Block) -> bool {
        self.block_size[to as usize] + self.weight[v as usize] <= self.max_block_size
    }

    #[inline]
    pub fn imbalanced(&self) -> bool {
        self.bs_viol != 0
    }
}
