// Partition state.

const SPARSE_MAX: u32 = 8;
const DENSE_LANES: usize = 16;
const MAX_BLOCKS: usize = 64;
const COUNTER_MAX: u32 = u16::MAX as u32;

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
    pin_count: Vec<u16>,
    nnz: Vec<u32>,
    b: Vec<u16>,
    pen: Vec<u16>,
    adj: Vec<u64>,
    room: [u64; 2],
    bs_max: u32,
    bs_max_cnt: u32,
    bs_viol: u32,
}

impl State {
    /// Builds the partition state, or `None` when `k` exceeds the block mask or a pin count or
    /// node degree exceeds the 16-bit counters.
    pub fn new(
        n: usize,
        m: usize,
        k: usize,
        he_off: Vec<u32>,
        he_nodes: Vec<Node>,
        part: Vec<Block>,
        max_block_size: u32,
        weight: Option<Vec<u32>>,
    ) -> Option<Self> {
        if k > MAX_BLOCKS
            || he_off
                .windows(2)
                .any(|w| w[1].saturating_sub(w[0]) > COUNTER_MAX)
        {
            return None;
        }
        let mut nd_off = vec![0u32; n + 1];
        for &v in &he_nodes {
            nd_off[v as usize + 1] += 1;
        }
        if nd_off.iter().any(|&d| d > COUNTER_MAX) {
            return None;
        }
        for i in 0..n {
            nd_off[i + 1] += nd_off[i];
        }
        let weight = weight.unwrap_or_else(|| vec![1u32; n]);
        Some(Self::build(
            n,
            m,
            k,
            he_off,
            he_nodes,
            nd_off,
            part,
            max_block_size,
            weight,
        ))
    }

    /// Same hypergraph and weights, new partition.
    pub fn with_partition(&self, part: Vec<Block>) -> Self {
        Self::build(
            self.n,
            self.m,
            self.k,
            self.he_off.clone(),
            self.he_nodes.clone(),
            self.nd_off.clone(),
            part,
            self.max_block_size,
            self.weight.clone(),
        )
    }

    fn build(
        n: usize,
        m: usize,
        k: usize,
        he_off: Vec<u32>,
        he_nodes: Vec<Node>,
        nd_off: Vec<u32>,
        part: Vec<Block>,
        max_block_size: u32,
        weight: Vec<u32>,
    ) -> Self {
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
        let mut b = vec![0u16; n * k];
        let mut pen = vec![0u16; n];
        let mut adj = vec![0u64; n];
        let mut blocks: Vec<usize> = Vec::with_capacity(k);
        for e in 0..m {
            let lo = he_off[e] as usize;
            let hi = he_off[e + 1] as usize;
            if lo == hi {
                continue;
            }
            let base = e * k;
            blocks.clear();
            let mut mask = 0u64;
            for i in lo..hi {
                let bl = part[he_nodes[i] as usize] as usize;
                if pin_count[base + bl] == 0 {
                    blocks.push(bl);
                    mask |= 1u64 << bl;
                }
                pin_count[base + bl] += 1;
            }
            nnz[e] = blocks.len() as u32;
            for i in lo..hi {
                let u = he_nodes[i] as usize;
                for &blk in blocks.iter() {
                    b[u * k + blk] += 1;
                }
                adj[u] |= mask;
                if pin_count[base + part[u] as usize] > 1 {
                    pen[u] += 1;
                }
            }
        }
        let mut block_size = vec![0u32; k];
        for (v, &bl) in part.iter().enumerate() {
            block_size[bl as usize] += weight[v];
        }
        let mut bs_max = 0u32;
        let mut bs_max_cnt = 0u32;
        let mut bs_viol = 0u32;
        let mut room = [0u64; 2];
        for (bl, &s) in block_size.iter().enumerate() {
            for (i, mask) in room.iter_mut().enumerate() {
                if s as u64 + i as u64 + 1 <= max_block_size as u64 {
                    *mask |= 1u64 << bl;
                }
            }
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
            room,
            bs_max,
            bs_max_cnt,
            bs_viol,
        }
    }

    #[inline]
    pub fn adj_mask(&self, v: Node) -> u64 {
        self.adj[v as usize]
    }

    #[inline(always)]
    pub fn gain_off(&self, v: Node, to: usize) -> i32 {
        debug_assert!((v as usize) < self.n && to < self.k && to != self.part[v as usize] as usize);
        unsafe {
            *self.b.get_unchecked(v as usize * self.k + to) as i32
                - *self.pen.get_unchecked(v as usize) as i32
        }
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
        (0..self.m)
            .map(|e| self.connectivity(e as Edge) as u64)
            .sum()
    }

    pub fn adjacent_blocks(&self, v: Node, out: &mut Vec<Block>) {
        out.clear();
        let mut msk = self.adj[v as usize];
        while msk != 0 {
            out.push(msk.trailing_zeros() as Block);
            msk &= msk - 1;
        }
    }

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
        let vi = v as usize;
        let mut g = 0i32;
        let mut np = 0u16;
        debug_assert!(vi < self.n && fb < k && tb < k);
        unsafe {
            let lo = *self.nd_off.get_unchecked(vi) as usize;
            let hi = *self.nd_off.get_unchecked(vi + 1) as usize;
            let ne = self.nd_edges.as_ptr();
            let pc = self.pin_count.as_mut_ptr();
            let nz = self.nnz.as_mut_ptr();
            for idx in lo..hi {
                let e = *ne.add(idx) as usize;
                let base = e * k;
                let cf = *pc.add(base + fb);
                let ct = *pc.add(base + tb);
                *pc.add(base + fb) = cf.saturating_sub(1);
                *pc.add(base + tb) = ct.wrapping_add(1);
                let f1 = (cf == 1) as u32;
                let t0 = (ct == 0) as u32;
                *nz.add(e) = nz.add(e).read().wrapping_add(t0).wrapping_sub(f1);
                g += f1 as i32 - t0 as i32;
                np = np.wrapping_add((t0 ^ 1) as u16);
            }
            let ho = self.he_off.as_ptr();
            let hn = self.he_nodes.as_ptr();
            let bb = self.b.as_mut_ptr();
            let adj = self.adj.as_mut_ptr();
            let part = self.part.as_ptr();
            let pen = self.pen.as_mut_ptr();
            let clr = !(1u64 << fb);
            let set = 1u64 << tb;
            for idx in lo..hi {
                let e = *ne.add(idx) as usize;
                let base = e * k;
                let c_from = *pc.add(base + fb);
                let c_to = *pc.add(base + tb);
                let elo = *ho.add(e) as usize;
                let ehi = *ho.add(e + 1) as usize;
                if c_from == 0 {
                    for i in elo..ehi {
                        let u = *hn.add(i) as usize;
                        let p = bb.add(u * k + fb);
                        let x = (*p).wrapping_sub(1);
                        *p = x;
                        *adj.add(u) &= if x == 0 { clr } else { u64::MAX };
                    }
                } else if c_from == 1 {
                    for i in elo..ehi {
                        let u = *hn.add(i) as usize;
                        if *part.add(u) == from && u != vi {
                            *pen.add(u) = (*pen.add(u)).wrapping_sub(1);
                            break;
                        }
                    }
                }
                if c_to == 1 {
                    for i in elo..ehi {
                        let u = *hn.add(i) as usize;
                        let p = bb.add(u * k + tb);
                        let x = *p;
                        *adj.add(u) |= if x == 0 { set } else { 0 };
                        *p = x.wrapping_add(1);
                    }
                } else if c_to == 2 {
                    for i in elo..ehi {
                        let u = *hn.add(i) as usize;
                        if *part.add(u) == to && u != vi {
                            *pen.add(u) = (*pen.add(u)).wrapping_add(1);
                            break;
                        }
                    }
                }
            }
            *pen.add(vi) = np;
        }
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
        for (i, mask) in self.room.iter_mut().enumerate() {
            *mask &= !((1u64 << f) | (1u64 << t));
            let weight = i as u64 + 1;
            if nf as u64 + weight <= cap as u64 {
                *mask |= 1u64 << f;
            }
            if nt as u64 + weight <= cap as u64 {
                *mask |= 1u64 << t;
            }
        }
        let vf_o = of > cap || of == 0;
        let vf_n = nf > cap || nf == 0;
        if vf_o != vf_n {
            if vf_n {
                self.bs_viol += 1;
            } else {
                self.bs_viol -= 1;
            }
        }
        let vt_o = ot > cap || ot == 0;
        let vt_n = nt > cap || nt == 0;
        if vt_o != vt_n {
            if vt_n {
                self.bs_viol += 1;
            } else {
                self.bs_viol -= 1;
            }
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

    #[inline(always)]
    pub fn best_move_of(&self, u: Node) -> Option<(i32, Block)> {
        let ui = u as usize;
        debug_assert!(ui < self.n);
        let (fu, wu, am) = unsafe {
            (
                *self.part.get_unchecked(ui) as usize,
                *self.weight.get_unchecked(ui),
                *self.adj.get_unchecked(ui),
            )
        };
        debug_assert!(fu < self.k);
        let cap = self.max_block_size;
        let bsf = unsafe { *self.block_size.get_unchecked(fu) };
        let lim = cap.wrapping_sub(wu);
        let wm1 = wu.wrapping_sub(1);
        let room = self.room[(wm1 & 1) as usize] | ((wm1 >= 2) as u64).wrapping_neg();
        let open = ((bsf > wu) & (wu <= cap)) as u64;
        let msk = am & !(1u64 << fu) & room & open.wrapping_neg();
        if msk == 0 {
            return None;
        }
        let k = self.k;
        let row = unsafe { self.b.get_unchecked(ui * k..ui * k + k) };
        let bs = unsafe { self.block_size.get_unchecked(..k) };
        let mut best = 0u64;
        if msk.count_ones() <= SPARSE_MAX {
            let mut m = msk;
            while m != 0 {
                let j = m.trailing_zeros() as usize;
                m &= m - 1;
                let (sz, r) = unsafe { (*bs.get_unchecked(j), *row.get_unchecked(j) as u64) };
                let c = (r << 38) | ((!sz as u64) << 6) | (63 - j as u64);
                best = best.max(c & ((sz <= lim) as u64).wrapping_neg());
            }
        } else {
            let mut j = 0usize;
            while j + DENSE_LANES <= k {
                let mb = msk >> j;
                for t in 0..DENSE_LANES {
                    let (sz, r) =
                        unsafe { (*bs.get_unchecked(j + t), *row.get_unchecked(j + t) as u64) };
                    let c = (r << 38) | ((!sz as u64) << 6) | (63 - (j + t) as u64);
                    let keep = ((mb >> t) & 1) & (sz <= lim) as u64;
                    best = best.max(c & keep.wrapping_neg());
                }
                j += DENSE_LANES;
            }
            while j < k {
                let (sz, r) = unsafe { (*bs.get_unchecked(j), *row.get_unchecked(j) as u64) };
                let c = (r << 38) | ((!sz as u64) << 6) | (63 - j as u64);
                let keep = ((msk >> j) & 1) & (sz <= lim) as u64;
                best = best.max(c & keep.wrapping_neg());
                j += 1;
            }
        }
        if best == 0 {
            return None;
        }
        let pu = unsafe { *self.pen.get_unchecked(ui) } as i32;
        Some(((best >> 38) as i32 - pu, (63 - (best & 63)) as Block))
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
