// Incremental km1 gain cache with an adjacent-block mask.

type PinCount = Vec<(Block, u32)>;

#[inline]
fn pc_get(pc: &PinCount, b: Block) -> u32 {
    match pc.binary_search_by_key(&b, |&(k, _)| k) {
        Ok(i) => pc[i].1,
        Err(_) => 0,
    }
}

#[inline]
fn pc_add(pc: &mut PinCount, b: Block, d: i32) {
    match pc.binary_search_by_key(&b, |&(k, _)| k) {
        Ok(i) => {
            let v = pc[i].1 as i32 + d;
            if v <= 0 { pc.remove(i); } else { pc[i].1 = v as u32; }
        }
        Err(i) => {
            if d > 0 { pc.insert(i, (b, d as u32)); }
        }
    }
}

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
    pub free: Vec<bool>,
    pin_count: Vec<PinCount>,
    b: Vec<u32>,
    pen: Vec<u32>,
    adj: Vec<u64>,
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

        let mut pin_count: Vec<PinCount> = vec![Vec::new(); m];
        for e in 0..m {
            for i in he_off[e]..he_off[e + 1] {
                let b = part[he_nodes[i as usize] as usize];
                pc_add(&mut pin_count[e], b, 1);
            }
        }
        let weight = weight.unwrap_or_else(|| vec![1u32; n]);
        let mut block_size = vec![0u32; k];
        for (v, &b) in part.iter().enumerate() {
            block_size[b as usize] += weight[v];
        }

        let mut b = vec![0u32; n * k];
        let mut pen = vec![0u32; n];
        for e in 0..m {
            let lo = he_off[e] as usize;
            let hi = he_off[e + 1] as usize;
            for i in lo..hi {
                let u = he_nodes[i] as usize;
                for &(blk, _) in pin_count[e].iter() {
                    b[u * k + blk as usize] += 1;
                }
                if pc_get(&pin_count[e], part[u]) > 1 {
                    pen[u] += 1;
                }
            }
        }

        assert!(k <= 64, "adjacent-block mask is limited to 64 blocks, k={}", k);
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
            free: Vec::new(),
            pin_count,
            b,
            pen,
            adj,
        }
    }

    #[inline]
    pub fn adj_mask(&self, v: Node) -> u64 {
        self.adj[v as usize]
    }

    #[inline]
    pub fn is_free(&self, v: Node) -> bool {
        self.free.is_empty() || self.free[v as usize]
    }

    #[inline]
    pub fn gain_cached(&self, v: Node, to: Block) -> i32 {
        if self.part[v as usize] == to {
            return 0;
        }
        self.b[v as usize * self.k + to as usize] as i32 - self.pen[v as usize] as i32
    }

    #[inline]
    pub fn edges_of(&self, v: Node) -> &[Edge] {
        &self.nd_edges[self.nd_off[v as usize] as usize..self.nd_off[v as usize + 1] as usize]
    }

    #[inline]
    pub fn count(&self, e: Edge, b: Block) -> u32 {
        pc_get(&self.pin_count[e as usize], b)
    }

    #[inline]
    pub fn connectivity(&self, e: Edge) -> u32 {
        self.pin_count[e as usize].len() as u32 - 1
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

    pub fn apply(&mut self, v: Node, to: Block) -> i32 {
        let from = self.part[v as usize];
        if from == to {
            return 0;
        }
        let mut g = 0i32;
        let lo = self.nd_off[v as usize] as usize;
        let hi = self.nd_off[v as usize + 1] as usize;
        for idx in lo..hi {
            let e = self.nd_edges[idx] as usize;
            let pc = &mut self.pin_count[e];
            if pc_get(pc, from) == 1 { g += 1; }
            pc_add(pc, from, -1);
            if pc_get(pc, to) == 0 { g -= 1; }
            pc_add(pc, to, 1);
        }
        for idx in lo..hi {
            let e = self.nd_edges[idx] as usize;
            let c_from = pc_get(&self.pin_count[e], from);
            let c_to = pc_get(&self.pin_count[e], to);
            let elo = self.he_off[e] as usize;
            let ehi = self.he_off[e + 1] as usize;
            if c_from == 0 {
                for i in elo..ehi {
                    let u = self.he_nodes[i] as usize;
                    let idx = u * self.k + from as usize;
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
                    let idx = u * self.k + to as usize;
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
        let mut np = 0u32;
        for idx in lo..hi {
            let e = self.nd_edges[idx] as usize;
            if pc_get(&self.pin_count[e], to) > 1 { np += 1; }
        }
        self.pen[v as usize] = np;

        self.part[v as usize] = to;
        let w = self.weight[v as usize];
        self.block_size[from as usize] -= w;
        self.block_size[to as usize] += w;
        g
    }

    #[inline]
    pub fn fits_node(&self, v: Node, to: Block) -> bool {
        self.block_size[to as usize] + self.weight[v as usize] <= self.max_block_size
    }

    pub fn imbalanced(&self) -> bool {
        self.block_size.iter().any(|&s| s > self.max_block_size || s == 0)
    }
}
