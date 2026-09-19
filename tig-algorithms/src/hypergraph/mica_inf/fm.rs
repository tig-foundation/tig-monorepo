// Fiduccia-Mattheyses: localized searches seeded from a shuffled border, with rollback to the
// best prefix and an adaptive stopping rule.

use super::gain::{Block, Node, State};

pub struct Config {
    pub stop_factor: f64,
    pub seeds_per_search: usize,
    pub rounds: usize,
    pub max_steps: usize,
}

#[inline]
fn degree(st: &State, v: Node) -> u32 {
    st.nd_off[v as usize + 1] - st.nd_off[v as usize]
}

// Stops a search once its step count exceeds factor * variance / mean^2 of the observed gains
// (never before ln(n) steps).
struct StopRule {
    beta: f64,
    factor: f64,
    steps: usize,
    mk: f64,
    sk: f64,
}

impl StopRule {
    fn new(n: usize, factor: f64) -> Self {
        StopRule { beta: (n as f64).ln(), factor, steps: 0, mk: 0.0, sk: 0.0 }
    }
    fn reset(&mut self) {
        self.steps = 0;
        self.mk = 0.0;
        self.sk = 0.0;
    }
    fn observe(&mut self, gain: i32) {
        self.steps += 1;
        let g = gain as f64;
        if self.steps == 1 {
            self.mk = g;
            self.sk = 0.0;
        } else {
            let prev = self.mk;
            self.mk = prev + (g - prev) / self.steps as f64;
            self.sk += (g - prev) * (g - self.mk);
        }
    }
    fn should_stop(&self) -> bool {
        if (self.steps as f64) <= self.beta {
            return false;
        }
        let var = if self.steps > 1 { self.sk / (self.steps - 1) as f64 } else { 0.0 };
        self.steps as f64 >= (var / (self.mk * self.mk)) * self.factor
    }
}

struct Move {
    node: Node,
    from: Block,
}

struct Scratch {
    log: Vec<Move>,
    heap: std::collections::BinaryHeap<(i32, u32, std::cmp::Reverse<Node>, Block)>,
    touched: Vec<Node>,
    seen: Vec<u32>,
    stamp: u32,
}

// One search from `seeds`, rolled back to its best prefix.
fn localized_search(
    st: &mut State,
    seeds: &[Node],
    locked: &mut [bool],
    cfg: &Config,
    sc: &mut Scratch,
) {
    sc.log.clear();
    sc.heap.clear();
    let mut stop = StopRule::new(st.n, cfg.stop_factor);
    let mut running = 0i32;
    let mut best = 0i32;
    let mut best_len = 0usize;

    // Equal gains are ranked by degree, then by node index.
    let rank = |st: &State, v: Node| -> u32 { degree(st, v) };

    for &u in seeds {
        if let Some((g, b)) = st.best_move_of(u) {
            sc.heap.push((g, rank(st, u), std::cmp::Reverse(u), b));
        }
    }

    while let Some((g_stale, _r_stale, std::cmp::Reverse(v), b_stale)) = sc.heap.pop() {
        if locked[v as usize] {
            continue;
        }
        let (g_now, to) = match st.best_move_of(v) {
            Some(x) => x,
            None => continue,
        };
        if g_now != g_stale || to != b_stale {
            sc.heap.push((g_now, rank(st, v), std::cmp::Reverse(v), to));
            continue;
        }
        let from = st.part[v as usize];
        let w_from = st.block_size[from as usize];
        let w_to = st.block_size[to as usize];
        let w_max = st.max_bs();
        let real = st.apply(v, to);
        locked[v as usize] = true;
        sc.log.push(Move { node: v, from });
        running += real;
        stop.observe(real);

        // An equal-gain prefix is kept when it moves a node out of a largest block into a smaller one.
        let bal_eq = running >= best && w_from == w_max && w_to + st.weight[v as usize] < w_max;
        if (running > best || bal_eq) && !st.imbalanced() {
            best = running;
            best_len = sc.log.len();
            stop.reset();
        }

        if stop.should_stop() || sc.log.len() >= cfg.max_steps {
            break;
        }

        if sc.stamp == u32::MAX {
            for s in sc.seen.iter_mut() {
                *s = 0;
            }
            sc.stamp = 0;
        }
        sc.stamp += 1;
        let tok = sc.stamp;
        sc.touched.clear();
        for idx in st.nd_off[v as usize]..st.nd_off[v as usize + 1] {
            let e = st.nd_edges[idx as usize] as usize;
            let lo = st.he_off[e] as usize;
            let hi = st.he_off[e + 1] as usize;
            if hi - lo > 64 {
                continue;
            }
            for i in lo..hi {
                let u = st.he_nodes[i];
                if !locked[u as usize] && u != v && sc.seen[u as usize] != tok {
                    sc.seen[u as usize] = tok;
                    sc.touched.push(u);
                }
            }
        }
        for i in 0..sc.touched.len() {
            let u = sc.touched[i];
            if let Some((g, b)) = st.best_move_of(u) {
                sc.heap.push((g, rank(st, u), std::cmp::Reverse(u), b));
            }
        }
    }

    while sc.log.len() > best_len {
        let mv = sc.log.pop().unwrap();
        st.apply(mv.node, mv.from);
    }
}

pub fn refine(st: &mut State, cfg: &Config) {
    let mut sc = Scratch {
        log: Vec::with_capacity(cfg.max_steps + 1),
        heap: std::collections::BinaryHeap::with_capacity(1024),
        touched: Vec::with_capacity(256),
        seen: vec![0u32; st.n],
        stamp: 0,
    };
    let mut locked = vec![false; st.n];
    let mut border: Vec<Node> = Vec::with_capacity(st.n);
    for round in 0..cfg.rounds {
        border.clear();
        for v in 0..st.n as Node {
            let p = st.part[v as usize];
            if st.adj_mask(v) & !(1u64 << p) != 0 {
                border.push(v);
            }
        }
        if border.is_empty() {
            break;
        }
        // Fisher-Yates with a splitmix64 stream seeded by the round.
        let mut z = (round as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        let mut next = move || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        for i in (1..border.len()).rev() {
            let j = (next() % (i as u64 + 1)) as usize;
            border.swap(i, j);
        }
        for f in locked.iter_mut() {
            *f = false;
        }
        for chunk in border.chunks(cfg.seeds_per_search) {
            localized_search(st, chunk, &mut locked, cfg, &mut sc);
        }
    }
}
