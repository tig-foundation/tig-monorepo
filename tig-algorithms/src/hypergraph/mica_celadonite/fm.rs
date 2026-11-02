use super::gain::{Block, Node, State};
use super::{shuffle, SPLITMIX_GAMMA};

pub struct Config {
    pub stop_factor: f64,
    pub seeds_per_search: usize,
    pub rounds: usize,
    pub max_steps: usize,
}

fn round_seed(round: usize) -> u64 {
    (round as u64).wrapping_mul(SPLITMIX_GAMMA).wrapping_add(1)
}

#[inline(always)]
fn degree(st: &State, v: Node) -> u32 {
    unsafe { *st.nd_off.get_unchecked(v as usize + 1) - *st.nd_off.get_unchecked(v as usize) }
}

struct StopRule {
    beta: f64,
    factor: f64,
    steps: usize,
    mk: f64,
    sk: f64,
}

impl StopRule {
    fn new(n: usize, factor: f64) -> Self {
        StopRule {
            beta: (n as f64).ln(),
            factor,
            steps: 0,
            mk: 0.0,
            sk: 0.0,
        }
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
        let var = if self.steps > 1 {
            self.sk / (self.steps - 1) as f64
        } else {
            0.0
        };
        self.steps as f64 >= (var / (self.mk * self.mk)) * self.factor
    }
}

struct Move {
    node: Node,
    from: Block,
}

struct KeyHeap<const FAST: bool> {
    a: Vec<u128>,
}

impl<const FAST: bool> KeyHeap<FAST> {
    fn with_capacity(c: usize) -> Self {
        let mut a = Vec::with_capacity(c + FAST as usize);
        if FAST {
            a.push(u128::MAX);
        }
        KeyHeap { a }
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.a.truncate(FAST as usize);
    }

    #[inline(always)]
    fn push(&mut self, x: u128) {
        let mut i = self.a.len();
        self.a.push(x);
        let p = self.a.as_mut_ptr();
        unsafe {
            while FAST || i > 0 {
                let up = if FAST { i >> 1 } else { (i - 1) >> 1 };
                let y = *p.add(up);
                if y >= x {
                    break;
                }
                *p.add(i) = y;
                i = up;
            }
            *p.add(i) = x;
        }
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<u128> {
        if FAST && self.a.len() == 1 {
            return None;
        }
        let last = self.a.pop()?;
        let n = self.a.len();
        if n == FAST as usize {
            return Some(last);
        }
        let p = self.a.as_mut_ptr();
        unsafe {
            *p.add(n) = 0;
            let top = *p.add(FAST as usize);
            let mut i = FAST as usize;
            loop {
                let c = 2 * i + usize::from(!FAST);
                if c >= n {
                    break;
                }
                let l = *p.add(c);
                let r = *p.add(c + 1);
                let (m, mv) = if r > l { (c + 1, r) } else { (c, l) };
                if mv <= last {
                    break;
                }
                *p.add(i) = mv;
                i = m;
            }
            *p.add(i) = last;
            Some(top)
        }
    }
}

#[inline(always)]
fn pack(g: i32, rank: u32, v: Node, b: Block) -> u128 {
    (((g as u32) ^ 0x8000_0000) as u128) << 96
        | (rank as u128) << 64
        | ((!v) as u128) << 32
        | b as u128
}

struct Scratch<const FAST: bool> {
    log: Vec<Move>,
    heap: KeyHeap<FAST>,
    touched: Vec<Node>,
    seen: Vec<u32>,
    stamp: u32,
}

impl<const FAST: bool> Scratch<FAST> {
    fn new(n: usize, log_cap: usize) -> Self {
        debug_assert!({
            let mut heap = KeyHeap::<FAST>::with_capacity(1);
            let mut values = [0, u128::MAX, 1, 0, 17, u128::MAX, 16, 2, 17];
            let mut valid = heap.pop().is_none();
            for &value in &values {
                heap.push(value);
            }
            values.sort_unstable();
            for &value in values.iter().rev() {
                valid &= heap.pop() == Some(value);
            }
            valid &= heap.pop().is_none();
            heap.push(1);
            heap.push(2);
            heap.clear();
            valid &= heap.pop().is_none();
            heap.push(0);
            valid && heap.pop() == Some(0) && heap.pop().is_none()
        });
        Scratch {
            log: Vec::with_capacity(log_cap),
            heap: KeyHeap::with_capacity(1024),
            touched: vec![0; n + 1],
            seen: vec![0u32; n],
            stamp: 0,
        }
    }
}

fn localized_search<const FAST: bool>(
    st: &mut State,
    seeds: &[Node],
    locked: &mut [bool],
    cfg: &Config,
    sc: &mut Scratch<FAST>,
) {
    sc.log.clear();
    sc.heap.clear();
    let mut stop = StopRule::new(st.n, cfg.stop_factor);
    let mut running = 0i32;
    let mut best = 0i32;
    let mut best_len = 0usize;

    for &u in seeds {
        if let Some((g, b)) = st.best_move_of(u) {
            sc.heap.push(pack(g, degree(st, u), u, b));
        }
    }

    while let Some(top) = sc.heap.pop() {
        let v = !((top >> 32) as u32);
        if unsafe { *locked.get_unchecked(v as usize) } {
            continue;
        }
        let (g_now, to) = match st.best_move_of(v) {
            Some(x) => x,
            None => continue,
        };
        let g_stale = (((top >> 96) as u32) ^ 0x8000_0000) as i32;
        let b_stale = top as u16 as Block;
        if (g_now != g_stale) | (to != b_stale) {
            sc.heap.push(pack(g_now, degree(st, v), v, to));
            continue;
        }
        let (from, w_from, w_to, weight) = unsafe {
            let from = *st.part.get_unchecked(v as usize);
            (
                from,
                *st.block_size.get_unchecked(from as usize),
                *st.block_size.get_unchecked(to as usize),
                *st.weight.get_unchecked(v as usize),
            )
        };
        let w_max = st.max_bs();
        let real = st.apply(v, to);
        unsafe {
            *locked.get_unchecked_mut(v as usize) = true;
        }
        sc.log.push(Move { node: v, from });
        running += real;
        stop.observe(real);

        let bal_eq = (running >= best) & (w_from == w_max) & (w_to + weight < w_max);
        if ((running > best) | bal_eq) & !st.imbalanced() {
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
        let mut tl = 0usize;
        {
            let seen = sc.seen.as_mut_ptr();
            let touched = sc.touched.as_mut_ptr();
            let lk = locked.as_ptr();
            let hn = st.he_nodes.as_ptr();
            let ho = st.he_off.as_ptr();
            let ne = st.nd_edges.as_ptr();
            unsafe {
                let e0 = *st.nd_off.get_unchecked(v as usize) as usize;
                let e1 = *st.nd_off.get_unchecked(v as usize + 1) as usize;
                for idx in e0..e1 {
                    let e = *ne.add(idx) as usize;
                    let lo = *ho.add(e) as usize;
                    let hi = *ho.add(e + 1) as usize;
                    if hi - lo > 64 {
                        continue;
                    }
                    if FAST {
                        let mut touch = |i: usize| {
                            let u = *hn.add(i);
                            let ui = u as usize;
                            let fresh = *seen.add(ui) != tok;
                            *seen.add(ui) = tok;
                            *touched.add(tl) = u;
                            tl += (fresh & !*lk.add(ui) & (u != v)) as usize;
                        };
                        let mut i = lo;
                        while hi - i >= 4 {
                            touch(i);
                            touch(i + 1);
                            touch(i + 2);
                            touch(i + 3);
                            i += 4;
                        }
                        while i < hi {
                            touch(i);
                            i += 1;
                        }
                    } else {
                        for i in lo..hi {
                            let u = *hn.add(i);
                            let ui = u as usize;
                            let fresh = *seen.add(ui) != tok;
                            *seen.add(ui) = tok;
                            *touched.add(tl) = u;
                            tl += (fresh & !*lk.add(ui) & (u != v)) as usize;
                        }
                    }
                }
            }
        }
        debug_assert!(sc.touched[..tl].iter().all(|&u| {
            u != v && !locked[u as usize] && sc.seen[u as usize] == tok
        }));
        for i in 0..tl {
            let u = unsafe { *sc.touched.get_unchecked(i) };
            if let Some((g, b)) = st.best_move_of(u) {
                sc.heap.push(pack(g, degree(st, u), u, b));
            }
        }
    }

    while sc.log.len() > best_len {
        let mv = sc.log.pop().unwrap();
        st.apply(mv.node, mv.from);
    }
}

pub fn refine_seeded<const FAST: bool>(st: &mut State, cfg: &Config, seeds: &[Node]) {
    if cfg.rounds == 0 || seeds.is_empty() {
        return;
    }
    let mut sc = Scratch::<FAST>::new(st.n, cfg.max_steps + 1);
    let mut locked = vec![false; st.n];
    let mut border = Vec::with_capacity(seeds.len());
    for round in 0..cfg.rounds {
        border.clear();
        for &v in seeds {
            if st.adj_mask(v) & !(1u64 << st.part[v as usize]) != 0 {
                border.push(v);
            }
        }
        if border.is_empty() {
            break;
        }
        shuffle(&mut border, round_seed(round));
        locked.fill(false);
        for chunk in border.chunks(cfg.seeds_per_search) {
            localized_search(st, chunk, &mut locked, cfg, &mut sc);
        }
    }
}

pub fn refine<const FAST: bool>(st: &mut State, cfg: &Config) {
    let mut sc = Scratch::<FAST>::new(st.n, cfg.max_steps + 1);
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
        shuffle(&mut border, round_seed(round));
        locked.fill(false);
        for chunk in border.chunks(cfg.seeds_per_search) {
            localized_search(st, chunk, &mut locked, cfg, &mut sc);
        }
    }
}

pub fn refine_filtered<const FAST: bool>(st: &mut State, cfg: &Config, seeds: Option<&[Node]>, min_gain: i32) {
    if cfg.rounds == 0 || seeds.map_or(false, |s| s.is_empty()) {
        return;
    }
    let mut sc = Scratch::<FAST>::new(st.n, cfg.max_steps.min(4096) + 1);
    let mut locked = vec![false; st.n];
    let mut border: Vec<Node> = Vec::with_capacity(seeds.map_or(st.n, |s| s.len()));
    let eligible = |st: &State, v: Node| -> bool {
        st.adj_mask(v) & !(1u64 << st.part[v as usize]) != 0
            && st.best_move_of(v).map_or(false, |(g, _)| g >= min_gain)
    };
    for round in 0..cfg.rounds {
        border.clear();
        match seeds {
            Some(list) => {
                for &v in list {
                    if eligible(st, v) {
                        border.push(v);
                    }
                }
            }
            None => {
                for v in 0..st.n as Node {
                    if eligible(st, v) {
                        border.push(v);
                    }
                }
            }
        }
        if border.is_empty() {
            break;
        }
        shuffle(&mut border, round_seed(round));
        locked.fill(false);
        for chunk in border.chunks(cfg.seeds_per_search) {
            localized_search(st, chunk, &mut locked, cfg, &mut sc);
        }
    }
}
