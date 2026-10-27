// Exact local-search polish of a quadratic-knapsack packing (idea 129, offline candidate; not submitted).
// Exhaustive neighbourhoods with exact gains from a maintained contribution vector contrib[i] = sum_{j in S} V[i][j]:
// add 1, swap 1-1, swap 2-1 (two out, one in), swap 1-2 (one out, two in; bounded by each candidate's largest
// interaction). Best improvement within a neighbourhood, back to the first neighbourhood after any move, until no
// move improves. Deterministic; the value never decreases.
use tig_challenges::knapsack::Challenge;

pub struct Polish<'a> {
    v: &'a [Vec<i32>],
    w: Vec<i64>,
    cap: i64,
    n: usize,
    pub in_set: Vec<bool>,
    contrib: Vec<i64>,
    weight: i64,
    pub value: i64,
    row_max: Vec<i64>,
    pub moves: u32,
    pub by_type: [u32; 6],
    comps: Vec<Vec<usize>>,
    swap_tries: usize,
}

/// Search settings of the polish (defaults = the frozen operating point, docs/tig-innovator.md §10.7).
#[derive(Clone, Debug)]
pub struct Cfg {
    /// Pair values at or above which two items count as one cluster (scaled Jaccard, 0..1000); one cluster set per
    /// threshold, duplicates dropped.
    pub taus: Vec<i32>,
    /// Clusters larger than this are not tried as a block (hub-chained components).
    pub cluster_max: usize,
    /// Positive-estimate cluster exchanges applied exactly per call.
    pub swap_tries: usize,
}

impl Default for Cfg {
    fn default() -> Self {
        Cfg { taus: vec![150], cluster_max: 24, swap_tries: 8 }
    }
}

/// Sum of pair values inside a set.
fn internal(v: &[Vec<i32>], set: &[usize]) -> i64 {
    let mut t = 0i64;
    for x in 0..set.len() {
        for y in (x + 1)..set.len() {
            t += v[set[x]][set[y]] as i64;
        }
    }
    t
}

impl<'a> Polish<'a> {
    pub fn new(ch: &'a Challenge, items: &[usize], cfg: &Cfg) -> Self {
        let n = ch.num_items;
        let v = &ch.interaction_values[..];
        let w: Vec<i64> = ch.weights.iter().map(|&x| x as i64).collect();
        let mut in_set = vec![false; n];
        for &i in items {
            in_set[i] = true;
        }
        let mut contrib = vec![0i64; n];
        let mut weight = 0;
        for &i in items {
            weight += w[i];
            let row = &v[i];
            for j in 0..n {
                contrib[j] += row[j] as i64;
            }
        }
        let value = items.iter().map(|&i| contrib[i]).sum::<i64>() / 2;
        let row_max = (0..n).map(|i| *v[i].iter().max().unwrap_or(&0) as i64).collect();
        let mut comps: Vec<Vec<usize>> = Vec::new();
        for &tau in &cfg.taus {
            for c in components(v, n, tau) {
                if c.len() <= cfg.cluster_max && !comps.contains(&c) {
                    comps.push(c);
                }
            }
        }
        Polish { v, w, cap: ch.max_weight as i64, n, in_set, contrib, weight, value, row_max, moves: 0, by_type: [0; 6],
                 comps, swap_tries: cfg.swap_tries }
    }

    fn flip(&mut self, i: usize, on: bool) {
        let sign = if on { 1 } else { -1 };
        self.in_set[i] = on;
        self.weight += sign * self.w[i];
        let row = &self.v[i];
        for j in 0..self.n {
            self.contrib[j] += sign * row[j] as i64;
        }
    }

    /// best[w] = largest contribution of an unselected item of weight <= w (w = 0..=10), 0 when none is positive.
    fn best_out_by_weight(&self) -> [i64; 11] {
        let mut best = [0i64; 11];
        for b in 0..self.n {
            if !self.in_set[b] {
                let wb = self.w[b] as usize;
                if wb <= 10 && self.contrib[b] > best[wb] {
                    best[wb] = self.contrib[b];
                }
            }
        }
        for w in 1..11 {
            best[w] = best[w].max(best[w - 1]);
        }
        best
    }

    pub fn items(&self) -> Vec<usize> {
        (0..self.n).filter(|&i| self.in_set[i]).collect()
    }

    /// Runs the neighbourhoods to a local optimum; returns the total gain.
    pub fn run(&mut self, max_moves: u32) -> i64 {
        let start = self.value;
        while self.moves < max_moves {
            let t = if self.try_add() { 0 } else if self.try_11() { 1 } else if self.try_21() { 2 }
                else if self.try_12() { 3 } else if self.try_cluster() { 4 } else if self.try_swap_clusters() { 5 }
                else { break };
            self.by_type[t] += 1;
            self.moves += 1;
        }
        self.value - start
    }

    /// Brings a whole outside cluster in, then ejects the items with the lowest contribution per weight until the
    /// packing fits, then fills the slack greedily; kept only if the value rises, otherwise undone exactly.
    fn try_cluster(&mut self) -> bool {
        for c in 0..self.comps.len() {
            let add: Vec<usize> = self.comps[c].iter().cloned().filter(|&i| !self.in_set[i]).collect();
            if add.len() < 2 {
                continue;
            }
            let (v0, mut log) = (self.value, Vec::<(usize, bool)>::new());
            for &i in &add {
                self.value += self.contrib[i];
                self.flip(i, true);
                log.push((i, true));
            }
            while self.weight > self.cap {
                let mut worst: Option<(i64, i64, usize)> = None; // (contrib, weight, item) minimising contrib / weight
                for r in 0..self.n {
                    if self.in_set[r] {
                        let (cr, wr) = (self.contrib[r], self.w[r]);
                        if worst.map_or(true, |(cb, wb, _)| cr * wb < cb * wr) {
                            worst = Some((cr, wr, r));
                        }
                    }
                }
                let (cr, _, r) = worst.unwrap();
                self.value -= cr;
                self.flip(r, false);
                log.push((r, false));
            }
            loop {
                let slack = self.cap - self.weight;
                let mut best: Option<(i64, usize)> = None;
                for b in 0..self.n {
                    if !self.in_set[b] && self.w[b] <= slack && self.contrib[b] > 0
                        && best.map_or(true, |(g, _)| self.contrib[b] > g) {
                        best = Some((self.contrib[b], b));
                    }
                }
                match best {
                    Some((g, b)) => {
                        self.value += g;
                        self.flip(b, true);
                        log.push((b, true));
                    }
                    None => break,
                }
            }
            if self.value > v0 {
                return true;
            }
            for &(i, on) in log.iter().rev() {
                if on {
                    self.flip(i, false);
                    self.value -= self.contrib[i];
                } else {
                    self.value += self.contrib[i];
                    self.flip(i, true);
                }
            }
            debug_assert_eq!(self.value, v0);
        }
        false
    }

    /// Cluster-for-cluster exchange: drop the selected part of one cluster, take the unselected part of another
    /// (or nothing), then fill the slack greedily. Candidates are ranked by an exact-before-repair estimate
    /// add(c2) - loss(c1) - cross(c1, c2) among weight-feasible pairs; the best few are applied exactly and the
    /// first that raises the value is kept.
    fn try_swap_clusters(&mut self) -> bool {
        let slack = self.cap - self.weight;
        let nc = self.comps.len();
        let mut ins: Vec<Vec<usize>> = Vec::with_capacity(nc);
        let mut outs: Vec<Vec<usize>> = Vec::with_capacity(nc);
        let (mut loss, mut gain, mut w_in, mut w_out) = (vec![0i64; nc], vec![0i64; nc], vec![0i64; nc], vec![0i64; nc]);
        for c in 0..nc {
            let (a, b): (Vec<usize>, Vec<usize>) = self.comps[c].iter().partition(|&&i| self.in_set[i]);
            loss[c] = a.iter().map(|&i| self.contrib[i]).sum::<i64>() - internal(self.v, &a);
            gain[c] = b.iter().map(|&i| self.contrib[i]).sum::<i64>() + internal(self.v, &b);
            w_in[c] = a.iter().map(|&i| self.w[i]).sum();
            w_out[c] = b.iter().map(|&i| self.w[i]).sum();
            ins.push(a);
            outs.push(b);
        }
        // (estimate, drop cluster, take cluster or usize::MAX for drop-and-refill)
        let mut cands: Vec<(i64, usize, usize)> = Vec::new();
        for c1 in 0..nc {
            if ins[c1].is_empty() {
                continue;
            }
            for c2 in 0..nc {
                if c2 == c1 || outs[c2].len() < 2 || w_out[c2] > w_in[c1] + slack || gain[c2] <= loss[c1] {
                    continue;
                }
                let cross: i64 = ins[c1].iter().map(|&r| outs[c2].iter().map(|&a| self.v[r][a] as i64).sum::<i64>()).sum();
                let est = gain[c2] - loss[c1] - cross;
                if est > 0 {
                    cands.push((est, c1, c2));
                }
            }
            cands.push((0, c1, usize::MAX));
        }
        cands.sort_unstable_by(|p, q| q.0.cmp(&p.0).then(p.1.cmp(&q.1)).then(p.2.cmp(&q.2)));
        let mut tried = 0;
        for &(est, c1, c2) in &cands {
            if est > 0 {
                if tried >= self.swap_tries {
                    continue;
                }
                tried += 1;
            }
            let v0 = self.value;
            let mut log = Vec::<(usize, bool)>::new();
            for &r in &ins[c1] {
                self.value -= self.contrib[r];
                self.flip(r, false);
                log.push((r, false));
            }
            if c2 != usize::MAX {
                for &a in &outs[c2] {
                    self.value += self.contrib[a];
                    self.flip(a, true);
                    log.push((a, true));
                }
            }
            self.fill(&mut log);
            if self.value > v0 {
                return true;
            }
            self.undo(&log);
            debug_assert_eq!(self.value, v0);
        }
        false
    }

    /// Greedy fill of the slack by the largest positive contribution, logging every flip.
    fn fill(&mut self, log: &mut Vec<(usize, bool)>) {
        loop {
            let slack = self.cap - self.weight;
            let mut best: Option<(i64, usize)> = None;
            for b in 0..self.n {
                if !self.in_set[b] && self.w[b] <= slack && self.contrib[b] > 0
                    && best.map_or(true, |(g, _)| self.contrib[b] > g) {
                    best = Some((self.contrib[b], b));
                }
            }
            match best {
                Some((g, b)) => {
                    self.value += g;
                    self.flip(b, true);
                    log.push((b, true));
                }
                None => break,
            }
        }
    }

    fn undo(&mut self, log: &[(usize, bool)]) {
        for &(i, on) in log.iter().rev() {
            if on {
                self.flip(i, false);
                self.value -= self.contrib[i];
            } else {
                self.value += self.contrib[i];
                self.flip(i, true);
            }
        }
    }

    fn try_add(&mut self) -> bool {
        let slack = self.cap - self.weight;
        let mut best: Option<(i64, usize)> = None;
        for b in 0..self.n {
            if !self.in_set[b] && self.w[b] <= slack && self.contrib[b] > 0 {
                if best.map_or(true, |(g, _)| self.contrib[b] > g) {
                    best = Some((self.contrib[b], b));
                }
            }
        }
        if let Some((g, b)) = best {
            self.flip(b, true);
            self.value += g;
            return true;
        }
        false
    }

    fn try_11(&mut self) -> bool {
        let slack = self.cap - self.weight;
        let ins: Vec<usize> = (0..self.n).filter(|&i| self.in_set[i]).collect();
        let outs: Vec<usize> = (0..self.n).filter(|&i| !self.in_set[i]).collect();
        let mut best: Option<(i64, usize, usize)> = None;
        let bw = self.best_out_by_weight();
        for &a in &ins {
            let room = slack + self.w[a];
            let ca = self.contrib[a];
            // V >= 0, so the gain is at most the best outside contribution that fits minus contrib[a]
            if bw[room.clamp(0, 10) as usize] - ca <= best.map_or(0, |(bg, ..)| bg) {
                continue;
            }
            let va = &self.v[a];
            for &b in &outs {
                if self.w[b] > room {
                    continue;
                }
                let g = self.contrib[b] - va[b] as i64 - ca;
                if g > 0 && best.map_or(true, |(bg, _, _)| g > bg) {
                    best = Some((g, a, b));
                }
            }
        }
        if let Some((g, a, b)) = best {
            self.flip(a, false);
            self.flip(b, true);
            self.value += g;
            return true;
        }
        false
    }

    fn try_21(&mut self) -> bool {
        let slack = self.cap - self.weight;
        let ins: Vec<usize> = (0..self.n).filter(|&i| self.in_set[i]).collect();
        let outs: Vec<usize> = (0..self.n).filter(|&i| !self.in_set[i]).collect();
        let mut best: Option<(i64, usize, usize, usize)> = None;
        let bw = self.best_out_by_weight();
        for x in 0..ins.len() {
            let a1 = ins[x];
            for y in (x + 1)..ins.len() {
                let a2 = ins[y];
                let room = slack + self.w[a1] + self.w[a2];
                let loss = self.contrib[a1] + self.contrib[a2] - self.v[a1][a2] as i64;
                if bw[room.clamp(0, 10) as usize] - loss <= best.map_or(0, |(bg, ..)| bg) {
                    continue;
                }
                let (v1, v2) = (&self.v[a1], &self.v[a2]);
                for &b in &outs {
                    if self.w[b] > room {
                        continue;
                    }
                    let g = self.contrib[b] - v1[b] as i64 - v2[b] as i64 - loss;
                    if g > 0 && best.map_or(true, |(bg, ..)| g > bg) {
                        best = Some((g, a1, a2, b));
                    }
                }
            }
        }
        if let Some((g, a1, a2, b)) = best {
            self.flip(a1, false);
            self.flip(a2, false);
            self.flip(b, true);
            self.value += g;
            return true;
        }
        false
    }

    fn try_12(&mut self) -> bool {
        let slack = self.cap - self.weight;
        let ins: Vec<usize> = (0..self.n).filter(|&i| self.in_set[i]).collect();
        let outs: Vec<usize> = (0..self.n).filter(|&i| !self.in_set[i]).collect();
        let mut best: Option<(i64, usize, usize, usize)> = None;
        let mut cand: Vec<(i64, usize)> = Vec::with_capacity(outs.len());
        let (mut top1, mut top2, mut rmo) = (0i64, 0i64, 0i64);
        for &b in &outs {
            let c = self.contrib[b].max(0);
            if c > top1 {
                top2 = top1;
                top1 = c;
            } else if c > top2 {
                top2 = c;
            }
            rmo = rmo.max(self.row_max[b]);
        }
        for &a in &ins {
            let room = slack + self.w[a];
            let ca = self.contrib[a];
            if top1 + top2 + rmo - ca <= best.map_or(0, |(bg, ..)| bg) {
                continue;
            }
            let va = &self.v[a];
            cand.clear();
            for &b in &outs {
                if self.w[b] < room {
                    cand.push((self.contrib[b] - va[b] as i64, b));
                }
            }
            cand.sort_unstable_by(|p, q| q.0.cmp(&p.0).then(p.1.cmp(&q.1)));
            let need = ca + best.map_or(0, |(bg, ..)| bg);
            let rm_all = cand.iter().map(|&(_, b)| self.row_max[b]).max().unwrap_or(0);
            for x in 0..cand.len() {
                let (g1, b1) = cand[x];
                // a partner adds at most the next candidate's g plus an interaction <= b1's row maximum
                if x + 1 >= cand.len() || g1 + cand[x + 1].0 + rm_all <= need {
                    break;
                }
                if g1 + cand[x + 1].0 + self.row_max[b1] <= need {
                    continue;
                }
                let vb1 = &self.v[b1];
                let r1 = room - self.w[b1];
                for y in (x + 1)..cand.len() {
                    let (g2, b2) = cand[y];
                    if g1 + g2 + self.row_max[b1] <= need {
                        break;
                    }
                    if self.w[b2] > r1 {
                        continue;
                    }
                    let g = g1 + g2 + vb1[b2] as i64 - ca;
                    if g > 0 && best.map_or(true, |(bg, ..)| g > bg) {
                        best = Some((g, a, b1, b2));
                    }
                }
            }
        }
        if let Some((g, a, b1, b2)) = best {
            self.flip(a, false);
            self.flip(b1, true);
            self.flip(b2, true);
            self.value += g;
            return true;
        }
        false
    }
}

/// Connected components of the graph with an edge where V[i][j] >= tau (union-find), components of size >= 2.
fn components(v: &[Vec<i32>], n: usize, tau: i32) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut Vec<usize>, mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for i in 0..n {
        let row = &v[i];
        for j in (i + 1)..n {
            if row[j] >= tau {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                if a != b {
                    parent[a.max(b)] = a.min(b);
                }
            }
        }
    }
    let mut by: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        by.entry(r).or_default().push(i);
    }
    by.into_values().filter(|c| c.len() >= 2).collect()
}
