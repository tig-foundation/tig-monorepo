use smallvec::SmallVec;
use crate::tig_adaptive::TIGState;
use rand::prelude::SliceRandom;
use rand::Rng;

/// Individual representation that holds a complete TIGState and its cost
pub struct Individual {
    pub state: TIGState,
    pub cost: i32,
}

impl Individual {
    pub fn new(state: TIGState, cost: i32) -> Self {
        Self { state, cost }
    }
}

/// Micro population container
pub struct MicroPopulation {
    pub indivs: SmallVec<[Individual; 16]>,
}

impl MicroPopulation {
    pub fn new() -> Self {
        Self { indivs: SmallVec::new() }
    }

    pub fn insert(&mut self, ind: Individual) {
        self.indivs.push(ind);
        // optional: maintain sorted by cost or prune
    }
}

impl MicroPopulation {
    /// Select two parents by tournament (k=3)
    pub fn select_parents<R: Rng>(&self, rng: &mut R) -> Option<(&Individual, &Individual)> {
        if self.indivs.len() < 2 { return None; }
        let _k = 3usize.min(self.indivs.len());
        let mut idxs: Vec<usize> = (0..self.indivs.len()).collect();
        idxs.shuffle(rng);
        let a = &self.indivs[idxs[0]];
        let b = &self.indivs[idxs[1]];
        Some((a, b))
    }

    /// Simple arc-preserving crossover: preserve arcs common to both parents, then greedily insert remaining nodes.
    pub fn arc_preserving_crossover(&self, parent_a: &Individual, parent_b: &Individual) -> Individual {
        let pa = &parent_a.state;
        let pb = &parent_b.state;
        let n = pa.route.len();
        let mut child_route: Vec<usize> = Vec::with_capacity(n);

        // collect common successor mapping
        let mut succ_common = vec![usize::MAX; n.max(pb.route.len())];
        for i in 0..pa.route.len() - 1 {
            let u = pa.route.nodes[i];
            let v = pa.route.nodes[i + 1];
            // find u in pb
            if let Some(pos) = pb.route.pos.get(u).copied() {
                if pos != usize::MAX && pos + 1 < pb.route.len() && pb.route.nodes[pos + 1] == v {
                    if u >= succ_common.len() { succ_common.resize(u + 1, usize::MAX); }
                    succ_common[u] = v;
                }
            }
        }

        // Start from depot if present (assume index 0)
        let depot = pa.route.nodes[0];
        child_route.push(depot);

        // follow common arcs
        while child_route.len() < n {
            let last = *child_route.last().unwrap();
            if last < succ_common.len() && succ_common[last] != usize::MAX {
                let nxt = succ_common[last];
                if !child_route.contains(&nxt) {
                    child_route.push(nxt);
                    continue;
                }
            }

            // otherwise pick the node that minimizes insertion cost (greedy)
            let mut best = None;
            let mut best_cost = i32::MAX;
            for &cand in pa.route.nodes.iter() {
                if child_route.contains(&cand) { continue; }
                let last_node = *child_route.last().unwrap();
                let cost = pa.travel_time(last_node, cand);
                if cost < best_cost {
                    best_cost = cost;
                    best = Some(cand);
                }
            }
            if let Some(bn) = best { child_route.push(bn); } else { break; }
        }

        let child_state = TIGState::new(
            child_route,
            pa.time,
            pa.max_capacity,
            pa.tw_start.as_ref().to_vec(),
            pa.tw_end.as_ref().to_vec(),
            pa.service.as_ref().to_vec(),
            pa.distance_matrix_nested(),
            pa.demands.as_ref().to_vec(),
        );

        Individual::new(child_state, 0)
    }

    /// Simple mutation: apply a random swap/relocate/2-opt to individual's state
    pub fn mutate_individual<R: Rng>(&self, ind: &mut Individual, rng: &mut R) {
        let n = ind.state.route.len();
        if n < 4 { return; }
        match rng.gen_range(0..3) {
            0 => {
                // swap
                let i = rng.gen_range(1..n-2);
                let j = rng.gen_range(i+1..n-1);
                ind.state.route.swap(i, j);
                ind.state.recompute_times_from(i.min(j));
            }
            1 => {
                // relocate
                let i = rng.gen_range(1..n-1);
                let j = rng.gen_range(1..n);
                let node = ind.state.route.remove(i);
                let insert = if j > i { j - 1 } else { j };
                ind.state.route.insert(insert, node);
                ind.state.recompute_times_from(std::cmp::min(i, insert));
            }
            _ => {
                // 2-opt small
                let i = rng.gen_range(1..n-3);
                let j = rng.gen_range(i+1..(i+3).min(n-1));
                ind.state.route.reverse(i, j);
                ind.state.recompute_times_from(i);
            }
        }
    }
}
