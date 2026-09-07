// Exact upper envelopes for the unchanged 1->2, 2->1 and 2->2 move loops.
// The caller uses them only when the public CSR proves nonnegative interactions,
// all weights are in 1..10 and the maintained state is within a bounded capacity.
struct AdditionEnvelope {
    rows: Vec<[i64; 11]>,
    by_capacity: [i64; 21],
    conditioned: bool,
}
impl AdditionEnvelope {
    fn new(state: &State, tsn: &TopNeighbors, unused: &[usize], conditioned: bool) -> Self {
        let mut rows = Vec::with_capacity(unused.len());
        let mut by_capacity = [i64::MIN; 21];
        for &a1 in unused {
            let mut row = [i64::MIN; 11];
            let wa1 = state.ch.weights[a1] as usize;
            for &a2 in &tsn.friends[a1] {
                if state.selected_bit[a2] || a1 == a2 { continue; }
                let wa2 = state.ch.weights[a2] as usize;
                let gain = state.contrib[a2] as i64 + state.ch.interaction_values[a1][a2] as i64;
                row[wa2] = row[wa2].max(gain);
                by_capacity[wa1 + wa2] = by_capacity[wa1 + wa2].max(state.contrib[a1] as i64 + gain);
            }
            for w in 1..=10 { row[w] = row[w].max(row[w - 1]); }
            rows.push(row);
        }
        for w in 1..=20 { by_capacity[w] = by_capacity[w].max(by_capacity[w - 1]); }
        Self { rows, by_capacity, conditioned }
    }
    #[inline(always)]
    fn row(&self, ai: usize, remaining: u32) -> Option<i64> {
        let w = if self.conditioned { remaining.min(10) as usize } else { 10 };
        let value = self.rows[ai][w];
        if value == i64::MIN { None } else { Some(value) }
    }
    #[inline(always)]
    fn overall(&self, budget: u32) -> Option<i64> {
        let w = if self.conditioned { budget.min(20) as usize } else { 20 };
        let value = self.by_capacity[w];
        if value == i64::MIN { None } else { Some(value) }
    }
}
struct RemovalEnvelope {
    rows: Vec<[i64; 11]>,
    by_required_weight: [i64; 21],
    conditioned: bool,
}
impl RemovalEnvelope {
    fn new(state: &State, tsn: &TopNeighbors, used: &[usize], conditioned: bool) -> Self {
        let mut rows = Vec::with_capacity(used.len());
        let mut by_required_weight = [i64::MIN; 21];
        for &r1 in used {
            let mut row = [i64::MIN; 11];
            let wr1 = state.ch.weights[r1] as usize;
            for &r2 in &tsn.friends[r1] {
                if !state.selected_bit[r2] || r1 == r2 { continue; }
                let wr2 = state.ch.weights[r2] as usize;
                let gain = state.ch.interaction_values[r1][r2] as i64 - state.contrib[r2] as i64;
                row[wr2] = row[wr2].max(gain);
                by_required_weight[wr1 + wr2] = by_required_weight[wr1 + wr2].max(gain - state.contrib[r1] as i64);
            }
            for w in (0..10).rev() { row[w] = row[w].max(row[w + 1]); }
            rows.push(row);
        }
        for w in (0..20).rev() { by_required_weight[w] = by_required_weight[w].max(by_required_weight[w + 1]); }
        Self { rows, by_required_weight, conditioned }
    }
    #[inline(always)]
    fn row(&self, ri: usize, required: u32) -> Option<i64> {
        let w = if self.conditioned { required as usize } else { 0 };
        if w > 10 { return None; }
        let value = self.rows[ri][w];
        if value == i64::MIN { None } else { Some(value) }
    }
    #[inline(always)]
    fn overall(&self, required: u32) -> Option<i64> {
        let w = if self.conditioned { required as usize } else { 0 };
        if w > 20 { return None; }
        let value = self.by_required_weight[w];
        if value == i64::MIN { None } else { Some(value) }
    }
}
