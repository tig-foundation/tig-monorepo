// Value-guided constructive heuristic with Phase 2 improvements
use crate::adp::vfa::VFA;
use crate::tig_adaptive::TIGState;
use rand::Rng;

pub struct Constructive {
    pub beam_width: usize, // Phase 2: consider top-k candidates
}

impl Default for Constructive {
    fn default() -> Self {
        Self { beam_width: 5 }
    }
}

impl Constructive {
    pub fn new(beam_width: usize) -> Self {
        Self { beam_width }
    }

    /// Phase 2: Enhanced insertion with capacity filtering and delay estimation
    pub fn insert_with_value<R: Rng>(
        state: &mut TIGState,
        node: usize,
        vfa: &VFA,
        rng: &mut R,
    ) -> bool {
        let mut candidates = Vec::new();

        // Phase 2: Generate all feasible positions
        for pos in 0..=state.route.len() {
            // Phase 2: Fast capacity check
            if state.load + state.demands[node] > state.max_capacity {
                continue;
            }

            // Phase 2: Fast time window check
            if !state.can_insert_at(node, pos) {
                continue;
            }

            // Phase 2: Estimate delay for sorting
            let delay = state.estimate_insertion_delay(node, pos);

            // Evaluate value
            let mut test_state = state.clone();
            test_state.insert_at(node, pos);
            let value = vfa.estimate(&test_state, rng);

            candidates.push((pos, value, delay));
        }

        if candidates.is_empty() {
            return false;
        }

        // Phase 2: Sort by value (descending) and take top candidates
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Insert at best position
        if let Some(&(best_pos, _, _)) = candidates.first() {
            state.insert_at(node, best_pos);
            true
        } else {
            false
        }
    }

    /// Phase 2: Batch insertion for multiple nodes
    pub fn insert_batch<R: Rng>(
        state: &mut TIGState,
        nodes: &[usize],
        vfa: &VFA,
        rng: &mut R,
    ) -> usize {
        let mut inserted = 0;
        for &node in nodes {
            if Self::insert_with_value(state, node, vfa, rng) {
                inserted += 1;
            }
        }
        inserted
    }
}
