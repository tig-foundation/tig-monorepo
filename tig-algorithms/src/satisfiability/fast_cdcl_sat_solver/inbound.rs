/*!
Copyright 2024 Chad Blanchard

Licensed under the TIG Inbound Game License v1.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/


use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::VecDeque;
use tig_challenges::satisfiability::*;

const VSIDS_INCREMENT: f64 = 0.5;
const VSIDS_DECAY: f64 = 0.95;

struct Solver {
    num_vars: usize,
    clauses: Vec<Vec<i32>>,
    assignments: Vec<bool>,
    decision_level: Vec<i32>,
    vsids_scores: Vec<f64>,
    watchers: Vec<Vec<usize>>,
    trail: VecDeque<i32>,
    rng: StdRng,
}

impl Solver {
    fn new(challenge: &Challenge) -> Self {
        let num_vars = challenge.difficulty.num_variables;
        let mut watchers = vec![Vec::new(); 2 * num_vars];
        let mut clauses = Vec::new();

        for (i, clause) in challenge.clauses.iter().enumerate() {
            let mut new_clause = clause.clone();
            new_clause.sort_unstable();
            new_clause.dedup();
            if new_clause.len() >= 2 {
                watchers[Self::lit_to_idx(new_clause[0])].push(i);
                watchers[Self::lit_to_idx(new_clause[1])].push(i);
            }
            clauses.push(new_clause);
        }

        Solver {
            num_vars,
            clauses,
            assignments: vec![false; num_vars],
            decision_level: vec![-1; num_vars],
            vsids_scores: vec![0.0; num_vars],
            watchers,
            trail: VecDeque::new(),
            rng: StdRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64),
        }
    }

    fn lit_to_idx(lit: i32) -> usize {
        (lit.abs() as usize - 1) * 2 + if lit > 0 { 0 } else { 1 }
    }

    fn assign(&mut self, lit: i32) {
        let var = lit.abs() as usize - 1;
        self.assignments[var] = lit > 0;
        self.trail.push_back(lit);
    }

    fn unassign(&mut self, lit: i32) {
        let var = lit.abs() as usize - 1;
        self.assignments[var] = false;
        self.decision_level[var] = -1;
    }

    fn propagate(&mut self) -> Option<Vec<i32>> {
        while let Some(lit) = self.trail.pop_front() {
            let neg_lit_idx = Self::lit_to_idx(-lit);
            let mut i = 0;
            while i < self.watchers[neg_lit_idx].len() {
                let clause_idx = self.watchers[neg_lit_idx][i];
                
                // Check if the first watched literal is satisfied
                if self.evaluate_lit(self.clauses[clause_idx][0]) {
                    i += 1;
                    continue;
                }
                
                // Try to find a new watch
                let mut found_new_watch = false;
                for j in 2..self.clauses[clause_idx].len() {
                    if !self.evaluate_lit(-self.clauses[clause_idx][j]) {
                        // Found a new watch, update watchers
                        self.clauses[clause_idx].swap(1, j);
                        let new_lit_idx = Self::lit_to_idx(self.clauses[clause_idx][1]);
                        self.watchers[new_lit_idx].push(clause_idx);
                        self.watchers[neg_lit_idx].swap_remove(i);
                        found_new_watch = true;
                        break;
                    }
                }
                
                if !found_new_watch {
                    // Check if the clause is unit or conflicting
                    if self.evaluate_lit(-self.clauses[clause_idx][1]) {
                        // Conflict found
                        self.trail.push_front(lit); // Restore the trail
                        return Some(self.clauses[clause_idx].clone());
                    } else {
                        // Unit clause, propagate the other watched literal
                        self.assign(self.clauses[clause_idx][1]);
                        i += 1;
                    }
                }
            }
        }
        None
    }
    fn evaluate_lit(&self, lit: i32) -> bool {
        let var = lit.abs() as usize - 1;
        self.decision_level[var] >= 0 && self.assignments[var] == (lit > 0)
    }

    fn analyze_conflict(&mut self, conflict: Vec<i32>) -> (Vec<i32>, i32) {
        let mut seen = vec![false; self.num_vars];
        let mut learnt_clause = Vec::new();
        let mut conflict_side = conflict;
        let mut counter = 0;
        let mut backtrack_level = 0;

        loop {
            for &lit in &conflict_side {
                let var = lit.abs() as usize - 1;
                if !seen[var] {
                    seen[var] = true;
                    self.vsids_scores[var] += VSIDS_INCREMENT;
                    if self.decision_level[var] == self.decision_level[conflict_side[0].abs() as usize - 1] {
                        counter += 1;
                    } else if self.decision_level[var] > 0 {
                        learnt_clause.push(-lit);
                        backtrack_level = backtrack_level.max(self.decision_level[var]);
                    }
                }
            }

            loop {
                let lit = self.trail.pop_back().unwrap();
                let var = lit.abs() as usize - 1;
                if seen[var] {
                    counter -= 1;
                    if counter == 0 {
                        learnt_clause.push(lit);
                        self.trail.push_back(lit);
                        return (learnt_clause, backtrack_level);
                    }
                }
                self.unassign(lit);
            }
        }
    }

    fn decay_vsids(&mut self) {
        for score in &mut self.vsids_scores {
            *score *= VSIDS_DECAY;
        }
    }

    fn pick_branching_variable(&mut self) -> i32 {
        let var = (0..self.num_vars)
            .filter(|&var| self.decision_level[var] < 0)
            .max_by(|&a, &b| self.vsids_scores[a].partial_cmp(&self.vsids_scores[b]).unwrap())
            .unwrap();
        (var as i32 + 1) * if self.rng.gen() { 1 } else { -1 }
    }

    fn solve(&mut self) -> Option<Solution> {
        loop {
            match self.propagate() {
                Some(conflict) => {
                    if self.decision_level.iter().all(|&level| level <= 0) {
                        return None;
                    }
                    let (learnt_clause, backtrack_level) = self.analyze_conflict(conflict);
                    while self.trail.back().map_or(false, |&lit| {
                        self.decision_level[lit.abs() as usize - 1] > backtrack_level
                    }) {
                        let lit = self.trail.pop_back().unwrap();
                        self.unassign(lit);
                    }
                    self.clauses.push(learnt_clause.clone());
                    let idx = self.clauses.len() - 1;
                    self.watchers[Self::lit_to_idx(learnt_clause[0])].push(idx);
                    if learnt_clause.len() > 1 {
                        self.watchers[Self::lit_to_idx(learnt_clause[1])].push(idx);
                    }
                    self.assign(learnt_clause[0]);
                }
                None => {
                    if self.trail.len() == self.num_vars {
                        return Some(Solution {
                            variables: self.assignments.clone(),
                        });
                    }
                    let lit = self.pick_branching_variable();
                    self.assign(lit);
                    let var = lit.abs() as usize - 1;
                    self.decision_level[var] = self.decision_level.iter().max().unwrap() + 1;
                }
            }
            self.decay_vsids();
        }
    }
}

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let mut solver = Solver::new(challenge);
    Ok(solver.solve())
}
#[cfg(feature = "cuda")]
mod gpu_optimisation {
    use super::*;
    use cudarc::driver::*;
    use std::{collections::HashMap, sync::Arc};
    use tig_challenges::CudaKernel;

    // set KERNEL to None if algorithm only has a CPU implementation
    pub const KERNEL: Option<CudaKernel> = None;

    // Important! your GPU and CPU version of the algorithm should return the same result
    pub fn cuda_solve_challenge(
        challenge: &Challenge,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> anyhow::Result<Option<Solution>> {
        solve_challenge(challenge)
    }
}
#[cfg(feature = "cuda")]
pub use gpu_optimisation::{cuda_solve_challenge, KERNEL};
