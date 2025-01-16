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
use tig_challenges::satisfiability::*;
use std::time::{Instant, Duration};

const MAX_FLIPS: usize = 10_000_000;
const MAX_TRIES: usize = 10;
const NOISE: f64 = 0.57; // Noise parameter for WalkSAT

struct WalkSAT {
    num_vars: usize,
    clauses: Vec<Vec<i32>>,
    assignment: Vec<bool>,
    unsat_clauses: Vec<usize>,
    var_scores: Vec<i32>,
    rng: StdRng,
    clause_states: Vec<u32>,
    var_last_flip: Vec<usize>,
}

impl WalkSAT {
    fn new(challenge: &Challenge) -> Self {
        let num_vars = challenge.difficulty.num_variables;
        let clauses = challenge.clauses.clone();
        let mut rng = StdRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64);
        
        let assignment = (0..num_vars).map(|_| rng.gen_bool(0.5)).collect();
        let clause_states = vec![0; clauses.len()];
        
        let mut solver = WalkSAT {
            num_vars,
            clauses,
            assignment,
            unsat_clauses: Vec::new(),
            var_scores: vec![0; num_vars],
            rng,
            clause_states,
            var_last_flip: vec![0; num_vars],
        };
        
        solver.initialize_clause_states();
        solver
    }

    fn initialize_clause_states(&mut self) {
        for (i, clause) in self.clauses.iter().enumerate() {
            let mut state = 0u32;
            for &lit in clause {
                let var = (lit.abs() - 1) as usize;
                if (lit > 0) == self.assignment[var] {
                    state |= 1;
                }
            }
            self.clause_states[i] = state;
            if state == 0 {
                self.unsat_clauses.push(i);
            }
        }
    }

    fn flip_var(&mut self, var: usize, step: usize) {
        self.assignment[var] = !self.assignment[var];
        self.var_last_flip[var] = step;

        for (i, clause) in self.clauses.iter().enumerate() {
            if clause.iter().any(|&lit| (lit.abs() - 1) as usize == var) {
                let old_state = self.clause_states[i];
                let new_state = old_state ^ 1;
                self.clause_states[i] = new_state;

                if old_state == 0 {
                    self.unsat_clauses.retain(|&x| x != i);
                } else if new_state == 0 {
                    self.unsat_clauses.push(i);
                }

                for &lit in clause {
                    let v = (lit.abs() - 1) as usize;
                    if v != var {
                        self.var_scores[v] += if new_state == 0 { 1 } else { -1 };
                    }
                }
            }
        }
    }

    fn solve(&mut self) -> Option<Solution> {
        let start_time = Instant::now();
        let timeout = Duration::from_secs(5); // 5 second timeout

        for _ in 0..MAX_TRIES {
            for step in 1..=MAX_FLIPS {
                if self.unsat_clauses.is_empty() {
                    return Some(Solution { variables: self.assignment.clone() });
                }

                if start_time.elapsed() > timeout {
                    return None; // Timeout reached
                }

                let unsat_clause = self.unsat_clauses[self.rng.gen_range(0..self.unsat_clauses.len())];
                let clause = &self.clauses[unsat_clause];

                if self.rng.gen_bool(NOISE) {
                    // Random walk
                    let var = (clause[self.rng.gen_range(0..clause.len())].abs() - 1) as usize;
                    self.flip_var(var, step);
                } else {
                    // Greedy move
                    let mut best_var = None;
                    let mut best_score = i32::MIN;

                    for &lit in clause {
                        let var = (lit.abs() - 1) as usize;
                        let score = self.var_scores[var];
                        if score > best_score || (score == best_score && self.var_last_flip[var] < self.var_last_flip[best_var.unwrap_or(0)]) {
                            best_var = Some(var);
                            best_score = score;
                        }
                    }

                    if let Some(var) = best_var {
                        self.flip_var(var, step);
                    }
                }
            }

            // Randomize assignment for next try
            for i in 0..self.num_vars {
                if self.rng.gen_bool(0.5) {
                    self.flip_var(i, 0);
                }
            }
        }

        None
    }
}

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let mut solver = WalkSAT::new(challenge);
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
