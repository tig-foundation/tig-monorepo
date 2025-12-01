use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::{Map, Value};
use std::time::{Duration, Instant};
use tig_challenges::satisfiability::*;

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
        let num_vars = challenge.num_variables;
        let clauses = challenge.clauses.clone();
        let mut rng = StdRng::seed_from_u64(u64::from_le_bytes(
            challenge.seed[..8].try_into().unwrap(),
        ) as u64);

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
                    return Some(Solution {
                        variables: self.assignment.clone(),
                    });
                }

                if start_time.elapsed() > timeout {
                    return None; // Timeout reached
                }

                let unsat_clause =
                    self.unsat_clauses[self.rng.gen_range(0..self.unsat_clauses.len())];
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
                        if score > best_score
                            || (score == best_score
                                && self.var_last_flip[var]
                                    < self.var_last_flip[best_var.unwrap_or(0)])
                        {
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

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let _ = save_solution(&Solution { variables: vec![false; challenge.num_variables] });
    let mut solver = WalkSAT::new(challenge);
    if let Some(s) = solver.solve() {
        let _ = save_solution(&s);
    }
    Ok(())
}

pub fn help() {
    println!("No help information available.");
}
