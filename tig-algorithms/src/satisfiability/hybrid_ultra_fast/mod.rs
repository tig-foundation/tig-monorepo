use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::{Map, Value};
use std::ptr;
use std::time::{Duration, Instant};
use tig_challenges::satisfiability::*;

const MAX_CONFLICTS: usize = 1_000_000;
const VSIDS_DECAY: f64 = 0.95;
const VSIDS_INCREMENT: f64 = 1.0;

#[derive(Clone, Copy)]
struct Clause {
    literals: *mut i32,
    len: u32,
}

struct Solver {
    num_vars: usize,
    clauses: Vec<Clause>,
    assignments: Vec<bool>,
    decision_levels: Vec<i32>,
    vsids_scores: Vec<f64>,
    watchers: Vec<Vec<usize>>,
    trail: Vec<i32>,
    propagation_queue: Vec<i32>,
    rng: StdRng,
    restarts: usize,
    conflicts: usize,
}

impl Solver {
    fn new(challenge: &Challenge) -> Self {
        let num_vars = challenge.num_variables;
        let mut clauses = Vec::with_capacity(challenge.clauses.len());
        let mut watchers = vec![Vec::new(); 2 * num_vars];

        for clause in &challenge.clauses {
            let mut clause_data = vec![0; clause.len()].into_boxed_slice();
            for (i, &lit) in clause.iter().enumerate() {
                clause_data[i] = lit;
            }
            let clause_ptr = Box::into_raw(clause_data) as *mut i32;
            clauses.push(Clause {
                literals: clause_ptr,
                len: clause.len() as u32,
            });

            if clause.len() >= 2 {
                watchers[Self::lit_to_idx(unsafe { *clause_ptr })].push(clauses.len() - 1);
                watchers[Self::lit_to_idx(unsafe { *clause_ptr.add(1) })].push(clauses.len() - 1);
            }
        }

        Solver {
            num_vars,
            clauses,
            assignments: vec![false; num_vars],
            decision_levels: vec![-1; num_vars],
            vsids_scores: vec![0.0; num_vars],
            watchers,
            trail: Vec::with_capacity(num_vars),
            propagation_queue: Vec::with_capacity(num_vars),
            rng: StdRng::seed_from_u64(
                u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64
            ),
            restarts: 0,
            conflicts: 0,
        }
    }

    #[inline(always)]
    fn lit_to_idx(lit: i32) -> usize {
        (lit.abs() as usize - 1) * 2 + if lit > 0 { 0 } else { 1 }
    }

    #[inline(always)]
    fn assign(&mut self, lit: i32, level: i32) {
        let var = lit.abs() as usize - 1;
        self.assignments[var] = lit > 0;
        self.decision_levels[var] = level;
        self.trail.push(lit);
        self.propagation_queue.push(lit);
    }

    unsafe fn propagate(&mut self) -> bool {
        while let Some(lit) = self.propagation_queue.pop() {
            let neg_lit_idx = Self::lit_to_idx(-lit);
            let mut i = 0;
            while i < self.watchers[neg_lit_idx].len() {
                let clause_idx = self.watchers[neg_lit_idx][i];
                let clause = self.clauses[clause_idx];

                if *clause.literals == -lit {
                    ptr::swap(clause.literals, clause.literals.add(1));
                }

                if self.evaluate_lit(*clause.literals) {
                    i += 1;
                    continue;
                }

                let mut found = false;
                for j in 2..clause.len as usize {
                    if !self.evaluate_lit(-*clause.literals.add(j)) {
                        ptr::swap(clause.literals.add(1), clause.literals.add(j));
                        self.watchers[Self::lit_to_idx(*clause.literals.add(1))].push(clause_idx);
                        self.watchers[neg_lit_idx].swap_remove(i);
                        found = true;
                        break;
                    }
                }

                if !found {
                    if self.evaluate_lit(-*clause.literals.add(1)) {
                        return false;
                    }
                    self.assign(
                        *clause.literals.add(1),
                        self.decision_levels[lit.abs() as usize - 1],
                    );
                    i += 1;
                }
            }
        }
        true
    }

    #[inline(always)]
    fn evaluate_lit(&self, lit: i32) -> bool {
        let var = lit.abs() as usize - 1;
        self.decision_levels[var] >= 0 && self.assignments[var] == (lit > 0)
    }

    fn pick_branching_variable(&mut self) -> i32 {
        let var = (0..self.num_vars)
            .filter(|&var| self.decision_levels[var] < 0)
            .max_by(|&a, &b| {
                self.vsids_scores[a]
                    .partial_cmp(&self.vsids_scores[b])
                    .unwrap()
            })
            .unwrap();
        (var as i32 + 1) * if self.rng.gen() { 1 } else { -1 }
    }

    fn solve(&mut self) -> Option<Solution> {
        let start_time = Instant::now();
        let timeout = Duration::from_secs(5);

        loop {
            if start_time.elapsed() > timeout {
                return None;
            }

            if unsafe { !self.propagate() } {
                self.conflicts += 1;
                if self.trail.is_empty() {
                    return None;
                }
                self.backtrack();
            } else if self.trail.len() == self.num_vars {
                return Some(Solution {
                    variables: self.assignments.clone(),
                });
            } else {
                if self.conflicts >= MAX_CONFLICTS {
                    self.restart();
                    continue;
                }
                let lit = self.pick_branching_variable();
                let level = self.trail.len() as i32;
                self.assign(lit, level);
            }

            if self.conflicts % 1000 == 0 {
                self.decay_vsids();
            }
        }
    }

    fn backtrack(&mut self) {
        let mut conflict_level = 0;
        for &lit in self.trail.iter().rev() {
            let var = lit.abs() as usize - 1;
            self.vsids_scores[var] += VSIDS_INCREMENT;
            if self.decision_levels[var] > conflict_level {
                conflict_level = self.decision_levels[var];
                break;
            }
        }

        while let Some(lit) = self.trail.pop() {
            let var = lit.abs() as usize - 1;
            if self.decision_levels[var] <= conflict_level {
                self.trail.push(lit);
                break;
            }
            self.assignments[var] = false;
            self.decision_levels[var] = -1;
        }
    }

    fn restart(&mut self) {
        self.trail.clear();
        self.propagation_queue.clear();
        for level in self.decision_levels.iter_mut() {
            *level = -1;
        }
        for assignment in self.assignments.iter_mut() {
            *assignment = false;
        }
        self.conflicts = 0;
        self.restarts += 1;
    }

    fn decay_vsids(&mut self) {
        for score in self.vsids_scores.iter_mut() {
            *score *= VSIDS_DECAY;
        }
    }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let _ = save_solution(&Solution { variables: vec![false; challenge.num_variables] });
    let mut solver = Solver::new(challenge);
    if let Some(s) = solver.solve() {
        let _ = save_solution(&s);
    }
    Ok(())
}

pub fn help() {
    println!("No help information available.");
}
