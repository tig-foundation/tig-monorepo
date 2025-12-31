use serde_json::{Map, Value};
use crate::utilities::vehicle_routing::{Challenge, Solution};

pub struct Solver;

impl Solver {
    pub fn solve_challenge_instance(
        challenge: &Challenge,
        hyperparameters: &Option<Map<String, Value>>,
        save_solution: Option<&dyn Fn(&Solution)>,
    ) -> Option<Solution> {
        // Adapter: convert external Challenge -> internal Problem via serde_json
        // Then run existing solver pipeline and convert internal solution -> external Solution
        // This wrapper is intentionally minimal and does not print, panic, or manage bundles.

        // Serialize challenge to value
        let val = match serde_json::to_value(challenge) {
            Ok(v) => v,
            Err(_) => return None,
        };

        // Try to convert into internal Problem structure
        let internal_problem: crate::problem_loader::Problem = match serde_json::from_value(val) {
            Ok(p) => p,
            Err(_) => return None,
        };

        // Build initial state
        let state = internal_problem.to_state();

        // Prepare solver seed and configuration
        let seed_bytes: [u8; 32] = [0u8; 32];
        let config = internal_problem.config.unwrap_or_default();

        // Construct underlying solver
        let mut solver = crate::solver::Solver::with_config(seed_bytes, config);

        // Use a conservative iteration cap; if hyperparameters specify iterations, try to respect it
        let max_iters = hyperparameters
            .as_ref()
            .and_then(|m| m.get("max_iterations"))
            .and_then(|v| v.as_u64())
            .map(|x| x as usize)
            .unwrap_or(100usize);

        // Attempt to solve; on any error return None
        match solver.solve_state(state, max_iters) {
            Ok(result_state) => {
                // Convert internal state to internal solution representation
                let internal_sol = crate::problem_loader::Solution {
                    routes: vec![result_state.route.nodes.to_vec()],
                    total_cost: result_state.total_cost(),
                    feasible: result_state.is_feasible(),
                    arrival_times: if result_state.arrival_times.is_empty() {
                        None
                    } else {
                        Some(result_state.arrival_times.clone())
                    },
                };

                // Convert internal solution to external tig_challenges Solution via serde
                let v = match serde_json::to_value(&internal_sol) {
                    Ok(v) => v,
                    Err(_) => return None,
                };

                let external_sol: Solution = match serde_json::from_value(v) {
                    Ok(s) => s,
                    Err(_) => return None,
                };

                // Optionally call the provided save_solution callback
                if let Some(cb) = save_solution {
                    cb(&external_sol);
                }

                Some(external_sol)
            }
            Err(_) => None,
        }
    }
}
