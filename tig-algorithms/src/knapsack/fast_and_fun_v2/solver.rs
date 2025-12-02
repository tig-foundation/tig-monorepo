use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;
use super::params::Params;
use super::state::State;
use super::construct::build_initial_solution;
use super::local_search::local_search_vnd;
use super::dp::dp_refinement;

fn run_one_instance(challenge: &Challenge, params: &Params) -> Solution {
    let mut state = State::new_empty(challenge);

    // Build initial solution and compute (locked, core, rejected) sets
    build_initial_solution(&mut state, params.n_it_construct, params.core_half);

    // Local search
    local_search_vnd(&mut state, params);

    // ILS: DP + Local Search refinements
    for _it in 0..params.n_maxils {
        let prev_sel     = state.selected_items();
        let prev_val     = state.total_value;
        let prev_contrib = state.contrib.clone();
        let prev_weight  = state.total_weight;
        dp_refinement(&mut state);
        local_search_vnd(&mut state, params);
        if state.total_value <= prev_val {
            state.restore_snapshot(&prev_sel, prev_contrib, prev_val, prev_weight);
            break;
        }
    }

    let mut items = state.selected_items();
    items.sort_unstable();
    Solution { items }
}

pub struct Solver;

impl Solver {
    pub fn solve(
        challenge: &Challenge,
        _save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
        hyperparameters: &Option<Map<String, Value>>,
    ) -> Result<Option<Solution>> {
        let params = Params::initialize(hyperparameters);
        let solution = run_one_instance(challenge, &params);
        Ok(Some(solution))
    }
}
