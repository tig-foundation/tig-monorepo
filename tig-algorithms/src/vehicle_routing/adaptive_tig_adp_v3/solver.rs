// Phase 2: Enhanced solver with full configuration system
use crate::adp::{rollout::RolloutPolicy, vfa::VFA};
use crate::config::SolverConfig;
use crate::constructive::Constructive;
use crate::utilities::IZS;
use crate::local_search::LocalSearch;
use crate::tig_adaptive::TIGState;
use crate::population::Population;
use anyhow::Result;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::time::Instant;

pub struct Solver {
    vfa: VFA,
    rollout: RolloutPolicy,
    izs: IZS,
    local_search: LocalSearch,
    _constructive: Constructive,
    rng: SmallRng,
    config: SolverConfig,
}

impl Solver {
    /// Phase 2: Constructor with default configuration
    pub fn new(seed: [u8; 32]) -> Self {
        Self::with_config(seed, SolverConfig::default())
    }

    /// Phase 2: Constructor with custom configuration
    pub fn with_config(seed: [u8; 32], config: SolverConfig) -> Self {
        let vfa = VFA::new(config.learning_rate, config.discount_factor, config.flexibility_weight)
            .with_penalty_weight(config.penalty_weight)
            .with_decay(config.learning_decay);

        let rollout = RolloutPolicy::new(config.rollout_depth)
            .with_fallback(config.rollout_fallback);

        let izs = IZS::new(config.izs_threshold);

        let local_search = LocalSearch::new(config.local_search_time_ms, config.neighborhood_size);

        let _constructive = Constructive::new(config.beam_width);

        Self {
            vfa,
            rollout,
            izs,
            local_search,
            _constructive,
            rng: SmallRng::from_seed(seed),
            config,
        }
    }

    /// Phase 2: Load solver from JSON problem file
    pub fn from_json(json_path: &str, seed: [u8; 32]) -> Result<(Self, TIGState)> {
        let problem = crate::problem_loader::load_problem(json_path)?;
        let state = problem.to_state();
        
        // Use problem config if available, otherwise default
        let config = problem.config.unwrap_or_default();
        let solver = Self::with_config(seed, config);
        
        Ok((solver, state))
    }

    pub fn solve(
        &mut self,
        initial_route: Vec<usize>,
        time: i32,
        max_capacity: i32,
        tw_start: Vec<i32>,
        tw_end: Vec<i32>,
        service_time: Vec<i32>,
        distance_matrix: Vec<Vec<i32>>,
        demands: Vec<i32>,
        _t0: &Instant,
        _timeout_ms: u128,
    ) -> Result<Vec<usize>> {
        let mut state = TIGState::new(
            initial_route,
            time,
            max_capacity,
            tw_start,
            tw_end,
            service_time,
            distance_matrix,
            demands,
        );

        // Apply local search
        self.local_search.optimize_with_config(&mut state, &self.izs);

        Ok(state.route.nodes.to_vec())
    }

    /// Phase 2: Solve from TIGState with iteration limit
    pub fn solve_state(&mut self, mut state: TIGState, max_iterations: usize) -> Result<TIGState> {
        // initialize micro-population for evolutionary improvement
        let pop_size = 8usize;
        let mut pop = Population::new(&mut self.rng, &self.vfa, &self.local_search, &self.izs);
        pop.initialize_from(state.clone(), pop_size);

        for i in 0..max_iterations {
            if i % 10 == 0 && self.config.verbose {
                println!("Iteration {}/{}: feasible={}, ftb={}",
                     i, max_iterations, state.is_feasible(), state.free_time_budget());
            }

            // Apply local search (use multi-route optimizer with a single route)
            let mut states = vec![state.clone()];
            self.local_search.optimize_multi_with_config(&mut states, &self.izs);
            state = states.remove(0);

            // Evolve population a few generations and pick best
            for _gen in 0..3 {
                pop.step_evolve();
            }
            if let Some(best) = pop.best() {
                state = best.state.clone();
            }

            // Check if feasible and good enough
            if state.is_feasible() && state.free_time_budget() > 1000 {
                break;
            }
        }

        if self.config.verbose {
            println!("Final state: feasible={}, violations=(time:{}, cap:{})",
                state.is_feasible(),
                state.time_violation_penalty(),
                state.capacity_violation_penalty()
            );
        }

        Ok(state)
    }

    pub fn insert_dynamic(
        &mut self,
        mut state: TIGState,
        node: usize,
        horizon: usize,
    ) -> (bool, TIGState) {
        let value_before = self.vfa.estimate(&state, &mut self.rng);

        if Constructive::insert_with_value(&mut state, node, &self.vfa, &mut self.rng) {
            self.local_search.optimize_with_config(&mut state, &self.izs);

            let rollout_target =
                self.rollout
                    .rollout(state.clone(), &mut self.rng, horizon, &self.vfa);
            let update_target = rollout_target + value_before;
            self.vfa.update_dlt(&state, update_target, node, node);

            (true, state)
        } else {
            (false, state)
        }
    }

    /// Phase 2: Get solver statistics
    pub fn stats(&self) -> String {
        format!(
            "Solver Statistics:\n{}",
            self.vfa.stats()
        )
    }

    /// Phase 2: Access configuration
    pub fn config(&self) -> &SolverConfig {
        &self.config
    }
}
