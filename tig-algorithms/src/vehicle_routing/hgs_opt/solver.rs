use super::genetic::Genetic;
use super::loader_tig::TigLoader;
use super::params::Params;
use super::problem::Problem;
use anyhow::Result;
use rand::{rngs::SmallRng, SeedableRng};
use serde_json::{Map, Value};
use std::time::Instant;
use tig_challenges::vehicle_routing::*;

pub struct Solver;

impl Solver {
    fn solve(
        data: Problem,
        params: Params,
        t0: &Instant,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    ) -> Result<Option<(Solution, i32, usize)>> {
        // Initialize parameters, RNG and run GA
        let mut rng = SmallRng::from_seed(data.seed);
        let mut ga = Genetic::new(&data, params);
        Ok(ga.run(&mut rng, t0, save_solution).map(|(routes, cost)| {
            let len = routes.len();
            (Solution { routes }, cost, len)
        }))
    }

    pub fn solve_challenge_instance(
        challenge: &Challenge,
        hyperparameters: &Option<Map<String, Value>>,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    ) -> Result<Option<Solution>> {
        let t0 = Instant::now();
        let data = TigLoader::load(&challenge);
        let params = Params::initialize(hyperparameters,data.nb_nodes);
        match Self::solve(data, params, &t0, save_solution) {
            Ok(Some((solution, _cost, _routes))) => Ok(Some(solution)),
            Ok(None) => Ok(None),
            Err(e) => {
                eprintln!("Error: {}", e);
                Ok(None)
            }
        }
    }
}
