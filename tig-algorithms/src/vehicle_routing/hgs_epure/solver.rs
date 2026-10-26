use super::genetic::Genetic;
use super::loader_tig::TigLoader;
use super::params::Params;
use super::problem::Problem;
use super::reverse_mode;
use anyhow::Result;
use rand::{rngs::SmallRng, SeedableRng};
use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;

pub struct Solver;

impl Solver {
    fn solve(
        data: Problem,
        params: Params,
        hyperparameters: &Option<Map<String, Value>>,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    ) -> Result<Option<Solution>> {
        if params.decomp_nb_phases > 0 {
            return reverse_mode::solve_reversed_mode(data, params, hyperparameters, save_solution);
        }

        let mut rng = SmallRng::from_seed(data.seed);
        let mut ga = Genetic::new(data, params);
        Ok(ga
            .run(&mut rng, save_solution, None)
            .map(|individual| Solution { routes: individual.routes }))
    }

    pub fn solve_challenge_instance(
        challenge: &Challenge,
        hyperparameters: &Option<Map<String, Value>>,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    ) -> Result<Option<Solution>> {
        let data = TigLoader::load(challenge);
        let params = Params::initialize(hyperparameters, &data);

        match Self::solve(data, params, hyperparameters, save_solution) {
            Ok(Some(solution)) => Ok(Some(solution)),
            Ok(None) => Ok(None),
            Err(_) => Ok(None),
        }
    }
}
