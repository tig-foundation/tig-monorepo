use super::instance::Instance;
use super::config::Config;
use super::evolution::Evolution;
use anyhow::Result;
use tig_challenges::vehicle_routing::*;
use serde_json::{Map, Value};
use rand::{rngs::SmallRng, SeedableRng};
use std::time::Instant;

pub struct TigLoader;

impl TigLoader {
    pub fn load(challenge: &Challenge) -> Instance {
        let nb_nodes = challenge.num_nodes;
        let nb_vehicles = challenge.fleet_size;

        let mut service_times = vec![challenge.service_time; nb_nodes];
        service_times[0] = 0;

        let total_demand: f64 = challenge.demands.iter().map(|&d| d as f64).sum();
        let ratio = total_demand / challenge.max_capacity as f64;
        let lb_vehicles = ratio.ceil() as usize;

        Instance {
            seed: challenge.seed,
            nb_nodes,
            nb_vehicles,
            lb_vehicles,
            demands: challenge.demands.clone(),
            node_positions: challenge.node_positions.clone(),
            max_capacity: challenge.max_capacity,
            distance_matrix: challenge.distance_matrix.clone(),
            service_times,
            start_tw: challenge.ready_times.clone(),
            end_tw: challenge.due_times.clone(),
        }
    }
}

pub struct Solver;

impl Solver {
    fn solve(
        data: Instance,
        params: Config,
        t0: &Instant,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    ) -> Result<Option<(Solution, i32, usize)>> {
        let mut rng = SmallRng::from_seed(data.seed);
        let mut ga = Evolution::new(&data, params);
        Ok(ga.run(&mut rng, t0, save_solution).map(|(routes, cost)| {
            (Solution { routes: routes.clone() }, cost, routes.len())
        }))
    }

    pub fn solve_challenge_instance(
        challenge: &Challenge,
        hyperparameters: &Option<Map<String, Value>>,
        save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
    ) -> Result<Option<Solution>> {
        let t0 = Instant::now();
        let data = TigLoader::load(challenge);
        let params = Config::initialize(hyperparameters, data.nb_nodes);
        match Self::solve(data, params, &t0, save_solution) {
            Ok(Some((solution, _cost, _routes))) => Ok(Some(solution)),
            Ok(None) => Ok(None),
            Err(_) => Ok(None),
        }
    }
}
