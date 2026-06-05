use super::instance::{Instance, NodeData};
use super::config::Config;
use super::evolution::Evolution;
use anyhow::Result;
use super::*;
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

        let node_data: Vec<NodeData> = (0..nb_nodes).map(|i| NodeData {
            start_tw: challenge.ready_times[i],
            end_tw: challenge.due_times[i],
            service_time: service_times[i],
            demand: challenge.demands[i],
        }).collect();

        Instance {
            seed: challenge.seed,
            nb_nodes,
            nb_vehicles,
            lb_vehicles,
            demands: challenge.demands.clone(),
            node_positions: challenge.node_positions.clone(),
            max_capacity: challenge.max_capacity,
            distance_matrix: challenge.distance_matrix.iter().flatten().copied().collect(),
            service_times,
            start_tw: challenge.ready_times.clone(),
            end_tw: challenge.due_times.clone(),
            node_data,
        }
    }
}

pub struct ScenarioProfiler;

impl ScenarioProfiler {
    pub fn profile(data: &Instance) -> (f64, f64) {
        // Calculate time window tightness
        let mut total_tw_width = 0.0;
        let mut total_tw_range = 0.0;
        for i in 1..data.nb_nodes {
            let width = (data.end_tw[i] - data.start_tw[i]) as f64;
            let range = (data.end_tw[i] - data.start_tw[0]) as f64;
            total_tw_width += width;
            total_tw_range += range;
        }
        let avg_tw_width = total_tw_width / (data.nb_nodes - 1) as f64;
        let avg_tw_range = total_tw_range / (data.nb_nodes - 1) as f64;
        let time_window_tightness = 1.0 - (avg_tw_width / avg_tw_range).min(1.0);

        // Calculate demand variance
        let mut mean_demand = 0.0;
        let mut variance = 0.0;
        let mut count = 0;
        for i in 1..data.nb_nodes {
            let demand = data.demands[i] as f64;
            mean_demand += demand;
            count += 1;
        }
        mean_demand /= count as f64;

        for i in 1..data.nb_nodes {
            let demand = data.demands[i] as f64;
            variance += (demand - mean_demand).powi(2);
        }
        let demand_variance = if count > 1 {
            (variance / (count - 1) as f64).sqrt() / mean_demand
        } else {
            0.0
        };

        (time_window_tightness, demand_variance)
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

        // Profile the scenario to determine algorithm complexity
        let (time_window_tightness, demand_variance) = ScenarioProfiler::profile(&data);

        // Select algorithm based on scenario complexity
        let params = if hyperparameters.is_none() {
            Config::select_algorithm(data.nb_nodes, data.nb_vehicles, time_window_tightness, demand_variance)
        } else {
            Config::initialize(hyperparameters, data.nb_nodes, data.nb_vehicles)
        };

        match Self::solve(data, params, &t0, save_solution) {
            Ok(Some((solution, _cost, _routes))) => Ok(Some(solution)),
            Ok(None) => Ok(None),
            Err(_) => Ok(None),
        }
    }
}