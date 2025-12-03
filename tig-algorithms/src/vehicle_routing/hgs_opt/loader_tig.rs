use super::problem::Problem;
use tig_challenges::vehicle_routing::*;

pub struct TigLoader;

impl TigLoader {
    pub fn load(challenge: &Challenge) -> Problem {
        #[cfg(debug_assertions)]
        println!("----- LOADING TIG VRPTW CHALLENGE");
        let nb_nodes = challenge.num_nodes;
        let nb_vehicles = challenge.fleet_size;

        #[cfg(debug_assertions)]
        println!(
            "----- TIG INSTANCE LOADED WITH {} CLIENTS AND {} VEHICLES",
            nb_nodes, nb_vehicles
        );

        let mut service_times = vec![challenge.service_time; nb_nodes];
        service_times[0] = 0;

        // Demand-based lower bound on fleet size for TIG
        let total_demand: f64 = challenge.demands.iter().map(|&d| d as f64).sum();
        let ratio = total_demand / challenge.max_capacity as f64;
        let lb_vehicles = ratio.ceil() as usize;

        Problem {
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
