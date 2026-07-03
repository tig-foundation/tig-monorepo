use super::problem::{NodeData, Problem};
use tig_challenges::vehicle_routing::*;

pub struct TigLoader;

impl TigLoader {
    pub fn load(challenge: &Challenge) -> Problem {
        let nb_nodes = challenge.num_nodes;
        let nb_vehicles = challenge.fleet_size;

        let mut node_data: Vec<NodeData> = (0..nb_nodes)
            .map(|i| NodeData {
                start_tw: challenge.ready_times[i],
                end_tw: challenge.due_times[i],
                service_time: if i == 0 { 0 } else { challenge.service_time },
                demand: challenge.demands[i],
            })
            .collect();

        // One-time TW tightening with depot reachability/return bounds:
        // start_i >= start_depot + d(0,i)
        // end_i   <= end_depot   - service_i - d(i,0)
        let depot_start = node_data[0].start_tw;
        let depot_end = node_data[0].end_tw;
        for i in 1..nb_nodes {
            let d0i = challenge.distance_matrix[0][i];
            let di0 = challenge.distance_matrix[i][0];
            let nd = &mut node_data[i];
            nd.start_tw = nd.start_tw.max(depot_start + d0i);
            nd.end_tw = nd.end_tw.min(depot_end - nd.service_time - di0);
            debug_assert!(
                nd.start_tw <= nd.end_tw,
                "Node {} has empty tightened TW: [{}, {}]",
                i,
                nd.start_tw,
                nd.end_tw
            );
        }

        // Demand-based lower bound on fleet size for TIG
        let total_demand: i64 = challenge.demands.iter().map(|&d| d as i64).sum();
        let lb_vehicles = ((total_demand + challenge.max_capacity as i64 - 1) / challenge.max_capacity as i64) as usize;
        Problem {
            seed: challenge.seed,
            nb_nodes,
            nb_vehicles,
            lb_vehicles,
            is_vrptw: true,
            fixed_distance_offset: 0,
            node_positions: challenge.node_positions.clone(),
            max_capacity: challenge.max_capacity,
            distance_matrix: challenge.distance_matrix.iter().flatten().copied().collect(),
            node_data,
        }
    }
}
