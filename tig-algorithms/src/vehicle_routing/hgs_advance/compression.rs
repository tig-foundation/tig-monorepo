use super::problem::{NodeData, Problem};
use super::sequence::Sequence;

pub struct ProblemCompression {
    pub compact: Problem,
    pub chains: Vec<Vec<usize>>, // compact client c maps to chains[c - 1]
}

impl ProblemCompression {
    pub fn from_chains(data: &Problem, chains: Vec<Vec<usize>>) -> Option<Self> {
        // Keep only compressible chains (TW-exact with a simple macro-node), split others.
        let mut normalized: Vec<Vec<usize>> = Vec::with_capacity(chains.len());
        for chain in chains {
            debug_assert!(!chain.is_empty(), "Consensus chain should never be empty");
            let seq = Self::sequence_on_chain(data, &chain);
            if seq.tw == 0 && seq.tau_minus <= seq.tau_plus {
                normalized.push(chain);
            } else {
                for id in chain {
                    normalized.push(vec![id]);
                }
            }
        }

        let compact_clients = normalized.len();
        debug_assert!(compact_clients <= data.nb_nodes - 1);
        if compact_clients == data.nb_nodes - 1 {
            return None;
        }

        let mut node_data: Vec<NodeData> = Vec::with_capacity(compact_clients + 1);
        let mut node_positions: Vec<(i32, i32)> = Vec::with_capacity(compact_clients + 1);
        let mut first_orig: Vec<usize> = vec![0usize; compact_clients + 1];
        let mut last_orig: Vec<usize> = vec![0usize; compact_clients + 1];
        let mut fixed_distance: i32 = 0;

        // Depot
        node_data.push(data.node_data[0]);
        node_positions.push(data.node_positions[0]);

        for chain in &normalized {
            let seq = Self::sequence_on_chain(data, chain);
            let demand_sum = seq.load;
            let mut sx: i64 = 0;
            let mut sy: i64 = 0;
            for &id in chain {
                sx += data.node_positions[id].0 as i64;
                sy += data.node_positions[id].1 as i64;
            }
            let len = chain.len() as i64;
            let bx = (sx / len) as i32;
            let by = (sy / len) as i32;

            fixed_distance += seq.distance;

            node_data.push(NodeData {
                start_tw: seq.tau_minus,
                end_tw: seq.tau_plus,
                service_time: seq.tmin,
                demand: demand_sum,
            });
            node_positions.push((bx, by));
            first_orig[node_data.len() - 1] = chain[0];
            last_orig[node_data.len() - 1] = chain[chain.len() - 1];
        }

        let nb_nodes = compact_clients + 1;
        let mut distance_matrix = vec![0i32; nb_nodes * nb_nodes];
        for i in 0..nb_nodes {
            let from = if i == 0 { 0 } else { last_orig[i] };
            for j in 0..nb_nodes {
                let to = if j == 0 { 0 } else { first_orig[j] };
                distance_matrix[i * nb_nodes + j] = data.dm(from, to);
            }
        }

        let total_demand = node_data.iter().skip(1).map(|nd| nd.demand as i64).sum::<i64>();
        let lb_vehicles = ((total_demand + data.max_capacity as i64 - 1) / data.max_capacity as i64) as usize;
        debug_assert!(lb_vehicles <= data.nb_vehicles, "lb_vehicles exceeds available vehicles");
        debug_assert!(lb_vehicles <= compact_clients, "lb_vehicles exceeds number of clients");

        let compact = Problem {
            seed: data.seed,
            nb_nodes,
            nb_vehicles: data.nb_vehicles,
            lb_vehicles,
            is_vrptw: data.is_vrptw,
            fixed_distance_offset: data.fixed_distance_offset + fixed_distance as i64,
            max_capacity: data.max_capacity,
            distance_matrix,
            node_positions,
            node_data,
        };

        Some(Self {
            compact,
            chains: normalized,
        })
    }

    #[inline]
    fn sequence_on_chain(data: &Problem, chain: &[usize]) -> Sequence {
        let mut seq = Sequence::singleton(data, chain[0]);
        for &id in chain.iter().skip(1) {
            let s = Sequence::singleton(data, id);
            seq = Sequence::join2(data, &seq, &s);
        }
        seq
    }
}
