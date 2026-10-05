use super::problem::{NodeData, Problem};
use super::sequence::Sequence;

pub struct ProblemCompression {
    pub compact: Problem,
    /// Compact client `c` expands to `chains[c - 1]` in the caller's index space.
    pub chains: Vec<Vec<usize>>,
}

impl ProblemCompression {
    pub fn from_chains(data: &Problem, chains: Vec<Vec<usize>>) -> Option<Self> {
        // A chain collapses into one macro-node only when its own traversal is time-window
        // exact; otherwise it is split back into singletons.
        let mut compact_chains: Vec<Vec<usize>> = Vec::with_capacity(chains.len());
        for chain in chains {
            debug_assert!(!chain.is_empty(), "Consensus chain should never be empty");
            let chain_seq = Self::sequence_on_chain(data, &chain);
            if chain_seq.tw == 0 && chain_seq.tau_minus <= chain_seq.tau_plus {
                compact_chains.push(chain);
            } else {
                for id in chain {
                    compact_chains.push(vec![id]);
                }
            }
        }

        let compact_clients = compact_chains.len();
        debug_assert!(compact_clients <= data.nb_nodes - 1);
        if compact_clients == data.nb_nodes - 1 {
            return None;
        }

        let mut node_data: Vec<NodeData> = Vec::with_capacity(compact_clients + 1);
        let mut node_positions: Vec<(i32, i32)> = Vec::with_capacity(compact_clients + 1);
        let mut first_orig: Vec<usize> = vec![0usize; compact_clients + 1];
        let mut last_orig: Vec<usize> = vec![0usize; compact_clients + 1];
        let mut fixed_distance: i32 = 0;

        node_data.push(data.node_data[0]);
        node_positions.push(data.node_positions[0]);

        for chain in &compact_chains {
            let chain_seq = Self::sequence_on_chain(data, chain);
            let demand_sum = chain_seq.load;
            let mut sum_x: i64 = 0;
            let mut sum_y: i64 = 0;
            for &id in chain {
                sum_x += data.node_positions[id].0 as i64;
                sum_y += data.node_positions[id].1 as i64;
            }
            let chain_len = chain.len() as i64;
            let barycenter_x = (sum_x / chain_len) as i32;
            let barycenter_y = (sum_y / chain_len) as i32;

            fixed_distance += chain_seq.distance;

            node_data.push(NodeData {
                start_tw: chain_seq.tau_minus,
                end_tw: chain_seq.tau_plus,
                service_time: chain_seq.tmin,
                demand: demand_sum,
            });
            node_positions.push((barycenter_x, barycenter_y));
            first_orig[node_data.len() - 1] = chain[0];
            last_orig[node_data.len() - 1] = chain[chain.len() - 1];
        }

        // The compact matrix leaves a macro-node by its chain's last original node and enters it
        // by the first: it is asymmetric even when the caller's matrix is symmetric, and its
        // diagonal is the last-to-first distance inside a chain, not zero.
        let nb_nodes = compact_clients + 1;
        let mut distance_matrix = vec![0i32; nb_nodes * nb_nodes];
        for i in 0..nb_nodes {
            let from_node = if i == 0 { 0 } else { last_orig[i] };
            for j in 0..nb_nodes {
                let to_node = if j == 0 { 0 } else { first_orig[j] };
                distance_matrix[i * nb_nodes + j] = data.dm(from_node, to_node);
            }
        }

        let total_demand = node_data
            .iter()
            .skip(1)
            .map(|nd| nd.demand as i64)
            .sum::<i64>();
        let lb_vehicles =
            ((total_demand + data.max_capacity as i64 - 1) / data.max_capacity as i64) as usize;
        debug_assert!(
            lb_vehicles <= data.nb_vehicles,
            "lb_vehicles exceeds available vehicles"
        );
        debug_assert!(
            lb_vehicles <= compact_clients,
            "lb_vehicles exceeds number of clients"
        );

        let distance_matrix_transposed = Problem::transpose_distances(&distance_matrix, nb_nodes);
        let compact = Problem {
            seed: data.seed,
            nb_nodes,
            nb_vehicles: data.nb_vehicles,
            lb_vehicles,
            is_vrptw: data.is_vrptw,
            fixed_distance_offset: data.fixed_distance_offset + fixed_distance as i64,
            max_capacity: data.max_capacity,
            distance_matrix,
            distance_matrix_transposed,
            node_positions,
            node_data,
        };

        Some(Self {
            compact,
            chains: compact_chains,
        })
    }

    #[inline]
    fn sequence_on_chain(data: &Problem, chain: &[usize]) -> Sequence {
        let mut seq = Sequence::singleton(data, chain[0]);
        for &id in chain.iter().skip(1) {
            let node = Sequence::singleton(data, id);
            seq = Sequence::join2(data, &seq, &node);
        }
        seq
    }
}
