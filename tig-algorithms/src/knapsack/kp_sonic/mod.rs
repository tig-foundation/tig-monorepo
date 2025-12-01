use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;


pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    Err(anyhow!("This algorithm is no longer compatible."))
}

// Old code that is no longer compatible
#[cfg(none)]
mod dead_code {
    use anyhow::{anyhow, Result};
    use tig_challenges::knapsack::*;

    const MAX_ITEMS: usize = 1000; // Adjust based on expected maximum number of items
    const SCALE_FACTOR: i32 = 1000; // Scaling factor to preserve precision


    pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
        let mut solution = Solution {
            sub_solutions: Vec::new(),
        };
        for sub_instance in &challenge.sub_instances {
            match solve_sub_instance(sub_instance)? {
                Some(sub_solution) => solution.sub_solutions.push(sub_solution),
                None => return Ok(None),
            }
        }
        Ok(Some(solution))
    }

    pub fn solve_sub_instance(challenge: &SubInstance) -> Result<Option<SubSolution>> {
        let vertex_count = challenge.weights.len();

        // Ensure that vertex_count does not exceed MAX_ITEMS
        if vertex_count > MAX_ITEMS {
            return Err(anyhow!(
                "vertex_count ({}) exceeds MAX_ITEMS ({})",
                vertex_count,
                MAX_ITEMS
            ));
        }

        // Precompute total flows and cost metrics using integer arithmetic
        let mut edge_costs: [(usize, i32); MAX_ITEMS] = [(0, 0); MAX_ITEMS];
        let weights = &challenge.weights;
        let values = &challenge.values;
        let interaction_values = &challenge.interaction_values;

        for flow_index in 0..vertex_count {
            let total_flow: i32 = values[flow_index] as i32
                + interaction_values[flow_index].iter().sum::<i32>();
            let cost_scaled = if weights[flow_index] != 0 {
                total_flow * SCALE_FACTOR / weights[flow_index] as i32
            } else {
                0
            };
            edge_costs[flow_index] = (flow_index, cost_scaled);
        }

        // Sort edge_costs in descending order based on cost_scaled
        let mut edge_costs_slice = &mut edge_costs[..vertex_count];
        edge_costs_slice.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        // Initialize fixed-size arrays for coloring and uncolored
        let mut coloring: [usize; MAX_ITEMS] = [0; MAX_ITEMS];
        let mut coloring_len = 0;
        let mut uncolored: [usize; MAX_ITEMS] = [0; MAX_ITEMS];
        let mut uncolored_len = 0;
        let mut current_entropy: i32 = 0;
        let mut current_temperature: i32 = 0;

        // Initial Selection Phase
        for &(flow_index, _) in edge_costs_slice.iter() {
            let weight = weights[flow_index] as i32;
            if current_entropy + weight <= challenge.max_weight as i32 {
                current_entropy += weight;
                current_temperature += values[flow_index] as i32;

                let interactions = &interaction_values[flow_index];
                for &colored in &coloring[..coloring_len] {
                    current_temperature += interactions[colored];
                }
                coloring[coloring_len] = flow_index;
                coloring_len += 1;
            } else {
                uncolored[uncolored_len] = flow_index;
                uncolored_len += 1;
            }
        }

        // Precompute Mutation Rates using fixed-size arrays
        let mut mutation_rates: [i32; MAX_ITEMS] = [0; MAX_ITEMS];
        for flow_index in 0..vertex_count {
            let value = values[flow_index] as i32;
            let interactions = &interaction_values[flow_index];
            let mut rate = value;
            for &colored in &coloring[..coloring_len] {
                rate += interactions[colored];
            }
            mutation_rates[flow_index] = rate;
        }

        // Initialize fixed-size cooling_schedule
        let mut cooling_schedule: [u8; MAX_ITEMS] = [0; MAX_ITEMS];

        // Mutation Phase Parameters
        const MAX_GENERATIONS: usize = 100;

        for _ in 0..MAX_GENERATIONS {
            let mut best_mutation: i32 = 0;
            let mut best_crossover: Option<(usize, usize)> = None;

            // Iterate over uncolored items
            for uncolored_index in 0..uncolored_len {
                let mutant = uncolored[uncolored_index];
                if cooling_schedule[mutant] > 0 {
                    continue;
                }

                let mutant_fitness = mutation_rates[mutant];
                if mutant_fitness < 0 {
                    continue;
                }

                let min_entropy_reduction = weights[mutant] as i32
                    - (challenge.max_weight as i32 - current_entropy as i32);

                // Iterate over colored items
                for colored_index in 0..coloring_len {
                    let gene_to_remove = coloring[colored_index];
                    if cooling_schedule[gene_to_remove] > 0 {
                        continue;
                    }

                    if min_entropy_reduction > 0 {
                        let removed_entropy = weights[gene_to_remove] as i32;
                        if removed_entropy < min_entropy_reduction {
                            continue;
                        }
                    }

                    let fitness_change = mutant_fitness
                        - mutation_rates[gene_to_remove]
                        - interaction_values[mutant][gene_to_remove];

                    if fitness_change > best_mutation {
                        best_mutation = fitness_change;
                        best_crossover = Some((uncolored_index, colored_index));
                    }
                }
            }

            if let Some((uncolored_index, colored_index)) = best_crossover {
                let gene_to_add = uncolored[uncolored_index];
                let gene_to_remove = coloring[colored_index];

                // Swap mutation
                coloring[colored_index] = gene_to_add;
                // Swap-remove for uncolored
                uncolored[uncolored_index] = uncolored[uncolored_len - 1];
                uncolored_len -= 1;
                uncolored[uncolored_len] = gene_to_remove;
                coloring_len -= 1;
                coloring[coloring_len] = gene_to_remove;
                coloring_len += 1;

                // Update current_temperature and current_entropy
                current_temperature += best_mutation;
                current_entropy = current_entropy
                    + weights[gene_to_add] as i32
                    - weights[gene_to_remove] as i32;

                // Update mutation_rates
                for flow_index in 0..vertex_count {
                    mutation_rates[flow_index] +=
                        interaction_values[flow_index][gene_to_add] as i32
                        - interaction_values[flow_index][gene_to_remove] as i32;
                }

                // Update cooling_schedule
                cooling_schedule[gene_to_add] = 3;
                cooling_schedule[gene_to_remove] = 3;
            } else {
                break;
            }

            // Early exit if solution meets the minimum value
            if current_temperature as u32 >= challenge.baseline_value {
                let solution_items = &coloring[..coloring_len];
                return Ok(Some(SubSolution { items: solution_items.to_vec() }));
            }

            // Decrement cooling_schedule
            for schedule in cooling_schedule.iter_mut() {
                if *schedule > 0 {
                    *schedule -= 1;
                }
            }
        }

        // Final Check
        if current_temperature as u32 >= challenge.baseline_value {
            let solution_items = &coloring[..coloring_len];
            Ok(Some(SubSolution { items: solution_items.to_vec() }))
        } else {
            Ok(None)
        }
    }
}

pub fn help() {
    println!("No help information available.");
}
