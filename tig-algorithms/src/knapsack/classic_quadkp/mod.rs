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
    use anyhow::Result;
    use tig_challenges::knapsack::*;


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

        let mut edge_costs: Vec<(usize, f32)> = (0..vertex_count)
            .map(|flow_index| {
                let total_flow = challenge.values[flow_index] as i32 + 
                    challenge.interaction_values[flow_index].iter().sum::<i32>();
                let cost = total_flow as f32 / challenge.weights[flow_index] as f32;
                (flow_index, cost)
            })
            .collect();

        edge_costs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut coloring = Vec::with_capacity(vertex_count);
        let mut uncolored = Vec::with_capacity(vertex_count);
        let mut current_entropy = 0;
        let mut current_temperature = 0;

        for &(flow_index, _) in &edge_costs {
            if current_entropy + challenge.weights[flow_index] <= challenge.max_weight {
                current_entropy += challenge.weights[flow_index];
                current_temperature += challenge.values[flow_index] as i32;
    
                for &colored in &coloring {
                    current_temperature += challenge.interaction_values[flow_index][colored];
                }
                coloring.push(flow_index);
            } else {
                uncolored.push(flow_index);
            }
        }

        let mut mutation_rates = vec![0; vertex_count];
        for flow_index in 0..vertex_count {
            mutation_rates[flow_index] = challenge.values[flow_index] as i32;
            for &colored in &coloring {
                mutation_rates[flow_index] += challenge.interaction_values[flow_index][colored];
            }
        }

        let max_generations = 100;
        let mut cooling_schedule = vec![0; vertex_count];
    
        for _ in 0..max_generations {
            let mut best_mutation = 0;
            let mut best_crossover = None;

            for uncolored_index in 0..uncolored.len() {
                let mutant = uncolored[uncolored_index];
                if cooling_schedule[mutant] > 0 {
                    continue;
                }
        
                unsafe {
                    let mutant_fitness = *mutation_rates.get_unchecked(mutant);
                    let min_entropy_reduction = *challenge.weights.get_unchecked(mutant) as i32 - (challenge.max_weight as i32 - current_entropy as i32);

                    if mutant_fitness < 0 {
                        continue;
                    }
                
                    for colored_index in 0..coloring.len() {
                        let gene_to_remove = *coloring.get_unchecked(colored_index);
                        if *cooling_schedule.get_unchecked(gene_to_remove) > 0 {
                            continue;
                        }

                        if min_entropy_reduction > 0 {
                            let removed_entropy = *challenge.weights.get_unchecked(gene_to_remove) as i32;
                            if removed_entropy < min_entropy_reduction {
                                continue;
                            }
                        }

                        let fitness_change = mutant_fitness - *mutation_rates.get_unchecked(gene_to_remove)
                        - *challenge.interaction_values.get_unchecked(mutant).get_unchecked(gene_to_remove);
            
                        if fitness_change > best_mutation {
                            best_mutation = fitness_change;
                            best_crossover = Some((uncolored_index, colored_index));
                        }
                    }
                }
            }

            if let Some((uncolored_index, colored_index)) = best_crossover {
                let gene_to_add = uncolored[uncolored_index];
                let gene_to_remove = coloring[colored_index];
            
                coloring.swap_remove(colored_index);
                uncolored.swap_remove(uncolored_index);
                coloring.push(gene_to_add);
                uncolored.push(gene_to_remove);
            
                current_temperature += best_mutation;
                current_entropy = current_entropy + challenge.weights[gene_to_add] - challenge.weights[gene_to_remove];

                unsafe {
                    for flow_index in 0..vertex_count {
                        *mutation_rates.get_unchecked_mut(flow_index) += 
                            challenge.interaction_values.get_unchecked(flow_index).get_unchecked(gene_to_add) - 
                            challenge.interaction_values.get_unchecked(flow_index).get_unchecked(gene_to_remove);
                    }
                }

                cooling_schedule[gene_to_add] = 3;
                cooling_schedule[gene_to_remove] = 3;
            } else {
                break;
            }

            if current_temperature as u32 >= challenge.baseline_value {
                return Ok(Some(SubSolution { items: coloring }));
            } 

            for cooling_rate in cooling_schedule.iter_mut() {
                *cooling_rate = if *cooling_rate > 0 { *cooling_rate - 1 } else { 0 };
            }
        }
    
        if current_temperature as u32 >= challenge.baseline_value {
            Ok(Some(SubSolution { items: coloring }))
        } else {
            Ok(None)
        }
    }
}

pub fn help() {
    println!("No help information available.");
}
