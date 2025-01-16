/*!
Copyright 2024 syebastian

Licensed under the TIG Benchmarker Outbound Game License v1.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

use anyhow::Result;
use tig_challenges::knapsack::{Challenge, Solution};

pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
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

        if current_temperature as u32 >= challenge.min_value {
            return Ok(Some(Solution { items: coloring }));
        } 

        for cooling_rate in cooling_schedule.iter_mut() {
            *cooling_rate = if *cooling_rate > 0 { *cooling_rate - 1 } else { 0 };
        }
    }
    
    if current_temperature as u32 >= challenge.min_value {
        Ok(Some(Solution { items: coloring }))
    } else {
        Ok(None)
    }
}

#[cfg(feature = "cuda")]
mod gpu_optimisation {
    use super::*;
    use cudarc::driver::*;
    use std::{collections::HashMap, sync::Arc};
    use tig_challenges::CudaKernel;

    // set KERNEL to None if algorithm only has a CPU implementation
    pub const KERNEL: Option<CudaKernel> = None;

    // Important! your GPU and CPU version of the algorithm should return the same result
    pub fn cuda_solve_challenge(
        challenge: &Challenge,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> anyhow::Result<Option<Solution>> {
        solve_challenge(challenge)
    }
}
#[cfg(feature = "cuda")]
pub use gpu_optimisation::{cuda_solve_challenge, KERNEL};