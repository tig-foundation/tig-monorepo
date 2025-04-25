/*!
Copyright 2024 Chad Blanchard

Licensed under the TIG Open Data License v1.0 or (at your option) any later version 
(the "License"); you may not use this file except in compliance with the License. 
You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

use tig_challenges::knapsack::*;
use std::cmp;
use std::collections::HashMap;

struct Item {
    index: usize,
    weight: usize,
    value: usize,
}


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

pub fn solve_sub_instance(challenge: &SubInstance) -> anyhow::Result<Option<SubSolution>> {
    let max_weight = challenge.max_weight as usize;
    let baseline_value = challenge.baseline_value as usize;
    let num_items = challenge.difficulty.num_items;

    let items: Vec<Item> = challenge.weights.iter().zip(challenge.values.iter()).enumerate()
        .map(|(i, (&w, &v))| Item {
            index: i,
            weight: w as usize,
            value: v as usize,
        })
        .collect();

    // Phase 1: Quantum-inspired superposition
    let superpositions = generate_superpositions(&items, max_weight);

    // Phase 2: Interference and amplification
    let amplified = amplify_solutions(&superpositions, baseline_value);

    // Phase 3: Measurement and solution reconstruction
    if let Some(solution) = measure_best_solution(&amplified, &items, max_weight, baseline_value) {
        Ok(Some(SubSolution { items: solution }))
    } else {
        Ok(None)
    }
}

fn generate_superpositions(items: &[Item], max_weight: usize) -> Vec<HashMap<usize, f64>> {
    let mut superpositions = vec![HashMap::new(); max_weight + 1];
    superpositions[0].insert(0, 1.0);

    for item in items {
        for w in (item.weight..=max_weight).rev() {
            let new_states: HashMap<usize, f64> = superpositions[w - item.weight].iter()
                .map(|(&v, &p)| (v + item.value, p * 0.5))
                .collect();

            if let Some(&max_new_value) = new_states.keys().max() {
                if max_new_value > *superpositions[w].keys().max().unwrap_or(&0) {
                    superpositions[w] = new_states;
                }
            }
        }
    }

    superpositions
}

fn amplify_solutions(superpositions: &[HashMap<usize, f64>], baseline_value: usize) -> HashMap<usize, f64> {
    let mut amplified = HashMap::new();
    
    for states in superpositions.iter() {
        for (&value, &probability) in states.iter() {
            if value >= baseline_value {
                *amplified.entry(value).or_insert(0.0) += probability * 1.5; // Amplify good solutions
            }
        }
    }

    amplified
}

fn measure_best_solution(amplified: &HashMap<usize, f64>, items: &[Item], max_weight: usize, baseline_value: usize) -> Option<Vec<usize>> {
    let best_value = *amplified.keys().max()?;
    if best_value < baseline_value {
        return None;
    }

    let mut solution = Vec::new();
    let mut remaining_value = best_value;
    let mut remaining_weight = max_weight;

    for item in items.iter().rev() {
        if remaining_weight >= item.weight && remaining_value >= item.value {
            let prob_with = amplified.get(&remaining_value).unwrap_or(&0.0);
            let prob_without = amplified.get(&(remaining_value - item.value)).unwrap_or(&0.0);
            
            if prob_with > prob_without {
                solution.push(item.index);
                remaining_weight -= item.weight;
                remaining_value -= item.value;
            }
        }
    }

    Some(solution)
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
        challenge: &SubInstance,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> anyhow::Result<Option<SubSolution>> {
        solve_challenge(challenge)
    }
}
#[cfg(feature = "cuda")]
pub use gpu_optimisation::{cuda_solve_challenge, KERNEL};
