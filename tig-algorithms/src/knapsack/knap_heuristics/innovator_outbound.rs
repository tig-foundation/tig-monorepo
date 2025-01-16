/*!
Copyright 2024 CodeAlchemist

Licensed under the TIG Innovator Outbound Game License v1.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

use tig_challenges::knapsack::{Challenge, Solution, calculate_total_value};
use rand::{rngs::SmallRng, Rng, SeedableRng};

fn calculate_value_change(
    current_solution: &[usize],
    item: usize,
    values: &[u32],
    interaction_values: &[Vec<i32>]
) -> i32 {
    let mut value_change = values[item] as i32;

    for &other_item in current_solution {
        value_change += interaction_values[item][other_item];
    }

    value_change
}

pub fn solve_challenge(challenge: &Challenge) ->  anyhow::Result<Option<Solution>> {
    {
        let coef = vec![7.685029664028169e-08, -0.00026933405380081394, 0.39570553839776795, 4.133197036353228, 8025.3579356210785];

        let mut lim = 0.0;
        let mut val = 1.0;
        for i in (0..coef.len()).rev(){
            lim += coef[i] * val;
            val *= challenge.difficulty.num_items as f32;
        }

        lim = lim * (1.0 as f32 + (challenge.difficulty.better_than_baseline as f32)/1000.0);

        if challenge.min_value as f32 > lim {
            return Ok(None);
        }
    }

    let mut current_solution = greedy_initial_solution(challenge);
    let mut current_value = calculate_total_value(&current_solution, &challenge.values, &challenge.interaction_values);
    let mut current_weight: u32 = current_solution.iter().map(|&i| challenge.weights[i]).sum();
    let mut best_value = current_value;
    let mut rng = SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64);

    let initial_temperature =  100.0;
    let cooling_rate = 0.925;
    let mut temperature = initial_temperature;

    while temperature > 1.0 {
        for i in 0..challenge.difficulty.num_items {
            let item_pos = current_solution.binary_search(&i);

            let (new_solution, new_weight, new_value) = match item_pos {
                Ok(pos) => {
                    // Remove item
                    let mut new_sol = current_solution.clone();
                    let new_weight = current_weight - challenge.weights[i];
                    new_sol.remove(pos);
                    let new_value = current_value as i32 - calculate_value_change(&current_solution, i, &challenge.values, &challenge.interaction_values);
                    (new_sol, new_weight, new_value)
                },
                Err(pos) => {
                    // Add item
                    let new_weight = current_weight + challenge.weights[i];
                    if new_weight <= challenge.max_weight {
                        let mut new_sol = current_solution.clone();
                        new_sol.insert(pos, i);

                        let new_value = current_value as i32 + calculate_value_change(&current_solution, i, &challenge.values, &challenge.interaction_values);
                        (new_sol, new_weight, new_value)
                    } else {
                        continue;
                    }
                }
            };

            let delta = new_value - current_value as i32;
            if delta > 0 || rng.gen::<f64>() < (delta as f64 / temperature).exp() {
                current_solution = new_solution;
                current_value = new_value as u32;
                current_weight = new_weight;

                if current_value > best_value && current_weight <= challenge.max_weight {
                    best_value = current_value;
                }
            }

            if current_value >= challenge.min_value && current_weight <= challenge.max_weight {
                return Ok(Some(Solution { items: current_solution }));
            }
        }

        temperature *= cooling_rate;
    }

    Ok(None)
}

fn greedy_initial_solution(challenge: &Challenge) -> Vec<usize> {
    let n = challenge.weights.len();
    let mut solution = Vec::new();
    let mut current_weight = 0;

    let mut items: Vec<(usize, f64)> = (0..n)
        .map(|i| {
            let total_value = challenge.values[i] as i32 + challenge.interaction_values[i].iter().sum::<i32>();
            let ratio = total_value as f64 / challenge.weights[i] as f64;
            (i, ratio)
        })
        .collect();

    items.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (i, _) in items {
        if current_weight + challenge.weights[i] <= challenge.max_weight {
            solution.push(i);
            current_weight += challenge.weights[i];
        }
    }

    solution.sort_unstable();
    solution
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

// Important! Do not include any tests in this file, it will result in your submission being rejected
