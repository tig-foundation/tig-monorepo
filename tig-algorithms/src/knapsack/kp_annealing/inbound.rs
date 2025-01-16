/*!
Copyright 2024 CodeAlchemist

Identity of Submitter CodeAlchemist

UAI null

Licensed under the TIG Inbound Game License v2.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::Result;
use rand::{SeedableRng, Rng, rngs::SmallRng};
use tig_challenges::knapsack::{Challenge, Solution, calculate_total_value};

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

pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
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
    let mut rng = SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into()?));

    let mut positions: Vec<i32> = vec![-1; challenge.difficulty.num_items];
    for (idx, &item) in current_solution.iter().enumerate() {
        positions[item] = idx as i32;
    }

    let mut temperature = 500.0;
    while temperature > 1.0 {
        for i in 0..challenge.difficulty.num_items {
            let pos = positions[i];
            if pos >= 0 {
                let new_weight = current_weight - challenge.weights[i];
                let new_value = current_value as i32 -  calculate_value_change(&current_solution, i, &challenge.values, &challenge.interaction_values);

                let delta = new_value - current_value as i32;
                if delta > 0 || rng.gen::<f64>() < (delta as f64 / temperature).exp() {
                    let last_idx = current_solution.len() - 1;
                    if pos as usize != last_idx {
                        current_solution.swap(pos as usize, last_idx);
                        positions[current_solution[pos as usize]] = pos;
                    }
                    current_solution.pop();
                    positions[i] = -1;
                    current_value = new_value as u32;
                    current_weight = new_weight;
                }
            } else if current_weight + challenge.weights[i] <= challenge.max_weight {
                let new_weight = current_weight + challenge.weights[i];
                let new_value = current_value as i32 + calculate_value_change(&current_solution, i, &challenge.values, &challenge.interaction_values);

                let delta = (new_value - current_value as i32);
                if delta > 0 || rng.gen::<f64>() < (delta as f64 / temperature).exp() {
                    positions[i] = current_solution.len() as i32;
                    current_solution.push(i);
                    current_value = new_value as u32;
                    current_weight = new_weight;

                }
            }

            if current_value as i32 >= challenge.min_value as i32 && current_weight <= challenge.max_weight {
                return Ok(Some(Solution { items: current_solution }));
            }
        }
        temperature *= 0.985;
    }
    Ok(None)
}

#[cfg(feature = "cuda")]
mod gpu_optimisation {
    use super::*;
    use cudarc::driver::*;
    use std::{collections::HashMap, sync::Arc};
    use tig_challenges::CudaKernel;

    pub const KERNEL: Option<CudaKernel> = None;

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