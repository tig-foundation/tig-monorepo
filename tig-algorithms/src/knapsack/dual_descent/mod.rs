// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::Result;

use serde_json::{Map, Value};
use tig_challenges::knapsack::*;

const DEBUG: bool = false;



pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    // return Err(<msg>) if your algorithm encounters an error
    // return Ok(()) if your algorithm finds no solution or needs to exit early
    // return Ok(Solution { .. }) if your algorithm finds a solution
    let weights: &Vec<u32> = &challenge.weights;
    let values: &Vec<u32> = &challenge.values;
    let capacity: u32 = challenge.max_weight;
    let iterations: usize = 1000;
    let mut alpha: f64 = 1.0;
    let convergence_threshold: f64 = 1e-5;

    let n = values.len();
    let mut dual_vars = vec![0.0; n];
    let mut final_selected_items = vec![false; n];
    let mut best_total_value = 0;
    let mut best_total_weight = 0;

    for iter in 0..iterations {
        let mut current_selected_items = vec![false; n];
        let mut current_total_weight = 0;
        let mut current_total_value = 0;

        let mut dp = vec![vec![0; (capacity + 1) as usize]; n + 1];
        let mut keep = vec![vec![false; (capacity + 1) as usize]; n + 1];

        // Solve the subproblem for each item
        for j in 1..=n {
            for w in 0..=capacity {
                if weights[j - 1] <= w {
                    let not_taking_item = dp[j - 1][w as usize];
                    let taking_item = ((values[j - 1] as f64 - dual_vars[j - 1]) + dp[j - 1][(w - weights[j - 1]) as usize] as f64) as i32;

                    if taking_item > not_taking_item {
                        dp[j][w as usize] = taking_item;
                        keep[j][w as usize] = true;
                    } else {
                        dp[j][w as usize] = not_taking_item;
                    }
                } else {
                    dp[j][w as usize] = dp[j - 1][w as usize];
                }
            }
        }

        // Reconstruct the solution to find the selected items
        let mut w = capacity;
        for j in (1..=n).rev() {
            if keep[j][w as usize] && (current_total_weight + weights[j - 1]) <= capacity {
                current_selected_items[j - 1] = true;
                current_total_weight += weights[j - 1];
                current_total_value += values[j - 1];
                w -= weights[j - 1];
            }
        }

        // Calculate total resource consumed
        let total_resource_consumed = current_total_weight;

        // Update dual variables
        let mut any_update: bool = false;
        for i in 0..dual_vars.len() {
            let old_dual_var = dual_vars[i];
            if current_selected_items[i] {
                dual_vars[i] += alpha * (total_resource_consumed as f64 - capacity as f64) / capacity as f64;
            } else {
                dual_vars[i] -= alpha * (capacity as f64 - total_resource_consumed as f64) / capacity as f64;
            }
            if (dual_vars[i] - old_dual_var).abs() > convergence_threshold {
                any_update = true;
            }

            if DEBUG {
                //println!("Iteration {}: Dual var {} updated from {} to {}", iter, i, old_dual_var, dual_vars[i]);
            }
        }

        // Update final selected items based on current iteration
        if current_total_weight <= capacity && current_total_value > best_total_value {
            best_total_value = current_total_value;
            best_total_weight = current_total_weight;
            final_selected_items = current_selected_items.clone();
        }

        // Debugging information for each iteration
        if DEBUG {
            //println!("Iteration {}: Resource consumed: {}", iter, total_resource_consumed);
            println!("Iteration {}: Current total value: {}", iter, current_total_value);
            println!("Iteration {}: Best total value so far: {}", iter, best_total_value);
            //println!("Iteration {}: Current selected items: {:?}", iter, current_selected_items);
        }

        // Early stopping if no dual variable is updated
        if !any_update {
            if DEBUG {
                println!("Iteration {}: No updates to dual variables, stopping early.", iter);
            }
            break;
        }

        if iter > 0 && iter % 100 == 0 && current_total_value <= best_total_value{
            alpha *= 0.5;
            if DEBUG {
                println!("Iteration {}: Reducing alpha to {}", iter, alpha);
            }
        }
    }

    if best_total_weight <= capacity {
        let _ = save_solution(&Solution {
            items: final_selected_items.iter().enumerate().filter_map(|(i, &selected)| if selected { Some(i) } else { None }).collect(),
        });
        return Ok(());
    } else {
        Ok(())
    }
}

pub fn help() {
    println!("No help information available.");
}
