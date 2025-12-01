// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;
use std::cmp::Ordering;



pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let num_items = challenge.weights.len();
    let mut remaining_capacity = challenge.max_weight as i32; 
    let mut densities = vec![0.0; num_items];
    let mut selected_items = Vec::with_capacity(num_items/2);

    for i in 0..num_items {
        densities[i] = challenge.values[i] as f64 / challenge.weights[i] as f64;
    }

    let mut sorted_indices: Vec<usize> = (0..num_items).collect();
    sorted_indices.sort_unstable_by(|&i, &j| densities[i].partial_cmp(&densities[j]).unwrap_or(Ordering::Equal));

    while let Some(i) = sorted_indices.pop() {
        let weight = challenge.weights[i] as i32;
        if weight > remaining_capacity {
            continue;
        }

        selected_items.push(i);
        remaining_capacity -= weight;

        for &j in &sorted_indices {
            let joint_profit = challenge.interaction_values[i][j] as f64;
            densities[j] += joint_profit / challenge.weights[j] as f64;
        }

        sorted_indices.sort_unstable_by(|&i, &j| densities[i].partial_cmp(&densities[j]).unwrap_or(Ordering::Equal));
    }

    let _ = save_solution(&Solution { items: selected_items });
    return Ok(());
}




// Important! Do not include any tests in this file, it will result in your submission being rejected

pub fn help() {
    println!("No help information available.");
}
