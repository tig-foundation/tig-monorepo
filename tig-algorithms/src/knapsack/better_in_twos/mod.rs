use std::collections::HashSet;
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;



pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let n = challenge.num_items;
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            pairs.push((i, j));
        }
    }
    let weights: Vec<u32> = pairs
        .iter()
        .map(|(i, j)| challenge.weights[*i] + challenge.weights[*j])
        .collect();
    let values: Vec<u32> = pairs
        .iter()
        .map(|(i, j)| challenge.values[*i] + challenge.values[*j])
        .collect();
    let ratios: Vec<f64> = weights
        .iter()
        .zip(values.iter())
        .map(|(w, v)| *v as f64 / *w as f64)
        .collect();
    let mut sorted_value_to_weight_ratio: Vec<usize> = (0..n).collect();
    sorted_value_to_weight_ratio.sort_by(|&a, &b| ratios[a].partial_cmp(&ratios[b]).unwrap());

    let items = HashSet::<usize>::new();
    let mut total_weight = 0;
    let max_weight = challenge.max_weight;
    for &idx in &sorted_value_to_weight_ratio {
        let mut additional_weight = 0;
        let p = pairs[idx];
        if !items.contains(&p.0) {
            additional_weight += challenge.weights[p.0];
        }
        if !items.contains(&p.1) {
            additional_weight += challenge.weights[p.1];
        }
        if total_weight + additional_weight > max_weight {
            continue;
        }
        total_weight += additional_weight;
    }
    let _ = save_solution(&Solution {
        items: items.into_iter().collect(),
    });
    return Ok(());
}

pub fn help() {
    println!("No help information available.");
}
