// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge

use std::collections::VecDeque;
use std::cmp::Ordering;
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;

#[derive(Clone, Debug)]
struct Node {
    level: usize,
    profit: u32,
    weight: u32,
    bound: f64,
}

impl Node {
    fn bound(&self, num_items: usize, max_weight: u32, weights: &[u32], values: &[u32]) -> f64 {
        if self.weight >= max_weight {
            return 0.0;
        }
        let mut profit_bound = self.profit as f64;
        let mut j = self.level;
        let mut total_weight = self.weight;
        while j < num_items && total_weight + weights[j] <= max_weight {
            total_weight += weights[j];
            profit_bound += values[j] as f64;
            j += 1;
        }
        if j < num_items {
            profit_bound += (max_weight - total_weight) as f64 * values[j] as f64 / weights[j] as f64;
        }
        profit_bound
    }
}



pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let num_items = challenge.num_items as usize;
    let max_weight = challenge.max_weight;
    let values = &challenge.values;
    let weights = &challenge.weights;

    let mut queue: VecDeque<Node> = VecDeque::new();
    let mut u = Node { level: 0, profit: 0, weight: 0, bound: 0.0 };
    u.bound = u.bound(num_items, max_weight, weights, values);
    queue.push_back(u.clone());

    let mut max_profit = 0;
    let mut best_items = vec![];

    while let Some(node) = queue.pop_front() {
        if node.bound > max_profit as f64 {
            u.level = node.level + 1;
            u.weight = node.weight + weights[u.level];
            u.profit = node.profit + values[u.level];
            u.bound = u.bound(num_items, max_weight, weights, values);

            if u.weight <= max_weight && u.profit > max_profit {
                max_profit = u.profit;
                best_items.push(u.level);
            }
            if u.bound > max_profit as f64 {
                queue.push_back(u.clone());
            }

            u.weight = node.weight;
            u.profit = node.profit;
            u.bound = u.bound(num_items, max_weight, weights, values);
            if u.bound > max_profit as f64 {
                queue.push_back(u.clone());
            }
        }
    }

    let _ = save_solution(&Solution { items: best_items });
    return Ok(());
}

pub fn help() {
    println!("No help information available.");
}
