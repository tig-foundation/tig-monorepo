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
    // TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
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
        let capacity = challenge.max_weight as usize;
        let passing_value = challenge.baseline_value as usize;
        let n = challenge.num_items;
        let weights: Vec<usize> = challenge.weights.iter().map(|&w| w as usize).collect();
        let values: Vec<usize> = challenge.values.iter().map(|&v| v as usize).collect();

        // sort
        let mut sorted_items: Vec<usize> = (0..n).collect();
        sorted_items.sort_by(|&a, &b| {
            let ratio_a = values[a] as f64 / weights[a] as f64;
            let ratio_b = values[b] as f64 / weights[b] as f64;
            ratio_b.partial_cmp(&ratio_a).unwrap()
        });

        // estimate optimal value
        let mut greedy_value: usize= 0;
        let mut limit1: f64 = 0.0;
        let mut limit1_flag: bool = true;
        let mut limit2: f64 = 0.0;
        let mut limit2_flag: u8 = 0;
        let mut remaining = capacity;
        for i in 0..n {
            let item = sorted_items[i];

            if weights[item] > remaining {
                // can't take it
                if limit1_flag && i > 0 {
                    let last = sorted_items[i - 1]; // optimism
                    limit1 = greedy_value as f64 + remaining as f64 * values[last] as f64 / weights[last] as f64;
                    limit1_flag = false;
                    continue;
                }

                limit2_flag += 1;
                if limit2_flag == 1 {
                    let last = sorted_items[i];
                    limit2 = greedy_value as f64 + remaining as f64 * values[last] as f64 / weights[last] as f64;
                }
                continue;
            }

            limit2_flag = 0;
            greedy_value += values[item];
            remaining -= weights[item];
        }

        let estimate: usize = (0.62 * limit1 + 0.38 * limit2).ceil() as usize;

        if estimate < passing_value { // no solution
            return Ok(None);
        }

        // dp
        let mut dp = vec![0; capacity + 1];
        let mut selected_items: Vec<Vec<usize>> = vec![Vec::new(); capacity + 1];

        for i in 0..n {
            let w = weights[i];
            let v = values[i];
            for c in (w..=capacity).rev() {
                let take = dp[c - w] + v;
                if dp[c] < take  {
                    dp[c] = take;
                    selected_items[c] = selected_items[c - w].clone();
                    selected_items[c].push(i);
                }
            }
        }

        if dp[capacity] >= passing_value{
            Ok(Some(SubSolution { items: selected_items[capacity].clone()}))
        } else {
            Ok(None)
        }

    }
}

pub fn help() {
    println!("No help information available.");
}
