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

    pub fn solve_sub_instance(challenge: &SubInstance) -> anyhow::Result<Option<SubSolution>> {
        let max_weight = challenge.max_weight as usize;
        let baseline_value = challenge.baseline_value as usize;
        let num_items = challenge.num_items;

        let weights: Vec<usize> = challenge
            .weights
            .iter()
            .map(|&weight| weight as usize)
            .collect();
        let values: Vec<usize> = challenge
            .values
            .iter()
            .map(|&value| value as usize)
            .collect();

        let mut sorted_items: Vec<(usize, f64)> = (0..num_items)
            .map(|i| (i, values[i] as f64 / weights[i] as f64))
            .collect();
        sorted_items.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut dp = vec![0; max_weight + 1];

        for &(item_index, _) in &sorted_items {
            let weight = weights[item_index];
            let value = values[item_index];
            for w in (weight..=max_weight).rev() {
                dp[w] = dp[w].max(dp[w - weight] + value);
                if dp[w] >= baseline_value {
                    return Ok(Some(SubSolution {
                        items: vec![item_index],
                    }));
                }
            }
        }

        let mut items = Vec::new();
        let mut w = max_weight;
        while w > 0 {
            if let Some(&(item_index, _)) = sorted_items
                .iter()
                .find(|&&(i, _)| weights[i] <= w && dp[w] == dp[w - weights[i]] + values[i])
            {
                items.push(item_index);
                w -= weights[item_index];
            } else {
                break;
            }
        }

        if dp[max_weight] >= baseline_value {
            Ok(Some(SubSolution { items }))
        } else {
            Ok(None)
        }
    }
}

pub fn help() {
    println!("No help information available.");
}
